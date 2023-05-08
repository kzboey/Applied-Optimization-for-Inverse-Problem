from os.path import join
import pyelsa as elsa
import aomip
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.image as mig
import utils
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from scipy import signal
from skimage.transform import radon, rescale
import preprocessing

data_path = '/srv/ceph/share-all/aomip/6983008_seashell'
projs_name = '20211124_seashell_{:04}.tif'
projs_rows = 2368
projs_cols = 2240
num_projections = 721
# create the numpy array which will receive projection data from tiff files
projs = np.zeros((num_projections, projs_rows, projs_cols), dtype=np.float32)
# load projection data
for i in range(1,num_projections):
    projs[i] = plt.imread(join(data_path, projs_name.format(i)))
    
new_projs_rows = (projs[0].shape[0] // 8)
new_projs_cols = (projs[0].shape[1] // 8)
bin_factor = 8
binned_projections = preprocessing.containerised_projections(num_projections,new_projs_rows,new_projs_cols,projs,4,bin_factor)


slice_idx_50 = 50
slice_idx_100 = 150
slice_idx_200 = 200
# slice_idx_400 = 400
# slice_idx_600 = 600

sliced_sinogram_50 = np.empty((num_projections, new_projs_cols), dtype=np.float32)
sliced_sinogram_100 = np.empty((num_projections, new_projs_cols), dtype=np.float32)
sliced_sinogram_200 = np.empty((num_projections, new_projs_cols), dtype=np.float32)
# sliced_sinogram_400 = np.empty((num_projections, new_projs_cols), dtype=np.float32)
# sliced_sinogram_600 = np.empty((num_projections, new_projs_cols), dtype=np.float32)

for i in range(num_projections):
    proj=binned_projections[i]
    row_50 = proj[slice_idx_50, :]
    sliced_sinogram_50[i, :] = row_50
    row_100 = proj[slice_idx_100, :]
    sliced_sinogram_100[i, :] = row_100
    row_200 = proj[slice_idx_200, :]
    sliced_sinogram_200[i, :] = row_200
#     row_400 = proj[slice_idx_400, :]
#     sliced_sinogram_400[i, :] = row_400
#     row_600 = proj[slice_idx_600, :]
#     sliced_sinogram_600[i, :] = row_600

utils.save_array_as_image(sliced_sinogram_50,'sliced_sinogram_50.png','Img')
utils.save_array_as_image(sliced_sinogram_100,'sliced_sinogram_100.png','Img')
utils.save_array_as_image(sliced_sinogram_200,'sliced_sinogram_200.png','Img')
# utils.save_array_as_image(sliced_sinogram_400,'sliced_sinogram_400.png','Img')
# utils.save_array_as_image(sliced_sinogram_600,'sliced_sinogram_600.png','Img')

def f(A,b,x):
    return 0.5* np.linalg.norm(A.dot(x) - b)**2

def df(A,b,x):
    return (A.T).dot(A.dot(x))- (A.T).dot(b)
    #return A.T.dot(A.dot(x)-b)
    
def gradientDescent(function,A,b, x0, iterations, alpha=1e-5):
    x = x0
    eps=1e-6
    stopIdx=0
    history = np.zeros((x0.size, iterations+1))
    history[:, 0] = x0
    
    value0 = function(A,b,x0)
    values = np.zeros(iterations+1)
    values[0] = value0
    
    for i in range(iterations):
        d = df(A,b,x) #A.T.dot(A.dot(x)-b)
        #alpha = 1e-7 #(d.T).dot(d) / (d.T).dot(A).dot(d)
        x = x - alpha*d
        history[:,i+1] = x
        values[i] = function(A,b,x)
        stopIdx=i
        if np.linalg.norm(d) < eps:
            stopIdx=i
            break
            
    return history, values, stopIdx

size = (new_projs_rows,new_projs_cols)
A = aomip.XrayOperator(size, [721], np.linspace(0, 360, new_projs_cols), size[0]*1, size[0]*0.5)
b = sliced_sinogram_100.flatten()
print("A has size = ", A.shape, ", and b has size = ", b.size)
initial_x0 = np.array(np.zeros(new_projs_rows*new_projs_cols))-0.1

H = np.linspace(-1, 1, sliced_sinogram_100.shape[0])

ram_lak = np.abs(H)
shepp_logan = np.abs(H) * np.sinc(H / 2)
cosine = np.abs(H) * np.cos(H * np.pi / 2)

h = np.tile(ram_lak, (280, 1)).T
fftsino = np.fft.fft(sliced_sinogram_100, axis=0)
projection = np.fft.fftshift(fftsino, axes=1) * np.fft.fftshift(h, axes=0)
fsino = np.real(np.fft.ifft(np.fft.ifftshift(projection, axes=1), axis=0))

iteration = 10
alpha=1e-3
hist, minv, stopIdx = gradientDescent(f,A,fsino.flatten(),initial_x0,iteration,alpha)
xreconstructed = (hist[:,stopIdx]).reshape(size)
plt.imshow(xreconstructed, cmap='gray')
utils.save_array_as_image(xreconstructed,'reconstructed_gd_sino150.png','Img')

## Part 4 : FInite difference 

def fair_grad(x, lambd):
    xTerm = x/lambd
    return x / (1 + xTerm)

def huber_loss_gradient(x, delta):
    """
    Computes the gradient of the Huber loss of x with parameter delta.
    """
    abs_x = np.abs(x)
    gradient = np.where(abs_x <= delta, x, delta * np.sign(x))
    return gradient

def finite_diff_op_1d(n):
    """
    Returns the finite difference operator for a 1D signal with n samples.
    """
    D = np.zeros((n-1,n))
    for i in range(n-1):
        D[i,i] = -1
        D[i,i+1] = 1
    return D

def generate_pattern_matrix(n):
    """
        Generate L transpose multiply L
    """
    diagonal = np.ones(n) * 2
    sub_diagonal = np.ones(n - 1) * -1
    A = np.diag(diagonal) + np.diag(sub_diagonal, k=-1) + np.diag(sub_diagonal, k=1)
    A[0, 1] = -1
    A[-1, -2] = -1
    A[0, 0] = 1
    A[-1, -1] = 1
    return A

def f_finDiff(A,x,b,L,beta=1):
    return 0.5* np.linalg.norm(A.dot(x) - b)**2 + 0.5*beta*np.linalg.norm(L.dot(x))**2

def dff_finDiff(A,x,b,L,beta):
    return A.T.dot(A.dot(x)-b) + beta*(generate_pattern_matrix(x.size)).dot(x)

def gradientDescentFinDiff(A,b,beta,L,x0, iterations, alpha=1e-5):
    x = x0
    eps=1e-6
    stopIdx=0
    history = np.zeros((x0.size, iterations+1))
    history[:, 0] = x0
    
    for i in range(iterations):
        d = dff_finDiff(A,x,b,L,beta) #A.T.dot(A.dot(x)-b)
        x = x - alpha*d
        history[:,i+1] = x
        stopIdx=i
        if np.linalg.norm(d) < eps:
            stopIdx=i
            break
            
    return history, stopIdx

def huber_loss_fd(A,b,x,L, delta,beta=1):
    abs_x = np.abs(x)
    loss = np.where(abs_x <= delta, 0.5 * x ** 2, delta * (abs_x - 0.5 * delta))
    return f_finDiff(A,x,b,L,beta) + np.sum(loss)

def gradientDescentHuberFd(A,b,delta,L,x0, iterations, alpha=1e-3):
    x = x0
    eps=1e-6
    stopIdx=0
    history = np.zeros((x0.size, iterations+1))
    history[:, 0] = x0
    
    for i in range(iterations):
        d = dff_finDiff(A,x,b,L,1)  + huber_loss_gradient(x, delta)
        x = x - alpha*d
        history[:,i+1] = x
        stopIdx=i
        if np.linalg.norm(d) < eps:
            stopIdx=i
            break
        
    return history, stopIdx

def gradientDescentFairFd(A,b,lambd,L,x0,iterations, alpha=1e-3):
    x = x0
    eps=1e-6
    stopIdx=0
    history = np.zeros((x0.size, iterations+1))
    history[:, 0] = x0

    for i in range(iterations):
        d = dff_finDiff(A,x,b,L,1) + fair_grad(x, lambd)
        x = x - alpha*d
        history[:,i+1] = x
        stopIdx=i
        if np.linalg.norm(d) < eps:
            stopIdx=i
            break
        
    return history, stopIdx


L=finite_diff_op_1d(initial_x0.size)
iteration = 10
beta=5
initial_x0 = np.array(np.random.random(new_projs_rows*new_projs_cols))
hist, stopIdx = gradientDescentFinDiff(A,fsino.flatten(),beta,L,initial_x0,iteration,1e-5)
xreconstructedFd = (hist[:,stopIdx]).reshape(size)
utils.save_array_as_image(xreconstructedFd,'reconstructed_FiniteDifference_sino150.png','Img')

hist, stopIdx = gradientDescentHuberFd(A,fsino.flatten(),beta,L,initial_x0,iteration,1e-5)
xreconstructedHuberFd = (hist[:,stopIdx]).reshape(size)
utils.save_array_as_image(xreconstructedHuberFd,'reconstructed_Huber_FiniteDifference_sino150.png','Img')

hist, stopIdx = gradientDescentFairFd(A,fsino.flatten(),beta,L,initial_x0,iteration,1e-5)
xreconstructedFairFd = (hist[:,stopIdx]).reshape(size)
utils.save_array_as_image(xreconstructedFairFd,'reconstructed_Fair_FiniteDifference_sino150.png','Img')