from os.path import join
import pyelsa as elsa
import aomip
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.image as mig
import utils
import matplotlib.pyplot as plt
import preprocessing

## Prepare data
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
    
""" Homework 1: preprocessing with slicing"""
""" Start """
slice_idx_50 = 50
slice_idx_100 = 150
slice_idx_150 = 150
slice_idx_200 = 200

sliced_sinogram_50 = np.empty((num_projections, new_projs_cols), dtype=np.float32)
sliced_sinogram_100 = np.empty((num_projections, new_projs_cols), dtype=np.float32)
sliced_sinogram_150 = np.empty((num_projections, new_projs_cols), dtype=np.float32)
sliced_sinogram_200 = np.empty((num_projections, new_projs_cols), dtype=np.float32)

for i in range(num_projections):
    proj=binned_projections[i]
    row_50 = proj[slice_idx_50, :]
    sliced_sinogram_50[i, :] = row_50
    row_100 = proj[slice_idx_100, :]
    sliced_sinogram_100[i, :] = row_100
    row_150 = proj[slice_idx_150, :]
    sliced_sinogram_150[i, :] = row_150
    row_200 = proj[slice_idx_200, :]
    sliced_sinogram_200[i, :] = row_200

utils.save_array_as_image(sliced_sinogram_50,'sliced_sinogram_row_50.png','Img')
utils.save_array_as_image(sliced_sinogram_100,'sliced_sinogram_row_100.png','Img')
utils.save_array_as_image(sliced_sinogram_150,'sliced_sinogram_row_150.png','Img')
utils.save_array_as_image(sliced_sinogram_200,'sliced_sinogram_row_200.png','Img')

""" End """

""" Homework 3: Solving CT Problems """
""" Start """

# (i) Gradient Descent

def f(A,b,x):
    return 0.5* np.linalg.norm(A.dot(x) - b)**2

def df(A,b,x):
    return (A.T).dot(A.dot(x))- (A.T).dot(b)
    #return A.T.dot(A.dot(x)-b)
    
def gradientDescent(function,A,b, x0, iterations, alpha=1e-3):
    x = x0
    eps=1e-6
    stopIdx=0
    history = np.zeros((x0.size, iterations+1))
    history[:, 0] = x0
    
    value0 = function(A,b,x0)
    values = np.zeros(iterations+1)
    values[0] = value0
    
    for i in range(iterations):
        d = df(A,b,x) 
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

H = np.linspace(-1, 1, sliced_sinogram_150.shape[0])

#filtering sinogram data
ram_lak = np.abs(H)
shepp_logan = np.abs(H) * np.sinc(H / 2)
cosine = np.abs(H) * np.cos(H * np.pi / 2)

h = np.tile(ram_lak, (280, 1)).T
fftsino = np.fft.fft(sliced_sinogram_150, axis=0)
projection = np.fft.fftshift(fftsino, axes=1) * np.fft.fftshift(h, axes=0)
fsino = np.real(np.fft.ifft(np.fft.ifftshift(projection, axes=1), axis=0))

# Start Gd algorithm
iteration = 10
alpha=1e-3
hist, minv, stopIdx = gradientDescent(f,A,sliced_sinogram_150.flatten(),initial_x0,iteration,alpha)
xreconstructednofilter = (hist[:,stopIdx]).reshape(size)
utils.save_array_as_image(xreconstructednofilter,'reconstructed_gd_sino150_nofilter.png','Img')

hist, minv, stopIdx = gradientDescent(f,A,fsino.flatten(),initial_x0,iteration,alpha)
xreconstructedfilter = (hist[:,stopIdx]).reshape(size)
utils.save_array_as_image(xreconstructedfilter,'reconstructed_gd_sino150.png','Img')

# (ii) L2-Norm squared

def f_tikohonov(A,x,b,beta):
    return 0.5* np.linalg.norm(A.dot(x) - b)**2 + 0.5*beta*x.dot(x)

def dff_tikohonov(A,x,b,beta):
    return A.T.dot(A.dot(x)-b) + beta*x

def gradientDescentTikhonov(function,A,b,beta,x0, iterations, alpha=1e-3):
    x = x0
    eps=1e-6
    stopIdx=0
    history = np.zeros((x0.size, iterations+1))
    history[:, 0] = x0
    
    value0 = function(A,x0,b,beta)
    values = np.zeros(iterations+1)
    values[0] = value0
    
    for i in range(iterations):
        d = dff_tikohonov(A,x,b,beta) 
        x = x - alpha*d
        history[:,i+1] = x
        values[i] = function(A,x,b,beta)
        stopIdx=i
        if np.linalg.norm(d) < eps:
            stopIdx=i
            break
            
    return history, values, stopIdx


beta=1
hist, minv, stopIdx = gradientDescentTikhonov(f_tikohonov,A,fsino.flatten(),beta,initial_x0,iteration, alpha)
xreconstructedTikonov = (hist[:,stopIdx]).reshape(size)
utils.save_array_as_image(xreconstructedTikonov,'reconstructed_tiknonov_sino150.png','Img')

# (iii) Huber Functional

def huber_loss(A,b,x, delta):
    abs_x = np.abs(x)
    loss = np.where(abs_x <= delta, 0.5 * x ** 2, delta * (abs_x - 0.5 * delta))
    return 0.5* np.linalg.norm(A.dot(x) - b)**2 + np.sum(loss)

def huber_loss_gradient(x, delta):
    """
    Computes the gradient of the Huber loss of x with parameter delta.
    """
    abs_x = np.abs(x)
    gradient = np.where(abs_x <= delta, x, delta * np.sign(x))
    return gradient

def gradientDescentHuber(function,A,b,delta,x0, iterations, alpha=1e-3):
    x = x0
    eps=1e-6
    stopIdx=0
    history = np.zeros((x0.size, iterations+1))
    history[:, 0] = x0
    
    value0 = function(A,b,x0,delta)
    values = np.zeros(iterations+1)
    values[0] = value0
    
    for i in range(iterations):
        d = df(A,b,x) + huber_loss_gradient(x, delta)
        x = x - alpha*d
        history[:,i+1] = x
        values[i] = function(A,b,x0,delta)
        stopIdx=i
        if np.linalg.norm(d) < eps:
            stopIdx=i
            break
        
    return history, values, stopIdx


delta=1
hist, minv, stopIdx = gradientDescentHuber(huber_loss,A,fsino.flatten(),delta,initial_x0,iteration,alpha)
xreconstructedHuber = (hist[:,stopIdx]).reshape(size)
utils.save_array_as_image(xreconstructedHuber,'reconstructed_Huber_sino150.png','Img')

# iv) Fair potential

def fair(A,b,x,lambd):
    lrTerm = 0.5* np.linalg.norm(A.dot(x) - b)**2
    absTerm = np.abs(x/lambd)
    reg = (lambd**2) * np.sum(absTerm - np.log(1 + absTerm))
    return lrTerm + reg

def fair_grad(x, lambd):
    xTerm = x/lambd
    return x / (1 + xTerm)

def gradientDescentFair(function,A,b,lambd,x0, iterations, alpha=1e-3):
    x = x0
    eps=1e-6
    stopIdx=0
    history = np.zeros((x0.size, iterations+1))
    history[:, 0] = x0
    
    value0 = function(A,b,x0,delta)
    values = np.zeros(iterations+1)
    values[0] = value0
    
    for i in range(iterations):
        d = df(A,b,x) + fair_grad(x, lambd)
        x = x - alpha*d
        history[:,i+1] = x
        values[i] = function(A,b,x0,delta)
        stopIdx=i
        if np.linalg.norm(d) < eps:
            stopIdx=i
            break
        
    return history, values, stopIdx


delta=1
hist, minv, stopIdx = gradientDescentFair(fair,A,fsino.flatten(),delta,initial_x0,iteration,alpha)
xreconstructedFair = (hist[:,stopIdx]).reshape(size)
utils.save_array_as_image(xreconstructedTikonov,'reconstructed_Fair_sino150.png','Img')

""" End """

""" Homework 4: Finite Differences """
""" Start """

# def finite_diff_op_1d(n):
#     """
#     Returns the finite difference operator for a 1D signal with n samples.
#     """
#     D = np.zeros((n-1,n))
#     for i in range(n-1):
#         D[i,i] = -1
#         D[i,i+1] = 1
#     return D

# def f_finDiff(A,x,b,L,beta):
#     return 0.5* np.linalg.norm(A.dot(x) - b)**2 + 0.5*beta*np.linalg.norm(L.dot(x))**2

# def dff_finDiff(A,x,b,L,beta):
#     return A.T.dot(A.dot(x)-b) + beta*(((L.T).dot(L)).dot(x))

# def gradientDescentFinDiff(function,A,b,beta,L,x0, iterations, alpha=1e-3):
#     x = x0
#     eps=1e-6
#     stopIdx=0
#     history = np.zeros((x0.size, iterations+1))
#     history[:, 0] = x0
    
#     value0 = function(A,x0,b,L,beta)
#     values = np.zeros(iterations+1)
#     values[0] = value0
    
#     for i in range(iterations):
#         d = dff_finDiff(A,x,b,L,beta) #A.T.dot(A.dot(x)-b)
#         x = x - alpha*d
#         history[:,i+1] = x
#         values[i] = function(A,x,b,L,beta)
#         stopIdx=i
#         if np.linalg.norm(d) < eps:
#             stopIdx=i
#             break
            
#     return history, values, stopIdx


# L=finite_diff_op_1d(initial_x0.size)
# iteration = 10
# beta=1
# hist, minv, stopIdx = gradientDescentFinDiff(f_finDiff,A,fsino.flatten(),beta,L,initial_x0,iteration,1e-3)
# xreconstructedFair = (hist[:,stopIdx]).reshape(size)
# plt.imshow(xreconstructedFair, cmap='gray')
# utils.save_array_as_image(xreconstructedTikonov,'reconstructed_FiniteDifference_sino150.png','Img')

""" End """