import aomip
import numpy as np
import json
from os.path import join
from tifffile import imsave, imread
import matplotlib.pyplot as plt
import GradientDescent
from challenge import utils
from utils import save_array_as_image, plot_convergence
from scipy.optimize import line_search

size = np.array([512, 512])
phantom = aomip.shepp_logan(size)
num_angles = 512
arc = 360

A = aomip.XrayOperator(size, [750], np.linspace(0, arc, num_angles), size[0]*100, size[0]*2)
sinogram = A.apply(phantom)

###################### Filter ##############################################
H = np.linspace(-1, 1, sinogram.shape[0])

ram_lak = np.abs(H)
shepp_logan = np.abs(H) * np.sinc(H / 2)
cosine = np.abs(H) * np.cos(H * np.pi / 2)

h = np.tile(shepp_logan, (512, 1)).T
fftsino = np.fft.fft(sinogram, axis=0)
projection = np.fft.fftshift(fftsino, axes=1) * np.fft.fftshift(h, axes=0)
fsino = np.real(np.fft.ifft(np.fft.ifftshift(projection, axes=1), axis=0))
###################### Filter ##############################################

img_vector = 512*512
x0 = np.random.rand(img_vector)
b = fsino.flatten()
groundtruth = phantom.flatten()

# exp1
xBacktrack, itr, history = GradientDescent.backtracking_linesearch(A,b,x0,1e-4,0.5,0.1,eps=1e-6,iteration=8000)
xBacktrack = xBacktrack.reshape((512,512))
print('iteration backtrack:',itr)
save_array_as_image(xBacktrack,'backtrack.png','img')
imsave(join('img','backtrack.tif'), xBacktrack)
plot_convergence(groundtruth,history,itr,'backtrack','img')

# exp2
xbb1, itr, history = GradientDescent.bb_linesearch(A,b,x0,1e-5,bb=1,eps=1e-6,iteration=8000)
xbb1 = xbb1.reshape((512,512))
print('iteration BB1:',itr)
save_array_as_image(xbb1,'bb1.png','img')
imsave(join('img','bb1.tif'), xbb1)
plot_convergence(groundtruth,history,itr,'bb1','img')

# exp3
xbb2, itr, history = GradientDescent.bb_linesearch(A,b,x0,1e-5,bb=2,eps=1e-6,iteration=8000)
xbb2 = xbb2.reshape((512,512))
print('iteration BB2:',itr)
save_array_as_image(xbb2,'bb2.png','img')
imsave(join('img','bb2.tif'), xbb2)
plot_convergence(groundtruth,history,itr,'bb2','img')

# exp4
xIsta, itr, history = GradientDescent.ista(A,b,x0,beta=0.1,eps=1e-6,iteration=8000)
xIsta = xIsta.reshape((512,512))
print('iteration ISTA:',itr)
save_array_as_image(xIsta,'ista.png','img')
imsave(join('img','istsa.tif'), xIsta)
plot_convergence(groundtruth,history,itr,'ista','img')

# exp4b
exp4b, itr, history = GradientDescent.ista(A,b,x0,beta=0.1,eps=1e-6,iteration=8000,backtrack=True)
exp4b = exp4b.reshape((512,512))
print('iteration ISTA:',itr)
save_array_as_image(exp4b,'exp4b.png','img')
imsave(join('img','exp4b.tif'), exp4b)
plot_convergence(groundtruth,history,itr,'ista with backtracking','img')

# exp5
lower_bound = np.zeros(x0.size) - 0.01 #-0.001
upper_bound = np.ones(x0.size) + 0.01 #+0.001
xpgd, itr, history = GradientDescent.projected_gradient_descent(A,b,x0,0.001,lower_bound,upper_bound,eps=1e-6,iteration=8000)
xpgd = xpgd.reshape((512,512))
print('iteration pgd:',itr)
save_array_as_image(xpgd,'pgd.png','img')
imsave(join('img','pgd.tif'), xpgd)
plot_convergence(groundtruth,history,itr,'pgd','img')

# exp6
exp6, itr, history = GradientDescent.compute_GD(A,b,0.001,x0,beta=5,regularizer='tikhonov',iteration=8000) 
exp6= exp6.reshape((512,512))
print('iteration:',itr)
save_array_as_image(exp6,'exp6.png','img')
imsave(join('img','exp6.tif'), exp6)
plot_convergence(groundtruth,history,itr,'GD_(tikhonov)','img')

# exp7
exp7, itr, history = GradientDescent.compute_GD(A,b,0.001,x0,beta=5,delta=1,regularizer='huber',iteration=8000) 
exp7= exp7.reshape((512,512))
print('iteration:',itr)
save_array_as_image(exp7,'exp7.png','img')
imsave(join('img','exp7.tif'), exp7)
plot_convergence(groundtruth,history,itr,'GD_(huber)','img')

# exp8
exp8, itr, history = GradientDescent.compute_GD(A,b,0.001,x0,beta=5,delta=1,regularizer='fair',iteration=8000) 
exp8= exp8.reshape((512,512))
print('iteration:',itr)
save_array_as_image(exp8,'exp8.png','img')
imsave(join('img','exp8.tif'), exp8)
plot_convergence(groundtruth,history,itr,'GD_(fair potential)','img')


