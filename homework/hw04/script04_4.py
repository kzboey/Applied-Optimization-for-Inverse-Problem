import aomip
import numpy as np
from tifffile import imsave, imread
from os.path import join
import GradientDescent
from challenge import utils
from utils import save_array_as_image, plot_convergence, filter_sinogram
from Operator import forward_diff, backward_diff, central_diff

size = np.array([512, 512])
phantom = aomip.shepp_logan(size)
num_angles = 512
arc = 360

A = aomip.XrayOperator(size, [750], np.linspace(0, arc, num_angles), size[0]*100, size[0]*2)
sinogram = A.apply(phantom)
fsino = filter_sinogram(sinogram,'shepp-logan')
b = fsino.flatten()

img_vector = 512*512
x0 = np.random.rand(img_vector)
groundtruth = phantom.flatten()

fd = forward_diff(img_vector) # forward difference operator
bd = backward_diff(img_vector) # backward difference operator
cd = central_diff(img_vector) # central difference operator

# without finite difference
exp1, itr, history = GradientDescent.compute_GD(A,b,0.001,x0,beta=5,regularizer='tikhonov',iteration=1000) 
exp1= exp1.reshape((512,512))
print('iteration:',itr)
save_array_as_image(exp1,'exp1.png','img4')
imsave(join('img4','exp1.tif'), exp1)
plot_convergence(groundtruth,history,itr,'GD_(tikhonov)','img4')

# with forward difference
exp2, itr, history = GradientDescent.compute_GD(A,b,0.001,x0,fd,beta=5,regularizer='tikhonov',iteration=1000) 
exp2= exp2.reshape((512,512))
print('iteration:',itr)
save_array_as_image(exp2,'exp2.png','img4')
imsave(join('img4','exp2.tif'), exp2)
plot_convergence(groundtruth,history,itr,'GD_(tikhonov)_forward_difference','img4')

# with backward difference
exp3, itr, history = GradientDescent.compute_GD(A,b,0.001,x0,bd,beta=5,regularizer='tikhonov',iteration=1000) 
exp3= exp3.reshape((512,512))
print('iteration:',itr)
save_array_as_image(exp3,'exp3.png','img4')
imsave(join('img4','exp3.tif'), exp2)
plot_convergence(groundtruth,history,itr,'GD_(tikhonov)_backward_difference','img4')

# with central difference
exp4, itr, history = GradientDescent.compute_GD(A,b,0.001,x0,cd,beta=5,regularizer='tikhonov',iteration=1000) 
exp4= exp4.reshape((512,512))
print('iteration:',itr)
save_array_as_image(exp4,'exp4.png','img4')
imsave(join('img4','exp4.tif'), exp4)
plot_convergence(groundtruth,history,itr,'GD_(tikhonov)_central_difference','img4')

# huber
exp5, itr, history = GradientDescent.compute_GD(A,b,0.001,x0,fd,beta=5,delta=1,regularizer='huber',iteration=1000) 
exp5= exp5.reshape((512,512))
print('iteration:',itr)
save_array_as_image(exp5,'exp5.png','img4')
imsave(join('img4','exp5.tif'), exp5)
plot_convergence(groundtruth,history,itr,'GD_(huber)_forward_difference_delta=1','img4')

exp6, itr, history = GradientDescent.compute_GD(A,b,0.001,x0,fd,beta=5,delta=5,regularizer='huber',iteration=1000) 
exp6= exp6.reshape((512,512))
print('iteration:',itr)
save_array_as_image(exp6,'exp6.png','img4')
imsave(join('img4','exp6.tif'), exp6)
plot_convergence(groundtruth,history,itr,'GD_(huber)_forward_difference_delta=5','img4')

exp7, itr, history = GradientDescent.compute_GD(A,b,0.001,x0,bd,beta=5,delta=1,regularizer='fair',iteration=1000) 
exp7= exp7.reshape((512,512))
print('iteration:',itr)
save_array_as_image(exp7,'exp7.png','img4')
imsave(join('img4','exp7.tif'), exp7)
plot_convergence(groundtruth,history,itr,'GD_(fair Potential)_backward_difference_delta=1','img4')

exp8, itr, history = GradientDescent.compute_GD(A,b,0.001,x0,bd,beta=5,delta=5,regularizer='fair',iteration=1000) 
exp8= exp8.reshape((512,512))
print('iteration:',itr)
save_array_as_image(exp8,'exp8.png','img4')
imsave(join('img4','exp8.tif'), exp8)
plot_convergence(groundtruth,history,itr,'GD_(fair Potential)_backward_difference_delta=5','img4')

