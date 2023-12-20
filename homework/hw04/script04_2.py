import aomip
import numpy as np
import json
from os.path import join
from tifffile import imsave, imread
import matplotlib.pyplot as plt
import GradientDescent
from challenge import utils
from utils import save_array_as_image, plot_convergence

size = np.array([512, 512])
phantom = aomip.shepp_logan(size)
num_angles = 512
arc = 360

A = aomip.XrayOperator(size, [750], np.linspace(0, arc, num_angles), size[0]*100, size[0]*2)
sinogram = A.apply(phantom)
gaussian_noise10 = np.random.normal(0, 5, sinogram.shape)     #add gaussian noise
noisy_sinogram = sinogram + gaussian_noise10

###################### Filter ##############################################
H = np.linspace(-1, 1, noisy_sinogram.shape[0])

ram_lak = np.abs(H)
shepp_logan = np.abs(H) * np.sinc(H / 2)
cosine = np.abs(H) * np.cos(H * np.pi / 2)

h = np.tile(shepp_logan, (512, 1)).T
fftsino = np.fft.fft(noisy_sinogram, axis=0)
projection = np.fft.fftshift(fftsino, axes=1) * np.fft.fftshift(h, axes=0)
fsino = np.real(np.fft.ifft(np.fft.ifftshift(projection, axes=1), axis=0))
###################### Filter ##############################################

img_vector = 512*512
x0 = np.random.rand(img_vector)
b = fsino.flatten()
groundtruth = phantom.flatten()

# # exp1
# gd, itr, history = GradientDescent.compute_GD(A,b,0.001,x0,beta=5,regularizer='tikhonov',iteration=500) 
# gd= gd.reshape((512,512))
# print('iteration tikhonov:',itr)
# save_array_as_image(gd,'exp1.png','img2')
# imsave(join('img2','exp1.tif'), gd)
# plot_convergence(groundtruth,history,itr,'Vanilla_GD_(tikhonov)_lambd=10','img2')

# # exp2
# ogm_fair, itr, history = GradientDescent.compute_OGM1(A,b,x0,beta=1,delta=1,regularizer='tikhonov',iteration=500)
# ogm_fair= ogm_fair.reshape((512,512))
# print('iteration huber:',itr)
# save_array_as_image(ogm_fair,'exp2.png','img2')
# imsave(join('img2','exp2.tif'), ogm_fair)
# plot_convergence(groundtruth,history,itr,'OGM_(tikhonov)_lambd=10','img2')

# # exp3
# lwb, itr, history = GradientDescent.compute_Landweber(A,b,0.001,x0,iteration=500)
# lwb= lwb.reshape((512,512))
# print('iteration lwb:',itr)
# save_array_as_image(lwb,'exp3.png','img2')
# imsave(join('img2','exp3.tif'), lwb)
# plot_convergence(groundtruth,history,itr,'Landweber_lambd=10','img2')

# # exp4
# cg, itr, history = GradientDescent.conjugate_gradient_normal(A,b,x0,iteration=200)
# cg= cg.reshape((512,512))
# print('iteration cg:',itr)
# save_array_as_image(cg,'exp4.png','img2')
# imsave(join('img2','exp4.tif'), cg)
# plot_convergence(groundtruth,history,itr,'Conjugate_gradient_lambd=10','img2')


# exp5
gd, itr, history = GradientDescent.compute_GD(A,b,0.001,x0,beta=5,regularizer='tikhonov',iteration=1000) 
gd= gd.reshape((512,512))
print('iteration tikhonov:',itr)
save_array_as_image(gd,'exp5.png','img2')
imsave(join('img2','exp5.tif'), gd)
plot_convergence(groundtruth,history,itr,'Vanilla_GD_(tikhonov)_lambd=5','img2')

# exp6
ogm_fair, itr, history = GradientDescent.compute_OGM1(A,b,x0,beta=1,delta=1,regularizer='tikhonov',iteration=750)
ogm_fair= ogm_fair.reshape((512,512))
print('iteration huber:',itr)
save_array_as_image(ogm_fair,'exp6.png','img2')
imsave(join('img2','exp6.tif'), ogm_fair)
plot_convergence(groundtruth,history,itr,'OGM_(tikhonov)_lambd=5','img2')

# exp7
lwb, itr, history = GradientDescent.compute_Landweber(A,b,0.001,x0,iteration=500)
lwb= lwb.reshape((512,512))
print('iteration lwb:',itr)
save_array_as_image(lwb,'exp7.png','img2')
imsave(join('img2','exp7.tif'), lwb)
plot_convergence(groundtruth,history,itr,'Landweber_lambd=5','img2')

# exp8
cg, itr, history = GradientDescent.conjugate_gradient_normal(A,b,x0,iteration=200)
cg= cg.reshape((512,512))
print('iteration cg:',itr)
save_array_as_image(cg,'exp8.png','img2')
imsave(join('img2','exp8.tif'), cg)
plot_convergence(groundtruth,history,itr,'Conjugate_gradient_lambd=5','img2')