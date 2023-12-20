import aomip
import numpy as np
from os.path import join
import matplotlib.pyplot as plt

step = 1
def callback1(x,y,i):
    if i % step == 0:
        fs.append(x)
        fy.append(y)
        
size = np.array([512, 512])
phantom = aomip.shepp_logan(size)
num_angles = 512
arc = 360
        
A = aomip.XrayOperator(size, [750], np.linspace(0, arc, num_angles), size[0]*100, size[0]*2)
sinogram = A.apply(phantom)
gaussian_noise10 = np.random.normal(0, 5, sinogram.shape)     #add gaussian noise
noisy_sinogram = sinogram + gaussian_noise10
fsino = aomip.filter_sinogram(noisy_sinogram,'shepp-logan')
b = fsino.flatten()

img_vector = 512*512
x0 = np.random.rand(img_vector)
groundtruth = phantom.flatten()

fs = []
fy = []
proxParams = {'v':x0, 'x':x0, 'beta':1}
exp1, stopIdx = aomip.fast_pgm(A,b,x0,aomip.proximalL2Squared,proxParams,iteration=3000,restart=True,callback=callback1)
exp1= exp1.reshape((512,512))
aomip.save_array_as_image(exp1,'exp1.png','img4')
aomip.plot_convergence(groundtruth,fs,stopIdx,'fast_pgm_l2squared_itr=1000','img4')
aomip.plot_objective_value(fy,stopIdx,'fast_pgm_l2squared_itr=1000','img4')

fs = []
fy = []
proxParams = {'v':x0, 'x':x0, 'beta':1}
exp2, stopIdx = aomip.fast_pgm(A,b,x0,aomip.proximalL2Squared,proxParams,iteration=3000,restart=True,restartItr=500,callback=callback1)
exp2= exp2.reshape((512,512))
aomip.save_array_as_image(exp2,'exp2.png','img4')
aomip.plot_convergence(groundtruth,fs,stopIdx,'fast_pgm_l2squared_itr=500','img4')
aomip.plot_objective_value(fy,stopIdx,'fast_pgm_l2squared_itr=500','img4')

fs = []
fy = []
proxParams = {'v':x0, 'x':x0, 'beta':1}
exp3, stopIdx = aomip.fast_pgm(A,b,x0,aomip.proximalL2Squared,proxParams,iteration=3000,restart=True,restartItr=250,callback=callback1)
exp3= exp3.reshape((512,512))
aomip.save_array_as_image(exp3,'exp3.png','img4')
aomip.plot_convergence(groundtruth,fs,stopIdx,'fast_pgm_l2squared_itr=250','img4')
aomip.plot_objective_value(fy,stopIdx,'fast_pgm_l2squared_itr=250','img4')


