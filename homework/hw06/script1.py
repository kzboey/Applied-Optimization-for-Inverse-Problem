import aomip
import numpy as np
from os.path import join
import matplotlib.pyplot as plt

size = np.array([512, 512])
phantom = aomip.shepp_logan(size)
num_angles = 512
arc = 360

A = aomip.XrayOperator(size, [750], np.linspace(0, arc, num_angles), size[0]*100, size[0]*2)
sinogram = A.apply(phantom)
fsino = aomip.filter_sinogram(sinogram,'shepp-logan')
b = fsino.flatten()

img_vector = 512*512
x0 = np.random.rand(img_vector)
groundtruth = phantom.flatten()

step = 1
def callback1(x,y,i):
    if i % step == 0:
        fs.append(x)
        fy.append(y)
        
fs = []
fy = []
proxParams = {'v':x0, 'x':x0, 'tau':1}
exp1, stopIdx = aomip.pgm(A,b,x0,aomip.proximalL1,proxParams,iteration=1000,callback=callback1)
print('iteration :',stopIdx) 
exp1= exp1.reshape((512,512))
aomip.save_array_as_image(exp1,'exp1.png','img1')
aomip.plot_convergence(groundtruth,fs,stopIdx,'pgm_l1','img1')
aomip.plot_objective_value(fy,stopIdx,'pgm_l1','img1')

fs = []
fy = []
proxParams = {'v':x0, 'x':x0, 'tau':1}
exp2, stopIdx = aomip.fast_pgm(A,b,x0,aomip.proximalL1,proxParams,iteration=1000,callback=callback1)
print('iteration :',stopIdx) 
exp2= exp2.reshape((512,512))
aomip.save_array_as_image(exp2,'exp2.png','img1')
aomip.plot_convergence(groundtruth,fs,stopIdx,'fast_pgm_l1','img1')
aomip.plot_objective_value(fy,stopIdx,'fast_pgm_l1','img1')
                           
fs = []
fy = []
proxParams = {'v':x0, 'x':x0, 'tau':0.1}
exp3, stopIdx = aomip.pogm(A,b,x0,aomip.proximalL1,proxParams,iteration=1000,callback=callback1)
print('iteration :',stopIdx) 
exp3= exp3.reshape((512,512))
aomip.save_array_as_image(exp3,'exp3.png','img1')
aomip.plot_convergence(groundtruth,fs,stopIdx,'pogm_l1','img1')
aomip.plot_objective_value(fy,stopIdx,'pogm_l1','img1')                           

fs = []
fy = []
proxParams = {'v':x0, 'x':x0, 'beta':1}
exp4, stopIdx = aomip.pogm(A,b,x0,aomip.proximalL2Squared,proxParams,iteration=500,callback=callback1)
print('iteration :',stopIdx) 
exp4= exp4.reshape((512,512))
aomip.save_array_as_image(exp4,'exp4.png','img1')
aomip.plot_convergence(groundtruth,fs,stopIdx,'pogm_l2','img1')
aomip.plot_objective_value(fy,stopIdx,'pogm_l2','img1') 

fs = []
fy = []
proxParams = {'v':x0, 'x':x0, 'beta':1}
exp5, stopIdx = aomip.pgm(A,b,x0,aomip.proximalL2Squared,proxParams,iteration=500,callback=callback1)
print('iteration :',stopIdx) 
exp5= exp5.reshape((512,512))
aomip.save_array_as_image(exp5,'exp5.png','img1')
aomip.plot_convergence(groundtruth,fs,stopIdx,'pgm_l2','img1')
aomip.plot_objective_value(fy,stopIdx,'pgm_l2','img1') 