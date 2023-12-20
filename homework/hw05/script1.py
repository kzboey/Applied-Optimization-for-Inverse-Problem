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

fs = []
fy = []
proxParams = {'v':x0, 'x':x0, 'beta':1}
exp1, stopIdx = aomip.pgm(A,b,x0,aomip.proximalL2Squared,proxParams,iteration=5000,callback=callback1)
exp1= exp1.reshape((512,512))
aomip.save_array_as_image(exp1,'exp1.png','img1')
aomip.plot_convergence(groundtruth,fs,stopIdx,'pgm_l2squared','img1')
aomip.plot_objective_value(fy,stopIdx,'pgm_l2squared','img1')

fs = []
fy = []
proxParams = {'v':x0, 'x':x0, 'beta':1, 'delta': 10 }
exp2, stopIdx = aomip.pgm(A,b,x0,aomip.proximalHuber,proxParams,iteration=5000,callback=callback1)
exp2= exp2.reshape((512,512))
aomip.save_array_as_image(exp2,'exp2.png','img1')
aomip.plot_convergence(groundtruth,fs,stopIdx,'pgm_huber','img1')
aomip.plot_objective_value(fy,stopIdx,'pgm_huber','img1')

fs = []
fy = []
proxParams = {'v':x0, 'x':x0, 'beta':1}
exp3, stopIdx = aomip.fast_pgm(A,b,x0,aomip.proximalL2Squared,proxParams,iteration=5000,callback=callback1)
exp3= exp3.reshape((512,512))
aomip.save_array_as_image(exp3,'exp3.png','img1')
aomip.plot_convergence(groundtruth,fs,stopIdx,'fast_pgm_l2squared','img1')
aomip.plot_objective_value(fy,stopIdx,'fast_pgm_l2squared','img1')

fs = []
fy = []
proxParams = {'v':x0, 'x':x0, 'beta':1}
exp3b, stopIdx = aomip.fast_pgm(A,b,x0,aomip.proximalL2Squared,proxParams,momentum=2,iteration=5000,callback=callback1)
exp3b= exp3b.reshape((512,512))
aomip.save_array_as_image(exp3b,'exp3b.png','img1')
aomip.plot_convergence(groundtruth,fs,stopIdx,'fast_pgm_l2squared_v2','img1')
aomip.plot_objective_value(fy,stopIdx,'fast_pgm_l2squared_v2','img1')

fs = []
fy = []
proxParams = {'v':x0, 'x':x0, 'beta':1, 'delta': 10 }
exp4, stopIdx = aomip.fast_pgm(A,b,x0,aomip.proximalHuber,proxParams,iteration=5000,callback=callback1)
exp4= exp4.reshape((512,512))
aomip.save_array_as_image(exp4,'exp4.png','img1')
aomip.plot_convergence(groundtruth,fs,stopIdx,'fast_pgm_huber','img1')
aomip.plot_objective_value(fy,stopIdx,'fast_pgm_huber','img1')

fs = []
fy = []
proxParams = {'v':x0, 'x':x0, 'beta':1, 'delta': 10 }
exp4b, stopIdx = aomip.fast_pgm(A,b,x0,aomip.proximalHuber,proxParams,momentum=2,iteration=5000,callback=callback1)
exp4b= exp4b.reshape((512,512))
aomip.save_array_as_image(exp4b,'exp4b.png','img1')
aomip.plot_convergence(groundtruth,fs,stopIdx,'fast_pgm_huber_v2','img1')
aomip.plot_objective_value(fy,stopIdx,'fast_pgm_huber_v2','img1')

fs = []
fy = []
proxParams = {'v':x0, 'x':x0, 'y':np.ones(img_vector), 'g':aomip.proximalL2Squared, 'beta':1}
exp5, stopIdx = aomip.pgm(A,b,x0,aomip.proximalTranslation,proxParams,iteration=5000,callback=callback1)
exp5 = exp5.reshape((512,512))
aomip.save_array_as_image(exp5,'exp5.png','img1')
aomip.plot_convergence(groundtruth,fs,stopIdx,'fast_pgm_translationl2','img1')
aomip.plot_objective_value(fy,stopIdx,'fast_pgm_translationl2','img1')


## (ii) Uniquesness of formulation
fs = []
fy = []
proxParams = {'v':x0, 'x':x0, 'beta':1}
exp6, stopIdx = aomip.pgm(A,b,x0,aomip.proximalIdentity,proxParams,beta=proxParams['beta'],regularizer='tikhonov',iteration=5000,callback=callback1)
exp6= exp6.reshape((512,512))
aomip.save_array_as_image(exp6,'exp6.png','img1')
aomip.plot_convergence(groundtruth,fs,stopIdx,'pgm_l2squared_h(x)=0','img1')
aomip.plot_objective_value(fy,stopIdx,'pgm_l2squared_h(x)=0','img1')

fs = []
fy = []
proxParams = {'v':x0, 'x':x0, 'beta':1}
exp7, stopIdx = aomip.fast_pgm(A,b,x0,aomip.proximalIdentity,proxParams,beta=proxParams['beta'],regularizer='tikhonov',iteration=5000,callback=callback1)
exp7= exp7.reshape((512,512))
aomip.save_array_as_image(exp7,'exp7.png','img1')
aomip.plot_convergence(groundtruth,fs,stopIdx,'fast_pgm_l2squared_h(x)=0','img1')
aomip.plot_objective_value(fy,stopIdx,'fast_pgm_l2squared_h(x)=0','img1')

fs = []
fy = []
proxParams = {'v':x0, 'x':x0, 'beta':1}
exp7b, stopIdx = aomip.fast_pgm(A,b,x0,aomip.proximalIdentity,proxParams,beta=proxParams['beta'],regularizer='tikhonov',momentum=2,iteration=5000,callback=callback1)
exp7b= exp7b.reshape((512,512))
aomip.save_array_as_image(exp7b,'exp7b.png','img1')
aomip.plot_convergence(groundtruth,fs,stopIdx,'fast_pgm_l2squared_h(x)=0_v2','img1')
aomip.plot_objective_value(fy,stopIdx,'fast_pgm_l2squared_h(x)=0_v2','img1')

fs = []
fy = []
proxParams = {'v':x0, 'x':x0, 'beta':1, 'delta': 10 }
exp8, stopIdx = aomip.pgm(A,b,x0,aomip.proximalIdentity,proxParams,beta=proxParams['beta'],delta=proxParams['delta'],regularizer='huber',iteration=5000, callback=callback1)
exp8= exp8.reshape((512,512))
aomip.save_array_as_image(exp8,'exp8.png','img1')
aomip.plot_convergence(groundtruth,fs,stopIdx,'pgm_huber_h(x)=0','img1')
aomip.plot_objective_value(fy,stopIdx,'pgm_huber_h(x)=0','img1')

fs = []
fy = []
proxParams = {'v':x0, 'x':x0, 'beta':1, 'delta': 10 }
exp9, stopIdx = aomip.fast_pgm(A,b,x0,aomip.proximalIdentity,proxParams,beta=proxParams['beta'],delta=proxParams['delta'],regularizer='huber',iteration=5000, callback=callback1)
exp9= exp9.reshape((512,512))
aomip.save_array_as_image(exp9,'exp9.png','img1')
aomip.plot_convergence(groundtruth,fs,stopIdx,'fast_pgm_huber_h(x)=0','img1')
aomip.plot_objective_value(fy,stopIdx,'fast_pgm_huber_h(x)=0','img1')

## (iii)  Elastic Net Formulation
fs = []
fy = []
proxParams = {'v':x0, 'x':x0, 'tau':0.1, 'beta':0.1}
exp10, stopIdx = aomip.pgm(A,b,x0,aomip.proximalElasticNet,proxParams,iteration=5000,callback=callback1)
exp10= exp10.reshape((512,512))
aomip.save_array_as_image(exp10,'exp10.png','img1')
aomip.plot_convergence(groundtruth,fs,stopIdx,'pgm_elastic_reg=0.1','img1')
aomip.plot_objective_value(fy,stopIdx,'pgm_elastic_reg=0.1','img1')


fs = []
fy = []
proxParams = {'v':x0, 'x':x0, 'tau':0.01, 'beta':0.01}
exp11, stopIdx = aomip.pgm(A,b,x0,aomip.proximalElasticNet,proxParams,iteration=5000,callback=callback1)
exp11= exp11.reshape((512,512))
aomip.save_array_as_image(exp11,'exp11.png','img1')
aomip.plot_convergence(groundtruth,fs,stopIdx,'pgm_elastic_reg=0.01','img1')
aomip.plot_objective_value(fy,stopIdx,'pgm_elastic_reg=0.01','img1')

fs = []
fy = []
proxParams = {'v':x0, 'x':x0, 'tau':0.1, 'beta':0.1}
exp12, stopIdx = aomip.fast_pgm(A,b,x0,aomip.proximalElasticNet,proxParams,iteration=5000,callback=callback1)
exp12= exp12.reshape((512,512))
aomip.save_array_as_image(exp12,'exp12.png','img1')
aomip.plot_convergence(groundtruth,fs,stopIdx,'fast_pgm_elastic_reg=0.1','img1')
aomip.plot_objective_value(fy,stopIdx,'fast_pgm_elastic_reg=0.1','img1')

fs = []
fy = []
proxParams = {'v':x0, 'x':x0, 'tau':0.01, 'beta':0.01}
exp13, stopIdx = aomip.fast_pgm(A,b,x0,aomip.proximalElasticNet,proxParams,iteration=5000,callback=callback1)
exp13= exp13.reshape((512,512))
aomip.save_array_as_image(exp13,'exp13.png','img1')
aomip.plot_convergence(groundtruth,fs,stopIdx,'fast_pgm_pgm_elastic_reg=0.01','img1')
aomip.plot_objective_value(fy,stopIdx,'fast_pgm_pgm_elastic_reg=0.01','img1')

