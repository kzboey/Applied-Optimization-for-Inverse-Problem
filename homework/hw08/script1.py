import aomip
import numpy as np
from os.path import join
import matplotlib.pyplot as plt
import json
from challenge import utils
from tifffile import imsave, imread

with open("config.json", "r") as json_data_file:
    data = json.load(json_data_file)
dataset = data["htc"]
data_path = (dataset["data_path"])["server"]
file_name = dataset["phantom7a"]
file = join(data_path,file_name)

## 60 degrees
sinogram, A = utils.load_htc2022data(file,arc=60) 
fsino = aomip.filter_sinogram(sinogram,'shepp-logan')
b = fsino
x0 = np.zeros([512,512])
        
L = aomip.get_largest_sigma(A,1000) # norm of K
alpha = 1/L
alpha = float(format(1/L, '.0e'))

# Subgradient method        
      
# fixed step size
iteration = 5000
alpha = np.ones(iteration)*0.01
beta = 0.01
grad = lambda x : A.applyAdjoint(A.apply(x) - sinogram)
exp1, stopIdx = aomip.subgradient(A,sinogram,x0,beta,alpha,grad,iteration=iteration)
print('iteration :',stopIdx) 
aomip.save_array_as_image(exp1,'exp1.png','img1')
imsave(join('img1','exp1.tif'), exp1)
# Elapsed time: 69.35283851623535 seconds
# iteration : 5000

# variable step size, 1/k
alpha = 1 / np.arange(1, iteration+1)
beta = 0.01
fs = []
exp2, stopIdx = aomip.subgradient(A,sinogram,x0,beta,alpha,grad,iteration=iteration)
print('iteration :',stopIdx) 
aomip.save_array_as_image(exp2,'exp2.png','img1')
imsave(join('img1','exp2.tif'), exp2)
# Elapsed time: 74.70258712768555 seconds
# iteration : 5000
        
# ADMM

L = aomip.get_largest_sigma(A,1000) # norm of K
alpha = 1/L
alpha = float(format(1/L, '.0e'))

tau = 1e-3

lambd = (0.95*tau)/ (L**2)
proxParamsf = {'v':x0, 'x':x0, 'sigma': alpha, 'tau':lambd}
proxParamsg = {'v':x0, 'x':x0, 'y':b, 'sigma': alpha, 'beta':1,'g':aomip.proximalL2Squared}
exp3, stopIdx = aomip.admm_lasso_ct(A,b,x0,aomip.proximalL1,aomip.proximalTranslation,proxParamsf,proxParamsg,lambd,tau,iteration=3000)
print('iteration :',stopIdx) 
aomip.save_array_as_image(exp3,'exp3.png','img1')
imsave(join('img1','exp3.tif'), exp3)
# Elapsed time: 57.06966423988342 seconds
# iteration : 3000

# FPGM

fs = []
grad = lambda x : A.applyAdjoint(A.apply(x) - sinogram)
proxParams = {'v':x0, 'x':x0, 'beta':5}
exp4, stopIdx = aomip.fast_pgm(A,b,x0,aomip.proximalL2Squared,proxParams,mygrad=grad,iteration=3000)
aomip.save_array_as_image(exp4,'exp4.png','img1')
imsave(join('img1','exp4.tif'), exp4)
# Elapsed time: 28.48592233657837 seconds

# POGM

fs = []
proxParams = {'v':x0, 'x':x0, 'beta':10}
exp5, stopIdx = aomip.pogm(A,b,x0,aomip.proximalL2Squared,proxParams,mygrad=grad,iteration=3000)
print('iteration :',stopIdx) 
aomip.save_array_as_image(exp5,'exp5.png','img1')
imsave(join('img1','exp5.tif'), exp5)
# Elapsed time: 1.261713981628418 seconds
# iteration : 24


## ISTA

exp6, stopIdx = aomip.ista(A,b,x0,beta=0.01,mygrad=grad,iteration=3000)
print('iteration :',stopIdx) 
aomip.save_array_as_image(exp6,'exp6.png','img1')
imsave(join('img1','exp6.tif'), exp6)
# Elapsed time: 31.162291526794434 seconds
# iteration : 3000

# bb2
# have to hard code this part to not use operator because it was previously implemented as such
b = fsino.flatten()
img_vector = 512*512
x0 = np.random.rand(img_vector)

exp7, stopIdx = aomip.bb_linesearch(A,b,x0,1e-3,bb=2,regularizer='tikhonov',beta=5,iteration=1000)
print('iteration :',stopIdx) 
exp7= exp7.reshape((512,512))
aomip.save_array_as_image(exp7,'exp7.png','img1')
imsave(join('img1','exp7.tif'), exp7)    
# Elapsed time: 1.7747914791107178 seconds
# iteration : 28


## ISTA again
grad = lambda x : A.applyAdjoint(A.apply(x) - sinogram)
exp8, stopIdx = aomip.ista(A,b,x0,beta=0.01,mygrad=grad,iteration=5000)
print('iteration :',stopIdx) 
aomip.save_array_as_image(exp8,'exp8.png','img1')
imsave(join('img1','exp8.tif'), exp8)
# Elapsed time: 54.936132192611694 seconds
# iteration : 5000

## FPGM with L1 
fs = []

proxParams = {'v':x0, 'x':x0, 'tau':1}
exp9, stopIdx = aomip.fast_pgm(A,b,x0,aomip.proximalL1,proxParams,mygrad=grad,iteration=5000)
print('iteration :',stopIdx) 
aomip.save_array_as_image(exp9,'exp9.png','img1')
imsave(join('img1','exp9.tif'), exp9)
# Elapsed time: 54.45972013473511 seconds
# iteration : 5000



