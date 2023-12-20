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
file_name = dataset["phantom7c"]
file = join(data_path,file_name)

## 60 degrees
sinogram, A = utils.load_htc2022data(file,arc=30) 
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
fs = []
grad = lambda x : A.applyAdjoint(A.apply(x) - sinogram)
exp1, stopIdx = aomip.subgradient(A,sinogram,x0,beta,alpha,grad,iteration=iteration)
print('iteration :',stopIdx) 
aomip.save_array_as_image(exp1,'exp1.png','img2')
imsave(join('img2','exp1.tif'), exp1)
# Elapsed time: 66.30176830291748 seconds
# iteration : 5000

# variable step size, 1/k
alpha = 1 / np.arange(1, iteration+1)
beta = 0.01
fs = []
exp2, stopIdx = aomip.subgradient(A,sinogram,x0,beta,alpha,grad,iteration=iteration)
print('iteration :',stopIdx) 
aomip.save_array_as_image(exp2,'exp2.png','img2')
imsave(join('img2','exp2.tif'), exp2)
# Elapsed time: 66.18613982200623 seconds
# iteration : 5000
        
# ADMM

L = aomip.get_largest_sigma(A,1000) # norm of K
alpha = 1/L
alpha = float(format(1/L, '.0e'))

tau = 1e-3

lambd = (0.95*tau)/ (L**2)
stackedOpt = aomip.StackedOperator()
grad = aomip.FirstDerivative()
gradShape = grad.apply(x0)
gradBlock = np.zeros_like(gradShape)  

opts = [A,grad]
xstacked = [x0,x0]
bstacked = [sinogram, gradBlock]

blockProxG = [aomip.proximalTranslation, aomip.proximalL1]
proxParamg1 = {'v':x0,'y':-sinogram, 'sigma':alpha, 'beta':lambd, 'g':aomip.proximalL2Squared}
proxParamg2 = {'v':x0, 'sigma':alpha, 'tau':lambd}
blockProxpramsG = [proxParamg1,proxParamg2]
exp3, stopIdx = aomip.admm_tv(bstacked,xstacked,stackedOpt,opts,blockProxG,blockProxpramsG,lambd,tau,iteration=5000)
print('iteration :',stopIdx) 
aomip.save_array_as_image(exp3,'exp3.png','img2')
imsave(join('img2','exp3.tif'), exp3)
# Elapsed time: 187.92652583122253 seconds
# iteration : 5000

# FPGM

fs = []
grad = lambda x : A.applyAdjoint(A.apply(x) - sinogram)
proxParams = {'v':x0, 'x':x0, 'beta':1, 'delta':10}
exp4, stopIdx = aomip.fast_pgm(A,b,x0,aomip.proximalHuber,proxParams,mygrad=grad,iteration=5000)
print('iteration :',stopIdx) 
aomip.save_array_as_image(exp4,'exp4.png','img2')
imsave(join('img2','exp4.tif'), exp4)
# Elapsed time: 50.826568841934204 seconds
# iteration : 5000

# POGM

proxParams = {'v':x0, 'x':x0, 'beta':10}
exp5, stopIdx = aomip.pogm(A,b,x0,aomip.proximalL2Squared,proxParams,mygrad=grad,iteration=5000)
print('iteration :',stopIdx) 
aomip.save_array_as_image(exp5,'exp5.png','img2')
imsave(join('img2','exp5.tif'), exp5)
# Elapsed time: 0.23142337799072266 seconds
# iteration : 22

## ISTA

exp6, stopIdx = aomip.ista(A,b,x0,beta=0.01,mygrad=grad,iteration=8000)
print('iteration :',stopIdx) 
aomip.save_array_as_image(exp6,'exp6.png','img2')
imsave(join('img2','exp6.tif'), exp6)
# Elapsed time: 75.92918658256531 seconds
# iteration : 8000

# bb1
# have to hard code this part to not use operator because it was previously implemented as such
b = fsino.flatten()
img_vector = 512*512
x0 = np.random.rand(img_vector)

exp7, stopIdx = aomip.bb_linesearch(A,b,x0,1e-3,bb=1,regularizer='tikhonov',beta=5,iteration=1000)
print('iteration :',stopIdx) 
exp7= exp7.reshape((512,512))
aomip.save_array_as_image(exp7,'exp7.png','img2')
imsave(join('img2','exp7.tif'), exp7)    
# Elapsed time: 1.7756826877593994 seconds
# iteration : 26






