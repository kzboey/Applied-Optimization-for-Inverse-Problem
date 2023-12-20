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
file_name = dataset["phantom3c"]
file = join(data_path,file_name)

## 60 degrees
sinogram, A = utils.load_htc2022data(file,arc=30)
fsino = aomip.filter_sinogram(sinogram,'shepp-logan')
b = fsino
x0 = np.zeros([512,512])

stackedOpt = aomip.StackedOperator()
grad = aomip.FirstDerivative()
gradShape = grad.apply(x0)
gradBlock = np.zeros_like(gradShape)  

opts = [A,grad]
xstacked = [x0,x0]
bstacked = [b, gradBlock]

step = 1
def callback1(x,i):
    if i % step == 0:
        fs.append(x)
        
L = aomip.get_largest_sigma(A,100) # norm of K
alpha = 1/L
alpha = float(format(1/L, '.0e'))

taus = np.logspace(-5, 4, 10)

# Anisotropic
for i,tau in enumerate(taus):
    lambd = (0.95*tau)/ (L**2)
    blockProxG = [aomip.proximalTranslation, aomip.proximalL1]
    proxParamg1 = {'v':x0,'y':-sinogram, 'sigma':alpha, 'beta':lambd, 'g':aomip.proximalL2Squared}
    proxParamg2 = {'v':x0, 'sigma':alpha, 'tau':lambd}
    blockProxpramsG = [proxParamg1,proxParamg2]
    exp, stopIdx = aomip.admm_tv(bstacked,xstacked,stackedOpt,opts,blockProxG,blockProxpramsG,lambd,tau,iteration=500)
    print("iteration: ",stopIdx)
    file = 'Anisotrophic_30degrees_exp{}.png'.format(i)
    tiff = 'Anisotrophic_30degrees_exp{}.tif'.format(i)
    aomip.save_array_as_image(exp,file,'img3')
    imsave(join('img3',tiff), exp)
    
# Isotrophic
for i,tau in enumerate(taus):
    lambd = (0.95*tau)/ (L**2)
    blockProxG = [aomip.proximalTranslation, aomip.proximalL21]
    proxParamg1 = {'v':x0,'y':-sinogram, 'sigma':alpha, 'beta':lambd, 'g':aomip.proximalL2Squared}
    proxParamg2 = {'v':x0, 'sigma':alpha, 'tau':lambd}
    blockProxpramsG = [proxParamg1,proxParamg2]
    exp, stopIdx = aomip.admm_tv(bstacked,xstacked,stackedOpt,opts,blockProxG,blockProxpramsG,lambd,tau,iteration=500)
    print("iteration: ",stopIdx)
    file = 'Isotrophic_30degrees_exp{}.png'.format(i)
    tiff = 'Isotrophic_30degrees_exp{}.tif'.format(i)
    aomip.save_array_as_image(exp,file,'img3')
    imsave(join('img3',tiff), exp)