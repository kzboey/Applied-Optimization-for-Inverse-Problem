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
sinogram, A = utils.load_htc2022data(file,arc=60) # change arc to 360
fsino = aomip.filter_sinogram(sinogram,'shepp-logan')
b = fsino
x0 = np.zeros([512,512])

step = 1
def callback1(x,i):
    if i % step == 0:
        fs.append(x)

L = aomip.get_largest_sigma(A,1000) # norm of K
alpha = 1/L
alpha = float(format(1/L, '.0e'))

taus = np.logspace(-5, 4, 10)

for i,tau in enumerate(taus):
    lambd = (0.95*tau)/ (L**2)
    fs = []
    proxParamsf = {'v':x0, 'x':x0, 'sigma': alpha, 'tau':lambd}
    proxParamsg = {'v':x0, 'x':x0, 'y':b, 'sigma': alpha, 'beta':1,'g':aomip.proximalL2Squared}
    exp, stopIdx = aomip.admm_lasso_ct(A,b,x0,aomip.proximalL1,aomip.proximalTranslation,proxParamsf,proxParamsg,lambd,tau,iteration=200)
    file = '60degrees_exp{}.png'.format(i)
    tiff = '60degrees_exp{}.tif'.format(i)
    print('iteration :',stopIdx) 
    aomip.save_array_as_image(exp,file,'img1')
    imsave(join('img1',tiff), exp)


