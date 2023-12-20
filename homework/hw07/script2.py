import aomip
import numpy as np
from os.path import join
import matplotlib.pyplot as plt

def smooth(N):
    I, J = np.meshgrid(np.arange(N), np.arange(N))
    sigma = 0.25 * N
    c = np.array([[0.6*N, 0.6*N], [0.5*N, 0.3*N], [0.2*N, 0.7*N], [0.8*N, 0.2*N]])
    a = np.array([1, 0.5, 0.7, 0.9])
    img = np.zeros((N, N))
    for i in range(4):
        term1 = (I - c[i, 0])**2 / (1.2 * sigma )**2
        term2 = (J - c[i, 1])**2 / sigma**2
        img += a[i] * np.exp(-term1 - term2)
    return img

syntheticImg = smooth(256) # groundtruth
size = np.array([256, 256])
num_angles = 512
arc = 360
A = aomip.XrayOperator(size, [150], np.linspace(0, arc, num_angles), size[0]*100, size[0]*2)
sinogram = A.apply(syntheticImg)

x0 = np.zeros_like(syntheticImg)

stackedOpt = aomip.StackedOperator()
grad = aomip.FirstDerivative()
gradShape = grad.apply(x0)
gradBlock = np.zeros_like(gradShape)  

opts = [A,grad]
xstacked = [x0,x0]
bstacked = [sinogram, gradBlock]

L = aomip.get_largest_sigma(A,100) # norm of K
alpha = 1/L
alpha = float(format(1/L, '.0e'))

taus = np.logspace(-6, 1, 8)

step = 1
def callback1(x,i):
    if i % step == 0:
        fs.append(x)
        
# Anisotropic
for i,tau in enumerate(taus):
    fs = []
    lambd = (0.95*tau)/ (L**2)
    blockProxG = [aomip.proximalTranslation, aomip.proximalL1]
    proxParamg1 = {'v':x0,'y':-sinogram, 'sigma':alpha, 'beta':lambd, 'g':aomip.proximalL2Squared}
    proxParamg2 = {'v':x0, 'sigma':alpha, 'tau':lambd}
    blockProxpramsG = [proxParamg1,proxParamg2]
    reconImg, stopIdx = aomip.admm_tv(bstacked,xstacked,stackedOpt,opts,blockProxG,blockProxpramsG,lambd,tau,iteration=1000,callback=callback1)
    print("iteration: ",stopIdx)
    file = 'synthetic_Anisotrophic_exp{}.png'.format(i)
    aomip.save_array_as_image(reconImg,file,'img2')
    aomip.plot_convergence(syntheticImg,fs,stopIdx,file,'img2')

# Isotrophic
for i,tau in enumerate(taus):
    fs = []
    lambd = (0.95*tau)/ (L**2)
    blockProxG = [aomip.proximalTranslation, aomip.proximalL21]
    proxParamg1 = {'v':x0,'y':-sinogram, 'sigma':alpha, 'beta':lambd, 'g':aomip.proximalL2Squared}
    proxParamg2 = {'v':x0, 'sigma':alpha, 'tau':lambd}
    blockProxpramsG = [proxParamg1,proxParamg2]
    reconImg2, stopIdx = aomip.admm_tv(bstacked,xstacked,stackedOpt,opts,blockProxG,blockProxpramsG,lambd,tau,iteration=1000,callback=callback1)
    print("iteration: ",stopIdx)
    file = 'synthetic_Isotrophic_exp{}.png'.format(i)
    aomip.save_array_as_image(reconImg2,file,'img2')
    aomip.plot_convergence(syntheticImg,fs,stopIdx,file,'img2')