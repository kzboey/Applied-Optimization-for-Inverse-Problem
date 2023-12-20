import aomip
import numpy as np
import json
from os.path import join
from tifffile import imsave, imread
from scipy.io import loadmat
import matplotlib.pyplot as plt
from preprocessing import bin_array
import GradientDescent
from utils import save_array_as_image

with open("config.json", "r") as json_data_file:
    data = json.load(json_data_file)
dataset = data["htc"]
data_path = (dataset["data_path"])["server"]
file_name = dataset["phantom1b"]

mat = loadmat(join(data_path,file_name))
data = mat['CtDataFull']
fullsino=data[0][0][1]
sino = bin_array(fullsino,2)
b=sino.flatten()


size = np.array([512, 512])

num_angles = 360
arc = 360

s2c = size[0] * 20
c2d = size[0] * 0.1
D = s2c + c2d
A = aomip.XrayOperator(size, [num_angles], np.linspace(0, arc, 280), s2c, c2d)

###################### Filter ##############################################
H = np.linspace(-1, 1, sino.shape[0])

ram_lak = np.abs(H)
shepp_logan = np.abs(H) * np.sinc(H / 2)
cosine = np.abs(H) * np.cos(H * np.pi / 2)

h = np.tile(shepp_logan, (280, 1)).T
fftsino = np.fft.fft(sino, axis=0)
projection = np.fft.fftshift(fftsino, axes=1) * np.fft.fftshift(h, axes=0)
fsino = np.real(np.fft.ifft(np.fft.ifftshift(projection, axes=1), axis=0))
###################### Filter ##############################################

b = fsino.flatten()

############################################################################################################################
x0 = np.ones(A.shape[1])*-50

"""
ogm_none, iteration_OGM = GradientDescent.compute_OGM1(A,fsino.flatten(),50000,x0,beta=5,delta=5,regularizer='gd',iteration=25)
ogm_none= ogm_none.reshape((512,512))
ogm_none[ogm_none<0] = 0 
print('iteration :',iteration_OGM)
save_array_as_image(ogm_none,'OGM_none.png','img2')
imsave(join('img2','OGM_none.tif'), ogm_none)

ogm_tikhonov, iteration_OGM = GradientDescent.compute_OGM1(A,fsino.flatten(),50000,x0,beta=5,delta=5,regularizer='tikhonov',iteration=25)
ogm_tikhonov= ogm_tikhonov.reshape((512,512))
ogm_tikhonov[ogm_tikhonov<0] = 0 
print('iteration huber:',iteration_OGM)
save_array_as_image(ogm_tikhonov,'OGM_tikhonov.png','img2')
imsave(join('img2','OGM_tikhonov.tif'), ogm_tikhonov)

ogm_huber, iteration_OGM = GradientDescent.compute_OGM1(A,fsino.flatten(),50000,x0,beta=5,delta=5,regularizer='huber',iteration=25)
ogm_huber= ogm_huber.reshape((512,512))
ogm_huber[ogm_huber<0] = 0 
print('iteration huber:',iteration_OGM)
save_array_as_image(ogm_huber,'OGM_huber.png','img2')
imsave(join('img2','OGM_huber.tif'), ogm_huber)

ogm_fair, iteration_OGM = GradientDescent.compute_OGM1(A,fsino.flatten(),50000,x0,beta=5,delta=5,regularizer='fair',iteration=25)
ogm_fair= ogm_fair.reshape((512,512))
ogm_fair[ogm_fair<0] = 0 
print('iteration huber:',iteration_OGM)
save_array_as_image(ogm_fair,'OGM_fair.png','img2')
imsave(join('img2','OGM_fair.tif'), ogm_fair)


gd, iteration_gd_1 = GradientDescent.compute_GD(A,b,0.000005,x0,beta=5,regularizer='gd',iteration=1000) 
gd= gd.reshape((512,512))
gd[gd<0] = 0 
print('iteration 1:',iteration_gd_1)
save_array_as_image(gd,'gd_vanilla.png','img')
imsave(join('img2','gd_vanilla.tif'), gd)

gd_tikhonov, iteration_gd_1 = GradientDescent.compute_GD(A,b,0.000005,x0,beta=5,regularizer='tikhonov',iteration=1000) 
gd_tikhonov= gd_tikhonov.reshape((512,512))
gd_tikhonov[gd_tikhonov<0] = 0 
print('iteration 1:',iteration_gd_1)
save_array_as_image(gd_tikhonov,'gd_tikhonov.png','img')
imsave(join('img2','gd_tikhonov.tif'), gd_tikhonov)

gd_huber, iteration_gd_1 = GradientDescent.compute_GD(A,b,0.000005,x0,beta=2,delta=2,regularizer='huber',iteration=1000) 
gd_huber= gd_huber.reshape((512,512))
gd_huber[gd_huber<0] = 0 
print('iteration 1:',iteration_gd_1)
save_array_as_image(gd_tikhonov,'gd_huber.png','img')
imsave(join('img2','gd_huber.tif'), gd_huber)

gd_fair, iteration_gd_1 = GradientDescent.compute_GD(A,b,0.000005,x0,beta=2,delta=2,regularizer='fair',iteration=1000) 
gd_fair= gd_fair.reshape((512,512))
gd_fair[gd_fair<0] = 0 
print('iteration 1:',iteration_gd_1)
save_array_as_image(gd_fair,'gd_fair.png','img')
imsave(join('img2','gd_fair.tif'), gd_fair)


fgm_none, iteration_FGM = GradientDescent.compute_FGM1(A,fsino.flatten(),50000,x0,beta=5,delta=5,regularizer='gd',iteration=25)
fgm_none= fgm_none.reshape((512,512))
fgm_none[fgm_none<0] = 0 
print('iteration :',iteration_FGM)
save_array_as_image(fgm_none,'FGM_none.png','img2')
imsave(join('img2','FGM_none.tif'), fgm_none)

fgm_tikhonov, iteration_FGM = GradientDescent.compute_FGM1(A,fsino.flatten(),50000,x0,beta=5,delta=5,regularizer='tikhonov',iteration=25)
fgm_tikhonov= fgm_tikhonov.reshape((512,512))
fgm_tikhonov[fgm_tikhonov<0] = 0 
print('iteration huber:',iteration_FGM)
save_array_as_image(fgm_tikhonov,'FGM_tikhonov.png','img2')
imsave(join('img2','FGM_tikhonov.tif'), fgm_tikhonov)

fgm_huber, iteration_FGM = GradientDescent.compute_FGM1(A,fsino.flatten(),50000,x0,beta=5,delta=5,regularizer='huber',iteration=25)
fgm_huber= fgm_huber.reshape((512,512))
fgm_huber[fgm_huber<0] = 0 
print('iteration huber:',iteration_FGM)
save_array_as_image(fgm_huber,'FGM_huber.png','img2')
imsave(join('img2','FGM_huber.tif'), fgm_huber)

fgm_fair, iteration_FGM = GradientDescent.compute_FGM1(A,fsino.flatten(),50000,x0,beta=5,delta=5,regularizer='fair',iteration=25)
fgm_fair= fgm_fair.reshape((512,512))
fgm_fair[fgm_fair<0] = 0 
print('iteration huber:',iteration_FGM)
save_array_as_image(fgm_fair,'FGM_fair.png','img2')
imsave(join('img2','FGM_fair.tif'), fgm_fair)

sigA = GradientDescent.get_largest_sigma(A,100)
print('largest singular value of A:',sigA)
bound = 2/(sigA**2)

lwb, iteration_lwb = GradientDescent.compute_Landweber(A,b,0.0001,x0,iteration=1000)
lwb= lwb.reshape((512,512))
lwb[lwb<0] = 0 
print('iteration lwb:',iteration_lwb)
save_array_as_image(lwb,'Landweber.png','img2')
imsave(join('img2','Landweber.tif'), lwb)
"""

sirt, iteration_sirt = GradientDescent.compute_SIRT(A,b,0.0001,x0,iteration=1000)
sirt= sirt.reshape((512,512))
sirt[sirt<0] = 0 
print('iteration sirt:',iteration_sirt)
save_array_as_image(sirt,'sirt.png','img2')
imsave(join('img2','sirt.tif'), sirt)

cg, iteration_cg = GradientDescent.conjugate_gradient_normal(A,b,x0,iteration=20)
cg= cg.reshape((512,512))
cg[cg<0] = 0 
print('iteration sirt:',iteration_cg)
save_array_as_image(cg,'cg.png','img2')
imsave(join('img2','cg.tif'), cg)