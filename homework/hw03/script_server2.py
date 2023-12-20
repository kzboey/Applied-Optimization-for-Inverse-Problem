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
file_name = dataset["phantom2a"]

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

x0 = -np.random.rand(A.shape[1])

ogm_tik, iteration_OGM = GradientDescent.compute_OGM1(A,b,10000,x0,beta=1,delta=0,regularizer='huber',iteration=20)
ogm_tik= ogm_tik.reshape((512,512))
ogm_tik[ogm_tik<0] = 0
print('iteration OGM:',iteration_OGM)
save_array_as_image(ogm_tik,'OGM_tikhonov.png','img2a')
imsave(join('img2a','1b_tikhonov.tif'), ogm_tik)