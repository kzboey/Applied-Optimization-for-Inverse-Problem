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
data_path = (dataset["data_path"])["local"]
file_name = dataset["phantom1b"]

mat = loadmat(join(data_path,file_name))
data = mat['CtDataFull']
fullsino=data[0][0][1]
sino = bin_array(fullsino,4)
b=sino.flatten()


size = np.array([512, 512])

num_angles = 360
arc = 360

s2c = size[0] * 20
c2d = size[0] * 0.1
D = s2c + c2d
A = aomip.XrayOperator(size, [num_angles], np.linspace(0, arc, 70), s2c, c2d)
fbp=A.applyAdjoint(b)
print(fbp)

sigA = GradientDescent.get_largest_sigma(A,5)
print('largest singular value of A:',sigA)

ogm, iteration_OGM = GradientDescent.compute_OGM1(A,b,1000,beta=5,delta=0,regularizer='gd',iteration=5)
ogm= ogm.reshape((512,512))
print('iteration OGM:',iteration_OGM)
save_array_as_image(ogm,'OGM.png','img')

ogm_tikhonov, iteration_OGM = GradientDescent.compute_OGM1(A,b,1000,beta=5,delta=1,regularizer='tikhonov',iteration=5)
ogm= ogm.reshape((512,512))
print('iteration OGM:',iteration_OGM)
save_array_as_image(ogm,'OGM_tikhonov.png','img')