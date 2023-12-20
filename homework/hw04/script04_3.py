import aomip
import numpy as np
import json
from os.path import join
from tifffile import imsave, imread
import matplotlib.pyplot as plt
import GradientDescent
from challenge import utils
from utils import save_array_as_image, plot_convergence, filter_sinogram

with open("config.json", "r") as json_data_file:
    data = json.load(json_data_file)
dataset = data["htc"]
data_path = (dataset["data_path"])["server"]
file_name = dataset["phantom1b"]
file = join(data_path,file_name)

## 90 degrees
sinogram, A = utils.load_htc2022data(file,arc=90)
fsino = filter_sinogram(sinogram,'shepp-logan')
b = fsino.flatten()
img_vector = 512*512
x0 = np.random.rand(img_vector)

# experiment 1
xBacktrack, itr, history = GradientDescent.backtracking_linesearch(A,b,x0,1e-3,0.1,0.1,eps=1e-6,iteration=200)
xBacktrack = xBacktrack.reshape((512,512))
print('iteration backtrack:',itr)
save_array_as_image(xBacktrack,'exp1.png','img3')
imsave(join('img3','exp1.tif'), xBacktrack)

# experiment 2
xBacktrack, itr, history = GradientDescent.backtracking_linesearch(A,b,x0,1e-3,0.1,0.1,beta=5,regularizer='tikhonov',eps=1e-6,iteration=200)
xBacktrack = xBacktrack.reshape((512,512))
print('iteration backtrack:',itr)
save_array_as_image(xBacktrack,'exp2.png','img3')
imsave(join('img3','exp2.tif'), xBacktrack)

# experiment 3
xbb2, itr, history = GradientDescent.bb_linesearch(A,b,x0,1e-5,bb=2,eps=1e-6,beta=5,regularizer='tikhonov',iteration=1000)
xbb2 = xbb2.reshape((512,512))
print('iteration BB2:',itr)
save_array_as_image(xbb2,'exp3.png','img3')
imsave(join('img3','exp3.tif'), xbb2)

# experiment 4
xIsta, itr, history = GradientDescent.ista(A,b,x0,beta=2,eps=1e-6,iteration=1000)
xIsta = xIsta.reshape((512,512))
print('iteration ISTA:',itr)
save_array_as_image(xIsta,'exp4.png','img3')
imsave(join('img3','exp4.tif'), xIsta)

# experiment 5
lower_bound = np.zeros(x0.size)-0.1
upper_bound = np.zeros(x0.size)+0.1
xpgd, itr, history = GradientDescent.projected_gradient_descent(A,b,x0,0.001,lower_bound,upper_bound,eps=1e-6,beta=5,regularizer='tikhonov',iteration=1000)
xpgd = xpgd.reshape((512,512))
print('iteration pgd:',itr)
save_array_as_image(xpgd,'exp5.png','img3')
imsave(join('img3','exp5.tif'), xpgd)

## 360 degrees
with open("config.json", "r") as json_data_file:
    data = json.load(json_data_file)
dataset = data["htc"]
data_path = (dataset["data_path"])["server"]
file_name = dataset["phantom5a"]
file = join(data_path,file_name)

sinogram, A = utils.load_htc2022data(file,arc=360)
fsino = filter_sinogram(sinogram,'shepp-logan')
b = fsino.flatten()
img_vector = 512*512
x0 = np.random.rand(img_vector)

# experiment 6
ogm_huber, iteration_OGM, _ = GradientDescent.compute_OGM1(A,b,x0,beta=5,delta=5,regularizer='huber',iteration=50)
ogm_huber= ogm_huber.reshape((512,512))
print('iteration huber:',iteration_OGM)
save_array_as_image(ogm_huber,'exp6.png','img3')
imsave(join('img3','exp6.tif'), ogm_huber)

# experiment 7
ogm_tikhonov, iteration_OGM, _ = GradientDescent.compute_OGM1(A,b,x0,beta=5,delta=5,regularizer='fair',iteration=100)
ogm_tikhonov= ogm_tikhonov.reshape((512,512))
print('iteration huber:',iteration_OGM)
save_array_as_image(ogm_tikhonov,'exp7.png','img3')
imsave(join('img3','exp7.tif'), ogm_tikhonov)

# experiment 8
lwb, iteration_lwb, _ = GradientDescent.compute_Landweber(A,b,0.001,x0,iteration=10000)
lwb= lwb.reshape((512,512))
print('iteration lwb:',iteration_lwb)
save_array_as_image(lwb,'exp8.png','img3')
imsave(join('img3','exp8.tif'), lwb)

# experiment 9
exp9, itr, history = GradientDescent.bb_linesearch(A,b,x0,1e-5,bb=2,eps=1e-6,beta=5,regularizer='tikhonov',iteration=1000)
exp9 = exp9.reshape((512,512))
print('iteration BB2:',itr)
save_array_as_image(xbb2,'exp9.png','img3')
imsave(join('img3','exp9.tif'), exp9)

# experiment 10
exp10, itr, _ = GradientDescent.conjugate_gradient_normal(A,b,x0,iteration=30000)
exp10= exp10.reshape((512,512))
print('iteration cg:',exp10)
save_array_as_image(exp10,'exp10.png','img3')
imsave(join('img3','exp10.tif'), exp10)

## 60 degrees
with open("config.json", "r") as json_data_file:
    data = json.load(json_data_file)
dataset = data["htc"]
data_path = (dataset["data_path"])["server"]
file_name = dataset["phantom3a"]
file = join(data_path,file_name)

sinogram, A = utils.load_htc2022data(file,arc=60)
fsino = filter_sinogram(sinogram,'shepp-logan')
b = fsino.flatten()
img_vector = 512*512
x0 = np.random.rand(img_vector)

# experiment 11
exp11, itr, history = GradientDescent.backtracking_linesearch(A,b,x0,1e-3,0.1,0.1,beta=5,regularizer='tikhonov',eps=1e-6,iteration=1000)
exp11 = exp11.reshape((512,512))
print('iteration backtrack:',itr)
save_array_as_image(exp11,'exp11.png','img3')
imsave(join('img3','exp11.tif'), exp11)

# experiment 12
exp12, itr, history = GradientDescent.bb_linesearch(A,b,x0,1e-5,bb=2,eps=1e-6,beta=5,regularizer='tikhonov',iteration=1000)
exp12 = exp12.reshape((512,512))
print('iteration BB2:',itr)
save_array_as_image(exp12,'exp12.png','img3')
imsave(join('img3','exp12.tif'), exp12)

# experiment 13
exp13, itr, history = GradientDescent.ista(A,b,x0,beta=2,eps=1e-6,iteration=5000)
exp13 = exp13.reshape((512,512))
print('iteration ISTA:',itr)
save_array_as_image(exp13,'exp13.png','img3')
imsave(join('img3','exp13.tif'), exp13)

# experiment 14
lower_bound = np.zeros(x0.size)-0.1
upper_bound = np.zeros(x0.size)+0.1
exp14, itr, history = GradientDescent.projected_gradient_descent(A,b,x0,0.001,lower_bound,upper_bound,eps=1e-6,beta=5,regularizer='tikhonov',iteration=5000)
exp14 = exp14.reshape((512,512))
print('iteration pgd:',itr)
save_array_as_image(exp14,'exp14.png','img3')
imsave(join('img3','exp14.tif'), exp14)
