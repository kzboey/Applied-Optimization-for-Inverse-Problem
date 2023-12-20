import aomip
import numpy as np
from os.path import join
import matplotlib.pyplot as plt
import json
from challenge import utils
from tifffile import imsave, imread

step = 1
def callback1(x,y,i):
    if i % step == 0:
        fs.append(x)
        fy.append(y)
        
with open("config.json", "r") as json_data_file:
    data = json.load(json_data_file)
dataset = data["htc"]
data_path = (dataset["data_path"])["server"]
file_name = dataset["phantom6a"]
file = join(data_path,file_name)

## 90 degrees
sinogram, A = utils.load_htc2022data(file,arc=90)
fsino = aomip.filter_sinogram(sinogram,'shepp-logan')
b = fsino.flatten()
img_vector = 512*512
x0 = np.random.rand(img_vector)

fs = []
fy = []
proxParams = {'v':x0, 'x':x0, 'beta':1}
exp1, stopIdx = aomip.fast_pgm(A,b,x0,aomip.proximalL2Squared,proxParams,iteration=5000,restart=True,restartItr=500,callback=callback1)
print('iteration :',stopIdx) # 408
exp1= exp1.reshape((512,512))
aomip.save_array_as_image(exp1,'exp1.png','img4')
imsave(join('img4','exp1.tif'), exp1)

fs = []
fy = []
proxParams = {'v':x0, 'x':x0, 'beta':1, 'delta': 10 }
exp2, stopIdx = aomip.fast_pgm(A,b,x0,aomip.proximalHuber,proxParams,iteration=5000,restart=True,restartItr=500,callback=callback1)
print('iteration :',stopIdx) # 3421
exp2= exp2.reshape((512,512))
aomip.save_array_as_image(exp2,'exp2.png','img4')
imsave(join('img4','exp2.tif'), exp2)

fs = []
fy = []
proxParams = {'v':x0, 'x':x0, 'beta':1, 'delta': 10 }
exp3, stopIdx = aomip.fast_pgm(A,b,x0,aomip.proximalIdentity,proxParams,delta=proxParams['delta'],regularizer='huber',iteration=5000,restart=True,restartItr=500,callback=callback1)
print('iteration :',stopIdx) # 385
exp3= exp3.reshape((512,512))
aomip.save_array_as_image(exp3,'exp3.png','img4')
imsave(join('img4','exp3.tif'), exp3)

fs = []
fy = []
proxParams = {'v':x0, 'x':x0, 'tau':0.1, 'beta':0.1}
exp4, stopIdx = aomip.fast_pgm(A,b,x0,aomip.proximalElasticNet,proxParams,momentum=2,iteration=5000,restart=True,restartItr=500,callback=callback1)
print('iteration :',stopIdx)  # 56
exp4= exp4.reshape((512,512))
aomip.save_array_as_image(exp4,'exp4.png','img4')
imsave(join('img4','exp4.tif'), exp4)

