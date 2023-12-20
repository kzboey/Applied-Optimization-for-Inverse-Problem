import aomip
import numpy as np
from os.path import join
import matplotlib.pyplot as plt
from skimage.io import imread, imsave
import tifffile
import json

fs = []
fy = []
step = 1
def callback1(x,y,i):
    if i % step == 0:
        fs.append(x)
        fy.append(y)

## Preprocessing of the low dose dataset ##
def load_tiff_stack_with_metadata(file):
    if not (file.name.endswith('.tif') or file.name.endswith('.tiff')):
        raise FileNotFoundError('File has to be tif.')
    with tifffile.TiffFile(file) as tif:
        data = tif.asarray()
        metadata = tif.pages[0].tags["ImageDescription"].value
    metadata = metadata.replace("'", "\"")
    try:
        metadata = json.loads(metadata)
    except:
        print('The tiff file you try to open does not seem to have metadata attached.')
        metadata = None
    return data, metadata

# Read corresponding file from config 
def getData(file):
    with open("config.json", "r") as json_data_file:
        data = json.load(json_data_file)
    dataset = data["mayo_clinical"]
    data_path = (dataset["data_path"])["server"]
    file_name = dataset[file]
    file = join(data_path,file_name)
    tif_file = open(file, "rb")
    data, metadata = load_tiff_stack_with_metadata(tif_file)
    return data, metadata

# Obtain the nth projection
def getSinogram(data, index):
    return data[:,index,:]
    
def preproccessed(sino_data, metadata):
    # reconstruction size
    image_size = [512] * 2

    # extract angles in degree
    angles = np.degrees(np.array(metadata["angles"])[: metadata["rotview"]])[::2]

    # setup some spacing and sizes
    voxel_size = 0.7 # can be adjusted
    vox_scaling = 1 / voxel_size
    vol_spacing = vox_scaling
    vol_size = image_size

    # size of detector
    det_count = sino_data.shape[:-1]
    det_spacing = vox_scaling * metadata["du"]

    # Distances from source to center, and center to detector
    ds2c = vox_scaling * metadata["dso"]
    dc2d = vox_scaling * metadata["ddo"]

    # Obtain the forward projection operator
    A = aomip.XrayOperator(
        vol_size,
        det_count,
        angles,
        ds2c,
        dc2d,
        vol_spacing = [vol_spacing]*2,
        sino_spacing = [det_spacing]
    )
    
    return A

## Preprocessing ends ##

# full dose
datafd, metadatafd = getData('L333')
sinogramfd = getSinogram(datafd,25)
Afd = preproccessed(sinogramfd, metadatafd)
fsinofd = aomip.filter_sinogram(sinogramfd,'shepp-logan')
bfd = (fsinofd.flatten())[::2]

# normalized for tranmission log likelihood
max_fd = np.max(sinogramfd)
min_fd = np.min(sinogramfd)
normalized_sinofd = (sinogramfd - min_fd) / (max_fd - min_fd)
Afd_log = preproccessed(sinogramfd, metadatafd)
fsinofd_log = aomip.filter_sinogram(normalized_sinofd,'shepp-logan')
bfd_log = (fsinofd_log.flatten())[::2]

# quarter dose
dataqd, metadataqd = getData('L096')
sinogramqd = getSinogram(dataqd,25)
Aqd = preproccessed(sinogramqd, metadataqd)
fsinoqd = aomip.filter_sinogram(sinogramqd,'shepp-logan')
bqd = (fsinoqd.flatten())[::2]

# normalized for tranmission log likelihood
max_qd = np.max(sinogramqd)
min_qd = np.min(sinogramqd)
normalized_sinoqd = (sinogramqd - min_qd) / (max_qd - min_qd)
Aqd_log = preproccessed(sinogramqd, metadataqd)
fsinoqd_log = aomip.filter_sinogram(normalized_sinoqd,'shepp-logan')
bqd_log = (fsinoqd_log.flatten())[::2]

# Assume blank scan
b = np.ones_like(bfd) * 1e+8
x0 = np.zeros(512*512)



"""
    Transmission log likelihood
"""


# Gradient of descent algorithm for transmisssion log likelihood, equation (5) from sheet
df = lambda x : (Afd.T).dot(bfd_log - np.exp(-Afd.dot(x)))
# fixed step size
fs = []
iteration = 2000
alpha = np.ones(iteration)*0.01
beta = 0.001
exp1, stopIdx = aomip.subgradient(Aqd,bqd_log,x0,beta,alpha,df,iteration=iteration,callback=None)
exp1= exp1.reshape((512,512))
print('iteration :',stopIdx) 
aomip.save_array_as_image(exp1,'exp1.png','img4')
# Elapsed time: 71.6659243106842 seconds
# iteration : 2000


iteration = 5000
alpha = 1 / np.arange(1, iteration+1)
beta = 0.001
fs = []
exp2, stopIdx = aomip.subgradient(Aqd,bqd_log,x0,beta,alpha,df,iteration=iteration,callback=None)
exp2= exp2.reshape((512,512))
print('iteration :',stopIdx) 
aomip.save_array_as_image(exp2,'exp2.png','img4')
# Elapsed time: 181.41402339935303 seconds
# iteration : 5000

# Quarter dose reconstruction with pogm
fs = []
proxParams = {'v':x0, 'x':x0, 'beta':1, 'delta': 5 }
exp3, stopIdx = aomip.pogm(Aqd,bqd_log,x0,aomip.proximalHuber,proxParams,mygrad=df,iteration=2000,callback=None)
print('iteration :',stopIdx) 
exp3= exp3.reshape((512,512))
aomip.save_array_as_image(exp3,'exp3.png','img4')
# Elapsed time: 69.2211275100708 seconds
# iteration : 2000


# Quarter dose reconstruction with pogm
fs = []
proxParams = {'v':x0, 'x':x0, 'beta':1}
exp4, stopIdx = aomip.pogm(Aqd,bqd_log,x0,aomip.proximalL2Squared,proxParams,mygrad=df,iteration=2000,callback=None)
print('iteration :',stopIdx) 
exp4= exp4.reshape((512,512))
aomip.save_array_as_image(exp4,'exp4.png','img4')
# Elapsed time: 56.85691261291504 seconds
# iteration : 1589







