from os.path import join
import pyelsa as elsa
import aomip
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.image as mig
import utils
import matplotlib.pyplot as plt

## Prepare data
local=True

data_path = '/srv/ceph/share-all/aomip/2686726_Walnut1/Walnut1/Projections/tubeV1'
projs_name = 'scan_{:06}.tif'
dark_name = 'di000000.tif'
flat_name = ['io000000.tif', 'io000001.tif']
vecs_name = 'scan_geom_corrected.geom'
projs_rows = 972
projs_cols = 768

# projection file indices, we need to read in the projection in reverse order due to the portrait mode acquision
projs_idx  = range(1200,0, -1)

num_projections = 1200 #vecs.shape[0]

# create the numpy array which will receive projection data from tiff files
projs = np.zeros((num_projections, projs_rows, projs_cols), dtype=np.float32)

# Changing orientation from landscape to portrait mode
trafo = lambda image : np.transpose(np.flipud(image))

# load flat-field and dark-fields
# there are two flat-field images (taken before and after acquisition), we simply average them
dark = trafo(plt.imread(join(data_path, dark_name)))
flat = np.zeros((2, projs_rows, projs_cols), dtype=np.float32)

# Combine avereate of the flat field image
for i, fn in enumerate(flat_name):
    flat[i] = trafo(plt.imread(join(data_path, fn)))
flat =  np.mean(flat,axis=0)

# load projection data
for i in range(num_projections):
    projs[i] = trafo(plt.imread(join(data_path, projs_name.format(projs_idx[i]))))

""" Homework 1: preprocessing with slicing"""
""" Start """

slice_idx_50 = 50
slice_idx_100 = 100
slice_idx_200 = 200
slice_idx_400 = 400
slice_idx_800 = 800

sliced_sinogram_50 = np.empty((num_projections, projs_cols), dtype=np.float32)
sliced_sinogram_100 = np.empty((num_projections, projs_cols), dtype=np.float32)
sliced_sinogram_200 = np.empty((num_projections, projs_cols), dtype=np.float32)
sliced_sinogram_400 = np.empty((num_projections, projs_cols), dtype=np.float32)
sliced_sinogram_800 = np.empty((num_projections, projs_cols), dtype=np.float32)

for i in range(num_projections):
    proj=projs[i]
    
    #slice row 50 for every projection
    row_50 = proj[slice_idx_50, :]
    sliced_sinogram_50[i, :] = row_50
    
    #slice row 100 for every projection
    row_100 = proj[slice_idx_100, :]
    sliced_sinogram_100[i, :] = row_100
    
    #slice row 200 for every projection
    row_200 = proj[slice_idx_200, :]
    sliced_sinogram_200[i, :] = row_200
    
    #slice row 400 for every projection
    row_400 = proj[slice_idx_400, :]
    sliced_sinogram_400[i, :] = row_400
    
    #slice row 800 for every projection
    row_800 = proj[slice_idx_800, :]
    sliced_sinogram_800[i, :] = row_800

utils.save_array_as_image(sliced_sinogram_50,'sliced_sinogram.png','Img')
utils.save_array_as_image(sliced_sinogram_100,'sliced_sinogram.png','Img')
utils.save_array_as_image(sliced_sinogram_200,'sliced_sinogram.png','Img')
utils.save_array_as_image(sliced_sinogram_400,'sliced_sinogram.png','Img')
utils.save_array_as_image(sliced_sinogram_800,'sliced_sinogram.png','Img')

""" End """

""" Homework 3: Solving CT Problems """
""" Start """

def f(A,b,x):
    return 0.5* np.linalg.norm(A.dot(x) - b)**2

def gradientDescent(function,A,b, x0, iterations):
    x = x0
    history = np.zeros((x0.size, iterations+1))
    history[:, 0] = x0
    
    value0 = function(A,b,x0)
    values = np.zeros(iterations+1)
    values[0] = value0
    
    for i in range(iterations):
        d = A.T.dot(A.dot(x)-b)
        alpha = (d.T).dot(d) / (d.T).dot(A).dot(d)
        x = x - alpha*d
        history[:,i+1] = x
        _, _, values[i] = function(A,b,x)
        
    return history, values

size = (projs_rows,projs_cols)
A = aomip.XrayOperator(size, [721], np.linspace(0, 360, projs_cols), size[0]*100, size[0]*2)
b = sliced_sinogram_200.flatten()
initial_x0 = np.array(np.zeros(projs_rows*projs_cols))
hist, minv = gradientDescent(f,A,b,initial_x0,1000)
iteration = 100
hist, minv = gradientDescent(f,A,b,initial_x0,iteration)
xreconstructed = (hist[iteration]).reshape(size)
utils.save_array_as_image(sliced_sinogram_50,'reconstructed_sinogram_200.png','Img')

""" End """