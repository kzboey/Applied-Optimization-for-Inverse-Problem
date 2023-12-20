import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from os.path import join

def save_array_as_image(array, filename, directory):
    # Create the directory if it doesn't exist
    if not os.path.exists(directory):
        os.makedirs(directory)

    # Normalize the array to 0-255
    array = ((array - np.min(array)) / (np.max(array) - np.min(array))) * 255
    array = array.astype(np.uint8)

    # Save the array as an image
    img = Image.fromarray(array)
    img.save(os.path.join(directory, filename))
    
def plot_convergence(phantom, history, iteration, desc, file_path):
    itr = np.arange(iteration)
    convergence_vect = np.zeros(iteration)
    
    for i in range(iteration):
        conv = np.linalg.norm(phantom - history[:,i])**2
        convergence_vect[i] = conv
    
    plt.clf()
    plt.plot(itr, convergence_vect, marker='.')
    plt.xlabel('iterations')
    plt.ylabel('Reconstruction error')
    plt.title('Convergence analysis of '+desc)
    plt.grid(True)
    bbox = dict(boxstyle ="round", fc ="0.8")
    arrowprops = dict(
    arrowstyle = "->",
    connectionstyle = "angle, angleA = 0, angleB = 90,\
    rad = 10")
    offset = 72

    lastX, lastY = iteration-1, convergence_vect[-1]
    plt.annotate('last iteration: = %.1f'%(lastY), xy=(lastX, lastY), xytext =(-2 * offset, offset),
            textcoords ='offset points',
            bbox = bbox, arrowprops = arrowprops)

    file_name = 'plot_'+desc+'.png'
    save_path = join(file_path,file_name)
    plt.savefig(save_path)
    plt.show()
    
def filter_sinogram(sinogram, filter_type='shepp-logan'):
    H = np.linspace(-1, 1, sinogram.shape[0])
    
    filtered = None
    
    if filter_type == 'ram-lak':
        filtered = np.abs(H)
    elif filter_type == 'shepp-logan':
        filtered = np.abs(H) * np.sinc(H / 2)
    elif filter_type == 'cosine':
        filtered = np.abs(H) * np.cos(H * np.pi / 2)

    h = np.tile(filtered, (sinogram.shape[1], 1)).T
    fftsino = np.fft.fft(sinogram, axis=0)
    projection = np.fft.fftshift(fftsino, axes=1) * np.fft.fftshift(h, axes=0)
    fsino = np.real(np.fft.ifft(np.fft.ifftshift(projection, axes=1), axis=0))
    
    return fsino