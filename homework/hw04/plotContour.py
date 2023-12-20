import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

# define a least squares function f: R^2 -> R
def myFunction(x):
    A = np.array([[2, 1], [1, 1]])
    b = np.array([1, 1])
    residual = A.dot(x) - b
    value =  0.5* np.dot(x, (A.dot(x))) - np.dot(x, b)
    
    return A, b, value
    
# create surface values
def createSurface(function, arange, steps):
    xGrid = np.linspace(arange[0], arange[1], steps)
    yGrid = np.linspace(arange[0], arange[1], steps)
    xx, yy = np.meshgrid(xGrid, yGrid)
    grid = np.array([xx.flatten(), yy.flatten()])
    
    surface = []
    for i in range(grid.shape[1]):
        A, b, value = function(grid[:, i])
        surface.append(value)
        
    surface = (np.asarray(surface)).reshape(steps, steps)
    return xx, yy, surface

# create a contour plot
def plotFunctionContour(function, arange, steps, axes):
    xx, yy, surface = createSurface(function, arange, steps)
    axes.contour(xx, yy, surface)
    axes.set_xlim(arange[0], arange[1]); axes.set_ylim(arange[0], arange[1])