import numpy as np

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

# create a 3d plot for the function
def plotFunction3d(function, arange, steps, axes):
    xx, yy, surface = createSurface(function, arange, steps)
    axes.plot_surface(xx, yy, surface, cmap=cm.coolwarm)
    
# create a contour plot
def plotFunctionContour(function, arange, steps, axes):
    xx, yy, surface = createSurface(function, arange, steps)
    axes.contour(xx, yy, surface)
    axes.set_xlim(arange[0], arange[1]); axes.set_ylim(arange[0], arange[1])