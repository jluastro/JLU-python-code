import pylab as py
import numpy as np

def dither_fov(offset_x, offset_y):
    """
    Plot up the GSAOI field of view at the specififed offset locations.
    """

    fov = {}
    fov['chip1'] = {'x': np.array([ 0.00,  0.00, 40.96, 40.96,  0.00]),
                    'y': np.array([ 0.00, 40.96, 40.96,  0.00,  0.00])}
    fov['chip2'] = {'x': np.array([ 0.00,  0.00, 40.96, 40.96,  0.00]),
                    'y': np.array([43.66, 84.62, 84.62, 43.66, 43.66])}
    fov['chip3'] = {'x': np.array([43.66, 43.66, 84.62, 84.62, 43.66]),
                    'y': np.array([ 0.00, 40.96, 40.96,  0.00,  0.00])}
    fov['chip4'] = {'x': np.array([43.66, 43.66, 84.62, 84.62, 43.66]),
                    'y': np.array([43.66, 84.62, 84.62, 43.66, 43.66])}


    color_array = ['red', 'orange', 'yellow', 'green', 'blue', 'purple', 
                   'cyan', 'magenta', 'pink', 'gray', 'black']
    
    py.clf()

    fmt = 'Pos {0:2d}, Chip {1:s}: dx = {2:6.2f}  dy = {3:6.2f}'
    for ii in range(len(offset_x)):
        for chip in fov:
            print fmt.format(ii, chip, fov[chip]['x'][0] + offset_x[ii], 
                             fov[chip]['y'][0] + offset_y[ii]), offset_x[ii], offset_y[ii]
            py.plot(fov[chip]['x'] + offset_x[ii], fov[chip]['y'] + offset_y[ii], 
                    '-', color=color_array[ii])

    axis_rng = [-5, 90]
    xlo = axis_rng[0] + offset_x.min()
    xhi = axis_rng[1] + offset_x.max()
    ylo = axis_rng[0] + offset_y.min()
    yhi = axis_rng[1] + offset_y.max()
    print xlo, xhi, ylo, yhi
    py.axis('equal')
    py.xlim(xlo, xhi)
    py.ylim(ylo, yhi)

    

