import pylab as py
import numpy as np
import atpy

wd1_root = '/Volumes/data1/dhuang/code/'
wd1_data = wd1_root + 'wd10.001.fits'
comp = [wd1_root + 'CompleVSF814W.fits',
        wd1_root + 'CompleVSF125W.fits',
        wd1_root + 'CompleVSF139M.fits',
        wd1_root + 'CompleVSF160W.fits']


def plot_pm_radial():
    """
    Plot a historgram of stars moving in a radial direction (in the
    cluster outskirts and compare to the average histogram in all other
    directions.
    """
    d = atpy.Table(wd1_data)

    idx = np.where(d.P > 0.9)

    # Identify the clusters central position
    # h, xedges, yedges = py.histogram2d(d.x_F814W, d.y_F814W, weights=d.P)
    h, xedges, yedges = py.histogram2d(d.x_F814W[idx], d.y_F814W[idx])
    extent = [yedges[0], yedges[-1], xedges[0], xedges[-1]]
    py.imshow(h, extent=extent, interpolation=None)
    py.colorbar()


def plot_radial_structure():
    """
    Plot the radial structure of Wd 1 down to XX Msun.
    """
    
