import numpy as np
import pylab as py
import pyfits


# Updated March 06, 2012 from HST website.
# All zeropoints given in Vega magnitudes
ZP = {'F105W': 25.6236,
      'F110W': 26.0628,
      'F125W': 25.3293,
      'F140W': 25.3761,
      'F160W': 24.6949,
      'F098M': 25.1057,
      'F127M': 23.6799,
      'F139M': 23.4006,
      'F153M': 23.2098,
      'F126N': 21.9396,
      'F128N': 21.9355,
      'F130N': 22.0138,
      'F132N': 21.9499,
      'F164N': 21.5239,
      'F167N': 21.5948,
      'F814W': 25.529}


def curveOfGrowth(psfFile):
    img = pyfits.getdata(psfFile)

    y, x = np.indices(img.shape)

    xcenter = (x.max() - x.min()) / 2.0
    ycenter = (y.max() - y.min()) / 2.0

    rpix = np.hypot(x - xcenter, y - ycenter)

    py.clf()
    py.plot(rpix.flatten(), img.flatten(), 'k.')

    maxRadius = np.min(img.shape)

    r = np.arange(maxRadius)

    flux = np.zeros(len(r), dtype=float)
    Npix = np.zeros(len(r), dtype=int)

    for ii in range(len(r)):
        idx = np.where(rpix < r[ii])

        Npix[ii] = len(idx[0])
        flux[ii] = img[idx].sum()

    return r, flux, Npix
    
      
