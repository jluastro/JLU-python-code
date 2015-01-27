import pylab as py
import numpy as np
from astropy.modeling.functional_models import AiryDisk2D
from astropy.modeling.functional_models import Gaussian2D
from astropy.modeling.models import PowerLaw1D


def make_nirc2_psf():
    scale = 0.01 # arcsec/pixel
    resolution = 0.05 # arcsec
    resolution /= scale

    x, y = np.mgrid[0:100, 0:100]
    psf = AiryDisk2D.eval(x, y, 1, 50, 50, resolution)

    return psf

def make_seeing_psf():
    scale = 0.2 # arcsec/pixel
    resolution = 0.65 # arcsec
    resolution /= scale

    x, y = np.mgrid[0:100, 0:100]
    psf = Gaussian2D.eval(x, y, 1, 50, 50, resolution)
    
    return psf

def random_stars():
    seed = 23498739
    nstars = 1e6

    img_size = 8400
    mrange = [9, 24]
    lum_function = PowerLaw1D(1.0, mrange[0], -0.5)

    np.random.seed(seed)
    x = np.random.rand(nstars) * img_size
    y = np.random.rand(nstars) * img_size
    m_ran = (np.random.rand(nstars) * np.diff(mrange)) + mrange[0]
    

    return x, y
