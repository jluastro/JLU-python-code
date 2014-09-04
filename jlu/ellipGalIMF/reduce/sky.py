import numpy as np
import pylab as py
import pyfits
from scipy.ndimage import filters
from scipy.optimize import fmin as simplex
from scipy.interpolate import interp1d
import pdb

# Constants that describe the various OH spectral bands.
l_boundary = np.array([0.9980,1.0670,1.1250,1.1960,1.2520,
                       1.2890,1.4000,1.4720,1.5543,1.6356,
                       1.7253,1.8400,1.9570,2.0950,2.3000, 
                       2.4000])

# Rotational Transitions
description_strings = np.array(['4-1', '5-2', '6-3', '7-4', 'O2', 
                                '8-5', '2-0', '3-1', '4-2', '5-3', 
                                '6-4', '7-5', '8-6', '9-7', 'final bit', 
                                'end'])
l_rotlow = np.array([1.00852, 1.03757, 1.09264, 1.15388, 1.22293,
                     1.30216, 1.45190, 1.52410, 1.60308, 1.69037,
                     1.78803, 2.02758, 2.18023, 1.02895, 1.08343, 
                     1.14399, 1.21226, 1.29057, 1.43444, 1.50555, 
                     1.58333, 1.66924, 1.76532, 2.00082, 2.15073])
l_rotmed = np.array([1.00282, 1.02139, 1.04212, 1.07539, 1.09753, 
                     1.13542, 1.15917, 1.20309, 1.22870, 1.28070,
                     1.30853, 1.41861, 1.46048, 1.48877, 1.53324,
                     1.56550, 1.61286, 1.65024, 1.70088, 1.74500, 
                     1.79940, 1.97719, 2.04127, 2.12496, 2.19956])

def scale_sky(sky_spec, sci_spec):
    """
    Fit the 1D sky spectrum to the 1D science spectrum allowing
    different families of OH lines to be scaled differently and
    the continuum to be scaled separately. 

    This module is code hijacked from the OSIRIS data redcution
    pipeline.
    """

    sky, sky_hdr = pyfits.getdata(sky_spec, header=True)
    sci, sci_hdr = pyfits.getdata(sci_spec, header=True)

    # Make a wavelength array for the sky object.
    sky_lambda = np.arange(len(sky), dtype=float)
    sky_lambda -= sky_hdr['CRPIX1'] - 1
    sky_lambda *= sky_hdr['CRDELT1']
    sky_lambda += sky_hdr['CRVAL1']

    # Full width in pixels of unresolved emission line
    line_half_width = 2
    npixw = 2 * line_half_width

    # Median filter the sky to get an estimate of the continuum.
    # Remove this continuum.
    backgnd = filters.median(sky, size=10*line_half_width, mode='nearest')
    sky2 = sky - backgnd
    
    for ii in range(len(l_boundary)-1):
        lr = np.where((sky_lambda >= l_boundary[ii]) & (sky_lambda < l_boundary[ii+1]))[0]

        if len(lr) == 0:
            continue

        sky_lr = sky[lr]
        sky2_lr = sky2[lr]
        sci_lr = sci[lr]
        lam_lr = sky_lamda[lr]
        
        sky_median = np.median(sky2_lr)
        sky_stddev = np.std(sky2)

        w_skylines = np.where(sky2_lr > (10 * sky_median) + sky_stddev)[0]

        if len(w_skylines) > 0:
            # Make a mask for where line regions are and for where the
            # continuum is.
            line_mask = np.zeros(len(lr), dtype=float)
            line_mask[w_skylines] = 10.0

            # Convolve the mask with a "resolution-width" window.
            window = np.ones(npixw)
            line_mask = np.convolve(line_mask, window, mode='same')
            line_regions = line_mask > 0
            line_count = line_regions.sum()

            if (line_count >= 3) and ((len(line_regions) - line_count) >= 3):
                # Remove the continuum from the science and object
                f_sci = interp1d(lam_lr[~line_regions], sci_lr[~line_regions])
                f_obj = inpterp1d(lam_lr[~line_regions], sky2[~line_regions])
                sci_no_cont = sci_lr[line_regions] - f_sci([lam_lr])
                

                out = simplex(fit_sky, init, args=(sci, sky))



    
def fit_sky():
    return



def trim_spectra():
    
    return
    

def plot_skycorr_output(output_dir, output_root, object_root):
    """
    Plot the output from skycorr
    """
    fit = pyfits.getdata(output_dir + output_root + '_fit.fits')
    astar = pyfits.getdata(output_dir + '../../A0V_0034.fits')

    wave = fit['LAMBDA']
    
    py.clf()
    py.plot(wave, fit['flux'], label='sci')
    py.plot(wave, fit['mflux'], label='sky')
    py.plot(wave, fit['scflux'], label='sci_sky-sub')
    py.plot(wave, fit['dev'], label='dev')
    py.legend()

    # Zoom in around the wing-ford band.
    py.xlim(0.9900, 1.0000)
    py.ylim(-100, 500)

    return
