import numpy as np
import pylab as plt
import math
import os
import pdb
from astropy import table
from astropy.table import Table, vstack
from astropy.modeling import models, fitting
import glob
from astropy.io import fits
import scipy.ndimage
import photutils
from photutils import psf
from photutils import morphology as morph
from photutils import DAOStarFinder
from photutils import CircularAperture
from photutils import CircularAnnulus
from photutils import aperture_photometry
from astropy.modeling.models import custom_model


directory = '/g/lu/data/MAOS/'
filenames = glob.glob(directory + '*.fits')


@custom_model
def Elliptical_Moffat2D(x, y, \
                        N_sky = 0., amplitude = 1., phi=0., power = 1.,\
                        x_0 = 0., y_0 = 0., width_x = 1., width_y = 1.):
    """
    A custom astropy model for a two dimensional elliptical moffat function.  
    N_sky: a constant background value
    Amplitude: A
    phi: rotation angle (in radians?)
    power: slope of model (beta)
    x_0, y_0: star's centroid coordinates
    width_x, width_y: core widths (alpha)
    """

    c = np.cos(phi)
    s = np.sin(phi)
    A = (c / width_x) ** 2 + (s / width_y)**2
    B = (s / width_x) ** 2 + (c/ width_y)**2
    C = 2 * s * c * (1/width_x**2 - 1/width_y**2)
    denom = (1 + A * (x-x_0)**2 + B * (y-y_0)**2 + C*(x-x_0)*(y-y_0))**power

    return N_sky + amplitude / denom


def psf_stats(img_files = filenames, output='psf_stats.fits'):
    '''
    Calculate statistics for simulated PSFs.
    '''

    N_files = len(img_files)

    # Load up the first file to get the number of
    # wavelength samples. We assume they are all the same.
    hdu_list = fits.open(img_files[0])
    N_wavelengths = len(hdu_list)

    # The total number of entries we expect in our final table.
    N_tot = N_files * N_wavelengths
    
    # Initialize all column arrays
    files = []
    wavelength = np.zeros(N_tot, dtype=float)
    scale = np.zeros(N_tot, dtype=float)
    samp = np.zeros(N_tot, dtype=float)
    ee50_rad = np.zeros(N_tot, dtype=float)
    NEA = np.zeros(N_tot, dtype=float)
    emp_fwhm = np.zeros(N_tot, dtype=float)
    mof_fwhm_maj = np.zeros(N_tot, dtype=float)
    mof_fwhm_min = np.zeros(N_tot, dtype=float)
    mof_phi = np.zeros(N_tot, dtype=float)
    mof_beta = np.zeros(N_tot, dtype=float)
    x_pos = np.zeros(N_tot, dtype=float)
    y_pos = np.zeros(N_tot, dtype=float)
    ao_on = np.zeros(N_tot, dtype=int)
    
    # Iterate through files. We will add them to one big table.
    ii = 0
    
    for nn in range(N_files):
        # Load up the image to work on.
        hdu_list = fits.open(img_files[nn])

        # SET the filename.
        filename = img_files[nn][len(directory):]
        print("Working on image: ", filename)
        
        # Iterate through images within file
        for jj in range(len(hdu_list)):
            files.append(filename)
            img = hdu_list[jj].data
            hdr = hdu_list[jj].header

            # Normalize the PSF.
            img /= img.sum()

            # SET the AO status
            if ('psfc' in filename):
                ao_on[ii] = 1

            # SET the wavelength
            wavelength[ii] = hdr['WAVEL'] * 1e9   # units? 
            scale[ii] = hdr['PIXSCALE']           # arcsec/pix^2
            
            # SET position of the PSF in the FOV
            x = hdr['COMMENT'][0].split()[3]
            x = x.replace('(', '')
            x = int(x.replace(',', ''))
            y = hdr['COMMENT'][0].split()[4]
            y = y.replace(')', '')
            y = int(y.replace(',', ''))

            x_pos[ii] = x
            y_pos[ii] = y
            
            # SET Sampling
            samp[ii] = float(hdr['COMMENT'][5].split()[2])
            
            # Edge length of this PSF, coords of center.
            size = img.shape[0] 
            coords = np.array(img.shape) / 2

            # Calculate the curve-of-growth... i.e. EE for a bunch of different radii.
            y, x = np.mgrid[:size, :size]
            r = np.hypot(x - coords[0], y - coords[1]).flatten()
            flux = img.flatten()

            ridx = r.argsort()
            r = r[ridx]
            flux = flux[ridx]

            enc_energy = np.cumsum(flux)

            # SET the EE50 radius.
            ee50_rad[ii] = r[ np.where(enc_energy >= 0.5)[0][0] ]

            # SET and calculate the NEA
            NEA[ii] = 1.0 / np.sum(img**2)

            # Calculate the empircal FWHM
            # Find the pixels where the flux is a above half max value.
            max_flux = np.amax(img)
            half_max = max_flux / 2.0
            idx = np.where(img >= half_max)[0]

            # Find the equivalent circle diameter for the area of pixels.
            area_HM = len(idx)   # area in pix**2
            emp_fwhm[ii] = 2.0 * (area_HM / np.pi)**0.5

            # Fit with Moffat function
            y, x = np.mgrid[:size, :size]
            z = img
            m_init = Elliptical_Moffat2D(x_0 = coords[0], y_0 = coords[1], amplitude=np.amax(z),
                                         width_x = emp_fwhm[ii], width_y = emp_fwhm[ii])
            fit_m = fitting.LevMarLSQFitter()
            m = fit_m(m_init, x, y, z)
            
            if np.abs(m.width_x.value) < np.abs(m.width_y.value):
                alpha_min = np.abs(m.width_x.value)
                alpha_maj = np.abs(m.width_y.value)
                phi = np.degrees(m.phi.value % (2.0 * np.pi))
            else:
                alpha_maj = np.abs(m.width_y.value)
                alpha_min = np.abs(m.width_x.value)
                phi = np.degrees((m.phi.value % (2.0 * np.pi)) + (np.pi/2))
    
            # SET the Moffat fit parameters
            mof_beta[ii] = m.power.value
            mof_phi[ii] = phi
            mof_fwhm_min[ii] = 2 * alpha_min * np.sqrt((2**(1/m.power.value))-1)
            mof_fwhm_maj[ii] = 2 * alpha_maj * np.sqrt((2**(1/m.power.value))-1)

            # Finished! Iterate our index for the next entry.
            ii += 1


    # Do a little house-keeping to convert to arcsec.
    NEA *= scale**2
    ee50_rad *= scale
    emp_fwhm *= scale
    mof_fwhm_maj *= scale
    mof_fwhm_min *= scale

    stats = table.Table([files, x_pos, y_pos, wavelength, ee50_rad, NEA, emp_fwhm,
                             mof_fwhm_min, mof_fwhm_maj, mof_beta, mof_phi,
                             scale, samp, ao_on],
                        names = ('Image', 'x_pos', 'y_pos', 'wavelength', 'ee50_rad', 'nea', 'emp_fwhm',
                                 'mof_fwhm_min', 'mof_fwhm_maj', 'mof_beta', 'mof_phi', 'scale', 'samp', 'ao_on'),
                        meta={'name': 'Stats Table'})
    stats['x_pos'].unit = 'arcsec'
    stats['x_pos'].format = '4.0f'
    stats['y_pos'].unit = 'arcsec'
    stats['y_pos'].format = '4.0f'
    stats['wavelength'].unit = 'nm'
    stats['wavelength'].format = '7.2f'
    stats['ee50_rad'].unit = 'arcsec'
    stats['ee50_rad'].format = '5.3f'
    stats['nea'].unit = 'arcsec^2'
    stats['nea'].format = '7.3f'
    stats['emp_fwhm'].unit = 'arcsec'
    stats['emp_fwhm'].format = '5.3f'
    stats['mof_fwhm_min'].unit = 'arcsec'
    stats['mof_fwhm_min'].format = '7.3f'
    stats['mof_fwhm_maj'].unit = 'arcsec'
    stats['mof_fwhm_maj'].format = '7.3f'
    stats['mof_beta'].unit = ''
    stats['mof_beta'].format = '7.3f'
    stats['mof_phi'].unit = 'deg'
    stats['mof_phi'].format = '7.3f'
    stats['scale'].unit = 'arcsec pix^-1'
    stats['samp'].unit = ''
            
    stats.write(output, overwrite=True)
                        
    return

def plot_psf_profiles(xpos, ypos):
    """
    Plot a 1D azimuthally averaged PSF at the designated X and Y position
    in the field. 

    xpos : int
        X Position in arceconds. This needs to match the filename.
    ypos : int
        Y Position in arceconds. This needs to match the filename.

    """
    pos_str = '_x{0:d}_y{1:d}'.format(xpos, ypos)
    print(pos_str)

    glao_idx = [i for i, elem in enumerate(filenames) if pos_str in elem][0]
    noao_idx = [i for i, elem in enumerate(filenames) if 'psfo' in elem][0]

    # glao_idx = filenames.index(pos_str)
    # noao_idx = filenames.index('psfo')

    glao_hdu = fits.open(filenames[glao_idx])
    noao_hdu = fits.open(filenames[noao_idx])

    # Define the radial bins we will use 
    
    # Iterate through the different wavelengths.
    color_idx = np.linspace(0, 1, len(glao_hdu))
    
    for jj in range(len(glao_hdu)):
        g_img = glao_hdu[jj].data
        g_hdr = glao_hdu[jj].header

        n_img = noao_hdu[jj].data
        n_hdr = noao_hdu[jj].header

        n_scale = n_hdr['PIXSCALE']           # arcsec/pix^2
        g_scale = g_hdr['PIXSCALE']           # arcsec/pix^2
        
        # Lets double check that the images are
        # the same size. 
        if (g_img.shape != n_img.shape):
            print('Problem with mis-matched image sizes! ')
            print('jj = ', jj)
            print('n_img.shape = ', n_img.shape)
            print('g_img.shape = ', g_img.shape)

        center = np.round(np.array(n_img.shape) / 2.0)
            
        # Make radial mean of the GLAO and NO-AO PSF.
        y, x = np.indices(n_img.shape)
        r = np.hypot(x - center[0], y - center[1])
        r = r.astype(np.int)

        n_tbin = np.bincount(r.ravel(), n_img.ravel())
        g_tbin = np.bincount(r.ravel(), g_img.ravel())
        num_r = np.bincount(r.ravel())
        n_radialprofile = n_tbin / num_r
        g_radialprofile = g_tbin / num_r
        n_rad = r * n_scale
        g_rad = r * g_scale

        plt.semilogy(n_rad, n_radialprofile, 'k--', color=plt.cm.jet(color_idx[jj]))
        plt.semilogy(g_rad, g_radialprofile, 'k-', color=plt.cm.jet(color_idx[jj]))

        
    return
