import numpy as np
import math
import os
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
    
    # radial bins for the EE curves

    num_wavelengths = 7
    
    # Initialize all column arrays
    samps = np.zeros(N_files*num_wavelengths, dtype=float)
    int_time = np.zeros(N_files*num_wavelengths, dtype=float)
    file = []
    wl = np.zeros(N_files*num_wavelengths, dtype=float)
    ee50 = np.zeros(N_files*num_wavelengths, dtype=float)
    NEA = np.zeros(N_files*num_wavelengths, dtype=float)
    emp_fwhm = np.zeros(N_files*num_wavelengths, dtype=float)
    mof_fwhm_maj = np.zeros(N_files*num_wavelengths, dtype=float)
    mof_fwhm_min = np.zeros(N_files*num_wavelengths, dtype=float)
    x_position = np.zeros(N_files*num_wavelengths, dtype=float)
    y_position = np.zeros(N_files*num_wavelengths, dtype=float)
    phi = np.zeros(N_files*num_wavelengths, dtype=float)
    beta = np.zeros(N_files*num_wavelengths, dtype=float)
    scale = np.zeros(N_files*num_wavelengths, dtype=float)
    
    # Iterate through files
    for ii in range(N_files):
        # Load up the image to work on.
        filename = img_files[ii]
        filename = filename[len(directory):]
        print("Working on image: ", filename)
        hdu_list = fits.open(img_files[ii])
        
        
        # Iterate through images within file
        for jj in range(len(hdu_list)):
            img = hdu_list[jj].data
            hdr = hdu_list[jj].header
    #         img, hdr = fits.getdata(img_files[ii], header=True)
            size = np.shape(img)[0] #edge length of a given frame
            
            max_radius = size/2
            radii = np.arange(0.05, max_radius, 0.05)
            
            coords = np.array([img.shape[0]/2, img.shape[1]/2])

            # Define the background annuli (typically from 2"-3"). Calculate mean background for each star.
            bkg_annuli = CircularAnnulus(coords, max_radius, (max_radius + 1))
            bkg = aperture_photometry(img, bkg_annuli)
            bkg_mean = bkg['aperture_sum'] / bkg_annuli.area()
            enc_energy = np.zeros((1, len(radii)), dtype=float)
            int_psf2_all = np.zeros(1, dtype=float)

            # Loop through radial bins and calculate EE
            for rr in range(len(radii)):
                radius_pixel = radii[rr]
                apertures = CircularAperture(coords, r=radius_pixel)
                phot_table = aperture_photometry(img, apertures)
                energy = phot_table['aperture_sum']
                bkg_sum = apertures.area() * bkg_mean
                enc_energy[:, rr] = energy - bkg_sum

                # Calculate the sum((PSF - bkg))^2 -- have to do this for each star.
                # Only do this on the last radius measurement.
                if rr == (len(radii)):
    #                 for ss in range(N_stars):
                    aperture = CircularAperture(coords, r=radius_pixel)
                    phot2_table = aperture_photometry((img - bkg_mean)**2, aperture)
                    int_psf2 = phot2_table['aperture_sum'][0]
                    int_psf2 /= enc_energy[rr]**2 # normalize
                    int_psf2_all = int_psf2

            # Normalize all the curves so that the mean of the last 5 bins = 1
            enc_energy /= np.tile(enc_energy[:, -5:].mean(axis=1), (len(radii), 1)).T
            enc_energy_final = np.median(enc_energy, axis=0)
            ee50_rad = radii[ np.where(enc_energy_final >= 0.5)[0][0]]


            # Calculate NEA
            nea = 1.0 / (np.diff(enc_energy_final)**2 / (2.0 * math.pi * radii[1:] * np.diff(radii))).sum()


            # Calculate FWHM
            x_cent = int(round(float(coords[0])))
            y_cent = int(round(float(coords[1])))
            one_star = img[int(y_cent-size*.45) : int(y_cent+size*.45), int(x_cent-size*.45) : int(x_cent+size*.45)]  # Odd box, with center in middle pixel.
            over_samp_5 = scipy.ndimage.zoom(one_star, 5, order = 1)

            # Find the pixels where the flux is a above half max value.
            max_flux = np.amax(over_samp_5)
            half_max = max_flux / 2.0
            idx = np.where(over_samp_5 >= half_max)

            # Find the equivalent circle diameter for the area of pixels.
            area_count = len(idx[0]) / 5**2   # area in pix**2 -- note we went back to raw pixels (not oversampled)
            emp_FWHM = 2.0 * (area_count / np.pi)**0.5
            med_emp_FWHM = np.median(emp_FWHM)


            # Fit with Moffat function
            y, x = np.mgrid[:size, :size]
            z = img
            m_init = Elliptical_Moffat2D(x_0 = coords[0], y_0 = coords[1], amplitude=np.amax(z),
                                         width_x = emp_FWHM, width_y = emp_FWHM)
            fit_m = fitting.LevMarLSQFitter()
            m = fit_m(m_init, x, y, z)
            
#             N_sky[(ii + jj)]     = m.N_sky.value
#             amplitude[(ii + jj)] = m.amplitude.value
#             power[(ii + jj)]     = m.power.value
#             x_0[(ii + jj)]       = m.x_0.value
#             y_0[(ii + jj)]       = m.y_0.value
#             if m.width_x.value < m.width_y.value:
#                 width_x[(ii + jj)]    = m.width_x.value
#                 width_y[(ii + jj)]    = m.width_y.value
#                 phi[(ii + jj)]       = m.phi.value
#             else:
#                 width_x[(ii + jj)]    = m.width_y.value
#                 width_y[(ii + jj)]    = m.width_x.value
#                 phi[(ii + jj)]       = m.phi.value + (np.pi/2)
            
            if m.width_x.value < m.width_y.value:
                alpha_min = m.width_x.value
                alpha_maj = m.width_y.value
                phi[(ii * num_wavelengths + jj)]       = m.phi.value
            else:
                alpha_maj = m.width_y.value
                alpha_min = m.width_x.value
                phi[(ii * num_wavelengths + jj)] = m.phi.value + (np.pi/2)
    
#             alpha_min = m.width_x.value
#             alpha_max = m.width_y.value
            
            # Fit Moffat, Calculate Moffat-FWHM
#             alpha = m.emp_fwhm.value
            beta[(ii * num_wavelengths + jj)] = m.power.value
            FWHM_min = 2 * alpha_min * np.sqrt((2**(1/m.power.value))-1)
            FWHM_maj = 2 * alpha_maj * np.sqrt((2**(1/m.power.value))-1)
            
            # Get position
            x = hdr['COMMENT'][0].split()[3]
            x = x.replace('(', '')
            x = int(x.replace(',', ''))
            y = hdr['COMMENT'][0].split()[4]
            y = y.replace(')', '')
            y = int(y.replace(',', ''))
            
            # Sampling
            samp = float(hdr['COMMENT'][5].split()[2])
            
            # Compile data table
            scale[(ii * num_wavelengths + jj)] = hdr['PIXSCALE']
            print(filename)
            file.append(filename)
            wl[(ii * num_wavelengths + jj)] = hdr['WAVEL'] * 1e9
            ee50[(ii * num_wavelengths + jj)] = ee50_rad * scale[(ii * num_wavelengths + jj)]
            NEA[(ii * num_wavelengths + jj)] = nea * scale[(ii * num_wavelengths + jj)] * scale[(ii * num_wavelengths + jj)]
            emp_fwhm[(ii * num_wavelengths + jj)] = med_emp_FWHM * scale[(ii * num_wavelengths + jj)]
            mof_fwhm_maj[(ii * num_wavelengths + jj)] = np.abs(FWHM_maj * scale[(ii * num_wavelengths + jj)])
            mof_fwhm_min[(ii * num_wavelengths + jj)] = np.abs(FWHM_min * scale[(ii * num_wavelengths + jj)])
            x_position[(ii * num_wavelengths + jj)] = x
            y_position[(ii * num_wavelengths + jj)] = y
#             x_position[(ii * num_wavelengths + jj)] = m.x_0.value * scale
#             y_position[(ii * num_wavelengths + jj)] = m.y_0.value * scale

    stats = table.Table([file, x_position, y_position, wl, ee50, NEA, emp_fwhm, mof_fwhm_min, mof_fwhm_maj, beta, phi, scale],
                        names = ('Image', 'x_position[arcsec]', 'y_position[arcsec]', 'wavelength[nm]',
                                 'EE50[arcsec]', 'NEA[arcsec^2]', 'emp_fwhm[arcsec]', 'mof_fwhm_min[arcsec]',
                                 'mof_fwhm_maj[arcsec]', 'beta', 'phi[rad]', 'pixscale'),
                        meta={'name':'Stats Table'})

    stats['EE50[arcsec]'].format = '7.3f'
    stats['NEA[arcsec^2]'].format = '7.3f'
    stats['emp_fwhm[arcsec]'].format = '7.3f'
    stats['mof_fwhm_min[arcsec]'].format = '7.3f'
    stats['mof_fwhm_maj[arcsec]'].format = '7.3f'
    
    stats.write(output, overwrite=True)
                        
    return