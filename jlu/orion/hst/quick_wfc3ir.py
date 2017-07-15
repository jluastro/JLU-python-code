import numpy as np
import pylab as py
from astropy.io import fits
from astropy.wcs import WCS
from photutils import DAOStarFinder
from astropy.modeling import models, fitting
from astropy.stats import gaussian_fwhm_to_sigma

def run_find_stars():
    img_dir = '/u/jlu/data/orion/hst/'

    img_file = img_dir + 'icol25ekq_flt.fits'

    find_stars_wfc3ir(img_file)

    return

def find_stars_wfc3ir(img_file, threshold=4, N_passes=2):
    print("Working on image: ", img_file)
    img = fits.getdata(img_file)

    # Calculate the bacgkround and noise (iteratively)
    print("\t Calculating background")
    bkg_threshold = 3
    for nn in range(5):
        if nn == 0:
            bkg_mean = img.mean()
            bkg_std = img.std()
        else:
            bkg_mean = img[good_pix].mean()
            bkg_std = img[good_pix].std()

        bad_hi = bkg_mean + (bkg_threshold * bkg_std)
        bad_lo = 0.0

        good_pix = np.where((img < bad_hi) & (img > bad_lo))

    bkg_mean = img[good_pix].mean()
    bkg_std = img[good_pix].std()
    img_threshold = threshold * bkg_std
    print('\t Bkg = {0:.2f} +/- {1:.2f}'.format(bkg_mean, bkg_std))
    print('\t Bkg Threshold = {0:.2f}'.format(img_threshold))

    # Detect stars
    print('\t Detecting Stars')
    radius_init_guess = 1.4

    # Each pass will have an updated fwhm for the PSF.
    for nn in range(N_passes):
        print('\t Pass {0:d} assuming FWHM = {1:.1f}'.format(nn, fwhm))
        daofind = DAOStarFinder(fwhm=fwhm, threshold = img_threshold, exclude_border=True)
        sources = daofind(img - bkg_mean)

        # Calculate FWHM for each detected star.
        x_fwhm = np.zeros(len(sources), dtype=float)
        y_fwhm = np.zeros(len(sources), dtype=float)
        theta = np.zeros(len(sources), dtype=float)

        cutout_half_size = int(round(fwhm * 2))
        cutout_size = 2 * cutout_half_size

        cutouts = np.zeros((len(sources), cutout_size, cutout_size), dtype=float)
        g2d_model = models.AiryDisk2D(1.0, cutout_half_size, cutout_half_size,
                                          radius_init_guess)
        g2d_fitter = fitting.LevMarLSQFitter()
        cut_y, cut_x = np.mgrid[:cutout_size, :cutout_size]

        for ss in range(len(sources)):
            x_lo = int(round(sources[ss]['xcentroid'] - cutout_half_size))
            x_hi = x_lo + cutout_size
            y_lo = int(round(sources[ss]['ycentroid'] - cutout_half_size))
            y_hi = y_lo + cutout_size

            cutout_tmp = img[y_lo:y_hi, x_lo:x_hi].astype(float)
            if ((cutout_tmp.shape[0] != cutout_size) | (cutout_tmp.shape[1] != cutout_size)):
                # Edge source... fitting is no good
                continue

            cutouts[ss] = cutout_tmp
            cutouts[ss] /= cutouts[ss].sum()


            # Fit an elliptical gaussian to the cutout image.
            g2d_params = g2d_fitter(g2d_model, cut_x, cut_y, cutouts[ss])

            x_fwhm[ss] = g2d_params.x_stddev.value / gaussian_fwhm_to_sigma
            y_fwhm[ss] = g2d_params.y_stddev.value / gaussian_fwhm_to_sigma
            theta[ss] = g2d_params.theta.value

        sources['x_fwhm'] = x_fwhm
        sources['y_fwhm'] = y_fwhm
        sources['theta'] = theta

        # Drop sources with flux (signifiance) that isn't good enough.
        # Empirically this is <1.2
        good = np.where(sources['flux'] > 1.2)[0]
        sources = sources[good]

        x_fwhm_med = np.median(sources['x_fwhm'])
        y_fwhm_med = np.median(sources['y_fwhm'])

        print('\t    Number of sources = ', len(sources))
        print('\t    Median x_fwhm = {0:.1f} +/- {1:.1f}'.format(x_fwhm_med,
                                                                 sources['x_fwhm'].std()))
        print('\t    Median y_fwhm = {0:.1f} +/- {1:.1f}'.format(y_fwhm_med,
                                                                 sources['y_fwhm'].std()))

        fwhm = np.mean([x_fwhm_med, y_fwhm_med])


        formats = {'xcentroid': '%8.3f', 'ycentroid': '%8.3f', 'sharpness': '%.2f',
                   'roundness1': '%.2f', 'roundness2': '%.2f', 'peak': '%10.1f',
                   'flux': '%10.6f', 'mag': '%6.2f', 'x_fwhm': '%5.2f', 'y_fwhm': '%5.2f',
                   'theta': '%6.3f'}

        sources.write(img_file.replace('.fits', '_stars.txt'), format='ascii.fixed_width',
                      delimiter=None, bookend=False, formats=formats)
        
    return


def shift_ir_image_wcs():
    ir_img = '/Users/jlu/data/orion/hst/icol25ekq_flt.fits'
    ir_img_new = '/Users/jlu/data/orion/hst/tmp.fits'

    img, hdr = fits.getdata(ir_img, header=True)

    wcs_old = WCS(hdr)
    wcs_new = wcs_old.deepcopy()
    wcs_new.rotateCD(2.0)
    wcs_new.wcs.crpix += np.array([40.0, -150.])

    hdr_new = wcs_new.to_header(relax=True)

    fits.writeto(ir_img_new, img, hdr_new, clobber=True)

    return
