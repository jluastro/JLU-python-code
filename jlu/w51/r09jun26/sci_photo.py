import pyfits
from jlu.nirc2 import photometry as ph
from pyraf import iraf as ir
import numpy as np
import pylab as py
import math, asciidata
import shutil
from scipy import fftpack
from jlu.util import radialProfile
import find_apertures
import pickle
from gcreduce import gcutil
import mtf

workDir = '/u/jlu/work/w51/photo_calib/09jun26/science_fields/'

def curve_of_growth_wide_h():
    """
    Make a plot of the radius of growth for the science exposures
    in h, kp, and lp bands. Do this from the PSF itself.
    """
    dataDir = '/u/jlu/data/w51/09jun26/combo/'

    ##############################
    # H-band wide image
    ##############################
    images = ['mag09jun26_w51a_wide_h', 'm09jun26_w51a_wide_h_1',
              'm09jun26_w51a_wide_h_2', 'm09jun26_w51a_wide_h_3']

    aper = np.arange(2, 24, 0.25)

    radius = np.zeros((len(aper),len(images)), dtype=float)
    flux = np.zeros((len(aper),len(images)), dtype=float)
    mag = np.zeros((len(aper),len(images)), dtype=float)
    merr = np.zeros((len(aper),len(images)), dtype=float)
    diff = np.zeros((len(aper),len(images)), dtype=float)

    for ii in range(len(images)):
        r, f, m, me, d = calc_curve_of_growth(dataDir, images[ii], aper)

        radius[:,ii] = r
        flux[:,ii] = f
        mag[:,ii] = m
        merr[:,ii] = me
        diff[:,ii] = d

        
    ##########
    # Plotting H-band
    ##########
    py.figure(2, figsize=(6,8))
    py.clf()
    py.subplots_adjust(left=0.15, bottom=0.08, top=0.95, right=0.95)
    
    py.subplot(2, 1, 1)
    py.plot(radius[:,0], mag[:,0], 'k-', linewidth=2)
    legendItems = ['Main']
    for ii in range(1, len(images)):
        py.plot(radius[:,ii], mag[:,ii])
        legendItems.append('Sub%d' % ii)
    py.xlabel('Aperture (pix)')
    py.ylabel('Magnitude')
    py.title('Image = %s' % images[0])
    py.xlim(radius[0,0], radius[-1,0])
    py.legend(legendItems)
    
    py.subplot(2, 1, 2)
    py.plot(radius[:,0], diff[:,0], 'k-', linewidth=2)
    for ii in range(1, len(images)):
        py.plot(radius[:,ii], diff[:,ii])
    lims = py.axis()
    py.plot([lims[0], lims[1]], [0, 0], 'b-')
    py.xlabel('Aperture (pix)')
    py.ylabel('Delta-Magnitude')
    py.xlim(radius[0,0], radius[-1,0])
    
    py.savefig(workDir + 'cog_' + images[0] + '.png')


    
def curve_of_growth_wide_kp():
    """
    Make a plot of the radius of growth for the science exposures
    in h, kp, and lp bands. Do this from the PSF itself.
    """
    dataDir = '/u/jlu/data/w51/09jun26/combo/'

    ##############################
    # Kp-band wide image
    ##############################
    images = ['mag09jun26_w51a_wide_kp', 'm09jun26_w51a_wide_kp_1',
              'm09jun26_w51a_wide_kp_2', 'm09jun26_w51a_wide_kp_3']

    aper = np.arange(2, 24, 0.25)

    radius = np.zeros((len(aper),len(images)), dtype=float)
    flux = np.zeros((len(aper),len(images)), dtype=float)
    mag = np.zeros((len(aper),len(images)), dtype=float)
    merr = np.zeros((len(aper),len(images)), dtype=float)
    diff = np.zeros((len(aper),len(images)), dtype=float)

    for ii in range(len(images)):
        r, f, m, me, d = calc_curve_of_growth(dataDir, images[ii], aper)

        radius[:,ii] = r
        flux[:,ii] = f
        mag[:,ii] = m
        merr[:,ii] = me
        diff[:,ii] = d

    ##########
    # Plotting Kp-band
    ##########
    py.figure(2, figsize=(6,8))
    py.clf()
    py.subplots_adjust(left=0.15, bottom=0.08, top=0.95, right=0.95)
    
    py.subplot(2, 1, 1)
    py.plot(radius[:,0], mag[:,0], 'k-', linewidth=2)
    legendItems = ['Main']
    for ii in range(1, len(images)):
        py.plot(radius[:,ii], mag[:,ii])
        legendItems.append('Sub%d' % ii)
    py.xlabel('Aperture (pix)')
    py.ylabel('Magnitude')
    py.title('Image = %s' % images[0])
    py.xlim(radius[0,0], radius[-1,0])
    py.legend(legendItems)
    
    py.subplot(2, 1, 2)
    py.plot(radius[:,0], diff[:,0], 'k-', linewidth=2)
    for ii in range(1, len(images)):
        py.plot(radius[:,ii], diff[:,ii])
    lims = py.axis()
    py.plot([lims[0], lims[1]], [0, 0], 'b-')
    py.xlabel('Aperture (pix)')
    py.ylabel('Delta-Magnitude')
    py.xlim(radius[0,0], radius[-1,0])
    
    py.savefig(workDir + 'cog_' + images[0] + '.png')

def curve_of_growth_lp():
    """
    Make a plot of the radius of growth for the science exposures
    in h, kp, and lp bands. Do this from the PSF itself.
    """
    dataDir = '/u/jlu/data/w51/09jun26/combo/'

    ##############################
    # Kp-band wide image
    ##############################
    allIm = [['mag09jun26_w51a_f1_lp', 'm09jun26_w51a_f1_lp_1',
              'm09jun26_w51a_f1_lp_2', 'm09jun26_w51a_f1_lp_3'],
             ['mag09jun26_w51a_f2_lp', 'm09jun26_w51a_f2_lp_1',
              'm09jun26_w51a_f2_lp_2', 'm09jun26_w51a_f2_lp_3'],
             ['mag09jun26_w51a_f3_lp', 'm09jun26_w51a_f3_lp_1',
              'm09jun26_w51a_f3_lp_2', 'm09jun26_w51a_f3_lp_3'],
             ['mag09jun26_w51a_f4_lp', 'm09jun26_w51a_f4_lp_1',
              'm09jun26_w51a_f4_lp_2', 'm09jun26_w51a_f4_lp_3']
             ]
#              ['mag09jun26_w51a_f2_n2_lp', 'm09jun26_w51a_f2_n2_lp_1',
#               'm09jun26_w51a_f2_n2_lp_2', 'm09jun26_w51a_f2_n2_lp_3']
#              ]

    aper = np.arange(2, 100, 1)

    for images in allIm:
        radius = np.zeros((len(aper), len(images)), dtype=float)
        flux = np.zeros((len(aper), len(images)), dtype=float)
        mag = np.zeros((len(aper), len(images)), dtype=float)
        merr = np.zeros((len(aper), len(images)), dtype=float)
        diff = np.zeros((len(aper), len(images)), dtype=float)
        
        for ii in range(len(images)):
            r, f, m, me, d = calc_curve_of_growth(dataDir, images[ii], aper)
            
            radius[:,ii] = r
            flux[:,ii] = f
            mag[:,ii] = m
            merr[:,ii] = me
            diff[:,ii] = d
            
            
        # #########
        #  Plotting Lp-band
        # #########
        py.figure(2, figsize=(6,8))
        py.clf()
        py.subplots_adjust(left=0.15, bottom=0.08, top=0.95, right=0.95)
        
        py.subplot(2, 1, 1)
        py.plot(radius[:,0], mag[:,0], 'k-', linewidth=2)
        legendItems = ['Main']
        for ii in range(1, len(images)):
            py.plot(radius[:,ii], mag[:,ii])
            legendItems.append('Sub%d' % ii)
        py.xlabel('Aperture (pix)')
        py.ylabel('Magnitude')
        py.title('Image = %s' % images[0])
        py.xlim(radius[0,0], radius[-1,0])
        py.legend(legendItems)
    
        py.subplot(2, 1, 2)
        py.plot(radius[:,0], diff[:,0], 'k-', linewidth=2)
        for ii in range(1, len(images)):
            py.plot(radius[:,ii], diff[:,ii])
        lims = py.axis()
        py.plot([lims[0], lims[1]], [0, 0], 'b-')
        py.xlabel('Aperture (pix)')
        py.ylabel('Delta-Magnitude')
        py.xlim(radius[0,0], radius[-1,0])
    
        py.savefig(workDir + 'cog_' + images[0] + '.png')

def curve_of_growth_h():
    """
    Make a plot of the radius of growth for the science exposures
    in h, kp, and lp bands. Do this from the PSF itself.
    """
    dataDir = '/u/jlu/data/w51/09jun26/combo/'

    ##############################
    # Kp-band wide image
    ##############################
    allIm = [['mag09jun26_w51a_f1_h', 'm09jun26_w51a_f1_h_1',
              'm09jun26_w51a_f1_h_2', 'm09jun26_w51a_f1_h_3'],
             ['mag09jun26_w51a_f2_h', 'm09jun26_w51a_f2_h_1',
              'm09jun26_w51a_f2_h_2', 'm09jun26_w51a_f2_h_3'],
             ['mag09jun26_w51a_f3_h', 'm09jun26_w51a_f3_h_1',
              'm09jun26_w51a_f3_h_2', 'm09jun26_w51a_f3_h_3'],
             ['mag09jun26_w51a_f4_h', 'm09jun26_w51a_f4_h_1',
              'm09jun26_w51a_f4_h_2', 'm09jun26_w51a_f4_h_3']
             ]

    aper = np.arange(2, 100, 1)

    for images in allIm:
        radius = np.zeros((len(aper), len(images)), dtype=float)
        flux = np.zeros((len(aper), len(images)), dtype=float)
        mag = np.zeros((len(aper), len(images)), dtype=float)
        merr = np.zeros((len(aper), len(images)), dtype=float)
        diff = np.zeros((len(aper), len(images)), dtype=float)
        
        for ii in range(len(images)):
            r, f, m, me, d = calc_curve_of_growth(dataDir, images[ii], aper)
            
            radius[:,ii] = r
            flux[:,ii] = f
            mag[:,ii] = m
            merr[:,ii] = me
            diff[:,ii] = d
            
            
        # #########
        #  Plotting Lp-band
        # #########
        py.figure(2, figsize=(6,8))
        py.clf()
        py.subplots_adjust(left=0.15, bottom=0.08, top=0.95, right=0.95)
        
        py.subplot(2, 1, 1)
        py.plot(radius[:,0], mag[:,0], 'k-', linewidth=2)
        for ii in range(1, len(images)):
            py.plot(radius[:,ii], mag[:,ii])
        py.xlabel('Aperture (pix)')
        py.ylabel('Magnitude')
        py.title('Image = %s' % images[0])
        py.xlim(radius[0,0], radius[-1,0])
        
        py.subplot(2, 1, 2)
        py.plot(radius[:,0], diff[:,0], 'k-', linewidth=2)
        for ii in range(1, len(images)):
            py.plot(radius[:,ii], diff[:,ii])
        lims = py.axis()
        py.plot([lims[0], lims[1]], [0, 0], 'b-')
        py.xlabel('Aperture (pix)')
        py.ylabel('Delta-Magnitude')
        py.xlim(radius[0,0], radius[-1,0])
    
        py.savefig(workDir + 'cog_' + images[0] + '.png')

def curve_of_growth_kp():
    """
    Make a plot of the radius of growth for the science exposures
    in h, kp, and lp bands. Do this from the PSF itself.
    """
    dataDir = '/u/jlu/data/w51/09jun10/combo/'

    ##############################
    # Kp-band wide image
    ##############################
    allIm = [['mag09jun10_w51a_f1_kp', 'm09jun10_w51a_f1_kp_1',
              'm09jun10_w51a_f1_kp_2', 'm09jun10_w51a_f1_kp_3'],
             ['mag09jun10_w51a_f2_kp', 'm09jun10_w51a_f2_kp_1',
              'm09jun10_w51a_f2_kp_2', 'm09jun10_w51a_f2_kp_3'],
             ['mag09jun10_w51a_f3_kp', 'm09jun10_w51a_f3_kp_1',
              'm09jun10_w51a_f3_kp_2', 'm09jun10_w51a_f3_kp_3']]

    aper = np.arange(2, 100, 1)

    for images in allIm:
        radius = np.zeros((len(aper), len(images)), dtype=float)
        flux = np.zeros((len(aper), len(images)), dtype=float)
        mag = np.zeros((len(aper), len(images)), dtype=float)
        merr = np.zeros((len(aper), len(images)), dtype=float)
        diff = np.zeros((len(aper), len(images)), dtype=float)
        
        for ii in range(len(images)):
            r, f, m, me, d = calc_curve_of_growth(dataDir, images[ii], aper)
            
            radius[:,ii] = r
            flux[:,ii] = f
            mag[:,ii] = m
            merr[:,ii] = me
            diff[:,ii] = d
            
            
        # #########
        #  Plotting Lp-band
        # #########
        py.figure(2, figsize=(6,8))
        py.clf()
        py.subplots_adjust(left=0.15, bottom=0.08, top=0.95, right=0.95)
        
        py.subplot(2, 1, 1)
        py.plot(radius[:,0], mag[:,0], 'k-', linewidth=2)
        for ii in range(1, len(images)):
            py.plot(radius[:,ii], mag[:,ii])
        py.xlabel('Aperture (pix)')
        py.ylabel('Magnitude')
        py.title('Image = %s' % images[0])
        py.xlim(radius[0,0], radius[-1,0])
        
        py.subplot(2, 1, 2)
        py.plot(radius[:,0], diff[:,0], 'k-', linewidth=2)
        for ii in range(1, len(images)):
            py.plot(radius[:,ii], diff[:,ii])
        lims = py.axis()
        py.plot([lims[0], lims[1]], [0, 0], 'b-')
        py.xlabel('Aperture (pix)')
        py.ylabel('Delta-Magnitude')
        py.xlim(radius[0,0], radius[-1,0])
    
        py.savefig(workDir + 'cog_' + images[0] + '.png')

def curve_of_growth_gc_kp():
    """
    Make a plot of the radius of growth for the science exposures
    in h, kp, and lp bands. Do this from the PSF itself.
    """
    dataDir = '/u/ghezgroup/data/gc_new/09junlgs/combo/'

    ##############################
    # Kp-band wide image
    ##############################
    allIm = [['mag09junlgs_msr_kp', 'm09junlgs_msr_kp_1', 
              'm09junlgs_msr_kp_2', 'm09junlgs_msr_kp_3']]

    aper = np.arange(2, 100, 1)

    for images in allIm:
        for ii in range(len(images)):
            r, f, m, me, d = calc_curve_of_growth(dataDir, images[ii], aper)
            
            


def calc_curve_of_growth(dataDir, image, apertures):
    aper = apertures

    # ==========
    # Make a coordinate file, PSF is in the middle
    # ==========
    # get image size
    data = pyfits.getdata(dataDir + image + '_psf.fits')
    imgSize = data.shape[0]
    
    # Write out coo file
    cooFile = image + '_psf.coo'
    _coo = open(cooFile, 'w')
    _coo.write('%.2f %.2f\n' % (imgSize/2.0, imgSize/2.0))
    _coo.close()
    
    
    # #########
    # Setup phot for running on PSF
    # #########
    ph.setup_phot(dataDir + image + '_psf', apertures=aper)
    ir.fitskypars.salgorithm = 'constant'
    ir.fitskypars.skyvalue = 0.0
    
    # Output into current directory, not data directory
    input = dataDir + image + '_psf.fits'
    output = workDir + image + '_psf.phot.mag'

    ir.phot(input, cooFile, output, verbose="no")
    
    (radius, flux, mag, merr) = ph.get_phot_output(output, silent=True)
    diff = np.zeros(len(radius), dtype=float)
    diff[1:] = mag[1:] - mag[:-1]
    
    # Save output
    pfile = open(workDir + 'cog_' + image + '.dat', 'w')
    pickle.dump(radius, pfile)
    pickle.dump(flux, pfile)
    pickle.dump(mag, pfile)
    pickle.dump(merr, pfile)
    pickle.dump(diff, pfile)
    pfile.close()

    return (radius, flux, mag, merr, diff)

def compare_psfs(image, filter, dir=workDir, pixScale=1.0):
    ##########
    # Load up information on the science fields
    ##########
    pfile = open(dir + 'cog_' + image + '.dat', 'r')
    dRadius = pickle.load(pfile)
    dFlux = pickle.load(pfile)
    dMag = pickle.load(pfile)
    dMerr = pickle.load(pfile)
    dGcurve = pickle.load(pfile)
    
    # convert to narrow camera pixels
    dRadius = dRadius * pixScale
    
    ##########
    # Read in the growth curves from the 
    # photometric standards.
    ##########
    sdir = dir + '../find_apertures/'
    sRadius, growthCurves = find_apertures.getGrowthCurve(filter, dir=sdir)
    sGcurve = growthCurves.mean(axis=0)
    sGcurveErr = growthCurves.std(axis=0)

    # Make photometry vs. radius curve for standards... arbitrarily normalize
    sMag = sGcurve.cumsum()
    idx = np.abs(dRadius - sRadius[0]).argmin()
    sMag += dMag[idx] - sMag[0]

    py.figure(1)
    py.clf()
    py.plot(sRadius, sMag, 'b-', label='Standards')
    py.plot(dRadius, dMag, 'r-', label='Science PSF')
    tmp = np.zeros(len(sMag), dtype=float)
    tmp += sMag[-1]
    py.plot(sRadius, tmp, 'k--', label='_none_')

    rng = py.axis()
    py.ylim(rng[3], rng[2])
    py.legend(('Standards', 'Science PSF'), loc='lower right')
    py.xlabel('Radius (narrow pixels)')
    py.ylabel('Brightness (mag)')
    py.savefig(dir + 'phot_compare_' + image + '.png')

def aperture_correction(image, filter, plot=True, silent=False, dir=workDir,
                        fixPixScale=None):
    """
    Calculate the aperture corrections to be applied to the 
    science data. Hard coded for H and Kp to be wide camera pixels
    and Lp to be narrow camera pixels. Also hard coded is the optimal
    aperture radius to extract photometry on the science images.
    """
    if fixPixScale == None:
        pixScale = 4.

        if (filter == 'lp2'):
            pixScale = 1.
    else:
        pixScale = fixPixScale

    # This is the radius we will use to get the aperture photometry
    # for the science field. Note that the L' science aperture is 
    # larger than the standard aperture... this is because we see
    # a huge difference in the PSF from the standards (NGS) to the 
    # science field (LGS), so to gather the same amount of light, 
    # we need to use different apertures.
    # FYI, future reference, we should use LGS even on the standards.
    extractRadius = {'h': 60, 'kp': 70, 'lp2': 80}
    standardRadius = find_apertures.standard_aperture

    ##########
    # Load up information on the science fields
    ##########
    pfile = open(dir + 'cog_' + image + '.dat', 'r')
    dRadius = pickle.load(pfile)
    dFlux = pickle.load(pfile)
    dMag = pickle.load(pfile)
    dMerr = pickle.load(pfile)
    dGcurve = pickle.load(pfile)
    
    # convert to narrow camera pixels
    dRadius = dRadius * pixScale
    
    ##########
    # Read in the growth curves from the 
    # photometric standards.
    ##########
    sdir = dir + '../find_apertures/'
    sRadius, growthCurves = find_apertures.getGrowthCurve(filter, dir=sdir)
    sGcurve = growthCurves.mean(axis=0)
    sGcurveErr = growthCurves.std(axis=0)

    if plot == True:
        # #########
        #  Overplot them to double check that they are both sensible
        # #########
        py.figure(1)
        py.clf()
        p1 = py.plot(sRadius[1:], sGcurve[1:], 'k-')
        
        gcurveLo = sGcurve[1:] - sGcurveErr[1:]
        gcurveHi = sGcurve[1:] + sGcurveErr[1:]
        py.fill_between(sRadius[1:], gcurveLo, gcurveHi, 
                        color='grey', alpha=0.3)
        # not quite right cuz in mags but close enough.

        p2 = py.plot(dRadius[1:], dGcurve[1:], 'r.')
        
        py.legend((p1, p2), ('Standards', 'Science PSF'), loc='lower right')
        py.xlabel('Radius (narrow pixels)')
        py.ylabel('Magnitude Difference')
        py.savefig(dir + 'cog_compare_' + image + '.png')


        py.xlim(15, np.array([dRadius.max(), sRadius.max()]).min())
        py.ylim(-0.03, 0)
        py.savefig(dir + 'cog_compare_' + image + '_zoom.png')

        # Plot a difference figure
        sidx = np.where(np.setmember1d(sRadius, dRadius) == True)[0]
        didx = np.where(np.setmember1d(dRadius, sRadius) == True)[0]
        tmp = np.zeros(len(sRadius[1:]))

        py.clf()
        p1 = py.plot(sRadius[1:], tmp, 'k-')
        py.fill_between(sRadius[1:], -1.*sGcurveErr[1:], sGcurveErr[1:], 
                        color='grey', alpha=0.3)
        p2 = py.plot(sRadius[sidx], dGcurve[didx] - sGcurve[sidx], 'r.')
        
        py.legend((p1, p2), ('Standards', 'Science PSF'), loc='lower right')
        py.xlabel('Radius (narrow pixels)')
        py.ylabel('Magnitude Difference')
        py.savefig(dir + 'cog_compare_diff_' + image + '.png')

        py.xlim(15, np.array([dRadius.max(), sRadius.max()]).min())
        py.ylim(-0.03, 0.03)
        py.savefig(dir + 'cog_compare_diff_' + image + '_zoom.png')
        

    ##########
    # Calc Aperture Corrections
    ##########
    # Calculate the aperture correction to get the scale factor between
    # the science aperture (PSF size) and the standard star aperture (outer).
    dataApSize = extractRadius[filter]
    stanApSize = standardRadius[filter]

    ### 1. Go from full-size to aperture-size on science PSF
    fluxApIdx = np.where(dRadius == dataApSize)[0][0]

    stf2aper_flux = dFlux[fluxApIdx]
    stf2aper_mags = -2.5*math.log10(dFlux[fluxApIdx])
    stf2aper_flux_err = 0.0
    stf2aper_mags_err = 0.0

    ### 2. Go from aperture-size on science PSF to aperture-size on Standards
    # Integrate the standard star Curve of Growth (from the outside in)
    # Don't do any of this for L' since the PSFs are so different from 
    # LGS to NGS.
#     if filter != 'lp2':
    if True:
        dataApIdx = np.where(sRadius == dataApSize)[0][0]
        stanApIdx = np.where(sRadius == stanApSize)[0][0]

        if dataApIdx > stanApIdx:
            magCorr = sGcurve[stanApIdx:dataApIdx].sum() * -1.0
        else:
            magCorr = sGcurve[stanApIdx:dataApIdx:-1].sum()

        fluxCorr = 10**(magCorr/-2.5)

        # Determine the uncertainty by calculating the aperture correction
        # from the individual growth curves and then using std().
        curveCount = growthCurves.shape[0]
        magCorr1 = np.zeros(curveCount, dtype=float)
        fluxCorr1 = np.zeros(curveCount, dtype=float)
        for aa in range(curveCount):
            magCorr1[aa] = growthCurves[aa, :dataApIdx:-1].sum()
            fluxCorr1[aa] = 10**(magCorr1[aa]/-2.5)

        aper2stan_flux = fluxCorr
        aper2stan_mags = magCorr
        aper2stan_flux_err = fluxCorr1.std()
        aper2stan_mags_err = 2.5 * math.log10(math.e) * aper2stan_flux_err
    else:
        aper2stan_flux = 1.
        aper2stan_mags = 0.
        aper2stan_flux_err = 0.
        aper2stan_mags_err = 0.
    
    ### 3. Combine the two effects
    stf2stan_flux = stf2aper_flux * aper2stan_flux
    stf2stan_mags = stf2aper_mags + aper2stan_mags
    stf2stan_flux_err = stf2aper_flux * aper2stan_flux_err
    stf2stan_mags_err = 2.5 * math.log10(math.e) * stf2stan_flux_err

    #*** NO APERTURE CORRECTION FOR L' ***
#     if filter == 'lp2':
#         stf2stan_flux = 1.
#         stf2stan_mags = 0.
#         stf2stan_flux_err = 0.
#         stf2stan_mags_err = 0.


    ##########
    # Output
    ##########
    if not silent:
        print '*** APERTURE CORRECTIONS FOR %s ***' % image
        print 'Science Aperture Size  = %d narrow pixels (%.3f arcsec)' % \
            (dataApSize, dataApSize * 0.00995)
        print 'Standard Aperture Size = %d narrow pixels (%.3f arcsec)' % \
            (stanApSize, stanApSize * 0.00995)
        
        print ''
        print 'Aperture Correction to go from Starfinder Magnitudes'
        print 'to Aperture Magnitudes:'
        print '    Flux Ratio = %.3f +/- %.3f' % \
            (stf2aper_flux, stf2aper_flux_err)
        print '    Mag Differ = %.3f +/- %.3f' % \
            (stf2aper_mags, stf2aper_mags_err)
        print '    Aper Flux = STF Flux * %.3f' % (stf2aper_flux)
        print '    Aper Mags = STF Mags + %.3f' % (stf2aper_mags)
        print ''
        print 'Aperture Correction to go from Aperture Magnitudes '
        print 'to Standard Apparent Magnitudes:'
        print '    Flux Ratio = %.3f +/- %.3f' % \
            (aper2stan_flux, aper2stan_flux_err)
        print '    Mag Differ = %.3f +/- %.3f' % \
            (aper2stan_mags, aper2stan_mags_err)
        print '    Stan Flux = Aper Flux * %.3f' % (aper2stan_flux)
        print '    Stan Mags = Aper Mags + %.3f + ZP' % (aper2stan_mags)
        print ''
        print 'Aperture Correction to go from Starfinder Magnitudes '
        print 'to Standard Apparent Magnitudes:'
        print '    Flux Ratio = %.3f +/- %.3f' % \
            (stf2stan_flux, stf2stan_flux_err)
        print '    Mag Differ = %.3f +/- %.3f' % \
            (stf2stan_mags, stf2stan_mags_err)
        print '    Stan Flux = STF Flux * %.3f' % (stf2stan_flux)
        print '    Stan Mags = STF Mags + %.3f + ZP' % (stf2stan_mags)
    

    return (stf2stan_flux, stf2stan_flux_err, stf2stan_mags, stf2stan_mags_err)

def makeCalibrators():
    """
    Assumes you are in the starlists/ directory
    """
    from jlu.starfinder import txt2lis
    import os

    dataDir = '/u/jlu/data/w51/09jun26/combo/'
    listDir = workDir + 'starlists/'

    filters = ['h', 'kp', 'lp2']
    imageRoots = {'h': ['09jun26_w51a_wide_h'],
                  'kp': ['09jun26_w51a_wide_kp'],
                  'lp2': ['09jun26_w51a_f4_lp', '09jun26_w51a_f3_lp', 
                          '09jun26_w51a_f2_lp', '09jun26_w51a_f1_lp']}
    alignType = {'h': 12, 'kp': 12, 'lp2': 8}

    for filter in filters:
        for root in imageRoots[filter]:
            lists = ['mag' + root + '_0.8_stf',
                     'm' + root + '_1_0.6_stf',
                     'm' + root + '_2_0.6_stf',
                     'm' + root + '_3_0.6_stf']

            images = ['mag' + root,
                      'm' + root + '_1',
                      'm' + root + '_2',
                      'm' + root + '_3']

            for ii in range(len(images)):
                # ==========
                # Get the airmass
                # ==========
                airmass = pyfits.getval(dataDir+images[ii]+'.fits', 'AIRMASS')

                # ==========
                # Run txt2lis (custom version) on the starfinder *.txt lists
                # in order to get out the raw starfinder fluxes.
                # ==========
                txt2lis.convert_with_fits(listDir + lists[ii], datadir=dataDir)
            
                # ==========
                # Apply the aperture correction and save to a new
                #   *.maglis file.
                # ==========
                photoCalibrate(listDir+lists[ii], images[ii], 
                               filter, airmass, dataDir=dataDir)


            # ==========
            # Run align_rms on the new main + 3 subset images 
            # ==========
            gcutil.mkdir(listDir + 'align')

            # Make align list
            alignRoot = listDir + 'align/align_' + images[0]

            _alignList = open(alignRoot + '.list', 'w')
            for ii in range(len(images)):
                _alignList.write('%s.maglis %d\n' % 
                                 (listDir + lists[ii], alignType[filter]))
            _alignList.close()

            # Run align
            os.system('java -Xmx1024m align -R 3 -v -p -a 0 -r %s %s.list' % 
                      (alignRoot, alignRoot))

            # Run align_rms
            os.system('align_rms -m %s 3 3' % (alignRoot))
            os.rename(alignRoot + '_rms.out', 
                      listDir + images[0] + '_rms.maglis')

            # Run align_rms with relative errors
            os.system('align_rms -m -r %s 3 3' % (alignRoot))
            os.rename(alignRoot + '_rms.out', 
                      listDir + images[0] + '_rms_rel.maglis')

def checkOnGC():
    """
    Assumes you are in the starlists/ directory
    """
    from jlu.starfinder import txt2lis
    import os

    dataDir = '/u/ghezgroup/data/gc_new/09junlgs/combo/'
    listDir = workDir + 'starlists/'

    root = '09junlgs_msr_kp'
    lists = ['mag' + root + '_0.8_stf',
             'm' + root + '_1_0.6_stf',
             'm' + root + '_2_0.6_stf',
             'm' + root + '_3_0.6_stf']

    images = ['mag' + root,
              'm' + root + '_1',
              'm' + root + '_2',
              'm' + root + '_3']

    filter = 'kp'
    for ii in range(len(images)):
        # ==========
        # Get the airmass
        # ==========
        airmass = pyfits.getval(dataDir+images[ii]+'.fits', 'AIRMASS')
        
        # ==========
        # Run txt2lis (custom version) on the starfinder *.txt lists
        # in order to get out the raw starfinder fluxes.
        # ==========
        txt2lis.convert_with_fits(listDir + lists[ii], datadir=dataDir)
            
        # ==========
        # Apply the aperture correction and save to a new
        #   *.maglis file.
        # ==========
        photoCalibrate(listDir+lists[ii], images[ii], 
                       filter, airmass, fixPixScale=1, dataDir=dataDir)

        # Make a ds9 region file for overplotting
        list = asciidata.open(listDir + lists[ii] + '.maglis')
        mag = list[1].tonumpy()
        x = list[3].tonumpy()
        y = list[4].tonumpy()

        reg = open(listDir + lists[ii] + '.reg', 'w')
        x += 1
        y += 1

        reg.write('# Region file format: DS9 version 3.0\n')
        reg.write('global color=green font="helvetica 10 normal" ')
        reg.write('edit=1 move=1 delete=1 include=1 fixed=0 ')
        reg.write('width=2\n')

        for s in range(len(x)):
            if mag[s] < 14.5:
                reg.write('image;circle(%.1f,%.1f,5) # text={%4.1f}\n' %
                          (x[s], y[s], mag[s]))

        reg.close()


def photoCalibrate(listRoot, imageRoot, filter, airmass, 
                   apCorrDir=workDir, dataDir='/u/jlu/data/w51/09jun26/combo/',
                   fixPixScale=None, useMTF=None, outDir='./',
                   innerAper=None, outerAper=None):
    """
    Apply aperture corrections and zeropoints to a starlist and save
    the results out to a new *.maglis file.

    useMTF -- specify a directory where the MTF analysis has been run
              on the input image in order to compute the aperture corrections.
    """
    zeropoints     = {'h': 25.409, 'kp': 24.771, 'lp2': 23.236}
    zeropointsErr  = {'h': 0.025,  'kp': 0.018,  'lp2': 0.074}
    extinctions    = {'h': -0.039, 'kp': -0.069, 'lp2': -0.074}
    extinctionsErr = {'h': 0.017,  'kp': 0.013,  'lp2': 0.045}

    # ==========
    # 7. Calibrate the rms.lis files
    # ==========
    tab = asciidata.open(listRoot + '.lis')
    s_name = tab[0].tonumpy()
    s_mag = tab[1].tonumpy()
    s_date = tab[2].tonumpy()
    s_xpos = tab[3].tonumpy()
    s_ypos = tab[4].tonumpy()
    s_snr = tab[5].tonumpy()
    s_corr = tab[6].tonumpy()
    s_nframes = tab[7].tonumpy()
    s_flux = tab[8].tonumpy()

    # ==========
    # 3. Read in the aperture correction information
    # ==========
    if useMTF != None:
        results = mtf.apertureCorrections(imageRoot, plot=False, silent=True, 
                                          stfDir=dataDir, mtfDir=useMTF,
                                          outDir=outDir, innerAper=innerAper,
                                          outerAper=outerAper)
    else:
        results = aperture_correction(imageRoot, filter, dir=apCorrDir,
                                      plot=False, silent=True, 
                                      fixPixScale=fixPixScale)
    fluxApCorr = results[0]
    fluxApCorrErr = results[1]
    magApCorr = results[2]
    magApCorrErr = results[3]

    # ==========
    # 4. calculate apparent magnitudes
    # ==========
    fitsFile = dataDir + imageRoot + '.fits'
    itime = float(pyfits.getval(fitsFile, 'ITIME'))
    coadds = float(pyfits.getval(fitsFile, 'COADDS'))
    tint = itime * coadds

    print 'm    = -2.5*log( f_STF / tint ) + AC + ZP + EX * X'
    print 'tint = %7.3f' % tint
    print 'AC   = %7.3f +/- %6.3f' % (magApCorr, magApCorrErr)
    print 'ZP   = %7.3f +/- %6.3f' % (zeropoints[filter], zeropointsErr[filter])
    print 'EX   = %7.3f +/- %6.3f' % (extinctions[filter], extinctionsErr[filter])
    print ' X   = %7.3f' % airmass

    mag = -2.5 * np.log10(s_flux/tint) 
    mag += magApCorr
    mag += zeropoints[filter]
    mag += extinctions[filter] * airmass
            
    staticError = magApCorrErr**2
    staticError += zeropointsErr[filter]**2
    staticError += (extinctionsErr[filter] * airmass)**2
    staticError = math.sqrt(staticError)

    magErrFromSNR = 1.0857 / s_snr
    magErr = magErrFromSNR**2 + staticError**2
    magErr = np.sqrt(magErr)

    snr = 1.0857 / magErr

    # ==========
    # 5. Write output to a file
    # ==========
    print 'Calibrated Starlist %s has %5.2f mag of static error (ZP, EX, AC)' %\
        (listRoot + '.maglis', staticError)

    outRoot = listRoot.split('/')[-1]
    _out = open(outDir + outRoot + '.maglis', 'w')
                
    # Loop through stars
    for ss in range(len(s_name)):
        _out.write('%-10s  ' % (s_name[ss]))
        _out.write('%6.3f  ' % (mag[ss]))
        _out.write('%8.3f %8.3f %8.3f   ' % 
                   (s_date[ss], s_xpos[ss], s_ypos[ss]))
        _out.write('%11.2f  %7.2f  %6d  %11.2f\n' %
                   (snr[ss], s_corr[ss], s_nframes[ss], s_snr[ss]))
    _out.close()

            
def plotCalibrators():
    """
    Run in starlists/ directory
    """
    roots = ['mag09jun26_w51a_wide_h',
             'mag09jun26_w51a_wide_kp',
             'mag09jun26_w51a_f1_lp',
             'mag09jun26_w51a_f2_lp',
             'mag09jun26_w51a_f3_lp',
             'mag09jun26_w51a_f4_lp']

    for root in roots:
        # First pull out the static error contribution from the first 
        # (bright) star in the original starlist.
        list = asciidata.open(workDir + 'starlists/'+root + '_0.8_stf.maglis')
        mag = list[1].tonumpy()
        snr = list[5].tonumpy()
        snrOrig = list[8].tonumpy()

        magErr = 2.5 * math.log10(math.e) / snr
        magErrOrig = 2.5 * math.log10(math.e) / snrOrig
        
        staticErr = magErr[0]

        py.clf()
        py.semilogy(mag, magErrOrig, 'c.')
        py.semilogy(mag, magErr, 'b.')


        # Now read in the *rms.maglis files
        list = asciidata.open(workDir + 'starlists/'+root + '_rms.maglis')
        mag = list[1].tonumpy()
        x = list[3].tonumpy()
        y = list[4].tonumpy()
        snr = list[7].tonumpy()
        r = np.hypot(x - (x.max()/2.0), y - (y.max()/2.0))

        listRel = asciidata.open(workDir+'starlists/'+root+'_rms_rel.maglis')
        magRel = listRel[1].tonumpy()
        xRel = listRel[3].tonumpy()
        yRel = listRel[4].tonumpy()
        snrRel = listRel[7].tonumpy()

        magErr = 2.5 * math.log10(math.e) / snr
        magErrRel = 2.5 * math.log10(math.e) / snrRel

        py.semilogy(mag, magErr, 'r.')
        py.semilogy(mag, magErrRel, 'm.')

        py.title('%s: static = %5.2f mag' % (root, staticErr))
        py.xlabel('Magnitude')
        py.ylabel('Magnitude Errors')
        py.legend(('STF', 'STF+static', 'with ZP', 'Relative'), 
                  numpoints=1, loc='lower right')
        py.savefig('%sstarlists/plots/phot_err_%s.png' % 
                   (workDir, root))


        py.clf()
        py.semilogy(r, magErr, 'k.')
        py.semilogy(r, magErrRel, 'r.')
        py.title('%s: static = %5.2f mag' % (root, staticErr))
        py.xlabel('Radius (pixels)')
        py.ylabel('Magnitude Errors')
        py.legend(('with ZP', 'Relative'), numpoints=1)
        py.savefig('%sstarlists/plots/phot_err_radius_%s_rms.png' % 
                   (workDir, root))
        
        # Make a ds9 region file for overplotting
        reg = open(workDir + 'starlists/' + root + '_rms_rel.reg', 'w')
        x += 1
        y += 1

        reg.write('# Region file format: DS9 version 3.0\n')
        reg.write('global color=green font="helvetica 10 normal" ')
        reg.write('edit=1 move=1 delete=1 include=1 fixed=0 ')
        reg.write('width=2\n')

        for s in range(len(x)):
            if magRel[s] < 14.5:
                reg.write('image;circle(%.1f,%.1f,5) # text={%4.1f}\n' %
                          (x[s], y[s], magRel[s]))

        reg.close()
        


def makePhotoDat_h():
    """
    Read in the calibrated starlists from this night of observing
    and make a w51a_photo.dat file with the best calibrators.
    """
    # ==========
    # H-band
    # ==========
    list = asciidata.open(workDir + 
                          'starlists/mag09jun26_w51a_wide_h_rms_rel.maglis')
    
    mag = list[1].tonumpy()
    x = list[3].tonumpy()
    y = list[4].tonumpy()
    snr = list[7].tonumpy()
    
    magErr = 2.5 * math.log10(math.e) / snr

    # The brightest star in the field is f1_psf1 and our coordinate reference.
    idx = mag.argmin()
    xref = x[idx]
    yref = y[idx]

    xarc = (x - xref) * 0.04 * -1.0
    yarc = (y - yref) * 0.04
    rarc = np.hypot(xarc, yarc)

    # Read in the old photometric calibration file and try
    # to get names of the sources using a crude coordinate matching.
    names = []
    oldCalib = asciidata.open('/u/jlu/data/w51/source_list/w51a_photo.dat')
    oldNames = oldCalib[0].tonumpy()
    oldX = oldCalib[1].tonumpy()
    oldY = oldCalib[2].tonumpy()

    for ii in range(len(xarc)):
        diff = np.hypot(xarc[ii] - oldX, yarc[ii] - oldY)
        minIdx = diff.argmin()

        if diff[minIdx] < 0.2:
            names.append(oldNames[minIdx])
        else:
            names.append('')
    names = np.array(names)

    # Output file
    _out = open('results_photo_calib_h.dat', 'w')

#     # Loop through stars and reject close pairs
#     keepStar = np.ones(len(rarc), dtype=bool)
#     pairCutRad = 0.2
#     pairCutMag = 2

#     for ss in range(len(rarc)):
#         diffx = xarc - xarc[ss]
#         diffy = yarc - yarc[ss]
#         diffr = np.hypot(diffx, diffy)
#         diffm = abs(mag - mag[ss])

#         idx = np.where((diffr != 0) & 
#                        (diffr < pairCutRad) & 
#                        (diffm < pairCutMag))[0]

#         keepStar[idx] = False
    
#     mag = mag[keepStar == True]
#     x = x[keepStar == True]
#     y = y[keepStar == True]
#     snr = snr[keepStar == True]
#     magErr = magErr[keepStar == True]
#     xarc = xarc[keepStar == True]
#     yarc = yarc[keepStar == True]
#     rarc = rarc[keepStar == True]
#     names = names[keepStar == True]

    # Figure out which stars we should use as calibrators.
    idx1 = np.where(mag < 16.5)[0]  # Get brightest stars
    magErrFloor = np.median(magErr[idx1]) # Find median

    magErrCut = 1.1
    magCut = 17
    rarcCut = 12

    # Print out the brightest stars as they might be coo/psf stars
    # and need to be in w51a_photo.dat even if they aren't calibrators.
    idx = np.where((mag < magCut) & (rarc < rarcCut) &
                   (magErr >= (magErrCut * magErrFloor)))[0]

    print '*** H-band Bright Sources *** (not necessarily calibrators)'
    for ii in idx:
        print '%-13s  %8.3f %8.3f  %5.2f +/- %5.2f' % \
            (names[ii], xarc[ii], yarc[ii], mag[ii], magErr[ii])

    _out.write('*** H-band Bright Sources *** (not necessarily calibrators)\n')
    for ii in idx:
        _out.write('%-13s  %8.3f %8.3f  %5.2f +/- %5.2f\n' % 
                   (names[ii], xarc[ii], yarc[ii], mag[ii], magErr[ii]))

    
    idx2 = np.where((magErr < (magErrCut * magErrFloor)) &
                    (mag < magCut) & 
                    (rarc < rarcCut))[0]

    mag = mag[idx2]
    x = x[idx2]
    y = y[idx2]
    snr = snr[idx2]
    magErr = magErr[idx2]
    xarc = xarc[idx2]
    yarc = yarc[idx2]
    rarc = rarc[idx2]
    names = names[idx2]


    print ''
    print '*** H-band Photometric Calibrators ***'
#     print 'Pairwise Cut: dr < %.1f and dm < %.1f' % (pairCutRad, pairCutMag)
    print 'Magnitude Error Cut: %.1f * %.2f = %.2f' % \
        (magErrCut, magErrFloor, magErrCut*magErrFloor)
    print 'Magnitude Cut: %.2f' % (magCut)
    print 'Radius Cut: %.1f' % (rarcCut)
    print 'Number of calibrators: %d' % len(mag)
    print 'Magnitude Range: %.2f - %.2f' % (mag.min(), mag.max())

    _out.write('\n')
    _out.write('*** H-band Photometric Calibrators ***\n')
#     _out.write('Pairwise Cut: dr < %.1f and dm < %.1f\n' % (pairCutRad, pairCutMag))
    _out.write('Magnitude Error Cut: %.1f * %.2f = %.2f\n' %
               (magErrCut, magErrFloor, magErrCut*magErrFloor))
    _out.write('Magnitude Cut: %.2f\n' % (magCut))
    _out.write('Radius Cut: %.1f\n' % (rarcCut))
    _out.write('Number of calibrators: %d\n' % len(mag))
    _out.write('Magnitude Range: %.2f - %.2f\n' % (mag.min(), mag.max()))
    
    # Over plot the calibrators on an image
    dataDir = '/u/jlu/data/w51/09jun26/combo/'
    img = pyfits.getdata(dataDir + 'mag09jun26_w51a_wide_h.fits')

    xaxis = (np.arange(img.shape[1]) - xref) * 0.04 * -1.0
    yaxis = (np.arange(img.shape[0]) - yref) * 0.04

    py.figure(2, figsize=(10,10))
    py.clf()
    py.imshow(np.sqrt(img), cmap=py.cm.gist_heat,
              extent=[xaxis.max(), xaxis.min(), yaxis.min(), yaxis.max()],
              interpolation=None, vmin=math.sqrt(0), vmax=math.sqrt(1000))

    py.plot(xarc, yarc, 'go', mfc='none', mec='green', ms=5, mew=2)
    py.xlabel('R.A. Offset from f1_psf1 (arcsec)')
    py.ylabel('Dec. Offset from f1_psf1 (arcsec)')
    py.title('H-band Photometric Calibrators')

    for ii in range(len(xarc)):
        py.text(xarc[ii], yarc[ii], '%.2f' % mag[ii], 
                fontsize=6, color='green')

    py.axis([rarcCut, -rarcCut, -rarcCut, rarcCut])
    py.savefig(workDir + 'starlists/plots/photo_calib_map_h.png')

    # Print out calibrators
    for ii in range(len(xarc)):
        print '%-13s  %8.3f %8.3f  %5.2f +/- %5.2f' % \
            (names[ii], xarc[ii], yarc[ii], mag[ii], magErr[ii])
        _out.write('%-13s  %8.3f %8.3f  %5.2f +/- %5.2f\n' % 
                   (names[ii], xarc[ii], yarc[ii], mag[ii], magErr[ii]))
        

def makePhotoDat_kp():
    """
    Read in the calibrated starlists from this night of observing
    and make a w51a_photo.dat file with the best calibrators.
    """
    # ==========
    # H-band
    # ==========
    list = asciidata.open(workDir + 
                          'starlists/mag09jun26_w51a_wide_kp_rms_rel.maglis')
    
    mag = list[1].tonumpy()
    x = list[3].tonumpy()
    y = list[4].tonumpy()
    snr = list[7].tonumpy()
    
    magErr = 2.5 * math.log10(math.e) / snr

    # The brightest star in the field is f1_psf1 and our coordinate reference.
    idx = mag.argmin()
    xref = x[idx]
    yref = y[idx]

    xarc = (x - xref) * 0.04 * -1.0
    yarc = (y - yref) * 0.04
    rarc = np.hypot(xarc, yarc)

    # Read in the old photometric calibration file and try
    # to get names of the sources using a crude coordinate matching.
    names = []
    oldCalib = asciidata.open('/u/jlu/data/w51/source_list/w51a_photo.dat')
    oldNames = oldCalib[0].tonumpy()
    oldX = oldCalib[1].tonumpy()
    oldY = oldCalib[2].tonumpy()

    for ii in range(len(xarc)):
        diff = np.hypot(xarc[ii] - oldX, yarc[ii] - oldY)
        minIdx = diff.argmin()

        if diff[minIdx] < 0.2:
            names.append(oldNames[minIdx])
        else:
            names.append('')
    names = np.array(names)

    # Output file
    _out = open('results_photo_calib_kp.dat', 'w')


    # Loop through stars and reject close pairs
    keepStar = np.ones(len(rarc), dtype=bool)
    pairCutRad = 0.2
    pairCutMag = 2

    for ss in range(len(rarc)):
        diffx = xarc - xarc[ss]
        diffy = yarc - yarc[ss]
        diffr = np.hypot(diffx, diffy)
        diffm = abs(mag - mag[ss])

        idx = np.where((diffr != 0) & 
                       (diffr < pairCutRad) & 
                       (diffm < pairCutMag))[0]

        keepStar[idx] = False
    
    mag = mag[keepStar == True]
    x = x[keepStar == True]
    y = y[keepStar == True]
    snr = snr[keepStar == True]
    magErr = magErr[keepStar == True]
    xarc = xarc[keepStar == True]
    yarc = yarc[keepStar == True]
    rarc = rarc[keepStar == True]
    names = names[keepStar == True]

    # Figure out which stars we should use as calibrators.
    idx1 = np.where(mag < 17)[0]  # Get brightest stars
    magErrFloor = np.median(magErr[idx1]) # Find median

    magErrCut = 0.9
    magCut = 16.3
    rarcCut = 12

    # Print out the brightest stars as they might be coo/psf stars
    # and need to be in w51a_photo.dat even if they aren't calibrators.
    idx = np.where((mag < magCut) & (rarc < rarcCut) &
                   (magErr >= (magErrCut * magErrFloor)))[0]

    print '*** Kp-band Bright Sources *** (not necessarily calibrators)'
    for ii in idx:
        print '%8.3f %8.3f  %5.2f +/- %5.2f' % \
            (xarc[ii], yarc[ii], mag[ii], magErr[ii])

    _out.write('*** Kp-band Bright Sources *** (not necessarily calibrators)\n')
    for ii in idx:
        _out.write('%-13s  %8.3f %8.3f  %5.2f +/- %5.2f\n' % 
                   (names[ii], xarc[ii], yarc[ii], mag[ii], magErr[ii]))

    idx2 = np.where((magErr < (magErrCut * magErrFloor)) &
                    (mag < magCut) & 
                    (rarc < rarcCut))[0]

    mag = mag[idx2]
    x = x[idx2]
    y = y[idx2]
    snr = snr[idx2]
    magErr = magErr[idx2]
    xarc = xarc[idx2]
    yarc = yarc[idx2]
    rarc = rarc[idx2]
    names = names[idx2]

    print ''
    print '*** Kp-band Photometric Calibrators ***'
    print 'Pairwise Cut: dr < %.1f and dm < %.1f' % (pairCutRad, pairCutMag)
    print 'Magnitude Error Cut: %.1f * %.2f = %.2f' % \
        (magErrCut, magErrFloor, magErrCut*magErrFloor)
    print 'Magnitude Cut: %.2f' % (magCut)
    print 'Radius Cut: %.1f' % (rarcCut)
    print 'Number of calibrators: %d' % len(mag)
    print 'Magnitude Range: %.2f - %.2f' % (mag.min(), mag.max())
    
    _out.write('\n')
    _out.write('*** Kp-band Photometric Calibrators ***\n')
    _out.write('Pairwise Cut: dr < %.1f and dm < %.1f\n' % (pairCutRad, pairCutMag))
    _out.write('Magnitude Error Cut: %.1f * %.2f = %.2f\n' %
               (magErrCut, magErrFloor, magErrCut*magErrFloor))
    _out.write('Magnitude Cut: %.2f\n' % (magCut))
    _out.write('Radius Cut: %.1f\n' % (rarcCut))
    _out.write('Number of calibrators: %d\n' % len(mag))
    _out.write('Magnitude Range: %.2f - %.2f\n' % (mag.min(), mag.max()))

    # Over plot the calibrators on an image
    dataDir = '/u/jlu/data/w51/09jun26/combo/'
    img = pyfits.getdata(dataDir + 'mag09jun26_w51a_wide_kp.fits')

    xaxis = (np.arange(img.shape[1]) - xref) * 0.04 * -1.0
    yaxis = (np.arange(img.shape[0]) - yref) * 0.04

    py.figure(2, figsize=(10,10))
    py.clf()
    py.imshow(np.sqrt(img), cmap=py.cm.gist_heat,
              extent=[xaxis.max(), xaxis.min(), yaxis.min(), yaxis.max()],
              interpolation=None, vmin=math.sqrt(0), vmax=math.sqrt(1000))

    py.plot(xarc, yarc, 'go', mfc='none', mec='green', ms=5, mew=2)
    py.xlabel('R.A. Offset from f1_psf1 (arcsec)')
    py.ylabel('Dec. Offset from f1_psf1 (arcsec)')
    py.title('Kp-band Photometric Calibrators')

    for ii in range(len(xarc)):
        py.text(xarc[ii], yarc[ii], '%.2f' % mag[ii], 
                fontsize=6, color='green')

    py.axis([rarcCut, -rarcCut, -rarcCut, rarcCut])
    py.savefig(workDir + 'starlists/plots/photo_calib_map_kp.png')

    # Print out calibrators
    for ii in range(len(xarc)):
        print '%8.3f %8.3f  %5.2f +/- %5.2f' % \
            (xarc[ii], yarc[ii], mag[ii], magErr[ii])

    # Print out calibrators
    for ii in range(len(xarc)):
        print '%-13s  %8.3f %8.3f  %5.2f +/- %5.2f' % \
            (names[ii], xarc[ii], yarc[ii], mag[ii], magErr[ii])
        _out.write('%-13s  %8.3f %8.3f  %5.2f +/- %5.2f\n' % 
                   (names[ii], xarc[ii], yarc[ii], mag[ii], magErr[ii]))

def makePhotoDat_lp():
    """
    Read in the calibrated starlists from this night of observing
    and make a w51a_photo.dat file with the best calibrators.
    """
    # ==========
    # H-band
    # ==========
    lists = [workDir + 'starlists/mag09jun26_w51a_f1_lp_rms_rel.maglis',
             workDir + 'starlists/mag09jun26_w51a_f2_lp_rms_rel.maglis',
             workDir + 'starlists/mag09jun26_w51a_f3_lp_rms_rel.maglis',
             workDir + 'starlists/mag09jun26_w51a_f4_lp_rms_rel.maglis']

    # Coo Star Coordinates relative to f1_psf1
    # Pull coo stars from /u/jlu/data/w51/09jun26/reduce/analysis.py
    #                    f1_psf1 f2_psf0 f3_psf2 f4_psf1
    cooStarX = np.array([ 0.000,  8.004, -6.399, -9.215])
    cooStarY = np.array([ 0.000,  2.748,  4.607,  0.448])

    fields = ['f1', 'f2', 'f3', 'f4']

    # Output file
    _out = open('results_photo_calib_lp.dat', 'w')

    
    for ii in range(len(lists)):
        list = asciidata.open(lists[ii])
        
        mag = list[1].tonumpy()
        x = list[3].tonumpy()
        y = list[4].tonumpy()
        snr = list[7].tonumpy()
        
        magErr = 2.5 * math.log10(math.e) / snr
        
        xref = x[0]
        yref = y[0]
        xarc = ((x - xref) * 0.01 * -1.0) + cooStarX[ii]
        yarc = ((y - yref) * 0.01) + cooStarY[ii]
        rarc = np.hypot(xarc, yarc)

        # Read in the old photometric calibration file and try
        # to get names of the sources using a crude coordinate matching.
        names = []
        oldCalib = asciidata.open('/u/jlu/data/w51/source_list/w51a_photo.dat')
        oldNames = oldCalib[0].tonumpy()
        oldX = oldCalib[1].tonumpy()
        oldY = oldCalib[2].tonumpy()

        for jj in range(len(xarc)):
            diff = np.hypot(xarc[jj] - oldX, yarc[jj] - oldY)
            minIdx = diff.argmin()
            
            if diff[minIdx] < 0.2:
                names.append(oldNames[minIdx])
            else:
                names.append('')
        names = np.array(names)


#         # Loop through stars and reject close pairs
#         keepStar = np.ones(len(rarc), dtype=bool)
#         pairCutRad = 0.2
#         pairCutMag = 2

#         for ss in range(len(rarc)):
#             diffx = xarc - xarc[ss]
#             diffy = yarc - yarc[ss]
#             diffr = np.hypot(diffx, diffy)
#             diffm = abs(mag - mag[ss])
            
#             idx = np.where((diffr != 0) & 
#                            (diffr < pairCutRad) & 
#                            (diffm < pairCutMag))[0]

#             keepStar[idx] = False
    
#         mag = mag[keepStar == True]
#         x = x[keepStar == True]
#         y = y[keepStar == True]
#         snr = snr[keepStar == True]
#         magErr = magErr[keepStar == True]
#         xarc = xarc[keepStar == True]
#         yarc = yarc[keepStar == True]
#         rarc = rarc[keepStar == True]
#         names = names[keepStar == True]
        
        # Figure out which stars we should use as calibrators.
        idx1 = np.where(mag < 14.5)[0]  # Get brightest stars
        magErrFloor = np.median(magErr[idx1]) # Find median
        
        magErrCut = 1.3
        magCut = 14.5
        rarcCut = 12

        # Print out the brightest stars as they might be coo/psf stars
        # and need to be in w51a_photo.dat even if they aren't calibrators.
        idx = np.where((mag < 20) & (rarc < rarcCut) &
                       (magErr >= (magErrCut * magErrFloor)))[0]

        print ''
        print '*** Lp-band Bright Sources *** (not necessarily calibrators)'
        for jj in idx:
            print '%8.3f %8.3f  %5.2f +/- %5.2f' % \
                (xarc[jj], yarc[jj], mag[jj], magErr[jj])

        _out.write('*** Lp-band Bright Sources *** (not necessarily calibrators)\n')
        for jj in idx:
            _out.write('%-13s  %8.3f %8.3f  %5.2f +/- %5.2f\n' % 
                       (names[ii], xarc[ii], yarc[ii], mag[ii], magErr[ii]))

        idx2 = np.where((magErr < (magErrCut * magErrFloor)) &
                        (mag < magCut) & 
                        (rarc < rarcCut))[0]

        mag = mag[idx2]
        x = x[idx2]
        y = y[idx2]
        snr = snr[idx2]
        magErr = magErr[idx2]
        xarc = xarc[idx2]
        yarc = yarc[idx2]
        rarc = rarc[idx2]
        names = names[idx2]

        print ''
        print '*** Lp-band Photometric Calibrators for %s ***' % (fields[ii])
#         print 'Pairwise Cut: dr < %.1f and dm < %.1f' % (pairCutRad, pairCutMag)
        print 'Magnitude Error Cut: %.1f * %.2f = %.2f' % \
            (magErrCut, magErrFloor, magErrCut*magErrFloor)
        print 'Magnitude Cut: %.2f' % (magCut)
        print 'Radius Cut: %.1f' % (rarcCut)
        print 'Number of calibrators: %d' % len(mag)
        print 'Magnitude Range: %.2f - %.2f' % (mag.min(), mag.max())

        _out.write('\n')
        _out.write('*** Lp-band Photometric Calibrators for %s***\n' % 
                   (fields[ii]))
#         _out.write('Pairwise Cut: dr < %.1f and dm < %.1f\n' % 
#                    (pairCutRad, pairCutMag))
        _out.write('Magnitude Error Cut: %.1f * %.2f = %.2f\n' %
                   (magErrCut, magErrFloor, magErrCut*magErrFloor))
        _out.write('Magnitude Cut: %.2f\n' % (magCut))
        _out.write('Radius Cut: %.1f\n' % (rarcCut))
        _out.write('Number of calibrators: %d\n' % len(mag))
        _out.write('Magnitude Range: %.2f - %.2f\n' % (mag.min(), mag.max()))
        
        # Over plot the calibrators on an image
        dataDir = '/u/jlu/data/w51/09jun26/combo/'
        img = pyfits.getdata(dataDir + 'mag09jun26_w51a_'+fields[ii]+'_lp.fits')
        
        xaxis = ((np.arange(img.shape[1]) - xref) * 0.01 * -1.0) + cooStarX[ii]
        yaxis = ((np.arange(img.shape[0]) - yref) * 0.01) + cooStarY[ii]
        
        py.figure(2, figsize=(10,10))
        py.clf()
        py.imshow(np.sqrt(img), cmap=py.cm.gist_heat,
                  extent=[xaxis.max(), xaxis.min(), yaxis.min(), yaxis.max()],
                  interpolation=None, vmin=math.sqrt(800), vmax=math.sqrt(1500))
        
        py.plot(xarc, yarc, 'go', mfc='none', mec='green', ms=10, mew=1)
        py.xlabel('R.A. Offset from f1_psf1 (arcsec)')
        py.ylabel('Dec. Offset from f1_psf1 (arcsec)')
        py.title('Lp-band Photometric Calibrators for ' + fields[ii])
        
        for jj in range(len(xarc)):
            py.text(xarc[jj], yarc[jj], '%.2f' % mag[jj], 
                    fontsize=6, color='yellow')
            
        py.savefig(workDir + 
                   'starlists/plots/photo_calib_map_lp_' + fields[ii] + '.png')
        
        # Print out calibrators
        for jj in range(len(xarc)):
            print '%8.3f %8.3f  %5.2f +/- %5.2f' % \
                (xarc[jj], yarc[jj], mag[jj], magErr[jj])

        # Print out calibrators
        for ii in range(len(xarc)):
            print '%-13s  %8.3f %8.3f  %5.2f +/- %5.2f' % \
                (names[ii], xarc[ii], yarc[ii], mag[ii], magErr[ii])
            _out.write('%-13s  %8.3f %8.3f  %5.2f +/- %5.2f\n' % 
                       (names[ii], xarc[ii], yarc[ii], mag[ii], magErr[ii]))


