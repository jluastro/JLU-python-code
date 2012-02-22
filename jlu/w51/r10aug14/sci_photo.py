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

workDir = '/u/jlu/work/w51/photo_calib/10aug14/science_fields/'

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


def aperture_correction(image, filter, plot=True, silent=False, dir=workDir):
    """
    Calculate the aperture corrections to be applied to the 
    science data. Hard coded for H and Kp to be wide camera pixels
    and Lp to be narrow camera pixels. Also hard coded is the optimal
    aperture radius to extract photometry on the science images.
    """
    pixScale = 1.

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
    zeropoints     = {'h': 25.703, 'kp': 25.547, 'k': 24.943}
    zeropointsErr  = {'h': 0.084,  'kp': 0.048,  'k': 0.046}
    extinctions    = {'h': -0.208, 'kp': -0.588, 'k': -0.136}
    extinctionsErr = {'h': 0.071,  'kp': 0.041,  'k': 0.036}

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

            
