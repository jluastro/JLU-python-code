from scipy import stats
import numpy as np
import pylab as py
import math
import pyfits
import shutil, os, glob
from gcwork import starTables
from gcwork import starset
from gcreduce import gcutil
from jlu.util import statsIter as istats
from jlu.starfinder import starPlant
import pdb

workDir = '/u/jlu/work/gc/ao_performance/ast_err_vs_snr/'

def plot_astrometric_errors(radiusCut=4.0, avgXY=True):
    dir = '/u/jlu/uni/syelda/gc/aligndir/06setup_only/10_05_23/'

    alignRoot = 'align/align_d_rms_1000_abs_t'
    polyRoot = 'polyfit_1000/fit'
    
    s = starset.StarSet(dir + alignRoot)
    s.loadPolyfit(dir + polyRoot)
    x = s.getArray('x')
    y = s.getArray('y')
    r = np.hypot(x, y)

    # Trim stars down to those within 4"
    print 'Before Trim: ', len(s.stars)
    velCount = s.getArray('velCnt')
#     idx = np.where((r > 0.5) & (r < radiusCut))[0]
    idx = np.where((r > 0.5) & (r < radiusCut) & (velCount == velCount.max()))[0]
    s.stars = [s.stars[ii] for ii in idx]
    print 'After Trim: ', len(s.stars)

    numEpochs = len(s.years)
    numStars = len(s.stars)

    xerr_p = s.getArrayFromAllEpochs('xerr_p') * 10**3
    yerr_p = s.getArrayFromAllEpochs('yerr_p') * 10**3
    xerr_a = s.getArrayFromAllEpochs('xerr_a') * 10**3
    yerr_a = s.getArrayFromAllEpochs('yerr_a') * 10**3

    # --- Correct for the additive error term
    # Not enough decimal points in align, so we are going
    # to assume that if xerr_p is <= addErr, then the actual
    # error on the mean centroid position is 0.041 mas, which
    # is the largest it could have been without showing up in 
    # our printed out precision.
    addErr = 0.17 # mas
    fixErr = 0.04 # mas

    x_idx = np.where((xerr_p > 0) & (xerr_p <= addErr))
    y_idx = np.where((yerr_p > 0) & (yerr_p <= addErr))

    nodata = np.where(xerr_p < 0)

    xerr_p = np.sqrt(xerr_p**2 - addErr**2)
    yerr_p = np.sqrt(yerr_p**2 - addErr**2)
    
    xerr_p[x_idx] = fixErr
    yerr_p[y_idx] = fixErr
    xerr_p[nodata] = -1
    yerr_p[nodata] = -1

    # Calculate the residuals from the line fits and
    # again average over all epochs.
    x0 = s.getArray('fitXv.p') * -1.0
    vx = s.getArray('fitXv.v') * -1.0
    y0 = s.getArray('fitYv.p')
    vy = s.getArray('fitYv.v')
    t0 = s.getArray('fitXv.t0')
    t = s.years
    
    res = np.zeros(numStars, dtype=float)

    err_p = np.zeros(numStars, dtype=float)
    err_a = np.zeros(numStars, dtype=float)

    for ss in range(numStars):
        # Average these over all epochs assuming that the
        # epochs are equivalent quality... close enough
        idx = np.where(xerr_p[:,ss] > 0)[0]

        if avgXY:
            err_p[ss] = ((xerr_p[idx,ss] + yerr_p[idx,ss]) / 2.0).mean()
            err_a[ss] = ((xerr_a[idx,ss] + yerr_a[idx,ss]) / 2.0).mean()
        else:
            err_p[ss] = np.hypot(xerr_p[idx,ss], yerr_p[idx,ss]).mean()
            err_a[ss] = np.hypot(xerr_a[idx,ss], yerr_a[idx,ss]).mean()

        xfit = x0[ss] + vx[ss] * (t[idx] - t0[ss])
        yfit = y0[ss] + vy[ss] * (t[idx] - t0[ss])

        xobs = np.array([s.stars[ss].e[ee].x for ee in idx])
        yobs = np.array([s.stars[ss].e[ee].y for ee in idx])
        
        xres = np.abs(xfit - xobs).mean() * 10**3 # in mas
        yres = np.abs(yfit - yobs).mean() * 10**3 # in mas

        if avgXY:
            res[ss] = (xres + yres)/2.0
        else:
            res[ss] = math.hypot(xres, yres)

    mag = s.getArray('mag')
    magBinStep = 1.0
    magBinCent = np.arange(10, 19, magBinStep)

    err_p_med = np.zeros(len(magBinCent), dtype=float)
    err_a_med = np.zeros(len(magBinCent), dtype=float)
    res_med = np.zeros(len(magBinCent), dtype=float)

    for ii in range(len(magBinCent)):
        lo = magBinCent[ii] - magBinStep/2.0
        hi = magBinCent[ii] + magBinStep/2.0
        tmp = np.where((mag >= lo) & (mag < hi))[0]

        err_p_med[ii] = np.median(err_p[tmp])
        err_a_med[ii] = np.median(err_a[tmp])

        res_med[ii] = np.median(res[tmp])

    # Alignment error does not really depend on magnitude, so 
    # make it a constant.
    alignErr = err_a_med.mean()

    # Make a theoretical curve that scales with signal to noise ratio.
    # There is some arbitrary scale factor here.
    mag4snr = np.arange(magBinCent.min(), magBinCent.max()+1, 0.1)
    flux4snr = 10**(-0.4 * mag4snr)
    err4snr = 0.8 * np.sqrt(flux4snr[-1] / flux4snr)
    
    # Get the best fit theoretical curve that scales with signal to noise
    # ratio from star platning tests.
    mag4snr = np.arange(magBinCent.min(), magBinCent.max()+1, 0.1)
    magParams, xyParams = gc_plot_ast_err_vs_snr()
    err4snr2 = 10**np.polyval(xyParams, mag4snr)   # This is in mas
    #err4snr2 *= 10.0

    # Get the best fit theoretical curve that scales with signal to noise
    # ratio from star platning tests.
    magParamsIso, xyParamsIso = plot_ast_err_vs_snr()
    err4snr2Iso = 10**np.polyval(xyParamsIso, mag4snr)   # This is in pixels
    err4snr2Iso *= 10.0


    # Get errors from PSF uncertainties from Fritz et al. (2010)
    psfErrRadius = np.array([0.22, 0.47, 0.77, 1.07, 1.36, 2.07, 3.18])
    psfErrNorm = np.array([0.273, 0.176, 0.122, 0.133, 0.110, 0.105, 0.82])

    psfErrItself = 0.050

    psf_mag = np.arange(10, 19, 0.1)
    psf_err_itself = np.ones(len(psf_mag), dtype=float) * psfErrItself
    psf_err_others = 0.082 * 10**(0.4 * (psf_mag - 14))
    


    py.figure(1)
    py.clf()
    py.plot(mag, err_p, 'k.')

    py.figure(2)
    py.clf()
    py.plot(magBinCent, res_med, 'b-', linewidth=2, label='Residuals Over Time')
    py.plot(magBinCent, err_p_med, 'k.-', linewidth=2, label='Centroid Errors')

    rng = py.axis()
    py.plot([rng[0], rng[1]], [alignErr, alignErr], 'r--', label='Align Errors')

    #py.plot(mag4snr, err4snr, 'g--', label=r'Errors from SNR')
    py.semilogy(mag4snr, err4snr2, 'g--', label=r'Photon Noise Errors')
#     py.semilogy(mag4snr, err4snr2Iso, 'b--', label=r'Errors from SNR (isolated)')

#     py.semilogy(psf_mag, psf_err_itself, 'r-', label='Uncertainty on Star from PSF Errors')
#     py.semilogy(psf_mag, psf_err_others, 'c--', label='Uncertainty on Neighbors from PSF Errors')
    
    py.ylim(0.01, 1)
    py.xlabel('K Magnitude')
    py.ylabel('Positional Errors (mas)')
    py.legend(loc='upper left')
    py.title('Radius Cut = %d, avgXY = %s' % (radiusCut, str(avgXY)))

    fileName = 'ast_err_vs_mag_r%d' % radiusCut
    if avgXY:
        fileName += '_avg'
    py.savefig(fileName + '.png')
    
    idx = np.where((magBinCent >= 11) & (magBinCent <= 14))[0]
    if avgXY:
        print 'Averaging X and Y errors and residuals.'
    else:
        print 'Quad-summing X and Y errors and residuals.'
    print 'Mean Residuals between 11<=K<=14:  %5.3f mas' % res_med[idx].mean()
    print 'Mean Pos. Error between 11<=K<=14: %5.3f mas' % err_p_med[idx].mean()
    print 'Mean Aln. Error between 11<=K<=14: %5.3f mas' % err_a_med[idx].mean()

def plot_photometric_errors(radiusCut=4.0):
    dir = '/u/jlu/uni/syelda/gc/aligndir/06setup_only/10_05_23/'

    alignRoot = 'align/align_d_rms_1000_abs_t'
    polyRoot = 'polyfit_1000/fit'
    
    s = starset.StarSet(dir + alignRoot)
    s.loadPolyfit(dir + polyRoot)
    x = s.getArray('x')
    y = s.getArray('y')
    r = np.hypot(x, y)

    # Trim stars down to those within 4"
    print 'Before Trim: ', len(s.stars)
    velCount = s.getArray('velCnt')
#     idx = np.where((r > 0.5) & (r < radiusCut))[0]
    idx = np.where((r > 0.5) & (r < radiusCut) & (velCount == velCount.max()))[0]
    s.stars = [s.stars[ii] for ii in idx]
    print 'After Trim: ', len(s.stars)

    numEpochs = len(s.years)
    numStars = len(s.stars)
    print numEpochs

    m = s.getArray('mag')
    name = s.getArray('name')
    mag = s.getArrayFromAllEpochs('mag')
    snr = s.getArrayFromAllEpochs('snr')
    magerr = 1.086 / snr

    res = np.zeros(numStars, dtype=float)
    err = np.zeros(numStars, dtype=float)

    for ss in range(numStars):
        # Average these over all epochs assuming that the
        # epochs are equivalent quality... close enough
        idx = np.where(mag[:,ss] > 0)[0]

        err[ss] = magerr[idx,ss].mean()
        res[ss] = mag[idx,ss].std()
        if (name[ss] == 'S2-333'):
            print mag[idx,ss]
            print magerr[idx,ss]

    magBinStep = 1.0
    magBinCent = np.arange(10, 19, magBinStep)

    err_med = np.zeros(len(magBinCent), dtype=float)
    res_med = np.zeros(len(magBinCent), dtype=float)

    for ii in range(len(magBinCent)):
        lo = magBinCent[ii] - magBinStep/2.0
        hi = magBinCent[ii] + magBinStep/2.0
        tmp = np.where((m >= lo) & (m < hi))[0]

        err_med[ii] = np.median(err[tmp])
        res_med[ii] = np.median(res[tmp])

    py.figure(2)
    py.clf()
    py.plot(m, res, 'k.', label='Individual Stars')
    py.plot(magBinCent, res_med, 'b-', linewidth=2, label='Median')
    py.ylim(0.0, 0.2)
    py.xlabel('K Magnitude')
    py.ylabel('Photometric Errors (mag)')
    py.legend(loc='upper left')
    py.title('Radius Cut = %d' % (radiusCut))

    fileName = 'phot_err_vs_mag_r%d' % radiusCut
    py.savefig(fileName + '.png')
    
    idx = np.where((magBinCent >= 11) & (magBinCent <= 14))[0]
    print 'Mean Residuals between 11<=K<=14:  %5.3f mas' % res_med[idx].mean()
    print 'Mean Phot. Error between 11<=K<=14: %5.3f mas' % err_med[idx].mean()
    
    
def image_noise_properites():
    print 'READ NOISE:'
    hdr = pyfits.getheader(workDir + 'mag10maylgs_kp.fits')
    itime = hdr['ITIME']
    coadds = hdr['COADDS']
    nsamp = hdr['MULTISAM']
    gain = float(hdr['GAIN'])

    readnoise = (16.0 / math.sqrt(nsamp)) * math.sqrt(coadds) # in DN
    readnoise *= gain # convert to e-
    
    print '    MCDS with %d Fowler samples and %d coadds' % (nsamp, coadds)
    print '    noise = %.1f DN (%.1f e-)' % (readnoise/gain, readnoise)
    print ''


    # Figure out the dark current in a 2.8 sec x 10 coadd image.
    darkImg = pyfits.getdata(workDir + 'dark_2.8s_10ca.fits')
    darkImgFlat = darkImg.flatten()
    darkCurrent = istats.mean(darkImg, lsigma=5, hsigma=5, iter=10)*gain
    darkNoise = istats.std(darkImg, lsigma=5, hsigma=5, iter=10)*gain
    print 'DARK:'
    print '    Dark Current = %.1f DN (%.1f e-)' % \
        (darkCurrent/gain, darkCurrent)
    print '    Dark Noise = %.1f DN (%.1f e-)' % \
        (darkNoise/gain, darkNoise)
    print '    !! Dark noise does not match expected readnoise.'
    print ''
    
    py.clf()
    bins = np.arange(-100, 100, 1)
    fit = stats.norm.pdf(bins, loc=darkCurrent/gain, scale=darkNoise/gain) 
    loCut = (darkCurrent - 3.0 * darkNoise) / gain
    hiCut = (darkCurrent + 3.0 * darkNoise) / gain
    idx = np.where((darkImgFlat > loCut) & (darkImgFlat < hiCut))[0]
    fit *= len(idx)
    py.hist(darkImgFlat, bins=bins, histtype='step', label='Observed')
    py.plot(bins, fit, 'k-', linewidth=2, label='Gaussian with Mean/STD')
    py.xlabel('Counts from Dark Image (DN)')
    py.ylabel('Number of Pixels')
    py.legend()
    py.title('Mean Current = %.1f DN, Stddev = %.1f DN' % \
                 (darkCurrent/gain, darkNoise/gain))

    # Reset so that readnoise = darkNoise
#     print 'SETTING READ NOISE = DARK NOISE'
#     readnoise = darkNoise

    # Sky background flux
    skyImg = pyfits.getdata(workDir + 'mag10maylgs_sky_kp.fits')
    skyCount = 10.0
    skyLevel = istats.mean(skyImg, lsigma=5, hsigma=5, iter=5) * gain
    skyStd = istats.std(skyImg, lsigma=5, hsigma=5, iter=5) * gain
    skyNoise = math.sqrt((skyLevel + readnoise**2) / skyCount)
    print 'SKY:'
    print '    Sky Level = %.1f DN (%.1f e-)' % (skyLevel/gain, skyLevel)
    print '    Sky Stddev = %.1f DN (%.1f e-)' % (skyStd/gain, skyStd)
    print '    Assuming %d sky exposures' % skyCount
    print '    Sky Noise = %.1f DN (%.1f e-)' % (skyNoise/gain, skyNoise)
    print ''

    # GC background flux
    backLevelDC = 90.0 * gain   # Subtracted DC offset
    backNoiseDC = math.sqrt(backLevelDC)
    backImg = pyfits.getdata(workDir + 'mag10maylgs_kp_back.fits')
    backLevel = istats.mean(backImg, lsigma=5, hsigma=5, iter=5) * gain
    backNoise = math.sqrt(backLevel)
    backStd = istats.std(backImg, lsigma=5, hsigma=5, iter=5) * gain
    print 'GC BACKGROUND:'
    print '    GC Background subtracted off is 90 DN (360 e-)'
    print '    Background Level = %.1f DN (%.1f e-)' % \
        (backLevel/gain, backLevel)
    print '    Background Stddev = %.1f DN (%.1f e-)' % \
        (backStd/gain, backStd)
    print '    Background Photon Noise = %.1f DN (%.1f e-)' % \
        (backNoise/gain, backNoise)
    print '    Actual Photon Noise = %.1f DN (%.1f e-)' % \
        (backNoiseDC/gain, backNoiseDC)
    print ''

    # Final Noise Estimate
    skyFact = 1.0 + (1.0 / skyCount)
    gcCount = 150
    gcNoise = skyFact * (skyLevel + readnoise**2)
    gcNoise += backLevelDC
    gcNoise /= gcCount
    gcNoise = math.sqrt(gcNoise)
    print 'FINAL GC NOISE (excluding photon noise from stars):'
    print '    %.1f DN (%.1f e-)' % (gcNoise/gain, gcNoise)
    print ''

    gcNoise = math.sqrt((skyNoise**2 + readnoise**2) / gcCount)
    print 'GC NOISE added to sky/DC/star photon noise (from sky subtraction and readnoise):'
    print '    %.1f DN (%.1f e-)' % (gcNoise/gain, gcNoise)
    print ''

    return skyLevel, backLevelDC, gcNoise, gain

def ast_err_vs_snr(n_init=0, n_sims=1000):
    """
    Analyze how starfinder astrometric error changes with signal-to-noise
    ratio. Do this by planting stars on a simulated sky background and
    readnoise/dark current image. Then try to recover the stars with
    starfinder and measure the astrometric errors.
    """
    sky, dc, noise, gain = image_noise_properites() # IN e-
    
    background = (sky + dc)  # in e-

    imgCount = 150 # The total number of individual exposures that went
                   # into this image.

    print ''
    print 'SIMULATED IMAGE PROPERTIES:'
    print '  assuming %d science exposures' % imgCount
    print '  constant = %.1f DN (%.1f e-)' % (background/gain, background)
    print '  additional noise approximated with gaussian'
    print '  sigma = %.1f DN (%.1f e-)' % (noise/gain, noise)
    
    img = np.zeros((1024, 1024), dtype=float)
    img += background

    # Noise from read out and sky subtraction (in e-)
    otherNoise = stats.norm.rvs(scale=noise, size=(1024,1024))

    # We are going to plant stars on this image using one of our
    # own PSFs. Each star we plant has to have photon noise from 
    # itself as well.
    psf, hdr = pyfits.getdata('mag10maylgs_kp_psf.fits', header=True)
    # Repair the PSF so we don't have negative values
    idx1 = np.where(psf <= 0)
    idx2 = np.where(psf > 0)
    psf[idx1] = psf[idx2].min()

    # Seperate the PSFs by 2" since this is the Starfinder box size.
    step = 200  # in pixels
    
    # Pulled these numbers from the calibrated starlists for 10maylgs.
    mag0 = 9.782
    flux0 = 20522900.000

    # Randomly select from a distribution of magnitudes.
    magMin = 9.0
    magMax = 22.0

    # Number of samples (stars planted)
    #numStars = 100
    stopN = n_sims + n_init

    allIDLroots = []

    nn = n_init
    while nn <= stopN:
        # Create a new image
        newImage = img.copy()
        brightCoords = None
        brightMag = 1000

        # Make a new starlist with the input info.
        newStars = starTables.StarfinderList(None, hasErrors=False)

        # Now plant the stars
        for xx in range(step, img.shape[1]-step, step):
            for yy in range(step, img.shape[0]-step, step):
                mag = np.random.uniform(magMin, magMax)
                flux = flux0 * 10**(-0.4 * (mag - mag0))

                # We need to add photon noise to the PSF.
                # Scale up the PSF temporarily since poisson() only 
                # returns integers.
                tmp = psf * flux * gain

                #print '%4s  x = %4d  y = %4d  mag = %5.2f  flux = %.2f' % \
                #    (nn, xx, yy, mag, flux)
                
                newName = 'sim_%s' % str(nn).zfill(4)

                # set flux=1 since we already scaled it.
                starPlant.addStar(newImage, tmp, xx, yy, 1.0) 
                newStars.append(newName, mag, xx-1, yy-1, 
                                counts=flux, epoch=2010.342)

                # Updare our record of the brightest star in the image.
                if mag < brightMag:
                    brightMag = mag
                    brightCoords = [xx, yy]

                nn += 1

        # Fix negative pixels
        idx = np.where(newImage < 0)
        newImage[idx] = 0

        # Now add noise to the image, compensate for large number of exposures
        newImage = stats.poisson.rvs((newImage * imgCount).tolist())
        newImage = np.array(newImage, dtype=float) # necessary for starfinder
        newImage /= imgCount
        newImage += otherNoise

        # Subtract off sky/dark
        newImage -= background
        
        # Convert back to DN
        newImage /= gain

        # Make a background image.
        img_back = np.zeros((1024, 1024), dtype=float)
        
        # Save the new image, and copy over all the necessary files.
        suffix = str(nn).zfill(4)
        fitsFile = 'sim_img_%s.fits' % suffix
        psfFile = 'sim_img_%s_psf.fits' % suffix
        backFile = 'sim_img_%s_back.fits' % suffix
        cooFile = 'sim_img_%s.coo' % suffix
        lisFile = 'sim_orig_%s.lis' % suffix

        print 'Writing ', fitsFile

        gcutil.rmall([fitsFile, backFile])
        pyfits.writeto(fitsFile, newImage, hdr,
                       output_verify='silentfix')
        pyfits.writeto(backFile, img_back, hdr,
                       output_verify='silentfix')
        
        # Resort the starlist so the brightest star is at the top
        mdx = newStars.mag.argsort()
        newStars.name = newStars.name[mdx]
        newStars.mag = newStars.mag[mdx]
        newStars.epoch = newStars.epoch[mdx]
        newStars.x = newStars.x[mdx]
        newStars.y = newStars.y[mdx]
        newStars.snr = newStars.snr[mdx]
        newStars.corr = newStars.corr[mdx]
        newStars.nframes = newStars.nframes[mdx]
        newStars.counts = newStars.counts[mdx]
        newStars.saveToFile(lisFile)
        
        shutil.copyfile('mag10maylgs_kp_psf.fits', psfFile)
        _coo = open(cooFile, 'w')
        _coo.write('%.2f %.2f\n' % (brightCoords[0], brightCoords[1]))
        _coo.close()

        # Write out a starfinder batch file
        idlRoot = 'idl_sim_img_%s' % suffix
        _batch = open(idlRoot + '.batch', 'w')
        _batch.write("find_stf, ")
        _batch.write("'" + fitsFile + "', 0.8, /trimfake\n")
        _batch.write("exit\n")
        _batch.close()

        gcutil.rmall([idlRoot + '.log'])

        allIDLroots.append(idlRoot)

    for root in allIDLroots:
        print 'idl < %s.batch >& %s.log' % (root, root)


def align_ast_err_vs_snr(inputHasErrors=False):
    newLists = glob.glob('sim_img*.lis')
    fileNum = []

    for ii in range(len(newLists)):
        tmp = newLists[ii].replace('sim_img_', '').replace('_0.8_stf.lis', '')
        fileNum.append(tmp)

    for ii in range(len(fileNum)):
        print 'Aligning ', fileNum[ii]
        inputList = 'sim_orig_%s.lis' % fileNum[ii]
        outputList = 'sim_img_%s_0.8_stf.lis' % fileNum[ii]

        if ((os.path.exists(inputList) == False) or
            (os.path.exists(outputList) == False)):
            continue
            
        alignRoot = 'align/aln_%s' % fileNum[ii]
        
        _alignList = open(alignRoot + '.list', 'w')
        if inputHasErrors:
            _alignList.write(inputList + ' 9 ref\n')
        else:
            _alignList.write(inputList + ' 8 ref\n')
        _alignList.write(outputList + ' 8\n')
        _alignList.close()

        cmd = 'java align -a 0 -v -p -r ' + alignRoot  + ' '
        cmd += alignRoot + '.list'
        os.system(cmd)
        

def plot_ast_err_vs_snr():
    alnVelFiles = glob.glob(workDir + 'align/aln_*.vel')

    # Here are the arrays we want to collect
    fratio = np.array([], dtype=float)
    xdiff = np.array([], dtype=float)
    ydiff = np.array([], dtype=float)
    mag = np.array([], dtype=float)
    flux = np.array([], dtype=float)

    for aa in range(len(alnVelFiles)):
        s = starset.StarSet(alnVelFiles[aa].replace('.vel', ''))
        velCount = s.getArray('velCnt')
        newStars = []
        for ss in range(len(s.stars)):
            if velCount[ss] == 2:
                newStars.append(s.stars[ss])

        s.stars = newStars
        print '%d stars in %s' % (len(newStars), alnVelFiles[aa])


        # Need to compare STF fluxes directly rather than 
        # the magnitudes since they aren't calibrated the same way
        # at all.
        name = s.getArray('name')
        f_in = s.getArrayFromEpoch(0, 'fwhm')
        f_out = s.getArrayFromEpoch(1, 'fwhm')
        m_in = s.getArrayFromEpoch(0, 'mag') # these are calibrated properly

        x_in = s.getArrayFromEpoch(0, 'xorig')
        y_in = s.getArrayFromEpoch(0, 'yorig')

        x_out = s.getArrayFromEpoch(1, 'xorig')
        y_out = s.getArrayFromEpoch(1, 'yorig')

#         for ii in range(len(f_in)):
#             print '%10s  mag = %5.2f  f_in = %10.1f  f_out = %10.1f' % \
#                 (name[ii], m_in[ii], f_in[ii], f_out[ii])

        fratio = np.append(fratio, f_in / f_out)
        xdiff = np.append(xdiff, x_in - x_out)
        ydiff = np.append(ydiff, y_in - y_out)
        mag = np.append(mag, m_in)
        flux = np.append(flux, f_in)

    # Calculate Magnitude Errors
    mdiff = -2.5 * np.log10(fratio)

    # Make some binned statistics
    mstep = 0.5
    mbins = np.arange(math.floor(mag.min()), math.ceil(mag.max()), mstep)
    
    mdiff_std = np.zeros(len(mbins))
    mdiff_absmed = np.zeros(len(mbins))
    xdiff_std = np.zeros(len(mbins))
    xdiff_absmed = np.zeros(len(mbins))
    ydiff_std = np.zeros(len(mbins))
    ydiff_absmed = np.zeros(len(mbins))

    for mm in range(len(mbins)):
        idx = np.where((mag > mbins[mm]) & (mag <= mbins[mm] + mstep))[0]
        
        mdiff_std[mm] = mdiff[idx].std()
        mdiff_absmed[mm] = np.median( np.abs( mdiff[idx] ) )

        xdiff_std[mm] = xdiff[idx].std()
        xdiff_absmed[mm] = np.median( np.abs( xdiff[idx] ) )

        ydiff_std[mm] = ydiff[idx].std()
        ydiff_absmed[mm] = np.median( np.abs( ydiff[idx] ) )


    # Make a combo xy since they are the same
    xydiff_std = (xdiff_std + ydiff_std) / 2.0

    mbinCenter = mbins + (mstep/2.0)

    # Fit polynomials to the photo/astro RMS errors vs. mag (in log space)
    idx = np.where(xydiff_std > 0)[0]

    print 'Finding best fit polynomial for photometry'
    mparams = np.polyfit(mbinCenter, np.log10(mdiff_std), 2)
    print 'Finding best fit polynomial for astrometry'
    xyparams = np.polyfit(mbinCenter[idx], np.log10(xydiff_std[idx]), 2)

    mfit = 10**np.polyval(mparams, mbinCenter)
    xyfit = 10**np.polyval(xyparams, mbinCenter)

    print 'Phometric Error = 10^(%f + %f*m + %f*m^2)' % (mparams[0],
                                                         mparams[1], 
                                                         mparams[2])
    print 'Astrometric Error = 10^(%f + %f*m + %f*m^2)' % (xyparams[0],
                                                           xyparams[1], 
                                                           xyparams[2])
    

    # ----------
    # Plotting
    # ----------
    print 'Plotting'

    # Brightness
    py.clf()
    py.plot(mag, mdiff, 'b.')
    py.xlabel('Input Brightness (mag)')
    py.ylabel('Input - Output Brightness (mag)')
    py.savefig('plots/sim_mdiff_vs_mag.png')

    py.clf()
    py.semilogy(mag, np.abs(mdiff), 'b.')
    py.xlabel('Input Brightness (mag)')
    py.ylabel('abs[ Input - Output ] Brightness (mag)')
    py.savefig('plots/sim_mdiff_vs_mag_abs.png')

    py.clf()
    py.semilogy(mbinCenter, mdiff_std, 'bo', label='RMS error')
    py.semilogy(mbinCenter, mdiff_absmed, 'rs', 
                label='Median of Abs. Value')
#     py.semilogy(mbinCenter, mfit, 'b--', label='Best Fit')
    py.xlabel('Input Brightness (mag)')
    py.ylabel('Brightness Error (mag)')
    py.legend(loc='upper left')
    py.savefig('plots/sim_mdiff_vs_mag_rms.png')

    # Astrometry
    py.clf()
    py.plot(mag, xdiff, 'r.', label='X')
    py.plot(mag, ydiff, 'b.', label='Y')
    py.xlabel('Input Brightness (mag)')
    py.ylabel('Input - Output Positions (pix)')
    py.legend()
    py.savefig('plots/sim_xydiff_vs_mag.png')

    py.clf()
    py.semilogy(mag, np.abs(xdiff), 'r.', label='X')
    py.semilogy(mag, np.abs(ydiff), 'b.', label='Y')
    py.xlabel('Input Brightness (mag)')
    py.ylabel('abs[ Input - Output ] Positions (pix)')
    py.legend()
    py.savefig('plots/sim_xydiff_vs_mag_abs.png')
    
    py.clf()
    py.semilogy(mbinCenter, xdiff_std, 'bs', label='X RMS error')
    py.semilogy(mbinCenter, xdiff_absmed, 'bo', 
                label='X Median of Abs. Value') 
    py.semilogy(mbinCenter, ydiff_std, 'rs', label='Y RMS error')
    py.semilogy(mbinCenter, ydiff_absmed, 'ro', 
                label='Y Median of Abs. Value')
#     py.semilogy(mbinCenter, xyfit, 'b--', label='XY Best Fit')
    py.xlabel('Input Brightness (mag)')
    py.ylabel('Positional Errors (pix)')
    py.legend(numpoints=1, loc='upper left')
    py.savefig('plots/sim_xydiff_vs_mag_rms.png')
    
    return mparams, xyparams
    
def gc_ast_err_vs_snr(n_init=0, n_sims=10):
    """
    Analyze how starfinder astrometric error changes with signal-to-noise
    ratio. Do this by planting known GC stars on a simulated sky background and
    readnoise/dark current image. Then try to recover the stars with
    starfinder and measure the astrometric errors.
    """
    workDir = '/u/jlu/work/gc/ao_performance/ast_err_vs_snr/gc/'

    sky, dc, noise, gain = image_noise_properites() # IN e-
    
    background = (sky + dc)  # in e-

    imgCount = 150 # The total number of individual exposures that went
                   # into this image.

    print ''
    print 'SIMULATED IMAGE PROPERTIES:'
    print '  assuming %d science exposures' % imgCount
    print '  constant = %.1f DN (%.1f e-)' % (background/gain, background)
    print '  additional noise approximated with gaussian'
    print '  sigma = %.1f DN (%.1f e-)' % (noise/gain, noise)

    # Start with the background that Starfinder fit... it is smoothed.
    img = pyfits.getdata('mag10maylgs_kp_back.fits')
    img += background * gain

    # Noise from read out and sky subtraction (in e-)
    otherNoise = stats.norm.rvs(scale=noise, size=img.shape)

    # We are going to plant stars on this image using one of our
    # own PSFs. Each star we plant has to have photon noise from 
    # itself as well.
    psf, hdr = pyfits.getdata('mag10maylgs_kp_psf.fits', header=True)
    # Repair the PSF so we don't have negative values
    idx1 = np.where(psf <= 0)
    idx2 = np.where(psf > 0)
    psf[idx1] = psf[idx2].min()

    # Read in the GC starlist 
    gcstars = starTables.StarfinderList('mag10maylgs_kp_rms.lis', 
                                        hasErrors=True)

    allIDLroots = []

    for nn in range(n_init, n_init+n_sims):
        # Create a new image
        newImage = img.copy()

        # Now plant the stars
        for ii in range(len(gcstars.x)):
            mag = gcstars.mag[ii]
            flux = gcstars.counts[ii]

            # We need to add photon noise to the PSF.
            # Scale up the PSF temporarily since poisson() only 
            # returns integers.
            tmp = psf * flux * gain

            # set flux=1 since we already scaled it.
            starPlant.addStar(newImage, tmp, 
                              gcstars.x[ii]+1.0, gcstars.y[ii]+1.0, 1.0) 
            # Planted positions are actually rounded to the nearest 
            # pixel to avoid interpolation noise.... do the same in the 
            # input star list.
            gcstars.x[ii] = round(gcstars.x[ii])
            gcstars.y[ii] = round(gcstars.y[ii])

        # Fix negative pixels
        idx = np.where(newImage < 0)
        newImage[idx] = 0

        # Now add noise to the image, compensate for large number of exposures
        newImage = stats.poisson.rvs((newImage * imgCount).tolist())
        newImage = np.array(newImage, dtype=float) # necessary for starfinder
        newImage /= imgCount
        newImage += otherNoise

        # Subtract off sky/dark
        newImage -= background
        
        # Convert back to DN
        newImage /= gain

        # Save the new image, and copy over all the necessary files.
        suffix = str(nn).zfill(4)
        fitsFile = 'sim_img_%s.fits' % suffix
        psfFile = 'sim_img_%s_psf.fits' % suffix
        backFile = 'sim_img_%s_back.fits' % suffix
        cooFile = 'sim_img_%s.coo' % suffix
        lisFile = 'sim_orig_%s.lis' % suffix

        print 'Writing ', fitsFile

        gcutil.rmall([fitsFile])
        pyfits.writeto(fitsFile, newImage, hdr,
                       output_verify='silentfix')
        shutil.copyfile('mag10maylgs_kp_psf.fits', psfFile)
        shutil.copyfile('mag10maylgs_kp_back.fits', backFile)
        shutil.copyfile('mag10maylgs_kp.coo', cooFile)
        gcstars.saveToFile(lisFile)

        # Write out a starfinder batch file
        idlRoot = 'idl_sim_img_%s' % suffix
        _batch = open(idlRoot + '.batch', 'w')
        _batch.write("find_stf, ")
        _batch.write("'" + fitsFile + "', 0.8, /trimfake\n")
        _batch.write("exit\n")
        _batch.close()

        gcutil.rmall([idlRoot + '.log'])

        allIDLroots.append(idlRoot)

    for root in allIDLroots:
        print 'idl < %s.batch >& %s.log' % (root, root)


def gc_align_ast_err_vs_snr():
    newLists = glob.glob('sim_img*.lis')
    fileNum = []

    for ii in range(len(newLists)):
        tmp = newLists[ii].replace('sim_img_', '').replace('_0.8_stf.lis', '')
        fileNum.append(tmp)

    # Align everything to a signle input list
    inputList = 'sim_orig_%s.lis' % fileNum[0]

    alignRoot = 'align/aln_all'

    _alignList = open(alignRoot + '.list', 'w')
    _alignList.write(inputList + ' 9 ref\n')

    # Write the *.list file
    for ii in range(len(fileNum)):
        outputList = 'sim_img_%s_0.8_stf.lis' % fileNum[ii]

        if (os.path.exists(outputList) == False):
            continue
        
        _alignList.write(outputList + ' 8\n')

    _alignList.close()

    # Run Align
    cmd = 'java -Xmx512m align -a 0 -v -p -r ' + alignRoot  + ' '
    cmd += alignRoot + '.list'
    os.system(cmd)
        
def gc_plot_ast_err_vs_snr():
    s = starset.StarSet(workDir + 'gc/align/aln_all_t')
    starCount = len(s.stars)
    epochCount = len(s.years)

    # Here are the arrays we want to collect
    df_mean = np.zeros(starCount, dtype=float)
    dx_mean = np.zeros(starCount, dtype=float)
    dy_mean = np.zeros(starCount, dtype=float)
    dm_mean = np.zeros(starCount, dtype=float)
    df_std = np.zeros(starCount, dtype=float)
    dx_std = np.zeros(starCount, dtype=float)
    dy_std = np.zeros(starCount, dtype=float)
    dm_std = np.zeros(starCount, dtype=float)
    df_emean = np.zeros(starCount, dtype=float)
    dx_emean = np.zeros(starCount, dtype=float)
    dy_emean = np.zeros(starCount, dtype=float)
    dm_emean = np.zeros(starCount, dtype=float)

    dx_mean_abs = np.zeros(starCount, dtype=float)
    dy_mean_abs = np.zeros(starCount, dtype=float)
    dx_std_abs = np.zeros(starCount, dtype=float)
    dy_std_abs = np.zeros(starCount, dtype=float)

    # Get the starfinder information
    mag = s.getArrayFromEpoch(0, 'mag') # properly calibrated
    x_in = s.getArrayFromEpoch(0, 'xorig') # properly calibrated
    y_in = s.getArrayFromEpoch(0, 'yorig') # properly calibrated
    x2d = s.getArrayFromAllEpochs('xorig')
    y2d = s.getArrayFromAllEpochs('yorig')
    f2d = s.getArrayFromAllEpochs('fwhm')

    # Now loop through the stars and get the appropriate information
    for ss in range(starCount):
        dx = x_in[ss] - x2d[1:,ss]
        dy = y_in[ss] - y2d[1:,ss]
        df = f2d[0,ss] / f2d[1:,ss]

        dm = -2.5 * np.log10(df)

        # Get the means just to check for zero mean
        df_mean[ss] = df.mean()
        dx_mean[ss] = dx.mean() * 10
        dy_mean[ss] = dy.mean() * 10
        dm_mean[ss] = -2.5 * np.log10(df_mean[ss])

        # Get the RMS error which we will take as one estimate of the 
        # positional errors for each star.
        df_std[ss] = df.std()
        dx_std[ss] = dx.std() * 10
        dy_std[ss] = dy.std() * 10
        dm_std[ss] = dm.std()

        # Get the error on the mean.
        df_emean[ss] = df_std[ss] / np.sqrt(epochCount)
        dx_emean[ss] = dx_std[ss] / np.sqrt(epochCount)
        dy_emean[ss] = dy_std[ss] / np.sqrt(epochCount)
        dm_emean[ss] = dm_std[ss] / np.sqrt(epochCount)

        # Now take the absolute value of the differences BEFORE 
        # calculating averages. Only applied to positions.
        dx_mean_abs[ss] = np.abs(dx).mean() * 10
        dy_mean_abs[ss] = np.abs(dy).mean() * 10
        dx_std_abs[ss] = np.abs(dx).std() * 10
        dy_std_abs[ss] = np.abs(dy).std() * 10
        

    # Make a combo xy since they are the same
    dxy_std = (dx_std + dy_std) / 2.0
    dxy_emean = (dx_emean + dy_emean) / 2.0

    idx = np.where((dx_std > 0) & (dy_std > 0))[0]

    # Fit polynomials to the photo/astro RMS errors vs. mag (in log space)
    print 'Finding best fit polynomial for photometry'
    mparams = np.polyfit(mag[idx], np.log10(dm_std[idx]), 2)
    print 'Finding best fit polynomial for astrometry'
    xyparams = np.polyfit(mag[idx], np.log10(dxy_std[idx]), 2)

    magFit = np.arange(8, 20, 0.1)
    mfit = 10**np.polyval(mparams, magFit)
    xyfit = 10**np.polyval(xyparams, magFit)

    print 'Phometric Error = 10^(%f + %f*m + %f*m^2)' % (mparams[0],
                                                         mparams[1], 
                                                         mparams[2])
    print 'Astrometric Error = 10^(%f + %f*m + %f*m^2)' % (xyparams[0],
                                                           xyparams[1], 
                                                           xyparams[2])
    
    # Check for zero means
    py.clf()
    py.plot(mag, dx_mean, 'r.')
    py.plot(mag, dy_mean, 'b.')
    py.xlabel('K Magnitude')
    py.ylabel('Mean( x_in - x_out ) in mas')
    py.ylim(-0.03, 0.03)
    py.savefig('plots/gc_dpos_mean.png')

    py.clf()
    py.plot(mag, dm_mean, 'k.')
    py.xlabel('K Magnitude')
    py.ylabel('Mean Brightness Difference (mag)')
    py.ylim(-0.1, 0.1)
    py.savefig('plots/gc_dmag_mean.png')

    # Plot all the stars up just to see the full distribution
    py.clf()
    py.errorbar(mag, dx_std, fmt='r.', yerr=dx_emean)
    py.errorbar(mag, dy_std, fmt='b.', yerr=dy_emean)
    py.plot(magFit, xyfit, 'k--') # plot fit
    py.gca().set_yscale('log')
    py.xlabel('K Magnitude')
    py.ylabel('Positional STD(in-out) in mas')
    py.savefig('plots/gc_dpos_std.png')
    
    py.clf()
    py.errorbar(mag, dm_std, fmt='k.', yerr=dm_emean)
    py.plot(magFit, mfit, 'r--') # plot fit
    py.gca().set_yscale('log')
    py.xlabel('K Magnitude')
    py.ylabel('Brightness STD(in-out) in mag')
    py.savefig('plots/gc_dmag_std.png')

    # Same for the absolute value metrics
    py.clf()
    py.errorbar(mag, dx_mean_abs, fmt='r.', yerr=dx_std_abs)
    py.errorbar(mag, dy_mean_abs, fmt='b.', yerr=dy_std_abs)
    py.gca().set_yscale('log')
    py.xlabel('K Magnitude')
    py.ylabel('Positional Mean |in-out| in mas')
    py.savefig('plots/gc_dpos_abs_mean.png')



    return mparams, xyparams


def plot_fritz_fig14():
    psfErrRadius = np.array([0.22, 0.47, 0.77, 1.07, .136, 2.07, 3.18])
    psfErrNorm = np.array([0.273, 0.176, 0.122, 0.133, 0.110, 0.105, 0.82])

    psfErrItself = 0.050

    mag = np.arange(10, 19, 0.1)
    psf_err_itself = np.ones(len(mag), dtype=float) * psfErrItself
    psf_err_others = psfErrNorm[2] * 10**(0.4 * (mag - 14))
    
    py.clf()
    py.semilogy(mag, psf_err_others, 'k-', label='Uncertainty on Star from PSF Errors')
    py.semilogy(mag, psf_err_itself, 'k--', label='Uncertainty on Neighbors from PSF Errors')
    py.legend()
    py.xlabel('K Magnitude')
    py.ylabel('Astrometric Errors (mas)')
    py.ylim(0.01, 1)
    py.savefig('/u/jlu/doc/papers/proceed2010spie/fritz_figure14_myversion.png')
