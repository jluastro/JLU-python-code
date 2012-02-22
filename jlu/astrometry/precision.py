import asciidata
import numpy as np
import pylab as py
from gcwork import starset

def plotAllAOvsMag(radius=4):
    rootDir = '/u/ghezgroup/data/gc/'
    rootDir = '/u/jlu/work/gc/proper_motion/align/09_06_01/lis/'
    epochs = ['06maylgs1', '06junlgs', '06jullgs', '07maylgs', '07auglgs',
              '08maylgs1', '08jullgs', '09maylgs1']
    legends = ['2006 May', '2006 Jun', '2006 Jul', '2007 May', '2007 Aug',
               '2008 May', '2008 Jul', '2009 May']

    # Define the magnitude bins.
    magStep = 1.0
    magBins = np.arange(10.0, 20.0, magStep)

    # Keep a record of the astrometry precision curves for each epoch
    errMag = np.zeros((len(epochs), len(magBins)), dtype=float)
    errMedian = np.zeros(len(epochs), dtype=float)

    # Loop through the epochs and calculate the curves.
    for e in range(len(epochs)):
        # Load up the starlist
        starlist = rootDir + 'mag' + epochs[e] + '_kp_rms.lis'
        lis = asciidata.open(starlist)

        # Assume this is NIRC2 data.
        scale = 0.00995
    
        name = lis[0]._data
        mag = lis[1].tonumpy()
        x = lis[3].tonumpy()
        y = lis[4].tonumpy()
        xerr = lis[5].tonumpy()
        yerr = lis[6].tonumpy()

        # Convert into arsec offset from field center
        # We determine the field center by assuming that stars
        # are detected all the way out the edge.
        xhalf = x.max() / 2.0
        yhalf = y.max() / 2.0
        x = (x - xhalf) * scale
        y = (y - yhalf) * scale
        xerr *= scale * 1000.0
        yerr *= scale * 1000.0

        r = np.hypot(x, y)
        err = (xerr + yerr) / 2.0

        ##########
        # Compute errors in magnitude bins
        ########## 
        for mm in range(len(magBins)):
            mMin = magBins[mm] - (magStep / 2.0)
            mMax = magBins[mm] + (magStep / 2.0)
            idx = (np.where((mag >= mMin) & (mag < mMax) & (r < radius)))[0]

            if (len(idx) > 0):
                errMag[e, mm] = np.median(err[idx])

        idx = (np.where((mag >= 11) & (mag <= 14) & (r < radius)))[0]
        errMedian[e] = np.median(err[idx])

    # Out Directory
    outdir = '/u/jlu/work/gc/astrometry/lgsao/plots/'

    # Record/Print output for each epoch
    _logfile = open(outdir + 'asterr_vs_mag_allao.log', 'w')
    for e in range(len(epochs)):
        message = '%10s - Median Precision for K=11-14: %5.2f' % \
            (legends[e], errMedian[e])
        print message
        _logfile.write(message + '\n')
    _logfile.close()

    # Plot linearly in astrometric error.
    py.clf()
    for e in range(len(epochs)):
        py.plot(magBins, errMag[e,:])

    py.legend(legends, loc='upper left')
    py.xlabel('K Magnitude for r < %4.1f"' % radius)
    py.ylabel('Pos. Uncertainty (mas)')
    py.title('Relative Astrometric Precision')
    py.savefig(outdir + 'asterr_vs_mag_allao_lin.png')

    # Plot log in astrometric error
    py.clf()
    for e in range(len(epochs)):
        py.semilogy(magBins, errMag[e,:])

    py.legend(legends, loc='upper left')
    py.xlabel('K Magnitude for r < %4.1f"' % radius)
    py.ylabel('Pos. Uncertainty (mas)')
    py.title('Relative Astrometric Precision')
    py.savefig(outdir + 'asterr_vs_mag_allao_log.png')


def accuracyFromResiduals(radiusCut=4):
    rootDir = '/u/jlu/work/gc/proper_motion/align/09_06_01/'
    alignRoot = 'align/align_d_rms_1000_abs_t'
    polyRoot = 'polyfit_d/fit'
    pointsRoot = 'points_d/'

    # Load starset
    s = starset.StarSet(rootDir + alignRoot)
    s.loadPolyfit(rootDir + polyRoot, accel=0, arcsec=0)

    # Keep only those stars in ALL epochs
    epochCnt = s.getArray('velCnt')
    idx = np.where(epochCnt == epochCnt.max())[0]
    newstars = []
    for i in idx:
        newstars.append(s.stars[i])
    s.stars = newstars

    mag = s.getArray('mag')
    x = s.getArray('x')
    y = s.getArray('y')
    r = np.hypot(x, y)

    # Trim on radius
    if (radiusCut != None):
        idx = np.where(r < radiusCut)[0]
        
        newstars = []
        for i in idx:
            newstars.append(s.stars[i])
        s.stars = newstars
        
        mag = mag[idx]
        x = x[idx]
        y = y[idx]
        r = r[idx]


    # Define some arrays we will keep.
    numStars = len(s.stars)

    medErrX = np.zeros(numStars, dtype=float)
    medErrY = np.zeros(numStars, dtype=float)
    medErr = np.zeros(numStars, dtype=float)
    medResX = np.zeros(numStars, dtype=float)
    medResY = np.zeros(numStars, dtype=float)
    medRes = np.zeros(numStars, dtype=float)

    for ss in range(numStars):
        star = s.stars[ss]
        starName = star.name
        
        pointsFile = rootDir + pointsRoot + starName + '.points'
        pointsTab = asciidata.open(pointsFile)

        # Observed Data
        t = pointsTab[0].tonumpy()
        x = pointsTab[1].tonumpy()
        y = pointsTab[2].tonumpy()
        xerr = pointsTab[3].tonumpy()
        yerr = pointsTab[4].tonumpy()
        
        # Best fit velocity model
        fitx = star.fitXv
        fity = star.fitYv

        dt = t - fitx.t0
        fitLineX = fitx.p + (fitx.v * dt)
        fitLineY = fity.p + (fity.v * dt)

        # Residuals
        diffx = x - fitLineX
        diffy = y - fitLineY
        diff = np.hypot(diffx, diffy)

        medErrX[ss] = np.median(xerr)
        medErrY[ss] = np.median(yerr)
        medErr[ss] = np.median((xerr + yerr) / 2.0)
        medResX[ss] = np.median(diffx)
        medResY[ss] = np.median(diffy)
        medRes[ss] = np.median(diff)

    # Convert into milliarcsec
    medErrX *= 10**3
    medErrY *= 10**3
    medErr *= 10**3
    medResX *= 10**3
    medResY *= 10**3
    medRes *= 10**3

    # Bins stars vs. magnitude
    magStep = 1.0
    magBins = np.arange(10.0, 20.0, magStep)

    medErrMag = np.zeros(len(magBins), dtype=float)
    medResMag = np.zeros(len(magBins), dtype=float)

    for mm in range(len(magBins)):
        mMin = magBins[mm] - (magStep / 2.0)
        mMax = magBins[mm] + (magStep / 2.0)
        idx = np.where((mag >= mMin) & (mag < mMax))[0]

        if (len(idx) > 0):
            medErrMag[mm] = np.median(medErr[idx])
            medResMag[mm] = np.median(medRes[idx])
    
    # Plot
    outdir = '/u/jlu/work/gc/astrometry/lgsao/plots/'

    py.clf()
    py.semilogy(magBins, medErrMag)
    py.semilogy(magBins, medResMag)
    py.legend(('Precision', 'Accuracy'))
    py.title('Keck AO Relative Astrometry')
    py.xlabel('K Magnitude for r < %4.1f"' % radiusCut)
    py.ylabel('Pos. Uncertainty (mas)')

    py.savefig(outdir + 'accuracy_from_residuals_log.png')

    py.clf()
    py.plot(magBins, medErrMag)
    py.plot(magBins, medResMag)
    py.legend(('Precision', 'Accuracy'))
    py.title('Keck AO Relative Astrometry')
    py.xlabel('K Magnitude for r < %4.1f"' % radiusCut)
    py.ylabel('Pos. Uncertainty (mas)')
    py.ylim(0, 2.0)

    py.savefig(outdir + 'accuracy_from_residuals_lin.png')

    foo = np.where((mag >= 11) & (mag <= 14))[0]
    print 'Median Precisions for K=11-14: ', np.median(medErr[foo])
    print 'Median Accuracy for K=11-14: ', np.median(medRes[foo])

    _logfile = open(outdir + 'accuracy_from_residuals.log', 'w')
    _logfile.write('Median Precisions for K=11-14: %5.2f\n' %
                   np.median(medErr[foo]))
    _logfile.write('Median Accuracy for K=11-14:   %5.2f\n' %
                   np.median(medRes[foo]))
    _logfile.close()
