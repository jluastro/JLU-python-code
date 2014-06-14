import asciidata, shutil, os
from pylab import *
import numpy as np
import scipy
import scipy.stats
from gcwork import starset
from gcwork import young
from gcwork import starTables
import pdb

def confusionThreshold(starName1, starName2, root='./',
                       align='align/align_d_rms1000_t',
                       poly='polyfit_d/fit'):
    s = starset.StarSet(root + align)
    s.loadPolyfit(root + poly, arcsec=1, accel=0)

    names = s.getArray('name')
    id1 = names.index(starName1)
    id2 = names.index(starName2)

    star1 = s.stars[id1]
    star2 = s.stars[id2]

    years = np.array(star1.years)
    xfit1 = star1.fitXv.p + (star1.fitXv.v * (years - star1.fitXv.t0))
    yfit1 = star1.fitYv.p + (star1.fitYv.v * (years - star1.fitYv.t0))
    xfit2 = star2.fitXv.p + (star2.fitXv.v * (years - star2.fitXv.t0))
    yfit2 = star2.fitYv.p + (star2.fitYv.v * (years - star2.fitYv.t0))

    xdiff = xfit1 - xfit2
    ydiff = yfit1 - yfit2
    diff = hypot(xdiff, ydiff)

    for ee in range(len(years)):
        detected1 = (star1.e[ee].xpix > -999)
        detected2 = (star2.e[ee].xpix > -999)

        if (diff[ee] < 0.075):
            print '%8.3f   Close Approach: sep = %5.3f' % \
                  (years[ee], diff[ee])

            if ((detected1 == False) and (detected2 == False)):
                print '\tNeither source found... do nothing'
            if ((detected1 == False) and (detected2 == True)):
                print '\t%13s: Not found in this epoch' % (starName1)
                print '\t%13s: Remove point for this epoch' % (starName2)
            if ((detected2 == False) and (detected1 == True)):
                print '\t%13s: Not found in this epoch' % (starName2)
                print '\t%13s: Remove point for this epoch' % (starName1)
            if ((detected1 == True) and (detected2 == True)):
                print '\t   Found both sources... do nothing'
    
def plotStar(starName, rootDir='./', align='align/align_d_rms_1000_abs_t',
             poly='polyfit_d/fit', points='points_d/', radial=False):

    s = starset.StarSet(rootDir + align)
    s.loadPolyfit(rootDir + poly, accel=0, arcsec=0)
    #s.loadPolyfit(rootDir + poly, accel=1, arcsec=0)

    names = s.getArray('name')
    
    ii = names.index(starName)
    star = s.stars[ii]

    pointsTab = asciidata.open(rootDir + points + starName + '.points')
	
    time = pointsTab[0].tonumpy()
    x = pointsTab[1].tonumpy()
    y = pointsTab[2].tonumpy()
    xerr = pointsTab[3].tonumpy()
    yerr = pointsTab[4].tonumpy()

    fitx = star.fitXv
    fity = star.fitYv
    dt = time - fitx.t0
    fitLineX = fitx.p + (fitx.v * dt)
    fitSigX = np.sqrt( fitx.perr**2 + (dt * fitx.verr)**2 )

    fitLineY = fity.p + (fity.v * dt)
    fitSigY = np.sqrt( fity.perr**2 + (dt * fity.verr)**2 )

    if (radial == True):
        # Lets also do radial/tangential
        x0 = fitx.p
        y0 = fity.p
        vx = fitx.v
        vy = fity.v
        x0e = fitx.perr
        y0e = fity.perr
        vxe = fitx.verr
        vye = fity.verr
        
        r0 = np.sqrt(x0**2 + y0**2)

        vr = ((vx*x0) + (vy*y0)) / r0
        vt = ((vx*y0) - (vy*x0)) / r0
        vre =  (vxe*x0/r0)**2 + (vye*y0/r0)**2
        vre += (y0*x0e*vt/r0**2)**2 + (x0*y0e*vt/r0**2)**2
        vre =  np.sqrt(vre)
        vte =  (vxe*y0/r0)**2 + (vye*x0/r0)**2
        vte += (y0*x0e*vr/r0**2)**2 + (x0*y0e*vr/r0**2)**2
        vte =  np.sqrt(vte)

        r = ((x*x0) + (y*y0)) / r0
        t = ((x*y0) - (y*x0)) / r0
        rerr = (xerr*x0/r0)**2 + (yerr*y0/r0)**2
        rerr += (y0*x0e*t/r0**2)**2 + (x0*y0e*t/r0**2)**2
        rerr =  np.sqrt(rerr)
        terr =  (xerr*y0/r0)**2 + (yerr*x0/r0)**2
        terr += (y0*x0e*r/r0**2)**2 + (x0*y0e*r/r0**2)**2
        terr =  np.sqrt(terr)

        fitLineR = ((fitLineX*x0) + (fitLineY*y0)) / r0
        fitLineT = ((fitLineX*y0) - (fitLineY*x0)) / r0
        fitSigR = ((fitSigX*x0) + (fitSigY*y0)) / r0
        fitSigT = ((fitSigX*y0) - (fitSigY*x0)) / r0

        diffR = r - fitLineR
        diffT = t - fitLineT
        sigR = diffR / rerr
        sigT = diffT / terr

        idxR = np.where(abs(sigR) > 4)
        idxT = np.where(abs(sigT) > 4)
        

    diffX = x - fitLineX
    diffY = y - fitLineY
    diff = np.hypot(diffX, diffY)
    rerr = np.sqrt((diffX*xerr)**2 + (diffY*yerr)**2) / diff
    sigX = diffX / xerr
    sigY = diffY / yerr
    sig = diff / rerr

    
    # Determine if there are points that are more than 5 sigma off
    idxX = np.where(abs(sigX) > 4)
    idxY = np.where(abs(sigY) > 4)
    idx = np.where(abs(sig) > 4)

    print 'Star:        ', starName
    print '\tX Chi^2 = %5.2f (%6.2f for %2d dof)' % \
          (fitx.chi2red, fitx.chi2, fitx.chi2/fitx.chi2red)
    print '\tY Chi^2 = %5.2f (%6.2f for %2d dof)' % \
          (fity.chi2red, fity.chi2, fity.chi2/fity.chi2red)
    print 'X  Outliers: ', time[idxX]
    print 'Y  Outliers: ', time[idxY]
    if (radial):
        print 'R  Outliers: ', time[idxX]
        print 'T  Outliers: ', time[idxY]
    print 'XY Outliers: ', time[idx]

        

    close(2)
    figure(2, figsize=(7, 8))
    clf()

    dateTicLoc = MultipleLocator(3)
    #dateTicRng = [2006, 2010]
    dateTicRng = [1995, 2010]

    maxErr = np.array([xerr, yerr]).max()
    resTicRng = [-3*maxErr, 3*maxErr]

    from matplotlib.ticker import FormatStrFormatter
    fmtX = FormatStrFormatter('%5i')
    fmtY = FormatStrFormatter('%6.2f')

    paxes = subplot(3, 2, 1)
    plot(time, fitLineX, 'b-')
    plot(time, fitLineX + fitSigX, 'b--')
    plot(time, fitLineX - fitSigX, 'b--')
    errorbar(time, x, yerr=xerr, fmt='k.')
    rng = axis()
    axis(dateTicRng + [rng[2], rng[3]])
    xlabel('Date (yrs)')
    ylabel('X (pix)')
    #paxes.get_xaxis().set_major_locator(dateTicLoc)
    paxes.xaxis.set_major_formatter(fmtX)
    paxes.get_xaxis().set_major_locator(dateTicLoc)
    paxes.yaxis.set_major_formatter(fmtY)
    
    paxes = subplot(3, 2, 2)
    plot(time, fitLineY, 'b-')
    plot(time, fitLineY + fitSigY, 'b--')
    plot(time, fitLineY - fitSigY, 'b--')
    errorbar(time, y, yerr=yerr, fmt='k.')
    rng = axis()
    axis(dateTicRng + [rng[2], rng[3]])
    xlabel('Date (yrs)')
    ylabel('Y (pix)')
    #paxes.get_xaxis().set_major_locator(dateTicLoc)
    paxes.xaxis.set_major_formatter(fmtX)
    paxes.get_xaxis().set_major_locator(dateTicLoc)
    paxes.yaxis.set_major_formatter(fmtY)
    
    paxes = subplot(3, 2, 3)
    plot(time, np.zeros(len(time)), 'b-')
    plot(time, fitSigX, 'b--')
    plot(time, -fitSigX, 'b--')
    errorbar(time, x - fitLineX, yerr=xerr, fmt='k.')
    axis(dateTicRng + resTicRng)
    xlabel('Date (yrs)')
    ylabel('X Residuals (pix)')
    paxes.get_xaxis().set_major_locator(dateTicLoc)

    paxes = subplot(3, 2, 4)
    plot(time, np.zeros(len(time)), 'b-')
    plot(time, fitSigY, 'b--')
    plot(time, -fitSigY, 'b--')
    errorbar(time, y - fitLineY, yerr=yerr, fmt='k.')
    axis(dateTicRng + resTicRng)
    xlabel('Date (yrs)')
    ylabel('Y Residuals (pix)')
    paxes.get_xaxis().set_major_locator(dateTicLoc)

    bins = np.arange(-7, 7, 1)
    subplot(3, 2, 5)
    (n, b, p) = hist(sigX, bins)
    setp(p, 'facecolor', 'k')
    axis([-5, 5, 0, 20])
    xlabel('X Residuals (sigma)')
    ylabel('Number of Epochs')

    subplot(3, 2, 6)
    (n, b, p) = hist(sigY, bins)
    axis([-5, 5, 0, 20])
    setp(p, 'facecolor', 'k')
    xlabel('Y Residuals (sigma)')
    ylabel('Number of Epochs')

    subplots_adjust(wspace=0.4, hspace=0.4, right=0.95, top=0.95)
    savefig(rootDir+'plots/plotStar_' + starName + '.png')
    #show()

    ##########
    #
    # Also plot radial/tangential
    #
    ##########
    if (radial == True):
        clf()

        dateTicLoc = MultipleLocator(3)
        
        maxErr = np.array([rerr, terr]).max()
        resTicRng = [-3*maxErr, 3*maxErr]
        
        from matplotlib.ticker import FormatStrFormatter
        fmtX = FormatStrFormatter('%5i')
        fmtY = FormatStrFormatter('%6.2f')
        
        paxes = subplot(3, 2, 1)
        plot(time, fitLineR, 'b-')
        plot(time, fitLineR + fitSigR, 'b--')
        plot(time, fitLineR - fitSigR, 'b--')
        errorbar(time, r, yerr=rerr, fmt='k.')
        rng = axis()
        axis(dateTicRng + [rng[2], rng[3]])
        xlabel('Date (yrs)')
        ylabel('R (pix)')
        paxes.xaxis.set_major_formatter(fmtX)
        paxes.get_xaxis().set_major_locator(dateTicLoc)
        paxes.yaxis.set_major_formatter(fmtY)
        
        paxes = subplot(3, 2, 2)
        plot(time, fitLineT, 'b-')
        plot(time, fitLineT + fitSigT, 'b--')
        plot(time, fitLineT - fitSigT, 'b--')
        errorbar(time, t, yerr=terr, fmt='k.')
        rng = axis()
        axis(dateTicRng + [rng[2], rng[3]])
        xlabel('Date (yrs)')
        ylabel('T (pix)')
        paxes.xaxis.set_major_formatter(fmtX)
        paxes.get_xaxis().set_major_locator(dateTicLoc)
        paxes.yaxis.set_major_formatter(fmtY)
        
        paxes = subplot(3, 2, 3)
        plot(time, np.zeros(len(time)), 'b-')
        plot(time, fitSigR, 'b--')
        plot(time, -fitSigR, 'b--')
        errorbar(time, r - fitLineR, yerr=rerr, fmt='k.')
        axis(dateTicRng + resTicRng)
        xlabel('Date (yrs)')
        ylabel('R Residuals (pix)')
        paxes.get_xaxis().set_major_locator(dateTicLoc)
        
        paxes = subplot(3, 2, 4)
        plot(time, np.zeros(len(time)), 'b-')
        plot(time, fitSigT, 'b--')
        plot(time, -fitSigT, 'b--')
        errorbar(time, t - fitLineT, yerr=terr, fmt='k.')
        axis(dateTicRng + resTicRng)
        xlabel('Date (yrs)')
        ylabel('T Residuals (pix)')
        paxes.get_xaxis().set_major_locator(dateTicLoc)
        
        bins = np.arange(-7, 7, 1)
        subplot(3, 2, 5)
        (n, b, p) = hist(sigR, bins)
        setp(p, 'facecolor', 'k')
        axis([-5, 5, 0, 20])
        xlabel('T Residuals (sigma)')
        ylabel('Number of Epochs')
        
        subplot(3, 2, 6)
        (n, b, p) = hist(sigT, bins)
        axis([-5, 5, 0, 20])
        setp(p, 'facecolor', 'k')
        xlabel('Y Residuals (sigma)')
        ylabel('Number of Epochs')
        
        subplots_adjust(wspace=0.4, hspace=0.4, right=0.95, top=0.95)
        savefig(rootDir+'plots/plotStarRadial_' + starName + '.png')

    
def sumAllStars(root='./', align='align/align_d_rms_1000_abs_t',
                poly='polyfit_d/fit', points='points_d/',
                youngOnly=False, trimOutliers=False, trimSigma=4,
                useAccFits=False, magCut=None, radCut=None):
    """Analyze the distribution of points relative to their best
    fit velocities. Optionally trim the largest outliers in each
    stars *.points file.  Optionally make a magnitude cut with
    magCut flag and/or a radius cut with radCut flag."""

    s = starset.StarSet(root + align)
    s.loadPolyfit(root + poly, accel=0, arcsec=0)
    s.loadPolyfit(root + poly, accel=1, arcsec=0)
    if (youngOnly):
        s.onlyYoungDisk()
    
    # Re-get the names array since we may have trimmed down to
    # only the young disk stars.
    names = s.getArray('name')

    # Check if we're doing any cutting
    if ((magCut != None) and (radCut != None)):
        mag = s.getArray('mag')
        x = s.getArray('x')
        y = s.getArray('y')
        r = hypot(x,y)
        idx = np.where((mag < magCut) & (r < radCut))[0]
        newstars = []
        for i in idx:
            newstars.append(s.stars[i])
        s.stars = newstars
        mag = mag[idx]
        x = x[idx]
        y = y[idx]
        r = r[idx]
        names = [names[ii] for ii in idx]
    elif ((magCut != None) and (radCut == None)):
        mag = s.getArray('mag')
        x = s.getArray('x')
        y = s.getArray('y')
        r = hypot(x,y)
        idx = np.where(mag < magCut)[0]
        newstars = []
        for i in idx:
            newstars.append(s.stars[i])
        s.stars = newstars
        mag = mag[idx]
        x = x[idx]
        y = y[idx]
        r = r[idx]
        names = [names[ii] for ii in idx]
    elif ((magCut == None) and (radCut != None)):
        mag = s.getArray('mag')
        x = s.getArray('x')
        y = s.getArray('y')
        r = hypot(x,y)
        idx = np.where(r < radCut)[0]
        newstars = []
        for i in idx:
            newstars.append(s.stars[i])
        s.stars = newstars
        mag = mag[idx]
        x = x[idx]
        y = y[idx]
        r = r[idx]
        names = [names[ii] for ii in idx]

    # Make some empty arrays to hold all our results.
    sigmaX = np.arange(0, dtype=float)
    sigmaY = np.arange(0, dtype=float)
    sigma  = np.arange(0, dtype=float)
    diffX_all = np.arange(0, dtype=float)
    diffY_all = np.arange(0, dtype=float)
    xerr_all = np.arange(0, dtype=float)
    yerr_all = np.arange(0, dtype=float)

    # Loop through all the stars and combine their residuals.
    for star in s.stars:
        starName = star.name
        
        pointsFile = root + points + starName + '.points'
        if os.path.exists(pointsFile + '.orig'):
            pointsTab = asciidata.open(pointsFile + '.orig')
        else:
            pointsTab = asciidata.open(pointsFile)

        # Observed Data
        t = pointsTab[0].tonumpy()
        x = pointsTab[1].tonumpy()
        y = pointsTab[2].tonumpy()
        xerr = pointsTab[3].tonumpy()
        yerr = pointsTab[4].tonumpy()

        # Best fit velocity model
        if (useAccFits == True):
            fitx = star.fitXa
            fity = star.fitYa
        else:
            fitx = star.fitXv
            fity = star.fitYv

        dt = t - fitx.t0
        fitLineX = fitx.p + (fitx.v * dt)
        fitSigX = sqrt( fitx.perr**2 + (dt * fitx.verr)**2 )

        fitLineY = fity.p + (fity.v * dt)
        fitSigY = sqrt( fity.perr**2 + (dt * fity.verr)**2 )

        if (useAccFits == True):
            fitLineX += (fitx.a * dt**2) / 2.0
            fitLineY += (fity.a * dt**2) / 2.0
            fitSigX = sqrt(fitSigX**2 + (dt**2 * fitx.aerr / 2.0)**2)
            fitSigY = sqrt(fitSigY**2 + (dt**2 * fity.aerr / 2.0)**2)

        # Residuals
        diffX = x - fitLineX
        diffY = y - fitLineY
        diff = hypot(diffX, diffY)
        rerr = sqrt((diffX*xerr)**2 + (diffY*yerr)**2) / diff
        sigX = diffX / xerr
        sigY = diffY / yerr
        sig = diff / rerr

        idxX = (np.where(abs(sigX) > trimSigma))[0]
        idxY = (np.where(abs(sigY) > trimSigma))[0]
        idx  = (np.where(abs(sig) > trimSigma))[0]


#         if (len(idxX) > 0):
#             print 'X %d sigma outliers for %13s: ' % \
#                   (trimSigma, starName), sigX[idxX], t[idxX]
#         if (len(idxY) > 0):
#             print 'Y %d sigma outliers for %13s: ' % \
#                   (trimSigma, starName), sigY[idxY], t[idxY]
#         if (len(idx) > 0):
#             print 'T %d sigma outliers for %13s: ' % \
#                   (trimSigma, starName), sig[idx], t[idx]

        if ((trimOutliers == True) and (len(idx) > 0)):
            if not os.path.exists(pointsFile + '.orig'):
                shutil.copyfile(pointsFile, pointsFile + '.orig')

            for ii in idx[::-1]:
                pointsTab.delete(ii)

            pointsTab.writeto(pointsFile)

        # Combine this stars information with all other stars.
        sigmaX = concatenate((sigmaX, sigX))
        sigmaY = concatenate((sigmaY, sigY))
        sigma = concatenate((sigma, sig))
        diffX_all = concatenate((diffX_all,diffX))
        diffY_all = concatenate((diffY_all,diffY))
        xerr_all = concatenate((xerr_all,xerr))
        yerr_all = concatenate((yerr_all,yerr))

    rmsDiffXY = (diffX_all.std() + diffY_all.std()) / 2.0 * 1000.0
    aveDiffR = np.sqrt(diffX_all**2 + diffY_all**2).mean()
    medDiffR = np.median(np.sqrt(diffX_all**2 + diffY_all**2))

    print diffX_all.mean(), diffY_all.mean()
    print diffX_all.std(), diffY_all.std()
    print rmsDiffXY, aveDiffR, medDiffR
    print np.median(xerr_all)

    # Residuals should have a gaussian probability distribution
    # with a mean of 0 and a sigma of 1. Overplot this to be sure.
    ggx = np.arange(-7, 7, 0.25)
    ggy = normpdf(ggx, 0, 1)

    print 'Mean   RMS residual: %5.2f sigma' % (sigma.mean())
    print 'Stddev RMS residual: %5.2f sigma' % (sigma.std())
    print 'Median RMS residual: %5.2f sigma' % (median(sigma))
    print
    print 'Mean X centroiding error: %5.4f mas (median %5.4f mas)' % \
        ((xerr_all*1000.0).mean(), np.median(xerr_all)*10**3)
    print 'Mean Y centroiding error: %5.4f mas (median %5.4f mas)' % \
        ((yerr_all*1000.0).mean(), np.median(yerr_all)*10**3)
    print 'Mean distance from velocity fit: %5.4f mas (median %5.4f mas)' % \
        (aveDiffR*10**3, medDiffR*10**3)

    ##########
    # Plot
    ##########
    bins = np.arange(-7, 7, 1.0)
    figure(1)
    clf()
    subplot(3, 1, 1)
    (nx, bx, px) = hist(sigmaX, bins)
    ggamp = ((sort(nx))[-2:]).sum() / (2.0 * ggy.max())
    plot(ggx, ggy*ggamp, 'k-')
    xlabel('X Residuals (sigma)')

    subplot(3, 1, 2)
    (ny, by, py) = hist(sigmaY, bins)
    ggamp = ((sort(ny))[-2:]).sum() / (2.0 * ggy.max())
    plot(ggx, ggy*ggamp, 'k-')
    xlabel('Y Residuals (sigma)')

    subplot(3, 1, 3)
    (ny, by, py) = hist(sigma, np.arange(0, 7, 0.5))
    xlabel('Total Residuals (sigma)')

    subplots_adjust(wspace=0.34, hspace=0.33, right=0.95, top=0.97)
    savefig(root+'plots/residualsDistribution.eps')
    savefig(root+'plots/residualsDistribution.png')

    # Put all residuals together in one histogram
    clf()
    sigmaA = []
    for ss in range(len(sigmaX)):
        sigmaA = np.concatenate([sigmaA,[sigmaX[ss]]])
        sigmaA = np.concatenate([sigmaA,[sigmaY[ss]]])
    (na, ba, pa) = hist(sigmaA, bins)
    ggamp = ((sort(na))[-2:]).sum() / (2.0 * ggy.max())
    plot(ggx, ggy*ggamp, 'k-')
    xlabel('Residuals (sigma)')
    savefig(root+'plots/residualsAll.eps')
    savefig(root+'plots/residualsAll.png')
    

def sigmaVsEpoch(root='./', align='align/align_d_rms_1000_abs_t',
                 poly='polyfit_d/fit', useAccFits=False):
    """
    Plot the average offset (in sigma) from the best fit
    velocity as a function of epoch.
    """
    s = starset.StarSet(root + align, relErr=1)
    s.loadPolyfit(root + poly, accel=0, arcsec=1)
    s.loadPolyfit(root + poly, accel=1, arcsec=1)

    numEpochs = len(s.stars[0].years)
    
    # Use only stars detected in all epochs
    epochCnt = s.getArray('velCnt')
    idx = (np.where(epochCnt == epochCnt.max()))[0]
    newStars = []
    for ii in idx:
        newStars.append(s.stars[ii])
    s.stars = newStars
    
    print 'Using %d out of %d stars detected in %d epochs' % \
          (len(newStars), len(epochCnt), epochCnt.max())

    # Make some empty arrays to hold all our results.
    sigmaX = np.zeros(numEpochs, float)
    sigmaY = np.zeros(numEpochs, float)
    sigma  = np.zeros(numEpochs, float)
    diffEpX = np.zeros(numEpochs, float)
    diffEpY = np.zeros(numEpochs, float)
    diffEp  = np.zeros(numEpochs, float)

    # Fetch the fit parameters for all the stars
    if (useAccFits == True):
        fitVarX = 'fitXa'
        fitVarY = 'fitYa'
    else:
        fitVarX = 'fitXv'
        fitVarY = 'fitYv'

    t0 = s.getArray(fitVarX + '.t0')
    x0 = s.getArray(fitVarX + '.p')
    vx = s.getArray(fitVarX + '.v')
    y0 = s.getArray(fitVarY + '.p')
    vy = s.getArray(fitVarY + '.v')

    x0e = s.getArray(fitVarX + '.perr')
    y0e = s.getArray(fitVarY + '.perr')
    vxe = s.getArray(fitVarX + '.verr')
    vye = s.getArray(fitVarY + '.verr')

    if (useAccFits == True):
        ax = s.getArray(fitVarX + 'a')
        ay = s.getArray(fitVarY + 'a')
        axe = s.getArray(fitVarX + 'aerr')
        aye = s.getArray(fitVarY + 'aerr')

    # Loop through all the epochs and determine average residuals
    for ee in range(numEpochs):
        # Observed data
        x = s.getArrayFromEpoch(ee, 'x')
        y = s.getArrayFromEpoch(ee, 'y')
        xerr_p = s.getArrayFromEpoch(ee, 'xerr_p')
        yerr_p = s.getArrayFromEpoch(ee, 'yerr_p')
        xerr_a = s.getArrayFromEpoch(ee, 'xerr_a')
        yerr_a = s.getArrayFromEpoch(ee, 'yerr_a')
        xerr = hypot(xerr_p, xerr_a)
        yerr = hypot(yerr_p, yerr_a)
        t = s.stars[0].years[ee]

        dt = t - t0
        fitLineX = x0 + (vx * dt)
        fitSigX = sqrt( x0e**2 + (dt * vxe)**2 )

        fitLineY = y0 + (vy * dt)
        fitSigY = sqrt( y0e**2 + (dt * vye)**2 )

        if (useAccFits == True):
            fitLineX += (ax * dt**2) / 2.0
            fitLineY += (ay * dt**2) / 2.0
            fitSigX = sqrt(fitSigX**2 + (dt**2 * axe / 2.0)**2)
            fitSigY = sqrt(fitSigY**2 + (dt**2 * aye / 2.0)**2)

        # Residuals
        diffX = x - fitLineX
        diffY = y - fitLineY
        diff = hypot(diffX, diffY)
        rerr = sqrt((diffX*xerr)**2 + (diffY*yerr)**2) / diff
        sigX = diffX / xerr
        sigY = diffY / yerr
        sig = diff / rerr

        print 'Epoch %d' % ee

        print '%8.5f +/- %8.5f   %8.5f +/- %8.5f  %8.5f (%5.1f sigma)' % \
              (x[0], xerr[0], fitLineX[0], fitSigX[0], diffX[0], sigX[0])
        print '%8.5f +/- %8.5f   %8.5f +/- %8.5f  %8.5f (%5.1f sigma)' % \
              (y[0], yerr[0], fitLineY[0], fitSigY[0], diffY[0], sigY[0])

        sigmaX[ee] = sigX.mean()
        sigmaY[ee] = sigY.mean()
        sigma[ee] = median(sig)

        diffEpX[ee] = diffX.mean()
        diffEpY[ee] = diffY.mean()
        diffEp[ee] = median(diff)

                        
    ##########
    # Plot
    ##########
    clf()
    years = s.stars[0].years
    plot(years, sigmaX, 'rx')
    plot(years, sigmaY, 'bx')
    plot(years, sigma, 'ko')
    xlabel('Epoch (years)')
    ylabel('Median Residual Error (sigma)')
    legend(('X', 'Y', 'Total'))

    savefig(root+'plots/residualsVsEpoch.eps')
    savefig(root+'plots/residualsVsEpoch.png')

    clf()
    plot(years, diffEpX*1000.0, 'rx')
    plot(years, diffEpY*1000.0, 'bx')
    plot(years, diffEp*1000.0, 'ko')
    xlabel('Epoch (years)')
    ylabel('Median Residual Error (mas)')
    legend(('X', 'Y', 'Total'))
    savefig(root+'plots/residualsVsEpochMAS.eps')
    savefig(root+'plots/residualsVsEpochMAS.png')


    # Print out epochs with higher than 3 sigma median residuals
    hdx = (np.where(sigma > 3))[0]
    print 'Epochs with median residuals > 3 sigma:'
    for hh in hdx:
        print '%8.3f  residual = %4.1f' % (s.stars[0].years[hh], sigma[hh])


