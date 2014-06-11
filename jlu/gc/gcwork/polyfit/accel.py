import numpy as np
import pylab as py
from pylab import *
from matplotlib.font_manager import FontProperties
import asciidata
from gcwork import starset
from gcwork import objects
from gcwork import young
from scipy import stats
import pdb
#import nmpfit_sy2 as nmpfit_sy
import random

def histAccel(rootDir='./', align = 'align/align_d_rms_1000_abs_t',
              poly='polyfit_d/fit', points='points_d/',
              youngOnly=False):
    """
    Make a histogram of the accelerations in the radial/tangential,
    X/Y, and inline/perp. direction of motion. Also, the large outliers
    (> 3 sigma) are also printed out.

    Inputs:
    rootDir   = The root directory of an astrometry analysis
                (e.g. '07_05_18/' or './' if you are in the directory).
    align     = The align root file name (including the directory relative
                to rootDir). Make sure that polyfit was run on this align
		output.
    poly      = The polyfit root file name (including the directory relative
                to rootDir). This should be run on the same align as above.
    points    = The points directory.
    youngOnly = Only plot the known young stars. This does not include 
                the central arcsecond sources.

    Output:
    plots/polyfit_hist_accel.eps (and png)
    -- Contains the histograms of the accelerations.

    plots/polyfit_hist_accel_nepochs.eps (ang png)
    -- Contains a plot of number of stars vs. acceleration significance.
    """
    
    s = starset.StarSet(rootDir + align)
    s.loadPolyfit(rootDir + poly, accel=0, arcsec=1)
    s.loadPolyfit(rootDir + poly, accel=1, arcsec=1)

    names = s.getArray('name')

    # In mas/yr^2
    x0 = s.getArray('fitXa.p')
    y0 = s.getArray('fitYa.p')
    x0e = s.getArray('fitXa.perr')
    y0e = s.getArray('fitYa.perr')
    vx = s.getArray('fitXa.v')
    vy = s.getArray('fitYa.v')
    ax = s.getArray('fitXa.a')
    ay = s.getArray('fitYa.a')
    axe = s.getArray('fitXa.aerr')
    aye = s.getArray('fitYa.aerr')
    r2d = s.getArray('r2d')
    cnt = s.getArray('velCnt')

    if (youngOnly == True):
        yngNames = young.youngStarNames()

        idx = []

        for ii in range(len(names)):
            if (r2d[ii] > 0.8 and
                names[ii] in yngNames):# and
                #cnt[ii] >= 24):

                idx.append(ii)

        names = [names[i] for i in idx]
        x0 = x0[idx]
        y0 = y0[idx]
        x0e = x0e[idx]
        y0e = y0e[idx]
        vx = vx[idx]
        vy = vy[idx]
        ax = ax[idx]
        ay = ay[idx]
        axe = axe[idx]
        aye = aye[idx]
        r2d = r2d[idx]
        cnt = cnt[idx]
        print 'Found %d young stars' % len(names)

    pntcnt = np.zeros(len(names))
    for ii in range(len(names)):
        pntFileName = '%s%s.points' % (rootDir+points, names[ii])
        pntFile = open(pntFileName)
        data = pntFile.readlines()
        pntcnt[ii] = len(data)


    # Lets also do radial/tangential
    r = np.sqrt(x0**2 + y0**2)
    ar = ((ax*x0) + (ay*y0)) / r
    at = ((ax*y0) - (ay*x0)) / r
    are =  (axe*x0/r)**2 + (aye*y0/r)**2
    are += (y0*x0e*at/r**2)**2 + (x0*y0e*at/r**2)**2
    are =  np.sqrt(are)
    ate =  (axe*y0/r)**2 + (aye*x0/r)**2
    ate += (y0*x0e*ar/r**2)**2 + (x0*y0e*ar/r**2)**2
    ate =  np.sqrt(ate)

    # Lets also do parallael/perpendicular to velocity
    v = np.sqrt(vx**2 + vy**2)
    am = ((ax*vx) + (ay*vy)) / v
    an = ((ax*vy) - (ay*vx)) / v
    ame = np.sqrt((axe*vx)**2 + (aye*vy)**2) / v
    ane = np.sqrt((axe*vy)**2 + (aye*vx)**2) / v

    # Total acceleration
    atot = py.hypot(ax, ay)
    atoterr = np.sqrt((ax*axe)**2 + (ay*aye)**2) / atot

    def plotHist(val, pos, label):
        py.subplot(3, 2, pos)
        py.hist(val, bins=range(-8, 8, 1))
        py.axis([-8, 8, 0, (len(val)/3) + 2])
        #py.plot(r2d, val, 'k^')
        #for ii in range(len(r2d)):
        #   py.text(r2d[ii], val[ii], names[ii])
        #py.axis([0.5, 4.0, -7, 7])
        py.title(label)

        print 'Significant Values for %s' % (label)
        idx = (np.where(abs(val) > 3.0))[0]
        for i in idx:
            print '%15s   %5.2f sigma    %2d epochs' % \
                  (names[i], val[i], pntcnt[i])

    py.figure(3, figsize=(7.5,10))
    py.clf()
    plotHist(atot / atoterr, 1, 'Total')

    py.clf()

    # XY
    plotHist(ax / axe, 1, 'X')
    plotHist(ay / aye, 2, 'Y')
    
    # Radial/Tangential
    plotHist(ar / are, 3, 'Radial')
    plotHist(at / ate, 4, 'Tangential')

    # In line of Motion and out
    plotHist(an / ane, 5, 'Perpendic. to v')
    plotHist(am / ame, 6, 'Parallel to v')

    py.savefig('plots/polyfit_hist_accel.eps')
    py.savefig('plots/polyfit_hist_accel.png')


    # Analyze the non-central arcsec sources for Nepochs threshold
    idx = (np.where(r2d > 0.8))[0]

    py.figure(1)
    py.clf()
    py.subplot(2, 1, 1)
    py.plot(ar[idx] / are[idx], pntcnt[idx], 'k.')
    py.axis([-10, 10, 0, 30])
    py.xlabel('Radial Acc. Sig. (sigma)')
    py.ylabel('Number of Epochs Detected')

    py.subplot(2, 1, 2)
    py.plot(at[idx] / ate[idx], pntcnt[idx], 'k.')
    py.axis([-10, 10, 0, 30])
    py.xlabel('Tangent. Acc. Sig. (sigma)')
    py.ylabel('Number of Epochs Detected')

    py.savefig('plots/polyfit_hist_accel_nepochs.eps')
    py.savefig('plots/polyfit_hist_accel_nepochs.png')

def plotCompareYoung(align1, poly1, align2, poly2):
    """
    Compare the young stars accelerations for two different runs
    of align/polyfit.
    """
    s1 = starset.StarSet(align1)
    s1.loadPolyfit(poly1, accel=0, arcsec=1)
    s1.loadPolyfit(poly1, accel=1, arcsec=1)

    s2 = starset.StarSet(align2)
    s2.loadPolyfit(poly2, accel=0, arcsec=1)
    s2.loadPolyfit(poly2, accel=1, arcsec=1)

    names1 = s1.getArray('name')
    names2 = s2.getArray('name')
    yngNames = young.youngStarNames()

    # We need to filter starlists to just the young'uns
    idx1 = []
    for ii in range(len(s1.stars)):
        if (s1.stars[ii].name in yngNames and
            s1.stars[ii].r2d > 0.8 and 
            s1.stars[ii].velCnt >= 31):
            idx1.append(ii)
    stars1 = [s1.stars[i] for i in idx1]
    s1.stars = stars1
    names1 = s1.getArray('name')
            
    idx2 = []
    for ii in range(len(s2.stars)):
        if (s2.stars[ii].name in yngNames and
            s2.stars[ii].r2d > 0.8 and 
            s2.stars[ii].velCnt >= 29):
            idx2.append(ii)
    stars2 = [s2.stars[i] for i in idx2]
    s2.stars = stars2

    names = s2.getArray('name')
    print names
    
    # Make arrays of the accelerations
    x0_1 = s1.getArray('fitXa.p')
    y0_1 = s1.getArray('fitYa.p')
    vx_1 = s1.getArray('fitXa.v')
    vy_1 = s1.getArray('fitYa.v')
    ax_1 = s1.getArray('fitXa.a') * 1000.0
    ay_1 = s1.getArray('fitYa.a') * 1000.0
    axe_1 = s1.getArray('fitXa.aerr') * 1000.0
    aye_1 = s1.getArray('fitYa.aerr') * 1000.0
    chi2x_1 = s1.getArray('fitXa.chi2red')
    chi2y_1 = s1.getArray('fitYa.chi2red')

    x0_2 = s2.getArray('fitXa.p')
    y0_2 = s2.getArray('fitYa.p')
    vx_2 = s2.getArray('fitXa.v')
    vy_2 = s2.getArray('fitYa.v')
    ax_2 = s2.getArray('fitXa.a') * 1000.0
    ay_2 = s2.getArray('fitYa.a') * 1000.0
    axe_2 = s2.getArray('fitXa.aerr') * 1000.0
    aye_2 = s2.getArray('fitYa.aerr') * 1000.0
    chi2x_2 = s2.getArray('fitXa.chi2red')
    chi2y_2 = s2.getArray('fitYa.chi2red')

    py.figure(1, figsize=(4,4))
    py.clf()

#     py.errorbar(ax_1, ax_2, xerr=axe_1, yerr=axe_2, fmt='k^')
#     py.plot([-1, 1], [-1, 1], 'k--')
#     py.axis('equal')

    # Determine how different the values are from each other
#     diffX = ax_1 - ax_2
#     diffY = ay_1 - ay_2
#     diffXe = np.sqrt(axe_1**2 + axe_2**2)
#     diffYe = np.sqrt(aye_1**2 + aye_2**2)

#     sigX = diffX / diffXe
#     sigY = diffY / diffYe

    py.clf()
    py.subplot(2, 1, 1)
    n1x, bins1x, patches1x = py.hist(chi2x_1, bins=np.arange(0,6,0.2))
    py.setp(patches1x, 'facecolor', 'r', 'alpha', 0.50)
    n1y, bins1y, patches1y = py.hist(chi2y_1, bins=np.arange(0,6,0.2))
    py.setp(patches1y, 'facecolor', 'b', 'alpha', 0.50)

    ysubplot(2, 1, 2)
    n2x, bins2x, patches2x = py.hist(chi2x_2, bins=np.arange(0,6,0.2))
    py.setp(patches2x, 'facecolor', 'r', 'alpha', 0.50)
    n2y, bins2y, patches2y = py.hist(chi2y_2, bins=np.arange(0,6,0.2))
    py.setp(patches2y, 'facecolor', 'b', 'alpha', 0.50)
    py.savefig('poly_compare_chi2.png')


    # Compare the number of points for each star
    cnt1 = s1.getArray('velCnt')
    cnt2 = s2.getArray('velCnt')

#     py.clf()
#     n1, bins1, patches1 = py.hist(cnt1, bins=np.arange(0,32,1))
#     n2, bins2, patches2 = py.hist(cnt2, bins=np.arange(0,32,1))
#     py.setp(patches1, 'facecolor', 'r', 'alpha', 0.50)
#     py.setp(patches2, 'facecolor', 'b', 'alpha', 0.50)


def plotLimitProps():
    """Make a table of the observed properties of those stars that
    have significant orbital acceleration limits.

    Currently hard coded to 06_02_14 and efit results.
    """
    # Load GC constants
    cc = objects.Constants()

    root = '/u/jlu/work/gc/proper_motion/align/06_02_14/'
    efitResults = root + 'efit/results/outer3'
    
    # Load up positional information from align. 
    s = starset.StarSet(root + 'align/align_all1000_t')
    s.loadPolyfit(root + 'polyfit/fit', arcsec=1)
    s.loadPolyfit(root + 'polyfit/fit', arcsec=1, accel=1)
    s.loadEfitResults(efitResults, trimStars=1)
    
    # Calculate acceleration limit from the 2D position
    x = s.getArray('fitXalign.p')
    y = s.getArray('fitYalign.p')
    xerr = s.getArray('fitXalign.perr')
    yerr = s.getArray('fitYalign.perr')
    r2d = py.hypot(x, y)

    # Convert into cm
    r2d *= cc.dist * cc.cm_in_au

    # acc1 in cm/s^2
    a2d = cc.G * cc.mass * cc.msun / r2d**2
    # acc1 in km/s/yr
    a2d *= cc.sec_in_yr / 1e5

    a1 = s.getArray('efit.at_lo')
    a2 = s.getArray('efit.at_hi')
    alim = [max(abs(a1[ii]), abs(a2[ii])) for ii in range(len(a1))]
    name = s.getArray('name')

    # The properties of interest
    cnt = s.getArray('velCnt')
    mag = s.getArray('mag')
    vx = s.getArray('fitXv.v') * 1000.0
    vxerr = s.getArray('fitXv.verr') * 1000.0
    vy = s.getArray('fitYv.v') * 1000.0
    vyerr = s.getArray('fitYv.verr') * 1000.0
    v2d = py.hypot(vx, vy)
    v2derr = np.sqrt((vx*vxerr)**2 + (vy*vyerr)**2) / v2d
    x = s.getArray('x')
    y = s.getArray('y')

    # Lets also read in definities only (no 05jullgs) as comparison
    root2 = root + '../06_02_14a/'
    s2 = starset.StarSet(root2 + 'align/align_d_rms_t')
    names2 = s2.getArray('name')

    idx = []
    stars2 = []
    for i in range(len(alim)):
        if (alim[i] < a2d[i]):
            idx.append(i)

        # Find this star in the new analysis
        new = names2.index(name[i])
        stars2.append(s2.stars[new])

    s2.stars = stars2
    cnt2 = s2.getArray('velCnt')
    mag2 = s2.getArray('mag')

        
    # Clear plot region
    py.figure(2, figsize=(7,6))
    py.clf()

    # Make a histogram of the number of epochs
    py.subplot(2, 2, 1)
    n, bins, patches1 = py.hist(cnt, bins=range(0, 32, 1))
    py.setp(patches1, 'facecolor', 'r')
    n, bins, patches2 = py.hist(cnt[idx], bins=range(0, 32, 1))
    py.setp(patches2, 'facecolor', 'b')
    py.legend((patches1[0], patches2[0]),
           ('All Young Stars', 'Young Stars With Acc. Limits'),
           loc='upper left', prop=FontProperties(size=8))
    py.xlabel('Number of Epochs')

    py.subplot(2, 2, 2)
    n, bins, patches = py.hist(mag, bins=range(8, 16, 1))
    py.setp(patches, 'facecolor', 'r')
    n, bins, patches = py.hist(mag[idx], bins=range(8, 16, 1))
    py.setp(patches, 'facecolor', 'b')
    py.xlabel('Magnitude')

    py.subplot(2, 2, 3)
    n, bins, patches1 = py.hist(v2d, bins=range(0,16,1))
    py.setp(patches1, 'facecolor', 'r')
    n, bins, patches2 = py.hist(v2d[idx], bins=range(0,16,1))
    py.setp(patches1, 'facecolor', 'r')
    py.xlabel('Velocity (mas/yr)')

    py.subplot(2, 2, 4)
    n, bins, patches1 = py.hist(v2derr, bins=np.arange(0,0.25,0.02))
    py.setp(patches1, 'facecolor', 'r')
    n, bins, patches2 = py.hist(v2derr[idx], bins=np.arange(0,0.25,0.02))
    py.setp(patches1, 'facecolor', 'r')
    py.xlabel('Velocity Error (mas/yr)')

    py.savefig(root + 'plots/acc_limits_properties.ps')

    py.clf()
    py.plot(x, y, 'k^')
    py.plot(x[idx], y[idx], 'r^')
    for i in range(len(x)):
        py.text(x[i], y[i], name[i])
    py.axis('equal')
    py.axis([4, -4, -4, 4])

    # Do this to see individual stars
#     py.plot(cnt, mag, 'r^')
#     py.plot(cnt[idx], mag[idx], 'r^')

#     for i in range(len(mag)):
#         py.text(cnt[i], mag[i], name[i])


    # Make a histogram of the number of epochs
#     clf()
#     subplot(2, 1, 1)
#     n, bins, patches1 = hist(cnt2, bins=range(0, 32, 1))
#     setp(patches1, 'facecolor', 'r')
#     n, bins, patches2 = hist(cnt2[idx], bins=range(0, 32, 1))
#     setp(patches2, 'facecolor', 'b')
#     legend((patches1[0], patches2[0]),
#            ('All Young Stars', 'Young Stars With Acc. Limits'),
#            loc='best', prop=FontProperties('smaller'))
#     xlabel('Number of Epochs')

#     subplot(2, 1, 2)
#     n, bins, patches = hist(mag2, bins=range(8, 16, 1))
#     setp(patches, 'facecolor', 'r')
#     n, bins, patches = hist(mag2[idx], bins=range(8, 16, 1))
#     setp(patches, 'facecolor', 'b')
#     xlabel('Magnitude')

#     savefig(root + 'plots/acc_limits_properties_new.ps')



def plotLimits(rootDir='./', align='align/align_d_rms1000_t',
               poly='polyfit_d_points/fit', points='points_d/',
               youngOnly=False, sigma=3):
    """
    Find the 4 sigma radial acceleration limits for each star.
    """
    # Load GC constants
    cc = objects.Constants()

    # Load up positional information from align. 
    s = starset.StarSet(rootDir + align)
    s.loadPolyfit(rootDir + poly, arcsec=1)
    s.loadPolyfit(rootDir + poly, arcsec=1, accel=1)

    names = s.getArray('name')

    # In mas/yr^2
    x = s.getArray('x')
    y = s.getArray('y')
    vx = s.getArray('fitXa.v')
    vy = s.getArray('fitYa.v')
    ax = s.getArray('fitXa.a') * 1000.0
    ay = s.getArray('fitYa.a') * 1000.0
    axe = s.getArray('fitXa.aerr') * 1000.0
    aye = s.getArray('fitYa.aerr') * 1000.0
    r2d = s.getArray('r2d')
    cnt = s.getArray('velCnt')

    if (youngOnly == True):
        yngNames = young.youngStarNames()

        idx = []

        for ii in range(len(names)):
            if (r2d[ii] > 0.8 and
                names[ii] in yngNames and
                cnt[ii] >= 24):
                idx.append(ii)

        names = [names[i] for i in idx]
        x = x[idx]
        y = y[idx]
        vx = vx[idx]
        vy = vy[idx]
        ax = ax[idx]
        ay = ay[idx]
        axe = axe[idx]
        aye = aye[idx]
        r2d = r2d[idx]
        print 'Found %d young stars' % len(names)

    # Lets do radial/tangential
    r = np.sqrt(x**2 + y**2)
    if ('Radial' in poly):
        at = ax
        ar = ay
        ate = axe
        are = aye
    else:
        ar = ((ax*x) + (ay*y)) / r
        at = ((ax*y) - (ay*x)) / r
        are = np.sqrt((axe*x)**2 + (aye*y)**2) / r
        ate = np.sqrt((axe*y)**2 + (aye*x)**2) / r

    # Total acceleration
    atot = py.hypot(ax, ay)
    atoterr = np.sqrt((ax*axe)**2 + (ay*aye)**2) / atot

    accLim = ar - (sigma * are)

    # Calculate the acceleration limit set by the projected radius
    # Convert into cm
    r2d = r * cc.dist * cc.cm_in_au
    # acc1 in cm/s^2
    a2d = -cc.G * cc.mass * cc.msun / r2d**2
    # acc1 in km/s/yr
    a2d *= cc.sec_in_yr / 1.0e5
    a2d *= 1000.0 / cc.asy_to_kms
    accLimR2d = a2d

    _f = open(rootDir + 'tables/accel_limits.txt', 'w')

    _f.write('###   Radial Acceleration Limits for Young Stars   ###\n')
    _f.write('%13s  %7s - (%d * %5s) =  %9s     %21s  Constrained?\n' % \
             ('Name', 'a_rad', sigma, 'aerr', 'a_obs_lim', 'a_proj_lim'))
    
    fmt = '%13s  %7.3f - (%d * %5.3f) =    %7.3f  '
    fmt += 'vs %10.3f (mas/yr^2)  %s'
    fmt2 = fmt + '\n'

    for i in range(len(ar)):
        constrained = ''
        if (accLim[i] > accLimR2d[i]):
            constrained = '**'
            
        print fmt % (names[i], ar[i], sigma, are[i], accLim[i],
                     accLimR2d[i], constrained)
        _f.write(fmt2 % (names[i], ar[i], sigma, are[i], accLim[i],
                         accLimR2d[i], constrained))

    _f.close()



def compareVelocity(rootDir='./', align = 'align/align_d_rms1000t',
                    poly='polyfit_d/fit', points='points_d/'):
    """
    Compare velocities from the 2D and 3D polynomial fits.

    rootDir -- An align root directory such as '06_14_02/' (def='./')
    align -- Align root name (def = 'align/align_all1000_t')
    poly -- Polyfit root name (def = 'polyfit/fit')
    points -- Points directory (def = 'points/')
    """
    
    s = starset.StarSet(rootDir + align)
    s.loadPolyfit(rootDir + poly, accel=0, arcsec=1)
    s.loadPolyfit(rootDir + poly, accel=1, arcsec=1)

    names = s.getArray('name')
    x = s.getArray('x')
    y = s.getArray('y')
    r = hypot(x, y)

    vx_vel = s.getArray('fitXv.v')
    vy_vel = s.getArray('fitYv.v')
    vxe_vel = s.getArray('fitXv.verr')
    vye_vel = s.getArray('fitYv.verr')

    vx_acc = s.getArray('fitXa.v')
    vy_acc = s.getArray('fitYa.v')
    vxe_acc = s.getArray('fitXa.verr')
    vye_acc = s.getArray('fitYa.verr')
    
    # Calculate the residuals
    diffx = vx_vel - vx_acc
    diffxErr = np.sqrt(vxe_vel**2 + vxe_acc**2)
    
    diffy = vy_vel - vy_acc
    diffyErr = np.sqrt(vye_vel**2 + vye_acc**2)
    
    diff = py.hypot(diffx, diffy)
    diffErr = np.sqrt((diffx*diffxErr)**2 + (diffy*diffyErr)**2) / diff

    yngNames = young.youngStarNames()
    
    idx = (np.where((r > 0.8) & ((diff/diffErr) > 2.0)))[0]

    print '** Stars with large velocity discrepencies: **'
    print '%15s  %5s  %5s  %5s  %3s' % ('Name', 'Sigma', 'X', 'Y', 'YNG?')
    for i in idx:
        try:
            foo = yngNames.index(names[i])
            yngString = 'yng'
        except ValueError, e:
            yngString = ''
            
        print '%15s  %5.1f  %5.2f  %5.2f  %3s' % \
              (names[i], diff[i]/diffErr[i], x[i], y[i], yngString)


    # Plot X and Y seperatly
    py.clf()
    py.subplot(2, 1, 1)
    py.errorbar(vx_vel, vx_acc, xerr=vxe_vel, yerr=vxe_acc, fmt='k.')
    rng = py.axis()
    py.plot(rng[0:2], rng[0:2], 'b--')

    py.subplot(2, 1, 2)
    py.errorbar(vy_vel, vy_acc, xerr=vye_vel, yerr=vye_acc, fmt='k.')
    rng = py.axis()
    py.plot(rng[0:2], rng[0:2], 'b--')
    
    
def highSigSrcs(radiusCut, sigmaCut, rootDir='./', poly='polyfit_d/fit',
                save=False, verbose=True):
    
    """
    Make a list of all sources with significant accelerations.
    Assumes a plate scale of 9.94 mas/pixel.

    Inputs:
    radiusCut - the largest radius to include (arcsec)
    sigmaCut - the lowest a/a_err to include
    save - Save to tables/accelHighSigSrcs.txt (def=False)
    verbose - Print to screen (def=True)

    Outputs:
    srcNames - returns a list of stars with significant radial
    accelerations based on the passed in criteria.
    """
    # Load up the accelPolar file
    fitFile = rootDir + poly + '.accelPolar'
    scale = 0.00995  # arcsec/pixel

    tab = asciidata.open(fitFile)

    name = tab[0]._data
    radius = tab[2].tonumarray() * scale
    acc = tab[10].tonumarray() * scale
    accErr = tab[12].tonumarray() * scale
    sigma = acc / accErr

    # Make cuts in radius and sigma
    idx = ( where((radius < radiusCut) & (sigma > sigmaCut)) )[0]

    # Sort
    rdx = sigma[idx].argsort()
    idx = idx[rdx[::-1]]

    # Print out to the screen
    if (verbose == True):
        print '** Found %d significantly accelerating sources **' % len(idx)
        print ''
        print '%-15s  %8s  %10s  %8s' % \
              ('Name', 'Radius', 'Accel', 'Signif.')
        print '%-15s  %8s  %10s  %8s' % \
              ('', '(arcsec)', '(mas/yr^2)', '(sigma)')
        print '%-15s  %8s  %10s  %8s' % \
              ('---------------', '--------', '----------', '--------')

        for ii in idx:
            print '%-15s  %8.3f  %10.5f  %8.1f' % \
                  (name[ii], radius[ii], acc[ii], sigma[ii])


    # Save data to an output file
    if (save == True):
        outFile = 'tables/accelHighSigSrcs.txt'
        _out = open(outFile, 'w')

        _out.write('# Python: gcwork.polyfit.accel.highSigSrcs')
        _out.write('(%5.2f, %5.2f)\n' % \
                   (radiusCut, sigmaCut))
        _out.write('#\n')
        _out.write('%-15s  %8s  %10s  %8s\n' % \
                   ('# Name', 'Radius', 'Accel', 'Signif.'))
        _out.write('%-15s  %8s  %10s  %8s\n' % \
                   ('#', '(arcsec)', '(mas/yr^2)', '(sigma)'))
        _out.write('%-15s  %8s  %10s  %8s\n' % \
                   ('#--------------', '--------', '----------', '--------'))

        for ii in idx:
            _out.write('%-15s  %8.3f  %10.5f  %8.1f\n' % \
                       (name[ii], radius[ii], acc[ii], sigma[ii]))

        _out.close()

    # Return the list of significantly accelerating sources.
    return [name[ii] for ii in idx]
    
    
def velVsAcc():
    """
    Plot v/v_circular vs. a/a_bound.
    """
    # Load up the accelPolar file
    fitFile = rootDir + poly + '.accelPolar'
    scale = 0.00995  # arcsec/pixel

    tab = asciidata.open(fitFile)

    name = tab[0]._data
    radius = tab[2].tonumarray() * scale

    velPhi = tab[5].tonumarray() * scale
    velRad = tab[6].tonumarray() * scale
    velPhiErr = tab[7].tonumarray() * scale
    velRadErr = tab[8].tonumarray() * scale
    acc = tab[10].tonumarray() * scale
    accErr = tab[12].tonumarray() * scale

    # Need to get the line-of-sight velocity from Paumard et al.
    

    vel = np.sqrt(velPhi**2 + velRad**2)
    velErr = np.sqrt((velPhi*velPhiErr)**2 + (velRad*velRadErr)**2) / vel

    # Determine the circular velocity 

def plotStar(star, root='./', align = 'align/align_d_rms_1000_abs_t',
             poly='polyfit_d/fit', points='points_d/'):

    pointsFile = root + points + star + '.points'
    fitFile = root + poly + '.' + star + '.pfit'

    _tabPoints = asciidata.open(pointsFile)
    date = _tabPoints[0].tonumpy()
    x = _tabPoints[1].tonumpy() * -1.0
    y = _tabPoints[2].tonumpy()
    xerr = _tabPoints[3].tonumpy()
    yerr = _tabPoints[4].tonumpy()

    _fitPoints = asciidata.open(fitFile)
    date_f = _fitPoints[0].tonumpy()
    x_f = _fitPoints[1].tonumpy() * -1.0
    y_f = _fitPoints[2].tonumpy()
    xerr_f = _fitPoints[3].tonumpy()
    yerr_f = _fitPoints[4].tonumpy()

    # Find range of plot
    halfRange = max([ abs(x.max() - x.min()), abs(y.max() - y.min()) ]) / 2.0

    padd = 0.02
    xmax = x.min() + ((x.max() - x.min())/2.0) + halfRange + padd
    ymax = y.min() + ((y.max() - y.min())/2.0) + halfRange + padd
    xmin = x.min() + ((x.max() - x.min())/2.0) - halfRange - 2*padd
    ymin = y.min() + ((y.max() - y.min())/2.0) - halfRange - padd

    legend_items = ['1995', '1996', '1997', '1998', '1999',
                    '2000', '2001', '2002', '2003', '2004',
                    '2005', '2006', '2007']
    legend_colors = ['olive', 'brown', 'purple', 'darkslateblue', 'mediumblue',
                     'steelblue', 'teal', 'green', 'greenyellow', 'gold',
                     'darkorange', 'orangered', 'red']

    # Assign color to each epoch
    year = [str(int(np.floor( d ))) for d in date]
    color_arr = []


    py.close(2)
    py.figure(2, figsize=(6.2,5.4))
    py.clf()
    py.plot(x_f, y_f, 'k-')
    py.plot(x_f + xerr_f, y_f + yerr_f, 'k--')
    py.plot(x_f - xerr_f, y_f - yerr_f, 'k--')
    
    for i in range(len(year)):
        # find out which year
        try:
            idx = legend_items.index( year[i] )
            color_arr.append( legend_colors[idx] )
        except ValueError:
            color_arr.append('black')

        py.errorbar([x[i]], [y[i]], fmt='ko', xerr=xerr[i], yerr=yerr[i],
                    ms=5,
                    color=color_arr[i], mec=color_arr[i], mfc=color_arr[i])

    # Set axis ranges
    py.title(star)
    ax = py.axis([xmax, xmin, ymin, ymax])
    py.gca().set_aspect('equal')
    py.xlabel('R.A. Offset from Sgr A* (arcsec)')
    py.ylabel('Dec. Offset from Sgr A* (arcsec)')

    # Draw legend
    py.legend(legend_items, numpoints=1)
    ltext = py.gca().get_legend().get_texts()
    lgdLines = py.gca().get_legend().get_lines()
    for j in range(len(ltext)):
        py.setp(ltext[j], color=legend_colors[j])
        lgdLines[j].set_marker('o')
        lgdLines[j].set_mec(legend_colors[j])
        lgdLines[j].set_mfc(legend_colors[j])
        lgdLines[j].set_ms(5)


    # Retrieve information on the acceleration fit from the .accelFormal file
    fitFile = root + poly + '.accelFormal'
    _fit = open(fitFile, 'r')

    for line in _fit:
        entries = line.split()

        if (entries[0] == star):
            chiSqX = float( entries[7] )
            Qx = float( entries[8] )
            chiSqY = float( entries[15] )
            Qy = float( entries[16] )
            break


    _fit.close()

def minDistance(xfit1,yfit1,xfit2,yfit2,t0,trange,pause=0):
    """
    From the coefficients of the fits to two stars, find the minimum
    distance between them by using the acceleration fits. 
    xfit1 = [x0, vx, ax]
    yfit1 = [y0, vy, ay]
    
    HISTORY: 2010-01-13 - T. Do
    """
    time = np.arange(trange[0]-t0,trange[1]-t0+1,0.1)
    xpos1 = poly2(xfit1, time)
    ypos1 = poly2(yfit1, time)
    xpos2 = poly2(xfit2, time)
    ypos2 = poly2(yfit2, time)

    distance = np.sqrt((xpos1-xpos2)**2 + (ypos1 - ypos2)**2)
    if pause:
        clf()
        plot(xpos1,ypos1)
        plot(xpos2,ypos2)
        #print time
        show()
        #pdb.set_trace()
    return np.amin(distance)
    
def nonPhysical(rootDir='./', align='align/align_d_rms_1000_abs_t',
                poly='polyfit_d/fit', points='points_d/', sigma=5,
                plotChiSq=False, plotPrefix=''):
    """
    Print out a list of stars that have non-physical accelerations.

    Return: returns a list of star names that are unphysical 
    """
    # Load GC constants
    cc = objects.Constants()

    # Load up positional information from align. 
    s = starset.StarSet(rootDir + align)
    s.loadPolyfit(rootDir + poly, arcsec=1)
    s.loadPolyfit(rootDir + poly, arcsec=1, accel=1)

    names = s.getArray('name')
    mag = s.getArray('mag')*1.0
    print len(mag)
    print '%7.4f mag' % mag[0]
    # In mas/yr^2
    x = s.getArray('x')
    y = s.getArray('y')


#    pdb.set_trace()
    vx = s.getArray('fitXa.v')
    vy = s.getArray('fitYa.v')
    ax = s.getArray('fitXa.a') * 1000.0
    ay = s.getArray('fitYa.a') * 1000.0
    axe = s.getArray('fitXa.aerr') * 1000.0
    aye = s.getArray('fitYa.aerr') * 1000.0
    r2d = s.getArray('r2d')
    cnt = s.getArray('velCnt')

#    chi2x = s.getArray('fitXa.chi2red')
#    chi2y = s.getArray('fitYa.chi2red')

    chi2x = s.getArray('fitXa.chi2')
    chi2y = s.getArray('fitYa.chi2')

    chi2xv = s.getArray('fitXv.chi2')
    chi2yv = s.getArray('fitYv.chi2')
    
    nEpochs = s.getArray('velCnt')
    
    # T0 for each of the acceleration fits
    epoch = s.getArray('fitXa.t0')

    # All epochs
    allEpochs = np.array(s.stars[0].years)
    
    # Lets do radial/tangential
    r = np.sqrt(x**2 + y**2)
    if ('Radial' in poly):
        at = ax
        ar = ay
        ate = axe
        are = aye
    else:
        ar = ((ax*x) + (ay*y)) / r
        at = ((ax*y) - (ay*x)) / r
        are = np.sqrt((axe*x)**2 + (aye*y)**2) / r
        ate = np.sqrt((axe*y)**2 + (aye*x)**2) / r

    # Total acceleration
    atot = py.hypot(ax, ay)
    atoterr = np.sqrt((ax*axe)**2 + (ay*aye)**2) / atot

    # Calculate the acceleration limit set by the projected radius
    # Convert into cm
    r2d = r * cc.dist * cc.cm_in_au
    rarc =np.arange(0.01,10.0,0.01)
    rsim = rarc * cc.dist * cc.cm_in_au
    
    # acc1 in cm/s^2
    a2d = -cc.G * cc.mass * cc.msun / r2d**2
    a2dsim = -cc.G * cc.mass * cc.msun / rsim**2
    # acc1 in km/s/yr
    
    a2d *= cc.sec_in_yr / 1.0e5
    a2d *= 1000.0 / cc.asy_to_kms

    a2dsim *= cc.sec_in_yr / 1.0e5
    a2dsim *= 1000.0 / cc.asy_to_kms

    ##########
    #
    # Non-physical accelerations.
    #
    ##########

    # 1. Look for signficant non-zero tangential accelerations.
    bad1 = (np.where( abs(at/ate) > sigma))[0]
    print 'Found %d stars with tangential accelerations > %4.1f sigma' % \
          (len(bad1), sigma)

    # 2. Look for significant positive radial accelerations
    bad2 = (np.where( ((ar - (sigma*are)) > 0) ))[0]
    print 'Found %d stars with positive radial accelerations > %4.1f sigma' % \
          (len(bad2), sigma)


    # 3. Look for negative radial accelerations that are higher
    # than physically allowed.
    bad3 = (np.where( ar + (sigma*are) < a2d))[0]
    print 'Found %d stars with too large negative radial accelerations' % \
          (len(bad3))

    bad = np.unique( np.concatenate((bad1, bad2, bad3)) )

    # chose only bright stars
    nbad = len(bad)
    bright = (np.where(mag[bad] < 16.0))[0]

    bad = bad[bright]
    
#    bad = bad2
    # Sort
    rdx = r2d[bad].argsort()
    bad = bad[rdx]

    ######### look for physical accelerations
    goodAccel = np.where(ar + sigma*are < 0)[0]

    
    # loop through the bad points and look at the closest stars
    nearestStar = np.zeros(len(x))
    nearestStarBad = np.zeros(len(x))

    # look for the minium distance between all stars according to the fit
    for ii in arange(0,len(x)):
        distances = sqrt((x[ii] - x)**2 + (y[ii] - y)**2)
        srt = distances.argsort()
        distances = distances[srt]
        #nearestStar[ii] = distances[1]

        #loop over the 5 closest sources to see which one is closest
        #print names[ii]
        fitMin = np.zeros(5)
        for rr in arange(1,6):
            #print names[srt[rr]]
            xfit1 = [x[ii],vx[ii],ax[ii]/1000.0]
            xfit2 = [x[srt[rr]],vx[srt[rr]],ax[srt[rr]]/1000.0]
            yfit1 = [y[ii],vy[ii],ay[ii]/1000.0]
            yfit2 = [y[srt[rr]],vy[srt[rr]],ay[srt[rr]]/1000.0]
            fitMin[rr-1] = minDistance(xfit1,yfit1,xfit2,yfit2,epoch[ii],[np.amin(allEpochs),np.amax(allEpochs)])
            #print 'fitMin: %f' % fitMin[rr-1]
##         if np.amin(fitMin) < 0.1:
##             indMin = np.argmin(fitMin)
##             print names[ii] +'  ' +names[srt[indMin+1]]+ ' minDist: %f fitMin: %f' % (distances[1], np.amin(fitMin))
#            pdb.set_trace()
        nearestStar[ii] = np.amin(fitMin)

    # check only the unphysical accelerations
    for jj in bad:
        distances = sqrt((x[jj] - x)**2 + (y[jj] - y)**2)
        srt = distances.argsort()
        distances = distances[srt]
        nearestStarBad[jj] = distances[1]

        fitMin = np.zeros(5)
        for rr in arange(1,6):
            #print names[srt[rr]]
            xfit1 = [x[jj],vx[jj],ax[jj]/1000.0]
            xfit2 = [x[srt[rr]],vx[srt[rr]],ax[srt[rr]]/1000.0]
            yfit1 = [y[jj],vy[jj],ay[jj]/1000.0]
            yfit2 = [y[srt[rr]],vy[srt[rr]],ay[srt[rr]]/1000.0]

##             if names[jj] == 'S1-10':
##                 print names[srt[rr]]
##                 pause = 1
##             else:
##                 pause = 0
##             fitMin[rr-1] = minDistance(xfit1,yfit1,xfit2,yfit2,epoch[jj],[np.amin(allEpochs),np.amax(allEpochs)], pause = pause)
            fitMin[rr-1] = minDistance(xfit1,yfit1,xfit2,yfit2,epoch[jj],[np.amin(allEpochs),np.amax(allEpochs)])
            #print 'fitMin: %f' % fitMin[rr-1]
        if np.amin(fitMin) < 0.1:
            indMin = np.argmin(fitMin)
            print names[jj] +'  ' +names[srt[indMin+1]]+ ' minDist: %f fitMin: %f' % (distances[1], np.amin(fitMin))
#            pdb.set_trace()
        nearestStar[jj] = np.amin(fitMin)


    print '%d faint stars with non-physical accelerations' % (nbad - len(bright))
    print '*** Found %d stars with non-physical accelerations ***' % len(bad)
    
    returnNames = []
    print chi2x
#    print 'Name     Mag     x      y    chi2x     chi2y      ar     are     at    ate     nearest'
    print '|Name     |Mag     |x      |y    |chi2x     |chi2y      |ar     |are     |at    |ate     |nearest|'
    for bb in bad:
        returnNames = np.concatenate([returnNames,[names[bb]]])

#        print '%6s %7.3f %7.3f %7.3f %7.3f %7.3f %7.3f %7.3f %7.3f %7.3f %7.3f' % (names[bb], mag[bb], x[bb], y[bb],chi2x[bb],chi2y[bb],ar[bb]*cc.asy_to_kms/1000.0,are[bb]*cc.asy_to_kms/1000.0,at[bb]*cc.asy_to_kms/1000.0,ate[bb]*cc.asy_to_kms/1000.0,nearestStar[bb])
        print '|%6s |%7.3f |%7.3f |%7.3f |%7.3f |%7.3f |%7.3f |%7.3f |%7.3f |%7.3f |%7.3f|' % (names[bb], mag[bb], x[bb], y[bb],chi2x[bb],chi2y[bb],ar[bb]*cc.asy_to_kms/1000.0,are[bb]*cc.asy_to_kms/1000.0,at[bb]*cc.asy_to_kms/1000.0,ate[bb]*cc.asy_to_kms/1000.0,nearestStar[bb])

    if plotChiSq:
        # plot some chi-square info
        clf()
        subplot(231)
        # look at the distribution of chi-squares for stars brighter than 16 magnitude and have maximum degree of freedom
        maxEpoch = np.amax(nEpochs)
        dof = maxEpoch - 3
        good = np.where((mag < 16.0) & (nEpochs == maxEpoch))[0]

        # filter out the accelerations that do not have the max number of epochs measured
        print shape(good)
        print shape(bad)
        bad = np.intersect1d(good, bad)
        goodAccel = np.intersect1d(goodAccel, good)
        
        # use mean degree of freedom:
        #dof = np.floor(np.mean(nEpochs)-3)

        print 'Degree of freedom %f' % dof
        maxChi=30
        n, bins, patches1 = hist(chi2x[good], bins = np.arange(0, maxChi, 0.5),normed=1,label='x',alpha=0.6,color='blue')
        n, bins, patches2 = hist(chi2y[good], bins = bins, normed=1, label='y',alpha=0.6,color='green')
        xlabel('Chi-Sq Acc')
        title('All stars')
        chiInd = np.arange(0.01,100,0.01)
        chi2Theory = stats.chi2.pdf(chiInd,dof)
        plot(chiInd, chi2Theory)
    
        legend()
        subplot(232)
        n, bins, patches1 = hist(chi2x[bad], bins = np.arange(0,maxChi,0.5), normed=1,label='x',alpha=0.6)
        n, bins, patches2 = hist(chi2y[bad], bins = bins,normed=1,label='y',alpha=0.6)
        title('Non-physical Accelerations')
        xlabel('Chi-Sq Acc')
        plot(chiInd, chi2Theory)

        legend()

        # chi-sq distribution of physical accelerations
        subplot(233)
        n, bins, patches1 = hist(chi2x[goodAccel], bins = np.arange(0,maxChi,0.5), normed=1,label='x',alpha=0.6)
        n, bins, patches2 = hist(chi2y[goodAccel], bins = bins,normed=1,label='y',alpha=0.6)
        title('Physical Accelerations')
        xlabel('Chi-Sq Acc')
        plot(chiInd, chi2Theory)

        legend()

        #savefig('./chiSq_dist_accel.png')
        
        #clf()
        # plot the velocities
        velDof = dof + 1
        subplot(234)
        n, bins, patches1 = hist(chi2xv[good], bins = np.arange(0, maxChi, 0.5),normed=1,label='x',alpha=0.6,color='blue')
        n, bins, patches2 = hist(chi2yv[good], bins = bins, normed=1, label='y',alpha=0.6,color='green')
        xlabel('Chi-Sq Vel')
        title('All stars')
        chiInd = np.arange(0.01,100,0.01)
        chi2Theory = stats.chi2.pdf(chiInd,velDof)
        plot(chiInd, chi2Theory)
    
        legend()
        subplot(235)
        n, bins, patches1 = hist(chi2xv[bad], bins = np.arange(0,maxChi,0.5), normed=1,label='x',alpha=0.6)
        n, bins, patches2 = hist(chi2yv[bad], bins = bins,normed=1,label='y',alpha=0.6)
        title('Non-physical Accelerations')
        xlabel('Chi-Sq Vel')
        plot(chiInd, chi2Theory)

        legend()

        subplot(236)
        n, bins, patches1 = hist(chi2xv[goodAccel], bins = np.arange(0,maxChi,0.5), normed=1,label='x',alpha=0.6)
        n, bins, patches2 = hist(chi2yv[goodAccel], bins = bins,normed=1,label='y',alpha=0.6)
        title('Physical Accelerations')
        xlabel('Chi-Sq Vel')
        plot(chiInd, chi2Theory)

        legend()

        savefig(plotPrefix+'chiSq_dist_accel_vel.pdf')

        clf()
        subplot(131)
        loglog(chi2xv[good]/(dof+1.0),chi2x[good]/dof,'bo',label='X',alpha=0.6)
        plot(chi2yv[good]/(dof+1.0),chi2y[good]/dof,'go',label='Y',alpha=0.6)
        plot([0.001,100],[0.001,100])
        xlim(0.01,maxChi)
        ylim(0.01,maxChi)
        xlabel('Vel. Fit Reduced Chi-Sq')
        ylabel('Acc. Fit Reduced Chi-Sq')
        title('All Stars')
        legend(loc=2)
        
        subplot(132)
        loglog(chi2xv[bad]/(nEpochs[bad]-2.0),chi2x[bad]/(nEpochs[bad]-3.0),'bo',label='X',alpha=0.6)
        plot(chi2yv[bad]/(nEpochs[bad]-2.0),chi2y[bad]/(nEpochs[bad]-3.0),'go',label='Y',alpha=0.6)
        plot([0.001,100],[0.001,100])
        xlim(0.01,maxChi)
        ylim(0.01,maxChi)
        xlabel('Vel. Fit Reduced Chi-Sq')
        ylabel('Acc. Fit Reduced Chi-Sq')
        title('Non-physical Accelerations')
        legend(loc=2)

        subplot(133)
        loglog(chi2xv[goodAccel]/(nEpochs[goodAccel]-2.0),chi2x[goodAccel]/(nEpochs[goodAccel]-3.0),'bo',label='X',alpha=0.6)
        plot(chi2yv[goodAccel]/(nEpochs[goodAccel]-2.0),chi2y[goodAccel]/(nEpochs[goodAccel]-3.0),'go',label='Y',alpha=0.6)
        plot([0.001,100],[0.001,100])
        xlim(0.01,maxChi)
        ylim(0.01,maxChi)
        xlabel('Vel. Fit Reduced Chi-Sq')
        ylabel('Acc. Fit Reduced Chi-Sq')
        title('Physical Accelerations')
        legend(loc=2)

        savefig(plotPrefix+'chi2_vel_vs_accel.pdf')

        # plot chi-sq as a function of magnitude
        clf()
        subplot(231)
        semilogy(mag[good],chi2x[good]/(nEpochs[good]-3.0),'bo',label='X',alpha=0.6)
        plot(mag[good],chi2y[good]/(nEpochs[good]-3.0),'go',label='Y',alpha=0.6)
        ylim(0.01,maxChi)
        xlabel('K mag')
        ylabel('Accel. Fit Chi-Sq')
        title('All Stars')
        legend(loc=3)

        subplot(232)
        semilogy(mag[bad],chi2x[bad]/(nEpochs[bad]-3.0),'bo',label='X',alpha=0.6)
        plot(mag[bad],chi2y[bad]/(nEpochs[bad]-3.0),'go',label='Y',alpha=0.6)
        ylim(0.01,maxChi)
        xlabel('K mag')
        ylabel('Accel. Fit Chi-Sq')
        title('Non-physical Accelerations')
        legend(loc=3)

        subplot(233)
        semilogy(mag[goodAccel],chi2x[goodAccel]/(nEpochs[goodAccel]-3.0),'bo',label='X',alpha=0.6)
        plot(mag[goodAccel],chi2y[goodAccel]/(nEpochs[goodAccel]-3.0),'go',label='Y',alpha=0.6)
        ylim(0.01,maxChi)
        xlabel('K mag')
        ylabel('Accel. Fit Chi-Sq')
        title('Physical Accelerations')
        legend(loc=3)

        subplot(234)
        semilogy(mag[good],chi2xv[good]/(nEpochs[good]-2.0),'bo',label='X',alpha=0.6)
        plot(mag[good],chi2yv[good]/(nEpochs[good]-2.0),'go',label='Y',alpha=0.6)
        ylim(0.01,maxChi)
        xlabel('K mag')
        ylabel('Vel. Fit Chi-Sq')
        title('All Stars')
        legend(loc=3)

        subplot(235)
        semilogy(mag[bad],chi2xv[bad]/(nEpochs[bad]-2.0),'bo',label='X',alpha=0.6)
        plot(mag[bad],chi2yv[bad]/(nEpochs[bad]-2.0),'go',label='Y',alpha=0.6)
        ylim(0.01,maxChi)
        xlabel('K mag')
        ylabel('Vel. Fit Chi-Sq')
        title('Non-physical Accelerations')
        legend(loc=3)

        subplot(236)
        semilogy(mag[goodAccel],chi2xv[goodAccel]/(nEpochs[goodAccel]-2.0),'bo',label='X',alpha=0.6)
        plot(mag[goodAccel],chi2yv[goodAccel]/(nEpochs[goodAccel]-2.0),'go',label='Y',alpha=0.6)
        ylim(0.01,maxChi)
        xlabel('K mag')
        ylabel('Vel. Fit Chi-Sq')
        title('Physical Acceleration')
        legend(loc=3)

        savefig(plotPrefix+'mag_vs_chi2.pdf')
    else:
        # plot the histogram
        clf()
        subplot(221)
        nAll, allbins, patches = hist(nearestStar,bins=50)
    #nBad, badbins, patches2 = hist(nearestStarBad[bad],bins=allbins)
        nBad, badbins, patches2 = hist(nearestStarBad[bad],bins=10)
        setp(patches2, 'facecolor','r')
        xlabel('Nearest Neighbor (arcsec)')
        ylabel('N Stars')
    
        # do a ks test of the two distributions
        print 'len(allbins) %f' % len(allbins)
        print 'len(badbins) %f' % len(badbins)
        print nAll
        print nBad
        ksTest = stats.ks_2samp(nAll, nBad)
        print ksTest
        
        subplot(222)
        plot(rarc,a2dsim)
        plot([0,10],[0,0],hold=True)
          
        #    plot(r[bad],ar[bad],are[bb],'o')
        py.errorbar(r[bad],ar[bad],are[bad],fmt='o')
        xlim(0,5)
        ylim(2,-2)
        xlabel('Distance from Sgr A* (arcsec)')
        ylabel('Acceleration (km/s/yr)')
    
    
        subplot(223)
        # look at the distribution of chi-squares for stars brighter than 16 magnitude
        good = np.where(mag < 16.0)

        # use mean degree of freedom:
        dof = np.floor(np.mean(nEpochs)-3)

        print 'Mean degree of freedom %f' % dof
    
        n, bins, patches1 = hist(chi2x[good], bins = np.arange(0, 20, 0.1),normed=1,label='x',alpha=0.6,color='blue')
        n, bins, patches2 = hist(chi2y[good], bins = bins, normed=1, label='y',alpha=0.6,color='green')
        xlabel('Reduced Chi^2')
        
        chiInd = np.arange(0.01,100,0.01)
        chi2Theory = stats.chi2.pdf(chiInd,dof)
        plot(chiInd, chi2Theory)
    
        legend()
        subplot(224)
        n, bins, patches1 = hist(chi2x[bad], bins = np.arange(0,20,0.5), normed=1,label='x',alpha=0.6)
        n, bins, patches2 = hist(chi2y[bad], bins = bins,normed=1,label='y',alpha=0.6)
        xlabel('Reduced Chi^2')
        plot(chiInd, chi2Theory)

        legend()
        savefig('./unphysicalAccel_stats.png')
    return returnNames


def fitfunPoly2(p, fjac=None, x=None, y=None, err=None):
    # second order polynomial
    fun = p[0] + p[1]*x + 0.5*p[2]*x**2

    #deviations from the model
    deviates = (y - fun)/err
    return [0, deviates]

def poly2(p, x):
    return p[0] + p[1]*x + 0.5*p[2]*x**2

def accelCheck(x,xerr,y,yerr,time, t0 = 0, n = 4, name = None):
    #takes in an array of x and y values and randomly add in points to
    #fit for velocities and accelerations to figure out if one epoch
    #is causing a problem

    #remove the bad points
    good = (np.where(abs(x) < 500))[0]
    if (len(good) > 1):
        x = x[good]
        xerr=xerr[good]
        y = y[good]
        yerr=yerr[good]
        time = time[good]

    # sample randomly without replacement
    draw = random.sample(arange(len(x)),len(x))

    # status array to record the state of the fit
    statusStack = np.zeros((len(draw),4))
    statusStack[:,0] = time[draw]
    print 'Mean X err: %10.7f, mean Y err: %10.7f' % (np.mean(xerr),np.mean(yerr))
    print 'Status flags: tan accel, + radial, > max radial'
    for dd in arange(len(draw)-n+1):
        inds = draw[0:n+dd]
        #print time[inds[-1]]
        xSub = x[inds]
        xSubErr = xerr[inds]

        ySub = y[inds]
        ySubErr= yerr[inds]

        # do the fit with delta t from reference epoch
        timeSub = time[inds]-t0
    
        # initial guess
        p0x = [np.mean(xSub),np.amax(xSub) - np.amin(xSub), 0.0]
        p0y = [np.mean(ySub),np.amax(ySub) - np.amin(ySub), 0.0]
    
        #print p0x
        functargsX = {'x':timeSub, 'y':xSub, 'err': xSubErr}
        functargsY = {'x':timeSub, 'y':ySub, 'err': ySubErr}
    
        #do the fitting
        xfit = nmpfit_sy.mpfit(fitfunPoly2, p0x, functkw=functargsX,quiet=1)
        yfit = nmpfit_sy.mpfit(fitfunPoly2, p0y, functkw=functargsY,quiet=1)
        dof = len(xSub) - len(xfit.params)
        xredChiSq = xfit.fnorm/dof
        yredChiSq = yfit.fnorm/dof
        
        try:
            xpcerror = xfit.perror #* xredChiSq
            ypcerror = yfit.perror #* yredChiSq
        except TypeError:
            print timeSub
            print xSub
            print ySub
        #print xfit.params
        #print xpcerror
        #print yfit.params
        #print ypcerror
        
        # check whether the acceleration is physical
        x0 = poly2(xfit.params, 0.0)
        y0 = poly2(yfit.params, 0.0)
        status = isPhysical(x0, y0, xfit.params[2], xpcerror[2], yfit.params[2], ypcerror[2],arcsec=1)
        #print 'chi-sq %f, reduced chi-sq: %f' % (xfit.fnorm, xredChiSq)
        
        print 'x %8.5f %8.5f+-%8.5f %8.5f+-%8.5f %10.7f+-%10.7f' % (time[inds[-1]], xfit.params[0], xpcerror[0], xfit.params[2], xpcerror[2], xfit.params[2], xpcerror[2])
        print 'y %8.5f %8.5f+-%8.5f %8.5f+-%8.5f %10.7f+-%10.7f %d %d %d' % (time[inds[-1]], yfit.params[0], ypcerror[0], yfit.params[1], ypcerror[1], yfit.params[2], ypcerror[2], status[0], status[1], status[2])
        #print status

        
    
    #plot(timeSub,xSub,'o')
    simTime = np.arange(np.amin(timeSub),np.amax(timeSub)+0.1,0.1)
    simX = poly2(xfit.params,simTime)
    simY = poly2(yfit.params,simTime)

    # figure out where the best fit location for each date are
    propX = poly2(xfit.params,timeSub)
    propY = poly2(yfit.params,timeSub)
    
    #plot(simTime,poly2(xfit.params,simTime),hold=True)

    #switch the axes back
    xSub = -xSub
    simX = -simX
    clf()
    plot(xSub,ySub,'o')
    py.errorbar(xSub,ySub,xSubErr,ySubErr,fmt='o')
    plot(-propX,propY,'o')
    print ySub
    print time
    for ii in arange(0,len(time)):
        text(-propX[ii],propY[ii],str(timeSub[ii]+t0),size='x-small',color='r')
        text(-x[ii],y[ii],str(time[ii]),size='x-small')
        
        
    plot(simX,simY,color='r')
    xlim(np.amax(xSub)+np.amax(xSubErr),np.amin(xSub)-np.amax(xSubErr))
    ylim(np.amin(ySub)-np.amax(ySubErr),np.amax(ySub)+np.amax(ySubErr))
    title(name)
    xlabel('X offest (arcsec)')
    ylabel('y offset (arcsec)')
    
    #py.errorbar(timeSub,xSub,xSubErr,fmt='o')
##     if name:
##         show()
##         #py.savefig('./checkAccel_'+name+'.png')
##     else:
##         show()
    
def isPhysical(x, y, ax, axe, ay, aye, arcsec = None, sigma = 3):
    """
    Return a True or False depending on whether the acceleration
    measurement is physical. Unphysical accelerations are defined as
    either: 1. Significant tangential acceleration 2. Significant
    positive radial acceleration 3. Negative acceleration greater than
    the maximum allowed at the 2D position.

    RETURN: status array where 1 is true [tangential, pos. radial, > max radial]
    """
    
    status = np.zeros(3)
    # Lets do radial/tangential
    r = np.sqrt(x**2 + y**2)
    ar = ((ax*x) + (ay*y)) / r
    at = ((ax*y) - (ay*x)) / r
    are = np.sqrt((axe*x)**2 + (aye*y)**2) / r
    ate = np.sqrt((axe*y)**2 + (aye*x)**2) / r

    # Total acceleration
    atot = py.hypot(ax, ay)
    atoterr = np.sqrt((ax*axe)**2 + (ay*aye)**2) / atot

    # Calculate the acceleration limit set by the projected radius

    if arcsec:
        #convert to mks
        cc = objects.Constants()
        
        # Convert into cm
        r2d = r * cc.dist * cc.cm_in_au

        rarc =np.arange(0.01,10.0,0.01)
        rsim = rarc * cc.dist * cc.cm_in_au
    
        # acc1 in cm/s^2
        a2d = -cc.G * cc.mass * cc.msun / r2d**2
        a2dsim = -cc.G * cc.mass * cc.msun / rsim**2
        
        # acc1 in km/s/yr
    
        a2d *= cc.sec_in_yr / 1.0e5

        #a2d *= 1000.0 / cc.asy_to_kms

        # convert between arcsec/yr^2 to km/s/yr
        ar *= cc.asy_to_kms
        are *= cc.asy_to_kms

        at *= cc.asy_to_kms
        ate *= cc.asy_to_kms

        a2dsim *= cc.sec_in_yr / 1.0e5
        #a2dsim *= 1000.0 / cc.asy_to_kms

    # tests to see if the accelerations are physical
    if (abs(at/ate) > sigma):
        #print 'significant tangential acceleration'
        status[0] = 1

    #print 'radial acceleration %f +- %f' % (ar, are)
    #print 'tangential acceleration %f +- %f' % (at, ate)
    if ((ar - (sigma*are)) > 0):
        #print 'positive radial acceleration'
        status[1] = 1

    if (ar + (sigma*are) < a2d):
        #print 'too large radial acceleration'
        status[2] = 1

    return status
