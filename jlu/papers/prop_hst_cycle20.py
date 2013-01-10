import pylab as py
import numpy as np
from jlu.wd1 import synthetic as syn
import math
import atpy
from jlu.util import statsIter
from jlu.util import constants as cc
import ephem

#AKs = 0.91
AKs = 0.75
distance = 5000
age = 4e6
logAge = math.log10(age)

scale = 61.0

propDir = '/u/jlu/doc/proposals/hst/cycle20/wd1/'
plotDir = propDir + 'plots/'

def mass_luminosity():
    # Load up observed data:
    wd1_data = '/u/jlu/data/Wd1/hst/from_jay/EXPORT_WEST1.2012.02.04/wd1_catalog.fits'
    data = atpy.Table(wd1_data)

    iso = load_isochrone()

    # Determine median astrometric and photometric error as a function of
    # magnitude for both F814W and F125W. Use Y as a proxy for astrometric error.
    magBinSize = 0.25
    magBinCenter = np.arange(12, 28, magBinSize)
    yerr814w = np.zeros(len(magBinCenter), dtype=float)
    yerr125w = np.zeros(len(magBinCenter), dtype=float)
    merr814w = np.zeros(len(magBinCenter), dtype=float)
    merr125w = np.zeros(len(magBinCenter), dtype=float)

    for ii in range(len(magBinCenter)):
        mlo = magBinCenter[ii] - (magBinSize/2.0)
        mhi = magBinCenter[ii] + (magBinSize/2.0)

        idx814 = np.where((data.mag814 >= mlo) & (data.mag814 < mhi) &
                          (np.isnan(data.y2005_e) == False))[0]
        idx125 = np.where((data.mag125 >= mlo) & (data.mag125 < mhi) &
                          (np.isnan(data.y2010_e) == False))[0]

        if len(idx814) > 1:
            yerr814w[ii] = statsIter.mean(data.y2005_e[idx814], hsigma=3, lsigma=5, iter=10)
            merr814w[ii] = statsIter.mean(data.mag814_e[idx814], hsigma=3, lsigma=5, iter=10)
        else:
            yerr814w[ii] = np.nan
            merr814w[ii] = np.nan

        if len(idx125) > 1:
            yerr125w[ii] = statsIter.mean(data.y2010_e[idx125], hsigma=3, lsigma=5, iter=10)
            merr125w[ii] = statsIter.mean(data.mag125_e[idx125], hsigma=3, lsigma=5, iter=10)
        else:
            yerr125w[ii] = np.nan
            merr125w[ii] = np.nan

    # Assume we get an additional sqrt(3) by combining the 3 filters together.
    yerr125w /= math.sqrt(3.0)
            
    py.clf()
    py.plot(magBinCenter, yerr814w*scale, 'b.', ms=10,
            label='ACS-WFC F814W')
    py.plot(magBinCenter, yerr125w*scale, 'r.', ms=10,
            label='WFC3-IR F125W')
    py.xlabel('Magnitude')
    py.ylabel('Positional Error (mas)')
    py.legend(numpoints=1, loc='upper left')
    py.ylim(0, 5)
    py.savefig(plotDir + 'avg_poserr_f814w_vs_f125w.png')

    py.clf()
    py.plot(magBinCenter, merr814w, 'b.', ms=10,
            label='ACS-WFC F814W')
    py.plot(magBinCenter, merr125w, 'r.', ms=10,
            label='WFC3-IR F125W')
    py.xlabel('Magnitude')
    py.ylabel('Photometric Error (mag)')
    py.legend(numpoints=1, loc='upper left')
    py.ylim(0, 0.1)
    py.savefig(plotDir + 'avg_magerr_f814w_vs_f125w.png')

    idx814 = np.where((yerr814w*scale < 2) & (merr814w < 0.04))[0]
    idx125 = np.where((yerr125w*scale < 2) & (merr125w < 0.04))[0]

    lim814 = magBinCenter[idx814[-1]]
    lim125 = magBinCenter[idx125[-1]]

    print 'Limit for F814W data (pos err < 2 mas, mag err < 0.04: %5.2f' % \
        lim814
    print 'Limit for F125W data (pos err < 2 mas, mag err < 0.04: %5.2f' % \
        lim125

    ii814 = np.abs(iso.mag814w - lim814).argmin()
    ii125 = np.abs(iso.mag125w - lim125).argmin()

    # massLim814 = iso.M[ii814]
    # magLim814 = iso.mag814w[ii814]
    massLim814 = 0.85
    magLim814 = 23.5
    massLim125 = iso.M[ii125]
    magLim125 = iso.mag125w[ii125]

    py.clf()
    py.semilogx(iso.M, iso.mag814w, 'b-', ms=2, label='ACS-WFC F814W',
                linewidth=2)
    py.semilogx(iso.M, iso.mag125w, 'r-', ms=2, label='WFC3-IR F125W',
                linewidth=2)
    py.gca().invert_yaxis()
    py.plot([massLim814, massLim814], [magLim814-2, magLim814+2], 'b-',
            linewidth=4)
    py.plot([massLim125, massLim125], [magLim125-2, magLim125+2], 'r-',
            linewidth=4)
    ar1 = py.Arrow(massLim814, magLim814+0.1, 1.0, 0, color='blue')
    ar2 = py.Arrow(massLim125, magLim125+0.1, 0.15, 0, color='red')
    py.gca().add_patch(ar1)
    py.gca().add_patch(ar2)
    py.xlabel('Mass (Msun)')
    py.ylabel('Magnitude (F814W or F125W)')
    py.legend(loc='upper left')
    py.savefig(plotDir + 'mass_magnitude_with_limits.png')

    # Also plot a color-magnitude diagram and show the two mass limits
    py.clf()
    py.plot(data.mag814 - data.mag160, data.mag814, 'k.', ms=2)
    py.plot(iso.mag814w - iso.mag160w, iso.mag814w, 'b-', ms=2,
            linewidth=2)
    py.plot(iso.mag814w[ii814] - iso.mag160w[ii814], iso.mag814w[ii814],
            'b^', ms=10)
    py.plot(iso.mag814w[ii125] - iso.mag160w[ii125], iso.mag814w[ii125],
            'r^', ms=10)
    py.gca().invert_yaxis()
    py.xlim(0, 10)
    py.ylim(28, 13)
    py.xlabel('F814W - F160W')
    py.ylabel('F814W')
    py.savefig(plotDir + 'cmd_optical_iso.png')

    py.clf()
    py.plot(data.mag125 - data.mag160, data.mag125, 'k.', ms=2)
    py.plot(iso.mag125w - iso.mag160w, iso.mag125w, 'r-', ms=2,
            linewidth=2)
    py.plot(iso.mag125w[ii814] - iso.mag160w[ii814], iso.mag125w[ii814],
            'b^', ms=10)
    py.plot(iso.mag125w[ii125] - iso.mag160w[ii125], iso.mag125w[ii125],
            'r^', ms=10)
    py.gca().invert_yaxis()
    py.xlim(0, 3)
    py.ylim(24, 10)
    py.xlabel('F125W - F160W')
    py.ylabel('F125W')
    py.savefig(plotDir + 'cmd_infrared_iso.png')

    
def cmd():
    # Load up observed data:
    wd1_data = '/u/jlu/data/Wd1/hst/from_jay/EXPORT_WEST1.2012.02.04/wd1_catalog.fits'
    data = atpy.Table(wd1_data)

    # Load up model isochrone
    iso = load_isochrone()

    # Select a region in NIR CMD for defining cluster center and
    # velocity dispersion.
    magLo = 15.0
    magHi = 16.5
    colorLo = 0.8
    colorHi = 0.88
    
    # Pull out a few key masses along the isochrone
    idxM1 = np.abs(iso.M - 1.0).argmin()
    idxM01 = np.abs(iso.M - 0.1).argmin()

    colM1 = iso.mag125w[idxM1] - iso.mag160w[idxM1]
    magM1 = iso.mag125w[idxM1]
    colM01 = iso.mag125w[idxM01] - iso.mag160w[idxM01]
    magM01 = iso.mag125w[idxM01]

    # Plot up the complete near-infrared color magnitude diagram, select out
    # a color region to show cluster members in the VPD.
    py.clf()
    py.subplots_adjust(left=0.11)
    py.plot(data.mag125 - data.mag160, data.mag125, 'k.', ms=2)
    py.plot(iso.mag125w - iso.mag160w, iso.mag125w, 'r-', ms=2,
            color='red', linewidth=2)
    py.plot([colM1], [magM1], 'r*', ms=25)
    py.plot([colM01], [magM01],'r*', ms=25)
    py.text(colM1+0.1, magM1, r'1 M$_\odot$', color='red', 
            fontweight='normal', fontsize=28)
    py.text(colM01+0.1, magM01, r'0.1 M$_\odot$', color='red',
            fontweight='normal', fontsize=28)
    py.gca().invert_yaxis()

    py.plot([colorLo, colorHi, colorHi, colorLo, colorLo,],
            [magLo, magLo, magHi, magHi, magLo],
            'k-', color='lightgreen', linewidth=4)

    py.xlim(0, 2)
    py.ylim(23, 12)
    py.xlabel('F125W - F160W')
    py.ylabel('F125W')
    py.savefig(plotDir + 'cmd_infrared_vpd_box.png')

    # Plot up the VPD with the CMD selected highlighted.
    # Also trim out anything without good astrometric and
    # photometric errors.
    idx = np.where((data.x2005_e*scale < 2) & (data.y2005_e*scale < 2) &
                   (data.x2010_e*scale < 2) & (data.y2010_e*scale < 2) &
                   (data.mag814_e < 0.04) & (data.mag125_e < 0.04))[0]

    mag = data.mag125[idx]
    color = data.mag125[idx] - data.mag160[idx]
    tdx = np.where((mag > magLo) & (mag < magHi) & 
                   (color > colorLo) & (color < colorHi))[0]

    dx_mean = statsIter.mean(data.dx[idx[tdx]], hsigma=5, lsigma=5, iter=5)
    dy_mean = statsIter.mean(data.dy[idx[tdx]], hsigma=5, lsigma=5, iter=5)
    dx_std = statsIter.std(data.dx[idx[tdx]], hsigma=5, lsigma=5, iter=5)
    dy_std = statsIter.std(data.dy[idx[tdx]], hsigma=5, lsigma=5, iter=5)

    print 'Cluster Center: '
    print '   x = %5.1f +/- %5.1f mas' % (dx_mean, dx_std)
    print '   y = %5.1f +/- %5.1f mas' % (dy_mean, dy_std)

    # Select out cluster members
    dr = np.hypot(data.dx[idx] - dx_mean, data.dy[idx] - dy_mean)
    dr_cluster = np.max([dx_std, dy_std])
    inCluster = np.where(dr <= dr_cluster)[0]

    dt = 2010.6521 - 2005.4846

    py.clf()
    py.subplots_adjust(left=0.13)
    py.plot(data.dx[idx]/dt, data.dy[idx]/dt, 'k.', ms=4)
    py.plot(data.dx[idx[tdx]]/dt, data.dy[idx[tdx]]/dt,
            'b.', color='lightgreen', ms=10)
    circ = py.Circle([dx_mean/dt, dy_mean/dt], radius=dr_cluster/dt,
                     fc='none', ec='yellow', linewidth=4, zorder=10)
    py.gca().add_patch(circ)
    lim = 20
    py.axis([-lim/dt, lim/dt, -lim/dt, lim/dt])
    py.xlabel('X Proper Motion (mas/yr)')
    py.ylabel('Y Proper Motion (mas/yr)')
    py.title('2010 F814W - 2005 F125W')
    py.savefig(plotDir + 'vpd_cmd_selected.png')
    
    # Plot up the complete near-infrared color magnitude diagram, 
    # and overplot cluster members as determined from proper motions
    # from optical/IR data set.
    py.clf()
    py.subplots_adjust(left=0.11)
    py.plot(data.mag125 - data.mag160, data.mag125, 'k.', ms=2)
    py.plot(data.mag125[idx[inCluster]] - data.mag160[idx[inCluster]],
            data.mag125[idx[inCluster]], 'y.', ms=5)
    py.plot(iso.mag125w - iso.mag160w, iso.mag125w, 'r-', ms=2,
            linewidth=2)
    py.plot([colM1], [magM1], 'r*', ms=25)
    py.plot([colM01], [magM01],'r*', ms=25)
    py.text(colM1+0.1, magM1, r'1 M$_\odot$', color='red', 
            fontweight='normal', fontsize=28)
    py.text(colM01+0.1, magM01, r'0.1 M$_\odot$', color='red',
            fontweight='normal', fontsize=28)
    py.gca().invert_yaxis()
    py.xlim(0, 2)
    py.ylim(23, 12)
    py.xlabel('F125W - F160W')
    py.ylabel('F125W')
    py.savefig(plotDir + 'cmd_infrared_members.png')


    
def load_isochrone(logAge=logAge, AKs=AKs, distance=distance):
    # Load up model isochrone
    iso = syn.load_isochrone(logAge=logAge, AKs=AKs, distance=4000)

    deltaDM = 5.0 * math.log10(distance / 4000.0)

    iso.mag814w += deltaDM
    iso.mag125w += deltaDM
    iso.mag139m += deltaDM
    iso.mag160w += deltaDM

    return iso

def clusterProperMotion():
    name = 'Wd1'
    dist = distance / 1.0e3
    ra = '16:47:04'
    dec = '-45:51:05'

    obj = ephem.FixedBody()
    obj._ra = ephem.hours(ra)
    obj._dec = ephem.degrees(dec)
    obj._epoch = 2000
    obj.compute()

    gal = ephem.Galactic(obj)

    long = math.degrees(float(gal.long))
    print ''
    print '%-10s  %s %s  l = %.1f' % (name, ra, dec, long)
    properMotions(long, dist, name)

def properMotions(galacticLat, distance, clusterName):
    """
    Calculate the range of proper motions for different
    possible distances. Assume a constant circular rotation 
    of the galaxy of 220 km/s.

    Inputs:
    galacticLat -- in degrees.
    """
    # R0 - distance from Earth to the GC (Ghez et al. 2009; Reid et al. 2009)
    # d  - distance from Earth to the object. 
    # R  - distance from the object to the GC. Need this to get velocity.
    # Theta0 - rotational velocity at solar circle (Reid et al. 2009)
    #
    # l  - galacticLat in radians
    # 
    l = math.radians(galacticLat)
    R0 = 8.4        # kpc
    Theta0 = 254.0  # km/s
    
    d = np.arange(1.0, 12, 0.05) # distance in kpc

    cosl = math.cos(l)
    sinl = math.sin(l)

    R = np.sqrt( d**2 + R0**2 - (2.0 * d * R0 * math.cos(l)) )

    x = R0 - d * cosl
    y = d * sinl

    # Assume a rotation curve based on a reasonable potential.
    # Pulled from Brunetti and Pfenniger (2010)
    vc = Theta0 * R / np.sqrt(1 + R**2)


    oneR_2 = np.sqrt(1.0 + R**2)
    oneR0_2 = math.sqrt(1.0 + R0**2)

    vt = (x*R0 - R**2) / (d * oneR_2) + (x*R0 - R0**2) / (d * oneR0_2)
    vt *= Theta0

    vr = ((-y * R0) / d) * ((1.0 / oneR_2) - (1.0 / oneR0_2))
    vr *= Theta0

    
    # Proper motion (mas/yr)
    kms_masyr = 1.0e5 * cc.sec_in_yr / (d * cc.cm_in_AU)
    pm = vt * kms_masyr
    pm_hi = (vt + 5.0) * kms_masyr
    pm_lo = (vt - 5.0) * kms_masyr

    py.clf()
    py.figure(linewidth=2)
    py.plot(d, pm, 'r-', linewidth=2)
    py.plot(d, pm_hi, 'r--', linewidth=2)
    py.plot(d, pm_lo, 'r--', linewidth=2)
    py.xlabel('Distance (kpc)', fontsize=22, fontweight='bold')
    py.ylabel('Proper Motion (mas/yr)', fontsize=22, fontweight='bold')
    title = '%s (l = %d)' % (clusterName, galacticLat)
    py.title(title, fontsize=22, fontweight='bold')
    

    # Calculate Proper Motion Errors
    # Positional Errors for each measurements
    posErr = 1.0 # mas

    # Times of measurements
    t = np.array([2010.65, 2013.5, 2015.5])

    # Proper Motion error
    pmErr = posErr / math.sqrt( ((t - t.mean())**2).sum() )

    print 'Proper Motion Error Calculation:'
    print '   Positional Error: %3.1f mas' % posErr
    print '   Time of Measurements: ', t
    print ''
    print '   Proper Motion Errors: %4.2f mas/yr' % pmErr

    diff = np.abs(d - distance)
    ii = diff.argmin()
    pm_max = pm - pm + pm[ii] + (pmErr/2.0)
    pm_min = pm - pm + pm[ii] - (pmErr/2.0)

    py.fill_between(d, pm_min, pm_max, color='grey', alpha=0.3)
    py.plot([distance], [pm[ii]], 'k*', ms=10)

    py.ylim(-9, 0)
    
    ax = py.gca()
    for tick in ax.xaxis.get_major_ticks(): 
        tick.label1.set_fontsize(16) 
        tick.label1.set_fontweight('bold') 
    for tick in ax.yaxis.get_major_ticks(): 
        tick.label1.set_fontsize(16) 
        tick.label1.set_fontweight('bold') 

    outfile = plotDir + 'hst_clusterPropMot_%s' % clusterName
    py.savefig(outfile + '.png')
    py.savefig(outfile + '.eps')


def spiral_dither_pattern(numSteps, stepSize, angle, x0=0, y0=0):
    """
    Plot up the positions in coordinates and pixel phase space of a spiral
    dither pattern. This is useful for checking for adequate pixel phase coverage.

    numSteps -

    stepSize - step size in arcsec as given in the APT

    angle - angle in degrees.

    """
    spiral_dx = np.array([ 0,  1,  1,  0, -1,
                          -1, -1,  0,  1,  2,
                           2,  2,  2,  1,  0,
                          -1, -2, -2, -2, -2,
                          -2, -1,  0,  1,  2], dtype=float)

    spiral_dy = np.array([ 0,  0,  1,  1,  1,
                           0, -1, -1, -1, -1,
                           0,  1,  2,  2,  2,
                           2,  2,  1,  0, -1,
                          -2, -2, -2, -2, -2], dtype=float)

    xscale = 0.136
    yscale = 0.121

    spiral_dx = spiral_dx[:numSteps]
    spiral_dy = spiral_dy[:numSteps]

    cosa = math.cos(math.radians(angle))
    sina = math.sin(math.radians(angle))

    x = spiral_dx * cosa + spiral_dy * sina
    y = -spiral_dx * sina + spiral_dy * cosa

    x *= stepSize
    y *= stepSize

    xPixPhase = (x/xscale) % 1.0
    yPixPhase = (y/yscale) % 1.0

    for i in range(numSteps):
        fmt = 'Position {0:2d}:  X = {1:7.3f}  Y = {2:7.3f}'
        print fmt.format(i+1, x0 + x[i], y0 + y[i])

    py.figure(1)
    py.clf()
    py.plot(x, y, 'ko-')
    py.xlabel('X Offset (arcsec)')
    py.ylabel('Y Offset (arcsec)')
    py.axis('equal')

    py.figure(2)
    py.clf()
    py.plot(x/xscale, y/yscale, 'ks-')
    py.xlabel('X Offset (pixels)')
    py.ylabel('Y Offset (pixels)')
    py.axis('equal')

    py.figure(3)
    py.clf()
    py.plot(xPixPhase, yPixPhase, 'ko')
    py.xlabel('X Pixel Phase')
    py.ylabel('Y Pixel Phase')
    py.plot([0, 0, 1, 1, 0], [0, 1, 1, 0, 0], 'k--')
    py.xlim(-0.05, 1.05)
    py.ylim(-0.05, 1.05)


def mosaic_subarray():
    """
    Calculate offsets for the WFC3-IR sub-array mosaic.
    """
    # Center positions of the 2 x 2 full-array mosaic
    x_mos4 = np.array([-55.0,  55.0, -55.0,  55.0])
    y_mos4 = np.array([ 55.0,  55.0, -55.0, -55.0])

    x_fov = 123.0
    y_fov = 136.0

    dx_mos8 = np.array([-x_fov/4.0,  x_fov/4.0, -x_fov/4.0,  x_fov/4.0])
    dy_mos8 = np.array([ y_fov/4.0,  y_fov/4.0, -y_fov/4.0, -y_fov/4.0])

    for ii in range(len(x_mos4)):
        print 'Positions for sub-array images covering pointing %d at [%5.1f, %5.1f]' % \
            (ii, x_mos4[ii], y_mos4[ii])

        dx = x_mos4[ii] + dx_mos8
        dy = y_mos4[ii] + dy_mos8

        for jj in range(len(dx)):
            print 'Offsets for pos %d, sub pos %d:  %6.1f  %6.1f' % \
                (ii, jj, dx[jj], dy[jj])
    


    
