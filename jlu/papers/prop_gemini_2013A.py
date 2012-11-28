import numpy as np
import pylab as py
import ephem
import math
from jlu.util import constants as cc
from mpl_toolkits.axes_grid1 import ImageGrid

propDir = '/u/jlu/doc/proposals/gemini/2013A/'
plotDir = propDir + 'plots/'

def clusterSample():
    clusters = [{'name': 'M17',   'distance': 2, 'ra': '18:20:26', 'dec': '-17:10:01'},
                {'name': 'Wd2',   'distance': 5, 'ra': '10:23:58', 'dec': '-57:45:49'},
                {'name': 'Wd1',   'distance': 5, 'ra': '16:47:04', 'dec': '-45:51:05'},
                {'name': 'RSGC1', 'distance': 6, 'ra': '18:37:58', 'dec': '-06:52:53'},
                {'name': 'RSGC2', 'distance': 6, 'ra': '18:39:20', 'dec': '-06:05:10'}
                ]

    py.close(1)
    py.figure(1, linewidth=2, figsize=(16, 10))
    py.subplots_adjust(left=0.05, right=0.97, bottom=0.1, top=0.95, wspace=0.2, hspace=0.25)

    for ii in range(len(clusters)):
        clust = clusters[ii]

        obj = ephem.FixedBody()
        obj._ra = ephem.hours(clust['ra'])
        obj._dec = ephem.degrees(clust['dec'])
        obj._epoch = 2000
        obj.compute()

        gal = ephem.Galactic(obj)

        longitude = math.degrees(float(gal.lon))
        print ''
        print '%-10s  %s %s  l = %.1f' % \
            (clust['name'], clust['ra'], clust['dec'], longitude)

        py.subplot(2, 3, ii+1)
        properMotions(longitude, clust['distance'], clust['name'])

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

    # Calculate Proper Motion Errors
    # Positional Errors for each measurements
    posErr = 0.5 # mas

    # Times of measurements
    t = np.array([2013.5, 2014.5, 2015.5])

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

    # Plotting
    py.fill_between(d, pm_min, pm_max, color='lightgrey')
    py.plot(d, pm, 'r-', linewidth=2)
    py.plot(d, pm_hi, 'r--', linewidth=2)
    py.plot(d, pm_lo, 'r--', linewidth=2)
    py.plot([distance], [pm[ii]], 'k*', ms=10)

    py.xlabel('Distance (kpc)', fontsize=22, fontweight='bold')
    py.ylabel('Proper Motion (mas/yr)', fontsize=22, fontweight='bold')
    
    py.ylim(-9, 0)

    lim = py.axis()
    py.text(lim[1]-0.5, lim[3]-0.5, clusterName, fontsize=22, fontweight='bold',
            horizontalalignment='right', verticalalignment='top')

    
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


def overheads(tint, coadds, dithers, nPerDither):
    overhead = 21 + 5.6 * coadds + 6.5 * (coadds - 1)
    overhead *= dithers * nPerDither
    overhead += 30 * dithers

    integration = tint * coadds * dithers * nPerDither

    # Convert to minutes
    overhead /= 60.0
    integration /= 60.0

    totalTime = overhead + integration

    print 'Total Clock Time: {0:5.0f} min  or  {1:5.1f} hr'.format(totalTime, totalTime/60.)
    print 'Overheads:        {0:5.0f} min  or  {1:5.1f} hr'.format(overhead, overhead/60.)
    print 'Integration Time: {0:5.0f} min  or  {1:5.1f} hr'.format(integration, integration/60.)
