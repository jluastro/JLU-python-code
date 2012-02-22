import asciidata
import math
import numpy as np
import pylab as py
import pyfits
import scipy.integrate
import scipy.stats
from scipy import interpolate
from gcwork import objects
import ephem

# Define some useful parameters

# AV conversions from
# Rieke & Lebofsky
AJ_AV = 0.282
AH_AV = 0.175
AK_AV = 0.112

def distanceModulus(distance):
    return 5.0 * math.log10( distance ) - 5.0

def calc_kmag(MV, VK, distance, AV):
    DM = distanceModulus(distance)
    AK = AV * AK_AV

    return DM + AK + MV - VK

def calc_hmag(MV, VK, HK, distance, AV):
    DM = distanceModulus(distance)
    AH = AV * AH_AV

    return DM + AH + MV - VK + HK

def calc_jmag(MV, VK, HK, JH, distance, AV):
    DM = distanceModulus(distance)
    AJ = AV * AJ_AV

    return DM + AJ + MV - VK + HK + JH


def ms_jhk(distance, AV):
    print 'Reddining: AV = %.1f  AK = %.1f  AH = %.1f  AJ = %.1f' % \
        (AV, AK_AV*AV, AH_AV*AV, AJ_AV*AV)
    print 'Distance Modulus = %.2f' % distanceModulus(distance)

    # Data taken from Allens Astrophysical Quantities:
    file = '/u/ghezgroup/code/python/papers/spectralTypes.dat'
    tab = asciidata.open(file)
    
    specType = tab[0]._data
    lumClass = tab[1]._data
    absVmag = tab[2].tonumarray()
    Teff = tab[3].tonumarray()
    colorVK = tab[4].tonumarray()
    colorJH = tab[5].tonumarray()
    colorHK = tab[6].tonumarray()

    kmag = calc_kmag(absVmag, colorVK, distance, AV)
    hmag = calc_hmag(absVmag, colorVK, colorHK, distance, AV)
    jmag = calc_jmag(absVmag, colorVK, colorHK, colorJH, distance, AV)

    print 'IR Magnitudes at d = %5d pc with AV = %4.1f: ' % \
        (distance, AV)
    print '%-8s  %5s  %5s  %5s' % ('Type', 'J', 'H', 'K')
    idx5 = []
    idx3 = []
    idx1 = []
    for i in range(len(specType)):
        if (lumClass[i] == 'V'):
            idx5.append(i)
            print '%-8s  %5.2f  %5.2f  %5.2f' % \
                (specType[i] + ' ' + lumClass[i], jmag[i], hmag[i], kmag[i])

        if (lumClass[i] == 'III'):
            idx3.append(i)
        if (lumClass[i] == 'I'):
            idx1.append(i)


    py.clf()
    py.subplot(3, 1, 1)
    py.plot(py.log10(Teff[idx5]), jmag[idx5], 'k.')
    py.plot(py.log10(Teff[idx3]), jmag[idx3], 'ks')
    py.plot(py.log10(Teff[idx1]), jmag[idx1], 'kd')
    ax = py.axis()
    py.ylim(ax[3], ax[2])
    py.xlim(math.log10(6e4), math.log10(2e3))
    py.ylabel('J (mag)')

    py.subplot(3, 1, 2)
    py.plot(py.log10(Teff[idx5]), hmag[idx5], 'k.')
    py.plot(py.log10(Teff[idx3]), hmag[idx3], 'ks')
    py.plot(py.log10(Teff[idx1]), hmag[idx1], 'kd')
    ax = py.axis()
    py.ylim(ax[3], ax[2])
    py.xlim(math.log10(6e4), math.log10(2e3))
    py.ylabel('H (mag)')

    py.subplot(3, 1, 3)
    py.plot(py.log10(Teff[idx5]), kmag[idx5], 'k.')
    py.plot(py.log10(Teff[idx3]), kmag[idx3], 'ks')
    py.plot(py.log10(Teff[idx1]), kmag[idx1], 'kd')
    ax = py.axis()
    py.ylim(ax[3], ax[2])
    py.xlim(math.log10(6e4), math.log10(2e3))
    py.xlabel('Log[Temperature (K)]')
    py.ylabel('K (mag)')

    py.savefig('stars_jhk_%dpc_%dred.png' % (distance, AV))


def clusterProperMotions():
    name = ['W49A', 'Wd1', 'W51', 'Wd2', 'DBS2003-179']
    dist = [11.4,   3.6,   5.5,   2.8,   7.9]
    ra = ['19:10:18', '16:47:04', '19:22:14', '10:23:58', '17:11:32']
    dec = ['+09:06:21', '-45:51:05', '+14:03:09', '-57:45:49', '-39:10:47']

    for ii in range(len(name)):
        obj = ephem.FixedBody()
        obj._ra = ephem.hours(ra[ii])
        obj._dec = ephem.degrees(dec[ii])
        obj._epoch = 2000
        obj.compute()

        gal = ephem.Galactic(obj)

        long = math.degrees(float(gal.long))
        print ''
        print '%-10s  %s %s  l = %.1f' % (name[ii], ra[ii], dec[ii], long)
        properMotions(long, dist[ii], name[ii])

def fieldDensity():
    name = ['W49A', 'Wd1', 'W51', 'Wd2', 'DBS2003-179']
    dist = [11.4,   3.6,   5.5,   2.8,   7.9]
    magLim = [20,    20,    20,  18.6,  18.6]

    root = '/u/jlu/doc/proposals/hst/cycle18/sim_star_fields/'

    d = np.arange(1.0, 12, 0.05) # distance in kpc

    py.clf()
    for ii in range(len(name)):
        simFile = 'sim_for_%s.dat' % name[ii]
        print simFile
        _sim = asciidata.open(simFile)
        age = _sim[1].tonumpy()
        DM = _sim[7].tonumpy()
        AV = _sim[8].tonumpy()
        f125w = _sim[14].tonumpy()
        f140w = _sim[14].tonumpy()
        f160w = _sim[15].tonumpy()

        idx = np.where(f160w <= magLim[ii])[0]
        age = age[idx]
        DM = DM[idx]
        AV = AV[idx]
        f125w = f125w[idx]
        f140w = f140w[idx]
        f160w = f160w[idx]

        # Bin up the stars in distance (actually distance modulus)
        dmBins = np.arange(0, 17, 0.3)
        hist = scipy.stats.histogram2(DM, dmBins)
        distBins = 10.0 * 10**(dmBins/5.0) / 10**3

        # Smooth it out and resample to match
        tck = interpolate.splrep(distBins[:-1], hist[:-1], s=0)
        histNew = interpolate.splev(d, tck, der=0)
        
        py.plot(d, histNew)

    py.legend(name, loc='upper left')
    py.show()
        
        

def properMotions(galacticLat, distance, clusterName):
    """
    Calculate the range of proper motions for different
    possible distances. Assume a constant circular rotation 
    of the galaxy of 220 km/s.

    Inputs:
    galacticLat -- in degrees.
    """
    cc = objects.Constants()

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
    kms_masyr = 1.0e5 * cc.sec_in_yr / (d * cc.cm_in_au)
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
    t = np.arange(2011, 2015.001, 2.0)

    # Proper Motion error
    pmErr = posErr / math.sqrt( ((t - t.mean())**2).sum() )

    print 'Proper Motion Error Calculation:'
    print '   Positional Error: %3.1f mas' % posErr
    print '   Time of Measurements: ', t
    print ''
    print '   Proper Motion Errors: %3.1f mas/yr' % pmErr

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

    outfile = 'hst_clusterPropMot_%s' % clusterName
    py.savefig(outfile + '.png')
    py.savefig(outfile + '.eps')

def byf73_properMotions():
    """
    Calculate the range of proper motions for different
    possible distances. Assume a constant circular rotation 
    of the galaxy of 220 km/s.

    Inputs:
    galacticLat -- in degrees.
    """
    galacticLat = 286.0
    distance = 2.5 # kpc
    clusterName = 'BYF73'

    cc = objects.Constants()

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
    
    d = np.arange(1.0, 10, 0.05) # distance in kpc

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
    kms_masyr = 1.0e5 * cc.sec_in_yr / (d * cc.cm_in_au)
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
    t2 = np.arange(2011, 2013.001, 2.0)
    t4 = np.arange(2011, 2015.001, 2.0)

    # Proper Motion error
    pmErr2 = posErr / math.sqrt( ((t2 - t2.mean())**2).sum() )
    pmErr4 = posErr / math.sqrt( ((t4 - t4.mean())**2).sum() )

    print 'Proper Motion Error Calculation:'
    print '   Positional Error: %3.1f mas' % posErr
    print ''
    print '   Time of Measurements: ', t2
    print '   Proper Motion Errors: %4.2f mas/yr' % pmErr2
    print ''
    print '   Time of Measurements: ', t4
    print '   Proper Motion Errors: %4.2f mas/yr' % pmErr4

    diff = np.abs(d - distance)
    ii = diff.argmin()
    pm_max2 = pm - pm + pm[ii] + (pmErr2/2.0)
    pm_min2 = pm - pm + pm[ii] - (pmErr2/2.0)
    pm_max4 = pm - pm + pm[ii] + (pmErr4/2.0)
    pm_min4 = pm - pm + pm[ii] - (pmErr4/2.0)

    py.fill_between(d, pm_min2, pm_max2, color='grey', alpha=0.3)
    py.fill_between(d, pm_min4, pm_max4, color='grey', alpha=0.5)
    py.plot([distance], [pm[ii]], 'k*', ms=10)

    py.ylim(-9, 0)
    
    ax = py.gca()
    for tick in ax.xaxis.get_major_ticks(): 
        tick.label1.set_fontsize(16) 
        tick.label1.set_fontweight('bold') 
    for tick in ax.yaxis.get_major_ticks(): 
        tick.label1.set_fontsize(16) 
        tick.label1.set_fontweight('bold') 

    outfile = 'hst_clusterPropMot_%s' % clusterName
    py.savefig(outfile + '.png')
    py.savefig(outfile + '.eps')


def intTimes(tint, dithers, orbitMins=54.):
    """
    tint in sec for a single exposure
    """
    tintTotal = tint * dithers
    tintTotalMins = tintTotal / 60.0
    clockTime = dithers * (60. + tint) + 20. * (dithers - 1)
    clockTimeMins = math.ceil( clockTime / 60.0 )
    
    orbits = clockTimeMins / orbitMins

    print 'Total Integration Time: %d sec (%d min)' % \
        (tintTotal, tintTotalMins)
    print 'Total Clock Time:       %d sec (%d min)' % \
        (clockTime, clockTimeMins)
    print 'Total Orbits (no GS):   %.2f' % (orbits)

def orbitAllocations(f153m, f139m, f127m, orbitMins=54.):
    """
    Give the clock time in minutes (without guide star acquisition
    overheads) for each filter for each cluster.
    """
    def gsOverhead(clockTime):
        orbits = clockTime / orbitMins
        orbInt = math.ceil(orbits)
        
        # 6 minutes for first orbit, 4 for subsequent
        orbWithGS = (clockTime + 6 + 4*(orbInt - 1)) / orbitMins

        if orbWithGS > orbInt:
            orbInt += 1
            orbWithGS = (clockTime + 6 + 4*(orbInt - 1)) / orbitMins

        return (orbWithGS, orbInt)

    print 'Orbits for Astrometry Only: '
    print '    %.1f orbits (request %d orbits)' % \
        (gsOverhead(f153m))

    print 'Orbits for Photometry Only: '
    print '    %.1f orbits (request %d orbits)' % \
        (gsOverhead(f139m + f127m))

    print 'Orbits for Combined: '
    print '    %.1f orbits (request %d orbits)' % \
        (gsOverhead(f153m + f139m + f127m))
    

def plotClusterPositions():
    py.clf()
    
