import pylab as py
import numpy as np
import asciidata
from gcwork import objects
import pdb
import math

def quint_data():
    dir = '/u/jlu/data/quint/2010wfc3/from_jay/'

    file_f127 = dir + 'MATCHUP.QUINT.F127M.txt'
    file_f139 = dir + 'MATCHUP.QUINT.F139M.txt'

    f127 = read_jay(file_f127, 'F127M')
    f139 = read_jay(file_f139, 'F139M')

    idx = np.where((f127.x > 0) & (f139.x > 0) & 
                   (f127.me < 0.1) & (f139.me < 0.1))[0]
    
    f127 = trim_jay(f127, idx)
    f139 = trim_jay(f139, idx)
    
    color = f127.m - f139.m
    colorErr = np.hypot(f127.me, f139.me)

    # CMD
    py.clf()
    py.errorbar(color, f127.m, fmt='k.', xerr=colorErr, yerr=f127.me, ms=3)
    rng = py.axis()
    py.ylim(rng[3], rng[2])
    py.xlabel('F127M - F139M (mag)')
    py.ylabel('F127M (mag)')
    py.title('Quintuplet')
    py.savefig('quint_cmd_f127_f139.png')

    # CMD - no error bars
    py.clf()
    py.plot(color, f127.m, 'k.', ms=3)
    rng = py.axis()
    py.ylim(rng[3], rng[2])
    py.xlabel('F127M - F139M (mag)')
    py.ylabel('F127M (mag)')
    py.title('Quintuplet')
    py.savefig('quint_cmd_f127_f139_noerr.png')

    scale = 120.0

    # Positional Errors
    py.clf()
    py.semilogy(f127.m, f127.xe*scale, 'r.', label='X')
    py.semilogy(f127.m, f127.ye*scale, 'b.', label='Y')
    py.legend(loc='upper left')
    py.xlabel('F127M (mag)')
    py.ylabel('Positional Errors (mas)')
    py.title('Quintuplet')
    py.savefig('quint_poserr_f127.png')

    py.clf()
    py.semilogy(f139.m, f139.xe*scale, 'r.', label='X')
    py.semilogy(f139.m, f139.ye*scale, 'b.', label='Y')
    py.legend(loc='upper left')
    py.xlabel('F139M (mag)')
    py.ylabel('Positional Errors (mas)')
    py.title('Quintuplet')
    py.savefig('quint_poserr_f139.png')

    # Photometric Errors
    py.clf()
    py.semilogy(f127.m, f127.me, 'r.', label='X')
    py.xlabel('F127M (mag)')
    py.ylabel('Photometric Errors (mag)')
    py.title('Quintuplet')
    py.savefig('quint_magerr_f127.png')

    py.clf()
    py.semilogy(f139.m, f139.me, 'r.', label='X')
    py.xlabel('F139M (mag)')
    py.ylabel('Photometric Errors (mag)')
    py.title('Quintuplet')
    py.savefig('quint_magerr_f139.png')


def read_jay(filename, filter):
    zp = {'F127M': 24.65, 'F139M': 24.49}

    tab = asciidata.open(filename)

    d = objects.DataHolder()

    d.x = tab[0].tonumpy()
    d.y = tab[1].tonumpy()
    d.m = tab[2].tonumpy()

    d.xe = tab[3].tonumpy()
    d.ye = tab[4].tonumpy()
    d.me = tab[5].tonumpy()

    # Photometrically calibrate.
    d.m += zp[filter]

    return d

def trim_jay(d, idx):
    d.x = d.x[idx]
    d.y = d.y[idx]
    d.m = d.m[idx]

    d.xe = d.xe[idx]
    d.ye = d.ye[idx]
    d.me = d.me[idx]

    return d
    

def plot_wd1():
    """
    Use the Hubble Legacy Archive Sextractor output on Wd1 to plot up
    a luminosity function. Remember that saturated stars are not included.
    """

    filename = '/u/jlu/data/Wd1/hst/2005wfc/HLA/HST_10172_01_ACS_WFC_F814W/'
    filename += 'HST_10172_01_ACS_WFC_F814W_sexphot_trm.cat'
    foo = asciidata.open(filename)
    
    x = foo[0].tonumpy()
    y = foo[1].tonumpy()
    m = foo[5].tonumpy()
    me = foo[6].tonumpy()

    # Get everything with proper aperture corrected photometry.
    idx = np.where(m < 90)[0]

    x = x[idx]
    y = y[idx]
    m = m[idx]
    me = me[idx]

    hbins = np.arange(19, 27, 0.5)

    py.clf()
    py.hist(m, normed=True, bins=hbins, histtype='step')

    # Plot up a histogram of the off-field population
    xmin = 3100
    ymin = 3300
    idx = np.where((x > xmin) & (y > ymin))[0]

    py.hist(m[idx], normed=True, bins=hbins, histtype='step')

    py.clf()
    py.semilogy(m, me, 'k.')


def wd1_pm_errors():
    cc = objects.Constants()

    times1 = np.array([2005.0, 2010.0, 2012.0, 2014.0])
    times2 = np.array([2010.0, 2012.0, 2014.0])
    
    poserr = 1.0 # mas

    pmerr1 = poserr / math.sqrt( ((times1 - times1.mean())**2).sum() )
    pmerr2 = poserr / math.sqrt( ((times2 - times2.mean())**2).sum() )

    # Conversion from mas/yr to km/s
    dist = 4000.0 # pc
    masyr_to_kms = dist * cc.cm_in_au / (10**5 * 10**3 * cc.sec_in_yr)
    print 'Conversion: 1 mas/yr = %.2f km/s' % masyr_to_kms

    print 'Proper Motion Errors:'
    print '  positional error  = %.1f mas' % poserr
    print '  epochs of obs.    =  ', times1
    print '  proper motion err = %.2f mas/yr (%.1f km/s)' % \
        (pmerr1, pmerr1 * masyr_to_kms)

    print 'Proper Motion Errors:'
    print '  positional error  = %.1f mas' % poserr
    print '  epochs of obs.    =  ', times2
    print '  proper motion err = %.2f mas/yr (%.1f km/s)' % \
        (pmerr2, pmerr2 * masyr_to_kms)

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
