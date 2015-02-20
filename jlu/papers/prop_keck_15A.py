import math
import ephem
import asciidata
import pylab as py
import numpy as np
import os
import pyfits
from gcreduce import gcutil
from gcwork import objects
from jlu.util import img_scale


def gc_plotAirmass():
    from pyraf import iraf

    iraf.noao()
    iraf.noao.obsutil()

    pairmass = iraf.noao.obsutil.pairmass
    
    # coordinates to GC
    ra = "17:45:40.04"
    dec = "-29:00:28.12"
    obs = "keck"

#     months = np.array([10, 11, 1])
#     days   = np.array([20, 20, 23])
#     years  = np.array([2010, 2010, 2011])
#     colors = ['r', 'b', 'g']
#     labels = ['Oct 20, 2010 (HST)', 'Nov 20, 2010 (HST)',
#               'Jan 23, 2011 (HST)']

    months = np.array([4, 5, 6, 7])
    days   = np.array([20, 20, 20, 20])
    years  = np.array([2015, 2015, 2015, 2015])
    colors = ['r', 'b', 'c', 'g']
    labels = ['Apr 20, 2015 (HST)', 
              'May 20, 2015 (HST)',
              'Jun 20, 2015 (HST)',
              'Jul 20, 2015 (HST)',
              ]

    # Get sunset and sunrise times on the first day
    scinName = 'skycalc.input'
    scoutName = 'skycalc.output'

    scin = open(scinName, 'w')
    scin.write('m\n')
    scin.write('y %4d %2d %2d a' % (years[0], months[0], days[0]))
    scin.write('Q\n')
    scin.close()

    # Spawn skycalc
    os.system('skycalc < %s > %s' % (scinName, scoutName))

    # Now read in skycalc data
    scout = open(scoutName, 'r')
    lines = scout.readlines()

    for line in lines:
        fields = line.split()

        if (len(fields) < 3):
            continue

        if (fields[0] == 'Sunset'):
            sunset = float(fields[5]) + float(fields[6]) / 60.0
            sunset -= 24.0
            sunrise = float(fields[9]) + float(fields[10]) / 60.0

        if (fields[0] == '12-degr'):
            twilite1 = float(fields[2]) + float(fields[3]) / 60.0
            twilite1 -= 24.0
            twilite2 = float(fields[6]) + float(fields[7]) / 60.0
            print twilite1, twilite2

        if ((fields[0] == 'The') and (fields[1] == 'sun')):
            darkTime = (twilite2 - twilite1) - 0.5 # 0.5=LGS checkout
            splittime = twilite1 + 0.5 + darkTime/2
            if (splittime > 24.0):
                splittime -= 24.0

    print 'Sunrise %4.1f   Sunset %4.1f' % (sunrise, sunset)
    print '12-degr %4.1f  12-degr %4.1f' % (twilite1, twilite2)
    

    py.clf()
    for ii in range(len(days)):
        foo = pairmass(ra=ra, dec=dec, observatory=obs, listout="yes",
                       timesys="Standard", Stdout=1, resolution=2,
                       year=years[ii], month=months[ii], day=days[ii],
                       wx1=-7, wx2=7)

        entries = foo[5:]
        times = np.zeros(len(entries), dtype=float)
        airmass = np.zeros(len(entries), dtype=float)

        for ee in range(len(entries)):
            vals = entries[ee].split()

            tt = vals[0].split(':')
            hour = float(tt[0])
            minu = float(tt[1])

            times[ee] = hour + (minu / 60.0)
            airmass[ee] = float(vals[1])


        # Wrap the times around
        idx = (np.where(times > 12))[0]
        ndx = (np.where(times <= 12))[0]
        times = np.concatenate((times[idx]-24, times[ndx]))
        airmass = np.concatenate((airmass[idx], airmass[ndx]))

        # Find the points beyond the Nasmyth deck
        transitTime = times[airmass.argmin()]
        belowDeck = (np.where((times > transitTime) & (airmass >= 1.8)))[0]
        aboveDeck = (np.where(((times > transitTime) & (airmass < 1.8)) |
                           (times < transitTime)))[0]

        py.plot(times[belowDeck], airmass[belowDeck], colors[ii] + 'o', mfc='w')
        py.plot(times[aboveDeck], airmass[aboveDeck], colors[ii] + 'o')
        py.plot(times, airmass, colors[ii] + '-')

        py.text(times[aboveDeck[3]] - 0.3,
                airmass[aboveDeck[3]] + 0.4 + (ii*0.2),
                labels[ii], color=colors[ii])
            

    py.title('ORION Source n (RA = %s, DEC = %s)' % (ra, dec), fontsize=12)
    py.xlabel('Local Time in Hours (0 = midnight)')
    py.ylabel('Air Mass')

    loAirmass = 1
    hiAirmass = 3

    # Draw on the 12-degree twilight limits
    py.plot([splittime, splittime], [loAirmass, hiAirmass], 'k--')
    py.plot([twilite1 + 0.5, twilite1 + 0.5], [loAirmass, hiAirmass], 'k--')
    py.plot([twilite2, twilite2], [loAirmass, hiAirmass], 'k--')

    py.axis([sunset, sunrise, loAirmass, hiAirmass])
    py.savefig('/u/jlu/doc/proposals/ifa/15A/gc_airmass.png')


def gc_plotMoon():
    """
    This will plot distance/illumination of moon
    for one specified month
    """
    from pyraf import iraf

    iraf.noao()
    iraf.noao.obsutil()

    obs = iraf.noao.observatory

    # coordinates to GC
    ra = "17:45:40.04"
    dec = "-29:00:28.12"

    # Setup Object
    obj = ephem.FixedBody()
    obj._ra = ephem.hours(ra)
    obj._dec = ephem.degrees(dec)
    obj._epoch = 2000
    obj.compute()
    
    # Setup dates of observations
    months = np.array([4, 5, 6, 7])
    days   = np.array([1, 1, 1, 1])
    years  = np.zeros(len(days)) + 2015
    colors = ['r', 'b', 'c', 'g']
    labels = ['Apr 20, 2015 (HST)', 
              'May 20, 2015 (HST)',
              'Jun 20, 2015 (HST)',
              'Jul 20, 2015 (HST)',
              ]

    years  = np.zeros(len(days)) + 2010
    sym = ['rD', 'bD']
    colors = ['r', 'b']
    labels = ['Oct', 'Nov']

    daysInMonth = np.arange(31)

    # Setup the observatory info
    iraf.noao.obsutil.obs(command="set", obsid="keck")
    keck = ephem.Observer()
    keck.long = -obs.longitude
    keck.lat = obs.latitude

    moondist = np.zeros(len(daysInMonth), dtype=float)
    moonillum = np.zeros(len(daysInMonth), dtype=float)

    moon = ephem.Moon()
 
    py.clf()

    for mm in range(len(months)):
        for dd in daysInMonth:
            # Set the date and time to midnight
            keck.date = '%d/%d/%d %d' % (years[0], months[mm], days[0]+dd,
                                         obs.timezone)

            moon.compute(keck)
            obj.compute(keck)
            sep = ephem.separation((obj.ra, obj.dec), (moon.ra, moon.dec))
            sep *= 180.0 / math.pi

            moondist[dd] = sep
            moonillum[dd] = moon.phase

            print 'Day: %2d   Moon Illum: %4.1f   Moon Dist: %4.1f' % \
                  (dd, moonillum[dd], moondist[dd])

        py.plot(daysInMonth, moondist, sym[mm],label=labels[mm])

        for dd in daysInMonth:
            py.text(dd+0.45, moondist[dd]-2, '%2d' % moonillum[dd], 
                    color=colors[mm])

    py.plot([0,31],[30,30],'k')
    py.legend(loc=2,numpoints=1)
    py.title('Moon distance and % Illumination')
    py.xlabel('Date (UTC)', fontsize=14)
    py.ylabel('Moon Distance (degrees)', fontsize=14)
    py.axis([0, 31, 0, 180])

    py.savefig('orion_moondist.ps')
    py.savefig('orion_moondist.png')

def orion_findTipTilt():
    catalog = '/u/jlu/doc/proposals/keck/uc/10B/orion/daRio_2009_table2.fits'

    f = pyfits.open(catalog)
    tbl = f[1].data

    # First select out only those with V and I magnitudes
    mask = tbl.field(12) > 0
    tbl = tbl[mask]

    mask = tbl.field(16) > 0
    tbl = tbl[mask]

    srcID = tbl.field(0)
    raHour = tbl.field(1)
    raMin = tbl.field(2)
    raSec = tbl.field(3)
    decDeg = tbl.field(5)
    decMin = tbl.field(6)
    decSec = tbl.field(7)
    vmag = tbl.field(12)
    imag = tbl.field(16)

    ra = (raHour + raMin/60. + raSec/3600.) * 15.0
    dec = (decDeg + decMin/60. + decSec/3600.) * -1.0  # negative dec for orion

    raRadians = np.radians(ra)
    decRadians = np.radians(dec)

    ra0 = (5 + 35/60. + 14.5/3600.) * 15.0
    dec0 = (5 + 22/60. + 30/3600.) * -1.0

    ra0radians = math.radians(ra0)
    dec0radians = math.radians(dec0)

    # Separation in arcsec
    raDiff = (ra0 - ra) * np.cos(dec0radians) * 3600.0
    decDiff = (dec0 - dec) * 3600.0
    sep = np.hypot(raDiff, decDiff)

    sid = sep.argsort()

    for ss in range(len(sid)):
        ii = sid[ss]

        if sep[ii] > 60:
            break
        
        if vmag[ii] > 14:
            continue

        print 'Source %4d  sep = %5.1f  V = %4.1f  I = %4.1f  V-I = %4.1f' % \
            (srcID[ii], sep[ii], vmag[ii], imag[ii], vmag[ii]-imag[ii])


    # Plot them up
    py.clf()

    idx1 = np.where(vmag <= 14)[0]
    idx2 = np.where(vmag > 14)[0]
    py.scatter(raDiff[idx2], decDiff[idx2], (-imag[idx2]+20)**2, 'b')
    py.scatter(raDiff[idx1], decDiff[idx1], (-imag[idx1]+20)**2, 'r')
    py.axis([60, -60, -60, 60])

    return raDiff, decDiff, imag

def lp_sensitivity():
    """
    Read in a number of GC L' data sets and plot the 
    SNR vs. mag with number of frames plotted.
    """
    rootDir = '/u/jlu/doc/proposals/keck/uc/10B/orion/'
    files = [rootDir + 'mag04jul_lp_rms.lis',
             rootDir + 'mag05jullgs_lp_rms.lis',
             rootDir + 'mag06jullgs_lp_rms.lis']
    legends = ['04jul', '05jullgs', '06jullgs']

    py.clf()
    
    magStep = 1.0
    magBins = np.arange(6, 18, magStep)
    snrAvg = np.zeros(len(magBins))
    for ff in range(len(files)):
        tab = asciidata.open(files[ff])

        mag = tab[1].tonumpy()
        snr = tab[7].tonumpy()
        cnt = tab[9].tonumpy()

        for mm in range(len(magBins)-1):
            magLo = magBins[mm] - magStep/2.0
            magHi = magBins[mm] + magStep/2.0
            idx = np.where((mag > magLo) & (mag <= magHi))[0]

            snrAvg[mm] = snr[idx].mean()

        py.semilogy(magBins, snrAvg)

        legends[ff] += ': N = %d' % cnt[0]
    py.legend(legends)
    py.show()
    


def m31_plotWithHST(rotateFITS=True):
    cc = objects.Constants()

    # Load up the images
    kFile = '/u/jlu/data/m31/09sep/combo/mag09sep_m31_kp.fits'
    f330File = '/u/jlu/work/m31/nucleus/align/m31_0330nm_rot.fits'
    f435File = '/u/jlu/work/m31/nucleus/align/m31_0435nm_rot.fits'

    # Load up the NIRC2 image
    k = pyfits.getdata(kFile)
    f330 = pyfits.getdata(f330File)
    f435 = pyfits.getdata(f435File)

    m31K = np.array([701.822, 583.696])
    scaleK = 0.00995

    img = np.zeros((k.shape[0], k.shape[1], 3), dtype=float)
    img[:,:,0] = img_scale.linear(k, scale_min=10, scale_max=3800)
    img[:,:,1] = img_scale.linear(f435, scale_min=0.01, scale_max=1.8)
    img[:,:,2] = img_scale.linear(f330, scale_min=0, scale_max=0.16)

    # Axes
    xaxis = (np.arange(img.shape[0], dtype=float) - (m31K[0] - 1.0))*scaleK
    yaxis = (np.arange(img.shape[1], dtype=float) - (m31K[1] - 1.0))*scaleK
    
    py.clf()
    py.imshow(img, aspect='equal', 
              extent=[xaxis[0],xaxis[-1],yaxis[0],yaxis[-1]],
              interpolation='nearest')
    py.plot([0], [0], 'c+', linewidth=2, ms=10, mew=2)
    py.xlabel('R.A. Offset from M31* (arcsec)', fontsize=16)
    py.ylabel('Dec. Offset from M31* (arcsec)', fontsize=16)
    py.title('Blue = F330W, Green = F435W, Red = NIRC2-K\'')

    # Overplot the OSIRIS fields of view.
    rec1xTmp = np.array([-1.6, -1.6, 1.6, 1.6, -1.6])
    rec1yTmp = np.array([-0.3, 0.5, 0.5, -0.3, -0.3])
    rec2xTmp = np.array([-1.6, -1.6, 1.6, 1.6, -1.6])
    rec2yTmp = np.array([-1.0, -0.2, -0.2, -1.0, -1.0])
    rec3xTmp = np.array([-1.6, -1.6, 1.6, 1.6, -1.6])
    rec3yTmp = np.array([0.4, 1.2, 1.2, 0.4, 0.4])

    # rotate
    pa = math.radians(-34.0)
    cospa = math.cos(pa)
    sinpa = math.sin(pa)
    rec1x = rec1xTmp * cospa - rec1yTmp * sinpa
    rec1y = rec1xTmp * sinpa + rec1yTmp * cospa
    rec2x = rec2xTmp * cospa - rec2yTmp * sinpa
    rec2y = rec2xTmp * sinpa + rec2yTmp * cospa
    rec3x = rec3xTmp * cospa - rec3yTmp * sinpa
    rec3y = rec3xTmp * sinpa + rec3yTmp * cospa


    py.plot(rec1x, rec1y, 'g-', linewidth=2)
    py.plot(rec2x, rec2y, 'g--', linewidth=2)
    py.plot(rec3x, rec3y, 'g--', linewidth=2)

    # Label
    py.text(0.1, 0.1, 'P3\nBH+A stars', color='cyan', fontweight='bold',
            verticalalignment='bottom', horizontalalignment='left',
            fontsize=16)
    py.text(0.5, -0.3, 'P2\nperiapse', color='white', fontweight='bold',
            verticalalignment='top', horizontalalignment='center',
            fontsize=16)
    py.text(-0.7, 0.5, 'apoapse\nP1', color='white', fontweight='bold',
            verticalalignment='bottom', horizontalalignment='center',
            fontsize=16)

    limit = 2.0
    py.axis([-limit, limit, -limit, limit])
    py.savefig('m31_hst_nirc2_rgb.png')
    py.savefig('m31_hst_nirc2_rgb.eps')
    py.show()


def m31_plotMoonForGC():
    """
    This will plot distance/illumination of moon
    for one specified month
    """
    from pyraf import iraf

    iraf.noao()
    iraf.noao.obsutil()

    obs = iraf.noao.observatory

    # coordinates to Orion source n from Gomez et al. 2005 (J2000)
    ra = "17:45:40.0"
    dec = "-29:00:28.0"

    # Setup Object
    obj = ephem.FixedBody()
    obj._ra = ephem.hours(ra)
    obj._dec = ephem.degrees(dec)
    obj._epoch = 2000
    obj.compute()
    
    # Setup dates of observations
    months = np.array([8, 9])
    days   = np.array([1])
    years  = np.zeros(len(days)) + 2010
    sym = ['rD', 'bD']
    colors = ['r', 'b']
    labels = ['Aug', 'Sep']

    daysInMonth = np.arange(31)

    # Setup the observatory info
    obs(command="set", obsid="keck")
    keck = ephem.Observer()
    keck.long = -obs.longitude
    keck.lat = obs.latitude

    moondist = np.zeros(len(daysInMonth), dtype=float)
    moonillum = np.zeros(len(daysInMonth), dtype=float)

    moon = ephem.Moon()
 
    py.clf()

    for mm in range(len(months)):
        for dd in daysInMonth:
            # Set the date and time to midnight
            keck.date = '%d/%d/%d %d' % (years[0], months[mm], days[0]+dd,
                                         obs.timezone)

            moon.compute(keck)
            obj.compute(keck)
            sep = ephem.separation((obj.ra, obj.dec), (moon.ra, moon.dec))
            sep *= 180.0 / math.pi

            moondist[dd] = sep
            moonillum[dd] = moon.phase

            print 'Day: %2d   Moon Illum: %4.1f   Moon Dist: %4.1f' % \
                  (dd, moonillum[dd], moondist[dd])

        py.plot(daysInMonth, moondist, sym[mm],label=labels[mm])

        for dd in daysInMonth:
            py.text(dd+0.45, moondist[dd]-2, '%2d' % moonillum[dd], 
                    color=colors[mm])

    py.plot([0,31],[30,30],'k')
    py.legend(loc=2,numpoints=1)
    py.title('Moon distance and % Illumination')
    py.xlabel('Date (UTC)', fontsize=14)
    py.ylabel('Moon Distance (degrees)', fontsize=14)
    py.axis([0, 31, 0, 180])

    py.savefig('m31_gc_moondist.ps')
    py.savefig('m31_gc_moondist.png')
