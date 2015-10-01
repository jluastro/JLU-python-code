import numpy as np
import pylab as py
import math
import ephem
import os

def plot_airmass(ra, dec, year, months, days, outfile='plot_airmass.png'):
    """
    ra =  R.A. value of target (e.g. '17:45:40.04')
    dec = Dec. value of target (e.g. '-29:00:28.12')
    year = int value of year you want to observe
    months = array of months (integers) where each month will have a curve.
    days = array of days (integers), of same length as months.

    Notes:
    Months are 1-based (i.e. 1 = January). Same for days.
    """
    from pyraf import iraf

    iraf.noao()
    iraf.noao.obsutil()

    month_labels = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                    'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

    pairmass = iraf.noao.obsutil.pairmass
    
    # Observatory (for symbol)
    obs = "keck"

    # Labels and colors for different months.
    labels = []
    label_fmt = '{0:s} {1:d}, {2:d} (HST)'
    for ii in range(len(months)):
        label = label_fmt.format(month_labels[months[ii]-1],
                                 days[ii], year)
        labels.append(label)

    colors = ['r', 'b', 'g', 'c', 'm', 'y']

    # Get sunset and sunrise times on the first day
    scinName = 'skycalc.input'
    scoutName = 'skycalc.output'

    scin = open(scinName, 'w')
    scin.write('m\n')
    scin.write('y %4d %2d %2d a' % (year, months[0], days[0]))
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
                       year=year, month=months[ii], day=days[ii],
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
                airmass[aboveDeck[3]] + 0.4 + (ii*0.1),
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
    py.savefig(outfile)


def plot_moon(ra, dec, year, months, outfile='plot_moon.png'):
    """
    This will plot distance/illumination of moon
    for one specified month
    """
    from pyraf import iraf

    iraf.noao()
    iraf.noao.obsutil()

    obs = iraf.noao.observatory

    # Setup Object
    obj = ephem.FixedBody()
    obj._ra = ephem.hours(ra)
    obj._dec = ephem.degrees(dec)
    obj._epoch = 2000
    obj.compute()
    
    month_labels = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                    'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

    # Labels and colors for different months.
    labels = []
    label_fmt = '{0:s} {1:d} (HST)'
    for ii in range(len(months)):
        label = label_fmt.format(month_labels[months[ii]-1], year)
        labels.append(label)

    sym = ['rD', 'bD', 'gD', 'cD', 'mD', 'yD']
    colors = ['r', 'b', 'g', 'c', 'm', 'y']

    daysInMonth = np.arange(1, 31)

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
        for dd in range(len(daysInMonth)):
            # Set the date and time to midnight
            keck.date = '%d/%d/%d %d' % (year, months[mm],
                                         daysInMonth[dd],
                                         obs.timezone)

            moon.compute(keck)
            obj.compute(keck)
            sep = ephem.separation((obj.ra, obj.dec), (moon.ra, moon.dec))
            sep *= 180.0 / math.pi

            moondist[dd] = sep
            moonillum[dd] = moon.phase

            print 'Day: %2d   Moon Illum: %4.1f   Moon Dist: %4.1f' % \
                  (daysInMonth[dd], moonillum[dd], moondist[dd])

        py.plot(daysInMonth, moondist, sym[mm],label=labels[mm])

        for dd in range(len(daysInMonth)):
            py.text(daysInMonth[dd] + 0.45, moondist[dd]-2, '%2d' % moonillum[dd], 
                    color=colors[mm])

    py.plot([0,31], [30,30], 'k')
    py.legend(loc=2, numpoints=1)
    py.title('Moon distance and % Illumination')
    py.xlabel('Day of Month', fontsize=14)
    py.ylabel('Moon Distance (degrees)', fontsize=14)
    py.axis([0, 31, 0, 180])

    py.savefig(outfile)
