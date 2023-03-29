import numpy as np
import pylab as py
import math
import ephem
import os
from astropy import units as u
from astropy.time import Time
from astropy.coordinates import SkyCoord, EarthLocation, AltAz, get_sun
import pdb

##################
# Added March 2020
##################
# Workaround for our out-of-date Astropy
# The USNO IERS services (needed by PyLIMA for
# calculating the sidereal time) are being modernized,
# so mirror sites have been set up to obtain the IERS tables.
# Updated Astropy has this internalized.
from astropy.utils.iers import conf as iers_conf
iers_conf.iers_auto_url = 'https://astroconda.org/aux/astropy_mirror/iers_a_1/finals2000A.all'
iers_conf.auto_max_age = None

def plot_airmass(ra, dec, year, months, days, observatory, outfile='plot_airmass.png', date_idx = 0, proposal_cycle = 'A'):
    """
    ra =  R.A. value of target (e.g. '17:45:40.04')
    dec = Dec. value of target (e.g. '-29:00:28.12')
    year = int value of year you want to observe
    months = array of months (integers) where each month will have a curve.
    days = array of days (integers), of same length as months.
    observatory = Either 'keck1' or 'keck2'
    date_idx = Index of day to use for twilight dashed lines.  Defaults to first day.
    proposal_cycle = A or B to determine where the text with dates is. Default to A.

    Notes:
    Months are 1-based (i.e. 1 = January). Same for days.
    """
    # Setup the target
    target = SkyCoord(ra, dec, unit=(u.hour, u.deg), frame='icrs')

    # Setup local time.
    utc_offset = -10 * u.hour   # Hawaii Standard Time
        
    month_labels = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                    'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

    # Observatory (for symbol)
    obs = 'keck'
    keck = EarthLocation.of_site('keck')

    # Labels and colors for different months.
    labels = []
    label_fmt = '{0:s} {1:d}, {2:d} (HST)'
    for ii in range(len(months)):
        label = label_fmt.format(month_labels[months[ii]-1],
                                 days[ii], year)
        labels.append(label)

    colors = ['r', 'b', 'g', 'c', 'm', 'y']
    heights = [1.45, 1.35, 1.25, 1.15]

    # Get sunset and sunrise times on the first specified day
    midnight = Time('{0:d}-{1:d}-{2:d} 00:00:00'.format(year, months[date_idx], days[date_idx])) - utc_offset
    delta_midnight = np.arange(-12, 12, 0.01) * u.hour
    times = midnight + delta_midnight
    altaz_frame = AltAz(obstime=times, location=keck)
    sun_altaz = get_sun(times).transform_to(altaz_frame)

    sun_down = np.where(sun_altaz.alt < -0*u.deg)[0]
    twilite = np.where(sun_altaz.alt < -12*u.deg)[0]
    sunset = delta_midnight[sun_down[0]].value
    sunrise = delta_midnight[sun_down[-1]].value
    twilite1 = delta_midnight[twilite[0]].value
    twilite2 = delta_midnight[twilite[-1]].value

    # Get the half-night split times
    splittime = twilite1 + ((twilite2 - twilite1) / 2.0)

    print( 'Sunrise %4.1f   Sunset %4.1f  (hours around midnight HST)' % (sunrise, sunset))
    print( '12-degr %4.1f  12-degr %4.1f  (hours around midnight HST)' % (twilite1, twilite2))

    py.close(3)
    py.figure(3, figsize=(7, 7))
    py.clf()
    py.subplots_adjust(left=0.15)
    for ii in range(len(days)):
        midnight = Time('{0:d}-{1:d}-{2:d} 00:00:00'.format(year, months[ii], days[ii])) - utc_offset
        delta_midnight = np.arange(-7, 7, 0.2) * u.hour
        times = delta_midnight.value

        target_altaz = target.transform_to(AltAz(obstime=midnight + delta_midnight,
                                                 location=keck))
        
        airmass = target_altaz.secz

        # Trim out junk where target is at "negative airmass"
        good = np.where(airmass > 0)[0]
        times = times[good]
        airmass = airmass[good]

        # Find the points beyond the Nasmyth deck. Also don't bother with anything above sec(z) = 3
        transitTime = times[airmass.argmin()]

        if observatory == 'keck2':
            belowDeck = (np.where((times >= transitTime) & (airmass >= 1.8)))[0]
            aboveDeck = (np.where(((times >= transitTime) & (airmass < 1.8)) |
                            (times < transitTime)))[0]
        else:
            belowDeck = (np.where((times <= transitTime) & (airmass >= 1.8)))[0]
            aboveDeck = (np.where(((times <= transitTime) & (airmass < 1.8)) |
                            (times > transitTime)))[0]
            
        py.plot(times[belowDeck], airmass[belowDeck], colors[ii] + 'o', mfc='w', mec=colors[ii], ms=12)
        py.plot(times[aboveDeck], airmass[aboveDeck], colors[ii] + 'o', mec=colors[ii], ms=12)
        py.plot(times, airmass, colors[ii] + '-')
        
        if proposal_cycle == 'A':
            text_offset = 1.1
        else:
            text_offset = 1.3
        py.text(-3.5,
                text_offset + (ii*0.1),
                labels[ii], color=colors[ii], fontsize=20)

    # Make observatory name nice for title
    if observatory == "keck1":
        observatory = 'Keck I'
    if observatory == "keck2":
        observatory = 'Keck II'

    py.tick_params(labelsize=20)      
    py.title('Obs. RA = 18:00, DEC = -30:00 from %s' %observatory, fontsize = 20, y=1.02)
#    py.title('Observing RA = %s, DEC = %s from %s' % (ra, dec, observatory), fontsize=14)
    py.xlabel('Local Time in Hours (0 = midnight)', fontsize=20)
    py.ylabel('Air Mass', fontsize=20)

    loAirmass = 1
    hiAirmass = 2.2

    # Draw on the 12-degree twilight limits
    py.axvline(splittime, color='k', linestyle='--')
    py.axvline(twilite1 + 0.5, color='k', linestyle='--')
    py.axvline(twilite2, color='k', linestyle='--')

    py.axis([sunset, sunrise, loAirmass, hiAirmass])
    py.savefig(outfile)


def plot_moon(ra, dec, year, months, outfile='plot_moon.png'):
    """
    This will plot distance/illumination of moon
    for one specified month
    """
    # Setup local time.
    utc_offset = 10 * u.hour   # Hawaii Standard Time
        
    month_labels = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                    'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

    # Observatory (for symbol)
    keck_loc = EarthLocation.of_site('keck')
    keck = ephem.Observer()
    keck.long = keck_loc.lon.value
    keck.lat = keck_loc.lat.value
    
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
    label_fmt = '{0:s} {1:d}'
    for ii in range(len(months)):
        label = label_fmt.format(month_labels[months[ii]-1], year)
        labels.append(label)

    sym = ['rD', 'bD', 'gD', 'cD', 'mD', 'yD']
    colors = ['r', 'b', 'g', 'c', 'm', 'y']

    daysInMonth = np.arange(1, 31)

    moondist = np.zeros(len(daysInMonth), dtype=float)
    moonillum = np.zeros(len(daysInMonth), dtype=float)

    moon = ephem.Moon()

    py.close(3)
    py.figure(3, figsize=(7, 7))
    py.clf()
    py.subplots_adjust(left=0.15)

    for mm in range(len(months)):
        for dd in range(len(daysInMonth)):
            # Set the date and time to midnight
            keck.date = '%d/%d/%d %d' % (year, months[mm],
                                         daysInMonth[dd],
                                         utc_offset.value)

            moon.compute(keck)
            obj.compute(keck)
            sep = ephem.separation((obj.ra, obj.dec), (moon.ra, moon.dec))
            sep *= 180.0 / math.pi

            moondist[dd] = sep
            moonillum[dd] = moon.phase

            print( 'Day: %2d   Moon Illum: %4.1f   Moon Dist: %4.1f' % \
                  (daysInMonth[dd], moonillum[dd], moondist[dd]))

        # py.plot(daysInMonth, moondist, sym[mm],label=labels[mm])
        py.scatter(daysInMonth, moondist, c=moonillum, s = 250, cmap='Greys_r',edgecolors=colors[mm], label=labels[mm])

        # for dd in range(len(daysInMonth)):
        #     py.text(daysInMonth[dd] + 0.45, moondist[dd]-2, '%2d' % moonillum[dd], 
        #             color=colors[mm], fontsize=14)

    py.plot([0,31], [30,30], 'k')
    legend = py.legend(loc=2, numpoints=1, fontsize=20, labelcolor=colors, handlelength=0, markerscale=0)
    [handle.set_facecolor('w') for handle in legend.legendHandles]
    py.title('Moon distance and %% Illumination \n (RA = %s, DEC = %s)' % (ra, dec), fontsize=20)
    py.xlabel('Day of Month (UT)', fontsize = 20)
    py.ylabel('Moon Distance (degrees)', fontsize = 20)
    py.tick_params(labelsize=20)  
    py.axis([0, 31, 0, 200])
    py.yticks([0,30,60,90,120,150,180])
    py.xticks([0,5,10,15,20,25,30])
    # py.show()
    py.savefig(outfile)
