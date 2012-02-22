import asciidata
import numpy as np
import pylab as py
from jlu.observe import weather
import pyfits
import math
import datetime


def summarize_observing_conditions(fitsFiles):
    """
    Summarize the observing conditions for a set of fits files. Select
    out the dates and times, range of airmasses, temperature,
    pressure, humidity, and water vapor column.
    """
    count = len(fitsFiles)

    # Here is the data we are going to collect from the fits headers
    year = np.zeros(count, dtype=int)
    month = np.zeros(count, dtype=int)
    day = np.zeros(count, dtype=int)
    hour = np.zeros(count, dtype=int)
    minute = np.zeros(count, dtype=int)
    airmass = np.zeros(count, dtype=float)
    water_column = np.zeros(count, dtype=float)
    
    for ii in range(len(fitsFiles)):
        # Get header info
        hdr = pyfits.getheader(fitsFiles[ii])

        airmass[ii] = float(hdr['AIRMASS'])

        date = hdr['DATE-OBS'].split('-')
        _year = int(date[0])
        _month = int(date[1])
        _day = int(date[2])

        utc = hdr['UTC'].split(':')
        _hour = int(utc[0])
        _minute = int(utc[1])
        _second = int(math.floor(float(utc[2])))

        utc = datetime.datetime(_year, _month, _day, _hour, _minute, _second)
        utc2hst = datetime.timedelta(hours=-10)
        hst = utc + utc2hst

        year[ii] = hst.year
        month[ii] = hst.month
        day[ii] = hst.day
        hour[ii] = hst.hour
        minute[ii] = hst.minute

        # Get the water column in mm of H2O
        water_column[ii] = weather.cso_water_column(_year, _month, _day, 
                                                    _hour, _minute)

    # Now lets fetch the CFHT weather data
    (temperature, pressure, humidity, wind_speed, wind_dir) = \
        weather.cfht_weather_data(year, month, day, hour, minute)

    # Print out a nicely formatted table
    print '%-20s %4s %2s %2s %2s %2s   %4s %4s %5s %5s %4s %4s %4s' % \
        ('Filename', 'Year', 'M', 'D', 'h', 'm', 'AirM', 'H2O', 'Temp', 
         'Press', 'Humi', 'Wind', 'Dir')
    print '%-20s %4s %2s %2s %2s %2s   %4s %4s %5s %5s %4s %4s %4s' % \
        ('HST', '', '', '', '', '', '', 'mm', 'C', 'mbar', '%', 'km/h', 'deg')
    print '%-20s %4s %2s %2s %2s %2s   %4s %4s %5s %5s %4s %4s %4s' % \
        ('--------', '----', '--', '--', '--', '--', '----', '----', '-----', 
         '-----', '----', '----', '----')

    for ii in range(len(fitsFiles)):
        print '%-20s %4d %2d %2d %2d %2d  ' % \
            (fitsFiles[ii], year[ii], month[ii], day[ii], hour[ii], minute[ii]),
        print '%4.2f %4.2f %5.1f %5.1f %4.1f %4.1f %4d' % \
            (airmass[ii], water_column[ii], temperature[ii], pressure[ii],
             humidity[ii], wind_speed[ii], wind_dir[ii])

    # Print out the average values
    print '%-20s %4s %2s %2s %2s %2s   %4s %4s %5s %5s %4s %4s %4s' % \
        ('--------', '----', '--', '--', '--', '--', '----', '----', '-----', 
         '-----', '----', '----', '----')
    print '%-20s %4d %2d %2d %2d %2d  ' % \
        ('Average', year.mean(), month.mean(), day.mean(), hour.mean(), 
         minute.mean()),
    print '%4.2f %4.2f %5.1f %5.1f %4.1f %4.1f %4d' % \
        (airmass.mean(), water_column.mean(), temperature.mean(), 
         pressure.mean(), humidity.mean(), wind_speed.mean(), wind_dir.mean())
    print '%-20s %4d %2d %2d %2d %2d  ' % \
        ('Std. Dev.', year.std(), month.std(), day.std(), hour.std(), 
         minute.std()),
    print '%4.2f %4.2f %5.1f %5.1f %4.1f %4.1f %4d' % \
        (airmass.std(), water_column.std(), temperature.std(), 
         pressure.std(), humidity.std(), wind_speed.std(), wind_dir.std())

        
