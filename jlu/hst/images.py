import numpy as np
from datetime import datetime as dt
import pyfits
from jlu.util import datetimeUtil as dtUtil

def calc_mean_year(fitsFiles, verbose=False):
    """
    Calculate the average decimal year for a list of HST fits files.

    An example call is:

    year = calc_mean_year(glob.glob('*_flt.fits')

    You can print out the individual years in verbose mode:

    year = calc_mean_year(glob.glob('*_flt.fits', verbose=True)
    
    """
    numObs = 0
    meanYear = 0.0

    nfiles = len(fitsFiles)
    for ii in range(len(fitsFiles)):
        hdr = pyfits.getheader(fitsFiles[ii])

        date = hdr['DATE-OBS']
        time = hdr['TIME-OBS']
        
        dateObj = dt.strptime(date + ' ' + time, '%Y-%m-%d %H:%M:%S')

        year = dtUtil.toYearFraction(dateObj)
        meanYear += year

        if verbose:
            print('{0:12s} {1:12s} {2:8.3f} {3}'.format(date, time, year,
                                                        fitsFiles[ii]))

    meanYear /= nfiles

    if verbose:
        print('*** AVERAGE YEAR = {0:8.3f} ***'.format(meanYear))

    return meanYear

def print_pa_v3(fitsFiles):
    """
    Print position angle (PA_V3) for a set of *_flt.fits images
    """
    for ii in range(len(fitsFiles)):
        hdr = pyfits.getheader(fitsFiles[ii])    

        print('{0}: PA_V3 = {1}'.format(fitsFiles[ii], hdr['PA_V3']))

    return

def total_expTime(fitsFiles, verbose=True):
    """
    Calculate total exposure time for a group of images
    """
    expTime = 0
    
    for ii in range(len(fitsFiles)):
        hdr = pyfits.getheader(fitsFiles[ii])

        if verbose:
            print('{0}: Exposure Time = {1} s'.format(fitsFiles[ii], hdr['EXPTIME']))

        expTime += hdr['EXPTIME']

    print('**** Total Exposure Time: {0} ****'.format(expTime))

    return

    
