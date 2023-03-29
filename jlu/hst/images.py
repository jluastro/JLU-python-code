import numpy as np
from datetime import datetime as dt
from astropy.time import Time
#import pyfits
from astropy.io import fits
from jlu.util import datetimeUtil as dtUtil
import pdb
from astropy.coordinates import SkyCoord

def calc_mean_year(fitsFiles, verbose=True):
    """
    Calculate the average decimal year for a list of HST fits files.

    An example call is:

    year = calc_mean_year(glob.glob('*_flt.fits')

    You can print out the individual years in verbose mode:

    year = calc_mean_year(glob.glob('*_flt.fits', verbose=True)
    
    """
    numObs = 0
    meanYear = 0.0
    mean_mjd = 0.0
    
    nfiles = len(fitsFiles)
    for ii in range(len(fitsFiles)):
        hdu = fits.open(fitsFiles[ii])
        hdr = hdu[0].header

        date = hdr['DATE-OBS']
        time = hdr['TIME-OBS']
        
        dateObj = dt.strptime(date + ' ' + time, '%Y-%m-%d %H:%M:%S')

        year = dtUtil.toYearFraction(dateObj)
        meanYear += year

        # Now to calculate MJD
        t = Time('{0} {1}'.format(date, time), format='iso', scale='utc')
        mjd = t.mjd
        mean_mjd += mjd

        if verbose:
            print('{0:12s} {1:12s} {2:8.3f} {3}'.format(date, time, year,
                                                        fitsFiles[ii]))
    
    meanYear /= nfiles
    mean_mjd /= nfiles
    
    if verbose:
        print('*** AVERAGE YEAR = {0:8.4f} ***'.format(meanYear))
        print('**** AVERAGE MJD: {0} *****'.format(mean_mjd))

    return meanYear

def print_pa_v3(fitsFiles, verbose=True):
    """
    Print position angle (PA_V3) for a set of *_flt.fits images
    """
    pav3_arr = []
    for ii in range(len(fitsFiles)):
        hdu = fits.open(fitsFiles[ii])
        hdr = hdu[0].header  
        pav3_arr.append(hdr['PA_V3'])
        
        if verbose:
            print('{0}: PA_V3 = {1}'.format(fitsFiles[ii], hdr['PA_V3']))


    print('**** Average PA_V3: {0} ****'.format(np.mean(np.array(pav3_arr))))
            
    return

def print_pa_aper(fitsFiles, verbose=True):
    """
    Print position angle (PA_APER) for a set of *_flt.fits images
    """
    pa_arr = []
    for ii in range(len(fitsFiles)):
        hdu = fits.open(fitsFiles[ii])
        hdr = hdu[1].header
        pa_arr.append(hdr['PA_APER'])
        
        if verbose:
            print('{0}: PA_APER = {1}'.format(fitsFiles[ii], hdr['PA_APER']))


    print('**** Average PA_APER: {0} ****'.format(np.mean(np.array(pa_arr))))
            
    return

def total_expTime(fitsFiles, verbose=True):
    """
    Calculate total exposure time for a group of images
    """
    expTime = 0
    
    for ii in range(len(fitsFiles)):
        hdu = fits.open(fitsFiles[ii])
        hdr = hdu[0].header

        if verbose:
            print('{0}: Exposure Time = {1} s'.format(fitsFiles[ii], hdr['EXPTIME']))

        expTime += hdr['EXPTIME']

    print('**** Total Exposure Time: {0} ****'.format(expTime))

    return

def print_filters(fitsFiles, verbose=True):
    """
    Print filters of the fitsFiles
    """
    for ii in range(len(fitsFiles)):
        hdu = fits.open(fitsFiles[ii])
        hdr = hdu[0].header

        if verbose:
            print('{0}: Filter = {1}'.format(fitsFiles[ii], hdr['FILTER']))

    return

    
def get_field_center(fitsFiles):
    """
    Get average field center
    """
    ra_arr = []
    dec_arr = []
    for ii in range(len(fitsFiles)):
        hdu = fits.open(fitsFiles[ii])
        hdr = hdu[0].header

        ra_arr.append(hdr['RA_TARG'])
        dec_arr.append(hdr['DEC_TARG'])

    ra_ave = np.mean(np.array(ra_arr))
    dec_ave = np.mean(np.array(dec_arr))
        
    print('**** Average RA: {0} ****'.format(ra_ave))
    print('**** Average DEC: {0} ****'.format(dec_ave))

    coord = SkyCoord(ra_ave, dec_ave, frame='icrs', unit='deg')

    print('RA: {0}h{1}m{2}s'.format(coord.ra.hms.h, coord.ra.hms.m, coord.ra.hms.s))
    print('DEC: {0}d{1}m{2}s'.format(coord.dec.dms.d, coord.dec.dms.m, coord.dec.dms.s))
    
    return


    
    

    
