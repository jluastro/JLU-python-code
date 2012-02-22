import asciidata
import pyfits
import os
import math

def convert_with_fits(fileroot, datadir='./'):
    """
    fileroot - the root of the Starfinder *.txt (and output *.lis) file.
    sigfits - If this is set to a *_sig.fits file, then the number of frames
              will be pulled from this file indivdiually for each star's
              position. Otherwise the number of frames is set to 1.
    year - Set the year (as a float) of the observations. This supersedes
           the yearFromFITS keyword. (default=None)
    yearFromFits - Set to the original data FITS file to pull the year from
           the header. This is superseded by the year keyword. (default=None)

    """
    # Clean up fileroot
    if fileroot.endswith('.txt'): fileroot = fileroot.replace('.txt', '')

    # Assumes that the file looks like *_0.8_stf.txt (correlation can change)
    # and that the file root just has the ".txt" rmoved.
    parts1 = fileroot.split('/')
    parts2 = parts1[-1].split('_')

    fitsRoot = datadir + '_'.join(parts2[:-2])

    # Look for file to pull the "number of frames" from
    sigFile = fitsRoot + '_sig.fits'
    if not os.path.isfile(sigFile): sigFile = fitsRoot + '_wgt.fits'
    if not os.path.isfile(sigFile): sigFile = fitsRoot + '.sig'
    if not os.path.isfile(sigFile): sigFile = None

    if sigFile:
        sig = pyfits.getdata(sigFile)


    # Calculate the year, only have to do this once
    hdr = pyfits.getheader(fitsRoot + '.fits')
    year = calc_year(hdr)

    # Read input and output files
    _txt = open(fileroot + '.txt', 'r')
    _lis = open(fileroot + '.lis', 'w')

    ii = 0

    while True:
        line1 = _txt.readline()

        # Figure out if we have reached the end of the file
        if (line1 == ''):
            break

        line2 = _txt.readline()

        fields1 = line1.split()
        fields2 = line2.split()
        
        x = float(fields1[0]) # x position
        y = float(fields1[1]) # y position
        f = float(fields1[2]) # flux
        xerr = float(fields1[3])
        yerr = float(fields1[4])
        ferr = float(fields1[5])
        corr = float(fields2[0])

        # Generate a star name
        starname = 'star_%d' % (ii)

        # Do special things if this is the first star.
        if (ii == 0):
            starname = '16ne'
            f0 = f

        
        # Calculate the brightness in magnitudes
        # Arbitrarily sets the first star to m=9.0
        mag = 2.5 * math.log10(f0 / f) + 9.0  
        
        # Signal to noise ratio
        snr = f / ferr

        # Determine the number of frames this star was detected in
        frames = 1
        if sigFile:
            frames = sig[int(y), int(x)]

        _lis.write('%-10s  %6.3f  %8.3f %8.3f %8.3f %11.2f %7.2f  %6d  %9d\n' %
                   (starname, mag, year, x, y, snr, corr, frames, f))
                       

        ii += 1


    _txt.close()
    _lis.close()
        

def calc_year(hdr):
    mjd = hdr['MJD-OBS']
    
    d_days = mjd - 51179.00

    # Correct for leap years by dividing by 365.242
    year = 1999.0000 + d_days / 365.242

    return year
