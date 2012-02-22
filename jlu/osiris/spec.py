import numpy as np
import pylab as py
import pyfits

def plotSpectrum(fitsfile):
    """
    Plot a 1D spectrum
    """
    data, head = pyfits.getdata(fitsfile, header=True)

    if (len(data.shape) > 1):
        print 'Data is not 1D... cannot plot.'
        return

    pixels = np.arange(0, len(data))
    wave = head['CRVAL1'] + head['CDELT1'] * (pixels - head['CRPIX1'] - 1)
    waveunit = head['CUNIT1']

    py.figure(2, figsize=(10, 6))
    py.clf()
    py.plot(wave, data)
    py.xlabel('Wavelength (%s)' % waveunit)
    py.title(fitsfile)
    
    
    

