import pyfits
import math
import pylab as py
import numpy as np

def extract1d(cube, center, boxsize=5, combine='sum', header=None):
    """
    Extract a 1D spectrum from an OSIRIS data cube using a 
    rectangular aperture. 

    Input Parameters:
    cube -- Either a FITS file or a data cube (3d array).
    center -- A 2 element vector with the [x, y] value of the
              center of the aperture to extract. NOTE:
              quicklook2 reports the x and y coordinates in 
              reverse! (swap x and y).
    
    Keywork Parameters:
    boxsize -- size of the aperture in pixels. If the boxsize
               is an even number, then the lower left corner
               of the "center" pixel will be taken as the center.
    combine -- The method for creating the 1D spectrum from
               the spectra in the aperture. The choices are:
               sum: Add spectra (default).
               average: Average the spectra.
               median: Median the spectra.
    header -- Use if cube is a 3d array. The header is where
              the wavelength information comes from.
    """
    # First check if the cube is a FITS file or a 3d array
    if type(cube) == type(''):
        cubefile = cube
        cube, hdr = pyfits.getdata(cubefile, header=True)

    else:
        # Cube is already a cube (3D array)
        hdr = header
        if header == None:
            print 'ERROR in extract1d: A header must be passed in when ' + \
                'passing in a cube as a 3d array.'
            return
    
    # Check for a sensible combine method
    if (combine != 'sum' and combine != 'median' and combine != 'average'):
        print 'ERROR in extract1d: Invalid combine method - ' + combine
        return

    # We will describe the image in 3 dimensions:
    #   X 
    #   Y
    #   S - spectral dimension

    # Get the size information in the image (XY) plane.
    xsize = cube.shape[0]
    ysize = cube.shape[1]
    ssize = cube.shape[2]

    # Find the good pixels
    halfSize = boxsize / 2.0
    lowx = math.ceil(center[0] - halfSize)
    uppx = lowx + boxsize # not inclusive of the last pixel
    lowy = math.ceil(center[1] - halfSize)
    uppy = lowy + boxsize # not inclusive of the last pixel

    aperture = cube[lowx:uppx,lowy:uppy]

    # Set extremely negative pixels to 0... I don't know why this happens
    idx = np.where(aperture < -1000)
    aperture[idx] = 0

    # Combine pixels in aperture
    if (combine == 'average'):
        # Average over X and then average over Y
        spectrum = aperture.mean(axis=0).mean(axis=0)

    if (combine == 'median'):
        spectrum = np.median(np.median(aperture, axis=0), axis=0)

    if (combine == 'sum'):
        spectrum = aperture.sum(axis=0).sum(axis=0)


    # Now get the wavelength array
    w0 = hdr['CRVAL1']
    dw = hdr['CDELT1']
    
    wavelength = w0 + dw * np.arange(ssize, dtype=float)
    wavelength *= 1e-3  # convert to microns

    return wavelength, spectrum

def extract1d(cube, center, boxsize=5, combine='sum', header=None):
    """
    Extract a 1D spectrum from an OSIRIS data cube using a 
    rectangular aperture. 

    Input Parameters:
    cube -- Either a FITS file or a data cube (3d array).
    center -- A 2 element vector with the [x, y] value of the
              center of the aperture to extract. NOTE:
              quicklook2 reports the x and y coordinates in 
              reverse! (swap x and y).
    
    Keywork Parameters:
    boxsize -- size of the aperture in pixels. If the boxsize
               is an even number, then the lower left corner
               of the "center" pixel will be taken as the center.
    combine -- The method for creating the 1D spectrum from
               the spectra in the aperture. The choices are:
               sum: Add spectra (default).
               average: Average the spectra.
               median: Median the spectra.
    header -- Use if cube is a 3d array. The header is where
              the wavelength information comes from.
    """
    # First check if the cube is a FITS file or a 3d array
    if type(cube) == type(''):
        cubefile = cube
        cube, hdr = pyfits.getdata(cubefile, header=True)

    else:
        # Cube is already a cube (3D array)
        hdr = header
        if header == None:
            print 'ERROR in extract1d: A header must be passed in when ' + \
                'passing in a cube as a 3d array.'
            return
    
    # Check for a sensible combine method
    if (combine != 'sum' and combine != 'median' and combine != 'average'):
        print 'ERROR in extract1d: Invalid combine method - ' + combine
        return

    # We will describe the image in 3 dimensions:
    #   X 
    #   Y
    #   S - spectral dimension

    # Get the size information in the image (XY) plane.
    xsize = cube.shape[0]
    ysize = cube.shape[1]
    ssize = cube.shape[2]

    # Find the good pixels
    halfSize = boxsize / 2.0
    lowx = math.ceil(center[0] - halfSize)
    uppx = lowx + boxsize # not inclusive of the last pixel
    lowy = math.ceil(center[1] - halfSize)
    uppy = lowy + boxsize # not inclusive of the last pixel

    aperture = cube[lowx:uppx,lowy:uppy]

    # Set extremely negative pixels to 0... I don't know why this happens
    idx = np.where(aperture < -1000)
    aperture[idx] = 0

    # Combine pixels in aperture
    if (combine == 'average'):
        # Average over X and then average over Y
        spectrum = aperture.mean(axis=0).mean(axis=0)

    if (combine == 'median'):
        spectrum = np.median(np.median(aperture, axis=0), axis=0)

    if (combine == 'sum'):
        spectrum = aperture.sum(axis=0).sum(axis=0)


    # Now get the wavelength array
    w0 = hdr['CRVAL1']
    dw = hdr['CDELT1']
    
    wavelength = w0 + dw * np.arange(ssize, dtype=float)
    wavelength *= 1e-3  # convert to microns

    return wavelength, spectrum


def collapse_to_image(cubefile, combine='sum'):
    """
    Extract a 2D image from an OSIRIS data cube by collapsing
    the cube along the wavelength direction.

    Input Parameters:
    cube -- a FITS file containing an OSIRIS data cube
    
    Keywork Parameters:
    combine -- The collapse method for creating the 2D image.
               The choices are:
               sum: Add images at each wavelength (default).
               average: Average the images at each wavelength.
               median: Median the images.
    """
    # First check if the cube is a FITS file or a 3d array
    cubefits = pyfits.open(cubefile)
    cube = cubefits[0].data
    hdr = cubefits[0].header
    badpix = cubefits[2].data
    
    cube = np.ma.masked_where(badpix != 9, cube)

    # Check for a sensible combine method
    if (combine != 'sum' and combine != 'median' and combine != 'average'):
        print 'ERROR in collapse_to_image: Invalid combine method - ' + combine
        return

    # We will describe the image in 3 dimensions:
    #   X 
    #   Y
    #   S - spectral dimension

    # Combine pixels in aperture
    if (combine == 'average'):
        image = cube.mean(axis=2)

    if (combine == 'median'):
        image = np.median(cube, axis=0)

    if (combine == 'sum'):
        image = cube.sum(axis=2)

    # Transpose the image to be in a more typical direction
    image = image.transpose()

    # Save the cube image to a file
    cubeimg_file = cubefile.replace('.fits', '_img.fits')
    pyfits.writeto(cubeimg_file, image.data, header=hdr, clobber=True)
    
    return image

    
def extractTelluric(cubefile, savefile=None, boxsize=5):
    cube, hdr = pyfits.getdata(cubefile, header=True)

    # Find the pixel with the maximum flux
    image = cube.sum(axis=2)

    maxPixel = np.where(image == image.max())

    if (boxsize % 2 is not 1):
        # This is an even number
        print 'ERROR in extractTelluric(): Boxsize must be an odd number.'
        return
    
    w, s = extract1d(cube, maxPixel, header=hdr, boxsize=boxsize)

    # Save to a file
    if savefile != None and savefile != '':
        pyfits.writeto(savefile, s, hdr)

    return w, s


def testTelluricAperture(cubefile):
    apertureSize = np.arange(19) + 2
    snr = np.zeros(len(apertureSize), dtype=float)
    
    for ss in range(len(apertureSize)):
        wave, spec = extractTelluric(cubefile, boxsize=apertureSize[ss])
        snr[ss] = spec[550:600].mean() / spec[550:600].std()

    py.clf()
    py.plot(apertureSize, snr)
    

