from astropy.io import fits as pyfits
import math
import pylab as py
import numpy as np
# from PyAstronomy import pyasl
import pdb

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
            print('ERROR in extract1d: A header must be passed in when ' + \
                'passing in a cube as a 3d array.')
            return
    
    # Check for a sensible combine method
    if (combine != 'sum' and combine != 'median' and combine != 'average'):
        print('ERROR in extract1d: Invalid combine method - ' + combine)
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
        #if header == None:
        #    print 'ERROR in extract1d: A header must be passed in when ' + \
        #        'passing in a cube as a 3d array.'
        #    return
    
    # Check for a sensible combine method
    if (combine != 'sum' and combine != 'median' and combine != 'average'):
        print('ERROR in extract1d: Invalid combine method - ' + combine)
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
    cube = np.nan_to_num(cube)

    # Check for a sensible combine method
    if (combine != 'sum' and combine != 'median' and combine != 'average'):
        print('ERROR in collapse_to_image: Invalid combine method - ' + combine)
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
    pyfits.writeto(cubeimg_file, image.data, header=hdr, clobber=True,output_verify='warn')
    
    return image

    
def extractTelluric(cubefile, savefile=None, boxsize=5):
    cube, hdr = pyfits.getdata(cubefile, header=True)

    # Find the pixel with the maximum flux
    image = cube.sum(axis=2)

    maxPixel = np.where(image == image.max())

    if (boxsize % 2 is not 1):
        # This is an even number
        print('ERROR in extractTelluric(): Boxsize must be an odd number.')
        return
    
    w, s = extract1d(cube, maxPixel, header=hdr, boxsize=boxsize)

    # Save to a file
    if savefile != None and savefile != '':
        pyfits.writeto(savefile, s, hdr, output_verify='warn')

    return w, s


def testTelluricAperture(cubefile):
    apertureSize = np.arange(19) + 2
    snr = np.zeros(len(apertureSize), dtype=float)
    
    for ss in range(len(apertureSize)):
        wave, spec = extractTelluric(cubefile, boxsize=apertureSize[ss])
        snr[ss] = spec[550:600].mean() / spec[550:600].std()

    py.clf()
    py.plot(apertureSize, snr)
    
def cubeDopplerShift(cubeFile=None,dv=None):
    # shifts spectra by a given velocity (dv, in km/s)
    # positive velocity: redshifts spectrum
    # negative velocity: blueshifts spectrum
    
    cubefits = pyfits.open(cubeFile)

    cube = cubefits[0].data
    hdr = cubefits[0].header
    errors = cubefits[1].data
    quality = cubefits[2].data

    w0 = hdr['CRVAL1']
    dw = hdr['CDELT1']
    wavelength = w0 + dw * np.arange(cube.shape[2], dtype=float)

    newCube = np.zeros(cube.shape, dtype=float)
    for ii in range(cube.shape[0]):
        for jj in range(cube.shape[1]):
            if cube[ii,jj,:].mean() != 0:
                nflux, nwave = pyasl.dopplerShift(wavelength, cube[ii,jj,:], dv, edgeHandling="firstlast")
                newCube[ii,jj,:] = nflux

    outFile = cubeFile.replace('.fits', '_vlsrcorr.fits')
    try:
        pyfits.writeto(outFile, newCube, header=hdr, clobber=True,output_verify='ignore')
    except:
        pass
        #pdb.set_trace()
    pyfits.append(outFile,errors)
    pyfits.append(outFile,quality)
    #pyfits.append(outFile,nframes)

def cubeEdgeMask(cubeFile=None):
    # set the bad pixel mask (ext 2) for the edge spaxels to 0 (=bad, vs. 9=good)
    # edge spaxel coordinates are hard coded for the broadband filters
    # masks just the outermost spaxel along all edges

    cubefits = pyfits.open(cubeFile)

    cube = cubefits[0].data
    hdr = cubefits[0].header
    err = cubefits[1].data
    quality = cubefits[2].data

    newqual = quality

    # outermost edges first
    newqual[:,0,:] = 0
    newqual[:,62,:] = 0
    newqual[0,:,:] = 0
    newqual[17,:,:] = 0
    # short edges
    newqual[0:3,1,:] = 0
    newqual[10:18,61,:] = 0
    # bottom edge, long axis
    newqual[1,16:-1,:] = 0
    newqual[2,32:-1,:] = 0
    newqual[3,48:-1,:] = 0
    # top edge, long axis
    newqual[14,0:16,:] = 0
    newqual[15,0:32,:] = 0
    newqual[16,0:48,:] = 0

    outFile = cubeFile.replace('.fits', '_mask.fits')
    pyfits.writeto(outFile, cube, header=hdr, clobber=True,output_verify='ignore')
    pyfits.append(outFile,err)
    pyfits.append(outFile,newqual)
