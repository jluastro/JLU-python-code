import numpy as np
from astropy import wcs
from astropy.io import fits
import pdb
import astropy.coordinates as coord
import astropy.units as u

def add_wcs_to_nirc2(nirc2_file, x_16C, y_16C):
    """
    x_16C -- pixel position
    y_16C -- pixel position
    """
    # Create a new WCS object.  The number of axes must be set
    # from the start
    w = wcs.WCS(naxis=2)

    scale = 0.00995   # arcsec / pixel
    scale /= 3600.0   # degrees / pixel

    img, hdr = fits.getdata(nirc2_file, header=True)
    pa = hdr['ROTPOSN'] - hdr['INSTANGL'] + 0.252   # degrees
    print pa
    cpa = np.cos(np.radians(-pa))
    spa = np.sin(np.radians(-pa))
    # cd = np.array([[-cpa, -spa], [-spa, cpa]])
    # cd *= scale
    cd = np.array([[-cpa, -spa], [-spa, cpa]])
    #cd *= scale
    print cd
    
    # Set up an "Airy's zenithal" projection
    # Vector properties may be set with Python lists, or Numpy arrays
    w.wcs.crpix = [x_16C, y_16C]
    w.wcs.crval = [266.41709, -29.007574]
    w.wcs.cdelt = np.array([scale, scale])  
    w.wcs.ctype = ['RA---TAN', 'DEC--TAN']
    #w.wcs.cd = cd
    w.wcs.pc = cd

    # Some pixel coordinates of interest.
    pixcrd = np.array([[528.25, 644.94], [438.0, 98.0]])

    # Convert pixel coordinates to world coordinates
    world = w.wcs_pix2world(pixcrd, 1)
    print(world)

    # Convert the same coordinates back to pixel coordinates.
    pixcrd2 = w.wcs_world2pix(world, 1)
    print(pixcrd2)

    c = coord.SkyCoord(ra=world[0, 0] * u.degree, dec=world[0, 1] * u.degree)
    print 'Sgr A*: ', c.to_string('hmsdms')
    c = coord.SkyCoord(ra=world[1, 0] * u.degree, dec=world[1, 1] * u.degree)
    print ' IRS 7: ', c.to_string('hmsdms')
    

    # Now, write out the WCS object as a FITS header
    header = w.to_header()
    
    for key in header.keys():
        hdr[key] = (header[key], header.comments[key])

    # # Save to FITS file
    fits.writeto(nirc2_file.replace('.fits', '_wcs.fits'), img, header=hdr,
                 clobber=True, output_verify='silentfix')

    return hdr
