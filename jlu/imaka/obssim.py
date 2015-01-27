"""
obssim.py
    Observation Simulator wrapper for imaka PSFs.

    This module lets you simulate complicated scenes of point sources using
    spatially variable PSFs.
"""

import numpy as np
import pylab as py
import pyfits
from scipy.interpolate import RectBivariateSpline
import math
import pdb

class Scene(object):
    """
    Allow the user to specify some scene consisting of a set of
    point sources with specifed pixel positions and fluxes.
    """

    def __init__(self, stars_x, stars_y, stars_f):
        """
        X Position (in pixels)
        Y Position (in pixels)
        Flux (in electrons/sec)
        """
        self.xpos = stars_x
        self.ypos = stars_y
        self.flux = stars_f

        return


class PSF_grid(object):
    """
    Container for a grid of 'imaka PSFs. This function hosts
    all of the interpolation routines such that you can "get_psf"
    anywhere in the focal plane.
    """


    def __init__(self, psf,
                 wave_array=[487, 625, 770, 870, 1020, 1250, 1650, 2120],
                 grid_shape=[11,11]):
        """
        Load up a FITS file that contains at least 1, or possibly a grid
        of PSFs. These are stored and you can interpolate between them
        if necessary.
        """
        self.img_scale = 0.10  # arcseconds per pixel
        
        # Fix wave_array to be a float
        wave_array = np.array(wave_array, dtype=float)
        wave_shape = psf.shape[0]

        if wave_shape != len(wave_array):
            print 'Problem with PSF shape and wave_array shape'
            
        # Reshape the array to get the X and Y positions
        psf = psf.reshape((wave_shape, grid_shape[0], grid_shape[1],
                           psf.shape[2], psf.shape[3]))
        psf = np.swapaxes(psf, 1, 2)

        # scale array = lambda / 2D (Nyquist sampled)
        tel_diam = 2.235  # meters
        psf_scale = wave_array * (206264.8 * 1e-9) / (2.0 * tel_diam) # arcsec / pixel
        
        # Calculate the positions of all these PSFs. We assume that the
        # outermost PSFs are at the corners such that all observed stars
        # are internal to these corners.
        x_pos = np.mgrid[0:grid_shape[0]]
        y_pos = np.mgrid[0:grid_shape[1]]

        # Need to multiply by some of the array size properties.
        # Note that this assumes a pixel scale.
        fov = 10.    # arcmin
        fov *= 60.   # arcsec
        fov /= self.img_scale # pixels

        x_pos *= fov / x_pos[-1]
        y_pos *= fov / y_pos[-1]

        
        self.psf = psf
        self.psf_x = x_pos
        self.psf_y = y_pos
        self.psf_wave = wave_array
        self.wave_shape = wave_shape
        self.grid_shape = grid_shape
        self.psf_scale = psf_scale

        return

    @classmethod
    def from_file(cls, psf_file,
                  wave_array=[487, 625, 770, 870, 1020, 1250, 1650, 2120],
                  grid_shape=[11,11]):
        # 4D array with [wave, grid_idx, flux_x, flux_y]
        psf = pyfits.getdata(psf_file)

        return cls(psf, wave_array=wave_array, grid_shape=grid_shape)

        
    def get_local_psf(self, x, y, wave_idx):
        psf_x = self.psf_x
        psf_y = self.psf_y
        
        # Find the nearest PSF
        xidx_lo = np.where(psf_x < x)[0][-1]
        yidx_lo = np.where(psf_y < y)[0][-1]
        xidx_hi = xidx_lo + 1
        yidx_hi = yidx_lo + 1

        psf_xlo_ylo = self.psf[wave_idx, xidx_lo, yidx_lo]
        psf_xlo_yhi = self.psf[wave_idx, xidx_lo, yidx_hi]
        psf_xhi_ylo = self.psf[wave_idx, xidx_hi, yidx_lo]
        psf_xhi_yhi = self.psf[wave_idx, xidx_hi, yidx_hi]

        dx = 1. * (x - psf_x[xidx_lo]) / (psf_x[xidx_hi] - psf_x[xidx_lo])
        dy = 1. * (y - psf_y[yidx_lo]) / (psf_y[yidx_hi] - psf_y[yidx_lo])

        psf_loc = ((1 - dx) * (1 - dy) * psf_xlo_ylo +
                   (1 - dx) * (  dy  ) * psf_xlo_yhi +
                   (  dx  ) * (1 - dy) * psf_xhi_ylo +
                   (  dx  ) * (  dy  ) * psf_xhi_yhi)
            
        return psf_loc

    def plot_psf_grid(self, wave_idx, psf_size=[50,50]):
        # Chop down the PSFs to the plotting region
        # and at the wavelength requested.
        psf_shape = self.psf.shape[-2:]
        psf_x_lo = int((psf_shape[0] / 2.0) - (psf_size[0] / 2.0))
        psf_y_lo = int((psf_shape[1] / 2.0) - (psf_size[1] / 2.0))
        psf_x_hi = psf_x_lo + psf_size[0]
        psf_y_hi = psf_y_lo + psf_size[1]

        psf = self.psf[wave_idx, :, :, psf_x_lo:psf_x_hi, psf_y_lo:psf_y_hi]
        psf_shape = psf.shape[-2:]
        grid_shape = psf.shape[0:2]

        img = np.zeros((psf_shape[0] * grid_shape[0],
                        psf_shape[1] * grid_shape[1]), dtype=float)

        for xx in range(grid_shape[0]):
            for yy in range(grid_shape[1]):
                xlo = 0 + (xx * psf_shape[0])
                xhi = xlo + psf_shape[0]
                ylo = 0 + (yy * psf_shape[1])
                yhi = ylo + psf_shape[1]
                
                img[xlo:xhi, ylo:yhi] = psf[xx, yy, :, :]

        py.clf()
        py.imshow(img)

        return


        
class Instrument(object):
    def __init__(self, array_size, readnoise, dark_current, gain):
        """
        array_size - in units of pixels
        readnoise - in units of electrons per read
        dark_current - in units of electrons per second
        gain - in units of electrons per DN
        """
        
        self.array_size = array_size
        self.readnoise = readnoise
        self.gain = gain
        self.dark_current = dark_current

        # Here are a bunch of default values setup.

        # Integration time in seconds
        self.itime = 1.0

        # Coadds
        self.coadds = 1

        # Fowler Samples for multi-CDS. This is the number of reads
        # in the beginning and repeated at the end.
        self.fowler = 1

        # Pixel Scale (arcsec / pixel)
        self.scale = 0.1 

        return


class Observation(object):
    def __init__(self, instrument, scene, psf_grid, wave, background):
        """
        background - Background in electrons per second
        """
        # This will be the image in electrons... convert to DN at the end.
        img = np.zeros(instrument.array_size, dtype=float)

        # Add the background and dark current in electrons
        itime_tot = instrument.itime * instrument.coadds
        img += (background + instrument.dark_current) * itime_tot

        # Total readnoise in electrons
        readnoise = instrument.readnoise / math.sqrt(instrument.fowler)

        # i and j are the coordinates into the PSF array. Make it 0 at the center.
        psf_i = np.arange(psf_grid.psf.shape[3]) - (psf_grid.psf.shape[3] / 2)
        psf_j = np.arange(psf_grid.psf.shape[4]) - (psf_grid.psf.shape[4] / 2)

        psf_i_scaled = psf_i * (psf_grid.psf_scale[wave] / instrument.scale)
        psf_j_scaled = psf_j * (psf_grid.psf_scale[wave] / instrument.scale)

        # Add the point sources
        print 'Observation: Adding stars one by one.'
        for ii in range(len(scene.xpos)):
            # Fetch the appropriate interpolated PSF and scale by flux.
            # This is only good to a single pixel.
            psf = psf_grid.get_local_psf(scene.xpos[ii], scene.ypos[ii], wave)
            psf *= scene.flux[ii]

            # Project this PSF onto the detector at this position.
            # This includes sub-pixel shifts and scale changes.

            # Coordinates of the PSF's pixels at this star's position
            psf_i_old = psf_i_scaled + scene.xpos[ii]
            psf_j_old = psf_j_scaled + scene.ypos[ii]

            # Make the interpolation object.
            # Can't keep this because we have a spatially variable PSF.
            psf_interp = RectBivariateSpline(psf_i_old, psf_j_old, psf, kx=1, ky=1)

            # New grid of points to evaluate at for this star.
            xlo = int(psf_i_old[0])
            xhi = int(psf_i_old[-1])
            ylo = int(psf_j_old[0]) + 1
            yhi = int(psf_j_old[-1]) + 1

            # Remove sections that will be off the edge of the image
            if xlo < 0:
                xlo = 0
            if xhi > img.shape[0]:
                xhi = img.shape[0]
            if ylo < 0:
                ylo = 0
            if yhi > img.shape[1]:
                yhi = img.shape[1]
                
            # Interpolate the PSF onto the new grid.
            psf_i_new = np.arange(xlo, xhi)
            psf_j_new = np.arange(ylo, yhi)
            psf_star = psf_interp(psf_i_new, psf_j_new, grid=True)

            # Add the PSF to the image.
            img[xlo:xhi, ylo:yhi] += psf_star
            
        print 'Observation: Finished adding stars.'

        #####
        # ADD NOISE: Up to this point, the image is complete; but noise free.
        #####
        # Add Poisson noise from dark, sky, background, stars.
        img_noise = np.random.poisson(img, img.shape)

        # Add readnoise
        img_noise += np.random.normal(loc=0, scale=readnoise, size=img.shape)
        
        
        self.img = img_noise

            

            
        
        
        

    
