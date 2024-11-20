import numpy as np
import pylab as plt
from astropy.io import fits
from astropy.stats import sigma_clipped_stats

def kast_red(filename, spec_pix_x, box_size_x):
    """
    Plot raw, sky, and sky-subtracted target spectra for
    KAST red side with quick and dirty rectangular box
    extraction.

    Parameters
    ----------
    filename : str
        FITS file name.

    spec_pix_x : int
        Pixel location of the center of the trace in the
        spatial direction (e.g. x for red side).

    box_size_x : int
        Diameter of the rectangular aperture over which
        to median the spectra. The target is extracted from
        this size of box. Skies are subtracted from
        (2 to 3)*boxsize away from the target on each side and
        then averaged before subtraction.
    
    """
    foo = fits.getdata(filename)

    spec_extract_and_plot(foo, spec_pix_x, box_size_x, fignum=1)
    plt.title(f'KAST Red: {filename}')

    return

    
def kast_blue(filename, spec_pix_y, box_size_y):
    """
    Plot raw, sky, and sky-subtracted target spectra for
    KAST blue side with quick and dirty rectangular box
    extraction.

    Parameters
    ----------
    filename : str
        FITS file name.

    spec_pix_y : int
        Pixel location of the center of the trace in the
        spatial direction (e.g. y for blue side).

    box_size_y : int
        Diameter of the rectangular aperture over which
        to median the spectra. The target is extracted from
        this size of box. Skies are subtracted from
        (2 to 3)*boxsize away from the target on each side and
        then averaged before subtraction.
    
    """
    foo = fits.getdata(filename)

    foo = foo.T

    spec_extract_and_plot(foo, spec_pix_y, box_size_y, fignum=2)
    plt.title(f'KAST Blue: {filename}')

    return


def spec_extract_and_plot(img, spec_pix, box_size, fignum=1):
    """
    Extract a long-slit spectra from a rectangular box
    on a 2D spectral image. Plot the raw, sky, and sky-subtracted
    target spectrum.

    Parameters
    ----------
    img : np.array (shape=2)
        Dispersion direction should run along x.
        Spatial direction is along y.

    spec_pix : int
        The pixel at the center of the trace along the y direction.

    box_size : int
        Diameter of the rectangular aperture over which
        to median the spectra. The target is extracted from
        this size of box. Skies are subtracted from
        (2 to 3)*boxsize away from the target on each side and
        then averaged before subtraction.
        
    """
    xlo = int(spec_pix - 0.5 * box_size)
    xhi = int(spec_pix + 0.5 * box_size)
    cutout = img[:, xlo:xhi]
    spec1d, fu, bar = sigma_clipped_stats(cutout, axis=1)
    spec1d = spec1d[::-1]

    xlo_s1 = int(spec_pix + 2 * box_size)
    xhi_s1 = int(spec_pix + 3 * box_size)
    sky_cutout1 = img[:, xlo_s1:xhi_s1]
    sky1d_1, fu, bar = sigma_clipped_stats(sky_cutout1, axis=1)
    sky1d_1 = sky1d_1[::-1]

    xlo_s2 = int(spec_pix - 3 * box_size)
    xhi_s2 = int(spec_pix - 2 * box_size)
    sky_cutout2 = img[:, xlo_s2:xhi_s2]
    sky1d_2, fu, bar = sigma_clipped_stats(sky_cutout2, axis=1)
    sky1d_2 = sky1d_2[::-1]

    sky1d = (sky1d_1 + sky1d_2) * 0.5

    # Sky subtract.
    spec = spec1d - sky1d
    
    plt.figure(fignum, figsize=(10, 4))
    plt.subplots_adjust(left=0.13, top=0.8)
    plt.clf()
    plt.plot(spec1d, label='raw')
    plt.plot(spec, label='target - sky')
    plt.plot(sky1d, label='sky')
    plt.ylabel('Flux (DN)')
    plt.legend()

    return
    
