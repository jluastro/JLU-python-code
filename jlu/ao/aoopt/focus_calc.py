import numpy as np
import pylab as plt
import math

def defocus_keck_ao(focus_coeff, wavelength=2.12, tel_diam=10.0, f_num=15):
    """
    Parameters
    ---------------
    focus_coeff : float (nm)
        Coefficient of the defocs term assuming that the optical path difference
        from defocus is given by:
            c * sqrt(3) * (2 \rho^2  - 1)

    Optional Parameters
    ---------------
    wavelength : float (microns)
    tel_diam : float (m)
    f_num : float
    """
    pix_scale = 206265. / (tel_diam * f_num)   # arcsec / m
    pix_scale *= 1.0e3 / 1e6   # mas / micron
    print("Pixel Scale: {0:.2f} mas / micron".format(pix_scale))

    fwhm_z0_asec = 0.21 * (wavelength / tel_diam)  # arcsec
    fwhm_z0_micron = 1.02 * wavelength * f_num     # microns
    str_fmt = "FWHM at z=0 mm: {0:.1f} mas = {1:.1f} micron"
    print(str_fmt.format(fwhm_z0_asec*1e3, fwhm_z0_micron))

    # Handy quantities:
    sqrt_2ln2 = math.sqrt(2.0 * math.log(2.0))
    
    # Gaussian Beam Approximation:
    # The beam radius at focus:
    w0 = fwhm_z0_micron / sqrt_2ln2  # micron
    print("Beam Radius at z=0 mm: w0 = {0:.1f} micron".format(w0))

    # Delta-focus length set by rayliegh criteria
    z_R = math.pi * w0**2 * 1e-3 / wavelength  # mm
    print("Focus shift at RC: z_R = {0:.2f} mm".format(z_R))
    print()
    
    # Peak-to-valley of defocused wavefront
    p2v = 2.0 * math.sqrt(3) * focus_coeff  # nm
    print("Peak-to-valley for c = {0:.0f}: {1:.1f} nm".format(focus_coeff, p2v))

    # The focal shift distance (along z)
    z = 8.0 * p2v * f_num**2   # nm
    z *= 1e-6 # mm
    print("Focus length for   c = {0:.0f}: {1:.2f} mm".format(focus_coeff, z))

    # Defocused beam radius
    w = w0 * math.sqrt(1.0 + (z / z_R)**2)   # micron
    print("Beam radius for    c = {0:.0f}: {1:.2f} micron".format(focus_coeff, w))

    # Defocused beam FWHM
    fwhm_z_micron = w * sqrt_2ln2  # micron
    fwhm_z_mas = fwhm_z_micron * pix_scale     # mas
    str_fmt = "Beam FWHM for      c = {0:.0f}: {1:.2f} micron = {2:.2f} mas"    
    print(str_fmt.format(focus_coeff, fwhm_z_micron, fwhm_z_mas))

    print("Fractional Change in Beam FWHM: {0:.2f}".format(fwhm_z_micron / fwhm_z0_micron))

    return

    
    
    
