import math
import numpy as np

def flux2mag(flux, flux_error, ZP=0):
    """
    Convert flux and flux error to magnitudes.

    ZP - Specify the zeropoint in magnitudes. Default = 0.
    """
    const = 2.5 / math.log(10.0)

    mag = -2.5 * np.log10(flux) + ZP
    mag_error =  const * flux_error / flux

    return (mag, mag_error)

def mag2flux(mag, mag_error, ZP=0):
    """
    Convert magnitude and magnitude errors to fluxes.

    ZP - Specify the zeropoint in magnitudes. Default = 0.
    """
    const = 2.5 / math.log(10.0)

    flux = 10**((mag - ZP) / -2.5)
    flux_error =  mag_error * flux / const

    return (flux, flux_error)
    
    
