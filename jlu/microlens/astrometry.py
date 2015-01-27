import pylab as py
import numpy as np
from jlu.util import constants as c
import pdb

def astrometric_shift(mass, d_lens, d_source):
    """
    mass in solar masses
    d_lens in kpc
    d_source in kpc

    Return astrometric shift in milli-arcseconds.
    """
    m_cgs = mass * c.Msun * 1.0e3

    DL = d_lens * 1.0e3 * c.cm_in_pc
    DS = d_source * 1.0e3 * c.cm_in_pc

    term1 = 4.0 * c.G * m_cgs / (c.c * 1.0e5)**2

    term2 = (DS - DL) / (DL * DS)

    e_radius = (term1 * term2)**0.5

    e_radius *= 206265.  # convert to arcsec

    astrom_shift = 0.354 * e_radius  # asec
    astrom_shift *= 1.0e3 # mas

    return astrom_shift


