import numpy as np
import pylab as py
import pdb

mjup_in_msun = 1047.93
au_in_pc = 206264.8
mas_in_radian = 206264.8 * 1e3


def astro_wobble(m_planet=1, m_star=1, distance=10, semi_major_axis=1):
    """
    Calculate the astrometric wobble of the star induced by the planet.
    m_planet - in jupiter masses
    m_star - in solar masses
    distance - in parsec
    semi_major_axis - in AU
    """
    m_p_msun = m_planet / mjup_in_msun
    a_pc = semi_major_axis / au_in_pc

    wobble = a_pc * m_p_msun / (distance * m_star) # in radians
    wobble *= mas_in_radian
    wobble *= 2.0  # switch to peak to peak

    period = (semi_major_axis**3 / m_star)**(1.0/2.0)

    print( 'Period of Planet = {0:.2f} yr'.format(period) )
    print( 'Astrometric Wobble (peak-to-peak) = {0:.3f} mas'.format(wobble) )

    return


def mass_ratio_for_wobble(semi_major_axis=1, distance=10, wobble=1):
    """
    wobble - in milli-arcseconds (peak-to-peak)
    """
    a_pc = semi_major_axis / au_in_pc

    # Switch from peak-to-peak to half-amplitude
    wobble /= 2.0

    # Convert to radians
    wobble_radians = wobble / mas_in_radian

    # Calculate the mass ratio
    mass_ratio = wobble_radians * distance / a_pc

    return mass_ratio
