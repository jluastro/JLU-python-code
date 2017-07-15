import numpy as np
import pylab as plt
from astropy import units as u

def proxima_b():
    # Data from
    # https://en.wikipedia.org/wiki/Proxima_Centauri_b
    aplanet = 0.0485  # AU
    mstar = 0.123 # Msun
    mplanet = 1.27 # Mearth (minimum mass)
    distance = 1.295 # pc
    pm_ra = -3775.75 # mas/yr
    pm_dec = 765.54 # mas/yr

    print('Proxima b')
    calc_wobble(aplanet, mstar, mplanet, distance)

    return

def lhs1140():
    mstar = 0.15 # Msun
    mplanet = 6.6 # Mearth (4.8 - 8.5)
    aplanet = 0.09 # AU
    distance = 12 # pc
    pm_ra = 317.0  # mas/yr
    pm_dec = -589.0  # mas/yr
    
    print('LHS 1140b')
    calc_wobble(aplanet, mstar, mplanet, distance)

    return
    

def calc_wobble(aplanet, mstar, mplanet, distance, method='full'):
    """
    aplanet - semi-major axis in AU
    mstar - mass of the host star in Msun
    mplanet - mass of the planet in Earth masses.
    distance - in parsec
    """
    q_aplanet = aplanet * u.AU
    q_mstar = mstar * u.Msun
    q_mplanet = mplanet * u.Mearth

    # Calculate the semi-major axis of the star orbiting around the
    # center of mass.
    q_astar = q_aplanet * (q_mplanet / q_mstar)

    q_astar_in_arcsec = q_astar.to(u.AU) / distance

    print('Semi-major axis of the star wobble: ')
    print('   {0:.5f} AU'.format( q_astar.to(u.AU)) )
    print('   {0:.5f} "'.format( q_astar_in_arcsec) )
    print('   {0:.5f} mas'.format( q_astar_in_arcsec * 1e3) )

    return
    
