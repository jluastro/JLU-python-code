from astropy import units as u
from astropy import constants as c
import numpy as np
import pylab as plt
import math

def astro_wobble(m_star=1*u.Msun, m_bh=1*u.Msun,
                 distance=8*u.kpc, semi_major_axis=1*u.AU):
    """
    Calculate the astrometric wobble of the star induced by a dark object
    (e.g. black hole).

    Inputs
    ------
    m_star : flt
      Mass of luminous star whose wobble will be calculated.
      Should have astropy units attached.
    m_bh : flt
      Mass of dark object (e.g. BH). Should have astropy units attached.
    distance : flt
      Distance in kpc with astropy units attached.
    semi_major_axis : flt
      Semi-major axis of binary in AU with astropy units attached.

    Example
    -------
    astro_wobble(m_star=1*u.Msun, m_bh=10*u.Msun,
                 distance=8*u.kpc, semi_major_axis=1*u.AU)
    """
    m_tot = m_bh + m_star
    a = semi_major_axis
    D = distance
    
    # Period is SQRT[ 4 * pi^2 * a^3 / G * Mtot ]
    period = (4. * math.pi**2 * a**3 / (c.G * m_tot))**(1.0/2.0)
    period = period.to(u.yr)

    # On-sky wobble is (a / d) * (Mbh / Mtot)
    wobble = (a / D) * (m_bh / m_tot)
    wobble *= 2.0  # switch to peak to peak
    wobble = wobble.to(u.mas, equivalencies=u.dimensionless_angles())

    if len(period) == 1 and len(wobble) == 1:
        print( 'Period of Binary = {0:.2f}'.format(period) )
        print( 'Astrometric Wobble (peak-to-peak) = {0:.3f}'.format(wobble) )

    return wobble, period

def astro_wobble2(m_star=1*u.Msun, m_bh=1*u.Msun,
                  distance=8*u.kpc, period=1*u.yr):
    """
    Calculate the astrometric wobble of the star induced by a dark object
    (e.g. black hole).

    Inputs
    ------
    m_star : flt
      Mass of luminous star whose wobble will be calculated.
      Should have astropy units attached.
    m_bh : flt
      Mass of dark object (e.g. BH). Should have astropy units attached.
    distance : flt
      Distance in kpc with astropy units attached.
    period : flt
      Period of binary in yr with astropy units attached.

    Example
    -------
    astro_wobble2(m_star=1*u.Msun, m_bh=10*u.Msun,
                 distance=8*u.kpc, period=1*u.yr)
    """
    m_tot = m_bh + m_star
    P = period
    D = distance
    
    # semi-major axis is [ P^2 * Mtot * G / 4 * pi^2 ]^1/3
    a = ( period**2 * m_tot * c.G / (4. * math.pi**2) )**(1./3.)
    a = a.to(u.AU)
    
    # On-sky wobble is (a / d) * (Mbh / Mtot)
    wobble = (a / D) * (m_bh / m_tot)
    wobble *= 2.0  # switch to peak to peak
    wobble = wobble.to(u.mas, equivalencies=u.dimensionless_angles())

    if len(a) == 1 and len(wobble) == 1:
        print( 'Semi-major Axis of Binary = {0:.1f}'.format(a) )
        print( 'Astrometric Wobble (peak-to-peak) = {0:.3f}'.format(wobble) )

    return wobble, period

def plot_period_wobble():
    """
    Plot astrometric wobble vs. period for a range of BH binaries. 
    """
    m_star = 1.0 * u.Msun
    m_bh = np.array([1., 10., 100.]) * u.Msun
    a = np.array([3., 6., 10., 15.]) * u.AU
    P = np.array([1., 5., 10.]) * u.yr
    D = 8. * u.kpc

    # Default index when looping through arrays on other parameters. 
    ii_def = 1

    plt.close(1)
    plt.figure(1, figsize=(4,4))
    plt.clf()

    # Loop through m_bh
    a_dense = np.geomspace(0.1, 1000., 500) * u.AU
    label_P_pos = [12, 11, 6]
    for ii in range(len(m_bh)):
        wob_ii, P_ii = astro_wobble(m_bh=m_bh[ii],
                                    m_star=m_star,
                                    distance=D,
                                    semi_major_axis=a_dense)

        plt.plot(P_ii, wob_ii, marker=None, ls='--', color='blue')

        mid = np.argmin(np.abs(P_ii.value - label_P_pos[ii]))
        #mid = int(len(wob_ii) / 2.) + 100
        
        dy = ((wob_ii[mid+1] - wob_ii[mid]) / u.mas).value
        dx = ((P_ii[mid+1] - P_ii[mid]) / u.yr).value

        angle = np.rad2deg(np.arctan2(dy, dx))

        # annotate with transform_rotates_text to align text and line
        plt.text(P_ii[mid].value, wob_ii[mid].value, 
                 f'M$_{{BH}}$={m_bh[ii].value:.0f} M$_\odot$',
                 ha='center', va='bottom',
                 transform_rotates_text=True,
                 rotation=angle, rotation_mode='anchor',
                 color='blue')

    # Loop through semi-major axis
    m_bh_dense = np.geomspace(0.01, 3e6, 500) * u.Msun
    label_P_pos = [2.5, 4.5, 6, 9]
    for ii in range(len(a)):
        wob_ii, P_ii = astro_wobble(m_bh=m_bh_dense,
                                    m_star=m_star,
                                    distance=D,
                                    semi_major_axis=a[ii])
        
        plt.plot(P_ii, wob_ii, marker=None, ls='-.', color='red')

        mid = np.argmin(np.abs(P_ii.value - label_P_pos[ii]))
        # mid = int(len(wob_ii) / 2.)# - 50
        
        dy = ((wob_ii[mid+1] - wob_ii[mid]) / u.mas).value
        dx = ((P_ii[mid+1] - P_ii[mid]) / u.yr).value

        angle = np.rad2deg(np.arctan2(dy, dx))
        if angle > 90 or angle < -90:
            angle += 180
        
        # annotate with transform_rotates_text to align text and line
        plt.text(P_ii[mid].value, wob_ii[mid].value, 
                 f'a={a[ii].value:.0f} AU',
                 ha='center', va='bottom',
                 transform_rotates_text=True,
                 rotation=angle, rotation_mode='anchor',
                 color='red')
        

    plt.xlabel('Period (yr)')
    plt.ylabel('Astrometric P2V Wobble (mas)')

    plt.xlim(0, 14)
    plt.ylim(0, 5)
    plt.title("BH + Star (1M$_\odot$) Binaries at the GC")
    plt.savefig('gc_bh_astrom_wobble.png')
    
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

def rv_sb1_v1(m_star, m_bh, semi_major_axis):
    """
    Calculate the RV wobble of the star induced by a ~dark companion.

    Inputs
    ------
    m_star : flt
      Mass of luminous star whose wobble will be calculated.
      Should have astropy units attached.
    m_bh : flt
      Mass of dark object (e.g. BH). Should have astropy units attached.
    semi_major_axis : flt
      Semi-major axis of binary in AU with astropy units attached.

    """
    m_tot = m_bh + m_star
    a = semi_major_axis    
    
    # Period is SQRT[ 4 * pi^2 * a^3 / G * Mtot ]
    period = (4. * math.pi**2 * a**3 / (c.G * m_tot))**(1.0/2.0)
    period = period.to(u.yr)

    RV_wobble = 2 * math.pi * a / period
    RV_wobble = RV_wobble.to('km/s')
    
    return RV_wobble, period
    
def rv_sb1_v2(m_star, m_bh, period):
    """
    Calculate the astrometric wobble of the star induced by a dark object
    (e.g. black hole).

    Inputs
    ------
    m_star : flt
      Mass of luminous star whose wobble will be calculated.
      Should have astropy units attached.
    m_bh : flt
      Mass of dark object (e.g. BH). Should have astropy units attached.
    period : flt
      Period of binary in yr with astropy units attached.
    """
    m_tot = m_bh + m_star
    P = period

    # semi-major axis is [ P^2 * Mtot * G / 4 * pi^2 ]^1/3
    a = ( period**2 * m_tot * c.G / (4. * math.pi**2) )**(1./3.)
    a = a.to(u.AU)

    RV_wobble = 2 * math.pi * a / period
    RV_wobble = RV_wobble.to('km/s')
    
    return RV_wobble, a
    
