import numpy as np
import pylab as py
import atpy
import pdb
from jlu.astrometry import planets

def orion_lf():
    data_dir = '/Users/jlu/data/orion/hst/orion_proposal/'

    
    foo = atpy.Table(data_dir + 'MATCHUP.MT', type='ascii')

    mag = foo[foo.columns.keys[0]]

    idx = np.where(mag > 0)[0]
    mag = mag[idx]

    py.hist(mag)
    py.xlabel("F775W (mag)")
    py.ylabel("Number of Stars")
    pdb.set_trace()


def planets_ukirt():
    """
    Make a plot of planet mass / planet star vs. semi-major axis.
    """

    # Astrometric Wobble (peak-to-peak) in milli-arcseconds
    wobble = np.array([0.050, 0.4, 2.000])

    # Assumed distance
    distance = 10 # pc

    # Semi-Major axis
    a = np.arange(0.01, 2.0, 0.01)

    # Get all unique combinations
    mass_ratio = np.zeros((len(wobble), len(a)), dtype=float)

    for ww in range(len(wobble)):
        mass_ratio[ww] = planets.mass_ratio_for_wobble(semi_major_axis=a, wobble=wobble[ww])


    ##########
    # Plot mass ratio vs. semi-major axis with constant astrometry curves.
    ##########
    py.clf()
    for ww in range(len(wobble)):
        py.semilogy(a, mass_ratio[ww,:], label='{0:.3f} mas'.format(wobble[ww]))

    py.xlabel('Semi-major Axis (AU)')
    py.ylabel('Mass Ratio')
    py.legend(loc='upper right')
    py.xlim(0, 1.0)
    py.ylim(1e-4, 0.5)
    py.title('d = 10 pc', verticalalignment='bottom')
    arr = py.arrow(0.8, 3.2e-4, 0, 5e-4, width=1.0e-2, head_width=5e-2, head_length=1e-3)
    py.savefig('/Users/jlu/doc/proposals/ifa/14B/ukirt_astro/ast_mass_ratio_vs_a.png')


    ##########
    # Plot mass vs. period with constant astrometry curves.
    # For 50 M-jupiter object.
    ##########
    mjup_in_msun = 1047.93
    m_star = 50.0 / mjup_in_msun

    period = (a**3 / m_star)**(1.0/2.0)

    py.clf()
    for ww in range(len(wobble)):
        py.semilogy(period, mass_ratio[ww,:] * m_star * mjup_in_msun, 
                  label='{0:.3f} mas'.format(wobble[ww]))

    py.xlabel('Period (yr)')
    py.ylabel(r'Planet Mass (M$_J$)')
    py.legend(loc='upper right')
    py.ylim(0.01, 10)
    py.xlim(0, 3)
    py.title(r'M$_{star}$ = 50 M$_J$, d = 10 pc', verticalalignment='bottom')
    arr = py.arrow(2.5, 1.9e-2, 0, 2e-2, width=0.03, head_width=0.15, head_length=0.04)
    py.savefig('/Users/jlu/doc/proposals/ifa/14B/ukirt_astro/ast_mass_vs_p.png')

    ##########
    # Plot planet mass detectable at different 
    # distances.
    ##########
    period = 1.0
    m_star = 0.3
    a = (period**2 * m_star)**(1.0/3.0)

    distance = np.arange(0.0, 400., 1.)
    
    
    # Get all unique combinations
    mass_ratio = np.zeros((len(wobble), len(distance)), dtype=float)

    for ww in range(len(wobble)):
        mass_ratio[ww] = planets.mass_ratio_for_wobble(semi_major_axis=a, distance=distance,
                                                       wobble=wobble[ww])
        
    py.clf()
    for ww in range(len(wobble)):
        py.plot(distance, mass_ratio[ww,:] * m_star * mjup_in_msun, 
                label='{0:.3f} mas'.format(wobble[ww]))

    py.xlabel('Distance (pc)')
    py.ylabel(r'Planet Mass (M$_J$)')
    py.legend(loc='upper right')
    py.ylim(0.01, 10)
    py.xlim(0, 200)
    py.arrow(100, 1.17, 0, 1.0, width=2, head_width=10, head_length=1.0)
    py.title(r'M$_{star}$ = 0.3 M$_{\odot}$, P = 1.00 yr', verticalalignment='bottom')
    py.savefig('/Users/jlu/doc/proposals/ifa/14B/ukirt_astro/ast_mass_vs_d.png')
