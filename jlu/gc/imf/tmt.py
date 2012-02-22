import pylab as py
import numpy as np
import atpy
from matplotlib.patches import FancyArrow

def klf_simulation():
    """
    Plot up a simulated KLF for the young star cluster at the Galactic Center
    and show spectroscopic and photometric detections today with Keck and in the
    future with TMT.
    """
    t = atpy.Table('/u/jlu/work/tmt/tmt_sim_klf_t6.78.txt', type='ascii')
    nrowsOld = len(t)

    # Chop off everything below 0.1 Msun since we don't have models there anyhow
    t = t.where(t['Mass'] >= 0.1)
    nrows = len(t)

    print 'Found %d out of %d stars above 0.1 Msun' % (nrows, nrowsOld)

    # Assign arbitrary magnitudes for WR stars. Uniformly distributed from Kp=9-12
    idx = np.where(t['isWR'] == 'True')[0]
    t['Kp'][idx] = 9.0 + (np.random.rand(len(idx))*3)
    print t['Kp'][idx]

    kbins = np.arange(9.0, 24, 0.5)

    # Plot the KLF
    py.clf()
    py.hist(t['Kp'], bins=kbins, histtype='step', linewidth=2)
    py.gca().set_yscale('log')
    py.xlabel('Kp (d=8 kpc, AKs=2.7)')
    py.ylabel('Number of Stars')
    py.title('Galactic Center Young Cluster')
    py.xlim(9, 23)
    rng = py.axis()

    py.plot([15, 15], [rng[2], rng[3]], 'k--', linewidth=2)
    py.text(15.1, 4e4, r'13 M$_\odot$', horizontalalignment='left')
    ar1 = FancyArrow(15, 10**3, -1, 0, width=1e2, color='black', head_length=0.3)
    py.gca().add_patch(ar1)
    py.text(14.8, 7e2, 'Keck\nSpectra', horizontalalignment='right', verticalalignment='top')


    py.plot([20, 20], [rng[2], rng[3]], 'k-', linewidth=2)
    py.text(20.1, 4e4, r'0.5 M$_\odot$', horizontalalignment='left')
    ar1 = FancyArrow(20, 3e4, -1, 0, width=3e3, color='black', head_length=0.3)
    py.gca().add_patch(ar1)
    py.text(19.8, 2.2e4, 'TMT\nSpectra', horizontalalignment='right', verticalalignment='top')

    ar1 = FancyArrow(18.2, 10, 0, 13, width=0.1, color='blue', head_length=10)
    py.gca().add_patch(ar1)
    py.text(18, 9, 'Pre-MS\nTurn-On', color='blue',
            horizontalalignment='center', verticalalignment='top')

    
    py.savefig('/u/jlu/work/tmt/gc_klf_spectral_sensitivity.png')
    
    
    
