import math
import numpy as np
from matplotlib.patches import FancyArrow
from matplotlib import pyplot as py

def plot_klf_with_jwst():
    """
    Plot up a simulated KLF for the young star cluster at the Galactic Center
    and show spectroscopic and photometric detections today with Keck and in the
    future with TMT.
    """

    from jlu.gc.imf import bayesian as b

    fitAlpha = 1.7
    fitAge = 3.9e6
    fitMcl = 9.2e3
    fitDist = 7900
    fitLogAge = math.log10(fitAge)

    theAKs = 2.7
    distance = 8000.0

    # Use our best fit IMF for 0.5 - 150 Msun
    # Use Weidner_Kroupa_2004 IMF between 0.1 - 0.5 Msun
    #     Stop at 0.1 Msun as we don't have evolution models below that mass.
    massLimits = np.array([0.1, 0.5, 150])
    powers = np.array([-1.3, -fitAlpha])

    Mcl1 = 3.4e4
    Mcl2 = 3.0e4

    cluster1 = b.model_young_cluster_new(fitLogAge, massLimits=massLimits, 
                                         imfSlopes=powers, clusterMass=Mcl1, 
                                         makeMultiples=True,
                                         AKs=theAKs, distance=distance)

    Mcl1_1_150 = cluster1.systemMasses[cluster1.mass >= 1].sum()
    Mcl1_01_1 = cluster1.systemMasses[cluster1.mass < 1].sum()

    print 'Cluster with multiples:'
    print '     Total Cluster Mass = %d' % Mcl1
    print '    Mass [1 - 150] Msun = %d' % Mcl1_1_150
    print '    Mass [0.1 - 1] Msun = %d' % Mcl1_01_1

    # Assign arbitrary magnitudes for WR stars. Uniformly distributed from Kp=9-11
    idx = np.where(cluster1.isWR == True)[0]
    cluster1.mag[idx] = 9.0 + (np.random.rand(len(idx))*2)

    # Plot the mass luminosity relationship
    py.clf()
    py.semilogy(cluster1.mag, cluster1.mass, 'k.', ms=2, label="Multiples")
    py.xlabel('Kp Magnitude')
    py.ylabel('Stellar Mass (Msun)')
    py.savefig('jwst_mass_luminosity.png')

    kbins = np.arange(9.0, 24, 0.5)

    # Plot the KLF
    py.clf()
    py.hist(cluster1.mag, bins=kbins, histtype='step', linewidth=2, label='Multiples')
    py.gca().set_yscale('log')
    py.xlabel('Kp Magnitude')
    py.ylabel('Number of Stars')
    py.title('Galactic Center Young Cluster')
    py.xlim(9, 28)
    py.ylim(1, 1e4)
    rng = py.axis()

    py.axvline(15.5, color='black', linestyle='--', linewidth=2)
    py.text(15.6, 6e3, r'13 M$_\odot$', horizontalalignment='center')
    ar1 = FancyArrow(15.5, 1e3, -1, 0, width=1e2, color='black', head_length=0.3)
    py.gca().add_patch(ar1)
    py.text(15.3, 7e2, 'Keck\nSpectra', horizontalalignment='right', verticalalignment='top')


    py.axvline(21, color='black', linestyle='-', linewidth=2)
    py.text(21.0, 6e3, r'0.4 M$_\odot$', horizontalalignment='center')
    ar1 = FancyArrow(21, 4e3, -1, 0, width=4e2, color='black', head_length=0.3)
    py.gca().add_patch(ar1)
    py.text(20.8, 3e3, 'TMT\nSpectra', horizontalalignment='right', verticalalignment='top')

    ar1 = FancyArrow(17.8, 10, 0, 13, width=0.1, color='blue', head_length=10)
    py.gca().add_patch(ar1)
    py.text(18, 9, 'Pre-MS\nTurn-On', color='blue',
            horizontalalignment='center', verticalalignment='top')


    py.savefig('jwst_klf_spectral_sensitivity.png')
    
