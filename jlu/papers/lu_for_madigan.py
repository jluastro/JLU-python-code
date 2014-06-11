import numpy as np
import pylab as py
import math
from jlu.gc.imf import bayesian as b

fitAlpha = 1.7
fitAge = 3.9e6
fitMcl = 9.2e3
fitDist = 7900
fitLogAge = math.log10(fitAge)
theAKs = 2.7

workDir = '/Users/jlu/work/gc/madigan/'
distance = 8000.0

def plot_gc_yng_klf_imf(age):
    """
    Plot up a simulated KLF, IMF, mass-luminosity relationship for the young star
    cluster at the Galactic Center.
    """

    from jlu.gc.imf import bayesian as b
    
    # Use our best fit IMF for 0.5 - 150 Msun
    # Use Weidner_Kroupa_2004 IMF between 0.1 - 0.5 Msun
    #     Stop at 0.1 Msun as we don't have evolution models below that mass.
    massLimits = np.array([0.1, 0.5, 150])
    powers = np.array([-1.3, -fitAlpha])

    Mcl1 = 3.4e4
    Mcl2 = 3.0e4

    logAge = math.log10(age * 1e6)

    cluster1 = b.model_young_cluster_new(logAge, massLimits=massLimits, imfSlopes=powers,
                                         clusterMass=Mcl1, makeMultiples=True,
                                         AKs=theAKs, distance=distance)
    cluster2 = b.model_young_cluster_new(logAge, massLimits=massLimits, imfSlopes=powers,
                                         clusterMass=Mcl2, makeMultiples=False,
                                         AKs=theAKs, distance=distance)

    Mcl1_1_150 = cluster1.systemMasses[cluster1.mass >= 1].sum()
    Mcl2_1_150 = cluster2.systemMasses[cluster2.mass >= 1].sum()
    Mcl1_01_1 = cluster1.systemMasses[cluster1.mass < 1].sum()
    Mcl2_01_1 = cluster2.systemMasses[cluster2.mass < 1].sum()

    print 'Cluster with multiples:'
    print '     Total Cluster Mass = %d' % Mcl1
    print '    Mass [1 - 150] Msun = %d' % Mcl1_1_150
    print '    Mass [0.1 - 1] Msun = %d' % Mcl1_01_1
    print 'Cluster with singles:'
    print '     Total Cluster Mass = %d' % Mcl2
    print '    Mass [1 - 150] Msun = %d' % Mcl2_1_150
    print '    Mass [0.1 - 1] Msun = %d' % Mcl2_01_1

    # Assign arbitrary magnitudes for WR stars. Uniformly distributed from Kp=9-11
    idx1 = np.where(cluster1.isWR == True)[0]
    cluster1.mag[idx1] = 9.0 + (np.random.rand(len(idx1))*2)
    idx2 = np.where(cluster2.isWR == True)[0]
    cluster2.mag[idx2] = 9.0 + (np.random.rand(len(idx2))*2)

    # Plot the mass luminosity relationship
    py.clf()
    py.semilogy(cluster1.mag, cluster1.mass, 'k.', ms=2, label="Multiples", mec='black')
    py.semilogy(cluster2.mag, cluster2.mass, 'b.', ms=2, label='Single', mec='blue')
    py.xlabel('Kp Magnitude')
    py.ylabel('Stellar Mass (Msun)')
    py.title('Age: {0:.1f} Myr'.format(age))
    py.legend(loc='upper right', numpoints=1)
    py.ylim(0.1, 200)
    py.savefig('{0}plots/mass_luminosity_{1:.1f}Myr.png'.format(workDir, age))

    kbins = np.arange(9.0, 24, 0.5)

    # Plot the KLF
    py.clf()
    py.hist(cluster1.mag, bins=kbins, histtype='step', linewidth=2, label='Multiples', color='blue')
    py.hist(cluster2.mag, bins=kbins, histtype='step', linewidth=2, label='Singles', color='black')
    py.gca().set_yscale('log')
    py.xlabel('Kp Magnitude')
    py.ylabel('Number of Stars')
    py.title('Age: {0:.1f} Myr'.format(age))
    py.legend(loc='upper left')
    py.xlim(7, 22)
    py.ylim(1, 1e4)
    rng = py.axis()

    outRoot = '{0}plots/klf_{1:.1f}Myr'.format(workDir, age)
    py.savefig('{0}.png'.format(outRoot))
