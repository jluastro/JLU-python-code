import pylab as py
import numpy as np
from jlu.arches import synthetic as arch_syn
from jlu.util import constants
from jlu.imf import imf
from jlu.util import dataUtil
import pickle

propdir = '/Users/jlu/doc/proposals/nsf/aag/2014_lu/'

def prep_clusters():
    arch_iso = arch_syn.load_isochrone(logAge=6.40, AKs=2.4, distance=8000)
    #quin_iso = arch_syn.load_isochrone(logAge=6.55, AKs=3.1, distance=8000)
    #cent_iso = arch_syn.load_isochrone(logAge=6.80, AKs=2.7, distance=8000)

    clust = model_young_cluster(arch_iso, makeMultiples=True)
    _out = open(propdir + 'clust_arhces_multi.pickle', 'w')
    pickle.dump(clust, _out)
    _out.close()

    clust = model_young_cluster(arch_iso, makeMultiples=False)
    _out = open(propdir + 'clust_arches_single.pickle', 'w')
    pickle.dump(clust, _out)
    _out.close()

    return

def plot_mass_luminosity():
    clust_m = pickle.load(open(propdir + 'clust_arches_multi.pickle', 'rb'))
    clust_s = pickle.load(open(propdir + 'clust_arches_single.pickle', 'rb'))

    color_m = clust_m.magJ - clust_m.magKp
    color_s = clust_s.magJ - clust_s.magKp
    
    py.figure(3)
    py.clf()
    py.plot(color_m, clust_m.magKp, 'k.')
    py.gca().invert_yaxis()
    py.ylim(22, 8)
    py.xlabel('J - Kp (mag)')
    py.ylabel('Kp (mag)')
    py.savefig(propdir + 'plots/arches_syn_cmd.png')
    
    py.figure(1, figsize=(7, 6))
    py.clf()
    #py.scatter(clust_s.magKp, clust_s.mass, c='black', s=10)
    py.scatter(clust_m.mass, clust_m.magKp, c=color_m, edgecolor='none', s=10)
    py.gca().set_xscale('log')
    py.gca().invert_yaxis()
    py.ylim(22, 8)
    py.xlim(0.1, 100)
    cbar = py.colorbar()
    cbar.set_label('J - Kp (mag)')
    py.ylabel('Kp (mag)')
    py.xlabel('Mass ' + r'(M$_\odot$)')
    py.savefig(propdir + 'plots/arches_syn_mass_lum.png')

    # Select out a slice of stars at a luminosity of Kp=19 and see what
    # range of masses they have. This represents intrisic uncertainty.
    # Can we resolve this with color?
    dmag = 0.1
    idx_m = np.where((clust_m.magKp > (19-dmag)) & (clust_m.magKp < (19+dmag)))[0]
    idx_s = np.where((clust_s.magKp > (19-dmag)) & (clust_s.magKp < (19+dmag)))[0]

    
    merr1 = 0.04
    merr2 = 0.1
    cerr1 = np.hypot(merr1, merr1)
    cerr2 = np.hypot(merr2, merr2)
    dmag1 = merr1
    dmag2 = merr2
    idx_1 = np.where((clust_m.magKp > (19-dmag1)) & (clust_m.magKp < (19+dmag1)))[0]
    idx_2 = np.where((clust_m.magKp > (19-dmag2)) & (clust_m.magKp < (19+dmag2)))[0]
    
    py.close(5)
    py.figure(5, figsize=(10,10))
    py.clf()
    py.subplots_adjust(left=0.1, bottom=0.1)
    py.subplot(1, 2, 1)
    py.errorbar(color_m[idx_1], clust_m.mass[idx_1], xerr=cerr1, fmt='ko')
    py.xlim(5.1, 6.0)
    py.xlabel('J - Kp (mag)')
    py.ylabel('Mass ' + r'(M$_\odot$)')
    py.title(r'$\sigma_{photo}$ = 0.04 mag')

    py.subplot(1, 2, 2)
    py.errorbar(color_m[idx_2], clust_m.mass[idx_2], xerr=cerr2, fmt='ko')
    py.xlim(5.1, 6.0)
    py.xlabel('J - Kp (mag)')
    # py.ylabel('Mass ' + r'(M$_\odot$)')
    py.title(r'$\sigma_{photo}$ = 0.1 mag')
        
    py.savefig(propdir + 'plots/arches_syn_mass_color.png')

    
    return

def model_young_cluster(iso, makeMultiples=True):
    c = constants

    massLimits = np.array([0.1, 0.5, 150])
    imfSlopes = np.array([-1.3, -2.3])
    clusterMass = 1e4
    MFamp = 0.44
    MFindex = 0.51
    CSFamp = 0.50
    CSFindex = 0.45
    CSFmax = 3
    qMin = 0.01
    qIndex = -0.4

    # Sample a power-law IMF randomly
    results = imf.sample_imf(massLimits, imfSlopes, clusterMass,
                             makeMultiples=makeMultiples,
                             multiMFamp=MFamp, multiMFindex=MFindex,
                             multiCSFamp=CSFamp, multiCSFindex=CSFindex,
                             multiCSFmax=CSFmax,
                             multiQmin=qMin, multiQindex=qIndex)

    mass = results[0]
    isMultiple = results[1]
    compMasses = results[2]
    systemMasses = results[3]

    mag127m = np.zeros(len(mass), dtype=float)
    mag139m = np.zeros(len(mass), dtype=float)
    mag153m = np.zeros(len(mass), dtype=float)
    magJ = np.zeros(len(mass), dtype=float)
    magH = np.zeros(len(mass), dtype=float)
    magKp = np.zeros(len(mass), dtype=float)
    magLp = np.zeros(len(mass), dtype=float)
    temp = np.zeros(len(mass), dtype=float)
    isWR = np.zeros(len(mass), dtype=bool)

    def match_model_mass(theMass):
        dm = np.abs(iso.M - theMass)
        mdx = dm.argmin()

        # Model mass has to be within 2% of the desired mass
        if (dm[mdx] / theMass) > 0.1:
            return None
        else:
            return mdx

    def combine_mag(mag1, mdx_cc, iso_mag):
        f1 = 10**(-mag1 / 2.5)
        f2 = 10**(-iso_mag[mdx_cc] / 2.5)
        new_mag = -2.5 * np.log10(f1 + f2)
        return new_mag

    for ii in range(len(mass)):
        # Find the closest model mass (returns None, if nothing with dm = 0.1
        mdx = match_model_mass(mass[ii])
        if mdx == None:
            continue

        mag127m[ii] = iso.mag127m[mdx]
        mag139m[ii] = iso.mag139m[mdx]
        mag153m[ii] = iso.mag153m[mdx]
        magJ[ii] = iso.magJ[mdx]
        magH[ii] = iso.magH[mdx]
        magKp[ii] = iso.magKp[mdx]
        magLp[ii] = iso.magLp[mdx]
        temp[ii] = iso.T[mdx]
        isWR[ii] = iso.isWR[mdx]

        # Determine if this system is a binary.
        if isMultiple[ii]:
            n_stars = len(compMasses[ii])
            for cc in range(n_stars):
                mdx_cc = match_model_mass(compMasses[ii][cc])
                if mdx_cc != None:
                    mag127m[ii] = combine_mag(mag127m[ii], mdx_cc, iso.mag127m)
                    mag139m[ii] = combine_mag(mag139m[ii], mdx_cc, iso.mag139m)
                    mag153m[ii] = combine_mag(mag153m[ii], mdx_cc, iso.mag153m)
                    magJ[ii] = combine_mag(magJ[ii], mdx_cc, iso.magJ)
                    magH[ii] = combine_mag(magH[ii], mdx_cc, iso.magH)
                    magKp[ii] = combine_mag(magKp[ii], mdx_cc, iso.magKp)
                    magLp[ii] = combine_mag(magLp[ii], mdx_cc, iso.magLp)
                else:
                    print 'Rejected a companion %.2f' % compMasses[ii][cc]
        

    # Get rid of the bad ones
    idx = np.where(temp != 0)[0]
    cdx = np.where(temp == 0)[0]

    if len(cdx) > 0 and verbose:
        print 'Found %d stars out of mass range: Minimum bad mass = %.1f' % \
            (len(cdx), mass[cdx].min())

    mass = mass[idx]
    mag127m = mag127m[idx]
    mag139m = mag139m[idx]
    mag153m = mag153m[idx]
    magJ = magJ[idx]
    magH = magH[idx]
    magKp = magKp[idx]
    magLp = magLp[idx]
    temp = temp[idx]
    isWR = isWR[idx]
    isMultiple = isMultiple[idx]
    systemMasses = systemMasses[idx]
    if makeMultiples:
        compMasses = [compMasses[ii] for ii in idx]
    idx_noWR = np.where(isWR == False)[0]

    mag127m_noWR = mag127m[idx_noWR]
    mag139m_noWR = mag139m[idx_noWR]
    mag153m_noWR = mag153m[idx_noWR]
    magJ_noWR = magJ[idx_noWR]
    magH_noWR = magH[idx_noWR]
    magKp_noWR = magKp[idx_noWR]
    magLp_noWR = magLp[idx_noWR]
    num_WR = len(mag127m) - len(idx_noWR)

    cluster = dataUtil.DataHolder()
    cluster.mass = mass
    cluster.Teff = temp
    cluster.isWR = isWR
    cluster.mag127m = mag127m
    cluster.mag139m = mag139m
    cluster.mag153m = mag153m
    cluster.magJ = magJ
    cluster.magH = magH
    cluster.magKp = magKp
    cluster.magLp = magLp
    cluster.isMultiple = isMultiple
    cluster.compMasses = compMasses
    cluster.systemMasses = systemMasses

    cluster.idx_noWR = idx_noWR
    cluster.mag127m_noWR = mag127m_noWR
    cluster.mag139m_noWR = mag139m_noWR
    cluster.mag153m_noWR = mag153m_noWR
    cluster.magJ_noWR = magJ_noWR
    cluster.magH_noWR = magH_noWR
    cluster.magKp_noWR = magKp_noWR
    cluster.magLp_noWR = magLp_noWR
    cluster.num_WR = num_WR

    # Summary parameters
    cluster.massLimits = massLimits
    cluster.imfSlopes = imfSlopes
    cluster.sumIMFmass = clusterMass
    cluster.makeMultiples = makeMultiples
    cluster.MFamp = MFamp
    cluster.MFindex = MFindex
    cluster.CSFamp = CSFamp
    cluster.CSFindex = CSFindex
    cluster.CSFmax = CSFmax
    cluster.qMin = qMin
    cluster.qIndex = qIndex
            
    return cluster


