import numpy as np
import pylab as py
from jlu.nirc2 import synthetic
import pdb

# parallactic distance from Sandstrom+ 2007
distance = 389.0 

# 1 Myr age from Lagrange+ 2004
logAge = 6.0

# Extinction AV=30 from Lagrange+ 2004 towards source n
AV = 30

# Metallicityfrom Biazzo+ 2011
# Actually this is ignored for now and we are using solar metallicity
metallicity = -0.3


def make_synthetic():
    synthetic.nearIR(distance, logAge, 
                     redlawClass=synthetic.RedLawRomanZuniga07)


def cmd():
    synFile = 'syn_nir_d389.0_a600.dat'
    results = synthetic.load_nearIR_dict(synFile)

    # Lets trim down to the extinction value of interest.
    AKs_AV = 0.062 # from Nishiyama+ 2008... GC, but good enough?
    AKs = AV * AKs_AV

    idx = np.argmin(np.abs(results['AKs'] - AKs))

    numAKs = len(results['AKs'])
    numTeff = len(results['Teff'])

    for key in results.keys():
        if ((len(results[key].shape) == 2) and 
            (results[key].shape[1] == numAKs)):

            results[key] = results[key][:,idx]
        else:
            if len(results[key]) == numAKs:
                results[key] = results[key][idx]

    HKp = results['H'] - results['Kp']
    KpLp = results['Kp'] - results['Lp']

    # Fetch a few points at specific masses
    massPoints = np.array([30, 10, 5, 2, 1, 0.5])
    massIndices = np.zeros(len(massPoints))
    massLabels = []
    for mm in range(len(massPoints)):
        mdx = np.argmin(np.abs(results['mass'] - massPoints[mm]))

        massIndices[mm] = mdx
        massPoints[mm] = results['mass'][mdx]
        massLabels.append('{0:4.1f} Msun'.format(massPoints[mm]))
        

    # Plots
    py.clf()
    py.plot(HKp, results['H'])
    rng = py.axis()
    py.xlim(0, 2)
    py.ylim(rng[3], rng[2])
    py.xlabel('H - Kp')
    py.ylabel('H')
    py.title('Orion BN/KL Region')
    for mm, mmLab in zip(massIndices, massLabels):
        py.plot(HKp[mm], results['H'][mm], 'kx')
        py.text(HKp[mm], results['H'][mm], mmLab)
    py.savefig('orion_cmd_hkp_h.png')

    py.clf()
    py.plot(HKp, results['Kp'])
    rng = py.axis()
    py.xlim(0, 2)
    py.ylim(rng[3], rng[2])
    py.xlabel('H - Kp')
    py.ylabel('Kp')
    py.title('Orion BN/KL Region')
    for mm, mmLab in zip(massIndices, massLabels):
        py.plot(HKp[mm], results['Kp'][mm], 'kx')
        py.text(HKp[mm], results['Kp'][mm], mmLab)
    py.savefig('orion_cmd_hkp_kp.png')

    py.clf()
    py.plot(KpLp, results['Kp'])
    rng = py.axis()
    py.xlim(0, 2)
    py.ylim(rng[3], rng[2])
    py.xlabel('Kp - Lp')
    py.ylabel('Kp')
    py.title('Orion BN/KL Region')
    for mm, mmLab in zip(massIndices, massLabels):
        py.plot(KpLp[mm], results['Kp'][mm], 'kx')
        py.text(KpLp[mm], results['Kp'][mm], mmLab)
    py.savefig('orion_cmd_kplp_kp.png')

    py.clf()
    py.semilogx(results['Teff'], results['logL'])
    rng = py.axis()
    py.xlim(rng[1], rng[0])
    py.xlabel('Teff')
    py.ylabel('log(L)')
    py.title('Orion BN/KL Region')
    for mm, mmLab in zip(massIndices, massLabels):
        py.plot(results['Teff'][mm], results['logL'][mm], 'kx')
        py.text(results['Teff'][mm], results['logL'][mm], mmLab)
    py.savefig('orion_hr.png')
    
    
    # Print to a text file for Breann
    _out = open('orion_model_isochrone.dat', 'w')
    
    fmt = '#{0:>9s}  {1:>10s}  {2:>10s}  {3:>10s}  {4:>10s} {5:>10s} {6:>10s} {7:>10s}\n'
    _out.write('# AKs = {0:.2f}   age = 1 Myr   d = 389 pc\n'.format(AKs))
    _out.write(fmt.format('Teff', 'mass', 'logL', 'logg',
                          'J', 'H', 'Kp', 'Lp'))
    for ii in range(len(results['Teff'])):
        fmt1 = '{0:10.1f}  {1:10.3f}  {2:10.4f}  {3:10.4f}  '
        fmt2 = '{0:10.2f} {1:10.2f} {2:10.2f} {3:10.2f}\n'
        _out.write(fmt1.format(results['Teff'][ii], results['mass'][ii],
                               results['logL'][ii], results['logg'][ii]))
        _out.write(fmt2.format(results['J'][ii], results['H'][ii],
                               results['Kp'][ii], results['Lp'][ii]))
