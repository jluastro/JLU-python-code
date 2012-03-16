import pylab as py
import numpy as np
from jlu.arches import synthetic as syn

plotDir = '/u/jlu/work/arches/red_clump/'

def make_models():
    # Make one as would be observed at the GC.
    syn.load_isochrone(logAge=9.78, AKs=3.1, distance=8000)

    # Make some for absolute photometry. 10 pc, no extinction.
    syn.load_isochrone(logAge=9.00, AKs=0.0, distance=10)
    syn.load_isochrone(logAge=9.30, AKs=0.0, distance=10)
    syn.load_isochrone(logAge=9.48, AKs=0.0, distance=10)
    syn.load_isochrone(logAge=9.60, AKs=0.0, distance=10)
    syn.load_isochrone(logAge=9.70, AKs=0.0, distance=10)
    syn.load_isochrone(logAge=9.78, AKs=0.0, distance=10)

def plot_red_clump_bolometric():
    ages = np.array([9.00, 8.95, 8.90, 8.85, 8.78, 8.70])
    isos = []

    for ii in range(len(ages)):
        iso = syn.load_isochrone(logAge=ages[ii], AKs=0, distance=10)
        isos.append(iso)

    # Make an HR diagram
    py.clf()
    for ii in range(len(ages)):
        iso = isos[ii]
        label = '%2d' % (10**(ages[ii]-9.0))
        py.plot(iso.T, iso.logL, '-', label=label)
    gca = py.gca()
    gca.set_xlim(gca.get_xlim()[::-1])
    py.xlabel('Teff (K)')
    py.ylabel('log (L/L_sun)')
    py.legend(loc='lower left')
    py.savefig(plotDir + 'hr_diagram.png')

    # Make an HR diagram
    py.clf()
    for ii in range(len(ages)):
        iso = isos[ii]
        label = '%2d' % (10**(ages[ii]-9.0))
        py.plot(iso.T, iso.logL, '.', label=label)
    gca = py.gca()
    gca.set_xlim(gca.get_xlim()[::-1])
    py.xlabel('Teff (K)')
    py.ylabel('log (L/L_sun)')
    py.legend(loc='lower left')
    py.savefig(plotDir + 'hr_diagram_points.png')
        

def plot_red_clump_arches():
    logAge = 9.0
    AKs = 3.1
    distance = 8000
    iso = syn.load_isochrone(logAge=logAge, AKs=AKs, distance=distance)

    # Magnitude-Temperature 
    py.clf()
    py.plot(iso.T, iso.magKp, 'k-')
    gca = py.gca()
    gca.set_xlim(gca.get_xlim()[::-1])
    gca.set_ylim(gca.get_ylim()[::-1])
    py.xlabel('Teff (K)')
    py.ylabel('Kp Magnitude')
    py.title('log(t)=%.2f, AKs=%.1f, d=%4d' % (logAge, AKs, distance))
    py.savefig(plotDir + 'rc_arches_mag_temp.png')

    # Magnitude-Temperature 
    py.clf()
    py.plot(iso.mag153m-iso.magKp, iso.magKp, 'k-')
    gca = py.gca()
    gca.set_ylim(gca.get_ylim()[::-1])
    py.xlabel('F153M - Kp')
    py.ylabel('Kp')
    py.title('log(t)=%.2f, AKs=%.1f, d=%4d' % (logAge, AKs, distance))
    py.savefig(plotDir + 'rc_arches_cmd_153m_kp.png')
