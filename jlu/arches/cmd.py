import pylab as py
import numpy as np
import math
from jlu.arches import synthetic as syn
from jlu.util import constants
from jlu.stellarModels import atmospheres as atm
import pdb

logAge = 6.4
AKs = 3.1
distance = 8000

def plot_cmd():
    iso = syn.load_isochrone(logAge=logAge, AKs=AKs, distance=distance)

    mag = iso.mag153m
    color = iso.mag127m - iso.mag153m

    idx1 = np.abs(iso.M - 1.0).argmin()
    idx01 = np.abs(iso.M - 0.1).argmin()

    py.clf()
    py.plot(color, mag, 'k.')
    py.plot(color[idx1], mag[idx1], 'r^', ms=20)
    py.plot(color[idx01], mag[idx01], 'r^', ms=20)
    py.xlabel('F117M - F153M')
    py.ylabel('F153M')
    py.gca().set_ylim(py.gca().get_ylim()[::-1])
    py.savefig('/u/jlu/work/arches/cmd_sim.png')


    py.clf()
    py.semilogx(iso.T, iso.logL, 'k.')
    py.gca().set_xlim(py.gca().get_xlim()[::-1])
    py.plot(iso.T[idx1], iso.logL[idx1], 'r^', ms=20)
    py.plot(iso.T[idx01], iso.logL[idx01], 'r^', ms=20)
    py.xlabel('Teff (K)')
    py.ylabel('log(L/Lsun')
    py.savefig('/u/jlu/work/arches/hr_sim.png')

    wave = np.array([1.27, 1.39, 1.53])
    py.clf()
    py.plot(wave, [iso.mag127m[idx1], iso.mag139m[idx1], iso.mag153m[idx1]], 'r-')
    py.plot(wave, [iso.mag127m[idx01], iso.mag139m[idx01], iso.mag153m[idx01]], 'b-')
    py.savefig('/u/jlu/work/arches/sed.png')
    
    c = constants

    # Convert luminosity to erg/s
    L_all = 10**(iso.logL) * c.Lsun # luminsoity in erg/s

    # Calculate radius
    R_all = np.sqrt(L_all / (4.0 * math.pi * c.sigma * iso.T**4))
    R_all /= (c.cm_in_AU * c.AU_in_pc)
    
    # Get the atmosphere model now. Wavelength is in Angstroms
    star1 = atm.get_merged_atmosphere(temperature=iso.T[idx1], 
                                     gravity=iso.logg[idx1])

    # Convert into flux observed at Earth (unreddened)
    star1 *= (R_all[idx1] / distance)**2  # in erg s^-1 cm^-2 A^-1


    # Get the atmosphere model now. Wavelength is in Angstroms
    star01 = atm.get_merged_atmosphere(temperature=iso.T[idx01], 
                                       gravity=iso.logg[idx01])

    # Convert into flux observed at Earth (unreddened)
    star01 *= (R_all[idx01] / distance)**2  # in erg s^-1 cm^-2 A^-1

    py.clf()
    py.semilogy(star1.wave, star1.flux, 'r-')
    py.plot(star01.wave, star01.flux, 'b-')
    py.xlim(10000, 20000)
    py.ylim(5e-19, 1e-16)
    py.savefig('/u/jlu/work/arches/spectra.png')
