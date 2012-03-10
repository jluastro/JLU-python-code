import numpy as np
import pylab as py
from jlu.stellarModels import evolution
from jlu.stellarModels import extinction
from jlu.stellarModels import atmospheres as atm
from scipy import interpolate
from scipy import stats
from gcwork import objects
from pysynphot import spectrum
from pysynphot import ObsBandpass
from pysynphot import observation as obs
import pysynphot
from jlu.util import constants
from jlu.util import plfit
from jlu.nirc2 import synthetic as nirc2syn
import pickle
import time, datetime
import math
import os, glob
import tempfile
import scipy
import matplotlib
import pymc
import pdb

defaultAKs = 0.91
defaultDist = 4000

def Vega():
    # Use Vega as our zeropoint... assume V=0.03 mag and all colors = 0.0
    vega = atm.get_kurucz_atmosphere(temperature=9550, 
                                     gravity=3.95,
                                     metallicity=-0.5)

    vega = spectrum.trimSpectrum(vega, 8000, 50000)

    # This is (R/d)**2 as reported by Girardi et al. 2002, page 198, col 1.
    # and is used to convert to flux observed at Earth.
    vega *= 6.247e-17 
    
    return vega

class RedLawNishiyama09(pysynphot.reddening.CustomRedLaw):
    """
    You can call reddening(AKs) which will return an ArraySpectralElement
    that can then be manipulated with spectra.
    """
    def __init__(self):
        # Fetch the extinction curve, pre-interpolate across 1-8 microns
        wave = np.arange(0.5, 8.0, 0.001)
        
        # This will eventually be scaled by AKs when you
        # call reddening(). Right now, calc for AKs=1
        Alambda_scaled = extinction.nishiyama09(wave, 1.0, makePlot=False)

        # Convert wavelength to angstrom
        wave *= 10**4

        pysynphot.reddening.CustomRedLaw.__init__(self, wave=wave, 
                                                  waveunits='angstrom',
                                                  Avscaled=Alambda_scaled,
                                                  name='Nishiyama09',
                                                  litref='Nishiyama+ 2009')

class RedLawRomanZuniga07(pysynphot.reddening.CustomRedLaw):
    """
    You can call reddening(AKs) which will return an ArraySpectralElement
    that can then be manipulated with spectra.
    """
    def __init__(self):
        # Fetch the extinction curve, pre-interpolate across 1-8 microns
        wave = np.arange(1.0, 8.0, 0.01)
        
        # This will eventually be scaled by AKs when you
        # call reddening(). Right now, calc for AKs=1
        Alambda_scaled = extinction.romanzuniga07(wave, 1.0, makePlot=False)

        # Convert wavelength to angstrom
        wave *= 10**4

        pysynphot.reddening.CustomRedLaw.__init__(self, wave=wave, 
                                                  waveunits='angstrom',
                                                  Avscaled=Alambda_scaled,
                                                  name='RomanZuniga07',
                                                  litref='Roman-Zuniga+ 2007')


vega = Vega()
redlaw = RedLawNishiyama09()


def make_observed_isochrone_hst(logAge, AKs=defaultAKs,
                                distance=defaultDist, verbose=False):
    startTime = time.time()

    print 'Making isochrone: log(t) = %.2f  AKs = %.2f  dist = %d' % \
        (logAge, AKs, distance)
    print '     Starting at: ', datetime.datetime.now(), '  Usually takes ~5 minutes'

    outFile = '/u/jlu/work/arches/models/iso/'
    outFile += 'iso_%.2f_hst_%4.2f_%4s.pickle' % (logAge, AKs,
                                                 str(distance).zfill(4))

    c = constants

    # Get solar mettalicity models for a population at a specific age.
    evol = evolution.get_merged_isochrone(logAge=logAge)

    # Lets do some trimming down to get rid of repeat masses or 
    # mass resolutions higher than 1/1000. We will just use the first
    # unique mass after rounding by the nearest 0.001.
    mass_rnd = np.round(evol.mass, decimals=2)
    tmp, idx = np.unique(mass_rnd, return_index=True)

    mass = evol.mass[idx]
    logT = evol.logT[idx]
    logg = evol.logg[idx]
    logL = evol.logL[idx]
    isWR = logT != evol.logT_WR[idx]

    temp = 10**logT

    # Output magnitudes for each temperature and extinction value.
    mag127m = np.zeros(len(temp), dtype=float)
    mag139m = np.zeros(len(temp), dtype=float)
    mag153m = np.zeros(len(temp), dtype=float)
    magH = np.zeros(len(temp), dtype=float)
    magKp = np.zeros(len(temp), dtype=float)
    magLp = np.zeros(len(temp), dtype=float)

    filt127m = get_filter_info('wfc3,ir,f127m')
    filt139m = get_filter_info('wfc3,ir,f139m')
    filt153m = get_filter_info('wfc3,ir,f153m')
    filtH = get_filter_info('nirc2,H')
    filtKp = get_filter_info('nirc2,Kp')
    filtLp = get_filter_info('nirc2,Lp')

    # Make reddening
    red127m = redlaw.reddening(AKs).resample(filt127m.wave)
    red139m = redlaw.reddening(AKs).resample(filt139m.wave)
    red153m = redlaw.reddening(AKs).resample(filt153m.wave)
    redH = redlaw.reddening(AKs).resample(filtH.wave)
    redKp = redlaw.reddening(AKs).resample(filtKp.wave)
    redLp = redlaw.reddening(AKs).resample(filtLp.wave)

    # Convert luminosity to erg/s
    L_all = 10**(logL) * c.Lsun # luminsoity in erg/s

    # Calculate radius
    R_all = np.sqrt(L_all / (4.0 * math.pi * c.sigma * temp**4))
    R_all /= (c.cm_in_AU * c.AU_in_pc)

    # For each temperature extract the synthetic photometry.
    for ii in range(len(temp)):
        gravity = logg[ii]
        L = L_all[ii] # in erg/s
        T = temp[ii]  # in Kelvin
        R = R_all[ii] # in pc

        # Get the atmosphere model now. Wavelength is in Angstroms
        star = atm.get_merged_atmosphere(temperature=T, 
                                         gravity=gravity)

        # Trim wavelength range down to JHKL range (0.5 - 4.25 microns)
        star = spectrum.trimSpectrum(star, 5000, 42500)

        # Convert into flux observed at Earth (unreddened)
        star *= (R / distance)**2  # in erg s^-1 cm^-2 A^-1

        # ----------
        # Now to the filter integrations
        # ----------
        mag127m[ii] = mag_in_filter(star, filt127m, red127m)
        mag139m[ii] = mag_in_filter(star, filt139m, red139m)
        mag153m[ii] = mag_in_filter(star, filt153m, red153m)
        magH[ii] = mag_in_filter(star, filtH, redH)
        magKp[ii] = mag_in_filter(star, filtKp, redKp)
        magLp[ii] = mag_in_filter(star, filtLp, redLp)

        if verbose:
            print 'M = %7.3f Msun  T = %5d K  R = %2.1f Rsun  logg = %4.2f  F127M = %4.2f  F139M = %4.2f  F153M = %4.2f' % \
                (mass[ii], T, R * c.AU_in_pc / c.Rsun, logg[ii], mag127m[ii], mag139m[ii], mag153m[ii])


    iso = objects.DataHolder()
    iso.M = mass
    iso.T = temp
    iso.logg = logg
    iso.logL = logL
    iso.mag127m = mag127m
    iso.mag139m = mag139m
    iso.mag153m = mag153m
    iso.magH = magH
    iso.magKp = magKp
    iso.magLp = magLp
    iso.isWR = isWR
    
    _out = open(outFile, 'wb')
    pickle.dump(mass, _out)
    pickle.dump(temp, _out)
    pickle.dump(logg, _out)
    pickle.dump(logL, _out)
    pickle.dump(mag127m, _out)
    pickle.dump(mag139m, _out)
    pickle.dump(mag153m, _out)
    pickle.dump(magH, _out)
    pickle.dump(magKp, _out)
    pickle.dump(magLp, _out)
    pickle.dump(isWR, _out)
    _out.close()

    endTime = time.time()
    print '      Time taken: %d seconds' % (endTime - startTime)

def load_isochrone(logAge=6.78, AKs=defaultAKs, distance=defaultDist):
    inFile = '/u/jlu/work/arches/models/iso/'
    inFile += 'iso_%.2f_hst_%4.2f_%4s.pickle' % (logAge, AKs,
                                                 str(distance).zfill(4))

    if not os.path.exists(inFile):
        make_observed_isochrone_hst(logAge=logAge, AKs=AKs, distance=distance)

    _in = open(inFile, 'rb')
    iso = objects.DataHolder()
    iso.M = pickle.load(_in)
    iso.T = pickle.load(_in)
    iso.logg = pickle.load(_in)
    iso.logL = pickle.load(_in)
    iso.mag127m = pickle.load(_in)
    iso.mag139m = pickle.load(_in)
    iso.mag153m = pickle.load(_in)
    iso.magH = pickle.load(_in)
    iso.magKp = pickle.load(_in)
    iso.magLp = pickle.load(_in)
    iso.isWR = pickle.load(_in)
    _in.close()

    return iso


# Little helper utility to get all the bandpass/zeropoint info.
def get_filter_info(name, vega=vega):
    if name.startswith('nirc2'):
        tmp = name.split(',')
        filterName = tmp[-1]
        filter = nirc2syn.filters[filterName]
        flux0 = nirc2syn.filter_flux0[filterName]
        mag0 = nirc2syn.filter_mag0[filterName]
    else:
        filter = ObsBandpass(name)

    vega_obs = obs.Observation(vega, filter, binset=filter.wave, force='taper')
    vega_flux = vega_obs.binflux.sum()
    vega_mag = 0.03

    filter.flux0 = vega_flux
    filter.mag0 = vega_mag
    
    return filter

# Little helper utility to get the magnitude of an object through a filter.
def mag_in_filter(star, filter, extinction):
    """
    Assumes that extinction is already resampled to same wavelengths
    as filter.
    """
    star_in_filter = obs.Observation(star, filter*extinction,
                                     binset=filter.wave, force='taper')
    star_flux = star_in_filter.binflux.sum()
    star_mag = -2.5 * math.log10(star_flux / filter.flux0) + filter.mag0

    return star_mag


