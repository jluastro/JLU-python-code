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

def nishiyama09(wavelength, AKs, makePlot=False):
    # Data pulled from Nishiyama et al. 2009, Table 1

    filters = ['V', 'J', 'H', 'Ks', '[3.6]', '[4.5]', '[5.8]', '[8.0]']
    wave =      np.array([0.551, 1.25, 1.63, 2.14, 3.545, 4.442, 5.675, 7.760])
    A_AKs =     np.array([16.13, 3.02, 1.73, 1.00, 0.500, 0.390, 0.360, 0.430])
    A_AKs_err = np.array([0.04,  0.04, 0.03, 0.00, 0.010, 0.010, 0.010, 0.010])

    # Interpolate over the curve
    spline_interp = interpolate.splrep(wave, A_AKs, k=3, s=0)

    A_AKs_at_wave = interpolate.splev(wavelength, spline_interp)
    A_at_wave = AKs * A_AKs_at_wave

    if makePlot:
        py.clf()
        py.errorbar(wave, A_AKs, yerr=A_AKs_err, fmt='bo', 
                    markerfacecolor='none', markeredgecolor='blue',
                    markeredgewidth=2)
        
        # Make an interpolated curve.
        wavePlot = np.arange(wave.min(), wave.max(), 0.1)
        extPlot = interpolate.splev(wavePlot, spline_interp)
        py.loglog(wavePlot, extPlot, 'k-')

        # Plot a marker for the computed value.
        py.plot(wavelength, A_AKs_at_wave, 'rs',
                markerfacecolor='none', markeredgecolor='red',
                markeredgewidth=2)
        py.xlabel('Wavelength (microns)')
        py.ylabel('Extinction (magnitudes)')
        py.title('Nishiyama et al. 2009')

    
    return A_at_wave


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

    outFile = '/u/jlu/work/wd1/models/iso/'
    outFile += 'iso_%.2f_hst_%4.2f_%4s.pickle' % (logAge, AKs,
                                                 str(distance).zfill(4))

    c = constants

    # Get solar mettalicity models for a population at a specific age.
    evol = evolution.get_merged_isochrone(logAge=logAge)

    # Lets do some trimming down to get rid of repeat masses or 
    # mass resolutions higher than 1/1000. We will just use the first
    # unique mass after rounding by the nearest 0.001.
    mass_rnd = np.round(evol.mass, decimals=3)
    tmp, idx = np.unique(mass_rnd, return_index=True)

    mass = evol.mass[idx]
    logT = evol.logT[idx]
    logg = evol.logg[idx]
    logL = evol.logL[idx]
    isWR = logT != evol.logT_WR[idx]

    temp = 10**logT

    # Output magnitudes for each temperature and extinction value.
    mag814w = np.zeros(len(temp), dtype=float)
    mag125w = np.zeros(len(temp), dtype=float)
    mag139m = np.zeros(len(temp), dtype=float)
    mag160w = np.zeros(len(temp), dtype=float)

    filt814w = get_filter_info('acs,f814w,wfc1')
    filt125w = get_filter_info('wfc3,ir,f125w')
    filt139m = get_filter_info('wfc3,ir,f139m')
    filt160w = get_filter_info('wfc3,ir,f160w')

    # Make reddening
    red814w = redlaw.reddening(AKs).resample(filt814w.wave)
    red125w = redlaw.reddening(AKs).resample(filt125w.wave)
    red139m = redlaw.reddening(AKs).resample(filt139m.wave)
    red160w = redlaw.reddening(AKs).resample(filt160w.wave)

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
        mag814w[ii] = mag_in_filter(star, filt814w, red814w)
        mag125w[ii] = mag_in_filter(star, filt125w, red125w)
        mag139m[ii] = mag_in_filter(star, filt139m, red139m)
        mag160w[ii] = mag_in_filter(star, filt160w, red160w)

        if verbose:
            print 'M = %7.3f Msun  T = %5d K  R = %2.1f Rsun  logg = %4.2f  F814W = %4.2f  F125W = %4.2f  F139M = %4.2f  F160W = %4.2f' % \
                (mass[ii], T, R * c.AU_in_pc / c.Rsun, logg[ii], mag814w[ii], mag125w[ii], mag139m[ii], mag160w[ii])


    iso = objects.DataHolder()
    iso.M = mass
    iso.T = temp
    iso.logg = logg
    iso.logL = logL
    iso.mag814w = mag814w
    iso.mag125w = mag125w
    iso.mag139m = mag139m
    iso.mag160w = mag160w
    iso.isWR = isWR
    
    _out = open(outFile, 'wb')
    pickle.dump(mass, _out)
    pickle.dump(temp, _out)
    pickle.dump(logg, _out)
    pickle.dump(logL, _out)
    pickle.dump(mag814w, _out)
    pickle.dump(mag125w, _out)
    pickle.dump(mag139m, _out)
    pickle.dump(mag160w, _out)
    pickle.dump(isWR, _out)
    _out.close()

    endTime = time.time()
    print '      Time taken: %d seconds' % (endTime - startTime)

def load_isochrone(logAge=6.60, AKs=defaultAKs, distance=defaultDist):
    inFile = '/u/jlu/work/wd1/models/iso/'
    inFile += 'iso_%.2f_hst_%4.2f_%4s.pickle' % (logAge, AKs,
                                                 str(distance).zfill(4))

    changeDistance = False

    if not os.path.exists(inFile):
        # File doesn't exist, but if only distance has changed we can simply rescale.
        inFile = '/u/jlu/work/wd1/models/iso/'
        inFile += 'iso_%.2f_hst_%4.2f_%4s.pickle' % (logAge, AKs,
                                                 str(defaultDist).zfill(4))
        if not os.path.exists(inFile):
            make_observed_isochrone_hst(logAge=logAge, AKs=AKs, distance=distance)
        else:
            changeDistance = True

    _in = open(inFile, 'rb')
    iso = objects.DataHolder()
    iso.M = pickle.load(_in)
    iso.T = pickle.load(_in)
    iso.logg = pickle.load(_in)
    iso.logL = pickle.load(_in)
    iso.mag814w = pickle.load(_in)
    iso.mag125w = pickle.load(_in)
    iso.mag139m = pickle.load(_in)
    iso.mag160w = pickle.load(_in)
    iso.isWR = pickle.load(_in)
    _in.close()

    if changeDistance:
        print 'Using existing isochrone with d = %d and changing to d = %d' % \
            (defaultDist, distance)
        deltaDM = 5.0 * math.log10(float(distance) / float(defaultDist))
        print '    delta DM = %.2f' % deltaDM
        iso.mag814w += deltaDM
        iso.mag125w += deltaDM
        iso.mag139m += deltaDM
        iso.mag160w += deltaDM

    return iso


# Little helper utility to get all the bandpass/zeropoint info.
def get_filter_info(name, vega=vega):
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


