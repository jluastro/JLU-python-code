import atpy
import pyfits
from jlu.util import constants as c
from jlu.stellarModels import extinction
from jlu.stellarModels import atmospheres as atm
from jlu.stellarModels import evolution
from jlu.nirc2 import photometry as nirc2phot
from scipy import interpolate
import numpy as np
import pylab as py
import math
import pickle
from mpl_toolkits.axes_grid import AxesGrid
import pdb
import pysynphot
from pysynphot import spectrum
from pysynphot import observation as obs
import os

# Define a function that will do filter integrations
def magnitude_in_filter(filter, star, ext, AKs, atm, vega):
    """
    Using pysynphot objects and functions.

    filter = NIRC2Filter object (wave in angstrom)
    star = TabularSourceSpectrum object (wave in angstrom, flux in FLAMs)
    ext = CustomRedLaw object
    atm = EarthAtmosphere object (wave in angstrom)
    vega = Vega object (wave in angstrom, flux in FLAMs)
    """
    bandpass = filter

    if atm != None:
        bandpass *= atm

    vega_in_filter = obs.Observation(vega, bandpass, binset=filter.wave)
    vega_flux = vega_in_filter.binflux.sum()
    vega_mag = 0.03

    if ext != None and AKs > 0:
        bandpass *= extinction.reddening(AKs)

    star_in_filter = obs.Observation(star, bandpass, binset=filter.wave)
    star_flux = star_in_filter.binflux.sum()
    star_mag = -2.5 * math.log10(star_flux/vega_flux) + vega_mag

    return star_mag


class FilterNIRC2(spectrum.ArraySpectralElement):
    def __init__(self, name):
        wave, trans = nirc2phot.get_filter_profile(name)

        # Convert to Angstroms
        wave *= 10**4

        spectrum.ArraySpectralElement.__init__(self, 
                                               wave=wave, throughput=trans,
                                               waveunits='angstrom',
                                               name='NIRC2-'+name)

def Vega():
    # Use Vega as our zeropoint... assume V=0.03 mag and all colors = 0.0
    vega = atm.get_kurucz_atmosphere(temperature=9550, 
                                     gravity=3.95,
                                     metallicity=-0.5)

    vega = spectrum.trimSpectrum(vega, 10000, 50000)

    # This is (R/d)**2 as reported by Girardi et al. 2002, page 198, col 1.
    # and is used to convert to flux observed at Earth.
    vega *= 6.247e-17 
    
    return vega


class EarthAtmosphere(spectrum.ArraySpectralElement):
    nirc2_base = os.path.dirname(nirc2phot.__file__)

    earth_file = nirc2_base + '/earth_transmission.fits'
    
    def __init__(self, dataFile=earth_file):
        self.data_file = dataFile

        # also get the atmospheric transmission curve
        atmData = atpy.Table(self.data_file)

        wave = atmData.Microns
        trans = atmData.Transmission

        # Convert wavelength to angstrom
        wave *= 10**4

        # Trim down to near-IR wavelengths
        idx = np.where((wave >= 10000) & (wave <= 50000))[0]
        wave = wave[idx]
        trans = trans[idx]

        spectrum.ArraySpectralElement.__init__(self, 
                                               wave=wave, throughput=trans,
                                               waveunits='angstrom',
                                               name='Earth Atmosphere')

class RedLawNishiyama09(pysynphot.reddening.CustomRedLaw):
    """
    You can call reddening(AKs) which will return an ArraySpectralElement
    that can then be manipulated with spectra.
    """
    def __init__(self):
        # Fetch the extinction curve, pre-interpolate across 1-8 microns
        wave = np.arange(1.0, 8.0, 0.01)
        
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



earth = EarthAtmosphere()
vega = Vega()
redlaw = RedLawNishiyama09()

# Little helper utility to get all the bandpass/zeropoint info.
def get_filter_info(name, earth=earth, vega=vega):
    filter = FilterNIRC2(name)

    earth2 = earth.resample(filter.wave)
    filter *= earth2

    vega_obs = obs.Observation(vega, filter, binset=filter.wave)
    vega_flux = vega_obs.binflux.sum()
    vega_mag = 0.03

    return filter, vega_flux, vega_mag


# Little helper utility to get the magnitude of an object through a filter.
def mag_in_filter(star, filter, extinction, flux0, mag0):
    """
    Assumes that extinction is already resampled to same wavelengths
    as filter.
    """
    star_in_filter = obs.Observation(star, filter*extinction,
                                     binset=filter.wave)
    star_flux = star_in_filter.binflux.sum()
    star_mag = -2.5 * math.log10(star_flux / flux0) + mag0

    return star_mag
        
filter_names = ['J', 'H', 'K', 'Kp', 'Ks', 'Lp']
filters = {}
filter_mag0 = {}
filter_flux0 = {}

for ff in range(len(filter_names)):
    filt = filter_names[ff]

    foo = get_filter_info(filt)
    
    filters[filt] = foo[0]
    filter_flux0[filt] = foo[1]
    filter_mag0[filt] = foo[2]


def nearIR(distance, logAge, redlawClass=RedLawNishiyama09, AKsGrid=None):
    """
    For a sampling of effective temperatures and extinctions, calculate the
    J, H, K, Kp, Ks, Lp magnitudes for a population at the specified
    distance and age. All output is stored in a pickle file.

    Input Parameters:
    distance in pc
    logAge

    Optional Input Parameters:
    redlawClass - default = RedLawNishiyama09
    AKsGrid -- default [0 - 5; 0.1 steps]

    Output stored in a pickle file named syn_nir_d#####_a####.dat.
    
    """
    pickleFile = 'syn_nir_d' + str(distance).zfill(5) + '_a' \
        + str(int(round(logAge*100))).zfill(3) + '.dat'

    # Get solar mettalicity models for a population at a specific age.
    evol = evolution.get_merged_isochrone(logAge=logAge)
    mass = evol.mass
    logT = evol.logT
    logg = evol.logg
    logL = evol.logL
    temp = 10**logT
    isWR = evol.logT != evol.logT_WR
    print 'nearIR: Getting rid of Wolf-Rayet stars, we cannot model their atmospheres'

    # First get rid of the WR stars, we can't connect atmospheres
    # to them anyhow.
    idx = np.where(isWR == False)[0]
    mass = mass[idx]
    logT = logT[idx]
    logg = logg[idx]
    logL = logL[idx]
    temp = temp[idx]
    isWR = isWR[idx]
    
    # Sample only 100 points along the whole isochrone
    interval = int(math.floor(len(mass) / 100.0))
    idx = np.arange(0, len(mass), interval, dtype=int)
    # Make sure to get the last point
    if idx[-1] != (len(mass) - 1):
        idx = np.append(idx, len(mass) - 1)
        
    mass = mass[idx]
    logT = logT[idx]
    logg = logg[idx]
    logL = logL[idx]
    temp = temp[idx]

    # We will also run through a range of extinctions
    if AKsGrid == None:
        AKsGrid = np.arange(0, 5, 0.1)

    # Fetch earth, vega, and extinction objects
    earth = EarthAtmosphere()
    vega = Vega()
    redlaw = redlawClass()

    # Get the transmission curve for NIRC2 filters and atmosphere.
    J_filter, J_flux0, J_mag0 = get_filter_info('J', earth, vega)
    H_filter, H_flux0, H_mag0 = get_filter_info('H', earth, vega)
    K_filter, K_flux0, K_mag0 = get_filter_info('K', earth, vega)
    Kp_filter, Kp_flux0, Kp_mag0 = get_filter_info('Kp', earth, vega)
    Ks_filter, Ks_flux0, Ks_mag0 = get_filter_info('Ks', earth, vega)
    Lp_filter, Lp_flux0, Lp_mag0 = get_filter_info('Lp', earth, vega)

    # Output magnitudes for each temperature and extinction value.
    J = np.zeros((len(temp), len(AKsGrid)), dtype=float)
    H = np.zeros((len(temp), len(AKsGrid)), dtype=float)
    K = np.zeros((len(temp), len(AKsGrid)), dtype=float)
    Kp = np.zeros((len(temp), len(AKsGrid)), dtype=float)
    Ks = np.zeros((len(temp), len(AKsGrid)), dtype=float)
    Lp = np.zeros((len(temp), len(AKsGrid)), dtype=float)

    # For each filter, lets pre-make reddening curves so we only
    # have to do the calculation once.
    J_red = []
    H_red = []
    K_red = []
    Kp_red = []
    Ks_red = []
    Lp_red = []
    
    print 'Making extinction curves'
    for aa in range(len(AKsGrid)):
        red = redlaw.reddening(AKsGrid[aa])

        J_red.append( red.resample(J_filter.wave) )
        H_red.append( red.resample(H_filter.wave) )
        K_red.append( red.resample(K_filter.wave) )
        Kp_red.append( red.resample(Kp_filter.wave) )
        Ks_red.append( red.resample(Ks_filter.wave) )
        Lp_red.append( red.resample(Lp_filter.wave) )
    
    # For each temperature extract the synthetic photometry.
    for ii in range(len(temp)):
        gravity = logg[ii]
        L = 10**(logL[ii]) * c.Lsun # luminosity in erg/s
        T = temp[ii]  # in Kelvin
        # Get the radius
        R = math.sqrt(L / (4.0 * math.pi * c.sigma * T**4))   # in cm
        R /= (c.cm_in_AU * c.AU_in_pc)   # in pc

        print 'M = %6.2f Msun   T = %5d   R = %2.1f Rsun   logg = %4.2f' % \
            (mass[ii], T, R * c.AU_in_pc / c.Rsun, logg[ii])

        # Get the atmosphere model now. Wavelength is in Angstroms
        star = atm.get_merged_atmosphere(temperature=T,
                                         gravity=gravity)

        # Trim wavelength range down to JHKL range (1.0 - 4.25 microns)
        star = spectrum.trimSpectrum(star, 10000, 42500)

        # Convert into flux observed at Earth (unreddened)
        star *= (R / distance)**2  # in erg s^-1 cm^-2 A^-1

        for aa in range(len(AKsGrid)):
            print 'Photometry for T = %5d, AKs = %3.1f' % \
                (temp[ii], AKsGrid[aa])
            
            # ----------
            # Now to the filter integrations
            # ----------
            # K
            J[ii, aa] = mag_in_filter(star, J_filter, J_red[aa], 
                                      J_flux0, J_mag0)
            H[ii, aa] = mag_in_filter(star, H_filter, H_red[aa], 
                                      H_flux0, H_mag0)
            K[ii, aa] = mag_in_filter(star, K_filter, K_red[aa], 
                                      K_flux0, K_mag0)
            Kp[ii, aa] = mag_in_filter(star, Kp_filter, Kp_red[aa], 
                                       Kp_flux0, Kp_mag0)
            Ks[ii, aa] = mag_in_filter(star, Ks_filter, Ks_red[aa], 
                                       Ks_flux0, Ks_mag0)
            Lp[ii, aa] = mag_in_filter(star, Lp_filter, Lp_red[aa], 
                                       Lp_flux0, Lp_mag0)


    # Save output to pickle file
    pf = open(pickleFile, 'w')
    pickle.dump(mass, pf)
    pickle.dump(logg, pf)
    pickle.dump(logL, pf)
    pickle.dump(temp, pf)
    pickle.dump(AKsGrid, pf)
    pickle.dump(J, pf)
    pickle.dump(H, pf)
    pickle.dump(K, pf)
    pickle.dump(Kp, pf)
    pickle.dump(Ks, pf)
    pickle.dump(Lp, pf)

    return pickleFile

def load_nearIR(filename):
    pf = open(filename, 'r')
    
    mass = pickle.load(pf)
    logg = pickle.load(pf)
    logL = pickle.load(pf)
    temp = pickle.load(pf)
    AKs = pickle.load(pf)
    J = pickle.load(pf)
    H = pickle.load(pf)
    K = pickle.load(pf)
    Kp = pickle.load(pf)
    Ks = pickle.load(pf)
    Lp = pickle.load(pf)

    return temp, AKs, J, H, K, Kp, Ks, Lp, mass, logg, logL

def load_nearIR_dict(filename):
    resultsArray = load_nearIR(filename)

    results = {}
    results['Teff'] = resultsArray[0]
    results['AKs'] = resultsArray[1]
    results['J'] = resultsArray[2]
    results['H'] = resultsArray[3]
    results['K'] = resultsArray[4]
    results['Kp'] = resultsArray[5]
    results['Ks'] = resultsArray[6]
    results['Lp'] = resultsArray[7]
    results['mass'] = resultsArray[8]
    results['logg'] = resultsArray[9]
    results['logL'] = resultsArray[10]
    
    return results


def plot_grid_Kp_Ks(filename='syn_nir_d06500_a619.dat'):
    temp, AKs, J, H, K, Kp, Ks, Lp, m, logg, logL = load_nearIR(filename)

    A, T = np.meshgrid(AKs, temp)

    Kp_Ks = Kp - Ks

    # Plot 2D grid 
    py.clf()
    py.subplots_adjust(bottom=0.05, left=0.16)
    py.imshow(Kp_Ks, extent=[A.min(), A.max(), T.min(), T.max()])
    py.axis('tight')
    py.xlabel('A_Ks (mag)')
    py.ylabel('T_eff (K)')
    py.title(filename)
    cbar = py.colorbar(orientation='horizontal', pad=0.1, fraction=0.1)
    cbar.set_label('Kp - Ks (mag)')
    py.savefig('plots/syn_nir_Kp-Ks_T_AKs.png')

def plot_grid_Kp_K(filename='syn_nir_d06500_a619.dat'):
    temp, AKs, J, H, K, Kp, Ks, Lp, m, logg, logL = load_nearIR(filename)

    A, T = np.meshgrid(AKs, temp)

    Kp_K = Kp - K

    # Plot 2D grid 
    py.clf()
    py.subplots_adjust(bottom=0.05, left=0.16)
    py.imshow(Kp_K, extent=[A.min(), A.max(), T.min(), T.max()])
    py.axis('tight')
    py.xlabel('A_Ks (mag)')
    py.ylabel('T_eff (K)')
    py.title(filename)
    cbar = py.colorbar(orientation='horizontal', pad=0.1, fraction=0.1)
    cbar.set_label('Kp - K (mag)')
    py.savefig('plots/syn_nir_Kp-K_T_AKs.png')



def plot_Kcolors_vs_temp_sample(filename='syn_nir_d06500_a619.dat'):
    temp, AKs, J, H, K, Kp, Ks, Lp, m, logg, logL = load_nearIR(filename)

    Kp_Ks = Kp - Ks

    cnorm = py.normalize(Kp_Ks.min(), Kp_Ks.max())

    # Plot in 1D, pick out specific extinctions and temperatures
    # as examples.
    plotAKs = [0, 10, 20, 30]

    py.clf()
    py.subplots_adjust(hspace=0.001, bottom=0.1)
    
    sharedAxis = None
    for ii in range(len(plotAKs)):
        aa = plotAKs[ii]
        color = py.cm.jet(cnorm(Kp_Ks[0, aa]))

        idx = np.where(Kp_Ks[:,aa] != 0)[0]
        tmpAx = py.subplot(len(plotAKs), 1, ii+1, sharex=sharedAxis)
        if sharedAxis == None:
            sharedAxis = tmpAx
        foo = py.plot(temp[idx], Kp_Ks[idx, aa], 'k-',
                      color=color, linewidth=2)

        py.setp(py.gca().get_xticklabels(), visible=False)

        minVal = Kp_Ks[idx, aa].min()
        py.ylim(minVal, minVal + 0.018)

        py.xlim(5000, 32000)
        py.ylabel('Kp - Ks')

        py.text(25000, minVal + 0.012, 'AKs = %.1f' % AKs[aa], 
                color=color, weight='bold')
        
    py.setp(py.gca().get_xticklabels(), visible=True)
    py.xlabel('T_eff (K)')

    py.savefig('plots/syn_nir_Kp-Ks_T_sample.png')


    # Kp - K
    Kp_K = Kp - K

    cnorm = py.normalize(Kp_K.min(), Kp_K.max())

    # Plot in 1D, pick out specific extinctions and temperatures
    # as examples.
    plotAKs = [0, 10, 20, 30]

    py.clf()
    py.subplots_adjust(hspace=0.001, bottom=0.1)
    
    sharedAxis = None
    for ii in range(len(plotAKs)):
        aa = plotAKs[ii]
        color = py.cm.jet(cnorm(Kp_K[0, aa]))

        idx = np.where(Kp_K[:,aa] != 0)[0]
        tmpAx = py.subplot(len(plotAKs), 1, ii+1, sharex=sharedAxis)
        if sharedAxis == None:
            sharedAxis = tmpAx
        foo = py.plot(temp[idx], Kp_K[idx, aa], 'k-',
                      color=color, linewidth=2)

        py.setp(py.gca().get_xticklabels(), visible=False)

        minVal = Kp_K[idx, aa].min()
        py.ylim(minVal, minVal + 0.018)

        py.xlim(5000, 32000)
        py.ylabel('Kp - K')

        py.text(25000, minVal + 0.012, 'AKs = %.1f' % AKs[aa], 
                color=color, weight='bold')
        
    py.setp(py.gca().get_xticklabels(), visible=True)
    py.xlabel('T_eff (K)')

    py.savefig('plots/syn_nir_Kp-K_T_sample.png')

def get_Kp_K(pickAKs, pickTemp, filename='syn_nir_d06500_a619.dat',
             verbose=False):
    temp, AKs, J, H, K, Kp, Ks, Lp, m, logg, logL = load_nearIR(filename)

    Kp_K = Kp - K

    adx = np.abs(AKs - pickAKs).argmin()
    tdx = np.abs(temp - pickTemp).argmin()

    if verbose:
        print 'Selected T = %.1f K for input of %.1f' % (temp[tdx], pickTemp)
        print 'Selected AKs = %.2f mag for input of %.2f' % (AKs[adx], pickAKs)
        print '   Kp - K = %.3f' % Kp_K[tdx, adx]

    return Kp_K[tdx, adx]

def get_Kp_Ks(pickAKs, pickTemp, filename='syn_nir_d06500_a619.dat',
              verbose=False):
    temp, AKs, J, H, K, Kp, Ks, Lp, m, logg, logL = load_nearIR(filename)

    Kp_Ks = Kp - Ks

    adx = np.abs(AKs - pickAKs).argmin()
    tdx = np.abs(temp - pickTemp).argmin()

    if verbose:
        print 'Selected T = %.1f K for input of %.1f' % (temp[tdx], pickTemp)
        print 'Selected AKs = %.2f mag for input of %.2f' % (AKs[adx], pickAKs)
        print '   Kp - Ks = %.3f' % Kp_Ks[tdx, adx]

    return Kp_Ks[tdx, adx]
    

def test_atmospheres(distance=6000, logAge=7.0, AKs=None):
    """
    Plot synthetic photometry for some main-sequence stars using
    various atmospheres. This is useful for checking for smooth overlap
    between atmosphere models valid at different temperature ranges.
    
    distance = distance in pc
    """
    # Get solar mettalicity models for a population at a specific age.
    print 'Loading Geneva Isochrone logAge=%.2f' % logAge 
    evol = evolution.get_geneva_isochrone(logAge=logAge)
    mass = evol.mass
    logT = evol.logT
    logg = evol.logg
    logL = evol.logL
    temp = 10**logT

    # Trim down to just every 10th entry
    idx = np.arange(0, len(mass), 10)
    mass = mass[idx]
    logT = logT[idx]
    logg = logg[idx]
    logL = logL[idx]
    temp = temp[idx]
    
    # Get the NIRC2 filter throughputs
    J_filter = FilterNIRC2('J')
    H_filter = FilterNIRC2('H')
    Kp_filter = FilterNIRC2('Kp')
    Lp_filter = FilterNIRC2('Lp')

    # Get the throughput spectrum of the Earth's atmosphere
    earth = EarthAtmosphere()

    # Get the spectrum of Vega
    vega = Vega()

    # Get the reddening law
    if AKs != None:
        redlaw = RedLawNishiyama09()
    else:
        redlaw = None
    
    J_kurucz = np.zeros(len(mass), dtype=float)
    H_kurucz = np.zeros(len(mass), dtype=float)
    Kp_kurucz = np.zeros(len(mass), dtype=float)
    Lp_kurucz = np.zeros(len(mass), dtype=float)

    J_castelli = np.zeros(len(mass), dtype=float)
    H_castelli = np.zeros(len(mass), dtype=float)
    Kp_castelli = np.zeros(len(mass), dtype=float)
    Lp_castelli = np.zeros(len(mass), dtype=float)

    J_nextgen = np.zeros(len(mass), dtype=float)
    H_nextgen = np.zeros(len(mass), dtype=float)
    Kp_nextgen = np.zeros(len(mass), dtype=float)
    Lp_nextgen = np.zeros(len(mass), dtype=float)


    # Loop through the models in the isochrone and derive their 
    # synthetic photometry.
    for ii in range(len(temp)):
        print 'Working on Teff=%5d  logg=%4.1f' % (temp[ii], logg[ii])

        L = 10**(logL[ii]) * c.Lsun # luminosity in erg/s
        T = temp[ii]  # in Kelvin
        # Get the radius
        R = math.sqrt(L / (4.0 * math.pi * c.sigma * T**4))   # in cm
        R /= (c.cm_in_AU * c.AU_in_pc)   # in pc
        scaleFactor = (R / distance)**2

        # Get the Kurucz atmosphere model
        k93 = pysynphot.Icat('k93models', temp[ii], 0, logg[ii])
        k93 *= scaleFactor

        # Get the Castelli atmosphere model
        ck04 = pysynphot.Icat('ck04models', temp[ii], 0, logg[ii])
        ck04 *= scaleFactor

        # Get the NextGen atmosphere model
        ngen = pysynphot.Icat('nextgen', temp[ii], 0, logg[ii])
        ngen *= scaleFactor

        # Calculate the photometry
        J_kurucz[ii] = magnitude_in_filter(J_filter, k93, 
                                           redlaw, AKs, None, vega)
        H_kurucz[ii] = magnitude_in_filter(H_filter, k93, 
                                           redlaw, AKs, None, vega)
        Kp_kurucz[ii] = magnitude_in_filter(Kp_filter, k93, 
                                            redlaw, AKs, None, vega)
        Lp_kurucz[ii] = magnitude_in_filter(Lp_filter, k93, 
                                            redlaw, AKs, None, vega)

        J_castelli[ii] = magnitude_in_filter(J_filter, ck04, 
                                             redlaw, AKs, None, vega)
        H_castelli[ii] = magnitude_in_filter(H_filter, ck04, 
                                             redlaw, AKs, None, vega)
        Kp_castelli[ii] = magnitude_in_filter(Kp_filter, ck04, 
                                              redlaw, AKs, None, vega)
        Lp_castelli[ii] = magnitude_in_filter(Lp_filter, ck04, 
                                              redlaw, AKs, None, vega)

        J_nextgen[ii] = magnitude_in_filter(J_filter, ngen, 
                                            redlaw, AKs, None, vega)
        H_nextgen[ii] = magnitude_in_filter(H_filter, ngen, 
                                            redlaw, AKs, None, vega)
        Kp_nextgen[ii] = magnitude_in_filter(Kp_filter, ngen, 
                                             redlaw, AKs, None, vega)
        Lp_nextgen[ii] = magnitude_in_filter(Lp_filter, ngen, 
                                             redlaw, AKs, None, vega)

        if np.isnan(Kp_nextgen[ii]):
            pdb.set_trace()
            foo = magnitude_in_filter(Kp_filter, nextgen, redlaw, AKs, 
                                      None, vega)


    # Now lets plot up some differences vs. effective temp
    # Everything is done with reference to Kurucz.
    J_diff_cast = J_castelli - J_kurucz
    J_diff_ngen = J_nextgen - J_kurucz
    J_diff_cast_ngen = J_nextgen - J_castelli
    H_diff_cast = H_castelli - H_kurucz
    H_diff_ngen = H_nextgen - H_kurucz
    H_diff_cast_ngen = H_nextgen - H_castelli
    Kp_diff_cast = Kp_castelli - Kp_kurucz
    Kp_diff_ngen = Kp_nextgen - Kp_kurucz
    Kp_diff_cast_ngen = Kp_nextgen - Kp_castelli
    Lp_diff_cast = Lp_castelli - Lp_kurucz
    Lp_diff_ngen = Lp_nextgen - Lp_kurucz
    Lp_diff_cast_ngen = Lp_nextgen - Lp_castelli
    
    outdir = '/u/jlu/work/models/test/atmospheres/'
    outsuffix = '_%dpc_%.1fage' % (distance, logAge)
    if AKs != None:
        outsuffix += '_AKs%.1f' % (AKs)
    outsuffix += '.png'

    # Plut luminosity differences:
    py.clf()
    py.plot(temp, Kp_diff_cast, 'r.', label='Castelli - Kurucz')
    py.plot(temp, Kp_diff_ngen, 'b.', label='NextGen - Kurucz')
    py.plot(temp, Kp_diff_cast_ngen, 'g.', label='NextGen - Castelli')
    py.xlabel('Effective Temperature (deg)')
    py.ylabel('Kp Magnitude Difference')
    py.title('Distance = %.1f logAge = %.2f' % (distance/10**3, logAge))
    py.legend(numpoints=1, loc='upper left')
    py.savefig(outdir + 'diff_kp' + outsuffix)
    py.xlim(3000, 10000)
    py.ylim(-0.2, 0.2)
    py.savefig(outdir + 'diff_kp_zoom' + outsuffix)

    py.clf()
    py.plot(temp, H_diff_cast, 'r.', label='Castelli - Kurucz')
    py.plot(temp, H_diff_ngen, 'b.', label='NextGen - Kurucz')
    py.plot(temp, H_diff_cast_ngen, 'g.', label='NextGen - Castelli')
    py.xlabel('Effective Temperature (deg)')
    py.ylabel('H Magnitude Difference')
    py.title('Distance = %.1f logAge = %.2f' % (distance/10**3, logAge))
    py.legend(numpoints=1, loc='upper left')
    py.savefig(outdir + 'diff_h' + outsuffix)
    py.xlim(3000, 10000)
    py.ylim(-0.2, 0.2)
    py.savefig(outdir + 'diff_h_zoom' + outsuffix)

    py.clf()
    py.plot(temp, J_diff_cast, 'r.', label='Castelli - Kurucz')
    py.plot(temp, J_diff_ngen, 'b.', label='NextGen - Kurucz')
    py.plot(temp, J_diff_cast_ngen, 'g.', label='NextGen - Castelli')
    py.xlabel('Effective Temperature (deg)')
    py.ylabel('J Magnitude Difference')
    py.title('Distance = %.1f logAge = %.2f' % (distance/10**3, logAge))
    py.legend(numpoints=1, loc='upper left')
    py.savefig(outdir + 'diff_j' + outsuffix)
    py.xlim(3000, 10000)
    py.ylim(-0.2, 0.2)
    py.savefig(outdir + 'diff_j_zoom' + outsuffix)

    py.clf()
    py.plot(temp, Lp_diff_cast, 'r.', label='Castelli - Kurucz')
    py.plot(temp, Lp_diff_ngen, 'b.', label='NextGen - Kurucz')
    py.plot(temp, Lp_diff_cast_ngen, 'g.', label='NextGen - Castelli')
    py.xlabel('Effective Temperature (deg)')
    py.ylabel('Lp Magnitude Difference')
    py.title('Distance = %.1f logAge = %.2f' % (distance/10**3, logAge))
    py.legend(numpoints=1, loc='upper left')
    py.savefig(outdir + 'diff_lp' + outsuffix)
    py.xlim(3000, 10000)
    py.ylim(-0.2, 0.2)
    py.savefig(outdir + 'diff_lp_zoom' + outsuffix)


    # Now calculate color differences.
    JKp_kurucz = J_kurucz - Kp_kurucz
    JKp_castelli = J_castelli - Kp_castelli
    JKp_nextgen = J_nextgen - Kp_nextgen

    HKp_kurucz = H_kurucz - Kp_kurucz
    HKp_castelli = H_castelli - Kp_castelli
    HKp_nextgen = H_nextgen - Kp_nextgen

    KpLp_kurucz = Kp_kurucz - Lp_kurucz
    KpLp_castelli = Kp_castelli - Lp_castelli
    KpLp_nextgen = Kp_nextgen - Lp_nextgen

    py.clf()
    py.plot(temp, JKp_castelli - JKp_kurucz, 'r.', label='Castelli - Kurucz')
    py.plot(temp, JKp_nextgen - JKp_kurucz, 'b.', label='NextGen - Kurucz')
    py.plot(temp, JKp_nextgen - JKp_castelli, 'g.', label='NextGen - Castelli')
    py.xlabel('Effective Temperature (deg)')
    py.ylabel('J - Kp Color Difference')
    py.title('Distance = %.1f logAge = %.2f' % (distance/10**3, logAge))
    py.legend(numpoints=1, loc='upper left')
    py.savefig(outdir + 'diff_j_kp' + outsuffix)
    py.xlim(3000, 10000)
    py.ylim(-0.2, 0.2)
    py.savefig(outdir + 'diff_j_kp_zoom' + outsuffix)

    py.clf()
    py.plot(temp, HKp_castelli - HKp_kurucz, 'r.', label='Castelli - Kurucz')
    py.plot(temp, HKp_nextgen - HKp_kurucz, 'b.', label='NextGen - Kurucz')
    py.plot(temp, HKp_nextgen - HKp_castelli, 'g.', label='NextGen - Castelli')
    py.xlabel('Effective Temperature (deg)')
    py.ylabel('H - Kp Color Difference')
    py.title('Distance = %.1f logAge = %.2f' % (distance/10**3, logAge))
    py.legend(numpoints=1, loc='upper left')
    py.savefig(outdir + 'diff_h_kp' + outsuffix)
    py.xlim(3000, 10000)
    py.ylim(-0.2, 0.2)
    py.savefig(outdir + 'diff_h_kp_zoom' + outsuffix)

    py.clf()
    py.plot(temp, KpLp_castelli - KpLp_kurucz, 'r.', label='Castelli - Kurucz')
    py.plot(temp, KpLp_nextgen - KpLp_kurucz, 'b.', label='NextGen - Kurucz')
    py.plot(temp, KpLp_nextgen - KpLp_castelli, 'g.', label='NextGen - Castelli')
    py.xlabel('Effective Temperature (deg)')
    py.ylabel('Kp - Lp Color Difference')
    py.title('Distance = %.1f logAge = %.2f' % (distance/10**3, logAge))
    py.legend(numpoints=1, loc='upper left')
    py.savefig(outdir + 'diff_kp_lp' + outsuffix)
    py.xlim(3000, 10000)
    py.ylim(-0.2, 0.2)
    py.savefig(outdir + 'diff_kp_lp_zoom' + outsuffix)


    # Plot the color-magnitude diagrams for each. Rather than the differences.
    py.clf()
    py.semilogx(temp, J_kurucz, 'r.', label='Kurucz')
    py.plot(temp, J_castelli, 'b.', label='Castelli')
    py.plot(temp, J_nextgen, 'g.', label='NextGen')
    py.xlabel('Effective Temperature (deg)')
    py.ylabel('J Magnitude')
    py.title('Distance = %.1f logAge = %.2f' % (distance/10**3, logAge))
    rng = py.axis()
    py.axis([rng[1], rng[0], rng[3], rng[2]])
    py.legend(numpoints=1, loc='lower left')
    py.savefig(outdir + 'temp_j' + outsuffix)

    py.clf()
    py.semilogx(temp, H_kurucz, 'r.', label='Kurucz')
    py.plot(temp, H_castelli, 'b.', label='Castelli')
    py.plot(temp, H_nextgen, 'g.', label='NextGen')
    py.xlabel('Effective Temperature (deg)')
    py.ylabel('H Magnitude')
    py.title('Distance = %.1f logAge = %.2f' % (distance/10**3, logAge))
    rng = py.axis()
    py.axis([rng[1], rng[0], rng[3], rng[2]])
    py.legend(numpoints=1, loc='lower left')
    py.savefig(outdir + 'temp_h' + outsuffix)

    py.clf()
    py.semilogx(temp, Kp_kurucz, 'r.', label='Kurucz')
    py.plot(temp, Kp_castelli, 'b.', label='Castelli')
    py.plot(temp, Kp_nextgen, 'g.', label='NextGen')
    py.xlabel('Effective Temperature (deg)')
    py.ylabel('Kp Magnitude')
    py.title('Distance = %.1f logAge = %.2f' % (distance/10**3, logAge))
    rng = py.axis()
    py.axis([rng[1], rng[0], rng[3], rng[2]])
    py.legend(numpoints=1, loc='lower left')
    py.savefig(outdir + 'temp_kp' + outsuffix)

    py.clf()
    py.semilogx(temp, Lp_kurucz, 'r.', label='Kurucz')
    py.plot(temp, Lp_castelli, 'b.', label='Castelli')
    py.plot(temp, Lp_nextgen, 'g.', label='NextGen')
    py.xlabel('Effective Temperature (deg)')
    py.ylabel('Lp Magnitude')
    py.title('Distance = %.1f logAge = %.2f' % (distance/10**3, logAge))
    rng = py.axis()
    py.axis([rng[1], rng[0], rng[3], rng[2]])
    py.legend(numpoints=1, loc='lower left')
    py.savefig(outdir + 'temp_lp' + outsuffix)

    py.clf()
    py.semilogx(temp, JKp_kurucz, 'r.', label='Kurucz')
    py.plot(temp, JKp_castelli, 'b.', label='Castelli')
    py.plot(temp, JKp_nextgen, 'g.', label='NextGen')
    py.xlabel('Effective Temperature (deg)')
    py.ylabel('J - Kp Color')
    py.title('Distance = %.1f logAge = %.2f' % (distance/10**3, logAge))
    rng = py.axis()
    py.xlim(rng[1], rng[0])
    py.legend(numpoints=1, loc='lower left')
    py.savefig(outdir + 'temp_jkp' + outsuffix)

    py.clf()
    py.semilogx(temp, HKp_kurucz, 'r.', label='Kurucz')
    py.plot(temp, HKp_castelli, 'b.', label='Castelli')
    py.plot(temp, HKp_nextgen, 'g.', label='NextGen')
    py.xlabel('Effective Temperature (deg)')
    py.ylabel('H - Kp Color')
    py.title('Distance = %.1f logAge = %.2f' % (distance/10**3, logAge))
    rng = py.axis()
    py.xlim(rng[1], rng[0])
    py.legend(numpoints=1, loc='lower left')
    py.savefig(outdir + 'temp_hkp' + outsuffix)

    py.clf()
    py.semilogx(temp, KpLp_kurucz, 'r.', label='Kurucz')
    py.plot(temp, KpLp_castelli, 'b.', label='Castelli')
    py.plot(temp, KpLp_nextgen, 'g.', label='NextGen')
    py.xlabel('Effective Temperature (deg)')
    py.ylabel('Kp - Lp Color')
    py.title('Distance = %.1f logAge = %.2f' % (distance/10**3, logAge))
    rng = py.axis()
    py.xlim(rng[1], rng[0])
    py.legend(numpoints=1, loc='lower left')
    py.savefig(outdir + 'temp_kplp' + outsuffix)

