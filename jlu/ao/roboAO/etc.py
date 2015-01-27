import numpy as np
import pylab as py
from astropy.table import Table
from pysynphot import spectrum
from pysynphot import observation
from pysynphot import units
import pysynphot

# Zeropoints from Bessel et al. 1998
#  photons photons cm^-2 s^-1 A^-1
ZP_Vega = {'B': 1392.6, 'V': 995.5, 'R': 702.0, 'I': 452.0,
           'J':  193.1, 'H':  93.3, 'K':  43.6}

# Convert from m_AB to m_Vega. Values reported below are
# m_AB - m_Vega and were taken from:
# http://www.astronomy.ohio-state.edu/~martini/usefuldata.html
# which mostly come from Blanton et al. 2007
Vega_to_AB = {'B': 0.79, 'V': 0.02, 'R': 0.21, 'I': 0.45,
              'J': 0.91, 'H': 1.39, 'K': 1.85}

def etc_uh_roboAO(mag, filt, phot_sys='Vega'):
    """
    Exposure time calculator for a UH Robo-AO system and a NIR IFU spectrograph.

    phot_sys - 'Vega' (default) or 'AB'
    """
    ifu_throughput = 0.35
    ao_throughput = 0.55
    tel_throughput = 0.85**2

    # Convert to Vega if in AB
    if phot_sys != 'Vega':
        mag += Vega_to_AB[filt]

    # Convert into flux units, still above the atmosphere.
    flux = ZP_Vega[filt] * 10**(-mag/2.5)   # photons cm^-2 s^-1 A^-1

    
    
    
    pass


def read_mk_sky_emission_ir(pwv=3, airmass=1.5):
    pwv_suffix = {1.0: '10', 1.6: '16', 3.0: '30', 5.0: '50'}
    am_suffix = {1.0: '10', 1.5: '15', 2.0: '20'}
    
    if pwv not in pwv_suffix.keys():
        print 'Invalid precipatable water vapor (PWV): ', pwv
        print 'Choose from:'
        print pwv_suffix.keys()
        return
    
    if airmass not in am_suffix.keys():
        print 'Invalid airmass (AM): ', airmass
        print 'Choose from:'
        print am_suffix.keys()
        return
    
    mk_sky_file = 'mk_skybg_zm_'
    mk_sky_file += pwv_suffix[pwv] + '_'
    mk_sky_file += am_suffix[airmass] + '_ph.dat'
    
    try:
        t = Table.read(mk_sky_file + '.fits')
    except IOError:
        t = Table.read(mk_sky_file + '.txt', format='ascii')
        t.rename_column('col1', 'wave') # Units in nm
        t.rename_column('col2', 'flux') # Units in ph/sec/arcsec^2/nm/m^2

        t['wave'].unit = 'nm'
        t['flux'].unit = 'photons s^-1 arcsec^-2 nm^-1 m^-2'
        t.write(mk_sky_file + '.fits')

    # Convert to photlam flux units (m->cm and nm->Ang).
    # Drop the arcsec density... but sill there
    t['flux'] *= (1.0 / 100.0)**2 * (1.0 / 10.0)
    t['flux'].unit = 'photons cm^-2 s^-1 Ang^-1 arcsec^-2'
    spec = spectrum.ArraySourceSpectrum(wave=t['wave'], flux=t['flux'], 
                                        waveunits='nm', fluxunits=units.Photlam, 
                                        name='Maunakea Sky')
    
    return spec

def read_mk_sky_transmission():
    pwv_suffix = {1.0: '10', 1.6: '16', 3.0: '30', 5.0: '50'}
    am_suffix = {1.0: '10', 1.5: '15', 2.0: '20'}

    if pwv not in pwv_suffix.keys():
        print 'Invalid precipatable water vapor (PWV): ', pwv
        print 'Choose from:'
        print pwv_suffix.keys()
        return
    
    if airmass not in am_suffix.keys():
        print 'Invalid airmass (AM): ', airmass
        print 'Choose from:'
        print am_suffix.keys()
        return
    
    mk_sky_file = 'mktrans_zm_'
    mk_sky_file += pwv_suffix[pwv] + '_'
    mk_sky_file += am_suffix[airmass] + '.dat'

    try:
        t = Table.read(mk_sky_file + '.fits')
    except IOError:
        t = Table.read(mk_sky_file + '.txt', format='ascii')
        t.rename_column('col1', 'wave') # Units in microns
        t.rename_column('col2', 'trans') # Units in percent transmitted

        t['wave'].unit = 'microns'
        t.write(mk_sky_file + '.fits')

    spec = spectrum.ArraySpectralElement(wave=t['wave'], throughput=t['trans'],
                                         waveunits='microns', name='Maunakea Trans')

    return spec


    
def Vega():
    # Use Vega as our zeropoint... assume V=0.03 mag and all colors = 0.0
    temperature = 9550
    metallicity = -0.5
    gravity = 3.95
    vega = pysynphot.Icat('k93models', temperature, metallicity, gravity)
    
    vega = spectrum.trimSpectrum(vega, 3000, 30000)

    # This is (R/d)**2 as reported by Girardi et al. 2002, page 198, col 1.
    # and is used to convert to flux observed at Earth.
    vega *= 6.247e-17 
    
    return vega
