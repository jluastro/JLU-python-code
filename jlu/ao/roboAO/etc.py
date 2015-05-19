import numpy as np
import pylab as py
from astropy.table import Table
from pysynphot import spectrum
from pysynphot import observation
from pysynphot import units
import pysynphot
import pdb
import math
import inspect
import os
import time

# Convert from m_AB to m_Vega. Values reported below are
# m_AB - m_Vega and were taken from:
# http://www.astronomy.ohio-state.edu/~martini/usefuldata.html
# which mostly come from Blanton et al. 2007
Vega_to_AB = {'B': 0.79, 'V': 0.02, 'R': 0.21, 'I': 0.45,
              'J': 0.91, 'H': 1.39, 'K': 1.85}

#
# Switch to 30 minutes integration time
# Spectral Resolution = 30
# Make a plot that shows the error on the slope of the spectrum
#    - R=30 and Z+Y+J+H (all at once)
#    - R=3000 for Z and then H broadband between the OH lines
# Email Brent and John and Shelley and Christoph outputs
#

def etc_uh_roboAO(mag, filt_name, tint, aper_radius=0.15, phot_sys='Vega', spec_res=300):
    """
    Exposure time calculator for a UH Robo-AO system and a NIR IFU spectrograph.

    phot_sys - 'Vega' (default) or 'AB'
    """
    
    ifu_throughput = 0.35
    ao_throughput = 0.55
    tel_throughput = 0.85**2
    sys_throughput = ifu_throughput * ao_throughput * tel_throughput
    
    # TO DO Need to add telescope secondary obscuration to correct the area.
    tel_area = math.pi * (2.2 * 100. / 2.)**2   # cm^2 for UH 2.2m tel
    read_noise = 3.0       # electrons
    dark_current = 0.01    # electrons s^-1

    # Get the filter
    filt = pysynphot.ObsBandpass(filt_name)

    # Calculate the wave set for the IFU. Include Nyquist sampled pixels.
    dlamda = (filt.avgwave() / spec_res) / 2.0
    ifu_wave = np.arange(filt.wave.min(), filt.wave.max()+dlamda, dlamda)
    
    # Get the Earth transmission spectrum. Sample everything
    # onto this grid.
    earth_trans =  read_mk_sky_transmission()

    # Get the Earth background emission spectrum
    earth_bkg = read_mk_sky_emission_ir()
    earth_bkg.resample(ifu_wave)

    # Convert to Vega if in AB
    if phot_sys != 'Vega':
        mag += Vega_to_AB[filt_name]

    # Assume this star is an A0V star, so just scale a Vega spectrum
    # to the appropriate magnitude. Rescale to match the magnitude.
    # pysynphot.renorm() and pysynphot.setMagnitude are all broken.
    star = pysynphot.Vega.renorm(mag, 'vegamag', filt)  # erg cm^2 s^-1 A^-1

    # Observe the star and background through a filter and resample
    # at the IFU spectral sampling.
    star_obs = observation.Observation(star, filt, binset=ifu_wave)
    bkg_obs = observation.Observation(earth_bkg, filt, binset=ifu_wave)
    vega_obs = observation.Observation(pysynphot.Vega, filt, binset=ifu_wave)

    # Propogate the star flux and background through the
    # atmosphere and telescope. 
    star_obs *= earth_trans                       # erg s^-1 A^-1 cm^-2
    star_obs *= tel_area * sys_throughput         # erg s^-1 A^-1
    vega_obs *= earth_trans                       # erg s^-1 A^-1 cm^-2
    vega_obs *= tel_area * sys_throughput         # erg s^-1 A^-1
    bkg_obs *= tel_area * sys_throughput          # erg s^-1 A^-1 arcsec^-2

    # Convert them into photlam
    star_obs.convert('photlam')            # photon s^-1 A^-1
    vega_obs.convert('photlam')
    bkg_obs.convert('photlam')             # photon s^-1 A^-1 arcsec^-2

    # Pull the arrays out of the Observation objects
    star_counts = star_obs.binflux
    bkg_counts = bkg_obs.binflux
    vega_counts = vega_obs.binflux

    # Integrate each spectral channel using the ifu_wave (dlamda defined above).
    star_counts *= dlamda                     # photon s^-1
    bkg_counts *= dlamda                      # photon s^-1 arcsec^-2
    vega_counts *= dlamda                     # photon s^-1 arcsec^-2

    # Integrate over the aperture for the background and make
    # an aperture correction for the star.
    ee = get_roboAO_ee(mag, filt_name, aper_radius)
    aper_area = math.pi * aper_radius**2
    star_counts *= ee                         # photon s^-1
    vega_counts *= ee
    bkg_counts *= aper_area                   # photon s^-1

    pix_scale = 0.075                      # arcsec per pixel
    npix = aper_area / pix_scale**2

    vega_mag = 0.03
    star_mag = -2.5 * math.log10(star_counts.sum() / vega_counts.sum()) + vega_mag
    bkg_mag = -2.5 * math.log10(bkg_counts.sum() / vega_counts.sum()) + vega_mag
    print star_mag, bkg_mag
    
    signal = star_counts * tint               # photon
    bkg = bkg_counts * tint                   # photon
    
    noise_variance = signal.copy()
    noise_variance += bkg
    noise_variance += read_noise**2 * npix
    noise_variance += dark_current * tint * npix
    noise = noise_variance**0.5
    snr_spec = signal / noise
    
    
    # Calculate average signal-to-noise per spectral channel        
    avg_signal = signal.sum()
    avg_noise = noise.sum()
    avg_snr = avg_signal / avg_noise
    print 'signal = ', avg_signal, ' bkg = ', bkg.mean(), ' SNR = ', avg_snr

    # Inter-OH gives an overal reduction in background of
    # 2.3 mag at H - probably slightly less than this because this was with NIRSPEC
    # 2.0 mag at J

    # Do R=100
    # Do R=30
    
    return avg_snr, star_mag, bkg_mag, ifu_wave, signal, bkg, snr_spec


def read_mk_sky_emission_ir(pwv=1.6, airmass=1.5):
    path_to_file = inspect.getsourcefile(Vega)
    directory = os.path.split(path_to_file)[0] + '/maunakea_files/'
    
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
        t = Table.read(directory + mk_sky_file + '.fits')
    except IOError:
        t = Table.read(directory + mk_sky_file, format='ascii')
        t.rename_column('col1', 'wave') # Units in nm
        t.rename_column('col2', 'flux') # Units in ph/sec/arcsec^2/nm/m^2

        t['wave'].unit = 'nm'
        t['flux'].unit = 'photons s^-1 arcsec^-2 nm^-1 m^-2'
        t.write(directory + mk_sky_file + '.fits')

    # Convert to photlam flux units (m->cm and nm->Ang).
    # Drop the arcsec density... but sill there
    t['flux'] *= (1.0 / 100.0)**2 * (1.0 / 10.0)
    t['flux'].unit = 'photons cm^-2 s^-1 Ang^-1 arcsec^-2'
    spec = spectrum.ArraySourceSpectrum(wave=t['wave'], flux=t['flux'], 
                                        waveunits='nm', fluxunits=units.Photlam, 
                                        name='Maunakea Sky')
    spec.convert(units.Angstrom)

    return spec

def read_mk_sky_transmission(pwv=1.6, airmass=1.5):
    path_to_file = inspect.getsourcefile(Vega)
    directory = os.path.split(path_to_file)[0] + '/maunakea_files/'
    
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
        t = Table.read(directory + mk_sky_file + '.fits')
    except IOError:
        t = Table.read(directory + mk_sky_file, format='ascii')
        t.rename_column('col1', 'wave') # Units in microns
        t.rename_column('col2', 'trans') # Units in percent transmitted

        t['wave'].unit = 'micron'
        t.write(directory + mk_sky_file + '.fits')

    spec = spectrum.ArraySpectralElement(wave=t['wave'], throughput=t['trans'],
                                         waveunits='micron', name='Maunakea Trans')
    spec.convert(units.Angstrom)

    return spec


    
def Vega():
    # Use Vega as our zeropoint... assume V=0.03 mag and all colors = 0.0
    temperature = 9550
    metallicity = -0.5
    gravity = 3.95
    vega = pysynphot.Icat('k93models', temperature, metallicity, gravity)
    
    vega = spectrum.trimSpectrum(vega, 8000, 25000)

    # This is (R/d)**2 as reported by Girardi et al. 2002, page 198, col 1.
    # and is used to convert to flux observed at Earth.
    vega *= 6.247e-17 
    
    return vega

vega = Vega()


# Little helper utility to get all the bandpass/zeropoint info.
def get_filter_info(name, vega=vega):
    filt = pysynphot.ObsBandpass(name)

    vega_obs = observation.Observation(vega, filt, binset=filt.wave)

    # Vega_flux in Flam units.
    vega_flux = vega_obs.effstim('flam')
    vega_mag = 0.03

    return filt, vega_flux, vega_mag

def get_roboAO_ee(mag, filt_name, aper_radius):
    """
    Get the ensquared energy.
    """
    aper_radius_valid = [0.075, 0.15]
    
    if aper_radius not in aper_radius_valid:
        print 'Invalid aperture radius. Choose from:'
        print aper_radius_valid
    
    guide_mag = np.array([10, 17, 18, 19, 20], dtype=float)
    roboAO_ee_075 = {}
    roboAO_ee_075['Z'] = np.array([0.08, 0.06, 0.05, 0.04, 0.03])
    roboAO_ee_075['Y'] = np.array([0.11, 0.08, 0.06, 0.05, 0.03])
    roboAO_ee_075['J'] = np.array([0.12, 0.09, 0.08, 0.06, 0.04])
    roboAO_ee_075['H'] = np.array([0.11, 0.10, 0.08, 0.06, 0.04])

    roboAO_ee_150 = {}
    roboAO_ee_150['Z'] = np.array([0.19, 0.17, 0.15, 0.12, 0.10])
    roboAO_ee_150['Y'] = np.array([0.26, 0.22, 0.20, 0.16, 0.12])
    roboAO_ee_150['J'] = np.array([0.33, 0.28, 0.25, 0.20, 0.14])
    roboAO_ee_150['H'] = np.array([0.35, 0.31, 0.27, 0.22, 0.16])
    
    if aper_radius == 0.15:
        roboAO_ee = roboAO_ee_150
    else:
        roboAO_ee = roboAO_ee_075

    ee = np.interp(mag, guide_mag, roboAO_ee[filt_name])

    return ee
    
def make_sensitivity_curves(tint=300, spec_res=300, aper_radius=0.075):
    mag = np.arange(10, 22)
    snr_J = np.zeros(len(mag), dtype=float)
    snr_H = np.zeros(len(mag), dtype=float)

    for mm in range(len(mag)):
        print 'Mag: ', mag[mm]
        blah = etc_uh_roboAO(mag[mm], 'J', tint,
                             spec_res=spec_res, aper_radius=aper_radius)
        snr_J[mm]  = blah[0]
        blah = etc_uh_roboAO(mag[mm], 'H', tint,
                             spec_res=spec_res, aper_radius=aper_radius)
        snr_H[mm]  = blah[0]

    py.clf()
    py.plot(mag, snr_J, label='J')
    py.plot(mag, snr_H, label='H')
    py.xlabel('Magnitude')
    py.ylabel('SNR')
    py.title('Tint={0:d} s, R={1:d}, aper={2:0.3f}"'.format(tint, spec_res, aper_radius))
    out_file = 'roboAO_sensitivity_t{0:d}_R{1:d}_ap{2:0.3f}'.format(tint, spec_res, aper_radius)
    print out_file

    py.savefig(out_file + '.png')
        

    
    
