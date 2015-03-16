import numpy as np
import pylab as py
from astropy.table import Table
from astropy.table import Column
from pysynphot import spectrum
from pysynphot import observation
from pysynphot import units
import pysynphot
import pdb
import math
import inspect
import os
import time
import pickle
from astropy.modeling.functional_models import Gaussian2D


# Convert from m_AB to m_Vega. Values reported below are
# m_AB - m_Vega and were taken from:
# http://www.astronomy.ohio-state.edu/~martini/usefuldata.html
# which mostly come from Blanton et al. 2007
Vega_to_AB = {'B': 0.79, 'V': 0.02, 'R': 0.21, 'I': 0.45,
              'J': 0.91, 'H': 1.39, 'K': 1.85}

# Central wavelength in Angstroms.
filt_wave = {'Z': 8818.1, 'Y': 10368.1, 'J': 12369.9, 'H': 16464.4}

#
# switch to 30 minutes integration time
# Spectral Resolution = 30
# Make a plot that shows the error on the slope of the spectrum
#    - R=30 and Z+Y+J+H (all at once)
#    - R=3000 for Z and then H broadband between the OH lines
# Email Brent and John and Shelley and Christoph outputs
#

def etc_uh_roboAO(mag, filt_name, tint, aper_radius=0.15, phot_sys='Vega', spec_res=300, 
                  seeing_limited=False):
    """
    Exposure time calculator for a UH Robo-AO system and a NIR IFU spectrograph.

    phot_sys - 'Vega' (default) or 'AB'
    """
    
    ifu_throughput = 0.35
    ao_throughput = 0.55
    tel_throughput = 0.85**2

    if seeing_limited:
        sys_throughput = ifu_throughput * tel_throughput
    else:
        sys_throughput = ifu_throughput * ao_throughput * tel_throughput
    
    # TO DO Need to add telescope secondary obscuration to correct the area.
    tel_area = math.pi * (2.2 * 100. / 2.)**2   # cm^2 for UH 2.2m tel
    read_noise = 3.0       # electrons
    dark_current = 0.01    # electrons s^-1

    # Get the filter
    if filt_name == 'Z' or filt_name == 'Y':
        filt = get_ukirt_filter(filt_name)
    else:
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
    vega = pysynphot.FileSpectrum(pysynphot.locations.VegaFile)
    vega_obs = observation.Observation(vega, filt, binset=ifu_wave)

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
    if seeing_limited:
        ee = get_seeing_ee(filt_name, aper_radius)
    else:
        ee = get_roboAO_ee(mag, filt_name, aper_radius)
        
    aper_area = math.pi * aper_radius**2
    star_counts *= ee                         # photon s^-1
    # vega_counts *= ee
    bkg_counts *= aper_area                   # photon s^-1

    pix_scale = 0.150                      # arcsec per pixel
    if seeing_limited:
        pix_scale = 0.200
        
    npix = aper_area / pix_scale**2

    vega_mag = 0.03
    star_mag = -2.5 * math.log10(star_counts.sum() / vega_counts.sum()) + vega_mag
    bkg_mag = -2.5 * math.log10(bkg_counts.sum() / vega_counts.sum()) + vega_mag
    
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

def get_ukirt_filter(filt_name):
    path_to_file = inspect.getsourcefile(Vega)
    directory = os.path.split(path_to_file)[0] + '/'
    
    filt_file = 'ukirt_'
    filt_file += filt_name.lower()
    filt_file += '.fits'
    
    t = Table.read(directory + filt_file)

    # Convert to photlam flux units (m->cm and nm->Ang).
    # Drop the arcsec density... but sill there
    spec = spectrum.ArraySpectralElement(wave=t['col1'], throughput=t['col2'], 
                                        waveunits='Angstrom', 
                                        name='UKIRT_'+filt_name)

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

# vega = Vega()


# Little helper utility to get all the bandpass/zeropoint info.
def get_filter_info(name, vega=None):
    filt = pysynphot.ObsBandpass(name)

    vega_obs = observation.Observation(vega, filt, binset=filt.wave)

    # Vega_flux in Flam units.
    vega_flux = vega_obs.effstim('flam')
    vega_mag = 0.03

    return filt, vega_flux, vega_mag

def get_seeing_ee(filt_name, aper_radius, overwrite=False):
    """
    Get the ensquared energy for a seeing limited 
    """
    psf_file = 'seeing_psf_{0:s}.pickle'.format(filt_name)

    if overwrite or not os.path.exists(psf_file):
        print 'Making new PSF', overwrite, os.path.exists(psf_file)
        make_seeing_psf(filt_name)

    _psf = open(psf_file, 'r')
    r = pickle.load(_psf)
    ee = pickle.load(_psf)
    _psf.close()
            
    # Integrate out to the desired radius
    dr_idx = np.abs(r - aper_radius).argmin()

    ee_good = ee[dr_idx]

    return ee_good
    
def make_seeing_psf(filt_name):
    psf_file = 'seeing_psf_{0:s}.pickle'.format(filt_name)

    seeing = 0.8 # arcsec specified at 5000 Angstrom
    wave = filt_wave[filt_name] # In Angstroms
    print 'Seeing at 5000 Angstroms:', seeing

    seeing *= (wave / 5000)**(-1./5.)
    print 'Seeing at {0:s} band:'.format(filt_name), seeing
    
    # Make a very over-sampled PSF and do integrals
    # on that... faster than scipy.integrate.
    print 'Prep'
    pix_scale = 0.01 # arcsec
    
    xy1d = np.arange(-10*seeing, 10*seeing, pix_scale)
    
    x, y = np.meshgrid(xy1d, xy1d)
    
    # Use a 2D Gaussian as our PSF.
    print 'Make Gaussian sigma=', seeing/2.35
    psf = Gaussian2D.eval(x, y, 1, 0, 0, seeing/2.35, seeing/2.35, 0)
    
    # Integrate over each pixel
    print 'Integrate and normalize'
    psf *= pix_scale**2
    
    # Normalize
    psf /= psf.sum()
    
    # Get the radius of each pixel
    r = np.hypot(x, y)

    # Make an encircled energy curve.
    r_save = np.arange(0.025, 2, 0.025)
    ee_save = np.zeros(len(r_save), dtype=float)

    for rr in range(len(r_save)):
        print 'Integrating for ', r_save[rr]
        idx = np.where(r < r_save[rr])

        ee_save[rr] = psf[idx].sum()
        
    print 'Save'
    _psf = open(psf_file, 'w')
    pickle.dump(r_save, _psf)
    pickle.dump(ee_save, _psf)
    _psf.close()

    return

    
def get_roboAO_ee(mag, filt_name, aper_radius):
    """
    Get the encircled energy.
    """
    aper_radius_valid = [0.0375, 0.075, 0.15]
    
    if aper_radius not in aper_radius_valid:
        print 'Invalid aperture radius. Choose from:'
        print aper_radius_valid
    
    guide_mag = np.array([10, 17, 18, 19, 20], dtype=float)

    # These are ensquared energies; but I am assuming they are encircled energies.
    # These come from Christoph's spreadsheet; but they have been renamed (from 
    # his column names) based on aperture radius rather than diameter as he had it.
    roboAO_ee_0375 = {}
    roboAO_ee_0375['Z'] = np.array([0.08, 0.06, 0.05, 0.04, 0.03])
    roboAO_ee_0375['Y'] = np.array([0.11, 0.08, 0.06, 0.05, 0.03])
    roboAO_ee_0375['J'] = np.array([0.12, 0.09, 0.08, 0.06, 0.04])
    roboAO_ee_0375['H'] = np.array([0.11, 0.10, 0.08, 0.06, 0.04])

    roboAO_ee_075 = {}
    roboAO_ee_075['Z'] = np.array([0.19, 0.17, 0.15, 0.12, 0.10])
    roboAO_ee_075['Y'] = np.array([0.26, 0.22, 0.20, 0.16, 0.12])
    roboAO_ee_075['J'] = np.array([0.33, 0.28, 0.25, 0.20, 0.14])
    roboAO_ee_075['H'] = np.array([0.35, 0.31, 0.27, 0.22, 0.16])
    
    roboAO_ee_150 = {}
    roboAO_ee_150['Z'] = np.array([0.35, 0.35, 0.35, 0.34, 0.31])
    roboAO_ee_150['Y'] = np.array([0.42, 0.42, 0.43, 0.41, 0.36])
    roboAO_ee_150['J'] = np.array([0.54, 0.54, 0.52, 0.42, 0.43])
    roboAO_ee_150['H'] = np.array([0.66, 0.64, 0.62, 0.57, 0.48])
    
    if aper_radius == 0.15:
        roboAO_ee = roboAO_ee_150
    if aper_radius == 0.075:
        roboAO_ee = roboAO_ee_075
    if aper_radius == 0.0375:
        roboAO_ee = roboAO_ee_0375

    ee = np.interp(mag, guide_mag, roboAO_ee[filt_name])

    return ee
    
def make_sensitivity_curves(tint=1200, spec_res=100, aper_radius=0.15, seeing_limited=False):
    mag = np.arange(10, 22)
    snr_y = np.zeros(len(mag), dtype=float)
    snr_j = np.zeros(len(mag), dtype=float)
    snr_h = np.zeros(len(mag), dtype=float)
    snr_sum_y = np.zeros(len(mag), dtype=float)
    snr_sum_j = np.zeros(len(mag), dtype=float)
    snr_sum_h = np.zeros(len(mag), dtype=float)
    bkg_y = np.zeros(len(mag), dtype=float)
    bkg_j = np.zeros(len(mag), dtype=float)
    bkg_h = np.zeros(len(mag), dtype=float)
    star_y = np.zeros(len(mag), dtype=float)
    star_j = np.zeros(len(mag), dtype=float)
    star_h = np.zeros(len(mag), dtype=float)

    spec_y_tab = None
    spec_j_tab = None
    spec_h_tab = None

    # Calculate the number of supernovae.
    N_SNe = 4500.0 * 0.6 * 10**(mag - 18.9)

    out_file = 'roboAO_sensitivity_t{0:d}_R{1:d}_ap{2:0.3f}'.format(tint, spec_res, aper_radius)
    
    # Save the output to a table.
    _out = open(out_file + '.txt', 'w')

    meta1 = '# tint = {0:5d}, R = {1:5d}, apRad = {2:5.3f}"\n'
    _out.write(meta1.format(tint, spec_res, aper_radius))
    _out.write('# Sensitivity integrated over broad band.')
    
    hdr = '{0:5s}  {1:6s}  {2:5s}  {3:5s}  {4:5s}  {5:5s}  {6:5s}  {7:5s}\n'
    fmt = '{0:5.1f}  {1:6.1f}  {2:5.1f}  {3:5.1f}  {4:5.1f}  {5:5.1f}  {6:5.1f}  {7:5.1f}\n'
    _out.write(hdr.format('# Mag', 'N_SNe', 'J_SNR', 'H_SNR', 'J_ms', 'H_ms', 'J_mb', 'H_mb'))
               
    for mm in range(len(mag)):
        print 'Mag: ', mag[mm]
        blah_y = etc_uh_roboAO(mag[mm], 'Y', tint,
                               spec_res=spec_res, aper_radius=aper_radius, 
                               seeing_limited=seeing_limited)
        blah_j = etc_uh_roboAO(mag[mm], 'J', tint,
                               spec_res=spec_res, aper_radius=aper_radius,
                               seeing_limited=seeing_limited)
        blah_h = etc_uh_roboAO(mag[mm], 'H', tint,
                               spec_res=spec_res, aper_radius=aper_radius,
                               seeing_limited=seeing_limited)
        
        col_y_suffix = '_Y_{0:d}'.format(mag[mm])
        col_j_suffix = '_J_{0:d}'.format(mag[mm])
        col_h_suffix = '_H_{0:d}'.format(mag[mm])

        spec_signal_y = Column(name='sig'+col_y_suffix, data=blah_y[4])
        spec_signal_j = Column(name='sig'+col_j_suffix, data=blah_j[4])
        spec_signal_h = Column(name='sig'+col_h_suffix, data=blah_h[4])
        spec_bkg_y = Column(name='bkg'+col_y_suffix, data=blah_y[5])
        spec_bkg_j = Column(name='bkg'+col_j_suffix, data=blah_j[5])
        spec_bkg_h = Column(name='bkg'+col_h_suffix, data=blah_h[5])
        spec_snr_y = Column(name='snr'+col_y_suffix, data=blah_y[6])
        spec_snr_j = Column(name='snr'+col_j_suffix, data=blah_j[6])
        spec_snr_h = Column(name='snr'+col_h_suffix, data=blah_h[6])

        
        if spec_y_tab == None:
            spec_y_tab = Table([blah_y[3]], names=['wave_Y'])
        if spec_j_tab == None:
            spec_j_tab = Table([blah_j[3]], names=['wave_J'])
        if spec_h_tab == None:
            spec_h_tab = Table([blah_h[3]], names=['wave_H'])

        spec_y_tab.add_columns([spec_signal_y, spec_bkg_y, spec_snr_y])
        spec_j_tab.add_columns([spec_signal_j, spec_bkg_j, spec_snr_j])
        spec_h_tab.add_columns([spec_signal_h, spec_bkg_h, spec_snr_h])

        snr_y[mm]  = blah_y[0]
        snr_j[mm]  = blah_j[0]
        snr_h[mm]  = blah_h[0]
        snr_sum_y[mm] = math.sqrt((spec_snr_y**2).sum())
        snr_sum_j[mm] = math.sqrt((spec_snr_j**2).sum())
        snr_sum_h[mm] = math.sqrt((spec_snr_h**2).sum())

        star_y[mm]  = blah_y[1]
        star_j[mm]  = blah_j[1]
        star_h[mm]  = blah_h[1]
        bkg_y[mm]  = blah_y[2]
        bkg_j[mm]  = blah_j[2]
        bkg_h[mm]  = blah_h[2]

    avg_tab = Table([mag, snr_y, snr_j, snr_h, 
                     snr_sum_y, snr_sum_j, snr_sum_h,
                     star_y, star_j, star_h, bkg_y, bkg_j, bkg_h],
                    names=['mag', 'snr_y', 'snr_j', 'snr_h', 
                           'snr_sum_y', 'snr_sum_j', 'snr_sum_h',
                           'star_y', 'star_j', 'star_h', 
                           'bkg_y', 'bkg_j', 'bkg_h'])


    out_file = 'roboAO_sensitivity_t{0:d}_R{1:d}_ap{2:0.3f}'.format(tint, spec_res, aper_radius)

    if seeing_limited:
        out_file += '_seeing'
    
    # Save the tables
    spec_y_tab.write(out_file + '_spec_y_tab.fits', overwrite=True)
    spec_j_tab.write(out_file + '_spec_j_tab.fits', overwrite=True)
    spec_h_tab.write(out_file + '_spec_h_tab.fits', overwrite=True)
    avg_tab.write(out_file + '_avg_tab.fits', overwrite=True)

    return


def plot_sensitivity_curves(tint=1200, spec_res=100, aper_radius=0.15, seeing_limited=False):
    in_file = 'roboAO_sensitivity_t{0:d}_R{1:d}_ap{2:0.3f}'.format(tint, spec_res, aper_radius)

    if seeing_limited:
        in_file += '_seeing'

    avg_tab = Table.read(in_file + '_avg_tab.fits')

    # Calculate the band-integrated SNR for each magnitude bin and filter.
    mag = avg_tab['mag']

    N_SNe = 4500.0 * 10**(0.6*(mag - 18.9))

    hdr1 = '# {0:3s}  {1:15s}   {2:15s}   {3:15s}  {4:8s}'
    hdr2 = '# {0:3s}  {1:7s} {2:7s}   {3:7s} {4:7s}   {5:7s} {6:7s}  {7:8s}'
    print hdr1.format('Mag', '  Y-band SNR', '  J-band SNR', '  H-band SNR', 'Number')
    print hdr2.format('', 'Per_Ch', 'Summed', 'Per_Ch', 'Summed', 'Per_Ch', 'Summed', 'of SN Ia')
    print hdr2.format('---', '-------', '-------', '-------', '-------', '-------', '-------', '--------')

    for mm in range(len(mag)):
        fmt = '  {0:3d}  {1:7.1f} {2:7.1f}   {3:7.1f} {4:7.1f}   {5:7.1f} {6:7.1f}  {7:8.0f}'
        print fmt.format(mag[mm], 
                         avg_tab['snr_y'][mm], avg_tab['snr_sum_y'][mm],
                         avg_tab['snr_j'][mm], avg_tab['snr_sum_j'][mm],
                         avg_tab['snr_h'][mm], avg_tab['snr_sum_h'][mm],
                         N_SNe[mm])

    py.figure(1)
    py.clf()
    py.semilogy(avg_tab['mag'], avg_tab['snr_y'], label='Y')
    py.plot(avg_tab['mag'], avg_tab['snr_j'], label='J')
    py.plot(avg_tab['mag'], avg_tab['snr_h'], label='H')
    py.xlabel('Magnitude')
    py.ylabel('SNR Per Spectral Channel')
    py.title('Tint={0:d} s, R={1:d}, aper={2:0.3f}"'.format(tint, spec_res, aper_radius))
    py.legend()
    py.ylim(1, 1e4)
    py.savefig(in_file + '_snr_per.png')
    
    py.clf()
    py.semilogy(avg_tab['mag'], avg_tab['snr_sum_y'], label='Y')
    py.plot(avg_tab['mag'], avg_tab['snr_sum_j'], label='J')
    py.plot(avg_tab['mag'], avg_tab['snr_sum_h'], label='H')
    py.xlabel('Magnitude')
    py.ylabel('SNR Over Filter')
    py.ylim(1, 1e4)
    py.legend()
    py.title('Tint={0:d} s, R={1:d}, aper={2:0.3f}"'.format(tint, spec_res, aper_radius))
    py.savefig(in_file + '_snr_sum.png')

    # Two-panel - in the presentation
    py.close(2)
    py.figure(2, figsize=(7,8))
    py.clf()
    py.subplots_adjust(hspace=0.07, bottom=0.1)
    ax1 = py.subplot(211)
    py.ylim(1e1, 1e5)
    ax1.semilogy(avg_tab['mag'], N_SNe)
    ax1.set_ylabel('Number of SN Ia')
    py.setp(ax1.get_xticklabels(), visible=False)
    py.axvline(19, color='black', linestyle='--')
    py.title('Tint={0:d} s, R={1:d}, aper={2:0.3f}"'.format(tint, spec_res, aper_radius))

    ax2 = py.subplot(212, sharex=ax1)
    py.semilogy(avg_tab['mag'], avg_tab['snr_sum_y'], label='Y')
    py.plot(avg_tab['mag'], avg_tab['snr_sum_j'], label='J')
    py.plot(avg_tab['mag'], avg_tab['snr_sum_h'], label='H')
    py.xlabel('Magnitude')
    py.ylabel('Signal-to-Noise')
    py.ylim(1e1, 1e3)
    py.xlim(15, 21)
    #py.plot([19.3, 20.4], [30, 30], 'k-', linewidth=3)
    # ax2.arrow(19, 30, -0.5, 0, head_width=5, head_length=0.2, fc='k')
    # ax2.arrow(19, 30, 0, 30, head_width=0.2, head_length=5, fc='k')
    py.axvline(19, color='black', linestyle='--')
    py.legend()
    
    py.savefig(in_file + '_snr_Nsne.png')


def plot_sensitivity_curves_noOH(tint=1200, spec_res=3000, aper_radius=0.15):
    in_file = 'roboAO_sensitivity_t{0:d}_R{1:d}_ap{2:0.3f}'.format(tint, spec_res, aper_radius)

    spec_y_tab = Table.read(in_file + '_spec_y_tab.fits')
    spec_j_tab = Table.read(in_file + '_spec_j_tab.fits')
    spec_h_tab = Table.read(in_file + '_spec_h_tab.fits')
    avg_tab = Table.read(in_file + '_avg_tab.fits')

    # Calculate the band-integrated SNR for each magnitude bin and filter.
    mag = avg_tab['mag']

    for mm in range(len(mag)):
        col_y_suffix = '_Y_{0:d}'.format(mag[mm])
        col_j_suffix = '_J_{0:d}'.format(mag[mm])
        col_h_suffix = '_H_{0:d}'.format(mag[mm])

        snr_spec_y = spec_y_tab['snr'+col_y_suffix]
        snr_spec_j = spec_j_tab['snr'+col_j_suffix]
        snr_spec_h = spec_h_tab['snr'+col_h_suffix]

        bkg_spec_y = spec_y_tab['bkg'+col_y_suffix]
        bkg_spec_j = spec_j_tab['bkg'+col_j_suffix]
        bkg_spec_h = spec_h_tab['bkg'+col_h_suffix]

        fmt = '{0:d}  {1:7.1f} {2:7.1f}   {3:7.1f} {4:7.1f}   {5:7.1f} {6:7.1f}'
        print fmt.format(mag[mm], 
                         avg_tab['snr_y'][mm], avg_tab['snr_sum_y'],
                         avg_tab['snr_j'][mm], avg_tab['snr_sum_j'],
                         avg_tab['snr_h'][mm], avg_tab['snr_sum_h'])
        
    py.clf()
    # py.plot(avg_tab['mag'], avg_tab['snr_j'], label='J')
    # py.plot(avg_tab['mag'], avg_tab['snr_h'], label='H')
    py.plot(avg_tab['mag'], avg_tab['snr_j'], label='J')
    py.plot(avg_tab['mag'], avg_tab['snr_h'], label='H')
    py.xlabel('Magnitude')
    py.ylabel('SNR')
    py.title('Tint={0:d} s, R={1:d}, aper={2:0.3f}"'.format(tint, spec_res, aper_radius))
    py.savefig(in_file + '.png')
    

def plot_seeing_vs_ao(tint=1200, spec_res=100):
    ap_rad_see = 0.4
    ap_rad_rao = 0.150
    
    in_file_see = 'roboAO_sensitivity_t{0:d}_R{1:d}_ap{2:0.3f}_seeing'.format(tint, spec_res, ap_rad_see)
    in_file_rao = 'roboAO_sensitivity_t{0:d}_R{1:d}_ap{2:0.3f}'.format(tint, spec_res, ap_rad_rao)
    print in_file_see
    print in_file_rao

    avg_tab_rao = Table.read(in_file_rao + '_avg_tab.fits')
    avg_tab_see = Table.read(in_file_see + '_avg_tab.fits')
    
    # Calculate the band-integrated SNR for each magnitude bin and filter.
    mag_rao = avg_tab_rao['mag']
    mag_see = avg_tab_see['mag']

    hdr1 = '# {0:3s}  {1:15s}   {2:15s}   {3:15s}'
    hdr2 = '# {0:3s}  {1:7s} {2:7s}   {3:7s} {4:7s}   {5:7s} {6:7s}'
    print hdr1.format('Mag', 'Y SNR (summed)', 'J SNR (summed)', 'H SNR (summed)')
    print hdr2.format('', 'Robo-AO', 'Seeing', 'Robo-AO', 'Seeing', 'Robo-AO', 'Seeing')
    print hdr2.format('---', '-------', '-------', '-------', '-------', '-------', '-------')

    for mm in range(len(mag_rao)):
        fmt = '  {0:3d}  {1:7.1f} {2:7.1f}   {3:7.1f} {4:7.1f}   {5:7.1f} {6:7.1f}'
        print fmt.format(mag_rao[mm], 
                         avg_tab_rao['snr_sum_y'][mm], avg_tab_see['snr_sum_y'][mm],
                         avg_tab_rao['snr_sum_j'][mm], avg_tab_see['snr_sum_j'][mm],
                         avg_tab_rao['snr_sum_h'][mm], avg_tab_see['snr_sum_h'][mm])
        
    N_SNe = 4500.0 * 0.6 * 10**(avg_tab_rao['mag'] - 18.9)
    
    py.clf()
    py.semilogy(avg_tab_rao['mag'], avg_tab_rao['snr_sum_y'], label='UH Robo-AO')
    py.semilogy(avg_tab_rao['mag'], avg_tab_see['snr_sum_y'], label='Seeing-Limited')
    py.xlabel('Y-band Magnitude')
    py.ylabel('Signal-to-Noise (Filter Integrated)')
    py.legend()
    py.title('Tint={0:d} s, R={1:d}'.format(tint, spec_res))
    py.savefig(in_file_rao + '_y_snr_sum_vs_seeing.png')

    py.clf()
    py.plot(avg_tab_rao['mag'], avg_tab_rao['snr_sum_y'] / avg_tab_see['snr_sum_y'])
    py.xlabel('Y-band Magnitude')
    py.ylabel('Gain in Signal-to-Noise')
    py.title('Tint={0:d} s, R={1:d}'.format(tint, spec_res))
    py.savefig(in_file_rao + '_y_snr_gain_vs_seeing.png')
    

    py.clf()
    py.semilogy(avg_tab_rao['mag'], avg_tab_rao['snr_sum_j'], label='UH Robo-AO')
    py.semilogy(avg_tab_rao['mag'], avg_tab_see['snr_sum_j'], label='Seeing-Limited')
    py.xlabel('J-band Magnitude')
    py.ylabel('Signal-to-Noise (Filter Integrated)')
    py.legend()
    py.title('Tint={0:d} s, R={1:d}'.format(tint, spec_res))
    py.savefig(in_file_rao + '_j_snr_sum_vs_seeing.png')

    py.clf()
    py.plot(avg_tab_rao['mag'], avg_tab_rao['snr_sum_j'] / avg_tab_see['snr_sum_j'])
    py.xlabel('J-band Magnitude')
    py.ylabel('Gain in Signal-to-Noise')
    py.title('Tint={0:d} s, R={1:d}'.format(tint, spec_res))
    py.savefig(in_file_rao + '_j_snr_gain_vs_seeing.png')
    

    py.clf()

    py.semilogy(avg_tab_rao['mag'], avg_tab_rao['snr_sum_h'], label='UH Robo-AO')
    py.semilogy(avg_tab_rao['mag'], avg_tab_see['snr_sum_h'], label='Seeing-Limited')
    py.xlabel('H-band Magnitude')
    py.ylabel('Signal-to-Noise (Filter Integrated)')
    py.legend()
    py.title('Tint={0:d} s, R={1:d}'.format(tint, spec_res))
    py.savefig(in_file_rao + '_h_snr_sum_vs_seeing.png')

    py.clf()
    py.plot(avg_tab_rao['mag'], avg_tab_rao['snr_sum_h'] / avg_tab_see['snr_sum_h'])
    py.xlabel('H-band Magnitude')
    py.ylabel('Gain in Signal-to-Noise')
    py.title('Tint={0:d} s, R={1:d}'.format(tint, spec_res))
    py.savefig(in_file_rao + '_h_snr_gain_vs_seeing.png')
    
    

def make_tint_curves(mag=20, spec_res=100, aper_radius=0.15, seeing_limited=False):
    tint = np.arange(300, 3600+1, 300)
    snr_y = np.zeros(len(tint), dtype=float)
    snr_j = np.zeros(len(tint), dtype=float)
    snr_h = np.zeros(len(tint), dtype=float)
    
    snr_sum_y = np.zeros(len(tint), dtype=float)
    snr_sum_j = np.zeros(len(tint), dtype=float)
    snr_sum_h = np.zeros(len(tint), dtype=float)
    
    spec_y_tab = None
    spec_j_tab = None
    spec_h_tab = None

    for tt in range(len(tint)):
        print 'Tint: ', tint[tt]
        blah_y = etc_uh_roboAO(mag, 'Y', tint[tt],
                               spec_res=spec_res, aper_radius=aper_radius, 
                               seeing_limited=seeing_limited)
        blah_j = etc_uh_roboAO(mag, 'J', tint[tt],
                               spec_res=spec_res, aper_radius=aper_radius,
                               seeing_limited=seeing_limited)
        blah_h = etc_uh_roboAO(mag, 'H', tint[tt],
                               spec_res=spec_res, aper_radius=aper_radius,
                               seeing_limited=seeing_limited)
        
        col_y_suffix = '_Y_{0:d}'.format(tint[tt])
        col_j_suffix = '_J_{0:d}'.format(tint[tt])
        col_h_suffix = '_H_{0:d}'.format(tint[tt])

        spec_signal_y = Column(name='sig'+col_y_suffix, data=blah_y[4])
        spec_signal_j = Column(name='sig'+col_j_suffix, data=blah_j[4])
        spec_signal_h = Column(name='sig'+col_h_suffix, data=blah_h[4])
        spec_bkg_y = Column(name='bkg'+col_y_suffix, data=blah_y[5])
        spec_bkg_j = Column(name='bkg'+col_j_suffix, data=blah_j[5])
        spec_bkg_h = Column(name='bkg'+col_h_suffix, data=blah_h[5])
        spec_snr_y = Column(name='snr'+col_y_suffix, data=blah_y[6])
        spec_snr_j = Column(name='snr'+col_j_suffix, data=blah_j[6])
        spec_snr_h = Column(name='snr'+col_h_suffix, data=blah_h[6])
        
        if spec_y_tab == None:
            spec_y_tab = Table([blah_y[3]], names=['wave_Y'])
        if spec_j_tab == None:
            spec_j_tab = Table([blah_j[3]], names=['wave_J'])
        if spec_h_tab == None:
            spec_h_tab = Table([blah_h[3]], names=['wave_H'])

        spec_y_tab.add_columns([spec_signal_y, spec_bkg_y, spec_snr_y])
        spec_j_tab.add_columns([spec_signal_j, spec_bkg_j, spec_snr_j])
        spec_h_tab.add_columns([spec_signal_h, spec_bkg_h, spec_snr_h])

        snr_y[tt]  = blah_y[0]
        snr_j[tt]  = blah_j[0]
        snr_h[tt]  = blah_h[0]

        snr_sum_y[tt] = math.sqrt((spec_snr_y**2).sum())
        snr_sum_j[tt] = math.sqrt((spec_snr_j**2).sum())
        snr_sum_h[tt] = math.sqrt((spec_snr_h**2).sum())
        

    avg_tab = Table([tint, 
                     snr_y, snr_sum_y, 
                     snr_j, snr_sum_j,
                     snr_h, snr_sum_h],
                    names=['tint', 
                           'snr_y', 'snr_sum_y', 
                           'snr_j', 'snr_sum_j',
                           'snr_h', 'snr_sum_h'])


    out_file = 'roboAO_tint_m{0:d}_R{1:d}_ap{2:0.3f}'.format(mag, spec_res, aper_radius)

    if seeing_limited:
        out_file += '_seeing'
    
    # Save the tables
    spec_y_tab.write(out_file + '_spec_y_tab.fits', overwrite=True)
    spec_j_tab.write(out_file + '_spec_j_tab.fits', overwrite=True)
    spec_h_tab.write(out_file + '_spec_h_tab.fits', overwrite=True)
    avg_tab.write(out_file + '_avg_tab.fits', overwrite=True)

    return
    
def plot_tint_curves(mag=20, spec_res=100, aper_radius=0.15, seeing_limited=False):
    in_file = 'roboAO_tint_m{0:d}_R{1:d}_ap{2:0.3f}'.format(mag, spec_res, aper_radius)

    if seeing_limited:
        in_file += '_seeing'

    avg_tab = Table.read(in_file + '_avg_tab.fits')

    # Calculate the band-integrated SNR for each magnitude bin and filter.
    tint = avg_tab['tint']

    hdr1 = '# {0:3s}  {1:15s}   {2:15s}   {3:15s}'
    hdr2 = '# {0:3s}  {1:7s} {2:7s}   {3:7s} {4:7s}   {5:7s} {6:7s}'
    print hdr1.format('Tint', '  Y-band SNR', '  J-band SNR', '  H-band SNR')
    print hdr2.format('', 'Per_Ch', 'Summed', 'Per_Ch', 'Summed', 'Per_Ch', 'Summed')
    print hdr2.format('---', '-------', '-------', '-------', '-------', '-------', '-------')

    for tt in range(len(tint)):
        fmt = '  {0:3d}  {1:7.1f} {2:7.1f}   {3:7.1f} {4:7.1f}   {5:7.1f} {6:7.1f}'
        print fmt.format(tint[tt], 
                         avg_tab['snr_y'][tt], avg_tab['snr_sum_y'][tt],
                         avg_tab['snr_j'][tt], avg_tab['snr_sum_j'][tt],
                         avg_tab['snr_h'][tt], avg_tab['snr_sum_h'][tt])
        
    py.clf()
    py.plot(avg_tab['tint'], avg_tab['snr_y'], label='Y')
    py.plot(avg_tab['tint'], avg_tab['snr_j'], label='J')
    py.plot(avg_tab['tint'], avg_tab['snr_h'], label='H')
    py.xlabel('Integration Time (s)')
    py.ylabel('SNR Per Spectral Channel')
    py.title('Mag={0:d}, R={1:d}, aper={2:0.3f}"'.format(mag, spec_res, aper_radius))
    py.legend()
    py.ylim(0, 80)
    py.savefig(in_file + '_snr_per.png')
    
    py.clf()
    py.plot(avg_tab['tint'], avg_tab['snr_sum_y'], label='Y')
    py.plot(avg_tab['tint'], avg_tab['snr_sum_j'], label='J')
    py.plot(avg_tab['tint'], avg_tab['snr_sum_h'], label='H')
    py.xlabel('Integration Time (s)')
    py.ylabel('SNR Over Filter')
    py.ylim(0, 80)
    py.legend()
    py.title('Mag={0:d}, R={1:d}, aper={2:0.3f}"'.format(mag, spec_res, aper_radius))
    py.savefig(in_file + '_snr_sum.png')


def plot_seeing_vs_ao_tint(mag=20, spec_res=100):
    ap_rad_see = 0.4
    ap_rad_rao = 0.15
    
    in_file_see = 'roboAO_tint_m{0:d}_R{1:d}_ap{2:0.3f}_seeing'.format(mag, spec_res, ap_rad_see)
    in_file_rao = 'roboAO_tint_m{0:d}_R{1:d}_ap{2:0.3f}'.format(mag, spec_res, ap_rad_rao)

    avg_tab_rao = Table.read(in_file_rao + '_avg_tab.fits')
    avg_tab_see = Table.read(in_file_see + '_avg_tab.fits')
    
    # Calculate the band-integrated SNR for each magnitude bin and filter.
    tint = avg_tab_rao['tint']
    
    hdr1 = '# {0:4s}  {1:15s}   {2:15s}   {3:15s}'
    hdr2 = '# {0:4s}  {1:7s} {2:7s}   {3:7s} {4:7s}   {5:7s} {6:7s}'
    print hdr1.format('Tint', 'Y SNR (summed)', 'J SNR (summed)', 'H SNR (summed)')
    print hdr2.format('', 'Robo-AO', 'Seeing', 'Robo-AO', 'Seeing', 'Robo-AO', 'Seeing')
    print hdr2.format('---', '-------', '-------', '-------', '-------', '-------', '-------')

    for tt in range(len(tint)):
        fmt = '  {0:4d}  {1:7.1f} {2:7.1f}   {3:7.1f} {4:7.1f}   {5:7.1f} {6:7.1f}'
        print fmt.format(tint[tt], 
                         avg_tab_rao['snr_sum_y'][tt], avg_tab_see['snr_sum_y'][tt],
                         avg_tab_rao['snr_sum_j'][tt], avg_tab_see['snr_sum_j'][tt],
                         avg_tab_rao['snr_sum_h'][tt], avg_tab_see['snr_sum_h'][tt])

    py.clf()
    py.semilogy(avg_tab_rao['tint'], avg_tab_rao['snr_sum_y'], label='UH Robo-AO')
    py.semilogy(avg_tab_rao['tint'], avg_tab_see['snr_sum_y'], label='Seeing-Limited')
    py.xlabel('Integration Time (s)')
    py.ylabel('Y-band Signal-to-Noise')
    py.legend()
    py.title('Mag={0:d}, R={1:d}'.format(mag, spec_res))
    py.savefig(in_file_rao + '_y_snr_sum_vs_seeing_tint.png')

    py.clf()
    py.plot(avg_tab_rao['tint'], avg_tab_rao['snr_sum_y'] / avg_tab_see['snr_sum_y'])
    py.xlabel('Integration Time (s)')
    py.ylabel('Y-band Gain in Signal-to-Noise')
    py.title('Mag={0:d}, R={1:d}'.format(mag, spec_res))
    py.savefig(in_file_rao + '_y_snr_gain_vs_seeing_tint.png')
    

    py.clf()
    py.semilogy(avg_tab_rao['tint'], avg_tab_rao['snr_sum_j'], label='UH Robo-AO')
    py.semilogy(avg_tab_rao['tint'], avg_tab_see['snr_sum_j'], label='Seeing-Limited')
    py.xlabel('Integration Time (s)')
    py.ylabel('J-band Signal-to-Noise')
    py.legend()
    py.title('Mag={0:d}, R={1:d}'.format(mag, spec_res))
    py.savefig(in_file_rao + '_j_snr_sum_vs_seeing_tint.png')

    py.clf()
    py.plot(avg_tab_rao['tint'], avg_tab_rao['snr_sum_j'] / avg_tab_see['snr_sum_j'])
    py.xlabel('Integration Time (s)')
    py.ylabel('J-band Gain in Signal-to-Noise')
    py.title('Mag={0:d}, R={1:d}'.format(mag, spec_res))
    py.savefig(in_file_rao + '_j_snr_gain_vs_seeing_tint.png')
    

    py.clf()
    py.semilogy(avg_tab_rao['tint'], avg_tab_rao['snr_sum_h'], label='UH Robo-AO')
    py.semilogy(avg_tab_rao['tint'], avg_tab_see['snr_sum_h'], label='Seeing-Limited')
    py.xlabel('Integration Time (s)')
    py.ylabel('H-band Signal-to-Noise')
    py.legend()
    py.title('Mag={0:d}, R={1:d}'.format(mag, spec_res))
    py.savefig(in_file_rao + '_h_snr_sum_vs_seeing_tint.png')

    py.clf()
    py.plot(avg_tab_rao['tint'], avg_tab_rao['snr_sum_h'] / avg_tab_see['snr_sum_h'])
    py.xlabel('Integration Time (s)')
    py.ylabel('H-band Gain in Signal-to-Noise')
    py.title('Mag={0:d}, R={1:d}'.format(mag, spec_res))
    py.savefig(in_file_rao + '_h_snr_gain_vs_seeing_tint.png')
    
    

    
