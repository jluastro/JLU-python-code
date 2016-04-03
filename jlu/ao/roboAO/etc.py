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
from astropy.modeling.functional_models import Moffat2D


# Convert from m_AB to m_Vega. Values reported below are
# m_AB - m_Vega and were taken from:
# http://www.astronomy.ohio-state.edu/~martini/usefuldata.html
# Which mostly come from Blanton et al. 2007
Vega_to_AB = {'B': 0.790, 'V': 0.02, 'R': 0.21, 'I': 0.45, 'z': 0.54,
              'Y': 0.634, 'J': 0.91, 'H': 1.39, 'K': 1.85}

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

def etc_uh_roboAO(mag, filt_name, tint, sq_aper_diam=0.3, phot_sys='Vega', spec_res=100, 
                  seeing_limited=False):
    """
    Exposure time calculator for a UH Robo-AO system and a NIR IFU spectrograph.

    phot_sys - 'Vega' (default) or 'AB'
    """
    
    ifu_throughput = 0.35
    #ao_throughput = 0.55
    ao_throughput = 0.76  # New Design
    tel_throughput = 0.85**2

    if seeing_limited:
        sys_throughput = ifu_throughput * tel_throughput
    else:
        sys_throughput = ifu_throughput * ao_throughput * tel_throughput
    
    # TO DO Need to add telescope secondary obscuration to correct the area.
    tel_area = math.pi * (2.22 * 100. / 2.)**2   # cm^2 for UH 2.2m tel
    sec_area = math.pi * (0.613 * 100. / 2.)**2   # cm^2 for UH 2.2m tel hole/secondary obscuration
    tel_area -= sec_area
    
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

    # Estimate the number of pixels across our spectrum.
    npix_spec = len(ifu_wave)
    
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
    bkg_obs = observation.Observation(earth_bkg, filt, binset=ifu_wave, force="extrap")
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
    vega_counts *= dlamda                     # photon s^-1 NO? arcsec^-2

    # Integrate over the aperture for the background and make
    # an aperture correction for the star.
    if seeing_limited:
        ee = get_seeing_ee(filt_name, sq_aper_diam)
    else:
        ee = get_roboAO_ee(mag, filt_name, sq_aper_diam)
        
    aper_area = sq_aper_diam**2               # square
    star_counts *= ee                         # photon s^-1
    # TODO... Don't I need to do this for vega as well?
    # vega_counts *= ee  
    bkg_counts *= aper_area                   # photon s^-1 arcsec^-2

    pix_scale = 0.150                      # arcsec per pixel
    if seeing_limited:
        pix_scale = 0.400
        
    npix = (aper_area / pix_scale**2)
    npix *= npix_spec

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

    msg = 'filt = {0:s}  signal = {1:13.1f}  bkg = {2:9.1f}  SNR = {3:7.1f}'
    print msg.format(filt_name, avg_signal, bkg.mean(), avg_snr)

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

def get_seeing_ee(filt_name, sq_aper_diam, overwrite=False):
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
    sq_diam = pickle.load(_psf)
    sq_ee = pickle.load(_psf)
    _psf.close()
            
    # Integrate out to the desired square diameter
    dd_idx = np.abs(sq_diam - sq_aper_diam).argmin()

    ee_good = sq_ee[dd_idx]

    return ee_good
    
def make_seeing_psf(filt_name):
    psf_file = 'seeing_psf_{0:s}.pickle'.format(filt_name)

    # Note that seeing is taken as FWHM.
    seeing = 0.86 # arcsec specified at 5000 Angstrom
    
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
    # print 'Make Gaussian sigma=', seeing/2.35
    # psf = Gaussian2D.evaluate(x, y, 1, 0, 0, seeing/2.35, seeing/2.35, 0)

    # Use a 2D Moffatt as our PSF.
    alpha = 2.5
    gamma = seeing / (2.0 * (2**(1.0/alpha) - 1)**0.5)
    print 'Make Moffat 2D alpha=2.5, gamma=', gamma
    psf = Moffat2D.evaluate(x, y, 1, 0, 0, gamma, alpha)
    
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

    # Make an ensquared energy curve.
    sq_diam_save = np.arange(0.025, 2, 0.025)
    sq_ee_save = np.zeros(len(sq_diam_save), dtype=float)

    x_dist = np.abs(x)
    y_dist = np.abs(y)
    
    for ii in range(len(sq_diam_save)):
        print 'Integrating over square diameter: ', sq_diam_save[ii]
        dist = sq_diam_save[ii] / 2.0
        idx = np.where((x_dist < dist) & (y_dist < dist))

        sq_ee_save[ii] = psf[idx].sum()
        
        
    print 'Save'
    _psf = open(psf_file, 'w')
    pickle.dump(r_save, _psf)
    pickle.dump(ee_save, _psf)
    pickle.dump(sq_diam_save, _psf)
    pickle.dump(sq_ee_save, _psf)
    _psf.close()

    return

def get_roboAO_ee(mag, filt_name, sq_aper_diam,
                  data_dir='/Users/jlu/work/ao/roboAO/2016_01_18/UHRoboAO_EE_with_cumulative/'):
    """
    Get the encircled energy.

    Note we assume zenith angle = 30 degrees.
    """
    #sq_aper_diam_valid = [0.0375, 0.075, 0.15]
    sq_aper_diam_valid = [0.15, 0.30, 0.40, 0.70]
    
    if sq_aper_diam not in sq_aper_diam_valid:
        print 'Invalid square-aperture diameter. Choose from:'
        print sq_aper_diam_valid

    # Hard-coded magnitudes of the files.
    guide_mag = np.array([10, 17, 18, 19, 20], dtype=int)
    ee_mag = np.zeros(len(guide_mag), dtype=float)

    for gg in range(len(guide_mag)):
        filename = 'ee_cdf_V{0:2d}_z30.txt'.format(guide_mag[gg])

        t = Table.read(data_dir + filename, format='ascii')

        col_name = '{0:s}_EE{1:03.0f}'.format(filt_name, sq_aper_diam*1e3)

        cprob = t['cprob']
        ee = t[col_name]

        idx = np.where(cprob == 0.5)[0]
        ee_mag[gg] = ee[idx[0]]
        
    ee = np.interp(mag, guide_mag, ee_mag)

    return ee
    
def make_sensitivity_curves(tint=1200, spec_res=100, sq_aper_diam=0.30, seeing_limited=False):
    mag = np.arange(10, 22)
    snr_z = np.zeros(len(mag), dtype=float)
    snr_y = np.zeros(len(mag), dtype=float)
    snr_j = np.zeros(len(mag), dtype=float)
    snr_h = np.zeros(len(mag), dtype=float)
    snr_sum_z = np.zeros(len(mag), dtype=float)
    snr_sum_y = np.zeros(len(mag), dtype=float)
    snr_sum_j = np.zeros(len(mag), dtype=float)
    snr_sum_h = np.zeros(len(mag), dtype=float)
    bkg_z = np.zeros(len(mag), dtype=float)
    bkg_y = np.zeros(len(mag), dtype=float)
    bkg_j = np.zeros(len(mag), dtype=float)
    bkg_h = np.zeros(len(mag), dtype=float)
    star_z = np.zeros(len(mag), dtype=float)
    star_y = np.zeros(len(mag), dtype=float)
    star_j = np.zeros(len(mag), dtype=float)
    star_h = np.zeros(len(mag), dtype=float)

    spec_z_tab = None
    spec_y_tab = None
    spec_j_tab = None
    spec_h_tab = None

    # Calculate the number of supernovae.
    N_SNe = 4500.0 * 0.6 * 10**(mag - 18.9)

    out_file = 'roboAO_sensitivity_t{0:d}_R{1:d}_ap{2:0.3f}'.format(tint, spec_res,
                                                                    sq_aper_diam)
    
    # Save the output to a table.
    _out = open(out_file + '.txt', 'w')

    meta1 = '# tint = {0:5d}, R = {1:5d}, sq_ap_diam = {2:5.3f}"\n'
    _out.write(meta1.format(tint, spec_res, sq_aper_diam))
    _out.write('# Sensitivity integrated over broad band.')
    
    hdr = '{0:5s}  {1:6s}  {2:5s}  {3:5s}  {4:5s}  {5:5s}  {6:5s}  {7:5s}\n'
    fmt = '{0:5.1f}  {1:6.1f}  {2:5.1f}  {3:5.1f}  {4:5.1f}  {5:5.1f}  {6:5.1f}  {7:5.1f}\n'
    _out.write(hdr.format('# Mag', 'N_SNe', 'J_SNR', 'H_SNR', 'J_ms', 'H_ms', 'J_mb', 'H_mb'))
               
    for mm in range(len(mag)):
        print 'Mag: ', mag[mm]
        blah_z = etc_uh_roboAO(mag[mm], 'Z', tint,
                               spec_res=spec_res, sq_aper_diam=sq_aper_diam, 
                               seeing_limited=seeing_limited)
        blah_y = etc_uh_roboAO(mag[mm], 'Y', tint,
                               spec_res=spec_res, sq_aper_diam=sq_aper_diam, 
                               seeing_limited=seeing_limited)
        blah_j = etc_uh_roboAO(mag[mm], 'J', tint,
                               spec_res=spec_res, sq_aper_diam=sq_aper_diam,
                               seeing_limited=seeing_limited)
        blah_h = etc_uh_roboAO(mag[mm], 'H', tint,
                               spec_res=spec_res, sq_aper_diam=sq_aper_diam,
                               seeing_limited=seeing_limited)
        
        col_z_suffix = '_Z_{0:d}'.format(mag[mm])
        col_y_suffix = '_Y_{0:d}'.format(mag[mm])
        col_j_suffix = '_J_{0:d}'.format(mag[mm])
        col_h_suffix = '_H_{0:d}'.format(mag[mm])

        spec_signal_z = Column(name='sig'+col_z_suffix, data=blah_z[4])
        spec_signal_y = Column(name='sig'+col_y_suffix, data=blah_y[4])
        spec_signal_j = Column(name='sig'+col_j_suffix, data=blah_j[4])
        spec_signal_h = Column(name='sig'+col_h_suffix, data=blah_h[4])
        spec_bkg_z = Column(name='bkg'+col_z_suffix, data=blah_z[5])
        spec_bkg_y = Column(name='bkg'+col_y_suffix, data=blah_y[5])
        spec_bkg_j = Column(name='bkg'+col_j_suffix, data=blah_j[5])
        spec_bkg_h = Column(name='bkg'+col_h_suffix, data=blah_h[5])
        spec_snr_z = Column(name='snr'+col_z_suffix, data=blah_z[6])
        spec_snr_y = Column(name='snr'+col_y_suffix, data=blah_y[6])
        spec_snr_j = Column(name='snr'+col_j_suffix, data=blah_j[6])
        spec_snr_h = Column(name='snr'+col_h_suffix, data=blah_h[6])

        
        if spec_z_tab == None:
            spec_z_tab = Table([blah_z[3]], names=['wave_Z'])
        if spec_y_tab == None:
            spec_y_tab = Table([blah_y[3]], names=['wave_Y'])
        if spec_j_tab == None:
            spec_j_tab = Table([blah_j[3]], names=['wave_J'])
        if spec_h_tab == None:
            spec_h_tab = Table([blah_h[3]], names=['wave_H'])

        spec_z_tab.add_columns([spec_signal_z, spec_bkg_z, spec_snr_z])
        spec_y_tab.add_columns([spec_signal_y, spec_bkg_y, spec_snr_y])
        spec_j_tab.add_columns([spec_signal_j, spec_bkg_j, spec_snr_j])
        spec_h_tab.add_columns([spec_signal_h, spec_bkg_h, spec_snr_h])

        snr_z[mm]  = blah_z[0]
        snr_y[mm]  = blah_y[0]
        snr_j[mm]  = blah_j[0]
        snr_h[mm]  = blah_h[0]
        snr_sum_z[mm] = math.sqrt((spec_snr_z**2).sum())
        snr_sum_y[mm] = math.sqrt((spec_snr_y**2).sum())
        snr_sum_j[mm] = math.sqrt((spec_snr_j**2).sum())
        snr_sum_h[mm] = math.sqrt((spec_snr_h**2).sum())

        star_z[mm]  = blah_z[1]
        star_y[mm]  = blah_y[1]
        star_j[mm]  = blah_j[1]
        star_h[mm]  = blah_h[1]
        bkg_z[mm]  = blah_z[2]
        bkg_y[mm]  = blah_y[2]
        bkg_j[mm]  = blah_j[2]
        bkg_h[mm]  = blah_h[2]

    avg_tab = Table([mag, snr_z, snr_y, snr_j, snr_h, 
                     snr_sum_z, snr_sum_y, snr_sum_j, snr_sum_h,
                     star_z, star_y, star_j, star_h, bkg_z, bkg_y, bkg_j, bkg_h],
                    names=['mag', 'snr_z', 'snr_y', 'snr_j', 'snr_h', 
                           'snr_sum_z', 'snr_sum_y', 'snr_sum_j', 'snr_sum_h',
                           'star_z', 'star_y', 'star_j', 'star_h', 
                           'bkg_z', 'bkg_y', 'bkg_j', 'bkg_h'])


    out_file = 'roboAO_sensitivity_t{0:d}_R{1:d}_ap{2:0.3f}'.format(tint, spec_res, sq_aper_diam)

    if seeing_limited:
        out_file += '_seeing'
    
    # Save the tables
    spec_z_tab.write(out_file + '_spec_z_tab.fits', overwrite=True)
    spec_y_tab.write(out_file + '_spec_y_tab.fits', overwrite=True)
    spec_j_tab.write(out_file + '_spec_j_tab.fits', overwrite=True)
    spec_h_tab.write(out_file + '_spec_h_tab.fits', overwrite=True)
    avg_tab.write(out_file + '_avg_tab.fits', overwrite=True)

    return


def plot_sensitivity_curves(tint=1200, spec_res=100, sq_aper_diam=0.30, seeing_limited=False):
    in_file = 'roboAO_sensitivity_t{0:d}_R{1:d}_ap{2:0.3f}'.format(tint, spec_res, sq_aper_diam)

    if seeing_limited:
        in_file += '_seeing'

    avg_tab = Table.read(in_file + '_avg_tab.fits')

    # Calculate the band-integrated SNR for each magnitude bin and filter.
    mag = avg_tab['mag']
    N_SNe = 4500.0 * 10**(0.6*(mag - 18.9))

    hdr1 = '# {0:3s}  {1:15s}   {2:15s}   {3:15s}   {4:15s}  {5:8s}'
    hdr2 = '# {0:3s}  {1:7s} {2:7s}   {3:7s} {4:7s}   {5:7s} {6:7s}   {7:7s} {8:7s}  {9:8s}'
    print hdr1.format('Mag', '  Z-band SNR', '  Y-band SNR', '  J-band SNR', '  H-band SNR', 'Number')
    print hdr2.format('', 'Per_Ch', 'Summed', 'Per_Ch', 'Summed', 'Per_Ch', 'Summed', 'Per_Ch', 'Summed', 'of SN Ia')
    print hdr2.format('---', '-------', '-------','-------', '-------',  '-------', '-------', '-------', '-------', '--------')

    for mm in range(len(mag)):
        fmt = '  {0:3d}  {1:7.1f} {2:7.1f}   {3:7.1f} {4:7.1f}   {5:7.1f} {6:7.1f}  {7:7.1f} {8:7.1f}   {9:8.0f}'
        print fmt.format(mag[mm], 
                         avg_tab['snr_z'][mm], avg_tab['snr_sum_z'][mm],
                         avg_tab['snr_y'][mm], avg_tab['snr_sum_y'][mm],
                         avg_tab['snr_j'][mm], avg_tab['snr_sum_j'][mm],
                         avg_tab['snr_h'][mm], avg_tab['snr_sum_h'][mm],
                         N_SNe[mm])

    py.figure(1)
    py.clf()
    py.semilogy(avg_tab['mag'], avg_tab['snr_z'], label='Z')
    py.plot(avg_tab['mag'], avg_tab['snr_y'], label='Y')
    py.plot(avg_tab['mag'], avg_tab['snr_j'], label='J')
    py.plot(avg_tab['mag'], avg_tab['snr_h'], label='H')
    py.xlabel('Magnitude')
    py.ylabel('SNR Per Spectral Channel')
    py.title('Tint={0:d} s, R={1:d}, aper_diam={2:0.3f}"'.format(tint, spec_res, sq_aper_diam))
    py.legend()
    py.ylim(1, 1e4)
    py.savefig(in_file + '_snr_per.png')
    
    py.clf()
    py.semilogy(avg_tab['mag'], avg_tab['snr_sum_z'], label='Z')
    py.plot(avg_tab['mag'], avg_tab['snr_sum_y'], label='Y')
    py.plot(avg_tab['mag'], avg_tab['snr_sum_j'], label='J')
    py.plot(avg_tab['mag'], avg_tab['snr_sum_h'], label='H')
    py.xlabel('Magnitude')
    py.ylabel('SNR Over Filter')
    py.ylim(1, 1e4)
    py.legend()
    py.title('Tint={0:d} s, R={1:d}, aper_diam={2:0.3f}"'.format(tint, spec_res, sq_aper_diam))
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
    py.title('Tint={0:d} s, R={1:d}, aper_diam={2:0.3f}"'.format(tint, spec_res, sq_aper_diam))

    ax2 = py.subplot(212, sharex=ax1)
    py.semilogy(avg_tab['mag'], avg_tab['snr_sum_z'], label='Z')
    py.plot(avg_tab['mag'], avg_tab['snr_sum_y'], label='Y')
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


def plot_sensitivity_curves_noOH(tint=1200, spec_res=3000, sq_aper_diam=0.30):
    in_file = 'roboAO_sensitivity_t{0:d}_R{1:d}_ap{2:0.3f}'.format(tint, spec_res, sq_aper_diam)

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
    py.title('Tint={0:d} s, R={1:d}, aper_diam={2:0.3f}"'.format(tint, spec_res, sq_aper_diam))
    py.savefig(in_file + '.png')
    

def plot_seeing_vs_ao(tint=1200, spec_res=100):
    ap_diam_see = 0.8
    ap_diam_rao = 0.30
    
    in_file_see = 'roboAO_sensitivity_t{0:d}_R{1:d}_ap{2:0.3f}_seeing'.format(tint, spec_res, ap_diam_see)
    in_file_rao = 'roboAO_sensitivity_t{0:d}_R{1:d}_ap{2:0.3f}'.format(tint, spec_res, ap_diam_rao)
    print in_file_see
    print in_file_rao

    avg_tab_rao = Table.read(in_file_rao + '_avg_tab.fits')
    avg_tab_see = Table.read(in_file_see + '_avg_tab.fits')
    
    # Calculate the band-integrated SNR for each magnitude bin and filter.
    mag_rao = avg_tab_rao['mag']
    mag_see = avg_tab_see['mag']

    hdr1 = '# {0:3s}  {1:15s}   {2:15s}   {3:15s}   {4:15s}'
    hdr2 = '# {0:3s}  {1:7s} {2:7s}   {3:7s} {4:7s}   {5:7s} {6:7s}   {7:7s} {8:7s}'
    print hdr1.format('Mag', 'Y SNR (summed)', 'X SNR (summed)', 'J SNR (summed)', 'H SNR (summed)')
    print hdr2.format('', 'Robo-AO', 'Seeing', 'Robo-AO', 'Seeing', 'Robo-AO', 'Seeing', 'Robo-AO', 'Seeing')
    print hdr2.format('---', '-------', '-------', '-------', '-------', '-------', '-------', '-------', '-------')

    for mm in range(len(mag_rao)):
        fmt = '  {0:3d}  {1:7.1f} {2:7.1f}   {3:7.1f} {4:7.1f}   {5:7.1f} {6:7.1f}   {7:7.1f} {8:7.1f}'
        print fmt.format(mag_rao[mm], 
                         avg_tab_rao['snr_sum_z'][mm], avg_tab_see['snr_sum_z'][mm],
                         avg_tab_rao['snr_sum_y'][mm], avg_tab_see['snr_sum_y'][mm],
                         avg_tab_rao['snr_sum_j'][mm], avg_tab_see['snr_sum_j'][mm],
                         avg_tab_rao['snr_sum_h'][mm], avg_tab_see['snr_sum_h'][mm])
        
    N_SNe = 4500.0 * 0.6 * 10**(avg_tab_rao['mag'] - 18.9)
    
    py.clf()
    py.semilogy(avg_tab_rao['mag'], avg_tab_rao['snr_sum_z'], label='UH Robo-AO')
    py.semilogy(avg_tab_rao['mag'], avg_tab_see['snr_sum_z'], label='Seeing-Limited')
    py.xlabel('Z-band Magnitude')
    py.ylabel('Signal-to-Noise (Filter Integrated)')
    py.legend()
    py.title('Tint={0:d} s, R={1:d}'.format(tint, spec_res))
    py.savefig(in_file_rao + '_z_snr_sum_vs_seeing.png')

    py.clf()
    py.plot(avg_tab_rao['mag'], avg_tab_rao['snr_sum_z'] / avg_tab_see['snr_sum_z'])
    py.xlabel('Z-band Magnitude')
    py.ylabel('Gain in Signal-to-Noise')
    py.title('Tint={0:d} s, R={1:d}'.format(tint, spec_res))
    py.savefig(in_file_rao + '_z_snr_gain_vs_seeing.png')



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


    # Combine across filters.
    py.clf()
    py.semilogy(avg_tab_rao['mag'], avg_tab_rao['snr_sum_z'], 'c-', label='Z UH Robo-AO')
    py.semilogy(avg_tab_rao['mag'], avg_tab_see['snr_sum_z'], 'c--', label='Z Seeing-Limited')
    py.semilogy(avg_tab_rao['mag'], avg_tab_rao['snr_sum_y'], 'b-', label='Y UH Robo-AO')
    py.semilogy(avg_tab_rao['mag'], avg_tab_see['snr_sum_y'], 'b--', label='Y Seeing-Limited')
    py.semilogy(avg_tab_rao['mag'], avg_tab_rao['snr_sum_j'], 'g-', label='J UH Robo-AO')
    py.semilogy(avg_tab_rao['mag'], avg_tab_see['snr_sum_j'], 'g--', label='J Seeing-Limited')
    py.semilogy(avg_tab_rao['mag'], avg_tab_rao['snr_sum_h'], 'r-', label='H UH Robo-AO')
    py.semilogy(avg_tab_rao['mag'], avg_tab_see['snr_sum_h'], 'r--', label='H Seeing-Limited')
    py.xlabel('Magnitude')
    py.ylabel('Signal-to-Noise (Filter Integrated)')
    py.legend()
    py.title('Tint={0:d} s, R={1:d}'.format(tint, spec_res))
    py.savefig(in_file_rao + '_all_snr_sum_vs_seeing.png')
    
    py.clf()
    py.plot(avg_tab_rao['mag'], avg_tab_rao['snr_sum_z'] / avg_tab_see['snr_sum_z'], label='Z')
    py.plot(avg_tab_rao['mag'], avg_tab_rao['snr_sum_y'] / avg_tab_see['snr_sum_y'], label='Y')
    py.plot(avg_tab_rao['mag'], avg_tab_rao['snr_sum_j'] / avg_tab_see['snr_sum_j'], label='J')
    py.plot(avg_tab_rao['mag'], avg_tab_rao['snr_sum_h'] / avg_tab_see['snr_sum_h'], label='H')
    py.legend(loc='upper left')
    py.xlabel('Magnitude')
    py.ylabel('Gain in Signal-to-Noise')
    py.title('Tint={0:d} s, R={1:d}'.format(tint, spec_res))
    py.savefig(in_file_rao + '_all_snr_gain_vs_seeing.png')
        
    

def make_tint_curves(mag=20, spec_res=100, sq_aper_diam=0.30, seeing_limited=False):
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
                               spec_res=spec_res, sq_aper_diam=sq_aper_diam, 
                               seeing_limited=seeing_limited)
        blah_j = etc_uh_roboAO(mag, 'J', tint[tt],
                               spec_res=spec_res, sq_aper_diam=sq_aper_diam,
                               seeing_limited=seeing_limited)
        blah_h = etc_uh_roboAO(mag, 'H', tint[tt],
                               spec_res=spec_res, sq_aper_diam=sq_aper_diam,
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


    out_file = 'roboAO_tint_m{0:d}_R{1:d}_ap{2:0.3f}'.format(mag, spec_res, sq_aper_diam)

    if seeing_limited:
        out_file += '_seeing'
    
    # Save the tables
    spec_y_tab.write(out_file + '_spec_y_tab.fits', overwrite=True)
    spec_j_tab.write(out_file + '_spec_j_tab.fits', overwrite=True)
    spec_h_tab.write(out_file + '_spec_h_tab.fits', overwrite=True)
    avg_tab.write(out_file + '_avg_tab.fits', overwrite=True)

    return
    
def plot_tint_curves(mag=20, spec_res=100, sq_aper_diam=0.30, seeing_limited=False):
    in_file = 'roboAO_tint_m{0:d}_R{1:d}_ap{2:0.3f}'.format(mag, spec_res, sq_aper_diam)

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
    py.title('Mag={0:d}, R={1:d}, aper_diam={2:0.3f}"'.format(mag, spec_res, sq_aper_diam))
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
    py.title('Mag={0:d}, R={1:d}, aper_diam={2:0.3f}"'.format(mag, spec_res, sq_aper_diam))
    py.savefig(in_file + '_snr_sum.png')


def plot_seeing_vs_ao_tint(mag=20, spec_res=100):
    ap_diam_see = 0.8
    ap_diam_rao = 0.3
    
    in_file_see = 'roboAO_tint_m{0:d}_R{1:d}_ap{2:0.3f}_seeing'.format(mag, spec_res, ap_diam_see)
    in_file_rao = 'roboAO_tint_m{0:d}_R{1:d}_ap{2:0.3f}'.format(mag, spec_res, ap_diam_rao)

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
    
    

    
