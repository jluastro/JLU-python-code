import numpy as np
import pylab as py
import pysynphot
import astropy.units as units
import astropy.constants as const
import math
from jlu.nirc2 import synthetic
import pdb


def snr_osiris_imager(tint, mag, filter, phot_sys='vegamag'):

    ##########
    # Detector Characteristics:
    ##########
    # http://www.teledyne-si.com/imaging/H4RG%20Brochure%20-%20rev3%20v2-2%20-%20OSR.pdf
    # http://proceedings.spiedigitallibrary.org/proceeding.aspx?articleid=1363315
    # Claire, personal communication (via Tuan)

    # Read noise (e- per read)
    read_noise = 5.0

    # Dark current (e- per second)
    dark_current = 0.05

    ##########
    # Throughput
    ##########
    # OSIRIS throughput
    # Keck is 77.9 m^2 (equivalent to 9.96 m circular aperture).
    # Assumed something between inscribed and circumscribed pupil cold stop.
    # Thanks to Tuan, Claire Max, and Fitz for this info.
    # Some from the OSIRIS manual.
    tp_window = 0.97
    tp_stop = 0.95
    tp_detector = 0.70
    tp_mirrors = 0.99**7

    osiris_throughput = tp_window * tp_stop * tp_mirrors * tp_detector
    ngao_throughput = 0.61
    tel_throughput = 0.80
    atm_throughput = 0.90

    # Total system throughput (spectrograph + AO system + telescope)
    throughput = osiris_throughput * ngao_throughput * tel_throughput * atm_throughput

    ##########
    # Background
    ##########
    # Used Tuan's calculations:
    # Temperatures:   Tel = 275  AO = 273
    # Emissivities:   Tel = 0.09 AO = 0.02
    # Units: mag / arcsec^2
    bkg_mag = {'z': 18.778,
               'y': 17.230,
               'J': 16.510,
               # 'J': 18.96,
               'H': 13.706,
               'K': 13.855,
               'Hcont': 13.607,
               'Kcont': 13.948}

    # Filter above the atmosphere (no telescope/instrument)
    #bandpass = synthetic.FilterNIRC2(filter)
    filterStr = filter
    if filter == 'z':
        filterStr = 'SDSS,z'
    if filter == 'y':
        filterStr = 'Stromgren,y'
    bandpass = pysynphot.ObsBandpass(filterStr)

    #star = pysynphot.Vega
    star = pysynphot.BlackBody(6000)
    star = star.renorm(mag, phot_sys, bandpass)

    # Observe the star through a filter
    star_in_filt = pysynphot.observation.Observation(star, bandpass,
                                                     binset=bandpass.wave)

    # Integrate over the filter
    star_flux = star_in_filt.integrate(fluxunits='photlam') # photons / s / cm^2
    print 'star_flux 1 = ', star_flux, ' photons s^-1 cm^-2'
    
    # Convert to observed flux on Keck primary mirror
    keck_area = 77.9 * 100.0 ** 2          # cm^2
    star_flux *= keck_area                 # photons / s
    print 'star_flux 2 = ', star_flux, ' photons s^-1'

    # Apply througput of atmosphere, telescope, AO, instrument.
    # This throughput already includes a quantum efficienciy correction.
    star_flux *= throughput                # e- / s
    print 'star_flux 3 = ', star_flux, ' e- s^-1'

    # Do the same for the background
    bkg = pysynphot.FlatSpectrum(1, waveunits='angstrom', fluxunits='flam')
    bkg = bkg.renorm(bkg_mag[filter], phot_sys, bandpass)
    bkg_in_filt = pysynphot.observation.Observation(bkg, bandpass,
                                                    binset=bandpass.wave)
    bkg_flux_dens = bkg_in_filt.integrate(fluxunits='photlam') # photons / s / cm^2 / arcsec^2
    bkg_flux_dens *= keck_area * throughput     # e- / s / arcsec^2
    

    # Aperture information
    # technically there should be an aperture correction on the star as well.
    # In the NIRC2 calculator, they just multiply by the Strehl.
    pix_scale = 0.01   # arcsec
    aper_radius = 0.1  # arcsec
    aper_area = math.pi * aper_radius ** 2  # arcsec^2
    npix = aper_area / pix_scale ** 2
    print 'npix = ', npix

    # Calculate signal-to-noise in the specified integration time.
    signal = star_flux * tint
    noise_variance = star_flux * tint
    noise_variance += bkg_flux_dens * tint * aper_area
    noise_variance += (read_noise ** 2) * npix
    noise_variance += dark_current * tint * npix
    noise = math.sqrt( noise_variance )

    snr = signal / noise

    print 'signal = ', signal
    print 'noise = ', noise
    print 'snr = ', snr

    return star_in_filt
    
    
