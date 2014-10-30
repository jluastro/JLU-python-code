from pyraf import iraf as ir
import pyfits
import math
import atpy
import numpy as np
import pylab as py
import pickle, glob
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import os
from scipy import interpolate
import jlu

def setup_phot(imageRoot, silent=False,
               apertures=[25,50,75,100,125,150,175,200],
               sky_annulus=200, sky_dannulus=50, zmag=0):

    # Load image header
    hdr = pyfits.getheader(imageRoot + '.fits')

    ir.digiphot()
    ir.daophot()
    ir.unlearn('phot')
    ir.unlearn('datapars')
    ir.unlearn('centerpars')
    ir.unlearn('fitskypars')
    ir.unlearn('photpars')

    ##########
    # Set up datapars
    ##########
    ir.datapars.fwhmpsf = 5.0 # shouldn't really matter
    ir.datapars.sigma = 'INDEF'
    ir.datapars.datamin = 'INDEF'

    if os.path.exists(imageRoot + '.max'):
        max_file = open(imageRoot + '.max', 'r')
        max_line = max_file.readline()
        max = float(max_line)
        ir.datapars.datamax = max

        if not silent:
            print 'Set ir.datapars.datamax = %d' % max

    # Pull gain from the header
    ir.datapars.gain = 'GAIN'
    ir.datapars.epadu = 'INDEF'

    # Assumes 43.1 electrons per read of noise
    nreads = 1.0
    if int(hdr['SAMPMODE']) == 3:
        nreads = int(hdr['MULTISAM'])
    
    ir.datapars.ccdread = ''
    ir.datapars.readnoise = 43.1 * math.sqrt(2.0) / math.sqrt(nreads)

    # Get exposure times from header
    ir.datapars.exposure = ''
    ir.datapars.itime = float(hdr['ITIME']) * int(hdr['COADDS'])

    # Other Header keywords
    ir.datapars.airmass = 'AIRMASS'
    ir.datapars.filter = 'FWINAME'
    ir.datapars.obstime = 'EXPSTART'

    
    ##########
    # Setup centerpars. We will use *.coo file for initial guess.
    ##########
    ir.centerpars.calgorithm = 'centroid'

    ##########
    # Setup fitskypars
    ##########
    ir.fitskypars.salgorithm = 'centroid'
    ir.fitskypars.annulus = sky_annulus
    ir.fitskypars.dannulus = sky_dannulus

    ##########
    # Setup photpars
    ##########
    # Setup a zeropoint... this assumes Strehl = 1, but good enough for now.
    ir.photpars.zmag = zmag
    ir.photpars.apertures = ','.join([str(aa) for aa in apertures])
    
    ##########
    # Setup phot
    ##########
    ir.phot.interactive = 'no'
    ir.phot.radplots = 'no'
    ir.phot.verify = 'No'

    if silent:
        ir.phot.verbose = 'no'
    else:
        ir.phot.verbose = 'yes'

def run_phot(imageRoot, silent=False,
             apertures=[25,50,75,100,125,150,175,200],
             sky_annulus=200, sky_dannulus=50, zmag=0):

    setup_phot(imageRoot, apertures=apertures, zmag=zmag, silent=silent,
               sky_annulus=sky_annulus, sky_dannulus=sky_dannulus)

    image = imageRoot + '.fits'
    coords = imageRoot + '.coo'

    # Output into current directory, not data directory
    rootSplit = imageRoot.split('/')
    output = rootSplit[-1] + '.phot.mag'

    ir.phot(image, coords, output)

    (radius, flux, mag, merr) = get_phot_output(output, silent=silent)

    return (radius, flux, mag, merr)

def get_phot_output(output, silent=False):
    # Now get the results using txdump
    radStr = ir.txdump(output, 'RAPERT', 'yes', Stdout=1)
    fluxStr = ir.txdump(output, 'FLUX', 'yes', Stdout=1)
    magStr = ir.txdump(output, 'MAG', 'yes', Stdout=1)
    merrStr = ir.txdump(output, 'MERR', 'yes', Stdout=1)
    pierStr = ir.txdump(output, 'PIER', 'yes', Stdout=1)

    radFields = radStr[0].split()
    fluxFields = fluxStr[0].split()
    magFields = magStr[0].split()
    merrFields = merrStr[0].split()
    pierFields = pierStr[0].split()

    count = len(radFields)

    radius = np.zeros(count, dtype=float)
    flux = np.zeros(count, dtype=float)
    mag = np.zeros(count, dtype=float)
    merr = np.zeros(count, dtype=float)

    for rr in range(count):
        radius[rr] = float(radFields[rr])

        if (int(pierFields[rr]) != 0 or magFields[rr] == 'INDEF' or
            merrFields[rr] == 'INDEF'):
            print 'Problem in image: ' + output

            # Error
            flux[rr] = 0
            mag[rr] = 0
            merr[rr] = 0
        else:
            flux[rr] = float(fluxFields[rr])
            mag[rr] = float(magFields[rr])
            merr[rr] = float(merrFields[rr])

    if not silent:
        print '%6s  %10s  %6s  %6s' % ('Radius', 'Flux', 'Mag', 'MagErr')
        for ii in range(count):
            print '%8.1f  %10d  %6.3f  %6.3f' % \
                (radius[ii], flux[ii], mag[ii], merr[ii])
    
    return (radius, flux, mag, merr)

def get_filter_profile(filter):
    """
    Returns the wavelength (in microns) and the transmission for 
    the specified NIRC2 filter.

    Example: 
    (wave, trans) = nirc2.photometry.get_filter_profile('Kp')
    py.clf()
    py.plot(wave, trans)
    py.xlabel('Wavelength (microns)')
    py.ylabel('Transmission')
    """
    base_path = os.path.dirname(jlu.__file__)
    rootDir = base_path + '/nirc2/filters/'

    filters = ['J', 'H', 'K', 'Kcont', 'Kp', 'Ks', 'Lp', 'Ms',
               'Hcont', 'Brgamma', 'FeII']

    if filter not in filters:
        print 'Could not find profile for filter %s.' % filter
        print 'Choices are: ', filters
        return

    table = atpy.Table(rootDir + filter + '.dat', type='ascii')

    wavelength = table[table.keys()[0]]
    transmission = table[table.keys()[1]]

    # Lets fix wavelength array for duplicate values
    diff = np.diff(wavelength)
    idx = np.where(diff <= 0)[0]
    wavelength[idx+1] += 1.0e-7

    # Get rid of all entries with negative transmission
    idx = np.where(transmission > 1)[0]
    wavelength = wavelength[idx]
    transmission = transmission[idx] / 100.0 # convert from % to ratio

    return (wavelength, transmission)

def test_filter_profile_interp():
    """
    Plot up the filter transmission curves and their interpolations
    for the three K-band filters (K, Kp, Ks).
    """
    # Get the transmission curve for NIRC2 filters and atmosphere.
    K_wave, K_trans = get_filter_profile('K')
    Kp_wave, Kp_trans = get_filter_profile('Kp')
    Ks_wave, Ks_trans = get_filter_profile('Ks')
    J_wave, J_trans = get_filter_profile('J')
    H_wave, H_trans = get_filter_profile('H')
    Lp_wave, Lp_trans = get_filter_profile('Lp')

    # We will need to resample these transmission curves.
    print 'Creating interp object'
    K_interp = interpolate.splrep(K_wave, K_trans, k=1, s=0)
    Kp_interp = interpolate.splrep(Kp_wave, Kp_trans, k=1, s=0)
    Ks_interp = interpolate.splrep(Ks_wave, Ks_trans, k=1, s=0)
    J_interp = interpolate.splrep(J_wave, J_trans, k=1, s=0)
    H_interp = interpolate.splrep(H_wave, H_trans, k=1, s=0)
    Lp_interp = interpolate.splrep(Lp_wave, Lp_trans, k=1, s=0)

    K_wave_new = np.arange(K_wave.min(), K_wave.max(), 0.0005)
    Kp_wave_new = np.arange(Kp_wave.min(), Kp_wave.max(), 0.0005)
    Ks_wave_new = np.arange(Ks_wave.min(), Ks_wave.max(), 0.0005)
    J_wave_new = np.arange(J_wave.min(), J_wave.max(), 0.0005)
    H_wave_new = np.arange(H_wave.min(), H_wave.max(), 0.0005)
    Lp_wave_new = np.arange(Lp_wave.min(), Lp_wave.max(), 0.0005)

    print 'Interpolating'
    K_trans_new = interpolate.splev(K_wave_new, K_interp)
    Kp_trans_new = interpolate.splev(Kp_wave_new, Kp_interp)
    Ks_trans_new = interpolate.splev(Ks_wave_new, Ks_interp)
    J_trans_new = interpolate.splev(J_wave_new, J_interp)
    H_trans_new = interpolate.splev(H_wave_new, H_interp)
    Lp_trans_new = interpolate.splev(Lp_wave_new, Lp_interp)

    print 'Plotting'
#     py.figure(2, figsize=(4,4))
#     py.subplots_adjust(left=0.2, bottom=0.14, top=0.95, right=0.94)
    py.clf()
    py.plot(K_wave, K_trans, 'bo', ms=4, label='_nolegend_', mec='blue')
    py.plot(K_wave_new, K_trans_new, 'b-', label='K', linewidth=2)

    py.plot(Kp_wave, Kp_trans, 'ro', ms=4, label='_nolegend_', mec='red')
    py.plot(Kp_wave_new, Kp_trans_new, 'r-', label='Kp', linewidth=2)

    py.plot(Ks_wave, Ks_trans, 'go', ms=4, label='_nolegend_', mec='green')
    py.plot(Ks_wave_new, Ks_trans_new, 'g-', label='Ks', linewidth=2)

    py.plot(J_wave, J_trans, 'go', ms=4, label='_nolegend_', mec='green')
    py.plot(J_wave_new, J_trans_new, 'g-', label='J', linewidth=2)

    py.plot(H_wave, H_trans, 'go', ms=4, label='_nolegend_', mec='green')
    py.plot(H_wave_new, H_trans_new, 'g-', label='H', linewidth=2)

    py.plot(Lp_wave, Lp_trans, 'go', ms=4, label='_nolegend_', mec='green')
    py.plot(Lp_wave_new, Lp_trans_new, 'g-', label='Lp', linewidth=2)
    
    py.legend(loc='lower right', numpoints=1, markerscale=0.1)
    py.xlabel('Wavelength (microns)')
    py.ylabel('Transmission (%)')

#     py.axis([2.110, 2.120, 0.928, 0.945])

def test_atmosphere_profile_interp():
    atmDir = '/u/jlu/data/w51/09jun26/weather/atmosphere_transmission.dat'
    atmData = atpy.Table(atmDir, type='ascii')
    atm_wave = atmData[atmData.keys()[0]]
    atm_trans = atmData[atmData.keys()[1]]

    atm_interp = interpolate.splrep(atm_wave, atm_trans, k=1, s=1)

    atm_wave_new = np.arange(2.0, 2.4, 0.0005)
    atm_trans_new = interpolate.splev(atm_wave_new, atm_interp)

    py.clf()
    py.plot(atm_wave, atm_trans, 'r.', ms=2)
    py.plot(atm_wave_new, atm_trans_new, 'b-')
    py.xlabel('Wavelength (microns)')
    py.ylabel('Transmission (%)')
    py.xlim(2, 2.4)
