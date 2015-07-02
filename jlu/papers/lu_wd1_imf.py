import numpy as np
import pylab as py
from astropy.table import Table
from astropy.io import fits
import glob

data_dir = '/u/jlu/data/wd1/hst/reduce_2015_01_05/'
cat_dir = '/u/jlu/data/wd1/hst/reduce_2015_01_05/50.ALIGN_KS2/'
catalog = 'wd1_catalog_EOM_wvelNone.fits'

def make_plots():
    plot_astrometric_error()

def plot_astrometric_error():
    """
    Plot the astrometric error vs. magnitude.
    """
    cat = Table.read(catalog)

    mag_bins = np.arange(12, 24, 0.25)
    epochs = ['2005_F814W', '2010_F125W', '2010_F139M', '2010_F160W', '2013_F160W']
    n_mag = len(mag_bins)
    n_epochs = len(epochs)
    
    xe = np.zeros((len(mag_bins), len(epochs)), dtype=float)
    ye = np.zeros((len(mag_bins), len(epochs)), dtype=float)
    me = np.zeros((len(mag_bins), len(epochs)), dtype=float)

    for ee in range(n_epochs):
        for mm in range(n_mag):
            m_all = t['m_' + epochs[ee]]
            xe_all = t['xe_' + epochs[ee]]
            ye_all = t['ye_' + epochs[ee]]
            me_all = t['me_' + epochs[ee]]

def get_position_angle():
    """
    Get the average position angle for each filter/epoch combo to
    populate the Observations table.
    """
    epochs = ['2005_F814W', '2010_F125W', '2010_F139M', '2010_F160W', '2013_F160W',
              '2013_F160Ws']

    for ep in epochs:
        fits_files = glob.glob(data_dir + ep + '/00.DATA/*_flt.fits')

        print ''
        print 'FITS Files for Epoch = ', ep

        pa_array = np.zeros(len(fits_files), dtype=float)

        for ff in range(len(fits_files)):
            pa_array[ff] = fits.getval(fits_files[ff], 'ORIENTAT', 1)

            # print '{0:20s} ORIENTAT = {1:6.2f}'.format(fits_files[ff],
            #                                            pa_array[ff])

        fmt = 'Epoch: {0:10s} mean PA = {1:6.2f} +/- {2:6.2f}'
        print fmt.format(ep, pa_array.mean(), pa_array.std())
    
