import numpy as np
import pylab as py
from astropy.table import Table

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
