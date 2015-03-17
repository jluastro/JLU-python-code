import numpy as np
import pylab as py
from astropy.table import Table
from jlu.util import statsIter
from hst_flystar import astrometry
from hst_flystar import completeness as comp
import os

reduce_dir = '/u/jlu/data/wd1/hst/reduce_2015_01_05/'
cat_dir = reduce_dir + '50.ALIGN_KS2/'
plot_dir = '/u/jlu/work/wd1/analysis_2015_01_05/plots/'

catalog = 'wd1_catalog_RMSE_wvelNone.fits'

def plot_err_vs_mag(epoch):
    t = Table.read(cat_dir + catalog)

    x = t['x_' + epoch]
    y = t['y_' + epoch]
    m = t['m_' + epoch]
    xe = t['xe_' + epoch] * astrometry.scale['WFC'] * 1e3
    ye = t['ye_' + epoch] * astrometry.scale['WFC'] * 1e3
    me = t['me_' + epoch]

    py.close('all')
    py.figure(1, figsize=(8,12))
    py.subplots_adjust(hspace=0.001, bottom=0.08, top=0.95, left=0.15)
    py.clf()

    ax1 = py.subplot(311)
    ax1.semilogy(m, xe, 'r.', ms=3, label='X', alpha=0.5)
    ax1.semilogy(m, ye, 'b.', ms=3, label='Y', alpha=0.5)
    ax1.legend(loc='upper left', numpoints=1)
    ax1.set_ylabel('Pos. Error (mas)')
    ax1.set_title(epoch)
    mlim = ax1.get_xlim()

    ax2 = py.subplot(312, sharex=ax1)
    ax2.semilogy(m, me, 'k.', ms=2)
    ax2.set_ylabel('Photo. Error (mag)')

    mbins = np.arange(mlim[0], mlim[1]+0.1, 0.5)    
    ax3 = py.subplot(313, sharex=ax1)
    ax3.hist(m, bins=mbins)
    ax3.set_ylabel('N Stars')
    ax3.set_xlabel('Magnitude')

    xticklabels = ax1.get_xticklabels() + ax2.get_xticklabels()
    py.setp(xticklabels, visible=False)
    
    py.savefig(plot_dir + 'err_vs_mag_' + epoch + '.png')
    
    

def map_of_errors(epoch, pos_err_cut=3.0):
    """
    pos_err_cut - Positional error cut in milli-arcseconds.
    Choose this based on the plots generated from plot_err_vs_mag().
    """
    t = Table.read(cat_dir + catalog)

    xbins = np.arange(0, 4251, 250)
    ybins = np.arange(0, 4200, 250)

    xb2d, yb2d = np.meshgrid(xbins, ybins)

    xe_mean = np.zeros(xb2d.shape, dtype=float)
    ye_mean = np.zeros(yb2d.shape, dtype=float)
    me_mean = np.zeros(yb2d.shape, dtype=float)
    xe_std = np.zeros(xb2d.shape, dtype=float)
    ye_std = np.zeros(yb2d.shape, dtype=float)
    me_std = np.zeros(yb2d.shape, dtype=float)

    x = t['x_' + epoch]
    y = t['y_' + epoch]
    xe = t['xe_' + epoch] * astrometry.scale['WFC'] * 1e3
    ye = t['ye_' + epoch] * astrometry.scale['WFC'] * 1e3
    me = t['me_' + epoch]

    for xx in range(len(xbins)-1):
        for yy in range(len(ybins)-1):
            idx = np.where((x > xbins[xx]) & (x <= xbins[xx+1]) &
                           (y > ybins[yy]) & (y <= ybins[yy+1]) &
                           (xe < pos_err_cut) & (ye < pos_err_cut))[0]

            if len(idx) > 0:
                xe_mean[yy, xx] = statsIter.mean(xe[idx], hsigma=3, iter=5)
                ye_mean[yy, xx] = statsIter.mean(ye[idx], hsigma=3, iter=5)
                me_mean[yy, xx] = statsIter.mean(me[idx], hsigma=3, iter=5)
                xe_std[yy, xx] = statsIter.std(xe[idx], hsigma=3, iter=5)
                ye_std[yy, xx] = statsIter.std(ye[idx], hsigma=3, iter=5)
                me_std[yy, xx] = statsIter.std(me[idx], hsigma=3, iter=5)


    py.close('all')
    py.figure(1, figsize=(12, 6))
    py.clf()
    py.subplots_adjust(left=0.05, bottom=0.05)
    py.subplot(1, 2, 1)
    py.imshow(xe_mean, extent=[xbins[0], xbins[-1], ybins[0], ybins[-1]],
              vmin=0.1, vmax=3)
    py.colorbar(orientation='horizontal')
    py.title('X Error Mean')

    py.subplot(1, 2, 2)
    py.imshow(xe_std, extent=[xbins[0], xbins[-1], ybins[0], ybins[-1]],
              vmin=0.1, vmax=3)
    py.colorbar(orientation='horizontal')
    py.title('X Error Std')
    py.savefig(plot_dir + 'map_of_errors_x_' + epoch + '.png')


    py.figure(2, figsize=(12, 6))
    py.clf()
    py.subplots_adjust(left=0.05, bottom=0.05)
    py.subplot(1, 2, 1)
    py.imshow(ye_mean, extent=[xbins[0], xbins[-1], ybins[0], ybins[-1]],
              vmin=0.1, vmax=3)
    py.colorbar(orientation='horizontal')
    py.title('Y Error Mean')

    py.subplot(1, 2, 2)
    py.imshow(ye_std, extent=[xbins[0], xbins[-1], ybins[0], ybins[-1]],
              vmin=0.1, vmax=3)
    py.colorbar(orientation='horizontal')
    py.title('Y Error Std')
    py.savefig(plot_dir + 'map_of_errors_y_' + epoch + '.png')



    py.figure(3, figsize=(12, 6))
    py.clf()
    py.subplots_adjust(left=0.05, bottom=0.05)
    py.subplot(1, 2, 1)
    py.imshow(me_mean, extent=[xbins[0], xbins[-1], ybins[0], ybins[-1]],
              vmin=0.01, vmax=0.07)
    py.colorbar(orientation='horizontal')
    py.title('M Error Mean')

    py.subplot(1, 2, 2)
    py.imshow(me_std, extent=[xbins[0], xbins[-1], ybins[0], ybins[-1]],
              vmin=0.01, vmax=0.07)
    py.colorbar(orientation='horizontal')
    py.title('M Error Std')
    py.savefig(plot_dir + 'map_of_errors_m_' + epoch + '.png')

    return

    

