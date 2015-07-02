import numpy as np
import pylab as py
from astropy.table import Table
from jlu.util import statsIter
from hst_flystar import astrometry
from hst_flystar import completeness as comp
import os
import pdb

reduce_dir = '/u/jlu/data/wd1/hst/reduce_2015_01_05/'
cat_dir = reduce_dir + '50.ALIGN_KS2/'
art_dir = reduce_dir + '51.ALIGN_ART/'
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
    py.colorbar(orientation='horizontal', label='(milli-arsec)')
    py.title('X Error Mean')

    py.subplot(1, 2, 2)
    py.imshow(xe_std, extent=[xbins[0], xbins[-1], ybins[0], ybins[-1]],
              vmin=0.1, vmax=3)
    py.colorbar(orientation='horizontal', label='(milli-arsec)')
    py.title('X Error Std')
    py.savefig(plot_dir + 'map_of_errors_x_' + epoch + '.png')


    py.figure(2, figsize=(12, 6))
    py.clf()
    py.subplots_adjust(left=0.05, bottom=0.05)
    py.subplot(1, 2, 1)
    py.imshow(ye_mean, extent=[xbins[0], xbins[-1], ybins[0], ybins[-1]],
              vmin=0.1, vmax=3)
    py.colorbar(orientation='horizontal', label='(milli-arsec)')
    py.title('Y Error Mean')

    py.subplot(1, 2, 2)
    py.imshow(ye_std, extent=[xbins[0], xbins[-1], ybins[0], ybins[-1]],
              vmin=0.1, vmax=3)
    py.colorbar(orientation='horizontal', label='(milli-arsec)')
    py.title('Y Error Std')
    py.savefig(plot_dir + 'map_of_errors_y_' + epoch + '.png')



    py.figure(3, figsize=(12, 6))
    py.clf()
    py.subplots_adjust(left=0.05, bottom=0.05)
    py.subplot(1, 2, 1)
    py.imshow(me_mean, extent=[xbins[0], xbins[-1], ybins[0], ybins[-1]],
              vmin=0.01, vmax=0.07)
    py.colorbar(orientation='horizontal', label='(magnitude)')
    py.title('M Error Mean')

    py.subplot(1, 2, 2)
    py.imshow(me_std, extent=[xbins[0], xbins[-1], ybins[0], ybins[-1]],
              vmin=0.01, vmax=0.07)
    py.colorbar(orientation='horizontal', label='(magnitude)')
    py.title('M Error Std')
    py.savefig(plot_dir + 'map_of_errors_m_' + epoch + '.png')

    return

def debug_2005_F814W_ks2_vs_nimfo2bar():
    t = Table.read(art_dir + 'artstar_2005_F814W.fits')
    det = np.where((t['n_out'] > 1) & (t['m_in'] < -11) &
                   (t['xe_out'] < 0.5) & (t['ye_out'] < 0.5))[0]
    td = t[det]
    print '****'
    print '  {0} planted     {1} detected'.format(len(t), len(td))
    print '****'
    print '   NOTE: detected stars are brighter than -11 with'
    print '   pos. errors < 0.5 pixels'
    print ''

    dx = td['x_out'] - td['x_in']
    dy = td['y_out'] - td['y_in']
    dm = td['m_out'] - td['m_in']

    # Look at the two sub-populations:
    #   one with big differences in y
    #   one with small differences in  y
    idx_b = np.where(dy > -10)[0] # big (top chip)
    idx_s = np.where(dy < -10)[0] # small (bottom chip)
    print '{0} with small delta-y, {1} with large delta y'.format(len(idx_s),
                                                                  len(idx_b))
    py.figure(1)
    py.clf()
    py.plot(dx[idx_s], dy[idx_s], 'k.', ms=2, alpha=0.5, label='Bottom Chip')
    py.plot(dx[idx_b], dy[idx_b], 'r.', ms=2, alpha=0.5, label='Top Chip')
    py.xlabel('x_out - x_in (pix)')
    py.ylabel('y_out - y_in (pix)')
    py.axis('equal')
    py.legend(numpoints=1)
    py.title('Positional Differences\nfor Artificial Stars  ')
    py.savefig(plot_dir + 'wfc_err_ks2_vs_nimfo2bar_dxy_vpd.png')

    py.figure(2)
    py.clf()
    py.plot(td['x_in'][idx_s], td['y_in'][idx_s], 'k.', ms=2, alpha=0.5,
            label='Small Offset (<10 pix)')
    py.plot(td['x_in'][idx_b], td['y_in'][idx_b], 'r.', ms=2, alpha=0.5,
            label='Large Offset (>10 pix)')
    py.xlabel('x_in (pix)')
    py.ylabel('y_in (pix)')
    py.title('Detected Artificial Stars')
    py.savefig(plot_dir + 'wfc_err_ks2_vs_nimfo2bar_xy.png')

    # Identify a sub-set of stars to see distortion patterns.
    foo_s = np.where((td['m_in'][idx_s] > -13) & (td['m_in'][idx_s] < -12.7))[0]
    foo_b = np.where((td['m_in'][idx_b] > -13) & (td['m_in'][idx_b] < -12.7))[0]
    
    py.figure(4, figsize=(10,10))
    py.clf()
    qs = py.quiver(td['x_in'][idx_s][foo_s], td['y_in'][idx_s][foo_s],
                   dx[idx_s][foo_s], dy[idx_s][foo_s], color='black',
                   scale=1000)
    qb = py.quiver(td['x_in'][idx_b][foo_b], td['y_in'][idx_b][foo_b],
                   dx[idx_b][foo_b], dy[idx_b][foo_b], color='red',
                   scale=20)

    py.quiverkey(qs, 0.15, 0.93, 50, '50 pixels')
    py.quiverkey(qb, 0.50, 0.93, 1, '1 pixels')
    py.xlabel('x_in (pix)')
    py.ylabel('y_in (pix)')
    py.title('Note Scale Change')
    py.savefig(plot_dir + 'wfc_err_ks2_vs_nimfo2bar_quiver.png')

    # Temporarily modify both dx and dy to get rid of the means in the two
    # regions.
    dx[idx_s] -= np.median(dx[idx_s])
    dy[idx_s] -= np.median(dy[idx_s])
    dx[idx_b] -= np.median(dx[idx_b])
    dy[idx_b] -= np.median(dy[idx_b])
    
    py.clf()
    qs = py.quiver(td['x_in'][idx_s][foo_s], td['y_in'][idx_s][foo_s],
                   dx[idx_s][foo_s], dy[idx_s][foo_s], color='black',
                   scale=1e2)
    qb = py.quiver(td['x_in'][idx_b][foo_b], td['y_in'][idx_b][foo_b],
                   dx[idx_b][foo_b], dy[idx_b][foo_b], color='red',
                   scale=10)

    py.quiverkey(qs, 0.15, 0.93, 5, '5 pixels')
    py.quiverkey(qb, 0.50, 0.93, 0.5, '0.5 pixels')
    py.xlabel('x_in (pix)')
    py.ylabel('y_in (pix)')
    py.title('Median Offset Removed From Each Section/n(note scale change)')
    py.savefig(plot_dir + 'wfc_err_ks2_vs_nimfo2bar_quiver_median_removed.png')
    
    
    py.figure(3)
    py.clf()
    py.plot(td['m_in'], dx, 'r.', ms=2, alpha=0.5)
    py.plot(td['m_in'], dy, 'b.', ms=2, alpha=0.5)

    return
    
def plot_artstar_in_vs_out():
    """
    Examine the positional differences and magnitude differences
    for the artificially planted stars vs. what has come out of the
    KS2 and alignment process. Also look at the number of epochs that
    each artificial star was detected in.
    """
    t = Table.read(art_dir + 'art_align_a4_t_combo_pos.fits')

    epochs = ['2005_F814W', '2010_F125W', '2010_F139M', '2010_F160W', '2013_F160W']

    # For each epoch/filter starlist, compare the input and output
    # positions and magnitudes of the detected stars.
    for ee in range(len(epochs)):
        suffix = epochs[ee]
        print 'Plotting ', suffix
        
        det = np.where(t['n_' + suffix] > 1)[0]
        t_det = t[det]
        
        dx = t_det['x_' + suffix] - t_det['xin_' + suffix]
        dy = t_det['y_' + suffix] - t_det['yin_' + suffix]
        dm = t_det['m_' + suffix] - t_det['min_' + suffix]

        dx_avg = dx.mean()
        dy_avg = dy.mean()
        lim = 0.5
        mlim = 0.1

        mag_lim = [-15, -7]
        if ee > 0:
            mag_lim = [-13, -5]

        py.figure(2)
        py.clf()
        py.plot(dx, dy, 'k.', ms=2, alpha=0.2)
        py.axis([dx_avg-lim, dx_avg+lim, dy_avg-lim, dy_avg+lim])
        py.xlabel('x_out - x_in (pix)')
        py.ylabel('y_out - y_in (pix)')
        py.title(suffix)
        py.savefig(plot_dir + 'art_in_out_dxy_vpd_' + suffix + '.png')

        py.figure(1)
        py.clf()
        py.plot(t_det['min_' + suffix], dx, 'r.', ms=2, alpha=0.05, label='X')
        py.plot(t_det['min_' + suffix], dy, 'b.', ms=2, alpha=0.05, label='Y')
        py.legend()
        py.xlabel('Instr. Magnitude')
        py.ylabel('In - Out Position (pix)')
        py.xlim(mag_lim[0], mag_lim[1])
        py.ylim(-lim, lim)
        py.title(suffix)
        py.savefig(plot_dir + 'art_in_out_dxy_mag_' + suffix + '.png')

        py.clf()
        py.plot(t_det['min_' + suffix], dm, 'k.', ms=2, alpha=0.05)
        py.xlabel('Instr. Magnitude')
        py.ylabel('In - Out Magnitude (mag)')
        py.xlim(mag_lim[0], mag_lim[1])
        py.ylim(-mlim, mlim)
        py.title(suffix)
        py.savefig(plot_dir + 'art_in_out_dm_mag_' + suffix + '.png')

        bins = np.arange(-2, 2, 0.05)
        py.clf()
        py.hist(dx, label='X', color='red', histtype='step', bins=bins, log=True)
        py.hist(dy, label='Y', color='blue', histtype='step', bins=bins, log=True)
        py.xlabel('In - Out Position (pix)')
        py.ylabel('Number of Stars')
        py.legend()
        py.title(suffix)
        py.savefig(plot_dir + 'art_in_out_hist_dxy_' + suffix + '.png')

        bins = np.arange(-2, 2, 0.02)
        py.clf()
        py.hist(dm, histtype='step', bins=bins, log=True)
        py.xlabel('In - Out Magnitude (mag)')
        py.ylabel('Number of Stars')
        py.title(suffix)
        py.savefig(plot_dir + 'art_in_out_hist_dm_' + suffix + '.png')

        # Trim down before we make a quiver plot
        qscale = 9
        mag_lim = [-12, -11.8]
        dr_lim = 0.7
        dm_lim = 0.02
        
        if ee > 0:
            qscale = 0.5
            mag_lim = [-9, -8.8]
            dr_lim = 0.05
        
        dr = np.hypot(dx, dy)
        idx = np.where((dr < dr_lim) & (dm < dm_lim) &
                       (t_det['min_'+suffix] > mag_lim[0]) &
                       (t_det['min_'+suffix] < mag_lim[1]))[0]
        print 'N stars in Quiver: ', len(idx)

        py.close(3)
        py.figure(3, figsize=(12,12))
        py.clf()
        q = py.quiver(t_det['xin_' + suffix][idx], t_det['yin_' + suffix][idx],
                      dx[idx], dy[idx], scale=qscale)
        py.xlabel('X Position (pix)')
        py.ylabel('Y Position (pix)')
        py.title(suffix)

        if ee == 0:
            py.quiverkey(q, 0.95, 0.95, 0.5, '0.5 pixel', color='red')
        else:
            py.quiverkey(q, 0.95, 0.95, 0.05, '0.05 pixel', color='red')
            
        py.savefig(plot_dir + 'art_in_out_quiver_' + suffix + '.png')


        
