import numpy as np
import pylab as py
from astropy.table import Table, Column
from jlu.util import statsIter
from hst_flystar import astrometry
from hst_flystar import completeness as comp
import os
import pdb
from jlu.wd1.analysis import membership
from scipy.stats import chi2
from scipy import interpolate, optimize
import matplotlib
from popstar import synthetic, reddening, evolution
from jlu.stellarModels import extinction
import math
import matplotlib.animation as animation

# On LHCC
reduce_dir = '/Users/jlu/data/wd1/hst/reduce_2015_01_05/'
work_dir = '/Users/jlu/work/wd1/analysis_2015_01_05/'
# Have two different isochrone directories depending on redlaw used
iso_dir_nishi09 = '/Users/jlu/work/wd1/models/iso_2015/nishi09/'
iso_dir_wd1 = '/Users/jlu/work/wd1/models/iso_2015/wd1/'

synthetic.redlaw = reddening.RedLawWesterlund1()
#synthetic.redlaw = reddening.RedLawNishiyama09()

# On Laptop
# synthetic.redlaw = reddening.RedLawWesterlund1()
# iso_dir = '/Users/jlu/work/wd1/iso_2015_wd1/'

# synthetic.redlaw = reddening.RedLawRiekeLebofsky()
# iso_dir = '/Users/jlu/work/wd1/iso_2015_rieke/'

# synthetic.redlaw = reddening.RedLawRomanZuniga07()
# iso_dir = '/Users/jlu/work/wd1/iso_2015_roman/'

# synthetic.redlaw = reddening.RedLawCardelli()
# iso_dir = '/Users/jlu/work/wd1/iso_2015_cardelli/'

# iso_dir = '/Users/jlu/work/wd1/iso_2015/'

# reduce_dir = '/Users/jlu/work/wd1/'
# work_dir = '/Users/jlu/work/wd1/'
# evolution.models_dir = '/Users/jlu/work/models/evolution/'

cat_dir = reduce_dir + '50.ALIGN_KS2/'
art_dir = reduce_dir + '51.ALIGN_ART/'
plot_dir = work_dir + 'plots/'

catalog = 'wd1_catalog_RMSE_wvelErr.fits'

# Catalogs after membership calculations
cat_pclust = work_dir + 'membership/gauss_3/catalog_membership_3_rot.fits'
cat_pclust_pcolor = work_dir + 'catalog_membership_3_rot_Pcolor.fits'

# Artificial catalog
art_cat = art_dir + 'wd1_art_catalog_RMSE_wvelErr_aln_art.fits'


# Best fit (by eye) cluster parameters
wd1_logAge = 6.7
wd1_AKs = 0.69
wd1_distance = 4000

def plot_err_vs_mag(epoch):
    t = Table.read(plat_dir + catalog)

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
    
def plot_artstar_in_vs_out(use_obs_align=False):
    """
    Examine the positional differences and magnitude differences
    for the artificially planted stars vs. what has come out of the
    KS2 and alignment process. Also look at the number of epochs that
    each artificial star was detected in.
    """
    in_file = art_dir + 'art_align'
    if use_obs_align:
        in_file += '_obs'
    else:
        in_file += '_art'
    in_file += '_combo_pos_det_newerr.fits'
    t = Table.read(in_file)

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

        mag_lim = [12, 24]
        if ee > 0:
            mag_lim = [13, 21]

        if use_obs_align:
            out_suffix = suffix + '_obs'
        else:
            out_suffix = suffix + '_art'
        
        py.close(2)
        py.figure(2)
        py.clf()
        py.plot(dx, dy, 'k.', ms=2, alpha=0.2)
        py.axis([dx_avg-lim, dx_avg+lim, dy_avg-lim, dy_avg+lim])
        py.xlabel('x_out - x_in (pix)')
        py.ylabel('y_out - y_in (pix)')
        py.title(suffix)
        py.savefig(plot_dir + 'art_in_out_dxy_vpd_' + out_suffix + '.png')

        py.close(1)
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
        py.savefig(plot_dir + 'art_in_out_dxy_mag_' + out_suffix + '.png')

        py.clf()
        py.plot(t_det['min_' + suffix], dm, 'k.', ms=2, alpha=0.05)
        py.xlabel('Instr. Magnitude')
        py.ylabel('In - Out Magnitude (mag)')
        py.xlim(mag_lim[0], mag_lim[1])
        py.ylim(-mlim, mlim)
        py.title(suffix)
        py.savefig(plot_dir + 'art_in_out_dm_mag_' + out_suffix + '.png')

        bins = np.arange(-2, 2, 0.05)
        py.clf()
        py.hist(dx, label='X', color='red', histtype='step', bins=bins, log=True)
        py.hist(dy, label='Y', color='blue', histtype='step', bins=bins, log=True)
        py.xlabel('In - Out Position (pix)')
        py.ylabel('Number of Stars')
        py.legend()
        py.title(suffix)
        py.savefig(plot_dir + 'art_in_out_hist_dxy_' + out_suffix + '.png')

        bins = np.arange(-2, 2, 0.02)
        py.clf()
        py.hist(dm, histtype='step', bins=bins, log=True)
        py.xlabel('In - Out Magnitude (mag)')
        py.ylabel('Number of Stars')
        py.title(suffix)
        py.savefig(plot_dir + 'art_in_out_hist_dm_' + out_suffix + '.png')

        # Trim down before we make a quiver plot
        qscale = 0.2
        mag_lim = [20, 20.4]
        dr_lim = 0.7
        dm_lim = 0.02
        
        if ee > 0:
            qscale = 0.5
            mag_lim = [13, 13.7]
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
            
        py.savefig(plot_dir + 'art_in_out_quiver_' + out_suffix + '.png')


def plot_velerr_vs_mag():
    """
    Plot the final proper motion error vs. magnitude. This plot
    is essential to choosing the stars that will go into the membership analysis.

    We will make a few other plots as well just to look for the expected
    correlations between the velocity errors and the positional errors.
    """
    t = Table.read(cat_dir + catalog)

    epoch = '2013_F160W'
    m = t['m_' + epoch]
    me = t['me_' + epoch]
    xe = t['xe_' + epoch] * astrometry.scale['WFC'] * 1e3
    ye = t['ye_' + epoch] * astrometry.scale['WFC'] * 1e3
    vxe = t['fit_vxe'] * astrometry.scale['WFC'] * 1e3
    vye = t['fit_vye'] * astrometry.scale['WFC'] * 1e3

    py.close('all')
    py.figure()
    py.clf()
    py.plot(m, vxe, 'r.', ms=3, label='X', alpha=0.5)
    py.plot(m, vye, 'b.', ms=3, label='Y', alpha=0.5)
    py.legend(loc='upper left', numpoints=1)
    py.ylabel('Proper Motion Error (mas/yr)')
    py.xlabel('F153M Magnitude from 2013')
    py.ylim(0, 1)
    py.xlim(12, 22)
    py.savefig(plot_dir + 'velerr_vs_mag.png')

    py.clf()
    py.plot(me, vxe, 'r.', ms=3, label='X', alpha=0.5)
    py.plot(me, vye, 'b.', ms=3, label='Y', alpha=0.5)
    py.legend(loc='upper left', numpoints=1)
    py.xlabel('2013 F153M Magnitude Error')
    py.ylabel('Proper Motion Error (mas/yr)')
    py.ylim(0, 1)
    py.xlim(0, 0.1)
    py.savefig(plot_dir + 'velerr_vs_magerr.png')

    py.clf()
    py.plot(xe, vxe, 'r.', ms=3, label='X', alpha=0.5)
    py.plot(ye, vye, 'b.', ms=3, label='Y', alpha=0.5)
    py.legend(loc='upper left', numpoints=1)
    py.xlabel('2013 F153M Positional Error (mas)')
    py.ylabel('Proper Motion Error (mas/yr)')
    py.ylim(0, 1)
    py.xlim(0, 5)
    py.savefig(plot_dir + 'velerr_vs_poserr.png')
        

def check_velocity_fits(magCut=18, outDir=plot_dir):
    """
    Check the goodness of the velocity fits by examining the residuals
    and chi-squared distributions. 
    """
    cat = Table.read(cat_dir + catalog)

    t = np.array([cat['t_2005_F814W'], cat['t_2010_F160W'], cat['t_2013_F160W']])
    x = np.array([cat['x_2005_F814W'], cat['x_2010_F160W'], cat['x_2013_F160W']])
    y = np.array([cat['y_2005_F814W'], cat['y_2010_F160W'], cat['y_2013_F160W']])
    xe = np.array([cat['xe_2005_F814W'], cat['xe_2010_F160W'], cat['xe_2013_F160W']])
    ye = np.array([cat['ye_2005_F814W'], cat['ye_2010_F160W'], cat['ye_2013_F160W']])
    m = cat['m_2013_F160W']

    dt = t - cat['fit_t0']
    x_fit = cat['fit_x0'] + (cat['fit_vx'] * dt)
    y_fit = cat['fit_y0'] + (cat['fit_vy'] * dt)

    dx = x - x_fit
    dy = y - y_fit

    # Convert to mas
    dx *= astrometry.scale['WFC'] * 1e3
    dy *= astrometry.scale['WFC'] * 1e3
    xe *= astrometry.scale['WFC'] * 1e3
    ye *= astrometry.scale['WFC'] * 1e3

    x_chi2 = (dx**2 / xe**2).sum(axis=0)
    y_chi2 = (dy**2 / ye**2).sum(axis=0)

    idx = np.where(m < magCut)[0]
    m = m[idx]
    dx = dx[:, idx]
    dy = dy[:, idx]
    xe = xe[:, idx]
    ye = ye[:, idx]
    x_chi2 = x_chi2[idx]
    y_chi2 = y_chi2[idx]

    # Get the residuals in units of sigma.
    dx_sig = dx / xe
    dy_sig = dy / ye

        
    bins = np.arange(-3, 3, 0.1)
    py.close(1)
    py.figure(1)
    py.clf()
    py.subplots_adjust(top=0.95, hspace=0.35)
    py.subplot(211)
    py.hist(dx[0,:], color='black', bins=bins, histtype='step', label='2005,F814W')
    py.hist(dx[1,:], color='red', bins=bins, histtype='step', label='2010,F160W')
    py.hist(dx[2,:], color='blue', bins=bins, histtype='step', label='2013,F160W')
    py.legend(loc='upper right', fontsize=12)
    py.xlabel('X Residuals (mas)')
    py.ylabel('Number of Stars')
    
    py.subplot(212)
    py.hist(dy[0,:], color='black', bins=bins, histtype='step')
    py.hist(dy[1,:], color='red', bins=bins, histtype='step')
    py.hist(dy[2,:], color='blue', bins=bins, histtype='step')
    py.xlabel('Y Residuals (mas)')
    py.ylabel('Number of Stars')

    py.savefig('{0:s}hist_vel_resids_m{1:.1f}.png'.format(outDir, magCut))
    
    
    bins = np.arange(0, 10, 0.1)
    bin_cent = bins[0:-1] + (np.diff(bins) / 2.0)
    dof = 1
    chi2_fit = chi2.pdf(bin_cent, dof)
    chi2_fit *= len(x_chi2) / chi2_fit.sum()

    py.close(2)    
    py.figure(2)
    py.clf()
    py.subplots_adjust(top=0.95, hspace=0.35)
    py.subplot(211)
    py.hist(x_chi2, bins=bins, color='black', log=True, histtype='step')
    py.plot(bin_cent, chi2_fit, 'r-', linewidth=2)
    py.ylim(0.1, 2000)
    py.xlabel(r'$\chi^2$ for X fits (DOF=1)')
    py.ylabel('Number of Stars')

    py.subplot(212)
    py.hist(y_chi2, bins=bins, color='black', log=True, histtype='step')
    py.plot(bin_cent, chi2_fit, 'r-', linewidth=2)
    py.ylim(0.1, 2000)
    py.ylabel('Number of Stars')
    py.xlabel(r'$\chi^2$ for Y fits (DOF=1)')
    
    py.savefig('{0:s}hist_vel_chi2_m{1:.1f}.png'.format(outDir, magCut))

    return

            
def run_membership(N_gauss=3):
    """
    Run the membership analysis.
    """
    mag_err_cut = 1.0 # (basically no cut)
    vel_err_cut = 0.5 # mas/yr

    out_dir = '{0}membership/gauss_{1:d}/'.format(work_dir, N_gauss)
    membership.run(cat_dir + catalog, vel_err_cut, mag_err_cut, N_gauss,
                   out_dir, rotate=True)
    membership.plot_posteriors(out_dir, N_gauss)

    return

def make_cluster_catalog():
    mag_err_cut = 1.0 # (basically no cut)
    # vel_err_cut = 0.5 # mas/yr
    vel_err_cut = 10 # mas/yr (basically no cut)
    prob = 0.3
    N_gauss = 3
    out_dir = '{0}membership/gauss_{1:d}/'.format(work_dir, N_gauss)
    
    membership.cluster_membership(cat_dir + catalog,
                                  vel_err_cut, mag_err_cut,
                                  out_dir, N_gauss, prob, 
                                  rotate=True)

    return
    

def make_cmd(catalog=cat_pclust_pcolor, cl_prob=0.3, usePcolor=False, suffix=''):
    """
    Plot the total CMD and then the CMD of only cluster members.

    Parameters:
    
    """
    # Read in data table
    d = Table.read(catalog)

    if usePcolor:
        d['Membership'] *= d['Membership_color']

    # Determine which we will call "cluster members"
    clust = np.where(d['Membership'] > cl_prob)[0]
    d_cl = d[clust]

    ##########
    # Plot CMD with everything in black.
    ##########
    mag1 = d['m_2013_F160W']
    mag2 = d['m_2005_F814W']
    color = mag2 - mag1

    py.close('all')
    py.figure()
    py.clf()
    py.plot(color, mag2, 'k.', ms=2)
    py.ylabel('F814W (mag)')
    py.xlabel('F814W - F160W (mag)')
    py.gca().invert_yaxis()
    py.xlim(2, 10)
    py.savefig(plot_dir + 'cmd_all' + suffix + '.png')
    
    # Plot CMD with everything in black
    # and cluster members in red. 
    py.clf()
    py.plot(color, mag2, 'k.', ms=2)
    py.plot(color[clust], mag2[clust], 'r.', ms=2)
    py.ylabel('F814W (mag)')
    py.xlabel('F814W - F160W (mag)')
    py.gca().invert_yaxis()
    py.xlim(2, 10)
    py.savefig(plot_dir + 'cmd_all_clust' + suffix + '.png')

    # Plot CMD with cluster members in black.
    py.clf()
    py.plot(color[clust], mag2[clust], 'k.', ms=2)
    py.ylabel('F814W (mag)')
    py.xlabel('F814W - F160W (mag)')
    py.gca().invert_yaxis()
    py.xlim(2, 10)
    py.savefig(plot_dir + 'cmd_clust' + suffix + '.png')




        
    ##########
    # IR Plot CMD with cluster members in black.
    ##########
    mag1 = d['m_2013_F160W']
    mag2 = d['m_2010_F125W']
    color = mag2 - mag1
    py.clf()
    py.plot(color[clust], mag2[clust], 'k.', ms=2)
    py.ylabel('F125W (mag)')
    py.xlabel('F125W - F160W (mag)')
    py.xlim(0, 2)
    py.ylim(23, 12)
    py.savefig(plot_dir + 'cmd_ir_clust' + suffix + '.png')

    # IR Plot CMD with everything in black.
    py.clf()
    py.plot(color, mag2, 'k.', ms=2)
    py.ylabel('F125W (mag)')
    py.xlabel('F125W - F160W (mag)')
    py.xlim(0, 2)
    py.ylim(23, 12)
    py.savefig(plot_dir + 'cmd_ir_all' + suffix + '.png')
    
    # IR Plot CMD with everything in black
    # and cluster members in red. 
    py.clf()
    py.plot(color, mag2, 'k.', ms=2)
    py.plot(color[clust], mag2[clust], 'r.', ms=2)
    py.ylabel('F125W (mag)')
    py.xlabel('F125W - F160W (mag)')
    py.xlim(0, 2)
    py.ylim(23, 12)
    py.savefig(plot_dir + 'cmd_ir_all_clust' + suffix + '.png')




    ##########
    # F814W - F125W Plot CMD with cluster members in black.
    ##########
    mag1 = d['m_2010_F125W']
    mag2 = d['m_2005_F814W']
    color = mag2 - mag1
    py.clf()
    py.plot(color[clust], mag2[clust], 'k.', ms=2)
    py.ylabel('F814W (mag)')
    py.xlabel('F814W - F125W (mag)')
    py.xlim(1, 7)
    py.ylim(26, 12)
    py.savefig(plot_dir + 'cmd_2_clust' + suffix + '.png')

    # F814W - F125W Plot CMD with everything in black.
    py.clf()
    py.plot(color, mag2, 'k.', ms=2)
    py.ylabel('F814W (mag)')
    py.xlabel('F814W - F125W (mag)')
    py.xlim(1, 7)
    py.ylim(26, 12)
    py.savefig(plot_dir + 'cmd_2_all' + suffix + '.png')
    
    # F814W - F125W Plot CMD with everything in black
    # and cluster members in red. 
    py.clf()
    py.plot(color, mag2, 'k.', ms=2)
    py.plot(color[clust], mag2[clust], 'r.', ms=2)
    py.ylabel('F814W (mag)')
    py.xlabel('F814W - F125W (mag)')
    py.xlim(1, 7)
    py.ylim(26, 12)
    py.savefig(plot_dir + 'cmd_2_all_clust' + suffix + '.png')



        
    
    ##########
    # Plot CMD Hess diagram of all
    ##########
    bins_col = np.arange(2, 10, 0.2)
    bins_mag = np.arange(14, 26, 0.2)
    mag1 = d['m_2013_F160W']
    mag2 = d['m_2005_F814W']
    color = mag2 - mag1
    py.clf()
    py.hist2d(color, mag2, bins=(bins_col, bins_mag),
              cmap=py.cm.gist_heat_r)
    py.ylabel('F814W (mag)')
    py.xlabel('F814W - F160W (mag)')
    py.gca().invert_yaxis()
    py.xlim(2, 10)
    py.savefig(plot_dir + 'cmd_hess_all' + suffix + '.png')

    # Plot CMD Hess diagram of cluster members
    py.clf()
    py.hist2d(color[clust], mag2[clust], bins=(bins_col, bins_mag),
              cmap=py.cm.gist_heat_r)
    py.ylabel('F814W (mag)')
    py.xlabel('F814W - F160W (mag)')
    py.gca().invert_yaxis()
    py.xlim(2, 10)
    py.savefig(plot_dir + 'cmd_hess_clust' + suffix + '.png')

            
            
    ##########
    # IR: Plot CMD Hess diagram of all.
    ##########
    bins_col = np.arange(0, 2, 0.05)
    bins_mag = np.arange(12, 23, 0.2)
    mag1 = d['m_2013_F160W']
    mag2 = d['m_2010_F125W']
    color = mag2 - mag1
    py.clf()
    py.hist2d(color, mag2, bins=(bins_col, bins_mag),
              cmap=py.cm.gist_heat_r)
    py.ylabel('F125W (mag)')
    py.xlabel('F125W - F160W (mag)')
    py.xlim(0, 2)
    py.ylim(23, 12)
    py.savefig(plot_dir + 'cmd_ir_hess_all' + suffix + '.png')

    # IR: Plot CMD Hess diagram of cluster members.
    py.clf()
    py.hist2d(color[clust], mag2[clust], bins=(bins_col, bins_mag),
              cmap=py.cm.gist_heat_r)
    py.ylabel('F125W (mag)')
    py.xlabel('F125W - F160W (mag)')
    py.xlim(0, 2)
    py.ylim(23, 12)
    py.savefig(plot_dir + 'cmd_ir_hess_clust' + suffix + '.png')

    return
        

def plot_color_color(catalog=cat_pclust_pcolor, cl_prob=0.6, usePcolor=True, suffix=''):
    # Read in data table
    d = Table.read(catalog)

    if usePcolor:
        d['Membership'] *= d['Membership_color']

    # Determine which we will call "cluster members"
    clust = np.where(d['Membership'] > cl_prob)[0]
    d_cl = d[clust]

    color1 = d['m_2005_F814W'] - d['m_2010_F125W']
    color2 = d['m_2010_F125W'] - d['m_2013_F160W']

    py.close('all')
    py.figure()    
    py.clf()
    py.plot(color2, color1, 'k.', ms=2)
    py.plot(color2[clust], color1[clust], 'r.', ms=2)
    py.xlim(0, 2.5)
    py.ylim(1, 8)
    py.xlabel('F125W - F160W (mag)')
    py.ylabel('F814W - F125W (mag)')
    py.savefig(plot_dir + 'colcol_all_clust' + suffix + '.png')

    return
    

def calc_color_members():
    """
    Using a proper motion cleaned sample, calculate a
    probability of membership based on color/magnitude
    information.
    """
    d = Table.read(cat_pclust)

    clust = np.where(d['Membership'] > 0.6)[0]
    
    good = np.where((d['Membership'] > 0.6) &
                     (np.isnan(d['m_2013_F160W']) == False) & 
                     (np.isnan(d['m_2010_F125W']) == False) & 
                     (np.isnan(d['m_2005_F814W']) == False))[0]

    # Define and empirical isochrone in F814W vs. F160W
    m160 = d['m_2013_F160W']
    m125 = d['m_2010_F125W']
    m814 = d['m_2005_F814W']
    color1 = m814 - m160
    color2 = m125 - m160

    # We are going to interpolate in the space:
    #    F814W vs. color1
    #    F814W vs. color2
    m814_cl = m814[good]
    color1_cl = color1[good]
    color2_cl = color2[good]
    sdx = np.argsort(m814_cl)
    foo = np.array([m814_cl[sdx], color1_cl[sdx], color2_cl[sdx]])
    # cb, p = interpolate.splprep(foo, s=780, k=3)
    cb, p = interpolate.splprep(foo, s=1150, k=3)

    pts = np.linspace(0, 1, 300)
    m814_fit, c1_fit, c2_fit = interpolate.splev(pts, cb)

    # Convert our model into F125W vs. F125W - F160W.
    m125_fit = m814_fit - c1_fit + c2_fit

    # Define our acceptable areas: F814W vx. F814W - F160W
    optical_neg = np.vstack((c1_fit - 0.6, m814_fit)).T
    optical_pos = np.vstack((c1_fit + 1.0, m814_fit)).T
    optical_verts = np.append(optical_neg, optical_pos[::-1], axis=0)
    optical_path = matplotlib.path.Path(optical_verts)
    optical_patch = matplotlib.patches.PathPatch(optical_path, alpha=0.5,
                                                 facecolor='blue', edgecolor='blue')

    py.close(1)
    py.figure(1)
    py.clf()
    py.plot(color1, m814, 'k.', ms=2)
    py.plot(color1[clust], m814[clust], 'r.', ms=2)
    py.plot(c1_fit, m814_fit, 'b-', linewidth=3)
    py.gca().add_patch(optical_patch)
    py.ylim(26, 18)
    py.xlim(1, 10)
    py.xlabel('F814W - F160W')
    py.ylabel('F814W')
    py.savefig(plot_dir + 'membership_color_optical.png')


    # Define our acceptable areas: F125W vs. F125W - F160W
    infrared_neg = np.vstack((c2_fit - 0.25, m125_fit)).T
    infrared_pos = np.vstack((c2_fit + 0.35, m125_fit)).T
    infrared_verts = np.append(infrared_neg, infrared_pos[::-1], axis=0)
    infrared_path = matplotlib.path.Path(infrared_verts)
    infrared_patch = matplotlib.patches.PathPatch(infrared_path, alpha=0.5,
                                                 facecolor='blue', edgecolor='blue')

    py.figure(1)
    py.clf()
    py.plot(color2, m125, 'k.', ms=2)
    py.plot(color2[clust], m125[clust], 'r.', ms=2)
    py.plot(c2_fit, m125_fit, 'b-', linewidth=3)
    py.gca().add_patch(infrared_patch)
    py.ylim(23, 14)
    py.xlim(0, 2.5)
    py.xlabel('F125W - F160W')
    py.ylabel('F125W')
    py.savefig(plot_dir + 'membership_color_infrared.png')

    ##########
    # Calculate p(color) based on F814W vs. F814W - F160W only.
    ##########
    pcolor = np.zeros(len(m814), dtype=int)
    
    points = np.vstack((color1, m814)).T
    idx = np.where(optical_path.contains_points(points) == True)[0]
    fmt = 'Setting P(color) = 0 for {0} of {1} stars based on color cut.'
    print fmt.format(len(m814) - len(idx), len(m814))

    pcolor[idx] = 1

    d['Membership_color'] = pcolor
    print 'Sum[ P(color) ] = ', d['Membership_color'].sum()
    print 'Sum[ P(color) * P(VPD) ] = ', (d['Membership_color'] * d['Membership']).sum()

    # Finally, make a new catalog with only cluster members
    path_info = os.path.split(cat_pclust)
    outfile = work_dir + path_info[1].replace('.fits', '_Pcolor.fits')
    print 'Writing: ', outfile
    d.write(outfile, format='fits', overwrite=True)

    clust = np.where((d['Membership'] * d['Membership_color']) > 0.6)[0]
    py.clf()
    py.plot(color1[good], m814[good], 'r.', ms=2)
    py.plot(color1[clust], m814[clust], 'k.', ms=2)
    py.gca().add_patch(optical_patch)
    py.ylim(26, 18)
    py.xlim(1, 10)
    py.xlabel('F814W - F160W')
    py.ylabel('F814W')

    return


def play_cmd_isochrone_red(logAge=wd1_logAge, AKs=wd1_AKs,
                                        distance=wd1_distance):
    d = Table.read(cat_pclust_pcolor)

    pmem = d['Membership'] * d['Membership_color']
    m160 = d['m_2013_F160W']
    m125 = d['m_2010_F125W']
    m139 = d['m_2010_F139M']
    m814 = d['m_2005_F814W']
    color1 = m814 - m160
    color2 = m125 - m160

    clust = np.where(pmem > 0.8)[0]

    # Load up the original IAU isochrone. We'll compare everything to this
    iso = load_isochrone(logAge=logAge, AKs=AKs, distance=distance, IAU=True)
    
    # Original Reddening Law (of isochrone we are loading)
    wave_0 = np.array([0.551, 0.814, 1.25, 1.63, 2.14, 3.545, 4.442, 5.675, 7.760])
    A_AKs_0 = np.array([16.13, 9.4, 2.82, 1.73, 1.00, 0.500, 0.390, 0.360, 0.430]) #IAU
    A_int_0 = interpolate.splrep(wave_0, A_AKs_0, k=3, s=0)
    
    # New Reddening Law
    wave_1 = np.array( [0.551, 0.814, 1.25, 1.63, 2.14, 3.545, 4.442, 5.675, 7.760])
    A_AKs_1 = np.array([16.13, 9.45, 2.80, 1.70, 1.00, 0.500, 0.390, 0.360, 0.430])
    A_int_1 = interpolate.splrep(wave_1, A_AKs_1, k=3, s=0)

    wave_obs = [0.814, 1.25, 1.39, 1.60]
    A_0 = interpolate.splev(wave_obs, A_int_0)
    A_1 = interpolate.splev(wave_obs, A_int_1)
    print A_0
    print A_1

    dA_F814W = A_1[0] - A_0[0]
    dA_F125W = A_1[1] - A_0[1]
    dA_F139M = A_1[2] - A_0[2]
    dA_F160W = A_1[3] - A_0[3]
    
    # Calculate a reddening vector
    filt_F814W = synthetic.get_filter_info('acs,wfc1,f814w')
    filt_F125W = synthetic.get_filter_info('wfc3,ir,f125w')
    filt_F139M = synthetic.get_filter_info('wfc3,ir,f139m')
    filt_F160W = synthetic.get_filter_info('wfc3,ir,f160w')

    AKs_0 = 0
    AKs_1 = 0.1
    red_F814W_0 = synthetic.redlaw.reddening(AKs_0).resample(filt_F814W.wave)
    red_F125W_0 = synthetic.redlaw.reddening(AKs_0).resample(filt_F125W.wave)
    red_F139M_0 = synthetic.redlaw.reddening(AKs_0).resample(filt_F139M.wave)
    red_F160W_0 = synthetic.redlaw.reddening(AKs_0).resample(filt_F160W.wave)

    red_F814W_1 = synthetic.redlaw.reddening(AKs_1).resample(filt_F814W.wave)
    red_F125W_1 = synthetic.redlaw.reddening(AKs_1).resample(filt_F125W.wave)
    red_F139M_1 = synthetic.redlaw.reddening(AKs_1).resample(filt_F139M.wave)
    red_F160W_1 = synthetic.redlaw.reddening(AKs_1).resample(filt_F160W.wave)
    
    m_F814W_AKs_0 = synthetic.mag_in_filter(synthetic.vega, filt_F814W, red_F814W_0)
    m_F125W_AKs_0 = synthetic.mag_in_filter(synthetic.vega, filt_F125W, red_F125W_0)
    m_F139M_AKs_0 = synthetic.mag_in_filter(synthetic.vega, filt_F139M, red_F139M_0)
    m_F160W_AKs_0 = synthetic.mag_in_filter(synthetic.vega, filt_F160W, red_F160W_0)

    m_F814W_AKs_1 = synthetic.mag_in_filter(synthetic.vega, filt_F814W, red_F814W_1)
    m_F125W_AKs_1 = synthetic.mag_in_filter(synthetic.vega, filt_F125W, red_F125W_1)
    m_F139M_AKs_1 = synthetic.mag_in_filter(synthetic.vega, filt_F139M, red_F139M_1)
    m_F160W_AKs_1 = synthetic.mag_in_filter(synthetic.vega, filt_F160W, red_F160W_1)


    py.close(1)
    py.figure(1, figsize=(10,10))
    # py.clf()
    py.subplots_adjust(left=0.08, bottom=0.08, right=0.95, top=0.93, wspace=0.25, hspace=0.25)
    
    # F814W vs. F160W CMD
    py.subplot(2, 2, 1)
    py.plot(color1[clust], m814[clust], 'k.', ms=2)
    py.plot(iso['mag814w'] - iso['mag160w'], iso['mag814w'], 'r.', ms=5)
    py.plot(iso['mag814w'] - iso['mag160w'] + dA_F814W - dA_F160W, 
            iso['mag814w'] + dA_F814W, 'g.', ms=5)
    py.ylim(26, 14)
    py.xlim(3, 7)
    py.xlabel('F814W - F160W')
    py.ylabel('F814W')

    red_vec_dx = (m_F814W_AKs_1 - m_F814W_AKs_0) - (m_F160W_AKs_1 - m_F160W_AKs_0)
    red_vec_dy = (m_F814W_AKs_1 - m_F814W_AKs_0)
    py.arrow(3.5, 22, red_vec_dx, red_vec_dy, head_width=0.2)

    # F814W vs. F125W CMD
    py.subplot(2, 2, 2)
    py.plot(m814[clust] - m125[clust], m814[clust], 'k.', ms=2)
    py.plot(iso['mag814w'] - iso['mag125w'], iso['mag814w'], 'r.', ms=5)
    py.plot(iso['mag814w'] - iso['mag125w'] + dA_F814W - dA_F125W, 
            iso['mag814w'] + dA_F814W, 'g.', ms=5)
    py.ylim(26, 14)
    py.xlim(2.5, 5.5)
    py.xlabel('F814W - F125W')
    py.ylabel('F814W')
    
    red_vec_dx = (m_F814W_AKs_1 - m_F814W_AKs_0) - (m_F125W_AKs_1 - m_F125W_AKs_0)
    red_vec_dy = (m_F814W_AKs_1 - m_F814W_AKs_0)
    py.arrow(3, 22, red_vec_dx, red_vec_dy, head_width=0.18)
    py.show()

    # # F125W vs. F139M CMD
    # py.subplot(2, 2, 2)
    # py.plot(m125[clust] - m139[clust], m125[clust], 'k.', ms=2)
    # py.plot(iso['mag125w'] - iso['mag139m'], iso['mag125w'], 'r.', ms=5)
    # py.plot(iso['mag125w'] - iso['mag139m'] + dA_F125W - dA_F139M, 
    #         iso['mag125w'] + dA_F125W, 'g.', ms=5)
    # py.ylim(22, 11)
    # py.xlim(0.0, 1.0)
    # py.xlabel('F125W - F139M')
    # py.ylabel('F125W')
    
    # red_vec_dx = (m_F125W_AKs_1 - m_F125W_AKs_0) - (m_F139M_AKs_1 - m_F139M_AKs_0)
    # red_vec_dy = (m_F125W_AKs_1 - m_F125W_AKs_0)
    # py.arrow(3, 22, red_vec_dx, red_vec_dy, head_width=0.18)
    
    # F125W vs. F160W CMD
    py.subplot(2, 2, 3)
    py.plot(color2[clust], m125[clust], 'k.', ms=2)
    py.plot(iso['mag125w'] - iso['mag160w'], iso['mag125w'], 'r.', ms=5)
    py.plot(iso['mag125w'] - iso['mag160w'] + dA_F125W - dA_F160W, 
            iso['mag125w'] + dA_F125W, 'g.', ms=5)
    py.ylim(21.5, 12)
    py.xlim(0.5, 1.7)
    py.xlabel('F125W - F160W')
    py.ylabel('F125W')
    
    red_vec_dx = (m_F125W_AKs_1 - m_F125W_AKs_0) - (m_F160W_AKs_1 - m_F160W_AKs_0)
    red_vec_dy = (m_F125W_AKs_1 - m_F125W_AKs_0)
    py.arrow(0.6, 18, red_vec_dx, red_vec_dy, head_width=0.1)

    print 'Slope of IR redvector: {0}'.format( red_vec_dy / red_vec_dx )
    print 'Length of IR redvector: {0}'.format(np.hypot(red_vec_dy, red_vec_dx))

    
    # Color-color
    py.subplot(2, 2, 4)
    py.plot(color2[clust], color1[clust], 'k.', ms=2)
    py.plot(iso['mag125w'] - iso['mag160w'], iso['mag814w'] - iso['mag160w'],
            'r.', ms=5)
    py.plot(iso['mag125w'] - iso['mag160w'] + dA_F125W - dA_F160W, 
            iso['mag814w'] - iso['mag160w'] + dA_F814W - dA_F160W,
            'g.', ms=5)
    py.ylim(3.0, 7)
    py.xlim(0.5, 1.7)
    py.xlabel('F125W - F160W')
    py.ylabel('F814W - F160W')

    red_vec_dx = (m_F125W_AKs_1 - m_F125W_AKs_0) - (m_F160W_AKs_1 - m_F160W_AKs_0)
    red_vec_dy = (m_F814W_AKs_1 - m_F814W_AKs_0) - (m_F160W_AKs_1 - m_F160W_AKs_0)
    py.arrow(0.6, 5.5, red_vec_dx, red_vec_dy, head_width=0.05)
    

    py.suptitle('log(t)={0:4.2f}, AKs={1:4.2f}, d={2:4.0f}'.format(logAge, AKs, distance),
                verticalalignment='top')


    outfile = 'cmd_play_isochrones_t{0:4.2f}_AKs{1:4.2f}_d{2:4.0f}.png'.format(logAge, AKs, distance)
    py.savefig(plot_dir + outfile)
    return

def plot_cmd_isochrone(logAge=wd1_logAge, AKs=wd1_AKs,
                                     distance=wd1_distance):
    d = Table.read(cat_pclust_pcolor)

    pmem = d['Membership'] * d['Membership_color']
    m160 = d['m_2013_F160W']
    m125 = d['m_2010_F125W']
    m139 = d['m_2010_F139M']
    m814 = d['m_2005_F814W']
    color1 = m814 - m160
    color2 = m125 - m160

    clust = np.where(pmem > 0.8)[0]

    # Load Wd1 isochrone
    synthetic.redlaw = reddening.RedLawWesterlund1()
    iso = load_isochrone(logAge=logAge, AKs=AKs, distance=distance)
    iso = iso[::3]
    
    # Calculate a reddening vector based on vega
    filt_F814W = synthetic.get_filter_info('acs,wfc1,f814w')
    filt_F125W = synthetic.get_filter_info('wfc3,ir,f125w')
    filt_F139M = synthetic.get_filter_info('wfc3,ir,f139m')
    filt_F160W = synthetic.get_filter_info('wfc3,ir,f160w')

    AKs_0 = 0
    AKs_1 = 0.1
    red_F814W_0 = synthetic.redlaw.reddening(AKs_0).resample(filt_F814W.wave)
    red_F125W_0 = synthetic.redlaw.reddening(AKs_0).resample(filt_F125W.wave)
    red_F139M_0 = synthetic.redlaw.reddening(AKs_0).resample(filt_F139M.wave)
    red_F160W_0 = synthetic.redlaw.reddening(AKs_0).resample(filt_F160W.wave)

    red_F814W_1 = synthetic.redlaw.reddening(AKs_1).resample(filt_F814W.wave)
    red_F125W_1 = synthetic.redlaw.reddening(AKs_1).resample(filt_F125W.wave)
    red_F139M_1 = synthetic.redlaw.reddening(AKs_1).resample(filt_F139M.wave)
    red_F160W_1 = synthetic.redlaw.reddening(AKs_1).resample(filt_F160W.wave)

    #-------------------------------------------------------------------#
    # Need to apply reddening to vega before passing into mag_in_filter
    #-------------------------------------------------------------------#
    star = synthetic.vega
    
    red_0 = synthetic.redlaw.reddening(AKs_0).resample(star.wave) 
    red_star_0 = star * red_0
    red_1 = synthetic.redlaw.reddening(AKs_1).resample(star.wave)
    red_star_1 = star * red_1
    
    m_F814W_AKs_0 = synthetic.mag_in_filter(red_star_0, filt_F814W)
    m_F125W_AKs_0 = synthetic.mag_in_filter(red_star_0, filt_F125W)
    m_F139M_AKs_0 = synthetic.mag_in_filter(red_star_0, filt_F139M)
    m_F160W_AKs_0 = synthetic.mag_in_filter(red_star_0, filt_F160W)

    m_F814W_AKs_1 = synthetic.mag_in_filter(red_star_1, filt_F814W)
    m_F125W_AKs_1 = synthetic.mag_in_filter(red_star_1, filt_F125W)
    m_F139M_AKs_1 = synthetic.mag_in_filter(red_star_1, filt_F139M)
    m_F160W_AKs_1 = synthetic.mag_in_filter(red_star_1, filt_F160W)

    # Get a couple of key masses for overplotting
    iso_mass = np.array([0.5, 1.0, 2.0, 5.0, 10.0])
    iso_idx = np.zeros(len(iso_mass), dtype=int)
    for ii in range(len(iso_mass)):
        dmass = np.abs(iso_mass[ii] - iso['mass'])
        iso_idx[ii] = dmass.argmin()

    # Also pull a Nishiyama+09 isochrone, for comparison
    synthetic.redlaw = reddening.RedLawNishiyama09()
    iso_N09 = load_isochrone(logAge=logAge, AKs=AKs, distance=distance)

    synthetic.redlaw = reddening.RedLawWesterlund1()
    
    py.close(1)
    py.figure(1, figsize=(10,10))
    # py.clf()
    py.subplots_adjust(left=0.08, bottom=0.08, right=0.95, top=0.93, wspace=0.25, hspace=0.25)
    
    # F814W vs. F160W CMD
    py.subplot(2, 2, 1)
    py.plot(color1[clust], m814[clust], 'k.', ms=2)
    py.plot(iso['mag814w'] - iso['mag160w'], iso['mag814w'], 'r-', linewidth=3,
            label='Wd1')
    py.plot(iso_N09['mag814w'] - iso_N09['mag160w'], iso_N09['mag814w'], 'k-',
            linewidth=3, label='Nishiyama+09')
    py.ylim(26, 14)
    py.xlim(3, 7)
    py.legend(numpoints=1)
    py.xlabel('F814W - F160W (mag)')
    py.ylabel('F814W (mag)')

    red_vec_dx = (m_F814W_AKs_1 - m_F814W_AKs_0) - (m_F160W_AKs_1 - m_F160W_AKs_0)
    red_vec_dy = (m_F814W_AKs_1 - m_F814W_AKs_0)
    py.arrow(3.8, 21, red_vec_dx, red_vec_dy, head_width=0.2)

    x_mass = iso['mag814w'][iso_idx] - iso['mag160w'][iso_idx]
    y_mass = iso['mag814w'][iso_idx]
    #py.plot(x_mass, y_mass, 'rs', ms=10)
    #for ii in range(len(iso_mass)):
    #    py.text(x_mass[ii] + 0.8, y_mass[ii], r'{0:.1f} M$_\odot$'.format(iso_mass[ii]), color='black')

    # F814W vs. F125W CMD
    py.subplot(2, 2, 2)
    py.plot(m814[clust] - m125[clust], m814[clust], 'k.', ms=2)
    py.plot(iso['mag814w'] - iso['mag125w'], iso['mag814w'], 'r-', linewidth=3,
            label='Wd1')
    py.plot(iso_N09['mag814w'] - iso_N09['mag125w'], iso_N09['mag814w'],
            'k-', linewidth=3, label='Nishiyama+09')
    py.ylim(26, 14)
    py.xlim(2.5, 5.5)
    py.legend(numpoints=1)
    py.xlabel('F814W - F125W (mag)')
    py.ylabel('F814W (mag)')
    
    red_vec_dx = (m_F814W_AKs_1 - m_F814W_AKs_0) - (m_F125W_AKs_1 - m_F125W_AKs_0)
    red_vec_dy = (m_F814W_AKs_1 - m_F814W_AKs_0)
    py.arrow(2.7, 21, red_vec_dx, red_vec_dy, head_width=0.18)

    x_mass = iso['mag814w'][iso_idx] - iso['mag125w'][iso_idx]
    y_mass = iso['mag814w'][iso_idx]
    #py.plot(x_mass, y_mass, 'rs', ms=10)
    #for ii in range(len(iso_mass)):
    #    py.text(x_mass[ii] + 0.6, y_mass[ii], r'{0:.1f} M$_\odot$'.format(iso_mass[ii]))

    # # F125W vs. F139M CMD
    # py.subplot(2, 2, 2)
    # py.plot(m125[clust] - m139[clust], m125[clust], 'k.', ms=2)
    # py.plot(iso['mag125w'] - iso['mag139m'], iso['mag125w'], 'r.', ms=5)
    # py.plot(iso['mag125w'] - iso['mag139m'] + dA_F125W - dA_F139M, 
    #         iso['mag125w'] + dA_F125W, 'g.', ms=5)
    # py.ylim(22, 11)
    # py.xlim(0.0, 1.0)
    # py.xlabel('F125W - F139M')
    # py.ylabel('F125W')
    
    # red_vec_dx = (m_F125W_AKs_1 - m_F125W_AKs_0) - (m_F139M_AKs_1 - m_F139M_AKs_0)
    # red_vec_dy = (m_F125W_AKs_1 - m_F125W_AKs_0)
    # py.arrow(3, 22, red_vec_dx, red_vec_dy, head_width=0.18)
    
    # F125W vs. F160W CMD
    py.subplot(2, 2, 3)
    py.plot(color2[clust], m125[clust], 'k.', ms=2)
    py.plot(iso['mag125w'] - iso['mag160w'], iso['mag125w'], 'r-',
            label='Wd1', linewidth=3)
    py.plot(iso_N09['mag125w'] - iso_N09['mag160w'], iso_N09['mag125w'], 'k-',
            label='Nishiyama+09', linewidth=3)
    py.ylim(21.5, 12)
    py.xlim(0.5, 1.7)
    py.legend(numpoints=1)
    py.xlabel('F125W - F160W (mag)')
    py.ylabel('F125W (mag)')
    
    red_vec_dx = (m_F125W_AKs_1 - m_F125W_AKs_0) - (m_F160W_AKs_1 - m_F160W_AKs_0)
    red_vec_dy = (m_F125W_AKs_1 - m_F125W_AKs_0)
    py.arrow(0.72, 18, red_vec_dx, red_vec_dy, head_width=0.1)

    x_mass = iso['mag125w'][iso_idx] - iso['mag160w'][iso_idx]
    y_mass = iso['mag125w'][iso_idx]
    #py.plot(x_mass, y_mass, 'rs', ms=10)
    #for ii in range(len(iso_mass)):
    #    py.text(x_mass[ii] + 0.2, y_mass[ii], r'{0:.1f} M$_\odot$'.format(iso_mass[ii]))
    
    # Color-color
    py.subplot(2, 2, 4)
    py.plot(color2[clust], color1[clust], 'k.', ms=2)
    py.plot(iso['mag125w'] - iso['mag160w'], iso['mag814w'] - iso['mag160w'],
            'r-', linewidth=3, label='Wd1')
    py.plot(iso_N09['mag125w'] - iso_N09['mag160w'],
            iso_N09['mag814w'] - iso_N09['mag160w'],
            'k-', linewidth=3, label='Nishiyama+09')
    py.ylim(3.0, 7)
    py.xlim(0.5, 1.7)
    py.legend(numpoints=1)
    py.xlabel('F125W - F160W (mag)')
    py.ylabel('F814W - F160W (mag)')

    red_vec_dx = (m_F125W_AKs_1 - m_F125W_AKs_0) - (m_F160W_AKs_1 - m_F160W_AKs_0)
    red_vec_dy = (m_F814W_AKs_1 - m_F814W_AKs_0) - (m_F160W_AKs_1 - m_F160W_AKs_0)
    py.arrow(0.65, 4.0, red_vec_dx, red_vec_dy, head_width=0.05)
    
    x_mass = iso['mag125w'][iso_idx] - iso['mag160w'][iso_idx]
    y_mass = iso['mag814w'][iso_idx] - iso['mag160w'][iso_idx]
    #py.plot(x_mass, y_mass, 'rs', ms=10)
    #for ii in range(len(iso_mass)):
    #    py.text(x_mass[ii], y_mass[ii]-0.5, r'{0:.1f} M$_\odot$'.format(iso_mass[ii]))

    #py.suptitle('log(t)={0:4.2f}, AKs={1:4.2f}, d={2:4.0f}'.format(logAge, AKs, distance),
    #            verticalalignment='top')


    outfile = 'cmd_isochrones_t{0:4.2f}_AKs{1:4.2f}_d{2:4.0f}_{3}.png'.format(logAge, AKs,
                                                                              distance,
                                                                              synthetic.redlaw.name)
    py.savefig(plot_dir + outfile)

    #py.close('all')
    
    return

def compare_art_vs_obs_vel(use_obs_align=False):
    """
    Compare the distribution of velocities in the artificial
    vs. the observed stars. These need to match if we are going
    to make a cut on the velocitiy error. 
    """
    art_catalog = art_dir + 'wd1_art_catalog_RMSE_wvelErr'
    if use_obs_align:
        art_catalog += '_aln_obs'
    else:
        art_catalog += '_aln_art'
    art_catalog += '.fits'

    obs = Table.read(cat_dir + catalog)
    art = Table.read(art_catalog)

    # Trim out un-detected in artificial star lists.    
    adx_det = np.where((art['fit_vxe'] != 1) & (art['fit_vye'] != 1))[0]
    art = art[adx_det]

    # Convert artificial star velocities into mas / yr.
    scale = astrometry.scale['WFC'] * 1e3
    art['fit_vx'] *= scale
    art['fit_vy'] *= scale
    art['fit_vxe'] *= scale
    art['fit_vye'] *= scale

    obs['fit_vx'] *= scale
    obs['fit_vy'] *= scale
    obs['fit_vxe'] *= scale
    obs['fit_vye'] *= scale

    ##########
    # proper motion error vs. F160W
    ##########
    py.close(1)
    py.figure(1, figsize=(6, 10))
    py.clf()
    py.subplot(2, 1, 1)
    py.semilogy(art['m_2013_F160W'], art['fit_vxe'], 'k.', ms=2, alpha=0.2,
                label='Simulated')
    py.semilogy(obs['m_2013_F160W'], obs['fit_vxe'], 'r.', ms=4, alpha=0.5,
                label='Observed')
    py.ylim(1e-2, 10)
    py.xlim(11, 22)
    py.xlabel('F160W 2013 (mag)')
    py.ylabel('X Velocity Error (mas/yr)')


    py.subplot(2, 1, 2)
    py.semilogy(art['m_2013_F160W'], art['fit_vye'], 'k.', ms=2, alpha=0.2,
                label='Simulated')
    py.semilogy(obs['m_2013_F160W'], obs['fit_vye'], 'r.', ms=4, alpha=0.5,
                label='Observed')
    py.ylim(1e-2, 10)
    py.xlim(11, 22)
    py.xlabel('F160W 2013 (mag)')
    py.ylabel('Y Velocity Error (mas/yr)')


    plot_file = plot_dir + 'compare_art_vs_obs_vel_xy_F160W'
    if use_obs_align:
        plot_file += '_obs'
    else:
        plot_file += '_art'
    plot_file += '.png'
    py.savefig(plot_file)


    ##########
    # proper motion error vs. F814W
    ##########
    py.close(2)
    py.figure(2, figsize=(6, 10))
    py.clf()
    py.subplot(2, 1, 1)
    py.semilogy(art['m_2005_F814W'], art['fit_vxe'], 'k.', ms=2, alpha=0.2,
                label='Simulated')
    py.semilogy(obs['m_2005_F814W'], obs['fit_vxe'], 'r.', ms=4, alpha=0.5,
                label='Observed')
    py.ylim(1e-2, 10)
    py.xlim(13, 26)
    py.xlabel('F814W 2005 (mag)')
    py.ylabel('X Velocity Error (mas/yr)')


    py.subplot(2, 1, 2)
    py.semilogy(art['m_2005_F814W'], art['fit_vye'], 'k.', ms=2, alpha=0.2,
                label='Simulated')
    py.semilogy(obs['m_2005_F814W'], obs['fit_vye'], 'r.', ms=4, alpha=0.5,
                label='Observed')
    py.ylim(1e-2, 10)
    py.xlim(13, 26)
    py.xlabel('F814W 2005 (mag)')
    py.ylabel('Y Velocity Error (mas/yr)')
    plot_file = plot_dir + 'compare_art_vs_obs_vel_xy_F814W'
    if use_obs_align:
        plot_file += '_obs'
    else:
        plot_file += '_art'
    plot_file += '.png'
    py.savefig(plot_file)

    mdx_art = np.where(art['m_2013_F160W'] < 17)[0]
    mdx_obs = np.where(obs['m_2013_F160W'] < 17)[0]
    
    med_art_x = np.median(art[mdx_art]['fit_vxe'])
    med_obs_x = np.median(obs[mdx_obs]['fit_vxe'])
    med_art_y = np.median(art[mdx_art]['fit_vye'])
    med_obs_y = np.median(obs[mdx_obs]['fit_vye'])

    print 'Median Velocity (mas/yr)'
    print '   X: Obs = {0:5.3f}  Art = {1:5.3f}'.format(med_obs_x, med_art_x)
    print '   Y: Obs = {0:5.3f}  Art = {1:5.3f}'.format(med_obs_y, med_art_y)

    pdb.set_trace()


def compare_art_vs_obs_cmds(use_obs_align=False, vel_err_cut=0.5, mag_err_cut=1.0):
    """
    vel_err_cut = mas/yr
    mag_err_cut = magnitudes 
    """
    art = Table.read(art_cat)
    obs = Table.read(work_dir + 'catalog_membership_3_rot_Pcolor.fits')

    # Get the "detected" stars that are within our error cuts and
    # detected in all three astrometric epochs.
    idx = np.where((art['det_2005_F814W'] == True) &
                   (art['det_2010_F125W'] == True) &
                   (art['det_2010_F160W'] == True) &
                   (art['det_2013_F160W'] == True) &
                   (art['fit_vxe'] < vel_err_cut) &
                   (art['fit_vye'] < vel_err_cut) &
                   (art['me_2005_F814W'] < mag_err_cut) &
                   (art['me_2010_F125W'] < mag_err_cut) &
                   (art['me_2010_F160W'] < mag_err_cut) &
                   (art['me_2013_F160W'] < mag_err_cut))[0]

    art_mag = art['min_2010_F160W']
    art_col1 = art['min_2005_F814W'] - art['min_2010_F160W']
    art_col2 = art['min_2010_F125W'] - art['min_2010_F160W']
    
    det_mag = art_mag[idx]
    det_col1 = art_col1[idx]
    det_col2 = art_col2[idx]
    
    obs_mag = obs['m_2010_F160W']
    obs_col1 = obs['m_2005_F814W'] - obs['m_2010_F160W']
    obs_col2 = obs['m_2010_F125W'] - obs['m_2010_F160W']
    
    py.figure(1)
    py.clf()
    py.plot(det_col1, det_mag, 'k.', alpha=0.2, ms=2)
    py.plot(obs_col1, obs_mag, 'r.')
    py.ylim(21.5, 9.5)
    py.xlim(2.5, 7.5)
    py.xlabel('F814W - F160W (mag)')
    py.ylabel('F160W (mag)')
    
    plot_file = plot_dir + 'compare_art_vs_obs_cmds_F160W_F814W'
    if use_obs_align:
        plot_file += '_obs'
    else:
        plot_file += '_art'
    plot_file += '.png'
    py.savefig(plot_file)

    py.figure(2)
    py.clf()
    py.plot(art_col2, art_mag, 'k.', alpha=0.2, ms=2)
    py.plot(obs_col2, obs_mag, 'r.')
    py.ylim(21.5, 9.5)
    py.xlim(0.2, 1.8)
    py.xlabel('F125W - F160W (mag)')
    py.ylabel('F160W (mag)')

    plot_file = plot_dir + 'compare_art_vs_obs_cmds_F160W_F125W'
    if use_obs_align:
        plot_file += '_obs'
    else:
        plot_file += '_art'
    plot_file += '.png'
    py.savefig(plot_file)
    
    return

    
def make_completeness_table(vel_err_cut=0.5, mag_err_cut=1.0):
    """
    vel_err_cut = mas/yr
    mag_err_cut = magnitudes 
    """
    art = Table.read(art_cat)

    # Make a completeness curve for each filter independently.
    epochs = ['2005_F814W', '2010_F125W', '2010_F139M', '2010_F160W', '2013_F160W']
    
    mag_bins = np.arange(10, 27.01, 0.25)
    mag_bins_left = mag_bins[:-1]
    mag_bins_right = mag_bins[1:]
    mag_bins_mid = (mag_bins_left + mag_bins_right) / 2.0

    comp = Table([mag_bins_left, mag_bins_right, mag_bins_mid],
                 names=['mag_left', 'mag_right', 'mag'],
                  meta={'name': 'Completeness Table'})

    for ee in range(len(epochs)):
        ep = epochs[ee]
        idx = np.where(art['det_' + ep] == True)[0]

        # Make some cuts based on the vxe (only X ... something is still wrong in Y).
        tmp = art[idx]
        vdx = np.where((tmp['fit_vxe'] < vel_err_cut) & (tmp['me_' + ep] < mag_err_cut))[0]

        print 'Cut {0:d} based on velocity error cuts'.format(len(idx) - len(vdx))
        
        det = tmp[vdx]
        
        n_all, b_all = np.histogram(art['min_' + ep], bins=mag_bins)
        n_det, b_det = np.histogram(det['min_' + ep], bins=mag_bins)

        c = (1.0 * n_det) / n_all
        ce = c * np.sqrt(1.0 * n_det) / n_det
        
        # Fix some issues at the bright end (where number of sources is small).
        # For later good behavior in interpolation, set it to
        # the first "good" value. But set errors to np.nan to notify.
        #     m=20 is arbitrary but catches both F814W and IR
        idx = np.where((mag_bins_mid < 20) & ((ce > 0.1) | (n_all < 10)))[0]
        if len(idx) > 0:
            c[idx] = c[idx[-1] + 1]
            ce[idx] = np.nan

        # Do the same for the faint end; but set to 0.
        idx = np.where((mag_bins_mid >= 20) & ((ce > 0.1) | (n_all < 10)))[0]
        if len(idx) > 0:
            c[idx] = 0.0
            ce[idx] = np.nan
        
        col_c = Column(c, name='c_'+ep)
        col_ce = Column(ce, name='cerr_'+ep)
        col_N = Column(n_all, name='Nplant_'+ep)

        comp.add_columns([col_c, col_ce, col_N], copy=False)
        
        
    # Plot
    py.close('all')
    py.figure()
    py.clf()
    for ee in range(len(epochs)):
        ep = epochs[ee]
        py.errorbar(comp['mag'], comp['c_' + ep], yerr=comp['cerr_' + ep],
                    label=ep, drawstyle='steps-mid')
        # py.plot(mag_bins_mid, comp[ep], 
        #             label=ep, drawstyle='steps-mid')
        
    py.ylim(0, 1)
    py.legend(loc='lower left')
    py.xlabel('Magnitude (mag)')
    py.ylabel('Completeness')
    py.savefig(plot_dir + 'completeness_vs_mag.png')

    comp.write(work_dir + 'completeness_vs_mag.fits', overwrite=True)
                
    return
        
def calc_mass_function(logAge=wd1_logAge, AKs=wd1_AKs, distance=wd1_distance):
    # Read in data table
    print 'Reading data table'
    d = Table.read(cat_pclust_pcolor)

    # Trim down to cluster stars.
    pmem = d['Membership'] * d['Membership_color']
    m125 = d['m_2010_F125W']
    good = np.where((pmem > 0.8) & (m125 > 0))[0]
    d = d[good]

    # Get variables for just cluster members.
    pmem = d['Membership'] * d['Membership_color']
    m125 = d['m_2010_F125W']
    m160 = d['m_2013_F160W']
    m814 = d['m_2005_F814W']
    color1 = m814 - m160
    color2 = m125 - m160
    
    # Read in isochrone
    print 'Loading Isochrone'
    #----FOR NISHIYAMA+09----#
    #synthetic.redlaw = reddening.RedLawNishiyama09()
    #------------------------#
    red_dAKs = 0.1
    iso = load_isochrone(logAge=logAge, AKs=AKs, distance=distance)
    iso_red = load_isochrone(logAge=logAge, AKs=AKs+red_dAKs, distance=distance)

    # Get the completeness (relevant for diff. de-reddened magnitudes).
    print 'Loading completeness table'
    comp = Table.read(work_dir + 'completeness_vs_mag.fits')

    # Make a finely sampled mass-luminosity relationship by
    # interpolating on the isochrone.
    print 'Setting up mass-luminosity interpolater'
    iso_mag1 = iso['mag814w']
    iso_mag2 = iso['mag125w']
    iso_col1 = iso['mag814w'] - iso['mag160w']
    iso_col2 = iso['mag125w'] - iso['mag160w']
    iso_mass = iso['mass']
    iso_WR = iso['isWR']

    # Remove duplicate values, if they occur
    bad = np.where( np.diff(iso_mass) == 0)
    iso_mass = np.delete(iso_mass, bad)
    iso_mag1 = np.delete(iso_mag1, bad)
    iso_col1 = np.delete(iso_col1, bad)
    iso_mag2 = np.delete(iso_mag2, bad)
    iso_col2 = np.delete(iso_col2, bad)
    iso_WR = np.delete(iso_WR, bad)
    
    iso_tck1, iso_u1 = interpolate.splprep([iso_mass, iso_mag1, iso_col1], s=0.01)
    iso_tck2, iso_u2 = interpolate.splprep([iso_mass, iso_mag2, iso_col2], s=0.01)

    # Same for the reddening vector isochrone
    iso_red_mag1 = iso_red['mag814w']
    iso_red_mag2 = iso_red['mag125w']
    iso_red_col1 = iso_red['mag814w'] - iso_red['mag160w']
    iso_red_col2 = iso_red['mag125w'] - iso_red['mag160w']
    iso_red_mass = iso_red['mass']
    iso_red_WR = iso_red['isWR']

    # Remove duplicate values, if they occur
    bad = np.where( np.diff(iso_red_mass) == 0)
    iso_red_mass = np.delete(iso_red_mass, bad)
    iso_red_mag1 = np.delete(iso_red_mag1, bad)
    iso_red_col1 = np.delete(iso_red_col1, bad)
    iso_red_mag2 = np.delete(iso_red_mag2, bad)
    iso_red_col2 = np.delete(iso_red_col2, bad)
    iso_red_WR = np.delete(iso_red_WR, bad)
    
    iso_red_tck1, iso_red_u1 = interpolate.splprep([iso_red_mass, iso_red_mag1, iso_red_col1], s=0.01)
    iso_red_tck2, iso_red_u2 = interpolate.splprep([iso_red_mass, iso_red_mag2, iso_red_col2], s=0.01)

    # Find the maximum mass that is NOT a WR star
    mass_max = iso_mass[iso_WR == False].max()

    u_fine = np.linspace(0, 1, 1e4)
    iso_mass_f1, iso_mag_f1, iso_col_f1 = interpolate.splev(u_fine, iso_tck1)
    iso_mass_f2, iso_mag_f2, iso_col_f2 = interpolate.splev(u_fine, iso_tck2)

    iso_red_mass_f1, iso_red_mag_f1, iso_red_col_f1 = interpolate.splev(u_fine, iso_red_tck1)
    iso_red_mass_f2, iso_red_mag_f2, iso_red_col_f2 = interpolate.splev(u_fine, iso_red_tck2)

    # Test interpolation, if desired
    testInterp=False
    if testInterp:
       py.figure(1, figsize=(10,10))
       py.clf()
       py.plot(iso_col1, iso_mag1, 'k.', ms=8)
       py.plot(iso_col_f1, iso_mag_f1, 'r-', linewidth=2)
       py.xlabel('F814W - F160W')
       py.ylabel('F814W')
       py.title('Optical Interpolation')
       py.axis([3, 7, 26, 14])
       py.savefig('interp_test_opt.png')

       py.figure(2, figsize=(10,10))
       py.clf()
       py.plot(iso_col2, iso_mag2, 'k.', ms=8)
       py.plot(iso_col_f2, iso_mag_f2, 'r-', linewidth=2)
       py.xlabel('F125W - F160W')
       py.ylabel('F125W')
       py.title('IR Interpolation')
       py.axis([0.5, 1.7, 21, 12])       
       py.savefig('interp_test_ir.png')

       pdb.set_trace()  

    red_vec_dx_f1 = iso_red_col_f1 - iso_col_f1
    red_vec_dy_f1 = iso_red_mag_f1 - iso_mag_f1
    red_vec_dx_f2 = iso_red_col_f2 - iso_col_f2
    red_vec_dy_f2 = iso_red_mag_f2 - iso_mag_f2
        
    # Define WR stars. 
    iso_WR_f1 = np.zeros(len(iso_mass_f1), dtype=bool)
    iso_WR_f2 = np.zeros(len(iso_mass_f2), dtype=bool)
    iso_WR_f1[iso_mass_f1 > mass_max] = True
    iso_WR_f2[iso_mass_f2 > mass_max] = True

    # If desired, look at reddening vector slope as a function
    # of mass
    test = False
    if test:
        # Calculate dx, dy on the same mass scale
        vec_dx_f1 = iso_red_col1 - iso_col1
        vec_dy_f1 = iso_red_mag1 - iso_mag1
        vec_dx_f2 = iso_red_col2 - iso_col2
        vec_dy_f2 = iso_red_mag2 - iso_mag2

        # Calculate slopes dy / dx
        opt_slope = vec_dy_f1 / vec_dx_f1
        ir_slope = vec_dy_f2 / vec_dx_f2

        # Caculate length of the vector
        opt_length = np.hypot(vec_dy_f1, vec_dx_f1)
        ir_length = np.hypot(vec_dy_f2, vec_dx_f2)
        
        # Rough determination of MS turn on, from color. This
        # is specific for Age = 6.7, Aks = 0.69, dist = 4000 iso
        PMS_opt = np.where( iso_col1 > 4.0 )
        PMS_ir = np.where( iso_col2 > 0.8 )
        
        py.close('all')
        py.figure(1, figsize=(20,10))
        py.subplot(121)
        py.subplots_adjust(left=0.1)
        py.plot(iso_mass, opt_slope, 'k.', ms=5, label='MS')
        py.plot(iso_mass[PMS_opt], opt_slope[PMS_opt], 'r.', ms=5,
                label='PMS')
        py.plot(iso_mass[iso_WR], opt_slope[iso_WR], 'b.', ms=5,
                label='WR')
        py.xlabel('Mass (M_sun)')
        py.ylabel('Reddening vector slope: Optical')
        py.legend()
        py.subplot(122)
        py.plot(iso_mass, ir_slope, 'k.', ms=5, label='MS')
        py.plot(iso_mass[PMS_ir], ir_slope[PMS_ir], 'r.', ms=5,
                label='PMS')
        py.plot(iso_mass[iso_WR], ir_slope[iso_WR], 'b.', ms=5,
                label='WR')
        py.xlabel('Mass (M_sun)')
        py.ylabel('Reddening vector slope: IR')
        py.legend()
        py.savefig('Redvector_slope.png')

        # Conparing the lengths of the reddening vectors
        py.figure(2, figsize=(20,10))
        py.subplot(121)
        py.subplots_adjust(left=0.1)
        py.plot(iso_mass, opt_length, 'k.', ms=5, label='MS')
        py.plot(iso_mass[PMS_opt], opt_length[PMS_opt], 'r.', ms=5,
                label='PMS')
        py.plot(iso_mass[iso_WR], opt_length[iso_WR], 'b.', ms=5,
                label='WR')
        py.xlabel('Mass (M_sun)')
        py.ylabel('Length of Optical Reddening Vector (mag)')
        py.legend()
        py.subplot(122)
        py.plot(iso_mass, ir_length, 'k.', ms=5, label='MS')
        py.plot(iso_mass[PMS_ir], ir_length[PMS_ir], 'r.', ms=5,
                label='PMS')
        py.plot(iso_mass[iso_WR], ir_length[iso_WR], 'b.', ms=5,
                label='WR')
        py.xlabel('Mass (M_sun)')
        py.ylabel('Length of IR Reddening Vector (mag)')
        py.legend()
        py.savefig('Redvector_length.png')        
    
        pdb.set_trace()

    print 'Setting up completeness interpolater for CMD space.'
    comp1_int = comp_interp_for_cmd(comp['mag'], comp['c_2005_F814W'],
                                    comp['c_2010_F160W'], 'F814W', 'F160W')

    comp2_int = comp_interp_for_cmd(comp['mag'], comp['c_2010_F125W'],
                                    comp['c_2010_F160W'], 'F125W', 'F160W')

    
    mass1, isWR1, comp1, AKs1 = calc_mass_isWR_comp(m814, color1,
                                                    iso_mag_f1, iso_col_f1,
                                                    iso_mass_f1, iso_WR_f1,
                                                    comp1_int, mass_max,
                                                    red_vec_dx_f1, red_vec_dy_f1, red_dAKs,
                                                    plot=False)
    mass2, isWR2, comp2, AKs2 = calc_mass_isWR_comp(m125, color2,
                                                    iso_mag_f2, iso_col_f2,
                                                    iso_mass_f2, iso_WR_f2,
                                                    comp2_int, mass_max,
                                                    red_vec_dx_f2, red_vec_dy_f2, red_dAKs,
                                                    plot=False)

    # At this point, the Optical and IR outputs still match up. If desired,
    # compare the distribution of delta-AKs values
    testhist = False
    if testhist == True:
        # Normalized histogram of all the AKs values for optical and IR
        py.figure(1, figsize=(10,10))
        py.clf()
        n, bins, patches = py.hist(AKs2, bins=25, color='red', alpha = 0.01,
                                   normed=True)
        py.hist(AKs2, bins=bins, color='red', alpha = 0.6, label = 'IR', normed=True)
        py.hist(AKs1, bins=bins, color='blue', alpha = 0.6, label = 'Optical', normed=True)
        py.xlabel(r'$\Delta$A$_{Ks}$ (mags)')
        py.ylabel('N stars, normalized')
        #py.title('Reddening Distribution, {0}'.format(synthetic.redlaw.name))
        py.axis([-0.5, 0.5, 0, 8])
        py.legend()
        outfile ='AKsHist_t{0:4.2f}_AKs{1:4.2f}_d{2:4.0f}_{3}.png'.format(logAge, AKs,
                                                                              distance,
                                                                              synthetic.redlaw.name)
        py.savefig(outfile)

        # Eliminate stars in the PMS bump, which aren't used in the redmap
        # anyway
        good_ir = np.where( (m125 < 15.3) | (m125 > 17.4) )
        good_opt = np.where( (m814 < 19) | (m814 > 21) )
        
        # Identify the outliers as those with large reddening vals
        outlier_ir = np.where( (AKs2[good_ir] < -0.1) | (AKs2[good_ir] > 0.2) )
        outlier_opt = np.where( (AKs1[good_opt] < -0.1) | (AKs1[good_opt] > 0.2) )

        # Look at the histogram of good AKs vals for optical and IR
        py.figure(2, figsize=(10,10))
        py.clf()
        n, bins, patches = py.hist(AKs2[good_ir], bins=25, color='red', alpha = 0.01,
                                   normed=True)
        py.hist(AKs2[good_ir], bins=bins, color='red', alpha = 0.6, label = 'IR', normed=True)
        py.hist(AKs1[good_opt], bins=bins, color='blue', alpha = 0.6, label = 'Optical', normed=True)
        py.xlabel(r'$\Delta$A$_{Ks}$ (mags)')
        py.ylabel('N stars, normalized')
        #py.title('Trimmed Reddening Distribution, {0}'.format(synthetic.redlaw.name))
        py.axis([-0.5, 0.5, 0, 8])
        py.legend()
        outfile ='AKsHist_t{0:4.2f}_AKs{1:4.2f}_d{2:4.0f}_{3}_trimmed.png'.format(logAge, AKs,
                                                                              distance,
                                                                              synthetic.redlaw.name)
        py.savefig(outfile)       

        # Look at positions of outlier stars in their respective trimmed CMDs
        py.figure(3, figsize=(10,10))
        py.clf()
        py.plot(color1, m814, 'b.', ms=5, alpha = 0.5)
        py.plot(color1[good_opt], m814[good_opt], 'k.', ms=5, alpha=0.5)
        py.plot(color1[good_opt][outlier_opt], m814[good_opt][outlier_opt], 'r.', ms=5)
        py.plot(iso_col_f1, iso_mag_f1, 'g-', linewidth=2)
        py.xlabel('F814W - F160W')
        py.ylabel('F814W')
        py.title('Optical AKs outliers: <-0.1, >0.2')
        py.axis([3, 7, 26, 14])
        outfile='AKs_cmd_optical_t{0:4.2f}_AKs{1:4.2f}_d{2:4.0f}_{3}.png'.format(logAge, AKs,
                                                                              distance,
                                                                              synthetic.redlaw.name)
        py.savefig(outfile)

        py.figure(4, figsize=(10,10))
        py.clf()
        py.plot(color2, m125, 'b.', ms=5, alpha = 0.5)
        py.plot(color2[good_ir], m125[good_ir], 'k.', ms=5, alpha=0.5)
        py.plot(color2[good_ir][outlier_ir], m125[good_ir][outlier_ir], 'r.', ms=5)
        py.plot(iso_col_f2, iso_mag_f2, 'g-', linewidth=2)
        py.xlabel('F125W - F160W')
        py.ylabel('F125W')
        py.title('IR AKs outliers: <-0.1, >0.2')
        py.axis([0.5, 1.7, 21, 12])
        outfile='AKs_cmd_ir_t{0:4.2f}_AKs{1:4.2f}_d{2:4.0f}_{3}.png'.format(logAge, AKs,
                                                                              distance,
                                                                              synthetic.redlaw.name)
        py.savefig(outfile)        

        
        pdb.set_trace()
    
    # Find the maximum mass where we don't have WR stars anymore
    print mass_max, mass1.max(), mass2.max()
    mass_max1 = mass1[isWR1 == False].max()
    mass_max2 = mass2[isWR2 == False].max()
    print mass_max1, mass_max2

    # Trim down to just the stars that aren't WR stars.
    idx1_noWR = np.where(mass1 <= mass_max1)[0]
    mag1_noWR = m814[idx1_noWR]
    color1_noWR = color1[idx1_noWR]
    mass1_noWR = mass1[idx1_noWR]
    isWR1_noWR = isWR1[idx1_noWR]
    comp1_noWR = comp1[idx1_noWR]
    pmem1_noWR = pmem[idx1_noWR]
    AKs1_noWR = AKs1[idx1_noWR]

    # Trim down to just the stars that aren't WR stars.
    idx2_noWR = np.where(mass2 <= mass_max2)[0]
    mag2_noWR = m125[idx2_noWR]
    color2_noWR = color2[idx2_noWR]
    mass2_noWR = mass2[idx2_noWR]
    isWR2_noWR = isWR2[idx2_noWR]
    comp2_noWR = comp2[idx2_noWR]
    pmem2_noWR = pmem[idx2_noWR]
    AKs2_noWR = AKs2[idx2_noWR]

    # Save everything to an output file for later plotting.
    imf1 = Table([mag1_noWR, color1_noWR, mass1_noWR, isWR1_noWR, comp1_noWR, pmem1_noWR, AKs1_noWR,
                  d['x_2013_F160W'][idx1_noWR], d['y_2013_F160W'][idx1_noWR]],
                 names=['mag', 'color', 'mass', 'isWR', 'comp', 'pmem', 'dAKs', 'x', 'y'],
                 meta={'name': 'IMF Table for F814W vs. F814W - F160W',
                       'logAge': logAge,
                       'AKs': AKs,
                       'distance': distance,
                       'magLabel': 'F814W',
                       'colLabel': 'F814W - F160W'})

    imf2 = Table([mag2_noWR, color2_noWR, mass2_noWR, isWR2_noWR, comp2_noWR, pmem2_noWR, AKs2_noWR,
                  d['x_2013_F160W'][idx2_noWR], d['y_2013_F160W'][idx2_noWR]],
                 names=['mag', 'color', 'mass', 'isWR', 'comp', 'pmem', 'dAKs', 'x', 'y'],
                 meta={'name': 'IMF Table for F125W vs. F125W - F160W',
                       'logAge': logAge,
                       'AKs': AKs,
                       'distance': distance,
                       'magLabel': 'F125W',
                       'colLabel': 'F125W - F160W'})

    if synthetic.redlaw.name == 'Nishiyama09':
        red_suf = 'nishi'
    elif synthetic.redlaw.name == 'Westerlund1':
        red_suf = 'wd1'

    suffix = '_t{0:4.2f}_AKs{1:4.2f}_d{2:4.0f}_{3}'.format(logAge, AKs, distance, red_suf)
    imf1.write(work_dir + 'imf_table_from_optical' + suffix + '.fits', overwrite=True)
    imf2.write(work_dir + 'imf_table_from_infrared' + suffix + '.fits', overwrite=True)

    return

def plot_mass_function(logAge=wd1_logAge, AKs=wd1_AKs, distance=wd1_distance):
    suffix = '_t{0:4.2f}_AKs{1:4.2f}_d{2:4.0f}'.format(logAge, AKs, distance)
    imf1 = Table.read(work_dir + 'imf_table_from_optical' + suffix + '.fits')
    imf2 = Table.read(work_dir + 'imf_table_from_infrared' + suffix + '.fits')

    # Read in isochrone
    print 'Loading Isochrone'
    iso = load_isochrone(logAge=imf1.meta['LOGAGE'], 
                         AKs=imf1.meta['AKS'], 
                         distance=imf1.meta['DISTANCE'])
    
    # Make a finely sampled mass-luminosity relationship by
    # interpolating on the isochrone.
    print 'Setting up mass-luminosity interpolater'
    iso_mag1 = iso['mag814w']
    iso_mag2 = iso['mag125w']
    iso_col1 = iso['mag814w'] - iso['mag160w']
    iso_col2 = iso['mag125w'] - iso['mag160w']
    iso_mass = iso['mass']
    iso_WR = iso['isWR']
    iso_tck1, iso_u1 = interpolate.splprep([iso_mass, iso_mag1, iso_col1], s=2)
    iso_tck2, iso_u2 = interpolate.splprep([iso_mass, iso_mag2, iso_col2], s=2)

    # Find the maximum mass that is NOT a WR star
    mass_max = iso_mass[iso_WR == False].max()

    u_fine = np.linspace(0, 1, 1e4)
    iso_mass_f1, iso_mag_f1, iso_col_f1 = interpolate.splev(u_fine, iso_tck1)
    iso_mass_f2, iso_mag_f2, iso_col_f2 = interpolate.splev(u_fine, iso_tck2)
    
    # Define our mag and mass bins.  We will need both for completeness
    # estimation and calculating the mass function. 
    bins_log_mass = np.arange(-1, 1.9, 0.15)
    bins_m814 = get_mag_for_mass(bins_log_mass, iso_mass_f1, iso_mag_f1)
    bins_m125 = get_mag_for_mass(bins_log_mass, iso_mass_f2, iso_mag_f2)

    plot_fit_mass_function(imf1, bins_log_mass, iso_mass, iso_mag1, iso_col1, mass_max, 
                           suffix)
    plot_fit_mass_function(imf2, bins_log_mass, iso_mass, iso_mag2, iso_col2, mass_max, 
                           suffix)

    return

def plot_fit_mass_function(imf, bins_log_mass, iso_mass, iso_mag, iso_color, mass_max,
                           suffix):
    mass_noWR = imf['mass']
    pmem_noWR = imf['pmem']
    comp_noWR = imf['comp']

    filter_name = imf.meta['MAGLABEL']
    color_label = imf.meta['COLLABEL']

    # compute a preliminary mass function with the propoer weights
    weights = pmem_noWR / comp_noWR

    idx = np.where(comp_noWR == 0)[0]
    print 'Bad completeness for N stars = ', len(idx)
    print mass_noWR[idx][0:10]
    weights[idx] = 0

    py.close(1)
    py.figure(1)
    py.clf()
    py.subplots_adjust(top=0.88)
    
    n_raw, be, p = py.hist(np.log10(mass_noWR), bins=bins_log_mass,
                           log=True, histtype='step',
                           label='Unweighted')
    py.clf()
    n_mem, be, p = py.hist(np.log10(mass_noWR), bins=bins_log_mass,
                            log=True, histtype='step', color='green',
                            weights=pmem_noWR,
                            label='Observed')
    
    n_fin, be, p = py.hist(np.log10(mass_noWR), bins=bins_log_mass,
                            log=True, histtype='step', color='black',
                            weights=weights,
                            label='Complete')

    mean_weight = n_fin / n_mem
    print 'mean_weight = ', mean_weight
    
    n_err = (n_raw**0.5) * mean_weight
    n_err[0] = 1000.0  # dumb fix for empty bin
    bc = bins_log_mass[0:-1] + (np.diff(bins_log_mass) / 2.0)

    py.errorbar(bc[1:], n_fin[1:], yerr=n_err[1:], linestyle='none', color='black')

    # Fit a powerlaw.
    powerlaw = lambda x, amp, index: amp * (x**index)
    fitfunc = lambda p, x: p[0] + p[1] * x
    errfunc = lambda p, x, y, err: (y - fitfunc(p, x)) / err

    idx = np.where((bc > 0.2) & (n_fin > 0) & 
                   (np.isnan(n_fin) == False) & (np.isinf(n_fin) == False))[0]
    print bc
    print n_fin
    print n_err
    print idx

    log_m = bc[idx]
    log_n = np.log10(n_fin)[idx]
    log_n_err = n_err[idx] / n_fin[idx]
    
    print 'log_m = ', log_m
    print 'log_n = ', log_n
    print 'log_n_err = ', log_n_err

    pinit = [1.0, -1.0]
    out = optimize.leastsq(errfunc, pinit, args=(log_m, log_n, log_n_err),
                           full_output=1)

    pfinal = out[0]
    covar = out[1]
    print 'pfinal = ', pfinal
    print 'covar = ', covar

    index = pfinal[1]
    amp = 10.0**pfinal[0]

    indexErr = math.sqrt( covar[0][0] )
    ampErr = math.sqrt( covar[1][1] ) * amp

    py.plot(log_m, 10**fitfunc(pfinal, log_m), 'k--')
    fmt = r'$\frac{{dN}}{{dm}}\propto m^{{{0:4.2f}}}$'.format(index)
    py.text(0.90, 200, fmt)

    py.axvline(np.log10(mass_max), linestyle='--')
    py.ylim(5, 9e2)
    py.xlim(-0.5, 1.3)
    py.xlabel('log( Mass [Msun])')
    py.ylabel('Number of Stars')
    py.legend()

    
    ax1 = py.gca()
    ax2 = ax1.twiny()
    
    top_tick_mag = np.array([14.0, 17, 19, 22, 25])
    top_tick_mass = np.zeros(len(top_tick_mag), dtype=float)
    top_tick_label = np.zeros(len(top_tick_mag), dtype='S13')

    for nn in range(len(top_tick_mag)):
        dm = np.abs(iso_mag - top_tick_mag[nn])
        dm_idx = dm.argmin()

        top_tick_mass[nn] = iso_mass[dm_idx]
        top_tick_label[nn] = '{0:3.1f}'.format(top_tick_mag[nn])

    print 'top_tick_mag = ', top_tick_mag
    print 'top_tick_msas = ', top_tick_mass
    print 'top_tick_label = ', top_tick_label

    ax2.set_xlim(ax1.get_xlim())
    ax2.set_xticks(np.log10(top_tick_mass))
    ax2.set_xticklabels(top_tick_label)

    ax2.set_xlabel(filter_name + ' (mags)', labelpad=10)
    py.savefig(plot_dir + 'wd1_imf_' + filter_name + suffix + '.png')

    py.close(2)
    py.figure(2)
    py.clf()
    py.plot(np.log10(iso_mass), iso_mag)
    py.xlim(0, 1.9)
    py.xlabel('Log( Mass [Msun] )')
    py.ylabel(filter_name + ' (mag)')
    py.savefig(plot_dir + 'wd1_mass_luminosity_' + filter_name + suffix + '.png')

    py.close(3)    
    py.figure(3)
    py.clf()
    py.scatter(imf['color'], imf['mag'], c=np.log10(imf['mass']), s=10, vmin=-0.8, vmax=1.3)
    py.plot(iso_color, iso_mag, 'r-', linewidth=2)
    if filter_name == 'F814W':
        py.ylim(13, 26)
    else:
        py.ylim(12, 22)
    py.gca().invert_yaxis()
    py.xlabel(color_label + ' (mag)')
    py.ylabel(filter_name + ' (mag)')
    py.colorbar(orientation='horizontal', fraction=0.05, label=r'log(mass [M$_\odot$])')
    py.savefig(plot_dir + 'wd1_cmd_iso_mass_' + filter_name + suffix + '.png')

    py.close(4)
    py.figure(4)
    py.clf()
    py.scatter(imf['color'], imf['mag'], c=imf.meta['AKS']+imf['dAKs'], s=10)
    py.plot(iso_color, iso_mag, 'r-', linewidth=2)
    if filter_name == 'F814W':
        py.ylim(13, 26)
    else:
        py.ylim(12, 22)
    py.gca().invert_yaxis()
    py.xlabel(color_label + ' (mag)')
    py.ylabel(filter_name + ' (mag)')
    py.colorbar(orientation='horizontal', fraction=0.05, label='AKs (mag)')
    py.savefig(plot_dir + 'wd1_cmd_iso_AKs_' + filter_name + suffix + '.png')
    

    return

    
    
    
def comp_interp_for_cmd(mag, comp_blue, comp_red, blue_name, red_name):
    
    # Make a completeness curve interpolater. First we have to make a
    # color-mag grid and figure out the lowest (worst) completeness at
    # each point in the grid. Then we can interpolate on that grid.
    n_bins = len(mag)
    c_comp_arr = np.zeros((n_bins, n_bins), dtype=float)
    c_mag_arr = np.zeros((n_bins, n_bins), dtype=float)
    c_col_arr = np.zeros((n_bins, n_bins), dtype=float)

    # Loop through an array of BLUE mag and BLUE-RED color and
    # determine the lowest completness.
    # ii = mag
    # jj = color
    for ii in range(n_bins):
        for jj in range(n_bins):
            c_mag_arr[ii, jj] = mag[ii]
            c_col_arr[ii, jj] = mag[ii] - mag[jj]
            print(ii, jj, 
                  '{0:s} = {1:4.2f}'.format(blue_name, mag[ii]), 
                  '{0:s} = {1:4.2f}'.format(red_name, mag[jj]), 
                  '{0:s} - {1:s} = {2:4.2f}'.format(blue_name, red_name, mag[ii] - mag[jj]),
                  'c_{0:s} = {1:4.2f}'.format(blue_name, comp_blue[ii]),
                  'c_{0:s} = {1:4.2f}'.format(red_name, comp_red[jj]))


            # Take whichever is lower, don't multiply because they aren't 
            # really independent.
            if comp_red[jj] < comp_blue[ii]:
                c_comp_arr[ii, jj] = comp_red[jj]
            else:
                c_comp_arr[ii, jj] = comp_blue[ii]
            # c_comp_arr[ii, jj] = comp_blue[ii] * comp_red[jj]

            if c_comp_arr[ii, jj] < 0:
                c_comp_arr[ii, jj] = 0
                
            if c_comp_arr[ii, jj] > 1:
                c_comp_arr[ii, jj] = 1

            if np.isnan(c_comp_arr[ii, jj]):
                c_comp_arr[ii, jj] = 0
                
    # Flatten the arrays and clean out invalid regions.

    # comp_int = interpolate.SmoothBivariateSpline(c_mag_arr.flatten(),
    #                                               c_col_arr.flatten(),
    #                                               c_comp_arr.flatten(), s=2)
    comp_int = interpolate.LinearNDInterpolator((c_mag_arr.flatten(),
                                                 c_col_arr.flatten()),
                                                 c_comp_arr.flatten())
    # comp_int = interpolate.interp2d(c_mag_arr.flatten(),
    #                                 c_col_arr.flatten(),
    #                                 c_comp_arr.flatten(), kind='linear')

    # Plot the raw completeness array
    py.close('all')
    py.figure()
    py.clf()
    py.imshow(c_comp_arr, extent=(c_mag_arr[0,0] + c_col_arr[0,0], 
                                  c_mag_arr[-1,-1] + c_col_arr[-1, -1],
                                  c_mag_arr[0,0], c_mag_arr[-1,-1]), 
              vmin=0, vmax=1, origin='lower')
    py.axis('tight')
    py.colorbar(label='Completeness')
    py.gca().invert_yaxis()
    py.xlabel(red_name + ' (mag)')
    py.ylabel(blue_name + ' (mag)')
    py.savefig(plot_dir + 'completeness_cmd_raw_' + blue_name + '_' + red_name + '.png')
    

    # Plot interpolated completeness image in CMD space: F814W vx. F160W
    if blue_name == 'F814W':
        mm_tmp = np.arange(18.5, 27, 0.1)
        cc_tmp = np.arange(2.0, 8.0, 0.1)
    else:
        mm_tmp = np.arange(14, 25, 0.1)
        cc_tmp = np.arange(0.0, 2.0, 0.1)

    # comp_tmp = comp_int(mm_tmp, cc_tmp)
    mm_tmp_2d, cc_tmp_2d = np.meshgrid(mm_tmp, cc_tmp)
    points = np.array([mm_tmp_2d, cc_tmp_2d]).T
    print points.shape
    comp_tmp = comp_int(points)
    print comp_tmp.shape
    py.clf()
    py.imshow(comp_tmp, extent=(cc_tmp[0], cc_tmp[-1],
                                mm_tmp[0], mm_tmp[-1]), 
              vmin=0, vmax=1, origin='lower')
    
    py.axis('tight')
    py.colorbar(label='Completeness')
    py.xlabel(blue_name + ' - ' + red_name + ' (mag)')
    py.ylabel(blue_name + ' (mag)')
    py.gca().invert_yaxis()
    py.savefig(plot_dir + 'completeness_cmd_' + blue_name + '_' + red_name + '.png')

    
    return comp_int

def make_completeness_ccmd(vel_err_cut=0.5, mag_err_cut=1.0):
    """
    vel_err_cut = mas/yr
    mag_err_cut = magnitudes 
    """
    art = Table.read(art_cat)

    # Get the "detected" stars that are within our error cuts and
    # detected in all three astrometric epochs.
    idx = np.where((art['det_2005_F814W'] == True) &
                   (art['det_2010_F125W'] == True) &
                   (art['det_2010_F160W'] == True) &
                   (art['det_2013_F160W'] == True) &
                   (art['fit_vxe'] < vel_err_cut) &
                   (art['fit_vye'] < vel_err_cut) &
                   (art['me_2005_F814W'] < mag_err_cut) &
                   (art['me_2010_F125W'] < mag_err_cut) &
                   (art['me_2010_F160W'] < mag_err_cut) &
                   (art['me_2013_F160W'] < mag_err_cut))[0]

    # Bins for our 3D mag-color-color (F160W vs. F814W-F160W vs. F125W-F160W) completness table.
    # Note these bins are bin edges. The ranges are only applicable to the cluster members and
    # were selected based on the observed CMD (plus padding).
    bins_mag = np.arange(10, 21, 0.25)    # F160W
    bins_col1 = np.arange(3.0, 7.0, 0.20) # F814W - F160W
    bins_col2 = np.arange(0.5, 1.6, 0.05) # F125W - F160W

    art_mag = art['min_2010_F160W']
    art_col1 = art['min_2005_F814W'] - art['min_2010_F160W']
    art_col2 = art['min_2010_F125W'] - art['min_2010_F160W']
    
    det_mag = art_mag[idx]
    det_col1 = art_col1[idx]
    det_col2 = art_col2[idx]
    
    art_data = np.array([art_mag, art_col1, art_col2]).T
    det_data = np.array([det_mag, det_col1, det_col2]).T

    obs = Table.read(work_dir + 'catalog_membership_3_rot_Pcolor.fits')
    # obs = Table.read(work_dir + 'catalog_diffDered_NN_opt_10.fits')
    obs_mag = obs['m_2010_F160W']
    obs_col1 = obs['m_2005_F814W'] - obs['m_2010_F160W']
    obs_col2 = obs['m_2010_F125W'] - obs['m_2010_F160W']
    
    py.figure(1)
    py.clf()
    py.plot(det_col1, det_mag, 'k.', alpha=0.2, ms=2)
    py.plot(obs_col1, obs_mag, 'r.')
    py.ylim(21.5, 9.5)
    py.xlim(2.5, 7.5)
    py.xlabel('F814W - F160W (mag)')
    py.ylabel('F160W (mag)')

    py.figure(2)
    py.clf()
    py.plot(art_col2, art_mag, 'k.', alpha=0.2, ms=2)
    py.plot(obs_col2, obs_mag, 'r.')
    py.ylim(21.5, 9.5)
    py.xlim(0.2, 1.8)
    py.xlabel('F125W - F160W (mag)')
    py.ylabel('F160W (mag)')

    pdb.set_trace()
    
    
    bins = np.array([bins_mag, bins_col1, bins_col2])

    n_art, b_art = np.histogramdd(art_data, bins=bins)
    n_det, b_det = np.histogramdd(det_data, bins=bins)

    comp = n_det / n_art

    bad = np.where(n_art == 0)

    comp[bad] = 0.0

    pdb.set_trace()

    return

   
def make_completeness_cmds():
    comp = Table.read(work_dir + 'completeness_vs_mag.fits')
    mag = comp['mag']

    n_bins = len(mag)
    c_comp_arr = np.zeros((n_bins, n_bins, n_bins), dtype=float)
    c_mag_arr  = np.zeros((n_bins, n_bins, n_bins), dtype=float)
    c_col1_arr = np.zeros((n_bins, n_bins, n_bins), dtype=float)
    c_col2_arr = np.zeros((n_bins, n_bins, n_bins), dtype=float)

    comp_814 = comp['c_2005_F814W']  #### HERE ####
    comp_125 = comp['c_2010_F125W']
    comp_160_10 = comp['c_2010_F160W']
    comp_160_13 = comp['c_2013_F160W']

    # Loop through an array of BLUE mag and BLUE-RED color and
    # determine the lowest completness.
    # mm = mag F160W
    # cc1 = color F814W - F160W
    # cc2 = color F125W - F160W
    for mm in range(n_bins):
        for cc1 in range(n_bins):
            for cc2 in range(n_bins):
                c_mag_arr[mm, cc1, cc2] = mag[mm]
                c_col1_arr[mm, cc1, cc2] = mag[cc1] - mag[mm]
                c_col2_arr[mm, cc1, cc2] = mag[cc2] - mag[mm]

                comp_all_filt = np.array([comp_160_10[mm], comp_160_13[mm],
                                          comp_814[cc1], comp_125[cc2]])

                # Take whichever is lower, don't multiply because they aren't 
                # really independent.
                c_comp_arr[mm, cc1, cc2] = comp_all_filt.min()

                if c_comp_arr[mm, cc1, cc2] < 0:
                    c_comp_arr[mm, cc1, cc2] = 0
                
                if c_comp_arr[mm, cc1, cc2] > 1:
                    c_comp_arr[mm, cc1, cc2] = 1

                if np.isnan(c_comp_arr[mm, cc1, cc2]):
                    c_comp_arr[mm, cc1, cc2] = 0
                
    print 'Plotting raw completeness mag vs. mag (comp_mF160W_mF814W.mp4).'
    # Plot the raw completeness array
    py.close(1)
    fig = py.figure(1)

    ii = 0
    im = py.imshow(c_comp_arr[:, :, ii], vmin=0, vmax=1, origin='lower',
                   extent=(c_col1_arr[0, 0, ii] + c_mag_arr[0, 0, ii], 
                           c_col1_arr[-1, -1, ii] + c_mag_arr[-1, -1, ii],
                           c_mag_arr[0, 0, ii], c_mag_arr[-1, -1, ii]))

    def update_fig(ii):
        im.set_array(c_comp_arr[:, :, ii])
        py.title('F125W = {0:5.2f}'.format(c_col2_arr[0, 0, ii] + c_mag_arr[0, 0, ii]))
        return im,

    py.axis('tight')
    py.colorbar(label='Completeness')
    py.gca().invert_yaxis()
    py.ylabel('F160W (mag)')
    py.xlabel('F814W (mag)')
    # ani = animation.FuncAnimation(fig, update_fig, np.arange(c_comp_arr.shape[2]),
    #                               interval=50, blit=True)
    # ani.save('comp_mF160W_mF814W.mp4')

    ##########
    # Flatten the arrays and clean out invalid regions.
    ##########

    print 'Interpolating to CMD. Setup'
    comp_int = interpolate.LinearNDInterpolator((c_mag_arr.flatten(),
                                                 c_col1_arr.flatten(),
                                                 c_col2_arr.flatten()),
                                                 c_comp_arr.flatten())
    bins_mag = np.arange(10, 20, 0.25)  # F160W
    bins_col1 = np.arange(3.0, 7.0, 0.20) # F814W - F160W
    bins_col2 = np.arange(0.5, 1.6, 0.05) # F125W - F160W

    points_3d = np.meshgrid(bins_mag, bins_col1, bins_col2, indexing='ij')

    # comp_tmp = interpolate.griddata((c_mag_arr.flatten(),
    #                                  c_col1_arr.flatten(),
    #                                  c_col2_arr.flatten()),
    #                                  c_comp_arr.flatten(), points_3d)
    # pdb.set_trace()

    
    print 'Interpolating to CMD. Calc.'
    comp_tmp = comp_int(points_3d)
    print points_3d.shape, comp_tmp.shape
    pdb.set_trace()

    # Plot
    py.clf()
        
    ii = 0
    im = py.imshow(comp_tmp[:, :, ii], vmin=0, vmax=1, origin='lower',
                   extent=(bins_col1[0], bins_col1[-1],
                           bins_mag[0], bins_mag[-1]))
                           
    def update_fig(ii):
        im.set_array(comp_tmp[:, :, ii])
        py.title('F125W = {0:5.2f}'.format(bins_col2[ii]))
        return im,

    py.axis('tight')
    py.colorbar(label='Completeness')
    py.gca().invert_yaxis()
    py.ylabel('F160W (mag)')
    py.xlabel('F814W - F160W (mag)')
    ani = animation.FuncAnimation(fig, update_fig, np.arange(comp_tmp.shape[2]),
                                  interval=50, blit=True)
    ani.save('comp_cmd_mF160W_mF814W.mp4')
    
    # return comp_int
    
    
    
def calc_mass_isWR_comp(mag, color, iso_mag_f, iso_col_f, iso_mass_f, iso_WR_f,
                         comp_int, mass_max, red_vec_dx_f, red_vec_dy_f, dAKs_0, plot=False):
    # F814W vs. F814W - F160W    
    # Loop through data and assign masses, extinctions, and completeness to each star.
    mass = np.zeros(len(mag), dtype=float)
    isWR = np.zeros(len(mag), dtype=float)
    comp = np.zeros(len(mag), dtype=float)
    dAKs = np.zeros(len(mag), dtype=float)
    delta_arr = np.zeros(len(mag), dtype=float)

    red_dAKs = np.arange(-0.5, 0.5, 0.01)
    red_col = red_dAKs / dAKs_0
    red_mag = red_dAKs / dAKs_0

    # Loop through observed stars.
    for ii in range(len(mass)):
        delta_min = 100  # junk
        rr_min = -1      # junk
        mass_min = 1000  # junk

        # Get the list of closest iso match points at
        # each reddening vector. 
        iso_idx_per_rr = np.ones(len(red_dAKs), dtype=int) * -1
        
        for rr in range(len(red_dAKs)):
            dmag = mag[ii] - (iso_mag_f + (red_mag[rr] * red_vec_dy_f))
            dcol = color[ii] - (iso_col_f + (red_col[rr] * red_vec_dx_f))

            delta = np.hypot(dmag, dcol)

            # Find the closest iso point (at this reddening). 
            sdx = delta.argsort()

            # If the color + mag difference is less than 0.02, then take
            # the lowest mass. This helps account for the missing IMF bias.
            idx = np.where(delta[sdx] < 0.02)[0]

            if (len(idx) > 1):
                # More than one in a tight radius... choose the lower mass.
                min_mass_idx = iso_mass_f[sdx[idx]].argmin()
                min_idx = sdx[idx][min_mass_idx]
                
                iso_idx_per_rr[rr] = min_idx
            else:
                # One or zero within the radius... just take the closest.
                iso_idx_per_rr[rr] = sdx[0]
            # iso_idx_per_rr[rr] = sdx[0]
                
        # From the candidates, choose the closest first, then the lowest
        # mass one (within 0.02).
        #dmag = mag[ii] - (iso_mag_f[iso_idx_per_rr] + red_mag)
        #dcol = color[ii] - (iso_col_f[iso_idx_per_rr] + red_col)
        dmag = mag[ii] - (iso_mag_f[iso_idx_per_rr] + (red_mag * red_vec_dy_f[iso_idx_per_rr]))
        dcol = color[ii] - (iso_col_f[iso_idx_per_rr] + (red_col * red_vec_dx_f[iso_idx_per_rr]))
        delta = np.hypot(dmag, dcol)
        
        sdx = delta.argsort()
        idx = np.where(delta[sdx] < 0.02)[0]

        if len(idx) > 1:
            #---------METHOD 1: Choose lower mass------#
            # More than one in a tight radius... choose the lower mass.
            #min_mass_idx = iso_mass_f[iso_idx_per_rr[sdx[idx]]].argmin()
            #--METHOD 2: Choose lower absolute AKs val--#
            min_Aks_idx = abs(red_mag[sdx[idx]]).argmin()

            min_rdx = sdx[idx][min_Aks_idx]
        else:
            min_rdx = sdx[0]
        # min_rdx = sdx[0]
        
        min_idx = iso_idx_per_rr[min_rdx]
        

        print '{0:4d} {1:4d} {2:4d} {3:4.2f} {4:5.1f} {5:4.2f}'.format(ii, min_idx, min_rdx,
                                                              delta[min_rdx],
                                                              iso_mass_f[min_idx],
                                                              red_dAKs[min_rdx])
        

        mass[ii] = iso_mass_f[min_idx]
        isWR[ii] = iso_WR_f[min_idx]
        dAKs[ii] = red_dAKs[min_rdx]
        comp[ii] = comp_int(mag[ii], color[ii])
        delta_arr[ii] = delta[min_rdx]

        # Sanity plot. Currently tuned to IR
        if plot:
            if ((mag[ii] > 17.5) & (mag[ii] < 18.5) & (color[ii]>1.3)):
            #if ((mag[ii] > 17.5) & (color[ii]<1.1)):
                py.figure(1, figsize=(10,10))
                py.clf()
                py.plot(color[ii], mag[ii], 'k*', ms=10)
                py.plot(iso_col_f, iso_mag_f, 'r-')
                py.plot(iso_col_f[iso_idx_per_rr], iso_mag_f[iso_idx_per_rr], 'k.', ms = 5)
                py.plot(iso_col_f[min_idx], iso_mag_f[min_idx], 'r*', ms=10)
                mag_tmp = (iso_mag_f[iso_idx_per_rr] + (red_mag * red_vec_dy_f[iso_idx_per_rr]))
                col_tmp =  (iso_col_f[iso_idx_per_rr] + (red_col * red_vec_dx_f[iso_idx_per_rr]))
                py.plot(col_tmp, mag_tmp, 'g.', ms=5)
                py.axis([min(color), max(color), max(mag), min(mag)])
                py.title('dAKs = {0}'.format(dAKs[ii]))
                py.savefig('Match.png')
                pdb.set_trace()
                
        if comp[ii] > 1:
            comp[ii] = 1
        if comp[ii] < 0:
            comp[ii] = 0
    
    print mag.min(), mag.max(), color.min(), color.max()
    print comp.shape, mag.shape, color.shape

    return mass, isWR, comp, dAKs
    
    
def get_mag_for_mass(log_mass, iso_mass, iso_mag):
    mag = np.zeros(len(log_mass), dtype=float)

    for ii in range(len(mag)):
        dmass = np.abs((10**log_mass[ii]) - iso_mass)
        dmass_min_idx = dmass.argmin()

        mag[ii] = iso_mag[dmass_min_idx]

    print 'log_mass = ', log_mass
    print 'mass = ', 10**log_mass
    print 'mag = ', mag

    return mag
    

def load_isochrone(logAge=wd1_logAge, AKs=wd1_AKs, distance=wd1_distance, IAU=False):
    tmp_dist = 4000

    filters={'814w': 'acs,wfc1,f814w',
             '139m': 'wfc3,ir,f127m',
             '125w': 'wfc3,ir,f125w',
             '160w': 'wfc3,ir,f160w'}

    print 'Using Red Law = ', synthetic.redlaw.name

    # Change iso_dir depending on redlaw used
    if synthetic.redlaw.name == 'Nishiyama09':
        iso_dir = iso_dir_nishi09
    elif synthetic.redlaw.name == 'Westerlund1':
        iso_dir = iso_dir_wd1
        if IAU==True:
            iso_dir = iso_dir_wd1+'IAU_law/'

    iso = synthetic.IsochronePhot(logAge, AKs, tmp_dist,
                                   iso_dir=iso_dir, mass_sampling=1,
                                   filters=filters, red_law = synthetic.redlaw)

    # Extract isochrone properties
    iso_f = iso.points
    col_names = iso_f.keys()

    # Correct for distance, if necessary
    for cc in range(len(col_names)):
        delta_DM = 5.0 * math.log10(float(distance) / tmp_dist)
        print 'Changing distance: delta_DM = ', delta_DM
        
        if col_names[cc].startswith('mag'):
            iso_f[col_names[cc]] += delta_DM

    return iso_f

        
def check_atmospheres():
    cdbs_dir = os.environ['PYSYN_CDBS']

    castelli_file = cdbs_dir + 'grid/ck04models/catalog.fits'
    phoenix_file = cdbs_dir + 'grid/phoenix_v16_rebin/catalog.fits'

    castelli = Table.read(castelli_file)
    phoenix = Table.read(phoenix_file)

    c_teff = np.array([int(castelli[ii]['INDEX'].split(',')[0]) for ii in range(len(castelli))])
    c_metal = np.array([float(castelli[ii]['INDEX'].split(',')[1]) for ii in range(len(castelli))])
    c_logg = np.array([float(castelli[ii]['INDEX'].split(',')[2]) for ii in range(len(castelli))])

    p_teff = np.array([int(phoenix[ii]['INDEX'].split(',')[0]) for ii in range(len(phoenix))])
    p_metal = np.array([float(phoenix[ii]['INDEX'].split(',')[1]) for ii in range(len(phoenix))])
    p_logg = np.array([float(phoenix[ii]['INDEX'].split(',')[2]) for ii in range(len(phoenix))])
    
    # Get the unique log-g and loop through to find Teff range for each.
    c_logg_uni = np.unique(c_logg)

    print 'Castelli: solar metallicity'
    for ii in range(len(c_logg_uni)):
        idx = np.where((c_logg == c_logg_uni[ii]) & (c_metal == 0))[0]

        teff_good = c_teff[idx] 
        min_teff = teff_good.min()
        max_teff = teff_good.max()

        fmt = 'logg = {0:4.2f}   Teff = [{1:5.0f} - {2:5.0f}]'
        print fmt.format(c_logg_uni[ii], min_teff, max_teff)


    # Get the unique log-g and loop through to find Teff range for each.
    p_logg_uni = np.unique(p_logg)
    print ''
    print 'Phoenix: solar metallicity'
    for ii in range(len(p_logg_uni)):
        idx = np.where((p_logg == p_logg_uni[ii]) & (p_metal == 0))[0]

        teff_good = p_teff[idx] 
        min_teff = teff_good.min()
        max_teff = teff_good.max()

        fmt = 'logg = {0:4.2f}   Teff = [{1:5.0f} - {2:5.0f}]'
        print fmt.format(p_logg_uni[ii], min_teff, max_teff)
        
def do_a_bunch():
    AKs = np.arange(0.67, 0.75, 0.01)
    # AKs = np.array([0.67, 0.73])

    for ii in range(len(AKs)):
        plot_cmd_isochrone(AKs=AKs[ii])
        calc_mass_function(AKs=AKs[ii])
        plot_mass_function(AKs=AKs[ii])    

    return
