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
from scipy import interpolate
import matplotlib
from popstar import synthetic
from jlu.stellarModels import extinction

reduce_dir = '/Users/jlu/data/wd1/hst/reduce_2015_01_05/'
cat_dir = reduce_dir + '50.ALIGN_KS2/'
art_dir = reduce_dir + '51.ALIGN_ART/'

work_dir = '/Users/jlu/work/wd1/analysis_2015_01_05/'
plot_dir = work_dir + 'plots/'

catalog = 'wd1_catalog_RMSE_wvelErr.fits'

# Catalogs after membership calculations
cat_pclust = work_dir + 'membership/gauss_3/catalog_membership_3_rot.fits'
cat_pclust_pcolor = work_dir + 'catalog_membership_3_rot_Pcolor.fits'

# Artificial catalog
art_cat = art_dir + 'wd1_art_catalog_RMSE_wvelErr.fits'

# Best fit (by eye) cluster parameters
wd1_logAge = 6.91
wd1_AKs = 0.75
wd1_distance = 4000

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
    vel_err_cut = 0.5 # mas/yr
    prob = 0.3
    N_gauss = 3
    out_dir = '{0}membership/gauss_{1:d}/'.format(work_dir, N_gauss)
    
    membership.cluster_membership(cat_dir + catalog,
                                  vel_err_cut, mag_err_cut,
                                  out_dir, N_gauss, prob, 
                                  rotate=True)

    return
    

def make_cmd(catalog=cat_pclust, cl_prob=0.3, usePcolor=False, suffix=''):
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
    py.ylim(23, 14)
    py.savefig(plot_dir + 'cmd_ir_clust' + suffix + '.png')

    # IR Plot CMD with everything in black.
    py.clf()
    py.plot(color, mag2, 'k.', ms=2)
    py.ylabel('F125W (mag)')
    py.xlabel('F125W - F160W (mag)')
    py.xlim(0, 2)
    py.ylim(23, 14)
    py.savefig(plot_dir + 'cmd_ir_all' + suffix + '.png')
    
    # IR Plot CMD with everything in black
    # and cluster members in red. 
    py.clf()
    py.plot(color, mag2, 'k.', ms=2)
    py.plot(color[clust], mag2[clust], 'r.', ms=2)
    py.ylabel('F125W (mag)')
    py.xlabel('F125W - F160W (mag)')
    py.xlim(0, 2)
    py.ylim(23, 14)
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
    py.ylim(26, 18)
    py.savefig(plot_dir + 'cmd_2_clust' + suffix + '.png')

    # F814W - F125W Plot CMD with everything in black.
    py.clf()
    py.plot(color, mag2, 'k.', ms=2)
    py.ylabel('F814W (mag)')
    py.xlabel('F814W - F125W (mag)')
    py.xlim(1, 7)
    py.ylim(26, 18)
    py.savefig(plot_dir + 'cmd_2_all' + suffix + '.png')
    
    # F814W - F125W Plot CMD with everything in black
    # and cluster members in red. 
    py.clf()
    py.plot(color, mag2, 'k.', ms=2)
    py.plot(color[clust], mag2[clust], 'r.', ms=2)
    py.ylabel('F814W (mag)')
    py.xlabel('F814W - F125W (mag)')
    py.xlim(1, 7)
    py.ylim(26, 18)
    py.savefig(plot_dir + 'cmd_2_all_clust' + suffix + '.png')



        
    
    ##########
    # Plot CMD Hess diagram of all
    ##########
    bins_col = np.arange(2, 10, 0.2)
    bins_mag = np.arange(18, 26, 0.2)
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
    bins_mag = np.arange(14, 23, 0.2)
    mag1 = d['m_2013_F160W']
    mag2 = d['m_2010_F125W']
    color = mag2 - mag1
    py.clf()
    py.hist2d(color, mag2, bins=(bins_col, bins_mag),
              cmap=py.cm.gist_heat_r)
    py.ylabel('F125W (mag)')
    py.xlabel('F125W - F160W (mag)')
    py.xlim(0, 2)
    py.ylim(23, 14)
    py.savefig(plot_dir + 'cmd_ir_hess_all' + suffix + '.png')

    # IR: Plot CMD Hess diagram of cluster members.
    py.clf()
    py.hist2d(color[clust], mag2[clust], bins=(bins_col, bins_mag),
              cmap=py.cm.gist_heat_r)
    py.ylabel('F125W (mag)')
    py.xlabel('F125W - F160W (mag)')
    py.xlim(0, 2)
    py.ylim(23, 14)
    py.savefig(plot_dir + 'cmd_ir_hess_clust' + suffix + '.png')

    return
        

def plot_color_color(cl_prob=0.6):
    # Read in data table
    d = Table.read(cat_pclust)

    # Determine which we will call "cluster members"
    clust = np.where(d['Membership'] > cl_prob)[0]
    d_cl = d[clust]

    color1 = d['m_2005_F814W'] - d['m_2010_F125W']
    color2 = d['m_2010_F125W'] - d['m_2013_F160W']
    
    py.clf()
    py.plot(color2, color1, 'k.', ms=2)
    py.plot(color2[clust], color1[clust], 'r.', ms=2)
    py.xlim(0, 2.5)
    py.ylim(1, 8)
    py.savefig(plot_dir + 'colcol_all_clust.png')

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
    cb, p = interpolate.splprep(foo, s=780, k=3)

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


def plot_cmd_cluster_with_isochrones(logAge=wd1_logAge, AKs=wd1_AKs,
                                     distance=wd1_distance):
    d = Table.read(cat_pclust_pcolor)

    pmem = d['Membership'] * d['Membership_color']
    m160 = d['m_2013_F160W']
    m125 = d['m_2010_F125W']
    m814 = d['m_2005_F814W']
    color1 = m814 - m160
    color2 = m125 - m160

    clust = np.where(pmem > 0.8)[0]

    iso = synthetic.load_isochrone(logAge=logAge, AKs=AKs, distance=distance,
                                   iso_dir='/u/jlu/work/wd1/models/iso_2015/')

    # F814W vs. F160W CMD
    py.figure(1)
    py.clf()
    py.plot(color1[clust], m814[clust], 'k.', ms=2)
    py.plot(iso['mag814w'] - iso['mag160w'], iso['mag814w'], 'r.', ms=10)
    py.ylim(26, 18)
    py.xlim(3.2, 7)
    py.xlabel('F814W - F160W')
    py.ylabel('F814W')

    py.figure(2)
    py.clf()
    py.plot(color2[clust], m125[clust], 'k.', ms=2)
    py.plot(iso['mag125w'] - iso['mag160w'], iso['mag125w'], 'r.', ms=10)
    py.ylim(21.5, 15)
    py.xlim(0.6, 1.7)
    py.xlabel('F125W - F160W')
    py.ylabel('F125W')

    py.figure(3)
    py.clf()
    py.plot(color2[clust], color1[clust], 'k.', ms=2)
    py.plot(iso['mag125w'] - iso['mag160w'], iso['mag814w'] - iso['mag160w'],
            'r.', ms=10)
    py.ylim(3.2, 7)
    py.xlim(0.6, 1.7)
    py.xlabel('F125W - F160W')
    py.ylabel('F125W')
        
    return

def compare_art_vs_obs_vel():
    """
    Compare the distribution of velocities in the artificial
    vs. the observed stars. These need to match if we are going
    to make a cut on the velocitiy error. 
    """
    obs = Table.read(cat_pclust_pcolor)
    art = Table.read(art_cat)

    # Trim out un-detected in artificial star lists.    
    adx_det = np.where((art['fit_vxe'] != 1) & (art['fit_vye'] != 1))[0]
    art = art[adx_det]

    # Convert artificial star velocities into mas / yr.
    scale = astrometry.scale['WFC'] * 1e3
    art['fit_vx'] *= scale
    art['fit_vy'] *= scale
    art['fit_vxe'] *= scale
    art['fit_vye'] *= scale

    py.clf()
    py.semilogy(art['m_2013_F160W'], art['fit_vxe'], 'k.', ms=2, alpha=0.2,
                label='Simulated')
    py.semilogy(obs['m_2013_F160W'], obs['fit_vxe'], 'r.', ms=4, alpha=0.5,
                label='Observed')
    py.ylim(1e-2, 5)
    py.xlim(13, 22)
    py.xlabel('F160W 2013 (mag)')
    py.ylabel('X Velocity Error (mas/yr)')
    py.savefig(plot_dir + 'compare_art_vs_obs_vel_x.png')

    py.clf()
    py.semilogy(art['m_2013_F160W'], art['fit_vye'], 'k.', ms=2, alpha=0.2,
                label='Simulated')
    py.semilogy(obs['m_2013_F160W'], obs['fit_vye'], 'r.', ms=4, alpha=0.5,
                label='Observed')
    py.ylim(1e-2, 5)
    py.xlim(13, 22)
    py.xlabel('F160W 2013 (mag)')
    py.ylabel('Y Velocity Error (mas/yr)')
    py.savefig(plot_dir + 'compare_art_vs_obs_vel_y.png')
    
    mdx_art = np.where(art['m_2013_F160W'] < 17)[0]
    mdx_obs = np.where(obs['m_2013_F160W'] < 17)[0]
    
    med_art_x = np.median(art[mdx_art]['fit_vxe'])
    med_obs_x = np.median(obs[mdx_obs]['fit_vxe'])
    med_art_y = np.median(art[mdx_art]['fit_vye'])
    med_obs_y = np.median(obs[mdx_obs]['fit_vye'])

    print 'Median Velocity (mas/yr)'
    print '   X: Obs = {0:5.3f}  Art = {1:5.3f}'.format(med_obs_x, med_art_x)
    print '   Y: Obs = {0:5.3f}  Art = {1:5.3f}'.format(med_obs_y, med_art_y)

def make_completeness_table():
    art = Table.read(art_cat)
    

    # Make a completeness curve for each filter independently.
    epochs = ['2005_F814W', '2010_F125W', '2010_F139M', '2010_F160W', '2013_F160W']
    
    mag_bins = np.arange(12, 27.01, 0.25)
    mag_bins_left = mag_bins[:-1]
    mag_bins_right = mag_bins[1:]
    mag_bins_mid = (mag_bins_left + mag_bins_right) / 2.0

    comp = Table([mag_bins_left, mag_bins_right, mag_bins_mid],
                 names=['mag_left', 'mag_right', 'mag'],
                  meta={'name': 'Completeness Table'})

    for ee in range(len(epochs)):
        ep = epochs[ee]
        idx = np.where(art['det_' + ep] == True)[0]
        det = art[idx]
        
        n_all, b_all = np.histogram(art['min_' + ep], bins=mag_bins)
        n_det, b_det = np.histogram(det['min_' + ep], bins=mag_bins)

        c = (1.0 * n_det) / n_all
        ce = c * np.sqrt(1.0 * n_det) / n_det
        
        # Fix some issues at the bright end (where number of sources is small).
        # For later good behavior in interpolation, set it to
        # the first "good" value. But set errors to np.nan to notify.
        #     m=20 is arbitrary but catches both F814W and IR
        idx = np.where((mag_bins_mid < 20) & ((ce > 0.1) | (n_all < 10)))[0]  
        c[idx] = c[idx[-1] + 1]
        ce[idx] = np.nan

        # Do the same for the faint end; but set to 0.
        idx = np.where((mag_bins_mid >= 20) & ((ce > 0.1) | (n_all < 10)))[0]  
        c[idx] = 0.0
        ce[idx] = np.nan
        
        
        col_c = Column(c, name='c_'+ep)
        col_ce = Column(ce, name='cerr_'+ep)
        col_N = Column(n_all, name='Nplant_'+ep)

        comp.add_columns([col_c, col_ce, col_N], copy=False)
        
        
    # Plot
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
        
def plot_mass_function(logAge=wd1_logAge, AKs=wd1_AKs, distance=wd1_distance):
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
    iso = synthetic.load_isochrone(logAge=logAge, AKs=AKs, distance=distance,
                                   iso_dir='/u/jlu/work/wd1/models/iso_2015/')

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
    iso_tck1, iso_u1 = interpolate.splprep([iso_mass, iso_mag1, iso_col1], s=2)
    iso_tck2, iso_u2 = interpolate.splprep([iso_mass, iso_mag2, iso_col2], s=2)

    # Find the maximum mass that is NOT a WR star
    mass_max = iso_mass[iso_WR == False].max()

    u_fine = np.linspace(0, 1, 1e4)
    iso_mass_f1, iso_mag_f1, iso_col_f1 = interpolate.splev(u_fine, iso_tck1)
    iso_mass_f2, iso_mag_f2, iso_col_f2 = interpolate.splev(u_fine, iso_tck2)

    # Define WR stars. 
    iso_WR_f1 = np.zeros(len(iso_mass_f1), dtype=bool)
    iso_WR_f2 = np.zeros(len(iso_mass_f2), dtype=bool)
    iso_WR_f1[iso_mass_f1 > mass_max] = True
    iso_WR_f2[iso_mass_f2 > mass_max] = True

    print 'Setting up completeness interpolater for CMD space.'
    comp1_int = comp_interp_for_cmd(comp['mag'], comp['c_2005_F814W'],
                                    comp['c_2010_F160W'])
    comp2_int = comp_interp_for_cmd(comp['mag'], comp['c_2010_F125W'],
                                    comp['c_2010_F160W'])


    # Plot completeness image in CMD space: F814W vx. F160W
    mm1_tmp = np.arange(17, 27, 0.1)
    cc1_tmp = np.arange(2.0, 8.0, 0.1)
    comp1_tmp = comp1_int(mm1_tmp, cc1_tmp)
    py.clf()
    py.imshow(comp1_tmp, extent=(cc1_tmp.min(), cc1_tmp.max(),
                                 mm1_tmp.min(), mm1_tmp.max()), vmin=0, vmax=1)
    py.axis('tight')
    py.colorbar(label='Completeness')
    py.xlabel('F814W - F160W (mag)')
    py.ylabel('F814W (mag)')
    py.gca().invert_yaxis()
    py.savefig(plot_dir + 'completeness_cmd_F814W_F160W.png')

    # Plot completeness image in CMD space: F125W vx. F160W
    mm2_tmp = np.arange(11, 25, 0.1)
    cc2_tmp = np.arange(0.0, 2.0, 0.1)
    comp2_tmp = comp2_int(mm2_tmp, cc2_tmp)
    py.clf()
    py.imshow(comp2_tmp, extent=(cc2_tmp.min(), cc2_tmp.max(),
                                 mm2_tmp.min(), mm2_tmp.max()), vmin=0, vmax=1)
    py.axis('tight')
    py.colorbar(label='Completeness')
    py.xlabel('F125W - F160W (mag)')
    py.ylabel('F125W')
    py.gca().invert_yaxis()
    py.savefig(plot_dir + 'completeness_cmd_F125W_F160W.png')

    
    mass1, isWR1, comp1 = calc_mass_isWR_comp(m814, color1,
                                              iso_mag_f1, iso_col_f1,
                                              iso_mass_f1, iso_WR_f1,
                                              comp1_int, mass_max)
    mass2, isWR2, comp2 = calc_mass_isWR_comp(m125, color2,
                                              iso_mag_f2, iso_col_f2,
                                              iso_mass_f2, iso_WR_f2,
                                              comp2_int, mass_max)

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

    # Trim down to just the stars that aren't WR stars.
    idx2_noWR = np.where(mass2 <= mass_max2)[0]
    mag2_noWR = m125[idx2_noWR]
    color2_noWR = color2[idx2_noWR]
    mass2_noWR = mass2[idx2_noWR]
    isWR2_noWR = isWR2[idx2_noWR]
    comp2_noWR = comp2[idx2_noWR]
    pmem2_noWR = pmem[idx2_noWR]
    
    # Define our mag and mass bins.  We will need both for completeness
    # estimation and calculating the mass function. 
    bins_log_mass = np.arange(0, 1.9, 0.15)
    bins_m814 = get_mag_for_mass(bins_log_mass, iso_mass_f1, iso_mag_f1)
    bins_m125 = get_mag_for_mass(bins_log_mass, iso_mass_f2, iso_mag_f2)

    plot_fit_mass_function(mass1_noWR, pmem1_noWR, comp1_noWR, bins_log_mass,
                           iso_mass, iso_mag1, iso_col1, 'F814W', 'F814W - F160W')
    plot_fit_mass_function(mass2_noWR, pmem2_noWR, comp2_noWR, bins_log_mass,
                           iso_mass, iso_mag2, iso_col2, 'F125W', 'F125W - F160W')

    return

def plot_fit_mass_function(mass_noWR, pmem_noWR, comp_noWR, bins_log_mass,
                           iso_mass, iso_mag, iso_color, filter_name, color_label):
    # compute a preliminary mass function with the propoer weights
    weights = pmem_noWR / comp_noWR

    idx = np.where(comp_noWR == 0)[0]
    print 'Bad completeness for N stars = ', len(idx)
    print mass_noWR[idx][0:10]
    weights[idx] = 0

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
                            label='Completeness Corr.')

    mean_weight = n_fin / n_mem
    
    n_err = (n_fin**0.5) * mean_weight
    n_err[0] = 1000.0  # dumb fix for empty bin
    bc = bins_log_mass[0:-1] + (np.diff(bins_log_mass) / 2.0)

    py.errorbar(bc[1:], n_fin[1:], yerr=n_err[1:], linestyle='none', color='black')

    # Fit a powerlaw.
    powerlaw = lambda x, amp, index: amp * (x**index)
    fitfunc = lambda p, x: p[0] + p[1] * x
    errfunc = lambda p, x, y, err: (y - fitfunc(p, x)) / err

    log_m = bc[5:]
    log_n = np.log10(n_fin)[5:]
    log_n_err = n_err[5:] / n_fin[5:]

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
    py.text(1.3, 100, r'$\frac{dN}{dm}\propto m^{-2.2}$')

    py.axvline(np.log10(mass_max), linestyle='--')
    py.ylim(5, 9e2)
    py.xlim(0, 1.8)
    py.xlabel('log( Mass [Msun])')
    py.ylabel('Number of Stars')
    py.legend()

    
    ax1 = py.gca()
    ax2 = ax1.twiny()
    
    top_tick_mag = np.array([14.0, 17, 20, 21.6])
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
    py.savefig(plot_dir + 'wd1_imf_' + filter_name + '.png')

    py.figure(2)
    py.clf()
    py.plot(np.log10(iso_mass), iso_mag)
    py.xlim(0, 1.9)
    py.xlabel('Log( Mass [Msun] )')
    py.ylabel(filter_name + ' (mag)')
    py.savefig(plot_dir + 'wd1_mass_luminosity_' + filter_name +  '.png')
    
    py.figure(3)
    py.clf()
    py.plot(color, mag, 'k.')
    py.plot(iso_color, iso_mag, 'r-')
    py.axvline(2)
    # py.ylim(22, 12)
    # py.xlim(1, 3.5)
    py.xlabel(color_label + ' (mag)')
    py.ylabel(filter_name + ' (mag)')
    py.savefig(prop_dir + 'arches_cmd_iso_test_' + filter_name + '.png')

    return clust

    
    
    
def comp_interp_for_cmd(mag, comp_blue, comp_red):
    
    # Make a completeness curve interpolater. First we have to make a
    # color-mag grid and figure out the lowest (worst) completeness at
    # each point in the grid. Then we can interpolate on that grid.
    n_bins = len(mag)
    c_comp_arr = np.zeros((n_bins, n_bins), dtype=float)
    c_mag_arr = np.zeros((n_bins, n_bins), dtype=float)
    c_col_arr = np.zeros((n_bins, n_bins), dtype=float)

    # Loop through an array of BLUE mag and BLUE-RED color and
    # determine the lowest completness.
    for ii in range(n_bins):
        for jj in range(n_bins):
            c_mag_arr[ii, jj] = mag[ii]
            c_col_arr[ii, jj] = mag[jj] - mag[ii]

            # I chose 2010_F160W because it has shallower completeness.
            c_comp_arr[ii, jj] = comp_red[ii] * comp_blue[jj]
                    
            if c_comp_arr[ii, jj] < 0:
                c_comp_arr[ii, jj] = 0
                
            if c_comp_arr[ii, jj] > 1:
                c_comp_arr[ii, jj] = 1

            if np.isnan(c_comp_arr[ii, jj]):
                c_comp_arr[ii, jj] = 0

    comp_int = interpolate.SmoothBivariateSpline(c_mag_arr.flatten(),
                                                  c_col_arr.flatten(),
                                                  c_comp_arr.flatten(), s=200)

    return comp_int
    
    
    
def calc_mass_isWR_comp(mag, color, iso_mag_f, iso_col_f, iso_mass_f, iso_WR_f,
                         comp_int, mass_max):
    # F814W vs. F814W - F160W    
    # Loop through data and assign masses and completeness to each star.
    mass = np.zeros(len(mag), dtype=float)
    isWR = np.zeros(len(mag), dtype=float)
    comp = np.zeros(len(mag), dtype=float)
    
    for ii in range(len(mass)):
        dmag = mag[ii] - iso_mag_f
        dcol = color[ii] - iso_col_f

        delta = np.hypot(dmag, dcol)


        # Some funny business - sort and get the closest masses reasonable.
        sdx = delta.argsort()

        # If the color + mag difference is less than 0.15, then take
        # the lowest mass. This helps account for the missing IMF bias.
        idx = np.where(delta[sdx] < 0.01)[0]

        if len(idx) == 0:
            min_idx = delta.argmin()
            print 'Potential problem', mag[ii], color[ii], dmag[sdx[0]], dcol[sdx[0]]
        else:
            min_mass_idx = iso_mass_f[sdx[idx]].argmin()
            min_idx = sdx[idx][min_mass_idx]
            
        print '{0:4d} {1:4d} {2:4d} {3:5.1f} {4:5.1f}'.format(ii,
                                                              min_idx,
                                                              delta.argmin(),
                                                              iso_mass_f[min_idx],
                                                              iso_mass_f[delta.argmin()])

        mass[ii] = iso_mass_f[min_idx]
        isWR[ii] = iso_WR_f[min_idx]
        
        comp[ii] = comp_int(mag[ii], color[ii])

        if comp[ii] > 1:
            comp[ii] = 1
        if comp[ii] < 0:
            comp[ii] = 0

    print mag.min(), mag.max(), color.min(), color.max()
    print comp.shape, mag.shape, color.shape

    return mass, isWR, comp
    
    
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
    

    
