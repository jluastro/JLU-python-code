import numpy as np
import pylab as py
from astropy.table import Table
from jlu.util import statsIter
from hst_flystar import astrometry
from hst_flystar import completeness as comp
import os
import pdb
from jlu.wd1.analysis import membership
from scipy.stats import chi2
from scipy import interpolate
import matplotlib

reduce_dir = '/Users/jlu/data/wd1/hst/reduce_2015_01_05/'
cat_dir = reduce_dir + '50.ALIGN_KS2/'
art_dir = reduce_dir + '51.ALIGN_ART/'

work_dir = '/Users/jlu/work/wd1/analysis_2015_01_05/'
plot_dir = work_dir + 'plots/'

catalog = 'wd1_catalog_RMSE_wvelErr.fits'

# Catalogs after membership calculations
cat_pclust = work_dir + 'membership/gauss_3/catalog_membership_3_rot.fits'


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
    

def make_cmd(cl_prob=0.3):
    """
    Plot the total CMD and then the CMD of only cluster members.

    Parameters:
    
    """
    # Read in data table
    d = Table.read(cat_pclust)

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
    py.savefig(plot_dir + 'cmd_all.png')
    
    # Plot CMD with everything in black
    # and cluster members in red. 
    py.clf()
    py.plot(color, mag2, 'k.', ms=2)
    py.plot(color[clust], mag2[clust], 'r.', ms=2)
    py.ylabel('F814W (mag)')
    py.xlabel('F814W - F160W (mag)')
    py.gca().invert_yaxis()
    py.xlim(2, 10)
    py.savefig(plot_dir + 'cmd_all_clust.png')

    # Plot CMD with cluster members in black.
    py.clf()
    py.plot(color[clust], mag2[clust], 'k.', ms=2)
    py.ylabel('F814W (mag)')
    py.xlabel('F814W - F160W (mag)')
    py.gca().invert_yaxis()
    py.xlim(2, 10)
    py.savefig(plot_dir + 'cmd_clust.png')




        
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
    py.savefig(plot_dir + 'cmd_ir_clust.png')

    # IR Plot CMD with everything in black.
    py.clf()
    py.plot(color, mag2, 'k.', ms=2)
    py.ylabel('F125W (mag)')
    py.xlabel('F125W - F160W (mag)')
    py.xlim(0, 2)
    py.ylim(23, 14)
    py.savefig(plot_dir + 'cmd_ir_all.png')
    
    # IR Plot CMD with everything in black
    # and cluster members in red. 
    py.clf()
    py.plot(color, mag2, 'k.', ms=2)
    py.plot(color[clust], mag2[clust], 'r.', ms=2)
    py.ylabel('F125W (mag)')
    py.xlabel('F125W - F160W (mag)')
    py.xlim(0, 2)
    py.ylim(23, 14)
    py.savefig(plot_dir + 'cmd_ir_all_clust.png')




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
    py.savefig(plot_dir + 'cmd_2_clust.png')

    # F814W - F125W Plot CMD with everything in black.
    py.clf()
    py.plot(color, mag2, 'k.', ms=2)
    py.ylabel('F814W (mag)')
    py.xlabel('F814W - F125W (mag)')
    py.xlim(1, 7)
    py.ylim(26, 18)
    py.savefig(plot_dir + 'cmd_2_all.png')
    
    # F814W - F125W Plot CMD with everything in black
    # and cluster members in red. 
    py.clf()
    py.plot(color, mag2, 'k.', ms=2)
    py.plot(color[clust], mag2[clust], 'r.', ms=2)
    py.ylabel('F814W (mag)')
    py.xlabel('F814W - F125W (mag)')
    py.xlim(1, 7)
    py.ylim(26, 18)
    py.savefig(plot_dir + 'cmd_2_all_clust.png')



        
    
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
    py.savefig(plot_dir + 'cmd_hess_all.png')

    # Plot CMD Hess diagram of cluster members
    py.clf()
    py.hist2d(color[clust], mag2[clust], bins=(bins_col, bins_mag),
              cmap=py.cm.gist_heat_r)
    py.ylabel('F814W (mag)')
    py.xlabel('F814W - F160W (mag)')
    py.gca().invert_yaxis()
    py.xlim(2, 10)
    py.savefig(plot_dir + 'cmd_hess_clust.png')

            
            
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
    py.savefig(plot_dir + 'cmd_ir_hess_all.png')

    # IR: Plot CMD Hess diagram of cluster members.
    py.clf()
    py.hist2d(color[clust], mag2[clust], bins=(bins_col, bins_mag),
              cmap=py.cm.gist_heat_r)
    py.ylabel('F125W (mag)')
    py.xlabel('F125W - F160W (mag)')
    py.xlim(0, 2)
    py.ylim(23, 14)
    py.savefig(plot_dir + 'cmd_ir_hess_clust.png')

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

    points = np.vstack((color1_cl, m814_cl)).T
    idx = optical_patch.contains_points(points)

    pcolor[clust][idx] = 1

    d['Membership_color'] = pcolor

    # Finally, make a new catalog with only cluster members
    outfile = cat_pclust.replace('.fits', '_Pcolor.fits')
    print 'Writing: ', outfile
    d.write(outfile, format='fits', overwrite=True)

    return


    

    
    
    
    
    
