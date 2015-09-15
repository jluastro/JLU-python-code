import numpy as np
import pylab as py
from astropy.table import Table
from astropy.io import fits
import glob
import math
from hst_flystar import astrometry, photometry
from scipy import stats
from jlu.wd1.analysis import analysis_2015_01_05 as ana
import pdb

data_dir = '/Users/jlu/data/wd1/hst/reduce_2015_01_05/'
work_dir = '/Users/jlu/work/wd1/analysis_2015_01_05/'
cat_dir = '/Users/jlu/data/wd1/hst/reduce_2015_01_05/50.ALIGN_KS2/'

catalog = cat_dir + 'wd1_catalog_RMSE_wvelErr.fits'
cat_pclust_pcolor = work_dir + 'catalog_membership_3_rot_Pcolor.fits'

out_dir = '/Users/jlu/doc/papers/wd1_imf/'

angle = 46.43
mag_cut = 21

def make_plots():
    plot_astrometric_error()
    plot_velocity_error()
    plot_vel_chi2()
    plot_cmd_vpd_all()
    plot_completeness_vs_mag()

    return

def plot_astrometric_error():
    """
    Plot the astrometric error vs. magnitude.
    """
    cat = Table.read(catalog)

    mag_bins_F814W = np.append(np.arange(18, 20, 0.5), np.arange(20, 26, 0.25))
    mag_bins_nearIR = np.append(np.arange(14, 16, 1.0), np.arange(16, 24, 0.25))

    epochs = ['2010_F125W', '2010_F139M', '2010_F160W', '2013_F160W', '2005_F814W']
    n_epochs = len(epochs)
    
    py.close('all')

    # Setup Figure 1 (positional errors vs. mag)
    f1 = py.figure(3, figsize=(8, 12))
    py.clf()
    py.subplots_adjust(left=0.13, bottom=0.07, top=0.97, wspace=0.05, hspace=0.28)
    f1_ax1 = py.subplot(321)
    f1_ax2 = py.subplot(322)
    f1_ax3 = py.subplot(323)
    f1_ax4 = py.subplot(324)
    f1_ax5 = py.subplot(325)
    f1_ax6 = py.subplot(326)
    f1_sub = [f1_ax1, f1_ax2, f1_ax3, f1_ax4, f1_ax5, f1_ax6]

    # Setup Figure 2 (magnitude errors vs. mag)    
    f2 = py.figure(4, figsize=(8, 12))
    py.clf()
    py.subplots_adjust(left=0.13, bottom=0.07, top=0.97, wspace=0.05, hspace=0.28)
    f2_ax1 = py.subplot(321)
    f2_ax2 = py.subplot(322)
    f2_ax3 = py.subplot(323)
    f2_ax4 = py.subplot(324)
    f2_ax5 = py.subplot(325)
    f2_ax6 = py.subplot(326)
    f2_sub = [f2_ax1, f2_ax2, f2_ax3, f2_ax4, f2_ax5, f2_ax6]

    def median_trim(val):
        val_med = np.median(val)
        val_std = np.std(val)

        offset = (val - val_med) / val_std
            
        idx = np.where(offset < 5)

        if len(idx) > 0:
            val_med = np.median(val[idx])
        else:
            val_med = None

        return val_med
        
    
    for ee in range(n_epochs):
        if epochs[ee] == '2005_F814W':
            mag_bins = mag_bins_F814W
        else:
            mag_bins = mag_bins_nearIR

        filt = epochs[ee].split('_')[-1]
        year = epochs[ee].split('_')[0]
        
        m_all = cat['m_' + epochs[ee]]
        xe_all = cat['xe_' + epochs[ee]] * astrometry.scale['WFC'] * 1e3
        ye_all = cat['ye_' + epochs[ee]] * astrometry.scale['WFC'] * 1e3
        me_all = cat['me_' + epochs[ee]]

        # Sort everything by magnitude.
        sdx = np.argsort(m_all)

        # Calculate a running average with a window of
        # 50 points.
        window_size = 500
        window = np.ones((window_size, )) / window_size

        f_all = 10**(-m_all / 2.5)
        f_run_mean = np.convolve(f_all[sdx], window, mode='valid')
        m_run_mean = -2.5 * np.log10(f_run_mean)

        xe_run_mean = np.convolve(xe_all[sdx], window, mode='valid')
        ye_run_mean = np.convolve(ye_all[sdx], window, mode='valid')
        me_run_mean = np.convolve(me_all[sdx], window, mode='valid')

        xe_run_mean, mbe, bn = stats.binned_statistic(m_all, xe_all,
                                                      statistic=median_trim,
                                                      bins=mag_bins)
        ye_run_mean, mbe, bn = stats.binned_statistic(m_all, ye_all,
                                                      statistic=median_trim,
                                                      bins=mag_bins)
        me_run_mean, mbe, bn = stats.binned_statistic(m_all, me_all,
                                                      statistic=median_trim,
                                                      bins=mag_bins)
        m_run_mean = mag_bins[0:-1] + (np.diff(mag_bins) / 2.0)
        xye_run_mean = (xe_run_mean + ye_run_mean) / 2.0

        label = filt + ' ' + year
            
        # Plot positional errors for this epoch.
        f1_sub[ee].semilogy(m_all, xe_all, 'r.', label='X', ms=2, alpha=0.5)
        f1_sub[ee].semilogy(m_all, ye_all, 'b.', label='Y', ms=2, alpha=0.5)
        f1_sub[ee].semilogy(m_run_mean, xye_run_mean, 'k--', linewidth=3)
        f1_sub[ee].set_ylim(1e-1, 10)
        f1_sub[ee].set_xlabel(label + ' (mag)')

        # Aggregate pos err plot
        f1_sub[-1].semilogy(m_run_mean, xye_run_mean, label=label)

        # Plot photometric errors.
        f2_sub[ee].semilogy(m_all, me_all, 'k.', ms=2, alpha=0.5)
        f2_sub[ee].semilogy(m_run_mean, me_run_mean, 'r--', linewidth=3)
        f2_sub[ee].set_ylim(1e-3, 0.3)
        f2_sub[ee].set_xlabel(label + ' (mag)')

        # Aggregate phot err plot
        f2_sub[-1].semilogy(m_run_mean, me_run_mean, label=label)

        if epochs[ee] == '2005_F814W':
            f1_sub[ee].set_xlim(17.5, 26)
            f2_sub[ee].set_xlim(17.5, 26)
        else:
            f1_sub[ee].set_xlim(13, 22.5)
            f2_sub[ee].set_xlim(13, 22.5)

    # Add axis titles. Remove tick labels.
    f1_sub[0].legend(numpoints=1, loc='upper left', fontsize=12, markerscale=3)
    f1_sub[0].set_ylabel('Pos. Err. (mas)')
    f1_sub[2].set_ylabel('Pos. Err. (mas)')
    f1_sub[4].set_ylabel('Pos. Err. (mas)')
    f1_sub[1].set_yticklabels([])
    f1_sub[3].set_yticklabels([])
    f1_sub[5].set_yticklabels([])

    f2_sub[0].set_ylabel('Phot. Err. (mas)')
    f2_sub[2].set_ylabel('Phot. Err. (mas)')
    f2_sub[4].set_ylabel('Phot. Err. (mas)')
    f2_sub[1].set_yticklabels([])
    f2_sub[3].set_yticklabels([])
    f2_sub[5].set_yticklabels([])
    
    # Fix up aggregate plot
    # f1_sub[-1].set_ylim(1e-1, 10)
    # f1_sub[-1].set_xlim(14, 26)
    f1_sub[-1].set_xlabel('All Filters (mag)')
    f1_sub[-1].legend(loc='lower right', fontsize=11)
    
    f2_sub[-1].set_ylim(1e-3, 0.3)
    f2_sub[-1].set_xlim(14, 26)
    f2_sub[-1].set_xlabel('All Filters (mag)')
    f2_sub[-1].legend(loc='lower right', fontsize=11)

    f1.savefig(out_dir + 'pos_err_vs_mag.png')
    f2.savefig(out_dir + 'phot_err_vs_mag.png')

    return


def plot_velocity_error():
    """
    Plot the astrometric error vs. magnitude.
    """
    t = Table.read(catalog)
    
    mag_bins = np.append(np.arange(12, 16, 0.5), np.arange(16, 19, 0.25))
    
    epoch = '2013_F160W'
    m = t['m_' + epoch]
    me = t['me_' + epoch]
    vx = t['fit_vx'] * astrometry.scale['WFC'] * 1e3
    vy = t['fit_vy'] * astrometry.scale['WFC'] * 1e3
    vxe = t['fit_vxe'] * astrometry.scale['WFC'] * 1e3
    vye = t['fit_vye'] * astrometry.scale['WFC'] * 1e3

    v = np.hypot(vx, vy)
    ve = np.hypot(vxe * vx / v, vye * vy / v)

    py.close('all')
    py.figure()
    py.clf()
    py.semilogy(m, ve, 'k.', ms=3)
    py.ylabel('Proper Motion Error (mas/yr)')
    py.xlabel('Observed F160W Magnitude')
    py.ylim(5e-3, 5)
    py.xlim(12, 22)

    py.axhline(0.5, color='r', linestyle='--', linewidth=2)
    py.axvline(21, color='g', linestyle='-', linewidth=2)
    
    py.savefig(out_dir + 'velerr_vs_mag.png')

    # Calculate the median error for several different magnitude cuts.
    mag_cuts = np.arange(16, 21.1, 1)

    for ii in range(len(mag_cuts)):
        idx = np.where(m < mag_cuts[ii])[0]

        ve_med = np.median(ve[idx])

        fmt = 'Median Proper Motion Error for m < {0:.0f} = {1:4.2f} mas/yr'
        print fmt.format(mag_cuts[ii], ve_med)

    return
            
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
    
def print_number_stars():
    t = Table.read(catalog)

    epochs = ['2005_F814W', '2010_F125W', '2010_F139M', '2010_F160W', '2013_F160W',
              '2013_F160Ws']

    for ee in range(len(epochs)):
        idx = np.where(t['n_' + epochs[ee]] > 0)[0]

        # Faintest star detected.
        minMag = t['m_' + epochs[ee]].max()

        # Saturation level.
        satPoint = -11
        if 'F814W' in epochs[ee]:
            satPoint = -13.5 + 7.261
        satPoint += photometry.ZP[epochs[ee].split('_')[-1]]
            
        fmt = '{0:s}  {1:d}  faintest = {2:5.2f}, saturated = {3:5.2f}'
        print fmt.format(epochs[ee], len(idx), minMag, satPoint)

    return
    
def print_mean_residuals():
    epochs = ['2005_F814W', '2010_F125W', '2010_F139M', '2010_F160W', '2013_F160W',
              '2013_F160Ws']

    for ee in range(len(epochs)):
        trans_files = data_dir + '03.MAT_POS/' + epochs[ee]

        if ('F814W' in epochs[ee]) or ('F160Ws' in epochs[ee]):
            trans_files += '/01.XYM/TRANS.xym2mat.*refClust'
        else:
            trans_files += '_*/01.XYM/TRANS.xym2mat.*.refClust'

        t_files = glob.glob(trans_files)

        t_err_all = np.zeros(len(t_files), dtype=float)
    
        for tt in range(len(t_files)):
            trans = Table.read(t_files[tt], format='ascii')

            t_err_all[tt] = trans['col13'].mean()

        fmt = '{0:12s}  {1:6.3f} pix   {2:5.2f} mas'
        print fmt.format(epochs[ee], t_err_all.mean(),
                         t_err_all.mean() * astrometry.scale['WFC'] * 1e3)
        
    return
    
        
def plot_vel_chi2():
    ana.check_velocity_fits(magCut=mag_cut, outDir=out_dir)
    
    return

    
def plot_cmd_vpd_all():
    """
    Plot a three-panel CMD, CMD, VPD.
    """
    # Note that in this catalog, proper motions are already in
    # R.A. and Dec (mas/yr) units. But the positions need to
    # be rotated.
    d = Table.read(cat_pclust_pcolor)

    m160 = d['m_2013_F160W']
    m125 = d['m_2010_F125W']
    m139 = d['m_2010_F139M']
    m814 = d['m_2005_F814W']
    color1 = m814 - m160
    color2 = m125 - m160
    
    py.close(1)
    py.figure(1, figsize=(18,6))
    py.subplots_adjust(left=0.05, bottom=0.14, right=0.97, top=0.93,
                       wspace=0.25, hspace=0.3)
    
    # F814W vs. F160W CMD
    py.subplot(1, 3, 1)
    py.plot(d['m_2005_F814W'] - d['m_2013_F160W'], d['m_2005_F814W'], 'k.', ms=3)
    py.ylim(26, 14)
    py.xlim(0, 10)
    py.xlabel(r'm$_\mathrm{F814W}$ - m$_\mathrm{F160W}$')
    py.ylabel(r'm$_\mathrm{F814W}$')
    
    
    # F125W vs. F160W CMD
    py.subplot(1, 3, 2)
    py.plot(d['m_2010_F125W'] - d['m_2013_F160W'], d['m_2010_F125W'], 'k.', ms=3)
    py.ylim(22.5, 12)
    py.xlim(0, 2.5)
    py.xlabel(r'm$_\mathrm{F125W}$ - m$_\mathrm{F160W}$')
    py.ylabel(r'm$_\mathrm{F125W}$')

    # VPD
    vx = d['fit_vx']
    vy = d['fit_vy']
    py.subplot(1, 3, 3)
    py.plot(d['fit_vx'], d['fit_vy'], 'k.', ms=3)
    py.axis([6, -6, -6, 6])
    py.xlabel(r'$\mu_\alpha \cos \delta$ (mas yr$^{-1}$)')
    py.ylabel(r'$\mu_\delta$ mas yr$^{-1}$)')
    
    py.savefig(out_dir + 'wd1_cmd_vpd_all.png')

def plot_completeness_vs_mag():
    # Make a completeness curve for each filter independently.
    epochs = ['2005_F814W', '2010_F125W', '2010_F139M', '2010_F160W', '2013_F160W']
    
    comp = Table.read(work_dir + 'completeness_vs_mag.fits')
    
    # Plot
    py.clf()
    for ee in range(len(epochs)):
        ep = epochs[ee]
        py.errorbar(comp['mag'], comp['c_' + ep], yerr=comp['cerr_' + ep],
                    label=ep, drawstyle='steps-mid', linewidth=2)
        # py.plot(mag_bins_mid, comp[ep], 
        #             label=ep, drawstyle='steps-mid')
        
    py.ylim(0, 1)
    py.legend(loc='lower left')
    py.xlabel('Magnitude (mag)')
    py.ylabel('Completeness')
    py.savefig(out_dir + 'completeness_vs_mag.png')

    return
    
