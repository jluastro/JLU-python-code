import numpy as np
import pylab as plt
from astropy.io import fits
from astropy import stats
import pdb
import pickle
from jlu.m31 import ppxf_m31
from mpl_toolkits.axes_grid1 import ImageGrid

data_dir = '/u/kel/m31/data/osiris_mosaics/drf/sigclip/all_NIRC2_CC_DTOTOFF_2/no100828/'
cube_no_bulg_sub = data_dir + 'mosaic_all_nirc2_cc_dtotoff_no100828_scalederr.fits'
maps_no_bulg_sub = data_dir + 'ppxf_renormlosvd_nobs.dat'
maps_bulg_sub = data_dir + 'ppxf_renormlosvd_c11m_bs.dat'
maps_bulg_sub_tess = data_dir + 'ppxf_renormlosvd_c11m_bs_tess_sn40.dat'

errors_no_bulg_sub = data_dir + 'ppxf_errors_mc_nsim100_nobs.dat'
errors_bulg_sub = data_dir + 'ppxf_errors_mc_nsim100_c11m.dat'


ppxf_m31.workdir = '/Users/jlu/work/m31/'
plot_dir = ppxf_m31.workdir + 'plots/'


def plot_map_correlations(bulge_sub=True):
    if bulge_sub:
        maps_file = maps_bulg_sub
        out_suffix = '_bs'
    else:
        maps_file = maps_no_bulg_sub
        out_suffix = '_nobs'
        
    maps = ppxf_m31.PPXFresults(maps_file)

    ppxf_m31.plotResults3(maps_file, out_suffix=out_suffix)

    plt.figure(3)
    plt.clf()
    plot_vel_vs_h3(maps, out_suffix=out_suffix)

    return

def plot_vel_vs_h3(maps, out_suffix=''):
    xaxis = (np.arange(len(maps.velocity[0])) - ppxf_m31.bhpos_pix[0]) * 0.05
    yaxis = (np.arange(len(maps.velocity)) - ppxf_m31.bhpos_pix[1]) * 0.05
    yy, xx = np.meshgrid(xaxis, yaxis)
    radius = np.hypot(xx, yy)
    good = np.where((np.abs(yy) < 0.5) & (np.abs(xx) < 1.0))
        
    plt.scatter(maps.velocity[good], maps.h3[good], c=maps.sigma[good], s=5,
                    marker='o', vmin=0, vmax=450)
    plt.xlim(-700, 0)
    plt.ylim(-0.5, 0.5)
    plt.colorbar(label='Sigma (km/s)')
    plt.axhline(linestyle='--', color='grey')
    plt.xlabel('Velocity (km/s)')
    plt.ylabel('h3')
    plt.savefig(plot_dir + 'vel_vs_h3' + out_suffix + '.png')

    plt.clf()
    plt.scatter(maps.sigma[good], maps.h3[good], c=maps.velocity[good], s=5,
                    marker='o', vmin=-700, vmax=0)
    plt.xlim(0, 450)
    plt.ylim(-0.5, 0.5)
    plt.colorbar(label='Velocity (km/s)')
    plt.axhline(linestyle='--', color='grey')
    plt.xlabel('Sigma (km/s)')
    plt.ylabel('h3')
    plt.savefig(plot_dir + 'sig_vs_h3' + out_suffix + '.png')

    return

def plot_vs_vs_h3_paper():
    maps_bs = ppxf_m31.PPXFresults(maps_bulg_sub)
    maps_nobs = ppxf_m31.PPXFresults(maps_no_bulg_sub)
    errs_bs = ppxf_m31.PPXFresults(errors_bulg_sub)
    errs_nobs = ppxf_m31.PPXFresults(errors_no_bulg_sub)
    
    xaxis_bs = (np.arange(len(maps_bs.velocity[0])) - ppxf_m31.bhpos_pix[0]) * 0.05
    yaxis_bs = (np.arange(len(maps_bs.velocity)) - ppxf_m31.bhpos_pix[1]) * 0.05
    yy_bs, xx_bs = np.meshgrid(xaxis_bs, yaxis_bs)
    radius_bs = np.hypot(xx_bs, yy_bs)
    good_bs = np.where((np.abs(yy_bs) < 0.5) & (np.abs(xx_bs) < 1.0) & (errs_bs.velocity < 50))

    xaxis_nobs = (np.arange(len(maps_nobs.velocity[0])) - ppxf_m31.bhpos_pix[0]) * 0.05
    yaxis_nobs = (np.arange(len(maps_nobs.velocity)) - ppxf_m31.bhpos_pix[1]) * 0.05
    yy_nobs, xx_nobs = np.meshgrid(xaxis_nobs, yaxis_nobs)
    radius_nobs = np.hypot(xx_nobs, yy_nobs)
    good_nobs = np.where((np.abs(yy_nobs) < 0.5) & (np.abs(xx_nobs) < 1.0) & (errs_nobs.velocity < 50))

    plt.close(1)
    fig = plt.figure(1, figsize=(14, 6))
    plt.subplots_adjust(left=0.1, right=0.93)
    
    grid = ImageGrid(fig, 111,
                         nrows_ncols=(1,2),
                         axes_pad=0, share_all=False,
                         cbar_location="right", cbar_mode="single",
                         cbar_size="7%", cbar_pad=0.2, aspect=False)

    a0, b0, c0 = grid[0].errorbar(maps_nobs.velocity[good_nobs], maps_nobs.h3[good_nobs],
                                  xerr=errs_nobs.velocity[good_nobs], yerr=errs_nobs.h3[good_nobs], linestyle='none')
    sc0 = grid[0].scatter(maps_nobs.velocity[good_nobs], maps_nobs.h3[good_nobs], c=maps_nobs.sigma[good_nobs],
                          s=5, marker='o', vmin=0, vmax=450)
    grid[0].set_xlim(-650, -80)
    grid[0].set_ylim(-0.25, 0.25)
    grid[0].axhline(linestyle='--', color='grey')
    grid[0].set_xlabel('Velocity (km/s)')
    grid[0].set_ylabel('h3')
    grid[0].set_title('Not bulge subtracted')


    a1, b1, c1 = grid[1].errorbar(maps_bs.velocity[good_bs], maps_bs.h3[good_bs],
                                  xerr=errs_bs.velocity[good_bs], yerr=errs_bs.h3[good_bs], linestyle='none')
    sc1 = grid[1].scatter(maps_bs.velocity[good_bs], maps_bs.h3[good_bs], c=maps_bs.sigma[good_bs],
                          s=5, marker='o', vmin=0, vmax=450)
    grid[1].set_xlim(-650, -80)
    grid[1].set_ylim(-0.25, 0.25)
    grid[1].axhline(linestyle='--', color='grey')
    grid[1].set_xlabel('Velocity (km/s)')
    grid[1].set_ylabel('h3')
    grid[1].set_title('Bulge subtracted')
        

    cbar = grid[1].cax.colorbar(sc1)
    cax = grid.cbar_axes[0]
    axis = cax.axis[cax.orientation]
    axis.label.set_text('Sigma (km/s)')

    err_color1 = cbar.to_rgba(maps_nobs.sigma[good_nobs])
    err_color2 = cbar.to_rgba(maps_bs.sigma[good_bs])
    c0[0].set_color(err_color1)
    c1[0].set_color(err_color2)
    c0[1].set_color(err_color1)
    c1[1].set_color(err_color2)

    plt.savefig(plot_dir + 'vel_vs_h3_compare.png')
    plt.savefig(plot_dir + 'vel_vs_h3_compare.eps')


    return
    
    
