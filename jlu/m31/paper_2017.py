import numpy as np
import pylab as plt
from astropy.io import fits
from astropy import stats
import pdb
import pickle
from jlu.m31 import ppxf_m31

data_dir = '/u/kel/m31/data/osiris_mosaics/drf/sigclip/all_NIRC2_CC_DTOTOFF_2/no100828/'
cube_no_bulg_sub = data_dir + 'mosaic_all_nirc2_cc_dtotoff_no100828_scalederr.fits'
maps_no_bulg_sub = data_dir + 'ppxf_renormlosvd_nobs.dat'
maps_bulg_sub = data_dir + 'ppxf_renormlosvd_c11m_bs.dat'
maps_bulg_sub_tess = data_dir + 'ppxf_renormlosvd_c11m_bs_tess_sn40.dat'


ppxf_m31.workdir = '/Users/jlu/work/m31/'
plot_dir = ppxf_m31.workdir + 'plots/'

def plot_bulge_spaxel_spectra():
    """
    Make plots of spectra from different spaxels that are dominated by bulge light. 
    """
    cube = fits.getdata(cube_no_bulg_sub)
    bulge_ratio = fits.getdata(data_dir + 'bulge_sbratiomap.fits')

    # Clean our arrays of any spectra without much data. As a proxy, just use the 500th spectral element.
    # Note this changes the shape to a [SPAX NUMBER, SPEC CHANNEL] array (2D instead of 3D).
    have_data = np.where(cube[:, :, 500] > 0)
    cube_good = cube[have_data[0], have_data[1], :]
    bratio_good = bulge_ratio[have_data[0], have_data[1]]

    # Sort according to the bulge-to-disk ratio.
    sdx = np.argsort(bratio_good)
    cube_good = cube_good[sdx]
    bratio_good = bratio_good[sdx]

    # Make a histogram of the bulge ratios just so we know how many pixels we have in each.
    plt.figure(1)
    plt.clf()
    plt.hist(bratio_good, bins=np.arange(0.1, 0.55, 0.025))
    plt.xlabel('Bulge SF Ratio')
    plt.ylabel('Number of pixels')
    plt.show()
    plt.savefig('')

    idx_40 = np.where((bulge_ratio >= 0.40))
    idx_45 = np.where((bulge_ratio >= 0.45))
    idx_48 = np.where((bulge_ratio >= 0.48))

    spec_avg_40 = stats.sigma_clip(cube[idx_40], sigma=5, axis=0).mean(axis=0)
    spec_avg_45 = stats.sigma_clip(cube[idx_45], sigma=5, axis=0).mean(axis=0)
    spec_avg_48 = stats.sigma_clip(cube[idx_48], sigma=5, axis=0).mean(axis=0)
    pdb.set_trace()

    # Normalize the spectra to the same point:
    spec_avg_40 /= spec_avg_40[500:600].mean()
    spec_avg_45 /= spec_avg_45[500:600].mean()
    spec_avg_48 /= spec_avg_48[500:600].mean()

    plt.figure(2)
    plt.clf()
    plt.plot(spec_avg_40)
    plt.plot(spec_avg_45)
    plt.plot(spec_avg_48)
    

    

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
