import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import matplotlib.gridspec as gridspec
import numpy as np
from astropy.io import fits
from astropy.table import Table
from flystar import starlists
from mpl_toolkits.axes_grid1.inset_locator import (inset_axes, InsetPosition,
                                                  mark_inset)
from mpl_toolkits.axes_grid1.colorbar import colorbar
from microlens.jlu import model_fitter, multinest_utils, multinest_plot, munge_ob150211, model

paper_dir = '/u/jlu/doc/papers/ob150211/'

combo_dir = '/g/lu/data/microlens/17jun05/combo/'
lis_file = combo_dir + '/starfinder/mag17jun05_ob150211_kp_rms_named.lis'
astrom_data = '/u/jlu/work/microlens/OB150211/a_2019_05_04/ob150211_astrom_p3_2019_05_04.fits'
pspl_ast_phot = '/u/jlu/work/microlens/OB150211/a_2019_05_04/model_fits/4_fit_phot_astrom_parallax/bb_'

# Get the data and model
data = munge_ob150211.getdata()

fitter = model_fitter.PSPL_parallax_Solver(data, outputfiles_basename=pspl_ast_phot)
fitter.load_mnest_results()
mymodel = fitter.get_best_fit_model(use_median=True)

# 1 day sampling over whole range
# Find the first and last data date across both photometry and astrometry,
# and give both a 100 day window.
tast = np.arange(data['t_ast'].min()-100, data['t_ast'].max()+100, 1)
tphot = np.arange(data['t_phot'].min()-100, data['t_phot'].max()+100, 1)
tmin = np.min([tast.min(), tphot.min()])
tmax = np.max([tast.max(), tphot.max()])
t_mod = np.arange(tmin, tmax, 1)

# Prep the colorbar                                                                                
cmap = plt.cm.viridis
norm = plt.Normalize(vmin=tmin, vmax=tmax)
smap = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
smap.set_array([])

def mainplot():
    
    # Get the image and stars 
    img = fits.getdata(combo_dir + 'mag17jun05_ob150211_kp.fits')
    psf_file = '/g/lu/data/microlens/source_list/ob150211_psf.list'
    psf_tab = Table.read(psf_file, format='ascii', header_start=-1)
    pdx = np.where(psf_tab['PSF?']==1)[0]
    psf_tab = psf_tab[pdx]

    lis_tab = starlists.StarList.from_lis_file(lis_file)

    # Find the target and get its coordinates
    tdx = np.where(lis_tab['name']=='ob150211')[0]
    assert tdx is not None, "This list is missing ob150211"
    coo_targ = np.array([lis_tab['x'][tdx[0]], lis_tab['y'][tdx[0]]])
    coo_targ -= 1   # Shift to a 0-based array system

    scale = 0.00996
    x_axis = np.arange(img.shape[0], dtype=float)
    y_axis = np.arange(img.shape[1], dtype=float)
    x_axis = (x_axis - coo_targ[0]) * scale * -1.0
    y_axis = (y_axis - coo_targ[1]) * scale

    # Get the astrometric model
    pos_out = mymodel.get_astrometry(t_mod)
    # Set (0,0) position at 17jun05 (MJD 57909)
    t0idx = np.where((t_mod>=57909) & (t_mod<57910))[0][0]
    xpos = (pos_out[:, 0] - pos_out[t0idx, 0])*-1e3
    ypos = (pos_out[:, 1] - pos_out[t0idx, 1])*1e3
    xpos_ins = xpos[(t_mod>=tast.min()) & (t_mod<=tast.max())]
    ypos_ins = ypos[(t_mod>=tast.min()) & (t_mod<=tast.max())]

    # Get the photometric model
    mag = mymodel.get_photometry(t_mod)

    ## PLOT ##
    fig = plt.figure(figsize=(16,6))
    gs = gridspec.GridSpec(3, 2)

    # Set a reference frame. This should agree with the (0,0) date from above.
    refidx = 7

    # Plot the star field
    ax1 = fig.add_subplot(gs[:, 0])

    ax1.imshow(img, cmap='gist_heat_r', norm=LogNorm(12, 1e6), extent=[x_axis[0], x_axis[-1], y_axis[0], y_axis[-1]])

    # plt.axis('equal')
    ax1.set_xlim(-4,2)
    ax1.set_ylim(-4,2)
    ax1.set_xlabel(r'$\Delta \alpha^*$ (")')
    ax1.set_ylabel(r'$\Delta \delta$ (")')
    ax1.invert_xaxis()

    # Label the target
    color1 = "#7f0f18"
    ax1.plot([-0.8, -0.2], [0.9, 0.2], linestyle='-', color=color1)
    ax1.text(-0.8, 0.9, 'ob150211', fontsize=16, color=color1)

    # Fake inset axes to control the inset marking, hide its ticks
    axf = inset_axes(ax1, 1, 1)
    axf.plot(xpos_ins/1e3, ypos_ins/1e3)
    axf.set_xticks([])
    axf.set_yticks([])
    axf.set_aspect('equal')

    # Plot the motion on the sky
    axins = inset_axes(ax1, 1.05, 1)

    axins.scatter(xpos_ins, ypos_ins, c=t_mod[(t_mod>=tast.min()) & (t_mod<=tast.max())],
                  cmap=cmap, norm=norm, s=1)
    axins.errorbar((data['xpos'] - data['xpos'][refidx])*-1e3, (data['ypos'] - data['ypos'][refidx])*1e3,
                       xerr = data['xpos_err']*1e3, yerr=data['ypos_err']*1e3, fmt='.k')
    axins.set_xticks([],[])
    axins.set_yticks([],[])
    axins.invert_xaxis()
    # Enlarge the lims to create space for the points
    axins.set_xlim(axins.get_xlim()[0]+1, axins.get_xlim()[1]-1)
    axins.set_ylim(axins.get_ylim()[0]-1, axins.get_ylim()[1]+1)
    axins.set_aspect('equal')
    # Plot the scale in the inset
    axins.plot([-1.8, 0.2], [3, 3], color=color1)
    axins.text(0.5, 3.4, '2 mas', color=color1, fontsize=14)

    # Tweak the limits of the fake axes to fit the inset markers
    axf.set_xlim((axins.get_xlim()[0]+0.15)/1e3, (axins.get_xlim()[1]-1)/1e3)
    axf.set_ylim((axins.get_ylim()[0]-1)/1e3, (axins.get_ylim()[1]+1)/1e3)

    # Manually set the position and relative size of the inset axes within ax1
    ip = InsetPosition(ax1, [0.719,0,0.3,0.45])
    axf.set_axes_locator(ip)
    axins.set_axes_locator(ip)
    # Mark the region corresponding to the inset axes on the parenta axes
    # and draw lines in grey linking the two axes.
    mark_inset(parent_axes=ax1, inset_axes=axf, loc1=1, loc2=3, fc="none", ec='0.45')

    # Plot the photometry
    axp = fig.add_subplot(gs[0, 1])
    axp.scatter(t_mod, mag, c=t_mod, cmap=cmap, norm=norm, s=1)
    axp.errorbar(data['t_phot'][data['t_phot']>tmin], data['mag'][data['t_phot']>tmin],
                     yerr=data['mag_err'][data['t_phot']>tmin], fmt='k.')
    axp.invert_yaxis()
    axp.set_ylabel('I-band (mag)')
    axp.set_xticks([])

    # Plot the astrometry (ra and dec)
    axra = fig.add_subplot(gs[1, 1])
    axra.scatter(t_mod, xpos, c=t_mod, cmap=cmap, norm=norm, s=1)
    axra.errorbar(data['t_ast'], (data['xpos'] - data['xpos'][refidx])*-1e3, yerr=data['xpos_err']*1e3, fmt='k.')
    axra.set_ylabel(r'$\Delta \alpha^*$ (mas)')
    axra.set_xticks([])

    axdec = fig.add_subplot(gs[2, 1])
    axdec.scatter(t_mod, ypos, c=t_mod, cmap=cmap, norm=norm, s=1)
    axdec.errorbar(data['t_ast'], (data['ypos'] - data['ypos'][refidx])*1e3, yerr=data['ypos_err']*1e3, fmt='k.')
    axdec.set_ylabel(r'$\Delta \delta$ (mas)')
    axdec.set_xlabel('t (MJD)')

    fig.subplots_adjust(left=0.05, hspace=0, wspace=0.05)

    plt.savefig(paper_dir + 'ob150211_mainplot.pdf')
    plt.show()

def plot_onSky():
    # Model the unlensed motion
    dp_tmod_unlens = mymodel.get_astrometry(t_mod) - mymodel.get_astrometry_unlensed(t_mod)
    x_mod_no_pm = dp_tmod_unlens[:, 0]*-1
    y_mod_no_pm = dp_tmod_unlens[:, 1]

    # Make the unlensed model for the data points
    p_mod_unlens_tdat = mymodel.get_astrometry_unlensed(data['t_ast'])
    x_mod_tdat = p_mod_unlens_tdat[:, 0]
    y_mod_tdat = p_mod_unlens_tdat[:, 1]
    x_no_pm = (data['xpos'] - x_mod_tdat)*-1
    y_no_pm = data['ypos'] - y_mod_tdat

    # Model the unlensed astrometry at t_ast
    dp_tdat_unlens = mymodel.get_astrometry(data['t_ast']) - p_mod_unlens_tdat
    x_no_pm_tdat = (dp_tdat_unlens[:, 0])*-1
    y_no_pm_tdat = dp_tdat_unlens[:, 1]

    # PLOT
    plt.figure(figsize=(8,8))

    # definitions for the axes
    left, width = 0.13, 0.55
    bottom, height = 0.11, 0.55
    bottom_h = left_h = left + width + 0.04

    axSky = plt.axes([left, bottom, width, height])
    axdec = plt.axes([left, bottom_h - 0.02, width, 0.14])
    axra = plt.axes([left_h, bottom, 0.14, height])

    axSky.scatter(x_no_pm*1e3, y_no_pm*1e3, c=data['t_ast'],
                cmap=cmap, norm=norm, s=10)
    axSky.errorbar(x_no_pm*1e3, y_no_pm*1e3,
                 xerr=data['xpos_err']*1e3, yerr=data['ypos_err']*1e3,
                 fmt='none', ecolor=smap.to_rgba(data['t_ast']))
    axSky.scatter(x_mod_no_pm*1e3, y_mod_no_pm*1e3, c=t_mod, cmap=cmap, norm=norm, s=4)
    axSky.invert_xaxis()
    axSky.axis('equal')
    axSky.set_xlabel(r'$\Delta \alpha^*$ (mas)')
    axSky.set_ylabel(r'$\Delta \delta$ (mas)')

    axra.scatter((x_mod_no_pm - x_mod_no_pm), t_mod, c=t_mod, cmap=cmap, norm=norm, s=2)
    axra.errorbar((x_no_pm - x_no_pm_tdat)*1e3, data['t_ast'], xerr = data['xpos_err']*1e3,
                 fmt='k.', ecolor=smap.to_rgba(data['t_ast']))
    axra.set_ylabel('t (MJD)')
    axra.yaxis.set_label_position('right')
    axra.yaxis.tick_right()
    axra.yaxis.set_tick_params(rotation=-25)

    axdec.scatter(t_mod, (y_mod_no_pm - y_mod_no_pm), c=t_mod, cmap=cmap, norm=norm, s=2)
    axdec.errorbar(data['t_ast'], (y_no_pm - y_no_pm_tdat)*1e3, yerr = data['ypos_err']*1e3,
                 fmt='k.', ecolor=smap.to_rgba(data['t_ast']))
    axdec.set_xlabel('t (MJD)')
    axdec.xaxis.set_label_position('top')
    axdec.xaxis.tick_top()
    # axdec.locator_params(axis='x', nbins=6)
    axdec.xaxis.set_tick_params(rotation=25)

    plt.savefig(paper_dir + 'ob150029_onsky.pdf')
    plt.show()

def plot_obs():
    # Set the (0,0) point at 17jun05
    xcenter = data['xpos'][7]
    ycenter = data['ypos'][7]
    # Get the astrometric model
    pos_out = mymodel.get_astrometry(tast)
    xpos = (pos_out[:, 0] - xcenter)*-1e3
    ypos = (pos_out[:, 1] - ycenter)*1e3
    pos_tdat = mymodel.get_astrometry(data['t_ast'])
    xpos_tdat = (pos_tdat[:, 0] - xcenter)*-1e3
    ypos_tdat = (pos_tdat[:, 1] - ycenter)*1e3

    # Get the photometric model
    mag = mymodel.get_photometry(tphot)
    mag_tdat = mymodel.get_photometry(data['t_phot'])

    fig, ax = plt.subplots(2, 3, gridspec_kw={'height_ratios': [2, 1]}, figsize=(20,6))

    ax[0, 0].scatter(tphot, mag, c=tphot, cmap=cmap, norm=norm, s=4)
    ax[0, 0].errorbar(data['t_phot'], data['mag'],
                      yerr=data['mag_err'], fmt='k.')
    ax[0, 0].invert_yaxis()
    ax[0, 0].set_ylabel('I-band (mag)')
    ax[0, 0].set_xticks([])
    ax[1, 0].scatter(tphot, mag-mag, c=tphot, cmap=cmap, norm=norm, s=4)
    ax[1, 0].errorbar(data['t_phot'], (data['mag'] - mag_tdat),
                      yerr=data['mag_err'], fmt='k.')
    ax[1, 0].invert_yaxis()
    ax[1, 0].set_xlabel('time (MJD)')
    ax[1, 0].set_ylabel('residuals')
    
    ax[0, 1].scatter(tast, xpos, c=tast, cmap=cmap, norm=norm, s=4)
    ax[0, 1].errorbar(data['t_ast'], (data['xpos'] - xcenter)*-1e3, yerr=data['xpos_err']*1e3, fmt='k.')
    ax[0, 1].set_ylabel(r'$\Delta \alpha^*$ (mas)')
    ax[0, 1].set_xticks([])
    ax[1, 1].scatter(tast, xpos-xpos, c=tast, cmap=cmap, norm=norm, s=4)
    ax[1, 1].errorbar(data['t_ast'], (data['xpos'] - xcenter)*-1e3 - xpos_tdat,
                          yerr=data['xpos_err']*1e3, fmt='k.')
    ax[1, 1].set_xlabel('time (MJD)')
    ax[1, 1].locator_params(axis='x', nbins=7)
    ax[1, 1].set_ylabel('residuals')

    ax[0, 2].scatter(tast, ypos, c=tast, cmap=cmap, norm=norm, s=4)
    ax[0, 2].errorbar(data['t_ast'], (data['ypos'] - ycenter)*1e3, yerr=data['ypos_err']*1e3, fmt='k.')
    ax[0, 2].set_ylabel(r'$\Delta \delta$ (mas)')
    ax[0, 2].set_xticks([])
    ax[1, 2].scatter(tast, ypos-ypos, c=tast, cmap=cmap, norm=norm, s=4)
    ax[1, 2].errorbar(data['t_ast'], (data['ypos'] - ycenter)*1e3 - ypos_tdat,
                          yerr=data['ypos_err']*1e3, fmt='k.')
    ax[1, 2].set_xlabel('time (MJD)')
    ax[1, 2].locator_params(axis='x', nbins=7)
    ax[1, 2].set_ylabel('residuals')

    fig.subplots_adjust(left=0.08, hspace=0, wspace=0.35)
    
    plt.savefig(paper_dir + 'ob150211_data.pdf')
    plt.show()
