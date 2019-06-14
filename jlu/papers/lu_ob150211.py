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

    # Get the data and model
    data = munge_ob150211.getdata()

    fitter = model_fitter.PSPL_parallax_Solver(data, outputfiles_basename=pspl_ast_phot)
    fitter.load_mnest_results()
    mymodel = fitter.get_best_fit_model(use_median=True)

    # 1 day sampling over whole range
    # Find the first and last data date across both photometry and astrometry
    tmin = 56700 #np.min([data['t_ast'].min(),data['t_phot'].min()])
    tmax = np.max([data['t_ast'].max(),data['t_phot'].max()])
    t_mod = np.arange(tmin, tmax+100, 1)

    # Prep the colorbar                                                                                
    cmap = plt.cm.viridis
    norm = plt.Normalize(vmin=tmin, vmax=tmax)
    smap = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    smap.set_array([])

    # Get the astrometric model
    pos_out = mymodel.get_astrometry(t_mod)
    # Set (0,0) position at 17jun05 (MJD 57909)
    t0idx = np.where((t_mod>=57909) & (t_mod<57910))[0][0]
    xpos = (pos_out[:, 0] - pos_out[t0idx, 0])*-1e3
    ypos = (pos_out[:, 1] - pos_out[t0idx, 1])*1e3

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
    axf.plot(xpos/1e3, ypos/1e3)
    axf.set_xticks([])
    axf.set_yticks([])
    axf.set_aspect('equal')

    # Plot the motion on the sky
    axins = inset_axes(ax1, 1.05, 1)

    axins.scatter(xpos, ypos, c=t_mod, cmap=cmap, norm=norm, s=1)
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
    axins.plot([-1.8, 0.2], [5, 5], color=color1)
    axins.text(0.8, 5.4, '2 mas', color=color1, fontsize=14)

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
    # Photometry data limited to only the beginning of the magnification
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
