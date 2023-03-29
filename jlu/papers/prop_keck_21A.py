import numpy as np
import pylab as plt
import pdb
import math
import os
from jlu.observe import skycalc
from microlens.jlu import munge
# from microlens.jlu import residuals
from microlens.jlu import model_fitter, model
import shutil, os, sys
import scipy
import scipy.stats
# from gcwork import starset
# from gcwork import starTables
from astropy.table import Table
from jlu.util import fileUtil
from astropy.table import Table, Column, vstack
from astropy.io import fits
import matplotlib.ticker
import matplotlib.colors
from matplotlib.pylab import cm
from matplotlib.colors import Normalize, LogNorm
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable
from mpl_toolkits.axes_grid1.colorbar import colorbar
from matplotlib.ticker import NullFormatter
from microlens.jlu import model_fitter, multinest_utils, multinest_plot, munge_ob150211, munge_ob150029, model
from matplotlib.colors import LinearSegmentedColormap, colorConverter
import pdb
import pickle
from scipy.stats import norm
from jlu.util import datetimeUtil as dtUtil
from datetime import datetime as dt
import lu_2019_lens
import copy
from astropy.time import Time

# Default matplotlib color cycles.
mpl_b = '#1f77b4'
mpl_o = '#ff7f0e'
mpl_g = '#2ca02c'
mpl_r = '#d62728'

tdir = lu_2019_lens.pspl_ast_multiphot

mdir = '/u/jlu/work/microlens/'
# New targets directory
phot_2020_dir = {'kb200101' : mdir + 'KB200101/a_2020_09_10/model_fits/kmtnet_phot_par/a0_',
                 'mb19284' : mdir + '',
                 'ob190017' : mdir + 'OB190017/a_2020_09_10/model_fits/ogle_phot_par/a0_',
                 'ob170019' : mdir + 'OB170019/a_2020_09_10/model_fits/ogle_phot_par/a0_',
                 'ob170095' : mdir + 'OB170095/a_2020_09_10/model_fits/ogle_phot_par/a0_'}

def piE_tE(fit_type = 'ast'):
    """
    Supports plotting for several different fit solutions:

    fit_type = 'ast'
        Keck + OGLE photometry, Keck astrometry
    fit_type = 'phot'
        OGLE photometry
    fit_type = 'multiphot'
        Keck + OGLE photometry
    """
    if fit_type is 'ast':
        data_dict = lu_2019_lens.pspl_ast_multiphot
    if fit_type is 'phot':
        data_dict = lu_2019_lens.pspl_phot
    if fit_type is 'multiphot':
        data_dict = lu_2019_lens.pspl_multiphot
        

    ##########
    # !!! NOTE: CHOICE OF THE quantiles_2d HAS A LARGE EFFECT 
    # ON THE WAY THIS PLOT LOOKS !!!
    # Plot piE-tE 2D posteriors from OGLE photometry only fits.
    # Also plot PopSyCLE simulations simultaneously.
    ##########
    span = 0.999999426697
    smooth = 0.04
    quantiles_2d = None
    hist2d_kwargs = None
    labels = None
    label_kwargs = None
    show_titles = False 
    title_fmt = ".2f" 
    title_kwargs = None
    
    # Initialize values.
    if label_kwargs is None:
        label_kwargs = dict()
    if title_kwargs is None:
        title_kwargs = dict()
    if hist2d_kwargs is None:
        hist2d_kwargs = dict()

    # Dictionary of dictionaries containing the tE and piE
    # label position for each of the text labels.
    # WARNING: THE 'PHOT' AND 'AST' ARE THE SAME FOR THE NEW
    # TARGETS... THEY ARE JUST PHOT THOUGH.
    label_pos = {'phot': {'ob120169': [150, 0.01],
                          'ob140613': [170, 0.1],
                          'ob150029': [150, 0.2],
                          'ob150211': [35, 0.04],
                          'ob170019': [0, 0.04],
                          'ob170095': [0, 0.04],
                          'ob190017': [0, 0.04],
                          'kb200101': [0, 0]},
                 'ast':  {'ob120169': [50, 0.1],
                          'ob140613': [190, 0.15],
                          'ob150029': [55, 0.18],
                          'ob150211': [150, 0.008],
                          'ob170019': [140, 0.04],
                          'ob170095': [35, 0.02],
                          'ob190017': [200, 0.28],
                          'kb200101': [180, 0.02]}}

    label_pos_ast = {'ob120169': [0.006, 0.25],
                     'ob140613': [0.04, 0.1],
                     'ob150029': [0.04, 0.2],
                     'ob150211': [0.006, 0.012]}
                 
    colors = {'ob120169': 'purple',
              'ob140613': 'red',
              'ob150029': 'coral',
              'ob150211': 'darkorange',
              'ob170019': 'navy',
              'ob170095': 'brown',
              'ob190017': 'green',
              'kb200101': 'magenta'}

    # Set defaults.
    hist2d_kwargs['alpha'] = hist2d_kwargs.get('alpha', 0.2)
    hist2d_kwargs['levels'] = hist2d_kwargs.get('levels', quantiles_2d)

    targets = ['ob120169', 'ob140613', 'ob150029', 'ob150211'] 
    new_targets = ['ob170019', 'ob170095', 'ob190017', 'kb200101']
    tE = {}
    piE = {}
    theta_E = {}
    weights = {}

    for targ in targets:
        fit_targ, dat_targ = lu_2019_lens.get_data_and_fitter(data_dict[targ])
        
        res_targ = fit_targ.load_mnest_modes()
        smy_targ = fit_targ.load_mnest_summary()

        # Get rid of the global mode in the summary table.
        smy_targ = smy_targ[1:]

        # Find which solution has the max likelihood.
        mdx = smy_targ['maxlogL'].argmax()
        res_targ = res_targ[mdx]
        smy_targ = smy_targ[mdx]

        tE[targ] = res_targ['tE']
        piE[targ] = np.hypot(res_targ['piE_E'], res_targ['piE_N'])
        theta_E[targ] = res_targ['thetaE']
        weights[targ] = res_targ['weights']

    for targ in new_targets:
        fit_targ, dat_targ = lu_2019_lens.get_data_and_fitter(phot_2020_dir[targ])
        
        res_targ = fit_targ.load_mnest_modes()
        smy_targ = fit_targ.load_mnest_summary()

        # Get rid of the global mode in the summary table.
        smy_targ = smy_targ[1:]

        # Find which solution has the max likelihood.
        mdx = smy_targ['maxlogL'].argmax()
        res_targ = res_targ[0]
        smy_targ = smy_targ
        tE[targ] = res_targ['tE']
        piE[targ] = np.hypot(res_targ['piE_E'], res_targ['piE_N'])
        weights[targ] = res_targ['weights']

    # Plot the piE-tE 2D posteriors.
    plt.close(1)
    fig = plt.figure(1, figsize=(6,6))
    plt.clf()
    axes = plt.gca()
    plt.subplots_adjust(bottom=0.15)

    sx = smooth
    sy = smooth

    hist2d_kwargs['fill_contours'] = hist2d_kwargs.get('fill_contours', False)
    hist2d_kwargs['plot_contours'] = hist2d_kwargs.get('plot_contours', True)

    for targ in targets + new_targets:
        model_fitter.contour2d_alpha(tE[targ], piE[targ], span=[span, span], quantiles_2d=quantiles_2d,
                                 weights=weights[targ], ax=axes, smooth=[sy, sx], color=colors[targ],
                                 **hist2d_kwargs, plot_density=False)

        axes.text(label_pos[fit_type][targ][0], label_pos[fit_type][targ][1],
                      targ.upper(), color=colors[targ])    

    # OB110022 from Lu+16.
    piEE_110022 = -0.393
    piEN_110022 = -0.071
    piE_110022 = np.hypot(piEE_110022, piEN_110022)
    tE_110022 = 61.4

    dcmax_110022 = 2.19/np.sqrt(8)
    dcmax_110022_pe = 1.06/np.sqrt(8)
    dcmax_110022_me = 1.17/np.sqrt(8)

    # MB19284 from Dave Bennett.
    piEE_19284 = -0.06339
    piEN_19284 = 0.05600
    piE_19284 = np.hypot(piEE_19284, piEN_19284)
    tE_19284 = 648.454

    # Plotting OB110022 and MB19284.
    plt.scatter(tE_110022, piE_110022, marker = 'o', s = 30, color='indigo')
    axes.text(18, 0.38, 'OB110022', color='indigo')

#    plt.scatter(tE_19284, piE_19284, marker = 'o', s = 30, color='lime')
#    axes.text(300, 0.1, 'MB19284', color='lime')

    # Add the PopSyCLE simulation points.
    # NEED TO UPDATE THIS WITH BUGFIX IN DELTAM
    t = Table.read('/u/casey/scratch/papers/microlens_2019/popsycle_rr_files/Mock_EWS_v2.fits') 

    bh_idx = np.where(t['rem_id_L'] == 103)[0]
    ns_idx = np.where(t['rem_id_L'] == 102)[0]
    wd_idx = np.where(t['rem_id_L'] == 101)[0]
    st_idx = np.where(t['rem_id_L'] == 0)[0]

    u0_arr = t['u0']
    thetaE_arr = t['theta_E']
    
    # Stores the maximum astrometric shift
    final_delta_arr = np.zeros(len(u0_arr))
    
    # Stores the lens-source separation corresponding
    # to the maximum astrometric shift
    final_u_arr = np.zeros(len(u0_arr))

    # Sort by whether the maximum astrometric shift happens
    # before or after the maximum photometric amplification
    big_idx = np.where(u0_arr > np.sqrt(2))[0]
    small_idx = np.where(u0_arr <= np.sqrt(2))[0]

    # Flux ratio of lens to source (and make it 0 if dark lens)
    g_arr = 10**(-0.4 * (t['ubv_i_app_L'] - t['ubv_i_app_S']))
    g_arr = np.nan_to_num(g_arr)

    for i in np.arange(len(u0_arr)):
        g = g_arr[i] 
        thetaE = thetaE_arr[i]    
        # Try all values between u0 and sqrt(2) to find max 
        # astrometric shift
        if u0_arr[i] < np.sqrt(2):
            u_arr = np.linspace(u0_arr[i], np.sqrt(2), 100)
            delta_arr = np.zeros(len(u_arr))
            for j in np.arange(len(u_arr)):
                u = u_arr[j] 
                numer = 1 + g * (u**2 - u * np.sqrt(u**2 + 4) + 3)
                denom = u**2 + 2 + g * u * np.sqrt(u**2 + 4)
                delta = (u * thetaE/(1 + g)) * (numer/denom)
                delta_arr[j] = delta
            max_idx = np.argmax(delta_arr)
            final_delta_arr[i] = delta_arr[max_idx]
            final_u_arr[i] = u_arr[max_idx]
        # Maximum astrometric shift will occur at sqrt(2)
        if u0_arr[i] > np.sqrt(2):
            u = u0_arr[i]
            numer = 1 + g * (u**2 - u * np.sqrt(u**2 + 4) + 3)
            denom = u**2 + 2 + g * u * np.sqrt(u**2 + 4)
            delta = (u * thetaE/(1 + g)) * (numer/denom)
            final_delta_arr[i] = delta
            final_u_arr[i] = u

    axes.scatter(t['t_E'][st_idx], t['pi_E'][st_idx], 
                 alpha = 0.4, marker = '.', s = 25, 
                 color = 'paleturquoise')
    axes.scatter(t['t_E'][wd_idx], t['pi_E'][wd_idx], 
                 alpha = 0.4, marker = '.', s = 25, 
                 color = 'aqua')
    axes.scatter(t['t_E'][ns_idx], t['pi_E'][ns_idx], 
                 alpha = 0.4, marker = '.', s = 25, 
                 color = 'blue')
    axes.scatter(t['t_E'][bh_idx], t['pi_E'][bh_idx],
                 alpha = 0.4, marker = '.', s = 25, 
                 color = 'dimgray')

    # Trickery to make the legend darker
    axes.scatter(0.01, 100, 
                 alpha = 0.8, marker = '.', s = 25, 
                 label = 'Star', color = 'paleturquoise')
    axes.scatter(0.01, 100, 
                 alpha = 0.8, marker = '.', s = 25,
                 label = 'WD', color = 'aqua')
    axes.scatter(0.01, 100,
                 alpha = 0.8, marker = '.', s = 25, 
                 label = 'NS', color = 'blue')
    axes.scatter(0.01, 100,
                 alpha = 0.8, marker = '.', s = 25, 
                 label = 'BH', color = 'dimgray')

    axes.set_xlim(10, 700)
    axes.set_ylim(0.005, 0.5)
    axes.set_xlabel('$t_E$ (days)')
    axes.set_ylabel('$\pi_E$')
    axes.set_xscale('log')
    axes.set_yscale('log')
    axes.legend(loc=3)
    plt.savefig('piE_tE_' + fit_type + '.png')
    plt.show()

    # Plot the deltac-piE 2D posteriors.
    plt.close(1)
    fig = plt.figure(1, figsize=(6,6))
    plt.clf()
    axes = plt.gca()
    plt.subplots_adjust(bottom=0.15)

    axes.errorbar(dcmax_110022, piE_110022, 
                   xerr = np.array([[dcmax_110022_me], [dcmax_110022_pe]]), 
                   fmt = 'o', color = 'indigo', markersize = 5,
                   xuplims = True)
    axes.text(0.5, 0.3, 'OB110022', color='indigo')

    for targ in targets:
        model_fitter.contour2d_alpha(theta_E[targ]/np.sqrt(8), piE[targ], span=[span, span], quantiles_2d=quantiles_2d,
                                 weights=weights[targ], ax=axes, smooth=[sy, sx], color=colors[targ],
                                 **hist2d_kwargs, plot_density=False)


        axes.text(label_pos_ast[targ][0], label_pos_ast[targ][1],
                  targ.upper(), color=colors[targ])    

    axes.scatter(final_delta_arr[st_idx], t['pi_E'][st_idx], 
                  alpha = 0.4, marker = '.', s = 25,
                  c = 'paleturquoise')
    axes.scatter(final_delta_arr[wd_idx], t['pi_E'][wd_idx], 
                  alpha = 0.4, marker = '.', s = 25,
                  c = 'aqua')
    axes.scatter(final_delta_arr[ns_idx], t['pi_E'][ns_idx], 
                  alpha = 0.4, marker = '.', s = 25,
                  c = 'blue')
    axes.scatter(final_delta_arr[bh_idx], t['pi_E'][bh_idx], 
                  alpha = 0.8, marker = '.', s = 25,
                  c = 'dimgray')

    axes.set_xlabel('$\delta_{c,max}$ (mas)')
    axes.set_ylabel('$\pi_E$')
    axes.set_xscale('log')
    axes.set_yscale('log')
    axes.set_xlim(0.005, 4)
    axes.set_ylim(0.009, 0.5)
    plt.savefig('piE_deltac.png')
    plt.show()


def deltac_vs_time_BH15_targets(target, t_obs_prop=None, dtmax=None):
    """
    target : 'ob120169', 'ob140613', 'ob150029', 'ob150211'
    """
    target_name = copy.deepcopy(target)
    target_name = target_name.replace('OB', 'ob')
    print(target_name)
    mod_fit, data = lu_2019_lens.get_data_and_fitter(tdir[target_name])
    mod_all = mod_fit.get_best_fit_modes_model(def_best = 'maxL')
    mod = mod_all[0]

    # Sample time
    tmax = np.max(np.append(data['t_phot1'], data['t_phot2'])) + 90.0
    if dtmax is not None:
        tmax += dtmax
    t_mod_ast = np.arange(data['t_ast'].min() - 180.0, tmax, 2)
    t_mod_pho = np.arange(data['t_phot1'].min(), tmax, 2)

    # Get the linear motion curves for the source (includes parallax)
    p_unlens_mod = mod.get_astrometry_unlensed(t_mod_ast)
    p_unlens_mod_at_ast = mod.get_astrometry_unlensed(data['t_ast'])

    # Get the lensed motion curves for the source
    p_lens_mod = mod.get_astrometry(t_mod_ast)
    p_lens_mod_at_ast = mod.get_astrometry(data['t_ast'])

    x = (data['xpos'] - p_unlens_mod_at_ast[:,0]) * -1e3
    xe = data['xpos_err'] * 1e3
    y = (data['ypos'] - p_unlens_mod_at_ast[:, 1]) * 1e3
    ye = data['ypos_err'] * 1e3

    r2 = x**2 + y**2
    r = np.sqrt(r2)

    xterm = (x * xe)**2/r2
    yterm = (y * ye)**2/r2
    re = np.sqrt(xterm + yterm)

    xmod = (p_lens_mod[:, 0] - p_unlens_mod[:, 0])*-1e3
    ymod = (p_lens_mod[:, 1] - p_unlens_mod[:, 1])*1e3

    # Convert to decimal dates
    t_ast_dec = Time(data['t_ast'], format='mjd', scale='utc')
    t_mod_ast_dec = Time(t_mod_ast, format='mjd', scale='utc')

    t_ast_dec.format='decimalyear'
    t_mod_ast_dec.format='decimalyear'

    t_ast_dec = t_ast_dec.value
    t_mod_ast_dec = t_mod_ast_dec.value
    
    if t_obs_prop is not None:
        # Turn the epoch YYYY-MM-DD into a decimal.
        t_obs_prop_dec = np.zeros(len(t_obs_prop))
        for idx, tt in enumerate(t_obs_prop):
            t_strp = dt.strptime(tt, '%Y-%m-%d')
            t_dec = dtUtil.toYearFraction(t_strp)
            t_obs_prop_dec[idx] = t_dec

    plt.figure(1)
    plt.clf()
    plt.errorbar(t_ast_dec, x, yerr=xe, fmt='k.', alpha=1, zorder = 1000, label='Data')
    plt.scatter(t_mod_ast_dec, xmod, s = 1)
    plt.title(target)
    plt.xlabel('Time (Year)')
    plt.ylabel('$\delta$RA (mas)')
    if t_obs_prop is not None:
        for obs in t_obs_prop_dec:
            plt.axvline(x = obs, color='red', ls=':')
        plt.axvline(x = 0, color='red', ls=':', label='Proposed')
        plt.xlim(t_mod_ast_dec.min() - 0.1, t_mod_ast_dec.max() + 0.1)
    plt.axhline(y=0, color='black', alpha=0.5)
    plt.legend(loc=1)
    plt.title(target)
    plt.savefig(target + '_deltac_RA_vs_time.png')
    plt.show()

    plt.figure(2)
    plt.clf()
    plt.errorbar(t_ast_dec, y, yerr=ye, fmt='k.', alpha=1, zorder = 1000, label='Data')
    plt.scatter(t_mod_ast_dec, ymod, s = 1)
    plt.title(target)
    plt.xlabel('Time (Year)')
    plt.ylabel('$\delta$Dec (mas)')
    if t_obs_prop is not None:
        for obs in t_obs_prop_dec:
            plt.axvline(x = obs, color='red', ls=':')
        plt.axvline(x = 0, color='red', ls=':', label='Proposed')
        plt.xlim(t_mod_ast_dec.min() - 0.1, t_mod_ast_dec.max() + 0.1)
    plt.axhline(y=0, color='black', alpha=0.5)
    plt.legend(loc=1)
    plt.title(target)
    plt.savefig(target + '_deltac_Dec_vs_time.png')
    plt.show()

    plt.figure(3)
    plt.clf()
    plt.errorbar(t_ast_dec, r, yerr=re, fmt='k.', alpha=1, zorder = 1000, label='Data')
    plt.scatter(t_mod_ast_dec, np.sqrt(xmod**2 + ymod**2), s = 1)
    plt.xlabel('Time (Year)')
    plt.ylabel('$\delta_c$ (mas)')
    if t_obs_prop is not None:
        for obs in t_obs_prop_dec:
            plt.axvline(x = obs, color='red', ls=':')
        plt.axvline(x = 0, color='red', ls=':', label='Proposed')
        plt.xlim(t_mod_ast_dec.min() - 0.1, t_mod_ast_dec.max() + 0.1)
    plt.gca().set_ylim(bottom=0)
    plt.axhline(y=0.15, color='gray', ls='--')
    plt.legend(loc=1)
    plt.title(target)
    plt.savefig(target + '_deltac_vs_time.png')
    plt.show()

def make_plots_BH15():
#    deltac_vs_time_BH15_targets('OB120169')
    deltac_vs_time_BH15_targets('OB140613',
                                t_obs_prop=['2021-04-01', '2021-07-30'],
                                dtmax=1000)
#    deltac_vs_time_BH15_targets('OB150029')
#    deltac_vs_time_BH15_targets('OB150211')

def make_plots():
    deltac_vs_time(0.061, 200/365.25, 4, '2020-06-22', 
                   ['2020-07-12', '2020-07-22', '2020-08-22', '2020-09-03'],
                   t_obs_prop=['2021-04-01', '2021-06-01',  '2021-07-30'],
                   title='KB200101')

    deltac_vs_time(0.18, 258.6/365.25, 4, '2019-04-13', 
                   ['2019-04-21', '2019-05-13', '2020-07-22', '2020-08-23', '2020-09-03'],
                   t_obs_prop=['2021-04-01', '2021-06-01', '2021-07-30'],
                   title='OB190017')

    deltac_vs_time(0.03, 105.9/365.25, 5, '2017-05-19', 
                   ['2017-05-21', '2017-06-08', '2017-07-14', '2017-07-19', 
                    '2020-06-13', '2020-08-23', '2020-09-04'],
                   t_obs_prop=['2021-04-01', '2021-07-30'],
                   title='OB170095')

    deltac_vs_time(0.165, 648.454/365.25, 5, '2020-12-23', 
                   ['2020-06-25', '2020-07-12', '2020-07-22', '2020-08-22'], 
                   t_obs_prop=['2021-04-01', '2021-06-01',  '2021-07-30'],
                   title='MB19284')

#    # u0 > 0, Keck + OGLE values.
#    deltac_vs_time(0.45, 128.3/365.25, 7, '2015-07-09', 
#                   ['2015-05-05', '2015-06-07', '2015-06-29', '2015-07-24',
#                    '2016-05-02', '2016-07-14', '2016-08-02', '2017-06-05',
#                    '2017-06-08', '2017-07-19', '2018-05-11', '2018-08-02',
#                    '2018-08-16'],
#                   title='OB150211')        
#
#    # Keck + OGLE values.
#    deltac_vs_time(0.11, 386.5/365.25, 8, '2015-06-22', 
#                   ['2015-06-07', '2015-06-29', '2016-04-17', '2016-05-24',
#                    '2016-08-02', '2017-06-05' ,'2017-07-14', '2018-05-11',
#                    '2018-08-16', '2019-04-17'],
#                   title='OB140613')        
#
#    # Keck + OGLE values.
#    deltac_vs_time(1E-4, 168.0/365.25, 10, '2012-04-01', 
#                   ['2012-06-23', '2012-07-10', '2013-04-30', '2013-07-15',
#                    '2015-05-05', '2015-06-07', '2016-05-24', '2016-07-14'],
#                   title='OB120169')
#
#    # Keck + OGLE values.
#    deltac_vs_time(0.90, 118.2/365.25, 7, '2015-08-12', 
#                   ['2015-06-07', '2015-07-23', '2016-05-24' ,'2016-07-14',
#                    '2017-05-21' ,'2017-07-14', '2017-07-19', '2018-08-21',
#                    '2019-04-17'],
#                   title='OB150029')        


def deltac_vs_time(u0, tE, dt_max, t0, t_obs, t_obs_prop=None, title=''):
    """
    u0 : dim'less impact parameter
    tE : in years
    t0 : str, in YYYY-MM-DD format
    t_obs : list of dates in str  YYYY-MM-DD format.
    dt_max : maximum time after photometric peak 
    """
    t0_strp = dt.strptime(t0, '%Y-%m-%d')
    t0_dec = dtUtil.toYearFraction(t0_strp)
    t_arr = np.linspace(t0_dec - 0.1, t0_dec + dt_max, 200) 
    if title=='MB19284':
        t_arr = np.linspace(t0_dec - 1, t0_dec + dt_max, 200) 

    # Turn the epoch YYYY-MM-DD into a decimal.
    t_obs_dec = np.zeros(len(t_obs))
    for idx, tt in enumerate(t_obs):
        t_strp = dt.strptime(tt, '%Y-%m-%d')
        t_dec = dtUtil.toYearFraction(t_strp)
        t_obs_dec[idx] = t_dec

    if t_obs_prop is not None:
        # Turn the epoch YYYY-MM-DD into a decimal.
        t_obs_prop_dec = np.zeros(len(t_obs_prop))
        for idx, tt in enumerate(t_obs_prop):
            t_strp = dt.strptime(tt, '%Y-%m-%d')
            t_dec = dtUtil.toYearFraction(t_strp)
            t_obs_prop_dec[idx] = t_dec
            
    u = np.sqrt(u0**2 + ((t_arr - t0_dec)/tE)**2)
    deltac = u/(u**2 + 2)

    fig, ax = plt.subplots()
    ax.plot(t_arr, deltac)
    ax.set_xlabel('Time (Year)')
    ax.set_ylabel(r'$\delta_c$/$\theta_E$')
    for obs in t_obs_dec:
        ax.axvline(x = obs, color='black')
    ax.axvline(x = 0, color='black', label='Completed')
    if t_obs_prop is not None:
        for obs in t_obs_prop_dec:
            ax.axvline(x = obs, color='red', ls=':')
        ax.axvline(x = 0, color='red', ls=':', label='Proposed')
    ax.set_xlim(t_arr[0], t_arr[-1])
    ax.legend()
    ax.set_ylim(bottom=0)
    ax.set_title(title)
    plt.savefig(title + '_deltac_vs_time.png')
    plt.show()

    return


def reduction_table():
    output_dir = ''
    table_file = output_dir + 'osiris_2020.txt'
    if os.path.exists(table_file):
        os.remove(table_file)

    ddir = '/g/lu/data/microlens/'
    log_names = ['20jun13os/combo/mag20jun13os_ob170095_kp_tdHband.log',
                 '20jun25os/combo/mag20jun25os_mb19284_kp_tdOpen.log',
                 '20jul12os/combo/mag20jul12os_kb200101_kp_tdOpen.log',
                 '20jul12os/combo/mag20jul12os_mb19284_kp_tdOpen.log',
                 '20jul14os/combo/mag20jul14os_kb200101_kp_tdOpen.log',
                 '20jul22os/combo/mag20jul22os_kb200101_kp_tdHband.log',
                 '20jul22os/combo/mag20jul22os_kb200101_kp_tdOpen.log',
                 '20jul22os/combo/mag20jul22os_kb200122_kp_tdOpen.log',
                 '20jul22os/combo/mag20jul22os_mb19284_kp_tdOpen.log',
                 '20jul22os/combo/mag20jul22os_ob190017_kp_tdOpen.log',
                 '20aug22os/combo/mag20aug22os_kb200101_kp_tdOpen.log',
                 '20aug22os/combo/mag20aug22os_mb19284_kp_tdOpen.log',
                 '20aug23os/combo/mag20aug23os_ob170095_l_kp_tdOpen.log',
                 '20aug23os/combo/mag20aug23os_ob170095_s_kp_tdOpen.log',
                 '20aug23os/combo/mag20aug23os_ob190017_kp_tdHband.log',
                 '20aug23os/combo/mag20aug23os_ob190017_kp_tdOpen.log',
                 '20sep03os/combo/mag20sep03os_kb200101_kp_tdOpen.log',
                 '20sep03os/combo/mag20sep03os_ob190017_kp_tdOpen.log',
                 '20sep04os/combo/mag20sep04os_ob110462_kp_tdOpen.log',
                 '20sep04os/combo/mag20sep04os_ob170095_kp_tdOpen.log']

    with open(table_file, 'a+') as tb:
        for log_name in log_names:
            t = Table.read(ddir + log_name, format='ascii')

            cfile = t['col1']
            fwhm = t['col2']
            strehl = t['col3']

            foo = log_name.split('/')
            bar = log_name.split('_', 1)

            hdul = fits.open(ddir + foo[0] + '/raw/' +  cfile[0][1:] + '.fits')
            hdr = hdul[0].header
                                    
            idx = np.where(fwhm > 0)[0]
            avg_fwhm = np.average(fwhm[idx])
            avg_strehl = np.average(strehl[idx])
            # Calculate total integration time, in minutes
            tint_tot = (hdr['truitime'] * hdr['coadds'] * len(idx))/60

            tb.write(foo[0] + ' & ' + 
                     bar[1][:-4] + ' & ' + 
                     '{:.2f}'.format(tint_tot) + ' & ' + 
                     str(len(idx)) + ' & ' + 
                     '{:.0f}'.format(avg_fwhm) + ' & ' + 
                     '{:.2f}'.format(avg_strehl) + r' \\ ' + '\n')

def plot_how_many():
    """
    How many BHs needed to detect in order to constrain
    number in MW to sigma/N
    """
    def tick_function(old, conversion):
        """
        Tick marks for the double axes
        """
        new = old * conversion
        return ["%.0f" % z for z in new]

    t = Table.read('/u/casey/scratch/papers/microlens_2019/popsycle_rr_files/Mock_EWS_v2.fits')

    n_long = len(np.where(t['t_E'] > 120)[0])
    n_long_bh = len(np.where((t['t_E'] > 120) & (t['rem_id_L'] == 103))[0])

    cf = n_long/n_long_bh
    
    NBH = 20

    N_detect = np.linspace(0, NBH, 1000)
    N_sigma = np.sqrt(N_detect)
    
    fig = plt.figure(12, figsize=(6,5))
    plt.clf()
    plt.subplots_adjust(left = 0.17, top = 0.8, bottom = 0.2)
    ax1 = fig.add_subplot(111)
    ax2 = ax1.twiny()
    
    ax1.plot(N_detect, N_sigma/N_detect, lw = 3)
    ax1.set_xlabel('$N_{BH}$ events observed')
    ax1.set_xticks(np.arange(N_detect[0], N_detect[-1] + 1, NBH/5))
    
    new_tick_locations = np.linspace(0, NBH, 6)
    
    ax2.set_xlim(ax1.get_xlim())
    ax2.set_xticks(new_tick_locations)
    ax2.set_xticklabels(tick_function(new_tick_locations, cf))
    ax2.set_xlabel("N events observed per year")
    
    ax1.set_ylabel('$\sigma_{N_{BH}}/N_{BH}$')
    plt.ylim(0, 1)
    fig.patch.set_facecolor('white')
    plt.show()

    return

def tE_BH_move():
    """
    Plot PopSyCLE tE distributions for two 
    different BH kick velocities.
    """
    # Fiducial model (BH kick = 100 km/s)
    t = Table.read('/u/casey/scratch/papers/microlens_2019/popsycle_rr_files/Mock_EWS_v2.fits')
    
    bh_idx = np.where(t['rem_id_L'] == 103)[0] # BHs 
    not_bh_idx = np.where(t['rem_id_L'] != 103)[0] # Not BHs 
    long_idx = np.where(t['t_E'] > 120)[0] # tE > 120 day events
    long_bh_idx = np.where((t['t_E'] > 120) & 
                           (t['rem_id_L'] == 103))[0] # tE > 120 events that are BHs

    long_bh_frac = len(long_bh_idx)/len(long_idx)
    print('BH kick = 100 km/s, long BH frac = ' + str(long_bh_frac))

    # Modified model (BH kick = 200 km/s)
    new_bh_tE = t['t_E'][bh_idx]/2.0 # BHs
    new_tE = np.concatenate((new_bh_tE, t['t_E'][not_bh_idx])) # All events
    new_long_idx = np.where(new_tE > 120)[0] # tE > 120 day events
    new_long_bh_idx = np.where(new_bh_tE > 120)[0] # tE > 120 events that are BHs

    new_long_bh_frac = len(new_long_bh_idx)/len(new_long_idx)
    print('BH kick = 200 km/s, long BH frac = ' + str(new_long_bh_frac))

    bins = np.logspace(-0.5, 2.7, 26)
    
    fig = plt.figure(1, figsize = (6,5))
    plt.clf()
    plt.subplots_adjust(left = 0.17, top = 0.8, bottom = 0.2)
    plt.hist(t['t_E'], bins = bins, label = 'BH Kick = 100 km/s',
             histtype = 'step', color = mpl_b)
    plt.hist(t['t_E'][bh_idx], bins = bins,
             histtype = 'step', color = mpl_o)
    plt.hist(new_tE, bins = bins, label = 'BH Kick = 200 km/s',
             histtype = 'step', color = mpl_b, linestyle = '--')
    plt.hist(new_bh_tE, bins = bins, 
             histtype = 'step', color = mpl_o, linestyle = '--')
    plt.text(0.3, 100, 'All events', color = mpl_b)
    plt.text(2.2, 8, 'BH events', color = mpl_o)
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('$t_E$ (days)')
    plt.ylabel('Number of events')
    plt.ylim(1, 5000)
    plt.axvline(x = 120, color = mpl_r)
    plt.text(130, 2000, '$t_E = 120$ days', color = mpl_r, rotation=90)
    plt.legend()

    return


def plot_airmass_moon():
    # coordinates to center of OGLE field BLG502 (2017 objects)
    ra = "18:00:00"
    dec = "-30:00:00"
    months = np.array([4, 5, 6, 7])
    days = np.array([10, 15, 15, 30])
    # outdir = '/Users/jlu/doc/proposals/keck/uc/18A/'
    outdir = ''

    # Keck 2
    skycalc.plot_airmass(ra, dec, 2021, months, days, 'keck2', 
                         outfile=outdir + 'microlens_airmass_keck2_21A.png', date_idx=-1)
    skycalc.plot_moon(ra, dec, 2021, months, 
                      outfile=outdir + 'microlens_moon_21A.png')

    # Keck 1
    skycalc.plot_airmass(ra, dec, 2021, months, days, 'keck1',
                         outfile=outdir + 'microlens_airmass_keck1_21A.png', date_idx=-1)
    
    return


def plot_3_targs():
    
    #name three objects
    targNames = ['ob140613', 'ob150211', 'ob150029']

    #object alignment directories
    an_dirs = ['/g/lu/microlens/cross_epoch/OB140613/a_2018_09_24/prop/',
               '/g/lu/microlens/cross_epoch/OB150211/a_2018_09_19/prop/',
               '/g/lu/microlens/cross_epoch/OB150029/a_2018_09_24/prop/']
    align_dirs = ['align/align_t', 'align/align_t', 'align/align_t']
    points_dirs = ['points_a/', 'points_d/', 'points_d/']
    poly_dirs = ['polyfit_a/fit', 'polyfit_d/fit', 'polyfit_d/fit']

    xlim = [1.2, 2.0, 1.5]
    ylim = [1.0, 7.0, 1.5]

    #Output file
    #filename = '/Users/jlu/doc/proposals/keck/uc/19A/plot_3_targs.png'
    filename = '/Users/jlu/plot_3_targs.png'
    #figsize = (15, 4.5)
    figsize = (10, 4.5)
    
    ps = 9.92

    plt.close(1)
    plt.figure(1, figsize=figsize)
    
    Ntarg = len(targNames) - 1
    for i in range(Ntarg):
        rootDir = an_dirs[i]
        starName = targNames[i]
        align = align_dirs[i]
        poly = poly_dirs[i]
        point = points_dirs[i]
    
        s = starset.StarSet(rootDir + align)
        s.loadPolyfit(rootDir + poly, accel=0, arcsec=0)

        names = s.getArray('name')
        mag = s.getArray('mag')
        x = s.getArray('x') 
        y = s.getArray('y') 

        ii = names.index(starName)
        star = s.stars[ii]

        pointsTab = Table.read(rootDir + point + starName + '.points', format='ascii')

        time = pointsTab[pointsTab.colnames[0]]
        x = pointsTab[pointsTab.colnames[1]]
        y = pointsTab[pointsTab.colnames[2]]
        xerr = pointsTab[pointsTab.colnames[3]]
        yerr = pointsTab[pointsTab.colnames[4]]

        if i == 0:
            print('Doing MJD')
            idx_2015 = np.where(time <= 57387)
            idx_2016 = np.where((time > 57387) & (time <= 57753))
            idx_2017 = np.where((time > 57753) & (time <= 58118))
            idx_2018 = np.where((time > 58119) & (time <= 58484))
        else:
            idx_2015 = np.where(time < 2016)
            idx_2016 = np.where((time >= 2016) & (time < 2017))
            idx_2017 = np.where((time >= 2017) & (time < 2018))
            idx_2018 = np.where((time >= 2018) & (time < 2019))

        fitx = star.fitXv
        fity = star.fitYv
        dt = time - fitx.t0
        fitLineX = fitx.p + (fitx.v * dt)
        fitSigX = np.sqrt( fitx.perr**2 + (dt * fitx.verr)**2 )

        fitLineY = fity.p + (fity.v * dt)
        fitSigY = np.sqrt( fity.perr**2 + (dt * fity.verr)**2 )

        from matplotlib.ticker import FormatStrFormatter
        fmtX = FormatStrFormatter('%5i')
        fmtY = FormatStrFormatter('%6.2f')
        fontsize1 = 16
        
        # Convert everything into relative coordinates:
        x -= fitx.p
        y -= fity.p
        fitLineX -= fitx.p
        fitLineY -= fity.p

        # Change plate scale.
        x = x * ps * -1.0
        y = y * ps
        xerr *= ps
        yerr *= ps
        fitLineX = fitLineX * ps * -1.0
        fitLineY = fitLineY * ps
        fitSigX *= ps
        fitSigY *= ps

        paxes = plt.subplot(1, Ntarg, i+1)
        plt.errorbar(x[idx_2015], y[idx_2015], xerr=xerr[idx_2015], yerr=yerr[idx_2015], fmt='r.', label='2015')  
        plt.errorbar(x[idx_2016], y[idx_2016], xerr=xerr[idx_2016], yerr=yerr[idx_2016], fmt='g.', label='2016')  
        plt.errorbar(x[idx_2017], y[idx_2017], xerr=xerr[idx_2017], yerr=yerr[idx_2017], fmt='b.', label='2017')  
        plt.errorbar(x[idx_2018], y[idx_2018], xerr=xerr[idx_2018], yerr=yerr[idx_2018], fmt='c.', label='2018')  

        # if i==1:
        #     plt.yticks(np.arange(np.min(y-yerr-0.1*ps), np.max(y+yerr+0.1*ps), 0.3*ps))
        #     plt.xticks(np.arange(np.min(x-xerr-0.1*ps), np.max(x+xerr+0.1*ps), 0.25*ps))
        # else:
        #     plt.yticks(np.arange(np.min(y-yerr-0.1*ps), np.max(y+yerr+0.1*ps), 0.15*ps))
        #     plt.xticks(np.arange(np.min(x-xerr-0.1*ps), np.max(x+xerr+0.1*ps), 0.15*ps))
        paxes.tick_params(axis='both', which='major', labelsize=fontsize1)
        paxes.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
        paxes.xaxis.set_major_formatter(FormatStrFormatter('%.2f'))
        plt.xlabel('X offset (mas)', fontsize=fontsize1)
        plt.ylabel('Y offset (mas)', fontsize=fontsize1)
        plt.plot(fitLineX, fitLineY, 'k-', label='_nolegend_')    
        plt.plot(fitLineX + fitSigX, fitLineY + fitSigY, 'k--', label='_nolegend_')
        plt.plot(fitLineX - fitSigX, fitLineY - fitSigY, 'k--',label='_nolegend_')

        # Plot lines between observed point and the best fit value along the model line.
        for ee in range(len(time)):
            if ee in idx_2015[0].tolist():
                color_line = 'red'
            if ee in idx_2016[0].tolist():
                color_line = 'green'
            if ee in idx_2017[0].tolist():
                color_line = 'blue'
            if ee in idx_2018[0].tolist():
                color_line = 'cyan'
                
            plt.plot([fitLineX[ee], x[ee]], [fitLineY[ee], y[ee]], color=color_line, linestyle='dashed', alpha=0.8)
        
        plt.axis([xlim[i], -xlim[i], -ylim[i], ylim[i]])
        
        plt.title(starName.upper())
        if i==0:
            plt.legend(loc=1, fontsize=12)


    
    plt.subplots_adjust(wspace=0.6, hspace=0.6, left = 0.08, bottom = 0.05, right=0.95, top=0.90)
    plt.tight_layout()
    plt.show()
    plt.savefig(filename)

    return

def plot_ob140613_phot_ast():
    data = munge.getdata('ob140613', use_astrom_phot=True)
    mod_base = '/u/jlu/work/microlens/OB140613/a_2019_06_26/model_fits/'
    mod_base += '8_fit_multiphot_astrom_parallax2/aa_'

    mod_fit = model_fitter.PSPL_multiphot_astrom_parallax2_Solver(data, outputfiles_basename=mod_base)
    #mod_fit.separate_modes()
    mod_fit.summarize_results_modes()

    mod_all = mod_fit.get_best_fit_modes_model(def_best='maxL')
    tab_all = mod_fit.load_mnest_modes()

    plot_4panel(data, mod_all[0], tab_all[0], 'ob140613_phot_astrom.png', r_min_k=4.0, mass_max_lim=2, log=False)

    return

def plot_ob150211_phot_ast():
    data = munge.getdata('ob150211', use_astrom_phot=True)
    mod_base = '/u/jlu/work/microlens/OB150211/a_2019_06_26/model_fits/'
    mod_base += '8_fit_multiphot_astrom_parallax2/tmp/aa_'

    mod_fit = model_fitter.PSPL_multiphot_astrom_parallax2_Solver(data, outputfiles_basename=mod_base)
    #mod_fit.separate_modes()
    #mod_fit.summarize_results_modes()

    mod_all = mod_fit.get_best_fit_modes_model(def_best='maxL')
    tab_all = mod_fit.load_mnest_modes()

    tab_all[0]['weights'] = tab_all[0]['weights'] / tab_all[0]['weights'].sum()

    plot_4panel(data, mod_all[0], tab_all[0], 'ob150211_phot_astrom.png', mass_max_lim=10, log=True)

    return

def plot_4panel(data, mod, tab, outfile, r_min_k=None, mass_max_lim=2, log=False):
    # Calculate the model on a similar timescale to the data.
    tmax = np.max(np.append(data['t_phot1'], data['t_phot2'])) + 90.0
    t_mod_ast = np.arange(data['t_ast'].min() - 180.0, tmax, 2)
    t_mod_pho = np.arange(data['t_phot1'].min(), tmax, 2)

    # Get the linear motion curves for the source (includes parallax)
    p_unlens_mod = mod.get_astrometry_unlensed(t_mod_ast)
    p_unlens_mod_at_ast = mod.get_astrometry_unlensed(data['t_ast'])

    # Get the lensed motion curves for the source
    p_lens_mod = mod.get_astrometry(t_mod_ast)
    p_lens_mod_at_ast = mod.get_astrometry(data['t_ast'])

    # Geth the photometry
    m_lens_mod = mod.get_photometry(t_mod_pho, filt_idx=0)
    m_lens_mod_at_phot1 = mod.get_photometry(data['t_phot1'], filt_idx=0)
    m_lens_mod_at_phot2 = mod.get_photometry(data['t_phot2'], filt_idx=1)

    # Calculate the delta-mag between R-band and K-band from the
    # flat part at the end.
    tidx = np.argmin(np.abs(data['t_phot1'] - data['t_ast'][-1]))
    if r_min_k == None:
        r_min_k = data['mag1'][tidx] - data['mag2'][-1]
    print('r_min_k = ', r_min_k)

    # Plotting        
    plt.close(2)
    plt.figure(2, figsize=(18, 4))

    pan_wid = 0.15
    pan_pad = 0.09
    fig_pos = np.arange(0, 4) * (pan_wid + pan_pad) + pan_pad

    # Brightness vs. time
    fm1 = plt.gcf().add_axes([fig_pos[0], 0.36, pan_wid, 0.6])
    fm2 = plt.gcf().add_axes([fig_pos[0], 0.18, pan_wid, 0.2])
    fm1.errorbar(data['t_phot1'], data['mag1'], yerr=data['mag_err1'],
                 color = mpl_b, fmt='.', alpha=0.05)
    # fm1.errorbar(data['t_phot2'], data['mag2'] + r_min_k, yerr=data['mag_err2'],
    #              fmt='k.', alpha=0.9)
    fm1.plot(t_mod_pho, m_lens_mod, 'r-')
    fm2.errorbar(data['t_phot1'], data['mag1'] - m_lens_mod_at_phot1, yerr=data['mag_err1'],
                 color = mpl_b, fmt='.', alpha=0.05)
    # fm2.errorbar(data['t_phot2'], data['mag2'] + r_min_k - m_lens_mod_at_phot2, yerr=data['mag_err2'],
    #              fmt='k.', alpha=0.9)
#    fm2.set_yticks(np.array([0.0, 0.2]))
    fm1.yaxis.set_major_locator(plt.MaxNLocator(4))
    fm2.xaxis.set_major_locator(plt.MaxNLocator(2))
    fm2.axhline(0, linestyle='--', color='r')
    fm2.set_xlabel('Time (HJD)')
    fm1.set_ylabel('Magnitude')
    fm1.invert_yaxis()
    fm2.set_ylabel('Res.')
    
    
    # RA vs. time
    f1 = plt.gcf().add_axes([fig_pos[1], 0.36, pan_wid, 0.6])
    f2 = plt.gcf().add_axes([fig_pos[1], 0.18, pan_wid, 0.2])
    f1.errorbar(data['t_ast'], data['xpos']*1e3,
                    yerr=data['xpos_err']*1e3, fmt='k.', zorder = 1000)
    f1.plot(t_mod_ast, p_lens_mod[:, 0]*1e3, 'r-')
    f1.plot(t_mod_ast, p_unlens_mod[:, 0]*1e3, 'r--')
    f1.get_xaxis().set_visible(False)
    f1.set_ylabel(r'$\Delta \alpha^*$ (mas)')
    f1.get_shared_x_axes().join(f1, f2)
    
    f2.errorbar(data['t_ast'], (data['xpos'] - p_unlens_mod_at_ast[:,0]) * 1e3,
                yerr=data['xpos_err'] * 1e3, fmt='k.', alpha=1, zorder = 1000)
    f2.plot(t_mod_ast, (p_lens_mod[:, 0] - p_unlens_mod[:, 0])*1e3, 'r-')
    f2.axhline(0, linestyle='--', color='r')
    f2.xaxis.set_major_locator(plt.MaxNLocator(3))
    f2.set_xlabel('Time (HJD)')
    f2.set_ylabel('Res.')

    
    # Dec vs. time
    f3 = plt.gcf().add_axes([fig_pos[2], 0.36, pan_wid, 0.6])
    f4 = plt.gcf().add_axes([fig_pos[2], 0.18, pan_wid, 0.2])
    f3.errorbar(data['t_ast'], data['ypos']*1e3,
                    yerr=data['ypos_err']*1e3, fmt='k.', zorder = 1000)
    f3.plot(t_mod_ast, p_lens_mod[:, 1]*1e3, 'r-')
    f3.plot(t_mod_ast, p_unlens_mod[:, 1]*1e3, 'r--')
    f3.set_ylabel(r'$\Delta \delta$ (mas)')
    f3.yaxis.set_major_locator(plt.MaxNLocator(4))
    f3.get_xaxis().set_visible(False)
    f3.get_shared_x_axes().join(f3, f4)
    
    f4.errorbar(data['t_ast'], (data['ypos'] - p_unlens_mod_at_ast[:,1]) * 1e3,
                yerr=data['ypos_err'] * 1e3, fmt='k.', alpha=1, zorder = 1000)
    f4.plot(t_mod_ast, (p_lens_mod[:, 1] - p_unlens_mod[:, 1])*1e3, 'r-')
    f4.axhline(0, linestyle='--', color='r')
    f4.xaxis.set_major_locator(plt.MaxNLocator(3))
#    f4.set_yticks(np.array([0.0, -0.2])) # For OB140613
    f4.set_xlabel('Time (HJD)')
    f4.set_ylabel('Res.')


    # Mass posterior
    masses = 10**tab['log_thetaE'] / (8.14 * 10**tab['log_piE'])
    weights = tab['weights']
    
    f5 = plt.gcf().add_axes([fig_pos[3], 0.18, pan_wid, 0.8])
    bins = np.arange(0., 10, 0.1)
    f5.hist(masses, weights=weights, bins=bins, alpha = 0.9, log=log)
    f5.set_xlabel('Mass (M$_\odot$)')
    f5.set_ylabel('Probability')
    f5.set_xlim(0, mass_max_lim)

    plt.savefig(outfile)

    return


def summarize_results(tab):
    if len(tab) < 1:
        print('Did you run multinest_utils.separate_mode_files yet?') 

    # Which params to include in table
    parameters = tab.colnames
    parameters.remove('weights')
    parameters.remove('logLike')
    
    weights = tab['weights']
    sumweights = np.sum(weights)
    weights = weights / sumweights

    sig1 = 0.682689
    sig2 = 0.9545
    sig3 = 0.9973
    sig1_lo = (1.-sig1)/2.
    sig2_lo = (1.-sig2)/2.
    sig3_lo = (1.-sig3)/2.
    sig1_hi = 1.-sig1_lo
    sig2_hi = 1.-sig2_lo
    sig3_hi = 1.-sig3_lo

    print(sig1_lo, sig1_hi)

    # Calculate the median, best-fit, and quantiles.
    best_idx = np.argmax(tab['logLike'])
    best = tab[best_idx]
    best_errors = {}
    med_best = {}
    med_errors = {}
    
    for n in parameters:
        # Calculate median, 1 sigma lo, and 1 sigma hi credible interval.
        tmp = model_fitter.weighted_quantile(tab[n], [0.5, sig1_lo, sig1_hi], sample_weight=weights)
        
        # Switch from values to errors.
        err_lo = tmp[0] - tmp[1]
        err_hi = tmp[2] - tmp[0]

        # Store into dictionaries.
        med_best[n] = tmp[0]
        med_errors[n] = np.array([err_lo, err_hi])
        #best_errors[n] = np.array([best[n] - tmp[1], tmp[2] - best[n]])
        best_errors[n] = np.array([tmp[1], tmp[2]])

    print('####################')
    print('Best-Fit Solution:')
    print('####################')
    fmt = '    {0:15s}  best = {1:10.3f}  68\% low = {2:10.3f} 68% hi = {3:10.3f}'
    for n in parameters:
        print(fmt.format(n, best[n], best_errors[n][0], best_errors[n][1]))

    
    return
