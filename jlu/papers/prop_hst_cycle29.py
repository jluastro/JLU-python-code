import numpy as np
import pylab as plt
import pdb
import math
import os
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
# import lu_2019_lens
import copy
from astropy.time import Time
from astropy.coordinates import SkyCoord
from astropy import units as u
from flystar import analysis
import yaml
import lu_2019_lens

def piE_tE():
    mdir = '/u/jlu/work/microlens/'
    data_dict = {'ob120169' : mdir + 'OB120169/a_2020_08_18/model_fits/120_phot_astrom_parallax_aerr_ogle_keck/base_a/a5_',
                 'ob140613' : mdir + 'OB140613/a_2020_08_18/model_fits/120_phot_astrom_parallax_merr_ogle_keck/base_a/a2_',
                 'ob150029' : mdir + 'OB150029/a_2020_08_18/model_fits/120_fit_phot_astrom_parallax_aerr_ogle_keck/base_a/a1_',
                 'ob150211' : mdir + 'OB150211/a_2020_08_18/model_fits/120_phot_astrom_parallax_aerr_ogle_keck/base_a/a5_',
                 'mb09260' : mdir + 'MB09260/a_2020_08_07/model_fits/all_phot_ast_merr/base_b/b0_',
                 'mb10364' : mdir + 'MB10364/a_2020_12_12/model_fits/moa_hst_gp_f814w/a0_',
                 'ob110037' : mdir + 'OB110037/a_2020_08_26/model_fits/all_phot_ast_merr/base_c/c0_',
                 'ob110310' : mdir + 'OB110310/a_2020_08_26/model_fits/all_phot_ast_merr/base_a/a0_',
                 'ob110462' : mdir + 'OB110462/a_2020_08_26/model_fits/all_phot_ast_merr/base_a/a0_'}

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

    colors = {'ob120169': 'red',
              'ob140613': 'red',
              'ob150029': 'red',
              'ob150211': 'red',
              'mb09260' : 'gray',
              'mb10364' : 'gray',
              'ob110037' : 'gray',
              'ob110310' : 'gray',
              'ob110462' : 'gray'}

    # Set defaults.
    hist2d_kwargs['alpha'] = hist2d_kwargs.get('alpha', 0.2)
    hist2d_kwargs['levels'] = hist2d_kwargs.get('levels', quantiles_2d)

    targets = ['ob120169', 'ob140613', 'ob150029', 'ob150211',
               'mb09260', 'mb10364', 'ob110037', 'ob110310', 'ob110462'] 
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

    # Plot the piE-tE 2D posteriors.
    fig = plt.figure(1, figsize=(9,6))
    plt.clf()
    axes = plt.gca()
    plt.subplots_adjust(left = 0.47, bottom=0.15, right=0.97)

    sx = smooth
    sy = smooth

    hist2d_kwargs['fill_contours'] = hist2d_kwargs.get('fill_contours', False)
    hist2d_kwargs['plot_contours'] = hist2d_kwargs.get('plot_contours', True)

    for targ in targets:
        model_fitter.contour2d_alpha(tE[targ], piE[targ], span=[span, span], quantiles_2d=quantiles_2d,
                                 weights=weights[targ], ax=axes, smooth=[sy, sx], color=colors[targ],
                                 **hist2d_kwargs, plot_density=False, sigma_levels=[1, 2, 3])

    axes.scatter(100.171, 0.35319,
                 color='gray', marker='*', s = 150, 
                 zorder=1000, edgecolors='none')
    axes.scatter(127.957, 0.32037,
                 color='gray', marker='*', s = 150, 
                 zorder=1000, edgecolors='none')
    axes.scatter(275.53, 0.31,
                 color='gray', marker='*', s = 150, 
                 zorder=1000, edgecolors='none')
    axes.scatter(178.804, 0.10864,
                 color='gray', marker='*', s = 150, 
                 zorder=1000, edgecolors='none',
                 label='HST (PI: Sahu, \nCycle 25)')

    # MB19284
    axes.scatter(499.482, 0.038397, 
                 color='red', marker='*', s = 150, 
                 zorder=1000, edgecolors='none', 
                 label = 'HST \n(priv. comm.)')

    axes.plot(200, 0.15, color='gray', label='HST (PI: Sahu, \nCycle 17)')
    axes.plot(20, 0.2, color='red', label='Keck \n(priv. comm.)')

    # OB110022 from Lu+16.
    piEE_110022 = -0.393
    piEN_110022 = -0.071
    piE_110022 = np.hypot(piEE_110022, piEN_110022)
    tE_110022 = 61.4

    dcmax_110022 = 2.19/np.sqrt(8)
    dcmax_110022_pe = 1.06/np.sqrt(8)
    dcmax_110022_me = 1.17/np.sqrt(8)

    # Plotting OB110022.
    plt.scatter(tE_110022, piE_110022, marker = 'o', s = 50, color='red')

    # Darken some small points
    plt.scatter(142.436, 0.236727, marker = 'o', s = 50, color='red')
    plt.scatter(320, 0.135921, marker = 'o', s = 50, color='red')
    plt.scatter(151.636, 0.220, marker = 'o', s = 50, color='gray')
    plt.scatter(74.7, 0.378, marker = 'o', s = 50, color='gray')

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
                 alpha = 0.4, marker = '.', s = 15, 
                 color = 'paleturquoise')
    axes.scatter(t['t_E'][wd_idx], t['pi_E'][wd_idx], 
                 alpha = 0.4, marker = '.', s = 15, 
                 color = 'aqua')
    axes.scatter(t['t_E'][ns_idx], t['pi_E'][ns_idx], 
                 alpha = 0.4, marker = '.', s = 15, 
                 color = 'blue')
    axes.scatter(t['t_E'][bh_idx], t['pi_E'][bh_idx],
                 alpha = 0.8, marker = '.', s = 15, 
                 color = 'black')

    # Trickery to make the legend darker
    axes.scatter(0.01, 100, 
                 alpha = 0.8, marker = '.', s = 25, 
                 label = 'Star', color = 'paleturquoise')
    axes.scatter(0.01, 100, 
                 alpha = 0.8, marker = '.', s = 25,
                 label = 'White dwarf', color = 'aqua')
    axes.scatter(0.01, 100,
                 alpha = 0.8, marker = '.', s = 25, 
                 label = 'Neutron star', color = 'blue')
    axes.scatter(0.01, 100,
                 alpha = 0.8, marker = '.', s = 25, 
                 label = 'Black hole', color = 'black')

    axes.set_xlim(10, 800)
    axes.set_ylim(0.005, 0.5)
    axes.set_xlabel('$t_E$ (days)')
    axes.set_ylabel('$\pi_E$')
    axes.set_xscale('log')
    axes.set_yscale('log')
#    axes.legend(loc=3)
    axes.legend(bbox_to_anchor=(-0.87, 0.9), loc='upper left', ncol=1)
    plt.savefig('piE_tE_cycle29.png')
    plt.show()

    # Plot the deltac-piE 2D posteriors.
#    plt.close(2)
    fig = plt.figure(2, figsize=(6,6))
    plt.clf()
    axes = plt.gca()
    plt.subplots_adjust(bottom=0.15)

    # OB110022
    axes.errorbar(dcmax_110022, piE_110022, 
                   xerr = np.array([[0.3], [0.3]]), 
#                   xerr = np.array([[dcmax_110022_me], [dcmax_110022_pe]]), 
                   fmt = 'o', color = 'red', markersize = 8,
                   xuplims = True)

    axes.errorbar(0.619, 0.379, xerr=np.array([[0.2], [0.2]]),
                   fmt = 'o', color = 'gray', markersize = 8,
                   xuplims = True)

    axes.errorbar(0.853, 0.263, xerr=np.array([[0.28], [0.28]]),
                   fmt = 'o', color = 'gray', markersize = 8,
                   xuplims = True)

    axes.errorbar(1.204, 0.120, xerr=np.array([[0.3], [0.3]]),
                   fmt = 'o', color = 'gray', markersize = 8,
                   xuplims = True)

    axes.errorbar(1.586, 0.239, xerr=np.array([[0.4], [0.4]]),
                   fmt = 'o', color = 'gray', markersize = 8,
                   xuplims = True)

    axes.errorbar(0.265, 0.237, xerr=np.array([[0.1], [0.1]]), 
                   fmt = 'o', color = 'red', markersize = 8,
                   xuplims = True)

    axes.errorbar(0.316, 0.135, xerr=np.array([[0.1], [0.1]]), 
                   fmt = 'o', color = 'red', markersize = 8,
                   xuplims = True)

    for targ in ['ob110462', 'ob120169', 'ob150211']:
        model_fitter.contour2d_alpha(theta_E[targ]/np.sqrt(8), piE[targ], span=[span, span], quantiles_2d=quantiles_2d,
                                 weights=weights[targ], ax=axes, smooth=[sy, sx], color=colors[targ],
                                 **hist2d_kwargs, plot_density=False, sigma_levels=[1, 2, 3])


#        axes.text(label_pos_ast[targ][0], label_pos_ast[targ][1],
#                  targ.upper(), color=colors[targ])    

    axes.scatter(final_delta_arr[st_idx], t['pi_E'][st_idx], 
                  alpha = 0.4, marker = '.', s = 15,
                  c = 'paleturquoise')
    axes.scatter(final_delta_arr[wd_idx], t['pi_E'][wd_idx], 
                  alpha = 0.4, marker = '.', s = 15,
                  c = 'aqua')
    axes.scatter(final_delta_arr[ns_idx], t['pi_E'][ns_idx], 
                  alpha = 0.4, marker = '.', s = 15,
                  c = 'blue')
    axes.scatter(final_delta_arr[bh_idx], t['pi_E'][bh_idx], 
                  alpha = 0.8, marker = '.', s = 15,
                  c = 'black')

    axes.set_xlabel('$\delta_{c,max}$ (mas)')
    axes.set_ylabel('$\pi_E$')
    axes.set_xscale('log')
    axes.set_yscale('log')
    axes.set_xlim(0.005, 4)
    axes.set_ylim(0.005, 0.5)
    plt.savefig('piE_deltac_cycle29.png')
    plt.show()


def EWS_select_BH():
    """
    Plot the fraction of long duration EWS events
    that will be due to black holes.
    """

    def binom(n,k,p):
        """
        n = total
        k = success
        p = prob of success
        """
        prob = scipy.special.comb(n,k) * p**k * (1-p)**(n-k)
        err = np.sqrt(n*p*(1-p))
        
        return prob, err

    def calc_fraction(t, mintE):
        """
        Calculate the fraction of long duration events
        that have black hole lenses
        """

        not_bh_idx = np.where((t['t_E'] > mintE) & 
                              (t['rem_id_L'] != 103))[0]
        
        bh_idx = np.where((t['t_E'] > mintE) & 
                          (t['rem_id_L'] == 103))[0]
        
        frac = len(bh_idx)/(len(not_bh_idx) + len(bh_idx))
        
        bh_err = np.sqrt(len(bh_idx))
        not_bh_err = np.sqrt(len(not_bh_idx))
        n_bh = np.array(len(bh_idx))
        n_not_bh = np.array(len(not_bh_idx))
        term1 = bh_err * n_not_bh
        term2 = not_bh_err * n_bh
        term3 = (n_bh + n_not_bh)**2
        frac_err = np.sqrt(term1**2 + term2**2)/term3

        return frac, frac_err

    t = Table.read('/u/casey/scratch/papers/microlens_2019/popsycle_rr_files/Mock_EWS_v2.fits')

#    times = np.linspace(60, 150, 20)
    times = np.linspace(60, 200, 20)

    frac_arr = np.zeros(len(times))
    frac_err_arr = np.zeros(len(times))

    for i in np.arange(len(times)):
        frac_arr[i], frac_err_arr[i] = calc_fraction(t, mintE = times[i])

    prob1, err1 = binom(10, 1, 0.4)
    print(prob1, err1)

    fig = plt.figure(11, figsize=(4,4))
    plt.subplots_adjust(bottom=0.2, top=0.97, left=0.25, right=0.95)
    plt.clf()
    plt.fill_between(times, frac_arr-frac_err_arr, frac_arr+frac_err_arr, alpha=0.5, color='tab:blue', label='Expectation')
    plt.errorbar(120, 1/10, yerr=0.09486832980505139, marker= 's', capsize=5, elinewidth = 2, ms = 10, ls = 'None', label='Sample', color='k')
#    plt.errorbar(122, 1/10, yerr=err1/10, marker= 's', capsize=5, elinewidth = 2, ms = 10, ls = 'None', label='Sample', color='r')
    plt.ylabel('Fraction of BH events')
    plt.xlabel('Minimum $t_E$ (days)')
#    plt.xlim(60, 150)
#    plt.ylim(0, 0.6)
    plt.xlim(60, 200)
    plt.ylim(0, 0.7)
    plt.legend(loc=2)
    plt.show()
    plt.savefig('expect_vs_detect.png')

    return

