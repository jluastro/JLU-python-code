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
from scipy.stats import chi2
# from gcwork import starset
# from gcwork import starTables
from jlu.papers import lu_2019_lens
from astropy.table import Table
from jlu.util import fileUtil
from astropy.table import Table, Column, vstack
from astropy.io import fits
import matplotlib.ticker
import matplotlib.colors
from matplotlib.pylab import cm
from matplotlib.colors import Normalize, LogNorm
import mpl_toolkits
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable
from matplotlib.ticker import NullFormatter
from microlens.jlu import model_fitter, multinest_utils, multinest_plot, munge_ob150211, munge_ob150029, model
from matplotlib.colors import LinearSegmentedColormap, colorConverter
import pdb
import pickle
from scipy.stats import norm
import yaml
from flystar import plots
from flystar import analysis
from astropy.coordinates import SkyCoord
import astropy.units as u
from astropy.io import fits
from astropy.wcs import WCS
from jlu.util import datetimeUtil as dtUtil
from datetime import datetime as dt

# Default matplotlib color cycles.
mpl_b = '#1f77b4'
mpl_o = '#ff7f0e'
mpl_g = '#2ca02c'
mpl_r = '#d62728'

# run directory
# ob120169_dir = '/u/jlu/work/microlens/OB120169/a_2019_06_26/model_fits/120_fit_multiphot_astrom_parallax_aerr/base_c/'
# ob140613_dir = '/u/jlu/work/microlens/OB140613/a_2019_06_26/model_fits/120_fit_multiphot_astrom_parallax_merr/base_b/'
# ob150029_dir = '/u/jlu/work/microlens/OB150029/a_2019_06_26/model_fits/120_fit_multiphot_astrom_parallax_aerr/base_b/'
# ob150211_dir = '/u/jlu/work/microlens/OB150211/a_2019_06_26/model_fits/120_fit_multiphot_astrom_parallax_aerr/base_b/'

ob120169_dir = '/u/jlu/work/microlens/OB120169/a_2020_08_18/model_fits/120_phot_astrom_parallax_aerr_ogle_keck/base_a/'
ob140613_dir = '/u/jlu/work/microlens/OB140613/a_2020_08_18/model_fits/120_phot_astrom_parallax_merr_ogle_keck/base_a/'
ob150029_dir = '/u/jlu/work/microlens/OB150029/a_2020_08_18/model_fits/120_fit_phot_astrom_parallax_aerr_ogle_keck/base_a/'
ob150211_dir = '/u/jlu/work/microlens/OB150211/a_2020_08_18/model_fits/120_phot_astrom_parallax_aerr_ogle_keck/base_a/'

# run id
# ob120169_id = 'c2'
# ob140613_id = 'b1'
# ob150029_id = 'b3'
# ob150211_id = 'b2'

ob120169_id = 'a5'
ob140613_id = 'a2'
ob150029_id = 'a1'
ob150211_id = 'a5'

mod_roots = {'ob120169': ob120169_dir + ob120169_id + '_',
             'ob140613': ob140613_dir + ob140613_id + '_',
             'ob150029': ob150029_dir + ob150029_id + '_',
             'ob150211': ob150211_dir + ob150211_id + '_'}

phot_ast_fits = {'MB09260' : '/u/jlu/work/microlens/MB09260/a_2021_07_08/model_fits/moa_hst_phot_ast_gp/base_a/a0_',
                 'MB10364' : '/u/jlu/work/microlens/MB10364/a_2021_07_08/model_fits/moa_hst_phot_ast_gp/base_a/a0_',
                 'OB110037' : '/u/jlu/work/microlens/OB110037/a_2021_07_08/model_fits/ogle_hst_phot_ast_gp/base_a/a0_',
                 'OB110310' : '/u/jlu/work/microlens/OB110310/a_2021_07_08/model_fits/ogle_hst_phot_ast_gp/base_a/a0_',
                 'OB110462' : '/u/jlu/work/microlens/OB110462/a_2021_07_08/model_fits/ogle_hst_phot_ast_gp/base_a/a0_'}

# prop_dir = '/u/casey/scratch/code/JLU-python-code/jlu/papers/'
prop_dir = '/u/jlu/doc/proposals/keck/uc/22A/'

# Target coordinates
ra_ob190017 = '17:59:03.5200'
dec_ob190017 = '-27:32:49.2000'

ra_ob170095 = '17:51:27.9400'
dec_ob170095 = '-33:08:06.600'

ra_kb200101 = '17:45:11.0300'
dec_kb200101 = '-25:24:28.9800'

ra_mb19284 = '18:05:55.0600'
dec_mb19284 = '-30:20:12.9400'

ra_ob170019 = '17:52:18.7400'
dec_ob170019 = '-33:00:04.000'

mdir = '/u/jlu/work/microlens/'

phot_2020_dir = {'kb200101' : mdir + 'KB200101/a_2020_09_10/model_fits/kmtnet_phot_par/a0_',
                 'mb19284' : mdir + '',
                 'ob190017' : mdir + 'OB190017/a_2020_09_10/model_fits/ogle_phot_par/a0_',
                 'ob170019' : mdir + 'OB170019/a_2020_09_10/model_fits/ogle_phot_par/a0_',
                 'ob170095' : mdir + 'OB170095/a_2021_09_18/model_fits/base_a/a0_'}

ob170095_dir = mdir + 'OB170095/a_2021_09_18/model_fits/base_a/a0_'

# Gaia stuff
# NOTE: OB190017 is NOT in Gaia.
# Still trying to figure out for OB170095
# gaia_ob190017 = analysis.query_gaia(ra_ob190017, dec_ob190017, search_radius=30.0, table_name='gaiaedr3')
# gaia_ob170095 = analysis.query_gaia(ra_ob170095, dec_ob170095, search_radius=30.0, table_name='gaiaedr3')
# gaia_kb200101 = analysis.query_gaia(ra_kb200101, dec_kb200101, search_radius=30.0, table_name='gaiaedr3')
# gaia_mb19284 = analysis.query_gaia(ra_mb19284, dec_mb19284, search_radius=30.0, table_name='gaiaedr3')
# gaia_ob170019 = analysis.query_gaia(ra_ob170019, dec_ob170019, search_radius=30.0, table_name='gaiaedr3')
# 
# gaia_ob190017.write('gaia_ob190017.gz', overwrite=True, format='ascii')
# gaia_ob170095.write('gaia_ob170095.gz', overwrite=True, format='ascii')
# gaia_kb200101.write('gaia_kb200101.gz', overwrite=True, format='ascii')
# gaia_mb19284.write('gaia_mb19284.gz', overwrite=True, format='ascii')
# gaia_ob170019.write('gaia_ob170019.gz', overwrite=True, format='ascii')

# gaia_ob190017 = Table.read('gaia_ob190017.gz', format='ascii')
# gaia_ob170095 = Table.read('gaia_ob170095.gz', format='ascii')
# gaia_kb200101 = Table.read('gaia_kb200101.gz', format='ascii')
# gaia_mb19284 = Table.read('gaia_mb19284.gz', format='ascii')
# gaia_ob170019 = Table.read('gaia_ob170019.gz', format='ascii')

def get_new_Raithel_files():
    #Raithel18
    R_527 = '/u/samrose/scratch/metal_ifmr_runs/OGLE527ub_v3/OGLE527ub_v3_my_Raithel_refined_events_ubv_I_Damineli16.fits'
    R_611 = '/u/samrose/scratch/metal_ifmr_runs/OGLE611ub_v3/my_Raithel611/OGLE611ub_v3_my_Raithel_refined_events_ubv_I_Damineli16.fits'
    R_629 = '/u/samrose/scratch/metal_ifmr_runs/OGLE629ub_v3/OGLE629ub_v3_my_Raithel_refined_events_ubv_I_Damineli16.fits'
    R_500 = '/u/samrose/scratch/metal_ifmr_runs/OGLE500ub_v3/OGLE500ub_v3_Raithel_refined_events_ubv_I_Damineli16.fits'
    R_506 = '/u/samrose/scratch/metal_ifmr_runs/OGLE506ub_v3/OGLE506ub_v3_Raithel_refined_events_ubv_I_Damineli16.fits'
    R_675 = '/u/samrose/scratch/metal_ifmr_runs/OGLE675ub_v3/OGLE675ub_v3_Raithel_refined_events_ubv_I_Damineli16.fits'
    R_504 = '/u/samrose/scratch/NERSC_runs/l2.150_b-1.770/seed42_raithel18_refined_events_ubv_I_Damineli16.fits'
    R_511 = '/u/samrose/scratch/NERSC_runs/l3.280_b-2.520/seed42_raithel18_refined_events_ubv_I_Damineli16.fits'
    R_648 = '/u/samrose/scratch/NERSC_runs/l1.960_b0.940/seed42_raithel18_refined_events_ubv_I_Damineli16.fits'

    all_events = None
    
    for rfield in [R_527, R_611, R_629, R_500, R_506, R_675, R_504, R_511, R_648]:
        cuts_table = Table.read(rfield)
        
        #perform magnitude cuts
        if 'ubv_I_app_LSN' in cuts_table.colnames:
            bad_m_idx = np.where(cuts_table['ubv_I_app_LSN'] > 21)[0]
            cuts_table.remove_rows([bad_m_idx])
            bad_del_idx = np.where(cuts_table['delta_m_I'] < 0.1)[0]
            cuts_table.remove_rows([bad_del_idx])
        elif 'ubv_i_app_LSN' in cuts_table.colnames:
            bad_m_idx = np.where(cuts_table['ubv_i_app_LSN'] > 21)[0]
            cuts_table.remove_rows([bad_m_idx])
            bad_del_idx = np.where(cuts_table['delta_m_i'] < 0.1)[0]
            cuts_table.remove_rows([bad_del_idx])        

        if all_events is not None:
            all_events = vstack([all_events, cuts_table])
        else:
            all_events = cuts_table

    all_events.write('from_sam_ews_raithel.fits')

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
                 'ast':  {'ob120169': [45, 0.12],
                          'ob140613': [250, 0.15],
                          'ob150029': [40, 0.22],
                          'ob150211': [140, 0.01],
                          'ob170019': [120, 0.045],
                          'ob170095': [30, 0.04],
                          'ob190017': [180, 0.28],
                          'kb200101': [180, 0.016]}}

    label_pos_ast = {'ob120169': [0.006, 0.06],
                     'ob140613': [0.04, 0.145],
                     'ob150029': [0.02, 0.25],
                     'ob150211': [0.03, 0.012]}

    colors = {'ob120169': 'gray',
              'ob140613': 'gray',
              'ob150029': 'gray',
              'ob150211': 'red',
              'ob170019': 'blue',
              'ob170095': 'blue',
              'ob190017': 'blue',
              'kb200101': 'blue',
              'MB09260' : 'gray',
              'MB10364' : 'gray',
              'OB110037' : 'gray',
              'OB110310' : 'gray',
              'OB110462' : 'red'}

    # Set defaults.
    hist2d_kwargs['alpha'] = hist2d_kwargs.get('alpha', 0.2)
    hist2d_kwargs['levels'] = hist2d_kwargs.get('levels', quantiles_2d)

    targets = ['ob120169', 'ob140613', 'ob150029', 'ob150211'] 
    hst_targets = ['MB09260', 'MB10364', 'OB110037', 'OB110310', 'OB110462']
    new_targets = ['ob170019', 'ob170095', 'ob190017', 'kb200101']
    tE = {}
    piE = {}
    theta_E = {}
    weights = {}

    for targ in hst_targets:
        fit_targ, dat_targ = multinest_utils.get_data_and_fitter(phot_ast_fits[targ]) 

        res_targ = fit_targ.load_mnest_results(remake_fits=True)

        tE[targ] = res_targ['tE']
        piE[targ] = np.hypot(res_targ['piE_E'], res_targ['piE_N'])
        theta_E[targ] = 10**res_targ['log10_thetaE']
        weights[targ] = res_targ['weights']

    for targ in targets:
        fitter, data = lu_2019_lens.get_data_and_fitter(lu_2019_lens.pspl_ast_multiphot[targ])
#        stats_ast, data_ast, mod_ast = lu_2019_lens.load_summary_statistics(lu_2019_lens.pspl_ast_multiphot[targ])
        tab_list = fitter.load_mnest_modes()
        mode = lu_2019_lens.pspl_ast_multiphot_mode[targ]
        fit_targ, dat_targ = multinest_utils.get_data_and_fitter(lu_2019_lens.pspl_ast_multiphot[targ])

        res_targ = tab_list[mode]
        
#        res_targ = fit_targ.load_mnest_results(remake_fits=True)
#        res_targ = fit_targ.load_mnest_modes()
#        res_targ = res_targ[0]
#
        tE[targ] = res_targ['tE']
        piE[targ] = np.hypot(res_targ['piE_E'], res_targ['piE_N'])
        theta_E[targ] = res_targ['thetaE_amp']
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

    # MB190284 fit results (from Dave Bennett)
    data_tab = '/u/jlu/doc/proposals/hst/cycle28_mid2/mcmc_bsopcnC_3.dat'

    # chi^2 1/t_E t0 umin sep theta eps1=q/(1+q) 1/Tbin dsxdt dsydt t_fix Tstar(=0) pi_E,N piE,E 0 0 0 0 0 0 0 0 0 A0ogleI A2ogleI A0ogleV A2ogleV A0moa2r A2moa2r A0moa2V
    data = Table.read(data_tab, format='ascii.fixed_width_no_header', delimiter=' ')
    data.rename_column('col1', 'chi2')
    data.rename_column('col2', 'tE_inv')
    data.rename_column('col3', 't0')
    data.rename_column('col4', 'u0')
    data.rename_column('col5', 'sep')
    data.rename_column('col6', 'theta')
    data.rename_column('col7', 'eps1')
    data.rename_column('col8', 'Tbin_inv')
    data.rename_column('col9', 'dsxdt')
    data.rename_column('col10', 'dsydt')
    data.rename_column('col11', 't_fix')
    data.rename_column('col12', 'Tstar')
    data.rename_column('col13', 'piEE')
    data.rename_column('col14', 'piEN')
    data['tE'] = 1.0 / data['tE_inv']
    data['piEE'] = data['piEE'].astype('float')
    data['weight'] = np.ones(len(data))
    data['piE'] = np.hypot(data['piEE'], data['piEN'])

    # Plot the piE-tE 2D posteriors.
#    plt.close(1)
    fig = plt.figure(1, figsize=(6,6))
    plt.clf()
    axes = plt.gca()
    plt.subplots_adjust(bottom=0.15)

    sx = smooth
    sy = smooth

    hist2d_kwargs['fill_contours'] = hist2d_kwargs.get('fill_contours', False)
    hist2d_kwargs['plot_contours'] = hist2d_kwargs.get('plot_contours', True)

    for targ in targets + new_targets + hst_targets:
        model_fitter.contour2d_alpha(tE[targ], piE[targ], span=[span, span], quantiles_2d=quantiles_2d,
                                 weights=weights[targ], ax=axes, smooth=[sy, sx], color=colors[targ],
                                 **hist2d_kwargs, plot_density=False, sigma_levels=[1, 2])

    for targ in new_targets:
        axes.text(label_pos[fit_type][targ][0], label_pos[fit_type][targ][1],
                      targ.upper(), color=colors[targ])    

    model_fitter.contour2d_alpha(data['tE'], data['piE'], span=[span, span], quantiles_2d=quantiles_2d,
                                 ax=axes, smooth=[sy, sx], color='blue',
                                 **hist2d_kwargs, plot_density=False, sigma_levels=[1, 2])
    axes.text(300, 0.025, 'MB19284', color='blue')

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
#    plt.scatter(tE_110022, piE_110022, marker = 'o', s = 30, color='indigo')
#    axes.text(18, 0.38, 'OB110022', color='indigo')
    plt.scatter(tE_110022, piE_110022, marker = 's', s = 40, color='gray')
    plt.scatter(129.4, 0.04, marker = 's', s = 40, color='blue')
#    axes.text(17, 0.38, 'OB110022', color='gray')

#    plt.scatter(tE_19284, piE_19284, marker = 'o', s = 30, color='lime')
#    axes.text(300, 0.1, 'MB19284', color='lime')

    # Add the PopSyCLE simulation points.
    t = Table.read('/u/casey/scratch/papers/microlens_2019/popsycle_rr_files/Mock_EWS_v2_NEW_DELTAM.fits') 
#    t = Table.read('from_sam_ews_raithel.fits')
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
#    g_arr = 10**(-0.4 * (t['ubv_I_app_L'] - t['ubv_I_app_S']))
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
                 color = 'tab:cyan')
    axes.scatter(t['t_E'][bh_idx], t['pi_E'][bh_idx],
                 alpha = 0.8, marker = '.', s = 25, 
                 color = 'black')

    # Trickery to make the legend darker
    axes.scatter(0.01, 100, 
                 alpha = 0.8, marker = '.', s = 25, 
                 label = 'Star', color = 'paleturquoise')
    axes.scatter(0.01, 100, 
                 alpha = 0.8, marker = '.', s = 25,
                 label = 'WD', color = 'aqua')
    axes.scatter(0.01, 100,
                 alpha = 0.8, marker = '.', s = 25, 
                 label = 'NS', color = 'tab:cyan')
    axes.scatter(0.01, 100,
                 alpha = 0.8, marker = '.', s = 25, 
                 label = 'BH', color = 'black')

    axes.set_xlim(10, 1000)
    axes.set_ylim(0.005, 0.5)
    axes.set_xlabel('$t_E$ (days)')
    axes.set_ylabel('$\pi_E$')
    axes.set_xscale('log')
    axes.set_yscale('log')
    axes.legend(loc=3)
    plt.savefig('piE_tE_22A.png')
    plt.show()

    # Plot the deltac-piE 2D posteriors.
#    plt.close(2)
    fig = plt.figure(2, figsize=(6,6))
    plt.clf()
    axes = plt.gca()
    plt.subplots_adjust(bottom=0.15)

    axes.errorbar(dcmax_110022, piE_110022, 
                   xerr = np.array([[dcmax_110022_me], [dcmax_110022_pe]]), 
                   fmt = 'o', color = 'gray', markersize = 5,
                   xuplims = True)
#    axes.text(0.5, 0.3, 'OB110022', color='gray')

    # MB10364, OB110037, OB11030
    axes.errorbar(0.56, 0.25, xerr = np.array([[0.3], [0.3]]), 
                  fmt = 'o', color = 'gray', markersize = 5,
                  xuplims = True)

    axes.errorbar(0.92, 0.11, xerr = np.array([[0.3], [0.3]]), 
                   fmt = 'o', color = 'gray', markersize = 5,
                   xuplims = True)

    axes.errorbar(0.89, 0.08, xerr = np.array([[0.3], [0.3]]), 
                   fmt = 'o', color = 'gray', markersize = 5,
                   xuplims = True)

    # ob120169 and ob155029
    axes.errorbar(1.06, 0.17, xerr = np.array([[0.3], [0.3]]), 
                  fmt = 'o', color = 'gray', markersize = 5,
                  xuplims = True)

    axes.errorbar(0.45, 0.17, xerr = np.array([[0.3], [0.3]]), 
                   fmt = 'o', color = 'gray', markersize = 5,
                  xuplims = True)

    for targ in ['ob140613', 'ob150211', 'OB110037', 'OB110462']:
        model_fitter.contour2d_alpha(theta_E[targ]/np.sqrt(8), piE[targ], span=[span, span], quantiles_2d=quantiles_2d,
                                 weights=weights[targ], ax=axes, smooth=[sy, sx], color=colors[targ],
                                 **hist2d_kwargs, plot_density=False, sigma_levels=[1, 2])

#        axes.text(label_pos_ast[targ][0], label_pos_ast[targ][1],
#                  targ.upper(), color=colors[targ])    

    xarr = np.linspace(0.001, 4, 1000)
    axes.fill_between(xarr, xarr*0.18, xarr*0.07, alpha=0.15, color='orange')
    axes.text(0.05, 0.006, 'Mass Gap', rotation=45)

    axes.scatter(final_delta_arr[st_idx], t['pi_E'][st_idx], 
                  alpha = 0.4, marker = '.', s = 25,
                  c = 'paleturquoise')
    axes.scatter(final_delta_arr[wd_idx], t['pi_E'][wd_idx], 
                  alpha = 0.4, marker = '.', s = 25,
                  c = 'aqua')
    axes.scatter(final_delta_arr[ns_idx], t['pi_E'][ns_idx], 
                  alpha = 0.4, marker = '.', s = 25,
                  c = 'tab:cyan')
    axes.scatter(final_delta_arr[bh_idx], t['pi_E'][bh_idx], 
                  alpha = 0.8, marker = '.', s = 25,
                  c = 'black')

    axes.set_xlabel('$\delta_{c,max}$ (mas)')
    axes.set_ylabel('$\pi_E$')
    axes.set_xscale('log')
    axes.set_yscale('log')
#    axes.set_xlim(0.005, 4)
#    axes.set_ylim(0.009, 0.5)
    axes.set_xlim(0.02, 2)
    axes.set_ylim(0.005, 0.5)
    plt.savefig('piE_deltac_22A.png')
    plt.show()


def gaia_info():
    print('OB190017 : too faint to be in Gaia')
    print('')
    print('OB170095 : no parallax or PMs')
    print('Mag (G) : ', gaia_ob170095['phot_g_mean_mag'][0])
#    print('OB170095 : not 100% sure this is the target')
#    print('Parallax : {0:2f} +/- {1:2f}'.format(gaia_ob170095['parallax'][0], gaia_ob170095['parallax_error'][0]))
#    print('PM RA : {0:2f} +/- {1:2f}'.format(gaia_ob170095['pmra'][0], gaia_ob170095['pmra_error'][0]))
#    print('PM Dec : {0:2f} +/- {1:2f}'.format(gaia_ob170095['pmdec'][0], gaia_ob170095['pmdec_error'][0]))
    print('')
    print('KB200101')
    print('Mag (G) : ', gaia_kb200101['phot_g_mean_mag'][0])
    print('Parallax : {0:2f} +/- {1:2f}'.format(gaia_kb200101['parallax'][0], gaia_kb200101['parallax_error'][0]))
    print('PM RA : {0:2f} +/- {1:2f}'.format(gaia_kb200101['pmra'][0], gaia_kb200101['pmra_error'][0]))
    print('PM Dec : {0:2f} +/- {1:2f}'.format(gaia_kb200101['pmdec'][0], gaia_kb200101['pmdec_error'][0]))
    print('')
    print('MB19284')
    print('Mag (G) : ', gaia_mb19284['phot_g_mean_mag'][0])
    print('Parallax : {0:2f} +/- {1:2f}'.format(gaia_mb19284['parallax'][0], gaia_mb19284['parallax_error'][0]))
    print('PM RA : {0:2f} +/- {1:2f}'.format(gaia_mb19284['pmra'][0], gaia_mb19284['pmra_error'][0]))
    print('PM Dec : {0:2f} +/- {1:2f}'.format(gaia_mb19284['pmdec'][0], gaia_mb19284['pmdec_error'][0]))
    print('')
    print('OB170019')
    print('Mag (G) : ', gaia_ob170019['phot_g_mean_mag'][0])
    print('Parallax : {0:2f} +/- {1:2f}'.format(gaia_ob170019['parallax'][0], gaia_ob170019['parallax_error'][0]))
    print('PM RA : {0:2f} +/- {1:2f}'.format(gaia_ob170019['pmra'][0], gaia_ob170019['pmra_error'][0]))
    print('PM Dec : {0:2f} +/- {1:2f}'.format(gaia_ob170019['pmdec'][0], gaia_ob170019['pmdec_error'][0]))


def gaia_CMDs():
    plt.figure(1)
    plt.clf()
    plt.title('OB170095')
    plt.plot(gaia_ob170095['bp_rp'], gaia_ob170095['phot_g_mean_mag'], '.')
    plt.plot(gaia_ob170095['bp_rp'][0], gaia_ob170095['phot_g_mean_mag'][0], 'o')
    plt.xlabel('BP - RP')
    plt.ylabel('G')
    plt.gca().invert_yaxis()

    plt.figure(2)
    plt.clf()
    plt.title('MB19284')
    plt.plot(gaia_mb19284['bp_rp'], gaia_mb19284['phot_g_mean_mag'], '.')
    plt.plot(gaia_mb19284['bp_rp'][0], gaia_mb19284['phot_g_mean_mag'][0], 'o')
    plt.xlabel('BP - RP')
    plt.ylabel('G')
    plt.gca().invert_yaxis()

    plt.figure(3)
    plt.clf()
    plt.title('KB200101')
    plt.plot(gaia_kb200101['bp_rp'], gaia_kb200101['phot_g_mean_mag'], '.')
    plt.plot(gaia_kb200101['bp_rp'][0], gaia_kb200101['phot_g_mean_mag'][0], 'o')
    plt.xlabel('BP - RP')
    plt.ylabel('G')
    plt.gca().invert_yaxis()

    plt.figure(4)
    plt.clf()
    plt.title('OB170019')
    plt.plot(gaia_ob170019['bp_rp'], gaia_ob170019['phot_g_mean_mag'], '.')
    plt.plot(gaia_ob170019['bp_rp'][0], gaia_ob170019['phot_g_mean_mag'][0], 'o')
    plt.xlabel('BP - RP')
    plt.ylabel('G')
    plt.gca().invert_yaxis()


def gaia_VPDs():
    plt.figure(2)
    plt.clf()
    plt.title('MB19284')
    plt.errorbar(gaia_mb19284['pmra'], gaia_mb19284['pmdec'], 
                 xerr=gaia_mb19284['pmra_error'], yerr=gaia_mb19284['pmdec_error'], 
                 marker='.', ls='none')
    plt.errorbar(gaia_mb19284['pmra'][0], gaia_mb19284['pmdec'][0], 
                 xerr=gaia_mb19284['pmra_error'][0], yerr=gaia_mb19284['pmdec_error'][0], 
                 marker='.', ls='none')
    plt.xlabel('PM RA (mas/yr)')
    plt.ylabel('PM Dec (mas/yr)')
    plt.grid(True)
    plt.gca().set_aspect('equal', adjustable='box')

    plt.figure(3)
    plt.clf()
    plt.title('KB200101')
    plt.errorbar(gaia_kb200101['pmra'], gaia_kb200101['pmdec'], 
                 xerr=gaia_kb200101['pmra_error'], yerr=gaia_kb200101['pmdec_error'], 
                 marker='.', ls='none')
    plt.errorbar(gaia_kb200101['pmra'][0], gaia_kb200101['pmdec'][0], 
                 xerr=gaia_kb200101['pmra_error'][0], yerr=gaia_kb200101['pmdec_error'][0], 
                 marker='.', ls='none')
    plt.xlabel('PM RA (mas/yr)')
    plt.ylabel('PM Dec (mas/yr)')
    plt.grid(True)
    plt.gca().set_aspect('equal', adjustable='box')

    plt.figure(4)
    plt.clf()
    plt.title('OB170019')
    plt.errorbar(gaia_ob170019['pmra'], gaia_ob170019['pmdec'], 
                 xerr=gaia_ob170019['pmra_error'], yerr=gaia_ob170019['pmdec_error'], 
                 marker='.', ls='none')
    plt.errorbar(gaia_ob170019['pmra'][0], gaia_ob170019['pmdec'][0], 
                 xerr=gaia_ob170019['pmra_error'][0], yerr=gaia_ob170019['pmdec_error'][0], 
                 marker='.', ls='none')
    plt.xlabel('PM RA (mas/yr)')
    plt.ylabel('PM Dec (mas/yr)')
    plt.grid(True)
    plt.gca().set_aspect('equal', adjustable='box')


def plot_ob170095():
    # THIS IS NO USE
    # gaia_ob170095 = analysis.query_gaia(ra_ob170095, dec_ob170095, search_radius=30.0, table_name='gaiaedr3')

    # hdu = fits.open('/g/lu/data/microlens/20sep04os/raw/i200904_a012003_flip.fits')[0]
    # wcs = WCS(hdu.header)

    # image_data = hdu.data

    # fig = plt.figure()
    # ax = fig.add_axes([0.15, 0.1, 0.8, 0.8], projection=wcs)
    # ax.imshow(hdu.data, cmap='gray', norm=LogNorm(), origin='lower')
    # overlay = ax.get_coords_overlay('icrs')
#    ax.scatter(gaia_ob170095['ra'][0], gaia_ob170095['dec'][0], transform=ax.get_transform(), s=300,
#               edgecolor='white', facecolor='none')
    # print('Min:', np.min(image_data))
    # print('Max:', np.max(image_data))
    # print('Mean:', np.mean(image_data))
    # print('Stdev:', np.std(image_data))

    target = 'ob170095'
    mod_root = ob170095_dir
    mod_yaml = open(mod_root +  'params.yaml').read() 
    params = yaml.safe_load(mod_yaml)

    # OB170095 fit results.
    data = munge.getdata2(target,
                          phot_data=params['phot_data'],
                          ast_data=['Kp_Keck'])  

    mod_fit = model_fitter.PSPL_Solver(data,
                                       getattr(model, params['model']),
                                       add_error_on_photometry=params['add_error_on_photometry'],
                                       multiply_error_on_photometry=params['multiply_error_on_photometry'],
                                       outputfiles_basename=mod_root)
    

    best = mod_fit.get_best_fit(def_best='maxL')
    print(best)
    # best['u0_amp'] *= 1.0
    # best['piE_E'] *= 1.0
    # best['piE_N'] *= -1.0
    # best['tE'] = 106.
    # best['piE_E'] = -0.15
    # best['piE_N'] = 0.14
    thetaE = 0.25  # mas
    mod = mod_fit.get_model(best)

    # Calculate the model on a similar timescale to the data.
    tmax = np.max(np.append(data['t_phot1'], data['t_phot2'])) + 90.0
    t_mod_ast = np.arange(data['t_ast1'].min() - 180.0, tmax, 2)
    t_mod_pho = np.arange(data['t_phot1'].min(), tmax, 2)

    def get_astrometry_and_shift(mod, t):
        u0 = mod.u0.reshape(1, len(mod.u0))
        thetaE_hat = mod.thetaE_hat.reshape(1, len(mod.thetaE_hat))
        tau = (t - mod.t0) / mod.tE
        tau = tau.reshape(len(tau), 1)
    
        # Get the parallax vector for each date.
        parallax_vec = model.parallax_in_direction(mod.raL, mod.decL, t)

        # Shape of u: [N_times, 2]
        u = u0 + tau * thetaE_hat
        u -= mod.piE_amp * parallax_vec
        u_amp = np.linalg.norm(u, axis=1)

        # Calculate centroid offset in units of delta_c / thetaE
        denom = u_amp ** 2 + 2.0
        shift = u / denom.reshape((len(u_amp), 1))  # thetaE units

        return u, shift

    # Get the linear motion curves for the source (includes parallax)
    p_unlens_mod, p_shift_mod = get_astrometry_and_shift(mod, t_mod_ast)
    p_unlens_mod_at_ast, p_shift_mod_at_ast = get_astrometry_and_shift(mod, data['t_ast1'])

    # Get the lensed motion curves for the source
    p_lens_mod = p_unlens_mod + p_shift_mod
    p_lens_mod_at_ast = p_unlens_mod_at_ast + p_shift_mod_at_ast

    # Geth the photometry
    m_lens_mod = mod.get_photometry(t_mod_pho, filt_idx=0)
    m_lens_mod_at_phot1 = mod.get_photometry(data['t_phot1'], filt_idx=0)
    m_lens_mod_at_phot2 = mod.get_photometry(data['t_phot2'], filt_idx=1)

    # Calculate the delta-mag between R-band and K-band from the
    # flat part at the end.
    tidx = np.argmin(np.abs(data['t_phot1'] - data['t_ast1'][-1]))
    r_min_k = data['mag1'][tidx] - data['mag2'][-1]
    r_min_k = best['mag_base1'] - best['mag_base2']
    print('r_min_k = ', r_min_k)

    # Fix up the model astrometry data to fall on the data (just pos offset and thetaE scaling).
    p_unlens_mod *= thetaE
    p_unlens_mod_at_ast *= thetaE
    p_lens_mod *= thetaE
    p_lens_mod_at_ast *= thetaE

    dra = np.mean(p_lens_mod_at_ast[:, 0] - data['xpos1']*1e3)
    ddec = np.mean(p_lens_mod_at_ast[:, 1] - data['ypos1']*1e3)

    data['xpos1'] += dra/1e3
    data['ypos1'] += ddec/1e3

    # p_unlens_mod[:, 0] -= dra
    # p_unlens_mod[:, 1] -= ddec
    # p_lens_mod[:, 0] -= dra
    # p_lens_mod[:, 1] -= ddec

    
    # Plotting        
    # plt.close(3)
    plt.figure(3, figsize=(14, 4))
    plt.clf()

    pan_wid = 0.22
    pan_pad = 0.09
    fig_pos = np.arange(0, 3) * (pan_wid + pan_pad) + 1.5*pan_pad

    plt.figtext(0.25*pan_pad, 0.55, target.upper(), rotation='vertical',
                    fontweight='bold', fontsize=20,
                    va='center', ha='center')

    # Brightness vs. time
    fm1 = plt.gcf().add_axes([fig_pos[0], 0.36, pan_wid, 0.6])
    fm2 = plt.gcf().add_axes([fig_pos[0], 0.18, pan_wid, 0.2], sharex=fm1)
    fm1.errorbar(data['t_phot1'], data['mag1'], yerr=data['mag_err1'],
                 color = mpl_b, fmt='.', alpha=0.05)
    fm1.errorbar(data['t_phot2'], data['mag2'] + r_min_k, yerr=data['mag_err2'],
                 fmt='k.', alpha=0.9)
    fm1.plot(t_mod_pho, m_lens_mod, 'r-')
    fm2.errorbar(data['t_phot1'], data['mag1'] - m_lens_mod_at_phot1, yerr=data['mag_err1'],
                 color = mpl_b, fmt='.', alpha=0.05)
    fm2.errorbar(data['t_phot2'], data['mag2'] - m_lens_mod_at_phot2, yerr=data['mag_err2'],
                 fmt='k.', alpha=0.9)
    fm2.set_yticks(np.array([0.0, 0.2]))
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
    f1.errorbar(data['t_ast1'], data['xpos1']*1e3,
                    yerr=data['xpos_err1']*1e3, fmt='k.', zorder = 1000)
    f1.plot(t_mod_ast, p_lens_mod[:, 0], 'r-')
    f1.plot(t_mod_ast, p_unlens_mod[:, 0], 'r--')
    f1.get_xaxis().set_visible(False)
    f1.set_ylabel(r'$\Delta \alpha^*$ (mas)')
    f1.get_shared_x_axes().join(f1, f2)
    
    f2.errorbar(data['t_ast1'], (data['xpos1'] - p_unlens_mod_at_ast[:,0]) * 1e3,
                yerr=data['xpos_err1'] * 1e3, fmt='k.', alpha=1, zorder = 1000)
    f2.plot(t_mod_ast, (p_lens_mod[:, 0] - p_unlens_mod[:, 0]), 'r-')
    f2.axhline(0, linestyle='--', color='r')
    f2.xaxis.set_major_locator(plt.MaxNLocator(3))
    f2.set_ylim(-0.8, 0.8)
    f2.set_xlabel('Time (HJD)')
    f2.set_ylabel('Res.')

    
    # Dec vs. time
    f3 = plt.gcf().add_axes([fig_pos[2], 0.36, pan_wid, 0.6])
    f4 = plt.gcf().add_axes([fig_pos[2], 0.18, pan_wid, 0.2])
    f3.errorbar(data['t_ast1'], data['ypos1']*1e3,
                    yerr=data['ypos_err1']*1e3, fmt='k.', zorder = 1000)
    f3.plot(t_mod_ast, p_lens_mod[:, 1], 'r-')
    f3.plot(t_mod_ast, p_unlens_mod[:, 1], 'r--')
    f3.set_ylabel(r'$\Delta \delta$ (mas)')
    f3.yaxis.set_major_locator(plt.MaxNLocator(4))
    f3.get_xaxis().set_visible(False)
    f3.get_shared_x_axes().join(f3, f4)
    
    f4.errorbar(data['t_ast1'], (data['ypos1'] - p_unlens_mod_at_ast[:,1]) * 1e3,
                yerr=data['ypos_err1'] * 1e3, fmt='k.', alpha=1, zorder = 1000)
    f4.plot(t_mod_ast, (p_lens_mod[:, 1] - p_unlens_mod[:, 1]), 'r-')
    f4.axhline(0, linestyle='--', color='r')
    f4.xaxis.set_major_locator(plt.MaxNLocator(3))
#    f4.set_yticks(np.array([0.0, -0.2])) # For OB140613
    f4.set_ylim(-0.8, 0.8)
    f4.set_xlabel('Time (HJD)')
    f4.set_ylabel('Res.')

    plt.savefig(target + '_3panel_test.png')
    
    
    
    return

def get_gaia_centered():
    ob190017 = SkyCoord(ra_ob190017, dec_ob190017, frame='icrs', unit=(u.hourangle, u.deg))
    ob170095 = SkyCoord(ra_ob170095, dec_ob170095, frame='icrs', unit=(u.hourangle, u.deg))
    kb200101 = SkyCoord(ra_kb200101, dec_kb200101, frame='icrs', unit=(u.hourangle, u.deg))
    mb19284 = SkyCoord(ra_mb19284, dec_mb19284, frame='icrs', unit=(u.hourangle, u.deg))
    ob170019 = SkyCoord(ra_ob170019, dec_ob170019, frame='icrs', unit=(u.hourangle, u.deg))

    plt.figure(1)
    plt.clf()
    plt.title('OB190017')
    plt.scatter((gaia_ob190017['ra'] - ob190017.ra.deg)*3600, 
                (gaia_ob190017['dec'] - ob190017.dec.deg)*3600,
                s=gaia_ob190017['phot_g_mean_flux']/100)
    plt.scatter((gaia_ob190017['ra'][0] - ob190017.ra.deg)*3600, 
                (gaia_ob190017['dec'][0] - ob190017.dec.deg)*3600, 
                s=gaia_ob190017['phot_g_mean_flux'][0]/100)
    plt.grid(True)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.gca().invert_xaxis()
    plt.xlabel('$\Delta$ RA (arcsec)')
    plt.ylabel('$\Delta$ Dec (arcsec)')

    plt.figure(2)
    plt.clf()
    plt.title('OB170095')
    plt.scatter((gaia_ob170095['ra'] - ob170095.ra.deg)*3600, 
                (gaia_ob170095['dec'] - ob170095.dec.deg)*3600,
                s=gaia_ob170095['phot_g_mean_flux']/100)
    plt.scatter((gaia_ob170095['ra'][0] - ob170095.ra.deg)*3600, 
                (gaia_ob170095['dec'][0] - ob170095.dec.deg)*3600, 
                s=gaia_ob170095['phot_g_mean_flux'][0]/100)
    plt.grid(True)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.gca().invert_xaxis()
    plt.xlabel('$\Delta$ RA (arcsec)')
    plt.ylabel('$\Delta$ Dec (arcsec)')

    plt.figure(3)
    plt.clf()
    plt.title('KB200101')
    plt.scatter((gaia_kb200101['ra'] - kb200101.ra.deg)*3600, 
                (gaia_kb200101['dec'] - kb200101.dec.deg)*3600,
                s=gaia_kb200101['phot_g_mean_flux']/100)
    plt.scatter((gaia_kb200101['ra'][0] - kb200101.ra.deg)*3600, 
                (gaia_kb200101['dec'][0] - kb200101.dec.deg)*3600, 
                s=gaia_kb200101['phot_g_mean_flux'][0]/100)
    plt.grid(True)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.gca().invert_xaxis()
    plt.xlabel('$\Delta$ RA (arcsec)')
    plt.ylabel('$\Delta$ Dec (arcsec)')

    plt.figure(4)
    plt.clf()
    plt.title('MB19284')
    plt.scatter((gaia_mb19284['ra'] - mb19284.ra.deg)*3600, 
                (gaia_mb19284['dec'] - mb19284.dec.deg)*3600,
                s=gaia_mb19284['phot_g_mean_flux']/100)
    plt.scatter((gaia_mb19284['ra'][0] - mb19284.ra.deg)*3600, 
                (gaia_mb19284['dec'][0] - mb19284.dec.deg)*3600, 
                s=gaia_mb19284['phot_g_mean_flux'][0]/100)
    plt.grid(True)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.gca().invert_xaxis()
    plt.xlabel('$\Delta$ RA (arcsec)')
    plt.ylabel('$\Delta$ Dec (arcsec)')

    plt.figure(5)
    plt.clf()
    plt.title('OB170019')
    plt.scatter((gaia_ob170019['ra'] - ob170019.ra.deg)*3600, 
                (gaia_ob170019['dec'] - ob170019.dec.deg)*3600,
                s=gaia_ob170019['phot_g_mean_flux']/100)
    plt.scatter((gaia_ob170019['ra'][0] - ob170019.ra.deg)*3600, 
                (gaia_ob170019['dec'][0] - ob170019.dec.deg)*3600, 
                s=gaia_ob170019['phot_g_mean_flux'][0]/100)
    plt.grid(True)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.gca().invert_xaxis()
    plt.xlabel('$\Delta$ RA (arcsec)')
    plt.ylabel('$\Delta$ Dec (arcsec)')

    return


def get_gaia():
    plt.figure(1)
    plt.clf()
    plt.title('OB190017')
#    plt.scatter(gaia_ob190017['ra'], gaia_ob190017['dec'], s=gaia_ob190017['phot_rp_mean_flux']/100)
    plt.scatter(gaia_ob190017['ra'], gaia_ob190017['dec'], s=gaia_ob190017['phot_g_mean_flux']/100)
    plt.scatter(gaia_ob190017['ra'][0], gaia_ob190017['dec'][0], s=gaia_ob190017['phot_g_mean_flux'][0]/100)
    plt.gca().invert_xaxis()

    plt.figure(2)
    plt.clf()
    plt.title('OB170095')
#    plt.scatter(gaia_ob170095['ra'], gaia_ob170095['dec'], s=gaia_ob170095['phot_rp_mean_flux']/100)
    plt.scatter(gaia_ob170095['ra'], gaia_ob170095['dec'], s=gaia_ob170095['phot_g_mean_flux']/100)
    plt.scatter(gaia_ob170095['ra'][0], gaia_ob170095['dec'][0], s=gaia_ob170095['phot_g_mean_flux'][0]/100)
    plt.gca().invert_xaxis()

    plt.figure(3)
    plt.clf()
    plt.title('KB200101')
#    plt.scatter(gaia_kb200101['ra'], gaia_kb200101['dec'], s=gaia_kb200101['phot_rp_mean_flux']/100)
    plt.scatter(gaia_kb200101['ra'], gaia_kb200101['dec'], s=gaia_kb200101['phot_g_mean_flux']/100)
    plt.scatter(gaia_kb200101['ra'][0], gaia_kb200101['dec'][0], s=gaia_kb200101['phot_g_mean_flux'][0]/100)
    plt.gca().invert_xaxis()

    plt.figure(4)
    plt.clf()
    plt.title('MB19284')
#    plt.scatter(gaia_mb19284['ra'], gaia_mb19284['dec'], s=gaia_mb19284['phot_rp_mean_flux']/100)
    plt.scatter(gaia_mb19284['ra'], gaia_mb19284['dec'], s=gaia_mb19284['phot_g_mean_flux']/100)
    plt.scatter(gaia_mb19284['ra'][0], gaia_mb19284['dec'][0], s=gaia_mb19284['phot_g_mean_flux'][0]/100)
    plt.gca().invert_xaxis()

    plt.figure(5)
    plt.clf()
    plt.title('OB170019')
#    plt.scatter(gaia_ob170019['ra'], gaia_ob170019['dec'], s=gaia_ob170019['phot_rp_mean_flux']/100)
    plt.scatter(gaia_ob170019['ra'], gaia_ob170019['dec'], s=gaia_ob170019['phot_g_mean_flux']/100)
    plt.scatter(gaia_ob170019['ra'][0], gaia_ob170019['dec'][0], s=gaia_ob170019['phot_g_mean_flux'][0]/100)
    plt.gca().invert_xaxis()

    return
 
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

def make_plots():
    deltac_vs_time(0.061, 200/365.25, 4, '2020-06-22', 
                   ['2020-07-12', '2020-07-22', '2020-08-22', '2020-09-03'],
                   t_obs_prop=['2021-08-01', '2021-09-01'],
                   title='KB200101')

    deltac_vs_time(0.18, 258.6/365.25, 4, '2019-04-13', 
                   ['2019-04-21', '2019-05-13', '2020-07-22', '2020-08-23', '2020-09-03'],
                   t_obs_prop=['2021-08-01', '2021-09-01'],
                   title='OB190017')

    deltac_vs_time(0.03, 105.9/365.25, 5, '2017-05-19', 
                   ['2017-05-21', '2017-06-08', '2017-07-14', '2017-07-19', 
                    '2020-06-13', '2020-08-23', '2020-09-04'],
                   t_obs_prop=['2021-08-01'],
                   title='OB170095')

    deltac_vs_time(0.165, 648.454/365.25, 5, '2020-12-23', 
                   ['2020-06-25', '2020-07-12', '2020-07-22', '2020-08-22'], 
                   t_obs_prop=['2021-08-01', '2021-09-01'],
                   title='MB19284')

def make_reg_file():
    table_file = 'mb19284.reg'
    if os.path.exists(table_file):
        os.remove(table_file)

    ra = gaia_mb19284['ra']
    dec = gaia_mb19284['dec']

    if os.path.exists(table_file):
        os.remove(table_file)

    with open(table_file, 'a+') as tb:
        tb.write('global color=green dashlist=8 3 width=1 font="helvetica 10 normal" select=1 highlite=1 dash=0 fixed=0 edit=1 move=1 delete=1 include=1 source=1 \n')
        tb.write('icrs \n')
        for i in np.arange(len(ra)):
            tb.write('circle( {0}, {1}, 0.0001) \n'.format(ra[i], dec[i]))

def tE_BH():
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

    bins = np.logspace(-0.5, 2.7, 26)
    
    fig = plt.figure(1, figsize = (6,5))
    plt.clf()
    plt.subplots_adjust(left = 0.17, top = 0.8, bottom = 0.2)
    plt.hist(t['t_E'], bins = bins,
             histtype = 'step', color = 'tab:blue')
    plt.hist(t['t_E'][bh_idx], bins = bins,
             histtype = 'step', color = 'tab:orange')
    plt.text(0.3, 100, 'All events', color = 'tab:blue')
    plt.text(2.2, 8, 'BH events', color = 'tab:orange')
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('$t_E$ (days)')
    plt.ylabel('Number of events')
    plt.ylim(1, 5000)
    plt.axvline(x = 120, color = 'tab:red')
    plt.text(130, 60, '$t_E = 120$ days', color = 'tab:red', rotation=90)
    plt.savefig('tE.png')

    return


def piE_tE_deltac():
    span=0.999999426697
    smooth=0.02
    quantiles_2d=None
    hist2d_kwargs=None
    labels=None
    label_kwargs=None
    show_titles=False 
    title_fmt=".2f" 
    title_kwargs=None
    """
    !!! NOTE: CHOICE OF THE quantiles_2d HAS A LARGE EFFECT 
    ON THE WAY THIS PLOT LOOKS !!!
    Plot piE-tE 2D posteriors from OGLE photometry only fits.
    Also plot PopSyCLE simulations simultaneously.
    """
    # Initialize values.
    if label_kwargs is None:
        label_kwargs = dict()
    if title_kwargs is None:
        title_kwargs = dict()
    if hist2d_kwargs is None:
        hist2d_kwargs = dict()

    # Set defaults.
    hist2d_kwargs['alpha'] = hist2d_kwargs.get('alpha', 0.2)
    hist2d_kwargs['levels'] = hist2d_kwargs.get('levels', quantiles_2d)

    ob120169_yaml = open(ob120169_dir + ob120169_id +  '_params.yaml').read() 
    params_120169 = yaml.safe_load(ob120169_yaml)
    ob140613_yaml = open(ob140613_dir + ob140613_id +  '_params.yaml').read() 
    params_140613 = yaml.safe_load(ob140613_yaml)
    ob150029_yaml = open(ob150029_dir + ob150029_id +  '_params.yaml').read() 
    params_150029 = yaml.safe_load(ob150029_yaml)
    ob150211_yaml = open(ob150211_dir + ob150211_id +  '_params.yaml').read() 
    params_150211 = yaml.safe_load(ob150211_yaml)

    # OB120169 fit results.
    data_120169 = munge.getdata2('ob120169',
                                 phot_data=params_120169['phot_data'],
                                 ast_data=params_120169['astrom_data'])  

    fitter_120169 = model_fitter.PSPL_Solver(data_120169,
                                             getattr(model, params_120169['model']),
                                             add_error_on_photometry=params_120169['add_error_on_photometry'],
                                             multiply_error_on_photometry=params_120169['multiply_error_on_photometry'],
                                             outputfiles_basename=ob120169_dir + ob120169_id + '_')

    results_120169 = fitter_120169.load_mnest_results_for_dynesty()
    smy_120169 = fitter_120169.load_mnest_summary()

    # OB140613 fit results.
    data_140613 = munge.getdata2('ob140613',
                                 phot_data=params_140613['phot_data'],
                                 ast_data=params_140613['astrom_data'])  

    fitter_140613 = model_fitter.PSPL_Solver(data_140613,
                                             getattr(model, params_140613['model']),
                                             add_error_on_photometry=params_140613['add_error_on_photometry'],
                                             multiply_error_on_photometry=params_140613['multiply_error_on_photometry'],
                                             outputfiles_basename=ob140613_dir + ob140613_id + '_')

    results_140613 = fitter_140613.load_mnest_results_for_dynesty()
    smy_140613 = fitter_140613.load_mnest_summary()

    # OB150029 fit results.
    data_150029 = munge.getdata2('ob150029',
                                 phot_data=params_150029['phot_data'],
                                 ast_data=params_150029['astrom_data'])  

    fitter_150029 = model_fitter.PSPL_Solver(data_150029,
                                             getattr(model, params_150029['model']),
                                             add_error_on_photometry=params_150029['add_error_on_photometry'],
                                             multiply_error_on_photometry=params_150029['multiply_error_on_photometry'],
                                             outputfiles_basename=ob150029_dir + ob150029_id + '_')

    results_150029 = fitter_150029.load_mnest_results_for_dynesty()
    smy_150029 = fitter_150029.load_mnest_summary()

    # OB150211 fit results.
    data_150211 = munge.getdata2('ob150211',
                                 phot_data=params_150211['phot_data'],
                                 ast_data=params_150211['astrom_data'])  

    fitter_150211 = model_fitter.PSPL_Solver(data_150211,
                                             getattr(model, params_150211['model']),
                                             add_error_on_photometry=params_150211['add_error_on_photometry'],
                                             multiply_error_on_photometry=params_150211['multiply_error_on_photometry'],
                                             outputfiles_basename=ob150211_dir + ob150211_id + '_')

    results_150211 = fitter_150211.load_mnest_results_for_dynesty()
    smy_150211 = fitter_150211.load_mnest_summary()

    # Extract weighted samples.
    samples_120169 = results_120169['samples']
    samples_140613 = results_140613['samples']
    samples_150029 = results_150029['samples']
    samples_150211 = results_150211['samples']

    try:
        weights_120169 = np.exp(results_120169['logwt'] - results_120169['logz'][-1])
        weights_140613 = np.exp(results_140613['logwt'] - results_140613['logz'][-1])
        weights_150029 = np.exp(results_150029['logwt'] - results_150029['logz'][-1])
        weights_150211 = np.exp(results_150211['logwt'] - results_150211['logz'][-1])
    except:
        weights_120169 = results_120169['weights']
        weights_140613 = results_140613['weights']
        weights_150029 = results_150029['weights']
        weights_150211 = results_150211['weights']

    # Deal with 1D results. A number of extra catches are also here
    # in case users are trying to plot other results besides the `Results`
    # instance generated by `dynesty`.
    samples_120169 = np.atleast_1d(samples_120169)
    if len(samples_120169.shape) == 1:
        samples_120169 = np.atleast_2d(samples_120169)
    else:
        assert len(samples_120169.shape) == 2, "Samples must be 1- or 2-D."
        samples_120169 = samples_120169.T
    assert samples_120169.shape[0] <= samples_120169.shape[1], "There are more " \
                                                 "dimensions than samples!"
    
    samples_140613 = np.atleast_1d(samples_140613)
    if len(samples_140613.shape) == 1:
        samples_140613 = np.atleast_2d(samples_140613)
    else:
        assert len(samples_140613.shape) == 2, "Samples must be 1- or 2-D."
        samples_140613 = samples_140613.T
    assert samples_140613.shape[0] <= samples_140613.shape[1], "There are more " \
                                                 "dimensions than samples!"

    samples_150029 = np.atleast_1d(samples_150029)
    if len(samples_150029.shape) == 1:
        samples_150029 = np.atleast_2d(samples_150029)
    else:
        assert len(samples_150029.shape) == 2, "Samples must be 1- or 2-D."
        samples_150029 = samples_150029.T
    assert samples_150029.shape[0] <= samples_150029.shape[1], "There are more " \
                                                 "dimensions than samples!"

    samples_150211 = np.atleast_1d(samples_150211)
    if len(samples_150211.shape) == 1:
        samples_150211 = np.atleast_2d(samples_150211)
    else:
        assert len(samples_150211.shape) == 2, "Samples must be 1- or 2-D."
        samples_150211 = samples_150211.T
    assert samples_150211.shape[0] <= samples_150211.shape[1], "There are more " \
                                                 "dimensions than samples!"

    # Maximum likelihood vals                                                                                         
    smy_list = [smy_120169, smy_140613, smy_150029, smy_150211]

    smy_name = ['OB120169','OB140613', 'OB150029', 'OB150211']
    maxl = {}

    for ss, smy in enumerate(smy_list):
        print(smy_name[ss])
        print('tE : ', smy['MaxLike_tE'][0])
        print('piE : ', np.hypot(smy['MaxLike_piE_E'][0], smy['MaxLike_piE_N'][0]))
        maxl[smy_name[ss]] = {'tE' : smy['MaxLike_tE'][0],
                              'piE' : np.hypot(smy['MaxLike_piE_E'][0], smy['MaxLike_piE_N'][0])}

    #############
    # The observations. 
    # OB110022 from Lu+16.
    # Finished targets are phot + astrom solutions
    # Ongoing targets are phot parallax solutions 
    # (global MEDIAN +/- 1 sigma)
    #############
    # OB110022
    piEE_110022 = -0.393
    piEE_110022_pe = 0.013
    piEE_110022_me = 0.012
    piEN_110022 = -0.071
    piEN_110022_pe = 0.013
    piEN_110022_me = 0.012
    piE_110022, piE_110022_pe, piE_110022_me = calc_hypot_and_err(piEE_110022, piEE_110022_pe, piEE_110022_me,
                                                                  piEN_110022, piEN_110022_pe, piEN_110022_me)
    tE_110022 = 61.4
    tE_110022_pe = 1.0
    tE_110022_me = 1.0
    
    # This is an upper limit.
    dcmax_110022 = 2.19/np.sqrt(8)
    dcmax_110022_pe = 1.06/np.sqrt(8)
    dcmax_110022_me = 1.17/np.sqrt(8)

    # Plot the piE-tE 2D posteriors.
    # tE = 2; piEE,N = 3, 4 
    fig, ax = plt.subplots(1, 2, figsize=(14,6), sharey=True,
                           gridspec_kw={'width_ratios': [1, 1.4]})
    plt.subplots_adjust(left=0.1, bottom=0.15, wspace=0.1)

    tE_120169 = samples_120169[2]
    tE_140613 = samples_140613[2]
    tE_150029 = samples_150029[2]
    tE_150211 = samples_150211[2]

    thetaE_120169 = samples_120169[3]
    thetaE_140613 = samples_140613[3]
    thetaE_150029 = samples_150029[3]
    thetaE_150211 = samples_150211[3]

    piE_120169 = np.hypot(samples_120169[5], samples_120169[6])
    piE_140613 = np.hypot(samples_140613[5], samples_140613[6])
    piE_150029 = np.hypot(samples_150029[5], samples_150029[6])
    piE_150211 = np.hypot(samples_150211[5], samples_150211[6])

    sx = smooth
    sy = smooth

    hist2d_kwargs['fill_contours'] = hist2d_kwargs.get('fill_contours',
                                                       False)
    hist2d_kwargs['plot_contours'] = hist2d_kwargs.get('plot_contours',
                                                       True)

    ax[0].errorbar(dcmax_110022, piE_110022, 
                   xerr = np.array([[dcmax_110022_me], [dcmax_110022_pe]]), 
                   fmt = '*', color = 'cyan', markersize = 15,
                   xuplims = True)
    model_fitter.contour2d_alpha(thetaE_120169/np.sqrt(8), piE_120169, span=[span, span], quantiles_2d=quantiles_2d,
                                 weights=weights_120169, ax=ax[0], smooth=[sy, sx], color='blue',
                                 **hist2d_kwargs, plot_density=False)
    model_fitter.contour2d_alpha(thetaE_140613/np.sqrt(8), piE_140613, span=[span, span], quantiles_2d=quantiles_2d,
                                 weights=weights_140613, ax=ax[0], smooth=[sy, sx], color='hotpink', 
                                 **hist2d_kwargs, plot_density=False)
    model_fitter.contour2d_alpha(thetaE_150029/np.sqrt(8), piE_150029, span=[span, span], quantiles_2d=quantiles_2d,
                                 weights=weights_150029, ax=ax[0], smooth=[sy, sx], color='red',
                                 **hist2d_kwargs, plot_density=False)
    model_fitter.contour2d_alpha(thetaE_150211/np.sqrt(8), piE_150211, span=[span, span], quantiles_2d=quantiles_2d,
                                 weights=weights_150211, ax=ax[0], smooth=[sy, sx], color='dodgerblue', 
                                 **hist2d_kwargs, plot_density=False)

    ax[1].plot(tE_110022, piE_110022, color='cyan', marker='*', ms = 15)
    model_fitter.contour2d_alpha(tE_120169, piE_120169, span=[span, span], quantiles_2d=quantiles_2d,
                                 weights=weights_120169, ax=ax[1], smooth=[sy, sx], color='blue',
                                 **hist2d_kwargs, plot_density=False)
    model_fitter.contour2d_alpha(tE_150211, piE_150211, span=[span, span], quantiles_2d=quantiles_2d,
                                 weights=weights_150211, ax=ax[1], smooth=[sy, sx], color='dodgerblue', 
                                 **hist2d_kwargs, plot_density=False)
#    ax[1].plot(maxl['OB140613']['tE'], maxl['OB140613']['piE'], color='hotpink', marker = '*', ms = 15)
#    ax[1].plot(maxl['OB150029']['tE'], maxl['OB150029']['piE'], color='red', marker='*', ms = 15)

    ax[1].plot(0.01, 100, color = 'cyan', label='OB110022')
    ax[1].plot(0.01, 100, color='blue', label='OB120169')
    ax[1].plot(0.01, 100, color='hotpink', label='OB140613')
    ax[1].plot(0.01, 100, color='red', label='OB150029')
    ax[1].plot(0.01, 100, color='dodgerblue', label='OB150211')
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

    ax[0].scatter(final_delta_arr[st_idx], t['pi_E'][st_idx], 
                  alpha = 0.4, marker = '.', s = 25,
                  c = 'gold')
    ax[0].scatter(final_delta_arr[wd_idx], t['pi_E'][wd_idx], 
                  alpha = 0.4, marker = '.', s = 25,
                  c = 'goldenrod')
    ax[0].scatter(final_delta_arr[ns_idx], t['pi_E'][ns_idx], 
                  alpha = 0.4, marker = '.', s = 25,
                  c = 'sienna')
    ax[0].scatter(final_delta_arr[bh_idx], t['pi_E'][bh_idx], 
                  alpha = 0.8, marker = '.', s = 25,
                  c = 'black')
    
    ax[1].scatter(t['t_E'][st_idx], t['pi_E'][st_idx], 
                alpha = 0.4, marker = '.', s = 25, 
                color = 'gold')
    ax[1].scatter(t['t_E'][wd_idx], t['pi_E'][wd_idx], 
                alpha = 0.4, marker = '.', s = 25, 
                color = 'goldenrod')
    ax[1].scatter(t['t_E'][ns_idx], t['pi_E'][ns_idx], 
                alpha = 0.4, marker = '.', s = 25, 
                color = 'sienna')
    ax[1].scatter(t['t_E'][bh_idx], t['pi_E'][bh_idx],
                alpha = 0.8, marker = '.', s = 25, 
                color = 'black')
    # Trickery to make the legend darker
    ax[1].scatter(0.01, 100, 
                alpha = 0.8, marker = 'o', s = 25, 
                label = 'Star', color = 'gold')
    ax[1].scatter(0.01, 100, 
                alpha = 0.8, marker = 'o', s = 25,
                label = 'WD', color = 'goldenrod')
    ax[1].scatter(0.01, 100,
                alpha = 0.8, marker = 'o', s = 25, 
                label = 'NS', color = 'sienna')
    ax[1].scatter(0.01, 100,
                alpha = 0.8, marker = 'o', s = 25, 
                label = 'BH', color = 'black')
    ax[0].set_xlabel('$\delta_{c,max}$ (mas)')
    ax[0].set_ylabel('$\pi_E$')
    ax[0].set_xscale('log')
    ax[0].set_yscale('log')
    ax[0].set_xlim(0.005, 4)
    ax[0].set_ylim(0.009, 0.5)
    ax[1].set_xlim(10, 400)
    ax[1].set_ylim(0.009, 0.5)
    ax[1].set_xlabel('$t_E$ (days)')
    ax[1].set_xscale('log')
    ax[1].set_yscale('log')
    box = ax[1].get_position()
    ax[1].set_position([box.x0, box.y0, box.width * 0.7, box.height])
    ax[1].legend(bbox_to_anchor=(1.5, 0.5), loc="center right")
    plt.savefig('piE_tE_deltac.png')

    results_mass_120169 = {}
    results_mass_120169['weights'] = results_120169['weights']
    results_mass_120169['logvol'] = results_120169['logvol']
    results_mass_120169['samples'] = results_120169['samples'][:,17].reshape(len(results_120169['samples']), 1)
    
#    # Plot the mL-?? 2D posteriors.
#    fig, ax = plt.subplots(1, 2, figsize=(14,6), sharey=True,
#                           gridspec_kw={'width_ratios': [1, 1.4]})
#    plt.subplots_adjust(left=0.1, bottom=0.15, wspace=0.1)
#
#    mL_120169 = samples_120169[17]
#    mL_140613 = samples_140613[17]
#    mL_150029 = samples_150029[17]
#    mL_150211 = samples_150211[17]
#
#    sx = smooth
#    sy = smooth
#
#    hist2d_kwargs['fill_contours'] = hist2d_kwargs.get('fill_contours',
#                                                       False)
#    hist2d_kwargs['plot_contours'] = hist2d_kwargs.get('plot_contours',
#                                                       True)
#    model_fitter.contour2d_alpha(thetaE_120169/np.sqrt(8), mL_120169, span=[span, span], quantiles_2d=quantiles_2d,
#                                 weights=weights_120169, ax=ax[0], smooth=[sy, sx], color='blue',
#                                 **hist2d_kwargs, plot_density=False)
#    model_fitter.contour2d_alpha(thetaE_140613/np.sqrt(8), mL_140613, span=[span, span], quantiles_2d=quantiles_2d,
#                                 weights=weights_140613, ax=ax[0], smooth=[sy, sx], color='hotpink', 
#                                 **hist2d_kwargs, plot_density=False)
#    model_fitter.contour2d_alpha(thetaE_150029/np.sqrt(8), mL_150029, span=[span, span], quantiles_2d=quantiles_2d,
#                                 weights=weights_150029, ax=ax[0], smooth=[sy, sx], color='red',
#                                 **hist2d_kwargs, plot_density=False)
#    model_fitter.contour2d_alpha(thetaE_150211/np.sqrt(8), mL_150211, span=[span, span], quantiles_2d=quantiles_2d,
#                                 weights=weights_150211, ax=ax[0], smooth=[sy, sx], color='dodgerblue', 
#                                 **hist2d_kwargs, plot_density=False)
#
#    model_fitter.contour2d_alpha(piE_120169, mL_120169, span=[span, span], quantiles_2d=quantiles_2d,
#                                 weights=weights_120169, ax=ax[1], smooth=[sy, sx], color='blue',
#                                 **hist2d_kwargs, plot_density=False)
#    model_fitter.contour2d_alpha(piE_140613, mL_140613, span=[span, span], quantiles_2d=quantiles_2d,
#                                 weights=weights_140613, ax=ax[1], smooth=[sy, sx], color='hotpink', 
#                                 **hist2d_kwargs, plot_density=False)
#    model_fitter.contour2d_alpha(piE_150029, mL_150029, span=[span, span], quantiles_2d=quantiles_2d,
#                                 weights=weights_150029, ax=ax[1], smooth=[sy, sx], color='red',
#                                 **hist2d_kwargs, plot_density=False)
#    model_fitter.contour2d_alpha(piE_150211, mL_150211, span=[span, span], quantiles_2d=quantiles_2d,
#                                 weights=weights_150211, ax=ax[1], smooth=[sy, sx], color='dodgerblue', 
#                                 **hist2d_kwargs, plot_density=False)
#
#    ax[1].plot(0.01, 100, color='blue', label='OB120169')
#    ax[1].plot(0.01, 100, color='hotpink', label='OB140613')
#    ax[1].plot(0.01, 100, color='red', label='OB150029')
#    ax[1].plot(0.01, 100, color='dodgerblue', label='OB150211')
#
#    ax[0].set_xlabel('$\delta_{c,max}$ (mas)')
#    ax[0].set_ylabel('$M_L$ ($M_\odot$)')
##    ax[0].set_xscale('log')
##    ax[0].set_yscale('log')
#    ax[0].set_ylim(0.08, 12)
#
#    ax[1].set_xlabel('$\pi_E$')
#    ax[1].set_xlim(0.009, 0.5)
#    ax[1].set_xscale('log')
##    ax[1].set_yscale('log')
#    ax[1].set_ylim(0.08, 12)
#    plt.legend()
#    box = ax[1].get_position()
#    ax[1].set_position([box.x0, box.y0, box.width * 0.7, box.height])
#    ax[1].legend(bbox_to_anchor=(1.5, 0.5), loc="center right")

def plot_mass_post():
    post_120169 = np.loadtxt(ob120169_dir + ob120169_id + '_.txt')
    post_140613 = np.loadtxt(ob140613_dir + ob140613_id + '_.txt')
    post_150029 = np.loadtxt(ob150029_dir + ob150029_id + '_.txt')
    post_150211 = np.loadtxt(ob150211_dir + ob150211_id + '_.txt')

    bins = np.linspace(0.08, 12, 50)    
    fig = plt.figure(1, figsize=(6,6))
    plt.clf()
    plt.hist(post_120169[:, 19], weights = post_120169[:,0], 
             bins = bins, histtype = 'step', density=True,
             label = 'OB120169', lw = 2, color='blue', alpha = 0.8)
    plt.hist(post_140613[:, 19], weights = post_140613[:,0], 
             bins = bins, histtype = 'step', density=True,
             label = 'OB140613', lw = 2, color='hotpink', alpha = 0.8)
    plt.hist(post_150029[:, 19], weights = post_150029[:,0], 
             bins = bins, histtype = 'step', density=True,
             label = 'OB150029', lw = 2, color='red', alpha = 0.8)
    plt.hist(post_150211[:, 19], weights = post_150211[:,0], 
             bins = bins, histtype = 'step', density=True,
             label = 'OB150211', lw = 2, color='dodgerblue', alpha = 0.8)
    plt.legend()
    plt.xlabel('$M_L (M_\odot)$')
    plt.ylabel('Probability density')
    plt.savefig('mass_post.png')

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


def calc_hypot_and_err(A, sig_A_p, sig_A_m, B, sig_B_p, sig_B_m):
    """
    For some quantities A and B, calculate f and sigma_f, where
    f = \sqrt(A^2 + B^2)
    sigma_f = \sqrt( (A/f)^2 sigma_A^2 + (B/f)^2 sigma B^2).
    
    Parameters
    ----------
    A, B : median value.
    sig_A,B_p, sig_A,B_m : +/- 1 sigma values.
    
    Return
    ------
    f : see formula for f above
    sigma_f_p : see formula for sigma_f above, calculate with +1 sigma value
    sigma_f_m : see formula for sigma_f above, calculate with -1 sigma value
    
    """
    Af2 = A**2/(A**2 + B**2)
    Bf2 = B**2/(A**2 + B**2)
    
    f = np.sqrt(A**2 + B**2)
    sigma_f_p = np.sqrt(Af2 * sig_A_p**2 + Bf2 * sig_B_p**2)
    sigma_f_m = np.sqrt(Af2 * sig_A_m**2 + Bf2 * sig_B_m**2)
    
    return f, sigma_f_p, sigma_f_m


def plot_airmass_moon():
    # coordinates to center of OGLE field BLG502 (2017 objects)
    ra = "18:00:00"
    dec = "-30:00:00"
    months = np.array([8, 9])
    days = np.array([1, 1])
    # outdir = '/Users/jlu/doc/proposals/keck/uc/18A/'
    outdir = '/u/casey/scratch/code/JLU-python-code/jlu/papers/'

    # Keck 2
    skycalc.plot_airmass(ra, dec, 2021, months, days, 'keck2', outfile=outdir + 'microlens_airmass_keck2_21B.png', date_idx=-1)
    skycalc.plot_moon(ra, dec, 2021, np.array([8, 9]), outfile=outdir + 'microlens_moon_21B.png')

    # Keck 1
    skycalc.plot_airmass(ra, dec, 2021, months, days, 'keck1', outfile=outdir + 'microlens_airmass_keck1_21B.png', date_idx=-1)
    
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
    mod_root = ob140613_dir + ob140613_id + '_'
    target = 'ob140613'
    
    mod_yaml = open(mod_root +  'params.yaml').read() 
    params = yaml.safe_load(mod_yaml)

    # OB140613 fit results.
    data = munge.getdata2(target,
                          phot_data=params['phot_data'],
                          ast_data=params['astrom_data'])  

    mod_fit = model_fitter.PSPL_Solver(data,
                                             getattr(model, params['model']),
                                             add_error_on_photometry=params['add_error_on_photometry'],
                                             multiply_error_on_photometry=params['multiply_error_on_photometry'],
                                             outputfiles_basename=mod_root)
    

    mod_all = mod_fit.get_best_fit_modes_model(def_best='maxL')
    tab_all = mod_fit.load_mnest_modes()

    plot_3panel(data, mod_all[0], tab_all[0], target + '_phot_astrom.png', target,
                    r_min_k=4.0, mass_max_lim=4, log=False)

    return

def plot_ob150211_phot_ast():
    mod_root = ob150211_dir + ob150211_id + '_'
    target = 'ob150211'
    
    mod_yaml = open(mod_root +  'params.yaml').read() 
    params = yaml.safe_load(mod_yaml)

    # fit results.
    data = munge.getdata2(target,
                          phot_data=params['phot_data'],
                          ast_data=params['astrom_data'])  

    mod_fit = model_fitter.PSPL_Solver(data,
                                             getattr(model, params['model']),
                                             add_error_on_photometry=params['add_error_on_photometry'],
                                             multiply_error_on_photometry=params['multiply_error_on_photometry'],
                                             outputfiles_basename=mod_root)
    

    mod_all = mod_fit.get_best_fit_modes_model(def_best='maxL')
    tab_all = mod_fit.load_mnest_modes()

    plot_3panel(data, mod_all[0], tab_all[0], target + '_phot_astrom.png', target,
                    r_min_k=4.0, mass_max_lim=4, log=False)

    return

def plot_ob120169_phot_ast():
    mod_root = ob120169_dir + ob120169_id + '_'
    target = 'ob120169'
    
    mod_yaml = open(mod_root +  'params.yaml').read() 
    params = yaml.safe_load(mod_yaml)

    # fit results.
    data = munge.getdata2(target,
                          phot_data=params['phot_data'],
                          ast_data=params['astrom_data'])  

    mod_fit = model_fitter.PSPL_Solver(data,
                                             getattr(model, params['model']),
                                             add_error_on_photometry=params['add_error_on_photometry'],
                                             multiply_error_on_photometry=params['multiply_error_on_photometry'],
                                             outputfiles_basename=mod_root)
    

    mod_all = mod_fit.get_best_fit_modes_model(def_best='maxL')
    tab_all = mod_fit.load_mnest_modes()

    plot_3panel(data, mod_all[0], tab_all[0], target + '_phot_astrom.png', target,
                    r_min_k=4.0, mass_max_lim=4, log=False)

    return

def plot_ob150029_phot_ast():
    mod_root = ob150029_dir + ob150029_id + '_'
    target = 'ob150029'
    
    mod_yaml = open(mod_root +  'params.yaml').read() 
    params = yaml.safe_load(mod_yaml)

    # fit results.
    data = munge.getdata2(target,
                          phot_data=params['phot_data'],
                          ast_data=params['astrom_data'])  

    mod_fit = model_fitter.PSPL_Solver(data,
                                             getattr(model, params['model']),
                                             add_error_on_photometry=params['add_error_on_photometry'],
                                             multiply_error_on_photometry=params['multiply_error_on_photometry'],
                                             outputfiles_basename=mod_root)
    

    mod_all = mod_fit.get_best_fit_modes_model(def_best='maxL')
    tab_all = mod_fit.load_mnest_modes()

    plot_3panel(data, mod_all[0], tab_all[0], target + '_phot_astrom.png', target,
                    r_min_k=4.0, mass_max_lim=4, log=False)

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
    # plt.close(3)
    plt.figure(3, figsize=(18, 4))
    plt.clf()

    pan_wid = 0.15
    pan_pad = 0.09
    fig_pos = np.arange(0, 4) * (pan_wid + pan_pad) + pan_pad

    # Brightness vs. time
    fm1 = plt.gcf().add_axes([fig_pos[0], 0.36, pan_wid, 0.6])
    fm2 = plt.gcf().add_axes([fig_pos[0], 0.18, pan_wid, 0.2])
    fm1.errorbar(data['t_phot1'], data['mag1'], yerr=data['mag_err1'],
                 color = 'tab:blue', fmt='.', alpha=0.05)
    # fm1.errorbar(data['t_phot2'], data['mag2'] + r_min_k, yerr=data['mag_err2'],
    #              fmt='k.', alpha=0.9)
    fm1.plot(t_mod_pho, m_lens_mod, 'r-')
    fm2.errorbar(data['t_phot1'], data['mag1'] - m_lens_mod_at_phot1, yerr=data['mag_err1'],
                 color = 'tab:blue', fmt='.', alpha=0.05)
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
    f2.set_ylim(-0.8, 0.8)
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
    f4.set_ylim(-0.8, 0.8)
    f4.set_xlabel('Time (HJD)')
    f4.set_ylabel('Res.')


    # Mass posterior
    # masses = 10**tab['log_thetaE'] / (8.14 * 10**tab['log_piE'])
    masses = tab['thetaE'] / (8.14 * np.hypot(tab['piE_E'], tab['piE_N']))
    weights = tab['weights']
    
    f5 = plt.gcf().add_axes([fig_pos[3], 0.18, pan_wid, 0.8])
    bins = np.arange(0., 10, 0.1)
    f5.hist(masses, weights=weights, bins=bins, alpha = 0.9, log=log)
    f5.set_xlabel('Mass (M$_\odot$)')
    f5.set_ylabel('Probability')
    f5.set_xlim(0, mass_max_lim)

    plt.savefig(outfile)

    return

def plot_3panel(data, mod, tab, outfile, target, r_min_k=None, mass_max_lim=2, log=False):
    # Calculate the model on a similar timescale to the data.
    tmax = np.max(np.append(data['t_phot1'], data['t_phot2'])) + 90.0
    t_mod_ast = np.arange(data['t_ast1'].min() - 180.0, tmax, 2)
    t_mod_pho = np.arange(data['t_phot1'].min(), tmax, 2)

    # Get the linear motion curves for the source (includes parallax)
    p_unlens_mod = mod.get_astrometry_unlensed(t_mod_ast)
    p_unlens_mod_at_ast = mod.get_astrometry_unlensed(data['t_ast1'])

    # Get the lensed motion curves for the source
    p_lens_mod = mod.get_astrometry(t_mod_ast)
    p_lens_mod_at_ast = mod.get_astrometry(data['t_ast1'])

    # Geth the photometry
    m_lens_mod = mod.get_photometry(t_mod_pho, filt_idx=0)
    m_lens_mod_at_phot1 = mod.get_photometry(data['t_phot1'], filt_idx=0)
    m_lens_mod_at_phot2 = mod.get_photometry(data['t_phot2'], filt_idx=1)

    # Calculate the delta-mag between R-band and K-band from the
    # flat part at the end.
    tidx = np.argmin(np.abs(data['t_phot1'] - data['t_ast1'][-1]))
    if r_min_k == None:
        r_min_k = data['mag1'][tidx] - data['mag2'][-1]
    print('r_min_k = ', r_min_k)

    # Plotting        
    # plt.close(3)
    plt.figure(3, figsize=(14, 4))
    plt.clf()

    pan_wid = 0.22
    pan_pad = 0.09
    fig_pos = np.arange(0, 3) * (pan_wid + pan_pad) + 1.5*pan_pad

    plt.figtext(0.25*pan_pad, 0.55, target.upper(), rotation='vertical',
                    fontweight='bold', fontsize=20,
                    va='center', ha='center')

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
    f1.errorbar(data['t_ast1'], data['xpos']*1e3,
                    yerr=data['xpos_err']*1e3, fmt='k.', zorder = 1000)
    f1.plot(t_mod_ast, p_lens_mod[:, 0]*1e3, 'r-')
    f1.plot(t_mod_ast, p_unlens_mod[:, 0]*1e3, 'r--')
    f1.get_xaxis().set_visible(False)
    f1.set_ylabel(r'$\Delta \alpha^*$ (mas)')
    f1.get_shared_x_axes().join(f1, f2)
    
    f2.errorbar(data['t_ast1'], (data['xpos'] - p_unlens_mod_at_ast[:,0]) * 1e3,
                yerr=data['xpos_err'] * 1e3, fmt='k.', alpha=1, zorder = 1000)
    f2.plot(t_mod_ast, (p_lens_mod[:, 0] - p_unlens_mod[:, 0])*1e3, 'r-')
    f2.axhline(0, linestyle='--', color='r')
    f2.xaxis.set_major_locator(plt.MaxNLocator(3))
    f2.set_ylim(-0.8, 0.8)
    f2.set_xlabel('Time (HJD)')
    f2.set_ylabel('Res.')

    
    # Dec vs. time
    f3 = plt.gcf().add_axes([fig_pos[2], 0.36, pan_wid, 0.6])
    f4 = plt.gcf().add_axes([fig_pos[2], 0.18, pan_wid, 0.2])
    f3.errorbar(data['t_ast1'], data['ypos']*1e3,
                    yerr=data['ypos_err']*1e3, fmt='k.', zorder = 1000)
    f3.plot(t_mod_ast, p_lens_mod[:, 1]*1e3, 'r-')
    f3.plot(t_mod_ast, p_unlens_mod[:, 1]*1e3, 'r--')
    f3.set_ylabel(r'$\Delta \delta$ (mas)')
    f3.yaxis.set_major_locator(plt.MaxNLocator(4))
    f3.get_xaxis().set_visible(False)
    f3.get_shared_x_axes().join(f3, f4)
    
    f4.errorbar(data['t_ast1'], (data['ypos'] - p_unlens_mod_at_ast[:,1]) * 1e3,
                yerr=data['ypos_err'] * 1e3, fmt='k.', alpha=1, zorder = 1000)
    f4.plot(t_mod_ast, (p_lens_mod[:, 1] - p_unlens_mod[:, 1])*1e3, 'r-')
    f4.axhline(0, linestyle='--', color='r')
    f4.xaxis.set_major_locator(plt.MaxNLocator(3))
#    f4.set_yticks(np.array([0.0, -0.2])) # For OB140613
    f4.set_ylim(-0.8, 0.8)
    f4.set_xlabel('Time (HJD)')
    f4.set_ylabel('Res.')

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

def plot_pm_chi2_hist():
    targets = ['ob120169', 'ob140613', 'ob150029', 'ob150211']
    # targets = ['ob150029']

    rcut = {'ob120169': 1.3,
            'ob140613': 1.5,
            'ob150029': 1.5,
            'ob150211': 1.5}
    mcut = {'ob120169': 19.7,
            'ob140613': 18,
            'ob150029': 18,
            'ob150211': 17.2}

    plt.close('all')

    chi2red_hist = None
    chi2red_bins = None
    chi2red_targ = []

    for tt in range(len(targets)):
        target = targets[tt]
        mod_root = mod_roots[target]
        print('*** ' + target + ' ***')
    
        mod_yaml = open(mod_root +  'params.yaml').read() 
        params = yaml.safe_load(mod_yaml)

        # Load up the table
        stars_tab = Table.read(munge.data_sets[target]['Kp_Keck'])

        # Trim out stars with too few epochs
        good = np.where(stars_tab['n_vfit'] > 5)[0]
        stars_tab = stars_tab[good]
        

        tdx = np.where(stars_tab['name'] == target)[0][0]
        
        print('Sample Orig: ', len(stars_tab))
        stars_tab['r0'] = np.hypot(stars_tab['x0'] - stars_tab['x0'][tdx],
                                   stars_tab['y0'] - stars_tab['y0'][tdx])

        # Fetch the 5 closest stars + target within a magnitude cut.
        mdx = np.where(stars_tab['m0'] < mcut[target])[0]
        stars_tab = stars_tab[mdx]
        print('Sample Mag Cut: ', len(stars_tab))
        
        rdx = stars_tab['r0'].argsort()[0:6]
        print('Sample Rad Cut: ', len(rdx))


        hn, hb, tc = plot_chi2_dist(stars_tab[rdx], stars_tab['x'][rdx].shape[1], target)

        if tt == 0:
            chi2red_hist = hn
            chi2red_bins = hb
        else:
            chi2red_hist += hn
            
        chi2red_targ.append(tc)

    chi2red_hist /= chi2red_hist.sum()

    plt.figure(figsize=(6,6))
    plt.step(chi2red_bins[1:], chi2red_hist, label='Data', where='pre')

    colors = ['red', 'orange', 'black', 'purple']
    for tt in range(len(targets)):
        arr_y = 0.1
        arr_dy = -0.05
        plt.arrow(chi2red_targ[tt], arr_y, dx=0, dy=arr_dy, label=targets[tt].upper(),
                      color=colors[tt], width=0.04, head_width=0.08, head_length=0.02)
    
    plt.xlim(0, 5)
    plt.legend()
    plt.xlabel(r'$\tilde{\chi}^2$')
    plt.ylabel('PDF')



def plot_chi2_dist(tab, Ndetect, target, xlim=5, n_bins=15):
    """
    tab = flystar table
    Ndetect = Number of epochs star detected in
    """
    chi2_x_list = []
    chi2_y_list = []
    chi2_list = []
    fnd_list = [] # Number of non-NaN error measurements

    tdx = np.where(tab['name'] == target)[0][0]

    for ii in range(len(tab['xe'])):
        # Ignore the NaNs 
        fnd = np.argwhere(~np.isnan(tab['xe'][ii,:]))
        fnd_list.append(len(fnd))
        
        time = tab['t'][ii, fnd]
        x = tab['x'][ii, fnd]
        y = tab['y'][ii, fnd]
        xerr = tab['xe'][ii, fnd]
        yerr = tab['ye'][ii, fnd]

        dt = tab['t'][ii, fnd] - tab['t0'][ii]
        fitLineX = tab['x0'][ii] + (tab['vx'][ii] * dt)
        fitLineY = tab['y0'][ii] + (tab['vy'][ii] * dt)

        diffX = x - fitLineX
        diffY = y - fitLineY
        sigX = diffX / xerr
        sigY = diffY / yerr
        
        chi2_x = np.sum(sigX**2)
        chi2_y = np.sum(sigY**2)
        chi2_x_list.append(chi2_x)
        chi2_y_list.append(chi2_y)
        chi2_list.append(chi2_x + chi2_y)

        print('{0:15s} {1:4.1f} {2:4.1f} {3:4.1f} {4:2d}'.format(tab['name'][ii], chi2_x, chi2_y, chi2_x + chi2_y, fnd_list[ii]))

    x = np.array(chi2_x_list)
    y = np.array(chi2_y_list)
    t = np.array(chi2_list)
    fnd = np.array(fnd_list)

    chi2red_x = x / (fnd - 2)
    chi2red_y = y / (fnd - 2)
    chi2red_t = t / (2.0 * (fnd - 2))

    idx = np.where((fnd == Ndetect) & (tab['name'] != target))[0]
    tdx = np.where(tab['name'] == target)[0]
    
    # Fitting position and velocity... so subtract 2 to get Ndof
    Ndof = 2 * (Ndetect - 2)
    chi2_xaxis = np.linspace(0, xlim, n_bins*10)
    chi2_bins = np.linspace(0, xlim, n_bins)

    plt.figure(figsize=(6,4))
    plt.clf()
    hn, hb, hp = plt.hist(chi2red_t[idx], bins=chi2_bins, histtype='step', label='Data', density=True)
    plt.arrow(chi2red_t[tdx], hn.max()*0.5, dx=0, dy=hn.max()*0.1)

    chi2_mod = chi2.pdf(chi2_xaxis*Ndof, Ndof) * Ndof
    plt.plot(chi2_xaxis, chi2_mod, 'r-', alpha=0.6, 
             label='$\chi^2$ ' + str(Ndof) + ' dof')
    
    plt.title('$N_{epoch} = $' + str(Ndetect) + ', $N_{dof} = $' + str(Ndof))
    plt.xlim(0, xlim)
    plt.legend()
    plt.xlabel(r'$\tilde{\chi}^2$')
    plt.ylabel('PDF')

    print('Ndetect = {0:d}'.format(Ndetect))
    print('Mean reduced chi^2: (Ndetect = {0:d} of {1:d})'.format(len(idx), len(tab)))
    fmt = '   {0:s} = {1:.1f} for N_detect and {2:.1f} for all'
    med_chi2red_x_f = np.median(chi2red_x[idx])
    med_chi2red_x_a = np.median(chi2red_x)
    med_chi2red_y_f = np.median(chi2red_y[idx])
    med_chi2red_y_a = np.median(chi2red_y)
    med_chi2red_t_f = np.median(chi2red_t[idx])
    med_chi2red_t_a = np.median(chi2red_t)
    print(fmt.format('  X', med_chi2red_x_f, med_chi2red_x_a))
    print(fmt.format('  Y', med_chi2red_y_f, med_chi2red_y_a))
    print(fmt.format('Tot', med_chi2red_t_f, med_chi2red_t_a))

    print('')
    print('Target chi^2: ')
    fmt = '   {0:s} = {1:.1f} raw or {2:.1f} reduced for {3:d} DOF'
    print(fmt.format('Targ Tot', t[tdx[0]], chi2red_t[tdx[0]], 2*(fnd[tdx[0]] - 2)))

    return hn, hb, chi2red_t[tdx]


def plot_mb19284():
    astrom_both = '/g2/scratch/jlu/microlens/MB19284/mb19284_astrom_p4_2021_08_11_hst_keck.fits'

    foo = Table.read(astrom_both)

    # data = munge.getdata2('mb19284',
    #                       phot_data=['MOA'],
    #                       use_astrom_file=astrom_both,
    #                       use_astrom_phot=False)
    plt.figure(1)
    plt.clf()
    plt.errorbar(foo[0]['x'], foo[0]['y'], xerr=foo[0]['xe'], yerr=foo[0]['ye'])

    #pdb.set_trace()

    return
