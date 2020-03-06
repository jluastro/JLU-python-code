# Microlensing
import numpy as np
import pylab as plt
from astropy.table import Table, Column, vstack
from astropy.io import fits
from flystar import starlists
from scipy.interpolate import UnivariateSpline
from scipy.optimize import curve_fit, least_squares
import matplotlib.ticker
import matplotlib.colors
from matplotlib.pylab import cm
from matplotlib.colors import Normalize, LogNorm
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable
from mpl_toolkits.axes_grid1.colorbar import colorbar
from matplotlib.ticker import NullFormatter
import os
from scipy.ndimage import gaussian_filter as norm_kde
from microlens.jlu import model_fitter, multinest_utils, multinest_plot, munge_ob150211, munge_ob150029, model, munge
import dynesty.utils as dyutil
from dynesty import plotting as dyplot
from matplotlib.colors import LinearSegmentedColormap, colorConverter
import pdb
import pickle
import math
from scipy.stats import norm
import glob
from collections import Counter

ob120169_dir = '/u/jlu/work/microlens/OB120169/a_2019_06_26/model_fits/103_fit_phot_parallax_aerr/base_c/'
ob140613_dir = '/u/jlu/work/microlens/OB140613/a_2019_06_26/model_fits/101_fit_phot_parallax_merr/base_c/'
ob150029_dir = '/u/jlu/work/microlens/OB150029/a_2019_06_26/model_fits/103_fit_phot_parallax_aerr/base_d/'
ob150211_dir = '/u/jlu/work/microlens/OB150211/a_2019_06_26/model_fits/103_fit_phot_parallax_aerr/base_a/'

mb09260_dir = '/u/jlu/work/microlens/MB09260/a_2020_02_19/model_fits/102_fit_phot_parallax/a0_'
mb10364_dir = '/u/jlu/work/microlens/MB10364/a_2020_02_18/model_fits/102_fit_phot_parallax/b0_'
ob110462_dir = '/u/jlu/work/microlens/OB110462/a_2020_01_22/model_fits/102_fit_phot_parallax/a0_'
ob110310_dir = '/u/jlu/work/microlens/OB110310/a_2020_01_22/model_fits/102_fit_phot_parallax/a0_'
ob110037_dir = '/u/jlu/work/microlens/OB110037/a_2020_01_22/model_fits/102_fit_phot_parallax/a0_'

data_dir = '/g/lu/data/microlens/hst/'
prop_dir = '/u/casey/scratch/code/JLU-python-code/jlu/papers/'

MB09260_ep = ['2009_10_01', '2009_10_19', '2010_03_22', 
              '2010_06_14', '2010_10_20', '2011_04_19', 
              '2011_10_24', '2012_09_25', '2013_06_17']

MB10364_ep = ['2010_09_13', '2010_10_26', '2011_04_13',
              '2011_07_22', '2011_10_31', '2012_09_25',
              '2013_10_24']

OB110037_ep = ['2011_08_15', '2011_09_26', '2011_11_01',
               '2012_05_07', '2012_09_25', '2013_10_21',
               '2014_10_26', '2017_03_13', '2017_09_04']

OB110310_ep = ['2011_09_21', '2011_10_31', '2012_04_24',
               '2012_09_24', '2013_10_21', '2017_03_14', 
               '2017_09_01']

OB110462_ep = ['2011_08_08', '2011_10_31', '2012_09_09', 
               '2012_09_25', '2013_05_13', '2013_10_22',
               '2014_10_26', '2017_08_11', '2017_08_29']

ra = {'MB09260' : '17:58:27.226',
      'MB10364' : '17:57:06.854',
      'OB110037' : '17:55:56.672',
      'OB110310' : '17:51:26.450',
      'OB110462' : '17:51:40.190'}

dec = {'MB09260' : '-26:50:36.98',
       'MB10364' : '-34:26:49.11',
       'OB110037' : '-30:33:33.63',
       'OB110310' : '-30:24:33.90',
       'OB110462' : '-29:53:26.30'}

def piE_tE_phot_only_fits():
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

    # MB09260 fit results.
    data_09260 = munge.getdata2('mb09260',
                                phot_data=['MOA'],
                                ast_data = [])

    fitter_09260 = model_fitter.PSPL_Solver(data_09260,
                                            model.PSPL_Phot_Par_Param1,
                                            outputfiles_basename = mb09260_dir)

    results_09260 = fitter_09260.load_mnest_results_for_dynesty()
    smy_09260 = fitter_09260.load_mnest_summary()

    # MB10364 fit results.
    data_10364 = munge.getdata2('mb10364',
                                phot_data=['MOA'],
                                ast_data = [])

    fitter_10364 = model_fitter.PSPL_Solver(data_10364,
                                            model.PSPL_Phot_Par_Param1,
                                            outputfiles_basename = mb10364_dir)

    results_10364 = fitter_10364.load_mnest_results_for_dynesty()
    smy_10364 = fitter_10364.load_mnest_summary()

    # OB110462 fit results.
    data_110462 = munge.getdata2('ob110462',
                                 phot_data=['I_OGLE'],
                                 ast_data=[])  

    fitter_110462 = model_fitter.PSPL_Solver(data_110462,
                                             model.PSPL_Phot_Par_Param1,
                                             outputfiles_basename = ob110462_dir)

    results_110462 = fitter_110462.load_mnest_results_for_dynesty()
    smy_110462 = fitter_110462.load_mnest_summary()

    
    # OB110310 fit results.
    data_110310 = munge.getdata2('ob110310',
                                 phot_data=['I_OGLE'],
                                 ast_data=[])  

    fitter_110310 = model_fitter.PSPL_Solver(data_110310,
                                             model.PSPL_Phot_Par_Param1,
                                             outputfiles_basename = ob110310_dir)

    results_110310 = fitter_110310.load_mnest_results_for_dynesty()
    smy_110310 = fitter_110310.load_mnest_summary()

    # OB110037 fit results.
    data_110037 = munge.getdata2('ob110037',
                                 phot_data=['I_OGLE'],
                                 ast_data=[])  

    fitter_110037 = model_fitter.PSPL_Solver(data_110037,
                                             model.PSPL_Phot_Par_Param1,
                                             outputfiles_basename = ob110037_dir)

    results_110037 = fitter_110037.load_mnest_results_for_dynesty()
    smy_110037 = fitter_110037.load_mnest_summary()

    # OB120169 fit results.
    data_120169 = munge.getdata2('ob120169',
                                 phot_data=['I_OGLE'],
                                 ast_data=['Kp_Keck'])  

    fitter_120169 = model_fitter.PSPL_Solver(data_120169,
                                             model.PSPL_Phot_Par_Param1,
                                             add_error_on_photometry=True,
                                             outputfiles_basename=ob120169_dir + 'c3_')

    results_120169 = fitter_120169.load_mnest_results_for_dynesty()
    smy_120169 = fitter_120169.load_mnest_summary()

    
    # OB140613 fit results.
    data_140613 = munge.getdata2('ob140613',
                                 phot_data=['I_OGLE'],
                                 ast_data=['Kp_Keck'])  

    fitter_140613 = model_fitter.PSPL_Solver(data_140613,
                                             model.PSPL_Phot_Par_Param1,
                                             multiply_error_on_photometry=True,
                                             outputfiles_basename=ob140613_dir + 'c8_')
    results_140613 = fitter_140613.load_mnest_results_for_dynesty()
    smy_140613 = fitter_140613.load_mnest_summary()

    # OB150029 fit results.
    data_150029 = munge.getdata2('ob150029',
                                 phot_data=['I_OGLE'],
                                 ast_data=['Kp_Keck'])  

    fitter_150029 = model_fitter.PSPL_Solver(data_150029,
                                             model.PSPL_Phot_Par_Param1,
                                             add_error_on_photometry=True,
                                             outputfiles_basename=ob150029_dir + 'd8_')
    results_150029 = fitter_150029.load_mnest_results_for_dynesty()
    smy_150029 = fitter_150029.load_mnest_summary()

    # OB150211 fit results.
    data_150211 = munge.getdata2('ob150211',
                                 phot_data=['I_OGLE'],
                                 ast_data=['Kp_Keck'])  

    fitter_150211 = model_fitter.PSPL_Solver(data_150211,
                                             model.PSPL_Phot_Par_Param1,
                                             add_error_on_photometry=True,
                                             outputfiles_basename=ob150211_dir + 'a4_')
    results_150211 = fitter_150211.load_mnest_results_for_dynesty()
    smy_150211 = fitter_150211.load_mnest_summary()

    # Extract weighted samples.
    samples_09260 = results_09260['samples']
    samples_10364 = results_10364['samples']
    samples_110462 = results_110462['samples']
    samples_110310 = results_110310['samples']
    samples_110037 = results_110037['samples']
    samples_120169 = results_120169['samples']
    samples_140613 = results_140613['samples']
    samples_150029 = results_150029['samples']
    samples_150211 = results_150211['samples']

    try:
        weights_09260 = np.exp(results_09260['logwt'] - results_09260['logz'][-1])
        weights_10364 = np.exp(results_10364['logwt'] - results_10364['logz'][-1])
        weights_110462 = np.exp(results_110462['logwt'] - results_110462['logz'][-1])
        weights_110310 = np.exp(results_110310['logwt'] - results_110310['logz'][-1])
        weights_110037 = np.exp(results_110037['logwt'] - results_110037['logz'][-1])
        weights_120169 = np.exp(results_120169['logwt'] - results_120169['logz'][-1])
        weights_140613 = np.exp(results_140613['logwt'] - results_140613['logz'][-1])
        weights_150029 = np.exp(results_150029['logwt'] - results_150029['logz'][-1])
        weights_150211 = np.exp(results_150211['logwt'] - results_150211['logz'][-1])
    except:
        weights_09260 = results_09260['weights']
        weights_10364 = results_10364['weights']
        weights_110462 = results_110462['weights']
        weights_110310 = results_110310['weights']
        weights_110037 = results_110037['weights']
        weights_120169 = results_120169['weights']
        weights_140613 = results_140613['weights']
        weights_150029 = results_150029['weights']
        weights_150211 = results_150211['weights']

    # Deal with 1D results. A number of extra catches are also here
    # in case users are trying to plot other results besides the `Results`
    # instance generated by `dynesty`.
    samples_09260 = np.atleast_1d(samples_09260)
    if len(samples_09260.shape) == 1:
        samples_09260 = np.atleast_2d(samples_09260)
    else:
        assert len(samples_09260.shape) == 2, "Samples must be 1- or 2-D."
        samples_09260 = samples_09260.T
    assert samples_09260.shape[0] <= samples_09260.shape[1], "There are more " \
                                                 "dimensions than samples!"

    samples_10364 = np.atleast_1d(samples_10364)
    if len(samples_10364.shape) == 1:
        samples_10364 = np.atleast_2d(samples_10364)
    else:
        assert len(samples_10364.shape) == 2, "Samples must be 1- or 2-D."
        samples_10364 = samples_10364.T
    assert samples_10364.shape[0] <= samples_10364.shape[1], "There are more " \
                                                 "dimensions than samples!"

    samples_110462 = np.atleast_1d(samples_110462)
    if len(samples_110462.shape) == 1:
        samples_110462 = np.atleast_2d(samples_110462)
    else:
        assert len(samples_110462.shape) == 2, "Samples must be 1- or 2-D."
        samples_110462 = samples_110462.T
    assert samples_110462.shape[0] <= samples_110462.shape[1], "There are more " \
                                                 "dimensions than samples!"
    
    samples_110310 = np.atleast_1d(samples_110310)
    if len(samples_110310.shape) == 1:
        samples_110310 = np.atleast_2d(samples_110310)
    else:
        assert len(samples_110310.shape) == 2, "Samples must be 1- or 2-D."
        samples_110310 = samples_110310.T
    assert samples_110310.shape[0] <= samples_110310.shape[1], "There are more " \
                                                 "dimensions than samples!"

    samples_110037 = np.atleast_1d(samples_110037)
    if len(samples_110037.shape) == 1:
        samples_110037 = np.atleast_2d(samples_110037)
    else:
        assert len(samples_110037.shape) == 2, "Samples must be 1- or 2-D."
        samples_110037 = samples_110037.T
    assert samples_110037.shape[0] <= samples_110037.shape[1], "There are more " \
                                                 "dimensions than samples!"
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

    # Plot the piE-tE 2D posteriors.
    # tE = 2; piEE,N = 3, 4 
    fig, ax = plt.subplots(1, 2, figsize=(14, 6), sharey=True,
                           gridspec_kw={'width_ratios': [1.2, 2]})
    plt.subplots_adjust(left=0.1, bottom=0.15, wspace=0.1, right=1)

    tE_09260 = samples_09260[2]
    tE_10364 = samples_10364[2]
    tE_110462 = samples_110462[2]
    tE_110310 = samples_110310[2]
    tE_110037 = samples_110037[2]
    tE_120169 = samples_120169[2]
    tE_140613 = samples_140613[2]
    tE_150029 = samples_150029[2]
    tE_150211 = samples_150211[2]

    piE_09260 = np.hypot(samples_09260[3], samples_09260[4])
    piE_10364 = np.hypot(samples_10364[3], samples_10364[4])
    piE_110462 = np.hypot(samples_110462[3], samples_110462[4])
    piE_110310 = np.hypot(samples_110310[3], samples_110310[4])
    piE_110037 = np.hypot(samples_110037[3], samples_110037[4])
    piE_120169 = np.hypot(samples_120169[3], samples_120169[4])
    piE_140613 = np.hypot(samples_140613[3], samples_140613[4])
    piE_150029 = np.hypot(samples_150029[3], samples_150029[4])
    piE_150211 = np.hypot(samples_150211[3], samples_150211[4])

    sx = smooth
    sy = smooth

    hist2d_kwargs['fill_contours'] = hist2d_kwargs.get('fill_contours',
                                                       False)
    hist2d_kwargs['plot_contours'] = hist2d_kwargs.get('plot_contours',
                                                       True)
    model_fitter.contour2d_alpha(tE_09260, piE_09260, span=[span, span], quantiles_2d=quantiles_2d,
                                 weights=weights_09260, ax=ax[1], smooth=[sy, sx], color='blue',
                                 **hist2d_kwargs, plot_density=False)
    model_fitter.contour2d_alpha(tE_110310, piE_110310, span=[span, span], quantiles_2d=quantiles_2d,
                                 weights=weights_110310, ax=ax[1], smooth=[sy, sx], color='blue', 
                                 **hist2d_kwargs, plot_density=False)
    model_fitter.contour2d_alpha(tE_120169, piE_120169, span=[span, span], quantiles_2d=quantiles_2d,
                                 weights=weights_120169, ax=ax[1], smooth=[sy, sx], color='red',
                                 **hist2d_kwargs, plot_density=False)
    model_fitter.contour2d_alpha(tE_150211, piE_150211, span=[span, span], quantiles_2d=quantiles_2d,
                                 weights=weights_150211, ax=ax[1], smooth=[sy, sx], color='red', 
                                 **hist2d_kwargs, plot_density=False)

    # Maximum likelihood vals
    smy_list = [smy_09260, smy_10364, smy_110037, 
                smy_110310, smy_110462, smy_120169, 
                smy_140613, smy_150029, smy_150211]

    smy_name = ['MB09260', 'MB10364', 'OB110037', 
                'OB110310', 'OB110462', 'OB120169', 
                'OB140613', 'OB150029', 'OB150211']
    maxl = {}
    for ss, smy in enumerate(smy_list):
        print(smy_name[ss])
        print('t0 : ', smy['MaxLike_t0'][0])
        print('tE : ', smy['MaxLike_tE'][0])
        print('piE : ', np.hypot(smy['MaxLike_piE_E'][0], smy['MaxLike_piE_N'][0]))
        maxl[smy_name[ss]] = {'tE' : smy['MaxLike_tE'][0], 
                              'piE' : np.hypot(smy['MaxLike_piE_E'][0], smy['MaxLike_piE_N'][0])}

    plt.plot(maxl['MB10364']['tE'], maxl['MB10364']['piE'], color='blue', marker='s')
    plt.plot(maxl['OB110037']['tE'], maxl['OB110037']['piE'], color='blue', marker='s')
    plt.plot(maxl['OB110462']['tE'], maxl['OB110462']['piE'], color='blue', marker='s', label = 'HST')
    plt.plot(61.4, np.hypot(-0.393, -0.071), color = 'red', marker='^') # OB110022
    plt.plot(maxl['OB140613']['tE'], maxl['OB140613']['piE'], color='red', marker='^')
    plt.plot(maxl['OB150029']['tE'], maxl['OB150029']['piE'], color='red', marker='^', label = 'Keck')

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
    ax[1].set_xlim(10, 400)
    ax[1].set_ylim(0.009, 0.5)
    ax[1].set_xlabel('$t_E$ (days)')
    ax[1].set_xscale('log')
    ax[1].set_yscale('log')
    box = ax[1].get_position()
    ax[1].set_position([box.x0, box.y0, box.width * 0.7, box.height])
    ax[1].legend(bbox_to_anchor=(1.4, 0.5), loc="center right")
    plt.savefig('hst_sahu_piE_tE_phot_only_fit.png')


def piE_tE_phot_only_fits_old():
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

    # MB09260 fit results.
    data_09260 = munge.getdata2('mb09260',
                                phot_data=['MOA'],
                                ast_data = [])

    fitter_09260 = model_fitter.PSPL_Solver(data_09260,
                                            model.PSPL_Phot_Par_Param1,
                                            outputfiles_basename = mb09260_dir)

    results_09260 = fitter_09260.load_mnest_results_for_dynesty()
    smy_09260 = fitter_09260.load_mnest_summary()

    # MB10364 fit results.
    data_10364 = munge.getdata2('mb10364',
                                phot_data=['MOA'],
                                ast_data = [])

    fitter_10364 = model_fitter.PSPL_Solver(data_10364,
                                            model.PSPL_Phot_Par_Param1,
                                            outputfiles_basename = mb10364_dir)

    results_10364 = fitter_10364.load_mnest_results_for_dynesty()
    smy_10364 = fitter_10364.load_mnest_summary()

    # OB110462 fit results.
    data_110462 = munge.getdata2('ob110462',
                                 phot_data=['I_OGLE'],
                                 ast_data=[])  

    fitter_110462 = model_fitter.PSPL_Solver(data_110462,
                                             model.PSPL_Phot_Par_Param1,
                                             outputfiles_basename = ob110462_dir)

    results_110462 = fitter_110462.load_mnest_results_for_dynesty()
    smy_110462 = fitter_110462.load_mnest_summary()

    
    # OB110310 fit results.
    data_110310 = munge.getdata2('ob110310',
                                 phot_data=['I_OGLE'],
                                 ast_data=[])  

    fitter_110310 = model_fitter.PSPL_Solver(data_110310,
                                             model.PSPL_Phot_Par_Param1,
                                             outputfiles_basename = ob110310_dir)

    results_110310 = fitter_110310.load_mnest_results_for_dynesty()
    smy_110310 = fitter_110310.load_mnest_summary()

    # OB110037 fit results.
    data_110037 = munge.getdata2('ob110037',
                                 phot_data=['I_OGLE'],
                                 ast_data=[])  

    fitter_110037 = model_fitter.PSPL_Solver(data_110037,
                                             model.PSPL_Phot_Par_Param1,
                                             outputfiles_basename = ob110037_dir)

    results_110037 = fitter_110037.load_mnest_results_for_dynesty()
    smy_110037 = fitter_110037.load_mnest_summary()

    # OB120169 fit results.
    data_120169 = munge.getdata2('ob120169',
                                 phot_data=['I_OGLE'],
                                 ast_data=['Kp_Keck'])  

    fitter_120169 = model_fitter.PSPL_Solver(data_120169,
                                             model.PSPL_Phot_Par_Param1,
                                             add_error_on_photometry=True,
                                             outputfiles_basename=ob120169_dir + 'c3_')

    results_120169 = fitter_120169.load_mnest_results_for_dynesty()
    smy_120169 = fitter_120169.load_mnest_summary()

    
    # OB140613 fit results.
    data_140613 = munge.getdata2('ob140613',
                                 phot_data=['I_OGLE'],
                                 ast_data=['Kp_Keck'])  

    fitter_140613 = model_fitter.PSPL_Solver(data_140613,
                                             model.PSPL_Phot_Par_Param1,
                                             multiply_error_on_photometry=True,
                                             outputfiles_basename=ob140613_dir + 'c8_')
    results_140613 = fitter_140613.load_mnest_results_for_dynesty()
    smy_140613 = fitter_140613.load_mnest_summary()

    # OB150029 fit results.
    data_150029 = munge.getdata2('ob150029',
                                 phot_data=['I_OGLE'],
                                 ast_data=['Kp_Keck'])  

    fitter_150029 = model_fitter.PSPL_Solver(data_150029,
                                             model.PSPL_Phot_Par_Param1,
                                             add_error_on_photometry=True,
                                             outputfiles_basename=ob150029_dir + 'd8_')
    results_150029 = fitter_150029.load_mnest_results_for_dynesty()
    smy_150029 = fitter_150029.load_mnest_summary()

    # OB150211 fit results.
    data_150211 = munge.getdata2('ob150211',
                                 phot_data=['I_OGLE'],
                                 ast_data=['Kp_Keck'])  

    fitter_150211 = model_fitter.PSPL_Solver(data_150211,
                                             model.PSPL_Phot_Par_Param1,
                                             add_error_on_photometry=True,
                                             outputfiles_basename=ob150211_dir + 'a4_')
    results_150211 = fitter_150211.load_mnest_results_for_dynesty()
    smy_150211 = fitter_150211.load_mnest_summary()

    # Extract weighted samples.
    samples_09260 = results_09260['samples']
    samples_10364 = results_10364['samples']
    samples_110462 = results_110462['samples']
    samples_110310 = results_110310['samples']
    samples_110037 = results_110037['samples']
    samples_120169 = results_120169['samples']
    samples_140613 = results_140613['samples']
    samples_150029 = results_150029['samples']
    samples_150211 = results_150211['samples']

    try:
        weights_09260 = np.exp(results_09260['logwt'] - results_09260['logz'][-1])
        weights_10364 = np.exp(results_10364['logwt'] - results_10364['logz'][-1])
        weights_110462 = np.exp(results_110462['logwt'] - results_110462['logz'][-1])
        weights_110310 = np.exp(results_110310['logwt'] - results_110310['logz'][-1])
        weights_110037 = np.exp(results_110037['logwt'] - results_110037['logz'][-1])
        weights_120169 = np.exp(results_120169['logwt'] - results_120169['logz'][-1])
        weights_140613 = np.exp(results_140613['logwt'] - results_140613['logz'][-1])
        weights_150029 = np.exp(results_150029['logwt'] - results_150029['logz'][-1])
        weights_150211 = np.exp(results_150211['logwt'] - results_150211['logz'][-1])
    except:
        weights_09260 = results_09260['weights']
        weights_10364 = results_10364['weights']
        weights_110462 = results_110462['weights']
        weights_110310 = results_110310['weights']
        weights_110037 = results_110037['weights']
        weights_120169 = results_120169['weights']
        weights_140613 = results_140613['weights']
        weights_150029 = results_150029['weights']
        weights_150211 = results_150211['weights']

    # Deal with 1D results. A number of extra catches are also here
    # in case users are trying to plot other results besides the `Results`
    # instance generated by `dynesty`.
    samples_09260 = np.atleast_1d(samples_09260)
    if len(samples_09260.shape) == 1:
        samples_09260 = np.atleast_2d(samples_09260)
    else:
        assert len(samples_09260.shape) == 2, "Samples must be 1- or 2-D."
        samples_09260 = samples_09260.T
    assert samples_09260.shape[0] <= samples_09260.shape[1], "There are more " \
                                                 "dimensions than samples!"

    samples_10364 = np.atleast_1d(samples_10364)
    if len(samples_10364.shape) == 1:
        samples_10364 = np.atleast_2d(samples_10364)
    else:
        assert len(samples_10364.shape) == 2, "Samples must be 1- or 2-D."
        samples_10364 = samples_10364.T
    assert samples_10364.shape[0] <= samples_10364.shape[1], "There are more " \
                                                 "dimensions than samples!"

    samples_110462 = np.atleast_1d(samples_110462)
    if len(samples_110462.shape) == 1:
        samples_110462 = np.atleast_2d(samples_110462)
    else:
        assert len(samples_110462.shape) == 2, "Samples must be 1- or 2-D."
        samples_110462 = samples_110462.T
    assert samples_110462.shape[0] <= samples_110462.shape[1], "There are more " \
                                                 "dimensions than samples!"
    
    samples_110310 = np.atleast_1d(samples_110310)
    if len(samples_110310.shape) == 1:
        samples_110310 = np.atleast_2d(samples_110310)
    else:
        assert len(samples_110310.shape) == 2, "Samples must be 1- or 2-D."
        samples_110310 = samples_110310.T
    assert samples_110310.shape[0] <= samples_110310.shape[1], "There are more " \
                                                 "dimensions than samples!"

    samples_110037 = np.atleast_1d(samples_110037)
    if len(samples_110037.shape) == 1:
        samples_110037 = np.atleast_2d(samples_110037)
    else:
        assert len(samples_110037.shape) == 2, "Samples must be 1- or 2-D."
        samples_110037 = samples_110037.T
    assert samples_110037.shape[0] <= samples_110037.shape[1], "There are more " \
                                                 "dimensions than samples!"
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

    # Plot the piE-tE 2D posteriors.
    # tE = 2; piEE,N = 3, 4 
    fig, ax = plt.subplots(1, 2, figsize=(14,6), sharey=True,
                           gridspec_kw={'width_ratios': [1, 2]})
    plt.subplots_adjust(left=0.1, bottom=0.15, wspace=0.1)

    tE_09260 = samples_09260[2]
    tE_10364 = samples_10364[2]
    tE_110462 = samples_110462[2]
    tE_110310 = samples_110310[2]
    tE_110037 = samples_110037[2]
    tE_120169 = samples_120169[2]
    tE_140613 = samples_140613[2]
    tE_150029 = samples_150029[2]
    tE_150211 = samples_150211[2]

    piE_09260 = np.hypot(samples_09260[3], samples_09260[4])
    piE_10364 = np.hypot(samples_10364[3], samples_10364[4])
    piE_110462 = np.hypot(samples_110462[3], samples_110462[4])
    piE_110310 = np.hypot(samples_110310[3], samples_110310[4])
    piE_110037 = np.hypot(samples_110037[3], samples_110037[4])
    piE_120169 = np.hypot(samples_120169[3], samples_120169[4])
    piE_140613 = np.hypot(samples_140613[3], samples_140613[4])
    piE_150029 = np.hypot(samples_150029[3], samples_150029[4])
    piE_150211 = np.hypot(samples_150211[3], samples_150211[4])

    sx = smooth
    sy = smooth

    hist2d_kwargs['fill_contours'] = hist2d_kwargs.get('fill_contours',
                                                       False)
    hist2d_kwargs['plot_contours'] = hist2d_kwargs.get('plot_contours',
                                                       True)
    model_fitter.contour2d_alpha(tE_09260, piE_09260, span=[span, span], quantiles_2d=quantiles_2d,
                                 weights=weights_09260, ax=ax[1], smooth=[sy, sx], color='green',
                                 **hist2d_kwargs, plot_density=False)
    model_fitter.contour2d_alpha(tE_10364, piE_10364, span=[span, span], quantiles_2d=quantiles_2d,
                                 weights=weights_10364, ax=ax[1], smooth=[sy, sx], color='paleturquoise',
                                 **hist2d_kwargs, plot_density=False)
    model_fitter.contour2d_alpha(tE_110462, piE_110462, span=[span, span], quantiles_2d=quantiles_2d,
                                 weights=weights_110462, ax=ax[1], smooth=[sy, sx], color='purple',
                                 **hist2d_kwargs, plot_density=False)
    model_fitter.contour2d_alpha(tE_110310, piE_110310, span=[span, span], quantiles_2d=quantiles_2d,
                                 weights=weights_110310, ax=ax[1], smooth=[sy, sx], color='red', 
                                 **hist2d_kwargs, plot_density=False)
    model_fitter.contour2d_alpha(tE_110037, piE_110037, span=[span, span], quantiles_2d=quantiles_2d,
                                 weights=weights_110037, ax=ax[1], smooth=[sy, sx], color='blue',
                                 **hist2d_kwargs, plot_density=False)
    model_fitter.contour2d_alpha(tE_120169, piE_120169, span=[span, span], quantiles_2d=quantiles_2d,
                                 weights=weights_120169, ax=ax[1], smooth=[sy, sx], color='darkorange',
                                 **hist2d_kwargs, plot_density=False)
    model_fitter.contour2d_alpha(tE_140613, piE_140613, span=[span, span], quantiles_2d=quantiles_2d,
                                 weights=weights_140613, ax=ax[1], smooth=[sy, sx], color='aqua', 
                                 **hist2d_kwargs, plot_density=False)
    model_fitter.contour2d_alpha(tE_150029, piE_150029, span=[span, span], quantiles_2d=quantiles_2d,
                                 weights=weights_150029, ax=ax[1], smooth=[sy, sx], color='lawngreen',
                                 **hist2d_kwargs, plot_density=False)
    model_fitter.contour2d_alpha(tE_150211, piE_150211, span=[span, span], quantiles_2d=quantiles_2d,
                                 weights=weights_150211, ax=ax[1], smooth=[sy, sx], color='deeppink', 
                                 **hist2d_kwargs, plot_density=False)

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
    ax[1].arrow(62.33 - 30, 0.258, 20, 0, 
                width = 0.01, head_width = 0.05, head_length = 5, 
                color = 'paleturquoise')# MB10364
    ax[1].arrow(77.04 - 30, 0.368, 20, 0, 
                width = 0.01, head_width = 0.05, head_length = 5, 
                color = 'blue') # OB110037
    ax[1].scatter(0.01, 100, marker = '_', color='green', label='MB09260')    
    ax[1].scatter(0.01, 100, marker = '_', color='paleturquoise', label='MB10364')    
    ax[1].scatter(0.01, 100, marker = '_', color='purple', label='OB110462')    
    ax[1].scatter(0.01, 100, marker = '_', color='red', label='OB110310')
    ax[1].scatter(0.01, 100, marker = '_', color='blue', label='OB110037')
    ax[1].scatter(0.01, 100, marker = '_', color='darkorange', label='OB120169')    
    ax[1].scatter(0.01, 100, marker = '_', color='aqua', label='OB140613')    
    ax[1].scatter(0.01, 100, marker = '_', color='lawngreen', label='OB150029')    
    ax[1].scatter(0.01, 100, marker = '_', color='deeppink', label='OB150211')    
    ax[1].annotate("", xy=(0.5, 0.5), xytext=(0, 0), arrowprops=dict(arrowstyle="->"))
    ax[0].set_xlabel('$\delta_{c,max}$ (mas)')
    ax[0].set_ylabel('$\pi_E$')
    ax[0].set_xscale('log')
    ax[0].set_yscale('log')
    ax[1].set_xlim(10, 400)
    ax[1].set_ylim(0.009, 0.5)
    ax[1].set_xlabel('$t_E$ (days)')
    ax[1].set_xscale('log')
    ax[1].set_yscale('log')
    box = ax[1].get_position()
    ax[1].set_position([box.x0, box.y0, box.width * 0.7, box.height])
    ax[1].legend(bbox_to_anchor=(1.3, 0.5), loc="center right")
    plt.savefig('hst_sahu_piE_tE_phot_only_fit.png')

    smy_list = [smy_09260, smy_10364, smy_110037, 
                smy_110310, smy_110462, smy_120169, 
                smy_140613, smy_150029, smy_150211]

    smy_name = ['MB09260', 'MB10364', 'OB110037', 
                'OB110310', 'OB110462', 'OB120169', 
                'OB140613', 'OB150029', 'OB150211']
    
    for ss, smy in enumerate(smy_list):
        print(smy_name[ss])
        print('t0 : ', smy['MaxLike_t0'][0])
        print('tE : ', smy['MaxLike_tE'][0])
        print('piE : ', np.hypot(smy['MaxLike_piE_E'][0], smy['MaxLike_piE_N'][0]))

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
    
    NBH = 10

    N_detect = np.linspace(0, NBH, 1000)
    N_sigma = np.sqrt(N_detect)
    
    fig = plt.figure(figsize=(6,5))
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
    plt.savefig('nbh_sigma.png')
    plt.show()

    return

def plot_lightcurves():
    """
    Plot the lightcurves for MB09260 and MB10364.
    """
    mb09260_data = munge.getdata2('mb09260', phot_data=['MOA', 'HST'], ast_data = [])  
    mb10364_data = munge.getdata2('mb10364', phot_data=['MOA', 'HST'], ast_data = [])  

    mb09260_dir = '/u/jlu/work/microlens/MB09260/a_2020_02_19/model_fits/202_fit_multiphot_parallax/a0_'
    mb10364_dir = '/u/jlu/work/microlens/MB10364/a_2020_02_18/model_fits/202_fit_multiphot_parallax/a0_'
     
    mb09260_fitter = model_fitter.PSPL_Solver(mb09260_data,
                                              model.PSPL_Phot_Par_Param1,
                                              importance_nested_sampling = False,
                                              n_live_points = 400,
                                              evidence_tolerance = 0.5,
                                              sampling_efficiency = 0.8,
                                              outputfiles_basename=mb09260_dir)

    mb10364_fitter = model_fitter.PSPL_Solver(mb10364_data,
                                              model.PSPL_Phot_Par_Param1,
                                              importance_nested_sampling = False,
                                              n_live_points = 400,
                                              evidence_tolerance = 0.5,
                                              sampling_efficiency = 0.8,
                                              outputfiles_basename=mb10364_dir)

    mb09260_mod = mb09260_fitter.get_best_fit_model(def_best='maxl')
    mb10364_mod = mb10364_fitter.get_best_fit_model(def_best='maxl')
    
    times_mod = np.arange(54000, 57000, 1)

    fig = plt.figure(1, figsize=(14,6))
    plt.clf()
    plt.subplots_adjust(left=0.07, right=0.99, wspace=0.15)
    ax1 = plt.subplot(1, 2, 1)
    ax2 = plt.subplot(1, 2, 2)

    mb09260_mod_mag1 = mb09260_mod.get_photometry(times_mod, 0)
    mb09260_mod_mag2 = mb09260_mod.get_photometry(times_mod, 1)
    mb10364_mod_mag1 = mb10364_mod.get_photometry(times_mod, 0)
    mb10364_mod_mag2 = mb10364_mod.get_photometry(times_mod, 1)

    # Rescaled MOA to match what's on Artemis
    mb09260_mag1_rs = mb09260_data['mag1'] + 3.25
    mb10364_mag1_rs = mb10364_data['mag1'] + 1.75
    mb09260_mag2_rs = mb09260_data['mag2'] - 1.05087
    mb10364_mag2_rs = mb10364_data['mag2'] - 1.8983

    mb09260_mod_mag1 += 3.25
    mb10364_mod_mag1 += 1.75
    mb09260_mod_mag2 += -1.0587
    mb10364_mod_mag2 += -1.8983

    ax1.errorbar(mb09260_data['t_phot1'], mb09260_mag1_rs, 
                 ls='none', yerr=mb09260_data['mag_err1'],
                 color='tab:blue', alpha = 0.3)
    ax1.errorbar(mb09260_data['t_phot2'], mb09260_mag2_rs, 
                 ls='none', yerr=mb09260_data['mag_err2'],
                 color='tab:orange', alpha = 0.9)
    ax1.plot(times_mod, mb09260_mod_mag1, '-', color = 'tab:blue') 
    ax1.plot(mb09260_data['t_phot1'], mb09260_mag1_rs, '.',
             color='tab:blue', alpha = 0.3)
    ax1.plot(1, 1, '.', color='tab:blue', alpha = 0.9, label='MOA I')
    ax1.plot(mb09260_data['t_phot2'], mb09260_mag2_rs, 'o',
             color='tab:orange', alpha = 0.9,
             label='HST F814W - 1.06')
    ax1.plot(times_mod, mb09260_mod_mag2, '--', color = 'tab:orange')
    ax1.set_ylabel('Magnitude')
    ax1.set_xlabel('Time (MJD)')
    ax1.set_title('MB09260')
    ax1.invert_yaxis()
    ax1.set_xlim(54447, 56647)
    ax1.xaxis.set_major_locator(plt.MultipleLocator(500))
    ax1.yaxis.set_major_locator(plt.MultipleLocator(1))
    ax1.set_ylim(18, 14)
    ax1.legend()

    ax2.errorbar(mb10364_data['t_phot1'], mb10364_mag1_rs, 
                 ls='none', yerr=mb10364_data['mag_err1'],
                 color='tab:blue', alpha = 0.3)
    ax2.errorbar(mb10364_data['t_phot2'], mb10364_mag2_rs, 
                 ls='none', yerr=mb10364_data['mag_err2'],
                 color='tab:orange', alpha = 0.9)
    ax2.plot(times_mod, mb10364_mod_mag1, '-', color = 'tab:blue') 
    ax2.plot(mb10364_data['t_phot1'], mb10364_mag1_rs, '.',
             color='tab:blue', alpha = 0.3)
    ax2.plot(1, 1, '.', color='tab:blue', alpha = 0.9, label='MOA I')
    ax2.plot(mb10364_data['t_phot2'], mb10364_mag2_rs, 'o',
             color='tab:orange', alpha = 0.9,
             label='HST F606W - 1.90')
    ax2.plot(times_mod, mb10364_mod_mag2, '--', color = 'tab:orange')
    ax2.set_title('MB10364')
    ax2.set_xlabel('Time (MJD)')
    ax2.invert_yaxis()
    ax2.set_xlim(54817, 56653)
    ax2.xaxis.set_major_locator(plt.MultipleLocator(500))
    ax2.yaxis.set_major_locator(plt.MultipleLocator(1))
    ax2.set_ylim(15, 11.4)
    ax2.legend()
    plt.savefig('moa_hst_photometry.png')
