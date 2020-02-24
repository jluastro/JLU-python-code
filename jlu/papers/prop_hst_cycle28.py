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

    # Extract weighted samples.
    samples_09260 = results_09260['samples']
    samples_10364 = results_10364['samples']
    samples_110462 = results_110462['samples']
    samples_110310 = results_110310['samples']
    samples_110037 = results_110037['samples']

    try:
        weights_09260 = np.exp(results_09260['logwt'] - results_09260['logz'][-1])
        weights_10364 = np.exp(results_10364['logwt'] - results_10364['logz'][-1])
        weights_110462 = np.exp(results_110462['logwt'] - results_110462['logz'][-1])
        weights_110310 = np.exp(results_110310['logwt'] - results_110310['logz'][-1])
        weights_110037 = np.exp(results_110037['logwt'] - results_110037['logz'][-1])
    except:
        weights_09260 = results_09260['weights']
        weights_10364 = results_10364['weights']
        weights_110462 = results_110462['weights']
        weights_110310 = results_110310['weights']
        weights_110037 = results_110037['weights']

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

    # Plot the piE-tE 2D posteriors.
    # tE = 2; piEE,N = 3, 4 
    fig, axes = plt.subplots(1, 1, figsize=(6,6))
    plt.subplots_adjust(bottom=0.15)

    tE_09260 = samples_09260[2]
    tE_10364 = samples_10364[2]
    tE_110462 = samples_110462[2]
    tE_110310 = samples_110310[2]
    tE_110037 = samples_110037[2]
    piE_09260 = np.hypot(samples_09260[3], samples_09260[4])
    piE_10364 = np.hypot(samples_10364[3], samples_10364[4])
    piE_110462 = np.hypot(samples_110462[3], samples_110462[4])
    piE_110310 = np.hypot(samples_110310[3], samples_110310[4])
    piE_110037 = np.hypot(samples_110037[3], samples_110037[4])

    sx = smooth
    sy = smooth

    hist2d_kwargs['fill_contours'] = hist2d_kwargs.get('fill_contours',
                                                       False)
    hist2d_kwargs['plot_contours'] = hist2d_kwargs.get('plot_contours',
                                                       True)
    model_fitter.contour2d_alpha(tE_09260, piE_09260, span=[span, span], quantiles_2d=quantiles_2d,
                                 weights=weights_09260, ax=axes, smooth=[sy, sx], color='green',
                                 **hist2d_kwargs, plot_density=False)
    model_fitter.contour2d_alpha(tE_10364, piE_10364, span=[span, span], quantiles_2d=quantiles_2d,
                                 weights=weights_10364, ax=axes, smooth=[sy, sx], color='goldenrod',
                                 **hist2d_kwargs, plot_density=False)
    model_fitter.contour2d_alpha(tE_110462, piE_110462, span=[span, span], quantiles_2d=quantiles_2d,
                                 weights=weights_110462, ax=axes, smooth=[sy, sx], color='purple',
                                 **hist2d_kwargs, plot_density=False)
    model_fitter.contour2d_alpha(tE_110310, piE_110310, span=[span, span], quantiles_2d=quantiles_2d,
                                 weights=weights_110310, ax=axes, smooth=[sy, sx], color='red', 
                                 **hist2d_kwargs, plot_density=False)
    model_fitter.contour2d_alpha(tE_110037, piE_110037, span=[span, span], quantiles_2d=quantiles_2d,
                                 weights=weights_110037, ax=axes, smooth=[sy, sx], color='darkorange',
                                 **hist2d_kwargs, plot_density=False)
    axes.text(50, 0.1, 'MB09260', color='green')    
    axes.text(22, 0.25, 'MB10364', color='goldenrod')    
    axes.text(150, 0.21, 'OB110462', color='purple')    
    axes.text(110, 0.4, 'OB110310', color='red')
    axes.text(25, 0.35, 'OB110037', color='darkorange')

    # Add the PopSyCLE simulation points.
    # NEED TO UPDATE THIS WITH BUGFIX IN DELTAM
    t = Table.read('/u/casey/scratch/papers/microlens_2019/popsycle_rr_files/Mock_EWS_v2.fits') 

    bh_idx = np.where(t['rem_id_L'] == 103)[0]
    ns_idx = np.where(t['rem_id_L'] == 102)[0]
    wd_idx = np.where(t['rem_id_L'] == 101)[0]
    st_idx = np.where(t['rem_id_L'] == 0)[0]

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

    axes.set_xlim(10, 500)
    axes.set_ylim(0.005, 0.5)
    axes.set_xlabel('$t_E$ (days)')
    axes.set_ylabel('$\pi_E$')
    axes.set_xscale('log')
    axes.set_yscale('log')
    axes.legend(loc=3)
    plt.savefig('hst_sahu_piE_tE_phot_only_fit.png')
