import numpy as np
import pylab as plt
import pdb
import math
import os
# from jlu.observe import skycalc
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

mpl_o = '#ff7f0e'
mpl_b = '#1f77b4'
mpl_g = '#2ca02c'
mpl_r = '#d62728'

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

def tE_piE_deltac():
    """
    Make PopSyCLE plots.
    """
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
                                                                
    ##########
    # OB120169
    ##########
    # Phot + astrom
    piEE_120169 = 0
    piEE_120169_pe = 0.015
    piEE_120169_me = 0.014
    piEN_120169 = -0.154
    piEN_120169_pe = 0.019
    piEN_120169_me = 0.023
    piE_120169, piE_120169_pe, piE_120169_me = calc_hypot_and_err(piEE_120169, piEE_120169_pe, piEE_120169_me,
                                                                  piEN_120169, piEN_120169_pe, piEN_120169_me)

    tE_120169 = 175.363
    tE_120169_pe = 10.981
    tE_120169_me = 17.388

    # This is based on logthetaE. I don't know how to rebin...
    logthetaE_120169 = -1.465
    logthetaE_120169_pe = 1.043
    logthetaE_120169_me = 1.123

    dcmax_120169 = 10**logthetaE_120169/np.sqrt(8)
    dcmax_120169_pe = 10**(logthetaE_120169 + logthetaE_120169_pe)/np.sqrt(8) - dcmax_120169
    dcmax_120169_me = dcmax_120169 - 10**(logthetaE_120169 - logthetaE_120169_me)/np.sqrt(8)


    ##########
    # OB140613
    ##########
    # Phot + astrom
    piEE_140613 = -0.118
    piEE_140613_pe = 0.001
    piEE_140613_me = 0.001
    piEN_140613 = 0.078
    piEN_140613_pe = 0.001
    piEN_140613_me = 0.001
    piE_140613, piE_140613_pe, piE_140613_me = calc_hypot_and_err(piEE_140613, piEE_140613_pe, piEE_140613_me,
                                                                  piEN_140613, piEN_140613_pe, piEN_140613_me)
    
    tE_140613 = 304.957
    tE_140613_pe = 1.639
    tE_140613_me = 1.618

    # This is based on logthetaE. I don't know how to rebin...
    logthetaE_140613 = -0.136
    logthetaE_140613_pe = 0.127
    logthetaE_140613_me = 0.098

    dcmax_140613 = 10**logthetaE_140613/np.sqrt(8)
    dcmax_140613_pe = 10**(logthetaE_140613 + logthetaE_140613_pe)/np.sqrt(8) - dcmax_140613
    dcmax_140613_me = dcmax_140613 - 10**(logthetaE_140613 - logthetaE_140613_me)/np.sqrt(8)

    ##########
    # OB150029
    ##########
    # Phot + astrom
    piEE_150029 = 0.066
    piEE_150029_pe = 0.001
    piEE_150029_me = 0.001
    piEN_150029 = 0.163
    piEN_150029_pe = 0.001
    piEN_150029_me = 0.001
    piE_150029, piE_150029_pe, piE_150029_me = calc_hypot_and_err(piEE_150029, piEE_150029_pe, piEE_150029_me,
                                                                  piEN_150029, piEN_150029_pe, piEN_150029_me)

    tE_150029 = 138.959
    tE_150029_pe = 0.816
    tE_150029_me = 0.815

    # This is based on logthetaE. I don't know how to rebin...
    logthetaE_150029 = -1.001
    logthetaE_150029_pe = 1.354
    logthetaE_150029_me = 0.735

    dcmax_150029 = 10**logthetaE_150029/np.sqrt(8)
    dcmax_150029_pe = 10**(logthetaE_150029 + logthetaE_150029_pe)/np.sqrt(8) - dcmax_150029
    dcmax_150029_me = dcmax_150029 - 10**(logthetaE_150029 - logthetaE_150029_me)/np.sqrt(8)


    ##########
    # OB150211
    ##########
    # Phot + astrom
    # This is the dd_ run
    piEE_150211 = 0.010
    piEE_150211_pe = 0.012
    piEE_150211_me = 0.011
    piEN_150211 = 0.005
    piEN_150211_pe = 0.021
    piEN_150211_me = 0.021
    piE_150211, piE_150211_pe, piE_150211_me = calc_hypot_and_err(piEE_150211, piEE_150211_pe, piEE_150211_me,
                                                                  piEN_150211, piEN_150211_pe, piEN_150211_me)

    tE_150211 = 123.002
    tE_150211_pe = 2.781
    tE_150211_me = 4.032

    logthetaE_150211 = -1.264
    logthetaE_150211_pe = 1.152
    logthetaE_150211_me = 0.953

    dcmax_150211 = 10**logthetaE_150211/np.sqrt(8)
    dcmax_150211_pe = 10**(logthetaE_150211 + logthetaE_150211_pe)/np.sqrt(8) - dcmax_150211
    dcmax_150211_me = dcmax_150211 - 10**(logthetaE_150211 - logthetaE_150211_me)/np.sqrt(8)

    print(dcmax_150211)
    print(dcmax_150211_pe)
    print(dcmax_150211_me)

    ##########
    # OB170019
    ##########
    # EWS phot only
    piEE_170019 = -0.008
    piEE_170019_pe = 0.002
    piEE_170019_me = 0.002
    piEN_170019 = -0.056
    piEN_170019_pe = 0.003
    piEN_170019_me = 0.002

    piE_170019, piE_170019_pe, piE_170019_me = calc_hypot_and_err(piEE_170019, piEE_170019_pe, piEE_170019_me,
                                                                  piEN_170019, piEN_170019_pe, piEN_170019_me)

    piE_170019 = np.hypot(piEE_170019, piEN_170019)

    tE_170019 = 129.038
    tE_170019_pe = 0.268
    tE_170019_me = 0.268

    ##########
    # OB170095
    ##########
    # EWS phot only
    piEE_170095 = -0.035
    piEE_170095_pe = 0.007
    piEE_170095_me = 0.008
    piEN_170095 = 0.024
    piEN_170095_pe = 0.018
    piEN_170095_me = 0.030

    piE_170095, piE_170095_pe, piE_170095_me = calc_hypot_and_err(piEE_170095, piEE_170095_pe, piEE_170095_me,
                                                                  piEN_170095, piEN_170095_pe, piEN_170095_me)

    tE_170095 = 105.582 
    tE_170095_pe = 1.542
    tE_170095_me = 1.614 

    ##########
    # OB190017
    ##########
    # EWS phot only
    piEE_190017 = -0.123
    piEE_190017_pe = 0.002
    piEE_190017_me = 0.002
    piEN_190017 = -0.131
    piEN_190017_pe = 0.003
    piEN_190017_me = 0.003

    piE_190017, piE_190017_pe, piE_190017_me = calc_hypot_and_err(piEE_190017, piEE_190017_pe, piEE_190017_me,
                                                                  piEN_190017, piEN_190017_pe, piEN_190017_me)

    tE_190017 = 258.70 
    tE_190017_pe = 4.028
    tE_190017_me = 4.081

    ##########
    # OB190033
    ##########
    # EWS phot only
    piEE_190033 = -0.060
    piEE_190033_pe = 0.007
    piEE_190033_me = 0.007
    piEN_190033 = 0.439
    piEN_190033_pe = 0.004
    piEN_190033_me = 0.004

    tE_190033 = 135.532
    tE_190033_pe = 0.789 
    tE_190033_me = 0.772

    piE_190033, piE_190033_pe, piE_190033_me = calc_hypot_and_err(piEE_190033, piEE_190033_pe, piEE_190033_me,
                                                                  piEN_190033, piEN_190033_pe, piEN_190033_me)

    #################
    # The simulation.
    #################
    t = Table.read('/u/casey/scratch/papers/microlens_2019/popsycle_rr_files/Mock_EWS_v2.fits')

    bh_idx = np.where(t['rem_id_L'] == 103)[0]
    ns_idx = np.where(t['rem_id_L'] == 102)[0]
    wd_idx = np.where((t['rem_id_L'] == 101) | 
                      (t['rem_id_L'] == 6))[0]
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

    plt.figure(10, figsize=(14, 6))
    plt.clf()
    plt.subplots_adjust(left = 0.25, bottom = 0.15, wspace = 0.3)
    ax1 = plt.subplot(1, 2, 1) 
    ax2 = plt.subplot(1, 2, 2)

    mindc = 0.005
    maxdc = 5
    minpiE = 0.003
    maxpiE = 5
    mintE = 0.8
    maxtE = 500

    # For labeling purposes, to make it darker in the legend.
    ax1.scatter(10**-3, 10**-3,
                alpha = 0.5, label = 'Star', marker = 's',  
                c = 'gold')
    ax1.scatter(10**-3, 10**-3,
                alpha = 0.5, label = 'WD', marker = 'P', 
                c = 'coral')
    ax1.scatter(10**-3, 10**-3,
                alpha = 0.5, label = 'NS', marker = 'v', 
                c = 'limegreen')
    ax1.scatter(10**-3, 10**-3,
                alpha = 0.5, label = 'BH', 
                c = 'k')
    
    ax1.scatter(t['t_E'][st_idx], t['pi_E'][st_idx],
                alpha = 0.2, marker = 's', c = 'gold', label = '')
    ax1.scatter(t['t_E'][wd_idx], t['pi_E'][wd_idx],
                alpha = 0.2, marker = 'P', c = 'coral', label = '')
    ax1.scatter(t['t_E'][ns_idx], t['pi_E'][ns_idx], 
                alpha = 0.2, marker = 'v', c = 'limegreen', label = '')
    ax1.scatter(t['t_E'][bh_idx], t['pi_E'][bh_idx], 
                alpha = 0.2, c = 'k', label = '')

    ax1.errorbar(tE_110022, piE_110022, 
                 xerr = np.array([[tE_110022_me], [tE_110022_pe]]), 
                 yerr = np.array([[piE_110022_me], [piE_110022_pe]]),
                 capsize = 5, fmt = 's', color = 'cyan', markersize = 12,
                 label = 'OB110022')
    ax1.errorbar(tE_120169, piE_120169, 
                 xerr = np.array([[tE_120169_me], [tE_120169_pe]]), 
                 yerr = np.array([[piE_120169_me], [piE_120169_pe]]),
                 capsize = 5, fmt = 's', color = 'dodgerblue', markersize = 12,
                 label = 'OB120169')
    ax1.errorbar(tE_140613, piE_140613, 
                 xerr = np.array([[tE_140613_me], [tE_140613_pe]]), 
                 yerr = np.array([[piE_140613_me], [piE_140613_pe]]),
                 capsize = 5, fmt = 's', color = 'navy', markersize = 12,
                 label = 'OB140613')
    ax1.errorbar(tE_150029, piE_150029, 
                 xerr = np.array([[tE_150029_me], [tE_150029_pe]]), 
                 yerr = np.array([[piE_150029_me], [piE_150029_pe]]),
                 capsize = 5, fmt = 's', color = 'blueviolet', markersize = 12,
                 label = 'OB150029')
    ax1.errorbar(tE_150211, piE_150211, 
                 xerr = np.array([[tE_150211_me], [tE_150211_pe]]), 
                 yerr = np.array([[piE_150211_me], [piE_150211_pe]]),
                 capsize = 5, fmt = 's', color = 'purple', markersize = 12,
                 label = 'OB150211')
    ax1.errorbar(tE_170019, piE_170019, 
                 xerr = np.array([[tE_170019_me], [tE_170019_pe]]), 
                 yerr = np.array([[piE_170019_me], [piE_170019_pe]]),
                 capsize = 5, fmt = 's', color = 'deeppink', markersize = 12,
                 label = 'OB170019')
    ax1.errorbar(tE_170095, piE_170095, 
                 xerr = np.array([[tE_170095_me], [tE_170095_pe]]), 
                 yerr = np.array([[piE_170095_me], [piE_170095_pe]]),
                 capsize = 5, fmt = 's', color = 'red', markersize = 12,
                 label = 'OB170095')
    ax1.errorbar(tE_190017, piE_190017, 
                 xerr = np.array([[tE_190017_me], [tE_190017_pe]]), 
                 yerr = np.array([[piE_190017_me], [piE_190017_pe]]),
                 capsize = 5, fmt = 's', color = 'fuchsia', markersize = 12,
                 label = 'OB190017')

    ax1.set_xlabel('$t_E$ (days)')
    ax1.set_ylabel('$\pi_E$')
    ax1.set_xscale('log')
    ax1.set_yscale('log')
    ax1.legend(bbox_to_anchor=(-0.7, 0), loc="lower left", borderaxespad=0)
    ax1.set_xlim(mintE, maxtE)
    ax1.set_ylim(minpiE, maxpiE)
    tEbins = np.logspace(-1, 2.5, 26)
    piEbins = np.logspace(-4, 1, 26)


    # For labeling purposes, to make it darker in the legend.
    ax2.scatter(10**-3, 10**-3,
                alpha = 0.5, label = 'Star', marker = 's',  
                c = 'gold')
    ax2.scatter(10**-3, 10**-3,
                alpha = 0.5, label = 'WD', marker = 'P', 
                c = 'coral')
    ax2.scatter(10**-3, 10**-3,
                alpha = 0.5, label = 'NS', marker = 'v', 
                c = 'limegreen')
    ax2.scatter(10**-3, 10**-3,
                alpha = 0.5, label = 'BH', 
                c = 'k')
    
    ax2.scatter(final_delta_arr[st_idx], t['pi_E'][st_idx], 
                      alpha = 0.1, marker = 's', c = 'gold')
    ax2.scatter(final_delta_arr[wd_idx], t['pi_E'][wd_idx], 
                      alpha = 0.1, marker = 'P', c = 'coral')
    ax2.scatter(final_delta_arr[ns_idx], t['pi_E'][ns_idx], 
                      alpha = 0.1, marker = 'v', c = 'limegreen')
    ax2.scatter(final_delta_arr[bh_idx], t['pi_E'][bh_idx], 
                      alpha = 0.2, c = 'k')

    ax2.errorbar(dcmax_110022, piE_110022, 
                 xerr = np.array([[dcmax_110022_me], [dcmax_110022_pe]]), 
                 yerr = np.array([[piE_110022_me], [piE_110022_pe]]),
                 capsize = 5, fmt = 's', color = 'cyan', markersize = 12,
                 label = 'OB110022')
    ax2.errorbar(dcmax_120169, piE_120169, 
                 xerr = np.array([[dcmax_120169_me], [dcmax_120169_pe]]), 
                 yerr = np.array([[piE_120169_me], [piE_120169_pe]]),
                 capsize = 5, fmt = 's', color = 'dodgerblue', markersize = 12,
                 label = 'OB120169')
    ax2.errorbar(dcmax_140613, piE_140613, 
                 xerr = np.array([[dcmax_140613_me], [dcmax_140613_pe]]), 
                 yerr = np.array([[piE_140613_me], [piE_140613_pe]]),
                 capsize = 5, fmt = 's', color = 'navy', markersize = 12,
                 label = 'OB140613')
    ax2.errorbar(dcmax_150029, piE_150029, 
                 xerr = np.array([[dcmax_150029_me], [dcmax_150029_pe]]), 
                 yerr = np.array([[piE_150029_me], [piE_150029_pe]]),
                 capsize = 5, fmt = 's', color = 'blueviolet', markersize = 12,
                 label = 'OB150029')
    ax2.errorbar(dcmax_150211, piE_150211, 
                 xerr = np.array([[dcmax_150211_me], [dcmax_150211_pe]]), 
                 yerr = np.array([[piE_150211_me], [piE_150211_pe]]),
                 capsize = 5, fmt = 's', color = 'purple', markersize = 12,
                 label = 'OB150211')

    ax2.set_xlabel('$\delta_{c,max}$ (mas)')
    ax2.set_ylabel('$\pi_E$')
    ax2.set_xscale('log')
    ax2.set_yscale('log')
    ax2.set_xlim(mindc, maxdc)
    ax2.set_ylim(minpiE, maxpiE)

    plt.show()

    ###########
    # piE vs delta_cmax
    ###########
    plt.figure(2, figsize=(6, 6))
    plt.clf()
    plt.subplots_adjust(bottom = 0.15, right = 0.95)

    mindc = 0.005
    maxdc = 5
    minpiE = 0.003
    maxpiE = 5

    # For labeling purposes, to make it darker in the legend.
    plt.scatter(10**-3, 10**-3,
                alpha = 0.5, label = 'Star', marker = 's',  
                c = 'gold')
    plt.scatter(10**-3, 10**-3,
                alpha = 0.5, label = 'WD', marker = 'P', 
                c = 'coral')
    plt.scatter(10**-3, 10**-3,
                alpha = 0.5, label = 'NS', marker = 'v', 
                c = 'limegreen')
    plt.scatter(10**-3, 10**-3,
                alpha = 0.5, label = 'BH', 
                c = 'k')
    
    plt.scatter(t['pi_E'][st_idx], final_delta_arr[st_idx],
                      alpha = 0.1, marker = 's', c = 'gold')
    plt.scatter(t['pi_E'][wd_idx], final_delta_arr[wd_idx],
                      alpha = 0.1, marker = 'P', c = 'coral')
    plt.scatter(t['pi_E'][ns_idx], final_delta_arr[ns_idx],
                      alpha = 0.1, marker = 'v', c = 'limegreen')
    plt.scatter(t['pi_E'][bh_idx], final_delta_arr[bh_idx],
                      alpha = 0.2, c = 'k')

    plt.errorbar(piE_110022, dcmax_110022, 
                 xerr = np.array([[piE_110022_me], [piE_110022_pe]]),
                 yerr = np.array([[dcmax_110022_me], [dcmax_110022_pe]]), 
                 capsize = 5, fmt = 's', color = 'cyan', markersize = 12,
                 label = 'OB110022')
    plt.errorbar(piE_120169, dcmax_120169,
                 xerr = np.array([[piE_120169_me], [piE_120169_pe]]),
                 yerr = np.array([[dcmax_120169_me], [dcmax_120169_pe]]), 
                 capsize = 5, fmt = 's', color = 'dodgerblue', markersize = 12,
                 label = 'OB120169')
    plt.errorbar(piE_140613, dcmax_140613,  
                 xerr = np.array([[piE_140613_me], [piE_140613_pe]]),
                 yerr = np.array([[dcmax_140613_me], [dcmax_140613_pe]]), 
                 capsize = 5, fmt = 's', color = 'navy', markersize = 12,
                 label = 'OB140613')
    plt.errorbar(piE_150029, dcmax_150029,  
                 xerr = np.array([[piE_150029_me], [piE_150029_pe]]),
                 yerr = np.array([[dcmax_150029_me], [dcmax_150029_pe]]), 
                 capsize = 5, fmt = 's', color = 'blueviolet', markersize = 12,
                 label = 'OB150029')
    plt.errorbar(piE_150211, dcmax_150211, 
                 xerr = np.array([[piE_150211_me], [piE_150211_pe]]),
                 yerr = np.array([[dcmax_150211_me], [dcmax_150211_pe]]), 
                 capsize = 5, fmt = 's', color = 'purple', markersize = 12,
                 label = 'OB150211')

    plt.xlabel('$\pi_E$')
    plt.ylabel('$\delta_{c,max}$ (mas)')
    plt.xscale('log')
    plt.yscale('log')
#    plt.legend(loc=4)
    plt.xlim(minpiE, maxpiE)
    plt.ylim(mindc, maxdc)

    plt.show()

    ###########
    # piE vs delta_cmax
    ###########
    plt.figure(2, figsize=(6, 6))
    plt.clf()

    mindc = 0.005
    maxdc = 5
    minpiE = 0.003
    maxpiE = 5

    # For labeling purposes, to make it darker in the legend.
    plt.scatter(10**-3, 10**-3,
                alpha = 0.5, label = 'Star', marker = 's',  
                c = 'gold')
    plt.scatter(10**-3, 10**-3,
                alpha = 0.5, label = 'WD', marker = 'P', 
                c = 'coral')
    plt.scatter(10**-3, 10**-3,
                alpha = 0.5, label = 'NS', marker = 'v', 
                c = 'limegreen')
    plt.scatter(10**-3, 10**-3,
                alpha = 0.5, label = 'BH', 
                c = 'k')
    
    plt.scatter(t['pi_E'][st_idx], final_delta_arr[st_idx],
                      alpha = 0.1, marker = 's', c = 'gold')
    plt.scatter(t['pi_E'][wd_idx], final_delta_arr[wd_idx],
                      alpha = 0.1, marker = 'P', c = 'coral')
    plt.scatter(t['pi_E'][ns_idx], final_delta_arr[ns_idx],
                      alpha = 0.1, marker = 'v', c = 'limegreen')
    plt.scatter(t['pi_E'][bh_idx], final_delta_arr[bh_idx],
                      alpha = 0.2, c = 'k')

    plt.errorbar(piE_110022, dcmax_110022, 
                 xerr = np.array([[piE_110022_me], [piE_110022_pe]]),
                 yerr = np.array([[dcmax_110022_me], [dcmax_110022_pe]]), 
                 capsize = 5, fmt = 's',
                 label = 'OB110022')
    plt.errorbar(piE_120169, dcmax_120169,
                 xerr = np.array([[piE_120169_me], [piE_120169_pe]]),
                 yerr = np.array([[dcmax_120169_me], [dcmax_120169_pe]]), 
                 capsize = 5, fmt = 's',
                 label = 'OB120169')
    plt.errorbar(piE_140613, dcmax_140613,  
                 xerr = np.array([[piE_140613_me], [piE_140613_pe]]),
                 yerr = np.array([[dcmax_140613_me], [dcmax_140613_pe]]), 
                 capsize = 5, fmt = 's',
                 label = 'OB140613')
    plt.errorbar(piE_150029, dcmax_150029,  
                 xerr = np.array([[piE_150029_me], [piE_150029_pe]]),
                 yerr = np.array([[dcmax_150029_me], [dcmax_150029_pe]]), 
                 capsize = 5, fmt = 's',
                 label = 'OB150029')
    plt.errorbar(piE_150211, dcmax_150211, 
                 xerr = np.array([[piE_150211_me], [piE_150211_pe]]),
                 yerr = np.array([[dcmax_150211_me], [dcmax_150211_pe]]), 
                 capsize = 5, fmt = 's',
                 label = 'OB150211')

    plt.xlabel('$\pi_E$')
    plt.ylabel('$\delta_{c,max}$ (mas)')
    plt.xscale('log')
    plt.yscale('log')
    plt.legend(loc=4)
    plt.xlim(minpiE, maxpiE)
    plt.ylim(mindc, maxdc)

    plt.legend(bbox_to_anchor=(-0.2,0), loc="lower left", borderaxespad=0)

    plt.show()

    ###########
    # piE vs tE
    ###########
    plt.figure(1, figsize=(8, 6))
    plt.clf()
    plt.subplots_adjust(left = 0.4, bottom = 0.15, right = 0.96)

    minpiE = 0.003
    maxpiE = 5
    mintE = 0.8
    maxtE = 500

    # For labeling purposes, to make it darker in the legend.
    plt.scatter(10**-3, 10**-3,
                alpha = 0.5, label = 'Star', marker = 's',  
                c = 'gold')
    plt.scatter(10**-3, 10**-3,
                alpha = 0.5, label = 'WD', marker = 'P', 
                c = 'coral')
    plt.scatter(10**-3, 10**-3,
                alpha = 0.5, label = 'NS', marker = 'v', 
                c = 'limegreen')
    plt.scatter(10**-3, 10**-3,
                alpha = 0.5, label = 'BH', 
                c = 'k')
    
    plt.scatter(t['t_E'][st_idx], t['pi_E'][st_idx],
                alpha = 0.2, marker = 's', c = 'gold', label = '')
    plt.scatter(t['t_E'][wd_idx], t['pi_E'][wd_idx],
                alpha = 0.2, marker = 'P', c = 'coral', label = '')
    plt.scatter(t['t_E'][ns_idx], t['pi_E'][ns_idx], 
                alpha = 0.2, marker = 'v', c = 'limegreen', label = '')
    plt.scatter(t['t_E'][bh_idx], t['pi_E'][bh_idx], 
                alpha = 0.2, c = 'k', label = '')

    plt.errorbar(tE_110022, piE_110022, 
                 xerr = np.array([[tE_110022_me], [tE_110022_pe]]), 
                 yerr = np.array([[piE_110022_me], [piE_110022_pe]]),
                 capsize = 5, fmt = 's', color = 'cyan', markersize = 12,
                 label = 'OB110022')
    plt.errorbar(tE_120169, piE_120169, 
                 xerr = np.array([[tE_120169_me], [tE_120169_pe]]), 
                 yerr = np.array([[piE_120169_me], [piE_120169_pe]]),
                 capsize = 5, fmt = 's', color = 'dodgerblue', markersize = 12,
                 label = 'OB120169')
    plt.errorbar(tE_140613, piE_140613, 
                 xerr = np.array([[tE_140613_me], [tE_140613_pe]]), 
                 yerr = np.array([[piE_140613_me], [piE_140613_pe]]),
                 capsize = 5, fmt = 's', color = 'navy', markersize = 12,
                 label = 'OB140613')
    plt.errorbar(tE_150029, piE_150029, 
                 xerr = np.array([[tE_150029_me], [tE_150029_pe]]), 
                 yerr = np.array([[piE_150029_me], [piE_150029_pe]]),
                 capsize = 5, fmt = 's', color = 'blueviolet', markersize = 12,
                 label = 'OB150029')
    plt.errorbar(tE_150211, piE_150211, 
                 xerr = np.array([[tE_150211_me], [tE_150211_pe]]), 
                 yerr = np.array([[piE_150211_me], [piE_150211_pe]]),
                 capsize = 5, fmt = 's', color = 'purple', markersize = 12,
                 label = 'OB150211')
    plt.errorbar(tE_170019, piE_170019, 
                 xerr = np.array([[tE_170019_me], [tE_170019_pe]]), 
                 yerr = np.array([[piE_170019_me], [piE_170019_pe]]),
                 capsize = 5, fmt = 's', color = 'deeppink', markersize = 12,
                 label = 'OB170019')
    plt.errorbar(tE_170095, piE_170095, 
                 xerr = np.array([[tE_170095_me], [tE_170095_pe]]), 
                 yerr = np.array([[piE_170095_me], [piE_170095_pe]]),
                 capsize = 5, fmt = 's', color = 'red', markersize = 12,
                 label = 'OB170095')
    plt.errorbar(tE_190017, piE_190017, 
                 xerr = np.array([[tE_190017_me], [tE_190017_pe]]), 
                 yerr = np.array([[piE_190017_me], [piE_190017_pe]]),
                 capsize = 5, fmt = 's', color = 'fuchsia', markersize = 12,
                 label = 'OB190017')

    plt.xlabel('$t_E$ (days)')
    plt.ylabel('$\pi_E$')
    plt.xscale('log')
    plt.yscale('log')
#    plt.legend(loc=2)
    plt.legend(bbox_to_anchor=(-0.7, 0), loc="lower left", borderaxespad=0)
    plt.xlim(mintE, maxtE)
    plt.ylim(minpiE, maxpiE)
    tEbins = np.logspace(-1, 2.5, 26)
    piEbins = np.logspace(-4, 1, 26)

    plt.show()

    ###########
    # piE vs delta_cmax
    ###########
    plt.figure(2, figsize=(6, 6))
    plt.clf()
    plt.subplots_adjust(bottom = 0.15, right = 0.95)

    mindc = 0.005
    maxdc = 5
    minpiE = 0.003
    maxpiE = 5

    # For labeling purposes, to make it darker in the legend.
    plt.scatter(10**-3, 10**-3,
                alpha = 0.5, label = 'Star', marker = 's',  
                c = 'gold')
    plt.scatter(10**-3, 10**-3,
                alpha = 0.5, label = 'WD', marker = 'P', 
                c = 'coral')
    plt.scatter(10**-3, 10**-3,
                alpha = 0.5, label = 'NS', marker = 'v', 
                c = 'limegreen')
    plt.scatter(10**-3, 10**-3,
                alpha = 0.5, label = 'BH', 
                c = 'k')
    
    plt.scatter(t['pi_E'][st_idx], final_delta_arr[st_idx],
                      alpha = 0.1, marker = 's', c = 'gold')
    plt.scatter(t['pi_E'][wd_idx], final_delta_arr[wd_idx],
                      alpha = 0.1, marker = 'P', c = 'coral')
    plt.scatter(t['pi_E'][ns_idx], final_delta_arr[ns_idx],
                      alpha = 0.1, marker = 'v', c = 'limegreen')
    plt.scatter(t['pi_E'][bh_idx], final_delta_arr[bh_idx],
                      alpha = 0.2, c = 'k')

    plt.errorbar(piE_110022, dcmax_110022, 
                 xerr = np.array([[piE_110022_me], [piE_110022_pe]]),
                 yerr = np.array([[dcmax_110022_me], [dcmax_110022_pe]]), 
                 capsize = 5, fmt = 's', color = 'cyan', markersize = 12,
                 label = 'OB110022')
    plt.errorbar(piE_120169, dcmax_120169,
                 xerr = np.array([[piE_120169_me], [piE_120169_pe]]),
                 yerr = np.array([[dcmax_120169_me], [dcmax_120169_pe]]), 
                 capsize = 5, fmt = 's', color = 'dodgerblue', markersize = 12,
                 label = 'OB120169')
    plt.errorbar(piE_140613, dcmax_140613,  
                 xerr = np.array([[piE_140613_me], [piE_140613_pe]]),
                 yerr = np.array([[dcmax_140613_me], [dcmax_140613_pe]]), 
                 capsize = 5, fmt = 's', color = 'navy', markersize = 12,
                 label = 'OB140613')
    plt.errorbar(piE_150029, dcmax_150029,  
                 xerr = np.array([[piE_150029_me], [piE_150029_pe]]),
                 yerr = np.array([[dcmax_150029_me], [dcmax_150029_pe]]), 
                 capsize = 5, fmt = 's', color = 'blueviolet', markersize = 12,
                 label = 'OB150029')
    plt.errorbar(piE_150211, dcmax_150211, 
                 xerr = np.array([[piE_150211_me], [piE_150211_pe]]),
                 yerr = np.array([[dcmax_150211_me], [dcmax_150211_pe]]), 
                 capsize = 5, fmt = 's', color = 'purple', markersize = 12,
                 label = 'OB150211')

    plt.xlabel('$\pi_E$')
    plt.ylabel('$\delta_{c,max}$ (mas)')
    plt.xscale('log')
    plt.yscale('log')
#    plt.legend(loc=4)
    plt.xlim(minpiE, maxpiE)
    plt.ylim(mindc, maxdc)

    plt.show()

    ###########
    # tE vs delta_cmax
    ###########
    plt.figure(3, figsize=(6, 6))
    plt.clf()

    mindc = 0.005
    maxdc = 5
    mintE = 0.8
    maxtE = 500    
    
    # For labeling purposes, to make it darker in the legend.
    plt.scatter(10**-3, 10**-3,
                alpha = 0.5, label = 'Star', marker = 's',  
                c = 'gold')
    plt.scatter(10**-3, 10**-3,
                alpha = 0.5, label = 'WD', marker = 'P', 
                c = 'coral')
    plt.scatter(10**-3, 10**-3,
                alpha = 0.5, label = 'NS', marker = 'v', 
                c = 'limegreen')
    plt.scatter(10**-3, 10**-3,
                alpha = 0.5, label = 'BH', 
                c = 'k')

    plt.scatter(t['t_E'][st_idx], final_delta_arr[st_idx],
                      alpha = 0.2, marker = 's', c = 'gold')
    plt.scatter(t['t_E'][wd_idx], final_delta_arr[wd_idx],
                      alpha = 0.2, marker = 'P', c = 'coral')
    plt.scatter(t['t_E'][ns_idx], final_delta_arr[ns_idx],
                      alpha = 0.2, marker = 'v', c = 'limegreen')
    plt.scatter(t['t_E'][bh_idx], final_delta_arr[bh_idx],
                      alpha = 0.2, c = 'k')

    plt.errorbar(tE_110022, dcmax_110022,  
                 xerr = np.array([[tE_110022_me], [tE_110022_pe]]),
                 yerr = np.array([[dcmax_110022_me], [dcmax_110022_pe]]), 
                 capsize = 5, fmt = 's',
                 label = 'OB110022')
    plt.errorbar(tE_120169, dcmax_120169,  
                 xerr = np.array([[tE_120169_me], [tE_120169_pe]]),
                 yerr = np.array([[dcmax_120169_me], [dcmax_120169_pe]]), 
                 capsize = 5, fmt = 's',
                 label = 'OB120169')
    plt.errorbar(tE_140613, dcmax_140613,  
                 xerr = np.array([[tE_140613_me], [tE_140613_pe]]),
                 yerr = np.array([[dcmax_140613_me], [dcmax_140613_pe]]), 
                 capsize = 5, fmt = 's',
                 label = 'OB140613')
    plt.errorbar(tE_150029, dcmax_150029, 
                 xerr = np.array([[tE_150029_me], [tE_150029_pe]]),
                 yerr = np.array([[dcmax_150029_me], [dcmax_150029_pe]]), 
                 capsize = 5, fmt = 's',
                 label = 'OB150029')
    plt.errorbar(tE_150211, dcmax_150211,  
                 xerr = np.array([[tE_150211_me], [tE_150211_pe]]),
                 yerr = np.array([[dcmax_150211_me], [dcmax_150211_pe]]), 
                 capsize = 5, fmt = 's',
                 label = 'OB150211')
 
    plt.xlabel('$t_E$ (days)')
    plt.ylabel('$\delta_{c,max}$ (mas)')
    plt.xscale('log')
    plt.yscale('log')
    plt.legend(loc=2)
    plt.xlim(mintE, maxtE)
    plt.ylim(mindc, maxdc)

    plt.show()

    return


def plot_airmass_moon():
    # coordinates to center of OGLE field BLG502 (2017 objects)
    ra = "18:00:00"
    dec = "-30:00:00"
    months = np.array([4, 5, 6, 7])
    days = np.array([10, 15, 15, 30])
    # outdir = '/Users/jlu/doc/proposals/keck/uc/18A/'
    outdir = '/Users/fatima/Desktop/'
    # outdir = '/Users/casey/scratch/'

    # Keck 2
    skycalc.plot_airmass(ra, dec, 2020, months, days, 'keck2', outfile=outdir + 'microlens_airmass_keck2_20A.png', date_idx=-1)
    skycalc.plot_moon(ra, dec, 2020, months, outfile=outdir + 'microlens_moon_20A.png')

    # Keck 1
    skycalc.plot_airmass(ra, dec, 2020, months, days, 'keck1', outfile=outdir + 'microlens_airmass_keck1_20A.png', date_idx=-1)
    
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
    #mod_fit.summarize_results_modes()

    mod_all = mod_fit.get_best_fit_modes_model(def_best='median')
    tab_all = mod_fit.load_mnest_modes()
    
    def plot_4panel(mod, tab):
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
        r_min_k = data['mag1'][tidx] - data['mag2'][-1]
        r_min_k = 4.0
        print('r_min_k = ', r_min_k)

        # Plotting        
        plt.figure(2, figsize=(18, 4))

        pan_wid = 0.15
        pan_pad = 0.09
        fig_pos = np.arange(0, 4) * (pan_wid + pan_pad) + pan_pad
        print(fig_pos)

        # Brightness vs. time
        fm1 = plt.gcf().add_axes([fig_pos[0], 0.36, pan_wid, 0.6])
        fm2 = plt.gcf().add_axes([fig_pos[0], 0.18, pan_wid, 0.2])
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
                        yerr=data['xpos_err']*1e3, fmt='k.')
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
                        yerr=data['ypos_err']*1e3, fmt='k.')
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
        f4.set_yticks(np.array([0.0, -0.2]))
        f4.set_xlabel('Time (HJD)')
        f4.set_ylabel('Res.')


        # Mass posterior
        masses = 10**tab['log_thetaE'] / (8.14 * 10**tab['log_piE'])
        weights = tab['weights']
        print(masses[0:10])
        print(weights[0:10])
        
        f5 = plt.gcf().add_axes([fig_pos[3], 0.18, pan_wid, 0.8])
        f5.hist(masses, weights=weights, bins=50, alpha = 0.9)
        f5.set_xlabel('Mass (M$_\odot$)')
        f5.set_ylabel('Probability')
        f5.set_xlim(0, 2)


    plt.close(2)
    plot_4panel(mod_all[1], tab_all[1])
    

    return
    

def OLD_MAP_NUMBERS_DONT_USE():
    #############
    # The observations. 
    # OB110022 from Lu+16.
    # Others are phot parallax solutions (global MAP)
    #############
    # OB110022
    piEE_110022 = -0.393
    piEN_110022 = -0.071
    piE_110022 = np.hypot(piEE_110022, piEN_110022)
    tE_110022 = 61.4
    dcmax_110022 = 2.19/np.sqrt(8)

    ##########
    # OB120169
    ##########
    # Phot only
    piEE_120169_p = 0.0130
    piEN_120169_p = -0.1367
    piE_120169_p = np.hypot(piEE_120169_p, piEN_120169_p)
    tE_120169_p = 163.26
    # Phot + astrom
    piEE_120169_pa = -0.0129
    piEN_120169_pa = -0.1639
    piE_120169_pa = np.hypot(piEE_120169_pa, piEN_120169_pa)
    tE_120169_pa = 185.27
    dcmax_120169_pa = (10**-1.38)/np.sqrt(8)
    # Nijaid's calculation on the astrometric shift
    dcmax_120169_ni = 0.155

    ##########
    # OB140613
    ##########
    # Phot only
    piEE_140613_p = -0.1128
    piEN_140613_p = 0.0752
    piE_140613_p = np.hypot(piEE_140613_p, piEN_140613_p)
    tE_140613_p = 320.97
    # Phot + astrom
    piEE_140613_pa = -0.1179
    piEN_140613_pa = 0.0787
    piE_140613_pa = np.hypot(piEE_140613_pa, piEN_140613_pa)
    tE_140613_pa = 304.14
    dcmax_140613_pa = (10**-0.0237)/np.sqrt(8)
    # Nijaid's calculation on the astrometric shift
    dcmax_140613_ni = 0.423

    ##########
    # OB150029
    ##########
    # Phot only
    piEE_150029_p = 0.0504
    piEN_150029_p = 0.1669
    piE_150029_p = np.hypot(piEE_150029_p, piEN_150029_p)
    tE_150029_p = 154.38
    # Phot + astrom
    piEE_150029_pa = 0.0670
    piEN_150029_pa = 0.163
    piE_150029_pa = np.hypot(piEE_150029_pa, piEN_150029_pa)
    tE_150029_pa = 138.30
    dcmax_150029_pa = (10**-0.188)/np.sqrt(8)
    # Nijaid's calculation on the astrometric shift
    dcmax_150029_ni = 0.276

    ##########
    # OB150211
    ##########
    # Phot only
    piEE_150211_p = 0.0463
    piEN_150211_p = -0.0294
    piE_150211_p = np.hypot(piEE_150211_p, piEN_150211_p)
    tE_150211_p = 111.78
    # Phot + astrom
    piEE_150211_pa = 0.030
    piEN_150211_pa = -0.017
    piE_150211_pa = np.hypot(piEE_150211_pa, piEN_150211_pa)
    tE_150211_pa = 115.49
    dcmax_150211_pa = (10**-0.294)/np.sqrt(8)
    # Nijaid's calculation on the astrometric shift
    dcmax_150211_ni = 0.407

    ##########
    # OB170019
    ##########
    # EWS phot only
    piEE_170019 = -0.0079
    piEN_170019 = -0.0568
    piE_170019 = np.hypot(piEE_170019, piEN_170019)
    tE_170019 = 129.03 

    ##########
    # OB170095
    ##########
    # EWS phot only
    piEE_170095 = -0.039
    piEN_170095 = 0.0058
    piE_170095 = np.hypot(piEE_170095, piEN_170095)
    tE_170095 = 105.50 

    ##########
    # OB190017
    ##########
    # EWS phot only
    piEE_190017 = -0.123
    piEN_190017 = -0.131
    piE_190017 = np.hypot(piEE_190017, piEN_190017)
    tE_190017 = 258.70 

    ##########
    # OB190033
    ##########
    # EWS phot only
    piEE_190033 = -0.059
    piEN_190033 = 0.438
    piE_190033 = np.hypot(piEE_190033, piEN_190033)
    tE_190033 = 135.52
