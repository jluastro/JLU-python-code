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


mb09260_data = munge.getdata2('mb09260', phot_data=['MOA', 'HST'], ast_data = [])  
mb10364_data = munge.getdata2('mb10364', phot_data=['MOA', 'HST'], ast_data = [])  
ob110037_data = munge.getdata2('ob110037', phot_data=['I_OGLE', 'MOA', 'HST'], ast_data = [])  
ob110310_data = munge.getdata2('ob110310', phot_data=['I_OGLE', 'MOA', 'HST'], ast_data = [])  
ob110462_data = munge.getdata2('ob110462', phot_data=['I_OGLE', 'MOA', 'HST'], ast_data = [])  

# Some of these may still require rerunning... 
mb09260_dir = '/u/jlu/work/microlens/MB09260/a_2020_03_26/model_fits/202_fit_multiphot_parallax/b0_'
mb10364_dir = '/u/jlu/work/microlens/MB10364/a_2020_02_18/model_fits/202_fit_multiphot_parallax/b0_'
ob110037_dir = '/u/jlu/work/microlens/OB110037/a_2020_01_22/model_fits/302_fit_3phot_parallax/b0_'
ob110310_dir = '/u/jlu/work/microlens/OB110310/a_2020_01_22/model_fits/302_fit_3phot_parallax/a0_'
ob110462_dir = '/u/jlu/work/microlens/OB110462/a_2020_01_22/model_fits/302_fit_3phot_parallax/a0_'

# run directory                                                                                             
ob120169_dir = '/u/jlu/work/microlens/OB120169/a_2019_06_26/model_fits/120_fit_multiphot_astrom_parallax_aerr/base_c/'
ob140613_dir = '/u/jlu/work/microlens/OB140613/a_2019_06_26/model_fits/120_fit_multiphot_astrom_parallax_merr/base_b/'
ob150029_dir = '/u/jlu/work/microlens/OB150029/a_2019_06_26/model_fits/120_fit_multiphot_astrom_parallax_aerr/base_b/'
ob150211_dir = '/u/jlu/work/microlens/OB150211/a_2019_06_26/model_fits/120_fit_multiphot_astrom_parallax_aerr/base_b/'

# run id                                                                                          
ob120169_id = 'c2'
ob140613_id = 'b1'
ob150029_id = 'b3'
ob150211_id = 'b2'


def piE_tE():
    # Plot the piE-tE 2D posteriors.
    plt.close(1)
    fig = plt.figure(1, figsize=(6,6))
    plt.clf()
    axes = plt.gca()
    plt.subplots_adjust(bottom=0.15)

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

    # Values are PSBL fit from email from Dave Bennett
    # Subject line is "Mid-Cycle HST proposal?", reply is Jan 25, 2021
    axes.scatter(499.482, 0.038397, marker='*', s=100, color='red')
    axes.text(300, 0.028, 'MB19284', color='red')

    axes.set_xlim(10, 1000)
    axes.set_ylim(0.005, 0.5)
    axes.set_xlabel('$t_E$ (days)')
    axes.set_ylabel('$\pi_E$')
    axes.set_xscale('log')
    axes.set_yscale('log')
    axes.legend(loc=1)
    plt.show()


def plot_psbl_model():
    """
    """
    # Parameters from Dave Bennett
    # t_E = Einstein radius for the total lens mass
    # t0  = time of closest approach between the source and lens center-of-mass
    # umin = the closest approach of the source to the lens center-of-mass in units of the 
    #     Einstein ring radius. I use the opposite convention of most people, so that umin = -u0.
    #     In units of thetaE
    # sep_0 = the separation of the two lens stars at t = t_fix
    # theta = the angle between the source trajectory and the lens axis (also at t = t_fix)
    # eps1 = mass fraction of lens 1 eps1 = M_1/M_total
    # eps2 = M_2/M_total = 1 - eps1 (so not a model parameter)
    # piEN = the North component of the 2-d microlensing parallax vector
    # piEE = the East component of the 2-d microlensing parallax vector
    # t_fix = the reference time for the orbital motion parameters and Geocentric coordinate system used
    # dsxdt = orbital velocity of lens 1 vs. lens 2 in the x-direction (the lens axis) at t = t_fix
    #     in units of Einstein radius per day
    # dsydt = orbital velocity of lens 1 vs. lens 2 in the y-direction at t = t_fix
    #     in units of Einstein radius per day
    # Tbin = orbital period of the lens system in days
    # 
    # origin is at center of mass
    #
    # Without parallax, the coordinates of the source are given by:
    # vfac = (t-t0)/t_E
    #  sx = cos(theta)*vfac - sin(theta)*umin
    #  sy = sin(theta)*vfac + cos(theta)*umin
    #
    # The x and y coordinates for masses 1 and 2 (on the lens axis) are given by
    #
    #       eps2 = 1.-eps1
    #       x1 = -eps2*sep
    #       x2 =  eps1*sep
    #       y1 = 0.
    #       y2 = 0.
    # I don’t define either mass 1 or mass 2 to be the primary. eps1 can change from
    # < 0.5 to > 0.5 during the modeling, and it would require a change of coordinates
    # to insist that either mass 1 or mass 2 be the primary.

    db_t_E = 499.482 # days
    db_t0 = 9156.1432
    db_umin = -0.737474
    db_sep_0 = 1.29465 # thetaE
    db_theta = 1.52948
    db_eps1 = 0.5117
    db_eps2 = 0.48830
    db_piEN = 0.029339
    db_piEE = -0.02477
    db_t_fix = 9000
    db_dsxdt = 0.00003707
    db_dsydt = 0.00012346
    db_Tbin = 6010.465

    # need this?
    pi_E = 0.038397

    # Properties we need:
    t0 = db_t0 + 50000.0   # get to MJD?
    u0_amp = db_umin * -1.0
    tE = db_t_E
    piE_E = db_piEE
    piE_N = db_piEN
    q = db_eps1 / db_eps2
    if db_eps1 > db_eps2:
        q = 1. / q
    sep = db_sep_0
    phi = np.degrees(db_theta)
    b_sff = 1.0    # Dave didn't send me this parameter.
    mag_src = 18.0  # I made this up.

    raL_str = '18:05:55.06'
    decL_str = '-30:20:12.94'

    target_coords = SkyCoord(raL_str, decL_str, unit=(u.hourangle, u.deg), frame='icrs')
    raL = target_coords.ra.degree
    decL = target_coords.dec.degree

    # Past and planned measurements.
    # t_type = np.array(['hst', 'hst', 'hst', 'keck', 'keck', 'keck'])
    # t_data = np.array([2458922.186, 2458986.572, 2459106.863, 2459025.387, 2459052.362, 2459083.267]) 
    t_type = np.array(['hst', 'hst', 'keck', 'keck', 'keck', 'new', 'new', 'new', 'new'])
    t_data = np.array([2458922.186, 2458986.572, 2459025.387, 2459052.362, 2459083.267,
                           2459298.0, 2459363, 2459423, 2459487]) 
    t_data -= 2400000.5

    # Gaia EDR3 values
    gaia_vx = 3.28  # error = 0.75 mas/yr
    gaia_vy = 1.36  # error = 0.75 mas/yr
    gaia_t0 = 2015.5
    gaia_par = 0.75 # error = 0.42 mas

    # MB190284 fit results (from Dave Bennett) -- full posteriors
    data_tab = '/u/jlu/doc/proposals/hst/cycle28_mid2/mcmc_bsopcnC_3.dat'

    # chi^2 1/t_E t0 umin sep theta eps1=q/(1+q) 1/Tbin dsxdt dsydt t_fix Tstar(=0)
    # pi_E,N piE,E 0 0 0 0 0 0 0 0 0 A0ogleI A2ogleI A0ogleV A2ogleV A0moa2r A2moa2r A0moa2V
    post = Table.read(data_tab, format='ascii.fixed_width_no_header', delimiter=' ')
    post.rename_column('col1', 'chi2')
    post.rename_column('col2', 'tE_inv')
    post.rename_column('col3', 't0')
    post.rename_column('col4', 'u0')
    post.rename_column('col5', 'sep')
    post.rename_column('col6', 'theta')
    post.rename_column('col7', 'eps1')
    post.rename_column('col8', 'Tbin_inv')
    post.rename_column('col9', 'dsxdt')
    post.rename_column('col10', 'dsydt')
    post.rename_column('col11', 't_fix')
    post.rename_column('col12', 'Tstar')
    post.rename_column('col13', 'piEN')
    post.rename_column('col14', 'piEE')
    post.rename_column('col24', 'A0ogleI')
    post.rename_column('col25', 'A2ogleI')
    
    # Switch to our preferred parameterization
    post['tE'] = 1.0 / post['tE_inv']
    post['piEE'] = post['piEE'].astype('float')
    post['weight'] = np.ones(len(post))
    post['piE'] = np.hypot(post['piEE'], post['piEN'])
    post['t0'] += 50000.0
    post['u0_amp'] = post['u0'] * -1.0
    post['q'] = post['eps1'] / (1.0 - post['eps1'])
    post['phi'] = np.degrees(post['theta'])
    post['mag_src'] = -2.5 * np.log10(post['A0ogleI']) + 21
    post['b_sff'] = post['A0ogleI'] / (post['A0ogleI'] + post['A2ogleI'])
    
    # Re-set source flux to posterior mean.
    mag_src = post['mag_src'].mean()
    b_sff = post['b_sff'].mean()
    

    #####
    # Make the model
    #####
    psbl = model.PSBL_Phot_Par_Param1(t0, u0_amp, tE, piE_E, piE_N,
                                          q, sep, phi, b_sff, mag_src,
                                          raL=raL, decL=decL)

    psbl_post = []
    for ii in np.random.randint(0, len(post), size=100):
        psbl_tmp = model.PSBL_Phot_Par_Param1(post['t0'][ii], post['u0_amp'][ii], post['tE'][ii],
                                              post['piEE'][ii], post['piEN'][ii],
                                              post['q'][ii], post['sep'][ii], post['phi'][ii],
                                              post['b_sff'][ii], post['mag_src'][ii],
                                              raL=raL, decL=decL)
        psbl_post.append(psbl_tmp)


    #####
    # Plotting
    #####
    t_obs = np.arange(57000, 61000, 3)

    
    images, amps = psbl.get_all_arrays(t_obs)

    images_post = []
    amps_post = []
    for ii in range(len(psbl_post)):
        foo = psbl_post[ii].get_all_arrays(t_obs)
        images_post.append(foo[0])
        amps_post.append(foo[1])
    

    ##########
    # Photometry
    ##########
    phot = psbl.get_photometry(t_obs, amp_arr=amps)

    # Plot the photometry
    plt.figure(1)
    plt.clf()
    
    for ii in range(len(psbl_post)):
        phot_tmp = psbl_post[ii].get_photometry(t_obs, amp_arr=amps_post[ii])
        plt.plot(t_obs, phot_tmp, 'r-', color='pink', alpha=0.05)
        
    plt.plot(t_obs, phot, 'r-', lw=2)
    
    plt.ylabel('Photometry (mag)')
    plt.xlabel('Time (MJD)')
    plt.gca().invert_yaxis()
    
    ##########
    # Astrometry Plots
    ##########
    # Find the points closest to t0
    t0idx = np.argmin(np.abs(t_obs - psbl.t0))


    xL1, xL2 = psbl.get_resolved_lens_astrometry(t_obs)
    xS_unlens = psbl.get_astrometry_unlensed(t_obs)
    xS_lensed = psbl.get_astrometry(t_obs, image_arr=images, amp_arr=amps)

    dxS = (xS_lensed - xS_unlens)


    # Plot the positions of everything
    plt.figure(2)
    plt.clf()
    plt.plot(xS_unlens[:, 0], xS_unlens[:, 1], 'b--', mfc='blue',
             mec='blue')
    plt.plot(xS_lensed[:, 0], xS_lensed[:, 1], 'b-')
    plt.plot(xL1[:, 0], xL1[:, 1], 'g--', mfc='none',
             mec='green')
    plt.plot(xL2[:, 0], xL2[:, 1], 'g--', mfc='none',
             mec='dark green')

    plt.plot(xS_unlens[t0idx, 0], xS_unlens[t0idx, 1], 'bx', mfc='blue',
             mec='blue',
             label='xS, unlensed')
    plt.plot(xS_lensed[t0idx, 0], xS_lensed[t0idx, 1], 'bo',
             label='xS, lensed')
    plt.plot(xL1[t0idx, 0], xL1[t0idx, 1], 'gs', mfc='green',
             mec='green',
             label='Primary lens')
    plt.plot(xL2[t0idx, 0], xL2[t0idx, 1], 'gs', mfc='none',
             mec='green',
             label='Secondary lens')

    for tt in range(len(t_data)):
        idx = np.argmin(np.abs(t_obs - t_data[tt]))
        if tt == 0:
            label = 'Past Data'
        else:
            label = ''

        mec = 'blue'
        if t_type[tt] == 'new':
            mfc = 'none'
        else:
            mfc = 'blue'
        plt.plot(xS_lensed[idx, 0], xS_lensed[idx, 1], 'bs', mfc=mfc, mec=mec, label=label)
    

    plt.legend(fontsize=14)
    plt.ylim(-2.0, 3.0)
    plt.xlim(-2.0, 3.0)
    plt.gca().invert_xaxis()
    plt.xlabel(r'R.A. ($\theta_E$)')
    plt.ylabel(r'Dec. ($\theta_E$)')
    plt.savefig('astrom_on_sky.png')

    

    # Check just the astrometric shift part.
    fig3 = plt.figure(3)
    plt.clf()
    ax_x = fig3.add_subplot(2, 1, 1)
    ax_y = fig3.add_subplot(2, 1, 2, sharex=ax_x, sharey=ax_x)
    ax_x.label_outer()
    ax_y.label_outer()

    for ii in range(len(psbl_post)):
        xS_unlens_tmp = psbl_post[ii].get_astrometry_unlensed(t_obs)
        xS_lensed_tmp = psbl_post[ii].get_astrometry(t_obs, image_arr=images_post[ii], amp_arr=amps_post[ii])

        dxS_tmp = (xS_lensed_tmp - xS_unlens_tmp)
        
        ax_x.plot(t_obs, dxS_tmp[:, 0], 'r-', color='pink', alpha=0.05)
        ax_y.plot(t_obs, dxS_tmp[:, 1], 'b-', color='cyan', alpha=0.05)
    
    ax_x.plot(t_obs, dxS[:, 0], 'r-', label='R.A.')
    ax_y.plot(t_obs, dxS[:, 1], 'b-', label='Dec.')

    for tt in range(len(t_data)):
        idx = np.argmin(np.abs(t_obs - t_data[tt]))
        if tt == 0:
            label = 'Past Data'
        else:
            label = ''

        mec = 'black'
        if t_type[tt] == 'new':
            mfc = 'none'
        else:
            mfc = 'black'
            
        ax_x.plot(t_obs[idx], dxS[idx, 0], 'ks', mec=mec, mfc=mfc, label=label)
        ax_y.plot(t_obs[idx], dxS[idx, 1], 'ks', mec=mec, mfc=mfc, label=label)
        

    ax_x.legend(fontsize=14)
    ax_y.legend(fontsize=14)
    fig3.text(0.04, 0.5, r'Astrometric Shift ($\theta_E$)', va='center', rotation='vertical', fontsize=16)
    ax_y.set_xlabel('Time (MJD)')

    ax_x.set_xlim(57600, 60400)
    plt.savefig('astrom_shift_v_time.png')

    
    # Just the shift in lens rest frame.
    plt.figure(4)
    plt.clf()
    plt.plot(dxS[:, 0], dxS[:, 1], 'r-')
    plt.axhline(0, linestyle='--')
    plt.axvline(0, linestyle='--')
    plt.gca().invert_xaxis()
    plt.axis('equal')

    for tt in range(len(t_data)):
        idx = np.argmin(np.abs(t_obs - t_data[tt]))
        if tt == 0:
            label = 'Past Data'
        else:
            label = ''

        mec = 'black'
        if t_type[tt] == 'new':
            mfc = 'none'
        else:
            mfc = 'black'
            
        plt.plot(dxS[idx, 0], dxS[idx, 1], 'ks', mfc=mfc, mec=mec, label=label)

    plt.xlabel(r'Shift RA ($\theta_E$)')
    plt.ylabel(r'Shift Dec ($\theta_E$)')
    plt.savefig('astrom_shift.png')


    return

def piE_tE_phot_only_fits():
    """
    !!! NOTE: CHOICE OF THE quantiles_2d HAS A LARGE EFFECT 
    ON THE WAY THIS PLOT LOOKS !!!
    Plot piE-tE 2D posteriors from OGLE photometry only fits.
    Also plot PopSyCLE simulations simultaneously.
    """

    mb09260_yaml = open(mb09260_dir + 'params.yaml').read() 
    params_09260 = yaml.safe_load(mb09260_yaml)
    mb10364_yaml = open(mb10364_dir + 'params.yaml').read() 
    params_10364 = yaml.safe_load(mb10364_yaml)
    ob110037_yaml = open(ob110037_dir + 'params.yaml').read() 
    params_110037 = yaml.safe_load(ob110037_yaml)
    ob110310_yaml = open(ob110310_dir + 'params.yaml').read() 
    params_110310 = yaml.safe_load(ob110310_yaml)
    ob110462_yaml = open(ob110462_dir + 'params.yaml').read() 
    params_110462 = yaml.safe_load(ob110462_yaml)

    # Cycle 17-24 Sahu results tE-piE.
    fitter_09260 = model_fitter.PSPL_Solver(mb09260_data,
                                            model.PSPL_Phot_Par_Param1,
                                            outputfiles_basename=mb09260_dir)

    smy_09260 = fitter_09260.load_mnest_summary()

    fitter_10364 = model_fitter.PSPL_Solver(mb10364_data,
                                            model.PSPL_Phot_Par_Param1,
                                            outputfiles_basename=mb10364_dir)

    smy_10364 = fitter_10364.load_mnest_summary()

    fitter_110037 = model_fitter.PSPL_Solver(ob110037_data,
                                              model.PSPL_Phot_Par_Param1,
                                              outputfiles_basename=ob110037_dir)

    smy_110037 = fitter_110037.load_mnest_summary()

    fitter_110310 = model_fitter.PSPL_Solver(ob110310_data,
                                             model.PSPL_Phot_Par_Param1,
                                             outputfiles_basename=ob110310_dir)

    smy_110310 = fitter_110310.load_mnest_summary()

    fitter_110462 = model_fitter.PSPL_Solver(ob110462_data,
                                             model.PSPL_Phot_Par_Param1,
                                             outputfiles_basename=ob110462_dir)
    smy_110462 = fitter_110462.load_mnest_summary()


    # Maximum likelihood vals                                                                                         
    smy_list = [smy_09260, smy_10364, smy_110037, smy_110310, smy_110462]

    smy_name = ['MB09260','MB10364', 'OB110037', 'OB110310', 'OB110462']
    maxl = {}

    for ss, smy in enumerate(smy_list):
        print(smy_name[ss])
        print('tE : ', smy['MaxLike_tE'][0])
        print('piE : ', np.hypot(smy['MaxLike_piE_E'][0], smy['MaxLike_piE_N'][0]))
        maxl[smy_name[ss]] = {'tE' : smy['MaxLike_tE'][0],
                              'piE' : np.hypot(smy['MaxLike_piE_E'][0], smy['MaxLike_piE_N'][0])}

    span = 0.999999426697
    smooth = 0.02
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

    # Set defaults.
    hist2d_kwargs['alpha'] = hist2d_kwargs.get('alpha', 0.2)
    hist2d_kwargs['levels'] = hist2d_kwargs.get('levels', quantiles_2d)

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
    fig = plt.figure(1)
    plt.clf()
    fig.set_size_inches(6.0, 6.0, forward=True)
    axes = fig.gca()
    plt.subplots_adjust(bottom=0.15, left=0.2)

    sx = smooth
    sy = smooth

    hist2d_kwargs['fill_contours'] = hist2d_kwargs.get('fill_contours', False)
    hist2d_kwargs['plot_contours'] = hist2d_kwargs.get('plot_contours', True)
    
    model_fitter.contour2d_alpha(data['tE'], data['piE'], span=[span, span], quantiles_2d=quantiles_2d,
                                 ax=axes, smooth=[sy, sx], color='red',
                                 **hist2d_kwargs, plot_density=False)
    axes.text(430, 0.025, 'MB190284', color='red')    

    axes.plot(maxl['MB09260']['tE'], maxl['MB09260']['piE'], 
               color='brown', marker = 'X', ms = 8, alpha=0.8)
    axes.plot(maxl['MB10364']['tE'], maxl['MB10364']['piE'], 
               color='brown', marker = 'X', ms = 8, alpha=0.8)
    axes.plot(maxl['OB110037']['tE'], maxl['OB110037']['piE'], 
               color='brown', marker='X', ms = 8, alpha=0.8)
    axes.plot(maxl['OB110310']['tE'], maxl['OB110310']['piE'], 
               color='brown', marker = 'X', ms = 8, alpha=0.8)
    axes.plot(maxl['OB110462']['tE'], maxl['OB110462']['piE'], 
               color='brown', marker='X', ms = 8, alpha=0.8)
    axes.text(12, 0.35, ' Cycle 17-24 \nTargets (Sahu)', color='brown')

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
                 alpha = 1.0, marker = '.', s = 25, 
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
                 alpha = 1.0, marker = '.', s = 25, 
                 label = 'BH', color = 'dimgray')

    axes.set_xlim(10, 2000)
    axes.set_ylim(0.005, 0.8)
    axes.set_xlabel('$t_E$ (days)')
    axes.set_ylabel('$\pi_E$')
    axes.set_xscale('log')
    axes.set_yscale('log')
    axes.legend(loc=1, markerscale=2)
    plt.savefig('piE_tE_phot_only_MB190284.png')

    return


def piE_tE_deltac_hst():
#    span=0.999999426697
#    smooth=0.02
#    quantiles_2d=None
#    hist2d_kwargs=None
#    labels=None
#    label_kwargs=None
#    show_titles=False 
#    title_fmt=".2f" 
#    title_kwargs=None
#    """
#    !!! NOTE: CHOICE OF THE quantiles_2d HAS A LARGE EFFECT 
#    ON THE WAY THIS PLOT LOOKS !!!
#    Plot piE-tE 2D posteriors from OGLE photometry only fits.
#    Also plot PopSyCLE simulations simultaneously.
#    """
#    # Initialize values.
#    if label_kwargs is None:
#        label_kwargs = dict()
#    if title_kwargs is None:
#        title_kwargs = dict()
#    if hist2d_kwargs is None:
#        hist2d_kwargs = dict()
#
#    # Set defaults.
#    hist2d_kwargs['alpha'] = hist2d_kwargs.get('alpha', 0.2)
#    hist2d_kwargs['levels'] = hist2d_kwargs.get('levels', quantiles_2d)

    mb09260_yaml = open(mb09260_dir + 'params.yaml').read() 
    params_09260 = yaml.safe_load(mb09260_yaml)
    mb10364_yaml = open(mb10364_dir + 'params.yaml').read() 
    params_10364 = yaml.safe_load(mb10364_yaml)
    ob110037_yaml = open(ob110037_dir + 'params.yaml').read() 
    params_110037 = yaml.safe_load(ob110037_yaml)
    ob110310_yaml = open(ob110310_dir + 'params.yaml').read() 
    params_110310 = yaml.safe_load(ob110310_yaml)
    ob110462_yaml = open(ob110462_dir + 'params.yaml').read() 
    params_110462 = yaml.safe_load(ob110462_yaml)

    # MB09260 fit results.
    fitter_09260 = model_fitter.PSPL_Solver(mb09260_data,
                                            model.PSPL_Phot_Par_Param1,
                                            outputfiles_basename=mb09260_dir)

    results_09260 = fitter_09260.load_mnest_results_for_dynesty()
    smy_09260 = fitter_09260.load_mnest_summary()

    # MB10364 fit results.
    fitter_10364 = model_fitter.PSPL_Solver(mb10364_data,
                                            model.PSPL_Phot_Par_Param1,
                                            outputfiles_basename=mb10364_dir)

    results_10364 = fitter_10364.load_mnest_results_for_dynesty()
    smy_10364 = fitter_10364.load_mnest_summary()

    # OB110037 fit results.
    fitter_110037 = model_fitter.PSPL_Solver(ob110037_data,
                                              model.PSPL_Phot_Par_Param1,
                                              outputfiles_basename=ob110037_dir)

    results_110037 = fitter_110037.load_mnest_results_for_dynesty()
    smy_110037 = fitter_110037.load_mnest_summary()

    # OB110310 fit results.
    fitter_110310 = model_fitter.PSPL_Solver(ob110310_data,
                                             model.PSPL_Phot_Par_Param1,
                                             outputfiles_basename=ob110310_dir)

    results_110310 = fitter_110310.load_mnest_results_for_dynesty()
    smy_110310 = fitter_110310.load_mnest_summary()

    # OB110462 fit results.
    fitter_110462 = model_fitter.PSPL_Solver(ob110462_data,
                                             model.PSPL_Phot_Par_Param1,
                                             outputfiles_basename=ob110462_dir)

    results_110462 = fitter_110462.load_mnest_results_for_dynesty()
    smy_110462 = fitter_110462.load_mnest_summary()

#    # Extract weighted samples.
#    samples_09260 = results_09260['samples']
#    samples_10364 = results_10364['samples']
#    samples_110037 = results_110037['samples']
#    samples_110310 = results_110310['samples']
#    samples_110462 = results_110462['samples']
#
#    try:
#        weights_09260 = np.exp(results_09260['logwt'] - results_09260['logz'][-1])
#        weights_10364 = np.exp(results_10364['logwt'] - results_10364['logz'][-1])
#        weights_110037 = np.exp(results_110037['logwt'] - results_110037['logz'][-1])
#        weights_110310 = np.exp(results_110310['logwt'] - results_110310['logz'][-1])
#        weights_110462 = np.exp(results_110462['logwt'] - results_110462['logz'][-1])
#    except:
#        weights_09260 = results_09260['weights']
#        weights_10364 = results_10364['weights']
#        weights_110037 = results_110037['weights']
#        weights_110310 = results_110310['weights']
#        weights_110462 = results_110462['weights']
#
#    # Deal with 1D results. A number of extra catches are also here
#    # in case users are trying to plot other results besides the `Results`
#    # instance generated by `dynesty`.
#    samples_09260 = np.atleast_1d(samples_09260)
#    if len(samples_09260.shape) == 1:
#        samples_09260 = np.atleast_2d(samples_09260)
#    else:
#        assert len(samples_09260.shape) == 2, "Samples must be 1- or 2-D."
#        samples_09260 = samples_09260.T
#    assert samples_09260.shape[0] <= samples_09260.shape[1], "There are more " \
#                                                 "dimensions than samples!"
#    
#    samples_10364 = np.atleast_1d(samples_10364)
#    if len(samples_10364.shape) == 1:
#        samples_10364 = np.atleast_2d(samples_10364)
#    else:
#        assert len(samples_10364.shape) == 2, "Samples must be 1- or 2-D."
#        samples_10364 = samples_10364.T
#    assert samples_10364.shape[0] <= samples_10364.shape[1], "There are more " \
#                                                 "dimensions than samples!"
#
#    samples_110037 = np.atleast_1d(samples_110037)
#    if len(samples_110037.shape) == 1:
#        samples_110037 = np.atleast_2d(samples_110037)
#    else:
#        assert len(samples_110037.shape) == 2, "Samples must be 1- or 2-D."
#        samples_110037 = samples_110037.T
#    assert samples_110037.shape[0] <= samples_110037.shape[1], "There are more " \
#                                                 "dimensions than samples!"
#
#    samples_110310 = np.atleast_1d(samples_110310)
#    if len(samples_110310.shape) == 1:
#        samples_110310 = np.atleast_2d(samples_110310)
#    else:
#        assert len(samples_110310.shape) == 2, "Samples must be 1- or 2-D."
#        samples_110310 = samples_110310.T
#    assert samples_110310.shape[0] <= samples_110310.shape[1], "There are more " \
#                                                 "dimensions than samples!"
#
#    samples_110462 = np.atleast_1d(samples_110462)
#    if len(samples_110462.shape) == 1:
#        samples_110462 = np.atleast_2d(samples_110462)
#    else:
#        assert len(samples_110462.shape) == 2, "Samples must be 1- or 2-D."
#        samples_110462 = samples_110462.T
#    assert samples_110462.shape[0] <= samples_110462.shape[1], "There are more " \
#                                                 "dimensions than samples!"

    # Maximum likelihood vals                                                                                         
    smy_list = [smy_09260, smy_10364, smy_110037, smy_110310, smy_110462]

    smy_name = ['MB09260','MB10364', 'OB110037', 'OB110310', 'OB110462']
    maxl = {}

    for ss, smy in enumerate(smy_list):
        print(smy_name[ss])
        print('tE : ', smy['MaxLike_tE'][0])
        print('piE : ', np.hypot(smy['MaxLike_piE_E'][0], smy['MaxLike_piE_N'][0]))
        maxl[smy_name[ss]] = {'tE' : smy['MaxLike_tE'][0],
                              'piE' : np.hypot(smy['MaxLike_piE_E'][0], smy['MaxLike_piE_N'][0])}

    #############
    # The observations. 
    # Finished targets are phot + astrom solutions
    # Ongoing targets are phot parallax solutions 
    # (global MEDIAN +/- 1 sigma)
    #############
    # Plot the piE-tE 2D posteriors.
    # tE = 2; piEE,N = 3, 4 
    fig, ax = plt.subplots(1, 2, figsize=(14,6), sharey=True,
                           gridspec_kw={'width_ratios': [1, 1.4]})
    plt.subplots_adjust(left=0.1, bottom=0.15, wspace=0.1)

#    tE_09260 = samples_09260[2]
#    tE_10364 = samples_10364[2]
#    tE_110037 = samples_110037[2]
#    tE_110310 = samples_110310[2]
#    tE_110462 = samples_110462[2]
#
#    piE_09260 = np.hypot(samples_09260[3], samples_09260[4])
#    piE_10364 = np.hypot(samples_10364[3], samples_10364[4])
#    piE_110037 = np.hypot(samples_110037[3], samples_110037[4])
#    piE_110310 = np.hypot(samples_110310[3], samples_110310[4])
#    piE_110462 = np.hypot(samples_110462[3], samples_110462[4])

#    sx = smooth
#    sy = smooth

#    hist2d_kwargs['fill_contours'] = hist2d_kwargs.get('fill_contours',
#                                                       False)
#    hist2d_kwargs['plot_contours'] = hist2d_kwargs.get('plot_contours',
#                                                       True)

#    model_fitter.contour2d_alpha(tE_09260, piE_09260, span=[span, span], quantiles_2d=quantiles_2d,
#                                 weights=weights_09260, ax=ax[1], smooth=[sy, sx], color='magenta',
#                                 **hist2d_kwargs, plot_density=False)
    ax[1].plot(maxl['MB09260']['tE'], maxl['MB09260']['piE'], 
               color='magenta', marker = 's', ms = 10, alpha=0.8)
    ax[1].plot(maxl['MB10364']['tE'], maxl['MB10364']['piE'], 
               color='darkgreen', marker = 's', ms = 10, alpha=0.8)
    ax[1].plot(maxl['OB110037']['tE'], maxl['OB110037']['piE'], 
               color='springgreen', marker='s', ms = 10, alpha=0.8)
#    model_fitter.contour2d_alpha(tE_110310, piE_110310, span=[span, span], quantiles_2d=quantiles_2d,
#                                 weights=weights_110310, ax=ax[1], smooth=[sy, sx], color='orangered', 
#                                 **hist2d_kwargs, plot_density=False)
    ax[1].plot(maxl['OB110310']['tE'], maxl['OB110310']['piE'], 
               color='orangered', marker = 's', ms = 10, alpha=0.8)
    ax[1].plot(maxl['OB110462']['tE'], maxl['OB110462']['piE'], 
               color='blueviolet', marker='s', ms = 10, alpha=0.8)

    ax[1].plot(0.01, 100, color='magenta', label='MB09260')
    ax[1].plot(0.01, 100, color='darkgreen', label='MB10364')
    ax[1].plot(0.01, 100, color='springgreen', label='OB110037')
    ax[1].plot(0.01, 100, color='orangered', label='OB110310')
    ax[1].plot(0.01, 100, color = 'blueviolet', label='OB110642')
    ax[1].set_xlim(10, 1000)
    ax[1].set_ylim(0.01, 1)
    ax[1].set_xscale('log')
    ax[1].set_yscale('log')

def plot_gaia():
    raL_str = '18:05:55.06'
    decL_str = '-30:20:12.94'

    search_rad = 60 # arcsec
    
    if not os.path.exists('gaia3.fits'):
        gaia = analysis.query_gaia(raL_str, decL_str, search_radius=search_rad, table_name='gaiaedr3')
        gaia['designation'] = gaia['designation'].astype(str)
        gaia.write('gaia3.fits', overwrite=True)
    else:
        gaia = Table.read('gaia3.fits')

    target_coords = SkyCoord(raL_str, decL_str, unit=(u.hourangle, u.deg), frame='icrs')
    ra = target_coords.ra.degree     # in decimal degrees
    dec = target_coords.dec.degree   # in decimal degrees
    
    cos_dec = np.cos(np.radians(dec))
    x = (gaia['ra'] - ra) * cos_dec * 3600.0   # arcsec
    y = (gaia['dec'] - dec) * 3600.0           # arcsec
    xe = gaia['ra_error'] * cos_dec / 1e3      # arcsec
    ye = gaia['dec_error'] / 1e3               # arcsec

    gaia['dra*'] = x
    gaia['ddec'] = y
    gaia['dra*_err'] = xe
    gaia['ddec_err'] = ye
    gaia['m'] = gaia['phot_g_mean_mag']

    tdx = np.argmin(np.hypot(gaia['dra*'], gaia['ddec']))

    print(gaia[tdx])

    plt.figure(1)
    plt.clf()
    plt.semilogy(gaia['m'], gaia['pmra_error'], 'r.', alpha=0.5)
    plt.plot(gaia['m'], gaia['pmdec_error'], 'b.', alpha=0.5)
    plt.plot(gaia['m'][tdx], gaia['pmra_error'][tdx], 'rs', ms=10)
    plt.plot(gaia['m'][tdx], gaia['pmdec_error'][tdx], 'bs', ms=10)
    plt.xlabel('Gaia G (mag)')
    plt.ylabel('PM Error (mas/yr)')

    plt.figure(2)
    plt.clf()
    plt.semilogy(gaia['m'], gaia['astrometric_excess_noise'], 'k.')
    plt.plot(gaia['m'][tdx], gaia['astrometric_excess_noise'][tdx], 'rs', ms=10)
    plt.xlabel('Gaia G (mag)')
    plt.ylabel('Astrometric Excess Noise')

    plt.figure(3)
    plt.clf()
    plt.plot(gaia['pmra'], gaia['pmdec'], 'k.')
    plt.plot(gaia['pmra'][tdx], gaia['pmdec'][tdx], 'rs', ms=10)
    plt.xlabel('pmra (mas/yr)')
    plt.ylabel('pmdec (mas/yr)')
    plt.gca().invert_xaxis()
    plt.axis('equal')

    plt.figure(4)
    plt.clf()
    plt.plot(gaia['bp_rp'], gaia['m'], 'k.')
    plt.plot(gaia['bp_rp'][tdx], gaia['m'][tdx], 'rs', ms=10)
    plt.gca().invert_yaxis()
    plt.xlabel('BP - RP (mag)')
    plt.ylabel('Gaia G (mag)')
    plt.xlim(0, 3)
    plt.ylim(20, 14)

    return


