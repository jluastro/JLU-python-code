import numpy as np
import pylab as plt
import pdb
import math
import os
from microlens.jlu import munge
from microlens.jlu import model_fitter, model
import shutil, os, sys
import scipy
import scipy.stats
from scipy.stats import chi2
from astropy.table import Table
from jlu.util import fileUtil
from astropy.table import Table, Column, vstack
from astropy.io import fits
from microlens.jlu import model_fitter, multinest_utils, multinest_plot, model
from jlu.papers import lu_2019_lens
from flystar import plots
import copy
from astropy.time import Time
from jlu.util import datetimeUtil as dtUtil
from datetime import datetime as dt
import yaml

def table_phot_astrom():
    """
    I was in a rush and didn't have time to make a proper table, so this just 
    prints out the values, and you manually type it into LaTeX itself. 
    """
    #####
    # Comment out the target you aren't working on
    #####
    ob110462_data = '/u/jlu/work/microlens/OB110462/a_2021_03_29/model_fits/hstf814w_phot_ast/base_p/p0_'
    mod_fit, data = lu_2019_lens.get_data_and_fitter(ob110462_data)

    mod_all = mod_fit.get_best_fit_modes_model(def_best = 'maxL')
    mod = mod_all[0]

    tab = mod_fit.load_mnest_results()
    tab['piE'] = np.hypot(tab['piE_E'], tab['piE_N'])
    tab['muRel'] = np.hypot(tab['muRel_E'], tab['muRel_N'])

    tab_best = {}
    tab_maxl = {}
    med_errors = {}
    sumweights = np.sum(tab['weights'])
    weights = tab['weights'] / sumweights
    
    sig1 = 0.682689
    sig2 = 0.9545
    sig3 = 0.9973
    sig1_lo = (1. - sig1) / 2.
    sig2_lo = (1. - sig2) / 2.
    sig3_lo = (1. - sig3) / 2.
    sig1_hi = 1. - sig1_lo
    sig2_hi = 1. - sig2_lo
    sig3_hi = 1. - sig3_lo

    #####
    # Comment out the target you aren't working on
    #####
    # For MB10364
    params = ['tE', 'thetaE', 'piE', 'muRel', 'mL', 'mag_src1', 'piL']

    maxl_idx = tab['logLike'].argmax()

    for n in params:
        tab_maxl[n] = tab[n][maxl_idx]
        # Calculate median, 1 sigma lo, and 1 sigma hi credible interval.                                                          
        tmp = model_fitter.weighted_quantile(tab[n], [0.5, sig1_lo, sig1_hi],
                                             sample_weight=weights)
        tab_best[n] = tmp[0]
        
        # Switch from values to errors.                                                                                            
#        err_lo = tmp[0] - tmp[1]
#        err_hi = tmp[2] - tmp[0]
#        med_errors[n] = np.array([err_lo, err_hi])

        err_lo = tab_maxl[n] - tmp[1]
        err_hi = tmp[2] - tab_maxl[n]
        med_errors[n] = np.array([err_lo, err_hi])

#    print(tab_maxl)
#    print(med_errors)
        
    for param in params:
        print(param + ' : ' +  str(tab_maxl[param]) + ', ' +  str(med_errors[param]))
    
    return


def ob110462_ast_v_time():
    target='OB110462'
    t_obs_prop=['2022-05-01', '2022-10-01']

    # Get the model
    ob110462_data = '/u/jlu/work/microlens/OB110462/a_2021_03_29/model_fits/hstf814w_phot_ast/base_p/p0_'
    mod_fit, data = lu_2019_lens.get_data_and_fitter(ob110462_data)

    keys = ['t_phot1', 'mag1', 'mag_err1',
            't_ast1', 'xpos1', 'ypos1', 'xpos_err1', 'ypos_err1']
    
    for key in keys:
        data[key] = np.delete(data[key], 4)

    mod_all = mod_fit.get_best_fit_modes_model(def_best = 'maxL')
    mod = mod_all[0]

    # Sample N random draws from the posterior
    ndraw = 100
    res = mod_fit.load_mnest_results()

    # Indices of the posterior we want
    pdx = np.random.choice(len(res), ndraw, replace=True, p=res['weights'])

    # Sample time
    tmax = np.max(data['t_phot1']) + 90.0
    tmax += 2500
    t_mod_ast = np.arange(data['t_ast1'].min() - 180.0, tmax, 0.1)
    t_mod_pho = np.arange(data['t_phot1'].min(), tmax, 5)

    days_per_year = 365.25
    dt_mod_ast = (t_mod_ast - mod.t0)/days_per_year
    dt_data_ast = (data['t_ast1'] - mod.t0)/days_per_year 

    # Get the linear motion curves for the source (NO parallax)
    p_unlens_mod = mod.xS0 + np.outer(dt_mod_ast, mod.muS) * 1e-3
    p_unlens_mod_at_ast = mod.xS0 + np.outer(dt_data_ast, mod.muS) * 1e-3

    # Get the lensed motion curves for the source
    p_lens_mod = mod.get_astrometry(t_mod_ast)
    p_lens_mod_at_ast = mod.get_astrometry(data['t_ast1'])

    x = (data['xpos1'] - p_unlens_mod_at_ast[:,0]) * -1e3
    xe = data['xpos_err1'] * 1e3
    y = (data['ypos1'] - p_unlens_mod_at_ast[:,1]) * 1e3
    ye = data['ypos_err1'] * 1e3

    xmod = (p_lens_mod[:,0] - p_unlens_mod[:,0]) * -1e3 
    ymod = (p_lens_mod[:,1] - p_unlens_mod[:,1]) * 1e3

    # Convert to decimal dates
    t_ast_dec = Time(data['t_ast1'], format='mjd', scale='utc')
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

    ###
    # X VS TIME
    ###

    plt.figure(1, figsize=(6,6))
    plt.clf()
    plt.errorbar(t_ast_dec, x, yerr=xe, fmt='r.', alpha=1, zorder = 1000, label='Proposed')
    plt.scatter(t_mod_ast_dec, xmod, s = 0.1, color='black')
    plt.title(target)
    plt.xlabel('Time (Year)')
    plt.ylabel(r'$\Delta \alpha^*$ (mas)')
    for ii in range(len(pdx)):
        # Plot the posterior draws
        pdraw = mod_fit.get_model(res[ii])

        # Get the lensed motion curves for the source
        p_lens_pdraw = pdraw.get_astrometry(t_mod_ast)
        p_lens_pdraw_at_ast = pdraw.get_astrometry(data['t_ast1'])

        xpdraw = (p_lens_pdraw[:,0] - p_unlens_mod[:,0]) * -1e3
        ypdraw = (p_lens_pdraw[:,1] - p_unlens_mod[:,1]) * 1e3

        plt.scatter(t_mod_ast_dec, xpdraw, s = 0.01, alpha=0.02, color='lightgray')

    plt.xticks([2011, 2014, 2017, 2020, 2023])
    for t_obs in t_obs_prop_dec:
        plt.axvline(x=t_obs, color='red', ls=':')
    plt.legend(loc=1)
    plt.title(target)
    plt.xlim(2010.75, 2023.25)
    plt.savefig(target + '_deltac_RA_vs_time.png')
    plt.show()

    ###
    # Y VS TIME
    ###
    plt.figure(2, figsize=(6,6))
    plt.clf()
    plt.errorbar(t_ast_dec, y, yerr=ye, fmt='r.', alpha=1, zorder = 1000, label='Data')
    plt.scatter(t_mod_ast_dec, ymod, s = 0.1, color='black')
    plt.title(target)
    plt.xlabel('Time (Year)')
    plt.ylabel('$\Delta \delta$ (mas)')
    for ii in range(len(pdx)):
        # Plot the posterior draws
        pdraw = mod_fit.get_model(res[ii])

        # Get the lensed motion curves for the source
        p_lens_pdraw = pdraw.get_astrometry(t_mod_ast)
        p_lens_pdraw_at_ast = pdraw.get_astrometry(data['t_ast1'])

        xpdraw = (p_lens_pdraw[:,0] - p_unlens_mod[:,0]) * -1e3
        ypdraw = (p_lens_pdraw[:,1] - p_unlens_mod[:,1]) * 1e3

        plt.scatter(t_mod_ast_dec, ypdraw, s = 0.01, alpha=0.02, color='lightgray')
        
    plt.xticks([2011, 2014, 2017, 2020, 2023])
    for t_obs in t_obs_prop_dec:
        plt.axvline(x=t_obs, color='red', ls=':')
    plt.legend(loc=1)
    plt.xlim(2010.75, 2023.25)
    plt.title(target)
    plt.savefig(target + '_deltac_Dec_vs_time.png')
    plt.show()

    # Single plot
    plt.close(3)
    fig, ax= plt.subplots(nrows=2, ncols=1, num=3, figsize=(6,6), sharex=True)
    plt.subplots_adjust(hspace=0.05, top=0.98)
    ax[0].errorbar(t_ast_dec, x, yerr=xe, fmt='r.', alpha=1, zorder = 1000, label='Data')
    ax[0].scatter(t_mod_ast_dec, xmod, s = 0.1, color='black')
    ax[0].set_ylabel(r'$\Delta \alpha^*$ (mas)')
    for ii in range(len(pdx)):
        # Plot the posterior draws
        pdraw = mod_fit.get_model(res[ii])

        # Get the lensed motion curves for the source
        p_lens_pdraw = pdraw.get_astrometry(t_mod_ast)
        p_lens_pdraw_at_ast = pdraw.get_astrometry(data['t_ast1'])

        xpdraw = (p_lens_pdraw[:,0] - p_unlens_mod[:,0]) * -1e3
        ypdraw = (p_lens_pdraw[:,1] - p_unlens_mod[:,1]) * 1e3

        ax[0].scatter(t_mod_ast_dec, xpdraw, s = 0.01, alpha=0.02, color='lightgray')

    for t_obs in t_obs_prop_dec:
        ax[0].axvline(x=t_obs, color='red', ls=':')
    ax[0].axvline(x=2000, color='red', ls=':', label='Proposed')
    ax[0].legend(loc=2)

    ax[1].errorbar(t_ast_dec, y, yerr=ye, fmt='r.', alpha=1, zorder = 1000, label='Data')
    ax[1].scatter(t_mod_ast_dec, ymod, s = 0.1, color='black')
    ax[1].set_xlabel('Time (Year)')
    ax[1].set_ylabel('$\Delta \delta$ (mas)')
    for ii in range(len(pdx)):
        # Plot the posterior draws
        pdraw = mod_fit.get_model(res[ii])

        # Get the lensed motion curves for the source
        p_lens_pdraw = pdraw.get_astrometry(t_mod_ast)
        p_lens_pdraw_at_ast = pdraw.get_astrometry(data['t_ast1'])

        xpdraw = (p_lens_pdraw[:,0] - p_unlens_mod[:,0]) * -1e3
        ypdraw = (p_lens_pdraw[:,1] - p_unlens_mod[:,1]) * 1e3

        ax[1].scatter(t_mod_ast_dec, ypdraw, s = 0.01, alpha=0.02, color='lightgray')
        
    ax[1].set_xticks([2011, 2014, 2017, 2020, 2023])
    for t_obs in t_obs_prop_dec:
        ax[1].axvline(x=t_obs, color='red', ls=':')
    ax[1].set_xlim(2010.75, 2023.25)
    plt.savefig(target + '_deltac_RA_Dec_vs_time.png')
    plt.show()


def plot_targets():
    # OB110462
    tdir = 'OB110462/a_2020_07_30/ob150462_astrom_p4aerr_2020_09_15_snipe.fits'
    starName = 'OB110462'
    plot_astrom(starName, tdir)

    # OB110310
    tdir = 'OB110310/a_2020_07_26/ob110310_astrom_p4aerr_2020_09_15_snipe.fits'
    starName = 'OB110310'
    plot_astrom(starName, tdir, ylim=[-1, 1])

    # OB110037
    tdir = 'OB110037/a_2020_08_26/ob110037_astrom_p5aerr_2020_09_15_snipe.fits'
    starName = 'OB110037'
    plot_astrom(starName, tdir, ylim=[-1, 1])

    # MB10364
    tdir = 'MB10364/a_2020_08_08/mb10364_astrom_p4aerr_2020_09_18_snipe.fits'
    starName = 'MB10364'
    plot_astrom(starName, tdir, ylim=[-1, 1], xlim=[0, 60])

    # MB09260
    tdir = 'MB09260/a_2020_08_07/mb09260_astrom_p4aerr_2020_09_18_snipe.fits'
    starName = 'MB09260'
    plot_astrom(starName, tdir, ylim=[-1, 1])


def plot_astrom(starName, tdir, xlim=[0,40], ylim=[-2,2]):
    tab = Table.read(mdir + tdir)
    starName = starName

    ##################################
    # Individual astrometry plot
    ##################################
    plt.close('all')
    plt.figure(1, figsize=(10,6))
    plt.subplots_adjust(wspace=0.4, left=0.15)
    plt.suptitle(starName)

    names = tab['name']
    mag = tab['m0']
    x = tab['x0']
    y = tab['y0']
    r = np.hypot(x, y)
    
    ii = np.where(tab['name'] == starName)[0][0]
    
    # Ignore the NaNs
    fnd = np.argwhere(~np.isnan(tab['xe'][ii,:]))

    time = tab['t'][ii, fnd]
    dtime = time.data % 1 
    x = tab['x'][ii, fnd]
    y = tab['y'][ii, fnd]
    m = tab['m'][ii, fnd]
    
    xerr = tab['xe'][ii, fnd]
    yerr = tab['ye'][ii, fnd]
    merr = tab['me'][ii, fnd]
    
    dt = tab['t'][ii, fnd] - tab['t0'][ii]
    fitLineX = tab['x0'][ii] + (tab['vx'][ii] * dt)
    fitLineY = tab['y0'][ii] + (tab['vy'][ii] * dt)
    
    fitSigX = np.hypot(tab['x0e'][ii], tab['vxe'][ii]*dt)
    fitSigY = np.hypot(tab['y0e'][ii], tab['vye'][ii]*dt)
    
    fitLineM = np.repeat(tab['m0'][ii], len(dt)).reshape(len(dt),1)
    fitSigM = np.repeat(tab['m0e'][ii], len(dt)).reshape(len(dt),1)
    
    diffX = x - fitLineX
    diffY = y - fitLineY
    diffM = m - fitLineM
    diff = np.hypot(diffX, diffY)
    rerr = np.sqrt((diffX*xerr)**2 + (diffY*yerr)**2) / diff
    sigX = diffX / xerr
    sigY = diffY / yerr
    sigM = diffM / merr
    sig = diff / rerr
    
    # Calculate chi^2 metrics
    chi2_x_targ = np.sum(sigX**2)
    chi2_y_targ = np.sum(sigY**2)
    chi2_m_targ = np.sum(sigM**2)
    
    dof = len(x) - 2
    dofM = len(m) - 1
    
    chi2_red_x_targ = chi2_x_targ / dof
    chi2_red_y_targ = chi2_y_targ / dof
    chi2_red_m_targ = chi2_m_targ / dofM
        
    print( 'Star:        ', starName )
    print( '\tX Chi^2 = %5.2f (%6.2f for %2d dof)' % 
           (chi2_red_x_targ, chi2_x_targ, dof))
    print( '\tY Chi^2 = %5.2f (%6.2f for %2d dof)' % 
           (chi2_red_y_targ, chi2_y_targ, dof))
    print( '\tM Chi^2 = %5.2f (%6.2f for %2d dof)' % 
           (chi2_red_m_targ, chi2_m_targ, dofM))
    
    tmin = time.min()
    tmax = time.max()
    
    dateTicLoc = plt.MultipleLocator(3)
    dateTicRng = [np.floor(tmin), np.ceil(tmax)]
    dateTics = np.arange(np.floor(tmin), np.ceil(tmax)+0.1)
    DateTicsLabel = dateTics
    
    # See if we are using MJD instead.
    if time[0] > 50000:
        print('MJD')
        dateTicLoc = plt.MultipleLocator(1000)
        t0 = int(np.round(np.min(time), 50))
        tO = int(np.round(np.max(time), 50))
        dateTicRng = [tmin-200, tmax+200]
        dateTics = np.arange(dateTicRng[0], dateTicRng[-1]+500, 1000)
        DateTicsLabel = dateTics
        
        
    maxErr = np.array([(diffX-xerr)*1e3, (diffX+xerr)*1e3,
                       (diffY-yerr)*1e3, (diffY+yerr)*1e3]).max()
    maxErrM = np.array([(diffM - merr), (diffM + merr)]).max()
    
    if maxErr > 2:
        maxErr = 2.0
    if maxErrM > 1.0:
        maxErr = 1.0
    resTicRng = [-1.1*maxErr, 1.1*maxErr]
    resTicRngM = [-1.1*maxErrM, 1.1*maxErrM]

    from matplotlib.ticker import FormatStrFormatter
    fmtX = FormatStrFormatter('%5i')
    fmtY = FormatStrFormatter('%6.3f')
    fmtM = FormatStrFormatter('%5.2f')
    fontsize1 = 16
    
    ##########
    # X residuals vs time
    ##########
    paxes = plt.subplot(1, 2, 1)
    plt.plot(time, np.zeros(len(time)), 'b-')
    plt.plot(time,  fitSigX*1e3, 'b--')
    plt.plot(time, -fitSigX*1e3, 'b--')
    plt.errorbar(time, (x - fitLineX)*1e3, yerr=xerr.reshape(len(xerr),)*1e3, fmt='k.')
    plt.axis(dateTicRng + resTicRng)
    plt.xticks(fontsize=fontsize1)
    plt.xlabel('Date (yrs)', fontsize=fontsize1)
    if time[0] > 50000:
        plt.xlabel('Date (MJD)', fontsize=fontsize1)
    plt.ylabel('X Residuals (mas)', fontsize=fontsize1)
    paxes.xaxis.set_major_formatter(fmtX)
    paxes.tick_params(axis='both', which='major', labelsize=fontsize1)
    plt.ylim(ylim[0], ylim[-1])

    ##########
    # Y residuals vs time
    ##########
    paxes = plt.subplot(1, 2, 2)
    plt.plot(time, np.zeros(len(time)), 'b-')
    plt.plot(time,  fitSigY*1e3, 'b--')
    plt.plot(time, -fitSigY*1e3, 'b--')
    plt.errorbar(time, (y - fitLineY)*1e3, yerr=yerr.reshape(len(yerr),)*1e3, fmt='k.')
    plt.axis(dateTicRng + resTicRng)
    plt.xticks(fontsize=fontsize1)
    plt.xlabel('Date (yrs)', fontsize=fontsize1)
    if time[0] > 50000:
        plt.xlabel('Date (MJD)', fontsize=fontsize1)
    plt.ylabel('Y Residuals (mas)', fontsize=fontsize1)
    paxes.xaxis.set_major_formatter(fmtX)
    paxes.tick_params(axis='both', which='major', labelsize=fontsize1)
    plt.ylim(ylim[0], ylim[-1])

    plt.savefig(starName + '_astrom.png')
    plt.show()

    ##################################
    # chi2 distribution
    ##################################
    r0 = np.hypot(tab['x0'], tab['y0'])
    keep_idx = np.where((tab['m0'] < np.max(m) + 0.5) & 
                        (tab['m0'] > np.min(m) - 0.5) & 
                        (r0 < 30))[0]

    tab = tab[keep_idx]
    Ndetect = len(m)

    chi2_x_list = []
    chi2_y_list = []
    fnd_list = [] # Number of non-NaN error measurements                                                                                           
    for ii in range(len(tab['xe'])):
        # Ignore the NaNs                                                                                                                          
        fnd = np.argwhere(~np.isnan(tab['xe'][ii,:]))
#        fnd = np.where(tab['xe'][ii, :] > 0)[0]                                                                                                   
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

    x = np.array(chi2_x_list)
    y = np.array(chi2_y_list)
    fnd = np.array(fnd_list)

    idx = np.where(fnd == Ndetect)[0]
    # Fitting position and velocity... so subtract 2 to get Ndof                                                                                   
    Ndof = Ndetect - 2
    chi2_xaxis = np.linspace(0, 40, 100)

    plt.figure(2, figsize=(6,6))
    plt.clf()
    plt.hist(x[idx], bins=np.arange(400), histtype='step', label='X', density=True, 
             ls='--', color='tab:blue', lw=1.5)
    plt.hist(y[idx], bins=np.arange(400), histtype='step', label='Y', density=True, 
             color='tab:orange', lw=1.5)
    plt.plot(chi2_xaxis, chi2.pdf(chi2_xaxis, Ndof), 'r-', alpha=0.6)
#             label='$\chi^2$ ' + str(Ndof) + ' dof')
#    plt.title('$N_{epoch} = $' + str(Ndetect) + ', $N_{dof} = $' + str(Ndof))
    plt.axvline(x=chi2_x_targ, ls='--', color='tab:blue', lw=1.5)
    plt.axvline(x=chi2_y_targ, color='tab:orange', lw=1.5)
    plt.ylabel('PDF')
    plt.xlabel('$\chi^2$')
    plt.xlim(xlim[0], xlim[1])
    plt.legend()
    plt.savefig(starName + '_chi2.png')
    plt.show()

    return


def posteriors():
    mL = {}
    muRel = {}
    weights = {}

    ob110462_data = '/u/jlu/work/microlens/OB110462/a_2021_03_29/model_fits/hstf814w_phot_ast/base_p/p0_'
    fit_targ, dat_targ = lu_2019_lens.get_data_and_fitter(ob110462_data)
    
    res_targ = fit_targ.load_mnest_modes()
    smy_targ = fit_targ.load_mnest_summary()
    
    # Get rid of the global mode in the summary table.
    smy_targ = smy_targ[1:]

    # Find which solution has the max likelihood.
    mdx = smy_targ['maxlogL'].argmax()
    res_targ = res_targ[mdx]
    smy_targ = smy_targ[mdx]
    
    mL = res_targ['mL']
    muRel = np.hypot(res_targ['muRel_N'], res_targ['muRel_E'])
    weights = res_targ['weights']
    
    plt.figure(4, figsize=(6,4))
    plt.subplots_adjust(bottom=0.2)
    plt.clf()
    plt.hist(mL, weights=weights, bins=50, histtype='step', lw=2)
    plt.axvspan(2, 5, color='gray', alpha=0.3)
    plt.xlabel('Lens mass ($M_\odot$)')
    plt.ylabel('PDF')
    plt.xlim(xmin=0)
#    plt.savefig('mL_posteriors.png')
    plt.show()

    plt.figure(4, figsize=(6,4))
    plt.subplots_adjust(bottom=0.2)
    plt.clf()
    plt.hist(mL, weights=weights, bins=np.logspace(0, 2.5))
    plt.xlabel('Lens mass ($M_\odot$)')
    plt.ylabel('PDF')
    plt.xscale('log')
#    plt.xlim(xmin=0)
#    plt.savefig('mL_posteriors.png')
    plt.show()


def piE_tE():        
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
    contour_kwargs = None
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
    if contour_kwargs is None:
        contour_kwargs = dict()

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

    # Set defaults.
    hist2d_kwargs['alpha'] = hist2d_kwargs.get('alpha', 0.2)
    hist2d_kwargs['levels'] = hist2d_kwargs.get('levels', quantiles_2d)

    # No need for it to be a loop but too lazy to change old code.
    hst_targets = ['ob110462'] 
    tE = {}
    piE = {}
    theta_E = {}
    weights = {}

    # Get data for plotting.
    ob110462_data = '/u/jlu/work/microlens/OB110462/a_2021_03_29/model_fits/hstf814w_phot_ast/base_p/p0_'
#    ob110462_data = '/u/jlu/work/microlens/OB110462/a_2020_08_26/model_fits/all_phot_ast_merr/base_a/a0_'
#    ob110462_data = '/u/jlu/work/microlens/OB110462/a_2021_03_29/model_fits/ogle_hstf814w_gp/a0_'
#    ob110462_data = '/u/jlu/work/microlens/OB110462/a_2020_08_26/model_fits/ogle_phot_gp/c0_'
    fit_targ, dat_targ = lu_2019_lens.get_data_and_fitter(ob110462_data)
    
    res_targ = fit_targ.load_mnest_modes()
    smy_targ = fit_targ.load_mnest_summary()
    
    # Get rid of the global mode in the summary table.
    smy_targ = smy_targ[1:]

    # Find which solution has the max likelihood.
    mdx = smy_targ['maxlogL'].argmax()
    res_targ = res_targ[mdx]
    smy_targ = smy_targ[mdx]
    
    tE = res_targ['tE']
    piE = np.hypot(res_targ['piE_E'], res_targ['piE_N'])
    weights = res_targ['weights']
    theta_E = res_targ['thetaE']
    
    plt.close(1)
    fig = plt.figure(1, figsize=(6,6))
    plt.clf()
    axes = plt.gca()
    plt.subplots_adjust(bottom=0.15)
    
    sx = smooth
    sy = smooth
    
    hist2d_kwargs['fill_contours'] = hist2d_kwargs.get('fill_contours', False)
    hist2d_kwargs['plot_contours'] = hist2d_kwargs.get('plot_contours', True)

    model_fitter.contour2d_alpha(tE, piE, span=[span, span], quantiles_2d=quantiles_2d,
                                 weights=weights, ax=axes, smooth=[sy, sx], color='red',
                                 **hist2d_kwargs, plot_density=False, sigma_levels=[1,2,3])
        
    axes.text(100, 0.2,
              'OB110462', color='red')


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
    plt.savefig('piE_tE.png')
    plt.show()
    
    # Plot the deltac-piE 2D posteriors.
    plt.close(2)
    fig = plt.figure(2, figsize=(6,6))
    plt.clf()
    axes = plt.gca()
    plt.subplots_adjust(bottom=0.15)
    
    model_fitter.contour2d_alpha(theta_E/np.sqrt(8), piE, span=[span, span], quantiles_2d=quantiles_2d,
                                 weights=weights, ax=axes, smooth=[sy, sx], color='red',
                                 **hist2d_kwargs, plot_density=False, sigma_levels=[1,2,3])

#    axes.arrow(5e-1, 5e-2, -5e-1 + 5e-2, -5e-2 + 5e-1)
#    axes.arrow(5e-2, 5e-1, 5e-1 - 5e-2, 5e-2 - 5e-1)
    axes.annotate('', xy=(5e-1, 5e-2), xytext=(5e-2, 5e-1),
                 arrowprops=dict(facecolor='black', shrink=0.),)
    axes.text(0.08, 0.08, 'Mass', rotation=-45, fontsize=24)

#    axes.text(100, 0.1,
#              'OB110462', color='red')
            
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
                 c = 'black')

#    xarr = np.linspace(0.001, 4, 1000)
#    axes.fill_between(xarr, xarr*0.18, xarr*0.07, alpha=0.15, color='orange')
#    axes.text(0.17, 0.02, 'Mass Gap', rotation=45)

    # Trickery to make the legend darker
    axes.scatter(0.01, 100, 
                 alpha = 0.95, marker = '.', s = 25, 
                 label = 'Star', color = 'paleturquoise')
    axes.scatter(0.01, 100, 
                 alpha = 0.8, marker = '.', s = 25,
                 label = 'WD', color = 'aqua')
    axes.scatter(0.01, 100,
                 alpha = 0.8, marker = '.', s = 25, 
                 label = 'NS', color = 'blue')
    axes.scatter(0.01, 100,
                 alpha = 0.8, marker = '.', s = 25, 
                 label = 'BH', color = 'black')
#    axes.scatter(0.5045, 0.0876, marker='.', color='black') # 2 solar mass object
    
    axes.set_xlabel('$\delta_{c,max}$ (mas)')
    axes.set_ylabel('$\pi_E$')
    axes.set_xscale('log')
    axes.set_yscale('log')
    axes.set_xlim(0.02, 4)
    axes.set_ylim(0.005, 1)
    axes.set_aspect('equal')
    plt.legend(loc=1)
    plt.savefig('piE_deltac.png')
    plt.show()

