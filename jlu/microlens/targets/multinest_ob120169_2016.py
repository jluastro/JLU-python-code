import pylab as py
import numpy as np
from astropy.table import Table
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt
from scipy import interpolate
import scipy
import scipy.stats
import pymultinest
import math
import pdb
import os
import sys
import time
from jlu.microlens import model_fitter
from jlu.microlens import model
from astropy.table import Table

starttime = time.time()

data_dir = '/Users/jlu/work/microlens/OB120169/analysis_2016_09_14//analysis_ob120169_2016_09_14_a4_m22_w4_MC100'
phot_file = data_dir + '/../photometry/OB120169.dat'
ast_file = data_dir + '/points_d/OB120169_mjd.points'

def load_data():
    scale = 0.009952
    
    phot_tab = Table.read(phot_file, format='ascii')
    t_jd = phot_tab['col1']  # need to convert from JD to MJFD
    t_phot = t_jd - 2400000.5
    mag = phot_tab['col2']    # mag
    mag_e = phot_tab['col3']  # mag

    ast_tab = Table.read(ast_file, format='ascii')
    t_ast = ast_tab['col1']
    xpos = ast_tab['col2'] * scale * -1.0    # arcsec
    ypos = ast_tab['col3'] * scale     # arcsec
    xpos_e = ast_tab['col4'] * scale  # arcsec
    ypos_e = ast_tab['col5'] * scale  # arcsec

    dec = -35.37500
    ra = 267.4634

    data = {}
    data['t_phot'] = t_phot.data
    data['imag'] = mag.data
    data['imag_err'] = mag_e.data
    data['t_ast'] = t_ast.data
    data['xpos'] = xpos.data
    data['ypos'] = ypos.data
    data['xpos_err'] = xpos_e.data
    data['ypos_err'] = ypos_e.data
    data['raL'] = ra
    data['decL'] = dec


    return data

def run_pspl_parallax_fit():    
    data = load_data()

    # model_fitter.multinest_pspl_parallax(data,
    #                                      n_live_points=300,
    #                                      saveto='./mnest_pspl_par/',
    #                                      runcode='aa')

    model_fitter.plot_posteriors('mnest_pspl_par/', 'aa')

    best = model_fitter.get_best_fit('mnest_pspl_par/', 'aa')

    
    pspl_out = model.PSPL_parallax(data['raL'], data['decL'],
                                   best['mL'], best['t0'],
                                   np.array([best['xS0_E'], best['xS0_N']]),
                                   best['beta'], 
                                   np.array([best['muL_E'], best['muL_N']]),
                                   np.array([best['muS_E'], best['muS_N']]),
                                   best['dL'], best['dS'], best['imag_base'])

    t_out = np.arange(55000, 58000, 1)
    imag_out = pspl_out.get_photometry(t_out)
    pos_out = pspl_out.get_astrometry(t_out)

    imag_out_data = pspl_out.get_photometry(data['t_phot'])
    pos_out_data = pspl_out.get_astrometry(data['t_ast'])

    lnL_phot_out = pspl_out.likely_photometry(data['t_phot'], data['imag'], data['imag_err'])
    lnL_ast_out = pspl_out.likely_astrometry(data['t_ast'], data['xpos'], data['ypos'],
                                             data['xpos_err'], data['ypos_err'])
    lnL_out = lnL_phot_out.mean() + lnL_ast_out.mean()
         
    chi2_phot = (((data['imag'] - imag_out_data) / data['imag_err'])**2).sum()
    chi2_ast = (((data['xpos'] - pos_out_data[:, 0]) / data['xpos_err'])**2).sum()
    chi2_ast += (((data['ypos'] - pos_out_data[:, 1]) / data['ypos_err'])**2).sum()

    chi2_tot = chi2_phot + chi2_ast
    
    dof_phot = len(data['imag']) - 12
    dof_ast = (len(data['xpos']) * 2) - 12
    dof_tot = (len(data['imag']) + 2*len(data['xpos'])) - 12
    
    print 'lnL for output: ', lnL_out
    print 'Photometry chi^2 = {0:4.1f} (dof={1:4d})'.format(chi2_phot, dof_phot)
    print 'Astrometry chi^2 = {0:4.1f} (dof={1:4d})'.format(chi2_ast, dof_ast)
    print 'Total      chi^2 = {0:4.1f} (dof={1:4d})'.format(chi2_tot, dof_tot)

    outroot = 'mnest_pspl_par/plots/aa'
    
    ##########
    # Photometry vs. time
    ##########
    fig = plt.figure(1)
    plt.clf()
    f1 = fig.add_axes((0.2, 0.3, 0.75, 0.6))
    plt.errorbar(data['t_phot'], data['imag'], yerr=data['imag_err'], fmt='r.')
    plt.plot(t_out, imag_out, 'k-')
    plt.ylabel('I (mag)', fontsize=12)
    plt.title('Input Data and Output Model', fontsize=12)
    plt.xlim(data['t_phot'].min(), data['t_phot'].max())
    plt.gca().invert_yaxis()
    f1.set_xticklabels([])
    
    f2 = fig.add_axes((0.2, 0.15, 0.75, 0.15))
    plt.errorbar(data['t_phot'], imag_out_data - data['imag'], yerr=data['imag_err'], fmt='r.')
    plt.axhline(0, linestyle='-', color='k')
    plt.xlim(data['t_phot'].min(), data['t_phot'].max())
    plt.ylim(-0.3, 0.3)
    plt.xlabel('t - t0 (days)', fontsize=12)
    plt.ylabel('Residual', fontsize=12)
    plt.gca().invert_yaxis()
    plt.savefig(outroot + '_phot.png')

    ##########
    # Astrometry 2D
    ##########
    fig = plt.figure(2)
    plt.clf()

    lens_pos = pspl_out.get_lens_astrometry(t_out)
    srce_pos_unlens = pspl_out.get_astrometry_unlensed(t_out)
    
    t0idx = np.argmin(np.abs(t_out - pspl_out.t0))
    x0 = lens_pos[t0idx, 0]
    y0 = lens_pos[t0idx, 1]
    
    plt.errorbar((data['xpos'] - x0) * 1e3, (data['ypos'] - y0) * 1e3,
                 xerr=data['xpos_err'] * 1e3, yerr=data['ypos_err'] * 1e3, fmt='r.')
    plt.plot((pos_out[:, 0] - x0) * 1e3, (pos_out[:, 1] - y0) * 1e3,
             'k-', label='Source (lensed)')
    plt.plot((srce_pos_unlens[:, 0] - x0) * 1e3, (srce_pos_unlens[:, 1] - y0) * 1e3,
             'k--', label='Source (unlensed)')
    plt.plot((lens_pos[:, 0] - x0) * 1e3, (lens_pos[:, 1] - y0) * 1e3,
             'b--', label='Lens')
    plt.legend(fontsize=12)
    plt.gca().invert_xaxis()
    plt.xlabel(r'$\Delta$RA (mas)', fontsize=12)
    plt.ylabel(r'$\Delta$Dec (mas)', fontsize=12)
    plt.xlim(5, -10)
    plt.ylim(-10, 5)
    plt.title('Input Data and Output Model', fontsize=12)
    plt.savefig(outroot + '_ast.png')

    ##########
    # Astrometry East vs. time
    ##########
    fig = plt.figure(3)
    plt.clf()
    f1 = fig.add_axes((0.2, 0.3, 0.75, 0.6))
    plt.errorbar(data['t_ast'], (data['xpos'] - x0) * 1e3, yerr=data['xpos_err'] * 1e3, fmt='r.')
    plt.plot(t_out, (pos_out[:, 0] - x0) * 1e3, 'k-')
    plt.xlim(56000, 57600)
    plt.ylabel(r'$\Delta$RA (mas)', fontsize=12)
    plt.title('Input Data and Output Model', fontsize=12)
    f1.set_xticklabels([])
    
    f2 = fig.add_axes((0.2, 0.15, 0.75, 0.15))
    plt.errorbar(data['t_ast'], (pos_out_data[:, 0] - data['xpos']) * 1e3,
                 yerr=data['xpos_err'] * 1e3, fmt='r.')
    plt.axhline(0, linestyle='--', color='k')
    plt.xlim(56000, 57600)
    plt.ylim(-1.5, 1.5)
    plt.xlabel('t - t0 (days)', fontsize=12)
    plt.ylabel('Residuals (mas)', fontsize=12)
    plt.savefig(outroot + '_t_vs_E.png')

    ##########
    # Astrometry East vs. time
    ##########
    fig = plt.figure(4)
    plt.clf()
    f1 = fig.add_axes((0.2, 0.3, 0.75, 0.6))
    plt.errorbar(data['t_ast'], (data['ypos'] - y0) * 1e3, yerr=data['ypos_err'] * 1e3, fmt='r.')
    plt.plot(t_out, (pos_out[:, 1] - y0) * 1e3, 'k-')
    plt.xlim(56000, 57600)
    plt.ylabel(r'$\Delta$Dec (mas)', fontsize=12)
    plt.title('Input Data and Output Model', fontsize=12)
    f1.set_xticklabels([])
    
    f2 = fig.add_axes((0.2, 0.15, 0.75, 0.15))
    plt.errorbar(data['t_ast'], (pos_out_data[:, 1] - data['ypos']) * 1e3,
                 yerr=data['ypos_err'] * 1e3, fmt='r.')
    plt.axhline(0, linestyle='--', color='k')
    plt.xlim(56000, 57600)
    plt.ylim(-1.5, 1.5)
    plt.xlabel('t - t0 (days)', fontsize=12)
    plt.ylabel('Residuals (mas)', fontsize=12)
    plt.savefig(outroot + '_t_vs_N.png')
    
    return
