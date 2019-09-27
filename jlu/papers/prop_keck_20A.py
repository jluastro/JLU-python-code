import numpy as np
import pylab as plt
import pdb
import math
import os
from jlu.observe import skycalc
from microlens.jlu import munge
from microlens.jlu import residuals
from microlens.jlu import model_fitter, model
import shutil, os, sys
import scipy
import scipy.stats
from gcwork import starset
from gcwork import starTables
from astropy.table import Table
from jlu.util import fileUtil

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
                         fmt='k.', alpha=0.2)
        fm1.errorbar(data['t_phot2'], data['mag2'] + r_min_k, yerr=data['mag_err2'],
                         fmt='g.', alpha=0.2)
        fm1.plot(t_mod_pho, m_lens_mod, 'r-')
        fm2.errorbar(data['t_phot1'], data['mag1'] - m_lens_mod_at_phot1, yerr=data['mag_err1'],
                         fmt='k.', alpha=0.2)
        fm2.errorbar(data['t_phot2'], data['mag2'] - m_lens_mod_at_phot2, yerr=data['mag_err2'],
                         fmt='g.', alpha=0.2)
        fm2.axhline(0, linestyle='--', color='r')
        fm2.set_xlabel('Time (HJD)')
        fm1.set_ylabel('Magnitude')
        fm1.invert_yaxis()
        
        
        # RA vs. time
        f1 = plt.gcf().add_axes([fig_pos[1], 0.36, pan_wid, 0.6])
        f2 = plt.gcf().add_axes([fig_pos[1], 0.18, pan_wid, 0.2])
        f1.errorbar(data['t_ast'], data['xpos']*1e3,
                        yerr=data['xpos_err']*1e3, fmt='k.')
        f1.plot(t_mod_ast, p_lens_mod[:, 0]*1e3, 'r-')
        f1.plot(t_mod_ast, p_unlens_mod[:, 0]*1e3, 'r--')
        f1.set_ylabel(r'$\Delta \alpha^*$ (mas)')
        f1.get_shared_x_axes().join(f1, f2)
        
        f2.errorbar(data['t_ast'], (data['xpos'] - p_unlens_mod_at_ast[:,0]) * 1e3,
                    yerr=data['xpos_err'] * 1e3, fmt='k.', alpha=0.2)
        f2.plot(t_mod_ast, (p_lens_mod[:, 0] - p_unlens_mod[:, 0])*1e3, 'r-')
        f2.axhline(0, linestyle='--', color='r')
        f2.set_xlabel('Time (HJD)')
        f2.set_ylabel('Obs - Mod')

        
        # Dec vs. time
        f3 = plt.gcf().add_axes([fig_pos[2], 0.36, pan_wid, 0.6])
        f4 = plt.gcf().add_axes([fig_pos[2], 0.18, pan_wid, 0.2])
        f3.errorbar(data['t_ast'], data['ypos']*1e3,
                        yerr=data['ypos_err']*1e3, fmt='k.')
        f3.plot(t_mod_ast, p_lens_mod[:, 1]*1e3, 'r-')
        f3.plot(t_mod_ast, p_unlens_mod[:, 1]*1e3, 'r--')
        f3.set_ylabel(r'$\Delta \delta$ (mas)')
        f3.get_shared_x_axes().join(f3, f4)
        
        f4.errorbar(data['t_ast'], (data['ypos'] - p_unlens_mod_at_ast[:,1]) * 1e3,
                    yerr=data['ypos_err'] * 1e3, fmt='k.', alpha=0.2)
        f4.plot(t_mod_ast, (p_lens_mod[:, 1] - p_unlens_mod[:, 1])*1e3, 'r-')
        f4.axhline(0, linestyle='--', color='r')
        f4.set_xlabel('Time (HJD)')


        # Mass posterior
        masses = 10**tab['log_thetaE'] / (8.14 * 10**tab['log_piE'])
        weights = tab['weights']
        print(masses[0:10])
        print(weights[0:10])
        
        f5 = plt.gcf().add_axes([fig_pos[3], 0.18, pan_wid, 0.8])
        f5.hist(masses, weights=weights, bins=50)
        f5.set_xlabel('Mass (M$_\odot$)')
        f5.set_ylabel('Probability')


    plt.close(2)
    plot_4panel(mod_all[1], tab_all[1])
    

    return
    
