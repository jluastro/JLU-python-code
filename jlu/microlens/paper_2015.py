import numpy as np
import pylab as py
from matplotlib.ticker import ScalarFormatter, FormatStrFormatter, MultipleLocator, NullFormatter
from mpl_toolkits.axes_grid1 import Grid
import pdb
import asciidata
from jlu.gc.gcwork import starset
from jlu.microlens import multinest_plot
from jlu.microlens import calc
from astropy.time import Time
from jlu.microlens import residuals
from jlu.microlens import MCMC_LensModel
from jlu.util import fileUtil
from scipy import stats
import ephem
import math
import scipy
import scipy.stats
from astropy.table import Table
import astropy.table as table
import shutil
# from jlu.microlens.multinest import run_ob110022_err_x2 as mnest_ob110022

plot_dir = '/Users/jlu/doc/papers/microlens/microlens_paper/Paper/'
pub_dir = plot_dir + '699-2_Publication_apj_v3/'
analysis_dir = '/Users/jlu/work/microlens/2015_evan/'

final_targets = ['ob110022', 'ob110125', 'ob120169']
final_Kcut = [22, 22, 22]
final_order = [1, 2, 2]

align_runs = {18: {1: 'ad', 2: 'ap', 3: 'bb'},
              20: {1: 'ah', 2: 'at', 3: 'bf'},
              22: {1: 'al', 2: 'ax', 3: 'bj'}}

final_flag = [align_runs[final_Kcut[0]][final_order[0]],
              align_runs[final_Kcut[1]][final_order[1]],
              align_runs[final_Kcut[2]][final_order[2]]]

mnest_runs = {'ob110022': {1: ['bf'], 2: ['bc']},
              'ob120169': {1: ['ax', 'ay'], 2: ['bh', 'ba']}}
# final_mnest = [mnest_runs[final_targets[0]][],
#                mnest_runs[final_targets[2]][]]


def remake_and_publish():
    remake_all()
    publish()
    
    return

def remake_all():
    plot_figure1a()
    plot_figure1b()
    table_ao_observations()
    plotPosError(target='ob110022')
    plotPosError(target='ob110125')
    plotPosError(target='ob120169')
    table_of_data('ob110022')
    table_of_data('ob110125')
    table_of_data('ob120169')
    plot_vel_err_vs_mag_all()
    plot_residuals_figures()
    plot_pos_on_sky('ob110022')
    plot_pos_on_sky('ob110125')
    plot_pos_on_sky('ob120169')
    table_proper_motions()

    ##### Run Multinest
    # from jlu.microlens.multinest import run_ob110022
    # run_ob110022.multinest_run()
    # from jlu.microlens.multinest import run_ob120169
    ## MODIFY run_ob120169 to do u0minus
    # reload(run_ob120169)
    # run_ob120169.multinest_run()
    ## MODIFY run_ob120169 to do u0plus
    # reload(run_ob120169)
    # run_ob120169.multinest_run()

    # Multinest Results    
    plot_OB120169(u0minus=True)
    plot_OB120169(u0minus=False)
    plot_OB110022()
    plot_posteriors_muRel_OB110022()

    # Appendices
    plot_align_weight_final()
    plot_align_order_final()
    
    return

def publish():
    shutil.copyfile(plot_dir + 'Phot_Ast_compare.pdf', pub_dir + 'f1a.pdf')
    shutil.copyfile(plot_dir + 'MassTrend.pdf', pub_dir + 'f1b.pdf')
    shutil.copyfile(plot_dir + 'tab_ao_obs.tex', pub_dir + 'table1.tex')
    shutil.copyfile(plot_dir + 'plotPosError_ob110022.pdf', pub_dir + 'f3.pdf')
    shutil.copyfile(plot_dir + 'plotPosError_ob110125.pdf', pub_dir + 'f4.pdf')
    shutil.copyfile(plot_dir + 'plotPosError_ob120169.pdf', pub_dir + 'f5.pdf')

    print '**** FIX ****'
    print 'Change Numbers in Text with: '
    print 'paper_2015.calc_average_pos_aln_error()'
    calc_average_pos_aln_error()
    
    shutil.copyfile(plot_dir + 'tab_data_ob110022.tex', pub_dir + 'table2.tex')
    shutil.copyfile(plot_dir + 'tab_data_ob110125.tex', pub_dir + 'table3.tex')
    shutil.copyfile(plot_dir + 'tab_data_ob120169.tex', pub_dir + 'table4.tex')
    shutil.copyfile(plot_dir + 'hist_chi2_final.pdf', pub_dir + 'f_new_6.pdf')
    shutil.copyfile(plot_dir + 'velErr_vs_mag_all.pdf', pub_dir + 'f6.pdf')

    print '**** FIX ****'
    print 'Change Numbers in Text with:'
    print 'paper_2015.compare_chi2("ob110022")'
    print 'paper_2015.compare_chi2("ob110125")'
    print 'paper_2015.compare_chi2("ob120169")'
    print ''
    compare_chi2("ob110022")    
    compare_chi2("ob110125")    
    compare_chi2("ob120169")    
    
    shutil.copyfile(plot_dir + 'plotStar_ob110022_all_pub_xcorr_Xinvert.pdf', pub_dir + 'f7.pdf')
    shutil.copyfile(plot_dir + 'plotStar_ob110125_all_pub_xcorr_Xinvert.pdf', pub_dir + 'f8.pdf')
    shutil.copyfile(plot_dir + 'plotStar_ob120169_all_pub_xcorr_Xinvert.pdf', pub_dir + 'f9.pdf')
    shutil.copyfile(plot_dir + 'plot_pos_on_sky_ob110022.pdf', pub_dir + 'f11a.pdf')
    shutil.copyfile(plot_dir + 'plot_pos_on_sky_ob110125.pdf', pub_dir + 'f11b.pdf')
    shutil.copyfile(plot_dir + 'plot_pos_on_sky_ob120169.pdf', pub_dir + 'f11c.pdf')
    shutil.copyfile(plot_dir + 'tab_proper_motions.tex', pub_dir + 'table5.tex')

    ##### Run Multinest
    # from jlu.microlens.multinest import run_ob110022
    # run_ob110022.multinest_run()
    # from jlu.microlens.multinest import run_ob120169
    ## MODIFY run_ob120169 to do u0minus
    # reload(run_ob120169)
    # run_ob120169.multinest_run()
    ## MODIFY run_ob120169 to do u0plus
    # reload(run_ob120169)
    # run_ob120169.multinest_run()
    
    # Write Chi^2 values for lensing fits.
    print '**** FIX ****'
    print 'Change chi^2 values from'
    print 'paper_2015.calc_chi2_lens_fit()'
    print ''
    paper_2015.calc_chi2_lens_fit()
    
    # Copy over Mnest results plots
    file_ob110022 = analysis_dir + 'analysis_ob110022_2014_03_22' + final_flag[0] + '_MC100_omit_1/'
    file_ob110022 += '/multiNest/' + mnest_runs['ob110022'][final_order[0]][0] + '/plots/plot_OB110022_data_vs_model.pdf'
    shutil.copyfile(file_ob110022, pub_dir + 'f13.pdf')
    
    file_ob120169_u0m = analysis_dir + 'analysis_ob120169_2014_03_22' + final_flag[2] + '_MC100_omit_1/'
    file_ob120169_u0m += '/multiNest/' + mnest_runs['ob120169'][final_order[2]][0] + '/plots/plot_OB20169_data_vs_model.pdf'
    shutil.copyfile(file_ob120169_u0m, pub_dir + 'f14.pdf')
    
    file_ob120169_u0p = analysis_dir + 'analysis_ob120169_2014_03_22' + final_flag[2] + '_MC100_omit_1/'
    file_ob120169_u0p += '/multiNest/' + mnest_runs['ob120169'][final_order[2]][1] + '/plots/plot_OB120169_data_vs_model.pdf'
    shutil.copyfile(file_ob120169_u0p, pub_dir + 'f17.pdf')

    
    # Write mass limits into text: abstract, results
    print '**** FIX ****'
    print 'Change mass limits in text: abstract, results'
    print '    look for: paper_2015.mass_posterior_OB110022()'
    print '    look for: paper_2015.mass_posterior_OB120169()'
    print ''
    paper_2015.mass_posterior_OB110022()
    paper_2015.mass_posterior_OB120169()

    print '**** FIX ****'
    print 'Change Tables With Lens Fit Parameters: requires LOTS of manual modifications.'
    print '   - merge OB120169 tables'
    print '   - modify significanat digits'
    print '   - fix t0 row header label'
    print 'table_lens_parameters()'
    print ''
    table_lens_parameters()

    shutil.copyfile(plot_dir + 'ob110022_post_muRelxy.png', '')
    
    

    # Appendices
    shutil.copyfile(plot_dir + 'align_weight_OB110022_res_sig.pdf', pub_dir + 'f20.pdf')

    print '**** FIX ****'
    print 'Change Table Values With: '
    print 'align_residuals_vs_order_summary()'
    print ''
    align_residuals_vs_order_summary(only_stars_in_fit=False)
    
    shutil.copyfile(plot_dir + 'align_order_all_res_sig.pdf', pub_dir + 'f22.pdf')
    shutil.copyfile(plot_dir + 'align_order_all_chi2.pdf', pub_dir + 'f23.pdf')
    shutil.copyfile(plot_dir + 'align_order_all_res_sig_a.pdf', pub_dir + 'f24.pdf')
    shutil.copyfile(plot_dir + 'align_order_all_chi2_a.pdf', pub_dir + 'f25.pdf')

    return
    

def plot_figure1b(M=[1,5,10], beta=0.0,
                  logxmin = -3, logxmax = 2., dl = 4.0, ds = 8.0,
                  plot_dir=plot_dir):

    M = np.array(M)
    x = np.logspace(logxmin, logxmax, 10000)
    pi_rel = 4.848e-9*(1./dl - 1./ds)*3600.*1000.*180./np.pi #mas
    kappa = 8.1459
    thetaE = np.sqrt(kappa*M*pi_rel)

    py.close(1)
    fig = py.figure(1, figsize=(8,6))
    py.subplots_adjust(left=0.15, bottom=0.15)
    ax = py.gca()
    ax.set_yscale('linear')
    ax.set_xscale('log')
    ax.tick_params(axis='both', which='major', pad=8)
    ax.grid(True, which='both', linestyle=':', color='grey')
    sep = np.sqrt(x**2.+beta**2.)
    shift = thetaE[0]*sep/(sep**2. +2)
    ax.plot(sep, shift, '-r', label= str(M[0]) + r' M$_\odot$', lw=2)
    shift = thetaE[1]*sep/(sep**2 + 2)
    ax.plot(sep, shift, '-b', label= str(M[1]) + r' M$_\odot$', lw=2)        
    shift = thetaE[2]*sep/(sep**2. +2)
    ax.plot(sep, shift, '-k', label= str(M[2]) + r' M$_\odot$', lw=2)
    py.legend(loc=2, fontsize=20)
    py.xlabel('$u$ (Einstein radii)', size = 24)
    py.ylabel('$\delta_c$ (mas)', size = 24, labelpad=10)
    py.xticks(fontsize=20)
    py.yticks(fontsize=20)
    ax.xaxis.set_tick_params(width=1.3, which='both')
    ax.yaxis.set_tick_params(width=.3)
    py.xlim(0.01, 30)
   
    py.savefig(plot_dir + 'MassTrend.pdf')
     
    return

     
def plot_figure1a(M=10., beta=0.1,
                  logxmin = -2, logxmax = 2., dl = 4.0, ds = 8.0,
                  plot_dir=plot_dir):

    #u is proxy for t/tE
    x = np.logspace(logxmin, logxmax, 10000)
    pi_rel = 4.848e-9*(1./dl - 1./ds)*3600.*1000.*180./np.pi #mas
    kappa = 8.1459
    thetaE = np.sqrt(kappa*M*pi_rel)
    
    py.close(1)
    fig = py.figure(1, figsize=(8,6))
    py.subplots_adjust(left=0.15, bottom=0.15)
    ax = py.gca()
    ax.set_yscale('linear')
    ax.set_xscale('log')
    ax.tick_params(axis='both', which='major', pad=8)
    ax.grid(True, which='both', linestyle=':', color='grey')

    sep = np.sqrt(x**2. + beta**2.) 
    shift = 10.*thetaE*sep/(sep**2. +2)
    ax.plot(x, shift, '-k', lw=2)
    mag = (sep**2. + 2)/(sep*np.sqrt(sep**2 + 4))
    ax.plot(x, mag, '--k', lw=2)
    py.xlim(0.01, 30)
    py.xlabel('$(t-t_0)\; / \; t_E$', size = 24)
    py.ylabel('$Test_c$', size = 24, color='white', labelpad=10)
    py.annotate('10x Shift [mas]', xy=(2.9, 9.0), rotation=-68,
                xycoords='data', fontsize=20)
    py.annotate('Magnification', xy=(0.06, 9.0), rotation=-62,
                xycoords='data', fontsize=20)
    py.xticks(fontsize=20)
    py.yticks(fontsize=20)
    ax.xaxis.set_tick_params(width=1.3, which='both')
    ax.yaxis.set_tick_params(width=.3)    

    py.savefig(plot_dir + 'Phot_Ast_compare.pdf')

    return

     
def plotPosError(rootDir='/u/jlu/data/microlens/', plot_dir=plot_dir,
                 radius=4, target = 'ob110022'):
    """
    Make three standard figures that show the data quality 
    from a *_rms.lis file. 

    1. astrometric error as a function of magnitude.
    2. photometric error as a function of magnitude.
    3. histogram of number of stars vs. magnitude.

    Use raw=True to plot the individual stars in plots 1 and 2.
    """
    
    if target == 'ob110022':
        epochs = ['11may','11jul','12jun','12jul','13apr','13jul']
        epoch_txt = ['2011 May','2011 Jul','2011 Jun','2012 Jul','2013 Apr','2013 Jul']
    else:    
        epochs = ['12may','12jun','12jul','13apr','13jul']
        epoch_txt = ['2012 May','2012 Jun', '2012 Jul','2013 Apr','2013 Jul']
    
    Nepochs = len(epochs)
    
    # Assume this is NIRC2 data.
    scale = 0.009952
    
    axfontsize = 18
    tickfontsize = 12

    py.close(1)
    fig = py.figure(1, figsize=(12,6))
    fig.subplots_adjust(left=0.1, right=0.95, top=0.95)
    grid = Grid(fig, 111,
                nrows_ncols=(2, 3),
                axes_pad=0.1,
                add_all=False,
                label_mode='all')

    majorFmt = FormatStrFormatter('%d')
    nullFmt = NullFormatter()

    for ee in range(Nepochs):
        starlist = rootDir + epochs[ee] + '/combo/starfinder/' + \
          'mag' + epochs[ee] + '_' + target + '_kp_rms_named.lis'
     
        # Load up the starlist
        lis = asciidata.open(starlist)
        
        name = lis[0]._data
        mag = lis[1].tonumpy()
        x = lis[3].tonumpy()
        y = lis[4].tonumpy()
        xerr = lis[5].tonumpy()
        yerr = lis[6].tonumpy()
        snr = lis[7].tonumpy()
        corr = lis[8].tonumpy()
        merr = 1.086 / snr
    
        # Convert into arsec offset from field center
        # We determine the field center by assuming that stars
        # are detected all the way out the edge.
        xhalf = x.max() / 2.0
        yhalf = y.max() / 2.0
        x = (x - xhalf) * scale
        y = (y - yhalf) * scale
        xerr *= scale * 1000.0
        yerr *= scale * 1000.0
    
        r = np.hypot(x, y)
        err = (xerr + yerr) / 2.0

        # Find the target
        if target == 'ob120169':
            target_name = 'ob120169_R'
        else:
            target_name = target
        ttt = np.where([nn == target_name for nn in name])[0][0]
        print 'Found Target: ', ttt
        print '  K = {0:.2f}'.format(mag[ttt])
        print '  err = {0:.2f} mas'.format(err[ttt])
    
        ##########
        #
        # Plot astrometry errors
        #
        ##########
        idx = (np.where(r < radius))[0]
        fig.add_axes(grid[ee])
        grid[ee].semilogy(mag[idx], err[idx], 'k.')
        grid[ee].semilogy(mag[ttt], err[ttt], 'r.', ms=10)
        
        grid[ee].axis([12.5, 23, 2e-2, 30.0])
        grid[ee].axhline(0.15, color='g') 
        grid[ee].xaxis.label.set_fontsize(tickfontsize)
        grid[ee].yaxis.label.set_fontsize(tickfontsize)

        # Turn off tick labels for the top row and right columns (with exceptions):
        if ee < 3:
            if (ee == 2) and ((target == 'ob110125') or (target == 'ob120169')):
                pass
            else:
                for xlabel_i in grid[ee].xaxis.get_ticklabels():
                    xlabel_i.set_visible(False)

        if (ee == 1) or (ee == 2) or (ee == 4) or (ee == 5):
            for ylabel_i in grid[ee].yaxis.get_ticklabels():
                ylabel_i.set_visible(False)
                
        if ee > 1:
            if (target == 'ob110022') and ee == 1:
                pass
            else:
                grid[ee].set_xlabel('K Magnitude',
                                    fontsize= axfontsize)

        # Write Epoch
        grid[ee].text(13, 10, epoch_txt[ee], fontsize=axfontsize, color='k')
        
    py.figtext(0.02, 0.55, 'Positional Uncertainty (mas)', fontsize=axfontsize,
                rotation='vertical', verticalalignment='center')
    py.savefig(plot_dir + 'plotPosError_%s.pdf' % (target))
            

    return


def plot_pos_on_sky(target, plot_dir=plot_dir):
    align = 'align/align_t'
    poly = 'polyfit_d/fit'
    points = 'points_d/'

    # Get the analysis directory to work on.    
    ff = final_targets.index(target)
    rootDir = analysis_dir
    rootDir += 'analysis_' + final_targets[ff] + '_2014_03_22' + final_flag[ff] + '_MC100_omit_1/'

    # Load the align and polyfit results for all stars.
    s = starset.StarSet(rootDir + align)
    s.loadPolyfit(rootDir + poly, accel=0, arcsec=0)
    
    names = s.getArray('name')
    mag = s.getArray('mag')
    x = s.getArray('x')
    y = s.getArray('y')
    r = np.hypot(x,y)

    # Find the target and fetch the points measurements.
    if target == 'ob120169':
        star_name='ob120169_R'
    else:
        star_name = target
    ii = names.index(star_name)
    star = s.stars[ii]
       
    pointsTab = asciidata.open(rootDir + points + star_name + '.points')
            
    time = pointsTab[0].tonumpy()
    x = pointsTab[1].tonumpy()
    y = pointsTab[2].tonumpy()
    xerr = pointsTab[3].tonumpy()
    yerr = pointsTab[4].tonumpy()

    # Make arrays to map the best fit positions
    # (at the times we have observations).        
    fitx = star.fitXv
    fity = star.fitYv
    dt = time - fitx.t0
    fitLineX = fitx.p + (fitx.v * dt)
    fitSigX = np.sqrt( fitx.perr**2 + (dt * fitx.verr)**2 )
    
    fitLineY = fity.p + (fity.v * dt)
    fitSigY = np.sqrt( fity.perr**2 + (dt * fity.verr)**2 )

    dt2 = np.arange(time[0] - 0.2, time[-1] + 0.2, 0.1) - fitx.t0
    fitLineX2 = fitx.p + (fitx.v * dt2)
    fitSigX2 = np.sqrt( fitx.perr**2 + (dt2 * fitx.verr)**2 )
    fitLineY2 = fity.p + (fity.v * dt2)
    fitSigY2 = np.sqrt( fity.perr**2 + (dt2 * fity.verr)**2 )
    
        
    from matplotlib.ticker import FormatStrFormatter
    fmtX = FormatStrFormatter('%3i')
    fmtY = FormatStrFormatter('%3i')
    fontsize1 = 18
    fontsize2 = 16
	                       
    plateScale = 9.952
    x = x * plateScale
    y = y * plateScale
    fitLineX = fitLineX * plateScale
    fitLineY = fitLineY * plateScale
    fitSigX  = fitSigX * plateScale
    fitSigY  = fitSigY * plateScale
    fitLineX2 = fitLineX2 * plateScale
    fitLineY2 = fitLineY2 * plateScale
    fitSigX2  = fitSigX2 * plateScale
    fitSigY2  = fitSigY2 * plateScale
    xerr = xerr * plateScale
    yerr = yerr * plateScale

    xmean = np.mean(x)
    ymean = np.mean(y)
    
    if target == 'ob110022':
        ytics = [-6,-3, 0, 3, 6]
        xtics = [-6,-3, 0, 3, 6]
        xlimit = (-7, 7)
        title = 'OB110022'
        
    if target == 'ob110125':
        ytics = [-1, 0, 1]
        xtics = [-1, 0, 1]
        xlimit = (-1.5, 1.5)
        title = 'OB110125'

    if target == 'ob120169':
        ytics = [-2, 0, 2]
        xtics = [-2, 0, 2]
        xlimit = (-2.5, 2.5)
        title = 'OB120169'

    # Plotting
    figsize = py.gcf().get_size_inches()
    if (figsize[0] != 6.175) and (figsize[1] != 6.175):
        print 'Closing and re-opening figure'
        py.close(1)
        py.figure(1)
    else:
        print 'Using existing figure'
        py.clf()
    py.subplots_adjust(left=0.15, top=0.925)

    dx = (x - xmean) * -1.0
    dy = (y - ymean)
    dx_fit_obs = (fitLineX - xmean) * -1.0
    dy_fit_obs = (fitLineY - ymean)
    dx_fit = (fitLineX2 - xmean) * -1.0
    dy_fit = (fitLineY2 - ymean)
    dx_fit_perr = (fitLineX2 + fitSigX2 - xmean) * -1.0
    dx_fit_nerr = (fitLineX2 - fitSigX2 - xmean) * -1.0
    dy_fit_perr = (fitLineY2 + fitSigY2 - ymean)
    dy_fit_nerr = (fitLineY2 - fitSigY2 - ymean)
    
    py.plot(dx_fit, dy_fit, 'b-')

    if target == 'ob120169':
        py.plot(dx_fit_perr, dy_fit_nerr, 'b--')
        py.plot(dx_fit_nerr, dy_fit_perr, 'b--')
    else:
        py.plot(dx_fit_perr, dy_fit_perr, 'b--')
        py.plot(dx_fit_nerr, dy_fit_nerr, 'b--')
        
    py.errorbar(dx, dy, yerr=yerr, xerr=xerr, fmt='k.', markersize=10)

    # Plot lines between the observed points and the fit points
    for ee in range(len(dx)):
        py.plot([dx[ee], dx_fit_obs[ee]], [dy[ee], dy_fit_obs[ee]], 'k-', color='grey')
                
    py.ylabel(r'$x_{N}$ (mas)', fontsize=fontsize1)
    py.xlabel(r'$x_{E}$ (mas)', fontsize=fontsize1)
    py.gca().xaxis.set_major_formatter(fmtX)
    py.gca().yaxis.set_major_formatter(fmtY)
    py.tick_params(axis='both', which='major', labelsize=fontsize2)
	
    py.axis('equal')
    py.xticks(xtics)
    py.yticks(xtics)                                
    py.ylim(xlimit)
    py.gca().invert_xaxis()
    py.title(title)

    py.savefig(plot_dir + 'plot_pos_on_sky_' + target + '.pdf')

def check_parallax_correction(target, u0pos=True):
    root_dir='/u/sinukoff/projects/microlens/analysis/'

    if target == 'ob110022':
        root_dir += 'analysis_ob110022_2014_03_22' + final_flag[0] + '_MC100_omit_1/'
        mnest_dir = 'multiNest/av/'
        mnest_root = 'av'
    if target == 'ob120169':
        root_dir += 'analysis_ob120169_2014_03_22' + final_flag[2] + '_MC100_omit_1/'
        if u0pos:
            mnest_dir = 'multiNest/ay/'
            mnest_root = 'ay'
        else:
            mnest_dir = 'multiNest/ax/'
            mnest_root = 'ax'
            
    params = multinest_plot.get_best_fit(root_dir, mnest_dir, mnest_root)

    calc.parallax_correction(target,
                             params['t0'], params['tE'],
                             np.array([params['thetaS0x'], params['thetaS0y']]),
                             np.array([params['muSx'], params['muSy']]),
                             np.array([params['muRelx'], params['muRely']]),
                             params['beta'],
                             np.array([params['piEE'], params['piEN']]))

    return


def convert_dates():
    """
    Convert the JD dates coming from Evan's code to fractional year.
    """
    def print_times(target, 
                    phot_p, phot_p_err_p, phot_p_err_n,
                    ast_p, ast_p_err_p, ast_p_err_n):
        hdr_f = '{0:10s} {1:13s} {2:13s}'
        row_f = '{0:10s} {1:12.4f} {2:12.4f}'

        print ''
        print '*****************'
        print target + ' in MJD at Earth:'
        print hdr_f.format('', 'Photometry', 'Astrometry')
        print row_f.format('Value', phot_p.utc.mjd, ast_p.utc.mjd)
        print row_f.format('Error+', 
                           phot_p_err_p.utc.mjd - phot_p.utc.mjd, 
                           ast_p_err_p.utc.mjd - ast_p.utc.mjd)
        print row_f.format('Error-', 
                           phot_p_err_n.utc.mjd - phot_p.utc.mjd, 
                           ast_p_err_n.utc.mjd - ast_p.utc.mjd)
    
        print target + ' in decimalYear at Earth:'
        print hdr_f.format('', 'Photometry', 'Astrometry')
        print row_f.format('Value', phot_p.utc.jyear, ast_p.utc.jyear)
        print row_f.format('Error+', 
                           phot_p_err_p.utc.jyear - phot_p.utc.jyear, 
                           ast_p_err_p.utc.jyear - ast_p.utc.jyear)
        print row_f.format('Error-', 
                           phot_p_err_n.utc.jyear - phot_p.utc.jyear, 
                           ast_p_err_n.utc.jyear - ast_p.utc.jyear)

        print target + ' in MJD at Sun:'
        print hdr_f.format('', 'Photometry', 'Astrometry')
        print row_f.format('Value', phot_p.tcb.mjd, ast_p.tcb.mjd)
        print row_f.format('Error+', 
                           phot_p_err_p.tcb.mjd - phot_p.tcb.mjd, 
                           ast_p_err_p.tcb.mjd - ast_p.tcb.mjd)
        print row_f.format('Error-', 
                           phot_p_err_n.tcb.mjd - phot_p.tcb.mjd, 
                           ast_p_err_n.tcb.mjd - ast_p.tcb.mjd)

        print 

        return

    mklat = 19.8207
    mklon = -155.4681
    
    # OB110022
    phot_p = Time(2455687.913, format='jd', scale='tcb', 
                  lat=mklat, lon=mklon)
    phot_p_err_p = Time(2455687.913 + 0.265, format='jd', scale='tcb', 
                        lat=mklat, lon=mklon)
    phot_p_err_n = Time(2455687.913 - 0.255, format='jd', scale='tcb', 
                        lat=mklat, lon=mklon)
    ast_p = Time(2011.3422, format='jyear', scale='utc', 
                 lat=mklat, lon=mklon)
    ast_p_err_p = Time(2011.3422 + 0.0007, format='jyear', scale='utc', 
                 lat=mklat, lon=mklon)
    ast_p_err_n = Time(2011.3422 - 0.0007, format='jyear', scale='utc', 
                 lat=mklat, lon=mklon)

    print_times('OB110022', 
                phot_p, phot_p_err_p, phot_p_err_n,
                ast_p, ast_p_err_p, ast_p_err_n)    
    pdb.set_trace()
    

    # OB 110125
    phot_p = Time(2455724.431, format='jd', scale='tcb', 
                  lat=mklat, lon=mklon)
    phot_p_err_p = Time(2455724.431 + 0.701, format='jd', scale='tcb', 
                        lat=mklat, lon=mklon)
    phot_p_err_n = Time(2455724.431 - 0.724, format='jd', scale='tcb', 
                        lat=mklat, lon=mklon)
    ast_p = Time(2455724.431, format='jd', scale='utc', 
                  lat=mklat, lon=mklon)
    ast_p_err_p = Time(2455724.431 + 0.701, format='jd', scale='utc', 
                        lat=mklat, lon=mklon)
    ast_p_err_n = Time(2455724.431 - 0.724, format='jd', scale='utc', 
                        lat=mklat, lon=mklon)

    print_times('OB110125',
                phot_p, phot_p_err_p, phot_p_err_n,
                ast_p, ast_p_err_p, ast_p_err_n)    


    # OB 120169 (u_0 > 0)
    phot_p = Time(2456026.247, format='jd', scale='tcb', 
                  lat=mklat, lon=mklon)
    phot_p_err_p = Time(2456026.247 + 0.379, format='jd', scale='tcb', 
                        lat=mklat, lon=mklon)
    phot_p_err_n = Time(2456026.247 - 0.435, format='jd', scale='tcb', 
                        lat=mklat, lon=mklon)
    ast_p = Time(2012.2604, format='jyear', scale='utc', 
                 lat=mklat, lon=mklon)
    ast_p_err_p = Time(2012.2604 + 0.0007, format='jyear', scale='utc', 
                 lat=mklat, lon=mklon)
    ast_p_err_n = Time(2012.2604 - 0.0008, format='jyear', scale='utc', 
                 lat=mklat, lon=mklon)

    print_times('OB120169 (u_0 > 0)',
                phot_p, phot_p_err_p, phot_p_err_n,
                ast_p, ast_p_err_p, ast_p_err_n)    

    # OB 120169 (u_0 < 0)
    phot_p = Time(2456026.029, format='jd', scale='tcb', 
                  lat=mklat, lon=mklon)
    phot_p_err_p = Time(2456026.029 + 0.401, format='jd', scale='tcb', 
                        lat=mklat, lon=mklon)
    phot_p_err_n = Time(2456026.029 - 0.428, format='jd', scale='tcb', 
                        lat=mklat, lon=mklon)
    ast_p = Time(2012.2597, format='jyear', scale='utc',
                        lat=mklat, lon=mklon)
    ast_p_err_p = Time(2012.2597 + 0.0007, format='jyear', scale='utc',
                        lat=mklat, lon=mklon)
    ast_p_err_n = Time(2012.2597 - 0.0008, format='jyear', scale='utc',
                        lat=mklat, lon=mklon)
    
    print_times('OB120169 (u_0 < 0)',
                phot_p, phot_p_err_p, phot_p_err_n,
                ast_p, ast_p_err_p, ast_p_err_n)    


def plot_residuals_figures_old():
    # OB120169
    # starNames = ['OB120169_R', 'p005_15_3.5', 'S2_16_2.5', 'p000_16_3.6', 'S6_17_2.4', 'S7_17_2.3']
    starNames = ['OB120169_R', 'OB120169_L', 'S10_17_1.4', 'S12_17_2.4', 'S6_17_2.4', 'S7_17_2.3']

    rootDir = '/Users/jlu/work/microlens/OB120169/analysis_2015_09_18/analysis_ob120169_2015_09_18_a3_m22_w4_MC100/'
    align = 'align/align_t'
    residuals.plotStar(starNames, rootDir=rootDir, align=align)

def plot_residuals_figures():

    # OB120169
    plotStar('ob120169')

    # OB110022
    plotStar('ob110022')

    # OB110125
    plotStar('ob110125')
        
    
def plotStar(targname, align='align/align_t',
             poly='polyfit_d/fit', points='points_d/', figsize=(15,13)):

    rootDir = analysis_dir

    if targname.lower() == 'ob110022': 
        rootDir += 'analysis_' + targname.lower() + '_2014_03_22' + final_flag[0] + '_MC100_omit_1/'
        starNames = ['ob110022', 'p001_14_1.8', 'p002_16_1.0', 's000_16_1.1', 'p003_16_2.3', 's002_17_1.5']
    if targname.lower() == 'ob110125': 
        rootDir += 'analysis_' + targname.lower() + '_2014_03_22' + final_flag[1] + '_MC100_omit_1/'
        starNames = ['ob110125', 'S1_16_3.9', 'S6_17_3.8', 'S13_18_1.7', 'S14_18_2.7', 'p004_18_3.0']
    if targname.lower() == 'ob120169': 
        rootDir += 'analysis_' + targname.lower() + '_2014_03_22' + final_flag[2] + '_MC100_omit_1/'
        starNames = ['ob120169_R', 'p005_15_3.5', 'S2_16_2.5', 'p000_16_3.6', 'S6_17_2.4', 'S7_17_2.3']
        
    print 'Creating residuals plots for star(s):'
    print starNames
    Nstars = len(starNames)
    
    s = starset.StarSet(rootDir + align)
    s.loadPolyfit(rootDir + poly, accel=0, arcsec=0)
    names = s.getArray('name')
    mag = s.getArray('mag')
    x = s.getArray('x')
    y = s.getArray('y')
    r = np.hypot(x,y)
    
    axwidth = 0.12 # fraction of whole figure
    axheight = 0.15
    xgap = 0.04
    ygap = 0.04
    ygap2 = 0.07
    labelpady = 0
    labelpadx = 10
    xleft = 0.06
    xright = 0.04
    ybottom = 0.06
    ytop = 0.04

    py.close(1)    
    fig1 = py.figure(1, figsize=figsize)

    for i in range(Nstars):
    
        starName = starNames[i]
        
        for j in range(1):    
            ii = names.index(starName)
            star = s.stars[ii]
       
            pointsTab = asciidata.open(rootDir + points + starName + '.points')
            
            time = pointsTab[0].tonumpy()
            x = pointsTab[1].tonumpy()
            y = pointsTab[2].tonumpy()
            xerr = pointsTab[3].tonumpy()
            yerr = pointsTab[4].tonumpy()
        
            fitx = star.fitXv
            fity = star.fitYv
            dt = time - fitx.t0
            fitLineX = fitx.p + (fitx.v * dt)
            fitSigX = np.sqrt( fitx.perr**2 + (dt * fitx.verr)**2 )
        
            fitLineY = fity.p + (fity.v * dt)
            fitSigY = np.sqrt( fity.perr**2 + (dt * fity.verr)**2 )
                
        
            diffX = x - fitLineX
            diffY = y - fitLineY
            diff = np.hypot(diffX, diffY)
            rerr = np.sqrt((diffX*xerr)**2 + (diffY*yerr)**2) / diff
            sigX = diffX / xerr
            sigY = diffY / yerr
            sig = diff / rerr
        
            
            # Determine if there are points that are more than 4 sigma off
            idxX = np.where(abs(sigX) > 4)
            idxY = np.where(abs(sigY) > 4)
            idx = np.where(abs(sig) > 4)
            
            dateTicLoc = MultipleLocator(3)
            dateTicRng = [2006, 2010]
            
            if starNames[0] == 'ob110022':
                dateTics = np.array([2011, 2012, 2013, 2014])
            else:
                dateTics = np.array([2012, 2013, 2014])
    	    DateTicsLabel = dateTics-2000
    	    
            maxErr = np.array([xerr, yerr]).max()
            resTicRng = [-3*maxErr, 3*maxErr]
        
            from matplotlib.ticker import FormatStrFormatter
            fmtX = FormatStrFormatter('%2i')
            fmtY = FormatStrFormatter('%6.0f')
            fontsize1 = 14
            fontsize2 = 14
	                               
            plateScale = 9.952
            x = x*plateScale
            y = y*plateScale
            fitLineX = fitLineX*plateScale
            fitLineY = fitLineY*plateScale
            fitSigX  = fitSigX*plateScale
            fitSigY  = fitSigY*plateScale
            xerr = xerr*plateScale
            yerr = yerr*plateScale
            diffX = diffX*plateScale
            diffY = diffY*plateScale

            if starNames[0] == 'ob120169_R':
                starNames[0] = 'ob120169'
            
            if starNames[0] == 'ob110022':
                ytics = [-4, 0, 4]
                xtics = [-4, 0, 4]
                xlimit = (-7, 7)
                xrestics = [-2, 0, 2]
                xreslimit = (-2.5, 2.5) 

            if starNames[0] == 'ob110125':
                ytics = [-3, 0, 3]
                xtics = [-3, 0, 3]
                xlimit = (-3.5, 3.5)
                xrestics = [-2, 0, 2]
                xreslimit = (-2.5, 2.5) 

            if starNames[0] == 'ob120169':
                ytics = [-2, 0, 2]
                xtics = [-2, 0, 2]
                xlimit = (-3.5, 3.5)
                xrestics = [-2, 0, 2]
                xreslimit = (-2.5, 2.5) 

            # X + residuals
            xmean = np.mean(x) 
            
            if starNames[0] == 'ob120169':
                if i == 1 or i == 3 or i == 4:
                    ytics = [-6,-3, 0, 3, 6]
                    xtics = [-6,-3, 0, 3, 6]
                    xlimit = (-7, 7)
                    xrestics = [-2, 0, 2]
                    xreslimit = (-2.5, 2.5) 

            left = xleft + i * (xgap + axwidth)
            bottom = 1.0 - (1.0 * (ygap + axheight))
            width = axwidth
            height = axheight
            resheight = height / 3.0
            resbottom = bottom - resheight
			
            frame1 = fig1.add_axes((left, bottom, width, height))
            #xstart, ystart, xend, yend [units are fraction of the image frame, from bottom left corner]
            py.annotate(starName, xy=(0.3,1.05), xycoords='axes fraction', fontsize=12, color='red')
            py.plot(time, (fitLineX - xmean)*-1., 'b-')
            py.plot(time, (fitLineX + fitSigX - xmean)*-1, 'b--')
            py.plot(time, (fitLineX - fitSigX - xmean)*-1, 'b--')
            py.errorbar(time, (x - xmean)*-1, yerr=xerr, fmt='k.')
            rng = py.axis()
            py.ylim(xlimit) 
            if i == 0:
                py.ylabel( r'$x_{E}$ (mas)', fontsize=fontsize1, labelpad = labelpady)
            frame1.yaxis.set_major_formatter(fmtY)
            frame1.tick_params(axis='both', which='major', labelsize=fontsize2)
            py.yticks(xtics)
            py.xticks(dateTics, DateTicsLabel)
            frame1.set_xticklabels([]) #Remove x-tic labels for the first frame        

            frame2 = fig1.add_axes((left, resbottom, width, resheight))
            py.plot(time, np.zeros(len(time)), 'b-')
            py.plot(time, (fitSigX)*-1, 'b--')
            py.plot(time, (-fitSigX)*-1, 'b--')
            py.yticks(xrestics)
            py.errorbar(time, (x - fitLineX), yerr=xerr, fmt='k.')
            if i == 0:
                py.ylabel('Residual', fontsize=fontsize1, labelpad = labelpady)
            frame2.get_xaxis().set_major_locator(dateTicLoc)
            frame2.xaxis.set_major_formatter(fmtX)
            frame2.yaxis.set_major_formatter(fmtY)
            frame2.tick_params(axis='both', which='major', labelsize=fontsize2)
			
            py.xticks(dateTics, DateTicsLabel)
            py.xlim(np.min(dateTics), np.max(dateTics))
            py.ylim(xreslimit)
            py.xticks(dateTics, DateTicsLabel)
            py.xlim(np.min(dateTics), np.max(dateTics))
              

            # Y + Residuals
            bottom = resbottom - ygap - height
            resbottom = bottom - resheight
            ymean = np.mean(y) 
                       
            frame3 = fig1.add_axes((left, bottom, width, height))
            #xstart, ystart, xend, yend [units are fraction of the image frame, from bottom left corner]
            py.plot(time, fitLineY - ymean, 'b-')
            py.plot(time, fitLineY + fitSigY - ymean, 'b--')
            py.plot(time, fitLineY - fitSigY - ymean, 'b--')
            py.errorbar(time, y - ymean, yerr=yerr, fmt='k.')
            rng = py.axis()
            py.ylim(xlimit) 
            if i == 0:
                py.ylabel(r'$x_{N}$ (mas)', fontsize=fontsize1, labelpad = labelpady)
            frame3.yaxis.set_major_formatter(fmtY)
            frame3.tick_params(axis='both', which='major', labelsize=fontsize2)
            py.yticks(xtics)
            py.xticks(dateTics, DateTicsLabel)
            frame3.set_xticklabels([]) #Remove x-tic labels for the first frame        

            frame4 = fig1.add_axes((left, resbottom, width, resheight))            
            py.plot(time, np.zeros(len(time)), 'b-')
            py.plot(time, fitSigY, 'b--')
            py.plot(time, -fitSigY, 'b--')
            py.yticks(xrestics)
            py.errorbar(time, y - fitLineY, yerr=yerr, fmt='k.')
            py.xlabel('Date - 2000 (yrs)', fontsize=fontsize1, labelpad = labelpadx)
            if i == 0:
                py.ylabel('Residual', fontsize=fontsize1, labelpad = labelpady)
            frame4.get_xaxis().set_major_locator(dateTicLoc)
            frame4.xaxis.set_major_formatter(fmtX)
            frame4.yaxis.set_major_formatter(fmtY)
            frame4.tick_params(axis='both', which='major', labelsize=fontsize2)
			
            py.xticks(dateTics, DateTicsLabel)
            py.xlim(np.min(dateTics), np.max(dateTics))
            py.ylim(xreslimit)
            py.xticks(dateTics, DateTicsLabel)
            py.xlim(np.min(dateTics), np.max(dateTics))

            
            if starNames[0] == 'ob110022':
                if i == 1 or i == 4:
                    ytics = [-2,-1, 0, 1, 2]
                    xtics = [-2,-1, 0, 1, 2]
                    xlimit = (-2, 2)
            if starNames[0] == 'ob110125':
                if i == 0:
                    ytics = [-2,-1, 0, 1, 2]
                    xtics = [-2,-1, 0, 1, 2]
                    xlimit = (-2, 2)
            if starNames[0] == 'ob120169':
                if i == 1 or i == 3 or i == 4:
                    ytics = [-4, 0, 4]
                    xtics = [-4, 0, 4]
                    xlimit = (-6, 6)
            
            bottom = resbottom - ygap2 - height
            
            frame5 = fig1.add_axes((left, bottom, width, height))
            py.plot((fitLineX-xmean)*-1, fitLineY-ymean, 'b-')
            py.errorbar((x-xmean)*-1, y - ymean, yerr=yerr, xerr=xerr, fmt='k.')
            if i == 0:
                py.ylabel(r'$x_{N}$ (mas)', fontsize=fontsize1, labelpad = labelpady)
            py.xlabel(r'$x_{E}$ (mas)', fontsize=fontsize1, labelpad = labelpadx)
            frame5.get_xaxis().set_major_locator(dateTicLoc)
            frame5.xaxis.set_major_formatter(fmtX)
            frame5.yaxis.set_major_formatter(fmtY)
            frame5.tick_params(axis='both', which='major', labelsize=fontsize2)
            [t.set_ha('center') for t in frame5.get_xticklabels()]
			
            py.xlim(xlimit)
            py.ylim(xlimit)
            py.xticks(xtics)
            py.yticks(xtics)                                
            py.xlim(py.xlim()[::-1])
            bottom = bottom - ygap2 - height

            frame6 = fig1.add_axes((left, bottom, width, height))
            bins = np.arange(-7, 7, 1)
            idx = np.where(diffY < 0)[0]
            sig[idx] = -1.*sig[idx] 
            (n, b, p) = py.hist(sigX, bins, histtype='stepfilled', color='b')
            py.setp(p, 'facecolor', 'b')
            (n, b, p) = py.hist(sigY, bins, histtype='step', color='r')
            py.axis([-7, 7, 0, 8], fontsize=fontsize1)
            py.xlabel('Residuals (sigma)', fontsize=fontsize1, labelpad = labelpadx)
            if i == 0:
                py.ylabel('# of Epochs', fontsize=fontsize1, labelpad = labelpady)
            py.ylim(0, len(time))
            frame6.tick_params(axis='both', which='major', labelsize=fontsize2)
            frame6.xaxis.set_major_formatter(fmtX)
            frame6.yaxis.set_major_formatter(fmtY)
            py.xticks([-6,-4,-2,0,2,4,6])
            [t.set_ha('center') for t in frame6.get_xticklabels()]
            
			       
    py.savefig(plot_dir + 'plotStar_' + targname + '_all_pub_xcorr_Xinvert' + '.pdf',
               bbox_inches='tight',pad_inches=0.5)

def plot_vel_err_vs_mag_all():
    """Analyze the distribution of points relative to their best
    fit velocities. Optionally trim the largest outliers in each
    stars *.points file.  Optionally make a magnitude cut with
    magCut flag and/or a radius cut with radCut flag."""
    
    rootDir = analysis_dir

    targets = ['ob110022', 'ob110125', 'ob120169']

    rootDirs = {'ob110022': rootDir + 'analysis_ob110022_2014_03_22' + final_flag[0] + '_MC100_omit_1/',
                'ob110125': rootDir + 'analysis_ob110125_2014_03_22' + final_flag[1] + '_MC100_omit_1/',
                'ob120169': rootDir + 'analysis_ob120169_2014_03_22' + final_flag[2] + '_MC100_omit_1/'}

    mag_arr = {}
    ve_arr = {}
        
    align = 'align/align_t'
    poly = 'polyfit_d/fit'
    points = 'points_d/'
    magCut = 100
    radCut = 100

    py.clf()

    for ii in range(len(targets)):
        rootDir = rootDirs[targets[ii]]
        
        s = starset.StarSet(rootDir + align)
        s.loadPolyfit(rootDir + poly, accel=0, arcsec=0)

        #####
        # 
        # Make Cuts to Sample of Stars
        #
        #####

        # Get som parameters we will cut on.
        mag = s.getArray('mag')
        x = s.getArray('x')
        y = s.getArray('y')
        r = np.hypot(x, y)

        # Trim out stars.
        idx = np.where((mag < magCut) & (r < radCut))[0]
        newstars = []
        for i in idx:
            newstars.append(s.stars[i])
        s.stars = newstars

        # Get arrays we want to plot    
        mag = s.getArray('mag')
        ve_x = s.getArray('fitXv.verr')
        ve_y = s.getArray('fitYv.verr')

        scale = 9.952  # mas/pixel
    
        ve_x_mas = ve_x * scale
        ve_y_mas = ve_y * scale
        ve_avg = (ve_x_mas + ve_y_mas) / 2.0

        mag_arr[targets[ii]] = mag
        ve_arr[targets[ii]] = ve_avg

        
    #####
    # Plot Velocity Errors vs. Kp
    #####    
    markersize = 8.0 
    xmin = 12.5
    xmax = 20.5
    ymin = 0
    ymax = 0.98

    fig2 = py.figure(2)
    fig2.set_size_inches(6, 9, forward=True)
    py.clf()
    py.subplots_adjust(left=0.15, bottom=0.09, hspace=0, wspace=0, right=0.95, top=0.95)

    nullFmt = NullFormatter()

    for ii in range(len(targets)):
        py.subplot(3, 1, ii+1)
        py.plot(mag_arr[targets[ii]], ve_arr[targets[ii]], 'ko', markersize=markersize)
        py.xlim(xmin, xmax)
        py.ylim(ymin, ymax)
        py.yticks(fontsize=16)
        py.xticks(fontsize=16)
        if ii == 1:
            py.ylabel('Proper Motion Uncertainty [mas/yr]')
            
        if ii == (len(targets) - 1):
            py.xlabel('K magnitude')
        else:
            py.gca().xaxis.set_major_formatter(nullFmt)

        py.text(18, 0.8, targets[ii].upper())
        
    py.savefig(plot_dir + 'velErr_vs_mag_all.pdf')

    return
    
def plot_vel_err_vs_mag(targname):
    """Analyze the distribution of points relative to their best
    fit velocities. Optionally trim the largest outliers in each
    stars *.points file.  Optionally make a magnitude cut with
    magCut flag and/or a radius cut with radCut flag."""
    
    # Get the analysis directory to work on.    
    ff = final_targets.index(target)
    rootDir = analysis_dir
    rootDir += 'analysis_' + final_targets[ff] + '_2014_03_22' + final_flag[ff] + '_MC100_omit_1/'

    align = 'align/align_t'
    poly = 'polyfit_d/fit'
    points = 'points_d/'
    magCut = 100
    radCut = 100
    
    s = starset.StarSet(rootDir + align)
    s.loadPolyfit(rootDir + poly, accel=0, arcsec=0)

    #####
    # 
    # Make Cuts to Sample of Stars
    #
    #####

    # Get som parameters we will cut on.
    mag = s.getArray('mag')
    x = s.getArray('x')
    y = s.getArray('y')
    r = np.hypot(x, y)

    # Trim out stars.
    idx = np.where((mag < magCut) & (r < radCut))[0]
    newstars = []
    for i in idx:
        newstars.append(s.stars[i])
    s.stars = newstars

    # Get arrays we want to plot    
    mag = s.getArray('mag')
    x = s.getArray('x')
    y = s.getArray('y')
    r = np.hypot(x, y)
    chi2red_x = s.getArray('fitXv.chi2red')
    chi2red_y = s.getArray('fitYv.chi2red')
    chi2_x = s.getArray('fitXv.chi2')
    chi2_y = s.getArray('fitYv.chi2')
    ve_x = s.getArray('fitXv.verr')
    ve_y = s.getArray('fitYv.verr')

    scale = 9.952  # mas/pixel
    
    ve_x_mas = ve_x * scale
    ve_y_mas = ve_y * scale
    
    #####
    # Plot Velocity Errors vs. Kp
    #####    
    markersize = 8.0 
    xmin = 12.5
    xmax = 20.5
    #ymin = 8e-2
    ymin = 0
    ymax = 1.0

    fig1 = py.figure(1)
    fig1.set_size_inches(6, 6, forward=True)
    py.clf()
    py.plot(mag, ve_x_mas, 'ro', markersize=markersize, label='X', markeredgecolor=None)
    py.plot(mag, ve_y_mas, 'bo', markersize=markersize, label='Y', markeredgecolor=None)
    py.xlim(xmin, xmax)
    py.ylim(ymin, ymax)
    py.ylabel('Proper Motion Uncertinaty [mas/yr]', fontsize=16)
    py.xlabel('K magnitude', fontsize=16)
    py.legend(loc='upper left')
    py.yticks(fontsize=16)
    py.xticks(fontsize=16)
    py.savefig(plot_dir + 'velErr_vs_mag_xy_' + targname + '.pdf')

    fig2 = py.figure(2)
    fig2.set_size_inches(6, 6, forward=True)
    py.clf()
    py.plot(mag, (ve_x_mas + ve_y_mas) / 2.0, 'ko', markersize=markersize)
    py.xlim(xmin, xmax)
    py.ylim(ymin, ymax)
    py.ylabel('Proper Motion Uncertainty [mas/yr]', fontsize=16)
    py.xlabel('K magnitude', fontsize=16)
    py.yticks(fontsize=16)
    py.xticks(fontsize=16)
    py.savefig(plot_dir + 'velErr_vs_mag_' + targname + '.pdf')

    return

def compare_chi2(target):
    """
    Print out the chi^2 values of the source and the
    relative fraction of other stars from a comparison
    sample with chi^2 values above and below the sources.
    """
    
    # Get the analysis directory to work on.    
    ff = final_targets.index(target)
    rootDir = analysis_dir
    rootDir += 'analysis_' + final_targets[ff] + '_2014_03_22' + final_flag[ff] + '_MC100_omit_1/'

    align = 'align/align_t'
    poly = 'polyfit_d/fit'
    points = 'points_d/'
    magCut = 22
    radCut = 10.0
    
    s = starset.StarSet(rootDir + align)
    s.loadPolyfit(rootDir + poly, accel=0, arcsec=0)

    #####
    # 
    # Make Cuts to Sample of Stars
    #
    #####

    # Get som parameters we will cut on.
    mag = s.getArray('mag')
    x = s.getArray('x')
    y = s.getArray('y')
    r = np.hypot(x, y)
    cnt = s.getArray('velCnt')

    cntCut = cnt.max()

    # Trim out stars.
    idx = np.where((mag < magCut) & (r < radCut) & (cnt <= cntCut))[0]
    newstars = []
    for i in idx:
        newstars.append(s.stars[i])
    s.stars = newstars

    # Get arrays we want to plot    
    name = s.getArray('name')
    
    # Find the target
    if target == 'ob120169':
        target_name = 'ob120169_R'
    else:
        target_name = target
    ttt = np.where([nn == target_name for nn in name])[0][0]
    
    mag = s.getArray('mag')
    x = s.getArray('x')
    y = s.getArray('y')
    rad = np.hypot(x - x[ttt], y - y[ttt])
    chi2red_x = s.getArray('fitXv.chi2red')
    chi2red_y = s.getArray('fitYv.chi2red')
    chi2_x = s.getArray('fitXv.chi2')
    chi2_y = s.getArray('fitYv.chi2')
    vx = s.getArray('fitXv.v')
    vy = s.getArray('fitYv.v')
    vxe = s.getArray('fitXv.verr')
    vye = s.getArray('fitYv.verr')

    scale = 9.952
    vx *= scale * -1.0
    vy *= scale
    vxe *= scale
    vye *= scale
    ve = np.hypot(vxe, vye)

    N_dof_x = np.round(chi2_x / chi2red_x)
    N_dof_y = np.round(chi2_y / chi2red_y)

    # Combine the X and Y chi-squared distributions.
    chi2red = chi2red_x + chi2red_y
    chi2 = chi2_x + chi2_y
    N_dof = N_dof_x + N_dof_y


    ##########
    # Plot a cumulative distribution function of the chi^2 values. 
    ##########
    chi2_bins = np.arange(0, 30, 0.25)
    chi2_model = scipy.stats.chi2.cdf(chi2_bins, N_dof[0])
    #chi2_model /= 3.0  # arbitrary

    py.clf()
    py.hist(chi2, bins=chi2_bins, cumulative=True, histtype='step', normed=True)
    py.axvline(chi2[ttt], color='red', linestyle='-')
    py.plot(chi2_bins, chi2_model, 'k--')
    py.ylim(0, 1)

    ##########
    # Print out the chi^2 comparison results.
    ##########

    higher = np.where(chi2 > chi2[ttt])[0]
    high_frac = 1.0 * len(higher) / len(chi2)

    print 'Chi^2 Value for {0:s}:'.format(target)
    print '     {0:5.1f} for {1:3.0f} DOF'.format(chi2[ttt], N_dof[ttt])
    print 'Fraction Higher = {0:.2f}'.format(high_frac)
    print 'Comparison Sample Size = {0:d}'.format(len(chi2))
    print 'Kcut = {0:d}  Rcut = {1:5.2f}  Ncut = {2:d}'.format(magCut, radCut, cntCut)
    print '{0:5s} {1:5s} {2:5s} {3:5s}'.format(' Mag', ' Chi2', ' vErr', ' rad')
    print '{0:5.2f} {1:5.1f} {2:5.2f} {3:5.2f}'.format(mag[ttt], chi2[ttt], ve[ttt], rad[ttt])
    print ''
    print 'Stars with Higher Chi^2:'
    print '{0:5s} {1:5s} {2:5s} {3:5s}'.format(' Mag', ' Chi2', ' vErr', ' rad')
    for ii in higher:
        print  '{0:5.2f} {1:5.1f} {2:5.2f} {3:5.2f}'.format(mag[ii], chi2[ii], ve[ii], rad[ii])
    
    return

def table_proper_motions():
    """Get the final proper motion for each of the targets."""

    outfile = plot_dir + 'tab_proper_motions.tex'
    _out = open(outfile, 'w')
    _out.write('\\begin{deluxetable}{lrrrc}\n')
    _out.write('\\tabletypesize{\\footnotesize}\n')
    _out.write('\\tablecaption{Proper Motions}\n')
    _out.write('\\tablecolumns{3}\n')
    _out.write('\\tablewidth{0pt}\n')
    _out.write('\\tablehead{\n')
    _out.write('Source & $\mu_{\\x}$ & $\mu_{\\y}$  \\\\ \n')
    _out.write('& [mas yr$^{-1}$] & [mas yr$^{-1}$] & $\chi_{vel}^2$ & DOF \\\\ \n')
    _out.write('}\n')
    _out.write('\\startdata\n')
    
    for ff in range(len(final_targets)):    
        # Get the analysis directory to work on.
        target = final_targets[ff]
        rootDir = analysis_dir
        rootDir += 'analysis_' + final_targets[ff] + '_2014_03_22' + final_flag[ff] + '_MC100_omit_1/'
        
        align = 'align/align_t'
        poly = 'polyfit_d/fit'
        points = 'points_d/'
    
        s = starset.StarSet(rootDir + align)
        s.loadPolyfit(rootDir + poly, accel=0, arcsec=0)
    
        # Get arrays we want to plot
        name = s.getArray('name')
        v_x = s.getArray('fitXv.v')
        v_y = s.getArray('fitYv.v')
        ve_x = s.getArray('fitXv.verr')
        ve_y = s.getArray('fitYv.verr')
        chi2_x = s.getArray('fitXv.chi2')
        chi2_y = s.getArray('fitYv.chi2')
        dof_x = s.getArray('fitXv.dof')
        dof_y = s.getArray('fitYv.dof')
        chi2 = chi2_x + chi2_y
        dof = dof_x + dof_y

        scale = 9.952  # mas/pixel
    
        ve_x_mas = ve_x * scale
        ve_y_mas = ve_y * scale
        v_x_mas = v_x * scale * -1.0
        v_y_mas = v_y * scale

        # Find the target
        if target == 'ob120169':
            target_name = 'ob120169_R'
        else:
            target_name = target
        ttt = np.where([nn == target_name for nn in name])[0][0]

        fmt = '{0:10s} & {1:5.2f} $\pm$ {2:5.2f} & {3:5.2f} $\pm$ {4:5.2f} & {5:5.1f} & {6:1d} '

        _out.write( fmt.format(target.upper(),
                               v_x_mas[ttt], ve_x_mas[ttt],
                               v_y_mas[ttt], ve_y_mas[ttt],
                               chi2[ttt], dof[ttt] ))
        
        if ff != (len(final_targets) - 1):
            _out.write(' \\\\')
        _out.write('\n')

    _out.write('\\enddata\n')
    _out.write('\\label{tb:pm}\n')
    _out.write('\\end{deluxetable}\n')

    return
    
def align_residuals_vs_order(analysis_dirs, only_stars_in_fit=True, out_suffix=''):
    data_all = []

    for aa in range(len(analysis_dirs)):
        data = residuals.check_alignment_fit(root_dir=analysis_dirs[aa])

        data_all.append(data)

    order = np.arange(1, len(data_all)+1)
    N_order = len(data_all)

    year = data_all[0]['year']
    scale = 9.952  # mas / pixel

    N_plots = len(year)
    if only_stars_in_fit:
        N_plots -= 1

    ##########
    # Plot Residuals for each epoch.
    ##########
    fig = py.figure(2)
    fig.set_size_inches(15, 3.5, forward=True)
    py.clf()
    py.subplots_adjust(left=0.08, bottom=0.2, hspace=0, wspace=0, right=0.95, top=0.88)

    idx_subplot = 0
    ymax = 0

    majorLoc = MultipleLocator(1)
    majorFmt = FormatStrFormatter('%d')
    nullFmt = NullFormatter()

    for ee in range(len(year)):
        end = 'all'
        if only_stars_in_fit:
            end = 'used'
        
        xres = np.array([data_all[oo]['xres_rms_'+end][ee] for oo in range(N_order)])
        yres = np.array([data_all[oo]['yres_rms_'+end][ee] for oo in range(N_order)])
        ares = np.hypot(xres, yres)
    
        xres_e = np.array([data_all[oo]['xres_err_p_'+end][ee] for oo in range(N_order)])
        yres_e = np.array([data_all[oo]['yres_err_p_'+end][ee] for oo in range(N_order)])
        ares_e = np.hypot(xres * xres_e, yres * yres_e) / ares

        xres *= scale
        yres *= scale
        ares *= scale
        xres_e *= scale
        yres_e *= scale
        ares_e *= scale

        if only_stars_in_fit and np.isnan(xres[0]):
            print 'Skipping Ref Epoch', ee
            continue

        idx_subplot += 1

        # Residuals.        
        if idx_subplot == 1:
            ax1 = py.subplot(1, N_plots, idx_subplot)
        else:
            py.subplot(1, N_plots, idx_subplot, sharex=ax1)
        py.errorbar(order, xres, yerr=xres_e, fmt='r.--', label='X')
        py.errorbar(order, yres, yerr=yres_e, fmt='b.--', label='Y')
        py.errorbar(order, ares, yerr=ares_e, fmt='g.--', label='Both')
        py.title(data_all[0]['year'][ee])
        py.xlim(0, 3.9)
        ax = py.gca()
        ax.xaxis.set_major_locator(majorLoc)
        ax.xaxis.set_major_formatter(majorFmt)
        py.xlabel('Order')

        ymax = np.ceil(np.max(ares.tolist() + [ymax]))
        py.ylim(0, ymax)
        
        if idx_subplot == 1:
            #ax.yaxis.set_major_formatter(majorFmt)
            py.ylabel('RMS Residuals (mas)')
        else:
            ax.yaxis.set_major_formatter(nullFmt)

        if idx_subplot == N_plots:
            py.legend(numpoints=1)
            
    py.savefig(plot_dir + 'align_resi_vs_order_' + out_suffix + '.png')

    
    ##########
    # Plot Chi^2 for each epoch.
    ##########
    fig = py.figure(3)
    fig.set_size_inches(15, 3.5, forward=True)
    py.clf()
    py.subplots_adjust(left=0.08, bottom=0.2, hspace=0, wspace=0, right=0.95, top=0.88)
    
    idx_subplot = 0
    ymax = 0

    majorLoc = MultipleLocator(1)
    majorFmt = FormatStrFormatter('%d')
    nullFmt = NullFormatter()

    for ee in range(len(year)):
        end = 'all'
        if only_stars_in_fit:
            end = 'used'
            
        chi2x = np.array([data_all[oo]['chi2x_'+end][ee] for oo in range(N_order)])
        chi2y = np.array([data_all[oo]['chi2y_'+end][ee] for oo in range(N_order)])
        chi2 = chi2x + chi2y
        N_stars = np.array([data_all[oo]['N_stars_'+end][ee] for oo in range(N_order)])

        if only_stars_in_fit and (N_stars[0] == 0):
            print 'Skipping Ref Epoch', ee
            continue

        idx_subplot += 1

        # Chi^2
        if idx_subplot == 1:
            ax1 = py.subplot(1, N_plots, idx_subplot)
        else:
            py.subplot(1, N_plots, idx_subplot, sharex=ax1)
        py.plot(order, chi2x, 'r.--', label='X')
        py.plot(order, chi2y, 'b.--', label='Y')
        py.plot(order, chi2, 'g.--', label='Both')
        py.title(data_all[0]['year'][ee])
        py.xlim(0, 3.9)
        ax = py.gca()
        ax.xaxis.set_major_locator(majorLoc)
        ax.xaxis.set_major_formatter(majorFmt)
        py.xlabel('Order')

        ymax = np.max(chi2 + [ymax])
        py.ylim(0, ymax+5)
        
        if idx_subplot == 1:
            py.ylabel(r'$\chi^2$')
        else:
            ax.yaxis.set_major_formatter(nullFmt)

        if idx_subplot == N_plots:
            py.legend(numpoints=1)
            
    py.savefig(plot_dir + 'align_chi2_vs_order_' + out_suffix + '.png')

    end = 'all'
    if only_stars_in_fit:
        end = 'used'
        
    N_par = np.array([3, 6, 10])

    outfile = plot_dir + 'align_resi_vs_order_' + out_suffix + '.txt'
    _out = open(outfile, 'w')
    
    # Fit Improvement Significance
    for ee in range(len(year)):
        chi2x = np.array([data_all[oo]['chi2x_'+end][ee] for oo in range(N_order)])
        chi2y = np.array([data_all[oo]['chi2y_'+end][ee] for oo in range(N_order)])
        chi2a = chi2x + chi2y
        N_stars = np.array([data_all[oo]['N_stars_'+end][ee] for oo in range(N_order)])

        if only_stars_in_fit and (N_stars[0] == 0):
            print 'Skipping Ref Epoch', ee
            continue

        # Print out some values for this epoch and each order.
        print ''
        print 'Epoch: {0:8.3f}'.format(year[ee])
        _out.write('\n')
        _out.write('Epoch: {0:8.3f}'.format(year[ee]))
        for mm in range(N_order):
            fmt = 'N_stars = {0:3d}  N_par = {1:3d}  N_dof = {2:3d}  Chi^2 X = {3:5.1f}  Chi^2 Y = {4:5.1f}  Chi^2 Tot = {5:5.1f}'
            print fmt.format(int(N_stars[mm]), int(N_par[mm]), int(N_stars[mm] - N_par[mm]),
                             chi2x[mm], chi2y[mm], chi2a[mm])
            
            fmt = 'N_stars = {0:3d}  N_par = {1:3d}  N_dof = {2:3d}  Chi^2 X = {3:5.1f}  Chi^2 Y = {4:5.1f}  Chi^2 Tot = {5:5.1f}\n'
            _out.write(fmt.format(int(N_stars[mm]), int(N_par[mm]), int(N_stars[mm] - N_par[mm]),
                             chi2x[mm], chi2y[mm], chi2a[mm]))
    
        # F-test for X and Y and total.
        # These have shape: len(order) - 1
        ftest_m = order[1:]
        ftest_dof1 = np.diff(N_par)
        ftest_dof2 = N_stars[1:] - N_par[1:]
        ftest_dof_ratio = 1.0 / (ftest_dof1 / ftest_dof2)
        ftest_x = (-1 * np.diff(chi2x) / chi2x[1:]) * ftest_dof_ratio
        ftest_y = (-1 * np.diff(chi2y) / chi2y[1:]) * ftest_dof_ratio
        ftest_a = (-1 * np.diff(chi2a) / chi2a[1:]) * ftest_dof_ratio

        px_value = np.zeros(len(ftest_m), dtype=float)
        py_value = np.zeros(len(ftest_m), dtype=float)
        pa_value = np.zeros(len(ftest_m), dtype=float)

        for mm in range(len(ftest_m)):
            px_value[mm] = stats.f.sf(ftest_x[mm], ftest_dof1[mm], ftest_dof2[mm])
            py_value[mm] = stats.f.sf(ftest_y[mm], ftest_dof1[mm], ftest_dof2[mm])
            pa_value[mm] = stats.f.sf(ftest_a[mm], 2*ftest_dof1[mm], 2*ftest_dof2[mm])
            fmt = 'M = {0}: M-1 --> M     Fa = {1:5.2f} pa = {2:7.4f}    Fx = {3:5.2f} px = {4:7.4f}   Fy = {5:5.2f} py = {6:7.4f}'
            print fmt.format(ftest_m[mm], ftest_a[mm], pa_value[mm],
                             ftest_x[mm], px_value[mm], ftest_y[mm], py_value[mm])

            fmt = fmt + '\n'
            _out.write(fmt.format(ftest_m[mm], ftest_a[mm], pa_value[mm],
                             ftest_x[mm], px_value[mm], ftest_y[mm], py_value[mm]))

    ##########
    # Combine over all epochs NOT including velocities:
    ##########
    chi2x = np.array([data_all[oo]['chi2x_'+end].sum() for oo in range(N_order)])
    chi2y = np.array([data_all[oo]['chi2y_'+end].sum() for oo in range(N_order)])
    chi2 = chi2x + chi2y

    # Number of stars is constant for all orders.
    N_data = 2 * data_all[0]['N_stars_'+end].sum()
    # Number of free parameters in the alignment fit.
    N_afit = (len(year) - 1) * N_par * 2
    N_free_param = N_afit
    N_dof = N_data - N_free_param

    print ''
    print '*** Combined F test across epochs (NO velocities) ***'
    _out.write('\n')
    _out.write('*** Combined F test across epochs (NO velocities) ***\n')
    
    for mm in range(N_order):
        fmt = 'Order={0:d}  N_stars = {1:3d}  N_par = {2:3d}  N_dof = {3:3d}  Chi^2 = {4:5.1f}'
        print fmt.format(order[mm], int(N_data), int(N_free_param[mm]), int(N_dof[mm]), chi2[mm])
        fmt = fmt + '\n'
        _out.write(fmt.format(order[mm], int(N_data), int(N_free_param[mm]), int(N_dof[mm]), chi2[mm]))
        
    ftest_m = order[1:]
    ftest_dof1 = np.diff(N_free_param)
    ftest_dof2 = N_data - N_free_param[1:]
    ftest_dof_ratio = 1.0 / (ftest_dof1 / ftest_dof2)
    ftest = (-1 * np.diff(chi2) / chi2[1:]) * ftest_dof_ratio
        
    p_value = np.zeros(len(ftest_m), dtype=float)

    # print 'ftest_m: ', ftest_m
    # print 'ftest_dof1: ', ftest_dof1
    # print 'ftest_dof2: ', ftest_dof2
    # print 'ftest_dof_ratio: ', ftest_dof_ratio
    # print 'ftest: ', ftest
    
    for mm in range(len(ftest_m)):
        p_value[mm] = stats.f.sf(ftest[mm], ftest_dof1[mm], ftest_dof2[mm])
        fmt = 'M = {0}: M-1 --> M   F = {1:5.2f}  p = {2:7.4f}'
        print fmt.format(ftest_m[mm], ftest[mm], p_value[mm])
    
        fmt = fmt + '\n'
        _out.write(fmt.format(ftest_m[mm], ftest[mm], p_value[mm]))
    
            
    ##########
    # Combine over all epochs to include velocities:
    ##########

    # Number of stars is constant for all orders.
    N_data = 2 * data_all[0]['N_stars_'+end].sum()
    # Get the maximum number of stars used in any epoch... this is close enough?    
    N_stars_max = data_all[0]['N_stars_'+end].max()
    # Number of free parameters in the velocity fit.
    N_vfit = 4 * N_stars_max   # (x0, vx, y0, vy for each star)
    # Number of free parameters in the alignment fit.
    N_afit = (len(year) - 1) * N_par * 2

    N_free_param = N_vfit + N_afit
    N_dof = N_data - N_free_param

    print ''
    print '*** Combined F test across epochs (WITH velocities) ***'
    _out.write('\n')
    _out.write('*** Combined F test across epochs (WITH velocities) ***\n')
    
    for mm in range(N_order):
        fmt = 'Order={0:d}  N_stars = {1:3d}  N_par = {2:3d}  N_dof = {3:3d}  Chi^2 = {4:5.1f}'
        print fmt.format(order[mm], int(N_data), int(N_free_param[mm]), int(N_dof[mm]), chi2[mm])

        fmt = fmt + '\n'
        _out.write(fmt.format(order[mm], int(N_data), int(N_free_param[mm]), int(N_dof[mm]), chi2[mm]))
        
    ftest_m = order[1:]
    ftest_dof1 = np.diff(N_free_param)
    ftest_dof2 = N_data - N_free_param[1:]
    ftest_dof_ratio = 1.0 / (ftest_dof1 / ftest_dof2)
    ftest = (-1 * np.diff(chi2) / chi2[1:]) * ftest_dof_ratio
        
    p_value = np.zeros(len(ftest_m), dtype=float)

    for mm in range(len(ftest_m)):
        p_value[mm] = stats.f.sf(ftest[mm], ftest_dof1[mm], ftest_dof2[mm])
        fmt = 'M = {0}: M-1 --> M   F = {1:5.2f}  p = {2:7.4f}'
        print fmt.format(ftest_m[mm], ftest[mm], p_value[mm])
        
        fmt = fmt + '\n'
        _out.write(fmt.format(ftest_m[mm], ftest[mm], p_value[mm]))

    return

def align_residuals_vs_order_summary(only_stars_in_fit=True, Kcut=22):
    root_dir = '/Users/jlu/work/microlens/2015_evan/'

    # Make 2D arrays for the runs with different Kcut and orders.
    Kcuts = np.array([18, 20, 22])
    orders = np.array([1, 2, 3])
    targets = ['ob110022', 'ob110125', 'ob120169']

    runs = {18: {1: 'ad', 2: 'ap', 3: 'bb'},
            20: {1: 'ah', 2: 'at', 3: 'bf'},
            22: {1: 'al', 2: 'ax', 3: 'bj'}}

    end = 'all'
    if only_stars_in_fit:
        end = 'used'

    # Number of free parameters in the alignment fit for each order
    N_par_aln = np.array([3, 6, 10])

    print '********'
    print '* Results for Kcut={0:d} and in_fit={1}'.format(Kcut, only_stars_in_fit)
    print '********'

    for tt in range(len(targets)):
        # make some variables we will need for the F-test
        N_free_all = np.zeros(len(orders), dtype=int)
        N_data_all = np.zeros(len(orders), dtype=int)
        chi2_all = np.zeros(len(orders), dtype=float)
        
        for oo in range(len(orders)):
            flag = runs[Kcut][orders[oo]]
            
            analysis_root_fmt = 'analysis_{0}_2014_03_22{1}_MC100_omit_1/'
            analysis_dir = root_dir + analysis_root_fmt.format(targets[tt], flag)

            data = residuals.check_alignment_fit(root_dir=analysis_dir)

            year = data['year']
            scale = 9.952  # mas / pixel
        
            # Get the total chi^2 for all stars in all epochs
            chi2x = data['chi2x_' + end].sum()
            chi2y = data['chi2y_' + end].sum()
            chi2 = chi2x + chi2y

            ##########
            # Combine over all epochs including velocities:
            ##########

            # Number of stars is constant for all orders.
            N_data = 2 * data['N_stars_' + end].sum()
            # Get the maximum number of stars used in any epoch... this is close enough?    
            N_stars_max = data['N_stars_'+end].max()
            # Number of free parameters in the velocity fit.
            N_vfit = 4 * N_stars_max   # (x0, vx, y0, vy for each star)
            # Number of free parameters in the alignment fit.
            N_afit = (len(year) - 1) * N_par_aln[oo] * 2

            N_free_param = N_vfit + N_afit
            N_dof = N_data - N_free_param

            fmt = 'Target={0:s}  Order={1:d}  N_stars = {2:3d}  N_par = {3:3d}  N_dof = {4:3d}  Chi^2 = {5:5.1f}'
            print fmt.format(targets[tt], orders[oo], int(N_data), int(N_free_param), int(N_dof), chi2)

            N_free_all[oo] = int(N_free_param)
            N_data_all[oo] = int(N_data)
            chi2_all[oo] = chi2
        
        # Done with all orders in this target... lets print F-test results
        ftest_m = orders[1:]
        ftest_dof1 = np.diff(N_free_all)
        ftest_dof2 = N_data_all[1:] - N_free_all[1:]
        ftest_dof_ratio = 1.0 / (1.0 * ftest_dof1 / ftest_dof2)
        ftest = (-1 * np.diff(chi2_all) / chi2_all[1:]) * ftest_dof_ratio
        
        p_value = np.zeros(len(ftest_m), dtype=float)

        for mm in range(len(ftest_m)):
            p_value[mm] = stats.f.sf(ftest[mm], ftest_dof1[mm], ftest_dof2[mm])
            fmt = 'M = {0}: M-1 --> M   F = {1:5.2f}  p = {2:7.4f}'
            print fmt.format(ftest_m[mm], ftest[mm], p_value[mm])

    return

def plot_align_order_final():
    root_dir = '/Users/jlu/work/microlens/2015_evan/'

    Kcut = 22
    orders = np.array([1, 2, 3])

    runs = {18: {1: 'ad', 2: 'ap', 3: 'bb'},
            20: {1: 'ah', 2: 'at', 3: 'bf'},
            22: {1: 'al', 2: 'ax', 3: 'bj'}}

    flags = [runs[Kcut][1], runs[Kcut][2], runs[Kcut][3]]
    targets = ['ob110022', 'ob110125', 'ob120169']

    analysis_dir_fmt = 'analysis_{0}_2014_03_22{1}_MC100_omit_1/'
    
    ##########
    # Load up data for all target/order combinations.
    ##########

    chi2 = {}
    resi = {}
    sigm = {}
    ndof = {}
    chi2a = {}
    sigma = {}
            
    for tt in range(len(targets)):
        chi2[targets[tt]] = {}
        resi[targets[tt]] = {}
        sigm[targets[tt]] = {}
        ndof[targets[tt]] = {}
        chi2a[targets[tt]] = {}
        sigma[targets[tt]] = {}
        
        for ii in range(len(orders)):
            analysis_dir = root_dir + analysis_dir_fmt.format(targets[tt], flags[ii])
            align_root = 'align/align_t'

            out = residuals.chi2_dist_all_epochs(align_root, root_dir=analysis_dir,
                                                 only_stars_in_fit=False,
                                                 plotfile=None)

            xe_ap = np.hypot(out['xe_p'], out['xe_a'])
            ye_ap = np.hypot(out['ye_p'], out['ye_a'])

            chi2x_a = ((out['xres'] / xe_ap)**2).sum(axis=0)
            chi2y_a = ((out['yres'] / ye_ap)**2).sum(axis=0)
            
            chi2[targets[tt]][orders[ii]] = np.concatenate([out['chi2x'], out['chi2y']])
            resi[targets[tt]][orders[ii]] = np.concatenate([out['xres'], out['yres']])
            sigm[targets[tt]][orders[ii]] = np.concatenate([out['xres'] / out['xe_p'],
                                                            out['yres'] / out['ye_p']])
            chi2a[targets[tt]][orders[ii]] = np.concatenate([chi2x_a, chi2y_a])
            sigma[targets[tt]][orders[ii]] = np.concatenate([out['xres'] / xe_ap,
                                                             out['yres'] / ye_ap])
            ndof[targets[tt]][orders[ii]] = out['Ndof_x']

    ##########
    # Plot Residuals with only positional error.
    ##########
    plot_ii = 0

    py.figure(1)
    py.clf()
    fig = py.gcf()
    fig.set_size_inches(12, 12, forward=True)
    py.subplots_adjust(bottom=0.08, left=0.08, top=0.95, wspace=0.28, hspace=0.23)

    sig_bin = 0.5
    sig_bins = np.arange(-6, 6, sig_bin)
    sig_mod_bin = 0.05
    sig_mod_bins = np.arange(-6, 6, sig_mod_bin)
    sig_plot = scipy.stats.norm.pdf(sig_mod_bins)
    
    for ii in range(len(orders)):
        for tt in range(len(targets)):
            sig = sigm[targets[tt]][orders[ii]].flatten()

            py.subplot(3, 3, plot_ii+1)
            py.hist(sig, bins=sig_bins, color='grey', normed=True)
            py.plot(sig_mod_bins, sig_plot, 'k--')

            if tt == 0:
                py.ylabel('PDF for O={0:d}'.format(orders[ii]))
            if ii == 2:
                py.xlabel(r'[p$_{obs}$ - p$_{fit}$] / $\sigma_p$')
            if ii == 0:
                py.title(targets[tt].upper())

            plot_ii += 1

            py.xlim(-6, 6)
            py.ylim(0, 0.6)
            
    py.savefig(plot_dir + 'align_order_all_res_sig.pdf')

    ##########
    # Plot Chi^2 with only positional error.
    ##########
    plot_ii = 0
    
    py.figure(2)
    py.clf()
    fig = py.gcf()
    fig.set_size_inches(12, 12, forward=True)
    py.subplots_adjust(bottom=0.08, left=0.08, top=0.95, wspace=0.28, hspace=0.23)

    chi2_bin = 3.0
    chi2_bins = np.arange(0, 50, chi2_bin)
    chi2_mod_bin = 0.25
    chi2_mod_bins = np.arange(0, 100, chi2_mod_bin)
    
    for ii in range(len(orders)):
        for tt in range(len(targets)):
            chi = chi2[targets[tt]][orders[ii]]
            chi2_plot = scipy.stats.chi2.pdf(chi2_mod_bins, ndof[targets[tt]][orders[ii]])
            #chi2_plot /= 2.0

            py.subplot(3, 3, plot_ii + 1)
            py.hist(chi, bins=chi2_bins, color='grey', normed=True)
            py.plot(chi2_mod_bins, chi2_plot, 'k--')

            if tt == 0:
                py.ylabel('PDF for O={0:d}'.format(orders[ii]))
            if ii == 2:
                py.xlabel(r'$\chi^2$')
            if ii == 0:
                py.title(targets[tt].upper())

            plot_ii += 1

            py.xlim(0, 50)
            py.ylim(0, 0.25)

    py.savefig(plot_dir + 'align_order_all_chi2.pdf')

    ##########
    # Plot Residuals with positional and alignment error.
    ##########
    plot_ii = 0

    py.figure(3)
    py.clf()
    fig = py.gcf()
    fig.set_size_inches(12, 12, forward=True)
    py.subplots_adjust(bottom=0.08, left=0.08, top=0.95, wspace=0.28, hspace=0.23)

    sig_bin = 0.5
    sig_bins = np.arange(-6, 6, sig_bin)
    sig_mod_bin = 0.05
    sig_mod_bins = np.arange(-6, 6, sig_mod_bin)
    sig_plot = scipy.stats.norm.pdf(sig_mod_bins)
    
    for ii in range(len(orders)):
        for tt in range(len(targets)):
            sig = sigma[targets[tt]][orders[ii]].flatten()

            py.subplot(3, 3, plot_ii+1)
            py.hist(sig, bins=sig_bins, color='grey', normed=True)
            py.plot(sig_mod_bins, sig_plot, 'k--')

            if tt == 0:
                py.ylabel('PDF for O={0:d}'.format(orders[ii]))
            if ii == 2:
                py.xlabel(r'[p$_{obs}$ - p$_{fit}$] / $\sigma_p$')
            if ii == 0:
                py.title(targets[tt].upper())

            plot_ii += 1

            py.xlim(-6, 6)
            py.ylim(0, 0.8)
            
    py.savefig(plot_dir + 'align_order_all_res_sig_a.pdf')

    ##########
    # Plot Chi^2 with only positional error.
    ##########
    plot_ii = 0

    py.figure(4)
    py.clf()
    fig = py.gcf()
    fig.set_size_inches(12, 12, forward=True)
    py.subplots_adjust(bottom=0.08, left=0.08, top=0.95, wspace=0.28, hspace=0.23)

    chi2_bin = 1.5
    chi2_bins = np.arange(0, 50, chi2_bin)
    chi2_mod_bin = 0.25
    chi2_mod_bins = np.arange(0, 100, chi2_mod_bin)
    
    for ii in range(len(orders)):
        for tt in range(len(targets)):
            chi = chi2a[targets[tt]][orders[ii]]
            chi2_plot = scipy.stats.chi2.pdf(chi2_mod_bins, ndof[targets[tt]][orders[ii]])
            #chi2_plot /= 1.5  # arbitrary

            py.subplot(3, 3, plot_ii + 1)
            py.hist(chi, bins=chi2_bins, color='grey', normed=True)
            py.plot(chi2_mod_bins, chi2_plot, 'k--')

            if tt == 0:
                py.ylabel('PDF for O={0:d}'.format(orders[ii]))
            if ii == 2:
                py.xlabel(r'$\chi^2$')
            if ii == 0:
                py.title(targets[tt].upper())

            plot_ii += 1

            py.xlim(0, 30)
            py.ylim(0, 0.40)

    py.savefig(plot_dir + 'align_order_all_chi2_a.pdf')

    return    


def plot_align_order(target, only_stars_in_fit=True, Kcut=22):
    root_dir = '/Users/jlu/work/microlens/2015_evan/'

    # Make 2D arrays for the runs with different Kcut and orders.
    Kcuts = np.array([18, 20, 22])
    orders = np.array([1, 2, 3])

    runs = {18: {1: 'ad', 2: 'ap', 3: 'bb'},
            20: {1: 'ah', 2: 'at', 3: 'bf'},
            22: {1: 'al', 2: 'ax', 3: 'bj'}}

    flags = [runs[Kcut][1], runs[Kcut][2], runs[Kcut][3]]
    
    analysis_root_fmt = 'analysis_{0}_2014_03_22{1}_MC100_omit_1/'

    analysis_roots = []
    align_roots = []
    
    for ii in range(len(flags)):
        analysis_roots.append( root_dir + analysis_root_fmt.format(target, flags[ii]) )
        align_roots.append( analysis_root_fmt.format(target, flags[ii]) + 'align/align_t' )

    out_suffix = 'K{0:2d}_{1:s}'.format(Kcut, target)
    
    align_residuals_vs_order(analysis_roots,
                             only_stars_in_fit=only_stars_in_fit,
                             out_suffix=out_suffix)
    
    for aa in range(len(analysis_roots)):
        outfile = plot_dir
        outfile += 'chi2_res_dist_{0:s}_K{1:d}_M{2:d}'.format(target, Kcut, orders[aa])
        if only_stars_in_fit:
            outfile += '_infit'
        outfile += '.png'
        print outfile
        
        residuals.chi2_dist_all_epochs('align/align_t', root_dir=analysis_roots[aa],
                                       only_stars_in_fit=only_stars_in_fit,
                                       plotfile=outfile)

    return    


    
def plot_alignment_errors(target, radius=4):
    align = 'align/align'

    # Get the analysis directory to work on.    
    ff = final_targets.index(target)
    rootDir = analysis_dir
    rootDir += 'analysis_' + final_targets[ff] + '_2014_03_22' + final_flag[ff] + '_MC100_omit_1/'
    
    if target == 'ob110022': 
        epochs = ['2011 May','2011 Jul','2011 Jun','2012 Jul','2013 Apr','2013 Jul']
    if target == 'ob110125': 
        epochs = ['2012 May','2012 Jun', '2012 Jul','2013 Apr','2013 Jul']
    if target == 'ob120169': 
        epochs = ['2012 May','2012 Jun', '2012 Jul','2013 Apr','2013 Jul']
        
    # Assume this is NIRC2 data.
    scale = 9.952  # mas / pixel
    
    # Load the align and polyfit results for all stars.
    s = starset.StarSet(rootDir + align)
    s.loadStarsUsed()
    
    x = s.getArrayFromAllEpochs('xpix') * scale
    y = s.getArrayFromAllEpochs('ypix') * scale
    xe_p = s.getArrayFromAllEpochs('xpixerr_p') * scale
    ye_p = s.getArrayFromAllEpochs('ypixerr_p') * scale
    xe_a = s.getArrayFromAllEpochs('xpixerr_a') * scale
    ye_a = s.getArrayFromAllEpochs('ypixerr_a') * scale
    isUsed = s.getArrayFromAllEpochs('isUsed')

    m = s.getArray('mag')
    name = s.getArray('name')
    cnt = s.getArray('velCnt')

    Nepochs = len(epochs)

    # Setup plot
    axfontsize = 18
    tickfontsize = 12

    py.close(1)
    fig = py.figure(1, figsize=(12,6))
    fig.subplots_adjust(left=0.1, right=0.95, top=0.95)
    grid = Grid(fig, 111,
                nrows_ncols=(2, 3),
                axes_pad=0.1,
                add_all=False,
                label_mode='L')

    # Calculate the radius of each star.
    xhalf = x.max() / 2.0
    yhalf = y.max() / 2.0
    r = np.hypot(x - xhalf, y - yhalf)

    # Combine x and y errors (mean)    
    err_p = (xe_p + ye_p) / 2.0
    err_a = (xe_a + ye_a) / 2.0

    err_p_median = np.zeros(Nepochs, dtype=float)
    err_a_median = np.zeros(Nepochs, dtype=float)

    for ee in range(Nepochs):
        # Find the target
        if target == 'ob120169':
            target_name = 'ob120169_R'
        else:
            target_name = target
            
        ttt = np.where([nn == target_name for nn in name])[0][0]
        ##########
        #
        # Plot alignment errors
        #
        ##########
        idx = (np.where(r[ee, :] < (radius*1e3)))[0]

        fig.add_axes(grid[ee])
        grid[ee].semilogy(m[idx], err_p[ee, idx], 'b.')
        grid[ee].semilogy(m[ttt], err_p[ee, ttt], 'b.', ms=10, label='Pos')
        grid[ee].semilogy(m[idx], err_a[ee, idx], 'r.')
        grid[ee].semilogy(m[ttt], err_a[ee, ttt], 'r.', ms=10, label='Aln')
        
        grid[ee].axis([12, 23, 2e-2, 30.0])
        grid[ee].axhline(0.15, color='g') 
        grid[ee].xaxis.label.set_fontsize(tickfontsize)
        grid[ee].yaxis.label.set_fontsize(tickfontsize)
        if ee == 2:
            grid[ee].legend(numpoints=1)
        
        if ee > 1:
            if (target == 'ob110022') and ee == 1:
                pass
            else:
                grid[ee].set_xlabel('K Magnitude',
                                    fontsize= axfontsize)

        # Write Epoch
        grid[ee].text(13, 10, epochs[ee], fontsize=axfontsize, color='k')

        print ''
        print 'Epoch: ', epochs[ee], ' -- errors on stars with r<4" and m<19'
        print '  N_stars = {0:3d}'.format(isUsed[ee, :].sum())

        idx = (np.where((r[ee, :] < (radius*1e3)) & (m < 19)))[0]
        fmt = '  mean err_{0:s} = {1:5.2f} mas     median err_{0:s} = {2:5.2f} mas'
        print fmt.format('p', err_p[ee, idx].mean(), np.median(err_p[ee, idx]))
        print fmt.format('a', err_a[ee, idx].mean(), np.median(err_a[ee, idx]))

        err_p_median[ee] = np.median(err_p[ee, idx])
        err_a_median[ee] = np.median(err_a[ee, idx])
        
    py.figtext(0.02, 0.55, 'Alignment Uncertainty (mas)', fontsize=axfontsize,
                rotation='vertical', verticalalignment='center')
    py.savefig(plot_dir + 'plotPosAlignError_%s.png' % (target))
            

    return err_p_median, err_a_median

def calc_aberration(target):
    if target.lower() == 'ob110022':
        ra = '17:53:17.93'
        dec = '-30:02:29.3'
        dec2 = '-30:02:14.3'
        dec3 = '-30:02:19.3'
    elif target.lower() == 'ob110125':
        ra = '18:03:32.95'
        dec = '-29:49:43.0'
        dec2 = '-29:49:38.0'
        dec3 = '-29:49:33.0'
    elif target.lower() == 'ob120169':
        ra = '17:49:51.38'
        dec = '-35:22:28.0'
        dec2 = '-35:22:23.0'
        dec3 = '-35:22:18.0'

    dates = ['2011/5/25 12:00:00', '2011/7/7 9:00:00',
             '2012/5/23 12:00:00', '2012/6/23 10:00:00', '2012/7/10 9:00:00',
             '2013/4/30 13:30:00', '2013/7/15 8:30:00']
        
    keck = ephem.Observer()
    keck.lat = '19:49:28'
    keck.long = '-155:28:24'
    keck.elevation = 4154.0
    keck.pressure = 0

    star1 = ephem.FixedBody()
    star1._ra = ephem.hours(ra)
    star1._dec = ephem.degrees(dec)

    star2 = ephem.FixedBody()
    star2._ra = ephem.hours(ra)
    star2._dec = ephem.degrees(dec2)

    star3 = ephem.FixedBody()
    star3._ra = ephem.hours(ra)
    star3._dec = ephem.degrees(dec3)
            
    for ii in range(len(dates)):
        keck.date = dates[ii]

        star1.compute(keck)
        star2.compute(keck)
        star3.compute(keck)

        sep12 = ephem.separation((star1.ra, star1.dec), (star2.ra, star2.dec))
        sep12 *= (180.0 / math.pi) * 3600.0 # convert to arcsec
        
        sep13 = ephem.separation((star1.ra, star1.dec), (star3.ra, star3.dec))
        sep13 *= (180.0 / math.pi) * 3600.0 # convert to arcsec

        sep_orig2 = ephem.separation((star1.a_ra, star1.a_dec), (star2.a_ra, star2.a_dec))
        sep_orig2 *= (180.0 / math.pi) * 3600.0 # convert to arcsec

        sep_orig3 = ephem.separation((star1.a_ra, star1.a_dec), (star3.a_ra, star3.a_dec))
        sep_orig3 *= (180.0 / math.pi) * 3600.0 # convert to arcsec
                
        delta12 = (sep12 - sep_orig2) * 1e3  # in milliarcseconds
        delta13 = (sep13 - sep_orig3) * 1e3  # in milliarcseconds
        
        print '{0:20s}  Delta(10") = {1:6.3f} mas  Delta(5") = {1:6.3f} mas'.format(dates[ii], delta12, delta13)

        
def plot_align_weight_final():
    root_dir = analysis_dir

    targets = final_targets
    weights = np.array([1, 2, 3, 4])
    orders = {'ob110022': final_order[0], 'ob110125': final_order[1], 'ob120169': final_order[2]}
    
    # runs dict = {order: {weight: flag}}
    runs = {1: {1: 'ai', 2: 'aj', 3: 'ak', 4: 'al'},
            2: {1: 'au', 2: 'av', 3: 'aw', 4: 'ax'}}

    analysis_root_fmt = 'analysis_{0}_2014_03_22{1}_MC100_omit_1/'
    
    ##########
    # Load up data for all target/order combinations.
    ##########

    chi2 = {}
    resi = {}
    sigm = {}
    ndof = {}
    chi2a = {}
    sigma = {}
            
    for tt in range(len(targets)):
        order = orders[targets[tt]]
        flags = [runs[order][1], runs[order][2], runs[order][3], runs[order][4]]
    
        chi2[targets[tt]] = {}
        resi[targets[tt]] = {}
        sigm[targets[tt]] = {}
        ndof[targets[tt]] = {}
        chi2a[targets[tt]] = {}
        sigma[targets[tt]] = {}
        
        for ii in range(len(weights)):
            analysis_root = analysis_root_fmt.format(targets[tt], flags[ii])

            out = residuals.chi2_dist_all_epochs('align/align_t', root_dir=root_dir + analysis_root,
                                                 only_stars_in_fit=False,
                                                 plotfile=None)

            xe_ap = np.hypot(out['xe_p'], out['xe_a'])
            ye_ap = np.hypot(out['ye_p'], out['ye_a'])

            chi2x_a = ((out['xres'] / xe_ap)**2).sum(axis=0)
            chi2y_a = ((out['yres'] / ye_ap)**2).sum(axis=0)
            
            chi2[targets[tt]][weights[ii]] = np.concatenate([out['chi2x'], out['chi2y']])
            resi[targets[tt]][weights[ii]] = np.concatenate([out['xres'], out['yres']])
            sigm[targets[tt]][weights[ii]] = np.concatenate([out['xres'] / out['xe_p'],
                                                            out['yres'] / out['ye_p']])
            chi2a[targets[tt]][weights[ii]] = np.concatenate([chi2x_a, chi2y_a])
            sigma[targets[tt]][weights[ii]] = np.concatenate([out['xres'] / xe_ap,
                                                             out['yres'] / ye_ap])
            ndof[targets[tt]][weights[ii]] = out['Ndof_x']

    ##########
    # Plot Residuals with only positional error.
    ##########
    plot_ii = 0

    py.figure(1)
    py.clf()
    fig = py.gcf()
    fig.set_size_inches(12, 16, forward=True)
    py.subplots_adjust(bottom=0.08, left=0.08, top=0.95, wspace=0.28, hspace=0.23)

    sig_bin = 0.5
    sig_bins = np.arange(-6, 6, sig_bin)
    sig_mod_bin = 0.05
    sig_mod_bins = np.arange(-6, 6, sig_mod_bin)
    sig_plot = scipy.stats.norm.pdf(sig_mod_bins)
    
    for ii in range(len(weights)):
        for tt in range(len(targets)):
            sig = sigm[targets[tt]][weights[ii]].flatten()

            py.subplot(4, 3, plot_ii+1)
            py.hist(sig, bins=sig_bins, color='grey', normed=True)
            py.plot(sig_mod_bins, sig_plot, 'k--')

            if tt == 0:
                py.ylabel('PDF for W={0:d}'.format(weights[ii]))
            if ii == 3:
                py.xlabel(r'[p$_{obs}$ - p$_{fit}$] / $\sigma_p$')
            if ii == 0:
                py.title(targets[tt].upper())

            plot_ii += 1

            py.xlim(-6, 6)
            py.ylim(0, 0.5)
            
    py.savefig(plot_dir + 'align_weight_all_res_sig.pdf')

    ##########
    # Plot Chi^2 with only positional error.
    ##########
    plot_ii = 0
    
    py.figure(2)
    py.clf()
    fig = py.gcf()
    fig.set_size_inches(12, 16, forward=True)
    py.subplots_adjust(bottom=0.08, left=0.08, top=0.95, wspace=0.28, hspace=0.23)

    chi2_bin = 4.0
    chi2_bins = np.arange(0, 50, chi2_bin)
    chi2_mod_bin = 0.25
    chi2_mod_bins = np.arange(0, 100, chi2_mod_bin)
    
    for ii in range(len(weights)):
        for tt in range(len(targets)):
            chi = chi2[targets[tt]][weights[ii]]
            chi2_plot = scipy.stats.chi2.pdf(chi2_mod_bins, ndof[targets[tt]][weights[ii]])
            chi2_plot /= 3.0  # arbitrary

            py.subplot(4, 3, plot_ii + 1)
            py.hist(chi, bins=chi2_bins, color='grey', normed=True)
            py.plot(chi2_mod_bins, chi2_plot, 'k--')

            if tt == 0:
                py.ylabel('PDF for W={0:d}'.format(weights[ii]))
            if ii == 3:
                py.xlabel(r'$\chi^2$')
            if ii == 0:
                py.title(targets[tt].upper())

            plot_ii += 1

            py.xlim(0, 50)
            py.ylim(0, 0.10)

    py.savefig(plot_dir + 'align_weight_all_chi2.pdf')

    ##########
    # Plot Residuals with positional and alignment error.
    ##########
    plot_ii = 0

    py.figure(3)
    py.clf()
    fig = py.gcf()
    fig.set_size_inches(12, 16, forward=True)
    py.subplots_adjust(bottom=0.08, left=0.08, top=0.95, wspace=0.28, hspace=0.23)

    sig_bin = 0.5
    sig_bins = np.arange(-6, 6, sig_bin)
    sig_mod_bin = 0.05
    sig_mod_bins = np.arange(-6, 6, sig_mod_bin)
    sig_plot = scipy.stats.norm.pdf(sig_mod_bins)
    
    for ii in range(len(weights)):
        for tt in range(len(targets)):
            sig = sigma[targets[tt]][weights[ii]].flatten()

            py.subplot(4, 3, plot_ii+1)
            py.hist(sig, bins=sig_bins, color='grey', normed=True)
            py.plot(sig_mod_bins, sig_plot, 'k--')

            if tt == 0:
                py.ylabel('PDF for W={0:d}'.format(weights[ii]))
            if ii == 3:
                py.xlabel(r'[p$_{obs}$ - p$_{fit}$] / $\sigma_p$')
            if ii == 0:
                py.title(targets[tt].upper())

            plot_ii += 1

            py.xlim(-6, 6)
            py.ylim(0, 0.5)
            
    py.savefig(plot_dir + 'align_weight_all_res_sig_a.pdf')

    ##########
    # Plot Chi^2 with only positional error.
    ##########
    plot_ii = 0

    py.figure(4)
    py.clf()
    fig = py.gcf()
    fig.set_size_inches(12, 16, forward=True)
    py.subplots_adjust(bottom=0.08, left=0.08, top=0.95, wspace=0.28, hspace=0.23)

    chi2_bin = 3.0
    chi2_bins = np.arange(0, 50, chi2_bin)
    chi2_mod_bin = 0.25
    chi2_mod_bins = np.arange(0, 100, chi2_mod_bin)
    
    for ii in range(len(weights)):
        for tt in range(len(targets)):
            chi = chi2a[targets[tt]][weights[ii]]
            chi2_plot = scipy.stats.chi2.pdf(chi2_mod_bins, ndof[targets[tt]][weights[ii]])
            chi2_plot /= 1.5  # arbitrary

            py.subplot(4, 3, plot_ii + 1)
            py.hist(chi, bins=chi2_bins, color='grey', normed=True)
            py.plot(chi2_mod_bins, chi2_plot, 'k--')

            if tt == 0:
                py.ylabel('PDF for W={0:d}'.format(weights[ii]))
            if ii == 3:
                py.xlabel(r'$\chi^2$')
            if ii == 0:
                py.title(targets[tt].upper())

            plot_ii += 1

            py.xlim(0, 50)
            py.ylim(0, 0.20)

    py.savefig(plot_dir + 'align_weight_all_chi2_a.pdf')


    ##########
    # PAPER PLOT: Plot Residuals with only positional error for OB110022.
    ##########
    plot_ii = 0

    py.figure(5)
    py.clf()
    fig = py.gcf()
    fig.set_size_inches(16, 4, forward=True)
    py.subplots_adjust(bottom=0.2, left=0.05, right=0.98, top=0.9, wspace=0, hspace=0.24)

    sig_bin = 0.5
    sig_bins = np.arange(-6, 6, sig_bin)
    sig_mod_bin = 0.05
    sig_mod_bins = np.arange(-6, 6, sig_mod_bin)
    sig_plot = scipy.stats.norm.pdf(sig_mod_bins)
    
    for ii in range(len(weights)):
        tt = 0
        sig = sigm[targets[tt]][weights[ii]].flatten()

        py.subplot(1, len(weights), plot_ii+1)
        py.hist(sig, bins=sig_bins, color='grey', normed=True)
        py.plot(sig_mod_bins, sig_plot, 'k--')

        if ii == 0:
            py.ylabel('PDF')
        else:
            py.gca().get_yaxis().set_visible(False)
                
        py.xlabel(r'[p$_{obs}$ - p$_{fit}$] / $\sigma_p$')
        py.title("W = {0:d}".format(weights[ii]))

        plot_ii += 1

        py.xlim(-6, 5.9)
        py.ylim(0, 0.5)
            
    py.savefig(plot_dir + 'align_weight_OB110022_res_sig.pdf')
    
    return    
    
def calc_average_pos_aln_error():
    """
    Calculate the average positional + alignment error
    for each of the targets for all orders. 
    """
    align_root_fmt = 'analysis_{0}_2014_03_22{1}_MC100_omit_1/align/align_t'
    points_root_fmt = 'analysis_{0}_2014_03_22{1}_MC100_omit_1/points_d/'

    # Assume this is NIRC2 data.
    scale = 9.952  # mas / pixel
    
    for tt in range(len(final_targets)):
        target = final_targets[tt]
        
        points_root = points_root_fmt.format(target, final_flag[tt])
        align_root = align_root_fmt.format(target, final_flag[tt])
        
        s = starset.StarSet(analysis_dir + align_root)
        name = s.getArray('name')
        xerr_p = s.getArrayFromAllEpochs('xpixerr_p') * scale
        yerr_p = s.getArrayFromAllEpochs('ypixerr_p') * scale
        xerr_a = s.getArrayFromAllEpochs('xpixerr_a') * scale
        yerr_a = s.getArrayFromAllEpochs('ypixerr_a') * scale

        # Find the target
        if target == 'ob120169':
            target_name = 'ob120169_R'
        else:
            target_name = target
            
        ttt = np.where([nn == target_name for nn in name])[0][0]

        xerr = xerr_p[:, ttt] + xerr_a[:, ttt]
        yerr = yerr_p[:, ttt] + yerr_a[:, ttt]
        err = (xerr + yerr) / 2.0
            
        fmt = '{0:10s} Kcut={1:2d} O={2:1d}:  xerr = {3:4.2f} mas   yerr = {4:4.2f}   err = {5:4.2f} mas'
        print fmt.format(target, final_Kcut[tt], final_order[tt], xerr.mean(), yerr.mean(), err.mean())

    return

def calc_chi2_lens_fit():
    analysis_dir_fmt = 'analysis_{0}_2014_03_22{1}_MC100_omit_1/'

    # Assume this is NIRC2 data.
    pixScale = 9.952  # mas / pixel

    for tt in range(len(final_targets)):
        target = final_targets[tt]
        order = final_order[tt]

        if target == 'ob110125':
            continue

        if target == 'ob120169':
            targname = 'ob120169_R'
        else:
            targname = target

        analysis_dir_tt = analysis_dir_fmt.format(final_targets[tt], final_flag[tt])

        #Plot astrometric observations
        pointsFile = analysis_dir + analysis_dir_tt + 'points_d/' + targname + '.points'
        pointsTab = Table.read(pointsFile, format='ascii')
        tobs = pointsTab['col1']
        xpix = pointsTab['col2']
        ypix = pointsTab['col3']
        xerr = pointsTab['col4']
        yerr = pointsTab['col5']

        thetaSx_data = xpix * pixScale
        thetaSy_data = ypix * pixScale
        xerr_data = xerr * pixScale
        yerr_data = yerr * pixScale

        # Load Best-Fit Microlensing Model
        for mm in range(len(mnest_runs[target][order])):
            mnest_run = mnest_runs[target][order][mm]
            
            mnest_dir = 'multiNest/' + mnest_run + '/'
            mnest_root = mnest_run
            tab = load_mnest_results(root_dir=analysis_dir + analysis_dir_tt,
                                     mnest_dir=mnest_dir,
                                     mnest_root=mnest_root)
        
            print '**********'
            print '* ' + target + ' O=' + str(order) + ' mnest=' + mnest_run
            print '**********'
            
            params = tab.keys()
            pcnt = len(params)
            Masslim = 0.0

            # Clean table and only keep valid results.
            ind = np.where((tab['Mass'] >= Masslim))[0]
            tab = tab[ind]

            best = np.argmax(tab['logLike'])
            maxLike = tab['logLike'][best]
            print 'Best-Fit Solution:'
            print '  Index = {0:d}'.format(best)
            print '  Log(L) = {0:6.2f}'.format(maxLike)
            print '  Params:'
            for i in range(pcnt):
                print '     {0:15s}  {1:10.3f}'.format(params[i], tab[params[i]][best])

            t0 = tab['t0'][best]
            tE = tab['tE'][best]
            thetaS0x = tab['thetaS0x'][best]
            thetaS0y = tab['thetaS0y'][best]
            muSx = tab['muSx'][best]
            muSy = tab['muSy'][best]
            muRelx = tab['muRelx'][best]
            muRely = tab['muRely'][best]
            beta = tab['beta'][best]
            piEN = tab['piEN'][best]
            piEE = tab['piEE'][best]
            print '     muLx = {0:5.2f} mas/yr'.format(muSx - muRelx)
            print '     muLy = {0:5.2f} mas/yr'.format(muSy - muRely)


            ##########
            # Get astrometry for best-fit model. Do this on a fine time grid
            # and also at the points of the observations.
            ##########
            tmod = np.arange(t0-20.0, t0+20.0, 0.01)
            model = MCMC_LensModel.LensModel_Trial1(tmod, t0, tE, [thetaS0x, thetaS0y],
                                                    [muSx,muSy], [muRelx, muRely],
                                                    beta, [piEN, piEE])
            model_tobs = MCMC_LensModel.LensModel_Trial1(tobs, t0, tE, [thetaS0x, thetaS0y],
                                                        [muSx,muSy], [muRelx, muRely],
                                                        beta, [piEN, piEE])

            thetaS_model = model[0]
            thetaE_amp = model[1]
            M = model[2]
            shift = model[3]
            thetaS_nolens = model[4]

            thetaS_model_tobs = model_tobs[0]
            thetaE_amp_tobs = model_tobs[1]
            M_tobs = model_tobs[2]
            shift_tobs = model_tobs[3]
            thetaS_nolens_tobs = model_tobs[4]

            thetaSx_model = thetaS_model[:,0]
            thetaSy_model = thetaS_model[:,1]
            thetaS_nolensx = thetaS_nolens[:,0]
            thetaS_nolensy = thetaS_nolens[:,1]
            shiftx = shift[:,0]
            shifty = shift[:,1]

            thetaSx_model_tobs = thetaS_model_tobs[:,0]
            thetaSy_model_tobs = thetaS_model_tobs[:,1]

            ##########
            # Calculate Chi-Squared Star for comparison to velocity-only fit.
            ##########
            # Find the model points closest to the data.
            dx = thetaSx_data - thetaSx_model_tobs
            dy = thetaSy_data - thetaSy_model_tobs
            chi2_x = ((dx / xerr_data)**2).sum()
            chi2_y = ((dy / yerr_data)**2).sum()
            N_dof = (2*len(dx)) - 9
            chi2_red = (chi2_x + chi2_y) / N_dof
            
            print 'Chi-Squared of Lens Model Fit:'
            print '        X Chi^2 = {0:5.2f}'.format(chi2_x)
            print '        Y Chi^2 = {0:5.2f}'.format(chi2_y)
            print '    Total Chi^2 = {0:5.2f} ({1:d} DOF)'.format(chi2_x + chi2_y, N_dof)
            print '  Reduced Chi^2 = {0:5.2f}'.format(chi2_red)
            print ''
            print ''

    return

def plot_final_chi2_distribution():
    analysis_dir_fmt = 'analysis_{0}_2014_03_22{1}_MC100_omit_1/'
    align = 'align/align_t'
    poly = 'polyfit_d/fit'
    points = 'points_d/'

    # Assume this is NIRC2 data.
    pixScale = 9.952  # mas / pixel

    ##########
    # Load up data for all targets.
    ##########

    chi2_1 = {}
    chi2_2 = {}
    ndof_1 = {}
    ndof_2 = {}
    limits = [0.25, 0.30, 0.5]
            
    for tt in range(len(final_targets)):
        target = final_targets[tt]
        
        new_root = analysis_dir + analysis_dir_fmt.format(final_targets[tt], final_flag[tt])

        # out = residuals.chi2_dist_all_epochs(align, root_dir=new_root,
        #                                      only_stars_in_fit=False,
        #                                      plotfile=None)

        # Load up the list of stars used in the transformation.
        s = starset.StarSet(new_root + align)
        s.loadStarsUsed()
        s.loadPolyfit(new_root + poly, accel=0, arcsec=0)

        name = s.getArray('name')
        cnt = s.getArray('velCnt')
        
        # Find the target
        if target == 'ob120169':
            target_name = 'ob120169_R'
        else:
            target_name = target
            
        chi2red_x = s.getArray('fitXv.chi2red')
        chi2red_y = s.getArray('fitYv.chi2red')
        chi2_x = s.getArray('fitXv.chi2')
        chi2_y = s.getArray('fitYv.chi2')

        idx = np.where(cnt == cnt.max())[0]
        
        # chi2_1[target] = chi2_x[idx] + chi2_y[idx]
        # ndof_1[target] = (chi2_x[idx] / chi2red_x[idx]) + (chi2_y[idx] / chi2red_y[idx])
        chi2_1[target] = np.concatenate([chi2_x[idx], chi2_y[idx]])
        ndof_1[target] = np.concatenate([(chi2_x[idx] / chi2red_x[idx]), (chi2_y[idx] / chi2red_y[idx])])

        # # Alternative from the residuals        
        # xe_ap = np.hypot(out['xe_p'], out['xe_a'])
        # ye_ap = np.hypot(out['ye_p'], out['ye_a'])
        # chi2x_a = ((out['xres'] / xe_ap)**2).sum(axis=0)
        # chi2y_a = ((out['yres'] / ye_ap)**2).sum(axis=0)

        # chi2_2[target] = chi2x_a + chi2y_a
        # ndof_2[target] = out['Ndof_x'] + out['Ndof_y']

    py.figure(1)
    py.clf()
    fig = py.gcf()
    fig.set_size_inches(6, 12, forward=True)
    py.subplots_adjust(bottom=0.08, left=0.18, top=0.95, wspace=0.28, hspace=0.15)

    chi2_bin = 1.5
    chi2_bins = np.arange(0, 50, chi2_bin)
    chi2_mod_bin = 0.25
    chi2_mod_bins = np.arange(0, 100, chi2_mod_bin)
    
    for tt in range(len(final_targets)):
        target = final_targets[tt]
        chi = chi2_1[target]
        chi2_plot = scipy.stats.chi2.pdf(chi2_mod_bins, ndof_1[target].max())

        py.subplot(3, 1, tt+1)
        py.hist(chi, bins=chi2_bins, color='grey', normed=True)
        py.plot(chi2_mod_bins, chi2_plot, 'k--')

        if tt == 2:
            py.xlabel(r'$\chi^2_{vel}$')
            
        py.ylabel('PDF')
        py.xlim(0, 30)
        py.ylim(0, limits[tt])

    for tt in range(len(final_targets)):
        # Calculate KS test.
        target = final_targets[tt]
        ks_p = stats.kstest(chi2_1[target], lambda x: stats.chi2.cdf(x, ndof_1[target].max()))
        print 'KS Test Results for {0:s}: D = {1:4.2f}  p = {2:8.2e}'.format(target, ks_p[0], ks_p[1])

    for tt in range(len(final_targets)):
        target = final_targets[tt]
        idx = np.where(chi2_1[target] < 15)[0]
        ks_p = stats.kstest(chi2_1[target][idx], lambda x: stats.chi2.cdf(x, ndof_1[target].max()))
        print 'KS Test Results for {0:s} trimmed: D = {1:4.2f}  p = {2:8.2e}'.format(target, ks_p[0], ks_p[1])
        

    py.savefig(plot_dir + 'hist_chi2_final.pdf')
        

    return
    

def table_of_data(target):
    # Assume this is NIRC2 data.
    scale = 9.952  # mas / pixel
    
    align = 'align/align_t'
    poly = 'polyfit_d/fit'
    points = 'points_d/'

    # Get the analysis directory to work on.    
    ff = final_targets.index(target)
    rootDir = analysis_dir
    rootDir += 'analysis_' + final_targets[ff] + '_2014_03_22' + final_flag[ff] + '_MC100_omit_1/'

    # Load the align and polyfit results for all stars.
    s = starset.StarSet(rootDir + align)
    s.loadPolyfit(rootDir + poly, accel=0, arcsec=0)
    
    names = s.getArray('name')

    # Find the target and fetch the points measurements.
    if target == 'ob120169':
        star_name='ob120169_R'
    else:
        star_name = target
        
    ii = names.index(star_name)
    star = s.stars[ii]

    # Get the positional and alignment errors (separately)
    xe_p = star.getArrayAllEpochs('xpixerr_p') * scale
    ye_p = star.getArrayAllEpochs('ypixerr_p') * scale
    xe_a = star.getArrayAllEpochs('xpixerr_a') * scale
    ye_a = star.getArrayAllEpochs('ypixerr_a') * scale

    # Get the position, convert to relative arcseoconds in arbitrary coordinate system.
    x = star.getArrayAllEpochs('xpix')
    y = star.getArrayAllEpochs('ypix')
    xmean = np.mean(x)
    ymean = np.mean(y)
    dx = (x - xmean) * -1.0 * scale
    dy = (y - ymean) * scale

    # Get the dates. Convert back to MJD. See algorithm in calc_year.pro
    year = star.getArrayAllEpochs('t')
    d_days = (year - 1999.0) * 365.242
    mjd = d_days + 51179.0

    # Get the brightness
    mag = star.getArrayAllEpochs('mag')

    ##########
    # Print out latex table.
    ##########
    outfile = plot_dir + 'tab_data_' + target + '.tex'
    _out = open(outfile, 'w')
    _out.write('% Code to make table:\n')
    _out.write('% jlu_python/\n')
    _out.write('% from jlu.microlens import paper_2015\n')
    _out.write('% paper_2015.table_of_data(target="' + target + '")\n')
    _out.write('%\n')
    _out.write('\\begin{deluxetable}{lrrrcccc}\n')
    _out.write('\\tabletypesize{\\footnotesize}\n')
    _out.write('\\tablecaption{' + target.upper() + ' Measurements \label{tab:obs_' + target + '}}\n')
    _out.write('\\tablehead{\n')
    _out.write('MJD & Kp & $\Delta\\xpos$ & $\Delta\\ypos$ & \n')
    _out.write('   \multicolumn{2}{c}{$\sigma_{\mathrm{\\x}}$} & \multicolumn{2}{c}{$\sigma_{\mathrm{\\y}}$} \\\\\n')
    _out.write(' & & [mas] & [mas] & \multicolumn{2}{c}{[mas]} & \multicolumn{2}{c}{[mas]} \\\\\n')
    _out.write(' & & & & pos & aln & pos & aln \n')
    _out.write('}\n')
    _out.write('\startdata\n')
    fmt = '{0:8.3f} & {1:5.1f} & {2:8.2f} & {3:8.2f} & {4:5.2f} & {5:5.2f} & {6:5.2f} & {7:5.2f} {8:4s}'
    for ii in range(len(mjd)):
        if (ii == (len(mjd)-1)):
            ender = '\n'
        else:
            ender = '\\\\ \n'
        _out.write(fmt.format(mjd[ii], mag[ii], dx[ii], dy[ii], xe_p[ii], xe_a[ii], ye_p[ii], ye_a[ii], ender))
    _out.write('\\enddata\n')
    _out.write('\\tablenotetext{}{}\n')
    _out.write('\end{deluxetable}\n')

    _out.close()

    return
    
def table_ao_observations():
    """Make Table 1 with the details of the observations."""

    outfile = plot_dir + 'tab_ao_obs.tex'
    
    _out = open(outfile, 'w')
    _out.write('% Code to make figure:\n')
    _out.write('% jlu_python/\n')
    _out.write('% from jlu.microlens import paper_2015\n')
    _out.write('% paper_2015.table_ao_observations()\n')
    _out.write('%\n')
    _out.write('\\begin{deluxetable*}{lrrrrrrrrr}\n')
    _out.write('\\tabletypesize{\\footnotesize}\n')
    _out.write('\\tablewidth{0pt}\n')
    _out.write('\\tablecaption{AO Observations}\n')
    _out.write('\\tablehead{\n')
    _out.write('Event  & RA (J2000) & Dec (J2000) & Date & $N_{\mathrm{exp}}$ & $N_{\star}$  & \n')
    _out.write('Strehl &  FWHM & $\sigma_{\mathrm{pos}}$ & $\sigma_{\mathrm{aln}}$ \\\\ \n')
    _out.write('& [hr] & [deg] & [UT] & & & & [mas] & [mas] & [mas]\n')
    _out.write('}\n')
    _out.write('\\startdata\n')
    _out.write('\\\\\n')

    targ_name = {'ob110022': 'OB110022', 'ob110125': 'OB110125', 'ob120169': 'OB120169'}
    ra  = {'ob110022': '17:53:17.93', 'ob110125': '18:03:32.95', 'ob120169': '17:49:51.38'}
    dec = {'ob110022': '-30:02:29.3', 'ob110125': '-29:49:43.0', 'ob120169': '-35:22:28.0'}
    date = {'ob110022': ['May 25, 2011', 'July 7, 2011', 'June 23, 2012', 'July 10, 2012', 'April 30, 2013', 'July 15, 2013'],
            'ob110125': ['May 23, 2012', 'June 23, 2012', 'July 10, 2012', 'April 30, 2013', 'July 15, 2013'],
            'ob120169': ['May 23, 2012', 'June 23, 2012', 'July 10, 2012', 'April 30, 2013', 'July 15, 2013']}
    Nexp = {'ob110022': [27, 16, 40, 34, 22, 30],
            'ob110125': [21, 33, 18, 48, 39],
            'ob120169': [ 5, 10, 22, 31, 11]}
    Nstar = {'ob110022': [285, 178, 701, 717, 485, 636],
             'ob110125': [104, 327, 221, 332, 329],
             'ob120169': [ 35, 122, 192, 207,  84]}
    strehl = {'ob110022': [0.14, 0.13, 0.24, 0.26, 0.24, 0.34],
              'ob110125': [0.10, 0.36, 0.21, 0.29, 0.36],
              'ob120169': [0.10, 0.24, 0.29, 0.29, 0.26]}
    fwhm = {'ob110022': [ 91, 69, 70, 68, 71, 60],
            'ob110125': [ 96, 57, 70, 64, 57],
            'ob120169': [110, 69, 64, 61, 74]}

    for tt in range(len(final_targets)):
        target = final_targets[tt]
        
        err_p, err_a = plot_alignment_errors(target)

        for ee in range(len(date[target])):
            if ee == 0:
                _out.write('{0:s} & {1:s} & {2:s} \n'.format(targ_name[target], ra[target], dec[target]))
                _out.write('    & ')
            else:
                _out.write('& & & ')
                
            
            fmt = '{0:15s} & {1:2d} & {2:3d} & {3:4.2f} & {4:3d} & {5:4.2f} & {6:4.2f} \\\\ \n'
            _out.write(fmt.format(date[target][ee], Nexp[target][ee], Nstar[target][ee],
                                  strehl[target][ee], fwhm[target][ee], err_p[ee], err_a[ee]))
            
        if tt != (len(final_targets) - 1):
            _out.write('\\\\ \n')
                            
    _out.write('\\enddata\n')
    _out.write('\\tablenotetext{}{$N_{\star}$: Number of stars detected. Strehl and\n')
    _out.write('FWHM are the average values over all individual\n')
    _out.write('exposures. $\sigma_{\mathrm{pos}}$ and $\sigma_{\mathrm{aln}}$ are\n')
    _out.write('calculated after cross-epoch transformation from the median \n')
    _out.write('of all stars with r$<$4'' and Kp$<$19 mag. \n')
    _out.write('}\n')
    _out.write('\\label{tb:AOobs}\n')
    _out.write('\\end{deluxetable*}\n')

    return    

def plot_OB110022():
    tt = 0
    target = final_targets[tt]
    order = final_order[tt]
    display_name = target.upper()
    analysis_dir_fmt = 'analysis_{0}_2014_03_22{1}_MC100_omit_1/'
        
    root_dir = analysis_dir
    analysis_dir_tt = analysis_dir_fmt.format(final_targets[tt], final_flag[tt])
    points_dir = 'points_d/'
    mnest_run = mnest_runs[target][order][0]
    mnest_dir = analysis_dir_tt + 'multiNest/' + mnest_run + '/'
    mnest_root = mnest_run
    
    #Plot astrometric observations
    pointsFile = root_dir + analysis_dir_tt + points_dir + target + '.points'
    pointsTab = Table.read(pointsFile, format='ascii')
    tobs = pointsTab['col1']
    xpix = pointsTab['col2']
    ypix = pointsTab['col3']
    xerr = pointsTab['col4']
    yerr = pointsTab['col5']

    pixScale = 9.952
    thetaSx_data = xpix * pixScale
    thetaSy_data = ypix * pixScale
    xerr_data = xerr * pixScale
    yerr_data = yerr * pixScale

    # Overplot Best-Fit Model
    tab = load_mnest_results_OB110022(root_dir=root_dir,
                                      mnest_dir=mnest_dir,
                                      mnest_root=mnest_root)
    params = tab.keys()
    pcnt = len(params)
    Masslim = 0.0

    # Clean table and only keep valid results.
    ind = np.where((tab['Mass'] >= Masslim))[0]
    tab = tab[ind]

    best = np.argmax(tab['logLike'])
    maxLike = tab['logLike'][best]
    print 'Best-Fit Solution:'
    print '  Index = {0:d}'.format(best)
    print '  Log(L) = {0:6.2f}'.format(maxLike)
    print '  Params:'
    for i in range(pcnt):
        print '     {0:15s}  {1:10.3f}'.format(params[i], tab[params[i]][best])

    t0 = tab['t0'][best]
    tE = tab['tE'][best]
    thetaS0x = tab['thetaS0x'][best]
    thetaS0y = tab['thetaS0y'][best]
    muSx = tab['muSx'][best]
    muSy = tab['muSy'][best]
    muRelx = tab['muRelx'][best]
    muRely = tab['muRely'][best]
    beta = tab['beta'][best]
    piEN = tab['piEN'][best]
    piEE = tab['piEE'][best]
    print '     muLx = {0:5.2f} mas/yr'.format(muSx - muRelx)
    print '     muLy = {0:5.2f} mas/yr'.format(muSy - muRely)
    

    ##########
    # Get astrometry for best-fit model. Do this on a fine time grid
    # and also at the points of the observations.
    ##########
    tmod = np.arange(t0-20.0, t0+20.0, 0.01)
    model = MCMC_LensModel.LensModel_Trial1(tmod, t0, tE, [thetaS0x, thetaS0y],
                                            [muSx,muSy], [muRelx, muRely],
                                            beta, [piEN, piEE])
    model_tobs = MCMC_LensModel.LensModel_Trial1(tobs, t0, tE, [thetaS0x, thetaS0y],
                                                 [muSx,muSy], [muRelx, muRely],
                                                 beta, [piEN, piEE])
    
    thetaS_model = model[0]
    thetaE_amp = model[1]
    M = model[2]
    shift = model[3]
    thetaS_nolens = model[4]

    thetaS_model_tobs = model_tobs[0]
    thetaE_amp_tobs = model_tobs[1]
    M_tobs = model_tobs[2]
    shift_tobs = model_tobs[3]
    thetaS_nolens_tobs = model_tobs[4]
        
    thetaSx_model = thetaS_model[:,0]
    thetaSy_model = thetaS_model[:,1]
    thetaS_nolensx = thetaS_nolens[:,0]
    thetaS_nolensy = thetaS_nolens[:,1]
    shiftx = shift[:,0]
    shifty = shift[:,1]

    thetaSx_model_tobs = thetaS_model_tobs[:,0]
    thetaSy_model_tobs = thetaS_model_tobs[:,1]
    
    ##########
    # Calculate Chi-Squared Stat for comparison to velocity-only fit.
    ##########
    # Find the model points closest to the data.
    dx = thetaSx_data - thetaSx_model_tobs
    dy = thetaSy_data - thetaSy_model_tobs
    chi2_x = ((dx / xerr_data)**2).sum()
    chi2_y = ((dy / yerr_data)**2).sum()
    N_dof = (2*len(dx)) - 9
    chi2_red = (chi2_x + chi2_y) / N_dof
    print 'Chi-Squared of Lens Model Fit:'
    print '        X Chi^2 = {0:5.2f}'.format(chi2_x)
    print '        Y Chi^2 = {0:5.2f}'.format(chi2_y)
    print '  Reduced Chi^2 = {0:5.2f}'.format(chi2_red)
    
    
    ##########
    # Plotting
    ##########
    fontsize1 = 18
    fontsize2 = 14       

    py.clf()
    fig1 = py.figure(2, figsize=(10,8))
    py.subplots_adjust(left=0.1, top=0.95, hspace=0.3, wspace=0.3)
    
    paxes = py.subplot(2, 2, 1)
    originX = np.mean(thetaSx_model)
    originY = np.mean(thetaSy_model)
    py.plot(tmod-t0, thetaSx_model-originX, 'r-')
    py.plot(tmod-t0, thetaS_nolensx-originX, 'r--')
    py.errorbar(tobs-t0, thetaSx_data-originX, yerr=xerr_data, fmt='k.')
    py.ylabel(r'x$_E$ (mas)', fontsize=fontsize1)
    py.xlabel('t - t$_{\mathrm{o}}$ (yr)', fontsize=fontsize1)
    py.ylim(-10, 2)
    py.xlim(-0.1, 2.5)
    xticks = np.arange(0, 2.5, 1, dtype=int)
    py.xticks(xticks, fontsize=fontsize2)
    py.yticks(fontsize=fontsize2)

    paxes = py.subplot(2, 2, 2)
    py.plot(tmod-t0, thetaSy_model-originY, 'b-')
    py.plot(tmod-t0, thetaS_nolensy-originY, 'b--')
    py.errorbar(tobs-t0, thetaSy_data-originY, yerr=yerr_data, fmt='k.')
    py.ylabel(r'x$_N$ (mas)', fontsize=fontsize1)
    py.xlabel('t - t$_{\mathrm{o}}$ (yr)', fontsize=fontsize1)  
    py.ylim(-1, 7)
    py.xlim(-0.1, 2.5)
    py.xticks(xticks, fontsize=fontsize2)
    py.yticks(fontsize=fontsize2)
        
    paxes = py.subplot(2, 2, 3)
    n = len(tmod)
    py.plot(thetaS_nolensx - originX,  thetaS_nolensy - originY, 'g--')
    py.plot(thetaSx_model - originX, thetaSy_model - originY, 'g-')
    py.errorbar(thetaSx_data - originX,  thetaSy_data - originY, xerr=xerr_data, yerr=yerr_data, fmt='k.')
    py.ylabel(r'x$_N$ (mas)', fontsize=fontsize1)
    py.xlabel(r'x$_E$ (mas)', fontsize=fontsize1)
    py.axis('equal')
    py.ylim(-1, 7)
    py.xlim(-10, 2)
    py.xticks(fontsize=fontsize2)
    py.yticks(fontsize=fontsize2)
    py.xlim(py.xlim()[::-1])
    

    # Make a histogram of the mass using the weights. This creates the 
    # marginalized 1D posteriors.
    paxes = py.subplot(2, 2, 4)
    bins = 75
    n, bins, patch = py.hist(tab['Mass'], normed=True,
                             histtype='step', weights=tab['weights'], bins=bins, linewidth=1.5)
    bin_centers = (bins[:-1] + bins[1:]) / 2
    py.xlabel('Mass')
    xtitle = r'Lens Mass (M$_{\odot}$)'
    py.xlabel(xtitle, fontsize=fontsize1, labelpad=10)
    py.xlim(0, 2.0)
    py.ylim(0, 1.4)
    py.ylabel('Relative probability', fontsize=fontsize1, labelpad=10)
    py.xticks(fontsize=fontsize2)
    py.yticks(fontsize=fontsize2)


    outdir = root_dir + mnest_dir + 'plots/'    
    outfile =  outdir + 'plot_OB110022_data_vs_model'
    fileUtil.mkdir(outdir)
    print 'writing plot to file ' + outfile
    
    py.savefig(outfile + '.png')
    py.savefig(outfile + '.pdf')

    return

def plot_OB120169(u0minus=True):
    tt = 2
    target = final_targets[tt]
    order = final_order[tt]
    display_name = target.upper()
    analysis_dir_fmt = 'analysis_{0}_2014_03_22{1}_MC100_omit_1/'
        
    root_dir = analysis_dir
    analysis_dir_tt = analysis_dir_fmt.format(final_targets[tt], final_flag[tt])
    points_dir = 'points_d/'
    if u0minus == True:
        mnest_run = mnest_runs[target][order][0]
    else:
        mnest_run = mnest_runs[target][order][1]
    mnest_dir = analysis_dir_tt + 'multiNest/' + mnest_run + '/'
    mnest_root = mnest_run
    
    #Plot astrometric observations
    pointsFile = root_dir + analysis_dir_tt + points_dir + target + '.points'
    pointsTab = Table.read(pointsFile, format='ascii')
    tobs = pointsTab['col1']
    xpix = pointsTab['col2']
    ypix = pointsTab['col3']
    xerr = pointsTab['col4']
    yerr = pointsTab['col5']

    pixScale = 9.952
    thetaSx_data = xpix * pixScale
    thetaSy_data = ypix * pixScale
    xerr_data = xerr * pixScale
    yerr_data = yerr * pixScale

    # Overplot Best-Fit Model
    tab = load_mnest_results(root_dir=root_dir,
                             mnest_dir=mnest_dir,
                             mnest_root=mnest_root)
    params = tab.keys()
    pcnt = len(params)
    Masslim = 0.0

    # Clean table and only keep valid results.
    ind = np.where((tab['Mass'] >= Masslim))[0]
    tab = tab[ind]

    best = np.argmax(tab['logLike'])
    maxLike = tab['logLike'][best]
    print 'Best-Fit Solution:'
    print '  Index = {0:d}'.format(best)
    print '  Log(L) = {0:6.2f}'.format(maxLike)
    print '  Params:'
    for i in range(pcnt):
        print '     {0:15s}  {1:10.3f}'.format(params[i], tab[params[i]][best])

    t0 = tab['t0'][best]
    tE = tab['tE'][best]
    thetaS0x = tab['thetaS0x'][best]
    thetaS0y = tab['thetaS0y'][best]
    muSx = tab['muSx'][best]
    muSy = tab['muSy'][best]
    muRelx = tab['muRelx'][best]
    muRely = tab['muRely'][best]
    beta = tab['beta'][best]
    piEN = tab['piEN'][best]
    piEE = tab['piEE'][best]
    print '     muLx = {0:5.2f} mas/yr'.format(muSx - muRelx)
    print '     muLy = {0:5.2f} mas/yr'.format(muSy - muRely)
    

    ##########
    # Get astrometry for best-fit model. Do this on a fine time grid
    # and also at the points of the observations.
    ##########
    tmod = np.arange(t0-20.0, t0+20.0, 0.01)
    model = MCMC_LensModel.LensModel_Trial1(tmod, t0, tE, [thetaS0x, thetaS0y],
                                            [muSx,muSy], [muRelx, muRely],
                                            beta, [piEN, piEE])
    model_tobs = MCMC_LensModel.LensModel_Trial1(tobs, t0, tE, [thetaS0x, thetaS0y],
                                                 [muSx,muSy], [muRelx, muRely],
                                                 beta, [piEN, piEE])
    
    thetaS_model = model[0]
    thetaE_amp = model[1]
    M = model[2]
    shift = model[3]
    thetaS_nolens = model[4]

    thetaS_model_tobs = model_tobs[0]
    thetaE_amp_tobs = model_tobs[1]
    M_tobs = model_tobs[2]
    shift_tobs = model_tobs[3]
    thetaS_nolens_tobs = model_tobs[4]
        
    thetaSx_model = thetaS_model[:,0]
    thetaSy_model = thetaS_model[:,1]
    thetaS_nolensx = thetaS_nolens[:,0]
    thetaS_nolensy = thetaS_nolens[:,1]
    shiftx = shift[:,0]
    shifty = shift[:,1]

    thetaSx_model_tobs = thetaS_model_tobs[:,0]
    thetaSy_model_tobs = thetaS_model_tobs[:,1]
    
    ##########
    # Calculate Chi-Squared Stat for comparison to velocity-only fit.
    ##########
    # Find the model points closest to the data.
    dx = thetaSx_data - thetaSx_model_tobs
    dy = thetaSy_data - thetaSy_model_tobs
    chi2_x = ((dx / xerr_data)**2).sum()
    chi2_y = ((dy / yerr_data)**2).sum()
    N_dof = (2*len(dx)) - 9
    chi2_red = (chi2_x + chi2_y) / N_dof
    print 'Chi-Squared of Lens Model Fit:'
    print '        X Chi^2 = {0:5.2f}'.format(chi2_x)
    print '        Y Chi^2 = {0:5.2f}'.format(chi2_y)
    print '  Reduced Chi^2 = {0:5.2f}'.format(chi2_red)
    
    
    ##########
    # Plotting
    ##########
    fontsize1 = 18
    fontsize2 = 14       

    py.clf()
    fig1 = py.figure(2, figsize=(10,8))
    py.subplots_adjust(left=0.1, top=0.95, hspace=0.3, wspace=0.3)
    
    paxes = py.subplot(2, 2, 1)
    originX = np.mean(thetaSx_model)
    originY = np.mean(thetaSy_model)
    py.plot(tmod-t0, thetaSx_model-originX, 'r-')
    py.plot(tmod-t0, thetaS_nolensx-originX, 'r--')
    py.errorbar(tobs-t0, thetaSx_data-originX, yerr=xerr_data, fmt='k.')
    py.ylabel(r'x$_E$ (mas)', fontsize=fontsize1)
    py.xlabel('t - t$_{\mathrm{o}}$ (yr)', fontsize=fontsize1)
    py.ylim(-10, 2)
    py.xlim(-0.1, 2.5)
    xticks = np.arange(0, 2.5, 1, dtype=int)
    py.xticks(xticks, fontsize=fontsize2)
    py.yticks(fontsize=fontsize2)

    paxes = py.subplot(2, 2, 2)
    py.plot(tmod-t0, thetaSy_model-originY, 'b-')
    py.plot(tmod-t0, thetaS_nolensy-originY, 'b--')
    py.errorbar(tobs-t0, thetaSy_data-originY, yerr=yerr_data, fmt='k.')
    py.ylabel(r'x$_N$ (mas)', fontsize=fontsize1)
    py.xlabel('t - t$_{\mathrm{o}}$ (yr)', fontsize=fontsize1)  
    py.ylim(-1, 7)
    py.xlim(-0.1, 2.5)
    py.xticks(xticks, fontsize=fontsize2)
    py.yticks(fontsize=fontsize2)
        
    paxes = py.subplot(2, 2, 3)
    n = len(tmod)
    py.plot(thetaS_nolensx - originX,  thetaS_nolensy - originY, 'g--')
    py.plot(thetaSx_model - originX, thetaSy_model - originY, 'g-')
    py.errorbar(thetaSx_data - originX,  thetaSy_data - originY, xerr=xerr_data, yerr=yerr_data, fmt='k.')
    py.ylabel(r'x$_N$ (mas)', fontsize=fontsize1)
    py.xlabel(r'x$_E$ (mas)', fontsize=fontsize1)
    py.axis('equal')
    py.ylim(-1, 7)
    py.xlim(-10, 2)
    py.xticks(fontsize=fontsize2)
    py.yticks(fontsize=fontsize2)
    py.xlim(py.xlim()[::-1])
    

    # Make a histogram of the mass using the weights. This creates the 
    # marginalized 1D posteriors.
    paxes = py.subplot(2, 2, 4)
    bins = 75
    n, bins, patch = py.hist(tab['Mass'], normed=True,
                             histtype='step', weights=tab['weights'], bins=bins, linewidth=1.5)
    bin_centers = (bins[:-1] + bins[1:]) / 2
    py.xlabel('Mass')
    xtitle = r'Lens Mass (M$_{\odot}$)'
    py.xlabel(xtitle, fontsize=fontsize1, labelpad=10)
    py.xlim(0, 2.0)
    py.ylim(0, 1.4)
    py.ylabel('Relative probability', fontsize=fontsize1, labelpad=10)
    py.xticks(fontsize=fontsize2)
    py.yticks(fontsize=fontsize2)


    outdir = root_dir + mnest_dir + 'plots/'    
    outfile =  outdir + 'plot_OB120169_data_vs_model'
    if u0minus == True:
        outfile += '_u0minus'
    else:
        outfile += '_u0plus'
        
    fileUtil.mkdir(outdir)
    print 'writing plot to file ' + outfile
    
    py.savefig(outfile + '.png')
    py.savefig(outfile + '.pdf')

    return

def load_mnest_results_OB110022(root_dir='/Users/jlu/work/microlens/2015_evan/analysis_2014_03_22al_MC100_omit_1/',
                                mnest_dir='multiNest/bc_big_murel/',
                                mnest_root='bc_big_murel'):
    return load_mnest_results(root_dir, mnest_dir, mnest_root)

def plot_posteriors_muRel_OB110022():
    """
    For OB110022, plot the 2D posteriors for
    Mass, thetaE, 
    """
    analysis_dir_fmt = 'analysis_{0}_2014_03_22{1}_MC100_omit_1/'

    # Assume this is NIRC2 data.
    pixScale = 9.952  # mas / pixel

    ##########
    # First directory
    ##########
    root_dir = analysis_dir + analysis_dir_fmt.format(final_targets[00], final_flag[00]) + 'multiNest/'
    mnest_dir = 'bf/'
    mnest_root = 'bf'
    
    tab = load_mnest_results(root_dir = root_dir,
                             mnest_dir = mnest_dir,
                             mnest_root = mnest_root)

    # Plot 1D posteriors with priors over-plotted.
    py.close(2)
    py.figure(2, figsize=(6, 12))
    py.clf()

    py.subplots_adjust(left=0.18, bottom=0.07, top=0.98, hspace=0.35)

    py.subplot(311)
    binsize = 1.5 # mas/yr
    bin_min = mnest_ob110022.muRelx_min - (1.1 * binsize)
    bin_max = mnest_ob110022.muRelx_max + (1.1 * binsize)
    bins = np.arange(bin_min, bin_max, binsize)
    py.hist(tab['muRelx'] * -1.0, bins=bins, weights=tab['weights'],
            histtype='step', normed=True)
    py.plot(-bins, mnest_ob110022.muRelx_gen.pdf(-bins), 'k--')
    py.xlabel(r'$\mu_{\mathrm{rel},E}$ (mas yr$^{-1}$)')
    py.ylabel('Probability Density')
    py.xlim(bin_min, bin_max)
    py.ylim(0, 0.055)
    
    py.subplot(312)
    binsize = 1.0 # mas/yr
    bin_min = mnest_ob110022.muRely_min - (1.1 * binsize)
    bin_max = mnest_ob110022.muRely_max + (1.1 * binsize)
    bins = np.arange(bin_min, bin_max, binsize)
    py.hist(tab['muRely'], bins=bins, weights=tab['weights'],
            histtype='step', normed=True)
    py.plot(bins, mnest_ob110022.muRely_gen.pdf(bins), 'k--')
    py.ylabel('Probability Density')
    py.xlabel(r'$\mu_{\mathrm{rel},N}$ (mas yr$^{-1}$)')
    py.xlim(bin_min, bin_max)
    py.ylim(0, 0.055)


    # Plot 2D posterior for Theta_E and \muRel.
    muRel = np.hypot(tab['muRelx'], tab['muRely'])

    py.subplot(313)
    py.hist2d(muRel, tab['Mass'], weights=tab['weights'], bins=100, cmap=py.cm.gist_yarg)
    py.ylabel(r'Mass (M$_\odot$)')
    py.xlabel(r'$|\vec{\mu}_{\mathrm{rel}}|$ (mas yr$^{-1}$)')
    py.xlim(0, 40)
    py.ylim(0, 2.1)
    
    py.savefig(plot_dir + 'ob110022_post_muRelxy.png')

    return
    
def load_mnest_results(root_dir, mnest_dir, mnest_root):
    
    root = root_dir + mnest_dir + mnest_root + '_'
    tab = Table.read(root + '.txt', format='ascii')
    
    # Convert to log(likelihood)
    tab['col2'] /= -2.0
    
    # Rename the parameter columns. This is hard-coded to match the
    # above run() function.

    tab.rename_column('col1', 'weights')
    tab.rename_column('col2', 'logLike')
    tab.rename_column('col3', 't0')
    tab.rename_column('col4', 'beta')
    tab.rename_column('col5', 'tE')
    tab.rename_column('col6', 'piEN')
    tab.rename_column('col7', 'piEE')
    tab.rename_column('col8', 'thetaS0x')
    tab.rename_column('col9', 'thetaS0y')
    tab.rename_column('col10', 'muSx')
    tab.rename_column('col11', 'muSy')
    tab.rename_column('col12', 'muRelx')
    tab.rename_column('col13', 'muRely')
    tab.rename_column('col14', 'thetaE')
    tab.rename_column('col15', 'Mass')
        
    return tab

def table_lens_parameters():
    root_dir = analysis_dir
    analysis_dir_fmt = 'analysis_{0}_2014_03_22{1}_MC100_omit_1/'
    
    # OB110022
    tt = 0
    target = final_targets[tt]
    order = final_order[tt]
    flag = final_flag[tt]
    analysis_dir_tt = analysis_dir_fmt.format(target, flag)
    phot_file = '/Users/jlu/doc/papers/microlens/microlens_paper/Photometry/mcmc/ob110022_mcmc.dat'
    mnest_run = mnest_runs[target][order][0]
    mnest_dir = 'multiNest/' + mnest_run + '/'
    mnest_root = mnest_run

    print '##########'
    print '# OB110022'
    print '##########'
    multinest_plot.summarize_results(root_dir = root_dir + analysis_dir_tt,
                                     mnest_dir = mnest_dir,
                                     mnest_root = mnest_root,
                                     phot_file = phot_file,
                                     target = target)
    multinest_plot.calc_chi2_lens_fit(root_dir=roto_dir,
                                      analysis_dir=analysis_dir_tt,
                                      mnest_run=mnest_root,
                                      target=target,
                                      useMedian=False, verbose=True)
    multinest_plot.calc_chi2_lens_fit(root_dir=roto_dir,
                                      analysis_dir=analysis_dir_tt,
                                      mnest_run=mnest_root,
                                      target=target,
                                      useMedian=True, verbose=True)
        

    # OB120169
    # u0-
    tt = 2
    target = final_targets[tt]
    order = final_order[tt]
    flag = final_flag[tt]
    analysis_dir_tt = analysis_dir_fmt.format(target, flag)
    phot_file = '/Users/jlu/doc/papers/microlens/microlens_paper/Photometry/mcmc/ob120169_mcmc_u0minus.dat'
    mnest_run = mnest_runs[target][order][0]
    mnest_dir = 'multiNest/' + mnest_run + '/'
    mnest_root = mnest_run

    print '##########'
    print '# OB120169 u0-'
    print '##########'
    multinest_plot.summarize_results(root_dir = root_dir + analysis_dir_tt,
                                     mnest_dir = mnest_dir,
                                     mnest_root = mnest_root,
                                     phot_file = phot_file,
                                     target = target)

    # OB120169
    # u0+
    phot_file = '/Users/jlu/doc/papers/microlens/microlens_paper/Photometry/mcmc/ob120169_mcmc_u0plus.dat'
    mnest_run = mnest_runs[target][order][1]
    mnest_dir = 'multiNest/' + mnest_run + '/'
    mnest_root = mnest_run

    print '##########'
    print '# OB120169 u0+'
    print '##########'
    multinest_plot.summarize_results(root_dir = root_dir + analysis_dir_tt,
                                     mnest_dir = mnest_dir,
                                     mnest_root = mnest_root,
                                     phot_file = phot_file,
                                     target = target)

    # OB110125
    tt = 1
    target = final_targets[tt]
    order = final_order[tt]
    flag = final_flag[tt]
    analysis_dir_tt = analysis_dir_fmt.format(target, flag)
    phot_file = '/Users/jlu/doc/papers/microlens/microlens_paper/Photometry/mcmc/ob110125_mcmc.dat'
    mnest_run = None
    mnest_dir = None
    mnest_root = None

    print '##########'
    print '# OB110125'
    print '##########'
    multinest_plot.summarize_results(root_dir = root_dir + analysis_dir_tt,
                                     mnest_dir = mnest_dir,
                                     mnest_root = mnest_root,
                                     phot_file = phot_file,
                                     target = target)
    
def mass_posterior_OB110022():
    analysis_dir_fmt = 'analysis_{0}_2014_03_22{1}_MC100_omit_1/'
    
    tt = 0
    target = final_targets[tt]
    order = final_order[tt]
    flag = final_flag[tt]
    analysis_dir_tt = analysis_dir_fmt.format(target, flag)
    mnest_run = mnest_runs[target][order][0]
    mnest_dir = 'multiNest/' + mnest_run + '/'
    
    tab = load_mnest_results(root_dir = analysis_dir + analysis_dir_tt,
                             mnest_dir = mnest_dir,
                             mnest_root = mnest_run)
    outfile = 'plot_ob110022_mass_posterior.png'
    multinest_plot.mass_posterior(tab, plot_dir, outfile)

    return

def mass_posterior_OB120169():
    analysis_dir_fmt = 'analysis_{0}_2014_03_22{1}_MC100_omit_1/'
    
    tt = 2
    target = final_targets[tt]
    order = final_order[tt]
    flag = final_flag[tt]
    analysis_dir_tt = analysis_dir_fmt.format(target, flag)
    
    mnest_run = mnest_runs[target][order][0]
    mnest_dir = 'multiNest/' + mnest_run + '/'
    tab_m = load_mnest_results(root_dir = analysis_dir + analysis_dir_tt,
                               mnest_dir = mnest_dir,
                               mnest_root = mnest_run)

    mnest_run = mnest_runs[target][order][1]
    mnest_dir = 'multiNest/' + mnest_run + '/'
    tab_p = load_mnest_results(root_dir=analysis_dir + analysis_dir_tt,
                               mnest_dir=mnest_dir,
                               mnest_root=mnest_run)
    
    # Clean table and only keep valid results.
    Masslim = 0.0
    ind_p = np.where((tab_p['Mass'] >= Masslim))[0]
    ind_m = np.where((tab_m['Mass'] >= Masslim))[0]
    tab_p = tab_p[ind_p]
    tab_m = tab_m[ind_m]

    # Combine the two tables together.
    tab = table.vstack(tab_p, tab_m)
    
    outfile = 'plot_ob120169_mass_posterior.png'
    print 'u0-'
    multinest_plot.mass_posterior(tab_m, plot_dir, outfile)
    print ''
    
    print 'u0-'
    multinest_plot.mass_posterior(tab_p, plot_dir, outfile)
    print ''

    print 'combined'
    multinest_plot.mass_posterior(tab, plot_dir, outfile)

    return
    
    

