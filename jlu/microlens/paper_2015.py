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
from scipy import stats
import ephem
import math

plotdir = '/Users/jlu/doc/papers/OTHERS/sinukoff/microlens/microlens_paper/Paper/'

def plot_figure1b(M=[1,5,10], beta=0.0,
                  logxmin = -3, logxmax = 2., dl = 4.0, ds = 8.0,
                  plotdir=plotdir):

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
   
    py.savefig(plotdir + 'MassTrend.pdf')
     
    return

     
def plot_figure1a(M=10., beta=0.1,
                  logxmin = -2, logxmax = 2., dl = 4.0, ds = 8.0,
                  plotdir=plotdir):

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

    py.savefig(plotdir + 'Phot_Ast_compare.pdf')

    return

     
def plotPosError(rootDir='/u/jlu/data/microlens/', plotDir=plotdir,
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
    py.savefig(plotDir + 'plotPosError_%s.pdf' % (target))
            

    return


def plot_pos_on_sky(target, plotdir=plotdir):
    align = 'align/align_t'
    poly = 'polyfit_d/fit'
    points = 'points_d/'

    rootDir = '/u/sinukoff/projects/microlens/analysis/'
    if target == 'ob110022': 
        rootDir += 'analysis_ob110022_2014_03_22ah_MC100_omit_1/'
    if target == 'ob110125': 
        rootDir += 'analysis_ob110125_2014_03_22al_MC100_omit_1/'
    if target == 'ob120169': 
        rootDir += 'analysis_ob120169_2014_03_22al_MC100_omit_1/'

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

    py.savefig(plotdir + 'plot_pos_on_sky_' + target + '.pdf')

def check_parallax_correction(target, u0pos=True):
    root_dir='/u/sinukoff/projects/microlens/analysis/'

    if target == 'ob110022':
        root_dir += 'analysis_ob110022_2014_03_22ah_MC100_omit_1/'
        mnest_dir = 'multiNest/av/'
        mnest_root = 'av'
    if target == 'ob120169':
        root_dir += 'analysis_ob120169_2014_03_22al_MC100_omit_1/'
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


def plot_residuals_figures():
    # OB120169
    # starNames = ['OB120169_R', 'p005_15_3.5', 'S2_16_2.5', 'p000_16_3.6', 'S6_17_2.4', 'S7_17_2.3']
    starNames = ['OB120169_R', 'OB120169_L', 'S10_17_1.4', 'S12_17_2.4', 'S6_17_2.4', 'S7_17_2.3']

    rootDir = '/Users/jlu/work/microlens/OB120169/analysis_2015_09_18/analysis_ob120169_2015_09_18_a3_m22_w4_MC100/'
    align = 'align/align_t'
    residuals.plotStar(starNames, rootDir=rootDir, align=align)


def align_residuals_vs_order(align_roots, root_dir='./', only_stars_in_fit=True,
                             out_suffix=''):
    data_all = []

    for aa in range(len(align_roots)):
        data = residuals.check_alignment_fit(align_roots[aa], root_dir=root_dir)

        data_all.append(data)

    py.figure(2, figsize=(15, 3.5))
    py.clf()
    py.subplots_adjust(left=0.08, bottom=0.2, hspace=0, wspace=0, right=0.95, top=0.88)

    order = np.arange(1, len(data_all)+1)
    N_order = len(data_all)

    year = data_all[0]['year']
    scale = 9.952  # mas / pixel

    idx_subplot = 0

    majorLoc = MultipleLocator(1)
    majorFmt = FormatStrFormatter('%d')
    nullFmt = NullFormatter()

    N_plots = len(year)
    if only_stars_in_fit:
        N_plots -= 1

    # Plot Residuals
    for ee in range(len(year)):
        end = 'all'
        if only_stars_in_fit:
            end = 'used'
        
        xres = np.array([data_all[oo]['xres_rms_'+end][ee] for oo in range(N_order)])
        yres = np.array([data_all[oo]['yres_rms_'+end][ee] for oo in range(N_order)])
    
        xres_e = np.array([data_all[oo]['xres_err_'+end][ee] for oo in range(N_order)])
        yres_e = np.array([data_all[oo]['yres_err_'+end][ee] for oo in range(N_order)])

        xres *= scale
        yres *= scale
        xres_e *= scale
        yres_e *= scale

        if only_stars_in_fit and np.isnan(xres[0]):
            print 'Skipping Ref Epoch', ee
            continue

        idx_subplot += 1
        py.subplot(1, N_plots, idx_subplot)
        py.errorbar(order, xres, yerr=xres_e, fmt='r.--', label='X')
        py.errorbar(order, yres, yerr=yres_e, fmt='b.--', label='Y')
        py.title(data_all[0]['year'][ee])
        py.xlim(0, 3.9)
        py.ylim(0, 2.1)

        ax = py.gca()
        ax.xaxis.set_major_locator(majorLoc)
        #ax.yaxis.set_major_locator(majorLoc)
        ax.xaxis.set_major_formatter(majorFmt)
        py.xlabel('Order')
        
        if idx_subplot == 1:
            #ax.yaxis.set_major_formatter(majorFmt)
            py.ylabel('RMS Residuals (mas)')
        else:
            ax.yaxis.set_major_formatter(nullFmt)

        if idx_subplot == N_plots:
            py.legend(numpoints=1)

    py.savefig(plotdir + 'align_residuals_vs_order_' + out_suffix + '.png')

    end = 'all'
    if only_stars_in_fit:
        end = 'used'
        
    N_par = np.array([3, 6, 10])


    # Fit Improvement Significance
    for ee in range(len(year)):
        chi2x = np.array([data_all[oo]['chi2x_'+end][ee] for oo in range(N_order)])
        chi2y = np.array([data_all[oo]['chi2y_'+end][ee] for oo in range(N_order)])
        N_stars = np.array([data_all[oo]['N_stars_'+end][ee] for oo in range(N_order)])
        print 'N_stars: ', N_stars
    
        if only_stars_in_fit and np.isnan(chi2x[0]):
            print 'Skipping Ref Epoch', ee
            continue

        # These have shape: len(order) - 1        
        ftest_m = order[1:]
        ftest_dof1 = np.diff(N_par)
        ftest_dof2 = N_stars[1:] - N_par[1:]
        ftest_dof_ratio = 1.0 / (ftest_dof1 / ftest_dof2)
        ftest_x = (-1 * np.diff(chi2x) / chi2x[1:]) * ftest_dof_ratio
        ftest_y = (-1 * np.diff(chi2y) / chi2y[1:]) * ftest_dof_ratio

        px_value = np.zeros(len(ftest_m), dtype=float)
        py_value = np.zeros(len(ftest_m), dtype=float)

        for mm in range(len(ftest_m)):
            px_value[mm] = stats.f.sf(ftest_x[mm], ftest_dof1[mm], ftest_dof2[mm])
            py_value[mm] = stats.f.sf(ftest_y[mm], ftest_dof1[mm], ftest_dof2[mm])
            fmt = 'M = {0}: M-1 --> M   Fx = {1:5.2f} px = {2:7.4f}   Fy = {3:5.2f} py = {4:7.4f}'
            print fmt.format(ftest_m[mm], ftest_x[mm], px_value[mm], ftest_y[mm], py_value[mm])

    # Combine over all epochs to include velocities:
    chi2x = np.array([data_all[oo]['chi2x_'+end].sum() for oo in range(N_order)])
    chi2y = np.array([data_all[oo]['chi2y_'+end].sum() for oo in range(N_order)])
    chi2 = chi2x + chi2y

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
    print '*** Combined F test across epochs ***'
    print 'N_data: ', N_data
    print 'N_vfit: ', N_vfit
    print 'N_afit: ', N_afit
    print 'N_free: ', N_free_param
    print 'chi2: ', chi2
    print 'N_dof: ', N_dof

    ftest_m = order[1:]
    ftest_dof1 = np.diff(N_free_param)
    ftest_dof2 = N_data - N_free_param[1:]
    ftest_dof_ratio = 1.0 / (ftest_dof1 / ftest_dof2)
    ftest = (-1 * np.diff(chi2) / chi2[1:]) * ftest_dof_ratio
        
    p_value = np.zeros(len(ftest_m), dtype=float)

    print 'ftest_m: ', ftest_m
    print 'ftest_dof1: ', ftest_dof1
    print 'ftest_dof2: ', ftest_dof2
    print 'ftest_dof_ratio: ', ftest_dof_ratio
    print 'ftest: ', ftest
    
    for mm in range(len(ftest_m)):
        p_value[mm] = stats.f.sf(ftest[mm], ftest_dof1[mm], ftest_dof2[mm])
        fmt = 'M = {0}: M-1 --> M   F = {1:5.2f} p = {2:7.4f}'
        print fmt.format(ftest_m[mm], ftest[mm], p_value[mm])

    return

def plot_align_order(target, only_stars_in_fit=True):
    root_dir = '/Users/jlu/work/microlens/2015_evan/'
    
    if target == 'ob110022':
        flags = ['ah', 'at', 'bf']
    else:
        flags = ['al', 'ax', 'bj']

    align_root_fmt = 'analysis_{0}_2014_03_22{1}_MC100_omit_1/align/align'

    align_roots = []
    
    for ii in range(len(flags)):
        align_roots.append( align_root_fmt.format(target, flags[ii]) )
        
    align_residuals_vs_order(align_roots, root_dir=root_dir,
                             only_stars_in_fit=only_stars_in_fit,
                             out_suffix=target)

    return    

    
def plot_alignment_errors(target, radius=4):
    align = 'align/align'

    rootDir = '/u/sinukoff/projects/microlens/analysis/'
    if target == 'ob110022': 
        rootDir += 'analysis_ob110022_2014_03_22ah_MC100_omit_1/'
        epochs = ['2011 May','2011 Jul','2011 Jun','2012 Jul','2013 Apr','2013 Jul']
    if target == 'ob110125': 
        rootDir += 'analysis_ob110125_2014_03_22al_MC100_omit_1/'
        epochs = ['2012 May','2012 Jun', '2012 Jul','2013 Apr','2013 Jul']
    if target == 'ob120169': 
        rootDir += 'analysis_ob120169_2014_03_22al_MC100_omit_1/'
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
        
    py.figtext(0.02, 0.55, 'Alignment Uncertainty (mas)', fontsize=axfontsize,
                rotation='vertical', verticalalignment='center')
    py.savefig(plotdir + 'plotPosAlignError_%s.png' % (target))
            

    return

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

        
