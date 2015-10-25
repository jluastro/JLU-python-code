import numpy as np
import pylab as py
from matplotlib.ticker import ScalarFormatter, FormatStrFormatter
from mpl_toolkits.axes_grid1 import Grid
import pdb
import asciidata
from jlu.gc.gcwork import starset
from jlu.microlens import multinest_plot
from jlu.microlens import calc

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
    scale = 0.00992
    
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
        
        grid[ee].axis([12, 23, 2e-2, 30.0])
        grid[ee].axhline(0.15, color='g') 
        grid[ee].xaxis.label.set_fontsize(tickfontsize)
        grid[ee].yaxis.label.set_fontsize(tickfontsize)
        if ee > 2:
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

    pdb.set_trace()

