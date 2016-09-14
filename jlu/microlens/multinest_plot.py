import pylab as py
import numpy as np
import os
from astropy.table import Table
from astropy import table
from jlu.microlens import MCMC_LensModel
from jlu.util import fileUtil
import pandas

def plot_OB120169(root_dir='/Users/jlu/work/microlens/OB120169/analysis_2016_06_22/',
                  analysis_dir='analysis_ob120169_2016_06_22_a4_m22_w4_MC100/',
                  points_dir='points_d/', mnest_dir='multiNest/up/',
                  mnest_root='up'):
        
    target = 'OB120169_R'
    display_name = 'OB120169'
    
    #Plot astrometric observations
    pointsFile = root_dir + analysis_dir + points_dir + target + '.points'
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
    tab = load_mnest_results_OB120169(root_dir=root_dir + analysis_dir,
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
    py.ylabel(r'$\Delta\alpha$ (mas)', fontsize=fontsize1)
    py.xlabel('t - t$_{\mathrm{o}}$ (yr)', fontsize=fontsize1)
    py.ylim(-4, 12)
    py.xlim(-0.5, 6.0)
    xticks = np.arange(0, 4.1, 1, dtype=int)
    py.xticks(xticks, fontsize=fontsize2)
    py.yticks(fontsize=fontsize2)

    paxes = py.subplot(2, 2, 2)
    py.plot(tmod-t0, thetaSy_model-originY, 'b-')
    py.plot(tmod-t0, thetaS_nolensy-originY, 'b--')
    py.errorbar(tobs-t0, thetaSy_data-originY, yerr=yerr_data, fmt='k.')
    py.ylabel(r'$\Delta\delta$ (mas)', fontsize=fontsize1)
    py.xlabel('t - t$_{\mathrm{o}}$ (yr)', fontsize=fontsize1)  
    py.ylim(-2, 6)
    py.xlim(-0.5, 6.0)
    py.xticks(xticks, fontsize=fontsize2)
    py.yticks(fontsize=fontsize2)
        
    paxes = py.subplot(2, 2, 3)
    n = len(tmod)
    py.plot(thetaS_nolensx - originX,  thetaS_nolensy - originY, 'g--')
    py.plot(thetaSx_model - originX, thetaSy_model - originY, 'g-')
    py.errorbar(thetaSx_data - originX,  thetaSy_data - originY, xerr=xerr_data, yerr=yerr_data, fmt='k.')
    py.ylabel(r'$\Delta\delta$ (mas)', fontsize=fontsize1)
    py.xlabel(r'$\Delta\alpha$ (mas)', fontsize=fontsize1)
    py.axis('equal')
    py.ylim(-5, 7)
    py.xlim(-4, 8)
    py.xticks(fontsize=fontsize2)
    py.yticks(fontsize=fontsize2)
    py.xlim(py.xlim()[::-1])
    

    # Make a histogram of the mass using the weights. This creates the 
    # marginalized 1D posteriors.
    paxes = py.subplot(2, 2, 4)
    bins = np.arange(0, 30, 0.25)
    n, bins, patch = py.hist(tab['Mass'], normed=True,
                             histtype='step', weights=tab['weights'], bins=bins, linewidth=1.5)
    bin_centers = (bins[:-1] + bins[1:]) / 2
    py.xlabel('Mass')
    xtitle = r'Lens Mass (M$_{\odot}$)'
    py.xlabel(xtitle, fontsize=fontsize1, labelpad=10)
    py.xlim(0, 20)
    py.ylim(0, 0.2)
    py.ylabel('Relative probability', fontsize=fontsize1, labelpad=10)
    py.xticks(fontsize=fontsize2)
    py.yticks(fontsize=fontsize2)


    outdir = root_dir + analysis_dir + mnest_dir + 'plots/'    
    outfile =  outdir + 'plot_OB120169_data_vs_model.png'
    fileUtil.mkdir(outdir)
    print 'writing plot to file ' + outfile
    
    py.savefig(outfile)

    return


def load_mnest_results_OB120169(root_dir='/Users/jlu/work/microlens/OB120169/analysis_2015_09_18/',
                                mnest_dir='mnest_evan_2015_08/ay/',
                                mnest_root='ay'):
    return load_mnest_results(root_dir, mnest_dir, mnest_root)




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

def mass_posterior_OB120169():

    #root_dir='/Users/jlu/work/microlens/OB120169/analysis_2015_09_18/'
    #analysis_dir='analysis_ob120169_2015_09_18_a4_m20_w4_MC100/'

    root_dir = '/Users/jlu/work/microlens/2015_evan/'
    analysis_dir = 'analysis_ob120169_2014_03_22'
    
    ##########
    # O=1 directory
    ##########
    an_dir = analysis_dir + 'al_MC100_omit_1/'
    
    # Load Model Posteriors for both
    #    u_o > 0 (p=plus) and
    #    u_o < 0 (m=minus).
    mnest_dir = an_dir + 'multiNest/ay/'
    mnest_root = 'ay'
    tab_p = load_mnest_results(root_dir=root_dir, mnest_dir=mnest_dir, mnest_root=mnest_root)
    
    mnest_dir = an_dir + 'multiNest/ax/'
    mnest_root = 'ax'
    tab_m = load_mnest_results(root_dir=root_dir, mnest_dir=mnest_dir, mnest_root=mnest_root)

    # Clean table and only keep valid results.
    Masslim = 0.0
    ind_p = np.where((tab_p['Mass'] >= Masslim))[0]
    ind_m = np.where((tab_m['Mass'] >= Masslim))[0]
    tab_p = tab_p[ind_p]
    tab_m = tab_m[ind_m]

    # Combine the two tables together.
    tab = table.vstack(tab_p, tab_m)

    outdir = root_dir + an_dir + 'multiNest/plots/'    
    outfile =  'plot_OB120169_mass_posterior_combo_O1.png'

    mass_posterior(tab, outdir, outfile, bins=500)

    ##########
    # O=2 directory
    ##########
    an_dir = analysis_dir + 'ax_MC100_omit_1/'
    
    # Load Model Posteriors for both
    #    u_o > 0 (p=plus) and
    #    u_o < 0 (m=minus).
    mnest_dir = an_dir + 'multiNest/bh/'
    mnest_root = 'bh'
    tab_p = load_mnest_results(root_dir=root_dir, mnest_dir=mnest_dir, mnest_root=mnest_root)
    
    mnest_dir = an_dir + 'multiNest/ba/'
    mnest_root = 'ba'
    tab_m = load_mnest_results(root_dir=root_dir, mnest_dir=mnest_dir, mnest_root=mnest_root)

    # Clean table and only keep valid results.
    Masslim = 0.0
    ind_p = np.where((tab_p['Mass'] >= Masslim))[0]
    ind_m = np.where((tab_m['Mass'] >= Masslim))[0]
    tab_p = tab_p[ind_p]
    tab_m = tab_m[ind_m]

    # Combine the two tables together.
    tab = table.vstack(tab_p, tab_m)

    outdir = root_dir + an_dir + 'multiNest/plots/'    
    outfile =  'plot_OB120169_mass_posterior_combo_O2.png'

    mass_posterior(tab, outdir, outfile, bins=500)
    

def mass_posterior_OB110022():
    root_dir = '/Users/jlu/work/microlens/2015_evan/'
    analysis_dir = 'analysis_ob110022_2014_03_22'

    ##########
    # O=1 directory
    ##########
    an_dir = analysis_dir + 'al_MC100_omit_1/'
    mnest_dir = an_dir + 'multiNest/bf/'
    mnest_root = 'bf'
    
    tab = load_mnest_results(root_dir = root_dir,
                             mnest_dir = mnest_dir,
                             mnest_root = mnest_root)
    outdir = root_dir + an_dir + 'multiNest/plots/'
    outfile = 'plot_ob110022_mass_posterior_combo_O1.png'
    mass_posterior(tab, outdir, outfile)

    ##########
    # O=2 directory
    ##########
    an_dir = analysis_dir + 'ax_MC100_omit_1/'
    mnest_dir = an_dir + 'multiNest/bc/'
    mnest_root = 'bc'

    tab = load_mnest_results(root_dir = root_dir,
                             mnest_dir = mnest_dir,
                             mnest_root = mnest_root)
    outdir = root_dir + an_dir + 'multiNest/plots/'
    outfile = 'plot_ob110022_mass_posterior_combo_O2.png'
    mass_posterior(tab, outdir, outfile)

def mass_posterior_OB110022_err_x2():
    root_dir = '/Users/jlu/work/microlens/2015_evan/'
    analysis_dir = 'analysis_ob110022_2014_03_22'
    
    ##########
    # O=1 directory, X and Y errors double.
    ##########
    an_dir = analysis_dir + 'al_MC100_omit_1/'
    mnest_dir = an_dir + 'multiNest/bf_err_x2/'
    mnest_root = 'bf_err_x2'
    
    tab = load_mnest_results(root_dir = root_dir,
                             mnest_dir = mnest_dir,
                             mnest_root = mnest_root)
    outdir = root_dir + an_dir + 'multiNest/plots/'
    outfile = 'plot_ob110022_mass_posterior_combo_O1_err_x2.png'
    mass_posterior(tab, outdir, outfile)

    

def mass_posterior(tab, outdir, outfile, bins=50):

    # Make a histogram of the mass using the weights. This creates the 
    # marginalized 1D posteriors.
    fontsize1 = 18
    fontsize2 = 14       

    massMax = np.ceil(tab['Mass'].max())
    
    py.figure(1)
    py.clf()
    bins = np.arange(0, massMax, massMax / bins)
    n, foo, patch = py.hist(tab['Mass'], normed=True,
                             histtype='step', weights=tab['weights'],
                             bins=bins, linewidth=1.5)
    bin_centers = (bins[:-1] + bins[1:]) / 2
    py.xlabel('Mass')
    xtitle = r'Lens Mass (M$_{\odot}$)'
    py.xlabel(xtitle, fontsize=fontsize1, labelpad=10)
    py.xlim(0, np.ceil(tab['Mass'].max()))
    py.ylabel('Relative probability', fontsize=fontsize1, labelpad=10)
    py.xticks(fontsize=fontsize2)
    py.yticks(fontsize=fontsize2)

    ##########
    # Calculate 3-sigma boundaries for mass limits.
    ##########
    sig1_hi = 0.682689
    sig_med = 0.5
    sig3_hi = 0.9973
    sig1_lo = 1.0 - sig1_hi
    sig3_lo = 1.0 - sig3_hi

    quantiles = [sig3_lo, sig1_lo, sig_med, sig1_hi, sig3_hi]

    mass_quants = weighted_quantile(tab['Mass'], quantiles,
                                    sample_weight=tab['weights'])    

    for qq in range(len(quantiles)):
        print 'Mass at {0:.1f}% quantiles:  M = {1:5.2f}'.format(quantiles[qq]*100, mass_quants[qq])

    ax = py.axis()
    py.axvline(mass_quants[0], color='red', linestyle='--')
    py.text(mass_quants[0] + 0.05, 0.9*ax[3],
            'M>{0:5.2f} with 99.7% confidence'.format(mass_quants[0]), fontsize=12)
    py.axvline(mass_quants[-1], color='green', linestyle='--')
    py.text(mass_quants[-1] + 0.05, 0.8*ax[3],
            'M<{0:5.2f} with 99.7% confidence'.format(mass_quants[-1]), fontsize=12)
    
    ##########
    # Save figure
    ##########
    outfile =  outdir + outfile
    fileUtil.mkdir(outdir)
    print 'writing plot to file ' + outfile
    
    py.savefig(outfile)

    return

def get_best_fit_OB120169(root_dir='/Users/jlu/work/microlens/OB120169/analysis_2015_09_18/',
                          mnest_dir='mnest_evan_2015_08/ay/',
                          mnest_root='ay'):
    return get_best_fit(root_dir, mnest_dir, mnest_root)

    
def get_best_fit(root_dir, mnest_dir, mnest_root):
    # Overplot Best-Fit Model
    tab = load_mnest_results(root_dir, mnest_dir, mnest_root)
    params = tab.keys()
    pcnt = len(params)
    Masslim = 0.0

    # Clean table and only keep valid results.
    ind = np.where((tab['Mass'] >= Masslim))[0]
    tab = tab[ind]

    best = np.argmax(tab['logLike'])
    maxLike = tab['logLike'][best]

    tab_best = tab[best]

    return tab_best


def plot_posteriors(root_dir, mnest_dir, mnest_root):
    """
    Plots posteriors using pair_posterior code
    """
    outdir = root_dir + mnest_dir
    tab = load_mnest_results(root_dir=root_dir, mnest_dir=mnest_dir, mnest_root=mnest_root)
    weights = tab['weights']
    tab.remove_columns(('weights', 'logLike'))
    if not os.path.exists(outdir + '/plots'):
        os.mkdir(outdir + '/plots')
    if not os.path.exists(outdir + '/plots/posteriors'):
        os.mkdir(outdir + 'plots/posteriors/')
    pair_posterior(tab, weights, outfile=outdir+'plots/posteriors/' + mnest_root +'_posteriors.png', title=mnest_root)

    return

def pair_posterior(atpy_table, weights, outfile=None, title=None):
    """
    pair_posterior(atpy_table)

    :Arguments:
    atpy_table:       Contains 1 column for each parameter with samples.

    Produces a matrix of plots. On the diagonals are the marginal
    posteriors of the parameters. On the off-diagonals are the
    marginal pairwise posteriors of the parameters.
    """

    params = atpy_table.keys()
    pcnt = len(params)

    fontsize = 6
    
    py.close(10)
    py.figure(10, figsize = (20,20))

    # Marginalized 1D
    for ii in range(pcnt):
        ax = py.subplot(pcnt, pcnt, ii*(pcnt+1)+1)
        py.setp(ax.get_xticklabels(), fontsize=fontsize)
        py.setp(ax.get_yticklabels(), fontsize=fontsize)
        n, bins, patch = py.hist(atpy_table[params[ii]], normed=True,
                                 histtype='step', weights=weights, bins=50)
        py.xlabel(params[ii], size=fontsize)
        py.ylim(0, n.max()*1.1)

    # Bivariates
    for ii in range(pcnt - 1):
        for jj in range(ii+1, pcnt):
            ax = py.subplot(pcnt, pcnt, ii*pcnt + jj+1)
            py.setp(ax.get_xticklabels(), fontsize=fontsize)
            py.setp(ax.get_yticklabels(), fontsize=fontsize)

            (H, x, y) = np.histogram2d(atpy_table[params[jj]], atpy_table[params[ii]],
                                       weights=weights, bins=50)
            xcenter = x[:-1] + (np.diff(x) / 2.0)
            ycenter = y[:-1] + (np.diff(y) / 2.0)

            py.contourf(xcenter, ycenter, H.T, cmap=py.cm.gist_yarg)

            py.xlabel(params[jj], size=fontsize)
            py.ylabel(params[ii], size=fontsize)

    #if title != None:
    #    py.suptitle(title)
    py.subplots_adjust(wspace=0.3, hspace=0.3)
    if outfile != None:
        py.savefig(outfile, bbox_inches='tight', pad_inches=0.05)

    return


def calc_chi2_lens_fit(root_dir='/Users/jlu/work/microlens/2015_evan/',
                       analysis_dir='analysis_ob110022_2014_03_22al_MC100_omit_1/',
                       mnest_run='bf',
                       target='OB110022',
                       useMedian=False, verbose=True):
    
    # Assume this is NIRC2 data.
    pixScale = 9.952  # mas / pixel

    if target == 'ob120169':
        targname = 'ob120169_R'
    else:
        targname = target

    #Plot astrometric observations
    pointsFile = root_dir + analysis_dir + 'points_d/' + targname + '.points'
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
    mnest_dir = 'multiNest/' + mnest_run + '/'
    mnest_root = mnest_run
    tab = load_mnest_results(root_dir=root_dir + analysis_dir,
                             mnest_dir=mnest_dir,
                             mnest_root=mnest_root)

    params = tab.keys()
    pcnt = len(params)
    Masslim = 0.0

    # Clean table and only keep valid results.
    ind = np.where((tab['Mass'] >= Masslim))[0]
    tab = tab[ind]

    if not useMedian:    
        best = np.argmax(tab['logLike'])
        maxLike = tab['logLike'][best]
        if verbose:
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

        if verbose:
            print '     muLx = {0:5.2f} mas/yr'.format(muSx - muRelx)
            print '     muLy = {0:5.2f} mas/yr'.format(muSy - muRely)
    else:
        t0 = weighted_quantile(tab['t0'], 0.5, sample_weight=tab['weights'])
        tE = weighted_quantile(tab['tE'], 0.5, sample_weight=tab['weights'])
        thetaS0x = weighted_quantile(tab['thetaS0x'], 0.5, sample_weight=tab['weights'])
        thetaS0y = weighted_quantile(tab['thetaS0y'], 0.5, sample_weight=tab['weights'])
        muSx = weighted_quantile(tab['muSx'], 0.5, sample_weight=tab['weights'])
        muSy = weighted_quantile(tab['muSy'], 0.5, sample_weight=tab['weights'])
        muRelx = weighted_quantile(tab['muRelx'], 0.5, sample_weight=tab['weights'])
        muRely = weighted_quantile(tab['muRely'], 0.5, sample_weight=tab['weights'])
        beta = weighted_quantile(tab['beta'], 0.5, sample_weight=tab['weights'])
        piEN = weighted_quantile(tab['piEN'], 0.5, sample_weight=tab['weights'])
        piEE = weighted_quantile(tab['piEE'], 0.5, sample_weight=tab['weights'])

        if verbose:
            print 'Median Solution:'
            print '   Params:'
            print '     {0:15s}  {1:10.3f}'.format('t0', t0)
            print '     {0:15s}  {1:10.3f}'.format('tE', tE)
            print '     {0:15s}  {1:10.3f}'.format('thetaS0x', thetaS0x)
            print '     {0:15s}  {1:10.3f}'.format('thetaS0y', thetaS0y)
            print '     {0:15s}  {1:10.3f}'.format('muSx', muSx)
            print '     {0:15s}  {1:10.3f}'.format('muSy', muSy)
            print '     {0:15s}  {1:10.3f}'.format('muRelx', muRelx)
            print '     {0:15s}  {1:10.3f}'.format('muRely', muRely)
            print '     {0:15s}  {1:10.3f}'.format('beta', beta)
            print '     {0:15s}  {1:10.3f}'.format('piEN', piEN)
            print '     {0:15s}  {1:10.3f}'.format('piEE', piEE)
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
    chi2_tot = chi2_x + chi2_y
    chi2_red = chi2_tot / N_dof

    if verbose:
        print 'Chi-Squared of Lens Model Fit:'
        print '        X Chi^2 = {0:5.2f}'.format(chi2_x)
        print '        Y Chi^2 = {0:5.2f}'.format(chi2_y)
        print '    Total Chi^2 = {0:5.2f} ({1:d} DOF)'.format(chi2_tot, N_dof)
        print '  Reduced Chi^2 = {0:5.2f}'.format(chi2_red)
        print ''
        print ''

    return chi2_tot, N_dof, chi2_red


def summarize_results(root_dir='/Users/jlu/work/microlens/2015_evan/',
                      analysis_dir='analysis_ob110022_2014_03_22al_MC100_omit_1/',
                      mnest_root='bf',
                      target='OB110022',
                      phot_file = '/Users/jlu/doc/papers/microlens/microlens_paper/Photometry/mcmc/ob110022_mcmc.dat'):


    sig1 = 0.682689
    sig2 = 0.9545
    sig3 = 0.9973
    sig1_lo = (1.-sig1)/2.
    sig2_lo = (1.-sig2)/2.
    sig3_lo = (1.-sig3)/2.
    sig1_hi = 1.-sig1_lo
    sig2_hi = 1.-sig2_lo
    sig3_hi = 1.-sig3_lo
    
    fmts = {}
    tblvals = {}

    dof_phot = {'OB110022': 8253, 'OB110125': 995, 'OB120169': 433}

    ##########
    # Photometry Fits: Subo's MCMC chains 
    ##########
    usepars_phot = []
    photdf = pandas.read_table(phot_file, sep = ' ', header=0, skipinitialspace=True)
    photpars = photdf.columns.values
    for j in photpars:
        # Skip the junk column
        if j.startswith('junk'):
            continue
            
        arr = photdf[j]

        # Switch from flux to magnitudes
        if j == 'fsource':
            arr = 18.0 - 2.5*np.log10(arr)
            j = 'Isource'

        # Add this column to our output dictionary and calculate the
        # median, upper, and lower values.
        usepars_phot.append(j)
        tblvals[target, 'phot', j] = np.percentile(arr, [50, sig1_lo*100, sig1_hi*100])

        # If this is the chi^2 column, use the min chisq instead.
        if j == 'chisq':
            tblvals[target, 'phot', j] = [np.min(arr), 0, 0]

        # Switch from values to errors.
        tblvals[target, 'phot', j][1] = tblvals[target, 'phot', j][0] - tblvals[target, 'phot', j][1]
        tblvals[target, 'phot', j][2] = tblvals[target, 'phot', j][2] - tblvals[target, 'phot', j][0]


    # Add the fblend / fsource results
    arr = photdf['fblend'] / photdf['fsource'] 
    usepars_phot.append('fratio')
    tblvals[target, 'phot', 'fratio'] = np.percentile(arr, [50, sig1_lo*100, sig1_hi*100])
    tblvals[target, 'phot', 'fratio'][1] = tblvals[target, 'phot', 'fratio'][0] - tblvals[target, 'phot', 'fratio'][1]
    tblvals[target, 'phot', 'fratio'][2] = tblvals[target, 'phot', 'fratio'][2] - tblvals[target, 'phot', 'fratio'][0]

    # Add the chi-squared DOF
    tblvals[target, 'phot', 'dof'] = [dof_phot[target.upper()], 0, 0]
    tblvals[target, 'phot', 'chisq'][1] = 0
    tblvals[target, 'phot', 'chisq'][2] = 0
    usepars_phot.append('dof')
    


    ##########        
    # MultiNest Astrometry Fits
    ##########
    if mnest_root != None:  
        mnest_dir = root_dir + analysis_dir + 'multiNest/' + mnest_root + '/'
        mnestdf = pandas.io.parsers.read_table(mnest_dir + mnest_root + '_.txt',
                                               header = None, sep = r"\s*")
    
        mnestdf.columns=['weights', 'logLike', 't0', 'u0', 'tE', 'piEN', 'piEE',
                         'thetaS0x', 'thetaS0y', 'muSx', 'muSy', 'muRelx', 'muRely',
                         'thetaE', 'Mass']
    
        # Convert to log(likelihood)
        mnestdf['logLike'] /= -2.0

        # Which params to include in table
        usepars_multinest = ['t0', 'u0', 'tE', 'piEN', 'piEE',
                             'thetaS0x', 'thetaS0y', 'muSx', 'muSy', 'muRelx', 'muRely',
                             'thetaE', 'Mass']
        
        weights = mnestdf['weights']
        sumweights = np.sum(weights)
        weights = weights / sumweights
    
        for n in usepars_multinest:
            arr = mnestdf[n].values
            
            # Convert t0 to modified MJD (same as photometry)
            # See algorithm in calc_year.pro
            if n == 't0':
                arr = year_to_mmjd(arr)
        
            # Calculate median, 1 sigma lo, and 1 sigma hi credible interval.
            tblvals[target, 'multinest', n] = weighted_quantile(arr, [0.5, sig1_lo, sig1_hi],
                                                                   sample_weight=weights.values)
            # Switch from values to errors.
            tblvals[target, 'multinest', n][1] = tblvals[target, 'multinest', n][0] - tblvals[target, 'multinest', n][1]
            tblvals[target, 'multinest', n][2] = tblvals[target, 'multinest', n][2] - tblvals[target, 'multinest', n][0]

        # Fetch the chi-sq value.
        chi2_tot, N_dof, chi2_red = calc_chi2_lens_fit(root_dir=root_dir,
                                                       analysis_dir=analysis_dir,
                                                       mnest_run=mnest_root,
                                                       target=target,
                                                       useMedian=True, verbose=False)
        tblvals[target, 'multinest', 'chisq'] = [chi2_tot, 0, 0]
        tblvals[target, 'multinest', 'dof'] = [N_dof, 0, 0]
        
                
    allpars = ['t0', 'u0', 'tE', 'piEE', 'piEN',
               'muSx', 'muSy', 'muRelx', 'muRely',
               'thetaE', 'Mass', 's', 'q', 'alpha', 'omega', 'sdot/s', 'Isource', 'fratio', 'chisq', 'dof']

    #Positive x direction should be East
    if tblvals.has_key((target, 'multinest', 'muSx')):
        tblvals[(target, 'multinest', 'muSx')][0] *= -1.
    if tblvals.has_key((target, 'multinest', 'muRelx')):
        tblvals[(target, 'multinest', 'muRelx')][0] *= -1.

    #Assign muber formats
    fmts[(target,'phot', 't0')]='%12.2f'
    fmts[(target,'phot', 'u0')]='%12.3f'
    fmts[(target,'phot', 'tE')]='%12.1f'
    fmts[(target,'phot', 'piEE')]='%12.3f'
    fmts[(target,'phot', 'piEN')]='%12.3f'
    fmts[(target,'phot', 's')]='%12.3f'
    fmts[(target,'phot', 'q')]='%12.3f'
    fmts[(target,'phot', 'alpha')]='%12.3f'
    fmts[(target,'phot', 'omega')]='%12.3f'
    fmts[(target,'phot', 'sdot/s')]='%12.3f'
    fmts[(target,'phot', 'chisq')]='%12.1f'
    fmts[(target,'phot', 'dof')]='%12d'
    fmts[(target,'phot', 'fsource')]='%12.3f'
    fmts[(target,'phot', 'fblend')]='%12.3f'
    fmts[(target,'phot', 'thetaS0x')]='%12.3f'
    fmts[(target,'phot', 'thetaS0y')]='%12.3f'
    fmts[(target,'phot', 'muSx')]='%12.3f'
    fmts[(target,'phot', 'muSy')]='%12.3f'
    fmts[(target,'phot', 'muRelx')]='%12.3f'
    fmts[(target,'phot', 'muRely')]='%12.3f'
    fmts[(target,'phot', 'thetaE')]='%12.3f'
    fmts[(target,'phot', 'Mass')]='%12.3f'
    fmts[(target,'phot', 'fratio')]='%12.3f'
    fmts[(target,'phot', 'Isource')]='%12.3f'

    fmts[(target,'multinest', 't0')]='%12.2f'
    fmts[(target,'multinest', 'u0')]='%12.3f'
    fmts[(target,'multinest', 'tE')]='%12.1f'
    fmts[(target,'multinest', 'piEE')]='%12.3f'
    fmts[(target,'multinest', 'piEN')]='%12.3f'
    fmts[(target,'multinest', 's')]='%12.3f'
    fmts[(target,'multinest', 'q')]='%12.3f'
    fmts[(target,'multinest', 'alpha')]='%12.3f'
    fmts[(target,'multinest', 'omega')]='%12.3f'
    fmts[(target,'multinest', 'sdot/s')]='%12.3f'
    fmts[(target,'multinest', 'chisq')]='%12.1f'
    fmts[(target,'multinest', 'dof')]='%12d'
    fmts[(target,'multinest', 'fsource')]='%12.3f'
    fmts[(target,'multinest', 'fblend')]='%12.3f'
    fmts[(target,'multinest', 'thetaS0x')]='%12.3f'
    fmts[(target,'multinest', 'thetaS0y')]='%12.3f'
    fmts[(target,'multinest', 'muSx')]='%12.3f'
    fmts[(target,'multinest', 'muSy')]='%12.3f'
    fmts[(target,'multinest', 'muRelx')]='%12.3f'
    fmts[(target,'multinest', 'muRely')]='%12.3f'
    fmts[(target,'multinest', 'thetaE')]='%12.3f'
    fmts[(target,'multinest', 'Mass')]='%12.3f'
    fmts[(target,'multinest', 'fratio')]='%12.3f'
    fmts[(target,'multinest', 'Isource')]='%12.3f'

    #blank entries    
    for p in range(len(allpars)):   
        if not tblvals.has_key((target, 'phot', allpars[p])):
            tblvals[(target, 'phot', allpars[p])] = ['---','','']
            fmts[(target,'phot', allpars[p])]='%12s'

        if not tblvals.has_key((target, 'multinest', allpars[p])):
            tblvals[(target, 'multinest', allpars[p])] = ['---','','']
            fmts[(target,'multinest', allpars[p])]='%12s'

    #Names for LateX table
    printnames = {}
    printnames['t0'] = '$t_0$ (HJD - 2450000)'
    printnames['u0'] = '$u_0$'
    printnames['tE'] = '$t_E$ (days)'
    printnames['piEN'] = '$\\pi_{E,N}$'
    printnames['piEE'] = '$\\pi_{E,E}$'
    printnames['thetaS0x'] = '$\\theta_{s,0,x}$'
    printnames['thetaS0y'] = '$\\theta_{s,0,y}$'
    printnames['muSx'] = '$\\mu_{s,E}~(\mathrm{mas~yr^{-1})}$'
    printnames['muSy'] = '$\\mu_{s,N}~(\mathrm{mas~yr^{-1})}$'
    printnames['muRelx'] = '$\\mu_{\\mathrm{rel},E}~(\mathrm{mas~yr^{-1})}$'
    printnames['muRely'] = '$\\mu_{\\mathrm{rel},N}~(\mathrm{mas~yr^{-1})}$'
    printnames['thetaE'] = '$\\theta_E$ (mas)' 
    printnames['Mass'] = 'Mass ($M_{\\odot}$)'
    printnames['s'] = 's'
    printnames['q'] = 'q'
    printnames['alpha'] = '$\\alpha$'
    printnames['omega'] = '$\\omega$'
    printnames['sdot/s'] = '$\\dot{s}/s$'
    printnames['chisq'] = '$\\chi^{2}$'
    printnames['dof'] = 'N$_{\mathrm{dof}}$'
    printnames['Isource'] = '$I_{\mathrm{OGLE}}$'
    printnames['fratio'] = '$f_{b}/f_{s}$'

    #Print LateX table
    t = target
    for pp in allpars:
        if tblvals[(t, 'phot', pp)] == ['---','',''] and tblvals[(t, 'multinest', pp)] == ['---','','']:
            continue

        cmd = '{0:45s}'.format(printnames[pp])
        cmd += ' & ' + fmts[(t, 'phot',  pp)] % (tblvals[(t, 'phot', pp)][0])
        if (tblvals[(t, 'phot', pp)][0] != '') and (tblvals[(t, 'phot', pp)][1] != 0):
            cmd += '$^{+' + fmts[(t,'phot', pp)] % (tblvals[(t, 'phot', pp)][1]) + '}'
            cmd += '_{-' + fmts[(t,'phot', pp)] % (tblvals[(t, 'phot', pp)][2]) + '}$'
        else:
            cmd += '{0:34s}'.format('')

        cmd += ' & ' + fmts[(t,'multinest', pp)] % (tblvals[(t, 'multinest', pp)][0])
        if (tblvals[(t, 'multinest', pp)][0] != '') and (tblvals[(t, 'multinest', pp)][1] != 0):
            cmd += '$^{+' + fmts[(t,'multinest', pp)] % (tblvals[(t, 'multinest', pp)][1]) + '}'
            cmd += '_{-' + fmts[(t,'multinest', pp)] % (tblvals[(t, 'multinest', pp)][2]) + '}$'
        else:
            cmd += '{0:34s}'.format('')

        cmd += ' \\\\ [0.2cm]'
        print cmd

    return


def weighted_quantile(values, quantiles, sample_weight=None, values_sorted=False, old_style=False):
    """ Very close to numpy.percentile, but supports weights.
    NOTE: quantiles should be in [0, 1]!
    :param values: numpy.array with data
    :param quantiles: array-like with many quantiles needed
    :param sample_weight: array-like of the same length as `array`
    :param values_sorted: bool, if True, then will avoid sorting of initial array
    :param old_style: if True, will correct output to be consistent with numpy.percentile.
    :return: numpy.array with computed quantiles.
    """
    values = np.array(values)
    quantiles = np.array(quantiles)
    if sample_weight is None:
        sample_weight = np.ones(len(values))
    sample_weight = np.array(sample_weight)
    assert np.all(quantiles >= 0) and np.all(quantiles <= 1), 'quantiles should be in [0, 1]'

    if not values_sorted:
        sorter = np.argsort(values)
        values = values[sorter]
        sample_weight = sample_weight[sorter]

    weighted_quantiles = np.cumsum(sample_weight) - 0.5 * sample_weight
    if old_style:
        # To be convenient with np.percentile
        weighted_quantiles -= weighted_quantiles[0]
        weighted_quantiles /= weighted_quantiles[-1]
    else:
        weighted_quantiles /= np.sum(sample_weight)
        
    return np.interp(quantiles, weighted_quantiles, values)            

def mmjd_to_year(t0_mmjd):
    """
    Input format is in
        MJD - 50000
    or
        HJD - 2450000
    Peculiar, but used by microlensing community.

    Return Value
    ------------
    fractional year
    
    """
    t_ref_yr = 2009.0
    t_ref_mmjd = 4832.5
    t_ref_mmjd_leap = 4833.0
    
    # # Convert HJD - 2450000 into calendar year using Jan 1 2009 as calibrator
    # t0_yr = t_ref_yr + (t0_mmjd - t_ref_mmjd) / 365.24

    # Convert HJD-245000 into calendar year using Jan 1 2009 as calibrator
    t0_yr = 2009.0 + (t0_mmjd - 4832.5) / 365.0
    leapYr = np.where((t0_mmjd > 5927.5) & (t0_mmjd < 6293.) )[0]
    if (len(leapYr) > 0):
        t0_yr[leapYr] =  2009. + (t0_mmjd[leapYr] - 4833.) / 366.
    
    
    return t0_yr

def year_to_mmjd(t0_yr):
    t_ref_yr = 2009.0
    t_ref_mmjd = 4832.5
    t_ref_mmjd_leap = 4833.0

    # # Convert HJD - 2450000 into calendar year using Jan 1 2009 as calibrator
    # t0_mmjd = ((t0_yr - t_ref_yr) * 365.24) + t_ref_mmjd
        
    # Convert HJD-245000 into calendar year using Jan 1 2009 as calibrator
    t0_mmjd = ((t0_yr - 2009.0) * 365.0) + 4832.5
    leapYr = np.where((t0_mmjd > 5927.5) & (t0_mmjd < 6293.) )[0]
    if (len(leapYr) > 0):
        t0_mmjd[leapYr] =  ((t0_yr - 2009.0) * 366.0) + 4833.0
    
    return t0_mmjd
            
