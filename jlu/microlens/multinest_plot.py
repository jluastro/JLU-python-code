import pylab as py
import numpy as np
from astropy.table import Table
from astropy import table
from jlu.microlens import MCMC_LensModel
from jlu.util import fileUtil

def plot_OB120169(root_dir='/Users/jlu/work/microlens/OB120169/analysis_2015_09_18/',
                  analysis_dir='analysis_ob120169_2015_09_18_a4_m20_w4_MC100/',
                  points_dir='points_d/', mnest_dir='mnest_evan_2015_08/ay/',
                  mnest_root='ay'):
        
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
    tab = load_mnest_results_OB120169(root_dir=root_dir,
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
    py.ylim(-4, 8)
    py.xlim(-0.5, 4.0)
    xticks = np.arange(0, 4.1, 1, dtype=int)
    py.xticks(xticks, fontsize=fontsize2)
    py.yticks(fontsize=fontsize2)

    paxes = py.subplot(2, 2, 2)
    py.plot(tmod-t0, thetaSy_model-originY, 'b-')
    py.plot(tmod-t0, thetaS_nolensy-originY, 'b--')
    py.errorbar(tobs-t0, thetaSy_data-originY, yerr=yerr_data, fmt='k.')
    py.ylabel(r'$\Delta\delta$ (mas)', fontsize=fontsize1)
    py.xlabel('t - t$_{\mathrm{o}}$ (yr)', fontsize=fontsize1)  
    py.ylim(-2, 4)
    py.xlim(-0.5, 4.0)
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
    py.ylim(-5, 5)
    py.xlim(-4, 8)
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
    py.xlim(0, 20)
    py.ylim(0, 0.2)
    py.ylabel('Relative probability', fontsize=fontsize1, labelpad=10)
    py.xticks(fontsize=fontsize2)
    py.yticks(fontsize=fontsize2)


    outdir = root_dir + mnest_dir + 'plots/'    
    outfile =  outdir + 'plot_OB120169_data_vs_model.png'
    fileUtil.mkdir(outdir)
    print 'writing plot to file ' + outfile
    
    py.savefig(outfile)

    return


def load_mnest_results_OB120169(root_dir='/Users/jlu/work/microlens/OB120169/analysis_2015_09_18/',
                                mnest_dir='mnest_evan_2015_08/ay/',
                                mnest_root='ay'):
    
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

def mass_posterior_OB120169(root_dir='/Users/jlu/work/microlens/OB120169/analysis_2015_09_18/',
                            analysis_dir='analysis_ob120169_2015_09_18_a4_m20_w4_MC100/'):
    
    target = 'OB120169_R'
    display_name = 'OB120169'
    
    # Load Model Posteriors for both
    #    u_o > 0 (p=plus) and
    #    u_o < 0 (m=minus).
    tab_p = load_mnest_results_OB120169(root_dir=root_dir,
                                        mnest_dir=analysis_dir + 'multiNest/ay/',
                                        mnest_root='ay')
    tab_m = load_mnest_results_OB120169(root_dir=root_dir,
                                        mnest_dir=analysis_dir + 'multiNest/ax/',
                                        mnest_root='ax')
    params = tab_p.keys()
    pcnt = len(params)

    # Clean table and only keep valid results.
    Masslim = 0.0
    ind_p = np.where((tab_p['Mass'] >= Masslim))[0]
    ind_m = np.where((tab_m['Mass'] >= Masslim))[0]
    tab_p = tab_p[ind_p]
    tab_m = tab_m[ind_m]

    # Combine the two tables together.
    tab = table.vstack(tab_p, tab_m)

    # Make a histogram of the mass using the weights. This creates the 
    # marginalized 1D posteriors.
    fontsize1 = 18
    fontsize2 = 14       
    
    py.figure(1)
    py.clf()
    bins = np.arange(0, 20, 0.2)
    n, foo, patch = py.hist(tab['Mass'], normed=True,
                             histtype='step', weights=tab['weights'],
                             bins=bins, linewidth=1.5)
    bin_centers = (bins[:-1] + bins[1:]) / 2
    py.xlabel('Mass')
    xtitle = r'Lens Mass (M$_{\odot}$)'
    py.xlabel(xtitle, fontsize=fontsize1, labelpad=10)
    py.xlim(0, 20)
    py.ylabel('Relative probability', fontsize=fontsize1, labelpad=10)
    py.xticks(fontsize=fontsize2)
    py.yticks(fontsize=fontsize2)

    ##########
    # Lower Mass Limit
    ##########
    mass_cdf_rev = n[::-1].cumsum()
    mass_cdf_rev /= mass_cdf_rev[-1]  # ensure normalized
    mass_bins_rev = bin_centers[::-1]

    idx = np.where(mass_cdf_rev <= 0.997)[0]
    mass_low_limit = mass_bins_rev[idx[-1]]
    print 'Mass Lower Limit (99.7%) - integrated from high mass to low mass'
    print ' M > {0:5.2f}'.format(mass_low_limit)

    py.axvline(mass_low_limit, color='red', linestyle='--')
    ax = py.axis()
    py.text(mass_low_limit + 0.1, 0.9*ax[3],
            'M>{0:5.2f} with 99.7% confidence'.format(mass_low_limit), fontsize=12)

    ##########
    # Save figure
    ##########
    outdir = root_dir + analysis_dir + 'multiNest/plots/'    
    outfile =  outdir + 'plot_OB120169_mass_posterior_combo.png'
    fileUtil.mkdir(outdir)
    print 'writing plot to file ' + outfile
    
    py.savefig(outfile)

    return

