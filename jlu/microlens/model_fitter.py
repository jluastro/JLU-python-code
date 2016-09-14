import numpy as np
import pylab as plt
import scipy.stats
import pdb
import pymultinest
import os
import errno
from jlu.microlens import model
import time
from astropy.table import Table
from matplotlib.ticker import MultipleLocator

def make_gen(min, max):
    return scipy.stats.uniform(loc=min, scale=max-min)

def make_norm_gen(mean, std):
    return scipy.stats.norm(loc=mean, scale=std)

def make_t0_gen(t, mag):
    """Get an approximate t0 search range by finding the brightest point
    and then searching days where flux is higher than 80% of this peak.
    """
    mag_min = np.min(mag)  # min mag = brightest
    delta_mag = np.max(mag) - mag_min
    idx = np.where(mag < (mag_min + (0.2 * delta_mag)))[0]
    t0_min = t[idx[0]]
    t0_max = t[idx[-1]]

    # Pad by and extra 40% in case of gaps.
    t0_min -= 0.4 * (t0_max - t0_min)
    t0_max += 0.4 * (t0_max - t0_min)

    return make_gen(t0_min, t0_max)

def make_muS_xy_gen(t, pos):
    """Get an approximate muS search range by looking at the best fit
    straight line to the astrometry. Then allows lots of free space.

    Inputs
    ======
    t: array of times in days
    pos: array of positions in arcsec

    Return
    ======
    uniform generator for velocity in mas/yr
    """
    # Convert t to years temporarily.
    t_yr = t / model.days_per_year
    par, cov = np.polyfit(t_yr, pos, 1, cov=True)
    vel = par[0] * 1e3  # mas/yr
    vel_err = (cov[0][0]**0.5) * 1e3 # mas/yr

    scale_factor = 100.0

    vel_lo = vel - scale_factor*vel_err
    vel_hi = vel + scale_factor*vel_err

    return make_gen(vel_lo, vel_hi)
    

def random_prob(generator, x):
    value = generator.ppf(x)
    ln_prob = generator.logpdf(value)
    return value, ln_prob

def multinest_pspl(data, n_live_points=1000, saveto='./mnest_pspl/', runcode='aa'):

    t_phot = data['t_phot']
    imag = data['imag']
    imag_err = data['imag_err']

    t_ast = data['t_ast']
    xpos = data['xpos']
    ypos = data['ypos']
    xpos_err = data['xpos_err']
    ypos_err = data['ypos_err']

    # Model Parameters: mL, t0, xS0, beta, muL, muS, dL, dS, imag_base
    mL_gen = make_gen(0, 50)
    t0_gen = make_t0_gen(t_phot, imag)
    xS0_x_gen = make_gen(data['xpos'].min(), data['xpos'].max())
    xS0_y_gen = make_gen(data['ypos'].min(), data['ypos'].max())
    beta_gen = make_gen(-1, 1)
    muL_x_gen = make_gen(-20, 20)
    muL_y_gen = make_gen(-20, 20)
    muS_x_gen = make_muS_xy_gen(t_ast, xpos)
    muS_y_gen = make_muS_xy_gen(t_ast, ypos)
    dL_gen = make_gen(3000, 5000)
    dS_gen = make_gen(6000, 10000)
    imag_base_gen = make_gen(18.5, 19.5)
	
    def priors(cube, ndim, nparams):
        cube[0] = mL_gen.ppf(cube[0])
        cube[1] = t0_gen.ppf(cube[1])
        cube[2] = xS0_x_gen.ppf(cube[2])
        cube[3] = xS0_y_gen.ppf(cube[3])
        cube[4] = beta_gen.ppf(cube[4])
        cube[5] = muL_x_gen.ppf(cube[5])
        cube[6] = muL_y_gen.ppf(cube[6])
        cube[7] = muS_x_gen.ppf(cube[7])
        cube[8] = muS_y_gen.ppf(cube[8])
        cube[9] = dL_gen.ppf(cube[9])
        cube[10] = dS_gen.ppf(cube[10])
        cube[11] = imag_base_gen.ppf(cube[11])
        
        return 
	
    def likelihood(cube, ndim, nparams):
        mL = cube[0]
        t0 = cube[1]
        xS0 = np.array([cube[2], cube[3]])
        beta = cube[4]
        muL = np.array([cube[5], cube[6]])
        muS = np.array([cube[7], cube[8]])
        dL = cube[9]
        dS = cube[10]
        imag_base = cube[11]

        pspl = model.PSPL(mL, t0, xS0, beta, muL, muS, dL, dS, imag_base)

        lnL_phot = pspl.likely_photometry(t_phot, imag, imag_err)
        lnL_ast = pspl.likely_astrometry(t_ast, xpos, ypos, xpos_err, ypos_err)

        lnL = lnL_phot.mean() + lnL_ast.mean()

        fmt = 'mL={0:4.1f} t0={1:7.1f} xS0=[{2:8.4f}, {3:8.4f}] beta={4:5.2f} '
        fmt += 'muL=[{5:6.2f}, {6:6.2f}] muS=[{7:6.2f}, {8:6.2f}] dL={9:5.0f} dS={10:5.0f} '
        fmt += 'imag={11:4.1f} lnL={12:12.2e}'
        
        print fmt.format(mL, t0, xS0[0], xS0[1], beta, muL[0], muL[1], muS[0], muS[1], dL, dS, imag_base, lnL)

        return lnL


    num_dims = 12
    num_params = 12  #cube will have this many dimensions
    ev_tol = 0.3
    samp_eff = 0.8
    n_live_points = 100

    #Create param file
    _run = open(saveto + runcode + '_params.run', 'w')
    _run.write('Num Dimensions: %d\n' % num_dims)
    _run.write('Num Params: %d\n' % num_params)
    _run.write('Evidence Tolerance: %.1f\n' % ev_tol)
    _run.write('Sampling Efficiency: %.1f\n' % samp_eff)
    _run.write('Num Live Points: %d\n' % n_live_points)
    _run.close()

    startdir = os.getcwd()
    os.chdir(saveto)
        
    pymultinest.run(likelihood, priors, num_dims, n_params=num_params,
					outputfiles_basename = runcode + '_', 
					verbose=True, resume=False, evidence_tolerance=ev_tol,
					sampling_efficiency=samp_eff, n_live_points=n_live_points,
					multimodal=True, n_clustering_params=num_dims,
                    importance_nested_sampling=False)              

    os.chdir(startdir)

    return


def load_mnest_results(mnest_dir, mnest_root):
    
    root = mnest_dir + mnest_root + '_'
    tab = Table.read(root + '.txt', format='ascii')
    
    # Convert to log(likelihood)
    tab['col2'] /= -2.0
    
    # Rename the parameter columns. This is hard-coded to match the
    # above run() function.

    tab.rename_column('col1', 'weights')
    tab.rename_column('col2', 'logLike')
    tab.rename_column('col3', 'mL')
    tab.rename_column('col4', 't0')
    tab.rename_column('col5', 'xS0_x')
    tab.rename_column('col6', 'xS0_y')
    tab.rename_column('col7', 'beta')
    tab.rename_column('col8', 'muL_x')
    tab.rename_column('col9', 'muL_y')
    tab.rename_column('col10', 'muS_x')
    tab.rename_column('col11', 'muS_y')
    tab.rename_column('col12', 'dL')
    tab.rename_column('col13', 'dS')
    tab.rename_column('col14', 'imag_base')
            
    return tab

def get_best_fit(mnest_dir, mnest_root):
    # Identify best-fit model
    tab = load_mnest_results(mnest_dir, mnest_root)
    
    params = tab.keys()
    pcnt = len(params)

    best = np.argmax(tab['logLike'])
    maxLike = tab['logLike'][best]

    tab_best = tab[best]

    return tab_best


def plot_posteriors(mnest_dir, mnest_root):
    """
    Plots posteriors using pair_posterior code
    """
    outdir = mnest_dir

    # Load up the table and remove the weights column
    tab = load_mnest_results(mnest_dir=mnest_dir, mnest_root=mnest_root)
    weights = tab['weights']
    tab.remove_columns(('weights', 'logLike'))

    # Make the subdirectory
    try:
        os.mkdir(outdir + '/plots')
        os.mkdir(outdir + '/plots/posteriors')
    except OSError as exc:
        if exc.errno != errno.EEXIST:
            raise exc
        pass
    
    # Make the plots
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
    
    plt.close(10)
    plt.figure(10, figsize = (20,20))

    # Marginalized 1D
    for ii in range(pcnt):
        ax = plt.subplot(pcnt, pcnt, ii*(pcnt+1)+1)
        plt.setp(ax.get_xticklabels(), fontsize=fontsize)
        plt.setp(ax.get_yticklabels(), fontsize=fontsize)
        n, bins, patch = plt.hist(atpy_table[params[ii]], normed=True,
                                 histtype='step', weights=weights, bins=50)
        plt.xlabel(params[ii], size=fontsize)
        plt.ylim(0, n.max()*1.1)
        
        ax.get_xaxis().get_major_formatter().set_useOffset(False)        
        ax.get_xaxis().get_major_formatter().set_scientific(False)

        x_limits = ax.get_xlim()
        y_limits = ax.get_ylim()

        x_sep = (x_limits[1] - x_limits[0]) / 2.1
        y_sep = (y_limits[1] - y_limits[0]) / 2.1
        x_majorLocator = MultipleLocator(x_sep)
        y_majorLocator = MultipleLocator(y_sep)

        ax.xaxis.set_major_locator(x_majorLocator)
        ax.yaxis.set_major_locator(y_majorLocator)

    # Bivariates
    for ii in range(pcnt - 1):
        for jj in range(ii+1, pcnt):
            ax = plt.subplot(pcnt, pcnt, ii*pcnt + jj+1)
            plt.setp(ax.get_xticklabels(), fontsize=fontsize)
            plt.setp(ax.get_yticklabels(), fontsize=fontsize)

            (H, x, y) = np.histogram2d(atpy_table[params[jj]], atpy_table[params[ii]],
                                       weights=weights, bins=50)
            xcenter = x[:-1] + (np.diff(x) / 2.0)
            ycenter = y[:-1] + (np.diff(y) / 2.0)

            plt.contourf(xcenter, ycenter, H.T, cmap=plt.cm.gist_yarg)

            plt.xlabel(params[jj], size=fontsize)
            plt.ylabel(params[ii], size=fontsize)

            ax.get_xaxis().get_major_formatter().set_useOffset(False)        
            ax.get_xaxis().get_major_formatter().set_scientific(False)        

            ax.get_yaxis().get_major_formatter().set_useOffset(False)        
            ax.get_yaxis().get_major_formatter().set_scientific(False)        
                        
            x_limits = ax.get_xlim()
            y_limits = ax.get_ylim()

            x_sep = (x_limits[1] - x_limits[0]) / 2.1
            y_sep = (y_limits[1] - y_limits[0]) / 2.1
            x_majorLocator = MultipleLocator(x_sep)
            y_majorLocator = MultipleLocator(y_sep)

            ax.xaxis.set_major_locator(x_majorLocator)
            ax.yaxis.set_major_locator(y_majorLocator)

    #if title != None:
    #    plt.suptitle(title)
    plt.subplots_adjust(wspace=0.3, hspace=0.3)
    if outfile != None:
        plt.savefig(outfile, bbox_inches='tight', pad_inches=0.05)

    return

