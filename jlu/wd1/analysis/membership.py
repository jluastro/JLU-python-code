import numpy as np
import pylab as py
from numpy import random
from scipy import stats
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse 
import pdb
import os
import pymultinest
from astropy.table import Table
from hst_flystar import astrometry
import time
import math
from jlu.util import fileUtil
from astropy.table import Table
from astropy.io import ascii
from scipy import optimize
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes, inset_axes
from mpl_toolkits.axes_grid1.inset_locator import mark_inset

angle = 46.43


def make_pi_gen():
    """
    piC = fraction of stars belonging to the gaussian
    """
    return stats.uniform(loc = 0, scale = 1)  #Uniform distribution from 0 - 1

def make_v_gen():
    """
    Generates prior for gaussian X velocity
    """
    return stats.uniform(loc=-4, scale=12)

def make_sig_gen():
    """
    Generates prior for gaussian semi-MAJOR axis vel standard deviation
    """
    return stats.uniform(loc=0, scale=8)

def make_theta_gen():
    """
    Generates prior for theta = angle between x axis and semi-MAJOR axis for the gaussian
    population. Returns angle in radians
    """
    #Will allow angle between 0 and 180 degrees (in radians)
    return stats.uniform(loc=0, scale=3.14159)

def make_clust_vx_gen():
    """
    Generates prior for gaussian Y velocity
    """
    clustVy_mean = 0.0 # since we will transform into the cluster's reference frame
    clustVy_std = 0.2  # Rough sigma of cluster 

    return stats.norm(loc=clustVy_mean, scale=clustVy_std)

def make_clust_vy_gen():
    """
    Generates prior for gaussian Y velocity
    """
    clustVy_mean = 0.0 # since we will transform into the cluster's reference frame
    clustVy_std = 0.2  # Rough sigma of cluster 

    return stats.norm(loc=clustVy_mean, scale=clustVy_std)

def make_clust_sigA_gen():
    """
    Generates prior for gaussian Y velocity
    """
    # clustSigA_mean = 0.0 # since we will transform into the cluster's reference frame
    # clustSigA_std = 0.2  # Rough sigma of cluster 
    # return stats.norm(loc=clustVy_mean, scale=clustVy_std)

    return stats.uniform(loc=0, scale=8)

def make_clust_sigB_gen():
    """
    Generates prior for gaussian Y velocity
    """
    # clustSigB_mean = 0.0 # since we will transform into the cluster's reference frame
    # clustSigB_std = 0.2  # Rough sigma of cluster 
    # return stats.norm(loc=clustVy_mean, scale=clustVy_std)

    return stats.uniform(loc=0, scale=8)

def make_clust_theta_gen():
    """
    Generates prior for theta = angle between x axis and semi-MAJOR axis for the gaussian
    population. Returns angle in radians
    """
    # #Will allow angle between 0 and 180 degrees (in radians)
    # mean = 0.0
    # stdev = 0.001
    # return stats.norm(loc=mean, scale=stdev)
    
    return stats.uniform(loc=0, scale=3.14159)



pi_gen = make_pi_gen()
v_gen = make_v_gen()
sig_gen = make_sig_gen()
theta_gen = make_theta_gen()
clust_vx_gen = make_clust_vx_gen()
clust_vy_gen = make_clust_vy_gen()
clust_sigA_gen = make_clust_sigA_gen()
clust_sigB_gen = make_clust_sigB_gen()
clust_theta_gen = make_clust_theta_gen()

    
def random_pi(randNum):
    """
    Given random number from 0-1, produces piC value at which probability of
    getting that value or less is equal to random number 

    Returns piC, log_prob_piC
    """
    pi = pi_gen.ppf(randNum)
    log_prob_pi = pi_gen.logpdf(pi)
    
    return pi, log_prob_pi

def random_v(randNum):
    """
    See comments for random_piC
    """
    v = v_gen.ppf(randNum)
    log_prob_v = v_gen.logpdf(v)

    return v, log_prob_v

def random_sig(randNum):
    """
    See comments for random_piC
    """
    sig = sig_gen.ppf(randNum)
    log_prob_sig = sig_gen.logpdf(sig)

    return sig, log_prob_sig

def random_theta(randNum):
    """
    Returns array [theta, prob_theta]
    """
    theta = theta_gen.ppf(randNum)
    log_prob_theta = theta_gen.logpdf(theta)
    
    return theta, log_prob_theta


def random_clust_vx(randNum):
    """
    See comments for random_piC
    """
    clust_vx = clust_vx_gen.ppf(randNum)
    log_prob_clust_vx = clust_vx_gen.logpdf(clust_vx)

    return clust_vx, log_prob_clust_vx

def random_clust_vy(randNum):
    """
    See comments for random_piC
    """
    clust_vy = clust_vy_gen.ppf(randNum)
    log_prob_clust_vy = clust_vy_gen.logpdf(clust_vy)

    return clust_vy, log_prob_clust_vy

def random_clust_sigA(randNum):
    """
    See comments for random_piC
    """
    sig = clust_sigA_gen.ppf(randNum)
    log_prob_sig = clust_sigA_gen.logpdf(sig)

    return sig, log_prob_sig

def random_clust_sigB(randNum):
    """
    See comments for random_piC
    """
    sig = clust_sigB_gen.ppf(randNum)
    log_prob_sig = clust_sigB_gen.logpdf(sig)

    return sig, log_prob_sig

def random_clust_theta(randNum):
    """
    Returns array [theta, prob_theta]
    """
    theta = clust_theta_gen.ppf(randNum)
    log_prob_theta = clust_theta_gen.logpdf(theta)
    
    return theta, log_prob_theta




def run(catalogfile, vel_err, mag_err, N_gauss, outdir, rotate=True):
    """
    PyMultiNest run to determine cluster membership, using PM catalog and
    applying vel_err and mag_err cuts. Output is put in newly-created outdir
    directory (must be a string).

    Parameters:
    catalogflie --> String containing the name of a FITS catalog.
    vel_err --> The maximum allowed velocity error for stars to be included.
    mag_err --> The maximum allowed magnitude error for stars to be included.
    N_gauss --> number bivariate gaussian, where N_gauss <= 4
    outdir --> The output directory name.
    
    Keywords:
    rotate = 1 --> rotate star velocities into RA/DEC format, as opposed
    to X,Y
    """
    # Load data for full field, extract velocities (already converted to mas)
    d = loadData(catalogfile, vel_err, mag_err, rotate=rotate)
    
    star_Vx = d['fit_vx']
    star_Vy = d['fit_vy']
    star_Sigx = d['fit_vxe']
    star_Sigy = d['fit_vye']

    N_stars = len(d)
        
    def print_param(pname, val, logp, headerFirst=False):
        rowHead = '{0:6s}  '
        colHead = ' val_{0} (  logp_{0} )'
        colVal = '{0:6.3f} ({1:9.2e})'

        if headerFirst:
            outhdr = '  '.join([colHead.format(k) for k in range(N_gauss)])
            print rowHead.format('') + outhdr

        outstr = '  '.join([colVal.format(val[k], logp[k]) for k in range(N_gauss)])
        print rowHead.format(pname) + outstr

        return

    def priors(cube, ndim, nparams):
        return

    def likelihood(cube, ndim, nparams):
        """
        Define the likelihood function (from Clarkson+12, Hosek+15)
        """
        #start the timer
        t0 = time.time()
        
        ####################
        #Set up model params
        ####################
        # Number of parameters per Gaussian:
        N_per_gauss = 6

        # Make arrays for the paramters of each Gaussian
        pi = np.arange(N_gauss, dtype=float)
        vx = np.arange(N_gauss, dtype=float)
        vy = np.arange(N_gauss, dtype=float)
        sigA = np.arange(N_gauss, dtype=float)
        sigB = np.arange(N_gauss, dtype=float)
        theta = np.arange(N_gauss, dtype=float)

        # Make arrays for the prior probability of each paramter
        logp_pi = np.arange(N_gauss, dtype=float)
        logp_vx = np.arange(N_gauss, dtype=float)
        logp_vy = np.arange(N_gauss, dtype=float)
        logp_sigA = np.arange(N_gauss, dtype=float)
        logp_sigB = np.arange(N_gauss, dtype=float)
        logp_theta = np.arange(N_gauss, dtype=float)

        # Set the fraction of stars in each Gaussian
        for kk in range(N_gauss):
            pi[kk], logp_pi[kk] = random_pi(cube[kk*N_per_gauss + 0])
            
        # Make sure all the sum(pi) = 1.
        pi /= pi.sum()

        # Sort the field pi values such that they are always ranked from
        # smallest to largest.
        sidx = pi[1:].argsort()
        pi[1:] = pi[1:][sidx]
        logp_pi[1:] = logp_pi[1:][sidx]

        # Re-set the cube values. Note this is AFTER sorting.
        for kk in range(N_gauss):
            cube[kk*N_per_gauss + 0] = pi[kk]

        
        # Set the other Gaussian parameters.
        for kk in range(N_gauss):
            # Treat the cluster gaussian (the first, most compact one)
            # with a special prior function.
            if kk == 0:
                rand_vx = random_clust_vx
                rand_vy = random_clust_vy
                rand_sigA = random_clust_sigA
                rand_sigB = random_clust_sigB
                rand_theta = random_clust_theta
            else:
                rand_vx = random_v
                rand_vy = random_v
                rand_sigA = random_sig
                rand_sigB = random_sig
                rand_theta = random_theta

            # Velocity centr
            vx[kk], logp_vx[kk] = rand_vx(cube[kk*N_per_gauss + 1])
            cube[kk*N_per_gauss + 1] = vx[kk]
            
            vy[kk], logp_vy[kk] = rand_vy(cube[kk*N_per_gauss + 2])
            cube[kk*N_per_gauss + 2] = vy[kk]

            # Major axis
            sigA[kk], logp_sigA[kk] = rand_sigA(cube[kk*N_per_gauss + 3])
            cube[kk*N_per_gauss + 3] = sigA[kk]

            # Minor axis
            sigB[kk], logp_sigB[kk] = rand_sigB(cube[kk*N_per_gauss + 4])
            cube[kk*N_per_gauss + 4] = sigB[kk]

            # Angle of major axis (in radians)
            theta[kk], logp_theta[kk] = rand_theta(cube[kk*N_per_gauss + 5])
            cube[kk*N_per_gauss + 5] = theta[kk]

            #Only want to consider gaussians where Sig A > Sig B
            if sigB[kk] > sigA[kk]:
                # print '#######################'
                # print '#######################'
                # print '#######################'
                # print '#######################'
                return -np.Inf

            # Check that all our prior probabilities are valid, otherwise abort
            # before expensive calculation.
            if ((logp_pi[kk] == -np.inf) or
                (logp_vx[kk] == -np.inf) or (logp_vy[kk] == -np.inf) or
                (logp_sigA[kk] == -np.inf) or (logp_sigB[kk] == -np.inf) or
                (logp_theta[kk] == -np.inf)):
                return -np.Inf
        
        ################################
        # Calculating likelihood function
        #  Likelihood = 
        #    \Sum(i=0 -> N_stars) \Sum(k=0 -> N_gauss)
        #        \pi_k * (2 \pi |\Sigma_k,i|)^{-1/2} *
        #        exp[ -1/2 * (\mu_i - \mu_k)^T \sigma_k,i (\mu_i - \mu_k) ]
        ################################        
        # Keep track of the probability for each star, each gaussian
        # component. We will add over components and multiply over stars.
        prob_gauss = np.zeros((N_gauss, N_stars), dtype=float)
        
        # L_{i,k}  Loop through the different gaussian components.
        for kk in range(N_gauss):
            # N_stars long array
            prob_gauss[kk, :] = prob_ellipse(star_Vx, star_Vy, star_Sigx, star_Sigy,
                                            pi[kk], vx[kk], vy[kk],
                                            sigA[kk], sigB[kk], theta[kk])

        # For each star, the total likelihood is the sum
        # of each component (before log).
        L_star = prob_gauss.sum(axis=0)  # This array should be N-stars long
        logL_star = np.log10(L_star)

        # Final likelihood
        logL = logL_star.sum()
        logL_tmp = logL

        # Add in log(prior probabilities) as well
        for kk in range(N_gauss):
            logL += logp_pi[kk]
            logL += logp_vx[kk]
            logL += logp_vy[kk]
            logL += logp_sigA[kk]
            logL += logp_sigB[kk]
            logL += logp_theta[kk]

        # Some printing
        print '*** logL = {0:9.2e}   w/priors = {1:9.2e}'.format(logL_tmp, logL)

        print_param('pi', pi, logp_pi, headerFirst=True)
        print_param('vx', vx, logp_vx)
        print_param('vy', vy, logp_vy)
        print_param('sigA', sigA, logp_sigA)
        print_param('sigB', sigB, logp_sigB)
        print_param('theta', theta, logp_theta)

        t1 = time.time()

        total = t1 - t0
        
        print 'TIME SPENT: ' + str(total)
        #pdb.set_trace()
        return logL


    #########################################
    # End Likelihoods
    # Begin running multinest
    #########################################
    #Make new directory to hold output
    fileUtil.mkdir(outdir)    
    outroot = outdir + '/mnest_'

    num_dims = 2 * 3 * N_gauss
    num_params = num_dims
    ev_tol = 0.3
    samp_eff = 0.8
    n_live_points = 300
    
    # Create param file
    _run = open(outroot + 'params.run', 'w')
    _run.write('Catalog: %s\n' % catalogfile)
    _run.write('Vel Err Cut: %.2f\n' % vel_err)
    _run.write('Mag Err Cut: %.2f\n' % mag_err)
    _run.write('Rotate: %s\n' % str(rotate))
    _run.write('Num Gauss: %d\n' % N_gauss)
    _run.write('Num Dimensions: %d\n' % num_dims)
    _run.write('Num Params: %d\n' % num_params)
    _run.write('Evidence Tolerance: %.1f\n' % ev_tol)
    _run.write('Sampling Efficiency: %.1f\n' % samp_eff)
    _run.write('Num Clustering Params: %d\n' % num_dims)
    _run.write('Num Live Points: %d\n' % n_live_points)
    _run.close()

    # Run multinest
    pymultinest.run(likelihood, priors, num_dims, n_params=num_params,
                    outputfiles_basename=outroot,
                    verbose=True, resume=False,
                    evidence_tolerance=ev_tol, sampling_efficiency=samp_eff,
                    n_live_points=n_live_points, multimodal=True,
                    n_clustering_params=num_dims,
                    importance_nested_sampling=False)

    return
    

def loadData(catalogfile, vel_err, mag_err, rotate=True):
    """
    Load proper motion catalog data, filtering using vel_err and 
    mag_err cuts. Converts proper motions (and errors) into mas/yr.
    
    Parameters:
    vel_err --> maximum allowed velocity error (in mas/yr).
    mag_err --> maximum allowed photometric error (in mag).
    """
    d = Table.read(catalogfile)
    
    # Converting from pix to mas, applying mag offsets
    pscale = astrometry.scale['WFC'] * 1e3  # mas/pix, from ACS comparison

    d['fit_vx'] *= pscale
    d['fit_vy'] *= pscale
    d['fit_vxe'] *= pscale 
    d['fit_vye'] *= pscale

    # Applying the error cuts, only to F153m filter
    lowErr = np.where((d['fit_vxe'] < vel_err) &
                      (d['fit_vye'] < vel_err) &
                      (d['me_2005_F814W'] < mag_err) &
                      (d['me_2010_F160W'] < mag_err) &
                      (d['me_2013_F160W'] < mag_err))
                        
    d_trim = d[lowErr]
    print 'Stars in trimmed catalog: {0} out of {1}'.format(len(d_trim), len(d))

    #--- If rotate flag, then rotate to RA/DEC ---#
    if rotate:
        vx_tmp, vy_tmp = vel_rotate(d_trim['fit_vx'], d_trim['fit_vy'], angle)  
        vxe_tmp, vye_tmp = velerr_rotate(d_trim['fit_vxe'], d_trim['fit_vye'], angle)

        d_trim['fit_vx'] = vx_tmp * -1.0
        d_trim['fit_vy'] = vy_tmp
        d_trim['fit_vxe'] = vxe_tmp
        d_trim['fit_vye'] = vye_tmp
        
    return d_trim


def load_results(rootdir, N_gauss):
    root = '%s/mnest_' % (rootdir)
    tab = Table.read(root + '.txt', format='ascii')

    # Convert to log(likelihood)
    tab['col2'] /= -2.0
    
    # Rename the parameter columns. This is hard-coded to match the
    # above run() function.
    tab.rename_column('col1', 'weights')
    tab.rename_column('col2', 'logLike')
    
    for kk in range(N_gauss):
        tab.rename_column('col{0}'.format(6*kk + 3), 'pi_{0}'.format(kk))
        tab.rename_column('col{0}'.format(6*kk + 4), 'vx_{0}'.format(kk))
        tab.rename_column('col{0}'.format(6*kk + 5), 'vy_{0}'.format(kk))
        tab.rename_column('col{0}'.format(6*kk + 6), 'sigA_{0}'.format(kk))
        tab.rename_column('col{0}'.format(6*kk + 7), 'sigB_{0}'.format(kk))
        tab.rename_column('col{0}'.format(6*kk + 8), 'theta_{0}'.format(kk))

    # Now sort based on logLikelihood
    tab.sort('logLike')

    return tab
        
def plot_posteriors(outdir, N_gauss):
    """
    Plots posteriors using pair_posterior code
    """
    tab = load_results(outdir, N_gauss)
    weights = tab['weights']
    tab.remove_columns(('weights', 'logLike'))
    fileUtil.mkdir(outdir + '/plots/')

    # Number of gaussians determines which pair_posterior code is used
    pair_posterior(tab, weights, N_gauss, outfile=outdir+'/plots/posteriors.png', title=outdir)
        
    return

def pair_posterior(tab, weights, N_gauss, outfile=None, title=None):
    """
    pair_posterior(astropy_table)

    :Arguments:
    astropy_table:       Contains 1 column for each parameter with samples.

    Produces a matrix of plots. On the diagonals are the marginal
    posteriors of the parameters. On the off-diagonals are the
    marginal pairwise posteriors of the parameters.
    """

    params = tab.keys()
    pcnt = len(params)

    fontsize = 10

    py.figure(figsize = (30,30))
    py.subplots_adjust(left=0.05, right=0.95, bottom=0.05, top=0.95)
    # Marginalized 1D
    for ii in range(pcnt):
        ax = py.subplot(pcnt, pcnt, ii*(pcnt+1)+1)
        py.setp(ax.get_xticklabels(), fontsize=fontsize)
        py.setp(ax.get_yticklabels(), fontsize=fontsize)
        n, bins, patch = py.hist(tab[params[ii]], normed=True,
                                 histtype='step', weights=weights, bins=50)
        py.xlabel(params[ii], size=fontsize)
        py.ylim(0, n.max()*1.1)

    # Bivariates
    for ii in range(pcnt - 1):
        for jj in range(ii+1, pcnt):
            ax = py.subplot(pcnt, pcnt, ii*pcnt + jj+1)
            py.setp(ax.get_xticklabels(), fontsize=fontsize)
            py.setp(ax.get_yticklabels(), fontsize=fontsize)

            (H, x, y) = np.histogram2d(tab[params[jj]], tab[params[ii]],
                                       weights=weights, bins=50)
            xcenter = x[:-1] + (np.diff(x) / 2.0)
            ycenter = y[:-1] + (np.diff(y) / 2.0)

            py.contourf(xcenter, ycenter, H.T, cmap=py.cm.gist_yarg)

            py.xlabel(params[jj], size=fontsize)
            py.ylabel(params[ii], size=fontsize)

    if title != None:
        py.suptitle(title)

    if outfile != None:
        py.savefig(outfile)

    return


def plot_posteriors_1D(outdir, N_gauss):
    """
    Makes separate plot for each of the 1D posteriors. Also produces 
    tale with fitted mu, sigma for each of the params

    Param file = parameters.txt, placed with plots
    """
    # Load results
    tab = load_results(outdir, N_gauss)
    
    # These will effectively tell us the likelihood of each model
    weights = tab['weights']
    tab.remove_columns(('weights', 'logLike'))

    # Get the parameters array
    params = tab.keys()
    N_params = len(params)
    N_params_per_gauss = 6
    
    # Dictionary with symbols for plotting purposes
    str_per_gauss = [r'$\pi_{{{0}}}$',
                     r'$\mu_{{RA,{0}}} \cos \delta$ (mas/yr)',
                     r'$\mu_{{Dec,{0}}}$ (mas/yr)',
                     r'$\sigma_{{a,{0}}}$ (mas/yr)',
                     r'$\sigma_{{b,{0}}}$ (mas/yr)',
                     r'$\theta_{{{0}}}$ (radians)']
        
    # Make a histogram for each parameter using the weights. This creates the 
    # marginalized 1D posteriors
    final_params = []
    
    py.close(2)
    py.figure(2)
    

    priors_clust = [pi_gen, clust_vx_gen, clust_vy_gen,
                    clust_sigA_gen, clust_sigB_gen, clust_theta_gen]
    priors_field = [pi_gen, v_gen, v_gen,
                    sig_gen, sig_gen, theta_gen]

    for kk in range(N_gauss):
        for gg in range(N_params_per_gauss):
            ii = (kk * N_params_per_gauss) + gg
            
            py.clf()
            ax = py.gca()
            n, bins, patch = py.hist(tab[params[ii]], weights=weights,
                                     histtype='step', normed=True,
                                     bins=300, linewidth=2)
            bin_centers = (bins[:-1] + bins[1:]) / 2.0
            
            # Plot priors
            if kk == 0:
                generator = priors_clust[gg]
            else:
                generator = priors_field[gg]
                                        
            x = bin_centers
            y = generator.pdf(bin_centers)
            py.plot(x, y, 'g-', linewidth = 2)

            py.axis([min(bin_centers) - 0.1, max(bin_centers) + 0.1, 0, max(n) + 0.3])
            py.xlabel(str_per_gauss[gg].format(kk))
            py.ylabel('Probability Density')

            # if N_gauss == 4:
            #     bad = 23
            # elif N_gauss == 5:
            #     bad = 30

            bad = -1
            
            if ((ii != bad)):
                # Attempt to fit data with a gaussian
                A, mu, sig = gaussfit(bin_centers, n)
    
                # Make the fit and plot
                fitcurve = A * np.exp(-(bin_centers - mu)**2 / (2. * sig**2))
                py.plot(bin_centers, fitcurve, 'r-', linewidth=2)
        
                # Place the determined mu, sig in the plot title
                py.title('Mean: {0:.4f}, STD: {1:.4f}'.format(mu, sig))
        
                # Save the params of the gauss fit
                result = [mu, sig]
                final_params.append(result)

            outfile = outdir+'/plots/posterior_'+params[ii]+'.png'
            py.savefig(outfile)
        
    # Output a text file with the final fit params
    final_params = np.transpose(np.array(final_params))
    outfile = outdir + '/plots/pop_parameters.txt'

    names = ('Param', 'Mu', 'Sigma')
    formats = {'Param': '%10s', 'Mu': '%8.5f', 'Sigma': '%8.5f'}
    t = Table([params, final_params[0], final_params[1]], names=names)
    ascii.write(t, outfile, Writer=ascii.FixedWidth, delimiter=None,
                formats=formats)
    return

def gaussfit(x, y):
    """
    Attempt to fit a 1D gaussian to the input data. y = Gaussian(x)

    Returns array with [Peak, mu, sigma]
    """
    #Define gaussian function
    def gauss(x, *p):
        A, mu, sigma = p
        return A * np.exp(-(x - mu)**2 / (2. * sigma**2))

    #Provide guesses for params
    A_guess = max(y)
    peakpos = np.where(y == A_guess)[0]
    mu_guess = x[peakpos]
    halfmax = np.where(y > A_guess / 3.0)[0]
    distance = np.array(x[halfmax][1:] - x[halfmax][:-1])
    if len(distance) == 0:
        sig_guess = (x[peakpos] - x[peakpos-1]) / 2.0
    else:            
        sig_guess = max(distance)
    
    p0 = [A_guess, mu_guess, sig_guess]
    params, var_matrix = optimize.curve_fit(gauss, x, y, p0=p0)
    
    return params


def show_distributions(catalogfile, velcut, magcut, outdir, N_gauss, rotate=True):
    """
    Given the catalog file and the parameter file from plot_posteriors_1D, plot the 
    corresponding cluster and field populations distributions. Applies the velocity cut and error cut

    Finds the parameter file in outdir/plots
    """
    d = loadData(catalogfile, velcut, magcut, rotate=rotate)
    star_Vx = d['fit_vx']
    star_Vy = d['fit_vy']
    
    # Transform velocities into reference frame of the cluster
    # Using velocity determined from UPDATED doit.py:
    # star_Vx -= cluster_Vx_doit
    # star_Vy -= cluster_Vy_doit
    
    # Load up the best fit ellipse paramters.
    params = Table.read(outdir+'/plots/pop_parameters.txt',format='ascii')

    pi_fit = np.zeros(N_gauss, dtype=float)
    vx_fit = np.zeros(N_gauss, dtype=float)
    vy_fit = np.zeros(N_gauss, dtype=float)
    sigA_fit = np.zeros(N_gauss, dtype=float)
    sigB_fit = np.zeros(N_gauss, dtype=float)
    theta_fit = np.zeros(N_gauss, dtype=float)
    
    for kk in range(N_gauss):
        # Extract the necessary parameters from the parameter file
        pname = params['Param']
        pi_idx = np.where(pname == ('pi_'+ str(kk)))[0]
        vx_idx = np.where(pname == ('vx_'+ str(kk)))[0]
        vy_idx = np.where(pname == ('vy_'+ str(kk)))[0]
        sigA_idx = np.where(pname == ('sigA_'+ str(kk)))[0]
        sigB_idx = np.where(pname == ('sigB_'+ str(kk)))[0]
        theta_idx = np.where(pname == ('theta_'+ str(kk)))[0]
        
        pi_fit[kk] = params['Mu'][pi_idx]
        vx_fit[kk] = params['Mu'][vx_idx]
        vy_fit[kk] = params['Mu'][vy_idx]
        sigA_fit[kk] = params['Mu'][sigA_idx]
        sigB_fit[kk] = params['Mu'][sigB_idx]
        theta_fit[kk] = np.degrees(params['Mu'][theta_idx]) - 90.0
        print theta_fit[kk]

    #---------------------------#
    # Plotting just VPD
    #---------------------------#

    # Plot full field VPD
    py.close('all')
    py.figure(1)
    py.clf()
    py.plot(star_Vx, star_Vy, 'k.', ms=4, alpha=0.5)
    py.xlabel(r'$\mu_{\alpha} \cos \delta$ (mas yr$^{-1}$)')
    py.ylabel(r'$\mu_{\delta}$ (mas yr$^{-1}$)')
    py.axis([6, -6, -6, 6])
    py.savefig(outdir + '/plots/VPD_field.png')

    #-----------------------------------#
    # Plot VPD with 1, 2, and 3 sigma contours for each population
    # overlaid, with cluster zoom.
    # Will also include Galactic plane (angle = -12.5 degrees)
    #-------------------------------------#
    # Function for rotation
    def line1(x, angle):
        # Angle is in degrees
        y = np.tan(np.radians(angle)) * x
        return y
    
    x_rot = np.arange(-5, 9, 1)
    y_rot = line1(x_rot, -12.5)

    py.figure(2, figsize = (12,6))
    py.clf()
    py.subplot(121)
    py.subplots_adjust(left=0.1)
    ax = py.gca()
    py.plot(star_Vx, star_Vy, 'k.', ms=4, alpha = 0.5)
    py.xlabel(r'$\mu_{\alpha}$cos$\delta$ (mas yr$^{-1}$)')
    py.ylabel(r'$\mu_{\delta}$ (mas yr$^{-1}$)')
    py.axis([6, -6, -6, 6])
    fig = py.gcf()

    # Plot the ellipses
    colors = ['r', 'b', 'g', 'c', 'y']
    for kk in range(N_gauss):
        scale1 = 2.0
        scale2 = 2.0 * np.sqrt(5.991)
        scale3 = 2.0 * np.sqrt(10.597)
        
        circ1 = Ellipse((vx_fit[kk], vy_fit[kk]),
                        sigA_fit[kk] * scale1, sigB_fit[kk] * scale1,
                        angle=theta_fit[kk], color=colors[kk], fill=False, linewidth=3)
        circ2 = Ellipse((vx_fit[kk], vy_fit[kk]),
                        sigA_fit[kk] * scale2, sigB_fit[kk] * scale2,
                        angle=theta_fit[kk], color=colors[kk], fill=False, linewidth=3)
        circ3 = Ellipse((vx_fit[kk], vy_fit[kk]),
                        sigA_fit[kk] * scale3, sigB_fit[kk] * scale3,
                        angle=theta_fit[kk], color=colors[kk], fill=False, linewidth=3)
        
        fig.gca().add_artist(circ1)
        fig.gca().add_artist(circ2)
        fig.gca().add_artist(circ3)
        
    py.subplot(122)
    ax2 = py.gca()
    py.axis([3,-3,-3, 3])
    fig = py.gcf()
    py.plot(star_Vx, star_Vy, 'k.', ms=4, alpha = 0.5)
    py.xlabel(r'$\mu_{\alpha}$cos$\delta$ (mas yr$^{-1}$)')
    py.ylabel(r'$\mu_{\delta}$ (mas yr$^{-1}$)')
    fig = py.gcf()

    # Plot the ellipses
    colors = ['r', 'b', 'g', 'c', 'y']
    for kk in range(N_gauss):
        scale1 = 2.0
        scale2 = 2.0 * np.sqrt(5.991)
        scale3 = 2.0 * np.sqrt(10.597)
        
        circ1 = Ellipse((vx_fit[kk], vy_fit[kk]),
                        sigA_fit[kk] * scale1, sigB_fit[kk] * scale1,
                        angle=theta_fit[kk], color=colors[kk], fill=False, linewidth=3)
        circ2 = Ellipse((vx_fit[kk], vy_fit[kk]),
                        sigA_fit[kk] * scale2, sigB_fit[kk] * scale2,
                        angle=theta_fit[kk], color=colors[kk], fill=False, linewidth=3)
        circ3 = Ellipse((vx_fit[kk], vy_fit[kk]),
                        sigA_fit[kk] * scale3, sigB_fit[kk] * scale3,
                        angle=theta_fit[kk], color=colors[kk], fill=False, linewidth=3)
        
        fig.gca().add_artist(circ1)
        fig.gca().add_artist(circ2)
        fig.gca().add_artist(circ3)
    
    outfile = outdir + '/plots/distributions_combo.png'
    py.savefig(outfile)

    return

def cluster_membership(catalogfile, velcut, magcut, outdir, N_gauss, prob, rotate=True):
    """
    Calculates cluster membership probabilities for each star given the population parameters 
    in parameter file (from plot_posteriors_1D). Assumes parameter file will be found 
    in <outdir>/plots/pop_parameters.txt

    Assumes catalogfile is untrimmed

    velcut: velocity error cut (mas/yr)
    magcut: phot error cut (mag)
    N_gauss: number of gaussians
    
    Identifies stars with a cluster membership probability GREATER THAN prob as high-prob members
    
    Produces new catalog cluster.fits which only contains cluster members. THIS HAS POSITIONS/PMs
    IN PIXELS. Membership probabilities are added in last column. Also makes new catalog
    *_prob_#.fits which is the original catalog with the cluster probabilities added on.
    This is the catalog we will work with extensively in the analysis

    Plots: Membership histogram, Spatial/velocity distributions of high-prob members

    Assumes gaussian 1 is the cluster gaussian

    If rotate = True, rotates velocities to RA/DEC to calculate membership probs.
    NOT COMPATABLE WITH MAKING VPD!!!
    """
    #Extract catalog data
    d = loadData(catalogfile, velcut, magcut, rotate=rotate)

    star_xpos_pix = d['x_2010_F160W']
    star_ypos_pix = d['y_2010_F160W']
    star_Vx = d['fit_vx']
    star_Vy = d['fit_vy']
    star_Sigx = d['fit_vxe']
    star_Sigy = d['fit_vye']

    # Transform velocities into reference frame of the cluster
    # Using velocity determined from UPDATED doit.py:
    # star_Vx -= cluster_Vx_doit
    # star_Vy -= cluster_Vy_doit

    # Extract the necessary parameters from the parameter file
    params = Table.read(outdir + '/plots/pop_parameters.txt', format='ascii')

    pscale = astrometry.scale['WFC'] * 1e3
    pi_fit = np.zeros(N_gauss, dtype=float)
    vx_fit = np.zeros(N_gauss, dtype=float)
    vy_fit = np.zeros(N_gauss, dtype=float)
    sigA_fit = np.zeros(N_gauss, dtype=float)
    sigB_fit = np.zeros(N_gauss, dtype=float)
    theta_fit = np.zeros(N_gauss, dtype=float)

    for kk in range(N_gauss):
        # Extract the necessary parameters from the parameter file
        pname = params['Param']
        pi_idx = np.where(pname == ('pi_'+ str(kk)))[0]
        vx_idx = np.where(pname == ('vx_'+ str(kk)))[0]
        vy_idx = np.where(pname == ('vy_'+ str(kk)))[0]
        sigA_idx = np.where(pname == ('sigA_'+ str(kk)))[0]
        sigB_idx = np.where(pname == ('sigB_'+ str(kk)))[0]
        theta_idx = np.where(pname == ('theta_'+ str(kk)))[0]
        
        pi_fit[kk] = params['Mu'][pi_idx][0]
        vx_fit[kk] = params['Mu'][vx_idx][0]
        vy_fit[kk] = params['Mu'][vy_idx][0]
        sigA_fit[kk] = params['Mu'][sigA_idx][0]
        sigB_fit[kk] = params['Mu'][sigB_idx][0]
        theta_fit[kk] = params['Mu'][theta_idx][0]

    # Calculate cluster membership probability assuming
    # 1st gaussian is the cluster.
    prob_all = np.zeros((N_gauss, len(d)), dtype=float)

    for kk in range(N_gauss):
        prob_all[kk, :] = prob_ellipse(star_Vx, star_Vy, star_Sigx, star_Sigy,
                                       pi_fit[kk], vx_fit[kk], vy_fit[kk],
                                       sigA_fit[kk], sigB_fit[kk], theta_fit[kk])

    # Calculate cluster membership probability
    p_cluster = prob_all[0, :] / prob_all.sum(axis=0)

    #Plot distribution of cluster membership probabilities
    xbins = np.arange(0,1.01,0.05)

    py.close('all')
    fig, ax = py.subplots(num=1)
    n, bins, patch = py.hist(p_cluster, bins=xbins)
    py.axis([0, 1, 0, 4000])
    py.xlabel('Cluster Membership Probability')
    py.ylabel(r'N$_{stars}$')
    axins = inset_axes(ax, width = "75%", height = 3, loc = 1) 
    py.hist(p_cluster, bins=xbins)
    py.axvline(prob, color='red', linestyle='--', linewidth=2)
    axins.set_xlim(0.1, 1.0)
    axins.set_ylim(0, 800)
    mark_inset(ax, axins, loc1=2, loc2=4, fc="none", ec="0.5")
    outfile = outdir + '/plots/membership_hist_rot.png'
    py.savefig(outfile)

    # Collect the stars where P(membership > prob)
    # Highlight these stars in VPD
    members_ind = np.array(np.where(bins >= prob)[0])
    members_ind = members_ind[:-1]
    members = n[members_ind].sum()
    print 'Number of cluster members: ' + str(members)

    # Plot spatial and VPD positions of ID'ed cluster members
    memberID = np.where(p_cluster >= prob)[0]
    
    # Change positions into arcsecs from cluster center for plotting purposes
    cl_x = star_xpos_pix[memberID].mean()
    cl_y = star_xpos_pix[memberID].mean()
    star_xpos_plot = (star_xpos_pix - cl_x) * pscale / 1e3
    star_ypos_plot = (star_ypos_pix - cl_y) * pscale / 1e3

    #--Now for spatial/velocity positions of high prob members---#
    outfile = outdir + '/plots/members_' + str(prob) + '_rot.png'
    py.close(2)
    py.figure(2, figsize =(12,6))
    py.clf()
    py.subplot(121)
    py.subplots_adjust(left=0.1)
    ax = py.gca()
    py.plot(star_xpos_plot, star_ypos_plot, 'k.', ms = 6, alpha = 0.2)     
    py.plot(star_xpos_plot[memberID], star_ypos_plot[memberID], 'r.', ms=8, alpha = 0.5)
    py.axis([125, -125, -125, 125])
    py.xlabel('X Position Offset (arcsec)')
    py.ylabel('Y Position Offset (arcsec)')
    
    py.subplot(122)

    py.plot(star_Vx, star_Vy, 'k.', ms = 4, alpha = 0.5)
    py.axis([6,-6,-6, 6])
    py.xlabel(r'$\mu_{\alpha}$cos$\delta$ (mas yr$^{-1}$)')
    py.ylabel(r'$\mu_{\delta}$ (mas yr$^{-1}$)')
    py.plot(star_Vx[memberID], star_Vy[memberID], 'r.', ms = 4, alpha = 0.3)
    #
    # Test orientation
    #
    # tmp1 = prob_all[2, :] / prob_all[2, :].sum()
    # sdx = tmp1.argsort()
    # tmp = tmp1[sdx].cumsum()
    # sig_lev_1 = tmp1[sdx[np.where(tmp > 0.01)[0][0]]]
    # sig_lev_2 = tmp1[sdx[np.where(tmp > 0.10)[0][0]]]
    # sig_lev_3 = tmp1[sdx[np.where(tmp > 0.50)[0][0]]]
    # f3 = np.where(tmp1 > sig_lev_1)[0]
    # f2 = np.where(tmp1 > sig_lev_2)[0]
    # f1 = np.where(tmp1 > sig_lev_3)[0]
    # py.plot(star_Vx[f3], star_Vy[f3], 'g.', ms=4, alpha=0.2)
    # py.plot(star_Vx[f2], star_Vy[f2], 'b.', ms=4, alpha=0.2)
    # py.plot(star_Vx[f1], star_Vy[f1], 'c.', ms=4, alpha=0.2)
    
    outfile = outdir + '/plots/membership_VPD_'+str(prob)+'.png'
    py.savefig(outfile)
    
    # Add column for membership probability in original cluster table
    d['Membership'] = p_cluster

    # HACK for BRITE cluster members.
    mag = d['m_2005_F814W']
    color = d['m_2005_F814W'] - d['m_2013_F160W']
    idx = np.where((mag < 18.8) & ((color > 3.1) & (color < 4.8)))[0]
    d['Membership'][idx] = 1.0

    # Finally, make a new catalog with only cluster members
    outfile = '{0}/catalog_membership_{1}_rot.fits'.format(outdir, N_gauss)
    print 'Writing: ', outfile
    d.write(outfile, format='fits', overwrite=True)

    outfile = '{0}/catalog_cluster_only_{1}_{2:.1f}_rot.fits'.format(outdir, N_gauss, prob)
    print 'Writing: ', outfile
    d_clust = d[memberID]
    d_clust.write(outfile, format='fits', overwrite=True)

    return 


def vel_rotate(x, y, angle):
    cosa = np.cos(angle)
    sina = np.sin(angle)
    
    x_new = (x * cosa) + (y * sina)
    y_new = (-1.0 * x * sina) + (y * cosa)
        
    return x_new, y_new

def velerr_rotate(xe, ye, angle):
    cosa = np.cos(angle)
    sina = np.sin(angle)
    
    x_new = np.hypot(xe * cosa, ye * sina)
    y_new = np.hypot(xe * sina, ye * cosa)
            
    return x_new, y_new


def prob_ellipse(star_vx, star_vy, star_sigx, star_sigy,
                 pi, vx, vy, sigA, sigB, theta):
    xdiff = star_vx - vx
    ydiff = star_vy - vy
        
    cos_a = math.cos(theta)
    sin_a = math.sin(theta)
    R = np.array([[cos_a, -sin_a],
                  [sin_a,  cos_a]])
    Z_AB = np.array([[sigA**2,         0.0],
                     [         0.0, sigB**2]])

    Z_xy = R.dot( Z_AB.dot( R.T ) )

    ##############################
    # Switch out of vector notation here since we have N_stars
    # to deal with.
    ##############################

    # Create the covariance matrix variables -- one for each
    # position in the covar matrix. Each will be N_stars long.
    # This is the combined Z_xy + S_xy where
    # S_xy = covariance matrix from stellar velocity errors
    cov_0_0 = Z_xy[0, 0] + star_sigx**2
    cov_0_1 = Z_xy[0, 1]
    cov_1_0 = Z_xy[1, 0]
    cov_1_1 = Z_xy[1, 1] + star_sigy**2

    # N_stars long arrays
    det_cov = (cov_0_0 * cov_1_1) - (cov_0_1 * cov_1_0)

    exp_term  = xdiff**2 * cov_0_0
    exp_term += ydiff**2 * cov_1_1
    exp_term += ydiff * xdiff * (cov_0_1 + cov_1_0)
    exp_term /= -2.0 * det_cov

    # N_stars long array
    prob_gauss = pi * np.exp(exp_term)
    prob_gauss /= 2.0 * np.pi * np.sqrt(det_cov)

    return prob_gauss

