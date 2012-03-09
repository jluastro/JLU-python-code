import pylab as py
import numpy as np
from mpl_toolkits import mplot3d
import pymc
import bayesian as b
from scipy import interpolate
from jlu.papers import lu_gc_imf
import scipy
import scipy.stats
import pymultinest
import math
import atpy
from gcreduce import gcutil
import pickle
import pdb
from gcwork import objects
from jlu.util import rebin

def random_distance(randNum):
    dist_mean = 8.096  # kpc
    dist_std = 0.483   # kpc
    dist_min = 6.793   # kpc
    dist_max = 9.510   # kpc
    dist_a = (dist_min - dist_mean) / dist_std
    dist_b = (dist_max - dist_mean) / dist_std
    dist = scipy.stats.truncnorm.ppf(randNum, dist_a, dist_b,
                                     loc=dist_mean, scale=dist_std)
    prob_dist = scipy.stats.truncnorm.pdf(dist, dist_a, dist_b,
                                          loc=dist_mean, scale=dist_std)
    return dist, prob_dist

def random_log_age(randNum):
    log_age_mean = 6.78
    log_age_std = 0.18
    log_age_min = 6.20
    log_age_max = 7.20
    log_age_a = (log_age_min - log_age_mean) / log_age_std
    log_age_b = (log_age_max - log_age_mean) / log_age_std
    log_age_cont = scipy.stats.truncnorm.ppf(randNum, log_age_a, log_age_b,
                                             loc=log_age_mean, scale=log_age_std)
    prob_log_age_cont = scipy.stats.truncnorm.pdf(log_age_cont, log_age_a, log_age_b,
                                                  loc=log_age_mean, scale=log_age_std)

    return log_age_cont, prob_log_age_cont

def random_alpha(randNum):
    alpha_min = 0.10
    alpha_max = 3.35
    alpha_diff = alpha_max - alpha_min
    alpha = scipy.stats.uniform.ppf(randNum, loc=alpha_min, scale=alpha_diff)
    prob_alpha = scipy.stats.uniform.pdf(alpha, loc=alpha_min, scale=alpha_diff)

    return alpha, prob_alpha

def random_mass(randNum):
    #Mcl_min = 1
    #Mcl_max = 100
    Mcl_min = 7
    Mcl_max = 13
    Mcl_diff = Mcl_max - Mcl_min
    Mcl = scipy.stats.uniform.ppf(randNum, loc=Mcl_min, scale=Mcl_diff)
    prob_Mcl = scipy.stats.uniform.pdf(Mcl, loc=Mcl_min, scale=Mcl_diff)

    return Mcl, prob_Mcl

def random_N_old(randNum):
    N_old_min = 0.2e3
    N_old_max = 3.0e3
    N_old_diff = N_old_max - N_old_min
    N_old = scipy.stats.uniform.ppf(randNum, loc=N_old_min, scale=N_old_diff)
    prob_N_old = scipy.stats.uniform.pdf(N_old, loc=N_old_min, scale=N_old_diff)

    return N_old, prob_N_old

def random_gamma(randNum):
    # Values from Schodel+ 2010
    gamma_mean = 0.27
    gamma_std = 0.02
    gamma = scipy.stats.norm.ppf(randNum, loc=gamma_mean, scale=gamma_std)
    prob_gamma = scipy.stats.norm.pdf(gamma, loc=gamma_mean, scale=gamma_std)
    
    return gamma, prob_gamma

def random_rcMean(randNum):
    # Values from Schodel+ 2010
    rcMean_mean = 15.57 + 0.03  # correction for Ks -> Kp (cool stars)
    rcMean_std = 0.06
    rcMean = scipy.stats.norm.ppf(randNum, loc=rcMean_mean, scale=rcMean_std)
    prob_rcMean = scipy.stats.norm.pdf(rcMean, loc=rcMean_mean, scale=rcMean_std)

    return rcMean, prob_rcMean

def random_rcSigma(randNum):
    # Values from Schodel+ 2010
    rcSigma_mean = 0.3  # correction for Ks -> Kp (cool stars)
    rcSigma_std = 0.2
    rcSigma_min = 0.1
    rcSigma_max = 0.8
    rcSigma_a = (rcSigma_min - rcSigma_mean) / rcSigma_std
    rcSigma_b = (rcSigma_max - rcSigma_mean) / rcSigma_std
    rcSigma = scipy.stats.truncnorm.ppf(randNum, rcSigma_a, rcSigma_b,
                                        loc=rcSigma_mean, scale=rcSigma_std)
    prob_rcSigma = scipy.stats.truncnorm.pdf(rcSigma, rcSigma_a, rcSigma_b,
                                             loc=rcSigma_mean, scale=rcSigma_std)

    return rcSigma, prob_rcSigma

def run(outdir, yng=None, rmin=0, rmax=30, n_live_points=4000, multiples=True):
    """
    Run MultiNest bayesian inference on the specified data <yng> and send output
    to the <outdir>.
    """
    magCut = 15.5

    # Load up the data arrays.
    if yng == None:
        yng = lu_gc_imf.load_yng_data_by_radius(rmin, rmax, magCut=magCut)
        yng_file = 'Observed Data'
    elif type(yng) == str:
        yng_file = yng
        foo = open(yng_file, 'r')
        yng = pickle.load(foo)
        foo.close()
    else:
        yng_file = 'Simulated Cluster Object'
        
    completeness = lu_gc_imf.load_image_completeness_by_radius(rmin, rmax)

    def priors(cube, ndim, nparams):
        return

    def likelihood(cube, ndim, nparams):
        # Set up variables (with priors)
        AKs = 2.7   # mag
        m_min = 1.0 # Msun
        m_max = 150 # Msun
        Z = 0.02

        # Distance to cluster
        dist, prob_dist = random_distance(cube[0])
        cube[0] = dist

        # Log Age of the cluster
        log_age_cont, prob_log_age_cont = random_log_age(cube[1])
        log_age = round(log_age_cont, 2)
        cube[1] = log_age_cont

        # Slope of the IMF
        alpha, prob_alpha = random_alpha(cube[2])
        cube[2] = alpha

        # Total Cluster Mass
        Mcl, prob_Mcl = random_mass(cube[3])
        cube[3] = Mcl

        # Number of young stars we observed. cube[4] contains total N(OB)
        N_yng_obs_definite = yng.prob[yng.prob == 1].sum()
        N_yng_obs_other_expect = yng.prob[yng.prob != 1].sum()
        N_yng_obs_other = scipy.stats.poisson.ppf(cube[4], N_yng_obs_other_expect)
        N_yng_obs = N_yng_obs_definite + N_yng_obs_other
        prob_N_yng_obs = scipy.stats.poisson.pmf(N_yng_obs_other, N_yng_obs_other_expect)
        cube[4] = N_yng_obs

        # Check that all our prior probabilities are valid, otherwise abort
        # before expensive calculation.
        if ((prob_alpha == 0) or (prob_log_age_cont == 0) or (prob_dist == 0) or
            (prob_Mcl == 0) or (prob_N_yng_obs == 0)):
            return -np.Inf

        # Set some variables so we don't have to change names around
        logAge = log_age
        distance = dist
        imfSlope = alpha
        clusterMass = Mcl
        minMass = m_min
        maxMass = m_max

        # Get the PDF(k|model) -- the simulated luminosity function
        mod_sims = b.fetch_model_from_sims(logAge, AKs, distance*10**3,
                                           imfSlope, clusterMass*10**3,
                                           minMass, maxMass, makeMultiples=multiples)

        sim_N_WR = mod_sims[0]
        sim_k_bins = mod_sims[1]
        sim_k_pdf = mod_sims[2]
        sim_k_pdf_norm = mod_sims[3]

        sim_k_bin_widths = np.diff(sim_k_bins)
        sim_k_bin_center = sim_k_bins[:-1] + (sim_k_bin_widths/2.0)

        # Multiply by the completeness curve (after re-sampling). And renormalize.
        Kp_interp = interpolate.splrep(completeness.mag, completeness.comp, k=1, s=0)
        comp_resamp = interpolate.splev(sim_k_bin_center, Kp_interp)
        comp_resamp[comp_resamp < 0] = 0.0
        comp_resamp[comp_resamp > 1] = 1.0

        sim_k_pdf *= comp_resamp
        sim_k_pdf_norm *= comp_resamp
        #sim_k_pdf_norm /= (sim_k_pdf_norm * sim_k_bin_widths).sum()
        sim_k_pdf_norm /= sim_k_pdf_norm.sum()

        # Trim down to a little more than magCut just to speed things
        # up and make computations easier.
        idx = np.where(sim_k_bins <= (magCut + (3.0 * yng.kp_err.max())))[0]
        #idx = np.where(sim_k_bins <= magCut)[0]

        sim_k_bins = np.append(sim_k_bins[idx], sim_k_bins[idx[-1]+1])
        sim_k_bin_center = sim_k_bin_center[idx]
        sim_k_bin_widths = sim_k_bin_widths[idx]
        sim_k_pdf = sim_k_pdf[idx]
        sim_k_pdf_norm = sim_k_pdf_norm[idx]

        #sim_k_pdf_norm /= (sim_k_pdf_norm * sim_k_bin_widths).sum()
        sim_k_pdf_norm /= sim_k_pdf_norm.sum()

        # Prob(N_WR | model)
        cube[5] = sim_N_WR
        log_L_N_WR = log_prob( scipy.stats.poisson.pmf(yng.N_WR, sim_N_WR) )

        # Prob(N_kp | model)
        idx = np.where(sim_k_bins <= magCut)[0]
        N_yng_sim = sim_k_pdf[idx[:-1]].sum()
        bb = idx[-1] # Take only a part of the last bin (depending on where the magCut falls)
        N_yng_sim += sim_k_pdf[bb] * (magCut - sim_k_bins[bb]) / sim_k_bin_widths[bb]
        cube[6] = N_yng_sim

        log_L_N_yng = log_prob( scipy.stats.poisson.pmf(N_yng_obs, N_yng_sim) )

        ##############################
        #
        # KLF shape is handled directly with the normalized Kp PDF
        # from the simulations. We account for errors by convolving
        # each measurement gaussian with the PDF(Kp).
        #
        ##############################
        log_L_k_detect = 0.0

        # Loop through each star and calc prob of detecting.
        for ii in range(len(yng.kp)):
            # Make gaussian around observed k mag
            obs_k_norm = scipy.stats.norm(loc=yng.kp[ii], scale=yng.kp_err[ii])
            obs_k_norm_cdf = obs_k_norm.cdf(sim_k_bins)
            obs_k_norm_pdf_binned = np.diff(obs_k_norm_cdf)

            # Renormalize over observing range
            # and convolve gaussian with PDF(K) from model
            if obs_k_norm_pdf_binned.sum() == 0:
                L_k_i = 0
            else:
                # Make it a true PDF
                #obs_k_norm_pdf_binned /= (obs_k_norm_pdf_binned * sim_k_bin_widths).sum()
                #obs_k_norm_pdf_binned /= obs_k_norm_pdf_binned.sum()

                # Multiple gaussian with PDF(K) from model and sum to get probability
                #L_k_i = (sim_k_pdf_norm * obs_k_norm_pdf_binned * sim_k_bin_widths**2).sum()
                L_k_i = (sim_k_pdf_norm * obs_k_norm_pdf_binned).sum()

            log_L_k_i = log_prob(L_k_i)

            log_L_k_detect += yng.prob[ii] * log_L_k_i

        #log_L = log_L_N_yng + log_L_k_detect + (yng.N_WR * log_L_N_WR)
        log_L = log_L_N_yng + log_L_k_detect + log_L_N_WR

        # Add in the log(Prior_Probabilities) as well
        log_L += math.log(prob_dist)
        log_L += math.log(prob_alpha)
        log_L += math.log(prob_Mcl)
        log_L += math.log(prob_log_age_cont)
        log_L += math.log(prob_N_yng_obs)

        return log_L


    gcutil.mkdir(outdir)

    outroot = outdir + 'mnest_'

    num_params = 7
    num_dims = 5
    ev_tol = 0.5
    samp_eff = 0.5
    n_clust_param = num_dims - 1

    _run = open(outroot + 'params.run', 'w')
    _run.write('Num Dimensions: %d\n' % num_dims)
    _run.write('Num Params: %d\n' % num_params)
    _run.write('Evidence Tolerance: %.1f\n' % ev_tol)
    _run.write('Sampling Efficiency: %.1f\n' % samp_eff)
    _run.write('Num Clustering Params: %d\n' % n_clust_param)
    _run.write('Num Live Points: %d\n' % n_live_points)
    _run.write('Young Star Data: %s\n' % yng_file)
    _run.write('Allowed Multiples in Fit: %s\n' % str(multiples))
    _run.close()

    pymultinest.run(likelihood, priors, num_dims, n_params=num_params,
                    outputfiles_basename=outroot,
                    verbose=True, resume=False,
                    evidence_tolerance=ev_tol, sampling_efficiency=samp_eff,
                    n_clustering_params=n_clust_param,
                    n_live_points=n_live_points)


def run_no_dist(outdir, yng=None, rmin=0, rmax=30, n_live_points=4000, multiples=True):
    """
    Run MultiNest bayesian inference on the specified data <yng> and send output
    to the <outdir>.
    """
    magCut = 15.5

    # Load up the data arrays.
    if yng == None:
        yng = lu_gc_imf.load_yng_data_by_radius(rmin, rmax, magCut=magCut)
        yng_file = 'Observed Data'
    elif type(yng) == str:
        yng_file = yng
        foo = open(yng_file, 'r')
        yng = pickle.load(foo)
        foo.close()
    else:
        yng_file = 'Simulated Cluster Object'
        
    completeness = lu_gc_imf.load_image_completeness_by_radius(rmin, rmax)

    def priors(cube, ndim, nparams):
        return

    def likelihood(cube, ndim, nparams):
        # Set up variables (with priors)
        AKs = 2.7   # mag
        m_min = 1.0 # Msun
        m_max = 150 # Msun
        Z = 0.02

        # Distance to cluster
        dist = 8.00 # kpc
        # dist_mean = 8.096  # kpc
        # dist_std = 0.483   # kpc
        # dist_min = 6.793   # kpc
        # dist_max = 9.510   # kpc
        # dist_a = (dist_min - dist_mean) / dist_std
        # dist_b = (dist_max - dist_mean) / dist_std
        # dist = scipy.stats.truncnorm.ppf(cube[0], dist_a, dist_b,
        #                                  loc=dist_mean, scale=dist_std)
        # prob_dist = scipy.stats.truncnorm.pdf(dist, dist_a, dist_b,
        #                                       loc=dist_mean, scale=dist_std)
        prob_dist = 1.0
        cube[0] = dist

        # Log Age of the cluster
        log_age_mean = 6.78
        log_age_std = 0.18
        log_age_min = 6.20
        log_age_max = 7.20
        log_age_a = (log_age_min - log_age_mean) / log_age_std
        log_age_b = (log_age_max - log_age_mean) / log_age_std
        log_age_cont = scipy.stats.truncnorm.ppf(cube[1], log_age_a, log_age_b,
                                                 loc=log_age_mean, scale=log_age_std)
        prob_log_age_cont = scipy.stats.truncnorm.pdf(log_age_cont, log_age_a, log_age_b,
                                                      loc=log_age_mean, scale=log_age_std)
        log_age = round(log_age_cont, 2)
        cube[1] = log_age_cont

        # Slope of the IMF
        alpha_min = 0.10
        alpha_max = 3.35
        alpha_diff = alpha_max - alpha_min
        alpha = scipy.stats.uniform.ppf(cube[2], loc=alpha_min, scale=alpha_diff)
        prob_alpha = scipy.stats.uniform.pdf(alpha, loc=alpha_min, scale=alpha_diff)
        cube[2] = alpha

        # Total Cluster Mass
        Mcl_min = 1
        Mcl_max = 100
        Mcl_diff = Mcl_max - Mcl_min
        Mcl = scipy.stats.uniform.ppf(cube[3], loc=Mcl_min, scale=Mcl_diff)
        prob_Mcl = scipy.stats.uniform.pdf(Mcl, loc=Mcl_min, scale=Mcl_diff)
        cube[3] = Mcl

        # Number of young stars we observed. cube[4] contains total N(OB)
        N_yng_obs_definite = yng.prob[yng.prob == 1].sum()
        N_yng_obs_other_expect = yng.prob[yng.prob != 1].sum()
        N_yng_obs_other = scipy.stats.poisson.ppf(cube[4], N_yng_obs_other_expect)
        N_yng_obs = N_yng_obs_definite + N_yng_obs_other
        prob_N_yng_obs = scipy.stats.poisson.pmf(N_yng_obs_other, N_yng_obs_other_expect)
        cube[4] = N_yng_obs

        # Check that all our prior probabilities are valid, otherwise abort
        # before expensive calculation.
        if ((prob_alpha == 0) or (prob_log_age_cont == 0) or (prob_dist == 0) or
            (prob_Mcl == 0) or (prob_N_yng_obs == 0)):
            return -np.Inf

        # Set some variables so we don't have to change names around
        logAge = log_age
        distance = dist
        imfSlope = alpha
        clusterMass = Mcl
        minMass = m_min
        maxMass = m_max

        # Get the PDF(k|model) -- the simulated luminosity function
        mod_sims = b.fetch_model_from_sims(logAge, AKs, distance*10**3,
                                           imfSlope, clusterMass*10**3,
                                           minMass, maxMass, makeMultiples=multiples)

        sim_N_WR = mod_sims[0]
        sim_k_bins = mod_sims[1]
        sim_k_pdf = mod_sims[2]
        sim_k_pdf_norm = mod_sims[3]

        sim_k_bin_widths = np.diff(sim_k_bins)
        sim_k_bin_center = sim_k_bins[:-1] + (sim_k_bin_widths/2.0)

        # Multiply by the completeness curve (after re-sampling). And renormalize.
        Kp_interp = interpolate.splrep(completeness.mag, completeness.comp, k=1, s=0)
        comp_resamp = interpolate.splev(sim_k_bin_center, Kp_interp)
        comp_resamp[comp_resamp < 0] = 0.0
        comp_resamp[comp_resamp > 1] = 1.0

        sim_k_pdf *= comp_resamp
        sim_k_pdf_norm *= comp_resamp
        sim_k_pdf_norm /= (sim_k_pdf_norm * sim_k_bin_widths).sum()

        # Trim down to a little more than magCut just to speed things
        # up and make computations easier.
        idx = np.where(sim_k_bins <= (magCut + (3.0 * yng.kp_err.max())))[0]

        sim_k_bins = np.append(sim_k_bins[idx], sim_k_bins[idx[-1]+1])
        sim_k_bin_center = sim_k_bin_center[idx]
        sim_k_bin_widths = sim_k_bin_widths[idx]
        sim_k_pdf = sim_k_pdf[idx]
        sim_k_pdf_norm = sim_k_pdf_norm[idx]

        sim_k_pdf_norm /= (sim_k_pdf_norm * sim_k_bin_widths).sum()

        # Prob(N_WR | model)
        cube[5] = sim_N_WR
        log_L_N_WR = log_prob( scipy.stats.poisson.pmf(yng.N_WR, sim_N_WR) )

        # Prob(N_kp | model)
        idx = np.where(sim_k_bins <= magCut)[0]
        N_yng_sim = sim_k_pdf[idx[:-1]].sum()
        bb = idx[-1] # Take only a part of the last bin (depending on where the magCut falls)
        N_yng_sim += sim_k_pdf[bb] * (magCut - sim_k_bins[bb]) / sim_k_bin_widths[bb]
        cube[6] = N_yng_sim

        log_L_N_yng = log_prob( scipy.stats.poisson.pmf(N_yng_obs, N_yng_sim) )

        ##############################
        #
        # KLF shape is handled directly with the normalized Kp PDF
        # from the simulations. We account for errors by convolving
        # each measurement gaussian with the PDF(Kp).
        #
        ##############################
        log_L_k_detect = 0.0

        # Loop through each star and calc prob of detecting.
        for ii in range(len(yng.kp)):
            # Make gaussian around observed k mag
            obs_k_norm = scipy.stats.norm(loc=yng.kp[ii], scale=yng.kp_err[ii])
            obs_k_norm_cdf = obs_k_norm.cdf(sim_k_bins)
            obs_k_norm_pdf_binned = np.diff(obs_k_norm_cdf)

            # Renormalize over observing range
            # and convolve gaussian with PDF(K) from model
            if obs_k_norm_pdf_binned.sum() == 0:
                L_k_i = 0
            else:
                # Make it a true PDF
                obs_k_norm_pdf_binned /= (obs_k_norm_pdf_binned * sim_k_bin_widths).sum()

                # Multiple gaussian with PDF(K) from model and sum to get probability
                L_k_i = (sim_k_pdf_norm * obs_k_norm_pdf_binned * sim_k_bin_widths**2).sum()

            log_L_k_i = log_prob(L_k_i)

            log_L_yng_i = log_prob(yng.prob[ii])

            log_L_k_detect += yng.prob[ii] * log_L_k_i

        #log_L = log_L_N_yng + log_L_k_detect + (yng.N_WR * log_L_N_WR)
        log_L = log_L_N_yng + log_L_k_detect + log_L_N_WR

        # Add in the log(Prior_Probabilities) as well
        log_L += math.log(prob_dist)
        log_L += math.log(prob_alpha)
        log_L += math.log(prob_Mcl)
        log_L += math.log(prob_log_age_cont)
        log_L += math.log(prob_N_yng_obs)

        return log_L


    gcutil.mkdir(outdir)

    outroot = outdir + 'mnest_'

    num_params = 7
    num_dims = 5
    ev_tol = 0.5
    samp_eff = 0.5
    n_clust_param = num_dims - 1

    _run = open(outroot + 'params.run', 'w')
    _run.write('Num Dimensions: %d\n' % num_dims)
    _run.write('Num Params: %d\n' % num_params)
    _run.write('Evidence Tolerance: %.1f\n' % ev_tol)
    _run.write('Sampling Efficiency: %.1f\n' % samp_eff)
    _run.write('Num Clustering Params: %d\n' % n_clust_param)
    _run.write('Num Live Points: %d\n' % n_live_points)
    _run.write('Young Star Data: %s\n' % yng_file)
    _run.write('Allowed Multiples in Fit: %s\n' % str(multiples))
    _run.close()

    pymultinest.run(likelihood, priors, num_dims, n_params=num_params,
                    outputfiles_basename=outroot,
                    verbose=True, resume=False,
                    evidence_tolerance=ev_tol, sampling_efficiency=samp_eff,
                    n_clustering_params=n_clust_param,
                    n_live_points=n_live_points)


def run2(outdir, yng=None, rmin=0, rmax=30, n_live_points=300, multiples=True):
    magCut = 15.5

    # Load up the data arrays.
    if yng == None:
        yng = lu_gc_imf.load_yng_data_by_radius(rmin, rmax, magCut=magCut)
        yng_file = 'Observed Data'
    elif type(yng) == str:
        yng_file = yng
        foo = open(yng_file, 'r')
        yng = pickle.load(foo)
        foo.close()
    else:
        yng_file = 'Simulated Cluster Object'
        
    completeness = lu_gc_imf.load_image_completeness_by_radius(rmin, rmax)

    def priors(cube, ndim, nparams):
        return

    def likelihood(cube, ndim, nparams):
        # Set up variables (with priors)
        AKs = 2.7   # mag
        m_min = 1.0 # Msun
        m_max = 150 # Msun
        Z = 0.02

        # Distance to cluster
        dist, prob_dist = random_distance(cube[0])
        cube[0] = dist

        # Log Age of the cluster
        log_age_cont, prob_log_age_cont = random_log_age(cube[1])
        log_age = round(log_age_cont, 2)
        cube[1] = log_age_cont

        # Slope of the IMF
        alpha, prob_alpha = random_alpha(cube[2])
        cube[2] = alpha

        # Total Cluster Mass
        Mcl, prob_Mcl = random_mass(cube[3])
        cube[3] = Mcl

        # Number of young stars we observed. cube[4] contains total N(OB)
        N_yng_obs_definite = yng.prob[yng.prob == 1].sum()
        N_yng_obs_other_expect = yng.prob[yng.prob != 1].sum()
        N_yng_obs_other = scipy.stats.poisson.ppf(cube[4], N_yng_obs_other_expect)
        N_yng_obs = N_yng_obs_definite + N_yng_obs_other
        prob_N_yng_obs = scipy.stats.poisson.pmf(N_yng_obs_other, N_yng_obs_other_expect)
        cube[4] = N_yng_obs

        # Check that all our prior probabilities are valid, otherwise abort
        # before expensive calculation.
        if ((prob_alpha == 0) or (prob_log_age_cont == 0) or (prob_dist == 0) or
            (prob_Mcl == 0) or (prob_N_yng_obs == 0)):
            return -np.Inf

        # Set some variables so we don't have to change names around
        logAge = log_age
        distance = dist
        imfSlope = alpha
        clusterMass = Mcl
        minMass = m_min
        maxMass = m_max

        # Get the PDF(k|model) -- the simulated luminosity function
        mod_sims = b.fetch_model_from_sims(logAge, AKs, distance*10**3,
                                           imfSlope, clusterMass*10**3,
                                           minMass, maxMass, makeMultiples=multiples)

        sim_N_WR = mod_sims[0]
        sim_k_bins = mod_sims[1]
        sim_k_pdf = mod_sims[2]
        sim_k_pdf_norm = mod_sims[3]

        sim_k_bin_widths = np.diff(sim_k_bins)
        sim_k_bin_center = sim_k_bins[:-1] + (sim_k_bin_widths/2.0)

        # Trim down to magCut
        idx = np.where(sim_k_bins <= (magCut + (3.0*yng.kp_err.max())))[0]

        sim_k_bins = np.append(sim_k_bins[idx], sim_k_bins[idx[-1]+1])
        sim_k_bin_center = sim_k_bin_center[idx]
        sim_k_bin_widths = sim_k_bin_widths[idx]
        sim_k_pdf = sim_k_pdf[idx]
        sim_k_pdf_norm = sim_k_pdf_norm[idx]

        sim_k_pdf_norm /= (sim_k_pdf_norm * sim_k_bin_widths).sum()

        # Completeness curve (after re-sampling at simulated Kp)
        Kp_interp = interpolate.splrep(completeness.mag, completeness.comp, k=1, s=0)
        comp_at_kp_sim = interpolate.splev(sim_k_bin_center, Kp_interp)
        comp_at_kp_sim[comp_at_kp_sim < 0] = 0.0
        comp_at_kp_sim[comp_at_kp_sim > 1] = 1.0

        # Completeness curve (after re-sampling at simulated Kp)
        comp_at_kp_obs = interpolate.splev(yng.kp, Kp_interp)
        comp_at_kp_obs[comp_at_kp_obs < 0] = 0.0
        comp_at_kp_obs[comp_at_kp_obs > 1] = 1.0

        # Prob(N_WR | model)
        N_WR_sim = sim_N_WR
        cube[5] = N_WR_sim
        log_L_N_WR = log_prob( scipy.stats.poisson.pmf(yng.N_WR, sim_N_WR) )

        # Prob(N_yng_sim | model)
        idx = np.where(sim_k_bins <= magCut)[0]
        N_yng_sim_expect = sim_k_pdf[idx[:-1]].sum()
        bb = idx[-1] # Take only a part of the last bin (depending on where the magCut falls)
        N_yng_sim_expect += sim_k_pdf[bb] * (magCut - sim_k_bins[bb]) / sim_k_bin_widths[bb]
        cube[6] = N_yng_sim_expect

        N_yng_sim = scipy.stats.poisson.rvs(N_yng_sim_expect)
        prob_N_yng_sim = scipy.stats.poisson.pmf(N_yng_sim, N_yng_sim_expect)
        log_L_N_yng_sim = log_prob( prob_N_yng_sim )

        # Non detections: log_L_k_non_detect
        if N_yng_sim <= N_yng_obs:
            # Jump straight out for performance reasons -- model is not possible
            return -np.Inf
        else:
            tmp = (1.0 - comp_at_kp_sim) * sim_k_pdf_norm * sim_k_bin_widths
            P_non_detect = tmp.sum()

            log_L_k_non_detect = (N_yng_sim - N_yng_obs) * log_prob(P_non_detect)

        # Detections: log_L_k_detect
        log_L_k_detect = 0.0

        # Loop through each star and calc prob of detecting.
        for ii in range(len(yng.kp)):
            # Make gaussian around observed k mag
            obs_k_norm = scipy.stats.norm(loc=yng.kp[ii], scale=yng.kp_err[ii])
            obs_k_norm_cdf = obs_k_norm.cdf(sim_k_bins)
            obs_k_norm_pdf_binned = np.diff(obs_k_norm_cdf)
            
            # Renormalize over observing range
            # and convolve gaussian with PDF(K) from model
            if obs_k_norm_pdf_binned.sum() == 0:
                L_k_i = 0
            else:
                # Make it a true PDF
                #obs_k_norm_pdf_binned /= (obs_k_norm_pdf_binned * sim_k_bin_widths).sum()
                #obs_k_norm_pdf_binned /= obs_k_norm_pdf_binned.sum()

                # Multiple gaussian with PDF(K) from model and sum to get probability
                #L_k_i = (sim_k_pdf_norm * obs_k_norm_pdf_binned * sim_k_bin_widths**2).sum()
                L_k_i = (sim_k_pdf_norm * obs_k_norm_pdf_binned).sum()

            log_L_k_i = log_prob(L_k_i)
            log_L_k_i_detect = log_prob(comp_at_kp_obs[ii])
            log_L_yng_i = log_prob(yng.prob[ii])

            log_L_k_detect += log_L_k_i + log_L_k_i_detect + log_L_yng_i


        # Binomial Coefficient
        if N_yng_obs >= N_yng_sim:
            log_binom_coeff = -np.Inf
        else:
            log_binom_coeff = scipy.special.gammaln(N_yng_sim + 1)
            log_binom_coeff -= scipy.special.gammaln(N_yng_obs + 1)
            log_binom_coeff -= scipy.special.gammaln(N_yng_sim - N_yng_obs + 1)

        log_L = log_L_N_WR + log_L_k_detect + log_binom_coeff
        log_L += log_L_k_non_detect + log_L_N_yng_sim

        # Add in the log(Prior_Probabilities) as well
        log_L += math.log(prob_dist)
        log_L += math.log(prob_alpha)
        log_L += math.log(prob_Mcl)
        log_L += math.log(prob_log_age_cont)
        log_L += math.log(prob_N_yng_obs)

        return log_L

    gcutil.mkdir(outdir)

    outroot = outdir + 'mnest_'

    num_dims = 5
    num_params = 7
    ev_tol = 0.7
    samp_eff = 0.8
    n_clust_param = num_dims - 1

    _run = open(outroot + 'params.run', 'w')
    _run.write('Num Dimensions: %d\n' % num_dims)
    _run.write('Num Params: %d\n' % num_params)
    _run.write('Evidence Tolerance: %.1f\n' % ev_tol)
    _run.write('Sampling Efficiency: %.1f\n' % samp_eff)
    _run.write('Num Clustering Params: %d\n' % n_clust_param)
    _run.write('Num Live Points: %d\n' % n_live_points)
    _run.write('Young Star Data: %s\n' % yng_file)
    _run.write('Allowed Multiples in Fit: %s\n' % str(multiples))
    _run.close()

    pymultinest.run(likelihood, priors, num_dims, n_params=num_params,
                    outputfiles_basename=outroot,
                    verbose=True, resume=False,
                    evidence_tolerance=ev_tol, sampling_efficiency=samp_eff,
                    n_clustering_params=n_clust_param,
                    n_live_points=n_live_points)

def run3(outdir, data=None, rmin=0, rmax=30, n_live_points=300, multiples=True,
         interact=False):
    magCut = 15.5

    # Load up the data arrays.
    if data == None:
        data = lu_gc_imf.load_all_catalog_by_radius(rmin, rmax, magCut=magCut)
        data_file = 'Observed Data'
    elif type(data) == str:
        data_file = data
        foo = open(data_file, 'r')
        data = pickle.load(foo)
        foo.close()

        idx = np.where(data.kp_ext <= magCut)
        data.kp = data.kp[idx]
        data.kp_ext = data.kp_ext[idx]
        data.kp_err = data.kp_err[idx]
        data.prob = data.prob[idx]
        data.mass = data.mass[idx]
        data.isYoung = data.isYoung[idx]
    else:
        data_file = 'Simulated Cluster Object'
        
    completeness = lu_gc_imf.load_image_completeness_by_radius(rmin, rmax)

    def priors(cube, ndim, nparams):
        return

    def likelihood(cube, ndim, nparams):
        # Set up variables (with priors)
        AKs = 2.7   # mag
        m_min = 1.0 # Msun
        m_max = 150 # Msun
        k_min = 8.0 # K' magnitude limit for old KLF powerlaw
        k_max = 18. # K' magnitude limit for old KLF powerlaw
        Z = 0.02

        ####################
        # Priors for model parameters
        ####################
        # Distance to cluster
        dist, prob_dist = random_distance(cube[0])
        cube[0] = dist

        # Log Age of the cluster
        log_age_cont, prob_log_age_cont = random_log_age(cube[1])
        log_age = round(log_age_cont, 2)
        cube[1] = log_age_cont

        # Slope of the IMF
        alpha, prob_alpha = random_alpha(cube[2])
        cube[2] = alpha

        # Total Cluster Mass
        Mcl, prob_Mcl = random_mass(cube[3])
        cube[3] = Mcl

        # Powerlaw slope for old population
        gamma, prob_gamma = random_gamma(cube[4])
        cube[4] = gamma

        # Number of old stars that exist (not just observed).
        # Remember this is from k_min brightness down to
        # the model magnitude cut (k_max).
        N_old, prob_N_old = random_N_old(cube[5])
        cube[5] = N_old

        # Mean of Red Clump: Note ratio of rec-clump to powerlaw is fixed.
        rcMean, prob_rcMean = random_rcMean(cube[6])
        rcSigma, prob_rcSigma = random_rcSigma(cube[7])
        cube[6] = rcMean
        cube[7] = rcSigma

        # Check that all our prior probabilities are valid, otherwise abort
        # before expensive calculation.
        if ((prob_alpha == 0) or (prob_log_age_cont == 0) or (prob_dist == 0) or
            (prob_Mcl == 0) or (prob_gamma == 0) or (prob_N_old == 0)):
            return -np.Inf

        # We will only be considering magnitudes down to some hard magnitude
        # limit that is well below our detection threshold. As long as we
        # renormalize everything, this should be fine. Just remember when
        # interpreting any Numbers of Young stars, etc.
        #model_mag_cut = magCut + (3.0 * data.kp_err.max())
        model_mag_cut = k_max

        ####################
        # PDF for young stars
        ####################
        # Get the PDF_y(k|model) -- the simulated luminosity function for young stars
        mod_sims = b.fetch_model_from_sims(log_age, AKs, dist*10**3,
                                           alpha, Mcl*10**3,
                                           m_min, m_max, makeMultiples=multiples)

        sim_N_WR = mod_sims[0]
        sim_k_bins = mod_sims[1]
        sim_k_pdf = mod_sims[2]
        sim_k_pdf_norm = mod_sims[3]
        sim_k_bin_widths = np.diff(sim_k_bins)
        sim_k_bin_center = sim_k_bins[:-1] + (sim_k_bin_widths/2.0)

        # Append bins at the bright end to always go up to at least
        # K' = k_min to cover the full range of observations.
        if sim_k_bins[0] > k_min:
            width = sim_k_bin_widths[0]
            new_bins = np.arange(sim_k_bins[0]-width, k_min, -width)
            if len(new_bins) > 0:
                tmp = np.zeros(len(new_bins), dtype=float)
                sim_k_bins = np.concatenate([new_bins[::-1], sim_k_bins])
                sim_k_pdf = np.concatenate([tmp, sim_k_pdf])
                sim_k_pdf_norm = np.concatenate([tmp, sim_k_pdf_norm])
                sim_k_bin_widths = np.diff(sim_k_bins)
                sim_k_bin_center = sim_k_bins[:-1] + (sim_k_bin_widths/2.0)
        
        # Trim down on the faint side to the model magnitude cut
        idx = np.where(sim_k_bins <= model_mag_cut)[0]

        sim_k_bins = np.append(sim_k_bins[idx], sim_k_bins[idx[-1]+1])
        sim_k_bin_center = sim_k_bin_center[idx]
        sim_k_bin_widths = sim_k_bin_widths[idx]
        sim_k_pdf = sim_k_pdf[idx]
        sim_k_pdf_norm = sim_k_pdf_norm[idx]
        sim_k_pdf_norm /= (sim_k_pdf_norm * sim_k_bin_widths).sum()

        ####################
        # PDF for old stars
        ####################
        pl_loc = math.e**k_min
        pl_scale = math.e**k_max - math.e**k_min
        pl_index = gamma * math.log(10)
        old_powerlaw = scipy.stats.powerlaw(pl_index, loc=pl_loc, scale=pl_scale)
        old_gaussian = scipy.stats.norm(loc=rcMean, scale=rcSigma)

        fracInRC = 0.12
        old_k_cdf = (1.0 - fracInRC) * old_powerlaw.cdf(math.e**sim_k_bins)
        old_k_cdf += fracInRC * old_gaussian.cdf(sim_k_bins)
        old_k_pdf_norm = np.diff(old_k_cdf)
        old_k_pdf_norm /= (old_k_pdf_norm * sim_k_bin_widths).sum()

        ####################
        # completeness curves
        ####################
        # Completeness curve (resampled to simulated Kp)
        Kp_interp = interpolate.splrep(completeness.mag, completeness.comp, k=1, s=0)
        comp_at_kp_sim = interpolate.splev(sim_k_bin_center, Kp_interp)
        comp_at_kp_sim[comp_at_kp_sim < 0] = 0.0
        comp_at_kp_sim[comp_at_kp_sim > 1] = 1.0
        comp_at_kp_sim[sim_k_bin_center > magCut] = 0.0

        # Completeness curve (resampled to observed Kp)
        comp_at_kp_obs = interpolate.splev(data.kp_ext, Kp_interp)
        comp_at_kp_obs[comp_at_kp_obs < 0] = 0.0
        comp_at_kp_obs[comp_at_kp_obs > 1] = 1.0
        comp_at_kp_obs[data.kp_ext > magCut] = 0.0

        # N_yng (down to model_magnitude cut)
        N_yng = int(np.round(sim_k_pdf.sum()))
        cube[8] = N_yng

        ####################
        # Different parts of the likelihood
        ####################
        #####
        # Keep track of the normalization constant (actually an array
        # that will be integrated at the end).
        log_norm_const = 0.0
        
        #####
        # Prob(N_WR | model)
        N_WR_sim = sim_N_WR
        cube[9] = N_WR_sim
        log_L_N_WR = log_prob( scipy.stats.poisson.pmf(data.N_WR, sim_N_WR) )

        N_tot = N_yng + N_old
        N_obs = len(data.kp_ext)
        fracYng = N_yng / N_tot   # this is our mixture model weight

        #####
        # Binomial Coefficient
        if N_obs >= N_tot:
            return -np.Inf
        else:
            log_binom_coeff = scipy.special.gammaln(N_tot + 1)
            log_binom_coeff -= scipy.special.gammaln(N_obs + 1)
            log_binom_coeff -= scipy.special.gammaln(N_tot - N_obs + 1)

        log_norm_const += log_binom_coeff

        #####
        # Non detections: log_L_k_non_detect
        ## Young part
        tmp_y = fracYng * sim_k_pdf_norm * sim_k_bin_widths
        tmp_y *= (1.0 - comp_at_kp_sim)
        P_I0_y = tmp_y.sum()

        ## Old part
        tmp_o = (1.0 - fracYng) * old_k_pdf_norm * sim_k_bin_widths
        tmp_o *= (1.0 - comp_at_kp_sim)
        P_I0_o = tmp_o.sum()

        ## log[ prob(I=0 | model)^(N-n) ]
        log_L_k_non_detect = (N_tot - N_obs) * log_prob(P_I0_y + P_I0_o)
        # log_L_k_non_detect = (N_tot - N_obs) * log_prob(P_I0_y) # TMP

        log_norm_const += log_L_k_non_detect

        #####
        # Detections: log_L_k_detect
        log_L_k_detect = 0.0

        tmp1 = fracYng * sim_k_pdf_norm * sim_k_bin_widths * comp_at_kp_sim
        tmp2 = (1.0 - fracYng) * old_k_pdf_norm * sim_k_bin_widths * comp_at_kp_sim
        tmp3 = N_obs * log_prob(tmp1.sum() + tmp2.sum())
        # TMP
        #tmp3 = N_obs * log_prob(tmp1.sum())

        log_norm_const += tmp3

        arr_L_k_y_np = np.zeros(len(data.kp_ext), dtype=float)
        arr_L_k_y = np.zeros(len(data.kp_ext), dtype=float)
        arr_L_k_o = np.zeros(len(data.kp_ext), dtype=float)
        arr_L_k = np.zeros(len(data.kp_ext), dtype=float)
        arr_L_k_comp = np.zeros(len(data.kp_ext), dtype=float)

        # Loop through each star and calc prob of detecting.
        for ii in range(len(data.kp_ext)):
            # Make gaussian around observed k mag
            obs_k_norm = scipy.stats.norm(loc=data.kp_ext[ii], scale=data.kp_err[ii])
            obs_k_norm_cdf = obs_k_norm.cdf(sim_k_bins)
            obs_k_norm_pdf = np.diff(obs_k_norm_cdf)
            
            # Convolve gaussian with PDF(K) from model
            if obs_k_norm_pdf.sum() == 0:
                print 'We have a problem... this should never happen.'
                pdb.set_trace()
                #L_k_i = 0
                return -np.Inf
            else:
                # Young part
                # Multiply gaussian with PDF(K) from model and sum to get probability
                L_k_i_y = (sim_k_pdf_norm * obs_k_norm_pdf * sim_k_bin_widths).sum()
                arr_L_k_y_np[ii] = log_prob(L_k_i_y)
                L_k_i_y *= data.prob[ii]

                # Old part
                L_k_i_o = (old_k_pdf_norm * obs_k_norm_pdf * sim_k_bin_widths).sum()
                L_k_i_o *= (1.0 - data.prob[ii])

                # Combine
                L_k_i = L_k_i_y + L_k_i_o
                #L_k_i = L_k_i_y # TMP

                arr_L_k_y[ii] = log_prob(L_k_i_y)
                arr_L_k_o[ii] = log_prob(L_k_i_o)
                arr_L_k[ii] = log_prob(L_k_i)
                arr_L_k_comp[ii] = log_prob(L_k_i) + log_prob(comp_at_kp_obs[ii])
                
            log_L_k_detect += log_prob(L_k_i) + log_prob(comp_at_kp_obs[ii])

        log_L = log_L_N_WR + log_L_k_detect + log_binom_coeff
        log_L += log_L_k_non_detect
        log_L -= log_norm_const

        # Add in the log(Prior_Probabilities) as well
        log_L += math.log(prob_dist)
        log_L += math.log(prob_alpha)
        log_L += math.log(prob_Mcl)
        log_L += math.log(prob_log_age_cont)
        log_L += math.log(prob_gamma)
        log_L += math.log(prob_N_old)
        log_L += math.log(prob_rcMean)
        log_L += math.log(prob_rcSigma)

        if log_L >= 0:
            pdb.set_trace()

        if interact:
            py.figure(1)
            py.clf()
            py.plot(data.kp_ext, arr_L_k_y, 'b.')
            py.plot(data.kp_ext, arr_L_k_o, 'r.')
            py.plot(data.kp_ext, arr_L_k, 'g.')
            py.plot(data.kp_ext, arr_L_k_comp, 'k.')
            py.ylim(-10, -2)
            
            py.figure(4)
            py.clf()
            bins = np.arange(-15, 0, 1)
            py.hist(arr_L_k_y[data.prob == 1], label='young',
                    histtype='step', bins=bins)
            # py.hist(arr_L_k[data.prob == 0], label='old',
            #         histtype='step', normed=True, bins=bins)
            idx = np.where((data.prob > 0) & (data.prob < 1))[0]
            if len(idx) > 0:
                py.hist(arr_L_k_y[idx], label='uncertain',
                        histtype='step', bins=bins, weights=data.prob[idx])
                py.hist(arr_L_k_y, label='all',
                        histtype='step', bins=bins, weights=data.prob)
            
            py.legend(loc='upper left')
            py.xlabel('log L')

            binsKp = np.arange(9.0, 17, 1.0)  # Bin Center Pointsklf_mag_bins
            binEdges = binsKp[0:-1] + (binsKp[1:] - binsKp[0:-1]) / 2.0

            tdx = np.where(sim_k_bins <= magCut)[0]
            k_bins_tmp = np.append(sim_k_bins[tdx], sim_k_bins[tdx[-1]+1])
            sim_k_pdf_tmp = (sim_k_pdf * comp_at_kp_sim)[tdx]
            old_k_pdf_tmp = (old_k_pdf_norm * N_old * sim_k_bin_widths * comp_at_kp_sim)[tdx]

            N_yng_sim = (sim_k_pdf_norm * N_yng * sim_k_bin_widths * comp_at_kp_sim)[tdx].sum()
            N_old_sim = (old_k_pdf_norm * N_old * sim_k_bin_widths * comp_at_kp_sim)[tdx].sum()
            N_tot_sim = N_yng_sim + N_old_sim

            py.figure(2)
            py.clf()
            py.hist(data.kp_ext, bins=binEdges, histtype='step', color='black',
                    weights=data.prob)
            sim_k_pdf_rebin = rebin.rebin(k_bins_tmp, sim_k_pdf_tmp, binEdges)
            rebin.edge_step(binEdges, sim_k_pdf_rebin, color='red')
            py.plot(9.5, N_WR_sim, 'rs', ms=10)
            py.plot(9.5, data.N_WR, 'ko', ms=9)
            py.title('Young')

            py.figure(3)
            py.clf()
            py.hist(data.kp_ext, bins=binEdges, histtype='step', color='black',
                    weights=(1.0-data.prob))
            old_k_pdf_rebin = rebin.rebin(k_bins_tmp, old_k_pdf_tmp, binEdges)
            rebin.edge_step(binEdges, old_k_pdf_rebin, color='red')
            py.title('Old')

            
            print 'dist    = %6.3f   log_prob = %8.1f' % (dist, math.log(prob_dist))
            print 'log_age = %6.2f   log_prob = %8.1f' % (log_age_cont, math.log(prob_log_age_cont))
            print 'alpha   = %6.2f   log_prob = %8.1f' % (alpha, math.log(prob_alpha))
            print 'Mcl     = %6.2f   log_prob = %8.1f' % (Mcl, math.log(prob_Mcl))
            print 'gamma   = %6.2f   log_prob = %8.1f' % (gamma, math.log(prob_gamma))
            print 'N_old   = %6d   log_prob = %8.1f' % (N_old, math.log(prob_N_old))
            print 'rcMean  = %6.2f   log_prob = %8.1f' % (rcMean, math.log(prob_rcMean))
            print 'rcSigma = %6.2f   log_prob = %8.1f' % (rcSigma, math.log(prob_rcSigma))
            print ''
            print 'Simulated Stars:'
            print 'N_yng = %6d' % (N_yng_sim)
            print 'N_old = %6d' % (N_old_sim)
            print 'N_tot = %6d' % (N_tot_sim)
            print 'N_WR = %6d' % (N_WR_sim)
            print ''
            print 'Observed Stars:'
            print 'N_yng = %6d' % (data.prob.sum())
            print 'N_old = %6d' % ((1.0 - data.prob).sum())
            print 'N_obs = %6d' % (N_obs)
            print 'N_WR = %6d' % (data.N_WR)
            print ''
            print 'Binomial: '
            print 'N_tot = %6d' % N_tot
            print 'N_obs = %6d' % N_obs
            print ''
            print 'Likelihood:'
            print 'log_L_N_WR         = %8.1f' % log_L_N_WR
            print 'log_L_binom_coeff  = %8.1f' % log_binom_coeff
            print 'log_L_k_detect     = %8.1f' % log_L_k_detect
            print 'log_L_k_non_detect = %8.1f' % log_L_k_non_detect
            print 'log_norm_const     = %8.1f' % log_norm_const
            print ''
            idx = np.where(data.prob > 0)[0]
            print 'YNG arr_L_k_y_np   = %8.1f' % arr_L_k_y_np[idx].sum()
            print 'YNG arr_L_k_y      = %8.1f' % arr_L_k_y[idx].sum()
            print 'FINAL: '
            print 'log_L = %8.1f' % log_L
            print ''
            
            #pdb.set_trace()
            
        cube[10] = log_L_N_WR
        cube[11] = log_binom_coeff
        cube[12] = log_L_k_detect
        cube[13] = log_L_k_non_detect
        cube[14] = log_norm_const

        return log_L

    gcutil.mkdir(outdir)

    outroot = outdir + 'mnest_'

    num_dims = 8
    num_params = 15
    ev_tol = 0.7
    samp_eff = 0.8
    n_clust_param = num_dims - 1

    _run = open(outroot + 'params.run', 'w')
    _run.write('Num Dimensions: %d\n' % num_dims)
    _run.write('Num Params: %d\n' % num_params)
    _run.write('Evidence Tolerance: %.1f\n' % ev_tol)
    _run.write('Sampling Efficiency: %.1f\n' % samp_eff)
    _run.write('Num Clustering Params: %d\n' % n_clust_param)
    _run.write('Num Live Points: %d\n' % n_live_points)
    _run.write('Young Star Data: %s\n' % data_file)
    _run.write('Allowed Multiples in Fit: %s\n' % str(multiples))
    _run.close()

    pymultinest.run(likelihood, priors, num_dims, n_params=num_params,
                    outputfiles_basename=outroot,
                    verbose=True, resume=False,
                    evidence_tolerance=ev_tol, sampling_efficiency=samp_eff,
                    n_clustering_params=n_clust_param,
                    n_live_points=n_live_points)


def load_results(rootdir):
    root = '%s/mnest_' % (rootdir)
    tab = atpy.Table(root + '.txt', type='ascii')

    # Convert to log(likelihood)
    tab['col2'] /= -2.0
    
    # Rename the parameter columns. This is hard-coded to match the
    # above run() function.
    tab.rename_column('col1', 'weights')
    tab.rename_column('col2', 'logLike')
    tab.rename_column('col3', 'distance')
    tab.rename_column('col4', 'logAge')
    tab.rename_column('col5', 'alpha')
    tab.rename_column('col6', 'Mcl')
    tab.rename_column('col7', 'gamma')
    tab.rename_column('col8', 'N_old')
    tab.rename_column('col9', 'rcMean')
    tab.rename_column('col10', 'rcSigma')
    tab.rename_column('col11', 'N_yng')
    tab.rename_column('col12', 'N_WR_sim')
    tab.rename_column('col13', 'log_L_N_WR')
    tab.rename_column('col14', 'log_L_binom_coeff')
    tab.rename_column('col15', 'log_L_k_detect')
    tab.rename_column('col16', 'log_L_k_non_detect')
    tab.rename_column('col17', 'log_L_norm_const')

    # Now sort based on logLikelihood
    tab.sort('logLike')

    return tab

def plot_posteriors(outdir):
    tab = load_results(outdir)
    tab.remove_columns(('weights', 'logLike'))

    pair_posterior(tab, weights, outfile=outroot+'posteriors.png', title=outdir)

def plot_posteriors_1D(outdir, sim=True):
    tab = load_results(outdir)

    # If this is simulated data, load up the simulation and overplot
    # the input values on each histogram.
    if sim:
        parts = outdir.split('_')

        logAge = float(parts[3][1:])
        distance = float(parts[5][1:]) / 10**3
        imfSlope = float(parts[6][1:])
        Mcl = int(parts[7][1:]) / 10**3
        
        tmp2 = 'cluster_' + '_'.join(parts[2:])
        tmp3 = tmp2.replace('/', '')
        tmp3 += '.pickle'

        foo = open(tmp3, 'r')
        sim = pickle.load(foo)
        foo.close()

        numWR = sim.N_WR
        numOB = sim.prob.sum()

    fontsize = 12

    py.close(1)
    py.figure(1, figsize = (10,10))
    py.subplots_adjust(left=0.05, right=0.95, bottom=0.07, top=0.95, wspace=0.5, hspace=0.3)

    ax1 = py.subplot2grid((3, 6), (0, 0), colspan=3)
    ax2 = py.subplot2grid((3, 6), (0, 3), colspan=3)
    ax3 = py.subplot2grid((3, 6), (1, 0), colspan=3)
    ax4 = py.subplot2grid((3, 6), (1, 3), colspan=3)
    ax5 = py.subplot2grid((3, 6), (2, 0), colspan=2)
    ax6 = py.subplot2grid((3, 6), (2, 2), colspan=2)
    ax7 = py.subplot2grid((3, 6), (2, 4), colspan=2)

    def plot_PDF(ax, paramName, counter=False):
        if counter:
            bins = np.arange(0, round(tab[paramName].max()))
        else:
            bins = 50
        n, bins, patch = ax.hist(tab[paramName], normed=True, histtype='step',
                                 weights=tab['weights'], bins=bins)
        py.setp(ax.get_xticklabels(), fontsize=fontsize)
        py.setp(ax.get_yticklabels(), fontsize=fontsize)
        ax.set_xlabel(paramName, size=fontsize+2)
        ax.set_ylim(0, n.max()*1.1)

    plot_PDF(ax1, 'alpha')
    plot_PDF(ax2, 'logAge')
    plot_PDF(ax3, 'Mcl')
    plot_PDF(ax4, 'distance')
    plot_PDF(ax5, 'N_WR_sim', counter=True)
    plot_PDF(ax6, 'gamma', counter=True)
    plot_PDF(ax7, 'N_old', counter=True)

    # Make some adjustments to the axes for Number of stars plots
    N_WR_sim_avg = np.average(tab['N_WR_sim'], weights=tab['weights'])
    N_WR_sim_std = math.sqrt( np.dot(tab['weights'], (tab['N_WR_sim']-N_WR_sim_avg)**2) / tab['weights'].sum() )
    N_WR_lo = N_WR_sim_avg - (3 * N_WR_sim_std)
    N_WR_hi = N_WR_sim_avg + (3 * N_WR_sim_std)
    if N_WR_lo < 0:
        N_WR_lo = 0
    ax5.set_xlim(N_WR_lo, N_WR_hi)

    if sim:
        ax1.axvline(imfSlope, color='red')
        ax2.axvline(logAge, color='red')
        ax3.axvline(Mcl, color='red')
        ax4.axvline(distance, color='red')
        ax5.axvline(numWR, color='red')

    py.suptitle(outdir)
    py.savefig(outdir + 'plot_posteriors_1D.png')

def plot_posteriors3(outdir):
    outroot = '%s/mnest_' % (outdir)
    num_params = 10

    tab = atpy.Table(outroot + '.txt', type='ascii')

    if num_params != (tab.shape[1] - 2):
        print 'N_params mismatch: ', outroot

    # First column is the weights
    weights = tab['col1']
    logLike = tab['col2'] / -2.0
    
    # Now delete the first two rows
    tab.remove_columns(('col1', 'col2'))

    # Rename the parameter columns. This is hard-coded to match the
    # above run() function.
    tab.rename_column('col3', 'distance')
    tab.rename_column('col4', 'logAge')
    tab.rename_column('col5', 'alpha')
    tab.rename_column('col6', 'Mcl')
    tab.rename_column('col7', 'gamma')
    tab.rename_column('col8', 'N_old')
    tab.rename_column('col9', 'rcMean')
    tab.rename_column('col10', 'rcSigma')
    tab.rename_column('col11', 'N_yng')
    tab.rename_column('col12', 'N_WR_sim')

    pair_posterior(tab, weights, outfile=outroot+'posteriors.png', title=outdir)

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

    fontsize = 10

    py.figure(figsize = (10,10))
    py.subplots_adjust(left=0.05, right=0.95, bottom=0.05, top=0.95)

    # Margenilized 1D
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

    if title != None:
        py.suptitle(title)

    if outfile != None:
        py.savefig(outfile)

    return

def log_prob(value):
    if (type(value) in [int, float]) or (value.shape == ()):
        if value == 0:
            return -np.Inf
        else:
            return np.log(value)
    else:
        log_val = np.zeros(len(value), dtype=float)

        idx = np.where(value == 0)[0]
        if len(idx) != 0:
            cdx = np.where(value != 0)[0]

            log_val[idx] = -np.Inf
            log_val[cdx] = np.log(value[cdx])
            return log_val
        else:
            return np.log(value)

    return math.log(value)

def plot_results_3d(outdir, param1='alpha', param2='logAge'):
    outroot = '%s/mnest_' % (outdir)
    num_params = 7

    tab = atpy.Table(outroot + '.txt', type='ascii')

    if num_params != (tab.shape[1] - 2):
        print 'N_params mismatch: ', outroot

    # First column is the weights
    weights = tab['col1']
    logLike = tab['col2'] / -2.0
    
    # Now delete the first two rows
    tab.remove_columns(('col1', 'col2'))

    # Rename the parameter columns. This is hard-coded to match the
    # above run() function.
    tab.rename_column('col3', 'distance')
    tab.rename_column('col4', 'logAge')
    tab.rename_column('col5', 'alpha')
    tab.rename_column('col6', 'Mcl')
    tab.rename_column('col7', 'N_yng_obs')
    tab.rename_column('col8', 'N_WR_sim')
    tab.rename_column('col9', 'N_yng_sim')

    py.close(1)
    fig = py.figure(1, figsize=(10,10))
    ax = fig.add_subplot(111, projection='3d')

    # Map the highest probability regions
    peak = logLike.max()
    idx1 = np.where(logLike > peak-1)[0]
    idx2 = np.where((logLike <= peak-1) & (logLike > (peak-2)))[0]
    idx3 = np.where((logLike <= peak-2) & (logLike > (peak-3)))[0]

    ax.scatter(tab[param1][idx3], tab[param2][idx3], logLike[idx3], color='blue')
    ax.scatter(tab[param1][idx2], tab[param2][idx2], logLike[idx2], color='green')
    ax.scatter(tab[param1][idx1], tab[param2][idx1], logLike[idx1], color='red')

def simulated_data(logAge=6.6, AKs=2.7, distance=8000, alpha=2.35, Mcl=10**4,
                   gamma=0.3, Nold=1.5e3, rcMean=15.6, rcSigma=0.3, multiples=True):
    if multiples:
        multiStr = 'multi'
    else:
        multiStr = 'single'

    out_root = 'cluster_sim_t%.2f_AKs%.1f_d%d_a%.2f_m%d_g%.2f_no%d_%s' % \
        (logAge, AKs, distance, alpha, Mcl, gamma, Nold, multiStr)

    # This is the magnitude range over which the mixture model weights
    # are determined.
    k_min = 8.0  # K' magnitude limit for old KLF powerlaw
    k_max = 18.0 # K' magnitude limit for old KLF powerlaw
    
    # Generate young stars.
    cluster = b.model_young_cluster(logAge, AKs=AKs, imfSlope=alpha,
                                  distance=distance, clusterMass=Mcl,
                                  makeMultiples=multiples)

    # Generate old stars
    pl_loc = math.e**k_min
    pl_scale = math.e**k_max - math.e**k_min
    pl_index = gamma * math.log(10.0)
    powerlaw = scipy.stats.powerlaw(pl_index, loc=pl_loc, scale=pl_scale)
    gaussian = scipy.stats.norm(loc=rcMean, scale=rcSigma)

    # Fix the relative fraction of stars in the Red Clump
    fracInRC = 0.12
    kp_PLAW = np.log( powerlaw.rvs((1.0 - fracInRC)*Nold) )
    kp_NORM = gaussian.rvs(fracInRC*Nold)
    old_sim_kp = np.concatenate([kp_PLAW, kp_NORM])

    # Assign photometric errors to both young and old.
    # Perturb the magnitudes by their photometric errors
    kp_yng = scipy.stats.norm.rvs(size=len(cluster.mag_noWR),
                                  scale=0.1, loc=cluster.mag_noWR)
    kp_old = scipy.stats.norm.rvs(size=Nold,
                                  scale=0.1, loc=old_sim_kp)

    # Load up imaging completness curve
    completeness = lu_gc_imf.load_image_completeness_by_radius()

    # Determine the probability of detection (in images) for each star
    Kp_interp = interpolate.splrep(completeness.mag, completeness.comp, k=1, s=0)
    comp_for_yng_stars = interpolate.splev(kp_yng, Kp_interp)
    comp_for_yng_stars[comp_for_yng_stars > 1] = 1.0
    comp_for_yng_stars[comp_for_yng_stars < 0] = 0.0

    comp_for_old_stars = interpolate.splev(kp_old, Kp_interp)
    comp_for_old_stars[comp_for_old_stars > 1] = 1.0
    comp_for_old_stars[comp_for_old_stars < 0] = 0.0

    # Randomly select detected stars based on the imaging completeness curve
    rand_number = np.random.rand(len(cluster.mag_noWR))
    detected_yng = np.where((rand_number < comp_for_yng_stars))[0]

    rand_number = np.random.rand(Nold)
    detected_old = np.where((rand_number < comp_for_old_stars))[0]
    

    # Merge together the two data sets
    data = objects.DataHolder()
    data.kp = np.concatenate([kp_yng[detected_yng], kp_old[detected_old]])
    data.kp_ext = np.concatenate([kp_yng[detected_yng], kp_old[detected_old]])
    data.kp_err = np.ones(len(detected_yng) + len(detected_old), dtype=float) * 0.1
    data.N_WR = cluster.num_WR
    data.mass = np.concatenate([cluster.mass[cluster.idx_noWR][detected_yng],
                                np.zeros(len(detected_old))])
    data.isYoung = np.concatenate([np.ones(len(detected_yng), dtype=bool),
                                   np.zeros(len(detected_old), dtype=bool)])

    # data = objects.DataHolder()
    # data.kp = kp_yng[detected_yng]
    # data.kp_ext = kp_yng[detected_yng]
    # data.kp_err = np.ones(len(detected_yng), dtype=float) * 0.1
    # data.N_WR = cluster.num_WR
    # data.mass = cluster.mass[cluster.idx_noWR][detected_yng]
    # data.isYoung = np.ones(len(detected_yng), dtype=bool)

    # Assign probability of youth. Assume everything is perfectly characterized
    # up to 14. Then an increasing fraction of the stars will have imperfect
    # characterization going from 100% with Prob(yng) = 0 or 1 at Kp=14 to
    # 0% at Kp=16.5. Assume all young stars with imperfect knowledge
    # have a randomly selected P(yng) between 0.4 - 1.0. Assume all old stars
    # with imperfect knowledge have a randomly selected P(yng) between 0 - 0.4.
    # This effectively means that we "know" the old stars better than the young
    # stars.
    prob_ID = (16.5 - data.kp) / (16.5 - 14.0)
    prob_ID[data.kp < 14] = 1
    prob_ID[data.kp > 16.5] = 0

    # Default, we know it's type
    data.prob = np.ones(len(data.kp), dtype=float)
    data.prob[data.isYoung == False] = 0.0

    # # randomly decide if each star is perfectly ID'ed
    # rand_number = np.random.rand(len(data.kp))
    # idx = np.where(rand_number > prob_ID)[0]      # all stars with uncertain types
    # ydx = np.where(data.isYoung[idx] == True)[0]  # young uncertains
    # odx = np.where(data.isYoung[idx] == False)[0] # old uncertains
    # data.prob[idx[ydx]] = np.random.uniform(low=0.4, high=1.0, size=len(ydx))
    # data.prob[idx[odx]] = np.random.uniform(low=0.0, high=0.4, size=len(odx))

    return data

def test_prob_old():
    """
    I need to figure out the exact prob_old(kp) functional form to use.
    """
    obs = lu_gc_imf.load_all_catalog_by_radius()

    idx = np.where((obs.kp_ext >= 10) & (obs.kp_ext < 14))[0]
    rdx = np.where((obs.kp_ext >= 14) & (obs.kp_ext < 15.5))[0]
    print 'OBS results:'
    print 'N_old with 10.0 <= Kp < 14.0: %d' % (1.0 - obs.prob[idx]).sum()
    print 'N_old with 14.0 <= Kp < 15.5: %d' % (1.0 - obs.prob[rdx]).sum()

    Nold = 1.5e3
    gamma = 0.3
    rcMean = 15.5
    rcSigma = 0.3
    k_min = 8.0
    k_max = 18.0

    pl_loc = math.e**k_min
    pl_scale = math.e**k_max - math.e**k_min
    pl_index = gamma * math.log(10.0)
    powerlaw = scipy.stats.powerlaw(pl_index, loc=pl_loc, scale=pl_scale)
    gaussian = scipy.stats.norm(loc=rcMean, scale=rcSigma)

    # Fix the relative fraction of stars in the Red Clump
    fracInRC = 0.12
    kp_PLAW = np.log( powerlaw.rvs((1.0 - fracInRC)*Nold) )
    kp_NORM = gaussian.rvs(fracInRC*Nold)
    kp = np.concatenate([kp_PLAW, kp_NORM])

    idx = np.where((kp >= 10.0) & (kp < 14.0))[0]
    rdx = np.where((kp >= 14.0) & (kp < 15.5))[0]
    print 'SIM results:'
    print 'N_old with 10.0 <= Kp < 14.0: %d' % len(idx)
    print 'N_old with 14.0 <= Kp < 15.5: %d' % len(rdx)

    outRoot = '/u/jlu/work/gc/imf/klf/2012_02_14/multinest/test_model/'

    sim = simulated_data()
    kp_bins = np.arange(8, 18, 1.0)
    py.figure(1)
    py.clf()
    py.hist(sim.kp, histtype='step', bins=kp_bins, label='Sim')
    py.hist(obs.kp_ext, histtype='step', bins=kp_bins, label='Obs')
    py.legend(loc='upper left')
    py.xlabel('Kp')
    py.xlim(8, 16)
    py.ylim(0, 400)
    py.title('All Stars')
    py.savefig(outRoot + 'klf_sim_vs_obs.png')

    py.figure(2)
    py.clf()
    py.hist(sim.kp, histtype='step', bins=kp_bins, label='Sim', weights=sim.prob)
    py.hist(obs.kp_ext, histtype='step', bins=kp_bins, label='Obs', weights=obs.prob)
    py.legend(loc='upper left')
    py.xlabel('Kp')
    py.xlim(8, 16)
    py.ylim(0, 150)
    py.title('Young Stars')
    py.savefig(outRoot + 'klf_sim_vs_obs_yng.png')

    py.figure(3)
    py.clf()
    py.hist(sim.kp, histtype='step', bins=kp_bins, label='Sim', weights=(1.0 - sim.prob))
    py.hist(obs.kp_ext, histtype='step', bins=kp_bins, label='Obs', weights=(1.0 - obs.prob))
    py.legend(loc='upper left')
    py.xlabel('Kp')
    py.xlim(8, 16)
    py.ylim(0, 400)
    py.title('Old Stars')
    py.savefig(outRoot + 'klf_sim_vs_obs_old.png')

    print len(sim.kp), sim.prob.sum() + (1.0 - sim.prob).sum()
    print len(obs.kp), obs.prob.sum() + (1.0 - obs.prob).sum()


def make_simulated_data_set(logAge, AKs, distance, imfSlope, clusterMass,
                            Nold, multiples=True, suffix=''):

    sim = simulated_data(logAge, AKs, distance, imfSlope, clusterMass,
                         Nold=Nold, multiples=multiples)
    if multiples:
        multi = 'multi'
    else:
        multi = 'single'

    out_root = 'cluster_sim_t%.2f_AKs%.1f_d%d_a%.2f_m%d_o%d_%s%s' % \
	(logAge, AKs, distance, imfSlope, clusterMass, Nold, multi, suffix)

    out_file = out_root + '.pickle'
    _out = open(out_file, 'w')
    pickle.dump(sim, _out)
    _out.close()

def run_simulated_data_set(logAge, AKs, distance, imfSlope, clusterMass,
                           Nold, multiples, fitMultiples, suffix=''):
    if multiples:
        multi = 'multi'
    else:
        multi = 'single'

    if fitMultiples:
        fit_multi = 'multi'
    else:
        fit_multi = 'single'

    info = 'sim_t%.2f_AKs%.1f_d%d_a%.2f_m%d_o%d_%s%s' % \
        (logAge, AKs, distance, imfSlope, clusterMass, Nold, multi, suffix)

    data_file = 'cluster_%s.pickle' % (info)
    out_dir = 'fit_%s_%s/' % (fit_multi, info)

    run3(out_dir, data=data_file, multiples=fitMultiples)

    return out_dir

