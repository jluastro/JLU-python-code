import numpy as np
import pylab as py
from jlu.stellarModels import evolution
from jlu.nirc2 import synthetic
from jlu.stellarModels import atmospheres
from jlu.papers import lu_gc_imf
from scipy import interpolate
from scipy import stats
from gcwork import objects
from pysynphot import spectrum
from jlu.util import constants
from jlu.util import plfit
import pickle
import time, datetime
import math
import os, glob
import tempfile
import scipy
import matplotlib
import pymc
import pdb

defaultAKs = 2.7
defaultDist = 8000
defaultFilter = 'Kp'
defaultMFamp = 0.44
defaultMFindex = 0.51
defaultCSFamp = 0.50
defaultCSFindex = 0.45
defaultCSFmax = 3


def pymc_run():
    """
    This should really be run from the command line. Use this as a cut and
    paste template.
    """
    mc_vars = pymc_model()

    dbname = 'mcmc_chain1'

    mc = pymc.MCMC(mc_vars, db='hdf5', dbname=dbname, dbmode='a', verbose=3)
    mc.isample(iter=10000, burn=0, verbose=3)
    mc.commit()

    return mc_vars, mc

def pymc_model(yng=None, rmin=0, rmax=30):
    # Set up variables (with priors)
    AKs = 2.7   # mag
    m_min = 1.0 # Msun
    m_max = 150 # Msun
    Z = 0.02
    magCut = 15.5

    verbose = 3

    # Distance to cluster
    dist_mean = 8096  # pc
    dist_std = 483    # pc
    dist = pymc.TruncatedNormal('dist', dist_mean, 1.0/dist_std**2, a=6793.0, b=9510.0,
                                verbose=verbose)

    # Age of the cluster
    age_mean = 6e6   # Myr
    age_std = 2e6    # Myr
    age = pymc.TruncatedNormal('age', age_mean, 1.0/age_std**2, a=2.0e6, b=1.2e7,
                               verbose=verbose)

    @pymc.deterministic(verbose=verbose)
    def logAge(age=age):
        " convert and round age"
        return round(np.log10(age), 2)
        
    # Slope of the IMF
    alpha = pymc.Uniform('alpha', 0.15, 3.0, verbose=verbose)

    # Total Cluster Mass
    Mcl = pymc.Uniform('Mcl', 3e3, 1e5, value=6e4, verbose=verbose)

    # Load up the data arrays.
    if yng == None:
        yng = lu_gc_imf.load_yng_data_by_radius(rmin, rmax, magCut=magCut)
    comp = lu_gc_imf.load_image_completeness_by_radius(rmin, rmax)

    # Setup the Likelihood
    @pymc.potential(verbose=verbose)
    def likelihood(yngData=yng, completeness=comp, 
                   logAge=logAge, AKs=AKs, distance=dist,
                   imfSlope=alpha, clusterMass=Mcl,
                   minMass=m_min, maxMass=m_max, magCut=magCut):
        print 'likelihood: '
        print '    logAge      = %.2f' % logAge
        print '    AKs         = %.2f' % AKs
        print '    distance    = %d' % dist
        print '    imfSlope    = %.2f' % alpha
        print '    clusterMass = %.2e' % Mcl
        print '    minMass     = %.1f' % m_min
        print '    maxMass     = %d' % m_max

        # Get the PDF(k|model) -- the simulated luminosity function
        mod_sims = fetch_model_from_sims(logAge, AKs, distance,
                                         imfSlope, clusterMass,
                                         minMass, maxMass)

        sim_N_WR = mod_sims[0]
        sim_k_bins = mod_sims[1]
        sim_k_pdf = mod_sims[2]
        sim_k_pdf_norm = mod_sims[3]

        # Resample and renormalize the PDF for the mag cut
        idx = np.where(sim_k_bins <= (magCut + 0.5))[0]
        sim_k_bins = sim_k_bins[idx]
        sim_k_pdf = sim_k_pdf[idx[:-1]]
        sim_k_pdf_norm = sim_k_pdf_norm[idx[:-1]]
        sim_k_bin_widths = np.diff(sim_k_bins)
        sim_k_bin_center = sim_k_bins[:-1] + (sim_k_bin_widths/2.0)
        sim_k_pdf_norm /= (sim_k_pdf_norm * sim_k_bin_widths).sum()


        # Multiply by the completeness curve (after re-sampling). And renormalize.
        Kp_interp = interpolate.splrep(completeness.mag, completeness.comp, k=1, s=0)
        comp_resamp = interpolate.splev(sim_k_bin_center, Kp_interp)
        comp_resamp[comp_resamp < 0] = 0.0
        comp_resamp[comp_resamp > 1] = 1.0

        # Prob(N_WR | model)
        log_L_N_WR = pymc.poisson_like(yng.N_WR, sim_N_WR)

        ##############################
        #
        # KLF is handled with a binomial distribution that accounts for
        # both detections and non-detections.
        #
        ##############################
        ###############
        # Here is the detection part.
        ###############
        log_L_k_detect = 1.0

        # Get the probability of detecting (completeness) at each star's Kp
        comp_at_kp = interpolate.splev(yng.kp, Kp_interp)
        comp_at_kp[comp_at_kp < 0] = 0.0
        comp_at_kp[comp_at_kp > 1] = 1.0

        # Loop through each star and calc prob of detecting.
        for ii in range(len(yng.kp)):
            # Make gaussian around observed k mag
            obs_k_norm = scipy.stats.norm(loc=yng.kp[ii], scale=yng.kp_err[ii])
            obs_k_norm_cdf = obs_k_norm.cdf(sim_k_bins)
            obs_k_norm_pdf_binned = np.diff(obs_k_norm_cdf)

            # Convolve gaussian with P(KLF) from model
            L_k_i = (sim_k_pdf_norm * obs_k_norm_pdf_binned * sim_k_bin_widths).sum()

            # Multiply times P(detect)
            L_k_i *= comp_at_kp[ii]

            # Multiply times P(yng)
            L_k_i *= yng.prob[ii]

            if L_k_i != 0:
                log_L_k_detect += np.log(L_k_i)
            else:
                log_L_k_detect = -np.Inf
                break

        ###############
        # Here is the non-detection part.
        ###############

        N_yng_expect = sim_k_pdf.sum()
        N_yng = scipy.stats.distributions.poisson.rvs(1, loc=N_yng_expect)
        log_L_N_yng = pymc.poisson_like(N_yng, N_yng_expect)

        N_yng_obs_expect = yng.prob.sum()
        N_yng_obs = scipy.stats.distributions.poisson.rvs(1, loc=N_yng_obs_expect)
        log_L_N_yng_obs = pymc.poisson_like(N_yng_obs, N_yng_obs_expect)

        if N_yng < N_yng_obs:
            log_binom_coeff = -np.Inf
        else:
            binomial_coeff = math.factorial(N_yng)
            binomial_coeff /= math.factorial(N_yng_obs) * math.factorial(N_yng - N_yng_obs)
            log_binom_coeff = math.log(binomial_coeff)


        tmp = (1.0 - comp_resamp) * sim_k_pdf_norm * sim_k_bin_widths
        P_non_detect = tmp.sum()
        if (N_yng >= N_yng_obs) and (P_non_detect != 0):
            log_L_k_non_detect = (N_yng - N_yng_obs) * np.log(P_non_detect)
        else:
            log_L_k_non_detect = -np.Inf
        

        log_L = log_binom_coeff + log_L_N_yng + log_L_N_yng_obs + \
            log_L_k_detect + log_L_k_non_detect + log_L_N_WR

        return log_L

    return vars()


def pymc_model2(yng=None, rmin=0, rmax=30):
    # Set up variables (with priors)
    AKs = 2.7   # mag
    m_min = 1.0 # Msun
    m_max = 150 # Msun
    Z = 0.02
    magCut = 15.5

    verbose = 3

    ###############
    # Priors
    ###############
    # Distance to cluster
    dist_mean = 8.096  # kpc
    dist_std = 0.483   # kpc
    dist = pymc.TruncatedNormal('dist', dist_mean, 1.0/dist_std**2, a=6.793, b=9.510,
                                verbose=1)

    # Log Age of the cluster
    log_age_mean = 6.78
    log_age_std = 0.18
    logAgeCont = pymc.TruncatedNormal('logAgeCont', log_age_mean, 1.0/log_age_std**2,
                                      a=6.30, b=7.08, verbose=1)

    @pymc.deterministic(verbose=1)
    def logAge(logAgeCont=logAgeCont):
        "round age"
        return round(logAgeCont, 2)
        
    # Slope of the IMF
    alpha = pymc.Uniform('alpha', 0.15, 3.0, value=2.01,
                         verbose=1)

    # Total Cluster Mass
    Mcl = pymc.Uniform('Mcl', 3, 100, value=60,
                       verbose=1)

    ###############
    # Data
    ###############
    # Load up the data arrays.
    if yng == None:
        yng = lu_gc_imf.load_yng_data_by_radius(rmin, rmax, magCut=magCut)
    comp = lu_gc_imf.load_image_completeness_by_radius(rmin, rmax)


    ###############
    # Useful Variables
    ###############
    @pymc.deterministic(trace=False)
    def mod_sims_no_mass(logAge=logAge, AKs=AKs, imfSlope=alpha,
                         minMass=1, maxMass=150):
        return fetch_model_from_sims_no_mass(logAge, AKs, imfSlope, minMass, maxMass)
        
    @pymc.deterministic(verbose=1)
    def sim_N_WR(mod_sims_no_mass=mod_sims_no_mass, clusterMass=Mcl, simClusterMass=1e6):
        return mod_sims_no_mass[0] * clusterMass*1000 / simClusterMass

    @pymc.deterministic(trace=False)
    def sim_k_bins_tmp(mod_sims_no_mass=mod_sims_no_mass, distance=dist, simClusterDist=8000):
        return mod_sims_no_mass[1] + 5.0 * np.log10(distance*1000 / simClusterDist)

    @pymc.deterministic(trace=False)
    def sim_k_idx(sim_k_bins_tmp=sim_k_bins_tmp, magCut=magCut):
        idx = np.where(sim_k_bins_tmp <= magCut)[0]
        return idx

    @pymc.deterministic(trace=False)
    def sim_k_bins(sim_k_bins_tmp=sim_k_bins_tmp, sim_k_idx=sim_k_idx):
        return sim_k_bins_tmp[sim_k_idx]

    @pymc.deterministic(trace=False)
    def sim_k_bin_widths(sim_k_bins=sim_k_bins):
        return np.diff(sim_k_bins)

    @pymc.deterministic(trace=False)
    def sim_k_bin_center(sim_k_bins=sim_k_bins, sim_k_bin_widths=sim_k_bin_widths):
        return sim_k_bins[:-1] + (sim_k_bin_widths/2.0)

    # Number of young stars in simulation
    @pymc.deterministic
    def N_yng_expected(mod_sims_no_mass=mod_sims_no_mass, clusterMass=Mcl, simClusterMass=1e6,
                       magCut=magCut, sim_k_bins_tmp=sim_k_bins_tmp, sim_k_idx=sim_k_idx):
        sim_k_bin_widths = np.diff(sim_k_bins_tmp)
        sim_k_pdf = mod_sims_no_mass[2] * clusterMass*1000 / simClusterMass

        N_yng_sim = sim_k_pdf[sim_k_idx[:-1]].sum()
        bb = sim_k_idx[-1] # Take only a part of the last bin 
        N_yng_sim += sim_k_pdf[bb] * (magCut - sim_k_bins_tmp[bb]) / sim_k_bin_widths[bb]

        return N_yng_sim

    @pymc.deterministic(trace=False)
    def sim_k_pdf_norm(mod_sims_no_mass=mod_sims_no_mass, sim_k_idx=sim_k_idx):
        tmp = mod_sims_no_mass[3][sim_k_idx[:-1]]
        tmp /= (tmp * sim_k_bin_widths.value).sum()
        return tmp

    @pymc.deterministic(trace=False)
    def comp_interp(completeness=comp):
        return interpolate.splrep(completeness.mag, completeness.comp, k=1, s=0)

    @pymc.deterministic(trace=False)
    def comp_at_kp_sim(comp_interp=comp_interp, sim_k_bin_center=sim_k_bin_center):
        # Multiply by the completeness curve (after re-sampling). And renormalize.
        comp_resamp = interpolate.splev(sim_k_bin_center, comp_interp)
        comp_resamp[comp_resamp < 0] = 0.0
        comp_resamp[comp_resamp > 1] = 1.0
        return comp_resamp

    @pymc.deterministic(trace=False)
    def comp_at_kp_obs(comp_interp=comp_interp, yng=yng):
        # Get the probability of detecting (completeness) at each star's Kp
        comp_at_kp = interpolate.splev(yng.kp, comp_interp)
        comp_at_kp[comp_at_kp < 0] = 0.0
        comp_at_kp[comp_at_kp > 1] = 1.0
        return comp_at_kp
    

    ###############
    # Likelihood terms
    ###############
    # Prob(N_WR | model)
    N_WR = pymc.Poisson('N_WR', mu=sim_N_WR, value=yng.N_WR, observed=True, verbose=1)

    # Number of young stars observed
    N_yng_obs = pymc.Poisson('N_yng_obs', mu=yng.prob.sum(), verbose=1)
    
    # Number of young stars simulated
    N_yng = pymc.Poisson('N_yng', mu=N_yng_expected, verbose=1)

    # Non detections
    @pymc.potential(verbose=verbose)
    def likely_non_detect(N_yng=N_yng, N_yng_obs=N_yng_obs, comp_at_kp_sim=comp_at_kp_sim,
                          sim_k_bin_widths=sim_k_bin_widths, sim_k_pdf_norm=sim_k_pdf_norm):
        if N_yng < N_yng_obs:
            log_L_k_non_detect = -np.Inf
        else:
            tmp = (1.0 - comp_at_kp_sim) * sim_k_pdf_norm * sim_k_bin_widths
            P_non_detect = tmp.sum()
            if P_non_detect != 0:
                log_L_k_non_detect = (N_yng - N_yng_obs) * np.log(P_non_detect)
            else:
                log_L_k_non_detect = -np.Inf

        return log_L_k_non_detect

    # Detections
    @pymc.potential(verbose=verbose)
    def likely_detect(yng=yng, comp_at_kp_obs=comp_at_kp_obs, 
                      sim_k_bins=sim_k_bins, sim_k_pdf_norm=sim_k_pdf_norm,
                      sim_k_bin_widths=sim_k_bin_widths, sim_k_bin_center=sim_k_bin_center):

        # Detections
        log_L_k_detect = 1.0

        # Loop through each star and calc prob of detecting.
        for ii in range(len(yng.kp)):
            # Make gaussian around observed k mag
            obs_k_norm = scipy.stats.norm(loc=yng.kp[ii], scale=yng.kp_err[ii])
            obs_k_norm_cdf = obs_k_norm.cdf(sim_k_bins)
            obs_k_norm_pdf_binned = np.diff(obs_k_norm_cdf)

            # Convolve gaussian with P(KLF) from model
            L_k_i = (sim_k_pdf_norm * obs_k_norm_pdf_binned * sim_k_bin_widths).sum()

            # Multiply times P(detect)
            L_k_i *= comp_at_kp_obs[ii]

            # Multiply times P(yng)
            L_k_i *= yng.prob[ii]

            if L_k_i != 0:
                log_L_k_detect += np.log(L_k_i)
            else:
                log_L_k_detect = -np.Inf
                break

        return log_L_k_detect

    # Binomial Coefficient
    @pymc.potential(verbose=verbose)
    def likely_binomial(yng=yng, N_yng=N_yng, N_yng_obs=N_yng_obs):
        print 'N_yng = %3d, N_yng_obs = %3d' % (N_yng, N_yng_obs)
        if N_yng_obs >= N_yng:
            log_binom_coeff = -np.Inf
        else:
            log_binom_coeff = scipy.special.gammaln(N_yng + 1)
            log_binom_coeff -= scipy.special.gammaln(N_yng_obs + 1)
            log_binom_coeff -= scipy.special.gammaln(N_yng - N_yng_obs + 1)

        return log_binom_coeff

    return vars()
        
def pymc_model3(yng=None, rmin=0, rmax=30):
    # Set up variables (with priors)
    AKs = 2.7   # mag
    m_min = 1.0 # Msun
    m_max = 150 # Msun
    Z = 0.02
    magCut = 15.5

    verbose=3

    # Distance to cluster
    dist_mean = 8096  # pc
    dist_std = 483    # pc
    dist = pymc.TruncatedNormal('dist', dist_mean, 1.0/dist_std**2, a=6793.0, b=9510.0,
                                verbose=1)

    # Age of the cluster
    age_mean = 6e6   # Myr
    age_std = 2e6    # Myr
    age = pymc.TruncatedNormal('age', age_mean, 1.0/age_std**2, a=2.0e6, b=1.2e7,
                               verbose=1)

    @pymc.deterministic(verbose=verbose)
    def logAge(age=age):
        " convert and round age"
        return round(np.log10(age), 2)
        
    # Slope of the IMF
    alpha = pymc.Uniform('alpha', 0.15, 3.0, verbose=1)

    # Total Cluster Mass
    Mcl = pymc.Uniform('Mcl', 3e3, 1e5, verbose=1)

    # Load up the data arrays.
    if yng == None:
        yng = lu_gc_imf.load_yng_data_by_radius(rmin, rmax, magCut=magCut)
    comp = lu_gc_imf.load_image_completeness_by_radius(rmin, rmax)

    # Setup the Likelihood
    @pymc.potential(verbose=verbose)
    def likelihood(yngData=yng, completeness=comp, 
                   logAge=logAge, AKs=AKs, distance=dist,
                   imfSlope=alpha, clusterMass=Mcl,
                   minMass=m_min, maxMass=m_max, magCut=magCut):
        print '    logAge      = %.2f' % logAge
        print '    AKs         = %.2f' % AKs
        print '    distance    = %d' % dist
        print '    imfSlope    = %.2f' % alpha
        print '    clusterMass = %.2e' % Mcl
        print '    minMass     = %.1f' % m_min
        print '    maxMass     = %d' % m_max

        # Get the PDF(k|model) -- the simulated luminosity function
        mod_sims = fetch_model_from_sims(logAge, AKs, distance,
                                         imfSlope, clusterMass,
                                         minMass, maxMass)

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

        # Prob(N_WR | model)
        log_L_N_WR = pymc.poisson_like(yng.N_WR, sim_N_WR)

        # Prob(N_kp | model)
        idx = np.where(sim_k_bins <= magCut)[0]
        N_yng_sim = sim_k_pdf[idx[:-1]].sum()
        bb = idx[-1] # Take only a part of the last bin (depending on where the magCut falls)
        N_yng_sim += sim_k_pdf[bb] * (magCut - sim_k_bins[bb]) / sim_k_bin_widths[bb]
        log_L_N_yng = pymc.poisson_like(yng.prob.sum(), N_yng_sim)

        ##############################
        #
        # KLF shape is handled directly with the normalized Kp PDF
        # from the simulations. We account for errors by convolving
        # each measurement gaussian with the PDF(Kp).
        #
        ##############################
        log_L_k_detect = 1.0

        # Loop through each star and calc prob of detecting.
        for ii in range(len(yng.kp)):
            # Make gaussian around observed k mag
            obs_k_norm = scipy.stats.norm(loc=yng.kp[ii], scale=yng.kp_err[ii])
            obs_k_norm_cdf = obs_k_norm.cdf(sim_k_bins)
            obs_k_norm_pdf_binned = np.diff(obs_k_norm_cdf)

            # Convolve gaussian with PDF(K) from model
            L_k_i = (sim_k_pdf_norm * obs_k_norm_pdf_binned * sim_k_bin_widths).sum()

            # Multiply times P(yng)
            L_k_i *= yng.prob[ii]

            if L_k_i != 0:
                log_L_k_detect += np.log(L_k_i)
            else:
                log_L_k_detect = -np.Inf
                break


        log_L = log_L_N_yng + log_L_k_detect + log_L_N_WR

        return log_L

    return vars()

def pymc_model4(yng=None, rmin=0, rmax=30):
    # Set up variables (with priors)
    AKs = 2.7   # mag
    m_min = 1.0 # Msun
    m_max = 150 # Msun
    Z = 0.02
    magCut = 15.5

    klf_mag_bins = np.arange(9.0, 17.0, 1.0)

    verbose=3

    # Distance to cluster
    dist_mean = 8.096  # kpc
    dist_std = 0.483   # kpc
    dist = pymc.TruncatedNormal('dist', dist_mean, 1.0/dist_std**2, a=6.793, b=9.510,
                                verbose=1)

    # Log Age of the cluster
    log_age_mean = 6.78
    log_age_std = 0.18
    logAgeCont = pymc.TruncatedNormal('logAgeCont', log_age_mean, 1.0/log_age_std**2,
                                      a=6.30, b=7.08, verbose=1)

    @pymc.deterministic()
    def logAge(logAgeCont=logAgeCont):
        "round age"
        return round(logAgeCont, 2)
        
    # Slope of the IMF
    alpha = pymc.Uniform('alpha', 0.15, 3.0, verbose=1)

    # Total Cluster Mass (in x10^3 Msun)
    Mcl = pymc.Uniform('Mcl', 3, 100, verbose=1)

    # Load up the data arrays.
    if yng == None:
        yng = lu_gc_imf.load_yng_data_by_radius(rmin, rmax, magCut=magCut)
    comp = lu_gc_imf.load_image_completeness_by_radius(rmin, rmax)

    @pymc.deterministic(trace=False)
    def sim_model(yng=yng, completeness=comp, logAge=logAge, AKs=AKs, distance=dist,
                  imfSlope=alpha, clusterMass=Mcl, minMass=m_min, maxMass=m_max,
                  magCut=magCut):

        print '    logAge             = %.2f' % logAge
        print '    AKs (mag)          = %.2f' % AKs
        print '    distance (kpc)     = %.2f' % dist
        print '    imfSlope           = %.2f' % alpha
        print '    Mcl (x10^3 Msun)   = %.2f' % Mcl
        print '    minMass (Msun)     = %.1f' % m_min
        print '    maxMass (Msun)     = %d' % m_max

        # Get the PDF(k|model) -- the simulated luminosity function
        mod_sims = fetch_model_from_sims(logAge, AKs, distance*10**3,
                                         imfSlope, clusterMass*10**3,
                                         minMass, maxMass)

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

        # def unbin(k_pdf):
        #     n_ints = np.array([int(round(k_pdf_n)) for k_pdf_n in k_pdf])

        #     return np.repeat(sim_k_bin_center, n_ints)

        # py.figure(1)
        # py.clf()
        # py.hist(yng.kp, bins=klf_mag_bins, histtype='step', label='Obs', color='black',
        #         linewidth=2, weights=yng.prob)
        # py.hist(unbin(sim_k_pdf), bins=klf_mag_bins, histtype='step',
        #         label='Sim, Comp', color='red')

        sim_k_pdf *= comp_resamp
        sim_k_pdf_norm *= comp_resamp

        # Trim down to max(magCut + 3*yng.kp_err)
        #magLim = (magCut + 3.0*yng.kp_err).max()
        magLim = magCut
        idx = np.where(sim_k_bins <= magLim)[0]

        sim_k_bins = np.append(sim_k_bins[idx], sim_k_bins[idx[-1]+1])
        sim_k_bin_center = sim_k_bin_center[idx]
        sim_k_bin_widths = sim_k_bin_widths[idx]
        sim_k_pdf = sim_k_pdf[idx]
        sim_k_pdf_norm = sim_k_pdf_norm[idx]

        sim_k_pdf_norm /= (sim_k_pdf_norm * sim_k_bin_widths).sum()


        # py.hist(unbin(sim_k_pdf), bins=klf_mag_bins, histtype='step',
        #         label='Sim, Incomp', color='blue')
        # py.plot([8.5], [sim_N_WR], 'bs')
        # py.plot([8.5], [yng.N_WR], 'ko')
        # py.xlabel('Kp Magnitude')
        # py.ylabel('Number of Stars')
        # py.title(r't=%.2f d=%.2f $\alpha$=%.2f M=%.1e' %
        #          (logAge, distance, imfSlope, clusterMass))
        # py.legend(loc='upper left')
        # py.show()

        # py.figure(2)
        # py.clf()
        # py.hist(yng.kp, bins=klf_mag_bins, histtype='step', label='Obs', color='black',
        #         linewidth=2, weights=yng.prob, normed=True)
        # py.hist(unbin(sim_k_pdf), bins=klf_mag_bins, histtype='step',
        #         label='Sim, Incomp', color='blue', normed=True)
        # py.xlabel('Kp Magnitude')
        # py.ylabel('Number of Stars')
        # py.title(r't=%.2f d=%.2f $\alpha$=%.2f M=%.1e' %
        #          (logAge, distance, imfSlope, clusterMass))
        # py.legend(loc='upper left')
        # py.ylim(0, 1)
        # py.show()

        return (sim_N_WR, sim_k_bins, sim_k_pdf, sim_k_pdf_norm)

    # This is here for book keeping purposes.
    @pymc.deterministic(verbose=1)
    def sim_N_WR(yng=yng, sim_model=sim_model):
        print 'sim_N_WR = %3d vs. %3d observed' % (sim_model[0], yng.N_WR)
        return sim_model[0]


    ###############
    # Likelihood
    ###############
    # First term in the likelihood
    N_WR = pymc.Poisson('N_WR', mu=sim_N_WR, value=yng.N_WR, observed=True, verbose=1)

    # Second term in the likelihood
    N_yng_obs = pymc.Poisson('N_yng_obs', mu=yng.prob.sum(), verbose=1)

    @pymc.deterministic(verbose=1)
    def sim_N_yng(sim_model=sim_model, magCut=magCut, N_yng_obs=N_yng_obs):
        # Prob(N_kp | model)
        sim_k_bins = sim_model[1]
        sim_k_pdf = sim_model[2]
        sim_k_bin_widths = np.diff(sim_k_bins)
        
        idx = np.where(sim_k_bins <= magCut)[0]
        N_yng_sim = sim_k_pdf[idx[:-1]].sum()
        bb = idx[-1] # Take only a part of the last bin (depending on where the magCut falls)
        N_yng_sim += sim_k_pdf[bb] * (magCut - sim_k_bins[bb]) / sim_k_bin_widths[bb]

        print 'sim_N_yng = %3d vs. %d observed' % (N_yng_sim, N_yng_obs)

        return N_yng_sim

    N_yng = pymc.Poisson('N_yng', mu=sim_N_yng, value=N_yng_obs, verbose=1)
        
    # Third term in the likelihood
    @pymc.potential(verbose=verbose)
    def likely_kp_dist(yng=yng, sim_model=sim_model):
        sim_k_bins = sim_model[1]
        sim_k_pdf_norm = sim_model[3]
        sim_k_bin_widths = np.diff(sim_k_bins)
        
        ##############################
        #
        # KLF shape is handled directly with the normalized Kp PDF
        # from the simulations. We account for errors by convolving
        # each measurement gaussian with the PDF(Kp).
        #
        ##############################
        log_L_k_detect = 0.0

        # Loop through each star and calc prob of detecting.
        tmp = []
        for ii in range(len(yng.kp)):
            # Make gaussian around observed k mag
            obs_k_norm = scipy.stats.norm(loc=yng.kp[ii], scale=yng.kp_err[ii])
            obs_k_norm_cdf = obs_k_norm.cdf(sim_k_bins)
            obs_k_norm_pdf_binned = np.diff(obs_k_norm_cdf)

            if obs_k_norm_pdf_binned.sum() == 0:
                L_k_i = 0
            else:
                obs_k_norm_pdf_binned /= (obs_k_norm_pdf_binned * sim_k_bin_widths).sum()

                # Multiple gaussian with PDF(K) from model and sum to get probability
                L_k_i = (sim_k_pdf_norm * obs_k_norm_pdf_binned * sim_k_bin_widths).sum()

            if L_k_i != 0:
                log_L_k_i = np.log(L_k_i)
            else:
                log_L_k_i = -np.Inf

            log_L_yng_i = np.log(yng.prob[ii])

            log_L_k_detect += log_L_k_i + log_L_yng_i

            tmp.append(log_L_k_i + log_L_yng_i)

            # print '%2d  Kp = %5.2f +/- %4.2f  log_L_y_i = %5.1f  log_L_k_i = %5.1f  log_L_k = %8.1f'  % \
            #     (ii, yng.kp[ii], yng.kp_err[ii], log_L_yng_i, log_L_k_i, log_L_k_detect)


        py.figure(3)
        py.clf()
        py.plot(yng.kp, tmp, 'k.')
        py.xlabel('Kp mag')
        py.title('%.1f' % log_L_k_detect)
        py.show()
        return log_L_k_detect

    return vars()


def sim_to_obs_klf(cluster, magCut=15.5, withErrors=True, yng_orig=None):
    # Load up the original young star data to use the observed distribution
    # of Kp errors for our simulated population.
    if yng_orig == None:
        yng_orig = lu_gc_imf.load_yng_data_by_radius(magCut=magCut)

    # Load up imaging completness curve
    completeness = lu_gc_imf.load_image_completeness_by_radius()

    starCount = len(cluster.mag_noWR)

    if withErrors:
        # Randomly assign error bars from the observed kp error distribution.
        # Use the same random array as before as detectability and magnitude error
        # are not correlated.
        rand_number = np.random.rand(starCount)
        rand_kp_err_idx = rand_number * (len(yng_orig.kp_err)-1)
        rand_kp_err_idx = np.array(np.round(rand_kp_err_idx), dtype=int)
        kp_err = yng_orig.kp_err[rand_kp_err_idx]

        # Perturb the magnitudes by their photometric errors
        noise = np.random.randn(starCount) * kp_err
        kp = cluster.mag_noWR + noise
    else:
        kp = cluster.mag_noWR
        kp_err = np.zeros(len(kp), dtype=float)

    # Determine the probability of detection (in images) for each star
    Kp_interp = interpolate.splrep(completeness.mag, completeness.comp, k=1, s=0)
    comp_for_stars = interpolate.splev(cluster.mag_noWR, Kp_interp)
    comp_for_stars[comp_for_stars > 1] = 1.0
    comp_for_stars[comp_for_stars < 0] = 0.0

    # Randomly select detected stars based on the imaging completeness curve
    rand_number = np.random.rand(starCount)
    detected = np.where((rand_number < comp_for_stars) & (kp <= magCut))[0]

    yng_sim = objects.DataHolder()
    yng_sim.kp = kp[detected]
    yng_sim.kp_err = kp_err[detected]
    yng_sim.N_WR = cluster.num_WR
    yng_sim.mass = cluster.mass[cluster.idx_noWR][detected]

    # We know all these stars are young. In principle, we could throw in
    # a contaminating population, but lets not do that for now.
    yng_sim.prob = np.ones(len(detected))

    return yng_sim

def simulated_klf(logAge, AKs, distance, imfSlope, clusterMass, minMass=1, maxMass=150,
                  magCut=15.5, makeMultiples=True, withErrors=True,
                  MFamp=defaultMFamp, MFindex=defaultMFindex,
                  CSFamp=defaultCSFamp, CSFindex=defaultCSFindex, CSFmax=defaultCSFmax):
    """
    Simulate a cluster similar to that at the Galactic Center, which is useful 
    for determining how well we can recover the input parameters.
    """
    # Get a simulated cluster
    cluster = model_young_cluster(logAge, AKs=AKs, imfSlope=imfSlope,
                                  distance=distance,
                                  clusterMass=clusterMass,
                                  minMass=minMass, maxMass=maxMass,
                                  makeMultiples=makeMultiples, MFamp=MFamp, MFindex=MFindex,
                                  CSFamp=CSFamp, CSFindex=CSFindex, CSFmax=CSFmax)

    return sim_to_obs_klf(cluster, magCut=magCut, withErrors=withErrors)

def scale_model_from_sims(sim_model_no_mass, distance, clusterMass,
                         simClusterMass=1e6, simClusterDist=8000):
    sim_N_WR = sim_model_no_mass[0]
    sim_k_bins = sim_model_no_mass[1]
    sim_k_pdf = sim_model_no_mass[2]
    sim_k_pdf_norm = sim_model_no_mass[3]

    N_WR = sim_N_WR * clusterMass / simClusterMass
    k_bins = sim_k_bins + 5.0 * np.log10(distance / simClusterDist)
    k_pdf = sim_k_pdf * clusterMass / simClusterMass
    k_pdf_norm = sim_k_pdf_norm

    return N_WR, k_bins, k_pdf, k_pdf_norm

def fetch_model_from_sims_no_mass(logAge, AKs, imfSlope, minMass, maxMass,
                                  clusterMass=1e6, distance=8000, makeMultiples=True,
                                  MFamp=defaultMFamp, MFindex=defaultMFindex,
                                  CSFamp=defaultCSFamp, CSFindex=defaultCSFindex,
                                  CSFmax=defaultCSFmax, filterName=defaultFilter):
    if makeMultiples:
        multi_str = 'multi'
    else:
        multi_str = 'single'

    # First see if we have this set of parameters already processed.
    # Recall that the simulations can be scaled linearly with
    # cluster mass and distance... use some default values and then
    # scale appropriately to avoid needlessly re-running.
    modelDir = '/u/jlu/work/gc/imf/klf/models/clusters/'
    modelFile = 'sim_t%.2f_a%.2f_m%d_aks%.2f_d%d_minM%.1f_maxM%d_%s.pickle' % \
        (logAge, imfSlope, clusterMass, AKs, distance, minMass, maxMass, multi_str)

    if os.path.exists(modelDir + modelFile):
        try:
            # print 'Using pre-existing model: %s' % modelFile
            _sim = open(modelDir + modelFile, 'rb')
            sim_N_WR = pickle.load(_sim)
            sim_k_bins = pickle.load(_sim)
            sim_k_pdf = pickle.load(_sim)
            sim_k_pdf_norm = pickle.load(_sim)
        except EOFError:
            print 'Bad file: ', modelFile
            raise

        # Summary parameters
        tmp_logAge = round(pickle.load(_sim), 2)
        tmp_filt = pickle.load(_sim)
        tmp_AKs = round(pickle.load(_sim), 2)
        tmp_distance = pickle.load(_sim)
        tmp_imfSlope = round(pickle.load(_sim), 2)
        tmp_sumIMFmass = pickle.load(_sim)
        tmp_minIMFmass = pickle.load(_sim)
        tmp_maxIMFmass = pickle.load(_sim)
        tmp_makeMultiples = pickle.load(_sim)
        tmp_MFamp = pickle.load(_sim)
        tmp_MFindex = pickle.load(_sim)
        tmp_CSFamp = pickle.load(_sim)
        tmp_CSFindex = pickle.load(_sim)
        tmp_CSFmax = pickle.load(_sim)

        _sim.close()

        if ((logAge != tmp_logAge) or (AKs != tmp_AKs) or (distance != tmp_distance) or
            (round(imfSlope, 2) != round(tmp_imfSlope, 2)) or
            (clusterMass != tmp_sumIMFmass) or (minMass != tmp_minIMFmass) or
            (maxMass != tmp_maxIMFmass) or (MFamp != tmp_MFamp) or (MFindex != tmp_MFindex) or
            (CSFamp != tmp_CSFamp) or (CSFindex != tmp_CSFindex) or (CSFmax != tmp_CSFmax)):

            print 'fetch_model_from_sims_no_mass: You need to re-generate this cluster.'
            print ' logAge       Need %10.2f, found %10.2f' % (logAge, tmp_logAge)
            print ' filter       Need %10s, found %10s' % (filterName, tmp_filt)
            print ' AKs          Need %10.2f, found %10.2f' % (AKs, tmp_AKs)
            print ' distance     Need %10d, found %10d' % (distance, tmp_distance)
            print ' imfSlope     Need %10.2f, found %10.2f' % (imfSlope, tmp_imfSlope)
            print ' clusterMass  Need %10d, found %10d' % (clusterMass, tmp_sumIMFmass)
            print ' minMass      Need %10.3f, found %10.3f' % (minMass, tmp_minIMFmass)
            print ' maxMass      Need %10.3f, found %10.3f' % (maxMass, tmp_maxIMFmass)
            print ' multiples    Need %10s, found %10s' % (str(makeMultiples), str(tmp_makeMultiples))
            print ' MFamp        Need %10.2f, found %10.2f' % (MFamp, tmp_MFamp)
            print ' MFindex      Need %10.2f, found %10.2f' % (MFindex, tmp_MFindex)
            print ' CSFamp       Need %10.2f, found %10.2f' % (CSFamp, tmp_CSFamp)
            print ' CSFindex     Need %10.2f, found %10.2f' % (CSFindex, tmp_CSFindex)
            print ' CSFmax       Need %10.2f, found %10.2f' % (CSFmax, tmp_CSFmax)
            pdb.set_trace()
    else:
        cluster = model_young_cluster(logAge, AKs=AKs, 
                                      distance=distance, imfSlope=imfSlope,
                                      clusterMass=clusterMass,
                                      minMass=minMass, maxMass=maxMass,
                                      makeMultiples=makeMultiples, MFamp=MFamp, MFindex=MFindex,
                                      CSFamp=CSFamp, CSFindex=CSFindex, CSFmax=CSFmax)

        sim_N_WR = cluster.num_WR
        sim_k_bins = np.arange(cluster.mag_noWR.min(), cluster.mag_noWR.max()+0.1, 0.1)
        sim_k_pdf, sim_k_bins = np.histogram(cluster.mag_noWR, bins=sim_k_bins, normed=False)
        sim_k_pdf_norm, sim_k_bins = np.histogram(cluster.mag_noWR, bins=sim_k_bins, normed=True)

        # Save to file
        _sim = open(modelDir + modelFile, 'wb')
        pickle.dump(sim_N_WR, _sim)
        pickle.dump(sim_k_bins, _sim)
        pickle.dump(sim_k_pdf, _sim)
        pickle.dump(sim_k_pdf_norm, _sim)

        # Summary parameters
        pickle.dump(cluster.logAge, _sim)
        pickle.dump(cluster.filter, _sim)
        pickle.dump(cluster.AKs, _sim)
        pickle.dump(cluster.distance, _sim)
        pickle.dump(cluster.imfSlope, _sim)
        pickle.dump(cluster.sumIMFmass, _sim)
        pickle.dump(cluster.minIMFmass, _sim)
        pickle.dump(cluster.maxIMFmass, _sim)
        pickle.dump(cluster.makeMultiples, _sim)
        pickle.dump(cluster.MFamp, _sim)
        pickle.dump(cluster.MFindex, _sim)
        pickle.dump(cluster.CSFamp, _sim)
        pickle.dump(cluster.CSFindex, _sim)
        pickle.dump(cluster.CSFmax, _sim)
        pickle.dump(cluster.qMin, _sim)
        pickle.dump(cluster.qIndex, _sim)

        _sim.close()

    return sim_N_WR, sim_k_bins, sim_k_pdf, sim_k_pdf_norm


def fetch_model_from_sims(logAge, AKs, distance, imfSlope,
                          clusterMass, minMass, maxMass,
                          makeMultiples=True, MFamp=defaultMFamp, MFindex=defaultMFindex,
                          CSFamp=defaultCSFamp, CSFindex=defaultCSFindex,
                          CSFmax=defaultCSFmax, filterName=defaultFilter):
    mod_sim = fetch_model_from_sims_no_mass(logAge, AKs, imfSlope, minMass, maxMass,
                                            makeMultiples=makeMultiples,
                                            MFamp=MFamp, MFindex=MFindex,
                                            CSFamp=CSFamp, CSFindex=CSFindex, CSFmax=CSFmax,
                                            filterName=defaultFilter)
    return scale_model_from_sims(mod_sim, distance, clusterMass)

def pre_make_observed_isochrones():
#    logAgeArray = np.arange(5.90, 7.51, 0.01)
    logAgeArray = np.arange(6.80, 7.51, 0.01)

    simClusterMass = 1e6
    simClusterDist = 8000

    for logAge in logAgeArray:
        # will make the isochrone if necessary
        load_isochrone(logAge, AKs=2.7, distance=simClusterDist)
    

def model_young_cluster(logAge, filterName=defaultFilter,
                        AKs=defaultAKs, distance=defaultDist,
                        imfSlope=2.35, clusterMass=10**4,
                        minMass=1, maxMass=150,
                        makeMultiples=True, MFamp=defaultMFamp, MFindex=defaultMFindex,
                        CSFamp=defaultCSFamp, CSFindex=defaultCSFindex,
                        CSFmax=defaultCSFmax,
                        qMin=0.01, qIndex=-0.4, verbose=False):
    c = constants

    logAgeString = '0%d' % (int(logAge * 100))

    iso = load_isochrone(logAge=logAge, filterName=filterName,
                         AKs=AKs, distance=distance)

    # Sample a power-law IMF randomly
    mass, isMultiple, compMasses  = sample_imf(clusterMass, minMass, maxMass, imfSlope,
                                               makeMultiples=makeMultiples,
                                               multiMFamp=MFamp, multiMFindex=MFindex,
                                               multiCSFamp=CSFamp, multiCSFindex=CSFindex,
                                               multiCSFmax=CSFmax,
                                               multiQmin=qMin, multiQindex=qIndex)

    mag = np.zeros(len(mass), dtype=float)
    temp = np.zeros(len(mass), dtype=float)
    isWR = np.zeros(len(mass), dtype=bool)

    def match_model_mass(theMass):
        dm = np.abs(iso.M - theMass)
        mdx = dm.argmin()

        # Model mass has to be within 2% of the desired mass
        if dm[mdx] / theMass > 0.02:
            return None
        else:
            return mdx
        
    for ii in range(len(mass)):
        # Find the closest model mass (returns None, if nothing with dm = 0.1
        mdx = match_model_mass(mass[ii])
        if mdx == None:
            continue

        mag[ii] = iso.mag[mdx]
        temp[ii] = iso.T[mdx]
        isWR[ii] = iso.isWR[mdx]

        # Determine if this system is a binary.
        if isMultiple[ii]:
            n_stars = len(compMasses[ii])
            for cc in range(n_stars):
                mdx_cc = match_model_mass(compMasses[ii][cc])
                if mdx_cc != None:
                    f1 = 10**(-mag[ii]/2.5)
                    f2 = 10**(-iso.mag[mdx_cc]/2.5)
                    mag[ii] = -2.5 * np.log10(f1 + f2)
                else:
                    print 'Rejected a companion %.2f' % compMasses[ii][cc]
        

    # Get rid of the bad ones
    idx = np.where(temp != 0)[0]
    cdx = np.where(temp == 0)[0]

    if len(cdx) > 0 and verbose:
        print 'Found %d stars out of mass range: Minimum bad mass = %.1f' % \
            (len(cdx), mass[cdx].min())

    mass = mass[idx]
    mag = mag[idx]
    temp = temp[idx]
    isWR = isWR[idx]
    isMultiple = isMultiple[idx]
    if makeMultiples:
        compMasses = [compMasses[ii] for ii in idx]

    idx_noWR = np.where(isWR == False)[0]

    mag_noWR = mag[idx_noWR]
    num_WR = len(mag) - len(idx_noWR)

    cluster = objects.DataHolder()
    cluster.mass = mass
    cluster.Teff = temp
    cluster.isWR = isWR
    cluster.mag = mag
    cluster.isMultiple = isMultiple
    cluster.compMasses = compMasses

    cluster.idx_noWR = idx_noWR
    cluster.mag_noWR = mag_noWR
    cluster.num_WR = num_WR

    # Summary parameters
    cluster.logAge = logAge
    cluster.filter = filterName
    cluster.AKs = AKs
    cluster.distance = distance
    cluster.imfSlope = imfSlope
    cluster.sumIMFmass = clusterMass
    cluster.minIMFmass = minMass
    cluster.maxIMFmass = maxMass
    cluster.makeMultiples = makeMultiples
    cluster.MFamp = MFamp
    cluster.MFindex = MFindex
    cluster.CSFamp = CSFamp
    cluster.CSFindex = CSFindex
    cluster.CSFmax = CSFmax
    cluster.qMin = qMin
    cluster.qIndex = qIndex
            
    return cluster

def make_observed_isochrone(logAge, filterName=defaultFilter,
                            AKs=defaultAKs, distance=defaultDist, verbose=False):
    startTime = time.time()

    print 'Making isochrone: log(t) = %.2f  filt = %s  AKs = %.2f  dist = %d' % \
        (logAge, filterName, AKs, distance)
    print '     Starting at: ', datetime.datetime.now(), '  Usually takes ~5 minutes'

    outFile = '/u/jlu/work/gc/imf/klf/models/iso/'
    outFile += 'iso_%.2f_%s_%4.2f_%4s.pickle' % (logAge, filterName, AKs,
                                                 str(distance).zfill(4))

    c = constants

    # Get solar mettalicity models for a population at a specific age.
    evol = evolution.get_merged_isochrone(logAge=logAge)

    # Lets do some trimming down to get rid of repeat masses or 
    # mass resolutions higher than 1/1000. We will just use the first
    # unique mass after rounding by the nearest 0.001.
    mass_rnd = np.round(evol.mass, decimals=2)
    tmp, idx = np.unique(mass_rnd, return_index=True)

    mass = evol.mass[idx]
    logT = evol.logT[idx]
    logg = evol.logg[idx]
    logL = evol.logL[idx]
    isWR = logT != evol.logT_WR[idx]

    temp = 10**logT

    # Output magnitudes for each temperature and extinction value.
    mag = np.zeros(len(temp), dtype=float)

    filt = synthetic.filters[filterName]
    flux0 = synthetic.filter_flux0[filterName]
    mag0 = synthetic.filter_mag0[filterName]

    # Make reddening
    red = synthetic.redlaw.reddening(AKs).resample(filt.wave)

    # Convert luminosity to erg/s
    L_all = 10**(logL) * c.Lsun # luminsoity in erg/s

    # Calculate radius
    R_all = np.sqrt(L_all / (4.0 * math.pi * c.sigma * temp**4))
    R_all /= (c.cm_in_AU * c.AU_in_pc)

    # For each temperature extract the synthetic photometry.
    for ii in range(len(temp)):
        gravity = logg[ii]
        L = L_all[ii] # in erg/s
        T = temp[ii]  # in Kelvin
        R = R_all[ii] # in pc

        # Get the atmosphere model now. Wavelength is in Angstroms
        star = atmospheres.get_merged_atmosphere(temperature=T, 
                                                  gravity=gravity)

        # Trim wavelength range down to JHKL range (1.0 - 4.25 microns)
        star = spectrum.trimSpectrum(star, 10000, 42500)

        # Convert into flux observed at Earth (unreddened)
        star *= (R / distance)**2  # in erg s^-1 cm^-2 A^-1

        # ----------
        # Now to the filter integrations
        # ----------
        mag[ii] = synthetic.mag_in_filter(star, filt, red, flux0, mag0)

        if verbose:
            print 'M = %7.3f Msun   T = %5d K   R = %2.1f Rsun   logg = %4.2f   mag = %4.2f' % \
                (mass[ii], T, R * c.AU_in_pc / c.Rsun, logg[ii], mag[ii])


    iso = objects.DataHolder()
    iso.M = mass
    iso.T = temp
    iso.logg = logg
    iso.logL = logL
    iso.mag = mag
    iso.isWR = isWR
    
    _out = open(outFile, 'wb')
    pickle.dump(mass, _out)
    pickle.dump(temp, _out)
    pickle.dump(logg, _out)
    pickle.dump(logL, _out)
    pickle.dump(mag, _out)
    pickle.dump(isWR, _out)
    _out.close()

    endTime = time.time()
    print '      Time taken: %d seconds' % (endTime - startTime)

def load_isochrone(logAge=6.78, filterName=defaultFilter,
                   AKs=defaultAKs, distance=defaultDist):
    inFile = '/u/jlu/work/gc/imf/klf/models/iso/'
    inFile += 'iso_%.2f_%s_%4.2f_%4s.pickle' % (logAge, filterName, AKs,
                                                 str(int(distance)).zfill(4))

    if not os.path.exists(inFile):
        make_observed_isochrone(logAge=logAge, filterName=filterName,
                                AKs=AKs, distance=distance)

    _in = open(inFile, 'rb')
    iso = objects.DataHolder()
    iso.M = pickle.load(_in)
    iso.T = pickle.load(_in)
    iso.logg = pickle.load(_in)
    iso.logL = pickle.load(_in)
    iso.mag = pickle.load(_in)
    iso.isWR = pickle.load(_in)
    _in.close()

    return iso

def grid_of_isochrones():
    """
    Generate a grid of isochrones over the useful range of parameters that I will
    be exploring for the GC IMF. Eventually we may be using a MCMC, but the grid is
    useful for visualization and for testing.

    
    """
    ageGrid = np.arange(3e6, 9e6, 5e5)
    imfSlopeGrid = np.arange(0.25, 2.85, 0.1)
    clusterMass = np.array([7500, 10000, 25000, 50000, 75000, 10**5])

    print len(ageGrid)*len(imfSlopeGrid)*len(clusterMass)


    # logAge, filterName=defaultFilter,
    #                     AKs=defaultAKs, distance=defaultDist,
    #                     imfSlope=2.35, clusterMass=10**4, minMass=1.0,
    #                     kperr=None


def sample_imf(totalMass, minMass, maxMass, imfSlope,
               multiMFamp=defaultMFamp, multiMFindex=defaultMFindex,
               multiCSFamp=defaultCSFamp, multiCSFindex=defaultCSFindex,
               multiCSFmax=defaultCSFmax,
               makeMultiples=True, multiQindex=-0.4, multiQmin=0.01,
               verbose=False):
    """
    Randomly sample from an IMF of the specified slope and mass
    limits until the desired total mass is reached. The maximum
    stellar mass is not allowed to exceed the total cluster mass.
    The simulated total mass will not be exactly equivalent to the
    desired total mass; but we will take one star above or below
    (whichever brings us closer to the desired total) the desired
    total cluster mass break point.

    IMF Slope is 2.35 for Salpeter.
    """
    if (maxMass > totalMass) and verbose:
        print 'sample_imf: Setting maximum allowed mass to %d' % \
            (totalMass)
        maxMass = totalMass

    # p(m) = A * m^-imfSlope
    # Setup useful variables
    nG = 1 - imfSlope  # This is also -Gamma, hence nG name

    if imfSlope != 1:
        A =  nG / (maxMass**nG - minMass**nG)
    else:
        A = 1.0 / (math.log(maxMass) - math.log(minMass))

    # Generative function for primary masses
    def cdf_inv_not_1(x, minMass, maxMass, nG):
        return (x * (maxMass**nG - minMass**nG) + minMass**nG)**(1.0/nG)

    # This is the special case for alpha = 1.0
    def cdf_inv_1(x, minMass, maxMass, nG):
        return minMass * (maxMass / minMass)**x

    # Generative function for companion mass ratio (q = m_comp/m_primary)
    def q_cdf_inv(x, qLo, beta):
        b = 1.0 + beta
        return (x * (1.0 - qLo**b) + qLo**b)**(1.0/b)

    # First estimate the mean number of stars expected
    if imfSlope != 1:
        if imfSlope != 2: 
            nGp1 = 1 + nG
            meanMass = A * (maxMass**nGp1 - minMass**nGp1) / nGp1
        else:
            meanMass = A * (math.log(maxMass) - math.log(minMass))

        cdf_inv = cdf_inv_not_1
    else:
        meanMass = A * (maxMass - minMass)
        cdf_inv = cdf_inv_1

    meanNumber = round(totalMass / meanMass)

    simTotalMass = 0
    newStarCount = round(meanNumber)
    if not makeMultiples:
        newStarCount *= 1.1

    masses = np.array([], dtype=float)
    isMultiple = np.array([], dtype=bool)
    compMasses = []
    systemMasses = np.array([], dtype=float)

    def binary_properties(mass):
        # Multiplicity Fraction
        mf = multiMFamp * mass**multiMFindex
        mf[mf > 1] = 1

        # Companion Star Fraction
        csf = multiCSFamp * mass**multiCSFindex
        csf[csf > 3] = multiCSFmax

        return mf, csf

    loopCnt = 0

    while simTotalMass < totalMass:
        # Generate a random distribution 20% larger than
        # the number we expect to need.
        uniX = np.random.rand(newStarCount)

        # Convert into the IMF from the inverted CDF
        newMasses = cdf_inv(uniX, minMass, maxMass, nG)

        if makeMultiples:
            compMasses = [[] for ii in range(len(newMasses))]

            # Determine the multiplicity of every star
            MF, CSF = binary_properties(newMasses)
            newIsMultiple = np.random.rand(newStarCount) < MF
            newSystemMasses = newMasses.copy()
        
            # Calculate number and masses of companions
            for ii in range(len(newMasses)):
                if newIsMultiple[ii]:
                    n_comp = 1 + np.random.poisson((CSF[ii]/MF[ii]) - 1)
                    q_values = q_cdf_inv(np.random.rand(n_comp), multiQmin, multiQindex)
                    m_comp = q_values * newMasses[ii]

                    # Only keep companions that are more than the minimum mass
                    mdx = np.where(m_comp >= minMass)
                    compMasses[ii] = m_comp[mdx]
                    newSystemMasses[ii] += compMasses[ii].sum()

                    # Double check for the case when we drop all companions.
                    # This happens a lot near the minimum allowed mass.
                    if len(mdx) == 0:
                        newIsMultiple[ii] == False

            newSimTotalMass = newSystemMasses.sum()
            isMultiple = np.append(isMultiple, newIsMultiple)
            systemMasses = np.append(systemMasses, newSystemMasses)
        else:
            newSimTotalMass = newMasses.sum()

        # Append to our primary masses array
        masses = np.append(masses, newMasses)

        if (loopCnt >= 0) and verbose:
            print 'sample_imf: Loop %d added %.2e Msun to previous total of %.2e Msun' % \
                (loopCnt, newSimTotalMass, simTotalMass)

        simTotalMass += newSimTotalMass
        newStarCount = meanNumber * 0.1  # increase by 20% each pass
        loopCnt += 1
        
    # Make a running sum of the system masses
    if makeMultiples:
        massCumSum = systemMasses.cumsum()
    else:
        massCumSum = masses.cumsum()

    # Find the index where we are closest to the desired
    # total mass.
    idx = np.abs(massCumSum - totalMass).argmin()

    masses = masses[:idx+1]

    if makeMultiples:
        systemMasses = systemMasses[:idx+1]
        isMultiple = isMultiple[:idx+1]
        compMasses = compMasses[:idx+1]
    else:
        isMultiple = np.zeros(len(masses), dtype=bool)

    return (masses, isMultiple, compMasses)

def test_distributions():
    test_dist_generate()
    test_dist_plot()

def test_dist_generate():
    """
    Generate 100 clusters and examine the distributions of the parameters
    of interest.
    """
    Nclusters = 1000

    # Model parameters:
    # Use the default filter, extinction, distance, and minMass, maxMass
    clusterMass = 5e4
    clusterAge = round(math.log10(6e6), 2)  # log(t)
    clusterIMF = 2.35

    outRoot = '/u/jlu/work/gc/imf/klf/models/test_distributions/clusters/'
    outRoot += 'sim_t%.2f_a%.2f_m%d.pickle' % (clusterAge, clusterIMF, clusterMass)

    for ii in range(Nclusters):
        cluster = model_young_cluster(clusterAge,
                                      imfSlope=clusterIMF,
                                      clusterMass=clusterMass)

        # Get rid of some stuff to save space, we can always re-calc
        cluster.Teff = None
        cluster.idx_noWR = None

        f, fname = open_unique_file(outRoot, 'wb')
        pickle.dump(cluster, f)
        f.close()
        

def test_dist_plot():
    # Model parameters:
    # Use the default filter, extinction, distance, and minMass, maxMass
    clusterMass = 5e4
    clusterAge = round(math.log10(6e6), 2)  # log(t)
    clusterIMF = 2.35

    outRoot = '/u/jlu/work/gc/imf/klf/models/test_distributions/clusters/'
    outRoot += 'sim_t%.2f_a%.2f_m%d_' % (clusterAge, clusterIMF, clusterMass)

    files = glob.glob(outRoot + '*.pickle')

    Nclusters = len(files)

    # Ouptput values
    kp = np.array([], dtype=float)
    N_WR = np.array([], dtype=int)

    # One of the outputs will be the binned Kp PDF for each cluster.
    kpBins = np.arange(4, 20, 0.1)
    kpBinCenters = kpBins[:-1] + np.diff(kpBins)/2.0
    kp_pdf = np.zeros((Nclusters, len(kpBins)-1), dtype=float)
    
    for ii in range(Nclusters):
        print 'Cluster ', ii
        _f = open(files[ii], 'rb')
        cluster = pickle.load(_f)
        _f.close()

        #kp = np.append(kp, cluster.mag_noWR)
        N_WR = np.append(N_WR, cluster.num_WR)

        foo1, foo2 = np.histogram(cluster.mag_noWR, bins=kpBins, normed=True)
        kp_pdf[ii,:] = foo1

    outDir = '/u/jlu/work/gc/imf/klf/models/test_distributions/plots/'

    # First plot the PDF of the luminosity function:
    mean_kp_pdf = kp_pdf.mean(axis=0)
    std_kp_pdf = kp_pdf.std(axis=0)

    py.clf()
    py.plot(kpBinCenters, mean_kp_pdf, 'k-')
    py.fill_between(kpBinCenters,
                    mean_kp_pdf - std_kp_pdf,
                    mean_kp_pdf + std_kp_pdf,
                    color='grey', alpha=0.2)
    py.gca().set_yscale('log')
    py.xlabel('Kp Magnitude')
    py.ylabel('Probability Density (0.1 mag bins)')
    py.title('log(t)=%.2f alpha=%.2f M=%.0e Nclust=%d' % \
             (clusterAge, clusterIMF, clusterMass, Nclusters), fontsize=10)
    py.savefig(outDir + 'test_distribution_kp_pdf.png')


    # Plot up the distribution of the number of WR stars
    py.clf()
    binsWR = np.arange(N_WR.min(), N_WR.max()+1, 1)
    py.hist(N_WR, bins=binsWR, histtype='step', normed=True)

    # Compare to poisson distribution
    poisson = scipy.stats.poisson(N_WR.mean())
    foo = poisson.pmf(binsWR)
    py.plot(binsWR+0.5, foo, 'rx-')
    py.xlabel('Number of WR stars')
    py.ylabel('Number of clusters')
    py.title('log(t)=%.2f alpha=%.2f M=%.0e Nclust=%d' % \
             (clusterAge, clusterIMF, clusterMass, Nclusters), fontsize=10)
    py.savefig(outDir + 'test_distribution_nWR_pdf.png')


def test_dist_numWR():
    # Model parameters:
    # Use the default filter, extinction, distance, and minMass, maxMass
    clusterMass = [5e3, 1e4, 2e4, 4e4, 5e4]
    clusterAge = round(math.log10(6e6), 2)  # log(t)
    clusterIMF = 2.35

    Nclusters = 100

    for mm in range(len(clusterMass)):
        outRoot = '/u/jlu/work/gc/imf/klf/models/test_distributions/clusters/'
        outRoot += 'sim_t%.2f_a%.2f_m%d.pickle' % (clusterAge, clusterIMF, clusterMass[mm])

        print 'Generating %d clusters with M = %.1e' % (Nclusters, clusterMass[mm])

        for ii in range(Nclusters):
            cluster = model_young_cluster(clusterAge,
                                          imfSlope=clusterIMF,
                                          clusterMass=clusterMass[mm])

            # Get rid of some stuff to save space, we can always re-calc
            cluster.Teff = None
            cluster.idx_noWR = None

            f, fname = open_unique_file(outRoot, 'wb')
            pickle.dump(cluster, f)
            f.close()
        

def test_dist_plot_numWR():
    # Model parameters:
    # Use the default filter, extinction, distance, and minMass, maxMass
    clusterMass = [5e3, 1e4, 2e4, 4e4, 5e4]
    clusterAge = round(math.log10(6e6), 2)  # log(t)
    clusterIMF = 2.35

    Nclusters = 100

    # Keep track of the number of WR stars for each set
    # of simulations at a given cluster mass.
    N_WR_all = []

    for mm in range(len(clusterMass)):
        outRoot = '/u/jlu/work/gc/imf/klf/models/test_distributions/clusters/'
        outRoot += 'sim_t%.2f_a%.2f_m%d_' % (clusterAge, clusterIMF, clusterMass[mm])

        files = glob.glob(outRoot + '*.pickle')

        Nclusters = len(files)
        if Nclusters > 300:
            Nclusters = 300

        print 'Gathering %d clusters with M = %.2e Msun' % (Nclusters, clusterMass[mm])
        
        N_WR = np.zeros(Nclusters)
        N_WR_all.append(N_WR)

        # Sample smaller clusters (by pulling from our existing simulations
        for ii in range(Nclusters):
            _f = open(files[ii], 'rb')
            cluster = pickle.load(_f)
            _f.close()

            N_WR[ii] = cluster.num_WR


    # Now re-loop through and plot up the distributions
    outDir = '/u/jlu/work/gc/imf/klf/models/test_distributions/plots/'

    py.clf()
    colors = ['red', 'green', 'blue', 'orange', 'cyan']

    N_WR_mean = np.zeros(len(clusterMass), dtype=float)

    for mm in range(len(clusterMass)):
        legendLabel = '%.1e' % clusterMass[mm]
        binsWR = np.arange(N_WR_all[mm].min(), N_WR_all[mm].max()+1, 1)
        py.hist(N_WR_all[mm], bins=binsWR, histtype='step', normed=True,
                label=legendLabel, color=colors[mm])

        N_WR_mean[mm] = N_WR_all[mm].mean()
        # Compare to poisson distribution
        poisson = scipy.stats.poisson(N_WR_mean[mm])
        foo = poisson.pmf(binsWR)
        py.plot(binsWR+0.5, foo, linestyle='--', color=colors[mm])

        print 'M = %.1e  <N_WR> = %.1f' % (clusterMass[mm], N_WR_mean[mm])

    py.legend()
    py.xlabel('Number of WR stars')
    py.ylabel('Number of clusters')
    py.title('log(t)=%.2f alpha=%.2f Nclust=%d' % \
             (clusterAge, clusterIMF, Nclusters), fontsize=10)
    py.savefig(outDir + 'test_distribution_numWR_pdf_v_Mcl.png')

    py.clf()
    py.plot(clusterMass, N_WR_mean, 'ks', label='Simulations')
    lineFit = np.polyfit(clusterMass, N_WR_mean, 1)
    legendLabel = 'Line Fit: %.2f + %.4f * (Mcl/1e3)' % (lineFit[1], lineFit[0]*10**3)
    lineX = np.arange(0, 1e5, 2500)
    py.plot(lineX, np.polyval(lineFit, lineX), 'k-',
            label=legendLabel)
    py.xlabel('Cluster Mass (Msun)')
    py.ylabel('Mean Number of WR stars')
    py.title('log(t)=%.2f alpha=%.2f Nclust=%d' % \
             (clusterAge, clusterIMF, Nclusters), fontsize=10)
    py.xlim(0, 1e5)
    py.ylim(0, 30)
    legFont = matplotlib.font_manager.FontProperties(size=10)
    leg = py.legend(loc='upper left', prop=legFont)
    py.savefig(outDir + 'test_distribution_numWR_mean_v_Mcl.png')

def test_dist_kp():
    """
    Generate a 10^6 Msun cluster and a 10^7 Msun cluster and look
    at the resulting PDF(kp). Can we get away with using 10^6 or do
    we need to use 10^7?
    """
    # Model parameters:
    # Use the default filter, extinction, distance, and minMass, maxMass
    clusterMass = [1e6, 1e7]
    clusterAge = round(math.log10(6e6), 2)  # log(t)
    clusterIMF = 2.35

    for mm in range(len(clusterMass)):
        outRoot = '/u/jlu/work/gc/imf/klf/models/test_distributions/clusters/'
        outRoot += 'sim_t%.2f_a%.2f_m%d.pickle' % (clusterAge, clusterIMF, clusterMass[mm])

        print 'Generating %d clusters with M = %.1e' % (1, clusterMass[mm])

        cluster = model_young_cluster(clusterAge,
                                      imfSlope=clusterIMF,
                                      clusterMass=clusterMass[mm])

        # Get rid of some stuff to save space, we can always re-calc
        cluster.Teff = None
        cluster.idx_noWR = None

        f, fname = open_unique_file(outRoot, 'wb')
        pickle.dump(cluster, f)
        f.close()

def test_dist_plot_kp():
    # Model parameters:
    # Use the default filter, extinction, distance, and minMass, maxMass
    clusterMass = [1e6, 1e7]
    clusterAge = round(math.log10(6e6), 2)  # log(t)
    clusterIMF = 2.35

    kpBins = np.arange(4, 20, 0.1)

    py.clf()

    for mm in range(len(clusterMass)):
        outRoot = '/u/jlu/work/gc/imf/klf/models/test_distributions/clusters/'
        outRoot += 'sim_t%.2f_a%.2f_m%d_' % (clusterAge, clusterIMF, clusterMass[mm])

        files = glob.glob(outRoot + '*.pickle')

        print 'Gathering %d clusters with M = %.2e Msun' % (1, clusterMass[mm])
        _f = open(files[0], 'rb')
        cluster = pickle.load(_f)
        _f.close()

        legendLabel = 'M=%.0e' % clusterMass[mm]
        py.hist(cluster.mag_noWR, bins=kpBins, histtype='step',
                label=legendLabel, normed=True)

    py.gca().set_yscale('log')
    py.legend(loc='upper left')
    py.xlabel('Kp')
    py.ylabel('PDF(Kp) for magBin=0.1')
    py.xlim(10, 16)
    py.ylim(1e-3, 1e-1)

    outDir = '/u/jlu/work/gc/imf/klf/models/test_distributions/plots/'
    py.savefig(outDir + 'test_distribution_kp_pdf_v_Mcl.png')

    
def open_unique_file(file_name, mode='r'):
    dirname, filename = os.path.split(file_name)
    prefix, suffix = os.path.splitext(filename)

    fd, filename = tempfile.mkstemp(suffix, prefix+"_", dirname)
    return os.fdopen(fd, mode), filename


