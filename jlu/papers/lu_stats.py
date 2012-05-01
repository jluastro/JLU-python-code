import pylab as py
import numpy as np
import pymultinest
import scipy.stats
import atpy
import math
import pdb

def make_random_data():
    # Dist 1
    rand_norm = scipy.stats.norm()
    
    # Dist 2
    rand_uni = scipy.stats.uniform(loc=-5, scale=7)
    
    # Generate 100 objects with normal distribution, p(yng) = 1 (e.g. yng = norm)
    rand_set_1 = rand_norm.rvs(size=100)
    #rand_set_1 = scipy.stats.powerlaw.rvs(2.0, size=100)
    p_yng_1 = np.ones(len(rand_set_1), dtype=float)

    # Generate 50 objects with uniform distribution, p(yng) = 0 (e.g. yng = norm)
    rand_set_2 = rand_uni.rvs(size=100)
    p_yng_2 = np.zeros(len(rand_set_2), dtype=float)

    # Generate another 50 each, but assign non-zero p(yng)
    rand_set_3 = rand_norm.rvs(size=50)
    tmp_p_yng_3 = rand_norm.pdf(rand_set_3)
    tmp_p_old_3 = rand_uni.pdf(rand_set_3)
    p_yng_3 = tmp_p_yng_3 / (tmp_p_yng_3 + tmp_p_old_3)

    rand_set_4 = rand_uni.rvs(size=50)
    tmp_p_yng_4 = rand_norm.pdf(rand_set_4)
    tmp_p_old_4 = rand_uni.pdf(rand_set_4)
    p_yng_4 = tmp_p_yng_4 / (tmp_p_yng_4 + tmp_p_old_4)

    # Gather all the data and p(yng) togeter into a single data set
    data = np.concatenate([rand_set_1, rand_set_2, rand_set_3, rand_set_4])
    p_yng = np.concatenate([p_yng_1, p_yng_2, p_yng_3, p_yng_4])

    bins = np.arange(-5, 5, 1)

    py.clf()
    py.hist(data, histtype='step', label='Unweighted', bins=bins)
    py.hist(data, histtype='step', weights=p_yng, label='Yng', bins=bins)
    py.hist(data, histtype='step', weights=(1.0 - p_yng), label='Old', bins=bins)
    py.legend(loc='upper left')
    py.savefig('/u/jlu/work/stats/test_prob_yng/random_data.png')

    out = atpy.Table()
    out.add_column('mydata', data)
    out.add_column('pyng', p_yng, dtype=np.float32)
    out.write('/u/jlu/work/stats/test_prob_yng/random_data.txt',
              type='ascii', overwrite=True)
                   
def test_membership_prob(test):
    """
    A self-contained test to figure out what we should be doing
    with the membership information (prob(yng)) in the bayesian
    analysis.
    """
    print 'Performing Test: ', test, test == 'mix'

    tab = atpy.Table('/u/jlu/work/stats/test_prob_yng/random_data.txt', type='ascii')
    data = tab.mydata
    p_yng = tab.pyng

    # Now we are going to run a multinest fitting program.
    # We will fit only the gaussian distribution but we need
    # to account for the probability of membership.
    def priors(cube, ndim, nparams):
        return

    def random_alpha(randNum):
        alpha_min = 0.1
        alpha_max = 5
        alpha_diff = alpha_max - alpha_min
        alpha = scipy.stats.uniform.ppf(randNum, loc=alpha_min, scale=alpha_diff)
        log_prob_alpha = scipy.stats.uniform.logpdf(alpha, loc=alpha_min, scale=alpha_diff)

        return alpha, log_prob_alpha

    def random_mean(randNum):
        mean_min = -1.0
        mean_max = 1.0
        mean_diff = mean_max - mean_min
        mean = scipy.stats.uniform.ppf(randNum, loc=mean_min, scale=mean_diff)
        log_prob_mean = scipy.stats.uniform.logpdf(mean, loc=mean_min, scale=mean_diff)
        
        return mean, log_prob_mean

    def random_sigma(randNum):
        sigma_min = 0.0
        sigma_max = 2.0
        sigma_diff = sigma_max - sigma_min
        sigma = scipy.stats.uniform.ppf(randNum, loc=sigma_min, scale=sigma_diff)
        log_prob_sigma = scipy.stats.uniform.logpdf(sigma, loc=sigma_min, scale=sigma_diff)

        return sigma, log_prob_sigma

    def random_uni_edge(randNum, edge_min, edge_max):
        edge_diff = edge_max - edge_min
        edge = scipy.stats.uniform.ppf(randNum, loc=edge_min, scale=edge_diff)
        log_prob_edge = scipy.stats.uniform.logpdf(edge, loc=edge_min, scale=edge_diff)

        return edge, log_prob_edge

    def logLikePL1(cube, ndim, nparams):
        alpha, log_prob_alpha = random_alpha(cube[0])
        cube[0] = alpha

        L_i = scipy.stats.powerlaw.pdf(data, alpha) * p_yng
        log_L = np.log10( L_i ).sum()
        log_L += log_prob_alpha

        return log_L

    def logLikePL2(cube, ndim, nparams):
        alpha, log_prob_alpha = random_alpha(cube[0])
        cube[0] = alpha

        L_i = scipy.stats.powerlaw.pdf(data, alpha)
        log_L = (p_yng * np.log10( L_i )).sum()
        log_L += log_prob_alpha

        return log_L

    def logLike1(cube, ndim, nparams):
        mean, log_prob_mean = random_mean(cube[0])
        cube[0] = mean

        sigma, log_prob_sigma = random_sigma(cube[1])
        cube[1] = sigma

        idx = np.where(p_yng != 0)[0]

        L_i = scipy.stats.norm.pdf(data[idx], loc=mean, scale=sigma) * p_yng[idx]
        log_L = np.log10( L_i ).sum()
        log_L += log_prob_mean
        log_L += log_prob_sigma

        return log_L

    def logLike2(cube, ndim, nparams):
        mean, log_prob_mean = random_mean(cube[0])
        cube[0] = mean

        sigma, log_prob_sigma = random_sigma(cube[1])
        cube[1] = sigma

        L_i = scipy.stats.norm.pdf(data, loc=mean, scale=sigma)
        log_L = (p_yng * np.log10( L_i )).sum()
        log_L += log_prob_mean
        log_L += log_prob_sigma

        return log_L

    def logLike3(cube, ndim, nparams):
        mean, log_prob_mean = random_mean(cube[0])
        cube[0] = mean

        sigma, log_prob_sigma = random_sigma(cube[1])
        cube[1] = sigma

        tmp = np.random.uniform(size=len(data))
        idx = np.where(tmp <= p_yng)
        L_i = scipy.stats.norm.pdf(data[idx], loc=mean, scale=sigma)
        log_L = np.log10( L_i ).sum()
        log_L += log_prob_mean
        log_L += log_prob_sigma

        return log_L

    def logLike4(cube, ndim, nparams):
        mean, log_prob_mean = random_mean(cube[0])
        cube[0] = mean

        sigma, log_prob_sigma = random_sigma(cube[1])
        cube[1] = sigma

        uni_l, log_prob_uni_l = random_uni_edge(cube[2], -10, -1)
        cube[2] = uni_l

        uni_h, log_prob_uni_h = random_uni_edge(cube[3], 0, 10)
        cube[3] = uni_h


        L_i_m1 = scipy.stats.norm.pdf(data, loc=mean, scale=sigma)
        L_i_m2 = scipy.stats.uniform.pdf(data, loc=uni_l, scale=(uni_h - uni_l))
        L_i = (p_yng * L_i_m1) + ((1 - p_yng) * L_i_m2)
        log_L = np.log10( L_i ).sum()
        log_L += log_prob_mean
        log_L += log_prob_sigma
        log_L += log_prob_uni_l
        log_L += log_prob_uni_h

        return log_L

    num_params = 2
    num_dims = 2
    ev_tol = 0.7
    samp_eff = 0.5
    n_live_points = 300

    #Now run all 3 tests.

    if test == 'multi':
        outroot = '/u/jlu/work/stats/test_prob_yng/multi_'
        pymultinest.run(logLike1, priors, num_dims, n_params=num_params,
                        outputfiles_basename=outroot,
                        verbose=True, resume=False,
                        evidence_tolerance=ev_tol, sampling_efficiency=samp_eff,
                        n_live_points=n_live_points)

    if test == 'power':
        outroot = '/u/jlu/work/stats/test_prob_yng/power_'
        pymultinest.run(logLike2, priors, num_dims, n_params=num_params,
                        outputfiles_basename=outroot,
                        verbose=True, resume=False,
                        evidence_tolerance=ev_tol, sampling_efficiency=samp_eff,
                        n_live_points=n_live_points)

    if test == 'mix':
        num_params = 4
        num_dims = 4
        n_clust_param = num_dims - 1
        outroot = '/u/jlu/work/stats/test_prob_yng/mix_'
        pymultinest.run(logLike4, priors, num_dims, n_params=num_params,
                        outputfiles_basename=outroot,
                        verbose=True, resume=False,
                        evidence_tolerance=ev_tol, sampling_efficiency=samp_eff,
                        n_live_points=n_live_points)

    if test == 'mc':
        outroot = '/u/jlu/work/stats/test_prob_yng/mc_'
        pymultinest.run(logLike3, priors, num_dims, n_params=num_params,
                        outputfiles_basename=outroot,
                        verbose=True, resume=False,
                        evidence_tolerance=ev_tol, sampling_efficiency=samp_eff,
                        n_clustering_params=n_clust_param,
                        n_live_points=n_live_points)


def plot_test_membership_prob(out_file_root):
    from jlu.gc.imf import multinest as m
    outroot = '/u/jlu/work/stats/test_prob_yng/' + out_file_root + '_'

    tab = atpy.Table(outroot + '.txt', type='ascii')

    # First column is the weights
    weights = tab['col1']
    logLike = tab['col2'] / -2.0
    
    # Now delete the first two rows
    tab.remove_columns(('col1', 'col2'))

    # Rename the parameter columns. This is hard-coded to match the
    # above run() function.
    tab.rename_column('col3', 'mean')
    tab.rename_column('col4', 'sigma')

    m.pair_posterior(tab, weights, outfile=outroot+'posteriors.png')


def test_bernoulli_prob():
    """
    A self-contained test to figure out why the Bernoulli distribution
    isn't working in the likelihood.
    """
    print 'Starting'
    ####################
    # Some random powerlaw populations to play with.
    ####################
    rand_set_1 = scipy.stats.powerlaw.rvs(3.0, size=10000)

    # Apply a linear completeness correction
    def comp(xval, x0=0.6, a=30):
        dx = xval - x0
        denom = np.sqrt(1.0 + a**2 * dx**2)
        f = 0.5 * (1.0 - ((a * dx) / denom))

        return f

    comp_at_rand = comp(rand_set_1)
    detect_rand = scipy.stats.uniform.rvs(size=len(rand_set_1))
    detect_idx = np.where(detect_rand <= comp_at_rand)[0]

    observed_set_1 = rand_set_1[detect_idx]
    data = observed_set_1
    N_obs = len(data)
    print 'Number of observed stars'

    # Now we are going to run a multinest fitting program.
    # We will fit only the gaussian distribution but we need
    # to account for the probability of membership.
    def priors(cube, ndim, nparams):
        return

    print 'Random Number generators'
    alpha_min = 1.0
    alpha_max = 3.2
    alpha_diff = alpha_max - alpha_min
    alpha_gen = scipy.stats.uniform(loc=alpha_min, scale=alpha_diff)

    log_N_min = np.log(1000)
    log_N_max = np.log(50000)
    log_N_diff = log_N_max - log_N_min
    log_N_gen = scipy.stats.uniform(loc=log_N_min, scale=log_N_diff)
    
    def random_alpha(randNum):
        alpha = alpha_gen.ppf(randNum)
        log_prob_alpha = alpha_gen.logpdf(alpha)
        return alpha, log_prob_alpha

    def random_N(randNum):
        log_N = log_N_gen.ppf(randNum)
        N = math.e**log_N
        log_prob_N = log_N_gen.logpdf(log_N)
        return N, log_prob_N


    # Bins for histograms of PDF
    bin_width = 0.025
    bins = np.arange(0, 1+bin_width, bin_width)
    bin_centers = bins[:-1] + (bin_width / 2.0)

    # Completeness at bin centers
    print 'completeness'
    comp_at_bins = comp(bin_centers)
    incomp_at_bins = 1.0 - comp_at_bins

    def logLike(cube, ndim, nparams):
        alpha, log_prob_alpha = random_alpha(cube[0])
        cube[0] = alpha

        N, log_prob_N = random_N(cube[1])
        cube[1] = N

        if N_obs >= N:
            return -np.Inf

        # Make a simulated data set - similar to what we do when we don't
        # have the analytic expression for the luminosity function.
        sim_plaw = scipy.stats.powerlaw(alpha)

        # Bin it up to make a normalized PDF
        sim_cdf = sim_plaw.cdf(bins)
        sim_pdf = np.diff(sim_cdf)
        sim_pdf_norm = sim_pdf / (sim_pdf * bin_width).sum()

        ##########
        # Parts of the Likelihood
        ##########
        # Binomial coefficient:
        log_L_binom_coeff = scipy.special.gammaln(N + 1)
        log_L_binom_coeff -= scipy.special.gammaln(N_obs + 1)
        log_L_binom_coeff -= scipy.special.gammaln(N - N_obs + 1)

        # Undetected part
        tmp = (sim_pdf_norm * incomp_at_bins * bin_width).sum()
        log_L_non_detect = (N - N_obs) * np.log(tmp)

        # Detected part
        log_L_detect = 0.0

        for ii in range(N_obs):
            # Find the closest bin in the PDF
            dx = np.abs(data[ii] - bin_centers)
            idx = dx.argmin()

            L_i = comp_at_bins[idx] * sim_pdf_norm[idx]

            if L_i == 0.0:
                log_L_detect += -np.Inf
            else:
                log_L_detect += np.log(L_i)

        log_L = log_L_binom_coeff + log_L_non_detect + log_L_detect
        log_L += log_prob_alpha
        log_L += log_prob_N

        cube[2] = log_L_binom_coeff
        cube[3] = log_L_non_detect
        cube[4] = log_L_detect

        return log_L


    def logLike2(cube, ndim, nparams):
        alpha, log_prob_alpha = random_alpha(cube[0])
        cube[0] = alpha

        N, log_prob_N = random_N(cube[1])
        cube[1] = N

        # Make a simulated data set - similar to what we do when we don't
        # have the analytic expression for the luminosity function.
        sim_plaw = scipy.stats.powerlaw(alpha)

        # Bin it up to make a normalized PDF
        sim_cdf = sim_plaw.cdf(bins)
        sim_pdf = np.diff(sim_cdf) * N
        sim_pdf *= comp_at_bins
        sim_pdf_norm = sim_pdf / (sim_pdf * bin_width).sum()

        N_obs_sim = sim_pdf.sum()

        ##########
        # Parts of the Likelihood
        ##########
        # Number of stars part
        log_L_N_obs = scipy.stats.poisson.logpmf(N_obs, N_obs_sim)
        
        # Detected part
        log_L_detect = 0.0

        for ii in range(N_obs):
            # Find the closest bin in the PDF
            dx = np.abs(data[ii] - bin_centers)
            idx = dx.argmin()

            L_i = sim_pdf_norm[idx]

            if L_i == 0.0:
                log_L_detect += -np.Inf
            else:
                log_L_detect += np.log(L_i)

        log_L = log_L_detect + log_L_N_obs
        log_L += log_prob_alpha
        log_L += log_prob_N

        cube[2] = log_L_detect
        cube[3] = log_L_N_obs
        cube[4] = N_obs_sim

        return log_L


    num_params = 5
    num_dims = 2
    n_clust_param = num_dims - 1
    ev_tol = 0.7
    samp_eff = 0.5
    n_live_points = 300

    # Now run the tests.
    outroot = '/u/jlu/work/stats/test_bernoulli/multi_'
    print 'running multinest'
    pymultinest.run(logLike2, priors, num_dims, n_params=num_params,
                    outputfiles_basename=outroot,
                    verbose=True, resume=False,
                    evidence_tolerance=ev_tol, sampling_efficiency=samp_eff,
                    n_live_points=n_live_points)


def plot_test_bernoulli(out_file_root):
    outroot = '/u/jlu/work/stats/test_bernoulli/' + out_file_root + '_'

    tab = atpy.Table(outroot + '.txt', type='ascii')

    tab['col2'] /= -2.0

    tab.rename_column('col1', 'weights')
    tab.rename_column('col2', 'logLike')
    tab.rename_column('col3', 'alpha')
    tab.rename_column('col4', 'N')
    tab.rename_column('col5', 'logL_b')
    tab.rename_column('col6', 'logL_nd')
    tab.rename_column('col7', 'logL_d')

    py.figure(1)
    py.clf()
    bins_alpha = np.arange(1.0, 5.0, 0.01)
    (n, b, p) = py.hist(tab.alpha, weights=tab.weights, histtype='step', bins=bins_alpha)
    py.ylim(0, 1.1 * n.max())
    idx = np.where(n > 0)[0]
    py.xlim(b[idx[0]-1], b[idx[-1]+1])
    py.axvline(3.0)

    py.figure(2)
    py.clf()
    bins_N = np.arange(1000, 50000, 100)
    (n, b, p) = py.hist(tab.N, weights=tab.weights, histtype='step', bins=bins_N)
    py.ylim(0, 1.1 * n.max())
    idx = np.where(n > 0)[0]
    py.xlim(b[idx[0]-1], b[idx[-1]+1])
    py.axvline(10000)


