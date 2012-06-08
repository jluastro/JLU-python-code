def run1(outdir, yng=None, rmin=0, rmax=30, n_live_points=4000, multiples=True):
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
        dist, log_prob_dist = random_distance(cube[0])
        cube[0] = dist

        # Log Age of the cluster
        log_age_cont, log_prob_log_age_cont = random_log_age(cube[1])
        log_age = round(log_age_cont, 2)
        cube[1] = log_age_cont

        # Slope of the IMF
        alpha, log_prob_alpha = random_alpha(cube[2])
        cube[2] = alpha

        # Total Cluster Mass
        Mcl, log_prob_Mcl = random_mass(cube[3])
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

