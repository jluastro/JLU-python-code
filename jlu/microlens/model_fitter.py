import numpy as np
import scipy.stats
import pdb
import pymultinest

def make_gen(min, max):
    return scipy.stats.uniform(loc=min, scale=max-min)

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
    beta_gen = make_gen(0.001, 5)
    muL_x_gen = make_gen(-20, 20)
    muL_y_gen = make_gen(-20, 20)
    muS_x_gen = make_gen(-20, 20)
    muS_y_gen = make_gen(-20, 20)
    dS_gen = make_gen(0, 6000)
    dL_gen = make_gen(6001, 12000)
    imag_base_gen = make_gen(17, 20)
	
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
        cube[9] = dS_gen.ppf(cube[9])
        cube[10] = dL_gen.ppf(cube[10])
        cube[11] = imag.ppf(cube[11])
        
        # mL, lnL_mL = random_prob(mL_gen, cube[0])
        # cube[0] = mL

        # t0, lnL_t0 = random_prob(t0_gen, cube[1])
        # cube[1] = t0

        # xS0_x, lnL_xS0_x = random_prob(xS0_x_gen, cube[2])
        # cube[2] = xS0_x

        # xS0_y, lnL_xS0_y = random_prob(xS0_y_gen, cube[3])
        # cube[3] = xS0_y
        
        # beta, lnL_beta = random_prob(beta_gen, cube[4])
        # cube[4] = beta

        # muL_x, lnL_muL_x = random_prob(muL_x_gen, cube[5])
        # cube[5] = muL_x

        # muL_y, lnL_muL_y = random_prob(muL_y_gen, cube[6])
        # cube[6] = muL_y

        # muS_x, lnL_muS_x = random_prob(muS_x_gen, cube[7])
        # cube[7] = muS_x

        # muS_y, lnL_muS_y = random_prob(muS_y_gen, cube[8])
        # cube[8] = xS0_y

        # dS, lnL_dS = random_prob(dS_gen, cube[9])
        # cube[9] = dS

        # dL, lnL_dS = random_prob(dL_gen, cube[10])
        # cube[10] = dL

        # imag, lnL_imag = random_prob(imag_gen, cube[11])
        # cube[11] = imag

        # lnL_prior = lnL_mL + lnL_t0 + lnL_xS0_x + lnL_xS0_y + lnL_muL_x + lnL_muL_y
        # lnL_prior += lnL_muS_x + lnL_muS_y + lnL_dS + lnL_dL + lnL_imag
                                                                        
        return 
	
    def likelihood(cube, ndim, nparams):
        mL = cube[0]
        t0 = cube[1]
        xS0 = np.array([cube[2], cube[3]])
        beta = cube[4]
        muL = np.array([cube[5], cube[6]])
        muS = np.array([cube[7], cube[8]])
        dS = cube[9]
        dL = cube[10]
        imag = cube[11]

        pspl = model.PSPL(mL, t0, xS0, beta, muL, muS, dL, dS, imag)

        lnL_phot = pspl.likely_photometry(t_phot, imag, imag_err)
        lnL_ast = pspl.likely_astrometry(t_ast, xpos, ypos, xpos_err, ypos_err)

        lnL = lnL_phot.mean() + lnL_ast.mean()

        return lnL

    
    num_dims = 11
    num_params = 11  #cube will have this many dimensions
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
