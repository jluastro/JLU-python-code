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

def make_distance_gen():
    dist_mean = 8.096  # kpc
    dist_std = 0.483   # kpc
    dist_min = 6.793   # kpc
    dist_max = 9.510   # kpc
    dist_a = (dist_min - dist_mean) / dist_std
    dist_b = (dist_max - dist_mean) / dist_std
    return scipy.stats.truncnorm(dist_a, dist_b, loc=dist_mean, scale=dist_std)

def random_distance(randNum):
    dist = dist_gen.ppf(randNum)
    log_prob_dist = dist_gen.logpdf(dist)
    return dist, log_prob_dist

def make_log_age_gen():
    log_age_mean = 6.78
    log_age_std = 0.18
    log_age_min = 6.20
    log_age_max = 7.20
    log_age_a = (log_age_min - log_age_mean) / log_age_std
    log_age_b = (log_age_max - log_age_mean) / log_age_std
    return scipy.stats.truncnorm(log_age_a, log_age_b,
                                 loc=log_age_mean, scale=log_age_std)

def random_log_age(randNum):
    log_age_cont = log_age_gen.ppf(randNum)
    log_prob_log_age_cont = log_age_gen.logpdf(log_age_cont)
    return log_age_cont, log_prob_log_age_cont

def make_alpha_gen():
    alpha_min = 0.10
    alpha_max = 3.35
    alpha_diff = alpha_max - alpha_min
    return scipy.stats.uniform(loc=alpha_min, scale=alpha_diff)

def random_alpha(randNum):
    alpha = alpha_gen.ppf(randNum)
    log_prob_alpha = alpha_gen.logpdf(alpha)
    return alpha, log_prob_alpha

def make_Mcl_gen():
    Mcl_min = 4
    Mcl_max = 100
    Mcl_diff = Mcl_max - Mcl_min
    return scipy.stats.uniform(loc=Mcl_min, scale=Mcl_diff)

def random_mass(randNum):
    Mcl = Mcl_gen.ppf(randNum)
    log_prob_Mcl = Mcl_gen.logpdf(Mcl)
    return Mcl, log_prob_Mcl

def make_N_old_gen():
    N_old_min = 0.2e3
    N_old_max = 1.0e4
    N_old_diff = N_old_max - N_old_min
    return scipy.stats.uniform(loc=N_old_min, scale=N_old_diff)

def random_N_old(randNum):
    N_old = int(round(N_old_gen.ppf(randNum)))
    log_prob_N_old = N_old_gen.logpdf(N_old)
    return N_old, log_prob_N_old

def make_gamma_gen():
    # Values from Schodel+ 2010
    gamma_mean = 0.27
    gamma_std = 0.02
    return scipy.stats.norm(loc=gamma_mean, scale=gamma_std)

def random_gamma(randNum):
    #gamma = gamma_gen.ppf(randNum)
    gamma = 0.27
    log_prob_gamma = gamma_gen.logpdf(gamma)
    return gamma, log_prob_gamma

def make_rcMean_gen():
    # Values from Schodel+ 2010 and Schodel private communication
    rcMean_mean = 15.71 + 0.03  # correction for Ks -> Kp (cool stars)
    rcMean_std = 0.06
    return scipy.stats.norm(loc=rcMean_mean, scale=rcMean_std)

def random_rcMean(randNum):
    #rcMean = rcMean_gen.ppf(randNum)
    rcMean = 15.71 + 0.03
    log_prob_rcMean = rcMean_gen.logpdf(rcMean)
    return rcMean, log_prob_rcMean

def make_rcSigma_gen():
    # Values from Schodel+ 2010
    rcSigma_mean = 0.36  # correction for Ks -> Kp (cool stars)
    rcSigma_std = 0.04
    rcSigma_min = 0.1
    rcSigma_max = 0.8
    rcSigma_a = (rcSigma_min - rcSigma_mean) / rcSigma_std
    rcSigma_b = (rcSigma_max - rcSigma_mean) / rcSigma_std
    return scipy.stats.truncnorm(rcSigma_a, rcSigma_b,
                                 loc=rcSigma_mean, scale=rcSigma_std)

def random_rcSigma(randNum):
    #rcSigma = rcSigma_gen.ppf(randNum)
    rcSigma = 0.36
    log_prob_rcSigma = rcSigma_gen.logpdf(rcSigma)
    return rcSigma, log_prob_rcSigma


# Instantiate random number generators
dist_gen = make_distance_gen()
log_age_gen = make_log_age_gen()
alpha_gen = make_alpha_gen()
Mcl_gen = make_Mcl_gen()
N_old_gen = make_N_old_gen()
gamma_gen = make_gamma_gen()
rcMean_gen = make_rcMean_gen()
rcSigma_gen = make_rcSigma_gen()


def run2(outdir, data=None, rmin=0, rmax=30, n_live_points=300, multiples=True,
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
    Kp_interp = interpolate.splrep(completeness.mag, completeness.comp, k=1, s=0)

    # Completeness curve (resampled to observed Kp)
    comp_at_kp_obs = interpolate.splev(data.kp_ext, Kp_interp)
    comp_at_kp_obs[comp_at_kp_obs < 0] = 0.0
    comp_at_kp_obs[comp_at_kp_obs > 1] = 1.0
    comp_at_kp_obs[data.kp_ext > magCut] = 0.0

    log_comp_at_kp_obs = log_prob(comp_at_kp_obs)

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

        # Number of old stars that exist (not just observed).
        # Remember this is from k_min brightness down to
        # the model magnitude cut (k_max).
        N_old, log_prob_N_old = random_N_old(cube[4])
        cube[4] = N_old

        # Powerlaw slope for old population
        gamma, log_prob_gamma = random_gamma(cube[5])
        cube[5] = gamma

        # Mean of Red Clump: Note ratio of rec-clump to powerlaw is fixed.
        rcMean, log_prob_rcMean = random_rcMean(cube[6])
        rcSigma, log_prob_rcSigma = random_rcSigma(cube[7])
        cube[6] = rcMean
        cube[7] = rcSigma

        # Check that all our prior probabilities are valid, otherwise abort
        # before expensive calculation.
        if ((log_prob_alpha == -np.inf) or (log_prob_log_age_cont == -np.inf) or
            (log_prob_dist == -np.inf) or (log_prob_Mcl == -np.inf) or
            (log_prob_gamma == -np.inf) or (log_prob_N_old == -np.inf) or
            (log_prob_rcMean == -np.inf) or (log_prob_rcSigma == -np.inf)):
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

        sim_k_bin_width = sim_k_bins[1] - sim_k_bins[0]
        
        # Append bins at the bright end to always go up to at least
        # K' = k_min to cover the full range of observations.
        if sim_k_bins[0] > k_min:
            new_bins = np.arange(sim_k_bins[0]-sim_k_bin_width, k_min, -sim_k_bin_width)
            if len(new_bins) > 0:
                tmp = np.zeros(len(new_bins), dtype=float)
                sim_k_bins = np.concatenate([new_bins[::-1], sim_k_bins])
                sim_k_pdf = np.concatenate([tmp, sim_k_pdf])
                sim_k_pdf_norm = np.concatenate([tmp, sim_k_pdf_norm])
        

        # Trim down on the faint side to the model magnitude cut
        idx = np.where(sim_k_bins <= model_mag_cut)[0]

        sim_k_bins = np.append(sim_k_bins[idx], sim_k_bins[idx[-1]+1])
        sim_k_pdf = sim_k_pdf[idx]
        sim_k_pdf_norm = sim_k_pdf_norm[idx]
        sim_k_pdf_norm /= (sim_k_pdf_norm * sim_k_bin_width).sum()
        sim_k_bin_center = sim_k_bins[:-1] + (sim_k_bin_width/2.0)

        ####################
        # PDF for old stars
        ####################
        pl_loc = math.e**k_min
        pl_scale = math.e**k_max - pl_loc
        pl_index = gamma * math.log(10)
        old_powerlaw = scipy.stats.powerlaw(pl_index, loc=pl_loc, scale=pl_scale)
        old_gaussian = scipy.stats.norm(loc=rcMean, scale=rcSigma)

        fracInRC = 0.12
        old_k_cdf = (1.0 - fracInRC) * old_powerlaw.cdf(math.e**sim_k_bins)
        old_k_cdf += fracInRC * old_gaussian.cdf(sim_k_bins)
        old_k_pdf_norm = np.diff(old_k_cdf)
        old_k_pdf_norm /= (old_k_pdf_norm * sim_k_bin_width).sum()

        ####################
        # completeness curves
        ####################
        # Completeness curve (resampled to simulated Kp)
        comp_at_kp_sim = interpolate.splev(sim_k_bin_center, Kp_interp)
        comp_at_kp_sim[comp_at_kp_sim < 0] = 0.0
        comp_at_kp_sim[comp_at_kp_sim > 1] = 1.0
        comp_at_kp_sim[sim_k_bin_center > magCut] = 0.0
        
        # N_yng (down to model_magnitude cut)
        N_yng = int(np.round(sim_k_pdf.sum()))
        cube[8] = N_yng

        ####################
        # Different parts of the likelihood
        ####################
        #####
        # Prob(N_WR | model)
        #####
        N_WR_sim = sim_N_WR
        cube[9] = N_WR_sim
        log_L_N_WR = scipy.stats.poisson.logpmf(data.N_WR, sim_N_WR)

        N_tot = N_yng + N_old
        N_obs = len(data.kp_ext)
        fracYng = float(N_yng) / float(N_tot)

        #####
        # Binomial Coefficient
        #####
        if N_obs >= N_tot:
            return -np.Inf
        else:
            log_binom_coeff = scipy.special.gammaln(N_tot + 1)
            log_binom_coeff -= scipy.special.gammaln(N_obs + 1)
            log_binom_coeff -= scipy.special.gammaln(N_tot - N_obs + 1)

        #####
        # Non detections: log_L_k_non_detect
        #####
        incomp_at_kp_sim = 1.0 - comp_at_kp_sim

        # number of young vs. mag
        n_yng_k = N_yng * sim_k_pdf_norm * sim_k_bin_width
        n_old_k = N_old * old_k_pdf_norm * sim_k_bin_width
        prob_Y1 = n_yng_k / (n_yng_k + n_old_k)
        prob_Y0 = 1.0 - prob_Y1
        #prob_Y1 = fracYng
        #prob_Y0 = 1.0 - prob_Y1

        # number of young and old that are unobserved
        N_yng_unobs = (incomp_at_kp_sim * sim_k_pdf_norm).sum() * N_yng * sim_k_bin_width
        N_old_unobs = (incomp_at_kp_sim * old_k_pdf_norm).sum() * N_old * sim_k_bin_width
        # Rescale to make sure N_yng_unobs + N_old_unobs = N_unobs = N_tot - N_obs
        tmp_N_unobs = N_yng_unobs + N_old_unobs
        N_yng_unobs = int(round( (N_tot - N_obs) * N_yng_unobs / tmp_N_unobs ))
        N_old_unobs = N_tot - N_obs - N_yng_unobs
        
        ## Young part
        tmp_y = fracYng * sim_k_pdf_norm * sim_k_bin_width
        P_I0_y = (incomp_at_kp_sim * tmp_y).sum()

        ## Old part
        tmp_o = (1.0 - fracYng) * old_k_pdf_norm * sim_k_bin_width
        P_I0_o = (incomp_at_kp_sim * tmp_o).sum()

        ## Total Integral
        P_I0 =  incomp_at_kp_sim * (tmp_y + tmp_o)# * sim_k_bin_width

        ## log[ prob(I=0 | model)^(N-n) ]
        log_L_k_non_detect = (N_tot - N_obs) * log_prob(P_I0.sum())
        #log_L_k_non_detect = N_yng_unobs * log_prob(P_I0_y)
        #log_L_k_non_detect += N_old_unobs * log_prob(P_I0_o)

        #####
        # Normalization Constant for the observed side
        #####
        P_I0_all = (tmp_y + tmp_o).sum()# * sim_k_bin_width
        P_I0_detect = (comp_at_kp_sim * (tmp_y + tmp_o)).sum()

        ## log[ prob(I=1 | model)^n ]
        #log_L_norm_coeff = N_tot * log_prob(P_I0_all)
        #log_L_norm_coeff = N_yng * log_prob(tmp_y.sum()) + N_old * log_prob(tmp_o.sum())
        log_L_norm_coeff = N_obs * log_prob(P_I0_detect) + log_L_k_non_detect + log_binom_coeff

        #####
        # Detections: log_L_k_detect
        #####
        log_L_k_detect = 0.0

        # Loop through each star and calc prob of detecting.
        for ii in range(len(data.kp_ext)):
            # Make gaussian around observed k mag
            obs_k_norm = scipy.stats.norm(loc=data.kp_ext[ii], scale=data.kp_err[ii])
            obs_k_norm_cdf = obs_k_norm.cdf(sim_k_bins)
            obs_k_norm_pdf = np.diff(obs_k_norm_cdf)
            obs_k_norm_pdf /= (obs_k_norm_pdf * sim_k_bin_width).sum()
            
            # Convolve gaussian with PDF(K) from model
            if obs_k_norm_pdf.sum() == 0:
                print 'We have a problem... this should never happen.'
                pdb.set_trace()
                return -np.Inf
            else:
                if data.isYoung[ii]:
                    prob = 1.0
                else:
                    prob = 0.0
                    
                # Young part
                # Multiply gaussian with PDF(K) from model and sum to get probability
                L_k_i_y = (sim_k_pdf_norm * obs_k_norm_pdf * sim_k_bin_width).sum()
                L_k_i_y *= prob * fracYng
                #L_k_i_y *= data.prob[ii] * fracYng
                #L_k_i_y *= fracYng * comp_at_kp_obs[ii]
                #log_L_k_i_y = data.prob[ii] * log_prob(L_k_i_y)

                # Old part
                L_k_i_o = (old_k_pdf_norm * obs_k_norm_pdf * sim_k_bin_width).sum()
                L_k_i_o *= (1.0 - prob) * (1.0 - fracYng)
                #L_k_i_o *= (1.0 - data.prob[ii]) * (1.0 - fracYng)
                #L_k_i_o *= (1.0 - fracYng) * comp_at_kp_obs[ii]
                #log_L_k_i_o = (1.0 - data.prob[ii]) * log_prob(L_k_i_o)

                # Combine
                L_k_i = L_k_i_y + L_k_i_o
                
            log_L_k_detect += log_prob(L_k_i) + log_comp_at_kp_obs[ii]
            #log_L_k_detect += log_L_k_i_o + log_L_k_i_y

        log_L = log_L_N_WR
        log_L += log_binom_coeff + log_L_k_non_detect + log_L_k_detect
        #log_L -= log_L_norm_coeff

        # Add in the log(Prior_Probabilities) as well
        log_L += log_prob_dist
        log_L += log_prob_alpha
        log_L += log_prob_Mcl
        log_L += log_prob_log_age_cont
        log_L += log_prob_N_old
        #log_L += log_prob_gamma
        #log_L += log_prob_rcMean
        #log_L += log_prob_rcSigma

        #pdb.set_trace()
        #if log_L >= 0:
        #    pdb.set_trace()

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
            old_k_pdf_tmp = (old_k_pdf_norm * N_old * sim_k_bin_width * comp_at_kp_sim)[tdx]

            N_yng_sim = (sim_k_pdf_norm * N_yng * sim_k_bin_width * comp_at_kp_sim)[tdx].sum()
            N_old_sim = (old_k_pdf_norm * N_old * sim_k_bin_width * comp_at_kp_sim)[tdx].sum()
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
            print ''
            idx = np.where(data.prob > 0)[0]
            print 'YNG arr_L_k_y_np   = %8.1f' % arr_L_k_y_np[idx].sum()
            print 'YNG arr_L_k_y      = %8.1f' % arr_L_k_y[idx].sum()
            print 'FINAL: '
            print 'log_L = %8.1f' % log_L
            print ''
            
            pdb.set_trace()
            
        cube[10] = log_L_N_WR
        cube[11] = log_binom_coeff
        cube[12] = log_L_k_detect
        cube[13] = log_L_k_non_detect
        cube[14] = log_L_norm_coeff

        return log_L


    def likelihood2(cube, ndim, nparams):
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

        # Number of old stars that exist (not just observed).
        # Remember this is from k_min brightness down to
        # the model magnitude cut (k_max).
        N_old, log_prob_N_old = random_N_old(cube[4])
        cube[4] = N_old

        # Powerlaw slope for old population
        gamma, log_prob_gamma = random_gamma(cube[5])
        cube[5] = gamma

        # Mean of Red Clump: Note ratio of rec-clump to powerlaw is fixed.
        rcMean, log_prob_rcMean = random_rcMean(cube[6])
        rcSigma, log_prob_rcSigma = random_rcSigma(cube[7])
        cube[6] = rcMean
        cube[7] = rcSigma

        # Check that all our prior probabilities are valid, otherwise abort
        # before expensive calculation.
        if ((log_prob_alpha == -np.inf) or (log_prob_log_age_cont == -np.inf) or
            (log_prob_dist == -np.inf) or (log_prob_Mcl == -np.inf) or
            (log_prob_gamma == -np.inf) or (log_prob_N_old == -np.inf) or
            (log_prob_rcMean == -np.inf) or (log_prob_rcSigma == -np.inf)):
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

        sim_k_bin_width = sim_k_bins[1] - sim_k_bins[0]
        
        # Append bins at the bright end to always go up to at least
        # K' = k_min to cover the full range of observations.
        if sim_k_bins[0] > k_min:
            new_bins = np.arange(sim_k_bins[0]-sim_k_bin_width, k_min, -sim_k_bin_width)
            if len(new_bins) > 0:
                tmp = np.zeros(len(new_bins), dtype=float)
                sim_k_bins = np.concatenate([new_bins[::-1], sim_k_bins])
                sim_k_pdf = np.concatenate([tmp, sim_k_pdf])
                sim_k_pdf_norm = np.concatenate([tmp, sim_k_pdf_norm])
        

        # Trim down on the faint side to the model magnitude cut
        idx = np.where(sim_k_bins <= model_mag_cut)[0]

        sim_k_bins = np.append(sim_k_bins[idx], sim_k_bins[idx[-1]+1])
        sim_k_pdf = sim_k_pdf[idx]
        sim_k_pdf_norm = sim_k_pdf_norm[idx]
        sim_k_pdf_norm /= (sim_k_pdf_norm * sim_k_bin_width).sum()
        sim_k_bin_center = sim_k_bins[:-1] + (sim_k_bin_width/2.0)

        ####################
        # PDF for old stars
        ####################
        pl_loc = math.e**k_min
        pl_scale = math.e**k_max - pl_loc
        pl_index = gamma * math.log(10)
        old_powerlaw = scipy.stats.powerlaw(pl_index, loc=pl_loc, scale=pl_scale)
        old_gaussian = scipy.stats.norm(loc=rcMean, scale=rcSigma)

        fracInRC = 0.12
        old_k_cdf = (1.0 - fracInRC) * old_powerlaw.cdf(math.e**sim_k_bins)
        old_k_cdf += fracInRC * old_gaussian.cdf(sim_k_bins)
        old_k_pdf_norm = np.diff(old_k_cdf)
        old_k_pdf_norm /= (old_k_pdf_norm * sim_k_bin_width).sum()

        ####################
        # completeness curves
        ####################
        # Completeness curve (resampled to simulated Kp)
        comp_at_kp_sim = interpolate.splev(sim_k_bin_center, Kp_interp)
        comp_at_kp_sim[comp_at_kp_sim < 0] = 0.0
        comp_at_kp_sim[comp_at_kp_sim > 1] = 1.0
        comp_at_kp_sim[sim_k_bin_center > magCut] = 0.0
        
        # N_yng (down to model_magnitude cut)
        N_yng = int(np.round(sim_k_pdf.sum()))
        cube[8] = N_yng

        ####################
        # Different parts of the likelihood
        ####################
        #####
        # Prob(N_WR | model)
        #####
        N_WR_sim = sim_N_WR
        cube[9] = N_WR_sim
        log_L_N_WR = scipy.stats.poisson.logpmf(data.N_WR, sim_N_WR)

        # Some useful numbers
        N_tot = N_yng + N_old
        N_obs = len(data.kp_ext)
        fracYng = float(N_yng) / float(N_tot)

        N_obs_yng_expect = data.prob.sum()
        N_obs_yng = scipy.stats.poisson.rvs(N_obs_yng_expect)
        log_N_obs_yng = scipy.stats.poisson.logpmf(N_obs_yng, N_obs_yng_expect)

        N_obs_old_expect = (1.0 - data.prob).sum()
        N_obs_old = scipy.stats.poisson.rvs(N_obs_old_expect)
        log_N_obs_old = scipy.stats.poisson.logpmf(N_obs_old, N_obs_old_expect)

        #####
        # Binomial Coefficients
        #####
        if (N_obs >= N_tot) or (N_obs_yng >= N_yng) or (N_obs_old >= N_old):
            return -np.Inf
        else:
            log_binom_coeff_yng = scipy.special.gammaln(N_yng + 1)
            log_binom_coeff_yng -= scipy.special.gammaln(N_obs_yng + 1)
            log_binom_coeff_yng -= scipy.special.gammaln(N_yng - N_obs_yng + 1)

            log_binom_coeff_old = scipy.special.gammaln(N_old + 1)
            log_binom_coeff_old -= scipy.special.gammaln(N_obs_old + 1)
            log_binom_coeff_old -= scipy.special.gammaln(N_old - N_obs_old + 1)

        #####
        # Non detections: log_L_k_non_detect
        #####
        incomp_at_kp_sim = 1.0 - comp_at_kp_sim

        ## Young part
        tmp_y = sim_k_pdf_norm # * sim_k_bin_width
        P_I0_y = incomp_at_kp_sim * tmp_y

        N_yng_in_bin = N_yng * sim_k_pdf_norm * sim_k_bin_width * incomp_at_kp_sim
        idx = np.where(N_yng_in_bin > 0)[0]
        log_L_k_non_detect_y = (N_yng_in_bin[idx] * log_prob(P_I0_y[idx])).sum()

        ## Old part
        tmp_o = old_k_pdf_norm # * sim_k_bin_width
        P_I0_o = incomp_at_kp_sim * tmp_o

        N_old_in_bin = N_old * old_k_pdf_norm * sim_k_bin_width * incomp_at_kp_sim
        idx = np.where(N_old_in_bin > 0)[0]
        log_L_k_non_detect_o = (N_old_in_bin[idx] * log_prob(P_I0_o[idx])).sum()

        log_L_k_non_detect_y_tmp = (N_yng - N_obs_yng) * log_prob(P_I0_y)
        log_L_k_non_detect_o_tmp = (N_old - N_obs_old) * log_prob(P_I0_o)

        #####
        # Normalization Constant
        #####
        ## log[ prob(I=1 | model)^n ]
        log_L_norm_coeff_y = N_yng * log_prob(tmp_y.sum())
        log_L_norm_coeff_o = N_old * log_prob(tmp_o.sum())

        #####
        # Detections: log_L_k_detect
        #####
        log_L_k_detect_y = 0.0
        log_L_k_detect_o = 0.0

        # Loop through each star and calc prob of detecting.
        for ii in range(len(data.kp_ext)):
            # Make gaussian around observed k mag
            obs_k_norm = scipy.stats.norm(loc=data.kp_ext[ii], scale=data.kp_err[ii])
            obs_k_norm_cdf = obs_k_norm.cdf(sim_k_bins)
            obs_k_norm_pdf = np.diff(obs_k_norm_cdf)
            obs_k_norm_pdf /= (obs_k_norm_pdf * sim_k_bin_width).sum()
            
            # Convolve gaussian with PDF(K) from model
            if obs_k_norm_pdf.sum() == 0:
                print 'We have a problem... this should never happen.'
                pdb.set_trace()
                return -np.Inf
            else:
                if data.isYoung[ii]:
                    prob = 1.0
                else:
                    prob = 0.0
                    
                # Young part
                # Multiply gaussian with PDF(K) from model and sum to get probability
                L_k_i_y = (sim_k_pdf_norm * obs_k_norm_pdf * sim_k_bin_width).sum()
                L_k_i_y *= comp_at_kp_obs[ii]

                # Old part
                L_k_i_o = (old_k_pdf_norm * obs_k_norm_pdf * sim_k_bin_width).sum()
                L_k_i_o *= comp_at_kp_obs[ii]
                
            log_L_k_detect_y += prob * log_prob(L_k_i_y)
            log_L_k_detect_o += (1.0  - prob) * log_prob(L_k_i_o)

        log_L = log_L_N_WR
        log_L += log_N_obs_yng + log_N_obs_old
        log_L += log_binom_coeff_yng + log_L_k_non_detect_y + log_L_k_detect_y
        log_L += log_binom_coeff_old + log_L_k_non_detect_o + log_L_k_detect_o
        #log_L -= log_L_norm_coeff

        # Add in the log(Prior_Probabilities) as well
        log_L += log_prob_dist
        log_L += log_prob_alpha
        log_L += log_prob_Mcl
        log_L += log_prob_log_age_cont
        log_L += log_prob_N_old
        #log_L += log_prob_gamma
        #log_L += log_prob_rcMean
        #log_L += log_prob_rcSigma

        #pdb.set_trace()
        #if log_L >= 0:
        #    pdb.set_trace()

        cube[10] = log_L_N_WR
        cube[11] = log_binom_coeff_yng
        cube[12] = log_L_k_detect_y
        cube[13] = log_L_k_non_detect_y
        cube[14] = log_L_norm_coeff_y
        cube[15] = log_binom_coeff_old
        cube[16] = log_L_k_detect_o
        cube[17] = log_L_k_non_detect_o
        cube[18] = log_L_norm_coeff_o

        return log_L

    gcutil.mkdir(outdir)

    outroot = outdir + 'mnest_'

    #num_dims = 8
    num_dims = 5
    num_params = 19
    ev_tol = 0.7
    samp_eff = 0.8
    n_clust_param = 4

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

    pymultinest.run(likelihood2, priors, num_dims, n_params=num_params,
                    outputfiles_basename=outroot,
                    verbose=True, resume=False,
                    evidence_tolerance=ev_tol, sampling_efficiency=samp_eff,
                    n_live_points=n_live_points)
        #n_clustering_params=n_clust_param,


def run(outdir, data=None, rmin=0, rmax=30, n_live_points=300, multiples=True,
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
    Kp_interp = interpolate.splrep(completeness.mag, completeness.comp, k=1, s=0)

    # Completeness curve (resampled to observed Kp)
    comp_at_kp_obs = interpolate.splev(data.kp_ext, Kp_interp)
    comp_at_kp_obs[comp_at_kp_obs < 0] = 0.0
    comp_at_kp_obs[comp_at_kp_obs > 1] = 1.0
    comp_at_kp_obs[data.kp_ext > magCut] = 0.0

    log_comp_at_kp_obs = log_prob(comp_at_kp_obs)

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

        # Check that all our prior probabilities are valid, otherwise abort
        # before expensive calculation.
        if ((log_prob_alpha == -np.inf) or (log_prob_log_age_cont == -np.inf) or
            (log_prob_dist == -np.inf) or (log_prob_Mcl == -np.inf)):
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

        sim_k_bin_width = sim_k_bins[1] - sim_k_bins[0]
        
        # Append bins at the bright end to always go up to at least
        # K' = k_min to cover the full range of observations.
        if sim_k_bins[0] > k_min:
            new_bins = np.arange(sim_k_bins[0]-sim_k_bin_width, k_min, -sim_k_bin_width)
            if len(new_bins) > 0:
                tmp = np.zeros(len(new_bins), dtype=float)
                sim_k_bins = np.concatenate([new_bins[::-1], sim_k_bins])
                sim_k_pdf = np.concatenate([tmp, sim_k_pdf])
                sim_k_pdf_norm = np.concatenate([tmp, sim_k_pdf_norm])
        

        # Trim down on the faint side to the model magnitude cut
        idx = np.where(sim_k_bins <= model_mag_cut)[0]

        sim_k_bins = np.append(sim_k_bins[idx], sim_k_bins[idx[-1]+1])
        sim_k_pdf = sim_k_pdf[idx]
        sim_k_pdf_norm = sim_k_pdf_norm[idx]
        sim_k_pdf_norm /= (sim_k_pdf_norm * sim_k_bin_width).sum()
        sim_k_bin_center = sim_k_bins[:-1] + (sim_k_bin_width/2.0)

        ####################
        # completeness curves
        ####################
        # Completeness curve (resampled to simulated Kp)
        comp_at_kp_sim = interpolate.splev(sim_k_bin_center, Kp_interp)
        comp_at_kp_sim[comp_at_kp_sim < 0] = 0.0
        comp_at_kp_sim[comp_at_kp_sim > 1] = 1.0
        comp_at_kp_sim[sim_k_bin_center > magCut] = 0.0

        ####################
        # Different parts of the likelihood
        ####################
        #####
        # Prob(N_WR | model)
        #####
        N_WR_sim = sim_N_WR
        cube[4] = N_WR_sim
        log_L_N_WR = scipy.stats.poisson.logpmf(data.N_WR, sim_N_WR)

        #####
        # Prob(N_yng_obs | model)
        #####
        sim_k_pdf_incomp = sim_k_pdf * comp_at_kp_sim
        N_yng_obs_expect = sim_k_pdf_incomp.sum()
        cube[5] = N_yng_obs_expect

        N_yng_obs = int(np.round(data.prob.sum()))
        log_L_N_yng_obs = scipy.stats.poisson.logpmf(N_yng_obs, N_yng_obs_expect)

        #####
        # Detections: log_L_k_detect
        #####
        log_L_k_detect = 0.0

        # Loop through each star and calc prob of detecting.
        for ii in range(len(data.kp_ext)):
            # Make gaussian around observed k mag
            obs_k_norm = scipy.stats.norm(loc=data.kp_ext[ii], scale=data.kp_err[ii])
            obs_k_norm_cdf = obs_k_norm.cdf(sim_k_bins)
            obs_k_norm_pdf = np.diff(obs_k_norm_cdf)
            obs_k_norm_pdf /= (obs_k_norm_pdf * sim_k_bin_width).sum()
            
            # Convolve gaussian with PDF(K) from model
            if obs_k_norm_pdf.sum() == 0:
                print 'We have a problem... this should never happen.'
                pdb.set_trace()
                return -np.Inf
            else:
                # Multiply gaussian with PDF(K) from model and sum to get probability
                L_k_i = (sim_k_pdf_norm * obs_k_norm_pdf * sim_k_bin_width).sum()

                # Apply completeness correction
                L_k_i *= comp_at_kp_obs[ii]

            log_L_k_detect += data.prob[ii] * log_prob(L_k_i)

        log_L = log_L_N_WR + log_L_N_yng_obs
        log_L += log_L_k_detect

        # Add in the log(Prior_Probabilities) as well
        log_L += log_prob_dist
        log_L += log_prob_alpha
        log_L += log_prob_Mcl
        log_L += log_prob_log_age_cont

        # pdb.set_trace()
        # if log_L >= 0:
        #    pdb.set_trace()

        cube[6] = log_L_N_WR
        cube[7] = log_L_N_yng_obs
        cube[8] = log_L_k_detect

        return log_L


    gcutil.mkdir(outdir)

    outroot = outdir + 'mnest_'

    num_dims = 4
    num_params = 9
    ev_tol = 0.7
    samp_eff = 0.8

    _run = open(outroot + 'params.run', 'w')
    _run.write('Num Dimensions: %d\n' % num_dims)
    _run.write('Num Params: %d\n' % num_params)
    _run.write('Evidence Tolerance: %.1f\n' % ev_tol)
    _run.write('Sampling Efficiency: %.1f\n' % samp_eff)
    _run.write('Num Live Points: %d\n' % n_live_points)
    _run.write('Young Star Data: %s\n' % data_file)
    _run.write('Allowed Multiples in Fit: %s\n' % str(multiples))
    _run.close()

    pymultinest.run(likelihood, priors, num_dims, n_params=num_params,
                    outputfiles_basename=outroot,
                    verbose=True, resume=False,
                    evidence_tolerance=ev_tol, sampling_efficiency=samp_eff,
                    n_live_points=n_live_points)


def load_results2(rootdir):
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
    tab.rename_column('col7', 'N_old')
    tab.rename_column('col8', 'gamma')
    tab.rename_column('col9', 'rcMean')
    tab.rename_column('col10', 'rcSigma')
    tab.rename_column('col11', 'N_yng')
    tab.rename_column('col12', 'N_WR_sim')
    tab.rename_column('col13', 'log_L_N_WR')
    tab.rename_column('col14', 'log_L_binom_coeff_y')
    tab.rename_column('col15', 'log_L_k_detect_y')
    tab.rename_column('col16', 'log_L_k_non_detect_y')
    tab.rename_column('col17', 'log_L_norm_coeff_y')
    tab.rename_column('col18', 'log_L_binom_coeff_o')
    tab.rename_column('col19', 'log_L_k_detect_o')
    tab.rename_column('col20', 'log_L_k_non_detect_o')
    tab.rename_column('col21', 'log_L_norm_coeff_o')

    # Now sort based on logLikelihood
    tab.sort('logLike')

    return tab

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
    tab.rename_column('col7', 'N_WR_sim')
    tab.rename_column('col8', 'N_yng_obs')
    tab.rename_column('col9', 'log_L_N_WR')
    tab.rename_column('col10', 'log_L_N_yng_obs')
    tab.rename_column('col11', 'log_L_k_detect')

    # Now sort based on logLikelihood
    tab.sort('logLike')

    return tab

def plot_posteriors(outdir):
    tab = load_results(outdir)
    weights = tab.weights
    tab.remove_columns(('weights', 'logLike'))
    tab.remove_columns(('log_L_N_WR', 'log_L_binom_coeff', 'log_L_k_detect',
                        'log_L_k_non_detect'))

    gcutil.mkdir(outdir + 'plots/')
    pair_posterior(tab, weights, outfile=outdir+'/plots/posteriors.png', title=outdir)

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
        #N_old = int(parts[8][1:])

        #gamma = 0.27
        #rcMean = 15.71 + 0.03
        #rcSigma = 0.36
        
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

    ax11 = py.subplot2grid((3, 3), (0, 0))
    ax12 = py.subplot2grid((3, 3), (0, 1))
    ax13 = py.subplot2grid((3, 3), (0, 2))
    ax21 = py.subplot2grid((3, 3), (1, 0))
    ax22 = py.subplot2grid((3, 3), (1, 1))
    ax23 = py.subplot2grid((3, 3), (1, 2))
    ax31 = py.subplot2grid((3, 3), (2, 0))
    ax32 = py.subplot2grid((3, 3), (2, 1))
    ax33 = py.subplot2grid((3, 3), (2, 2))

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

    plot_PDF(ax11, 'alpha')
    plot_PDF(ax12, 'logAge')
    plot_PDF(ax13, 'Mcl')
    plot_PDF(ax21, 'distance')
    plot_PDF(ax22, 'N_WR_sim', counter=True)
    # plot_PDF(ax23, 'N_old')
    # plot_PDF(ax31, 'gamma')
    # plot_PDF(ax32, 'rcMean')
    # plot_PDF(ax33, 'rcSigma')

    # Make some adjustments to the axes for Number of stars plots
    N_WR_sim_avg = np.average(tab['N_WR_sim'], weights=tab['weights'])
    N_WR_sim_std = math.sqrt( np.dot(tab['weights'], (tab['N_WR_sim']-N_WR_sim_avg)**2) / tab['weights'].sum() )
    N_WR_lo = N_WR_sim_avg - (3 * N_WR_sim_std)
    N_WR_hi = N_WR_sim_avg + (3 * N_WR_sim_std)
    if N_WR_lo < 0:
        N_WR_lo = 0
    ax22.set_xlim(N_WR_lo, N_WR_hi)

    if sim:
        ax11.axvline(imfSlope, color='red')
        ax12.axvline(logAge, color='red')
        ax13.axvline(Mcl, color='red')
        ax21.axvline(distance, color='red')
        ax22.axvline(numWR, color='red')
        # ax23.axvline(N_old, color='red')
        # ax31.axvline(gamma, color='red')
        # ax32.axvline(rcMean, color='red')
        # ax33.axvline(rcSigma, color='red')

    gcutil.mkdir(outdir + 'plots/')

    py.suptitle(outdir)
    py.savefig(outdir + 'plots/plot_posteriors_1D.png')

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
                   gamma=0.27, Nold=1.5e3, rcMean=15.74, rcSigma=0.36, multiples=True):
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
    rand_number = np.random.rand(len(data.kp))
    # young uncertains
    ydx = np.where((data.isYoung == True) & (rand_number > prob_ID))[0]
    # old uncertains
    odx = np.where((data.isYoung == False) & (rand_number > prob_ID))[0] 

    # Sort the young and old untyped stars by brightness
    ydx_s = data.kp[ydx].argsort()
    odx_s = data.kp[odx].argsort()

    ydx = ydx[ydx_s]
    odx = odx[odx_s]

    # Go through the young stars one at a time and find the closest old star
    # in brightness. Assign them complimentary probabilities.
    oldAssigned = []
    for yy in ydx:
        mdiff = np.abs(data.kp[yy] - data.kp[odx])
        mdx = mdiff.argsort()

        for mm in mdx:
            oo = odx[mm]

            if oo not in oldAssigned:
                data.prob[yy] = np.random.uniform(low=0.4, high=1.0)
                data.prob[oo] = 1.0 - data.prob[yy]
                oldAssigned.append(oo)
                print 'Probability Pair:'
                print '   Young Kp = %.2f  Prob = %.2f' % (data.kp[yy], data.prob[yy])
                print '     Old Kp = %.2f  Prob = %.2f' % (data.kp[oo], data.prob[oo])
                break

    ydx = np.where(data.isYoung == True)[0]
    odx = np.where(data.isYoung == False)[0]
    bins = np.arange(8.5, 18.5, 1.0)  # Bin Edges

    py.figure(1)
    py.clf()
    py.hist(data.kp[ydx], histtype='step', label='No Weights', bins=bins)
    py.hist(data.kp, histtype='step', weights=data.prob,
            label='Prob Weights', bins=bins)
    py.xlabel('Kp')
    py.ylabel('Number of Stars')
    py.title('Young Stars')
    py.legend(loc='upper left')

    py.figure(2)
    py.clf()
    py.hist(data.kp[odx], histtype='step', label='No Weights', bins=bins)
    py.hist(data.kp, histtype='step', weights=(1.0 - data.prob),
            label='Prob Weights', bins=bins)
    py.xlabel('Kp')
    py.ylabel('Number of Stars')
    py.title('Old Stars')
    py.legend(loc='upper left')

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


def get_data_file(logAge, AKs, distance, imfSlope, clusterMass, Nold,
                  multiples, suffix=''):
    if multiples:
        multi = 'multi'
    else:
        multi = 'single'

    out_root = 'cluster_sim_t%.2f_AKs%.1f_d%d_a%.2f_m%d_o%d_%s%s' % \
	(logAge, AKs, distance, imfSlope, clusterMass, Nold, multi, suffix)

    out_file = out_root + '.pickle'

    return out_file

def get_out_dir(logAge, AKs, distance, imfSlope, clusterMass, Nold,
                multiples, fitMultiples, suffix=''):

    if multiples:
        multi = 'multi'
    else:
        multi = 'single'

    if fitMultiples:
        fit_multi = 'multi'
    else:
        fit_multi = 'single'

    out_dir = 'fit_%s_sim_t%.2f_AKs%.1f_d%d_a%.2f_m%d_o%d_%s%s/' % \
        (fit_multi, logAge, AKs, distance, imfSlope, clusterMass, Nold, multi, suffix)

    return out_dir
    

def make_simulated_data_set(logAge, AKs, distance, imfSlope, clusterMass,
                            Nold, multiples=True, suffix=''):

    sim = simulated_data(logAge, AKs, distance, imfSlope, clusterMass,
                         Nold=Nold, multiples=multiples)

    out_file = get_data_file(logAge, AKs, distance, imfSlope, clusterMass,
                             Nold, multiples=multiples, suffix=suffix)

    _out = open(out_file, 'w')
    pickle.dump(sim, _out)
    _out.close()

    # Also write two scripts for fitting the simulated data.
    fit_single = get_out_dir(logAge, AKs, distance, imfSlope, clusterMass,
                          Nold, multiples, False, suffix=suffix)
    py_single_file = fit_single.replace('/', '.py')
    _single = open(py_single_file, 'w')
    _single.write('from jlu.gc.imf import multinest as m\n')
    _single.write('from jlu.papers import lu_gc_imf as imf\n')
    _single.write('\n')
    _single.write('logAge = %.2f\n' % logAge)
    _single.write('AKs = %.1f\n' % AKs)
    _single.write('distance = %d\n' % distance)
    _single.write('imfSlope = %.2f\n' % imfSlope)
    _single.write('clusterMass = %.1e\n' % clusterMass)
    _single.write('Nold = %.2e\n' % Nold)
    _single.write('multiples = %s\n' % multiples)
    _single.write('fitMultiples = False\n')
    _single.write('\n')
    _single.write('out_dir = m.run_simulated_data_set(logAge, AKs, distance, imfSlope, clusterMass, Nold, multiples, fitMultiples, suffix="%s")' % suffix)
    _single.write('\n')
    _single.write('print out_dir\n')
    _single.close()

    fit_multi = get_out_dir(logAge, AKs, distance, imfSlope, clusterMass,
                          Nold, multiples, True, suffix=suffix)
    py_multi_file = fit_multi.replace('/', '.py')
    _multi = open(py_multi_file, 'w')
    _multi.write('from jlu.gc.imf import multinest as m\n')
    _multi.write('from jlu.papers import lu_gc_imf as imf\n')
    _multi.write('\n')
    _multi.write('logAge = %.2f\n' % logAge)
    _multi.write('AKs = %.1f\n' % AKs)
    _multi.write('distance = %d\n' % distance)
    _multi.write('imfSlope = %.2f\n' % imfSlope)
    _multi.write('clusterMass = %.1e\n' % clusterMass)
    _multi.write('Nold = %.2e\n' % Nold)
    _multi.write('multiples = %s\n' % multiples)
    _multi.write('fitMultiples = True\n')
    _multi.write('\n')
    _multi.write('out_dir = m.run_simulated_data_set(logAge, AKs, distance, imfSlope, clusterMass, Nold, multiples, fitMultiples, suffix="%s")' % suffix)
    _multi.write('\n')
    _multi.write('print out_dir\n')
    _multi.close()

    return (py_single_file, py_multi_file)

def run_simulated_data_set(logAge, AKs, distance, imfSlope, clusterMass,
                           Nold, multiples, fitMultiples, suffix='', interact=False):
    if multiples:
        multi = 'multi'
    else:
        multi = 'single'

    data_file = get_data_file(logAge, AKs, distance, imfSlope, clusterMass,
                              Nold, multiples, suffix=suffix)
    out_dir = get_out_dir(logAge, AKs, distance, imfSlope, clusterMass,
                          Nold, multiples, fitMultiples, suffix=suffix)

    run(out_dir, data=data_file, multiples=fitMultiples, interact=interact)

    return out_dir


def plot_results_detail(rootdir):
    res = load_results(rootdir)

    x = res.Mcl
    y = res.alpha
    xlabel = 'Mcl'
    ylabel = 'alpha'

    plotStuff = (('-norm', -res.log_L_norm_coeff),
                 ('binom', res.log_L_binom_coeff),
                 ('detect', res.log_L_k_detect),
                 ('undetect', res.log_L_k_non_detect),
                 ('binom + undetect', res.log_L_binom_coeff + res.log_L_k_non_detect),
                 ('binom + undetect + detect', res.log_L_binom_coeff + res.log_L_k_non_detect + res.log_L_k_detect),
                 ('binom + undetect + detect - norm', res.log_L_binom_coeff + res.log_L_k_non_detect + res.log_L_k_detect - res.log_L_norm_coeff),
                 ('logLike', res.logLike))

    py.close('all')

    titles = [plotStuff[ii][0] for ii in range(len(plotStuff))]
    values = [plotStuff[ii][1] for ii in range(len(plotStuff))]
    print titles

    py.figure(1, figsize=(22,12))
    py.subplots_adjust(left=0.05, right=0.98, bottom=0.05, top=0.95)

    for ii in range(len(titles)):
        py.subplot(2, 4, (ii+1))
        py.scatter(x, y, s=20, marker='.', edgecolor='none', c=values[ii])
        py.title(titles[ii])
        py.colorbar()
        py.xlabel(xlabel)
        py.ylabel(ylabel)
    
    
