import pylab as plt
import numpy as np
import dynesty
from dynesty import plotting as dyplot
from dynesty import utils as dyfunc
from multiprocessing import Pool
import pymultinest
import os
import scipy
import pdb
import scipy.stats
import pickle
from astropy.table import Table
import sys
sys.path.append('/g/lu/scratch/siyao/code')
import sjia
from scipy import stats
import time
from astropy.modeling import powerlaws
from popstar.imf import imf
from sklearn.neighbors import KernelDensity
import math


def prep_data_from_siyao():
    # Write now, I can only run this on galley because of pymysql

    # Assume disk properties
    i_disk1 = 123.0
    o_disk1 = 95.0
    ie_disk1 = 14.0 
    oe_disk1 = 15.0

    i_plane2 = 88.0
    o_plane2 = 244.0
    ie_plane2 = 18.0
    oe_plane2 = 19.0
    
    # Directory containing 1 <star>.mc.dat file for each star. These are pickle files.
    #pdffile = '{0}{1}.mc.dat'.format(root+'analyticOrbits/', star)
    #pdf = pickle.load(open(pdffile, 'rb'))
    mc_dir = '/g/lu/scratch/siyao/work/3_young/analyticOrbits/'

    # Load up disk membership probabilities
    d1_prob_mem_file = '/g/lu/scratch/siyao/work/3_young/disk/disk_membership_prob1.dat'
    d2_prob_mem_file = '/g/lu/scratch/siyao/work/3_young/disk/disk_membership_prob2.dat'

    d1_prob_mem = Table.read(d1_prob_mem_file, format='ascii')
    d2_prob_mem = Table.read(d2_prob_mem_file, format='ascii')

    # Join the tables and add a properly normalized non-disk membership probability.
    prob_mem = d1_prob_mem['col1', 'col2']
    prob_mem.rename_column('col1', 'name')
    prob_mem.rename_column('col2', 'p_d1')
    prob_mem['p_d2'] = d2_prob_mem['col2']
    prob_mem['p_nd'] = 1.0 - prob_mem['p_d1'] - prob_mem['p_d2']

    # Load up the posteriors for all the stars in the sample. Can we keep this all in memory? 
    N_stars = len(prob_mem)
    N_samp = int(1e5)

    i_all = np.zeros((N_stars, N_samp), dtype=float)
    o_all = np.zeros((N_stars, N_samp), dtype=float)
    e_all = np.zeros((N_stars, N_samp), dtype=float)
    p_all = np.zeros((N_stars, N_samp), dtype=float)
    # w_all = np.zeros((N_stars, N_samp), dtype=float)
    # t0_all = np.zeros((N_stars, N_samp), dtype=float)
    # ph_all = np.zeros((N_stars, N_samp), dtype=float)

    for nn in range(N_stars):
        pdf = pickle.load(open(mc_dir + prob_mem['name'][nn] + '.mc.dat', 'rb'))
    
        i_all[nn, :] = pdf.i
        o_all[nn, :] = pdf.o
        e_all[nn, :] = pdf.e
        p_all[nn, :] = pdf.p
        # w_all[nn, :] = pdf.w
        # t0_all[nn, :] = pdf.t0
        # ph_all[nn, :] = pdf.ph

    # Calc semi-major axis.
    mass = 3.984e6  # Do et al. 2019

    a_all = (p_all**2 * mass)**(1./3.)

    # Make weight arrays for each disk/group/population.
    wgt_d1 = np.zeros((N_stars, N_samp), dtype=float)
    wgt_d2 = np.zeros((N_stars, N_samp), dtype=float)
    wgt_nd = np.zeros((N_stars, N_samp), dtype=float)

    for ss in range(len(prob_mem)):
        wgt_d1[ss, :]  = stats.norm.pdf(i_all[ss, :], loc=i_disk1, scale=ie_disk1)
        wgt_d1[ss, :] *= stats.norm.pdf(o_all[ss, :], loc=o_disk1, scale=oe_disk1)
        wgt_d1[ss, :] *= prob_mem['p_d1'][ss] / wgt_d1[ss, :].sum()
    
        wgt_d2[ss, :]  = stats.norm.pdf(i_all[ss, :], loc=i_plane2, scale=ie_plane2)
        wgt_d2[ss, :] *= stats.norm.pdf(o_all[ss, :], loc=o_plane2, scale=oe_plane2)
        wgt_d2[ss, :] *= prob_mem['p_d2'][ss] / wgt_d2[ss, :].sum()
    
        wgt_nd[ss, :]  = np.repeat(1.0 / N_samp, N_samp)
        wgt_nd[ss, :] *= prob_mem['p_nd'][ss] / wgt_nd[ss, :].sum()

    p_thresh = 0.1
    s_d1 = np.where(prob_mem['p_d1'] > p_thresh)[0]
    s_d2 = np.where(prob_mem['p_d2'] > p_thresh)[0]
    s_nd = np.where(prob_mem['p_nd'] > p_thresh)[0]

    # Save output
    prob_mem.write('membership_probs.fits', overwrite=True)

    _pdfs = open('all_pdfs_weights.pkl', 'wb')

    pdf_dict = {'i': i_all, 'e': e_all, 'o': o_all, 'p': p_all, 'a': a_all}
    wgt_dict = {'d1': wgt_d1, 'd2': wgt_d2, 'nd': wgt_nd}
    d1_dict = {'i': i_disk1, 'ie': ie_disk1, 'o': o_disk1, 'oe': oe_disk1}
    d2_dict = {'i': i_plane2, 'ie': ie_plane2, 'o': o_plane2, 'oe': oe_plane2}
    grp_dict = {'s_d1': s_d1, 's_d2': s_d1, 's_nd': s_nd}

    pickle.dump( pdf_dict, _pdfs )
    pickle.dump( wgt_dict, _pdfs )
    pickle.dump( d1_dict, _pdfs )
    pickle.dump( d2_dict, _pdfs )
    pickle.dump( grp_dict, _pdfs )

    _pdfs.close()

    return

def load_pdfs_weights_pickle(filename):
    _pkl = open(filename, 'rb')

    pdf_dict = pickle.load(_pkl)
    wgt_dict = pickle.load(_pkl)
    d1_dict = pickle.load(_pkl)
    d2_dict = pickle.load(_pkl)
    grp_dict = pickle.load(_pkl)

    return pdf_dict, wgt_dict, d1_dict, d2_dict, grp_dict


class Eccentricity_Solver(object):
    """
    Use methods of Hogg et al. 2010 and Bowler et al. 2020 to fit the 
    eccentricity distribution of a population based on importance
    sampling the existing individual posteriors. 
    
    We will define the eccentricity distribution using a beta function
    with two parameters $\alpha$ and $\beta$. 
    
    $\mathcal{L}_{\alpha,\beta} \approx \Pi_{n=1}^{N_{stars}} \frac{1}{K} \Sigma_{k=1}^{K} f_\alpha(e_{nk}) * w_{nk}
    """
    
    default_priors = {
        'alpha': ('make_gen', 0, 10),
        'beta':  ('make_gen', 0, 10)
    }

    def __init__(self, e_pdfs, weights, star_names):
        """
        ecc_pdfs: numpy array (N_stars, N_samples)
            2D array containing N_stars with N_samples of the 
            eccentricity distribution.
            
        weights: numpy array (N_stars, N_samples)
            2D array of weights (usually from disk membership analysis)
            for each sample and each star. 
            
        star_names: numpy array or list
            List of star names with length of N_stars.
            
        Once initialized, call solve() or run dynesty using the 
        Prior() and LogLikelihood() functions.
            
        """
        
        # Data we will use in fitting.
        self.e_pdfs = e_pdfs
        self.weights = weights
        self.star_names = star_names

        # Setup parameters to fit.
        self.fitter_param_names = ['alpha', 'beta']
        self.n_dims = len(self.fitter_param_names)
        self.n_params = self.n_dims # nothing extra to carry.
        self.n_clustering_params = self.n_dims

        # Setup priors
        self.make_default_priors()
            

        return
        
    def make_default_priors(self):
        """
        Setup our prior distributions (i.e. random samplers). We will
        draw from these in the Prior() function. We set them up in advance
        because they depend on properties of the data. Also,
        they can be over-written by custom priors as desired.

        To make your own custom priors, use the make_gen() functions
        with different limits.
        """
        self.priors = {}
        for param_name in self.fitter_param_names:
            prior_type, prior_min, prior_max = self.default_priors[param_name]
            
            if prior_type == 'make_gen':
                self.priors[param_name] = make_gen(prior_min, prior_max)
                
        return

    def Prior(self, cube, ndim=None, nparams=None):
        for i, param_name in enumerate(self.fitter_param_names):
            cube[i] = self.priors[param_name].ppf(cube[i])

        return cube

    def LogLikelihood(self, cube, ndim=None, n_params=None):
        """
        This is just a wrapper because PyMultinest requires passing in
        the ndim and nparams.
        """
        alpha = cube[0]
        beta = cube[1]
        
        # First, lets evaluate the beta function at all of our
        # eccentricities. We are still 2D here. (N_stars, K_samples)
        p_e_nk = scipy.stats.beta.pdf(self.e_pdfs, alpha, beta)
        
        # Weight by disk membership
        p_e_nk_wgt = p_e_nk * self.weights
        
        # Sum over all the samples for each star.
        # Also normalize by K
        p_e_n = p_e_nk_wgt.sum(axis=1) / self.e_pdfs.shape[1]

        zdx = np.where(p_e_n == 0)[0]
        if len(zdx) > 0:
            pdb.set_trace()
        
        # Switch into log space
        log_p_e_n = np.log(p_e_n)
                
        # Sum to get final log likelihood.
        lnL = log_p_e_n.sum()
        
        return lnL


    def solve(self, n_cpu=1, pool=None):
        if (n_cpu > 1):
            self.sampler = dynesty.NestedSampler(self.LogLikelihood, self.Prior, 
                                                 ndim=self.n_dims, bound='multi',
                                                 sample='unif',
                                                 pool = pool, queue_size = n_cpu)
        else:
            self.sampler = dynesty.NestedSampler(self.LogLikelihood, self.Prior, 
                                                 ndim=self.n_dims, bound='multi',
                                                 sample='unif')

        self.sampler.run_nested(print_progress=True, maxiter=2000)

        return
        

    def save(self, outfile):
        _pkl = open(outfile, 'wb')

        pickle.dump(self.sampler.results, _pkl)

        _pkl.close()

        return


    
def make_gen(min, max):
    return scipy.stats.uniform(loc=min, scale=max - min)

def make_norm_gen(mean, std):
    return scipy.stats.norm(loc=mean, scale=std)

def make_loguniform_gen(min, max):
    """
    min and max are before you take the log.
    This will be uniform in any log basis set. 
    """
    return scipoy.stats.loguniform(min, max)
    
def make_lognorm_gen(mean_log, std_log):
    return scipy.stats.lognorm(std_log, scale=np.exp(mean_log))

def load_data_and_sampler_results(pdfs_weights_file, sampler_file):
    """
    Load up data and sampler results from pickle files.
    """
    tmp =  load_pdfs_weights_pickle(pdfs_weights_file)

    results = pickle.load(open(sampler_file, 'rb'))

    stuff = list(tmp) + [results]
    
    return stuff

def fit_e_disk1():
    pdfs_weights_file = 'all_pdfs_weights.pkl'
    membership_file = 'membership_probs.fits'
    
    tmp =  load_pdfs_weights_pickle(pdfs_weights_file)

    pdf_dict = tmp[0]
    wgt_dict = tmp[1]
    d1_dict = tmp[2]
    d2_dict = tmp[3]
    grp_dict = tmp[4]

    prob_mem = Table.read(membership_file)

    # Fit only stars with non-zero membership probability.
    p_thresh = 0.1
    s_d1 = np.where(prob_mem['p_d1'] > p_thresh)[0]
    e_solver = Eccentricity_Solver(pdf_dict['e'][s_d1, :], wgt_dict['d1'][s_d1, :], prob_mem['name'][s_d1])

    t0 = time.time()
    # n_cpu = 4
    # pool = Pool(n_cpu)
    e_solver.solve()
    t1 = time.time()
    print('Runtime: ', t1 - t0)

    e_solver.save('dnest_ecc_d1.pkl')

    return

def fit_e_disk2():
    pdfs_weights_file = 'all_pdfs_weights.pkl'
    membership_file = 'membership_probs.fits'
    
    tmp =  load_pdfs_weights_pickle(pdfs_weights_file)

    pdf_dict = tmp[0]
    wgt_dict = tmp[1]
    d1_dict = tmp[2]
    d2_dict = tmp[3]
    grp_dict = tmp[4]

    prob_mem = Table.read(membership_file)

    # Fit only stars with non-zero membership probability.
    p_thresh = 0.1
    s_d2 = np.where(prob_mem['p_d2'] > p_thresh)[0]
    e_solver = Eccentricity_Solver(pdf_dict['e'][s_d2, :], wgt_dict['d2'][s_d2, :], prob_mem['name'][s_d2])
    e_solver.priors['alpha'] = make_gen(0, 50)
    e_solver.priors['beta'] = make_gen(0, 30)

    t0 = time.time()
    # n_cpu = 4
    # pool = Pool(n_cpu)
    sampler = dynesty.DynamicNestedSampler(e_solver.LogLikelihood, e_solver.Prior, 
                                        ndim=e_solver.n_dims, bound='multi',
                                        sample='unif')

    sampler.run_nested(print_progress=True, dlogz_init=0.05, nlive_init=1000, nlive_batch=500,
                    maxiter_init=20000, maxiter_batch=2000, maxbatch=10)
    
    e_solver.sampler = sampler

    t1 = time.time()
    print('Runtime: ', t1 - t0)

    e_solver.save('dnest_ecc_d2.pkl')

    return

def fit_e_nondisk():
    pdfs_weights_file = 'all_pdfs_weights.pkl'
    membership_file = 'membership_probs.fits'
    
    tmp =  load_pdfs_weights_pickle(pdfs_weights_file)

    pdf_dict = tmp[0]
    wgt_dict = tmp[1]
    d1_dict = tmp[2]
    d2_dict = tmp[3]
    grp_dict = tmp[4]

    prob_mem = Table.read(membership_file)

    # Fit only stars with non-zero membership probability.
    p_thresh = 0.1
    s_nd = np.where(prob_mem['p_nd'] > p_thresh)[0]
    e_solver = Eccentricity_Solver(pdf_dict['e'][s_nd, :], wgt_dict['nd'][s_nd, :], prob_mem['name'][s_nd])

    t0 = time.time()
    # n_cpu = 4
    # pool = Pool(n_cpu)
    e_solver.solve()
    t1 = time.time()
    print('Runtime: ', t1 - t0)

    e_solver.save('dnest_ecc_nd.pkl')

    return

def plot_e_fit_results(group):
    pdfs_weights_file = 'all_pdfs_weights.pkl'
    sampler_results = 'dnest_ecc_' + group + '.pkl'
    membership_file = 'membership_probs.fits'
    
    tmp =  load_pdfs_weights_pickle(pdfs_weights_file)

    pdf_dict = tmp[0]
    wgt_dict = tmp[1]
    d1_dict = tmp[2]
    d2_dict = tmp[3]
    grp_dict = tmp[4]

    prob_mem = Table.read(membership_file)

    results = pickle.load(open(sampler_results, 'rb'))

    # Fit only stars with non-zero membership probability.
    p_thresh = 0.1
    sdx = np.where(prob_mem['p_' + group] > p_thresh)[0]
    plot_ecc_results(results, pdf_dict['e'][sdx, :], wgt_dict[group][sdx, :], group)

    return
    

def plot_ecc_results(results, pdfs, pdf_weights, suffix):
    samples = results.samples
    weights = np.exp(results.logwt - results.logz[-1])
    samples_equal = dyfunc.resample_equal(samples, weights)
    
    # results.summary()
    mean, cov = dyfunc.mean_and_cov(samples, weights)
    errors = np.diagonal(cov)**0.5

    maxL_index = results['logl'].argmax()
    maxL_params = samples[maxL_index]

    param_names = ['alpha', 'beta']
    labels = ['$\\alpha$', '$\\beta$']

    for ii in range(len(mean)):
        print('{0:5s} = {1:5.2f} +/- {2:5.2f}, maxL = {3:5.2f}'.format(param_names[ii],
                                                      mean[ii], errors[ii], maxL_params[ii]))
    plt.close('all')
    
    dyplot.runplot(results)
    plt.savefig('dnest_ecc_run_' + suffix + '.png')
    
    dyplot.traceplot(results, labels=labels)
    plt.savefig('dnest_ecc_trace_' + suffix + '.png')
    
    dyplot.cornerplot(results, labels=labels)
    plt.savefig('dnest_ecc_corner_' + suffix + '.png')

    # Make a plot of the resulting distributions.
    # Note these bins have to match what we used to make the PDFs in the first place.
    e_bin = np.arange(0, 1, 0.01)

    # Calculate the "best-fit" PDF.
    # p_e = scipy.stats.beta.pdf(e_bin, mean[0], mean[1])
    p_e = scipy.stats.beta.pdf(e_bin, maxL_params[0], maxL_params[1])

    # Make samples drawn from the posteriors.
    N_samp = 1000
    p_e_nk = np.zeros((len(e_bin), N_samp), dtype=float)
    for ss in range(N_samp):
        p_e_nk[:, ss] = scipy.stats.beta.pdf(e_bin, samples_equal[ss][0], samples_equal[ss][1])

    fix, ax = plt.subplots(2, 1, sharex=True)
    plt.subplots_adjust(hspace=0)
    
    for ss in range(N_samp):
        ax[0].plot(e_bin, p_e_nk[:, ss], 'r-', linewidth=1, alpha=0.05)

    ax[0].plot(e_bin, p_e, 'r-', linewidth=5)

    # Plot the individual star PDFs 
    e_bin_edges = np.append(e_bin, 1.0)
    e_bin_widths = np.diff(e_bin_edges)
    
    for ss in range(pdfs.shape[0]):
        
        # # instantiate and fit the KDE model
        # kde = KernelDensity(bandwidth=1e-2, kernel='gaussian')
        # kde.fit(pdfs[ss][:, None], sample_weight=pdf_weights[ss])

        # # score_samples returns the log of the probability density
        # e_bin_kde = np.arange(0, 1.0, 5e-3)
        # logprob = kde.score_samples(e_bin_kde[:, None])
        # prob = np.exp(logprob)
        # prob *= pdf_weights[ss].sum()
        
        # ax[1].plot(e_bin_kde, prob, 'k-', color='green', linewidth=2, alpha=0.5)
        
        en, eb = np.histogram(pdfs[ss], bins=e_bin_edges, weights=pdf_weights[ss], density=False)
        en /= e_bin_widths
        ax[1].plot(e_bin + (e_bin_widths/2.0), en, 'k-', linewidth=2, alpha=0.5)

    ax[1].set_xlabel('Eccentricity')
    ax[1].set_ylabel('PDF')
    ax[0].set_ylabel('PDF')

    # ax[0].set_ylim(0, 5)
    ylim1 = ax[1].get_ylim()
    ax[1].set_ylim(0, ylim1[1])

    plt.savefig('dnest_ecc_dist_' + suffix + '.png')

    return


def broken_powerlaw_trunc(a, alpha1, alpha2, a_break, a_min=2e3, a_max=2e7):
    # Use the IMF broken powerlaw we already have.
    a_limits = np.array([a_min, a_break, a_max])
    a_powers = np.array([alpha1, alpha2])

    plaw = imf.IMF_broken_powerlaw(a_limits, a_powers)
    orig_shape = a.shape
    tmp = a.ravel()
    prob_flat = plaw.xi(tmp)
    prob = prob_flat.reshape(orig_shape)

    # a_norm = a / a_break

    # amp * a_norm**(-gamma) * (1 + a_norm**delta)**((gamma - alpha) / delta)
    
    return prob

def test_broken_powerlaw_trunc():
    N_points = int(1e7)
    a_min = 2e3
    a_max = 2e7
    a_pdfs = np.linspace(a_min, a_max, N_points)

    t1 = time.time()
    p_a_nk = broken_powerlaw_trunc(a_pdfs, 1.3, 2.3, 1e6,
                                           a_min=a_min, a_max=a_max)
    t2 = time.time()

    print('runtime = {0:.1f} sec for {1:e} points'.format(t2 - t1, N_points))
    print('runtime per point = {0:.2e} sec'.format((t2 - t1) / N_points))

    return

def powerlaw_trunc(a, alpha1, a_min=2e3, a_max=2e7):
    # Use the IMF broken powerlaw we already have.
    a_limits = np.array([a_min, a_max])
    a_powers = np.array([alpha1])

    plaw = imf.IMF_broken_powerlaw(a_limits, a_powers)
    orig_shape = a.shape
    tmp = a.ravel()
    prob_flat = plaw.xi(tmp)
    prob = prob_flat.reshape(orig_shape)

    # a_norm = a / a_break

    # amp * a_norm**(-gamma) * (1 + a_norm**delta)**((gamma - alpha) / delta)
    
    return prob

class SemiMajorAxis_Solver(object):
    """
    Use methods of Hogg et al. 2010 and Bowler et al. 2020 to fit the 
    semi-major axis distribution of a population based on importance
    sampling the existing individual posteriors. 

    semi-major axes are assumed to be in AU for default priors.
    
    We will define the semi-major axis distribution using the 
    astropy BrokenPowerLaw1D distribution with free parameters for
    amplitude, alpha1, alpha2, a_break
    
    $\mathcal{L}_{params} \approx \Pi_{n=1}^{N_{stars}} \frac{1}{K} \Sigma_{k=1}^{K} f_\alpha(a_{nk}) * w_{nk}
    """
    
    default_priors = {
        'alpha1': ('make_gen', -4, 4),
        'alpha2':  ('make_gen', -4, -0.1),
        'log_a_min': ('make_gen', math.log(1e3), math.log(5e4)),
        'log_a_max': ('make_gen', math.log(1e6), math.log(1e8)),
        'log_a_break': ('make_gen', 4, 5),
        'amp': ('make_gen', 1e-3, 1e3)
    }

    def __init__(self, a_pdfs, weights, star_names, a_min=2e3, a_max=2e7):
        """
        a_pdfs: numpy array (N_stars, N_samples)
            2D array containing N_stars with N_samples of the 
            semi-major axis distribution.
            
        weights: numpy array (N_stars, N_samples)
            2D array of weights (usually from disk membership analysis)
            for each sample and each star. 
            
        star_names: numpy array or list
            List of star names with length of N_stars.
            
        Once initialized, call solve() or run dynesty using the 
        Prior() and LogLikelihood() functions.
            
        """
        
        # Data we will use in fitting.
        self.a_pdfs = a_pdfs
        self.weights = weights
        self.star_names = star_names

        # Setup parameters to fit.
        self.fitter_param_names = ['alpha1'] # , 'alpha2', 'log_a_break'] #, 'amp']
        self.n_dims = len(self.fitter_param_names)
        self.n_params = self.n_dims # nothing extra to carry.
        self.n_clustering_params = self.n_dims

        # Save some statics
        self.a_min = a_min
        self.a_max = a_max

        # Setup priors
        self.make_default_priors()
            

        return
        
    def make_default_priors(self):
        """
        Setup our prior distributions (i.e. random samplers). We will
        draw from these in the Prior() function. We set them up in advance
        because they depend on properties of the data. Also,
        they can be over-written by custom priors as desired.

        To make your own custom priors, use the make_gen() functions
        with different limits.
        """
        self.priors = {}
        for param_name in self.fitter_param_names:
            prior_type, prior_min, prior_max = self.default_priors[param_name]
            
            if prior_type == 'make_gen':
                self.priors[param_name] = make_gen(prior_min, prior_max)
                
        return

    def Prior(self, cube, ndim=None, nparams=None):
        for i, param_name in enumerate(self.fitter_param_names):
            cube[i] = self.priors[param_name].ppf(cube[i])

        return cube

    def LogLikelihood(self, cube, ndim=None, n_params=None):
        """
        This is just a wrapper because PyMultinest requires passing in
        the ndim and nparams.
        """
        alpha1 = cube[0]
        # alpha2 = cube[1]
        # a_break = 10**cube[2]
        # print('alpha1 = {0:5.2f}  alpha2 = {1:5.2f}  a_break = {2:8.2e}'.format(alpha1, alpha2, a_break))
        # amp = cube[3]

        # First, lets evaluate the probability at all of our
        # semi-major axes. We are still 2D here. (N_stars, K_samples)
        # p_a_nk = broken_powerlaw_trunc(self.a_pdfs, alpha1, alpha2, a_break,
        #                                    a_min=self.a_min, a_max=2e7)
        p_a_nk = powerlaw_trunc(self.a_pdfs, alpha1,
                                           a_min=self.a_min, a_max=2e7)
        
        # Weight by disk membership
        p_a_nk_wgt = p_a_nk * self.weights
        
        # Sum over all the samples for each star.
        # Also normalize by K
        p_a_n = p_a_nk_wgt.sum(axis=1) / self.a_pdfs.shape[1]

        zdx = np.where(p_a_n == 0)[0]
        if len(zdx) > 0:
            pdb.set_trace()
        
        # Switch into log space
        log_p_a_n = np.log(p_a_n)
                
        # Sum to get final log likelihood.
        lnL = log_p_a_n.sum()

        # print('lnL = ', lnL)
        
        return lnL


    def solve(self, n_cpu=1, pool=None):
        if (n_cpu > 1):
            self.sampler = dynesty.NestedSampler(self.LogLikelihood, self.Prior, 
                                                 ndim=self.n_dims, bound='multi',
                                                 sample='unif',
                                                 pool = pool, queue_size = n_cpu)
        else:
            self.sampler = dynesty.NestedSampler(self.LogLikelihood, self.Prior, 
                                                 ndim=self.n_dims, bound='multi',
                                                 sample='unif', nlive=1000)

        print('run nested')
        self.sampler.run_nested(print_progress=True, maxiter=3000)

        return
        

    def save(self, outfile):
        _pkl = open(outfile, 'wb')

        pickle.dump(self.sampler.results, _pkl)

        _pkl.close()

        return
    

def fit_a_dist(group, continue_run=False):
    pdfs_weights_file = 'all_pdfs_weights.pkl'
    membership_file = 'membership_probs.fits'
    sampler_results = 'dnest_a_plaw_' + group + '.pkl'
    
    tmp =  load_pdfs_weights_pickle(pdfs_weights_file)

    pdf_dict = tmp[0]
    wgt_dict = tmp[1]
    d1_dict = tmp[2]
    d2_dict = tmp[3]
    grp_dict = tmp[4]

    prob_mem = Table.read(membership_file)

    # Hard-code range to fit.
    a_min = 1e3
    a_max = 82506.0

    # Fit only stars with non-zero membership probability.
    p_thresh = 0.1
    sdx = np.where(prob_mem['p_' + group] > p_thresh)[0]
    a_solver = SemiMajorAxis_Solver(pdf_dict['a'][sdx, :], wgt_dict[group][sdx, :], prob_mem['name'][sdx],
                                    a_min=a_min, a_max=a_max)

    t0 = time.time()
    sampler = a_solver.solve()
    pdb.set_trace()
    t1 = time.time()
    print('Runtime: ', t1 - t0)

    a_solver.save(sampler_results)

    return


def plot_a_fit_results(group):
    pdfs_weights_file = 'all_pdfs_weights.pkl'
    sampler_results = 'dnest_a_plaw_' + group + '.pkl'
    membership_file = 'membership_probs.fits'
    
    tmp =  load_pdfs_weights_pickle(pdfs_weights_file)

    pdf_dict = tmp[0]
    wgt_dict = tmp[1]
    d1_dict = tmp[2]
    d2_dict = tmp[3]
    grp_dict = tmp[4]

    prob_mem = Table.read(membership_file)

    # Hard-code range to fit.
    # a_min = 1e3
    # a_max = 1e7
    a_min = 1e3
    a_max = 82506.0
    
    results = pickle.load(open(sampler_results, 'rb'))

    # Fit only stars with non-zero membership probability.
    p_thresh = 0.1
    sdx = np.where(prob_mem['p_' + group] > p_thresh)[0]
    plot_a_results(results, pdf_dict['a'][sdx, :], wgt_dict[group][sdx, :], group, a_min, a_max)

    return

def plot_a_results(results, pdfs, pdf_weights, suffix, a_min, a_max):
    samples = results.samples
    weights = np.exp(results.logwt - results.logz[-1])
    samples_equal = dyfunc.resample_equal(samples, weights)
    
    # results.summary()
    mean, cov = dyfunc.mean_and_cov(samples, weights)
    errors = np.diagonal(cov)**0.5

    maxL_index = results['logl'].argmax()
    maxL_params = samples[maxL_index]
    
    param_names = ['alpha1'] #, 'alpha2', 'log_a_break', 'amp']

    for ii in range(len(mean)):
        print('{0:5s} = {1:5.2f} +/- {2:5.2f}, maxL = {3:5.2f}'.format(param_names[ii],
                                                      mean[ii], errors[ii], maxL_params[ii]))

    plt.close('all')
    
    # dyplot.runplot(results)
    # plt.savefig('dnest_a_run_' + suffix + '.png')
    
    dyplot.traceplot(results)
    plt.savefig('dnest_a_trace_' + suffix + '.png')
    
    dyplot.cornerplot(results)
    plt.savefig('dnest_a_corner_' + suffix + '.png')

    # Make a plot of the resulting distributions.
    # Note these bins have to match what we used to make the PDFs in the first place.
    a_bin = np.logspace(3, 8, 50)
    # a_bin = np.linspace(1e3, 1e6, 100)
    a_bin_mid = a_bin[:-1] + np.diff(a_bin)

    alpha1 = mean[0]
    # alpha2 = mean[1]
    # a_break = 10**mean[2]

    # p_a = broken_powerlaw_trunc(a_bin_mid, alpha1, alpha2, a_break, a_min=a_min, a_max=a_max)   
    p_a = powerlaw_trunc(a_bin_mid, alpha1, a_min=a_min, a_max=a_max)   

    N_samp = 1000
    p_a_nk = np.zeros((len(a_bin_mid), N_samp), dtype=float)
    for ss in range(N_samp):
        # p_a_nk[:, ss] = broken_powerlaw_trunc(a_bin_mid,
        #                                           samples_equal[ss, 0],
        #                                           samples_equal[ss, 1],
        #                                           10**samples_equal[ss, 2],
        #                                           a_min=a_min, a_max=a_max)
        p_a_nk[:, ss] = powerlaw_trunc(a_bin_mid,
                                           samples_equal[ss, 0],
                                           a_min=a_min, a_max=a_max)
        
    fix, ax = plt.subplots(2, 1, sharex=True)
    plt.subplots_adjust(hspace=0)

    for ss in range(N_samp):
        ax[0].loglog(a_bin_mid, p_a_nk[:, ss], 'r-', linewidth=1, alpha=0.05)

    ax[0].loglog(a_bin_mid, p_a, 'r-', linewidth=5)

    # Plot the individual star PDFs
    a_bin_widths = np.diff(a_bin)
    
    for ss in range(pdfs.shape[0]):
        an, ab = np.histogram(pdfs[ss], bins=a_bin, weights=pdf_weights[ss], density=False)
        an /= a_bin_widths
        ax[1].loglog(a_bin_mid, an, 'k-', linewidth=2, alpha=0.5)

    # Joint PDF:
    an, ab = np.histogram(pdfs.ravel(), bins=a_bin, weights=pdf_weights.ravel(), density=False)
    an /= a_bin_widths
    ax[1].loglog(a_bin_mid, an, 'g-', linewidth=3)
    

    ax[1].set_xlabel('Semi-major Axis (AU)')
    ax[1].set_ylabel('PDF')
    ax[0].set_ylabel('PDF')

    plt.savefig('dnest_a_dist_' + suffix + '.png')

    return
