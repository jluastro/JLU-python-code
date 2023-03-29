import numpy as np
import pylab as plt
from jlu.gc.imf import pop_fitter
from dynesty import plotting as dyplot
from dynesty import utils as dyfunc
from astropy.table import Table
from sklearn.neighbors import KernelDensity
from scipy import stats
import pickle

work_dir = '/u/jlu/work/gc/imf/dynamics/'

def plot_ecc_corner():
    """
    Plot the eccentricity distributions and alpha, beta posteriors
    all together for disk1, disk2 (plane2), and non-disk.
    """

    # Load up the huge set of i, omega samples for all
    # the stars.
    pdfs_weights_file = work_dir + 'all_pdfs_weights.pkl'
    tmp = pop_fitter.load_pdfs_weights_pickle(pdfs_weights_file)

    pdf_dict = tmp[0]
    wgt_dict = tmp[1]
    d1_dict = tmp[2]
    d2_dict = tmp[3]
    grp_dict = tmp[4]

    # Load up the membership probabilities. 
    membership_file = work_dir + 'membership_probs.fits'
    prob_mem = Table.read(membership_file)

    # Define the groups and load up the sampler results for
    # each one. 
    groups = ['d2', 'nd', 'd1']
    # groups = ['d2', 'nd']
    ecc_results = {}

    # Load up the sampler results for all through groups.
    for group in groups:
        sampler_results_file = work_dir + 'dnest_ecc_' + group + '.pkl'
        _in = open(sampler_results_file, 'rb')
        ecc_results[group] = pickle.load(_in)
        _in.close()

    # Make a plot of the alpha-beta corners with all three shown.
    plt.close(1)
    foo = plt.subplots(2, 2, figsize=(6, 6), num=1)

    colors = {'d1': 'red', 'd2': 'blue', 'nd': 'grey'}

    for group in groups:
        print('')
        print('*** Results for ', group)
        results = ecc_results[group]
        samples = results.samples
        weights = np.exp(results.logwt - results.logz[-1])
        samples_equal = dyfunc.resample_equal(samples, weights)

        try:
            results.nlive
        except AttributeError:
            results.nlive = results.batch_nlive[-1]
    
        results.summary()
        mean, cov = dyfunc.mean_and_cov(samples, weights)
        errors = np.diagonal(cov)**0.5

        param_names = ['alpha', 'beta']

        for ii in range(len(mean)):
            print('{0:5s} = {1:5.2f} +/- {2:5.2f}'.format(param_names[ii],
                                                          mean[ii], errors[ii]))
                                                      
        dyplot.cornerplot(results, fig=foo, labels=[r'$\alpha$', r'$\beta$'],
                          color=colors[group], quantiles=None)

    # Make a legend.
    plt.text(0.65, 0.75, 'disk1', color=colors['d1'],
                 fontsize=18, fontweight='bold',
                 transform=plt.gcf().transFigure)
    plt.text(0.65, 0.7, 'plane2', color=colors['d2'],
                 fontsize=18, fontweight='bold',
                 transform=plt.gcf().transFigure)
    plt.text(0.65, 0.65, 'other', color=colors['nd'],
                 fontsize=18, fontweight='bold',
                 transform=plt.gcf().transFigure)
    
    print('Existing axis limits: ')
    print('axes: 0, 0')
    print('    X = ', foo[1][0, 0].get_xlim())
    print('    Y = ', foo[1][0, 0].get_ylim())
    print('axes: 0, 1')
    print('    X = ', foo[1][0, 1].get_xlim())
    print('    Y = ', foo[1][0, 1].get_ylim())
    print('axes: 1, 0')
    print('    X = ', foo[1][1, 0].get_xlim())
    print('    Y = ', foo[1][1, 0].get_ylim())
    print('axes: 1, 1')
    print('    X = ', foo[1][1, 1].get_xlim())
    print('    Y = ', foo[1][1, 1].get_ylim())

    axgrid = foo[1]
    
    axgrid[1, 0].set_xlim(5e-1, 50)
    axgrid[0, 0].set_xlim(5e-1, 50)

    axgrid[1, 0].set_ylim(5e-1, 50)
    axgrid[1, 1].set_xlim(5e-1, 50)
    
    axgrid[0, 0].set_ylim(0, 0.013)
    axgrid[1, 1].set_ylim(0, 0.013)

    axgrid[0, 0].set_xscale('log')
    axgrid[1, 1].set_xscale('log')
    axgrid[1, 0].set_xscale('log')
    axgrid[1, 0].set_yscale('log')

    plt.savefig(work_dir + 'fig_ecc_corner.png')

    return
        
def plot_ecc_dist():
    groups = ['d1', 'd2', 'nd']
    titles = {'d1': 'disk1', 'd2': 'plane2', 'nd': 'other'}
    colors = {'d1': 'red', 'd2': 'blue', 'nd': 'grey'}
    colors_d = {'d1': 'darkred', 'd2': 'darkblue', 'nd': 'black'}

    ylim = {'d1': [0, 5], 'd2': [0, 9], 'nd': [0, 3]}

    pdfs_weights_file = work_dir + 'all_pdfs_weights.pkl'
    membership_file = work_dir + 'membership_probs.fits'
    
    # Load invdividual PDFs
    tmp =  pop_fitter.load_pdfs_weights_pickle(pdfs_weights_file)
    pdf_dict = tmp[0]
    wgt_dict = tmp[1]
    d1_dict = tmp[2]
    d2_dict = tmp[3]
    grp_dict = tmp[4]

    prob_mem = Table.read(membership_file)

    # Load up the sampler results for all through groups.
    for group in groups:
        print('Working on ', group)
        
        # PDFs and weights:
        # Include only stars with non-zero membership probability.
        p_thresh = 0.1
        sdx = np.where(prob_mem['p_' + group] > p_thresh)[0]
        pdfs = pdf_dict['e'][sdx, :]
        pdf_weights = wgt_dict[group][sdx, :]
        
        # Load sample results
        sampler_results_file = work_dir + 'dnest_ecc_' + group + '.pkl'

        _in = open(sampler_results_file, 'rb')
        ecc_results = pickle.load(_in)
        _in.close()
        
        results = ecc_results
        samples = results.samples
        weights = np.exp(results.logwt - results.logz[-1])
        samples_equal = dyfunc.resample_equal(samples, weights)
    
        maxL_index = results['logl'].argmax()
        maxL_params = samples[maxL_index]

        # Calculate the best-fit (maxL) distributions.
        e_bin = np.arange(0, 1, 0.01)
        p_e = stats.beta.pdf(e_bin, maxL_params[0], maxL_params[1])

        # Make samples drawn from the posteriors.
        N_samp = 1000
        p_e_nk = np.zeros((len(e_bin), N_samp), dtype=float)
        for ss in range(N_samp):
            p_e_nk[:, ss] = stats.beta.pdf(e_bin, samples_equal[ss][0], samples_equal[ss][1])

        #####
        # Plot
        #####
        fig = plt.figure(1)
        plt.clf()
        fix, ax = plt.subplots(2, 1, sharex=True, num=1)
        plt.subplots_adjust(hspace=0, left=0.15)
    
        for ss in range(N_samp):
            ax[0].plot(e_bin, p_e_nk[:, ss], 'r-', color=colors[group],
                           linewidth=1, alpha=0.05)

        ax[0].plot(e_bin, p_e, 'r-', color=colors_d[group],
                       linewidth=5)

        # Plot the individual star PDFs 
        e_bin_edges = np.append(e_bin, 1.0)
        e_bin_widths = np.diff(e_bin_edges)
    
        for ss in range(pdfs.shape[0]):
        
            # instantiate and fit the KDE model
            kde = KernelDensity(bandwidth=1e-2, kernel='gaussian')
            kde.fit(pdfs[ss][:, None], sample_weight=pdf_weights[ss])

            # score_samples returns the log of the probability density
            e_bin_kde = np.arange(0, 1.0, 5e-3)
            logprob = kde.score_samples(e_bin_kde[:, None])
            prob = np.exp(logprob)
            prob *= pdf_weights[ss].sum()
        
            ax[1].plot(e_bin_kde, prob, 'k-', color='grey', linewidth=2, alpha=0.5)
        
        ax[1].set_xlabel('Eccentricity')
        ax[1].set_ylabel('PDF')
        ax[0].set_ylabel('PDF')
        ax[0].set_title(titles[group])

        # ax[0].set_ylim(0, 5)
        ylim1 = ax[1].get_ylim()
        ax[1].set_ylim(0, ylim1[1])
        ax[1].set_xlim(0, 1)
        ax[0].set_ylim(ylim[group][0], ylim[group][1])

        plt.savefig(work_dir + 'fig_dnest_ecc_dist_' + group + '.png')

    return

