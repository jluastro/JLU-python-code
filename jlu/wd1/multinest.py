import pylab as py
import numpy as np
from popstar import evolution, atmospheres, synthetic, reddening
from popstar.imf import imf, multiplicity
from astropy.table import Table
import matplotlib.pyplot as plt
from scipy import interpolate
import scipy
import scipy.stats
import pymultinest
import math
import pyfits
import pdb
import os
import pickle
import random

defaultAge = 7.0
defaultDist = 3400
defaultAKs = 0.75
defaultFilter = (['F814W','F125W','F139M','F160W'])
defaultMassLimits = np.array([1.,10.])
defaultClusterMass = 5e4
defaultIMF = np.array([-2.3])
makeMultiples = False
count139 = False

def make_gen(min,max):
    return scipy.stats.uniform(loc=min, scale=max-min)

def make_gen2(x,a):
    return scipy.stats.powerlaw(a=a)

def mass_generator():
    massLimits = np.array([0.5, 1, 150])
    powers = np.array([-2.3, -2.35])
    imfPrior = imf.IMF_broken_powerlaw(massLimits, powers)

    return imfPrior


dist_gen = make_gen(3000, 6000)
logAge_gen = make_gen(6.3, 7.2)
alpha1_gen = make_gen(-3.0, -0.5)
alpha2_gen = make_gen(-3.0, -1.0)
mbreak_gen = make_gen(0.1, 3.0)
AKs_gen = make_gen(0.70, 0.79)
dAKs_gen = make_gen(0.00, 0.10)
Mcl_gen = make_gen(40000, 60000)
#mass_gen = mass_generator()

def random_distance(x):
    dist = dist_gen.ppf(x)
    log_prob_dist = dist_gen.logpdf(dist)
    dist = round(dist/100.,0)*100.
    return dist, log_prob_dist

def random_LogAge(x):
    logAge = logAge_gen.ppf(x)
    log_prob_logAge = logAge_gen.logpdf(logAge)
    logAge = round(logAge,1)
    return logAge, log_prob_logAge

def random_alpha1(x):
    alpha1 = alpha1_gen.ppf(x)
    log_prob_alpha1 = alpha1_gen.logpdf(alpha1)
    alpha1 = round(alpha1,1)
    return alpha1, log_prob_alpha1

def random_alpha2(x):
    alpha2 = alpha2_gen.ppf(x)
    log_prob_alpha2 = alpha2_gen.logpdf(alpha2)
    alpha2 = round(alpha2,1)
    return alpha2, log_prob_alpha2

def random_mbreak(x):
    mbreak = mbreak_gen.ppf(x)
    log_prob_mbreak = mbreak_gen.logpdf(mbreak)
    mbreak = round(mbreak,1)
    return mbreak, log_prob_mbreak

def random_AKs(x):
    AKs = AKs_gen.ppf(x)
    log_prob_AKs = AKs_gen.logpdf(AKs)
    AKs = round(AKs,2)
    return AKs, log_prob_AKs

def random_dAKs(x):
    dAKs = dAKs_gen.ppf(x)
    log_prob_dAKs = dAKs_gen.logpdf(dAKs)
    dAKs = round(dAKs, 2)
    return dAKs, log_prob_dAKs

def random_Mcl(x):
    Mcl = Mcl_gen.ppf(x)
    log_prob_Mcl = Mcl_gen.logpdf(Mcl)
    Mcl = round(Mcl/10000.,0)*10000.
    return Mcl, log_prob_Mcl

# def random_mass(x):
#     mass = imfPrior.imf_dice_star_cl(x)
#     log_prob_mass = np.log10( imfPrior.imf_xi(mass) )

#     return mass,log_prob_mass

def multinest_run(root_dir='/Users/jlu/work/wd1/analysis_2015_01_05/',
                  data_tab='catalog_diffDered_NN_opt_10.fits',
                  comp_tab='completeness_ccmd.fits',
                  out_dir='multinest/fit_0001/'):
    
    if not os.path.exists(root_dir + out_dir):
        os.makedirs(root_dir + out_dir)

    # Input the observed data
    t = Table.read(root_dir + data_tab)

    # Input the completeness table and bins.
    completeness_map = pyfits.getdata(root_dir + comp_tab)
    completeness_map = completeness_map.T
    _in_bins = open(root_dir + comp_tab.replace('.fits', '_bins.pickle'), 'r')
    bins_mag = pickle.load(_in_bins)
    bins_col1 = pickle.load(_in_bins)
    bins_col2 = pickle.load(_in_bins)

    # Some components of our model are static.
    imf_multi = multiplicity.MultiplicityUnresolved()
    imf_mmin = 0.1   # msun
    imf_mmax = 150.0 # msun
    evo_model = evolution.MergedBaraffePisaEkstromParsec()
    red_law = reddening.RedLawNishiyama09()
    atm_func = atmospheres.get_merged_atmosphere
    Mcl_sim = 5.0e6


    # Our data vs. model comparison will be done in
    # magnitude-color-color space. Models will be binned
    # to construct 3D probability density spaces.
    # These are the bin sizes for the models.
    #
    # Note Dimensions:
    #   mag = m_2010_F160W
    #   col1 = m_2005_F814W - m_2010_F160W
    #   col2 = m_2010_F125W - m_2010_F160W
    #
    bins = np.array([bins_mag, bins_col1, bins_col2])
    

    def priors(cube, ndim, nparams):
        return   
    
    def likelihood(cube, ndim, nparams):
        ##########
        # Priors (I think order matters)
        ##########
        parName = ['distance', 'LogAge', 'AKs', 'dAKs',
                   'alpha1', 'alpha2', 'mbreak', 'Mcl']
        par, par_prior_logp = get_prior_info(cube, parName)

        sysMass = np.zeros(len(t))

        ##########
        # Load up the model cluster.
        ##########
        imf_mass_limits = np.array([imf_mmin, par['mbreak'], imf_mmax])
        imf_powers = np.array([par['alpha2'], par['alpha1']])
        imf_multi = None
        new_imf = imf.IMF_broken_powerlaw(imf_mass_limits, imf_powers, imf_multi)

        print 'Getting Isochrone'
        new_iso = synthetic.IsochronePhot(par['LogAge'], par['AKs'], par['distance'],
                                          evo_model=evo_model, atm_func=atm_func,
                                          red_law=red_law)
        
        print 'Getting Cluster'
        cluster = synthetic.ResolvedClusterDiffRedden(new_iso, new_imf, Mcl_sim, 
                                                      par['dAKs'], red_law=red_law)

        # Convert simulated cluster into agnitude-color-color histogram
        mag = cluster.star_systems['mag160w']
        col1 = cluster.star_systems['mag814w'] - mag
        col2 = cluster.star_systems['mag125w'] - mag

        data = np.array([mag, col1, col2]).T
        bins = np.array([bins_mag, bins_col1, bins_col2])

        H_sim_c, edges = np.histogramdd(data, bins=bins, normed=True)
        H_sim = H_sim_c * completeness_map
        
        # Convert Observed cluster into magnitude-color-color histogram
        mag = t['m_2010_F160W']
        col1 = t['m_2005_F814W'] - t['m_2010_F160W']
        col2 = t['m_2010_F125W'] - t['m_2010_F160W']
        
        data = np.array([mag, col1, col2]).T
        bins = np.array([bins_mag, bins_col1, bins_col2])

        H_obs, edges = np.histogramdd(data, bins=bins)

        # Plotting
        extent = (bins_col1[0], bins_col2[-1], bins_mag[0], bins_mag[-1])
        py.figure(1)
        py.clf()
        py.imshow(H_sim_c.sum(axis=2), extent=extent)
        py.gca().invert_yaxis()
        py.colorbar()
        py.axis('tight')
        py.title('Sim Complete')

        py.figure(2)
        py.clf()
        py.imshow(H_sim.sum(axis=2), extent=extent)
        py.gca().invert_yaxis()
        py.colorbar()
        py.axis('tight')
        py.title('Sim Incomplete')
        
        py.figure(3)
        py.clf()
        py.imshow(H_obs.sum(axis=2), extent=extent)
        py.gca().invert_yaxis()
        py.colorbar()
        py.axis('tight')
        py.title('Obs Incomplete')

        py.figure(4)
        py.clf()
        py.imshow(completeness_map.mean(axis=2), extent=extent,
                  vmin=0, vmax=1)
        py.gca().invert_yaxis()
        py.colorbar()
        py.axis('tight')
        py.title('Completeness Map')
                
        pdb.set_trace()
        
        mcc_cluster = 1


        print likei.sum()
        return likei.sum()

    num_dims = 8
    num_params = 8
    ev_tol = 0.3
    samp_eff = 0.8
    n_live_points = 300

    # pymultinest.run(likelihood, priors, num_dims, n_params=num_params,
    #                 outputfiles_basename=out_dir + 'test',
    #                 verbose=True, resume=False, evidence_tolerance=ev_tol,
    #                 sampling_efficiency=samp_eff, n_live_points=n_live_points,
    #                 multimodal=True, n_clustering_params=num_dims)
    cube_test = [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]
    likelihood(cube_test, num_dims, num_params)
        


def plot_results_detail(rootdir):
    res = load_results(rootdir)

    poltStuff = (('distance', res.distance),
                 ('logAge', res.logAge),
                 ('Aks', res.Aks),
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

def load_results(rootdir):
    root = '%s' % (rootdir)
    tab = Table(root, type='ascii')

    # Convert to log(likelihood)
    tab['col2'] /= -2.0
    
    # Rename the parameter columns. This is hard-coded to match the
    # above run() function.
    tab.rename_column('col1', 'weights')
    tab.rename_column('col2', 'logLike')
    tab.rename_column('col3', 'distance')
    tab.rename_column('col4', 'logAge')
    tab.rename_column('col5', 'Aks')
#    tab.rename_column('col6', 'alpha')
#    tab.rename_column('col7', 'Mcl')


    # Now sort based on logLikelihood
    tab.sort('logLike')

    return tab    

def get_prior_info(cube, parName):
    """Get parameter and prior probability from prior functions.
    """
    par = {}
    par_prior_logp = {}
    
    # Loop through each free parameter and convert our cube random number
    # into a random selection from the prior function and its prior probability.
    # Save everything into a dictionary (par and par_prior_logp) accessible with
    # the parameter names (parName).
    for ii in range(len(parName)):
        prior_function = globals()['random_' + parName[ii]]

        par_tmp, log_prob_par_tmp = prior_function(cube[ii])

        cube[ii] = par_tmp

        par[parName[ii]] = par_tmp
        par_prior_logp[parName[ii]] = log_prob_par_tmp

    # Round some priors
    par['distance'] = int(par['distance'])
                        
    return par, par_prior_logp


    
