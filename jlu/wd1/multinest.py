import pylab as py
import numpy as np
from popstar import evolution, atmospheres, synthetic, reddening
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt
import pymc
from jlu.gc.imf import bayesian as by
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
import simCluster
import os
import makemodel as mm
import random
from jlu.imf import imf

defaultAge = 7.0
defaultDist = 3400
defaultAKs = 0.75
defaultFilter = (['F814W','F125W','F139M','F160W'])
defaultMassLimits = np.array([1.,10.])
defaultClusterMass = 5e4
defaultIMF = np.array([-2.3])
makeMultiples = False
count139 = False

do = 1

def make_gen(min,max):
    return scipy.stats.uniform(loc=min, scale=max-min)

def make_gen2(x,a):
    return scipy.stats.powerlaw(a=a)

def mass_generator():
    massLimits = np.array([0.5, 1, 150])
    powers = np.array([-2.3, -2.35])
    imfPrior = imf.IMF_broken_powerlaw(massLimits, powers)

    return imfPrior


dist_gen=make_gen(3000, 6000)
logAge_gen=make_gen(6.3, 7.2)
alpha_gen=make_gen(1.8, 3.0)
AKs_gen=make_gen(0.70, 0.79)
Mcl_gen=make_gen(40000, 60000)
mass_gen=make_generator()

def random_distance(x):
    dist=dist_gen.ppf(x)
    log_prob_dist=dist_gen.logpdf(dist)
    dist=round(dist/100.,0)*100.
    return dist, log_prob_dist

def random_logAge(x):
    logAge=logAge_gen.ppf(x)
    log_prob_logAge=logAge_gen.logpdf(logAge)
    logAge=round(logAge,1)
    return logAge, log_prob_logAge

def random_alpha(x):
    alpha=alpha_gen.ppf(x)
    log_prob_alpha=alpha_gen.logpdf(alpha)
    alpha=round(alpha,1)
    return alpha, log_prob_alpha

def random_AKs(x):
    AKs=AKs_gen.ppf(x)
    log_prob_AKs=AKs_gen.logpdf(AKs)
    AKs=round(AKs,2)
    return AKs, log_prob_AKs

def random_Mcl(x):
    Mcl=Mcl_gen.ppf(x)
    log_prob_Mcl=Mcl_gen.logpdf(Mcl)
    Mcl=round(Mcl/10000.,0)*10000.
    return Mcl, log_prob_Mcl

def random_mass(x):
    mass = imfPrior.imf_dice_star_cl(x)
    log_prob_mass = np.log10( imfPrior.imf_xi(mass) )

    return mass,log_prob_mass

def multinest_run(root_dir='/Users/jlu/work/wd1/analysis_2015_01_05',
                  data_tab='catalog_diffDered_NN_opt_10.fits',
                  out_dir='multinest/fit_0001'):
    
    if not os.path.exists(root_dir + out_dir):
        os.makedirs(root_dir + out_dir)

    # Input the observed data
    t = Table.read(root_dir + data_tab)

    # Some components of our model are static.
    imf_multi = multiplicity.MultiplicyUnresolved()
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
    
 
    def priors(cube, ndim, nparams):
        return   
    
    def likelihood(cube, ndim, nparams):
        ##########
        # Priors (I think order matters)
        ##########
        parName = ['distance', 'LogAge', 'AKs', 'dAKs', 'alpha1', 'alpha2', 'mbreak', 'Mcl']
        par, par_prior_logp = get_prior_info(cube, parName)

        sysMass = np.zeros(len(t))

        ##########
        # Load up the model cluster.
        ##########
        imf_mass_limits = np.array([mmin, par['mbreak'], mmax])
        imf_powers = np.array([par['alpha2'], par['alpha1']])
        imf = imf.IMF_broken_powerlaw(imf_mass_limits, imf_powers, imf_multi)
        
        iso = syn.IsochronePhot(par['LogAge'], par['AKs'], par['distance'],
                                evo_model=evo, atm_func=atm_func,
                                red_law=red_law)
        
        cluster = synthetic.ResolvedClusterDiffRedden(iso, imf, Mcl_sim, 
                                                      par['dAKs'], red_law=red_law)

        # Convert simulated cluster into magnitude-color-color histogram
        mcc_cluster = 
        

        ## Find the relationship of magnitudes as a function of mass.
        obj125=interpolate.splrep(iso.mass,iso.mag125,k=1,s=0)
        obj139=interpolate.splrep(iso.mass,iso.mag139,k=1,s=0)
        obj160=interpolate.splrep(iso.mass,iso.mag160,k=1,s=0)
        obj814=interpolate.splrep(iso.mass,iso.mag814,k=1,s=0)
        
        u125=interpolate.splev(t.mass,obj125)
        u139=interpolate.splev(t.mass,obj139)
        u160=interpolate.splev(t.mass,obj160)
        u814=interpolate.splev(t.mass,obj814)
        t.add_column('u125',u125)
        t.add_column('u139',u139)
        t.add_column('u160',u160)
        t.add_column('u814',u814)

        likei=np.log10(1./(2.*np.pi*t.mag125_e**2.)**0.5)+np.log10(np.e)*(-1.*(t.mag125-t.u125)**2./2./t.mag125_e**2.)
        +np.log10(1./(2.*np.pi*t.mag160_e**2.)**0.5)+np.log10(np.e)*(-1.*(t.mag160-t.u160)**2./2./t.mag160_e**2.)
        +np.log10(1./(2.*np.pi*t.mag814_e**2.)**0.5)+np.log10(np.e)*(-1.*(t.mag814-t.u814)**2./2./t.mag814_e**2.)

        if count139==True:
            likei+=np.log10(1./(2.*np.pi*t.mag139_e**2.)**0.5)+np.log10(np.e)*(-1.*(t.mag139-t.u139)**2./2./t.mag139_e**2.)

        print likei.sum()
        return likei.sum()

    num_dims=3+len(t)
    num_params=3+len(t)
    ev_tol=0.3
    samp_eff=0.8

    pymultinest.run(likelihood, priors, num_dims, n_params=num_params,
                    outputfiles_basename=saveto+'test',
                    verbose=True, resume=False, evidence_tolerance=ev_tol,
                    sampling_efficiency=samp_eff, n_live_points=n_live_points,
                    multimodal=True, n_clustering_params=num_dims)              
        


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
    tab = atpy.Table(root, type='ascii')

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

if do==1:
    file='MC6.7_0.76_4500.fits'
    multinest_run(fitsfile=file, n_live_points=1300, count139=False, massLimits=defaultMassLimits, 
                   saveto='/users/dhuang/work/Temporary/'+file+'_1/')

## fitsfile: the input data. 'MC6.8_0.77_5000.fits' is a synthetic cluster with logAge=6.8, Aks=0.77 and distance=5000
## count139: if we should take F139M into consideration.
## saveto  : the output address that the .txt and .dat files to be saved to.


def get_prior_info(cube, parNames):
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
                        
