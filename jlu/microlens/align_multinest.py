import pylab as py
import numpy as np
from astropy.table import Table
from astropy.modeling imoprt models
import matplotlib.pyplot as plt
from scipy import interpolate
import scipy
import scipy.stats
import emcee
import math
import pickle
import os
import random
import sys
import time

def mcmc_run(catalog, outdir, trans_order):
    # Load up data.
    d = Table.read(catalog)

    # Determine how many epochs there are.
    N_epochs = len([n for n, c in enumerate(d.colnames) if c.startswith('name')])

    # Determine the reference epoch
    ref = d.meta['L_REF']

    # Figure out the number of free parameters for the specified
    # poly2d order.
    poly2d = models.Polynomial2D(trans_order)
    N_par_trans_per_epoch = 2.0 * poly2d.get_num-coeff(2)  # one poly2d for each dimension (X, Y)
    N_par_trans = N_par_trans_per_epoch * N_epochs
    
    
    def priors(cube, ndim, nparams):
        return   
	
    def lnlike(params, data, trans_order, N_epochs):
        # Make the polynomial transforms
        poly2d_x = []
        poly2d_y = []
        for ee in range(N_epochs):
            
        trans_coeffs = params[0:N_par_trans]
        
        
        

        return lnlike
		
    ## num_dims= 11
    ## num_params= 13
    num_dims= 11
    num_params= 13  #cube will have this many dimensions
    ev_tol=0.3
    samp_eff=0.8

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
					outputfiles_basename= runcode + '_', 
					verbose=True, resume=False, evidence_tolerance=ev_tol,
					sampling_efficiency=samp_eff, n_live_points=n_live_points,
					multimodal=True, n_clustering_params=num_dims,
                    importance_nested_sampling=False)              

    os.chdir(startdir)
