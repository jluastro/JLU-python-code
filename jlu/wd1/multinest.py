import pylab as py
import numpy as np
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
import getmass
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

# The completeness file need to be changed later.
def multinest_run(fitsfile='4784.fits', n_live_points=1000, count139=False,
                  massLimits=defaultMassLimits, 
                  saveto='/users/dhuang/work/Temporary/'):

    if not os.path.exists(saveto):
        os.makedirs(saveto)
    
    ## input our data.
    t = atpy.Table(fitsfile)
 
    def priors(cube, ndim, nparams):
        return   
    
    def likelihood(cube, ndim, nparams):
        # Distance:
        distance, log_prob_distance = random_distance(cube[0])
        cube[0]=distance
        distance=int(distance)

        # Age:
        LogAge, log_prob_LogAge = random_logAge(cube[1])
        cube[1] = LogAge

        # AKs:
        AKs, log_prob_AKs = random_AKs(cube[2])
        cube[2] = AKs

        # alpha:
#        alpha, log_prob_alpha = random_alpha(cube[3])
#        cube[3] = alpha
        # Mcl:
#        Mcl, log_prob_Mcl = random_Mcl(cube[4])
#        cube[4] = Mcl

        # Mass[i]:
        mass, log_prob_mass = random_mass(cube[3:3+len(t)])

        sysMass = np.zeros(len(t))
        
        for i in range(len(t)):
            idx = int(cube[i+3]*len(masscc))
            cube[3+i]=masscc[idx]
        print cube[4], cube[5], cube[500]


    cc = by.imf.sample_imf(massLimits, imfSlopes=defaultIMF,
                           totalMass=defaultClusterMass, makeMultiples=makeMultiples)     
    masscc = cc[0]



        ## generate an isochrone
        iso=mm.ModelTable(LogAge, AKs, distance, saveto='/users/dhuang/work/model/modelfitsfile/')

##-----------------------------------------------------------------
## Mass could not be a free parameter, i think....it should be:
#        ## Find the relationship of mass as a function of magnitudes. Try to estimate the mass of our data.
#        objmass125=interpolate.splrep(iso.mag125,iso.mass,k=1,s=0)   ####!!!! splrep(x,y) and splrep(y,x) 
#        objmass139=interpolate.splrep(iso.mag139,iso.mass,k=1,s=0)
#        objmass160=interpolate.splrep(iso.mag160,iso.mass,k=1,s=0)
#        objmass814=interpolate.splrep(iso.mag814,iso.mass,k=1,s=0)
#        mass125=interpolate.splev(t.mag125,objmass125,k=1,s=0)
#        mass139=interpolate.splev(t.mag139,objmass139,k=1,s=0)
#        mass160=interpolate.splev(t.mag160,objmass160,k=1,s=0)
#        mass814=interpolate.splev(t.mag814,objmass814,k=1,s=0)
#        meanMass=(mass125+mass160+mass814)/3.
#        sigmaMass=(((mass125-meanMass)**2.+(mass160-meanMass)**2.+(mass814-meanMass)**2.)/3.)**0.5
#        if count139==True:
#            meanMass=(meanMass*3.+mass139)/4.
#            sigmaMass=(((mass125-meanMass)**2.+(mass139-meanMass)**2.+(mass160-meanMass)**2.+(mass814-meanMass)**2.)/4.)**0.5
#        mass=np.zeros(len(t))
#        for i in range(len(t)):
#            mass[i]=pymc.TruncateNormal('mass',meanMass[i],1.0/sigmaMass**2.)
#        t.add_column('mass',mass)
##------------------------------------------------------------------

        t.add_column('mass',mass)
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
