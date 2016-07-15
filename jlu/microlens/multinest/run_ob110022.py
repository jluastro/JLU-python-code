import pylab as py
import numpy as np
from astropy.table import Table
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt
from scipy import interpolate
import scipy
import scipy.stats
import pymultinest
import math
import pickle
import pdb
import os
import random
import asciidata
import sys
from jlu.microlens import MCMC_LensModel
import time
import copy

starttime = time.time()
'''
#############################################
PRIORS DEFINED HERE
#############################################
'''
def make_gen(min,max):
    return scipy.stats.uniform(loc=min, scale=max-min)

def make_Gauss(mu,sig):
    return scipy.stats.norm(loc=mu, scale=sig)

# USER INPUT
target = 'ob110022'
rootdir = '/Users/jlu/work/microlens/2015_evan/'
pointsdir = 'points_d/' 
mcmcPhotDir = 'photometry/posteriors/'
mcmcPhotFile = 'ob110022_mcmc.dat'

# O=1, Kcut=22
analysisdir = 'analysis_ob110022_2014_03_22al_MC100_omit_1/'
runcode = 'bf'
# O=2, Kcut=22
# analysisdir = 'analysis_ob110022_2014_03_22ax_MC100_omit_1/'   
# runcode = 'bc'

savedir = 'multiNest/' + runcode + '/' 
MakePhotPriors = False
MultiDimPriors = True
Nbins = 40

if MultiDimPriors == True:
    PhotPriorsFile = 'PhotPriors_MultiDim_' + target + '.txt'
else:
    PhotPriorsFile = 'PhotPriors_' + target + '.txt'
               
dpix=10.0 
pixScale = 9.952
mu_mid = 0.0 # mas/yr
dmu = 40. # mas/yr
PhotparNames = ['t0', 'beta', 'tE', 'piEN', 'piEE']
#END OF USER INPUT

if not os.path.exists(rootdir + analysisdir):
	print 'Error: Path does not exist: ', rootdir + analysisdir 
	sys.exit(0)
   
multiNest_saveto = rootdir + analysisdir + savedir
photpriors_saveto = rootdir + mcmcPhotDir

nPhotpars = len(PhotparNames)

if not os.path.exists(multiNest_saveto):
   os.makedirs(multiNest_saveto)


# READ IN ASTROMETRIC DATA
print 'run_ob110022_err_x2: Read in Astrometric Data'
if target == 'ob120169':
    pointsFile = rootdir + analysisdir + pointsdir + target+'_R' + '.points'
else:
    pointsFile = rootdir + analysisdir + pointsdir + target + '.points'
if os.path.exists(pointsFile + '.orig'):
	pointsTab = Table.read(pointsFile + '.orig', format='ascii')
else:
	pointsTab = Table.read(pointsFile, format='ascii')
	
tobs = pointsTab['col1']
xpix = pointsTab['col2']
ypix = pointsTab['col3']
xerr = pointsTab['col4']
yerr = pointsTab['col5']

Npts = len(tobs)

thetaSx_data = xpix*pixScale
thetaSy_data = ypix*pixScale
xerr_data = xerr*pixScale
yerr_data = yerr*pixScale



# thetaS_data = np.array([[[xpix[i],ypix[i]] for i in range(Npts)]])
# err_data = np.array([[[xerr[i],yerr[i]] for i in range(Npts)]])  
# 
# thetaSx_data = thetaS_data[:,0]
# thetaSy_data = thetaS_data[:,1]
# xerr_data = err_data[:,0]
# yerr_data = err_data[:,1]

       
#SET ORIGIN OF COORD SYS
print 'run_ob110022_err_x2: Set Origin of Coord Sys'
orig_x_pix, orig_y_pix  = np.mean(xpix), np.mean(ypix)
print 'run_ob110022_err_x2:     ', orig_x_pix, orig_y_pix

# SET REASONABLE BOUNDS ON UNCONSTRAINED PARAMETERS
# 1. source position at t0 (xpix0, ypix0) will be confined to box dpix by dpix  
thetaS0x_min, thetaS0x_max = (orig_x_pix - dpix/2.0)*pixScale, (orig_x_pix + dpix/2.0)*pixScale 
thetaS0y_min, thetaS0y_max = (orig_y_pix - dpix/2.0)*pixScale, (orig_y_pix + dpix/2.0)*pixScale

# 2. Unlensed source proper motion, will be confined to box dmu by dmu centered at mu_mid 
muSx_min, muSy_min = mu_mid-dmu/2., mu_mid - dmu/2.
muSx_max, muSy_max = mu_mid+dmu/2., mu_mid + dmu/2.

# 3. Source-lens relative proper motion, will be confined to box 2*dmu by 2*dmu centered at mu_mid
muRelx_min, muRely_min = mu_mid-dmu, mu_mid-dmu
muRelx_max, muRely_max = mu_mid+dmu, mu_mid+dmu

print 'run_ob110022_err_x2: Make Random Number Generators'
thetaS0x_gen = make_gen(thetaS0x_min, thetaS0x_max)
thetaS0y_gen = make_gen(thetaS0y_min, thetaS0y_max)
muSx_gen = make_gen(muSx_min, muSx_max)
muSy_gen = make_gen(muSy_min, muSy_max)
muRelx_gen = make_gen(muRelx_min, muRelx_max)
muRely_gen = make_gen(muRely_min, muRely_max)


def MultiDimPrior_Gen(intable):
    Nrows = len(intable)
    Ncols = len(intable.colnames)

    # Set bin-edges for each column in the table.
    bins = []
    for i in range(Ncols):
        bins.append(np.linspace(intable[:,i].min(), intable[:,i].max(), Nbins))

    #multidim histogram
    H, edges = np.histogramdd(intable, bins=bins)
    Hnorm = H/np.sum(H)

    #indices where histogram != 0
    indpdf = np.transpose(np.array((Hnorm).nonzero()))
    
    ## binmids = np.empty([Ncols, Nbins-1], dtype='float')
    ## for i in range(Ncols): binmids[i,:] = (bins[i][0:Nbins-1]+bins[i][1:Nbins])/2.

    ## pdfflat = pdf.flatten()
    ## print len(pdfflat)
    ## print lkqsl
    Nrows = len(indpdf[:,0])
    cdf = np.zeros(Nrows, dtype='float')
    pdf = np.zeros(Nrows, dtype='float')

    for i in range(Ncols): print np.transpose(np.nonzero(Hnorm != 0))[i]

    # 1D cdf and pdf using only non-zero bins    
    for j in range(Nrows):
        print str(j)+'/' + str(Nrows)
        cdf[j] = cdf[j-1] + Hnorm[tuple(indpdf[j,:])]
        pdf[j] = Hnorm[tuple(indpdf[j,:])]
    indmax = np.where(pdf == np.max(pdf))[0]
    print indpdf[indmax,:]

    plotbins=[]
    plotslice = np.empty([Nrows, Nbins-1])
    indpdftemp = copy.copy(indpdf)
    for i in range(Ncols):
        # plot slice through dimension i
        #keep = [x for x in range(Ncols) if x != i]
        #print keep
        #print indpdf[indmax[0],keep]
        plotbins.append(bins[i][0:-1])
        for j in range(Nbins-1):
           indpdftemp = copy.copy(indpdf) 
           indpdftemp[indmax[0],i]=j
           print indpdftemp[indmax[0]]
           plotslice[i,j] = H[tuple(indpdftemp[indmax[0]])]
           #plotslice.append(H[tuple(indpdf[indmax[0],keep])])
    
        
   
    py.clf()
    fig3 = py.figure(figsize=[10,12])
    for i in range(Ncols):
        ax = fig3.add_subplot(3, 2, i+1)
        py.xticks(fontsize=14)
        py.yticks(fontsize=14)
        py.xlim(plotbins[i].min(), plotbins[i].max()) 
        py.bar(plotbins[i], plotslice[i,:], (plotbins[i][1]-plotbins[i][0])*0.8)  #bx = bin edges
    ax = fig3.add_subplot(3, 2, Ncols+1)     
    nx, bx, px = py.hist(np.sum(H)*pdf.flatten(), 40, ec='b', histtype='step', linewidth=3)
    py.xticks(fontsize=14)
    py.yticks(fontsize=14)
    #print np.shape(cdf)
    #py.clf()
    #py.plot(range(len(cdf)), cdf)
    py.savefig(rootdir + analysisdir + 'plots/testhist_Nbins'+ str(int(Nbins)) + '.pdf')
    py.clf()
    
    savearr = indpdf[:,0].astype(int)
    for i in range(Ncols-1):savearr = np.column_stack((savearr.astype(int), indpdf[:,i+1]))
    savearr = np.column_stack((pdf, cdf, savearr))  
    
    # bins = bin edges, savearr columns are normalized pdf, cdf, index[0], index[1], index[2]...
    # where indices are multi-dimensional bin indices    
    return bins, savearr  


    
def GenPhotPriors():
	# READ IN PHOTOMETRICALLY DERIVED POSTERIORS
    parnames = Photpars['parnames']
    # print parnames
	
    # Column numbers in mcmcPhotFile corresponding to the photometric parameters parnames
    if target == 'ob110022':  
        idx = [10, 11, 12, 16, 17] 
    elif target == 'ob110125':  
        idx = [8,9,10,14,15]
    elif target == 'ob120169':  
        idx = [5,6,7,8,9]
    else: 
        print 'Error: Unrecognized target name'
        sys.exit(0)
        
    # Get MCMC chains for each parameter
    mcmcPhotTab = Table.read(rootdir + mcmcPhotDir + mcmcPhotFile, format='ascii')
    t0temp = mcmcPhotTab[mcmcPhotTab.colnames[idx[0]]]
    
    # Convert HJD-245000 into calendar year using Jan 1 2009 as calibrator
    t0calYr = 2009. + (t0temp - 4832.5)/365.
    leapYr = np.where((t0temp > 5927.5) & (t0temp < 6293.) )[0]
    t0calYr[leapYr] =  2009. + (t0temp[leapYr] - 4833.) / 366.
    
    #GENERATE PDFs, CDFs AND SAVE TO FILE
    Ncols = len(mcmcPhotTab.colnames)
    Nrows = len(t0temp)
    intable = np.empty([Nrows, len(idx)])
    for i in range(len(idx)):
        intable[:, i] = mcmcPhotTab[mcmcPhotTab.colnames[idx[i]]]
    intable[:, 0] = t0calYr
    
    if MultiDimPriors == True:
        for i in range(nPhotpars):   
            Photpars[parnames[i]] = intable[i]
            
        # bins = bin edges, savearr columns are normalized
        #    pdf, cdf, index[0], index[1], index[2]...
        # where indices are multi-dimensional bin indices.            
        bins, savearr = MultiDimPrior_Gen(intable)  
        np.savetxt(photpriors_saveto + PhotPriorsFile, savearr, fmt='%f')
        np.savetxt(photpriors_saveto + PhotPriorsFile[0:-4] + '_bins.txt', np.column_stack(bins), fmt='%f')
        bins = asciidata.open(photpriors_saveto + PhotPriorsFile[0:-4] + '_bins.txt')
        Photpars['MultiDimPrior_Bins'] = np.array(bins)
        Photpars['MultiDimPrior'] = np.array(savearr)
        #temp = Photpars['MultiDimPrior'][:,-nPhotpars:]
        
        
    else:
        for i in range(nPhotpars):
            Photpars[parnames[i]] = mcmcPhotTab[idx[i]].tonumpy()	
            temp = Photpars[parnames[i]]
            
            bins = np.linspace(temp.min(), temp.max(), Nbins)  #bin edges
            binstep = bins[1]-bins[0]
            fig3 = py.figure(figsize=[8,6])
            ax = fig3.add_subplot(2, 1, 1)
            (nx, bx, px) = ax.hist(temp, bins)  #bx = bin edges
            nx = np.array(nx, dtype=float)
            temppdf = nx/nx.sum()  #Correct normalization?
            binmids = (bins[0:Nbins-1]+bins[1:Nbins])/2.
            tempcdf = np.zeros(Nbins-1, dtype=float)   #cdf 
            for j in range(Nbins-1):
                tempcdf[j] = temppdf[0:j+1].sum()
            Photpars[parnames[i]] = np.column_stack((binmids,temppdf, tempcdf))
            if  i == 0: 
                savearr = np.column_stack((binmids,temppdf, tempcdf))      
            else: 
                savearr = np.column_stack((savearr,binmids,temppdf, tempcdf))

        np.savetxt(photpriors_saveto + PhotPriorsFile, savearr, fmt='%f') 
    return Photpars
	

def ReadPhotPriors():
    parnames = Photpars['parnames']
    
    if MultiDimPriors == True:
        Photpars['MultiDimPrior']  = np.loadtxt(photpriors_saveto + PhotPriorsFile)
        Photpars['MultiDimPrior_Bins']  = np.loadtxt(photpriors_saveto + PhotPriorsFile[0:-4] + '_bins.txt')
    else:
        data = np.loadtxt(photpriors_saveto + PhotPriorsFile)
        for i in range(nPhotpars):
            binmids = data[data.colnames[3*i]]
            temppdf = data[data.colnames[3*i+1]]
            tempcdf = data[data.colnames[3*i+2]]
            Photpars[parnames[i]] = np.column_stack((binmids,temppdf,tempcdf))
    return Photpars

def GetPhotPriors():
    if MakePhotPriors == True:
        PhotPriors = GenPhotPriors()
        parnames = Photpars['parnames']
    else:
	    PhotPriors = ReadPhotPriors()
	    parnames = Photpars['parnames']
	    # for i in range(len(parnames)):
	    #     print parnames[i]
	    #     print PhotPriors[parnames[i]]
    return PhotPriors

def makedict_PhotPars():
    Photpars = {}
    Photpars['parnames'] = PhotparNames
    print Photpars['parnames']
    for i in range(nPhotpars):
        Photpars.update({PhotparNames[i]:np.array([0,1])})
    return Photpars

# GET PRIORS FROM PHOTOMETRY USING SUBO'S MCMC POSTERIORS
print 'run_ob110022_err_x2: Get Photometric Prior Distributions'
Photpars = makedict_PhotPars()
PhotPriors = GetPhotPriors()



def random_photpar(x,bins,pdf,cdf):  #array of columns (bins, pdf, cdf) #bins are bin mids
    if len(np.where(cdf > x)[0]) == 0: 
        idx = 0
    else:
        idx = np.min(np.where(cdf > x)[0])
    t0 = bins[idx] + np.random.uniform()*(bins[1]-bins[0])
    ln_prob_t0 = np.log(pdf[idx])
    return t0, ln_prob_t0


def random_photpar_multidim(x,bins,pdf,cdf,bininds):  #array of columns (bins, pdf, cdf) #bins are bin mids
    if len(np.where(cdf > x)[0]) == 0: 
        idx = 0
    else:
        idx = np.min(np.where(cdf > x)[0])
    inds = bininds[idx]

    pars = np.empty(nPhotpars, dtype=float)

    for i in range(nPhotpars):
        pars[i] = bins[inds[i], i]
        pars[i] += np.random.uniform() * (bins[1, i] - bins[0, i])
        
    ln_prob_pars = np.log(pdf[idx])
    return pars, ln_prob_pars    
    
	
#thetaS0x = initial x position of source (mas)
def random_thetaS0x(x):
    thetaS0x=thetaS0x_gen.ppf(x)
    ln_prob_thetaS0x=thetaS0x_gen.logpdf(thetaS0x)
    return thetaS0x, ln_prob_thetaS0x

#thetaS0y = initial y position of source (mas)
def random_thetaS0y(x):
    thetaS0y=thetaS0y_gen.ppf(x)
    ln_prob_thetaS0y=thetaS0y_gen.logpdf(thetaS0y)
    return thetaS0y, ln_prob_thetaS0y

# Source proper motion (x dimension)
def random_muSx(x):
    muSx=muSx_gen.ppf(x)
    ln_prob_muSx=muSx_gen.logpdf(muSx)
    return muSx, ln_prob_muSx

# Source proper motion (y dimension)
def random_muSy(y):
    muSy=muSy_gen.ppf(y)
    ln_prob_muSy=muSy_gen.logpdf(muSy)
    return muSy, ln_prob_muSy

# Source-lens relative proper motion (x dimension) 
def random_muRelx(x):
    muRelx=muRelx_gen.ppf(x)
    ln_prob_muRelx=muRelx_gen.logpdf(muRelx)
    return muRelx, ln_prob_muRelx
 
# Source-lens relative proper motion (y dimension)  
def random_muRely(x):
    muRely=muRely_gen.ppf(x)
    ln_prob_muRely=muRely_gen.logpdf(muRely)
    return muRely, ln_prob_muRely 
 
 
#FOR TESTING WITH UNIFORM PHOT PRIORS       
def random_t0(x):
    t0=t0_gen.ppf(x)
    ln_prob_t0=t0_gen.logpdf(t0)
    return t0, ln_prob_t0
        
def random_tE(x):
    tE=tE_gen.ppf(x)
    ln_prob_tE=tE_gen.logpdf(tE)
    return tE, ln_prob_tE

def random_beta(x):
    beta=beta_gen.ppf(x)
    ln_prob_beta=beta_gen.logpdf(beta)
    return beta, ln_prob_beta
    
def random_piEN(x):
    piEN=piEN_gen.ppf(x)
    ln_prob_piEN=piEN_gen.logpdf(piEN)
    return piEN, ln_prob_piEN    
         
def random_piEE(x):
    piEE=piEE_gen.ppf(x)
    ln_prob_piEE=piEE_gen.logpdf(piEE)
    return piEE, ln_prob_piEE  

parnames = Photpars['parnames']

# for i in range(nPhotpars):
#     print parnames[i]
#     print Photpars[parnames[i]]
    

# The completeness file need to be changed later.
def multinest_run(n_live_points=1000, 
                  target = 'ob110022', saveto=multiNest_saveto):
	
	
    def priors(cube, ndim, nparams):
        return   
	
    def likelihood(cube, ndim, nparams):

        lnlikePhot = 0.0
        lnlike_nonPhot = 0.0
        parnames = Photpars['parnames']
        
        # Photometric params
        if MultiDimPriors == True:
            params, lnlikePhot = random_photpar_multidim(cube[0],
                                                         Photpars['MultiDimPrior_Bins'],
                                                         Photpars['MultiDimPrior'][:,0],
                                                         Photpars['MultiDimPrior'][:,1],
                                                         Photpars['MultiDimPrior'][:, -nPhotpars:])
            for i in range(nPhotpars): cube[i] = params[i]
        else:
            for i in range(nPhotpars):
                param, ln_prob_param = random_photpar(cube[i],
                                                      Photpars[parnames[i]][:,0],
                                                      Photpars[parnames[i]][:,1],
                                                      Photpars[parnames[i]][:,2])
                cube[i]=param
                lnlikePhot += ln_prob_param

        idx = nPhotpars

        # x Position at t0:
        thetaS0x, ln_prob_thetaS0x = random_thetaS0x(cube[idx])
        cube[idx] = thetaS0x
        idx += 1
        lnlike_nonPhot += ln_prob_thetaS0x

        # y Position at t0:
        thetaS0y, ln_prob_thetaS0y = random_thetaS0y(cube[idx])
        cube[idx] = thetaS0y
        idx += 1
        lnlike_nonPhot += ln_prob_thetaS0y

        # Source proper motion (x dimension)
        muSx, ln_prob_muSx = random_muSx(cube[idx])
        cube[idx] = muSx
        idx += 1
        lnlike_nonPhot += ln_prob_muSx

        # Source proper motion (y dimension)
        muSy, ln_prob_muSy = random_muSy(cube[idx])
        cube[idx] = muSy
        idx += 1
        lnlike_nonPhot += ln_prob_muSy

        # Source-lens relative proper motion (x dimension)
        muRelx, ln_prob_muRelx = random_muRelx(cube[idx])
        cube[idx] = muRelx
        idx += 1
        lnlike_nonPhot += ln_prob_muRelx

        # Source-lens relative proper motion (y dimension)
        muRely, ln_prob_muRely = random_muRely(cube[idx])
        cube[idx] = muRely
        idx += 1
        lnlike_nonPhot += ln_prob_muRely


        t0 = cube[0]
        beta = cube[1]
        tE = cube[2]
        piEN = cube[3]
        piEE= cube[4]


        #Create astrometric model of source
        thetaS_model, thetaE_amp, M, shift, thetaS_nolens = MCMC_LensModel.LensModel_Trial1(tobs, t0, tE,
                                                                                            [thetaS0x, thetaS0y],
                                                                                            [muSx,muSy],
                                                                                            [muRelx, muRely],
                                                                                            beta,
                                                                                            [piEN, piEE])
        cube[11] = thetaE_amp
        cube[12] = M
        thetaSx_model = thetaS_model[:,0]
        thetaSy_model = thetaS_model[:,1]

        lnlike =  lnlikePhot + lnlike_nonPhot + \
          MCMC_LensModel.lnLikelihood(thetaSx_model, thetaSx_data, xerr_data) + \
          MCMC_LensModel.lnLikelihood(thetaSy_model, thetaSy_data, yerr_data)

        # print "Log Likelihood:  ", lnlike

        return lnlike
		
    ## num_dims = 11
    ## num_params = 13
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

    # Testing
    # lnlike = multinest_run.likelihood([0.01,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.95], 12, 12)
    # print lnlike
    return

def run():
    multinest_run(n_live_points=1000, target=target, 
                       saveto=multiNest_saveto)
    endtime = time.time()
    print 'Total runtime (min):', (endtime-starttime)/60. 
    return
 


   
