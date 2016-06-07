import numpy as np
import pylab as py
import pidly
import pyfits, pickle
import time
import math, glob
import scipy
import scipy.interpolate
from scipy.optimize import curve_fit#, OptimizeWarning
from scipy import signal,ndimage
from gcwork import objects
import pdb
import ppxf
import pp
import itertools
import pandas
import astropy
import os
import warnings
from jlu.m31 import ifu


# datadir = '/u/jlu/data/m31/08oct/081021/SPEC/reduce/m31/ss/'
# workdir = '/u/jlu/work/m31/nucleus/ifu_09_02_24/'
# cuberoot = 'm31_08oct_Kbb_050'

workdir = '/Users/kel/Documents/Projects/M31/analysis_new/ifu_11_11_30/'
datadir = workdir
mctmpdir = workdir+'tmp_mc/'
modeldir = '/Users/kel/Documents/Projects/M31/models/Peiris/2003/'
contmpdir = workdir+'tmp_convert/'
modelworkdir = '/Users/kel/Documents/Projects/M31/analysis_new/modeling/'

#cuberoot = 'm31_all_semerr'
#cuberoot = 'm31_all'
#cuberoot = 'm31_all_halforgerr'
#cuberoot = 'm31_all_seventherr'
#cuberoot = 'm31_all_scalederr'
cuberoot = 'm31_all_scalederr_cleanhdr'

cc = objects.Constants()

# Modified black hole position from 2009 sep alignment 
# analysis between NIRC2 and HST. This is the position
# in the osiris cubes.
#bhpos = np.array([8.7, 39.1]) * 0.05 # python coords, not ds9
#bhpos = np.array([22.5, 37.5]) * 0.05 # guessed offset for new M31 mosaic
# position in new mosaic, 2016/05
#bhpos = np.array([20.,41.]) * 0.05
# position in new mosaic, using new Lauer F435 frame
bhpos = np.array([19.1,40.7]) * 0.05
# for plotting horizontally
bhpos_hor = np.array([(83-40.7), 19.1]) * 0.05

def run():
    """
    Run the PPXF analysis the M31 OSIRIS data cube, using the IDL
    implementation of pPXF.
    """
    # Start an IDL session
    idl = pidly.IDL(long_delay=0.1)
    
    # Read in the data cube.
    cubefits = pyfits.open(datadir + cuberoot + '.fits')
    
    cube = cubefits[0].data
    hdr = cubefits[0].header
    errors = cubefits[1].data
    quality = cubefits[2].data
    nframes = cubefits[3].data

    # Convert the errors to the error on the mean vs. the stddev
    # nframes is wrong (too low by 1)... Why?
    # This produces errors that are still a little large compared to what
    # I get if I look at the noise in a line-less region of spectra.
    # Different is nearly a factor of 2 or so.

    # 20160218 KEL - removing - errors already calced as SEofM
    #errors /= (np.sqrt(nframes+1) * 2.0)

    # Get the wavelength solution so we can specify range:
    w0 = hdr['CRVAL1']
    dw = hdr['CDELT1']
    wavelength = w0 + dw * np.arange(cube.shape[2], dtype=float)
    wavelength /= 1000.0   # Convert to microns
    
    # Test #2
    logWaveCube, newCube, velScale = log_rebin_cube(wavelength, cube)

    # Load templates
    logWaveTemps, templates = load_templates(velScale)

    # Trim down galaxy spectrum to 2.18 microns and greater.
    # This is necessary for the ppxf routine which demands the templates
    # have broader wavelength coverage than the galaxy spectrum.
    waveCut = 2.185
    print 'blue wavelength cutoff = %.2f microns' % waveCut
    idx = np.where(np.exp(logWaveCube) > waveCut)[0]
    newCube = newCube[:,:,idx]
    logWaveCube = logWaveCube[idx]

    print 'logWaveCube.shape = ', logWaveCube.shape
    print 'logWaveTemps.shape = ', logWaveTemps.shape

    py.clf()
    py.plot(np.log(wavelength), cube[10,30,:], 'r-')
    py.plot(logWaveCube, newCube[10,30,:], 'b-')
    py.plot(logWaveTemps, templates[0,:], 'g-')
    py.show()

    # Mark good pixels as those with wavelength > 2.18.
    # Below this, atmosphere (and templates) get icky.
    # We have already made the waveCut, so all pixels in
    # the galaxy spectrum are good.
    #goodPixels = np.where(np.exp(logWaveCube) > waveCut)[0]
    #print 'goodPixels size = ', goodPixels.shape
    #goodPixels = np.array(goodPixels, dtype=np.int32)

    # Run ppxf
    print 'IDL PPXF:'
    print '  setting templates'
    idl.templates = templates
    print '  setting velocity scale = ', velScale
    idl.velScale = velScale
    print '  setting start velocity'
    idl.start = np.array([-300.0, 180.0])
    print '  setting good pixels'
    #idl.goodPixels = goodPixels
    idl('goodPixels = indgen(%d)' % (len(logWaveCube)))
    print '  setting vsyst'
    idl.dv = (logWaveTemps[0] - logWaveCube[0]) * cc.c
    idl('loadct, 12')

    # Do the whole cube
    imgShape = (newCube.shape[0], newCube.shape[1])
    velocity = np.zeros(imgShape, dtype=float)
    sigma = np.zeros(imgShape, dtype=float)
    h3 = np.zeros(imgShape, dtype=float)
    h4 = np.zeros(imgShape, dtype=float)
    h5 = np.zeros(imgShape, dtype=float)
    h6 = np.zeros(imgShape, dtype=float)
    chi2red = np.zeros(imgShape, dtype=float)

    pweights = np.zeros((newCube.shape[0], newCube.shape[1], 5), dtype=float)
    tweights = np.zeros((newCube.shape[0], newCube.shape[1], templates.shape[0]), dtype=float)


    #for xx in range(4, newCube.shape[0]-4):
    for xx in range(8, newCube.shape[0]-8):
        #for yy in range(10, newCube.shape[1]-10):
        for yy in range(10, newCube.shape[1]-10):
            print 'STARTING ppxf  ', time.ctime(time.time())

            print ''
            print '  PIXEL: xx = %d, yy = %d' % (xx, yy)
            print '    setting galaxy spectra'
            tmp = newCube[xx,yy,:]
            tmperr = np.zeros(len(tmp), dtype=float) + \
                np.median(errors[xx,yy,:])

            # Add a constant to the galaxy spectrum to make it above zero
            minFlux = tmp.mean() - (3.0 * tmp.std())
            tmp -= minFlux
            
            idl.galaxy = tmp
            print '    setting errors'
            idl.error = tmperr
            print '    calling ppxf'

            idl('ppxf, templates, galaxy, error, velScale, start, sol, GOODPIXELS=goodPixels, /plot, moments=4, degree=4, vsyst=dv, polyweights=polyweights, weights=weights')

            solution = idl.sol
            velocity[xx, yy] = solution[0]
            sigma[xx, yy] = solution[1]
            h3[xx, yy] = solution[2]
            h4[xx, yy] = solution[3]
            h5[xx, yy] = solution[4]
            h6[xx, yy] = solution[5]
            chi2red[xx, yy] = solution[6]

            pweights[xx, yy, :] = idl.polyweights
            tweights[xx, yy, :] = idl.weights

            print '  SOLUTION:'
            print '    velocity = %6.1f km/s' % velocity[xx, yy]
            print '    sigma    = %5.1f km/s' % sigma[xx, yy]
            print '    h3       = %6.3f' % h3[xx, yy]
            print '    h4       = %6.3f' % h4[xx, yy]
            print '    h5       = %6.3f' % h5[xx, yy]
            print '    h6       = %6.3f' % h6[xx, yy]
            print ''
            print '    Reduced Chi^2 = %6.3f' % chi2red[xx, yy]

    output = open(workdir + '/ppxf.dat', 'w')
    pickle.dump(velocity, output)
    pickle.dump(sigma, output)
    pickle.dump(h3, output)
    pickle.dump(h4, output)
    pickle.dump(h5, output)
    pickle.dump(h6, output)
    pickle.dump(chi2red, output)
    pickle.dump(pweights, output)
    pickle.dump(tweights, output)
    output.close()
    
def run_py(inputFile=None,verbose=True,newTemplates=True,blue=False,red=False,twocomp=False):
    """
    Run the PPXF analysis the M31 OSIRIS data cube, using the Python implementation of pPXF.
    """
    # Read in the data cube.
    if inputFile:
        cubefits = pyfits.open(inputFile)
    else:
        cubefits = pyfits.open(datadir + cuberoot + '.fits')
    
    cube = cubefits[0].data
    hdr = cubefits[0].header
    errors = cubefits[1].data
    quality = cubefits[2].data
    nframes = cubefits[3].data

    # Convert the errors to the error on the mean vs. the stddev
    # nframes is wrong (too low by 1)... Why?
    # This produces errors that are still a little large compared to what
    # I get if I look at the noise in a line-less region of spectra.
    # Different is nearly a factor of 2 or so.

    # 20160218 KEL - removing - errors already calced as SEofM
    #errors /= (np.sqrt(nframes+1) * 2.0)

    # Get the wavelength solution so we can specify range:
    w0 = hdr['CRVAL1']
    dw = hdr['CDELT1']
    wavelength = w0 + dw * np.arange(cube.shape[2], dtype=float)
    wavelength /= 1000.0   # Convert to microns
    
    # Test #2
    logWaveCube, newCube, velScale = log_rebin_cube(wavelength, cube)

    # Load templates
    if newTemplates:
        logWaveTemps, templates = load_templates(velScale,IDL=False)
    else:
        logWaveTemps, templates = load_templates_old(velScale,IDL=False)

    # Trim down galaxy spectrum to 2.18 microns and greater.
    # This is necessary for the ppxf routine which demands the templates
    # have broader wavelength coverage than the galaxy spectrum.
    #if newTemplates:
    #    waveCut = 2.185
    #else:
    #    waveCut = 2.05
    if red:
        waveCut = 2.285
    else: 
        waveCut = 2.185
    print 'blue wavelength cutoff = %.2f microns' % waveCut
    if blue:
        waveCutRed = 2.285
        idx = np.where((np.exp(logWaveCube) > waveCut) & (np.exp(logWaveCube) < waveCutRed))[0]
    else:
        idx = np.where(np.exp(logWaveCube) > waveCut)[0]
    newCube = newCube[:,:,idx]
    logWaveCube = logWaveCube[idx]
    newErrors = errors[:,:,idx]

    print 'logWaveCube.shape = ', logWaveCube.shape
    print 'logWaveTemps.shape = ', logWaveTemps.shape

    py.clf()
    py.plot(np.log(wavelength), cube[10,30,:], 'r-')
    py.plot(logWaveCube, newCube[10,30,:], 'b-')
    py.plot(logWaveTemps, templates[:,0], 'g-')
    py.show()

    # Mark good pixels as those with wavelength > 2.18.
    # Below this, atmosphere (and templates) get icky.
    # We have already made the waveCut, so all pixels in
    # the galaxy spectrum are good.
    #goodPixels = np.where(np.exp(logWaveCube) > waveCut)[0]
    #print 'goodPixels size = ', goodPixels.shape
    #goodPixels = np.array(goodPixels, dtype=np.int32)

    # Run ppxf
    print 'Python pPXF:'
    print '  templates set'
    print '  velocity scale = ', velScale
    print '  setting start velocity'
    start = np.array([-300.0, 180.0])
    print '  setting good pixels'
    goodPixels = np.arange(len(logWaveCube))
    dv = (logWaveTemps[0] - logWaveCube[0]) * cc.c
    print '  vsyst = ', dv

    # Do the whole cube
    imgShape = (newCube.shape[0], newCube.shape[1])
    print ' Cube size', newCube.shape[0], ' ', newCube.shape[1]
    velocity = np.zeros(imgShape, dtype=float)
    sigma = np.zeros(imgShape, dtype=float)
    h3 = np.zeros(imgShape, dtype=float)
    h4 = np.zeros(imgShape, dtype=float)
    if twocomp:
        velocity2 = np.zeros(imgShape, dtype=float)
        sigma2 = np.zeros(imgShape, dtype=float)
        h3_2 = np.zeros(imgShape, dtype=float)
        h4_2 = np.zeros(imgShape, dtype=float)
    h5 = np.zeros(imgShape, dtype=float)
    h6 = np.zeros(imgShape, dtype=float)
    chi2red = np.zeros(imgShape, dtype=float)

    pweights = np.zeros((newCube.shape[0], newCube.shape[1], 5), dtype=float)
    if twocomp:
        tweights = np.zeros((newCube.shape[0], newCube.shape[1], templates.shape[1]*2), dtype=float)
    else:
        tweights = np.zeros((newCube.shape[0], newCube.shape[1], templates.shape[1]), dtype=float)
    bestfit = np.zeros((newCube.shape[0], newCube.shape[1], len(idx)), dtype=float)

    # get all the xx,yy pair possiblities - setup for parallel processing
    #xx = np.arange(8, newCube.shape[0]-8)
    #yy = np.arange(10, newCube.shape[1]-10)
    xx = np.arange(newCube.shape[0])
    yy = np.arange(newCube.shape[1])
    allxxyylist = list(itertools.product(xx, yy))
    allxxyy = np.array(allxxyylist)
    allxx, allyy = zip(*itertools.product(xx, yy))

    #pdb.set_trace()

    # pp implementation
    job_server = pp.Server()
    print "Starting pp with", job_server.get_ncpus(), "workers"
    t1=time.time()
    jobs = [(i, job_server.submit(run_once, (newCube[i[0],i[1],:],newErrors[i[0],i[1],:],templates,velScale,start,goodPixels,dv,twocomp,i), (), ('numpy as np','time','ppxf'))) for i in allxxyy]
    #test=np.array([(0,0),(0,5),(10,30)])
    #test=np.array([(10,30)])
    #jobs = [(i, job_server.submit(run_once, (newCube[i[0],i[1],:],newErrors[i[0],i[1],:],templates,velScale,start,goodPixels,dv,twocomp,i), (), ('numpy as np','time','ppxf','pdb'))) for i in test]
    #test = run_once(newCube[20,45,:],newErrors[20,45,:]+1.,templates,velScale,start,goodPixels,dv,twocomp,[20,45])
    job_server.wait()
    #pdb.set_trace()
    for i, job in jobs:
        print "Setting output of ", i
        job()
        #pdb.set_trace()
        xx = i[0]
        yy = i[1]
        if job.result==None:
            # don't actually need this, since they're all 0 initially
            velocity[xx, yy] = 0
            sigma[xx, yy] = 0
            h3[xx, yy] = 0
            h4[xx, yy] = 0
            #h5[xx, yy] = 0
            #h6[xx, yy] = 0
            chi2red[xx, yy] = 0

            pweights[xx, yy, :] = 0
            tweights[xx, yy, :] = 0
            bestfit[xx, yy, :] = 0
        else:
            if twocomp:
                solution = job.result.sol[0]
                velocity[xx, yy] = solution[0]
                sigma[xx, yy] = solution[1]
                #h3[xx, yy] = solution[2]
                #h4[xx, yy] = solution[3]
                solution2 = job.result.sol[1]
                velocity2[xx, yy] = solution2[0]
                sigma2[xx, yy] = solution2[1]
                #h3_2[xx, yy] = solution2[2]
                #h4_2[xx, yy] = solution2[3]
            else:
                solution = job.result.sol
                velocity[xx, yy] = solution[0]
                sigma[xx, yy] = solution[1]
                h3[xx, yy] = solution[2]
                h4[xx, yy] = solution[3]
            #h5[xx, yy] = solution[4]
            #h6[xx, yy] = solution[5]
            chi2red[xx, yy] = job.result.chi2

            pweights[xx, yy, :] = job.result.polyweights
            tweights[xx, yy, :] = job.result.weights
            bestfit[xx, yy, :] = job.result.bestfit
            
        
        
    #pdb.set_trace()
    
    output = open(workdir + '/ppxf.dat', 'w')
    pickle.dump(velocity, output)
    pickle.dump(sigma, output)
    pickle.dump(h3, output)
    pickle.dump(h4, output)
    if twocomp:
        pickle.dump(velocity2, output)
        pickle.dump(sigma2, output)
        pickle.dump(h3_2, output)
        pickle.dump(h4_2, output)
    pickle.dump(h5, output)
    pickle.dump(h6, output)
    pickle.dump(chi2red, output)
    pickle.dump(pweights, output)
    pickle.dump(tweights, output)
    pickle.dump(newCube, output)
    pickle.dump(bestfit, output)
    output.close()

    print "Time elapsed: ", time.time() - t1, "s"

def run_once(newCube,errors,templates,velScale,start,goodPixels,vsyst,twocomp,allxxyy,verbose=False):
    xx = allxxyy[0]
    yy = allxxyy[1]

    if verbose:
        print 'STARTING ppxf  ', time.ctime(time.time())
        t1 = time.time()
        print ''
        print '  PIXEL: xx = %d, yy = %d' % (xx, yy)
        print '    setting galaxy spectra'
    
    tmp = newCube
    tmperr = errors
    
    # Add a constant to the galaxy spectrum to make it above zero
    minFlux = tmp.mean() - (3.0 * tmp.std())
    tmp2 = tmp - minFlux
            
    galaxy = tmp2
    error = tmperr

    if twocomp:
        templates2 = np.concatenate((templates,templates),axis=1)
        outppxf = ppxf.ppxf(templates2, galaxy, error, velScale, [start,start], goodpixels=goodPixels, plot=False, moments=[2,2], mdegree=4, vsyst=vsyst,component=[0]*23+[1]*23)
    else:
        outppxf = ppxf.ppxf(templates, galaxy, error, velScale, start, goodpixels=goodPixels, plot=False, moments=4, mdegree=4, vsyst=vsyst)
    

    #pdb.set_trace()
    return outppxf

def runErrorMC(newTemplates=True,jackknife=False,test=False):
    """
    Run the PPXF analysis the M31 OSIRIS data cube.
    """
    # Read in the data cube.
    cubefits = pyfits.open(datadir + cuberoot + '.fits')
    
    cube = cubefits[0].data
    hdr = cubefits[0].header
    errors = cubefits[1].data
    quality = cubefits[2].data
    nframes = cubefits[3].data

    # Convert the errors to the error on the mean vs. the stddev
    # nframes is wrong (too low by 1)... Why?
    # This produces errors that are still a little large compared to what
    # I get if I look at the noise in a line-less region of spectra.
    # Different is nearly a factor of 2 or so.
    # 20160218 KEL - removing - error already calced as SEotM
    #errors /= (np.sqrt(nframes+1) * 2.0)

    # Get the wavelength solution so we can specify range:
    w0 = hdr['CRVAL1']
    dw = hdr['CDELT1']
    wavelength = w0 + dw * np.arange(cube.shape[2], dtype=float)
    wavelength /= 1000.0   # Convert to microns
    
    # Test #2
    logWaveCube, newCube, velScale = log_rebin_cube(wavelength, cube)

    # Load templates
    if newTemplates:
        logWaveTemps, templates = load_templates(velScale,IDL=False)
    else:
        logWaveTemps, templates = load_templates_old(velScale,IDL=False)


    # Trim down galaxy spectrum to 2.18 microns and greater.
    # This is necessary for the ppxf routine which demands the templates
    # have broader wavelength coverage than the galaxy spectrum.
    waveCut = 2.185
    print 'blue wavelength cutoff = %.2f microns' % waveCut
    idx = np.where(np.exp(logWaveCube) > waveCut)[0]
    newCube = newCube[:,:,idx]
    logWaveCube = logWaveCube[idx]
    newErrors = errors[:,:,idx]

    print 'logWaveCube.shape = ', logWaveCube.shape
    print 'logWaveTemps.shape = ', logWaveTemps.shape

    # Mark good pixels as those with wavelength > 2.1.
    # Below this, atmosphere (and templates) get icky
    #goodPixels = np.where(np.exp(logWaveCube) > 2.1)[0]
    #print 'goodPixels = ', goodPixels

    # Run ppxf
    print 'Python pPXF:'
    print '  templates set'
    print '  velocity scale = ', velScale
    print '  setting start velocity'
    start = np.array([-300.0, 180.0])
    print '  setting good pixels'
    goodPixels = np.arange(len(logWaveCube))
    dv = (logWaveTemps[0] - logWaveCube[0]) * cc.c
    print '  vsyst = ', dv
    
    # Error distributions
    #nsim = 3
    nsim = 100
    if jackknife:
        niter = templates.shape[1]
    else:
        niter = nsim

    imgShape = (newCube.shape[0], newCube.shape[1])

    # set array to hold MC errors
    if test:
        npix = 6
        velocity = np.zeros((npix, niter), dtype=float)
        sigma = np.zeros((npix, niter), dtype=float)
        h3 = np.zeros((npix, niter), dtype=float)
        h4 = np.zeros((npix, niter), dtype=float)
        h5 = np.zeros((npix, niter), dtype=float)
        h6 = np.zeros((npix, niter), dtype=float)
        chi2red = np.zeros((npix, niter), dtype=float)

        pweights = np.zeros((npix, 5, niter), 
                           dtype=float)
        tweights = np.zeros((npix, 
                            templates.shape[1], niter), dtype=float)
    
    velocityErr = np.zeros(imgShape, dtype=float)
    sigmaErr = np.zeros(imgShape, dtype=float)
    h3Err = np.zeros(imgShape, dtype=float)
    h4Err = np.zeros(imgShape, dtype=float)
    h5Err = np.zeros(imgShape, dtype=float)
    h6Err = np.zeros(imgShape, dtype=float)
    chi2redErr = np.zeros(imgShape, dtype=float)

    pweightsErr = np.zeros((newCube.shape[0], newCube.shape[1], 5), 
                           dtype=float)
    tweightsErr = np.zeros((newCube.shape[0], newCube.shape[1], 
                            templates.shape[1]), dtype=float)

    # set arrays to hold average outputs from MC run
    velocityAvg = np.zeros(imgShape, dtype=float)
    sigmaAvg = np.zeros(imgShape, dtype=float)
    h3Avg = np.zeros(imgShape, dtype=float)
    h4Avg = np.zeros(imgShape, dtype=float)
    h5Avg = np.zeros(imgShape, dtype=float)
    h6Avg = np.zeros(imgShape, dtype=float)
    chi2redAvg = np.zeros(imgShape, dtype=float)

    pweightsAvg = np.zeros((newCube.shape[0], newCube.shape[1], 5), 
                           dtype=float)
    tweightsAvg = np.zeros((newCube.shape[0], newCube.shape[1], 
                            templates.shape[1]), dtype=float)
    
    # get all the xx,yy pair possiblities - setup for parallel processing
    if test:
        #xx = [17,18,19,20,21,22]
        yy = [35]
        #xx = [17,18]
        #yy = [35,35]
        xx = [0,1]
    else:
        #xx = np.arange(8, newCube.shape[0]-8)
        #yy = np.arange(10, newCube.shape[1]-10)
        xx = np.arange(newCube.shape[0])
        yy = np.arange(newCube.shape[1])
        
    allxxyylist = list(itertools.product(xx,yy))
    allxxyy = np.array(allxxyylist)

    job_server = pp.Server()
    print "Starting pp with", job_server.get_ncpus(), "workers"
    t1=time.time()
    jobs = [(i,job_server.submit(run_once_mc, (newCube[i[0],i[1],:],newErrors[i[0],i[1],:],templates,velScale,start,goodPixels,dv,nsim,mctmpdir,jackknife,test,i), (), ('numpy as np','time','ppxf'))) for i in allxxyy]
    #test = run_once_mc(newCube[allxxyy[0,0],allxxyy[0,1]],newErrors[allxxyy[0,0],allxxyy[0,1]],templates,velScale,start,goodPixels,dv,nsim,mctmpdir,jackknife,test,allxxyy[0])
    #pdb.set_trace()
    # wait for all the jobs to finish before proceeding
    job_server.wait()
    # pdb.set_trace()
    
    for i in range(len(allxxyy)):
        print "Setting output of ", i
        xx = allxxyy[i][0]
        yy = allxxyy[i][1]
        
        p = PPXFresultsMC(inputFile=mctmpdir + 'mc_'+str(xx)+'_'+str(yy)+'.dat')

        if test:
            for j in range(niter):
                n = PPXFresults(inputFile=mctmpdir + 'mc_'+str(xx)+'_'+str(yy)+'_iter'+str(j)+'.dat')
                velocity[i,j] = n.velocity
                sigma[i,j] = n.sigma
                h3[i,j] = n.h3
                h4[i,j] = n.h4
                chi2red[i,j] = n.chi2red
                pweights[i,:, j] = n.pweights
                tweights[i,:, j] = n.tweights
        
        velocityErr[xx, yy] = p.velocityErr
        sigmaErr[xx,yy] = p.sigmaErr
        h3Err[xx,yy] = p.h3Err
        h4Err[xx,yy] = p.h4Err
        chi2redErr[xx,yy] = p.chi2redErr
        pweightsErr[xx,yy,:] = p.pweightsErr
        tweightsErr[xx,yy,:] = p.tweightsErr

        velocityAvg[xx, yy] = p.velocity
        sigmaAvg[xx,yy] = p.sigma
        h3Avg[xx,yy] = p.h3
        h4Avg[xx,yy] = p.h4
        chi2redAvg[xx,yy] = p.chi2red
        pweightsAvg[xx,yy,:] = p.pweights
        tweightsAvg[xx,yy,:] = p.tweights       
                    
    output = open(workdir + '/ppxf_errors_mc_nsim'+str(nsim)+'.dat', 'w')
    pickle.dump(velocityErr, output)
    pickle.dump(sigmaErr, output)
    pickle.dump(h3Err, output)
    pickle.dump(h4Err, output)
    pickle.dump(h5Err, output)
    pickle.dump(h6Err, output)
    pickle.dump(chi2redErr, output)
    pickle.dump(pweightsErr, output)
    pickle.dump(tweightsErr, output)
    output.close()

    output = open(workdir + '/ppxf_avg_mc_nsim'+str(nsim)+'.dat', 'w')
    pickle.dump(velocityAvg, output)
    pickle.dump(sigmaAvg, output)
    pickle.dump(h3Avg, output)
    pickle.dump(h4Avg, output)
    pickle.dump(h5Avg, output)
    pickle.dump(h6Avg, output)
    pickle.dump(chi2redAvg, output)
    pickle.dump(pweightsAvg, output)
    pickle.dump(tweightsAvg, output)
    output.close()

    if test:
        output = open(workdir + '/ppxf_test_mc_nsim'+str(nsim)+'.dat','w')
        pickle.dump(velocity, output)
        pickle.dump(sigma, output)
        pickle.dump(h3, output)
        pickle.dump(h4, output)
        pickle.dump(h5, output)
        pickle.dump(h6, output)
        pickle.dump(chi2red, output)
        pickle.dump(pweights, output)
        pickle.dump(tweights, output)
        output.close()

    print "Time elapsed: ", time.time() - t1, "s"
            

def run_once_mc(newCube,errors,templates,velScale,start,goodPixels,vsyst,nsim,mctmpdir,jackknife,test,allxxyy,verbose=False):
    # code can run either MC simulations (e.g. spectrum is perturbed within the errors, fits are calculated nsim number of times)
    # or jackknife simulations (e.g. one spectral template at a time is dropped from the fits; nsim is disregarded in this
    # case and the fits are calculated n_template number of times)
    xx = allxxyy[0]
    yy = allxxyy[1]

    if verbose:
        print 'STARTING ppxf  ', time.ctime(time.time())
        t1 = time.time()
        print ''
        print '  PIXEL: xx = %d, yy = %d' % (xx, yy)
        print '    setting galaxy spectra'
    
    tmp = newCube
    tmperr = errors
    
    # Add a constant to the galaxy spectrum to make it above zero
    minFlux = tmp.mean() - (3.0 * tmp.std())
    tmp -= minFlux

    if jackknife:
        iter = templates.shape[1]
    else:
        iter = nsim

    velocity = np.zeros(iter, dtype=float)
    sigma = np.zeros(iter, dtype=float)
    h3 = np.zeros(iter, dtype=float)
    h4 = np.zeros(iter, dtype=float)
    h5 = np.zeros(iter, dtype=float)
    h6 = np.zeros(iter, dtype=float)
    chi2red = np.zeros(iter, dtype=float)

    pweights = np.zeros((iter, 5), dtype=float)
    tweights = np.zeros((iter, templates.shape[1]), dtype=float)

    # check if the error array is all 0 - if so, bad spaxel so
    # just leave the zeros in the initialized array
    if tmperr.mean() != 0.:
        for n in range(iter):
            if jackknife:
                specSim = tmp
            else:
                specSim = tmp + (np.random.randn(len(tmp)) * tmperr)
            
            galaxy = specSim
            error = tmperr

            if jackknife:
                templatesJK = np.zeros((templates.shape[0],templates.shape[1]), dtype=float)
                templatesJK[:,0:(n-1)] = templates[:,0:(n-1)]
                templatesJK[:,(n+1):] = templates[:,(n+1):]
            else:
                templatesJK = templates
        
            p = ppxf.ppxf(templatesJK, galaxy, error, velScale, start, goodpixels=goodPixels, plot=False, moments=4, mdegree=4, vsyst=vsyst)

            solution = p.sol
            velocity[n] = solution[0]
            sigma[n] = solution[1]
            h3[n] = solution[2]
            h4[n] = solution[3]
            chi2red[n] = p.chi2
            pweights[n,:] = p.polyweights
            tweights[n,:] = p.weights

    velocityAvg = np.average(velocity)
    velocityErr = np.std(velocity)
    sigmaAvg = np.average(sigma)
    sigmaErr = np.std(sigma)
    h3Avg = np.average(h3)
    h3Err = np.std(h3)
    h4Avg = np.average(h4)
    h4Err = np.std(h4)
    h5Avg = np.average(h4)
    h5Err = np.std(h4)
    h6Avg = np.average(h4)
    h6Err = np.std(h4)
    chi2redAvg = np.average(chi2red)
    chi2redErr = np.std(chi2red)
    pweightsAvg = np.average(pweights,axis=0)
    pweightsErr = np.std(pweights,axis=0)
    tweightsAvg = np.average(tweights,axis=0)
    tweightsErr = np.std(tweights,axis=0) 

    output = open(mctmpdir + 'mc_'+str(xx)+'_'+str(yy)+'.dat', 'w')
    pickle.dump(velocityAvg, output)
    pickle.dump(velocityErr, output)
    pickle.dump(sigmaAvg, output)
    pickle.dump(sigmaErr, output)
    pickle.dump(h3Avg, output)
    pickle.dump(h3Err, output)
    pickle.dump(h4Avg, output)
    pickle.dump(h4Err, output)
    pickle.dump(h5Avg, output)
    pickle.dump(h5Err, output)
    pickle.dump(h6Avg, output)
    pickle.dump(h6Err, output)
    pickle.dump(chi2redAvg, output)
    pickle.dump(chi2redErr, output)
    pickle.dump(pweightsAvg, output)
    pickle.dump(pweightsErr, output)
    pickle.dump(tweightsAvg, output)
    pickle.dump(tweightsErr, output)
    output.close()

    if test:
        for n in range(iter):
            output = open(mctmpdir + 'mc_'+str(xx)+'_'+str(yy)+'_iter'+str(n)+'.dat', 'w')
            pickle.dump(velocity[n], output)
            pickle.dump(sigma[n], output)
            pickle.dump(h3[n], output)
            pickle.dump(h4[n], output)
            pickle.dump(h5[n], output)
            pickle.dump(h6[n], output)
            pickle.dump(chi2red[n], output)
            pickle.dump(pweights[n,:], output)
            pickle.dump(tweights[n,:], output)
            output.close()
            
    #pdb.set_trace()
    
class PPXFresults(object):
    def __init__(self, inputFile=workdir+'ppxf.dat',bestfit=False,twocomp=False):
        self.inputFile = inputFile
        
        input = open(inputFile, 'r')
        self.velocity = pickle.load(input)
        self.sigma = pickle.load(input)
        self.h3 = pickle.load(input)
        self.h4 = pickle.load(input)
        if twocomp:
            self.velocity2 = pickle.load(input)
            self.sigma2 = pickle.load(input)
            self.h3_2 = pickle.load(input)
            self.h4_2 = pickle.load(input)
        self.h5 = pickle.load(input)
        self.h6 = pickle.load(input)
        self.chi2red = pickle.load(input)
        self.pweights = pickle.load(input)
        self.tweights = pickle.load(input)
        if bestfit:
            self.galaxy = pickle.load(input)
            self.bestfit = pickle.load(input)

class PPXFresultsMC(object):
    def __init__(self, inputFile=mctmpdir + '/mc_00_00.dat'):
        self.inputFile = inputFile

        input = open(inputFile, 'r')
        self.velocity = pickle.load(input)
        self.velocityErr = pickle.load(input)
        self.sigma = pickle.load(input)
        self.sigmaErr = pickle.load(input)
        self.h3 = pickle.load(input)
        self.h3Err = pickle.load(input)
        self.h4 = pickle.load(input)
        self.h4Err = pickle.load(input)
        self.h5 = pickle.load(input)
        self.h5Err = pickle.load(input)
        self.h6 = pickle.load(input)
        self.h6Err = pickle.load(input)
        self.chi2red = pickle.load(input)
        self.chi2redErr = pickle.load(input)
        self.pweights = pickle.load(input)
        self.pweightsErr = pickle.load(input)
        self.tweights = pickle.load(input)
        self.tweightsErr = pickle.load(input)

def plotResults(inputFile):
    cubeimg = pyfits.getdata(datadir + cuberoot + '_img.fits')
    snrimg = pyfits.getdata(datadir + cuberoot + '_snr.fits')

    p = PPXFresults(inputFile)

    print p.velocity.shape
    print cubeimg.shape
    print bhpos
    xaxis = np.arange(cubeimg.shape[0]) * 0.05
    yaxis = np.arange(cubeimg.shape[1]) * 0.05
    
    xtickLoc = py.MultipleLocator(0.5)

    #py.figure(2, figsize=(12,5))
    py.figure(2, figsize=(6,17))
    py.subplots_adjust(left=0.01, right=0.94, top=0.95)
    py.clf()

    ##########
    # Plot Cube Image
    ##########
    py.subplot(4, 1, 1)
    #cubeimg=cubeimg.transpose()
    cubeimg=py.ma.masked_where(cubeimg==0.,cubeimg)
    py.imshow(np.rot90(py.ma.masked_where(cubeimg<-10000, cubeimg),3), 
              extent=[xaxis[0], xaxis[-1], yaxis[0], yaxis[-1]],
              cmap=py.cm.hot)
    py.plot([bhpos_hor[0]], [bhpos_hor[1]], 'kx')
    py.ylabel('Y (arcsec)')
    py.xlabel('X (arcsec)')
    py.gca().get_xaxis().set_major_locator(xtickLoc)
    py.axis('image')
    cbar = py.colorbar(orientation='vertical')
    cbar.set_label('Flux (cts/sec)')

    # Make a compass rose
    pa = 56.0
    cosSin = np.array([ math.cos(math.radians(pa)), 
                        math.sin(math.radians(pa)) ])
    arr_base = np.array([ xaxis[-1]-0.2, yaxis[-1]-0.6 ])
    arr_n = cosSin * 0.2
    arr_w = cosSin[::-1] * 0.2
    arr_n_rot90 = cosSin[::-1] * 0.2
    arr_w_rot90 = cosSin * 0.2
    #py.arrow(arr_base[0], arr_base[1], arr_n[0], arr_n[1],
    py.arrow(arr_base[0], arr_base[1], -arr_n_rot90[0], arr_n_rot90[1],
             edgecolor='w', facecolor='w', width=0.03, head_width=0.08)
    #py.arrow(arr_base[0], arr_base[1], -arr_w[0], arr_w[1],
    py.arrow(arr_base[0], arr_base[1], -arr_w_rot90[0], -arr_w_rot90[1],
             edgecolor='w', facecolor='w', width=0.03, head_width=0.08)
    py.text(arr_base[0]-arr_n[0]-.25, arr_base[1]-arr_n[1]-0.28, 'E', 
            color='white', 
            horizontalalignment='left', verticalalignment='bottom')
    #py.text(arr_base[0]-arr_w[0]-0.15, arr_base[1]+arr_w[1]+0.1, 'E',
    py.text(arr_base[0]-arr_n_rot90[0]-0.18, arr_base[1]+arr_n_rot90[1]+0.1, 'N', 
            color='white',
            horizontalalignment='right', verticalalignment='center')
    py.title('K Image')


    ##########
    # Plot SNR Image
    ##########
    print snrimg[30,10]
    py.subplot(4, 1, 2)
    snrimg=snrimg.transpose()
    snrimg=np.rot90(snrimg,3)
    py.imshow(py.ma.masked_invalid(snrimg), 
              extent=[xaxis[0], xaxis[-1], yaxis[0], yaxis[-1]],
              cmap=py.cm.jet)
    py.plot([bhpos_hor[0]], [bhpos_hor[1]], 'kx')
    py.xlabel('X (arcsec)')
    py.gca().get_xaxis().set_major_locator(xtickLoc)
    py.axis('image')
    cbar = py.colorbar(orientation='vertical')
    cbar.set_label('SNR')
    py.title('SNR')

    ##########
    # Plot Velocity
    ##########
    py.subplot(4, 1, 3)
    velimg = p.velocity.transpose()+308.0
    py.imshow(np.rot90(py.ma.masked_where(velimg==308.,velimg),3), vmin=-250., vmax=250.,
              extent=[xaxis[0], xaxis[-1], yaxis[0], yaxis[-1]])
    py.plot([bhpos_hor[0]], [bhpos_hor[1]], 'kx')
    py.xlabel('X (arcsec)')
    py.gca().get_xaxis().set_major_locator(xtickLoc)
    py.axis('image')
    cbar = py.colorbar(orientation='vertical')
    cbar.set_label('Velocity (km/s)')
    py.title('Velocity')

    ##########
    # Plot Dispersion
    ##########
    py.subplot(4, 1, 4)
    sigimg = p.sigma.transpose()
    py.imshow(np.rot90(py.ma.masked_where(sigimg==0.,sigimg),3), vmin=0., vmax=250.,
              extent=[xaxis[0], xaxis[-1], yaxis[0], yaxis[-1]],
              cmap=py.cm.jet)
    py.plot([bhpos_hor[0]], [bhpos_hor[1]], 'kx')
    py.xlabel('X (arcsec)')
    py.gca().get_xaxis().set_major_locator(xtickLoc)
    py.axis('image')
    cbar = py.colorbar(orientation='vertical')
    cbar.set_label('Dispersion (km/s)')
    py.title('Dispersion')

#     for xx in range(4, velocity.shape[0]-5):
#         for yy in range(10, velocity.shape[1]-11):
#             print 'xx=%2d, yy=%2d, velocity: mean=%6.1f  std = %6.1f' % \
#                 (xx, yy, 
#                  p.velocity[xx:xx+2,yy:yy+2].mean(), 
#                  p.velocity[xx:xx+2,yy:yy+2].std())

    py.tight_layout()
    
    py.savefig(workdir + 'plots/kinematic_maps.png')
    py.savefig(workdir + 'plots/kinematic_maps.eps')
    py.show()



def plotResults2(inputFile):
    cubeimg = pyfits.getdata(datadir + cuberoot + '_img.fits')
    #snrimg = pyfits.getdata(datadir + cuberoot + '_snr.fits')

    p = PPXFresults(inputFile)

    print p.velocity.shape
    print cubeimg.shape
    print bhpos

    xaxis = np.arange(cubeimg.shape[0]) * 0.05
    yaxis = np.arange(cubeimg.shape[1]) * 0.05
    
    xtickLoc = py.MultipleLocator(0.5)

    py.close(2)
    py.figure(2, figsize=(7,15))
    py.subplots_adjust(left=0.05, right=0.94, top=0.95)
    py.clf()

    ##########
    # Plot Cube Image
    ##########
    py.subplot(3, 1, 1)
    #plcube = cubeimg.transpose()
    cubeimg=py.ma.masked_where(cubeimg==0.,cubeimg)
    py.imshow(np.rot90(py.ma.masked_where(cubeimg<-10000, cubeimg),3), 
              extent=[xaxis[0], xaxis[-1], yaxis[0], yaxis[-1]],
              cmap=py.cm.hot)
    py.plot([bhpos_hor[0]], [bhpos_hor[1]], 'kx', markeredgewidth=3)
    py.ylabel('Y (arcsec)')
    py.xlabel('X (arcsec)')
    py.gca().get_xaxis().set_major_locator(xtickLoc)
    py.axis('image')
    cbar = py.colorbar(orientation='vertical')
    cbar.set_label('Flux (cts/sec)')

    # Make a compass rose
    pa = 56.0
    cosSin = np.array([ math.cos(math.radians(pa)), 
                        math.sin(math.radians(pa)) ])
    arr_base = np.array([ xaxis[-1]-0.2, yaxis[-1]-0.6 ])
    arr_n = cosSin * 0.2
    arr_w = cosSin[::-1] * 0.2
    arr_n_rot90 = cosSin[::-1] * 0.2
    arr_w_rot90 = cosSin * 0.2
    #py.arrow(arr_base[0], arr_base[1], arr_n[0], arr_n[1],
    py.arrow(arr_base[0], arr_base[1], -arr_n_rot90[0], arr_n_rot90[1],
             edgecolor='w', facecolor='w', width=0.03, head_width=0.08)
    #py.arrow(arr_base[0], arr_base[1], -arr_w[0], arr_w[1],
    py.arrow(arr_base[0], arr_base[1], -arr_w_rot90[0], -arr_w_rot90[1],
             edgecolor='w', facecolor='w', width=0.03, head_width=0.08)
    py.text(arr_base[0]-arr_n[0]-.25, arr_base[1]-arr_n[1]-0.28, 'E', 
            color='white', 
            horizontalalignment='left', verticalalignment='bottom')
    #py.text(arr_base[0]-arr_w[0]-0.15, arr_base[1]+arr_w[1]+0.1, 'E',
    py.text(arr_base[0]-arr_n_rot90[0]-0.18, arr_base[1]+arr_n_rot90[1]+0.1, 'N', 
            color='white',
            horizontalalignment='right', verticalalignment='center')

    ##########
    # Plot Velocity
    ##########
    py.subplot(3, 1, 2)
    velimg = p.velocity.transpose()+308.0
    py.imshow(np.rot90(py.ma.masked_where(velimg==308.,velimg),3), vmin=-250., vmax=250., 
              extent=[xaxis[0], xaxis[-1], yaxis[0], yaxis[-1]])
    py.plot([bhpos_hor[0]], [bhpos_hor[1]], 'kx', markeredgewidth=3)
    py.xlabel('X (arcsec)')
    py.gca().get_xaxis().set_major_locator(xtickLoc)
    py.axis('image')
    cbar = py.colorbar(orientation='vertical')
    cbar.set_label('Velocity (km/s)')

    ##########
    # Plot Dispersion
    ##########
    py.subplot(3, 1, 3)
    sigimg = p.sigma.transpose()
    py.imshow(np.rot90(py.ma.masked_where(sigimg==0.,sigimg),3), vmin=0., vmax=250., 
              extent=[xaxis[0], xaxis[-1], yaxis[0], yaxis[-1]],
              cmap=py.cm.jet)
    py.plot([bhpos_hor[0]], [bhpos_hor[1]], 'kx', markeredgewidth=3)
    py.xlabel('X (arcsec)')
    py.gca().get_xaxis().set_major_locator(xtickLoc)
    py.axis('image')
    cbar = py.colorbar(orientation='vertical')
    cbar.set_label('Dispersion (km/s)')

#     for xx in range(4, velocity.shape[0]-5):
#         for yy in range(10, velocity.shape[1]-11):
#             print 'xx=%2d, yy=%2d, velocity: mean=%6.1f  std = %6.1f' % \
#                 (xx, yy, 
#                  p.velocity[xx:xx+2,yy:yy+2].mean(), 
#                  p.velocity[xx:xx+2,yy:yy+2].std())

    py.tight_layout()

    py.savefig(workdir + 'plots/kinematic_maps2.png')
    py.savefig(workdir + 'plots/kinematic_maps2.eps')
    py.show()

def plotResults3(inputFile,zoom=False,twocomp=False):
    cubeimg = pyfits.getdata(datadir + cuberoot + '_img.fits')
       
    p = PPXFresults(inputFile,twocomp=twocomp)

    xaxis = np.arange(cubeimg.shape[0]) * 0.05
    yaxis = np.arange(cubeimg.shape[1]) * 0.05

    if zoom:
        x0 = np.abs(xaxis-(bhpos_hor[0]-0.5)).argmin()
        x1 = np.abs(xaxis-(bhpos_hor[0]+0.5)).argmin()
        y0 = np.abs(yaxis-(bhpos_hor[1]-0.5)).argmin()
        y1 = np.abs(yaxis-(bhpos_hor[1]+0.5)).argmin()
    else:
        x0 = 0
        x1 = -1
        y0 = 0
        y1 = -1
    
    xtickLoc = py.MultipleLocator(0.5)

    py.close(2)
    if zoom:
        py.figure(2, figsize=(9,7))
    else:
        py.figure(2, figsize=(15,7))
    py.subplots_adjust(left=0.05, right=0.94, top=0.95)
    py.clf()

    ##########
    # Plot Velocity
    ##########
    py.subplot(2, 2, 1)
    if twocomp:
        velimg = p.velocity2.transpose()+308.0
    else:
        velimg = p.velocity.transpose()+308.0
    py.imshow(np.rot90(py.ma.masked_where(velimg[x0:x1,y0:y1]==308.,velimg[x0:x1,y0:y1]),3), vmin=-250., vmax=250.,
              extent=[xaxis[x0], xaxis[x1], yaxis[y0], yaxis[y1]])
    py.plot([bhpos_hor[0]], [bhpos_hor[1]], 'kx', markeredgewidth=3)
    py.ylabel('Y (arcsec)')
    py.xlabel('X (arcsec)')
    py.gca().get_xaxis().set_major_locator(xtickLoc)
    py.axis('image')
    cbar = py.colorbar(orientation='vertical')
    cbar.set_label('Velocity (km/s)')

    ##########
    # Plot Dispersion
    ##########
    py.subplot(2, 2, 2)
    if twocomp:
        sigimg = p.sigma2.transpose()
    else:
       sigimg = p.sigma.transpose()
    py.imshow(np.rot90(py.ma.masked_where(sigimg[x0:x1,y0:y1]==0.,sigimg[x0:x1,y0:y1]),3), vmin=0., vmax=250.,
              extent=[xaxis[x0], xaxis[x1], yaxis[y0], yaxis[y1]],
              cmap=py.cm.jet)
    py.plot([bhpos_hor[0]], [bhpos_hor[1]], 'kx', markeredgewidth=3)
    py.xlabel('X (arcsec)')
    py.gca().get_xaxis().set_major_locator(xtickLoc)
    py.axis('image')
    cbar = py.colorbar(orientation='vertical')
    cbar.set_label('Dispersion (km/s)')

    ##########
    # Plot h3
    ##########
    py.subplot(2, 2, 3)
    if twocomp:
        h3img = p.h3_2.transpose()
    else:
        h3img = p.h3.transpose()
    py.imshow(np.rot90(py.ma.masked_where(h3img[x0:x1,y0:y1]==0, h3img[x0:x1,y0:y1]),3),vmin=-.2,vmax=.2, 
              extent=[xaxis[x0], xaxis[x1], yaxis[y0], yaxis[y1]])
    py.plot([bhpos_hor[0]], [bhpos_hor[1]], 'kx', markeredgewidth=3)
    py.ylabel('Y (arcsec)')
    py.xlabel('X (arcsec)')
    py.gca().get_xaxis().set_major_locator(xtickLoc)
    py.axis('image')
    cbar = py.colorbar(orientation='vertical')
    cbar.set_label('h3')

    ##########
    # Plot h4
    ##########
    py.subplot(2, 2, 4)
    if twocomp:
        h4img = p.h4_2.transpose()
    else:
        h4img = p.h4.transpose()
    py.imshow(np.rot90(py.ma.masked_where(h4img[x0:x1,y0:y1]==0, h4img[x0:x1,y0:y1]),3),vmin=-.2,vmax=.2, 
              extent=[xaxis[x0], xaxis[x1], yaxis[y0], yaxis[y1]])
    py.plot([bhpos_hor[0]], [bhpos_hor[1]], 'kx', markeredgewidth=3)
    py.xlabel('X (arcsec)')
    py.gca().get_xaxis().set_major_locator(xtickLoc)
    py.axis('image')
    cbar = py.colorbar(orientation='vertical')
    cbar.set_label('h4')

    py.tight_layout()

    if zoom:
        py.savefig(workdir + 'plots/kinematic_maps3_zoom.png')
        py.savefig(workdir + 'plots/kinematic_maps3_zoom.eps')
    else:
        py.savefig(workdir + 'plots/kinematic_maps3.png')
        py.savefig(workdir + 'plots/kinematic_maps3.eps')
    py.show()

def plotErr1(inputResults=workdir+'/ppxf.dat',inputAvg=workdir+'/ppxf_avg_mc_nsim100.dat',inputErr=workdir+'/ppxf_errors_mc_nsim100.dat'):
    ### Plots error on velocity and velocity dispersion
    cubeimg = pyfits.getdata(datadir + cuberoot + '_img.fits')
    
    p = PPXFresults(inputResults)
    a = PPXFresults(inputAvg)
    e = PPXFresults(inputErr)

    xaxis = np.arange(cubeimg.shape[0]) * 0.05
    yaxis = np.arange(cubeimg.shape[1]) * 0.05
    
    xtickLoc = py.MultipleLocator(0.5)

    py.close(2)
    py.figure(2, figsize=(15,13))
    py.subplots_adjust(left=0.05, right=0.94, top=0.95)
    py.clf()

    ##########
    # Plot Velocity, MC velocity average, and MC velocity error
    ##########
    py.subplot(3, 2, 1)
    velimg = p.velocity.transpose()+308.0
    py.imshow(np.rot90(py.ma.masked_where(velimg==308.,velimg),3), vmin=-250., vmax=250., 
              extent=[xaxis[0], xaxis[-1], yaxis[0], yaxis[-1]])
    py.plot([bhpos_hor[0]], [bhpos_hor[1]], 'kx', markeredgewidth=3)
    py.ylabel('Y (arcsec)')
    py.xlabel('X (arcsec)')
    py.gca().get_xaxis().set_major_locator(xtickLoc)
    py.axis('image')
    cbar = py.colorbar(orientation='vertical')
    cbar.set_label('Velocity (km/s)')

    #pdb.set_trace()
    
    py.subplot(3,2,3)
    velavg = a.velocity.transpose()+308.0
    py.imshow(np.rot90(py.ma.masked_where(velavg==308.,velavg),3), vmin=-250., vmax=250.,
              extent=[xaxis[0], xaxis[-1], yaxis[0], yaxis[-1]])
    py.plot([bhpos_hor[0]], [bhpos_hor[1]], 'kx', markeredgewidth=3)
    py.ylabel('Y (arcsec)')
    py.xlabel('X (arcsec)')
    py.gca().get_xaxis().set_major_locator(xtickLoc)
    py.axis('image')
    cbar = py.colorbar(orientation='vertical')
    cbar.set_label('Velocity, MC avg')

    py.subplot(3,2,5)
    velerr = e.velocity.transpose()
    py.imshow(np.rot90(py.ma.masked_where(velerr==0.,velerr),3), vmin=0., vmax=40.,
              extent=[xaxis[0], xaxis[-1], yaxis[0], yaxis[-1]])
    py.plot([bhpos_hor[0]], [bhpos_hor[1]], 'kx', markeredgewidth=3)
    py.ylabel('Y (arcsec)')
    py.xlabel('X (arcsec)')
    py.gca().get_xaxis().set_major_locator(xtickLoc)
    py.axis('image')
    cbar = py.colorbar(orientation='vertical')
    cbar.set_label('Velocity, MC err')

    ##########
    # Plot Dispersion
    ##########
    py.subplot(3, 2, 2)
    sigimg = p.sigma.transpose()
    py.imshow(np.rot90(py.ma.masked_where(sigimg==0.,sigimg),3), vmin=0., vmax=250.,
              extent=[xaxis[0], xaxis[-1], yaxis[0], yaxis[-1]],
              cmap=py.cm.jet)
    py.plot([bhpos_hor[0]], [bhpos_hor[1]], 'kx', markeredgewidth=3)
    py.xlabel('X (arcsec)')
    py.gca().get_xaxis().set_major_locator(xtickLoc)
    py.axis('image')
    cbar = py.colorbar(orientation='vertical')
    cbar.set_label('Dispersion (km/s)')

    py.subplot(3, 2, 4)
    sigavg = a.sigma.transpose()
    py.imshow(np.rot90(py.ma.masked_where(sigavg==0.,sigimg),3), vmin=0., vmax=250.,
              extent=[xaxis[0], xaxis[-1], yaxis[0], yaxis[-1]],
              cmap=py.cm.jet)
    py.plot([bhpos_hor[0]], [bhpos_hor[1]], 'kx', markeredgewidth=3)
    py.xlabel('X (arcsec)')
    py.gca().get_xaxis().set_major_locator(xtickLoc)
    py.axis('image')
    cbar = py.colorbar(orientation='vertical')
    cbar.set_label('Dispersion, MC avg')

    py.subplot(3, 2, 6)
    sigerr = e.sigma.transpose()
    py.imshow(np.rot90(py.ma.masked_where(sigerr==0.,sigerr),3), vmin=0., vmax=50.,
              extent=[xaxis[0], xaxis[-1], yaxis[0], yaxis[-1]],
              cmap=py.cm.jet)
    py.plot([bhpos_hor[0]], [bhpos_hor[1]], 'kx', markeredgewidth=3)
    py.xlabel('X (arcsec)')
    py.gca().get_xaxis().set_major_locator(xtickLoc)
    py.axis('image')
    cbar = py.colorbar(orientation='vertical')
    cbar.set_label('Dispersion, MC err')

    py.tight_layout()
    
    py.savefig(workdir + 'plots/mc_err1.png')
    py.savefig(workdir + 'plots/mc_err1.eps')
    py.show()
    
def plotErr2(inputResults=workdir+'/ppxf.dat',inputAvg=workdir+'/ppxf_avg_mc_nsim100.dat',inputErr=workdir+'/ppxf_errors_mc_nsim100.dat'):
    ### Plots error on h3 and h4
    cubeimg = pyfits.getdata(datadir + cuberoot + '_img.fits')
        
    p = PPXFresults(inputResults)
    a = PPXFresults(inputAvg)
    e = PPXFresults(inputErr)

    xaxis = np.arange(cubeimg.shape[0]) * 0.05
    yaxis = np.arange(cubeimg.shape[1]) * 0.05
    
    xtickLoc = py.MultipleLocator(0.5)

    py.close(2)
    py.figure(2, figsize=(15,13))
    py.subplots_adjust(left=0.05, right=0.94, top=0.95)
    py.clf()

    ##########
    # Plot h3, MC h3 average, and MC h3 error
    ##########
    py.subplot(3, 2, 1)
    h3img = p.h3.transpose()
    py.imshow(np.rot90(py.ma.masked_where(h3img == 0.,h3img),3), vmin=-.2, vmax=.2, 
              extent=[xaxis[0], xaxis[-1], yaxis[0], yaxis[-1]])
    py.plot([bhpos_hor[0]], [bhpos_hor[1]], 'kx', markeredgewidth=3)
    py.ylabel('Y (arcsec)')
    py.xlabel('X (arcsec)')
    py.gca().get_xaxis().set_major_locator(xtickLoc)
    py.axis('image')
    cbar = py.colorbar(orientation='vertical')
    cbar.set_label('h3')

    #pdb.set_trace()
    
    py.subplot(3,2,3)
    h3avg = a.h3.transpose()
    py.imshow(np.rot90(py.ma.masked_where(h3avg==0.,h3avg),3), vmin=-.2, vmax=.2,
              extent=[xaxis[0], xaxis[-1], yaxis[0], yaxis[-1]])
    py.plot([bhpos_hor[0]], [bhpos_hor[1]], 'kx', markeredgewidth=3)
    py.ylabel('Y (arcsec)')
    py.xlabel('X (arcsec)')
    py.gca().get_xaxis().set_major_locator(xtickLoc)
    py.axis('image')
    cbar = py.colorbar(orientation='vertical')
    cbar.set_label('h3, MC avg')

    py.subplot(3,2,5)
    h3err = e.h3.transpose()
    py.imshow(np.rot90(py.ma.masked_where(h3err==0.,h3err),3), vmin=0., vmax=.05,
              extent=[xaxis[0], xaxis[-1], yaxis[0], yaxis[-1]])
    py.plot([bhpos_hor[0]], [bhpos_hor[1]], 'kx', markeredgewidth=3)
    py.ylabel('Y (arcsec)')
    py.xlabel('X (arcsec)')
    py.gca().get_xaxis().set_major_locator(xtickLoc)
    py.axis('image')
    cbar = py.colorbar(orientation='vertical')
    cbar.set_label('h3, MC err')

    ##########
    # Plot h4
    ##########
    py.subplot(3,2,2)
    h4img = p.h4.transpose()
    py.imshow(np.rot90(py.ma.masked_where(h4img==0.,h4img),3), vmin=-.2, vmax=.2,
              extent=[xaxis[0], xaxis[-1], yaxis[0], yaxis[-1]],
              cmap=py.cm.jet)
    py.plot([bhpos_hor[0]], [bhpos_hor[1]], 'kx', markeredgewidth=3)
    py.xlabel('X (arcsec)')
    py.gca().get_xaxis().set_major_locator(xtickLoc)
    py.axis('image')
    cbar = py.colorbar(orientation='vertical')
    cbar.set_label('h4')

    py.subplot(3,2,4)
    h4avg = a.h4.transpose()
    py.imshow(np.rot90(py.ma.masked_where(h4avg==0.,h4avg),3), vmin=-.2, vmax=.2,
              extent=[xaxis[0], xaxis[-1], yaxis[0], yaxis[-1]],
              cmap=py.cm.jet)
    py.plot([bhpos_hor[0]], [bhpos_hor[1]], 'kx', markeredgewidth=3)
    py.xlabel('X (arcsec)')
    py.gca().get_xaxis().set_major_locator(xtickLoc)
    py.axis('image')
    cbar = py.colorbar(orientation='vertical')
    cbar.set_label('h4, MC avg')

    py.subplot(3,2,6)
    h4err = e.h4.transpose()
    py.imshow(np.rot90(py.ma.masked_where(h4err==0.,h4err),3), vmin=0., vmax=.05,
              extent=[xaxis[0], xaxis[-1], yaxis[0], yaxis[-1]],
              cmap=py.cm.jet)
    py.plot([bhpos_hor[0]], [bhpos_hor[1]], 'kx', markeredgewidth=3)
    py.xlabel('X (arcsec)')
    py.gca().get_xaxis().set_major_locator(xtickLoc)
    py.axis('image')
    cbar = py.colorbar(orientation='vertical')
    cbar.set_label('h4, MC err')

    py.tight_layout()

    py.savefig(workdir + 'plots/mc_err2.png')
    py.savefig(workdir + 'plots/mc_err2.eps')
    py.show()

def plotQuality():
    cubeimg = pyfits.getdata(datadir + cuberoot + '_img.fits')
    expimg = pyfits.getdata(datadir + cuberoot + '.fits',ext=3)
    snrimg = pyfits.getdata(datadir + cuberoot + '_snr.fits')

    print cubeimg.shape
    print bhpos
    xaxis = np.arange(cubeimg.shape[0]) * 0.05
    yaxis = np.arange(cubeimg.shape[1]) * 0.05
    
    xtickLoc = py.MultipleLocator(0.5)

    #py.figure(2, figsize=(15,8))
    py.figure(2, figsize=(7,15))
    py.subplots_adjust(left=0.01, right=0.94, top=0.95)
    py.clf()

    ##########
    # Plot Cube Image
    ##########
    #py.subplot(1, 3, 1)
    py.subplot(3, 1, 1)
    #cubeimg=cubeimg.transpose()
    #cubeimg=np.rot90(cubeimg)
    py.imshow(np.rot90((py.ma.masked_where(cubeimg<-10000, cubeimg)),3), 
              extent=[xaxis[0], xaxis[-1], yaxis[0], yaxis[-1]],
              cmap=py.cm.hot)
    py.plot([bhpos_hor[0]], [bhpos_hor[1]], 'kx')
    py.ylabel('Y (arcsec)')
    py.xlabel('X (arcsec)')
    py.gca().get_xaxis().set_major_locator(xtickLoc)
    py.axis('image')
    cbar = py.colorbar(orientation='vertical')
    cbar.set_label('Flux (cts/sec)')

    # Make a compass rose
    pa = 56.0
    cosSin = np.array([ math.cos(math.radians(pa)), 
                        math.sin(math.radians(pa)) ])
    arr_base = np.array([ xaxis[-1]-0.2, yaxis[-1]-0.6 ])
    arr_n = cosSin * 0.2
    arr_w = cosSin[::-1] * 0.2
    arr_n_rot90 = cosSin[::-1] * 0.2
    arr_w_rot90 = cosSin * 0.2
    #py.arrow(arr_base[0], arr_base[1], arr_n[0], -arr_n[1],
    py.arrow(arr_base[0], arr_base[1], -arr_n_rot90[0], arr_n_rot90[1],
             edgecolor='w', facecolor='w', width=0.03, head_width=0.08)
    #py.arrow(arr_base[0], arr_base[1], -arr_w[0], arr_w[1],
    py.arrow(arr_base[0], arr_base[1], -arr_w_rot90[0], -arr_w_rot90[1],
             edgecolor='w', facecolor='w', width=0.03, head_width=0.08)
    py.text(arr_base[0]-arr_n[0]-.25, arr_base[1]-arr_n[1]-0.28, 'E', 
            color='white', 
            horizontalalignment='left', verticalalignment='bottom')
    py.text(arr_base[0]-arr_n_rot90[0]-0.18, arr_base[1]+arr_n_rot90[1]+0.1, 'N', 
            color='white',
            horizontalalignment='right', verticalalignment='center')
    py.title('K Image')


    ##########
    # Plot SNR Image
    ##########
    print snrimg[30,10]
    #py.subplot(1, 3, 2)
    py.subplot(3, 1, 2)
    snrimg=snrimg.transpose()
    snrimg=np.rot90(snrimg,3) 
    py.imshow(py.ma.masked_invalid(snrimg), 
              extent=[xaxis[0], xaxis[-1], yaxis[0], yaxis[-1]],
              cmap=py.cm.jet)
    py.plot([bhpos_hor[0]], [bhpos_hor[1]], 'kx')
    py.xlabel('X (arcsec)')
    py.gca().get_xaxis().set_major_locator(xtickLoc)
    py.axis('image')
    cbar = py.colorbar(orientation='vertical')
    cbar.set_label('SNR')
    py.title('SNR')

    ##########
    # Plot number of exposures
    ##########
    #py.subplot(1, 3, 3)
    py.subplot(3, 1, 3)
    expimg=expimg[:,:,700]
    expimg=expimg.transpose()
    py.imshow(np.rot90(expimg,3),
              extent=[xaxis[0], xaxis[-1], yaxis[0], yaxis[-1]])
    py.plot([bhpos_hor[0]], [bhpos_hor[1]], 'kx')
    py.xlabel('X (arcsec)')
    py.gca().get_xaxis().set_major_locator(xtickLoc)
    py.axis('image')
    cbar = py.colorbar(orientation='vertical')
    cbar.set_label('Exposures')
    py.title('Exposures')

    py.tight_layout()

    py.savefig(workdir + 'plots/quality_maps.png')
    py.savefig(workdir + 'plots/quality_maps.eps')
    py.show()


    
def plotPDF(inputResults=workdir+'/ppxf_test_mc_nsim100.dat',inputError=workdir+'/ppxf_errors_mc_nsim100.dat'):
    ### Plot PDF of results from runErrorMC(test=True)

    p = PPXFresults(inputResults)
    e = PPXFresults(inputError)

    spax = np.where(e.velocity != 0.)
    nspax = len(spax[0])
    
    py.close(2)
    py.figure(2)
    py.clf()

    for i in range(nspax):
        py.clf()
        ny, bins, patches = py.hist(p.velocity[i,:]+308., 50, normed=1, facecolor='green', alpha=0.75)
        py.xlabel('Velocity (km/s)')
        py.ylabel('Frequency')
        titpl = 'Spaxel [%(xx)s,%(yy)s]' % {"xx": spax[0][i], "yy": spax[1][i]}
        py.title(titpl)
        spl = 'Err = %6.2f' % e.velocity[spax[0][i],spax[1][i]]
        py.annotate(xy=(.05,.9),s=spl,xycoords='axes fraction')
        outstem = workdir + 'plots/velocity_pdf_%(xx)s_%(yy)s' % {"xx": spax[0][i], "yy": spax[1][i]}
        py.savefig(outstem + '.png')
        #py.savefig(outstem + '.eps')

    for i in range(nspax):
        py.clf()
        ny, bins, patches = py.hist(p.sigma[i,:], 50, normed=1, facecolor='green', alpha=0.75)
        py.xlabel('Sigma (km/s)')
        py.ylabel('Frequency')
        titpl = 'Spaxel [%(xx)s,%(yy)s]' % {"xx": spax[0][i], "yy": spax[1][i]}        
        py.title(titpl)
        spl = 'Err = %6.2f' % e.sigma[spax[0][i],spax[1][i]]
        py.annotate(xy=(.05,.9),s=spl,xycoords='axes fraction')
        outstem = workdir + 'plots/sigma_pdf_%(xx)s_%(yy)s'  % {"xx": spax[0][i], "yy": spax[1][i]}
        py.savefig(outstem + '.png')
        #py.savefig(outstem + '.eps')
    
def precessionSpeed():
    cubeimg = pyfits.getdata(datadir + cuberoot + '_img.fits')
    snrimg = pyfits.getdata(datadir + cuberoot + '_snr.fits')

    p = PPXFresults()

    #bhpos = np.array([9.0, 38.0])
    xaxis = (np.arange(p.velocity.shape[0]) - bhpos[0]) * 0.05
    yaxis = (np.arange(p.velocity.shape[1]) - bhpos[1]) * 0.05
    
    # Pull out a profile along the line-of-nodes (Y-axis)
    binSize = 0.05 # * 3.77
    # Convert Surface Brightness from cts/s to mag/arcsec^2
    cubeimg += cubeimg.mean() - (cubeimg.std() * 3.0)
    fluxCube = cubeimg / binSize**2
    magCube = -2.5 * np.log10(cubeimg) + 23.7

    light = fluxCube[:,bhpos[0]]
    velocity = p.velocity[bhpos[0],:] + 308.0
    dispersion = p.sigma[bhpos[0],:]

    py.figure(1)
    py.clf()
    py.subplots_adjust(left=0.13)
    py.plot(yaxis, light)
    limits = py.axis()
    py.plot([0, 0], [limits[2], limits[3]], 'k--')
    py.xlabel('Distance Along Line-of-Nodes (")')
    py.ylabel('Surface Brightness (arbitrary flux/square arcsec)')
    py.xlim(-1.3, 1.3)
    py.savefig(workdir + 'plots/line_of_nodes_flux.png')
    #py.show()

    py.clf()
    py.subplots_adjust(left=0.13)
    py.plot(yaxis, velocity)
    py.xlabel('Distance Along Line-of-Nodes (")')
    py.ylabel('Velocity (km/s)')
    limits = py.axis()
    py.plot([0, 0], [limits[2], limits[3]], 'k--')
    py.plot([limits[0], limits[1]], [0, 0], 'k--')
    py.ylim(-300, 300)
    py.xlim(-1.3, 1.3)
    py.savefig(workdir + 'plots/line_of_nodes_velocity.png')
    #py.show()

    py.clf()
    py.subplots_adjust(left=0.13)
    py.plot(yaxis, dispersion)
    py.xlabel('Distance Along Line-of-Nodes (")')
    py.ylabel('Velocity Dispersion (km/s)')
    limits = py.axis()
    py.plot([0, 0], [limits[2], limits[3]], 'k--')
    py.plot([limits[0], limits[1]], [0, 0], 'k--')
    py.xlim(-1.3, 1.3)
    py.savefig(workdir + 'plots/line_of_nodes_dispersion.png')
    #py.show()

    # Calculate the Sigma_P * sin(i) (precession speed)
    numerator = light * velocity
    denominator = light * (yaxis * 3.77)

    # Sum it all up, but only over a fixed radial extent
    idx = np.where(np.abs(yaxis) < 1.2)[0]
    
    sigPsinI = numerator[idx].sum() / denominator[idx].sum()
    sigP = sigPsinI / math.sin(math.radians(55.0))
    print 'Precession Speed * sin(i) = %6.2f km/s/pc' % sigPsinI
    print 'Precession Speed (i=55 d) = %6.2f km/s/pc' % sigP
    
    # Calculate for several lines
    numAll = np.zeros(fluxCube.shape[0], dtype=float)
    denomAll = np.zeros(fluxCube.shape[0], dtype=float)
    sigPsinIall = np.zeros(fluxCube.shape[0], dtype=float)
    for xx in range(4, 17):
        lightTmp = fluxCube[idx,xx]
        velocityTmp = p.velocity[xx,idx] + 308.0
        numAll[xx] = (lightTmp * velocityTmp).sum() / lightTmp.sum()
        denomAll[xx] = (lightTmp * yaxis[idx] * 3.77).sum() / lightTmp.sum()
        sigPsinIall[xx] = numAll[xx] / denomAll[xx]
    print sigPsinIall
    foo = np.where(numAll != 0)[0]
    py.clf()
    py.plot(denomAll[foo]/3.77, numAll[foo], 'k.')
    py.show()

    # Calculate for a bunch of different velocity errors
    verrs = np.arange(3.0, 30.0, 1.0)

    denomSum = denominator[idx].sum()

    sigPErrs = np.zeros(len(verrs), dtype=float)

    for ii in range(len(verrs)):
        sigPsinIErr = np.sqrt((light[idx]**2).sum())
        sigPsinIErr *= verrs[ii] * binSize / denomSum
        sigPErr = sigPsinIErr / abs(math.sin(math.radians(55.0)))

        sigPErrs[ii] = sigPErr

#         print '   verr = %4.1f, error in precession speed = %6.2f km/s/pc' % \
#             (verrs[ii], sigPErr)

    py.clf()
    py.plot(verrs, sigPErrs)
    #py.show()

def plotModelKinematics(inputFile=None,nonaligned=True,clean=False,trim=False):

    if inputFile:
        inputFile=inputFile
    else:
        if nonaligned:
            if clean:
                inputFile = modeldir + 'nonaligned_OSIRIScoords_fits_clean.dat'
            else:
                inputFile = modeldir + 'nonaligned_OSIRIScoords_fits_full.dat'
        else:
            if clean:
                inputFile = modeldir + 'aligned_OSIRIScoords_fits_clean.dat'
            else:
                inputFile = modeldir + 'aligned_OSIRIScoords_fits_full.dat'

    model = modelFitResults(inputFile)

    # trim model results to match OSIRIS FOV
    #trimrange = [[41.,88.],[21.,102.]]
    # updated trim to match new BH coordinates
    #trimrange = [[43.5,84.5],[17.5,101.5]]
    # use trimModel() instead
    if trim:
        modbhpos = bhpos_hor
        #xaxis = np.arange(trimrange[0][1]-trimrange[0][0]) * 0.05
        #yaxis = np.arange(trimrange[1][1]-trimrange[1][0]) * 0.05
        xaxis = np.arange(84) * 0.05
        yaxis = np.arange(41) * 0.05
    else:
        # modbhpos given by modelBin
        # PA = -34
        if nonaligned:
            modbhpos = [(model.velocity.shape[1]-59.035) * 0.05, 63.955 * 0.05]
        else:
            modbhpos = [(model.velocity.shape[1]-59.035) * 0.05, 54.955 * 0.05] 
        xaxis = np.arange(model.velocity.shape[1]) * 0.05
        yaxis = np.arange(model.velocity.shape[0]) * 0.05
    
    xtickLoc = py.MultipleLocator(0.5)

    py.close(2)
    py.figure(2, figsize=(10,12))
    py.subplots_adjust(left=0.05, right=0.94, top=0.95)
    py.clf()

    ##########
    # Plot Velocity
    ##########
    py.subplot(2, 2, 1)
    if trim:
        velimg = trimModel(model.velocity,nonaligned=nonaligned)
    else:
        velimg = model.velocity
    py.imshow(np.rot90(velimg.transpose(),3), vmin=-250., vmax=250.,
    #py.imshow(velimg.transpose(), vmin=-50., vmax=50.,
              extent=[xaxis[0], xaxis[-1], yaxis[0], yaxis[-1]])
    py.plot([modbhpos[0]], [modbhpos[1]], 'kx', markeredgewidth=3)
    py.ylabel('Y (arcsec)')
    py.xlabel('X (arcsec)')
    py.gca().get_xaxis().set_major_locator(xtickLoc)
    cbar = py.colorbar(orientation='vertical')
    cbar.set_label('Velocity (km/s)')    

    ##########
    # Plot Dispersion
    ##########
    py.subplot(2, 2, 2)
    if trim:
        sigimg = trimModel(model.sigma,nonaligned=nonaligned)
    else:
        sigimg = model.sigma
    py.imshow(np.rot90(sigimg.transpose(),3), vmin=0., vmax=250.,
                extent=[xaxis[0], xaxis[-1], yaxis[0], yaxis[-1]],
                cmap=py.cm.jet)
    py.plot([modbhpos[0]], [modbhpos[1]], 'kx', markeredgewidth=3)
    py.xlabel('X (arcsec)')
    py.gca().get_xaxis().set_major_locator(xtickLoc)
    cbar = py.colorbar(orientation='vertical')
    cbar.set_label('Dispersion (km/s)')

    ##########
    # Plot h3
    ##########
    py.subplot(2, 2, 3)
    if trim:
        h3img = trimModel(model.h3,nonaligned=nonaligned)
        h3img = np.rot90(h3img.transpose(),3)
    else:
        h3img = np.rot90(model.h3.transpose(),3)
    py.imshow(py.ma.masked_where(h3img == 0.,h3img), vmin=-.2, vmax=.2,
    #py.imshow(py.ma.masked_where(h3img == 0.,h3img), vmin=-.05, vmax=.05, 
              extent=[xaxis[0], xaxis[-1], yaxis[0], yaxis[-1]])
    py.plot([modbhpos[0]], [modbhpos[1]], 'kx', markeredgewidth=3)
    py.ylabel('Y (arcsec)')
    py.xlabel('X (arcsec)')
    py.gca().get_xaxis().set_major_locator(xtickLoc)
    cbar = py.colorbar(orientation='vertical')
    cbar.set_label('h3')

    ##########
    # Plot h4
    ##########
    py.subplot(2, 2, 4)
    if trim:
        h4img = trimModel(model.h4,nonaligned=nonaligned)
        h4img = np.rot90(h4img.transpose(),3)
    else:
        h4img = np.rot90(model.h4.transpose(),3)
    py.imshow(py.ma.masked_where(h4img==0.,h4img), vmin=-.2, vmax=.2,
    #py.imshow(py.ma.masked_where(h4img==0.,h4img), vmin=-.05, vmax=.05,
              extent=[xaxis[0], xaxis[-1], yaxis[0], yaxis[-1]],
              cmap=py.cm.jet)
    py.plot([modbhpos[0]], [modbhpos[1]], 'kx', markeredgewidth=3)
    py.xlabel('X (arcsec)')
    py.gca().get_xaxis().set_major_locator(xtickLoc)
    cbar = py.colorbar(orientation='vertical')
    cbar.set_label('h4')

    #pdb.set_trace()

    py.tight_layout()

    py.savefig(modelworkdir + 'plots/model_kinematics.png')
    py.savefig(modelworkdir + 'plots/model_kinematics.eps')
    py.show()

def plotDataModelResiduals(inputData=workdir+'ppxf.dat',inputModel=modeldir+'nonaligned_OSIRIScoords_fits_full_smooth.dat',inputRes=workdir+'model_residuals.dat',nonaligned=True):

    data = PPXFresults(inputData)
    model = modelFitResults(inputModel)
    res = modelFitResults(inputRes)

    cubeimg = pyfits.getdata(datadir + cuberoot + '_img.fits')

    xaxis = np.arange(cubeimg.shape[0]) * 0.05
    yaxis = np.arange(cubeimg.shape[1]) * 0.05
    
    xtickLoc = py.MultipleLocator(0.5)

    py.close(2)
    py.figure(2, figsize=(7,15))
    py.subplots_adjust(left=0.05, right=0.94, top=0.95)
    py.clf()

    ##########
    # Plot flux/nstar
    ##########
    py.subplot(3,1,1)
    py.imshow(np.rot90(py.ma.masked_where(cubeimg==0.,cubeimg/cubeimg.max()),3),vmin=0.,vmax=1.,
              extent=[xaxis[0], xaxis[-1], yaxis[0], yaxis[-1]])
    py.plot([bhpos_hor[0]], [bhpos_hor[1]], 'kx', markeredgewidth=3)
    #py.ylabel('Y (arcsec)')
    py.ylabel('Data')
    py.xlabel('X (arcsec)')
    py.gca().get_xaxis().set_major_locator(xtickLoc)
    py.axis('image')
    cbar = py.colorbar(orientation='vertical')
    cbar.set_label('Flux (norm)')

    py.subplot(3,1,2)
    py.imshow(np.rot90((trimModel(model.nstar,nonaligned=nonaligned).transpose()/trimModel(model.nstar,nonaligned=nonaligned).max().transpose()),3),vmin=0.,vmax=1.,
            extent=[xaxis[0], xaxis[-1], yaxis[0], yaxis[-1]])
    py.plot([bhpos_hor[0]], [bhpos_hor[1]], 'kx', markeredgewidth=3)
    #py.ylabel('Y (arcsec)')
    py.ylabel('Model')
    py.xlabel('X (arcsec)')
    py.gca().get_xaxis().set_major_locator(xtickLoc)
    py.axis('image')
    cbar = py.colorbar(orientation='vertical')
    cbar.set_label('Model, number of stars (norm)')

    py.subplot(3,1,3)
    py.imshow(np.rot90(py.ma.masked_where(cubeimg==0.,res.nstar.transpose()),3),vmin=-.25,vmax=.25,
              extent=[xaxis[0], xaxis[-1], yaxis[0], yaxis[-1]])
    py.plot([bhpos_hor[0]], [bhpos_hor[1]], 'kx', markeredgewidth=3)
    #py.ylabel('Y (arcsec)')
    py.ylabel('Residuals')
    py.xlabel('X (arcsec)')
    py.gca().get_xaxis().set_major_locator(xtickLoc)
    py.axis('image')
    cbar = py.colorbar(orientation='vertical')
    cbar.set_label('Normed flux residuals')

    py.tight_layout()

    py.savefig(modelworkdir + 'plots/residuals_flux.png')
    py.savefig(modelworkdir + 'plots/residuals_flux.eps')
    py.show()
    pdb.set_trace()
    ##########
    # Plot velocity
    ##########  
    py.clf()

    py.subplot(3,1,1)
    py.imshow(np.rot90(py.ma.masked_where((data.velocity.transpose()+308.)==308.,data.velocity.transpose()+308.),3),vmin=-250.,vmax=250., 
              extent=[xaxis[0], xaxis[-1], yaxis[0], yaxis[-1]])
    py.plot([bhpos_hor[0]], [bhpos_hor[1]], 'kx', markeredgewidth=3)
    #py.ylabel('Y (arcsec)')
    py.ylabel('Data')
    py.xlabel('X (arcsec)')
    py.gca().get_xaxis().set_major_locator(xtickLoc)
    py.axis('image')
    cbar = py.colorbar(orientation='vertical')
    cbar.set_label('Velocity (km/s)')
    
    py.subplot(3,1,2)
    py.imshow(np.rot90(trimModel(model.velocity,nonaligned=nonaligned).transpose(),3),vmin=-250.,vmax=250., 
            extent=[xaxis[0], xaxis[-1], yaxis[0], yaxis[-1]])
    py.plot([bhpos_hor[0]], [bhpos_hor[1]], 'kx', markeredgewidth=3)
    #py.ylabel('Y (arcsec)')
    py.ylabel('Model')
    py.xlabel('X (arcsec)')
    py.gca().get_xaxis().set_major_locator(xtickLoc)
    py.axis('image')
    cbar = py.colorbar(orientation='vertical')
    cbar.set_label('Model velocity (km/s)')

    py.subplot(3,1,3)
    py.imshow(np.rot90(res.velocity.transpose(),3),vmin=-100.,vmax=200., 
              extent=[xaxis[0], xaxis[-1], yaxis[0], yaxis[-1]])
    py.plot([bhpos_hor[0]], [bhpos_hor[1]], 'kx', markeredgewidth=3)
    #py.ylabel('Y (arcsec)')
    py.ylabel('Residuals')
    py.xlabel('X (arcsec)')
    py.gca().get_xaxis().set_major_locator(xtickLoc)
    py.axis('image')
    cbar = py.colorbar(orientation='vertical')
    cbar.set_label('Velocity residuals (km/s)')

    py.tight_layout()

    py.savefig(modelworkdir + 'plots/residuals_velocity.png')
    py.savefig(modelworkdir + 'plots/residuals_velocity.eps')
    py.show()

    ##########
    # Plot sigma
    ##########  
    py.clf()

    py.subplot(3,1,1)
    py.imshow(np.rot90(py.ma.masked_where(data.sigma.transpose()==0.,data.sigma.transpose()),3),vmin=0.,vmax=250., 
              extent=[xaxis[0], xaxis[-1], yaxis[0], yaxis[-1]])
    py.plot([bhpos_hor[0]], [bhpos_hor[1]], 'kx', markeredgewidth=3)
    #py.ylabel('Y (arcsec)')
    py.ylabel('Data')
    py.xlabel('X (arcsec)')
    py.gca().get_xaxis().set_major_locator(xtickLoc)
    py.axis('image')
    cbar = py.colorbar(orientation='vertical')
    cbar.set_label('Sigma (km/s)')

    py.subplot(3,1,2)
    py.imshow(np.rot90(trimModel(model.sigma,nonaligned=nonaligned).transpose(),3),vmin=0.,vmax=250., 
            extent=[xaxis[0], xaxis[-1], yaxis[0], yaxis[-1]])
    py.plot([bhpos_hor[0]], [bhpos_hor[1]], 'kx', markeredgewidth=3)
    #py.ylabel('Y (arcsec)')
    py.ylabel('Model')
    py.xlabel('X (arcsec)')
    py.gca().get_xaxis().set_major_locator(xtickLoc)
    py.axis('image')
    cbar = py.colorbar(orientation='vertical')
    cbar.set_label('Model sigma (km/s)')

    py.subplot(3,1,3)
    py.imshow(np.rot90(res.sigma.transpose(),3),vmin=-100.,vmax=100., 
              extent=[xaxis[0], xaxis[-1], yaxis[0], yaxis[-1]])
    py.plot([bhpos_hor[0]], [bhpos_hor[1]], 'kx', markeredgewidth=3)
    #py.ylabel('Y (arcsec)')
    py.ylabel('Residuals')
    py.xlabel('X (arcsec)')
    py.gca().get_xaxis().set_major_locator(xtickLoc)
    py.axis('image')
    cbar = py.colorbar(orientation='vertical')
    cbar.set_label('Sigma residuals (km/s)')

    py.tight_layout()

    py.savefig(modelworkdir + 'plots/residuals_sigma.png')
    py.savefig(modelworkdir + 'plots/residuals_sigma.eps')
    py.show()

    ##########
    # Plot h3
    ##########
    py.close(2)
    py.figure(2, figsize=(7,15))
    py.subplots_adjust(left=0.05, right=0.94, top=0.95) 
    py.clf()

    py.subplot(3,1,1)
    py.imshow(np.rot90(py.ma.masked_where(data.h3.transpose()==0.,data.h3.transpose()),3),vmin=-0.2,vmax=0.2, extent=[xaxis[0], xaxis[-1], yaxis[0], yaxis[-1]])
    py.plot([bhpos_hor[0]], [bhpos_hor[1]], 'kx', markeredgewidth=3)
    #py.ylabel('Y (arcsec)')
    py.ylabel('Data')
    py.xlabel('X (arcsec)')
    py.gca().get_xaxis().set_major_locator(xtickLoc)
    py.axis('image')
    cbar = py.colorbar(orientation='vertical')
    cbar.set_label('h3')

    py.subplot(3,1,2)
    py.imshow(np.rot90(trimModel(model.h3,nonaligned=nonaligned).transpose(),3),vmin=-0.2,vmax=0.2, extent=[xaxis[0], xaxis[-1], yaxis[0], yaxis[-1]])
    py.plot([bhpos_hor[0]], [bhpos_hor[1]], 'kx', markeredgewidth=3)
    #py.ylabel('Y (arcsec)')
    py.ylabel('Model')
    py.xlabel('X (arcsec)')
    py.gca().get_xaxis().set_major_locator(xtickLoc)
    py.axis('image')
    cbar = py.colorbar(orientation='vertical')
    cbar.set_label('Model h3')

    py.subplot(3,1,3)
    py.imshow(np.rot90(res.h3.transpose(),3),vmin=-0.2,vmax=0.2, extent=[xaxis[0], xaxis[-1], yaxis[0], yaxis[-1]])
    py.plot([bhpos_hor[0]], [bhpos_hor[1]], 'kx', markeredgewidth=3)
    #py.ylabel('Y (arcsec)')
    py.ylabel('Residuals')
    py.xlabel('X (arcsec)')
    py.gca().get_xaxis().set_major_locator(xtickLoc)
    py.axis('image')
    cbar = py.colorbar(orientation='vertical')
    cbar.set_label('h3 residuals')

    py.tight_layout()

    py.savefig(modelworkdir + 'plots/residuals_h3.png')
    py.savefig(modelworkdir + 'plots/residuals_h3.eps')
    py.show()

    py.close(2)
    py.figure(2, figsize=(7,15))
    py.subplots_adjust(left=0.05, right=0.94, top=0.95)
    py.clf()
    
    ##########
    # Plot h4
    ##########  
    py.clf()

    py.subplot(3,1,1)
    py.imshow(np.rot90(py.ma.masked_where(data.h4.transpose()==0.,data.h4.transpose()),3),vmin=-0.2,vmax=0.2, extent=[xaxis[0], xaxis[-1], yaxis[0], yaxis[-1]])
    py.plot([bhpos_hor[0]], [bhpos_hor[1]], 'kx', markeredgewidth=3)
    #py.ylabel('Y (arcsec)')
    py.ylabel('Data')
    py.xlabel('X (arcsec)')
    py.gca().get_xaxis().set_major_locator(xtickLoc)
    py.axis('image')
    cbar = py.colorbar(orientation='vertical')
    cbar.set_label('h4')

    py.subplot(3,1,2)
    py.imshow(np.rot90(trimModel(model.h4,nonaligned=nonaligned).transpose(),3),vmin=-0.2,vmax=0.2, extent=[xaxis[0], xaxis[-1], yaxis[0], yaxis[-1]])
    py.plot([bhpos_hor[0]], [bhpos_hor[1]], 'kx', markeredgewidth=3)
    #py.ylabel('Y (arcsec)')
    py.ylabel('Model')
    py.xlabel('X (arcsec)')
    py.gca().get_xaxis().set_major_locator(xtickLoc)
    py.axis('image')
    cbar = py.colorbar(orientation='vertical')
    cbar.set_label('Model h4')

    py.subplot(3,1,3)
    py.imshow(np.rot90(res.h4.transpose(),3),vmin=-0.2,vmax=0.2, extent=[xaxis[0], xaxis[-1], yaxis[0], yaxis[-1]])
    py.plot([bhpos_hor[0]], [bhpos_hor[1]], 'kx', markeredgewidth=3)
    #py.ylabel('Y (arcsec)')
    py.ylabel('Residuals')
    py.xlabel('X (arcsec)')
    py.gca().get_xaxis().set_major_locator(xtickLoc)
    py.axis('image')
    cbar = py.colorbar(orientation='vertical')
    cbar.set_label('h4 residuals')

    py.tight_layout()

    py.savefig(modelworkdir + 'plots/residuals_h4.png')
    py.savefig(modelworkdir + 'plots/residuals_h4.eps')
    py.show()



def trimModel(input,nonaligned=True):

    # trim models to match OSIRIS FOV
    # set up for PA = -34
    #trimrange = [[41.,88.],[21.,102.]]
    # new trimrange to match new BH position
    #trimrange = [[43.5,84.5],[17.5,101.5]]
    # BHpos (data) = [19.1,  40.7]
    if nonaligned:
        # BHpos (nonaligned) = [63.954999999999998, 59.034999999999997]
        trimrange = [[44.9,85.9],[18.3,102.3]]
    else:
        # BHpos (aligned) = [54.954999999999998, 59.034999999999997]
        trimrange = [[35.9, 75.9],[18.3,102.3]]
        
    return input[trimrange[0][0]:trimrange[0][1],trimrange[1][0]:trimrange[1][1]]
    
    
def modelBin(inputFile=None,nonaligned=True,clean=False):
    ### Reads in model results from Peiris & Tremaine 2003 (coordinates transformed to
    #### match OSIRIS observations), bins individual stellar particles to match the
    #### spaxel size in observations, and fits LOSVDs to the resulting velocity
    #### histograms.

    ### clean: keyword to control how LOSVD are fit. With clean=False, LOSVDs are fit
    #### to all bins. If no fit is possible (e.g. if there are too few particles in the
    #### bin), the mean velocity is taken as the ensemble velocity and 0 is recorded for
    #### the higher moments (sigma, h3, h4). With clean=True, 0 is also recorded for the
    #### ensemble velocity in these bins. In addition, marginal fits (where OptimizeWarning
    #### is returned by the fitter) are also discarded and 0 recorded for all kinematic
    #### parameters.

    if clean:
        from scipy.optimize import OptimizeWarning

    if inputFile:
        model=modelResults(inputFile)
    else:
        model = modelResults(nonaligned=nonaligned,skycoords=True,OSIRIS=True)

    # bin size = 0.05" = 0.1865 pc
    binpc = 0.1865
    
    # BH position in the OSIRIS data: [22.5, 37.5]. Matching pixel phase, 0.5*0.05" = 0.09325 pc:
    #model.x += 0.09325
    #model.y += 0.09325
    # new BH position: [19.1, 40.6]
    xfrac = bhpos[0]-np.floor(bhpos[0])
    yfrac = bhpos[1]-np.floor(bhpos[1])
    model.x += (binpc*xfrac)
    model.y += (binpc*yfrac)

    # get the full size of the binned array, but making sure to leave bin boundaries on the axes
    posxbin = np.ceil(np.max(model.x)/binpc)
    negxbin = np.ceil(np.abs(np.min(model.x)/binpc))
    nxbin = posxbin+negxbin
    posybin = np.ceil(np.max(model.y)/binpc)
    negybin = np.ceil(np.abs(np.min(model.y)/binpc))
    nybin = posybin + negybin

    #modbhpos = [negxbin+0.5,negybin+0.5]
    modbhpos = [negxbin+xfrac,negybin+yfrac]
    print "Model BH is at ", modbhpos
    #pdb.set_trace()

    nstar = np.zeros((nxbin,nybin))
    losv = np.zeros((nxbin,nybin))
    sigma = np.zeros((nxbin,nybin))
    h3 = np.zeros((nxbin,nybin))
    h4 = np.zeros((nxbin,nybin))

    leftxbound = np.arange(-1.*negxbin*binpc,posxbin*binpc,binpc)
    bottomybound = np.arange(-1*negybin*binpc,posybin*binpc,binpc)

    t1 = time.time()

    # create the irange array
    # this array starts in the center then steps its way out, alternating sides
    # the maximum number of particles are near the center, so this should speed up the where
    # statements (b/c when the loop hits the vast deserted outskirts, there's little left in the master array)
    for i in range(int(nxbin)):
        if i==0:
            irangetmp = np.array([np.floor(nxbin/2)])
        else:
            tmp = irangetmp[-1] + (((-1)**i)*i)
            irangetmp = np.append(irangetmp,tmp)
    badilo = np.where(irangetmp < 0)
    irangetmp2 = np.delete(irangetmp,badilo)
    badihi = np.where(irangetmp2 > len(leftxbound))
    irange = np.delete(irangetmp2,badihi)
    irange = irange.astype(int)
    for j in range(int(nybin)):
        if j == 0:
            jrangetmp = np.array([np.floor(nybin/2)])
        else:
            tmp = jrangetmp[-1] + (((-1)**j)*j)
            jrangetmp = np.append(jrangetmp,tmp)
    badjlo = np.where(jrangetmp < 0)
    jrangetmp2 = np.delete(jrangetmp,badjlo)
    badjhi = np.where(jrangetmp2 > len(bottomybound))
    jrange = np.delete(jrangetmp2,badjhi)
    jrange = jrange.astype(int)
    
    # this loop is solely for binning the particles and writing their locations/velocities
    # to a separate file for each bin
    # first, check to see if the files already exist (checking the first one)
    if os.path.isfile(modeldir + 'spax/spax_0_0.dat') is False:
        # make array copies 
        modx = np.array(model.x)
        mody = np.array(model.y)
        modz = np.array(model.z)
        modvx = np.array(model.vx)
        modvy = np.array(model.vy)
        modvz = np.array(model.vz)
        for i in irange:
            for j in jrange:
                good = np.where((modx >= leftxbound[i]) & (modx < leftxbound[i]+binpc) & (mody >= bottomybound[j]) & (mody < bottomybound[j]+binpc))
                # nstar is the number of star particles in the bin
                ntmp = good[0].shape[0]
                nstar[i,j] = ntmp
                # write just these particles out to a file
                outputFile = modeldir + 'spax/spax_'+str(int(i))+'_'+str(int(j))+'.dat'
                tmpx = modx[good[0]]
                tmpy = mody[good[0]]
                tmpz = modz[good[0]]
                tmpvx = modvx[good[0]]
                tmpvy = modvy[good[0]]
                tmpvz = modvz[good[0]]
                np.savetxt(outputFile,np.c_[tmpx,tmpy,tmpz,tmpvx,tmpvy,tmpvz],fmt='%8.6f',delimiter='\t')
                # remove the written out particles from the master arrays, to improve computational time
                modx = np.delete(modx,good[0])
                mody = np.delete(mody,good[0])
                modz = np.delete(modz,good[0])
                modvx = np.delete(modvx,good[0])
                modvy = np.delete(modvy,good[0])
                modvz = np.delete(modvz,good[0])
                
    print "Time elapsed for binning: ", time.time() - t1, "s"
    t2 = time.time()

    #pdb.set_trace()

    if clean:
        warnings.simplefilter("error", OptimizeWarning)            
    # this loop does the LOSVD fits for each bin, to get the kinematics
    for i in range(int(nxbin)):
        print "Starting column ", i
        for j in range(int(nybin)):
            model = modelResults(inputFile = modeldir + 'spax/spax_'+str(i)+'_'+str(j)+'.dat')
            # make sure that nstar is populated - if binning code and outputs were run separately
            # previously, it will not yet have been filled in
            if nstar[i,j] == 0.:
                nstar[i,j] = model.x.shape[0]
            # v_LOS = -v_z
            modvLOS = -1.* model.vz
            if model.x.shape[0] == 0:
                losv[i,j] = 0.
                sigma[i,j] = 0.
                h3[i,j] = 0.
                h4[i,j] = 0.
            elif model.x.shape[0] <= 6:
                if clean:
                    losv[i,j] = 0.
                else:
                    losv[i,j] = np.mean(modvLOS)
                sigma[i,j] = 0.
                h3[i,j] = 0.
                h4[i,j] = 0.
            else:
                # binning in widths of 5 km/s, as in Peiris & Tremaine 2003
                #binsize = 5.
                binsize = 1.
                ny, bins = py.histogram(modvLOS,bins=np.arange(min(modvLOS),max(modvLOS)+binsize,binsize))
                gamma0 = ny.sum()
                v0 = np.mean(modvLOS)
                s0 = np.std(modvLOS)
                h3_0 = 0.
                h4_0 = 0.
                guess = [gamma0, v0, s0, h3_0, h4_0]
                if clean:
                    try:
                        popt, pcov = curve_fit(gaussHermite, bins[0:-1], ny, p0=guess)
                        # popt = [gamma, v, sigma, h3, h4]
                        losv[i,j] = popt[1]
                        sigma[i,j] = popt[2]
                        h3[i,j] = popt[3]
                        h4[i,j] = popt[4]
                    except (RuntimeError, OptimizeWarning):
                        # if get an error on the fit, drop the bin
                        losv[i,j] = 0.
                        sigma[i,j] = 0.
                        h3[i,j] = 0.
                        h4[i,j] = 0.
                else:
                    try:
                        popt, pcov = curve_fit(gaussHermite, bins[0:-1], ny, p0=guess)
                        # popt = [gamma, v, sigma, h3, h4]
                        losv[i,j] = popt[1]
                        sigma[i,j] = popt[2]
                        h3[i,j] = popt[3]
                        h4[i,j] = popt[4]
                    except (RuntimeError):
                        # if no fit is possible, take the mean
                        losv[i,j] = np.mean(modvLOS)
                        sigma[i,j] = 0.
                        h3[i,j] = 0.
                        h4[i,j] = 0.                    
                py.close()
        # code crashes after exiting this loop (why??), so setting output here for now
        if i == (nxbin-1):
            if nonaligned:
                if clean:
                    output = open(modeldir + 'nonaligned_OSIRIScoords_fits_clean.dat', 'w')
                else:
                    output = open(modeldir + 'nonaligned_OSIRIScoords_fits_full.dat', 'w')
            else:
                if clean:
                    output = open(modeldir + 'aligned_OSIRIScoords_fits_clean.dat', 'w')
                else:
                    output = open(modeldir + 'aligned_OSIRIScoords_fits_full.dat', 'w')                    
        
            pickle.dump(nstar, output)
            pickle.dump(losv, output)
            pickle.dump(sigma, output)
            pickle.dump(h3, output)
            pickle.dump(h4, output)
            output.close()

    print "Time elapsed for LOSVD fitting: ", time.time() - t2, "s"
    #pdb.set_trace()
    
def gaussHermite(x,gamma,v,sigma,h3,h4):
    # Hermite polynomials are from van der Marel & Franx 1993, Appendix A
    a = (x - v)/sigma
    H3 = (1./np.sqrt(6.)) * (2.*np.sqrt(2)*(a**3) - 3.*np.sqrt(2)*a)
    H4 = (1./np.sqrt(24)) * (4.*(a**4) - 12.*(a**2) + 3.)
    #return (gamma / np.sqrt(2.*math.pi*(sigma**0.5))) * np.exp(-(x-v)**2./(2.*(sigma**2.))) * (1. + h3*H3 + h4*H4)
    return (gamma / (sigma*np.sqrt(2.*math.pi))) * np.exp((-0.5)*(a**2.)) * (1. + h3*H3 + h4*H4)

class modelFitResults(object):
    def __init__(self, inputFile=modeldir+'nonaligned_OSIRIScoords_fits_full.dat'):
        self.inputFile = inputFile
        
        input = open(inputFile, 'r')
        self.nstar = pickle.load(input)
        self.velocity = pickle.load(input)
        self.sigma = pickle.load(input)
        self.h3 = pickle.load(input)
        self.h4 = pickle.load(input)

def smoothModels(inputModel=modeldir+'nonaligned_OSIRIScoords_fits_full.dat', inputPSF=workdir+'plots/osir_perf_m31_all_scalederr_cleanhdr_params.txt', twoGauss=False):

    model = modelFitResults(inputFile=inputModel)

    if twoGauss:
        PSFparams = readPSFparams(inputFile=inputPSF,twoGauss=True)
    else:
        PSFparams = readPSFparams(inputFile=inputPSF,twoGauss=False)

    PSF = ifu.gauss_kernel(PSFparams.sig1[0], PSFparams.amp1[0], half_box=50)
    nstar = signal.convolve(model.nstar,PSF,mode='same')
    velocity = signal.convolve(model.velocity,PSF,mode='same')
    sigma = signal.convolve(model.sigma,PSF,mode='same')
    h3 = signal.convolve(model.h3,PSF,mode='same')
    h4 = signal.convolve(model.h4,PSF,mode='same')

    output = open(modeldir + 'aligned_OSIRIScoords_fits_full_smooth.dat', 'w')                    
        
    pickle.dump(nstar, output)
    pickle.dump(velocity, output)
    pickle.dump(sigma, output)
    pickle.dump(h3, output)
    pickle.dump(h4, output)
    output.close()

def modelResiduals(inputModel=modeldir+'nonaligned_OSIRIScoords_fits_full_smooth.dat',inputData=workdir+'ppxf.dat'):

    model = modelFitResults(inputFile=inputModel)
    data = PPXFresults(inputFile=inputData)
    cubeimg = pyfits.getdata(datadir + cuberoot + '_img.fits')
    cubeimg = cubeimg.transpose()

    # normalized flux residuals
    nstar = (cubeimg/cubeimg.max()) - (trimModel(model.nstar)/trimModel(model.nstar).max())
    velocity = (data.velocity+308.) - py.ma.masked_where(data.velocity == 0.,trimModel(model.velocity))
    sigma = data.sigma - py.ma.masked_where(data.sigma == 0.,trimModel(model.sigma))
    h3 = data.h3 - py.ma.masked_where(data.h3 == 0.,trimModel(model.h3))
    h4 = data.h4 - py.ma.masked_where(data.h4 == 0.,trimModel(model.h4))

    output = open(workdir+'model_residuals.dat','w')

    pickle.dump(nstar, output)
    pickle.dump(velocity, output)
    pickle.dump(sigma, output)
    pickle.dump(h3, output)
    pickle.dump(h4, output)
    output.close()    
        
class readPSFparams(object):
    def __init__(self, inputFile = workdir+'plots/osir_perf_m31_all_scalederr_cleanhdr_params.txt',twoGauss=False):
        self.inputFile = inputFile

        if twoGauss:
            input = pandas.read_csv(inputFile,delim_whitespace=True,header=None,names=['h1','sig1','h3','h4','h5','amp1','h6','h7','r1','r2','x1','x2','y1','y2','s1','s2'],usecols=['sig1','amp1'])
            self.sig1 = input.sig1
            self.amp1 = input.amp1
        else:
            input = pandas.read_csv(inputFile,delim_whitespace=True,header=None,names=['h1','sig1','h2','amp1','r1','r2','x1','x2','y1','y2','s1','s2'],usecols=['sig1','amp1'])
            self.sig1 = input.sig1
            self.amp1 = input.amp1
    
def modelConvertCoordinates(nonaligned=True,test=False):
    if nonaligned:
        inputFile = modeldir + 'nonaligned_model.dat'
    else:
        inputFile = modeldir + 'aligned_model.dat'

    model = pandas.read_csv(inputFile,delim_whitespace=True,header=None,names=['x','y','z','v_x','v_y','v_z'])

    t1 = time.time()
    ### angles, transformations from Peiris & Tremaine 2003 (table 2; eq. 4)
    ### angles originally measured in degrees
    if nonaligned:
        if test:
            thetaL = np.radians(0.)
            thetaI = np.radians(10.)
            thetaA = np.radians(45.)
        else:
            thetaL = np.radians(-42.8)
            thetaI = np.radians(54.1)
            thetaA = np.radians(-34.5)

    else:
        thetaL = np.radians(-52.3)
        thetaI = np.radians(77.5)
        thetaA = np.radians(-11.0)
        
    matL = np.matrix([[np.cos(thetaL),-np.sin(thetaL),0.],[np.sin(thetaL),np.cos(thetaL),0.],[0.,0.,1.]])
    matI = np.matrix([[1.,0.,0.],[0.,np.cos(thetaI),-np.sin(thetaI)],[0.,np.sin(thetaI),np.cos(thetaI)]])
    matA = np.matrix([[np.cos(thetaA),-np.sin(thetaA),0.],[np.sin(thetaA),np.cos(thetaA),0.],[0.,0.,1.]])

    # create subsets of the inputs in chunks of 5000 
    isets = np.arange(math.ceil(len(model.x)/5000.))
    istart = np.arange(0,len(model.x),5000)
    istop = np.arange(5000,len(model.x),5000)
    istop = np.concatenate((istop,[len(model.x)+1]))
    startstop = np.concatenate(([istart],[istop]),axis=0)
    startstop = startstop.transpose()

    xyz = np.column_stack((model.x,model.y,model.z))
    vxyz = np.column_stack((model.v_x,model.v_y,model.v_z))
    
    #job_server = pp.Server()
    #print "Starting pp with", job_server.get_ncpus(), "workers"
    #jobs = [(i,job_server.submit(run_once_convert, args=(xyz[i[0]:i[1]],vxyz[i[0]:i[1]],matL,matI,matA,i[0],contmpdir), depfuncs=(), modules=('numpy as np',))) for i in startstop]
    #test = run_once_convert(xyz[startstop[0][0]:startstop[0][1]],vxyz[startstop[0][0]:startstop[0][1]],matL,matI,matA,startstop[0][0],contmpdir)
    #pdb.set_trace()

    #job_server.wait()

    # job_server.submit keeps failing - doing a for loop for now
    for i in startstop:
        job = run_once_convert(xyz[i[0]:i[1]],vxyz[i[0]:i[1]],matL,matI,matA,i[0],contmpdir)

    bigX = np.zeros(len(model.x))
    bigY = np.zeros(len(model.x))
    bigZ = np.zeros(len(model.x))
    bigVX = np.zeros(len(model.x))
    bigVY = np.zeros(len(model.x))
    bigVZ = np.zeros(len(model.x))
    
    for i in range(len(istart)):
        print "Setting output of ", i
        set = istart[i]

        tmpInput = contmpdir + 'convert_'+str(set)+'.dat'
    
        tmpmodel = pandas.read_csv(tmpInput,delim_whitespace=True,header=None,names=['x','y','z','v_x','v_y','v_z'])

        bigX[istart[i]:istop[i]] = tmpmodel.x
        bigY[istart[i]:istop[i]] = tmpmodel.y
        bigZ[istart[i]:istop[i]] = tmpmodel.z
        bigVX[istart[i]:istop[i]] = tmpmodel.v_x
        bigVY[istart[i]:istop[i]] = tmpmodel.v_y
        bigVZ[istart[i]:istop[i]] = tmpmodel.v_z
    
    if nonaligned:
        if test:
            outputFile = modeldir + 'nonaligned_model_skycoords_testrotate.dat'
        else:
            outputFile = modeldir + 'nonaligned_model_skycoords.dat'
        
    else:
        outputFile = modeldir + 'aligned_model_skycoords.dat'
        
    np.savetxt(outputFile,np.c_[bigX,bigY,bigZ,bigVX,bigVY,bigVZ],fmt='%8.6f',delimiter='\t')

    print "Time elapsed: ", time.time() - t1, "s"
    
def run_once_convert(xyz,vxyz,matL,matI,matA,set,contmpdir):
    
    for i in range(len(xyz)):
        tmp = matL.dot(matI).dot(matA).dot(xyz[i])
        tmpv = matL.dot(matI).dot(matA).dot(vxyz[i])
        if i==0:
            bigXYZ = tmp
            bigV_XYZ = tmpv
        else:
            bigXYZ = np.squeeze(bigXYZ)
            bigV_XYZ = np.squeeze(bigV_XYZ)
            bigXYZ = np.append(bigXYZ,tmp,axis=0)
            bigV_XYZ = np.append(bigV_XYZ,tmpv,axis=0)

    tmpx, tmpy, tmpz = np.hsplit(bigXYZ, 3)
    tmpvx, tmpvy, tmpvz = np.hsplit(bigV_XYZ, 3)

    np.savetxt(contmpdir + 'convert_'+str(set)+'.dat',np.c_[tmpx,tmpy,tmpz,tmpvx,tmpvy,tmpvz],fmt='%8.6f',delimiter='\t')

def modelOSIRISrotation(inputFile=None,nonaligned=True):
    # transforming from sky coordinates (+Y is north, +X is west) to OSIRIS coordinates,
    ### taking into account the PA

    if inputFile:
        model = modelResults(inputFile)
    else:
        if nonaligned:
            model = modelResults(nonaligned=True,skycoords=True)
        else:
            model = modelResults(nonaligned=False,skycoords=True)

    t1 = time.time()

    # counterclockwise rotation (from model skycoords to OSIRIS coords) is positive,
    # by definition of the rotation matrix
    cpa = -34.
    
    thetaCPA = np.radians(cpa)

    rotMat = np.matrix([[np.cos(thetaCPA),-np.sin(thetaCPA)],[np.sin(thetaCPA),np.cos(thetaCPA)]])

    # create subsets of the inputs in chunks of 5000 
    isets = np.arange(math.ceil(len(model.x)/5000.))
    istart = np.arange(0,len(model.x),5000)
    istop = np.arange(5000,len(model.x),5000)
    istop = np.concatenate((istop,[len(model.x)+1]))
    startstop = np.concatenate(([istart],[istop]),axis=0)
    startstop = startstop.transpose()

    # rotating in the plane of the sky, so only transforming x and y (and v_x and v_y)
    xy = np.column_stack((model.x,model.y))
    vxy = np.column_stack((model.vx,model.vy))

    #job_server = pp.Server()
    #print "Starting pp with", job_server.get_ncpus(), "workers"
    #jobs = [(i, job_server.submit(run_once_osiris_convert, (xy[i[0]:i[1]],vxy[i[0]:i[1]],rotMat,i[0],contmpdir,i), (), ('numpy as np'))) for i in startstop]

    #job_server.wait()

    for i in startstop:
        job = run_once_osiris_convert(xy[i[0]:i[1]],vxy[i[0]:i[1]],rotMat,i[0],contmpdir)
    
    
    bigX = np.zeros(len(model.x))
    bigY = np.zeros(len(model.x))
    bigZ = model.z
    bigVX = np.zeros(len(model.x))
    bigVY = np.zeros(len(model.x))
    bigVZ = model.vz

    for i in range(len(istart)):
        print "Setting output of ", i
        set = istart[i]

        tmpInput = contmpdir + 'convertOSIRIS_'+str(set)+'.dat'
    
        tmpmodel = pandas.read_csv(tmpInput,delim_whitespace=True,header=None,names=['x','y','vx','vy'])

        bigX[istart[i]:istop[i]] = tmpmodel.x
        bigY[istart[i]:istop[i]] = tmpmodel.y
        bigVX[istart[i]:istop[i]] = tmpmodel.vx
        bigVY[istart[i]:istop[i]] = tmpmodel.vy
      
    if nonaligned:
        outputFile = modeldir + 'nonaligned_model_OSIRIScoords.dat'
    else:
        outputFile = modeldir + 'aligned_model_OSIRIScoords.dat'
        
    np.savetxt(outputFile,np.c_[bigX,bigY,bigZ,bigVX,bigVY,bigVZ],fmt='%8.6f',delimiter='\t')

    print "Time elapsed: ", time.time() - t1, "s"

def run_once_osiris_convert(xy,vxy,rotMat,set,contmpdir):
    
    for i in range(len(xy)):
        tmp = rotMat.dot(xy[i])
        tmpv = rotMat.dot(vxy[i])
        if i==0:
            outXY = tmp
            outV_XY = tmpv
        else:
            outXY = np.squeeze(outXY)
            outV_XY = np.squeeze(outV_XY)
            outXY = np.append(outXY,tmp,axis=0)
            outV_XY = np.append(outV_XY,tmpv,axis=0)

    tmpx, tmpy = np.hsplit(outXY, 2)
    tmpvx, tmpvy = np.hsplit(outV_XY, 2)

    outputFile = contmpdir + 'convertOSIRIS_'+str(set)+'.dat'    
    np.savetxt(outputFile,np.c_[tmpx,tmpy,tmpvx,tmpvy],fmt='%8.6f',delimiter='\t')
    
class modelResults(object):
    def __init__(self,inputFile=None,nonaligned=True,skycoords=True,OSIRIS=False):

        if inputFile:
            inputFile = inputFile
        else:
            if nonaligned:
                if skycoords:
                    inputFile = modeldir + 'nonaligned_model_skycoords.dat'
                    if OSIRIS:
                        inputFile = modeldir + 'nonaligned_model_OSIRIScoords.dat'
                else:
                    inputFile = modeldir + 'nonaligned_model.dat'
            else:
                if skycoords:
                    inputFile = modeldir + 'aligned_model_skycoords.dat'
                    if OSIRIS:
                        inputFile = modeldir + 'aligned_model_OSIRIScoords.dat'
                else:
                    inputFile = modeldir + 'aligned_model.dat'

        model = pandas.read_csv(inputFile,delim_whitespace=True,header=None,names=['x','y','z','v_x','v_y','v_z'])
    
        self.x = model.x
        self.y = model.y
        self.z = model.z
        self.vx = model.v_x
        self.vy = model.v_y
        self.vz = model.v_z
    
def load_templates(velScale, resolution=3241, IDL=True):
    # IDL and Python versions of pPXF require different formats for the input templates
    templateDir = '/Users/kel/Documents/Library/IDL/ppxf/templates/GNIRS/library_v15_gnirs_combined/'

    files = glob.glob(templateDir + '*.fits')
    print 'Using %d templates' % len(files)

    templates = None
    for ff in range(len(files)):
        newWave, newSpec = load_spectrum_gnirs(files[ff], velScale, resolution)

        if templates == None:
            if IDL:
                templates = np.zeros(( len(files),len(newSpec)), dtype=float)
            else: 
                templates = np.zeros(( len(newSpec),len(files)), dtype=float)

        if IDL:
            templates[ff,:] = newSpec
        else:
            templates[:,ff] = newSpec

    return (newWave, templates)

def load_templates_old(velScale, IDL=False):
    templateDir = '/Users/kel/Documents/Library/IDL/ppxf/templates/atlasSpectra/medresIR/K_band'

    files = glob.glob(templateDir + '/spec.*')
    specfiles = ['HR6406', 'KY_Cyg', 'HR8316', 'HR8726', 'HR8089',
                 'HR8465', 'HR2473', 'HR969', 'HR8313', 'HR8626', 
                 'HR8232', 'HR1017', 'HR7387', 'HR382', 'HR7924',
                 'HR1713', 'HR6714', 'HR7977', 'HR1903', 'HR8469',
                 'RX_Boo', 'HR7886', 'HR8621', 'HR7635',
                 'HR3275', 'HR7806', 'HR6703', 'HR3212', 'HR3323',
                 'HR4255', 'HR4883', 'HR8905', 'HR7495', 'HR5017',
                 'HR4031', 'HR1412', 'HR5291', 'HR1791', 'HR2294',
                 'HR6165', 'HR3982', 'HR4295', 'HR7557', 'HR4931',
                 'HR6927']
    files = []
    for ss in range(len(specfiles)):
        files.append( templateDir + '/spec.' + specfiles[ss] )

    print 'Using %d templates' % len(files)

    templates = None
    for ff in range(len(files)):
        newWave, newSpec = load_spectrum_medresIR(files[ff], velScale)

        if templates == None:
            if IDL:
                templates = np.zeros((len(files), len(newSpec)), dtype=float)
            else:
                templates = np.zeros((len(newSpec), len(files)), dtype=float)

        if IDL:
            templates[ff,:] = newSpec
        else:
            templates[:,ff] = newSpec

    return (newWave, templates)

def load_spectrum_gnirs(file, velScale, resolution):
    """
    Load up a spectrum from the Gemini GNIRS library.
    """
    spec, hdr = pyfits.getdata(file, header=True)

    pixScale = hdr['CD1_1'] * 1.0e-4    # microns

    wavelength = np.arange(len(spec), dtype=float)
    wavelength -= (hdr['CRPIX1']-1.0)   # get into pixels relative to the reference pix
    wavelength *= hdr['CD1_1']          # convert to the proper wavelength scale (Ang)
    wavelength += hdr['CRVAL1']         # shift to the wavelength zeropoint

    # Convert from Angstroms to microns
    wavelength *= 1.0e-4

    deltaWave = 2.21344 / resolution         # microns
    resInPixels = deltaWave / pixScale       # pixels
    sigmaInPixels = resInPixels / 2.355
    psfBins = np.arange(-4*math.ceil(resInPixels), 4*math.ceil(resInPixels))
    psf = py.normpdf(psfBins, 0, sigmaInPixels)
    specLowRes = np.convolve(spec, psf, mode='same')

    # Rebin into equals logarithmic steps in wavelength
    logWave, specNew, vel = log_rebin2(wavelength, specLowRes,
                                       inMicrons=True, velScale=velScale)

    return logWave, specNew


def load_spectrum_medresIR(file, velScale):
    """
    Load spectrum from the R~3000 Wallace & Hinkle library.
    Return:
        logWave -- log(wavelength in microns)
        specNew -- the spectral flux
    """
    f_spec = open(file, 'r')
    
    lineCount = 0
    waveNumber = []
    flux = []
    
    for line in f_spec:
        lineCount += 1
        if lineCount <= 6:
            continue
        
        fields = line.split()
        waveNumber.append( float(fields[0]) )
        flux.append( float(fields[1]) )
        
    waveNumber = np.array(waveNumber)
    flux = np.array(flux)
    
    print file
    logWave, specNew, vel = log_rebin2(waveNumber, flux, 
                                       inMicrons=False, velScale=velScale)

    return logWave, specNew

def log_rebin_cube(wavelength, cube):
    """
    Read in data cube and re-bin the spectrum at each spaxel
    to be sampled uniformly in log(wavelength).
    """
    newCube = np.zeros(cube.shape, dtype=float)
    for ii in range(cube.shape[0]):
        for jj in range(cube.shape[1]):
            logWave, spec, vel = log_rebin2(wavelength, cube[ii,jj,:])
            newCube[ii, jj, :] = spec

    return logWave, newCube, vel

def log_rebin2(xaxis, spectrum, inMicrons=True, velScale=None, plot=False):
    
    if inMicrons:
        # xaxis is currently expressed in wavelength with units of microns
        # We will assume that it is linear in wavelength
        wavelength = xaxis
    else:
        # Otherwise we assume it is wavenumbers with units of cm^-1
        # We will assume that it is linear in wavenumber.
        # Convert into wavelength in microns
        frequency = xaxis[::-1] # Reverse
        spectrum = spectrum[::-1]
        wavelength = 1.0e4 / frequency

    pixCnt = len(spectrum)

    tck = scipy.interpolate.splrep(wavelength, spectrum, s=0)
    
    # Define new wavelength scale with even steps in log(wavelength)
    if velScale == None:
        logDw = (np.log(wavelength[-1]) - np.log(wavelength[0])) 
        logDw /= (pixCnt - 1)
        velScale = logDw * cc.c
    else:
        logDw = velScale/cc.c
        pixCnt = (np.log(wavelength[-1]) - np.log(wavelength[0])) 
        pixCnt /= logDw

    logWave = np.log(wavelength[0]) + np.arange(pixCnt)*logDw

    waveNew = np.exp(logWave)
    specNew = scipy.interpolate.splev(waveNew, tck)

    if plot:
        py.clf()
        py.plot(wavelength, spectrum, 'r-')
        py.plot(waveNew, specNew, 'b-')
        py.show()

    return (logWave, specNew, velScale)


def log_rebin(wavelength, spectrum):

    cnt = len(wavelength)
    print 'cnt = ', cnt

    ###
    tck = scipy.interpolate.splrep(wavelength, spectrum, s=0)
    
    # Define new wavelength scale with even steps in log(wavelength)
    logDw = (np.log(wavelength[-1]) - np.log(wavelength[0])) 
    logDw /= (cnt - 1)

    logWave = np.log(wavelength[0]) + np.arange(cnt)*logDw

    waveNew = np.exp(logWave)
    specNew = scipy.interpolate.splev(waveNew, tck)
    print 'waveNew: ', waveNew
    print 'logWave: ', logWave
    print 'specNew: ', specNew
    print 'total flux = ', specNew.sum()
    py.clf()
    py.plot(waveNew, specNew, 'b-')
    ###

    waveRange = wavelength[[0,-1]]
    print 'waveRange = ', waveRange

    # Get the wavelength bin size (or scale)
    dw = wavelength[1] - wavelength[0]
    print 'dw = ', wavelength[0] - wavelength[1]
    print 'dw = ', wavelength[1] - wavelength[2]
    print 'dw = ', wavelength[2] - wavelength[3]

    # Wavelength in units of delta-lambda (dw)
    limits = (waveRange / dw) + np.array([-0.5, 0.5])
    print 'limits = ', limits

    dwWave = wavelength / dw
    borders = np.append(dwWave - 0.5, dwWave[-1] + 0.5)
    print 'borders = ', borders[0:5], borders[-1]

    # Get the natural log
    logLimits = np.log(limits)
    print 'logLimits = ', logLimits

    velScale = cc.c * (logLimits[1] - logLimits[0]) / cnt
    print 'velScale = ', velScale
    
    logDw = (logLimits[1]-logLimits[0]) / (cnt)
    newBorders = np.exp(logLimits[0] + np.arange(cnt+1) * logDw)
    print 'newBorders.shape = ', newBorders.shape
    print 'newBorders = ', newBorders[0:5], newBorders[-1]
    
    newBordersTmp = newBorders[0:-1]

    k = np.floor(newBorders - limits[0])
    k = np.where(k > 0, k, 0)
    k = np.where(k < cnt-1, k, cnt-1)

    print 'cnt-1 = ', cnt-1
    print 'k[0:15] = ', k[0:15] 
    print 'k[-5:] = ', k[-5:]
                        
    logWave = np.log(np.sqrt(newBorders[1:] * newBordersTmp) * dw)
    print 'logWave = ', logWave[0:5], logWave[-1]

    specNew = np.zeros(spectrum.shape, dtype=float)
    qualityNew = np.zeros(spectrum.shape, dtype=float)
    
    for ii in range(cnt):
        a = newBorders[ii] - borders[k[ii]]
        b = borders[k[ii+1]+1] - newBorders[ii+1]

        if len(spectrum.shape) == 3: 
            specNew[:,:,ii] = spectrum[:,:,k[ii]:k[ii+1]+1].sum(axis=2)
            specNew[:,:,ii] -= a*spectrum[:,:,k[ii]] 
            specNew[:,:,ii] -= b*spectrum[:,:,k[ii+1]]
        else:
            specNew[ii] = spectrum[k[ii]:k[ii+1]+1].sum()
            specNew[ii] -= a*spectrum[k[ii]] 
            specNew[ii] -= b*spectrum[k[ii+1]]

    py.plot(np.exp(logWave), specNew, 'g-')
    py.plot(wavelength, spectrum, 'r-')
    py.show()
    print 'total flux = ', specNew.sum()
    return (logWave, specNew)

def velErr(inputResults=workdir+'/ppxf.dat'):

    ### smooths input velocity map and subtracts it from itself to get residuals
    ### as a measure of the error on the velocity
    p = PPXFresults(inputResults)

    tmp = p.velocity
    # set the zeroes around the edges to NaN so convolve handles them properly
    bad = np.where(tmp == 0)    
    tmp[bad] = np.nan

    # boxcar kernel, width=3
    kern = astropy.convolution.Box2DKernel(3)
    v_smooth = astropy.convolution.convolve(tmp+308.,kern,boundary='extend')
    v_smooth[bad] = 0.

    v_err = p.velocity - v_smooth
    good = np.where(np.isfinite(v_err))

    result = np.std(v_err[good])

    print 'Standard deviation of the residuals is %s' % result

    return result
    

def play(frequency, spectrum):

    # Reverse.
    # Remember that these are the center values of each spectral channel.
    frequency = frequency[::-1]
    spectrum = spectrum[::-1]
    wavelength = 1.0e4 / frequency

    cnt = len(wavelength)

    # Get the borders
    # assume uniform sampling in frequency
    dFreq = frequency[1] - frequency[0]

    freq1 = frequency[0] - dFreq
    freq2 = frequency[-1] + dFreq

    # Extra center values
    freqTmp = np.concatenate(([freq1], frequency, [freq2]))
    waveTmp = 1.0e4 / freqTmp

    print 'frequency.shape = ', frequency.shape
    print 'freqTmp.shape = ', freqTmp.shape
    print 'wavelength.shape = ', wavelength.shape
    print 'waveTmp.shape = ', waveTmp.shape

    # Now calculate delta-Lambda for each central value, by taking
    # finding delta Lambda from the pixel to the left and right and
    # averaging the deltaLambda.
    dWave1 = waveTmp[1:-1] - waveTmp[0:-2] # length same as original
    dWave2 = waveTmp[2:] - waveTmp[1:-1]
    dWave = (dWave1 + dWave2) / 2.0

    print 'dWave1.shape = ', dWave1.shape
    print 'dWave2.shape = ', dWave2.shape
    print 'dWave.shape = ', dWave.shape
    print ''
    print 'dWave = ', dWave
    
    # Construct the current borders
    borders = np.append(wavelength - (dWave/2.0), 
                        wavelength[-1] + (dWave[-1]/2.0))

    # We will be working with log(wave) so lets convert
    logWave = np.log(wavelength)
    logBorders = np.log(borders)
                             
    # Calculate the new dlogWave by assuming same range and same number
    # of pixels as input spectrum.
    dlogWave = (logBorders[-1] - logBorders[0]) / cnt

    # Calculate new borders and wavelengths
    newLogBorders = logBorders[0] + dlogWave * np.arange(cnt+1)
    newLogWave = newLogBorders[0:-1] + (dlogWave/2.0)
    newBorders = np.exp(newLogBorders)
    newWave = np.exp(newLogWave)

    print 'wavelength = ', wavelength
    print ''
    print 'borders = ', borders
    print 'newBorders = ', newBorders
    print ''
    print 'logWave = ', logWave
    print 'newLogWave = ', newLogWave
    print ''
    print 'logBorders = ', logBorders
    print 'newLogBorders = ', newLogBorders

    # Calculate the new flux
    specNew = np.zeros(spectrum.shape, dtype=float)
    
    residual = 0.0
    for ii in range(cnt):
        idxLo = ((np.where(borders <= newBorders[ii]))[0]).max()
        idxHi = ((np.where(borders >= newBorders[ii+1]))[0]).min()

        specNew[ii] = spectrum[idxLo:idxHi].sum()
        
        binSizeLo = borders[idxLo+1] - borders[idxLo]
        binSizeHi = borders[idxHi] - borders[idxHi-1]

        a = (newBorders[ii] - borders[idxLo]) / binSizeLo
        b = (borders[idxHi] - newBorders[ii+1]) / binSizeHi
        
        print 'ii = ', ii, ' idxLo = ', idxLo, ' idxHi = ', idxHi
        print '    a = ', a, ' b = ', b
        print '    specNew = ', specNew[ii]
        specNew[ii] -= a * spectrum[idxLo]
        print '    specNew = ', specNew[ii]
        specNew[ii] -= b * spectrum[idxHi-1]
        print '    specNew = ', specNew[ii]
    
    py.clf()
    py.plot(logWave, spectrum, 'b-')
    py.plot(newLogWave, specNew, 'g-')
    py.show()

    return newLogWave, specNew


def test():
    cubefits = pyfits.open(datadir + cuberoot + '.fits')
    
    cube = cubefits[0].data
    hdr = cubefits[0].header
    errors = cubefits[1].data
    quality = cubefits[2].data

    # Get the wavelength solution so we can specify range:
    w0 = hdr['CRVAL1']
    dw = hdr['CDELT1']
    wavelength = w0 + dw * np.arange(cube.shape[2], dtype=float)
    wavelength /= 1000.0   # Convert to microns
    
    # Lets temporary run a single spectrum
    spectrum = cube[10,30,:]

#     print 'IDL version:'
#     idl.specNew1 = np.zeros(cube.shape[2], dtype=float)
#     idl.logWave1 = np.zeros(len(wavelength), dtype=float)
#     idl.waveRange = [wavelength[0], wavelength[-1]]
#     idl.spec = spectrum
#     idl('log_rebin, waveRange, spec, specNew1, logWave1')

#     print 'PYTHON VERSION:'
#     (logWave2, specNew2) = log_rebin(wavelength, cube[10,30,:])
#     print 'logWave2 = ', logWave2
#     print 'specNew2 = ', specNew2
#     (logWave3, specNew3) = log_rebin(wavelength, cube)

#     print idl.logWave1
#     print logWave2
#     print idl.specNew1
#     print specNew2
#     print specNew3[10,30,:]

#     logWave = idl.logWave1
#     logSpec = idl.specNew1

    # Load templates
    #templates = load_templates()
    print 'Templates:'
    load_spectrum('/u/jlu/work/atlasSpectra/medresIR/K_band/spec.HR21')

    # Test #2
    logWave3, specNew3 = log_rebin2(wavelength, cube[10,30,:])


