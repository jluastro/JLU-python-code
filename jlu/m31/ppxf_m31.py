import numpy as np
import pylab as py
# import pidly
from astropy.io import fits as pyfits
from astropy.table import Table
import pickle
import time
import math, glob
import scipy
import scipy.interpolate
from scipy.interpolate import interp1d
from scipy.optimize import curve_fit#, OptimizeWarning
from scipy import signal,ndimage
from gcwork import objects
import pdb
# import ppxf
import pp
import itertools
import pandas
import astropy
import os
import warnings
import matplotlib as mpl
import colormaps as cmaps
from matplotlib.colors import LogNorm
from jlu.m31 import ifu
import ppxf


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
cuberoot = 'm31_all_scalederr'
#cuberoot = 'm31_all_scalederr_cleanhdr'
#cuberoot = 'm31_all_scalederr_cleanhdr_bulgesub_vorcube_20160825'
#cuberoot = 'm31_all_scalederr_cleanhdr_bulgesub_20160825'
#cuberoot = 'm31_mosaic_telshift'

cc = objects.Constants()

# Modified black hole position from 2009 sep alignment 
# analysis between NIRC2 and HST. This is the position
# in the osiris cubes.
#bhpos = np.array([8.7, 39.1]) * 0.05 # python coords, not ds9
#bhpos = np.array([22.5, 37.5]) * 0.05 # guessed offset for new M31 mosaic
# position in new mosaic, 2016/05
#bhpos = np.array([20.,41.]) * 0.05
# position in new mosaic, using new Lauer F435 frame
#bhpos = np.array([19.1,40.7]) * 0.05
# for plotting horizontally
#bhpos_hor = np.array([(83-40.7), 19.1]) * 0.05
#bhpos_pix = np.array([(83-40.7),19.1])
# new shifts, numclip 5, numclip 10
#bhpos = np.array([19.2,39.6]) * 0.05
#bhpos_hor = np.array([(83-39.6), 19.2]) * 0.05
#bhpos_pix = np.array([(83-39.6), 19.2])
# numclip 15
#bhpos = np.array([17.1,39.5]) * 0.05
#bhpos_hor = np.array([(83-39.5), 17.1]) * 0.05
#bhpos_pix = np.array([(83-39.5), 17.1])
# max PSF = 2
#bhpos = np.array([17.0,39.5]) * 0.05
#bhpos_hor = np.array([(82-39.5), 17.0]) * 0.05
#bhpos_pix = np.array([(82-39.5), 17.0])
# tel shift
#bhpos = np.array([23.,  44.9]) * 0.05
#bhpos_hor = np.array([(87-44.9), 23.]) * 0.05
#bhpos_pix = np.array([(87-44.9), 23.])
# 081021
#bhpos = np.array([15.0,33.0])*0.05
#bhpos_hor = np.array([(76-33.0),15.0])*0.05
#bhpos_pix = np.array([(76-33.0),15.0])
#bhpos_pix = np.array([(76-37.0),9.0]) # by eye, because the analysis.registerOSIRIStoNIRC2 clearly wasn't right
# 100815
#bhpos = np.array([16.3,34.5])*0.05
#bhpos_hor = np.array([(76-34.5),16.3])*0.05
#bhpos_pix = np.array([(76-34.5),15.6])
# 100828
#bhpos = np.array([24.,42.])*0.05
#bhpos_hor = np.array([(79-42.),24.])*0.05
#bhpos_pix = np.array([(79-42.),24.])
# 100829
#bhpos = np.array([23.5,35.2])*0.05
#bhpos_hor = np.array([(78-35.2),23.5])*0.05
#bhpos_pix = np.array([(78-35.2),23.5])
# comb shift
#bhpos = np.array([24.2,38.3])*0.05
#bhpos_hor = np.array([(83-38.3),24.2])*0.05
#bhpos_pix = np.array([(83-38.3),24.2])
# comb 2 shift
#bhpos = np.array([23.9,37.6])*0.05
#bhpos_hor = np.array([(82-37.6),23.9])*0.05
#bhpos_pix = np.array([(82-37.6),23.9])
# comb 3 shift
#bhpos = np.array([20.1,37.5])*0.05
#bhpos_hor = np.array([(82-37.5),20.1])*0.05
#bhpos_pix = np.array([(82-37.5),20.1])
# NIRC2 DTOTOFF
#bhpos = np.array([40.1,18.0])*0.05
#bhpos_hor = np.array([(85-40.1),18.])*0.05
#bhpos_pix = np.array([(85-40.1),18.])
# CC mos DTOTOFF
#bhpos = np.array([39.4,15.7])*0.05
#bhpos_hor = np.array([(84-39.4),15.7])*0.05
#bhpos_pix = np.array([(84-39.4),15.7])
# NIRC2 DTOTOFF 2
#bhpos = np.array([38.7,20.2])*0.05
#bhpos_hor = np.array([(83-38.7),20.2])*0.05
#bhpos_pix = np.array([(83-38.7),20.2])
# CC mos DTOTOFF 2
#bhpos = np.array([38.4,19.1])*0.05
#bhpos_hor = np.array([(82-38.4),19.1])*0.05
#bhpos_pix = np.array([(82-38.4),19.1])
# NIRC2 DTOTOFF 2 - no 100828, 100829
#bhpos = np.array([38.8,15.2])*0.05
#bhpos_hor = np.array([(82-38.8),15.2])*0.05
#bhpos_pix = np.array([(82-38.8),15.2])
# CC mos DTOTOFF 2 - no 100828
#bhpos = np.array([38.3,18.8])*0.05
#bhpos_hor = np.array([(81-38.3),18.8])*0.05
#bhpos_pix = np.array([(81-38.3),18.8])
# CC mos DTOTOFF 2 - no 100828, 100829
#bhpos = np.array([38.4,15.2])*0.05
#bhpos_hor = np.array([(81-38.4),15.2])*0.05
#bhpos_pix = np.array([(81-38.4),15.2])
# NIRC2 DTOTOFF 2 - no 100828
bhpos = np.array([38.7,20.0])*0.05
bhpos_hor = np.array([(82-38.7),20.0])*0.05
bhpos_pix = np.array([(82-38.7),20.0])
# NIRC2 DTOTOFF 2 - no 100828, x+1,y-1
#bhpos = np.array([37.7,19.0])*0.05
#bhpos_hor = np.array([(82-37.7),19.0])*0.05
#bhpos_pix = np.array([(82-37.7),19.0])
# NIRC2 DTOTOFF 2 - no 100828, x+1
#bhpos = np.array([37.7,20.0])*0.05
#bhpos_hor = np.array([(82-37.7),20.0])*0.05
#bhpos_pix = np.array([(82-37.7),20.0])
# NIRC2 DTOTOFF 2 - no 100828, y-1
#bhpos = np.array([38.7,19.0])*0.05
#bhpos_hor = np.array([(82-38.7),19.0])*0.05
#bhpos_pix = np.array([(82-38.7),19.0])

vsys = 340.

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
    
def run_py(inputFile=None,verbose=True,newTemplates=True,blue=False,red=False,mid=False,twocomp=False,selectTemp=None,flagWave=False,moments=4):
    """
    Run the PPXF analysis the M31 OSIRIS data cube, using the Python implementation of pPXF.
    """
    # Read in the data cube.
    if inputFile:
        cubefits = pyfits.open(inputFile)
    else:
        inputFile = datadir + cuberoot + '.fits'
        cubefits = pyfits.open(inputFile)
    
    cube = cubefits[0].data
    hdr = cubefits[0].header
    errors = cubefits[1].data
    quality = cubefits[2].data
    #nframes = cubefits[3].data

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
        logWaveTemps, templates = load_templates(velScale,IDL=False,selectTemp=selectTemp)
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
    elif mid:
        waveCut = 2.220
    else: 
        waveCut = 2.185
    print 'blue wavelength cutoff = %.2f microns' % waveCut
    if blue:
        waveCutRed = 2.285
        idx = np.where((np.exp(logWaveCube) > waveCut) & (np.exp(logWaveCube) < waveCutRed))[0]
    elif mid:
        waveCutRed = 2.312
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
    if flagWave:
        # set good pixels - start with the whole range set to unflagged
        flag = np.exp(logWaveCube) < 0
        # flag the red end - lots of telluric lines
        #flag = np.exp(logWaveCube) > 2.358
        # cut between 2nd and 3rd CO bandhead
        flag = np.exp(logWaveCube) > 2.329
        # cut between 1st and 2nd CO bandhead
        #flag = np.exp(logWaveCube) > 2.312
        # then mask out the badly corrected telluric lines blueward of the CO bandheads (range is inclusive)
        #tellLinesLo = [2.2865, 2.31375]
        #tellLinesLo = [2.286, 2.313, 2.329, 2.2975]
        #tellLinesHi = [2.2905, 2.319, 2.344, 2.3015]
        # just masking the blue bumps by the CO band heads
        #tellLinesLo = [2.285, 2.312]
        #tellLinesHi = [2.2895, 2.318]
        # flagwave 1
        #tellLinesLo = [2.288,2.314]
        #tellLinesHi = [2.2915,2.325]
        # flagwave 2
        tellLinesLo = [2.2913]
        tellLinesHi = [2.293]
    
        for i in np.arange(len(tellLinesLo)):
            # set flag to 1 within the line to be masked
            flag |= (np.exp(logWaveCube) >= tellLinesLo[i]) & (np.exp(logWaveCube) <= tellLinesHi[i])
            #pdb.set_trace()
        # return the unflagged channels
        goodPixels = np.where(flag == 0)[0]
    else:
        # set good pixels - start with the whole range set to unflagged
        flag = np.exp(logWaveCube) < 0
        # flag the red end no matter if masking is turned off - lots of telluric lines
        flag = np.exp(logWaveCube) > 2.329    
        #flag = np.exp(logWaveCube) > 2.312
        goodPixels = np.where(flag == 0)[0]
        # before, set all pixels to good pixels
        #goodPixels = np.arange(len(logWaveCube))
    #pdb.set_trace()
    
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
    jobs = [(i, job_server.submit(run_once, (newCube[i[0],i[1],:],newErrors[i[0],i[1],:],templates,velScale,start,goodPixels,dv,twocomp,moments,i), (), ('numpy as np','time','ppxf'))) for i in allxxyy]
    #test=np.array([(0,0),(0,5),(10,30)])
    #test=np.array([(10,30)])
    #jobs = [(i, job_server.submit(run_once, (newCube[i[0],i[1],:],newErrors[i[0],i[1],:],templates,velScale,start,goodPixels,dv,twocomp,i), (), ('numpy as np','time','ppxf','pdb'))) for i in test]
    #test = run_once(newCube[20,40,:],newErrors[20,40,:],templates,velScale,start,goodPixels,dv,twocomp,[20,40])
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
                if moments == 4:
                    h3[xx, yy] = solution[2]
                    h4[xx, yy] = solution[3]
            #h5[xx, yy] = solution[4]
            #h6[xx, yy] = solution[5]
            chi2red[xx, yy] = job.result.chi2

            pweights[xx, yy, :] = job.result.polyweights
            tweights[xx, yy, :] = job.result.weights
            bestfit[xx, yy, :] = job.result.bestfit
            
        
        
    #pdb.set_trace()

    outfile = os.path.dirname(inputFile) + '/ppxf.dat'
    output = open(outfile, 'w')
    pickle.dump(velocity, output)
    pickle.dump(sigma, output)
    if moments == 4:
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

def run_once(newCube,errors,templates,velScale,start,goodPixels,vsyst,twocomp,moments,allxxyy,verbose=False):
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
    #minFlux = tmp.mean() - (3.0 * tmp.std())
    #tmp2 = tmp - minFlux
            
    #galaxy = tmp2
    galaxy = tmp
    error = tmperr

    if twocomp:
        templates2 = np.concatenate((templates,templates),axis=1)
        outppxf = ppxf.ppxf(templates2, galaxy, error, velScale, [start,start], goodpixels=goodPixels, plot=False, moments=[2,2], mdegree=4, vsyst=vsyst,component=[0]*23+[1]*23)
    else:
        #outppxf = ppxf.ppxf(templates, galaxy, error, velScale, start, goodpixels=goodPixels, plot=False, moments=4, mdegree=4, vsyst=vsyst)
        #start = [0.,236.]
        outppxf = ppxf.ppxf(templates, galaxy, error, velScale, start, goodpixels=goodPixels, plot=False, moments=moments, vsyst=vsyst)

    #pdb.set_trace()
    return outppxf

def runErrorMC(inCube=None,newTemplates=True,jackknife=False,test=False,flagWave=False,inputVoronoiFile=None,rerun=True):
    """
    Run the PPXF analysis the M31 OSIRIS data cube.
    """
    # Read in the data cube.
    if inCube is None:
        cubefits = pyfits.open(datadir + cuberoot + '.fits')
    else:
        cubefits = pyfits.open(inCube)
    
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
    #goodPixels = np.arange(len(logWaveCube))
    if flagWave:
        # set good pixels - start with the whole range set to unflagged
        flag = np.exp(logWaveCube) < 0
        # flag the red end - lots of telluric lines
        #flag = np.exp(logWaveCube) > 2.358
        flag = np.exp(logWaveCube) > 2.329
        # then mask out the badly corrected telluric lines blueward of the CO bandheads (range is inclusive)
        #tellLinesLo = [2.2865, 2.31375]
        #tellLinesLo = [2.286, 2.313, 2.329, 2.2975]
        #tellLinesHi = [2.2905, 2.319, 2.344, 2.3015]
        #tellLinesLo = [2.285, 2.312]
        #tellLinesHi = [2.2895, 2.318]
        tellLinesLo = [2.288,2.314]
        tellLinesHi = [2.2915,2.325]
    
        for i in np.arange(len(tellLinesLo)):
            # set flag to 1 within the line to be masked
            flag |= (np.exp(logWaveCube) >= tellLinesLo[i]) & (np.exp(logWaveCube) <= tellLinesHi[i])
            #pdb.set_trace()
        # return the unflagged channels
        goodPixels = np.where(flag == 0)[0]
    else:
        # set good pixels - start with the whole range set to unflagged
        flag = np.exp(logWaveCube) < 0
        # flag the red end no matter if masking is turned off - lots of telluric lines
        flag = np.exp(logWaveCube) > 2.329    
        #flag = np.exp(logWaveCube) > 2.312
        goodPixels = np.where(flag == 0)[0]
        # before, set all pixels to good pixels
        #goodPixels = np.arange(len(logWaveCube))
    #pdb.set_trace()
    
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
        #yy = [35]
        #xx = [17,18]
        #yy = [35,35]
        #xx = [0,1]
        xx = [17]
        yy = [76]

        allxxyylist = list(itertools.product(xx,yy))
    else:
        if inputVoronoiFile is None:
            # then fit all spaxels
            #xx = np.arange(8, newCube.shape[0]-8)
            #yy = np.arange(10, newCube.shape[1]-10)
            xx = np.arange(newCube.shape[0])
            yy = np.arange(newCube.shape[1])

            allxxyylist = list(itertools.product(xx,yy))
        else:
            # grab only 1 value per Voronoi bin
            inyy, inxx, binnum = np.loadtxt(inputVoronoiFile,unpack=True)
            inxx = inxx.astype(int)
            inyy = inyy.astype(int)
            # only need one spaxel for each bin (they're all the same within a bin)
            u, idx = np.unique(binnum, return_index=True)
            yy = inyy[idx]
            xx = inxx[idx]
        
            allxxyylist = list(zip(xx,yy))
            
    allxxyy = np.array(allxxyylist)

    #pdb.set_trace()
    t1=time.time()
    if rerun is True:
        job_server = pp.Server()
        print "Starting pp with", job_server.get_ncpus(), "workers"
        jobs = [(i,job_server.submit(run_once_mc, (newCube[i[0],i[1],:],newErrors[i[0],i[1],:],templates,velScale,start,goodPixels,dv,nsim,mctmpdir,jackknife,test,i), (), ('numpy as np','time','ppxf'))) for i in allxxyy]
        #test = run_once_mc(newCube[allxxyy[0,0],allxxyy[0,1]],newErrors[allxxyy[0,0],allxxyy[0,1]],templates,velScale,start,goodPixels,dv,nsim,mctmpdir,jackknife,test,allxxyy[0])
        #pdb.set_trace()
        # wait for all the jobs to finish before proceeding
        job_server.wait()
        #pdb.set_trace()
    
    for i in range(len(allxxyy)):
        print "Setting output of ", i
        xx = allxxyy[i][0]
        yy = allxxyy[i][1]

        if ((newErrors[xx,yy,:].mean() !=0.) and (np.isinf(newErrors[xx,yy,:].mean()) == 0.)):
        
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

            if inputVoronoiFile is not None:
                binxxyyidx = np.where((inxx == xx) & (inyy == yy))
                binxxyy = binnum[binxxyyidx]
                binidx = np.where(binnum == binxxyy)
                xxtmp = inxx[binidx]
                yytmp = inyy[binidx]
                for j in range(len(xxtmp)):
                    velocityErr[xxtmp[j], yytmp[j]] = p.velocityErr
                    sigmaErr[xxtmp[j],yytmp[j]] = p.sigmaErr
                    h3Err[xxtmp[j],yytmp[j]] = p.h3Err
                    h4Err[xxtmp[j],yytmp[j]] = p.h4Err
                    chi2redErr[xxtmp[j],yytmp[j]] = p.chi2redErr
                    pweightsErr[xxtmp[j],yytmp[j],:] = p.pweightsErr
                    tweightsErr[xxtmp[j],yytmp[j],:] = p.tweightsErr
        
                    velocityAvg[xxtmp[j], yytmp[j]] = p.velocity
                    sigmaAvg[xxtmp[j],yytmp[j]] = p.sigma
                    h3Avg[xxtmp[j],yytmp[j]] = p.h3
                    h4Avg[xxtmp[j],yytmp[j]] = p.h4
                    chi2redAvg[xxtmp[j],yytmp[j]] = p.chi2red
                    pweightsAvg[xxtmp[j],yytmp[j],:] = p.pweights
                    tweightsAvg[xxtmp[j],yytmp[j],:] = p.tweights
                    
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
    #minFlux = tmp.mean() - (3.0 * tmp.std())
    #tmp -= minFlux

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
    def __init__(self, inputFile=workdir+'ppxf.dat',bestfit=False,twocomp=False,moments=4):
        self.inputFile = inputFile
        
        input = open(inputFile, 'r')
        self.velocity = pickle.load(input)
        self.sigma = pickle.load(input)
        if moments == 4:
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
    xaxis = np.arange(cubeimg.shape[0]) - bhpos_pix[0] * 0.05
    yaxis = np.arange(cubeimg.shape[1]) - bhpos_pix[1] * 0.05
    
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
    py.plot([bhpos_hor[0]], [bhpos_hor[1]], 'ko', markeredgewidth=2,markerfacecolor='None')
    py.ylabel('Y (arcsec)')
    py.xlabel('X (arcsec)')
    py.gca().get_xaxis().set_major_locator(xtickLoc)
    py.axis('image')
    cbar = py.colorbar(orientation='vertical')
    cbar.set_label('Flux (cts/sec)')

    # Make a compass rose
    #pa = 56.0
    pa = 34.0
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
    #snrimg=snrimg.transpose()
    snrimg=np.rot90(snrimg,3)
    py.imshow(py.ma.masked_invalid(snrimg), 
              extent=[xaxis[0], xaxis[-1], yaxis[0], yaxis[-1]],
              cmap=py.cm.jet)
    py.plot([bhpos_hor[0]], [bhpos_hor[1]], 'ko', markeredgewidth=2,markerfacecolor='None')
    py.xlabel('X (arcsec)')
    py.gca().get_xaxis().set_major_locator(xtickLoc)
    py.axis('image')
    cbar = py.colorbar(orientation='vertical')
    cbar.set_label('SNR')
    py.title('SNR')

    #pdb.set_trace()

    ##########
    # Plot Velocity
    ##########
    py.subplot(4, 1, 3)
    velimg = p.velocity.transpose()+vsys
    py.imshow(np.rot90(py.ma.masked_where(velimg==vsys,velimg),3), vmin=-250., vmax=250.,
              extent=[xaxis[0], xaxis[-1], yaxis[0], yaxis[-1]])
    py.plot([bhpos_hor[0]], [bhpos_hor[1]], 'ko', markeredgewidth=2,markerfacecolor='None')
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
    py.plot([bhpos_hor[0]], [bhpos_hor[1]], 'ko', markeredgewidth=2,markerfacecolor='None')
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

    xaxis = np.arange(cubeimg.shape[0]) - bhpos_pix[0] * 0.05
    yaxis = np.arange(cubeimg.shape[1]) - bhpos_pix[1] * 0.05
    
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
    py.plot([bhpos_hor[0]], [bhpos_hor[1]], 'ko', markeredgewidth=2,markerfacecolor='None')
    py.ylabel('Y (arcsec)')
    py.xlabel('X (arcsec)')
    py.gca().get_xaxis().set_major_locator(xtickLoc)
    py.axis('image')
    cbar = py.colorbar(orientation='vertical')
    cbar.set_label('Flux (cts/sec)')

    # Make a compass rose
    #pa = 56.0
    pa = 34.0
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
    velimg = p.velocity.transpose()+vsys
    py.imshow(np.rot90(py.ma.masked_where(velimg==vsys,velimg),3), vmin=-250., vmax=250., 
              extent=[xaxis[0], xaxis[-1], yaxis[0], yaxis[-1]])
    py.plot([bhpos_hor[0]], [bhpos_hor[1]], 'ko', markeredgewidth=2,markerfacecolor='None')
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
    py.plot([bhpos_hor[0]], [bhpos_hor[1]], 'ko', markeredgewidth=2,markerfacecolor='None')
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

def plotResults3(inputFile,zoom=False,twocomp=False,moments=4):
    #cubeimg = pyfits.getdata(datadir + cuberoot + '_img.fits')
    #cubeimg = pyfits.getdata('/Users/kel/Documents/Projects/M31/data/081021/SPEC/reduce_new/cleanrecmat/m31/woscalecont/ppxf/m31_081021_mosaic_woscalecont_img.fits')
    #cubeimg=pyfits.getdata('/Users/kel/Documents/Projects/M31/data/osiris_mosaics/drf/sigclip/all_telshift/woscalecont/m31_mosaic_telshift_woscalecont_img.fits')
      
    p = PPXFresults(inputFile,twocomp=twocomp,moments=moments)

    xaxis = (np.arange(len(p.velocity[0])) - bhpos_pix[0]) * 0.05
    yaxis = (np.arange(len(p.velocity)) - bhpos_pix[1]) * 0.05

    if zoom:
        xaxistmp = np.arange(len(p.velocity[0])) * 0.05
        yaxistmp = np.arange(len(p.velocity)) * 0.05
        x0 = np.abs(xaxistmp-(bhpos_hor[0]-0.5)).argmin()
        x1 = np.abs(xaxistmp-(bhpos_hor[0]+0.5)).argmin()
        y0 = np.abs(yaxistmp-(bhpos_hor[1]-0.5)).argmin()
        y1 = np.abs(yaxistmp-(bhpos_hor[1]+0.5)).argmin()
    else:
        x0 = 0
        x1 = -1
        y0 = 0
        y1 = -1
    
    xtickLoc = py.MultipleLocator(0.5)

    py.close(2)
    if zoom:
        py.figure(2, figsize=(10,7))
    else:
        py.figure(2, figsize=(15,7))
    py.subplots_adjust(left=0.07, right=0.94, top=0.95)
    py.clf()

    ##########
    # Plot Velocity
    ##########
    py.subplot(2, 2, 1)
    if twocomp:
        velimg = p.velocity2.transpose()+vsys
    else:
        velimg = p.velocity.transpose()+vsys
    plvel = np.rot90(velimg,3)
    py.imshow(py.ma.masked_where(plvel[y0:y1,x0:x1]==vsys,plvel[y0:y1,x0:x1]), vmin=-250., vmax=250.,
              extent=[xaxis[x0], xaxis[x1], yaxis[y0], yaxis[y1]])
    py.plot([0], 'ko', markeredgewidth=2,markerfacecolor='None')
    #py.imshow(np.rot90(py.ma.masked_where(velimg==vsys,velimg),3), vmin=-250., vmax=250.)
    #py.plot([bhpos_pix[0]],[bhpos_pix[1]],'ko', markeredgewidth=2,markerfacecolor='None')
    py.ylabel('Y (arcsec)')
    #py.xlabel('X (arcsec)')
    py.gca().get_xaxis().set_major_locator(xtickLoc)
    py.axis('image')
    cbar = py.colorbar(orientation='vertical',ticks=[-250.,0.,250])
    cbar.set_label('Velocity (km/s)')

    ##########
    # Plot Dispersion
    ##########
    py.subplot(2, 2, 2)
    if twocomp:
        sigimg = p.sigma2.transpose()
    else:
       sigimg = p.sigma.transpose()
    plsig = np.rot90(sigimg,3)
    py.imshow(py.ma.masked_where(plsig[y0:y1,x0:x1]==0.,plsig[y0:y1,x0:x1]), vmin=0., vmax=300.,
              extent=[xaxis[x0], xaxis[x1], yaxis[y0], yaxis[y1]],
              cmap=py.cm.jet)
    py.plot([0], 'ko', markeredgewidth=2,markerfacecolor='None')
    #py.xlabel('X (arcsec)')
    py.gca().get_xaxis().set_major_locator(xtickLoc)
    py.axis('image')
    cbar = py.colorbar(orientation='vertical',ticks=[0.,150.,300])
    cbar.set_label('Dispersion (km/s)')

    if moments == 4:
        ##########
        # Plot h3
        ##########
        py.subplot(2, 2, 3)
        if twocomp:
            h3img = p.h3_2.transpose()
        else:
            h3img = p.h3.transpose()
        plh3 = np.rot90(h3img,3)
        py.imshow(py.ma.masked_where(plh3[y0:y1,x0:x1]==0, plh3[y0:y1,x0:x1]),vmin=-.2,vmax=.2, 
                extent=[xaxis[x0], xaxis[x1], yaxis[y0], yaxis[y1]])
        py.plot([0], 'ko', markeredgewidth=2,markerfacecolor='None')
        py.ylabel('Y (arcsec)')
        py.xlabel('X (arcsec)')
        py.gca().get_xaxis().set_major_locator(xtickLoc)
        py.axis('image')
        cbar = py.colorbar(orientation='vertical',ticks=[-.2,0.,.2])
        cbar.set_label('h3')

        ##########
        # Plot h4
        ##########
        py.subplot(2, 2, 4)
        if twocomp:
            h4img = p.h4_2.transpose()
        else:
            h4img = p.h4.transpose()
        plh4 = np.rot90(h4img,3)
        py.imshow(py.ma.masked_where(plh4[y0:y1,x0:x1]==0, plh4[y0:y1,x0:x1]),vmin=-.2,vmax=.2, 
                extent=[xaxis[x0], xaxis[x1], yaxis[y0], yaxis[y1]])
        py.plot([0], 'ko', markeredgewidth=2,markerfacecolor='None')
        py.xlabel('X (arcsec)')
        py.gca().get_xaxis().set_major_locator(xtickLoc)
        py.axis('image')
        cbar = py.colorbar(orientation='vertical',ticks=[-.2,0.,.2])
        cbar.set_label('h4')

    #py.tight_layout()
    pdb.set_trace()

    if zoom:
        py.savefig(workdir + 'plots/kinematic_maps3_zoom.png')
        py.savefig(workdir + 'plots/kinematic_maps3_zoom.eps')
    else:
        py.savefig(workdir + 'plots/kinematic_maps3.png')
        py.savefig(workdir + 'plots/kinematic_maps3.eps')
    py.show()

def plotSpaxResults(inputResults=workdir+'ppxf.dat',inSpax=[20,40],blue=False,losvd=False,bs=True,mask=False):
    # given spaxel coordinates, plot the input spectrum and the best fit ppxf fit (templates
    # convolved with the LOSVD), plus the residuals.

    # inSpax is given in coordinates showing the correct orientation (i.e. after being flipped
    # along the x axis)

    # if /blue is set, only plots the blue end of the spectrum (blueward of the CO bandheads)
    # if /losvd is set, plots the spectrum+fit and the LOSVD 
    # if /bs is set, clips the wavelength vector to match the bulge-subtracted spectrum range
    # if /mask is set, plots gray boxes on the LOSVD plot where the telluric mask is

    p = PPXFresults(inputResults,bestfit=True)

    cubefits = pyfits.open(datadir + cuberoot + '.fits')
    
    cube = cubefits[0].data
    hdr = cubefits[0].header
    errors = cubefits[1].data

    #pdb.set_trace()

    # flip x coordinate
    XX = len(p.velocity[0])-inSpax[0]
    YY = inSpax[1]
    
    #spec = cube[XX,YY]
    spec = p.galaxy[YY,XX]
    bestfit = p.bestfit[YY,XX]

    # Get the wavelength solution so we can specify range:
    w0 = hdr['CRVAL1']
    dw = hdr['CDELT1']
    wavelength = w0 + dw * np.arange(cube.shape[2], dtype=float)
    wavelength /= 1000.0   # Convert to microns

    # trim telluric contaminated red end if not already trimmed
    #if len(spec) == 1665:
    waveCutTell = 2.329
    idx = np.where(wavelength < waveCutTell)[0]
    wavelength = wavelength[idx]
    #spec = spec[idx]
    #bestfit = bestfit[idx]

    if bs:
        waveCutBS = 2.185
        idx = np.where(wavelength > waveCutBS)[0]
        wavelength = wavelength[idx]
    #    spec = spec[idx]
    #    bestfit = bestfit[idx]
    #    spec = spec[idx]

    waveCut = 2.185
    if blue:
        waveCutRed = 2.285
        #idx = np.where((wavelength > waveCut) & (wavelength < waveCutRed))[0]
        #wavelength = wavelength[idx]
        wavelength = (waveCut*1000.) + dw*np.arange(spec.shape[0], dtype=float)
        wavelength /= 1000.
    else:
        waveCutRed = wavelength.max()
        wavelength = (waveCut*1000.) + dw*np.arange(spec.shape[0], dtype=float)
        wavelength /= 1000.

    # flux calibrate
    #spec = -2.5*np.log10(spec)+23.7
    #bestfit = -2.5*np.log10(bestfit)+23.7
    #res = -2.5*np.log10(spec-bestfit)+23.7
    #print np.median(spec),np.median(bestfit),(np.median(spec)-np.median(bestfit))
    minidx = np.argmin(np.abs(wavelength - waveCut))
    maxidx = np.argmin(np.abs(wavelength - waveCutRed))
    #bestfit += (np.median(spec[minidx:maxidx]) - np.median(bestfit[minidx:maxidx]))
    res = spec - bestfit

    py.close(2)
    if losvd:
        py.figure(2, figsize=(15,7))
        py.subplots_adjust(left=0.05, right=0.94, top=0.95)
        py.subplot(1,2,1)
    else:
        py.figure(2)
        
    py.plot(wavelength,spec,'k-')
    py.plot(wavelength,bestfit,'r-')
    py.plot(wavelength,res,'g-')
    py.legend(('Spectrum','Best fit','Residuals'),loc=0)
    py.xlim(waveCut,waveCutRed)
    py.xlabel('Wavelength ($\mu$m)')
    py.ylabel('Flux')
    pltitle = 'Spectrum with pPXF best fit for spaxel [%i, %i]' % (inSpax[0], YY)
    py.title(pltitle)

    if mask:
        tellLinesLo = [2.285, 2.312]
        tellLinesHi = [2.2895, 2.318]
        for i in range(len(tellLinesLo)):
            py.axvspan(tellLinesLo[i], tellLinesHi[i], facecolor='k', alpha=0.25)
                       

    if losvd:
        py.subplot(1,2,2)
        x = np.arange(0,2000)-1000.
        y = gaussHermite(x,100.,p.velocity[YY,XX]+vsys,p.sigma[YY,XX],p.h3[YY,XX],p.h4[YY,XX])
        
        py.plot(x,y)
        veltext = 'v = %.1f' % (p.velocity[YY,XX]+vsys)
        sigtext = '$\sigma$ = %.1f' % p.sigma[YY,XX]
        h3text = 'h3 = %.2f' % p.h3[YY,XX]
        h4text = 'h4 = %.2f' % p.h4[YY,XX]
        py.figtext(.55,.85,veltext)
        py.figtext(.55,.8,sigtext)
        py.figtext(.55,.75,h3text)
        py.figtext(.55,.7,h4text)
        py.xlabel('Velocity (km s$^{-1}$)')
        pltitle = 'LOSVD for spaxel [%i, %i]' % (inSpax[0], YY)
        py.title(pltitle)

    #pdb.set_trace()

def plotSpaxMaskTell(inResults1=None,inResults2=None,inTell=None,inSpax=None,inCube='/Users/kel/Documents/Projects/M31/data/osiris_mosaics/drf/sigclip/all_NIRC2_CC_DTOTOFF_2/no100828/mosaic_all_nirc2_cc_dtotoff_no100828_scalederr.fits'):
    # generally, inResults1 = no mask, inResults2 = w mask
    p1 = PPXFresults(inResults1,bestfit=True)
    p2 = PPXFresults(inResults2,bestfit=True)

    spec = p1.galaxy[inSpax[0],inSpax[1]]
    bestfit1 = p1.bestfit[inSpax[0],inSpax[1]]
    bestfit2 = p2.bestfit[inSpax[0],inSpax[1]]

    cube, hdr = pyfits.getdata(inCube,header=True)

    #w0 = hdr['CRVAL1']
    w0 = 2185.25
    dw = hdr['CDELT1']
    wavelength = w0 + dw * np.arange(spec.shape[0], dtype=float)
    wavelength /= 1000.0

    waveCut = 2.185
    idx = np.where(wavelength > waveCut)
    wavelength = wavelength[idx]

    tell,thdr = pyfits.getdata(inTell,header=True)

    w0 = thdr['CRVAL1']
    dw = thdr['CDELT1']
    twave = w0 + dw * np.arange(tell.shape[0], dtype=float)
    twave /= 1000.0

    py.close(2)
    py.figure(2, figsize=(15,7))
    py.subplots_adjust(left=0.05, right=0.94, top=0.95)
    py.subplot(1,2,1)

    py.plot(wavelength,spec,'k-',label='Spectrum')
    py.plot(wavelength,bestfit1,'b-',label='No telluric masking')
    py.plot(wavelength,bestfit2,'g-',label='With masking')
    py.legend(loc=0)
    py.xlim(2.185,2.329)
    pltitle = 'Spectrum with pPXF best fit for spaxel [%i, %i]' % (inSpax[0], inSpax[1])
    py.title(pltitle)
    # flagwave 1
    py.axvspan(2.288,2.2915, facecolor='k', alpha=0.25)
    py.axvspan(2.314,2.325, facecolor='k', alpha=0.25)
    # flagwave 2
    #py.axvspan(2.2913,2.293,facecolor='k', alpha=0.25)

    py.subplot(1,2,2)

    py.plot(twave,tell)
    py.title('Telluric spectrum')
    py.xlim(2.185,2.329)
    # flagwave 1
    py.axvspan(2.288,2.2915, facecolor='k', alpha=0.25)
    py.axvspan(2.314,2.325, facecolor='k', alpha=0.25)
    # flagwave 2
    #py.axvspan(2.2913,2.293,facecolor='k', alpha=0.25)

    #pdb.set_trace()
def plotFluxKin(inputResults=None,incubeimg=None):
    # plot a flux map with the 0 km/s velocity line and then the 200 km/s dispersion line
    cubeimg = pyfits.getdata(incubeimg)
    cubeimg = np.rot90(cubeimg,3)

    xaxis = (np.arange(cubeimg.shape[1]) - bhpos_pix[0]) * 0.05
    yaxis = (np.arange(cubeimg.shape[0]) - bhpos_pix[1]) * 0.05

    xtickLoc = py.MultipleLocator(0.5)

    p = PPXFresults(inputResults,bestfit=True)
    vel = np.rot90(p.velocity.T,3)+vsys
    sig = np.rot90(p.sigma.T,3)

    # smooth data for contours
    smsig = .5
    vel = scipy.ndimage.filters.gaussian_filter(vel, smsig)
    sig = scipy.ndimage.filters.gaussian_filter(sig, smsig)

    py.close(2)
    py.figure(2,figsize=(9,9))
    py.subplots_adjust(left=0.07, right=0.94, top=0.95)
    py.subplot(211)
    py.imshow(py.ma.masked_where(cubeimg==0,cubeimg),extent=[xaxis[0], xaxis[-1], yaxis[0], yaxis[-1]])
    py.plot([0], 'ko', markeredgewidth=2,markerfacecolor='None')
    cs = py.contour(vel,[-200,-100,0,100,200],extent=[xaxis[0], xaxis[-1], yaxis[0], yaxis[-1]])
    py.setp(cs.collections,linewidth=2)
    py.setp(cs.collections[2],linewidth=3)
    py.ylabel('Velocity contours')
    #py.clabel(cs)
    #py.title('OSIRIS flux map w/ systemic velocity contours')
    
    #pdb.set_trace()

    #py.close(2)
    #py.figure(2)
    py.subplot(212)
    py.imshow(py.ma.masked_where(cubeimg==0,cubeimg),extent=[xaxis[0], xaxis[-1], yaxis[0], yaxis[-1]])
    py.plot([0], 'ko', markeredgewidth=2,markerfacecolor='None')
    cs = py.contour(sig,[200,300],extent=[xaxis[0], xaxis[-1], yaxis[0], yaxis[-1]])
    py.setp(cs.collections,linewidth=2)
    py.ylabel('Dispersion contours')
    py.xlabel('X (arcsec)')
    #py.clabel(cs)
    #py.title('OSIRIS flux map w/ dispersion contours')

    #pdb.set_trace()
    
def plotTempWeights(inputResults=workdir+'ppxf.dat'):
    # Creates n/6 plots (6 plots per page), where n=# of templates used, colored by the weight

    p = PPXFresults(inputResults)
    tw = p.tweights

    ntw = tw.shape[2]
    nplot = np.ceil(ntw/6.)

    goodtemp = np.zeros((tw.shape[0],tw.shape[1],tw.shape[2]),dtype=int)
    good = np.where(tw != 0)
    goodtemp[good] = 1
    ntemp = np.sum(goodtemp,axis=2)

    names, spectype = load_template_names()

    xaxis = (np.arange(tw.shape[1]) - bhpos_pix[0]) * 0.05
    yaxis = (np.arange(tw.shape[0]) - bhpos_pix[1]) * 0.05
    
    xtickLoc = py.MultipleLocator(0.5)

    # discrete colormap
    tmpcmap = py.get_cmap('jet',10)
    
    # plot the number of templates used at each bin
    py.close(2)
    py.figure(2, figsize=(7,4))
    py.imshow(np.rot90(ntemp.T,3), extent=[xaxis[0], xaxis[-1], yaxis[0], yaxis[-1]], cmap=tmpcmap)
    py.plot([bhpos_hor[0]], [bhpos_hor[1]], 'ko', markeredgewidth=2,markerfacecolor='None')
    py.ylabel('Y (arcsec)')
    py.xlabel('X (arcsec)')
    py.gca().get_xaxis().set_major_locator(xtickLoc)
    py.axis('image')
    cbar = py.colorbar(orientation='vertical',cmap=tmpcmap)
    cbar.set_label('Number of templates')

    #py.savefig(workdir + 'plots/tempweights_ntemp.png')
    py.show()
    pdb.set_trace()
    
    py.close(2)
    py.figure(2, figsize=(15,13))
    py.subplots_adjust(left=0.05, right=0.94, top=0.95)

    # plot the weights of each template
    for pagen in np.arange(nplot):
        py.clf()
        for plotn in np.arange(6):
            py.subplot(3,2,plotn+1)
            tmpn = (pagen*6) + plotn
            tmpn = tmpn.astype('int')
            if tmpn <= (ntw-1):
                tmp = tw[:,:,tmpn]
                tmpimg = np.rot90(tmp.T,3)
                py.imshow(tmpimg, extent=[xaxis[0], xaxis[-1], yaxis[0], yaxis[-1]],cmap=cmaps.inferno)
                py.plot([bhpos_hor[0]], [bhpos_hor[1]], 'cx', markeredgewidth=3)
                py.ylabel('Y (arcsec)')
                py.xlabel('X (arcsec)')
                py.gca().get_xaxis().set_major_locator(xtickLoc)
                py.axis('image')
                cbar = py.colorbar(orientation='vertical')
                lab = '%s (%s)'  % (names[tmpn], spectype[tmpn])
                cbar.set_label(lab)
        
        #filename = 'plots/tempweights_%1.0f.png' % pagen
        #py.savefig(workdir + filename)
        py.show()
        pdb.set_trace()

def plotChi2(inputResults=workdir+'/ppxf.dat'):
    p = PPXFresults(inputResults)

    xaxis = np.arange(p.chi2red.shape[1]) - bhpos_pix[0] * 0.05
    yaxis = np.arange(p.chi2red.shape[0]) - bhpos_pix[1] * 0.05
    
    xtickLoc = py.MultipleLocator(0.5)

    py.close(2)
    py.figure(2, figsize=(8,4))
    py.subplots_adjust(left=0.1, right=0.96, top=0.95)
    py.clf()

    chi2 = np.rot90(p.chi2red.T,3)
    py.imshow(chi2, extent=[xaxis[0], xaxis[-1], yaxis[0], yaxis[-1]], vmin=0.,vmax=5.)
    py.plot([bhpos_hor[0]], [bhpos_hor[1]], 'ko', markeredgewidth=2,markerfacecolor='None')
    py.ylabel('Y (arcsec)')
    py.xlabel('X (arcsec)')
    py.gca().get_xaxis().set_major_locator(xtickLoc)
    py.axis('image')
    cbar = py.colorbar(orientation='vertical')
    cbar.set_label('Reduced $\chi^2$')

def plotSpec(inputCube=workdir+'m31_all_scalederr_cleanhdr_bulgesub_vorcube_20160825.fits',spax=[20,45]):
    # plot the spectrum in a given spaxel
    cubefits = pyfits.open(inputCube)

    cube = cubefits[0].data
    hdr = cubefits[0].header

    # Get the wavelength solution so we can specify range:
    w0 = hdr['CRVAL1']
    dw = hdr['CDELT1']
    wavelength = w0 + dw * np.arange(cube.shape[2], dtype=float)
    wavelength /= 1000.0   # Convert to microns

    py.close(2)
    py.figure(2,figsize=(11,6))
    py.clf()
    
    py.plot(wavelength,cube[spax[0],spax[1],:])
    py.xlim(2.18,2.36)
    py.xlabel('Wavelength ($\mu$m)')
    py.ylabel('Flux (DN s$^{-1}$)')

    
    py.text(2.205,0.39,'Na')
    py.text(2.222,0.38,'Fe')
    py.text(2.235,0.38,'Fe')
    py.text(2.246,0.38,'Fe')
    py.text(2.26,0.36,'Ca')
    py.text(2.281,0.36,'Mg')
    py.text(2.29,0.31,'$^{12}$CO')
    py.text(2.32,0.31,'$^{12}$CO')
    py.text(2.345,0.31,'$^{12}$CO')

    pdb.set_trace()

def plotCubeComp(inCube1=None,inCube2=None,shift1=[0.,0.],shift2=[0.,0.],spax1=None,leg1=None,leg2=None):
    # spax: [long axis, short axis]
    # shift: [long axis, short axis]
    
    cubefits = pyfits.open(inCube1)

    cube1 = cubefits[0].data
    hdr = cubefits[0].header

    cube2= pyfits.getdata(inCube2)

    # Get the wavelength solution so we can specify range:
    w0 = hdr['CRVAL1']
    dw = hdr['CDELT1']
    wavelength = w0 + dw * np.arange(cube1.shape[2], dtype=float)
    wavelength /= 1000.0   # Convert to microns

    # figure out the matching spaxels using the shifts
    spax1 = np.array(spax1)
    shift1 = np.array(shift1)
    shift2 = np.array(shift2)
    spax2 = np.array(([0.,0.]),dtype=float)
    spax2[0] = spax1[0] + (shift1[0] - shift2[0])
    spax2[1] = spax1[1] + (shift1[1] - shift2[1])

    # grab just the matching spaxels
    spec1 = cube1[spax1[1],spax1[0],:]
    spec2 = cube2[spax2[1],spax2[0],:]

    # grab some neighboring spaxels to boost S/N
    # this creates a 3x3 cube, centered on the given spaxel
    #spec1all = cube1[(spax1[1]-1):(spax1[1]+2),(spax1[0]-1):(spax1[0]+2),:]
    #spec2all = cube2[(spax2[1]-1):(spax2[1]+2),(spax2[0]-1):(spax2[0]+2),:]
    # this creates a 2x5 horizontal (along the long axis) strip, with the given spaxel centered in the bottom row
    spec1all = cube1[spax1[1]:(spax1[1]+2),(spax1[0]-2):(spax1[0]+3),:]
    spec2all = cube2[spax2[1]:(spax2[1]+2),(spax2[0]-2):(spax2[0]+3),:]

    spec1combtmp = np.median(spec1all,axis=0)
    spec1comb = np.median(spec1combtmp,axis=0)
    spec2combtmp = np.median(spec2all,axis=0)
    spec2comb = np.median(spec2combtmp,axis=0)

    # smooth lightly to get rid of some noise
    spec1sm = scipy.signal.medfilt(spec1,3)
    spec2sm = scipy.signal.medfilt(spec2,3)

    spec1combsm = scipy.signal.medfilt(spec1comb,3)
    spec2combsm = scipy.signal.medfilt(spec2comb,3)

    # heavily smooth to get a continuum estimate
    #cont1 = scipy.signal.medfilt(spec1,401)
    #cont2 = scipy.signal.medfilt(spec2,401)

    #cont1c = scipy.signal.medfilt(spec1comb,401)
    #cont2c = scipy.signal.medfilt(spec2comb,401)
    
    # fit a polynomial to get a continuum estimate
    cont1p = np.polyfit(wavelength,np.nan_to_num(spec1),2)
    cont2p = np.polyfit(wavelength,np.nan_to_num(spec2),2)

    cont1 = np.polyval(cont1p,wavelength)
    cont2 = np.polyval(cont2p,wavelength)

    cont1cp = np.polyfit(wavelength,np.nan_to_num(spec1comb),2)
    cont2cp = np.polyfit(wavelength,np.nan_to_num(spec2comb),2)

    cont1c = np.polyval(cont1cp,wavelength)
    cont2c = np.polyval(cont2cp,wavelength)
    
    # divide by continuum to get just absorption lines
    lines1 = np.nan_to_num(spec1sm) - cont1
    lines2 = np.nan_to_num(spec2sm) - cont2

    lines1c = np.nan_to_num(spec1combsm) - cont1c
    lines2c = np.nan_to_num(spec2combsm) - cont2c

    if leg1 is None:
        leg1 = '1'
    if leg2 is None:
        leg2 = '2'

    py.figure(3)
    py.clf()
    py.plot(wavelength,spec1sm,'b-')
    py.plot(wavelength,spec2sm,'g-')

    py.plot(wavelength,cont1,'r-')
    py.plot(wavelength,cont2,'r-')

    py.plot(wavelength,lines1,'m-')
    py.plot(wavelength,lines2,'c-')

    py.title('Single spectrum comparison')
    py.xlabel('Wavelength ($\mu$m)')
    py.ylabel('Flux (DN/s, or normalized)')
    py.legend(('Smooth spec '+leg1,'Smooth spec '+leg2,'Cont '+leg1,'Cont '+leg2,'Lines only, '+leg1,'Lines only, '+leg2),loc=0)

    py.figure(4)
    py.clf()
    py.plot(wavelength,spec1combsm,'b-')
    py.plot(wavelength,spec2combsm,'g-')

    py.plot(wavelength,cont1c,'r-')
    py.plot(wavelength,cont2c,'r-')

    py.plot(wavelength,lines1c,'m-')
    py.plot(wavelength,lines2c,'c-')

    py.title('2x5 spectrum comparison')
    py.xlabel('Wavelength ($\mu$m)')
    py.ylabel('Flux (DN/s, or normalized)')
    py.legend(('Smooth spec '+leg1,'Smooth spec '+leg2,'Cont '+leg1,'Cont '+leg2,'Lines only, '+leg1,'Lines only, '+leg2),loc=0)
    
    pdb.set_trace()

def plotFrameComp(inPath=None,shiftFileSE=None,shiftFileNW=None,spax=None,shifts=False):
    # Plots spectra from multiple frames together, specifically to check
    # whether spectra from the SE/NW pointings are different from each other.
    # Frames are input by entering the shifts files, which contain the file names
    # (assumed to be just file names, relative to the inPath keyword)
    # Spax is the spaxel to plot ([long axis,short axis]). If shifts=False,
    # uses the detector coordinates (so spaxels are the same in all frames).
    # If shifts=True, uses the shifts from the shifts file (so

    SEshifts = pandas.read_csv(shiftFileSE,delim_whitespace=True,header=None,names=['l','s','f'])
    NWshifts = pandas.read_csv(shiftFileNW,delim_whitespace=True,header=None,names=['l','s','f'])

    cube,hdr = pyfits.getdata(inPath+'/'+SEshifts['f'][0],header=True)
    w0 = hdr['CRVAL1']
    dw = hdr['CDELT1']
    wavelength = w0 + dw * np.arange(cube.shape[2], dtype=float)
    wavelength /= 1000.0   # Convert to microns

    py.close(3)
    py.figure(3)
    py.clf()

    py.plot([], label='SE frames', color='b')
    py.plot([], label='NW frames', color='g')
    
    for i in np.arange(len(SEshifts['f'])):
        if shifts:
            if i==0:
                lshift0 = SEshifts['l'][0]
                sshift0 = SEshifts['s'][0]
            lshift = SEshifts['l'][i]
            sshift = SEshifts['s'][i]
            spax2se = np.array(([0.,0.]),dtype=float)
            spax2se[0] = spax[0] + (lshift0 - lshift)
            spax2se[1] = spax[1] + (sshift0 - sshift)
        else:
            spax2se = spax
        tmp = pyfits.getdata(inPath+'/'+SEshifts['f'][i])
        py.plot(wavelength,tmp[spax2se[1],spax2se[0],:],'b-')

    for j in np.arange(len(NWshifts['f'])):
        if shifts:
            if j==0:
                lshift0 = NWshifts['l'][0]
                sshift0 = NWshifts['s'][0]
            lshift = NWshifts['l'][j]
            sshift = NWshifts['s'][j]
            spax2nw = np.array(([0.,0.]),dtype=float)
            spax2nw[0] = spax[0] + (lshift0 - lshift)
            spax2nw[1] = spax[1] + (sshift0 - sshift)
        else:
            spax2nw = spax
        tmp = pyfits.getdata(inPath+'/'+NWshifts['f'][j])
        py.plot(wavelength,tmp[spax2nw[1],spax2nw[0],:],'g-')

    py.xlabel('Wavelength')
    if shifts:
        titpl = 'SE/NW frames, spaxel [%(xx)s,%(yy)s] (physical)' % {"xx": spax[0], "yy": spax[1]}
    else:
        titpl = 'SE/NW frames, spaxel [%(xx)s,%(yy)s] (detector)' % {"xx": spax[0], "yy": spax[1]}
    py.title(titpl)
    py.legend()

    pdb.set_trace()
        

def plotCubeTell(inOrgCube=None,inTLCCube=None,inTLC=None,spax=None):
    ### Plots original cube (not telluric correction), telluric-corrected cube, and the telluric spectrum,
    ### all at the given spaxel

    org, hdr = pyfits.getdata(inOrgCube,header=True)
    tlccube = pyfits.getdata(inTLCCube)
    tlcall = pyfits.getdata(inTLC)

    # Get the wavelength solution so we can specify range:
    w0 = hdr['CRVAL1']
    dw = hdr['CDELT1']
    wavelength = w0 + dw * np.arange(org.shape[2], dtype=float)
    wavelength /= 1000.0   # Convert to microns

    spec1 = org[spax[1],spax[0],:]/np.median(org[spax[1],spax[0],:])
    spec2 = tlccube[spax[1],spax[0],:]/np.median(tlccube[spax[1],spax[0],:])
    #tlc = tlcall[spax[1],spax[0],:]#/np.median(tlcall[spax[1],spax[0],:])
    tlc = tlcall
    
    # grab some neighboring spaxels to boost S/N
    # this creates a 3x3 cube, centered on the given spaxel
    #spec1all = org[(spax[1]-1):(spax[1]+2),(spax[0]-1):(spax[0]+2),:]
    #spec2all = tlccub[(spax[1]-1):(spax[1]+2),(spax[0]-1):(spax[0]+2),:]
    # this creates a 2x5 horizontal (along the long axis) strip, with the given spaxel centered in the bottom row
    spec1all = org[spax[1]:(spax[1]+2),(spax[0]-2):(spax[0]+3),:]
    spec2all = tlccube[spax[1]:(spax[1]+2),(spax[0]-2):(spax[0]+3),:]

    spec1combtmp = np.median(spec1all,axis=0)
    spec1comb = np.median(spec1combtmp,axis=0)/np.median(spec1all)
    spec2combtmp = np.median(spec2all,axis=0)
    spec2comb = np.median(spec2combtmp,axis=0)/np.median(spec2all)

    # heavily smooth and normalize the telluric continuum to get a continuum estimate
    #conttlc = scipy.signal.medfilt(tlc,401)/np.median(tlc)
    # fit a polynomial to get a continuum estimate
    conttlcp = np.polyfit(wavelength,tlc,2)
    conttlctmp = np.polyval(conttlcp,wavelength)
    # normalize
    conttlc = conttlctmp/np.median(conttlctmp)
 
    # subtract continuum to get just absorption lines
    lines1 = spec1 - conttlc
    lines1comb = spec1comb - conttlc

    py.figure(2)
    py.clf()
    py.plot(wavelength,spec1)
    py.plot(wavelength,spec2)
    #py.plot(wavelength,lines1)
    py.plot(wavelength,(tlc))#/np.median(tlc))+1.5)
    #py.plot(wavelength,conttlc+1.5)
    
    py.xlabel('Wavelength ($\mu$m)')
    py.ylabel('Flux (norm)')
    py.title('Single spaxel TLC comparison')
    #py.legend(('Cube, no TLC/SS','Cube, TLC/SS','Cube, no TLC, TLC cont corr','Tell spectrum + offset','Tell continuum + offset'),loc=0)
    py.legend(('Cube, no TLC/SS','Cube, TLC/SS','Tell spectrum + offset'),loc=0)

    py.figure(3)
    py.clf()
    py.plot(wavelength,spec1comb)
    py.plot(wavelength,spec2comb)
    #py.plot(wavelength,lines1comb)
    py.plot(wavelength,(tlc))#/np.median(tlc))+1.5)
    #py.plot(wavelength,conttlc+1.5)
    
    py.xlabel('Wavelength ($\mu$m)')
    py.ylabel('Flux (norm)')
    py.title('2x5 spaxel TLC comparison')
    #py.legend(('Cube, no TLC/SS','Cube, TLC/SS','Cube, no TLC, TLC cont corr','Tell spectrum + offset','Tell continuum + offset'),loc=0)
    py.legend(('Cube, no TLC','Cube, TLC','Tell spectrum '),loc=0)
    
    pdb.set_trace()
    
def plotErr1(inputResults=workdir+'/ppxf.dat',inputAvg=workdir+'/ppxf_avg_mc_nsim100.dat',inputErr=workdir+'/ppxf_errors_mc_nsim100.dat'):
    ### Plots error on velocity and velocity dispersion
    cubeimg = pyfits.getdata(datadir + cuberoot + '_img.fits')
    
    p = PPXFresults(inputResults)
    a = PPXFresults(inputAvg)
    e = PPXFresults(inputErr)

    xaxis = np.arange(cubeimg.shape[0]) - bhpos_pix[0] * 0.05
    yaxis = np.arange(cubeimg.shape[1]) - bhpos_pix[1] * 0.05
    
    xtickLoc = py.MultipleLocator(0.5)

    py.close(2)
    py.figure(2, figsize=(15,13))
    py.subplots_adjust(left=0.05, right=0.94, top=0.95)
    py.clf()

    ##########
    # Plot Velocity, MC velocity average, and MC velocity error
    ##########
    py.subplot(3, 2, 1)
    velimg = p.velocity.transpose()+vsys
    py.imshow(np.rot90(py.ma.masked_where(velimg==vsys,velimg),3), vmin=-250., vmax=250., 
              extent=[xaxis[0], xaxis[-1], yaxis[0], yaxis[-1]])
    py.plot([bhpos_hor[0]], [bhpos_hor[1]], 'ko', markeredgewidth=2,markerfacecolor='None')
    py.ylabel('Y (arcsec)')
    py.xlabel('X (arcsec)')
    py.gca().get_xaxis().set_major_locator(xtickLoc)
    py.axis('image')
    cbar = py.colorbar(orientation='vertical')
    cbar.set_label('Velocity (km/s)')

    #pdb.set_trace()
    
    py.subplot(3,2,3)
    velavg = a.velocity.transpose()+vsys
    py.imshow(np.rot90(py.ma.masked_where(velavg==vsys,velavg),3), vmin=-250., vmax=250.,
              extent=[xaxis[0], xaxis[-1], yaxis[0], yaxis[-1]])
    py.plot([bhpos_hor[0]], [bhpos_hor[1]], 'ko', markeredgewidth=2,markerfacecolor='None')
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
    py.plot([bhpos_hor[0]], [bhpos_hor[1]], 'ko', markeredgewidth=2,markerfacecolor='None')
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
    py.plot([bhpos_hor[0]], [bhpos_hor[1]], 'ko', markeredgewidth=2,markerfacecolor='None')
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
    py.plot([bhpos_hor[0]], [bhpos_hor[1]], 'ko', markeredgewidth=2,markerfacecolor='None')
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
    py.plot([bhpos_hor[0]], [bhpos_hor[1]], 'ko', markeredgewidth=2,markerfacecolor='None')
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

    xaxis = np.arange(cubeimg.shape[0]) - bhpos_pix[0] * 0.05
    yaxis = np.arange(cubeimg.shape[1]) - bhpos_pix[1] * 0.05
    
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
    py.plot([bhpos_hor[0]], [bhpos_hor[1]], 'ko', markeredgewidth=2,markerfacecolor='None')
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
    py.plot([bhpos_hor[0]], [bhpos_hor[1]], 'ko', markeredgewidth=2,markerfacecolor='None' )
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
    py.plot([bhpos_hor[0]], [bhpos_hor[1]], 'ko', markeredgewidth=2,markerfacecolor='None' )
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
    py.plot([bhpos_hor[0]], [bhpos_hor[1]], 'ko', markeredgewidth=2,markerfacecolor='None' )
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
    py.plot([bhpos_hor[0]], [bhpos_hor[1]], 'ko', markeredgewidth=2,markerfacecolor='None' )
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
    py.plot([bhpos_hor[0]], [bhpos_hor[1]], 'ko', markeredgewidth=2,markerfacecolor='None' )
    py.xlabel('X (arcsec)')
    py.gca().get_xaxis().set_major_locator(xtickLoc)
    py.axis('image')
    cbar = py.colorbar(orientation='vertical')
    cbar.set_label('h4, MC err')

    py.tight_layout()

    py.savefig(workdir + 'plots/mc_err2.png')
    py.savefig(workdir + 'plots/mc_err2.eps')
    py.show()

def plotErr3(inputErr=workdir+'/ppxf_errors_mc_nsim100.dat',JK=False):
    # plots errors in the same format as the results in plotResults3

    #cubeimg = pyfits.getdata(datadir + cuberoot + '_img.fits')
       
    e = PPXFresults(inputErr)

    xaxis = (np.arange(len(e.velocity[0])) - bhpos_pix[0]) * 0.05
    yaxis = (np.arange(len(e.velocity)) - bhpos_pix[1]) * 0.05

    xtickLoc = py.MultipleLocator(0.5)

    py.close(2)
    py.figure(2, figsize=(15,7))
    py.subplots_adjust(left=0.07, right=0.94, top=0.95)
    py.clf()

    ##########
    # Plot Velocity
    ##########
    py.subplot(2, 2, 1)
    velerr = e.velocity.transpose()
    py.imshow(np.rot90(py.ma.masked_where(velerr==0.,velerr),3), vmin=0., vmax=40.,
              extent=[xaxis[0], xaxis[-1], yaxis[0], yaxis[-1]])
    if JK:
        py.plot([0],'ko', markeredgewidth=2,markerfacecolor='None',markeredgecolor='white')
    else:
        py.plot([0], 'ko', markeredgewidth=2,markerfacecolor='None' )
    py.ylabel('Y (arcsec)')
    py.xlabel('X (arcsec)')
    py.gca().get_xaxis().set_major_locator(xtickLoc)
    py.axis('image')
    cbar = py.colorbar(orientation='vertical',ticks=[0,20,40])
    cbar.set_label('$\sigma_{\mathrm{vel}}$ (km/s)')

    ##########
    # Plot Dispersion
    ##########
    py.subplot(2, 2, 2)
    sigerr = e.sigma.transpose()
    py.imshow(np.rot90(py.ma.masked_where(sigerr==0.,sigerr),3), vmin=0., vmax=50.,
              extent=[xaxis[0], xaxis[-1], yaxis[0], yaxis[-1]],
              cmap=py.cm.jet)
    if JK:
        py.plot([0], 'wo', markeredgewidth=2,markerfacecolor='None',markeredgecolor='white' )
    else:
        py.plot([0], 'ko', markeredgewidth=2,markerfacecolor='None' )
    py.xlabel('X (arcsec)')
    py.gca().get_xaxis().set_major_locator(xtickLoc)
    py.axis('image')
    cbar = py.colorbar(orientation='vertical',ticks=[0,25,50])
    cbar.set_label('$\sigma_{\mathrm{disp}}$ (km/s)')

    ##########
    # Plot h3
    ##########
    py.subplot(2, 2, 3)
    h3err = e.h3.transpose()
    py.imshow(np.rot90(py.ma.masked_where(h3err==0.,h3err),3),vmin=0.,vmax=.05, 
              extent=[xaxis[0], xaxis[-1], yaxis[0], yaxis[-1]])
    if JK:
        py.plot([0], 'wo', markeredgewidth=2,markerfacecolor='None',markeredgecolor='white' )
    else:
        py.plot([0], 'ko', markeredgewidth=2,markerfacecolor='None' )
    py.ylabel('Y (arcsec)')
    py.xlabel('X (arcsec)')
    py.gca().get_xaxis().set_major_locator(xtickLoc)
    py.axis('image')
    cbar = py.colorbar(orientation='vertical',ticks=[0.,.025,.05])
    cbar.set_label('$\sigma_{h3}$')

    ##########
    # Plot h4
    ##########
    py.subplot(2, 2, 4)
    h4err = e.h4.transpose()
    py.imshow(np.rot90(py.ma.masked_where(h4err==0.,h4err),3),vmin=0.,vmax=.05, 
              extent=[xaxis[0], xaxis[-1], yaxis[0], yaxis[-1]])
    if JK:
        py.plot([0], 'wo', markeredgewidth=2,markerfacecolor='None',markeredgecolor='white' )
    else:
        py.plot([0], 'ko', markeredgewidth=2,markerfacecolor='None' )
    py.xlabel('X (arcsec)')
    py.gca().get_xaxis().set_major_locator(xtickLoc)
    py.axis('image')
    cbar = py.colorbar(orientation='vertical',ticks=[0,.025,.05])
    cbar.set_label('$\sigma_{h4}$')

    #py.tight_layout()
    pdb.set_trace()

    py.savefig(workdir + 'plots/error_maps3.png')
    py.savefig(workdir + 'plots/error_maps3.eps')
    py.show()

def plotErrHist(inputErr=workdir+'/ppxf_errors_mc_nsim100.dat',inputVoronoiFile=datadir+'voronoi_2d_binning_output.txt',plotdir=datadir,incuberoot=cuberoot):

    cubefits = pyfits.open(plotdir + incuberoot + '.fits')
    cube = cubefits[0].data
    # flux errors
    ferrcube = cubefits[1].data
    #ferr = ferrcube.sum(axis=2)
    ferr = np.median(ferrcube,axis=2)
    # convert to mag
    #ferr = -2.5*np.log10(ferrdns)+23.7

    # kinematic errors
    kerrors = PPXFresults(inputErr)
    verr = kerrors.velocity
    serr = kerrors.sigma
    h3err = kerrors.h3
    h4err = kerrors.h4

    # grab only 1 value per Voronoi bin
    yy, xx, binnum = np.loadtxt(inputVoronoiFile,unpack=True)
    xx = xx.astype(int)
    yy = yy.astype(int)
    # only need one spaxel for each bin (they're all the same within a bin)
    u, idx = np.unique(binnum, return_index=True)
    sampleYY = yy[idx]
    sampleXX = xx[idx]

    ferrbin = ferr[sampleXX,sampleYY].flatten()
    verrbin = verr[sampleXX,sampleYY].flatten()
    serrbin = serr[sampleXX,sampleYY].flatten()
    h3errbin = h3err[sampleXX,sampleYY].flatten()
    h4errbin = h4err[sampleXX,sampleYY].flatten()

    py.close(2)
    py.figure(2)
    py.hist(ferrbin,bins=20)
    py.xlabel('Flux errors (DN s$^{-1}$)')
    py.ylabel('Number of tessellated bins')
    py.title('Flux error distribution')

    #pdb.set_trace()
    py.savefig(plotdir+'plots/err_hist_flux.png')
    py.show

    py.clf()
    py.hist(verrbin,bins=20)
    py.xlabel('Velocity errors (km s$^{-1}$)')
    py.ylabel('Number of tessellated bins')
    py.title('MC velocity error distribution')

    pdb.set_trace()
    py.savefig(plotdir+'plots/err_hist_velocity.png')
    py.show

    py.clf()
    py.hist(serrbin,bins=20)
    py.xlabel('Sigma errors (km s$^{-1}$)')
    py.ylabel('Number of tessellated bins')
    py.title('Sigma error distribution')

    #pdb.set_trace()
    py.savefig(plotdir+'plots/err_hist_sigma.png')
    py.show

    print "Median errors"
    print "Flux: ", np.median(ferrbin)
    print "Velocity: ", np.median(verrbin)
    print "Sigma: ", np.median(serrbin)
    print "h3: ", np.median(h3errbin)
    print "h4: ", np.median(h4errbin)
    print "Mean errors"
    print "Flux: ", ferrbin.mean()
    print "Velocity: ", verrbin.mean()
    print "Sigma: ", serrbin.mean()
    print "h3: ", h3errbin.mean()
    print "h4: ", h4errbin.mean()
    
def plotQuality(datadir=datadir,cuberoot=cuberoot,workdir=workdir,saveFC=False,errext=False):
    cubeimg, hdr = pyfits.getdata(datadir + cuberoot + '_img.fits',header=True)
    expimg = pyfits.getdata(datadir + cuberoot + '.fits',ext=3)
    if errext is False:
        snrimg = pyfits.getdata(datadir + cuberoot + '_snr.fits')
    else:
        snrimg = pyfits.getdata(datadir + cuberoot + '_errext_snr.fits')

    cubeimg = cubeimg / (0.05*0.05)
    cubeimg = -2.5*np.log10(cubeimg) + 23.7

    if saveFC:
        pyfits.writeto(datadir+cuberoot+'_img_FC.fits',cubeimg,header=hdr,clobber=True,output_verify='warn')
        
    print cubeimg.shape
    print bhpos
    xaxis = (np.arange(cubeimg.shape[0]) - bhpos_pix[0]) * 0.05
    yaxis = (np.arange(cubeimg.shape[1]) - bhpos_pix[1]) * 0.05
    
    xtickLoc = py.MultipleLocator(1.0)

    #py.figure(2, figsize=(15,8))
    py.figure(2, figsize=(8,14))
    py.subplots_adjust(left=0.05, right=0.94, top=0.95, hspace=0.3)
    py.clf()

    ##########
    # Plot Cube Image
    ##########
    #py.subplot(1, 3, 1)
    py.subplot(3, 1, 1)
    #cubeimg=cubeimg.transpose()
    #cubeimg=np.rot90(cubeimg)
    #py.imshow(np.rot90((py.ma.masked_where(cubeimg<-10000, cubeimg)),3),
    py.imshow(np.rot90(cubeimg,3), 
              extent=[xaxis[0], xaxis[-1], yaxis[0], yaxis[-1]],
              #cmap=py.cm.hot_r,norm=LogNorm(vmin=13.,vmax=9.5))
              cmap=py.cm.hot_r,vmin=11.5,vmax=9.5)
    py.plot([0], 'ko', markeredgewidth=2,markerfacecolor='None')
    py.ylabel('Y (arcsec)')
    py.xlabel('X (arcsec)')
    #py.xticks([-2,-1,0,1])
    py.gca().get_xaxis().set_major_locator(xtickLoc)
    py.axis('image')
    cbar = py.colorbar(orientation='vertical',ticks=[11.5,10.5,9.5])
    cbar.ax.invert_yaxis()
    cbar.set_label('$\Sigma_K$ (mag arcsec$^{-2}$)')

    # Make a compass rose
    #pa = 56.0
    pa = 34.0
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
    #py.title('K Image')


    ##########
    # Plot SNR Image
    ##########
    print snrimg[30,10]
    #py.subplot(1, 3, 2)
    py.subplot(3, 1, 2)
    #snrimg=snrimg.transpose()
    snrimg=np.rot90(snrimg,3) 
    py.imshow(py.ma.masked_invalid(snrimg), 
              extent=[xaxis[0], xaxis[-1], yaxis[0], yaxis[-1]],
              cmap=py.cm.jet)
    py.plot([0], 'ko', markeredgewidth=2,markerfacecolor='None')
    py.xlabel('X (arcsec)')
    py.gca().get_xaxis().set_major_locator(xtickLoc)
    py.axis('image')
    cbar = py.colorbar(orientation='vertical',ticks=[0,40,80])
    cbar.set_label('SNR')
    py.ylabel('Y (arcsec)')
    #py.title('SNR')

    ##########
    # Plot number of exposures
    ##########
    #py.subplot(1, 3, 3)
    py.subplot(3, 1, 3)
    expimg=expimg[:,:,700]
    expimg=expimg.transpose()
    py.imshow(np.rot90(expimg,3),
              extent=[xaxis[0], xaxis[-1], yaxis[0], yaxis[-1]])
    py.plot([0], 'ko', markeredgewidth=2,markerfacecolor='None')
    py.xlabel('X (arcsec)')
    py.gca().get_xaxis().set_major_locator(xtickLoc)
    py.axis('image')
    cbar = py.colorbar(orientation='vertical',ticks=[0,expimg.max()/2,expimg.max()])
    cbar.set_label('Exposures')
    py.ylabel('Y (arcsec)')
    #py.title('Exposures')

    #py.tight_layout()
    pdb.set_trace()
    py.savefig(workdir + 'plots/quality_maps.png')
    py.savefig(workdir + 'plots/quality_maps.eps')
    py.show()


    
def plotErrPDF(inputResults=workdir+'/ppxf_test_mc_nsim100.dat',inputError=workdir+'/ppxf_errors_mc_nsim100.dat'):
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
        ny, bins, patches = py.hist(p.velocity[i,:]+vsys, 50, normed=1, facecolor='green', alpha=0.75)
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
    
def precessionSpeed(inCube=None,inPPXF=None,inMod=None,B01=False):

    #bhpos_pix = bhpos/0.05
    if B01:
        astep = 0.051
    else:
        astep = 0.05
        
    binSize = astep * 3.73
    
    if inMod:
        # read in models
        # manipulate to get in normal orientation
        pres = modelFitResults(inMod)
        cubeimg = np.rot90((pres.nstar).T,3)
        fluxCube = cubeimg / binSize**2
        light = fluxCube[bhpos_pix[1],:]
        velimg = np.rot90((pres.velocity).T,3)
        velocity = velimg[bhpos_pix[1],:]
        sigimg = np.rot90((pres.sigma).T,3)
        dispersion = sigimg[bhpos_pix[1],:]
    else:
        if B01:
            N = 79
            cubeimg = np.zeros((N,N))
            velimg = np.zeros((N,N))
            bflux = pandas.read_csv('/Users/kel/Documents/Projects/M31/HP_pattern_speed/B01_analysis/TIGERTeam/oasis_flux_m8.out',delim_whitespace=True,header=None,names=['y','x','f'])
            bvel = pandas.read_csv('/Users/kel/Documents/Projects/M31/HP_pattern_speed/B01_analysis/TIGERTeam/oasis_vel_m8.out',delim_whitespace=True,header=None,names=['y','x','v'])
            for i in np.arange(len(bflux['f'])):
                tx = int(bflux['x'][i])
                ty = int(bflux['y'][i])
                cubeimg[tx-1,ty-1] = float(bflux['f'][i])
                velimg[tx-1,ty-1] = float(bvel['v'][i])

            idx = np.where(cubeimg > 1000)
            cubeimg[idx] = 0.
            fluxCube = cubeimg / binSize**2
            #bh = [N/2+1,N/2+1]
            bh = [N/2,N/2]
            light = fluxCube[bh[1],:]
            velocity = velimg[bh[1],:]
            #pdb.set_trace() 
        else:
            # read in data
            # manipulate to get in normal orientation
            cubeimg = pyfits.getdata(inCube)
            cubeimg = np.rot90(cubeimg,3)
            fluxCube = cubeimg / binSize**2
            light = fluxCube[bhpos_pix[1],:]
            #snrimg = pyfits.getdata(inSNR)
            pres = PPXFresults(inPPXF)
            velimg = np.rot90((pres.velocity + vsys).T,3)
            velocity = velimg[bhpos_pix[1],:]
            sigimg = np.rot90((pres.sigma).T,3)
            dispersion = sigimg[bhpos_pix[1],:]

    if B01:
        xaxis = (np.arange(N) - bh[0]) * astep
        yaxis = (np.arange(N) - bh[1]) * astep
    else:
        #bhpos = np.array([9.0, 38.0])
        xaxis = (np.arange(cubeimg.shape[1]) - (bhpos_hor[0]/.05)) * astep
        yaxis = (np.arange(cubeimg.shape[0]) - (bhpos_hor[1]/.05)) * astep

    # Pull out a profile along the line-of-nodes (Y-axis)
    
    # Convert Surface Brightness from cts/s to mag/arcsec^2
    #cubeimg += cubeimg.mean() - (cubeimg.std() * 3.0)
    
    #magCube = -2.5 * np.log10(cubeimg) + 23.7
    #magSB = -2.5 * np.log10(fluxCube) + 23.7
    

    py.figure(1)
    py.clf()
    py.subplots_adjust(left=0.13)
    py.plot(xaxis, light)
    limits = py.axis()
    py.plot([0, 0], [limits[2], limits[3]], 'k--')
    py.xlabel('Distance Along Line-of-Nodes (")')
    py.ylabel('Surface Brightness (arbitrary flux/square arcsec)')
    py.xlim(-1.3, 1.3)
    py.savefig(workdir + 'plots/line_of_nodes_flux.png')
    #py.show()
    #pdb.set_trace()

    py.clf()
    py.subplots_adjust(left=0.13)
    py.plot(xaxis, velocity)
    py.xlabel('Distance Along Line-of-Nodes (")')
    py.ylabel('Velocity (km/s)')
    limits = py.axis()
    py.plot([0, 0], [limits[2], limits[3]], 'k--')
    py.plot([limits[0], limits[1]], [0, 0], 'k--')
    py.ylim(-300, 300)
    py.xlim(-1.3, 1.3)
    py.savefig(workdir + 'plots/line_of_nodes_velocity.png')
    #py.show()
    #pdb.set_trace()

    if not B01:
        py.clf()
        py.subplots_adjust(left=0.13)
        py.plot(xaxis, dispersion)
        py.xlabel('Distance Along Line-of-Nodes (")')
        py.ylabel('Velocity Dispersion (km/s)')
        limits = py.axis()
        py.plot([0, 0], [limits[2], limits[3]], 'k--')
        py.plot([limits[0], limits[1]], [0, 0], 'k--')
        py.xlim(-1.3, 1.3)
        py.savefig(workdir + 'plots/line_of_nodes_dispersion.png')
        #py.show()
        #pdb.set_trace()

    # Calculate the Sigma_P * sin(i) (precession speed)
    # Sambhus & Sridhar 2000
    numerator = light * velocity * (astep * 3.73)
    denominator = light * (xaxis * 3.73) * (astep * 3.73)

    tnum = light * velocity
    tden = light * (xaxis * 3.73)
    xc = xaxis * 3.73

    # Sum it all up, but only over a fixed radial extent
    #idx = np.where(np.abs(yaxis) < 1.2)[0]
    #idx = np.where(np.abs(xaxis) < 1.3)[0]
    #idx = np.where(np.abs(xaxis) < 4.)[0]
    idx = np.where(np.abs(xaxis) < 1.975)[0]
    
    sigPsinI = numerator[idx].sum() / denominator[idx].sum()
    tsigPsinI = np.trapz(tnum[idx],xc[idx])/np.trapz(tden[idx],xc[idx])
    
    sigP = sigPsinI / math.sin(math.radians(54.1))
    print 'Precession Speed * sin(i) = %6.2f km/s/pc' % sigPsinI
    print 'Precession speed * sin i, np.trapz = %6.2f km/s/pc' % tsigPsinI
    print 'Precession Speed (i=55 d) = %6.2f km/s/pc' % sigP
    
    # Calculate for several lines
    # Tremaine & Weinberg 1984
    numAll = np.zeros(fluxCube.shape[0], dtype=float)
    denomAll = np.zeros(fluxCube.shape[0], dtype=float)
    sigPsinIall = np.zeros(fluxCube.shape[0], dtype=float)
    if B01:
        xxlo = 13
        xxhi = 65
    else:
        xxlo = 4
        xxhi = 35
    for xx in range(xxlo, xxhi):
        lightTmp = fluxCube[xx,idx]
        velocityTmp = velimg[xx,idx]
        #numAll[xx] = (lightTmp * velocityTmp * (astep * 3.73)).sum() #/ lightTmp.sum()
        numAll[xx] = np.trapz(lightTmp*velocityTmp, xc[idx])
        #denomAll[xx] = (lightTmp * np.abs(xaxis[idx]) * 3.73 * (astep * 3.73)).sum() #/ lightTmp.sum()
        #denomAll[xx] = (lightTmp * xaxis[idx] * 3.73 * (astep * 3.73)).sum() #/ lightTmp.sum()
        #denomAll[xx] = (lightTmp * (xaxis[idx]-.2) * 3.73 * (astep * 3.73)).sum() #/ lightTmp.sum()
        denomAll[xx] = np.trapz(lightTmp * (xaxis[idx] * 3.73), xc[idx])
        sigPsinIall[xx] = numAll[xx] / denomAll[xx]
    print 'sigPsinIall = '
    print sigPsinIall
    sigPall = sigPsinIall / np.sin(np.radians(54.1))
    print 'sigPall (i = 54.1) = '
    print sigPall
    foo = np.where(numAll != 0)[0]
    py.clf()
    py.plot(denomAll[foo]/3.73, numAll[foo], 'k.')
    py.show()

    py.clf()
    #py.plot(yaxis,sigPsinIall,'ko')
    py.plot(yaxis*3.73,sigPsinIall,'ko-')
    #py.xlabel('arcsec')
    py.xlabel('pc')
    py.ylabel('precession speed (km/s/pc)')
    py.title('Precession speed at strips parallel to line of nodes')
    pdb.set_trace()
    
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

    pdb.set_trace()
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
        print 'check cube dimensions before proceeding with plot'
        pdb.set_trace()
        xaxis = (np.arange(84) - bhpos_pix[0]) * 0.05
        yaxis = (np.arange(41) - bhpos_pix[1]) * 0.05
    else:
        # modbhpos given by modelBin
        # PA = -34
        if nonaligned:
            #modbhpos = [(model.velocity.shape[1]-59.035) * 0.05, 63.955 * 0.05]
            modbhpos = [(model.velocity.shape[1]-58.3) * 0.05, 57.5 * 0.05]
        else:
            modbhpos = [(model.velocity.shape[1]-59.035) * 0.05, 54.955 * 0.05] 
        xaxis = (np.arange(model.velocity.shape[1]) - bhpos_pix[0]) * 0.05
        yaxis = (np.arange(model.velocity.shape[0]) - bhpos_pix[1]) * 0.05
    
    xtickLoc = py.MultipleLocator(0.5)

    py.close(2)
    py.figure(2, figsize=(15,7))
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
    py.plot([0], 'ko', markeredgewidth=2,markerfacecolor='None' )
    py.axis('image')
    py.ylabel('Y (arcsec)')
    py.xlabel('X (arcsec)')
    py.gca().get_xaxis().set_major_locator(xtickLoc)
    cbar = py.colorbar(orientation='vertical')
    cbar.set_label('Velocity (km/s)')

    #pdb.set_trace()

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
    py.plot([0], 'ko', markeredgewidth=2,markerfacecolor='None' )
    py.axis('image')
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
    py.plot([0], 'ko', markeredgewidth=2,markerfacecolor='None' )
    py.axis('image')
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
    py.plot([0], 'ko', markeredgewidth=2,markerfacecolor='None' )
    py.axis('image')
    py.xlabel('X (arcsec)')
    py.gca().get_xaxis().set_major_locator(xtickLoc)
    cbar = py.colorbar(orientation='vertical')
    cbar.set_label('h4')

    pdb.set_trace()

    py.tight_layout()

    #py.savefig(modelworkdir + 'plots/model_kinematics.png')
    #py.savefig(modelworkdir + 'plots/model_kinematics.eps')
    py.show()

def plotL98(incubeimg=None):
    # plot OSIRIS and NIRC2 data against the Lauer 1998 data

    # OSIRIS data
    #cubeimg = pyfits.getdata(datadir + cuberoot + '_img.fits')
    cubeimg = pyfits.getdata(incubeimg)
    # ang from top to north (counterclockwise)
    oang = -56.
    # pixel scale
    opix = 0.05
    # BH position
    obh = bhpos/0.05
    # found new BH position via testing
    obhrot = [74.,105.]

    # NIRC2 data - narrow field camera, rotated so N is up
    nirc2 = pyfits.getdata('/Users/kel/Documents/Projects/M31/analysis_old/align/m31_2125nm.fits')
    nang = 0.
    npix = 0.00995
    nbh = [700.,583.]
    # clip to same size as L98 data, with BH at the center
    nrad = 4./npix
    nirc2clip = nirc2[nbh[1]-nrad:nbh[1]+nrad,nbh[0]-nrad:nbh[0]+nrad]
    nbhclip = [nirc2clip.shape[0]/2.,nirc2clip.shape[1]/2.]

    # nonaligned models, PA = -34
    mod = modelFitResults('/Users/kel/Documents/Projects/M31/models/Peiris/2003/nonaligned_grid_rotate/nonaligned_OSIRIScoords_fit_full_smooth_-42.8_54.1_-34.5_-34.0_l98.dat')
    modflux = np.rot90(mod.nstar.T,3)
    modbh = [138.5,modflux.shape[1]-128.]

    # Lauer+ 1998 data
    # 4" on a side, centered on BH
    l98 = pyfits.getdata('/Users/kel/Documents/Projects/M31/data/hst_lauer/1998/L98/data4.fits')
    lang = 55.7
    lpix = 0.0228
    lbh = [l98.shape[0]/2.,l98.shape[1]/2.]

    # rotate to L98 orientation
    cuberot = scipy.ndimage.rotate(cubeimg,oang-lang)
    nirc2rot = scipy.ndimage.rotate(nirc2clip,nang-lang)

    # rebin to L98 pixel scale
    cubecomp = scipy.ndimage.zoom(cuberot,opix/lpix)
    nirc2comp = scipy.ndimage.zoom(nirc2rot,npix/lpix)
    nbhcomp = [nirc2comp.shape[0]/2.,nirc2comp.shape[1]/2.]

    # clip L98 image for plotting with OSIRIS data
    l98oclip = l98[lbh[0]-obhrot[0]:(lbh[0]-obhrot[0])+cubecomp.shape[0],lbh[1]-obhrot[1]:(lbh[1]-obhrot[1])+cubecomp.shape[1]]

    # clip NIRC2 image for plotting with L98 data
    nclip = nirc2comp[nbhcomp[0]-lbh[0]:(nbhcomp[0]-lbh[0])+l98.shape[0],nbhcomp[1]-lbh[1]:(nbhcomp[1]-lbh[1])+l98.shape[1]]

    # clip everything to rad~2" (90 pix = 2.052" at 0.0228"/pix)
    rado = 74.
    cubepl = cubecomp[obhrot[0]-rado:obhrot[0]+rado,obhrot[1]-rado:obhrot[1]+rado]
    l98oplorg = l98oclip[obhrot[0]-rado:obhrot[0]+rado,obhrot[1]-rado:obhrot[1]+rado]
    npl = nclip[lbh[0]-rado:lbh[0]+rado,lbh[1]-rado:lbh[1]+rado]
    l98pl = l98[lbh[0]-rado:lbh[0]+rado,lbh[1]-rado:lbh[1]+rado]
    modpl = modflux[modbh[0]-rado:modbh[0]+rado,modbh[1]-rado:modbh[1]+rado]

    # convolve L98 data to OSIRIS resolution
    PSFparams = readPSFparams('/Users/kel/Documents/Projects/M31/data/osiris_mosaics/drf/sigclip/all_NIRC2_CC_DTOTOFF_2/no100828/data/osiris_perf/sig_1.05/params.txt',twoGauss=False)
    PSFsig = PSFparams.sig1[0] * (opix/lpix)
    psf = ifu.gauss_kernel(PSFsig, PSFparams.amp1[0],half_box=PSFsig*5.)
    l98opl = signal.convolve(l98oplorg, psf, mode='same')
    
    #pdb.set_trace()
    # plot OSIRIS comparison
    xaxis = np.arange(cubepl.shape[0]) * 0.0228 - (rado*.0228)
    yaxis = np.arange(cubepl.shape[1]) * 0.0228 - (rado*.0228)
    
    py.close(2)
    py.figure(2, figsize=(5,14))
    py.subplots_adjust(left=0.01, right=0.88, top=0.95, hspace=0.3)
    py.clf()
    py.subplot(3,1,1)
    py.ylabel('Y (arcsec)')
    py.rc('axes', labelsize=15)
    py.rc('xtick', labelsize=15)    
    py.rc('ytick', labelsize=15)
    py.rc('legend', fontsize=15)
    py.imshow(np.ma.masked_where((cubepl/cubepl.max())<0.05,cubepl/cubepl.max()),vmin=0.,vmax=1.,extent=[xaxis[0], xaxis[-1], yaxis[0], yaxis[-1]])
    py.autoscale(False)
    py.xticks([-1.5,0,1.5])
    py.yticks([-1.5,0,1.5])
    py.plot([0], [0], 'ko', markeredgewidth=1,markerfacecolor='None')
    
    cbar = py.colorbar(orientation='vertical',ticks=[0.,.5,1.])
    cbar.set_label('OSIRIS flux, norm')

    # Make a compass rose
    #pa = 56.0
    pa = lang
    cosSin = np.array([ math.cos(math.radians(pa)), 
                        math.sin(math.radians(pa)) ])
    arr_base = np.array([ xaxis[-1]-0.3, yaxis[-1]-0.7 ])
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

    py.subplot(3,1,2)
    py.imshow(l98opl/l98opl.max(),vmin=0.,vmax=1.,extent=[xaxis[0], xaxis[-1], yaxis[0], yaxis[-1]])
    py.autoscale(False)
    py.xticks([-1.5,0,1.5])
    py.yticks([-1.5,0,1.5])
    py.ylabel('Y (arcsec)')
    py.plot([0],[0], 'ko', markeredgewidth=1,markerfacecolor='None')
    cbar = py.colorbar(orientation='vertical',ticks=[0.,.5,1.])
    cbar.set_label('L98 flux, norm')

    py.subplot(3,1,3)
    py.imshow(np.ma.masked_where((cubepl/cubepl.max())<0.05,cubepl/cubepl.max())-np.ma.masked_where((cubepl/cubepl.max())<0.05,l98opl/l98opl.max()),vmin=-.25,vmax=.25,extent=[xaxis[0], xaxis[-1], yaxis[0], yaxis[-1]])
    py.autoscale(False)
    py.xticks([-1.5,0,1.5])
    py.yticks([-1.5,0,1.5])
    py.ylabel('Y (arcsec)')
    py.xlabel('X (arcsec)')
    py.plot([0], [0], 'ko', markeredgewidth=1,markerfacecolor='None')
    cbar = py.colorbar(orientation='vertical',ticks=[-.25,0.,.25])
    cbar.set_label('Residuals')

    pdb.set_trace()
    py.savefig(workdir + 'plots/L98_vs_OSIRIS.png')
    
    # plot NIRC2 comparison
    
    py.clf()
    py.subplot(3,1,1)
    py.imshow(npl/npl.max(),vmin=0.,vmax=1.,extent=[xaxis[0], xaxis[-1], yaxis[0], yaxis[-1]])
    py.autoscale(False)
    py.xticks([-1.5,0,1.5])
    py.yticks([-1.5,0,1.5])
    py.plot([0],[0], 'ko', markeredgewidth=2,markerfacecolor='None')
    cbar = py.colorbar(orientation='vertical')
    cbar.set_label('NIRC2 flux, norm')

    py.subplot(3,1,2)
    py.imshow(l98pl/l98pl.max(),vmin=0.,vmax=1.,extent=[xaxis[0], xaxis[-1], yaxis[0], yaxis[-1]])
    py.autoscale(False)
    py.xticks([-1.5,0,1.5])
    py.yticks([-1.5,0,1.5])
    py.plot([0],[0], 'ko', markeredgewidth=2,markerfacecolor='None')
    cbar = py.colorbar(orientation='vertical')
    cbar.set_label('L98 flux, norm')

    py.subplot(3,1,3)
    py.imshow((npl/npl.max())-l98pl/l98pl.max(),vmin=-.3,vmax=.3,extent=[xaxis[0], xaxis[-1], yaxis[0], yaxis[-1]])
    py.autoscale(False)
    py.xticks([-1.5,0,1.5])
    py.yticks([-1.5,0,1.5])
    py.plot([0],[0], 'ko', markeredgewidth=2,markerfacecolor='None')
    cbar = py.colorbar(orientation='vertical')
    cbar.set_label('Residuals (NIRC2 - L98)')

    #pdb.set_trace()
    py.savefig(workdir + 'plots/L98_vs_NIRC2.png')

    py.close(3)
    py.figure(3, figsize=(13,3))
    py.subplots_adjust(left=0.05, right=0.92, top=0.93,wspace=.25,bottom=.21)
    # plot comparison with original nonaligned models
    py.clf()
    py.subplot(1,3,1)
    py.imshow(l98pl/l98pl.max(),vmin=0.,vmax=1.,extent=[xaxis[0], xaxis[-1], yaxis[0], yaxis[-1]])
    py.autoscale(False)
    py.ylabel('Y (arcsec)')
    py.xlabel('X (arcsec)')
    py.xticks([-1.5,0,1.5])
    py.yticks([-1.5,0,1.5])
    py.plot([0],[0], 'ko', markeredgewidth=2,markerfacecolor='None')
    cbar = py.colorbar(orientation='vertical',ticks=[0,.5,1])
    cbar.set_label('L98 flux, norm')

    # Make a compass rose
    #pa = 56.0
    pa = lang
    cosSin = np.array([ math.cos(math.radians(pa)), 
                        math.sin(math.radians(pa)) ])
    arr_base = np.array([ xaxis[-1]-0.3, yaxis[-1]-0.7 ])
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

    py.subplot(1,3,2)
    modx = (np.arange(modflux.shape[0]) - modbh[1])*0.0228 
    mody = (np.arange(modflux.shape[1]) - modbh[0])*0.0228
    py.imshow(modpl/modpl.max(),vmin=0.,vmax=1.,extent=[xaxis[0], xaxis[-1], yaxis[0], yaxis[-1]])
    py.plot([0],[0], 'ko', markeredgewidth=2,markerfacecolor='None')
    py.axis('image')
    #py.ylabel('Y (arcsec)')
    py.xlabel('X (arcsec)')
    py.xticks([-1.5,0,1.5])
    py.yticks([-1.5,0,1.5])
    cbar = py.colorbar(orientation='vertical',ticks=[0,.5,1])
    cbar.set_label('Model nstar, norm')

    py.subplot(1,3,3)
    py.imshow((l98pl/l98pl.max()) - (modpl/modpl.max()), vmin=-.25,vmax=.25,extent=[xaxis[0], xaxis[-1], yaxis[0], yaxis[-1]])
    py.plot([0],[0], 'ko', markeredgewidth=2,markerfacecolor='None')
    py.axis('image')
    py.xticks([-1.5,0,1.5])
    py.yticks([-1.5,0,1.5])
    #py.ylabel('Y (arcsec)')
    py.xlabel('X (arcsec)')
    cbar = py.colorbar(orientation='vertical',ticks=[-.25,0,.25])
    cbar.set_label('Residuals')
    #pdb.set_trace()
    

def plotSTIS(inData=workdir+'ppxf_tess_bs_2_20160825.dat',inData2=None,inErr=None,inErr2=None,inModel2=None,l98=False,leg=None, indvPlot=False, hAlpha=False, B01=False):
    # setting only inData compares the OSIRIS kinematic fits to the STIS measurements
    # setting both inData and inData2 compares the two different OSIRIS kinematic fits to each other (no STIS comparison),
    #    but along the STIS PA still; assumes cube size and SMBH position are the same as in inData
    
    ppxf = PPXFresults(inData)
    #ppxf = modelFitResults(inData)
    if inData2 is not None:
        ppxf2 = PPXFresults(inData2)
    if inModel2 is not None:
        ppxf2 = modelFitResults(inModel2)
    if inErr is not None:
        err = PPXFresults(inErr)
    if inErr2 is not None:
        err2 = PPXFresults(inErr2)

    # OSIRIS data - just used for plotting
    #cubeimg = pyfits.getdata(datadir + cuberoot + '_img.fits')
    #cubeimg = pyfits.getdata('/Users/kel/Documents/Projects/M31/data/osiris_mosaics/drf/sigclip/all_telshift/woscalecont/m31_mosaic_telshift_woscalecont_img.fits')
    cubeimg = pyfits.getdata('/Users/kel/Documents/Projects/M31/data/osiris_mosaics/drf/sigclip/all_comb3/mosaic_all_comb3_img.fits')

    # reorient data
    vel = np.rot90(ppxf.velocity.T,3) + vsys
    sig = np.rot90(ppxf.sigma.T,3)
    h3 = np.rot90(ppxf.h3.T,3)
    h4 = np.rot90(ppxf.h4.T,3)
    
    chi2 = np.rot90(ppxf.chi2red.T,3)

    cubeshape = vel.shape
    
    if ((inData2 is not None) or (inModel2 is not None)):
        if inData2 is not None:
            vel_2 = np.rot90(ppxf2.velocity.T,3) + vsys
        if inModel2 is not None:
            vel_2 = np.rot90(ppxf2.velocity.T,3)
        sig_2 = np.rot90(ppxf2.sigma.T,3)
        h3_2 = np.rot90(ppxf2.h3.T,3)
        h4_2 = np.rot90(ppxf2.h4.T,3)
    else:
        inSTIS = '/Users/kel/Documents/Projects/M31/data/hst/STIS/stisFCQ.dat'
        stis = Table.read(inSTIS,format='ascii')
        # column numbers/names are from Hiranya's script.pro
        stis.rename_column('col1','r')
        stis.rename_column('col2','vel')
        stis.rename_column('col3','dv')
        stis.rename_column('col4','sig')
        stis.rename_column('col5','ds')
        stis.rename_column('col8','h4')
        stis.rename_column('col9','dh4')
        stis.rename_column('col10','h3')
        stis.rename_column('col11','dh3')

        # constants are also from Hiranya's script
        stis['r'] += 0.02
        stis['r'] *= -1.
        stis['vel'] += 340.

    if hAlpha:
        inHAlpha = pandas.read_csv('/Users/kel/Documents/Projects/M31/data/menezes2013/fig3_scrape.txt',delim_whitespace=True,header=None,names=['r','vel'])
    if B01:
        #inB01_v = pandas.read_csv('/Users/kel/Documents/Projects/M31/data/bacon2001/fig13_scrape_velocity.txt',delim_whitespace=True,header=None,names=['r','vel'])
        inB01_v = pandas.read_csv('/Users/kel/Documents/Projects/M31/data/bacon2001/fig10_scrape_velocity_bs.txt',delim_whitespace=True,header=None,names=['r','vel'])
        inB01_v['r'] *= -1.
        #inB01_s = pandas.read_csv('/Users/kel/Documents/Projects/M31/data/bacon2001/fig13_scrape_sigma..txt',delim_whitespace=True,header=None,names=['r','sig'])
        inB01_s = pandas.read_csv('/Users/kel/Documents/Projects/M31/data/bacon2001/fig10_scrape_sigma_bs.txt',delim_whitespace=True,header=None,names=['r','sig'])
        inB01_s['r'] *= -1.

    if inErr is not None:
        velErr = np.rot90(err.velocity.T,3)
        sigErr = np.rot90(err.sigma.T,3)
        h3Err = np.rot90(err.h3.T,3)
        h4Err = np.rot90(err.h4.T,3)
    if inErr2 is not None:
        velErr2 = np.rot90(err2.velocity.T,3)
        sigErr2 = np.rot90(err2.sigma.T,3)
        h3Err2 = np.rot90(err2.h3.T,3)
        h4Err2 = np.rot90(err2.h4.T,3)

    # STIS PA = angle between N and long axis of the STIS slit
    spa = 39.
    # rotate the slit a bit
    #spa=60.
    # OSIRIS PA = angle between top (along long end of the OSIRIS cube) and N
    opa = 34.
    # slit PA = angle between top (along long end of the cube) and slit
    slitpa = np.radians(spa + opa)
    cpa = np.radians(90. - spa - opa)
    if hAlpha:
        # PA from Menezes 2013
        hpa = 59.3
        hslitpa = np.radians(hpa + opa)
        chpa = np.radians(90. - hpa - opa)

    # using trig arguments, can get the endpoints of the slit at the edge of the FOV in terms of OSIRIS coordinates
    x1 = 0.
    y1 = (bhpos_pix[0]*np.tan(cpa)) + bhpos_pix[1]
    x2 = cubeshape[1]-1.
    y2 = bhpos_pix[1] - ((x2 - bhpos_pix[0])*np.tan(cpa))
    # shift the slit up and down a bit
    #y1 += 1
    #y2 += 1
    #x1 -=1
    #x2 -=1

    # get the high and low ends of the slit - in this case, the slit is 0.1" wide, so taking +/- 0.05" = 1 pixel
    dw = 1.
    dx = dw*np.cos(cpa)
    dy = dw*np.sin(cpa)
    # get the high (u) and low (d) endpoints of the slit on both the left and the right ends of the cube
    x1u = x1+dx
    y1u = y1+dy
    x1d = x1-dx
    y1d = y1-dy
    x2u = x2+dx
    y2u = y2+dy
    x2d = x2-dx
    y2d = y2-dy
    
    # plotting the slit on the flux/velocity map
    py.close(1)
    py.figure(1)
    py.clf()
    py.imshow(py.ma.masked_where( np.rot90(ppxf.velocity.T,3)==0,vel),vmin=-250.,vmax=250.)
    py.title('OSIRIS velocity map w/ STIS slit marked')
    #py.imshow(np.rot90(cubeimg,3))
    #py.title('OSIRIS flux map w/ STIS slit PA marked')
    #py.imshow(py.ma.masked_where( np.rot90(ppxf.sigma.T,3)==0,sig),vmin=0.,vmax=300.)
    #py.title('OSIRIS dispersion map s/ STIS slit PA marked')
    py.plot([x1,x2],[y1,y2],'k-')
    py.plot([x1u,x2u],[y1u,y2u],'k--')
    py.plot([x1d,x2d],[y1d,y2d],'k--')
    py.plot([bhpos_pix[0]],[bhpos_pix[1]],'ko', markeredgewidth=2,markerfacecolor='None')
    py.axis('image')

    #pdb.set_trace()

    if l98:
        # plot the STIS slit on the L98 data
        l98 = pyfits.getdata('/Users/kel/Documents/Projects/M31/data/hst_lauer/1998/L98/data4.fits')
        # angle between top and N
        lang = 55.7
        lpix = 0.0228
        lbh = [l98.shape[0]/2.,l98.shape[1]/2.]
        # cpaL98 ends up being >90, so adjust
        cpaL98 = np.radians(lang+spa-90.)
        x1_l98 = 0.
        # multiply by -1 to make up for the >90
        y1_l98 = (-1.*lbh[0]*np.tan(cpaL98)) + lbh[1]
        x2_l98 = l98.shape[1]-1.
        # again, multiply by -1 to make up for the >90 angle
        y2_l98 = lbh[1] + ((x2_l98 - lbh[0])*np.tan(cpaL98))

        py.close(3)
        py.figure(3)
        py.clf()
        py.imshow(l98)
        py.title('L98 flux map w/ STIS slit PA marked')
        py.plot([x1_l98,x2_l98],[y1_l98,y2_l98],'k-')
        py.plot([lbh[0]],[lbh[1]],'ko', markeredgewidth=2,markerfacecolor='None')
        py.axis('image')

        pdb.set_trace()

    # interpolate data so I can take a cut across at a weird angle
    XX = np.arange(cubeshape[1])
    YY = np.arange(cubeshape[0])
    #vint = scipy.interpolate.interp2d(XX,YY,vel)
    #sint = scipy.interpolate.interp2d(XX,YY,sig)
    #h3int = scipy.interpolate.interp2d(XX,YY,h3)
    #h4int = scipy.interpolate.interp2d(XX,YY,h4)
    vint = vel
    sint = sig
    h3int = h3
    h4int = h4
    if ((inData2 is not None) or (inModel2 is not None)):
        #vint_2 = scipy.interpolate.interp2d(XX,YY,vel_2)
        #sint_2 = scipy.interpolate.interp2d(XX,YY,sig_2)
        #h3int_2 = scipy.interpolate.interp2d(XX,YY,h3_2)
        #h4int_2 = scipy.interpolate.interp2d(XX,YY,h4_2)
        vint_2 = vel_2
        sint_2 = sig_2
        h3int_2 = h3_2
        h4int_2 = h4_2
    if inErr is not None:
        #verrint = scipy.interpolate.interp2d(XX,YY,velErr)
        #serrint = scipy.interpolate.interp2d(XX,YY,sigErr)
        #h3errint = scipy.interpolate.interp2d(XX,YY,h3Err)
        #h4errint = scipy.interpolate.interp2d(XX,YY,h4Err)
        verrint = velErr
        serrint = sigErr
        h3errint = h3Err
        h4errint = h4Err
    if inErr2 is not None:
        #verrint2 = scipy.interpolate.interp2d(XX,YY,velErr2)
        #serrint2 = scipy.interpolate.interp2d(XX,YY,sigErr2)
        #h3errint2 = scipy.interpolate.interp2d(XX,YY,h3Err2)
        #h4errint2 = scipy.interpolate.interp2d(XX,YY,h4Err2)
        verrint2 = velErr2
        serrint2 = sigErr2
        h3errint2 = h3Err2
        h4errint2 = h4Err2

    # this samples them roughly once per pixel
    npoints = 87
    xvals = np.linspace(x1, x2, npoints)
    xvalsu = np.linspace(x1u, x2u, npoints)
    xvalsd = np.linspace(x1d, x2d, npoints)
    yvals = np.linspace(y1, y2, npoints)
    yvalsu = np.linspace(y1u, y2u, npoints)
    yvalsd = np.linspace(y1d, y2d, npoints)

    if hAlpha:
        x1h = 0.
        x2h = (bhpos_pix[0]*np.tan(chpa)) + bhpos_pix[1]
        y1h = cubeshape[1]-1.
        y2h = bhpos_pix[1] - ((x2h - bhpos_pix[0])*np.tan(chpa))

    #pdb.set_trace()

    vslit = []
    vsd = []
    sslit = []
    ssd = []
    h3slit = []
    h3sd = []
    h4slit = []
    h4sd = []
    chi2slit = []
    for i in np.arange(npoints):
        # these should techincally be flux-weighted averages, but the slit is pretty
        # narrow and there aren't big flux changes from spaxel to spaxel, so probably
        # close enough
        # make sure we're not out of the FOV
        if ((yvalsu[i] <= vint.shape[0]) and (xvalsu[i] <= vint.shape[1])):
            v = vint[yvals[i], xvals[i]]
            vu = vint[yvalsu[i], xvalsu[i]]
            vd = vint[yvalsd[i], xvalsd[i]]
            s = sint[yvals[i], xvals[i]]
            su = sint[yvalsu[i], xvalsu[i]]
            sd = sint[yvalsd[i], xvalsd[i]]
            h3tmp = h3int[yvals[i], xvals[i]]
            h3tmpu = h3int[yvalsu[i], xvalsu[i]]
            h3tmpd = h3int[yvalsd[i], xvalsd[i]]
            h4tmp = h4int[yvals[i], xvals[i]]
            h4tmpu = h4int[yvalsu[i], xvalsu[i]]
            h4tmpd = h4int[yvalsd[i], xvalsd[i]]
            chi2tmp = chi2[yvals[i], xvals[i]]
        else:
            # if we're out of the FOV, set v = vsys, which is equivalent to it being in a hole (and will be caught by the next if statement)
            # everything else set to 0, for the errors
            v = vsys
            vu = vsys
            vd = vsys
            s = 0.
            su = 0.
            sd = 0.
            h3tmp = 0.
            h3tmpu = 0.
            h3tmpd = 0.
            h4tmp = 0.
            h4tmpu = 0.
            h4tmpd = 0.
            chi2tmp = 0.
            
        # check if we're in a hole in the tessellation (v == 0 before adding vsys, so v==vsys in this frame)
        if ((v == vsys) or (vu == vsys) or (vd == vsys)):
            # if any of the slit is in a hole, set everything to a placeholder
            vslit.append(-500.)
            sslit.append(-500.)
            h3slit.append(-500.)
            h4slit.append(-500.)
            chi2slit.append(-500.)
        else:
            # otherwise continue as usual
            vslit.append((v+vu+vd)/3.)
            sslit.append((s+su+sd)/3.)
            h3slit.append((h3tmp+h3tmpu+h3tmpd)/3.)
            h4slit.append((h4tmp+h4tmpu+h4tmpd)/3.)

            chi2slit.append(chi2tmp)
            
        if ((inErr is not None) and (yvalsu[i] <= vint.shape[0]) and (xvalsu[i] <= vint.shape[1])):
            verrtmp = verrint[yvals[i],xvals[i]]
            verrutmp = verrint[yvalsu[i],xvalsu[i]]
            verrdtmp = verrint[yvalsd[i],xvalsd[i]]
            serrtmp = serrint[yvals[i],xvals[i]]
            serrutmp = serrint[yvalsu[i],xvalsu[i]]
            serrdtmp = serrint[yvalsd[i],xvalsd[i]]
            h3errtmp = h3errint[yvals[i],xvals[i]]
            h3errutmp = h3errint[yvalsu[i],xvalsu[i]]
            h3errdtmp = h3errint[yvalsd[i],xvalsd[i]]
            h4errtmp = h4errint[yvals[i],xvals[i]]
            h4errutmp = h4errint[yvalsu[i],xvalsu[i]]
            h4errdtmp = h4errint[yvalsd[i],xvalsd[i]]
            vsd.append(np.sqrt(verrtmp**2+verrutmp**2+verrdtmp**2))
            ssd.append(np.sqrt(serrtmp**2+serrutmp**2+serrdtmp**2))
            h3sd.append(np.sqrt(h3errtmp**2+h3errutmp**2+h3errdtmp**2))
            h4sd.append(np.sqrt(h4errtmp**2+h4errutmp**2+h4errdtmp**2))
        else:
            vsd.append(np.std([v,vu,vd]))
            ssd.append(np.std([s,su,sd]))
            h3sd.append(np.std([h3tmp,h3tmpu,h3tmpd]))
            h4sd.append(np.std([h4tmp,h4tmpu,h4tmpd]))
    
    vslit = np.array(vslit)
    sslit = np.array(sslit)
    h3slit = np.array(h3slit)
    h4slit = np.array(h4slit)
    vsd = np.array(vsd)
    ssd = np.array(ssd)
    h3sd = np.array(h3sd)
    h4sd = np.array(h4sd)
    chi2slit = np.array(chi2slit)
    
    vpl = np.ma.masked_where(vslit==-500,vslit)
    spl = np.ma.masked_where(sslit==-500,sslit)
    h3pl = np.ma.masked_where(h3slit==-500,h3slit)
    h4pl = np.ma.masked_where(h4slit==-500,h4slit)
    chi2pl = np.ma.masked_where(chi2slit==-500,chi2slit)
    
    # radius vector for plotting the OSIRIS cut, in arcsec
    ras = np.arange(npoints)*0.05 - (bhpos_pix[0]/np.cos(cpa))*0.05
    rasshift = np.arange(npoints)*0.05 - (44./np.cos(cpa))*0.05
        
    if ((inData2 is not None) or (inModel2 is not None)):
        vslit_2 = []
        vsd_2 = []
        sslit_2 = []
        ssd_2 = []
        h3slit_2 = []
        h3sd_2 = []
        h4slit_2 = []
        h4sd_2 = []
        for i in np.arange(npoints):
            # these should techincally be flux-weighted averages, but the slit is pretty
            # narrow and there aren't big flux changes from spaxel to spaxel, so probably
            # close enough
            # make sure we're not out of the FOV
            if ((yvalsu[i] <= vint.shape[0]) and (xvalsu[i] <= vint.shape[1])):
                v_2 = vint_2[yvals[i], xvals[i]]
                vu_2 = vint_2[yvalsu[i], xvalsu[i]]
                vd_2 = vint_2[yvalsd[i], xvalsd[i]]
                s_2 = sint_2[yvals[i], xvals[i]]
                su_2 = sint_2[yvalsu[i], xvalsu[i]]
                sd_2 = sint_2[yvalsd[i], xvalsd[i]]
                h3tmp_2 = h3int_2[yvals[i], xvals[i]]
                h3tmpu_2 = h3int_2[yvalsu[i], xvalsu[i]]
                h3tmpd_2 = h3int_2[yvalsd[i], xvalsd[i]]
                h4tmp_2 = h4int_2[yvals[i], xvals[i]]
                h4tmpu_2 = h4int_2[yvalsu[i], xvalsu[i]]
                h4tmpd_2 = h4int_2[yvalsd[i], xvalsd[i]]
            else:
                v_2 = vsys
                vu_2 = vsys
                vd_2 = vsys
                s_2 = 0.
                su_2 = 0.
                sd_2 = 0.
                h3tmp_2 = 0.
                h3tmpu_2 = 0.
                h3tmpd_2 = 0.
                h4tmp_2 = 0.
                h4tmpu_2 = 0.
                h4tmpd_2 = 0.
                
            if ((v_2 == vsys) or (vu_2 == vsys) or (vd_2 == vsys)):
                vslit_2.append(-500.)
                sslit_2.append(-500.)
                h3slit_2.append(-500.)
                h4slit_2.append(-500.)
            else:
                vslit_2.append((v_2+vu_2+vd_2)/3.)
                sslit_2.append((s_2+su_2+sd_2)/3.)
                h3slit_2.append((h3tmp_2+h3tmpu_2+h3tmpd_2)/3.)
                h4slit_2.append((h4tmp_2+h4tmpu_2+h4tmpd_2)/3.)

            if inErr2 is not None:
                verrtmp = verrint2[yvals[i],xvals[i]]
                verrutmp = verrint2[yvalsu[i],xvalsu[i]]
                verrdtmp = verrint2[yvalsd[i],xvalsd[i]]
                serrtmp = serrint2[yvals[i],xvals[i]]
                serrutmp = serrint2[yvalsu[i],xvalsu[i]]
                serrdtmp = serrint2[yvalsd[i],xvalsd[i]]
                h3errtmp = h3errint2[yvals[i],xvals[i]]
                h3errutmp = h3errint2[yvalsu[i],xvalsu[i]]
                h3errdtmp = h3errint2[yvalsd[i],xvalsd[i]]
                h4errtmp = h4errint2[yvals[i],xvals[i]]
                h4errutmp = h4errint2[yvalsu[i],xvalsu[i]]
                h4errdtmp = h4errint2[yvalsd[i],xvalsd[i]]
                vsd_2.append(np.sqrt(verrtmp**2+verrutmp**2+verrdtmp**2))
                ssd_2.append(np.sqrt(serrtmp**2+serrutmp**2+serrdtmp**2))
                h3sd_2.append(np.sqrt(h3errtmp**2+h3errutmp**2+h3errdtmp**2))
                h4sd_2.append(np.sqrt(h4errtmp**2+h4errutmp**2+h4errdtmp**2))
            else:
                vsd_2.append(np.std([v_2,vu_2,vd_2]))
                ssd_2.append(np.std([s_2,su_2,sd_2]))
                h3sd_2.append(np.std([h3tmp_2,h3tmpu_2,h3tmpd_2]))
                h4sd_2.append(np.std([h4tmp_2,h4tmpu_2,h4tmpd_2]))
    
        vslit_2 = np.array(vslit_2)
        sslit_2 = np.array(sslit_2)
        h3slit_2 = np.array(h3slit_2)
        h4slit_2 = np.array(h4slit_2)
        vsd_2 = np.array(vsd_2)
        ssd_2 = np.array(ssd_2)
        h3sd_2 = np.array(h3sd_2)
        h4sd_2 = np.array(h4sd_2)

        vpl2 = np.ma.masked_where(vslit_2==-500,vslit_2)
        spl2 = np.ma.masked_where(sslit_2==-500,sslit_2)
        h3pl2 = np.ma.masked_where(h3slit_2==-500,h3slit_2)
        h4pl2 = np.ma.masked_where(h4slit_2==-500,h4slit_2)
        
    else:
        # interpolate and smooth the STIS data to the same grid and with the same
        # instrumental resolution as the OSIRIS data
        # first get the PSF
        #PSFparams = readPSFparams(inputFile=workdir+'plots/osir_perf_m31_all_scalederr_cleanhdr_params.txt',twoGauss=False)
        #PSFparams = readPSFparams(inputFile='/Users/kel/Documents/Projects/M31/data/osiris_mosaics/drf/sigclip/numclip_5/data/osiris_perf/twogauss/osir_perf_s081021_a019001__mosaic_Kbb_050_img_params.txt',twoGauss=True)
        PSFparams = readPSFparams(inputFile='/Users/kel/Documents/Projects/M31/data/osiris_mosaics/drf/sigclip/all_NIRC2_CC_DTOTOFF_2/no100828/data/osiris_perf/sig_1.05/params.txt',twoGauss=False)
        # grab sigma of the PSF in units of pixels and convert to arcsec
        PSFsig = PSFparams.sig1[0]#*0.05
        gauss = ifu.gauss_kernel1D(PSFsig,PSFparams.amp1[0],half_box=PSFsig*5.)
    
        # interpolation: first try linear
        v1int = interp1d(stis['r'],stis['vel'])
        v1 = np.convolve(v1int.__call__(ras),gauss,mode='same')
        s1int = interp1d(stis['r'],stis['sig'])
        s1 = np.convolve(s1int.__call__(ras),gauss,mode='same')
        h31int = interp1d(stis['r'],stis['h3'])
        h31 = np.convolve(h31int.__call__(ras),gauss,mode='same')
        h41int = interp1d(stis['r'],stis['h4'])
        h41 = np.convolve(h41int.__call__(ras),gauss,mode='same')

        # also try a cubic spline interpolation
        v2int = interp1d(stis['r'],stis['vel'],kind='cubic')
        v2 = np.convolve(v2int.__call__(ras),gauss,mode='same')
        s2int = interp1d(stis['r'],stis['sig'],kind='cubic')
        s2 = np.convolve(s2int.__call__(ras),gauss,mode='same')
        h32int = interp1d(stis['r'],stis['h3'],kind='cubic')
        h32 = np.convolve(h32int.__call__(ras),gauss,mode='same')
        h42int = interp1d(stis['r'],stis['h4'],kind='cubic')
        h42 = np.convolve(h42int.__call__(ras),gauss,mode='same')

    if indvPlot:
        py.figure(2)
        py.clf()
        py.plot(ras,vpl,'ro')
        py.errorbar(ras,vpl,yerr=vsd,ecolor='k',fmt='none')
        #py.plot(rasshift,vslit,'go')
        if ((inData2 is None) and (inModel2 is None)):
            py.plot(stis['r'],stis['vel'],'ko')
            py.errorbar(stis['r'],stis['vel'],yerr=stis['dv'],ecolor='k',fmt='none')
            #py.plot(ras,v1,'k.')
            py.plot(ras,v2,'k-')
        else:
            py.plot(ras,vpl2,'ko')
            if inErr2 is not None:
                py.errorbar(ras,vpl2,yerr=vsd_2,ecolor='k',fmt='none')
        py.plot([-2,2],[0,0],'k--')
        py.plot([0,0],[-300,300],'k--')
        py.xlim(-2,2)
        py.ylim(-300,300)
        py.xlabel('r (arcsec)')
        py.ylabel('velocity (km s$^{-1}$)')

        #py.savefig(workdir+'plots/osiris_vs_stis_vel_340-1.png')
        #py.savefig(workdir+'plots/osiris_vs_stis_vel_blue.png')
    
        #pdb.set_trace()

        py.clf()
        py.plot(ras,spl,'ro-')
        py.errorbar(ras,spl,yerr=ssd,ecolor='k',fmt='none')
        #py.plot(rasshift,sslit,'go-')
        if ((inData2 is None) and (inModel2 is None)):
            py.plot(stis['r'],stis['sig'],'ko')
            py.errorbar(stis['r'],stis['sig'],yerr=stis['ds'],ecolor='k',fmt='none')
            #py.plot(ras,s1,'k.')
            py.plot(ras,s2,'k-')
        else:
            py.plot(ras,spl2,'ko')
            if inErr2 is not None:
                py.errorbar(ras,spl2,yerr=ssd_2,ecolor='k',fmt='none')
        py.plot([-2,2],[0,0],'k--')
        py.plot([0,0],[0,350],'k--')
        py.xlim(-2,2)
        py.ylim(0,350)
        py.xlabel('r (arcsec)')
        py.ylabel('$\sigma$ (km s$^{-1}$)')

        #py.savefig(workdir+'plots/osiris_vs_stis_sig_blue.png')
        
        #pdb.set_trace()

        py.clf()
        py.plot(ras,h3pl,'ro-')
        py.errorbar(ras,h3pl,yerr=h3sd,ecolor='k',fmt='none')
        #py.plot(rasshift,h3slit,'go-')
        if ((inData2 is None) and (inModel2 is None)):
            py.plot(stis['r'],stis['h3'],'ko')
            py.errorbar(stis['r'],stis['h3'],yerr=stis['dh3'],ecolor='k',fmt='none')
            #py.plot(ras,h31,'k.')
            py.plot(ras,h32,'k-')
        else:
            py.plot(ras,h3pl2,'ko')
            if inErr2 is not None:
                py.errorbar(ras,h3pl2,yerr=h3sd_2,ecolor='k',fmt='none')
        py.plot([-2,2],[0,0],'k--')
        py.plot([0,0],[-0.3,0.3],'k--')
        py.xlim(-2,2)
        py.ylim(-0.3,0.3)
        py.xlabel('r (arcsec)')
        py.ylabel('h3')

        #py.savefig(workdir+'plots/osiris_vs_stis_h3_blue.png')
    
        #pdb.set_trace()
    
        py.clf()
        py.plot(ras,h4pl,'ro-')
        py.errorbar(ras,h4pl,yerr=h4sd,ecolor='k',fmt='none')
        #py.plot(rasshift,h4slit,'go-')
        if ((inData2 is None) and (inModel2 is None)):
            py.plot(stis['r'],stis['h4'],'ko')
            py.errorbar(stis['r'],stis['h4'],yerr=stis['dh4'],ecolor='k',fmt='none')
            #py.plot(ras,h41,'k.')
            py.plot(ras,h42,'k-')
        else:
            py.plot(ras,h4pl2,'ko')
            if inErr2 is not None:
                py.errorbar(ras,h4pl2,yerr=h4sd_2,ecolor='k',fmt='none')
        py.plot([-2,2],[0,0],'k--')
        py.plot([0,0],[-0.3,0.3],'k--')
        py.xlim(-2,2)
        py.ylim(-0.3,0.3)
        py.xlabel('r (arcsec)')
        py.ylabel('h4')

        #py.savefig(workdir+'plots/osiris_vs_stis_h4_blue.png')

        #pdb.set_trace()

    py.close(3)
    py.figure(3, figsize=(7,11))
    ax1 = py.subplot(411)
    py.plot(ras,vpl,'ro')
    py.errorbar(ras,vpl,yerr=vsd,ecolor='r',fmt='none')
    #py.plot(rasshift,vslit,'go')
    if ((inData2 is None) and (inModel2 is None)):
        py.plot(stis['r'],stis['vel'],'ko')
        py.errorbar(stis['r'],stis['vel'],yerr=stis['dv'],ecolor='k',fmt='none')
        #py.plot(ras,v1,'k.')
        py.plot(ras,v2,'k-')
        #if B01:
        # bulge-sub B01 data is not at the same PA as the STIS data
            #py.plot(inB01_v['r'],inB01_v['vel'],'b-')
            #py.legend(('OSIRIS','STIS','STIS smooth','OASIS'),loc=0)
        #else:
        py.legend(('OSIRIS','STIS'),loc=0)
    else:
        py.plot(ras,vpl2,'ko')
        if inErr2 is not None:
            py.errorbar(ras,vpl2,yerr=ssd_2,ecolor='k',fmt='none')
        if leg is not None:
            py.legend((leg[0],leg[1]),loc=0)
        else:
            py.legend(('OSIRIS 1','OSIRIS 2'),loc=0)
    py.plot([-2,2],[0,0],'k--')
    py.plot([0,0],[-300,300],'k--')
    py.xlim(-2,2)
    py.ylim(-300,300)
    #py.ylim(-500,400)
    ax1.set_xticklabels([])
    py.yticks([-200,0,200])
    py.ylabel('v (km s$^{-1}$)')
    #pdb.set_trace()
    
    ax2 = py.subplot(412)
    py.plot(ras,spl,'ro-')
    py.errorbar(ras,spl,yerr=ssd,ecolor='r',fmt='none')
    #py.plot(rasshift,sslit,'go-')
    if ((inData2 is None) and (inModel2 is None)):
        py.plot(stis['r'],stis['sig'],'ko')
        py.errorbar(stis['r'],stis['sig'],yerr=stis['ds'],ecolor='k',fmt='none')
        #py.plot(ras,s1,'k.')
        py.plot(ras,s2,'k-')
        #if B01:
        #    py.plot(inB01_s['r'],inB01_s['sig'],'b-')
    else:
        py.plot(ras,spl2,'ko')
        if inErr2 is not None:
            py.errorbar(ras,spl2,yerr=ssd_2,ecolor='k',fmt='none')
    py.plot([-2,2],[0,0],'k--')
    py.plot([0,0],[0,350],'k--')
    py.xlim(-2,2)
    py.ylim(0,350)
    #py.ylim(0,500)
    ax2.set_xticklabels([])
    py.yticks([50,150,250])
    py.ylabel('$\sigma$ (km s$^{-1}$)')
    
    ax3 = py.subplot(413)
    py.plot(ras,h3pl,'ro-')
    py.errorbar(ras,h3pl,yerr=h3sd,ecolor='r',fmt='none')
    #py.plot(rasshift,h3slit,'go-')
    if ((inData2 is None) and (inModel2 is None)):
        py.plot(stis['r'],stis['h3'],'ko')
        py.errorbar(stis['r'],stis['h3'],yerr=stis['dh3'],ecolor='k',fmt='none')
        #py.plot(ras,h31,'k.')
        py.plot(ras,h32,'k-')
    else:
        py.plot(ras,h3pl2,'ko')
        if inErr2 is not None:
            py.errorbar(ras,h3pl2,yerr=h3sd_2,ecolor='k',fmt='none')
    py.plot([-2,2],[0,0],'k--')
    py.plot([0,0],[-0.3,0.3],'k--')
    py.xlim(-2,2)
    py.ylim(-0.3,0.3)
    ax3.set_xticklabels([])
    py.yticks([-.2,0.,.2])
    py.ylabel('h3')
    
    ax4 = py.subplot(414)
    py.plot(ras,h4pl,'ro-')
    py.errorbar(ras,h4pl,yerr=h4sd,ecolor='r',fmt='none')
    #py.plot(rasshift,h4slit,'go-')
    if ((inData2 is None) and (inModel2 is None)):
        py.plot(stis['r'],stis['h4'],'ko')
        py.errorbar(stis['r'],stis['h4'],yerr=stis['dh4'],ecolor='k',fmt='none')
        #py.plot(ras,h41,'k.')
        py.plot(ras,h42,'k-')
    else:
        py.plot(ras,h4pl2,'ko')
        if inErr2 is not None:
            py.errorbar(ras,h4pl2,yerr=h4sd_2,ecolor='k',fmt='none')
    py.plot([-2,2],[0,0],'k--')
    py.plot([0,0],[-0.3,0.3],'k--')
    py.xlim(-2,2)
    py.ylim(-0.3,0.3)
    #py.xticks([-1.5,0,1.5])
    py.yticks([-.2,0,.2])
    py.xlabel('r (arcsec)')
    py.ylabel('h4')
    py.tight_layout()
    py.subplots_adjust(wspace=0, hspace=0)
    
    #pdb.set_trace()

    if B01:
        # bulge-sub B01 data (their fig 10) is at PA=56, same as ours
        py.close(4)
        py.figure(4,figsize=(7,6))
        xaxis = (np.arange(vel.shape[1]) - bhpos_pix[0]) * 0.05
        # FWHM of B01 data (M8 mosaic) = .5 arcsec = 210 mas dispersion = 5 OSIRIS pixels
        disp_B01 = 5
        gaussB01 = ifu.gauss_kernel1D(disp_B01,1,half_box=disp_B01*3.)
        pdb.set_trace()
        v_c_B01 = np.convolve(vel[bhpos_pix[1],:],gaussB01,mode='same')
        ax1 = py.subplot(211)
        py.plot(xaxis,vel[bhpos_pix[1],:],'ro')
        py.plot(xaxis,v_c_B01,'r-')
        py.errorbar(xaxis,vel[bhpos_pix[1],:],yerr=velErr[bhpos_pix[1],:],ecolor='r',fmt='none')
        py.plot(inB01_v['r']-.1,inB01_v['vel'],'b^')
        py.ylim(-300,300)
        py.xlim(-2,2)
        py.plot([-2,2],[0,0],'k--')
        py.plot([0,0],[-300,300],'k--')
        ax1.set_xticklabels([])
        py.yticks([-200,0,200])
        py.ylabel('v (km s$^{-1}$)')
        py.legend(('OSIRIS','OSIRIS smooth','OASIS'),loc=0)

        ax2 = py.subplot(212)
        s_c_B01 = np.convolve(sig[bhpos_pix[1],:],gaussB01,mode='same')
        py.plot(xaxis,sig[bhpos_pix[1],:],'ro')
        py.plot(xaxis,s_c_B01,'r-')
        py.errorbar(xaxis,sig[bhpos_pix[1],:],yerr=sigErr[bhpos_pix[1],:],ecolor='r',fmt='none')
        py.plot(inB01_s['r']-.1,inB01_s['sig'],'b^')
        py.xlim(-2,2)
        py.ylim(0,350)
        py.plot([-2,2],[0,0],'k--')
        py.plot([0,0],[-300,300],'k--')
        py.yticks([50,150,250])
        py.ylabel('$\sigma$ (km s$^{-1}$)')
        py.xlabel('r (arcsec)')

        py.tight_layout()
        
    pdb.set_trace()
    if hAlpha:
        py.close(4)
        py.figure(4)
        py.plot(ras,vpl,'ro')
        #py.errorbar(ras,vpl,yerr=vsd,ecolor='r',fmt='none')
        py.plot(inHAlpha['r']-.1,inHAlpha['vel'],'ko')
        py.plot([-2,2],[0,0],'k--')
        py.plot([0,0],[-300,300],'k--')
        py.xlim(-2,2)
        py.ylim(-300,300)
        py.ylabel('v (km s$^{-1}$)')
        py.xlabel('Distance from BH (arcsec)')
        py.legend(('OSIRIS, cut PA=59.3 degrees','Menezes 2013 fig 3'),loc=0)
        py.title('Comparison with Menezes 2013 H$\\alpha$ velocities')
    
def plotDataModelResiduals(inputData=workdir+'ppxf.dat',inputModel=modeldir+'nonaligned_OSIRIScoords_fits_full_smooth.dat',inputRes=workdir+'model_residuals.dat',nonaligned=True,trimTess=False,outStem=None,incubeimg=None,pltitle=None):

    data = PPXFresults(inputData)
    model = modelFitResults(inputModel)
    res = modelFitResults(inputRes)

    if incubeimg is None:
        cubeimg = pyfits.getdata(datadir + cuberoot + '_img.fits')
    else:
        cubeimg = pyfits.getdata(incubeimg)

    xaxis = (np.arange(cubeimg.shape[0]) - bhpos_pix[0]) * 0.05
    yaxis = (np.arange(cubeimg.shape[1]) - bhpos_pix[1]) * 0.05
    
    xtickLoc = py.MultipleLocator(0.5)

    py.close(2)
    py.figure(2, figsize=(8,14))
    py.subplots_adjust(left=0.05, right=0.94, top=0.95, hspace=0.3)
    py.clf()

    ##########
    # Plot flux/nstar
    ##########
    py.subplot(3,1,1)
    py.imshow(np.rot90(py.ma.masked_where(cubeimg==0.,cubeimg/cubeimg.max()),3),vmin=0.,vmax=1.,
              extent=[xaxis[0], xaxis[-1], yaxis[0], yaxis[-1]])
    py.plot([0], 'ko', markeredgewidth=2,markerfacecolor='None')
    #py.ylabel('Y (arcsec)')
    py.ylabel('Data')
    py.xlabel('X (arcsec)')
    if pltitle:
        py.title(pltitle)
    py.gca().get_xaxis().set_major_locator(xtickLoc)
    py.axis('image')
    cbar = py.colorbar(orientation='vertical',ticks=[0.,.5,1.])
    cbar.set_label('Flux (norm)')

    # Make a compass rose
    #pa = 56.0
    pa = 34.0
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

    py.subplot(3,1,2)
    if trimTess:
        modFplot = np.rot90(model.nstar.transpose()/model.nstar.max(),3)
        py.imshow(py.ma.masked_where(modFplot==0.,modFplot),vmin=0.,vmax=1., extent=[xaxis[0], xaxis[-1], yaxis[0], yaxis[-1]])
    else:
        py.imshow(np.rot90((trimModel(model.nstar,nonaligned=nonaligned).transpose()/trimModel(model.nstar,nonaligned=nonaligned).max().transpose()),3),vmin=0.,vmax=1., extent=[xaxis[0], xaxis[-1], yaxis[0], yaxis[-1]])
    py.plot([0], 'ko', markeredgewidth=2,markerfacecolor='None')
    #py.ylabel('Y (arcsec)')
    py.ylabel('Model')
    py.xlabel('X (arcsec)')
    py.gca().get_xaxis().set_major_locator(xtickLoc)
    py.axis('image')
    cbar = py.colorbar(orientation='vertical',ticks=[0.,.5,1.])
    cbar.set_label('Flux (norm)')

    #testcubeimg = pyfits.getdata(datadir + cuberoot + '_img.fits')
    #testcubeimg = testcubeimg.transpose()
    #nstarres = (testcubeimg/testcubeimg.max()) - .7*(trimModel(model.nstar)/trimModel(model.nstar).max())
    py.subplot(3,1,3)
    py.imshow(np.rot90(py.ma.masked_where(cubeimg==0.,res.nstar.transpose()),3),vmin=-.25,vmax=.25,
    #py.imshow(np.rot90(py.ma.masked_where(cubeimg==0.,nstarres.transpose()),3),vmin=-.25,vmax=.25,
              extent=[xaxis[0], xaxis[-1], yaxis[0], yaxis[-1]])
    py.plot([0], 'ko', markeredgewidth=2,markerfacecolor='None')
    #py.ylabel('Y (arcsec)')
    py.ylabel('Residuals')
    py.xlabel('X (arcsec)')
    py.gca().get_xaxis().set_major_locator(xtickLoc)
    py.axis('image')
    cbar = py.colorbar(orientation='vertical',ticks=[-.25,0.,.25])
    cbar.set_label('Flux residuals')

    #py.tight_layout()

    pdb.set_trace()
    
    if outStem:
        py.savefig(outStem+'_flux.png')
        py.savefig(outStem+'_flux.eps')
    else:
        py.savefig(modelworkdir + 'plots/residuals_flux.png')
        py.savefig(modelworkdir + 'plots/residuals_flux.eps')
    py.show()

    
    
    ##########
    # Plot velocity
    ##########  
    py.clf()

    py.subplot(3,1,1)
    py.imshow(np.rot90(py.ma.masked_where((data.velocity.transpose()+vsys)==vsys,data.velocity.transpose()+vsys),3),vmin=-250.,vmax=250., 
              extent=[xaxis[0], xaxis[-1], yaxis[0], yaxis[-1]])
    py.plot([0], 'ko', markeredgewidth=2,markerfacecolor='None')
    #py.ylabel('Y (arcsec)')
    py.ylabel('Data')
    py.xlabel('X (arcsec)')
    if pltitle:
        py.title(pltitle)
    py.gca().get_xaxis().set_major_locator(xtickLoc)
    py.axis('image')
    cbar = py.colorbar(orientation='vertical',ticks=[-250.,0.,250])
    cbar.set_label('Velocity (km/s)')

    # Make a compass rose
    #pa = 56.0
    pa = 34.0
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
    
    py.subplot(3,1,2)
    if trimTess:
        modVplot = np.rot90(model.velocity.transpose(),3)
        py.imshow(py.ma.masked_where(modVplot==0.,modVplot),vmin=-250.,vmax=250., 
                extent=[xaxis[0], xaxis[-1], yaxis[0], yaxis[-1]])
    else:
        py.imshow(np.rot90(trimModel(model.velocity,nonaligned=nonaligned).transpose(),3),vmin=-250.,vmax=250., 
                extent=[xaxis[0], xaxis[-1], yaxis[0], yaxis[-1]])
    py.plot([0], 'ko', markeredgewidth=2,markerfacecolor='None' )
    #py.ylabel('Y (arcsec)')
    py.ylabel('Model')
    py.xlabel('X (arcsec)')
    py.gca().get_xaxis().set_major_locator(xtickLoc)
    py.axis('image')
    cbar = py.colorbar(orientation='vertical',ticks=[-250.,0.,250])
    cbar.set_label('Velocity (km/s)')

    py.subplot(3,1,3)
    py.imshow(np.rot90(res.velocity.transpose(),3),vmin=-100.,vmax=200., 
              extent=[xaxis[0], xaxis[-1], yaxis[0], yaxis[-1]])
    py.plot([0], 'ko', markeredgewidth=2,markerfacecolor='None' )
    #py.ylabel('Y (arcsec)')
    py.ylabel('Residuals')
    py.xlabel('X (arcsec)')
    py.gca().get_xaxis().set_major_locator(xtickLoc)
    py.axis('image')
    cbar = py.colorbar(orientation='vertical',ticks=[-100.,0.,200])
    cbar.set_label('Residuals (km/s)')

    #py.tight_layout()

    if outStem:
        py.savefig(outStem+'_velocity.png')
        py.savefig(outStem+'_velocity.eps')
    else:
        py.savefig(modelworkdir + 'plots/residuals_velocity.png')
        py.savefig(modelworkdir + 'plots/residuals_velocity.eps')
    py.show()

    pdb.set_trace()
    
    ##########
    # Plot sigma
    ##########  
    py.clf()

    py.subplot(3,1,1)
    py.imshow(np.rot90(py.ma.masked_where(data.sigma.transpose()==0.,data.sigma.transpose()),3),vmin=0.,vmax=300., 
              extent=[xaxis[0], xaxis[-1], yaxis[0], yaxis[-1]])
    py.plot([0], 'ko', markeredgewidth=2,markerfacecolor='None' )
    #py.ylabel('Y (arcsec)')
    py.ylabel('Data')
    py.xlabel('X (arcsec)')
    if pltitle:
        py.title(pltitle)
    py.gca().get_xaxis().set_major_locator(xtickLoc)
    py.axis('image')
    cbar = py.colorbar(orientation='vertical',ticks=[0.,150.,300.])
    cbar.set_label('Sigma (km/s)')

    # Make a compass rose
    #pa = 56.0
    pa = 34.0
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
    
    py.subplot(3,1,2)
    if trimTess:
        modSigplot = np.rot90(model.sigma.transpose(),3)
        py.imshow(py.ma.masked_where(modSigplot==0.,modSigplot),vmin=0.,vmax=300., 
                extent=[xaxis[0], xaxis[-1], yaxis[0], yaxis[-1]])
    else:
        py.imshow(np.rot90(trimModel(model.sigma,nonaligned=nonaligned).transpose(),3),vmin=0.,vmax=300., 
                extent=[xaxis[0], xaxis[-1], yaxis[0], yaxis[-1]])
    py.plot([0], 'ko', markeredgewidth=2,markerfacecolor='None' )
    #py.ylabel('Y (arcsec)')
    py.ylabel('Model')
    py.xlabel('X (arcsec)')
    py.gca().get_xaxis().set_major_locator(xtickLoc)
    py.axis('image')
    cbar = py.colorbar(orientation='vertical',ticks=[0.,150.,300.])
    cbar.set_label('Sigma (km/s)')

    py.subplot(3,1,3)
    py.imshow(np.rot90(res.sigma.transpose(),3),vmin=-150.,vmax=150., 
              extent=[xaxis[0], xaxis[-1], yaxis[0], yaxis[-1]])
    py.plot([0], 'ko', markeredgewidth=2,markerfacecolor='None' )
    #py.ylabel('Y (arcsec)')
    py.ylabel('Residuals')
    py.xlabel('X (arcsec)')
    py.gca().get_xaxis().set_major_locator(xtickLoc)
    py.axis('image')
    cbar = py.colorbar(orientation='vertical',ticks=[-150.,0.,150])
    cbar.set_label('Residuals (km/s)')

    #py.tight_layout()

    if outStem:
        py.savefig(outStem+'_sigma.png')
        py.savefig(outStem+'_sigma.eps')
    else:
        py.savefig(modelworkdir + 'plots/residuals_sigma.png')
        py.savefig(modelworkdir + 'plots/residuals_sigma.eps')
    py.show()

    ##########
    # Plot h3
    ##########
    py.close(2)
    py.figure(2, figsize=(8,14))
    py.subplots_adjust(left=0.05, right=0.94, top=0.95, hspace=0.3) 
    py.clf()

    py.subplot(3,1,1)
    py.imshow(np.rot90(py.ma.masked_where(data.h3.transpose()==0.,data.h3.transpose()),3),vmin=-0.2,vmax=0.2, extent=[xaxis[0], xaxis[-1], yaxis[0], yaxis[-1]])
    py.plot([0], 'ko', markeredgewidth=2,markerfacecolor='None' )
    #py.ylabel('Y (arcsec)')
    py.ylabel('Data')
    py.xlabel('X (arcsec)')
    if pltitle:
        py.title(pltitle)
    py.gca().get_xaxis().set_major_locator(xtickLoc)
    py.axis('image')
    cbar = py.colorbar(orientation='vertical',ticks=[-.2,0.,.2])
    cbar.set_label('h3')

    # Make a compass rose
    #pa = 56.0
    pa = 34.0
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

    py.subplot(3,1,2)
    if trimTess:
        modH3plot = np.rot90(model.h3.transpose(),3)
        py.imshow(py.ma.masked_where(modVplot==0.,modH3plot),vmin=-0.2,vmax=0.2, extent=[xaxis[0], xaxis[-1], yaxis[0], yaxis[-1]])
    else:
        py.imshow(np.rot90(trimModel(model.h3,nonaligned=nonaligned).transpose(),3),vmin=-0.2,vmax=0.2, extent=[xaxis[0], xaxis[-1], yaxis[0], yaxis[-1]])
    py.plot([0], 'ko', markeredgewidth=2,markerfacecolor='None' )
    #py.ylabel('Y (arcsec)')
    py.ylabel('Model')
    py.xlabel('X (arcsec)')
    py.gca().get_xaxis().set_major_locator(xtickLoc)
    py.axis('image')
    cbar = py.colorbar(orientation='vertical',ticks=[-.2,0.,.2])
    cbar.set_label('h3')

    py.subplot(3,1,3)
    py.imshow(np.rot90(res.h3.transpose(),3),vmin=-0.2,vmax=0.2, extent=[xaxis[0], xaxis[-1], yaxis[0], yaxis[-1]])
    py.plot([0], 'ko', markeredgewidth=2,markerfacecolor='None' )
    #py.ylabel('Y (arcsec)')
    py.ylabel('Residuals')
    py.xlabel('X (arcsec)')
    py.gca().get_xaxis().set_major_locator(xtickLoc)
    py.axis('image')
    cbar = py.colorbar(orientation='vertical',ticks=[-.2,0,.2])
    cbar.set_label('Residuals')

    #py.tight_layout()

    if outStem:
        py.savefig(outStem+'_h3.png')
        py.savefig(outStem+'_h3.eps')
    else:
        py.savefig(modelworkdir + 'plots/residuals_h3.png')
        py.savefig(modelworkdir + 'plots/residuals_h3.eps')
    py.show()

    py.close(2)
    py.figure(2, figsize=(8,14))
    py.subplots_adjust(left=0.05, right=0.94, top=0.95, hspace=0.3)
    py.clf()
    
    ##########
    # Plot h4
    ##########  
    py.clf()

    py.subplot(3,1,1)
    py.imshow(np.rot90(py.ma.masked_where(data.h4.transpose()==0.,data.h4.transpose()),3),vmin=-0.2,vmax=0.2, extent=[xaxis[0], xaxis[-1], yaxis[0], yaxis[-1]])
    py.plot([0], 'ko', markeredgewidth=2,markerfacecolor='None' )
    #py.ylabel('Y (arcsec)')
    py.ylabel('Data')
    py.xlabel('X (arcsec)')
    if pltitle:
        py.title(pltitle)
    py.gca().get_xaxis().set_major_locator(xtickLoc)
    py.axis('image')
    cbar = py.colorbar(orientation='vertical',ticks=[-.2,0,.2])
    cbar.set_label('h4')

    # Make a compass rose
    #pa = 56.0
    pa = 34.0
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

    py.subplot(3,1,2)
    if trimTess:
        modH4plot = np.rot90(model.h4.transpose(),3)
        py.imshow(py.ma.masked_where(modVplot==0.,modH4plot),vmin=-0.2,vmax=0.2, extent=[xaxis[0], xaxis[-1], yaxis[0], yaxis[-1]])
    else:
        py.imshow(np.rot90(trimModel(model.h4,nonaligned=nonaligned).transpose(),3),vmin=-0.2,vmax=0.2, extent=[xaxis[0], xaxis[-1], yaxis[0], yaxis[-1]])
    py.plot([0], 'ko', markeredgewidth=2,markerfacecolor='None' )
    #py.ylabel('Y (arcsec)')
    py.ylabel('Model')
    py.xlabel('X (arcsec)')
    py.gca().get_xaxis().set_major_locator(xtickLoc)
    py.axis('image')
    cbar = py.colorbar(orientation='vertical',ticks=[-.2,0,.2])
    cbar.set_label('h4')

    py.subplot(3,1,3)
    py.imshow(np.rot90(res.h4.transpose(),3),vmin=-0.2,vmax=0.2, extent=[xaxis[0], xaxis[-1], yaxis[0], yaxis[-1]])
    py.plot([0], 'ko', markeredgewidth=2,markerfacecolor='None' )
    #py.ylabel('Y (arcsec)')
    py.ylabel('Residuals')
    py.xlabel('X (arcsec)')
    py.gca().get_xaxis().set_major_locator(xtickLoc)
    py.axis('image')
    cbar = py.colorbar(orientation='vertical',ticks=[-.2,0.,.2])
    cbar.set_label('Residuals')

    py.tight_layout()

    if outStem:
        py.savefig(outStem+'_h4.png')
        py.savefig(outStem+'_h4.eps')
    else:
        py.savefig(modelworkdir + 'plots/residuals_h4.png')
        py.savefig(modelworkdir + 'plots/residuals_h4.eps')
    py.show()

def plotDataModel1D(inputData=workdir+'ppxf.dat',inputModel=modeldir+'nonaligned_OSIRIScoords_fits_full_smooth.dat'):
    # 1D version of plotDataModelResiduals

    data = PPXFresults(inputData)
    model = modelFitResults(inputModel)

    dataV = np.rot90(data.velocity.T,3)
    dataSig = np.rot90(data.sigma.T,3)
    dataH3 = np.rot90(data.h3.T,3)
    dataH4 = np.rot90(data.h4.T,3)

    modNstar = np.rot90(model.nstar.T,3)
    modV = np.rot90(model.velocity.T,3)
    modSig = np.rot90(model.sigma.T,3)
    modH3 = np.rot90(model.h3.T,3)
    modH4 = np.rot90(model.h4.T,3)

    cubeimg = pyfits.getdata(datadir + cuberoot + '_img.fits')
    cubeimg = np.rot90(cubeimg,3)

    yaxis = (np.arange(modNstar.shape[1]) - (bhpos_hor[0]/0.05)) * 0.05

    py.close(2)
    py.figure(2)
    py.clf()

    ####
    # flux
    ####

    bhcutpos = bhpos_hor[1]/.05
    py.plot(yaxis,modNstar[bhcutpos,:]/modNstar[bhcutpos,:].max())
    py.plot(yaxis,cubeimg[bhcutpos,:]/cubeimg[bhcutpos,:].max(),'ko')
    py.xlabel('Distance along cut at the SMBH (arcsec)')
    py.ylabel('Flux (scaled)')
    py.ylim(0.,1.05)
    py.xlim(yaxis[0],yaxis[-1])
    py.legend(('Model','OSIRIS data'))

    py.savefig(modelworkdir + 'plots/residuals_flux_1D.png')
    py.savefig(modelworkdir + 'plots/residuals_flux_1D.eps')

    #pdb.set_trace()
    ####
    # velocity
    ####

    py.clf()
    py.plot(yaxis,py.ma.masked_where(modV[bhcutpos,:]==0,modV[bhcutpos,:]))
    py.plot(yaxis,py.ma.masked_where(dataV[bhcutpos,:]==0.,dataV[bhcutpos,:]+vsys),'ko')
    py.xlabel('Distance along cut at the SMBH (arcsec)')
    py.ylabel('Velocity (km s$^{-1}$)')
    py.xlim(yaxis[0],yaxis[-1])
    py.ylim(-260,260)
    py.legend(('Model','OSIRIS data'))

    py.savefig(modelworkdir + 'plots/residuals_velocity_1D.png')
    py.savefig(modelworkdir + 'plots/residuals_velocity_1D.eps')
    
    #pdb.set_trace()
    ####
    # sigma
    ####

    py.clf()
    py.plot(yaxis,py.ma.masked_where(modSig[bhcutpos,:]==0,modSig[bhcutpos,:]))
    py.plot(yaxis,py.ma.masked_where(dataSig[bhcutpos,:]==0.,dataSig[bhcutpos,:]),'ko')
    py.xlabel('Distance along cut at the SMBH (arcsec)')
    py.ylabel('Dispersion (km s$^{-1}$)')
    py.xlim(yaxis[0],yaxis[-1])
    py.ylim(0,250)
    py.legend(('Model','OSIRIS data'))

    py.savefig(modelworkdir + 'plots/residuals_sigma_1D.png')
    py.savefig(modelworkdir + 'plots/residuals_sigma_1D.eps')

    #pdb.set_trace()
    ####
    # h3
    ####

    py.clf()
    py.plot(yaxis,py.ma.masked_where(modSig[bhcutpos,:]==0,modH3[bhcutpos,:]))
    py.plot(yaxis,py.ma.masked_where(dataSig[bhcutpos,:]==0.,dataH3[bhcutpos,:]),'ko')
    py.xlabel('Distance along cut at the SMBH (arcsec)')
    py.ylabel('h3')
    py.xlim(yaxis[0],yaxis[-1])
    py.ylim(-.2,.2)
    py.legend(('Model','OSIRIS data'))

    py.savefig(modelworkdir + 'plots/residuals_h3_1D.png')
    py.savefig(modelworkdir + 'plots/residuals_h3_1D.eps')

    #pdb.set_trace()
    ####
    # h4
    ####

    py.clf()
    py.plot(yaxis,py.ma.masked_where(modSig[bhcutpos,:]==0,modH4[bhcutpos,:]))
    py.plot(yaxis,py.ma.masked_where(dataSig[bhcutpos,:]==0.,dataH4[bhcutpos,:]),'ko')
    py.xlabel('Distance along cut at the SMBH (arcsec)')
    py.ylabel('h4')
    py.xlim(yaxis[0],yaxis[-1])
    py.ylim(-.2,.2)
    py.legend(('Model','OSIRIS data'))

    py.savefig(modelworkdir + 'plots/residuals_h4_1D.png')
    py.savefig(modelworkdir + 'plots/residuals_h4_1D.eps')

    #pdb.set_trace()
    
def plotQuiver(inputFile=None,nonaligned=True,binsize=0.25,inputFile2=None):
    # makes a quiver plot (x and y motions on the sky) of the model velocities
    # use binsize to specify bin size (in arcsec) for plotting

    # to get the difference between two different models, input a second model as inputFile2
    # results will be inputFile - inputFile2

    #xbinsize = binsize*2.
    xbinsize = binsize
    ybinsize = binsize
    
    if inputFile:
        model=modelResults(inputFile)
    else:
        model = modelResults(nonaligned=nonaligned,skycoords=True,OSIRIS=True)

    if inputFile2:
        model2 = modelResults(inputFile2)

    # 1" = 3.73 pc
    xbinpc = xbinsize*3.73
    ybinpc = ybinsize*3.73
    
    # Setting the BH pixel phase to match that of the data
    xfrac = bhpos[0]-np.floor(bhpos[0])
    yfrac = bhpos[1]-np.floor(bhpos[1])
    # correct for binsize != 0.05: divide by the ratio of the two pixel scales and
    # calc the new pixel phase
    xfrac = (xfrac*(.05/xbinsize))-np.floor(xfrac*(0.05/xbinsize))
    yfrac = (yfrac*(.05/ybinsize))-np.floor(yfrac*(0.05/ybinsize))
    # reposition BH (originally at origin in model) to the correct pixel phase
    model.x += (xbinpc*xfrac)
    model.y += (ybinpc*yfrac)
    if inputFile2:
        model2.x += (xbinpc*xfrac)
        model2.y += (ybinpc*yfrac)

    # get the full size of the binned array, but making sure to leave bin boundaries on the axes
    # positive and negative extent of the x axis
    posxbin = np.ceil(np.max(model.x)/xbinpc)
    negxbin = np.ceil(np.abs(np.min(model.x)/xbinpc))
    nxbin = posxbin+negxbin
    # and y axis
    posybin = np.ceil(np.max(model.y)/ybinpc)
    negybin = np.ceil(np.abs(np.min(model.y)/ybinpc))
    nybin = posybin + negybin

    # new BH position: (0,0) + (xfrac,yfrac)
    modbhpos = [negxbin+xfrac,negybin+yfrac]
    #print "Model BH is at ", modbhpos
    #pdb.set_trace()

    # initializing kinematic vectors
    vx = np.zeros((nxbin,nybin))
    vy = np.zeros((nxbin,nybin))

    # left/bottom edges of each spaxel (in units of pc)
    leftxbound = np.arange(-1.*negxbin*xbinpc, (posxbin*xbinpc) + (0.5*xbinpc), xbinpc)
    bottomybound = np.arange(-1*negybin*ybinpc, (posybin*ybinpc) + (0.5*ybinpc), ybinpc)
    # binning x and y velocity in bins of 5 km/s, with cuts at +/- 1000 km/s
    vxbins = np.arange(-1000., 1005., 5.)
    vybins = np.arange(-1000., 1005., 5.)

    t1 = time.time()

    iny = model.y
    inx = model.x
    if inputFile2:
        invx = model.vx - model2.vx
        invy = model.vy - model2.vy
    else:
        invx = model.vx
        invy = model.vy

    # create the histogram and average the x velocities in each bin
    #binsx = (leftxbound, bottomybound, vxbins)
    binsx = (bottomybound, leftxbound, vxbins)
    #sumvx, bins_vx, bin_num = scipy.stats.binned_statistic_dd((model.x, model.y, model.vx),
    sumvx, bins_vx, bin_num = scipy.stats.binned_statistic_dd((iny, inx, invx),
                                                               invx,
                                                               statistic='sum',
                                                               bins=binsx)


    # create the histogram and average the y velocities in each bin
    #binsy = (leftxbound, bottomybound, vybins)
    binsy = (bottomybound, leftxbound, vybins)
    #sumvy, bins_vy, bin_num2 = scipy.stats.binned_statistic_dd((model.x, model.y, model.vy),
    sumvy, bins_vy, bin_num2 = scipy.stats.binned_statistic_dd((iny, inx, invy),
                                                               invy,
                                                               statistic='sum',
                                                               bins=binsy)

    # create the histogram and count up the number of particles in each spatial bin
    #nstar, bins_count, bin_num3 = scipy.stats.binned_statistic_dd((model.x, model.y),
    nstar, bins_count, bin_num3 = scipy.stats.binned_statistic_dd((iny, inx),
                                                                  invx,
                                                                  statistic='count',
                                                                  #bins=(leftxbound, bottomybound))
                                                                  bins=(bottomybound, leftxbound))
    
    print 'Time Point 1: dt = {0:.0f} s'.format(time.time() - t1)
    #pdb.set_trace()

    meanvx = np.nan_to_num(sumvx.sum(axis=2)/nstar)
    meanvy = np.nan_to_num(sumvy.sum(axis=2)/nstar)

    meanvxf = meanvx.flatten('c')
    meanvyf = meanvy.flatten('c')
    nstarf = nstar.flatten('c')
    
    py.figure(3)
    py.clf()

    yy,xx=np.meshgrid(bins_vx[0][0:-1],bins_vx[1][0:-1])
    xxf=xx.flatten('f')
    yyf=yy.flatten('f')
    
    #py.quiver(bins_vx[0][0:-1],bins_vy[1][0:-1],-1.*meanvy,meanvx,nstar)
    #py.quiver(bins_vx[0][0:-1],bins_vy[1][0:-1],meanvx.T,meanvy.T,nstar.T)
    py.quiver(xxf,yyf,meanvxf,meanvyf,nstarf)
    cbar = py.colorbar(orientation='vertical')
    cbar.set_label('Number of Stars')
    py.xlabel('X (arcsec)')
    py.ylabel('Y (arcsec)')
    #py.xlim(5,-5)
    py.xlim(-5,5)
    py.ylim(-5,5)
    py.axes().set_aspect('equal', 'datalim')
    pdb.set_trace()

def trimModel(input,nonaligned=True):

    # trim models to match OSIRIS FOV
    # set up for PA = -34
    #trimrange = [[41.,88.],[21.,102.]]
    # new trimrange to match new BH position
    #trimrange = [[43.5,84.5],[17.5,101.5]]
    # BHpos (data) = [19.1,  40.7]
    if nonaligned:
        # BHpos (nonaligned, PA=-34) = [63.954999999999998, 59.034999999999997]
        #trimrange = [[44.9,85.9],[18.3,102.3]]
        # BHpos (nonaligned, PA=-56) = [57.954999999999998, 58.034999999999997]
        #trimrange = [[38.9,79.9],[17.3,101.3]]
        # BHps (nonaligned, new BH position) = [58.2, 57.6]
        #trimrange = [[39.0,80.0],[19.0,102.0]]
        # BHps (nonaligned, new BH position) = [58.7, 58.0]
        trimrange = [[38.7,79.7],[14.7,96.7]]
    else:
        # BHpos (aligned) = [54.954999999999998, 59.034999999999997]
        trimrange = [[35.9, 75.9],[18.3,102.3]]
        
    return input[trimrange[0][0]:trimrange[0][1],trimrange[1][0]:trimrange[1][1]]
    
    
def modelBin(inputFile=None,nonaligned=True,clean=False,l98bin=False):
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
    # if matching L98 photometry, use 0.0228" = 0.08504 pc
    if l98bin:
        binpc = 0.08504
    else:
        binpc = 0.1865
    
    # Setting the BH pixel phase to match that of the data
    xfrac = bhpos[0]-np.floor(bhpos[0])
    yfrac = bhpos[1]-np.floor(bhpos[1])
    # if L98 bin size, divide by the ratio of the two pixel scales and
    # calc the new pixel phase
    if l98bin:
        xfrac = (xfrac*(.05/.0228))-np.floor(xfrac*(0.05/0.0228))
        yfrac = (yfrac*(.05/.0228))-np.floor(yfrac*(0.05/0.0228))
    # reposition BH (originally at origin in model) to the correct pixel phase
    model.x += (binpc*xfrac)
    model.y += (binpc*yfrac)

    # get the full size of the binned array, but making sure to leave bin boundaries on the axes
    # positive and negative extent of the x axis
    posxbin = np.ceil(np.max(model.x)/binpc)
    negxbin = np.ceil(np.abs(np.min(model.x)/binpc))
    nxbin = posxbin+negxbin
    # and y axis
    posybin = np.ceil(np.max(model.y)/binpc)
    negybin = np.ceil(np.abs(np.min(model.y)/binpc))
    nybin = posybin + negybin

    # new BH position: (0,0) + (xfrac,yfrac)
    modbhpos = [negxbin+xfrac,negybin+yfrac]
    print "Model BH is at ", modbhpos
    #pdb.set_trace()

    # initializing flux and kinematic arrays
    losv = np.zeros((nxbin,nybin))
    sigma = np.zeros((nxbin,nybin))
    h3 = np.zeros((nxbin,nybin))
    h4 = np.zeros((nxbin,nybin))

    # left/bottom edges of each spaxel (in units of pc)
    leftxbound = np.arange(-1.*negxbin*binpc, (posxbin*binpc) + (0.5*binpc), binpc)
    bottomybound = np.arange(-1*negybin*binpc, (posybin*binpc) + (0.5*binpc), binpc)
    # binning LOS velocity in bins of 5 km/s, with cuts at +/- 1000 km/s
    vzbins = np.arange(-1000., 1005., 5.)
    pdb.set_trace()
    # by definition, v_LOS = -v_z
    modvLOS = -1.* model.vz

    t1 = time.time()

    # create the histogram and count up the number of particles in each velocity bin
    bins = (leftxbound, bottomybound, vzbins)
    losvd, bins_new, bin_num = scipy.stats.binned_statistic_dd((model.x, model.y, modvLOS),
                                                               modvLOS,
                                                               statistic='count',
                                                               bins=bins)

    # create the histogram and count up the number of particles in each spatial bin
    nstar, bins_new2, bin_num2 = scipy.stats.binned_statistic_dd((model.x, model.y),
                                                                  modvLOS,
                                                                  statistic='count',
                                                                  bins=(leftxbound, bottomybound))
    
    print 'Time Point 1: dt = {0:.0f} s'.format(time.time() - t1)
    pdb.set_trace()
       
    t2 = time.time()

    if clean:
        warnings.simplefilter("error", OptimizeWarning)            
    # this loop does the LOSVD fits for each bin, to get the kinematics
    # using the normal bin order for this
    for i in range(int(nxbin)):
        print "Starting column ", i
        for j in range(int(nybin)):
            # if there are no particles, set everything to zero
            if nstar[i,j] == 0:
                losv[i,j] = 0.
                sigma[i,j] = 0.
                h3[i,j] = 0.
                h4[i,j] = 0.
            # if there are too few particles to perform a fit, hack something together 
            elif losvd[i,j,:].max() <= 5:
                if clean:
                    losv[i,j] = 0.
                else:
                    losv[i,j] = (losvd[i,j,:]*bins_new[2][0:-1]).sum()/nstar[i,j]
                sigma[i,j] = 0.
                h3[i,j] = 0.
                h4[i,j] = 0.
            # for everything else, fit the LOSVD histogram
            else:
                # set the initial values
                gamma0 = nstar[i,j]
                # approximate the average velocity of the particles
                v0 = (losvd[i,j,:]*bins_new[2][0:-1]).sum()/nstar[i,j]
                # pull out the standard deviation directly from the histogram
                s0 = np.sqrt(((bins_new[2][0:-1]-v0)**2.*losvd[i,j,:]).sum()/nstar[i,j])
                h3_0 = 0.
                h4_0 = 0.
                guess = [gamma0, v0, s0, h3_0, h4_0]
                if clean:
                    try:
                        popt, pcov = curve_fit(gaussHermite, bins_new[2][0:-1], losvd[i,j,:], p0=guess)
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
                        popt, pcov = curve_fit(gaussHermite, bins_new[2][0:-1], losvd[i,j,:], p0=guess)
                        # popt = [gamma, v, sigma, h3, h4]
                        losv[i,j] = popt[1]
                        sigma[i,j] = popt[2]
                        h3[i,j] = popt[3]
                        h4[i,j] = popt[4]
                    except (RuntimeError):
                        # if no fit is possible, take the mean
                        losv[i,j] = v0
                        sigma[i,j] = 0.
                        h3[i,j] = 0.
                        h4[i,j] = 0.                    
            #py.close()
        # code crashes after exiting this loop (why??), so setting output here for now
        # update: it shouldn't crash here anymore, but this is fine here
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

        
def modelConvert2CSV(inputFile=None,smooth=False,l98bin=False,centerBH=None,reorient=True,normFlux=False):
    # convert pickle file to flat file
    # structure matches that of Hiranya's CSVs:
    ### flux file: [bin_y, bin_x, flux (or, here, nstar)]
    ### velocity file: [bin_y, bin_x, pos_y (in arcsec), pos_x, velocity, sigma, h3, h4]

    # Keywords:
    # smooth: smooth maps before flattening
    # l98bin: when False (default), pixel scale=0.05" (OSIRIS scale), when True,
    ### pixel scale=0.0228" (L98 pixel scale)
    # centerBH: if BH coordinates are given ([x,y]), pos_x and pos_y place the BH
    ### at [0.0, 0.0]
    # reorient: if True (default), orientation is N is 56 degrees to the left of top, 
    ###  E 90 degrees to the left of that
    # fluxNorm: normalize flux so peak flux = 1

    #inputPSF=workdir+'plots/osir_perf_m31_all_scalederr_cleanhdr_params.txt'
    inputPSF = '/Users/kel/Documents/Projects/M31/data/osiris_mosaics/drf/sigclip/all_NIRC2_CC_DTOTOFF_2/no100828/data/osiris_perf/sig_1.05/params.txt'
    
    mod = modelFitResults(inputFile)
    gridShape = mod.nstar.shape
    bingrid = np.indices((gridShape[0],gridShape[1]))
    rowgrid = bingrid[0,:,:]
    colgrid = bingrid[1,:,:]
    tmpbinx = colgrid.flatten()
    tmpbiny = rowgrid.flatten()
    tmpbinx = tmpbinx.astype(int)
    tmpbiny = tmpbiny.astype(int)
    PSFparams = readPSFparams(inputFile=inputPSF,twoGauss=False)
    if l98bin:
        pixscale = 0.0228
        PSF = ifu.gauss_kernel(PSFparams.sig1[0]*2.19, PSFparams.amp1[0], half_box=50)
    else:
        pixscale = 0.05
        PSF = ifu.gauss_kernel(PSFparams.sig1[0], PSFparams.amp1[0], half_box=50)
    if centerBH:
        tmpposx = (tmpbinx - centerBH[0])*pixscale
        tmpposy = (tmpbiny - centerBH[1])*pixscale
    else:
        tmpposx = tmpbinx*pixscale
        tmpposy = tmpbiny*pixscale
    
    if reorient:
        tmpnstar = np.rot90(mod.nstar.T,3)
        tmpv = np.rot90(mod.velocity.T,3)
        tmpsig = np.rot90(mod.sigma.T,3)
        tmph3 = np.rot90(mod.h3.T,3)
        tmph4 = np.rot90(mod.h4.T,3)
    else:
        tmpnstar = mod.nstar
        tmpv = mod.velocity
        tmpsig = mod.sigma
        tmph3 = mod.h3
        tmph4 = mod.h4
        
    if smooth:
        tmpnstar = signal.convolve(tmpnstar,PSF,mode='same')
        tmpv = signal.convolve(tmpv,PSF,mode='same')
        tmpsig = signal.convolve(tmpsig,PSF,mode='same')
        tmph3 = signal.convolve(tmph3,PSF,mode='same')
        tmph4 = signal.convolve(tmph4,PSF,mode='same')
        
    tmpnstar = tmpnstar.flatten()
    if normFlux:
        tmpnstar = tmpnstar/tmpnstar.max()
    tmpv = tmpv.flatten()
    tmpsig = tmpsig.flatten()
    tmph3 = tmph3.flatten()
    tmph4 = tmph4.flatten()
    #pdb.set_trace()
    outputFileFlux = inputFile.replace('.dat', '_fluxCSV.dat')
    outputFileVel = inputFile.replace('.dat', '_velCSV.dat')
    np.savetxt(outputFileFlux,np.c_[tmpbinx,tmpbiny,tmpnstar],fmt=('%d','%d','%8.6f'),delimiter='\t')
    np.savetxt(outputFileVel,np.c_[tmpbinx,tmpbiny, tmpposx, tmpposy, tmpv,tmpsig,tmph3,tmph4],fmt=('%d','%d','%8.6f','%8.6f','%8.6f','%8.6f','%8.6f','%8.6f'),delimiter='\t')

class modelReadCSV(object):
    def __init__(self, inputFile=None, vel=False):
        self.inputFile = inputFile
        # read in the file created by modelConvert2CSV

        if vel is False:
            mod = pandas.read_csv(inputFile,delim_whitespace=True,header=None,names=['y','x','f'])
        else:
            mod = pandas.read_csv(inputFile,delim_whitespace=True,header=None,names=['y','x','posy','posx','v','sigma','h3','h4'])

        imgShape = (mod.x.max()-mod.x.min()+1,mod.y.max()-mod.y.min()+1)

        off = mod.x.min()

        if vel:
            tmpposx = np.zeros(imgShape,dtype=float)
            tmpposy = np.zeros(imgShape,dtype=float)
            tmpv = np.zeros(imgShape,dtype=float)
            tmpsig = np.zeros(imgShape,dtype=float)
            tmph3 = np.zeros(imgShape,dtype=float)
            tmph4 = np.zeros(imgShape,dtype=float)

            
            for i in range(mod.y.shape[0]):
                tmpposx[mod.x[i]-off,mod.y[i]-off] = mod.posx[i]
                tmpposy[mod.x[i]-off,mod.y[i]-off] = mod.posy[i]
                tmpv[mod.x[i]-off,mod.y[i]-off] = mod.v[i]
                tmpsig[mod.x[i]-off,mod.y[i]-off] = mod.sigma[i]
                tmph3[mod.x[i]-off,mod.y[i]-off] = mod.h3[i]
                tmph4[mod.x[i]-off,mod.y[i]-off] = mod.h4[i]

            # check the order of the sorting - if x is increasing faster, use 'f',
            # if y is increasing faster, use 'c'
            #if mod.x[0]==mod.y[1]:
            #    sortorder = 'c'
            #else:
            #    sortorder = 'f'
            #tmpposx = np.reshape(mod.posx,imgShape,order=sortorder)
            #tmpposy = np.reshape(mod.posy,imgShape,order=sortorder)
            #tmpv = np.reshape(mod.v,imgShape,order=sortorder)
            #tmpsig = np.reshape(mod.sigma,imgShape,order=sortorder)
            #tmph3 = np.reshape(mod.h3,imgShape,order=sortorder)
            #tmph4 = np.reshape(mod.h4,imgShape,order=sortorder)
            
            self.posx = tmpposx
            self.posy = tmpposy
            self.v = tmpv
            self.sigma = tmpsig
            self.h3 = tmph3
            self.h4 = tmph4
            
        else:
            tmpnstar = np.zeros(imgShape,dtype=float)

            for i in range(mod.y.shape[0]):
                tmpnstar[mod.x[i]-off,mod.y[i]-off] = mod.f[i]

            #if mod.x[0]==mod.y[1]:
            #    sortorder = 'c'
            #else:
            #    sortorder = 'f'
            #tmpnstar = np.reshape(mod.f,imgShape,order=sortorder)
            
            self.nstar = tmpnstar
            
def smoothModels(inputModel=modeldir+'nonaligned_OSIRIScoords_fits_full.dat', inputPSF=workdir+'plots/osir_perf_m31_all_scalederr_cleanhdr_params.txt', inModArr=None, twoGauss=False):

    if not inModArr[0][0][0]:
        model = modelFitResults(inputFile=inputModel)
        modNstar = model.nstar
        modV = model.velocity
        modS = model.sigma
        modH3 = model.h3
        modH4 = model.h4
    else:
        modNstar = inModArr[0][:][:]
        modV = inModArr[1][:][:]
        modS = inModArr[2][:][:]
        modH3 = inModArr[3][:][:]
        modH4 = inModArr[4][:][:]
        
    #pdb.set_trace()
    if twoGauss:
        PSFparams = readPSFparams(inputFile=inputPSF,twoGauss=True)
    else:
        PSFparams = readPSFparams(inputFile=inputPSF,twoGauss=False)

    PSF = ifu.gauss_kernel(PSFparams.sig1[0], PSFparams.amp1[0], half_box=50)
    nstar = signal.convolve(modNstar,PSF,mode='same')
    velocity = signal.convolve(modV,PSF,mode='same')
    sigma = signal.convolve(modS,PSF,mode='same')
    h3 = signal.convolve(modH3,PSF,mode='same')
    h4 = signal.convolve(modH4,PSF,mode='same')

    if not inModArr[0][0][0]:
        #output = open(modeldir + 'nonaligned_OSIRIScoords_fits_full_smooth.dat', 'w')
        outFile = inputModel.replace('.dat', '_smooth.dat')
        output = open(outFile, 'w')
        
        pickle.dump(nstar, output)
        pickle.dump(velocity, output)
        pickle.dump(sigma, output)
        pickle.dump(h3, output)
        pickle.dump(h4, output)
        output.close()

    else:
        out = [modNstar,modV,modS,modH3,modH4]
        return out

def modelResiduals(inputModel=modeldir+'nonaligned_OSIRIScoords_fits_full_smooth.dat',inputData=workdir+'ppxf.dat',trimTess=False,incubeimg=None):

    model = modelFitResults(inputFile=inputModel)
    data = PPXFresults(inputFile=inputData)
    if incubeimg is None:
        cubeimg = pyfits.getdata(datadir + cuberoot + '_img.fits')
    else:
        cubeimg = pyfits.getdata(incubeimg)
    cubeimg = cubeimg.transpose()

    # normalized flux residuals
    if trimTess:
        nstar = (cubeimg/cubeimg.max()) - (model.nstar/model.nstar.max())
        velocity = (data.velocity+vsys) - py.ma.masked_where(data.velocity == 0.,model.velocity)
        sigma = data.sigma - py.ma.masked_where(data.sigma == 0.,model.sigma)
        h3 = data.h3 - py.ma.masked_where(data.h3 == 0.,model.h3)
        h4 = data.h4 - py.ma.masked_where(data.h4 == 0.,model.h4)
    else:
        nstar = (cubeimg/cubeimg.max()) - (trimModel(model.nstar)/trimModel(model.nstar).max())
        velocity = (data.velocity+vsys) - py.ma.masked_where(data.velocity == 0.,trimModel(model.velocity))
        sigma = data.sigma - py.ma.masked_where(data.sigma == 0.,trimModel(model.sigma))
        h3 = data.h3 - py.ma.masked_where(data.h3 == 0.,trimModel(model.h3))
        h4 = data.h4 - py.ma.masked_where(data.h4 == 0.,trimModel(model.h4))

    #output = open(workdir+'model_residuals.dat','w')
    outFile = inputModel.replace('.dat','_residuals.dat')
    output = open(outFile, 'w')

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
    
def modelConvertCoordinates(nonaligned=True,test=False,testLIA=None):
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
            thetaL = np.radians(testLIA[0])
            thetaI = np.radians(testLIA[1])
            thetaA = np.radians(testLIA[2])
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
    #cpa = -34.
    cpa = -56.
    #testing a model rotation thing - reset to -56 when done
    #cpa = -45.
    
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

def create_model_templates(tWeights,IDL=False,norm=False,normmask=None,rebinWave=True):
    # create model templates by inputting the template weights (output from ppxf run)
    # output is the linear combinations of the template set

    # hard-coded from prior run of ppxf
    velScale = 34.596540091951731

    # read in templates
    logWaveTemps, templates = load_templates(velScale,IDL=IDL,rebinWave=rebinWave)
    tempShape = templates.shape

    twShape = tWeights.shape
    combSpec = np.zeros((twShape[0],twShape[1],tempShape[0]),dtype=float)
    
    if norm:
        # if set, normalizes the weights in each spaxel so they sum to 1 (or the value in normmask, if given)
        totTW = tWeights.sum(axis=2)
    #    if normmask is None:
    #        normmask=np.ones([tWeights.shape[0],tWeights.shape[1]])
        for i in range(twShape[2]):
            tWeights[:,:,i] /= totTW
    #        tWeights[:,:,i] *= normmask

        
    for i in range(twShape[0]):
        tmp = tWeights[i,:,:].dot(templates.T)
        combSpec[i,:,:] = tmp

    #pdb.set_trace()
    return logWaveTemps, combSpec
        
def load_templates(velScale, resolution=3241, IDL=True, selectTemp=None, rebinWave=True):
    # IDL and Python versions of pPXF require different formats for the input templates
    templateDir = '/Users/kel/Documents/Library/IDL/ppxf/templates/GNIRS/library_v15_gnirs_combined/'

    files = glob.glob(templateDir + '*.fits')
    

    templates = None
    for ff in range(len(files)):
        newWave, newSpec = load_spectrum_gnirs(files[ff], velScale, resolution, rebinWave=rebinWave)

        if templates == None:
            if IDL:
                templates = np.zeros(( len(files),len(newSpec)), dtype=float)
            else: 
                templates = np.zeros(( len(newSpec),len(files)), dtype=float)

        if IDL:
            templates[ff,:] = newSpec
        else:
            templates[:,ff] = newSpec

    if selectTemp:
        print 'Using %d templates' % len(selectTemp)
        if IDL:
            newTemp = templates[selectTemp,:]
            templates = newTemp
        else:
            newTemp = templates[:,selectTemp]
            templates = newTemp
    else:
        print 'Using %d templates' % len(files)
            
    return (newWave, templates)

def load_template_names():
    templateDir = '/Users/kel/Documents/Library/IDL/ppxf/templates/GNIRS/library_v15_gnirs_combined/'
    nc = len(templateDir)
    
    files = glob.glob(templateDir + '*.fits')
    nf = len(files)

    specdict = {'hd113538': 'K8 V', 'hd173764': 'G4 IIa', 'hd1737': 'G5 III', 'hd20038': 'F7 IIIw', 'hd206067': 'K0 III', 'hd212320': 'G6 III', 'hd218594': 'K1 III', 'hd224533': 'G9 III', 'hd2490': 'M0 III', 'hd26965': 'K1 V(a)', 'hd32440': 'K6 III', 'hd34642': 'K0 IV', 'hd35369': 'G8 III', 'hd36079': 'G5 II', 'hd38392': 'K2 V', 'hd39425': 'K2 III', 'hd4188': 'K0 III', 'hd4730': 'K3 III', 'hd63425B': 'K7 III', 'hd64606': 'G8 V', 'hd6461': 'G3 V', 'hd720': 'K5 III', 'hd9138': 'K4 III'}
    
    names = []
    spectype = []
    for i in range(nf):
        tmpname = files[i][nc:-1]
        tmpname = tmpname.split('_')
        tmpname = tmpname[0]
        names.append(tmpname)
        tmpspec = specdict[tmpname]
        spectype.append(tmpspec)

    return (names, spectype)

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

def load_spectrum_gnirs(file, velScale, resolution, rebinWave=True):
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

    if rebinWave:
        # Rebin into equals logarithmic steps in wavelength
        logWave, specNew, vel = log_rebin2(wavelength, specLowRes,
                                        inMicrons=True, velScale=velScale)
        outWave = logWave
    else:
        outWave = wavelength
        specNew = specLowRes

    return outWave, specNew


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

    spectrum = np.nan_to_num(spectrum)
    
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
    v_smooth = astropy.convolution.convolve(tmp+vsys,kern,boundary='extend')
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

