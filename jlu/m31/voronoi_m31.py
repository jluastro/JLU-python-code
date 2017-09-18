import math
from astropy.io import fits as pyfits
import numpy as np
import pylab as py
import scipy
from scipy import signal
import time
import pickle
from scipy.optimize import curve_fit
from jlu.osiris import cube as cube_code
from jlu.osiris import spec
from jlu.m31 import ppxf_m31
import glob
import os
import pdb
import atpy
import voronoi_2d_binning as v2d

datadir = '/Users/kel/Documents/Projects/M31/analysis_new/ifu_11_11_30/'
modeldir = '/Users/kel/Documents/Projects/M31/models/Peiris/2003/'
cuberoot = 'm31_all_scalederr_cleanhdr_bulgesub'

bhpos_hor = ppxf_m31.bhpos_hor
bhpos_horpix = bhpos_hor*20.

def createVoronoiInput(cubeFile=None):
    # makes a version of the median flux map that is all positive, and a
    # version of the error array that is directly scaled from this new
    # flux map and the SNR map created empirically (via ifu.map_snr)
    if cubeFile:
        cubefits = pyfits.open(cubeFile)
    else:
        cubefits = pyfits.open(datadir + cuberoot + '.fits')
    
    cube = cubefits[0].data
    hdr = cubefits[0].header
    errors = cubefits[1].data
    quality = cubefits[2].data
    #nframes = cubefits[3].data

    #snrimg = pyfits.getdata(datadir + cuberoot + '_snr.fits')
    if cubeFile:
        #snrimg = pyfits.getdata(cubeFile.replace('_bulgesub.fits','_snr.fits'))
        snrimg = pyfits.getdata(cubeFile.replace('.fits','_snr.fits'))
    else:
        snrimg = pyfits.getdata(datadir+'m31_all_scalederr_cleanhdr_snr.fits')
    
    xx = np.arange(cube.shape[0])
    yy = np.arange(cube.shape[1])
    imgShape = (cube.shape[0],cube.shape[1])

    cubeVor = np.zeros(imgShape, dtype=float)
    errVor = np.zeros(imgShape, dtype=float)

    for nx in xx:
        for ny in yy:
            tmpcube = cube[nx,ny,:]
            # add a constant to the spectrum to make it above 0
            #print "old cube mean is %f " % tmpcube.mean()
            #minFlux = tmpcube.mean() - (1.0 * tmpcube.std())
            #print "minflux is %f" % minFlux
            #tmpcube += np.abs(minFlux)
            #print "new cube mean is %f " % tmpcube.mean()
            tmpcubeavg = tmpcube.mean()
            tmpcubeavg = np.median(tmpcube)
            tmpsnr = np.abs(snrimg[ny,nx])
            #tmpsnr = np.abs(snrimg[nx,ny])
            # calc errors for tessellation based on the empirical
            # S/N already calculated
            tmperr = tmpcubeavg/tmpsnr
            cubeVor[nx,ny] = tmpcubeavg
            errVor[nx,ny] = tmperr
            #if ny==71:
            #    print 'ny = 71'
            #    if nx==28:
            #        pdb.set_trace()
            
    # change NaN to 0
    errVor = np.nan_to_num(errVor)

    if cubeFile:
        outfile = cubeFile.replace('.fits','_vor.fits')
    else:
        outfile = datadir + cuberoot + '_vor.fits'
    pyfits.writeto(outfile, cubeVor, header=hdr)
    pyfits.append(outfile, errVor)
            
def tessellate(inputFile=datadir+cuberoot+'_vor.fits',targetSN=50):

    cubefits = pyfits.open(inputFile)
    
    cube = cubefits[0].data
    hdr = cubefits[0].header
    errors = cubefits[1].data

    #pdb.set_trace()
    #xx = np.arange(cube.shape[0])
    #yy = np.arange(cube.shape[1])
    good = np.where((cube > 0) & (errors > 0) & ((cube/errors) >= 2.))
    goodbad=np.zeros((cube.shape[0],cube.shape[1]),dtype=float)
    goodbad[good]=1.
    xx = good[1]
    yy = good[0]
    #pdb.set_trace()
    binNum, xNode, yNode, xBar, yBar, sn, nPixels, scale, pixSize = v2d.voronoi_2d_binning(xx, yy, cube[good], errors[good], targetSN, plot=1, quiet=0)
        
    py.close(1)
    py.figure(1,figsize=(8,8))
    py.subplots_adjust(left=0.16, right=0.94, top=0.95)
    py.subplot(211)
    rnd = np.argsort(np.random.random(xNode.size))  # Randomize bin colors
    # added in flips to display in the same orientation as the data
    # divide by 20 to put in arcsec (0.05 arcsec per pixel)
    xxpl = (xx - ppxf_m31.bhpos_pix[0]) * 0.05
    yypl = (np.flipud(yy) - ppxf_m31.bhpos_pix[1]) * 0.05
    xnpl = (xNode - ppxf_m31.bhpos_pix[0]) * 0.05
    ynpl = (yNode - ppxf_m31.bhpos_pix[1]) * 0.05
    v2d._display_pixels(xxpl, yypl, rnd[binNum], pixSize/20., horflip=True)
    py.plot(xnpl, ynpl, '+w', scalex=False, scaley=False) # do not rescale after imshow()
    py.plot([0], 'ko', markeredgewidth=2,markerfacecolor='None')
    py.axis('image')
    py.ylabel('Y (arcsec)')
    py.xlabel('X (arcsec)')
    py.title('Map of Voronoi bins')
    
    py.subplot(212)
    rad = np.sqrt(np.abs(xBar-bhpos_horpix[0])**2 + np.abs(yBar-bhpos_horpix[1])**2)  # Use centroids, NOT generators
    w = nPixels == 1
    py.plot(rad[~w]/20., sn[~w], 'or', label='Voronoi bins')
    py.xlabel('Distance from SMBH (arcsec)')
    py.ylabel('Bin S/N')
    py.axis([np.min(rad/20.), np.max(rad/20.), 0, np.max(sn)])  # x0, x1, y0, y1
    if np.sum(w) > 0:
        py.plot(rad[w]/20., sn[w], 'xb', label='single spaxels')
    py.axhline(targetSN)
    py.legend()
    py.pause(0.01)  # allow plot to appear in certain cases

    pdb.set_trace()
    outfile = os.path.dirname(inputFile)+'/voronoi_2d_binning_output.txt'
    np.savetxt(outfile, np.column_stack([xx, yy, binNum]),
               fmt=b'%10.6f %10.6f %8i')
    #np.savetxt(datadir+'voronoi_2d_binning_output.txt', np.column_stack([xx, yy, binNum]),
    #           fmt=b'%10.6f %10.6f %8i')

def createVoronoiOutput(inputFile=datadir+cuberoot+'.fits',inputVoronoiFile=datadir+'voronoi_2d_binning_output.txt'):
    cubefits = pyfits.open(inputFile)
    
    cube = cubefits[0].data
    hdr = cubefits[0].header
    errors = cubefits[1].data
    quality = cubefits[2].data
    nframes = cubefits[3].data

    cubeShape = (cube.shape[0],cube.shape[1],cube.shape[2])

    yy, xx, binnum = np.loadtxt(inputVoronoiFile,unpack=True)
    xx = xx.astype(int)
    yy = yy.astype(int)
    binnum = binnum.astype(int)

    newCube = np.zeros(cubeShape, dtype=float)
    newErr = np.zeros(cubeShape, dtype=float)

    for nb in range(binnum.max()+1):
        idx = np.where(binnum == nb)
        nx = xx[idx]
        ny = yy[idx]
        nbins = len(idx[0])
        tmpCube = np.sum(cube[nx,ny,:],axis=0)/nbins
        tmpErr = np.sqrt(np.sum(errors[nx,ny,:]**2,axis=0))/nbins
        newCube[nx,ny,:] = tmpCube
        newErr[nx,ny,:] = tmpErr

    #pdb.set_trace()
    outfile = inputFile.replace('.fits','_vorcube.fits')
    pyfits.writeto(outfile,newCube,header=hdr)
    pyfits.append(outfile,newErr)
    pyfits.append(outfile,quality)
    pyfits.append(outfile,nframes)
    
def tessModels(inputModel=modeldir+'nonaligned_model_OSIRIScoords.dat',inputModelArr=None,inputVoronoiFile=datadir+'voronoi_2d_binning_output.txt',l98bin=False,outFile=None):
    
    # read in the full model file (not fitted)
    if inputModelArr is None:
        model = ppxf_m31.modelResults(inputFile=inputModel)
        X = model.x
        Y = model.y
        Z = model.z
        VX = model.vx
        VY = model.vy
        VZ = model.vz
    else:
        X = inputModelArr[:,0]
        Y = inputModelArr[:,1]
        Z = inputModelArr[:,2]
        VX = inputModelArr[:,3]
        VY = inputModelArr[:,4]
        VZ = inputModelArr[:,5]

    # this section is all taken from ppxf_m31.modelBin()
    # bin size = 0.05" = 0.1865 pc
    # if matching L98 photometry, use 0.0228" = 0.08504 pc
    if l98bin:
        binpc = 0.08504
    else:
        binpc = 0.1865
    
    # Setting the BH pixel phase to match that of the data
    xfrac = (ppxf_m31.bhpos[0]/0.05)-np.floor(ppxf_m31.bhpos[0]/0.05)
    yfrac = (ppxf_m31.bhpos[1]/0.05)-np.floor(ppxf_m31.bhpos[1]/0.05)
    # if L98 bin size, divide by the ratio of the two pixel scales and
    # calc the new pixel phase
    if l98bin:
        xfrac = (xfrac*(.05/.0228))-np.floor(xfrac*(0.05/0.0228))
        yfrac = (yfrac*(.05/.0228))-np.floor(yfrac*(0.05/0.0228))
    # reposition BH (originally at origin in model) to the correct pixel phase
    X += (binpc*xfrac)
    Y += (binpc*yfrac)

    # get the full size of the binned array, but making sure to leave bin boundaries on the axes
    # positive and negative extent of the x axis
    posxbin = np.ceil(np.max(X)/binpc)
    negxbin = np.ceil(np.abs(np.min(X)/binpc))
    nxbin = posxbin+negxbin
    # and y axis
    posybin = np.ceil(np.max(Y)/binpc)
    negybin = np.ceil(np.abs(np.min(Y)/binpc))
    nybin = posybin + negybin

    # new BH position: (0,0) + (xfrac,yfrac)
    modbhpos = [negxbin+xfrac,negybin+yfrac]
    print("Model BH is at ", modbhpos)

    # trim to only cover the OSIRIS FOV
    newnegxbin = 0. - np.floor(ppxf_m31.bhpos[1]/0.05)
    newnegybin = 0. - np.floor(ppxf_m31.bhpos[0]/0.05)

    xlen = 41
    ylen = 82
    goodTrim = np.where((X/binpc >= newnegxbin) & (X/binpc <= (newnegxbin + xlen)) & (Y/binpc >= newnegybin) & (Y/binpc <= (newnegybin + ylen)))

    xClip = X[goodTrim[0]] - newnegxbin*binpc
    yClip = Y[goodTrim[0]] - newnegybin*binpc
    zClip = Z[goodTrim[0]]
    vxClip = VX[goodTrim[0]]
    vyClip = VY[goodTrim[0]]
    vzClip = VZ[goodTrim[0]]

    # convert x,y positions to bin numbers
    xBin = np.floor(xClip/binpc)
    yBin = np.floor(yClip/binpc)
    xClippc = xClip/binpc
    yClippc = yClip/binpc
    xyBin = list(zip(xBin,yBin))

    #pdb.set_trace()

    # grab the Voronoi bins
    yy, xx, binnum = np.loadtxt(inputVoronoiFile,unpack=True)
    xx = xx.astype(int)
    yy = yy.astype(int)
    binnum = binnum.astype(int)

    # initiate the output arrays
    modShape = (xlen,ylen)

    newNstar = np.zeros(modShape, dtype=float)
    newVel = np.zeros(modShape, dtype=float)
    newSigma = np.zeros(modShape, dtype=float)
    newH3 = np.zeros(modShape, dtype=float)
    newH4 = np.zeros(modShape, dtype=float)

    #pdb.set_trace()
    # binning LOS velocity in bins of 5 km/s, with cuts at +/- 1000 km/s
    vzbins = np.arange(-1000., 1005., 5.)
    
    # by definition, v_LOS = -v_z
    modvLOS = -1.* vzClip

    xbins = np.arange(xlen+1)
    ybins = np.arange(ylen+1)
    # create the histogram and count up the number of particles in each velocity bin
    bins = (xbins, ybins, vzbins)
    losvd, bins_new, bin_num = scipy.stats.binned_statistic_dd((xClippc, yClippc, modvLOS),
                                                               modvLOS,
                                                               statistic='count',
                                                               bins=bins)

    # create the histogram and count up the number of particles in each spatial bin
    nstar, bins_new2, bin_num2 = scipy.stats.binned_statistic_dd((xClippc, yClippc),
                                                                  modvLOS,
                                                                  statistic='count',
                                                                  bins=(xbins, ybins))

    # initiate the temporary arrays sorted by bin number
    nstarnb = np.zeros(binnum.max()+1)
    losvdnb = np.zeros((binnum.max()+1,len(vzbins)-1))
    losv = np.zeros(binnum.max()+1)
    sigma = np.zeros(binnum.max()+1)
    h3 = np.zeros(binnum.max()+1)
    h4 = np.zeros(binnum.max()+1)

    #pdb.set_trace()

    # do the LOSVD fits, by bin number
    for i in range(binnum.max()+1):
        # sum up nstar and losvd for each bin
        #print "Starting bin ", i
        idx = np.where(binnum == i)
        nx = xx[idx]
        ny = yy[idx]
        tmpnstar = nstar[nx,ny].sum()
        nstarnb[i] = tmpnstar
        if len(nx) == 1:
            tmplosvd = losvd[nx,ny,:].flatten()
        else:
            tmplosvd = losvd[nx,ny,:].sum(axis=0)
        if len(tmplosvd) != len(vzbins)-1:
            tmplosvd = tmplosvd.sum(axis=0)
        losvdnb[i] = tmplosvd
        # if there are no particles in the bin, set everything to 0
        if tmpnstar == 0:
            losv[i] = 0.
            sigma[i] = 0.
            h3[i] = 0.
            h4[i] = 0.
        # if there are too few particles to perform a fit, hack something together 
        elif losvdnb[i].max() <= 5:
            losv[i] = (losvdnb[i]*bins_new[2][0:-1]).sum()/tmpnstar
            sigma[i] = 0.
            h3[i] = 0.
            h4[i] = 0.
        # for everything else, fit the LOSVD histogram
        else:
            # set the initial values
            gamma0 = tmpnstar
            # approximate the average velocity of the particles
            v0 = (losvdnb[i]*bins_new[2][0:-1]).sum()/tmpnstar
            # pull out the standard deviation directly from the histogram
            s0 = np.sqrt(((bins_new[2][0:-1]-v0)**2.*losvdnb[i]).sum()/tmpnstar)
            h3_0 = 0.
            h4_0 = 0.
            guess = [gamma0, v0, s0, h3_0, h4_0]
            popt, pcov = curve_fit(ppxf_m31.gaussHermite, bins_new[2][0:-1], losvdnb[i], p0=guess)
            # popt = [gamma, v, sigma, h3, h4]
            losv[i] = popt[1]
            sigma[i] = popt[2]
            h3[i] = popt[3]
            h4[i] = popt[4]             
    
    for nb in range(binnum.max()+1):
        #print 'round 2, starting bin', nb
        idx = np.where(binnum == nb)
        nx = xx[idx]
        ny = yy[idx]
        nbins = len(idx[0])
        newNstar[nx,ny] = nstarnb[nb]/nbins
        newVel[nx,ny] = losv[nb]
        newSigma[nx,ny] = sigma[nb]
        newH3[nx,ny] = h3[nb]
        newH4[nx,ny] = h4[nb]

    #pdb.set_trace()

    if outFile:
        outputFile = outFile
    else:
        outFile = inputModel.replace('.dat','_fits_trim_tess.dat')
    output = open(outputFile, 'wb')
    pickle.dump(newNstar, output)
    pickle.dump(newVel, output)
    pickle.dump(newSigma, output)
    pickle.dump(newH3, output)
    pickle.dump(newH4, output)
    output.close()

