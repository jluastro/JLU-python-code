import math
import pyfits
import numpy as np
import pylab as py
import scipy
import asciidata
from scipy import signal
from pyraf import iraf as ir
from jlu.osiris import cube as cube_code
from jlu.osiris import spec
from jlu.m31 import ppxf_m31
import glob
import os
import pdb
import atpy
import voronoi_2d_binning as v2d

datadir = '/Users/kel/Documents/Projects/M31/analysis_new/ifu_11_11_30/'

cuberoot = 'm31_all_scalederr_cleanhdr'

def createVoronoiInput():
    # makes a version of the median flux map that is all positive, and a
    # version of the error array that is directly scaled from this new
    # flux map and the SNR map created empirically (via ifu.map_snr)
    cubefits = pyfits.open(datadir + cuberoot + '.fits')
    
    cube = cubefits[0].data
    hdr = cubefits[0].header
    errors = cubefits[1].data
    quality = cubefits[2].data
    nframes = cubefits[3].data

    snrimg = pyfits.getdata(datadir + cuberoot + '_snr.fits')
    
    xx = np.arange(cube.shape[0])
    yy = np.arange(cube.shape[1])
    imgShape = (cube.shape[0],cube.shape[1])

    cubeVor = np.zeros(imgShape, dtype=float)
    errVor = np.zeros(imgShape, dtype=float)

    for nx in xx:
        for ny in yy:
            tmpcube = cube[nx,ny]
            # add a constant to the spectrum to make it above 0
            #print "old cube mean is %f " % tmpcube.mean()
            minFlux = tmpcube.mean() - (1.0 * tmpcube.std())
            #print "minflux is %f" % minFlux
            tmpcube += np.abs(minFlux)
            #print "new cube mean is %f " % tmpcube.mean()
            tmpcubeavg = tmpcube.mean()
            tmpsnr = np.abs(snrimg[nx,ny])
            # calc errors for tessellation based on the empirical
            # S/N already calculated
            tmperr = tmpcubeavg/tmpsnr
            cubeVor[nx,ny] = tmpcubeavg
            errVor[nx,ny] = tmperr

    # change NaN to 0
    errVor = np.nan_to_num(errVor)
    
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
    good = np.where((cube > 0) & (errors > 0) & ((cube/errors) >=2.))
    goodbad=np.zeros((cube.shape[0],cube.shape[1]),dtype=float)
    goodbad[good]=1.
    xx = good[1]
    yy = good[0]
    #pdb.set_trace()
    binNum, xNode, yNode, xBar, yBar, sn, nPixels, scale, pixSize = v2d.voronoi_2d_binning(xx, yy, cube[good], errors[good], targetSN, plot=1, quiet=0)

    py.clf()
    py.subplot(211)
    rnd = np.argsort(np.random.random(xNode.size))  # Randomize bin colors
    # added in flips to display in the same orientation as the data
    # divide by 20 to put in arcsec (0.05 arcsec per pixel)
    v2d._display_pixels(xx/20., np.flipud(yy)/20., rnd[binNum], pixSize/20., horflip=True)
    py.plot(xNode/20., yNode/20., '+w', scalex=False, scaley=False) # do not rescale after imshow()
    py.xlabel('R (arcsec)')
    py.ylabel('R (arcsec)')
    py.title('Map of Voronoi bins')
    
    py.subplot(212)
    rad = np.sqrt(xBar**2 + yBar**2)  # Use centroids, NOT generators
    w = nPixels == 1
    py.plot(rad[~w]/20., sn[~w], 'or', label='Voronoi bins')
    py.xlabel('R (arcsec)')
    py.ylabel('Bin S/N')
    py.axis([np.min(rad/20.), np.max(rad/20.), 0, np.max(sn)])  # x0, x1, y0, y1
    if np.sum(w) > 0:
        py.plot(rad[w]/20., sn[w], 'xb', label='single spaxels')
    py.axhline(targetSN)
    py.legend()
    py.pause(0.01)  # allow plot to appear in certain cases

    pdb.set_trace()
    np.savetxt(datadir+'voronoi_2d_binning_output.txt', np.column_stack([xx, yy, binNum]),
               fmt=b'%10.6f %10.6f %8i')

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

    for nb in range(binnum.max()):
        idx = np.where(binnum == nb)
        nx = xx[idx]
        ny = yy[idx]
        nbins = len(idx[0])
        tmpCube = np.sum(cube[nx,ny,:],axis=0)/nbins
        tmpErr = np.sqrt(np.sum(errors[nx,ny,:]**2,axis=0))
        newCube[nx,ny,:] = tmpCube
        newErr[nx,ny,:] = tmpErr

    pdb.set_trace()
    outfile = datadir + cuberoot + '_vorcube.fits'
    pyfits.writeto(outfile,newCube,header=hdr)
    pyfits.append(outfile,newErr)
    pyfits.append(outfile,quality)
    pyfits.append(outfile,nframes)
    
        
