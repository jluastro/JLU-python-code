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
            minFlux = tmpcube.mean() - (3.0 * tmpcube.std())
            tmpcube += minFlux
            tmpcubeavg = tmpcube.mean()
            tmpsnr = snrimg[nx,ny]
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

    #xx = np.arange(cube.shape[0])
    #yy = np.arange(cube.shape[1])
    good = np.where(cube > 0)
    xx = good[1]
    yy = good[0]

    binNum, xNode, yNode, xBar, yBar, sn, nPixels, scale = v2d.voronoi_2d_binning(xx, yy, cube[good], errors[good], targetSN, plot=1, quiet=0)

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
    
        
