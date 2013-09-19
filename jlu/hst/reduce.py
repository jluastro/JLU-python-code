import math
import atpy
import pylab as py
import numpy as np
from jlu.hst import starlists
from jlu.hst import images
from jlu.hst import astrometry as ast
import glob
from matplotlib import colors
from jlu.util import statsIter
import pdb
import os

def run_img2xym_acswfc(directory):
    """
    Run img2xym on ACS WFC data in the specified directory, <dir>. There is a specific
    directory structure that is expected within <dir>:

    00.DATA/
    01.XYM/
    """
    os.chdir(directory + '/01.XYM')

    program = 'img2xym_WFC'

    ## Arguments include:
    # hmin - dominance of peak. Something like the minimum SNR of a peak w.r.t. background.
    hmin = 5
    # fmin - minumum peak flux above the background
    fmin = 500
    # pmax - maximum flux allowed (absolute)
    pmax = 99999
    # psf - the library
    psf = codeDir + 'img2xym_WFC.09x10/PSFEFF.F814W.fits'

    ## Files to operate on:
    dataDir = '../00.DATA/*flt.fits'

    cmd_tmp = '{program} {hmin} {fmin} {pmax} {psf} {dataDir}'
    cmd = cmd_tmp.format(program=program, hmin=hmin, fmin=fmin, pmax=pmax, psf=psf,
                         dataDir=dataDir)

    try:
        os.system(cmd)
    finally:
        os.chdir('../../')
    

def run_img2xym_wfc3ir(year, filter, suffix=''):
    """
    Run img2xym on WFC3-IR data in the specified directory, <dir>. There is a specific
    directory structure that is expected within <dir>:

    00.DATA/
    01.XYM/
    """
    os.chdir('{0}_{1}{2}/01.XYM'.format(year, filter, suffix))

    program = 'img2xym_wfc3ir_pert'
    cmd_fmt = '{program} {hmin} {fmin} {pmax} {psf} {data}'

    ##########
    # First pass Arguments
    ##########
    ## Arguments include
    # hmin - dominance of peak. Something like the minimum SNR of a peak w.r.t. background.
    hmin1 = 8
    # fmin - minumum peak flux above the background
    fmin1 = 5000
    # pmax - maximum flux allowed (absolute)
    pmax1 = 20000
    # psf - the library
    filt_low = filter.lower()
    psf1 = codeDir + 'PSFs/psf_wfc3ir_' + filt_low + '.fits'

    ##########
    # Second pass Arguments
    ##########
    ## Arguments include
    # hmin - dominance of peak. Something like the minimum SNR of a peak w.r.t. background.
    hmin2 = 4
    # fmin - minumum peak flux above the background
    fmin2 = 2
    # pmax - maximum flux allowed (absolute)
    pmax2 = 1e15


    # Keep a log file
    _log = open('img2xym_log.txt', 'w')

    # Loop through the individual files and make a new PSF for each one.
    data_dir = '../00.DATA/'
    fits_files = glob.glob(data_dir + '*flt.fits')

    try:
        for fits in fits_files:
            fits_root = os.path.split(fits)[-1].replace('_flt.fits', '')
            psf2 = 'psf_wfc3ir_{0}_{1}.fits'.format(filt_low, fits_root)

            # Pass 1 with library PSF
            cmd = cmd_fmt.format(program=program, hmin=hmin1, fmin=fmin1, pmax=pmax1,
                                 psf=psf1, data=fits)
            _log.write(cmd + '\n')
            os.system(cmd)
            os.system('cp ' + fits_root + '_flt.xym ' + fits_root + '_flt.xym.old')

            # Copy over the new PSF
            cmd = 'mv psft.fits ' + psf2
            _log.write(cmd + '\n')
            os.system(cmd)

            # Pass 2 with image-specific PSF
            cmd = cmd_fmt.format(program=program, hmin=hmin2, fmin=fmin2, pmax=pmax2,
                                 psf=psf2, data=fits)
            _log.write(cmd + '\n')
            os.system(cmd)
            
    finally:
        os.chdir('../../')
        _log.close()
            
