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
import subprocess
import shutil
from jlu.util import fileUtil

codeDir = '~jlu/code/fortran/hst/'

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
            


def xym2mat(run, year, filt, suffix='', ref=None,
            ref_camera='c9', ref_mag='',
            camera='c9', mag='',
            clobber=False):
    """
    Run xym2mat data in the specified directory, <year>_<filter><suffix>. 
    There is a specific directory structure that is expected within <dir>:

    00.DATA/
    01.XYM/
    """
    os.chdir('{0}_{1}{2}/01.XYM'.format(year, filter, suffix))

    program = 'xym2mat'

    # Clean up some old files if necessaary
    if clobber:
        print('*** Deleting old IN.* TRANS.* MAT.* and MATCHUP.XYMEEE files\n')
        old_files = ['IN.xym2mat', 'TRANS.xym2mat']
        fileUtil.rmall(old_files)
        fileUtil.rmall( glob.glob('MAT.*') )

    # Fetch all the *.xym files we will be working with
    xym_files = glob.glob('*.xym')
    Nxym = len(xym_files)

    # Make the camera and magnitude limits into arrays if they aren't already.
    if (hasattr(camera, '__iter__') == False) or (len(camera) < Nxym):
        camera = np.repeat(camera, Nxym)

    if (hasattr(mag, '__iter__') == False) or (len(mag) < Nxym):
        mag = np.repeat(mag, Nxym)

    # Keep a log file
    _log = open('flystar_xym2mat.log', 'w')

    try: 
        ########## 
        # xym2mat -- IN.xym2mat
        ##########
        logger(_log, '*** Writing IN.xym2mat')
        
        f_in_mat = open('IN.xym2mat', 'w')

        in_fmt = '{idx:03d} "{ff}" {cam}'
        if mag[0] != None:
            in_fmt += ' "{mag}"'
        in_fmt += '\n'
    
        # Build the reference epoch line for xym2mat
        if ref == None:
            ref = xym_files[0]
            ref_camera = camera[0]
            ref_mag = mag[0]

        f_in_mat.write( in_fmt.format(idx=0, ff=ref, cam=ref_camera, mag=ref_mag) )
        for ii in range(Nxym):
            f_in_mat.write( in_fmt.format(idx=ii+1, ff=xym_files[ii], cam=camera[ii], mag=mag[ii]) )
        f_in_mat.close()

        ##########
        # xym2mat -- Run
        ##########
        logger(_log, '*** Calling xym2mat')

        # First call a little more open
        cmd = ['xym2mat', '20']
        logger(_log, '*** Running:')
        logger(_log, ' '.join(cmd))
        subprocess.call(cmd, stdout=_log, stderr=_log)

        # Second call a little more constrained
        cmd[1] = '22'
        logger(_log, 'Running:')
        logger(_log, ' '.join(cmd))
        subprocess.call(cmd, stdout=_log, stderr=_log)

        ##########
        # Copy files over.
        ##########
        logger('*** Copying files to <file>.{0}'.format(run))
        shutil.copy("IN.xym2mat", "IN.xym2mat." + run)
        shutil.copy("TRANS.xym2mat", "TRANS.xym2mat." + run)
        

    finally:
        os.chdir('../../')
        _log.close()



def xym2bar(run, year, filt, suffix='', Nepochs=1, camera='c9',
            zeropoint='z0', clobber=False, ref_xym2bar=None):
    """
    Run xym2bar on the contents of the directory given by:
    <year>_<filt><suffix>/
        00.DATA/
        01.XYM/
    """
    os.chdir('{0}_{1}{2}/01.XYM'.format(year, filt, suffix))

    program = 'xym2bar'

    _log = open('flystar_xym2bar.log', 'w')
    
    # Clean up some old files if necessaary
    if clobber:
        logger(_log, '*** Deleting old IN.* TRANS.* MAT.* and MATCHUP.XYMEEE files')
        old_files = ['IN.xym2bar',
                     'TRANS.xym2bar',
                     'MATCHUP.XYMEEE']
        fileUtil.rmall(old_files)

    # Fetch all the *.xym files we will be working with
    xym_files = glob.glob('*.xym')
    Nxym = len(xym_files)

    # Make the camera and magnitude limits into arrays if they aren't already.
    if (hasattr(camera, '__iter__') == False) or (len(camera) < Nxym):
        camera = np.repeat(camera, Nxym)

    if (hasattr(zeropoint, '__iter__') == False) or (len(zeropoint) < Nxym):
        zeropoint = np.repeat(zeropoint, Nxym)

    try: 
        ##########
        # xym2bar -- Make IN.xym2bar
        ##########
        logger(_log, '*** Writing IN.xym2bar')

        in_fmt = '{idx:03d} "{ff}" {cam} {zp}\n'
        f_in_bar = open('IN.xym2bar', 'w')
        if ref_xym2bar != None:
            f_in_bar.write( in_fmt.format(idx=0, ff=ref_xym2bar, cam='', zp='z0') )
        for ii in range(Nxym):
            f_in_bar.write( in_fmt.format(idx=ii+1, ff=xym_files[ii], cam=camera[ii],
                                          zp=zeropoint[ii]) )
        f_in_bar.close()

        ##########
        # xym2bar -- Run
        ##########
        cmd = ['xym2bar', str(Nepochs)]
        if ref_xym2bar != None:
            cmd += ['I']
            
        logger(_log, '*** Running:  \n' + ' '.join(cmd))

        subprocess.call(cmd, stdout=_log, stderr=_log)

        ##########
        # Copy files over.
        ##########
        logger('*** Copying files to <file>.{0}'.format(run))
        shutil.copy("IN.xym2bar", "IN.xym2bar." + run)
        shutil.copy("TRANS.xym2bar", "TRANS.xym2bar." + run)
        shutil.copy("MATCHUP.XYMEEE", "MATCHUP.XYMEEE." + run)
        

    finally:
        os.chdir('../../')
        _log.close()


    return
