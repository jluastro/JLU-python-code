"""
Research Note: HST Data Reduction (2024_03_26)
Working Directory: /u/jlu/data/Wd1/hst/reduce_2024_03_26/
"""
import os
import pdb
import warnings
import math
import copy
import glob
import shutil
import subprocess
from weakref import ref
import pylab as py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from flystar.startables import StarTable
# from flystar.fit_velocity import linear_fit
from flystar.motion_model import Linear
from jlu import photometry as photo
from matplotlib import colors
from tqdm import tqdm
from jlu.util import statsIter
from jlu.util import fileUtil
from hst_flystar import astrometry as ast
from hst_flystar import photometry
from hst_flystar import reduce as flystar
from hst_flystar import starlists
from hst_flystar import completeness as comp
from hst_flystar.starlists import read_matchup
from flystar import starlists as fly_starlists
import astropy.table
from astropy.table import Table
from astropy.table import Column
from jlu.astrometry import align
from scipy.stats import binned_statistic, chi2
from scipy.optimize import least_squares
from sklearn.mixture import GaussianMixture
from astropy.stats import sigma_clip
from matplotlib import colors as mcolors
from matplotlib import colorbar
from tqdm import tqdm

user_path = os.path.expanduser('~')
work_dir = f'{user_path}/Westerlund1/data/reduce_2024_10_23/'
# code_dir = '/u/jlu/code/fortran/hst/'

# Load this variable with outputs from calc_years()
years = {
    '2005_F814W':  2005.485,
    '2010_F125W':  2010.652,
    '2010_F139M':  2010.652,
    '2010_F160W':  2010.652,
    '2013_F160W':  2013.199,
    '2013_F160Ws': 2013.202,
    '2015_F160W':  2015.148,
    '2015_F160Ws': 2015.149,
    '2024_F160W':  2024.109,
    '2024_F160Ws': 2024.113
}

topStars = [{'name':'wd1_00001', 'x': 1838.81, 'y':  568.98, 'm160': -10.2},
            {'name':'wd1_00002', 'x': 3396.91, 'y': 1389.27, 'm160': -10.1},
            {'name':'wd1_00003', 'x': 3824.63, 'y': 3347.88, 'm160': -10.4},
            {'name':'wd1_00004', 'x':  717.67, 'y': 3033.63, 'm160':  -9.9},
            {'name':'wd1_00005', 'x': 2030.72, 'y': 2683.57, 'm160':  -9.7},
            {'name':'wd1_00006', 'x':  676.98, 'y':  663.25, 'm160':  -9.6}]


def run_img2xym_acswfc(directory):
    """
    Run img2xym on ACS WFC data in the specified directory, <dir>. There is a specific
    directory structure that is expected within <dir>:

    00.DATA/
    01.XYM/
    """
    os.chdir(directory + '/01.XYM')

    program = 'img2xymrduv'

    ## Arguments include:
    # hmin - dominance of peak. Something like the minimum SNR of a peak w.r.t. background.
    hmin = 5
    # fmin - minumum peak flux above the background
    fmin = 500
    # pmax - maximum flux allowed (absolute)
    pmax = 99999
    # psf - the library
    # psf = code_dir + 'PSFs/PSFSTD_ACSWFC_F814W_SM3.fits'
    psf = work_dir + '/PSFSTD_ACSWFC_F814W_SM3.fits'

    ## Files to operate on:
    dataDir = '../00.DATA/*flt.fits'

    cmd_tmp = '{program} {hmin} {fmin} {pmax} {psf} {dataDir}'
    cmd = cmd_tmp.format(program=program, hmin=hmin, fmin=fmin, pmax=pmax, psf=psf,
                         dataDir=dataDir)

    try:
        os.system(cmd)
    finally:
        os.chdir('../../')
    

def xym_acswfc_pass1():
    """
    Match and align the 3 exposures. Make a new star list (requiring 2 out of
    3 images). Also make sure the new starlist has only positive pxel values.
    """
    year = '2005'
    filt = 'F814W'

    xym_dir = '{0}_{1}/01.XYM/'.format(year, filt)

    flystar.xym2mat('ref1', year, filt, camera='f5 c5', mag='m-13.5,-10', clobber=True)
    flystar.xym2bar('ref1', year, filt, camera='f5 c5', Nepochs=2, clobber=True)

    starlists.make_matchup_positive(xym_dir + 'MATCHUP.XYMEEE.ref1')
    
def xym_acswfc_pass2():
    """
    Re-do the alignment with the new positive master file. Edit IN.xym2mat to 
    change 00 epoch to use the generated matchup file with f5, c0 
    (MATCHUP.XYMEEE.01.all.positive). Make sure to remove all the old MAT.* and 
    TRANS.xym2mat files because there is a big shift in the transformation from 
    the first time we ran it.
    """
    year = '2005'
    filt = 'F814W'

    xym_dir = '{0}_{1}/01.XYM/'.format(year, filt)

    flystar.xym2mat('ref2', year, filt, camera='f5 c5', mag='m-13.5,-10',
                    ref='MATCHUP.XYMEEE.ref1.positive', ref_mag='m-13.5,-10',
                    ref_camera='f5 c0', clobber=True)
    flystar.xym2bar('ref2', year, filt, Nepochs=2, zeropoint='', 
                    camera='f5 c5', clobber=True)

def plot_cmd_one_pass(reread=False):
    """
    Plot CMDs for all filter combinations from the one-pass analysis.
    This is just to get a sense of the magnitude differences.
    """
    if reread:
        # Read in the text files and save as FITS tables for speed.
        t2005_814 = starlists.read_matchup('MATCHUP.XYMEEE.F814W.2005.ref5')
        t2010_160 = starlists.read_matchup('MATCHUP.XYMEEE.F160W.2010.ref5')
        t2010_139 = starlists.read_matchup('MATCHUP.XYMEEE.F139M.2010.ref5')
        t2010_125 = starlists.read_matchup('MATCHUP.XYMEEE.F125W.2010.ref5')
        t2013_160 = starlists.read_matchup('MATCHUP.XYMEEE.F160W.2013.ref5')
        t2013_160s = starlists.read_matchup('MATCHUP.XYMEEE.F160Ws.2013.ref5')
        t2015_160 = starlists.read_matchup('MATCHUP.XYMEEE.F160W.2015.ref5')
        t2015_160s = starlists.read_matchup('MATCHUP.XYMEEE.F160Ws.2015.ref5')

        t2005_814.write('MATCHUP.XYMEEE.F814W.2005.ref5.fits', overwrite=True)
        t2010_160.write('MATCHUP.XYMEEE.F160W.2010.ref5.fits', overwrite=True)
        t2010_139.write('MATCHUP.XYMEEE.F139M.2010.ref5.fits', overwrite=True)
        t2010_125.write('MATCHUP.XYMEEE.F125W.2010.ref5.fits', overwrite=True)
        t2013_160.write('MATCHUP.XYMEEE.F160W.2013.ref5.fits', overwrite=True)
        t2013_160s.write('MATCHUP.XYMEEE.F160Ws.2013.ref5.fits', overwrite=True)
        t2015_160.write('MATCHUP.XYMEEE.F160W.2015.ref5.fits', overwrite=True)
        t2015_160s.write('MATCHUP.XYMEEE.F160Ws.2015.ref5.fits', overwrite=True)
    else:
        # Read in the FITS versions of the tables.
        t2005_814 = Table.read('MATCHUP.XYMEEE.F814W.2005.ref5.fits')
        t2010_160 = Table.read('MATCHUP.XYMEEE.F160W.2010.ref5.fits')
        t2010_139 = Table.read('MATCHUP.XYMEEE.F139M.2010.ref5.fits')
        t2010_125 = Table.read('MATCHUP.XYMEEE.F125W.2010.ref5.fits')
        t2013_160 = Table.read('MATCHUP.XYMEEE.F160W.2013.ref5.fits')
        t2013_160s = Table.read('MATCHUP.XYMEEE.F160Ws.2013.ref5.fits')
        t2015_160 = Table.read('MATCHUP.XYMEEE.F160W.2015.ref5.fits')
        t2015_160s = Table.read('MATCHUP.XYMEEE.F160Ws.2015.ref5.fits')

    # Trim down to only those stars with well measured positions. 
    good = np.where((t2005_814['xe'] < 0.1)  & (t2005_814['ye'] < 0.1) &
                    (t2010_160['xe'] < 0.1)  & (t2010_160['ye'] < 0.1) & 
                    (t2010_139['xe'] < 0.1)  & (t2010_139['ye'] < 0.1) & 
                    (t2010_125['xe'] < 0.1)  & (t2010_125['ye'] < 0.1) & 
                    (t2013_160['xe'] < 0.1)  & (t2013_160['ye'] < 0.1) & 
                    (t2013_160s['xe'] < 0.1) & (t2013_160s['ye'] < 0.1) &
                    (t2015_160s['xe'] < 0.1) & (t2015_160s['ye'] < 0.1) &
                    (t2015_160s['xe'] < 0.1) & (t2015_160s['ye'] < 0.1))[0]

    t2005_814 = t2005_814[good]
    t2010_160 = t2010_160[good]
    t2010_139 = t2010_139[good]
    t2010_125 = t2010_125[good]
    t2013_160 = t2013_160[good]
    t2013_160s = t2013_160s[good]
    t2015_160 = t2015_160[good]
    t2015_160s = t2015_160s[good]

    # Put all the tables in a list so that we can loop through
    # the different combinations.
    t_all = [t2005_814, t2010_125, t2010_139, t2010_160, t2013_160, t2013_160s, t2015_160, t2015_160s]
    label = ['F814W 2005', 'F125W 2010', 'F139M 2010', 'F160W 2010',
             'F160W 2013', 'F160Ws 2013', 'F160W 2015', 'F160Ws 2015']
        
    ##########
    # CMDs
    ##########
    for ii in range(len(t_all)):
        for jj in range(ii+1, len(t_all)):
            t1 = t_all[ii]
            t2 = t_all[jj]
            
            plt.clf()
            plt.plot(t1['m'] - t2['m'], t1['m'], 'k.')
            plt.xlabel(label[ii] + ' - ' + label[jj])
            plt.ylabel(label[ii])
            ax = plt.gca()
            ax.invert_yaxis()

            outfile = 'cmd_'
            outfile += label[ii].replace(' ', '_')
            outfile += '_vs_'
            outfile += label[jj].replace(' ', '_')
            outfile += '.png'
            plt.savefig(outfile)

    return


def plot_quiver_one_pass(path, reread=False):
    """
    Plot a quiver vector plot between F814W in 2005 and F160W in 2010 and 2013.
    This is just to check that we don't hae any gross flows between the two cameras.
    """
    
    ref5_paths = [
        f'{path}/MATCHUP.XYMEEE.F814W.2005.ref5',
        f'{path}/MATCHUP.XYMEEE.F160W.2010.ref5',
        f'{path}/MATCHUP.XYMEEE.F160W.2013.ref5',
        f'{path}/MATCHUP.XYMEEE.F160W.2015.ref5',
        f'{path}/MATCHUP.XYMEEE.F160W.2024.ref5',
    ]
    
    if reread:
        t2005 = starlists.read_matchup(ref5_paths[0])
        t2010 = starlists.read_matchup(ref5_paths[1])
        t2013 = starlists.read_matchup(ref5_paths[2])
        t2015 = starlists.read_matchup(ref5_paths[3])
        t2024 = starlists.read_matchup(ref5_paths[4])

        for ref5_path, t in zip(ref5_paths, [t2005, t2010, t2013, t2015, t2024]):
            if not os.path.exists(f'{ref5_path}.fits'):
                t.write(f'{ref5_path}.fits')
    else:
        t2005 = Table.read(f'{ref5_paths[0]}.fits')
        t2010 = Table.read(f'{ref5_paths[1]}.fits')
        t2013 = Table.read(f'{ref5_paths[2]}.fits')
        t2015 = Table.read(f'{ref5_paths[3]}.fits')
        t2024 = Table.read(f'{ref5_paths[4]}.fits')

    good = np.where((t2005['m'] < -8) & (t2010['m'] < -8) & (t2013['m'] < -8) & (t2015['m'] < -8) & (t2024['m'] < -8) &
                    (t2005['xe'] < 0.05) & (t2010['xe'] < 0.05) & (t2013['xe'] < 0.05) & (t2015['xe'] < 0.05) & (t2024['xe'] < 0.05) &
                    (t2005['ye'] < 0.05) & (t2010['ye'] < 0.05) & (t2013['ye'] < 0.05) & (t2015['ye'] < 0.05) & (t2024['ye'] < 0.05) &
                    (t2005['me'] < 0.05) & (t2010['me'] < 0.05) & (t2013['me'] < 0.05) & (t2015['me'] < 0.05) & (t2024['me'] < 0.05))[0]

    g2005 = t2005[good]
    g2010 = t2010[good]
    g2013 = t2013[good]
    g2015 = t2015[good]
    g2024 = t2024[good]

    # Compute all pairwise differences (mas)
    dx_05_10 = (g2010['x'] - g2005['x']) * ast.scale['WFC'] * 1e3
    dy_05_10 = (g2010['y'] - g2005['y']) * ast.scale['WFC'] * 1e3

    dx_05_13 = (g2013['x'] - g2005['x']) * ast.scale['WFC'] * 1e3
    dy_05_13 = (g2013['y'] - g2005['y']) * ast.scale['WFC'] * 1e3

    dx_05_15 = (g2015['x'] - g2005['x']) * ast.scale['WFC'] * 1e3
    dy_05_15 = (g2015['y'] - g2005['y']) * ast.scale['WFC'] * 1e3

    dx_05_24 = (g2024['x'] - g2005['x']) * ast.scale['WFC'] * 1e3
    dy_05_24 = (g2024['y'] - g2005['y']) * ast.scale['WFC'] * 1e3

    dx_10_13 = (g2013['x'] - g2010['x']) * ast.scale['WFC'] * 1e3
    dy_10_13 = (g2013['y'] - g2010['y']) * ast.scale['WFC'] * 1e3

    dx_10_15 = (g2015['x'] - g2010['x']) * ast.scale['WFC'] * 1e3
    dy_10_15 = (g2015['y'] - g2010['y']) * ast.scale['WFC'] * 1e3

    dx_10_24 = (g2024['x'] - g2010['x']) * ast.scale['WFC'] * 1e3
    dy_10_24 = (g2024['y'] - g2010['y']) * ast.scale['WFC'] * 1e3

    dx_13_15 = (g2015['x'] - g2013['x']) * ast.scale['WFC'] * 1e3
    dy_13_15 = (g2015['y'] - g2013['y']) * ast.scale['WFC'] * 1e3

    dx_13_24 = (g2024['x'] - g2013['x']) * ast.scale['WFC'] * 1e3
    dy_13_24 = (g2024['y'] - g2013['y']) * ast.scale['WFC'] * 1e3

    dx_15_24 = (g2024['x'] - g2015['x']) * ast.scale['WFC'] * 1e3
    dy_15_24 = (g2024['y'] - g2015['y']) * ast.scale['WFC'] * 1e3

    # Keep only stars with all differences < 20 mas in all pairs
    diffs = [
        dx_05_10, dy_05_10, dx_05_13, dy_05_13, dx_05_15, dy_05_15, dx_05_24, dy_05_24,
        dx_10_13, dy_10_13, dx_10_15, dy_10_15, dx_10_24, dy_10_24,
        dx_13_15, dy_13_15, dx_13_24, dy_13_24,
        dx_15_24, dy_15_24
    ]
    small = np.where(np.all([np.abs(d) < 20 for d in diffs], axis=0))[0]

    print(len(g2005), len(small), len(dx_05_10))
    g2005 = g2005[small]
    g2010 = g2010[small]
    g2013 = g2013[small]
    g2015 = g2015[small]
    g2024 = g2024[small]
    dx_05_10 = dx_05_10[small]
    dx_05_13 = dx_05_13[small]
    dx_05_15 = dx_05_15[small]
    dx_05_24 = dx_05_24[small]
    dx_10_13 = dx_10_13[small]
    dx_10_15 = dx_10_15[small]
    dx_10_24 = dx_10_24[small]
    dx_13_15 = dx_13_15[small]
    dx_13_24 = dx_13_24[small]
    dx_15_24 = dx_15_24[small]
    
    dy_05_10 = dy_05_10[small]
    dy_05_13 = dy_05_13[small]
    dy_05_15 = dy_05_15[small]
    dy_05_24 = dy_05_24[small]
    dy_10_13 = dy_10_13[small]
    dy_10_15 = dy_10_15[small]
    dy_10_24 = dy_10_24[small]
    dy_13_15 = dy_13_15[small]
    dy_13_24 = dy_13_24[small]
    dy_15_24 = dy_15_24[small]
    
    print(len(g2005), len(small), len(dx_05_10))

    qscale = 1e2
        
    plt.clf()
    q = plt.quiver(g2005['x'], g2005['y'], dx_05_10, dy_05_10, scale=qscale)
    plt.quiverkey(q, 0.95, 0.95, 5, '5 mas', color='red', labelcolor='red')
    plt.title('2010 - 2005')
    plt.savefig(f'{path}/vec_diff_ref5_05_10.png')

    plt.clf()
    q = plt.quiver(g2005['x'], g2005['y'], dx_05_13, dy_05_13, scale=qscale)
    plt.quiverkey(q, 0.95, 0.95, 5, '5 mas', color='red', labelcolor='red')
    plt.title('2013 - 2005')
    plt.savefig(f'{path}/vec_diff_ref5_05_13.png')

    plt.clf()
    q = plt.quiver(g2005['x'], g2005['y'], dx_05_15, dy_05_15, scale=qscale)
    plt.quiverkey(q, 0.95, 0.95, 5, '5 mas', color='red', labelcolor='red')
    plt.title('2015 - 2005')
    plt.savefig(f'{path}/vec_diff_ref5_05_15.png')
    
    plt.clf()
    q = plt.quiver(g2005['x'], g2005['y'], dx_05_24, dy_05_24, scale=qscale)
    plt.quiverkey(q, 0.95, 0.95, 5, '5 mas', color='red', labelcolor='red')
    plt.title('2024 - 2005')
    plt.savefig(f'{path}/vec_diff_ref5_05_24.png')

    plt.clf()
    q = plt.quiver(g2005['x'], g2005['y'], dx_10_13, dy_10_13, scale=qscale)
    plt.quiverkey(q, 0.95, 0.95, 5, '5 mas', color='red', labelcolor='red')
    plt.title('2013 - 2010')
    plt.savefig(f'{path}/vec_diff_ref5_10_13.png')

    plt.clf()
    q = plt.quiver(g2005['x'], g2005['y'], dx_10_15, dy_10_15, scale=qscale)
    plt.quiverkey(q, 0.95, 0.95, 5, '5 mas', color='red', labelcolor='red')
    plt.title('2015 - 2010')
    plt.savefig(f'{path}/vec_diff_ref5_10_15.png')

    plt.clf()
    q = plt.quiver(g2005['x'], g2005['y'], dx_10_24, dy_10_24, scale=qscale)
    plt.quiverkey(q, 0.95, 0.95, 5, '5 mas', color='red', labelcolor='red')
    plt.title('2024 - 2010')
    plt.savefig(f'{path}/vec_diff_ref5_10_24.png')

    plt.clf()
    q = plt.quiver(g2005['x'], g2005['y'], dx_13_15, dy_13_15, scale=qscale)
    plt.quiverkey(q, 0.95, 0.95, 5, '5 mas', color='red', labelcolor='red')
    plt.title('2015 - 2013')
    plt.savefig(f'{path}/vec_diff_ref5_13_15.png')

    plt.clf()
    q = plt.quiver(g2005['x'], g2005['y'], dx_13_24, dy_13_24, scale=qscale)
    plt.quiverkey(q, 0.95, 0.95, 5, '5 mas', color='red', labelcolor='red')
    plt.title('2024 - 2013')
    plt.savefig(f'{path}/vec_diff_ref5_13_24.png')
    
    plt.clf()
    q = plt.quiver(g2005['x'], g2005['y'], dx_15_24, dy_15_24, scale=qscale)
    plt.quiverkey(q, 0.95, 0.95, 5, '5 mas', color='red', labelcolor='red')
    plt.title('2024 - 2015')
    plt.savefig(f'{path}/vec_diff_ref5_15_24.png')


    ##########
    # VPD
    ##########
    plt.clf()
    plt.plot(dx_05_10, dy_05_10, 'k.', ms=2)
    lim = 10
    plt.axis([-lim, lim, -lim, lim])
    plt.xlabel('X Proper Motion (mas)')
    plt.ylabel('Y Proper Motion (mas)')
    plt.title('2010 - 2005')
    plt.savefig(f'{path}/pm_diff_ref5_05_10.png')
    plt.close()

    plt.plot(dx_05_13, dy_05_13, 'k.', ms=2)
    plt.axis([-lim, lim, -lim, lim])
    plt.xlabel('X Proper Motion (mas)')
    plt.ylabel('Y Proper Motion (mas)')
    plt.title('2013 - 2005')
    plt.savefig(f'{path}/pm_diff_ref5_05_13.png')
    plt.close()

    plt.plot(dx_05_15, dy_05_15, 'k.', ms=2)
    plt.axis([-lim, lim, -lim, lim])
    plt.xlabel('X Proper Motion (mas)')
    plt.ylabel('Y Proper Motion (mas)')
    plt.title('2015 - 2005')
    plt.savefig(f'{path}/pm_diff_ref5_05_15.png')
    plt.close()

    plt.plot(dx_05_24, dy_05_24, 'k.', ms=2)
    plt.axis([-lim, lim, -lim, lim])
    plt.xlabel('X Proper Motion (mas)')
    plt.ylabel('Y Proper Motion (mas)')
    plt.title('2024 - 2005')
    plt.savefig(f'{path}/pm_diff_ref5_05_24.png')
    plt.close()

    plt.plot(dx_10_13, dy_10_13, 'k.', ms=2)
    plt.axis([-lim, lim, -lim, lim])
    plt.xlabel('X Proper Motion (mas)')
    plt.ylabel('Y Proper Motion (mas)')
    plt.title('2013 - 2010')
    plt.savefig(f'{path}/pm_diff_ref5_10_13.png')
    plt.close()

    plt.plot(dx_10_15, dy_10_15, 'k.', ms=2)
    plt.axis([-lim, lim, -lim, lim])
    plt.xlabel('X Proper Motion (mas)')
    plt.ylabel('Y Proper Motion (mas)')
    plt.title('2015 - 2010')
    plt.savefig(f'{path}/pm_diff_ref5_10_15.png')
    plt.close()

    plt.plot(dx_10_24, dy_10_24, 'k.', ms=2)
    plt.axis([-lim, lim, -lim, lim])
    plt.xlabel('X Proper Motion (mas)')
    plt.ylabel('Y Proper Motion (mas)')
    plt.title('2024 - 2010')
    plt.savefig(f'{path}/pm_diff_ref5_10_24.png')
    plt.close()

    plt.plot(dx_13_15, dy_13_15, 'k.', ms=2)
    plt.axis([-lim, lim, -lim, lim])
    plt.xlabel('X Proper Motion (mas)')
    plt.ylabel('Y Proper Motion (mas)')
    plt.title('2015 - 2013')
    plt.savefig(f'{path}/pm_diff_ref5_13_15.png')
    plt.close()

    plt.plot(dx_13_24, dy_13_24, 'k.', ms=2)
    plt.axis([-lim, lim, -lim, lim])
    plt.xlabel('X Proper Motion (mas)')
    plt.ylabel('Y Proper Motion (mas)')
    plt.title('2024 - 2013')
    plt.savefig(f'{path}/pm_diff_ref5_13_24.png')
    plt.close()

    plt.plot(dx_15_24, dy_15_24, 'k.', ms=2)
    plt.axis([-lim, lim, -lim, lim])
    plt.xlabel('X Proper Motion (mas)')
    plt.ylabel('Y Proper Motion (mas)')
    plt.title('2024 - 2015')
    plt.savefig(f'{path}/pm_diff_ref5_15_24.png')
    plt.close()

    print('2010 - 2005')
    print('   dx = {dx:6.2f} ± {dxe:6.2f} mas'.format(dx=dx_05_10.mean(), dxe=dx_05_10.std()))
    print('   dy = {dy:6.2f} ± {dye:6.2f} mas'.format(dy=dy_05_10.mean(), dye=dy_05_10.std()))

    print('2013 - 2005')
    print('   dx = {dx:6.2f} ± {dxe:6.2f} mas'.format(dx=dx_05_13.mean(), dxe=dx_05_13.std()))
    print('   dy = {dy:6.2f} ± {dye:6.2f} mas'.format(dy=dy_05_13.mean(), dye=dy_05_13.std()))

    print('2015 - 2005')
    print('   dx = {dx:6.2f} ± {dxe:6.2f} mas'.format(dx=dx_05_15.mean(), dxe=dx_05_15.std()))
    print('   dy = {dy:6.2f} ± {dye:6.2f} mas'.format(dy=dy_05_15.mean(), dye=dy_05_15.std()))
    
    print('2024 - 2005')
    print('   dx = {dx:6.2f} ± {dxe:6.2f} mas'.format(dx=dx_05_24.mean(), dxe=dx_05_24.std()))
    print('   dy = {dy:6.2f} ± {dye:6.2f} mas'.format(dy=dy_05_24.mean(), dye=dy_05_24.std()))
    
    print('2013 - 2010')
    print('   dx = {dx:6.2f} ± {dxe:6.2f} mas'.format(dx=dx_10_13.mean(), dxe=dx_10_13.std()))
    print('   dy = {dy:6.2f} ± {dye:6.2f} mas'.format(dy=dy_10_13.mean(), dye=dy_10_13.std()))

    print('2015 - 2010')
    print('   dx = {dx:6.2f} ± {dxe:6.2f} mas'.format(dx=dx_10_15.mean(), dxe=dx_10_15.std()))
    print('   dy = {dy:6.2f} ± {dye:6.2f} mas'.format(dy=dy_10_15.mean(), dye=dy_10_15.std()))

    print('2024 - 2010')
    print('   dx = {dx:6.2f} ± {dxe:6.2f} mas'.format(dx=dx_10_24.mean(), dxe=dx_10_24.std()))
    print('   dy = {dy:6.2f} ± {dye:6.2f} mas'.format(dy=dy_10_24.mean(), dye=dy_10_24.std()))

    print('2015 - 2013')
    print('   dx = {dx:6.2f} ± {dxe:6.2f} mas'.format(dx=dx_13_15.mean(), dxe=dx_13_15.std()))
    print('   dy = {dy:6.2f} ± {dye:6.2f} mas'.format(dy=dy_13_15.mean(), dye=dy_13_15.std()))
    
    print('2024 - 2013')
    print('   dx = {dx:6.2f} ± {dxe:6.2f} mas'.format(dx=dx_13_24.mean(), dxe=dx_13_24.std()))
    print('   dy = {dy:6.2f} ± {dye:6.2f} mas'.format(dy=dy_13_24.mean(), dye=dy_13_24.std()))
    
    print('2024 - 2015')
    print('   dx = {dx:6.2f} ± {dxe:6.2f} mas'.format(dx=dx_15_24.mean(), dxe=dx_15_24.std()))
    print('   dy = {dy:6.2f} ± {dye:6.2f} mas'.format(dy=dy_15_24.mean(), dye=dy_15_24.std()))


def make_refClust_catalog():
    """
    Make mat_all_good.fits where we matchup all the refClust starlists and
    get a preliminary look at the VPD and CMD. Do this on a per-position basis.
    """
    for ii in range(1, 4+1):
        pos_dir = 'match_refClust_pos{0}'.format(ii)
        os.chdir(work_dir + '/03.MAT_POS/')
        
        fileUtil.mkdir(pos_dir)
        shutil.copy('2005_F814W/01.XYM/MATCHUP.XYMEEE.F814W.2005.refClust', pos_dir)
        os.chdir(work_dir + '/03.MAT_POS/' + pos_dir)
        
        # Call xym1mat to match against F814W for all combos
        f814w = 'MATCHUP.XYMEEE.F814W.2005.refClust'

        years = ['2010', '2010', '2010', '2013']
        filts = ['F160W', 'F139M', 'F125W', 'F160W']

        for jj in range(len(years)):
            match = 'MATCHUP.XYMEEE.'
            match += '{0}.{1}.pos{2}.refClust'.format(filts[jj], years[jj], ii)
            out_root = 'mat_2005_{0}_{1}_pos{2}'.format(years[jj], filts[jj], ii)

            match_dir = '../{0}_{1}_pos{2}/01.XYM/'.format(years[jj], filts[jj], ii)
            shutil.copy(match_dir + match, './')
            
            cmd = ['xym1mat', f814w, match,
                   out_root + '.mat', out_root + '.xym1', out_root + '.lnk',
                   '14', '0.99']
            subprocess.call(cmd)

        # Process the output.
        fmt = '{name:8s}  {x:10.4f} {y:10.4f} {m:8.4f}  '
        fmt += '{xe:7.4f} {ye:7.4f} {me:7.4f}  '
        fmt += '{n:2d}\n'
        
        # First 2005 (treated differently)
        tab = starlists.read_matchup(f814w)
        _final_2005 = open('final_2005_814_pos{0}.txt'.format(ii), 'w')
        for tt in tab:
            _final_2005.write(fmt.format(name=tt[13], x=tt[0], y=tt[1], m=tt[2],
                                         xe=tt[3], ye=tt[4], me=tt[5], n=tt[10]))
        _final_2005.close()

        # All 2010 and 2013 data
        for jj in range(len(years)):
            suffix = '{0}_{1}_pos{2}'.format(years[jj], filts[jj], ii)
            _final = open('final_' + suffix + '.txt', 'w')
            tab = Table.read('mat_2005_' + suffix + '.lnk', format='ascii')

            for tt in tab:
                _final.write(fmt.format(name=tt[30], x=tt[19], y=tt[20], m=tt[21],
                                        xe=tt[22], ye=tt[23], me=tt[24], n=tt[29]))
            _final.close()
                                        
    return

def plot_quiver_one_pass_refClust(path):
    """
    Plot a quiver vector plot between F814W in 2005 and F160W in 2010 and 2013 and 2015.
    This is just to check that we don't hae any gross flows between the two cameras.

    See notes from "HST Data Reduction (2014-06) for creation of the FITS table.
    """
    t = Table.read(f'{path}/mat_all_good.fits')
    t.rename_column('col02', 'x_2005')
    t.rename_column('col03', 'y_2005')
    t.rename_column('col04', 'm_2005')
    t.rename_column('col05', 'xe_2005')
    t.rename_column('col06', 'ye_2005')
    t.rename_column('col07', 'me_2005')

    t.rename_column('col10', 'x_2010')
    t.rename_column('col11', 'y_2010')
    t.rename_column('col12', 'm_2010')
    t.rename_column('col13', 'xe_2010')
    t.rename_column('col14', 'ye_2010')
    t.rename_column('col15', 'me_2010')

    t.rename_column('col18', 'x_2013')
    t.rename_column('col19', 'y_2013')
    t.rename_column('col20', 'm_2013')
    t.rename_column('col21', 'xe_2013')
    t.rename_column('col22', 'ye_2013')
    t.rename_column('col23', 'me_2013')

    t.rename_column('col26', 'x_2015')
    t.rename_column('col27', 'y_2015')
    t.rename_column('col28', 'm_2015')
    t.rename_column('col29', 'xe_2015')
    t.rename_column('col30', 'ye_2015')
    t.rename_column('col31', 'me_2015')
    
    t.rename_column('col34', 'x_2024')
    t.rename_column('col35', 'y_2024')
    t.rename_column('col36', 'm_2024')
    t.rename_column('col37', 'xe_2024')
    t.rename_column('col38', 'ye_2024')
    t.rename_column('col39', 'me_2024')
    
    good = np.where(
        (t['m_2005'] < -8) & (t['m_2010'] < -8) & (t['m_2013'] < -8) & (t['m_2015'] < -8) & (t['m_2024'] < -8) &
        (t['xe_2010'] < 0.05) & (t['xe_2013'] < 0.05) & (t['xe_2015'] < 0.05) & (t['xe_2024'] < 0.05) &
        (t['ye_2010'] < 0.05) & (t['ye_2013'] < 0.05) & (t['ye_2015'] < 0.05) & (t['ye_2024'] < 0.05) &
        (t['me_2010'] < 0.05) & (t['me_2013'] < 0.05) & (t['me_2015'] < 0.05) & (t['me_2024'] < 0.05)
    )[0]

    g = t[good]

    dx_05_10 = (g['x_2010'] - g['x_2005']) * ast.scale['WFC'] * 1e3
    dy_05_10 = (g['y_2010'] - g['y_2005']) * ast.scale['WFC'] * 1e3

    dx_05_13 = (g['x_2013'] - g['x_2005']) * ast.scale['WFC'] * 1e3
    dy_05_13 = (g['y_2013'] - g['y_2005']) * ast.scale['WFC'] * 1e3

    dx_05_15 = (g['x_2015'] - g['x_2005']) * ast.scale['WFC'] * 1e3
    dy_05_15 = (g['y_2015'] - g['y_2005']) * ast.scale['WFC'] * 1e3

    dx_05_24 = (g['x_2024'] - g['x_2005']) * ast.scale['WFC'] * 1e3
    dy_05_24 = (g['y_2024'] - g['y_2005']) * ast.scale['WFC'] * 1e3
    
    dx_10_13 = (g['x_2013'] - g['x_2010']) * ast.scale['WFC'] * 1e3
    dy_10_13 = (g['y_2013'] - g['y_2010']) * ast.scale['WFC'] * 1e3

    dx_10_15 = (g['x_2015'] - g['x_2010']) * ast.scale['WFC'] * 1e3
    dy_10_15 = (g['y_2015'] - g['y_2010']) * ast.scale['WFC'] * 1e3
    
    dx_10_24 = (g['x_2024'] - g['x_2010']) * ast.scale['WFC'] * 1e3
    dy_10_24 = (g['y_2024'] - g['y_2010']) * ast.scale['WFC'] * 1e3

    dx_13_15 = (g['x_2015'] - g['x_2013']) * ast.scale['WFC'] * 1e3
    dy_13_15 = (g['y_2015'] - g['y_2013']) * ast.scale['WFC'] * 1e3
    
    dx_13_24 = (g['x_2024'] - g['x_2013']) * ast.scale['WFC'] * 1e3
    dy_13_24 = (g['y_2024'] - g['y_2013']) * ast.scale['WFC'] * 1e3
    
    dx_15_24 = (g['x_2024'] - g['x_2015']) * ast.scale['WFC'] * 1e3
    dy_15_24 = (g['y_2024'] - g['y_2015']) * ast.scale['WFC'] * 1e3
    
    # Keep only stars with all pairwise differences < 20 mas
    diffs = [
        dx_05_10, dy_05_10, dx_05_13, dy_05_13, dx_05_15, dy_05_15, dx_05_24, dy_05_24,
        dx_10_13, dy_10_13, dx_10_15, dy_10_15, dx_10_24, dy_10_24,
        dx_13_15, dy_13_15, dx_13_24, dy_13_24,
        dx_15_24, dy_15_24
    ]
    small = np.where(np.all([np.abs(d) < 20 for d in diffs], axis=0))[0]

    print(len(g), len(small), len(dx_05_10))
    g = g[small]
    dx_05_10 = dx_05_10[small]
    dx_05_13 = dx_05_13[small]
    dx_05_15 = dx_05_15[small]
    dx_05_24 = dx_05_24[small]
    dx_10_13 = dx_10_13[small]
    dx_10_15 = dx_10_15[small]
    dx_10_24 = dx_10_24[small]
    dx_13_15 = dx_13_15[small]
    dx_13_24 = dx_13_24[small]
    dx_15_24 = dx_15_24[small]
    
    dy_05_10 = dy_05_10[small]
    dy_05_13 = dy_05_13[small]
    dy_05_15 = dy_05_15[small]
    dy_05_24 = dy_05_24[small]
    dy_10_13 = dy_10_13[small]
    dy_10_15 = dy_10_15[small]
    dy_10_24 = dy_10_24[small]
    dy_13_15 = dy_13_15[small]
    dy_13_24 = dy_13_24[small]
    dy_15_24 = dy_15_24[small]
    print(len(g), len(small), len(dx_05_10))

    qscale = 2e2
    
    q = plt.quiver(g['x_2005'], g['y_2005'], dx_05_10, dy_05_10, scale=qscale)
    plt.quiverkey(q, 0.95, 0.95, 5, '5 mas', color='red', labelcolor='red')
    plt.title('2010 - 2005')
    plt.savefig(f'{path}/vec_diff_ref5_05_10.png')
    plt.close()

    q = plt.quiver(g['x_2005'], g['y_2005'], dx_05_13, dy_05_13, scale=qscale)
    plt.quiverkey(q, 0.95, 0.95, 5, '5 mas', color='red', labelcolor='red')
    plt.title('2013 - 2005')
    plt.savefig(f'{path}/vec_diff_ref5_05_13.png')
    plt.close()

    q = plt.quiver(g['x_2005'], g['y_2005'], dx_05_15, dy_05_15, scale=qscale)
    plt.quiverkey(q, 0.95, 0.95, 5, '5 mas', color='red', labelcolor='red')
    plt.title('2015 - 2005')
    plt.savefig(f'{path}/vec_diff_ref5_05_15.png')
    plt.close()
    
    q = plt.quiver(g['x_2005'], g['y_2005'], dx_05_24, dy_05_24, scale=qscale)
    plt.quiverkey(q, 0.95, 0.95, 5, '5 mas', color='red', labelcolor='red')
    plt.title('2024 - 2005')
    plt.savefig(f'{path}/vec_diff_ref5_05_24.png')
    plt.close()
    
    q = plt.quiver(g['x_2005'], g['y_2005'], dx_10_13, dy_10_13, scale=qscale)
    plt.quiverkey(q, 0.95, 0.95, 5, '5 mas', color='red', labelcolor='red')
    plt.title('2013 - 2010')
    plt.savefig(f'{path}/vec_diff_ref5_10_13.png')
    plt.close()
    
    q = plt.quiver(g['x_2005'], g['y_2005'], dx_10_15, dy_10_15, scale=qscale)
    plt.quiverkey(q, 0.95, 0.95, 5, '5 mas', color='red', labelcolor='red')
    plt.title('2015 - 2010')
    plt.savefig(f'{path}/vec_diff_ref5_10_15.png')
    plt.close()
    
    q = plt.quiver(g['x_2005'], g['y_2005'], dx_10_24, dy_10_24, scale=qscale)
    plt.quiverkey(q, 0.95, 0.95, 5, '5 mas', color='red', labelcolor='red')
    plt.title('2024 - 2010')
    plt.savefig(f'{path}/vec_diff_ref5_10_24.png')
    plt.close()
    
    q = plt.quiver(g['x_2005'], g['y_2005'], dx_13_15, dy_13_15, scale=qscale)
    plt.quiverkey(q, 0.95, 0.95, 5, '5 mas', color='red', labelcolor='red')
    plt.title('2015 - 2013')
    plt.savefig(f'{path}/vec_diff_ref5_13_15.png')
    plt.close()
    
    q = plt.quiver(g['x_2005'], g['y_2005'], dx_13_24, dy_13_24, scale=qscale)
    plt.quiverkey(q, 0.95, 0.95, 5, '5 mas', color='red', labelcolor='red')
    plt.title('2024 - 2013')
    plt.savefig(f'{path}/vec_diff_ref5_13_24.png')
    plt.close()
    
    q = plt.quiver(g['x_2005'], g['y_2005'], dx_15_24, dy_15_24, scale=qscale)
    plt.quiverkey(q, 0.95, 0.95, 5, '5 mas', color='red', labelcolor='red')
    plt.title('2024 - 2015')
    plt.savefig(f'{path}/vec_diff_ref5_15_24.png')
    plt.close()

    ##########
    # VPD
    ##########
    plt.plot(dx_05_10, dy_05_10, 'k.', ms=2)
    lim = 10
    plt.axis([-lim, lim, -lim, lim])
    plt.xlabel('X Proper Motion (mas)')
    plt.ylabel('Y Proper Motion (mas)')
    plt.title('2010 - 2005')
    plt.savefig(f'{path}/pm_diff_ref5_05_10.png')
    plt.close()

    plt.plot(dx_05_13, dy_05_13, 'k.', ms=2)
    plt.axis([-lim, lim, -lim, lim])
    plt.xlabel('X Proper Motion (mas)')
    plt.ylabel('Y Proper Motion (mas)')
    plt.title('2013 - 2005')
    plt.savefig(f'{path}/pm_diff_ref5_05_13.png')
    plt.close()

    plt.plot(dx_05_15, dy_05_15, 'k.', ms=2)
    plt.axis([-lim, lim, -lim, lim])
    plt.xlabel('X Proper Motion (mas)')
    plt.ylabel('Y Proper Motion (mas)')
    plt.title('2015 - 2005')
    plt.savefig(f'{path}/pm_diff_ref5_05_15.png')
    plt.close()
    
    plt.plot(dx_05_24, dy_05_24, 'k.', ms=2)
    plt.axis([-lim, lim, -lim, lim])
    plt.xlabel('X Proper Motion (mas)')
    plt.ylabel('Y Proper Motion (mas)')
    plt.title('2024 - 2005')
    plt.savefig(f'{path}/pm_diff_ref5_05_24.png')
    plt.close()
    
    plt.plot(dx_10_13, dy_10_13, 'k.', ms=2)
    plt.axis([-lim, lim, -lim, lim])
    plt.xlabel('X Proper Motion (mas)')
    plt.ylabel('Y Proper Motion (mas)')
    plt.title('2013 - 2010')
    plt.savefig(f'{path}/pm_diff_ref5_10_13.png')
    plt.close()
    
    plt.plot(dx_10_15, dy_10_15, 'k.', ms=2)
    plt.axis([-lim, lim, -lim, lim])
    plt.xlabel('X Proper Motion (mas)')
    plt.ylabel('Y Proper Motion (mas)')
    plt.title('2015 - 2010')
    plt.savefig(f'{path}/pm_diff_ref5_10_15.png')
    plt.close()
    
    plt.plot(dx_10_24, dy_10_24, 'k.', ms=2)
    plt.axis([-lim, lim, -lim, lim])
    plt.xlabel('X Proper Motion (mas)')
    plt.ylabel('Y Proper Motion (mas)')
    plt.title('2024 - 2010')
    plt.savefig(f'{path}/pm_diff_ref5_10_24.png')
    plt.close()

    plt.plot(dx_13_15, dy_13_15, 'k.', ms=2)
    plt.axis([-lim, lim, -lim, lim])
    plt.xlabel('X Proper Motion (mas)')
    plt.ylabel('Y Proper Motion (mas)')
    plt.title('2015 - 2013')
    plt.savefig(f'{path}/pm_diff_ref5_13_15.png')
    plt.close()
    
    plt.plot(dx_13_24, dy_13_24, 'k.', ms=2)
    plt.axis([-lim, lim, -lim, lim])
    plt.xlabel('X Proper Motion (mas)')
    plt.ylabel('Y Proper Motion (mas)')
    plt.title('2024 - 2013')
    plt.savefig(f'{path}/pm_diff_ref5_13_24.png')
    plt.close()
    
    plt.plot(dx_15_24, dy_15_24, 'k.', ms=2)
    plt.axis([-lim, lim, -lim, lim])
    plt.xlabel('X Proper Motion (mas)')
    plt.ylabel('Y Proper Motion (mas)')
    plt.title('2024 - 2015')
    plt.savefig(f'{path}/pm_diff_ref5_15_24.png')
    plt.close()

    print('2010 - 2005')
    print('   dx = {dx:6.2f} ± {dxe:6.2f} mas'.format(dx=dx_05_10.mean(), dxe=dx_05_10.std()))
    print('   dy = {dy:6.2f} ± {dye:6.2f} mas'.format(dy=dy_05_10.mean(), dye=dy_05_10.std()))

    print('2013 - 2005')
    print('   dx = {dx:6.2f} ± {dxe:6.2f} mas'.format(dx=dx_05_13.mean(), dxe=dx_05_13.std()))
    print('   dy = {dy:6.2f} ± {dye:6.2f} mas'.format(dy=dy_05_13.mean(), dye=dy_05_13.std()))

    print('2015 - 2005')
    print('   dx = {dx:6.2f} ± {dxe:6.2f} mas'.format(dx=dx_05_15.mean(), dxe=dx_05_15.std()))
    print('   dy = {dy:6.2f} ± {dye:6.2f} mas'.format(dy=dy_05_15.mean(), dye=dy_05_15.std()))

    print('2024 - 2005')
    print('   dx = {dx:6.2f} ± {dxe:6.2f} mas'.format(dx=dx_05_24.mean(), dxe=dx_05_24.std()))
    print('   dy = {dy:6.2f} ± {dye:6.2f} mas'.format(dy=dy_05_24.mean(), dye=dy_05_24.std()))
    
    print('2013 - 2010')
    print('   dx = {dx:6.2f} ± {dxe:6.2f} mas'.format(dx=dx_10_13.mean(), dxe=dx_10_13.std()))
    print('   dy = {dy:6.2f} ± {dye:6.2f} mas'.format(dy=dy_10_13.mean(), dye=dy_10_13.std()))

    print('2015 - 2010')
    print('   dx = {dx:6.2f} ± {dxe:6.2f} mas'.format(dx=dx_10_15.mean(), dxe=dx_10_15.std()))
    print('   dy = {dy:6.2f} ± {dye:6.2f} mas'.format(dy=dy_10_15.mean(), dye=dy_10_15.std()))

    print('2024 - 2010')
    print('   dx = {dx:6.2f} ± {dxe:6.2f} mas'.format(dx=dx_10_24.mean(), dxe=dx_10_24.std()))
    print('   dy = {dy:6.2f} ± {dye:6.2f} mas'.format(dy=dy_10_24.mean(), dye=dy_10_24.std()))
    
    print('2015 - 2013')
    print('   dx = {dx:6.2f} ± {dxe:6.2f} mas'.format(dx=dx_13_15.mean(), dxe=dx_13_15.std()))
    print('   dy = {dy:6.2f} ± {dye:6.2f} mas'.format(dy=dy_13_15.mean(), dye=dy_13_15.std()))

    print('2024 - 2013')
    print('   dx = {dx:6.2f} ± {dxe:6.2f} mas'.format(dx=dx_13_24.mean(), dxe=dx_13_24.std()))
    print('   dy = {dy:6.2f} ± {dye:6.2f} mas'.format(dy=dy_13_24.mean(), dye=dy_13_24.std()))
    
    print('2024 - 2015')
    print('   dx = {dx:6.2f} ± {dxe:6.2f} mas'.format(dx=dx_15_24.mean(), dxe=dx_15_24.std()))
    print('   dy = {dy:6.2f} ± {dye:6.2f} mas'.format(dy=dy_15_24.mean(), dye=dy_15_24.std()))

    return
    
def plot_quiver_align(pos, orig='', catalog='catalog_align_a4_t.fits'):
    """
    Set orig='orig' to plot xorig and yorig (straight from ks2) rather than
    x and y from align output.
    """
    t = Table.read(work_dir + '/50.ALIGN_KS2/' + catalog)

    epoch_names = t.meta['EPNAMES']
    ast_epochs = []
    mtrim = []
    for ee in range(len(epoch_names)):
        if '2005_F814W' in epoch_names[ee]:
            ast_epochs.append(ee)
            mtrim.append(23)
        if ((pos in epoch_names[ee]) and ('F160W' in epoch_names[ee])):
            ast_epochs.append(ee)
            mtrim.append(20)


    # Successively trim down the table to only those stars
    # detected in all epochs, bright, and with small errors. 
    for ii in range(len(ast_epochs)):
        ee = ast_epochs[ii]
        mm = mtrim[ii]
        
        idx = np.where((t['m'][:,ee] < mm) &
                       (t['x'][:,ee] > -1) &
                       (t['y'][:,ee] > -1) &
                       (t['xe'][:,ee] < 0.05) &
                       (t['ye'][:,ee] < 0.05) &
                       (t['me'][:,ee] < 0.05))[0]
        # print( ee, len(idx))

        t = t[idx]
        
    print('Number of Good Stars = ', len(t))
                       
    # Plot all possible combinations of these epochs.
    qscale = 2e2
    plot_dir = work_dir + '/50.ALIGN_KS2/plots/quiver_' + catalog.replace('.fits', '/')
    fileUtil.mkdir(plot_dir)
    _out = open(plot_dir + 'stats_' + pos + orig + '.txt', 'w')

    plt.close(1)
    plt.figure(1, figsize=(6,6))
    
    for ia in range(len(ast_epochs)):
        for ib in range(ia + 1, len(ast_epochs)):
            aa = ast_epochs[ia]
            bb = ast_epochs[ib]
            
            dx = (t['x' + orig][:, bb] - t['x' + orig][:, aa]) * ast.scale['WFC'] * 1e3
            dy = (t['y' + orig][:, bb] - t['y' + orig][:, aa]) * ast.scale['WFC'] * 1e3

            small = np.where((np.abs(dx) < 20) & (np.abs(dy) < 20))[0]
            # print('Small = ', len(small))

            g = t[small]
            dx = dx[small]
            dy = dy[small]

            ename_aa = epoch_names[aa]
            ename_bb = epoch_names[bb]

            plt.clf()
            q = plt.quiver(g['x' + orig][:,ast_epochs[0]], g['y' + orig][:,ast_epochs[0]], dx, dy, scale=qscale)
            plt.quiverkey(q, 0.95, 0.95, 5, '5 mas', color='red', labelcolor='red')
            plt.title(ename_bb + ' - ' + ename_aa)
            plt.savefig(plot_dir + 'quiver_' + ename_bb + '_' + ename_aa + orig + '.png')

            title = '{0:s} - {1:s}'.format(ename_bb, ename_aa)
            
            plt.clf()
            plt.plot(dx, dy, 'k.', ms=2)
            lim = 20
            plt.axis([-lim, lim, -lim, lim])
            plt.xlabel('X Proper Motion (mas)')
            plt.ylabel('Y Proper Motion (mas)')
            plt.title(title)
            plt.savefig(plot_dir + 'vpd_' + ename_bb + '_' + ename_aa + orig + '.png')
    
            # print(title)
            # print('   dx = {dx:6.2f} ± {dxe:6.2f} mas'.format(dx=dx.mean(), dxe=dx.std()))
            # print('   dy = {dy:6.2f} ± {dye:6.2f} mas'.format(dy=dy.mean(), dye=dy.std()))

            _out.write(title + '\n')
            _out.write('   dx = {dx:6.2f} ± {dxe:6.2f} mas\n'.format(dx=dx.mean(), dxe=dx.std()))
            _out.write('   dy = {dy:6.2f} ± {dye:6.2f} mas\n'.format(dy=dy.mean(), dye=dy.std()))
    
    _out.close()
    
    return

def plot_vpd_across_field(nside=4, interact=False):
    """
    Plot the VPD at different field positions so we can see if there are
    systematic discrepancies due to residual distortions.
    """
    # Read in matched and aligned star lists from the *.ref5 analysis.
    # Recall these data sets are in the F814W reference frame with a 50 mas plate scale.
    t2005 = starlists.read_matchup(work_dir + '/02.MAT/MATCHUP.XYMEEE.F814W.ref5')
    t2010 = starlists.read_matchup(work_dir + '/02.MAT/MATCHUP.XYMEEE.F125W.ref5')

    scale = 50.0 # mas per pixel
    

    # Trim down to only those stars that are detected in both epochs.
    # Also make cuts on astrometric/photometric errors, etc.
    # We only want the well measured stars in this analysis.
    perrLim814 = 1.0 / scale
    perrLim125 = 4.0 / scale
    merrLim814 = 0.05
    merrLim125 = 0.1
    
    cond = ((t2005.m != 0) & (t2010.m != 0) &
            (t2005.xe < perrLim814) & (t2005.ye < perrLim814) &
            (t2010.xe < perrLim125) & (t2010.ye < perrLim125) &
            (t2005.me < merrLim814) & (t2010.me < merrLim125))

    t2005 = t2005.where(cond)
    t2010 = t2010.where(cond)

    # Calculate proper motions
    dt = years['2010_F125W'] - years['2005_F814W']
    dx = t2010.x - t2005.x
    dy = t2010.y - t2005.y
    pmx = dx * scale / dt
    pmy = dy * scale / dt
    pm = np.hypot(pmx, pmy)

    t2005.add_column('pmx', pmx)
    t2005.add_column('pmy', pmy)
    t2005.add_column('pm', pm)


    # Divide up the region into N x N boxes and plot up the VPD for each.
    xlo = math.floor(t2005.x.min())
    xhi = math.ceil(t2005.x.max())
    ylo = math.floor(t2005.y.min())
    yhi = math.ceil(t2005.y.max())
    xboxsize = round((xhi - xlo) / nside)
    yboxsize = round((yhi - ylo) / nside)

    # Setup colors
    jet = plt.get_cmap('jet')
    cNorm = colors.Normalize(vmin=0, vmax=nside**2)
    colorMap = plt.cm.ScalarMappable(norm=cNorm, cmap=jet)

    # Save the average proper motions in each box
    pmx = np.zeros((nside, nside), dtype=float)
    pmy = np.zeros((nside, nside), dtype=float)
    pmxe = np.zeros((nside, nside), dtype=float)
    pmye = np.zeros((nside, nside), dtype=float)
    xcen = np.zeros((nside, nside), dtype=float)
    ycen = np.zeros((nside, nside), dtype=float)

    pmCut = 1.0

    # Calculate the global mean proper motion
    # Start by trimming down to a 1 mas/yr radius
    idx2 = np.where(pm < pmCut)[0]
    pmx_all = np.median( t2005.pmx[idx2] )
    pmy_all = np.median( t2005.pmy[idx2] )
    
    out = 'All X:{0:5.0f}-{1:5.0f}  Y:{2:5.0f}-{3:5.0f}  '
    out += 'PMX:{4:5.2f} ± {5:5.2f} PMY:{6:5.2f} ± {7:5.2f}  '
    out += 'N:{8:5d}'
    print((out.format(xlo, xhi, ylo, yhi, pmx_all, 0.0, pmy_all, 0.0, len(idx2))))

    # Make a global proper motion diagram of star with a proper motion within
    # 1 mas/yr. This is mainly to see systematic flows due to residual distortion.
    pmTot = np.hypot(t2005.pmx, t2005.pmy)
    clust = np.where(pmTot < pmCut)[0]
    plt.clf()
    q = plt.quiver(t2005.x[clust], t2005.y[clust], t2005.pmx[clust], t2005.pmy[clust],
                  scale=18)
    plt.quiverkey(q, 0.5, 0.98, 1, '1 mas/yr', color='red', labelcolor='red')
    plt.xlabel('X Position (pixels)')
    plt.ylabel('Y Position (pixels)')
    plt.xlim(xlo, xhi)
    plt.ylim(ylo, yhi)
    out = '{0}/plots/vec_proper_motion_all.png'
    plt.savefig(out.format(work_dir))
    
    plt.clf()
    for xx in range(nside):
        for yy in range(nside):
            xlo_box = xlo + xx * xboxsize
            ylo_box = ylo + yy * yboxsize
            xhi_box = xlo + (1+xx) * xboxsize
            yhi_box = ylo + (1+yy) * yboxsize

            idx = np.where((t2005.x > xlo_box) & (t2005.x <= xhi_box) &
                           (t2005.y > ylo_box) & (t2005.y <= yhi_box))[0]


            if interact:
                color = colorMap.to_rgba(yy + xx * nside)
                lim = 5

                plt.plot(t2005.pmx[idx], t2005.pmy[idx], 'k.', ms=2, color=color)
                plt.axis([-lim, lim, -lim, lim])

                plt.xlabel('X Proper Motion (mas/yr)')
                plt.ylabel('Y Proper Motion (mas/yr)')

            # Lets get the mean and std-dev (iterative) for the box.
            # Start by trimming down to a 1 mas/yr circle.
            idx2 = np.where(t2005.pm[idx] < pmCut)[0]
            xmean = np.median( t2005.pmx[idx][idx2] )
            ymean = np.median( t2005.pmy[idx][idx2] )
            xstd = t2005.pmx[idx][idx2].std()
            ystd = t2005.pmy[idx][idx2].std()
            xmean_err = xstd / np.sqrt(len(idx2))
            ymean_err = ystd / np.sqrt(len(idx2))

            xcen[xx, yy] = xlo_box + (xboxsize / 2.0)
            ycen[xx, yy] = ylo_box + (yboxsize / 2.0)
            pmx[xx, yy] = xmean - pmx_all
            pmy[xx, yy] = ymean - pmx_all
            pmxe[xx, yy] = xmean_err
            pmye[xx, yy] = ymean_err

            out = 'Box X:{0:5.0f}-{1:5.0f}  Y:{2:5.0f}-{3:5.0f}  '
            out += 'PMX:{4:5.2f} ± {5:5.2f} PMY:{6:5.2f} ± {7:5.2f}  '
            out += 'N:{8:5d}  '

            if interact:
                out += 'Continue?'
                input(out.format(xlo_box, xhi_box, ylo_box, yhi_box,
                                     xmean, xmean_err, ymean, ymean_err, len(idx2)))
            else:
                print((out.format(xlo_box, xhi_box, ylo_box, yhi_box,
                                 xmean, xmean_err, ymean, ymean_err, len(idx2))))


    if interact:
        out = '{0}/plots/vpd_grid_nside{1}.png'
        plt.savefig(out.format(work_dir, nside))

    plt.clf()
    q = plt.quiver(xcen, ycen, pmx, pmy)
    plt.quiverkey(q, 0.5, 0.98, 0.1, '0.1 mas/yr', color='red', labelcolor='red')
    plt.xlabel('X Position (pixels)')
    plt.ylabel('Y Position (pixels)')
    plt.xlim(xlo, xhi)
    plt.ylim(ylo, yhi)
    for xx in range(nside+1):
        plt.axvline(xlo + xx * xboxsize, linestyle='--', color='grey')
    for yy in range(nside+1):
        plt.axhline(ylo + yy * yboxsize, linestyle='--', color='grey')
    out = '{0}/plots/vec_proper_motion_grid_nside{1}.png'
    plt.savefig(out.format(work_dir, nside))

    plt.clf()
    q = plt.quiver(xcen, ycen, pmx/pmxe, pmy/pmye)
    plt.quiverkey(q, 0.5, 0.98, 3, '3 sigma', color='red', labelcolor='red')
    plt.xlabel('X Position (pixels)')
    plt.ylabel('Y Position (pixels)')
    plt.xlim(xlo, xhi)
    plt.ylim(ylo, yhi)
    for xx in range(nside+1):
        plt.axvline(xlo + xx * xboxsize, linestyle='--', color='grey')
    for yy in range(nside+1):
        plt.axhline(ylo + yy * yboxsize, linestyle='--', color='grey')
    out = '{0}/plots/vec_proper_motion_grid_sig_nside{1}.png'
    plt.savefig(out.format(work_dir, nside))

def calc_years(data_dir):
    """
    Calculate the epoch for each data set.
    """
    # years = ['2005', '2010', '2010', '2010', '2013', '2013', '2015', '2015', '2024', '2024']
    # filts = ['F814W', 'F125W', 'F139M', 'F160W', 'F160W', 'F160Ws', 'F160W', 'F160Ws', 'F160W', 'F160Ws']
    
    years = ['2024', '2024']
    filts = ['F160W', 'F160Ws']
    
    for ii in range(len(years)):
        search_path = f'{data_dir}/{years[ii]}_{filts[ii]}/00.DATA'
        search_path_list = glob.glob(f'{search_path}/*_flt.fits')
        # print(f'Searching {search_path}: {search_path_list}')
        assert len(search_path_list) > 0, f'No files found in {search_path}'
        epoch = flystar.calc_mean_year(glob.glob(f'{search_path}/*_flt.fits'))

        print(('{0}_{1} at {2:8.3f}'.format(years[ii], filts[ii], epoch)))


def make_master_list(data_dir, years=years, pscale=50, vel_cut=0.7, vel_err_cut=0.2, outdir_list=None):
    """Trim stars with preliminary proper motions within vel_cut and vel_err_cut

    Parameters
    ----------
    data_dir : str
        Directory containing the 02.MAT/MATCHUP.XYMEEE.YEAR.ref files.
    years : dict
        Dictionary of observation time in the form of 'year_filt': yeardecimal, e.g. '2005_F814W': 2005.0
    pscale : float, optional
        Pixel scale in mas/pixel, by default 50.0
    vel_cut : float, optional
        Velocity cut in mas/yr, by default 0.7
    vel_err_cut : float, optional
        Velocity error cut in mas/yr, by default 0.2
    outdir_list : list of str, optional
        Output directory list to save the master list, by default None
    """
    t2005_814 = starlists.read_matchup(f'{data_dir}/02.MAT/MATCHUP.XYMEEE.F814W.2005.ref5')
    t2010_125 = starlists.read_matchup(f'{data_dir}/02.MAT/MATCHUP.XYMEEE.F125W.2010.ref5')
    t2010_139 = starlists.read_matchup(f'{data_dir}/02.MAT/MATCHUP.XYMEEE.F139M.2010.ref5')
    t2010_160 = starlists.read_matchup(f'{data_dir}/02.MAT/MATCHUP.XYMEEE.F160W.2010.ref5')
    t2013_160 = starlists.read_matchup(f'{data_dir}/02.MAT/MATCHUP.XYMEEE.F160W.2013.ref5')
    t2013_160s = starlists.read_matchup(f'{data_dir}/02.MAT/MATCHUP.XYMEEE.F160Ws.2013.ref5')
    t2015_160 = starlists.read_matchup(f'{data_dir}/02.MAT/MATCHUP.XYMEEE.F160W.2015.ref5')
    t2015_160s = starlists.read_matchup(f'{data_dir}/02.MAT/MATCHUP.XYMEEE.F160Ws.2015.ref5')
    t2024_160 = starlists.read_matchup(f'{data_dir}/02.MAT/MATCHUP.XYMEEE.F160W.2024.ref5')
    t2024_160s = starlists.read_matchup(f'{data_dir}/02.MAT/MATCHUP.XYMEEE.F160Ws.2024.ref5')
    
    # Assert all tables have the same length
    nrows = len(t2005_814)
    assert all(len(tbl) == nrows for tbl in [
        t2010_125, t2010_139, t2010_160, t2013_160, t2013_160s,
        t2015_160, t2015_160s, t2024_160, t2024_160s
    ]), "All tables must have the same number of rows"
    
    # Trim down to only those stars that are detected in all epochs.
    # Also make cuts on astrometric/photometric errors, etc.
    # We only want the well measured stars in this analysis.
    perrLim2005 = 3.0 / pscale # unit: pix, equivalent to error < 3 mas
    perrLim2010 = 4.0 / pscale
    perrLim2013 = 5.0 / pscale
    perrLim2015 = 5.0 / pscale
    perrLim2024 = 5.0 / pscale
    merrLim2005 = 0.05
    merrLim2010 = 0.1
    merrLim2013 = 0.1
    merrLim2015 = 0.1
    merrLim2024 = 0.1


    detect_2005 = (
        (t2005_814['m'] != 0) & 
        (t2005_814['xe'] < perrLim2005) & (t2005_814['ye'] < perrLim2005) &
        (t2005_814['me'] < merrLim2005)
    )

    detect_2010 = (
        (t2010_125['m'] != 0) & (t2010_139['m'] != 0) & (t2010_160['m'] != 0) &
        (t2010_125['xe'] < perrLim2010) & (t2010_125['ye'] < perrLim2010) &
        (t2010_139['xe'] < perrLim2010) & (t2010_139['ye'] < perrLim2010) &
        (t2010_160['xe'] < perrLim2010) & (t2010_160['ye'] < perrLim2010) &
        (t2010_125['me'] < merrLim2010) & (t2010_139['me'] < merrLim2010) & (t2010_160['me'] < merrLim2010)
    )

    detect_2013 = (
        (t2013_160['m'] != 0) &
        (t2013_160['xe'] < perrLim2013) & (t2013_160['ye'] < perrLim2013) &
        (t2013_160['me'] < merrLim2013)
    )

    detect_2015 = (
        (t2015_160['m'] != 0) &
        (t2015_160['xe'] < perrLim2015) & (t2015_160['ye'] < perrLim2015) &
        (t2015_160['me'] < merrLim2015)
    )

    detect_2024 = (
        (t2024_160['m'] != 0) &
        (t2024_160['xe'] < perrLim2024) & (t2024_160['ye'] < perrLim2024) &
        (t2024_160['me'] < merrLim2024)
    )

    detect_all = np.logical_and.reduce([
        detect_2005, detect_2010, detect_2013, detect_2015, detect_2024
    ])

    detect_ir = np.logical_and.reduce([
        detect_2010, detect_2013, detect_2015
    ])

    detect_old = np.logical_and.reduce([
        detect_2005, detect_2010, detect_2013, detect_2015
    ])

    print(f'Detected in all years: {detect_all.sum()} out of {len(detect_all)}')
    print(f'Detected in all IR filters: {detect_ir.sum()} out of {len(detect_ir)}')
    print(f'Detected in all years except 2024: {detect_old.sum()} out of {len(detect_old)}')

    t2005_814 = t2005_814[detect_ir]
    t2010_125 = t2010_125[detect_ir]
    t2010_139 = t2010_139[detect_ir]
    t2010_160 = t2010_160[detect_ir]
    t2013_160 = t2013_160[detect_ir]
    t2013_160s = t2013_160s[detect_ir]
    t2015_160 = t2015_160[detect_ir]
    t2015_160s = t2015_160s[detect_ir]
    t2024_160 = t2024_160[detect_ir]
    t2024_160s = t2024_160s[detect_ir]

    table_list = [t2005_814, t2010_160, t2013_160, t2015_160, t2024_160]
    # Calculate proper motions
    t = np.array([
        years['2005_F814W'], 
        years['2010_F160W'], 
        years['2013_F160W'], 
        years['2015_F160W'], 
        years['2024_F160W']
    ])
    x = np.array([table['x'] for table in table_list]).T
    y = np.array([table['y'] for table in table_list]).T
    xe = np.array([table['xe'] for table in table_list]).T
    ye = np.array([table['ye'] for table in table_list]).T

    t_reference = 2010.
    N = len(x)
    vx = np.zeros(N)
    vy = np.zeros(N)
    vxe = np.zeros(N)
    vye = np.zeros(N)
    x0 = np.zeros(N)
    y0 = np.zeros(N)
    x0e = np.zeros(N)
    y0e = np.zeros(N)
    chi2_vx = np.zeros(N)
    chi2_vy = np.zeros(N)
    t0 = np.zeros(N)

    for i in tqdm(range(len(x))):
        mm = Linear()
        params, param_errors = mm.fit(
            t=t[1:],
            x=x[i, 1:] * pscale,
            y=y[i, 1:] * pscale,
            xe=xe[i, 1:] * pscale,
            ye=ye[i, 1:] * pscale,
            t0=t_reference
        )

        x0[i], vx[i], y0[i], vy[i] = params
        x0e[i], vxe[i], y0e[i], vye[i] = param_errors
        chi2_vx[i], chi2_vy[i] = mm.calc_chi2(params, np.array([t_reference]), t[1:], x[i, 1:], y[i, 1:], xe[i, 1:], ye[i, 1:])

        # vx_result = linear_fit(t[1:] - t_reference, x[i, 1:] * pscale, sigma=xe[i, 1:] * pscale)
        # vy_result = linear_fit(t[1:] - t_reference, y[i, 1:] * pscale, sigma=ye[i, 1:] * pscale)
        # vx[i] = vx_result['slope']
        # vy[i] = vy_result['slope']
        # vxe[i] = vx_result['e_slope']
        # vye[i] = vy_result['e_slope']
        # x0[i] = vx_result['intercept']
        # y0[i] = vy_result['intercept']
        # x0e[i] = vx_result['e_intercept']
        # y0e[i] = vy_result['e_intercept']
        # chi2_vx[i] = vx_result['chi2']
        # chi2_vy[i] = vy_result['chi2']
        t0[i] = np.average(t, weights=1. / np.hypot(xe[i], ye[i]))

    fig, ax = plt.subplots()
    ax.scatter(vx, vy, s=1, c='k', alpha=0.2)
    circle = plt.Circle((0, 0), 0.7, color='red', fill=False, linewidth=1.5)
    ax.add_patch(circle)
    ax.set_aspect('equal')
    ax.set_xlim(-1.5, 1.5)
    ax.set_ylim(-1.5, 1.5)
    ax.set_xlabel('Proper Motion in X (mas/yr)')
    ax.set_ylabel('Proper Motion in Y (mas/yr)')
    ax.set_title('Proper Motion Vector Field')
    plt.tight_layout()
    plt.show()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4))
    ax1.scatter(t2005_814['m'], vxe, s=1, alpha=0.2, label='X')
    ax2.scatter(t2005_814['m'], vye, s=1, alpha=0.2, label='Y', color='C3')
    ax1.set_yscale('log')
    ax2.set_yscale('log')
    ax1.set_xlabel('Magnitude', fontsize=12)
    ax1.set_ylabel('Proper Motion Error in X (mas/yr)', fontsize=12)
    ax2.set_xlabel('Magnitude', fontsize=12)
    ax2.set_ylabel('Proper Motion Error in Y (mas/yr)', fontsize=12)
    ax1.set_title('Proper Motion Error in X vs Magnitude', fontsize=12)
    ax2.set_title('Proper Motion Error in Y vs Magnitude', fontsize=12)
    ax1.legend()
    ax2.legend()
    plt.tight_layout()
    plt.show()

    v = np.hypot(vx, vy)
    # ve = np.sqrt((vxe * vx / v)**2 + (vye * vy / v)**2)
    
    good = (
        (v <= vel_cut) & (vxe <= vel_err_cut) & (vye <= vel_err_cut)
    )
    t2005_814 = t2005_814[good]
    t2010_125 = t2010_125[good]
    t2010_139 = t2010_139[good]
    t2010_160 = t2010_160[good]
    t2013_160 = t2013_160[good]
    t2013_160s = t2013_160s[good]
    t2015_160 = t2015_160[good]
    t2015_160s = t2015_160s[good]
    t2024_160 = t2024_160[good]
    t2024_160s = t2024_160s[good]
    
    all_tables = [t2005_814, t2010_125, t2010_139, t2010_160, t2013_160, t2013_160s, t2015_160, t2015_160s, t2024_160, t2024_160s]

    output_format = {
        'x': '10.4f', 'y': '10.4f', 'm': '8.4f',
        'xe': '10.4f', 'ye': '10.4f', 'me': '8.4f'
    }
    columns = ['x', 'y', 'm', 'xe', 'ye', 'me', 'name']


    if outdir_list:
        for table, outdir in zip(all_tables, outdir_list):
            table[columns].write(
                outdir,
                delimiter='',
                format='ascii.fixed_width_no_header',
                formats=output_format,
                overwrite=True
            )
    return


# def make_master_lists():
#     """
#     Trim the ref5 master lists for each filter down to just stars with
#     proper motions within 1 mas/yr of the cluster motion.
#     """
#     # Read in matched and aligned star lists from the *.ref5 analysis.
#     # Recall these data sets are in the F814W reference frame with a 50 mas plate scale.
#     print('Loading Data')
#     t2005_814 = starlists.read_matchup(work_dir + '/02.MAT/MATCHUP.XYMEEE.F814W.2005.ref5')
#     t2010_125 = starlists.read_matchup(work_dir + '/02.MAT/MATCHUP.XYMEEE.F125W.2010.ref5')
#     t2010_139 = starlists.read_matchup(work_dir + '/02.MAT/MATCHUP.XYMEEE.F139M.2010.ref5')
#     t2010_160 = starlists.read_matchup(work_dir + '/02.MAT/MATCHUP.XYMEEE.F160W.2010.ref5')
#     t2013_160 = starlists.read_matchup(work_dir + '/02.MAT/MATCHUP.XYMEEE.F160W.2013.ref5')
#     t2013_160s = starlists.read_matchup(work_dir + '/02.MAT/MATCHUP.XYMEEE.F160Ws.2013.ref5')
#     t2015_160 = starlists.read_matchup(work_dir + '/02.MAT/MATCHUP.XYMEEE.F160W.2015.ref5')
#     t2015_160s = starlists.read_matchup(work_dir + '/02.MAT/MATCHUP.XYMEEE.F160Ws.2015.ref5')

#     scale = 50.0 # mas per pixel

#     # Trim down to only those stars that are detected in all epochs.
#     # Also make cuts on astrometric/photometric errors, etc.
#     # We only want the well measured stars in this analysis.
#     perrLim2005 = 3.0 / scale
#     perrLim2010 = 4.0 / scale
#     perrLim2013 = 5.0 / scale
#     perrLim2015 = 5.0 / scale
#     merrLim2005 = 0.05
#     merrLim2010 = 0.1
#     merrLim2013 = 0.1
#     merrLim2015 = 0.1
    
#     print('Trimming Data')
#     cond = ((t2005_814['m'] != 0) & (t2010_125['m'] != 0) &
#             (t2010_139['m'] != 0) & (t2010_160['m'] != 0) &
#             (t2013_160['m'] != 0) & (t2015_160['m'] != 0) &
#             (t2005_814['xe'] < perrLim2005) & (t2005_814['ye'] < perrLim2005) &
#             (t2010_125['xe'] < perrLim2010) & (t2010_125['ye'] < perrLim2010) &
#             (t2010_139['xe'] < perrLim2010) & (t2010_139['ye'] < perrLim2010) &
#             (t2010_160['xe'] < perrLim2010) & (t2010_160['ye'] < perrLim2010) &
#             (t2013_160['xe'] < perrLim2013) & (t2013_160['ye'] < perrLim2013) &
#             (t2015_160['xe'] < perrLim2015) & (t2015_160['ye'] < perrLim2015) &
#             (t2005_814['me'] < merrLim2005) & (t2010_125['me'] < merrLim2010) &
#             (t2010_139['me'] < merrLim2010) & (t2010_160['me'] < merrLim2010) &
#             (t2013_160['me'] < merrLim2013) & (t2015_160['me'] < merrLim2015))
#     print('    Cutting down to {0} from {1}'.format(cond.sum(), len(t2005_814)))

#     t2005_814 = t2005_814[cond]
#     t2010_125 = t2010_125[cond]
#     t2010_139 = t2010_139[cond]
#     t2010_160 = t2010_160[cond]
#     t2013_160 = t2013_160[cond]
#     t2013_160s = t2013_160s[cond]
#     t2015_160 = t2015_160[cond]
#     t2015_160s = t2015_160s[cond]

#     # Calculate proper motions
#     print('Calculating velocities')
#     t = np.array([years['2005_F814W'], years['2010_F160W'], years['2013_F160W'], years['2015_F160W']])
#     x = np.array([t2005_814['x'], t2010_160['x'], t2013_160['x'], t2015_160['x']]).T
#     y = np.array([t2005_814['y'], t2010_160['y'], t2013_160['y'], t2015_160['y']]).T
#     xe = np.array([t2005_814['xe'], t2010_160['xe'], t2013_160['xe'], t2015_160['xe']]).T
#     ye = np.array([t2005_814['ye'], t2010_160['ye'], t2013_160['ye'], t2015_160['ye']]).T

#     def linefit2(time, pos, pos_err):
#         epochs = np.tile(time, (pos_err.shape[1], 1)).T

#         # t_w = epochs / (pos_err ** 2)
#         # p_w = pos / (pos_err ** 2)

#         w = 1.0 / (pos_err ** 2)

#         w_sum = w.sum(axis=0)

#         wxy = (w * epochs * pos).sum(axis=0)
#         wx = (w * epochs).sum(axis=0)
#         wy = (w * pos).sum(axis=0)
#         wxx = (w * epochs ** 2).sum(axis=0)

#         denom = (w_sum * wxx) - (wx ** 2)
        
#         vel = (w_sum * wxy - wx * wy) / denom
#         pos0 = (wxx * wy - wx * wxy) / denom

#         vel_err = np.sqrt( w_sum / denom )
#         pos0_err = np.sqrt( wxx / denom )

#         return pos0, vel, pos0_err, vel_err
        
    
#     # Lets set a t0 value:
#     t0 = 2010.0
#     x0, vx, x0e, vxe = linefit2(t - t0, x.T, xe.T)
#     y0, vy, y0e, vye = linefit2(t - t0, y.T, ye.T)
#     vx *= scale
#     vy *= scale
#     vxe *= scale
#     vye *= scale

#     # Add the velocity fits to the 2005 table
#     t2005_814['vx'] = vx
#     t2005_814['vy'] = vy
#     t2005_814['vxe'] = vxe
#     t2005_814['vye'] = vye

#     # Get rid of stars without velocities
#     good = ((np.isnan(vx) == False) & (np.isnan(vy) == False))
#     print('    Cutting down to {0} from {1}'.format(good.sum(), len(t2005_814)))
#     t2005_814 = t2005_814[good]
#     t2010_125 = t2010_125[good]
#     t2010_139 = t2010_139[good]
#     t2010_160 = t2010_160[good]
#     t2013_160 = t2013_160[good]
#     t2013_160s = t2013_160s[good]
#     t2015_160 = t2015_160[good]
#     t2015_160s = t2015_160s[good]

#     vx = vx[good]
#     vy = vy[good]
#     vxe = vxe[good]
#     vye = vye[good]

#     # Trim down to a 1 mas/yr radius
#     # vx_mean = statsIter.mean(vx, lsigma=4, hsigma=4, iter=10, verbose=True)
#     # vy_mean = statsIter.mean(vy, lsigma=4, hsigma=4, iter=10, verbose=True)
#     vx_mean = 0.0
#     vy_mean = 0.0 
#     velCut = 0.7
#     velErrCut = 0.2

#     # Make a couple of plots to decide on (and show) the cuts
#     plt.clf()
#     plt.plot(vx, vy, 'k.', alpha=0.2)
#     circ = plt.Circle([vx_mean, vy_mean], radius=velCut, color='red', fill=False)
#     plt.gca().add_artist(circ)
#     plt.xlabel('X Velocity (mas/yr)')
#     plt.ylabel('Y Velocity (mas/yr)')
#     plt.axis([-2, 2, -2, 2])
#     plt.savefig('plots/make_master_vpd_cuts.png')

#     plt.clf()
#     plt.plot(t2005_814['m'], vxe, 'r.', alpha=0.2, label='X')
#     plt.plot(t2005_814['m'], vye, 'b.', alpha=0.2, label='Y')
#     plt.axhline(velErrCut, color='black')
#     plt.xlabel('F814W Magnitude')
#     plt.ylabel('Velocity Error (mas/yr)')
#     plt.legend()
#     plt.savefig('plots/make_master_verr_cuts.png')

#     dv = np.hypot(vx - vx_mean, vy - vy_mean)
#     idx2 = ((dv < velCut) & (vxe < velErrCut) & (vye < velErrCut))
#     print('Making Velocity Cuts: v < {0:3.1f} mas/yr and verr < {1:3.1f} mas/yr'.format(velCut, velErrCut))
#     print('    Cutting down to {0} from {1}'.format(idx2.sum(), len(t2005_814)))

#     t2005_814 = t2005_814[idx2]
#     t2010_125 = t2010_125[idx2]
#     t2010_139 = t2010_139[idx2]
#     t2010_160 = t2010_160[idx2]
#     t2013_160 = t2013_160[idx2]
#     t2013_160s = t2013_160s[idx2]
#     t2015_160 = t2015_160[idx2]
#     t2015_160s = t2015_160s[idx2]

#     _o814_05 = open(work_dir + '/02.MAT/MASTER.F814W.2005.ref5', 'w')
#     _o125_10 = open(work_dir + '/02.MAT/MASTER.F125W.2010.ref5', 'w')
#     _o139_10 = open(work_dir + '/02.MAT/MASTER.F139M.2010.ref5', 'w')
#     _o160_10 = open(work_dir + '/02.MAT/MASTER.F160W.2010.ref5', 'w')
#     _o160_13 = open(work_dir + '/02.MAT/MASTER.F160W.2013.ref5', 'w')
#     _o160s_13 = open(work_dir + '/02.MAT/MASTER.F160Ws.2013.ref5', 'w')
#     _o160_15 = open(work_dir + '/02.MAT/MASTER.F160W.2015.ref5', 'w')
#     _o160s_15 = open(work_dir + '/02.MAT/MASTER.F160Ws.2015.ref5', 'w')

#     _o_align = open(work_dir + '/50.ALIGN_KS2/wd1_1pass_label.dat', 'w')

#     ofmt = '{0:10.4f} {1:10.4f} {2:8.4f} {3:10.4f} {4:10.4f} {5:8.4f} {6}\n'
    
#     for ii in range(len(t2005_814)):
#         _o814_05.write(ofmt.format(t2005_814['x'][ii], t2005_814['y'][ii], t2005_814['m'][ii],
#                                    t2005_814['xe'][ii], t2005_814['ye'][ii], t2005_814['me'][ii],
#                                    t2005_814['name'][ii]))
#         _o125_10.write(ofmt.format(t2010_125['x'][ii], t2010_125['y'][ii], t2010_125['m'][ii],
#                                    t2010_125['xe'][ii], t2010_125['ye'][ii], t2010_125['me'][ii],
#                                    t2010_125['name'][ii]))
#         _o139_10.write(ofmt.format(t2010_139['x'][ii], t2010_139['y'][ii], t2010_139['m'][ii],
#                                    t2010_139['xe'][ii], t2010_139['ye'][ii], t2010_139['me'][ii],
#                                    t2010_139['name'][ii]))
#         _o160_10.write(ofmt.format(t2010_160['x'][ii], t2010_160['y'][ii], t2010_160['m'][ii],
#                                    t2010_160['xe'][ii], t2010_160['ye'][ii], t2010_160['me'][ii],
#                                    t2010_160['name'][ii]))
#         _o160_13.write(ofmt.format(t2013_160['x'][ii], t2013_160['y'][ii], t2013_160['m'][ii],
#                                    t2013_160['xe'][ii], t2013_160['ye'][ii], t2013_160['me'][ii],
#                                    t2013_160['name'][ii]))
#         _o160s_13.write(ofmt.format(t2013_160s['x'][ii], t2013_160s['y'][ii], t2013_160s['m'][ii],
#                                     t2013_160s['xe'][ii], t2013_160s['ye'][ii], t2013_160s['me'][ii],
#                                     t2013_160s['name'][ii]))
#         _o160_15.write(ofmt.format(t2015_160['x'][ii], t2015_160['y'][ii], t2015_160['m'][ii],
#                                    t2015_160['xe'][ii], t2015_160['ye'][ii], t2015_160['me'][ii],
#                                    t2015_160['name'][ii]))
#         _o160s_15.write(ofmt.format(t2015_160s['x'][ii], t2015_160s['y'][ii], t2015_160s['m'][ii],
#                                     t2015_160s['xe'][ii], t2015_160s['ye'][ii], t2015_160s['me'][ii],
#                                     t2015_160s['name'][ii]))


        
#     _o814_05.close()
#     _o125_10.close()
#     _o139_10.close()
#     _o160_10.close()
#     _o160_13.close()
#     _o160s_13.close()
#     _o160_15.close()
#     _o160s_15.close()

#     _o_align.close()

#     return

def make_pos_directories(reduce_path):
    """
    Make 03.MAT_POS directory, and the year_filter_pos1 through pos4 directories.
    
    Parameters
    ----------
    reduce_path : str
        The path to the reduction directory where the 03.MAT_POS directory will be created.
    """
    mat_pos_path = os.path.join(reduce_path, '03.MAT_POS')
    if not os.path.exists(mat_pos_path):
        os.makedirs(mat_pos_path)

    years = ['2005', '2010', '2013', '2015', '2024']

    filts = {'2005': ['F814W'],
             '2010': ['F125W', 'F139M', 'F160W'],
             '2013': ['F160W', 'F160Ws'],
             '2015': ['F160W', 'F160Ws'],
             '2024': ['F160W', 'F160Ws']}

    pos4_exceptions = ['2005_F814W', '2013_F160Ws', '2015_F160Ws', '2024_F160Ws']
    mat_path = f'{reduce_path}/02.MAT'

    for year in years:
        filts_in_year = filts[year]

        for filt in filts_in_year:
            epoch_filt = f'{year}_{filt}'

            if epoch_filt in pos4_exceptions:
                posits = ['']
            else:
                posits = ['pos1', 'pos2', 'pos3', 'pos4']

            # Define the master file for this combo
            master_file = f'{mat_path}/MASTER.{filt}.{year}.ref5'

            for pos in posits:
                ep_filt_pos = epoch_filt
                if pos != '':
                    ep_filt_pos += f'_{pos}'

                # Make the directory
                if not os.path.exists(f'{mat_pos_path}/{ep_filt_pos}'):
                    os.makedirs(f'{mat_pos_path}/{ep_filt_pos}')
                if not os.path.exists(f'{mat_pos_path}/{ep_filt_pos}/01.XYM'):
                    os.makedirs(f'{mat_pos_path}/{ep_filt_pos}/01.XYM')

                # Copy over the master file
                shutil.copy(master_file, f'{mat_pos_path}/{ep_filt_pos}/01.XYM/')

    return

def calc_pos_number(reduce_path, verbose=True):
    """
    Calculate the position number (in 2005 ACS coordinate system) of all the
    MAT.*** files in 02.MAT directory. This can then be used to separate out
    which files are in pos1/ pos2/ pos3/ and pos4/. The schematic for the
    positions is:

        pos4    pos1
        pos3    pos2

    Just read in the MAT.*** file, take the average of columns 3 and 4 and use
    rough, hard-coded quadrant separations to pick out the position. 
    """
    mat_files = glob.glob(f'{reduce_path}/02.MAT/MAT.*')
    epoch_filt = np.zeros(len(mat_files), dtype='U12')
    pos = np.zeros(len(mat_files), dtype='U4')
    mat_root = np.zeros(len(mat_files), dtype='U8')
    xavg = np.zeros(len(mat_files), dtype=float)
    yavg = np.zeros(len(mat_files), dtype=float)

    for mm in range(len(mat_files)):
        tab = starlists.read_mat(mat_files[mm])

        # Get the average X and average Y position for this file
        xavg[mm] = tab['xref'].mean()
        yavg[mm] = tab['yref'].mean()

        # Get the MAT root file name and number for printing
        mat_root[mm] = os.path.split(mat_files[mm])[1]
        mat_num = os.path.splitext(mat_root[mm])[1]
        mat_num = int(mat_num[1:])

        # Decide the epoch and the filter.
        if (mat_num >= 0) and (mat_num <= 99):
            epoch_filt[mm] = '2005_F814W'
        elif (mat_num >= 100) and (mat_num <= 129):
            epoch_filt[mm] = '2010_F125W'
        elif (mat_num >= 130) and (mat_num <= 161):
            epoch_filt[mm] = '2010_F139M'
        elif (mat_num >= 162) and (mat_num <= 199):
            epoch_filt[mm] = '2010_F160W'
        elif (mat_num >= 200) and (mat_num <= 259):
            epoch_filt[mm] = '2013_F160W'
        elif (mat_num >= 270) and (mat_num <= 299):
            epoch_filt[mm] = '2013_F160Ws'
        elif (mat_num >= 300) and (mat_num <= 359):
            epoch_filt[mm] = '2015_F160W'
        elif (mat_num >= 370) and (mat_num <= 399):
            epoch_filt[mm] = '2015_F160Ws'
        elif (mat_num >= 400) and (mat_num <= 459):
            epoch_filt[mm] = '2024_F160W'
        elif (mat_num >= 470) and (mat_num <= 499):
            epoch_filt[mm] = '2024_F160Ws'
        else:
            raise ValueError('Unknown MAT file number: {0}'.format(mat_num))
            
        # Decide the position
        if (xavg[mm] > 2000) and (yavg[mm] > 2000):
            pos[mm] = 'pos1'
        if (xavg[mm] > 2000) and (yavg[mm] < 2000):
            pos[mm] = 'pos2'
        if (xavg[mm] < 2000) and (yavg[mm] < 2000):
            pos[mm] = 'pos3'
        if (xavg[mm] < 2000) and (yavg[mm] > 2000):
            pos[mm] = 'pos4'



    # Print output
    efilt_unique = np.unique(epoch_filt)
    pos_unique = np.unique(pos)
    fmt = '{0:s} {1:s} {2:s}     {3:4.0f} {4:4.0f}'

    for ee in efilt_unique:
        for pp in pos_unique:
            idx = np.where((epoch_filt == ee) & (pos == pp))[0]
            
            if verbose:
                print() 
                for ii in idx:
                    print(fmt.format(mat_root[ii], epoch_filt[ii], pos[ii], xavg[ii], yavg[ii]))
                    
    return mat_root, epoch_filt, pos

def setup_xym_by_pos(reduce_path, mat_root=None, epoch_filt=None, pos=None):
    """
    Something
    """
    mat_pos_path = os.path.join(reduce_path, '03.MAT_POS')

    pos4_exceptions = ['2005_F814W', '2013_F160Ws', '2015_F160Ws', '2024_F160Ws']
    mat_path = f'{reduce_path}/02.MAT/'

    # Get the MAT file names and positions.
    if any([mat_root is None, epoch_filt is None, pos is None]):
        mat_root, epoch_filt, pos = calc_pos_number(reduce_path=reduce_path, verbose=False)

    # Read the old IN.xym2bar file to match MAT.??? to the *.xym files
    mat_xym_files = read_in_xym2mat(f'{mat_path}/IN.xym2mat')
    open_logs = {}
    
    for ii in tqdm(range(len(mat_root)), desc="Copying MAT/XYM files"):
    # for ii in range(1):
        # copy stuff to this directory
        to_dir = epoch_filt[ii]

        if epoch_filt[ii] not in pos4_exceptions:
            to_dir += '_' + pos[ii]

        to_dir = f'{mat_pos_path}/{to_dir}/01.XYM/'
        # print('Copying to {0} for {1}'.format(to_dir, mat_root[ii]))
        
        # Get rid of the filter-related letters for the 02.MAT/ sub-dir.
        efilt_strip = epoch_filt[ii].replace('W', '').replace('F', '').replace('M', '')

        # Copy the old MAT file
        from_file = f'{mat_path}{efilt_strip}/{mat_root[ii]}'
        shutil.copy(from_file, to_dir)
        # print('    Copy ' + from_file + ' to ' + to_dir)

        # Copy the xym file
        mat_num = mat_root[ii].split('.')[1]

        xym_file = mat_xym_files[mat_num]
        # Convert xym_file to absolute path using reduce_path
        # Remove leading "../"
        xym_path = os.path.join(reduce_path, xym_file[3:])
        shutil.copy(xym_path, to_dir)
        # print('    Copy ' + xym_path + ' to ' + to_dir)

        # Keep a record of the match between XYM and MAT files
        logfile = f'{to_dir}/xym_mat.txt'

        if logfile not in list(open_logs.keys()):
            _log = open(logfile, 'w')
            open_logs[logfile] = _log
        else:
            _log = open_logs[logfile]

        # Just save the xym_file file name... not the path.
        xym_file_base = os.path.split(xym_file)[1]
        _log.write('{0} {1}\n'.format(mat_num, xym_file_base))

    # Close all the log files
    for key in open_logs:
        open_logs[key].close()
        
    return


def read_in_xym2mat(in_file):
    _in = open(in_file, "r")
    lines = _in.readlines()
    _in.close()
    
    cnt_mat = len(lines) - 1

    mat_num = np.zeros(cnt_mat, dtype='U3')
    xym_files = np.zeros(cnt_mat, dtype='U80')

    # Skip the first line
    jj = 0
    for ii in range(1, len(lines)):
        # Split the line by spaces
        line_split = lines[ii].split()
        
        # First entry is the MAT file number
        mat_num[jj] = line_split[0]
        xym_files[jj] = line_split[1][1:-1]

        jj += 1

    mat_xym_files = dict(list(zip(mat_num, xym_files)))

    return mat_xym_files


            
def xym_by_pos(reduce_path, year, filt, pos, Nepochs):
    """
    Re-do the alignment with the ref5 master frames (cluster members only) on each position separately.
    Doesn't work for 2005_F814W or 2013_F160Ws or 2015_F160Ws or 2024_F160Ws.
    
    Parameters
    ----------
    reduce_path : str
        The path to the reduction directory where the 03.MAT_POS directory is located.
    year : str
        The year of the observations (e.g., '2005', '2010', '2013', '2015', '2024').
    filt : str
        The filter used in the observations (e.g., 'F125W', 'F139M', 'F160W', 'F160W').
    pos : str
        The position of the observations (e.g., 'pos1', 'pos2', 'pos3', 'pos4').
    Nepochs : int
        The number of images where an object must be detected to make the list, passed as rNIMMIN to xym2bar.
    """
    
    mat_dir = f'{reduce_path}/03.MAT_POS/{year}_{filt}_{pos}/01.XYM'

    master_file = f'MASTER.{filt}.{year}.ref5'

    # Read in the xym_mat.txt file that maps the MAT and XYM files together.
    tab = Table.read(f'{mat_dir}/xym_mat.txt', format='ascii')

    # Make IN.xym2mat and IN.xym2bar file
    _mat = open(f'{mat_dir}/IN.xym2mat', 'w')
    _bar = open(f'{mat_dir}/IN.xym2bar', 'w')

    _mat.write(f'000 {master_file} c0\n')

    for ii in range(len(tab)):
        _mat.write(f'{tab[ii][0]} {tab[ii][1]} c9\n')
        _bar.write(f'{tab[ii][0]} {tab[ii][1]} c9\n')

    _mat.close()
    _bar.close()

    # Run xym2mat
    cwd = os.getcwd()
    os.chdir(mat_dir)
    subprocess.call(['xym2mat', '22'])
    subprocess.call(['xym2mat', '24'])
    subprocess.call(['xym2mat', '25'])

    # Make IN.xym2bar file
    subprocess.call(['xym2bar', str(Nepochs)])

    # Copy final output files
    suffix = f'.{filt}.{year}.{pos}.refClust'
    shutil.copy('MATCHUP.XYMEEE', f'MATCHUP.XYMEEE{suffix}')
    shutil.copy('TRANS.xym2mat', f'TRANS.xym2mat{suffix}')
    shutil.copy('TRANS.xym2bar', f'TRANS.xym2bar{suffix}')
    shutil.copy('IN.xym2mat', f'IN.xym2mat{suffix}')
    shutil.copy('IN.xym2bar', f'IN.xym2bar{suffix}')
    
    os.chdir(cwd)

    return

def make_brite_list_2010(reduce_path):
    """
    Copy over the MATCHUP ref5 files from the 02.MAT directory. It has to be ref5
    because we want the starlists to be matched. The difference between ref5 and
    refClust is very small.
    
    Take an input list of MATCHUP files (assumes they have the same stars, and the
    same length) and trim out only the bright stars. The resulting output file contains
    the X and Y position (from the first file) and the list of all magnitudes for each star.

    trimMags is a list of brightness criteria for each of the matchup files. Any star
    that satisfies this criteria in any one of the filters will be added to the global
    bright list.

    This is a modified version of the code that is in hst_flystar. The modifications
    include:
    - for bright stars, detection in only 1 filter is required.
    - a set of hand selected brite stars are validated and added to the list.
    """
    matchup_file = ['MATCHUP.XYMEEE.F160W.2010.ref5',
                    'MATCHUP.XYMEEE.F139M.2010.ref5',
                    'MATCHUP.XYMEEE.F125W.2010.ref5']
    trimMags = [-8, -7, -8]

    # os.chdir(f'{reduce_path}/12.KS2_2010')
    shutil.copy(f'{reduce_path}/02.MAT/{matchup_file[0]}', f'{reduce_path}/12.KS2_2010/')
    shutil.copy(f'{reduce_path}/02.MAT/{matchup_file[1]}', f'{reduce_path}/12.KS2_2010/')
    shutil.copy(f'{reduce_path}/02.MAT/{matchup_file[2]}', f'{reduce_path}/12.KS2_2010/')

    # Read in the matchup files.
    list_160 = starlists.read_matchup(f'{reduce_path}/12.KS2_2010/{matchup_file[0]}')
    list_139 = starlists.read_matchup(f'{reduce_path}/12.KS2_2010/{matchup_file[1]}')
    list_125 = starlists.read_matchup(f'{reduce_path}/12.KS2_2010/{matchup_file[2]}')

    stars = astropy.table.hstack([list_160, list_139, list_125])
    print('Loaded {0} stars'.format(len(stars)))

    # Trim down based on the magnitude cuts. Non-detections will pass
    # through as long as there is a valid detection in at least one filter.
    good1 = np.where((stars['m_1'] < trimMags[0]) |
                     (stars['m_2'] < trimMags[1]) |
                     (stars['m_3'] < trimMags[2]))[0]

    stars = stars[good1]
    print('Keeping {0} stars that meet brightness cuts'.format(len(stars)))

    # For sources fainter than [-10, -9, -10], they must be in
    # all three epochs.
    keep = np.zeros(len(stars), dtype=bool)

    mag_rng = [-10, -9, -10]
    good2 = np.where((stars['m_1'] > mag_rng[0]) & (stars['m_1'] < 0) &
                     (stars['m_2'] > mag_rng[1]) & (stars['m_2'] < 0) &
                     (stars['m_3'] > mag_rng[2]) & (stars['m_3'] < 0))[0]
    keep[good2] = True

    # For sources between [-14, -13, -14] and [-10, -9, -10], they must be in
    # at least two epochs.
    mag_lo = [-10, -9, -10]
    mag_hi = [-14, -13, -14]
    in12 = np.where((stars['m_1'] > mag_hi[0]) & (stars['m_1'] < mag_lo[0]) &
                    (stars['m_2'] > mag_hi[1]) & (stars['m_2'] < mag_lo[1]))[0]
    in13 = np.where((stars['m_1'] > mag_hi[0]) & (stars['m_1'] < mag_lo[0]) &
                    (stars['m_3'] > mag_hi[2]) & (stars['m_3'] < mag_lo[2]))[0]
    in23 = np.where((stars['m_2'] > mag_hi[1]) & (stars['m_2'] < mag_lo[1]) &
                    (stars['m_3'] > mag_hi[2]) & (stars['m_3'] < mag_lo[2]))[0]

    keep[in12] = True
    keep[in13] = True
    keep[in23] = True

    # For sources brighter than [-14, -13, -14], they must be detected in
    # at least one epoch.
    mag_lim = [-14, -13, -14]
    good3 = np.where((stars['m_1'] < mag_lim[0]) |
                     (stars['m_2'] < mag_lim[1]) |
                     (stars['m_3'] < mag_lim[2]))[0]
    keep[good3] = True

    stars = stars[keep]

    # Save the matchup stars to a file (with errors). Later on,
    # these will be appended to the ks2 output if ks2 doesn't find
    # the bright source. In other wordes, trust the one pass output
    # at the brightest stars.
    brite_obs_125 = open(f'{reduce_path}/12.KS2_2010/BRITE_F125W.XYMEEE', 'w')
    brite_obs_139 = open(f'{reduce_path}/12.KS2_2010/BRITE_F139M.XYMEEE', 'w')
    brite_obs_160 = open(f'{reduce_path}/12.KS2_2010/BRITE_F160W.XYMEEE', 'w')

    for ii in range(len(stars)):
        fmt = '{x:10.4f}  {y:10.4f}  {m:10.4f}  {me:10.4f}  {ye:10.4f}  {ye:10.4f}\n'

        brite_obs_160.write(fmt.format(x=stars['x_1'][ii],
                                       y=stars['y_1'][ii],
                                       m=stars['m_1'][ii],
                                       xe=stars['xe_1'][ii],
                                       ye=stars['ye_1'][ii],
                                       me=stars['me_1'][ii]))
        brite_obs_139.write(fmt.format(x=stars['x_2'][ii],
                                       y=stars['y_2'][ii],
                                       m=stars['m_2'][ii],
                                       xe=stars['xe_2'][ii],
                                       ye=stars['ye_2'][ii],
                                       me=stars['me_2'][ii]))
        brite_obs_125.write(fmt.format(x=stars['x_3'][ii],
                                       y=stars['y_3'][ii],
                                       m=stars['m_3'][ii],
                                       xe=stars['xe_3'][ii],
                                       ye=stars['ye_3'][ii],
                                       me=stars['me_3'][ii]))
    
    brite_obs_125.close()
    brite_obs_139.close()
    brite_obs_160.close()
    

    # Now we need to fix the magnitudes for the non detections amongst the
    # bright sources. I know the rough relationships between brightnesses
    # between these 3 filters, so I will just use those:
    #    F160W = F125W
    #    F139M = F160W + 1.5
    #
    # 1.  detected in F160W but not in F125W
    idx = np.where((stars['m_1'] != 0) & (stars['m_3'] == 0))[0]
    stars['m_3'][idx] = stars['m_1'][idx]
    stars['x_3'][idx] = stars['x_1'][idx]
    stars['y_3'][idx] = stars['y_1'][idx]

    # 2.  detected in F160W but not in F139M
    idx = np.where((stars['m_1'] != 0) & (stars['m_2'] == 0))[0]
    stars['m_2'][idx] = stars['m_1'][idx] + 1.5
    stars['x_2'][idx] = stars['x_1'][idx]
    stars['y_2'][idx] = stars['y_1'][idx]

    # 3.  detected in F125W but not in F160W
    idx = np.where((stars['m_3'] != 0) & (stars['m_1'] == 0))[0]
    stars['m_1'][idx] = stars['m_3'][idx]
    stars['x_1'][idx] = stars['x_3'][idx]
    stars['y_1'][idx] = stars['y_3'][idx]

    # 4.  detected in F125W but not in F139M
    idx = np.where((stars['m_3'] != 0) & (stars['m_2'] == 0))[0]
    stars['m_2'][idx] = stars['m_3'][idx] + 1.5
    stars['x_2'][idx] = stars['x_3'][idx]
    stars['y_2'][idx] = stars['y_3'][idx]

    # 5. detected in F139M but not in F160W
    idx = np.where((stars['m_2'] != 0) & (stars['m_1'] == 0))[0]
    stars['m_1'][idx] = stars['m_2'][idx] - 1.5
    stars['x_1'][idx] = stars['x_2'][idx]
    stars['y_1'][idx] = stars['y_2'][idx]

    # 6. detected in F139M but not in F125W
    idx = np.where((stars['m_2'] != 0) & (stars['m_3'] == 0))[0]
    stars['m_3'][idx] = stars['m_2'][idx] - 1.5
    stars['x_3'][idx] = stars['x_2'][idx]
    stars['y_3'][idx] = stars['y_2'][idx]

    # Double check that everything has a valid magnitude
    idx = np.where((stars['m_1'] < 0) &
                   (stars['m_2'] < 0) &
                   (stars['m_3'] < 0) &
                   (stars['x_1'] != 0) &
                   (stars['y_1'] != 0))[0]
    if len(idx) != len(stars):
        print('FAILED: finding some bright stars with ')
        print('no magnitudes in some filters')
        pdb.set_trace()

    brite = open(f'{reduce_path}/12.KS2_2010/BRITE.XYM', 'w')

    for ii in range(len(stars)):
        fmt = '{x:10.4f}  {y:10.4f}  {m1:10.4f}  {m2:10.4f}  {m3:10.4f}\n'

        brite.write(fmt.format(x=stars['x_1'][ii],
                               y=stars['y_1'][ii],
                               m1=stars['m_1'][ii],
                               m2=stars['m_2'][ii],
                               m3=stars['m_3'][ii]))
    
    brite.close()

    return

def make_brite_list_2024_F160Ws(reduce_path):
    """
    Copy over the MATCHUP ref5 files from the 02.MAT directory. It has to be ref5
    because we want the starlists to be matched. The difference between ref5 and
    refClust is very small.
    
    Take an input list of MATCHUP files (assumes they have the same stars, and the
    same length) and trim out only the bright stars. The resulting output file contains
    the X and Y position (from the first file) and the list of all magnitudes for each star.

    trimMags is a list of brightness criteria for each of the matchup files. Any star
    that satisfies this criteria in any one of the filters will be added to the global
    bright list.

    This is a modified version of the code that is in hst_flystar. The modifications
    include:
    - for bright stars, detection in only 1 filter is required.
    - a set of hand selected brite stars are validated and added to the list.
    """
    matchup_file = ['MATCHUP.XYMEEE.F160Ws.2024.ref5']
    trimMags = [-8, -7, -8]

    # os.chdir(f'{reduce_path}/12.KS2_2010')
    shutil.copy(f'{reduce_path}/02.MAT/{matchup_file[0]}', f'{reduce_path}/15.KS2_2024/')

    # Read in the matchup files.
    list_160 = starlists.read_matchup(f'{reduce_path}/15.KS2_2024/{matchup_file[0]}')

    stars = astropy.table.hstack([list_160])
    print('Loaded {0} stars'.format(len(stars)))

    # Trim down based on the magnitude cuts. Non-detections will pass
    # through as long as there is a valid detection in at least one filter.
    good1 = np.where((stars['m'] < trimMags[0]))[0]

    stars = stars[good1]
    print('Keeping {0} stars that meet brightness cuts'.format(len(stars)))

    # For sources fainter than [-10, -9, -10], they must be in
    # all three epochs.
    keep = np.zeros(len(stars), dtype=bool)

    mag_rng = [-10, -9, -10]
    good2 = np.where((stars['m'] > mag_rng[0]) & (stars['m'] < 0))[0]
    keep[good2] = True

    # For sources between [-14, -13, -14] and [-10, -9, -10], they must be in
    # at least two epochs.
    mag_lo = [-10, -9, -10]
    mag_hi = [-14, -13, -14]
    in12 = np.where((stars['m'] > mag_hi[0]) & (stars['m'] < mag_lo[0]))[0]

    keep[in12] = True

    # For sources brighter than [-14, -13, -14], they must be detected in
    # at least one epoch.
    mag_lim = [-14, -13, -14]
    good3 = np.where((stars['m'] < mag_lim[0]))[0]
    keep[good3] = True

    stars = stars[keep]

    # Save the matchup stars to a file (with errors). Later on,
    # these will be appended to the ks2 output if ks2 doesn't find
    # the bright source. In other wordes, trust the one pass output
    # at the brightest stars.
    brite_obs_160 = open(f'{reduce_path}/15.KS2_2024/BRITE_F160Ws.XYM', 'w')

    for ii in range(len(stars)):
        fmt = '{x:10.4f}  {y:10.4f}  {m:10.4f}\n'

        brite_obs_160.write(fmt.format(x=stars['x'][ii],
                                       y=stars['y'][ii],
                                       m=stars['m'][ii]))

    brite_obs_160.close()
    

    # Double check that everything has a valid magnitude
    idx = np.where((stars['m'] < 0) &
                   (stars['x'] != 0) &
                   (stars['y'] != 0))[0]
    if len(idx) != len(stars):
        print('FAILED: finding some bright stars with ')
        print('no magnitudes in some filters')
        pdb.set_trace()

    return

def find_top_stars(reread=False):
    """
    Select some stars to be our topStars named sources.
    Verify that they are detected in all of the
    epochs and all of the filters.
    """
    root_2005_814 = work_dir + '/11.KS2_2005/nimfo2bar.xymeee.ks2.F1'
    root_2010_160 = work_dir + '/12.KS2_2010/nimfo2bar.xymeee.F1'
    root_2010_139 = work_dir + '/12.KS2_2010/nimfo2bar.xymeee.F2'
    root_2010_125 = work_dir + '/12.KS2_2010/nimfo2bar.xymeee.F3'
    root_2013_160 = work_dir + '/13.KS2_2013/nimfo2bar.xymeee.F1'
    root_2013_160s = work_dir + '/13.KS2_2013/nimfo2bar.xymeee.F2'
    root_2015_160 = work_dir + '/14.KS2_2015/nimfo2bar.xymeee.F1'
    root_2015_160s = work_dir + '/14.KS2_2015/nimfo2bar.xymeee.F2'
    root_2024_160 = work_dir + '/15.KS2_2024/nimfo2bar.xymeee.F1'
    root_2024_160s = work_dir + '/15.KS2_2024/nimfo2bar.xymeee.F2'

    print('Reading data')
    if reread == True:
        t_2005_814 = starlists.read_nimfo2bar(root_2005_814)
        t_2010_160 = starlists.read_nimfo2bar(root_2010_160)
        t_2010_139 = starlists.read_nimfo2bar(root_2010_139)
        t_2010_125 = starlists.read_nimfo2bar(root_2010_125)
        t_2013_160 = starlists.read_nimfo2bar(root_2013_160)
        t_2013_160s = starlists.read_nimfo2bar(root_2013_160s)
        t_2015_160 = starlists.read_nimfo2bar(root_2015_160)
        t_2015_160s = starlists.read_nimfo2bar(root_2015_160s)
        t_2024_160 = starlists.read_nimfo2bar(root_2024_160)
        t_2024_160s = starlists.read_nimfo2bar(root_2024_160s)

        t_2005_814.write(root_2005_814 + '.fits', overwrite=True)
        t_2010_160.write(root_2010_160 + '.fits', overwrite=True)
        t_2010_139.write(root_2010_139 + '.fits', overwrite=True)
        t_2010_125.write(root_2010_125 + '.fits', overwrite=True)
        t_2013_160.write(root_2013_160 + '.fits', overwrite=True)
        t_2013_160s.write(root_2013_160s + '.fits', overwrite=True)
        t_2015_160.write(root_2015_160 + '.fits', overwrite=True)
        t_2015_160s.write(root_2015_160s + '.fits', overwrite=True)
        t_2024_160.write(root_2024_160 + '.fits', overwrite=True)
        t_2024_160s.write(root_2024_160s + '.fits', overwrite=True)
    else:
        t_2005_814 = Table.read(root_2005_814 + '.fits')
        t_2010_160 = Table.read(root_2010_160 + '.fits')
        t_2010_139 = Table.read(root_2010_139 + '.fits')
        t_2010_125 = Table.read(root_2010_125 + '.fits')
        t_2013_160 = Table.read(root_2013_160 + '.fits')
        t_2013_160s = Table.read(root_2013_160s + '.fits')
        t_2015_160 = Table.read(root_2015_160 + '.fits')
        t_2015_160s = Table.read(root_2015_160s + '.fits')
        t_2024_160 = Table.read(root_2024_160 + '.fits')
        

    # Trim them all down to some decent magnitude ranges. These
    # were chosen by a quick look at the CMD from the one-pass analysis.
    print('Cutting out faint stars')
    lo814 = -13 #+ 3
    lo160 = -9 #+ 3
    lo139 = -7.35 #+ 3
    lo125 = -9.2 #+ 3

    # First pass, cuts are courser so that we can measure crowding.
    g_2005_814 = np.where(t_2005_814['m'] < lo814)[0]
    g_2010_125 = np.where(t_2010_125['m'] < lo125)[0]
    g_2010_139 = np.where(t_2010_139['m'] < lo139)[0]
    g_2010_160 = np.where(t_2010_160['m'] < lo160)[0]
    g_2013_160 = np.where(t_2013_160['m'] < lo160)[0]
    g_2013_160s = np.where(t_2013_160s['m'] < lo160)[0]
    g_2015_160 = np.where(t_2015_160['m'] < lo160)[0]
    g_2015_160s = np.where(t_2015_160s['m'] < lo160)[0]
    g_2024_160 = np.where(t_2024_160['m'] < lo160)[0]
    g_2024_160s = np.where(t_2024_160s['m'] < lo160)[0]

    t_2005_814 = t_2005_814[g_2005_814]
    t_2010_125 = t_2010_125[g_2010_125]
    t_2010_139 = t_2010_139[g_2010_139]
    t_2010_160 = t_2010_160[g_2010_160]
    t_2013_160 = t_2013_160[g_2013_160]
    t_2013_160s = t_2013_160s[g_2013_160s]
    t_2015_160 = t_2015_160[g_2015_160]
    t_2015_160s = t_2015_160s[g_2015_160s]
    t_2024_160 = t_2024_160[g_2024_160]
    t_2024_160s = t_2024_160s[g_2024_160s]

    # Cross match all the sources. All positions should agree to within
    # a pixel. Do this consecutively so that you build up the good matches.
    print('Matching 2010_160 and 2010_125')
    r1 = align.match(t_2010_160['x'], t_2010_160['y'], t_2010_160['m'],
                     t_2010_125['x'], t_2010_125['y'], t_2010_125['m'] - 0.2,
                     dr_tol=0.5, dm_tol=1.0)
    print('    found {0:d} matches'.format(len(r1[0])))
    t_2010_160 = t_2010_160[r1[0]]
    t_2010_125 = t_2010_125[r1[1]]
    
    print('Matching 2010_160 and 2010_139')
    r2 = align.match(t_2010_160['x'], t_2010_160['y'], t_2010_160['m'],
                     t_2010_139['x'], t_2010_139['y'], t_2010_139['m'] - 1.65,
                     dr_tol=0.5, dm_tol=1.0)
    print('    found {0:d} matches'.format(len(r2[0])))
    t_2010_160 = t_2010_160[r2[0]]
    t_2010_125 = t_2010_125[r2[0]]
    t_2010_139 = t_2010_139[r2[1]]

    print('Matching 2010_160 and 2013_160')
    r3 = align.match(t_2010_160['x'], t_2010_160['y'], t_2010_160['m'],
                     t_2013_160['x'], t_2013_160['y'], t_2013_160['m'],
                     dr_tol=0.5, dm_tol=1.0)
    print('    found {0:d} matches'.format(len(r3[0])))
    t_2010_160 = t_2010_160[r3[0]]
    t_2010_125 = t_2010_125[r3[0]]
    t_2010_139 = t_2010_139[r3[0]]
    t_2013_160 = t_2013_160[r3[1]]

    print('Matching 2010_160 and 2013_160s')
    r4 = align.match(t_2010_160['x'], t_2010_160['y'], t_2010_160['m'],
                     t_2013_160s['x'], t_2013_160s['y'], t_2013_160s['m'],
                     dr_tol=0.5, dm_tol=1.0)
    print('    found {0:d} matches'.format(len(r4[0])))
    t_2010_160 = t_2010_160[r4[0]]
    t_2010_125 = t_2010_125[r4[0]]
    t_2010_139 = t_2010_139[r4[0]]
    t_2013_160 = t_2013_160[r4[0]]
    t_2013_160s = t_2013_160s[r4[1]]

    print('Matching 2010_160 and 2015_160')
    r5 = align.match(t_2010_160['x'], t_2010_160['y'], t_2010_160['m'],
                     t_2015_160['x'], t_2015_160['y'], t_2015_160['m'],
                     dr_tol=0.5, dm_tol=1.0)
    print('    found {0:d} matches'.format(len(r5[0])))
    t_2010_160 = t_2010_160[r5[0]]
    t_2010_125 = t_2010_125[r5[0]]
    t_2010_159 = t_2010_159[r5[0]]
    t_2013_160 = t_2013_160[r5[0]]
    t_2013_160s = t_2013_160s[r5[0]]
    t_2015_160 = t_2015_160[r5[1]]

    print('Matching 2010_160 and 2015_160s')
    r6 = align.match(t_2010_160['x'], t_2010_160['y'], t_2010_160['m'],
                     t_2015_160s['x'], t_2015_160s['y'], t_2015_160s['m'],
                     dr_tol=0.5, dm_tol=1.0)
    print('    found {0:d} matches'.format(len(r6[0])))
    t_2010_160 = t_2010_160[r6[0]]
    t_2010_125 = t_2010_125[r6[0]]
    t_2010_159 = t_2010_159[r6[0]]
    t_2013_160 = t_2013_160[r6[0]]
    t_2013_160s = t_2013_160s[r6[0]]
    t_2015_160 = t_2015_160[r6[0]]
    t_2015_160s = t_2015_160s[r6[1]]
    
    print('Matching 2010_160 and 2024_160')
    r7 = align.match(t_2010_160['x'], t_2010_160['y'], t_2010_160['m'],
                     t_2024_160['x'], t_2024_160['y'], t_2024_160['m'],
                     dr_tol=0.5, dm_tol=1.0)
    print('    found {0:d} matches'.format(len(r6[0])))
    t_2010_160 = t_2010_160[r7[0]]
    t_2010_125 = t_2010_125[r7[0]]
    t_2010_159 = t_2010_159[r7[0]]
    t_2013_160 = t_2013_160[r7[0]]
    t_2013_160s = t_2013_160s[r7[0]]
    t_2015_160 = t_2015_160[r7[0]]
    t_2015_160s = t_2015_160s[r7[0]]
    t_2024_160 = t_2024_160[r7[1]]
    
    print('Matching 2010_160 and 2024_160s')
    r8 = align.match(t_2010_160['x'], t_2010_160['y'], t_2010_160['m'],
                     t_2024_160s['x'], t_2024_160s['y'], t_2024_160s['m'],
                     dr_tol=0.5, dm_tol=1.0)
    print('    found {0:d} matches'.format(len(r6[0])))
    t_2010_160 = t_2010_160[r8[0]]
    t_2010_125 = t_2010_125[r8[0]]
    t_2010_159 = t_2010_159[r8[0]]
    t_2013_160 = t_2013_160[r8[0]]
    t_2013_160s = t_2013_160s[r8[0]]
    t_2015_160 = t_2015_160[r8[0]]
    t_2015_160s = t_2015_160s[r8[0]]
    t_2024_160 = t_2024_160[r8[0]]
    t_2024_160s = t_2024_160s[r8[1]]

    print('Matching 2010_160 and 2005_814')
    r9 = align.match(t_2010_160['x'], t_2010_160['y'], t_2010_160['m'],
                     t_2005_814['x'], t_2005_814['y'], t_2005_814['m'] + 4.2,
                     dr_tol=0.5, dm_tol=1.0)
    print('    found {0:d} matches'.format(len(r7[0])))
    t_2010_160 = t_2010_160[r9[0]]
    t_2010_125 = t_2010_125[r9[0]]
    t_2010_139 = t_2010_139[r9[0]]
    t_2013_160 = t_2013_160[r9[0]]
    t_2013_160s = t_2013_160s[r9[0]]
    t_2015_160 = t_2015_160[r9[0]]
    t_2015_160s = t_2015_160s[r9[0]]
    t_2005_814 = t_2005_814[r9[1]]

    # Sort by brightness and print
    sdx = t_2010_160['m'].argsort()
    
    fmt = '{name:10s}  {x:8.2f} {y:8.2f}  '
    fmt += '{m160:5.1f} {m139:5.1f} {m125:5.1f} {m814:5.1f}'

    nameIdx = 1

    _out = open('wd1_named_stars.txt', 'w')
    for ii in sdx:
        name = 'wd1_{0:05d}'.format(nameIdx)
        print(fmt.format(name = name,
                         x = t_2010_160['x'][ii],
                         y = t_2010_160['y'][ii],
                         m160 = t_2010_160['m'][ii],
                         m139 = t_2010_139['m'][ii],
                         m125 = t_2010_125['m'][ii],
                         m814 = t_2005_814['m'][ii]))
        _out.write(fmt.format(name = name,
                            x = t_2010_160['x'][ii],
                            y = t_2010_160['y'][ii],
                            m160 = t_2010_160['m'][ii],
                            m139 = t_2010_139['m'][ii],
                            m125 = t_2010_125['m'][ii],
                            m814 = t_2005_814['m'][ii]))
        _out.write('\n')
                            
        nameIdx += 1

    _out.close()

    return

def combine_nimfo2bar_brite(reduce_path):
    """
    For all the data, we need to add back in the missing bright stars
    from the one-pass analysis.
    """
    epochs = ['2005_F814W', 
              '2010_F125W_pos1', '2010_F125W_pos2', '2010_F125W_pos3', '2010_F125W_pos4',
              '2010_F139M_pos1', '2010_F139M_pos2', '2010_F139M_pos3', '2010_F139M_pos4',
              '2010_F160W_pos1', '2010_F160W_pos2', '2010_F160W_pos3', '2010_F160W_pos4',
              '2013_F160W_pos1', '2013_F160W_pos2', '2013_F160W_pos3', '2013_F160W_pos4',
              '2013_F160Ws',
              '2015_F160W_pos1', '2015_F160W_pos2', '2015_F160W_pos3', '2015_F160W_pos4',
              '2015_F160Ws',
              '2024_F160W_pos1', '2024_F160W_pos2', '2024_F160W_pos3', '2024_F160W_pos4',
              '2024_F160Ws'
              ]
    magCuts = [-13.5, 
               -10, -10, -10, -10,
                -9,  -9,  -9,  -9,
               -10, -10, -10, -10,
               -10, -10, -10, -10,
               -12,
               -10, -10, -10, -10,
               -12,
               -10, -10, -10, -10,
               -12
               ]
               
    
    for ee in range(len(epochs)):
        ep = epochs[ee]
        year = ep.split('_')[0]
        filt = ep.split('_')[1]

        mat_dir =  f'{reduce_path}/03.MAT_POS/' + ep + '/01.XYM/'
        mat_file = 'MATCHUP.XYMEEE'
        ks2_dir = reduce_path
        ks2_file = 'nimfo2bar.xymeee.F1'
        N_filter = 1
        if year == '2005':
            ks2_dir += '/11.KS2_2005/'
            ks2_file = ks2_file.replace('.F1', '.ks2.F1')

        if year == '2010':
            ks2_dir += '/12.KS2_2010/0{pos}.pos{pos}/'.format(pos=ep[-1])
            N_filter = 3
            if filt == 'F139M':
                ks2_file = ks2_file.replace('.F1', '.F2')
            if filt == 'F125W':
                ks2_file = ks2_file.replace('.F1', '.F3')

        if year == '2013':
            ks2_dir += '/13.KS2_2013/'
            if ep.endswith('s'):
                ks2_dir += '05.F160Ws/'
            else:
                ks2_dir += '0{pos}.pos{pos}/'.format(pos=ep[-1])

        if year == '2015':
            ks2_dir += '/14.KS2_2015/'
            if ep.endswith('s'):
                ks2_dir += '05.F160Ws/'
            else:
                ks2_dir += '0{pos}.pos{pos}/'.format(pos=ep[-1])
        
        if year == '2024':
            ks2_dir += '/15.KS2_2024/'
            if ep.endswith('s'):
                ks2_dir += '05.F160Ws/'
            else:
                ks2_dir += '0{pos}.pos{pos}/'.format(pos=ep[-1])
        
        one = starlists.read_matchup(mat_dir + mat_file)
        ks2 = starlists.read_nimfo2bar(ks2_dir + ks2_file)

        if 'F160Ws' in ks2_dir:
            # In the short-exposure data, most sources only detected
            # in a single frame and has no errors. We still want them.
            idx = np.where(one['m'] <= magCuts[ee])[0]
        else:
            print('magCut = ', magCuts[ee], ' for ', ep)
            idx = np.where((one['m'] <= magCuts[ee]) & (one['me'] < 1))[0]

        # Trim down to just the brite sources in the MATCHUP
        one_brite = one[idx]

        # Get the last ks2 star name. 
        last_name = int(ks2[-1]['name'])

        # Loop through each brite star and see if it 
        # was detected in ks2.
        cnt_add = 0
        cnt_replace = 0

        for ii in range(len(one_brite)):
            dx = one_brite[ii]['x'] - ks2['x']
            dy = one_brite[ii]['y'] - ks2['y']
            dm = one_brite[ii]['m'] - ks2['m']
            dr = np.hypot(dx, dy)

            rdx = dr.argmin()

            dr_min = dr[rdx]
            dm_min = dm[rdx]

            # Decide whether to add or replace this star in KS2
            add = False
            replace = False

            # Star doesn't exist in KS2 - add it.
            if (dr_min > 1):
                add = True
            else:
                # There is a star, but it has a wildly different
                # magnitude. Replace it.
                if (dm_min > 1):
                    replace = True

                # There is a star in KS2; but it doesn't have errors
                # and the one-pass analysis does. Replace it.
                if ((ks2[rdx]['xe'] == 1) and (one_brite[ii]['xe'] != 1)):
                    replace = True
            


            if add:
                new_name = last_name + 1

                ks2.add_row([one_brite[ii]['x'], one_brite[ii]['y'],
                             one_brite[ii]['m'], one_brite[ii]['xe'],
                             one_brite[ii]['ye'], one_brite[ii]['me'],
                             one_brite[ii]['N_fnd'], one_brite[ii]['N_xywell'],
                             str(new_name).zfill(6)])
                last_name += 1

                cnt_add += 1

            if replace:
                ks2[rdx]['x'] = one_brite[ii]['x']
                ks2[rdx]['y'] = one_brite[ii]['y']
                ks2[rdx]['m'] = one_brite[ii]['m']
                ks2[rdx]['xe'] = one_brite[ii]['xe']
                ks2[rdx]['ye'] = one_brite[ii]['ye']
                ks2[rdx]['me'] = one_brite[ii]['me']
                ks2[rdx]['N_fnd'] = one_brite[ii]['N_fnd']
                ks2[rdx]['N_xywell'] = one_brite[ii]['N_xywell']

                cnt_replace += 1
                
        fmt =  '{0:s}: Added {1:d}, replace {2:d} out of {3:d} brite stars'
        print(fmt.format(ks2_dir + ks2_file, cnt_add, cnt_replace, len(one_brite)))
        ks2.write(ks2_dir + ks2_file.replace('2bar', '2bar_brite'), 
                  format='ascii.fixed_width_no_header', delimiter=' ',
                  formats={'x': '%10.4f', 'y': '%10.4f', 'm': '%10.3f', 
                           'xe': '%10.4f', 'ye': '%10.4f', 'me': '%10.3f',
                           'N_fnd': '%3d', 'N_xywell': '%3d', 'name': lambda n: n.zfill(6)},
                  overwrite=True)

            
    return
    
def make_catalog_from_align(align_root):
    """
    Given starlist output from align, returns to MATCHUP format. Inputs are the
    align *.pos, *.err, *.mag, *.param, and *.name files. Will return 3 MATCHUP
    files, one for each epoch. Align run order:

    2005_F814W, 2010_F125W, 2010_F139M, 2010_F160W, 2013_F160W, 2013_f160Ws

    """
    root_dir = work_dir + '/50.ALIGN_KS2/'

    # Setup the mapping between epoch, filter, pos and the align index.
    align_idx = ['2005_F814W',
                 '2010_F125W_pos1',
                 '2010_F125W_pos2',
                 '2010_F125W_pos3',
                 '2010_F125W_pos4',
                 '2010_F139M_pos1',
                 '2010_F139M_pos2',
                 '2010_F139M_pos3',
                 '2010_F139M_pos4',
                 '2010_F160W_pos1',
                 '2010_F160W_pos2',
                 '2010_F160W_pos3',
                 '2010_F160W_pos4',
                 '2013_F160W_pos1',
                 '2013_F160W_pos2',
                 '2013_F160W_pos3',
                 '2013_F160W_pos4',
                 '2013_F160Ws',
                 '2015_F160W_pos1',
                 '2015_F160W_pos2',
                 '2015_F160W_pos3',
                 '2015_F160W_pos4',
                 '2015_F160Ws']

    ast.make_catalog_from_align(align_root, root_dir=root_dir, epoch_names=align_idx)

    return

def combine_mosaic_pos(catalog_name_in, catalog_name_out):
    t = Table.read(work_dir + '/50.ALIGN_KS2/' + catalog_name_in)
    
    # First we need to combine all the positions of the mosaic together.
    mosaic_epochs = ['2010_F125W', '2010_F139M', '2010_F160W', '2013_F160W', '2015_F160W']
    pos = ['pos1', 'pos2', 'pos3', 'pos4']

    n_epochs = len(mosaic_epochs)
    n_stars = len(t)
    x_wavg = np.zeros((n_epochs, n_stars), dtype=float)
    y_wavg = np.zeros((n_epochs, n_stars), dtype=float)
    f_wavg = np.zeros((n_epochs, n_stars), dtype=float)
    m_wavg = np.zeros((n_epochs, n_stars), dtype=float)
    xe_wavg = np.zeros((n_epochs, n_stars), dtype=float)
    ye_wavg = np.zeros((n_epochs, n_stars), dtype=float)
    fe_wavg = np.zeros((n_epochs, n_stars), dtype=float)
    me_wavg = np.zeros((n_epochs, n_stars), dtype=float)
    wx_wavg = np.zeros((n_epochs, n_stars), dtype=float)
    wy_wavg = np.zeros((n_epochs, n_stars), dtype=float)
    wf_wavg = np.zeros((n_epochs, n_stars), dtype=float)
    nframes = np.zeros((n_epochs, n_stars), dtype=float)

    # We will need to keep track of the modified years area.
    years_orig = np.array(t.meta['YEARS'])  # original
    years = np.zeros(n_epochs, dtype=float) # Keep new years.

    # Loop through each year+filter combination    
    for ii in range(n_epochs):
        print('Working on: ' + mosaic_epochs[ii])

        # This will store the columns we will need to delete.
        epochs_idx = []
        
        for pp in range(len(pos)):
            ee = t.meta['EPNAMES'].index(mosaic_epochs[ii] + '_' + pos[pp])
            
            epochs_idx.append(ee)
            
            x = t['x'][:, ee]
            y = t['y'][:, ee]
            m = t['m'][:, ee]
            xe = t['xe'][:, ee]
            ye = t['ye'][:, ee]
            me = t['me'][:, ee]
            n = t['n'][:, ee]
                  
            # Identify the stars with detections in this list.
            det = np.where((x > -1e4) & (xe > 0) &
                           (y > -1e4) & (ye > 0) &
                           (m > 0) & (me > 0))[0]

            # Convert the magnitudes to fluxes
            f_det, fe_det = photo.mag2flux(m[det], me[det])
            
            # Calculate the weights
            w_x = 1.0 / xe[det]**2
            w_y = 1.0 / ye[det]**2
            # w_f = 1.0 / fe_det**2   - Too big... mags are good enough.
            w_f = 1.0 / me[det]**2

            # Adding to the weighted average calculation
            x_wavg[ii, det] += x[det] * w_x
            y_wavg[ii, det] += y[det] * w_y
            f_wavg[ii, det] += f_det * w_f
            # me = inf carries no photometric weight (w_f = 1/inf**2 = 0), but
            # mag2flux turns it into fe_det = inf, so the product is inf*0 = nan
            # and would poison fe_wavg -- and with it the combined me. Zero the
            # operand rather than the product: np.where evaluates both branches,
            # so guarding the product still triggers the invalid multiply.
            fe_wavg[ii, det] += np.where(np.isfinite(fe_det), fe_det, 0.0) * w_f

            wx_wavg[ii, det] += w_x
            wy_wavg[ii, det] += w_y
            wf_wavg[ii, det] += w_f

            nframes[ii, det] += n[det]

        # Get the average year for all the positions
        # years[ii] = years_orig[epochs_idx].mean() # Wrong!
        years[ii] = np.array(t.meta['YEARS'])[epochs_idx].mean()
            
        # Finished this epoch, finish the weighted average calculation.
        # Stars with no detections in this epoch have wavg == 0 and weight
        # == 0, so this is a deliberate 0/0 -> nan (undetected position) and
        # 1/0 -> inf (undetected, infinite uncertainty); suppress the
        # resulting warnings rather than the (correct) nan/inf results.
        with np.errstate(invalid='ignore', divide='ignore'):
            x_wavg[ii, :] /= wx_wavg[ii, :]
            xe_wavg[ii, :] = (1.0 / wx_wavg[ii, :])**0.5

            y_wavg[ii, :] /= wy_wavg[ii, :]
            ye_wavg[ii, :] = (1.0 / wy_wavg[ii, :])**0.5

            f_wavg[ii, :] /= wf_wavg[ii, :]
            # fe_wavg[ii, :] = (1.0 / wf_wavg[ii, :])**0.5
            fe_wavg[ii, :] /= wf_wavg[ii, :]  # Spe

        m_wavg[ii, :], me_wavg[ii, :] = photo.flux2mag(f_wavg[ii, :], fe_wavg[ii, :])

        # Delete the old columns with individual positions
        t['x'] = np.delete(t['x'], epochs_idx, axis=1)
        t['y'] = np.delete(t['y'], epochs_idx, axis=1)
        t['m'] = np.delete(t['m'], epochs_idx, axis=1)
        t['xe'] = np.delete(t['xe'], epochs_idx, axis=1)
        t['ye'] = np.delete(t['ye'], epochs_idx, axis=1)
        t['me'] = np.delete(t['me'], epochs_idx, axis=1)
        t['n'] = np.delete(t['n'], epochs_idx, axis=1)
        t.meta['EPNAMES'] = np.delete(t.meta['EPNAMES'], epochs_idx).tolist()
        t.meta['YEARS'] = np.delete(t.meta['YEARS'], epochs_idx).tolist()

        # Add in the new column with the combined list.
        t['x'] = np.append(t['x'], x_wavg[[ii], :].T, axis=1)
        t['y'] = np.append(t['y'], y_wavg[[ii], :].T, axis=1)
        t['m'] = np.append(t['m'], m_wavg[[ii], :].T, axis=1)
        t['xe'] = np.append(t['xe'], xe_wavg[[ii], :].T, axis=1)
        t['ye'] = np.append(t['ye'], ye_wavg[[ii], :].T, axis=1)
        t['me'] = np.append(t['me'], me_wavg[[ii], :].T, axis=1)
        t['n'] = np.append(t['n'], nframes[[ii], :].T, axis=1)
        t.meta['EPNAMES'] = np.append(t.meta['EPNAMES'], mosaic_epochs[ii]).tolist()
        t.meta['YEARS'] = np.append(t.meta['YEARS'], years[ii]).tolist()

    # Remove the xorig and yorig for the other epochs
    t.remove_column('xorig')
    t.remove_column('yorig')

    t.write(work_dir + '/50.ALIGN_KS2/' + catalog_name_out,
            format='fits', overwrite=True)


def combine_mosaic_positions(wd1_catalog, mosaic_epochs):
    wd1 = wd1_catalog.copy()
    with_m_orig = 'm_orig' in wd1.colnames
    if with_m_orig:
        wd1 = wd1[['name', 'x', 'y', 'm', 'm_orig', 'xe', 'ye', 'me', 'n']]
    else:
        wd1 = wd1[['name', 'x', 'y', 'm', 'xe', 'ye', 'me', 'n']]
    # wd1.rename_column('m_orig', 'm')
    # First we need to combine all the positions of the mosaic together.
    # mosaic_epochs = ['2010_F125W', '2010_F139M', '2010_F160W', '2013_F160W', '2015_F160W', '2024_F160W']
    pos = ['pos1', 'pos2', 'pos3', 'pos4']
    n_pos = len(pos)

    n_epochs = len(mosaic_epochs)
    n_stars = len(wd1)
    x_wavg = np.zeros((n_epochs, n_stars), dtype=float)
    y_wavg = np.zeros((n_epochs, n_stars), dtype=float)
    f_wavg = np.zeros((n_epochs, n_stars), dtype=float)
    m_wavg = np.zeros((n_epochs, n_stars), dtype=float)
    if with_m_orig:
        f_wavg_orig = np.zeros((n_epochs, n_stars), dtype=float)
        m_wavg_orig = np.zeros((n_epochs, n_stars), dtype=float)
    xe_wavg = np.zeros((n_epochs, n_stars), dtype=float)
    ye_wavg = np.zeros((n_epochs, n_stars), dtype=float)
    fe_wavg = np.zeros((n_epochs, n_stars), dtype=float)
    me_wavg = np.zeros((n_epochs, n_stars), dtype=float)
    wx_wavg = np.zeros((n_epochs, n_stars), dtype=float)
    wy_wavg = np.zeros((n_epochs, n_stars), dtype=float)
    wf_wavg = np.zeros((n_epochs, n_stars), dtype=float)
    nframes = np.zeros((n_epochs, n_stars), dtype=float)

    # We will need to keep track of the modified years area.
    years_orig = np.array(wd1.meta['list_times'])  # original
    years = np.zeros(n_epochs, dtype=float) # Keep new years.

    # Loop through each year+filter combination    
    for ii in range(n_epochs):
        print('Working on: ' + mosaic_epochs[ii])

        # This will store the columns we will need to delete.
        epochs_idx = []

        for pp in range(n_pos):
            ee = wd1.meta['EPNAMES'].index(mosaic_epochs[ii] + '_' + pos[pp])

            epochs_idx.append(ee)

            x = wd1['x'][:, ee]
            y = wd1['y'][:, ee]
            m = wd1['m'][:, ee]
            if 'm_orig' in wd1.colnames:
                m_orig = wd1['m_orig'][:, ee]
            xe = wd1['xe'][:, ee]
            ye = wd1['ye'][:, ee]
            me = wd1['me'][:, ee]
            n = wd1['n'][:, ee]

            # Identify the stars with detections in this list.
            det = np.where((x > -1e4) & (xe > 0) &
                           (y > -1e4) & (ye > 0) &
                           (m > 0) & (me > 0))[0]
            print(f'Number of detections in {mosaic_epochs[ii]}_{pos[pp]}: {len(det)}')

            # Convert the magnitudes to fluxes. m_orig only exists when the
            # input table carried it, so its conversion has to sit behind the
            # same guard as its use below -- otherwise this is a NameError for
            # any table without that column. Its flux error is never needed:
            # the m_orig average reuses w_f, computed from me.
            f_det, fe_det = photo.mag2flux(m[det], me[det])
            if with_m_orig:
                f_det_orig, _ = photo.mag2flux(m_orig[det], me[det])

            # Calculate the weights
            w_x = 1.0 / xe[det]**2
            w_y = 1.0 / ye[det]**2
            # w_f = 1.0 / fe_det**2   - Too big... mags are good enough.
            w_f = 1.0 / me[det]**2

            # Adding to the weighted average calculation
            x_wavg[ii, det] += x[det] * w_x
            y_wavg[ii, det] += y[det] * w_y
            f_wavg[ii, det] += f_det * w_f
            # A measurement with me = inf carries no photometric weight
            # (w_f = 1/inf**2 = 0), but mag2flux turns it into fe_det = inf, so
            # fe_det * w_f is inf*0 = nan -- which poisons fe_wavg, and with it
            # the combined me for that star. Contributing 0 is exactly what zero
            # weight means. Such a star deliberately stays in det: w_x/w_y come
            # from xe/ye rather than me, so its POSITION is still perfectly
            # usable and only the photometry has to be skipped.
            # np.where would still evaluate inf*0 in the discarded branch and
            # warn, so zero the operand instead of the product.
            fe_wavg[ii, det] += np.where(np.isfinite(fe_det), fe_det, 0.0) * w_f
            if with_m_orig:
                f_wavg_orig[ii, det] += f_det_orig * w_f
            nframes[ii, det] += n[det]

            wx_wavg[ii, det] += w_x
            wy_wavg[ii, det] += w_y
            wf_wavg[ii, det] += w_f


        # Get the average year for all the positions
        # years[ii] = years_orig[epochs_idx].mean() # Wrong!
        years[ii] = np.array(wd1.meta['list_times'])[epochs_idx].mean()

        # Finished this epoch, finish the weighted average calculation.
        # Stars with no detections in this epoch have wavg == 0 and weight
        # == 0, so this is a deliberate 0/0 -> nan (undetected position) and
        # 1/0 -> inf (undetected, infinite uncertainty); suppress the
        # resulting warnings rather than the (correct) nan/inf results.
        with np.errstate(invalid='ignore', divide='ignore'):
            x_wavg[ii, :] /= wx_wavg[ii, :]
            xe_wavg[ii, :] = (1.0 / wx_wavg[ii, :])**0.5

            y_wavg[ii, :] /= wy_wavg[ii, :]
            ye_wavg[ii, :] = (1.0 / wy_wavg[ii, :])**0.5

            f_wavg[ii, :] /= wf_wavg[ii, :]
            if with_m_orig:
                f_wavg_orig[ii, :] /= wf_wavg[ii, :]
            # fe_wavg[ii, :] = (1.0 / wf_wavg[ii, :])**0.5
            fe_wavg[ii, :] /= wf_wavg[ii, :]  # Spe

        m_wavg[ii, :], me_wavg[ii, :] = photo.flux2mag(f_wavg[ii, :], fe_wavg[ii, :])
        if with_m_orig:
            m_wavg_orig[ii, :], me_wavg[ii, :] = photo.flux2mag(f_wavg_orig[ii, :], fe_wavg[ii, :])

        # Delete the old columns with individual positions
        wd1['x'] = np.delete(wd1['x'], epochs_idx, axis=1)
        wd1['y'] = np.delete(wd1['y'], epochs_idx, axis=1)
        wd1['m'] = np.delete(wd1['m'], epochs_idx, axis=1)
        if with_m_orig:
            wd1['m_orig'] = np.delete(wd1['m_orig'], epochs_idx, axis=1)
        wd1['xe'] = np.delete(wd1['xe'], epochs_idx, axis=1)
        wd1['ye'] = np.delete(wd1['ye'], epochs_idx, axis=1)
        wd1['me'] = np.delete(wd1['me'], epochs_idx, axis=1)
        wd1['n'] = np.delete(wd1['n'], epochs_idx, axis=1)
        wd1.meta['EPNAMES'] = np.delete(wd1.meta['EPNAMES'], epochs_idx).tolist()
        wd1.meta['list_times'] = np.delete(wd1.meta['list_times'], epochs_idx).tolist()

        # Add in the new column with the combined list.
        wd1['x'] = np.append(wd1['x'], x_wavg[[ii], :].T, axis=1)
        wd1['y'] = np.append(wd1['y'], y_wavg[[ii], :].T, axis=1)
        wd1['m'] = np.append(wd1['m'], m_wavg[[ii], :].T, axis=1)
        if with_m_orig:
            wd1['m_orig'] = np.append(wd1['m_orig'], m_wavg_orig[[ii], :].T, axis=1)
        wd1['xe'] = np.append(wd1['xe'], xe_wavg[[ii], :].T, axis=1)
        wd1['ye'] = np.append(wd1['ye'], ye_wavg[[ii], :].T, axis=1)
        wd1['me'] = np.append(wd1['me'], me_wavg[[ii], :].T, axis=1)
        wd1['n'] = np.append(wd1['n'], nframes[[ii], :].T, axis=1)
        wd1.meta['EPNAMES'] = np.append(wd1.meta['EPNAMES'], mosaic_epochs[ii]).tolist()
        wd1.meta['list_times'] = np.append(wd1.meta['list_times'], years[ii]).tolist()

    return wd1


def add_velocities_to_catalog(catalog_in_dir, catalog_out_dir, epochs=['2005_F814W', '2010_F160W', '2013_F160W', '2015_F160W'], weighting='var', absolute_sigma=False, bootstrap=0, fixed_t0=False, verbose=False, mask_val=-100000.0, show_progress=True):
    """
    Call after make_catalog_from_align(). 
    """
    catalog_in_dir = work_dir + '/50.ALIGN_KS2/' + catalog_in_dir
    catalog_out_dir = work_dir + '/50.ALIGN_KS2/' + catalog_out_dir
    
    startable = StarTable(Table.read(catalog_in_dir))
    startable.meta['HIERARCH LIST_TIMES'] = startable.meta['YEARS']
    epoch_cols = [startable.meta['EPNAMES'].index(_) for _ in epochs]
    mask_lists = [_ for _ in range(len(startable.meta['YEARS'])) if _ not in epoch_cols]
    # startable.fit_velocities(weighting=weighting, absolute_sigma=absolute_sigma, bootstrap=bootstrap, fixed_t0=fixed_t0, verbose=verbose, mask_val=mask_val, mask_lists=mask_lists, show_progress=show_progress)
    startable.fit_velocities_all_detected(
        motion_model_to_fit=Linear(),
        weighting=weighting,
        absolute_sigma=absolute_sigma,
        epoch_cols='all'
    )
    startable.write(catalog_out_dir, overwrite=True)
    return


def identify_kinematic_members(catalog):
    t = copy.deepcopy(catalog)
    scale = ast.scale['WFC'] * 10**3  # mas/yr per pix/yr
    members = np.full(len(t), False, dtype=bool)  # Start with all non-members

    # 
    # Convert to "/yr and mas/yr. Keep the same coordinates as before.
    #

    # Read in the old list and find the first star (by name) in our new list. 
    label_old = fly_starlists.read_label(work_dir + '/50.ALIGN_KS2/wd1_top_stars_label.dat')
    first_star = list(t['name']).index(label_old['name'][0])
    # first_star = np.where(t['name'].data == label_old['name'][0])[0] # error due to difference between byte string and string
    # print(first_star)

    # Find the named stars that match in label_old. We need to preserve these
    # no matter what. (but we can set use=0 if they aren't members).
    in_old = np.zeros(len(t), dtype=bool)
    for oo in range(len(label_old)):
        ndx = np.where(t['name'] == label_old['name'][oo])[0]
        in_old[ndx] = True

    # Convert coords. +x increases to the East/left.
    t['x0'] = (t['x0'] - t['x0'][first_star]) * ast.scale['WFC'] * -1.0
    t['y0'] = (t['y0'] - t['y0'][first_star]) * ast.scale['WFC']

    # Get rid of the big outliers first. We know cluster members should be within.
    # 0.1 pix/yr. Also get rid of anything with large uncertainties. Note, if it is
    # in our old list, we keep it no matter what.
    # idx = np.where( (in_old == True) | 
    #                 (( np.abs(t['vx']) < 0.1 ) &
    #                  ( np.abs(t['vy']) < 0.1 ) &
    #                  ( t['vxe'] < 0.008)  &
    #                  ( t['vye'] < 0.008)) )[0]

    idx1 = np.where( (in_old == True) | 
                    (( np.abs(t['vx']) < 0.1 ) &
                     ( np.abs(t['vy']) < 0.1 )))[0]

    t = t[idx1]
    in_old = in_old[idx1]

    # Determine the mean and std of the velocities. We really want to iter
    vx = t['vx']
    vy = t['vy']

    # Fit a 3 component 2D Gaussian Mixture Model to the velocity distribution
    X = np.vstack((vx, vy)).T
    gmm = GaussianMixture(n_components=3, covariance_type='full', random_state=42)
    gmm.fit(X)
    labels = gmm.predict(X)


    # Select the component with the smallest std in both vx and vy as the cluster members
    stds = np.array([np.sqrt(gmm.covariances_[i].diagonal()) for i in range(gmm.n_components)])
    cluster_label = np.argmin(stds.sum(axis=1))
    members[idx1] = (labels == cluster_label)
    # Extract mean and std of the member component
    vx_mean, vy_mean = gmm.means_[cluster_label]
    member_cov = gmm.covariances_[cluster_label]
    vx_std, vy_std = np.sqrt(member_cov.diagonal())    

    # vx_clip = sigma_clip(t['vx'], sigma=2, maxiters=20)
    # vy_clip = sigma_clip(t['vy'], sigma=2, maxiters=20)

    # vx_mean = vx_clip.mean()
    # vy_mean = vy_clip.mean()
    # vx_std = vx_clip.std()
    # vy_std = vy_clip.std()
    vel_cut = ((vx_std + vy_std) / 2.0) * 3.0  # 3 sigma

    # Convert velocities to have cluster mean at 0, 0
    print('Adjusting cluster to 0 velocity')
    t['vx'] = (t['vx'] - vx_mean) * scale * -1.0
    t['vy'] = (t['vy'] - vy_mean) * scale
    t['vx_err'] *= scale
    t['vy_err'] *= scale
    vel_cut *= scale

    # Plot Gaussian
    circle = plt.Circle((0, 0), vel_cut, color='C3', linewidth=2, fill=False, label='GMM std')
    fig, ax = plt.subplots(figsize=(6, 6))
    scatter = ax.scatter(t['vx'], t['vy'], c=labels, cmap='viridis', s=1, alpha=0.5)
    ax.set_xlabel('vx (mas/yr)')
    ax.set_ylabel('vy (mas/yr)')
    ax.set_title('GMM of VPD')
    ax.legend(*scatter.legend_elements(), title='Component')
    ax.add_artist(circle)
    if not os.path.exists(os.path.join(work_dir, '50.ALIGN_KS2/plots')):
        os.makedirs(os.path.join(work_dir, '50.ALIGN_KS2/plots'))
    plt.tight_layout()
    plt.savefig(os.path.join(work_dir, '50.ALIGN_KS2/plots/make_new_labels_with_members_vpd.png'))
    plt.show()

    # Make the membership cut.
    vtot = np.hypot(t['vx'], t['vy'])
    t = t[members[idx1]]

    # Make a plot to show the distribution of members. This can
    # effect our alignment process.
    plt.figure(2, figsize=(6,6))
    plt.plot(t['x0'], t['y0'], 'k.', ms=1)
    plt.xlabel('X (")')
    plt.ylabel('Y (")')
    plt.savefig(os.path.join(work_dir, '50.ALIGN_KS2/plots/make_new_labels_with_members_xy.png'))
    plt.show()

    print('vx: {0:7.4f} ± {1:7.4f} pix/yr'.format(vx_mean, vx_std))
    print('vy: {0:7.4f} ± {1:7.4f} pix/yr'.format(vy_mean, vy_std))
    print('Velocity Cut: vtot < {0:7.4f} pix/yr'.format(vel_cut))

    print('vx: {0:7.2f} ± {1:7.2f} mas/yr'.format(vx_mean * scale * -1., vx_std * scale))
    print('vy: {0:7.2f} ± {1:7.2f} mas/yr'.format(vy_mean * scale, vy_std * scale))
    print('Velocity Cut: vtot < {0:7.4f} mas/yr'.format(vel_cut * scale))

    print('Saving {0:d} Wd 1 members'.format(len(t)))
    # Make the new label.dat ... we can make it straight from our
    # astropy table, after some modifications.
    # Fetch the 2005_F814W epoch index.
    ee = t.meta['EPNAMES'].index('2005_F814W')
    name = Column(data = t['name'], name = 'name')
    m   = Column(data = t['m'][:,ee], name = 'm')
    x0  = Column(data = t['x0'], name = 'x0')
    y0  = Column(data = t['y0'], name = 'y0')
    x0e = Column(data = t['x0_err'], name = 'x0_err')
    y0e = Column(data = t['y0_err'], name = 'y0_err')
    vx  = Column(data = t['vx'], name = 'vx')
    vy  = Column(data = t['vy'], name = 'vy')
    vxe = Column(data = t['vx_err'], name = 'vx_err')
    vye = Column(data = t['vy_err'], name = 'vy_err')
    t0  = Column(data = t['t0'], name = 't0')
    use = Column(data = np.ones(len(x0), dtype=int)*18, name = 'use')
    r0  = Column(data = np.hypot(x0, y0), name = 'r0')

    label_new = Table([name, m, x0, y0, x0e, y0e, vx, vy, vxe, vye, t0, use, r0])

    # Set the column formats:
    label_new['name'].format = '11s'
    label_new['m'].format = '6.3f'
    label_new['x0'].format = '9.4f'
    label_new['y0'].format = '9.4f'
    label_new['x0_err'].format = '6.4f'
    label_new['y0_err'].format = '6.4f'
    label_new['vx'].format = '7.4f'
    label_new['vy'].format = '7.4f'
    label_new['vx_err'].format = '7.4f'
    label_new['vy_err'].format = '7.4f'
    label_new['t0'].format = '9.4f'
    label_new['use'].format = '3d'
    label_new['r0'].format = '9.4f'

    # Change the stars' names.
    star_idx = len(label_old) + 1
    for ss in range(len(label_new)):
        if not label_new['name'][ss].startswith('wd1'):
            label_new['name'][ss] = 'wd1_{0:05d}'.format(star_idx)
            star_idx += 1
        
    label_new.rename_column('name', '#      name')
    label_new.write(work_dir + '/50.ALIGN_KS2/wd1_members_label.dat', format='ascii.fixed_width',
                        delimiter='  ', delimiter_pad=None, overwrite=True, strip_whitespace=True,
                        bookend=False)
    return members

def make_new_labels_with_members(catalog):
    """
    Produce a new label.dat file for Wd 1 that contains proper motion-selected cluster
    members based on a previous alignment. This will then be used to run align again using
    ONLY cluster members in the alignment.
    """
    t = copy.deepcopy(catalog)
    members = np.full(len(t), False, dtype=bool)  # Start with all non-members

    # 
    # Convert to "/yr and mas/yr. Keep the same coordinates as before.
    #

    # Read in the old list and find the first star (by name) in our new list. 
    label_old = fly_starlists.read_label(work_dir + '/50.ALIGN_KS2/wd1_top_stars_label.dat')
    first_star = list(t['name']).index(label_old['name'][0])
    # first_star = np.where(t['name'].data == label_old['name'][0])[0] # error due to difference between byte string and string
    # print(first_star)

    # Find the named stars that match in label_old. We need to preserve these
    # no matter what. (but we can set use=0 if they aren't members).
    in_old = np.zeros(len(t), dtype=bool)
    for oo in range(len(label_old)):
        ndx = np.where(t['name'] == label_old['name'][oo])[0]
        in_old[ndx] = True

    # Convert coords. +x increases to the East/left.
    t['x0'] = (t['x0'] - t['x0'][first_star]) * ast.scale['WFC'] * -1.0
    t['y0'] = (t['y0'] - t['y0'][first_star]) * ast.scale['WFC']

    # Get rid of the big outliers first. We know cluster members should be within.
    # 0.1 pix/yr. Also get rid of anything with large uncertainties. Note, if it is
    # in our old list, we keep it no matter what.
    # idx = np.where( (in_old == True) | 
    #                 (( np.abs(t['vx']) < 0.1 ) &
    #                  ( np.abs(t['vy']) < 0.1 ) &
    #                  ( t['vxe'] < 0.008)  &
    #                  ( t['vye'] < 0.008)) )[0]

    idx1 = np.where( (in_old == True) | 
                    (( np.abs(t['vx']) < 0.1 ) &
                     ( np.abs(t['vy']) < 0.1 )))[0]

    t = t[idx1]
    in_old = in_old[idx1]

    # Determine the mean and std of the velocities. We really want to iter
    vx_clip = sigma_clip(t['vx'], sigma=2, maxiters=20)
    vy_clip = sigma_clip(t['vy'], sigma=2, maxiters=20)

    vx_mean = vx_clip.mean()
    vy_mean = vy_clip.mean()
    vx_std = vx_clip.std()
    vy_std = vy_clip.std()
    vel_cut = ((vx_std + vy_std) / 2.0) * 3.0  # 3 sigma

    print('vx: {0:7.4f} ± {1:7.4f} pix/yr'.format(vx_mean, vx_std))
    print('vy: {0:7.4f} ± {1:7.4f} pix/yr'.format(vy_mean, vy_std))
    print('Velocity Cut: vtot < {0:7.4f} pix/yr'.format(vel_cut))

    print('vx: {0:7.2f} ± {1:7.2f} mas/yr'.format(vx_mean*ast.scale['WFC']*10**3,
                                                        vx_std*ast.scale['WFC']*10**3))
    print('vy: {0:7.2f} ± {1:7.2f} mas/yr'.format(vy_mean*ast.scale['WFC']*10**3,
                                                        vy_std*ast.scale['WFC']*10**3))
    print('Velocity Cut: vtot < {0:7.4f} mas/yr'.format(vel_cut*ast.scale['WFC']*10**3))
    
    # Convert velocities to have cluster mean at 0, 0
    print('Adjusting cluster to 0 velocity')
    t['vx'] = (t['vx'] - vx_mean) * ast.scale['WFC'] * 10**3 * -1.0
    t['vy'] = (t['vy'] - vy_mean) * ast.scale['WFC'] * 10**3
    t['vx_err'] *= ast.scale['WFC'] * 10**3
    t['vy_err'] *= ast.scale['WFC'] * 10**3
    vel_cut *= ast.scale['WFC'] * 10**3

    if not os.path.exists(os.path.join(work_dir, '50.ALIGN_KS2/plots')):
        os.makedirs(os.path.join(work_dir, '50.ALIGN_KS2/plots'))
    # Make a figure to illustrated the membership cut.
    plt.figure(figsize=(6, 6))
    plt.plot(t['vx'], t['vy'], 'k.', ms=2, alpha=0.5)
    circ = plt.Circle([0, 0], radius=vel_cut, color='red', fill=False)
    plt.gca().add_artist(circ)
    lim = 2.5
    plt.axis([-lim, lim, -lim, lim])
    plt.savefig(os.path.join(work_dir, '50.ALIGN_KS2/plots/make_new_labels_with_members_vpd.png'))
    plt.show()

    # Make the membership cut.
    vtot = np.hypot(t['vx'], t['vy'])
    idx2 = np.where((in_old == True) | (vtot < vel_cut))
    t = t[idx2]
    members[idx1[idx2]] = True
    print('Saving {0:d} Wd 1 members'.format(len(idx2[0])))

    # Make a plot to show the distribution of members. This can
    # effect our alignment process.
    plt.figure(2, figsize=(6,6))
    plt.plot(t['x0'], t['y0'], 'k.', ms=4)
    plt.xlabel('X (")')
    plt.ylabel('Y (")')
    plt.savefig(os.path.join(work_dir, '50.ALIGN_KS2/plots/make_new_labels_with_members_xy.png'))
    plt.show()

    # Make the new label.dat ... we can make it straight from our
    # astropy table, after some modifications.
    # Fetch the 2005_F814W epoch index.
    ee = t.meta['EPNAMES'].index('2005_F814W')
    name = Column(data = t['name'], name = 'name')
    m   = Column(data = t['m'][:,ee], name = 'm')
    x0  = Column(data = t['x0'], name = 'x0')
    y0  = Column(data = t['y0'], name = 'y0')
    x0e = Column(data = t['x0_err'], name = 'x0_err')
    y0e = Column(data = t['y0_err'], name = 'y0_err')
    vx  = Column(data = t['vx'], name = 'vx')
    vy  = Column(data = t['vy'], name = 'vy')
    vxe = Column(data = t['vx_err'], name = 'vx_err')
    vye = Column(data = t['vy_err'], name = 'vy_err')
    t0  = Column(data = t['t0'], name = 't0')
    use = Column(data = np.ones(len(x0), dtype=int)*18, name = 'use')
    r0  = Column(data = np.hypot(x0, y0), name = 'r0')

    label_new = Table([name, m, x0, y0, x0e, y0e, vx, vy, vxe, vye, t0, use, r0])

    # Set the column formats:
    label_new['name'].format = '11s'
    label_new['m'].format = '6.3f'
    label_new['x0'].format = '9.4f'
    label_new['y0'].format = '9.4f'
    label_new['x0_err'].format = '6.4f'
    label_new['y0_err'].format = '6.4f'
    label_new['vx'].format = '7.4f'
    label_new['vy'].format = '7.4f'
    label_new['vx_err'].format = '7.4f'
    label_new['vy_err'].format = '7.4f'
    label_new['t0'].format = '9.4f'
    label_new['use'].format = '3d'
    label_new['r0'].format = '9.4f'

    # Change the stars' names.
    star_idx = len(label_old) + 1
    for ss in range(len(label_new)):
        if not label_new['name'][ss].startswith('wd1'):
            label_new['name'][ss] = 'wd1_{0:05d}'.format(star_idx)
            star_idx += 1
        
    label_new.rename_column('name', '#      name')
    label_new.write(work_dir + '/50.ALIGN_KS2/wd1_members_label.dat', format='ascii.fixed_width',
                        delimiter='  ', delimiter_pad=None, overwrite=True, strip_whitespace=True,
                        bookend=False)
    
    return members





##################################################
# OLD STUFF
##################################################
                    
def make_catalog(use_RMSE=True, vel_weight=None):
    """
    Read the align FITS table from all three epochs and fit a velocity to the
    positions.

    use_RMSE - If True, use RMS error (standard deviation) for the positional error.
               If False, convert the positional errors in "error on the mean" and
               save to a different catalog name.
    vel_weight - None, 'error', 'variance'
            None = no weighting by errors in the velocity fit.
            'error' = Weight by 1.0 / positional error in the velocity fit.
            'variance' = Weight by 1.0 / (positional error)^2 in the velocity fit. 
    """
    
    final = None
    good = None

    d_all = Table.read(work_dir + '/50.ALIGN_KS2/align_a4_t_combo_pos.fits')

    # TRIM out all stars that aren't detected in all 3 epochs:
    #    2005_814
    #    2010_160
    #    2013_160
    #    2015_160
    idx = np.where((d_all['x_2005_F814W'] > -999) &
                   (d_all['x_2010_F160W'] > -999) &
                   (d_all['x_2013_F160W'] > -999) &
                   (d_all['x_2015_F160W'] > -999) &
                   (d_all['n_2005_F814W'] > 1) &
                   (d_all['n_2010_F160W'] > 1) &
                   (d_all['n_2013_F160W'] > 1) &
                   (d_all['n_2015_F160W'] > 1) &
                   (d_all['xe_2005_F814W'] > 0) &
                   (d_all['xe_2010_F160W'] > 0) &
                   (d_all['xe_2013_F160W'] > 0) &
                   (d_all['xe_2015_F160W'] > 0) &
                   (d_all['ye_2005_F814W'] > 0) &
                   (d_all['ye_2010_F160W'] > 0) &
                   (d_all['ye_2013_F160W'] > 0) &
                   (d_all['ye_2015_F160W'] > 0))[0]

    tmp_2005 = np.where((d_all['x_2005_F814W'] > -999) &
                        (d_all['n_2005_F814W'] > 1) &
                        (d_all['xe_2005_F814W'] > 0) &
                        (d_all['ye_2005_F814W'] > 0))[0]
    tmp_2010 = np.where((d_all['x_2010_F160W'] > -999) &
                        (d_all['n_2010_F160W'] > 1) &
                        (d_all['xe_2010_F160W'] > 0) &
                        (d_all['ye_2010_F160W'] > 0))[0]
    tmp_2013 = np.where((d_all['x_2013_F160W'] > -999) &
                        (d_all['n_2013_F160W'] > 1) &
                        (d_all['xe_2013_F160W'] > 0) &
                        (d_all['ye_2013_F160W'] > 0))[0]
    tmp_2015 = np.where((d_all['x_2015_F160W'] > -999) &
                        (d_all['n_2015_F160W'] > 1) &
                        (d_all['xe_2015_F160W'] > 0) &
                        (d_all['ye_2015_F160W'] > 0))[0]
    print('Found {0:3} in 2005 F814W'.format(len(tmp_2005)))
    print('Found {0:3} in 2010 F160W'.format(len(tmp_2010)))
    print('Found {0:3} in 2013 F160W'.format(len(tmp_2013)))
    print('Found {0:3} in 2015 F160W'.format(len(tmp_2015)))
    print('')
    print('Kept {0:d} of {1:d} stars in all 4 epochs.'.format(len(idx), len(d_all)))
    
    d = d_all[idx]
    
    #Changing rms errors into standard errors for the f153m data
    xeom_2005_814 = d['xe_2005_F814W'] / np.sqrt(d['n_2005_F814W'])
    yeom_2005_814 = d['ye_2005_F814W'] / np.sqrt(d['n_2005_F814W'])
    xeom_2010_160 = d['xe_2010_F160W'] / np.sqrt(d['n_2010_F160W'])
    yeom_2010_160 = d['ye_2010_F160W'] / np.sqrt(d['n_2010_F160W'])
    xeom_2013_160 = d['xe_2013_F160W'] / np.sqrt(d['n_2013_F160W'])
    yeom_2013_160 = d['ye_2013_F160W'] / np.sqrt(d['n_2013_F160W'])
    xeom_2015_160 = d['xe_2015_F160W'] / np.sqrt(d['n_2015_F160W'])
    yeom_2015_160 = d['ye_2015_F160W'] / np.sqrt(d['n_2015_F160W'])
    
    # Fit velocities. Will use an error-weighted t0, specified to each object
    t = np.array([years['2005_F814W'], years['2010_F160W'], years['2013_F160W'], years['2015_F160W']])

    if use_RMSE:
        # Shape = (nepochs, nstars)
        xerr = np.array([d['xe_2005_F814W'], d['xe_2010_F160W'], d['xe_2013_F160W'], d['xe_2015_F160W']])
        yerr = np.array([d['ye_2005_F814W'], d['ye_2010_F160W'], d['ye_2013_F160W'], d['ye_2015_F160W']])
    else:
        xerr = np.array([xeom_2005_814, xeom_2010_160, xeom_2013_160, xeom_2015_160])
        yerr = np.array([yeom_2005_814, yeom_2010_160, yeom_2013_160, yeom_2015_160])

    w = 1.0 / (xerr**2 + yerr**2)
    w = np.transpose(w) #Getting the dimensions of w right
    numerator = np.sum(t * w, axis = 1)
    denominator = np.sum(w, axis = 1)
    t0_arr = numerator / denominator

    nstars = len(d)
    nepochs = len(t)
    
    # 2D arrays Shape = (nepochs, nstars)
    t = np.tile(t, (nstars, 1)).T
    t0 = np.tile(t0_arr, (nepochs, 1))

    #Calculating dt for each object
    dt = t - t0

    d.add_column(Column(data=t[0],name='t_2005_F814W'))
    d.add_column(Column(data=t[1],name='t_2010_F160W'))
    d.add_column(Column(data=t[2],name='t_2013_F160W'))
    d.add_column(Column(data=t[2],name='t_2015_F160W'))
    d.add_column(Column(data=t0[0],name='fit_t0'))

    d.add_column(Column(data=np.ones(nstars),name='x0'))
    d.add_column(Column(data=np.ones(nstars),name='vx'))
    d.add_column(Column(data=np.ones(nstars),name='y0'))
    d.add_column(Column(data=np.ones(nstars),name='vy'))

    d.add_column(Column(data=np.ones(nstars),name='x0e'))
    d.add_column(Column(data=np.ones(nstars),name='vxe'))
    d.add_column(Column(data=np.ones(nstars),name='y0e'))
    d.add_column(Column(data=np.ones(nstars),name='vye'))

    for ii in range(len(d)):
        x = np.array([d['x_2005_F814W'][ii], d['x_2010_F160W'][ii], d['x_2013_F160W'][ii], d['x_2015_F160W'][ii]])
        y = np.array([d['y_2005_F814W'][ii], d['y_2010_F160W'][ii], d['y_2013_F160W'][ii], d['y_2015_F160W'][ii]])
        xe = xerr[:, ii]
        ye = yerr[:, ii]

        if (vel_weight != 'error') and (vel_weight != 'variance'):
            vxOpt, vxCov = np.polyfit(dt[:, ii], x, 1, cov=True)
            vyOpt, vyCov = np.polyfit(dt[:, ii], y, 1, cov=True)
        if vel_weight == 'error':
            vxOpt, vxCov = np.polyfit(dt[:, ii], x, 1, w=1/xe, cov=True)
            vyOpt, vyCov = np.polyfit(dt[:, ii], y, 1, w=1/ye, cov=True)
        if vel_weight == 'variance':
            vxOpt, vxCov = np.polyfit(dt[:, ii], x, 1, w=1/xe**2, cov=True)
            vyOpt, vyCov = np.polyfit(dt[:, ii], y, 1, w=1/ye**2, cov=True)
            
        vxErr = np.sqrt(-1.0 * vxCov.diagonal())
        vyErr = np.sqrt(-1.0 * vyCov.diagonal())

        d['x0'][ii] = vxOpt[1]
        d['vx'][ii] = vxOpt[0]
        d['x0e'][ii] = vxErr[1]
        d['vxe'][ii] = vxErr[0]

        d['y0'][ii] = vyOpt[1]
        d['vy'][ii] = vyOpt[0]
        d['y0e'][ii] = vyErr[1]
        d['vye'][ii] = vyErr[0]


    # Fix the F814W magnitudes (adjust for integration time)
    fix_magnitudes(d)
        
    catalog_name = 'wd1_catalog'
    if use_RMSE:
        catalog_name += '_RMSE'
    else:
        catalog_name += '_EOM'

    if vel_weight == None:
        catalog_name += '_wvelNone'
    else:
        if vel_weight == 'error':
            catalog_name += '_wvelErr'
        if vel_weight == 'variance':
            catalog_name += '_wvelVar'
    catalog_name += '.fits'

    d.write(work_dir + '/50.ALIGN_KS2/' + catalog_name, format='fits', overwrite=True)
    
    return

def art_set_detected(use_obs_align=False):
    """
    Create the "detected" columsn in the artificial star list.
    
    Apply a set of criteria to call a star detected based on on how
    closely the input and output position/flux match.
    """
    in_file = work_dir + '/51.ALIGN_ART/art_align'
    if use_obs_align:
        in_file += '_obs'
    else:
        in_file += '_art'
    in_file += '_combo_pos.fits'
    
    d = Table.read(in_file)

    # Make a "detected" column for each epoch/filter.
    # Set detected = True for all sources that are detected and whose
    # positions/fluxes match within certain criteria. These are loose criteria.
    epochs = ['2005_F814W', '2010_F125W', '2010_F139M', '2010_F160W', '2013_F160W']
    dr_cuts = [1.0, 0.5, 0.5, 0.5, 0.5] # pixels
    dm_cuts = [0.5, 0.5, 0.5, 0.5, 0.5] # magnitudes
    n_cut   = 1

    for ee in range(len(epochs)):
        epoch = epochs[ee]
        det = np.zeros(len(d), dtype=bool)

        n = d['n_' + epoch]
        dx = d['x_' + epoch] - d['xin_' + epoch]
        dy = d['y_' + epoch] - d['yin_' + epoch]
        dm = d['m_' + epoch] - d['min_' + epoch]
        dr = np.hypot(dx, dy)

        idx = np.where((n > n_cut) & (dr < dr_cuts[ee]) & (dm < dm_cuts[ee]))
        det[idx] = True

        # Add this column to the table.
        d.add_column(Column(det, name='det_' + epoch))

    catalog_name = work_dir + '/51.ALIGN_ART/art_align' 
    if use_obs_align:
        catalog_name += '_obs'
    else:
        catalog_name += '_art'
    catalog_name += '_combo_pos_det.fits'
    d.write(catalog_name, format='fits', overwrite=True)

    return
    
def fix_artstar_errors(use_obs_align=False):
    """
    Little snippet of code to compare the artificial star errors with the
    observed star errors to determine if a constant offset exists between
    the two. Reports the value of that offset.

    Specialized to work with Wd1 data (filter sets, etc)

    art_obs mags assumed to be instrumental
    """
    real_obs = work_dir + '/50.ALIGN_KS2/align_a4_t_combo_pos.fits'
    art_obs = work_dir + '/51.ALIGN_ART/art_align'
    if use_obs_align:
        art_obs += '_obs'
    else:
        art_obs += '_art'
    art_obs += '_combo_pos_det.fits'
    binsize = 0.5
    

    # Read the catalogs, create new artificial star lists for
    # each filter/epoch only containing the recovered stars
    print('Reading input')
    obs = Table.read(real_obs, format='fits')
    art = Table.read(art_obs, format='fits')
    print('Done')

    # Fix F814W mag in observed data temporariliy
    fix_magnitudes(obs)
    
    epochs = ['2005_F814W', '2010_F125W', '2010_F139M', '2010_F160W', '2013_F160W']

    def mean_clip(vals):
        val_mean, val_std, n = statsIter.mean_std_clip(vals,
                                                       clipsig=3.0,
                                                       maxiter=10,
                                                       converge_num=0.01,
                                                       verbose=False,
                                                       return_nclip=True)
        return val_mean

    def plot_pos_errors(epoch, obs_mag, obs_err, art_mag, art_err, mag_cent,
                        obs_pos_err_mean, art_pos_err_mean, suffix=''):
        
        plt.figure(1, figsize = (10,10))
        plt.clf()
        plt.semilogy(art_mag, art_err, 'k.', ms=4, label='Artificial', alpha=0.2)
        plt.semilogy(obs_mag, obs_err, 'r.', ms=4, label='Observed', alpha=0.5)
        plt.plot(mag_cent, obs_pos_err_mean, 'b-', linewidth=2,
                label='Observed Median')
        plt.plot(mag_cent, art_pos_err_mean, 'g-', linewidth=2,
                label='Artificial Median')
        plt.xlabel('Observed Mag')
        plt.ylabel('Positional Error (pix)')
        plt.title('Positional Errors,' + epoch)
        plt.legend(loc=2, numpoints=1)
        plt.xlim(13, max(art_mag))
        plt.ylim(1e-4, 1)
        plot_file = work_dir + '/plots/Pos_errcomp' + suffix + '_' + epoch
        if use_obs_align:
            plot_file += '_obs'
        else:
            plot_file += '_art'
        plot_file += '.png'
        plt.savefig(plot_file)

        return

    def plot_mag_errors(epoch, obs_mag, obs_err, art_mag, art_err, mag_cent,
                        obs_mag_err_mean, art_mag_err_mean, suffix=''):
        
        plt.figure(2, figsize=(10,10))
        plt.clf()
        plt.semilogy(art_mag, art_merr, 'k.', ms=4, label='Artificial', alpha=0.2)
        plt.semilogy(obs_mag, obs_merr, 'r.', ms=4, label='Observed', alpha=0.5)
        plt.plot(mag_cent, obs_mag_err_mean, 'b-', linewidth=2,
                label='Observed Median')
        plt.plot(mag_cent, art_mag_err_mean, 'g-', linewidth=2,
                label='Artificial Median')
        plt.xlabel('Observed Mag')
        plt.ylabel('Photometric Error (mag)')
        plt.title('Photometric Errors,' + epoch)
        plt.legend(loc=2, numpoints=1)
        plt.xlim(13, max(art_mag))
        plt.ylim(1e-4, 1)
        plot_file = work_dir + '/plots/Mag_errcomp' + suffix + '_' + epoch
        if use_obs_align:
            plot_file += '_obs'
        else:
            plot_file += '_art'
        plot_file += '.png'
        plt.savefig(plot_file)

        return
        
    for epoch in epochs:
        # Convert art star mags from instrumental to apparent
        filt = epoch.split('_')[-1]
        art['m_' + epoch] += photometry.ZP[filt]
        art['min_' + epoch] += photometry.ZP[filt]

        det = np.where(art['det_' + epoch] == True)
                
        # Extract observed/artificial errors for each filter.
        # X and Y errors are added in quadrature.
        obs_err = np.hypot(obs['xe_' + epoch], obs['ye_' + epoch])
        obs_mag = obs['m_' + epoch]
        obs_merr = obs['me_' + epoch]

        art_err = np.hypot(art['xe_' + epoch], art['ye_' + epoch])
        art_mag = art['m_' + epoch]
        art_merr = art['me_' + epoch]

        art_err = art_err[det]
        art_mag = art_mag[det]
        art_merr = art_merr[det]

        # For each epoch/filter, calculate the median position and
        # magnitude error in each magbin for both observed and artificial.
        mag_bins = np.arange(min(art_mag), max(art_mag) + binsize, binsize)
        mag_cent = mag_bins[:-1] + (np.diff(mag_bins) / 2.0)

        obs_pos_err_mean, f1, f2 = binned_statistic(obs_mag, obs_err,
                                                    bins=mag_bins,
                                                    statistic=mean_clip)
        obs_mag_err_mean, f1, f2 = binned_statistic(obs_mag, obs_merr,
                                                    bins=mag_bins,
                                                    statistic=mean_clip)
        art_pos_err_mean, f1, f2 = binned_statistic(art_mag, art_err,
                                                    bins=mag_bins,
                                                    statistic=mean_clip)
        art_mag_err_mean, f1, f2 = binned_statistic(art_mag, art_merr,
                                                    bins=mag_bins,
                                                    statistic=mean_clip)

        plot_pos_errors(epoch, obs_mag, obs_err, art_mag, art_err, mag_cent,
                        obs_pos_err_mean, art_pos_err_mean, suffix='_orig')
        plot_mag_errors(epoch, obs_mag, obs_err, art_mag, art_err, mag_cent,
                        obs_mag_err_mean, art_mag_err_mean, suffix='_orig')

        
        # Now, calculate error floor of observations, add in quadrature to artificial errors
        # For WFC3IR, we are in the error floor for everything brighter than 16th mag
        # For ACS, we are in error floor for evertything brighter than 14th mag

        if epoch == '2005_F814W':
            brite_obs = np.where((obs_mag > 10) & (obs_mag < 21))[0]
            brite_art = np.where((art_mag > 10) & (art_mag < 21))[0]
        else:
            brite_obs = np.where((obs_mag > 0) & (obs_mag < 16))[0]
            brite_art = np.where((art_mag > 0) & (art_mag < 16))[0]
            
        # Take the median error for all the bright bins.
        perr_obs = np.median(obs_err[brite_obs])
        merr_obs = np.median(obs_merr[brite_obs])
        perr_art = np.median(art_err[brite_art])
        merr_art = np.median(art_merr[brite_art])

        # Calculate the final "floor" errors that will get
        # added in quadrature.
        pos_err = 0.0
        mag_err = 0.0
        pos_err_1d = 0.0
        if perr_obs > perr_art:
            pos_err = np.sqrt(perr_obs**2 - perr_art**2)
            pos_err_1d = pos_err / np.sqrt(2.0)
        if merr_obs > merr_art:
            mag_err = np.sqrt(merr_obs**2 - merr_art**2)

        print('**********************************')
        fmt = 'The {0:s} error floor of {1:s} is {2:5.4f} {3:s} vs. obs of {4:5.4f} {3:s}'
        print(fmt.format('1D positional', epoch, pos_err_1d, '(pix)', perr_obs))
        print(fmt.format('  photometric', epoch, mag_err, '(mag)', merr_obs))
        print('**********************************')

        # Adding error floors in quadrature to artificial errors
        art_err = np.hypot(art_err, pos_err)
        art_merr = np.hypot(art_merr, mag_err)
        
        # Recalculate median pos and mag errors for the artificial stars
        art_pos_err_mean, f1, f2 = binned_statistic(art_mag, art_err,
                                                    bins=mag_bins,
                                                    statistic=mean_clip)
        art_mag_err_mean, f1, f2 = binned_statistic(art_mag, art_merr,
                                                    bins=mag_bins,
                                                    statistic=mean_clip)
        
        plot_pos_errors(epoch, obs_mag, obs_err, art_mag, art_err, mag_cent,
                        obs_pos_err_mean, art_pos_err_mean, suffix='_corr')
        plot_mag_errors(epoch, obs_mag, obs_err, art_mag, art_err, mag_cent,
                        obs_mag_err_mean, art_mag_err_mean, suffix='_corr')

        # Update the original artificial star table with the error floors,
        # write new table. Only update errors for stars detected in that
        # particular filter.
        art['me_' + epoch][det] = art_merr

        # Will split pos error floor evenly among X,Y
        art['xe_' + epoch][det] = np.hypot(art['xe_' + epoch][det], pos_err_1d)
        art['ye_' + epoch][det] = np.hypot(art['ye_' + epoch][det], pos_err_1d)

        # Find any detected stars with zero errors. Set them to 0.1 * obs floor.
        det_zero_merr = np.where(art['me_' + epoch][det] == 0)[0]
        det_zero_xerr = np.where(art['xe_' + epoch][det] == 0)[0]
        det_zero_yerr = np.where(art['ye_' + epoch][det] == 0)[0]
        art['me_' + epoch][det][det_zero_merr] = 0.1 * merr_obs
        art['xe_' + epoch][det][det_zero_xerr] = 0.1 * perr_obs
        art['ye_' + epoch][det][det_zero_yerr] = 0.1 * perr_obs

        
    out_file = work_dir + '/51.ALIGN_ART/art_align'
    if use_obs_align:
        out_file += '_obs'
    else:
        out_file += '_art'
    out_file += '_combo_pos_det_newerr.fits'
    art.write(out_file, format='fits', overwrite=True)

    return


def make_artificial_catalog(use_RMSE=True, vel_weight=None, use_obs_align=False):
    """
    Read the align FITS table from the artificial star catalog and fit a
    velocity to the positions.

    use_RMSE - If True, use RMS error (standard deviation) for the positional error.
               If False, convert the positional errors in "error on the mean" and
               save to a different catalog name.
    vel_weight - None, 'error', 'variance'
            None = no weighting by errors in the velocity fit.
            'error' = Weight by 1.0 / positional error in the velocity fit.
            'variance' = Weight by 1.0 / (positional error)^2 in the velocity fit. 
    """
    
    final = None
    good = None

    in_file = work_dir + '/51.ALIGN_ART/art_align'
    if use_obs_align:
        in_file += '_obs'
    else:
        in_file += '_art'
    in_file += '_combo_pos_det_newerr.fits'
    
    d = Table.read(in_file)
    
    # Fetch all stars that are in all 3 epochs.
    #    2005_814
    #    2010_160
    #    2013_160
    idx = np.where((d['det_2005_F814W'] == True) &
                   (d['det_2010_F160W'] == True) &
                   (d['det_2013_F160W'] == True))[0]
    
    print('Found {0:d} of {1:d} stars in all 3 epochs.'.format(len(idx), len(d)))
    
    # Changing rms errors into standard errors for the f153m data
    xeom_2005_814 = d['xe_2005_F814W'][idx] / np.sqrt(d['n_2005_F814W'][idx])
    yeom_2005_814 = d['ye_2005_F814W'][idx] / np.sqrt(d['n_2005_F814W'][idx])
    xeom_2010_160 = d['xe_2010_F160W'][idx] / np.sqrt(d['n_2010_F160W'][idx])
    yeom_2010_160 = d['ye_2010_F160W'][idx] / np.sqrt(d['n_2010_F160W'][idx])
    xeom_2013_160 = d['xe_2013_F160W'][idx] / np.sqrt(d['n_2013_F160W'][idx])
    yeom_2013_160 = d['ye_2013_F160W'][idx] / np.sqrt(d['n_2013_F160W'][idx])
    
    # Fit velocities. Will use an error-weighted t0, specified to each object
    t = np.array([years['2005_F814W'], years['2010_F160W'], years['2013_F160W']])

    if use_RMSE:
        # Shape = (nepochs, nstars)
        xerr = np.array([d['xe_2005_F814W'][idx],
                         d['xe_2010_F160W'][idx],
                         d['xe_2013_F160W'][idx]])
        yerr = np.array([d['ye_2005_F814W'][idx],
                         d['ye_2010_F160W'][idx],
                         d['ye_2013_F160W'][idx]])
    else:
        xerr = np.array([xeom_2005_814**2, xeom_2010_160**2, xeom_2013_160**2])
        yerr = np.array([yeom_2005_814**2, yeom_2010_160**2, yeom_2013_160**2])

    w = 1.0 / (xerr**2 + yerr**2)
    w = np.transpose(w) #Getting the dimensions of w right
    numerator = np.sum(t * w, axis = 1)
    denominator = np.sum(w, axis = 1)

    nstars = len(d)
    nepochs = len(t)
    
    t0_arr = np.zeros(nstars, dtype=float)
    t0_arr[idx] = numerator / denominator
    
    # 2D arrays Shape = (nepochs, nstars)
    t = np.tile(t, (nstars, 1)).T
    t0 = np.tile(t0_arr, (nepochs, 1))

    #Calculating dt for each object
    dt = t - t0
    
    d.add_column(Column(data=t[0], name='t_2005_F814W'))
    d.add_column(Column(data=t[1], name='t_2010_F160W'))
    d.add_column(Column(data=t[2], name='t_2013_F160W'))
    d.add_column(Column(data=t0[0], name='fit_t0'))

    d.add_column(Column(data=np.ones(nstars), name='x0'))
    d.add_column(Column(data=np.ones(nstars), name='vx'))
    d.add_column(Column(data=np.ones(nstars), name='y0'))
    d.add_column(Column(data=np.ones(nstars), name='vy'))

    d.add_column(Column(data=np.ones(nstars), name='x0e'))
    d.add_column(Column(data=np.ones(nstars), name='vxe'))
    d.add_column(Column(data=np.ones(nstars), name='y0e'))
    d.add_column(Column(data=np.ones(nstars), name='vye'))

    for i_idx in range(len(idx)):
        # Note i_idx is the index into the "idx" array.
        # Note ii    is the index into the original table arrays.
        ii = idx[i_idx]

        if (ii % 1e4) == 0:
            print('Working on ', i_idx, ii)
        x = np.array([d['x_2005_F814W'][ii], d['x_2010_F160W'][ii], d['x_2013_F160W'][ii]])
        y = np.array([d['y_2005_F814W'][ii], d['y_2010_F160W'][ii], d['y_2013_F160W'][ii]])
        xe = xerr[:, i_idx]
        ye = yerr[:, i_idx]

        if (vel_weight != 'error') and (vel_weight != 'variance'):
            vxOpt, vxCov = np.polyfit(dt[:, ii], x, 1, cov=True)
            vyOpt, vyCov = np.polyfit(dt[:, ii], y, 1, cov=True)
        if vel_weight == 'error':
            vxOpt, vxCov = np.polyfit(dt[:, ii], x, 1, w=1/xe, cov=True)
            vyOpt, vyCov = np.polyfit(dt[:, ii], y, 1, w=1/ye, cov=True)
        if vel_weight == 'variance':
            vxOpt, vxCov = np.polyfit(dt[:, ii], x, 1, w=1/xe**2, cov=True)
            vyOpt, vyCov = np.polyfit(dt[:, ii], y, 1, w=1/ye**2, cov=True)

        # if (ii == 35391):
        #     pdb.set_trace()
        
        vxErr = np.sqrt(-1.0 * vxCov.diagonal())
        vyErr = np.sqrt(-1.0 * vyCov.diagonal())

        d['x0'][ii] = vxOpt[1]
        d['vx'][ii] = vxOpt[0]
        d['x0e'][ii] = vxErr[1]
        d['vxe'][ii] = vxErr[0]

        d['y0'][ii] = vyOpt[1]
        d['vy'][ii] = vyOpt[0]
        d['y0e'][ii] = vyErr[1]
        d['vye'][ii] = vyErr[0]

    catalog_name = work_dir + '/51.ALIGN_ART/wd1_art_catalog'
    if use_RMSE:
        catalog_name += '_RMSE'
    else:
        catalog_name += '_EOM'

    if vel_weight == None:
        catalog_name += '_wvelNone'
    else:
        if vel_weight == 'error':
            catalog_name += '_wvelErr'
        if vel_weight == 'variance':
            catalog_name += '_wvelVar'
            
    if use_obs_align:
        catalog_name += '_aln_obs'
    else:
        catalog_name += '_aln_art'
        
    catalog_name += '.fits'

    d.write(catalog_name, format='fits', overwrite=True)
    
    return


def plot_vpd(use_RMSE=False, vel_weight=None):
    """
    Check the VPD and quiver plots for our KS2-extracted, re-transformed astrometry.
    """
    catalog_name = 'wd1_catalog'
    catalog_suffix = ''
    if use_RMSE:
        catalog_suffix += '_RMSE'
    else:
        catalog_suffix += '_EOM'

    if vel_weight == None:
        catalog_suffix += '_wvelNone'
    else:
        if vel_weight == 'error':
            catalog_suffix += '_wvelErr'
        if vel_weight == 'variance':
            catalog_suffix += '_wvelVar'
    catalog_name += catalog_suffix + '.fits'
    
    catFile = work_dir + '/50.ALIGN_KS2/' + catalog_name
    tab = Table.read(catFile)

    good = (tab['vxe'] < 0.01) & (tab['vye'] < 0.01) & \
        (tab['me_2005_F814W'] < 0.1) & (tab['me_2010_F160W'] < 0.1)

    tab2 = tab[good]

    vx = tab2['vx'] * ast.scale['WFC'] * 1e3
    vy = tab2['vy'] * ast.scale['WFC'] * 1e3

    plt.figure(1)
    plt.clf()
    q = plt.quiver(tab2['x_2005_F814W'], tab2['y_2005_F814W'], vx, vy, scale=1e2)
    plt.quiverkey(q, 0.95, 0.85, 5, '5 mas/yr', color='red', labelcolor='red')
    plt.savefig(work_dir + '/50.ALIGN_KS2/plots/vec_diffs' + catalog_suffix + '.png')

    plt.close(3)
    plt.figure(3, figsize=(8,6))
    plt.clf()
    nz = mcolors.Normalize()
    nz.autoscale(tab2['m_2005_F814W'])
    q = plt.quiver(tab2['x_2005_F814W'], tab2['y_2005_F814W'], vx, vy, scale=1e2,
                  color=plt.cm.gist_stern(nz(tab2['m_2005_F814W'])))
    plt.quiverkey(q, 0.95, 0.85, 5, '5 mas/yr', color='black', labelcolor='black')
    plt.axis('equal')
    cax, foo = colorbar.make_axes(plt.gca(), orientation='vertical', fraction=0.2, pad=0.04)
    cb = colorbar.ColorbarBase(cax, cmap=plt.cm.gist_stern, norm=nz,
                               orientation='vertical')
    cb.set_label('F814W')
    plt.savefig(work_dir + '/50.ALIGN_KS2/plots/vec_diffs_color' + catalog_suffix + '.png')

        
    plt.figure(2)
    plt.clf()
    plt.plot(vx, vy, 'k.', ms=2)
    lim = 5
    plt.axis([-lim, lim, -lim, lim])
    plt.xlabel('X Proper Motion (mas/yr)')
    plt.ylabel('Y Proper Motion (mas/yr)')
    plt.savefig(work_dir + '/50.ALIGN_KS2/plots/vpd' + catalog_suffix + '.png')

    plt.figure(3)
    plt.clf()
    nz = mcolors.Normalize()
    nz.autoscale(tab2['m_2005_F814W'])
    plt.scatter(vx, vy, c=nz(tab2['m_2005_F814W']), s=5, edgecolor='',
               cmap=plt.cm.gist_stern)
    plt.xlabel('X Proper Motion (mas/yr)')
    plt.ylabel('Y Proper Motion (mas/yr)')
    plt.axis('equal')
    lim = 3.5
    plt.axis([-lim, lim, -lim, lim])
    
    cax, foo = colorbar.make_axes(plt.gca(), orientation='vertical', fraction=0.2, pad=0.04)
    cb = colorbar.ColorbarBase(cax, cmap=plt.cm.gist_stern, norm=nz,
                               orientation='vertical')
    cb.set_label('F814W')
    plt.savefig(work_dir + '/50.ALIGN_KS2/plots/vpd_color' + catalog_suffix + '.png')
    

    idx = np.where((np.abs(vx) < 3) & (np.abs(vy) < 3))[0]
    print('Cluster Members (within vx < 10 mas/yr and vy < 10 mas/yr)')
    print(('   vx = {vx:6.2f} ± {vxe:6.2f} mas/yr'.format(vx=vx[idx].mean(),
                                                           vxe=vx[idx].std())))
    print(('   vy = {vy:6.2f} ± {vye:6.2f} mas/yr'.format(vy=vy[idx].mean(),
                                                           vye=vy[idx].std())))
    
    return

# def setup_artstar_info():
    # comp_dirs = {
    #     '21.KS2_2005_ART': ['ks2_25']
    # }
    # art_lists = {
    #     '21.KS2_2005_ART': ['Artstarlist_2005_25.txt']
    # }
    # filters = {
    #     '21.KS2_2005_ART': [1]
    # }
    # comp_dirs = {'21.KS2_2005_ART': ['01.ks2', '02.ks2', '03.ks2',
    #                                  '04.ks2', '05.ks2', '06.ks2', '07.ks2',
    #                                  '08.ks2', '09.ks2', '10.ks2',
    #                                  '11.ks2', '12.ks2', '13.ks2', '14.ks2'],
    #              '22.KS2_2010_ART': ['01.pos1_a', '01.pos1_b', '01.pos1_c',
    #                                  '02.pos2_a', '02.pos2_b', '02.pos2_c',
    #                                  '03.pos3_a', '03.pos3_b', '03.pos3_c',
    #                                  '04.pos4_a', '04.pos4_b', '04.pos4_c',
    #                                  '01.pos1_d', '01.pos1_e', '01.pos1_f',                                    
    #                                  '02.pos2_d', '02.pos2_e', '02.pos2_f',                                    
    #                                  '03.pos3_d', '03.pos3_e', '03.pos3_f',                                    
    #                                  '04.pos4_d', '04.pos4_e', '04.pos4_f'],
    #              '23.KS2_2013_ART': ['01.pos1_a', '01.pos1_b', '01.pos1_c',
    #                                  '02.pos2_a', '02.pos2_b', '02.pos2_c',
    #                                  '03.pos3_a', '03.pos3_b', '03.pos3_c',
    #                                  '04.pos4_a', '04.pos4_b', '04.pos4_c',
    #                                  '01.pos1_d', '01.pos1_e', '01.pos1_f',
    #                                  '02.pos2_d', '02.pos2_e', '02.pos2_f',
    #                                  '03.pos3_d', '03.pos3_e', '03.pos3_f',
    #                                  '04.pos4_d', '04.pos4_e', '04.pos4_f'],
    #              '24.KS2_2015_ART': ['01.pos1_a', '01.pos1_b', '01.pos1_c',
    #                                  '02.pos2_a', '02.pos2_b', '02.pos2_c',
    #                                  '03.pos3_a', '03.pos3_b', '03.pos3_c',
    #                                  '04.pos4_a', '04.pos4_b', '04.pos4_c',
    #                                  '01.pos1_d', '01.pos1_e', '01.pos1_f',
    #                                  '02.pos2_d', '02.pos2_e', '02.pos2_f',
    #                                  '03.pos3_d', '03.pos3_e', '03.pos3_f',
    #                                  '04.pos4_d', '04.pos4_e', '04.pos4_f'],
    #              '25.KS2_2024_ART': ['01.pos1_a', '01.pos1_b', '01.pos1_c',
    #                                  '02.pos2_a', '02.pos2_b', '02.pos2_c',
    #                                  '03.pos3_a', '03.pos3_b', '03.pos3_c',
    #                                  '04.pos4_a', '04.pos4_b', '04.pos4_c',
    #                                  '01.pos1_d', '01.pos1_e', '01.pos1_f',
    #                                  '02.pos2_d', '02.pos2_e', '02.pos2_f',
    #                                  '03.pos3_d', '03.pos3_e', '03.pos3_f',
    #                                  '04.pos4_d', '04.pos4_e', '04.pos4_f'],
    #             }
    # art_lists = {'21.KS2_2005_ART': ['Artstarlist_2005_1.txt',
    #                                  'Artstarlist_2005_2.txt',
    #                                  'Artstarlist_2005_3.txt',
    #                                  'Artstarlist_2005_4.txt',
    #                                  'Artstarlist_2005_5.txt',
    #                                  'Artstarlist_2005_6.txt',
    #                                  'Artstarlist_2005_7.txt',
    #                                  'Artstarlist_2005_8.txt',
    #                                  'Artstarlist_2005_9.txt',
    #                                  'Artstarlist_2005_10.txt',
    #                                  'Artstarlist_2005_11.txt',
    #                                  'Artstarlist_2005_12.txt',
    #                                  'Artstarlist_2005_13.txt',
    #                                  'Artstarlist_2005_14.txt'],
    #              '22.KS2_2010_ART': ['Artstarlist_2010_pos1_a.txt',
    #                                  'Artstarlist_2010_pos1_b.txt',
    #                                  'Artstarlist_2010_pos1_c.txt',
    #                                  'Artstarlist_2010_pos2_a.txt',
    #                                  'Artstarlist_2010_pos2_b.txt',
    #                                  'Artstarlist_2010_pos2_c.txt',
    #                                  'Artstarlist_2010_pos3_a.txt',
    #                                  'Artstarlist_2010_pos3_b.txt',
    #                                  'Artstarlist_2010_pos3_c.txt',
    #                                  'Artstarlist_2010_pos4_a.txt',
    #                                  'Artstarlist_2010_pos4_b.txt',
    #                                  'Artstarlist_2010_pos4_c.txt',
    #                                  'Artstarlist_2010_pos1_d.txt',
    #                                  'Artstarlist_2010_pos1_e.txt',
    #                                  'Artstarlist_2010_pos1_f.txt',
    #                                  'Artstarlist_2010_pos2_d.txt',
    #                                  'Artstarlist_2010_pos2_e.txt',
    #                                  'Artstarlist_2010_pos2_f.txt',
    #                                  'Artstarlist_2010_pos3_d.txt',
    #                                  'Artstarlist_2010_pos3_e.txt',
    #                                  'Artstarlist_2010_pos3_f.txt',
    #                                  'Artstarlist_2010_pos4_d.txt',
    #                                  'Artstarlist_2010_pos4_e.txt',
    #                                  'Artstarlist_2010_pos4_f.txt'],
    #              '23.KS2_2013_ART': ['Artstarlist_2013_pos1_a.txt',
    #                                  'Artstarlist_2013_pos1_b.txt',
    #                                  'Artstarlist_2013_pos1_c.txt',
    #                                  'Artstarlist_2013_pos2_a.txt',
    #                                  'Artstarlist_2013_pos2_b.txt',
    #                                  'Artstarlist_2013_pos2_c.txt',
    #                                  'Artstarlist_2013_pos3_a.txt',
    #                                  'Artstarlist_2013_pos3_b.txt',
    #                                  'Artstarlist_2013_pos3_c.txt',
    #                                  'Artstarlist_2013_pos4_a.txt',
    #                                  'Artstarlist_2013_pos4_b.txt',
    #                                  'Artstarlist_2013_pos4_c.txt',
    #                                  'Artstarlist_2013_pos1_d.txt',
    #                                  'Artstarlist_2013_pos1_e.txt',
    #                                  'Artstarlist_2013_pos1_f.txt',
    #                                  'Artstarlist_2013_pos2_d.txt',
    #                                  'Artstarlist_2013_pos2_e.txt',
    #                                  'Artstarlist_2013_pos2_f.txt',
    #                                  'Artstarlist_2013_pos3_d.txt',
    #                                  'Artstarlist_2013_pos3_e.txt',
    #                                  'Artstarlist_2013_pos3_f.txt',
    #                                  'Artstarlist_2013_pos4_d.txt',
    #                                  'Artstarlist_2013_pos4_e.txt',
    #                                  'Artstarlist_2013_pos4_f.txt'],
    #              '24.KS2_2015_ART': ['Artstarlist_2015_pos1_a.txt',
    #                                  'Artstarlist_2015_pos1_b.txt',
    #                                  'Artstarlist_2015_pos1_c.txt',
    #                                  'Artstarlist_2015_pos2_a.txt',
    #                                  'Artstarlist_2015_pos2_b.txt',
    #                                  'Artstarlist_2015_pos2_c.txt',
    #                                  'Artstarlist_2015_pos3_a.txt',
    #                                  'Artstarlist_2015_pos3_b.txt',
    #                                  'Artstarlist_2015_pos3_c.txt',
    #                                  'Artstarlist_2015_pos4_a.txt',
    #                                  'Artstarlist_2015_pos4_b.txt',
    #                                  'Artstarlist_2015_pos4_c.txt',
    #                                  'Artstarlist_2015_pos1_d.txt',
    #                                  'Artstarlist_2015_pos1_e.txt',
    #                                  'Artstarlist_2015_pos1_f.txt',
    #                                  'Artstarlist_2015_pos2_d.txt',
    #                                  'Artstarlist_2015_pos2_e.txt',
    #                                  'Artstarlist_2015_pos2_f.txt',
    #                                  'Artstarlist_2015_pos3_d.txt',
    #                                  'Artstarlist_2015_pos3_e.txt',
    #                                  'Artstarlist_2015_pos3_f.txt',
    #                                  'Artstarlist_2015_pos4_d.txt',
    #                                  'Artstarlist_2015_pos4_e.txt',
    #                                  'Artstarlist_2015_pos4_f.txt'],
    #              '25.KS2_2024_ART': ['Artstarlist_2024_pos1_a.txt',
    #                                  'Artstarlist_2024_pos1_b.txt',
    #                                  'Artstarlist_2024_pos1_c.txt',
    #                                  'Artstarlist_2024_pos2_a.txt',
    #                                  'Artstarlist_2024_pos2_b.txt',
    #                                  'Artstarlist_2024_pos2_c.txt',
    #                                  'Artstarlist_2024_pos3_a.txt',
    #                                  'Artstarlist_2024_pos3_b.txt',
    #                                  'Artstarlist_2024_pos3_c.txt',
    #                                  'Artstarlist_2024_pos4_a.txt',
    #                                  'Artstarlist_2024_pos4_b.txt',
    #                                  'Artstarlist_2024_pos4_c.txt',
    #                                  'Artstarlist_2024_pos1_d.txt',
    #                                  'Artstarlist_2024_pos1_e.txt',
    #                                  'Artstarlist_2024_pos1_f.txt',
    #                                  'Artstarlist_2024_pos2_d.txt',
    #                                  'Artstarlist_2024_pos2_e.txt',
    #                                  'Artstarlist_2024_pos2_f.txt',
    #                                  'Artstarlist_2024_pos3_d.txt',
    #                                  'Artstarlist_2024_pos3_e.txt',
    #                                  'Artstarlist_2024_pos3_f.txt',
    #                                  'Artstarlist_2024_pos4_d.txt',
    #                                  'Artstarlist_2024_pos4_e.txt',
    #                                  'Artstarlist_2024_pos4_f.txt']                
    #             }

    # filters = {'21.KS2_2005_ART': [1],
    #            '22.KS2_2010_ART': [1, 2, 3],
    #            '23.KS2_2013_ART': [1],
    #            '24.KS2_2015_ART': [1],
    #            '25.KS2_2024_ART': [1]}

    # return comp_dirs, art_lists, filters


def call_process_ks2_artstar(comp_dirs, art_lists, filters, ks2_update):
    # comp_dirs, art_lists, filters = setup_artstar_info()

    # Loop through each year
    for epoch in list(comp_dirs.keys()):
        art_dirs = comp_dirs[epoch]
                
        # Loop through each position, sublist combo
        for ii in range(len(art_dirs)):
            directory = work_dir + '/' + epoch + '/' + art_dirs[ii]
            os.chdir(directory)

            filt = filters[epoch]

            # Loop through each filter            
            for ff in tqdm(range(len(filt)), desc=f'Processing epoch {epoch}, subset {ii+1}/{len(art_dirs)}'):
                filt_string = 'F{0:d}'.format(filt[ff])
                print('Working on {0}, filter F{1}'.format(directory, filt_string))

                art_star_list = art_lists[epoch][ii]

                if '2005' in epoch:
                    nimfo2bar_file = 'nimfo2bar.xymeee.ks2.' + filt_string
                else:
                    nimfo2bar_file = 'nimfo2bar.xymeee.' + filt_string
                comp.process_ks2_artstar(art_star_list, nimfo2bar_file,
                                         filter_num=filt[ff], ks2_update=ks2_update)

            
    return

def recombine_artstar_subsets(comp_dirs, art_lists, filters):
    """
    Take the subsets of the artificial star lists and put them
    back together again. We will only work with the final output
    catalogs form call_process_ks2_artstar().

    Last step is top save the catalogs in 51.ALIGN_ART/.
    """
    # comp_dirs, art_lists, filters = setup_artstar_info()

    ##################################################
    #
    # Process 21.KS2_2005_ART -- combine everything (single filter)
    #
    ##################################################
    epoch = '21.KS2_2005_ART'
    t_2005_F814W = None
    subdirs = comp_dirs[epoch]
    alists = art_lists[epoch]

    N_largest = 0
    for ii in range(len(subdirs)):
        table_file = work_dir + '/' + epoch + '/' + subdirs[ii] + '/'
        table_file += alists[ii].replace('.txt', '')
        table_file += '_F1_joined.fits'

        t = Table.read(table_file)

        if ii == 0:
            t_2005_F814W = t
        else:
            # Modify the names (which are actually indices).
            t['name'] += N_largest
            t_2005_F814W = astropy.table.vstack((t_2005_F814W, t), join_type='exact')
            
        N_largest = t_2005_F814W['name'][-1]

    # Save output to file.
    outfile = 'artstar_2005_F814W.fits'
    outfile = work_dir + '/51.ALIGN_ART/' + outfile
    t_2005_F814W.write(outfile, overwrite=True)



    ##################################################
    #
    # Process 22.KS2_2010_ART -- combine each position (three filters)
    #
    ##################################################
    epoch = '22.KS2_2010_ART'
    positions = ['pos1_1', 'pos2_1', 'pos3_1', 'pos4_1', 'pos1_2', 'pos2_2', 'pos3_2', 'pos4_2']
    subsets = {'pos1_1': ['a', 'b', 'c'],
               'pos2_1': ['a', 'b', 'c'],
               'pos3_1': ['a', 'b', 'c'],
               'pos4_1': ['a', 'b', 'c'],
               'pos1_2': ['d', 'e', 'f'],
               'pos2_2': ['d', 'e', 'f'],
               'pos3_2': ['d', 'e', 'f'],
               'pos4_2': ['d', 'e', 'f']}
    filter_names = {1: 'F160W', 2: 'F139M', 3: 'F125W'}

    for pp in range(len(positions)):
        # The name of this mosaic position.
        pos_out = positions[pp]
        pos = positions[pp].split('_')[0]
        pos_idx = int(pos[-1])
        
        # List of subset directory suffixes for this mosaic position.
        subs = subsets[pos_out]

        for filt in filters[epoch]:
            t_2010 = None
            N_largest = 0

            for ii in range(len(subs)):
                table_file = work_dir + '/' + epoch
                table_file += '/{0:02d}.{1:s}_{2:s}/'.format(pos_idx, pos, subs[ii])
                table_file += 'Artstarlist_2010_{0:s}_{1:s}'.format(pos, subs[ii])
                table_file += '_F{0:d}'.format(filt)
                table_file += '_joined.fits'
            
                t = Table.read(table_file)

                if ii == 0:
                    t_2010 = t
                else:
                    # Modify the names (which are actually indices).
                    t['name'] += N_largest
                    t_2010 = astropy.table.vstack((t_2010, t), join_type='exact')
            
                N_largest = t_2010['name'][-1]

            # Save output to file.
            outfile = 'artstar_2010_{0}_{1}.fits'.format(filter_names[filt], pos_out)
            outfile = work_dir + '、51.ALIGN_ART/' + outfile
            t_2010.write(outfile, overwrite=True)

    ##################################################
    #
    # Process 23.KS2_2013_ART -- combine each position (three filters)
    #
    ##################################################
    epoch = '23.KS2_2013_ART'
    positions = ['pos1_1', 'pos2_1', 'pos3_1', 'pos4_1', 'pos1_2', 'pos2_2', 'pos3_2', 'pos4_2']
    subsets = {'pos1_1': ['a', 'b', 'c'],
               'pos2_1': ['a', 'b', 'c'],
               'pos3_1': ['a', 'b', 'c'],
               'pos4_1': ['a', 'b', 'c'],
               'pos1_2': ['d', 'e', 'f'],
               'pos2_2': ['d', 'e', 'f'],
               'pos3_2': ['d', 'e', 'f'],
               'pos4_2': ['d', 'e', 'f']}
    filter_names = {1: 'F160W'}

    for pp in range(len(positions)):
        # The name of this mosaic position.
        pos_out = positions[pp]
        pos = positions[pp].split('_')[0]
        pos_idx = int(pos[-1])
        
        # List of subset directory suffixes for this mosaic position.
        subs = subsets[pos_out]

        for filt in filters[epoch]:
            t_2013 = None
            N_largest = 0

            for ii in range(len(subs)):
                table_file = work_dir + '/' + epoch
                table_file += '/{0:02d}.{1:s}_{2:s}/'.format(pos_idx, pos, subs[ii])
                table_file += 'Artstarlist_2013_{0:s}_{1:s}'.format(pos, subs[ii])
                table_file += '_F{0:d}'.format(filt)
                table_file += '_joined.fits'
            
                t = Table.read(table_file)

                if ii == 0:
                    t_2013 = t
                else:
                    # Modify the names (which are actually indices).
                    t['name'] += N_largest
                    t_2013 = astropy.table.vstack((t_2013, t), join_type='exact')
            
                N_largest = t_2013['name'][-1]

            # Save output to file.
            outfile = 'artstar_2013_{0}_{1}.fits'.format(filter_names[filt], pos_out)
            outfile = work_dir + '/51.ALIGN_ART/' + outfile
            t_2013.write(outfile, overwrite=True)

    return

def align_artstars(use_obs_align=False):
    """
    Read in the artificial star catalogs, transform them using the
    alignments derived from the observed data, cross-match the stars across
    epochs, and apply trim_align cuts in a similar manner to the observed data.

    Deliver a FITS catalog that parallels the one delivered from
        align_to_fits()
        combine_mosaic_pos()
    This will have the mosaic positions combined together. 
    """
    mosaic_epochs = ['2005_F814W', '2010_F125W', '2010_F139M',
                     '2010_F160W', '2013_F160W']
    pos_for_epoch = {'2005_F814W': [''],
                     '2010_F125W': ['pos1_1', 'pos2_1', 'pos3_1', 'pos4_1', 
                                    'pos1_2', 'pos2_2', 'pos3_2', 'pos4_2'],
                     '2010_F139M': ['pos1_1', 'pos2_1', 'pos3_1', 'pos4_1',
                                    'pos1_2', 'pos2_2', 'pos3_2', 'pos4_2'],
                     '2010_F160W': ['pos1_1', 'pos2_1', 'pos3_1', 'pos4_1',
                                    'pos1_2', 'pos2_2', 'pos3_2', 'pos4_2'],
                     '2013_F160W': ['pos1_1', 'pos2_1', 'pos3_1', 'pos4_1',
                                    'pos1_2', 'pos2_2', 'pos3_2', 'pos4_2']}
    
    # WARNING Hard-coding the order -- arrays above need to be matched to
    # the align.list file from the observed data. No checks!!
    align_indices = {'2005_F814W': [0], 
                     '2010_F125W': [1, 2, 3, 4, 1, 2, 3, 4], 
                     '2010_F139M': [5, 6, 7, 8, 5, 6, 7, 8], 
                     '2010_F160W': [9, 10, 11, 12, 9, 10, 11, 12],
                     '2013_F160W': [13, 14, 15, 16, 13, 14, 15, 16]}

    if use_obs_align:
        align_root = 'align_a4_t'
        align_dir = work_dir + '/50.ALIGN_KS2/'
    else:
        align_root = 'align_a4_t'
        align_dir = work_dir + '/51.ALIGN_ART/align_a4_t/'

    # This table contains the transformations that will be applied.
    trans = Table.read(align_dir + align_root + '.trans',
                       format='ascii')

    # This will be used to validate that we are working on the right list. 
    align_list_file = open(align_dir + align_root + '.list', 'r')
    align_list = align_list_file.readlines()

    def transform(xin, yin, tr):
        """
        xin - array
        yin - array
        tr - row from an align.trans file
        """
        xout = tr['a0']
        xout += (tr['a1'] * xin) + (tr['a2'] * yin)
        xout += (tr['a3'] * xin**2) + (tr['a4'] * xin * yin) + (tr['a5'] * yin**2)
        
        yout = tr['b0']
        yout += (tr['b1'] * yin) + (tr['b2'] * yin)
        yout += (tr['b3'] * yin**2) + (tr['b4'] * yin * xin) + (tr['b5'] * xin**2)

        return (xout, yout)

    def transform_error(xin, yin, xin_e, yin_e, tr):
        """
        xin - array
        yin - array
        tr - row from an align.trans file
        """
        xe_term1 = (tr['a1'] * xin_e)
        xe_term1 += (tr['a3'] * 2 * xin * xin_e) + (tr['a4'] * yin * xin_e)

        xe_term2 = (tr['a2'] * yin_e)
        xe_term2 += (tr['a5'] * 2 * yin * yin_e) + (tr['a4'] * xin * yin_e)
        
        xout_e = np.hypot(xe_term1, xe_term2)

        ye_term1 = (tr['b1'] * yin_e)
        ye_term1 += (tr['b3'] * 2 * yin * yin_e) + (tr['b4'] * xin * yin_e)

        ye_term2 = (tr['b2'] * xin_e)
        ye_term2 += (tr['b5'] * 2 * xin * xin_e) + (tr['b4'] * yin * xin_e)
        
        yout_e = np.hypot(ye_term1, ye_term2)
        
        return (xout_e, yout_e)
    
    # This will be the final table
    t_out = Table()
    t_out.table_name = 'art_align'

    if use_obs_align:
        t_out.table_name = t_out.table_name + '_obs'
    else:
        t_out.table_name = t_out.table_name + '_art'

    # Loop through each epoch (2005, 2010, 2013)
    for ee in range(len(mosaic_epochs)):
        epoch = mosaic_epochs[ee]
        positions = pos_for_epoch[epoch]
        print('')
        print('Combining all positions from epoch = ', epoch)
        

        # Define the final output arrays 
        x = np.array([], dtype=float)
        y = np.array([], dtype=float)
        m = np.array([], dtype=float)
        xe = np.array([], dtype=float)
        ye = np.array([], dtype=float)
        me = np.array([], dtype=float)
        n = np.array([], dtype=int)
        x_in = np.array([], dtype=float)
        y_in = np.array([], dtype=float)
        m_in = np.array([], dtype=float)
        name = np.array([], dtype='U10')

        # Loop through the mosaic positions (pos1, pos2, pos3, pos4)
        for pp in range(len(positions)):
            # Get the current position
            pos = positions[pp]
            align_idx = align_indices[epoch][pp]
            pos_only = pos.split('_')[0]

            # Get the transformation for this epoch
            tr = trans[align_idx]

            # Make a suffix string from the current position
            if pos != '':
                pos_suffix = '_' + pos
            else:
                pos_suffix = pos

            # Read in the table
            artstar_list = 'artstar_{0}{1}.fits'.format(epoch, pos_suffix)

            print('   ks2 list = ', artstar_list, '  align list = ', align_list[align_idx].split()[0])

            t = Table.read(work_dir + '/51.ALIGN_ART/' + artstar_list)

            # Transfrom the stars' input positions.
            x_in_t, y_in_t = transform(t['x_in'], t['y_in'], tr)

            # Identify the detected stars and only transform them.
            det = np.where(t['n_out'] > 0)[0]
            x_out_t = t['x_out']
            y_out_t = t['y_out']
            xe_out_t = t['xe_out']
            ye_out_t = t['ye_out']
            
            x_out_t[det], y_out_t[det] = transform(t['x_out'][det], t['y_out'][det],
                                                   tr)
            xe_out_t[det], ye_out_t[det] = transform_error(t['x_out'][det],
                                                           t['y_out'][det],
                                                           t['xe_out'][det],
                                                           t['ye_out'][det],
                                                           tr)

            # Append input values to the final arrays
            x_in = np.append(x_in, x_in_t)
            y_in = np.append(y_in, y_in_t)
            m_in = np.append(m_in, t['m_in'])

            # Append detected values to the final arrays
            x = np.append(x, x_out_t)
            y = np.append(y, y_out_t)
            m = np.append(m, t['m_out'])
            n = np.append(n, t['n_out'])

            xe = np.append(xe, xe_out_t)
            ye = np.append(ye, ye_out_t)
            me = np.append(me, t['me_out'])

            new_name = [str(nn) + pos_suffix for nn in t['name']]

            name = np.append(name, new_name)

            print('   position = {0}, added {1} star'.format(pos, len(x_out_t)))
            print('')

        # Add new columns to the table
        t_out.add_column(Column(x, name='x_' + epoch))
        t_out.add_column(Column(y, name='y_' + epoch))
        t_out.add_column(Column(m, name='m_' + epoch))

        t_out.add_column(Column(xe, name='xe_' + epoch))
        t_out.add_column(Column(ye, name='ye_' + epoch))
        t_out.add_column(Column(me, name='me_' + epoch))
        
        t_out.add_column(Column(n, name='n_' + epoch))

        t_out.add_column(Column(x_in, name='xin_' + epoch))
        t_out.add_column(Column(y_in, name='yin_' + epoch))
        t_out.add_column(Column(m_in, name='min_' + epoch))

        t_out.add_column(Column(name, name='name_' + epoch))

            
    #fix_art_magnitudes(t_out)

    file_name = work_dir + '/51.ALIGN_ART/art_align'
    if use_obs_align:
        file_name += '_obs'
    else:
        file_name += '_art'
    file_name += '_combo_pos.fits'

    print(file_name)
    
    t_out.write(file_name, overwrite=True)

    return


def fix_art_magnitudes(t):
    t['m_2005_F814W'] -= -2.5 * np.log10(2407. / 3.)
    t['min_2005_F814W'] -= -2.5 * np.log10(2407. / 3.)

    return
    
def fix_magnitudes(t):
    t['m_2005_F814W'] -= -2.5 * np.log10(2407. / 3.)

    return

def catalog_fix_magnitudes(catalog_in, catalog_out):
    # Read in the table.
    d = Table.read(catalog_in)
    
    # Fix the F814W magnitudes (adjust for integration time)
    fix_magnitudes(d)

    d.write(catalog_out, overwrite=True)

    return


def plot_chi2_dist(startable, chi2_lim=10, bins=20, log=True, n_epoch=4, dof=2,
                   truncate_at=None, save_path=None):
    """chi2 histogram against its expected distribution.

    Both sides are densities over the WHOLE sample, not over the plotted range:
    each axis is weighted by 1 / (N_axis * bin_width) using the count before
    any trimming, and the reference is the plain chi2 pdf. So the bars in view
    integrate to the fraction of the sample that is in view, and the curve to
    chi2.cdf(upper, dof) -- a sample that really follows chi2(dof) puts them on
    top of each other, and any mass outside the range is simply missing from
    the bars.

    That is the point of normalizing this way. Renormalizing each side to its
    own in-view mass instead (density=True against a truncated pdf) makes the
    comparison conditional on landing in view, and then a heavy tail comes back
    as a peak EXCESS: pass 3's dof=3 group puts 12.5% of its chi2 past 4*dof
    where 0.7% was expected, which inflated the bars by 1/0.875 against 1/0.993
    for the curve and lifted the peak ~13% above it. Read that way the plot
    reported a peak problem when what it had was a tail problem. Here the same
    sample instead reads as bars sitting below the curve, with the deficit
    equal to the tail that overflowed -- which is what it is.

    truncate_at : float, optional
        Cut the sample here (see flag_chi2_outliers), per axis -- the flag is
        per STAR, so trimming the table first would pull good x values out of
        the x histogram. The cut values stay in the normalization (they are
        part of the sample, just not of the histogram), and the reference is
        drawn to the same point.
    """
    xs = np.linspace(0, chi2_lim, 100)
    fig, ax = plt.subplots()
    c2x = np.asarray(startable['chi2_x']); c2y = np.asarray(startable['chi2_y'])
    c2x = c2x[np.isfinite(c2x)]; c2y = c2y[np.isfinite(c2y)]

    # Diagnostics come off the un-cut sample; the histogram gets the cut one.
    # n_all is taken here, before trimming, because it is what the bars are
    # normalized by -- counting only the survivors would put the tail's mass
    # back into the peak, which is exactly what this plot is built to avoid.
    raw = np.concatenate([c2x, c2y])
    n_all = (len(c2x), len(c2y))
    cut = np.mean(raw >= truncate_at) if truncate_at else 0.0
    if truncate_at:
        c2x = c2x[c2x < truncate_at]; c2y = c2y[c2y < truncate_at]
    upper = min(chi2_lim, truncate_at) if truncate_at else chi2_lim

    width = chi2_lim / bins
    for c2, n, label in ((c2x, n_all[0], r'$\chi^2(v_x)$'),
                         (c2y, n_all[1], r'$\chi^2(v_y)$')):
        ax.hist(c2, bins=bins, range=(0, chi2_lim), histtype='step',
                weights=np.full(len(c2), 1.0 / (n * width)),
                log=log, label=label)
    keep = xs <= upper
    ax.plot(xs[keep], chi2.pdf(xs[keep], dof), color='C3',
            label=r'$\chi_{' + str(dof) + '}^2$')
    ax.set_xlabel(r'$\chi^2$')
    ax.set_ylabel('Density')
    ax.legend()
    # Medians are immune to binning and to the normalization above, so they are
    # the honest check on whether the errors are scaled correctly. The tail
    # numbers say how much of the sample is off the right-hand edge.
    sub = (f'{100*np.mean(raw > chi2_lim):.1f}% beyond {chi2_lim:g} '
           f'(expect {100*chi2.sf(chi2_lim, dof):.1f}%)')
    if truncate_at:
        sub += (f', {100*cut:.1f}% cut at {truncate_at:.1f} '
                f'(expect {100*chi2.sf(truncate_at, dof):.1f}%)')
    ax.set_title(r'$N_\mathrm{epoch}=' + str(n_epoch) + r', N_\mathrm{dof}=' + str(dof) + r'$'
                 + f'   median {np.median(raw):.2f} (expect {chi2.median(dof):.2f})\n' + sub,
                 fontsize='medium')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    plt.show()


def plot_chi2_dist_parallax(startable, chi2_lim=10, bins=20, log=True, n_epoch=4, dof=2,
                            truncate_at=None, save_path=None):
    """As plot_chi2_dist, for the joint (both-axis) parallax chi2.

    Normalized the same way -- see plot_chi2_dist for how to read it.
    """
    xs = np.linspace(0, chi2_lim, 100)
    fig, ax = plt.subplots()
    c2x = np.asarray(startable['chi2_x']); c2y = np.asarray(startable['chi2_y'])
    c2x = c2x[np.isfinite(c2x)]; c2y = c2y[np.isfinite(c2y)]
    c2j = np.hypot(c2x, c2y)
    if truncate_at:
        c2x = c2x[c2x < truncate_at]; c2y = c2y[c2y < truncate_at]
        c2j = c2j[c2j < truncate_at]
    upper = min(chi2_lim, truncate_at) if truncate_at else chi2_lim
    for c2, kw in ((c2x, dict(histtype='step', label=r'$\chi^2(v_x)$')),
                   (c2y, dict(histtype='step', label=r'$\chi^2(v_y)$')),
                   (c2j, dict(color='C3', alpha=0.3, label=r'$\chi^2_\mathrm{parallax}$'))):
        ax.hist(c2, bins=bins, range=(0, chi2_lim), density=True, log=log, **kw)
    keep = xs <= upper
    ax.plot(xs[keep], chi2.pdf(xs[keep], dof) / chi2.cdf(upper, dof), color='C3',
            label=r'$\chi_{' + str(dof) + '}^2$')
    ax.set_xlabel(r'$\chi^2$')
    ax.set_ylabel('Density')
    ax.legend()
    ax.set_title(r'$N_\mathrm{epoch}=' + str(n_epoch) + r', N_\mathrm{dof}=' + str(dof) + r'$')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    plt.show()

# def junk():
#         ahdr = '{0:<10s}  {1:5s}  {2:8s}  {3:8s}  {4:8s}  {5:8s}  ' + \
#       '{6:8s}  {7:8s}  {8:8s}  {9:8s}  {10:4s}  {11:8s}\n'
#     afmt = '{0:<10s}  {1:5.2f}  {2:8.3f}  {3:8.3f}  {4:8.3f}  {5:8.3f}  ' + \
#       '{6:8.3f}  {7:8.3f}  {8:8.3f}  {9:8.3f}  {10:4d}  {11:8.2f}\n'

#     _o_align.write(ahdr.format('#Name', 'mag', 'x', 'y', 'xerr', 'yerr',
#                                    'vx', 'vy', 'vxerr', 'vyerr', 't0', 'use?', 'r2d'))
#     _o_align.write(ahdr.format('#()', '(mag)', '(asec)', '(asec)', '(asec)', '(asec)',
#                                    '(mas/yr)', '(mas/yr)', '(mas/yr)', '(mas/yr)', '(year)', '()', '(asec)'))
#         _o_align.write(afmt.format(t2005_814['name'][ii].decode('utf-8'),
#                                    t2005_814['m'][ii] + photometry.ZP['F814W'],
#                                    t2005_814['x'][ii] * ast.scale['WFC'],
#                                    t2005_814['y'][ii] * ast.scale['WFC'],
#                                    t2005_814['xe'][ii] * ast.scale['WFC'],
#                                    t2005_814['ye'][ii] * ast.scale['WFC'],
#                                    0.0, 0.0, 0.0, 0.0, 18,
#                                    np.hypot(t2005_814['x'][ii], t2005_814['y'][ii]) * ast.scale['WFC']))

# def fit_velocity():
#     catalog_name = 'wd1_catalog'
#     if use_RMSE:
#         catalog_name += '_RMSE'
#     else:
#         catalog_name += '_EOM'

#     if vel_weight == None:
#         catalog_name += '_wvelNone'
#     else:
#         if vel_weight == 'error':
#             catalog_name += '_wvelErr'
#         if vel_weight == 'variance':
#             catalog_name += '_wvelVar'
#     catalog_name += '.fits'

#     d.write(work_dir + '/50.ALIGN_KS2/' + catalog_name, format='fits', overwrite=True)
    
def plot_matchup_error(matchup_file, mag_range=None, xlabel=None, title=None):
    table = read_matchup(matchup_file)
    table = table[(table['xe'] != 1.) & (table['ye'] != 1.)]
    N = len(table)
    if mag_range is None:
        mag_range = (np.nanmin(table['m']), np.nanmax(table['m']))
    
    table_trim = table[(table['m'] >= mag_range[0]) & (table['m'] <= mag_range[1])]
    
    mag_min = None
    # mag_max = -14
    
    fig, ax = plt.subplots()
    ax.plot(table['m'], table['xe'], color='C0', marker='.', ms=2, alpha=0.5, ls='none', label='X')
    ax.plot(table['m'], table['ye'], color='C1', marker='.', ms=2, alpha=0.5, ls='none', label='Y')
    # plt.axvline(mag_max, ls='--', color='C3')
    if xlabel is None:
        ax.set_xlabel('Mag', fontsize=12)
    else:
        ax.set_xlabel(xlabel, fontsize=12)
    
    if title:
        ax.set_title(title, fontsize=14)
    ax.set_ylabel('Astrometric Uncertainty (pix)', fontsize=12)
    ax.legend()
    # ax.set_xlim([-22, 0])
    # ax.set_ylim([1e-3, 10])
    ax.set_yscale('log')
    plt.tight_layout()
    plt.show()

    # mag_range = (-13.5, -5)
    # within_mag_range = (table['m'] >= mag_range[0]) & (table['m'] <= mag_range[1])
    # table_within_range = table[within_mag_range]
    xerr_median = np.median(table_trim['xe'])
    yerr_median = np.median(table_trim['ye'])
    merr_median = np.median(table_trim['me'])
    print(f'Median Errors for {len(table_trim)} of {N} stars between {mag_range[0]:.2f} - {mag_range[-1]:.2f}:')
    print(f'Median X error: {xerr_median:.3f} mas')
    print(f'Median Y error: {yerr_median:.3f} mas')
    print(f'Median Mag error: {merr_median:.3f} mag')





# ============================== #
# === Added by Lingfeng 2026 === #
# ============================== #

def get_valid_idx(wd1, epoch_cols):
    """Get valid indices for the specified epochs."""
    valid_x = np.isfinite(wd1['x'][:, epoch_cols])
    valid_y = np.isfinite(wd1['y'][:, epoch_cols])
    valid_xe = (wd1['xe'][:, epoch_cols]!=0) & (np.isfinite(wd1['xe'][:, epoch_cols]))
    valid_ye = (wd1['ye'][:, epoch_cols]!=0) & (np.isfinite(wd1['ye'][:, epoch_cols]))
    return valid_x & valid_y & valid_xe & valid_ye

# 1.4826 = 1/0.6745, the factor that makes the median absolute deviation an
# unbiased estimate of sigma for Gaussian data. Unlike rms it has a 50%
# breakdown point, so blends and mismatches don't set the width.
nmad = lambda a: 1.4826 * np.median(np.abs(a - np.median(a)))


def _wls_linear(pos, err, valid, times):
    """Per-star weighted linear fit. Returns (residuals, leverage h).

    h is the diagonal of the hat matrix H = A (A^T W A)^-1 A^T W, so that
    Var(r_j) = sigma_j^2 (1 - h_j) and sum_j (1 - h_j) = dof. Residuals must
    be divided by sigma_j*sqrt(1 - h_j) -- not sigma_j -- before their width
    means anything, because the fit absorbs part of each point's noise in
    proportion to that point's lever arm.
    """
    wt = np.where(valid, 1.0 / err**2, 0.0)
    t0 = (wt * times).sum(1) / wt.sum(1)
    dt = times[None, :] - t0[:, None]
    Sw, Swt, Swtt = wt.sum(1), (wt * dt).sum(1), (wt * dt * dt).sum(1)
    det = Sw * Swtt - Swt**2
    C00, C01, C11 = Swtt / det, -Swt / det, Sw / det
    r0, r1 = (wt * pos).sum(1), (wt * pos * dt).sum(1)
    a0, a1 = C00 * r0 + C01 * r1, C01 * r0 + C11 * r1
    resid = np.where(valid, pos - (a0[:, None] + a1[:, None] * dt), np.nan)
    h = wt * (C00[:, None] + 2 * C01[:, None] * dt + C11[:, None] * dt * dt)
    return resid, h


def _robust_cov(M):
    """Outlier-resistant covariance via NMAD: Cov(a,b) = [nmad(a+b)^2 - nmad(a-b)^2]/4."""
    n = M.shape[1]
    C = np.empty((n, n))
    for i in range(n):
        C[i, i] = nmad(M[:, i])**2
        for j in range(i + 1, n):
            C[i, j] = C[j, i] = (nmad(M[:, i] + M[:, j])**2 - nmad(M[:, i] - M[:, j])**2) / 4
    return C


def _solve_block(pos, err, times):
    """Per-epoch inflation factors for one homogeneous block of stars.

    Matches the full observed residual covariance to (I-H) Sigma (I-H)^T rather
    than matching per-epoch residual *widths*. The widths alone are not enough:
    r = (I-H)eps has rank n-2, so two of the n width equations are dependent and
    a width-matching iteration is underdetermined -- in this data it ran away to
    f=0.35 for 2015_F160W and f=4.8 for 2024_F160W, deflating one epoch to pay
    for another. The covariance has n(n+1)/2 entries for n unknowns, which is
    over-determined and well behaved.
    """
    n = len(times)
    resid, _ = _wls_linear(pos, err, np.ones_like(pos, bool), times)
    Cobs = _robust_cov(resid)
    q = np.median(err, axis=0)
    iu = np.triu_indices(n)
    scale = np.abs(Cobs[iu]).max()

    def model(th):
        sig = np.exp(th) * q
        wt = 1.0 / sig**2
        A = np.column_stack([np.ones(n), times - np.average(times, weights=wt)])
        H = A @ np.linalg.inv(A.T @ np.diag(wt) @ A) @ A.T @ np.diag(wt)
        IH = np.eye(n) - H
        return (IH @ np.diag(sig**2) @ IH.T)[iu]

    # Bounded: an unbounded solve can drive sigma to overflow, which zeroes the
    # weights and makes A^T W A singular.
    sol = least_squares(lambda th: (model(th) - Cobs[iu]) / scale, np.zeros(n),
                        bounds=(np.log(0.2), np.log(20.0)), max_nfev=40000)
    return np.exp(sol.x)


def calibrate_epoch_errors(tab, epoch_cols=None,
                           mag_bins=(12, 19, 20.5, 21.5, 22.5, 23.5, 26),
                           min_stars=250, mag_col='m0', apply=True, verbose=True):
    """Rescale per-epoch xe/ye so that fit residuals follow the expected chi2.

    Solves for a multiplicative factor per (epoch, axis, magnitude), which is the
    right functional form here: the residual pull is roughly flat in magnitude
    while the implied additive floor tracks sigma, i.e. the errors are wrong by a
    factor rather than missing a constant systematic term.

    Stars are grouped into blocks sharing a detection pattern and a magnitude bin
    so that every star in a block has the same epochs and a comparable error
    scale; each block is solved independently and the factors are then
    interpolated in magnitude per epoch.

    epoch_cols selects which epochs take part; the default is all of them. Pass
    the SAME columns that will enter fit_motion_models (i.e. everything not in
    its mask_lists), because the factors are measured from the residuals of a
    fit to exactly these epochs -- calibrating against a 7-epoch fit and then
    fitting only 4 measures the errors under the wrong model. Epochs outside
    epoch_cols are left untouched.

    min_stars gates both the detection pattern and each magnitude bin within it,
    so raising it silently drops whole bins -- and a bin with no block is not
    left alone, it is flat-extrapolated from the nearest bin that did solve. The
    bright end is where that bites: bright stars saturate out of individual
    epochs, so few of them share the common detection patterns (Wd1 pass 3 has
    467 stars brighter than m0=19 in the 4-epoch pattern and 764 in the
    5-epoch one). At the previous default of 800 every bright bin failed, the
    brightest surviving block centre was 19.75, and everything above it inherited
    that factor: m0 < 19 came out at 2.4x the expected median chi2 uncalibrated
    and still 1.6x calibrated. At 250 those bins solve and it lands at 1.18,
    with the chi2-outlier rate there dropping from 29.6% to 20.9% and no bin
    elsewhere moving by more than 0.01. Blocks this size are fine for the solve:
    _robust_cov is NMAD-based, so ~250 stars give its entries a few per cent.

    Do NOT apply the result twice. The factors are measured against the errors
    they are derived from, so a second pass re-measures already-inflated errors:
    tested on pass 3 it pulls the bright end further in (1.18 -> 1.06) but
    over-inflates the faint bulk (1.20 -> 1.29) and raises the overall outlier
    rate from 13.6% to 14.0%.

    The input table is never modified. Returns a copy of `tab` whose xe/ye
    carry the factors, or `tab` itself when apply=False -- which makes
    apply=False a dry run: verbose still prints the factors it would have
    applied.
    """
    epnames = list(tab.meta['EPNAMES'])
    times = np.asarray(tab.meta['list_times'])
    n_ep = len(epnames)
    work_cols = (np.arange(n_ep) if epoch_cols is None
                 else np.asarray(epoch_cols, dtype=int))
    valid = get_valid_idx(tab, work_cols)
    n_valid = valid.sum(1)
    mags = np.asarray(tab[mag_col])

    # Blocks: one per (detection pattern, magnitude bin). Only patterns with
    # >= 4 epochs give a usable covariance (need dof >= 2).
    # A block needs >= 4 epochs at >= 3 distinct times. Two distinct times give a
    # residual covariance too rank-poor to separate the per-epoch variances --
    # e.g. [2010 F125W, F139M, F160W, 2024 F160W] is 4 epochs but only 2 times.
    patterns = {}
    for row in np.unique(valid[n_valid >= 4], axis=0):
        cols = work_cols[np.where(row)[0]]   # row indexes work_cols, not the table
        sel = (valid == row).all(1)
        if sel.sum() >= min_stars and len(set(times[cols])) >= 3:
            patterns[tuple(cols)] = sel

    tables = {ax: {j: [] for j in range(n_ep)} for ax in 'xy'}
    for cols, sel in patterns.items():
        cols = np.array(cols)
        for lo, hi in zip(mag_bins[:-1], mag_bins[1:]):
            m = sel & (mags >= lo) & (mags < hi)
            if m.sum() < min_stars:
                continue
            for ax in 'xy':
                f = _solve_block(tab[ax][np.ix_(m, cols)],
                                 tab[ax + 'e'][np.ix_(m, cols)], times[cols])
                for k, j in enumerate(cols):
                    tables[ax][j].append((0.5 * (lo + hi), f[k], m.sum()))
            if verbose:
                print(f'  block {[epnames[c][:9] for c in cols]} '
                      f'{mag_col}=[{lo}, {hi}) N={m.sum()}')

    # Collapse repeated magnitude centres (several patterns can contribute the
    # same bin) into one N-weighted value, and keep them sorted -- np.interp
    # silently misbehaves on unsorted or duplicated x.
    curves = {}
    for ax in 'xy':
        curves[ax] = {}
        for j in range(n_ep):
            if not tables[ax][j]:
                continue
            arr = np.array(tables[ax][j])
            mc = np.unique(arr[:, 0])
            fv = np.array([np.average(arr[arr[:, 0] == c, 1],
                                      weights=arr[arr[:, 0] == c, 2]) for c in mc])
            curves[ax][j] = (mc, fv)

    if verbose:
        print('\n  inflation factors (faint -> bright):')
        for j, e in enumerate(epnames):
            if j in curves['x']:
                fx, fy = curves['x'][j][1], curves['y'][j][1]
                print(f'    {e:14s} f_x {fx.min():.2f}-{fx.max():.2f}'
                      f'   f_y {fy.min():.2f}-{fy.max():.2f}')

    tab_out = tab
    if apply:
        # Work on a copy: the caller keeps the un-inflated table, so the
        # calibration can be re-run or dropped without re-deriving xe/ye.
        tab_out = tab.copy()
        # np.interp returns nan for a nan magnitude, which would multiply that
        # star's xe/ye into nan and silently drop it from every later fit. A
        # star can legitimately reach here with no m0 (undetected, or every
        # epoch's me non-finite), so fall back to the faintest bin's factor
        # rather than poisoning the errors.
        finite_mag = np.isfinite(mags)
        if not finite_mag.all():
            print(f'  {np.sum(~finite_mag)} star(s) have no {mag_col}; '
                  f'using the faintest-bin factor for them')
        for ax in 'xy':
            for j, (mc, fv) in curves[ax].items():
                factor = np.interp(mags, mc, fv)
                factor[~finite_mag] = fv[-1]
                tab_out[ax + 'e'][:, j] *= factor
    return tab_out

def post_process_ref_table(ref_table, mosaic_epochs, epnames):
    wd1_catalog = ref_table.copy()
    wd1_catalog.meta['EPNAMES'] = epnames
    wd1_catalog['n'] = np.isfinite(wd1_catalog['x']) & np.isfinite(wd1_catalog['y'])
    wd1_position_combined = combine_mosaic_positions(wd1_catalog, mosaic_epochs)
    return wd1_position_combined

def count_matches(wd1_position_combined):
    # Test matches between F814W and 2010 F160W
    f160w_idx = wd1_position_combined.meta['EPNAMES'].index('2010_F160W')
    t1 = {wd1_position_combined.meta['EPNAMES'][i]: sum(np.isfinite(wd1_position_combined['x'][:, i])) for i in [0, f160w_idx]}
    x1 = np.ma.masked_invalid(wd1_position_combined['x'])
    print(t1)
    print(f'F814W and 2010 F160W matched stars: {sum(np.all(~x1.mask[:, [0, f160w_idx]], axis=1))}')

def all_detected(wd1_position_combined, cols):
    # Test matches between F814W and all F160W epochs
    valid_idx = get_valid_idx(wd1_position_combined, cols)
    valid_idx = np.all(valid_idx, axis=1)
    print(f'All detected in specified columns {[wd1_position_combined.meta["EPNAMES"][i] for i in cols]}: {np.sum(valid_idx)}')
    return valid_idx

def group_detected(wd1_position_combined, col_groups):
    # e.g. col_groups = [[0], [1, 2], [3, 4]]: Detected in [0] and [1 or 2] and [3 or 4]
    valid_indices = []
    for col_group in col_groups:
        valid_idx = get_valid_idx(wd1_position_combined, col_group)
        valid_idx = np.any(valid_idx, axis=1)
        valid_indices.append(valid_idx)
    valid_idx = np.all(valid_indices, axis=0)
    print(f'All detected in specified column groups {col_groups}: {np.sum(valid_idx)}')
    return valid_idx

def atleast_n_detected(wd1_position_combined, cols, n):
    # Atleast n detected in the specified columns
    valid = get_valid_idx(wd1_position_combined, cols)
    valid_count = np.sum(valid, axis=1)
    valid_idx = valid_count >= n
    print(f'At least {n} detected in specified columns {cols}: {np.sum(valid_idx)}')
    return valid_idx

def epoch_pull_stats(tab, fit_cols, model='Linear', apply_global_scale=False,
                     fixed_params_dict=None, verbose=True):
    """Leverage-corrected residual pulls per epoch, to check the error scale.

    A fitted point's residual is smaller than its measurement error by roughly
    sqrt(1 - h), where h is that point's leverage in its own star's fit. An
    epoch at the end of the time baseline can carry h ~ 0.7, so its raw pull
    looks half the size it should even when the errors are exactly right --
    which is why raw residual/error ratios cannot be read as error scales.
    Dividing by sqrt(1 - h) removes that, leaving a width of 1.0 when the
    errors are correct.

    Use this to CHECK an error scale, after flag_chi2_outliers has removed the
    non-linear movers: on pass 3 the pooled width was 1.17 with the outliers in
    and 1.01 with them out, i.e. the apparent scale error was the outlier tail
    leaking into the IQR, not the errors. Rescale only if it stays away from
    1.0 on a cleaned sample -- rescaling to cover outliers inflates every good
    star's error bars and still leaves the tail.

    The per-epoch widths are returned as a DIAGNOSTIC only, and only the pooled
    width is offered for rescaling. Solving for n per-epoch factors from n
    residual widths is underdetermined -- r = (I-H)eps has rank n-2 -- and runs
    away, deflating one epoch to pay for another; that is exactly what
    calibrate_epoch_errors' covariance matching exists to avoid, and it should
    be used for per-epoch work. One pooled factor is a single unknown fitted to
    n equations, so it does not suffer that.

    Measured on `model` stars detected in every one of fit_cols, so the design
    matrix is the same for all of them.

    Returns {'per_epoch': {axis: {col: width}}, 'pooled': float,
             'frac_gt_3sigma': float}. With apply_global_scale=True every
    fit_cols epoch's xe/ye is multiplied by the pooled width in place.
    """
    epnames = list(tab.meta['EPNAMES'])
    times = np.asarray(tab.meta['list_times'])
    cols = np.asarray(fit_cols, dtype=int)
    models_used = np.asarray(tab['motion_model_used']).astype(str)

    valid = get_valid_idx(tab, cols)
    sel = (models_used == model) & valid.all(axis=1)
    if sel.sum() < 50:
        raise ValueError(f'epoch_pull_stats: only {sel.sum()} {model} stars have all '
                         f'{len(cols)} epochs; too few to measure a pull width.')

    t0 = np.asarray(tab['t0'])[sel]
    dt = times[cols][None, :] - t0[:, None]
    out = {'per_epoch': {}}
    pooled = []
    for ax in ('x', 'y'):
        pos = np.asarray(tab[ax])[np.ix_(sel, cols)]
        err = np.asarray(tab[ax + 'e'])[np.ix_(sel, cols)]
        p0 = np.asarray(tab[ax + '0'])[sel]
        v = np.asarray(tab['v' + ax])[sel]
        resid = pos - (p0[:, None] + v[:, None] * dt)

        # Leverage of a per-star weighted straight-line fit:
        #   h_i = w_i (S2 - 2 dt_i S1 + dt_i^2 S0) / (S0 S2 - S1^2)
        w = 1.0 / err**2
        S0, S1, S2 = w.sum(1), (w * dt).sum(1), (w * dt**2).sum(1)
        h = w * (S2[:, None] - 2 * dt * S1[:, None] + dt**2 * S0[:, None]) / (S0 * S2 - S1**2)[:, None]
        pull = resid / (err * np.sqrt(np.clip(1.0 - h, 1e-6, None)))

        widths = {}
        for k, j in enumerate(cols):
            widths[int(j)] = float(_robust_width(pull[:, k]))
        out['per_epoch'][ax] = widths
        pooled.append(pull.ravel())

    allpull = np.concatenate(pooled)
    allpull = allpull[np.isfinite(allpull)]
    out['pooled'] = float(_robust_width(allpull))
    out['frac_gt_3sigma'] = float(np.mean(np.abs(allpull) > 3))
    out['n_stars'] = int(sel.sum())

    if verbose:
        print(f'epoch_pull_stats: {out["n_stars"]} {model} stars with all {len(cols)} epochs')
        print(f'  {"epoch":<14}{"leverage":>10}{"pull x":>9}{"pull y":>9}')
        for k, j in enumerate(cols):
            print(f'  {epnames[j]:<14}{np.median(h[:, k]):>10.2f}'
                  f'{out["per_epoch"]["x"][int(j)]:>9.2f}{out["per_epoch"]["y"][int(j)]:>9.2f}')
        print(f'  pooled width {out["pooled"]:.3f} (1.0 = errors correct); '
              f'{100*out["frac_gt_3sigma"]:.1f}% of points beyond 3 sigma (0.3% if Gaussian)')

    if apply_global_scale:
        for ax in ('x', 'y'):
            tab[ax + 'e'][:, cols] *= out['pooled']
        if verbose:
            print(f'  applied x{out["pooled"]:.3f} to xe/ye of {len(cols)} epochs')
    return out


def _robust_width(v):
    """IQR-based sigma, so an outlier tail does not set the width."""
    v = np.asarray(v)
    v = v[np.isfinite(v)]
    return 0.7413 * (np.percentile(v, 75) - np.percentile(v, 25))


def flag_chi2_outliers(tab, fit_cols, p=0.99, verbose=True):
    """Stars whose fit residuals are too large for their own dof to be noise.

    An error rescaling cannot remove these -- it slides the whole chi2
    distribution rather than trimming a tail -- so they are excluded from the
    science sample instead. These are stars the motion model does not describe:
    blends, unresolved binaries, mismatches, real curvature.

    dof follows the model: n_valid - n_params per axis, except Parallax, whose
    chi2 is joint over both axes with pi shared (dof = 2*n_valid - 5).

    Returns a boolean array, True where the star is an outlier.
    """
    cols = np.asarray(fit_cols, dtype=int)
    n_valid = get_valid_idx(tab, cols).sum(axis=1)
    models_used = np.asarray(tab['motion_model_used']).astype(str)
    c2x, c2y = np.asarray(tab['chi2_x']), np.asarray(tab['chi2_y'])
    n_params = np.asarray(tab['n_params'])

    bad = np.zeros(len(tab), dtype=bool)
    is_par = models_used == 'Parallax'
    dof_joint = 2 * n_valid - 5
    ok = is_par & (dof_joint > 0)
    bad[ok] = (c2x[ok] + c2y[ok]) > chi2.ppf(p, dof_joint[ok])

    dof = n_valid - n_params
    ok = (~is_par) & (dof > 0)
    bad[ok] = (c2x[ok] > chi2.ppf(p, dof[ok])) | (c2y[ok] > chi2.ppf(p, dof[ok]))

    if verbose:
        exp = 2 * (1 - p) * 100
        print(f'flag_chi2_outliers: {int(bad.sum())} of {len(tab)} stars '
              f'({100*bad.mean():.1f}%) above the p={p} threshold '
              f'({exp:.1f}% expected if the model held for every star)')
    return bad


def select_analysis_sample(tab, analysis_epochs):
    """Stars detected in every one of analysis_epochs.

    Selection only: which epochs the motion fit was allowed to USE is a
    separate question (fit_motion_models' mask_lists), and the two need not
    agree -- an epoch can be a bonus measurement for the stars that have it
    without being required of every star.

    Returns the trimmed table and the boolean index, the latter because
    downstream steps (kinematic membership) map their own indices back
    through it.
    """
    epnames = list(tab.meta['EPNAMES'])   # ndarray when read back from HDF5
    epoch_cols = [epnames.index(ep) for ep in analysis_epochs]
    valid_idx = all_detected(tab, epoch_cols)
    return tab[valid_idx], valid_idx


def plot_chi2_by_dof(tab, fit_cols=None, min_stars=50, outlier_p=None):
    """chi2 distributions grouped by how many measurements entered each fit.

    fit_cols must be the epochs that actually reached fit_motion_models, i.e.
    everything NOT in its mask_lists (default: all of them). Counting epochs
    the fit never saw would overstate dof.

    Group by the measurement count, NOT the distinct-time count: run_fit forms
    dof = n_valid - n_params from the number of unmasked epochs, while
    tab['n_fit'] counts unique times. The three 2010 filters (F125W/F139M/
    F160W) all sit at t=2010.652, so whenever they are in the fit n_valid runs
    above n_fit and grouping on n_fit would use too few degrees of freedom.

    Parallax is split out because its chi2 is joint over both axes
    (dof = 2*n_valid - n_fit_params, since pi is shared) rather than per-axis.

    outlier_p, if given, is the p that flag_chi2_outliers was run at; it is
    handed to the plot, which cuts there per axis. Pass the table the flag was
    computed ON, un-filtered -- filtering it first would take the cut twice and
    hide how much the plot is missing.
    """
    cols = (np.arange(len(tab.meta['EPNAMES'])) if fit_cols is None
            else np.asarray(fit_cols, dtype=int))
    n_valid = get_valid_idx(tab, cols).sum(axis=1)

    # A table read back from HDF5/FITS returns this column as bytes (b'Linear'),
    # which matches no str literal -- the Empty/Fixed skip and the Parallax test
    # below would both silently fail. StarTable.read normalizes it now; this
    # keeps the helper working on a table loaded any other way.
    models_used = np.asarray(tab['motion_model_used']).astype(str)

    for model in np.unique(models_used):
        if model in ('Empty', 'Fixed'):
            continue
        for n_meas in np.unique(n_valid):
            in_group = (n_valid == n_meas) & (models_used == model)
            if in_group.sum() < min_stars:
                continue
            n_params = int(tab['n_params'][in_group][0])
            if model == 'Parallax':
                dof = 2 * int(n_meas) - 5
                plot_fn = plot_chi2_dist_parallax
            else:
                dof = int(n_meas) - n_params
                plot_fn = plot_chi2_dist
            if dof < 1:
                continue
            # outlier_p is applied by the plot itself, per axis and without
            # renormalizing the bars, so pass the group un-trimmed.
            plot_fn(
                tab[in_group], chi2_lim=max(10, 4 * dof), bins=40, log=False,
                n_epoch=n_meas, dof=dof,
                truncate_at=(chi2.ppf(outlier_p, dof) if outlier_p else None)
            )


def atleast_n_group_detected(wd1_position_combined, col_groups, n):
     # e.g. col_groups = [[0], [1, 2], [3, 4]]: Detected in at least n epochs in [0], [1 or 2], [3 or 4]
    valid_indices = []
    for col_group in col_groups:
        valid_idx = get_valid_idx(wd1_position_combined, col_group)
        valid_idx = np.any(valid_idx, axis=1)
        valid_indices.append(valid_idx)
    valid_indices = np.array(valid_indices)
    valid_count = np.sum(valid_indices, axis=0)
    valid_idx = valid_count >= n
    print(f'At least {n} detected in specified column groups {col_groups}: {np.sum(valid_idx)}')
    return valid_idx

def calculate_m0(ref_table, list_indices, mag_col='m', mag_err_col='me'):
    # Adapted from flystar.startables.StarTable.combine_lists
    mags = np.ma.masked_invalid(ref_table[mag_col][:, list_indices])
    mag_errs = np.ma.masked_invalid(ref_table[mag_err_col][:, list_indices])
    fluxes = 10**(-0.4 * mags)
    flux_errs = 0.4 * np.log(10) * fluxes * mag_errs
    unified_mask = fluxes.mask | flux_errs.mask
    fluxes.mask = unified_mask
    flux_errs.mask = unified_mask
    weights = np.ma.masked_invalid(1. / flux_errs**2)
    fluxes_mean = np.ma.average(fluxes, weights=weights, axis=1)
    fluxes_std = np.ma.sqrt(1 / np.ma.sum(weights, axis=1))
    # Convert back to magnitudes
    m0 = -2.5 * np.ma.log10(fluxes_mean)
    m0_err = 2.5 / np.log(10) * fluxes_std / fluxes_mean

    # A star is masked here when every one of its detections in list_indices is
    # unusable: no magnitude, or no usable magnitude error (nan, inf, or -- in
    # lists written before convert_nimfo2bar_to_align learned to write inf -- a
    # 0 that becomes an infinite weight). Hand that back as plain arrays in the
    # convention the rest of the pipeline uses, nan for a missing value and inf
    # for an unknown error, rather than as masked arrays: assigning a masked
    # array into a StarTable makes a MaskedColumn, and np.isfinite() on one
    # returns a masked array that every downstream test then has to unwrap.
    m0 = np.ma.filled(m0, np.nan)
    m0_err = np.ma.filled(m0_err, np.inf)
    m0_err[~np.isfinite(m0_err)] = np.inf
    return m0, m0_err


def mag_lim_per_list(n_iters, n_lists, default=(0, 22), overrides=None):
    """Build align's mag_lim in its (N_iters, N_lists, 2) per-list form.

    align applies mag_lim to the RAW starlist magnitudes, before any magnitude
    transformation, so a single limit means different things in different
    filters: in a reddened field an optical list can sit several magnitudes
    fainter than an IR one for the same stars, and a flat cut then selects
    transformation anchors by colour. `overrides` gives those lists their own
    limit.

    Only the stars used to DERIVE each transformation are affected -- the final
    match that fills ref_table always uses the untrimmed list.

    Parameters
    ----------
    n_iters : int
        Number of alignment iterations (len(trans_args) where that is a list).
    n_lists : int
        Number of starlists.
    default : (min, max), optional
        Limit for every list without an override, by default (0, 22).
    overrides : sequence of (indices, (min, max)), optional
        Per-list limits, e.g. [(F814W_2005_idxs, (0, 25))]. Indices may be a
        scalar or an array. By default None.
    """
    lim = np.tile(np.asarray(default, dtype=float), (n_iters, n_lists, 1))
    for idxs, val in (overrides or []):
        lim[:, np.atleast_1d(idxs), :] = np.asarray(val, dtype=float)
    return lim


def prepare_reflist(ref_table, list_idxs, detect_idxs=None, mag_idxs=None, mag_lim=None):
    """
    detect_idxs : list of index arrays, optional
        A star is used in the transformation only if it is detected in at
        least one list of each group, e.g.
        [F160W_2010_idxs, F160W_2013_idxs, F160W_2015_idxs, F160W_2024_idxs].
        If None, all stars are used.
    mag_idxs, mag_lim : optional
        Additionally require mean m_orig over mag_idxs to be <= mag_lim.
    """

    # list_idxs names the lists to use; everything else is masked out of the
    # motion fit. The two are complements by construction -- calculate_m0 takes
    # the lists to INCLUDE and fit_motion_models the lists to EXCLUDE -- so the
    # mask is derived rather than passed. Taking the list count from the table
    # also means it tracks the table itself: a list added to the alignment is
    # masked automatically instead of waiting for a hand-maintained index array
    # to be regenerated, which would otherwise leave it silently inside the
    # astrometric fit.
    mask_idxs = np.setdiff1d(np.arange(ref_table['x'].shape[1]), list_idxs)

    ref_list = ref_table.copy()
    m0, m0_err = calculate_m0(ref_list, list_indices=list_idxs)
    ref_list['m0'] = m0
    ref_list['m0_err'] = m0_err

    ref_list.fit_motion_models(
        motion_models=['Linear'],
        mask_lists=mask_idxs
    )
    ref_list = ref_list[
        'name', 't0',
        'm0', 'm0_err', 
        'x0', 'x0_err', 'vx', 'vx_err', 
        'y0', 'y0_err', 'vy', 'vy_err', 
        'use_in_trans'
    ]

    if detect_idxs is None:
        good_ref_stars = np.ones(len(ref_list), dtype=bool)
    else:
        good_ref_stars = np.logical_and.reduce([
            (
                np.isfinite(ref_table['x'][:, idxs])
                & np.isfinite(ref_table['y'][:, idxs])
                & np.isfinite(ref_table['m'][:, idxs])
            ).any(axis=1) for idxs in detect_idxs
        ])
    if mag_lim is not None:
        # Stars undetected in every mag_idxs list give an all-NaN slice; keep
        # np.nanmean quiet about it and drop them explicitly below.
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', RuntimeWarning)
            mags = np.nanmean(ref_table['m_orig'][:, mag_idxs], axis=1)
        good_ref_stars &= np.isfinite(mags) & (mags <= mag_lim)

    # Stars with no valid epoch left after mask_lists come out of the fit with
    # NaN x0/y0, which is exactly flystar's 'Empty' condition: every motion
    # model needs x0 and y0, and Fixed needs only those, so finite x0/y0 means
    # some model can still place the star. Empty rows propagate as NaN, match
    # nothing, and would only ride along as blank rows -- their real detections
    # get re-added as new stars by add_rows_for_new_stars anyway. A star whose
    # Linear fit failed for want of a second epoch keeps finite x0/y0 with NaN
    # vx/vy and is deliberately KEPT: it propagates as Fixed, held at its
    # measured position. Subset good_ref_stars to stay aligned with the rows
    # that survive.
    # m0 is in the test because the matching uses magnitudes (mag_trans /
    # dm_tol): a row with no m0 gives dm = nan and never matches, so it is dead
    # weight in exactly the same way. m0 goes nan not only when a star has no
    # magnitude, but also when every one of its detections has me == 0 (a
    # single-frame KS2 measurement has no scatter to quote), which masks out
    # the inverse-variance weights in calculate_m0. Drop this requirement if
    # the list is ever matched on position alone.
    keep = (np.isfinite(np.ma.filled(ref_list['x0'], np.nan))
            & np.isfinite(np.ma.filled(ref_list['y0'], np.nan))
            & np.isfinite(np.ma.filled(ref_list['m0'], np.nan)))
    n_drop = int(np.sum(~keep))
    if n_drop > 0:
        print(f'Dropping {n_drop} unmatchable stars (no fitted position, or no usable magnitude)')
    ref_list = ref_list[keep]
    good_ref_stars = good_ref_stars[keep]

    print(f'Number of good reference stars: {np.sum(good_ref_stars)}')
    ref_list['use_in_trans'] = False
    ref_list['use_in_trans'][good_ref_stars] = True

    return ref_list

def prepare_reflist_kinematic_members(ref_table, members, list_idxs, mag_idxs,
                                      motion_models=['Linear'], mag_lim=18):
    # list_idxs names the lists to use; everything else is masked out of the
    # motion fit. The two are complements by construction -- calculate_m0 takes
    # the lists to INCLUDE and fit_motion_models the lists to EXCLUDE -- so the
    # mask is derived rather than passed. Taking the list count from the table
    # also means it tracks the table itself: a list added to the alignment is
    # masked automatically instead of waiting for a hand-maintained index array
    # to be regenerated, which would otherwise leave it silently inside the
    # astrometric fit.
    mask_idxs = np.setdiff1d(np.arange(ref_table['x'].shape[1]), list_idxs)

    ref_list = ref_table.copy()
    m0, m0_err = calculate_m0(ref_list, list_indices=list_idxs)
    ref_list['m0'] = m0
    ref_list['m0_err'] = m0_err

    ref_list.fit_motion_models(
        motion_models=motion_models,
        mask_lists=mask_idxs
    )
    ref_list = ref_list[
        'name', 't0',
        'm0', 'm0_err', 
        'x0', 'x0_err', 'vx', 'vx_err', 
        'y0', 'y0_err', 'vy', 'vy_err', 
        'use_in_trans'
    ]

    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        mags = np.nanmean(ref_table['m_orig'][:, mag_idxs], axis=1)
    is_member = np.full(len(ref_list), False, dtype=bool)
    is_member[members] = True
    good_ref_stars = np.isfinite(mags) & (mags <= mag_lim) & is_member

    # Stars with no valid epoch left after mask_lists come out of the fit with
    # NaN x0/y0, which is exactly flystar's 'Empty' condition: every motion
    # model needs x0 and y0, and Fixed needs only those, so finite x0/y0 means
    # some model can still place the star. Empty rows propagate as NaN, match
    # nothing, and would only ride along as blank rows -- their real detections
    # get re-added as new stars by add_rows_for_new_stars anyway. A star whose
    # Linear fit failed for want of a second epoch keeps finite x0/y0 with NaN
    # vx/vy and is deliberately KEPT: it propagates as Fixed, held at its
    # measured position. Subset good_ref_stars to stay aligned with the rows
    # that survive.
    # m0 is in the test because the matching uses magnitudes (mag_trans /
    # dm_tol): a row with no m0 gives dm = nan and never matches, so it is dead
    # weight in exactly the same way. m0 goes nan not only when a star has no
    # magnitude, but also when every one of its detections has me == 0 (a
    # single-frame KS2 measurement has no scatter to quote), which masks out
    # the inverse-variance weights in calculate_m0. Drop this requirement if
    # the list is ever matched on position alone.
    keep = (np.isfinite(np.ma.filled(ref_list['x0'], np.nan))
            & np.isfinite(np.ma.filled(ref_list['y0'], np.nan))
            & np.isfinite(np.ma.filled(ref_list['m0'], np.nan)))
    n_drop = int(np.sum(~keep))
    if n_drop > 0:
        print(f'Dropping {n_drop} unmatchable stars (no fitted position, or no usable magnitude)')
    ref_list = ref_list[keep]
    good_ref_stars = good_ref_stars[keep]

    ref_list['use_in_trans'] = False
    ref_list['use_in_trans'][good_ref_stars] = True
    print(f'Number of good reference stars: {np.sum(good_ref_stars)}')

    return ref_list

def pm_uncertainty_vs_mag(ref_table):
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(ref_table['m0'], ref_table['vx_err'], s=1, edgecolors='none', alpha=0.5, label='vx_err')
    ax.scatter(ref_table['m0'], ref_table['vy_err'], s=1, edgecolors='none', alpha=0.5, label='vy_err')
    ax.set_yscale('log')
    ax.set_xlabel('Magnitude')
    ax.set_ylabel('Proper Motion Uncertainty')
    ax.legend()
    plt.show()

def plot_chi2(wd1_all_epochs, epoch_cols):
    times = np.floor(wd1_all_epochs.meta['list_times'])[epoch_cols]
    valid_idx = get_valid_idx(wd1_all_epochs, epoch_cols)
    star_times = np.ma.masked_array(
        np.broadcast_to(times, valid_idx.shape), mask=~valid_idx
    )
    n_unique = np.array([
        len(set(star_time.compressed())) for star_time in star_times
    ])
    for n in np.unique(n_unique):
        subset = wd1_all_epochs[n_unique == n]
        dof = n - 2
        if dof == 0:
            continue
        plot_chi2_dist(subset, chi2_lim=10, bins=40, log=False, n_epoch=n, dof=dof)
    return

def plot_chi2_parallax(wd1_all_epochs, epoch_cols):
    times = np.floor(wd1_all_epochs.meta['list_times'])[epoch_cols]
    valid_idx = get_valid_idx(wd1_all_epochs, epoch_cols)
    star_times = np.ma.masked_array(
        np.broadcast_to(times, valid_idx.shape), mask=~valid_idx
    )
    n_unique = np.array([
        len(set(star_time.compressed())) for star_time in star_times
    ])
    for n in np.unique(n_unique):
        subset = wd1_all_epochs[n_unique == n]
        dof = 2*n - 5
        if dof == 0:
            continue
        plot_chi2_dist(subset, chi2_lim=10, bins=40, log=False, n_epoch=n, dof=dof)
    return

def individual_stars(ref_table, n_select=10, cols='all', seed=42):
    n_stars = len(ref_table)
    # Randomly select n_select stars
    rng = np.random.default_rng(seed)
    selected_indices = rng.choice(n_stars, size=n_select, replace=False)
    # Each star 3*3 block of subplots
    nrows = 3*((n_select + 1) // 2)
    ncols = 6
    times = np.array(ref_table.meta['list_times'])
    n_times = len(times)
    if cols == 'all':
        cols = np.arange(n_times)
    else:
        cols = np.array(cols)

    times_dense = np.linspace(times[cols].min(), times[cols].max(), 200)

    fig, axes = plt.subplots(nrows, ncols, figsize=(15, 10*n_select//2))
    for i, idx in enumerate(selected_indices):
        single_table = ref_table[[idx]]
        x_inferred, y_inferred, xe_inferred, ye_inferred = single_table.infer_positions(times_dense)
        x_compare, y_compare, xe_compare, ye_compare = single_table.infer_positions(times[cols])
        x_residual = ref_table['x'][idx, cols] - x_compare
        y_residual = ref_table['y'][idx, cols] - y_compare
        m_residual = ref_table['m'][idx, cols] - ref_table['m0'][idx]

        row = (i // 2) * 3
        col = (i % 2) * 3

        # x vs time
        axes[row, col].errorbar(times[cols], ref_table['x'][idx, cols], yerr=ref_table['xe'][idx, cols], fmt='o', color='C0')
        axes[row, col].plot(times_dense, x_inferred, color='C3')
        axes[row, col].plot(times_dense, x_inferred - xe_inferred, color='C3', linestyle='--')
        axes[row, col].plot(times_dense, x_inferred + xe_inferred, color='C3', linestyle='--')
        axes[row, col].fill_between(times_dense, x_inferred - xe_inferred, x_inferred + xe_inferred, color='C3', alpha=0.2)
        axes[row, col].set_xlabel('Time')
        axes[row, col].set_ylabel('x (pixels)')

        # y vs time
        axes[row, col + 1].errorbar(times[cols], ref_table['y'][idx, cols], yerr=ref_table['ye'][idx, cols], fmt='o', color='C0')
        axes[row, col + 1].plot(times_dense, y_inferred, color='C3')
        axes[row, col + 1].plot(times_dense, y_inferred - ye_inferred, color='C3', linestyle='--')
        axes[row, col + 1].plot(times_dense, y_inferred + ye_inferred, color='C3', linestyle='--')
        axes[row, col + 1].fill_between(times_dense, y_inferred - ye_inferred, y_inferred + ye_inferred, color='C3', alpha=0.2)
        axes[row, col + 1].set_xlabel('Time')
        axes[row, col + 1].set_ylabel('y (pixels)')
        axes[row, col + 1].set_title(f"Star {ref_table['name'][idx]}, idx={idx}")

        # m vs time
        axes[row, col + 2].errorbar(times[cols], ref_table['m'][idx, cols], yerr=ref_table['me'][idx, cols], fmt='o', color='C0')
        axes[row, col + 2].axhline(ref_table['m0'][idx], color='C3')
        axes[row, col + 2].plot(times[cols], ref_table['m0'][idx] - ref_table['me'][idx, cols], color='C3', linestyle='--')
        axes[row, col + 2].plot(times[cols], ref_table['m0'][idx] + ref_table['me'][idx, cols], color='C3', linestyle='--')
        axes[row, col + 2].fill_between(times[cols], ref_table['m0'][idx] - ref_table['me'][idx, cols], ref_table['m0'][idx] + ref_table['me'][idx, cols], color='C3', alpha=0.2)
        axes[row, col + 2].set_xlabel('Time')
        axes[row, col + 2].set_ylabel('Magnitude')

        # x residual vs time
        axes[row + 1, col].errorbar(times[cols], x_residual, yerr=ref_table['xe'][idx, cols], fmt='o', color='C0')
        axes[row + 1, col].axhline(0, color='C3')
        axes[row + 1, col].plot(times_dense, -xe_inferred, color='C3', linestyle='--')
        axes[row + 1, col].plot(times_dense, xe_inferred, color='C3', linestyle='--')
        axes[row + 1, col].fill_between(times_dense, -xe_inferred, xe_inferred, color='C3', alpha=0.2)
        axes[row + 1, col].set_xlabel('Time')
        axes[row + 1, col].set_ylabel('x Residual (pixels)')

        # y residual vs time
        axes[row + 1, col + 1].errorbar(times[cols], y_residual, yerr=ref_table['ye'][idx, cols], fmt='o', color='C0')
        axes[row + 1, col + 1].axhline(0, color='C3')
        axes[row + 1, col + 1].plot(times_dense, -ye_inferred, color='C3', linestyle='--')
        axes[row + 1, col + 1].plot(times_dense, ye_inferred, color='C3', linestyle='--')
        axes[row + 1, col + 1].fill_between(times_dense, -ye_inferred, ye_inferred, color='C3', alpha=0.2)
        axes[row + 1, col + 1].set_xlabel('Time')
        axes[row + 1, col + 1].set_ylabel('y Residual (pixels)')

        # m residual vs time
        axes[row + 1, col + 2].errorbar(times[cols], m_residual, yerr=ref_table['me'][idx, cols], fmt='o', color='C0')
        axes[row + 1, col + 2].axhline(0, color='C3')
        axes[row + 1, col + 2].plot(times[cols], -ref_table['me'][idx, cols], color='C3', linestyle='--')
        axes[row + 1, col + 2].plot(times[cols], ref_table['me'][idx, cols], color='C3', linestyle='--')
        axes[row + 1, col + 2].fill_between(times[cols], -ref_table['me'][idx, cols], ref_table['me'][idx, cols], color='C3', alpha=0.2)
        axes[row + 1, col + 2].set_xlabel('Time')
        axes[row + 1, col + 2].set_ylabel('m Residual (pixels)')

        # x vs y
        axes[row + 2, col].errorbar(ref_table['x'][idx, cols], ref_table['y'][idx, cols], xerr=ref_table['xe'][idx, cols], yerr=ref_table['ye'][idx, cols], fmt='o', color='C0')
        axes[row + 2, col].plot(x_inferred, y_inferred, color='C3')
        axes[row + 2, col].set_xlabel('x (pixels)')
        axes[row + 2, col].set_ylabel('y (pixels)')
        
        # x y histogram of residuals
        sigma_x = x_residual / ref_table['xe'][idx, cols]
        sigma_y = y_residual / ref_table['ye'][idx, cols]
        axes[row + 2, col + 1].hist(sigma_x, bins=10, range=(-5, 5), color='C0', alpha=0.5, label='X')
        axes[row + 2, col + 1].hist(sigma_y, bins=10, range=(-5, 5), color='C1', alpha=0.5, label='Y')
        axes[row + 2, col + 1].set_xlabel('Residual / Uncertainty')
        axes[row + 2, col + 1].set_ylabel('Number of Epochs')
        axes[row + 2, col + 1].legend()
    
        # mag histogram of residuals
        sigma_m = m_residual / ref_table['me'][idx, cols]
        axes[row + 2, col + 2].hist(sigma_m, bins=10, range=(-5, 5), alpha=0.5, color='C0')
        axes[row + 2, col + 2].set_xlabel('Residual / Uncertainty')
        axes[row + 2, col + 2].set_ylabel('Number of Epochs')

    plt.tight_layout()
    plt.show()

def plot_residuals_per_epoch(tab, log=False):
    times = np.array(tab.meta['list_times'])
    epnames = tab.meta['EPNAMES']
    n_times = len(times)
    inferred_x, inferred_y, inferred_xe, inferred_ye = tab.infer_positions(times)
    residuals_x = tab['x'] - inferred_x
    residuals_y = tab['y'] - inferred_y
    chi2_x = (residuals_x / tab['xe'])**2
    chi2_y = (residuals_y / tab['ye'])**2
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 5))
    ax1.boxplot([chi2_x[:, i][np.isfinite(chi2_x[:, i])] for i in range(n_times)], tick_labels=epnames, showfliers=False)
    if log:
        ax1.set_yscale('log')
    ax1.set_ylabel('Chi2 X')
    ax1.tick_params(axis='x', labelrotation=30)
    ax2.boxplot([chi2_y[:, i][np.isfinite(chi2_y[:, i])] for i in range(n_times)], tick_labels=epnames, showfliers=False)
    if log:
        ax2.set_yscale('log')
    ax2.set_ylabel('Chi2 Y')
    ax2.tick_params(axis='x', labelrotation=30)
    plt.tight_layout()
    plt.show()

def combine_short_exposures(tab):
    """
    Merge each '*_F160Ws' short/safety-exposure epoch into its matching
    '*_F160W' position-combined epoch (inverse-variance-weighted for x/y,
    flux-space-weighted for magnitude -- same scheme as
    combine_mosaic_positions), then drop the now-redundant F160Ws column.

    Pairs are found by name via tab.meta['EPNAMES'] rather than a fixed
    column order, so this isn't tied to the current column layout: it picks
    up however many '*_F160Ws' epochs are actually present (currently 2013
    and 2015; a future 2024_F160Ws would be picked up automatically once
    it's added to the input table). All other epochs/columns (F814W, F125W,
    F139M, ...) are left untouched.
    """
    tab = copy.deepcopy(tab)
    epnames = list(tab.meta['EPNAMES'])

    short_exp_pairs = []
    for short_idx, ep in enumerate(epnames):
        if not ep.endswith('F160Ws'):
            continue
        long_name = ep[:-1]  # 'YYYY_F160Ws' -> 'YYYY_F160W'
        if long_name not in epnames:
            raise ValueError(f"combine_short_exposures: no '{long_name}' column to merge '{ep}' into")
        short_exp_pairs.append((epnames.index(long_name), short_idx))

    with np.errstate(invalid='ignore', divide='ignore'):
        for long_idx, short_idx in short_exp_pairs:
            x = np.ma.masked_invalid(tab['x'][:, [long_idx, short_idx]])
            y = np.ma.masked_invalid(tab['y'][:, [long_idx, short_idx]])
            m = np.ma.masked_invalid(tab['m'][:, [long_idx, short_idx]])
            xe = np.ma.masked_invalid(tab['xe'][:, [long_idx, short_idx]])
            ye = np.ma.masked_invalid(tab['ye'][:, [long_idx, short_idx]])
            me = np.ma.masked_invalid(tab['me'][:, [long_idx, short_idx]])

            tab['x'][:, long_idx] = np.ma.average(
                x, axis=1, weights=1/xe**2
            ).filled(np.nan)
            tab['y'][:, long_idx] = np.ma.average(
                y, axis=1, weights=1/ye**2
            ).filled(np.nan)
            tab['xe'][:, long_idx] = np.ma.sqrt(1/np.ma.sum(1/xe**2, axis=1)).filled(np.inf)
            tab['ye'][:, long_idx] = np.ma.sqrt(1/np.ma.sum(1/ye**2, axis=1)).filled(np.inf)

            flux = 10**(-0.4 * m)
            flux_err = 0.4 * np.log(10) * flux * me
            combined_flux = np.ma.average(flux, axis=1, weights=1/flux_err**2)
            combined_flux_err = np.ma.sqrt(1 / np.ma.sum(1/flux_err**2, axis=1))
            tab['m'][:, long_idx] = (-2.5 * np.ma.log10(combined_flux)).filled(np.nan)
            me_combined = (2.5 / np.log(10) * combined_flux_err / combined_flux).filled(np.inf)
            tab['me'][:, long_idx] = np.where(np.isfinite(tab['m'][:, long_idx]), me_combined, np.inf)

            if 'n' in tab.colnames:
                tab['n'][:, long_idx] += tab['n'][:, short_idx]

    # Drop the now-redundant F160Ws columns. All 2D per-epoch columns need
    # to shrink together, or they'd disagree with each other and with the
    # trimmed EPNAMES on which column index means which epoch.
    drop_idx = [short_idx for _, short_idx in short_exp_pairs]
    epoch_cols = [col for col in ('x', 'y', 'm', 'm_orig', 'xe', 'ye', 'me', 'n') if col in tab.colnames]
    for col in epoch_cols:
        tab[col] = np.delete(tab[col], drop_idx, axis=1)
    if 'list_times' in tab.meta:
        tab.meta['list_times'] = np.delete(np.asarray(tab.meta['list_times']), drop_idx).tolist()
    tab.meta['EPNAMES'] = [ep for i, ep in enumerate(epnames) if i not in drop_idx]

    return tab