"""
Research Note: Proper Motion Test (2014-06)
Working Directory: /u/jlu/data/Wd1/hst/reduce_2014_06_17/
"""
import math
import pylab as py
import numpy as np
from hst_flystar import reduce as flystar
from jlu import photometry as photo
import glob
from matplotlib import colors
from jlu.util import statsIter
from jlu.util import fileUtil
import pdb
import os
import shutil
import subprocess
from hst_flystar import astrometry as ast
from hst_flystar import photometry
from hst_flystar import reduce as flystar
from hst_flystar import starlists
from hst_flystar import completeness as comp
import astropy.table
from astropy import table
from astropy.table import Table
from astropy.table import Column
from jlu.astrometry import align
from jlu.gc.gcwork import starset
from scipy.stats import binned_statistic
from matplotlib import colors as mcolors
from matplotlib import colorbar

work_dir = '/u/jlu/data/wd1/hst/reduce_2015_01_05/'
code_dir = '/u/jlu/code/fortran/hst/'

# Load this variable with outputs from calc_years()
years = {'2005_F814W': 2005.485,
         '2010_F125W': 2010.652,
         '2010_F139M': 2010.652,
         '2010_F160W': 2010.652,
         '2013_F160W': 2013.199,
         '2013_F160Ws': 2013.202}

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
    psf = code_dir + 'PSFs/PSFSTD_ACSWFC_F814W_4SM3.fits'

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

        t2005_814.write('MATCHUP.XYMEEE.F814W.2005.ref5.fits', overwrite=True)
        t2010_160.write('MATCHUP.XYMEEE.F160W.2010.ref5.fits', overwrite=True)
        t2010_139.write('MATCHUP.XYMEEE.F139M.2010.ref5.fits', overwrite=True)
        t2010_125.write('MATCHUP.XYMEEE.F125W.2010.ref5.fits', overwrite=True)
        t2013_160.write('MATCHUP.XYMEEE.F160W.2013.ref5.fits', overwrite=True)
        t2013_160s.write('MATCHUP.XYMEEE.F160Ws.2013.ref5.fits', overwrite=True)
    else:
        # Read in the FITS versions of the tables.
        t2005_814 = Table.read('MATCHUP.XYMEEE.F814W.2005.ref5.fits')
        t2010_160 = Table.read('MATCHUP.XYMEEE.F160W.2010.ref5.fits')
        t2010_139 = Table.read('MATCHUP.XYMEEE.F139M.2010.ref5.fits')
        t2010_125 = Table.read('MATCHUP.XYMEEE.F125W.2010.ref5.fits')
        t2013_160 = Table.read('MATCHUP.XYMEEE.F160W.2013.ref5.fits')
        t2013_160s = Table.read('MATCHUP.XYMEEE.F160Ws.2013.ref5.fits')

    # Trim down to only those stars with well measured positions. 
    good = np.where((t2005_814['xe'] < 0.1)  & (t2005_814['ye'] < 0.1) &
                    (t2010_160['xe'] < 0.1)  & (t2010_160['ye'] < 0.1) & 
                    (t2010_139['xe'] < 0.1)  & (t2010_139['ye'] < 0.1) & 
                    (t2010_125['xe'] < 0.1)  & (t2010_125['ye'] < 0.1) & 
                    (t2013_160['xe'] < 0.1)  & (t2013_160['ye'] < 0.1) & 
                    (t2013_160s['xe'] < 0.1) & (t2013_160s['ye'] < 0.1))[0]

    t2005_814 = t2005_814[good]
    t2010_160 = t2010_160[good]
    t2010_139 = t2010_139[good]
    t2010_125 = t2010_125[good]
    t2013_160 = t2013_160[good]
    t2013_160s = t2013_160s[good]

    # Put all the tables in a list so that we can loop through
    # the different combinations.
    t_all = [t2005_814, t2010_125, t2010_139, t2010_160, t2013_160, t2013_160s]
    label = ['F814W 2005', 'F125W 2010', 'F139M 2010', 'F160W 2010',
             'F160W 2013', 'F160Ws 2013']
        
    ##########
    # CMDs
    ##########
    for ii in range(len(t_all)):
        for jj in range(ii+1, len(t_all)):
            t1 = t_all[ii]
            t2 = t_all[jj]
            
            py.clf()
            py.plot(t1['m'] - t2['m'], t1['m'], 'k.')
            py.xlabel(label[ii] + ' - ' + label[jj])
            py.ylabel(label[ii])
            ax = py.gca()
            ax.invert_yaxis()

            outfile = 'cmd_'
            outfile += label[ii].replace(' ', '_')
            outfile += '_vs_'
            outfile += label[jj].replace(' ', '_')
            outfile += '.png'
            py.savefig(outfile)

    return


def plot_quiver_one_pass(reread=False):
    """
    Plot a quiver vector plot between F814W in 2005 and F160W in 2010 and 2013.
    This is just to check that we don't hae any gross flows between the two cameras.
    """
    if reread:
        t2005 = starlists.read_matchup('MATCHUP.XYMEEE.F814W.2005.ref5')
        t2010 = starlists.read_matchup('MATCHUP.XYMEEE.F160W.2010.ref5')
        t2013 = starlists.read_matchup('MATCHUP.XYMEEE.F160W.2013.ref5')

        t2005.write('MATCHUP.XYMEEE.F814W.2005.ref5.fits')
        t2010.write('MATCHUP.XYMEEE.F160W.2010.ref5.fits')
        t2013.write('MATCHUP.XYMEEE.F160W.2013.ref5.fits')
    else:
        t2005 = Table.read('MATCHUP.XYMEEE.F814W.2005.ref5.fits')
        t2010 = Table.read('MATCHUP.XYMEEE.F160W.2010.ref5.fits')
        t2013 = Table.read('MATCHUP.XYMEEE.F160W.2013.ref5.fits')

    good = np.where((t2005['m'] < -8) & (t2010['m'] < -8) & (t2013['m'] < -8) &
                    (t2005['xe'] < 0.05) & (t2010['xe'] < 0.05) & (t2013['xe'] < 0.05) & 
                    (t2005['ye'] < 0.05) & (t2010['ye'] < 0.05) & (t2013['ye'] < 0.05) & 
                    (t2005['me'] < 0.05) & (t2010['me'] < 0.05) & (t2013['me'] < 0.05))[0]

    g2005 = t2005[good]
    g2010 = t2010[good]
    g2013 = t2013[good]

    dx_05_10 = (g2010['x'] - g2005['x']) * ast.scale['WFC'] * 1e3
    dy_05_10 = (g2010['y'] - g2005['y']) * ast.scale['WFC'] * 1e3

    dx_05_13 = (g2013['x'] - g2005['x']) * ast.scale['WFC'] * 1e3
    dy_05_13 = (g2013['y'] - g2005['y']) * ast.scale['WFC'] * 1e3

    dx_10_13 = (g2013['x'] - g2010['x']) * ast.scale['WFC'] * 1e3
    dy_10_13 = (g2013['y'] - g2010['y']) * ast.scale['WFC'] * 1e3

    small = np.where((np.abs(dx_05_10) < 20) & (np.abs(dy_05_10) < 20) & 
                     (np.abs(dx_05_13) < 20) & (np.abs(dy_05_13) < 20) & 
                     (np.abs(dx_10_13) < 20) & (np.abs(dy_10_13) < 20))[0]

    print(len(g2005), len(small), len(dx_05_10))
    g2005 = g2005[small]
    g2010 = g2010[small]
    g2013 = g2013[small]
    dx_05_10 = dx_05_10[small]
    dx_05_13 = dx_05_13[small]
    dx_10_13 = dx_10_13[small]
    dy_05_10 = dy_05_10[small]
    dy_05_13 = dy_05_13[small]
    dy_10_13 = dy_10_13[small]
    print(len(g2005), len(small), len(dx_05_10))

    qscale = 1e2
        
    py.clf()
    q = py.quiver(g2005['x'], g2005['y'], dx_05_10, dy_05_10, scale=qscale)
    py.quiverkey(q, 0.95, 0.95, 5, '5 mas', color='red', labelcolor='red')
    py.title('2010 - 2005')
    py.savefig('vec_diff_ref5_05_10.png')

    py.clf()
    q = py.quiver(g2005['x'], g2005['y'], dx_05_13, dy_05_13, scale=qscale)
    py.quiverkey(q, 0.95, 0.95, 5, '5 mas', color='red', labelcolor='red')
    py.title('2013 - 2005')
    py.savefig('vec_diff_ref5_05_13.png')

    py.clf()
    q = py.quiver(g2005['x'], g2005['y'], dx_10_13, dy_10_13, scale=qscale)
    py.quiverkey(q, 0.95, 0.95, 5, '5 mas', color='red', labelcolor='red')
    py.title('2013 - 2010')
    py.savefig('vec_diff_ref5_10_13.png')

    py.clf()
    py.plot(dx_05_10, dy_05_10, 'k.', ms=2)
    lim = 10
    py.axis([-lim, lim, -lim, lim])
    py.xlabel('X Proper Motion (mas)')
    py.ylabel('Y Proper Motion (mas)')
    py.title('2010 - 2005')
    py.savefig('pm_diff_ref5_05_10.png')

    py.clf()
    py.plot(dx_05_13, dy_05_13, 'k.', ms=2)
    py.axis([-lim, lim, -lim, lim])
    py.xlabel('X Proper Motion (mas)')
    py.ylabel('Y Proper Motion (mas)')
    py.title('2013 - 2005')
    py.savefig('pm_diff_ref5_05_13.png')

    py.clf()
    py.plot(dx_10_13, dy_10_13, 'k.', ms=2)
    py.axis([-lim, lim, -lim, lim])
    py.xlabel('X Proper Motion (mas)')
    py.ylabel('Y Proper Motion (mas)')
    py.title('2013 - 2010')
    py.savefig('pm_diff_ref5_10_13.png')

    print('2010 - 2005')
    print('   dx = {dx:6.2f} +/- {dxe:6.2f} mas'.format(dx=dx_05_10.mean(), dxe=dx_05_10.std()))
    print('   dy = {dy:6.2f} +/- {dye:6.2f} mas'.format(dy=dy_05_10.mean(), dye=dy_05_10.std()))

    print('2013 - 2005')
    print('   dx = {dx:6.2f} +/- {dxe:6.2f} mas'.format(dx=dx_05_13.mean(), dxe=dx_05_13.std()))
    print('   dy = {dy:6.2f} +/- {dye:6.2f} mas'.format(dy=dy_05_13.mean(), dye=dy_05_13.std()))

    print('2013 - 2010')
    print('   dx = {dx:6.2f} +/- {dxe:6.2f} mas'.format(dx=dx_10_13.mean(), dxe=dx_10_13.std()))
    print('   dy = {dy:6.2f} +/- {dye:6.2f} mas'.format(dy=dy_10_13.mean(), dye=dy_10_13.std()))

def make_refClust_catalog():
    """
    Make mat_all_good.fits where we matchup all the refClust starlists and
    get a preliminary look at the VPD and CMD. Do this on a per-position basis.
    """
    for ii in range(1, 4+1):
        pos_dir = 'match_refClust_pos{0}'.format(ii)
        os.chdir(work_dir + '03.MAT_POS/')
        
        fileUtil.mkdir(pos_dir)
        shutil.copy('2005_F814W/01.XYM/MATCHUP.XYMEEE.F814W.2005.refClust', pos_dir)
        os.chdir(work_dir + '03.MAT_POS/' + pos_dir)
        
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

def plot_quiver_one_pass_refClust():
    """
    Plot a quiver vector plot between F814W in 2005 and F160W in 2010 and 2013.
    This is just to check that we don't hae any gross flows between the two cameras.

    See notes from "HST Data Reduction (2014-06) for creation of the FITS table.
    """
    t = Table.read('mat_all_good.fits')
    t.rename_column('col05', 'x_2005')
    t.rename_column('col06', 'y_2005')
    t.rename_column('col07', 'm_2005')

    t.rename_column('col11', 'x_2010')
    t.rename_column('col12', 'y_2010')
    t.rename_column('col13', 'm_2010')
    t.rename_column('col23', 'xe_2010')
    t.rename_column('col24', 'ye_2010')
    t.rename_column('col25', 'me_2010')

    t.rename_column('col44', 'x_2013')
    t.rename_column('col45', 'y_2013')
    t.rename_column('col46', 'm_2013')
    t.rename_column('col56', 'xe_2013')
    t.rename_column('col57', 'ye_2013')
    t.rename_column('col58', 'me_2013')
    
    good = np.where((t['m_2005'] < -8) & (t['m_2010'] < -8) & (t['m_2013'] < -8) &
                    (t['xe_2010'] < 0.05) & (t['xe_2013'] < 0.05) & 
                    (t['ye_2010'] < 0.05) & (t['ye_2013'] < 0.05) & 
                    (t['me_2010'] < 0.05) & (t['me_2013'] < 0.05))[0]

    g = t[good]

    dx_05_10 = (g['x_2010'] - g['x_2005']) * ast.scale['WFC'] * 1e3
    dy_05_10 = (g['y_2010'] - g['y_2005']) * ast.scale['WFC'] * 1e3

    dx_05_13 = (g['x_2013'] - g['x_2005']) * ast.scale['WFC'] * 1e3
    dy_05_13 = (g['y_2013'] - g['y_2005']) * ast.scale['WFC'] * 1e3

    dx_10_13 = (g['x_2013'] - g['x_2010']) * ast.scale['WFC'] * 1e3
    dy_10_13 = (g['y_2013'] - g['y_2010']) * ast.scale['WFC'] * 1e3

    small = np.where((np.abs(dx_05_10) < 20) & (np.abs(dy_05_10) < 20) & 
                     (np.abs(dx_05_13) < 20) & (np.abs(dy_05_13) < 20) & 
                     (np.abs(dx_10_13) < 20) & (np.abs(dy_10_13) < 20))[0]

    print(len(g), len(small), len(dx_05_10))
    g = g[small]
    dx_05_10 = dx_05_10[small]
    dx_05_13 = dx_05_13[small]
    dx_10_13 = dx_10_13[small]
    dy_05_10 = dy_05_10[small]
    dy_05_13 = dy_05_13[small]
    dy_10_13 = dy_10_13[small]
    print(len(g), len(small), len(dx_05_10))

    qscale = 2e2
        
    py.clf()
    q = py.quiver(g['x_2005'], g['y_2005'], dx_05_10, dy_05_10, scale=qscale)
    py.quiverkey(q, 0.95, 0.95, 5, '5 mas', color='red', labelcolor='red')
    py.title('2010 - 2005')
    py.savefig('vec_diff_ref5_05_10.png')

    py.clf()
    q = py.quiver(g['x_2005'], g['y_2005'], dx_05_13, dy_05_13, scale=qscale)
    py.quiverkey(q, 0.95, 0.95, 5, '5 mas', color='red', labelcolor='red')
    py.title('2013 - 2005')
    py.savefig('vec_diff_ref5_05_13.png')

    py.clf()
    q = py.quiver(g['x_2005'], g['y_2005'], dx_10_13, dy_10_13, scale=qscale)
    py.quiverkey(q, 0.95, 0.95, 5, '5 mas', color='red', labelcolor='red')
    py.title('2013 - 2010')
    py.savefig('vec_diff_ref5_10_13.png')

    py.clf()
    py.plot(dx_05_10, dy_05_10, 'k.', ms=2)
    lim = 10
    py.axis([-lim, lim, -lim, lim])
    py.xlabel('X Proper Motion (mas)')
    py.ylabel('Y Proper Motion (mas)')
    py.title('2010 - 2005')
    py.savefig('pm_diff_ref5_05_10.png')

    py.clf()
    py.plot(dx_05_13, dy_05_13, 'k.', ms=2)
    py.axis([-lim, lim, -lim, lim])
    py.xlabel('X Proper Motion (mas)')
    py.ylabel('Y Proper Motion (mas)')
    py.title('2013 - 2005')
    py.savefig('pm_diff_ref5_05_13.png')

    py.clf()
    py.plot(dx_10_13, dy_10_13, 'k.', ms=2)
    py.axis([-lim, lim, -lim, lim])
    py.xlabel('X Proper Motion (mas)')
    py.ylabel('Y Proper Motion (mas)')
    py.title('2013 - 2010')
    py.savefig('pm_diff_ref5_10_13.png')

    print('2010 - 2005')
    print('   dx = {dx:6.2f} +/- {dxe:6.2f} mas'.format(dx=dx_05_10.mean(), dxe=dx_05_10.std()))
    print('   dy = {dy:6.2f} +/- {dye:6.2f} mas'.format(dy=dy_05_10.mean(), dye=dy_05_10.std()))

    print('2013 - 2005')
    print('   dx = {dx:6.2f} +/- {dxe:6.2f} mas'.format(dx=dx_05_13.mean(), dxe=dx_05_13.std()))
    print('   dy = {dy:6.2f} +/- {dye:6.2f} mas'.format(dy=dy_05_13.mean(), dye=dy_05_13.std()))

    print('2013 - 2010')
    print('   dx = {dx:6.2f} +/- {dxe:6.2f} mas'.format(dx=dx_10_13.mean(), dxe=dx_10_13.std()))
    print('   dy = {dy:6.2f} +/- {dye:6.2f} mas'.format(dy=dy_10_13.mean(), dye=dy_10_13.std()))

def plot_quiver_align(pos, orig=''):
    """
    Set orig='orig' to plot xorig and yorig (straight from ks2) rather than
    x and y from align output.
    """
    t = Table.read(work_dir + '50.ALIGN_KS2/align_a4_t.fits')
    
    good = np.where((t['m_2005_F814W'] < 17.5) &
                    (t['m_2010_F160W_' + pos] < 18) &
                    (t['m_2013_F160W_' + pos] < 18) &
                    (t['x_2005_F814W'] > -1) &
                    (t['x_2010_F160W_' + pos] > -1) &
                    (t['x_2013_F160W_' + pos] > -1) &
                    (t['y_2005_F814W'] > -1) &
                    (t['y_2010_F160W_' + pos] > -1) &
                    (t['y_2013_F160W_' + pos] > -1) &
                    (t['xe_2010_F160W_' + pos] < 0.05) &
                    (t['xe_2013_F160W_' + pos] < 0.05) &
                    (t['ye_2010_F160W_' + pos] < 0.05) &
                    (t['ye_2013_F160W_' + pos] < 0.05) & 
                    (t['me_2010_F160W_' + pos] < 0.05) &
                    (t['me_2013_F160W_' + pos] < 0.05))[0]
    print('Good = ', len(good))
    g = t[good]

    dx_05_10 = (g['x'+orig+'_2010_F160W_' + pos] - g['x'+orig+'_2005_F814W']) * ast.scale['WFC'] * 1e3
    dy_05_10 = (g['y'+orig+'_2010_F160W_' + pos] - g['y'+orig+'_2005_F814W']) * ast.scale['WFC'] * 1e3

    dx_05_13 = (g['x'+orig+'_2013_F160W_' + pos] - g['x'+orig+'_2005_F814W']) * ast.scale['WFC'] * 1e3
    dy_05_13 = (g['y'+orig+'_2013_F160W_' + pos] - g['y'+orig+'_2005_F814W']) * ast.scale['WFC'] * 1e3

    dx_10_13 = (g['x'+orig+'_2013_F160W_' + pos] - g['x'+orig+'_2010_F160W_' + pos]) * ast.scale['WFC'] * 1e3
    dy_10_13 = (g['y'+orig+'_2013_F160W_' + pos] - g['y'+orig+'_2010_F160W_' + pos]) * ast.scale['WFC'] * 1e3

    small = np.where((np.abs(dx_05_10) < 20) & (np.abs(dy_05_10) < 20) & 
                     (np.abs(dx_05_13) < 20) & (np.abs(dy_05_13) < 20) & 
                     (np.abs(dx_10_13) < 20) & (np.abs(dy_10_13) < 20))[0]
    print('Small = ', len(small))

    g = g[small]
    dx_05_10 = dx_05_10[small]
    dx_05_13 = dx_05_13[small]
    dx_10_13 = dx_10_13[small]
    dy_05_10 = dy_05_10[small]
    dy_05_13 = dy_05_13[small]
    dy_10_13 = dy_10_13[small]

    qscale = 2e2

    plot_dir = work_dir + '50.ALIGN_KS2/plots/'
            
    py.clf()
    q = py.quiver(g['x_2005_F814W'], g['y_2005_F814W'], dx_05_10, dy_05_10, scale=qscale)
    py.quiverkey(q, 0.95, 0.95, 5, '5 mas', color='red', labelcolor='red')
    py.title('2010 - 2005')
    py.savefig(plot_dir + 'quiver_align_05_10_' + pos + orig + '.png')

    py.clf()
    q = py.quiver(g['x_2005_F814W'], g['y_2005_F814W'], dx_05_13, dy_05_13, scale=qscale)
    py.quiverkey(q, 0.95, 0.95, 5, '5 mas', color='red', labelcolor='red')
    py.title('2013 - 2005')
    py.savefig(plot_dir + 'quiver_align_05_13_' + pos + orig + '.png')

    py.clf()
    q = py.quiver(g['x_2005_F814W'], g['y_2005_F814W'], dx_10_13, dy_10_13, scale=qscale)
    py.quiverkey(q, 0.95, 0.95, 5, '5 mas', color='red', labelcolor='red')
    py.title('2013 - 2010')
    py.savefig(plot_dir + 'quiver_align_10_13_' + pos + orig + '.png')

    py.clf()
    py.plot(dx_05_10, dy_05_10, 'k.', ms=2)
    lim = 10
    py.axis([-lim, lim, -lim, lim])
    py.xlabel('X Proper Motion (mas)')
    py.ylabel('Y Proper Motion (mas)')
    py.title('2010 - 2005')
    py.savefig(plot_dir + 'quiver_align_vpd_05_10_' + pos + orig + '.png')
    
    py.clf()
    py.plot(dx_05_13, dy_05_13, 'k.', ms=2)
    py.axis([-lim, lim, -lim, lim])
    py.xlabel('X Proper Motion (mas)')
    py.ylabel('Y Proper Motion (mas)')
    py.title('2013 - 2005')
    py.savefig(plot_dir + 'quiver_align_vpd_05_13_' + pos + orig + '.png')

    py.clf()
    py.plot(dx_10_13, dy_10_13, 'k.', ms=2)
    py.axis([-lim, lim, -lim, lim])
    py.xlabel('X Proper Motion (mas)')
    py.ylabel('Y Proper Motion (mas)')
    py.title('2013 - 2010')
    py.savefig(plot_dir + 'quiver_align_vpd_10_13_' + pos + orig + '.png')

    print('2010 - 2005')
    print('   dx = {dx:6.2f} +/- {dxe:6.2f} mas'.format(dx=dx_05_10.mean(), dxe=dx_05_10.std()))
    print('   dy = {dy:6.2f} +/- {dye:6.2f} mas'.format(dy=dy_05_10.mean(), dye=dy_05_10.std()))

    print('2013 - 2005')
    print('   dx = {dx:6.2f} +/- {dxe:6.2f} mas'.format(dx=dx_05_13.mean(), dxe=dx_05_13.std()))
    print('   dy = {dy:6.2f} +/- {dye:6.2f} mas'.format(dy=dy_05_13.mean(), dye=dy_05_13.std()))

    print('2013 - 2010')
    print('   dx = {dx:6.2f} +/- {dxe:6.2f} mas'.format(dx=dx_10_13.mean(), dxe=dx_10_13.std()))
    print('   dy = {dy:6.2f} +/- {dye:6.2f} mas'.format(dy=dy_10_13.mean(), dye=dy_10_13.std()))

    _out = open(plot_dir + 'quiver_align_stats_' + pos + orig + '.txt', 'w')
    _out.write('2010 - 2005\n')
    _out.write('   dx = {dx:6.2f} +/- {dxe:6.2f} mas\n'.format(dx=dx_05_10.mean(), dxe=dx_05_10.std()))
    _out.write('   dy = {dy:6.2f} +/- {dye:6.2f} mas\n'.format(dy=dy_05_10.mean(), dye=dy_05_10.std()))

    _out.write('2013 - 2005\n')
    _out.write('   dx = {dx:6.2f} +/- {dxe:6.2f} mas\n'.format(dx=dx_05_13.mean(), dxe=dx_05_13.std()))
    _out.write('   dy = {dy:6.2f} +/- {dye:6.2f} mas\n'.format(dy=dy_05_13.mean(), dye=dy_05_13.std()))

    _out.write('2013 - 2010\n')
    _out.write('   dx = {dx:6.2f} +/- {dxe:6.2f} mas\n'.format(dx=dx_10_13.mean(), dxe=dx_10_13.std()))
    _out.write('   dy = {dy:6.2f} +/- {dye:6.2f} mas\n'.format(dy=dy_10_13.mean(), dye=dy_10_13.std()))
    _out.close()

    

    
    return

def plot_vpd_across_field(nside=4, interact=False):
    """
    Plot the VPD at different field positions so we can see if there are
    systematic discrepancies due to residual distortions.
    """
    # Read in matched and aligned star lists from the *.ref5 analysis.
    # Recall these data sets are in the F814W reference frame with a 50 mas plate scale.
    t2005 = starlists.read_matchup(work_dir + '02.MAT/MATCHUP.XYMEEE.F814W.ref5')
    t2010 = starlists.read_matchup(work_dir + '02.MAT/MATCHUP.XYMEEE.F125W.ref5')

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
    jet = py.get_cmap('jet')
    cNorm = colors.Normalize(vmin=0, vmax=nside**2)
    colorMap = py.cm.ScalarMappable(norm=cNorm, cmap=jet)

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
    out += 'PMX:{4:5.2f} +/- {5:5.2f} PMY:{6:5.2f} +/- {7:5.2f}  '
    out += 'N:{8:5d}'
    print((out.format(xlo, xhi, ylo, yhi, pmx_all, 0.0, pmy_all, 0.0, len(idx2))))

    # Make a global proper motion diagram of star with a proper motion within
    # 1 mas/yr. This is mainly to see systematic flows due to residual distortion.
    pmTot = np.hypot(t2005.pmx, t2005.pmy)
    clust = np.where(pmTot < pmCut)[0]
    py.clf()
    q = py.quiver(t2005.x[clust], t2005.y[clust], t2005.pmx[clust], t2005.pmy[clust],
                  scale=18)
    py.quiverkey(q, 0.5, 0.98, 1, '1 mas/yr', color='red', labelcolor='red')
    py.xlabel('X Position (pixels)')
    py.ylabel('Y Position (pixels)')
    py.xlim(xlo, xhi)
    py.ylim(ylo, yhi)
    out = '{0}/plots/vec_proper_motion_all.png'
    py.savefig(out.format(work_dir))
    
    py.clf()
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

                py.plot(t2005.pmx[idx], t2005.pmy[idx], 'k.', ms=2, color=color)
                py.axis([-lim, lim, -lim, lim])

                py.xlabel('X Proper Motion (mas/yr)')
                py.ylabel('Y Proper Motion (mas/yr)')

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
            out += 'PMX:{4:5.2f} +/- {5:5.2f} PMY:{6:5.2f} +/- {7:5.2f}  '
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
        py.savefig(out.format(work_dir, nside))

    py.clf()
    q = py.quiver(xcen, ycen, pmx, pmy)
    py.quiverkey(q, 0.5, 0.98, 0.1, '0.1 mas/yr', color='red', labelcolor='red')
    py.xlabel('X Position (pixels)')
    py.ylabel('Y Position (pixels)')
    py.xlim(xlo, xhi)
    py.ylim(ylo, yhi)
    for xx in range(nside+1):
        py.axvline(xlo + xx * xboxsize, linestyle='--', color='grey')
    for yy in range(nside+1):
        py.axhline(ylo + yy * yboxsize, linestyle='--', color='grey')
    out = '{0}/plots/vec_proper_motion_grid_nside{1}.png'
    py.savefig(out.format(work_dir, nside))

    py.clf()
    q = py.quiver(xcen, ycen, pmx/pmxe, pmy/pmye)
    py.quiverkey(q, 0.5, 0.98, 3, '3 sigma', color='red', labelcolor='red')
    py.xlabel('X Position (pixels)')
    py.ylabel('Y Position (pixels)')
    py.xlim(xlo, xhi)
    py.ylim(ylo, yhi)
    for xx in range(nside+1):
        py.axvline(xlo + xx * xboxsize, linestyle='--', color='grey')
    for yy in range(nside+1):
        py.axhline(ylo + yy * yboxsize, linestyle='--', color='grey')
    out = '{0}/plots/vec_proper_motion_grid_sig_nside{1}.png'
    py.savefig(out.format(work_dir, nside))

def calc_years():
    """
    Calculate the epoch for each data set.
    """
    years = ['2005', '2010', '2010', '2010', '2013', '2013']
    filts = ['F814W', 'F125W', 'F139M', 'F160W', 'F160W', 'F160Ws']
    
    for ii in range(len(years)):
        dataDir = '{0}/{1}_{2}/00.DATA/'.format(work_dir, years[ii], filts[ii])

        epoch = flystar.calc_mean_year(glob.glob(dataDir + '*_flt.fits'))

        print(('{0}_{1} at {2:8.3f}'.format(years[ii], filts[ii], epoch)))
        
    
def make_master_lists():
    """
    Trim the ref5 master lists for each filter down to just stars with
    proper motions within 1 mas/yr of the cluster motion.
    """
    # Read in matched and aligned star lists from the *.ref5 analysis.
    # Recall these data sets are in the F814W reference frame with a 50 mas plate scale.
    print('Loading Data')
    t2005_814 = starlists.read_matchup(work_dir + '02.MAT/MATCHUP.XYMEEE.F814W.2005.ref5')
    t2010_125 = starlists.read_matchup(work_dir + '02.MAT/MATCHUP.XYMEEE.F125W.2010.ref5')
    t2010_139 = starlists.read_matchup(work_dir + '02.MAT/MATCHUP.XYMEEE.F139M.2010.ref5')
    t2010_160 = starlists.read_matchup(work_dir + '02.MAT/MATCHUP.XYMEEE.F160W.2010.ref5')
    t2013_160 = starlists.read_matchup(work_dir + '02.MAT/MATCHUP.XYMEEE.F160W.2013.ref5')
    t2013_160s = starlists.read_matchup(work_dir + '02.MAT/MATCHUP.XYMEEE.F160Ws.2013.ref5')

    scale = 50.0 # mas per pixel

    # Trim down to only those stars that are detected in all epochs.
    # Also make cuts on astrometric/photometric errors, etc.
    # We only want the well measured stars in this analysis.
    perrLim2005 = 3.0 / scale
    perrLim2010 = 4.0 / scale
    perrLim2013 = 5.0 / scale
    merrLim2005 = 0.05
    merrLim2010 = 0.1
    merrLim2013 = 0.1
    
    print('Trimming Data')
    cond = ((t2005_814['m'] != 0) & (t2010_125['m'] != 0) &
            (t2010_139['m'] != 0) & (t2010_160['m'] != 0) &
            (t2013_160['m'] != 0) &
            (t2005_814['xe'] < perrLim2005) & (t2005_814['ye'] < perrLim2005) &
            (t2010_125['xe'] < perrLim2010) & (t2010_125['ye'] < perrLim2010) &
            (t2010_139['xe'] < perrLim2010) & (t2010_139['ye'] < perrLim2010) &
            (t2010_160['xe'] < perrLim2010) & (t2010_160['ye'] < perrLim2010) &
            (t2013_160['xe'] < perrLim2013) & (t2013_160['ye'] < perrLim2013) &
            (t2005_814['me'] < merrLim2005) & (t2010_125['me'] < merrLim2010) &
            (t2010_139['me'] < merrLim2010) & (t2010_160['me'] < merrLim2010) &
            (t2013_160['me'] < merrLim2013))
    print('    Cutting down to {0} from {1}'.format(cond.sum(), len(t2005_814)))

    t2005_814 = t2005_814[cond]
    t2010_125 = t2010_125[cond]
    t2010_139 = t2010_139[cond]
    t2010_160 = t2010_160[cond]
    t2013_160 = t2013_160[cond]
    t2013_160s = t2013_160s[cond]

    # Calculate proper motions
    print('Calculating velocities')
    t = np.array([years['2005_F814W'], years['2010_F160W'], years['2013_F160W']])
    x = np.array([t2005_814['x'], t2010_160['x'], t2013_160['x']]).T
    y = np.array([t2005_814['y'], t2010_160['y'], t2013_160['y']]).T
    xe = np.array([t2005_814['xe'], t2010_160['xe'], t2013_160['xe']]).T
    ye = np.array([t2005_814['ye'], t2010_160['ye'], t2013_160['ye']]).T

    def linefit2(time, pos, pos_err):
        epochs = np.tile(time, (pos_err.shape[1], 1)).T

        # t_w = epochs / (pos_err ** 2)
        # p_w = pos / (pos_err ** 2)

        w = 1.0 / (pos_err ** 2)

        w_sum = w.sum(axis=0)

        wxy = (w * epochs * pos).sum(axis=0)
        wx = (w * epochs).sum(axis=0)
        wy = (w * pos).sum(axis=0)
        wxx = (w * epochs ** 2).sum(axis=0)

        denom = (w_sum * wxx) - (wx ** 2)
        
        vel = (w_sum * wxy - wx * wy) / denom
        pos0 = (wxx * wy - wx * wxy) / denom

        vel_err = np.sqrt( w_sum / denom )
        pos0_err = np.sqrt( wxx / denom )

        return pos0, vel, pos0_err, vel_err
        
    
    # Lets set a t0 value:
    t0 = 2010.0
    x0, vx, x0e, vxe = linefit2(t - t0, x.T, xe.T)
    y0, vy, y0e, vye = linefit2(t - t0, y.T, ye.T)
    vx *= scale
    vy *= scale
    vxe *= scale
    vye *= scale

    # Add the velocity fits to the 2005 table
    t2005_814['vx'] = vx
    t2005_814['vy'] = vy
    t2005_814['vxe'] = vxe
    t2005_814['vye'] = vye

    # Get rid of stars without velocities
    good = ((np.isnan(vx) == False) & (np.isnan(vy) == False))
    print('    Cutting down to {0} from {1}'.format(good.sum(), len(t2005_814)))
    t2005_814 = t2005_814[good]
    t2010_125 = t2010_125[good]
    t2010_139 = t2010_139[good]
    t2010_160 = t2010_160[good]
    t2013_160 = t2013_160[good]
    t2013_160s = t2013_160s[good]

    vx = vx[good]
    vy = vy[good]
    vxe = vxe[good]
    vye = vye[good]

    # Trim down to a 1 mas/yr radius
    # vx_mean = statsIter.mean(vx, lsigma=4, hsigma=4, iter=10, verbose=True)
    # vy_mean = statsIter.mean(vy, lsigma=4, hsigma=4, iter=10, verbose=True)
    vx_mean = 0.0
    vy_mean = 0.0 
    velCut = 0.7
    velErrCut = 0.2

    # Make a couple of plots to decide on (and show) the cuts
    py.clf()
    py.plot(vx, vy, 'k.', alpha=0.2)
    circ = py.Circle([vx_mean, vy_mean], radius=velCut, color='red', fill=False)
    py.gca().add_artist(circ)
    py.xlabel('X Velocity (mas/yr)')
    py.ylabel('Y Velocity (mas/yr)')
    py.axis([-2, 2, -2, 2])
    py.savefig('plots/make_master_vpd_cuts.png')

    py.clf()
    py.plot(t2005_814['m'], vxe, 'r.', alpha=0.2, label='X')
    py.plot(t2005_814['m'], vye, 'b.', alpha=0.2, label='Y')
    py.axhline(velErrCut, color='black')
    py.xlabel('F814W Magnitude')
    py.ylabel('Velocity Error (mas/yr)')
    py.legend()
    py.savefig('plots/make_master_verr_cuts.png')

    dv = np.hypot(vx - vx_mean, vy - vy_mean)
    idx2 = ((dv < velCut) & (vxe < velErrCut) & (vye < velErrCut))
    print('Making Velocity Cuts: v < {0:3.1f} mas/yr and verr < {1:3.1f} mas/yr'.format(velCut, velErrCut))
    print('    Cutting down to {0} from {1}'.format(idx2.sum(), len(t2005_814)))

    t2005_814 = t2005_814[idx2]
    t2010_125 = t2010_125[idx2]
    t2010_139 = t2010_139[idx2]
    t2010_160 = t2010_160[idx2]
    t2013_160 = t2013_160[idx2]
    t2013_160s = t2013_160s[idx2]

    _o814_05 = open(work_dir + '02.MAT/MASTER.F814W.2005.ref5', 'w')
    _o125_10 = open(work_dir + '02.MAT/MASTER.F125W.2010.ref5', 'w')
    _o139_10 = open(work_dir + '02.MAT/MASTER.F139M.2010.ref5', 'w')
    _o160_10 = open(work_dir + '02.MAT/MASTER.F160W.2010.ref5', 'w')
    _o160_13 = open(work_dir + '02.MAT/MASTER.F160W.2013.ref5', 'w')
    _o160s_13 = open(work_dir + '02.MAT/MASTER.F160Ws.2013.ref5', 'w')

    ofmt = '{0:10.4f} {1:10.4f} {2:8.4f} {3:10.4f} {4:10.4f} {5:8.4f} {6}\n'
    for ii in range(len(t2005_814)):
        _o814_05.write(ofmt.format(t2005_814['x'][ii], t2005_814['y'][ii], t2005_814['m'][ii],
                                   t2005_814['xe'][ii], t2005_814['ye'][ii], t2005_814['me'][ii],
                                   t2005_814['name'][ii]))
        _o125_10.write(ofmt.format(t2010_125['x'][ii], t2010_125['y'][ii], t2010_125['m'][ii],
                                   t2010_125['xe'][ii], t2010_125['ye'][ii], t2010_125['me'][ii],
                                   t2010_125['name'][ii]))
        _o139_10.write(ofmt.format(t2010_139['x'][ii], t2010_139['y'][ii], t2010_139['m'][ii],
                                   t2010_139['xe'][ii], t2010_139['ye'][ii], t2010_139['me'][ii],
                                   t2010_139['name'][ii]))
        _o160_10.write(ofmt.format(t2010_160['x'][ii], t2010_160['y'][ii], t2010_160['m'][ii],
                                   t2010_160['xe'][ii], t2010_160['ye'][ii], t2010_160['me'][ii],
                                   t2010_160['name'][ii]))
        _o160_13.write(ofmt.format(t2013_160['x'][ii], t2013_160['y'][ii], t2013_160['m'][ii],
                                   t2013_160['xe'][ii], t2013_160['ye'][ii], t2013_160['me'][ii],
                                   t2013_160['name'][ii]))
        _o160s_13.write(ofmt.format(t2013_160s['x'][ii], t2013_160s['y'][ii], t2013_160s['m'][ii],
                                    t2013_160s['xe'][ii], t2013_160s['ye'][ii], t2013_160s['me'][ii],
                                    t2013_160s['name'][ii]))

    _o814_05.close()
    _o125_10.close()
    _o139_10.close()
    _o160_10.close()
    _o160_13.close()
    _o160s_13.close()


def make_pos_directories():
    """
    Make the individual position directories. The structure is complicated
    enough that it is worth doing in code. All of this goes under

        03.MAT_POS/

    """
    os.chdir(work_dir + '/03.MAT_POS')

    years = ['2005', '2010', '2013']

    filts = {'2005': ['F814W'],
             '2010': ['F125W', 'F139M', 'F160W'],
             '2013': ['F160W', 'F160Ws']}

    pos4_exceptions = ['2005_F814W', '2013_F160Ws']
    old_mat_dir = '../02.MAT'

    for year in years:
        filts_in_year = filts[year]

        for filt in filts_in_year:
            epoch_filt = '{0}_{1}'.format(year, filt)

            if epoch_filt in pos4_exceptions:
                posits = ['']
            else:
                posits = ['pos1', 'pos2', 'pos3', 'pos4']

            # Define the master file for this combo
            master_file = '{0}/MASTER.{1}.{2}.ref5'.format(old_mat_dir, filt, year)
            
            for pos in posits:
                ep_filt_pos = epoch_filt
                if pos != '':
                    ep_filt_pos += '_{0}'.format(pos)

                # Make the directory
                fileUtil.mkdir(ep_filt_pos)
                fileUtil.mkdir(ep_filt_pos + '/01.XYM')

                # Copy over the master file
                shutil.copy(master_file, ep_filt_pos + '/01.XYM/')

    return

def calc_pos_number():
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
    mat_files = glob.glob(work_dir + '02.MAT/MAT.*')
    epoch_filt = np.zeros(len(mat_files), dtype='S12')
    pos = np.zeros(len(mat_files), dtype='S4')
    mat_root = np.zeros(len(mat_files), dtype='S8')
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
        if (mat_num >= 100) and (mat_num <= 129):
            epoch_filt[mm] = '2010_F125W'
        if (mat_num >= 130) and (mat_num <= 161):
            epoch_filt[mm] = '2010_F139M'
        if (mat_num >= 162) and (mat_num <= 199):
            epoch_filt[mm] = '2010_F160W'
        if (mat_num >= 200) and (mat_num <= 259):
            epoch_filt[mm] = '2013_F160W'
        if (mat_num >= 270) and (mat_num <= 299):
            epoch_filt[mm] = '2013_F160Ws'
            
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

            print() 
            for ii in idx:
                print(fmt.format(mat_root[ii], epoch_filt[ii], pos[ii],
                                 xavg[ii], yavg[ii]))
                    
    return mat_root, epoch_filt, pos

def setup_xym_by_pos():
    """
    Something
    """
    os.chdir(work_dir + '/03.MAT_POS')

    pos4_exceptions = ['2005_F814W', '2013_F160Ws']
    old_mat_dir = '../02.MAT/'

    # Get the MAT file names and positions.
    mat_root, epoch_filt, pos = calc_pos_number()

    # Read the old IN.xym2bar file to match MAT.??? to the *.xym files
    mat_xym_files = read_in_xym2mat(old_mat_dir + 'IN.xym2mat')
    
    open_logs = {}
    
    for ii in range(len(mat_root)):
        # copy stuff to this directory
        to_dir = epoch_filt[ii]

        if epoch_filt[ii] not in pos4_exceptions:
            to_dir += '_' + pos[ii]

        to_dir += '/01.XYM/'
        print('Copying to {0} for {1}'.format(to_dir, mat_root[ii]))
        
        # Get rid of the filter-related letters for the 02.MAT/ sub-dir.
        efilt_strip = epoch_filt[ii].replace('W', '').replace('F', '').replace('M', '')

        # Copy the old MAT file
        from_file = old_mat_dir + efilt_strip + '/' + mat_root[ii]
        shutil.copy(from_file, to_dir)
        print('    Copy ' + from_file)

        # Copy the xym file
        mat_num = mat_root[ii].split('.')[1]

        xym_file = mat_xym_files[mat_num]
        
        shutil.copy(xym_file, to_dir)
        print('    Copy ' + xym_file)

        # Keep a record of the match between XYM and MAT files
        logfile = to_dir + 'xym_mat.txt'

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

    mat_num = np.zeros(cnt_mat, dtype='S3')
    xym_files = np.zeros(cnt_mat, dtype='S80')

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


            
def xym_by_pos(year, filt, pos, Nepochs):
    """
    Re-do the alignment with the ref5 master frames (cluster members only);
    but do the alignments on each position separately.

    Doesn't work for 2005_F814W or 2013_F160Ws.
    """
    mat_dir = '{0}/03.MAT_POS/{1}_{2}_{3}/01.XYM/'.format(work_dir, year, filt, pos)

    master_file = 'MASTER.{0}.{1}.ref5'.format(filt, year)

    # Read in the xym_mat.txt file that maps the MAT and XYM files together.
    tab = Table.read(mat_dir + 'xym_mat.txt', format='ascii')
    
    # Make IN.xym2mat and IN.xym2bar file
    _mat = open(mat_dir + 'IN.xym2mat', 'w')
    _bar = open(mat_dir + 'IN.xym2bar', 'w')
    
    _mat.write('000 ' + master_file + ' c0\n')

    for ii in range(len(tab)):
        _mat.write('{0} {1} c9\n'.format(tab[ii][0], tab[ii][1]))
        _bar.write('{0} {1} c9\n'.format(tab[ii][0], tab[ii][1]))

    _mat.close()
    _bar.close()

    # Run xym2mat
    os.chdir(mat_dir)
    subprocess.call(['xym2mat', '22'])
    subprocess.call(['xym2mat', '24'])
    subprocess.call(['xym2mat', '25'])

    # Make IN.xym2bar file
    subprocess.call(['xym2bar', str(Nepochs)])

    # Copy final output files
    suffix = '.{0}.{1}.{2}.refClust'.format(filt, year, pos)
    shutil.copy('MATCHUP.XYMEEE', 'MATCHUP.XYMEEE' + suffix)
    shutil.copy('TRANS.xym2mat', 'TRANS.xym2mat' + suffix)
    shutil.copy('TRANS.xym2bar', 'TRANS.xym2bar' + suffix)
    shutil.copy('IN.xym2mat', 'IN.xym2mat' + suffix)
    shutil.copy('IN.xym2bar', 'IN.xym2bar' + suffix)

    return
            
def make_brite_list_2010():
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

    os.chdir(work_dir + '12.KS2_2010')
    shutil.copy('{0}02.MAT/{1}'.format(work_dir, matchup_file[0]), './')
    shutil.copy('{0}02.MAT/{1}'.format(work_dir, matchup_file[1]), './')
    shutil.copy('{0}02.MAT/{1}'.format(work_dir, matchup_file[2]), './')

    # Read in the matchup files.
    list_160 = starlists.read_matchup(matchup_file[0])
    list_139 = starlists.read_matchup(matchup_file[1])
    list_125 = starlists.read_matchup(matchup_file[2])

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
    brite_obs_125 = open('BRITE_F125W.XYMEEE', 'w')
    brite_obs_139 = open('BRITE_F139M.XYMEEE', 'w')
    brite_obs_160 = open('BRITE_F160W.XYMEEE', 'w')
    
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

    brite = open('BRITE.XYM', 'w')
    
    for ii in range(len(stars)):
        fmt = '{x:10.4f}  {y:10.4f}  {m1:10.4f}  {m2:10.4f}  {m3:10.4f}\n'

        brite.write(fmt.format(x=stars['x_1'][ii],
                               y=stars['y_1'][ii],
                               m1=stars['m_1'][ii],
                               m2=stars['m_2'][ii],
                               m3=stars['m_3'][ii]))
    
    brite.close()

    return

def find_top_stars(reread=False):
    """
    Select some stars to be our topStars named sources.
    Verify that they are detected in all of the
    epochs and all of the filters.
    """
    root_2005_814 = work_dir + '11.KS2_2005/nimfo2bar.xymeee.ks2.F1'
    root_2010_160 = work_dir + '12.KS2_2010/nimfo2bar.xymeee.F1'
    root_2010_139 = work_dir + '12.KS2_2010/nimfo2bar.xymeee.F2'
    root_2010_125 = work_dir + '12.KS2_2010/nimfo2bar.xymeee.F3'
    root_2013_160 = work_dir + '13.KS2_2013/nimfo2bar.xymeee.F1'
    root_2013_160s = work_dir + '13.KS2_2013/nimfo2bar.xymeee.F2'

    print('Reading data')
    if reread == True:
        t_2005_814 = starlists.read_nimfo2bar(root_2005_814)
        t_2010_160 = starlists.read_nimfo2bar(root_2010_160)
        t_2010_139 = starlists.read_nimfo2bar(root_2010_139)
        t_2010_125 = starlists.read_nimfo2bar(root_2010_125)
        t_2013_160 = starlists.read_nimfo2bar(root_2013_160)
        t_2013_160s = starlists.read_nimfo2bar(root_2013_160s)

        t_2005_814.write(root_2005_814 + '.fits', overwrite=True)
        t_2010_160.write(root_2010_160 + '.fits', overwrite=True)
        t_2010_139.write(root_2010_139 + '.fits', overwrite=True)
        t_2010_125.write(root_2010_125 + '.fits', overwrite=True)
        t_2013_160.write(root_2013_160 + '.fits', overwrite=True)
        t_2013_160s.write(root_2013_160s + '.fits', overwrite=True)
    else:
        t_2005_814 = Table.read(root_2005_814 + '.fits')
        t_2010_160 = Table.read(root_2010_160 + '.fits')
        t_2010_139 = Table.read(root_2010_139 + '.fits')
        t_2010_125 = Table.read(root_2010_125 + '.fits')
        t_2013_160 = Table.read(root_2013_160 + '.fits')
        t_2013_160s = Table.read(root_2013_160s + '.fits')
        

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

    t_2005_814 = t_2005_814[g_2005_814]
    t_2010_125 = t_2010_125[g_2010_125]
    t_2010_139 = t_2010_139[g_2010_139]
    t_2010_160 = t_2010_160[g_2010_160]
    t_2013_160 = t_2013_160[g_2013_160]
    t_2013_160s = t_2013_160s[g_2013_160s]

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

    print('Matching 2010_160 and 2005_814')
    r5 = align.match(t_2010_160['x'], t_2010_160['y'], t_2010_160['m'],
                     t_2005_814['x'], t_2005_814['y'], t_2005_814['m'] + 4.2,
                     dr_tol=0.5, dm_tol=1.0)
    print('    found {0:d} matches'.format(len(r5[0])))
    t_2010_160 = t_2010_160[r5[0]]
    t_2010_125 = t_2010_125[r5[0]]
    t_2010_139 = t_2010_139[r5[0]]
    t_2013_160 = t_2013_160[r5[0]]
    t_2013_160s = t_2013_160s[r5[0]]
    t_2005_814 = t_2005_814[r5[1]]

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

def combine_nimfo2bar_brite():
    """
    For all the data, we need to add back in the missing bright stars
    from the one-pass analysis.
    """
    epochs = ['2005_F814W', 
              '2010_F125W_pos1', '2010_F125W_pos2', '2010_F125W_pos3', '2010_F125W_pos4',
              '2010_F139M_pos1', '2010_F139M_pos2', '2010_F139M_pos3', '2010_F139M_pos4',
              '2010_F160W_pos1', '2010_F160W_pos2', '2010_F160W_pos3', '2010_F160W_pos4',
              '2013_F160W_pos1', '2013_F160W_pos2', '2013_F160W_pos3', '2013_F160W_pos4',
              '2013_F160Ws']
    magCuts = [-13.5, 
               -10, -10, -10, -10,
                -9,  -9,  -9,  -9,
               -10, -10, -10, -10,
               -10, -10, -10, -10,
               -10, -10, -10, -10]
               
    
    for ee in range(len(epochs)):
        ep = epochs[ee]
        year = ep.split('_')[0]
        filt = ep.split('_')[1]

        mat_dir = work_dir + '03.MAT_POS/' + ep + '/01.XYM/'
        mat_file = 'MATCHUP.XYMEEE'
        ks2_dir = work_dir
        ks2_file = 'nimfo2bar.xymeee.F1'
        N_filter = 1
        if year == '2005':
            ks2_dir += '11.KS2_2005/'
            ks2_file = ks2_file.replace('.F1', '.ks2.F1')

        if year == '2010':
            ks2_dir += '12.KS2_2010/0{pos}.pos{pos}/'.format(pos=ep[-1])
            N_filter = 3
            if filt == 'F139M':
                ks2_file = ks2_file.replace('.F1', '.F2')
            if filt == 'F125W':
                ks2_file = ks2_file.replace('.F1', '.F3')

        if year == '2013':
            ks2_dir += '13.KS2_2013/'
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
                           'N_find': '%3d', 'N_xywell': '%3d', 'name': lambda n: n.zfill(6)})

            
    return
    
def align_to_fits(align_root):
    """
    Given starlist output from align, returns to MATCHUP format. Inputs are the
    align *.pos, *.err, *.mag, *.param, and *.name files. Will return 3 MATCHUP
    files, one for each epoch. Align run order:

    2005_F814W, 2010_F125W, 2010_F139M, 2010_F160W, 2013_F160W, 2013_f160Ws

    """
    s = starset.StarSet(work_dir + '50.ALIGN_KS2/' + align_root)

    name = s.getArray('name')

    # Setup the mapping between epoch, filter, pos and the align index.
    align_idx = {0: '2005_F814W',
                 1: '2010_F125W_pos1',
                 2: '2010_F125W_pos2',
                 3: '2010_F125W_pos3',
                 4: '2010_F125W_pos4',
                 5: '2010_F139M_pos1',
                 6: '2010_F139M_pos2',
                 7: '2010_F139M_pos3',
                 8: '2010_F139M_pos4',
                 9: '2010_F160W_pos1',
                 10: '2010_F160W_pos2',
                 11: '2010_F160W_pos3',
                 12: '2010_F160W_pos4',
                 13: '2013_F160W_pos1',
                 14: '2013_F160W_pos2',
                 15: '2013_F160W_pos3',
                 16: '2013_F160W_pos4',
                 17: '2013_F160Ws'}

    num_epochs = len(align_idx)

    # Construct a list of columns and column names
    columns = []
    col_names = []

    # For each epoch, add all the columns we want to keep.
    for ii in range(num_epochs):
        columns.append(s.getArrayFromEpoch(ii, 'xpix'))
        col_names.append('x_' + align_idx[ii])
        
        columns.append(s.getArrayFromEpoch(ii, 'ypix'))
        col_names.append('y_' + align_idx[ii])

        columns.append(s.getArrayFromEpoch(ii, 'mag'))
        col_names.append('m_' + align_idx[ii])

        columns.append(s.getArrayFromEpoch(ii, 'xpixerr_p'))
        col_names.append('xe_' + align_idx[ii])

        columns.append(s.getArrayFromEpoch(ii, 'ypixerr_p'))
        col_names.append('ye_' + align_idx[ii])

        columns.append(s.getArrayFromEpoch(ii, 'snr'))
        col_names.append('me_' + align_idx[ii])

        columns.append(s.getArrayFromEpoch(ii, 'nframes'))
        col_names.append('n_' + align_idx[ii])
                
        columns.append(s.getArrayFromEpoch(ii, 'xorig'))
        col_names.append('xorig_' + align_idx[ii])

        columns.append(s.getArrayFromEpoch(ii, 'yorig'))
        col_names.append('yorig_' + align_idx[ii])
    
    t = Table(columns, names=col_names)

    t.table_name = align_root
    t.write(work_dir + '50.ALIGN_KS2/' + align_root + '.fits', overwrite=True)

    return

def combine_mosaic_pos(align_root='align_a4_t'):
    t = Table.read(work_dir + '50.ALIGN_KS2/' + align_root + '.fits')
    
    # First we need to combine all the positions of the mosaic together.
    mosaic_epochs = ['2010_F125W', '2010_F139M', '2010_F160W', '2013_F160W']
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

    # Loop through each year+filter combination    
    for ee in range(n_epochs):
        print('Working on: ' + mosaic_epochs[ee])
        
        for pp in range(len(pos)):
            suffix = '_' + mosaic_epochs[ee] + '_' + pos[pp]
            
            x = t['x' + suffix]
            y = t['y' + suffix]
            m = t['m' + suffix]
            xe = t['xe' + suffix]
            ye = t['ye' + suffix]
            me = t['me' + suffix]
            n = t['n' + suffix]
                  
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
            x_wavg[ee, det] += x[det] * w_x
            y_wavg[ee, det] += y[det] * w_y
            f_wavg[ee, det] += f_det * w_f
            fe_wavg[ee, det] += fe_det * w_f

            wx_wavg[ee, det] += w_x
            wy_wavg[ee, det] += w_y
            wf_wavg[ee, det] += w_f

            nframes[ee, det] += n[det]

        # Finished this epoch, finish the weighted average calculation
        x_wavg[ee, :] /= wx_wavg[ee, :]
        xe_wavg[ee, :] = (1.0 / wx_wavg[ee, :])**0.5
        
        y_wavg[ee, :] /= wy_wavg[ee, :]
        ye_wavg[ee, :] = (1.0 / wy_wavg[ee, :])**0.5
        
        f_wavg[ee, :] /= wf_wavg[ee, :]
        # fe_wavg[ee, :] = (1.0 / wf_wavg[ee, :])**0.5
        fe_wavg[ee, :] /= wf_wavg[ee, :]  # Spe

        m_wavg[ee, :], me_wavg[ee, :] = photo.flux2mag(f_wavg[ee, :], fe_wavg[ee, :])

        # Add new columns to the table
        t.add_column(Column(x_wavg[ee, :], name='x_' + mosaic_epochs[ee]))
        t.add_column(Column(y_wavg[ee, :], name='y_' + mosaic_epochs[ee]))
        t.add_column(Column(m_wavg[ee, :], name='m_' + mosaic_epochs[ee]))

        t.add_column(Column(xe_wavg[ee, :], name='xe_' + mosaic_epochs[ee]))
        t.add_column(Column(ye_wavg[ee, :], name='ye_' + mosaic_epochs[ee]))
        t.add_column(Column(me_wavg[ee, :], name='me_' + mosaic_epochs[ee]))
        
        t.add_column(Column(nframes[ee, :], name='n_' + mosaic_epochs[ee]))

        
    # Table Clean Up:
    # Remove the position dependent columns from the table.
    for ee in range(n_epochs):
        for pp in range(len(pos)):
            suffix = '_' + mosaic_epochs[ee] + '_' + pos[pp]

            # Identify the stars with detections in this list.
            # Used for checking zeropoint offsets.
            x = t['x' + suffix]
            y = t['y' + suffix]
            m = t['m' + suffix]
            xe = t['xe' + suffix]
            ye = t['ye' + suffix]
            me = t['me' + suffix]
            n = t['n' + suffix]
            
            det = np.where((x > -1e4) & (xe > 0) &
                           (y > -1e4) & (ye > 0) &
                           (m > 0) & (me > 0))[0]

            dx = x[det] - t['x_' + mosaic_epochs[ee]][det]
            dy = y[det] - t['y_' + mosaic_epochs[ee]][det]
            dm = m[det] - t['m_' + mosaic_epochs[ee]][det]

            print('Stats for: ' + mosaic_epochs[ee] + ' ' + pos[pp])
            print('    dx = {0:7.3f} +/- {1:7.3f}'.format(dx.mean(), dx.std()))
            print('    dy = {0:7.3f} +/- {1:7.3f}'.format(dy.mean(), dy.std()))
            print('    dm = {0:7.3f} +/- {1:7.3f}'.format(dm.mean(), dm.std()))
            
            t.remove_column('x' + suffix)
            t.remove_column('y' + suffix)
            t.remove_column('m' + suffix)
            t.remove_column('xe' + suffix)
            t.remove_column('ye' + suffix)
            t.remove_column('me' + suffix)
            t.remove_column('n' + suffix)
            t.remove_column('xorig' + suffix)
            t.remove_column('yorig' + suffix)

    # Remove the xorig and yorig for the other epochs
    t.remove_column('xorig_2005_F814W')
    t.remove_column('yorig_2005_F814W')
    t.remove_column('xorig_2013_F160Ws')
    t.remove_column('yorig_2013_F160Ws')

    t.write(work_dir + '50.ALIGN_KS2/' + align_root + '_combo_pos.fits',
            format='fits', overwrite=True)

                    
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

    d_all = Table.read(work_dir + '50.ALIGN_KS2/align_a4_t_combo_pos.fits')

    # TRIM out all stars that aren't detected in all 3 epochs:
    #    2005_814
    #    2010_160
    #    2013_160
    idx = np.where((d_all['x_2005_F814W'] > -999) &
                   (d_all['x_2010_F160W'] > -999) &
                   (d_all['x_2013_F160W'] > -999) &
                   (d_all['n_2005_F814W'] > 1) &
                   (d_all['n_2010_F160W'] > 1) &
                   (d_all['n_2013_F160W'] > 1) &
                   (d_all['xe_2005_F814W'] > 0) &
                   (d_all['xe_2010_F160W'] > 0) &
                   (d_all['xe_2013_F160W'] > 0) &
                   (d_all['ye_2005_F814W'] > 0) &
                   (d_all['ye_2010_F160W'] > 0) &
                   (d_all['ye_2013_F160W'] > 0))[0]

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
    print('Found {0:3} in 2005 F814W'.format(len(tmp_2005)))
    print('Found {0:3} in 2010 F160W'.format(len(tmp_2010)))
    print('Found {0:3} in 2013 F160W'.format(len(tmp_2013)))
    print('')
    print('Kept {0:d} of {1:d} stars in all 3 epochs.'.format(len(idx), len(d_all)))
    
    d = d_all[idx]
    
    #Changing rms errors into standard errors for the f153m data
    xeom_2005_814 = d['xe_2005_F814W'] / np.sqrt(d['n_2005_F814W'])
    yeom_2005_814 = d['ye_2005_F814W'] / np.sqrt(d['n_2005_F814W'])
    xeom_2010_160 = d['xe_2010_F160W'] / np.sqrt(d['n_2010_F160W'])
    yeom_2010_160 = d['ye_2010_F160W'] / np.sqrt(d['n_2010_F160W'])
    xeom_2013_160 = d['xe_2013_F160W'] / np.sqrt(d['n_2013_F160W'])
    yeom_2013_160 = d['ye_2013_F160W'] / np.sqrt(d['n_2013_F160W'])
    
    # Fit velocities. Will use an error-weighted t0, specified to each object
    t = np.array([years['2005_F814W'], years['2010_F160W'], years['2013_F160W']])

    if use_RMSE:
        # Shape = (nepochs, nstars)
        xerr = np.array([d['xe_2005_F814W'], d['xe_2010_F160W'], d['xe_2013_F160W']])
        yerr = np.array([d['ye_2005_F814W'], d['ye_2010_F160W'], d['ye_2013_F160W']])
    else:
        xerr = np.array([xeom_2005_814, xeom_2010_160, xeom_2013_160])
        yerr = np.array([yeom_2005_814, yeom_2010_160, yeom_2013_160])

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
    d.add_column(Column(data=t0[0],name='fit_t0'))

    d.add_column(Column(data=np.ones(nstars),name='fit_x0'))
    d.add_column(Column(data=np.ones(nstars),name='fit_vx'))
    d.add_column(Column(data=np.ones(nstars),name='fit_y0'))
    d.add_column(Column(data=np.ones(nstars),name='fit_vy'))

    d.add_column(Column(data=np.ones(nstars),name='fit_x0e'))
    d.add_column(Column(data=np.ones(nstars),name='fit_vxe'))
    d.add_column(Column(data=np.ones(nstars),name='fit_y0e'))
    d.add_column(Column(data=np.ones(nstars),name='fit_vye'))

    for ii in range(len(d)):
        x = np.array([d['x_2005_F814W'][ii], d['x_2010_F160W'][ii], d['x_2013_F160W'][ii]])
        y = np.array([d['y_2005_F814W'][ii], d['y_2010_F160W'][ii], d['y_2013_F160W'][ii]])
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

        d['fit_x0'][ii] = vxOpt[1]
        d['fit_vx'][ii] = vxOpt[0]
        d['fit_x0e'][ii] = vxErr[1]
        d['fit_vxe'][ii] = vxErr[0]

        d['fit_y0'][ii] = vyOpt[1]
        d['fit_vy'][ii] = vyOpt[0]
        d['fit_y0e'][ii] = vyErr[1]
        d['fit_vye'][ii] = vyErr[0]


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

    d.write(work_dir + '50.ALIGN_KS2/' + catalog_name, format='fits', overwrite=True)
    
    return

def art_set_detected(use_obs_align=False):
    """
    Create the "detected" columsn in the artificial star list.
    
    Apply a set of criteria to call a star detected based on on how
    closely the input and output position/flux match.
    """
    in_file = work_dir + '51.ALIGN_ART/art_align'
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

    catalog_name = work_dir + '51.ALIGN_ART/art_align' 
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
    real_obs = work_dir + '50.ALIGN_KS2/align_a4_t_combo_pos.fits'
    art_obs = work_dir + '51.ALIGN_ART/art_align'
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
        
        py.figure(1, figsize = (10,10))
        py.clf()
        py.semilogy(art_mag, art_err, 'k.', ms=4, label='Artificial', alpha=0.2)
        py.semilogy(obs_mag, obs_err, 'r.', ms=4, label='Observed', alpha=0.5)
        py.plot(mag_cent, obs_pos_err_mean, 'b-', linewidth=2,
                label='Observed Median')
        py.plot(mag_cent, art_pos_err_mean, 'g-', linewidth=2,
                label='Artificial Median')
        py.xlabel('Observed Mag')
        py.ylabel('Positional Error (pix)')
        py.title('Positional Errors,' + epoch)
        py.legend(loc=2, numpoints=1)
        py.xlim(13, max(art_mag))
        py.ylim(1e-4, 1)
        plot_file = work_dir + 'plots/Pos_errcomp' + suffix + '_' + epoch
        if use_obs_align:
            plot_file += '_obs'
        else:
            plot_file += '_art'
        plot_file += '.png'
        py.savefig(plot_file)

        return

    def plot_mag_errors(epoch, obs_mag, obs_err, art_mag, art_err, mag_cent,
                        obs_mag_err_mean, art_mag_err_mean, suffix=''):
        
        py.figure(2, figsize=(10,10))
        py.clf()
        py.semilogy(art_mag, art_merr, 'k.', ms=4, label='Artificial', alpha=0.2)
        py.semilogy(obs_mag, obs_merr, 'r.', ms=4, label='Observed', alpha=0.5)
        py.plot(mag_cent, obs_mag_err_mean, 'b-', linewidth=2,
                label='Observed Median')
        py.plot(mag_cent, art_mag_err_mean, 'g-', linewidth=2,
                label='Artificial Median')
        py.xlabel('Observed Mag')
        py.ylabel('Photometric Error (mag)')
        py.title('Photometric Errors,' + epoch)
        py.legend(loc=2, numpoints=1)
        py.xlim(13, max(art_mag))
        py.ylim(1e-4, 1)
        plot_file = work_dir + 'plots/Mag_errcomp' + suffix + '_' + epoch
        if use_obs_align:
            plot_file += '_obs'
        else:
            plot_file += '_art'
        plot_file += '.png'
        py.savefig(plot_file)

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

        
    out_file = work_dir + '51.ALIGN_ART/art_align'
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

    in_file = work_dir + '51.ALIGN_ART/art_align'
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

    d.add_column(Column(data=np.ones(nstars), name='fit_x0'))
    d.add_column(Column(data=np.ones(nstars), name='fit_vx'))
    d.add_column(Column(data=np.ones(nstars), name='fit_y0'))
    d.add_column(Column(data=np.ones(nstars), name='fit_vy'))

    d.add_column(Column(data=np.ones(nstars), name='fit_x0e'))
    d.add_column(Column(data=np.ones(nstars), name='fit_vxe'))
    d.add_column(Column(data=np.ones(nstars), name='fit_y0e'))
    d.add_column(Column(data=np.ones(nstars), name='fit_vye'))

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

        d['fit_x0'][ii] = vxOpt[1]
        d['fit_vx'][ii] = vxOpt[0]
        d['fit_x0e'][ii] = vxErr[1]
        d['fit_vxe'][ii] = vxErr[0]

        d['fit_y0'][ii] = vyOpt[1]
        d['fit_vy'][ii] = vyOpt[0]
        d['fit_y0e'][ii] = vyErr[1]
        d['fit_vye'][ii] = vyErr[0]

    catalog_name = work_dir + '51.ALIGN_ART/wd1_art_catalog'
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
    
    catFile = work_dir + '50.ALIGN_KS2/' + catalog_name
    tab = Table.read(catFile)

    good = (tab['fit_vxe'] < 0.01) & (tab['fit_vye'] < 0.01) & \
        (tab['me_2005_F814W'] < 0.1) & (tab['me_2010_F160W'] < 0.1)

    tab2 = tab[good]

    vx = tab2['fit_vx'] * ast.scale['WFC'] * 1e3
    vy = tab2['fit_vy'] * ast.scale['WFC'] * 1e3

    py.figure(1)
    py.clf()
    q = py.quiver(tab2['x_2005_F814W'], tab2['y_2005_F814W'], vx, vy, scale=1e2)
    py.quiverkey(q, 0.95, 0.85, 5, '5 mas/yr', color='red', labelcolor='red')
    py.savefig(work_dir + '50.ALIGN_KS2/plots/vec_diffs' + catalog_suffix + '.png')

    py.close(3)
    py.figure(3, figsize=(8,6))
    py.clf()
    nz = mcolors.Normalize()
    nz.autoscale(tab2['m_2005_F814W'])
    q = py.quiver(tab2['x_2005_F814W'], tab2['y_2005_F814W'], vx, vy, scale=1e2,
                  color=py.cm.gist_stern(nz(tab2['m_2005_F814W'])))
    py.quiverkey(q, 0.95, 0.85, 5, '5 mas/yr', color='black', labelcolor='black')
    py.axis('equal')
    cax, foo = colorbar.make_axes(py.gca(), orientation='vertical', fraction=0.2, pad=0.04)
    cb = colorbar.ColorbarBase(cax, cmap=py.cm.gist_stern, norm=nz,
                               orientation='vertical')
    cb.set_label('F814W')
    py.savefig(work_dir + '50.ALIGN_KS2/plots/vec_diffs_color' + catalog_suffix + '.png')

        
    py.figure(2)
    py.clf()
    py.plot(vx, vy, 'k.', ms=2)
    lim = 5
    py.axis([-lim, lim, -lim, lim])
    py.xlabel('X Proper Motion (mas/yr)')
    py.ylabel('Y Proper Motion (mas/yr)')
    py.savefig(work_dir + '50.ALIGN_KS2/plots/vpd' + catalog_suffix + '.png')

    py.figure(3)
    py.clf()
    nz = mcolors.Normalize()
    nz.autoscale(tab2['m_2005_F814W'])
    py.scatter(vx, vy, c=nz(tab2['m_2005_F814W']), s=5, edgecolor='',
               cmap=py.cm.gist_stern)
    py.xlabel('X Proper Motion (mas/yr)')
    py.ylabel('Y Proper Motion (mas/yr)')
    py.axis('equal')
    lim = 3.5
    py.axis([-lim, lim, -lim, lim])
    
    cax, foo = colorbar.make_axes(py.gca(), orientation='vertical', fraction=0.2, pad=0.04)
    cb = colorbar.ColorbarBase(cax, cmap=py.cm.gist_stern, norm=nz,
                               orientation='vertical')
    cb.set_label('F814W')
    py.savefig(work_dir + '50.ALIGN_KS2/plots/vpd_color' + catalog_suffix + '.png')
    

    idx = np.where((np.abs(vx) < 3) & (np.abs(vy) < 3))[0]
    print('Cluster Members (within vx < 10 mas/yr and vy < 10 mas/yr)')
    print(('   vx = {vx:6.2f} +/- {vxe:6.2f} mas/yr'.format(vx=vx[idx].mean(),
                                                           vxe=vx[idx].std())))
    print(('   vy = {vy:6.2f} +/- {vye:6.2f} mas/yr'.format(vy=vy[idx].mean(),
                                                           vye=vy[idx].std())))
    
    return

def setup_artstar_info():
    comp_dirs = {'21.KS2_2005_ART': ['01.ks2', '02.ks2', '03.ks2',
                                     '04.ks2', '05.ks2', '06.ks2', '07.ks2',
                                     '08.ks2', '09.ks2', '10.ks2',
                                     '11.ks2', '12.ks2', '13.ks2', '14.ks2'],
                 '22.KS2_2010_ART': ['01.pos1_a', '01.pos1_b', '01.pos1_c',
                                     '02.pos2_a', '02.pos2_b', '02.pos2_c',
                                     '03.pos3_a', '03.pos3_b', '03.pos3_c',
                                     '04.pos4_a', '04.pos4_b', '04.pos4_c',
                                     '01.pos1_d', '01.pos1_e', '01.pos1_f',                                    
                                     '02.pos2_d', '02.pos2_e', '02.pos2_f',                                    
                                     '03.pos3_d', '03.pos3_e', '03.pos3_f',                                    
                                     '04.pos4_d', '04.pos4_e', '04.pos4_f'],
                 '23.KS2_2013_ART': ['01.pos1_a', '01.pos1_b', '01.pos1_c',
                                     '02.pos2_a', '02.pos2_b', '02.pos2_c',
                                     '03.pos3_a', '03.pos3_b', '03.pos3_c',
                                     '04.pos4_a', '04.pos4_b', '04.pos4_c',
                                     '01.pos1_d', '01.pos1_e', '01.pos1_f',
                                     '02.pos2_d', '02.pos2_e', '02.pos2_f',
                                     '03.pos3_d', '03.pos3_e', '03.pos3_f',
                                     '04.pos4_d', '04.pos4_e', '04.pos4_f']}
    art_lists = {'21.KS2_2005_ART': ['Artstarlist_2005_1.txt',
                                     'Artstarlist_2005_2.txt',
                                     'Artstarlist_2005_3.txt',
                                     'Artstarlist_2005_4.txt',
                                     'Artstarlist_2005_5.txt',
                                     'Artstarlist_2005_6.txt',
                                     'Artstarlist_2005_7.txt',
                                     'Artstarlist_2005_8.txt',
                                     'Artstarlist_2005_9.txt',
                                     'Artstarlist_2005_10.txt',
                                     'Artstarlist_2005_11.txt',
                                     'Artstarlist_2005_12.txt',
                                     'Artstarlist_2005_13.txt',
                                     'Artstarlist_2005_14.txt'],
                 '22.KS2_2010_ART': ['Artstarlist_2010_pos1_a.txt',
                                     'Artstarlist_2010_pos1_b.txt',
                                     'Artstarlist_2010_pos1_c.txt',
                                     'Artstarlist_2010_pos2_a.txt',
                                     'Artstarlist_2010_pos2_b.txt',
                                     'Artstarlist_2010_pos2_c.txt',
                                     'Artstarlist_2010_pos3_a.txt',
                                     'Artstarlist_2010_pos3_b.txt',
                                     'Artstarlist_2010_pos3_c.txt',
                                     'Artstarlist_2010_pos4_a.txt',
                                     'Artstarlist_2010_pos4_b.txt',
                                     'Artstarlist_2010_pos4_c.txt',
                                     'Artstarlist_2010_pos1_d.txt',
                                     'Artstarlist_2010_pos1_e.txt',
                                     'Artstarlist_2010_pos1_f.txt',
                                     'Artstarlist_2010_pos2_d.txt',
                                     'Artstarlist_2010_pos2_e.txt',
                                     'Artstarlist_2010_pos2_f.txt',
                                     'Artstarlist_2010_pos3_d.txt',
                                     'Artstarlist_2010_pos3_e.txt',
                                     'Artstarlist_2010_pos3_f.txt',
                                     'Artstarlist_2010_pos4_d.txt',
                                     'Artstarlist_2010_pos4_e.txt',
                                     'Artstarlist_2010_pos4_f.txt'],
                 '23.KS2_2013_ART': ['Artstarlist_2013_pos1_a.txt',
                                     'Artstarlist_2013_pos1_b.txt',
                                     'Artstarlist_2013_pos1_c.txt',
                                     'Artstarlist_2013_pos2_a.txt',
                                     'Artstarlist_2013_pos2_b.txt',
                                     'Artstarlist_2013_pos2_c.txt',
                                     'Artstarlist_2013_pos3_a.txt',
                                     'Artstarlist_2013_pos3_b.txt',
                                     'Artstarlist_2013_pos3_c.txt',
                                     'Artstarlist_2013_pos4_a.txt',
                                     'Artstarlist_2013_pos4_b.txt',
                                     'Artstarlist_2013_pos4_c.txt',
                                     'Artstarlist_2013_pos1_d.txt',
                                     'Artstarlist_2013_pos1_e.txt',
                                     'Artstarlist_2013_pos1_f.txt',
                                     'Artstarlist_2013_pos2_d.txt',
                                     'Artstarlist_2013_pos2_e.txt',
                                     'Artstarlist_2013_pos2_f.txt',
                                     'Artstarlist_2013_pos3_d.txt',
                                     'Artstarlist_2013_pos3_e.txt',
                                     'Artstarlist_2013_pos3_f.txt',
                                     'Artstarlist_2013_pos4_d.txt',
                                     'Artstarlist_2013_pos4_e.txt',
                                     'Artstarlist_2013_pos4_f.txt']}
        
    filters = {'21.KS2_2005_ART': [1],
               '22.KS2_2010_ART': [1, 2, 3],
               '23.KS2_2013_ART': [1]}

    return comp_dirs, art_lists, filters


def call_process_ks2_artstar():
    comp_dirs, art_lists, filters = setup_artstar_info()

    # Loop through each year
    for epoch in list(comp_dirs.keys()):
        art_dirs = comp_dirs[epoch]
                
        # Loop through each position, sublist combo
        for ii in range(len(art_dirs)):
            directory = work_dir + epoch + '/' + art_dirs[ii]
            os.chdir(directory)

            filt = filters[epoch]

            # Loop through each filter            
            for ff in range(len(filt)):
                filt_string = 'F{0:d}'.format(filt[ff])
                print('Working on {0}, filter F{1}'.format(directory, filt_string))
                
                art_star_list = art_lists[epoch][ii]

                if '2005' in epoch:
                    nimfo2bar_file = 'nimfo2bar.xymeee.ks2.' + filt_string
                else:
                    nimfo2bar_file = 'nimfo2bar.xymeee.' + filt_string
                comp.process_ks2_artstar(art_star_list, nimfo2bar_file,
                                         filter_num=filt[ff])

            
    return

def recombine_artstar_subsets():
    """
    Take the subsets of the artificial star lists and put them
    back together again. We will only work with the final output
    catalogs form call_process_ks2_artstar().

    Last step is top save the catalogs in 51.ALIGN_ART/.
    """
    comp_dirs, art_lists, filters = setup_artstar_info()

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
        table_file = work_dir + epoch + '/' + subdirs[ii] + '/'
        table_file += alists[ii].replace('.txt', '')
        table_file += '_F1_joined.fits'

        t = Table.read(table_file)

        if ii == 0:
            t_2005_F814W = t
        else:
            # Modify the names (which are actually indices).
            t['name'] += N_largest
            t_2005_F814W = table.vstack((t_2005_F814W, t), join_type='exact')
            
        N_largest = t_2005_F814W['name'][-1]

    # Save output to file.
    outfile = 'artstar_2005_F814W.fits'
    outfile = work_dir + '51.ALIGN_ART/' + outfile
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
                table_file = work_dir + epoch
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
                    t_2010 = table.vstack((t_2010, t), join_type='exact')
            
                N_largest = t_2010['name'][-1]

            # Save output to file.
            outfile = 'artstar_2010_{0}_{1}.fits'.format(filter_names[filt], pos_out)
            outfile = work_dir + '51.ALIGN_ART/' + outfile
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
                table_file = work_dir + epoch
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
                    t_2013 = table.vstack((t_2013, t), join_type='exact')
            
                N_largest = t_2013['name'][-1]

            # Save output to file.
            outfile = 'artstar_2013_{0}_{1}.fits'.format(filter_names[filt], pos_out)
            outfile = work_dir + '51.ALIGN_ART/' + outfile
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
        align_dir = work_dir + '50.ALIGN_KS2/'
    else:
        align_root = 'align_a4_t'
        align_dir = work_dir + '51.ALIGN_ART/align_a4_t/'

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
        name = np.array([], dtype='S10')

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

            t = Table.read(work_dir + '51.ALIGN_ART/' + artstar_list)

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

    file_name = work_dir + '51.ALIGN_ART/art_align'
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

