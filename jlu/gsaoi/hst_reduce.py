"""
Research Note: Create HST astrometric catalog for NGC 1851.
Working Directory: /u/jlu/data/HST/ngc1851/reduce_2014_07_03/
"""
import math
import atpy
import pylab as py
import numpy as np
from jlu.hst import images
from jlu.hst import astrometry as ast
import glob
from matplotlib import colors
from jlu.util import statsIter
import pdb
import os
from hst_flystar import reduce as flystar
from hst_flystar import starlists
from astropy.table import Table

workDir = '/u/jlu/data/HST/ngc1851/reduce_2014_07_03/'
codeDir = '/u/jlu/code/fortran/hst/'

# Load this variable with outputs from calc_years()
year =  2010.922

def run_img2xym_acswfc():
    """
    Run img2xym on ACS WFC data in the specified directory, <dir>. There is a specific
    directory structure that is expected within <dir>:

    00.DATA/
    01.XYM/
    """
    os.chdir(workDir + '/2010_F814W/01.XYM')

    program = 'img2xymrduv'

    ## Arguments include:
    # hmin - dominance of peak. Something like the minimum SNR
    # of a peak w.r.t. background.
    hmin = 5
    # fmin - minumum peak flux above the background
    fmin = 500
    # pmax - maximum flux allowed (absolute)
    pmax = 99999
    # psf - the library
    psf = codeDir + 'PSFs/PSFSTD_ACSWFC_F814W_SM3.fits'

    ## Files to operate on:
    dataDir = '../00.DATA/*flt.fits'

    cmd_tmp = '{program} {hmin} {fmin} {pmax} {psf} {dataDir}'
    cmd = cmd_tmp.format(program=program, hmin=hmin, fmin=fmin,
                          pmax=pmax, psf=psf, dataDir=dataDir)

    try:
        os.system(cmd)
    finally:
        os.chdir('../../')
    
def xym_acswfc_pass1():
    """
    Match and align all of the exposures. Make a new star list (requiring 6 out of
    7 images). Also make sure the new starlist has only positive pxel values.
    """
    year = '2010'
    filt = 'F814W'

    flystar.xym2mat('ref1', year, filt, camera='f5 c5', mag='m-18,-14', clobber=True)
    flystar.xym2bar('ref1', year, filt, camera='f5 c5', Nepochs=6, clobber=True)

    xym_dir = '{0}/2010_F814W/01.XYM/'.format(workDir)
    starlists.make_matchup_positive(xym_dir + 'MATCHUP.XYMEEE.ref1')
    
def xym_acswfc_pass2():
    """
    Re-do the alignment with the new positive master file. Edit IN.xym2mat to 
    change 00 epoch to use the generated matchup file with f5, c0 
    (MATCHUP.XYMEEE.01.all.positive). Make sure to remove all the old MAT.* and 
    TRANS.xym2mat files because there is a big shift in the transformation from 
    the first time we ran it.
    """
    year = '2010'
    filt = 'F814W'

    xym_dir = '{0}_{1}/01.XYM/'.format(year, filt)

    flystar.xym2mat('ref2', year, filt, camera='f5 c5', mag='m-14,-13',
                    ref='MATCHUP.XYMEEE.ref1.positive', ref_mag='m-14,-13',
                    ref_camera='f5 c0', clobber=True)
    flystar.xym2bar('ref2', year, filt, Nepochs=6, zeropoint='', 
                    camera='f5 c5', clobber=True)


def xym_acswfc_pass3():
    """
    Re-do the alignment with the new positive master file. Edit IN.xym2mat to 
    change 00 epoch to use the generated matchup file with f5, c0 
    (MATCHUP.XYMEEE.01.all.positive). Make sure to remove all the old MAT.* and 
    TRANS.xym2mat files because there is a big shift in the transformation from 
    the first time we ran it.
    """
    year = '2010'
    filt = 'F814W'

    xym_dir = '{0}_{1}/01.XYM/'.format(year, filt)

    flystar.xym2mat('ref3', year, filt, camera='f5 c5', mag='m-14,-11',
                    ref='MATCHUP.XYMEEE.ref1.positive', ref_mag='m-14,-11',
                    ref_camera='f5 c0', clobber=True)
    flystar.xym2bar('ref3', year, filt, Nepochs=6, zeropoint='', 
                    camera='f5 c5', clobber=True)



def calc_years():
    """
    Calculate the epoch for each data set.
    """
    dataDir = '{0}/00.DATA/'.format(workDir)

    epoch = images.calc_mean_year(glob.glob(dataDir + '*_flt.fits'))

    print('{0:8.3f}'.format(epoch))
        
    
def make_master_lists():
    """
    Trim the ref5 master lists for each filter down to just stars with
    proper motions within 1 mas/yr of the cluster motion.
    """
    # Read in matched and aligned star lists from the *.ref5 analysis.
    # Recall these data sets are in the F814W reference frame with a 50 mas plate scale.
    t2005_814 = starlists.read_matchup(workDir + '02.PMA/MATCHUP.XYMEEE.F814W.ref5')
    t2010_125 = starlists.read_matchup(workDir + '02.PMA/MATCHUP.XYMEEE.F125W.ref5')
    t2010_139 = starlists.read_matchup(workDir + '02.PMA/MATCHUP.XYMEEE.F139M.ref5')
    t2010_160 = starlists.read_matchup(workDir + '02.PMA/MATCHUP.XYMEEE.F160W.ref5')

    scale = 50.0 # mas per pixel

    # Trim down to only those stars that are detected in both epochs.
    # Also make cuts on astrometric/photometric errors, etc.
    # We only want the well measured stars in this analysis.
    perrLim814 = 1.0 / scale
    perrLim125 = 4.0 / scale
    merrLim814 = 0.05
    merrLim125 = 0.1
    
    cond = ((t2005_814.m != 0) & (t2010_125.m != 0) &
            (t2010_139.m != 0) & (t2010_160.m != 0) &
            (t2005_814.xe < perrLim814) & (t2005_814.ye < perrLim814) &
            (t2010_125.xe < perrLim125) & (t2010_125.ye < perrLim125) &
            (t2010_139.xe < perrLim125) & (t2010_139.ye < perrLim125) &
            (t2010_160.xe < perrLim125) & (t2010_160.ye < perrLim125) &
            (t2005_814.me < merrLim814) & (t2010_125.me < merrLim125) &
            (t2010_139.me < merrLim125) & (t2010_160.me < merrLim125))

    t2005_814 = t2005_814.where(cond)
    t2010_125 = t2010_125.where(cond)
    t2010_139 = t2010_139.where(cond)
    t2010_160 = t2010_160.where(cond)

    # Calculate proper motions
    dt = years['2010_F125W'] - years['2005_F814W']
    dx = t2010_125.x - t2005_814.x
    dy = t2010_125.y - t2005_814.y
    pmx = dx * scale / dt
    pmy = dy * scale / dt
    pm = np.hypot(pmx, pmy)

    t2005_814.add_column('pmx', pmx)
    t2005_814.add_column('pmy', pmy)
    t2005_814.add_column('pm', pm)
    
    # Trim down to a 1 mas/yr radius
    pmCut = 1.0
    idx2 = np.where(pm < pmCut)[0]

    t2005_814 = t2005_814.where(idx2)
    t2010_125 = t2010_125.where(idx2)
    t2010_139 = t2010_139.where(idx2)
    t2010_160 = t2010_160.where(idx2)

    _o814 = open(workDir + '02.PMA/MASTER.F814W.ref5', 'w')
    _o125 = open(workDir + '02.PMA/MASTER.F125W.ref5', 'w')
    _o139 = open(workDir + '02.PMA/MASTER.F139M.ref5', 'w')
    _o160 = open(workDir + '02.PMA/MASTER.F160W.ref5', 'w')

    ofmt = '{0:10.4f} {1:10.4f} {2:8.4f} {3:10.4f} {4:10.4f} {5:8.4f} {6}\n'
    for ii in range(len(t2005_814)):
        _o814.write(ofmt.format(t2005_814.x[ii], t2005_814.y[ii], t2005_814.m[ii],
                                t2005_814.xe[ii], t2005_814.ye[ii], t2005_814.me[ii],
                                t2005_814.name[ii]))
        _o125.write(ofmt.format(t2010_125.x[ii], t2010_125.y[ii], t2010_125.m[ii],
                                t2010_125.xe[ii], t2010_125.ye[ii], t2010_125.me[ii],
                                t2010_125.name[ii]))
        _o139.write(ofmt.format(t2010_139.x[ii], t2010_139.y[ii], t2010_139.m[ii],
                                t2010_139.xe[ii], t2010_139.ye[ii], t2010_139.me[ii],
                                t2010_139.name[ii]))
        _o160.write(ofmt.format(t2010_160.x[ii], t2010_160.y[ii], t2010_160.m[ii],
                                t2010_160.xe[ii], t2010_160.ye[ii], t2010_160.me[ii],
                                t2010_160.name[ii]))

    _o814.close()
    _o125.close()
    _o139.close()
    _o160.close()
        

def make_catalog():
    """
    Combine 4 MATCHUP files into a single FITS file catalog.  Stars have to be
    detected in all 4 filters in order to be included.

    Output is wd1_catalog.fits
    """
    files = ['MATCHUP.XYMEEE.F814W.ks2_all',
             'MATCHUP.XYMEEE.F160W.ks2_all',
             'MATCHUP.XYMEEE.F139M.ks2_all',
             'MATCHUP.XYMEEE.F125W.ks2_all']

    suffixes = ['814', '160', '139', '125']

    print 'Reading in star lists.'
    final = None
    for ff in range(len(files)):
        tab = atpy.Table(files[ff], type='ascii')
        tab.rename_column('col1', 'x_'+suffixes[ff])
        tab.rename_column('col2', 'y_'+suffixes[ff])
        tab.rename_column('col3', 'm_'+suffixes[ff])
        tab.rename_column('col4', 'xe_'+suffixes[ff])
        tab.rename_column('col5', 'ye_'+suffixes[ff])
        tab.rename_column('col6', 'me_'+suffixes[ff])
        tab.rename_column('col9', 'cntPos_'+suffixes[ff])
        tab.rename_column('col10', 'cntMag_'+suffixes[ff])
        tab.rename_column('col12', 'name')

        tab.remove_columns(['col7', 'col8', 'col11', 'col13', 'col14'])

        if final == None:
            final = tab
        else:
            final.add_column('x_'+suffixes[ff], tab['x_'+suffixes[ff]])
            final.add_column('y_'+suffixes[ff], tab['y_'+suffixes[ff]])
            final.add_column('m_'+suffixes[ff], tab['m_'+suffixes[ff]])
            final.add_column('xe_'+suffixes[ff], tab['xe_'+suffixes[ff]])
            final.add_column('ye_'+suffixes[ff], tab['ye_'+suffixes[ff]])
            final.add_column('me_'+suffixes[ff], tab['me_'+suffixes[ff]])
            final.add_column('cntPos_'+suffixes[ff], tab['cntPos_'+suffixes[ff]])
            final.add_column('cntMag_'+suffixes[ff], tab['cntMag_'+suffixes[ff]])


    # Trim down the table to only those stars in all filters.
    print 'Trimming stars not in all 4 filters.'
    final2 = final.where((final.m_814 != 0) & (final.m_160 != 0) &
                         (final.m_139 != 0) & (final.m_125 != 0) &
                         (final.xe_814 < 9) & (final.xe_160 < 9) &
                         (final.xe_139 < 9) & (final.xe_125 < 9))
    final2.table_name = 'wd1_catalog'

    final2.write('wd1_catalog.fits')
  

