"""
Research Note: Proper Motion Test (2013-03)
Working Directory: /u/jlu/data/Wd1/hst/reduce_2013_03_12/
"""
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

workDir = '/u/jlu/data/Wd1/hst/reduce_2013_03_12/'

# Load this variable with outputs from calc_years()
years = {'2005_F814W': 2005.485,
         '2010_F125W': 2010.652,
         '2010_F139M': 2010.652,
         '2010_F160W': 2010.652}

def plot_vpd_across_field(nside=4, interact=False):
    """
    Plot the VPD at different field positions so we can see if there are
    systematic discrepancies due to residual distortions.
    """
    # Read in matched and aligned star lists from the *.ref5 analysis.
    # Recall these data sets are in the F814W reference frame with a 50 mas plate scale.
    t2005 = starlists.read_matchup(workDir + '02.PMA/MATCHUP.XYMEEE.F814W.ref5')
    t2010 = starlists.read_matchup(workDir + '02.PMA/MATCHUP.XYMEEE.F125W.ref5')

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
    print(out.format(xlo, xhi, ylo, yhi, pmx_all, 0.0, pmy_all, 0.0, len(idx2)))

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
    py.savefig(out.format(workDir))
    
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
                raw_input(out.format(xlo_box, xhi_box, ylo_box, yhi_box,
                                     xmean, xmean_err, ymean, ymean_err, len(idx2)))
            else:
                print(out.format(xlo_box, xhi_box, ylo_box, yhi_box,
                                 xmean, xmean_err, ymean, ymean_err, len(idx2)))


    if interact:
        out = '{0}/plots/vpd_grid_nside{1}.png'
        py.savefig(out.format(workDir, nside))

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
    py.savefig(out.format(workDir, nside))

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
    py.savefig(out.format(workDir, nside))

def calc_years():
    """
    Calculate the epoch for each data set.
    """
    years = ['2005', '2010', '2010', '2010']
    filts = ['F814W', 'F125W', 'F139M', 'F160W']
    
    for ii in range(len(years)):
        dataDir = '{0}/{1}_{2}/00.DATA/'.format(workDir, years[ii], filts[ii])

        epoch = images.calc_mean_year(glob.glob(dataDir + '*_flt.fits'))

        print('{0}_{1} at {2:8.3f}'.format(years[ii], filts[ii], epoch))
        input('Continue?')
    
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
        
def cross_match_ks2():
    """
    Read in the IR and optical catalogs produced from our ks2 analysis.
    Cross-match all the stars (remember that they are all already in the same
    reference frame; but there is motion). Make a new table with the combined
    information.
    """
    irFile = '/u/jlu/data/Wd1/hst/reduce_2013_03_12/12.KS2_2010/LOGR_catalog.fits'
    opFile = '/u/jlu/data/Wd1/hst/reduce_2013_03_12/13.KS2_2005/LOGR_catalog.fits'

    irTable = atpy.Table(irFile)
    opTable = atpy.Table(opFile)

    # We will add the optical data to the infrared
    irTable.add_column('x_4', np.zeros(len(irTable), dtype=float))
    irTable.add_column('y_4', np.zeros(len(irTable), dtype=float))
    irTable.add_column('m_4', np.zeros(len(irTable), dtype=float))
    irTable.add_column('xe_4', np.zeros(len(irTable), dtype=float))
    irTable.add_column('ye_4', np.zeros(len(irTable), dtype=float))
    irTable.add_column('me_4', np.zeros(len(irTable), dtype=float))
    irTable.add_column('fsrc_4', np.zeros(len(irTable), dtype=float))

    # Set initial values for missing sources.
    irTable.m_4 = np.inf
    irTable.me_4 = np.inf
    irTable.xe_4 = 99999.0
    irTable.ye_4 = 99999.0
    
    matchRadius = 2.0

    outDir = '/u/jlu/data/Wd1/hst/reduce_2013_03_12/'
    _confused = open(outDir + 'wd1_confused.txt', 'w')
    _nameMatch = open(outDir + 'wd1_match_names.txt', 'w')

    _confused.write('{0:13s}  {1}\n'.format('Infrared', 'OpticalCandidates'))
    _nameMatch.write('{0:13s}  {1:13s}\n'.format('Infrared', 'Optical'))
    
    # Loop through each IR star and find an optical match (by position).
    for ii in range(len(irTable)):
        if (ii % 1000) == 0:
            print('Working on IR star {0}'.format(ii))
        
        # Use the x_0, y_0 positions
        dr = np.hypot(irTable.x_0[ii] - opTable.x_0, irTable.y_0[ii] - opTable.y_0)
        idx = np.where(dr < matchRadius)[0]

        if len(idx) > 1:
            candidates = ','.join(opTable.name[idx])
            _confused.write('{0:13s}  {1}\n'.format(irTable.name[ii], candidates))

        if len(idx) == 1:
            _nameMatch.write('{0:13s}  {1:13s}\n'.format(irTable.name[ii],
                                                         opTable.name[idx[0]]))

            irTable.x_4[ii] = opTable.x_1[idx[0]]
            irTable.y_4[ii] = opTable.y_1[idx[0]]
            irTable.m_4[ii] = opTable.m_1[idx[0]]
            irTable.xe_4[ii] = opTable.xe_1[idx[0]]
            irTable.ye_4[ii] = opTable.ye_1[idx[0]]
            irTable.me_4[ii] = opTable.me_1[idx[0]]
            irTable.fsrc_4[ii] = opTable.fsrc_1[idx[0]]
            
    irTable.table_name = 'Wd1'
    irTable.write(outDir + 'wd1_catalog.fits')

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
  

def check_vpd_ks2_astrometry():
    """
    Check the VPD and quiver plots for our KS2-extracted, re-transformed astrometry.
    """
    catFile = workDir + '20.KS2_PMA/wd1_catalog.fits'
    tab = atpy.Table(catFile)

    good = (tab.xe_160 < 0.05) & (tab.ye_160 < 0.05) & \
        (tab.xe_814 < 0.05) & (tab.ye_814 < 0.05) & \
        (tab.me_814 < 0.05) & (tab.me_160 < 0.05)

    tab2 = tab.where(good)

    dx = (tab2.x_160 - tab2.x_814) * ast.scale['WFC'] * 1e3
    dy = (tab2.y_160 - tab2.y_814) * ast.scale['WFC'] * 1e3

    py.clf()
    q = py.quiver(tab2.x_814, tab2.y_814, dx, dy, scale=5e2)
    py.quiverkey(q, 0.95, 0.85, 5, '5 mas', color='red', labelcolor='red')
    py.savefig(workDir + '20.KS2_PMA/vec_diffs_ks2_all.png')

    py.clf()
    py.plot(dy, dx, 'k.', ms=2)
    lim = 30
    py.axis([-lim, lim, -lim, lim])
    py.xlabel('Y Proper Motion (mas)')
    py.ylabel('X Proper Motion (mas)')
    py.savefig(workDir + '20.KS2_PMA/vpd_ks2_all.png')

    idx = np.where((np.abs(dx) < 10) & (np.abs(dy) < 10))[0]
    print('Cluster Members (within dx < 10 mas and dy < 10 mas)')
    print('   dx = {dx:6.2f} +/- {dxe:6.2f} mas'.format(dx=dx[idx].mean(),
                                                        dxe=dx[idx].std()))
    print('   dy = {dy:6.2f} +/- {dye:6.2f} mas'.format(dy=dy[idx].mean(),
                                                        dye=dy[idx].std()))
    

def map_of_errors():
    t = atpy.Table(workDir + '20.KS2_PMA/wd1_catalog.fits')

    xbins = np.arange(0, 4251, 250)
    ybins = np.arange(0, 4200, 250)

    xb2d, yb2d = np.meshgrid(xbins, ybins)

    xe_mean = np.zeros(xb2d.shape, dtype=float)
    ye_mean = np.zeros(yb2d.shape, dtype=float)
    me_mean = np.zeros(yb2d.shape, dtype=float)
    xe_std = np.zeros(xb2d.shape, dtype=float)
    ye_std = np.zeros(yb2d.shape, dtype=float)
    me_std = np.zeros(yb2d.shape, dtype=float)

    for xx in range(len(xbins)-1):
        for yy in range(len(ybins)-1):
            idx = np.where((t.x_160 > xbins[xx]) & (t.x_160 <= xbins[xx+1]) &
                           (t.y_160 > ybins[yy]) & (t.y_160 <= ybins[yy+1]) &
                           (t.xe_160 < 0.2) & (t.ye_160 < 0.2))[0]

            if len(idx) > 0:
                xe_mean[yy, xx] = statsIter.mean(t.xe_160[idx], hsigma=3, iter=5)
                ye_mean[yy, xx] = statsIter.mean(t.ye_160[idx], hsigma=3, iter=5)
                me_mean[yy, xx] = statsIter.mean(t.me_160[idx], hsigma=3, iter=5)
                xe_std[yy, xx] = statsIter.std(t.xe_160[idx], hsigma=3, iter=5)
                ye_std[yy, xx] = statsIter.std(t.ye_160[idx], hsigma=3, iter=5)
                me_std[yy, xx] = statsIter.std(t.me_160[idx], hsigma=3, iter=5)


    py.close('all')
    py.figure(1, figsize=(12, 6))
    py.clf()
    py.subplots_adjust(left=0.05, bottom=0.05)
    py.subplot(1, 2, 1)
    py.imshow(xe_mean, extent=[xbins[0], xbins[-1], ybins[0], ybins[-1]],
              vmin=0.01, vmax=0.07)
    py.colorbar(orientation='horizontal')
    py.title('X Error Mean')

    py.subplot(1, 2, 2)
    py.imshow(xe_std, extent=[xbins[0], xbins[-1], ybins[0], ybins[-1]],
              vmin=0.01, vmax=0.07)
    py.colorbar(orientation='horizontal')
    py.title('X Error Std')


    py.figure(2, figsize=(12, 6))
    py.clf()
    py.subplots_adjust(left=0.05, bottom=0.05)
    py.subplot(1, 2, 1)
    py.imshow(ye_mean, extent=[xbins[0], xbins[-1], ybins[0], ybins[-1]],
              vmin=0.01, vmax=0.07)
    py.colorbar(orientation='horizontal')
    py.title('Y Error Mean')

    py.subplot(1, 2, 2)
    py.imshow(ye_std, extent=[xbins[0], xbins[-1], ybins[0], ybins[-1]],
              vmin=0.01, vmax=0.07)
    py.colorbar(orientation='horizontal')
    py.title('Y Error Std')




    py.figure(3, figsize=(12, 6))
    py.clf()
    py.subplots_adjust(left=0.05, bottom=0.05)
    py.subplot(1, 2, 1)
    py.imshow(me_mean, extent=[xbins[0], xbins[-1], ybins[0], ybins[-1]],
              vmin=0.01, vmax=0.07)
    py.colorbar(orientation='horizontal')
    py.title('M Error Mean')

    py.subplot(1, 2, 2)
    py.imshow(me_std, extent=[xbins[0], xbins[-1], ybins[0], ybins[-1]],
              vmin=0.01, vmax=0.07)
    py.colorbar(orientation='horizontal')
    py.title('M Error Std')

def remake_loga_input(loga_file, artstar_file):
    """
    This makes an input.xym file from a LOGA.INPUT file and the ARTSTAR_a?.XYM file.
    """
    # Load up the input file (ARTSTAR) as a table... we are going to sort 
    # this at the end to give the same order as in the LOGA.INPUT file.
    print 'Loading Table: {0}'.format(loga_file)
    astar = atpy.Table(artstar_file, type='ascii')

    # This will contain the indices for re-sorting the ARTSTAR file
    # to match the LOGA order.
    sidx = np.ones(len(astar)) * -1

    # This one is already sorted correctly (same as LOGA)
    names = np.zeros(len(astar), dtype='S7')

    # Open the LOGA.INPUT file
    f_loga = open(loga_file, 'r')

    # Loop through each line, get the name of the star, convert
    # to an index into astar.
    aa = 0
    for line in f_loga:
        if line.startswith('#'):
            continue

        fields = line.split()
        names[aa] = fields[16]

        # Parse the name to get the line index in ARTSTAR file.
        # First star name is 1 (not 0).
        sidx[aa] = int(names[aa][1:].lstrip("0")) - 1

        aa += 1

    # Get rid of any stars that weren't succesfully planted (e.g. not in LOGA.INPUT)
    good = np.where(sidx >= 0)
    sidx = sidx[good]
    names = names[good]
    msg = 'Dropped {0} of {1} ARTSTAR entries that are not in LOGA.INPUT'
    print msg.format(len(astar) - len(sidx), len(astar))
 
    # Make a new table that has the contents of ARTSTAR sorted
    # in the same order as LOGA.INPUT.
    astar_new = astar.rows(sidx)
    astar_new.rename_column('col1', 'x')
    astar_new.rename_column('col2', 'y')
    for ii in range(2, len(astar.columns)):
        astar_new.rename_column('col{0}'.format(ii+1), 'm{0}'.format(ii-1))

    astar_new.add_column('name', names)
    astar_new.table_name = ''

    # Write to an output file.
    astar_new.write(loga_file + '_mag.fits', overwrite=True)
        
        
def make_completeness_table(loga_mag_file, matchup_files, filt_names=None):
    """
    Read in the output of KS2 (post processed with xym2mat and xym2bar)
    and match it up with the input planted stars.

    Input
    ---------
    loga_mag_file : string
        The name of the LOGA_MAGS.INPUT.fits file (produced by remake_loga_input()

    matchup_files : list
        A list of matchup files that corresponds to the filters in the LOGA file.
        The matchup files are assumed to be in the same order as the LOGA magnitude
        columns.
    """

    # Read files
    loga = atpy.Table(loga_mag_file)
    mat = [starlists.read_matchup(mat_file) for mat_file in matchup_files]

    # Figure out the number of filters... make sure the match in input/output
    num_filt = len(matchup_files)
    num_filt2 = len(loga.columns) - 3

    if num_filt != num_filt2:
        print 'Filter mismatch: {0} in input, {1} in output'.format()
        return

    # First modify the column names in loga to reflect that these are
    # input values
    loga.rename_column('x', 'x_in')
    loga.rename_column('y', 'y_in')

    for ff in range(num_filt):
        print 'Processing Filter #{0}'.format(ff+1)
        if filt_names != None:
            filt = '_{0}'.format(filt_names[ff])
        else:
            filt = '_{0}'.format(ff+1)

        # Rename the input magnitude columns
        old_col_name = 'm{0}'.format(ff+1)
        new_col_name = 'm_in' + filt
        loga.rename_column(old_col_name, new_col_name)

        # Add the new columns to the table for all of the output data.
        loga.add_column('x_out' + filt, mat[ff].x)
        loga.add_column('y_out' + filt, mat[ff].y)
        loga.add_column('m_out' + filt, mat[ff].m)
        loga.add_column('xe_out' + filt, mat[ff].xe)
        loga.add_column('ye_out' + filt, mat[ff].ye)
        loga.add_column('me_out' + filt, mat[ff].me)
    

    loga.table_name = ''
    outfile_name = os.path.dirname(loga_mag_file) + 'completeness_matched.fits'
    loga.write(outfile_name)

    return loga


def ir_completeness_all():
    root_dir = workDir + '22.KS2_ART_2010/'

    # This will be the final output table with EVERYTHING
    comp_all = None

    # Collect info from the 5 subdirs
    for ss in range(5):
        directory = '{0}ks2_a{1}/'.format(root_dir, ss+1)

        artstar_file = directory + 'ARTSTAR_a{0}.XYMMM'.format(ss+1)
        loga_file = directory + 'LOGA.INPUT'
        remake_loga_input(loga_file, artstar_file)
        
        input_file = loga_file + '_mag.fits'
        output_files = [directory + 'F160W/MATCHUP.XYMEEE.F160W.ks2',
                        directory + 'F139M/MATCHUP.XYMEEE.F139M.ks2',
                        directory + 'F125W/MATCHUP.XYMEEE.F125W.ks2']
        filter_names = ['F160W', 'F139M', 'F125W']

        # Make and load completeness table
        comp = make_completeness_table(input_file, output_files,
                                       filt_names=filter_names)

        if comp_all == None:
            comp_all = comp
            comp_all.table_name = ''
        else:
            comp_all.append(comp)

    comp_all.write('{0}completeness_matched_all.fits'.format(root_dir))


def opt_completeness_all():
    root_dir = workDir + '23.KS2_ART_2005/'

    # This will be the final output table with EVERYTHING
    comp_all = None

    # Collect info from the 5 subdirs
    for ss in range(5):
        directory = '{0}ks2_a{1}/'.format(root_dir, ss+1)

        artstar_file = directory + 'ARTSTAR_a{0}.XYM'.format(ss+1)
        loga_file = directory + 'LOGA.INPUT'
        remake_loga_input(loga_file, artstar_file)
        
        input_file = loga_file + '_mag.fits'
        output_files = [directory + 'F814W/MATCHUP.XYMEEE.F814W.ks2']
        filter_names = ['F814W']

        # Make and load completeness table
        comp = make_completeness_table(input_file, output_files,
                                       filt_names=filter_names)

        if comp_all == None:
            comp_all = comp
            comp_all.table_name = ''
        else:
            comp_all.append(comp)

    comp_all.write('{0}completeness_matched_all.fits'.format(root_dir))

    
