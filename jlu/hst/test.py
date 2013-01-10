import numpy as np
import pylab as py
import atpy
from jlu.hst import starlists
from jlu.util import statsIter
from jlu.util import transforms
from matplotlib import transforms as mplTrans
import math


def matchup_ks2_pass1(ks2Root, ks2FilterIdx, matchupFile):
    """
    Read in ks2 output and xym2mat files.
    Matchup to search for common sources.
    Return a table with all the measurements for comparison.

    The matchup files should be listed in the exact same filter order
    as is processed in KS2 and in the brite file.
    """

    # Load the KS2 star list
    ks2 = atpy.Table(ks2Root + '.FIND_AVG_UV1_F.fits')
    suffix = '_%d' % ks2FilterIdx

    # Trim down the KS2 list to just stuff well-measured in the filter of interest.
    keepIdx = np.where((ks2['x'+suffix] != -1000) &
                       (ks2['m'+suffix] != 0) &
                       (ks2['me'+suffix] < 1))[0]
    ks2 = ks2.rows(keepIdx)

    pass1 = starlists.read_matchup(matchupFile)
    keepIdx = np.where((pass1.x != -1000) &
                       (pass1.m != 0) &
                       (pass1.me < 1))[0]
    pass1 = pass1.rows(keepIdx)

    #####
    # For each star in the matchup list (with a valid measurement), search the ks2
    # data to find whether the star is measured. If it is, then compare the measurements
    # and unceratinties.
    #####
    ks2Indices = np.zeros(len(pass1), dtype=int)
    ks2Indices += -1   # -1 indicates no match

    print 'Starting search'
    for ss in range(len(pass1)):
        dr = np.hypot( pass1.x[ss] - ks2['x_0'], pass1.y[ss] - ks2['y_0'] )
        rminIdx = dr.argmin()
            
        if dr[rminIdx] < 1:
            ks2Indices[ss] = rminIdx
                
    # Make a new table with x, y, m, xe, ye, me  First in ks2, then in pass1.
    in_p1 = np.where(ks2Indices >= 0)[0]
    in_ks2 = ks2Indices[in_p1]
            
    stars = atpy.Table()
    stars.add_column('x_ks2', ks2['x'+suffix][in_ks2])
    stars.add_column('y_ks2', ks2['y'+suffix][in_ks2])
    stars.add_column('m_ks2', ks2['m'+suffix][in_ks2])
    stars.add_column('xe_ks2', ks2['xe'+suffix][in_ks2])
    stars.add_column('ye_ks2', ks2['ye'+suffix][in_ks2])
    stars.add_column('me_ks2', ks2['me'+suffix][in_ks2])
    stars.add_column('x_pass1', pass1['x'][in_p1])
    stars.add_column('y_pass1', pass1['y'][in_p1])
    stars.add_column('m_pass1', pass1['m'][in_p1])
    stars.add_column('xe_pass1', pass1['xe'][in_p1])
    stars.add_column('ye_pass1', pass1['ye'][in_p1])
    stars.add_column('me_pass1', pass1['me'][in_p1])
    stars.table_name = ''

    stars.write('stars_ks2_pass1_f%d.fits' % ks2FilterIdx, overwrite=True)

    # Just for kicks, also produce a table of stars that WERE NOT in ks2.
    # This table will have the same format as the pass1 list.
    stars_pass1_only = pass1.where(ks2Indices == -1)
    stars_pass1_only.table_name = ''
    
    stars_pass1_only.write('stars_pass1_only_f%d.fits' % ks2FilterIdx, overwrite=True)

def plot_comparison(tableSuffix):
    """
    Load up starlists produced by matchup_ks2_pass1() and plot some parameters
    of interest. Just pass in the filter suffix (e.g. "_f0"). Plots include:

    delta-x vs. m
    delta-y vs. m
    delta-m vs. m
    uncertainties vs. m for both lists
    me vs. m for stars in ks2 vs. stars not in ks2
    """
    stars = atpy.Table('stars_ks2_pass1%s.fits' % tableSuffix)
    other = atpy.Table('stars_pass1_only%s.fits' % tableSuffix)

    print '%6d stars in ks2' % len(stars)
    print '%6d stars not found in ks2' % len(other)

    # Plot delta-x vs. m
    dx = stars.x_ks2 - stars.x_pass1
    dxe = stars.xe_ks2

    #py.close('all')

    py.figure(1)
    py.clf()
    ax1 = py.subplot(2, 1, 1)
    py.plot(stars.m_pass1, dx, 'k.', ms=2)
    py.ylabel('dx (pix)')
    py.ylim(-0.3, 0.3)
    py.axhline(y=0, color='r', linewidth=2)
    py.title('KS2 - One Pass')

    ax2 = py.subplot(2, 1, 2, sharex=ax1)
    py.plot(stars.m_pass1, dx/dxe, 'k.', ms=2)
    py.ylabel('dx/dxe (sigma)')
    py.ylim(-3, 3)
    py.axhline(y=0, color='r', linewidth=2)
    py.xlabel('Magnitude')
    py.xlim(-13, -1)
    py.savefig('plot_dx_m%s.png' % tableSuffix)

    idx = np.where(stars.m_ks2 < -5)[0]
    print 'Mean dx = %7.3f +/- %7.3f (pix)' % \
        (statsIter.mean(dx[idx], lsigma=4, hsigma=4, iter=5),
         statsIter.std(dx[idx], lsigma=4, hsigma=4, iter=5))
    
    # Plot delta-y vs. m
    dy = stars.y_ks2 - stars.y_pass1
    dye = stars.ye_ks2

    py.figure(2)
    py.clf()
    ax1 = py.subplot(2, 1, 1)
    py.plot(stars.m_pass1, dy, 'k.', ms=2)
    py.ylabel('dy (pix)')
    py.ylim(-0.3, 0.3)
    py.axhline(y=0, color='r', linewidth=2)
    py.title('KS2 - One Pass')

    ax2 = py.subplot(2, 1, 2, sharex=ax1)
    py.plot(stars.m_pass1, dy/dye, 'k.', ms=2)
    py.ylabel('dy/dye (sigma)')
    py.ylim(-3, 3)
    py.axhline(y=0, color='r', linewidth=2)
    py.xlabel('Magnitude')
    py.xlim(-13, -1)

    py.savefig('plot_dy_m%s.png' % tableSuffix)
    
    idx = np.where(stars.m_ks2 < -5)[0]
    print 'Mean dy = %7.3f +/- %7.3f (pix)' % \
        (statsIter.mean(dy[idx], lsigma=4, hsigma=4, iter=5),
         statsIter.std(dy[idx], lsigma=4, hsigma=4, iter=5))

    # Plot delta-m vs. m
    dm = stars.m_ks2 - stars.m_pass1
    dme = stars.me_ks2

    py.figure(3)
    py.clf()
    ax1 = py.subplot(2, 1, 1)
    py.plot(stars.m_pass1, dm, 'k.', ms=2)
    py.ylabel('dm (mag)')
    py.ylim(-0.3, 0.3)
    py.axhline(y=0, color='r', linewidth=2)
    py.title('KS2 - One Pass')

    ax2 = py.subplot(2, 1, 2, sharex=ax1)
    py.plot(stars.m_pass1, dm/dme, 'k.', ms=2)
    py.ylabel('dm/dme (sigma)')
    py.ylim(-3, 3)
    py.axhline(y=0, color='r', linewidth=2)
    py.xlabel('Magnitude')
    py.xlim(-13, -1)

    py.savefig('plot_dm_m%s.png' % tableSuffix)


    # Plot differences in errors vs. m
    py.close(4)
    py.figure(4, figsize=(9,12))
    py.clf()
    py.subplots_adjust(left=0.12, bottom=0.07, top=0.95)
    ax1 = py.subplot(3, 1, 1)
    py.plot(stars.m_ks2, stars.xe_ks2 - stars.xe_pass1, 'k.', ms=2)
    py.ylabel('xe diff')
    py.ylim(-0.2, 0.2)
    py.axhline(y=0, color='r', linewidth=2)
    py.title('KS2 - One Pass')

    ax2 = py.subplot(3, 1, 2, sharex=ax1)
    py.plot(stars.m_ks2, stars.ye_ks2 - stars.ye_pass1, 'k.', ms=2)
    py.ylabel('ye diff')
    py.ylim(-0.2, 0.2)
    py.axhline(y=0, color='r', linewidth=2)

    ax3 = py.subplot(3, 1, 3, sharex=ax1)
    py.plot(stars.m_ks2, stars.me_ks2 - stars.me_pass1, 'k.', ms=2)
    py.ylabel('me diff')
    py.ylim(-0.2, 0.2)
    py.axhline(y=0, color='r', linewidth=2)

    py.xlabel('Magnitude')
    py.xlim(-13, -1)

    py.savefig('plot_compare_errors%s.png' % tableSuffix)

    
    # Plot stars that are NOT in KS2 vs. those that are.
    py.figure(5)
    py.clf()

    py.semilogy(stars.m_pass1, stars.me_pass1, 'k.', label='In KS2', ms=2)
    py.semilogy(other.m, other.me, 'r.', label='Not In KS2', ms=3)
    py.legend(numpoints=1, loc='upper left')
    py.xlabel('Magnitude')
    py.xlim(-23, -1)
    py.ylabel('Magnitude Error')
    py.ylim(0.004, 0.1)
    py.savefig('plot_m_me_others%s.png' % tableSuffix)

def fit_transform_matfile(matfile):
    data = atpy.Table(matfile, type='ascii')

    points_in = np.array([data.col1, data.col2]).T
    points_out = np.array([data.col3, data.col4]).T

    params, perrors = transforms.fit_affine2d_noshear(points_in, points_out)

    print params
    print perrors

    trans = mplTrans.Affine2D()
    trans.rotate_deg(params['angle'])
    trans.scale(params['scale'])
    trans.translate(params['transX'], params['transY'])

    points_new = trans.transform(points_in)

    diff = points_out - points_new
    print diff


def plot_pos_diff_xym1mat(xym1mat_cfile, scale=8, errMax=0.05):
    """
    Send in a match_c.txt file from xym1mat (3rd file) that has been trimmed
    of all bogus entries. Then plot of the differences in the positions
    from the 1st and 2nd starlists that went into the list matching process.
    This allows me to see any large scale systematics due to residual distortion
    or alignment errors.

    This assumes that the input starlists were in the XYMEEE format. We will
    use the frame #2 astrometric errors to trim out bad points.
    """
    t = atpy.Table(xym1mat_cfile, type='ascii')

    if errMax != None:
        x2e = t.col17
        y2e = t.col18

        err = np.hypot(x2e, y2e)

        origLen = len(t)
        t = t.where(err < errMax)
        newLen = len(t)

        print 'Trimmed %d of %d sources with errors > %.2f pix' % \
            (origLen-newLen, origLen, errMax)
        
    x1 = t.col8
    y1 = t.col9
    x2 = t.col11
    y2 = t.col12
    dx = x1 - x2
    dy = y1 - y2


    py.clf()
    q = py.quiver(x1, y1, dx, dy, scale=scale)
    py.quiverkey(q, 0.9, 0.95, 0.1, '0.1 pix', coordinates='axes', color='red')
    py.xlabel('X (pix)')
    py.ylabel('Y (pix)')
    py.savefig('pos_diff_vecs_' + xym1mat_cfile.replace('txt', 'png'))
