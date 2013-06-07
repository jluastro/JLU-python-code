import numpy as np
import pylab as py
import atpy
from jlu.hst import starlists
from jlu.util import statsIter
from jlu.util import transforms
from matplotlib import transforms as mplTrans
import math
from gcwork import starset


def matchup_ks2_pass1(ks2Catalog, ks2FilterIdx, matchupFile, outSuffix=None):
    """
    Read in ks2 output and xym2mat files.
    Matchup to search for common sources.
    Return a table with all the measurements for comparison.
    """

    # Load the KS2 star list
    # Should be LOGR_catalo.fits produced by jlu.hst.starlists.process_ks2_output()
    #ks2 = atpy.Table(ks2Root + '.FIND_AVG_UV1_F.fits')
    ks2 = atpy.Table(ks2Catalog)  
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

    print 'Starting search for {0} stars'.format(len(pass1))
    for ss in range(len(pass1)):
        if ss % 5000 == 0:
            print 'Reached star {0}'.format(ss)
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

    if outSuffix == None:
        outSuffix = 'f{0}'.format(ksFilterIdx)

    stars.write('stars_ks2_pass1_{0}.fits'.format(outSuffix), overwrite=True)

    # Just for kicks, also produce a table of stars that WERE NOT in ks2.
    # This table will have the same format as the pass1 list.
    stars_pass1_only = pass1.where(ks2Indices == -1)
    stars_pass1_only.table_name = ''
    
    stars_pass1_only.write('stars_pass1_only_{0}.fits'.format(outSuffix), overwrite=True)

def plot_comparison(tableSuffix, posLim=0.15, magLimits=[-13,-1], errLim=0.04,
                    quiverScale=2):
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

    #####
    # Plot delta-x vs. m
    #####
    dx = stars.x_ks2 - stars.x_pass1
    dxe = stars.xe_ks2

    #py.close('all')

    py.figure(1)
    py.clf()
    ax1 = py.subplot(2, 1, 1)
    py.plot(stars.m_pass1, dx, 'k.', ms=2)
    py.ylabel('dx (pix)')
    py.ylim(-posLim, posLim)
    py.axhline(y=0, color='r', linewidth=2)
    py.title('KS2 - One Pass')

    ax2 = py.subplot(2, 1, 2, sharex=ax1)
    py.plot(stars.m_pass1, dx/dxe, 'k.', ms=2)
    py.ylabel('dx/dxe (sigma)')
    py.ylim(-3, 3)
    py.axhline(y=0, color='r', linewidth=2)
    py.xlabel('Magnitude')
    py.xlim(magLimits[0], magLimits[1])
    py.savefig('plot_dx_m%s.png' % tableSuffix)

    
    #####
    # Plot delta-y vs. m
    #####
    dy = stars.y_ks2 - stars.y_pass1
    dye = stars.ye_ks2

    py.figure(2)
    py.clf()
    ax1 = py.subplot(2, 1, 1)
    py.plot(stars.m_pass1, dy, 'k.', ms=2)
    py.ylabel('dy (pix)')
    py.ylim(-posLim, posLim)
    py.axhline(y=0, color='r', linewidth=2)
    py.title('KS2 - One Pass')

    ax2 = py.subplot(2, 1, 2, sharex=ax1)
    py.plot(stars.m_pass1, dy/dye, 'k.', ms=2)
    py.ylabel('dy/dye (sigma)')
    py.ylim(-3, 3)
    py.axhline(y=0, color='r', linewidth=2)
    py.xlabel('Magnitude')
    py.xlim(magLimits[0], magLimits[1])

    py.savefig('plot_dy_m%s.png' % tableSuffix)
    
    #####
    # Plot delta-m vs. m
    #####
    dm = stars.m_ks2 - stars.m_pass1
    dme = stars.me_ks2

    py.figure(3)
    py.clf()
    ax1 = py.subplot(2, 1, 1)
    py.plot(stars.m_pass1, dm, 'k.', ms=2)
    py.ylabel('dm (mag)')
    py.ylim(-0.1, 0.1)
    py.axhline(y=0, color='r', linewidth=2)
    py.title('KS2 - One Pass')

    ax2 = py.subplot(2, 1, 2, sharex=ax1)
    py.plot(stars.m_pass1, dm/dme, 'k.', ms=2)
    py.ylabel('dm/dme (sigma)')
    py.ylim(-3, 3)
    py.axhline(y=0, color='r', linewidth=2)
    py.xlabel('Magnitude')
    py.xlim(magLimits[0], magLimits[1])

    py.savefig('plot_dm_m%s.png' % tableSuffix)


    #####
    # Plot differences in errors vs. m
    #####
    dxerr = stars.xe_ks2 - stars.xe_pass1
    dyerr = stars.ye_ks2 - stars.ye_pass1
    dmerr = stars.me_ks2 - stars.me_pass1
    
    py.close(4)
    py.figure(4, figsize=(9,12))
    py.clf()
    py.subplots_adjust(left=0.12, bottom=0.07, top=0.95)
    ax1 = py.subplot(3, 1, 1)
    py.plot(stars.m_ks2, dxerr, 'k.', ms=2)
    py.ylabel('xe diff')
    py.ylim(-0.05, 0.05)
    py.axhline(y=0, color='r', linewidth=2)
    py.title('KS2 - One Pass')

    ax2 = py.subplot(3, 1, 2, sharex=ax1)
    py.plot(stars.m_ks2, dyerr, 'k.', ms=2)
    py.ylabel('ye diff')
    py.ylim(-0.05, 0.05)
    py.axhline(y=0, color='r', linewidth=2)

    ax3 = py.subplot(3, 1, 3, sharex=ax1)
    py.plot(stars.m_ks2, dmerr, 'k.', ms=2)
    py.ylabel('me diff')
    py.ylim(-0.05, 0.05)
    py.axhline(y=0, color='r', linewidth=2)

    py.xlabel('Magnitude')
    py.xlim(magLimits[0], magLimits[1])

    py.savefig('plot_compare_errors%s.png' % tableSuffix)
    
    #####
    # Plot stars that are NOT in KS2 vs. those that are.
    #####
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


    #####
    # Plot vector point diagram of positional differences
    #####
    idx = np.where((stars.m_ks2 < -5) & (np.abs(dx) < posLim) & (np.abs(dy) < posLim) &
                   (stars.xe_pass1 < errLim) & (stars.ye_pass1 < errLim))[0]

    py.figure(6)
    py.clf()

    py.plot(dx[idx], dy[idx], 'k.', ms=2)
    py.xlabel('X KS2 - One Pass')
    py.xlabel('Y KS2 - One Pass')
    py.axis([-posLim, posLim, -posLim, posLim])
    py.savefig('plot_dxdy_vpd%s.png' % tableSuffix)


    #####
    # Plot vector field of positional differences
    #####
    py.figure(7)
    py.clf()

    q = py.quiver(stars.x_pass1[idx], stars.y_pass1[idx], dx[idx], dy[idx], scale=quiverScale)
    py.quiverkey(q, 0.5, 0.95, 0.05, '0.05 pix', color='red')
    py.title('KS2 - One Pass')
    py.xlabel('X (pixels)')
    py.xlabel('Y (pixels)')
    py.savefig('plot_dxdy_quiver%s.png' % tableSuffix)

    #####
    # Print out some statistics
    #####
    idx = np.where(stars.m_ks2 < -5)[0]
    print 'Mean dx = %7.3f +/- %7.3f (pix)' % \
        (statsIter.mean(dx[idx], lsigma=4, hsigma=4, iter=5),
         statsIter.std(dx[idx], lsigma=4, hsigma=4, iter=5))
    print 'Mean dy = %7.3f +/- %7.3f (pix)' % \
        (statsIter.mean(dy[idx], lsigma=4, hsigma=4, iter=5),
         statsIter.std(dy[idx], lsigma=4, hsigma=4, iter=5))
    print 'Mean dxerr = %7.3f +/- %7.3f (pix)' % \
        (statsIter.mean(dxerr[idx], lsigma=4, hsigma=4, iter=5),
         statsIter.std(dxerr[idx], lsigma=4, hsigma=4, iter=5))
    print 'Mean dyerr = %7.3f +/- %7.3f (pix)' % \
        (statsIter.mean(dyerr[idx], lsigma=4, hsigma=4, iter=5),
         statsIter.std(dyerr[idx], lsigma=4, hsigma=4, iter=5))


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


def ks2_bright_stars(ks2Root):
    uv1_file = ks2Root + '.FIND_AVG_UV1_F'
    z0_file = ks2Root + '.FIND_AVG_Z0_F'
    z2_file = ks2Root + '.FIND_AVG_Z2_F'
    
    uv1 = atpy.Table(uv1_file + '.fits')
    z0 = atpy.Table(z0_file + '.fits')
    z2 = atpy.Table(z2_file + '.fits')

    nfilt = uv1.keywords['NFILT']

    filterNames = ['F160W', 'F139M', 'F125W']

    # Trim down to just those stars with low photometric errors
    # in all 3 filters.
    good = ((uv1.me_1 < 0.05) & (uv1.me_2 < 0.05) & (uv1.me_3 < 0.05)) | (z0.pass_1 == 0)
    z0 = z0.where(good)
    print 'Keeping %d of %d with good photometry' % (len(z0), len(uv1))
    z2 = z2.where(good)
    uv1 = uv1.where(good)
    

    ##########
    # Plot the difference between the photometric methods.
    # Do this in all 3 filters.
    # Adopt the uv1 photometric errors.
    ##########
    py.close(2)
    py.figure(2, figsize=(16, 6))
    py.subplots_adjust(left=0.05)

    for ff in range(1, nfilt+1):
        mCol = 'm_%d' % ff
        meCol = 'm_%d' % ff

        uv1_z0 = uv1[mCol] - z0[mCol]
        uv1_z0_err = uv1[meCol]
        
        uv1_z2 = uv1[mCol] - z2[mCol]
        uv1_z2_err = uv1[meCol]

        z0_z2 = z0[mCol] - z2[mCol]
        z0_z2_err = uv1[meCol]

        py.clf()

        py.subplot(1, 3, 1)
        py.plot(z0[mCol], uv1_z0, 'k.', ms=2)
        py.xlabel('Z0 (mag)')
        py.ylabel('UV1 - Z0 (mag)')
        py.xlim(-18, -2)
        py.ylim(-2, 2)
        py.axhline(0, linestyle='--')

        py.subplot(1, 3, 2)
        py.plot(z0[mCol], uv1_z2, 'k.', ms=2)
        py.xlabel('Z0 (mag)')
        py.ylabel('UV1 - Z2 (mag)')
        py.title('Filter: %s' % filterNames[ff-1])
        py.xlim(-18, -2)
        py.ylim(-2, 2)
        py.axhline(0, linestyle='--')

        py.subplot(1, 3, 3)
        py.plot(z0[mCol], z0_z2, 'k.', ms=2)
        py.xlabel('Z0 (mag)')
        py.ylabel('Z0 - Z2 (mag)')
        py.xlim(-18, -2)
        py.ylim(-2, 2)
        py.axhline(0, linestyle='--')

        py.savefig('plot_ks2_brite_comp_phot_filt%d.png' % ff)

        py.subplot(1, 3, 1)
        py.ylim(-0.1, 0.1)
        py.subplot(1, 3, 2)
        py.ylim(-0.1, 0.1)
        py.subplot(1, 3, 3)
        py.ylim(-0.1, 0.1)
        py.savefig('plot_ks2_brite_comp_phot_filt%d_zoom.png' % ff)

        
    ##########
    # Plot the CMDs for the 3 possible filter combinations.
    # Do this for all 3 photometric methods.
    ##########

    methods = [z0, uv1, z2]

    for ii in range(len(methods)):
        py.clf()

        mm = methods[ii]
        py.subplot(1, 3, 1)
        py.plot(mm.m_3 - mm.m_2, mm.m_2, 'k.', ms=2)
        py.xlabel(filterNames[3-1] + ' - ' + filterNames[2-1])
        py.ylabel(filterNames[2-1])
        py.ylim(-4, -16)
        py.xlim(-1.9, -0.8)

        py.subplot(1, 3, 2)
        py.plot(mm.m_3 - mm.m_1, mm.m_2, 'k.', ms=2)
        py.xlabel(filterNames[3-1] + ' - ' + filterNames[1-1])
        py.ylabel(filterNames[2-1])
        py.title('Method %d' % ii)
        py.ylim(-4, -16)
        py.xlim(-0.5, 1.6)

        py.subplot(1, 3, 3)
        py.plot(mm.m_2 - mm.m_1, mm.m_2, 'k.', ms=2)
        py.xlabel(filterNames[2-1] + ' - ' + filterNames[1-1])
        py.ylabel(filterNames[2-1])
        py.ylim(-4, -16)
        py.xlim(1.4, 2.4)

        py.savefig('plot_ks2_brite_cmds_meth%d.png' % ii)

def plot_vpd_align(alignRoot, epoch1, epoch2, outRoot='plot_align'):
    """
    Read in an aligned data set. Plot a vector-point-diagram for the 
    positional differences between two different epochs within the data set.
    """
    s = starset.StarSet(alignRoot)

    x1 = s.getArrayFromEpoch(epoch1, 'xorig')
    y1 = s.getArrayFromEpoch(epoch1, 'yorig')
    x2 = s.getArrayFromEpoch(epoch2, 'xorig')
    y2 = s.getArrayFromEpoch(epoch2, 'yorig')

    x1e = s.getArrayFromEpoch(epoch1, 'xpixerr_p')
    y1e = s.getArrayFromEpoch(epoch1, 'ypixerr_p')
    x2e = s.getArrayFromEpoch(epoch2, 'xpixerr_p')
    y2e = s.getArrayFromEpoch(epoch2, 'ypixerr_p')

    dx = x2 - x1
    dy = y2 - y1
    dxe = np.hypot(x1e, x2e)
    dye = np.hypot(y1e, y2e)
    dr = np.hypot(dx, dy)

    idx = np.where((dxe < 0.02) & (dye < 0.02) & (dr < 3.0))[0]
    print len(x1), len(idx)

    drmax = 1.5

    py.figure(1)
    py.clf()
    py.plot(dx[idx], dy[idx], 'k.', ms=2)
    py.xlabel('Delta X (pixels)')
    py.ylabel('Delta Y (pixels)')
    py.axis([-drmax, drmax, -drmax, drmax])
    py.savefig(outRoot + '_vpd.png')

    py.figure(2)
    py.clf()
    q = py.quiver(x1[idx], y1[idx], dx[idx], dy[idx], scale=10)
    py.quiverkey(q, 0.5, 0.95, 0.05, '0.05 pix', color='red')
    py.savefig(outRoot + '_dr_vec.png')

    

    
