import pyfits
import numpy as np
import pylab as py
import atpy
import pdb
import os
from jlu.gc.gcwork import starset

"""
- Copy over the starlists from the GSAOI data directory.
- Fix the G3 starlists (find and move up named sources).
- Calibrate the jlu starlists
- reformat all the starlists -- makes the S*_#_jlu.lis (etc.)
- make_top_stars() -- makes the S*_#_jlu_named.lis
- align
- trim align

"""

work_dir = '/u/jlu/work/gsaoi/compare_starlists/'

def reformat_benoit():
    btab = pyfits.getdata(work_dir + 'ngc1851_benoit.fits')

    # Add back in the 12 columns and 12 rows that Benoit trimmed off
    btab[:, 0, :] += 12
    btab[:, 1, :] += 12

    nfiles = btab.shape[0]
    nstars = btab.shape[2]
    
    name = np.ones(nstars, dtype='S10')
    for ii in range(nstars):
        name[ii] = 'stars_{0:04d}'.format(ii+1) 

    year = 2012.95
            
    files = [58, 59, 60]
    for ff in range(nfiles):
        file_root_1 = 'S20121230S00{0}_{1}_ben.lis'.format(files[ff], 1)
        file_root_2 = 'S20121230S00{0}_{1}_ben.lis'.format(files[ff], 2)
        file_root_3 = 'S20121230S00{0}_{1}_ben.lis'.format(files[ff], 3)
        file_root_4 = 'S20121230S00{0}_{1}_ben.lis'.format(files[ff], 4)

        _f1 = open(file_root_1, 'w')
        _f2 = open(file_root_2, 'w')
        _f3 = open(file_root_3, 'w')
        _f4 = open(file_root_4, 'w')

        #    G3  G4
        #    G2  G1
        x = btab[ff, 0, :]
        y = btab[ff, 1, :]
        f = btab[ff, 2, :]
        m = -2.5 * np.log10(f) + 23.0

        split = 2100
        idx_1 = np.where((x > split) & (y < split))[0]
        idx_2 = np.where((x < split) & (y < split))[0]
        idx_3 = np.where((x < split) & (y > split))[0]
        idx_4 = np.where((x > split) & (y > split))[0]

        fmt = '{0:10s}  {1:6.3f}  {2:8.3f}  {3:10.3f} {4:10.3f}  1 1 1 {5:12.1f}\n'

        for ii in idx_1:
            _f1.write(fmt.format(name[ii], m[ii], year, x[ii], y[ii], f[ii]))

        for ii in idx_2:
            _f2.write(fmt.format(name[ii], m[ii], year, x[ii], y[ii], f[ii]))

        for ii in idx_3:
            _f3.write(fmt.format(name[ii], m[ii], year, x[ii], y[ii], f[ii]))

        for ii in idx_4:
            _f4.write(fmt.format(name[ii], m[ii], year, x[ii], y[ii], f[ii]))

        _f1.close()
        _f2.close()
        _f3.close()
        _f4.close()

    return

def reformat_jlu():
    """
    Make sure to run this first on all the *_badcoo.lis files:
    python ~/code/python/jlu_python/jlu/photometry/calibrate.py -N ~/data/gsaoi/commission/reduce/ngc1851/ngc1851_photo_calib.dat -I psf032 -c 13 -M 4 -V S20121230S0060_3_jlu_badcoo.lis
    """
    files = [58, 59, 60]

    year = 2012.95
    for ff in files:
        file_root_1 = 'S20121230S00{0}_1_jlu'.format(ff)
        file_root_2 = 'S20121230S00{0}_2_jlu'.format(ff)
        file_root_3 = 'S20121230S00{0}_3_jlu'.format(ff)
        file_root_4 = 'S20121230S00{0}_4_jlu'.format(ff)

        tab_1 = atpy.Table(file_root_1 + '_badcoo_cal.lis', type='ascii')
        tab_2 = atpy.Table(file_root_2 + '_badcoo_cal.lis', type='ascii')
        tab_3 = atpy.Table(file_root_3 + '_badcoo_cal.lis', type='ascii')
        tab_4 = atpy.Table(file_root_4 + '_badcoo_cal.lis', type='ascii')

        # Fix X positions of G1 and G4
        tab_1['col4'] += 2048 + 170.0
        tab_4['col4'] += 2048 + 170.0

        # Fix Y positions of G3 and G4
        tab_3['col5'] += 2048 + 170.0
        tab_4['col5'] += 2048 + 170.0

        _f1 = open(file_root_1 + '.lis', 'w')
        _f2 = open(file_root_2 + '.lis', 'w')
        _f3 = open(file_root_3 + '.lis', 'w')
        _f4 = open(file_root_4 + '.lis', 'w')

        fmt = '{0:10s}  {1:6.3f}  {2:8.3f}  {3:10.3f} {4:10.3f}  1 1 1 {5:12.1f}\n'

        for ii in range(len(tab_1)):
            _f1.write(fmt.format(tab_1['col1'][ii], tab_1['col2'][ii], tab_1['col3'][ii],
                                 tab_1['col4'][ii], tab_1['col5'][ii], tab_1['col9'][ii]))

        for ii in range(len(tab_2)):
            _f2.write(fmt.format(tab_2['col1'][ii], tab_2['col2'][ii], tab_2['col3'][ii],
                                 tab_2['col4'][ii], tab_2['col5'][ii], tab_2['col9'][ii]))

        for ii in range(len(tab_3)):
            _f3.write(fmt.format(tab_3['col1'][ii], tab_3['col2'][ii], tab_3['col3'][ii],
                                 tab_3['col4'][ii], tab_3['col5'][ii], tab_3['col9'][ii]))

        for ii in range(len(tab_4)):
            _f4.write(fmt.format(tab_4['col1'][ii], tab_4['col2'][ii], tab_4['col3'][ii],
                                 tab_4['col4'][ii], tab_4['col5'][ii], tab_4['col9'][ii]))
                                                                  

        _f1.close()
        _f2.close()
        _f3.close()
        _f4.close()
        
def reformat_mark():
    files = [58, 59, 60]

    year = 2012.95
    for ff in files:
        tab = atpy.Table(work_dir + 'ngc1851_d{0}_mark.dat'.format(ff), type='ascii')
        x = tab['col1']
        y = tab['col2']
        f = tab['col3']
        m = -2.5 * np.log10(f) + 23.0
        
        split = 2100
        idx_1 = np.where((x > split) & (y < split))[0]
        idx_2 = np.where((x < split) & (y < split))[0]
        idx_3 = np.where((x < split) & (y > split))[0]
        idx_4 = np.where((x > split) & (y > split))[0]

        file_root_1 = 'S20121230S00{0}_{1}_amm.lis'.format(ff, 1)
        file_root_2 = 'S20121230S00{0}_{1}_amm.lis'.format(ff, 2)
        file_root_3 = 'S20121230S00{0}_{1}_amm.lis'.format(ff, 3)
        file_root_4 = 'S20121230S00{0}_{1}_amm.lis'.format(ff, 4)
        
        _f1 = open(file_root_1, 'w')
        _f2 = open(file_root_2, 'w')
        _f3 = open(file_root_3, 'w')
        _f4 = open(file_root_4, 'w')

        fmt = '{0:10s}  {1:6.3f}  {2:8.3f}  {3:10.3f} {4:10.3f}  1 1 1 {5:12.1f}\n'

        for ii in idx_1:
            name = 'star_{0:04d}'.format(ii+1)
            _f1.write(fmt.format(name, m[ii], year, x[ii], y[ii], f[ii]))

        for ii in idx_2:
            name = 'star_{0:04d}'.format(ii+1)
            _f2.write(fmt.format(name, m[ii], year, x[ii], y[ii], f[ii]))

        for ii in idx_3:
            name = 'star_{0:04d}'.format(ii+1)
            _f3.write(fmt.format(name, m[ii], year, x[ii], y[ii], f[ii]))

        for ii in idx_4:
            name = 'star_{0:04d}'.format(ii+1)
            _f4.write(fmt.format(name, m[ii], year, x[ii], y[ii], f[ii]))

        _f1.close()
        _f2.close()
        _f3.close()
        _f4.close()
            

def make_top_stars():
    """
    Shuffle the stars around so that the same two named
    sources are at the top of every file.
    I selected these by hand to be relatively isolated.

    Note G3 images have mis-labeled star at the top and I had to
    change things manually.
    """
    topStars = {'1': {'amm': [{'name': 'G1_1', 'x': 2509.0, 'y':  555.0},
                              {'name': 'G1_2', 'x': 2294.0, 'y': 1638.5}],
                      'ben': [{'name': 'G1_1', 'x': 2534.5, 'y':  563.5},
                              {'name': 'G1_2', 'x': 2320.0, 'y': 1643.0}],
                      'jlu': [{'name': 'G1_1', 'x': 2532.0, 'y':  561.0},
                              {'name': 'G1_2', 'x': 2317.5, 'y': 1644.0}]},
                '2': {'amm': [{'name': 'G2_1', 'x': 1659.0, 'y': 1595.4},
                              {'name': 'G2_2', 'x':  507.0, 'y': 1394.5}],
                      'ben': [{'name': 'G2_1', 'x': 1665.5, 'y': 1602.0},
                              {'name': 'G2_2', 'x':  513.5, 'y': 1401.0}],
                      'jlu': [{'name': 'G2_1', 'x': 1663.0, 'y': 1599.5},
                              {'name': 'G2_2', 'x':  511.0, 'y': 1398.0}]},
                '3': {'amm': [{'name': 'G3_1', 'x':  507.0, 'y': 2344.0},
                              {'name': 'G3_2', 'x': 1819.0, 'y': 2877.5}],
                      'ben': [{'name': 'G3_1', 'x':  522.0, 'y': 2386.5},
                              {'name': 'G3_2', 'x': 1828.5, 'y': 2933.0}],
                      'jlu': [{'name': 'G3_1', 'x':  519.0, 'y': 2384.0},
                              {'name': 'G3_2', 'x': 1826.0, 'y': 2930.5}]},
                '4': {'amm': [{'name': 'G4_1', 'x': 3470.0, 'y': 2781.5},
                              {'name': 'G4_2', 'x': 2441.0, 'y': 3481.0}],
                      'ben': [{'name': 'G4_1', 'x': 3509.0, 'y': 2834.0},
                              {'name': 'G4_2', 'x': 2469.3, 'y': 3519.5}],
                      'jlu': [{'name': 'G4_1', 'x': 3505.0, 'y': 2831.5},
                              {'name': 'G4_2', 'x': 2467.0, 'y': 3516.9}]}}

    users = ['amm', 'ben', 'jlu']
    frames = ['58', '59', '60']
    chips = ['1', '2', '3', '4']

    root = 'S20121230S00'
    for user in users:
        for frame in frames:
            for chip in chips:
                starlist = '{0}{1}_{2}_{3}.lis'.format(root, frame,
                                                       chip, user)

                move_stars_up(starlist, topStars[chip][user])

    return

def move_stars_up(starlist, topStars):
    t = atpy.Table(starlist, type='ascii')
    t.rename_column('col4', 'x')
    t.rename_column('col5', 'y')
    
    # Find the "named" sources and shift them to the top
    idxInit = np.ones(len(topStars)) * -1
    for ss in range(len(topStars)):
        dr = np.hypot(t['x'] - topStars[ss]['x'],
                      t['y'] - topStars[ss]['y'])
        drmin = dr.argmin()

        # If this star is closer than 2 pixels, then call it a match.
        if dr[drmin] < 2.5:
            if idxInit[ss] != -1:
                print 'Confused!'
                pdb.set_trace()

            idxInit[ss] = drmin
            print 'Renamed {0} to {1} in {2}'.format(t['col1'][drmin],
                                                     topStars[ss]['name'],
                                                     starlist)

    topIdx = idxInit[idxInit >= 0]
    if len(topIdx) == 0:
        print 'No stars named!'
        pdb.set_trace()
        
    topNames = np.array([topStar['name'] for topStar in topStars])
    topNames = topNames[idxInit >= 0]

    outName = os.path.basename(starlist).replace('.lis', '_named.lis')
    _out = open(outName, 'w')

    # Write out the top stars first
    fmt = '{0:10s}  {1:6.3f}  {2:8.3f}  '
    fmt += '{3:10.3f} {4:10.3f}  1 1 1 {5:12.1f}\n'

    for ss in range(len(topIdx)):
        dd = topIdx[ss]

        _out.write(fmt.format(topNames[ss], t['col2'][dd], t['col3'][dd],
                              t['x'][dd], t['y'][dd], t['col9'][dd]))
    
    for dd in range(len(t)):
        # Make sure this wasn't one of the top stars.
        if dd in topIdx:
            continue

        _out.write(fmt.format(t['col1'][dd], t['col2'][dd], t['col3'][dd],
                              t['x'][dd], t['y'][dd], t['col9'][dd]))

    _out.close()

    return


def process_align_output(suffix='_t'):
    """
    Deal with the trim_align output that has all 3 users' starlists
    and all 3 frames and all 4 chips (separate align for each chip).
    Make them into a FITS catalog.
    """
    for ii in range(1, 4+1):
        s = starset.StarSet('align_{0}{1}'.format(ii, suffix))
        print 'Loaded starset ', ii

        n_users = 3
        n_frames = 3
        n_stars = len(s.stars)

        x = np.zeros((n_users, n_frames, n_stars), dtype=float)
        y = np.zeros((n_users, n_frames, n_stars), dtype=float)
        m = np.zeros((n_users, n_frames, n_stars), dtype=float)

        for uu in range(n_users):
            for ff in range(n_frames):
                icol = ff*n_frames + uu

                x[uu, ff, :] = s.getArrayFromEpoch(icol, 'xpix')
                y[uu, ff, :] = s.getArrayFromEpoch(icol, 'ypix')
                m[uu, ff, :] = s.getArrayFromEpoch(icol, 'mag')

        cat = np.array((x, y, m))
        pyfits.writeto('catalog_quad_{0}{1}.fits'.format(ii, suffix), cat,
                       output_verify='silentfix', clobber=True)

        
def plot_pos_diff(suffix='_t'):
    for ii in range(1, 4+1):
        t = pyfits.getdata('catalog_quad_{0}{1}.fits'.format(ii, suffix))

        x = t[0]
        y = t[1]
        m = t[2]

        n_users = x.shape[0]
        n_frames = x.shape[1]
        n_stars = x.shape[2]

        xerr_0 = x[0, :, :].std(axis=0)
        xerr_1 = x[1, :, :].std(axis=0)
        xerr_2 = x[2, :, :].std(axis=0)
        yerr_0 = y[0, :, :].std(axis=0)
        yerr_1 = y[1, :, :].std(axis=0)
        yerr_2 = y[2, :, :].std(axis=0)
        merr_0 = m[0, :, :].std(axis=0)
        merr_1 = m[1, :, :].std(axis=0)
        merr_2 = m[2, :, :].std(axis=0)
        m_0 = m[0, :, :].mean(axis=0)
        m_1 = m[1, :, :].mean(axis=0)
        m_2 = m[1, :, :].mean(axis=0)

        py.figure(1)
        py.clf()
        
        f, (ax1, ax2) = py.subplots(2, 1, sharex=True)
        f.subplots_adjust(hspace=0.)

        ax1.plot(xerr_2, xerr_1 - xerr_2, 'ro', label='X', mec='none', ms=5)
        ax1.plot(yerr_2, yerr_1 - yerr_2, 'bo', label='Y', mec='none', ms=5)
        ax1.axhline(0, linestyle='--', color='black')
        ax1.legend(numpoints=1)
        ax1.set_title('Compare Pos RMS Error')
        ax1.set_ylabel(r'$\sigma_{BN} - \sigma_{JL}$ (pix)')
        ax1.set_ylim(-0.3, 0.3)
        ax1.set_xlim(0, 0.3)

        ax2.plot(xerr_2, xerr_0 - xerr_2, 'ro', label='X', mec='none', ms=5)
        ax2.plot(yerr_2, yerr_0 - yerr_2, 'bo', label='Y', mec='none', ms=5)
        ax2.axhline(0, linestyle='--', color='black')
        ax2.set_xlabel(r'$\sigma_{JL}$ (pix)')
        ax2.set_ylabel(r'$\sigma_{BN} - \sigma_{JL}$ (pix)')
        ax2.set_ylim(-0.3, 0.3)
        ax2.set_xlim(0, 0.3)
        py.savefig('plots/pos_err_compare_q{0}.png'.format(ii))
        py.close()
    

        print '      X_RMS_Error    Y_RMS_Error   (Quad={0})'.format(ii)
        print ' JL   {0:11.2f}    {1:11.2f}'.format(xerr_2.mean(), yerr_2.mean())
        print ' BN   {0:11.2f}    {1:11.2f}'.format(xerr_1.mean(), yerr_1.mean())
        print ' MA   {0:11.2f}    {1:11.2f}'.format(xerr_0.mean(), yerr_0.mean())
                

        for ff in range(n_frames):
            dx1 = x[0, ff, :] - x[2, ff, :]
            dy1 = y[0, ff, :] - y[2, ff, :]
            dr1 = np.hypot(dx1, dy1)

            dx2 = x[1, ff, :] - x[2, ff, :]
            dy2 = y[1, ff, :] - y[2, ff, :]
            dr2 = np.hypot(dx2, dy2)

            dm1 = m[0, ff, :] - m[2, ff, :]
            dm2 = m[1, ff, :] - m[2, ff, :]
            
            # Make a histogram of dx and dy
            py.figure(1)
            py.clf()
            bins = np.arange(-1, 1, 0.05)

            py.subplot(211)
            py.hist(dx1, histtype='step', color='red', bins=bins, label='X')
            py.hist(dy1, histtype='step', color='blue', bins=bins, label='Y')
            py.ylabel('N stars (MA - JLU)')
            py.title('Frame ' + str(58+ff) + ', Quad ' + str(ii))
            py.legend()

            py.subplot(212)
            py.hist(dx2, histtype='step', color='red', bins=bins)
            py.hist(dy2, histtype='step', color='blue', bins=bins)
            py.ylabel('N stars (BN - JLU)')
            py.xlabel('Position Difference (pix)')

            py.savefig('plots/hist_pos_q{0}_f{1}.png'.format(ii, ff))

            # Make a histogram of dm
            py.figure(2)
            py.clf()
            py.subplot(111)
            bins = np.arange(-4.6, -3.2, 0.1)
            py.hist(dm1, histtype='step', color='red', bins=bins, label='MA-JLU')
            py.hist(dm2, histtype='step', color='blue', bins=bins, label='BN-JLU')
            py.xlabel('Magnitude Difference')
            py.ylabel('N stars')
            py.legend()
            py.title('Frame ' + str(58+ff) + ', Quad ' + str(ii))
            py.savefig('plots/hist_mag_q{0}_f{1}.png'.format(ii, ff))

            print 'Median Zeropoint (Quad={0}, Frame={1}):'.format(ii, 58+ff)
            print 'Nstars = {0}'.format(x.shape[2])
            print '   MA = {0:.2f}'.format(dm1.mean())
            print '   BN = {0:.2f}'.format(dm2.mean())

            # Make a 2 panel quiver plot with positional offsets relative to
            # JLU position.
            py.close(3)
            py.figure(3, figsize=(10,5))
            qscale = 1
            py.subplots_adjust(left=0.1, wspace=0.3, bottom=0.13)

            py.subplot(121)
            q = py.quiver(x[2, ff, :], y[2, ff, :], dx1, dy1, scale=qscale)
            py.quiverkey(q, 0.9, 0.9, 0.1, '2 mas')
            py.xlabel('X JLU (pixels)')
            py.ylabel('X JLU (pixels)')
            py.axis('equal')
            rng = py.axis()
            py.axis([rng[0], rng[0]+2500, rng[2], rng[2]+2500])
            py.title('MA - JLU: Frame ' + str(58+ff) + ', Quad ' + str(ii))

            py.subplot(122)
            q = py.quiver(x[2, ff, :], y[2, ff, :], dx2, dy2, scale=qscale)
            py.quiverkey(q, 0.9, 0.9, 0.1, '2 mas')
            py.xlabel('X JLU (pixels)')
            py.ylabel('X JLU (pixels)')
            py.axis('equal')
            rng = py.axis()
            py.axis([rng[0], rng[0]+2500, rng[2], rng[2]+2500])
            py.title('BN - JLU: Frame ' + str(58+ff) + ', Quad ' + str(ii))

            py.savefig('plots/quiver_q{0}_f{1}.png'.format(ii, ff))

