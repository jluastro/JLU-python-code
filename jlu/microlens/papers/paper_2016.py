import numpy as np
import pylab as plt
from jlu.microlens import residuals
from jlu.microlens import align_compare
from jlu.microlens import model
from jlu.util import fileUtil
from astropy.table import Table
import numpy as np
import shutil
from gcwork import starTables
from gcwork import starset
from scipy import spatial
import scipy
import scipy.stats
import pdb

# Final alignments and Directories
root_dir = '/Users/jlu/work/microlens/OB120169/a_2016_10_10/'

an_dir1 = root_dir + 'a_ob120169_2016_10_10_a3_m22_w4_MC100/'
an_dir2 = root_dir + 'a_ob120169_2016_10_10_a4_m22_w4_MC100/'

an_dir = an_dir1
points_dir = an_dir + 'points_a/'
poly_dir = an_dir + 'polyfit_a/'

def compare_align_order(only_stars_in_fit=True):
    plot_dir = root_dir + 'plots/'

    # prefix for the analysis directories
    prefix = 'a'
    target = 'ob120169'
    date = '2016_10_10'

    # Make 2D arrays for the runs with different Kcut and orders.
    Kcuts = np.array([22])
    orders = np.array([3, 4, 5])
    weights = np.array([1, 2, 3, 4])

    analysis_root_fmt = '{root:s}{prefix:s}_{target:s}_{date:s}_a{order:d}_m{kcut:d}_w{weight:d}_MC100/'

    for kk in range(len(Kcuts)):
        for ww in range(len(weights)):
            analysis_roots = []
            align_roots = []

            print("Kcut = {kcut:d}  weight = {weight:d}".format(kcut=Kcuts[kk], weight=weights[ww]))
            for oo in range(len(orders)):
                tmp = analysis_root_fmt.format(root=root_dir, prefix=prefix, 
                                               target=target, date=date,
                                               order=orders[oo], kcut=Kcuts[kk], weight=weights[ww])
                analysis_roots.append(tmp)
                align_roots.append( tmp + 'align/align_t' )

                out_suffix = 'm{kcut:d}_w{weight:d}'.format(kcut=Kcuts[kk], weight=weights[ww])

            align_compare.align_residuals_vs_order(analysis_roots,
                                                   only_stars_in_fit=only_stars_in_fit,
                                                   out_suffix=out_suffix, plot_dir=plot_dir)

            for oo in range(len(orders)):
                residuals.chi2_dist_all_epochs('align/align_t', root_dir=analysis_roots[oo],
                                               only_stars_in_fit=only_stars_in_fit)


    return    

def stars_used_in_align():
    """Calculate the mean number of stars used per epoch to align
    to a common coordinate system.
    """
    trans = Table.read(an_dir + 'align/align_t.trans', format='ascii')

    num_stars = trans['NumStars']

    rdx = np.where(num_stars != 0)[0]

    avg = num_stars[rdx].mean()
    std = num_stars[rdx].std()

    print("Mean Number of Stars Used (not in ref epoch): {0:.1f} +/- {1:.1f}".format(avg, std))
    print(num_stars)

    return

def align_errors():
    """Calculate the mean alignment error on the positions for 
    the set of stars used in the transformations and detected
    in all epochs. 
    """
    s = starset.StarSet(an_dir + 'align/align_t')
    s.loadStarsUsed()

    scale = 9.952
    N_epochs = len(s.dates)

    pe_avg_all = np.zeros(N_epochs, dtype=float)
    pe_std_all = np.zeros(N_epochs, dtype=float)

    for ee in range(N_epochs):
        is_used = s.getArrayFromEpoch(ee, 'isUsed')
        xe_a = s.getArrayFromEpoch(ee, 'xpixerr_a')
        ye_a = s.getArrayFromEpoch(ee, 'ypixerr_a')

        good = np.where(is_used == True)[0]
        pe_avg = (xe_a[good].mean() + ye_a[good].mean()) / 2.0
        pe_std = (xe_a[good].std() + xe_a[good].std()) / 2.0

        pe_avg *= scale
        pe_std *= scale

        pe_avg_all[ee] = pe_avg
        pe_std_all[ee] = pe_std

        fmt = "Mean align error (1D) for epoch {0:d}:  {1:6.3f} +- {2:6.3f} mas"
        print(fmt.format(ee, pe_avg, pe_std))


    gdx = np.where(np.isfinite(pe_avg_all) == True)[0]
    
    fmt = "  Mean align error (1D) for all epochs:  {0:6.3f} +- {1:6.3f} mas"
    print(fmt.format(pe_avg_all[gdx].mean(), pe_std_all[gdx].mean()))

    fmt = "Median align error (1D) for all epochs:  {0:6.3f} +- {1:6.3f} mas"
    print(fmt.format(np.median(pe_avg_all[gdx]), np.median(pe_std_all[gdx])))
    
    return


def apply_local_astrometry(an_dir):
    s = starset.StarSet(an_dir + 'align/align_t')
    s.loadPolyfit(an_dir + 'polyfit_d/fit', accel=0, arcsec=0)

    name = s.getArray('name')

    x = s.getArrayFromAllEpochs('xpix')
    y = s.getArrayFromAllEpochs('ypix')
    xe_p = s.getArrayFromAllEpochs('xpixerr_p')
    ye_p = s.getArrayFromAllEpochs('ypixerr_p')
    xe_a = s.getArrayFromAllEpochs('xpixerr_a')
    ye_a = s.getArrayFromAllEpochs('ypixerr_a')

    x0 = s.getArray('fitpXv.p')
    vx = s.getArray('fitpXv.v')
    t0x = s.getArray('fitpXv.t0')
    y0 = s.getArray('fitpYv.p')
    vy = s.getArray('fitpYv.v')
    t0y = s.getArray('fitpYv.t0')
    
    N_epochs = len(s.years)
    N_stars = len(s.stars)

    dx = np.zeros((N_epochs, N_stars), dtype=float)
    dy = np.zeros((N_epochs, N_stars), dtype=float)
    dxe = np.zeros((N_epochs, N_stars), dtype=float)
    dye = np.zeros((N_epochs, N_stars), dtype=float)

    for ee in range(N_epochs):
        # Identify reference stars that can be used for calculating local distortions.
        # We use all the stars (not restricted to just the alignment stars).
        # Use a KD Tree.
        # First clean out undetected stars.
        idx = np.where((xe_p[ee, :] != 0) & (ye_p[ee, :] != 0))[0]

        # Put together observed data. 
        coords = np.empty((len(idx), 2))
        coords[:, 0] = x[ee, idx]
        coords[:, 1] = y[ee, idx]

        tree = spatial.KDTree(coords)

        # For every star, calculate the best fit position at each epoch.
        dt_x = s.years[ee] - t0x   # N_stars long array
        dt_y = s.years[ee] - t0y

        x_fit = x0 + (vx * dt_x)
        y_fit = y0 + (vy * dt_y)

        # Query the KD tree for the nearest 3 neighbors (including self) for every star.
        nn_r, nn_i = tree.query(coords, 3, p=2)

        id1 = idx[nn_i[:, 1]]
        dx1 = x[ee, id1] - x_fit[id1]
        dy1 = y[ee, id1] - y_fit[id1]

        id2 = idx[nn_i[:, 2]]
        dx2 = x[ee, id2] - x_fit[id2]
        dy2 = y[ee, id2] - y_fit[id2]

        dx12 = np.array([dx1, dx2])
        dy12 = np.array([dy1, dy2])
    
        dx_avg = dx12.mean(axis=0)
        dy_avg = dy12.mean(axis=0)
        dx_std = dx12.std(axis=0)
        dy_std = dy12.std(axis=0)

        # Print the deltas specifically for OB120169
        lens = np.where(np.array(name) == 'OB120169')[0][0]
        msg = 'Mean {0:4s} = {1:6.3f} +- {2:6.3f} pix   ({3:6.2f} sigma)'
        print('')
        print('Epoch = {0:d}'.format(ee))
        print(msg.format('dx', dx_avg[lens], dx_std[lens], dx_avg[lens] / dx_std[lens]))
        print(msg.format('dy', dy_avg[lens], dy_std[lens], dy_avg[lens] / dy_std[lens]))

        dx[ee, idx] = dx_avg
        dy[ee, idx] = dy_avg
        dxe[ee, idx] = dx_std
        dye[ee, idx] = dy_std
        

    # Get the lens and two nearest sources
    targets = ['OB120169', 'OB120169_L', 'S24_18_0.8']
    tdx = [name.index(targets[0]), name.index(targets[1]), name.index(targets[2])]

    plt.figure(1)
    plt.clf()
    plt.subplot(211)
    plt.errorbar(s.years, dx[:, tdx[0]], yerr=dxe[:, tdx[0]], color='red', linestyle='none', marker='.', label=targets[0])
    plt.errorbar(s.years, dx[:, tdx[1]], yerr=dxe[:, tdx[1]], color='blue', linestyle='none', marker='.', label=targets[1])
    plt.errorbar(s.years, dx[:, tdx[2]], yerr=dxe[:, tdx[2]], color='green', linestyle='none', marker='.', label=targets[2])
    plt.legend(numpoints=1, fontsize=8)
    plt.ylim(-0.22, 0.22)
    plt.ylabel(r'$\Delta$x (pix)')
    plt.axhline(0, color='k', linestyle='--')

    plt.subplot(212)
    plt.errorbar(s.years, dy[:, tdx[0]], yerr=dye[:, tdx[0]], color='red', linestyle='none', marker='.', label=targets[0])
    plt.errorbar(s.years, dy[:, tdx[1]], yerr=dye[:, tdx[1]], color='blue', linestyle='none', marker='.', label=targets[1])
    plt.errorbar(s.years, dy[:, tdx[2]], yerr=dye[:, tdx[2]], color='green', linestyle='none', marker='.', label=targets[2])
    plt.ylim(-0.22, 0.22)
    plt.ylabel(r'$\Delta$y (pix)')
    plt.xlabel('Year')
    plt.axhline(0, color='k', linestyle='--')
    plt.savefig(root_dir + 'plots/plot_local_astrometry_3.png')
    plt.show()
    
    x_old = x
    y_old = y
    xe_p_old = xe_p
    ye_p_old = ye_p
    xe_a_old = xe_a
    ye_a_old = ye_a
    xe_old = np.hypot(xe_p_old, xe_a_old)
    ye_old = np.hypot(ye_p_old, ye_a_old)

    x_new = x_old - dx
    y_new = y_old - dy
    xe_new = np.hypot(xe_old, dxe)
    ye_new = np.hypot(ye_old, dye) 
    
    msg = '{0:16s} x = {1:8.3f} +- {2:6.3f}    y = {3:8.3f} +- {4:6.3f}'
    for ee in range(N_epochs):
        print('')
        print('##### EPOCH: {0:d} #####'.format(ee))
        for tt in range(len(targets)):
            print(msg.format(targets[tt] + ' OLD',
                                 x_old[ee, tdx[tt]], xe_old[ee, tdx[tt]],
                                 y_old[ee, tdx[tt]], ye_old[ee, tdx[tt]]))
            print(msg.format(targets[tt] + ' NEW',
                                 x_new[ee, tdx[tt]], xe_new[ee, tdx[tt]],
                                 y_new[ee, tdx[tt]], ye_new[ee, tdx[tt]]))    

    plot_three_stars_old_new(targets, s.years,
                                 x_old[:, tdx], y_old[:, tdx], xe_old[:, tdx], ye_old[:, tdx],
                                 x_new[:, tdx], y_new[:, tdx], xe_new[:, tdx], ye_new[:, tdx], outsuffix='_a3')

    # Save to new *.points files (in a new points directory) for these three sources.
    # The other sources just get copied over. Also use this as an opportunity to convert to MJD.
    fileUtil.mkdir(an_dir1 + 'points_a/')
    
    for nn in range(len(name)):
        #### POINTS file ####
        pts = starTables.read_points(an_dir + 'points_d/' + name[nn] + '.points')
        
        # Convert to MJD
        d_days = (pts['epoch'] - 1999.0) * 365.242
        mjd = d_days + 51179.0
        pts['epoch'] = mjd

        # Save new astrometry
        pts['x'] = x_new[:, nn]
        pts['y'] = y_new[:, nn]
        pts['xerr'] = xe_new[:, nn]
        pts['yerr'] = ye_new[:, nn]

        # Drop the first (crappy) epoch.
        pts = pts[1:]
            
        starTables.write_points(pts, an_dir + 'points_a/' + name[nn] + '.points')

        #### PHOT file ####
        phot = starTables.read_phot_points(an_dir + 'points_d/' + name[nn] + '.phot')
        phot['epoch'] = mjd
        phot = phot[1:]
        starTables.write_phot_points(phot, an_dir + 'points_a/' + name[nn] + '.phot')
        
    return


def apply_local_astrometry_compare():
    s1 = starset.StarSet(an_dir1 + 'align/align_t')
    s1.loadPolyfit(an_dir1 + 'polyfit_d/fit', accel=0, arcsec=0)

    s2 = starset.StarSet(an_dir2 + 'align/align_t')
    s2.loadPolyfit(an_dir2 + 'polyfit_d/fit', accel=0, arcsec=0)

    name1 = s1.getArray('name')
    name2 = s2.getArray('name')

    # Fetch the positions for the three sources of interest
    targets = ['OB120169', 'OB120169_L', 'S24_18_0.8']

    idx1 = np.nonzero(np.in1d(name1, targets))[0]
    idx2 = np.nonzero(np.in1d(name2, targets))[0] 

    x1 = s1.getArrayFromAllEpochs('xpix')
    y1 = s1.getArrayFromAllEpochs('ypix')
    xe_p1 = s1.getArrayFromAllEpochs('xpixerr_p')
    ye_p1 = s1.getArrayFromAllEpochs('ypixerr_p')
    xe_a1 = s1.getArrayFromAllEpochs('xpixerr_a')
    ye_a1 = s1.getArrayFromAllEpochs('ypixerr_a')

    x01 = s1.getArray('fitpXv.p')
    vx1 = s1.getArray('fitpXv.v')
    t0x1 = s1.getArray('fitpXv.t0')
    y01 = s1.getArray('fitpYv.p')
    vy1 = s1.getArray('fitpYv.v')
    t0y1 = s1.getArray('fitpYv.t0')

    x2 = s2.getArrayFromAllEpochs('xpix')
    y2 = s2.getArrayFromAllEpochs('ypix')
    xe_p2 = s2.getArrayFromAllEpochs('xpixerr_p')
    ye_p2 = s2.getArrayFromAllEpochs('ypixerr_p')
    xe_a2 = s2.getArrayFromAllEpochs('xpixerr_a')
    ye_a2 = s2.getArrayFromAllEpochs('ypixerr_a')

    x02 = s2.getArray('fitpXv.p')
    vx2 = s2.getArray('fitpXv.v')
    t0x2 = s2.getArray('fitpXv.t0')
    y02 = s2.getArray('fitpYv.p')
    vy2 = s2.getArray('fitpYv.v')
    t0y2 = s2.getArray('fitpYv.t0')

    N_epochs = len(s1.years)

    dx1 = np.zeros((len(idx1), N_epochs), dtype=float)
    dy1 = np.zeros((len(idx1), N_epochs), dtype=float)
    dx2 = np.zeros((len(idx2), N_epochs), dtype=float)
    dy2 = np.zeros((len(idx2), N_epochs), dtype=float)

    for ee in range(N_epochs):
        # For the three targets of interest, calculate the residuals in this epoch.
        dt_x1 = s1.years[ee] - t0x1[idx1]
        dt_y1 = s1.years[ee] - t0y1[idx1]
        dt_x2 = s2.years[ee] - t0x2[idx2]
        dt_y2 = s2.years[ee] - t0y2[idx2]

        x_fit1 = x01[idx1] + (vx1[idx1] * dt_x1)
        y_fit1 = y01[idx1] + (vy1[idx1] * dt_y1)
        x_fit2 = x02[idx2] + (vx2[idx2] * dt_x2)
        y_fit2 = y02[idx2] + (vy2[idx2] * dt_y2)

        dx1[:, ee] = x1[ee, idx1] - x_fit1
        dy1[:, ee] = y1[ee, idx1] - y_fit1
        dx2[:, ee] = x2[ee, idx2] - x_fit2
        dy2[:, ee] = y2[ee, idx2] - y_fit2

    # Mean delta-x and delta-y for the OTHER two sources.
    dx1_mean3 = dx1.mean(axis=0)
    dx1_mean2 = np.mean(dx1[1:], axis=0)
    dx1_stdv3 = dx1.std(axis=0)
    dx1_stdv2 = np.std(dx1[1:], axis=0)

    dx2_mean3 = dx2.mean(axis=0)
    dx2_mean2 = np.mean(dx2[1:], axis=0)
    dx2_stdv3 = dx2.std(axis=0)
    dx2_stdv2 = np.std(dx2[1:], axis=0)

    dy1_mean3 = dy1.mean(axis=0)
    dy1_mean2 = np.mean(dy1[1:], axis=0)
    dy1_stdv3 = dy1.std(axis=0)
    dy1_stdv2 = np.std(dy1[1:], axis=0)

    dy2_mean3 = dy2.mean(axis=0)
    dy2_mean2 = np.mean(dy2[1:], axis=0)
    dy2_stdv3 = dy2.std(axis=0)
    dy2_stdv2 = np.std(dy2[1:], axis=0)

    msg = 'Mean {0:13s} = {1:6.3f} +- {2:6.3f}   ({3:6.2f} sigma)'
    for ee in range(N_epochs):
        print('')
        print('Epoch = {0:d}'.format(ee))
        print(msg.format('dx (3 stars, a=3)', dx1_mean3[ee], dx1_stdv3[ee], dx1_mean3[ee] / dx1_stdv3[ee]))
        print(msg.format('dx (2 stars, a=3)', dx1_mean2[ee], dx1_stdv2[ee], dx1_mean2[ee] / dx1_stdv2[ee]))
        print(msg.format('dx (3 stars, a=4)', dx2_mean3[ee], dx2_stdv3[ee], dx2_mean3[ee] / dx2_stdv3[ee]))
        print(msg.format('dx (2 stars, a=4)', dx2_mean2[ee], dx2_stdv2[ee], dx2_mean2[ee] / dx2_stdv2[ee]))

        print(msg.format('dy (3 stars, a=3)', dy1_mean3[ee], dy1_stdv3[ee], dy1_mean3[ee] / dy1_stdv3[ee]))
        print(msg.format('dy (2 stars, a=3)', dy1_mean2[ee], dy1_stdv2[ee], dy1_mean2[ee] / dy1_stdv2[ee]))
        print(msg.format('dy (3 stars, a=4)', dy2_mean3[ee], dy2_stdv3[ee], dy2_mean3[ee] / dy2_stdv3[ee]))
        print(msg.format('dy (2 stars, a=4)', dy2_mean2[ee], dy2_stdv2[ee], dy2_mean2[ee] / dy2_stdv2[ee]))


    plt.figure(1)
    plt.clf()
    plt.subplot(211)
    plt.errorbar(s1.years, dx1_mean2, yerr=dx1_stdv2, color='red', linestyle='none', marker='.', label='a=3, 2 stars')
    plt.errorbar(s1.years, dx1_mean3, yerr=dx1_stdv3, color='blue', linestyle='none', marker='.', label='a=3, 3 stars')
    plt.errorbar(s2.years, dx2_mean2, yerr=dx2_stdv2, color='magenta', linestyle='none', marker='.', label='a=4, 2 stars')
    plt.errorbar(s2.years, dx2_mean3, yerr=dx2_stdv3, color='cyan', linestyle='none', marker='.', label='a=4, 3 stars')
    plt.legend(numpoints=1, fontsize=8)
    plt.ylim(-0.22, 0.22)
    plt.ylabel(r'$\Delta$x (pix)')
    plt.axhline(0, color='k', linestyle='--')

    plt.subplot(212)
    plt.errorbar(s1.years, dy1_mean2, yerr=dx1_stdv2, color='red', linestyle='none', marker='.', label='a=3, 2 stars')
    plt.errorbar(s1.years, dy1_mean3, yerr=dx1_stdv3, color='blue', linestyle='none', marker='.', label='a=3, 3 stars')
    plt.errorbar(s2.years, dy2_mean2, yerr=dx2_stdv2, color='magenta', linestyle='none', marker='.', label='a=4, 2 stars')
    plt.errorbar(s2.years, dy2_mean3, yerr=dx2_stdv3, color='cyan', linestyle='none', marker='.', label='a=4, 3 stars')
    plt.ylim(-0.22, 0.22)
    plt.ylabel(r'$\Delta$y (pix)')
    plt.xlabel('Year')
    plt.axhline(0, color='k', linestyle='--')
    plt.savefig(root_dir + 'plots/plot_local_astrometry.png')
    plt.show()
    
    
    x_old1 = x1[:, idx1]
    y_old1 = y1[:, idx1]
    xe_p_old1 = xe_p1[:, idx1]
    ye_p_old1 = ye_p1[:, idx1]
    xe_a_old1 = xe_a1[:, idx1]
    ye_a_old1 = ye_a1[:, idx1]
    xe_old1 = np.hypot(xe_p_old1, xe_a_old1)
    ye_old1 = np.hypot(ye_p_old1, ye_a_old1)

    x_old2 = x2[:, idx2]
    y_old2 = y2[:, idx2]
    xe_p_old2 = xe_p2[:, idx2]
    ye_p_old2 = ye_p2[:, idx2]
    xe_a_old2 = xe_a2[:, idx2]
    ye_a_old2 = ye_a2[:, idx2]
    xe_old2 = np.hypot(xe_p_old2, xe_a_old2)
    ye_old2 = np.hypot(ye_p_old2, ye_a_old2)

    x_new1 = x_old1 - np.tile(dx1_mean2, [3, 1]).T
    y_new1 = y_old1 - np.tile(dy1_mean2, [3, 1]).T

    x_new2 = x_old2 - np.tile(dx2_mean2, [3, 1]).T
    y_new2 = y_old2 - np.tile(dy2_mean2, [3, 1]).T

    xe_new1 = np.hypot(xe_old1, np.tile(dx1_stdv2, [3, 1]).T)
    ye_new1 = np.hypot(ye_old1, np.tile(dy1_stdv2, [3, 1]).T)
    xe_new2 = np.hypot(xe_old2, np.tile(dx2_stdv2, [3, 1]).T)
    ye_new2 = np.hypot(ye_old2, np.tile(dy2_stdv2, [3, 1]).T)
    # xe_new1 = xe_old1
    # ye_new1 = ye_old1
    # xe_new2 = xe_old2
    # ye_new2 = ye_old2

    msg = '{0:16s} x = {1:8.3f} +- {2:6.3f}    y = {3:8.3f} +- {4:6.3f}'
    for ee in range(N_epochs):
        print('')
        print('##### EPOCH: {0:d} #####'.format(ee))
        print('*** a = 3 ***')
        for tt in range(len(targets)):
            print(msg.format(targets[tt] + ' OLD', x_old1[ee, tt], xe_old1[ee, tt], y_old1[ee, tt], ye_old1[ee, tt]))
            print(msg.format(targets[tt] + ' NEW', x_new1[ee, tt], xe_new1[ee, tt], y_new1[ee, tt], ye_new1[ee, tt]))    

        print('')
        print('*** a = 4 ***')
        for tt in range(len(targets)):
            print(msg.format(targets[tt] + ' OLD', x_old2[ee, tt], xe_old2[ee, tt], y_old2[ee, tt], ye_old2[ee, tt]))
            print(msg.format(targets[tt] + ' NEW', x_new2[ee, tt], xe_new2[ee, tt], y_new2[ee, tt], ye_new2[ee, tt]))

    plot_three_stars_old_new(targets, s1.years, x_old1, y_old1, xe_old1, ye_old1, x_new1, y_new1, xe_new1, ye_new1, outsuffix='_a3')
    plot_three_stars_old_new(targets, s2.years, x_old2, y_old2, xe_old2, ye_old2, x_new2, y_new2, xe_new2, ye_new2, outsuffix='_a4')

    return
            

def plot_three_stars_old_new(targets, years, x_old, y_old, xe_old, ye_old, x_new, y_new, xe_new, ye_new, outsuffix=''):
    dateTicLoc = plt.MultipleLocator(2)
    dateTicRng = [2012, 2017]
    dateTics = np.arange(2012, 2018)
    DateTicsLabel = dateTics
    	    
    maxErr_x = np.array([xe_old, xe_new]).max()
    maxErr_y = np.array([ye_old, ye_new]).max()
    print('maxErr_x, maxErr_y = ', maxErr_x, maxErr_y)
        
    from matplotlib.ticker import FormatStrFormatter
    fmtX = FormatStrFormatter('%5i')
    fmtY = FormatStrFormatter('%6.2f')
    fontsize1 = 10

    ### a=3 results
    plt.figure(2, figsize=(10,10))
    plt.clf()

    N_rows = len(targets)
    
    for tt in range(len(targets)):
        paxes = plt.subplot(N_rows, 3, tt*3 + 1)
        plt.errorbar(years, x_old[:, tt], yerr=xe_old[:, tt], fmt='b.', label='Old')
        plt.errorbar(years, x_new[:, tt], yerr=xe_new[:, tt], fmt='r.', label='New')
        rng = plt.axis()
        plt.xlabel('Date (yrs)', fontsize=fontsize1)
        plt.ylabel('X (pix)', fontsize=fontsize1)
        paxes.xaxis.set_major_formatter(fmtX)
        paxes.get_xaxis().set_major_locator(dateTicLoc)
        paxes.yaxis.set_major_formatter(fmtY)
        paxes.tick_params(axis='both', which='major', labelsize=fontsize1)
        plt.xlim(np.min(dateTics), np.max(dateTics))
        if tt == 0:
            plt.legend(loc='upper left', fontsize=fontsize1)
            
        paxes = plt.subplot(N_rows, 3, tt*3 + 2)
        plt.errorbar(years, y_old[:, tt], yerr=ye_old[:, tt], fmt='b.')
        plt.errorbar(years, y_new[:, tt], yerr=ye_new[:, tt], fmt='r.')
        rng = plt.axis()
        plt.axis(dateTicRng + [rng[2], rng[3]], fontsize=fontsize1)
        plt.xlabel('Date (yrs)', fontsize=fontsize1)
        plt.ylabel('Y (pix)', fontsize=fontsize1)
        paxes.xaxis.set_major_formatter(fmtX)
        paxes.get_xaxis().set_major_locator(dateTicLoc)
        paxes.yaxis.set_major_formatter(fmtY)
        paxes.tick_params(axis='both', which='major', labelsize=fontsize1)
        plt.xlim(np.min(dateTics), np.max(dateTics))
        plt.title(targets[tt] + outsuffix)
        
        paxes = plt.subplot(N_rows, 3, tt*3 + 3)
        plt.errorbar(x_old[:, tt], y_old[:, tt], xerr=xe_old[:, tt], yerr=ye_old[:, tt], fmt='b.')
        plt.errorbar(x_new[:, tt], y_new[:, tt], xerr=xe_new[:, tt], yerr=ye_new[:, tt], fmt='r.')
        plt.xticks(np.arange(np.min(x_old - maxErr_x), np.max(x_old + maxErr_x), 0.4))
        plt.axis('equal')
        paxes.tick_params(axis='both', which='major', labelsize=fontsize1)
        paxes.xaxis.set_major_formatter(fmtY)
        paxes.yaxis.set_major_formatter(fmtY)
        plt.xlabel('X (pix)', fontsize=fontsize1)
        plt.ylabel('Y (pix)', fontsize=fontsize1)
        
    plt.subplots_adjust(wspace=0.4, hspace=0.4, left = 0.15, bottom = 0.1, right=0.9, top=0.9)
    plt.savefig(root_dir + 'plots/plot_three_stars_old_new' + outsuffix + '.png')
    plt.show()

    return

def check_final_chi2_distribution(only_stars_in_fit=False):
    # Load up the list of stars used in the transformation.	
    s = starset.StarSet(an_dir + 'align/align_t')
    s.loadPolyfit(poly_dir + 'fit', accel=0, arcsec=0)
    s.loadPoints(points_dir, useMJD=True)
    s.loadStarsUsed()

    trans = Table.read(an_dir + 'align/align_t.trans', format='ascii')
    N_par = trans['NumParams'][0]
    
    # Keep only stars detected in all epochs.
    cnt = s.getArray('velCnt')
    used = s.getArray('isUsed')
    N_epochs = cnt.max()

    idx = np.where(cnt == N_epochs)[0]
    msg = 'Keeping {0:d} of {1:d} stars in all epochs'

    if only_stars_in_fit:
        isUsed = s.getArrayFromAllEpochs('isUsed')
        cnt_used = isUsed.sum(axis=0)
        idx = np.where(cnt_used == cnt_used.max())[0]

        msg += ' and used'
        
    newstars = [s.stars[i] for i in idx]
    s.stars = newstars
    N_stars = len(newstars)
    print( msg.format(N_stars, len(cnt)) )

    # Now that we have are final list of stars, fetch all the
    # relevant variables.     
    cnt = s.getArray('velCnt')

    x = s.getArrayFromAllEpochs('pnt_x')
    y = s.getArrayFromAllEpochs('pnt_y')
    xe = s.getArrayFromAllEpochs('pnt_xe')
    ye = s.getArrayFromAllEpochs('pnt_ye')
    isUsed = s.getArrayFromAllEpochs('isUsed')

    x0 = s.getArray('fitpXv.p')
    vx = s.getArray('fitpXv.v')
    t0x = s.getArray('fitpXv.t0')
    y0 = s.getArray('fitpYv.p')
    vy = s.getArray('fitpYv.v')
    t0y = s.getArray('fitpYv.t0')

    m = s.getArray('mag')

    N_epochs = x.shape[0]
    N_stars = x.shape[1]
    year = np.zeros(N_epochs, dtype=float)

    # Make some output variables with the right shape/size.
    chi2x = np.zeros((N_epochs, N_stars), dtype=float)
    chi2y = np.zeros((N_epochs, N_stars), dtype=float)
    chi2 = np.zeros((N_epochs, N_stars), dtype=float)

    xresid = np.zeros((N_epochs, N_stars), dtype=float)
    yresid = np.zeros((N_epochs, N_stars), dtype=float)
    resid = np.zeros((N_epochs, N_stars), dtype=float)

    mjd = year_to_mjd(s.years)

    for ee in range(N_epochs):
        # Everything below should be arrays sub-indexed by "idx"
        dt_x = mjd[ee] - t0x
        dt_y = mjd[ee] - t0y

        x_fit = x0 + (vx * dt_x)
        y_fit = y0 + (vy * dt_y)

        xresid[ee, :] = x[ee, :] - x_fit
        yresid[ee, :] = y[ee, :] - y_fit

    # Calculate chi^2 (contains all sources of error)
    chi2x = ((xresid / xe)**2).sum(axis=0)
    chi2y = ((yresid / ye)**2).sum(axis=0)
    chi2 = chi2x + chi2y

    # Total residuals for each star.
    resid = np.hypot(xresid, yresid)

    # Total error for each star.
    xye = np.hypot(xresid * xe, yresid * ye) / resid

    # Figure out the number of degrees of freedom expected
    # for this chi^2 distribution.
    # N_data = N_stars * N_epochs * 2.0   # times 2 for X and Y measurements
    # N_free = N_par * N_epochs * 2.0
    N_data = N_epochs * 2.0   # times 2 for X and Y measurements
    N_free = 4.0
    N_dof = N_data - N_free
    N_dof_1 = N_dof / 2.0
    print( 'N_data = {0:.0f}  N_free = {1:.0f}  N_dof = {2:.0f}'.format(N_data, N_free, N_dof))

    # Setup some bins for making chi2 and residuals histograms
    chi2_lim = int(np.ceil(chi2[np.isfinite(chi2)].max()))
    if chi2_lim > (N_dof * 20):
        chi2_lim = N_dof * 20
    chi2_bin = chi2_lim / 20.0
    chi2_bins = np.arange(0, chi2_lim, chi2_bin)
    chi2_mod_bin = 0.25
    chi2_mod_bins = np.arange(0, chi2_lim, chi2_mod_bin)

    res_lim = 1.1 * resid.max()
    if res_lim > 1:
        res_lim = 1
    res_bin = 2.0 * res_lim / 40.0
    res_bins = np.arange(-res_lim, res_lim + res_bin, res_bin)

    sig_lim = 1.1 * (resid / xye).max()
    if sig_lim > 6:
        sig_lim = 6
    sig_bin = 2.0 * sig_lim / 20.0
    sig_bins = np.arange(-sig_lim, sig_lim + sig_bin, sig_bin)
    sig_mod_bin = sig_bin / 10.0
    sig_mod_bins = np.arange(-sig_lim, sig_lim + sig_mod_bin, sig_mod_bin)

    # Setup theoretical chi^2 distributions for X, Y, total.
    chi2_dist_a = scipy.stats.chi2(N_dof)
    chi2_dist_1 = scipy.stats.chi2(N_dof_1)
    chi2_plot_a = chi2_dist_a.pdf(chi2_mod_bins)
    chi2_plot_1 = chi2_dist_1.pdf(chi2_mod_bins)
    chi2_plot_a *= N_stars * chi2_bin
    chi2_plot_1 *= N_stars * chi2_bin

    # Setup theoretical normalized residuals distribution for X and Y
    sig_plot_1 = scipy.stats.norm.pdf(sig_mod_bins)
    sig_plot_1 *= N_stars * N_epochs * sig_bin / (sig_plot_1 * sig_mod_bin).sum()

    ##########
    # Plot Chi^2 Distribution
    ##########
    plt.clf()
    fig = plt.gcf()
    fig.set_size_inches(12, 12, forward=True)
    plt.subplots_adjust(bottom=0.08, left=0.08, top=0.95, wspace=0.28, hspace=0.32)

    ax_chi2 = plt.subplot(3, 3, 1)
    plt.hist(chi2x, bins=chi2_bins, color='blue')
    plt.plot(chi2_mod_bins, chi2_plot_1, 'k--')
    plt.xlabel(r'$\chi^2$')
    plt.ylabel(r'N$_{obs}$')
    plt.title('X')

    plt.subplot(3, 3, 2, sharex=ax_chi2, sharey=ax_chi2)
    plt.hist(chi2y, bins=chi2_bins, color='green')
    plt.plot(chi2_mod_bins, chi2_plot_1, 'k--')
    plt.xlabel(r'$\chi^2$')
    plt.title('Y')

    plt.subplot(3, 3, 3, sharex=ax_chi2, sharey=ax_chi2)
    plt.hist(chi2.flatten(), bins=chi2_bins, color='red')
    plt.plot(chi2_mod_bins, chi2_plot_a, 'k--')
    plt.xlabel(r'$\chi^2$')
    plt.title('X and Y')

    ax_res = plt.subplot(3, 3, 4)
    plt.hist(xresid.flatten(), bins=res_bins, color='blue')
    plt.xlabel('X Residuals (mas)')
    plt.ylabel(r'N$_{obs}$')
    plt.xlim(-res_lim, res_lim)

    plt.subplot(3, 3, 5, sharex=ax_res, sharey=ax_res)
    plt.hist(yresid.flatten(), bins=res_bins, color='green')
    plt.xlabel('Y Residuals (mas)')
    plt.xlim(-res_lim, res_lim)

    plt.subplot(3, 3, 6, sharey=ax_res)
    plt.hist(resid.flatten(), bins=res_bins, color='red')
    plt.xlim(0, res_lim)
    plt.xlabel('Total Residuals (mas)')

    ax_nres = plt.subplot(3, 3, 7)
    plt.hist((xresid / xe).flatten(), bins=sig_bins, color='blue')
    plt.plot(sig_mod_bins, sig_plot_1, 'k--')
    plt.xlabel('Normalized X Res.')

    plt.subplot(3, 3, 8, sharex=ax_nres, sharey=ax_nres)
    plt.hist((yresid / ye).flatten(), bins=sig_bins, color='green')
    plt.plot(sig_mod_bins, sig_plot_1, 'k--')
    plt.xlabel('Normalized Y Res.')

    plt.subplot(3, 3, 9, sharey=ax_nres)
    plt.hist((resid / xye).flatten(), bins=sig_bins, color='red')
    plt.xlim(0, sig_lim)
    plt.xlabel('Normalized Total Res.')

    fileUtil.mkdir(root_dir + 'plots/')

    outfile = an_dir + 'plots/'
    outfile += 'chi2_dist_final'
    if only_stars_in_fit:
        outfile += '_infit'
    outfile += '.png'
    print( outfile)

    plt.show()
    plt.savefig(outfile)

    return
    


def plot_three_stars():
    """Plot three panel figure with the target and two nearby reference 
    stars with their best-fit linear motions on top.
    """
    scale = 9.952 # mas/pix
    
    # Load up the points files
    targets = np.array(['OB120169', 'OB120169_L', 'S24_18_0.8'])
    pts_table = []

    for tt in targets:
        pts = starTables.read_points(points_dir + tt + '.points')

        pts_table.append(pts)

    # Load up the models from polyfit.
    pfit = Table.read(poly_dir + 'fit.linearFormal', format='ascii')
    pfit_t0 = Table.read(poly_dir + 'fit.lt0', format='ascii')

    idx = []
    for nn in range(len(pfit['star_name'])):
        if pfit['star_name'][nn] in targets:
            idx.append(nn)
    pfit = pfit[idx]
    pfit_t0 = pfit_t0[idx]


    plt.close(1)
    f1 = plt.figure(1, figsize=(12, 4))
    f1.clf()

    # plt.close(2)
    # f2 = plt.figure(2, figsize=(12, 12))
    # f2.clf()
    
    for tt in range(len(targets)):
        pts = pts_table[tt]
        
        # Convert the positions to units of mas offset from the mean position.
        x0 = pts['x'].mean()
        y0 = pts['y'].mean()

        t_obs = pts['epoch']
        x_obs = (pts['x'] - x0) * scale * -1.0
        y_obs = (pts['y'] - y0) * scale
        xe_obs = pts['xerr'] * scale
        ye_obs = pts['yerr'] * scale

        t_mod = np.arange(t_obs.min() - 200, t_obs.max() + 200)
        dtx = t_mod - pfit_t0[tt]['t0X']
        dty = t_mod - pfit_t0[tt]['t0Y']
        x_mod = pfit[tt]['a_x0'] + pfit[tt]['a_x1'] * dtx
        y_mod = pfit[tt]['a_y0'] + pfit[tt]['a_y1'] * dty
        xe_mod = np.sqrt( pfit[tt]['sig_a_x0']**2 + (pfit[tt]['sig_a_x1'] * dtx)**2 )
        ye_mod = np.sqrt( pfit[tt]['sig_a_y0']**2 + (pfit[tt]['sig_a_y1'] * dty)**2 )
        x_mod = (x_mod - x0) * scale * -1.0
        y_mod = (y_mod - y0) * scale
        xe_mod *= scale
        ye_mod *= scale

        # Plot
        ax1 = f1.add_subplot(1, 3, tt+1)
        ax1.errorbar(x_obs, y_obs, xerr=xe_obs, yerr=ye_obs, fmt='k.')
        ax1.plot(x_mod, y_mod, 'b-')
        ax1.plot(x_mod - xe_mod, y_mod + ye_mod, 'b--')
        ax1.plot(x_mod + xe_mod, y_mod - ye_mod, 'b--')
        ax1.set_title(targets[tt])
        # ax.axis('equal')

        # ax2_1 = f2.add_subplot(3, 3, 1 + tt)
        # ax2_1.errorbar(t_obs, x_obs, yerr=xe_obs, fmt='k.')
        # ax2_1.plot(t_mod, x_mod, 'b-')
        # ax2_1.plot(t_mod, x_mod - xe_mod, 'b--')
        # ax2_1.plot(t_mod, x_mod + xe_mod, 'b--')
        # ax2_1.set_title(targets[tt])

        # ax2_2 = f2.add_subplot(3, 3, 4 + tt)
        # ax2_2.errorbar(t_obs, y_obs, yerr=ye_obs, fmt='k.')
        # ax2_2.plot(t_mod, y_mod, 'b-')
        # ax2_2.plot(t_mod, y_mod - ye_mod, 'b--')
        # ax2_2.plot(t_mod, y_mod + ye_mod, 'b--')

        # ax2_3 = f2.add_subplot(3, 3, 7 + tt)
        # ax2_3.errorbar(x_obs, y_obs, xerr=xe_obs, yerr=ye_obs, fmt='k.')
        # ax2_3.plot(x_mod, y_mod, 'b-')
        # ax2_3.plot(x_mod - xe_mod, y_mod + ye_mod, 'b--')
        # ax2_3.plot(x_mod + xe_mod, y_mod - ye_mod, 'b--')
        
        
    f1.subplots_adjust(left=0.1, bottom=0.2)
    # f2.subplots_adjust(left=0.1, bottom=0.2)

    return

def plot_three_stars_big():
    """Call residuals.plotStar
    """
    targets = np.array(['OB120169', 'OB120169_L', 'S24_18_0.8'])
    
    residuals.plotStar(targets, rootDir=an_dir, align='align/align_t',
                       poly='polyfit_a/fit', points='points_a/',
                       figsize=(15,8))

    return


def year_to_mjd(years):
    """Convert year to MJD.
    """
    return ((years - 1999.0) * 365.242) + 51179.0


def plot_lens_geometry_ob110022():
    raL = 17 + (53. / 60.) + (17.93 / 3600.)
    decL = -30 + (2. / 60.) + (29.3 / 3600.)
    mL = 0.67          # Msun
    thetaE = 2.19      # mas
    t0 = 2455687.91    # JD
    t0 -= 2400000.5    # MJD
    xS0 = np.array([0.0003, 0.002])          # arcsec
    u0 = 0.574         # Einstein radii
    beta = u0 * thetaE # Einstein radii
    imag_base = 16.26  # I mag

    piE = np.array([-0.393, -0.071])  # Einstein radii
    tE = 61.4                         # piRel
    piRel = np.hypot(piE[0], piE[1]) * thetaE    # mas

    muS = np.array([4.06, 3.02])    # mas/yr
    muRel_amp = 12.0
    muRel = piE * thetaE * muRel_amp / piRel
    
    muL = muS - muRel               # mas/yr
    
    dS = 6000.0    # pc
    dL = 1.0 / ((1.0 / dS) + (piRel * 1e-3))

    plot_lens_geometry(raL, decL, mL, t0, xS0, beta, muL, muS, dL, dS, imag_base)

    return
    
def plot_lens_geometry(raL, decL, mL, t0, xS0, beta, muL, muS, dL, dS, imag_base):
    mod = model.PSPL_parallax(raL, decL, mL, t0, xS0, beta, muL, muS, dL, dS, imag_base)
    print('thetaE = ', mod.thetaE, mod.thetaE_amp)
    print('piRel / thetaE = ', mod.piRel / mod.thetaE_amp)
    print('thetaS0 = ', mod.thetaS0[0], mod.thetaS0[1])
    print('muL = ', mod.muL[0], mod.muL[1])
    print('muS = ', mod.muS[0], mod.muS[1])
    print('muRel = ', mod.muRel[0], mod.muRel[1])
    print('u0 = ', mod.u0_amp)
    print('piRel = ', mod.piRel)
    print('piE = ', mod.piE)
    print('dL = ', dL, ' dS = ', dS)

    # Astrometry of the unlensed source.
    t_obs = np.arange(t0 - 1000, t0 + 1000, 10)
    xS_unlens = mod.get_astrometry_unlensed(t_obs)
    xL_unlens = mod.get_lens_astrometry(t_obs)
    xS_restL0 = xS_unlens - mod.xL0

    tidx = np.argmin(np.abs(t_obs - t0))
    
    vel_scale = 0.05
    axis_lim_scale = 2.0
    
    plt.clf()
    cir = plt.Circle((xL_unlens[tidx, 0]*1e3, xL_unlens[tidx, 1]*1e3), mod.thetaE_amp,
                         color='red', fill=False)
    plt.gca().add_artist(cir)
    plt.plot([xL_unlens[tidx, 0]*1e3], [xL_unlens[tidx, 1]*1e3], 'ko')
    plt.arrow(xL_unlens[tidx, 0]*1e3, xL_unlens[tidx, 1]*1e3, mod.muL[0]*vel_scale, mod.muL[1]*vel_scale,
                  width=0.04, head_width=0.2, head_length=0.2, fc='k', ec='k')
    plt.arrow(xS_unlens[tidx, 0]*1e3, xS_unlens[tidx, 1]*1e3, mod.muS[0]*vel_scale, mod.muS[1]*vel_scale,
                  width=0.04, head_width=0.2, head_length=0.2, fc='b', ec='b')
    plt.plot(xS_unlens[:,0]*1e3, xS_unlens[:,1]*1e3, 'b--')
    plt.plot(xL_unlens[:,0]*1e3, xL_unlens[:,1]*1e3, 'k--')
    plt.xlabel(r'$\Delta\alpha$ (mas)')
    plt.ylabel(r'$\Delta\delta$ (mas)')
    plt.axis('equal')
    plt.xlim(mod.thetaE_amp * axis_lim_scale, -mod.thetaE_amp * axis_lim_scale)
    plt.ylim(-mod.thetaE_amp * axis_lim_scale, mod.thetaE_amp * axis_lim_scale)
    plt.savefig('/Users/jlu/doc/present/2017_02_stsci/OB110022/geometry.png')

    from jlu.microlens import test_model
    test_model.test_pspl_parallax(raL, decL, mL, t0, xS0, beta, muS, muL, dL, dS, imag_base,
                                      outdir='/Users/jlu/doc/present/2017_02_stsci/OB110022/')    

    return
