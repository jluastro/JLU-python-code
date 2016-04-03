import numpy as np
from astropy.table import Table
from astropy.modeling import polynomial as poly
from flystar import align
from scipy import stats
import pylab as py
import pdb

def monte_carlo():
    N_sims = 100

    chi2_1 = np.zeros(N_sims, dtype=float)
    chi2_red_1 = np.zeros(N_sims, dtype=float)
    N_dof_1 = np.zeros(N_sims, dtype=float)
    chi2_2 = np.zeros(N_sims, dtype=float)
    chi2_red_2 = np.zeros(N_sims, dtype=float)
    N_dof_2 = np.zeros(N_sims, dtype=float)
    ftest = np.zeros(N_sims, dtype=float)
    p_value = np.zeros(N_sims, dtype=float)
    
    for nn in range(N_sims):
        print '***'
        print '*** N_sim = ', nn
        print '***'
        out = test_order()

        chi2_1[nn] = out[0]
        chi2_red_1[nn] = out[1]
        N_dof_1[nn] = out[2]
        chi2_2[nn] = out[3]
        chi2_red_2[nn] = out[4]
        N_dof_2[nn] = out[5]
        ftest[nn] = out[6]
        p_value[nn] = out[7]

    binsize = (chi2_red_1.max() - chi2_red_1.min()) / 20.0
    bins = np.arange(0, chi2_red_1.max(), binsize)
    py.clf()
    py.hist(chi2_red_1, histtype='step', label=r'M=1 (\nu=134)', bins=bins)
    py.hist(chi2_red_2, histtype='step', label=r'M=2 (\nu=128)', bins=bins)
    py.xlabel(r'\chi^2_{red}')
    py.ylabel('Number of Sims')
    py.legend()
    py.savefig('sim_mc_chi2.png')

    py.clf()
    py.hist(ftest, bins=np.arange(-13, 13, 1), histtype='step')
    py.xlabel('F Statistic')
    py.ylabel('Number of Sims')
    py.savefig('sim_mc_ftest.png')

    bins = np.logspace(-3, 0, 20)
    py.clf()
    idx = np.where(ftest > 0)[0]
    py.hist(p_value[idx], bins=bins, histtype='step')
    py.gca().set_xscale('log')
    py.xlabel('p_value')
    py.ylabel('Number of Sims')
    py.title('Dropped {0:3d} Invalid p-values'.format(len(ftest) - len(idx)))
    py.savefig('sim_mc_ftest_pvalue.png')
    

def test_order():
    root_dir = '/Users/jlu/work/microlens/OB110022/analysis_2016_03_18/'

    # Load up real data.
    cat = Table.read(root_dir + 'ref_list.fits')

    N_stars = 70

    # Make a set of stars at random positions over a 1024 x 1024 grid
    x_epo1 = np.random.rand(N_stars) * 1024.0
    y_epo1 = np.random.rand(N_stars) * 1024.0

    # Propogate to epoch 2 with some small velocities.
    scale = 0.01  # arcsec/pixel
    vmean = 0.0    # arcsec/yr
    vdisp = 0.004  # arcsec/yr
    dt = 1.0 # years

    vmean *= dt / scale # convert to pixels
    vdisp *= dt / scale # convert to pixels

    # With velocities
    # x_epo2 = x_epo1 + (np.random.randn(N_stars) * vdisp) + vmean
    # y_epo2 = y_epo1 + (np.random.randn(N_stars) * vdisp) + vmean
    # No velocities
    x_epo2 = x_epo1
    y_epo2 = y_epo1

    # Make a couple of Polynomial2D transformations and apply them to both epochs.
    # Make them slightly different for each epoch.
    # P(x,y) = c0_0 + ( c1_0 * x ) + ( c0_1 *  y )
    t1_x_coeffs = {'c0_0':  0.0, 'c1_0': 0.993, 'c0_1': 0.004}
    t1_y_coeffs = {'c0_0':  0.0, 'c1_0': -0.004, 'c0_1': 0.993}
    # t1_x_coeffs = {'c0_0':  0.0, 'c1_0': 1.0, 'c0_1': 0.0}
    # t1_y_coeffs = {'c0_0':  0.0, 'c1_0': 0.0, 'c0_1': 1.0}
    t2_x_coeffs = {'c0_0': 15.0, 'c1_0': 0.998, 'c0_1': 0.012}
    t2_y_coeffs = {'c0_0': -7.0, 'c1_0':-0.012, 'c0_1': 0.998}
    
    poly2d_t1_x = poly.Polynomial2D(degree=1, **t1_x_coeffs)
    poly2d_t1_y = poly.Polynomial2D(degree=1, **t1_y_coeffs)
    poly2d_t2_x = poly.Polynomial2D(degree=1, **t2_x_coeffs)
    poly2d_t2_y = poly.Polynomial2D(degree=1, **t2_y_coeffs)

    x_t1 = poly2d_t1_x(x_epo1, y_epo1)
    y_t1 = poly2d_t1_y(x_epo1, y_epo1)
    x_t2 = poly2d_t2_x(x_epo2, y_epo2)
    y_t2 = poly2d_t2_y(x_epo2, y_epo2)

    # Assign error bars from the real data.
    idx1 = np.array(np.random.rand(N_stars) * len(cat), dtype=int)
    idx2 = np.array(np.random.rand(N_stars) * len(cat), dtype=int)

    # xe_t1 = np.ones(N_stars) * 0.05
    # ye_t1 = np.ones(N_stars) * 0.05
    # xe_t2 = np.ones(N_stars) * 0.05
    # ye_t2 = np.ones(N_stars) * 0.05

    # xe_t1 = np.random.randn(N_stars) * 0.05
    # ye_t1 = np.random.randn(N_stars) * 0.05
    # xe_t2 = np.random.randn(N_stars) * 0.05
    # ye_t2 = np.random.randn(N_stars) * 0.05
    
    xe_t1 = cat['xe'][idx1]
    ye_t1 = cat['ye'][idx1]
    xe_t2 = cat['xe'][idx2]
    ye_t2 = cat['ye'][idx2]

    # Perturb both data sets by error bars.
    x_t1 += np.random.randn(N_stars) * xe_t1
    y_t1 += np.random.randn(N_stars) * ye_t1
    x_t2 += np.random.randn(N_stars) * xe_t2
    y_t2 += np.random.randn(N_stars) * ye_t2
    
    ##
    ## DONE WITH MAKING FAKE DATA
    ##  -- start fitting the fake data
    ## 
    
    # Solve for parameters, p, to transform T(x1 | p) --> x2
    weights = 1.0 / (xe_t1**2 + ye_t1**2 + xe_t2**2 + ye_t2**2)
    tfit_1 = align.transforms.PolyTransform(x_t1, y_t1, x_t2, y_t2, order=1, weights=weights)
    tfit_2 = align.transforms.PolyTransform(x_t1, y_t1, x_t2, y_t2, order=2, weights=weights)

    # Calculate the residuals and chi-squared for each of the transformations.
    xfit_1, yfit_1 = tfit_1.evaluate(x_t1, y_t1)
    xfit_2, yfit_2 = tfit_2.evaluate(x_t1, y_t1)

    # **Note**  the error propogation isn't quite right. Just use the first order transformation.
    xefit_1, yefit_1 = tfit_1.evaluate(xe_t1, ye_t1)
    xefit_2, yefit_2 = tfit_1.evaluate(xe_t1, ye_t1)
    xefit_1 -= tfit_1.px.parameters[0]
    yefit_1 -= tfit_1.py.parameters[0]
    xefit_2 -= tfit_1.px.parameters[0]
    yefit_2 -= tfit_1.py.parameters[0]

    # Calculate the residuals, chi**2 distributions, F test
    xres_1 = xfit_1 - x_t2
    yres_1 = yfit_1 - y_t2
    xres_2 = xfit_2 - x_t2
    yres_2 = yfit_2 - y_t2

    chi2_1 = (xres_1**2 / (xefit_1**2 + xe_t2**2)).sum() + (yres_1**2 / (yefit_1**2 + ye_t2**2)).sum()
    chi2_2 = (xres_2**2 / (xefit_2**2 + xe_t2**2)).sum() + (yres_2**2 / (yefit_2**2 + ye_t2**2)).sum()

    N_data = 2 * N_stars
    N_freepar_1 = 2 * 3
    N_freepar_2 = 2 * 6

    N_dof_1 = N_data - N_freepar_1
    N_dof_2 = N_data - N_freepar_2

    ftest_dof1 = N_freepar_2 - N_freepar_1
    ftest_dof2 = N_data - N_freepar_2
    ftest_dof_ratio = 1.0 / (1.0 * ftest_dof1 / ftest_dof2)
    ftest = (-1 * (chi2_2 - chi2_1) / chi2_2) * ftest_dof_ratio

    p_value = stats.f.sf(ftest, ftest_dof1, ftest_dof2)

    print 'N_stars = ', N_stars
    print 'Positional Errors, Epoch 1'
    print '    X: Mean = {0:6.3f}, Median={1:6.3f}'.format(xe_t1.mean(), np.median(xe_t1))
    print '    Y: Mean = {0:6.3f}, Median={1:6.3f}'.format(ye_t1.mean(), np.median(ye_t1))
    print 'Positional Errors, Epoch 2'
    print '    X: Mean = {0:6.3f}, Median={1:6.3f}'.format(xe_t2.mean(), np.median(xe_t2))
    print '    Y: Mean = {0:6.3f}, Median={1:6.3f}'.format(ye_t2.mean(), np.median(ye_t2))
    print 'Input Coefficients:'
    print '    X, Epoch 1: ', t1_x_coeffs
    print '    Y, Epoch 1: ', t1_y_coeffs
    print '    X, Epoch 2: ', t2_x_coeffs
    print '    Y, Epoch 2: ', t2_y_coeffs
    print 'Output Coefficients 1st Order Fit:'
    print '    X: {0:7.3f} {1:7.3f} {2:7.3f}'.format( tfit_1.px.parameters[0], tfit_1.px.parameters[1], tfit_1.px.parameters[2])
    print '    Y: {0:7.3f} {1:7.3f} {2:7.3f}'.format( tfit_1.py.parameters[0], tfit_1.py.parameters[1], tfit_1.py.parameters[2])
    print 'Output Coefficients 2nd Order Fit:'
    print '    X:  {0:7.3f} {1:7.3f} {2:7.3f} {3:9.2e} {4:9.2e} {5:9.2e}'.format(tfit_2.px.parameters[0], tfit_2.px.parameters[1], tfit_2.px.parameters[3],
                                                                                 tfit_2.px.parameters[2], tfit_2.px.parameters[4], tfit_2.px.parameters[5])
    print '    Y:  {0:7.3f} {1:7.3f} {2:7.3f} {3:9.2e} {4:9.2e} {5:9.2e}'.format(tfit_2.py.parameters[0], tfit_2.py.parameters[1], tfit_2.py.parameters[3],
                                                                                 tfit_2.py.parameters[2], tfit_2.py.parameters[4], tfit_2.py.parameters[5])
    print 'Residuals After 1st Order Fit (M=1)'
    print '    X: Mean = {0:6.3f}, Median = {1:6.3f}, STD = {2:6.3f}'.format(xres_1.mean(), np.median(xres_1), xres_1.std())
    print '    Y: Mean = {0:6.3f}, Median = {1:6.3f}, STD = {2:6.3f}'.format(yres_1.mean(), np.median(yres_1), yres_1.std())
    print 'Residuals After 2nd Order Fit (M=2)'
    print '    X: Mean = {0:6.3f}, Median = {1:6.3f}, STD = {2:6.3f}'.format(xres_2.mean(), np.median(xres_2), xres_2.std())
    print '    Y: Mean = {0:6.3f}, Median = {1:6.3f}, STD = {2:6.3f}'.format(yres_2.mean(), np.median(yres_2), yres_2.std())
    print 'Chi-Squared Values:'
    print '    M = 1: Chi**2 = {0:6.2f}  ({1:6.2f} reduced, {2:3d} DOF)'.format(chi2_1, chi2_1 / N_dof_1, N_dof_1)
    print '    M = 2: Chi**2 = {0:6.2f}  ({1:6.2f} reduced, {2:3d} DOF)'.format(chi2_2, chi2_2 / N_dof_2, N_dof_2)
    print 'ftest_dof1: ', ftest_dof1
    print 'ftest_dof2: ', ftest_dof2
    print 'ftest_dof_ratio: ', ftest_dof_ratio
    print 'ftest: ', ftest
    fmt = 'M = 2: M-1 --> M   F = {0:5.2f} p = {1:7.4f}'
    print fmt.format(ftest, p_value)

    # Plotting
    bins = np.arange(-10, 10, 0.5)
    py.figure(1)
    py.clf()
    py.subplots_adjust(hspace=0.25)
    py.subplot(2, 1, 1)
    py.hist(xres_1 / np.hypot(xefit_1, xe_t2), bins=bins, label='X', histtype='step')
    py.hist(yres_1 / np.hypot(yefit_1, ye_t2), bins=bins, label='Y', histtype='step')
    py.title('1st order')
    py.ylabel('Number of Stars')
    py.legend()
    
    py.subplot(2, 1, 2)
    py.hist(xres_2 / np.hypot(xefit_2, xe_t2), bins=bins, label='X', histtype='step')
    py.hist(yres_2 / np.hypot(yefit_2, ye_t2), bins=bins, label='Y', histtype='step')
    py.title('2nd order')
    py.xlabel('Residuals (sigma)')
    py.ylabel('Number of Stars')
    py.legend()

    py.figure(2)
    py.clf()
    py.quiver(x_t2, y_t2, xres_1, yres_1, color='black', label='1st order')
    py.quiver(x_t2, y_t2, xres_2, yres_2, color='red', label='2nd order')
    py.legend()
    py.xlabel('X')
    py.ylabel('Y')


    out = [chi2_1, chi2_1 / N_dof_1, N_dof_1, chi2_2, chi2_2 / N_dof_2, N_dof_2, ftest, p_value]

    return out
