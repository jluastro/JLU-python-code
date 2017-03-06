import numpy as np
import pylab as plt
from scipy import stats
from jlu.microlens import residuals
from matplotlib.ticker import ScalarFormatter, FormatStrFormatter, MultipleLocator, NullFormatter

def align_residuals_vs_order(analysis_dirs, only_stars_in_fit=True, out_suffix='', plot_dir=''):
    data_all = []

    for aa in range(len(analysis_dirs)):
        data = residuals.check_alignment_fit(root_dir=analysis_dirs[aa])

        data_all.append(data)

    order = np.arange(1, len(data_all)+1)
    N_order = len(data_all)

    year = data_all[0]['year']
    scale = 9.952  # mas / pixel

    N_plots = len(year)
    if only_stars_in_fit:
        N_plots -= 1

    ##########
    # Plot Residuals for each epoch.
    ##########
    fig = plt.figure(2)
    fig.set_size_inches(15, 3.5, forward=True)
    plt.clf()
    plt.subplots_adjust(left=0.08, bottom=0.2, hspace=0, wspace=0, right=0.95, top=0.88)

    idx_subplot = 0
    ymax = 0

    majorLoc = MultipleLocator(1)
    majorFmt = FormatStrFormatter('%d')
    nullFmt = NullFormatter()

    for ee in range(len(year)):
        end = 'all'
        if only_stars_in_fit:
            end = 'used'
        
        xres = np.array([data_all[oo]['xres_rms_'+end][ee] for oo in range(N_order)])
        yres = np.array([data_all[oo]['yres_rms_'+end][ee] for oo in range(N_order)])
        ares = np.hypot(xres, yres)
    
        xres_e = np.array([data_all[oo]['xres_err_p_'+end][ee] for oo in range(N_order)])
        yres_e = np.array([data_all[oo]['yres_err_p_'+end][ee] for oo in range(N_order)])
        ares_e = np.hypot(xres * xres_e, yres * yres_e) / ares

        xres *= scale
        yres *= scale
        ares *= scale
        xres_e *= scale
        yres_e *= scale
        ares_e *= scale

        if only_stars_in_fit and np.isnan(xres[0]):
            print( 'Skipping Ref Epoch', ee)
            continue

        idx_subplot += 1

        # Residuals.        
        if idx_subplot == 1:
            ax1 = plt.subplot(1, N_plots, idx_subplot)
        else:
            plt.subplot(1, N_plots, idx_subplot, sharex=ax1)
        plt.errorbar(order, xres, yerr=xres_e, fmt='r.--', label='X')
        plt.errorbar(order, yres, yerr=yres_e, fmt='b.--', label='Y')
        plt.errorbar(order, ares, yerr=ares_e, fmt='g.--', label='Both')
        plt.title(data_all[0]['year'][ee])
        plt.xlim(0, 3.9)
        ax = plt.gca()
        ax.xaxis.set_major_locator(majorLoc)
        ax.xaxis.set_major_formatter(majorFmt)
        plt.xlabel('Order')

        ymax = np.ceil(np.max(ares.tolist() + [ymax]))
        plt.ylim(0, ymax)
        
        if idx_subplot == 1:
            #ax.yaxis.set_major_formatter(majorFmt)
            plt.ylabel('RMS Residuals (mas)')
        else:
            ax.yaxis.set_major_formatter(nullFmt)

        if idx_subplot == N_plots:
            plt.legend(numpoints=1)
            
    plt.show()
    plt.savefig(plot_dir + 'align_resi_vs_order_' + out_suffix + '.png')

    
    ##########
    # Plot Chi^2 for each epoch.
    ##########
    fig = plt.figure(3)
    fig.set_size_inches(15, 3.5, forward=True)
    plt.clf()
    plt.subplots_adjust(left=0.08, bottom=0.2, hspace=0, wspace=0, right=0.95, top=0.88)
    
    idx_subplot = 0
    ymax = 0

    majorLoc = MultipleLocator(1)
    majorFmt = FormatStrFormatter('%d')
    nullFmt = NullFormatter()

    for ee in range(len(year)):
        end = 'all'
        if only_stars_in_fit:
            end = 'used'
            
        chi2x = np.array([data_all[oo]['chi2x_'+end][ee] for oo in range(N_order)])
        chi2y = np.array([data_all[oo]['chi2y_'+end][ee] for oo in range(N_order)])
        chi2 = chi2x + chi2y
        N_stars = np.array([data_all[oo]['N_stars_'+end][ee] for oo in range(N_order)])

        if only_stars_in_fit and (N_stars[0] == 0):
            print( 'Skipping Ref Epoch', ee)
            continue

        idx_subplot += 1

        # Chi^2
        if idx_subplot == 1:
            ax1 = plt.subplot(1, N_plots, idx_subplot)
        else:
            plt.subplot(1, N_plots, idx_subplot, sharex=ax1)
        plt.plot(order, chi2x, 'r.--', label='X')
        plt.plot(order, chi2y, 'b.--', label='Y')
        plt.plot(order, chi2, 'g.--', label='Both')
        plt.title(data_all[0]['year'][ee])
        plt.xlim(0, 3.9)
        ax = plt.gca()
        ax.xaxis.set_major_locator(majorLoc)
        ax.xaxis.set_major_formatter(majorFmt)
        plt.xlabel('Order')

        ymax = np.max(chi2 + [ymax])
        plt.ylim(0, ymax+5)
        
        if idx_subplot == 1:
            plt.ylabel(r'$\chi^2$')
        else:
            ax.yaxis.set_major_formatter(nullFmt)

        if idx_subplot == N_plots:
            plt.legend(numpoints=1)

    plt.show()
    plt.savefig(plot_dir + 'align_chi2_vs_order_' + out_suffix + '.png')

    end = 'all'
    if only_stars_in_fit:
        end = 'used'
        
    N_par = np.array([3, 6, 10])

    outfile = plot_dir + 'align_resi_vs_order_' + out_suffix + '.txt'
    _out = open(outfile, 'w')
    
    # Fit Improvement Significance
    for ee in range(len(year)):
        chi2x = np.array([data_all[oo]['chi2x_'+end][ee] for oo in range(N_order)])
        chi2y = np.array([data_all[oo]['chi2y_'+end][ee] for oo in range(N_order)])
        chi2a = chi2x + chi2y
        N_stars = np.array([data_all[oo]['N_stars_'+end][ee] for oo in range(N_order)])

        if only_stars_in_fit and (N_stars[0] == 0):
            print( 'Skipping Ref Epoch', ee)
            continue

        # Print out some values for this epoch and each order.
        print( '')
        print( 'Epoch: {0:8.3f}'.format(year[ee]))
        _out.write('\n')
        _out.write('Epoch: {0:8.3f}'.format(year[ee]))
        for mm in range(N_order):
            fmt = 'N_stars = {0:3d}  N_par = {1:3d}  N_dof = {2:3d}  Chi^2 X = {3:5.1f}  Chi^2 Y = {4:5.1f}  Chi^2 Tot = {5:5.1f}'
            print( fmt.format(int(N_stars[mm]), int(N_par[mm]), int(N_stars[mm] - N_par[mm]),
                             chi2x[mm], chi2y[mm], chi2a[mm]))
            
            fmt = 'N_stars = {0:3d}  N_par = {1:3d}  N_dof = {2:3d}  Chi^2 X = {3:5.1f}  Chi^2 Y = {4:5.1f}  Chi^2 Tot = {5:5.1f}\n'
            _out.write(fmt.format(int(N_stars[mm]), int(N_par[mm]), int(N_stars[mm] - N_par[mm]),
                             chi2x[mm], chi2y[mm], chi2a[mm]))
    
        # F-test for X and Y and total.
        # These have shape: len(order) - 1
        ftest_m = order[1:]
        ftest_dof1 = np.diff(N_par)
        ftest_dof2 = N_stars[1:] - N_par[1:]
        ftest_dof_ratio = 1.0 / (ftest_dof1 / ftest_dof2)
        ftest_x = (-1 * np.diff(chi2x) / chi2x[1:]) * ftest_dof_ratio
        ftest_y = (-1 * np.diff(chi2y) / chi2y[1:]) * ftest_dof_ratio
        ftest_a = (-1 * np.diff(chi2a) / chi2a[1:]) * ftest_dof_ratio

        px_value = np.zeros(len(ftest_m), dtype=float)
        py_value = np.zeros(len(ftest_m), dtype=float)
        pa_value = np.zeros(len(ftest_m), dtype=float)

        for mm in range(len(ftest_m)):
            px_value[mm] = stats.f.sf(ftest_x[mm], ftest_dof1[mm], ftest_dof2[mm])
            py_value[mm] = stats.f.sf(ftest_y[mm], ftest_dof1[mm], ftest_dof2[mm])
            pa_value[mm] = stats.f.sf(ftest_a[mm], 2*ftest_dof1[mm], 2*ftest_dof2[mm])
            fmt = 'M = {0}: M-1 --> M     Fa = {1:5.2f} pa = {2:7.4f}    Fx = {3:5.2f} px = {4:7.4f}   Fy = {5:5.2f} py = {6:7.4f}'
            print( fmt.format(ftest_m[mm], ftest_a[mm], pa_value[mm],
                             ftest_x[mm], px_value[mm], ftest_y[mm], py_value[mm]))

            fmt = fmt + '\n'
            _out.write(fmt.format(ftest_m[mm], ftest_a[mm], pa_value[mm],
                             ftest_x[mm], px_value[mm], ftest_y[mm], py_value[mm]))

    ##########
    # Combine over all epochs NOT including velocities:
    ##########
    chi2x = np.array([data_all[oo]['chi2x_'+end].sum() for oo in range(N_order)])
    chi2y = np.array([data_all[oo]['chi2y_'+end].sum() for oo in range(N_order)])
    chi2 = chi2x + chi2y

    # Number of stars is constant for all orders.
    N_data = 2 * data_all[0]['N_stars_'+end].sum()
    # Number of free parameters in the alignment fit.
    N_afit = (len(year) - 1) * N_par * 2
    N_free_param = N_afit
    N_dof = N_data - N_free_param

    print( '')
    print( '*** Combined F test across epochs (NO velocities) ***')
    _out.write('\n')
    _out.write('*** Combined F test across epochs (NO velocities) ***\n')
    
    for mm in range(N_order):
        fmt = 'Order={0:d}  N_stars = {1:3d}  N_par = {2:3d}  N_dof = {3:3d}  Chi^2 = {4:5.1f}'
        print( fmt.format(order[mm], int(N_data), int(N_free_param[mm]), int(N_dof[mm]), chi2[mm]))
        fmt = fmt + '\n'
        _out.write(fmt.format(order[mm], int(N_data), int(N_free_param[mm]), int(N_dof[mm]), chi2[mm]))
        
    ftest_m = order[1:]
    ftest_dof1 = np.diff(N_free_param)
    ftest_dof2 = N_data - N_free_param[1:]
    ftest_dof_ratio = 1.0 / (ftest_dof1 / ftest_dof2)
    ftest = (-1 * np.diff(chi2) / chi2[1:]) * ftest_dof_ratio
        
    p_value = np.zeros(len(ftest_m), dtype=float)

    # print( 'ftest_m: ', ftest_m)
    # print( 'ftest_dof1: ', ftest_dof1)
    # print( 'ftest_dof2: ', ftest_dof2)
    # print( 'ftest_dof_ratio: ', ftest_dof_ratio)
    # print( 'ftest: ', ftest)
    
    for mm in range(len(ftest_m)):
        p_value[mm] = stats.f.sf(ftest[mm], ftest_dof1[mm], ftest_dof2[mm])
        fmt = 'M = {0}: M-1 --> M   F = {1:5.2f}  p = {2:7.4f}'
        print( fmt.format(ftest_m[mm], ftest[mm], p_value[mm]))
    
        fmt = fmt + '\n'
        _out.write(fmt.format(ftest_m[mm], ftest[mm], p_value[mm]))
    
            
    ##########
    # Combine over all epochs to include velocities:
    ##########

    # Number of stars is constant for all orders.
    N_data = 2 * data_all[0]['N_stars_'+end].sum()
    # Get the maximum number of stars used in any epoch... this is close enough?    
    N_stars_max = data_all[0]['N_stars_'+end].max()
    # Number of free parameters in the velocity fit.
    N_vfit = 4 * N_stars_max   # (x0, vx, y0, vy for each star)
    # Number of free parameters in the alignment fit.
    N_afit = (len(year) - 1) * N_par * 2

    N_free_param = N_vfit + N_afit
    N_dof = N_data - N_free_param

    print( '')
    print( '*** Combined F test across epochs (WITH velocities) ***')
    _out.write('\n')
    _out.write('*** Combined F test across epochs (WITH velocities) ***\n')
    
    for mm in range(N_order):
        fmt = 'Order={0:d}  N_stars = {1:3d}  N_par = {2:3d}  N_dof = {3:3d}  Chi^2 = {4:5.1f}'
        print( fmt.format(order[mm], int(N_data), int(N_free_param[mm]), int(N_dof[mm]), chi2[mm]))

        fmt = fmt + '\n'
        _out.write(fmt.format(order[mm], int(N_data), int(N_free_param[mm]), int(N_dof[mm]), chi2[mm]))
        
    ftest_m = order[1:]
    ftest_dof1 = np.diff(N_free_param)
    ftest_dof2 = N_data - N_free_param[1:]
    ftest_dof_ratio = 1.0 / (ftest_dof1 / ftest_dof2)
    ftest = (-1 * np.diff(chi2) / chi2[1:]) * ftest_dof_ratio
        
    p_value = np.zeros(len(ftest_m), dtype=float)

    for mm in range(len(ftest_m)):
        p_value[mm] = stats.f.sf(ftest[mm], ftest_dof1[mm], ftest_dof2[mm])
        fmt = 'M = {0}: M-1 --> M   F = {1:5.2f}  p = {2:7.4f}'
        print( fmt.format(ftest_m[mm], ftest[mm], p_value[mm]))
        
        fmt = fmt + '\n'
        _out.write(fmt.format(ftest_m[mm], ftest[mm], p_value[mm]))

    return

def align_residuals_vs_order_summary(root='/Users/jlu/work/microlens/OB120169/',
                                     prefix='a', target='ob120169', date='2016_10_10',
                                     orders=[3,4,5], weights=[1,2,3,4], Kcut=22, only_stars_in_fit=True):
    # Make 2D arrays for the runs with different Kcut and orders.
    orders = np.array(orders)
    weights = np.array(weights)

    end = 'all'
    if only_stars_in_fit:
        end = 'used'

    work_dir = root + prefix + '_' + date + '/'

    # Number of free parameters in the alignment fit for each order
    N_par_aln_dict = {3: 3, 4: 6, 5: 10}
    N_par_aln = [N_par_aln_dict[order] for order in orders]

    print( '********' )
    print( '* Results for Kcut={0:d} and in_fit={1}'.format(Kcut, only_stars_in_fit) )
    print( '********' )

    for tt in range(len(targets)):
        # make some variables we will need for the F-test
        N_free_all = np.zeros((len(weights), len(orders)), dtype=int)
        N_data_all = np.zeros((len(weights), len(orders)), dtype=int)
        chi2_all = np.zeros((len(weights), len(orders)), dtype=float)
        
        for ww in range(len(weights)):
            for oo in range(len(orders)):
                analysis_root_fmt = '{0:s}_{1:s}_{2:s}_a{3:d}_m{4:d}_w{5:d}_MC100/'
                analysis_dir = work_dir + analysis_root_fmt.format(prefix, target, date, orders[oo], Kcut, weights[ww])

                data = residuals.check_alignment_fit(root_dir=analysis_dir)

                year = data['year']
                scale = 9.952  # mas / pixel
        
                # Get the total chi^2 for all stars in all epochs
                chi2x = data['chi2x_' + end].sum()
                chi2y = data['chi2y_' + end].sum()
                chi2 = chi2x + chi2y

                ##########
                # Combine over all epochs including velocities:
                ##########

                # Number of stars is constant for all orders.
                N_data = 2 * data['N_stars_' + end].sum()
                # Get the maximum number of stars used in any epoch... this is close enough?    
                N_stars_max = data['N_stars_'+end].max()
                # Number of free parameters in the velocity fit.
                N_vfit = 4 * N_stars_max   # (x0, vx, y0, vy for each star)
                # Number of free parameters in the alignment fit.
                N_afit = (len(year) - 1) * N_par_aln[oo] * 2

                N_free_param = N_vfit + N_afit
                N_dof = N_data - N_free_param

                fmt = 'Target={0:s}  Order={1:d}  Weight={2:d} N_stars = {3:3d}  N_par = {4:3d}  N_dof = {5:3d}  Chi^2 = {6:5.1f}'
                print( fmt.format(targets[tt], orders[oo], weights[ww], int(N_data), int(N_free_param), int(N_dof), chi2) )

                N_free_all[ww, oo] = int(N_free_param)
                N_data_all[ww, oo] = int(N_data)
                chi2_all[ww, oo] = chi2
        
            # Done with all orders in this target... lets print F-test results
            ftest_m = orders[1:]
            ftest_dof1 = np.diff(N_free_all)
            ftest_dof2 = N_data_all[1:] - N_free_all[1:]
            ftest_dof_ratio = 1.0 / (1.0 * ftest_dof1 / ftest_dof2)
            ftest = (-1 * np.diff(chi2_all) / chi2_all[1:]) * ftest_dof_ratio
        
            p_value = np.zeros(len(ftest_m), dtype=float)

            for mm in range(len(ftest_m)):
                p_value[mm] = stats.f.sf(ftest[mm], ftest_dof1[mm], ftest_dof2[mm])
                fmt = 'M = {0}: M-1 --> M   F = {1:5.2f}  p = {2:7.4f}'
                print( fmt.format(ftest_m[mm], ftest[mm], p_value[mm]) )

    return
