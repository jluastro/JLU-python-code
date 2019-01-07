import matplotlib.pyplot as plt
import numpy as np
from astropy.table import Table, vstack
from matplotlib.patches import Ellipse
from matplotlib.ticker import ScalarFormatter
import matplotlib.ticker as ticker
from microlens.popsyn import synthetic
import h5py

##########
# Extended Data Table 4 from Mroz 2017
# Number of events detected in individual timescale bins
##########
log_tE = np.array([-0.93, -0.79, -0.65, -0.51, -0.37,
                   -0.23, -0.09, 0.05, 0.19, 0.33, 
                    0.47, 0.61, 0.75, 0.89, 1.03, 
                    1.17, 1.31, 1.45, 1.59, 1.73, 
                    1.87, 2.01, 2.15, 2.29, 2.43])

BLG500 = np.array([0, 0, 1, 0, 0,
                   0, 1, 1, 0, 3,
                   4, 10, 17, 22, 25,
                   26, 29, 23, 15, 12,
                   7, 3, 5, 0, 0])
                   
BLG501 = np.array([1, 0, 1, 1, 0,
                   1, 0, 1, 4, 5,
                   9, 19, 40, 32, 39,
                   35, 62, 42, 39, 25,
                   13, 9, 2, 0, 0])

BLG504 = np.array([0, 0, 0, 0, 0,
                   0, 0, 1, 0, 4,
                   7, 13, 17, 24, 30, 
                   46, 38, 39, 27, 20,
                   11, 6, 3, 1, 0])

BLG505 = np.array([0, 1, 0, 0, 0,
                   0, 0, 0, 2, 4,
                   8, 28, 39, 55, 78,
                   57, 62, 53, 40, 39,
                   20, 11, 2, 1, 1])

BLG506 = np.array([0, 0, 0, 0, 0,
                   0, 1, 0, 0, 1,
                   5, 10, 19, 26, 34,
                   44, 39, 32, 32, 19,
                   10, 6, 7, 3, 2])

BLG511 = np.array([0, 0, 0, 0, 0,
                   0, 0, 0, 4, 0,
                   8, 10, 11, 19, 22,
                   33, 30, 32, 24, 21,
                   10, 7, 1, 2, 0])

BLG512 = np.array([0, 1, 0, 0, 0,
                   0, 0, 0, 1, 3,
                   3, 13, 13, 28, 40,
                   46, 40, 41, 25, 31,
                   12, 3, 4, 3, 0])

BLG534 = np.array([0, 0, 0, 0, 0,
                   0, 0, 0, 1, 5,
                   8, 6, 17, 20, 22,
                   24, 24, 33, 20, 18,
                   6, 2, 2, 0, 0])
                   
BLG611 = np.array([0, 0, 0, 0, 0, 
                   0, 0, 0, 3, 2,
                   5, 3, 9, 20, 25,
                   23, 38, 36, 21, 11,
                   8, 4, 2, 1, 0])

##########
# Extended Data Table 5 from Mroz 2017
# Detection efficiencies for the analyzed fields
##########
BLG500de = np.array([0.0016, 0.0030, 0.0041, 0.0061, 0.0096,
                     0.0130, 0.0194, 0.0278, 0.0371, 0.0447,
                     0.0508, 0.0608, 0.0658, 0.0737, 0.0760, 
                     0.0858, 0.0872, 0.0949, 0.0964, 0.1024,
                     0.1000, 0.1029, 0.0989, 0.0853, 0.0618])

BLG501de = np.array([0.0033, 0.0071, 0.0086, 0.0118, 0.0144,
                     0.0209, 0.0279, 0.0365, 0.0423, 0.0506,
                     0.0557, 0.0630, 0.0669, 0.0746, 0.0769,
                     0.0826, 0.0831, 0.0898, 0.0940, 0.0973,
                     0.1004, 0.0965, 0.0928, 0.0788, 0.0539])

BLG504de = np.array([0.0021, 0.0046, 0.0061, 0.0089, 0.0126,
                     0.0176, 0.0255, 0.0368, 0.0461, 0.0559,
                     0.0630, 0.0701, 0.0750, 0.0855, 0.0910,
                     0.0939, 0.1026, 0.1099, 0.1145, 0.1192,
                     0.1207, 0.1182, 0.1122, 0.0979, 0.0638])

BLG505de = np.array([0.0045, 0.0078, 0.0110, 0.0139, 0.0180,
                     0.0248, 0.0343, 0.0423, 0.0506, 0.0571,
                     0.0692, 0.0753, 0.0816, 0.0876, 0.0940,
                     0.0950, 0.1014, 0.1055, 0.1108, 0.1134,
                     0.1174, 0.1124, 0.1072, 0.0890, 0.0538])

BLG506de = np.array([0.0015, 0.0043, 0.0057, 0.0086, 0.0120,
                     0.0189, 0.0299, 0.0396, 0.0495, 0.0593,
                     0.0675, 0.0784, 0.0866, 0.0937, 0.1011,
                     0.1035, 0.1079, 0.1184, 0.1191, 0.1264,
                     0.1288, 0.1253, 0.1148, 0.0998, 0.0596])

BLG511de = np.array([0.0016, 0.0038, 0.0057, 0.0077, 0.0119,
                     0.0181, 0.0278, 0.0388, 0.0486, 0.0596,
                     0.0680, 0.0758, 0.0832, 0.0907, 0.0949,
                     0.1035, 0.1067, 0.1151, 0.1212, 0.1249,
                     0.1254, 0.1218, 0.1146, 0.0914, 0.0560])

BLG512de = np.array([0.0039, 0.0085, 0.0126, 0.0144, 0.0186,
                     0.0297, 0.0381, 0.0503, 0.0603, 0.0705,
                     0.0790, 0.0876, 0.0940, 0.0990, 0.1056,
                     0.1113, 0.1131, 0.1206, 0.1286, 0.1302,
                     0.1336, 0.1331, 0.1160, 0.0906, 0.0548])

BLG534de = np.array([0.0013, 0.0033, 0.0047, 0.0068, 0.0095,
                     0.0160, 0.0226, 0.0335, 0.0395, 0.0484,
                     0.0592, 0.0641, 0.0746, 0.0772, 0.0838,
                     0.0899, 0.0913, 0.1012, 0.1048, 0.1105,
                     0.1111, 0.1085, 0.1029, 0.0888, 0.0578])

BLG611de = np.array([0.0013, 0.0041, 0.0053, 0.0084, 0.0121,
                     0.0180, 0.0290, 0.0390, 0.0525, 0.0631,
                     0.0755, 0.0863, 0.0874, 0.1025, 0.1107,
                     0.1204, 0.1252, 0.1361, 0.1389, 0.1470,
                     0.1525, 0.1500, 0.1458, 0.1295, 0.0891])

def plot_mroz2017():
    """
    Plot results from Mroz 2017.

    Returns
    -------
    NOTE: you can plot things over
    other things. Hence, to see the plot, you need to say
    matplotlib.pyplot.show()

    """
    all_events = BLG500 + BLG501 + BLG504 + BLG505 + BLG506 + BLG511 + BLG512 + BLG534 + BLG611
    plt.step(10**log_tE, all_events, where='mid')
    plt.errorbar(10**log_tE, all_events, yerr = np.sqrt(all_events), fmt = 'none', capsize = 5)
    plt.xscale('log')
    plt.yscale('log')
    plt.show()
    return

# FIXME : figure out a way to normalize the OGLE plot. 

def plot_mroz2017_de():
    """
    Plot results from Mroz 2017, with detection efficiency correction.

    Returns
    -------
    NOTE: you can plot things over
    other things. Hence, to see the plot, you need to say
    matplotlib.pyplot.show()

    """
    all_events_de = (BLG500/BLG500de + BLG501/BLG501de + BLG504/BLG504de + BLG505/BLG505de + BLG506/BLG506de + BLG511/BLG511de + BLG512/BLG512de + BLG534/BLG534de + BLG611/BLG611de) * 0.1
    plt.step(10**log_tE, all_events_de, where='mid')
    plt.errorbar(10**log_tE, all_events_de, yerr = np.sqrt(all_events_de), fmt = 'none', capsize = 5, color = 'black')
    plt.xscale('log')
    plt.yscale('log')
    plt.show()

    return


def make_BHMF(output_root):
    bh_mass_array = np.zeros(10**5)
    counter = 0

    hf = h5py.File(output_root + '.h5', 'r')
    l_array = hf['long_bin_edges']
    b_array = hf['lat_bin_edges']

    for ll in range(len(l_array)-1):
        print(ll)
        for bb in range(len(b_array)-1):
            dset_name = 'l' + str(ll) + 'b' + str(bb) + output_root
            dataset = hf[dset_name]

            BH_id = np.where(dataset[1] == 3)[0]
            nbh = len(BH_id)
            bh_mass_array[counter:counter+nbh] = dataset[2][BH_id]

            counter += nbh

    hf.close()

    np.savetxt(output_root + 'BHMF.txt', bh_mass_array)

    return

def best_tE(mag, f_blend, u0):
    t = Table.read('combined_patch.fits')

    tE_min = np.linspace(1, 300, 299)
    frac_array = np.zeros(len(tE_min))

    for i in np.arange(len(tE_min)):
        bh_id = np.where((t['rem_id_L'] == 103) &
                         (t['ubv_i_app_S'] != -99) &
                         (t['ubv_i_app_LSN'] < mag) &
                         (t['t_E'] < 300) &
                         (t['t_E'] > tE_min[i]) &
                         (t['u0'] < u0) &
                         (t['f_blend_i'] > f_blend))[0]

        all_id = np.where((t['ubv_i_app_S'] != -99) &
                          (t['ubv_i_app_LSN'] < mag) &
                          (t['t_E'] < 300) &
                          (t['t_E'] > tE_min[i]) &
                          (t['u0'] < u0) &
                          (t['f_blend_i'] > f_blend))[0]
        if len(all_id) > 0:
            frac_array[i] = 1.0 * len(bh_id)/len(all_id)
    
    plt.plot(tE_min, frac_array)
    plt.xlabel('lower limit on tE')
    plt.ylabel('fraction of black hole events')
    plt.show()

    return

def plot_bh_mass_function():
    output_root = 'OGLE_8_8_18'
    
    bh_mass_array = np.zeros(190747)
    counter = 0
    
    hf = h5py.File(output_root + '.h5', 'r')
    l_array = hf['long_bin_edges']
    b_array = hf['lat_bin_edges']
    
    for ll in range(len(l_array)-1):
        for bb in range(len(b_array)-1):
            print(ll, bb)
            dset_name = 'l' + str(ll) + 'b' + str(bb) + output_root
            dataset = hf[dset_name]
            
            BH_id = np.where(dataset[1] == 3)[0]
            nbh = len(BH_id)
            bh_mass_array[counter:counter+nbh] = dataset[2][BH_id]
            
            counter += nbh
            
    hf.close()

    plt.hist(bh_mass_array)
    plt.show()

    return bh_mass_array

def plot_how_many_we_need_submit():
    plot_how_many_we_need(120, 22, 0.1, 2)
    plot_how_many_we_need(120, 30, 0.1, 2)
    plot_how_many_we_need(120, 30, 0.1, 2)

def plot_how_many_we_need(tE_min, mag, f_blend, u0):
    # All the different patches we want to work with.
    dir_name = '/u/casey/scratch/work/microlens/galaxia_test/'

    patches = ['OGLE_10_30_18_refined_events.fits', 
               'OB150211_refined_events.fits',
               'OB150029_refined_events.fits',
               'OB140613i_refined_events.fits']

    ##########
    # Making the uncertainty plot.
    ##########
    N_detect = np.linspace(0, 10, 50)
    N_sigma = np.sqrt(N_detect)

    plt.close(1)
    fig = plt.figure(1, figsize = (6,4))
    ax1 = fig.add_subplot(111)
    ax2 = ax1.twiny()
    
    ax1.plot(N_detect, N_detect**-0.5)
    ax1.set_xlabel('$N_{BH}$ events observed per year', fontsize = 16)
    ax1.set_xticks(np.arange(N_detect[0], N_detect[-1] + 1, 5))
    
    new_tick_locations = np.linspace(0, 10, 5)
    
    # Start the counter for number of events and BH events, increment per loop.
    n_bh = 0
    n_events = 0

    for i in np.arange(len(patches)):
        t = Table.read(dir_name + patches[i])

        bh_id_tE_cut = np.where((t['rem_id_L'] == 103) &
                                (t['ubv_i_app_S'] != -99) &
                                (t['ubv_i_app_LSN'] < mag) &
                                (t['t_E'] > tE_min) &
                                (t['u0'] < u0) &
                                (t['f_blend_i'] > f_blend))[0]
        
        all_id_tE_cut = np.where((t['ubv_i_app_S'] != -99) &
                                 (t['ubv_i_app_LSN'] < mag) &
                                 (t['t_E'] > tE_min) &
                                 (t['u0'] < u0) &
                                 (t['f_blend_i'] > f_blend))[0]
        
        n_bh += len(bh_id_tE_cut)
        n_events += len(all_id_tE_cut)
        
    conversion = 1.0 * n_events/n_bh

    def tick_function(old):
        new = old * conversion
        return ["%.0f" % z for z in new]

    ax2.set_xlim(ax1.get_xlim())
    ax2.set_xticks(new_tick_locations)
    ax2.set_xticklabels(tick_function(new_tick_locations))
    ax2.set_xlabel("N events observed per year", fontsize = 16)
    ax1.set_ylabel('$\sigma_{N_{BH}}/N_{BH}$', fontsize = 16)
    plt.ylim(0, 1)
    fig.patch.set_facecolor('white')
#    fig.suptitle('tE_min: ' + str(tE_min) + ', mag: ' + str(mag) + ', f_blend: ' + str(f_blend) + ', u0: ' + str(u0))
#    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    # plt.tight_layout()
    plt.show()
    
    ##########
    # Making the tE distribution plot.
    ##########

    # Initialize array for storing tE of all events, BH, and NS.
    all_tE = None
    BH_tE = None
    NS_tE = None

    plt.figure(figsize=(6, 4))

    for i in np.arange(len(patches)):
        t = Table.read(patches[i])

        bh_id = np.where((t['rem_id_L'] == 103) &
                         (t['ubv_i_app_S'] != -99) &
                         (t['ubv_i_app_LSN'] < mag) &
                         (t['u0'] < u0) &
                         (t['f_blend_i'] > f_blend))[0]
        
        ns_id = np.where((t['rem_id_L'] == 102) &
                         (t['ubv_i_app_S'] != -99) &
                         (t['ubv_i_app_LSN'] < mag) &
                         (t['u0'] < u0) &
                         (t['f_blend_i'] > f_blend))[0]
        
        all_id = np.where((t['ubv_i_app_S'] != -99) &
                          (t['ubv_i_app_LSN'] < mag) &
                          (t['u0'] < u0) &
                          (t['f_blend_i'] > f_blend))[0]

        if all_tE is not None:
            all_tE = np.concatenate((all_tE, t['t_E'][all_id]))
        else:
            all_tE = t['t_E'][all_id]

        if BH_tE is not None:
            BH_tE = np.concatenate((BH_tE, t['t_E'][bh_id]))
        else:
            BH_tE = t['t_E'][bh_id]

        if NS_tE is not None:
            NS_tE = np.concatenate((NS_tE, t['t_E'][ns_id]))
        else:
            NS_tE = t['t_E'][ns_id]

    all_hist, all_binedges = np.histogram(all_tE, bins = np.logspace(-1, 3, 30))
    bh_hist, bh_binedges = np.histogram(BH_tE, bins = np.logspace(-1, 3, 30))
    ns_hist, ns_binedges = np.histogram(NS_tE, bins = np.logspace(-1, 3, 30))
    logtE_binedges = np.logspace(-1, 3, 30)
    logtE = (np.log10(logtE_binedges)[:-1] + np.log10(logtE_binedges)[1:])/2

    # Rescale to match OGLE distribution
    scale_factor = 2617.0/len(all_hist)
    all_hist = all_hist/scale_factor
    bh_hist = bh_hist/scale_factor
    ns_hist = ns_hist/scale_factor

    plt.step(10**logtE, all_hist, where='mid', label = 'All', color = 'black')
    plt.plot((10**logtE[-1], 10**logtE[-1]), (0, all_hist[-1]), 'k-')
    plt.step(10**logtE, bh_hist, where='mid', label = 'BH', color = 'red')
    plt.plot((10**logtE[-1], 10**logtE[-1]), (0, bh_hist[-1]), 'r-')
#    plt.step(10**logtE, ns_hist, where='mid', label = 'NS', color = 'blue')
#    plt.plot((10**logtE[-1], 10**logtE[-1]), (0, ns_hist[-1]), 'b-')
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel(r'Einstein crossing time $t_{E}$ (days)', fontsize = 16)
    plt.ylabel('Number of OGLE events', fontsize = 16)
    plt.legend()
    ax = plt.gca()
    for axis in [ax.xaxis, ax.yaxis]:
        axis.set_major_formatter(ScalarFormatter())
        ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda y,pos: ('{{:.{:1d}f}}'.format(int(np.maximum(-np.log10(y),0)))).format(y)))
    plt.xlim(5, 1000)
    plt.tight_layout()
#    plt.xlim(5, 400)
    plt.show()
    
    ##########
    # With a linear scaling
    ##########
    all_hist, all_binedges = np.histogram(all_tE, bins = np.arange(120, 501, 20))
    bh_hist, bh_binedges = np.histogram(BH_tE, bins = np.arange(120, 501, 20))
    lintE_binedges = np.arange(120, 501, 20)
#    lintE = (lintE_binedges[:-1] + lintE_binedges[1:])/2

    longidx = np.where(all_tE > 120)[0]
    print(all_tE[longidx])

    plt.figure(figsize=(6, 4))
    plt.step(lintE_binedges[:-1], all_hist, where='post', label = 'All', color = 'black')
    plt.step(lintE_binedges[:-1], bh_hist, where='post', label = 'BH', color = 'red')
    plt.xlabel(r'Einstein crossing time $t_{E}$ (days)', fontsize = 16)
    plt.ylabel('Number of events', fontsize = 16)
    plt.legend()
    ax = plt.gca()
    plt.xlim(120, 300)
    plt.ylim(0, 15)
    plt.tight_layout()
    plt.show()

#    bins = np.logspace(np.log10(0.5), np.log10(1100), 30)
#    plt.hist(all_tE, bins = bins, histtype = 'step', 
#             linewidth = 2, label = 'All Events')
#    plt.hist(BH_tE, bins = bins, histtype = 'step', 
#             linewidth = 2, label = 'BHs')
#    plt.hist(NS_tE, bins = bins, histtype = 'step', 
#             linewidth = 2, label = 'NSs')
#    plt.xlabel('$t_E$', fontsize = 16)
#    plt.xscale('log')
#    plt.yscale('log')
#    plt.xlim(100, 1100)
#    plt.legend()
#    plt.title('mag: ' + str(mag) + ', f_blend: ' + str(f_blend) + ', u0: ' + str(u0))
#    plt.show()

def plot_variations(tE_min, mag, f_blend, u0):
    ##########
    # Make the plot showing how t_E changes with more black holes
    ##########

    # All the different patches we want to work with.
    patches = ['OGLE_10_30_18_refined_events.fits', 
               'OB150211_refined_events.fits',
               'OB150029_refined_events.fits',
               'OB140613i_refined_events.fits']

    # Initialize array for storing tE.
    all_tE = None
    bh_tE = None

    for i in np.arange(len(patches)):
        t = Table.read(patches[i])

        bh_id = np.where((t['rem_id_L'] == 103) &
                         (t['ubv_i_app_S'] != -99) &
                         (t['ubv_i_app_LSN'] < mag) &
                         (t['u0'] < u0) &
                         (t['f_blend_i'] > f_blend))[0]
                
        all_id = np.where((t['ubv_i_app_S'] != -99) &
                          (t['ubv_i_app_LSN'] < mag) &
                          (t['u0'] < u0) &
                          (t['f_blend_i'] > f_blend))[0]

        if all_tE is not None:
            all_tE = np.concatenate((all_tE, t['t_E'][all_id]))
        else:
            all_tE = t['t_E'][all_id]

        if bh_tE is not None:
            bh_tE = np.concatenate((bh_tE, t['t_E'][bh_id]))
        else:
            bh_tE = t['t_E'][bh_id]

    # For the binaries
    bh_tE_binary = bh_tE.copy()
    all_tE_binary = all_tE.copy()
    bh_tE_binary[0::2] = np.sqrt(2) * bh_tE[0::2]
    for i in np.arange(len(bh_tE[0::2])):
        idx = np.where(all_tE_binary == bh_tE[0::2][i])[0]
        if len(idx) != 1:
            print(i)
        all_tE_binary[idx] = np.sqrt(2) * bh_tE[0::2][i]

    long_idx = np.where(bh_tE > 120)[0]
    print(bh_tE[long_idx])

    # For the primordial
    bh_tE_prim = np.concatenate((bh_tE, np.sqrt(5) *  bh_tE[0::10]))
    all_tE_prim = np.concatenate((all_tE, np.sqrt(5) * bh_tE[0::10]))

    tEbins = np.arange(120, 1081, 60)

    all_fid, all_edges_fid = np.histogram(all_tE, bins = tEbins)
    bh_fid, bh_edges_fid = np.histogram(bh_tE, bins = tEbins)

    print(all_fid)
    print(bh_fid)

    all_binary, all_edges_binary = np.histogram(all_tE_binary, bins = tEbins)
    bh_binary, bh_edges_binary = np.histogram(bh_tE_binary, bins = tEbins)

    all_prim, all_edges_prim = np.histogram(all_tE_prim, bins = tEbins)
    bh_prim, bh_edges_prim = np.histogram(bh_tE_prim, bins = tEbins)

    plt.figure(figsize=(6, 4))

    bad_idx_prim = np.where(all_prim == 0)[0]
    all_prim[bad_idx_prim] = 1
    bh_prim[bad_idx_prim] = 0
    frac_prim = (bh_prim * 1.0)/all_prim

    bad_idx_binary = np.where(all_binary == 0)[0]
    all_binary[bad_idx_binary] = 1
    bh_binary[bad_idx_binary] = 0
    frac_binary = (bh_binary * 1.0)/all_binary

    bad_idx_fid = np.where(all_fid == 0)[0]
    all_fid[bad_idx_fid] = 1
    bh_fid[bad_idx_binary] = 0
    frac_fid = (bh_fid * 1.0)/all_fid

    plt.step(tEbins[:-1], frac_prim, where='post', 
             label = 'Primordial BHs', color = '#2ca02c')
    plt.step(tEbins[:-1], frac_binary, where='post', 
             label = '50% Merged Binaries', color = '#ff7f0e', linestyle = '--')
    plt.step(tEbins[:-1], frac_fid, where='post', 
             label = 'Fiducial', color = '#1f77b4', linestyle = ':')

    plt.xlabel(r'Einstein crossing time $t_{E}$ (days) ')
    plt.ylabel('Fraction of BH events')
    plt.xlim(120, 500)
    plt.legend()
    plt.show()
    return

##########

#    bins = np.logspace(np.log10(0.5), np.log10(1100), 50)
#    plt.hist(all_tE_mod2, bins = bins, histtype = 'step', 
#             linewidth = 2, label = 'Primordial BHs')
#
#    plt.hist(all_tE_mod1, bins = bins, histtype = 'step', 
#             linewidth = 2, label = '10% Merged binaries')
#
#    plt.hist(all_tE, bins = bins, histtype = 'step', 
#             linewidth = 2, label = 'Fiducial')
#    plt.xscale('log')
#    plt.yscale('log')
#    plt.xlim(100, 1200)
#    plt.ylim(10**-1, 2 * 10**3)
#    plt.legend()
#    plt.ylabel('Number of OGLE events')
#    plt.xlabel('Einstein crossing time $t_E$ (days)')
#    plt.title('mag: ' + str(mag) + ', f_blend: ' + str(f_blend) + ', u0: ' + str(u0))
#    plt.suptitle('All Events')
#    plt.show()
#
#    plt.subplot(1, 2, 1)
#    bins = np.logspace(np.log10(0.5), np.log10(np.max(t['t_E'][all_id])), 30)
#    plt.hist(t['t_E'][all_id_full], bins = bins, histtype = 'step', 
#             linewidth = 2, label = 'Fiducial')
#    plt.hist(all_tE_mod1, bins = bins, histtype = 'step', 
#             linewidth = 2, label = 'Primordial BHs')
#    plt.hist(all_tE_mod2, bins = bins, histtype = 'step', 
#             linewidth = 2, label = '10% Merged binaries')
#    plt.xlabel('$t_E$', fontsize = 16)
#    plt.xscale('log')
#    plt.yscale('log')
#    plt.xlim(10, 300)
#    plt.title('All events')
#    plt.legend()
#
#    plt.subplot(1, 2, 2)
#    plt.hist(t['t_E'][bh_id_full], bins = bins, histtype = 'step', 
#             linewidth = 2, label = 'Fiducial')
#    plt.hist(bh_tE_mod1, bins = bins, histtype = 'step', 
#             linewidth = 2, label = 'Primordial BHs')
#    plt.hist(bh_tE_mod2, bins = bins, histtype = 'step', 
#             linewidth = 2, label = '10% Merged binaries')
#    plt.xlabel('$t_E$', fontsize = 16)
#    plt.xscale('log')
#    plt.yscale('log')
#    plt.xlim(10, 300)
#    plt.title('BH events')
#    plt.legend()
#    plt.show()
#
# Constraints on the conversion: 120 days < t_E < 500 days, no magnitude cut, u0 < 2

def plot_OGLE_v_WFIRST_v2():
    ogle_patches = ['OGLE_10_30_18_refined_events.fits', 
                    'OB150211_refined_events.fits',
                    'OB150029_refined_events.fits',
                    'OB140613i_refined_events.fits']

    wfirst_patches = ['OB170095_diff_limit_refined_events.fits', 
                      'OB150211_diff_limit_refined_events.fits',
                      'OB150029_diff_limit_refined_events.fits',
                      'OB140613_diff_limit_refined_events.fits']
    
    # Start the counter for number of events and BH events, increment per loop.
    all_ogle = None
    BH_ogle = None
    all_wfirst = None
    BH_wfirst = None

    for i in np.arange(len(ogle_patches)):
        t = Table.read(ogle_patches[i])

        bh_id = np.where((t['rem_id_L'] == 103) &
                         (t['ubv_i_app_S'] != -99) &
                         (t['ubv_i_app_LSN'] < 22) &
                         (t['u0'] < 1) &
                         (t['f_blend_i'] > 0.1))[0]
        
        all_id = np.where((t['ubv_i_app_S'] != -99) &
                          (t['ubv_i_app_LSN'] < 22) &
                          (t['u0'] < 1) &
                          (t['f_blend_i'] > 0.1))[0]
        
        if all_ogle is not None:
            all_ogle = np.concatenate((all_ogle, t['t_E'][all_id]))
        else:
            all_ogle = t['t_E'][all_id]

        if BH_ogle is not None:
            BH_ogle = np.concatenate((BH_ogle, t['t_E'][bh_id]))
        else:
            BH_ogle = t['t_E'][bh_id]


    for i in np.arange(len(wfirst_patches)):
        t = Table.read(wfirst_patches[i])

        bh_id = np.where((t['rem_id_L'] == 103) &
                         (t['ubv_h_app_S'] != -99) &
                         (t['ubv_h_app_LSN'] < 26) &
                         (t['u0'] < 1) &
                         (t['f_blend_h'] > 0.1))[0]
        
        all_id = np.where((t['ubv_h_app_S'] != -99) &
                          (t['ubv_h_app_LSN'] < 26) &
                          (t['u0'] < 1) &
                          (t['f_blend_h'] > 0.1))[0]
        
        if all_wfirst is not None:
            all_wfirst = np.concatenate((all_wfirst, t['t_E'][all_id]))
        else:
            all_wfirst = t['t_E'][all_id]

        if BH_wfirst is not None:
            BH_wfirst = np.concatenate((BH_wfirst, t['t_E'][bh_id]))
        else:
            BH_wfirst = t['t_E'][bh_id]

    print(np.max(all_ogle))

    plt.hist(all_ogle, bins = np.logspace(np.log10(0.5), np.log10(500), 20),
             histtype = 'step', label = 'All OGLE', color = 'black', linestyle = '--', linewidth = 2)
    plt.hist(all_wfirst,  bins = np.logspace(np.log10(0.5), np.log10(500), 20), 
             histtype = 'step', label = 'All WFIRST', color = 'red', linestyle = '--', linewidth = 2)
    plt.hist(BH_ogle, bins = np.logspace(np.log10(0.5), np.log10(500), 20), 
             histtype = 'step', label = 'BH OGLE', color = 'black', linestyle = '-', linewidth = 2)
    plt.hist(BH_wfirst, bins = np.logspace(np.log10(0.5), np.log10(500), 20), 
             histtype = 'step', label = 'BH WFIRST', color = 'red', linestyle = '-', linewidth = 2)
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('$t_E$ (days)')
    plt.ylabel('Number of events')
    plt.legend()
    plt.show()
    
    return


def plot_OGLE_v_WFIRST():
    ogle = Table.read('aggregate_3events_w_mroz_cuts.fits')
    wfirst = Table.read('aggregate_3events_w_wfirst_cuts.fits')

    ogle_bh = np.where(ogle['rem_id_L'] == 103)[0]
    wfirst_bh = np.where(wfirst['rem_id_L'] == 103)[0]

    print(len(ogle['t_E']))
    print(len(wfirst['t_E']))
    print(len(ogle_bh))
    print(len(wfirst_bh))

    plt.hist(ogle['t_E'], bins = np.logspace(np.log10(0.5), np.log10(500), 20),
             histtype = 'step', label = 'All OGLE', color = 'black', linestyle = '--', linewidth = 2)
    plt.hist(wfirst['t_E'],  bins = np.logspace(np.log10(0.5), np.log10(500), 20), 
             histtype = 'step', label = 'All WFIRST', color = 'red', linestyle = '--', linewidth = 2)
    plt.hist(ogle['t_E'][ogle_bh], bins = np.logspace(np.log10(0.5), np.log10(500), 20), 
             histtype = 'step', label = 'BH OGLE', color = 'black', linestyle = '-', linewidth = 2)
    plt.hist(wfirst['t_E'][wfirst_bh], bins = np.logspace(np.log10(0.5), np.log10(500), 20), 
             histtype = 'step', label = 'BH WFIRST', color = 'red', linestyle = '-', linewidth = 2)
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('$t_E$ (days)')
    plt.ylabel('Number of events')
    plt.legend()
    plt.show()
    
    return

def plot_mroz_rate():
    rate_mroz = 100 * np.array([1.28, 1.86, 0.949, 1.42, 0.963, 0.765, 0.759, 1.04, 0.653])
    glon_mroz = np.array([0.999, -0.0608, 2.1491, 1.0870, 0.0103, 3.2835, 2.2154, -1.1356, 0.3282])
    glat_mroz = np.array([-1.0293, -1.64, -1.7747, -2.3890, -2.9974, -2.5219, -3.1355, -2.2547, 2.2842])

    rate = 100 * np.array([0.346, 0.419, 0.238, 0.630])
    glon = np.array([357.25 - 360, 1.83, 356.39 - 360, 1.252])
    glat = np.array([-3.32, -2.52, 1.86, -1.38])

    plt.scatter(glon_mroz, glat_mroz, s=rate_mroz, color='red', label = 'Mroz')
    plt.scatter(glon, glat, s=rate, color='blue', label = 'PopSyCLE')
    plt.xlabel('longitude')
    plt.ylabel('latitude')
    plt.legend()
    plt.title('Radius $\propto$ event rate')
    plt.show()

def aggregate_patch():
    agg_patch = None
    patches = ['OGLE_10_30_18_refined_events.fits', 
               'OB150211_refined_events.fits',
               'OB150029_refined_events.fits',
               'OB140613i_refined_events.fits']
    for patch in patches:
        tt = Table.read(patch)
        tt.meta['H5_INPUT'] = None
        tt.meta['EVENT'] = None
        tt.meta['EVENT_IN'] = None
        tt.meta['BLEND_IN'] = None
        tt.meta['REFINED'] = None
        all_id = np.where((tt['ubv_i_app_S'] != -99) &
                          (tt['ubv_i_app_LSN'] < 30) &
#                          (tt['t_E'] < 360) &
                          (tt['t_E'] > 0.5) &
                          (tt['u0'] < 2) &
                          (tt['f_blend_i'] > 0.1))[0]

        if agg_patch is not None:
            agg_patch = vstack((agg_patch, tt[all_id]))
        else:
            agg_patch = tt[all_id]

    agg_patch.write('aggregate_events_u0_2.fits')

#    if agg_patch is not None:
#        agg_patch = vstack((agg_patch, tt))
#    else:
#        agg_patch = tt
#
#    agg_patch.write('combined_patch.fits')
    
    return

def aggregate_patch_wfirst():
    agg_patch = None
    patches = ['OB170095_diff_limit_refined_events.fits', 
               'OB150211_diff_limit_refined_events.fits',
               'OB150029_diff_limit_refined_events.fits']
#               'OB140613_refined_events.fits']
    for patch in patches:
        tt = Table.read(patch)
        tt.meta['H5_INPUT'] = None
        tt.meta['EVENT'] = None
        tt.meta['EVENT_IN'] = None
        tt.meta['BLEND_IN'] = None
        tt.meta['REFINED'] = None
        all_id = np.where((tt['ubv_h_app_S'] != -99) &
                          (tt['ubv_h_app_LSN'] < 26) &
                          (tt['t_E'] > 0.5) &
                          (tt['u0'] < 1) &
                          (tt['f_blend_h'] > 0.1))[0]

        if agg_patch is not None:
            agg_patch = vstack((agg_patch, tt[all_id]))
        else:
            agg_patch = tt[all_id]

    agg_patch.write('aggregate_3events_w_wfirst_cuts.fits')
    
    return

def neighbors(refined_events, blends, filter):
    """
    bloop
    """

    e = Table.read(refined_events)
    b = Table.read(blends)

    bh_id = np.where((e['rem_id_L'] == 103) &
                     (e['ubv_' + filter + '_app_S'] != -99) &
                     (e['ubv_' + filter + '_app_S'] < 26))[0]

    other_id = np.where((e['rem_id_L'] != 103) &
                        (e['ubv_' + filter + '_app_S'] != -99) &
                        (e['ubv_' + filter + '_app_S'] < 26))[0]


    plt.hist(e['px_L'][other_id]**-1 - e['px_S'][other_id]**-1, bins = np.linspace(0, 6), histtype = 'step')
    plt.hist(e['px_L'][bh_id]**-1 - e['px_S'][bh_id]**-1, bins = np.linspace(0, 6), histtype = 'step')
    plt.yscale('log')
    plt.show()

    sorc_id = e['obj_id_S'][bh_id]
    lens_id = e['obj_id_L'][bh_id]

    s = sorc_id[0]
    l = lens_id[0]

    nebr_id = np.where((b['obj_id_S'] == s) & (b['obj_id_L'] == l))[0]

#    plt.plot(e['px_S'][bh_id], e['py_S'][bh_id],'.',  ms = 5, color = 'red')
#    plt.plot(e['px_L'][bh_id], e['py_L'][bh_id], '.', ms = 10, color = 'black')
#    plt.plot(e['px_L'][other_id], e['py_L'][other_id], '.', ms = 5, color = 'red')
#    plt.show()

#    plt.plot(b['px_N'][nebr_id], b['py_N'][nebr_id], '.', color = 'yellow')
#    plt.plot(e['px_S'][bh_id[0]], e['py_S'][bh_id[0]],'.',  ms = 5, color = 'red')
#    plt.plot(e['px_L'][bh_id[0]], e['py_L'][bh_id[0]], '.', ms = 5, color = 'black')
#    plt.show()

#    print(e['px_S'][bh_id[0]])
#    print(e['py_S'][bh_id[0]])

    return

def rates(ebf_file, filter_name, red_law, refined_events, survey_duration, title):
    mag_vec = np.arange(18, 28.1, 1) 
    rate_vec = np.zeros(len(mag_vec))
    bhrate_vec = np.zeros(len(mag_vec))
    for i in np.arange(len(mag_vec)):
        rate_vec[i], bhrate_vec[i] = synthetic.calc_event_rate(ebf_file, filter_name, red_law, refined_events, mag_vec[i], survey_duration)

    plt.plot(mag_vec, rate_vec, label = 'All')
    plt.plot(mag_vec, bhrate_vec, label = 'BH')
    plt.xlabel('Magnitude cut')
    plt.ylabel('Rate (events/star/year)')
    plt.yscale('log')
    plt.title(title)
    plt.legend()
    plt.show()

    # Also to add to this: BH event rates
    # How the number of stars and events changes (separately)

    return

def BH_frac(fitsfile, title, filter = 'i', mag = 22, t_E_max = 300, u0 = 1, f_blend = 0.1):
    """
    Makes a plot of the fraction of black holes vs all other types of events.

    Parameters
    ----------
    fitsfile : the file that comes out of refine_events
    filter : 'i', 'h', etc.
    mag : float
    t_E_max : upper limit for t_E, in days
    u0 : upper limit on impact parameter
    f_blend: lower limit on blend fraction
    """
    t = Table.read(fitsfile)

    t_E_min_array = np.arange(20, t_E_max + 1, 20)

    bh_num = np.zeros(len(t_E_min_array))
    all_num = np.zeros(len(t_E_min_array))

    number = np.where((t['ubv_' + filter + '_app_S'] != -99) &
                      (t['ubv_' + filter + '_app_LSN'] < mag) &
                      (t['t_E'] < 1000) &
                      (t['t_E'] > 0.5) &
                      (t['u0'] < u0) &
                      (t['f_blend_' + filter] > f_blend))[0]
    
    print(len(number))

    for i in np.arange(len(t_E_min_array)):
        t_E_min = t_E_min_array[i]

        bh_id = np.where((t['rem_id_L'] == 103) &
                         (t['ubv_' + filter + '_app_S'] != -99) &
                         (t['ubv_' + filter + '_app_LSN'] < mag) &
                         (t['t_E'] < t_E_max) &
                         (t['t_E'] > t_E_min) &
                         (t['u0'] < u0) &
                         (t['f_blend_' + filter] > f_blend))[0]

        all_id = np.where((t['ubv_' + filter + '_app_S'] != -99) &
                          (t['ubv_' + filter + '_app_LSN'] < mag) &
                          (t['t_E'] < t_E_max) &
                          (t['t_E'] > t_E_min) &
                          (t['u0'] < u0) &
                          (t['f_blend_' + filter] > f_blend))[0]

        bh_num[i] = len(bh_id)
        all_num[i] = len(all_id)

    good_id = np.where(all_num != 0)[0]
    t_E_min_array = t_E_min_array[good_id]
    bh_num = bh_num[good_id]
    all_num = all_num[good_id]

    bh_id_hist = np.where((t['rem_id_L'] == 103) &
                          (t['ubv_' + filter + '_app_S'] != -99) &
                          (t['ubv_' + filter + '_app_LSN'] < mag) &
                          (t['t_E'] < 300) &
                          (t['t_E'] > 0.5) &
                          (t['u0'] < u0) &
                          (t['f_blend_' + filter] > f_blend))[0]

    all_id_hist = np.where((t['ubv_' + filter + '_app_S'] != -99) &
                           (t['ubv_' + filter + '_app_LSN'] < mag) &
                           (t['t_E'] < 300) &
                           (t['t_E'] > 0.5) &
                           (t['u0'] < u0) &
                           (t['f_blend_' + filter] > f_blend))[0]

    plt.figure(1, figsize=(10, 5))

    plt.subplot(1, 2, 1)
    plt.hist(t['t_E'][all_id_hist], bins = np.logspace(np.log10(0.5), np.log10(300), 25), 
             histtype = 'step', linewidth = 2, label = 'All')
    plt.hist(t['t_E'][bh_id_hist], bins = np.logspace(np.log10(0.5), np.log10(300), 25), 
             histtype = 'step', linewidth = 2, label = 'BH')
    plt.legend()
    plt.xlabel('$t_E$ (days)')
    plt.ylabel('Number of events')
    plt.xscale('log')
    plt.yscale('log')
    plt.title(title)
    
    plt.subplot(1, 2, 2)
    plt.bar(t_E_min_array, (1.0 * bh_num)/all_num, width = 10)
    plt.xlabel('Minimum $t_E$ (days)')
    plt.ylabel('Fraction of BH events')
    plt.title(title)
    plt.show()

def scatter_pairs(fitsfile, filter, variable1, variable2, mag = 22, logx = 'no'):
    """ 
    Makes a scatter plot comparing variable 1 and 2, and individual histograms
    for each. For each of these, one has a magnitude cut and the other doesn't.

    Parameters
    ----------
    fitsfile : the file that comes out of refine_events
    filter : 'i', 'h', etc.
    variable 1 and 2 : 'f_blend_i', 't_E', 'ubv_h_app_N', etc.
    mag : float
    logx : 'yes' or 'no', if you want the x-scale to be logarithmic
    """
    t = Table.read(fitsfile)
    x = t[variable1]
    y = t[variable2]
    
    bh_id = np.where((t['rem_id_L'] == 103) &
                     (t['ubv_' + filter + '_app_S'] != -99))[0]
    other_id = np.where((t['rem_id_L'] != 103) &
                        (t['ubv_' + filter + '_app_S'] != -99))[0]

    bh_magcut_id = np.where((t['rem_id_L'] == 103) & 
                            (t['ubv_' + filter + '_app_S'] < mag) & 
                            (t['ubv_' + filter + '_app_S'] != -99))[0]
    other_magcut_id = np.where((t['rem_id_L'] != 103) & 
                               (t['ubv_' + filter + '_app_S'] < mag) & 
                               (t['ubv_' + filter + '_app_S'] != -99))[0]

    x_lower = np.min(np.concatenate((x[other_id], x[bh_id])))
    x_upper = np.max(np.concatenate((x[other_id], x[bh_id])))
    y_lower = np.min(np.concatenate((y[other_id], y[bh_id])))
    y_upper = np.max(np.concatenate((y[other_id], y[bh_id])))
    xbins = np.linspace(x_lower, x_upper, 20)
    ybins = np.linspace(y_lower, y_upper, 20)

    x_lower_magcut = np.min(np.concatenate((x[other_magcut_id], x[bh_magcut_id])))
    x_upper_magcut = np.max(np.concatenate((x[other_magcut_id], x[bh_magcut_id])))
    y_lower_magcut = np.min(np.concatenate((y[other_magcut_id], y[bh_magcut_id])))
    y_upper_magcut = np.max(np.concatenate((y[other_magcut_id], y[bh_magcut_id])))
    xbins_magcut = np.linspace(x_lower_magcut, x_upper_magcut, 20)
    ybins_magcut = np.linspace(y_lower_magcut, y_upper_magcut, 20)

    plt.subplot(3, 2, 1)
    plt.plot(x[other_id], y[other_id],'.', color = 'cyan',alpha = 0.8, ms = 3, label = 'Other')
    plt.plot(x[bh_id], y[bh_id], '.', color = 'orange', alpha = 0.8, ms = 3, label = 'BH')
    plt.xlabel(variable1)
    plt.ylabel(variable2)
    if logx == 'yes':
        plt.yscale('log')
    plt.title('No magnitude cut')
    plt.legend()

    plt.subplot(3, 2, 2)
    plt.plot(x[other_magcut_id], y[other_magcut_id], '.', color = 'cyan', alpha = 0.8, ms = 3, label = 'Other')
    plt.plot(x[bh_magcut_id], y[bh_magcut_id], '.', color = 'orange', alpha = 0.8, ms = 3, label = 'BH')
    plt.xlabel(variable1)
    plt.ylabel(variable2)
    if logx == 'yes':
        plt.yscale('log')
    plt.title('m_source < ' + str(mag))
    plt.legend()

    plt.subplot(3, 2, 3)
    plt.hist(x[other_id], bins = xbins, histtype = 'step', label = 'Other')
    plt.hist(x[bh_id], bins = xbins, histtype = 'step', label = 'BH')
    plt.xlabel(variable1)
    plt.ylabel('Number')
    plt.title('No magnitude cut')
    plt.yscale('log')
    plt.legend()

    plt.subplot(3, 2, 4)
    plt.hist(x[other_magcut_id], bins = xbins_magcut, histtype = 'step', label = 'Other')
    plt.hist(x[bh_magcut_id], bins = xbins_magcut, histtype = 'step', label = 'BH')
    plt.xlabel(variable1)
    plt.ylabel('Number')
    plt.yscale('log')
    plt.title('m_source < ' + str(mag))
    plt.legend()

    plt.subplot(3, 2, 5)
    plt.hist(y[other_id], bins = ybins, histtype = 'step', label = 'Other')
    plt.hist(y[bh_id], bins = ybins, histtype = 'step', label = 'BH')
    plt.xlabel(variable2)
    plt.ylabel('Number')
    plt.yscale('log')
    plt.title('No magnitude cut')
    plt.legend()

    plt.subplot(3, 2, 6)
    plt.hist(y[other_magcut_id], bins = ybins_magcut, histtype = 'step', label = 'Other')
    plt.hist(y[bh_magcut_id], bins = ybins_magcut, histtype = 'step', label = 'BH')
    plt.xlabel(variable2)
    plt.ylabel('Number')
    plt.yscale('log')
    plt.title('m_source < ' + str(mag))
    plt.legend()

    plt.tight_layout()
    plt.show()

def plot_blend():
    ogle = Table.read('aggregate_3events_w_mroz_cuts.fits')
    wfirst = Table.read('aggregate_3events_w_wfirst_cuts.fits')

    ogle_bh = np.where(ogle['rem_id_L'] == 103)[0]
    wfirst_bh = np.where(wfirst['rem_id_L'] == 103)[0]

    plt.hist(ogle['f_blend_i'], bins = np.linspace(0, 1, 20),
             histtype = 'step', label = 'All OGLE', color = 'black', linewidth = 2)
    plt.hist(ogle['f_blend_i'][ogle_bh], bins = np.linspace(0, 1, 20), 
             histtype = 'step', label = 'BH OGLE', color = 'red', linewidth = 2)
    plt.yscale('log')
    plt.xlabel('$f_{blend,i}$xc')
    plt.ylabel('Number of events')
    plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
               ncol=2, mode="expand", borderaxespad=0.)
    plt.show()

    plt.hist(wfirst['f_blend_h'],  bins = np.linspace(0, 1, 20), 
             histtype = 'step', label = 'All WFIRST', color = 'black', linewidth = 2)
    plt.hist(wfirst['f_blend_h'][wfirst_bh], bins = np.linspace(0, 1, 20), 
             histtype = 'step', label = 'BH WFIRST', color = 'red', linewidth = 2)
    plt.yscale('log')
    plt.xlabel('$f_{blend,h}$')
    plt.ylabel('Number of events')
    plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
               ncol=2, mode="expand", borderaxespad=0.)
    plt.show()
    
    return    

def A_v_delta(fitsfile):
    """ 
    Makes a scatter plot comparing variable 1 and 2, and individual histograms
    for each. For each of these, one has a magnitude cut and the other doesn't.

    Parameters
    ----------
    fitsfile : the file that comes out of refine_events
    filter : 'i', 'h', etc.
    variable 1 and 2 : 'f_blend_i', 't_E', 'ubv_h_app_N', etc.
    mag : float
    logx : 'yes' or 'no', if you want the x-scale to be logarithmic
    """
    t = Table.read(fitsfile)
    u0 = t['u0']
    thetaE = t['theta_E']
    filter = 'h'
    mag = 26
    
    bh_id = np.where((t['rem_id_L'] == 103) &
                     (t['ubv_' + filter + '_app_S'] != -99))[0]
    other_id = np.where((t['rem_id_L'] != 103) &
                        (t['ubv_' + filter + '_app_S'] != -99))[0]

    bh_magcut_id = np.where((t['rem_id_L'] == 103) & 
                            (t['ubv_' + filter + '_app_S'] < mag) & 
                            (t['ubv_' + filter + '_app_S'] != -99))[0]
    other_magcut_id = np.where((t['rem_id_L'] != 103) & 
                               (t['ubv_' + filter + '_app_S'] < mag) & 
                               (t['ubv_' + filter + '_app_S'] != -99))[0]

    delta_c = (u0 * thetaE)/(u0**2 + 2)
    A = (u0**2 + 2)/(u0 * np.sqrt(u0**2 + 4))

    plt.subplot(1, 3, 1)
    plt.plot(A[other_id], delta_c[other_id], '.', label = 'Other')
    plt.plot(A[bh_id], delta_c[bh_id], '.', label = 'BH')
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('A')
    plt.ylabel('delta_c')
    plt.legend()

    plt.subplot(1, 3, 2)
    plt.hist(A[other_id], bins = np.linspace(0, 6000, 20), histtype = 'step', label = 'Other')
    plt.hist(A[bh_id], bins = np.linspace(0, 6000, 20), histtype = 'step', label = 'BH')
    plt.xlabel('A')
    plt.yscale('log')
    plt.legend()

    plt.subplot(1, 3, 3)
    plt.hist(delta_c[other_id], bins = np.linspace(0, 2.2, 20), histtype = 'step', label = 'Other')
    plt.hist(delta_c[bh_id], bins = np.linspace(0, 2.2, 20), histtype = 'step', label = 'BH')
    plt.xlabel('delta_c')
    plt.yscale('log')
    plt.legend()

    plt.show()

    return

