import numpy as np
import pylab as py
from astropy.table import Table
from astropy.table import Column
import pdb

work_dir = '/u/jlu/work/ao/roboAO/2016_01_18/'

# Central wavelength in Angstroms.
filt_wave = {'Z': 8818.1, 'Y': 10368.1, 'J': 12369.9, 'H': 16464.4}

def plot_seeing_vs_ao(tint=1200, spec_res=100):
    ap_diam_see = 0.8
    ap_diam_rao = 0.3
    
    in_file_root = '{0:s}/roboAO_sensitivity_t{1:d}_R{2:d}'.format(work_dir, tint, spec_res)

    in_file_see = '{0:s}_ap{1:0.3f}_seeing'.format(in_file_root, ap_diam_see)
    in_file_rao = '{0:s}_ap{1:0.3f}'.format(in_file_root, ap_diam_rao)

    avg_tab_rao = Table.read(in_file_rao + '_avg_tab.fits')
    avg_tab_see = Table.read(in_file_see + '_avg_tab.fits')
    
    # Calculate the band-integrated SNR for each magnitude bin and filter.
    mag_rao = avg_tab_rao['mag']
    mag_see = avg_tab_see['mag']

    hdr1 = '# {0:3s}  {1:15s}   {2:15s}   {3:15s}   {4:15s}'
    hdr2 = '# {0:3s}  {1:7s} {2:7s}   {3:7s} {4:7s}   {5:7s} {6:7s}   {7:7s} {8:7s}'
    print hdr1.format('Mag', 'Z SNR (summed)', 'Y SNR (summed)', 'J SNR (summed)', 'H SNR (summed)')
    print hdr2.format('', 'Robo-AO', 'Seeing', 'Robo-AO', 'Seeing', 'Robo-AO', 'Seeing', 'Robo-AO', 'Seeing')
    print hdr2.format('---', '-------', '-------', '-------', '-------', '-------', '-------', '-------', '-------')

    for mm in range(len(mag_rao)):
        fmt = '  {0:3d}  {1:7.1f} {2:7.1f}   {3:7.1f} {4:7.1f}   {5:7.1f} {6:7.1f}   {7:7.1f} {8:7.1f}'
        print fmt.format(mag_rao[mm], 
                         avg_tab_rao['snr_sum_z'][mm], avg_tab_see['snr_sum_z'][mm],
                         avg_tab_rao['snr_sum_y'][mm], avg_tab_see['snr_sum_y'][mm],
                         avg_tab_rao['snr_sum_j'][mm], avg_tab_see['snr_sum_j'][mm],
                         avg_tab_rao['snr_sum_h'][mm], avg_tab_see['snr_sum_h'][mm])

    py.figure(1)
    py.clf()
    py.semilogy(avg_tab_rao['mag'], avg_tab_rao['snr_sum_j'], label='UH Robo-AO', 
                linewidth=2)
    py.semilogy(avg_tab_rao['mag'], avg_tab_see['snr_sum_j'], label='Seeing-Limited',
                linewidth=2)
    py.xlabel('J-band Magnitude')
    py.ylabel('Signal-to-Noise in 1200 sec')
    py.xlim(16, 21)
    py.ylim(1, 500)
    py.arrow(16.1, 30, 4.05, 0, color='red', width=2, head_width=10, head_length=0.5, 
             length_includes_head=True, fc='red', ec='red')
    py.text(17.3, 11, '250x More\nSupernovae Ia', color='red', 
            horizontalalignment='left', fontweight='bold', fontsize=16)
    py.legend()
    py.savefig('{0:s}_prop_j_snr.png'.format(in_file_root))


    # Calculate the slope uncertainties
    # y = mx + b
    # X = [[ b ]
    #      [ m ]]
    # A = [[ 1  x_1 ]
    #      [ 1  x_2 ]
    #      [ 1  x_3 ]
    #      [ 1  x_4 ]]
    # C = [[ ye_1^2    0       0       0   ]
    #      [   0     ye_2^2    0       0   ]
    #      [   0       0     ye_3^2    0   ]
    #      [   0       0       0     ye_4^2]]
    #
    # Covariance matrix on fit parameters:
    # cov_mat = [A^T C A]^-1
    #
    # m_wave = m_wave,0 + A_wave
    # A_wave = A_Ks * (wave / wave_Ks)**alpha
    # alpha = -1.95 (Wang & Jiang 2014)
    #
    # for line fit:
    # y = m_wave
    # x = (wave / wave_Ks)**alpha
    # m = A_Ks
    # b = m_wave,0
    # 

    AKs_err_rao = np.zeros(len(mag_rao), dtype=float)
    AKs_err_see = np.zeros(len(mag_rao), dtype=float)
    AKs_snr_rao = np.zeros(len(mag_rao), dtype=float)
    AKs_snr_see = np.zeros(len(mag_rao), dtype=float)

    wave_Ks = 21500.0
    alpha = -1.95
    
    A = np.matrix([[ 1.0, (filt_wave['Z'] / wave_Ks)**alpha],
                   [ 1.0, (filt_wave['Y'] / wave_Ks)**alpha],
                   [ 1.0, (filt_wave['J'] / wave_Ks)**alpha],
                   [ 1.0, (filt_wave['H'] / wave_Ks)**alpha]])
    
    for mm in range(len(mag_rao)):
        C_rao = np.matrix([[(1.0 / avg_tab_rao['snr_sum_z'][mm])**2, 0, 0, 0],
                           [0, (1.0 / avg_tab_rao['snr_sum_y'][mm])**2, 0, 0],
                           [0, 0, (1.0 / avg_tab_rao['snr_sum_j'][mm])**2, 0],
                           [0, 0, 0, (1.0 / avg_tab_rao['snr_sum_h'][mm])**2]])

        C_see = np.matrix([[(1.0 / avg_tab_see['snr_sum_z'][mm])**2, 0, 0, 0],
                           [0, (1.0 / avg_tab_see['snr_sum_y'][mm])**2, 0, 0],
                           [0, 0, (1.0 / avg_tab_see['snr_sum_j'][mm])**2, 0],
                           [0, 0, 0, (1.0 / avg_tab_see['snr_sum_h'][mm])**2]])

        COV_rao = (A.T * (C_rao.I * A)).I
        COV_see = (A.T * (C_see.I * A)).I

        AKs_err_rao[mm] = COV_rao[1,1]**0.5
        AKs_err_see[mm] = COV_see[1,1]**0.5
        AKs_snr_rao[mm] = 1.0 / AKs_err_rao[mm]
        AKs_snr_see[mm] = 1.0 / AKs_err_see[mm]

        
    py.clf()
    py.plot(avg_tab_rao['mag'], AKs_err_rao, label='UH Robo-AO', linewidth=2)
    py.plot(avg_tab_rao['mag'], AKs_err_see, label='Seeing-Limited', linewidth=2)
    py.xlabel('Vega Magnitude')
    py.ylabel(r'Error on A$_{Ks}$')
    py.xlim(16, 21)
    py.ylim(0, 0.05)
    # py.arrow(16.1, 30, 4.05, 0, color='red', width=2, head_width=10, head_length=0.5, 
    #          length_includes_head=True, fc='red', ec='red')
    # py.text(17.3, 11, '250x More\nSupernovae Ia', color='red', 
    #         horizontalalignment='left', fontweight='bold', fontsize=16)
    py.legend()
    py.savefig('{0:s}_prop_AKs_err.png'.format(in_file_root))

    py.clf()
    py.semilogy(avg_tab_rao['mag'], AKs_snr_rao, label='UH Robo-AO', linewidth=2)
    py.plot(avg_tab_rao['mag'], AKs_snr_see, label='Seeing-Limited', linewidth=2)
    py.xlabel('Vega Magnitude')
    py.ylabel(r'SNR on A$_{Ks}$ Measurement')
    py.xlim(16, 21)
    py.ylim(0, 2000)
    # py.arrow(16.1, 30, 4.05, 0, color='red', width=2, head_width=10, head_length=0.5, 
    #          length_includes_head=True, fc='red', ec='red')
    # py.text(17.3, 11, '250x More\nSupernovae Ia', color='red', 
    #         horizontalalignment='left', fontweight='bold', fontsize=16)
    py.legend()
    py.savefig('{0:s}_prop_AKs_snr.png'.format(in_file_root))
    

        
    return

def plot_sensitivity_curve(tint=1200, spec_res=100):
    ap_diam_rao = 0.3
    
    in_file_root = '{0:s}/roboAO_sensitivity_t{1:d}_R{2:d}'.format(work_dir, tint, spec_res)
    in_file = '{0:s}_ap{1:0.3f}'.format(in_file_root, ap_diam_rao)

    avg_tab = Table.read(in_file + '_avg_tab.fits')
    
    # Calculate the band-integrated SNR for each magnitude bin and filter.
    mag = avg_tab['mag']

    py.figure(1)
    py.clf()
    py.semilogy(avg_tab['mag'], avg_tab['snr_sum_z'], label='Z Filter', linewidth=2, color='cyan')
    py.semilogy(avg_tab['mag'], avg_tab['snr_sum_y'], label='Y Filter', linewidth=2, color='blue')
    py.semilogy(avg_tab['mag'], avg_tab['snr_sum_j'], label='J Filter', linewidth=2, color='red')
    py.semilogy(avg_tab['mag'], avg_tab['snr_sum_h'], label='H Filter', linewidth=2, color='green')
    py.xlabel('Vega Magnitude in Filter')
    py.ylabel('Signal to Noise in 20 minutes')
    py.ylim(10, 1e3)
    py.xlim(16, 21)
    py.legend()
    print 'Tint={0:d} s, R={1:d}, aper={2:0.3f}"'.format(tint, spec_res, ap_diam_rao)
    py.savefig('{0:s}_prop_snr_sum.png'.format(in_file_root))
        
    return

def plot_seeing_vs_ao_gain(tint=1200, spec_res=100):
    ap_diam_see = 0.8
    ap_diam_rao = 0.3
    
    in_file_root = '{0:s}/roboAO_sensitivity_t{1:d}_R{2:d}'.format(work_dir, tint, spec_res)

    in_file_see = '{0:s}_ap{1:0.3f}_seeing'.format(in_file_root, ap_diam_see)
    in_file_rao = '{0:s}_ap{1:0.3f}'.format(in_file_root, ap_diam_rao)

    avg_tab_rao = Table.read(in_file_rao + '_avg_tab.fits')
    avg_tab_see = Table.read(in_file_see + '_avg_tab.fits')
    
    # Calculate the band-integrated SNR for each magnitude bin and filter.
    mag_rao = avg_tab_rao['mag']
    mag_see = avg_tab_see['mag']

    hdr1 = '# {0:3s}  {1:15s}   {2:15s}   {3:15s}'
    hdr2 = '# {0:3s}  {1:7s} {2:7s}   {3:7s} {4:7s}   {5:7s} {6:7s}'
    print hdr1.format('Mag', 'Y SNR (summed)', 'J SNR (summed)', 'H SNR (summed)')
    print hdr2.format('', 'Robo-AO', 'Seeing', 'Robo-AO', 'Seeing', 'Robo-AO', 'Seeing')
    print hdr2.format('---', '-------', '-------', '-------', '-------', '-------', '-------')

    for mm in range(len(mag_rao)):
        fmt = '  {0:3d}  {1:7.1f} {2:7.1f}   {3:7.1f} {4:7.1f}   {5:7.1f} {6:7.1f}'
        print fmt.format(mag_rao[mm], 
                         avg_tab_rao['snr_sum_y'][mm], avg_tab_see['snr_sum_y'][mm],
                         avg_tab_rao['snr_sum_j'][mm], avg_tab_see['snr_sum_j'][mm],
                         avg_tab_rao['snr_sum_h'][mm], avg_tab_see['snr_sum_h'][mm])

    py.close(3)
    py.figure(3, figsize=(12,6))
    py.subplots_adjust(left=0.1)
    py.subplot(121)
    
    py.plot(avg_tab_rao['mag'], avg_tab_rao['snr_sum_z'] / avg_tab_see['snr_sum_z'], label='Z')
    py.plot(avg_tab_rao['mag'], avg_tab_rao['snr_sum_y'] / avg_tab_see['snr_sum_y'], label='Y')
    py.plot(avg_tab_rao['mag'], avg_tab_rao['snr_sum_j'] / avg_tab_see['snr_sum_j'], label='J')
    py.plot(avg_tab_rao['mag'], avg_tab_rao['snr_sum_h'] / avg_tab_see['snr_sum_h'], label='H')
    py.legend(loc='upper left')
    py.xlabel('Vega Magnitude')
    py.ylabel('Gain in Signal-to-Noise')
    py.ylim(0, 3.0)
    py.xlim(10, 21)
    # py.title('t={0:d} s, R={1:d}'.format(tint, spec_res))

    py.subplot(122)
    py.plot(avg_tab_rao['mag'], (avg_tab_rao['snr_sum_z'] / avg_tab_see['snr_sum_z'])**2, label='Z')
    py.plot(avg_tab_rao['mag'], (avg_tab_rao['snr_sum_y'] / avg_tab_see['snr_sum_y'])**2, label='Y')
    py.plot(avg_tab_rao['mag'], (avg_tab_rao['snr_sum_j'] / avg_tab_see['snr_sum_j'])**2, label='J')
    py.plot(avg_tab_rao['mag'], (avg_tab_rao['snr_sum_h'] / avg_tab_see['snr_sum_h'])**2, label='H')
    py.xlabel('Vega Magnitude')
    py.ylabel('Gain in Observing Efficiency')
    py.xlim(10, 21)
    
    py.savefig('{0:s}_prop_gain_ao_vs_see.png'.format(in_file_root))
    
    # py.xlim(13, 21)
    # py.ylim(0, 3)
    # py.arrow(16.1, 30, 4.05, 0, color='red', width=2, head_width=10, head_length=0.5, 
    #          length_includes_head=True, fc='red', ec='red')
    # py.text(17.3, 11, '250x More\nSupernovae Ia', color='red', 
    #         horizontalalignment='left', fontweight='bold', fontsize=16)
    # py.legend()
    # py.savefig(in_file_rao + '_proposal.png')

    return
