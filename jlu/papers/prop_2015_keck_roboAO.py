import numpy as np
import pylab as py
from astropy.table import Table
from astropy.table import Column

work_dir = '/u/jlu/work/ao/roboAO'


def plot_seeing_vs_ao(tint=1200, spec_res=100):
    ap_rad_see = 0.4
    ap_rad_rao = 0.15
    
    in_file_root = '{0:s}/roboAO_sensitivity_t{1:d}_R{2:d}'.format(work_dir, tint, spec_res)

    in_file_see = '{0:s}_ap{1:0.3f}_seeing'.format(in_file_root, ap_rad_see)
    in_file_rao = '{0:s}_ap{1:0.3f}'.format(in_file_root, ap_rad_rao)

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
        
    py.clf()
    py.semilogy(avg_tab_rao['mag'], avg_tab_rao['snr_sum_j'], label='UH Robo-AO', 
                linewidth=2)
    py.semilogy(avg_tab_rao['mag'], avg_tab_see['snr_sum_j'], label='Seeing-Limited',
                linewidth=2)
    py.xlabel('J-band Magnitude')
    py.ylabel('Signal-to-Noise in 1200 sec')
    py.xlim(13, 21)
    py.ylim(1, 1e4)
    py.arrow(16.1, 30, 4.05, 0, color='red', width=2, head_width=10, head_length=0.5, 
             length_includes_head=True, fc='red', ec='red')
    py.text(17.3, 11, '250x More\nSupernovae Ia', color='red', 
            horizontalalignment='left', fontweight='bold', fontsize=16)
    py.legend()
    # py.title('Tint={0:d} s, R={1:d}'.format(tint, spec_res))
    py.savefig(in_file_rao + '_proposal.png')

    return

def plot_sensitivity_curve(tint=1200, spec_res=100):
    ap_rad_rao = 0.15
    
    in_file_root = '{0:s}/roboAO_sensitivity_t{1:d}_R{2:d}'.format(work_dir, tint, spec_res)
    in_file = '{0:s}_ap{1:0.3f}'.format(in_file_root, ap_rad_rao)

    avg_tab = Table.read(in_file + '_avg_tab.fits')
    
    # Calculate the band-integrated SNR for each magnitude bin and filter.
    mag = avg_tab['mag']

    py.clf()
    py.semilogy(avg_tab['mag'], avg_tab['snr_sum_y'], label='Y Filter', linewidth=2, color='blue')
    py.semilogy(avg_tab['mag'], avg_tab['snr_sum_j'], label='J Filter', linewidth=2, color='red')
    py.semilogy(avg_tab['mag'], avg_tab['snr_sum_h'], label='H Filter', linewidth=2, color='green')
    py.xlabel('Magnitude in Filter')
    py.ylabel('Signal to Noise')
    py.ylim(10, 1e3)
    py.xlim(15, 21)
    py.legend()
    print 'Tint={0:d} s, R={1:d}, aper={2:0.3f}"'.format(tint, spec_res, ap_rad_rao)
    py.savefig(in_file + '_snr_sum_proposal.png')
    
    py.clf()
    py.semilogy(avg_tab['mag'], avg_tab['snr_sum_j'], label='J Filter', linewidth=2, color='red')
    py.xlabel('J-Band Magnitude')
    py.ylabel('Signal to Noise')
    py.ylim(10, 1e3)
    py.xlim(15, 21)
    py.savefig(in_file + '_snr_sum_j_proposal.png')

    return
