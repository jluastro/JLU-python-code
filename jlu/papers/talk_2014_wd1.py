import numpy as np
import pylab as py
from astropy.table import Table
from jlu.hst import photometry
import math

# work_dir = '/u/jlu/data/wd1/hst/reduce_2014_06_17/'
# catalog = work_dir + '21.ALIGN_KS2/wd1_catalog.fits'
# out_dir = work_dir + 'plots/'
work_dir = '/u/jlu/work/wd1/'
catalog = work_dir + 'wd1_catalog_onepass_vel.fits'
cluster = work_dir + 'wd1_cluster_0.75.fits'
out_dir = work_dir + 'plots/'

def plot_pos_vel_errors():
    t = Table.read(catalog)

    yr_filt = ['2005_814', '2010_160', '2010_125', '2010_139', '2013_160']
    filt = ['F814W', 'F160W', 'F125W', 'F139W', 'F160W']
    year = ['2005', '2010', '2010', '2010', '2013']

    for ii in range(len(yr_filt)):
        ff = yr_filt[ii]
        
        # Velocity Errors
        py.clf()
        py.semilogy(t['m_' + ff], t['xe_' + ff]*50., 'r.', label='X',
                    ms=2, alpha=0.2)
        py.semilogy(t['m_' + ff], t['xe_' + ff]*50., 'b.', label='Y',
                    ms=2, alpha=0.2)
        py.xlabel(filt[ii] + ' (magnitude)')
        py.ylabel('Pos. Error (mas)')
        py.title(year[ii])

        if filt[ii] == 'F814W':
            py.xlim(11, 19)
        else:
            py.xlim(12, 23)
        py.ylim(0.1, 10)
        py.legend()
        py.savefig(out_dir + 'pos_err_' + ff + '.png')

        # Magnitude errors
        py.clf()
        py.semilogy(t['m_' + ff], t['me_' + ff], 'r.', label='X',
                    ms=2, alpha=0.2)
        py.xlabel(filt[ii] + ' (magnitude)')
        py.ylabel('Phot. Error (mag)')
        py.title(year[ii])

        if filt[ii] == 'F814W':
            py.xlim(11, 19)
        else:
            py.xlim(12, 23)
        py.ylim(0.01, 1)
        py.savefig(out_dir + 'mag_err_' + ff + '.png')

    py.clf()
    py.plot(t['m_2013_160'], t['fit_vxe']*50., 'r.', label='X',
            ms=5, alpha=0.2)
    py.plot(t['m_2013_160'], t['fit_vye']*50., 'b.', label='Y',
            ms=5, alpha=0.2)
    py.xlabel('F160W (magnitude)')
    py.ylabel('Vel. Error (mas/yr)')
    py.title('2013 - 2005')
    py.xlim(12, 22)
    py.ylim(0, 0.6)
    py.legend()
    py.savefig(out_dir + 'vel_err.png')

    
def make_cmd():
    """
    """    
    t = Table.read(catalog)
    # c = Table.read(cluster)

    fix_magnitudes(t)

    idx_t = np.where((t['xe_2005_814'] > 0) & (t['xe_2010_125'] > 0) & (t['xe_2010_139'] > 0) & 
                     (t['xe_2010_160'] > 0) & (t['xe_2013_160'] > 0) & 
                     (t['fit_vx'] < 0.01) & (t['fit_vy'] < 0.01))[0]
    # idx_c = np.where((c['xe_2005_814'] > 0) & (c['xe_2010_125'] > 0) & (c['xe_2010_139'] > 0) & 
    #                  (c['xe_2010_160'] > 0) & (c['xe_2013_160'] > 0) & 
    #                  (c['fit_vxe'] < 0.01) & (c['fit_vye'] < 0.01))[0]

    print 'Keeping {0:d} of {1:d} stars in complete catalog'.format(len(idx_t), len(t))
    # print 'Keeping {0:d} of {1:d} stars in cluster catalog'.format(len(idx_c), len(c))
    t = t[idx_t]
    # c = c[idx_c]
    
    py.clf()
    py.plot(t['m_2005_814'] - t['m_2010_125'], t['m_2005_814'], 'k.', ms=3)
    # py.plot(c['m_2005_814'] - c['m_2013_125'], c['m_2005_814'], 'r.', ms=3)
    py.xlabel('F814W - F125W (mag)')
    py.ylabel('F814W (mag)')
    py.ylim(12, 25)
    py.xlim(0, 8)
    py.xlim(2, 5)
    py.gca().invert_yaxis()
    py.savefig(out_dir + 'cmd_814_125.png')

    py.clf()
    py.plot(t['m_2010_125'] - t['m_2013_160'], t['m_2010_125'], 'k.', ms=3)
    py.xlabel('F125W - F160W (mag)')
    py.ylabel('F125W (mag)')
    py.ylim(9, 22)
    py.xlim(0, 2)
    py.gca().invert_yaxis()
    py.savefig(out_dir + 'cmd_125_160.png')

def clean_catalog_xym1mat():
    tab = Table.read(work_dir + 'mat_all_good.fits')
    
    suffix = ['2005_814', '2010_160', '2013_160', '2010_139', '2010_125']
    filt = ['F814W', 'F160W', 'F160W', 'F139M', 'F125W']

    for ii in range(len(suffix)):
        tab.rename_column('col{0:02d}'.format((ii*8)+1), 'name_{0:s}'.format(suffix[ii]))
        tab.rename_column('col{0:02d}'.format((ii*8)+2), 'x_{0:s}'.format(suffix[ii]))
        tab.rename_column('col{0:02d}'.format((ii*8)+3), 'y_{0:s}'.format(suffix[ii]))
        tab.rename_column('col{0:02d}'.format((ii*8)+4), 'm_{0:s}'.format(suffix[ii]))
        tab.rename_column('col{0:02d}'.format((ii*8)+5), 'xe_{0:s}'.format(suffix[ii]))
        tab.rename_column('col{0:02d}'.format((ii*8)+6), 'ye_{0:s}'.format(suffix[ii]))
        tab.rename_column('col{0:02d}'.format((ii*8)+7), 'me_{0:s}'.format(suffix[ii]))
        tab.rename_column('col{0:02d}'.format((ii*8)+8), 'n_{0:s}'.format(suffix[ii]))

        tab['m_{0:s}'.format(suffix[ii])] += photometry.ZP[filt[ii]]
        

    tab.write('wd1_catalog_onepass.fits', overwrite=True)
                   
def fix_magnitudes(t):
    t['m_2005_814'] -= -2.5 * math.log10(2407. / 3.)

    

    
