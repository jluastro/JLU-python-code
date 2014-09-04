import numpy as np
import pylab as py
from astropy.table import Table

work_dir = '/u/jlu/data/wd1/hst/reduce_2014_06_17/'
catalog = work_dir + '21.ALIGN_KS2/wd1_catalog.fits'
out_dir = work_dir + 'plots/'

def plot_pos_vel_errors():
    t = Table.read(catalog)

    yr_filt = ['2005_814', '2010_160', '2010_125', '2010_139', '2013_160']
    filt = ['F814W', 'F160W', 'F125W', 'F139W', 'F160W']
    year = ['2005', '2010', '2010', '2010', '2013']

    for ii in range(len(yr_filt)):
        ff = yr_filt[ii]
        
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

    
    
