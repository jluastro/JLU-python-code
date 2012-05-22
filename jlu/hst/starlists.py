import numpy as np
import pylab as py
import atpy

def read_matchup(filename):
    tab = atpy.Table(filename, type='ascii')

    tab.rename_column('col1', 'x')
    tab.rename_column('col2', 'y')
    tab.rename_column('col3', 'm')
    tab.rename_column('col4', 'xe')
    tab.rename_column('col5', 'ye')
    tab.rename_column('col6', 'me')
    tab.rename_column('col7', 'aux')
    tab.rename_column('col8', 'N_fnd')
    tab.rename_column('col9', 'N_xywell')
    tab.rename_column('col10', 'N_mwell')
    tab.rename_column('col11', 'N_min')
    tab.rename_column('col12', 'name')

    return tab


def plot_matchup_err(matchupFile, minMagGood=-13.5, maxMagGood=-5):
    tab = read_matchup(matchupFile)

    # Photometric errors
    py.figure(1)
    py.clf()
    py.semilogy(tab.m, tab.me, 'k.')
    py.xlabel('Instrumental Magnitude')
    py.ylabel('Photometric RMS Error (pix)')
    py.ylim(0.001, 1)
    py.xlim(-22, 0)
    py.title(matchupFile)
    py.savefig('plot_mag_err_%s.png' % matchupFile)

    # Astrometric errors
    py.figure(2)
    py.clf()
    py.semilogy(tab.m, tab.xe, 'r.', label='X')
    py.semilogy(tab.m, tab.ye, 'b.', label='Y')
    py.xlabel('Instrumental Magnitude')
    py.ylabel('Astrometric RMS Error (pix)')
    py.ylim(0.001, 1)
    py.xlim(-22, 0)
    py.title(matchupFile)
    py.legend(loc='upper left', numpoints=1)
    py.savefig('plot_pos_err_%s.png' % matchupFile)

    # Lets get the median errors inside the acceptable range.
    tabTrim = tab.where((tab.m > minMagGood) & (tab.m < maxMagGood))
    xerr = np.median(tab.xe)
    yerr = np.median(tab.ye)
    merr = np.median(tab.me)

    print 'Median Errors for Stars Between [%.1f, %.1f] in %s' % \
        (minMagGood, maxMagGood, matchupFile)
    print '     Photometric Error: %.2f mag' % (merr)
    print '   X Astrometric Error: %.3f pix' % (xerr)
    print '   Y Astrometric Error: %.3f pix' % (yerr)
    

    
