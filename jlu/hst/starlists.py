import numpy as np
import pylab as py
import atpy
from jlu.hst import photometry
import math

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

def read_xyviq1(filename, saveToFits=False, filtNames=None):
    tab = atpy.Table(filename, type='ascii')

    ncols = len(tab.columns)
    nfilt = (ncols - 3) / 10

    if filtNames != None and len(filtNames) != nfilt:
        print 'Please specify %d filter names to return Vega magnitudes.' % \
            (nfilt)
        return

    tab.rename_column('col1', 'x')
    tab.rename_column('col2', 'y')
    tab.rename_column('col3', 'name')

    for ff in range(nfilt):
        suffix = '_%d' % ff
        if filtNames != None:
            suffix = '_%s' % filtNames[ff]

        tab.rename_column('col%d' % (4+10*ff), 'x%s' % suffix)
        tab.rename_column('col%d' % (5+10*ff), 'y%s' % suffix)
        tab.rename_column('col%d' % (6+10*ff), 'xe%s' % suffix)
        tab.rename_column('col%d' % (7+10*ff), 'ye%s' % suffix)
        tab.rename_column('col%d' % (8+10*ff), 'f%s' % suffix)
        tab.rename_column('col%d' % (9+10*ff), 'fe%s' % suffix)
        tab.rename_column('col%d' % (10+10*ff), 'q%s' % suffix)
        tab.rename_column('col%d' % (11+10*ff), 'f_other%s' % suffix)
        tab.rename_column('col%d' % (12+10*ff), 'Nfound%s' % suffix)
        tab.rename_column('col%d' % (13+10*ff), 'Nsurvive%s' % suffix)

        f_col = 'f%s' % suffix
        fe_col = 'fe%s' % suffix

        mag = -2.5 * np.log10(tab[f_col])
        magErr = (2.5 / math.log(10.)) * tab[fe_col] / tab[f_col]

        tab.add_column('m%s' % suffix, mag, after=f_col)
        tab.add_column('me%s' % suffix, magErr, after=fe_col)
            
    if saveToFits:
        tab.write(filename + '.fits', overwrite=True)

    return tab
        


def read_ks2_avg_uv(filename, Nfilt, saveToFits=False, filtNames=None):
    if filtNames != None and Nfilt != len(filtNames):
        print 'Please specify %d filter names to return Vega magnitudes.' % \
            (Nfilt)
        return

    tab = atpy.Table(filename, type='ascii')
    
    # Extract some temporary information and dlete it from the table.
    filterNum = tab.col14
    tab.remove_columns(['col14', 'col15'])

    # Cleanup some funny behavior with '**'
    xe = tab.col4
    ye = tab.col5
    
    idx = np.where(np.core.defchararray.startswith(xe, '*') == True)[0]
    xe[idx] = 99999.0
    ye[idx] = 99999.0

    xe = np.array(xe, dtype=float)
    ye = np.array(ye, dtype=float)

    tab.remove_columns(['col4', 'col5'])
    tab.add_column('col4', xe, before='col6')
    tab.add_column('col5', ye, before='col6')

    final = atpy.Table()

    for ii in range(Nfilt):
        ff = ii + 1

        other = tab.where(filterNum == ff)

        # These are the columns that are not filter dependent
        if ii == 0:
            final.add_column('x_0', other.col12)
            final.add_column('y_0', other.col13)
            final.add_column('name', other.col16)
            final.add_column('tile', other.col11)
        else:
            if len(other) != len(final):
                print 'read_ks2: PROBLEM with multiple filters'

        mag = -2.5 * np.log10(other.col3)
        magErr = (2.5 / math.log(10)) * (other.col6 / other.col3)
        magSuffix = '%d' % ff

        if filtNames != None:
            mag += photometry.ZP[filtNames[ii]]
            magSuffix = filtNames[ii]


        final.add_column('x_%d' % ff, other.col1)
        final.add_column('y_%d' % ff, other.col2)
        final.add_column('f_%d' % ff, other.col3)
        final.add_column('m_%s' % magSuffix, mag)
        final.add_column('xe_%d' % ff, other.col4)
        final.add_column('ye_%d' % ff, other.col5)
        final.add_column('fe_%d' % ff, other.col6)
        final.add_column('me_%s' % magSuffix, magErr)
        final.add_column('q_%d' % ff, other.col7)
        final.add_column('f_other_%d' % ff, other.col8)
        final.add_column('Nfound_%d' % ff, other.col9)
        final.add_column('Nsurvive_%d' % ff, other.col10)

    if saveToFits:
        final.write(filename + '.fits', overwrite=True)

    return final


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
    xerr = np.median(tabTrim.xe)
    yerr = np.median(tabTrim.ye)
    merr = np.median(tabTrim.me)

    print 'Median Errors for Stars Between [%.1f, %.1f] in %s' % \
        (minMagGood, maxMagGood, matchupFile)
    print '     Photometric Error: %.2f mag' % (merr)
    print '   X Astrometric Error: %.3f pix' % (xerr)
    print '   Y Astrometric Error: %.3f pix' % (yerr)
    

    
def make_brite(matchup_files, trimIdx=0, trimMag=-7):
    """
    Take an input list of MATCHUP files (assumes they have the same stars, and the
    same length) and trim out only the bright stars. The resulting output file contains
    the X and Y position (from the first file) and the list of all magnitudes for each star.

    The brightness criteria is done on the specified file (matchup_files[trimIdx]) at the
    specified magnitude (trimMag).
    """
    if not hasattr(matchup_files, '__iter__'):
        matchup_files = [matchup_files]

    starlists = []

    for mfile in matchup_files:
        starlists.append( read_matchup(mfile) )

    idx = np.where(starlists[trimIdx].m < trimMag)[0]

    brite = open('BRITE.XYMMM', 'w')
    
    for ii in idx:
        brite.write('%10.4f  %10.4f ' % (starlists[0].x[ii], starlists[0].y[ii]))
        for mm in range(len(starlists)):
            brite.write('  %10.2f' % (starlists[mm].m[ii]))
        brite.write('\n')
    
    brite.close()
    
