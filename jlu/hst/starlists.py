import numpy as np
import pylab as py
import atpy
from jlu.hst import photometry
import math
import time
import os, shutil
from jlu.util import statsIter
from jlu.util import CatalogFinder

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

def read_xyviq0(filename, saveToFits=False, filtNames=None):
    tab = atpy.Table(filename, type='ascii')

    ncols = len(tab.columns)
    nfilt = (ncols - 9) / 2

    tab.add_keyword('NFILT', nfilt)

    if filtNames != None and len(filtNames) != nfilt:
        print 'Please specify %d filter names to return Vega magnitudes.' % \
            (nfilt)
        return

    tab.rename_column('col1', 'x')
    tab.rename_column('col2', 'y')
    tab.rename_column('col3', 'm')
    tab.rename_column('col7', 'name')
    tab.remove_columns(['col4', 'col5', 'col6', 'col8', 'col9'])
    tab.table_name = ''

    for ff in range(nfilt):
        suffix = '_%d' % ff
        if filtNames != None:
            suffix = '_%s' % filtNames[ff]

        tab.rename_column('col%d' % (10+2*ff), 'f%s' % suffix)
        tab.rename_column('col%d' % (11+2*ff), 'fe%s' % suffix)

        f_col = 'f%s' % suffix
        fe_col = 'fe%s' % suffix

        mag = -2.5 * np.log10(tab[f_col])
        magErr = (2.5 / math.log(10.)) * tab[fe_col] / tab[f_col]

        tab.add_column('m%s' % suffix, mag, after=f_col)
        tab.add_column('me%s' % suffix, magErr, after=fe_col)
            
    if saveToFits:
        tab.write(filename + '.fits', overwrite=True)

    return tab
        

def read_xyviq1(filename, saveToFits=False, filtNames=None):
    tab = atpy.Table(filename, type='ascii')

    ncols = len(tab.columns)
    nfilt = (ncols - 3) / 6

    tab.add_keyword('NFILT', nfilt)

    if filtNames != None and len(filtNames) != nfilt:
        print 'Please specify %d filter names to return Vega magnitudes.' % \
            (nfilt)
        return

    tab.rename_column('col1', 'x')
    tab.rename_column('col2', 'y')
    tab.rename_column('col3', 'm')
    tab.rename_column('col4', 'name')
    tab.table_name = ''

    for ff in range(nfilt):
        suffix = '_%d' % ff
        if filtNames != None:
            suffix = '_%s' % filtNames[ff]

        # tab.rename_column('col%d' % (4+10*ff), 'x%s' % suffix)
        # tab.rename_column('col%d' % (5+10*ff), 'y%s' % suffix)
        # tab.rename_column('col%d' % (6+10*ff), 'xe%s' % suffix)
        # tab.rename_column('col%d' % (7+10*ff), 'ye%s' % suffix)
        # tab.rename_column('col%d' % (8+10*ff), 'f%s' % suffix)
        # tab.rename_column('col%d' % (9+10*ff), 'fe%s' % suffix)
        # tab.rename_column('col%d' % (10+10*ff), 'q%s' % suffix)
        # tab.rename_column('col%d' % (11+10*ff), 'f_other%s' % suffix)
        # tab.rename_column('col%d' % (12+10*ff), 'Nfound%s' % suffix)
        # tab.rename_column('col%d' % (13+10*ff), 'Nsurvive%s' % suffix)

        tab.rename_column('col%d' % (5+6*ff), 'f%s' % suffix)
        tab.rename_column('col%d' % (6+6*ff), 'fe%s' % suffix)
        tab.rename_column('col%d' % (7+6*ff), 'q%s' % suffix)
        tab.rename_column('col%d' % (8+6*ff), 'f_other%s' % suffix)
        tab.rename_column('col%d' % (9+6*ff), 'Nfound%s' % suffix)
        tab.rename_column('col%d' % (10+6*ff), 'Nsurvive%s' % suffix)

        f_col = 'f%s' % suffix
        fe_col = 'fe%s' % suffix

        mag = -2.5 * np.log10(tab[f_col])
        magErr = (2.5 / math.log(10.)) * tab[fe_col] / tab[f_col]

        tab.add_column('m%s' % suffix, mag, after=f_col)
        tab.add_column('me%s' % suffix, magErr, after=fe_col)
            
    if saveToFits:
        tab.write(filename + '.fits', overwrite=True)

    return tab

def read_xyviq2(filename, saveToFits=False, filtNames=None):
    tab = read_xyviq1(filename, saveToFits=saveToFits, filtNames=filtNames)

    return tab


def read_ks2_avg_uv1(filename, saveToFits=False, filtNames=None):
    tab = atpy.Table(filename, type='ascii')

    Nfilt = tab.col14.max()
    tab.add_keyword('NFILT', Nfilt)

    if filtNames != None and Nfilt != len(filtNames):
        print 'Please specify %d filter names to return Vega magnitudes.' % \
            (Nfilt)
        return

    # Extract some temporary information and delete it from the table.
    filterNum = tab.col14
    tab.remove_columns(['col14'])

    # Cleanup some funny behavior with '**'
    xe = tab.col4
    ye = tab.col5

    try:
        idx = np.where(np.core.defchararray.startswith(xe, '*') == True)[0]
        xe[idx] = 99999.0
        ye[idx] = 99999.0
    except TypeError:
        # Everything is okay already.
        pass

    xe = np.array(xe, dtype=float)
    ye = np.array(ye, dtype=float)

    tab.remove_columns(['col4', 'col5'])
    tab.add_column('col4', xe, before='col6')
    tab.add_column('col5', ye, before='col6')

    final = atpy.Table()
    final.add_keyword('NFILT', tab.keywords['NFILT'])

    for ii in range(Nfilt):
        ff = ii + 1

        other = tab.where(filterNum == ff)

        # These are the columns that are not filter dependent
        if ii == 0:
            final.add_column('x_0', other.col1)
            final.add_column('y_0', other.col2)
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
        final.add_column('pass_%d' % ff, other.col15)
        final.table_name = ''

    if saveToFits:
        final.write(filename + '.fits', overwrite=True)

    return final


def read_ks2_avg_z0(filename, saveToFits=False, filtNames=None):
    tab = atpy.Table(filename, type='ascii')

    Nfilt = tab.col8.max()
    tab.add_keyword('NFILT', Nfilt)

    if filtNames != None and Nfilt != len(filtNames):
        print 'Please specify %d filter names to return Vega magnitudes.' % \
            (Nfilt)
        return

    # Extract some temporary information and delete it from the table.
    filterNum = tab.col8
    tab.remove_columns(['col8'])

    # Cleanup some funny behavior with '**'
    z = tab.col3

    try:
        idx = np.where(np.core.defchararray.startswith(z, '*') == True)[0]
        z[idx] = 0.0
        z = np.array(z, dtype=float)

        tab.remove_columns(['col3'])
        tab.add_column('col3', z, before='col4')
    except TypeError:
        # Everything is okay already.
        pass

    final = atpy.Table()
    final.add_keyword('NFILT', tab.keywords['NFILT'])

    for ii in range(Nfilt):
        ff = ii + 1

        other = tab.where(filterNum == ff)

        # These are the columns that are not filter dependent
        if ii == 0:
            final.add_column('x_0', other.col1)
            final.add_column('y_0', other.col2)
            final.add_column('name', other.col10)
            final.add_column('tile', other.col7)
        else:
            if len(other) != len(final):
                print 'read_ks2: PROBLEM with multiple filters'

        mag = -2.5 * np.log10(other.col3)
        magErr = (2.5 / math.log(10)) * (other.col4 / other.col3)
        magSuffix = '%d' % ff

        if filtNames != None:
            mag += photometry.ZP[filtNames[ii]]
            magSuffix = filtNames[ii]

        final.add_column('f_%d' % ff, other.col3)
        final.add_column('m_%s' % magSuffix, mag)
        final.add_column('fe_%d' % ff, other.col4)
        final.add_column('me_%s' % magSuffix, magErr)
        final.add_column('Nfound_%d' % ff, other.col5)
        final.add_column('Nsurvive_%d' % ff, other.col6)
        final.add_column('pass_%d' % ff, other.col9)
        final.table_name = ''

    if saveToFits:
        final.write(filename + '.fits', overwrite=True)

    return final

def read_ks2_avg_z2(filename, saveToFits=False, filtNames=None):
    tab = atpy.Table(filename, type='ascii')

    # Extract some temporary information and delete it from the table.
    filterNum = tab.col10
    tab.remove_columns(['col10'])

    Nfilt = filterNum.max()
    tab.add_keyword('NFILT', Nfilt)

    if filtNames != None and Nfilt != len(filtNames):
        print 'Please specify %d filter names to return Vega magnitudes.' % \
            (Nfilt)
        return

    # Cleanup some funny behavior with '**'
    z = tab.col3
    try:
        idx = np.where(np.core.defchararray.startswith(z, '*') == True)[0]
        z[idx] = 0.0
        z = np.array(z, dtype=float)

        tab.remove_columns(['col3'])
        tab.add_column('col3', z, before='col4')
    except TypeError:
        # Everything is okay already.
        pass

    final = atpy.Table()
    final.add_keyword('NFILT', tab.keywords['NFILT'])

    for ii in range(Nfilt):
        ff = ii + 1

        other = tab.where(filterNum == ff)

        # These are the columns that are not filter dependent
        if ii == 0:
            final.add_column('x_0', other.col1)
            final.add_column('y_0', other.col2)
            final.add_column('name', other.col12)
            final.add_column('tile', other.col9)
        else:
            if len(other) != len(final):
                print 'read_ks2: PROBLEM with multiple filters'

        mag = -2.5 * np.log10(other.col3)
        magErr = (2.5 / math.log(10)) * (other.col4 / other.col3)
        magSuffix = '%d' % ff

        if filtNames != None:
            mag += photometry.ZP[filtNames[ii]]
            magSuffix = filtNames[ii]

        final.add_column('f_%d' % ff, other.col3)
        final.add_column('m_%s' % magSuffix, mag)
        final.add_column('fe_%d' % ff, other.col4)
        final.add_column('me_%s' % magSuffix, magErr)
        final.add_column('q_%d' % ff, other.col5)
        final.add_column('f_other_%d' % ff, other.col6)
        final.add_column('Nfound_%d' % ff, other.col7)
        final.add_column('Nsurvive_%d' % ff, other.col8)
        final.add_column('pass_%d' % ff, other.col11)
        final.table_name = ''

    if saveToFits:
        final.write(filename + '.fits', overwrite=True)

    return final

def read_xym1mat(xym1mat_afile, xym1mat_cfile):
    """
    Read in starlist 1 and starlist 2 (output from xym1mat) and return a final
    list of the matched sources with x, y, m, xe, ye, me in both of the starlists
    in a common coordinate system.

    This routine assumes that the input starlists for xym1mat both had
    formats of [x, y, m, xe, ye, me, name]
    """
    # Search the A file to pull out the first star list.
    for line in open(xym1mat_afile, 'r'):
        if re.search('# ARG1:', line):
            xymeee1_tmp = line.split()[-1]
            xymeee1 = xymeee1_tmp.split('(')[0]
            print 'Starlist 1 (XYMEEE) = ', xymeee1

    # Read in the first starlist. These are untransformed coordinates.
    stars1 = atpy.Table(xymeee1, type='ascii')
    stars1.rename_column('col1', 'x')
    stars1.rename_column('col2', 'y')
    stars1.rename_column('col3', 'm')
    stars1.rename_column('col4', 'xe')
    stars1.rename_column('col5', 'ye')
    stars1.rename_column('col6', 'me')
    stars1.rename_column('col7', 'name')

    # Loop through the lines in the C file. In principle there should be one
    # C file line for every starlist 1 entry. But in practice, some stuff gets
    # accidentally thrown out (not sure why).
    idx1 = 0
    idx2 = 0

    cfile = open(xym1mat_cfile, 'r')

    for line in cfile:
        fields = line.split()

        x1_orig = float(fields[4])
        y1_orig = float(fields[5])
        m1_orig = float(fields[6])
        x1 = float(fields[7])
        y1 = float(fields[8])
        x2 = float(fields[13])
        y2 = float(fields[14])
        m2 = float(fields[15])
        xe2 = float(fields[16])
        ye2 = float(fields[17])
        me2 = float(fields[18])
        name2 = fields[19]

        # Catch non-matches
        if (float(fields[9]) == 0) and (float(fields[1]) == 0):
            idx1 += 1
            idx2 += 1
            continue

        # Check that the positions are consistent for the first starlist
        # and this isn't a case of a missing source.
        dr = np.hypot()
            
    

def process_ks2_output(ks2Root, readFromFits=True):
    """
    Positional information comes from the UV1 file.
    Photometric information comes from the XYVIQ1 file, except for bright
    saturated stars, which come from the input XYVIQ0 file (input fluxes).
    Photometric errors for the bright stars come from the XYVIQ1 file.
    """
    uv1_file = ks2Root + '.FIND_AVG_UV1_F'
    xyviq0_file = ks2Root + '.XYVIQ0'
    z0_file = ks2Root + '.FIND_AVG_Z0_F'

    # These will hold the atpy tables.
    uv1 = None
    xyviq0 = None
    z0 = None
    
    # Read existing atpy tables, unless specified otherwise.
    if readFromFits:
        # Read the UV1 file
        if os.path.exists(uv1_file + '.fits'):
            uv1 = atpy.Table(uv1_file + '.fits')

        # Readh the XYVIQ0 file
        if os.path.exists(xyviq0_file + '.fits'):
            xyviq0 = atpy.Table(xyviq0_file + '.fits')

        # Read the Z0 file
        if os.path.exists(z0_file + '.fits'):
            z0 = atpy.Table(z0_file + '.fits')

    
    if uv1 == None:
        print 'Reading UV1 file: ' + time.ctime()
        uv1 = read_ks2_avg_uv1(ks2Root + '.FIND_AVG_UV1_F', saveToFits=True)

    if xyviq0 == None:
        print 'Reading XYVIQ0 file: ' + time.ctime()
        xyviq0 = read_xyviq0(ks2Root + '.XYVIQ0', saveToFits=True)

    if z0 == None:
        print 'Reading Z0 file: ' + time.ctime()
        z0 = read_ks2_avg_z0(ks2Root + '.FIND_AVG_Z0_F', saveToFits=True)

    # Go ahead and make FITS tables for the other files as well.
    if not readFromFits:
        print 'Reading Z2 file: ' + time.ctime()
        read_ks2_avg_z2(ks2Root + '.FIND_AVG_Z2_F', saveToFits=True)
        print 'Reading XYVIQ1 file: ' + time.ctime()
        read_xyviq1(ks2Root + '.XYVIQ1', saveToFits=True)
        print 'Reading XYVIQ2 file: ' + time.ctime()
        read_xyviq2(ks2Root + '.XYVIQ2', saveToFits=True)
 
    print 'Creating output table: ' + time.ctime()

    # Make a table that contains the following columns
    # X, Y (averaged over all filters)
    # X in filt 1
    # Y in filt 1
    # M in filt 1
    # Xe in filt 1
    # Ye in filt 1
    # Me in filt 1
    # Repeat for other filters.

    nfilt = xyviq0.keywords['NFILT']
    
    final = atpy.Table()
    final.table_name = ''
    final.add_keyword('NFILT', nfilt)

    final.add_column('x_0', xyviq0.x)
    final.add_column('y_0', xyviq0.y)
    final.add_column('name', xyviq0.name)
    final.add_column('tile', uv1.tile)
    
    for ii in range(nfilt):
        ff = ii + 1
        print 'Processing filter %d: ' % ff + time.ctime()

        # For each filter, get the positions
        xCol = 'x_%d' % ff
        yCol = 'y_%d' % ff
        mCol = 'm_%d' % ff
        xeCol = 'xe_%d' % ff
        yeCol = 'ye_%d' % ff
        meCol = 'me_%d' % ff
        
        final.add_column(xCol, uv1[xCol])
        final.add_column(yCol, uv1[yCol])
        final.add_column(mCol, uv1[mCol])
        final.add_column(xeCol, uv1[xeCol])
        final.add_column(yeCol, uv1[yeCol])
        final.add_column(meCol, uv1[meCol])

        # Add a column to keep track of the
        # source of the photometry.
        photCol = 'fsrc_%d' % ff
        final.add_column(photCol, np.zeros(len(final)))

        # Now cleanup the bright stars
        passCol = 'pass_%d' % ff
        idx = np.where(uv1[passCol] == 0)
        print '    Dealing with %d bright stars: ' % len(idx[0]) + time.ctime()

        final[photCol][idx] = np.ones(len(idx))
        final[mCol][idx] = z0[mCol][idx]
        final[meCol][idx] = z0[meCol][idx]

    final.write(ks2Root + '_catalog.fits', overwrite=True)

    return final


def plot_matchup_err(matchupFile, minMagGood=-13.5, maxMagGood=-5):
    """
    Plot up photometric and astrometric errors from the results of
    an xym2mat pass. Save the plots to files in the current directory.
    """
    tab = read_matchup(matchupFile)

    # Photometric errors
    py.figure(1)
    py.clf()
    py.semilogy(tab.m, tab.me, 'k.', alpha=0.2)
    py.xlabel('Instrumental Magnitude')
    py.ylabel('Photometric RMS Error (pix)')
    py.ylim(0.001, 1)
    py.xlim(-22, 0)
    py.title(matchupFile)
    py.savefig('plot_mag_err_%s.png' % matchupFile)

    # Astrometric errors
    py.figure(2)
    py.clf()
    py.semilogy(tab.m, tab.xe, 'r.', label='X', alpha=0.2)
    py.semilogy(tab.m, tab.ye, 'b.', label='Y', alpha=0.2)
    py.xlabel('Instrumental Magnitude')
    py.ylabel('Astrometric RMS Error (pix)')
    py.ylim(0.001, 1)
    py.xlim(-22, 0)
    py.title(matchupFile)
    py.legend(loc='upper left', numpoints=1)
    py.savefig('plot_pos_err_%s.png' % matchupFile)

    # Lets get the median errors inside the acceptable range.
    tabTrim = tab.where((tab.m > minMagGood) & (tab.m < maxMagGood) & (tab.me < 9))
    xerr = np.median(tabTrim.xe)
    yerr = np.median(tabTrim.ye)
    merr = np.median(tabTrim.me)

    print 'Median Errors for %d of %d Stars Between [%.1f, %.1f] in %s' % \
        (len(tabTrim), len(tab), minMagGood, maxMagGood, matchupFile)
    print '     Photometric Error: %.2f mag' % (merr)
    print '   X Astrometric Error: %.3f pix' % (xerr)
    print '   Y Astrometric Error: %.3f pix' % (yerr)
    

    
def shift_matchup(filename):
    stars = read_matchup(filename)

    idx = np.where((stars.x > -999) & (stars.y > -999))[0]

    minX = stars.x[idx].min()
    minY = stars.y[idx].min()

    shiftX = 20.0 - minX
    shiftY = 20.0 - minY

    if shiftX > 0:
        stars.x += shiftX
	print 'Shifting X by %.2f' % shiftX
    if shiftY > 0:
        stars.y += shiftY
	print 'Shifting Y by %.2f' % shiftY

    _out = open(filename + '.shift', 'w')
    fmt = '%10.4f %10.4f %10.4f %10.4f %10.4f %10.4f %10.4f %3d %3d %3d %3d %7s %5d %5d\n'
    for star in stars:
        _out.write(fmt % (star[0], star[1], star[2], star[3], star[4], star[5],
                          star[6], star[7], star[8], star[9], star[10], star[11],
                          star[12], star[13]))
    _out.close()

    # stars.write(filename + '.shift', type='ascii', overwrite=True,
    # 	        formats=formatDict, quotechar=' ', names=None)
		

    return

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

def make_brite_multi_filter(matchup_files, trimMags):
    """
    Take an input list of MATCHUP files (assumes they have the same stars, and the
    same length) and trim out only the bright stars. The resulting output file contains
    the X and Y position (from the first file) and the list of all magnitudes for each star.

    trimMags is a list of brightness criteria for each of the matchup files. Any star
    that satisfies this criteria in any one of the filters will be added to the global
    bright list.
    """
    if not hasattr(matchup_files, '__iter__'):
        matchup_files = [matchup_files]

    if len(trimMags) != len(matchup_files):
        print 'Error calling make_brite_multi_filter'

    starlists = []

    # Read in the matchup files and keep a record of those
    # stars detected in all the epochs.
    inAll = None
    for mfile in matchup_files:
        table = read_matchup(mfile)
        starlists.append( table )

        if inAll == None:
            inAll = np.ones(len(starlists[0]), dtype=bool)
        inAll[ np.where(table.m == 0) ] = False

    # Trim down the lists to just those stars in all epochs.
    foo = 'Trimming {0} of {1} stars that are not in all epochs'
    print(foo.format(inAll.sum(), len(inAll)))
    for ff in range(len(starlists)):
        starlists[ff] = starlists[ff].where(inAll)

    # Trim down to just the brightest stars.
    keep = np.zeros(len(starlists[0]), dtype=bool)
    for ff in range(len(starlists)):
        idx = np.where(starlists[ff].m < trimMags[ff])[0]

        keep[idx] = True

    # Drop brite stars that aren't detected in every list. These are
    # probably fakes anyhow.
    idx = np.where(keep == True)[0]
    print len(idx)

    brite = open('BRITE.XYM', 'w')
    
    for ii in idx:
        brite.write('%10.4f  %10.4f ' % (starlists[0].x[ii], starlists[0].y[ii]))
        for mm in range(len(starlists)):
            brite.write('  %10.2f' % (starlists[mm].m[ii]))
        brite.write('\n')
    
    brite.close()

    

def combine_brite_ks2(ks2Root, briteFile, matchupFiles):
    """
    Read in the ks2 output, a brite star list, and matchup files.
    Combine all of these into a single output starlist.

    The matchup files should be listed in the exact same filter order
    as is processed in KS2 and in the brite file.
    """

    #####
    # Read in files
    #####
    print 'Reading in files'
    brite = atpy.Table(briteFile, type='ascii')
    brite.rename_column('col1', 'x')
    brite.rename_column('col2', 'y')

    nfilt = brite.shape[1] - 2
    if nfilt != len(matchupFiles):
        print 'Matchup file list does not match filter set in ks2 output.'
        print '     Number of matchup files  = %d' % len(matchupFiles)
        print '     Number of filters in ks2 = %d' % nfile
        return

    ks2_list = atpy.Table(ks2Root + '.XYVIQ1.fits')

    match_list = []

    for nn in range(nfilt):
        tmp = read_matchup(matchupFiles[nn])

        match_list.append(tmp)
    
    #####
    # Filter down the matchup files to only the brite stars.
    # We assume that the coordinates came from the first file (as is true
    # in make_brite().
    #####
    print 'Filtering matchup files to brite sources.'
    idxMatchup = np.zeros(len(brite), dtype=int)
    for bb in range(len(brite)):
        # Find the corresponding star in the matchup files.
        dr = np.hypot(brite.x[bb] - match_list[0].x,
                      brite.y[bb] - match_list[0].y)

        idx = dr.argmin()

        if dr[idx] > 1:
            print 'Problem finding match for brite star: '
            print '   [%.4f, %.4f]' % (brite.x[bb], brite.y[bb])
            print 'The closest match in %s: ' % matchupFiles[0]
            print '   [%.4f, %.4f]' % (match_list[0].x[idx], match_list[0].y[idx])

        idxMatchup[bb] = idx

    for ff in range(nfilt):
        match_list[ff] = match_list[ff].rows(idxMatchup)

    #####
    # Create output table. It is just the original ks2 table, with some columns
    # removed and the brite stars added. Lets also add a column to keep track of the
    # source of the astrometry/photometry (which code).
    #####
    # Remove the f_other columns.
    columnNames = ks2_list.columns.keys

    columnsToDelete = []
    for col in columnNames:
        if col.startswith('f_other'):
            columnsToDelete.append(col)
    ks2_list.remove_columns(columnsToDelete)


    # Combine the brite stars with the ks2 stars.
    all_X = np.array([mm.x for mm in match_list])
    all_Y = np.array([mm.y for mm in match_list])
    avg_X = all_X.mean(axis=0)
    avg_Y = all_Y.mean(axis=0)

    stars = atpy.Table()
    stars.add_column('x', np.append(ks2_list.x, avg_X))
    stars.add_column('y', np.append(ks2_list.y, avg_Y))
    stars.add_column('name', np.append(ks2_list.name, match_list[0].name))

    for ff in range(nfilt):
        suffix = '_%d' % ff
        mm = match_list[ff]

        flux = 10.0**(-mm.m / 2.5)
        fluxErr = mm.me * (math.log(10.) / 2.5) * flux

        stars.add_column('x'+suffix, np.append(ks2_list['x'+suffix], mm.x))
        stars.add_column('y'+suffix, np.append(ks2_list['y'+suffix], mm.y))
        stars.add_column('xe'+suffix, np.append(ks2_list['xe'+suffix], mm.xe))
        stars.add_column('ye'+suffix, np.append(ks2_list['ye'+suffix], mm.ye))
        stars.add_column('f'+suffix, np.append(ks2_list['f'+suffix], flux))
        stars.add_column('m'+suffix, np.append(ks2_list['m'+suffix], mm.m))
        stars.add_column('fe'+suffix, np.append(ks2_list['fe'+suffix], fluxErr))
        stars.add_column('me'+suffix, np.append(ks2_list['me'+suffix], mm.me))
        # Quality flag: 2.0 means BRITE MATCHUP star. Usually ks2 produces 0 - 1.
        stars.add_column('q'+suffix, np.append(ks2_list['q'+suffix], np.ones(len(mm.x))*2.0))
        stars.add_column('Nfound'+suffix, np.append(ks2_list['Nfound'+suffix], mm.N_fnd))
        stars.add_column('Nused'+suffix, np.append(ks2_list['Nsurvive'+suffix], mm.N_mwell))

    # Sort by brightness in the first filter
    stars.sort('f_0')

    stars.write(ks2Root + '_BRITE.XYVIQ1.fits')

    return stars

    
def convert_matchup_to_align(matchupFile, filter, year, topStars=''):
    """
    Read in a matchup file and print out a file suitable for use in align.

    INPUTS:
    matchupFile -- The matchup.XYMEEE file.
    filter -- The name of the filter for fetching the zeropoint.
    date -- The date to print in the align file.
    topStars -- a list of dictionaries, one for each "named" star that should
        be moved to the top of the starlist (in the order of topStars. Each
        dictionary should have the format:
        {'name': 'wd1_1',
         'x': 3001.3,
         'y': 240.1
         }
    """
    t = read_matchup(matchupFile)

    # Rough photometric calibration, align doesn't do well with instrumental
    # magnitudes (anything < 0).
    t.m += photometry.ZP[filter]

    # Find the "named" sources and shift them to the top
    idxInit = np.ones(len(topStars)) * -1
    for ss in range(len(topStars)):
        dr = np.hypot(t.x - topStars[ss]['x'],
                      t.y - topStars[ss]['y'])
        drmin = dr.argmin()

        # If this star is closer than 2 pixels, then call it a match.
        if dr[drmin] < 2.0:
            idxInit[ss] = drmin

    topIdx = idxInit[idxInit >= 0]
    topNames = np.array([topStar['name'] for topStar in topStars])
    topNames = topNames[idxInit >= 0]

    outName = os.path.basename(matchupFile) + '.lis'
    _out = open(outName, 'w')

    # Write out the top stars first
    for ss in range(len(topIdx)):
        dd = topIdx[ss]
        _out.write('%-13s  %7.3f  %9.4f  %10.4f %10.4f  %7.4f %7.4f  %7.3f  1 1 1\n' %
                   (topNames[ss], t.m[dd], year, t.x[dd], t.y[dd],
                    t.xe[dd], t.ye[dd], t.me[dd]))
            
    rejectCnt = 0
    for dd in range(len(t)):
        if (t.xe[dd] > 5) and (t.ye[dd] > 5) and (t.me[dd] > 5):
            rejectCnt += 1
            continue

        # Make sure this wasn't one of the top stars.
        if dd in topIdx:
            continue

        newName = 'star' + t.name[dd]
        _out.write('%-13s  %7.3f  %9.4f  %10.4f %10.4f  %7.4f %7.4f  %7.3f  1 1 1\n' %
                   (newName, t.m[dd], year, t.x[dd], t.y[dd],
                    t.xe[dd], t.ye[dd], t.me[dd]))

    _out.close()
    print('Rejected {0} sources in {1}'.format(rejectCnt, outName))

def convert_ks2_to_align(ks2catalog, filtIndex, filter, year, topStars=''):
    """
    Read in a matchup file and print out a file suitable for use in align.

    INPUTS:
    ks2catalog -- The output catalog from a KS2 run a produced by process_ks2_output()
    filtIndex -- the filter index (1 based) to use from the KS2 output (e.g. x_1)
    filter -- The name of the filter for fetching the zeropoint.
    date -- The date to print in the align file.
    topStars -- a list of dictionaries, one for each "named" star that should
        be moved to the top of the starlist (in the order of topStars. Each
        dictionary should have the format:
        {'name': 'wd1_1',
         'x': 3001.3,
         'y': 240.1
         }
    """
    t = atpy.Table(ks2catalog)

    suffix = '_{0}'.format(filtIndex)
    xcol = 'x' + suffix
    ycol = 'y' + suffix
    mcol = 'm' + suffix
    xecol = 'xe' + suffix
    yecol = 'ye' + suffix
    mecol = 'me' + suffix

    # Rough photometric calibration, align doesn't do well with instrumental
    # magnitudes (anything < 0).
    t[mcol] += photometry.ZP[filter]

    # Find the "named" sources and shift them to the top
    idxInit = np.ones(len(topStars)) * -1
    for ss in range(len(topStars)):
        dr = np.hypot(t[xcol] - topStars[ss]['x'],
                      t[ycol] - topStars[ss]['y'])
        drmin = dr.argmin()

        # If this star is closer than 2 pixels, then call it a match.
        if dr[drmin] < 2.0:
            idxInit[ss] = drmin

    topIdx = idxInit[idxInit >= 0]
    topNames = np.array([topStar['name'] for topStar in topStars])
    topNames = topNames[idxInit >= 0]

    rootFileName = os.path.splitext(os.path.basename(ks2catalog))[0]
    outName = '{0}_{1}.lis'.format(rootFileName, filtIndex)
    _out = open(outName, 'w')

    # Write out the top stars first
    for ss in range(len(topIdx)):
        dd = topIdx[ss]
        _out.write('%-13s  %7.3f  %9.4f  %10.4f %10.4f  %7.4f %7.4f  %7.3f  1 1 1\n' %
                   (topNames[ss], t[mcol][dd], year, t[xcol][dd], t[ycol][dd],
                    t[xecol][dd], t[yecol][dd], t[mecol][dd]))
            
    rejectCnt = 0
    for dd in range(len(t)):
        if ((t[xecol][dd] > 4.9) or (t[yecol][dd] > 4.9) or (t[mecol][dd] > 5) or
            (np.isnan(t[mcol][dd])) or (t[mcol][dd] > 100)):
            rejectCnt += 1
            continue

        # Make sure this wasn't one of the top stars.
        if dd in topIdx:
            continue

        newName = 'star' + t.name[dd]
        _out.write('%-13s  %7.3f  %9.4f  %10.4f %10.4f  %7.4f %7.4f  %7.3f  1 1 1\n' %
                   (newName, t[mcol][dd], year, t[xcol][dd], t[ycol][dd],
                    t[xecol][dd], t[yecol][dd], t[mecol][dd]))

    _out.close()
    print('Rejected {0} sources in {1}'.format(rejectCnt, outName))


def split_nimfo(filename):

    starlists = {}

    with open(filename) as f:
        for line in f:
            if line[0] == '#':
                continue

            fields = line.split()

            xraw = float(fields[5])
            yraw = float(fields[6])
            mraw = float(fields[2])
            name = fields[15]

            if mraw == 0:
                continue

            imageName = fields[17]
            imageIdx = int(imageName[1:])

            newStr = '{x:9.3f} {y:9.3f} {m:9.3f} {name}\n'
            newLine = newStr.format(x=xraw, y=yraw, m=mraw, name=name)

            try:
                starlists[imageIdx].write(newLine)
            except KeyError:
                starlists[imageIdx] = open('stars_{0:03d}.xym'.format(imageIdx), 'w')


    for key in starlists:
        starlists[key].close()


def read_nimfo(filename, saveToFits=True):
    """
    DON'T USE!!! These NIMFO files are too large to read in via atpy.

    Read the NIMFO file which contains position/flux information for
    every exposure and every star. Save to FITS file. 
    """
    tab = atpy.Table(filename, type='ascii')
    print 'Finished reading table'

    tab.rename_column('col1', 'x')
    tab.rename_column('col2', 'y')
    tab.rename_column('col3', 'm')
    tab.rename_column('col6', 'xraw')
    tab.rename_column('col7', 'yraw')
    tab.rename_column('col10', 'f0')
    tab.rename_column('col11', 'f0err')
    tab.rename_column('col12', 'f1')
    tab.rename_column('col13', 'f1err')
    tab.rename_column('col14', 'f2')
    tab.rename_column('col15', 'f2err')
    tab.rename_column('col16', 'name')

    tab.remove_columns(['col4', 'col5', 'col8', 'col9', 'col17', 'col18', 'col19'])
    tab.table_name = ''
    
    if saveToFits:
        tab.write(filename + '.fits', overwrite=True)

    return tab

# def read_nimfo(filename, saveToFits=True):
#     """
#     Read the NIMFO file which contains position/flux information for
#     every exposure and every star. Save to FITS file. 
#     """
#     _in = open(filename, 'r')

#     x = []
#     y = []
#     f1 = []
#     f2 = []
#     xe = []
#     ye = []
#     f1e = []
#     f2e = []
#     name = []
#     cnt = []

#     ii = 0
#     starName = None

#     for line in _in.readlines():
#         if line.startswith('#'):
#             continue

#         fields = line.split()

#         starNameNew = fields[15]

#         if (starNameNew != starName) and (starName != None):
#             # Finish up the previous star
#             xbar, xstd = statsIter.mean_std_clip(x_i)
#             ybar, ystd = statsIter.mean_std_clip(y_i)
#             f1bar, f1std = statsIter.mean_std_clip(f1_i)
#             f2bar, f2std = statsIter.mean_std_clip(f2_i)

#             x.append(xbar)
#             y.append(ybar)
#             f1.append(f1bar)
#             f2.append(f2bar)

#             xe.append(xstd)
#             ye.append(ystd)
#             f1.append(f1std)
#             f2.append(f2std)

#             name.append(starName)
#             cnt.append(len(x_i))

#             # Create arrays for new star
#             x_i = []
#             y_i = []
#             f1_i = []
#             f2_i = []
#             starName = starNameNew

#         x_i.append(fields[0])
#         y_i.append(fields[1])
#         f1_i.append(fields[14])
#         f2_i.append(fields[16])


#     # Write output
#     outfile = filename.replace('FIND_NIMFO', 'NIFMO')
#     _out = open(outfile, 'w')

#     _out.iwrte('%12s  %12s  %8s  %7s  %7s  %7s  %11s  %10s  %11s  %10s  %3s  %-13s\n' %
#                ('x', 'y', 'm', 'xerr', 'yerr', 'merr', 'z1', 'z1err', 'z2', 'z2err', 'cnt', 'name'))

#     m = -2.5 * np.log10(z1)
#     #me = 
    
#     for ii in range(len(x)):
#         _out.write('%12.4f  %12.4f  %8.4f  %7.4f  %7.4f  %7.4f  ' %
#                    (x[ii], y[ii], ))


def ks2_astrometry_by_filter(inputFile, pass1CameraCode):
    """
    After having run split_nimfo(), this code does everything to produce
    astrometrically registered starlists from the KS2 output starlists.

    Pass in a INPUT.KS2 file after having run split_nimfo(). 
    """
    _in = open(inputFile, 'r')
    inLines = _in.readlines()
    _in.close()

    img_lists = {}
    filterName = None

    for line in inLines:
        # New Filter
        if line.startswith('FILTER'):
            filterFields = line.split()
            filterName = filterFields[-1].strip('"')
            img_lists[filterName] = []

            # make directory
            if not os.path.isdir(filterName):
                os.makedirs(filterName)

            barFile = filterName + '/IN.xym2bar'
            print 'Opening barFile' + barFile
            _bar = open(barFile, 'w')
            

        # Check if this is a line specifiying an image.
        if 'PIX=' in line:
            # Image Index
            fields = line.split()
            ks2Index = int(fields[0])
            
            # Process MAT file
            srchString = 'MAT="'
            matStart = line.find(srchString) + len(srchString)
            matStop = line.find('"', matStart)
            matFile = line[matStart:matStop]
            matFileBase, matFileExt = os.path.splitext(os.path.basename(matFile))
            matFileExt = matFileExt.strip('.')

            # Construct stars_###.xym file
            oldStarList = 'stars_{0:03d}.xym'.format(ks2Index)
            newStarList = 'stars_ks{0:03d}_mat{1}.xym'.format(ks2Index, matFileExt)

            # copy files
            shutil.copyfile(matFile, filterName + '/' + os.path.basename(matFile))
            shutil.copyfile(oldStarList, filterName + '/' + newStarList)

            barStr = '{0} "{1}" {2}\n'
            _bar.write(barStr.format(matFileExt, newStarList, pass1CameraCode))
                       


def make_matchup_positive(matchupFile, padding=10):
    tab = read_matchup(matchupFile)

    idx = np.where(tab.m != 0)[0]

    shiftX = tab.x[idx].min() - padding  # add N pixel padding
    shiftY = tab.y[idx].min() - padding  # add N pixel padding

    if shiftX <= 0:
        tab.x -= shiftX
    if shiftY <= 0:
        tab.y -= shiftY

    out = open(matchupFile + '.positive', 'w')
    for ii in range(len(tab)):
        out.write('%10.4f  %10.4f  %10.2f  %10.4f  %10.4f  %10.2f\n' %
                  (tab.x[ii], tab.y[ii], tab.m[ii],
                   tab.xe[ii], tab.ye[ii], tab.me[ii]))
    out.close()
        


def match_xym_starlists_plot(t1, t2, magHi1=0, magHi2=0, magLo1=-99, magLo2=-99):
    if type(t1) != atpy.basetable.Table:
        t1 = atpy.Table(t1, type='ascii')
        t1.rename_column('col1', 'x')
        t1.rename_column('col2', 'y')
        t1.rename_column('col3', 'm')

    if type(t2) != atpy.basetable.Table:
        t2 = atpy.Table(t2, type='ascii')
        t2.rename_column('col1', 'x')
        t2.rename_column('col2', 'y')
        t2.rename_column('col3', 'm')

    g1 = t1.where((t1.m > magLo1) & (t1.m < magHi1))
    g2 = t2.where((t2.m > magLo2) & (t2.m < magHi2))

    sz1 = 1.4**np.abs(g1.m)
    sz2 = 1.4**np.abs(g2.m)
    
    fig1 = py.figure(1)
    fig1.clf()
    ax1 = fig1.gca()
    ax1.scatter(g1.x, g1.y, s=sz1)
    annotations1 = [str(star) for star in g1]
    cat1 = CatalogFinder.CatalogFinder(g1.x, g1.y, annotations1, axis=ax1)
    fig1.canvas.callbacks.connect('button_press_event', cat1)

    fig2 = py.figure(2)
    fig2.clf()
    ax2 = fig2.gca()
    ax2.scatter(g2.x, g2.y, s=sz2)
    annotations2 = [str(star) for star in g2]
    cat2 = CatalogFinder.CatalogFinder(g2.x, g2.y, annotations2, axis=ax2)
    fig2.canvas.callbacks.connect('button_press_event', cat2)

    return t1, t2
    
    
