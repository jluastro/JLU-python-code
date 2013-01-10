import numpy as np
import pylab as py
import atpy
from jlu.hst import photometry
import math
from jlu.util import statsIter

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
    nfilt = (ncols - 3) / 6

    tab.add_keyword('Nfilt', nfilt)

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

    tab.add_keyword('Nfilt', nfilt)

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

    Nfilt = tab.col8.max()

    if filtNames != None and Nfilt != len(filtNames):
        print 'Please specify %d filter names to return Vega magnitudes.' % \
            (Nfilt)
        return

    # Extract some temporary information and delete it from the table.
    filterNum = tab.col10
    tab.remove_columns(['col10'])

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
            
    

def process_ks2_output(ks2Root, Nfilt):
    xyviq1 = read_xyviq1(ks2Root + '.XYVIQ1', saveToFits=False)
    
    # Extract some temporary information and delete it from the table.
    filterNum = tab.col14
    tab.remove_columns(['col14', 'col15'])

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
        final.table_name = ''

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

    
# def read_nimfo(filename):
#     """
#     Read the NIMFO file which contains position/flux information for every exposure and every star.
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

#         starNameNew = fields[18]

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
#                    %11.1f  %10.1f  %11.1f  %10.1f  %3d  %-13s\n' %
#                ())
