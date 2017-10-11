import os, sys
import math, copy
from astropy.table import Table # asciidata
import numpy as np
from jlu.gc.gcwork import objects
#from pysqlite2 import dbapi2 as sqlite
from sqlite3 import dbapi2 as sqlite
import pdb

tablesDir = '/u/ghezgroup/data/gc/source_list/'
database = '/u/ghezgroup/data/gc/database/stars.sqlite'

class StarTable(object):
    def fixNames(self):
        for i in range(len(self.ourName)):
            self.ourName[i] = self.ourName[i].strip()
            self.name[i] = self.name[i].strip()

            if self.ourName[i] == '-':
                self.ourName[i] = ''

    def take(self, indices):
        """
        Loop through all the numpy arrays on this StarTable
        object and pull out only the items with the specified
        indices. This modifies the lists on this object.

        This code assumes there is always a 'name' list on the
        StarTable object.
        """
        origNameCnt = len(self.name)

        for dd in dir(self):
            obj = getattr(self, dd)
            objType = type(obj)

            # Check that this is a listable object (numpy or list)
            if ((objType is 'numpy.ndarray' or objType is 'list') and
                (len(obj) is origNameCnt)):
                # Trim this numpy list.
                obj = obj[indices]


class StarfinderList(StarTable):
    def __init__(self, listFile, hasErrors=False):
        self.file = listFile
        self.hasErrors = hasErrors

        # We can create a list file from scratch with nothing in it.
        if listFile == None:
            self.name = np.array([], dtype=str)
            self.mag = np.array([], dtype=float)
            self.epoch = np.array([], dtype=float)
            self.x = np.array([], dtype=float)
            self.y = np.array([], dtype=float)

            if self.hasErrors:
                self.xerr = np.array([], dtype=float)
                self.yerr = np.array([], dtype=float)

            self.snr = np.array([], dtype=float)
            self.corr = np.array([], dtype=float)
            self.nframes = np.array([], dtype=float)
            self.counts = np.array([], dtype=float)

        else:
            tab = Table.read(self.file)

            self.name = tab[0].tonumpy()
            for rr in range(len(self.name)):
                self.name[rr] = self.name[rr].strip()
            self.mag = tab[1].tonumpy()
            self.epoch = tab[2].tonumpy()
            self.x = tab[3].tonumpy()
            self.y = tab[4].tonumpy()

            tabIdx = 5
            if self.hasErrors == True:
                self.xerr = tab[tabIdx+0].tonumpy()
                self.yerr = tab[tabIdx+1].tonumpy()
                tabIdx += 2

            self.snr = tab[tabIdx+0].tonumpy()
            self.corr = tab[tabIdx+1].tonumpy()
            self.nframes = tab[tabIdx+2].tonumpy()
            self.counts = tab[tabIdx+3].tonumpy()

    def append(self, name, mag, x, y, epoch=None, xerr=0, yerr=0,
               snr=0, corr=1, nframes=1, counts=0):
        if epoch == None:
            epoch = self.epoch[0]

        self.name = np.append(self.name, name)
        self.mag = np.append(self.mag, mag)
        self.epoch = np.append(self.epoch, epoch)
        self.x = np.append(self.x, x)
        self.y = np.append(self.y, y)

        if self.hasErrors:
            self.xerr = np.append(self.xerr, xerr)
            self.yerr = np.append(self.yerr, yerr)

        self.snr = np.append(self.snr, snr)
        self.corr = np.append(self.corr, corr)
        self.nframes = np.append(self.nframes, nframes)
        self.counts = np.append(self.counts, counts)


    def saveToFile(self, outfile):
        _out = open(outfile, 'w')

        for ii in range(len(self.x)):
            _out.write('%-13s  ' % self.name[ii])
            _out.write('%6.3f  ' % self.mag[ii])
            _out.write('%9.4f  ' % self.epoch[ii])
            _out.write('%11.5f  %11.5f   ' % (self.x[ii], self.y[ii]))

            if self.hasErrors == True:
                _out.write('%8.5f  %8.5f  ' % (self.xerr[ii], self.yerr[ii]))

            _out.write('%15.4f  ' % self.snr[ii])
            _out.write('%4.2f  ' % self.corr[ii])
            _out.write('%5d  ' % self.nframes[ii])
            _out.write('%15.3f\n' % self.counts[ii])

        _out.close()

class Genzel2000(StarTable):
    def __init__(self):
        self.file = tablesDir + 'ucla_genzel2000.dat'
        tab = Table.read(self.file)

        self.ourName = [tab[0][d].strip() for d in range(tab.nrows)]
        self.name = [tab[1][d].strip() for d in range(tab.nrows)]
        self.r = tab[2].tonumpy()
        self.x = tab[3].tonumpy()
        self.y = tab[4].tonumpy()
        self.vx1 = tab[5].tonumpy()
        self.vx1err = tab[6].tonumpy()
        self.vy1 = tab[7].tonumpy()
        self.vy1err = tab[8].tonumpy()
        self.vx2 = tab[9].tonumpy()
        self.vx2err = tab[10].tonumpy()
        self.vy2 = tab[11].tonumpy()
        self.vy2err = tab[12].tonumpy()
        self.vx = tab[13].tonumpy()
        self.vxerr = tab[14].tonumpy()
        self.vy = tab[15].tonumpy()
        self.vyerr = tab[16].tonumpy()
        self.vz = tab[17].tonumpy()
        self.vzerr = tab[18].tonumpy()

        self.fixNames()

class Paumard2001(StarTable):
    def __init__(self):
        self.file = tablesDir + 'ucla_paumard2001.dat'
        tab = Table.read(self.file)

        self.ourName = [tab[0][d].strip() for d in range(tab.nrows)]
        self.name = [tab[1][d].strip() for d in range(tab.nrows)]
        self.x = tab[2].tonumpy()
        self.y = tab[3].tonumpy()
        self.vz = tab[4].tonumpy()
        self.vzerr = tab[5].tonumpy()

        x = self.x
        y = self.y
        self.r = np.sqrt(x**2 + y**2)

        self.fixNames()

def loadColumnFromDB(dbColumn, dbTable, dbCursor):
    sqlCommand = 'select ' + dbColumn + ' from ' + dbTable

    dbCursor.execute(sqlCommand)

    results = []
    for row in dbCursor:
        if type(row[0]) is 'str':
            results.append(row[0].strip())
        else:
            results.append(row[0])

    # Convert to numpy
    results = np.array(results)

    return results


class UCLAstars(StarTable):
    def __init__(self):
        # Create a connection to the database file
        connection = sqlite.connect(database)

        # Create a cursor object
        cur = connection.cursor()

        self.ourName = loadColumnFromDB('name', 'stars', cur)
        self.Kmag = loadColumnFromDB('kp', 'stars', cur)
        self.t0_astrom = loadColumnFromDB('ddate', 'stars', cur)
        self.x = loadColumnFromDB('x', 'stars', cur)
        self.xerr = loadColumnFromDB('x_err', 'stars', cur)
        self.y = loadColumnFromDB('y', 'stars', cur)
        self.yerr = loadColumnFromDB('y_err', 'stars', cur)
        self.r2d = loadColumnFromDB('r2d', 'stars', cur)
        self.vx = loadColumnFromDB('vx', 'stars', cur)
        self.vxerr = loadColumnFromDB('vx_err', 'stars', cur)
        self.vy = loadColumnFromDB('vy', 'stars', cur)
        self.vyerr = loadColumnFromDB('vy_err', 'stars', cur)
        self.vz = loadColumnFromDB('vz', 'stars', cur)
        self.vzerr = loadColumnFromDB('vz_err', 'stars', cur)
        self.t0_spectra = loadColumnFromDB('vz_ddate', 'stars', cur)


class Bartko2009(StarTable):
    def __init__(self):
        # Create a connection to the database file
        connection = sqlite.connect(database)

        # Create a cursor object
        cur = connection.cursor()

        self.ourName = loadColumnFromDB('ucla_name', 'bartko2009', cur)
        self.Kmag = loadColumnFromDB('Kmag', 'bartko2009', cur)
        self.t0_astrom = loadColumnFromDB('t0_astrometry', 'bartko2009', cur)
        self.x = loadColumnFromDB('x', 'bartko2009', cur)
        self.y = loadColumnFromDB('y', 'bartko2009', cur)
        self.r2d = loadColumnFromDB('r', 'bartko2009', cur)
        self.vx = loadColumnFromDB('vx', 'bartko2009', cur)
        self.vxerr = loadColumnFromDB('vx_err', 'bartko2009', cur)
        self.vy = loadColumnFromDB('vy', 'bartko2009', cur)
        self.vyerr = loadColumnFromDB('vy_err', 'bartko2009', cur)
        self.t0_spectra = loadColumnFromDB('t0_spectra', 'bartko2009', cur)
        self.vz = loadColumnFromDB('vz', 'bartko2009', cur)
        self.vzerr = loadColumnFromDB('vz_err', 'bartko2009', cur)


class Paumard2006(StarTable):
    def __init__(self):
        # Create a connection to the database file
        connection = sqlite.connect(database)

        # Create a cursor object
        cur = connection.cursor()

        self.ourName = loadColumnFromDB('ucla', 'paumard2006', cur)
        self.name = loadColumnFromDB('name', 'paumard2006', cur)
        self.r2d = loadColumnFromDB('r2d', 'paumard2006', cur)
        self.x = loadColumnFromDB('x', 'paumard2006', cur)
        self.y = loadColumnFromDB('y', 'paumard2006', cur)
        self.z = loadColumnFromDB('z', 'paumard2006', cur)
        self.zerr = loadColumnFromDB('z_err', 'paumard2006', cur)
        self.Kmag = loadColumnFromDB('Kmag', 'paumard2006', cur)
        self.vx = loadColumnFromDB('vx', 'paumard2006', cur)
        self.vxerr = loadColumnFromDB('vx_err', 'paumard2006', cur)
        self.vy = loadColumnFromDB('vy', 'paumard2006', cur)
        self.vyerr = loadColumnFromDB('vy_err', 'paumard2006', cur)
        self.vz = loadColumnFromDB('vz', 'paumard2006', cur)
        self.vzerr = loadColumnFromDB('vz_err', 'paumard2006', cur)
        self.jz = loadColumnFromDB('jz', 'paumard2006', cur)
        self.jzerr = loadColumnFromDB('jz_err', 'paumard2006', cur)
        self.e = loadColumnFromDB('e', 'paumard2006', cur)
        self.eerr = loadColumnFromDB('e_err', 'paumard2006', cur)
        self.type = loadColumnFromDB('type', 'paumard2006', cur)
        self.quality = loadColumnFromDB('quality', 'paumard2006', cur)
        self.MK = loadColumnFromDB('MK', 'paumard2006', cur)
        self.MKerr = loadColumnFromDB('MK_err', 'paumard2006', cur)
        self.t0_astrom = loadColumnFromDB('t0_astrometry', 'paumard2006', cur)
        self.t0_spectra = loadColumnFromDB('t0_spectra', 'paumard2006', cur)

        self.fixNames()

    def matchNames(self, labelFile=tablesDir+'label.dat'):
        cc = objects.Constants()

        # Load up our label.dat file
        labels = Labels(labelFile)

        # Convert Paumard Velocities to asec/yr.
        vxPaum = self.vx / cc.asy_to_kms
        vyPaum = self.vy / cc.asy_to_kms

        # Epoch to match at:
        t = 2008.0
        t0Paum = 2005.0  # just a guess

        xPaum = self.x + vxPaum * (t - t0Paum)
        yPaum = self.y + vyPaum * (t - t0Paum)

        xOurs = labels.x + (labels.vx * (t - labels.t0) / 1.0e3)
        yOurs = labels.y + (labels.vy * (t - labels.t0) / 1.0e3)

        for ii in range(len(xPaum)):
#             if self.ourName[ii] is not '-':
#                 continue

            dr = np.hypot(xOurs - xPaum[ii], yOurs - yPaum[ii])
            dm = labels.mag - self.Kmag[ii]

            # Find the closest source
            rdx = dr.argsort()[0]

            # Find thoses sources within 0.1"
            cdx = np.where(dr < 0.1)[0]

            print('')
            print('Match %10s at [%5.2f, %5.2f] and mag = %5.2f (ourName = %s)' % \
                (self.name[ii], xPaum[ii], yPaum[ii], self.Kmag[ii],
                 self.ourName[ii]))
            print('   Closest Star:')
            print('      %10s at [%5.2f, %5.2f] and mag = %5.2f' % \
                (labels.name[rdx], xOurs[rdx], yOurs[rdx], labels.mag[rdx]))
            print('   Stars within 0.1"')
            for kk in cdx:
                print('      %10s at [%5.2f, %5.2f] and mag = %5.2f' % \
                    (labels.name[kk], xOurs[kk], yOurs[kk], labels.mag[kk]))

class Ott2003(StarTable):
    def __init__(self):
        self.file = tablesDir + 'ucla_ott2003.dat'
        tab = Table.read(self.file)

        self.ourName = [tab[0][d].strip() for d in range(tab.nrows)]
        self.id = [tab[1][d].strip() for d in range(tab.nrows)]
        self.name = [tab[2][d].strip() for d in range(tab.nrows)]
        self.r = tab[3].tonumpy()
        self.x = tab[4].tonumpy()
        self.y = tab[5].tonumpy()
        self.xerr = tab[6].tonumpy()
        self.yerr = tab[7].tonumpy()
        self.mag = tab[8].tonumpy()
        self.magerr = tab[9].tonumpy()
        self.mHK = tab[10].tonumpy()
        self.mCO = tab[11].tonumpy()
        self.vx = tab[12].tonumpy()
        self.vy = tab[13].tonumpy()
        self.vz = tab[14].tonumpy()
        self.vxerr = tab[15].tonumpy()
        self.vyerr = tab[16].tonumpy()
        self.vzerr = tab[17].tonumpy()
        self.type = tab[18].tonumpy()

        self.fixNames()

class Tanner2006(StarTable):
    def __init__(self):
        self.file = tablesDir + 'ucla_tanner2006.dat'
        tab = Table.read(self.file)

        self.ourName = [tab[0][d].strip() for d in range(tab.nrows)]
        self.name = [tab[1][d].strip() for d in range(tab.nrows)]
        self.x = tab[2].tonumpy()
        self.y = tab[3].tonumpy()
        self.xerr = tab[4].tonumpy()
        self.yerr = tab[5].tonumpy()
        self.vx = tab[6].tonumpy()
        self.vxerr = tab[7].tonumpy()
        self.vy = tab[8].tonumpy()
        self.vyerr = tab[9].tonumpy()
        self.vz = tab[10].tonumpy()
        self.vzerr = tab[11].tonumpy()

        self.fixNames()

def youngStarNames(datfile='/u/ghezgroup/data/gc/source_list/young.dat'):
    """Load list of young stars.

    Retrieves the list from /u/ghezgroup/data/gc/source_list/young.dat
    and returns a list of the names.
    """
    f_yng = open(datfile, 'r')

    names = []
    for line in f_yng:
        _yng = line.split()

        names.append(_yng[0])

    names.sort()
    return names

def lateStarNames(datfile='/u/ghezgroup/data/gc/source_list/late.dat'):
    """Load list of late-type stars.

    Retrieves the list from /u/ghezgroup/data/gc/source_list/late.dat
    and returns a list of the names.
    """
    f_yng = open(datfile, 'r')

    names = []
    for line in f_yng:
        _yng = line.split()

        names.append(_yng[0])

    names.sort()
    return names

class Orbits(StarTable):
    """
    Loads up an orbits.dat file. File is assumed to reside in
    /u/ghezgroup/data/gc/source_list/.

    Optional Input:
    orbitFile: Default is 'orbits.dat'
    """
    def __init__(self, orbitFile='orbits.dat'):
        self.file = tablesDir + orbitFile
        tab = Table.read(self.file)

        self.ourName = [tab[0][d].strip() for d in range(tab.nrows)]
        self.name = [tab[0][d].strip() for d in range(tab.nrows)]
        self.p = tab[1].tonumpy()
        self.a = tab[2].tonumpy()
        self.t0 = tab[3].tonumpy()
        self.e = tab[4].tonumpy()
        self.i = tab[5].tonumpy()
        self.o = tab[6].tonumpy()
        self.w = tab[7].tonumpy()
        self.searchRadius = tab[8].tonumpy()

class Labels(StarTable):
    """
    Loads up a label.dat file. File is assumed to reside in
    /u/ghezgroup/data/gc/source_list/.

    Optional Input:
    labelFile: Default is 'label.dat'
    """
    def __init__(self, labelFile=tablesDir+'label.dat'):
        self.file = labelFile

        if labelFile != None:
            tab = Table.read(self.file)

            self.headerString = str(tab.header)

            self.ourName = tab[0].tonumpy()
            self.name = tab[0].tonumpy()
            self.mag = tab[1].tonumpy()
            self.x = tab[2].tonumpy()
            self.y = tab[3].tonumpy()
            self.xerr = tab[4].tonumpy()
            self.yerr = tab[5].tonumpy()
            self.vx = tab[6].tonumpy()
            self.vy = tab[7].tonumpy()
            self.vxerr = tab[8].tonumpy()
            self.vyerr = tab[9].tonumpy()
            self.t0 = tab[10].tonumpy()
            self.useToAlign = tab[11].tonumpy()
            self.r = tab[12].tonumpy()
        else:
            self.headerString = ''
            self.ourName = np.array([], dtype=str)
            self.name = np.array([], dtype=str)
            self.mag = np.array([], dtype=float)
            self.x = np.array([], dtype=float)
            self.y = np.array([], dtype=float)
            self.xerr = np.array([], dtype=float)
            self.yerr = np.array([], dtype=float)
            self.vx = np.array([], dtype=float)
            self.vy = np.array([], dtype=float)
            self.vxerr = np.array([], dtype=float)
            self.vyerr = np.array([], dtype=float)
            self.t0 = np.array([], dtype=float)
            self.useToAlign = np.array([], dtype=str)
            self.r = np.array([], dtype=float)

    def saveToFile(self, outfile):
        _out = open(outfile, 'w')

        _out.write(self.headerString)

        for ii in range(len(self.x)):
            _out.write('%-11s  ' % self.name[ii])
            _out.write('%4.1f    ' % self.mag[ii])
            _out.write('%9.5f  %9.5f   ' % (self.x[ii], self.y[ii]))
            _out.write('%8.5f  %8.5f  ' % (self.xerr[ii], self.yerr[ii]))
            _out.write('%8.3f %8.3f  ' % (self.vx[ii], self.vy[ii]))
            _out.write('%7.3f  %7.3f   ' % (self.vxerr[ii], self.vyerr[ii]))
            _out.write('%8.3f    ' % (self.t0[ii]))
            _out.write('%-10s   ' % str(self.useToAlign[ii]))
            _out.write('%6.3f\n' % self.r[ii])

        _out.close()

def makeLabelDat(root='./', align='align/align_d_rms_t', poly='polyfit_d/fit',
                 oldLabelFile='/u/ghezgroup/data/gc/source_list/label.dat',
                 addNewStars=True, keepOldStars=True, updateStarPosVel=True,
                 newUse=0,
                 stars=None, newLabelFile='label_new.dat'):
    """
    Make a new label.dat file using output from align and polyfit.

    Optional Inputs:
    root: The root of align analysis (e.g. './' or '07_05_18.')
    align: The root filename of the align output.
    poly: The root filename of the polyfit output.
    stars: A starset.StarSet() object with polyfit already loaded.
           This overrides align/poly/root values and is useful for
           custom cuts that trim_align can't handle such as magnitude
           dependent velocity error cuts. BEWARE: stars may be modified.

    Outputs:
    source_list/label_new.dat

    Dependencies:
    Polyfit and align must contain the same numbers/names of stars. Also,
    making the label.dat file depends on having the absolute astrometry
    done correctly. See gcwork.starset to learn about how the absolute
    astrometry is loaded (it depends on a specific reference epoch in align).

    You MUST run this on something that has already been run through
    java align_absolute.
    """
    from gcwork import starset

    if stars == None:
        s = starset.StarSet(root + align, relErr=0)

        if (poly != None):
            s.loadPolyfit(root + poly)
            s.loadPolyfit(root + poly, accel=1)
    else:
        s = stars

    # Trim out the new stars if we aren't going to add them
    if not addNewStars:
        idx = []
        for ss in range(len(s.stars)):
            if 'star' not in s.stars[ss].name:
                idx.append(ss)
        s.stars = [s.stars[ss] for ss in idx]

    # Get the 2D radius of all stars and sort
    radius = s.getArray('r2d')
    ridx = radius.argsort()
    s.stars = [s.stars[ss] for ss in ridx]


    # Get info for all the stars.
    names = np.array(s.getArray('name'))

    if poly != None:
        t0 = s.getArray('fitXv.t0')
        x = s.getArray('fitXv.p')# * -1.0
        y = s.getArray('fitYv.p')
        xerr = s.getArray('fitXv.perr')
        yerr = s.getArray('fitYv.perr')
        vx = s.getArray('fitXv.v') * 1000.0# * -1.0
        vy = s.getArray('fitYv.v') * 1000.0
        vxerr = s.getArray('fitXv.verr') * 1000.0
        vyerr = s.getArray('fitYv.verr') * 1000.0
    else:
        t0 = s.getArray('fitXalign.t0')
        x = s.getArray('fitXalign.p')# * -1.0
        y = s.getArray('fitYalign.p')
        xerr = s.getArray('fitXalign.perr')
        yerr = s.getArray('fitYalign.perr')
        vx = s.getArray('fitXalign.v') * 1000.0# * -1.0
        vy = s.getArray('fitYalign.v') * 1000.0
        vxerr = s.getArray('fitXalign.verr') * 1000.0
        vyerr = s.getArray('fitYalign.verr') * 1000.0

    r2d = np.sqrt(x**2 + y**2)
    mag = s.getArray('mag')

    # Fix Sgr A*
    idx = np.where(names == 'SgrA')[0]
    if (len(idx) > 0):
        x[idx] = 0
        y[idx] = 0
        vx[idx] = 0
        vy[idx] = 0
        vxerr[idx] = 0
        vyerr[idx] = 0
        r2d[idx] = 0

    # Clean up xerr and yerr so that they are at least 1 mas
    idx = np.where(xerr < 0.00001)[0]
    xerr[idx] = 0.00001
    idx = np.where(yerr < 0.00001)[0]
    yerr[idx] = 0.00001

    ##########
    # Load up the old star list and find the starting
    # point for new names.
    ##########
    oldLabels = Labels(labelFile=oldLabelFile)
    alnLabels = Labels(labelFile=oldLabelFile)
    newLabels = Labels(labelFile=oldLabelFile)

    if addNewStars:
        newNumber = calcNewNumbers(oldLabels.name, names)

    # Sort the old label list by radius just in case it
    # isn't already. We will update the radii first since
    # these sometimes get out of sorts.
    oldLabels.r = np.hypot(oldLabels.x, oldLabels.y)
    sidx = oldLabels.r.argsort()
    oldLabels.take(sidx)

    # Clean out the new label lists.
    newLabels.ourName = []
    newLabels.name = []
    newLabels.mag = []
    newLabels.x = []
    newLabels.y = []
    newLabels.xerr = []
    newLabels.yerr = []
    newLabels.vx = []
    newLabels.vy = []
    newLabels.vxerr = []
    newLabels.vyerr = []
    newLabels.t0 = []
    newLabels.useToAlign = []
    newLabels.r = []

    # Load up the align info into the alnLabels object
    alnLabels.ourName = names
    alnLabels.name = names
    alnLabels.mag = mag
    alnLabels.x = x
    alnLabels.y = y
    alnLabels.xerr = xerr
    alnLabels.yerr = yerr
    alnLabels.vx = vx
    alnLabels.vy = vy
    alnLabels.vxerr = vxerr
    alnLabels.vyerr = vyerr
    alnLabels.t0 = t0
    alnLabels.r = r2d


    def addStarFromAlign(alnLabels, ii, use):
        newLabels.ourName.append(alnLabels.ourName[ii])
        newLabels.name.append(alnLabels.name[ii])
        newLabels.mag.append(alnLabels.mag[ii])
        newLabels.x.append(alnLabels.x[ii])
        newLabels.y.append(alnLabels.y[ii])
        newLabels.xerr.append(alnLabels.xerr[ii])
        newLabels.yerr.append(alnLabels.yerr[ii])
        newLabels.vx.append(alnLabels.vx[ii])
        newLabels.vy.append(alnLabels.vy[ii])
        newLabels.vxerr.append(alnLabels.vxerr[ii])
        newLabels.vyerr.append(alnLabels.vyerr[ii])
        newLabels.t0.append(alnLabels.t0[ii])
        newLabels.useToAlign.append(use)
        newLabels.r.append(alnLabels.r[ii])

    def addStarFromOldLabels(oldLabels, ii):
        newLabels.ourName.append(oldLabels.name[ii])
        newLabels.ourName.append(oldLabels.ourName[ii])
        newLabels.name.append(oldLabels.name[ii])
        newLabels.mag.append(oldLabels.mag[ii])
        newLabels.x.append(oldLabels.x[ii])
        newLabels.y.append(oldLabels.y[ii])
        newLabels.xerr.append(oldLabels.xerr[ii])
        newLabels.yerr.append(oldLabels.yerr[ii])
        newLabels.vx.append(oldLabels.vx[ii])
        newLabels.vy.append(oldLabels.vy[ii])
        newLabels.vxerr.append(oldLabels.vxerr[ii])
        newLabels.vyerr.append(oldLabels.vyerr[ii])
        newLabels.t0.append(oldLabels.t0[ii])
        newLabels.useToAlign.append(oldLabels.useToAlign[ii])
        newLabels.r.append(oldLabels.r[ii])

    def deleteFromAlign(alnLabels, idx):
        # Delete them from the align lists.
        alnLabels.ourName = np.delete(alnLabels.ourName, idx)
        alnLabels.name = np.delete(alnLabels.name, idx)
        alnLabels.mag = np.delete(alnLabels.mag, idx)
        alnLabels.x = np.delete(alnLabels.x, idx)
        alnLabels.y = np.delete(alnLabels.y, idx)
        alnLabels.xerr = np.delete(alnLabels.xerr, idx)
        alnLabels.yerr = np.delete(alnLabels.yerr, idx)
        alnLabels.vx = np.delete(alnLabels.vx, idx)
        alnLabels.vy = np.delete(alnLabels.vy, idx)
        alnLabels.vxerr = np.delete(alnLabels.vxerr, idx)
        alnLabels.vyerr = np.delete(alnLabels.vyerr, idx)
        alnLabels.t0 = np.delete(alnLabels.t0, idx)
        alnLabels.r = np.delete(alnLabels.r, idx)


    nn = 0
    while nn < len(oldLabels.name):
        #
        # First see if there are any new stars that should come
        # before this star.
        #
        if addNewStars:
            def filterFunction(i):
                return (alnLabels.r[i] < oldLabels.r[nn]) and ('star' in alnLabels.name[i])
            idx = list(filter(filterFunction, list(range(len(alnLabels.name)))))

            for ii in idx:
                rAnnulus = int(math.floor(alnLabels.r[ii]))
                number = newNumber[rAnnulus]
                alnLabels.name[ii] = 'S%d-%d' % (rAnnulus, number)
                newNumber[rAnnulus] += 1

                # Insert these new stars.
                addStarFromAlign(alnLabels, ii, newUse)

            # Delete these stars from the align info.
            deleteFromAlign(alnLabels, idx)

        #
        # Now look for this star in the new align info
        #
        idx = np.where(alnLabels.name == oldLabels.name[nn])[0]

        if len(idx) > 0:
            # Found the star

            if updateStarPosVel:
                # Update with align info
                addStarFromAlign(alnLabels, idx[0], oldLabels.useToAlign[nn])
            else:
                # Don't update with align info
                addStarFromOldLabels(oldLabels, nn)

            deleteFromAlign(alnLabels, idx[0])

        elif keepOldStars:
            # Did not find the star. Only keep if user said so.
            addStarFromOldLabels(oldLabels, nn)

        nn += 1

    # Quick verification that we don't have repeat names.
    uniqueNames = np.unique(newLabels.name)
    if len(uniqueNames) != len(newLabels.name):
        print('Problem, we have a repeat name!!')

    # Write to output
    newLabels.saveToFile(root + 'source_list/' + newLabelFile)


def calcNewNumbers(oldNames, newNames):
    # Loop through annuli of 1 arcsecond and find last name
    rRange = np.arange(20)
    newNumber = np.zeros(len(rRange))
    for rr in range(len(rRange)):
        substring = 'S%d-' % rr
        rNameOld = [x for x in oldNames if x.find(substring) != -1]
        rNameNew = [x for x in newNames if x.find(substring) != -1]

        if (len(rNameOld) == 0):
            newNumber[rr] = 1
        else:
            rNumberOld = np.zeros(len(rNameOld))
            for nn in range(len(rNameOld)):
                tmp = rNameOld[nn].split('-')
                rNumberOld[nn] = int(tmp[-1])
            rNumberOld.sort()

            if (len(rNameNew) != 0):
                rNumberNew = np.zeros(len(rNameNew))
                for nn in range(len(rNameNew)):
                    tmp = rNameNew[nn].split('-')
                    rNumberNew[nn] = int(tmp[-1])
                rNumberNew.sort()
            else:
                rNumberNew = np.array([1])

            newNumber[rr] = max([rNumberOld[-1], rNumberNew[-1]]) + 1

        print('First New Number is S%d-%d' % (rRange[rr], newNumber[rr]))

    return newNumber

def makeOrbitsDat(root='./', efit='efit3_d/output/efit3.log',
                  poly='polyfit_d/fit', onlyHighAccel=True):
    """
    Make a new orbits.dat file using output from polyfit and efit3.

    Optional Inputs:
    root: The root of align analysis (e.g. './' or '07_05_18.')
    poly: The root filename of the polyfit output.
    efit: The efit3.log file containing the new orbit solutions.

    Outputs:
    source_list/orbits_new.dat

    Dependencies:
    Only sources in the central arcsecond with significant accelerations
    are included in our list of stellar orbits. To determine which stars
    these are, we run

    gcwork.polyfit.accel.highSigSrcs(0.5, 4)

    and then use all the named sources in the resulting list.
    """
    from gcwork.polyfit import accel

    # Now read in the efit3.log file
    tab = Table.read(root + efit)

    name = tab[0]._data
    dist = tab[1].tonumpy()  # pc
    a = tab[4].tonumpy()     # mas
    p = tab[5].tonumpy()     # yr
    e = tab[6].tonumpy()     #
    t0 = tab[7].tonumpy()    # yr
    w = tab[8].tonumpy()     # deg
    i = tab[9].tonumpy()     # deg
    o = tab[10].tonumpy()    # deg

    if onlyHighAccel == True:
        # Find the significantly accelerating sources within the
        # central arcsecond.
        srcs = accel.highSigSrcs(0.5, 4, verbose=False, rootDir=root, poly=poly)
    else:
        # Use ALL stars in this list
        srcs = name

    _out = open(root + 'source_list/orbits_new.dat', 'w')
    _out.write('# Python gcwork.starTables.makeOrbitsDat()\n')
    _out.write('%-10s  %7s  %7s  %8s  %7s  %7s  %7s  %7s  %7s\n' % \
               ('#Star', 'P', 'A', 't0', 'e', 'i', 'Omega', 'omega', 'search'))
    _out.write('%-10s  %7s  %7s  %8s  %7s  %7s  %7s  %7s  %7s\n' % \
               ('#Name', '(yrs)', '(mas)', '(yrs)', '()',
                '(deg)', '(deg)', '(deg)', '(pix)'))


    # Loop through every src and if it is named, output into a
    # new orbits_new.dat file.
    for ss in range(len(srcs)):
        try:
            idx = name.index(srcs[ss])
        except ValueError:
            #print 'Failed to find match for %s in %s' % (srcs[ss], efit)
            continue

        # Skip if this isn't a named source
        if (('star' in srcs[ss]) and (onlyHighAccel == True)):
            continue

        # Write output
        _out.write('%-10s  ' % (srcs[ss]))
        _out.write('%7.2f  %7.1f  %8.3f  ' % (p[idx], a[idx], t0[idx]))
        _out.write('%7.5f  %7.3f  %7.3f  ' % (e[idx], i[idx], o[idx]))
        _out.write('%7.3f  %7d\n' % (w[idx], 2))

    _out.close()


def labelNoYoung(input_labels, output_labels):
    """
    Take an existing label.dat file and set all the known young stars
    to NOT be used in alignment.
    """

    labels = Labels(labelFile=input_labels)

    # Load up the list of young stars
    yng = youngStarNames(datfile='/u/ghezgroup/data/gc/source_list/young_new.dat')

    _out = open(output_labels, 'w')
    _out.write('%-10s  %5s   ' % ('#Name', 'K'))
    _out.write('%7s %7s %7s %7s   ' % ('x', 'y', 'xerr', 'yerr'))
    _out.write('%8s %8s %8s %8s   ' % ('vx', 'vy', 'vxerr', 'vyerr'))
    _out.write('%8s %4s %7s\n' %  ('t0', 'use?', 'r2d'))

    _out.write('%-10s  %5s   ' % ('#()', '(mag)'))
    _out.write('%7s %7s %7s %7s   ' % \
               ('(asec)', '(asec)', '(asec)', '(asec)'))
    _out.write('%8s %8s %8s %8s   ' % \
               ('(mas/yr)', '(mas/yr)', '(mas/yr)', '(mas/yr)'))
    _out.write('%8s %4s %7s\n' %  ('(year)', '()', '(asec)'))

    for i in range(len(labels.name)):
        if (labels.name[i] in yng):
            labels.useToAlign[i] = 0   # Don't use for alignment

        _out.write('%-10s  %5.1f   ' % (labels.name[i], labels.mag[i]))
        _out.write('%7.3f %7.3f ' % (labels.x[i], labels.y[i]))
        _out.write('%7.3f %7.3f   ' % (labels.xerr[i], labels.yerr[i]))
        _out.write('%8.3f %8.3f ' % (labels.vx[i], labels.vy[i]))
        _out.write('%8.3f %8.3f   ' % (labels.vxerr[i], labels.vyerr[i]))
        _out.write('%8.3f %4d %7.3f\n' %  \
                   (labels.t0[i], labels.useToAlign[i], labels.r[i]))


    _out.close()


def labelRestrict(inputLabel, outputLabel, alignInput,
                  numSpeck=None, numAO=None):
    """
    Modify an existing label.dat file to be used with the align
    -restrict flag. This is the program that chooses which stars
    are to be used for speckle alignment and which are to be
    used for AO alignment. The stars are chosen based on the
    number of detections in either speckle or AO.
    We will use the subset of stars in ALL speckle epochs as speckle
    alignment sources; and those stars that are in ALL AO epochs
    are used as AO alignment sources. The input align files should
    not have been trimmed for the most part.

    Be sure that there is an <alignInput>.list file containing
    the epochs and their data types.

    Paramters:
    inputLabel -- the input label.dat file. This will not be modified.
    outputLabel -- the output label.dat file. Only the use? column is changed.
    alignInput -- the root name of the align files to be used when
                  determining how many speckle and AO maps a stars is found in.
    numSpeck -- if None then only stars in ALL speckle epochs are used
                as alignment sources.
    numAO -- if None then only stars in ALL AO epochs are used as
             alignment sources.
    """
    from gcwork import starset

    labels = Labels(labelFile=inputLabel)
    s = starset.StarSet(alignInput)

    # Figure out the data/camera type for each epoch (speckle or AO)
    _list = open(alignInput + '.list', 'r')
    aoEpochs = []
    spEpochs = []

    i = 0
    for line in _list:
        info = line.split()
        aoType = int( info[1] )

        if ((aoType == 2) or (aoType == 3)):
            spEpochs.append(i)
        if ((aoType == 8) or (aoType == 9)):
            aoEpochs.append(i)

        i += 1

    if (numSpeck == None):
        numSpeck = len(spEpochs)
    if (numAO == None):
        numAO = len(aoEpochs)

    # For each star, count up the number of speckle and AO epochs it is
    # detected in.
    names = s.getArray('name')
    velCnt = s.getArray('velCnt')

    numStars = len(names)
    numEpochs = len(s.stars[0].years)

    print('Initial:  Nstars = %4d  Nepochs = %2d' % (numStars, numEpochs))
    print('Number of Epochs of Type:')
    print('   Speckle = %d' % len(spEpochs))
    print('   AO      = %d' % len(aoEpochs))


    aoCnt = np.zeros(numStars)
    spCnt = np.zeros(numStars)

    for e in range(numEpochs):
        pos = s.getArrayFromEpoch(e, 'x')

        idx = (np.where(pos > -1000))[0]

        if (e in aoEpochs):
            aoCnt[idx] += 1
        if (e in spEpochs):
            spCnt[idx] += 1

    # Now lets write the output
    _out = open(outputLabel, 'w')
    _out.write('%-10s  %5s   ' % ('#Name', 'K'))
    _out.write('%7s %7s %7s %7s   ' % ('x', 'y', 'xerr', 'yerr'))
    _out.write('%8s %8s %8s %8s   ' % ('vx', 'vy', 'vxerr', 'vyerr'))
    _out.write('%8s %4s %7s\n' %  ('t0', 'use?', 'r2d'))

    _out.write('%-10s  %5s   ' % ('#()', '(mag)'))
    _out.write('%7s %7s %7s %7s   ' % \
               ('(asec)', '(asec)', '(asec)', '(asec)'))
    _out.write('%8s %8s %8s %8s   ' % \
               ('(mas/yr)', '(mas/yr)', '(mas/yr)', '(mas/yr)'))
    _out.write('%8s %4s %7s\n' %  ('(year)', '()', '(asec)'))

    spNumStars = 0
    aoNumStars = 0
    use = '1'
    for i in range(len(labels.name)):
        # Find this star in our align output
        try:
            foo = names.index(labels.name[i])

            if (labels.useToAlign[i] == 0):
                # Preserve any pre-existing use?=0 stars
                use = '0'

                if (spCnt[foo] >= numSpeck):
                    print('%-13s is in all speckle epochs, but use=0' % \
                          names[foo])
            else:
                if (spCnt[foo] >= numSpeck):
                    use = '2'
                    spNumStars += 1

                if (aoCnt[foo] >= numAO):
                    aoNumStars += 1

                    if (use == '2'):
                        # Speckle and AO
                        use += ',8'
                    else:
                        # AO only
                        use = '8'

        except ValueError:
            # Don't change anything if we didn't find it.
            # Reformat to string for ease of use
            use = str(labels.useToAlign[i])


        _out.write('%-10s  %5.1f   ' % (labels.name[i], labels.mag[i]))
        _out.write('%7.3f %7.3f ' % (labels.x[i], labels.y[i]))
        _out.write('%7.3f %7.3f   ' % (labels.xerr[i], labels.yerr[i]))
        _out.write('%8.3f %8.3f ' % (labels.vx[i], labels.vy[i]))
        _out.write('%8.3f %8.3f   ' % (labels.vxerr[i], labels.vyerr[i]))
        _out.write('%8.3f %-4s %7.3f\n' %  \
                   (labels.t0[i], use, labels.r[i]))


    _out.close()

    print('Final:   Nstars Speckle = %4d  AO = %4d' % \
          (spNumStars, aoNumStars))




def updateLabelInfoWithAbsRefs(oldLabelFile, newLabelFile, outputFile,
                               newUse=1, oldUse=0, appendNew=False):
    """
    Modify an existing label.dat file with updated positions and
    velocities from a different label.dat (or absolute_refs.dat) file.

    Input Parameters:
    oldLabelFile - The name of the input old label.dat file
    newLabelFile - The name of the input file from which to pull new vel info.
    outputFile - Save the results to a new file.

    Optional Parameters:
    newUse - (def=1) set to this value for stars that are modified,
             or if None, preserve what was in the old label file.
    oldUse - (def=0) set to this value for stars that are not modified,
             or if None, preserve what was in the old label file.
    """
    old = Labels(oldLabelFile)
    new = Labels(newLabelFile)
    print('%5d stars in list with old velocities' % len(old.name))
    print('%5d stars in list with new velocities' % len(new.name))

    if oldUse != None:
        old.useToAlign[:] = oldUse

    if appendNew:
        newNumber = calcNewNumbers(old.name, new.name)

    updateCount = 0
    newCount = 0

    for nn in range(len(new.name)):
        idx = np.where(old.name == new.name[nn])[0]

        if len(idx) > 0:
            old.mag[idx] = new.mag[nn]
            old.x[idx] = new.x[nn]
            old.y[idx] = new.y[nn]
            old.xerr[idx] = new.xerr[nn]
            old.yerr[idx] = new.yerr[nn]
            old.vx[idx] = new.vx[nn]
            old.vy[idx] = new.vy[nn]
            old.vxerr[idx] = new.vxerr[nn]
            old.vyerr[idx] = new.vyerr[nn]
            old.t0[idx] = new.t0[nn]
            old.r[idx] = new.r[nn]

            if newUse != None:
                old.useToAlign[idx] = newUse

            updateCount += 1
        else:
            if appendNew:
                rAnnulus = int(math.floor(new.r[nn]))
                number = newNumber[rAnnulus]
                new.name[nn] = 'S%d-%d' % (rAnnulus, number)
                newNumber[rAnnulus] += 1

                old.name = np.append(old.name, new.name[nn])
                old.mag = np.append(old.mag, new.mag[nn])
                old.x = np.append(old.x, new.x[nn])
                old.y = np.append(old.y, new.y[nn])
                old.xerr = np.append(old.xerr, new.xerr[nn])
                old.yerr = np.append(old.yerr, new.yerr[nn])
                old.vx = np.append(old.vx, new.vx[nn])
                old.vy = np.append(old.vy, new.vy[nn])
                old.vxerr = np.append(old.vxerr, new.vxerr[nn])
                old.vyerr = np.append(old.vyerr, new.vyerr[nn])
                old.t0 = np.append(old.t0, new.t0[nn])
                old.r = np.append(old.r, new.r[nn])
                old.useToAlign = np.append(old.useToAlign, newUse)

                newCount += 1
    print('%5d stars in the NEW starlist created' % len(old.name))
    print('   %5d updated' % updateCount)
    print('   %5d added'   % newCount)


    old.saveToFile(outputFile)

def checkLabelsForDuplicates(labels='/u/ghezgroup/data/gc/source_list/label.dat'):
    """
    Read in a label.dat file (or a labels object) and search for duplicates
    based on position and magnitude.
    """
    if type(labels) is 'gcwork.starTables.Labels':
        lab = labels
    else:
        lab = Labels(labelFile=labels)

    rdx = lab.r.argsort()
    lab.take(rdx)

    duplicateCnt = 0

    dummy = np.arange(len(lab.name))

    for ii in dummy:
        dx = lab.x - lab.x[ii]
        dy = lab.y - lab.y[ii]
        dm = np.abs(lab.mag - lab.mag[ii])

        dr = np.hypot(dx, dy)

        # Search for stars within 50 mas
        rdx = np.where((dr < 0.05) & (dm < 1) & (dummy >= ii))[0]

        if len(rdx) > 1:
            duplicateCnt += 1

            print('')
            print('Found stars close to %s' % lab.name[ii])
            print('    %-13s  %5s  %7s %7s  %7s %7s' % \
                ('Name', 'mag', 'x', 'y', 'vx', 'vy'))

            for rr in rdx:
                print('    %-13s  %5.2f  %7.3f %7.3f   %7.3f %7.3f' % \
                    (lab.name[rr], lab.mag[rr], lab.x[rr], lab.y[rr],
                     lab.vx[rr], lab.vy[rr]))

    print('')
    print('Found %d duplicates' % duplicateCnt)

def updateLabelInfoWithDeepMosaic(oldLabelFile, dpMscAlignDir, outputFile,
                                  alignRoot='align/align_d_rms_100_abs_t',
                                  polyRoot = 'polyfit_100/fit'):

    """
    Modify an existing label.dat file with updated positions,
    velocities, and magnitudes from a dp_msc alignment.

    Input Parameters:
    oldLabelFile - The name of the input old label.dat file
    dpMscAlignDir - the root directory name of a dp_msc alignment
                    and polyfit (e.g. /u/jlu/work/gc/do_msc/2011_05_29/)
    outputFile - Save the modified label.dat content to a new file.
    """
    old = Labels(oldLabelFile)

    s = starset.StarSet(dpMscAlign + alignRoot)
    s.loadPolyfit(dpMscAlign + polyRoot, accel=0)


    ### NOT DONE YET ###
