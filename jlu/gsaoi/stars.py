import numpy as np
import pylab as py
from gcwork import starset
import os
import atpy
import cPickle as pickle

class Starlist(object):
    def __init__(self, alignRoot):
        """
        Read in a file created by Starlist.process_align_output()
        """
        _in = open(alignRoot + '_xym.pickle', 'r')
        self.name = pickle.load(_in)
        self.ndet = pickle.load(_in)
        self.xref = pickle.load(_in)
        self.yref = pickle.load(_in)
        self.mref = pickle.load(_in)
        self.dxm = pickle.load(_in)
        self.dym = pickle.load(_in)
        self.mm = pickle.load(_in)
        self.roots = pickle.load(_in)
        self.quads = pickle.load(_in)
        self.dates = pickle.load(_in)
        self.tints = pickle.load(_in)
        _in.close()

        self.alignRoot = alignRoot

    def process_align_output(alignRoot):
        """
        Read in align output, store in a starset, and save in a digital format
        for fast reading in the future.
        """
        print "Fetching lists, dates, and integration times."
        roots, quads, dates, tints = Starlist.read_align_list(alignRoot)

        print "Reading in align output."
        s = starset.StarSet(alignRoot)

        x = s.getArrayFromAllEpochs('x')
        y = s.getArrayFromAllEpochs('y')
        m = s.getArrayFromAllEpochs('mag')

        print "Processing align output."
        # Save the first epoch as a reference.
        xref = x[0,:]
        yref = y[0,:]
        mref = m[0,:]

        # Drop the first epoch, a reference.
        x = x[1:,:]
        y = y[1:,:]
        m = m[1:,:]

        name = s.getArray('name')
        ndet = np.array(s.getArray('velCnt'), dtype=np.int16)

        # Set up mask to get rid of invalid values
        idx = x <= -1e5

        # For performance reasons, convert to dx and dy
        # and store in smaller array format.
        dx = np.array(x - xref, dtype=np.float32)
        dy = np.array(y - yref, dtype=np.float32)
        m = np.array(m, dtype=np.float16)

        # Create masked arrays
        dxm = np.ma.masked_where(idx, dx)
        dym = np.ma.masked_where(idx, dy)
        mm = np.ma.masked_where(idx, m)

        _out = open(alignRoot + '_xym.pickle', 'w')
        pickle.dump(name, _out)
        pickle.dump(ndet, _out)
        pickle.dump(xref, _out)
        pickle.dump(yref, _out)
        pickle.dump(mref, _out)
        pickle.dump(dxm, _out)
        pickle.dump(dym, _out)
        pickle.dump(mm, _out)
        pickle.dump(roots, _out)
        pickle.dump(quads, _out)
        pickle.dump(dates, _out)
        pickle.dump(tints, _out)
        _out.close()


    def read_align_list(alignRoot):
        """
        Read in a *.list file
        """

        alignList = alignRoot + '.list'

        # Read in the *.list file with the names of the starlists.
        # Drop the first one as it is the reference and just a repeat.
        starlistsTab = atpy.Table(alignList, type='ascii',
                                  delimiter=' ', data_start=1)
        starlists = starlistsTab.col1

        # Loop through and figure out the night and integration time.
        roots = np.zeros(len(starlists), dtype='S14')
        quads = np.zeros(len(starlists), dtype='S4')
        dates = np.zeros(len(starlists), dtype='S8')
        tints = np.zeros(len(starlists), dtype=np.float16)

        for ii in range(len(starlists)):
            tmp = starlists[ii].split('/')
            tmp2 = tmp[-1].split('_cr_')
            tmp3 = tmp2[0].split('_')

            roots[ii] = tmp3[0]
            quads[ii] = '_'.join(tmp3[1:])
            dates[ii] = roots[ii][1:7]

            starlistRoot = os.path.basename(starlists[ii])
            starlistRoot = os.path.splitext(starlistRoot)[0]
            starlistRoot = starlistRoot.replace('_0.9_stf_cal', '')
            fitsFile = 'starfinder/' + starlistRoot + '.fits'
            if os.path.exists(fitsFile):
                hdr = pyfits.getheader(fitsFile)
                tints[ii] = float(hdr['EXPTIME']) * float(hdr['COADDS'])

        return (roots, quads, dates, tints)
        

    process_align_output = staticmethod(process_align_output)
    read_align_list = staticmethod(read_align_list)


