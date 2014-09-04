import numpy as np
import pylab as py
import atpy
import os
import shutil
import cPickle as pickle

xym_dir = '/u/jlu/data/Wd1/hst/reduce_2013_03_12/2010_F160W/01.XYM'
mat_dir = '/u/jlu/data/Wd1/hst/reduce_2013_03_12/02.PMA/2010_160'
test_dir = mat_dir + '/test_overlap'

def xym2bar_singles():
    """
    Loop through and run xym2bar on a single star list.
    This will transform the entire star list into the
    coordinate system of the average star list. Also
    run with the "I" flag so that only stars with matches
    in the average list and printed out... this allows
    us to cross-match.

    This needs to be run in the test_overlap directory.
    """
    images = range(301, 328+1)
    # images = range(301, 302)


    master = '../MASTER.F160W.ref5'

    old_in = atpy.Table(mat_dir + '/IN.xym2mat', type='ascii')

    for ii in images:
        mat_file = '{0}/MAT.{1}'.format(mat_dir, ii+1)
        xym_file = old_in.col2[old_in.col1 == ii][0]
        in_file = test_dir + '/IN.xym2bar'

        _in = open(in_file, 'w')
        _in.write('000 "{0}"\n'.format(master))
        _in.write('{0} "../{1}" c9\n'.format(ii, xym_file))
        _in.close()

        os.system('xym2bar 1 I')

        shutil.copyfile(in_file, '{0}.{1}'.format(in_file, ii))
        shutil.copyfile('MATCHUP.XYMEEE', 'MATCHUP.XYMEEE.{0}'.format(ii))
        

def make_catalog():
    """
    Make a catalog of all the stars that are in at least 7 of the
    exposures. Also load up the master frame coordinates.
    """
    # Load up the master frame
    master = atpy.Table('../MASTER.F160W.ref5', type='ascii')
    x_mast = master.col1
    y_mast = master.col2
    m_mast = master.col3
    xe_mast = master.col4
    ye_mast = master.col5
    me_mast = master.col6

    images = range(301, 328+1)

    # Make output arrays.
    x = np.zeros((len(images)+1, len(x_mast)), dtype=np.float32)
    y = np.zeros((len(images)+1, len(x_mast)), dtype=np.float32)
    m = np.zeros((len(images)+1, len(x_mast)), dtype=np.float32)

    xe = np.zeros(len(x_mast), dtype=np.float32)
    ye = np.zeros(len(x_mast), dtype=np.float32)
    me = np.zeros(len(x_mast), dtype=np.float32)
    cnt = np.zeros(len(x_mast), dtype=np.int8)

    # Put the master coords in as the 0th element
    x[0,:] = x_mast
    y[0,:] = y_mast
    m[0,:] = m_mast
    xe = xe_mast
    ye = ye_mast
    me = me_mast
    
    # Load up all the other lists
    for ii in range(len(images)):
        match_file = 'MATCHUP.XYMEEE.{0}'.format(images[ii])
        print 'Processing ' + match_file
        
        stars = atpy.Table(match_file, type='ascii')

        idx = np.where(stars.col4 < 9)[0]

        x[ii+1,idx] = stars.col1[idx]
        y[ii+1,idx] = stars.col2[idx]
        m[ii+1,idx] = stars.col3[idx]

        cnt[idx] += 1

    # Trim down to everything that is in at least 7 exposures
    idx = np.where(cnt >= 7)[0]
    print 'Keeping {0} of {1} stars in 7 or more lists.'.format(len(idx), len(cnt))
    x = x[:,idx]
    y = y[:,idx]
    m = m[:,idx]
    xe = xe[:,idx]
    ye = ye[:,idx]
    me = me[:,idx]
    cnt = cnt[idx]
     
    _out = open('stars_aligned.pickle', 'w')
    pickle.dump(x, _out)
    pickle.dump(y, _out)
    pickle.dump(m, _out)
    pickle.dump(xe, _out)
    pickle.dump(ye, _out)
    pickle.dump(me, _out)
    pickle.dump(cnt, _out)
    _out.close()


def load_catalog():
    _in = open('stars_aligned.pickle', 'r')
    x = pickle.load(_in)
    y = pickle.load(_in)
    m = pickle.load(_in)
    xe = pickle.load(_in)
    ye = pickle.load(_in)
    me = pickle.load(_in)
    cnt = pickle.load(_in)

    _in.close

    return (x, y, m, xe, ye, me, cnt)
    
def plot_vector_diff_F160W():
    """
    Using the xym2mat analysis on the F160W filter in the 2010 data set,
    plot up the positional offset vectors for each reference star in
    each star list relative to the average list. This should show
    us if there are systematic problems with the distortion solution
    or PSF variations.
    """
    x, y, m, xe, ye, me, cnt = load_catalog()

    dx = x - x[0]
    dy = y - y[0]
    dr = np.hypot(dx, dy)

    py.clf()
    py.subplots_adjust(left=0.05, bottom=0.05, right=0.95, top=0.95)

    for ii in range(1, x.shape[0]):
        idx = np.where((x[ii,:] != 0) & (y[ii,:] != 0) & (dr[ii,:] < 0.1) &
                       (xe < 0.05) & (ye < 0.05))[0]

        py.clf()
        q = py.quiver(x[0,idx], y[0,idx], dx[ii, idx], dy[ii, idx], scale=1.0)
        py.quiverkey(q, 0.9, 0.9, 0.03333, label='2 mas', color='red')
        py.title('{0} stars in list {1}'.format(len(idx), ii))
        py.xlim(0, 4500)
        py.ylim(0, 4500)

        foo = raw_input('Continue?')
        if foo == 'q' or foo == 'Q':
            break
