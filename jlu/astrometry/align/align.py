from __future__ import division
import numpy as np
from collections import Counter
import pdb

# try:
#     from scipy.spatial import cKDTree as KDT
# except ImportError:
    # from scipy.spatial import KDTree as KDT
 
from scipy.spatial import KDTree as KDT
 
 
def match(x1, y1, m1, x2, y2, m2, dr_tol, dm_tol=None):
    """
    Finds matches between two different catalogs. No transformations are done and it
    is assumed that the two catalogs are already on the same coordinate system
    and magnitude system.

    For two stars to be matched, they must be within a specified radius (dr_tol) and
    delta-magnitude (dm_tol). For stars with more than 1 neighbor (within the tolerances),
    if one is found that is the best match in both brightness and positional offsets
    (closest in both), then the match is made. Otherwise,
    their is a conflict and no match is returned for the star.
    
 
    Parameters
    x1 : array-like
        X coordinate in the first catalog
    y1 : array-like
        Y coordinate in the first catalog (shape of array must match `x1`)
    m1 : array-like
        Magnitude in the first catalog. Must have the same shape as x1.
    x2 : array-like
        X coordinate in the second catalog
    y2 : array-like
        Y coordinate in the second catalog (shape of array must match `x2`)
    m2 : array-like
        Magnitude in the second catalog. Must have the same shape as x2.
    dr_tol : float
        How close (in units of the first catalog) a match has to be to count as a match.
        For stars with more than one nearest neighbor, the delta-magnitude is checked
        and the closest in delta-mag is chosen.
    dm_tol : float or None, optional
        How close in delta-magnitude a match has to be to count as a match.
        If None, then any delta-magnitude is allowed.
 
    Returns
    -------
    idx1 : int array
        Indicies into the first catalog of the matches. Will never be
        larger than `x1`/`y1`.
    idx2 : int array
        Indicies into the second catalog of the matches. Will never be
        larger than `x1`/`y1`.
    dr : float array
        Distance between the matches.
    dm : float array
        Delta-mag between the matches. (m1 - m2)
 
    """
 
    x1 = np.array(x1, copy=False)
    y1 = np.array(y1, copy=False)
    m1 = np.array(m1, copy=False)
    x2 = np.array(x2, copy=False)
    y2 = np.array(y2, copy=False)
    m2 = np.array(m2, copy=False)
 
    if x1.shape != y1.shape:
        raise ValueError('x1 and y1 do not match!')
    if x2.shape != y2.shape:
        raise ValueError('x2 and y2 do not match!')
 
    # Setup coords1 pairs and coords 2 pairs
    # this is equivalent to, but faster than just doing np.array([x1, y1])
    coords1 = np.empty((x1.size, 2))
    coords1[:, 0] = x1
    coords1[:, 1] = y1
 
    # this is equivalent to, but faster than just doing np.array([x1, y1])
    coords2 = np.empty((x2.size, 2))
    coords2[:, 0] = x2
    coords2[:, 1] = y2

    # Utimately we will generate arrays of indices.
    # idxs1 is the indices for matches into catalog 1. This
    # is just a place holder for which stars actually
    # have matches.
    idxs1 = np.ones(x1.size, dtype=int) * -1
    idxs2 = np.ones(x1.size, dtype=int) * -1

    # The matching will be done using a KDTree.
    kdt = KDT(coords2)

    # This returns the number of neighbors within the specified
    # radius. We will use this to find those stars that have no or one
    # match and deal with them easily. The more complicated conflict
    # cases will be dealt with afterward.
    i2_match = kdt.query_ball_point(coords1, dr_tol)
    Nmatch = np.array([len(idxs) for idxs in i2_match])

    # What is the largest number of matches we have for a given star?
    Nmatch_max = Nmatch.max()


    # Loop through and handle all the different numbers of matches.
    # This turns out to be the most efficient so we can use numpy
    # array operations. Remember, skip the Nmatch=0 objects... they
    # already have indices set to -1.
    for nn in range(1, Nmatch_max+1):
        i1_nn = np.where(Nmatch == nn)[0]

        if len(i1_nn) == 0:
            continue

        if nn == 1:
            i2_nn = np.array([i2_match[mm][0] for mm in i1_nn])
            if dm_tol != None:
                dm = np.abs(m1[i1_nn] - m2[i2_nn])
                keep = dm < dm_tol
                idxs1[i1_nn[keep]] = i1_nn[keep]
                idxs2[i1_nn[keep]] = i2_nn[keep]
            else:
                idxs1[i1_nn] = i1_nn
                idxs2[i1_nn] = i2_nn
        else:
            i2_tmp = np.array([i2_match[mm] for mm in i1_nn])

            # Repeat star list 1 positions and magnitudes
            # for nn times (tile then transpose) 
            x1_nn = np.tile(x1[i1_nn], (nn, 1)).T
            y1_nn = np.tile(y1[i1_nn], (nn, 1)).T
            m1_nn = np.tile(m1[i1_nn], (nn, 1)).T

            # Get out star list 2 positions and magnitudes
            x2_nn = x2[i2_tmp]
            y2_nn = y2[i2_tmp]
            m2_nn = m2[i2_tmp]
            dr = np.abs(x1_nn - x2_nn, y1_nn - y2_nn)
            dm = np.abs(m1_nn - m2_nn)

            if dm_tol != None:
                # Don't even consider stars that exceed our
                # delta-mag threshold. 
                dr_msk = np.ma.masked_where(dm > dm_tol, dr)
                dm_msk = np.ma.masked_where(dm > dm_tol, dm)

                # Remember that argmin on masked arrays can find
                # one of the masked array elements if ALL are masked.
                # But our subsequent "keep" check should get rid of all
                # of these.
                dm_min = dm_msk.argmin(axis=1)
                dr_min = dr_msk.argmin(axis=1)

                # Double check that "min" choice is still within our
                # detla-mag tolerence.
                dm_tmp = np.choose(dm_min, dm.T)

                keep = (dm_min == dr_min) & (dm_tmp < dm_tol)
            else:
                dm_min = dm.argmin(axis=1)
                dr_min = dr.argmin(axis=1)

                keep = (dm_min == dr_min)

            i2_keep_2D = i2_tmp[keep]
            dr_keep = dr_min[keep]  # which i2 star for a given i1 star
            ii_keep = np.arange(len(dr_keep))  # a running index for the i2 keeper stars.

            idxs1[i1_nn[keep]] = i1_nn[keep]
            idxs2[i1_nn[keep]] = i2_keep_2D[ii_keep, dr_keep]

    idxs1 = idxs1[idxs1 >= 0]
    idxs2 = idxs2[idxs2 >= 0]        

    dr = np.hypot(x1[idxs1] - x2[idxs2], y1[idxs1] - y2[idxs2])
    dm = m1[idxs1] - m2[idxs2]

    # Deal with duplicates
    duplicates = [item for item, count in Counter(idxs2).iteritems() if count > 1]
    print 'Found {0:d} out of {1:d} duplicates'.format(len(duplicates), len(dm))
    # for dd in range(len(duplicates)):
    #     dups = np.where(idxs2 == duplicates[dd])[0]

    #     # Handle them in brightness order -- brightest first in the first starlist
    #     fsort = m1[dups].argsort()

    #     # For every duplicate, match to the star that is closest in space and 
    #     # magnitude. HMMMM.... this doesn't seem like it will work optimally.

 
    return idxs1, idxs2, dr, dm
 
 
