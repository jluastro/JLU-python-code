from . import align
import numpy as np
import pdb

def test_match_easy():
    """
    Test when all stars match almost exactly.
    """
    # First star list spread over 10 pixels.
    x1 = np.array([2.0, 4.0, 6.0, 8.0, 10.0])
    y1 = np.array([2.0, 4.0, 6.0, 8.0, 10.0])
    m1 = np.array([8.0, 10.0, 9.0, 12.0, 13.0])
    nstars = len(x1)

    # Second star list that has noise (of 0.05 pixel).
    x2 = x1 + (np.random.randn(nstars) * 0.05)
    y2 = y1 + (np.random.randn(nstars) * 0.05)
    m2 = m1

    dr_tol = 1.0
    idx1, idx2, dr, dm = align.match(x1, y1, m1, x2, y2, m2, dr_tol)
    ngood = np.sum(idx1 == idx2)
    assert ngood == nstars
    assert np.sum(idx1 == idx2) == nstars

def test_match_lengths():
    """
    Test wehn one star list has a different length.

    """
    # First star list spread over 10 pixels.
    x1 = np.array([2.0, 4.0, 6.0, 8.0, 10.0])
    y1 = np.array([2.0, 4.0, 6.0, 8.0, 10.0])
    m1 = np.array([8.0, 10.0, 9.0, 12.0, 13.0])
    nstars = len(x1)

    # Second star list that has noise (of 0.05 pixel).
    x2 = np.append(x1 + (np.random.randn(nstars) * 0.05), [7.0])
    y2 = np.append(y1 + (np.random.randn(nstars) * 0.05), [2.0])
    m2 = m1

    # x1 first, then x2
    dr_tol = 1.0
    idx1, idx2, dr, dm = align.match(x1, y1, m1, x2, y2, m2, dr_tol)
    ngood = np.sum(idx1 == idx2)
    assert ngood == nstars
    assert np.sum(idx1 == idx2) == nstars

    # x2 first, then x1
    dr_tol = 1.0
    idx1, idx2, dr, dm = align.match(x2, y2, m2, x1, y1, m1, dr_tol)
    ngood = np.sum(idx1 == idx2)
    assert ngood == nstars
    assert np.sum(idx1 == idx2) == nstars



def test_match_missing():
    """
    Test when one star is missing.
    """
    # First star list spread over 10 pixels.
    x1 = np.array([2.0, 4.0, 6.0, 8.0, 10.0])
    y1 = np.array([2.0, 4.0, 6.0, 8.0, 10.0])
    m1 = np.array([8.0, 10.0, 9.0, 12.0, 13.0])
    nstars = len(x1)

    # Second star list that has noise (of 0.05 pixel).
    x2 = x1 + (np.random.randn(nstars) * 0.05)
    y2 = y1 + (np.random.randn(nstars) * 0.05)
    x2 = x2[1:]
    y2 = y2[1:]
    m2 = m1[1:]

    # x1 first, then x2
    dr_tol = 1.0
    idx1, idx2, dr, dm = align.match(x1, y1, m1, x2, y2, m2, dr_tol)
    ngood = np.sum(idx1 == (idx2 + 1))
    assert ngood == (nstars - 1)
    assert np.sum(idx1 == (idx2 + 1)) == (nstars - 1)

    # # x2 first, then x1
    dr_tol = 1.0
    idx1, idx2, dr, dm = align.match(x2, y2, m2, x1, y1, m1, dr_tol)
    ngood = np.sum((idx1 + 1) == idx2)
    assert ngood == (nstars - 1)
    assert np.sum((idx1 + 1) == idx2) == (nstars - 1)


def test_match_confused():
    """
    Test when there is confusion.
    The 5th and 6th source are confused by position, but very
    different in brightness.
    """
    # First star list spread over 10 pixels.
    x1 = np.array([2.0, 4.0, 6.0, 8.0, 10.0, 10.1])
    y1 = np.array([2.0, 4.0, 6.0, 8.0, 10.0, 9.9])
    m1 = np.array([8.0, 10.0, 9.0, 12.0, 13.0, 16.0])
    nstars = len(x1)

    # Second star list that has star 4 and star 5 confused... 
    x2 = x1.copy()
    y2 = y1.copy()
    x2[4] = 10.1
    x2[5] = 10.0
    y2[4] = 9.9
    y2[5] = 10.0
    m2 = m1

    # First check while allowing all delta-mag combinations. This results
    # in unrecoverably confusion since one is closer in position, but the other
    # is closer in brightness).
    dr_tol = 1.0
    idx1, idx2, dr, dm = align.match(x1, y1, m1, x2, y2, m2, dr_tol)
    ngood = np.sum(idx1 == idx2)
    assert ngood == (nstars - 2)
    assert np.sum(idx1 == idx2) == (nstars - 2)

    # Now check while allowing only small delta-mag combinations. This results
    # in recoverable confusion since one that is closer in position
    # has a large brightness offset (3 magnitudes).
    dr_tol = 1.0
    dm_tol = 1.0
    idx1, idx2, dr, dm = align.match(x1, y1, m1, x2, y2, m2, dr_tol, dm_tol=dm_tol)
    ngood = np.sum(idx1 == idx2)
    assert ngood == nstars
    assert np.sum(idx1 == idx2) == nstars


def test_match_duplicate():
    """
    Test when one stars in list 2 matches with two stars in list 1.
    """
    # First star list spread over 10 pixels.
    x1 = np.array([2.0, 4.0, 6.0, 8.0, 10.0, 10.1])
    y1 = np.array([2.0, 4.0, 6.0, 8.0, 10.0, 9.9])
    m1 = np.array([8.0, 10.0, 9.0, 12.0, 13.0, 13.1])
    nstars = len(x1)

    # Second star list that has star 4 and star 5 confused... 
    x2 = x1[:-1]
    y2 = y1[:-1]
    m2 = m1

    # First check while allowing all delta-mag combinations. This results
    # in unrecoverably confusion since one is closer in position, but the other
    # is closer in brightness).
    dr_tol = 1.0
    idx1, idx2, dr, dm = align.match(x1, y1, m1, x2, y2, m2, dr_tol)
    print idx1
    print idx2
    ngood = np.sum(idx1 == idx2)
    assert ngood == (nstars - 2)
    assert np.sum(idx1 == idx2) == (nstars - 2)





    

    


    

