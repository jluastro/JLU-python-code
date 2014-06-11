from . import jay
import numpy as np
import pdb
import time
import matplotlib.transforms
from skimage import transform
import math

def test_miracle_match50():
    """
    Test the default behaviour (and timing) of miracle_match50().
    """
    nstars = 1e4
    x1 = np.random.random(nstars) * 2048
    y1 = np.random.random(nstars) * 2048
    m1 = 9.0 + (np.random.random(nstars) * 13.0)

    ##########
    # Test #1
    ##########
    # Sanity check identical starlists
    x2 = x1
    y2 = y1
    m2 = m1

    t1 = time.time()
    N, x1m, y1m, m1m, x2m, y2m, m2m = jay.miracle_match_briteN(x1, y1, m1, x2, y2, m2, 50)
    t2 = time.time()
    print 'Test 1: Matching took {0:8.2f} sec'.format(t2 - t1)

    assert len(x1m) == 50
    assert x1m[0] == x2m[0]
    assert N > 1e3

    ##########
    # Test #2
    ##########
    # Perturb x positions by very small amounts
    x2 = x1 + (np.random.randn(nstars) * 0.001)  
    y2 = y1 + (np.random.randn(nstars) * 0.001)
    m2 = m1 + (np.random.randn(nstars) * 0.001)

    t1 = time.time()
    N, x1m, y1m, m1m, x2m, y2m, m2m = jay.miracle_match_briteN(x1, y1, m1, x2, y2, m2, 50)
    t2 = time.time()
    print 'Test 2: Matching took {0:8.2f} sec'.format(t2 - t1)

    assert abs(x1m[0] - x2m[0]) < 0.1
    assert abs(y1m[0] - y2m[0]) < 0.1
    assert abs(x1m[-1] - x2m[-1]) < 0.1
    assert abs(y1m[-1] - y2m[-1]) < 0.1
    assert N > 1e3


    ##########
    # Test #3
    ##########
    # Perturb x positions by big amounts
    x2 = x1 + (np.random.randn(nstars) * 0.1)  
    y2 = y1 + (np.random.randn(nstars) * 0.1)
    m2 = m1 + (np.random.randn(nstars) * 0.03)

    t1 = time.time()
    N, x1m, y1m, m1m, x2m, y2m, m2m = jay.miracle_match_briteN(x1, y1, m1, x2, y2, m2, 50)
    t2 = time.time()
    print 'Test 2: Matching took {0:8.2f} sec'.format(t2 - t1)

    assert abs(x1m[0] - x2m[0]) < 1
    assert abs(y1m[0] - y2m[0]) < 1
    assert abs(x1m[-1] - x2m[-1]) < 1
    assert abs(y1m[-1] - y2m[-1]) < 1
    assert N > 1e3


def test_assume_orient():
    """
    Test the basics of assume_orient().
    """
    nstars = 1e2
    x1 = np.random.random(nstars) * 2048
    y1 = np.random.random(nstars) * 2048
    m1 = 9.0 + (np.random.random(nstars) * 13.0)

    ##########
    # Test #1
    ##########
    # Sanity check identical starlists
    x2 = x1 + 101.0
    y2 = y1 + 50.2

    t1 = time.time()
    x2_new, y2_new = jay.assume_orient(x1, y1, x2, y2)
    t2 = time.time()
    print 'Test 1: took {0:8.2f} sec'.format(t2 - t1)

    # Offset is only good to 0.5 pixels.
    assert np.sum(x2_new - x1) < (nstars * 1)
    assert np.abs(x2_new[0] - x1[0]) <= 1

    ##########
    # Test #2
    ##########
    # Perturb x positions by very small amounts
    x2 = x1 + (np.random.randn(nstars) * 0.001) + 101.0
    y2 = y1 + (np.random.randn(nstars) * 0.001) + 50.2

    t1 = time.time()
    x2_new, y2_new = jay.assume_orient(x1, y1, x2, y2)
    t2 = time.time()
    print 'Test 2: took {0:8.2f} sec'.format(t2 - t1)

    # Offset is only good to 0.5 pixels.
    assert np.sum(x2_new - x1) < (nstars * 1)
    assert np.abs(x2_new[0] - x1[0]) <= 1

    
def test_find_offsets():
    """
    Test the basics of find_offsets().
    """
    nstars = 1e2
    x1 = np.random.random(nstars) * 2048
    y1 = np.random.random(nstars) * 2048
    m1 = 9.0 + (np.random.random(nstars) * 13.0)

    ##########
    # Test #1
    ##########
    # Sanity check identical starlists
    x2 = x1 + 101.0
    y2 = y1 + 50.2
    m2 = m1

    t1 = time.time()
    jay.find_offset(x1, y1, m1, x2, y2, m2)
    t2 = time.time()
    print 'Test 1: took {0:8.2f} sec'.format(t2 - t1)
    

    ##########
    # Test #2
    ##########
    # Add a rotation and make sure it returns
    # something sensible.
    t = matplotlib.transforms.Affine2D()
    t.rotate_deg(45.0)
    t.translate(101.0, 50.2)
    coords_in = np.array([x1, y1]).T
    coords_out = t.transform(coords_in)

    x2 = coords_out[:,0]
    y2 = coords_out[:,1]

    t1 = time.time()
    jay.find_offset(x1, y1, m1, x2, y2, m2)
    t2 = time.time()
    print 'Test 2: took {0:8.2f} sec'.format(t2 - t1)

def test_miracle_match_3d():
    """
    Test the basics of miracle_match_3d()
    """
    nstars = 1e4
    x1 = np.random.random(nstars) * 2048
    y1 = np.random.random(nstars) * 2048
    m1 = -14.0 + (np.random.random(nstars) * 10.0)

    ##########
    # Test #1
    ##########
    # Sanity check identical starlists
    x2 = x1
    y2 = y1
    m2 = m1

    print 'Test 1: BEGIN'
    t1 = time.time()
    N, x1m, y1m, m1m, x2m, y2m, m2m = jay.miracle_match_3d(x1, y1, m1, x2, y2, m2)
    t2 = time.time()
    print 'Test 1: Matching took {0:8.2f} sec'.format(t2 - t1)

    assert len(x1m) == len(x2m)
    assert len(x1m) == len(x1)
    assert x1m[0] == x2m[0]
    assert N > 1e3
    

    ##########
    # Test #2
    ##########
    # Add positional noise.
    x2 = x1 + (np.random.randn(len(x1)) * 0.2)
    y2 = y1 + (np.random.randn(len(x1)) * 0.2)
    m2 = m1 + (np.random.randn(len(x1)) * 0.1)

    print 'Test 2: BEGIN'
    t1 = time.time()
    N, x1m, y1m, m1m, x2m, y2m, m2m = jay.miracle_match_3d(x1, y1, m1, x2, y2, m2)
    t2 = time.time()
    print 'Test 2: Matching took {0:8.2f} sec'.format(t2 - t1)

    avg_offset = np.abs(x1m - x2m).mean()
    assert avg_offset < 0.25
    assert N > 1e3


    ##########
    # Test #3
    ##########
    # Randomize list #2
    x2 = x1 + (np.random.randn(len(x1)) * 0.2)
    y2 = y1 + (np.random.randn(len(x1)) * 0.2)
    m2 = m1

    indices = np.arange(len(x1))
    np.random.shuffle(indices)
    x2 = x2[indices]
    y2 = y2[indices]
    m2 = m2[indices]

    print 'Test 3: BEGIN'
    t1 = time.time()
    N, x1m, y1m, m1m, x2m, y2m, m2m = jay.miracle_match_3d(x1, y1, m1, x2, y2, m2)
    t2 = time.time()
    print 'Test 3: Matching took {0:8.2f} sec'.format(t2 - t1)

    assert m1m[0] == m2m[0] 
    avg_offset = np.abs(x1m - x2m).mean()
    assert avg_offset < 0.25
    assert N > 1e3

    ##########
    # Test #4
    ##########
    # Transform to make the match harder.
    trans = transform.AffineTransform(scale=[1.0, 1.0], rotation=math.radians(90.0),
                                      shear=0, translation=[30.0, -120])
    coords1 = np.array([x1, y1]).T
    coords2 = trans(coords1)
    x2 = coords2[:, 0] + (np.random.randn(len(x1)) * 0.2)
    y2 = coords2[:, 1] + (np.random.randn(len(x1)) * 0.2)
    m2 = m1

    print 'Test 4: BEGIN'
    t1 = time.time()
    N, x1m, y1m, m1m, x2m, y2m, m2m = jay.miracle_match_3d(x1, y1, m1, x2, y2, m2)
    t2 = time.time()
    print 'Test 4: Matching took {0:8.2f} sec'.format(t2 - t1)

    assert m1m[0] == m2m[0] 
    assert m1m[1] == m2m[1] 
    assert m1m[2] == m2m[2]   
    assert N > 1e3


def test_miracle_match_99():
    """
    Test the basics of miracle_match_99()
    """
    nstars = 1e4
    x1 = np.random.random(nstars) * 2048
    y1 = np.random.random(nstars) * 2048
    m1 = -14.0 + (np.random.random(nstars) * 10.0)

    ##########
    # Test #1
    ##########
    # Sanity check identical starlists
    x2 = x1
    y2 = y1
    m2 = m1

    print 'Test 1: BEGIN'
    t1 = time.time()
    N, x1m, y1m, m1m, x2m, y2m, m2m = jay.miracle_match_99(x1, y1, m1, x2, y2, m2)
    t2 = time.time()
    print 'Test 1: Matching took {0:8.2f} sec'.format(t2 - t1)

    assert len(x1m) == len(x2m)
    assert x1m[0] == x2m[0]
    assert N > 1e3
    

    ##########
    # Test #2
    ##########
    # Add positional noise.
    x2 = x1 + (np.random.randn(len(x1)) * 0.2)
    y2 = y1 + (np.random.randn(len(x1)) * 0.2)
    m2 = m1 + (np.random.randn(len(x1)) * 0.1)

    print 'Test 2: BEGIN'
    t1 = time.time()
    N, x1m, y1m, m1m, x2m, y2m, m2m = jay.miracle_match_99(x1, y1, m1, x2, y2, m2)
    t2 = time.time()
    print 'Test 2: Matching took {0:8.2f} sec'.format(t2 - t1)

    avg_offset = np.abs(x1m - x2m).mean()
    assert N > 1e3
    assert avg_offset < 0.25


    ##########
    # Test #3
    ##########
    # Randomize list #2
    x2 = x1 + (np.random.randn(len(x1)) * 0.2)
    y2 = y1 + (np.random.randn(len(x1)) * 0.2)
    m2 = m1

    indices = np.arange(len(x1))
    np.random.shuffle(indices)
    x2 = x2[indices]
    y2 = y2[indices]
    m2 = m2[indices]

    print 'Test 3: BEGIN'
    t1 = time.time()
    N, x1m, y1m, m1m, x2m, y2m, m2m = jay.miracle_match_3d(x1, y1, m1, x2, y2, m2)
    t2 = time.time()
    print 'Test 3: Matching took {0:8.2f} sec'.format(t2 - t1)

    assert m1m[0] == m2m[0] 
    assert m1m[1] == m2m[1] 
    assert m1m[2] == m2m[2] 
    avg_offset = np.abs(x1m - x2m).mean()
    assert avg_offset < 0.25
    assert N > 1e3

    ##########
    # Test #4
    ##########
    # Transform to make the match harder.
    trans = transform.AffineTransform(scale=[1.0, 1.0], rotation=math.radians(90.0),
                                      shear=0, translation=[30.0, -120])
    coords1 = np.array([x1, y1]).T
    coords2 = trans(coords1)
    x2 = coords2[:, 0] + (np.random.randn(len(x1)) * 0.2)
    y2 = coords2[:, 1] + (np.random.randn(len(x1)) * 0.2)
    m2 = m1

    print 'Test 4: BEGIN'
    t1 = time.time()
    N, x1m, y1m, m1m, x2m, y2m, m2m = jay.miracle_match_3d(x1, y1, m1, x2, y2, m2)
    t2 = time.time()
    print 'Test 4: Matching took {0:8.2f} sec'.format(t2 - t1)

    assert m1m[0] == m2m[0] 
    assert m1m[1] == m2m[1] 
    assert m1m[2] == m2m[2]   
    assert N > 1e3


def test_miracle_match_5C():
    """
    Test the basics of miracle_match_5C()
    """
    nstars = 1e4
    x1 = np.random.random(nstars) * 2048
    y1 = np.random.random(nstars) * 2048
    m1 = -14.0 + (np.random.random(nstars) * 10.0)

    # ##########
    # # Test #1
    # ##########
    # # Sanity check identical starlists
    # x2 = x1
    # y2 = y1
    # m2 = m1

    # print 'Test 1: BEGIN'
    # t1 = time.time()
    # N, x1m, y1m, m1m, x2m, y2m, m2m = jay.miracle_match_5C(x1, y1, m1, x2, y2, m2)
    # t2 = time.time()
    # print 'Test 1: Matching took {0:8.2f} sec'.format(t2 - t1)

    # assert len(x1m) == len(x2m)
    # assert x1m[0] == x2m[0]
    

    # ##########
    # # Test #2
    # ##########
    # # Add positional noise.
    # x2 = x1 + (np.random.randn(len(x1)) * 0.2)
    # y2 = y1 + (np.random.randn(len(x1)) * 0.2)
    # m2 = m1 + (np.random.randn(len(x1)) * 0.1)

    # print 'Test 2: BEGIN'
    # t1 = time.time()
    # N, x1m, y1m, m1m, x2m, y2m, m2m = jay.miracle_match_5C(x1, y1, m1, x2, y2, m2)
    # t2 = time.time()
    # print 'Test 2: Matching took {0:8.2f} sec'.format(t2 - t1)

    # avg_offset = np.abs(x1m - x2m).mean()
    # assert avg_offset < 0.25


    ##########
    # Test #3
    ##########
    # Randomize list #2
    x2 = x1 + (np.random.randn(len(x1)) * 0.2)
    y2 = y1 + (np.random.randn(len(x1)) * 0.2)
    m2 = m1

    indices = np.arange(len(x1))
    np.random.shuffle(indices)
    x2 = x2[indices]
    y2 = y2[indices]
    m2 = m2[indices]

    print 'Test 3: BEGIN'
    t1 = time.time()
    N, x1m, y1m, m1m, x2m, y2m, m2m = jay.miracle_match_5C(x1, y1, m1, x2, y2, m2)
    t2 = time.time()
    print 'Test 3: Matching took {0:8.2f} sec'.format(t2 - t1)

    assert m1m[0] == m2m[0] 
    assert m1m[1] == m2m[1] 
    assert m1m[2] == m2m[2] 
    avg_offset = np.abs(x1m - x2m).mean()
    assert avg_offset < 0.25
    assert N > 2

    # ##########
    # # Test #4
    # ##########
    # # Transform to make the match harder.
    # trans = transform.AffineTransform(scale=[1.0, 1.0], rotation=math.radians(90.0),
    #                                   shear=0, translation=[30.0, -120])
    # coords1 = np.array([x1, y1]).T
    # coords2 = trans(coords1)
    # x2 = coords2[:, 0] + (np.random.randn(len(x1)) * 0.2)
    # y2 = coords2[:, 1] + (np.random.randn(len(x1)) * 0.2)
    # m2 = m1

    # print 'Test 4: BEGIN'
    # t1 = time.time()
    # N, x1m, y1m, m1m, x2m, y2m, m2m = jay.miracle_match_5C(x1, y1, m1, x2, y2, m2)
    # t2 = time.time()
    # print 'Test 4: Matching took {0:8.2f} sec'.format(t2 - t1)

    # assert m1m[0] == m2m[0] 
    # assert m1m[1] == m2m[1] 
    # assert m1m[2] == m2m[2]
    # assert N > 2


def test_purify_buoy():
    nstars = 1e3
    x1 = np.random.random(nstars) * 2048
    y1 = np.random.random(nstars) * 2048

    ##########
    # Test #1
    ##########
    # Transform to make the match harder. Add some noise.
    trans = transform.AffineTransform(scale=[1.0, 1.0], rotation=math.radians(90.0),
                                      shear=0, translation=[30.0, -120])
    coords1 = np.array([x1, y1]).T
    coords2 = trans(coords1)
    x2 = coords2[:, 0] + (np.random.randn(len(x1)) * 0.2)
    y2 = coords2[:, 1] + (np.random.randn(len(x1)) * 0.2)

    print 'Test 1: BEGIN'
    t1 = time.time()
    x1m, y1m, x2m, y2m = jay.purify_buoy(x1, y1, x2, y2, 0.4)
    t2 = time.time()
    print 'Test 1: Matching took {0:8.2f} sec'.format(t2 - t1)

    # Confirm that at least half of the stars are retained.
    assert len(x1m) > (0.5 * nstars)

    # Check that the original positions were maintained
    assert x1m[0] in x1
    assert y1m[0] in y1
    assert x2m[0] in x2
    assert y2m[0] in y2

    # Check that the lengths of the arrays are correct
    assert len(x1m) == len(y1m)
    assert len(x1m) == len(x2m)
    assert len(x1m) == len(y2m)


def test_incorp_buoy():
    nstars = 1e4
    x1 = np.random.random(nstars) * 2048
    y1 = np.random.random(nstars) * 2048
    m1 = np.random.random(nstars) * 5.0 - 15.0

    ##########
    # Test #1
    ##########
    # Transform to make the match harder. Add some noise.
    trans = transform.AffineTransform(scale=[1.0, 1.0], rotation=math.radians(90.0),
                                      shear=0, translation=[30.0, -120])
    coords1 = np.array([x1, y1]).T
    coords2 = trans(coords1)
    x2 = coords2[:, 0] + (np.random.randn(len(x1)) * 0.2)
    y2 = coords2[:, 1] + (np.random.randn(len(x1)) * 0.2)
    m2 = m1

    # Select out the brightest 100.... they are already matched.
    x1bt = x1[:100]
    y1bt = y1[:100]
    m1bt = m1[:100]
    x2bt = x2[:100]
    y2bt = y2[:100]
    m2bt = m2[:100]

    print 'Test 1: BEGIN'
    t1 = time.time()
    x1m, y1m, m1m, x2m, y2m, m2m = jay.incorp_buoy(x1bt, y1bt, x2bt, y2bt,
                                                   x1, y1, m1, x2, y2, m2, 0.4)
    t2 = time.time()
    print 'Test 1: Matching took {0:8.2f} sec'.format(t2 - t1)

    # Confirm that at least half of the stars are retained.
    assert len(x1m) > (0.5 * nstars)

    # Check that the original positions were maintained
    assert x1m[0] in x1
    assert y1m[0] in y1
    assert x2m[0] in x2
    assert y2m[0] in y2

    # Check that the lengths of the arrays are correct
    assert len(x1m) == len(y1m)
    assert len(x1m) == len(x2m)
    assert len(x1m) == len(y2m)

    # Check that matches were made correctly (via magnitudes)
    assert m1m[0] == m2m[0]
    assert m1m[10] == m2m[10]

    
