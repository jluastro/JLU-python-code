import numpy as np
import itertools
import time
import pdb

def test_sum_two_lists_all_pairs():
    x1 = np.array(np.random.random(1e3) * 0.1, dtype=np.float16)
    x2 = np.array(np.random.random(5e3) * 1.0, dtype=np.float16)
    y1 = np.array(np.random.random(1e3) * 0.1, dtype=np.float16)
    y2 = np.array(np.random.random(5e3) * 1.0, dtype=np.float16)

    N1 = len(x1)
    N2 = len(x2)

    # Test 1
    t1 = time.time()
    indices = np.indices((N1, N2))
    dx = np.ravel(x1[indices[0]] - x2[indices[1]])
    dy = np.ravel(y1[indices[0]] - y2[indices[1]])
    t2 = time.time()
    print 'Test 1 took {0:8.2f} sec     shape = {1}'.format(t2 - t1, dx.shape)

    # Test 2
    t1 = time.time()
    pairs = np.meshgrid(x1, x2)
    dx = np.ravel(pairs[0] - pairs[1])
    pairs = np.meshgrid(y1, y2)
    dy = np.ravel(pairs[0] - pairs[1])
    t2 = time.time()
    print 'Test 2 took {0:8.2f} sec     shape = {1}'.format(t2 - t1, dx.shape)

    # Test 3 -- fastest
    t1 = time.time()
    pairs = cartesian_product((x1, x2))
    dx = pairs[:,0] - pairs[:,1]
    pairs = cartesian_product((y1, y2))
    dy = pairs[:,0] - pairs[:,1]
    t2 = time.time()
    print 'Test 3 took {0:8.2f} sec     shape = {1}'.format(t2 - t1, dx.shape)

    # Test 4 -- same thing as test 3... slightly slower for some reason
    # t1 = time.time()
    # pairs = np.empty([N1, N2, 2])
    # for i, a in enumerate(np.ix_(*(x1, x2))):
    #     pairs[...,i] = a
    # dx = (pairs[:,:,0] - pairs[:,:,1]).reshape(-1)
    # t2 = time.time()
    # print 'Test 4 took {0:8.2f} sec     shape = {1}'.format(t2 - t1, dx.shape)

    # Test 5 -- BAD!!!
    # t1 = time.time()
    # pairs = np.array(list(itertools.product(x1, x2)))
    # dx = pairs[:,0] - pairs[:,1]
    # t2 = time.time()
    # print 'Test 5 took {0:8.2f} sec     shape = {1}'.format(t2 - t1, dx.shape)


def cartesian_product(arrays):
    la = len(arrays)
    arr = np.empty([len(a) for a in arrays] + [la])
    for i, a in enumerate(np.ix_(*arrays)):
        arr[...,i] = a
    return arr.reshape(-1, la)
