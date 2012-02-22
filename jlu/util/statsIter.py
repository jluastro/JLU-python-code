import numpy as np
import math

def mean(array, hsigma=None, lsigma=None, iter=0, verbose=False):
    if hsigma==None and lsigma==None:
        return arr.mean()

    # Make a copy of the array (flattened) since we are going
    # to modify it as we go.
    arr = array.flatten().copy()

    lo = arr.min()
    hi = arr.max()

    if verbose:
        print '    Original: mean = %.3g' % (arr.mean())

    for ii in range(iter):
        mean = arr.mean()
        sigma = arr.std()

        if lsigma:
            lo = mean - (lsigma * sigma)
        if hsigma:
            hi = mean + (hsigma * sigma)

        idx = np.where((arr >= lo) & (arr <= hi))
        arr = arr[idx]

        new_mean = arr.mean()

        if verbose:
            print 'Iteration %2d: mean = %.3g' % (ii, new_mean)

    return new_mean

def std(array, hsigma=None, lsigma=None, iter=0, verbose=False):
    if hsigma==None and lsigma==None:
        return arr.mean()

    # Make a copy of the array (flattened) since we are going
    # to modify it as we go.
    arr = array.flatten().copy()

    lo = arr.min()
    hi = arr.max()

    if verbose:
        print '    Original: std = %.3g' % (arr.std())

    for ii in range(iter):
        mean = arr.mean()
        sigma = arr.std()

        if lsigma:
            lo = mean - (lsigma * sigma)
        if hsigma:
            hi = mean + (hsigma * sigma)
        idx = np.where((arr >= lo) & (arr <= hi))
        arr = arr[idx]

        new_std = arr.std()

        if verbose:
            print 'Iteration %2d: std = %.3g' % (ii, new_std)

    return new_std
