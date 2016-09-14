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
    cnt = len(arr)

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
            print 'Iteration %2d: mean = %.3g (cut %d of %d)' % \
                (ii, new_mean, cnt-len(arr), cnt)

    return new_mean

def std(array, hsigma=None, lsigma=None, iter=0, verbose=False):
    if hsigma==None and lsigma==None:
        return arr.std()

    # Make a copy of the array (flattened) since we are going
    # to modify it as we go.
    arr = array.flatten().copy()

    lo = arr.min()
    hi = arr.max()
    cnt = len(arr)

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
            print 'Iteration %2d: std = %.3g (cut %d of %d)' % \
                (ii, new_std, cnt-len(arr), cnt)

    return new_std

def mean_std_clip(indata, clipsig=3.0, maxiter=5, converge_num=0.02,
                  verbose=0, return_nclip=False):
    """
    Computes an iteratively sigma-clipped mean on a
    data set. Clipping is done about median, but mean
    is returned.

    .. note:: MYMEANCLIP routine from ACS library.

    :History:
        * 21/10/1998 Written by RSH, RITSS
        * 20/01/1999 Added SUBS, fixed misplaced paren on float call, improved doc. RSH
        * 24/11/2009 Converted to Python. PLL.

    Examples
    --------
    >>> mean, sigma = mean_std_clip(indata)

    Parameters
    ----------
    indata: array_like
        Input data.

    clipsig: float
        Number of sigma at which to clip.

    maxiter: int
        Ceiling on number of clipping iterations.

    converge_num: float
        If the proportion of rejected pixels is less than
        this fraction, the iterations stop.

    verbose: {0, 1}
        Print messages to screen?

    Returns
    -------
    mean: float
        N-sigma clipped mean.

    sigma: float
        Standard deviation of remaining pixels.

    """
    # Flatten array
    skpix = indata.reshape( indata.size, )

    ct = indata.size
    iter = 0
    c1 = 1.0
    c2 = 0.0

    while (c1 >= c2) and (iter < maxiter):
        lastct = ct
        medval = np.median(skpix)
        sig = np.std(skpix)
        wsm = np.where( abs(skpix-medval) < clipsig*sig )
        ct = len(wsm[0])
        if ct > 0:
            skpix = skpix[wsm]

        c1 = abs(ct - lastct)
        c2 = converge_num * lastct
        iter += 1

    mean  = np.mean( skpix )
    sigma = np.std( skpix )

    if verbose:
        prf = 'MEANCLIP:'
        print '%s %.1f-sigma clipped mean' % (prf, clipsig)
        print '%s Mean computed in %i iterations' % (prf, iter)
        print '%s Clipped %i of %i stars' % (prf, indata.size - len(skpix), indata.size)
        print '%s Mean = %.6f, sigma = %.6f' % (prf, mean, sigma)

    if return_nclip:
        return mean, sigma, len(skpix)
    else:
        return mean, sigma
