import numpy as np

def order_by_brite(xi, yi, mi, Nout, verbose=True):
    # Length of the input starlists.
    Nin = len(xi)
    if verbose:
        print 'order_by_brite: nstars in =', Nin
        print 'order_by_brite: desired nstars out =', Nout

    if Nout > Nin:
        Nout = Nin

    if verbose:
        print 'order_by_brite: return nstars out =', Nout

    sdx = mi.argsort()
    brite = sdx[:Nout]

    if verbose:
        print 'order_by_brite: faintest star m =', mi[brite[-1]]

    xo = xi[brite]
    yo = yi[brite]
    mo = mi[brite]

    return xo, yo, mo
