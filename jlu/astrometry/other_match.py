import gmatch
from jlu.astrometry import ord_brite
import numpy as np
import logging

def miracle_match_briteN_slow(xin1, yin1, min1, xin2, yin2, min2, Nbrite, verbose=False):
    print ''
    print '  miracle_match:  '
    print '  miracle_match: use brightest', Nbrite
    print '  miracle_match:  '

    # Get/check the lengths of the two starlists
    nin1 = len(xin1)
    nin2 = len(xin2)

    if (nin1 < Nbrite) or (nin2 < Nbrite):
        print 'You need at least {0} to '.format(Nbrite)
        print 'find the matches...'
        print 'NIN1: ', nin1
        print 'NIN2: ', nin2
        return

    x1, y1, m1 = ord_brite.order_by_brite(xin1, yin1, min1, Nbrite, verbose=verbose)
    x2, y2, m2 = ord_brite.order_by_brite(xin2, yin2, min2, Nbrite, verbose=verbose)
    
    print '  miracle_match: '
    print '  miracle_match: ORD_BRITE: '
    print '  miracle_match: '
    print '  miracle_match: '
    print '  miracle_match: DO V/VMAX-type search...'
    print '  miracle_match: '

    cat1 = np.array((x1, y1)).T
    cat2 = np.array((x2, y2)).T

    # create logger with 'spam_application'
    gmatch._logger.setLevel(logging.DEBUG)
    # create console handler with a higher log level
    ch = logging.StreamHandler()
    ch.setLevel(logging.ERROR)
    # create formatter and add it to the handlers
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    ch.setFormatter(formatter)
    # add the handlers to the logger
    gmatch._logger.addHandler(ch)    

    matches = gmatch.gmatch(cat1, cat2, eps=1e3, reject_scale=1.1)

    idx1 = matches[0]
    idx2 = matches[1]

    return x1[idx1], y1[idx1], m1[idx1], x2[idx2], y2[idx2], m2[idx2]
    
