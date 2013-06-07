import pylab as py
import numpy as np
from jlu.wd1 import synthetic as syn
import math
import atpy
from jlu.util import statsIter
from jlu.util import constants as cc
import ephem

scale = 61.0

plotDir = '/u/jlu/doc/proposals/hst/cycle21/g286/plots/'


def hst_wfc3_pos_error():
    # Load up observed data:
    wd1_data = '/u/jlu/data/Wd1/hst/from_jay/EXPORT_WEST1.2012.02.04/wd1_catalog.fits'
    data = atpy.Table(wd1_data)

    # Determine median astrometric and photometric error as a function of
    # magnitude for both F814W and F125W. Use Y as a proxy for astrometric error.
    magBinSize = 0.25
    magBinCenter = np.arange(12, 28, magBinSize)
    yerr160w = np.zeros(len(magBinCenter), dtype=float)
    yerr125w = np.zeros(len(magBinCenter), dtype=float)
    merr160w = np.zeros(len(magBinCenter), dtype=float)
    merr125w = np.zeros(len(magBinCenter), dtype=float)

    for ii in range(len(magBinCenter)):
        mlo = magBinCenter[ii] - (magBinSize/2.0)
        mhi = magBinCenter[ii] + (magBinSize/2.0)

        idx160 = np.where((data.mag160 >= mlo) & (data.mag160 < mhi) &
                          (np.isnan(data.y2005_e) == False))[0]
        idx125 = np.where((data.mag125 >= mlo) & (data.mag125 < mhi) &
                          (np.isnan(data.y2010_e) == False))[0]

        if len(idx160) > 1:
            yerr160w[ii] = statsIter.mean(data.y2005_e[idx160], hsigma=3, lsigma=5, iter=10)
            merr160w[ii] = statsIter.mean(data.mag160_e[idx160], hsigma=3, lsigma=5, iter=10)
        else:
            yerr160w[ii] = np.nan
            merr160w[ii] = np.nan

        if len(idx125) > 1:
            yerr125w[ii] = statsIter.mean(data.y2010_e[idx125], hsigma=3, lsigma=5, iter=10)
            merr125w[ii] = statsIter.mean(data.mag125_e[idx125], hsigma=3, lsigma=5, iter=10)
        else:
            yerr125w[ii] = np.nan
            merr125w[ii] = np.nan



            
    py.clf()
    py.plot(magBinCenter, yerr160w*scale, 'b.', ms=10)
    py.xlabel('Magnitude')
    py.ylabel('F160W Positional Error (mas)')
    py.ylim(0, 5)
    py.xlim(12, 20)
    py.savefig(plotDir + 'avg_poserr_f160w.png')

