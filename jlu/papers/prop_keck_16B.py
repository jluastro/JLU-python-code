import numpy as np
import pylab as py
import pdb
import math
import ephem
import os
from jlu.observe import skycalc

def gc_plot_airmass_moon():
    # coordinates to GC
    ra = "05:35:17.30"
    dec = "-05:23:28.0"
    months = np.array([10, 11, 12])
    days = np.array([10, 10, 10])
    outdir = '/Users/jlu/doc/proposals/ifa/16B/'

    skycalc.plot_airmass(ra, dec, 2016, months, days, outfile=outdir + 'orion_airmass.png')
    skycalc.plot_moon(ra, dec, 2016, months, outfile=outdir + 'orion_moon.png')

    return
