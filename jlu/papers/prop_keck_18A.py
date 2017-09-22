import numpy as np
import pylab as py
import pdb
import math
import ephem
import os
from jlu.observe import skycalc

def plot_airmass_moon():
    # coordinates to center of OGLE field BLG502 (2017 objects)
    ra = "18:00:00"
    dec = "-30:00:00"
    months = np.array([4, 5, 6, 7])
    days = np.array([10, 10, 10, 30])
    #outdir = '/Users/jlu/doc/proposals/keck/uc/18A/'
    outdir = '/Users/fatima/Desktop/'
    
    # Keck 2
    skycalc.plot_airmass(ra, dec, 2018, months, days, 'keck2', outfile=outdir + 'microlens_airmass_keck2.png', date_idx=0)
    skycalc.plot_moon(ra, dec, 2018, months, outfile=outdir + 'microlens_moon.png')

    # Keck 1
    skycalc.plot_airmass(ra, dec, 2018, months, days, 'keck1', outfile=outdir + 'microlens_airmass_keck1.png', date_idx=0)
    
    return
