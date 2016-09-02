import numpy as np
import pylab as py
import pdb
import math
import ephem
import os
from jlu.observe import skycalc

def new_young_stars():
    """
    Calculate the expected number of new young stars in the
    proposed fields of view.
    """
    fov = [2.27, 3.12]  # arcsec

    field_name = ['C', 'E', 'S', 'W', 'SE', 'N', 'NE', 'SW', 'NW',
                  'E2-1', 'E2-2', 'E2-3', 'E3-1', 'E3-2', 'E3-3',
                  'E4-1', 'E4-2', 'E4-3',
                  'S2-1', 'S2-2', 'S2-3', 'S3-1', 'S3-2', 'S3-3',
                  'S4-1', 'S4-2', 'S4-3',
                  'N1-1', 'N1-2', 'NE1-1']
    field_x = np.array([0.0, 2.88, -0.69, -2.70, 1.67, 0.33, 2.55, -2.9, -1.99,
                        5.43, 4.8, 4.16, 8.59, 7.94, 7.31,
                        11.73, 11.08, 10.44,
                        0.69, -1.49, -3.80, -0.03, -2.35, -4.66,
                        -0.87, -3.19, -5.50,
                        2.73, 5.36, 10.08])
    field_y = np.array([0.0, -0.67, -2.00, 0.74, -2.23, 2.01, 1.27, -1.12, 2.42,
                        0.99, -1.4, -3.75, 0.15, -2.21, -4.57,
                        -0.68, -3.04, -5.41,
                        -4.16, -4.21, -3.59, -7.95, -7.31, -6.67,
                        -11.03, -10.41, -9.79,
                        8.67, 7.24, 3.80])
    observed = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1,
                         1, 1, 1, 1, 1, 1,
                         1, 1, 1,
                         1, 0, 0, 0, 1, 0,
                         1, 1, 0,
                         1, 1, 1])
    gcows_south = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0,
                         0, 0, 0, 0, 0, 0,
                         0, 0, 0,
                         1, 1, 1, 1, 1, 1,
                         1, 1, 1,
                         0, 0, 0])
    

    # These are very rough.
    field_xsize = np.array([2.4, 2.2, 2.2, 2.2, 2.2, 2.2, 2.2, 2.2, 2.2,
                            3.1, 3.1, 3.1, 3.1, 3.1, 3.1,
                            3.1, 3.1, 3.1,
                            2.3, 2.3, 2.3, 2.3, 2.3, 2.3,
                            2.3, 2.3, 2.3,
                            2.9, 2.9, 2.9])
    field_ysize = np.array([1.8, 1.6, 1.6, 1.6, 1.6, 1.6, 1.6, 1.6, 1.6,
                            2.3, 2.3, 2.3, 2.3, 2.3, 2.3,
                            2.3, 2.3, 2.3,
                            3.1, 3.1, 3.1, 3.1, 3.1, 3.1,
                            3.1, 3.1, 3.1,
                            2.9, 2.9, 2.9])
    
    field_r = np.hypot(field_x, field_y)
    field_r[0] = 0.55
    field_area = field_xsize * field_ysize
    r_ref = 1.0 # arcsec

    obs = np.where(observed == 1)[0]
    noobs = np.where(observed == 0)[0]
    gcowsS = np.where(gcows_south == 1)[0]
    nogcowsS = np.where(gcows_south == 0)[0]
    
    yng_surf_dens = 3 * (field_r / r_ref)**-0.93
    yng_number = yng_surf_dens * field_area

    py.clf()
    py.loglog(field_r, yng_surf_dens, 'k.')

    for ii in range(len(field_area)):
        print '{0:5s}  {1:4.1f}  {2:5.2f}  {3:4.1f}'.format(field_name[ii], yng_number[ii], field_area[ii], field_r[ii])

    N_obs = yng_number[obs].sum()
    N_noobs = yng_number[noobs].sum()
    N_obs_disk = (N_obs - 11) * 0.2
    N_obs_off = N_obs - 11 - N_obs_disk

    N_obs_noS = yng_number[nogcowsS].sum()
    N_obs_noS_disk = (N_obs_noS - 11) * 0.2
    N_obs_noS_off = N_obs_noS - 11 - N_obs_noS_disk
    N_gcowsS = yng_number[gcowsS].sum()

    print 'N_yng[obs] = ', N_obs
    print 'N_yng[obs,disk] =', N_obs_disk
    print 'N_yng[obs,off disk,off Sclust] =', N_obs_off
    print 'N_yng[no-obs] = ', N_noobs
    print ''
    print 'N_yng[obs,noS] =', N_obs_noS
    print 'N_yng[obs,noS,disk] =', N_obs_noS_disk
    print 'N_yng[obs,noS,off disk, off Sclust] =', N_obs_noS_off
    print 'N_yng[gcows_south] = ', N_gcowsS
    
    return

def gc_plot_airmass_moon():
    # coordinates to GC
    ra = "17:45:40.04"
    dec = "-29:00:28.12"
    months = np.array([4, 5, 6, 7])
    days = np.array([10, 10, 10, 30])
    outdir = '/Users/jlu/doc/proposals/keck/uc/17A/'

    skycalc.plot_airmass(ra, dec, 2017, months, days, outfile=outdir + 'gc_airmass.png')
    skycalc.plot_moon(ra, dec, 2017, months, outfile=outdir + 'gc_moon.png')

    return
