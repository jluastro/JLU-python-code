import numpy as np
from matplotlib import pyplot as py
from astropy.io import ascii
from astropy.coordinates import ICRS, Galactic
from astropy import units as u
import pdb

#laserStrehl = 0.3  # 20%-ile seeing conditions
laserStrehl = 0.2  # 50%-ile seeing conditions

def plot_strehl_skycov(b_range=None):
    t = ascii.read('SkyCov_AllSky_final.dat')

    ra = t['rh'] + (t['rm'] / 60.) + (t['rs'] / 3600.)
    dec = t['dd'] + (t['dm'] / 60.) + (t['ds'] / 3600.)

    coords = ICRS(ra, dec, unit=(u.hour, u.degree))

    # Convert to Galactic coordinates
    b = coords.galactic.b.degree

    if b_range != None and len(b_range) == 2:
        b_abs = np.abs(b)
        idx = np.where((b_abs > b_range[0]) & (b_abs <= b_range[1]))[0]
        print 'Trimming down to {0} points'.format(len(idx))

        t = t[idx]
        ra = ra[idx]
        dec = dec[idx]
        coords = coords[idx]
        b = b[idx]
    
    ngsStrehl = np.copy(t['Savg']) / 100.
    ngsStrehl_rms = np.copy(t['Srms']) / 100.
    strehl = ngsStrehl * laserStrehl
    strehl_lo = (ngsStrehl - ngsStrehl_rms) * laserStrehl
    strehl_hi = (ngsStrehl + ngsStrehl_rms) * laserStrehl

    bins = np.arange(0.0, 0.3001, 0.02)

    n, s = np.histogram(strehl, bins=bins, normed=True)
    n_lo, s_lo = py.np.histogram(strehl_lo, bins=bins, normed=True)
    n_hi, s_hi = py.np.histogram(strehl_hi, bins=bins, normed=True)
    
    sr = s[::-1]
    sr_mid = sr[:-1] + (np.diff(sr)/2.0)

    n = n[::-1]
    n_cumsum = n.cumsum()
    sky_cov = n_cumsum / n_cumsum[-1]

    n_lo = n_lo[::-1]
    n_lo_cumsum = n_lo.cumsum()
    sky_cov_lo = n_lo_cumsum / n_lo_cumsum[-1]

    n_hi = n_hi[::-1]
    n_hi_cumsum = n_hi.cumsum()
    sky_cov_hi = n_hi_cumsum / n_hi_cumsum[-1]

    py.clf()
    py.plot(sky_cov, sr_mid, 'k.-')
    py.plot(sky_cov_lo, sr_mid, 'k--')
    py.plot(sky_cov_hi, sr_mid, 'k--')
    py.xlabel('Sky Coverage')
    py.ylabel('Strehl')
    py.ylim(0, 1)
    py.xlim(0, 1)
    outfile = '/Users/jlu/doc/proposals/nsf/msip/2014/gems_sky_coverage_MC'
    if b_range != None:
        py.title('b = [{0} - {1}]'.format(b_range[0], b_range[1]))
        outfile += '_b_{0}_{1}'.format(b_range[0], b_range[1])
    outfile += '.png'
    py.savefig(outfile)
    
    print '{0:8s}  {1:8s}'.format('SkyCov', 'Strehl')
    for ii in range(len(sr_mid)):
        print '{0:6.2f}  {1:6.2f}'.format(sky_cov[ii], sr_mid[ii])
    
    return



def plot_strehl_skycov_vs_b():
    t = ascii.read('SkyCov_AllSky_final.dat')

    ra = t['rh'] + (t['rm'] / 60.) + (t['rs'] / 3600.)
    dec = t['dd'] + (t['dm'] / 60.) + (t['ds'] / 3600.)

    coords = ICRS(ra, dec, unit=(u.hour, u.degree))

    # Convert to Galactic coordinates
    b = coords.galactic.b.degree

    ngsStrehl = np.copy(t['Savg']) / 100.
    ngsStrehl_rms = np.copy(t['Srms']) / 100.
    strehl = ngsStrehl * laserStrehl
    strehl_lo = (ngsStrehl - ngsStrehl_rms) * laserStrehl
    strehl_hi = (ngsStrehl + ngsStrehl_rms) * laserStrehl

    b_sets = [[0, 10], [25, 35], [55, 65], [80, 90]]
    b_label = ['b=0 deg', 'b=30 deg', 'b=60 deg', 'b=90 deg']
    bins = np.arange(0.0, 0.3001, 0.02)

    py.clf()
    for ib in range(len(b_sets)):
        b_range = b_sets[ib]
        idx = np.where((b > b_range[0]) & (b <= b_range[1]))[0]
    
        n, s = np.histogram(strehl[idx], bins=bins, normed=True)
    
        sr = s[::-1]
        sr_mid = sr[:-1] + (np.diff(sr)/2.0)

        n = n[::-1]
        n_cumsum = n.cumsum()
        sky_cov = n_cumsum / n_cumsum[-1]

        py.plot(sky_cov, sr_mid, label=b_label[ib], linewidth=2)

    py.legend()
    py.xlabel('Sky Coverage')
    py.ylabel('Strehl')
    py.ylim(0, 1)
    py.xlim(0, 1)
    py.savefig('/Users/jlu/doc/proposals/nsf/msip/2014/gems_sky_coverage_MC_vs_b.png')

    return


def plot_strehl_vs_nstars():
    t = ascii.read('SkyCov_AllSky_final.dat')

    ngsStrehl = np.copy(t['Savg']) / 100.
    strehl = ngsStrehl * laserStrehl

    py.clf()
    py.plot(ngsStrehl, t['N'], 'k.')
    py.ylim(0, 30)
    py.xlim(0, 1)
    py.xlabel('NGS TT Strehl')
    py.ylabel("Number of Guide Stars in R<1'")
    py.savefig('/Users/jlu/doc/proposals/nsf/msip/2014/gems_ngsstrehl_vs_nstars.png')
    
