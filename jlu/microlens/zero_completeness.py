import numpy as np
import pylab as plt
from scipy import stats


def prove_ref_wrong():
    """
    The referee for Casey's paper insists that they are 
    able to correct for the observational incompleteness
    'perfectly'! This is physically not possible. It is lame;
    but, we need to show this in a figure to somehow convince
    them that this doesn't work. 
    """
    n_stars = int(1e4)
    
    # Artificial luminosity function is just a normal
    # distribution. Note this is the "perfect" sample
    # with no observational incomplenetess.
    mags = stats.norm.rvs(size=n_stars, loc=19, scale=8)

    # Draw tE distribution from a normal that shifts
    # to longer timescales with fainter stars... this
    # is just arbitrary to show the effects.
    med_tE_v_mag = 1.4 + (0.2 * (mags - 19) / 9)
    std_tE_v_mag = 0.3 + (0.1 * (mags - 5) / 9)
        
    log_tE = stats.norm.rvs(size=n_stars, loc=med_tE_v_mag, scale=std_tE_v_mag)

    tE = 10**log_tE

    # Make a magnitude cut at I=20.
    det = np.where(mags < 20)[0]

    # Make a completeness curve as a function of tE:
    logbins = np.logspace(np.log10(tE.min()), np.log10(tE.max()), 50)

    comp_v_tE = np.zeros(len(log_bins) - 1, dtype=float)
    weights = np.ones(
    for bb in range(len(logbins)-1):
        all_in_bin = np.where((tE > logbins[bb]) & (tE <= logbins[bb+1]))[0]
        det_in_bin = np.where((tE > logbins[bb]) & (tE <= logbins[bb+1]))[0]
        comp_v_tE[bb] = 1.0 * all_in_bin / det_in_bin

    plt.figure(1) 
    plt.clf()
    plt.semilogy(mags, tE, 'k.', alpha=0.2, color='black', label='All')
    plt.semilogy(mags[det], tE[det], 'r.', alpha=0.2, color='red', label='Detected')
    plt.xlabel('Mag')
    plt.ylabel('tE')
    plt.legend()

    plt.figure(2)
    plt.clf()
    plt.hist(tE, bins=logbins, color='black', histtype='step', label='All')
    (nd, bd, pd) = plt.hist(tE[det], bins=logbins, color='red', histtype='step', label='Detected')
    plt.gca().set_xscale('log')
    plt.gca().set_yscale('log')
    plt.xlim(0.3, 3e3)
    #plt.ylim(1e-3, 1e-1)
    plt.xlabel('tE (days)')
    plt.ylabel('Probability Density')
    plt.legend()
    

    return


    
