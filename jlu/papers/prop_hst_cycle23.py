import numpy as np
import pylab as py
from astropy.table import Table
from jlu.arches import synthetic as syn
from jlu.arches.mwhosek import arches_completeness_2015_03 as completeness
from jlu.arches.mwhosek import arches_profile_2015_03 as profile
import matplotlib.patches as patches
from scipy import interpolate
import pdb
import math
from scipy import optimize

prop_dir = '/Users/jlu/doc/proposals/hst/cycle23/arches/'

def plot_arches_ast_err():
    catalog = '/u/mwhosek/Desktop/699-2/ks2/submitted_2015_04/ks2_poly/'
    catalog += 'catalog.fits'


    # Read in data table
    t = Table.read(catalog)

    mag = t['m_2010_f153m']
    vxe = t['fit_vxe'] * 120.
    vye = t['fit_vye'] * 120.

    ve = np.hypot(vxe, vye)

    # Read in isochrone
    iso = syn.load_isochrone(logAge=6.4, AKs=2.4, distance=8000)

    # Make an array of magnitudes and convert to masses.
    mag_arr = np.arange(12, 24, 0.05)
    mass_arr = np.zeros(len(mag_arr), dtype=float)

    for ii in range(len(mag_arr)):
        dm = np.abs(iso.mag153m - mag_arr[ii])
        dm_idx = dm.argmin()

        mass_arr[ii] = iso.M[dm_idx]


    # Establish new limits
    lim_mag = 21.7
    lim_pme = 1.95

    good = np.where((mag < lim_mag) & (ve < lim_pme))[0]
    bad = np.where((mag >= lim_mag) | (ve >= lim_pme))[0]
    
    py.clf()
    py.subplots_adjust(top=0.88)
    py.plot(mag[good], ve[good], 'k.', ms=2)
    py.plot(mag[bad], ve[bad], 'k.', ms=2, alpha=0.3)
    py.xlim(13, 23)
    py.ylim(0.0, 3)
    py.xlabel('F153M Magnitude', labelpad=10)
    py.ylabel('Proper Motion Error (mas/yr)', labelpad=15)

    ax1 = py.gca()
    ax2 = ax1.twiny()
    
    top_tick_mag = np.array([14.021, 17, 19.9, 21.65])
    top_tick_mass = np.zeros(len(top_tick_mag), dtype=float)
    top_tick_label = np.zeros(len(top_tick_mag), dtype='S13')

    for nn in range(len(top_tick_mag)):
        dm = np.abs(iso.mag153m - top_tick_mag[nn])
        dm_idx = dm.argmin()

        top_tick_mass[nn] = iso.M[dm_idx]

        if top_tick_mass[nn] > 10:
            top_tick_label[nn] = '{0:2.0f}'.format(top_tick_mass[nn])
        else:
            top_tick_label[nn] = '{0:4.1f}'.format(top_tick_mass[nn])

    print top_tick_mag
    print top_tick_mass
    print top_tick_label

    # py.arrow(20, 0.65, 0, -0.25, fc='cyan', ec='cyan',
    #          head_width=0.3, head_length=0.1, linewidth=3)
    # py.arrow(20, 0.65, -1, 0.00, fc='cyan', ec='cyan',
    #          head_width=0.1, head_length=0.3, linewidth=3)

    # py.arrow(21.65, 1.95, 0, -0.25, fc='red', ec='red',
    #          head_width=0.3, head_length=0.1, linewidth=3)
    # py.arrow(21.65, 1.95, -1, 0.00, fc='red', ec='red',
    #          head_width=0.1, head_length=0.3, linewidth=3)

    py.plot([13, 20, 20], [0.65, 0.65, 0.0], color='cyan',
            linewidth=3)
    py.plot([13, 21.7, 21.7], [1.95, 1.95, 0.0], color='red',
            linewidth=3)

    
    py.xlim(13, 23)
    py.ylim(0.0, 3)

    ax2.set_xlim(ax1.get_xlim())
    ax2.set_xticks(top_tick_mag)
    ax2.set_xticklabels(top_tick_label)

    ax2.set_xlabel(r'Mass (M$_\odot$)', labelpad=10)

    py.text(18, 2.02, 'New Limits', color='red', fontweight='bold',
            backgroundcolor='white')

    py.savefig(prop_dir + 'arches_pm_error.png')

    return

def plot_mass_function():
    catalog = '/u/mwhosek/Desktop/699-2/ks2/submitted_2015_04/tidal_radius/'
    catalog += 'catalog_probs_4.fits'

    prop_dir = '/Users/jlu/doc/proposals/hst/cycle23/arches/'

    # Read in data table
    t = Table.read(catalog)

    # Read in isochrone
    iso = syn.load_isochrone(logAge=6.4, AKs=2.4, distance=8000)

    # Get the completeness (relevant for diff. de-reddened magnitudes).
    comp_file = prop_dir + 'arches_completeness_mag.txt'
    comp = Table.read(comp_file, format='ascii')

    # Pick out the "good" cluster members.
    good = np.where((t['Membership'] > 0.3) & (t['m_2010_f127m'] > 0))[0]
    clust = t[good]

    # Apply a reddening correction (differential only)
    magCut = 99
    blueCut = -99
    clust = profile.apply_redcut(clust, 1, 2.4, blueCut, magCut, 1)

    mag = clust['m_2010_f153m']
    color = clust['m_2010_f127m'] - clust['m_2010_f153m']
    member = clust['Membership']

    # Drop things with too blue colors... this reduces contamination some.
    good = np.where(color > 2)[0]
    clust = clust[good]
    mag = clust['m_2010_f153m']
    color = clust['m_2010_f127m'] - clust['m_2010_f153m']
    member = clust['Membership']
    

    # Make a finely sampled mass-luminosity relationship by
    # interpolating on the isochrone.
    iso_mag = iso.mag153m
    iso_col = iso.mag127m - iso.mag153m
    iso_mass = iso.M.data
    iso_WR = iso.isWR
    iso_tck, iso_u = interpolate.splprep([iso_mass, iso_mag, iso_col], s=2)

    # Find the maximum mass that is NOT a WR star
    mass_max = iso_mass[iso_WR == False].max()

    u_fine = np.linspace(0, 1, 1e4)
    iso_mass_f, iso_mag_f, iso_col_f = interpolate.splev(u_fine, iso_tck)

    # Define WR stars. 
    iso_WR_f = np.zeros(len(iso_mass_f), dtype=bool)
    iso_WR_f[iso_mass_f > mass_max] = True

    # Make a completeness curve interpolater. First we have to make a
    # color-mag grid and figure out the lowest (worst) completeness at
    # each point in the grid. Then we can interpolate on that grid.
    c_m153_arr = (comp['OuterMag'] + comp['InnerMag']) / 2.0
    c_m127_arr = (comp['OuterMag'] + comp['InnerMag']) / 2.0
    c_comp_arr = np.zeros((len(c_m153_arr), len(c_m127_arr)), dtype=float)
    c_mag_arr = np.zeros((len(c_m153_arr), len(c_m127_arr)), dtype=float)
    c_col_arr = np.zeros((len(c_m153_arr), len(c_m127_arr)), dtype=float)

    # Loop through an array of F153M mag and F127M-F153M color and
    # determine the lowest completness.
    for ii in range(len(c_m153_arr)):
        for jj in range(len(c_m127_arr)):
            c_mag_arr[ii, jj] = c_m153_arr[ii]
            c_col_arr[ii, jj] = c_m127_arr[jj] - c_m153_arr[ii]

            # if comp['F127m_comp'][jj] < comp['F153m_comp'][ii]:
            #     c_comp_arr[ii, jj] = comp['F127m_comp'][jj]
            # else:
            #     c_comp_arr[ii, jj] = comp['F153m_comp'][ii]

            c_comp_arr[ii, jj] = comp['F153m_comp'][ii] * comp['F127m_comp'][jj]
                    
            if c_comp_arr[ii, jj] < 0:
                c_comp_arr[ii, jj] = 0
                
            if c_comp_arr[ii, jj] > 1:
                c_comp_arr[ii, jj] = 1
            
    comp_int = interpolate.SmoothBivariateSpline(c_mag_arr.flatten(),
                                                 c_col_arr.flatten(),
                                                 c_comp_arr.flatten(), s=200)

    mm_tmp = np.arange(14, 23, 0.1)
    cc_tmp = np.arange(0.1, 3.4, 0.1)
    comp_tmp = comp_int(mm_tmp, cc_tmp)
    py.clf()
    py.imshow(comp_tmp, extent=(cc_tmp.min(), cc_tmp.max(),
                                mm_tmp.min(), mm_tmp.max()))
    py.colorbar()
    
    # Loop through data and assign masses and completeness to each star.
    mass = np.zeros(len(mag), dtype=float)
    isWR = np.zeros(len(mag), dtype=float)
    comp = np.zeros(len(mag), dtype=float)
    # print mag.min(), mag.max(), color.min(), color.max()
    # comp = comp_int(mag, color)
    # print comp.shape, mag.shape, color.shape
    for ii in range(len(mass)):
        dmag = mag[ii] - iso_mag_f
        dcol = color[ii] - iso_col_f

        delta = np.hypot(dmag, dcol)

        # Some funny business - sort and get the closest masses reasonable.
        sdx = delta.argsort()

        # If the color + mag difference is less than 0.15, then take
        # the lowest mass. This helps account for the missing IMF bias.
        idx = np.where(delta[sdx] < 0.01)[0]

        if len(idx) == 0:
            min_idx = delta.argmin()
        else:
            min_mass_idx = iso_mass_f[sdx[idx]].argmin()
            min_idx = sdx[idx][min_mass_idx]
            
        print '{0:4d} {1:4d} {2:4d} {3:5.1f} {4:5.1f}'.format(ii,
                                                              min_idx,
                                                              delta.argmin(),
                                                              iso_mass_f[min_idx],
                                                              iso_mass_f[delta.argmin()])

        mass[ii] = iso_mass_f[min_idx]
        isWR[ii] = iso_WR_f[min_idx]
        
        comp[ii] = comp_int(mag[ii], color[ii])

        if comp[ii] > 1:
            comp[ii] = 1
        if comp[ii] < 0:
            comp[ii] = 0

    print mag.min(), mag.max(), color.min(), color.max()
    print comp.shape, mag.shape, color.shape
    
    # Find the maximum mass where we don't have WR stars anymore
    print mass_max, mass.max()
    mass_max = mass[isWR == False].max()
    print mass_max

    # Trim down to just the stars that aren't WR stars.
    idx_noWR = np.where(mass <= mass_max)[0]
    mass_noWR = mass[idx_noWR]
    mag_noWR = mag[idx_noWR]
    color_noWR = color[idx_noWR]
    isWR_noWR = isWR[idx_noWR]
    member_noWR = member[idx_noWR]
    comp_noWR = comp[idx_noWR]

    # Define our mag and mass bins.  We will need both for completeness
    # estimation and calculating the mass function. 
    bins_log_mass = np.arange(0, 1.9, 0.15)
    bins_mag = np.zeros(len(bins_log_mass), dtype=float)

    for ii in range(len(bins_mag)):
        dmass = np.abs((10**bins_log_mass[ii]) - iso_mass_f)
        dmass_min_idx = dmass.argmin()

        bins_mag[ii] = iso_mag_f[dmass_min_idx]

    print 'bins_log_mass = ', bins_log_mass
    print '10**bins_log_mass = ', 10**bins_log_mass
    print 'bins_mag = ', bins_mag
    

    # compute a preliminary mass function with the propoer weights
    weights = member_noWR / comp_noWR
    py.figure(1)
    py.clf()
    py.subplots_adjust(top=0.88)
    
    n_raw, be, p = py.hist(np.log10(mass_noWR), bins=bins_log_mass,
                           log=True, histtype='step',
                           label='Unweighted')
    py.clf()
    n_mem, be, p = py.hist(np.log10(mass_noWR), bins=bins_log_mass,
                            log=True, histtype='step', color='green',
                            weights=member_noWR,
                            label='Observed')
    
    n_fin, be, p = py.hist(np.log10(mass_noWR), bins=bins_log_mass,
                            log=True, histtype='step', color='black',
                            weights=weights,
                            label='Completeness Corr.')

    mean_weight = n_fin / n_mem
    
    n_err = (n_fin**0.5) * mean_weight
    n_err[0] = 1000.0  # dumb fix for empty bin
    bc = bins_log_mass[0:-1] + (np.diff(bins_log_mass) / 2.0)

    py.errorbar(bc[1:], n_fin[1:], yerr=n_err[1:], linestyle='none', color='black')

    # Fit a powerlaw.
    powerlaw = lambda x, amp, index: amp * (x**index)
    fitfunc = lambda p, x: p[0] + p[1] * x
    errfunc = lambda p, x, y, err: (y - fitfunc(p, x)) / err

    log_m = bc[5:]
    log_n = np.log10(n_fin)[5:]
    log_n_err = n_err[5:] / n_fin[5:]

    print 'log_m = ', log_m
    print 'log_n = ', log_n
    print 'log_n_err = ', log_n_err

    pinit = [1.0, -1.0]
    out = optimize.leastsq(errfunc, pinit, args=(log_m, log_n, log_n_err), full_output=1)

    pfinal = out[0]
    covar = out[1]
    print 'pfinal = ', pfinal
    print 'covar = ', covar

    index = pfinal[1]
    amp = 10.0**pfinal[0]

    indexErr = math.sqrt( covar[0][0] )
    ampErr = math.sqrt( covar[1][1] ) * amp

    py.plot(log_m, 10**fitfunc(pfinal, log_m), 'k--')
    py.text(1.3, 100, r'$\frac{dN}{dm}\propto m^{-2.2}$')

    py.axvline(np.log10(mass_max), linestyle='--')
    py.ylim(5, 9e2)
    py.xlim(0, 1.8)
    py.xlabel('log( Mass [Msun])')
    py.ylabel('Number of Stars')
    py.legend()

    
    ax1 = py.gca()
    ax2 = ax1.twiny()
    
    top_tick_mag = np.array([14.0, 17, 20, 21.6])
    top_tick_mass = np.zeros(len(top_tick_mag), dtype=float)
    top_tick_label = np.zeros(len(top_tick_mag), dtype='S13')

    for nn in range(len(top_tick_mag)):
        dm = np.abs(iso.mag153m - top_tick_mag[nn])
        dm_idx = dm.argmin()

        top_tick_mass[nn] = iso.M[dm_idx]
        top_tick_label[nn] = '{0:3.1f}'.format(top_tick_mag[nn])

    print 'top_tick_mag = ', top_tick_mag
    print 'top_tick_msas = ', top_tick_mass
    print 'top_tick_label = ', top_tick_label

    ax2.set_xlim(ax1.get_xlim())
    ax2.set_xticks(np.log10(top_tick_mass))
    ax2.set_xticklabels(top_tick_label)

    ax2.set_xlabel(r'F153M (mags)', labelpad=10)
    py.savefig(prop_dir + 'arches_imf.png')

    py.figure(2)
    py.clf()
    py.plot(np.log10(iso.M), iso.mag153m)
    py.xlim(0, 1.9)
    py.xlabel('Log( Mass [Msun] )')
    py.ylabel('F153M (mag)')
    py.savefig(prop_dir + 'arches_mass_luminosity.png')
    
    py.figure(3)
    py.clf()
    py.plot(color, mag, 'k.')
    py.plot(iso.mag127m - iso.mag153m, iso.mag153m, 'r-')
    py.axvline(2)
    py.ylim(22, 12)
    py.xlim(1, 3.5)
    py.xlabel('F127M - F153M (mag)')
    py.ylabel('F153M (mag)')
    py.savefig(prop_dir + 'arches_cmd_iso_test.png')

    return clust

    
    
    
