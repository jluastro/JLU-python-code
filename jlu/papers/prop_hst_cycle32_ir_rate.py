import os.path

import numpy as np
import pylab as plt
from astropy.time import Time
from astropy.table import Table
from astropy.coordinates import SkyCoord
from astropy import units as u
from bagle import sensitivity
from bagle import model
import math
import pdb
import pandas as pd
import matplotlib.dates as mdates
from matplotlib.patches import Rectangle, Circle
from matplotlib.colors import Normalize
from popsycle import analysis
import time
import pickle


def get_obs_times(cadence, duration, start_time='2024-10-01T00:00:00.0'):
    """
    Parameters
    ----------
    cadence : float
        Cadence (time between observations) in days.
    duration : float
        Duration (time from start to end of survey) in days.
    """
    # Start at the beginning of Cycle 32
    t32_start = Time(start_time)

    # Total number of observing times, before any filtering for Sun exclusion.
    n_times_tmp = int(np.ceil(duration / cadence))

    # Time array (before filtering for Sun exclusion).
    times = Time(t32_start.mjd + np.arange(0, duration, cadence),
                 format='mjd')

    # Establish Sun exclusion zone. Bulge isn't visible to HST from
    # Nov-March (only decimal_year between [0.3-0.8] is visible)
    t_bulge_vis_start = Time('2024-03-01T00:00:00.0')
    t_bulge_vis_stop = Time('2024-10-22T00:00:00.0')

    # Identify good times.
    annual_phase = times.decimalyear % 1
    idx = np.where((annual_phase > (t_bulge_vis_start.decimalyear % 1)) &
                   (annual_phase < (t_bulge_vis_stop.decimalyear % 1)))[0]

    times = times[idx]

    return times

def get_obs_times_apt(cadence, duration, start_time='2024-10-01T00:00:00.0',
                      roll_report='/u/jlu/work/microlens/surveys/hst_cy32_ir_event_rate/hst_cy32_bulge_roll_report.txt'):
    """
    Get observation times at the desired cadence and survey duration; but
    filter out times when the Galactic Bulge is not visible (based on an
    APT roll angle report).

    Parameters
    ----------
    cadence : float
        Desired cadence in days.
    duration : float
        Desired survey duration in days.
    start_time : str
        Start date and time in string format.

    Returns
    -------
    t_obs : array, float
        Times in MJD.
    """
    t_obs = get_obs_times(cadence, duration, start_time = start_time)

    # Load roll angle report.
    vis_report = Table.read(roll_report, format='ascii.fixed_width', delimiter='\t\t',
                            data_start=0, header_start=None)

    vis_report['time'] = Time( pd.to_datetime(vis_report['col1']) ).mjd
    vis_report['is_vis'] = [not val.startswith('Date') for val in vis_report['col2']]

    # is_vis_mask = np.ma.masked_equal(vis_report['is_vis'], True)
    # is_vis_slices = np.ma.clump_masked(is_vis_mask)

    # For each time in our time array, check to see where it falls in the roll report
    # and then check to see if it is visible.
    sdx = np.searchsorted(vis_report['time'], t_obs.mjd)
    is_vis_t_obs = vis_report['is_vis'][sdx]

    # Trim invisible times
    N_before = len(t_obs)
    t_obs = t_obs[is_vis_t_obs]
    N_after = len(t_obs)

    print(f'Trimming {N_before - N_after} of {N_before} times based on roll report ({N_after} left).')

    return t_obs

def get_pspl_lightcurve_parameters(event_table, photometric_system, filter_name, event_id):
    """
    Find the parameters for PSPL_PhotAstrom_Par_Param1 from event_table.

    Parameters
    ----------
    event_table : Astropy table
        Table containing the events calculated from refine_events.

    photometric_system : str
        The name of the photometric system in which the filter exists.

    filter_name : str
        The name of the filter in which to calculate all the
        microlensing events. The filter name convention is set
        in the global filt_dict parameter at the top of this module.

    event_id : float
        Corresponding event_id in event_table (just the row index in the table).

    Returns
    -------
    pspl_parameter_dict : dict
        Dictionary of the PSPL_PhotAstrom_Par_Param1 parameters

    obj_id_L : int
        Object id of the lens associated with event

    obj_id_S : int
        Object id of the source associated with event

    """
    L_coords = SkyCoord(l=event_table[event_id]['glon_L'] * u.degree, b=event_table[event_id]['glat_L'] * u.degree,
                        pm_l_cosb=event_table[event_id]['mu_lcosb_L'] * u.mas / u.year,
                        pm_b=event_table[event_id]['mu_b_L'] * u.mas / u.year, frame='galactic')
    S_coords = SkyCoord(l=event_table[event_id]['glon_S'] * u.degree, b=event_table[event_id]['glat_S'] * u.degree,
                        pm_l_cosb=event_table[event_id]['mu_lcosb_S'] * u.mas / u.year,
                        pm_b=event_table[event_id]['mu_b_S'] * u.mas / u.year, frame='galactic')

    raL = L_coords.icrs.ra.value  # Lens R.A.
    decL = L_coords.icrs.dec.value  # Lens dec
    mL = event_table[event_id]['mass_L']  # msun (Lens current mass)
    t0 = event_table[event_id]['t0']  # mjd
    beta = event_table[event_id]['u0'] * event_table[event_id]['theta_E']  # 5.0
    dL = event_table[event_id]['rad_L'] * 10 ** 3  # Distance to lens
    dL_dS = dL / (event_table[event_id]['rad_S'] * 10 ** 3)  # Distance to lens/Distance to source
    muL_E = L_coords.icrs.pm_ra_cosdec.value  # lens proper motion mas/year
    muL_N = L_coords.icrs.pm_dec.value  # lens proper motion mas/year
    muS_E = S_coords.icrs.pm_ra_cosdec.value  # lens proper motion mas/year
    muS_N = S_coords.icrs.pm_dec.value  # lens proper motion mas/year
    muRel_E = muS_E - muL_E
    muRel_N = muS_N - muL_N
    muRel = np.array([muRel_E, muRel_N])
    muRelHat = muRel / np.linalg.norm(muRel)

    u0 = event_table[event_id]['u0']  # Closest approach in thetaE units
    tE = event_table[event_id]['t_E']  # Einstein crossing time in days
    thetaE = event_table[event_id]['theta_E']
    piS = (event_table[event_id]['rad_S'] * u.kpc).to('mas', equivalencies=u.parallax()).value
    piE_E = event_table[event_id]['pi_E'] * muRelHat[0]
    piE_N = event_table[event_id]['pi_E'] * muRelHat[1]
    xS0_E = 0.001  # source position on sky, arbitrary offset in arcsec
    xS0_N = 0.001  # source position on sky, arbitrary offset in arcsec

    mag_src = event_table[event_id]['%s_%s_app_S' % (photometric_system, filter_name)]
    b_sff = event_table[event_id]['f_blend_%s' % filter_name]
    model_name = 'PSPL_Phot_noPar_Param1'

    pspl_parameter_dict = {'model': model_name, 'raL': raL, 'decL': decL,
                           't0': t0, 'u0': u0, 'tE': tE, 'thetaE': thetaE,
                           'piS': piS, 'piE_E': piE_E, 'piE_N': piE_N,
                           'xS0_E': xS0_E, 'xS0_N': xS0_N,
                           'muS_E': muS_E, 'muS_N': muS_N,
                           'b_sff': b_sff, 'mag_src': mag_src}

    return pspl_parameter_dict


def get_detectable_sim_events(events_table, t_obs, mH_max=24, bump_mag_cut=0.3, min_tE_snr=3,
                              verbose=False):
    # Add a detectable column:
    events_table['hst_det'] = np.ones(len(events_table), dtype=bool)
    print(f'{"No cuts:":20s} N_events = {len(events_table)}')

    # Apply magnitude cut.
    mag_col = 'ubv_H_app_LSN'
    idx = np.where(events_table[mag_col] > mH_max)[0]
    events_table['hst_det'][idx] = False
    print(f'{"Brightness cut:":20s} N_events = {events_table["hst_det"].sum()}')

    # Apply bump magnitude cut.
    binary_filt = (events_table['isMultiple_L'] == 1) | (events_table['isMultiple_S'] == 1)
    single_filt = (events_table['isMultiple_L'] == 0) & (events_table['isMultiple_S'] == 0)
    delta_m_cut = (((events_table['bin_delta_m'] > bump_mag_cut) & binary_filt) |
                   ((events_table['delta_m_H'] > bump_mag_cut) & single_filt))
    events_table['hst_det'][~delta_m_cut] = False
    print(f'{"Delta-mag cut:":20s} N_events = {events_table["hst_det"].sum()}')

    # Trim out masked or empty magnitude values. Typically happens when
    # the source is a compact object.
    # events_table['hst_det'][events_table['delta_m_H'].mask] = False
    # print(f'Dark source cut: N_events = {events_table["hst_det"].sum()}')

    # Trim out sources with negative blend fluxes. Something went wrong in
    # PopSyCLE (question to Natasha).
    events_table['hst_det'][events_table['f_blend_H'] <= 0] = False
    print(f'{"Neg. blend cut:":20s} N_events = {events_table["hst_det"].sum()}')

    # Trim out sources with tE < cadence... can't detect those anyhow.
    # Assume cadence is just the smallest delta-t.
    dt_min = np.diff(t_obs).min()
    events_table['hst_det'][events_table['t_E'] <= dt_min] = False
    tmp = f'tE < {dt_min:.1f} cut:'
    print(f'{tmp:20s} N_events = {events_table["hst_det"].sum()}')


    # Define model class used in sensitivity analysis.
    model_class = model.PSPL_Phot_Par_Param1

    # Add a tE_snr column.
    events_table['tE_snr'] = np.zeros(len(events_table), dtype=float)

    # Determine tE uncertainty for each event.
    idx = np.where(events_table['hst_det'] == True)[0]

    for ee in idx:
        event = events_table[ee]

        # Fetch params from even table.
        params = get_pspl_lightcurve_parameters(events_table, 'ubv', 'H', ee)

        # Reformat the parameters.
        keys_var = ['t0', 'u0', 'tE', 'piE_E', 'piE_N', 'b_sff', 'mag_src']
        keys_fix = ['raL', 'decL']
        params_var = {k: params[k] for k in keys_var}
        params_fix = {k: params[k] for k in keys_fix}

        # Figure out if this is an event we should print out:
        print_ev =  ((verbose is True) and
                     (params_var['t0'] > t_obs.min()) and
                     (params_var['t0'] < t_obs.max()) and
                     (np.abs(params_var['u0']) < 1) and
                     (params_var['tE'] > 10) and
                     (params_var['mag_src'] < 20))

        if print_ev:
            print('----------')
            print('Good Object:', ee)
            print(f'mag = {event[mag_col]:.2f}',
                  f'bump = {event["delta_m_H"]:.2f}',
                  f't0 = {event["t0"]:.0f}',
                  f'detectable = {event["hst_det"]}')


        # Make a model lightcurve at the appropriate times to estimate flux errors.
        pspl_mod = model.get_model(model_class, params_var, params_fix)

        # Assign Roman flux errors.
        # Following conventions of Wilson et al. 2023 Eq. 12-13
        merr_146, perr_146 = get_mag_pos_error(pspl_mod, t_obs)

        # Define steps used in numerical derivative in fisher matrix.
        param_delta = params_var.copy()
        param_delta['t0'] = 10.0 # day
        param_delta['u0'] = 0.1
        param_delta['tE'] = params_var['tE'] * 0.1
        param_delta['piE_E'] = 0.05
        param_delta['piE_N'] = 0.05
        param_delta['b_sff'] = 0.01
        param_delta['mag_src'] = 0.01

        # double check a couple so we don't push parameters outside of valid range.
        # b_sff shouldn't be negative.
        if param_delta['b_sff'] >= params_var['b_sff']:
            param_delta['b_sff'] = params_var['b_sff'] * 0.95
        # tE shouldn't be negative.
        if param_delta['tE'] >= params_var['tE']:
            param_delta['tE'] = params_var['tE'] * 0.95

        cov_mat = sensitivity.fisher_matrix(t_obs, merr_146,
                                            model_class,
                                            params_var,
                                            params_fix,
                                            num_deriv_frac=0.01,
                                            param_delta=param_delta,
                                            verbose=False)
        if (np.isinf(cov_mat[0,0])) and verbose:
            print('Bad covariance matrix, infinite errors:')
            print(ee, params_var, param_delta)

        err_on_params = np.sqrt(np.diagonal(cov_mat))

        tE_snr = params_var['tE'] / err_on_params[2]
        if np.isnan(tE_snr):
            tE_snr = 0
        events_table['tE_snr'][ee] = tE_snr

        # Filter out those events that don't have well-measured tE.
        if tE_snr < min_tE_snr:
            events_table[ee]['hst_det'] = False

        # Examine some sensible events:
        if print_ev:
            with np.printoptions(precision=3, suppress=True):
                print(f'mag error range = [{merr_146.min():.3f}, {merr_146.max():.3f}]')
                print('params = ', np.array(list(params_var.values())))
                print('errors = ', err_on_params)
                #print('covariance matrix')
                #print(cov_mat[0:3,0:3])

            print(f'tE_snr = {tE_snr:.2f}')


    print(f'{"tE SNR cut":20s}: N_events = {events_table["hst_det"].sum()}')

    return events_table


def get_mag_pos_error(pspl_mod, t_obs):
    """
    Get photometric and astrometric errors from the input model and
    at specified times.

    Parameters
    ----------
    pspl_mod : bagle.model
        Model instance that supports photometry.

    t_obs : array-like
        Array of times (in MJD).

    Returns
    -------
    mag_err : array-like
        Magnitude errors at specified times. Same shape as t_obs.

    pos_err : array-like
        Position errors at specified times. Same shape as t_obs.
    """
    zp_w146 = 27.648  # 1 e- / sec
    F_aper = 0.5

    # Assign Roman flux errors.
    # Following conventions of Wilson et al. 2023 Eq. 12-13
    mag_w146 = pspl_mod.get_photometry(t_obs)
    flux_w146 = F_aper * 10 ** (-0.4 * (mag_w146 - zp_w146))

    merr_146 = (2.5 / math.log(10)) * (1.0 / flux_w146 ** 0.5)
    perr_146 = np.array([100.0 * merr_146, 100.0 * merr_146]).T

    return merr_146, perr_146


def plot_event_lightcurve(events_table, ee, t_obs, oversamp=3):
    """
    Plot an event lightcurve.

    Parameters
    ----------
    events_table
    ee
    t_obs

    Returns
    -------

    """
    # Define model class used in sensitivity analysis.
    model_class = model.PSPL_Phot_Par_Param1

    # Fetch params from even table.
    params = get_pspl_lightcurve_parameters(events_table, 'ubv', 'H', ee)

    # Reformat the parameters.
    keys_var = ['t0', 'u0', 'tE', 'piE_E', 'piE_N', 'b_sff', 'mag_src']
    keys_fix = ['raL', 'decL']
    params_var = {k: params[k] for k in keys_var}
    params_fix = {k: params[k] for k in keys_fix}

    # Make a model lightcurve at the appropriate times to estimate flux errors.
    pspl_mod = model.get_model(model_class, params_var, params_fix)

    mag = pspl_mod.get_photometry(t_obs)
    merr, perr = get_mag_pos_error(pspl_mod, t_obs)

    # Make a more densely sampled model curve as well
    dt = t_obs[1] - t_obs[0]
    model_t = np.arange(t_obs.min(), t_obs.max(), dt/oversamp)
    model_mag = pspl_mod.get_photometry(model_t)

    plt.figure()
    markers, caps, bars = plt.errorbar(t_obs, mag, yerr=merr,
                                       ls="none", marker='.', ms=2,
                                       color='black', label='Sim Data')
    plt.plot(model_t, model_mag, ls='-', color='red',
             lw=1, marker='none', label='Model')
    plt.xlabel('Time (MJD)')
    plt.ylabel('H mag')
    plt.title(f'Event #{ee}: SNR(tE) = {events_table[ee]["tE_snr"]:.1f}')
    plt.gca().invert_yaxis()
    plt.legend()

    [bar.set_alpha(0.3) for bar in bars]
    [cap.set_alpha(0.3) for cap in caps]

    return

def plot_tE_snr_hist(sims, cadence, duration):
    # Trim down to just those events that have tE_snr != 0
    idx = np.where(sims['tE_snr'] > 0)[0]

    # Calculate the number of objects per HST WFC3-IR field.
    sim_area = 0.1 * u.deg ** 2
    wfc3_area = 123 * u.arcsec * 137 * u.arcsec

    sim_per_wfc3ir = (sim_area / wfc3_area).to('').value
    weights = np.repeat(1./sim_per_wfc3ir, len(idx))

    n_snr3_wfc3ir = np.sum(sims["hst_det"]) / sim_per_wfc3ir

    plt.figure()
    logbins = np.logspace(-1, 3, 30)
    plt.hist(sims['tE_snr'][idx], bins=logbins, weights=weights)
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('$t_E$ SNR')
    plt.ylabel('Events per WFC3-IR Field')
    plt.title(f'Cadence={cadence} d, Duration={duration} d, $N_{{tE>3,WFC3-IR}}$={n_snr3_wfc3ir:.1f}',
              fontsize=12)

    return

def analyze_field(l=-0.5, b=-0.5, min_tE_snr=3, outdir='/u/jlu/work/microlens/surveys/hst_cy32_ir_event_rate/'):
    # Read in a PopSyCLE field.
    sim_dir = '/u/nsabrams/work/PopSyCLE_runs/v2023/roman_grid/'
    field = f'l_{l:.1f}_b_{b:.1f}'
    root = f'{field}_refined_events_ubv_H_Damineli16_rb'
    ev_file = sim_dir + field + '/' + root + '.fits'
    sim_area = 0.1 * u.deg ** 2

    if not os.path.exists((ev_file)):
        print(f'Event file does not exist: {ev_file}')
        return

    sims = Table.read(ev_file)

    # Define time array
    cadence_dense = 1.0
    duration_dense = 92
    t1 = get_obs_times_apt(8, 40, start_time='2024-10-01T00:00:00.00')
    t2 = get_obs_times_apt(16, 40, start_time='2025-03-29T00:00:00.00')
    t3 = get_obs_times_apt(10, 40, start_time='2025-05-20T00:00:00.00')
    t4 = get_obs_times_apt(cadence_dense, duration_dense, start_time='2025-07-01T00:00:00.00')

    t_new = Time(np.concatenate([t1, t2, t3, t4]))

    # Modify the simulation time steps to be in the observing window.
    t_min = t_new.min()  # Add this time to all relevant time columns in the popsycle table.

    sims['t0'] += t_min.mjd

    # Find detections.
    print(f'Detecting events for {field}')
    sims_new = get_detectable_sim_events(sims, t_new.mjd, min_tE_snr=min_tE_snr, verbose=False)

    plot_tE_snr_hist(sims_new, cadence_dense, duration_dense)
    plt.savefig(outdir + root + '_tEsnr_hist.png')

    # Pull out the detectable events and plot a few
    idx = np.where(sims_new['hst_det'] == True)[0]
    if len(idx) > 3:
        plot_event_lightcurve(sims_new, idx[0], t_new.mjd)
        plt.savefig(outdir + root + f'_lcurve_{idx[0]}.png')
        plot_event_lightcurve(sims_new, idx[1], t_new.mjd)
        plt.savefig(outdir + root + f'_lcurve_{idx[1]}.png')
        plot_event_lightcurve(sims_new, idx[2], t_new.mjd)
        plt.savefig(outdir + root + f'_lcurve_{idx[2]}.png')

    # Save the output.
    sims_new.write(outdir + root + f'_tEsnr{min_tE_snr:.1f}.fits', overwrite=True)

    return t_new, sims_new

def plot_times(t, t_all):
    plt.figure(figsize=(10, 2))
    plt.vlines(t_all.datetime64, 1, 1.1, color='black')
    plt.vlines(t.datetime64, 1.01, 1.09, color='red')
    plt.gca().xaxis.set_major_locator(mdates.MonthLocator())
    plt.grid()
    foo = plt.xticks(rotation=30)

    return

def reset_detectability(ev_tab_file, min_tE_snr=3, mH_max=24, bump_mag_cut=0.3):
    """
    Recalculate detectability using all the same criteria except
    modify the SNR(tE) threshold.

    Parameters
    ----------
    ev_tab_file : str
        Name of the events table file with the tE_snr and hst_det columns
        added. Usually this is the output from get_detectable_sim_events().

    min_tE_snr : float
        Minimum signal-to-noise of tE that counts as detectable.

    Returns
    -------
    Return events table object (also saved to new file).
    """

    # Read in the events table.
    events_table = Table.read(ev_tab_file)

    # Calculate the number of objects per HST WFC3-IR field.
    sim_area = 0.1 * u.deg ** 2
    wfc3_area = 123 * u.arcsec * 137 * u.arcsec

    sim_per_wfc3ir = (sim_area / wfc3_area).to('').value

    # Add a detectable column:
    events_table['hst_det'] = np.ones(len(events_table), dtype=bool)
    print(f'{"No cuts:":20s} N_events = {len(events_table) / sim_per_wfc3ir:.1f} per WFC3-IR')

    # Apply magnitude cut.
    mag_col = 'ubv_H_app_LSN'
    idx = np.where(events_table[mag_col] > mH_max)[0]
    events_table['hst_det'][idx] = False
    print(f'{"Brightness cut:":20s} N_events = {events_table["hst_det"].sum() / sim_per_wfc3ir:.1f} per WFC3-IR')

    # Apply bump magnitude cut.
    binary_filt = (events_table['isMultiple_L'] == 1) | (events_table['isMultiple_S'] == 1)
    single_filt = (events_table['isMultiple_L'] == 0) & (events_table['isMultiple_S'] == 0)
    delta_m_cut = (((events_table['bin_delta_m'] > bump_mag_cut) & binary_filt) |
                   ((events_table['delta_m_H'] > bump_mag_cut) & single_filt))
    events_table['hst_det'][~delta_m_cut] = False
    print(f'{"Delta-mag cut:":20s} N_events = {events_table["hst_det"].sum() / sim_per_wfc3ir:.1f} per WFC3-IR')

    # Trim out sources with negative blend fluxes. Something went wrong in
    # PopSyCLE (question to Natasha).
    events_table['hst_det'][events_table['f_blend_H'] <= 0] = False
    print(f'{"Neg. blend cut:":20s} N_events = {events_table["hst_det"].sum() / sim_per_wfc3ir:.1f} per WFC3-IR')

    events_table['hst_det'][events_table['tE_snr'] < min_tE_snr] = False
    print(f'{"tE SNR cut:":20s} N_events = {events_table["hst_det"].sum() / sim_per_wfc3ir:.1f} per WFC3-IR')

    events_table.write(ev_tab_file.replace('tEsnr.fits', f'tEsnr{min_tE_snr:.1f}.fits'),
                       overwrite=True)

    return events_table

def plot_map_detections(min_tE_snr=1.0,
                        outdir = '/u/jlu/work/microlens/surveys/hst_cy32_ir_event_rate/',
                        shape='Rectangle'):
    """
    Make a map of the number of detections per WFC3-IR field.

    Parameters
    ----------
    min_tE_snr : float
    outdir : str
    """

    l = [-0.5, 0.0, 0.5, 1.0, 1.5]
    b = [0.0, -0.5, -1.0, -1.5, -2.0]

    # Array to save number of detections at SNR(tE) > 1.0
    n_det = np.empty((len(l), len(b)), dtype=float)
    l_2d =  np.empty((len(l), len(b)), dtype=float)
    b_2d = np.empty((len(l), len(b)), dtype=float)

    outdir = '/u/jlu/work/microlens/surveys/hst_cy32_ir_event_rate/'

    # Calculate the number of objects per HST WFC3-IR field.
    sim_area = 0.1 * u.deg ** 2
    wfc3ir_dx = 123 * u.arcsec
    wfc3ir_dy = 137 * u.arcsec
    wfc3_area = wfc3ir_dx * wfc3ir_dy

    sim_per_wfc3ir = (sim_area / wfc3_area).to('').value

    for ll in range(len(l)):
        for bb in range(len(b)):
            ev_file = f'l_{l[ll]:.1f}_b_{b[bb]:.1f}_refined_events_ubv_H_Damineli16_rb_tEsnr1.0.fits'

            l_2d[ll, bb] = l[ll]
            b_2d[ll, bb] = b[bb]

            if not os.path.exists(outdir + ev_file):
                print(f'Event file does not exist: {ev_file}')
                n_det[ll, bb] = np.nan
                continue

            sims = Table.read(outdir + ev_file)

            n_det_sim = sims['hst_det'].sum()

            n_det_wfc3ir = n_det_sim / sim_per_wfc3ir

            # Save the number of detections per wfc3ir field.
            n_det[ll, bb] = n_det_wfc3ir

    # Make plot
    plt.figure(figsize=(8,6))

    cmap = plt.cm.jet  # or any other colormap
    norm = Normalize(vmin=0, vmax=np.nanmax(n_det))

    # Scatter plot to get colorbar properly later.
    plt.scatter(l_2d, b_2d, c=n_det, marker='s', s=2, norm=norm)

    for ll in range(len(l)):
        for bb in range(len(b)):
            if np.isnan(n_det[ll, bb]):
                continue

            color = cmap(norm(n_det[ll, bb]))

            if shape == 'Rectangle':
                rect_dx = 0.1**0.5
                rect_dy = 0.1**0.5
                rect_x0 = l[ll] - rect_dx / 2.0
                rect_y0 = b[bb] - rect_dy / 2.0

                # Create a Rectangle patch
                ptch = Rectangle((rect_x0, rect_y0), rect_dx, rect_dy,
                                 linewidth=1, edgecolor=color, facecolor=color)
            else:
                dr = (0.1 / math.pi)**0.5
                ptch = Circle((l[ll], b[bb]), radius=dr,
                              linewidth=1, edgecolor=color, facecolor=color)

            # Add the patch to the Axes
            plt.gca().add_patch(ptch)

    plt.colorbar(label='N Detected per WFC3-IR Field')
    plt.axhline(0, ls='--', color='grey')
    plt.axvline(0, ls='--', color='grey')
    plt.axis('equal')
    plt.gca().invert_xaxis()
    plt.title("PopSyCLE Patch Size = 0.1 deg^2")
    plt.xlabel('l (deg)')
    plt.ylabel('b (deg)')

    return

def map_N_stars(mH_max=24, bump_mag_cut=0.3, u0_cut=2.0, recalc=True):
    l = [-0.5, 0.0, 0.5, 1.0, 1.5]
    b = [-2.0, -1.5, -1.0, -0.5, 0.0]

    hdf5dir = '/u/nsabrams/work/PopSyCLE_runs/v2023/roman_grid'
    outdir = '/u/jlu/work/microlens/surveys/hst_cy32_ir_event_rate/'

    # Assume all the patches have the same area.
    # Use the first patch to figure out the area and radius.
    field = f'l_{l[0]:.1f}_b_{b[-1]:.1f}'
    par_file = f'{hdf5dir}/{field}/{field}_galaxia_params.txt'
    params = read_galaxia_params(par_file)

    A_patch = float(params['surveyArea']) * u.deg**2
    r_patch = ((A_patch / math.pi) ** 0.5).to('deg')

    # Define fine bins for 2D histogram.
    binsize = 0.05 # deg
    l_bins = np.arange(l[0]-r_patch.value, l[-1]+r_patch.value, binsize)
    b_bins = np.arange(b[0]-r_patch.value, b[-1]+r_patch.value, binsize)

    # Array to save number of stars and number of events.
    n_map_det = np.zeros((len(l_bins)-1, len(b_bins)-1), dtype=float)
    n_map_evt = np.zeros((len(l_bins)-1, len(b_bins)-1), dtype=float)

    # For the 2d histogram, for each bin, calculate the vertices of the
    # square. Later, we will use this to check for complete area
    # coverage.
    hist_vertices = np.array([np.meshgrid(l_bins[0:-1], b_bins[0:-1]),
                              np.meshgrid(l_bins[1:], b_bins[0:-1]),
                              np.meshgrid(l_bins[0:-1], b_bins[1:]),
                              np.meshgrid(l_bins[1:], b_bins[1:])])

    for ll in range(len(l)):
        for bb in range(len(b)):
            print(f'Starting loop l={l[ll]:.1f}, b={b[bb]:.1f}')
            t_loop_start = time.time()
            field = f'l_{l[ll]:.1f}_b_{b[bb]:.1f}'
            h5_file = f'{hdf5dir}/{field}/{field}.h5'
            ev_file = f'{hdf5dir}/{field}/{field}_refined_events_ubv_H_Damineli16_rb.fits'

            par_file = f'{hdf5dir}/{field}/{field}_galaxia_params.txt'

            if not os.path.exists(par_file):
                continue

            params = read_galaxia_params(par_file)

            # Determine which histogram squares are inside this patch area.
            # First shift the bin square vertices to the l,b center of this patch.
            lb_tmp = np.array([l[ll], b[bb]])
            hist_vert_patch = hist_vertices - lb_tmp[np.newaxis, :, np.newaxis, np.newaxis]

            # Calculate the radius of each vertex.
            hist_vert_patch_r = np.hypot(hist_vert_patch[:, 0, :, :], hist_vert_patch[:, 1, :, :])

            # Figure out whether all the vertices for each square are inside the circle.
            hist_in_patch = np.all(hist_vert_patch_r < r_patch.value, axis=0).T
            print(f'Number of bins in patch = {np.sum(hist_in_patch)}')

            ######
            # Read in the list of stars for this patch and make histogram.
            ######
            print(f'Counting up all stars: {time.time() - t_loop_start} sec from loop start.')
            stars = analysis.get_star_system_pos_mag(h5_file, filt='ubv_H', recalc=recalc)
            print(f'Dropping too-faint stars: {time.time() - t_loop_start} sec from loop start.')
            stars = stars[stars['m_ubv_H_app'] <= mH_max]

            print(f'Histogramming stars: {time.time() - t_loop_start} sec from loop start.')
            H, le, be = np.histogram2d(stars['glon'], stars['glat'], bins=(l_bins, b_bins))
            del stars  # memory management

            # Convert to density
            H /= binsize**2

            # Figure out which bins fall entirely in this patch and haven't already
            # been populated from other patches.
            print(f'Adding stars to map: {time.time() - t_loop_start} sec from loop start.')
            idx = np.where((hist_in_patch == True) & (n_map_det == 0))
            n_map_det[idx[0], idx[1]] = H[idx[0], idx[1]]
            del H

            ######
            # Read in events and make histogram.
            ######
            if not os.path.exists(ev_file):
                prop.analyze_field(l[ll], b[bb], min_tE_snr=1)

            print(f'Counting up all events: {time.time() - t_loop_start} sec from loop start.')
            events = Table.read(ev_file).to_pandas()

            # Cutout undetectable events.
            print(f'Dropping too-faint events: {time.time() - t_loop_start} sec from loop start.')
            events = events[events['ubv_H_app_LSN'] <= mH_max]  # bright enough
            events = events[events['u0'] < u0_cut]              # strong amplification

            binary_filt = (events['isMultiple_L'] == 1) | (events['isMultiple_S'] == 1)
            single_filt = (events['isMultiple_L'] == 0) & (events['isMultiple_S'] == 0)
            delta_m_cut = (((events['bin_delta_m'] > bump_mag_cut) & binary_filt) |
                           ((events['delta_m_H'] > bump_mag_cut) & single_filt))
            events = events[delta_m_cut]                        # strong bump magnitude (not too blended)
            events = events[events['f_blend_H'] > 0]            # no negative blending (something went wrong).


            print(f'Histogramming events: {time.time() - t_loop_start} sec from loop start.')
            H2, le, be = np.histogram2d(events['glon_S'], events['glat_S'], bins=(l_bins, b_bins))
            del events

            H2 /= binsize ** 2

            print(f'Adding events to map: {time.time() - t_loop_start} sec from loop start.')
            idx = np.where((hist_in_patch == True) & (n_map_evt == 0))
            n_map_evt[idx[0], idx[1]] = H2[idx[0], idx[1]]
            del H2


            # Save binned maps.
            with open(f'{outdir}/map_stars_events_mH{mH_max:.1f}.pkl', 'wb') as f:
                pickle.dump(l_bins, f)
                pickle.dump(b_bins, f)
                pickle.dump(n_map_det, f)
                pickle.dump(n_map_evt, f)
                pickle.dump(mH_max, f)
                pickle.dump(bump_mag_cut, f)
                pickle.dump(u0_cut, f)
                pickle.dump(l, f)
                pickle.dump(b, f)
                print('Saved PICKLE file.')

    return

def plot_map_stars(mH_max=24):
    outdir = '/u/jlu/work/microlens/surveys/hst_cy32_ir_event_rate/'

    DT_years = 1000 / 365.25  # survey duration in years

    with open(f'{outdir}/map_stars_events_mH{mH_max:.1f}.pkl', 'rb') as f:
        l_bins = pickle.load(f)
        b_bins = pickle.load(f)
        n_map_det = pickle.load(f)
        n_map_evt = pickle.load(f)
        mH_max = pickle.load(f)
        bump_mag_cut = pickle.load(f)
        u0_cut = pickle.load(f)
        l = pickle.load(f)
        b = pickle.load(f)

    # Clean up the zeros.
    n_map_det[n_map_det == 0] = np.nan
    n_map_evt[n_map_evt == 0] = np.nan

    # Make plot of stellar density
    plt.figure(1, figsize=(8,6))
    plt.axis('equal')
    plt.imshow(n_map_det.T / 1e6, origin='lower',
               extent=[l_bins[0], l_bins[-1], b_bins[0], b_bins[-1]])
    plt.gca().invert_xaxis()
    plt.xlabel('l (deg)')
    plt.ylabel('b (deg)')
    plt.colorbar(label=r'Stellar Density ($\times 10^{6}$ deg$^{-2}$)')
    plt.title(f'Unblended Stars with m$_{{H,vega}}$<{mH_max:.1f}')

    # Make plot of density of microlensing events.
    plt.figure(2, figsize=(8, 6))
    plt.axis('equal')
    plt.imshow(n_map_evt.T / DT_years, origin='lower',
               extent=[l_bins[0], l_bins[-1], b_bins[0], b_bins[-1]])
    plt.gca().invert_xaxis()
    plt.xlabel('l (deg)')
    plt.ylabel('b (deg)')
    plt.colorbar(label=r'Event Rate (deg$^{-2}$ yr$^{-1}$)')
    title = f'm$_{{H,vega}}$<{mH_max:.1f} u$_0$<{u0_cut:.0f} $\Delta m_{{peak}}$>{bump_mag_cut:.1f}'
    plt.title(title)

    # Make plot of event rate
    plt.figure(3, figsize=(8, 6))
    plt.axis('equal')
    plt.imshow(n_map_evt.T * 1e6 / (DT_years * n_map_det.T), origin='lower',
               extent=[l_bins[0], l_bins[-1], b_bins[0], b_bins[-1]])
    plt.gca().invert_xaxis()
    plt.xlabel('l (deg)')
    plt.ylabel('b (deg)')
    plt.colorbar(label=r'Event Rate ($\times 10^{-6}$ star$^{-1}$ yr$^{-1}$)')
    plt.title(title)

    return

def read_galaxia_params(param_file):
    # Fetch some values from the params file
    lines = []
    with open(param_file, 'r') as f:
        lines = f.readlines()

    # Split and strip white spaces
    strip_list = [line.replace('\n', '').split(' ') for line in lines if line != '\n']

    # Make the final dictionary.
    params = dict()
    for strip in strip_list:
        params[strip[0]] = strip[1]

    return params
