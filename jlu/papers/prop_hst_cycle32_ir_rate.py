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


def get_detectable_sim_events(events_table, t_obs, mH_max=24, bump_mag_cut=0.3,
                              verbose=False):
    # Add a detectable column:
    events_table['hst_det'] = np.ones(len(events_table), dtype=bool)

    # Apply magnitude cut.
    mag_col = 'ubv_H_app_LSN'
    idx = np.where(events_table[mag_col] > mH_max)[0]
    events_table['hst_det'][idx] = False
    print(f'Brightness cut: N_events = {events_table["hst_det"].sum()}')

    # Apply bump magnitude cut.
    events_table['hst_det'][events_table['delta_m_H'] < bump_mag_cut] = False
    print(f'Delta-mag cut:  N_events = {events_table["hst_det"].sum()}')

    # Trim out masked or empty magnitude values. Typically happens when
    # the source is a compact object.
    events_table['hst_det'][events_table['delta_m_H'].mask] = False
    print(f'Dark source cut: N_events = {events_table["hst_det"].sum()}')

    # Trim out sources with negative blend fluxes. Something went wrong in
    # PopSyCLE (question to Natasha).
    events_table['hst_det'][events_table['f_blend_H'] <= 0] = False
    print(f'Neg. blend cut:  N_events = {events_table["hst_det"].sum()}')

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
        param_delta['tE'] = 5.0
        param_delta['piE_E'] = 0.1
        param_delta['piE_N'] = 0.1
        param_delta['b_sff'] = 0.01
        param_delta['mag_src'] = 0.1

        cov_mat = sensitivity.fisher_matrix(t_obs, merr_146,
                                            model_class,
                                            params_var,
                                            params_fix,
                                            num_deriv_frac=0.01,
                                            param_delta=param_delta)

        err_on_params = np.sqrt(np.diagonal(cov_mat))

        tE_snr = params_var['tE'] / err_on_params[2]
        events_table['tE_snr'][ee] = tE_snr

        # Filter out those events that don't have well-measured tE.
        if tE_snr < 3:
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


    print(f'tE SNR cut:    N_events = {events_table["hst_det"].sum()}')

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
    plt.title(f'Event #{ee}')
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
    logbins = np.logspace(0, 3, 20)
    plt.hist(sims['tE_snr'][idx], bins=logbins, weights=weights)
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('$t_E$ SNR')
    plt.ylabel('Events per WFC3-IR Field')
    plt.title(f'Cadence={cadence} d, Duration={duration} d, $N_{{tE>3,WFC3-IR}}$={n_snr3_wfc3ir:.1f}')

    return