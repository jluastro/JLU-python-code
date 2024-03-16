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



def get_obs_times(cadence, duration):
    """
    Parameters
    ----------
    cadence : float
        Cadence (time between observations) in days.
    duration : float
        Duration (time from start to end of survey) in days.
    """
    # Start at the beginning of Cycle 32
    t32_start = Time('2024-10-01T00:00:00.0')

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
        Corresponding event_id in event_table

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
    model_name = 'PSPL_PhotAstrom_Par_Param1'

    pspl_parameter_dict = {'model': model_name, 'raL': raL, 'decL': decL,
                           't0': t0, 'u0': u0, 'tE': tE, 'thetaE': thetaE,
                           'piS': piS, 'piE_E': piE_E, 'piE_N': piE_N,
                           'xS0_E': xS0_E, 'xS0_N': xS0_N,
                           'muS_E': muS_E, 'muS_N': muS_N,
                           'b_sff': b_sff, 'mag_src': mag_src}

    return pspl_parameter_dict


def pspl_model_gen(pspl_parameter_dict):
    """
    Generate pspl_photastrom_par_param1 model from parameter dict

    Parameters
    ----------
    pspl_parameter_dict : dict
        Dictionary of the PSPL_PhotAstrom_Par_Param1 parameters

    Returns
    -------
    pspl model
    """
    ##########
    # Calculate PSPL model and photometry
    ##########
    raL = pspl_parameter_dict['raL']  # Lens R.A.
    decL = pspl_parameter_dict['decL']  # Lens dec
    t0 = pspl_parameter_dict['t0']  # mjd
    u0 = pspl_parameter_dict['u0']
    tE = pspl_parameter_dict['tE']
    thetaE = pspl_parameter_dict['thetaE']
    piS = pspl_parameter_dict['piS']
    piE_E = pspl_parameter_dict['piE_E']
    piE_N = pspl_parameter_dict['piE_N']
    xS0_E = pspl_parameter_dict['xS0_E']
    xS0_N = pspl_parameter_dict['xS0_N']
    muS_E = pspl_parameter_dict['muS_E']  # lens proper motion mas/year
    muS_N = pspl_parameter_dict['muS_N']  # lens proper motion mas/year
    b_sff = pspl_parameter_dict['b_sff']
    mag_src = pspl_parameter_dict['mag_src']

    pspl = model.PSPL_PhotAstrom_Par_Param2(t0, u0, tE, thetaE,
                                            piS, piE_E, piE_N,
                                            xS0_E, xS0_N, muS_E, muS_N,
                                            b_sff, mag_src,
                                            raL=raL, decL=decL)

    return pspl



def get_detectable_sim_events(events_table, t_obs, mH_max=24, bump_mag_cut=0.3):
    # Add a detectable column:
    events_table['hst_det'] = np.ones(len(events_table), dtype=bool)

    # Apply magnitude cut.
    mag_col = 'ubv_I_app_LSN'
    idx = np.where(events_table[mag_col] < mH_max)[0]
    print(len(idx))
    events_table['hst_det'][idx] = False

    # Apply bump magnitude cut.
    events_table['hst_det'][events_table['delta_m_I'] < bump_mag_cut] = False

    # Determine tE uncertainty for each event.
    for ee in range(len(events_table)):
        print(ee, events_table[ee][mag_col], events_table[ee]['hst_det'])
        if events_table['hst_det'][ee] == False:
            pass

        model_class = model.PSPL_PhotAstrom_Par_Param2
        params = get_pspl_lightcurve_parameters(events_table, 'ubv', 'I', ee)

        keys_var = ['t0', 'u0', 'tE', 'thetaE', 'piS', 'piE_E', 'piE_N',
                    'xS0_E', 'xS0_N', 'muS_E', 'muS_N', 'b_sff', 'mag_src']
        keys_fix = ['raL', 'decL']
        params_var = {k: params[k] for k in keys_var}
        params_fix = {k: params[k] for k in keys_fix}

        print(params_var)

        # Make a model lightcurve at the appropriate times
        pspl_mod = pspl_model_gen(params)


        # Following conventions of Wilson et al. 2023 Eq. 12-13
        zp_w146 = 27.648  # 1 e- / sec
        F_aper = 0.5

        mag_w146 = pspl_mod.get_photometry(t_obs)
        flux_w146 = F_aper * 10 ** (-0.4 * (mag_w146 - zp_w146))
        merr_146 = (2.5 / math.log(10)) * (1.0 / flux_w146 ** 0.5)

        perr_146 = np.array([100.0 * merr_146, 100.0 * merr_146]).T

        cov_mat = sensitivity.fisher_cov_matrix_phot_astrom(t_obs, merr_146, perr_146,
                                            model_class,
                                            params_var,
                                            params_fix,
                                            num_deriv_frac=1.0e-4)

        err_on_params = np.sqrt(np.diagonal(cov_mat))

        tE_snr = params_var['tE'] / err_on_params[2]
        print(err_on_params)

        print('tE_snr = ', tE_snr)
        pdb.set_trace()

    return

