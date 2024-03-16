import numpy as np
import pylab as plt
from microlens.jlu import model
from astropy.time import Time

def sample_model_as_ogle(pspl):
    # Simulate
    # photometric observations every 1 day and
    # for the bulge observing window. Observations missed
    # for 125 days out of 365 days for photometry and missed
    t_phot = np.array([], dtype=float)
    
    for year_start in np.arange(56000, 58000, 365.25):
        phot_win = 240.0
        phot_start = (365.25 - phot_win) / 2.0
        t_phot_new = np.arange(year_start + phot_start,
                               year_start + phot_start + phot_win, 1)
        t_phot = np.concatenate([t_phot, t_phot_new])

    # Make the photometric observations.
    # Assume 0.05 mag photoemtric errors at I=19.
    # This means Signal = 400 e- at I=19.
    flux0 = 4000.0
    imag0 = 19.0
    imag_obs = pspl_par_in.get_photometry(t_phot)
    imag_obs_err = np.zeros(len(t_phot))
    if noise:
        flux_obs = flux0 * 10 ** ((imag_obs - imag0) / -2.5)
        flux_obs_err = flux_obs ** 0.5
        flux_obs += np.random.randn(len(t_phot)) * flux_obs_err
        imag_obs = -2.5 * np.log10(flux_obs / flux0) + imag0
        imag_obs_err = 1.087 / flux_obs_err

    data = {}
    data['t_phot1'] = t_phot
    data['mag1'] = imag_obs
    data['mag_err1'] = imag_obs_err

    data['raL'] = raL_in
    data['decL'] = decL_in
    data['target'] = 'sim'
    data['phot_data'] = 'sim'
    data['phot_files'] = ['sim']
    
    return data


def sample_model_as_keck(pspl, noise=True):
    # Simulate
    # astrometric observations every 14 days
    # for the bulge observing window. Observations missed
    # for 245 days out of 365 days for astrometry.
    t_ast = np.array([], dtype=float)
    for year_start in np.arange(56000, 58000, 365.25):
        ast_win = 120.0
        ast_start = (365.25 - ast_win) / 2.0
        t_ast_new = np.arange(year_start + ast_start,
                              year_start + ast_start + ast_win, 14)
        t_ast = np.concatenate([t_ast, t_ast_new])

    # Make the photometric observations.
    # Assume 0.05 mag photoemtric errors at K=21.
    # This means Signal = 400 e- at I=21.
    flux0 = 4000.0
    imag0 = 21.0
    imag_obs = pspl_par_in.get_photometry(t_ast)
    imag_obs_err = np.zeros(len(t_ast))
    if noise:
        flux_obs = flux0 * 10 ** ((imag_obs - imag0) / -2.5)
        flux_obs_err = flux_obs ** 0.5
        flux_obs += np.random.randn(len(t_ast)) * flux_obs_err
        imag_obs = -2.5 * np.log10(flux_obs / flux0) + imag0
        imag_obs_err = 1.087 / flux_obs_err

    # Make the astrometric observations.
    # Assume 0.15 milli-arcsec astrometric errors in each direction at all epochs. 
    if noise:
        pos_obs_tmp = pspl_par_in.get_astrometry(t_ast)
        pos_obs_err = np.ones((len(t_ast), 2), dtype=float) * 0.01 * 1e-3
        pos_obs = pos_obs_tmp + pos_obs_err * np.random.randn(len(t_ast), 2)
    else: 
        pos_obs = pspl_par_in.get_astrometry(t_ast)
        pos_obs_err = np.zeros((len(t_ast), 2))

    data = {}
    data['t_phot1'] = t_phot
    data['mag1'] = imag_obs
    data['mag_err1'] = imag_obs_err

    data['t_ast1'] = t_ast
    data['xpos1'] = pos_obs[:, 0]
    data['ypos1'] = pos_obs[:, 1]
    data['xpos_err1'] = pos_obs_err[:, 0]
    data['ypos_err1'] = pos_obs_err[:, 1]
    
    data['raL'] = raL_in
    data['decL'] = decL_in
    data['target'] = 'sim'
    data['phot_data'] = 'sim'
    data['ast_data'] = 'sim'
    data['phot_files'] = ['sim']
    data['ast_files'] = ['sim']

    
    return data, params

def sample_model_as_roman(pspl):
    # Simulate
    # photometric observations every 1 day and
    # for the bulge observing window. Observations missed
    # for 125 days out of 365 days for photometry and missed
    t_phot = np.array([], dtype=float)

    Roman_launch = Time('2027-01-01')
    print(roman_launch.mjd)
    
    for year_start in np.arange(roman_launch.decimalyear, roman_launch.decimalyear + 5, 365.25):
        phot_win = 240.0
        phot_start = (365.25 - phot_win) / 2.0
        t_phot_new = np.arange(year_start + phot_start,
                               year_start + phot_start + phot_win, 1)
        t_phot = np.concatenate([t_phot, t_phot_new])

    # Make the photometric observations.
    # Assume 0.05 mag photoemtric errors at I=19.
    # This means Signal = 400 e- at I=19.
    flux0 = 4000.0
    imag0 = 19.0
    imag_obs = pspl_par_in.get_photometry(t_phot)
    imag_obs_err = np.zeros(len(t_phot))
    if noise:
        flux_obs = flux0 * 10 ** ((imag_obs - imag0) / -2.5)
        flux_obs_err = flux_obs ** 0.5
        flux_obs += np.random.randn(len(t_phot)) * flux_obs_err
        imag_obs = -2.5 * np.log10(flux_obs / flux0) + imag0
        imag_obs_err = 1.087 / flux_obs_err

    data = {}
    data['t_phot1'] = t_phot
    data['mag1'] = imag_obs
    data['mag_err1'] = imag_obs_err

    data['raL'] = raL_in
    data['decL'] = decL_in
    data['target'] = 'sim'
    data['phot_data'] = 'sim'
    data['phot_files'] = ['sim']
    
    return data



def sample_model_as_curios(pspl):
    # Simulate
    # photometric observations every 1 day and
    # for the bulge observing window. Observations missed
    # for 125 days out of 365 days for photometry and missed
    t_phot = np.array([], dtype=float)
    
    for year_start in np.arange(56000, 58000, 365.25):
        phot_win = 240.0
        phot_start = (365.25 - phot_win) / 2.0
        t_phot_new = np.arange(year_start + phot_start,
                               year_start + phot_start + phot_win, 1)
        t_phot = np.concatenate([t_phot, t_phot_new])

    # Make the photometric observations.
    # Assume 0.05 mag photoemtric errors at I=19.
    # This means Signal = 400 e- at I=19.
    flux0 = 4000.0
    imag0 = 19.0
    imag_obs = pspl_par_in.get_photometry(t_phot)
    imag_obs_err = np.zeros(len(t_phot))
    if noise:
        flux_obs = flux0 * 10 ** ((imag_obs - imag0) / -2.5)
        flux_obs_err = flux_obs ** 0.5
        flux_obs += np.random.randn(len(t_phot)) * flux_obs_err
        imag_obs = -2.5 * np.log10(flux_obs / flux0) + imag0
        imag_obs_err = 1.087 / flux_obs_err

    data = {}
    data['t_phot1'] = t_phot
    data['mag1'] = imag_obs
    data['mag_err1'] = imag_obs_err

    data['raL'] = raL_in
    data['decL'] = decL_in
    data['target'] = 'sim'
    data['phot_data'] = 'sim'
    data['phot_files'] = ['sim']
    
    return data
