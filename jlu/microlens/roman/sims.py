import numpy as np
import pylab as plt

from bagle import model
from astropy.table import Table

def make_binary_orbit_lc():
    """
    Make Roman lightcurves with binary lenses, binary sources, and orbital motion.
    Per request by Dave Bennett on 2024-07-26.
    """
    return

def generate_roman_time_samples():
    return


def make_roman_lightcurve(mod, t, mod_filt_idx=0, filter_name='F146',
                          noise=True, tint=57, verbose=False, plot=True):
    """
    Given a BAGLE model, generate photometric and astrometric data

    Parameters
    ----------
    mod : model.ModelABC
        BAGLE model instance.
    t : numpy.array
        Array of times in MJD to sample the photometry and astrometry at.

    Optional Parameters
    -------------------
    mod_filt_idx : int
        The index of the filter on the model to query.
    filter_name : str
        The name of the Roman filter. Used to set zeropoints.
    tint : float
        The integration time of the individual time stamps.
    verbose : bool
        Print out quantities along the way.
    plot : bool
        Make plots of the photometry and astrometry lightcurves.

    """
    # https: // roman.gsfc.nasa.gov / science / WFI_technical.html
    # Zeropoints are the 57 sec point source, 5 sigma.
    zeropoint_all = {'F062': 24.77, 'F087': 24.46, 'F106': 24.46, 'F129': 24.43,
                     'F158': 24.36, 'F184': 23.72, 'F213': 23.14, 'F146': 25.37}
    flux0 = 5**2 * (tint / 57.0)
    fwhm_all = {'F062': 58.0, 'F087': 73.0, 'F106': 87.0, 'F129': 106.0,
                'F158': 128., 'F184': 146., 'F213': 169., 'F146': 105.}

    zp = zeropoint_all[filter_name]  # SNR=5
    fwhm = fwhm_all[filter_name] # mas

    # Get all synthetic magnitude and positions from the model object.
    try:
        img, amp = mod.get_all_arrays(t, filt_idx=mod_filt_idx)
        mag = mod.get_photometry(t, filt_idx=mod_filt_idx, amp_arr=amp)
        ast = 1e3 * mod.get_astrometry(t, filt_idx=mod_filt_idx, image_arr=img, amp_arr=amp)

        mag = mag.reshape(len(t))
    except:
        mag = mod.get_photometry(t, filt_idx=mod_filt_idx)
        ast = 1e3 * mod.get_astrometry(t, filt_idx=mod_filt_idx)


    ##
    ## Synthetic photometry with noise. Establish a photometric floor of 0.001 mag
    ##
    flux = flux0 * 10 ** ((mag - zp) / -2.5)
    snr = flux ** 0.5
    mag_err = 1.0857 / snr
    mag_err[mag_err < 0.001] = 0.001
    if noise:
        mag += np.random.normal(0, mag_err)

    if verbose:
        print(f'Mean {filter_name} Mag = {mag.mean():.1f} +/- {mag_err.mean():.2f} mag')

    ##
    ## Synthetic Astrometry (in mas) with noise
    ##
    # Assign astrometric errors as FWHM / 2*SNR or 0.1 mas minimum.
    ast_err = fwhm / (2 * snr)  # mas
    ast_err = np.vstack([ast_err, ast_err]).T
    ast_err[ast_err < 0.1] = 0.1
    if noise:
        ast += np.random.normal(size=ast_err.shape) * ast_err

    if verbose:
        print(f'Mean {filter_name} ast err = {ast_err.mean():.2f} mas')



    ##
    ## Make an output table.
    ##
    tab = Table((t, mag, mag_err),
                     names=(f't_{filter_name}', f'm_{filter_name}', f'me_{filter_name}'))

    tab[f'x_{filter_name}'] = ast[:, 0]
    tab[f'y_{filter_name}'] = ast[:, 1]
    tab[f'xe_{filter_name}'] = ast_err[:, 0]
    tab[f'ye_{filter_name}'] = ast_err[:, 1]

    ##
    ## Plot
    ##
    if plot:
        zoom_dt = [mod.t0 - 3*mod.tE, mod.t0 + 3*mod.tE]
        print(zoom_dt, mod.t0, mod.tE)

        plt_msc = [['F1', 'AA', 'A1'],
                   ['F2', 'AA', 'A2']]
        fig, axs = plt.subplot_mosaic(plt_msc,
                                      figsize=(16, 5),
                                      tight_layout=True)
        # Photometry vs. time
        axs['F1'].errorbar(t, mag, yerr=mag_err, label=filter_name,
                           ls='none', marker='.', alpha=0.2)
        axs['F1'].set_ylabel(f'{filter_name} mag')
        axs['F1'].invert_yaxis()

        # Photometry vs. time -- zoomed
        axs['F2'].errorbar(t, mag, yerr=mag_err, ls='none', marker='.', alpha=0.2)
        axs['F2'].axvline(mod.t0, ls='-', color='grey')
        axs['F2'].axvline(mod.t0 - mod.tE, ls='--', color='grey')
        axs['F2'].axvline(mod.t0 + mod.tE, ls='--', color='grey')
        axs['F2'].set_ylabel(f'{filter_name} mag')
        axs['F2'].set_xlabel('Time (MJD)')
        axs['F2'].invert_yaxis()
        axs['F2'].set_xlim(zoom_dt)

        # Astrometry on sky
        axs['AA'].errorbar(tab[f'x_{filter_name}'], tab[f'y_{filter_name}'],
                           xerr=tab[f'xe_{filter_name}'], yerr=tab[f'ye_{filter_name}'],
                           ls='none', marker='.', alpha=0.2)
        axs['AA'].set_xlabel(f'$\Delta\\alpha \cos \delta$ (mas)')
        axs['AA'].set_ylabel(f'$\Delta\delta$ (mas)')

        # Astrometry vs. time - East
        axs['A1'].errorbar(tab[f't_{filter_name}'], tab[f'x_{filter_name}'], yerr=tab[f'xe_{filter_name}'],
                           ls='none', marker='.', alpha=0.2)
        axs['A1'].set_xlabel(f'Time (MJD)')
        axs['A1'].set_ylabel(f'$\Delta\\alpha \cos \delta$ (mas)')
        axs['A2'].sharex(axs['F1'])

        # Astrometry vs. time - North
        axs['A2'].errorbar(tab[f't_{filter_name}'], tab[f'y_{filter_name}'], yerr=tab[f'ye_{filter_name}'],
                           ls='none', marker='.', alpha=0.2)
        axs['A2'].set_xlabel(f'Time (MJD)')
        axs['A2'].set_ylabel(f'$\Delta\delta$ (mas)')
        axs['A2'].sharex(axs['F1'])

        # Print out all the parameters to the screen and in a YAML file.
        params_mod = mod.fitter_param_names + mod.phot_param_names
        params_mod_fix = mod.fixed_param_names

        # loc_vars = locals()
        # pdict_mod = {}
        # for par in params_mod:
        #     pdict_mod[par] = mod_vars[par]
        #
        # pdict_mod_fix = {}
        # for par in params_mod_fix:
        #     pdict_mod_fix[par] = loc_vars[par]
        #
        # print(pdict_mod)
        # print(pdict_mod_fix)

        #plt.savefig(f'{outdir}/roman_event_lcurves_{ff:04d}.png')

        # Make lens geometry plot.
        #plt.close('all')
        #plot_models.plot_PSBL(psbl_par, outfile=f'{outdir}/roman_event_geom_{ff:04d}.png')

        # Save parameters to YAML file.
        # param_save_file = f'{outdir}/roman_event_params_{ff:04d}.pkl'
        # param_save_data = {}
        # param_save_data['model_class'] = psbl_par.__class__
        # param_save_data['model_params'] = pdict_mod
        # param_save_data['model_params_fix'] = pdict_mod_fix
        # param_save_data['model_params_add'] = pdict_add
        #
        # with open(param_save_file, 'wb') as f:
        #     pickle.dump(param_save_data, f)
        #
        # # Save the data to an astropy FITS table. We have one for each filter.
        # tab.write(f'{outdir}/roman_event_w149_data_{ff:04d}.fits', overwrite=True)
        # tab_f087.write(f'{outdir}/roman_event_f087_data_{ff:04d}.fits', overwrite=True)
        #
        # print(tab.colnames)
        #

    return tab



