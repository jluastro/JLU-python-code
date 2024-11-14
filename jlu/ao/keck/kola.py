import pickle

import numpy as np
import pylab as plt
from maosi import psf
from maosi import scene
from maosi import instrument
from maosi import observation
from astropy.modeling import models
from matplotlib.colors import LogNorm
import matplotlib as mpl
import scipy.ndimage as ndimage


def plot_metrics_actcnt_lgscnt(tab, lgs_pow, wfs_rate, filt, r_ensqE):
    """
    Plot performance metrics on a grid of actuator count and LGS beacon
    count. Note, we fix the LGS power per beacon and the WFS rate.

    Parameters
    ----------
    tab : astropy.table
        Astropy table containing results. Usually produced by the
        kola_sims_grid.ipynb Jupyter notebook in this same directory.

    lgs_pow : float
        Only show results for this fixed power per LSG beacon.
        Example : 20 = 20 Watts

    wfs_rate : float
        Only show results for this fixed loop rate (in Hz).

    filt : str
        Only show metrics for this filter.

    r_ensqE : int
        Only show the encircled energy within this radius (mas).
    """
    plot_metrics3_any_pair(table=tab, axis1='act_cnt', axis2='lgs_cnt',
                           lgs_pow=lgs_pow, wfs_rate=wfs_rate, filter=filt,
                           r_ensqE=r_ensqE)
    return


def plot_metrics_actcnt_lgscnt2(tab, lgs_pow_tot, wfs_rate, filt, r_ensqE):
    """
    Plot performance metrics on a grid of actuator count and LGS beacon
    count. Note, we fix the total LGS power (summed over all beacons)
    and the WFS rate.

    Parameters
    ----------
    tab : astropy.table
        Astropy table containing results. Usually produced by the
        kola_sims_grid.ipynb Jupyter notebook in this same directory.

    lgs_pow_tot : float
        Only show results for this total LGS power.
        Example : 180 = 180 Watts or 30 W per beacon for 6 beacons.

    wfs_rate : float
        Only show results for this fixed loop rate (in Hz).

    filt : str
        Only show metrics for this filter.

    r_ensqE : int
        Only show the encircled energy within this radius (mas).
    """
    plot_metrics3_any_pair(table=tab, axis1='act_cnt', axis2='lgs_cnt',
                           lgs_pow_tot=lgs_pow_tot, wfs_rate=wfs_rate, filter=filt,
                           r_ensqE=r_ensqE)

    return

def plot_metrics_actcnt_looprate(tab, lgs_pow, lgs_cnt, filt, r_ensqE):
    """
    Plot performance metrics on a grid of actuator count and LGS beacon
    count. Note, we fix the total LGS power (summed over all beacons)
    and the WFS rate.

    Parameters
    ----------
    tab : astropy.table
        Astropy table containing results. Usually produced by the
        kola_sims_grid.ipynb Jupyter notebook in this same directory.

    lgs_pow : float
        Only show results for this LGS power per beacon.
        Example : 30 = 30 Watts per beacon or 180 W total for 6 beacons.

    lgs_cnt : int
        The number of LGS beacons.

    filt : str
        Only show metrics for this filter.

    r_ensqE : int
        Only show the encircled energy within this radius (mas).
    """
    plot_metrics3_any_pair(table=tab, axis1='act_cnt', axis2='wfs_rate',
                           lgs_pow=lgs_pow, lgs_cnt=lgs_cnt, filter=filt,
                           r_ensqE=r_ensqE)

    return


def plot_metrics3_any_pair(interpolate=False, contour_levels=None, **kwargs):
    """
    Plot metrics for any arbitrary pair of columns from the table.
    """
    labels = {'act_cnt': 'Acutator Count',
              'wfs_rate': 'Loop Rate (Hz)',
              'lgs_pow': 'LGS Power per Beacon (W)',
              'lgs_pow_tot': 'Total Laser Power (W)',
              'lgs_cnt': 'Number of LGS Beacons',
              'cost': 'Total Project Cost ($)'
              }

    units = {'act_cnt': '',
             'wfs_rate': 'Hz',
             'lgs_pow': 'W',
             'lgs_pow_tot': 'W',
             'lgs_cnt': '',
             'cost': '$'
             }

    filters = np.array(["u", "g'", "r'", "i'", "Z", "Y", "J", "H", "K'"])
    r_ee = np.array([10, 35, 50, 70, 90, 120, 240, 400, 800]) # mas

    # Get the specified filter
    if 'filter' in kwargs:
        filt = kwargs['filter']
        del kwargs['filter']
    else:
        filt = "r'"

    # Get the specified ensquared energy radius
    if 'r_ensqE' in kwargs:
        r_ensqE = kwargs['r_ensqE']
        del kwargs['r_ensqE']
    else:
        r_ensqE = 50

    # Get the table.
    tab = kwargs['table']
    del kwargs['table']

    # Only remaining keywords should be the pair of parameters of interest.
    if 'axis1' not in kwargs or 'axis2' not in kwargs:
        raise RuntimeError('Need axis1 and axis2 keywords', kwargs)

    # Build up the conditions on the table rows we want to keep.
    keep = tab['r_ensqE50'] != 0     # Keep filled rows.
    fixed_keys = []
    axis_keys = []
    for key in kwargs:
        if 'axis' in key:
            # Name of column to plot on one of the axes.
            axis_keys.append(kwargs[key])
        else:
            # Fixed parameter, value pair.
            fixed_keys.append(key)
            # Modify the condition.
            keep *= tab[key] == kwargs[key]

    print(keep.sum(), len(tab))
    tab_t = tab[keep]

    # Figure out filter and EE columns to plot
    ff = np.where(filters == filt)[0][0]  # filter index
    rr = np.where(r_ee == r_ensqE)[0][0]  # EE radius to plot

    xval = tab_t[axis_keys[0]]
    yval = tab_t[axis_keys[1]]
    strehl = tab_t['strehl'][:, ff]
    fwhm = tab_t['fwhm'][:, ff]
    ensqE = tab_t['ensqE'][:, rr]*100

    mark_size = 100

    if interpolate:
        from scipy.interpolate import interp2d
        from scipy.interpolate import CloughTocher2DInterpolator

        s_interp = CloughTocher2DInterpolator(list(zip(xval, yval)), strehl)
        f_interp = CloughTocher2DInterpolator(list(zip(xval, yval)), fwhm)
        e_interp = CloughTocher2DInterpolator(list(zip(xval, yval)), ensqE)

        # Make new grid with finer sampling.
        xval_tmp = np.linspace(xval.min(), xval.max(), 100)
        yval_tmp = np.linspace(yval.min(), yval.max(), 100)
        xval, yval = np.meshgrid(xval_tmp, yval_tmp)
        strehl = s_interp(xval, yval)
        fwhm   = f_interp(xval, yval)
        ensqE  = e_interp(xval, yval)

        mark_size = 5


    plt.figure(figsize=(16, 5))
    plt.subplots_adjust(wspace=0.5, left=0.08, bottom=0.15, top=0.85)

    # Strehl
    plt.subplot(1, 3, 1)
    plt.scatter(xval, yval, c=strehl,
                s=mark_size, marker='s', cmap='plasma')
    plt.colorbar(label=f"{filters[ff]} Strehl")
    if interpolate and contour_levels:
        con = plt.contour(xval, yval, 1.0 - (strehl / strehl.max()), contour_levels, 
                          cmap='binary_r')
        plt.clabel(con, fmt='%4.2f', fontsize=12)
    plt.xlabel(labels[axis_keys[0]])
    plt.ylabel(labels[axis_keys[1]])
    plt.title('\n'.join([f'{fkey} = {kwargs[fkey]} ({units[fkey]})' for fkey in fixed_keys]))

    # FWHM
    plt.subplot(1, 3, 2)
    plt.scatter(xval, yval, c=fwhm,
                s=mark_size, marker='s', cmap='plasma_r')
    plt.colorbar(label=f"{filters[ff]} FWHM (mas)")
    if interpolate and contour_levels:
        con = plt.contour(xval, yval, (fwhm / fwhm.min()) - 1.0, contour_levels, 
                          cmap='binary_r')
        plt.clabel(con, fmt='%4.2f', fontsize=12)
    plt.xlabel(labels[axis_keys[0]])
    plt.ylabel(labels[axis_keys[1]])
    plt.title('\n'.join([f'{fkey} = {kwargs[fkey]} ({units[fkey]})' for fkey in fixed_keys]))

    # EE
    plt.subplot(1, 3, 3)
    plt.scatter(xval, yval, c=ensqE,
                s=mark_size, marker='s', cmap='plasma')
    plt.colorbar(label=f"{filters[ff]} r={r_ee[rr]} mas Ensquared Energy (%)")
    if interpolate and contour_levels:
        con = plt.contour(xval, yval, 1.0 - (ensqE / ensqE.max()), contour_levels, 
                          cmap='binary_r')
        plt.clabel(con, fmt='%4.2f', fontsize=12)
    plt.xlabel(labels[axis_keys[0]])
    plt.ylabel(labels[axis_keys[1]])
    plt.title('\n'.join([f'{fkey} = {kwargs[fkey]} ({units[fkey]})' for fkey in fixed_keys]))

    return

def plot_star_cluster_compare_res(wave_idx=4, recalc_scene=False, recalc_psfs=False, recalc_image=False):
    # Variables MAOSI needs
    img_scale = .005  # arcseconds per pixel
    img_array_size = 6000  # pixels
    readnoise = 1.0  # e-
    dark_current = 0.05  # e-/sec
    gain = 1  # e- / ADU
    tel_diam = 10.  # m
    background = 5e-7  # e-/s/m^2

    cluster_mass = 1e5 # Msun
    cluster_fwhm = 13. # arcsec

    if recalc_scene:
        _cl = open('star_cluster_scene.pkl', 'wb')
        cl_scene = scene.SPISEAClusterScene(cluster_mass=cluster_mass, cluster_fwhm=cluster_fwhm)
        pickle.dump(cl_scene, _cl)
        _cl.close()
    else:
        _cl = open('star_cluster_scene.pkl', 'rb')
        cl_scene = pickle.load(_cl)
        _cl.close()

    if recalc_psfs:
        psf_kapa_grid = make_kapa_psf_grid_no_halo()
        psf_kola_grid = make_kola_psf_grid_with_halo()
    else:
        psf_kapa_grid = load_kapa_psf_grid_no_halo()
        psf_kola_grid = load_kola_psf_grid_with_halo()

    kapa_obs = make_kapa_obs_star_cluster(cl_scene, psf_kapa_grid, img_array_size, img_scale,
                                          gain, background, tel_diam, dark_current,
                                          readnoise, wave_idx, recalc=recalc_image)
    kola_obs = make_kola_obs_star_cluster(cl_scene, psf_kola_grid, img_array_size, img_scale,
                                          gain, background, tel_diam, dark_current,
                                          readnoise, wave_idx, recalc=recalc_image)

    # Calculate seeing-limited, GLAO-corrected.
    # Convolve the image with the seeing-limited Gaussian kernel.
    width_see = 0.6 / (img_scale * 2.355) # STD
    width_glao = width_see / (2.0 * 2.355) # STD
    image_see  = ndimage.gaussian_filter(kapa_obs.img, sigma=width_see)
    image_glao = ndimage.gaussian_filter(kapa_obs.img, sigma=width_glao)

    # 2 panel plot
    plt.close(4)
    fig, axs = plt.subplots(1, 2, figsize=(12, 6), sharex=True, sharey=True)
    plt.subplots_adjust(left=0.1, right=0.9)

    img_x = np.arange(img_array_size, dtype=float)
    img_x -= (img_array_size / 2.0)
    img_x *= img_scale # arcsec


    vmin = np.percentile(kola_obs.img, 1)
    vmax = np.percentile(kola_obs.img, 99.0)

    #vmin = np.percentile(kola_obs.img, 5)
    #vmax = np.percentile(kola_obs.img, 99.99)
    norm = mpl.colors.PowerNorm(gamma=0.5, vmin=vmin, vmax=vmax)
    axs[0].imshow(kapa_obs.img, origin='lower', norm=norm, cmap='Greys',
                  extent = (img_x.min(), img_x.max(), img_x.min(), img_x.max()))
    axs[0].set_title(f'KAPA ($\lambda = ${psf_kapa_grid.psf_wave[wave_idx]:.0f} nm)')

    #norm = LogNorm()
    #norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
    norm = mpl.colors.PowerNorm(gamma=0.5, vmin=vmin, vmax=vmax)
    axs[1].imshow(kola_obs.img, origin='lower', norm=norm, cmap='Greys',
                  extent=(img_x.min(), img_x.max(), img_x.min(), img_x.max()))
    axs[1].set_title(f'KOLA ($\lambda = ${psf_kola_grid.psf_wave[wave_idx]:.0f} nm)')

    plt.savefig(f'star_cluster_compare_{psf_kola_grid.psf_wave[wave_idx]:.0f}nm.png')

    return


def make_kapa_obs_star_cluster(cl_scene, psf_kapa_grid, img_array_size, img_scale,
                               gain, background, tel_diam, dark_current, readnoise,
                               wave_idx, recalc=False):
    outfile = f'kapa_cluster_{psf_kapa_grid.psf_wave[wave_idx]:.0f}nm.pkl'

    if not recalc:
        _out = open(outfile, 'rb')
        obs = pickle.load(_out)
        _out.close()

        return obs


    # Make the instrument
    print("Loading Instrument...\n")
    h4rg = instrument.Instrument((img_array_size, img_array_size),
                                 readnoise,
                                 dark_current,
                                 gain,
                                 tel_diam,
                                 img_scale)
    h4rg.tint = 30.0  # sec
    print("DEBUGGING: PSF Information:")
    print("-----------------------------------------")
    print(f"PSF wavelength (nm): {psf_kapa_grid.psf_wave[wave_idx]}")
    print(f"Wave array     (nm): {psf_kapa_grid.psf_wave}")
    print(f"Wave array shape: {psf_kapa_grid.wave_shape}")
    print(f"PSF grid shape: {psf_kapa_grid.grid_shape}")
    print(f"PSF pixel scale ('' / pix): {psf_kapa_grid.psf_scale}")
    print(f"fov (arcseconds): {psf_kapa_grid.fov}")
    print("-----------------------------------------\n")
    print('Making Observation:')
    obs = observation.Observation(h4rg, cl_scene, psf_kapa_grid, wave_idx, background,
                                  origin=[img_array_size / 2, img_array_size / 2])

    _out = open(outfile, 'wb')
    pickle.dump(obs, _out)
    _out.close()

    return obs


def make_kola_obs_star_cluster(cl_scene, psf_kola_grid, img_array_size, img_scale,
                               gain, background, tel_diam, dark_current, readnoise,
                               wave_idx, recalc=False):

    outfile = f'kola_cluster_{psf_kola_grid.psf_wave[wave_idx]:.0f}nm.pkl'

    if not recalc:
        _out = open(outfile, 'rb')
        obs = pickle.load(_out)
        _out.close()

        return obs


    # Make the instrument
    print("Loading Instrument...\n")
    h4rg = instrument.Instrument((img_array_size, img_array_size),
                                 readnoise,
                                 dark_current,
                                 gain,
                                 tel_diam,
                                 img_scale)
    h4rg.tint = 30.0  # sec
    print("DEBUGGING: PSF Information:")
    print("-----------------------------------------")
    print(f"PSF wavelength (nm): {psf_kola_grid.psf_wave[wave_idx]}")
    print(f"Wave array     (nm): {psf_kola_grid.psf_wave}")
    print(f"Wave array shape: {psf_kola_grid.wave_shape}")
    print(f"PSF grid shape: {psf_kola_grid.grid_shape}")
    print(f"PSF pixel scale ('' / pix): {psf_kola_grid.psf_scale}")
    print(f"fov (arcseconds): {psf_kola_grid.fov}")
    print("-----------------------------------------\n")
    print('Making Observation:')
    obs = observation.Observation(h4rg, cl_scene, psf_kola_grid, wave_idx, background,
                                  origin=[img_array_size / 2, img_array_size / 2])

    _out = open(outfile, 'wb')
    pickle.dump(obs, _out)
    _out.close()

    return obs


def make_kapa_psf_grid_no_halo():
    psf_file = '/u/jlu/work/ao/keck/maos/keck/k1_lgs_ltao_kapa/A_keck_kapa_line/'
    my_psf_grid = psf.MAOS_PSF_grid_from_line(psf_file, 1)
    my_psf_grid.plot_psf_grid(1)

    # Save the PSF grid object for easier re-loading.
    _out = open('kapa_psf_grid.pkl', 'wb')
    pickle.dump(my_psf_grid, _out)
    _out.close()

    return


def load_kapa_psf_grid_no_halo():
    # Save the PSF grid object for easier re-loading.
    _out = open('kapa_psf_grid.pkl', 'rb')
    my_psf_grid = pickle.load(_out)
    _out.close()

    return my_psf_grid


def make_kola_psf_grid_with_halo():
    """
    Make a KOLA PSF grid and save it to a pickle file. This
    PSF grid will be padded with a Moffatt halo that approximates a seeing-limited halo.
    """
    # Load the MAOS PSF
    print("Loading PSF...\n")
    psf_file = '/u/bpeck/work/mcao/vismcao/dm2_study/4000m/'
    my_psf_grid = psf.MAOS_PSF_grid_from_line(psf_file, 1)
    my_psf_grid.plot_psf_grid(1)

    # Resize and add a Moffatt
    target_shape = (300, 300)
    target_half = target_shape[0] / 2.0

    # Loop through each PSF and attach the Moffatt halo.
    # Rescale each one individually.
    new_psf_grid = np.zeros((my_psf_grid.psf.shape[0],
                             my_psf_grid.psf.shape[1], my_psf_grid.psf.shape[2],
                             target_shape[0], target_shape[1]), dtype=float)

    # Loop through and update the PSFs.
    for pp in range(my_psf_grid.psf.shape[0]):
        print(f'Working on filter {pp}')

        current_psf = my_psf_grid.psf[pp, 0, 0]

        # Get some arrays related to the current PSF.
        psf_size = current_psf.shape[0]
        psf_x1d = (np.arange(0, psf_size) - (psf_size / 2.0)) * my_psf_grid.psf_scale[pp]
        psf_y1d = psf_x1d
        psf_x2d, psf_y2d = np.meshgrid(psf_x1d, psf_y1d)
        psf_r2d = np.hypot(psf_x2d, psf_y2d)

        # Create a Moffatt-shaped PSF for the halo.
        mf_amp = 1.0
        mf_x0 = 0
        mf_y0 = 0
        mf_alpha = 2.0  # (what we typically call beta)
        mf_fwhm = 1.0  # arcsec
        mf_gamma = mf_fwhm / (2 * np.sqrt(2 ** (1 / mf_alpha) - 1))  # (what we typically call alpha)
        psf_halo_obj = models.Moffat2D(mf_amp, mf_x0, mf_y0, mf_gamma, mf_alpha)

        # Get some arrays related to the Moffatt halo PSF.
        mf_x1d = np.arange(-target_half, target_half, 1) * my_psf_grid.psf_scale[pp]  # arcsec
        mf_y1d = mf_x1d
        mf_x2d, mf_y2d = np.meshgrid(mf_x1d, mf_y1d)
        mf_r2d = np.hypot(mf_x2d, mf_y2d)
        psf_halo_2d = psf_halo_obj(mf_x2d, mf_y2d)

        # Determine the radii over which we will normalize the halo PSF.
        r_min = psf_x1d.max() * 0.8
        r_max = psf_x1d.max()
        mrdx = np.where((mf_r2d > r_min) & (mf_r2d < r_max))[0]

        # Calculate the necessary padding for each axis
        pad_height = (target_shape[0] - current_psf.shape[0]) // 2
        pad_width = (target_shape[1] - current_psf.shape[1]) // 2

        # Ensure paddings are non-negative and even
        padding = ((max(0, pad_height), max(0, pad_height)),
                   (max(0, pad_width), max(0, pad_width)))

        # Now calculate cos filter.
        # Initialize the filter with ones
        filter = np.ones_like(psf_halo_2d)

        # Apply the radial cosine transition
        mask_transition = (mf_r2d > r_min) & (mf_r2d <= r_max)
        filter[mask_transition] = 0.5 * (np.cos(np.pi * (mf_r2d[mask_transition] - r_min) / (r_max - r_min)))

        # Set outside the max radius to zero
        filter[mf_r2d > r_max] = 0
        filter[filter < 0] = 0

        for xx in range(my_psf_grid.psf.shape[1]):
            for yy in range(my_psf_grid.psf.shape[2]):
                current_psf = my_psf_grid.psf[pp, xx, yy]

                # Re-normalize the halo PSFs based on the outer 90% of the input PSF.
                psf_halo = psf_halo_2d * current_psf.min() / psf_halo_2d[mrdx].mean()

                # Apply the padding
                padded_psf = np.pad(current_psf, padding, mode='constant', constant_values=0)

                # If the target shape is odd and the padding calculated is even, this will adjust for that
                if padded_psf.shape != target_shape:
                    padded_psf = np.pad(padded_psf, ((0, target_shape[0] - padded_psf.shape[0]),
                                                     (0, target_shape[1] - padded_psf.shape[1])),
                                        mode='constant')

                new_psf_grid[pp, xx, yy] = (padded_psf * filter) + (psf_halo * (1 - filter))

    my_psf_grid.psf = new_psf_grid
    my_psf_grid.grid_shape = np.array(my_psf_grid.psf.shape[1:3])
    my_psf_grid.fov = my_psf_grid.psf.shape[3]

    # Save the PSF grid object for easier re-loading.
    _out = open('kola_psf_grid_padded.pkl', 'wb')
    pickle.dump(my_psf_grid, _out)
    _out.close()

    return


def load_kola_psf_grid_with_halo():
    # Save the PSF grid object for easier re-loading.
    _out = open('kola_psf_grid_padded.pkl', 'rb')
    my_psf_grid = pickle.load(_out)
    _out.close()

    return my_psf_grid
