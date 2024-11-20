import pickle

import numpy as np
import pylab as plt
from astropy.modeling import models
from matplotlib.colors import LogNorm
import matplotlib as mpl
import scipy.ndimage as ndimage
from PIL import Image, ImageFilter
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
from astropy.io import fits
import scipy
from astropy.convolution import convolve_fft
import matplotlib.animation as animation

from astropy.visualization import (MinMaxInterval, SqrtStretch, ImageNormalize)

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

def plot_star_cluster_compare_res(wave_idx=4, recalc_scene=False, recalc_psfs=False,
                                  recalc_image=False, recalc_blur=False):
    # Variables MAOSI needs
    img_scale = .005  # arcseconds per pixel
    img_array_size = int(60 / img_scale) # 60" converted to pixels
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

    print('Make or load PSFs')
    if recalc_psfs:
        psf_kapa_grid = make_kapa_psf_grid_no_halo()
        psf_kola_grid = make_kola_psf_grid_with_halo()
    else:
        psf_kapa_grid = load_kapa_psf_grid_no_halo()
        psf_kola_grid = load_kola_psf_grid_with_halo()

    print('Make or load images')
    kapa_obs = make_kapa_obs_star_cluster(cl_scene, psf_kapa_grid, img_array_size, img_scale,
                                          gain, background, tel_diam, dark_current,
                                          readnoise, wave_idx, recalc=recalc_image)
    kola_obs = make_kola_obs_star_cluster(cl_scene, psf_kola_grid, img_array_size, img_scale,
                                          gain, background, tel_diam, dark_current,
                                          readnoise, wave_idx, recalc=recalc_image)

    img_x = np.arange(img_array_size, dtype=float)
    img_x -= (img_array_size / 2.0)
    img_x *= img_scale # arcsec

    if recalc_blur:
        # Calculate seeing-limited, GLAO-corrected.
        # Convolve the image with the seeing-limited Gaussian kernel.
        width_see = 0.6 / (img_scale * 2.355) # STD - equivalent to 0.6" seeing
        width_glao = width_see / 2.0 # STD - equivalent to 0.3" FWHM
        width_kapa = 0.05 / (img_scale * 2.355) # STD - equivalent to 50 mas FWHM. just smoothing.

        print('Make blurred images -- seeing-limited')
        image_see  = ndimage.gaussian_filter(kapa_obs.img, sigma=width_see)
        print('Make blurred images -- GLAO')
        image_glao = ndimage.gaussian_filter(kapa_obs.img, sigma=width_glao)
        print('Make blurred images -- KAPA')
        image_kapa = ndimage.gaussian_filter(kapa_obs.img, sigma=width_kapa)
        image_kola = kola_obs.img

        _out = open(f'star_cluster_images_{psf_kapa_grid.psf_wave[wave_idx]:.0f}nm.pkl', 'wb')
        pickle.dump(image_see, _out)
        pickle.dump(image_glao, _out)
        pickle.dump(image_kapa, _out)
        pickle.dump(image_kola, _out)
        _out.close()
    else:
        _out = open(f'star_cluster_images_{psf_kapa_grid.psf_wave[wave_idx]:.0f}nm.pkl', 'rb')
        image_see = pickle.load(_out)
        image_glao = pickle.load(_out)
        image_kapa = pickle.load(_out)
        image_kola = pickle.load(_out)
        _out.close()

    print('Plotting')
    plt.close('all')

    images = [image_see, image_glao, image_kapa, image_kola]
    titles = ['Seeing-Limited', 'GLAO', 'KAPA', 'KOLA']
    suffix = ['see', 'glao', 'kapa', 'kola']

    print('getting percentile')
    vmin = np.percentile(kola_obs.img, 0)
    vmax = np.percentile(kola_obs.img, 99.9)
    norm = mpl.colors.PowerNorm(gamma=0.5, vmin=vmin, vmax=vmax)
    if wave_idx < 6:
        cmap = 'cividis'
    else:
        cmap = 'gist_heat'

    print('Plotting images')
    for ii in range(len(images)):
        print('fubar, ii', ii)
        img = images[ii]

        plt.figure(figsize=(6,6))

        plt.imshow(img, cmap=cmap, norm=norm,
                   extent = (img_x.min(), img_x.max(), img_x.min(), img_x.max()))
        ax = plt.gca()
        ax.set_axis_off()

        plt.xlim([-2, 2])
        plt.ylim([-2, 2])

        plt.subplots_adjust(left=0.05, right=0.95, bottom=0.05, top=0.9)
        plt.title(titles[ii])

        scalebar = AnchoredSizeBar(ax.transData,
                                   1, '2 arcsec', 'lower right',
                                   pad=0.05,
                                   color='white',
                                   frameon=False,
                                   size_vertical=0.1)

        ax.add_artist(scalebar)
        plt.savefig(f'star_cluster_img_single_{suffix[ii]}_{psf_kapa_grid.psf_wave[wave_idx]:.0f}nm.png')

    # # 2 panel plot
    # fig, axs = plt.subplots(1, 2, figsize=(12, 6), sharex=True, sharey=True)
    # plt.subplots_adjust(left=0.1, right=0.9)
    #
    # vmin = np.percentile(kola_obs.img, 1)
    # vmax = np.percentile(kola_obs.img, 99.99)
    #
    # norm = mpl.colors.PowerNorm(gamma=0.5, vmin=vmin, vmax=vmax)
    # axs[0].imshow(kapa_obs.img, origin='lower', norm=norm, cmap='Greys',
    #               extent = (img_x.min(), img_x.max(), img_x.min(), img_x.max()))
    # axs[0].set_title(f'KAPA ($\lambda = ${psf_kapa_grid.psf_wave[wave_idx]:.0f} nm)')
    #
    # #norm = LogNorm()
    # #norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
    # norm = mpl.colors.PowerNorm(gamma=0.5, vmin=vmin, vmax=vmax)
    # axs[1].imshow(kola_obs.img, origin='lower', norm=norm, cmap='Greys',
    #               extent=(img_x.min(), img_x.max(), img_x.min(), img_x.max()))
    # axs[1].set_title(f'KOLA ($\lambda = ${psf_kola_grid.psf_wave[wave_idx]:.0f} nm)')
    #
    # plt.savefig(f'star_cluster_compare_{psf_kola_grid.psf_wave[wave_idx]:.0f}nm.png')

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


def make_kapa_psf_grid_no_halo(seed=1):
    psf_file = '/Users/jlu/work/ao/keck/maos/kapa_psf_grid/A_keck_kapa_grid/'
    my_psf_grid = psf.MAOS_PSF_grid_from_quadrant(psf_file, seed)
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


def make_kola_psf_grid_with_halo(seed=1):
    """
    Make a KOLA PSF grid and save it to a pickle file. This
    PSF grid will be padded with a Moffatt halo that approximates a seeing-limited halo.
    """
    # Load the MAOS PSF
    print("Loading PSF...\n")
    psf_file = '/Users/jlu/work/ao/keck/maos/kola_psf_grid/A_4000_8x7mag_60asdiam_tno/'
    #psf_file = '/u/bpeck/work/mcao/vismcao/dm2_study/4000m/'
    my_psf_grid = psf.MAOS_PSF_grid_from_quadrant(psf_file, seed)
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

def blur_galaxies(input='jwst_galaxies.tiff'):
    from astropy.convolution import convolve, Moffat2DKernel

    # Total guess at image scale based on mega-zoom of images and knowing
    # that the optical is from HST.
    img_scale = 0.025  # mas/pixel

    # # Open the image
    # image = np.array(Image.open(input))
    #
    # # Define the blurring kernels for Seeing-Limited
    # kern_fwhm = 0.6 / img_scale # pix
    # kern_beta = 5.0 # also known as alpha in Astropy -- the power
    # kern_gamma = kern_fwhm / (2.0 * (2**(1./kern_beta) - 1)**0.5) # pix
    # kernel_see = Moffat2DKernel(kern_gamma, kern_beta)
    #
    # # Define the blurring kernels for GLAO
    # kern_fwhm /= 2.0 # pix -- assumes 2x seeing-enhancement
    # kern_beta = 3.0 # also known as alpha in Astropy -- the power. GLAO flattens the wings.
    # kern_gamma = kern_fwhm / (2.0 * (2**(1./kern_beta) - 1)**0.5) # pix
    # kernel_glao = Moffat2DKernel(kern_gamma, kern_beta)
    #
    # img_see = image.copy()
    # img_glao = image.copy()
    # for cc in range(image.shape[-1]):
    #     img_see[:, :, cc] = convolve(image[:, :, cc], kernel_see, normalize_kernel=True)
    #     img_glao[:, :, cc] = convolve(image[:, :, cc], kernel_glao, normalize_kernel=True)

    # Fast Way
    image = Image.open(input)
    img_see  = image.filter(ImageFilter.GaussianBlur(0.6 / (img_scale * 2.355)))
    img_glao = image.filter(ImageFilter.GaussianBlur(0.3 / (img_scale * 2.355)))

    ext_x = np.array([0, image.size[0]]) * img_scale
    ext_x -= ext_x[-1] / 2.0
    ext_y = np.array([0, image.size[1]]) * img_scale
    ext_y -= ext_y[-1] / 2.0

    images = [img_see, img_glao, image]
    titles = ['Seeing-Limited, Visible', 'GLAO, Visible', '50 mas']
    suffix = ['see', 'glao', 'orig']

    for ii in range(len(images)):
        img = images[ii]

        plt.figure()
        plt.imshow(img, extent=(ext_x[0], ext_x[1], ext_y[0], ext_y[1]))
        ax = plt.gca()
        ax.set_axis_off()

        plt.subplots_adjust(left=0.05, right=0.95, bottom=0.05, top=0.9)
        plt.title(titles[ii])

        plt.xlim([-10, 10])
        plt.ylim([-10, 10])

        scalebar = AnchoredSizeBar(ax.transData,
                               5, '5 arcsec', 'lower right',
                               pad=0.1,
                               color='white',
                               frameon=False,
                               size_vertical=0.1)

        ax.add_artist(scalebar)
        plt.savefig(input.replace('.tiff', f'_{suffix[ii]}.png'))

    return


def plot_microlens_event_compare(kola_psf_wave_idx=5, kapa_psf_wave_idx=None, remake_images = True):
    from bagle import model
    from matplotlib import patches
    from skimage.draw import polygon

    # KOLA
    # evl.wvl=[0.432e-6 0.544e-6 0.652e-6 0.810e-6 0.877e-6 1.020e-6 1.248e-6 1.673e-6 2.200e-6]
    # KAPA
    # evl.wvl=[0.8e-6 1.0e-6 1.25e-6 1.65e-6 2.12e-6]
    kola2kapa_idx = {0:0, 1:0, 2:0, 3:0, 4:0, 5:1, 6:2, 7:3, 8:4}
    if kapa_psf_wave_idx is None:
        kapa_psf_wave_idx = kola2kapa_idx[kola_psf_wave_idx]

    mL = 20.0  # msun
    t0 = 60000  # MJD
    beta = 7.0  # mas
    dL = 1000.0  # pc
    dS = 8000.0  # pc
    xS0_E = 0.0  # arcsec
    xS0_N = beta / 1e3  # arcsec
    muL_E = 0.0  # mas/yr
    muL_N = 0.0  # mas/yr
    muS_E = 40.0  # mas/yr
    muS_N = 0.0  # mas/yr
    radiusS = 4.0  # mas
    b_sff = 1.0
    mag_src = 20.0
    n_outline = 30
    raL = 17.75 * 15.0
    decL = -29.0

    fspl = model.FSPL_PhotAstrom_noPar_Param1(mL, t0, beta, dL, dS,
                                              xS0_E, xS0_N,
                                              muL_E, muL_N, muS_E, muS_N,
                                              radiusS,
                                              b_sff, mag_src, n_outline,
                                              raL=raL, decL=decL)

    # dt = 1
    # dt_max = 1
    dt = 20
    dt_max = 400
    t_obs = np.arange(60000 - dt_max, 60001 + dt_max, dt)
    xS_images = fspl.get_resolved_astrometry_outline(t_obs) * 1e3  # mas
    xS = fspl.get_astrometry_unlensed(t_obs) * 1e3  # mas
    xL = fspl.get_lens_astrometry(t_obs) * 1e3  # mas
    A = fspl.get_resolved_amplification(t_obs).T
    thetaE = fspl.thetaE_amp  # mas

    kapa_psf_file = '/Users/jlu/work/ao/keck/maos/keck_maos/base/A_keck_scao_lgs/'
    kapa_psf_file += 'evlpsfcl_1_x0_y0.fits'
    #kapa_psf_file = '/Users/jlu/work/ao/keck/maos/kapa_psf_grid/A_keck_kapa_grid/'
    #kapa_psf_file += 'evlpsfcl_1731523196_x0_y0.fits'
    kola_psf_file = '/Users/jlu/work/ao/keck/maos/kola_psf_grid/A_4000_8x7mag_60asdiam_tno/'
    kola_psf_file += 'evlpsfcl_1_x0_y0.fits'

    kola_psf_fits = fits.open(kola_psf_file)
    kola_psf = kola_psf_fits[kola_psf_wave_idx].data
    kola_psf_hdr = kola_psf_fits[kola_psf_wave_idx].header
    kola_psf_scale = kola_psf_hdr['DP'] * 1e3  # mas/pixel
    kola_psf_wave = kola_psf_hdr['wvl'] * 1e9  # nm

    kapa_psf_fits = fits.open(kapa_psf_file)
    kapa_psf = kapa_psf_fits[kapa_psf_wave_idx].data
    kapa_psf_hdr = kapa_psf_fits[kapa_psf_wave_idx].header
    kapa_psf_scale = kapa_psf_hdr['DP'] * 1e3  # mas/pixel
    kapa_psf_wave = kapa_psf_hdr['wvl'] * 1e9  # nm

    img_scale = 2  # mas/pix sampling
    img_size = 100  # pixels (about 200 mas)
    kola_rescale = kola_psf_scale / img_scale
    kapa_rescale = kapa_psf_scale / img_scale
    print(f'image scale = {img_scale:.2f} mas/pix')
    print(f'orig psf scale = {kola_psf_scale:.2f} mas/pix')
    print(f'kola rescale = {kola_rescale:.2f}')
    print(f'kapa rescale = {kapa_rescale:.2f}')
    print(f'image size = {img_size} pix')
    print(f'thetaE = {thetaE:.2f} mas')
    print(f'tE = {fspl.tE:.2f} days')

    if remake_images:
        # Make an empty image.

        img = np.zeros((len(t_obs), img_size, img_size), dtype=float)
        kola_img_c = np.zeros((len(t_obs), img_size, img_size), dtype=float)
        kapa_img_c = np.zeros((len(t_obs), img_size, img_size), dtype=float)

        # # Rescale the PSF, cutof first 2 pixels to recenter.
        kola_psf = scipy.ndimage.zoom(kola_psf, zoom=kola_rescale, order=3)
        kola_trim_ii = int(kola_rescale)
        kola_psf = kola_psf[kola_trim_ii:, kola_trim_ii + 1:-1]  # why is it different in x and y?

        # # Rescale the PSF, cutof first 2 pixels to recenter.
        kapa_psf = scipy.ndimage.zoom(kapa_psf, zoom=kapa_rescale, order=3)
        kapa_trim_ii = int(kapa_rescale)
        kapa_psf = kapa_psf[kapa_trim_ii:, kapa_trim_ii + 1:-1]  # why is it different in x and y?

        for tt in range(len(t_obs)):
            # Make the image polygons
            poly_p_verts = np.append(xS_images[tt, :, 0, :],
                                 [xS_images[tt, 0, 0, :]], axis=0)
            poly_n_verts = np.append(xS_images[tt, :, 1, :],
                                 [xS_images[tt, 0, 1, :]], axis=0)
            poly_p_verts *= 1.0 / img_scale
            poly_n_verts *= 1.0 / img_scale
            poly_p_verts += img_size / 2.0
            poly_n_verts += img_size / 2.0

            rr_p, cc_p = polygon(poly_p_verts[:, 1], poly_p_verts[:, 0], img[tt].shape)
            rr_n, cc_n = polygon(poly_n_verts[:, 1], poly_n_verts[:, 0], img[tt].shape)
            img[tt, rr_p, cc_p] = A[tt, 0] / len(rr_p)
            img[tt, rr_n, cc_n] = A[tt, 1] / len(rr_n)
            print(f'image {tt} flux after = {img[tt].sum()}, pix_cnt = {len(rr_p)}')

            # Convolve our image with the PSF.
            kola_img_c[tt, :, :] = convolve_fft(img[tt, :, :], kola_psf, boundary='wrap')
            kapa_img_c[tt, :, :] = convolve_fft(img[tt, :, :], kapa_psf, boundary='wrap')

        _pkl = open(f'microlens_data_kapa{kapa_psf_wave:04.0f}nm_kola{kola_psf_wave:04.0f}nm.pkl', 'wb')
        pickle.dump(img, _pkl)
        pickle.dump(kapa_img_c, _pkl)
        pickle.dump(kola_img_c, _pkl)
        _pkl.close()

    else:
        _pkl = open(f'microlens_data_kapa{kapa_psf_wave:04.0f}nm_kola{kola_psf_wave:04.0f}nm.pkl', 'rb')
        img = pickle.load(_pkl)
        kapa_img_c = pickle.load(_pkl)
        kola_img_c = pickle.load(_pkl)
        _pkl.close()


    ##########
    # Plot
    ##########
    ##
    ## Plot schematic.
    ##
    plt.close('all')
    fig1 = plt.figure(1)
    plt.clf()

    plt.plot([0], [0], 'k.')

    # Plot Enstein radius in mas and the black hole.
    f1_thetaE = plt.gca().add_patch(patches.Circle(xL[0], thetaE,
                                                   ec='purple', fc='none'))
    f1_xL = plt.gca().add_patch(patches.Circle(xL[0], thetaE / 20.0,
                                               ec='none', fc='black'))
    f1_xS = plt.gca().add_patch(patches.Circle(xS[0], thetaE / 20.0,
                                               ec='red', fc='none'))

    f1_img_p = patches.Polygon(xS_images[0, :, 0, :], fc='orange', ec='darkorange')
    f1_img_n = patches.Polygon(xS_images[0, :, 1, :], fc='orange', ec='darkorange')
    plt.gca().add_patch(f1_img_p)
    plt.gca().add_patch(f1_img_n)

    plt.axis('equal')
    lim = thetaE * 2
    plt.xlim(-lim, lim)
    plt.ylim(-lim, lim)
    plt.xlabel('(mas)')
    plt.ylabel('(mas)')
    plt.text(0.05, 0.95, f'M$_{{BH}}$ = {mL:.0f} M$_\odot$', color='black',
             transform=fig1.gca().transAxes, fontsize=14)
    plt.text(0.05, 0.90, f't$_{{E}}$ = {fspl.tE:.0f} days', color='black',
             transform=fig1.gca().transAxes, fontsize=14)
    plt.text(0.05, 0.85, f'$\\theta_{{E}}$ = {thetaE:.0f} mas', color='black',
             transform=fig1.gca().transAxes, fontsize=14)
    plt.text(0.05, 0.80, f'$\lambda_{{PSF}}$ = {kola_psf_hdr["wvl"] * 1e3:.0f} nm', color='black',
             transform=fig1.gca().transAxes, fontsize=14)

    ##
    ## Plot the intrinsic image.
    ##
    fig2 = plt.figure(2)
    plt.clf()
    img_ext = np.array([-img[0].shape[0], img[0].shape[0]]) * img_scale / 2.0
    img_ext = np.append(img_ext, img_ext)
    f2_img = plt.imshow(img[0, :, :], cmap='binary_r', extent=img_ext)

    # Plot Enstein radius in mas, the black hole, and the true source position.
    f2_thetaE = plt.gca().add_patch(patches.Circle(xL[0], thetaE,
                                                   ec='cyan', fc='none', ls='--'))
    f2_xL = plt.gca().add_patch(patches.Circle(xL[0], thetaE / 20.0,
                                               ec='lightgrey', fc='black'))
    f2_xS = plt.gca().add_patch(patches.Circle(xS[0], thetaE / 20.0,
                                               ec='yellow', fc='yellow'))
    plt.xlabel('(mas)')
    plt.ylabel('(mas)')
    plt.title('Intrinsic Flux')

    ##
    ## Plot the KOLA PSF
    ##
    fig3 = plt.figure(3)
    plt.clf()
    psf_ext = np.array([-kola_psf.shape[0], kola_psf.shape[0]]) * img_scale / 2.0
    psf_ext = np.append(psf_ext, psf_ext)
    plt.imshow(kola_psf, cmap='binary_r', extent=psf_ext)
    plt.xlim(img_ext[0], img_ext[1])
    plt.ylim(img_ext[0], img_ext[1])
    plt.xlabel('(mas)')
    plt.ylabel('(mas)')
    plt.title('KOLA PSF')

    ##
    ## Plot the KOLA PSF
    ##
    fig3 = plt.figure(6)
    plt.clf()
    psf_ext = np.array([-kapa_psf.shape[0], kapa_psf.shape[0]]) * img_scale / 2.0
    psf_ext = np.append(psf_ext, psf_ext)
    plt.imshow(kapa_psf, cmap='binary_r', extent=psf_ext)
    plt.xlim(img_ext[0], img_ext[1])
    plt.ylim(img_ext[0], img_ext[1])
    plt.xlabel('(mas)')
    plt.ylabel('(mas)')
    plt.title('KAPA PSF')

    ##
    ## Plot the convolved image.
    ##
    fig4 = plt.figure(4)
    plt.clf()
    f4_img = plt.imshow(kola_img_c[0, :, :], cmap='binary_r', extent=img_ext)

    # Plot Enstein radius in mas, the black hole, and the true source position.
    f4_thetaE = plt.gca().add_patch(patches.Circle(xL[0], thetaE,
                                                   ec='cyan', fc='none', ls='--'))
    f4_xL = plt.gca().add_patch(patches.Circle(xL[0], thetaE / 20.0,
                                               ec='lightgrey', fc='black'))
    f4_xS = plt.gca().add_patch(patches.Circle(xS[0], thetaE / 20.0,
                                               ec='yellow', fc='yellow'))
    plt.xlabel('(mas)')
    plt.ylabel('(mas)')
    plt.title('Observed Flux')
    plt.text(0.05, 0.95, f'M$_{{BH}}$ = {mL:.0f} M$_\odot$', color='white',
             transform=fig4.gca().transAxes, fontsize=14)
    plt.text(0.05, 0.90, f't$_{{E}}$ = {fspl.tE:.0f} days', color='white',
             transform=fig4.gca().transAxes, fontsize=14)
    plt.text(0.05, 0.85, f'$\\theta_{{E}}$ = {thetaE:.0f} mas', color='white',
             transform=fig4.gca().transAxes, fontsize=14)
    plt.text(0.05, 0.80, f'$\lambda_{{PSF}}$ = {kola_psf_hdr["wvl"] * 1e6:.0f} nm', color='white',
             transform=fig4.gca().transAxes, fontsize=14)

    ##
    ## Plot 3 panel
    ##
    fig5 = plt.figure(5, figsize=(18, 6))
    plt.clf()
    f5a1 = plt.subplot(1, 3, 1)
    f5a2 = plt.subplot(1, 3, 2)
    f5a3 = plt.subplot(1, 3, 3)
    plt.subplots_adjust(left=0.1, right=0.95)

    #
    # Left Panel - Schematic
    #
    f5a1.plot([0], [0], 'k.')

    # Plot Enstein radius in mas and the black hole.
    f5a1_thetaE = f5a1.add_patch(patches.Circle(xL[0], thetaE,
                                                ec='purple', fc='none'))
    f5a1_xL = f5a1.add_patch(patches.Circle(xL[0], thetaE / 20.0,
                                            ec='none', fc='black'))
    f5a1_xS = f5a1.add_patch(patches.Circle(xS[0], thetaE / 20.0,
                                            ec='red', fc='none'))

    f5a1_img_p = patches.Polygon(xS_images[0, :, 0, :], fc='orange', ec='darkorange')
    f5a1_img_n = patches.Polygon(xS_images[0, :, 1, :], fc='orange', ec='darkorange')
    f5a1.add_patch(f5a1_img_p)
    f5a1.add_patch(f5a1_img_n)

    lim = thetaE * 2
    f5a1.set_xlabel('(mas)')
    f5a1.set_ylabel('(mas)')
    f5a1.set_title('Schematic')
    f5a1.text(0.05, 0.95, f'M$_{{BH}}$ = {mL:.0f} M$_\odot$', color='black',
              transform=f5a1.transAxes, fontsize=14)
    f5a1.text(0.05, 0.90, f't$_{{E}}$ = {fspl.tE:.0f} days', color='black',
              transform=f5a1.transAxes, fontsize=14)
    f5a1.text(0.05, 0.85, f'$\\theta_{{E}}$ = {thetaE:.0f} mas', color='black',
              transform=f5a1.transAxes, fontsize=14)
    f5a1.text(0.05, 0.80, f'$\lambda_{{PSF}}$ = {kola_psf_hdr["wvl"] * 1e3:.0f} nm',
              color='black',
              transform=f5a1.transAxes, fontsize=14)
    f5a1.set_title('Microlens Geometry')

    #
    # Middle Panel
    #
    # Create an ImageNormalize object
    kapa_norm = ImageNormalize(kapa_img_c, interval=MinMaxInterval(), stretch=SqrtStretch())

    f5a2_img = f5a2.imshow(kapa_img_c[0, :, :], norm=kapa_norm, cmap='Greys', extent=img_ext)

    # Plot Enstein radius in mas, the black hole, and the true source position.
    f5a2_thetaE = f5a2.add_patch(patches.Circle(xL[0], thetaE,
                                                ec='cyan', fc='none', ls='--'))
    f5a2_xL = f5a2.add_patch(patches.Circle(xL[0], thetaE / 20.0,
                                            ec='lightgrey', fc='black'))
    f5a2_xS = f5a2.add_patch(patches.Circle(xS[0], thetaE / 20.0,
                                            ec='yellow', fc='yellow'))
    f5a2.set_xlabel('(mas)')
    f5a2.set_title(f'KAPA $\lambda=${kapa_psf_wave:.0f}nm')

    #
    # Right Panel
    #
    kola_norm = ImageNormalize(kola_img_c, interval=MinMaxInterval(), stretch=SqrtStretch())

    f5a3_img = f5a3.imshow(kola_img_c[0, :, :], norm=kola_norm, cmap='Greys', extent=img_ext)

    # Plot Enstein radius in mas, the black hole, and the true source position.
    f5a3_thetaE = f5a2.add_patch(patches.Circle(xL[0], thetaE,
                                                ec='cyan', fc='none', ls='--'))
    f5a3_xL = f5a2.add_patch(patches.Circle(xL[0], thetaE / 20.0,
                                            ec='lightgrey', fc='black'))
    f5a3_xS = f5a2.add_patch(patches.Circle(xS[0], thetaE / 20.0,
                                            ec='yellow', fc='yellow'))
    f5a3.set_xlabel('(mas)')
    f5a3.set_title(f'KOLA $\lambda=${kola_psf_wave:.0f}nm')



    f5a1.axis('equal')
    f5a1.set_xlim(img_ext[0], img_ext[1])
    f5a1.set_ylim(img_ext[0], img_ext[1])
    f5a2.axis('equal')
    f5a2.set_xlim(img_ext[0], img_ext[1])
    f5a2.set_ylim(img_ext[0], img_ext[1])
    f5a3.axis('equal')
    f5a3.set_xlim(img_ext[0], img_ext[1])
    f5a3.set_ylim(img_ext[0], img_ext[1])

    ##########
    # Animate
    ##########

    plt_objs1 = [f1_thetaE, f1_xL, f1_xS, f1_img_p, f1_img_n]
    plt_objs2 = [f2_thetaE, f2_xL, f2_xS, f2_img]
    plt_objs4 = [f4_thetaE, f4_xL, f4_xS, f4_img]
    plt_objs5 = [f5a1_thetaE, f5a1_xL, f5a1_xS, f5a1_img_p, f5a1_img_n,
                 f5a2_thetaE, f5a2_xL, f5a2_xS, f5a2_img,
                 f5a3_thetaE, f5a3_xL, f5a3_xS, f5a3_img]

    def f1_update(t, xL, xS, p_outline, n_outline, plt_objs1):
        f1_thetaE, f1_xL, f1_xS, f1_img_p, f1_img_n = plt_objs1
        f1_thetaE.center = xL[t]
        f1_xL.center = xL[t]
        f1_xS.center = xS[t]
        f1_img_p.xy = p_outline[t]
        f1_img_n.xy = n_outline[t]
        return plt_objs1

    def f2_update(t, xL, xS, img, plt_objs2):
        f2_thetaE, f2_xL, f2_xS, f2_img = plt_objs2
        f2_thetaE.center = xL[t]
        f2_xL.center = xL[t]
        f2_xS.center = xS[t]
        f2_img.set_array(img[t])
        return plt_objs2

    def f4_update(t, xL, xS, img_c, plt_objs4):
        f4_thetaE, f4_xL, f4_xS, f4_img = plt_objs4
        f4_thetaE.center = xL[t]
        f4_xL.center = xL[t]
        f4_xS.center = xS[t]
        f4_img.set_array(img_c[t])
        return plt_objs4

    def f5_update(t, xL, xS, p_outline, n_outline, kapa_img_c, kola_img_c, plt_objs5):
        f5a1_thetaE, f5a1_xL, f5a1_xS, f5a1_img_p, f5a1_img_n, f5a2_thetaE, f5a2_xL, f5a2_xS, f5a2_img, f5a3_thetaE, f5a3_xL, f5a3_xS, f5a3_img = plt_objs5

        f5a1_thetaE.center = xL[t]
        f5a1_xL.center = xL[t]
        f5a1_xS.center = xS[t]
        f5a1_img_p.xy = p_outline[t]
        f5a1_img_n.xy = n_outline[t]

        f5a2_thetaE.center = xL[t]
        f5a2_xL.center = xL[t]
        f5a2_xS.center = xS[t]
        f5a2_img.set_array(kapa_img_c[t])

        f5a3_thetaE.center = xL[t]
        f5a3_xL.center = xL[t]
        f5a3_xS.center = xS[t]
        f5a3_img.set_array(kola_img_c[t])

        return plt_objs5

    p_outline = xS_images[:, :, 0, :]
    n_outline = xS_images[:, :, 1, :]
    frame_time = 100  # ms

    ani1 = animation.FuncAnimation(fig1, f1_update, len(t_obs),
                                   fargs=[xL, xS, p_outline, n_outline, plt_objs1],
                                   blit=True, interval=frame_time)
    ani1.save(f'fspl_schematic_{kola_psf_wave:04.0f}nm.gif')

    ani2 = animation.FuncAnimation(fig2, f2_update, len(t_obs),
                                   fargs=[xL, xS, img, plt_objs2],
                                   blit=True, interval=frame_time)
    ani2.save(f'fspl_image_raw_{kola_psf_wave:04.0f}nm.gif')

    ani4 = animation.FuncAnimation(fig4, f4_update, len(t_obs),
                                   fargs=[xL, xS, kola_img_c, plt_objs4],
                                   blit=True, interval=frame_time)
    ani4.save(f'fspl_image_conv_{kola_psf_wave:04.0f}nm.gif')

    ani5 = animation.FuncAnimation(fig5, f5_update, len(t_obs),
                                   fargs=[xL, xS, p_outline, n_outline, kapa_img_c, kola_img_c, plt_objs5],
                                   blit=True, interval=frame_time)
    ani5.save(f'fspl_3panel_kapa{kapa_psf_wave:04.0f}nm_kola{kola_psf_wave:04.0f}nm.gif')

    return fspl
