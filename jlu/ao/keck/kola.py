import numpy as np
import pylab as plt

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
