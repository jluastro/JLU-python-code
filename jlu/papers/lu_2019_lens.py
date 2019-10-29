import numpy as np
import pylab as plt
from astropy.table import Table, Column, vstack
from astropy.io import fits
from flystar import starlists
from scipy.interpolate import UnivariateSpline
from scipy.optimize import curve_fit, least_squares
import matplotlib.ticker
import matplotlib.colors
from matplotlib.pylab import cm
from matplotlib.colors import Normalize, LogNorm
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable
from mpl_toolkits.axes_grid1.colorbar import colorbar
from matplotlib.ticker import NullFormatter
import os
from scipy.ndimage import gaussian_filter as norm_kde
from microlens.jlu import model_fitter, multinest_utils, multinest_plot, munge_ob150211, munge_ob150029, model, munge
import dynesty.utils as dyutil
from matplotlib.colors import LinearSegmentedColormap, colorConverter
import pdb
import pickle
import math
from scipy.stats import norm

mpl_o = '#ff7f0e'
mpl_b = '#1f77b4'
mpl_g = '#2ca02c'
mpl_r = '#d62728'

ep_ob120169 = ['12jun',   '12jul',   '13apr',   '13jul', '15may05',
               '15jun07', '16may24', '16jul14']

ep_ob150211 = ['15may05', '15jun07', '15jun28', '15jul23', '16may03',
               '16jul14', '16aug02', '17jun05', '17jun08', '17jul19',
               '18may11', '18aug02', '18aug16']

ep_ob150029 = ['15jun07', '15jul23', '16may24', '16jul14', '17may21',
               '17jul14', '17jul19', '18aug21']

ep_ob140613 = ['15jun07', '15jun28', '16apr17', '16may24', '16aug02',
               '17jun05', '17jul14', '18may11', '18aug16']


epochs = {'ob120169': ep_ob120169, 'ob140613': ep_ob140613, 'ob150029': ep_ob150029, 'ob150211': ep_ob150211}

# paper_dir = '/u/jlu/doc/papers/ob150211/'
paper_dir = '/u/jlu/doc/papers/2015_bh_lenses/'

mlens_dir = '/u/jlu/work/microlens/'

a_date = {'ob120169': '2019_06_26',
          'ob140613': '2019_06_26',
          'ob150029': '2019_06_26',
          'ob150211': '2019_06_26'}

comp_stars = {'ob120169': ['ob120169_L', 'S24_18_0.8'],
              'ob140613': ['S002_15_0.7', 'S001_15_0.9'],
              'ob150029': ['S002_16_0.3', 'S003_16_0.9'],
              'ob150211': ['S001_11_1.3', 'S003_14_1.4']}

astrom_pass = {'ob120169': 'p5',
               'ob140613': 'p5',
               'ob150029': 'p4',
               'ob150211': 'p5'}

a_dir = {}
astrom_data = {}

for targ in a_date:
    a_dir[targ] = mlens_dir + targ.upper() + '/a_' + a_date[targ] + '/'
    astrom_data[targ] = a_dir[targ] + targ + '_astrom_' + astrom_pass[targ] + '_' + a_date[targ] + '.fits'

photom_spitzer = {'ob120169': None,
                  'ob140613': '/g/lu/data/microlens/spitzer/calchi_novati_2015/ob140613_phot_2.txt',
                  'ob150029': '/g/lu/data/microlens/spitzer/calchi_novati_2015/ob150029_phot_2.txt',
                  'ob150211': '/g/lu/data/microlens/spitzer/calchi_novati_2015/ob150211_phot_3.txt'}

    
pspl_phot = {'ob120169' : a_dir['ob120169'] + 'model_fits/3_fit_phot_parallax/u0_plusminus/aa_',
             'ob140613' : a_dir['ob140613'] + 'model_fits/3_fit_phot_parallax/u0_plusminus/aa_',
             'ob150029' : a_dir['ob150029'] + 'model_fits/3_fit_phot_parallax/u0_plusminus/aa_',
             'ob150211' : a_dir['ob150211'] + 'model_fits/3_fit_phot_parallax/u0_plusminus/aa_'}

pspl_ast_multiphot = {'ob120169' : a_dir['ob120169'] + 'model_fits/8_fit_multiphot_astrom_parallax2/aa_',
                      'ob140613' : a_dir['ob140613'] + 'model_fits/8_fit_multiphot_astrom_parallax2/aa_',
                      'ob150029' : a_dir['ob150029'] + 'model_fits/8_fit_multiphot_astrom_parallax2/aa_',
                      'ob150211' : a_dir['ob150211'] + 'model_fits/8_fit_multiphot_astrom_parallax2/dd_'}

pspl_multiphot = {'ob120169' : a_dir['ob120169'] + 'model_fits/9_fit_multiphot_only_parallax/bb_',
                  'ob140613' : a_dir['ob140613'] + 'model_fits/9_fit_multiphot_only_parallax/bb_',
                  'ob150029' : a_dir['ob150029'] + 'model_fits/9_fit_multiphot_only_parallax/aa_',
                  'ob150211' : a_dir['ob150211'] + 'model_fits/9_fit_multiphot_only_parallax/aa_'}

def all_paper():
    plot_images()
    make_obs_table()
    plot_pos_err()

    plot_ob120169_phot_ast()
    plot_ob140613()
    plot_ob150029_phot_ast()
    plot_ob150211_phot_ast()

    tE_piE()
    
    return
    
def make_obs_table():
    """
    Make a LaTeX table for all of the observations of the three targets from 2014/2015.
    """

    targets = list(epochs.keys())

    tables = {}

    # Build three different tables with all the columns we need.
    for ii in range(len(targets)):
        target = targets[ii]
        n_epochs = len(epochs[target])

        obj_name = np.repeat(target.upper(), n_epochs)
        obj_name[1:] = ''

        date = np.zeros(n_epochs, dtype='S10')
        tint = np.zeros(n_epochs, dtype=int)
        n_exp = np.zeros(n_epochs, dtype=int)
        strehl = np.zeros(n_epochs, dtype=float)
        fwhm = np.zeros(n_epochs, dtype=float)
        strehl_e = np.zeros(n_epochs, dtype=float)
        fwhm_e = np.zeros(n_epochs, dtype=float)
        n_star = np.zeros(n_epochs, dtype=int)
        m_base = np.zeros(n_epochs, dtype=float)
        ast_err = np.zeros(n_epochs, dtype=float)
        phot_err = np.zeros(n_epochs, dtype=float)

        # Loop through each epoch and grab information to populate our table.
        for ee in range(n_epochs):
            epoch = epochs[target][ee]
            img_file = '/g/lu/data/microlens/{0:s}/combo/mag{0:s}_{1:s}_kp.fits'.format(epoch, target)
            log_file = '/g/lu/data/microlens/{0:s}/combo/mag{0:s}_{1:s}_kp.log'.format(epoch, target)
            pos_file = '/g/lu/data/microlens/{0:s}/combo/starfinder/plotPosError_{1:s}_kp.txt'.format(epoch, target)

            # Fetch stuff from the image header.
            hdr = fits.getheader(img_file)
            date[ee] = hdr['DATE-OBS'].strip()
            tint[ee] = np.round(float(hdr['ITIME']) * float(hdr['COADDS']), 0)

            # From the log file, average Strehl and FWHM
            _log = Table.read(log_file, format='ascii')
            _log.rename_column('col2', 'fwhm')
            _log.rename_column('col3', 'strehl')
            strehl[ee] = _log['strehl'].mean()
            strehl_e[ee] = _log['strehl'].std()
            fwhm[ee] = _log['fwhm'].mean()
            fwhm_e[ee] = _log['fwhm'].std()
            n_exp[ee] = len(_log)

            # Read in the stats file from the analysis of the AIROPA starlist.
            _pos = open(pos_file, 'r')
            lines = _pos.readlines()
            _pos.close()

            n_star[ee] = int(lines[0].split(':')[-1])
            ast_err[ee] = float(lines[1].split(':')[-1])
            phot_err[ee] = float(lines[2].split(':')[-1])
            m_base[ee] = float(lines[3].split('=')[-1])

        # Make our table
        c_obj_name = Column(data=obj_name, name='Object', format='{:13s}')
        c_date = Column(data=date, name='Date', format='{:10s}')
        c_tint = Column(data=tint, name='t$_{int}$', format='{:3.0f}', unit='s')
        c_nexp = Column(data=n_exp, name='N$_{exp}$', format='{:3d}')
        c_fwhm = Column(data=fwhm, name='FWHM', format='{:3.0f}', unit='mas')
        c_fwhm_err = Column(data=fwhm_e, name='FWHM$_{err}$', format='{:3.0f}', unit='mas')
        c_strehl = Column(data=strehl, name='Strehl', format='{:4.2f}')
        c_strehl_err = Column(data=strehl_e, name='Strehl$_{err}$', format='{:4.2f}')
        c_nstar = Column(data=n_star, name='N$_{star}$', format='{:4d}')
        c_mbase = Column(data=m_base, name='Kp$_{turn}$', format='{:4.1f}', unit='mag')
        c_asterr = Column(data=ast_err, name='$\sigma_{ast}$', format='{:5.2f}', unit='mas')
        c_photerr = Column(data=phot_err, name='$\sigma_{phot}$', format='{:5.2f}', unit='mag')

        tt = Table((c_obj_name, c_date, c_tint, c_nexp,
                    c_fwhm, c_fwhm_err, c_strehl, c_strehl_err,
                    c_nstar, c_mbase, c_asterr, c_photerr))

        tables[target] = tt

    # Smash all the tables together.
    final_table = vstack([tables['ob120169'], tables['ob140613'], tables['ob150029'], tables['ob150211']])

    print(final_table)

    final_table.write(paper_dir + 'data_table.tex', format='aastex', overwrite=True)

    return

def epoch_figure():
    '''
    Makes a nice illustration of the OGLE and Keck data obtained for all targets.
    '''
    from astropy.time import Time
    import datetime

    # Obtain all dates
    targets = list(epochs.keys())
    ast_dates = np.array([])
    pho_dates = np.array([])
    ast_per_target = np.zeros(len(targets), dtype=int)
    pho_per_target = np.zeros(len(targets), dtype=int)
    for t in range(len(targets)):
        data = munge.getdata(targets[t], time_format='jyear')
        ast_dates = np.append(ast_dates, data['t_ast'].data)
        pho_dates = np.append(pho_dates, data['t_phot'].data)
        ast_per_target[t] = len(data['t_ast'])
        pho_per_target[t] = len(data['t_phot'])

    # Convert to astropy Time objects
    ast_dates = Time(ast_dates, format='jyear', scale='utc').datetime
    pho_dates = Time(pho_dates, format='jyear', scale='utc').datetime
    t_min = np.min([ast_dates.min(), pho_dates.min()])
    t_max = np.max([ast_dates.max(), pho_dates.max()])
    years = np.arange(t_min.year, t_max.year + 1)
    num_t = (years[-1] - years[0])*12 + (t_max.month - t_min.month)
    month_arr = np.arange(0, num_t+1)

    # Make grid
    grid = np.zeros((3*len(targets), num_t+1))

    # Find differences in months
    delta_ast = np.zeros(len(ast_dates), dtype=int)
    for i in range(len(delta_ast)):
        delta_ast[i] = (ast_dates[i].year - years[0])*12 + (ast_dates[i].month - t_min.month)
    delta_pho = np.zeros(len(pho_dates), dtype=int)
    for i in range(len(delta_pho)):
        delta_pho[i] = (pho_dates[i].year - years[0])*12 + (pho_dates[i].month - t_min.month)

    a = 0
    p = 0
    black = 1
    red = 2
    for t in range(len(targets)):
        in_idx = 3*t
        # Identify Keck astrometry
        for d in delta_ast[a:a+ast_per_target[t]]:
            if d in month_arr:
                grid[in_idx:in_idx+2, d] = black
        a += ast_per_target[t]
        # Identify OGLE photometry
        for d in delta_pho[p:p+pho_per_target[t]]:
            if d in month_arr:
                grid[in_idx+1, d] = red
        p += pho_per_target[t]

    cmap = matplotlib.colors.ListedColormap(['#ffffff', 'black', 'red'])
    boundaries = [-0.5, 0.5, 1.5, 2.5]
    norm = matplotlib.colors.BoundaryNorm(boundaries, cmap.N, clip=True)

    x = np.linspace(Time(t_min).jyear, Time(t_max).jyear, num_t+1)
    y = np.linspace(0, len(targets), grid.shape[0])

    fig, ax = plt.subplots(figsize=(8,6))
    cax = ax.pcolormesh(x, y, grid, cmap=cmap, norm=norm)
    plt.yticks(y[1::3], targets)

    #cbar = fig.colorbar(cax, ticks=[1, 2])
    #cbar.set_clim(black, red)
    #cbar.ax.set_yticklabels(['Keck', 'OGLE'])

    plt.show()
    plt.savefig(paper_dir + 'epochs.pdf')

def perr_v_mag(mag, amp, index, mag_const, adderr):
    """
    Model for a positional error vs. magnitude curve. The
    functional form is:

    pos_err = adderr  +  amp * e^(index * (mag - mag_const))
    """
    perr = amp * np.exp(index * (mag - mag_const))
    perr += adderr

    return perr

def fit_perr_v_mag(params, mag, perr_obs):
    amp = params[0]
    index = params[1]
    mag_const = params[2]
    adderr = params[3]

    perr_mod = perr_v_mag(mag, amp, index, mag_const, adderr)

    resid = perr_obs - perr_mod

    return resid

def plot_pos_err():
    """
    Make one plot per target that shows the positional errors vs. magnitude for every epoch.
    """
    targets = list(epochs.keys())

    plt.close('all')

    # NIRC2 plate scale (for 2015 Apr 13 and later)
    scale = 0.009971

    # Make a color scale that is the same for all of them.
    color_norm = Normalize(2015, 2019)
    cmap = cm.get_cmap('rainbow')

    # Loop through targets and make 1 plot each
    for tt in range(len(targets)):
    # for tt in range(1):
        target = targets[tt]
        n_epochs = len(epochs[target])

        plt.figure()

        # Calculate the pos error vs. mag curves for each epoch.
        # for ee in range(4):
        for ee in range(n_epochs):
            epoch = epochs[target][ee]

            pos_file = '/g/lu/data/microlens/{0:s}/combo/starfinder/mag{0:s}_{1:s}_kp_rms_named.lis'.format(epoch, target)

            lis = starlists.StarList.from_lis_file(pos_file)

            # Calc the 1D astrometric error by using the average over X and Y
            perr = 0.5 * (lis['xe'] + lis['ye'])
            kmag = lis['m'] + np.random.rand(len(lis))*1e-5
            # col_idx = np.argmin(np.abs(color_times - lis['t'][0]))
            col = cmap(color_norm(lis['t'][0]))
            # print(ee, col_idx, col, color_times[col_idx], lis['t'][0])

            # REJECTED: Analytic Functional Forms and Fitting
            # Determine relation for perr vs. mag.
            # sdx = kmag.argsort()
            # spl = UnivariateSpline(kmag[sdx], perr[sdx], k=4)

            # # Determine relation for perr vs. mag from our exponential functional form
            # # (see Jia et al. 2019).
            # p0 = [1e-7, 0.7, 5, 1e-3]
            # popt, pcov = curve_fit(perr_v_mag, kmag, perr, p0=p0)
            # print(popt)

            # # Determine relation for perr vs. mag from our exponential functional form
            # # (see Jia et al. 2019). Now with outlier rejection.
            # res_lsq = least_squares(fit_perr_v_mag, p0, args=(kmag, perr))
            # res_lsq2 = least_squares(fit_perr_v_mag, p0, args=(kmag, perr), loss='cauchy', f_scale=0.01)

            # plt.plot(p_mag, spl(p_mag), label='spline')
            # plt.semilogy(p_mag, perr_v_mag(p_mag, *popt), label='curve_fit')
            # plt.semilogy(p_mag, perr_v_mag(p_mag, *res_lsq.x), label='lsq')


            # FIT a functional form similar to Jia et al. 2009 with outlier rejection.
            p0 = [1e-7, 0.7, 5, 1e-3]
            res_lsq2 = least_squares(fit_perr_v_mag, p0, args=(kmag, perr), loss='cauchy', f_scale=0.01)

            epoch_label = '20{0:2s} {1:s}{2:s} {3:2s}'.format(epoch[0:2], epoch[2].upper(), epoch[3:-2], epoch[-2:])

            # Make a magnitude array for the model curves.
            p_mag = np.arange(kmag.min(), kmag.max(), 0.05)

            # Plot the data
            # if ee == 0:
            #     plt.plot(kmag, perr * scale * 1e3, '.', label=epoch_label + ' (obs)', color=col, ms=2)

            plt.semilogy(p_mag, scale * 1e3 * perr_v_mag(p_mag, *res_lsq2.x), label=epoch_label, color=col)

        plt.legend(fontsize=12, loc='upper left', ncol=2)
        plt.xlabel('Kp mag')
        plt.ylabel('$\sigma_{ast}$ (mas)')
        plt.title(target.upper())
        plt.ylim(3e-2, 10)
        plt.savefig(paper_dir + 'pos_err_' + target + '.pdf')

    return

def plot_images():
    img_ob120169 = '/g/lu/data/microlens/16may24/combo/mag16may24_ob120169_kp.fits'
    img_ob140613 = '/g/lu/data/microlens/18aug16/combo/mag18aug16_ob140613_kp.fits'
    img_ob150029 = '/g/lu/data/microlens/17jul19/combo/mag17jul19_ob150029_kp.fits'
    img_ob150211 = '/g/lu/data/microlens/17jun05/combo/mag17jun05_ob150211_kp.fits'

    images = {'ob120169': img_ob120169,
              'ob140613': img_ob140613,
              'ob150029': img_ob150029,
              'ob150211': img_ob150211}

    def plot_image_for_source(target, vmin, vmax):
        combo_dir = os.path.dirname(images[target])
        img_base = os.path.basename(images[target])

        img = fits.getdata(images[target])

        psf_file = '/g/lu/data/microlens/source_list/' + target + '_psf.list'
        psf_tab = Table.read(psf_file, format='ascii', header_start=-1)
        pdx = np.where(psf_tab['PSF?'] == 1)[0]
        psf_tab = psf_tab[pdx]

        lis_file = combo_dir + '/starfinder/' + img_base.replace('.fits', '_rms_named.lis')
        lis_tab = starlists.StarList.from_lis_file(lis_file)

        # Find the target and get its pixel coordinates in this image.
        tdx = np.where(lis_tab['name'] == target)[0]
        coo_targ = np.array([lis_tab['x'][tdx[0]], lis_tab['y'][tdx[0]]])
        coo_targ -= 1   # Shift to a 0-based array system

        # Define the axes
        scale = 0.00996
        x_axis = np.arange(img.shape[0], dtype=float)
        y_axis = np.arange(img.shape[1], dtype=float)
        x_axis = (x_axis - coo_targ[0]) * scale * -1.0
        y_axis = (y_axis - coo_targ[1]) * scale

        norm = LogNorm(vmin, vmax)

        plt.imshow(img, cmap='gist_heat_r', norm=norm, extent=[x_axis[0], x_axis[-1], y_axis[0], y_axis[-1]])
        plt.plot([0], [0], 'c*', ms=15, mec='black', mfc='none', mew=2)
        plt.plot(psf_tab['Xarc'], psf_tab['Yarc'], 'go', ms=15, mec='green', mfc='none', mew=2)

        plt.axis('equal')
        plt.xlabel(r'$\Delta \alpha^*$ (")')
        plt.ylabel(r'$\Delta \delta$ (")')
        plt.title(target.upper())

        date_label = '20{0:2s} {1:3s} {2:2s}'.format(img_base[3:5], img_base[5:8].upper(), img_base[8:10])
        plt.gcf().text(0.2, 0.8, date_label, color='black')

        # plt.xlim(0.5, -0.5)
        # plt.ylim(-0.5, 0.5)

        return

    plt.figure(4)
    plt.clf()
    plot_image_for_source('ob120169', 10, 5e5)
    plt.savefig(paper_dir + 'img_ob120169.pdf')

    plt.figure(1)
    plt.clf()
    plot_image_for_source('ob140613', 10, 5e5)
    plt.savefig(paper_dir + 'img_ob140613.pdf')

    plt.figure(2)
    plt.clf()
    plot_image_for_source('ob150029', 10, 1e5)
    plt.savefig(paper_dir + 'img_ob150029.pdf')

    plt.figure(3)
    plt.clf()
    plot_image_for_source('ob150211', 12, 1e6)
    plt.savefig(paper_dir + 'img_ob150211.pdf')

    return


def plot_ob120169_phot_ast():
    data = munge.getdata('ob120169', use_astrom_phot=True)

    mod_fit = model_fitter.PSPL_multiphot_astrom_parallax2_Solver(data,
                           outputfiles_basename = pspl_ast_multiphot['ob120169'])
    mod_all = mod_fit.get_best_fit_modes_model(def_best = 'maxL')
    img_f = '/g/lu/data/microlens/16may24/combo/mag16may24_ob120169_kp.fits'

    inset_kw = {'labelp1': [-0.8, -0.2], 'labelp2': [0.9, 0.2],
                'scalex': [-8, -6], 'scaley': [-2, -2],
                'textx': -5, 'texty': -1.3}
    plot_4panel(data, mod_all[0], 'ob120169', 1, img_f, inset_kw)

def plot_ob140613_phot_ast():
    data = munge.getdata('ob140613', use_astrom_phot=True)

    mod_fit = model_fitter.PSPL_multiphot_astrom_parallax2_Solver(data,
                           outputfiles_basename = pspl_ast_multiphot['ob140613'])
    mod_all = mod_fit.get_best_fit_modes_model(def_best = 'maxL')
    img_f = '/g/lu/data/microlens/18aug16/combo/mag18aug16_ob140613_kp.fits'

    inset_kw = {'labelp1': [-0.8, -0.2], 'labelp2': [0.9, 0.2],
                'scalex': [-2, 0], 'scaley': [6, 6],
                'textx': 1.4, 'texty': 6.5}
    plot_4panel(data, mod_all[0], 'ob140613', 6, img_f, inset_kw)

def plot_ob150029_phot_ast():
    data = munge.getdata('ob150029', use_astrom_phot=True)

    mod_fit = model_fitter.PSPL_multiphot_astrom_parallax2_Solver(data,
                           outputfiles_basename = pspl_ast_multiphot['ob150029'])
    mod_all = mod_fit.get_best_fit_modes_model(def_best = 'maxL')
    img_f = '/g/lu/data/microlens/17jul19/combo/mag17jul19_ob150029_kp.fits'

    inset_kw = {'labelp1': [-0.8, -0.2], 'labelp2': [0.9, 0.2],
                'scalex': [-5, -3], 'scaley': [-7, -7],
                'textx': 1, 'texty': -10}
    plot_4panel(data, mod_all[0], 'ob150029', 6, img_f, inset_kw)

def plot_ob150211_phot_ast():
    data = munge.getdata('ob150211', use_astrom_phot=True)

    mod_fit = model_fitter.PSPL_multiphot_astrom_parallax2_Solver(data,
                           outputfiles_basename = pspl_ast_multiphot['ob150211'])
    mod_all = mod_fit.get_best_fit_modes_model(def_best = 'maxL')
    img_f = '/g/lu/data/microlens/17jun05/combo/mag17jun05_ob150211_kp.fits'

    inset_kw = {'labelp1': [-0.8, -0.2], 'labelp2': [0.9, 0.2],
                'scalex': [2.0, 4.0], 'scaley': [-2, -2],
                'textx': 4.5, 'texty': -1.5}
    plot_4panel(data, mod_all[0], 'ob150211', 7, img_f, inset_kw)

    
def plot_4panel(data, mod, target, ref_epoch, img_f, inset_kw):
    '''
    Plots a 2x2 figure of 1) the Keck image @ ref_epoch (which must coincide
    with img_f), 2) magnitude vs time, 3) RA vs time, and 4) DEC vs time,
    where the latter three have a residual to the model.
    The astrometry plots include the lens and unlensed models.
    inset_kw is a dictionary for plotting 1) with the following keywords:
    {'labelp1': [x, y] list of first anchor for the label line,
     'labelp2': [x, y] list of the second anchor,
     'scalex': [x1, x2] list for plotting the pixel scale,
     'scaley': [y1, y2] list for plotting the pixel scale
               (y1 = y2 plots a flat line),
     'textx': the x coord of the scale text, and
     'texty': the y coord of the scale text}.
     '''
    from mpl_toolkits.axes_grid1.inset_locator import (inset_axes, InsetPosition,
                                                      mark_inset)
    from mpl_toolkits.axes_grid1.colorbar import colorbar

    # Sample time
    tmax = np.max(np.append(data['t_phot1'], data['t_phot2'])) + 90.0
    t_mod_ast = np.arange(data['t_ast'].min() - 180.0, tmax, 2)
    t_mod_pho = np.arange(data['t_phot1'].min(), tmax, 2)

    # Get the linear motion curves for the source (includes parallax)
    p_unlens_mod = mod.get_astrometry_unlensed(t_mod_ast)
    p_unlens_mod_at_ast = mod.get_astrometry_unlensed(data['t_ast'])

    # Get the lensed motion curves for the source
    p_lens_mod = mod.get_astrometry(t_mod_ast)
    p_lens_mod_at_ast = mod.get_astrometry(data['t_ast'])

    # Get the photometry
    m_lens_mod = mod.get_photometry(t_mod_pho, filt_idx=0)
    m_lens_mod_at_phot1 = mod.get_photometry(data['t_phot1'], filt_idx=0)
    m_lens_mod_at_phot2 = mod.get_photometry(data['t_phot2'], filt_idx=1)

    t_mod_all = np.append(t_mod_ast, t_mod_pho)
    # Set the colorbar
    cmap = plt.cm.viridis
    norm = plt.Normalize(vmin=t_mod_all.min(), vmax=t_mod_all.max())
    smap = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    smap.set_array([])

    # Find the closest model date to the ref_epoch and
    # center inset positions on it
    mod_ref_epoch = np.abs(t_mod_ast - data['t_ast'][ref_epoch]).argmin()
    xpos_ins = (p_lens_mod[:, 0] - p_lens_mod[mod_ref_epoch, 0])*-1e3
    ypos_ins = (p_lens_mod[:, 1] - p_lens_mod[mod_ref_epoch, 1])*1e3

    # Load the Keck image
    combo_dir = os.path.dirname(img_f)
    img_base = os.path.basename(img_f)

    img = fits.getdata(img_f)

    psf_file = '/g/lu/data/microlens/source_list/' + target + '_psf.list'
    psf_tab = Table.read(psf_file, format='ascii', header_start=-1)
    pdx = np.where(psf_tab['PSF?'] == 1)[0]
    psf_tab = psf_tab[pdx]

    lis_file = combo_dir + '/starfinder/' + img_base.replace('.fits', '_rms_named.lis')
    lis_tab = starlists.StarList.from_lis_file(lis_file)

    # Find the target and get its pixel coordinates in this image.
    tdx = np.where(lis_tab['name'] == target)[0]
    coo_targ = np.array([lis_tab['x'][tdx[0]], lis_tab['y'][tdx[0]]])
    coo_targ -= 1   # Shift to a 0-based array system

    # Define the axes
    scale = 0.00996
    x_axis = np.arange(img.shape[0], dtype=float)
    y_axis = np.arange(img.shape[1], dtype=float)
    x_axis = (x_axis - coo_targ[0]) * scale * -1.0
    y_axis = (y_axis - coo_targ[1]) * scale

    # Set figure
    plt.close(1)
    fig = plt.figure(1, figsize = (12,10))
    wpad = 0.15
    hpad = 0.1
    ax_width = 0.37*10/12
    ax_height = 0.37

    # TARGET IMAGE
    ax1 = fig.add_axes([wpad, 1.0 - hpad/2 - ax_height, ax_width, ax_height])
    ax1.imshow(img, cmap='gist_heat_r', norm=LogNorm(12, 1e6),
               extent=[x_axis[0], x_axis[-1], y_axis[0], y_axis[-1]])
    ax1.set_xlim(-4,2)
    ax1.set_ylim(-4,2)
    ax1.set_xlabel(r'$\Delta \alpha^*$ (")')
    ax1.set_ylabel(r'$\Delta \delta$ (")')
    ax1.invert_xaxis()

    # Label the target
    line_color = "#7f0f18"
    ax1.plot(inset_kw['labelp1'], inset_kw['labelp2'],
             linestyle='-', color=line_color)
    ax1.text(inset_kw['labelp1'][0], inset_kw['labelp2'][0], target.upper(),
             fontsize=16, color=line_color)

    # Fake inset axes to control the inset marking, hide its ticks
    axf = inset_axes(ax1, 1, 1)
    axf.plot(xpos_ins/1e3, ypos_ins/1e3)
    axf.set_xticks([])
    axf.set_yticks([])
    axf.set_aspect('equal')

    # Plot the motion on the sky
    axins = inset_axes(ax1, 1.05, 1)

    axins.scatter(xpos_ins, ypos_ins, c=t_mod_ast, cmap=cmap, norm=norm, s=1)
    axins.errorbar((data['xpos'] - data['xpos'][ref_epoch])*-1e3, (data['ypos'] - data['ypos'][ref_epoch])*1e3,
                   xerr = data['xpos_err']*1e3, yerr=data['ypos_err']*1e3, fmt='.k')

    axins.set_xticks([],[])
    axins.set_yticks([],[])
    axins.invert_xaxis()

    # Enlarge the lims to create space for the points
    axins.set_ylim(axins.get_ylim()[0]-2.0, axins.get_ylim()[1]+1.0)
    axins.set_aspect('equal')

    # Plot the scale in the inset
    axins.plot(inset_kw['scalex'], inset_kw['scaley'], color=line_color)
    axins.text(inset_kw['textx'], inset_kw['texty'], '2 mas', color=line_color, fontsize=12)

    # Tweak the limits of the fake axes to fit the inset markers
    axf.set_ylim((axins.get_ylim()[0])/1e3, (axins.get_ylim()[1])/1e3)
    # Manually set the position and relative size of the inset axes within ax1
    ip = InsetPosition(ax1, [0.40,0,0.55,0.45])
    axf.set_axes_locator(ip)
    axins.set_axes_locator(ip)
    # Mark the region corresponding to the inset axes on the parenta axes
    # and draw lines in grey linking the two axes.
    mark_inset(parent_axes=ax1, inset_axes=axf, loc1=1, loc2=3, fc="none", ec='0.45')

    # MAGNITUDE VS TIME
    ax10 = fig.add_axes([1.0 - wpad/2 - ax_width, 1.0 - hpad/2 - 0.75*ax_height, ax_width, 0.75*ax_height])
    ax11 = fig.add_axes([1.0 - wpad/2 - ax_width, 1.0 - hpad/2 - ax_height, ax_width, 0.25*ax_height])
    ax10.errorbar(data['t_phot1'], data['mag1'], yerr=data['mag_err1'],
                  fmt='k.', alpha=0.05)
    ax10.scatter(t_mod_pho, m_lens_mod, c = t_mod_pho, cmap = cmap, norm = norm, s = 1)
    ax10.invert_yaxis()
    ax10.set_ylabel('Magnitude')
    ax10.set_aspect('auto', adjustable='box')
    ax10.set_xticks([])
    ax11.errorbar(data['t_phot1'], data['mag1'] - m_lens_mod_at_phot1, yerr=data['mag_err1'],
                  fmt='k.', alpha=0.05)
    ax11.scatter(t_mod_pho, m_lens_mod - m_lens_mod, c = t_mod_pho, cmap = cmap, norm = norm, s = 1)
    ax11.yaxis.set_major_locator(plt.MaxNLocator(2))
    ax11.set_xlabel('Time (MJD)')
    ax11.set_ylabel('res')

    # Center the position data and model off the reference epoch
    p_lens_mod -= [data['xpos'][ref_epoch], data['ypos'][ref_epoch]]
    p_unlens_mod -= [data['xpos'][ref_epoch], data['ypos'][ref_epoch]]
    p_unlens_mod_at_ast -= [data['xpos'][ref_epoch], data['ypos'][ref_epoch]]
    data['xpos'] -= data['xpos'][ref_epoch]
    data['ypos'] -= data['ypos'][ref_epoch]

    # RA VS TIME
    ax20 = fig.add_axes([wpad, hpad + 0.25*ax_height, ax_width, 0.75*ax_height])
    ax21 = fig.add_axes([wpad, hpad, ax_width, 0.25*ax_height])
    ax20.errorbar(data['t_ast'], data['xpos']*-1e3,
                  yerr=data['xpos_err']*1e3, fmt='k.', zorder = 1000)
    ax20.scatter(t_mod_ast, p_lens_mod[:, 0]*-1e3, c = t_mod_ast, cmap = cmap, norm = norm, s = 1)
    ax20.plot(t_mod_ast, p_unlens_mod[:, 0]*-1e3, 'r--')
    ax20.get_xaxis().set_visible(False)
    ax20.set_ylabel(r'$\Delta \alpha^*$ (mas)')
    ax20.get_shared_x_axes().join(ax20, ax21)
    ax21.errorbar(data['t_ast'], (data['xpos'] - p_unlens_mod_at_ast[:,0]) * -1e3,
                  yerr=data['xpos_err'] * 1e3, fmt='k.', alpha=1, zorder = 1000)
    ax21.scatter(t_mod_ast, (p_lens_mod[:, 0] - p_unlens_mod[:, 0])*-1e3, c = t_mod_ast, cmap = cmap, norm = norm, s = 1)
    ax21.axhline(0, linestyle='--', color='r')
    ax21.set_xlabel('Time (MJD)')
    ax21.set_ylabel('res')

    # DEC VS TIME
    ax30 = fig.add_axes([1.0 - wpad/2 - ax_width, hpad + 0.25*ax_height, ax_width, 0.75*ax_height])
    ax31 = fig.add_axes([1.0 - wpad/2 - ax_width, hpad, ax_width, 0.25*ax_height])
    ax30.errorbar(data['t_ast'], data['ypos']*1e3,
                  yerr=data['ypos_err']*1e3, fmt='k.', zorder = 1000)
    ax30.scatter(t_mod_ast, p_lens_mod[:, 1]*1e3, c = t_mod_ast, cmap = cmap, norm = norm, s = 1)
    ax30.plot(t_mod_ast, p_unlens_mod[:, 1]*1e3, 'r--')
    ax30.get_xaxis().set_visible(False)
    ax30.set_ylabel(r'$\Delta \delta$ (mas)')
    ax30.get_shared_x_axes().join(ax30, ax31)
    ax31.errorbar(data['t_ast'], (data['ypos'] - p_unlens_mod_at_ast[:, 1]) * 1e3,
                  yerr=data['ypos_err'] * 1e3, fmt='k.', alpha=1, zorder = 1000)
    ax31.scatter(t_mod_ast, (p_lens_mod[:, 1] - p_unlens_mod[:, 1])*1e3, c = t_mod_ast, cmap = cmap, norm = norm, s = 1)
    ax31.axhline(0, linestyle='--', color='r')
    ax31.set_xlabel('Time (MJD)')
    ax31.set_ylabel('res')

    plt.savefig(paper_dir + target + '_phot_astrom.png')


def plot_comparison_stars_all():
    plot_comparison_stars('ob120169', res_rng=1.1)
    plot_comparison_stars('ob140613', res_rng=0.4)
    plot_comparison_stars('ob150029', res_rng=1.1)
    plot_comparison_stars('ob150211', res_rng=1.3)
    return

def plot_comparison_stars(target, res_rng=0.8):
    """
    target : str
        Target name (lowercase)

    res_rng : float
        +/- range of residuals in milli-arcseconds.
    """
    ast_data_file = astrom_data[target]

    data = Table.read(ast_data_file)

    # Flip the coordinates to what we see on sky (+x increase to the East)
    data['x'] *= -1.0
    data['x0'] *= -1.0
    data['vx'] *= -1.0

    # Make a list of the source and the two nearby comparison stars (3 all together)
    targets = np.append([target], comp_stars[target])

    # Figure out the min/max of the times for these sources.
    tdx = np.where(data['name'] == target)[0][0]
    tmin = data['t'][tdx].min() - 0.5   # in days
    tmax = data['t'][tdx].max() + 0.5   # in days

    # Setup figure and color scales
    fig = plt.figure(1, figsize=(13, 7.5))
    plt.clf()
    grid_t = plt.GridSpec(1, 3, hspace=5.0, wspace=0.5, bottom=0.60, top=0.95, left=0.12, right=0.86)
    grid_b = plt.GridSpec(2, 3, hspace=0.1, wspace=0.5, bottom=0.10, top=0.45, left=0.12, right=0.86)

    cmap = plt.cm.plasma
    norm = plt.Normalize(vmin=tmin, vmax=tmax)
    smap = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    smap.set_array([])

    def plot_each_star(star_num, star_name):
        # Make two boxes for each star
        ax_sky = fig.add_subplot(grid_t[0, star_num])
        ax_resX = fig.add_subplot(grid_b[1, star_num])
        ax_resY = fig.add_subplot(grid_b[0, star_num])

        # Fetch the data
        tdx = np.where(data['name'] == star_name)[0][0]
        star = data[tdx]

        # Make the model curves
        tmod = np.arange(tmin, tmax, 0.1)
        xmod = star['x0'] + star['vx'] * (tmod - star['t0'])
        ymod = star['y0'] + star['vy'] * (tmod - star['t0'])
        xmode = np.hypot(star['x0e'], star['vxe'] * (tmod - star['t0']))
        ymode = np.hypot(star['y0e'], star['vye'] * (tmod - star['t0']))

        xmod_at_t = star['x0'] + star['vx'] * (star['t'] - star['t0'])
        ymod_at_t = star['y0'] + star['vy'] * (star['t'] - star['t0'])

        # Plot Positions on Sky
        ax_sky.plot(xmod, ymod, 'k-', color='grey', zorder=1)
        ax_sky.plot(xmod + xmode, ymod + ymode, 'k--', color='grey', zorder=1)
        ax_sky.plot(xmod - xmode, ymod - ymode, 'k--', color='grey', zorder=1)
        sc = ax_sky.scatter(star['x'], star['y'], c=star['t'], cmap=cmap, norm=norm, s=20, zorder=2)
        ax_sky.errorbar(star['x'], star['y'], xerr=star['xe'], yerr=star['ye'],
                            ecolor=smap.to_rgba(star['t']), fmt='none', elinewidth=2, zorder=2)
        ax_sky.set_aspect('equal', adjustable='datalim')

        # Figure out which axis has the bigger data range.
        xrng = np.abs(star['x'].max() - star['x'].min())
        yrng = np.abs(star['y'].max() - star['y'].min())
        if xrng > yrng:
            ax_sky.set_xlim(star['x'].min() - 0.001, star['x'].max() + 0.001)
        else:
            ax_sky.set_ylim(star['y'].min() - 0.001, star['y'].max() + 0.001)

        # Set labels
        ax_sky.invert_xaxis()
        ax_sky.set_title(star_name.upper())
        ax_sky.set_xlabel(r'$\Delta\alpha*$ (")')
        if star_num == 0:
            ax_sky.set_ylabel(r'$\Delta\delta$ (")')


        # Plot Residuals vs. Time
        xres = (star['x'] - xmod_at_t) * 1e3
        yres = (star['y'] - ymod_at_t) * 1e3
        xrese = star['xe'] * 1e3
        yrese = star['ye'] * 1e3
        ax_resX.errorbar(star['t'], xres, yerr=xrese, fmt='r.', label=r'$\alpha*$', elinewidth=2)
        ax_resY.errorbar(star['t'], yres, yerr=yrese, fmt='b.', label=r'$\delta$', elinewidth=2)
        ax_resX.plot(tmod, xmod - xmod, 'r-')
        ax_resX.plot(tmod, xmode*1e3, 'r--')
        ax_resX.plot(tmod, -xmode*1e3, 'r--')
        ax_resY.plot(tmod, ymod - ymod, 'b-')
        ax_resY.plot(tmod, ymode*1e3, 'b--')
        ax_resY.plot(tmod, -ymode*1e3, 'b--')
        ax_resX.set_xlabel('Date (yr)')
        ax_resX.set_ylim(-res_rng, res_rng)
        ax_resY.set_ylim(-res_rng, res_rng)
        ax_resY.get_xaxis().set_visible(False)
        if star_num == 0:
            ax_resX.set_ylabel(r'$\alpha^*$')
            ax_resY.set_ylabel(r'$\delta$')
            plt.gcf().text(0.015, 0.3, 'Residuals (mas)', rotation=90, fontsize=24,
                               ha='center', va='center')
        # if star_num == 2:
        #     ax_res.legend(loc='right', bbox_to_anchor= (1.5, 0.5),
        #                       borderaxespad=0, frameon=True, numpoints=1,
        #                       handletextpad=0.1)
            # leg_ax = fig.add_axes([0.88, 0.12, 0.1, 0.1])
            # leg_ax.text(


        return sc



    sc = plot_each_star(0, targets[0])
    sc = plot_each_star(1, targets[1])
    sc = plot_each_star(2, targets[2])
    cb_ax = fig.add_axes([0.88, 0.60, 0.02, 0.35])
    plt.colorbar(sc, cax=cb_ax, label='Year')

    plt.savefig(paper_dir + 'comparison_star_' + target + '.png')

    return


def weighted_avg_and_std(values, weights):
    """
    Return the weighted average and standard deviation.

    values, weights -- Numpy ndarrays with the same shape.
    """
    average = np.average(values, weights=weights)
    # Fast and numerically precise:
    variance = np.average((values-average)**2, weights=weights)

    return (average, np.sqrt(variance))


def make_tab_9_fit():
    """
    Make a table with phot only (Keck + OGLE) fit
    9_fit_multiphot_only_parallax
    """
    # In the order we want them in the table. Use an empty param to indicate
    # break between fit parameters and unfit parameters.
    params_list = ['t0', 'u0_amp', 'tE', 'piE_E', 'piE_N',
                   'b_sff1', 'mag1', 'b_sff2', 'mag2', '',
                   'log_piE']

    params = {'t0' : [r'$t_0$ (MJD)', '${0:.2f}$'],
              'u0_amp' : [r'$u_0$ ', '${0:.3f}$'],
              'tE' : [r'$t_E$ (days)', '${0:.1f}$'],
              'piE_E' : [r'$\pi_{E,E}$ ', '${0:.3f}$'],
              'piE_N' : [r'$\pi_{E,N}$ ', '${0:.3f}$'],
              'piE' : [r'$\pi_E$', '${0:.3f}$'],
              'b_sff1' : [r'$b_{SFF}$', '${0:.3f}$'],
              'mag1' : [r'$I_{OGLE}$ (mag)', '${0:.4f}$'],
              'b_sff2' : [r'$b_{SFF}$', '${0:.3f}$'],
              'mag2' : [r'$I_{OGLE}$ (mag)', '${0:.4f}$'],
              'log_piE' : [r'$\mathrm{log}\pi_E$', '${0:.3f}$']}

    # We need a string description of each parameter.
    desc = {'t0'      : r'Time of closest approach.',
            'u0_amp'  : r'Closest approach in $\theta_E$ units.',
            'tE'      : r'Einstein crossing time.',
            'piE_E'   : r'Microlensing parallax in the $\alpha^*$ direction.',
            'piE_N'   : r'Microlensing parallax in the $\delta$ direction.',
            'piE'     : r'Microlensing parallax.',
            'b_sff1'  : r'The source flux fraction in the OGLE aperture, unlensed.',
            'mag1'    : r'OGLE I-band magnitude of the unlensed source.',
            'b_sff2'  : r'The source flux fraction in the Keck aperture, unlensed.',
            'mag2'    : r'Keck Kp-band magnitude of the unlensed source.',
            'log_piE' : r'Log (base 10) of $\pi_E$.'
            }

    # Get the data
    targets = list(epochs.keys())
    for target in targets:
        tab = Table.read(pspl_multiphot[target] + 'summary.fits')
        output = open(paper_dir + target + '_tab_9_fit.txt', 'w')
        ##########################################
        # This stuff needs to be done by hand.
        # If one solution, use global.
        # If degenerate solutions, give them both.
        ##########################################
        if ((target == 'ob120169') | (target == 'ob140613') | (target == 'ob150029')):
            output.write('Fit & {} & {} \\\\\n')
            output.write('\hline \n')
            for pp in params_list:
                # Check if we should switch to derived parameters when we
                # encounter a '' in the parameters list.
                if pp == '':
                    output.write('\hline\n')
                    output.write('Derived & {} & {} \\\\\n')
                    output.write('\hline\n')
                    continue

                p = params[pp]

                sol1 = p[1].format(tab['MAP_' + pp][0])

                output.write(p[0] + ' & ' + sol1 + ' & ' + desc[pp] + ' \\\\\n')

            output.write('log$\mathcal{Z}$' + ' & ' +
                         '{:.2f}'.format(tab['logZ'][0]) + ' & ' +
                         'Local Evidence')

            output.close()

        if target == 'ob150211':
            output.write('Fit & {}  & {} & {} \\\\\n')
            output.write('\hline \n')
            for pp in params_list:
                # Check if we should switch to derived parameters when we
                # encounter a '' in the parameters list.
                if pp == '':
                    output.write('\hline\n')
                    output.write('Derived & {} & {} & {} \\\\\n')
                    output.write('\hline\n')
                    continue

                p = params[pp]

                sol1 = p[1].format(tab['MAP_' + pp][1])
                sol2 = p[1].format(tab['MAP_' + pp][2])

                output.write(p[0] + ' & ' + sol1 + ' & ' + sol2 + ' & ' + desc[pp] + ' \\\\\n')

            output.write('log$\mathcal{Z}$' + ' & ' +
                         '{:.2f}'.format(tab['logZ'][1]) + ' & ' +
                         '{:.2f}'.format(tab['logZ'][2]) + ' & ' +
                         'Local Evidence')

            output.close()


def make_tab_8_fit():
    """
    Make a table with the Kains parametrization
    8_fit_multiphot_astrom_parallax2
    """
    # In the order we want them in the table. Use an empty param to indicate
    # break between fit parameters and unfit parameters.
    params_list = ['t0', 'u0_amp', 'tE', 'piE_E', 'piE_N',
                   'b_sff1', 'mag_src1', 'b_sff2', 'mag_src2', 'log_thetaE',
                   'xS0_E', 'xS0_N', 'muS_E', 'muS_N', 'piS', '',
                   'log_mL', 'muL_E', 'muL_N', 'muRel_E',
                   'muRel_N', 'piL', 'piRel', 'log_piE']

    params = {'t0' : [r'$t_0$ (MJD)', '${0:.2f}$'],
              'u0_amp' : [r'$u_0$ ', '${0:.3f}$'],
              'tE' : [r'$t_E$ (days)', '${0:.1f}$'],
              'piE_E' : [r'$\pi_{E,E}$ ', '${0:.3f}$'],
              'piE_N' : [r'$\pi_{E,N}$ ', '${0:.3f}$'],
              'log_piE' : [r'$\mathrm{log}\pi_E$', '${0:.3f}$'],
              'b_sff1' : [r'$b_{SFF}$', '${0:.3f}$'],
              'mag_src1' : [r'$I_{OGLE}$ (mag)', '${0:.4f}$'],
              'b_sff2' : [r'$b_{SFF}$', '${0:.3f}$'],
              'mag_src2' : [r'$I_{OGLE}$ (mag)', '${0:.4f}$'],
              'log_mL' : [r'$\mathrm{log}M_L (M_\odot$)', '${0:.1f}$'],
              'xS0_E' : [r"$x_{S,0,E}$ ($''$)", '${0:.4f}$'],
              'xS0_N' : [r"$x_{S,0,N}$ ($''$)", '${0:.4f}$'],
              'beta' : [r'$\beta$ (mas)', '${0:.2f}$'],
              'muL_E' : [r'$\mu_{L,E}$ (mas/yr)', '${0:.2f}$'],
              'muL_N' : [r'$\mu_{L,N}$ (mas/yr)', '${0:.2f}$'],
              'muS_E' : [r'$\mu_{S,E}$ (mas/yr)', '${0:.2f}$'],
              'muS_N' : [r'$\mu_{S,N}$ (mas/yr)', '${0:.2f}$'],
              'dL' : [r'$d_L$ (pc)', '${0:.0f}$'],
              'dS' : [r'$d_S$ (pc)', '${0:.0f}$'],
              'piL' : [r'$\pi_L$ (mas)', '${0:.4f}$'],
              'piS' : [r'$\pi_S$ (mas)', '${0:.4f}$'],
              'piRel' : [r'$\pi_{rel}$ (mas)', '${0:.4f}$'],
              'dL_dS' : [r'$d_L/d_S$', '${0:.3f}$'],
              'log_thetaE' : [r'$\mathrm{log}\theta_E$ (mas)', '${0:.2f}$'],
              'muRel_E' : [r'$\mu_{rel,E}$ (mas/yr)', '${0:.2f}$'],
              'muRel_N' : [r'$\mu_{rel,N}$ (mas/yr)', '${0:.2f}$']}

    # We need a string description of each parameter.
    desc = {'t0'      : r'Time of closest approach.',
            'u0_amp'  : r'Closest approach in $\theta_E$ units.',
            'tE'      : r'Einstein crossing time.',
            'piE_E'   : r'Microlensing parallax in the $\alpha^*$ direction.',
            'piE_N'   : r'Microlensing parallax in the $\delta$ direction.',
            'log_piE'     : r'Log (base 10) of the microlensing parallax.',
            'b_sff1'  : r'The source flux fraction in the OGLE aperture, unlensed.',
            'mag_src1'    : r'OGLE I-band magnitude of the unlensed source.',
            'b_sff2'  : r'The source flux fraction in the Keck aperture, unlensed.',
            'mag_src2'    : r'Keck Kp-band magnitude of the unlensed source.',
            'log_mL'      : r'Log (base 10) of the mass of the lens.',
            'xS0_E'   : r'Relative $\alpha^*$ source position at $t_0$.',
            'xS0_N'   : r'Relative $\delta$ source positions at $t_0$.',
            'beta'    : r'Closest angular approach distance.',
            'muL_E'   : r'Proper motion of the lens in the $\alpha^*$ direction',
            'muL_N'   : r'Proper motion of the lens in the $\delta$ direction',
            'muS_E'   : r'Proper motion of the source in the $\alpha^*$ direction',
            'muS_N'   : r'Proper motion of the source in the $\delta$ direction',
            'dL'      : r'Distance to the lens.',
            'dS'      : r'Distance to the source.',
            'piL'     : r'Lens parallax.',
            'piS'     : r'Source parallax.',
            'piRel'   : r'Relative parallax.',
            'dL_dS'   : r'Distance ratio of lens to source.',
            'log_thetaE'  : r'Log (base 10) of the angular Einstein radius',
            'muRel_E' : r'Relative source-lens proper motion in the $\alpha^*$ direction.',
            'muRel_N' : r'Relative source-lens proper motion in the $\delta$ direction.'
            }

    # Get the data
    targets = list(epochs.keys())
    for target in targets:
        tab = Table.read(pspl_ast_multiphot[target] + 'summary.fits')
        output = open(paper_dir + target + '_tab_8_fit.txt', 'w')
        ##########################################
        # This stuff needs to be done by hand.
        # If one solution, use global.
        # If degenerate solutions, give them both.
        ##########################################
        if ((target == 'ob120169') | (target == 'ob140613') | (target == 'ob150029')):
            output.write('Fit & {} & {} \\\\\n')
            output.write('\hline \n')
            for pp in params_list:
                # Check if we should switch to derived parameters when we
                # encounter a '' in the parameters list.
                if pp == '':
                    output.write('\hline\n')
                    output.write('Derived & {} & {} \\\\\n')
                    output.write('\hline\n')
                    continue

                p = params[pp]

                sol1 = p[1].format(tab['MAP_' + pp][0])

                output.write(p[0] + ' & ' + sol1 + ' & ' + desc[pp] + ' \\\\\n')

            output.write('log$\mathcal{Z}$' + ' & ' +
                         '{:.2f}'.format(tab['logZ'][0]) + ' & ' +
                         'Local Evidence')

            output.close()

        if target == 'ob150211':
            output.write('Fit & {}  & {} & {} \\\\\n')
            output.write('\hline \n')
            for pp in params_list:
                # Check if we should switch to derived parameters when we
                # encounter a '' in the parameters list.
                if pp == '':
                    output.write('\hline\n')
                    output.write('Derived & {} & {} & {} \\\\\n')
                    output.write('\hline\n')
                    continue

                p = params[pp]

                sol1 = p[1].format(tab['MAP_' + pp][1])
                sol2 = p[1].format(tab['MAP_' + pp][2])

                output.write(p[0] + ' & ' + sol1 + ' & ' + sol2 + ' & ' + desc[pp] + ' \\\\\n')

            output.write('log$\mathcal{Z}$' + ' & ' +
                         '{:.2f}'.format(tab['logZ'][1]) + ' & ' +
                         '{:.2f}'.format(tab['logZ'][2]) + ' & ' +
                         'Local Evidence')

            output.close()

def make_tab_3_fit():
    """
    Make a table with OGLE phot only fit
    3_fit_phot_parallax
    """
    # In the order we want them in the table. Use an empty param to indicate
    # break between fit parameters and unfit parameters.
    params_list = ['t0', 'u0_amp', 'tE', 'piE_E', 'piE_N',
                   'b_sff', 'mag_src']

    params = {'t0' : [r'$t_0$ (MJD)', '${0:.2f}$'],
              'u0_amp' : [r'$u_0$ ', '${0:.3f}$'],
              'tE' : [r'$t_E$ (days)', '${0:.1f}$'],
              'piE_E' : [r'$\pi_{E,E}$ ', '${0:.3f}$'],
              'piE_N' : [r'$\pi_{E,N}$ ', '${0:.3f}$'],
              'piE' : [r'$\pi_E$', '${0:.3f}$'],
              'b_sff' : [r'$b_{SFF}$', '${0:.3f}$'],
              'mag_src' : [r'$I_{OGLE}$ (mag)', '${0:.4f}$'],
              'piE' : [r'$\mathrm{log}\pi_E$', '${0:.3f}$']}

    # We need a string description of each parameter.
    desc = {'t0'      : r'Time of closest approach.',
            'u0_amp'  : r'Closest approach in $\theta_E$ units.',
            'tE'      : r'Einstein crossing time.',
            'piE_E'   : r'Microlensing parallax in the $\alpha^*$ direction.',
            'piE_N'   : r'Microlensing parallax in the $\delta$ direction.',
            'piE'     : r'Microlensing parallax.',
            'b_sff'  : r'The source flux fraction in the OGLE aperture, unlensed.',
            'mag_src'    : r'OGLE I-band magnitude of the unlensed source.',
            'log_piE' : r'Microlensing parallax.'
            }

    # Get the data
    targets = list(epochs.keys())
    for target in targets:
        tab = Table.read(pspl_phot[target] + 'summary.fits')
        output = open(paper_dir + target + '_tab_3_fit.txt', 'w')
        ##########################################
        # This stuff needs to be done by hand.
        # If one solution, use global.
        # If degenerate solutions, give them both.
        ##########################################
        if ((target == 'ob140613') | (target == 'ob150029')):
            output.write('Fit & {} & {} \\\\\n')
            output.write('\hline \n')
            for pp in params_list:
                # Check if we should switch to derived parameters when we
                # encounter a '' in the parameters list.
                if pp == '':
                    output.write('\hline\n')
                    output.write('Derived & {} & {} \\\\\n')
                    output.write('\hline\n')
                    continue

                p = params[pp]

                sol1 = p[1].format(tab['MAP_' + pp][0])

                output.write(p[0] + ' & ' + sol1 + ' & ' + desc[pp] + ' \\\\\n')

            output.write('log$\mathcal{Z}$' + ' & ' +
                         '{:.2f}'.format(tab['logZ'][0]) + ' & ' +
                         'Local Evidence')

            output.close()

        if ((target == 'ob120169') | (target == 'ob150211')):
            output.write('Fit & {}  & {} & {} \\\\\n')
            output.write('\hline \n')
            for pp in params_list:
                # Check if we should switch to derived parameters when we
                # encounter a '' in the parameters list.
                if pp == '':
                    output.write('\hline\n')
                    output.write('Derived & {} & {} & {} \\\\\n')
                    output.write('\hline\n')
                    continue

                p = params[pp]

                sol1 = p[1].format(tab['MAP_' + pp][1])
                sol2 = p[1].format(tab['MAP_' + pp][2])

                output.write(p[0] + ' & ' + sol1 + ' & ' + sol2 + ' & ' + desc[pp] + ' \\\\\n')

            output.write('log$\mathcal{Z}$' + ' & ' +
                         '{:.2f}'.format(tab['logZ'][1]) + ' & ' +
                         '{:.2f}'.format(tab['logZ'][2]) + ' & ' +
                         'Local Evidence')

            output.close()


def tE_piE():
    t = Table.read('/u/casey/scratch/papers/microlens_2019/plot_files/Mock_EWS.fits')

    mas_to_rad = 4.848 * 10**-9

    bh_idx = np.where(t['rem_id_L'] == 103)[0]
    ns_idx = np.where(t['rem_id_L'] == 102)[0]
    wd_idx = np.where((t['rem_id_L'] == 101) |
                      (t['rem_id_L'] == 6))[0]
    st_idx = np.where(t['rem_id_L'] == 0)[0]

    # start with a rectangular Figure
    plt.figure(15, figsize=(8, 8))
    plt.clf()

    minpiE = 0.003
    maxpiE = 5
    mintE = 0.8
    maxtE = 500

    # the scatter plot:
    plt.scatter(t['t_E'][st_idx], t['pi_E'][st_idx]/mas_to_rad,
                      alpha = 0.3, label = 'Star', marker = 's',
                      c = 'gold')
    plt.scatter(t['t_E'][wd_idx], t['pi_E'][wd_idx]/mas_to_rad,
                      alpha = 0.3, label = 'WD', marker = 'P',
                      c = 'coral')
    plt.scatter(t['t_E'][ns_idx], t['pi_E'][ns_idx]/mas_to_rad,
                      alpha = 0.3, label = 'NS', marker = 'v',
                      c = 'mediumblue')
    plt.scatter(t['t_E'][bh_idx], t['pi_E'][bh_idx]/mas_to_rad,
                      alpha = 0.3, label = 'BH',
                      c = 'k')

    #############
    # Add the observations.
    # OB110022 from Lu+16.
    # Others are phot parallax solutions (global)
    #############
    # OB110022
    piEE_110022 = -0.393
    piEN_110022 = -0.071
    piE_110022 = np.hypot(piEE_110022, piEN_110022)
    tE_110022 = 61.4

    # OB120169
    piEE_120169 = 0.0130
    piEN_120169 = -0.1367
    piE_120169 = np.hypot(piEE_120169, piEN_120169)
    tE_120169 = 163.26

    # OB140613
    piEE_140613 = -0.1128
    piEN_140613 = 0.0752
    piE_140613 = np.hypot(piEE_140613, piEN_140613)
    tE_140613 = 320.97

    # OB150029
    piEE_150029 = 0.0504
    piEN_150029 = 0.1669
    piE_150029 = np.hypot(piEE_150029, piEN_150029)
    tE_150029 = 154.38

    # OB150211
    piEE_150211 = 0.0463
    piEN_150211 = -0.0294
    piE_150211 = np.hypot(piEE_150211, piEN_150211)
    tE_150211 = 111.78

    plt.scatter(tE_110022, piE_110022, label = 'OB110022',
                marker = '*', s = 400)
    plt.scatter(tE_120169, piE_120169, label = 'OB120169',
                marker = '*', s = 400)
    plt.scatter(tE_140613, piE_140613, label = 'OB140613',
                marker = '*', s = 400)
    plt.scatter(tE_150029, piE_150029, label = 'OB150029',
                marker = '*', s = 400)
    plt.scatter(tE_150211, piE_150211, label = 'OB150211',
                marker = '*', s = 400)

    plt.xlabel('$t_E$ (days)')
    plt.ylabel('$\pi_E$')
    plt.xscale('log')
    plt.yscale('log')
    plt.legend(loc=2)
    plt.xlim(mintE, maxtE)
    plt.ylim(minpiE, maxpiE)
    tEbins = np.logspace(-1, 2.5, 26)
    piEbins = np.logspace(-4, 1, 26)

    plt.show()
    plt.savefig(paper_dir + 'tE_piE.pdf')

    return

def shift_vs_piE():
    # This assumes blending from source and lens.
    t = Table.read('/u/casey/scratch/papers/microlens_2019/plot_files/Mock_EWS.fits')

    u0_arr = t['u0']
    thetaE_arr = t['theta_E']

    # Stores the maximum astrometric shift
    final_delta_arr = np.zeros(len(u0_arr))

    # Stores the lens-source separation corresponding
    # to the maximum astrometric shift
    final_u_arr = np.zeros(len(u0_arr))

    # Sort by whether the maximum astrometric shift happens
    # before or after the maximum photometric amplification
    big_idx = np.where(u0_arr > np.sqrt(2))[0]
    small_idx = np.where(u0_arr <= np.sqrt(2))[0]

    # Flux ratio of lens to source (and make it 0 if dark lens)
    g_arr = 10**(-0.4 * (t['ubv_i_app_L'] - t['ubv_i_app_S']))
    dark_L_idx = np.where(t['ubv_i_app_L'] == -99)
    g_arr[dark_L_idx] = 0

    # Uncomment next line if want to do this without blending
    # g_arr = np.zeros(len(t['ubv_i_app_L']))

    for i in np.arange(len(u0_arr)):
        g = g_arr[i]
        thetaE = thetaE_arr[i]
        # Try all values between u0 and sqrt(2) to find max
        # astrometric shift
        if u0_arr[i] < np.sqrt(2):
            u_arr = np.linspace(u0_arr[i], np.sqrt(2), 100)
            delta_arr = np.zeros(len(u_arr))
            for j in np.arange(len(u_arr)):
                u = u_arr[j]
                numer = 1 + g * (u**2 - u * np.sqrt(u**2 + 4) + 3)
                denom = u**2 + 2 + g * u * np.sqrt(u**2 + 4)
                delta = (u * thetaE/(1 + g)) * (numer/denom)
                delta_arr[j] = delta
            max_idx = np.argmax(delta_arr)
            final_delta_arr[i] = delta_arr[max_idx]
            final_u_arr[i] = u_arr[max_idx]
        # Maximum astrometric shift will occur at sqrt(2)
        if u0_arr[i] > np.sqrt(2):
            u = u0_arr[i]
            numer = 1 + g * (u**2 - u * np.sqrt(u**2 + 4) + 3)
            denom = u**2 + 2 + g * u * np.sqrt(u**2 + 4)
            delta = (u * thetaE/(1 + g)) * (numer/denom)
            final_delta_arr[i] = delta
            final_u_arr[i] = u

    mas_to_rad = 4.848 * 10**-9

    bh_idx = np.where(t['rem_id_L'] == 103)[0]
    other_idx = np.where(t['rem_id_L'] != 103)[0]

    plt.figure(1)
    plt.clf()
    fig, ax = plt.subplots()
    norm = matplotlib.colors.Normalize(np.log10(np.min(t['t_E'])), np.log10(np.max(t['t_E'])))
    plt.subplots_adjust(bottom = 0.15, right = 0.95, top = 0.9)
    plt.set_cmap('inferno_r')
    ax.scatter(t['pi_E'][bh_idx]/mas_to_rad, final_delta_arr[bh_idx],
               alpha = 0.2, c = np.log10(t['t_E'][bh_idx]), label = 'PopSyCLE BH', norm = norm)
    ax.scatter(t['pi_E'][other_idx]/mas_to_rad, final_delta_arr[other_idx],
               alpha = 0.2, c = np.log10(t['t_E'][other_idx]), label = 'PopSyCLE Other', s = 2, norm = norm)
    #############
    # Add the observations
    #############
    # OB110022, astrometry solution
    piEE_ob11 = -0.393
    piEE_erru_ob11 = 0.013
    piEE_errl_ob11 = 0.012
    piEN_ob11 = -0.071
    piEN_erru_ob11 = 0.014
    piEN_errl_ob11 = 0.014

    piE_ob11 = np.hypot(piEE_ob11, piEN_ob11)
    piE_ob11_erru = calc_pythag_err(piEE_ob11, piEE_erru_ob11, piEN_ob11, piEN_erru_ob11)
    piE_ob11_errl = calc_pythag_err(piEE_ob11, piEE_errl_ob11, piEN_ob11, piEN_errl_ob11)

    deltac_ob11 = 2.19/np.sqrt(8) # max shift calc'd from thetaE.
    deltac_ob11_errl = 1.06/np.sqrt(8)
    deltac_ob11_erru = 1.17/np.sqrt(8)
    tE_ob11 = 61.4
    ax.errorbar(piE_ob11, deltac_ob11,
                xerr = np.array([piE_ob11_errl, piE_ob11_erru]).reshape((2,1)),
                yerr = np.array([deltac_ob11_errl, deltac_ob11_erru]).reshape((2,1)),
                color = 'k', alpha = 0.99, capsize=5)
    im = ax.scatter(piE_ob11, deltac_ob11,
               alpha = 0.99, c = np.log10(tE_ob11), label = 'OB110022', norm = norm,
               marker = 's', s = 60)
    # OB150211, larger logZ solution
    # piE component, lower and upper errors
    piEE_ob15 = 0.012
    piEE_erru_ob15 = 0.007
    piEE_errl_ob15 = 0.007
    piEN_ob15 = 0.012
    piEN_erru_ob15 = 0.008
    piEN_errl_ob15 = 0.006

    piE_ob15 = np.hypot(piEE_ob15, piEN_ob15)
    piE_ob15_erru = calc_pythag_err(piEE_ob15, piEE_erru_ob15, piEN_ob15, piEN_erru_ob15)
    piE_ob15_errl = calc_pythag_err(piEE_ob15, piEE_errl_ob15, piEN_ob15, piEN_errl_ob15)

    deltac_ob15 = 1.13/np.sqrt(8) # max shift calc'd from thetaE.
    deltac_ob15_errl = 0.31/np.sqrt(8)
    deltac_ob15_erru = 0.26/np.sqrt(8)
    tE_ob15 = 137.5

    ax.errorbar(piE_ob15, deltac_ob15,
                xerr = np.array([piE_ob15_errl, piE_ob15_erru]).reshape((2,1)),
                yerr = np.array([deltac_ob15_errl, deltac_ob15_erru]).reshape((2,1)),
                color = 'k', alpha = 0.99, capsize=5)
    ax.scatter(piE_ob15, deltac_ob15,
               alpha = 0.99, c = np.log10(tE_ob15), label = 'OB150211', norm = norm,
               marker = 'v', s = 60)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('$\pi_E$')
    ax.set_ylabel('$\delta_{c,max}$ (mas)')
    ax.set_xlim(5 * 10**-4, 1)
    ax.set_ylim(5 * 10**-3, 4)
    ax.legend(loc=3, fontsize=12)
    ax_divider = make_axes_locatable(ax)
    cax = ax_divider.append_axes("top", size = "5%", pad = "15%")
    cb = colorbar(im, cax = cax, ticks = np.log10([1, 3, 10, 30, 100, 300]), orientation = 'horizontal')
    cax.xaxis.set_ticks_position('top')
    cax.set_xticklabels([1, 3, 10, 30, 100, 300])
    cax.set_xlabel('$t_E$ (days)')
#    plt.savefig('deltac_vs_piE_vs_tE.png')
    plt.show()

    return

def calc_pythag_err(A, sigA, B, sigB):
    """
    For a function of the form f = sqrt(A^2 + B^2) where
    A and B have errors sigA and sigB, calculate the
    error on f, assuming A and B are uncorrelated.
    """

    f = np.hypot(A, B)
    sigf = np.hypot(A * sigA, B * sigB)/f

    return sigf

def calc_ob110022_shift():
    # THIS IS DUMB
    # REDO
    # Values from webplot scraper...

    #####
    # Try 1
    #####
    x1 = np.array([2.6779 * 10**-1, -4.5596 * 10**-1])
    x2 = np.array([5.8423 * 10**-2, 1.0510 * 10**-1])

    y1 = np.array([-4.4928, -3.0920])
    y2 = np.array([-5.0524, -3.9513])

    #####
    # Try 2
    #####
    z1 = np.array([-2.5699 * 10**-1, -4.6569 * 10**-1])
    z2 = np.array([6.7393 * 10**-2, 8.7739 * 10**-2])

    w1 = np.array([-4.4914, -3.1114])
    w2 = np.array([-5.0585, -2.9629])

    print(np.linalg.norm(x1 - x2))
    print(np.linalg.norm(y1 - y2))
    print(np.linalg.norm(z1 - z2))
    print(np.linalg.norm(w1 - w2))

    return

def calc_velocity():
    import astropy.coordinates as coord
    import astropy.units as u

    data = munge_ob150211.getdata()
    mnest_root = pspl_ast_phot['ob150211']

    # Load up the best-fit data
    _in = open(mnest_root + '_best.fits', 'rb')

    pars1 = pickle.load(_in)
    values1 = pickle.load(_in)
    logZ_sol1 = pickle.load(_in)
    maxL_sol1 = pickle.load(_in)

    pars2 = pickle.load(_in)
    values2 = pickle.load(_in)
    logZ_sol2 = pickle.load(_in)
    maxL_sol2 = pickle.load(_in)

    _in.close()

    # Fetch the lens proper motions. Only for the 1st solution
    # as this is the one we will adopt for the paper.
    dL = values2['dL'][0]
    muL_E = values2['muL_E'][0]
    muL_N = values2['muL_N'][0]
    muLe_E = np.mean([values2['muL_E'][1], values2['muL_E'][2]])
    muLe_N = np.mean([values2['muL_N'][1], values2['muL_N'][2]])
    print(dL, muL_E, muL_N)

    c1 = coord.ICRS(ra=data['raL']*u.degree, dec=data['decL']*u.degree,
                distance=dL*u.pc,
                pm_ra_cosdec=muL_E*u.mas/u.yr,
                pm_dec=muL_N*u.mas/u.yr)

    galcen_distance = 8*u.kpc
    pm_en = [muL_E, muL_N] * u.mas/u.yr
    v_e, v_n = -(galcen_distance * pm_en).to(u.km/u.s, u.dimensionless_angles())
    ve_e = -(galcen_distance * muLe_E * u.mas/u.yr).to(u.km/u.s, u.dimensionless_angles())
    ve_n = -(galcen_distance * muLe_N * u.mas/u.yr).to(u.km/u.s, u.dimensionless_angles())

    gal = c1.transform_to(coord.Galactic)
    muL_l = gal.pm_l_cosb
    muL_b = gal.pm_b
    v_l = -(galcen_distance * muL_l).to(u.km/u.s, u.dimensionless_angles())
    v_b = -(galcen_distance * muL_b).to(u.km/u.s, u.dimensionless_angles())

    fmt = '    {0:8s} = {1:8.3f} +/- {2:8.3f}  {3:8s}'

    print('Proper Motion for OB150322:')
    print('  Celestial:')
    print(fmt.format('muL_E', muL_E, muLe_E, 'mas/yr'))
    print(fmt.format('muL_N', muL_N, muLe_N, 'mas/yr'))
    print('  Galactic:')
    print(fmt.format('muL_l', muL_l, 0.0, 'mas/yr'))
    print(fmt.format('muL_b', muL_b, 0.0, 'mas/yr'))

    print('Velocity for OB150322 at dL=', dL)
    print('  Celestial:')
    print(fmt.format('vL_E', v_e, ve_e, 'km/s'))
    print(fmt.format('vL_N', v_n, ve_n, 'km/s'))
    print('  Galactic:')
    print(fmt.format('vL_l', v_l, 0.0, 'km/s'))
    print(fmt.format('vL_b', v_b, 0.0, 'km/s'))

    return


def plot_cmds():
    """
    Everything is in
    /Users/jlu/work/microlens/OB150211/a_2019_05_04/notes/7_other_phot.ipynb
    """

    # Read in the Gaia and 2MASS catalogs.
    tmass = Table.read('/Users/jlu/work/microlens/OB150211/tmass.fits')
    gaia = Table.read('/Users/jlu/work/microlens/OB150211/gaia.fits')

    tt_t = np.where(tmass['name'] == 'ob150211')
    tt_g = np.where(gaia['name'] == 'ob150211')

    plt.close(1)
    plt.figure(1, figsize=(10, 4))
    plt.subplots_adjust(bottom=0.2, left=0.1, right=0.8, wspace=0.4)

    plt.subplot(1, 2, 1)
    plt.plot(tmass['Jmag'] - tmass['Kmag'], tmass['Jmag'], 'k.', alpha=0.5)
    plt.plot(tmass['Jmag'][tt_t] - tmass['Kmag'][tt_t], tmass['Jmag'][tt_t], 'ro', ms=10)
    plt.xlim(0, 3)
    plt.gca().invert_yaxis()
    plt.xlabel('J-K (mag)')
    plt.ylabel('J (mag)')
    plt.title('2MASS')

    plt.subplot(1, 2, 2)
    sc = plt.scatter(gaia['bp_rp'], gaia['phot_g_mean_mag'], c=gaia['parallax'], s=10,
                     vmin=-1, vmax=2, cmap=plt.cm.viridis)
    plt.plot(gaia['bp_rp'][tt_g], gaia['phot_g_mean_mag'][tt_g], 'ro', ms=10)
    plt.xlim(0, 6.5)
    plt.gca().invert_yaxis()
    plt.xlabel('G$_{BP}$ - G$_{RP}$ (mag)')
    plt.ylabel('G (mag)')
    plt.title('Gaia')

    fig = plt.gcf()
    cb_ax = fig.add_axes([0.83, 0.13, 0.02, 0.77])
    cbar = fig.colorbar(sc, cax=cb_ax, label='Parallax (mas)')

    plt.savefig(paper_dir + 'cmds.png')

    return

def calc_blending_kp():
    """
    Read in a NIRC2 catalog and add up the flux
    from all the sources that are within 1.3" from
    the target. This should match the b_sff derived
    from the fits.
    """
    scale = 0.00995 # arcsec / pixel

    # Read in a NIRC2 starlist (high-ish quality)
    nirc2_lis = '/g/lu/data/microlens/17jun05/combo/starfinder/'
    nirc2_lis += 'mag17jun05_ob150211_kp_rms_named.lis'

    foo = starlists.read_starlist(nirc2_lis, error=True)

    # Find the target
    tdx = np.where(foo['name'] == 'ob150211')[0][0]

    # Calculate the distance from the target for each star
    r2d = np.hypot(foo['x'] - foo['x'][tdx],
                   foo['y'] - foo['y'][tdx])
    r2d *= scale

    foo['r2d'] = r2d

    # Find those targets within the OGLE aperture
    rdx = np.where(r2d < 1.5)[0]
    print(foo[rdx])

    # Calculate the source flux fraction assuming
    # that the lens is dark.
    f_src_lens = foo['flux'][tdx]
    f_neighbors = foo['flux'][rdx[1:]].sum()
    f_total = f_src_lens + f_neighbors

    print('f_src_lens  = ', f_src_lens)
    print('f_neighbors = ', f_neighbors)
    print('f_total     = ', f_total)
    print('')
    print('f_N / f_tot = {0:.2f}'.format( f_neighbors / f_total))
    print('(f_S + f_L) / f_tot = {0:.2f}'.format( f_src_lens / f_total))

    b_sff = foo['flux'][tdx] / (foo['flux'][rdx].sum())
    print('b_sff = {0:.2f}'.format( b_sff ))

    fit_b_sff = 0.90

    f_lens = f_src_lens - (fit_b_sff * f_total)
    print('f_lens = ', f_lens)
    f_src = f_src_lens - f_lens
    print('f_src  = ', f_src)

    print('f_S / (f_S + f_L) = {0:.2f}'.format(f_src / f_src_lens))


    return

def compare_all_linear_motions(save_all=False):
    """
    Plot and calculate the significance of the astrometric signal for each target.
    See fit_velocities.py in jlu/microlens for more info.
    Saves the calculations in a table.

    Parameters
    ----------
    save_all : bool, optional
      If True, all plots are saved. If False, only the figure of the off-peak
      linear fit of the target is saved.
      Default is False.
    """
    from jlu.microlens import fit_velocities

    # Make directory to hold table and figures
    out_dir = paper_dir+ 'compare_linear_motion'
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)

    ret_dir = os.getcwd()
    os.chdir(out_dir)

    targets = list(epochs.keys())
    objects = []
    signal = []
    average_deviation = np.zeros(len(targets))
    average_deviation_error = np.zeros(len(targets))
    all_chi2 = np.zeros(len(targets))
    all_chi2_red = np.zeros(len(targets))
    cut_chi2 = np.zeros(len(targets))
    cut_chi2_red = np.zeros(len(targets))

    n = 1
    for t, target in enumerate(targets):
        tab = fit_velocities.StarTable(target)

        average, var, all_chi2s, cut_chi2s = tab.compare_linear_motion(fign_start=n,
                                                                       return_results=True,
                                                                       save_all=save_all)
        average_deviation[t] = average
        average_deviation_error[t] = np.sqrt(var)
        sig = np.abs(average_deviation[t] / average_deviation_error[t])
        signal.append("${:.1f}\sigma$".format(sig))
        all_chi2[t] = all_chi2s[0]
        all_chi2_red[t] = all_chi2s[1]
        cut_chi2[t] = cut_chi2s[0]
        cut_chi2_red[t] = cut_chi2s[1]

        objects.append(target.upper())
        n += 3

    av_dev = Column(data=average_deviation, name='$\overline{\Delta r}$',
                    format='{:.3f}', unit='mas')
    av_deve = Column(data=average_deviation_error, name='$\sigma_{\overline{\Delta r}}$',
                     format='{:.3f}', unit='mas')
    signal = Column(data=signal, name='significance')
    all_chi2 = Column(data=all_chi2, name='$\chi^2$a', format='{:.2f}')
    all_chi2_red = Column(data=all_chi2_red, name='$\chi^2_{red}$a', format='{:.2f}')
    cut_chi2 = Column(data=cut_chi2, name='$\chi^2$', format='{:.2f}')
    cut_chi2_red = Column(data=cut_chi2_red, name='$\chi^2_{red}$', format='{:.2f}')

    tab = Table((Column(data=objects, name='Object'), all_chi2, all_chi2_red, cut_chi2, cut_chi2_red,\
                     av_dev, av_deve, signal))
    tab.write('astrom_significance.tex', format='aastex', overwrite=True)

    print(tab)

    os.chdir(ret_dir)


####################################
### PopSyCLE Visualization Stuff ###
####################################
def murel_popsycle():
    t = Table.read('/Users/casey/scratch/papers/microlens_2019/plot_files/Mock_EWS.fits')

    mas_to_rad = 4.848 * 10**-9

    long_idx = np.where(t['t_E'] > 100)[0]

    idx = np.where((t['t_E'] > 100) &
                   (t['pi_E']/mas_to_rad < 0.1))[0]

    print(len(t['t_E']))
    print(len(long_idx))
    print(len(idx))

    murel = t['mu_rel']

    murel_long = t['mu_rel'][long_idx]

    murel_long_piE = t['mu_rel'][idx]

#    fig = plt.figure(1, figsize=(6,6))
#    plt.clf()
#    plt.hist(murel,
#             bins = 50,
#             normed = True)
#    xmin, xmax = plt.xlim()
#    x = np.linspace(xmin, xmax, 100)
#    y = norm.pdf(x, mean, std)
#    plt.plot(x,y)
#    plt.xlabel('murel (mas/yr)')

    fig = plt.figure(2, figsize=(6,6))
    plt.clf()
    plt.hist(murel,
#             bins = np.linspace(0, 20, 51),
             bins = np.logspace(-1, 1.3, 51),
             histtype = 'step',
             density=True,
             label = 'All')
    plt.hist(murel_long,
#             bins = np.linspace(0, 20, 51),
             bins = np.logspace(-1, 1.3, 51),
             histtype = 'step',
             density=True,
             label = 'Long')
    plt.hist(murel_long_piE,
#             bins = np.linspace(0, 20, 51),
             bins = np.logspace(-1, 13., 51),
             histtype = 'step',
             density=True,
             label = 'Long, piE < 0.1')
    plt.xlabel('murel (mas/yr)')
    plt.xscale('log')
    plt.legend()

def dLdS_popsycle():
    t = Table.read('/Users/casey/scratch/papers/microlens_2019/plot_files/Mock_EWS.fits')

    long_idx = np.where(t['t_E'] > 100)[0]

    dL = t['rad_L']
    dS = t['rad_S']
    dLdS = (1.0 * dL)/(1.0 * dS)

    dL_long = t['rad_L'][long_idx]
    dS_long = t['rad_S'][long_idx]
    dLdS_long = (1.0 * dL_long)/(1.0 * dS_long)

    fig = plt.figure(1, figsize=(6,6))
    plt.clf()
    plt.hist(dLdS, bins = np.linspace(0, 1, 11), histtype = 'step', density=True, label = 'All')
    plt.hist(dLdS_long, bins = np.linspace(0, 1, 11), histtype = 'step', density=True, label = 'Long')
    plt.xlabel('dL/dS')
    plt.legend()

    fig = plt.figure(2, figsize=(6,6))
    plt.clf()
    plt.hist(dL, bins = np.linspace(0, 12, 13), histtype = 'step', density=True, label = 'All')
    plt.hist(dL_long, bins = np.linspace(0, 12, 13), histtype = 'step', density=True, label = 'Long')
    plt.xlabel('dL (kpc)')
    plt.legend()

    fig = plt.figure(3, figsize=(6,6))
    plt.clf()
    plt.hist(dLdS, bins = np.logspace(-2, 1))
    plt.xlabel('dL/dS')
    plt.xscale('log')
    plt.show()

    plt.show()

def dS_popsycle():
    t = Table.read('/Users/casey/scratch/papers/microlens_2019/plot_files/Mock_EWS.fits')

    long_idx = np.where(t['t_E'] > 100)[0]

    dS = t['rad_S']

    piS = 1.0/dS

    dS_long = t['rad_S'][long_idx]

    mean, std = norm.fit(piS)
    print(mean, std)

    fig = plt.figure(1, figsize=(6,6))
    plt.clf()
    plt.hist(piS,
             bins = 50,
             normed = True)
    xmin, xmax = plt.xlim()
    x = np.linspace(xmin, xmax, 100)
    y = norm.pdf(x, mean, std)
    plt.plot(x,y)
    plt.xlabel('piS (mas)')

    fig = plt.figure(2, figsize=(6,6))
    plt.clf()
    plt.hist(dS,
             bins = np.linspace(0, 20, 21),
             histtype = 'step',
             density=True,
             label = 'All')
    plt.hist(dS_long,
             bins = np.linspace(0, 20, 21),
             histtype = 'step',
             density=True,
             label = 'Long')
    plt.xlabel('dS (kpc)')
    plt.legend()

    fig = plt.figure(3, figsize=(6,6))
    plt.clf()
    plt.hist(1.0/dS,
             bins = np.linspace(0, 0.25, 21),
             histtype = 'step',
             density=True,
             label = 'All')
    plt.hist(1.0/dS_long,
             bins = np.linspace(0, 0.25, 21),
             histtype = 'step',
             density=True,
             label = 'Long')
    plt.xlabel('piS (mas)')
    plt.legend()

    plt.show()

def pi_popsycle():
    t = Table.read('/Users/casey/scratch/papers/microlens_2019/plot_files/Mock_EWS.fits')

    long_idx = np.where(t['t_E'] > 100)[0]

    dS = t['rad_S']
    piS = 1.0/dS
    dS_long = t['rad_S'][long_idx]
    piS_long = 1.0/dS_long

    dL = t['rad_L']
    piL = 1.0/dL
    dL_long = t['rad_L'][long_idx]
    piL_long = 1.0/dL_long

    meanS, stdS = norm.fit(piS)
    print('Mean, STD for piS')
    print(meanS, stdS)

    meanL, stdL = norm.fit(piL)
    print('Mean, STD for piL')
    print(meanL, stdL)

    bins = np.linspace(0, 0.5, 50)
#    bins = np.logspace(-4, 0, 50)

    print('Max piL')
    print(np.max(piL))

    print('Max piS')
    print(np.max(piS))

    fig = plt.figure(1, figsize=(6,6))
    plt.clf()
    plt.hist(piS,
             bins = bins,
#             normed = True,
             label = 'piS',
             histtype = 'step')
    plt.hist(piL,
             bins = bins,
#             normed = True,
             label = 'piL',
             histtype='step')
    plt.hist(piL - piS,
             bins = bins,
             label = 'piRel',
             histtype = 'step')
#    xmin, xmax = plt.xlim()
#    x = np.linspace(xmin, xmax, 100)
#    y = norm.pdf(x, mean, std)
#    plt.plot(x,y)
    plt.legend()
#    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('pi (mas)')

#    fig = plt.figure(2, figsize=(6,6))
#    plt.clf()
#    plt.hist(dS,
#             bins = np.linspace(0, 20, 21),
#             histtype = 'step',
#             density=True,
#             label = 'All')
#    plt.hist(dS_long,
#             bins = np.linspace(0, 20, 21),
#             histtype = 'step',
#             density=True,
#             label = 'Long')
#    plt.xlabel('dS (kpc)')
#    plt.legend()
#
#    fig = plt.figure(3, figsize=(6,6))
#    plt.clf()
#    plt.hist(1.0/dS,
#             bins = np.linspace(0, 0.25, 21),
#             histtype = 'step',
#             density=True,
#             label = 'All')
#    plt.hist(1.0/dS_long,
#             bins = np.linspace(0, 0.25, 21),
#             histtype = 'step',
#             density=True,
#             label = 'Long')
#    plt.xlabel('piS (mas)')
#    plt.legend()
#
    plt.show()


##################################
### OB150211 solo paper things ###
##################################

def make_obs_table_ob150211():
    """
    Make a LaTeX table for all of the observations of the three targets from 2014/2015.
    """

    targets = list(epochs.keys())

    tables = {}

    # Build only OB150211
    target = 'ob150211'

    n_epochs = len(epochs[target])

    obj_name = np.repeat(target.upper(), n_epochs)
    obj_name[1:] = ''

    date = np.zeros(n_epochs, dtype='S10')
    tint = np.zeros(n_epochs, dtype=int)
    n_exp = np.zeros(n_epochs, dtype=int)
    strehl = np.zeros(n_epochs, dtype=float)
    fwhm = np.zeros(n_epochs, dtype=float)
    strehl_e = np.zeros(n_epochs, dtype=float)
    fwhm_e = np.zeros(n_epochs, dtype=float)
    n_star = np.zeros(n_epochs, dtype=int)
    m_base = np.zeros(n_epochs, dtype=float)
    ast_err = np.zeros(n_epochs, dtype=float)
    phot_err = np.zeros(n_epochs, dtype=float)

    # Loop through each epoch and grab information to populate our table.
    for ee in range(n_epochs):
        epoch = epochs[target][ee]
        img_file = '/g/lu/data/microlens/{0:s}/combo/mag{0:s}_{1:s}_kp.fits'.format(epoch, target)
        log_file = '/g/lu/data/microlens/{0:s}/combo/mag{0:s}_{1:s}_kp.log'.format(epoch, target)
        pos_file = '/g/lu/data/microlens/{0:s}/combo/starfinder/plotPosError_{1:s}_kp.txt'.format(epoch, target)

        # Fetch stuff from the image header.
        hdr = fits.getheader(img_file)
        date[ee] = hdr['DATE-OBS'].strip()
        tint[ee] = np.round(float(hdr['ITIME']) * float(hdr['COADDS']), 0)

        # From the log file, average Strehl and FWHM
        _log = Table.read(log_file, format='ascii')
        _log.rename_column('col2', 'fwhm')
        _log.rename_column('col3', 'strehl')
        strehl[ee] = _log['strehl'].mean()
        strehl_e[ee] = _log['strehl'].std()
        fwhm[ee] = _log['fwhm'].mean()
        fwhm_e[ee] = _log['fwhm'].std()
        n_exp[ee] = len(_log)

        # Read in the stats file from the analysis of the AIROPA starlist.
        _pos = open(pos_file, 'r')
        lines = _pos.readlines()
        _pos.close()

        n_star[ee] = int(lines[0].split(':')[-1])
        ast_err[ee] = float(lines[1].split(':')[-1])
        phot_err[ee] = float(lines[2].split(':')[-1])
        m_base[ee] = float(lines[3].split('=')[-1])

    # Make our table
    c_date = Column(data=date, name='Date', format='{:10s}')
    c_tint = Column(data=tint, name='t$_{int}$', format='{:3.0f}', unit='s')
    c_nexp = Column(data=n_exp, name='N$_{exp}$', format='{:3d}')
    c_fwhm = Column(data=fwhm, name='FWHM', format='{:3.0f}', unit='mas')
    c_fwhm_err = Column(data=fwhm_e, name='FWHM$_{err}$', format='{:3.0f}', unit='mas')
    c_strehl = Column(data=strehl, name='Strehl', format='{:4.2f}')
    c_strehl_err = Column(data=strehl_e, name='Strehl$_{err}$', format='{:4.2f}')
    c_nstar = Column(data=n_star, name='N$_{star}$', format='{:4d}')
    c_mbase = Column(data=m_base, name='Kp$_{turn}$', format='{:4.1f}', unit='mag')
    c_asterr = Column(data=ast_err, name='$\sigma_{ast}$', format='{:5.2f}', unit='mas')
    c_photerr = Column(data=phot_err, name='$\sigma_{phot}$', format='{:5.2f}', unit='mag')

    final_table = Table((c_date, c_tint, c_nexp,
                         c_fwhm, c_fwhm_err, c_strehl, c_strehl_err,
                         c_nstar, c_mbase, c_asterr, c_photerr))

    print(final_table)

    final_table.write(paper_dir + 'data_table.tex', format='aastex', overwrite=True)

    return

def plot_astrometry(target):
    data = munge_ob150211.getdata()

    best = {}
    best['mL'] = 10.386
    best['t0'] = 57211.845
    # best['xS0_E'] = 0.023101
    best['xS0_E'] = 0.022981
    # best['xS0_N'] = -0.113297
    best['xS0_N'] = -0.113357
    best['beta'] = -0.462895
    best['muL_E'] = -0.129050
    best['muL_N'] = -3.92588
    best['muS_E'] = 0.7187671
    best['muS_N'] = -1.77336
    best['dL'] = 5841.39
    best['dS'] = best['dL'] / 0.8772
    best['b_sff'] = 0.6451
    best['mag_src'] = 17.77

    # mymodel = model.PSPL_parallax(data['raL'], data['decL'],
    #                                    best['mL'], best['t0'], np.array([best['xS0_E'], best['xS0_N']]),
    #                                    best['beta'], np.array([best['muL_E'], best['muL_N']]),
    #                                    np.array([best['muS_E'], best['muS_N']]),
    #                                    best['dL'], best['dS'], best['b_sff'], best['mag_src'])

    fitter = model_fitter.PSPL_parallax_Solver(data, outputfiles_basename=pspl_ast_phot[target])
    fitter.load_mnest_results()
    mymodel = fitter.get_best_fit_model(use_median=True)

    #####
    # Astrometry on the sky
    #####
    plt.figure(2)
    plt.clf()

    # Data
    plt.errorbar(data['xpos']*1e3, data['ypos']*1e3,
                     xerr=data['xpos_err']*1e3, yerr=data['ypos_err']*1e3, fmt='k.')

    # 1 day sampling over whole range
    t_mod = np.arange(data['t_ast'].min(), data['t_ast'].max(), 1)

    # Model - usually from fitter
    pos_out = mymodel.get_astrometry(t_mod)
    plt.plot(pos_out[:, 0]*1e3, pos_out[:, 1]*1e3, 'r-')

    plt.gca().invert_xaxis()
    plt.xlabel(r'$\Delta \alpha^*$ (mas)')
    plt.ylabel(r'$\Delta \delta$ (mas)')
    plt.title('Input Data and Output Model')

    #####
    # astrometry vs. time
    #####
    plt.figure(3)
    plt.clf()
    plt.errorbar(data['t_ast'], data['xpos']*1e3,
                     yerr=data['xpos_err']*1e3, fmt='k.')
    plt.plot(t_mod, pos_out[:, 0]*1e3, 'r-')
    plt.xlabel('t - t0 (days)')
    plt.ylabel(r'$\Delta \alpha^*$ (mas)')
    plt.title('Input Data and Output Model')

    plt.figure(4)
    plt.clf()
    plt.errorbar(data['t_ast'], data['ypos']*1e3,
                     yerr=data['ypos_err']*1e3, fmt='k.')
    plt.plot(t_mod, pos_out[:, 1]*1e3, 'r-')
    plt.xlabel('t - t0 (days)')
    plt.ylabel(r'$\Delta \delta$ (mas)')
    plt.title('Input Data and Output Model')

    #####
    # Remove the unlensed motion (proper motion)
    # astrometry vs. time
    #####
    # Make the model unlensed points.
    p_mod_unlens_tdat = mymodel.get_astrometry_unlensed(data['t_ast'])
    x_mod_tdat = p_mod_unlens_tdat[:, 0]
    y_mod_tdat = p_mod_unlens_tdat[:, 1]
    x_no_pm = data['xpos'] - x_mod_tdat
    y_no_pm = data['ypos'] - y_mod_tdat

    # Make the dense sampled model for the same plot
    dp_tmod_unlens = mymodel.get_astrometry(t_mod) - mymodel.get_astrometry_unlensed(t_mod)
    x_mod_no_pm = dp_tmod_unlens[:, 0]
    y_mod_no_pm = dp_tmod_unlens[:, 1]

    # Prep some colorbar stuff
    cmap = plt.cm.viridis
    norm = plt.Normalize(vmin=data['t_ast'].min(), vmax=data['t_ast'].max())
    smap = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    smap.set_array([])

    # Average the data for the last three years.
    t_year = []
    x_year = []
    y_year = []
    xe_year = []
    ye_year = []

    t_dat_yr = multinest_plot.mmjd_to_year(data['t_ast'])

    min_year = np.floor(t_dat_yr.min())
    max_year = np.ceil(t_dat_yr.max())

    years = np.arange(min_year, max_year + 1, 1)
    for tt in range(len(years)-1):
        ii = np.where((years[tt] < t_dat_yr) & (t_dat_yr <= years[tt+1]))[0]

        if tt == 100:
            t_year = data['t_ast'][ii].data
            x_year = x_no_pm[ii]
            y_year = y_no_pm[ii]
            xe_year = data['xpos_err'][ii]
            ye_year = data['ypos_err'][ii]
        else:
            #wgts = np.hypot(data['xpos_err'][ii], data['ypos_err'][ii])
            wgts = (data['xpos_err'][ii] + data['ypos_err'][ii]) / 2.0
            t_avg, t_std = weighted_avg_and_std(data['t_ast'][ii], wgts )
            x_avg, x_std = weighted_avg_and_std(x_no_pm[ii], wgts )
            y_avg, y_std = weighted_avg_and_std(y_no_pm[ii], wgts )

            t_avg = data['t_ast'][ii].mean()
            t_std = data['t_ast'][ii].std()
            x_avg = x_no_pm[ii].mean()
            x_std = x_no_pm[ii].std()
            y_avg = y_no_pm[ii].mean()
            y_std = y_no_pm[ii].std()

            t_year = np.append( t_year, t_avg )
            x_year = np.append( x_year, x_avg )
            y_year = np.append( y_year, y_avg)
            xe_year = np.append( xe_year, x_std / np.sqrt(len(ii) - 1) )
            ye_year = np.append( ye_year, y_std / np.sqrt(len(ii) - 1) )


    t_year = np.array(t_year)
    x_year = np.array(x_year)
    y_year = np.array(y_year)
    xe_year = np.array(xe_year)
    ye_year = np.array(ye_year)
    print(t_year)
    print(x_year)

    plt.figure(5)
    plt.clf()
    plt.scatter(x_year*1e3, y_year*1e3, c=t_year,
                cmap=cmap, norm=norm, s=10)
    plt.errorbar(x_year*1e3, y_year*1e3,
                 xerr=xe_year*1e3, yerr=ye_year*1e3,
                 fmt='none', ecolor=smap.to_rgba(t_year))
    plt.scatter(x_mod_no_pm*1e3, y_mod_no_pm*1e3, c=t_mod, cmap=cmap, norm=norm, s=4)
    plt.gca().invert_xaxis()
    plt.axis('equal')
    plt.xlabel(r'$\Delta \alpha^*$ (mas)')
    plt.ylabel(r'$\Delta \delta$ (mas)')
    plt.colorbar()


    plt.figure(6)
    plt.clf()
    plt.scatter(x_no_pm*1e3, y_no_pm*1e3, c=data['t_ast'],
                cmap=cmap, norm=norm, s=10)
    plt.errorbar(x_no_pm*1e3, y_no_pm*1e3,
                 xerr=data['xpos_err']*1e3, yerr=data['ypos_err']*1e3,
                 fmt='none', ecolor=smap.to_rgba(data['t_ast']))
    plt.scatter(x_mod_no_pm*1e3, y_mod_no_pm*1e3, c=t_mod, cmap=cmap, norm=norm, s=4)
    plt.gca().invert_xaxis()
    plt.axis('equal')
    plt.xlabel(r'$\Delta \alpha^*$ (mas)')
    plt.ylabel(r'$\Delta \delta$ (mas)')
    plt.colorbar()

    return

def plot_ob150211_mass_posterior():
    """
    # Lines are median, +/- 3 sigma.
    Line is MAP solution, global/mode 1 (identical).
    """

    fontsize1 = 18
    fontsize2 = 14

    tab = Table.read(pspl_ast_phot['ob150211'] + '.fits')
    smy = Table.read(pspl_ast_phot['ob150211'] + 'summary.fits')
    print(smy['MAP mL'])
    mmax = tab['mL'].max()
    mmin = tab['mL'].min()

    plt.figure(1)
    plt.clf()
    plt.subplots_adjust(bottom = 0.15)
    xliml = 0.9 * mmin
    xlimu = 1.1 * mmax
    bins = np.logspace(np.log10(xliml), np.log10(xlimu), 500)

    n, b = np.histogram(tab['mL'], bins = bins,
                        weights = tab['weights'], normed = True)

    n = norm_kde(n, 10.)
    b0 = 0.5 * (b[1:] + b[:-1])

    n, b, _ = plt.hist(b0, bins = b, weights = n)

    plt.xlabel('Lens Mass $(M_\odot)$', fontsize=fontsize1, labelpad=10)
    plt.xlim(xliml, xlimu)
    plt.ylabel('Relative probability', fontsize=fontsize1, labelpad=10)
    plt.xticks(fontsize=fontsize2)
    plt.yticks(fontsize=fontsize2)

    plt.xscale('log')


#    ##########
#    # Calculate 3-sigma boundaries for mass limits.
#    ##########
#
#    sig1_hi = 0.682689
#    sig1_lo = 1.0 - sig1_hi
#    sig_med = 0.5
#    sig2_hi = 0.9545
#    sig2_lo = 1.0 - sig2_hi
#    sig3_hi = 0.9973
#    sig3_lo = 1.0 - sig3_hi
#
#    quantiles = [sig3_lo, sig2_lo, sig1_lo, sig_med, sig1_hi, sig2_hi, sig3_hi]
#
#    mass_quants = model_fitter.weighted_quantile(tab['mL'], quantiles,
#                                                 sample_weight=tab['weights'])
#
#    for qq in range(len(quantiles)):
#        print( 'Mass at {0:.1f}% quantiles:  M = {1:5.2f}'.format(quantiles[qq]*100, mass_quants[qq]))
#
#    ax = plt.axis()
#    # plot median and +/- 3 sigma
#    plt.axvline(mass_quants[0], color='k', linestyle='--')
#    plt.axvline(mass_quants[3], color='k', linestyle='--')
#    plt.axvline(mass_quants[-1], color='k', linestyle='--')
#    plt.show()

    plt.axvline(smy['MAP mL'][0], color='k', linestyle='-', lw = 3)
    plt.show()

    plt.savefig(paper_dir + 'ob150211_mass_posterior.pdf')
    plt.savefig(paper_dir + 'ob150211_mass_posterior.png')

    return

def plot_ob150029_mass_posterior():
    """
    Lines are median, +/- 3 sigma.
    """

    fontsize1 = 18
    fontsize2 = 14

    tab = Table.read('/g/lu/scratch/jlu/work/microlens/OB150029/a_2019_04_19/notes/4_fit_phot_astrom_parallax/aa_.fits')
    mmax = tab['mL'].max()
    mmin = tab['mL'].min()

    plt.figure(1)
    plt.clf()
    plt.subplots_adjust(bottom = 0.15)
    xliml = 0.9 * mmin
    xlimu = 1.1 * mmax
#    bins = np.logspace(np.log10(xliml), np.log10(xlimu), 500)
    bins = np.linspace(xliml, xlimu, 500)

    n, b = np.histogram(tab['mL'], bins = bins,
                        weights = tab['weights'], normed = True)

    n = norm_kde(n, 10.)
    b0 = 0.5 * (b[1:] + b[:-1])

    n, b, _ = plt.hist(b0, bins = b, weights = n)

    plt.xlabel('Lens Mass $(M_\odot)$', fontsize=fontsize1, labelpad=10)
    plt.xlim(xliml, 2)
    plt.ylabel('Relative probability', fontsize=fontsize1, labelpad=10)
    plt.xticks(fontsize=fontsize2)
    plt.yticks(fontsize=fontsize2)

#    plt.xscale('log')

    ##########
    # Calculate 3-sigma boundaries for mass limits.
    ##########
    sig1_hi = 0.682689
    sig1_lo = 1.0 - sig1_hi
    sig_med = 0.5
    sig2_hi = 0.9545
    sig2_lo = 1.0 - sig2_hi
    sig3_hi = 0.9973
    sig3_lo = 1.0 - sig3_hi

    quantiles = [sig3_lo, sig2_lo, sig1_lo, sig_med, sig1_hi, sig2_hi, sig3_hi]

    mass_quants = model_fitter.weighted_quantile(tab['mL'], quantiles,
                                                 sample_weight=tab['weights'])

    for qq in range(len(quantiles)):
        print( 'Mass at {0:.1f}% quantiles:  M = {1:5.2f}'.format(quantiles[qq]*100, mass_quants[qq]))

    ax = plt.axis()
    # plot median and +/- 3 sigma
    plt.axvline(mass_quants[0], color='k', linestyle='--')
    plt.axvline(mass_quants[3], color='k', linestyle='--')
    plt.axvline(mass_quants[-1], color='k', linestyle='--')
    plt.show()
    plt.savefig(paper_dir + 'ob150029_mass_posterior.pdf')
    plt.savefig(paper_dir + 'ob150029_mass_posterior.png')

    return

def plot_ob150211_mass_posterior():
    """
    # Lines are median, +/- 3 sigma.
    Line is MAP solution, global/mode 1 (identical).
    """

    fontsize1 = 18
    fontsize2 = 14

    tab = Table.read(pspl_ast_phot['ob150211'] + '.fits')
    smy = Table.read(pspl_ast_phot['ob150211'] + 'summary.fits')
    print(smy['MAP mL'])
    mmax = tab['mL'].max()
    mmin = tab['mL'].min()

    plt.figure(1)
    plt.clf()
    plt.subplots_adjust(bottom = 0.15)
    xliml = 0.9 * mmin
    xlimu = 1.1 * mmax
    bins = np.logspace(np.log10(xliml), np.log10(xlimu), 500)

    n, b = np.histogram(tab['mL'], bins = bins,
                        weights = tab['weights'], normed = True)

    n = norm_kde(n, 10.)
    b0 = 0.5 * (b[1:] + b[:-1])

    n, b, _ = plt.hist(b0, bins = b, weights = n)

    plt.xlabel('Lens Mass $(M_\odot)$', fontsize=fontsize1, labelpad=10)
    plt.xlim(xliml, xlimu)
    plt.ylabel('Relative probability', fontsize=fontsize1, labelpad=10)
    plt.xticks(fontsize=fontsize2)
    plt.yticks(fontsize=fontsize2)

    plt.xscale('log')


#    ##########
#    # Calculate 3-sigma boundaries for mass limits.
#    ##########
#
#    sig1_hi = 0.682689
#    sig1_lo = 1.0 - sig1_hi
#    sig_med = 0.5
#    sig2_hi = 0.9545
#    sig2_lo = 1.0 - sig2_hi
#    sig3_hi = 0.9973
#    sig3_lo = 1.0 - sig3_hi
#
#    quantiles = [sig3_lo, sig2_lo, sig1_lo, sig_med, sig1_hi, sig2_hi, sig3_hi]
#
#    mass_quants = model_fitter.weighted_quantile(tab['mL'], quantiles,
#                                                 sample_weight=tab['weights'])
#
#    for qq in range(len(quantiles)):
#        print( 'Mass at {0:.1f}% quantiles:  M = {1:5.2f}'.format(quantiles[qq]*100, mass_quants[qq]))
#
#    ax = plt.axis()
#    # plot median and +/- 3 sigma
#    plt.axvline(mass_quants[0], color='k', linestyle='--')
#    plt.axvline(mass_quants[3], color='k', linestyle='--')
#    plt.axvline(mass_quants[-1], color='k', linestyle='--')
#    plt.show()

    plt.axvline(smy['MAP mL'][0], color='k', linestyle='-', lw = 3)
    plt.show()

    plt.savefig(paper_dir + 'ob150211_mass_posterior.pdf')
    plt.savefig(paper_dir + 'ob150211_mass_posterior.png')

    return

def plot_ob150029_mass_posterior():
    """
    Lines are median, +/- 3 sigma.
    """

    fontsize1 = 18
    fontsize2 = 14

    tab = Table.read('/g/lu/scratch/jlu/work/microlens/OB150029/a_2019_04_19/notes/4_fit_phot_astrom_parallax/aa_.fits')
    mmax = tab['mL'].max()
    mmin = tab['mL'].min()

    plt.figure(1)
    plt.clf()
    plt.subplots_adjust(bottom = 0.15)
    xliml = 0.9 * mmin
    xlimu = 1.1 * mmax
#    bins = np.logspace(np.log10(xliml), np.log10(xlimu), 500)
    bins = np.linspace(xliml, xlimu, 500)

    n, b = np.histogram(tab['mL'], bins = bins,
                        weights = tab['weights'], normed = True)

    n = norm_kde(n, 10.)
    b0 = 0.5 * (b[1:] + b[:-1])

    n, b, _ = plt.hist(b0, bins = b, weights = n)

    plt.xlabel('Lens Mass $(M_\odot)$', fontsize=fontsize1, labelpad=10)
    plt.xlim(xliml, 2)
    plt.ylabel('Relative probability', fontsize=fontsize1, labelpad=10)
    plt.xticks(fontsize=fontsize2)
    plt.yticks(fontsize=fontsize2)

#    plt.xscale('log')

    ##########
    # Calculate 3-sigma boundaries for mass limits.
    ##########
    sig1_hi = 0.682689
    sig1_lo = 1.0 - sig1_hi
    sig_med = 0.5
    sig2_hi = 0.9545
    sig2_lo = 1.0 - sig2_hi
    sig3_hi = 0.9973
    sig3_lo = 1.0 - sig3_hi

    quantiles = [sig3_lo, sig2_lo, sig1_lo, sig_med, sig1_hi, sig2_hi, sig3_hi]

    mass_quants = model_fitter.weighted_quantile(tab['mL'], quantiles,
                                                 sample_weight=tab['weights'])

    for qq in range(len(quantiles)):
        print( 'Mass at {0:.1f}% quantiles:  M = {1:5.2f}'.format(quantiles[qq]*100, mass_quants[qq]))

    ax = plt.axis()
    # plot median and +/- 3 sigma
    plt.axvline(mass_quants[0], color='k', linestyle='--')
    plt.axvline(mass_quants[3], color='k', linestyle='--')
    plt.axvline(mass_quants[-1], color='k', linestyle='--')
    plt.show()
    plt.savefig(paper_dir + 'ob150029_mass_posterior.pdf')
    plt.savefig(paper_dir + 'ob150029_mass_posterior.png')

    return

def plot_ob150211_mass_posterior():
    """
    # Lines are median, +/- 3 sigma.
    Line is MAP solution, global/mode 1 (identical).
    """

    fontsize1 = 18
    fontsize2 = 14

    tab = Table.read(pspl_ast_phot['ob150211'] + '.fits')
    smy = Table.read(pspl_ast_phot['ob150211'] + 'summary.fits')
    print(smy['MAP mL'])
    mmax = tab['mL'].max()
    mmin = tab['mL'].min()

    plt.figure(1)
    plt.clf()
    plt.subplots_adjust(bottom = 0.15)
    xliml = 0.9 * mmin
    xlimu = 1.1 * mmax
    bins = np.logspace(np.log10(xliml), np.log10(xlimu), 500)

    n, b = np.histogram(tab['mL'], bins = bins,
                        weights = tab['weights'], normed = True)

    n = norm_kde(n, 10.)
    b0 = 0.5 * (b[1:] + b[:-1])

    n, b, _ = plt.hist(b0, bins = b, weights = n)

    plt.xlabel('Lens Mass $(M_\odot)$', fontsize=fontsize1, labelpad=10)
    plt.xlim(xliml, xlimu)
    plt.ylabel('Relative probability', fontsize=fontsize1, labelpad=10)
    plt.xticks(fontsize=fontsize2)
    plt.yticks(fontsize=fontsize2)

    plt.xscale('log')


#    ##########
#    # Calculate 3-sigma boundaries for mass limits.
#    ##########
#
#    sig1_hi = 0.682689
#    sig1_lo = 1.0 - sig1_hi
#    sig_med = 0.5
#    sig2_hi = 0.9545
#    sig2_lo = 1.0 - sig2_hi
#    sig3_hi = 0.9973
#    sig3_lo = 1.0 - sig3_hi
#
#    quantiles = [sig3_lo, sig2_lo, sig1_lo, sig_med, sig1_hi, sig2_hi, sig3_hi]
#
#    mass_quants = model_fitter.weighted_quantile(tab['mL'], quantiles,
#                                                 sample_weight=tab['weights'])
#
#    for qq in range(len(quantiles)):
#        print( 'Mass at {0:.1f}% quantiles:  M = {1:5.2f}'.format(quantiles[qq]*100, mass_quants[qq]))
#
#    ax = plt.axis()
#    # plot median and +/- 3 sigma
#    plt.axvline(mass_quants[0], color='k', linestyle='--')
#    plt.axvline(mass_quants[3], color='k', linestyle='--')
#    plt.axvline(mass_quants[-1], color='k', linestyle='--')
#    plt.show()

    plt.axvline(smy['MAP mL'][0], color='k', linestyle='-', lw = 3)
    plt.show()

    plt.savefig(paper_dir + 'ob150211_mass_posterior.pdf')
    plt.savefig(paper_dir + 'ob150211_mass_posterior.png')

    return

def plot_ob150029_mass_posterior():
    """
    Lines are median, +/- 3 sigma.
    """

    fontsize1 = 18
    fontsize2 = 14

    tab = Table.read('/g/lu/scratch/jlu/work/microlens/OB150029/a_2019_04_19/notes/4_fit_phot_astrom_parallax/aa_.fits')
    mmax = tab['mL'].max()
    mmin = tab['mL'].min()

    plt.figure(1)
    plt.clf()
    plt.subplots_adjust(bottom = 0.15)
    xliml = 0.9 * mmin
    xlimu = 1.1 * mmax
#    bins = np.logspace(np.log10(xliml), np.log10(xlimu), 500)
    bins = np.linspace(xliml, xlimu, 500)

    n, b = np.histogram(tab['mL'], bins = bins,
                        weights = tab['weights'], normed = True)

    n = norm_kde(n, 10.)
    b0 = 0.5 * (b[1:] + b[:-1])

    n, b, _ = plt.hist(b0, bins = b, weights = n)

    plt.xlabel('Lens Mass $(M_\odot)$', fontsize=fontsize1, labelpad=10)
    plt.xlim(xliml, 2)
    plt.ylabel('Relative probability', fontsize=fontsize1, labelpad=10)
    plt.xticks(fontsize=fontsize2)
    plt.yticks(fontsize=fontsize2)

#    plt.xscale('log')

    ##########
    # Calculate 3-sigma boundaries for mass limits.
    ##########
    sig1_hi = 0.682689
    sig1_lo = 1.0 - sig1_hi
    sig_med = 0.5
    sig2_hi = 0.9545
    sig2_lo = 1.0 - sig2_hi
    sig3_hi = 0.9973
    sig3_lo = 1.0 - sig3_hi

    quantiles = [sig3_lo, sig2_lo, sig1_lo, sig_med, sig1_hi, sig2_hi, sig3_hi]

    mass_quants = model_fitter.weighted_quantile(tab['mL'], quantiles,
                                                 sample_weight=tab['weights'])

    for qq in range(len(quantiles)):
        print( 'Mass at {0:.1f}% quantiles:  M = {1:5.2f}'.format(quantiles[qq]*100, mass_quants[qq]))

    ax = plt.axis()
    # plot median and +/- 3 sigma
    plt.axvline(mass_quants[0], color='k', linestyle='--')
    plt.axvline(mass_quants[3], color='k', linestyle='--')
    plt.axvline(mass_quants[-1], color='k', linestyle='--')
    plt.show()
    plt.savefig(paper_dir + 'ob150029_mass_posterior.pdf')
    plt.savefig(paper_dir + 'ob150029_mass_posterior.png')

    return

def plot_ob150211_mass_posterior():
    """
    # Lines are median, +/- 3 sigma.
    Line is MAP solution, global/mode 1 (identical).
    """

    fontsize1 = 18
    fontsize2 = 14

    tab = Table.read(pspl_ast_phot['ob150211'] + '.fits')
    smy = Table.read(pspl_ast_phot['ob150211'] + 'summary.fits')
    print(smy['MAP mL'])
    mmax = tab['mL'].max()
    mmin = tab['mL'].min()

    plt.figure(1)
    plt.clf()
    plt.subplots_adjust(bottom = 0.15)
    xliml = 0.9 * mmin
    xlimu = 1.1 * mmax
    bins = np.logspace(np.log10(xliml), np.log10(xlimu), 500)

    n, b = np.histogram(tab['mL'], bins = bins,
                        weights = tab['weights'], normed = True)

    n = norm_kde(n, 10.)
    b0 = 0.5 * (b[1:] + b[:-1])

    n, b, _ = plt.hist(b0, bins = b, weights = n)

    plt.xlabel('Lens Mass $(M_\odot)$', fontsize=fontsize1, labelpad=10)
    plt.xlim(xliml, xlimu)
    plt.ylabel('Relative probability', fontsize=fontsize1, labelpad=10)
    plt.xticks(fontsize=fontsize2)
    plt.yticks(fontsize=fontsize2)

    plt.xscale('log')


#    ##########
#    # Calculate 3-sigma boundaries for mass limits.
#    ##########
#
#    sig1_hi = 0.682689
#    sig1_lo = 1.0 - sig1_hi
#    sig_med = 0.5
#    sig2_hi = 0.9545
#    sig2_lo = 1.0 - sig2_hi
#    sig3_hi = 0.9973
#    sig3_lo = 1.0 - sig3_hi
#
#    quantiles = [sig3_lo, sig2_lo, sig1_lo, sig_med, sig1_hi, sig2_hi, sig3_hi]
#
#    mass_quants = model_fitter.weighted_quantile(tab['mL'], quantiles,
#                                                 sample_weight=tab['weights'])
#
#    for qq in range(len(quantiles)):
#        print( 'Mass at {0:.1f}% quantiles:  M = {1:5.2f}'.format(quantiles[qq]*100, mass_quants[qq]))
#
#    ax = plt.axis()
#    # plot median and +/- 3 sigma
#    plt.axvline(mass_quants[0], color='k', linestyle='--')
#    plt.axvline(mass_quants[3], color='k', linestyle='--')
#    plt.axvline(mass_quants[-1], color='k', linestyle='--')
#    plt.show()

    plt.axvline(smy['MAP mL'][0], color='k', linestyle='-', lw = 3)
    plt.show()

    plt.savefig(paper_dir + 'ob150211_mass_posterior.pdf')
    plt.savefig(paper_dir + 'ob150211_mass_posterior.png')

    return

def plot_ob150029_mass_posterior():
    """
    Lines are median, +/- 3 sigma.
    """

    fontsize1 = 18
    fontsize2 = 14

    tab = Table.read('/g/lu/scratch/jlu/work/microlens/OB150029/a_2019_04_19/notes/4_fit_phot_astrom_parallax/aa_.fits')
    mmax = tab['mL'].max()
    mmin = tab['mL'].min()

    plt.figure(1)
    plt.clf()
    plt.subplots_adjust(bottom = 0.15)
    xliml = 0.9 * mmin
    xlimu = 1.1 * mmax
#    bins = np.logspace(np.log10(xliml), np.log10(xlimu), 500)
    bins = np.linspace(xliml, xlimu, 500)

    n, b = np.histogram(tab['mL'], bins = bins,
                        weights = tab['weights'], normed = True)

    n = norm_kde(n, 10.)
    b0 = 0.5 * (b[1:] + b[:-1])

    n, b, _ = plt.hist(b0, bins = b, weights = n)

    plt.xlabel('Lens Mass $(M_\odot)$', fontsize=fontsize1, labelpad=10)
    plt.xlim(xliml, 2)
    plt.ylabel('Relative probability', fontsize=fontsize1, labelpad=10)
    plt.xticks(fontsize=fontsize2)
    plt.yticks(fontsize=fontsize2)

#    plt.xscale('log')

    ##########
    # Calculate 3-sigma boundaries for mass limits.
    ##########
    sig1_hi = 0.682689
    sig1_lo = 1.0 - sig1_hi
    sig_med = 0.5
    sig2_hi = 0.9545
    sig2_lo = 1.0 - sig2_hi
    sig3_hi = 0.9973
    sig3_lo = 1.0 - sig3_hi

    quantiles = [sig3_lo, sig2_lo, sig1_lo, sig_med, sig1_hi, sig2_hi, sig3_hi]

    mass_quants = model_fitter.weighted_quantile(tab['mL'], quantiles,
                                                 sample_weight=tab['weights'])

    for qq in range(len(quantiles)):
        print( 'Mass at {0:.1f}% quantiles:  M = {1:5.2f}'.format(quantiles[qq]*100, mass_quants[qq]))

    ax = plt.axis()
    # plot median and +/- 3 sigma
    plt.axvline(mass_quants[0], color='k', linestyle='--')
    plt.axvline(mass_quants[3], color='k', linestyle='--')
    plt.axvline(mass_quants[-1], color='k', linestyle='--')
    plt.show()
    plt.savefig(paper_dir + 'ob150029_mass_posterior.pdf')
    plt.savefig(paper_dir + 'ob150029_mass_posterior.png')

    return

def plot_ob150211_posterior_tE_piE_phot_astrom():
    span=None
    levels=None
    color='gray'
    plot_datapoints=False
    plot_density=True
    plot_contours=True
    no_fill_contours=False
    fill_contours=True
    contour_kwargs=None
    contourf_kwargs=None
    data_kwargs=None

    try:
        str_type = types.StringTypes
        float_type = types.FloatType
        int_type = types.IntType
    except:
        str_type = str
        float_type = float
        int_type = int

    """
    Basically the _hist2d function from dynesty, but with a few mods I made.
    https://github.com/joshspeagle/dynesty/blob/master/dynesty/plotting.py
    """

    tab = Table.read('/u/jlu/work/microlens/OB150211/a_2019_05_04/notes/4_fit_phot_astrom_parallax/bb_.fits')

    x = tab['tE']
    y = (tab['piE_E']**2 + tab['piE_N']**2)**0.5
    weights = tab['weights']

    fig = plt.figure(1)
    plt.clf()
    ax = plt.gca()

    # Determine plotting bounds.
    data = [x, y]
    if span is None:
        span = [0.999999426697 for i in range(2)]
    span = list(span)
    if len(span) != 2:
        raise ValueError("Dimension mismatch between samples and span.")
    for i, _ in enumerate(span):
        try:
            xmin, xmax = span[i]
        except:
            q = [0.5 - 0.5 * span[i], 0.5 + 0.5 * span[i]]
            span[i] = dyutil.quantile(data[i], q, weights=weights)

    # The default "sigma" contour levels.
    if levels is None:
        levels = 1.0 - np.exp(-0.5 * np.arange(0.5, 2.1, 0.5) ** 2)

    # Color map for the density plot, over-plotted to indicate the
    # density of the points near the center.
    density_cmap = LinearSegmentedColormap.from_list(
        "density_cmap", [color, (1, 1, 1, 0)])

    # Color map used to hide the points at the high density areas.
    white_cmap = LinearSegmentedColormap.from_list(
        "white_cmap", [(1, 1, 1), (1, 1, 1)], N=2)

    # This "color map" is the list of colors for the contour levels if the
    # contours are filled.
    rgba_color = colorConverter.to_rgba(color)
    contour_cmap = [list(rgba_color) for l in levels] + [rgba_color]
    for i, l in enumerate(levels):
        contour_cmap[i][-1] *= float(i) / (len(levels)+1)

    xmax = x.max()
    xmin = x.min()

    ymax = y.max()
    ymin = y.min()

    xliml = 0.9 * xmin
    xlimu = 1.1 * xmax
    xbins = np.linspace(xliml, xlimu, 500)

    yliml = 0.9 * ymin
    ylimu = 1.1 * ymax
    ybins = np.linspace(yliml, ylimu, 500)

    # We'll make the 2D histogram to directly estimate the density.
    H, X, Y = np.histogram2d(x.flatten(), y.flatten(), bins=[xbins, ybins],
                             range=list(map(np.sort, span)),
                             weights=weights)
    # Smooth the results.
    H = norm_kde(H, [3, 3])

    # Compute the density levels.
    Hflat = H.flatten()
    inds = np.argsort(Hflat)[::-1]
    Hflat = Hflat[inds]
    sm = np.cumsum(Hflat)
    sm /= sm[-1]
    V = np.empty(len(levels))
    for i, v0 in enumerate(levels):
        try:
            V[i] = Hflat[sm <= v0][-1]
        except:
            V[i] = Hflat[0]
    V.sort()
    m = (np.diff(V) == 0)
    if np.any(m) and plot_contours:
        logging.warning("Too few points to create valid contours.")
        logging.warning("Make xnbin or ynbin bigger!!!!!!!")
    while np.any(m):
        V[np.where(m)[0][0]] *= 1.0 - 1e-4
        m = (np.diff(V) == 0)
    V.sort()

    # Compute the bin centers.
    X1, Y1 = 0.5 * (X[1:] + X[:-1]), 0.5 * (Y[1:] + Y[:-1])

    # Extend the array for the sake of the contours at the plot edges.
    H2 = H.min() + np.zeros((H.shape[0] + 4, H.shape[1] + 4))
    H2[2:-2, 2:-2] = H
    H2[2:-2, 1] = H[:, 0]
    H2[2:-2, -2] = H[:, -1]
    H2[1, 2:-2] = H[0]
    H2[-2, 2:-2] = H[-1]
    H2[1, 1] = H[0, 0]
    H2[1, -2] = H[0, -1]
    H2[-2, 1] = H[-1, 0]
    H2[-2, -2] = H[-1, -1]
    X2 = np.concatenate([X1[0] + np.array([-2, -1]) * np.diff(X1[:2]), X1,
                         X1[-1] + np.array([1, 2]) * np.diff(X1[-2:])])
    Y2 = np.concatenate([Y1[0] + np.array([-2, -1]) * np.diff(Y1[:2]), Y1,
                         Y1[-1] + np.array([1, 2]) * np.diff(Y1[-2:])])

    # Plot the data points.
    if plot_datapoints:
        if data_kwargs is None:
            data_kwargs = dict()
        data_kwargs["color"] = data_kwargs.get("color", color)
        data_kwargs["ms"] = data_kwargs.get("ms", 2.0)
        data_kwargs["mec"] = data_kwargs.get("mec", "none")
        data_kwargs["alpha"] = data_kwargs.get("alpha", 0.1)
        ax.plot(x, y, "o", zorder=-1, rasterized=True, **data_kwargs)

    # Plot the base fill to hide the densest data points.
    if (plot_contours or plot_density) and not no_fill_contours:
        ax.contourf(X2, Y2, H2.T, [V.min(), H.max()],
                    cmap=white_cmap, antialiased=False)

    if plot_contours and fill_contours:
        if contourf_kwargs is None:
            contourf_kwargs = dict()
        contourf_kwargs["colors"] = contourf_kwargs.get("colors", contour_cmap)
        contourf_kwargs["antialiased"] = contourf_kwargs.get("antialiased",
                                                             False)
        ax.contourf(X2, Y2, H2.T, np.concatenate([[0], V, [H.max()*(1+1e-4)]]),
                    **contourf_kwargs)

    # Plot the density map. This can't be plotted at the same time as the
    # contour fills.
    elif plot_density:
        ax.pcolor(X, Y, H.max() - H.T, cmap=density_cmap)

    # Plot the contour edge colors.
    if plot_contours:
        if contour_kwargs is None:
            contour_kwargs = dict()
        contour_kwargs["colors"] = contour_kwargs.get("colors", color)
        ax.contour(X2, Y2, H2.T, V, **contour_kwargs)

    ax.set_xlabel('$t_E$ (days)')
    ax.set_ylabel('$\pi_E$')

    ax.set_xlim(132, 144)
    ax.set_ylim(0, 0.04)

    plt.show()

    return

def plot_ob150211_posterior_deltac_piE_phot_astrom():
    span=None
    levels=None
    color='gray'
    plot_datapoints=False
    plot_density=True
    plot_contours=True
    no_fill_contours=False
    fill_contours=True
    contour_kwargs=None
    contourf_kwargs=None
    data_kwargs=None

    try:
        str_type = types.StringTypes
        float_type = types.FloatType
        int_type = types.IntType
    except:
        str_type = str
        float_type = float
        int_type = int

    """
    Basically the _hist2d function from dynesty, but with a few mods I made.
    https://github.com/joshspeagle/dynesty/blob/master/dynesty/plotting.py
    """

    tab = Table.read('/u/jlu/work/microlens/OB150211/a_2019_05_04/notes/4_fit_phot_astrom_parallax/bb_.fits')

    x = tab['tE']
    y = (tab['piE_E']**2 + tab['piE_N']**2)**0.5
    weights = tab['weights']

    fig = plt.figure(1)
    plt.clf()
    ax = plt.gca()

    # Determine plotting bounds.
    data = [x, y]
    if span is None:
        span = [0.999999426697 for i in range(2)]
    span = list(span)
    if len(span) != 2:
        raise ValueError("Dimension mismatch between samples and span.")
    for i, _ in enumerate(span):
        try:
            xmin, xmax = span[i]
        except:
            q = [0.5 - 0.5 * span[i], 0.5 + 0.5 * span[i]]
            span[i] = dyutil.quantile(data[i], q, weights=weights)

    # The default "sigma" contour levels.
    if levels is None:
        levels = 1.0 - np.exp(-0.5 * np.arange(0.5, 2.1, 0.5) ** 2)

    # Color map for the density plot, over-plotted to indicate the
    # density of the points near the center.
    density_cmap = LinearSegmentedColormap.from_list(
        "density_cmap", [color, (1, 1, 1, 0)])

    # Color map used to hide the points at the high density areas.
    white_cmap = LinearSegmentedColormap.from_list(
        "white_cmap", [(1, 1, 1), (1, 1, 1)], N=2)

    # This "color map" is the list of colors for the contour levels if the
    # contours are filled.
    rgba_color = colorConverter.to_rgba(color)
    contour_cmap = [list(rgba_color) for l in levels] + [rgba_color]
    for i, l in enumerate(levels):
        contour_cmap[i][-1] *= float(i) / (len(levels)+1)

    xmax = x.max()
    xmin = x.min()

    ymax = y.max()
    ymin = y.min()

    xliml = 0.9 * xmin
    xlimu = 1.1 * xmax
    xbins = np.linspace(xliml, xlimu, 500)

    yliml = 0.9 * ymin
    ylimu = 1.1 * ymax
    ybins = np.linspace(yliml, ylimu, 500)

    # We'll make the 2D histogram to directly estimate the density.
    H, X, Y = np.histogram2d(x.flatten(), y.flatten(), bins=[xbins, ybins],
                             range=list(map(np.sort, span)),
                             weights=weights)
    # Smooth the results.
    H = norm_kde(H, [3, 3])

    # Compute the density levels.
    Hflat = H.flatten()
    inds = np.argsort(Hflat)[::-1]
    Hflat = Hflat[inds]
    sm = np.cumsum(Hflat)
    sm /= sm[-1]
    V = np.empty(len(levels))
    for i, v0 in enumerate(levels):
        try:
            V[i] = Hflat[sm <= v0][-1]
        except:
            V[i] = Hflat[0]
    V.sort()
    m = (np.diff(V) == 0)
    if np.any(m) and plot_contours:
        logging.warning("Too few points to create valid contours.")
        logging.warning("Make xnbin or ynbin bigger!!!!!!!")
    while np.any(m):
        V[np.where(m)[0][0]] *= 1.0 - 1e-4
        m = (np.diff(V) == 0)
    V.sort()

    # Compute the bin centers.
    X1, Y1 = 0.5 * (X[1:] + X[:-1]), 0.5 * (Y[1:] + Y[:-1])

    # Extend the array for the sake of the contours at the plot edges.
    H2 = H.min() + np.zeros((H.shape[0] + 4, H.shape[1] + 4))
    H2[2:-2, 2:-2] = H
    H2[2:-2, 1] = H[:, 0]
    H2[2:-2, -2] = H[:, -1]
    H2[1, 2:-2] = H[0]
    H2[-2, 2:-2] = H[-1]
    H2[1, 1] = H[0, 0]
    H2[1, -2] = H[0, -1]
    H2[-2, 1] = H[-1, 0]
    H2[-2, -2] = H[-1, -1]
    X2 = np.concatenate([X1[0] + np.array([-2, -1]) * np.diff(X1[:2]), X1,
                         X1[-1] + np.array([1, 2]) * np.diff(X1[-2:])])
    Y2 = np.concatenate([Y1[0] + np.array([-2, -1]) * np.diff(Y1[:2]), Y1,
                         Y1[-1] + np.array([1, 2]) * np.diff(Y1[-2:])])

    # Plot the data points.
    if plot_datapoints:
        if data_kwargs is None:
            data_kwargs = dict()
        data_kwargs["color"] = data_kwargs.get("color", color)
        data_kwargs["ms"] = data_kwargs.get("ms", 2.0)
        data_kwargs["mec"] = data_kwargs.get("mec", "none")
        data_kwargs["alpha"] = data_kwargs.get("alpha", 0.1)
        ax.plot(x, y, "o", zorder=-1, rasterized=True, **data_kwargs)

    # Plot the base fill to hide the densest data points.
    if (plot_contours or plot_density) and not no_fill_contours:
        ax.contourf(X2, Y2, H2.T, [V.min(), H.max()],
                    cmap=white_cmap, antialiased=False)

    if plot_contours and fill_contours:
        if contourf_kwargs is None:
            contourf_kwargs = dict()
        contourf_kwargs["colors"] = contourf_kwargs.get("colors", contour_cmap)
        contourf_kwargs["antialiased"] = contourf_kwargs.get("antialiased",
                                                             False)
        ax.contourf(X2, Y2, H2.T, np.concatenate([[0], V, [H.max()*(1+1e-4)]]),
                    **contourf_kwargs)

    # Plot the density map. This can't be plotted at the same time as the
    # contour fills.
    elif plot_density:
        ax.pcolor(X, Y, H.max() - H.T, cmap=density_cmap)

    # Plot the contour edge colors.
    if plot_contours:
        if contour_kwargs is None:
            contour_kwargs = dict()
        contour_kwargs["colors"] = contour_kwargs.get("colors", color)
        ax.contour(X2, Y2, H2.T, V, **contour_kwargs)

    ax.set_xlabel('$t_E$ (days)')
    ax.set_ylabel('$\pi_E$')

    ax.set_xlim(132, 144)
    ax.set_ylim(0, 0.04)

    plt.show()

    return


def plot_ob150211_posterior_tE_piE_phot_only():
    span=None
    levels=None
    color='gray'
    plot_datapoints=False
    plot_density=True
    plot_contours=True
    no_fill_contours=False
    fill_contours=True
    contour_kwargs=None
    contourf_kwargs=None
    data_kwargs=None

    try:
        str_type = types.StringTypes
        float_type = types.FloatType
        int_type = types.IntType
    except:
        str_type = str
        float_type = float
        int_type = int

    """
    Basically the _hist2d function from dynesty, but with a few mods I made.
    https://github.com/joshspeagle/dynesty/blob/master/dynesty/plotting.py
    """

    tab = Table.read('/u/jlu/work/microlens/OB150211/a_2019_05_04/notes/3_fit_phot_parallax/u0_plusminus/aa_.fits')

    x = tab['tE']
    y = (tab['piE_E']**2 + tab['piE_N']**2)**0.5
    weights = tab['weights']

    fig = plt.figure(2)
    plt.clf()
    ax = plt.gca()

    # Determine plotting bounds.
    data = [x, y]
    if span is None:
        span = [0.999999426697 for i in range(2)]
    span = list(span)
    if len(span) != 2:
        raise ValueError("Dimension mismatch between samples and span.")
    for i, _ in enumerate(span):
        try:
            xmin, xmax = span[i]
        except:
            q = [0.5 - 0.5 * span[i], 0.5 + 0.5 * span[i]]
            span[i] = dyutil.quantile(data[i], q, weights=weights)

    # The default "sigma" contour levels.
    if levels is None:
        levels = 1.0 - np.exp(-0.5 * np.arange(0.5, 2.1, 0.5) ** 2)

    # Color map for the density plot, over-plotted to indicate the
    # density of the points near the center.
    density_cmap = LinearSegmentedColormap.from_list(
        "density_cmap", [color, (1, 1, 1, 0)])

    # Color map used to hide the points at the high density areas.
    white_cmap = LinearSegmentedColormap.from_list(
        "white_cmap", [(1, 1, 1), (1, 1, 1)], N=2)

    # This "color map" is the list of colors for the contour levels if the
    # contours are filled.
    rgba_color = colorConverter.to_rgba(color)
    contour_cmap = [list(rgba_color) for l in levels] + [rgba_color]
    for i, l in enumerate(levels):
        contour_cmap[i][-1] *= float(i) / (len(levels)+1)

    xmax = x.max()
    xmin = x.min()

    ymax = y.max()
    ymin = y.min()

    xliml = 0.9 * xmin
    xlimu = 1.1 * xmax
    xbins = np.linspace(xliml, xlimu, 500)

    yliml = 0.9 * ymin
    ylimu = 1.1 * ymax
    ybins = np.linspace(yliml, ylimu, 500)

    # We'll make the 2D histogram to directly estimate the density.
    H, X, Y = np.histogram2d(x.flatten(), y.flatten(), bins=[xbins, ybins],
                             range=list(map(np.sort, span)),
                             weights=weights)
    # Smooth the results.
    H = norm_kde(H, [3, 3])

    # Compute the density levels.
    Hflat = H.flatten()
    inds = np.argsort(Hflat)[::-1]
    Hflat = Hflat[inds]
    sm = np.cumsum(Hflat)
    sm /= sm[-1]
    V = np.empty(len(levels))
    for i, v0 in enumerate(levels):
        try:
            V[i] = Hflat[sm <= v0][-1]
        except:
            V[i] = Hflat[0]
    V.sort()
    m = (np.diff(V) == 0)
    if np.any(m) and plot_contours:
        logging.warning("Too few points to create valid contours.")
        logging.warning("Make xnbin or ynbin bigger!!!!!!!")
    while np.any(m):
        V[np.where(m)[0][0]] *= 1.0 - 1e-4
        m = (np.diff(V) == 0)
    V.sort()

    # Compute the bin centers.
    X1, Y1 = 0.5 * (X[1:] + X[:-1]), 0.5 * (Y[1:] + Y[:-1])

    # Extend the array for the sake of the contours at the plot edges.
    H2 = H.min() + np.zeros((H.shape[0] + 4, H.shape[1] + 4))
    H2[2:-2, 2:-2] = H
    H2[2:-2, 1] = H[:, 0]
    H2[2:-2, -2] = H[:, -1]
    H2[1, 2:-2] = H[0]
    H2[-2, 2:-2] = H[-1]
    H2[1, 1] = H[0, 0]
    H2[1, -2] = H[0, -1]
    H2[-2, 1] = H[-1, 0]
    H2[-2, -2] = H[-1, -1]
    X2 = np.concatenate([X1[0] + np.array([-2, -1]) * np.diff(X1[:2]), X1,
                         X1[-1] + np.array([1, 2]) * np.diff(X1[-2:])])
    Y2 = np.concatenate([Y1[0] + np.array([-2, -1]) * np.diff(Y1[:2]), Y1,
                         Y1[-1] + np.array([1, 2]) * np.diff(Y1[-2:])])

    # Plot the data points.
    if plot_datapoints:
        if data_kwargs is None:
            data_kwargs = dict()
        data_kwargs["color"] = data_kwargs.get("color", color)
        data_kwargs["ms"] = data_kwargs.get("ms", 2.0)
        data_kwargs["mec"] = data_kwargs.get("mec", "none")
        data_kwargs["alpha"] = data_kwargs.get("alpha", 0.1)
        ax.plot(x, y, "o", zorder=-1, rasterized=True, **data_kwargs)

    # Plot the base fill to hide the densest data points.
    if (plot_contours or plot_density) and not no_fill_contours:
        ax.contourf(X2, Y2, H2.T, [V.min(), H.max()],
                    cmap=white_cmap, antialiased=False)

    if plot_contours and fill_contours:
        if contourf_kwargs is None:
            contourf_kwargs = dict()
        contourf_kwargs["colors"] = contourf_kwargs.get("colors", contour_cmap)
        contourf_kwargs["antialiased"] = contourf_kwargs.get("antialiased",
                                                             False)
        ax.contourf(X2, Y2, H2.T, np.concatenate([[0], V, [H.max()*(1+1e-4)]]),
                    **contourf_kwargs)

    # Plot the density map. This can't be plotted at the same time as the
    # contour fills.
    elif plot_density:
        ax.pcolor(X, Y, H.max() - H.T, cmap=density_cmap)

    # Plot the contour edge colors.
    if plot_contours:
        if contour_kwargs is None:
            contour_kwargs = dict()
        contour_kwargs["colors"] = contour_kwargs.get("colors", color)
        ax.contour(X2, Y2, H2.T, V, **contour_kwargs)

    ax.set_xlabel('$t_E$ (days)')
    ax.set_ylabel('$\pi_E$')

    ax.set_xlim(100, 160)
    ax.set_ylim(0, 0.12)

    plt.show()

    return


def make_ob150211_tab():
# For this one, the negative solutions much favored over the positive solutions
# Which are also super unphysical.

    """
    Table of fit parameters with medians and 1 sigma uncertainties
    """
    # Get the photometry-only data
    mnest_root_phot_only = pspl_phot['ob150211']

    best_arr_phot_only = np.loadtxt(mnest_root_phot_only + 'summary.txt')
    best_phot_only_sol1 = best_arr_phot_only[1][14:21]
    logZ_phot_only_sol1 = best_arr_phot_only[1][28]
    maxL_phot_only_sol1 = best_arr_phot_only[1][29]
    best_phot_only_sol2 = best_arr_phot_only[2][14:21]
    logZ_phot_only_sol2 = best_arr_phot_only[2][28]
    maxL_phot_only_sol2 = best_arr_phot_only[2][29]

    # FIXME: Make these files and make sure they match up with the old indices.
    mnest_tab_phot_only_sol1 = np.loadtxt(mnest_root_phot_only + 'mode0.dat')
    mnest_tab_phot_only_sol2 = np.loadtxt(mnest_root_phot_only + 'mode1.dat')

    # Get the phot+astrom data
    mnest_root_phot_astr = pspl_ast_phot['ob150211']

    best_arr_phot_astr = np.loadtxt(mnest_root_phot_astr + 'summary.txt')
    best_phot_astr_sol1 = best_arr_phot_astr[1][42:63]
    logZ_phot_astr_sol1 = best_arr_phot_astr[1][84]
    maxL_phot_astr_sol1 = best_arr_phot_astr[1][85]
#    best_phot_astr_sol2 = best_arr_phot_astr[2][42:63]
#    logZ_phot_astr_sol2 = best_arr_phot_astr[2][84]
#    maxL_phot_astr_sol2 = best_arr_phot_astr[2][85]

    # FIXME: Make these files and make sure they match up with the old indices.
    mnest_tab_phot_astr_sol1 = np.loadtxt(mnest_root_phot_astr + 'mode0.dat')
    mnest_tab_phot_astr_sol2 = np.loadtxt(mnest_root_phot_astr + 'mode1.dat')

    # Get 1sigma errors
#    phot_only_pars1, phot_only_med_vals1 = model_fitter.quantiles(mnest_tab_phot_only_sol1)
    phot_only_pars2, phot_only_med_vals2 = model_fitter.quantiles(mnest_tab_phot_only_sol2)

    phot_astr_pars1, phot_astr_med_vals1 = model_fitter.quantiles(mnest_tab_phot_astr_sol1)
#    phot_astr_pars2, phot_astr_med_vals2 = model_fitter.quantiles(mnest_tab_phot_astr_sol2)

    params_list = ['t0', 'u0_amp', 'tE',
                   'piE_E', 'piE_N', 'b_sff', 'mag_src',
                   'mL', 'xS0_E', 'xS0_N', 'beta',
                   'muL_E', 'muL_N', 'muS_E', 'muS_N',
                   'dL', 'dL_dS', 'dS', 'thetaE', 'muRel_E', 'muRel_N']

    params = {'t0' : [r'$t_0$ (MJD)', '${0:.2f}^{{+{1:.2f}}}_{{-{2:.2f}}}$'],
              'u0_amp' : [r'$u_0$ $^\dagger$', '${0:.3f}^{{+{1:.3f}}}_{{-{2:.3f}}}$'],
              'tE' : [r'$t_E$ (days)$^\dagger$', '${0:.2f}^{{+{1:.2f}}}_{{-{2:.2f}}}$'],
              'piE_E' : [r'$\pi_{E,E}$ $^\dagger$', '${0:.3f}^{{+{1:.3f}}}_{{-{2:.3f}}}$'],
              'piE_N' : [r'$\pi_{E,N}$ $^\dagger$', '${0:.3f}^{{+{1:.3f}}}_{{-{2:.3f}}}$'],
              'b_sff' : [r'$b_{SFF}$', '${0:.2f}^{{+{1:.2f}}}_{{-{2:.2f}}}$'],
              'mag_src' : [r'$I_{OGLE}$ (mag)', '${0:.3f}^{{+{1:.3f}}}_{{-{2:.3f}}}$'],
              'mL' : [r'$M_L (M_\odot$)', '${0:.2f}^{{+{1:.2f}}}_{{-{2:.2f}}}$'],
              'xS0_E' : [r"$x_{S,0,E}$ ($''$)", '${0:.2f}^{{+{1:.2f}}}_{{-{2:.2f}}}$'],
              'xS0_N' : [r"$x_{S,0,N}$ ($''$)", '${0:.2f}^{{+{1:.2f}}}_{{-{2:.2f}}}$'],
              'beta' : [r'$\beta$ (mas)', '${0:.3f}^{{+{1:.3f}}}_{{-{2:.3f}}}$'],
              'muL_E' : [r'$\mu_{L,E}$ (mas/yr)', '${0:.2f}^{{+{1:.2f}}}_{{-{2:.2f}}}$'],
              'muL_N' : [r'$\mu_{L,N}$ (mas/yr)', '${0:.2f}^{{+{1:.2f}}}_{{-{2:.2f}}}$'],
              'muS_E' : [r'$\mu_{S,E}$ (mas/yr)', '${0:.2f}^{{+{1:.2f}}}_{{-{2:.2f}}}$'],
              'muS_N' : [r'$\mu_{S,N}$ (mas/yr)', '${0:.2f}^{{+{1:.2f}}}_{{-{2:.2f}}}$'],
              'dL' : [r'$d_L$ (pc)', '${0:.0f}^{{+{1:.0f}}}_{{-{2:.0f}}}$'],
              'dS' : [r'$d_S$ (pc) $^\dagger$', '${0:.0f}^{{+{1:.0f}}}_{{-{2:.0f}}}$'],
              'dL_dS' : [r'$d_L/d_S$', '${0:.3f}^{{+{1:.3f}}}_{{-{2:.3f}}}$'],
              'thetaE' : [r'$\theta_E$ (mas)$^\dagger$', '${0:.1f}^{{+{1:.1f}}}_{{-{2:.1f}}}$'],
              'muRel_E' : [r'$\mu_{rel,E}$ (mas/yr)$^\dagger$', '${0:.2f}^{{+{1:.2f}}}_{{-{2:.2f}}}$'],
              'muRel_N' : [r'$\mu_{rel,N}$ (mas/yr)$^\dagger$', '${0:.2f}^{{+{1:.2f}}}_{{-{2:.2f}}}$']}

    output = open(paper_dir + 'ob150029_param_fits_table.txt', 'w')
    for pp in params_list:
        p = params[pp]
        if pp in phot_only_pars2:
#            phot_only_sol1 = p[1].format(phot_only_med_vals1[pp][0],
#                                         phot_only_med_vals1[pp][2],
#                                         phot_only_med_vals1[pp][1])
            phot_only_sol2 = p[1].format(phot_only_med_vals2[pp][0],
                                         phot_only_med_vals2[pp][2],
                                         phot_only_med_vals2[pp][1])

        else:
#            phot_only_sol1 = '--'
            phot_only_sol2 = '--'
        if pp in phot_astr_pars1:
            phot_astr_sol1 = p[1].format(phot_astr_med_vals1[pp][0],
                                         phot_astr_med_vals1[pp][2],
                                         phot_astr_med_vals1[pp][1])
#            phot_astr_sol2 = p[1].format(phot_astr_med_vals2[pp][0],
#                                         phot_astr_med_vals2[pp][2],
#                                         phot_astr_med_vals2[pp][1])
        else:
            phot_astr_sol1 = '--'
#            phot_astr_sol2 = '--'

        output.write(p[0] + ' & ' +
                         phot_only_sol1 + ' & ' + phot_astr_sol2 + ' & ' +
                         phot_only_sol2 + ' & ' + phot_astr_sol1 + ' \\\\\n')

    output.write('log$\mathcal{Z}$' + ' & ' +
                     '{:.2f}'.format(logZ_phot_only_sol1) + '& ' +
                     '{:.2f}'.format(logZ_phot_astr_sol2) + ' & ' +
                     '{:.2f}'.format(logZ_phot_only_sol2) + ' & ' +
                     '{:.2f}'.format(logZ_phot_astr_sol1) + ' \\\\\n')

    output.write('log$\mathcal{Z}$' + '& ' + '{:.2f}'.format(logZ_phot_only_sol2) + ' & ' + '{:.2f}'.format(logZ_phot_astr_sol1) + ' \\\\\n')


def make_ob150211_astrom_fit_tab(recalc=False):
    """
    Make a table with only the astrometric + photometric fit solution.
    """
    # Get the phot+astrom data

    mnest_root = pspl_ast_phot['ob150211']
    data = munge_ob150211.getdata()

    if os.path.exists(mnest_root + '_best.fits') and recalc is False:
        _in = open(mnest_root + '_best.fits', 'rb')

        pars1 = pickle.load(_in)
        values1 = pickle.load(_in)
        logZ_sol1 = pickle.load(_in)
        maxL_sol1 = pickle.load(_in)

        pars2 = pickle.load(_in)
        values2 = pickle.load(_in)
        logZ_sol2 = pickle.load(_in)
        maxL_sol2 = pickle.load(_in)

        _in.close()
    else:
        mnest_root = pspl_ast_phot['ob150211']
        data = munge_ob150211.getdata()

        mfit = model_fitter.PSPL_parallax_Solver(data, outputfiles_basename=mnest_root)

        # We also need to fetch the logZ and maxL... pull from the summary plot.
        # But are these the same solutions?
        best_arr = np.loadtxt(mnest_root + 'summary.txt')
        best_sol1 = best_arr[1][42:63]
        logZ_sol1 = best_arr[1][84]
        maxL_sol1 = best_arr[1][85]
        best_sol2 = best_arr[2][42:63]
        logZ_sol2 = best_arr[2][84]
        maxL_sol2 = best_arr[2][85]

        # FIXME: Make these files and make sure they match up with the old indices.
        mnest_tab_list = mfit.load_mnest_modes()
        mnest_tab_sol1 = mnest_tab_list[0]
        mnest_tab_sol2 = mnest_tab_list[1]

        # Get 1sigma errors
        pars1, values1 = model_fitter.quantiles(mnest_tab_sol1, sigma=1)
        pars2, values2 = model_fitter.quantiles(mnest_tab_sol2, sigma=1)

        # Save to a pickle file for easy reloading.
        _out = open(mnest_root + '_best.fits', 'wb')
        pickle.dump(pars1, _out)
        pickle.dump(values1, _out)
        pickle.dump(logZ_sol1, _out)
        pickle.dump(maxL_sol1, _out)
        pickle.dump(pars2, _out)
        pickle.dump(values2, _out)
        pickle.dump(logZ_sol2, _out)
        pickle.dump(maxL_sol2, _out)
        _out.close()

    # In the order we want them in the table. Use an empty param to indicate
    # break between fit parameters and unfit parameters.
    params_list = ['t0', 'b_sff', 'mag_src',
                   'mL', 'dL', 'dL_dS', 'beta',
                   'muL_E', 'muL_N', 'muS_E', 'muS_N',
                   'piE_E', 'piE_N', 'xS0_E', 'xS0_N', '',
                   'tE', 'dS', 'thetaE', 'u0_amp', 'muRel_E', 'muRel_N']

    params = {'t0' : [r'$t_0$ (MJD)', '${0:.2f}^{{+{1:.2f}}}_{{-{2:.2f}}}$'],
              'u0_amp' : [r'$u_0$ ', '${0:.3f}^{{+{1:.3f}}}_{{-{2:.3f}}}$'],
              'tE' : [r'$t_E$ (days)', '${0:.1f}^{{+{1:.1f}}}_{{-{2:.1f}}}$'],
              'piE_E' : [r'$\pi_{E,E}$ ', '${0:.3f}^{{+{1:.3f}}}_{{-{2:.3f}}}$'],
              'piE_N' : [r'$\pi_{E,N}$ ', '${0:.3f}^{{+{1:.3f}}}_{{-{2:.3f}}}$'],
              'b_sff' : [r'$b_{SFF}$', '${0:.3f}^{{+{1:.3f}}}_{{-{2:.3f}}}$'],
              'mag_src' : [r'$I_{OGLE}$ (mag)', '${0:.4f}^{{+{1:.4f}}}_{{-{2:.4f}}}$'],
              'mL' : [r'$M_L (M_\odot$)', '${0:.1f}^{{+{1:.1f}}}_{{-{2:.1f}}}$'],
              'xS0_E' : [r"$x_{S,0,E}$ ($''$)", '${0:.4f}^{{+{1:.4f}}}_{{-{2:.4f}}}$'],
              'xS0_N' : [r"$x_{S,0,N}$ ($''$)", '${0:.4f}^{{+{1:.4f}}}_{{-{2:.4f}}}$'],
              'beta' : [r'$\beta$ (mas)', '${0:.2f}^{{+{1:.2f}}}_{{-{2:.2f}}}$'],
              'muL_E' : [r'$\mu_{L,E}$ (mas/yr)', '${0:.2f}^{{+{1:.2f}}}_{{-{2:.2f}}}$'],
              'muL_N' : [r'$\mu_{L,N}$ (mas/yr)', '${0:.2f}^{{+{1:.2f}}}_{{-{2:.2f}}}$'],
              'muS_E' : [r'$\mu_{S,E}$ (mas/yr)', '${0:.2f}^{{+{1:.2f}}}_{{-{2:.2f}}}$'],
              'muS_N' : [r'$\mu_{S,N}$ (mas/yr)', '${0:.2f}^{{+{1:.2f}}}_{{-{2:.2f}}}$'],
              'dL' : [r'$d_L$ (pc)', '${0:.0f}^{{+{1:.0f}}}_{{-{2:.0f}}}$'],
              'dS' : [r'$d_S$ (pc) ', '${0:.0f}^{{+{1:.0f}}}_{{-{2:.0f}}}$'],
              'dL_dS' : [r'$d_L/d_S$', '${0:.3f}^{{+{1:.3f}}}_{{-{2:.3f}}}$'],
              'thetaE' : [r'$\theta_E$ (mas)', '${0:.2f}^{{+{1:.2f}}}_{{-{2:.2f}}}$'],
              'muRel_E' : [r'$\mu_{rel,E}$ (mas/yr)', '${0:.2f}^{{+{1:.2f}}}_{{-{2:.2f}}}$'],
              'muRel_N' : [r'$\mu_{rel,N}$ (mas/yr)', '${0:.2f}^{{+{1:.2f}}}_{{-{2:.2f}}}$']}

    # We need a string description of each parameter.
    desc = {'t0'      : r'Time of closest approach.',
            'u0_amp'  : r'Closest approach in $\theta_E$ units.',
            'tE'      : r'Einstein crossing time.',
            'piE_E'   : r'Microlensing parallax in the $\alpha^*$ direction.',
            'piE_N'   : r'Microlensing parallax in the $\delta$ direction.',
            'b_sff'   : r'The source flux fraction in the OGLE aperture, unlensed.',
            'mag_src' : r'OGLE I-band magnitude of the unlensed source.',
            'mL'      : r'Mass of the lens.',
            'xS0_E'   : r'Relative $\alpha^*$ source position at $t_0$.',
            'xS0_N'   : r'Relative $\delta$ source positions at $t_0$.',
            'beta'    : r'Closest angular approach distance.',
            'muL_E'   : r'Proper motion of the lens in the $\alpha^*$ direction',
            'muL_N'   : r'Proper motion of the lens in the $\delta$ direction',
            'muS_E'   : r'Proper motion of the source in the $\alpha^*$ direction',
            'muS_N'   : r'Proper motion of the source in the $\delta$ direction',
            'dL'      : r'Distance to the lens.',
            'dS'      : r'Distance to the source.',
            'dL_dS'   : r'Distance ratio of lens to source.',
            'thetaE'  : r'Angular einstein radius',
            'muRel_E' : r'Relative source-lens proper motion in the $\alpha^*$ direction.',
            'muRel_N' : r'Relative source-lens proper motion in the $\delta$ direction.'
            }

    output = open(paper_dir + 'ob150211_params_ast_phot.txt', 'w')
    output.write('Fit & & & \\\\n')
    output.write('\hline\n')
    for pp in params_list:
        # Check if we should switch to derived parameters when we
        # encounter a '' in the parameters list.
        if pp == '':
            output.write('\hline\n')
            output.write('Derived & & & \\\\\n')
            output.write('\hline\n')
            continue

        p = params[pp]


        if pp in pars1:
            sol1 = p[1].format(values1[pp][0],
                               values1[pp][2],
                               values1[pp][1])
            sol2 = p[1].format(values2[pp][0],
                               values2[pp][2],
                               values2[pp][1])
        else:
            sol1 = '--'
            sol2 = '--'

        output.write(p[0] + ' & ' + sol2 + ' & ' + sol1 + ' & ' + desc[pp] + ' \\\\\n')

    output.write('log$\mathcal{Z}$' + ' & ' +
                     '{:.2f}'.format(logZ_sol2) + ' & ' +
                     '{:.2f}'.format(logZ_sol1) + ' & Local Evidence  \\\\\n')

    output.close()

def make_ob150211_map_fit_tab():
    """
    Make a table with only the astrometric + photometric fit solution.
    Using the MAP solution.
    """
    # Get the phot+astrom data

    smy = Table.read(pspl_ast_phot['ob150211'] + 'summary.fits')
    smy_err = Table.read(pspl_ast_phot_err['ob150211'] + 'summary.fits')
    print(smy.keys())

    # In the order we want them in the table. Use an empty param to indicate
    # break between fit parameters and unfit parameters.
    params_list = ['t0', 'b_sff', 'mag_src',
                   'mL', 'dL', 'dL_dS', 'beta',
                   'muL_E', 'muL_N', 'muS_E', 'muS_N',
                   'piE_E', 'piE_N', 'xS0_E', 'xS0_N', 'add_err', '',
                   'tE', 'dS', 'thetaE', 'u0_amp',
                   'muRel_E', 'muRel_N', 'logZ']

    params = {'t0' : [r'$t_0$ (MJD)', '${0:.2f}$'],
              'u0_amp' : [r'$u_0$ ', '${0:.3f}$'],
              'tE' : [r'$t_E$ (days)', '${0:.1f}$'],
              'piE_E' : [r'$\pi_{E,E}$ ', '${0:.3f}$'],
              'piE_N' : [r'$\pi_{E,N}$ ', '${0:.3f}$'],
              'b_sff' : [r'$b_{SFF}$', '${0:.3f}$'],
              'mag_src' : [r'$I_{OGLE}$ (mag)', '${0:.4f}$'],
              'mL' : [r'$M_L (M_\odot$)', '${0:.1f}$'],
              'xS0_E' : [r"$x_{S,0,E}$ ($''$)", '${0:.4f}$'],
              'xS0_N' : [r"$x_{S,0,N}$ ($''$)", '${0:.4f}$'],
              'beta' : [r'$\beta$ (mas)', '${0:.2f}$'],
              'muL_E' : [r'$\mu_{L,E}$ (mas/yr)', '${0:.2f}$'],
              'muL_N' : [r'$\mu_{L,N}$ (mas/yr)', '${0:.2f}$'],
              'muS_E' : [r'$\mu_{S,E}$ (mas/yr)', '${0:.2f}$'],
              'muS_N' : [r'$\mu_{S,N}$ (mas/yr)', '${0:.2f}$'],
              'dL' : [r'$d_L$ (pc)', '${0:.0f}$'],
              'dS' : [r'$d_S$ (pc) ', '${0:.0f}$'],
              'dL_dS' : [r'$d_L/d_S$', '${0:.3f}$'],
              'thetaE' : [r'$\theta_E$ (mas)', '${0:.2f}$'],
              'muRel_E' : [r'$\mu_{rel,E}$ (mas/yr)', '${0:.2f}$'],
              'muRel_N' : [r'$\mu_{rel,N}$ (mas/yr)', '${0:.2f}$'],
              'add_err' : [r'$\varepsilon$ (mag)', '${0:.4f}$'],
              'logZ' : [r'log$\mathcal{Z}$', '${0:.1f}$']}

    output = open(paper_dir + 'ob150211_params_ast_phot_MAP.txt', 'w')
    output.write('Fit & & \\\\\n')
    output.write('\hline\n')

    output_err = open(paper_dir + 'ob150211_params_ast_phot_err_MAP.txt', 'w')
    output_err.write('Fit & \\\\\n')
    output_err.write('\hline\n')

    for pp in params_list:
        # Check if we should switch to derived parameters when we
        # encounter a '' in the parameters list.
        if pp == '':
            output.write('\hline\n')
            output.write('Derived & & \\\\\n')
            output.write('\hline\n')

            output_err.write('\hline\n')
            output_err.write('Derived & \\\\\n')
            output_err.write('\hline\n')

            continue

        p = params[pp]

        # Fill out the table for NO additive error
        # that has two solutions.
        if (pp != 'add_err') & (pp != 'logZ'):
            sol1 = p[1].format(smy['MAP ' + pp][1])
            sol2 = p[1].format(smy['MAP ' + pp][2])

            output.write(p[0] + ' & ' + sol2 + ' & ' + sol1 + ' \\\\\n')

        # Take care of logZ
        if pp == 'logZ':
            sol1 = p[1].format(smy[pp][1])
            sol2 = p[1].format(smy[pp][2])

            output.write(p[0] + ' & ' + sol2 + ' & ' + sol1 + ' \\\\\n')

        # Fill out the table for WITH additive error
        # that has one solution.

        # Take care of logZ
        if pp == 'logZ':
            sol = p[1].format(smy_err[pp][0])

            output_err.write(p[0] + ' & ' + sol + ' \\\\\n')

        if pp != 'logZ':
            sol = p[1].format(smy_err['MAP_' + pp][0])
            output_err.write(p[0] + ' & ' + sol + ' \\\\\n')

    output.close()


def plot_ob150211_phot():
    data = munge_ob150211.getdata()

    mnest_dir = '/u/jlu/work/microlens/OB150211/a_2019_05_04/notes/3_fit_phot_parallax/u0_plusminus/'
    mnest_root = 'aa_'

    fitter = model_fitter.PSPL_phot_parallax_Solver(data,
                                                    outputfiles_basename = mnest_dir + mnest_root)

    tab_phot_par = fitter.load_mnest_results()

    multinest_plot.plot_phot_fit(data, mnest_dir, mnest_root, outdir=mnest_dir, parallax=True)

    return

def marginalize_tE_piE():
    t = Table.read('/u/casey/scratch/papers/microlens_2019/plot_files/Mock_EWS.fits')

    mas_to_rad = 4.848 * 10**-9

    bh_idx = np.where(t['rem_id_L'] == 103)[0]
    ns_idx = np.where(t['rem_id_L'] == 102)[0]
    wd_idx = np.where((t['rem_id_L'] == 101) |
                      (t['rem_id_L'] == 6))[0]
    st_idx = np.where(t['rem_id_L'] == 0)[0]

    # start with a rectangular Figure
    plt.figure(15, figsize=(8, 8))
    plt.clf()

    minpiE = 0.003
    maxpiE = 5
    mintE = 0.8
    maxtE = 300

    norm = matplotlib.colors.Normalize(np.log10(np.min(t['mu_rel'])), np.log10(np.max(t['mu_rel'])))
    plt.set_cmap('inferno_r')

    # the scatter plot:
    plt.scatter(t['t_E'][st_idx], t['pi_E'][st_idx]/mas_to_rad,
                      alpha = 0.4, label = 'Star', marker = 's', s = 1,
                      c = np.log10(t['mu_rel'][st_idx]), norm = norm)
    plt.scatter(t['t_E'][wd_idx], t['pi_E'][wd_idx]/mas_to_rad,
                      alpha = 0.4, label = 'WD', marker = 'P',
                      c = np.log10(t['mu_rel'][wd_idx]), norm = norm)
    plt.scatter(t['t_E'][ns_idx], t['pi_E'][ns_idx]/mas_to_rad,
                      alpha = 0.4, label = 'NS', marker = 'v',
                      c = np.log10(t['mu_rel'][ns_idx]), norm = norm)
    plt.scatter(t['t_E'][bh_idx], t['pi_E'][bh_idx]/mas_to_rad,
                      alpha = 0.4, label = 'BH',
                      c = np.log10(t['mu_rel'][bh_idx]), norm = norm)
    plt.xlabel('$t_E$ (days)')
    plt.ylabel('$\pi_E$')
    plt.xscale('log')
    plt.yscale('log')
    plt.legend()
    plt.xlim(mintE, maxtE)
    plt.ylim(minpiE, maxpiE)
    tEbins = np.logspace(-1, 2.5, 26)
    piEbins = np.logspace(-4, 1, 26)

    plt.colorbar()

    plt.show()
#    plt.savefig('margin_tE_vs_piE.png')

    return

def make_all_comparison_plots():
    data_150029 = munge_ob150029.getdata()
    data_150211 = munge_ob150211.getdata()

    fit_ob150211_phot_only = model_fitter.PSPL_phot_parallax_Solver(data_150211,
                                                                    outputfiles_basename = '/u/jlu/work/microlens/OB150211/model_fits/3_fit_phot_parallax/u0_plusminus/aa_')

    fit_ob150211_phot_astr = model_fitter.PSPL_parallax_Solver(data_150211,
                                                               outputfiles_basename = '/u/jlu/work/microlens/OB150211/model_fits/4_fit_phot_astrom_parallax/bb_')

    fit_ob150029_phot_only = model_fitter.PSPL_phot_parallax_Solver(data_150029,
                                                                    outputfiles_basename = '/u/jlu/work/microlens/OB150029/model_fits/3_fit_phot_parallax/u0_plusminus/cc_')

    fit_ob150029_phot_astr = model_fitter.PSPL_parallax_Solver(data_150029,
                                                               outputfiles_basename = '/u/jlu/work/microlens/OB150029/model_fits/4_fit_phot_astrom_parallax/aa_')

    # Best fit = median
#    fit_ob150211_phot_only.plot_model_and_data_modes()
#    fit_ob150211_phot_astr.plot_model_and_data_modes()
#    fit_ob150029_phot_only.plot_model_and_data_modes()
    fit_ob150029_phot_astr.plot_model_and_data_modes()


def plot_all_photometry_single():
    plot_photometry_single('ob120169')
    plot_photometry_single('ob150029')
    plot_photometry_single('ob150211')
    plot_photometry_single('ob140613')

    return

def plot_photometry_single(target):
    data = munge.getdata(target, use_astrom_phot=True)

    fitter = model_fitter.PSPL_multiphot_astrom_parallax2_Solver(data, outputfiles_basename=pspl_ast_multiphot[target])
    fitter.load_mnest_results()
    fitter.separate_modes()
    par_all = fitter.get_best_fit_modes(def_best='median')
    par = par_all[0]
    mod_all = fitter.get_best_fit_modes_model(def_best='median')
    mod = mod_all[0]

    ##########
    # Make amplification plot
    ##########
    amp1 = 10**((data['mag1'] - par['mag_src1']) / -2.5) - ((1 - par['b_sff1']) / par['b_sff1'])
    amp2 = 10**((data['mag2'] - par['mag_src2']) / -2.5) - ((1 - par['b_sff2']) / par['b_sff2'])

    const = 2.5 / math.log(10)   # 1.08574
    f1 = 10**(data['mag1'] / -2.5)
    f2 = 10**(data['mag2'] / -2.5)
    f_err1 = (1. / const) * data['mag_err1'] * f1
    f_err2 = (1. / const) * data['mag_err2'] * f2

    f_src1 = 10**(par['mag_src1'] / -2.5)
    f_src2 = 10**(par['mag_src2'] / -2.5)

    amp_err1 = f_err1 / f_src1
    amp_err2 = f_err2 / f_src2

    t_min = np.min(np.append(data['t_phot1'], data['t_phot2']))
    t_max = np.max(np.append(data['t_phot1'], data['t_phot2']))
    t_mod = np.arange(t_min, t_max, 1)
    amp_mod = mod.get_amplification(t_mod)
    amp_mod_t_dat1 = mod.get_amplification(data['t_phot1'])
    amp_mod_t_dat2 = mod.get_amplification(data['t_phot2'])

    #plt.close(1)
    fig_a = plt.figure(1, figsize=(6,6))
    fig_a.clf()

    f1 = fig_a.add_axes([0.2, 0.33, 0.75, 0.6])
    f2 = fig_a.add_axes([0.2, 0.13, 0.75, 0.2])

    # Amplification Curve 
    f1.plot(t_mod, amp_mod, 'k-', label='model')
    f1.errorbar(data['t_phot1'], amp1, yerr=amp_err1, color='red',
                   label='OGLE I', fmt='r.', alpha=0.4, zorder=3000)
    f1.errorbar(data['t_phot2'], amp2, yerr=amp_err2, color='blue',
                   label='Keck Kp', fmt='b.', alpha=0.6, zorder=3001)
    f1.set_ylabel('Amplification')
    f1.legend()
    f1.set_title(target.upper())

    # Residuals
    f1.get_shared_x_axes().join(f1, f2)
    f2.errorbar(data['t_phot1'], amp1 - amp_mod_t_dat1,
                    yerr=amp_err1, fmt='r.', alpha=0.2)
    f2.errorbar(data['t_phot2'], amp2 - amp_mod_t_dat2,
                    yerr=amp_err2, fmt='b.', alpha=0.2)
    f2.axhline(0, linestyle='--', color='r')
    f2.set_xlabel('Time (HJD)')
    f2.set_ylabel('Obs - Mod')

    ##########
    # Make magntidue plot
    ##########
    mag_mod1 = mod.get_photometry(t_mod, 0)
    mag_mod2 = mod.get_photometry(t_mod, 1)

    mag_mod_t_dat1 = mod.get_photometry(data['t_phot1'], 0)
    mag_mod_t_dat2 = mod.get_photometry(data['t_phot2'], 1)

    # Calculate the baseline-mag correction factor.
    base1 = par['mag_src1'] - (-2.5 * math.log10( par['b_sff1'] ))
    base2 = par['mag_src2'] - (-2.5 * math.log10( par['b_sff2'] ))

    dbase = base2 - base1
    print(par['mag_src1'], par['mag_src2'])
    print(par['b_sff1'], par['b_sff2'])
    print(base1, base2, dbase)
    
    #plt.close(2)
    fig_b = plt.figure(2, figsize=(6,6))
    fig_b.clf()

    f3 = fig_b.add_axes([0.2, 0.33, 0.75, 0.6])
    f4 = fig_b.add_axes([0.2, 0.13, 0.75, 0.2])

    # light curve
    f3.errorbar(data['t_phot1'], data['mag1'], yerr=data['mag_err1'], color='red',
                     label='OGLE I', fmt='r.', alpha=0.4, zorder=3000)
    f3.errorbar(data['t_phot2'], data['mag2'] - dbase, yerr=data['mag_err2'], color='blue',
                     label='Keck Kp + {0:.1f}'.format(-dbase), fmt='b.', alpha=0.6, zorder=3000)
    f3.invert_yaxis()
    f3.plot(t_mod, mag_mod1, 'r--')
    f3.plot(t_mod, mag_mod2 - dbase, 'b--')
    f3.set_ylabel('Observed Magnitude')
    f3.legend()
    f3.set_title(target.upper())

    # residuals
    f3.get_shared_x_axes().join(f3, f4)
    f4.errorbar(data['t_phot1'], data['mag1'] - mag_mod_t_dat1,
                    yerr=data['mag_err1'], fmt='r.', alpha=0.2)
    f4.errorbar(data['t_phot2'], data['mag2'] - mag_mod_t_dat2,
                    yerr=data['mag_err2'], fmt='b.', alpha=0.2)
    f4.axhline(0, linestyle='--', color='r')
    f4.set_xlabel('Time (HJD)')
    f4.set_ylabel('Obs - Mod')

    return
    
def get_base_photometry():

    fmt = '{0:s}:  Kp = {1:6.3f} +/- {2:6.3f} mag'
    for targ, ep in epochs.items():
        # Fetch the photometry for this target
        data = munge.getdata(targ, use_astrom_phot=True)

        # Photometry 2 is the Kp photometry. Only get the last 4 epochs.
        times = data['t_phot2']
        mags = data['mag2']

        sdx = times.argsort()
        times = times[sdx]
        mags = mags[sdx]

        mags = mags[-4:]

        m_mean = mags.mean()
        m_err = mags.std()
        
        print(fmt.format(targ, m_mean, m_err))

    return
        

def plot_vpd():
    all_targets = epochs.keys()
    all_targets = ['ob150029']
    for target in all_targets:
        ast_data_file = astrom_data[target]

        data = Table.read(ast_data_file)

        # Flip the coordinates to what we see on sky (+x increase to the East)
        # Convert proper motions to mas/yr
        data['x'] *= -1.0
        data['x0'] *= -1.0
        data['vx'] *= -1.0

        data['vx'] *= 1e3 # mas/yr
        data['vy'] *= 1e3 # mas/yr
        data['vxe'] *= 1e3 # mas/yr
        data['vye'] *= 1e3 # mas/yr

        # Trim out junky data:
        # -- stars detected in too few epochs
        # -- stars with too-large proper motion errors.
        verr_max = 0.5
        idx = np.where((data['n_vfit'] > 5) &
                       (data['vxe'] < verr_max) &
                       (data['vxe'] < verr_max))[0]
        fmt = '{0:s}: Trimming out {1:d} of {2:d} junk stars.'
        print(fmt.format(target, len(data) - len(idx), len(data)))
        
        data = data[idx]

        # Find the target. 
        tdx = np.where(data['name'] == target)[0]

        # Plot the proper motion VPD.
        plt.figure(1)
        plt.clf()
        plt.errorbar(data['vx'], data['vy'], xerr=data['vxe'], yerr=data['vye'], fmt='k.')
        plt.errorbar(data['vx'][tdx], data['vy'][tdx], xerr=data['vxe'][tdx], yerr=data['vye'][tdx], fmt='r.', ecolor='red')
        plt.axis('equal')
        plt.gca().invert_xaxis()
        plt.xlabel(r'$\mu_{\alpha^*}$ (mas/yr)')
        plt.xlabel(r'$\mu_{\delta}$ (mas/yr)')
        plt.ylim(-15, 15)
        plt.xlim(15, -15)

    return
        
        
    
    return
