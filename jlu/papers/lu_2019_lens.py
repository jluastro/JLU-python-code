import numpy as np
import pylab as plt
from astropy.table import Table, Column, vstack
from astropy.io import fits
from dynesty import plotting as dyplot
from flystar import starlists
from scipy.interpolate import UnivariateSpline
import scipy.stats
from scipy.optimize import curve_fit, least_squares
from popsycle import ebf
import matplotlib.ticker
import matplotlib.colors
from matplotlib.pylab import cm
from matplotlib.colors import Normalize, LogNorm
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable
from matplotlib.ticker import NullFormatter
import os, shutil
from scipy.ndimage import gaussian_filter as norm_kde
from microlens.jlu import model_fitter, multinest_utils, multinest_plot, munge_ob150211, munge_ob150029, model, munge
import dynesty.utils as dyutil
from astropy.stats import sigma_clipped_stats
from matplotlib.colors import LinearSegmentedColormap, colorConverter
import pdb
import pickle
import math
import copy
import yaml
from scipy.stats import norm, poisson
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
from astropy.coordinates import SkyCoord
from astropy import units as u
from scipy import interpolate

mpl_o = '#ff7f0e'
mpl_b = '#1f77b4'
mpl_g = '#2ca02c'
mpl_r = '#d62728'

ep_ob120169 = ['12jun',   '12jul',   '13apr',   '13jul', '15may05',
               '15jun07', '16may24', '16jul14']

ep_ob140613 = ['15jun07', '15jun28', '16apr17', '16may24', '16aug02',
               '17jun05', '17jul14', '18may11', '18aug16', '19apr17',
               '19apr21os']

ep_ob150029 = ['15jun07', '15jul23', '16may24', '16jul14', '17may21',
               '17jul14', '17jul19', '18aug21', '19apr17']

ep_ob150211 = ['15may05', '15jun07', '15jun28', '15jul23', '16may03',
               '16jul14', '16aug02', '17jun05', '17jun08', '17jul19',
               '18may11', '18aug02', '18aug16']


epochs = {'ob120169': ep_ob120169, 'ob140613': ep_ob140613, 'ob150029': ep_ob150029, 'ob150211': ep_ob150211}

# paper_dir = '/u/jlu/doc/papers/ob150211/'
paper_dir = '/u/jlu/doc/papers/2015_bh_lenses/'
# paper_dir = '/u/casey/scratch/code/JLU-python-code/jlu/papers/'
mlens_dir = '/u/jlu/work/microlens/'

a_date = {'ob120169': '2020_08_18',
          'ob140613': '2020_08_18',
          'ob150029': '2020_08_18',
          'ob150211': '2020_08_18'}

comp_stars = {'ob120169': ['ob120169_L', 'S24_18_0.8'],
              'ob140613': ['S002_15_0.7', 'S001_15_0.9'],
              'ob150029': ['S002_16_0.3', 'S003_16_0.9'],
              'ob150211': ['S001_13_1.3', 'S003_15_1.4']}

astrom_pass = {'ob120169': 'p5',
               'ob140613': 'p5',
               'ob150029': 'p4',
               'ob150211': 'p5'}

astrom_suffix = {'ob120169': '',
                 'ob140613': '_os',
                 'ob150029': '',
                 'ob150211': ''}

a_dir = {}
astrom_data = {}

for targ in a_date:
    a_dir[targ] = mlens_dir + targ.upper() + '/a_' + a_date[targ] + '/'
    astrom_data[targ] = a_dir[targ] + targ + '_astrom_' + astrom_pass[targ] + '_' + a_date[targ] + astrom_suffix[targ] + '.fits'

ogle_phot_all = {'ob120169_none' : a_dir['ob120169'] + 'model_fits/102_fit_phot_parallax/base_d/',
                 'ob120169_add'  : a_dir['ob120169'] + 'model_fits/103_fit_phot_parallax_aerr/base_c/c3_',
                 'ob120169_mult' : a_dir['ob120169'] + 'model_fits/101_fit_phot_parallax_merr/base_a/',
                 'ob120169_gp'   : a_dir['ob120169'] + 'model_fits/201_phot_ogle_gp/base_a/a2_',
                 'ob140613_none' : a_dir['ob140613'] + 'model_fits/102_fit_phot_parallax/base_c/',
                 'ob140613_add'  : a_dir['ob140613'] + 'model_fits/103_fit_phot_parallax_aerr/base_b/',
                 'ob140613_mult' : a_dir['ob140613'] + 'model_fits/101_fit_phot_parallax_merr/base_c/c8_',
                 'ob140613_gp'   : a_dir['ob140613'] + 'model_fits/201_phot_ogle_gp/base_pos/pos3_',
                 'ob150029_none' : a_dir['ob150029'] + 'model_fits/102_fit_phot_parallax/base_b/',
                 'ob150029_add'  : a_dir['ob150029'] + 'model_fits/103_fit_phot_parallax_aerr/base_d/d8_', 
                 'ob150029_mult' : a_dir['ob150029'] + 'model_fits/101_fit_phot_parallax_merr/base_d/',
                 'ob150029_gp'   : a_dir['ob150029'] + 'model_fits/201_phot_ogle_gp/base_a/a3_',
                 'ob150211_none' : a_dir['ob150211'] + 'model_fits/102_fit_phot_parallax/base_a/',
                 'ob150211_add'  : a_dir['ob150211'] + 'model_fits/103_fit_phot_parallax_aerr/base_a/a4_',
                 'ob150211_mult' : a_dir['ob150211'] + 'model_fits/101_fit_phot_parallax_merr/base_d/',
                 'ob150211_gp'   : a_dir['ob150211'] + 'model_fits/201_phot_ogle_gp/base_b/b4_'}
    
    
photom_spitzer = {'ob120169': None,
                  'ob140613': '/g/lu/data/microlens/spitzer/calchi_novati_2015/ob140613_phot_2.txt',
                  'ob150029': '/g/lu/data/microlens/spitzer/calchi_novati_2015/ob150029_phot_2.txt',
                  'ob150211': '/g/lu/data/microlens/spitzer/calchi_novati_2015/ob150211_phot_3.txt'}

# With GP.
pspl_phot = {'ob120169' : ogle_phot_all['ob120169_gp'],
             'ob140613' : ogle_phot_all['ob140613_gp'],
             'ob150029' : ogle_phot_all['ob150029_gp'],
             'ob150211' : ogle_phot_all['ob150211_gp']}

# With GP: Not done running -- need to update.
pspl_ast_multiphot = {'ob120169' : a_dir['ob120169'] + 'model_fits/220_phot_astrom_gp/base_b/b1_',
                      'ob140613' : a_dir['ob140613'] + 'model_fits/220_phot_astro_gp/base_a/a1_',
                      'ob150029' : a_dir['ob150029'] + 'model_fits/220_phot_astrom_gp/base_b/b4_',
                      'ob150211' : a_dir['ob150211'] + 'model_fits/220_phot_astrom_gp/base_a/a1_split_',
                      'ob150211_unsplit' : a_dir['ob150211'] + 'model_fits/220_phot_astrom_gp/base_a/a1_'}

# 0-based... so 0 = first mode (after the global solution).
pspl_ast_multiphot_mode = {'ob120169': 0,
                           'ob140613': 0,
                           'ob150029': 0,
                           'ob150211': 0}

# With GP: Not done running
pspl_multiphot = {'ob120169' : a_dir['ob120169'] + 'model_fits/211_phot_ogle_keck_gp/base_a/a5_',
                  'ob140613' : a_dir['ob140613'] + 'model_fits/213_phot_ogle_keck_gp/base_a/a5_',
                  'ob150029' : a_dir['ob150029'] + 'model_fits/211_phot_ogle_keck_gp/base_a/a4_',
                  'ob150211' : a_dir['ob150211'] + 'model_fits/213_phot_ogle_keck_gp/base_a/a3_'}

ogle_phot = {}
# ogle_phot['ob120169'] = ogle_phot_all['ob120169_add']
# ogle_phot['ob140613'] = ogle_phot_all['ob140613_mult']
# ogle_phot['ob150029'] = ogle_phot_all['ob150029_add']
# ogle_phot['ob150211'] = ogle_phot_all['ob150211_add']

ogle_phot['ob120169'] = ogle_phot_all['ob120169_gp']
ogle_phot['ob140613'] = ogle_phot_all['ob140613_gp']
ogle_phot['ob150029'] = ogle_phot_all['ob150029_gp']
ogle_phot['ob150211'] = ogle_phot_all['ob150211_gp']

popsycle_events = '/u/casey/scratch/papers/microlens_2019/popsycle_rr_files/Mock_EWS_v2_NEW_DELTAM.fits'

keck_phot_2020 = {'kb200101' : mlens_dir + 'KB200101/a_2020_09_10/model_fits/kmtnet_phot_par/a0_',
                  'mb19284' : mlens_dir + '',
                  'ob190017' : mlens_dir + 'OB190017/a_2020_09_10/model_fits/ogle_phot_par/a0_',
                  'ob170019' : mlens_dir + 'OB170019/a_2020_09_10/model_fits/ogle_phot_par/a0_',
                  'ob170095' : mlens_dir + 'OB170095/a_2021_09_18/model_fits/base_a/a0_'}
hst_phot = {'MB09260' : mlens_dir + 'MB09260/a_2021_07_08/model_fits/moa_hst_phot_ast_gp/base_a/a0_',
            'MB10364' :  mlens_dir +'MB10364/a_2021_07_08/model_fits/moa_hst_phot_ast_gp/base_a/a0_',
            'OB110037' : mlens_dir +'OB110037/a_2021_07_08/model_fits/ogle_hst_phot_ast_gp/base_a/a0_',
            'OB110310' : mlens_dir +'OB110310/a_2021_07_08/model_fits/ogle_hst_phot_ast_gp/base_a/a0_',
            'OB110462' : mlens_dir +'OB110462/a_2021_07_08/model_fits/ogle_hst_phot_ast_gp/base_a/a0_'}


def all_paper():
    # plot_images()
    # make_obs_table()
    # calc_base_mag()
    # plot_pos_err()

    # compare_all_linear_motions()

    # separate_modes_all()
    separate_ob150211_modes()

    plot_ob120169_phot_ast()
    plot_ob140613_phot_ast()
    plot_ob150029_phot_ast()
    plot_ob150211_phot_ast()

    # PSPL Fit Tables and Results Values
    # org_solutions_for_table()
    # -- Manually adjust which solutions are positive/negative/best
    # -- and the order to display the solutions in the table functions below.
    table_ob120169_phot_astrom()
    table_ob140613_phot_astrom()
    table_ob150029_phot_astrom()
    table_ob150211_phot_astrom()
    table_ob150211_phot()

    # Parameters and confidence intervals for the results text.
    results_best_params_all()

    # Lens Geometry, velocity plots
    targets = ['ob120169', 'ob140613', 'ob150029', 'ob150211']
    for targ in targets:
        plot_linear_motion(targ)
        plot_lens_geometry(targ, axis_lim_scale=1, vel_scale=0.25)
        calc_velocity(targ)
        plot_trace_corner(targ)
        
    # Mass Posteriors
    plot_ob150211_mass_posterior_modes()
    plot_all_mass_posteriors()

    # Statistics
    calc_bayes_factor()

    # tE vs. piE vs. deltaC plots
    piE_tE_deltac(fit_type='ast')

    # CMDs
    plot_cmds()

    #####
    # OLD BROKEN STUFF
    #####
    # plot_linear_fits()

    # Appendix -- DROP THIS TABLE
    # make_BIC_comparison_table()

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

            if epoch.endswith('os'):
                tint[ee] = np.round(float(hdr['ITIME0']) * float(hdr['COADDS']) / 1e6, 0)

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

def calc_base_mag(plot=True):
    targets = ['ob120169', 'ob140613', 'ob150029', 'ob150211']

    base_mags = {}

    if plot:
        fig, axs = plt.subplots(nrows=1, ncols=4, figsize=(16, 6))
        plt.subplots_adjust(left=0.1)

    for tt in range(len(targets)):
        target = targets[tt]
        
        fitter, data = get_data_and_fitter(pspl_ast_multiphot[target])

        # Figure out the first MJD day of NIRC2 observations for this target.
        keck_first_mjd = np.floor(data['t_phot2'].min())

        # Get out all epochs that are after the first year.
        keck_base_idx = np.where(data['t_phot2'] > (keck_first_mjd + 365))[0]

        # Figure out the peak yar of the OGLE observations for this target.
        ogle_peak_mjd = data['t_phot1'][data['mag1'].argmin()]
        ogle_base_idx = np.where(data['t_phot1'] > (ogle_peak_mjd + 365))[0]

        # Average the photometry.
        I_base = data['mag1'][ogle_base_idx].mean()
        kp_base = data['mag2'][keck_base_idx].mean()

        base_mags[target] = {}
        base_mags[target]['OGLE_I'] = I_base
        base_mags[target]['NIRC2_Kp'] = kp_base


        if plot:
            print('{0:10s} OGLE_I_base = {1:.2f}  Kp_base = {2:.2f}'.format(target, I_base, kp_base))
            axs[tt].errorbar(data['t_phot2'], data['mag2'], yerr=data['mag_err2'])
            axs[tt].axhline(kp_base, linestyle='--')
            axs[tt].set_title(target)
            axs[tt].invert_yaxis()
            axs[tt].set_xlabel('MJD')

            if tt == 0:
                axs[tt].set_ylabel('Kp (mag)')

    return base_mags

def calc_poisson_prob_detection():
    """
    Calculate the probability of finding 0 black holes from 
    a sample of 5 events with $t_E > $ 120 days.
    """
    samp_size = 5
    sim_prob_bh_mean = 0.42
    sim_prob_bh_err = 0.06

    sigma = np.arange(-3, 3.1, 1)
    colors = plt.cm.get_cmap('plasma_r', len(sigma)).colors    # discrete colors
    
    sim_prob_bh = sim_prob_bh_mean + sigma * sim_prob_bh_err

    mu_all = samp_size * sim_prob_bh

    n_bh = np.arange(0, 11)
    
    plt.figure(1)
    for ss in range(len(sigma)):
        prob_dist = poisson(mu_all[ss])
        legend_str = '{0:.0f}$\sigma$'.format(sigma[ss])

        plt.vlines(n_bh + (ss*0.1), 0, prob_dist.pmf(n_bh),
                     colors=colors[ss], linestyles='-', lw=4,
                     label=legend_str)
        
    plt.legend(loc='best', frameon=False)
    plt.xlabel('Number of black holes')
    plt.ylabel('Probability of detection')

    return

    
def calc_date_resolved():
    """
    Calculate 
    """
    targets = ['ob120169', 'ob140613', 'ob150029', 'ob150211']
    
    for tt in range(len(targets)):
        target = targets[tt]
        
        fitter, data = get_data_and_fitter(pspl_ast_multiphot[target])
        stats_ast, data_ast, mod_ast = load_summary_statistics(pspl_ast_multiphot[target])
        tab_list = fitter.load_mnest_modes()
        mode = pspl_ast_multiphot_mode[target]

        # Get the magnitude of muRel for the maximum-likelihood solution.
        muRel = np.hypot( stats_ast['MaxLike_muRel_E'][mode], stats_ast['MaxLike_muRel_N'][mode])

        # Calculate the credicble intervals on muRel
        sigma_vals = np.array([0.682689, 0.9545, 0.9973])
        credi_ints_lo = (1.0 - sigma_vals) / 2.0
        credi_ints_hi = (1.0 + sigma_vals) / 2.0
        credi_ints_med = np.array([0.5])
        credi_ints = np.concatenate([credi_ints_med, credi_ints_lo, credi_ints_hi])

        sumweights = np.sum(tab_list[mode]['weights'])
        weights = tab_list[mode]['weights'] / sumweights
                
        # Calculate median, 1 sigma lo, and 1 sigma hi credible interval.
        muRel_all = np.hypot( tab_list[mode]['muRel_E'], tab_list[mode]['muRel_N'] )
        tmp = model_fitter.weighted_quantile(muRel_all, credi_ints, sample_weight=weights)

        print('')
        print('*** ' + target + ' ***')
        print('Best muRel = {0:.3f} mas/yr'.format(muRel))
        print('    68.3% CI: [{0:.3f} - {1:.3f} mas/yr]'.format(tmp[1], tmp[4]))
        print('    95.5% CI: [{0:.3f} - {1:.3f} mas/yr]'.format(tmp[2], tmp[5]))
        print('    99.7% CI: [{0:.3f} - {1:.3f} mas/yr]'.format(tmp[3], tmp[6]))

        # Figure out when the objects will be resolvable using
        # ~Keck 2 micron diffraction limit.
        dr_resolve = 0.25 * 2.0 * 1e3 / 10.0  # mas

        from astropy.time import Time
        import datetime
        
        t0 = Time(stats_ast['MaxLike_t0'][mode], format='mjd', scale='utc')
        t0_yr = t0.decimalyear

        t0_resolve = (dr_resolve / muRel) + t0_yr

        print('')
        print('Resolvable in {0:.1f} (at Keck 2 micron resolution of {1:.1f} mas)'.format(t0_resolve, dr_resolve))

        # Figure out when the objects will be resolvable using
        # ~Keck 1 micron diffraction limit.
        dr_resolve = 0.25 * 1.0 * 1e3 / 10.0  # mas
        t0_resolve = (dr_resolve / muRel) + t0_yr

        print('Resolvable in {0:.1f} (at Keck 1 micron resolution of {1:.1f} mas)'.format(t0_resolve, dr_resolve))
        
        # Figure out when the objects will be resolvable using
        # ~TMT 1 micron diffraction limit.
        dr_resolve = 0.25 * 1.0 * 1e3 / 30.0  # mas
        t0_resolve = (dr_resolve / muRel) + t0_yr

        print('Resolvable in {0:.1f} (at TMT 1 micron resolution of {1:.1f} mas)'.format(t0_resolve, dr_resolve))
        
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
        ast_dates = np.append(ast_dates, data['t_ast1'].data)
        pho_dates = np.append(pho_dates, data['t_phot'].data)
        ast_per_target[t] = len(data['t_ast1'])
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

            # Trim down to stars within 4" of the target
            tdx = np.where(lis['name'] == target)[0][0]
            r2d = np.hypot(lis['x'] - lis['x'][tdx], lis['y'] - lis['y'][tdx])
            idx = np.where(r2d < 450)[0]

            lis = lis[idx]

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

            if epoch.endswith('os'):
                day = epoch[-4:-2]
            else:
                day = epoch[-2:]
                
            if target == 'ob120169':
                if epoch == '12jun': day = '23'
                if epoch == '12jul': day = '10' 
                if epoch == '13apr': day = '30'
                if epoch == '13jul': day = '15'
                
            epoch_label = '20{0:2s} {1:s}{2:s} {3:2s}'.format(epoch[0:2], epoch[2:3].upper(), epoch[3:5], day)

            # Make a magnitude array for the model curves.
            p_mag = np.arange(kmag.min(), kmag.max(), 0.05)

            # Plot the data
            # if target == 'ob140613' and ee == n_epochs-1:
            #     plt.plot(kmag, perr * scale * 1e3, '.', label=epoch_label + ' (obs)', color=col, ms=2)

            plt.semilogy(p_mag, scale * 1e3 * perr_v_mag(p_mag, *res_lsq2.x), label=epoch_label, color=col)

        plt.legend(fontsize=12, loc='upper left', ncol=2)
        plt.xlabel('Kp (mag)')
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
    
    plt.close('all')
    
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

def separate_modes_all():
    targets = ['ob120169', 'ob140613', 'ob150029', 'ob150211_unsplit']
    
    for targ in targets:
        print(targ.upper() + ':')
        
        # OGLE phot, Keck phot + astrom
        print('  OGLE phot, Keck phot + astrom')
        mod_fit_ast, data_ast = get_data_and_fitter(pspl_ast_multiphot[targ])
        # Force remake of all FITS files. 
        tab_ast = mod_fit_ast.load_mnest_results(remake_fits=True)
        smy_ast = mod_fit_ast.load_mnest_summary(remake_fits=True)
        mod_fit_ast.separate_modes()
        mod_fit_ast.plot_dynesty_style()

        # OGLE phot
        print('  OGLE phot')
        mod_fit_phot, data_phot = get_data_and_fitter(pspl_phot[targ])
        tab_phot = mod_fit_phot.load_mnest_results(remake_fits=True)
        smy_phot = mod_fit_phot.load_mnest_summary(remake_fits=True)
        mod_fit_phot.separate_modes()
        mod_fit_phot.plot_dynesty_style()

        # # OGLE phot, Keck phot
        # print('  OGLE phot, Keck phot')
        # mod_fit_mphot, data_mphot = get_data_and_fitter(pspl_multiphot[targ])
        # mod_fit_mphot.separate_modes()
        # mod_fit_mphot.plot_dynesty_style()

    return

def separate_ob150211_modes():
    """
    The u0 > 0 mode of OB150211 shows two modes that weren't split properly.
    In this code, we will split them 
    """
    mod_fit_ast, data_ast = get_data_and_fitter(pspl_ast_multiphot['ob150211_unsplit'])

    smy = mod_fit_ast.load_mnest_summary()
    tab = mod_fit_ast.load_mnest_modes()

    smy_global = smy[0]
    
    # Trim out the global mode of smy.
    smy = smy[1:]

    # Find the u0 > 0 (maxL) solution
    idx = np.where(smy['Mean_u0_amp'] > 0)[0][0]
    ndx = np.where(smy['Mean_u0_amp'] <= 0)[0][0]

    smy_g = smy[idx]
    tab_g = tab[idx]
    tab_neg = tab[ndx]

    ##########
    # Split criteria:
    ##########
    
    #
    # Add amplitude of muRel to table and summary
    #
    muRel = np.hypot(tab_g['muRel_E'], tab_g['muRel_N'])

    bins_murel = np.arange(0, 7, 0.04)

    plt.figure(1)
    plt.clf()
    n, b, p = plt.hist(muRel, weights=tab_g['weights'], bins=bins_murel)

    # Find the min point between 0-2 mas/yr.
    min_murel_idx = n[b[:-1]<2].argmin()
    min_murel = b[:-1][min_murel_idx] 
    print(f'Minimum muRel = {min_murel:.3f}')

    plt.axvline(min_murel, color='k', linestyle='--', linewidth=2)

    
    # Split the mode along the minimum muRel.
    mode1_mask = muRel > min_murel
    tab_g_hi_murel = tab_g[mode1_mask]
    tab_g_lo_murel = tab_g[~mode1_mask]
    
    plt.figure(2)
    plt.clf()
    plt.hist(muRel[~mode1_mask], weights=tab_g_lo_murel['weights'],
                 bins=bins_murel, color='orange', label='lo')
    plt.hist(muRel[mode1_mask], weights=tab_g_hi_murel['weights'],
                 bins=bins_murel, color='green', label='hi')
    plt.legend()

    # Find the maximum likelihood solution for OB150211... both modes.
    maxL_idx_lo = np.argmax(tab_g_lo_murel['logLike'])
    maxL_idx_hi = np.argmax(tab_g_hi_murel['logLike'])

    print('Low muRel Solution (u0>0):')
    print(tab_g_lo_murel[maxL_idx_lo]['logLike', 'mL'])
    print('High muRel Solution (u0>0):')
    print(tab_g_hi_murel[maxL_idx_hi]['logLike', 'mL'])

    print()
    print('Weights sum:')
    print('lo = {0:.3f}'.format(tab_g_lo_murel['weights'].sum()))
    print('hi = {0:.3f}'.format(tab_g_hi_murel['weights'].sum()))

    log_ev_neg_u0 = np.log(np.sum(tab_neg['weights'])) + smy['logZ'][ndx]
    log_ev_lo_murel = np.log(np.sum(tab_g_lo_murel['weights'])) + smy['logZ'][idx]
    log_ev_hi_murel = np.log(np.sum(tab_g_hi_murel['weights'])) + smy['logZ'][idx]

    print('\n Old evidence:')
    print(smy['logZ'])
    print('\n New evidence:')
    print('neg_murel', log_ev_neg_u0)
    print('pos_lo_murel', log_ev_lo_murel)
    print('pos_hi_murel', log_ev_hi_murel)
    

    ##########
    # Make a new "run" with the proper mode split.
    ##########
    old_output = pspl_ast_multiphot['ob150211_unsplit']
    new_output = old_output + 'split_'

    # copy over the global files.
    import shutil
    shutil.copyfile(old_output + '.txt', new_output + '.txt')
    shutil.copyfile(old_output + '.fits', new_output + '.fits')
    shutil.copyfile(old_output + 'ev.dat', new_output + 'ev.dat')
    shutil.copyfile(old_output + 'live.points', new_output + 'live.points')
    shutil.copyfile(old_output + 'params.yaml', new_output + 'params.yaml')
    shutil.copyfile(old_output + 'phys_live.points', new_output + 'phys_live.points')

    # Add the new modes as separate tables.
    tab = [tab_g_hi_murel, tab_g_lo_murel, tab_neg]

    N_modes = len(tab)
    print(f'N_modes = {N_modes}')
    
    # Make new mode files.
    for mm in range(N_modes):
        tab[mm].write(new_output + f'mode{mm}.fits', overwrite=True)


    # Make the summary table
    smy_add = vstack([smy[idx], smy[idx]])    # two new solutions.
    
    # Replace the values.
    # First, we want the statistics for the following types of solutions.
    sol_types = ['maxl', 'mean', 'map']
    sol_prefix = {'maxl': 'MaxLike_',
                  'mean': 'Mean_',
                  'map': 'MAP_'}

    for sol in sol_types:
        par_hi = mod_fit_ast.calc_best_fit(tab_g_hi_murel, smy_add, s_idx=0, def_best=sol)
        par_lo = mod_fit_ast.calc_best_fit(tab_g_lo_murel, smy_add, s_idx=1, def_best=sol)

        if sol == 'maxl' or sol == 'map':
            best_par_hi = par_hi
            best_par_lo = par_lo
        else:
            best_par_hi = par_hi[0]
            best_par_lo = par_lo[0]
            best_parerr_hi = par_hi[1]
            best_parerr_lo = par_lo[1]

        for param in mod_fit_ast.all_param_names:
            if sol_prefix[sol] + param not in smy_add.colnames:
                smy_add[sol_prefix[sol] + param] = 0.0
                
            smy_add[sol_prefix[sol] + param][0] = best_par_hi[param]
            smy_add[sol_prefix[sol] + param][1] = best_par_lo[param]

            if sol == 'mean':
                smy_add['StDev_' + param][0] = best_parerr_hi[param]
                smy_add['StDev_' + param][1] = best_parerr_lo[param]
                

    smy_add['logZ'][0] = log_ev_hi_murel
    smy_add['logZ'][1] = log_ev_lo_murel

    smy_add['maxlogL'][0] = tab_g_hi_murel['logLike'].max()
    smy_add['maxlogL'][1] = tab_g_lo_murel['logLike'].max()

    # order is global, murel hi + u0 pos, murel lo + u0 pos, u0 neg
    # matches the order of the modes/table.
    smy_new = vstack([smy_global, smy_add, smy[ndx]])

    print('Final Summary Table (with global)')
    print(smy_new['Mean_mL', 'MAP_mL', 'MaxLike_mL', 'logZ', 'maxlogL'])

    smy_new.write(new_output + 'summary.fits', overwrite=True)


    ##########
    # Test new format
    ##########
    mod_fit, data = get_data_and_fitter(pspl_ast_multiphot['ob150211_unsplit'] + 'split_')
    tabs_new = mod_fit.load_mnest_modes()

    plt.figure(3)
    plt.clf()
    bins_mL = np.logspace(-2, 2, 50)

    plt.hist(tabs_new[0]['mL'], weights=tabs_new[0]['weights'], bins=bins_mL, label='hi murel', alpha=0.4)
    plt.hist(tabs_new[1]['mL'], weights=tabs_new[1]['weights'], bins=bins_mL, label='lo murel', alpha=0.4)
    plt.hist(tabs_new[2]['mL'], weights=tabs_new[2]['weights'], bins=bins_mL, label='u0 neg', alpha=0.4)
    plt.xscale('log')
    plt.xlabel('$m_L$ (M$_\odot$)')
    plt.ylabel('Probability')
    plt.legend()

    return

def calc_bayes_factor(target):
    fitter, data = get_data_and_fitter(pspl_ast_multiphot[target])
    stats_ast, data_ast, mod_ast = load_summary_statistics(pspl_ast_multiphot[target])
    tab_list = fitter.load_mnest_modes()
    best_mode = pspl_ast_multiphot_mode[target]

    # Modify the priors to match what is in the run.py
    if target == 'ob150211':
        # Adjust the priors to encompass both possible solutions
        t0_guess = 57225
        fitter.priors['t0'] = model_fitter.make_gen(t0_guess - 100, t0_guess + 100) 
        fitter.priors['u0_amp'] = model_fitter.make_gen(-1.5, 1.5)
        fitter.priors['tE'] = model_fitter.make_gen(1, 2000)
        fitter.priors['piE_E'] = model_fitter.make_gen(-1, 1)
        fitter.priors['piE_N'] = model_fitter.make_gen(-1, 1)

        # Phot Set 1
        mean1, med1, std1 = sigma_clipped_stats(data['mag1'], sigma_lower=2, sigma_upper=4)
        fitter.priors['b_sff1'] = model_fitter.make_gen(0, 1.2)
        fitter.priors['mag_base1'] = model_fitter.make_norm_gen(mean1, 2 * std1)

        # Phot Set 2
        mean2, med2, std2 = sigma_clipped_stats(data['mag2'], sigma_lower=2, sigma_upper=4)
        fitter.priors['b_sff2'] = model_fitter.make_gen(0, 1.0)
        fitter.priors['mag_base2'] = model_fitter.make_norm_gen(mean2, 2 * std2)
    elif target == 'ob120169':
        # Adjust the priors to encompass both possible solutions 
        fitter.priors['t0'] = model_fitter.make_gen(56020 - 100, 56020 + 100) 
        fitter.priors['u0_amp'] = model_fitter.make_gen(-1.5, 1.5)
        fitter.priors['tE'] = model_fitter.make_gen(1, 2000)
        fitter.priors['piE_E'] = model_fitter.make_gen(-1, 1)
        fitter.priors['piE_N'] = model_fitter.make_gen(-1, 1)
        fitter.priors['b_sff1'] = model_fitter.make_gen(0, 1.2)
        fitter.priors['mag_base1'] = model_fitter.make_norm_gen(19.35, 0.1)
        fitter.priors['b_sff2'] = model_fitter.make_gen(0, 1.0)
        fitter.priors['mag_base2'] = model_fitter.make_norm_gen(17.95, 0.1)
    elif target == 'ob140613':
        # Adjust the priors to encompass both possible solutions 
        fitter.priors['t0'] = model_fitter.make_gen(57150 - 100, 57150 + 100) 
        fitter.priors['u0_amp'] = model_fitter.make_gen(-1.0, 1.0)
        fitter.priors['tE'] = model_fitter.make_gen(1, 2000)
        fitter.priors['piE_E'] = model_fitter.make_gen(-1, 1)
        fitter.priors['piE_N'] = model_fitter.make_gen(-1, 1)
        fitter.priors['b_sff1'] = model_fitter.make_gen(0, 1.2)
        fitter.priors['mag_base1'] = model_fitter.make_gen(18, 18.5)
        fitter.priors['b_sff2'] = model_fitter.make_gen(0, 1.0)
        fitter.priors['mag_base2'] = model_fitter.make_gen(14, 14.5)
    elif target == 'ob150029':
        # Adjust the priors
        fitter.priors['t0'] = model_fitter.make_gen(57230 - 100, 57230 + 10) 
        fitter.priors['u0_amp'] = model_fitter.make_gen(-1.5, 1.5)
        fitter.priors['tE'] = model_fitter.make_gen(1, 2000)
        fitter.priors['piE_E'] = model_fitter.make_gen(-1, 1)
        fitter.priors['piE_N'] = model_fitter.make_gen(-1, 1)

        # Phot Set 1
        mean1, med1, std1 = sigma_clipped_stats(data['mag1'], sigma_lower=2, sigma_upper=4)
        fitter.priors['b_sff1'] = model_fitter.make_gen(0, 1.2)
        fitter.priors['mag_base1'] = model_fitter.make_norm_gen(mean1, 2 * std1)

        # Phot Set 2
        mean2, med2, std2 = sigma_clipped_stats(data['mag2'], sigma_lower=2, sigma_upper=4)
        fitter.priors['b_sff2'] = model_fitter.make_gen(0, 1.0)
        fitter.priors['mag_base2'] = model_fitter.make_norm_gen(mean2, 2 * std2)
        

    # Draw samples on the priors.
    N_samps = 10000
    N_params = len(fitter.fitter_param_names)

    cube = np.random.rand(N_samps, N_params)

    prior_samps = {}

    plt.close('all')
    for i, param_name in enumerate(fitter.fitter_param_names):
        prior_samps[param_name] = fitter.priors[param_name].ppf(cube[:, i])

        plt.figure()
        n, b, p = plt.hist(prior_samps[param_name], bins=50, density=True,
                               label='Prior', histtype='step')
        plt.xlabel(param_name)

        for tt in range(len(tab_list)):
            plt.hist(tab_list[tt][param_name], weights=tab_list[tt]['weights'], bins=b,
                         label=f'Obs Mode {tt}', density=True, histtype='step')

        plt.legend()

    # print out the Bayes Factors
    if target == 'ob120160':
        print('Single mode... no Bayes Factor')
        prior_vol = 1.0
        log_BF = None
        
    elif target == 'ob140613' or target == 'ob150029':
        # Same prior volume, so Bayes Factor is just ratio of evidence.
        log_BF = stats_ast['logZ'][0] - stats_ast['logZ'][1:]
        print('Bayes Factor mode 1 / mode N: ', np.exp(log_BF))
        print(stats_ast['logZ', 'MaxLike_logL', 'Mean_logL'])

        prior_vol = np.ones(len(stats_ast), dtype=float) * (1.0 / len(stats_ast))

    elif target == 'ob150211':
        # Calculate muRel amplitude on the prior samples.
        prior_samps['muRel'] = 10**prior_samps['log10_thetaE'] / prior_samps['tE']  # mas/day
        prior_samps['muRel'] *= 365.25  # mas/yr

        # Solution 1:
        muRel_cut = np.hypot(tab_list[0]['muRel_E'], tab_list[0]['muRel_N']).min()
        idx1 = np.where((prior_samps['u0_amp'] > 0) & (prior_samps['muRel'] >= muRel_cut))[0]

        # Solution 2:
        idx2 = np.where((prior_samps['u0_amp'] > 0) & (prior_samps['muRel'] < muRel_cut))[0]

        # Solution 3:
        idx3 = np.where(prior_samps['u0_amp'] <= 0)[0]

        prior_vol1 = len(idx1) / N_samps
        prior_vol2 = len(idx2) / N_samps
        prior_vol3 = len(idx3) / N_samps

        prior_vol = np.array([prior_vol1, prior_vol2, prior_vol3])
        log_prior_vol = np.log(prior_vol)

        print('Prior volumes:')
        print(f'  Mode 1 - u0 > 0, hi muRel: {prior_vol1:.2f}')
        print(f'  Mode 2 - u0 > 0, lo muRel: {prior_vol2:.2f}')
        print(f'  Mode 3 - u0 <= 0:          {prior_vol3:.2f}')

        print('')

        log_BF = stats_ast['logZ'][0] - stats_ast['logZ'][1:]
        log_BF -= log_prior_vol[0] - log_prior_vol[1:]
        BF = np.exp(log_BF)
        print(f'Bayes Factor mode 1 / mode 2: {BF[0]:.3f}')
        print(f'Bayes Factor mode 1 / mode 3: {BF[1]:.3f}')
        print(stats_ast['logZ', 'MaxLike_logL', 'Mean_logL'])

    return prior_vol, log_BF

def plot_ob120169_phot_ast():
    target = 'ob120169'
    mod_fit, data = get_data_and_fitter(pspl_ast_multiphot[target])
    mod_all = mod_fit.get_best_fit_modes_model(def_best = 'median')
    mode = pspl_ast_multiphot_mode[target]
    img_f = '/g/lu/data/microlens/16may24/combo/mag16may24_ob120169_kp.fits'

    inset_kw = {'labelp1': [-0.8, -0.2], 'labelp2': [0.9, 0.2],  # 
                'scalex': [-8, -6], 'scaley': [-2, -2],
                'textx': -7, 'texty': -1.95,
                'padd': 1}
    plot_4panel(data, mod_all[mode], target, 1, img_f, inset_kw) #ref: 2016-05-24
    return

def plot_ob140613_phot_ast():
    target = 'ob140613'
    mod_fit, data = get_data_and_fitter(pspl_ast_multiphot[target])
    mod_all = mod_fit.get_best_fit_modes_model(def_best = 'median')
    mode = pspl_ast_multiphot_mode[target]
    img_f = '/g/lu/data/microlens/18aug16/combo/mag18aug16_ob140613_kp.fits'

    inset_kw = {'labelp1': [-0.8, -0.2], 'labelp2': [0.9, 0.2],
                'scalex': [-8, -6], 'scaley': [-1.5, -1.5],
                'textx': -7, 'texty': -1.45,
                'padd': 1}
    plot_4panel(data, mod_all[mode], target, 6, img_f, inset_kw) #ref: 2018-08-16
    return

def plot_ob150029_phot_ast():
    target = 'ob150029'
    mod_fit, data = get_data_and_fitter(pspl_ast_multiphot[target])
    mod_all = mod_fit.get_best_fit_modes_model(def_best = 'median')
    mode = pspl_ast_multiphot_mode[target]
    img_f = '/g/lu/data/microlens/17jul19/combo/mag17jul19_ob150029_kp.fits'

    inset_kw = {'labelp1': [-0.8, -0.2], 'labelp2': [0.9, 0.2],
                'scalex': [-1, -3], 'scaley': [-1, -1],
                'textx': -2, 'texty': -0.95,
                'padd': 0.2}
    plot_4panel(data, mod_all[mode], target, 6, img_f, inset_kw) #ref: 2017-07-19
    return

def plot_ob150211_phot_ast():
    target = 'ob150211'
    mod_fit, data = get_data_and_fitter(pspl_ast_multiphot[target])
    mod_all = mod_fit.get_best_fit_modes_model(def_best = 'median')
    mode = pspl_ast_multiphot_mode[target]
    img_f = '/g/lu/data/microlens/17jun05/combo/mag17jun05_ob150211_kp.fits'

    inset_kw = {'labelp1': [-0.8, -0.2], 'labelp2': [0.9, 0.2],
                'scalex': [3, 5], 'scaley': [4, 4],
                'textx': 4, 'texty': 3.95,
                'padd': 2}
    plot_4panel(data, mod_all[mode], target, 7, img_f, inset_kw) #ref: 2017-06-05
    return


def plot_4panel(data, mod, target, ref_epoch, img_f, inset_kw):
    '''
    Plots a 2x2 figure of 1) the Keck image @ ref_epoch (which must coincide
    with img_f), 2) magnitude vs time, 3) RA vs time, and 4) DEC vs time,
    where the latter three have a residual to the model.
    The astrometry plots include the lens and unlensed models.
    inset_kw is a dictionary for plotting 1) with the following keywords:
    {'labelp1': [x, y] list of first anchor for the label line,
     'labelp2': [x, y] list of the second anchor,
     'scalex': [x1, x2] list for plotting the pixel scale in mas,
     'scaley': [y1, y2] list for plotting the pixel scale in mas,
               (y1 = y2 plots a flat line),
     'textx': the x coord of the scale text in mas,
     'texty': the y coord of the scale text in mas,
     'padd': the padding to add in mas,
     }.
     '''
    from mpl_toolkits.axes_grid1.inset_locator import (inset_axes, InsetPosition,
                                                      mark_inset)

    # Sample time
    tmax = np.max(np.append(data['t_phot1'], data['t_phot2'])) + 90.0
    t_mod_ast = np.arange(data['t_ast1'].min() - 180.0, tmax, 2)
    t_mod_pho = np.arange(data['t_phot1'].min(), tmax, 2)

    # Get the linear motion curves for the source (includes parallax)
    p_unlens_mod = mod.get_astrometry_unlensed(t_mod_ast)
    p_unlens_mod_at_ast = mod.get_astrometry_unlensed(data['t_ast1'])

    # Get the lensed motion curves for the source
    p_lens_mod = mod.get_astrometry(t_mod_ast)
    p_lens_mod_at_ast = mod.get_astrometry(data['t_ast1'])

    # Get the predicted photometry
    m_lens_mod_gp = mod.get_photometry_with_gp(data['t_phot1'], data['mag1'], data['mag_err1'], filt_index=0, t_pred=t_mod_pho)[0]
    m_lens_mod = mod.get_photometry(t_mod_pho, filt_idx=0)    

    m_lens_mod_at_phot1 = mod.get_photometry(data['t_phot1'], filt_idx=0)
    m_lens_mod_at_phot2 = mod.get_photometry(data['t_phot2'], filt_idx=1)
    m_lens_mod_gp_at_phot1 = mod.get_photometry_with_gp(data['t_phot1'], data['mag1'], data['mag_err1'], filt_index=0)[0]
    m_lens_mod_gp_only_at_phot1 = m_lens_mod_gp_at_phot1 - m_lens_mod_at_phot1

    # Get the observed photometry, de-trended (GP noise removed)
    m_lens_obs1_detrend = data['mag1'] - m_lens_mod_gp_only_at_phot1

    t_mod_all = np.append(t_mod_ast, t_mod_pho)
    # Set the colorbar
    cmap = plt.cm.viridis
    norm = plt.Normalize(vmin=t_mod_all.min(), vmax=t_mod_all.max())
    smap = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    smap.set_array([])

    # Find the closest model date to the ref_epoch and
    # center inset positions on it
    mod_ref_epoch = np.abs(t_mod_ast - data['t_ast1'][ref_epoch]).argmin()
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
    fig = plt.figure(1, figsize = (11,10))
    wpad = 0.14
    hpad = 0.11
    ax_width = 0.37 * 10. / 11.
    ax_height = 0.37

    #####
    # TARGET IMAGE
    #####
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

    #####
    # INSET
    #####
    # Note: the scale of the inset should be the same as the image plot.
    axf = inset_axes(ax1, 1.05, 1)
    #     Model
    axf.scatter(xpos_ins/1e3, ypos_ins/1e3, c=t_mod_ast, cmap=cmap, norm=norm, s=1)
    #     Data
    axf.errorbar((data['xpos1'] - data['xpos1'][ref_epoch])*-1, (data['ypos1'] - data['ypos1'][ref_epoch])*1,
                   xerr = data['xpos_err1']*1, yerr=data['ypos_err1']*1, fmt='.k')
    axf.set_xticks([])
    axf.set_yticks([])
    axf.invert_xaxis()
    axf.set_aspect('equal')

    # Plot the scale in the inset
    axf.plot(np.array(inset_kw['scalex'])/1e3, np.array(inset_kw['scaley'])/1e3, color=line_color)
    axf.text(inset_kw['textx']/1e3, inset_kw['texty']/1e3, '2 mas', color=line_color, fontsize=12,
                 ha='center', va='bottom')

    # Enlarge the lims to create space for the points
    axf.set_ylim(axf.get_ylim()[0] - (inset_kw['padd']/1e3), axf.get_ylim()[1] + (inset_kw['padd']/1e3))
    axf.set_xlim(axf.get_xlim()[0] + (inset_kw['padd']/1e3), axf.get_xlim()[1] - (inset_kw['padd']/1e3))
    axf.set_aspect('equal')

    
    # Manually set the position and relative size of the inset axes within ax1
    ip = InsetPosition(ax1, [0.40, 0.05, 0.55, 0.45])
    axf.set_axes_locator(ip)
    # axins.set_axes_locator(ip)
    
    # Mark the region corresponding to the inset axes on the parental axes
    # and draw lines in grey linking the two axes.
    mark_inset(parent_axes=ax1, inset_axes=axf, loc1=1, loc2=3, fc="none", ec='0.45')

    #####
    # MAGNITUDE VS TIME
    #####
    ax10 = fig.add_axes([1.0 - wpad/2 - ax_width, 1.0 - hpad/2 - 0.75*ax_height, ax_width, 0.75*ax_height])
    ax11 = fig.add_axes([1.0 - wpad/2 - ax_width, 1.0 - hpad/2 - ax_height, ax_width, 0.25*ax_height])
    # ax10.errorbar(data['t_phot1'], data['mag1'], yerr=data['mag_err1'],
    #               fmt='k.', alpha=0.05)
    ax10.errorbar(data['t_phot1'], m_lens_obs1_detrend, yerr=data['mag_err1'],
                  fmt='k.', alpha=0.05)
    ax10.scatter(t_mod_pho, m_lens_mod, c = t_mod_pho, cmap = cmap, norm = norm, s = 1)
    ax10.invert_yaxis()
    ax10.set_ylabel('$m_I$ (mag)')
    ax10.set_aspect('auto', adjustable='box')
    ax10.set_xticks([])
    ax11.axhline(0, color='grey', linestyle='--', zorder=1)
    ax11.errorbar(data['t_phot1'], data['mag1'] - m_lens_mod_at_phot1, yerr=data['mag_err1'],
                  fmt='k.', alpha=0.05, zorder=2)
    ax11.scatter(t_mod_pho, m_lens_mod_gp - m_lens_mod, c = t_mod_pho, cmap = cmap, norm = norm, s = 1, zorder=3)
    ax11.yaxis.set_major_locator(plt.MaxNLocator(2))
    ax11.xaxis.set_major_locator(plt.MultipleLocator(1000))
    ax11.set_xlabel('Time (MJD)')
    ax11.set_ylabel('GP')
    

    # Center the position data and model off the reference epoch
    p_lens_mod -= [data['xpos1'][ref_epoch], data['ypos1'][ref_epoch]]
    p_unlens_mod -= [data['xpos1'][ref_epoch], data['ypos1'][ref_epoch]]
    p_unlens_mod_at_ast -= [data['xpos1'][ref_epoch], data['ypos1'][ref_epoch]]
    data['xpos1'] -= data['xpos1'][ref_epoch]
    data['ypos1'] -= data['ypos1'][ref_epoch]

    #####
    # RA VS TIME
    #####
    ax20 = fig.add_axes([wpad, hpad + 0.25*ax_height, ax_width, 0.75*ax_height])
    ax21 = fig.add_axes([wpad, hpad, ax_width, 0.25*ax_height])
    ax20.errorbar(data['t_ast1'], data['xpos1']*-1e3,
                  yerr=data['xpos_err1']*1e3, fmt='k.', zorder = 1000)
    ax20.scatter(t_mod_ast, p_lens_mod[:, 0]*-1e3, c = t_mod_ast, cmap = cmap, norm = norm, s = 1)
    ax20.plot(t_mod_ast, p_unlens_mod[:, 0]*-1e3, 'r--')
    ax20.get_xaxis().set_visible(False)
    ax20.set_ylabel(r'$\Delta \alpha^*$ (mas)')
    ax20.get_shared_x_axes().join(ax20, ax21)
    ax21.errorbar(data['t_ast1'], (data['xpos1'] - p_unlens_mod_at_ast[:,0]) * -1e3,
                  yerr=data['xpos_err1'] * 1e3, fmt='k.', alpha=1, zorder = 1000)
    ax21.scatter(t_mod_ast, (p_lens_mod[:, 0] - p_unlens_mod[:, 0])*-1e3, c = t_mod_ast, cmap = cmap, norm = norm, s = 1)
    ax21.axhline(0, linestyle='--', color='r')
    ax21.set_xlabel('Time (MJD)')
    ax21.set_ylabel('Res.')

    #####
    # DEC VS TIME
    #####
    ax30 = fig.add_axes([1.0 - wpad/2 - ax_width, hpad + 0.25*ax_height, ax_width, 0.75*ax_height])
    ax31 = fig.add_axes([1.0 - wpad/2 - ax_width, hpad, ax_width, 0.25*ax_height])
    ax30.errorbar(data['t_ast1'], data['ypos1']*1e3,
                  yerr=data['ypos_err1']*1e3, fmt='k.', zorder = 1000)
    ax30.scatter(t_mod_ast, p_lens_mod[:, 1]*1e3, c = t_mod_ast, cmap = cmap, norm = norm, s = 1)
    ax30.plot(t_mod_ast, p_unlens_mod[:, 1]*1e3, 'r--')
    ax30.get_xaxis().set_visible(False)
    ax30.set_ylabel(r'$\Delta \delta$ (mas)')
    ax30.get_shared_x_axes().join(ax30, ax31)
    ax31.errorbar(data['t_ast1'], (data['ypos1'] - p_unlens_mod_at_ast[:, 1]) * 1e3,
                  yerr=data['ypos_err1'] * 1e3, fmt='k.', alpha=1, zorder = 1000)
    ax31.scatter(t_mod_ast, (p_lens_mod[:, 1] - p_unlens_mod[:, 1])*1e3, c = t_mod_ast, cmap = cmap, norm = norm, s = 1)
    ax31.axhline(0, linestyle='--', color='r')
    ax31.set_xlabel('Time (MJD)')
    ax31.set_ylabel('Res.')

    plt.savefig(paper_dir + target + '_phot_astrom.pdf')
    #plt.close(1)

    return


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
    plt.close('all')
    
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
        ax_sky.plot(xmod, ymod, '-', color='grey', zorder=1)
        ax_sky.plot(xmod + xmode, ymod + ymode, '--', color='grey', zorder=1)
        ax_sky.plot(xmod - xmode, ymod - ymode, '--', color='grey', zorder=1)
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



##########
# PREFER the piE_tE() version with contours.
# This one only has stars.... good for public talk. 
##########
def OLD_tE_piE():
    plt.close('all')
    
    t = Table.read(popsycle_events)

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

    plt.scatter(tE_110022, piE_110022, label = 'OB110022',
                    marker = '*', s = 400)
    

    # Load up best-fits: OGLE + Keck photometry and astrometry fits
    targets = ['ob120169', 'ob140613', 'ob150029', 'ob150211']
    piE = {}
    tE = {}

    for targ in targets:
        stats_ast, data_ast, mod_ast = load_summary_statistics(pspl_ast_multiphot[targ])

        # Figure out which solution is the max logL solution.
        mdx = stats_ast['MaxLike_logL'].argmax()
        piE[targ] = np.hypot(stats_ast['MaxLike_piE_E'][mdx], stats_ast['MaxLike_piE_N'][mdx])
        tE[targ] = stats_ast['MaxLike_tE'][mdx]

        plt.scatter(tE[targ], piE[targ], label = targ.upper(),
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

def piE_tE_deltac(fit_type = 'ast'):
    """
    Supports plotting for several different fit solutions:

    fit_type = 'ast'
        Keck + OGLE photometry, Keck astrometry
    fit_type = 'phot'
        OGLE photometry
    fit_type = 'multiphot'
        Keck + OGLE photometry
    """
    if fit_type is 'ast':
        data_dict = pspl_ast_multiphot
    if fit_type is 'phot':
        data_dict = pspl_phot
    if fit_type is 'multiphot':
        data_dict = pspl_multiphot
        

    ##########
    # !!! NOTE: CHOICE OF THE quantiles_2d HAS A LARGE EFFECT 
    # ON THE WAY THIS PLOT LOOKS !!!
    # Plot piE-tE 2D posteriors from OGLE photometry only fits.
    # Also plot PopSyCLE simulations simultaneously.
    ##########
    span = 0.999999426697
    smooth = 0.04
    quantiles_2d = None
    hist2d_kwargs = None
    labels = None
    label_kwargs = None
    show_titles = False 
    title_fmt = ".2f" 
    title_kwargs = None
    
    # Initialize values.
    if label_kwargs is None:
        label_kwargs = dict()
    if title_kwargs is None:
        title_kwargs = dict()
    if hist2d_kwargs is None:
        hist2d_kwargs = dict()

    # Dictionary of dictionaries containing the tE and piE
    # label position for each of the text labels.
    # WARNING: THE 'PHOT' AND 'AST' ARE THE SAME FOR THE NEW
    # TARGETS... THEY ARE JUST PHOT THOUGH.
    label_pos = {'phot': {'ob120169': [150, 0.01],
                          'ob140613': [170, 0.1],
                          'ob150029': [150, 0.2],
                          'ob150211': [35, 0.04],
                          'ob170019': [0, 0.04],
                          'ob170095': [0, 0.04],
                          'ob190017': [0, 0.04],
                          'kb200101': [0, 0]},
                 'ast':  {'ob110022': [17, 0.38],
                          'ob120169': [130, 0.25],
                          'ob140613': [190, 0.09],
                          'ob150029': [30, 0.15],
                          'ob150211': [150, 0.02],
                          'ob170019': [120, 0.045],
                          'ob170095': [30, 0.04],
                          'ob190017': [180, 0.28],
                          'kb200101': [180, 0.016],
                          'mb190284': [300, 0.025]}
                }

    label_pos_ast = {'ob120169': [0.006, 0.06],
                     'ob140613': [0.04, 0.145],
                     'ob150029': [0.02, 0.25],
                     'ob150211': [0.03, 0.012]}

    # These are hard-coded points from previous papers (no full posteriors).
    tE_points = {'ob110022': 61.4}
    tE_err_points = {'ob110022': 5}  # CHECK
    piE_points = {'ob110022': np.hypot(-0.393, -0.071)}
    piE_err_points = {'ob110022': 0.05} # CHECK
    deltac_points = {'ob110022': 2.19 / np.sqrt(8)}
    deltac_err_points = {'ob110022': 1.06 / np.sqrt(8)}  # CHECK
    
    colors = {'ob110022': 'gray',
              'ob120169': 'purple',
              'ob140613': 'red',
              'ob150029': 'darkorange',
              'ob150211': 'black',
              'ob170019': 'blue',
              'ob170095': 'blue',
              'ob190017': 'blue',
              'kb200101': 'blue',
              'MB09260' : 'gray',
              'MB10364' : 'gray',
              'mb190284' : 'gray',
              'OB110037' : 'gray',
              'OB110310' : 'gray',
              'OB110462' : 'red'}

    include = {'ob110022': True,
              'ob120169': True,
              'ob140613': True,
              'ob150029': True,
              'ob150211': True,
              'ob170019': False,
              'ob170095': False,
              'ob190017': False,
              'kb200101': False,
              'MB09260' : False,
              'mb190284' : False,
              'MB10364' : False,
              'OB110037' : False,
              'OB110310' : False,
              'OB110462' : False}

    include_ast = {'ob110022': True,
                   'ob120169': True,
                   'ob140613': True,
                   'ob150029': True,
                   'ob150211': True,
                   'ob170019': False,
                   'ob170095': False,
                   'ob190017': False,
                   'kb200101': False,
                   'MB09260' : False,
                   'mb190284' : False,
                   'MB10364' : False,
                   'OB110037' : False,
                   'OB110310' : False,
                   'OB110462' : False}
        
    use_label = {'ob110022': False,
                 'ob120169': True,
                 'ob140613': True,
                 'ob150029': True,
                 'ob150211': True,
                 'ob170019': False,
                 'ob170095': False,
                 'ob190017': False,
                 'kb200101': False,
                 'MB09260' : False,
                 'mb190284': False,
                 'MB10364' : False,
                 'OB110037': False,
                 'OB110310': False,
                 'OB110462': False}
        

    ast_ulim = {'ob110022': True,
                'ob120169': False,
                'ob140613': False,
                'ob150029': False,
                'ob150211': False,
                'ob170019': False,
                'ob170095': False,
                'ob190017': False,
                'kb200101': False,
                'MB09260' : False,
                'mb190284': False,
                'MB10364' : False,
                'OB110037': True,
                'OB110310': False,
                'OB110462': False}
        
    # Set defaults.
    hist2d_kwargs['alpha'] = hist2d_kwargs.get('alpha', 0.2)
    hist2d_kwargs['levels'] = hist2d_kwargs.get('levels', quantiles_2d)

    ##########
    # Load up the data
    ##########
    targets = ['ob120169', 'ob140613', 'ob150029', 'ob150211'] 
    hst_targets = ['MB09260', 'MB10364', 'OB110037', 'OB110310', 'OB110462']
    phot_targets = ['ob170019', 'ob170095', 'ob190017', 'kb200101', 'ob110022']
    tE = {}
    piE = {}
    theta_E = {}
    deltaC = {}
    weights = {}

    for targ in hst_targets:
        if not include[targ] or targ in tE_points:
            continue
        
        fit_targ, dat_targ = get_data_and_fitter(hst_phot[targ])
        
        stats_targ = calc_summary_statistics(fit_targ)
        samps_targ = fit_targ.load_mnest_modes()

        mode = stats_targ['logZ'].argmax()
        # mode = pspl_ast_multiphot_mode[targ]
        
        # Find the best-fit solution
        samps_targ = samps_targ[mode]
        stats_targ = stats_targ[mode]
        
        param_targ = get_best_fit_from_stats(fit_targ, stats_targ, def_best='median')

        # Add thetaE (from log thetaE)
        if ('log10_thetaE' in samps_targ.colnames) and ('thetaE' not in samps_targ.colnames):
            theta_E[targ] = 10**samps_targ['log10_thetaE']
        else:
            theta_E[targ] = samps_targ['thetaE']

        tE[targ] = samps_targ['tE']
        piE[targ] = np.hypot(samps_targ['piE_E'], samps_targ['piE_N'])
        deltaC[targ] = theta_E[targ] * 2**0.5 / 4.0
        weights[targ] = samps_targ['weights']
        

    for targ in targets:
        if not include[targ] or targ in tE_points:
            continue

        fit_targ, dat_targ = get_data_and_fitter(pspl_ast_multiphot[targ])
        mode = pspl_ast_multiphot_mode[targ]
        
        stats_targ = calc_summary_statistics(fit_targ)
        samps_targ = fit_targ.load_mnest_modes()
        
        # Find the best-fit solution
        samps_targ = samps_targ[mode]
        stats_targ = stats_targ[mode]
        
        param_targ = get_best_fit_from_stats(fit_targ, stats_targ, def_best='median')

        # Add thetaE (from log thetaE)
        if ('log10_thetaE' in samps_targ.colnames) and ('thetaE' not in samps_targ.colnames):
            theta_E[targ] = 10**samps_targ['log10_thetaE']
        else:
            theta_E[targ] = samps_targ['thetaE']

        tE[targ] = samps_targ['tE']
        piE[targ] = np.hypot(samps_targ['piE_E'], samps_targ['piE_N'])
        deltaC[targ] = theta_E[targ] * 2**0.5 / 4.0
        weights[targ] = samps_targ['weights']
        
    for targ in phot_targets:
        if not include[targ] or targ in tE_points:
            continue

        fit_targ, dat_targ = get_data_and_fitter(keck_phot_2020[targ])
        
        stats_targ = calc_summary_statistics(fit_targ)
        samps_targ = fit_targ.load_mnest_modes()

        mode = stats_targ['logZ'].argmax()
        # mode = pspl_ast_multiphot_mode[targ]
        
        # Find the best-fit solution
        samps_targ = samps_targ[mode]
        stats_targ = stats_targ[mode]
        
        param_targ = get_best_fit_from_stats(fit_targ, stats_targ, def_best='median')

        tE[targ] = samps_targ['tE']
        piE[targ] = np.hypot(samps_targ['piE_E'], samps_targ['piE_N'])
        weights[targ] = samps_targ['weights']


    # MB190284 fit results (from Dave Bennett)
    data_tab = '/u/jlu/doc/proposals/hst/cycle28_mid2/mcmc_bsopcnC_3.dat'

    # chi^2 1/t_E t0 umin sep theta eps1=q/(1+q) 1/Tbin dsxdt dsydt t_fix Tstar(=0) pi_E,N piE,E 0 0 0 0 0 0 0 0 0 A0ogleI A2ogleI A0ogleV A2ogleV A0moa2r A2moa2r A0moa2V
    data = Table.read(data_tab, format='ascii.fixed_width_no_header', delimiter=' ')
    data.rename_column('col1', 'chi2')
    data.rename_column('col2', 'tE_inv')
    data.rename_column('col3', 't0')
    data.rename_column('col4', 'u0')
    data.rename_column('col5', 'sep')
    data.rename_column('col6', 'theta')
    data.rename_column('col7', 'eps1')
    data.rename_column('col8', 'Tbin_inv')
    data.rename_column('col9', 'dsxdt')
    data.rename_column('col10', 'dsydt')
    data.rename_column('col11', 't_fix')
    data.rename_column('col12', 'Tstar')
    data.rename_column('col13', 'piEE')
    data.rename_column('col14', 'piEN')
    data['tE'] = 1.0 / data['tE_inv']
    data['piEE'] = data['piEE'].astype('float')
    data['weight'] = np.ones(len(data))
    data['piE'] = np.hypot(data['piEE'], data['piEN'])

    targ = 'mb190284'
    tE[targ] = data['tE']
    piE[targ] = data['piE']
    weights[targ] = data['weight']
    phot_targets.append(targ)
    

    ##########
    # Load up the PopSyCLE simulation points.
    ##########
    popsyc = Table.read(popsycle_events) 
    bh_idx = np.where(popsyc['rem_id_L'] == 103)[0]
    ns_idx = np.where(popsyc['rem_id_L'] == 102)[0]
    wd_idx = np.where(popsyc['rem_id_L'] == 101)[0]
    st_idx = np.where(popsyc['rem_id_L'] == 0)[0]

    # Stores the maximum astrometric shift
    popsyc_delta_arr = np.zeros(len(popsyc))
    
    # Stores the lens-source separation corresponding
    # to the maximum astrometric shift
    popsyc_u_ast_max_arr = np.zeros(len(popsyc))

    # Flux ratio of lens to source (and make it 0 if dark lens)
    popsyc_g_arr = 10**(-0.4 * (popsyc['ubv_i_app_L'] - popsyc['ubv_i_app_S']))
    popsyc_g_arr = np.nan_to_num(popsyc_g_arr)

    # First calculate max astrometric shift using
    # approximation that only works for u0 > 2**0.5
    g = popsyc_g_arr
    u = popsyc['u0']
    numer = 1 + g * (u**2 - u * np.sqrt(u**2 + 4) + 3)
    denom = u**2 + 2 + g * u * np.sqrt(u**2 + 4)
    popsyc['deltaC'] = (u * popsyc['theta_E']/(1 + g)) * (numer/denom)

    # Now fix those that have u0 < 2**0.5.
    idx_fix = np.where(popsyc['u0'] < 2**0.5)[0]
    for i in idx_fix:
        g = popsyc_g_arr[i] 
        thetaE = popsyc['theta_E'][i]
        # Try all values between u0 and sqrt(2) to find max 
        # astrometric shift
        u = np.linspace(popsyc['u0'][i], np.sqrt(2), 100)
        numer = 1 + g * (u**2 - u * np.sqrt(u**2 + 4) + 3)
        denom = u**2 + 2 + g * u * np.sqrt(u**2 + 4)
        delta = (u * thetaE/(1 + g)) * (numer/denom)
        max_idx = np.argmax(delta)
        popsyc['deltaC'][i] = delta[max_idx]


    ##########
    # Plot the piE-tE 2D posteriors from observed targets.
    ##########
    fig = plt.figure(1, figsize=(6,6))
    plt.clf()
    axes = plt.gca()
    plt.subplots_adjust(bottom=0.15)

    sx = smooth
    sy = smooth

    hist2d_kwargs['fill_contours'] = hist2d_kwargs.get('fill_contours', False)
    hist2d_kwargs['plot_contours'] = hist2d_kwargs.get('plot_contours', True)

    for targ in targets + phot_targets + hst_targets:
        if not include[targ]:
            continue

        if targ in tE_points:
            plt.errorbar(tE_points[targ], piE_points[targ],
                             xerr=tE_err_points[targ], yerr=piE_err_points[targ],
                             marker = 's', ms = 4, color=colors[targ])
        else:
            model_fitter.contour2d_alpha(tE[targ], piE[targ], span=[span, span], quantiles_2d=quantiles_2d,
                                         weights=weights[targ], ax=axes, smooth=[sy, sx], color=colors[targ],
                                         **hist2d_kwargs, plot_density=False, sigma_levels=[1, 2])

        if include[targ] and use_label[targ]:
            axes.text(label_pos[fit_type][targ][0], label_pos[fit_type][targ][1],
                          targ.upper(), color=colors[targ])    



    axes.scatter(popsyc['t_E'][st_idx], popsyc['pi_E'][st_idx], 
                 alpha = 0.4, marker = '.', s = 25, 
                 color = 'paleturquoise')
    axes.scatter(popsyc['t_E'][wd_idx], popsyc['pi_E'][wd_idx], 
                 alpha = 0.4, marker = '.', s = 25, 
                 color = 'aqua')
    axes.scatter(popsyc['t_E'][ns_idx], popsyc['pi_E'][ns_idx], 
                 alpha = 0.4, marker = '.', s = 25, 
                 color = 'tab:cyan')
    axes.scatter(popsyc['t_E'][bh_idx], popsyc['pi_E'][bh_idx],
                 alpha = 0.8, marker = '.', s = 25, 
                 color = 'black')

    # Trickery to make the legend darker
    axes.scatter(0.01, 100, 
                 alpha = 0.8, marker = '.', s = 25, 
                 label = 'Star', color = 'paleturquoise')
    axes.scatter(0.01, 100, 
                 alpha = 0.8, marker = '.', s = 25,
                 label = 'WD', color = 'aqua')
    axes.scatter(0.01, 100,
                 alpha = 0.8, marker = '.', s = 25, 
                 label = 'NS', color = 'tab:cyan')
    axes.scatter(0.01, 100,
                 alpha = 0.8, marker = '.', s = 25, 
                 label = 'BH', color = 'black')

    axes.set_xlim(10, 1000)
    axes.set_ylim(0.005, 0.5)
    axes.set_xlabel('$t_E$ (days)')
    axes.set_ylabel('$\pi_E$')
    axes.set_xscale('log')
    axes.set_yscale('log')
    axes.legend(loc=3)
    plt.savefig(paper_dir + 'piE_tE_' + fit_type + '.png')
    plt.show()

    
    ##########
    # delta_c vs. piE
    ##########
    fig = plt.figure(2, figsize=(6,6))
    plt.clf()
    axes = plt.gca()
    plt.subplots_adjust(bottom=0.15)

    # Observed
    for targ in targets + phot_targets + hst_targets:
        if not include_ast[targ]:
            continue

        # Targets with only upper limits to deltaC.
        if ast_ulim[targ]:
            # Check to see if we have posteriors and
            # calculate the 3 sigma upper limits.
            if targ not in tE_points:
                deltac_conf_ints = get_CIs(deltaC[targ], weights[targ])
                piE_conf_ints = get_CIs(piE[targ], weights[targ])
                
                deltac_3sig_hi = deltac_conf_ints[-1]
                piE_median = model_fitter.weighted_quantile(piE[targ], [0.5], sample_weight=weights[targ])[0]
                piE_err = np.array([piE_conf_ints[0:2]]).T
            else:
                deltac_3sig_hi = deltac_points[targ] + 3*deltac_err_points[targ]
                piE_median = piE_points[targ]
                piE_err = piE_err_points[targ]

            axes.errorbar(deltac_3sig_hi, piE_median,
                          xerr = deltac_3sig_hi * 0.25,
                          yerr = piE_err,
                          fmt = 'o', color = colors[targ], markersize = 5,
                          xuplims = True)
        else:
            # Targets with posteriors and contours.
            model_fitter.contour2d_alpha(deltaC[targ], piE[targ], span=[span, span], quantiles_2d=quantiles_2d,
                                             weights=weights[targ], ax=axes, smooth=[sy, sx], color=colors[targ],
                                             **hist2d_kwargs, plot_density=False, sigma_levels=[1, 2])

    
    # Mass Gap Band
    xarr = np.linspace(0.001, 4, 1000)
    axes.fill_between(xarr, xarr*0.18, xarr*0.07, alpha=0.15, color='orange')
    axes.text(0.05, 0.006, 'Mass Gap', rotation=45)

    # PopSyCLE
    axes.scatter(popsyc['deltaC'][st_idx], popsyc['pi_E'][st_idx], 
                  alpha = 0.4, marker = '.', s = 25,
                  c = 'paleturquoise')
    axes.scatter(popsyc['deltaC'][wd_idx], popsyc['pi_E'][wd_idx], 
                  alpha = 0.4, marker = '.', s = 25,
                  c = 'aqua')
    axes.scatter(popsyc['deltaC'][ns_idx], popsyc['pi_E'][ns_idx], 
                  alpha = 0.4, marker = '.', s = 25,
                  c = 'tab:cyan')
    axes.scatter(popsyc['deltaC'][bh_idx], popsyc['pi_E'][bh_idx], 
                  alpha = 0.8, marker = '.', s = 25,
                  c = 'black')

    axes.set_xlabel('$\delta_{c,max}$ (mas)')
    axes.set_ylabel('$\pi_E$')
    axes.set_xscale('log')
    axes.set_yscale('log')
#    axes.set_xlim(0.005, 4)
#    axes.set_ylim(0.009, 0.5)
    axes.set_xlim(0.02, 2)
    axes.set_ylim(0.005, 0.5)
    plt.savefig('piE_deltac_ast.png')
    plt.show()

    return

def shift_vs_piE():
    plt.close('all')
    
    # This assumes blending from source and lens.
    t = Table.read(popsycle_events)

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

    fig = plt.figure(1, figsize=(6, 6))
    plt.clf()
    
    norm = matplotlib.colors.Normalize(np.log10(np.min(t['t_E'])), np.log10(np.max(t['t_E'])))
    plt.subplots_adjust(bottom = 0.15, right = 0.95, top = 0.9)
    plt.set_cmap('inferno_r')
    plt.scatter(t['pi_E'][bh_idx]/mas_to_rad, final_delta_arr[bh_idx],
                alpha = 0.2, c = 'black', label = 'PopSyCLE BH', norm = norm)
    plt.scatter(t['pi_E'][other_idx]/mas_to_rad, final_delta_arr[other_idx],
                alpha = 0.2, c = 'grey', label = 'PopSyCLE Other', s = 2, norm = norm)
    ax = plt.gca()
    
    #############
    # Add the observations
    #############
    targets = ['ob120169', 'ob140613', 'ob150029', 'ob150211']
    deltaC = {}
    tE = {}
    piE = {}
    weights = {}

    for targ in targets:
        fit_targ, dat_targ = get_data_and_fitter(pspl_ast_multiphot[targ])
        mode = pspl_ast_multiphot_mode[targ]
        
        stats_targ = calc_summary_statistics(fit_targ)
        samps_targ = fit_targ.load_mnest_modes()
        
        # Find the best-fit solution
        samps_targ = samps_targ[mode]
        stats_targ = stats_targ[mode]
        
        param_targ = get_best_fit_from_stats(fit_targ, stats_targ, def_best='median')

        # Add thetaE (from log thetaE)
        if ('log10_thetaE' in samps_targ.colnames) and ('thetaE' not in samps_targ.colnames):
            thetaE_samp = 10**samps_targ['log10_thetaE']
        else:
            thetaE_samp = samps_targ['thetaE']

        tE[targ] = samps_targ['tE']
        deltaC[targ] = thetaE_samp * 2**0.5 / 4.0
        piE[targ] = np.hypot(samps_targ['piE_E'], samps_targ['piE_N'])
        weights[targ] = samps_targ['weights']

    # Get ready for some plotting
    span = 0.999999426697
    smooth = 0.04
    quantiles_2d = None
    hist2d_kwargs = None
    labels = None
    label_kwargs = None
    show_titles = False 
    title_fmt = ".2f" 
    title_kwargs = None
    
    # Initialize values.
    if label_kwargs is None:
        label_kwargs = dict()
    if title_kwargs is None:
        title_kwargs = dict()
    if hist2d_kwargs is None:
        hist2d_kwargs = dict()
    
    sx = smooth
    sy = smooth

    hist2d_kwargs['fill_contours'] = hist2d_kwargs.get('fill_contours', False)
    hist2d_kwargs['plot_contours'] = hist2d_kwargs.get('plot_contours', True)

    label_pos = {'ob120169': [0.1, 1.8],
                 'ob140613': [0.15, 0.15],
                 'ob150029': [0.2, 0.4],
                 'ob150211': [0.01, 0.6]}
        
    colors = {'ob120169': 'purple',
              'ob140613': 'red',
              'ob150029': 'darkorange',
              'ob150211': 'black'}

    for targ in targets:
        model_fitter.contour2d_alpha(piE[targ], deltaC[targ], span=[span, span], quantiles_2d=quantiles_2d,
                                 weights=weights[targ], ax=ax, smooth=[sy, sx], color=colors[targ],
                                 **hist2d_kwargs, plot_density=False)

        ax.text(label_pos[targ][0], label_pos[targ][1],
                      targ.upper(), color=colors[targ])    
    
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('$\pi_E$')
    ax.set_ylabel('$\delta_{c,max}$ (mas)')
    ax.set_xlim(5 * 10**-3, 1)
    ax.set_ylim(5 * 10**-2, 4)
    ax.legend(loc='upper left', fontsize=12)
    
    plt.savefig(paper_dir + 'deltac_vs_piE.png')


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

def calc_velocity(target):
    import astropy.coordinates as coord
    import astropy.units as u

    fitter, data = get_data_and_fitter(pspl_ast_multiphot[target])
    stats = calc_summary_statistics(fitter, verbose=False)
    mod_all = fitter.get_best_fit_modes_model(def_best = 'median')
    mode = pspl_ast_multiphot_mode[target]
    bf_mod = mod_all[mode]

    # Hack for old models
    fitter.all_param_names.remove('thetaE')
    fitter.all_param_names.remove('mag_src1')
    fitter.all_param_names.remove('mag_src2')

    bf_mod = fitter.get_best_fit_model()
    
    def print_vel_info(ra, dec, muE, muN, muE_e, muN_e, dist):
        c1 = coord.ICRS(ra=ra*u.degree, dec=data['decL']*u.degree,
                    distance=dist*u.pc,
                    pm_ra_cosdec=muE*u.mas/u.yr,
                    pm_dec=muN*u.mas/u.yr)

        galcen_distance = 8*u.kpc
        pm_en = [muE, muN] * u.mas/u.yr
        v_e, v_n = -(galcen_distance * pm_en).to(u.km/u.s, u.dimensionless_angles())
        ve_e = -(galcen_distance * muE_e * u.mas/u.yr).to(u.km/u.s, u.dimensionless_angles())
        ve_n = -(galcen_distance * muN_e * u.mas/u.yr).to(u.km/u.s, u.dimensionless_angles())

        gal = c1.transform_to(coord.Galactic)
        mu_l = gal.pm_l_cosb
        mu_b = gal.pm_b
        v_l = -(galcen_distance * mu_l).to(u.km/u.s, u.dimensionless_angles())
        v_b = -(galcen_distance * mu_b).to(u.km/u.s, u.dimensionless_angles())

        fmt = '    {0:8s} = {1:8.3f} +/- {2:8.3f}  {3:8s}'

        print('Proper Motion for ' + target + ':')
        print('  Celestial:')
        print(fmt.format('mu_E', muE, muE_e, 'mas/yr'))
        print(fmt.format('mu_N', muN, muN_e, 'mas/yr'))
        print('  Galactic:')
        print(fmt.format('mu_l', mu_l, 0.0, 'mas/yr'))
        print(fmt.format('mu_b', mu_b, 0.0, 'mas/yr'))

        print('Velocity for ' + target + ' at dist={0:.1f}'.format(dist))
        print('  Celestial:')
        print(fmt.format('v_E', v_e, ve_e, 'km/s'))
        print(fmt.format('v_N', v_n, ve_n, 'km/s'))
        print('  Galactic:')
        print(fmt.format('v_l', v_l, 0.0, 'km/s'))
        print(fmt.format('v_b', v_b, 0.0, 'km/s'))

        return
        
    
    # Fetch the lens proper motions. Only for the "best" solution
    # as this is the one we will adopt for the paper.
    dL = bf_mod.dL
    muL_E = bf_mod.muL[0]
    muL_N = bf_mod.muL[1]
    muLe_E = np.diff([stats['lo68_muL_E'][mode], stats['hi68_muL_E'][mode]])[0] / 2.0
    muLe_N = np.diff([stats['lo68_muL_N'][mode], stats['hi68_muL_N'][mode]])[0] / 2.0

    print('\n*** Lens ***')
    print_vel_info(data['raL'], data['decL'], muL_E, muL_N, muLe_E, muLe_N, dL)

    # Fetch the source proper motions. Only for the "best" solution
    # as this is the one we will adopt for the paper.
    dS = bf_mod.dS
    muS_E = bf_mod.muS[0]
    muS_N = bf_mod.muS[1]
    muSe_E = np.diff([stats['lo68_muS_E'][mode], stats['hi68_muS_E'][mode]])[0] / 2.0
    muSe_N = np.diff([stats['lo68_muS_N'][mode], stats['hi68_muS_N'][mode]])[0] / 2.0

    print('\n*** Source ***')
    print_vel_info(data['raL'], data['decL'], muS_E, muS_N, muSe_E, muSe_N, dS)
    
    # Fetch the base magnitudes.
    base_mags = calc_base_mag(plot=False)
    
    # Load up PopSyCLE
    sim = Table.read(popsycle_events)

    # Filter out really faint stuff. CHOOSE NOT TO TRIM becuase we are
    # looking at the whole field of stars, not just the target. 
    # sdx = np.where(sim['ubv_i_app_LSN'] < (base_mags[target]['OGLE_I'] + 0.5))[0]
    # print('Keeping sims: {0:d} of {1:d}'.format(len(sdx), len(sim)))
    # sim = sim[sdx]

    sim_coo = coord.Galactic(l=sim['glon_S']*u.degree, b=sim['glat_S']*u.degree,
                distance=sim['px_S']*u.pc,
                pm_l_cosb=sim['mu_lcosb_S']*u.mas/u.yr,
                pm_b=sim['mu_b_S']*u.mas/u.yr)

    sim_icrs = sim_coo.transform_to(coord.ICRS)
    sim_muS_astar = sim_icrs.pm_ra_cosdec
    sim_muS_delta = sim_icrs.pm_dec
    
    # Load up the data to make a VPD:
    ast = Table.read(data['ast_files'][0])
    
    # Get rid of stuff without proper motion errors.
    idx = np.where((ast['vxe'] > 0) & (ast['vye'] > 0))[0]
    ast = ast[idx]
    
    tdx = np.where(ast['name'] == target)[0][0]
    ast['vx'] *= -1e3   # mas/yr increasing to the East
    ast['vy'] *= 1e3
    ast['vxe'] *= 1e3
    ast['vye'] *= 1e3

    # Shift the observed astrometry into the popsycle reference frame.
    weights = 1.0 / np.hypot(ast['vxe'], ast['vye'])
    vx_obs_mean = np.average(ast['vx'], weights=weights)
    vy_obs_mean = np.average(ast['vy'], weights=weights)
    vx_sim_mean = sim_muS_astar.mean().value
    vy_sim_mean = sim_muS_delta.mean().value

    dvx = vx_sim_mean - vx_obs_mean
    dvy = vy_sim_mean - vy_obs_mean
    print('dvx = {0:6.2f} mas/yr'.format(dvx))
    print('dvy = {0:6.2f} mas/yr'.format(dvy))
    
    ast['vx'] += dvx
    ast['vy'] += dvy
    muL_E += dvx
    muL_N += dvy
    muS_E += dvx
    muS_N += dvy

    plt.close(1)
    plt.figure(1, figsize=(6, 6))
    plt.clf()
    plt.plot(ast['vx'][tdx], ast['vy'][tdx], 'r*', ms=10, label='Src, Keck')
    plt.errorbar([muS_E], [muS_N], xerr=[muSe_E], yerr=[muSe_N],
                     marker='s', color='red', label='Src, Fit', ms=10, lw=2, mfc='none', mew=4)
    plt.errorbar([muL_E], [muL_N], xerr=[muLe_E], yerr=[muLe_N],
                     marker='s', color='black', label='Lens, Fit', ms=10, lw=2, mfc='none', mew=4)
    
    plt.errorbar(ast['vx'], ast['vy'], xerr=ast['vxe'], yerr=ast['vye'], fmt='b.', color='b',
                     label='Other, Keck', alpha=0.3)
    plt.scatter(sim_muS_astar, sim_muS_delta, 
                 alpha = 0.2, marker = '.', s = 10, 
                 color = 'grey', label='PopSyCLE')

    plt.gca().invert_xaxis()

    plt.axis('equal')
    plt.xlim(7, -12)
    plt.ylim(-15, 5)
    plt.xlabel(r'$\mu_{\alpha^*}$ (mas/yr)')
    plt.ylabel(r'$\mu_{\delta}$ (mas/yr)')

    handles, labels = plt.gca().get_legend_handles_labels()
    leg_idx = [0, 3, 4, 1, 2]
    handles = [handles[ii] for ii in leg_idx]
    labels = [labels[ii] for ii in leg_idx]
    plt.legend(handles, labels, fontsize=14, ncol=2, loc='upper center')
    
    plt.title(target.upper())

    plt.savefig(paper_dir + 'pm_vpd_' + target + '.png')

    return

def calc_zp_ob150211():
    # Load up the Keck photometry (on arbitrary flux scale) and the
    # 2MASS photometry. 
    ast = Table.read(data['ast_files'][0])
    tmass = Table.read('/Users/jlu/work/microlens/OB150211/tmass.fits')

    tt_t = np.where(tmass['name'] == 'ob150211')[0][0]
    tt_a = np.where(ast['name'] == 'ob150211')[0][0]

    # Get the Keck stars within 4" radius.
    rad = np.hypot(ast['x'] - ast['x'][tt_a], ast['y'] - ast['y'][tt_a])
    rdx = np.where(rad < 4.)


def plot_cmds():
    plot_cmd_ob150211()
    return

def plot_cmd_ob150211():
    """
    Everything is in
    /Users/jlu/work/microlens/OB150211/a_2019_05_04/notes/7_other_phot.ipynb
    """
    # Read in the Gaia and 2MASS catalogs.
    tmass = Table.read('/Users/jlu/work/microlens/OB150211/tmass.fits')
    gaia = Table.read('/Users/jlu/work/microlens/OB150211/gaia.fits')

    tt_t = np.where(tmass['name'] == 'ob150211')
    tt_g = np.where(gaia['name'] == 'ob150211')

    # Also fetch our best fit and figure out the source and lens/neighbor
    # brightness.
    foo = get_data_fitter_params_models_samples('ob150211', return_mode = 'best')
    fitter = foo[0]
    data = foo[1]
    stats = foo[2]
    params = foo[3]
    models = foo[4]
    sampls = foo[5]
    
    magS_I = params['mag_src1']
    magS_Kp = params['mag_src2']
    magLN_I = magS_I - 2.5 * math.log10((1.0 - params['b_sff1']) / params['b_sff1']**2)
    magLN_Kp = magS_Kp - 2.5 * math.log10((1.0 - params['b_sff2']) / params['b_sff2']**2)

    # Get the baseline magnitude (with all blending)
    magSLN_I = magS_I + 2.5 * math.log10(params['b_sff1'])
    magSLN_Kp = magS_Kp + 2.5 * math.log10(params['b_sff2'])
    
    # Assume J-K color is the same for the lens, source, and baseline.
    jk_color = tmass['Jmag'][tt_t[0][0]] - tmass['Kmag'][tt_t[0][0]]
    magS_J = magS_Kp + jk_color
    magLN_J = magLN_Kp + jk_color
    magSLN_J = magSLN_Kp + jk_color

    print('        {0:5s} {1:5s} {2:5s}    {3:7s} {4:7s}'.format('I', 'J', 'Kp', 'J_2MASS', 'K_2MASS'))
    print('b_SFF:  {0:5.2f} {1:5.2f} {2:5.2f}'.format(params['b_sff1'], -1, params['b_sff2']))
    print('magS:   {0:5.2f} {1:5.2f} {2:5.2f}'.format(magS_I, magS_J, magS_Kp))
    print('magLN:  {0:5.2f} {1:5.2f} {2:5.2f}'.format(magLN_I, magLN_J, magLN_Kp))
    print('magSLN: {0:5.2f} {1:5.2f} {2:5.2f}    {3:7.2f} {4:7.2f}'.format(magSLN_I, magSLN_J, magSLN_Kp,
                                                                               tmass['Jmag'][tt_t[0][0]],
                                                                               tmass['Kmag'][tt_t[0][0]]
                                                                               ))

    plt.close(1)
    plt.figure(1, figsize=(10, 4))
    plt.subplots_adjust(bottom=0.2, left=0.1, right=0.8, wspace=0.4)

    plt.subplot(1, 2, 1)
    plt.plot(tmass['Jmag'] - tmass['Kmag'], tmass['Kmag'], 'k.', alpha=0.5, mec='none')
    # plt.plot(tmass['Jmag'][tt_t] - tmass['Kmag'][tt_t], tmass['Kmag'][tt_t], 'ro', ms=5, label='2MASS @ target')
    plt.plot(magS_J - magS_Kp, magS_Kp, marker='s', color='orange', linestyle='none', ms=5, label='Source')
    plt.plot(magLN_J - magLN_Kp, magLN_Kp, 'bs', ms=5, label='Lens + Neighbors')
    plt.arrow(magLN_J - magLN_Kp, magLN_Kp, 0, 0.5, color='blue',
                  width=0.05, head_width=0.15, head_length=0.3)
    plt.plot(magSLN_J - magSLN_Kp, magSLN_Kp, marker='*', linestyle='none', color='magenta',
                 mec='deeppink', ms=10, label='Total')
    plt.xlim(0, 3)
    plt.gca().invert_yaxis()
    plt.xlabel('J-K (mag)')
    plt.ylabel('K (mag)')
    plt.title('2MASS')
    plt.legend(fontsize=10)

    plt.subplot(1, 2, 2)

    # color code only those sources with significant parallax
    good_par = np.where(np.abs(gaia['parallax']) > (3*gaia['parallax_error']))[0]
    plt.plot(gaia['bp_rp'], gaia['phot_g_mean_mag'], 'k.', ms=8, alpha=0.5, zorder=1, mec='none')
    sc = plt.scatter(gaia['bp_rp'][good_par], gaia['phot_g_mean_mag'][good_par],
                     c=gaia['parallax'][good_par], s=10, zorder=2, edgecolors='none',
                     vmin=-1, vmax=2, cmap=plt.cm.viridis)
    plt.plot(gaia['bp_rp'][tt_g], gaia['phot_g_mean_mag'][tt_g], linestyle='none', marker='*', color='magenta',
                 mec='deeppink', ms=10, zorder=3)
    plt.xlim(0, 6.5)
    plt.gca().invert_yaxis()
    plt.xlabel('G$_{BP}$ - G$_{RP}$ (mag)')
    plt.ylabel('G (mag)')
    plt.title('Gaia')

    fig = plt.gcf()
    cb_ax = fig.add_axes([0.83, 0.13, 0.02, 0.77])
    cbar = fig.colorbar(sc, cax=cb_ax, label='Parallax (mas)')

    plt.savefig(paper_dir + 'ob150211_cmds.png')

    return

def plot_cmd_other3():
    """
    Everything is in
    /Users/jlu/work/microlens/OB150211/a_2019_05_04/notes/7_other_phot.ipynb
    """
    # Read in the Gaia and 2MASS catalogs.
    tmass = Table.read('/Users/jlu/work/microlens/OB150211/tmass.fits')
    gaia = Table.read('/Users/jlu/work/microlens/OB150211/gaia.fits')

    tt_t = np.where(tmass['name'] == 'ob150211')
    tt_g = np.where(gaia['name'] == 'ob150211')

    # Also fetch our best fit and figure out the source and lens/neighbor
    # brightness.
    foo = get_data_fitter_params_models_samples('ob150211', return_mode = 'best')
    fitter = foo[0]
    data = foo[1]
    stats = foo[2]
    params = foo[3]
    models = foo[4]
    sampls = foo[5]
    
    magS_I = params['mag_src1']
    magS_Kp = params['mag_src2']
    magLN_I = magS_I - 2.5 * math.log10((1.0 - params['b_sff1']) / params['b_sff1']**2)
    magLN_Kp = magS_Kp - 2.5 * math.log10((1.0 - params['b_sff2']) / params['b_sff2']**2)

    # Get the baseline magnitude (with all blending)
    magSLN_I = magS_I + 2.5 * math.log10(params['b_sff1'])
    magSLN_Kp = magS_Kp + 2.5 * math.log10(params['b_sff2'])
    
    # Assume J-K color is the same for the lens, source, and baseline.
    jk_color = tmass['Jmag'][tt_t[0][0]] - tmass['Kmag'][tt_t[0][0]]
    magS_J = magS_Kp + jk_color
    magLN_J = magLN_Kp + jk_color
    magSLN_J = magSLN_Kp + jk_color

    print('        {0:5s} {1:5s} {2:5s}    {3:7s} {4:7s}'.format('I', 'J', 'Kp', 'J_2MASS', 'K_2MASS'))
    print('b_SFF:  {0:5.2f} {1:5.2f} {2:5.2f}'.format(params['b_sff1'], -1, params['b_sff2']))
    print('magS:   {0:5.2f} {1:5.2f} {2:5.2f}'.format(magS_I, magS_J, magS_Kp))
    print('magLN:  {0:5.2f} {1:5.2f} {2:5.2f}'.format(magLN_I, magLN_J, magLN_Kp))
    print('magSLN: {0:5.2f} {1:5.2f} {2:5.2f}    {3:7.2f} {4:7.2f}'.format(magSLN_I, magSLN_J, magSLN_Kp,
                                                                               tmass['Jmag'][tt_t[0][0]],
                                                                               tmass['Kmag'][tt_t[0][0]]
                                                                               ))

    plt.close(1)
    plt.figure(1, figsize=(10, 4))
    plt.subplots_adjust(bottom=0.2, left=0.1, right=0.8, wspace=0.4)

    plt.subplot(1, 2, 1)
    plt.plot(tmass['Jmag'] - tmass['Kmag'], tmass['Kmag'], 'k.', alpha=0.5, mec='none')
    plt.plot(tmass['Jmag'][tt_t] - tmass['Kmag'][tt_t], tmass['Kmag'][tt_t], 'ro', ms=5, label='2MASS @ target')
    plt.plot(magS_J - magS_Kp, magS_Kp, 'rs', ms=5, label='Source')
    plt.plot(magLN_J - magLN_Kp, magLN_Kp, 'bs', ms=5, label='Lens + Neighbors')
    plt.plot(magSLN_J - magSLN_Kp, magSLN_Kp, 'g*', ms=10, label='Total')
    plt.xlim(0, 3)
    plt.gca().invert_yaxis()
    plt.xlabel('J-K (mag)')
    plt.ylabel('K (mag)')
    plt.title('2MASS')
    plt.legend(fontsize=10)

    plt.subplot(1, 2, 2)

    # color code only those sources with significant parallax
    good_par = np.where(np.abs(gaia['parallax']) > (3*gaia['parallax_error']))[0]
    plt.plot(gaia['bp_rp'], gaia['phot_g_mean_mag'], 'k.', ms=8, alpha=0.5, zorder=1, mec='none')
    sc = plt.scatter(gaia['bp_rp'][good_par], gaia['phot_g_mean_mag'][good_par],
                     c=gaia['parallax'][good_par], s=10, zorder=2, edgecolors='none',
                     vmin=-1, vmax=2, cmap=plt.cm.viridis)
    plt.plot(gaia['bp_rp'][tt_g], gaia['phot_g_mean_mag'][tt_g], 'ro', ms=5, zorder=3)
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

def run_galaxia_all_targets():
    """
    Must run in the paper directory.
    """
    targets = ['ob120169', 'ob140613', 'ob150029', 'ob150211']

    gal_param_file = paper_dir + 'galaxy_model_params.txt'
    if not os.path.exists(gal_param_file):
        orig_file = '/Users/casey/scratch/papers/hst_sahu_2020/scratch_work/vpd/galaxy_model_params.txt'
        shutil.copyfile(orig_file, gal_param_file)

    from popsycle import synthetic

    for target in targets:
        output_root = f'popsycle_{target:s}'

        if os.path.exists(output_root + '.ebf'):
            continue

        print(f'Running Galaxia for {target} to save in {output_root}.')
        
        # Get the galactic lat and lon of the target
        ra = munge.ra[target]
        dec = munge.dec[target]
        sim_radius = 60.0   # arcsec

        sim_area = math.pi * sim_radius**2
        sim_area_deg2 = sim_area / 3600**2

        print(sim_area, ' arcsec^2')
        print(sim_area_deg2, ' deg^2')

        target_coords = SkyCoord(ra, dec, unit=(u.hourangle, u.deg), frame='icrs')

        gal_l = target_coords.galactic.l.degree
        gal_b = target_coords.galactic.b.degree


        synthetic.run_galaxia(output_root,
                      gal_l,
                      gal_b,
                      sim_area_deg2,
                      gal_param_file)

    return

def dark_lens_prob(target, plot=True, mode='global'):
    ##########
    # Load up all the data and fit samples.
    ##########
    if mode == 'global':
        fitter, data = get_data_and_fitter(pspl_ast_multiphot[target])
        sampls = fitter.load_mnest_results()
    elif mode == 'best':
        foo = get_data_fitter_params_models_samples(target, return_mode = 'best')
        fitter = foo[0]
        data = foo[1]
        stats = foo[2]
        params = foo[3]
        models = foo[4]
        sampls = foo[5]
    else:
        foo = get_data_fitter_params_models_samples(target, return_mode = 'all')
        fitter = foo[0]
        data = foo[1]
        stats = foo[2]
        params = foo[3]
        models = foo[4]
        sampls = foo[5]

        sampls = sampls[mode]


    # We will be using PopSyCLE (Galaxia) runs to sample on the age. 
    # We need to get indices for ages. This means we need a distance-dependent
    # age weighting (or sampling) scheme.
    ebf_list = {'ob120169': paper_dir + 'popsycle_ob120169.ebf',
                'ob140613': paper_dir + 'popsycle_ob140613.ebf',
                'ob150029': paper_dir + 'popsycle_ob150029.ebf',
                'ob150211': paper_dir + 'popsycle_ob150211.ebf'
               }

    ##########
    # Make a 3D grid on distance, mass, and age containing
    # the Kp and OLGE I magnitude of each star. Populate
    # the grid from model isochrones.
    # Distance grid
    ##########
    dL_arr_list = {'ob120169': np.arange(4.0, 12.01, 0.5),
                   'ob140613': np.arange(4.0, 12.01, 0.5),
                   'ob150029': np.arange(4.0, 12.01, 0.5),
                   'ob150211': np.arange(4.0, 12.01, 0.5)
                  }

    dL_arr = dL_arr_list[target]

    # Age grid
    age_arr = np.arange(7, 10.01, 0.5)

    # Mass grid
    mass_arr = np.append(np.arange(0.08, 3.0, 0.05), 
                         np.logspace(np.log10(3.0), np.log10(120), num=32))

    spisea_kp_oI_file = 'spisea_kp_oI_3d_' + target + '.pkl'

    if os.path.exists(spisea_kp_oI_file):
        print('Loading existing SPISEA 3D arrays')
        _tmp = open(spisea_kp_oI_file, 'rb')
        kp_3d = pickle.load(_tmp)
        oI_3d = pickle.load(_tmp)
        _tmp.close()
    else:
        kp_3d = np.empty((len(dL_arr), len(age_arr), len(mass_arr)), dtype=float)
        oI_3d = np.empty((len(dL_arr), len(age_arr), len(mass_arr)), dtype=float)

        # Calculate the extinction at each distance using a 3D exitnction map.
        coords_arr = SkyCoord(data['raL'] * u.deg, 
                              data['decL'] * u.deg, 
                              distance = dL_arr * u.kpc, frame='icrs')

        import dustmaps.marshall
        from dustmaps.marshall import MarshallQuery
        from spisea import synthetic, evolution, atmospheres, reddening #, ifmr
        from spisea.imf import imf #, multiplicityimport dustmaps

        dustmaps.marshall.fetch()
        q = MarshallQuery()
        AKs_arr = q(coords_arr)

        # Define evolution/atmosphere models and extinction law
        from spisea import synthetic, evolution, atmospheres, reddening
        evo_model = evolution.MISTv1() 
        atm_func = atmospheres.get_merged_atmosphere
        red_law = reddening.RedLawDamineli16()


        # Loop through every distance and age and make an isochrone. 
        for dd in range(len(dL_arr)):
            dL = dL_arr[dd]
            AKs = AKs_arr[dd]

            for aa in range(len(age_arr)):
                logAge = age_arr[aa]

                # Make Isochrone object. Note that is calculation will take a few minutes, unless the 
                # isochrone has been generated previously.
                print(f'Making isochrone: dist={dL:.2f} kpc, logAge={logAge:.2f}, AKs={AKs:.2f}')
                my_iso = synthetic.IsochronePhot(logAge, AKs, dL*1000, 
                                                 metallicity=0,
                                                 evo_model=evo_model, atm_func=atm_func,
                                                 red_law=red_law, filters=['ubv,I', 'nirc2,Kp'],
                                                 recomp=False,
                                                 iso_dir=paper_dir + 'iso/')

                iso_interp_kp = interpolate.interp1d(my_iso.points['mass'], 
                                                     my_iso.points['m_nirc2_Kp'],
                                                     kind='linear', 
                                                     bounds_error=False, 
                                                     fill_value=np.nan)
                iso_interp_I  = interpolate.interp1d(my_iso.points['mass'], 
                                                     my_iso.points['m_ubv_I'],
                                                     kind='linear', 
                                                     bounds_error=False, 
                                                     fill_value=np.nan)

                # Loop through each mass bin and find the minimum kp and I magnitudes.
                for mm in range(len(mass_arr)):
                    mass = mass_arr[mm]
                    kp_3d[dd, aa, mm] = iso_interp_kp(mass)
                    oI_3d[dd, aa, mm] = iso_interp_I(mass)

        _tmp = open(spisea_kp_oI_file, 'wb')
        pickle.dump(kp_3d, _tmp)
        pickle.dump(oI_3d, _tmp)
        _tmp.close()
        
    #########
    # Figure out which bins our posterior samples fall into.
    # 
    # Sample age from the distance-dependent age distribution from PopSyCLE.
    #########
    # Setup bins for histogramming.
    dL_delta = np.diff(dL_arr)
    dL_bins = dL_arr[1:] - (dL_delta / 2.0)
    dL_bins = np.insert(dL_bins, 0, dL_arr[0] - dL_delta[0] / 2.0)
    dL_bins = np.append(dL_bins, dL_arr[-1] + dL_delta[-1] / 2.0)

    mL_delta = np.diff(mass_arr)
    mL_bins = mass_arr[1:] - (mL_delta / 2.0)
    mL_bins = np.insert(mL_bins, 0, mass_arr[0] - mL_delta[0] / 2.0)
    mL_bins = np.append(mL_bins, mass_arr[-1] + mL_delta[-1] / 2.0)

    age_delta = np.diff(age_arr)
    age_bins = age_arr[1:] - (age_delta / 2.0)
    age_bins = np.insert(age_bins, 0, age_arr[0] - age_delta[0] / 2.0)
    age_bins = np.append(age_bins, age_arr[-1] + age_delta[-1] / 2.0)


    # Re-normalize the posterior weights array to sum to 1.0
    sampls['weights'] /= sampls['weights'].sum()

    # Take a random sample of the lens fits (weighted) so that we
    # have a proper posterior sample.
    samp_idx = np.random.choice(np.arange(len(sampls['weights'])),
                                size=5000, replace=False, p=sampls['weights'])
    resamp = sampls[samp_idx]

    # Define our sample for the posteriors in 
    # all filters for:
    #    Kp( mag_base_kp, b_sff_kp )
    #    oI( mag_base_oI, b_sff_oI )
    bsff2_idx = np.where(resamp['b_sff2'] > 1)[0] 
    resamp['b_sff2'][bsff2_idx] = 0.999999
    kpL_fit = resamp['mag_base2'] + 2.5*np.log10(1/(1 - resamp['b_sff2']))
    
    bsff1_idx = np.where(resamp['b_sff1'] > 1)[0] 
    resamp['b_sff1'][bsff1_idx] = 0.999999
    oIL_fit = resamp['mag_base1'] + 2.5*np.log10(1/(1 - resamp['b_sff1']))

    # For the same sample, loop through and figure out which bins each
    # falls in (in our 3D model array).
    #    Kp( dL, mL, age(dL | popsycle) ) -- random sampling on age from PopSyCLE.
    #    oI( dL, mL, age(dL | popsycle) ) -- random sampling on age from PopSyCLE.
    mL_fit = resamp['mL']
    dL_fit = 1.0 / resamp['piL']

    mL_idx = np.digitize(mL_fit, bins=mL_bins) - 1
    dL_idx = np.digitize(dL_fit, bins=dL_bins) - 1

    sim = ebf.read(ebf_list[target], '/')

    # Loop through our distance bins and make an age array.
    wgt_dL_age = np.empty((len(dL_arr), len(age_arr)), dtype=float)

    age_idx = np.ones(len(mL_idx), dtype=int) * -1

    import pdb
    for dd in range(len(dL_arr)):
        # Select PopSyCLE stars in this radius bin. 
        sdx = np.where((sim['rad'] >= dL_bins[dd]) & (sim['rad'] < dL_bins[dd+1]))[0]

        age_hist, age_be = np.histogram(sim['age'], bins=age_bins)
        age_hist = age_hist / age_hist.sum()   # normalize

        wgt_dL_age[dd, :] = age_hist

        idx = np.where(dL_idx == dd)[0]
        age_idx[idx] = np.random.choice(np.arange(len(age_arr)),
                                    size=1, replace=False, p=age_hist)

    print("Total points in sample: ", len(dL_fit))
    print('Too small distances: ', (dL_fit < dL_bins[0]).sum())
    print('Too large distances: ', (dL_fit > dL_bins[-1]).sum())
    print('Too small masses: ', (mL_fit < mL_bins[0]).sum())
    print('Too large masses: ', (mL_fit > mL_bins[-1]).sum())

    good = np.where((age_idx >= 0))[0]
    print('Bad entries: ', len(age_idx) - len(good))
    dL_fit = dL_fit[good]
    mL_fit = mL_fit[good]
    kpL_fit = kpL_fit[good]
    oIL_fit = oIL_fit[good]
    age_idx = age_idx[good]
    dL_idx = dL_idx[good]
    mL_idx = mL_idx[good]

    kpL_mod = np.full(len(dL_fit), np.nan, dtype=float)
    oIL_mod = np.full(len(dL_fit), np.nan, dtype=float)

    # Only query the 3D arrays where we have good masses. 
    gm = np.where((mL_fit >= mL_bins[0]) & (mL_fit <= mL_bins[-1]))[0]

    age_mod = age_arr[age_idx]
    kpL_mod[gm] = kp_3d[dL_idx[gm], age_idx[gm], mL_idx[gm]]
    oIL_mod[gm] = oI_3d[dL_idx[gm], age_idx[gm], mL_idx[gm]]


    # Kp probabilities
    no_mass_match_idx_kp = np.argwhere(np.isnan(kpL_mod))
    dark_idx_kp = np.where(kpL_fit > kpL_mod)[0]
    lum_idx_kp = np.where(kpL_fit <= kpL_mod)[0]

    p_no_mass_match_kp = len(no_mass_match_idx_kp)/len(mL_fit)
    p_dark_kp = len(dark_idx_kp)/len(mL_fit)
    p_lum_kp = len(lum_idx_kp)/len(mL_fit)

    # OGLE I probabilities
    no_mass_match_idx_oI = np.argwhere(np.isnan(oIL_mod))
    dark_idx_oI = np.where(oIL_fit > oIL_mod)[0]
    lum_idx_oI = np.where(oIL_fit <= oIL_mod)[0]

    p_no_mass_match_oI = len(no_mass_match_idx_oI)/len(mL_fit)
    p_dark_oI = len(dark_idx_oI)/len(mL_fit)
    p_lum_oI = len(lum_idx_oI)/len(mL_fit)

    # Joint probabilities
    no_mass_match_idx = np.argwhere(np.isnan(kpL_mod) | np.isnan(oIL_mod))
    dark_idx = np.where((kpL_fit > kpL_mod) | (oIL_fit > oIL_mod))[0]
    lum_idx = np.where((kpL_fit <= kpL_mod) & (oIL_fit <= oIL_mod))[0]

    p_no_mass_match = len(no_mass_match_idx)/len(mL_fit)
    p_dark = len(dark_idx)/len(mL_fit)
    p_lum = len(lum_idx)/len(mL_fit)
    
    print('')
    print('Total points in sample after final cuts: ', len(dL_fit))
    print(f'Dark stars (kp):     N = {len(dark_idx_kp):4d}  p = {p_dark_kp:.4f}') 
    print(f'Dead stars (kp):     N = {len(no_mass_match_idx_kp):4d}  p = {p_no_mass_match_kp:.4f}') 
    print(f'Luminous stars (kp): N = {len(lum_idx_kp):4d}  p = {p_lum_kp:.4f}')
    print('Total Prob (kp):     p = {0:.2f}'.format(p_dark_kp + p_no_mass_match_kp + p_lum_kp))
    print('')
    print(f'Dark stars (oI):     N = {len(dark_idx_oI):4d}  p = {p_dark_oI:.4f}') 
    print(f'Dead stars (oI):     N = {len(no_mass_match_idx_oI):4d}  p = {p_no_mass_match_oI:.4f}') 
    print(f'Luminous stars (oI): N = {len(lum_idx_oI):4d}  p = {p_lum_oI:.4f}') 
    print('Total Prob (oI):     p = {0:.2f}'.format(p_dark_oI + p_no_mass_match_oI + p_lum_oI))
    print('')
    print(f'Dark stars (joint):     N = {len(dark_idx):4d}  p = {p_dark:.4f}') 
    print(f'Dead stars (joint):     N = {len(no_mass_match_idx):4d}  p = {p_no_mass_match:.4f}') 
    print(f'Luminous stars (joint): N = {len(lum_idx):4d}  p = {p_lum:.4f}') 
    print('Total Prob (joint):     p = {0:.2f}'.format(p_dark + p_no_mass_match + p_lum))
    print('')

    
    ##########
    # Plotting
    ##########
    mass_bins_plot = {'ob120169': np.logspace(-2, 0.5, 10),
                      'ob140613': np.logspace(-1, 0.5, 15),
                      'ob150029': np.logspace(-2.5, 0.5, 15),
                      'ob150211': np.logspace(np.log10(mL_fit.min()), np.log10(mL_fit.max()+0.01), 11)
                      }

    mass_bins = mass_bins_plot[target]
    n_no_mass_kp = np.zeros(len(mass_bins) - 1)
    n_dark_kp = np.zeros(len(mass_bins) - 1)
    n_lum_kp = np.zeros(len(mass_bins) - 1)

    n_no_mass_oI = np.zeros(len(mass_bins) - 1)
    n_dark_oI = np.zeros(len(mass_bins) - 1)
    n_lum_oI = np.zeros(len(mass_bins) - 1)

    n_no_mass = np.zeros(len(mass_bins) - 1)
    n_dark = np.zeros(len(mass_bins) - 1)
    n_lum = np.zeros(len(mass_bins) - 1)
    
    mass_bin_centers = (mass_bins[:-1] + mass_bins[1:])/2
    bin_width = np.diff(mass_bins)
    
    # mass_bins = mass_bins_plot[target]
    # mass_bin_centers = mass_bins[0:-1] + np.diff(mass_bins) / 2.0
    # bin_width = mass_bins[1] - mass_bins[0]

    for ii in np.arange(len(mass_bins) - 1):
        # Classify based on Kp only
        no_mass_idx_kp = np.where((mL_fit < mass_bins[ii+1]) & 
                            (mL_fit > mass_bins[ii]) & 
                            (np.isnan(kpL_mod)))[0]
        
        dark_idx_kp = np.where((mL_fit < mass_bins[ii+1]) & 
                               (mL_fit > mass_bins[ii]) & 
                               (kpL_fit > kpL_mod))[0]

        lum_idx_kp = np.where((mL_fit < mass_bins[ii+1]) & 
                               (mL_fit > mass_bins[ii]) & 
                               (kpL_fit < kpL_mod))[0]

        # Classify based on ogle I only
        no_mass_idx_oI = np.where((mL_fit < mass_bins[ii+1]) & 
                            (mL_fit > mass_bins[ii]) & 
                            (np.isnan(oIL_mod)))[0]
        
        dark_idx_oI = np.where((mL_fit < mass_bins[ii+1]) & 
                               (mL_fit > mass_bins[ii]) & 
                               (oIL_fit > oIL_mod))[0]

        lum_idx_oI = np.where((mL_fit < mass_bins[ii+1]) & 
                               (mL_fit > mass_bins[ii]) & 
                               (oIL_fit < oIL_mod))[0]

        # Classify based on joint Kp and ogleI
        no_mass_idx = np.where((mL_fit < mass_bins[ii+1]) & 
                            (mL_fit > mass_bins[ii]) & 
                            (np.isnan(kpL_mod) | 
                             np.isnan(oIL_mod)))[0]
        
        dark_idx = np.where((mL_fit < mass_bins[ii+1]) & 
                               (mL_fit > mass_bins[ii]) & 
                               ((kpL_fit > kpL_mod) |
                                (oIL_fit > oIL_mod)))[0]

        lum_idx = np.where((mL_fit < mass_bins[ii+1]) & 
                               (mL_fit > mass_bins[ii]) & 
                               ((kpL_fit < kpL_mod) & 
                                (oIL_fit < oIL_mod)))[0]
        
        n_no_mass_kp[ii] = len(no_mass_idx_kp)/len(mL_fit)
        n_dark_kp[ii] = len(dark_idx_kp)/len(mL_fit)
        n_lum_kp[ii] = len(lum_idx_kp)/len(mL_fit)

        n_no_mass_oI[ii] = len(no_mass_idx_oI)/len(mL_fit)
        n_dark_oI[ii] = len(dark_idx_oI)/len(mL_fit)
        n_lum_oI[ii] = len(lum_idx_oI)/len(mL_fit)
        
        n_no_mass[ii] = len(no_mass_idx)/len(mL_fit)
        n_dark[ii] = len(dark_idx)/len(mL_fit)
        n_lum[ii] = len(lum_idx)/len(mL_fit)

    if plot:
        # Individual filters
        fig, ax = plt.subplots(2, 1, num=1, figsize=(6, 6), sharex='col')
        plt.clf()
        fig, ax = plt.subplots(2, 1, num=1, figsize=(6, 6), sharex='col')
        plt.subplots_adjust(hspace=0.05)
        ax[0].bar(mass_bin_centers, n_no_mass_kp, bin_width, 
                label='Dark$_A$ ({0:.0f}%)'.format(100*p_no_mass_match_kp), color='gray',
                      hatch='++', edgecolor='k')
        ax[0].bar(mass_bin_centers, n_dark_kp, bin_width, bottom=n_no_mass_kp,
                label='Dark$_B$ ({0:.0f}%)'.format(100*p_dark_kp), color='gray',
                      hatch='xx', edgecolor='k')
        ax[0].bar(mass_bin_centers, n_lum_kp, bin_width, bottom=n_no_mass_kp+n_dark_kp,
                label='Lum. ({0:.0f}%)'.format(100*p_lum_kp), color='red', edgecolor='k')
        #ax[0].legend(loc='upper right')
        ax[0].legend()
        ax[0].set_ylabel('Prob. (Kp)')
        ax[0].set_title(target.upper())
        plt.show()
    
        ax[1].bar(mass_bin_centers, n_no_mass_oI, bin_width, 
                label='Dark$_A$ ({0:.0f}%)'.format(100*p_no_mass_match_oI), color='gray',
                      hatch='++', edgecolor='k')
        ax[1].bar(mass_bin_centers, n_dark_oI, bin_width, bottom=n_no_mass_oI,
                label='Dark$_B$ ({0:.0f}%)'.format(100*p_dark_oI), color='gray',
                      hatch='xx', edgecolor='k')
        ax[1].bar(mass_bin_centers, n_lum_oI, bin_width, bottom=n_no_mass_oI+n_dark_oI,
                label='Lum. ({0:.0f}%)'.format(100*p_lum_oI), color='red', edgecolor='k')
        #ax[1].legend(loc='upper right')
        ax[1].legend()
        ax[1].set_ylabel('Prob. (I)')
        ax[1].set_xlabel('Lens mass ($M_\odot$)')

        ax[0].set_xscale('log')
        ax[1].set_xscale('log')
            
        plt.savefig(paper_dir + target + '_dark_lens_prob_' + str(mode) + '.png')
        plt.show()

        # Joint (2 filters)
        plt.figure(3)
        plt.clf()
        plt.bar(mass_bin_centers, n_no_mass, bin_width, 
                label='Dark$_A$ ({0:.0f}%)'.format(100*p_no_mass_match), color='gray',
                      hatch='++', edgecolor='k')
        plt.bar(mass_bin_centers, n_dark, bin_width, bottom=n_no_mass,
                label='Dark$_B$ ({0:.0f}%)'.format(100*p_dark), color='gray',
                      hatch='xx', edgecolor='k')
        plt.bar(mass_bin_centers, n_lum, bin_width, bottom=n_no_mass+n_dark,
                label='Lum. ({0:.0f}%)'.format(100*p_lum), color='red', edgecolor='k')
        plt.legend()
        plt.ylabel('Prob.')
        plt.xlabel('Lens mass ($M_\odot$)')
        plt.title(target.upper())
        
        # if target == 'ob150211':
        plt.gca().set_xscale('log')
            
        plt.savefig(paper_dir + target + '_dark_lens_prob_joint_' + str(mode) + '.png')
        plt.show()

        
        # Age histogram    
        plt.figure(2)
        plt.clf()
        plt.hist(age_mod, bins=np.arange(6, 10.51, 0.5))
        plt.yscale('log')
        plt.xlabel('logage')
        plt.ylabel('number')
        plt.title(target + ' star age distribution')
        plt.savefig(paper_dir + target + '_star_age_distribution_' + str(mode) + '.png')

    ##########
    # Calculate probabilites
    ##########
    print(f'*** MODE = {mode} ***')
    
    # Kp
    star_idx_kp = np.where(kpL_fit <= kpL_mod)[0]

    bd_mass1_idx_kp = np.where((mL_fit <= 0.2) & 
                             (mL_fit > 0) & 
                             (np.isnan(kpL_mod)))[0]
        
    bd_mass2_idx_kp = np.where((mL_fit <= 0.2) & 
                             (mL_fit > 0) &
                             (kpL_fit > kpL_mod))[0]

    wd_mass1_idx_kp = np.where((mL_fit <= 1.2) & 
                             (mL_fit > 0.2) &
                             (np.isnan(kpL_mod)))[0]
        
    wd_mass2_idx_kp = np.where((mL_fit <= 1.2) & 
                             (mL_fit > 0.2) &
                             (kpL_fit > kpL_mod))[0]

    ns_mass1_idx_kp = np.where((mL_fit <= 2.2) & 
                             (mL_fit > 1.2) & 
                             (np.isnan(kpL_mod)))[0]
        
    ns_mass2_idx_kp = np.where((mL_fit <= 2.2) &
                             (mL_fit > 1.2) &
                             (kpL_fit > kpL_mod))[0]

    bh_mass1_idx_kp = np.where((mL_fit > 2.2) & 
                             (np.isnan(kpL_mod)))[0]
        
    bh_mass2_idx_kp = np.where((mL_fit > 2.2) &
                             (kpL_fit > kpL_mod))[0]

    p_st_kp = 100*len(star_idx_kp)/len(mL_fit)
    p_bd_kp = 100*(len(bd_mass1_idx_kp) + len(bd_mass2_idx_kp))/len(mL_fit)
    p_wd_kp = 100*(len(wd_mass1_idx_kp) + len(wd_mass2_idx_kp))/len(mL_fit)
    p_ns_kp = 100*(len(ns_mass1_idx_kp) + len(ns_mass2_idx_kp))/len(mL_fit)
    p_bh_kp = 100*(len(bh_mass1_idx_kp) + len(bh_mass2_idx_kp))/len(mL_fit)

    print('')
    print('** Kp Probabilities **')
    print('Kp P_total = {0:.0f}'.format(p_st_kp + p_bd_kp + p_wd_kp + p_ns_kp + p_bh_kp))
    print('Kp P(Star) = {0:.0f}'.format(p_st_kp))
    print('Kp P(BD) = {0:.0f}'.format(p_bd_kp))
    print('Kp P(WD) = {0:.0f}'.format(p_wd_kp))
    print('Kp P(NS) = {0:.0f}'.format(p_ns_kp))
    print('Kp P(BH) = {0:.0f}'.format(p_bh_kp))

    # OGLE I
    star_idx_oI = np.where(oIL_fit <= oIL_mod)[0]

    bd_mass1_idx_oI = np.where((mL_fit > 0) &
                               (mL_fit <= 0.2) & 
                               (np.isnan(oIL_mod)))[0]
        
    bd_mass2_idx_oI = np.where((mL_fit > 0) &
                               (mL_fit <= 0.2) & 
                               (oIL_fit > oIL_mod))[0]

    wd_mass1_idx_oI = np.where((mL_fit > 0.2) &
                               (mL_fit <= 1.2) & 
                               (np.isnan(oIL_mod)))[0]
        
    wd_mass2_idx_oI = np.where((mL_fit > 0.2) &
                               (mL_fit <= 1.2) & 
                               (oIL_fit > oIL_mod))[0]

    ns_mass1_idx_oI = np.where((mL_fit <= 2.2) & 
                             (mL_fit > 1.2) & 
                             (np.isnan(oIL_mod)))[0]
        
    ns_mass2_idx_oI = np.where((mL_fit <= 2.2) &
                             (mL_fit >= 1.2) &
                             (oIL_fit > oIL_mod))[0]

    bh_mass1_idx_oI = np.where((mL_fit > 2.2) & 
                             (np.isnan(oIL_mod)))[0]
        
    bh_mass2_idx_oI = np.where((mL_fit > 2.2) &
                             (oIL_fit > oIL_mod))[0]

    p_st_oI = 100*len(star_idx_oI)/len(mL_fit)
    p_bd_oI = 100*(len(bd_mass1_idx_oI) + len(bd_mass2_idx_oI))/len(mL_fit)
    p_wd_oI = 100*(len(wd_mass1_idx_oI) + len(wd_mass2_idx_oI))/len(mL_fit)
    p_ns_oI = 100*(len(ns_mass1_idx_oI) + len(ns_mass2_idx_oI))/len(mL_fit)
    p_bh_oI = 100*(len(bh_mass1_idx_oI) + len(bh_mass2_idx_oI))/len(mL_fit)

    print('')
    print('** I Probabilities **')
    print('I P_total = {0:.0f}'.format(p_st_oI + p_bd_oI + p_wd_oI + p_ns_oI + p_bh_oI))
    print('I P(Star) = {0:.0f}'.format(p_st_oI))
    print('I P(BD) = {0:.0f}'.format(p_bd_oI))
    print('I P(WD) = {0:.0f}'.format(p_wd_oI))
    print('I P(NS) = {0:.0f}'.format(p_ns_oI))
    print('I P(BH) = {0:.0f}'.format(p_bh_oI))

    # Joint filter analysis.
    star_idx = np.where((kpL_fit <= kpL_mod) & (oIL_fit <= oIL_mod))[0]

    bd_mass1_idx = np.where((mL_fit <= 0.2) & 
                             (mL_fit > 0) & 
                             (np.isnan(kpL_mod) | 
                              np.isnan(oIL_mod)))[0]
        
    bd_mass2_idx = np.where((mL_fit <= 0.2) & 
                            (mL_fit > 0) &
                            ((kpL_fit > kpL_mod) |
                             (oIL_fit > oIL_mod)))[0]

    wd_mass1_idx = np.where((mL_fit <= 1.2) & 
                             (mL_fit > 0.2) &
                             (np.isnan(kpL_mod) | 
                              np.isnan(oIL_mod)))[0]
        
    wd_mass2_idx = np.where((mL_fit <= 1.2) & 
                             (mL_fit > 0.2) &
                            ((kpL_fit > kpL_mod) |
                             (oIL_fit > oIL_mod)))[0]

    ns_mass1_idx = np.where((mL_fit <= 2.2) & 
                             (mL_fit > 1.2) & 
                             (np.isnan(kpL_mod) | 
                              np.isnan(oIL_mod)))[0]
        
    ns_mass2_idx = np.where((mL_fit <= 2.2) &
                             (mL_fit > 1.2) &
                            ((kpL_fit > kpL_mod) |
                             (oIL_fit > oIL_mod)))[0]

    bh_mass1_idx = np.where((mL_fit > 2.2) & 
                             (np.isnan(kpL_mod) | 
                              np.isnan(oIL_mod)))[0]
        
    bh_mass2_idx = np.where((mL_fit > 2.2) &
                            ((kpL_fit > kpL_mod) |
                             (oIL_fit > oIL_mod)))[0]

    p_st = 100*len(star_idx)/len(mL_fit)
    p_bd = 100*(len(bd_mass1_idx) + len(bd_mass2_idx))/len(mL_fit)
    p_wd = 100*(len(wd_mass1_idx) + len(wd_mass2_idx))/len(mL_fit)
    p_ns = 100*(len(ns_mass1_idx) + len(ns_mass2_idx))/len(mL_fit)
    p_bh = 100*(len(bh_mass1_idx) + len(bh_mass2_idx))/len(mL_fit)

    print('')
    print('** Kp and I Probabilities **')
    print('Kp and oI P_total = {0:.0f}'.format(p_st + p_bd + p_wd + p_ns + p_bh))
    print('Kp and oI P(Star) = {0:.0f}'.format(p_st))
    print('Kp and oI P(BD) = {0:.0f}'.format(p_bd))
    print('Kp and oI P(WD) = {0:.0f}'.format(p_wd))
    print('Kp and oI P(NS) = {0:.0f}'.format(p_ns))
    print('Kp and oI P(BH) = {0:.0f}'.format(p_bh))
    

    n_age10 = len(np.where(age_mod == 10)[0])
    p_age10 = 100 * n_age10/len(age_mod)

    n_age95 = len(np.where(age_mod == 9.5)[0])
    p_age95 = 100 * n_age95/len(age_mod)

    n_age9 = len(np.where(age_mod == 9)[0])
    p_age9 = 100 * n_age9/len(age_mod)

    n_age85 = len(np.where(age_mod == 8.5)[0])
    p_age85 = 100 * n_age85/len(age_mod)

    n_age8 = len(np.where(age_mod == 8)[0])
    p_age8 = 100 * n_age8/len(age_mod)

    print('Percent with logage 10 : ', p_age10)
    print('Percent with logage 9.5 : ', p_age95)
    print('Percent with logage 9 : ', p_age9)
    print('Percent with logage 8.5 : ', p_age85)
    print('Percent with logage 8 : ', p_age8)
    
    return (p_st_kp, p_st_oI, p_st,
            p_bd_kp, p_bd_kp, p_bd,
            p_wd_kp, p_wd_kp, p_wd,
            p_ns_oI, p_ns_oI, p_ns,
            p_bh_oI, p_bh_oI, p_bh)


def make_lens_probability_table():
    # Delete old file if exists                                                     
    table_file = paper_dir + 'lens_type_prob.txt'
    if os.path.exists(table_file):
        os.remove(table_file)

    with open(table_file, 'a+') as tb:
        # for targ in targets:
        for targ in ['ob120169', 'ob140613', 'ob150029', 'ob150211']:
            foo = dark_lens_prob(targ, plot=False)
            p_st_kp = foo[0]
            p_st_oI = foo[1]
            p_st    = foo[2]
            p_bd_kp = foo[3]
            p_bd_oI = foo[4]
            p_bd    = foo[5]
            p_wd_kp = foo[6]
            p_wd_oI = foo[7]
            p_wd    = foo[8]
            p_ns_kp = foo[9]
            p_ns_oI = foo[10]
            p_ns    = foo[11]
            p_bh_kp = foo[12]
            p_bh_oI = foo[13]
            p_bh    = foo[14]
            
            tb.write('{0} & {1:.0f} & {2:.0f} & {3:.0f} & {4:.0f} & {5:.0f} & {6:.0f} & {7:.0f} & ' \
                         '{8:.0f} & {9:.0f} & {10:.0f} \\\ \n'.format(targ, p_st_kp, p_st_oI,
                                                                      p_bd_kp, p_bd_oI, 
                                                                      p_wd_kp, p_wd_oI,
                                                                      p_ns_kp, p_ns_oI, 
                                                                      p_bh_kp, p_bh_oI))

    # Delete old file if exists                                                     
    table_file = paper_dir + 'lens_type_prob_avg.txt'
    if os.path.exists(table_file):
        os.remove(table_file)

    with open(table_file, 'a+') as tb:
        # for targ in targets:
        for targ in ['ob120169', 'ob140613', 'ob150029', 'ob150211']:
            foo = dark_lens_prob(targ, plot=False)
            p_st_kp = foo[0]
            p_st_oI = foo[1]
            p_bd_kp = foo[2]
            p_wd_kp = foo[3]
            p_ns_kp = foo[4]
            p_bh_kp = foo[5]
            p_bd_oI = foo[6]
            p_wd_oI = foo[7]
            p_ns_oI = foo[8]
            p_bh_oI = foo[9]

            tb.write('{0} & {1:.0f} & {2:.0f} & {3:.0f} & ' \
                         '{4:.0f} & {5:.0f} \\\ \n'.format(targ, (p_st_kp + p_st_oI)/2.0, 
                                                           (p_bd_kp + p_bd_oI)/2.0, 
                                                           (p_wd_kp + p_wd_oI)/2.0, 
                                                           (p_ns_kp + p_ns_oI)/2.0, 
                                                           (p_bh_kp + p_bh_oI)/2.0))

    return

# def elppd(target):
#     targets = ['ob120169', 'ob140613', 'ob150029', 'ob150211']
    
#     for targ in targets:
#         import warnings
#         warnings.simplefilter(action='ignore', category=FutureWarning)

#         fitter, data = get_data_and_fitter(pspl_ast_multiphot[target])
#         stats = calc_summary_statistics(fitter)
#         tab = fitter.load_mnest_modes()

#         # Loop through each of the modes for this target
#         # and calculate the ELPPD.
#         for mode in range(len(stats)):
#             # Make an EQUAL WEIGHT POSTERIOR sample
#             sampls = tab[mode]
#             samp_idx = np.random.choice(np.arange(len(sampls['weights'])),
#                                 size=5000, replace=False, p=sampls['weights'])
#             resamp = sampls[samp_idx]

#             # Number of samples.
#             nsamp = len(resamp)

#             # Get RA and Dec of target.
#             raL, decL = data['raL'], data['decL']

#             # Initialize an array to store the pointwise LOG likelihood samples.
#             # Each row corresponds to a posterior sample, and so
#             #     pw_logL_arr[s].sum() gives the total logL for the s-th posterior sample.
#             # The number of rows equals the number of posterior samples.
#             # Each column corresponds to a data point (i.e. time), and so the length
#             #     of each row equals the number of data points.
#             pw_logL_arr = np.zeros((len(ramp), len(data['t_phot1'])))

#         # Fill out the table by looping over posterior samples.
#         for sdx, samp in enumerate(t):
#             print('Sample number ', sdx)
#             params = t[t.colnames[:21]][sdx]
#             params_dict = model_fitter.generate_params_dict(params, fit.fitter_param_names)
#             mod = fit.model_class(*params_dict.values(),
#                                    raL=raL,
#                                    decL=decL)
#             pw_logL = model_fitter.pointwise_likelihood(fit.data, mod, filt_index=0)
#             pw_logL_arr[sdx] = pw_logL

#         np.savetxt(targ + '.txt', pw_logL_arr)
#         pdb.set_trace()
#         # Calculate the LIKELIHOOD (not log) by exponentiating. 
#         pw_L_arr = np.exp(pw_logL_arr)

#         # Sum over the pointwise likelihood estimates.
#         # pw_logL_arr.sum(axis=0).shape = number of data points (since summed over samples)
#         pw_L_arr_sampsum = pw_L_arr.sum(axis=0)
#         pw_logL_arr_sampsum = pw_logL_arr.sum(axis=0)

#         elppd = 2 * pw_logL_arr_sampsum.sum()/nsamp - np.log(pw_L_arr_sampsum/nsamp).sum()

#         print(targ)
#         print('elppd = ', elppd)

#     return


def dark_lens_prob_bad_age(target):
    """
    Use a Galaxia EBF file to estimate an age distribution. Then
    sample from that age distribution, make synthetic stellar populations
    using SPISEA, and check which ones hold up against our light-curve
    fitting results.
    """
    # Load up microlensing fits (joint phot + astrom).
    # Also fetch our best fit and figure out the source and lens/neighbor
    # brightness.
    foo = get_data_fitter_params_models_samples('ob150211', return_mode = 'best')
    fitter = foo[0]
    data = foo[1]
    stats = foo[2]
    params = foo[3]
    models = foo[4]
    sampls = foo[5]

    sampls['weights'] /= sampls['weights'].sum()
    
    # Get rid of negative blending which leads to nan
    # for lens magnitude and replace them with 0.9999 (i.e. super close to 1).
    bsff2_idx = np.where(sampls['b_sff2'] > 1)[0] 
    sampls['b_sff2'][bsff2_idx] = 0.999999

    # Lens fit values
    dL_fit = 1 / sampls['piL']
    mL_fit = sampls['mL']
    kpL_fit   = sampls['mag_base2'] + 2.5*np.log10(1/(1 - sampls['b_sff2']))
    ogleL_fit = sampls['mag_base1'] + 2.5*np.log10(1/(1 - sampls['b_sff1']))

    # Setup some target-specific details
    dL_arr_list = {'ob120169': np.arange(4.0, 12.01, 0.5),
                   'ob140613': np.arange(4.0, 12.01, 0.5),
                   'ob150029': np.arange(4.0, 12.01, 0.5),
                   'ob150211': np.arange(4.0, 12.01, 0.5)
                  }
    right_list = {'ob120169': None,
                  'ob140613': None,
                  'ob150029': None,
                  'ob150211': None
                 }
    top_list = {'ob120169': None,
                'ob140613': None,
                'ob150029': None,
                'ob150211': None
                 }
    bottom_list = {'ob120169': None,
                   'ob140613': None,
                   'ob150029': None,
                   'ob150211': None
                 }
    ebf_list = {'ob120169': None,
                'ob140613': None,
                'ob150029': None,
                'ob150211': '/u/jlu/work/microlens/OB150211/current/notes/ob150211.ebf'
                 }

    ##########
    # Define isochrone parameters. Use the age distribution from the EBF file.
    ##########
    sim = ebf.read(ebf_list[target], '/')

    # We will step through this list of distances.
    dL_arr_edge = dL_arr_list[target]
    dL_arr = dL_arr_edge[:-1] + (np.diff(dL_arr_edge) / 2.0)
        
    coords_arr = SkyCoord(data['raL'] * u.deg, 
                          data['decL'] * u.deg, 
                          distance = dL_arr * u.kpc, frame='icrs')

    import dustmaps.marshall
    from dustmaps.marshall import MarshallQuery
    from spisea import synthetic, evolution, atmospheres, reddening #, ifmr
    from spisea.imf import imf #, multiplicity

    # Fetch the extinction at each distance.
    dustmaps.marshall.fetch()
    q = MarshallQuery()
    AKs_arr = q(coords_arr)

    # Define evolution/atmosphere models and extinction law
    evo_model = evolution.MISTv1() 
    atm_func = atmospheres.get_merged_atmosphere
    red_law = reddening.RedLawDamineli16()
    # IMF power law boundaries. changed start from 0.08 to 0.1 because of MIST.
    mass_limits = np.array([0.1, 0.5, 120])
    # IMF powers
    powers = np.array([-1.3, -2.3])
    # Cluster mass
    cluster_mass = 1E4

    # Take a random sample of the lens fits (weighted)
    samp_idx = np.random.choice(np.arange(len(sampls['weights'])),
                                size=1000, replace=False, p=sampls['weights'])

    dL_fit = dL_fit[samp_idx]
    mL_fit = mL_fit[samp_idx]
    kpL_fit = kpL_fit[samp_idx]
    ogleL_fit = ogleL_fit[samp_idx]

    # Only keep things within dL bin
    # FIXME: Change this ST we won't need this line.
    keep_idx = np.where((dL_fit > dL_arr[0]) & 
                        (dL_fit < dL_arr[-1]))[0]

    print('len(keep_idx) = ', len(keep_idx))
    dL_fit = dL_fit[keep_idx]
    mL_fit = mL_fit[keep_idx]
    kpL_fit = kpL_fit[keep_idx]
    kpL_mod = np.zeros(len(keep_idx))
    ogleL_fit = ogleL_fit[keep_idx]
    ogleL_mod = np.zeros(len(keep_idx))
    log_age_mod = np.zeros(len(keep_idx))

    # We will simulate a population over a grid of fit points and age.
    # For each fit point, we will try all the age bins. 
    # Here are the age bins.
    age_bin_delta = 0.5
    age_bin_cents = np.arange(7, 10.01, age_bin_delta)
    age_bin_edges = age_bin_cents + (age_bin_delta / 2.0)
    age_bin_edges = np.insert(age_bin_edges, 0, age_bin_cents[0] - (age_bin_delta / 2.0))
    print(f'age_bin_edges = {age_bin_edges}')

    # Now expand the arrays to have 2D = [N_fit_points, N_age_bins]

    # Sort the dL into the defined distance bins
    bin_idx = np.digitize(dL_fit, dL_arr_edge)

    # Loop through each distance bin.
    for ii in np.arange(len(dL_arr)):

        # Select PopSyCLE stars in this radius bin. 
        sdx = np.where((sim['rad'] >= dL_arr_edge[ii]) & (sim['rad'] < dL_arr_edge[ii+1]))[0]

        age_hist, age_be = np.histogram(sim['age'], bins=age_bin_edges)
        age_hist /= age_hist.sum()   # normalize
        age_idx = np.random.choice(np.arange(len(age_bin_cents)),
                                size=100, replace=False, p=age_hist)

        # Draw the logAge from PopSyCLE, round to the nearest delta-logAge=0.5 so we don't
        # make too many isochrones (expensive).
        logAge = np.random.choice(sim['age'][sdx])
        logAge = round(logAge * 2) / 2   # Rounds to the nearest 0.5 or 1.0
        
        bin_ii_idx = np.where(bin_idx == ii+1)[0]
        dL = dL_arr[ii]
        AKs = AKs_arr[ii]
        
        # Make Isochrone object. Note that is calculation will take a few minutes, unless the 
        # isochrone has been generated previously.
        print(f'Making isochrone: dist={dL:.2f} kpc, logAge={logAge:.2f}, AKs={AKs:.2f}')
        my_iso = synthetic.IsochronePhot(logAge, AKs, dL*1000, metallicity=0,
                                         evo_model=evo_model, atm_func=atm_func,
                                         red_law=red_law, filters=['ubv,I', 'nirc2,Kp'],
                                         recomp=True,
                                         iso_dir=paper_dir + 'iso/')

        # define IMF
        trunc_kroupa = imf.IMF_broken_powerlaw(mass_limits, powers)
        
        # make cluster
        cluster = synthetic.ResolvedCluster(my_iso, trunc_kroupa, cluster_mass, seed=1)
        output = cluster.star_systems
        print('max mass = ', output['mass'].max())
        if np.isnan(output['m_nirc2_Kp']).sum() > 0:
            pdb.set_trace()

        # plt.figure(3)
        # plt.clf()
        # plt.plot(output['mass'], output['m_nirc2_Kp'], 'k.', label='model,Kp')
        # plt.plot(output['mass'], output['m_ubv_I'], 'b.', label='model,I')
        
        # For each star in this particular distance bin...
        # Figure out which mass bins
        for jj in bin_ii_idx:
            idx = np.where((output['mass'] < mL_fit[jj] + 0.1) & 
                           (output['mass'] > mL_fit[jj] - 0.1))[0]

            if len(idx) == 0:
                kpL_mod[jj] = np.nan
                ogleL_mod[jj] = np.nan
                log_age_mod[jj] = np.nan
            else:
                kpL_mod[jj] = output['m_nirc2_Kp'][idx].min()
                ogleL_mod[jj] = output['m_ubv_I'][idx].min()
                log_age_mod[jj] = logAge
                
        # plt.plot(mL_fit[bin_ii_idx], kpL_mod[bin_ii_idx], 'rs', label='fit, Kp')
        # plt.plot(mL_fit[bin_ii_idx], ogleL_mod[bin_ii_idx], 'cs', label='fit, I')
        # plt.xscale('log')
        # plt.legend()
        # plt.xlabel('mass')
        # plt.ylabel('Kp or I (mag)')
        # plt.show()
        

    plt.close(1)
    fig, ax = plt.subplots(1, 3, num=1, figsize=(18, 6))
    # plt.clf()
    plt.subplots_adjust(wspace=0.27, left=0.08, right=0.98, top=0.93)
    ax[0].scatter(dL_fit, mL_fit, c=mL_fit, alpha=0.5, cmap='viridis')
    model_fitter.contour2d_alpha(1/sampls['piL'], sampls['mL'], weights=sampls['weights'],
                                     ax=ax[0], color='k',
                                 plot_density=False, sigma_levels=[1, 2, 3])
    ax[0].set_xlabel('$d_L$ [kpc]')
    ax[0].set_ylabel('$M_L [M_\odot]$')
    ax[0].set_title(target.upper())
    if right_list[target] is not None:
        ax[0].set_xlim(right=right_list[target])
    if top_list[target] is not None:
        ax[0].set_ylim(top=top_list[target])
    if bottom_list[target] is not None:
        ax[0].set_ylim(bottom=bottom_list[target])

    max1 = np.nanmax(kpL_mod)
    max2 = np.nanmax(kpL_fit)
    min1 = np.nanmin(kpL_mod)
    min2 = np.nanmin(kpL_fit)
    max = np.max([max1, max2])
    min = np.min([min1, min2])
    ax[1].set_xlabel('$Kp_L$ ($d_L$, $M_L$, $\\bigstar$) [mag]')
    ax[1].set_ylabel('$Kp_L$ ($b_{SFF}$, $m_{base}$) [mag]')
    ax[1].set_title('Kp')
    ax[1].plot([min, max], [min, max], color='gray')
    ax[1].fill_between(x=[min, max], y1=[max, max], y2=[min, max], color='gray', alpha=0.3)
    ax[1].scatter(kpL_mod, kpL_fit, c=mL_fit, alpha=0.5, cmap='viridis')
    ax[1].invert_yaxis()
    plt.text(0.6, 0.1,
               'Dark Lens',
               transform=ax[1].transAxes)
    plt.text(0.5, 0.9, 'P(dark lens) = {0:.0f}%'.format(100*len(np.where(kpL_fit > kpL_mod)[0])/len(kpL_fit)), 
             transform=ax[1].transAxes,
             ha='center')
    ax[1].axis('equal')

    
    max1 = np.nanmax(ogleL_mod)
    max2 = np.nanmax(ogleL_fit)
    min1 = np.nanmin(ogleL_mod)
    min2 = np.nanmin(ogleL_fit)
    max = np.max([max1, max2])
    min = np.min([min1, min2])
    ax[2].set_xlabel('$I_L$ ($d_L$, $M_L$, $\\bigstar$) [mag]')
    ax[2].set_ylabel('$I_L$ ($b_{SFF}$, $m_{base}$) [mag]')
    ax[2].set_title('OGLE I')
    ax[2].plot([min, max], [min, max], color='gray')
    ax[2].fill_between(x=[min, max], y1=[max, max], y2=[min, max], color='gray', alpha=0.3)
    ax[2].scatter(ogleL_mod, ogleL_fit, c=mL_fit, alpha=0.5, cmap='viridis')
    ax[2].invert_yaxis()
    plt.text(0.6, 0.1,
               'Dark Lens',
               transform=ax[2].transAxes)
    plt.text(0.5, 0.9, 'P(dark lens) = {0:.0f}%'.format(100*len(np.where(ogleL_fit > ogleL_mod)[0])/len(ogleL_fit)), 
             transform=ax[2].transAxes,
             ha='center')
    ax[2].axis('equal')
    
    plt.savefig(paper_dir + target + '_dark_lens_prob.png')
    
    print(len(np.where(kpL_fit > kpL_mod)[0]))
    print(len(kpL_fit))
    print('Probability of dark lens (KP): {0:.0f}%'.format(100*len(np.where(kpL_fit > kpL_mod)[0])/len(kpL_fit)))

    print(len(np.where(ogleL_fit > ogleL_mod)[0]))
    print(len(ogleL_fit))
    print('Probability of dark lens (I): {0:.0f}%'.format(100*len(np.where(ogleL_fit > ogleL_mod)[0])/len(ogleL_fit)))
    
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


def plot_lens_geometry(target, axis_lim_scale=3, vel_scale=0.05):
    """
    target : str
        Target name (lower case).
    axis_lim_scale : float
        Axis limits in units of thetaE
    vel_scale : float
        Scale factor for the velocity arrows. 
    """

    
    fitter, data = get_data_and_fitter(pspl_ast_multiphot[target])
    mod_all = fitter.get_best_fit_modes_model(def_best = 'median')
    mode = pspl_ast_multiphot_mode[target]
    mod = mod_all[mode]

    print('tE      = {0:5.2f} days'.format(mod.tE))
    print('piE     = {0:5.2f} = {1:5.2f}'.format(mod.piRel / mod.thetaE_amp, mod.piE_amp))
    print('vec_piE = [{0:5.2f}, {1:5.2f}]'.format(mod.piE[0], mod.piE[1]))
    print('xL0     = [{0:7.4f}, {1:7.4f}] asec'.format(mod.xL0[0], mod.xL0[1]))
    print('xS0     = [{0:7.4f}, {1:7.4f}] asec'.format(mod.xS0[0], mod.xS0[1]))
    print('muL     = [{0:7.4f}, {1:7.4f}] mas/yr'.format(mod.muL[0], mod.muL[1]))
    print('muS     = [{0:7.4f}, {1:7.4f}] mas/yr'.format(mod.muS[0], mod.muS[1]))
    print('thetaE  = |{1:6.3f}, {2:6.3f}| mas = {0:6.3f} mas'.format(mod.thetaE_amp,
                                                              mod.thetaE[0],
                                                              mod.thetaE[1]))
    print('thetaS0 = |{0:7.4f}, {1:7.4f}| mas = {2:7.4f} mas'.format(mod.thetaS0[0],
                                                                    mod.thetaS0[1],
                                                                    mod.beta))
    print('muRel   = |{0:7.4f}, {1:7.4f}| mas/yr = {2:7.4f} mas/yr'.format(mod.muRel[0],
                                                                           mod.muRel[1],
                                                                           mod.muRel_amp))
    print('u0      = {0:6.2f}'.format(mod.u0_amp))
    print('piRel   = {0:7.4f} mas'.format(mod.piRel))
    print('dL      = {0:7.1f} pc'.format(mod.dL))
    print('dS      = {0:7.1f} pc'.format(mod.dS))

    # Time samples for all curves.
    t_obs = np.arange(mod.t0 - 1000, mod.t0 + 1000, 10)
    
    # Parallax vector (normalized to 1).
    parallax_vec = model.parallax_in_direction(mod.raL, mod.decL, t_obs)
    
    # Astrometry of the unlensed source in (1) geo, (2) helio system
    xS_unlens_geo = mod.get_astrometry_unlensed(t_obs)
    xS_unlens_sun = xS_unlens_geo - (mod.piS * parallax_vec) * 1e-3  # arcsec

    # Astrometry of the lens in (1) geo, (2) helio system
    xL_unlens_geo = mod.get_lens_astrometry(t_obs)
    xL_unlens_sun = xL_unlens_geo - (mod.piL * parallax_vec) * 1e-3

    # Astrometry of the source in the heliocentric lens rest frame as seen from (1) geo, (2) helio
    xS_restL0_geo = xS_unlens_geo - mod.xL0
    xS_restL0_sun = xS_unlens_sun - mod.xL0

    tidx = np.argmin(np.abs(t_obs - mod.t0))

    # Fix up all positions to be relative to the source location (no parallax)
    # at time t0.
    xL_ref = xL_unlens_sun[tidx, :]

    xS_unlens_geo -= xL_ref
    xS_unlens_sun -= xL_ref
    xL_unlens_geo -= xL_ref
    xL_unlens_sun -= xL_ref

    # Plot params
    arr_head_len = 0.1
    arr_head_wid = 0.1
    arrow_width = 0.02
    

    ##########
    # Geocentric plot
    ##########
    plt.close(1)
    plt.figure(1)
    plt.clf()
    cir = plt.Circle((xL_unlens_geo[tidx, 0]*1e3,
                      xL_unlens_geo[tidx, 1]*1e3),
                      mod.thetaE_amp,
                      color='grey', fill=False)
    plt.gca().add_artist(cir)
    plt.plot([xL_unlens_geo[tidx, 0]*1e3], [xL_unlens_geo[tidx, 1]*1e3], 'ko')
    plt.plot([xS_unlens_geo[tidx, 0]*1e3], [xS_unlens_geo[tidx, 1]*1e3], 'r*')
    plt.arrow(xL_unlens_geo[tidx, 0]*1e3, xL_unlens_geo[tidx, 1]*1e3,
                  mod.muL[0]*vel_scale, mod.muL[1]*vel_scale,
                  width=arrow_width, head_width=arr_head_wid, head_length=arr_head_len, fc='k', ec='k')
    plt.arrow(xS_unlens_geo[tidx, 0]*1e3, xS_unlens_geo[tidx, 1]*1e3,
                  mod.muS[0]*vel_scale, mod.muS[1]*vel_scale,
                  width=arrow_width, head_width=arr_head_wid, head_length=arr_head_len, fc='r', ec='r')
    plt.plot(xS_unlens_geo[:,0]*1e3, xS_unlens_geo[:,1]*1e3, 'r--', label='Src, unlensed')
    plt.plot(xL_unlens_geo[:,0]*1e3, xL_unlens_geo[:,1]*1e3, 'k--', label='Lens')
    plt.xlabel(r'$\Delta\alpha^*$ (mas)')
    plt.ylabel(r'$\Delta\delta$ (mas)')
    plt.axis('equal')
    plt.gca().invert_xaxis()
    # xlim_lo = xL_unlens_geo[tidx, 0]*1e3 + mod.thetaE_amp * axis_lim_scale
    # xlim_hi = xL_unlens_geo[tidx, 0]*1e3 - mod.thetaE_amp * axis_lim_scale
    # ylim_lo = xL_unlens_geo[tidx, 1]*1e3 - mod.thetaE_amp * axis_lim_scale
    # ylim_hi = xL_unlens_geo[tidx, 1]*1e3 + mod.thetaE_amp * axis_lim_scale
    half_lim = 5 # mas
    xlim_lo = xL_unlens_geo[tidx, 0]*1e3 + half_lim * axis_lim_scale
    xlim_hi = xL_unlens_geo[tidx, 0]*1e3 - half_lim * axis_lim_scale
    ylim_lo = xL_unlens_geo[tidx, 1]*1e3 - half_lim * axis_lim_scale
    ylim_hi = xL_unlens_geo[tidx, 1]*1e3 + half_lim * axis_lim_scale
    plt.xlim(xlim_lo, xlim_hi)
    plt.ylim(ylim_lo, ylim_hi)
    plt.gca().xaxis.set_major_locator(plt.MultipleLocator(1))
    plt.gca().yaxis.set_major_locator(plt.MultipleLocator(1))
    plt.legend()
    plt.title(target.upper())
    plt.savefig(paper_dir + 'geometry_geo_' + target + '.png')

    ##########
    # Geocentric plot
    ##########
    plt.close(2)
    plt.figure(2)
    plt.clf()
    cir = plt.Circle((xL_unlens_sun[tidx, 0]*1e3,
                      xL_unlens_sun[tidx, 1]*1e3),
                      mod.thetaE_amp,
                      color='grey', fill=False)
    plt.gca().add_artist(cir)
    plt.plot([xL_unlens_sun[tidx, 0]*1e3], [xL_unlens_sun[tidx, 1]*1e3], 'ko')
    plt.plot([xS_unlens_sun[tidx, 0]*1e3], [xS_unlens_sun[tidx, 1]*1e3], 'r*')
    plt.arrow(xL_unlens_sun[tidx, 0]*1e3, xL_unlens_sun[tidx, 1]*1e3,
                  mod.muL[0]*vel_scale, mod.muL[1]*vel_scale,
                  width=arrow_width, head_width=arr_head_wid, head_length=arr_head_len, fc='k', ec='k')
    plt.arrow(xS_unlens_sun[tidx, 0]*1e3, xS_unlens_sun[tidx, 1]*1e3,
                  mod.muS[0]*vel_scale, mod.muS[1]*vel_scale,
                  width=arrow_width, head_width=arr_head_wid, head_length=arr_head_len, fc='r', ec='r')
    plt.plot(xS_unlens_sun[:,0]*1e3, xS_unlens_sun[:,1]*1e3, 'r--', label='Src, unlensed')
    plt.plot(xL_unlens_sun[:,0]*1e3, xL_unlens_sun[:,1]*1e3, 'k--', label='Lens')
    plt.xlabel(r'$\Delta\alpha^*$ (mas)')
    plt.ylabel(r'$\Delta\delta$ (mas)')
    plt.axis('equal')
    plt.gca().invert_xaxis()
    # xlim_lo = xL_unlens_sun[tidx, 0]*1e3 + mod.thetaE_amp * axis_lim_scale
    # xlim_hi = xL_unlens_sun[tidx, 0]*1e3 - mod.thetaE_amp * axis_lim_scale
    # ylim_lo = xL_unlens_sun[tidx, 1]*1e3 - mod.thetaE_amp * axis_lim_scale
    # ylim_hi = xL_unlens_sun[tidx, 1]*1e3 + mod.thetaE_amp * axis_lim_scale
    xlim_lo = xL_unlens_sun[tidx, 0]*1e3 + half_lim * axis_lim_scale
    xlim_hi = xL_unlens_sun[tidx, 0]*1e3 - half_lim * axis_lim_scale
    ylim_lo = xL_unlens_sun[tidx, 1]*1e3 - half_lim * axis_lim_scale
    ylim_hi = xL_unlens_sun[tidx, 1]*1e3 + half_lim * axis_lim_scale
    plt.xlim(xlim_lo, xlim_hi)
    plt.ylim(ylim_lo, ylim_hi)
    plt.gca().xaxis.set_major_locator(plt.MultipleLocator(1))
    plt.gca().yaxis.set_major_locator(plt.MultipleLocator(1))
    plt.legend()
    plt.title(target.upper())
    plt.savefig(paper_dir + 'geometry_sun_' + target + '.png')

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
        signal.append("${:.1f}\sigma$ \\\\".format(sig))
        
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

    # Header details are hard-coded in the paper. 
    # tab.write('astrom_significance.tex', format='aastex', overwrite=True)
    tab.write('astrom_significance.tex', format='ascii.fixed_width_no_header', delimiter='&', overwrite=True, delimiter_pad=' ', bookend=False)

    print(tab)

    os.chdir(ret_dir)

def plot_linear_motion(target, fig_num = 1):
    """
    Plot the linear motion of the target
    with a proper motion fit that excludes the peak year.
    """
    from jlu.microlens import fit_velocities

    out_dir = paper_dir + 'compare_linear_motion/'

    def plot_target(fitob, fig_num):
        '''
        Plots the linear fit for the target only. The figure is saved.
        
        Parameters
        ----------
        figob : StarTable from fit_velocities
        '''
        res_rng = fit_velocities.res_dict[fitob.target]

        stars = np.append([fitob.target], comp_stars[fitob.target])

        # Figure out the min/max of the times for these sources.
        tdx = np.where(fitob['name'] == fitob.target)[0][0]
        tmin = fitob['t'][tdx].min() - 0.5   # in days
        tmax = fitob['t'][tdx].max() + 0.5   # in days

        # Setup figure and color scales
        figsize = (6, 9.5)
        plt.close(fig_num)
        fig = plt.figure(fig_num, figsize=figsize)
        plt.clf()

        # st = fig.suptitle(fitob.target + " astrometry", fontsize = 20)

        grid_t = plt.GridSpec(1, 1, bottom=0.60, top=0.95, left=0.25, right=0.79)
        grid_b = plt.GridSpec(2, 1, hspace=0.1, wspace=0.5, bottom=0.10, top=0.5, left=0.25, right=0.79)

        cmap = plt.cm.plasma
        norm = plt.Normalize(vmin=tmin, vmax=tmax)
        smap = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        smap.set_array([])

        ax_sky = fig.add_subplot(grid_t[0, 0])
        ax_resX = fig.add_subplot(grid_b[1, 0])
        ax_resY = fig.add_subplot(grid_b[0, 0])

        # Fetch the data
        tdx = np.where(fitob['name'] == fitob.target)[0][0]
        star = fitob[tdx]

        # Change signs of the East
        x = star['x']*-1.0
        y = star['y']
        x0 = star['x0']*-1.0
        y0 = star['y0']
        vx = star['vx']*-1.0
        vy = star['vy']
        
        # Make the model curves
        tmod = np.arange(tmin, tmax, 0.1)
        xmod = x0 + vx * (tmod - star['t0'])
        ymod = y0 + vy * (tmod - star['t0'])
        xmode = np.hypot(star['x0e'], star['vxe'] * (tmod - star['t0']))
        ymode = np.hypot(star['y0e'], star['vye'] * (tmod - star['t0']))

        xmod_at_t = x0 + vx * (star['t'] - star['t0'])
        ymod_at_t = y0 + vy * (star['t'] - star['t0'])

        # Plot Positions on Sky
        ax_sky.plot(xmod, ymod, '-', color='grey', zorder=1)
        ax_sky.plot(xmod + xmode, ymod + ymode, '--', color='grey', zorder=1)
        ax_sky.plot(xmod - xmode, ymod - ymode, '--', color='grey', zorder=1)
        sc = ax_sky.scatter(x, y, c=star['t'], cmap=cmap, norm=norm, s=20, zorder=2)
        ax_sky.errorbar(x, y, xerr=star['xe'], yerr=star['ye'],
                            ecolor=smap.to_rgba(star['t']), fmt='none', elinewidth=2, zorder=2)
        ax_sky.set_aspect('equal', adjustable='datalim')

        # Figure out which axis has the bigger data range.
        xy_rng = 0.020  # this is the same range for all the stars.
        xmin = x0 - (xy_rng / 2.0)
        xmax = x0 + (xy_rng / 2.0)
        ymin = y0 - (xy_rng / 2.0)
        ymax = y0 + (xy_rng / 2.0)
        ax_sky.set_xlim(xmin, xmax)
        ax_sky.set_ylim(ymin, ymax)

        # Set labels
        ax_sky.invert_xaxis()
        ax_sky.set_title(fitob.target.upper())
        ax_sky.set_xlabel(r'$\Delta\alpha*$ (")')
        ax_sky.set_ylabel(r'$\Delta\delta$ (")')

        # Plot Residuals vs. Time
        xres = (x - xmod_at_t) * 1e3
        yres = (y - ymod_at_t) * 1e3
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

        xresrng = xres + np.sign(xres)*(xrese + 0.1)
        yresrng = yres + np.sign(yres)*(yrese + 0.1)
        # ax_resX.set_ylim(xresrng.min(), xresrng.max())
        # ax_resY.set_ylim(yresrng.min(), yresrng.max())
        ax_resX.set_ylim(-0.95, 0.95)
        ax_resY.set_ylim(-0.95, 0.95)
        ax_resY.get_xaxis().set_visible(False)
        ax_resX.set_ylabel(r'$\alpha^*$')
        ax_resY.set_ylabel(r'$\delta$')
        plt.gcf().text(0.04, 0.3, 'Residuals (mas)', rotation=90, fontsize=18,
                           ha='center', va='center')

        cb_ax = fig.add_axes([0.8, 0.60, 0.02, 0.35])
        plt.colorbar(sc, cax=cb_ax, label='Year')

        plt.savefig(f"{out_dir}/{fitob.target}_linear_fit.png")

        return

    fitob = fit_velocities.StarTable(target)
    fitob.fit(time_cut=fitob.time_cut) # Fit without the peak year
    
    plot_target(fitob, fig_num)

    return

def org_solutions_for_table():
    targets = ['ob120169', 'ob140613', 'ob150029', 'ob150211']
    
    for targ in targets:
        import warnings
        warnings.simplefilter(action='ignore', category=FutureWarning)
        
        stats_pho, data_pho, mod_pho = load_summary_statistics(pspl_phot[targ])
        stats_ast, data_ast, mod_ast = load_summary_statistics(pspl_ast_multiphot[targ])

        for col in stats_pho.itercols():
            if col.info.dtype.kind == 'f':        
                col.info.format = '.3f'        
        for col in stats_ast.itercols():
            if col.info.dtype.kind == 'f':        
                col.info.format = '.3f'        
                
        print('')
        print('******************')
        print(f'*** {targ:s} ***')
        print('******************')
        print('')
        print('Photometry Solutions')
        print(stats_pho['MaxLike_logL', 'MaxLike_u0_amp', 'MaxLike_piE_E', 'MaxLike_piE_N', 'Med_logL', 'Med_u0_amp', 'Med_piE_E', 'Med_piE_N'])

        print('')
        print('Astrometry Solutions')
        print(stats_ast['MaxLike_logL', 'MaxLike_u0_amp', 'MaxLike_piE_E', 'MaxLike_piE_N', 'Med_logL', 'Med_u0_amp', 'Med_piE_E', 'Med_piE_N'])

        print('')
        print('Astrometry Solutions Mass Info')
        print(stats_ast['logZ', 'Mean_mL', 'MaxLike_mL', 'MAP_mL', 'Mean_thetaE', 'MaxLike_thetaE', 'MAP_thetaE', 'Mean_logL', 'MaxLike_logL', 'MAP_logL'])

    return
            
def table_ob120169_phot_astrom():
    # Load up the params file so we know what kind of 
    # data and model we are working with. Note that we 
    # are assuming that all the Nruns are using the same
    # parameters.
    target = 'ob120169'
    
    stats_pho, data_pho, mod_pho = load_summary_statistics(pspl_phot[target])
    stats_ast, data_ast, mod_ast = load_summary_statistics(pspl_ast_multiphot[target])

    # No longer using
              # 'add_err1': '$\\varepsilon_{a,I}$ (mmag)',
              # 'add_err2': '$\\varepsilon_{a,Kp}$ (mmag)',

    labels = {'t0':       '$t_0$ (MJD)',
              'u0_amp':   '$u_0$',
              'tE':       '$t_E$ (days)',
              'piE_E':    '$\pi_{E,E}$',
              'piE_N':    '$\pi_{E,N}$',
              'b_sff1':   '$b_{SFF,I}$',
              'mag_base1': '$I_{base}$ (mag)',
              'b_sff2':   '$b_{SFF,Kp}$',
              'mag_base2': '$Kp_{base}$ (mag)',
              'thetaE':   '$\\theta_E$ (mas)',
              'piS':      '$\pi_S$ (mas)',
              'muS_E':    '$\mu_{S,\\alpha*}$ (mas/yr)',
              'muS_N':    '$\mu_{S,\delta}$ (mas/yr)',
              'xS0_E':    '$x_{S0,\\alpha*}$ (mas)',
              'xS0_N':    '$x_{S0,\delta}$ (mas)',
              'break1':   '',
              'mL':       '$M_L$ ($\msun$)',
              'piL':      '$\pi_L$ (mas)',
              'piRel':    '$\pi_{rel}$ (mas)',
              'muL_E':    '$\mu_{L,\\alpha*}$ (mas/yr)',
              'muL_N':    '$\mu_{L,\delta}$ (mas/yr)',
              'muRel_E':  '$\mu_{rel,\\alpha*}$ (mas/yr)',
              'muRel_N':  '$\mu_{rel,\delta}$ (mas/yr)',
              'mag_src1': '$I_{src}$ (mag)',
              'mag_src2': '$Kp_{src}$ (mag)'
             }
             #  'break2':   '',
             #  'gp_log_sigma1':      '$\log \sigma_{GP, I} (mag)$',
             #  'gp_rho1':            '$\\rho_{GP, I}$ (days)',
             #  'gp_log_omega04_S01': '$\log S_{0, GP, I} \omega_{0, GP, I}^4$ (mag$^2$ days$^{-2}$)',  
             #  'gp_log_omega01':     '$\log \omega_{0, GP, I}$ (days$^{-1}$)'
             # }
             
    scale = {'t0':      1.0,
             'u0_amp':  1.0,
             'tE':      1.0,
             'piE_E':   1.0,
             'piE_N':   1.0,
             'b_sff1':  1.0,
             'mag_src1':1.0,
             'mag_base1':1.0,
             'add_err1':1e3,
             'b_sff2':  1.0,
             'mag_src2':1.0,
             'mag_base2':1.0,
             'add_err2':1e3,
             'thetaE':  1.0,
             'piS':     1.0,
             'muS_E':   1.0,
             'muS_N':   1.0,
             'xS0_E':   1e3,
             'xS0_N':   1e3,
             'mL':      1.0,
             'piL':     1.0,
             'piRel':   1.0,
             'muL_E':   1.0,
             'muL_N':   1.0,
             'muRel_E': 1.0,
             'muRel_N': 1.0,
             'gp_log_sigma1':      1.0,
             'gp_rho1':            1.0,
             'gp_log_omega04_S01': 1.0,
             'gp_log_omega01':     1.0
            }
        
    sig_digits = {'t0':       '0.2f',
                  'u0_amp':   '0.2f',
                  'tE':       '0.1f',
                  'piE_E':    '0.3f',
                  'piE_N':    '0.3f',
                  'b_sff1':   '0.3f',
                  'mag_src1': '0.3f',
                  'mag_base1': '0.3f',
                  'add_err1': '0.1f',
                  'thetaE':   '0.2f',
                  'piS':      '0.3f',
                  'muS_E':    '0.2f',
                  'muS_N':    '0.2f',
                  'xS0_E':    '0.2f',
                  'xS0_N':    '0.2f',
                  'b_sff2':   '0.2f',
                  'mag_src2': '0.2f',
                  'mag_base2': '0.2f',
                  'add_err2': '0.1f',
                  'mL':       '0.2f',
                  'piL':      '0.3f',
                  'piRel':    '0.3f',
                  'muL_E':    '0.2f',
                  'muL_N':    '0.2f',
                  'muRel_E':  '0.2f',
                  'muRel_N':  '0.2f',
                  'gp_log_sigma1':      '0.1f',
                  'gp_rho1':            '0.1f',
                  'gp_log_omega04_S01': '0.1f',
                  'gp_log_omega01':     '0.1f'
                  }

    pho_u0p = 0
    pho_u0m = 1 # doesn't exist
    ast_u0p = 0
    ast_u0m = 1 # doesn't exist

    tab_file = open(paper_dir + target + '_OGLE_phot_ast.txt', 'w')
    tab_file.write('log$\mathcal{L}$ '
                   + '& {0:.2f} & {1:.2f} & {2:.2f} & '.format(stats_pho['MaxLike_logL'][pho_u0p],
                                                               stats_pho['MAP_logL'][pho_u0p],
                                                               stats_pho['Med_logL'][pho_u0p])
                   + '& {0:.2f} & {1:.2f} & {2:.2f} & '.format(stats_ast['MaxLike_logL'][ast_u0p],
                                                               stats_ast['MAP_logL'][ast_u0p],
                                                               stats_ast['Med_logL'][ast_u0p])
                   + '\\\ \n')
    tab_file.write('$\chi^2_{dof}$ ' 
                   + '& {0:.2f} & {1:.2f} & {2:.2f} & '.format(stats_pho['MaxLike_rchi2'][pho_u0p],
                                                               stats_pho['MAP_rchi2'][pho_u0p],
                                                               stats_pho['Med_rchi2'][pho_u0p])
                   + '& {0:.2f} & {1:.2f} & {2:.2f} & '.format(stats_ast['MaxLike_rchi2'][ast_u0p],
                                                               stats_ast['MAP_rchi2'][ast_u0p],
                                                               stats_ast['Med_rchi2'][ast_u0p])
                   + ' \\\ \n')
    tab_file.write('log$\mathcal{Z}$ ' 
                   + '& & & {0:.1f} & '.format(stats_pho['logZ'][pho_u0p])
                   + '& & & {0:.1f} & '.format(stats_ast['logZ'][ast_u0p])
                   + ' \\\ \n')
    tab_file.write('$N_{dof}$ ' 
                   + '& & & {0:.0f} & '.format(stats_pho['N_dof'][pho_u0p])
                   + '& & & {0:.0f} & '.format(stats_ast['N_dof'][ast_u0p])
                   + ' \\\ \n'
                   + r'\hline ' + '\n')

    # Keep track of when we finish off the fitted parameters (vs. additional parameters).
    start_extra_params = False
    start_gp_params = False
    
    for key, label in labels.items():
        # We will have 4 solutions... each has a value and error bar.
        # Setup an easy way to walk through them (and rescale) as necessary.
        # val_dict = [stats_pho, stats_pho, stats_ast]
        # val_mode = [  pho_u0p,   pho_u0m,   ast_u0m]
        val_dict = [stats_pho, stats_ast]
        val_mode = [  pho_u0p,   ast_u0p]

        if (key in mod_ast.additional_param_names) and not start_extra_params:
            tab_file.write('\\tableline\n')
            start_extra_params = True

        if ('gp' in key) and not start_gp_params:
            tab_file.write('\\tableline\n')
            start_gp_params = True
            
        tab_file.write(label)
                           
        for ss in range(len(val_dict)):
            stats = val_dict[ss]
            
            if ('MaxLike_' + key in stats.colnames) and (val_mode[ss] < len(stats)):
                fmt = ' & {0:' + sig_digits[key] + '} & {1:' + sig_digits[key] + '} & '
                fmt +=   '{2:' + sig_digits[key] + '} & [{3:' + sig_digits[key] + '}, {4:' + sig_digits[key] + '}] '
                
                val_mli = stats['MaxLike_' + key][val_mode[ss]]
                val_map = stats['MAP_' + key][val_mode[ss]]
                val_med = stats['Med_' + key][val_mode[ss]]
                elo = stats['lo68_'    + key][val_mode[ss]] - stats['Med_' + key][val_mode[ss]]
                ehi = stats['hi68_'    + key][val_mode[ss]] - stats['Med_' + key][val_mode[ss]]

                val_mli *= scale[key]
                val_map *= scale[key]
                val_med *= scale[key]
                elo *= scale[key]
                ehi *= scale[key]

                tab_file.write(fmt.format(val_mli, val_map, val_med, elo, ehi))
            else:
                fmt = ' & & & & '
                tab_file.write(fmt)

        tab_file.write(' \\\ \n')
    
    return

def table_ob140613_phot_astrom():
    # Load up the params file so we know what kind of 
    # data and model we are working with. Note that we 
    # are assuming that all the Nruns are using the same
    # parameters.
    target = 'ob140613'
    
    stats_pho, data_pho, mod_pho = load_summary_statistics(pspl_phot[target])
    stats_ast, data_ast, mod_ast = load_summary_statistics(pspl_ast_multiphot[target])

    labels = {'t0':       '$t_0$ (MJD)',
              'u0_amp':   '$u_0$',
              'tE':       '$t_E$ (days)',
              'piE_E':    '$\pi_{E,E}$',
              'piE_N':    '$\pi_{E,N}$',
              'b_sff1':   '$b_{SFF,I}$',
              'mag_src1': '$I_{src}$ (mag)',
              'mult_err1': '$\\varepsilon_{m,I}$',
              'thetaE':   '$\\theta_E$ (mas)',
              'piS':      '$\pi_S$ (mas)',
              'muS_E':    '$\mu_{S,\\alpha*}$ (mas/yr)',
              'muS_N':    '$\mu_{S,\delta}$ (mas/yr)',
              'xS0_E':    '$x_{S0,\\alpha*}$ (mas)',
              'xS0_N':    '$x_{S0,\delta}$ (mas)',
              'b_sff2':   '$b_{SFF,Kp}$',
              'mag_src2': '$Kp_{src}$ (mag)',
              'mult_err2': '$\\varepsilon_{m,Kp}$',
              'mL':       '$M_L$ ($\msun$)',
              'piL':      '$\pi_L$ (mas)',
              'piRel':    '$\pi_{rel}$ (mas)',
              'muL_E':    '$\mu_{L,\\alpha*}$ (mas/yr)',
              'muL_N':    '$\mu_{L,\delta}$ (mas/yr)',
              'muRel_E':  '$\mu_{rel,\\alpha*}$ (mas/yr)',
              'muRel_N':  '$\mu_{rel,\delta}$ (mas/yr)'
             }
    scale = {'t0':      1.0,
             'u0_amp':  1.0,
             'tE':      1.0,
             'piE_E':   1.0,
             'piE_N':   1.0,
             'b_sff1':  1.0,
             'mag_src1':1.0,
             'mult_err1':1.0,
             'thetaE':  1.0,
             'piS':     1.0,
             'muS_E':   1.0,
             'muS_N':   1.0,
             'xS0_E':   1e3,
             'xS0_N':   1e3,
             'b_sff2':  1.0,
             'mag_src2':1.0,
             'mult_err2':1.0,
             'mL':      1.0,
             'piL':     1.0,
             'piRel':   1.0,
             'muL_E':   1.0,
             'muL_N':   1.0,
             'muRel_E': 1.0,
             'muRel_N': 1.0
        }
    sig_digits = {'t0':       '0.2f',
                  'u0_amp':   '0.2f',
                  'tE':       '0.1f',
                  'piE_E':    '0.3f',
                  'piE_N':    '0.3f',
                  'b_sff1':   '0.3f',
                  'mag_src1': '0.3f',
                  'mult_err1': '0.1f',
                  'thetaE':   '0.2f',
                  'piS':      '0.3f',
                  'muS_E':    '0.2f',
                  'muS_N':    '0.2f',
                  'xS0_E':    '0.2f',
                  'xS0_N':    '0.2f',
                  'b_sff2':   '0.2f',
                  'mag_src2': '0.2f',
                  'mult_err2': '0.1f',
                  'mL':       '0.2f',
                  'piL':      '0.3f',
                  'piRel':    '0.3f',
                  'muL_E':    '0.2f',
                  'muL_N':    '0.2f',
                  'muRel_E':  '0.2f',
                  'muRel_N':  '0.2f'
                  }

    pho_u0m = 1 # no solution
    pho_u0p = 0 # Best of 3 positive solutions
    ast_u0m = 1 # no solution
    ast_u0p = 0 # best of 3 positive solutions
    
    tab_file = open(paper_dir + target + '_OGLE_phot_ast.txt', 'w')
    tab_file.write('log$\mathcal{L}$ '
                   + '& {0:.2f} & {1:.2f} & {2:.2f} & '.format(stats_pho['MaxLike_logL'][pho_u0p],
                                                               stats_pho['MAP_logL'][pho_u0p], 
                                                               stats_pho['Med_logL'][pho_u0p]) 
                   + '& {0:.2f} & {1:.2f} & {2:.2f} & '.format(stats_ast['MaxLike_logL'][ast_u0p],
                                                               stats_ast['MAP_logL'][ast_u0p],
                                                               stats_ast['Med_logL'][ast_u0p])
                   + ' \\\ \n')
    tab_file.write('$\chi^2_{dof}$ ' 
                   + '& {0:.2f} & {1:.2f} & {2:.2f} & '.format(stats_pho['MaxLike_rchi2'][pho_u0p],
                                                               stats_pho['MAP_rchi2'][pho_u0p],
                                                               stats_pho['Med_rchi2'][pho_u0p])
                   + '& {0:.2f} & {1:.2f} & {2:.2f} & '.format(stats_ast['MaxLike_rchi2'][ast_u0p],
                                                               stats_ast['MAP_rchi2'][ast_u0p],
                                                               stats_ast['Med_rchi2'][ast_u0p])
                   + ' \\\ \n')
    tab_file.write('log$\mathcal{Z}$ ' 
                   + '& & & {0:.1f} & '.format(stats_pho['logZ'][pho_u0p])
                   + '& & & {0:.1f} & '.format(stats_ast['logZ'][ast_u0p])
                   + ' \\\ \n')
    tab_file.write('$N_{dof}$ ' 
                   + '& & & {0:.0f} & '.format(stats_pho['N_dof'][pho_u0p])
                   + '& & & {0:.0f} & '.format(stats_ast['N_dof'][ast_u0p])
                   + ' \\\ \n'
                   + r'\hline ' + '\n')
    
    # Keep track of when we finish off the fitted parameters (vs. additional parameters).
    start_extra_params = False
    
    for key, label in labels.items():
        # We will have 4 solutions... each has a value and error bar.
        # Setup an easy way to walk through them (and rescale) as necessary.
        val_dict = [stats_pho, stats_ast]
        val_mode = [pho_u0p, ast_u0p]

        if (key in mod_ast.additional_param_names) and not start_extra_params:
            tab_file.write('\\tableline\n')
            start_extra_params = True

        tab_file.write(label)
                           
        for ss in range(len(val_dict)):
            stats = val_dict[ss]
            
            if ('MaxLike_' + key in stats.colnames):
                fmt = ' & {0:' + sig_digits[key] + '} & {1:' + sig_digits[key] + '} '
                fmt += '& {2:' + sig_digits[key] + '} & [{3:' + sig_digits[key] + '}, {4:' + sig_digits[key] + '}] '
                
                val_mli = stats['MaxLike_' + key][val_mode[ss]]
                val_map = stats['MAP_' + key][val_mode[ss]]
                val_med = stats['Med_' + key][val_mode[ss]]
                elo = stats['lo68_'    + key][val_mode[ss]] - stats['Med_' + key][val_mode[ss]]
                ehi = stats['hi68_'    + key][val_mode[ss]] - stats['Med_' + key][val_mode[ss]]

                val_mli *= scale[key]
                val_map *= scale[key]
                val_med *= scale[key]
                elo *= scale[key]
                ehi *= scale[key]

                tab_file.write(fmt.format(val_mli, val_map, val_med, elo, ehi))
            else:
                fmt = ' & & & & '
                tab_file.write(fmt)

        tab_file.write(' \\\ \n')
    
    return

def table_ob150029_phot_astrom():
    # Load up the params file so we know what kind of 
    # data and model we are working with. Note that we 
    # are assuming that all the Nruns are using the same
    # parameters.
    target = 'ob150029'
    
    stats_pho, data_pho, mod_pho = load_summary_statistics(pspl_phot[target])
    stats_ast, data_ast, mod_ast = load_summary_statistics(pspl_ast_multiphot[target])

    labels = {'t0':       '$t_0$ (MJD)',
              'u0_amp':   '$u_0$',
              'tE':       '$t_E$ (days)',
              'piE_E':    '$\pi_{E,E}$',
              'piE_N':    '$\pi_{E,N}$',
              'b_sff1':   '$b_{SFF,I}$',
              'mag_src1': '$I_{src}$ (mag)',
              'add_err1': '$\\varepsilon_{a,I}$ (mmag)',
              'thetaE':   '$\\theta_E$ (mas)',
              'piS':      '$\pi_S$ (mas)',
              'muS_E':    '$\mu_{S,\\alpha*}$ (mas/yr)',
              'muS_N':    '$\mu_{S,\delta}$ (mas/yr)',
              'xS0_E':    '$x_{S0,\\alpha*}$ (mas)',
              'xS0_N':    '$x_{S0,\delta}$ (mas)',
              'b_sff2':   '$b_{SFF,Kp}$',
              'mag_src2': '$Kp_{src}$ (mag)',
              'add_err2': '$\\varepsilon_{a,Kp}$ (mmag)',
              'mL':       '$M_L$ ($\msun$)',
              'piL':      '$\pi_L$ (mas)',
              'piRel':    '$\pi_{rel}$ (mas)',
              'muL_E':    '$\mu_{L,\\alpha*}$ (mas/yr)',
              'muL_N':    '$\mu_{L,\delta}$ (mas/yr)',
              'muRel_E':  '$\mu_{rel,\\alpha*}$ (mas/yr)',
              'muRel_N':  '$\mu_{rel,\delta}$ (mas/yr)'
             }
    scale = {'t0':      1.0,
             'u0_amp':  1.0,
             'tE':      1.0,
             'piE_E':   1.0,
             'piE_N':   1.0,
             'b_sff1':  1.0,
             'mag_src1':1.0,
             'add_err1':1e3,
             'thetaE':  1.0,
             'piS':     1.0,
             'muS_E':   1.0,
             'muS_N':   1.0,
             'xS0_E':   1e3,
             'xS0_N':   1e3,
             'b_sff2':  1.0,
             'mag_src2':1.0,
             'add_err2':1e3,
             'mL':      1.0,
             'piL':     1.0,
             'piRel':   1.0,
             'muL_E':   1.0,
             'muL_N':   1.0,
             'muRel_E': 1.0,
             'muRel_N': 1.0
        }
    sig_digits = {'t0':       '0.2f',
                  'u0_amp':   '0.2f',
                  'tE':       '0.1f',
                  'piE_E':    '0.3f',
                  'piE_N':    '0.3f',
                  'b_sff1':   '0.3f',
                  'mag_src1': '0.3f',
                  'add_err1': '0.1f',
                  'thetaE':   '0.2f',
                  'piS':      '0.3f',
                  'muS_E':    '0.2f',
                  'muS_N':    '0.2f',
                  'xS0_E':    '0.2f',
                  'xS0_N':    '0.2f',
                  'b_sff2':   '0.2f',
                  'mag_src2': '0.2f',
                  'add_err2': '0.1f',
                  'mL':       '0.2f',
                  'piL':      '0.3f',
                  'piRel':    '0.3f',
                  'muL_E':    '0.2f',
                  'muL_N':    '0.2f',
                  'muRel_E':  '0.2f',
                  'muRel_N':  '0.2f'
                  }

    pho_u0m = 1
    pho_u0p = 0
    ast_u0m = 0 
    ast_u0p = 1  # No solution
    
    tab_file = open(paper_dir + target + '_OGLE_phot_ast.txt', 'w')
    tab_file.write('log$\mathcal{L}$ '
                   + '& {0:.2f} & {1:.2f} & {2:.2f} & '.format(stats_pho['MaxLike_logL'][pho_u0m],
                                                               stats_pho['MAP_logL'][pho_u0m],
                                                               stats_pho['Med_logL'][pho_u0m])
                   + '& {0:.2f} & {1:.2f} & {2:.2f} & '.format(stats_ast['MaxLike_logL'][ast_u0m],
                                                               stats_ast['MAP_logL'][ast_u0m],
                                                               stats_ast['Med_logL'][ast_u0m])
                   + ' \\\ \n')
    tab_file.write('$\chi^2_{dof}$ ' 
                   + '& {0:.2f} & {1:.2f} & {2:.2f} & '.format(stats_pho['MaxLike_rchi2'][pho_u0m],
                                                               stats_pho['MAP_rchi2'][pho_u0m],
                                                               stats_pho['Med_rchi2'][pho_u0m])
                   + '& {0:.2f} & {1:.2f} & {2:.2f} & '.format(stats_ast['MaxLike_rchi2'][ast_u0m],
                                                               stats_ast['MAP_rchi2'][ast_u0m],
                                                               stats_ast['Med_rchi2'][ast_u0m])
                   + ' \\\ \n')
    tab_file.write('log$\mathcal{Z}$ ' 
                   + '& & & {0:.1f} & '.format(stats_pho['logZ'][pho_u0m])
                   + '& & & {0:.1f} & '.format(stats_ast['logZ'][ast_u0m])
                   + ' \\\ \n')
    tab_file.write('$N_{dof}$ ' 
                   + '& {0:.0f} & & & '.format(stats_pho['N_dof'][pho_u0m])
                   + '& {0:.0f} & & & '.format(stats_ast['N_dof'][ast_u0m])
                   + ' \\\ \n'
                   + r'\hline ' + '\n')
    
    # Keep track of when we finish off the fitted parameters (vs. additional parameters).
    start_extra_params = False

    for key, label in labels.items():
        # We will have 4 solutions... each has a value and error bar.
        # Setup an easy way to walk through them (and rescale) as necessary.
        val_dict = [stats_pho, stats_ast]
        val_mode = [pho_u0m, ast_u0m]

        if (key in mod_ast.additional_param_names) and not start_extra_params:
            tab_file.write('\\tableline\n')
            start_extra_params = True

        tab_file.write(label)
                           
        for ss in range(len(val_dict)):
            stats = val_dict[ss]
            
            if ('MaxLike_' + key in stats.colnames):
                fmt = ' & {0:' + sig_digits[key] + '} & {1:' + sig_digits[key] + '} '
                fmt += '& {2:' + sig_digits[key] + '} & [{3:' + sig_digits[key] + '}, {4:' + sig_digits[key] + '}] '
                
                val_mli = stats['MaxLike_' + key][val_mode[ss]]
                val_map = stats['MAP_' + key][val_mode[ss]]
                val_med = stats['Med_' + key][val_mode[ss]]
                elo = stats['lo68_'    + key][val_mode[ss]] - stats['Med_' + key][val_mode[ss]]
                ehi = stats['hi68_'    + key][val_mode[ss]] - stats['Med_' + key][val_mode[ss]]

                val_mli *= scale[key]
                val_map *= scale[key]
                val_med *= scale[key]
                elo *= scale[key]
                ehi *= scale[key]

                tab_file.write(fmt.format(val_mli, val_map, val_med, elo, ehi))
            else:
                fmt = ' & & & & '
                tab_file.write(fmt)

        tab_file.write(' \\\ \n')
    
    return

def table_ob150211_phot():
    """
    Print the latex table for the photometry-only fit.
    """
    target = 'ob150211'
    
    stats_pho, data_pho, mod_pho = load_summary_statistics(pspl_phot[target])

    # No longer using
              # 'add_err1': '$\\varepsilon_{a,I}$ (mmag)',
              # 'add_err2': '$\\varepsilon_{a,Kp}$ (mmag)',

    # This dictionary sets the order of the parameters in the table.
    # If I want a horizontal line in between the two, I just add 'break#' where
    # the "#" symbol is just any arbitrary number. 
    labels = {'t0':       '$t_0$ (MJD)',
              'u0_amp':   '$u_0$',
              'tE':       '$t_E$ (days)',
              'piE_E':    '$\pi_{E,E}$',
              'piE_N':    '$\pi_{E,N}$',
              'b_sff1':   '$b_{SFF,I}$',
              'mag_base1': '$I_{base}$ (mag)',
              'break1':   '',
              'mag_src1': '$I_{src}$ (mag)',
             }
        
    scale = {'t0':      1.0,
             'u0_amp':  1.0,
             'tE':      1.0,
             'piE_E':   1.0,
             'piE_N':   1.0,
             'b_sff1':  1.0,
             'mag_src1':1.0,
             'mag_base1':1.0,
             'add_err1':1e3,
             'thetaE':  1.0,
             'piS':     1.0,
             'muS_E':   1.0,
             'muS_N':   1.0,
             'xS0_E':   1e3,
             'xS0_N':   1e3,
             'b_sff2':  1.0,
             'mag_src2':1.0,
             'mag_base2':1.0,
             'add_err2':1e3,
             'mL':      1.0,
             'piL':     1.0,
             'piRel':   1.0,
             'muL_E':   1.0,
             'muL_N':   1.0,
             'muRel_E': 1.0,
             'muRel_N': 1.0,
             'gp_log_sigma1':      1.0,
             'gp_rho1':            1.0,
             'gp_log_omega04_S01': 1.0,
             'gp_log_omega01':     1.0
        }
    sig_digits = {'t0':       '0.2f',
                  'u0_amp':   '0.2f',
                  'tE':       '0.1f',
                  'piE_E':    '0.3f',
                  'piE_N':    '0.3f',
                  'b_sff1':   '0.3f',
                  'mag_src1': '0.3f',
                  'mag_base1': '0.3f',
                  'add_err1': '0.1f',
                  'thetaE':   '0.2f',
                  'piS':      '0.3f',
                  'muS_E':    '0.2f',
                  'muS_N':    '0.2f',
                  'xS0_E':    '0.2f',
                  'xS0_N':    '0.2f',
                  'b_sff2':   '0.2f',
                  'mag_src2': '0.2f',
                  'mag_base2': '0.2f',
                  'add_err2': '0.1f',
                  'mL':       '0.1f',
                  'piL':      '0.3f',
                  'piRel':    '0.3f',
                  'muL_E':    '0.2f',
                  'muL_N':    '0.2f',
                  'muRel_E':  '0.2f',
                  'muRel_N':  '0.2f',
                  'gp_log_sigma1':      '0.1f',
                  'gp_rho1':            '0.1f',
                  'gp_log_omega04_S01': '0.1f',
                  'gp_log_omega01':     '0.1f'
                  }

    pho_u0m = 0
    pho_u0p = 1

    tab_file = open(paper_dir + target + '_OGLE_phot.txt', 'w')
    tab_file.write('log$\mathcal{L}$ '
                   + '& {0:.2f} & {1:.2f} & {2:.2f} & '.format(stats_pho['MaxLike_logL'][pho_u0p],
                                                               stats_pho['MAP_logL'][pho_u0p],
                                                               stats_pho['Med_logL'][pho_u0p])
                   + '& {0:.2f} & {1:.2f} & {2:.2f} & '.format(stats_pho['MaxLike_logL'][pho_u0m],
                                                               stats_pho['MAP_logL'][pho_u0m],
                                                               stats_pho['Med_logL'][pho_u0m])
                   + ' \\\ \n')
    tab_file.write('$\chi^2_{dof}$ ' 
                   + '& {0:.2f} & {1:.2f} & {2:.2f} & '.format(stats_pho['MaxLike_rchi2'][pho_u0p],
                                                               stats_pho['MAP_rchi2'][pho_u0p],
                                                               stats_pho['Med_rchi2'][pho_u0p])
                   + '& {0:.2f} & {1:.2f} & {2:.2f} & '.format(stats_pho['MaxLike_rchi2'][pho_u0m],
                                                               stats_pho['MAP_rchi2'][pho_u0m],
                                                               stats_pho['Med_rchi2'][pho_u0m])
                   + ' \\\ \n')
    tab_file.write('log$\mathcal{Z}$ ' 
                   + '& & & {0:.1f} & '.format(stats_pho['logZ'][pho_u0p])
                   + '& & & {0:.1f} & '.format(stats_pho['logZ'][pho_u0m])
                   + ' \\\ \n')
    tab_file.write('$N_{dof}$ ' 
                   + '& & & {0:.0f} & '.format(stats_pho['N_dof'][pho_u0p])
                   + '& & & {0:.0f} & '.format(stats_pho['N_dof'][pho_u0m])
                   + ' \\\ \n'
                   + r'\hline ' + '\n')
    
    # Keep track of when we finish off the fitted parameters (vs. additional parameters).
    start_extra_params = False
    
    for key, label in labels.items():
        # We will have 4 solutions... each has a value and error bar.
        # Setup an easy way to walk through them (and rescale) as necessary.
        val_dict = [stats_pho, stats_pho]
        val_mode = [pho_u0p, pho_u0m]

        if 'break' in key:
            tab_file.write('\\tableline\n')

        tab_file.write(label)
                           
        for ss in range(len(val_dict)):
            stats = val_dict[ss]
            
            if ('MaxLike_' + key in stats.colnames):
                fmt = ' & {0:' + sig_digits[key] + '} & {1:' + sig_digits[key] + '} '
                fmt += '& {2:' + sig_digits[key] + '} & [{3:' + sig_digits[key] + '}, {4:' + sig_digits[key] + '}] '
                
                val_mli = stats['MaxLike_' + key][val_mode[ss]]
                val_map = stats['MAP_' + key][val_mode[ss]]
                val_med = stats['Med_' + key][val_mode[ss]]
                elo = stats['lo68_'    + key][val_mode[ss]] - stats['Med_' + key][val_mode[ss]]
                ehi = stats['hi68_'    + key][val_mode[ss]] - stats['Med_' + key][val_mode[ss]]

                val_mli *= scale[key]
                val_map *= scale[key]
                val_med *= scale[key]
                elo *= scale[key]
                ehi *= scale[key]

                tab_file.write(fmt.format(val_mli, val_map, val_med, elo, ehi))
            else:
                fmt = ' & & & & '
                tab_file.write(fmt)

        tab_file.write(' \\\ \n')
    
    return

    
    
    
def table_ob150211_phot_astrom():
    # Load up the params file so we know what kind of 
    # data and model we are working with. Note that we 
    # are assuming that all the Nruns are using the same
    # parameters.
    target = 'ob150211'
    
    stats_ast, data_ast, mod_ast = load_summary_statistics(pspl_ast_multiphot[target])

    # No longer using
              # 'add_err1': '$\\varepsilon_{a,I}$ (mmag)',
              # 'add_err2': '$\\varepsilon_{a,Kp}$ (mmag)',

    # This dictionary sets the order of the parameters in the table.
    # If I want a horizontal line in between the two, I just add 'break#' where
    # the "#" symbol is just any arbitrary number. 
    labels = {'t0':       '$t_0$ (MJD)',
              'u0_amp':   '$u_0$',
              'tE':       '$t_E$ (days)',
              'piE_E':    '$\pi_{E,E}$',
              'piE_N':    '$\pi_{E,N}$',
              'b_sff1':   '$b_{SFF,I}$',
              'mag_base1': '$I_{base}$ (mag)',
              'b_sff2':   '$b_{SFF,Kp}$',
              'mag_base2': '$Kp_{base}$ (mag)',
              'thetaE':   '$\\theta_E$ (mas)',
              'piS':      '$\pi_S$ (mas)',
              'muS_E':    '$\mu_{S,\\alpha*}$ (mas/yr)',
              'muS_N':    '$\mu_{S,\delta}$ (mas/yr)',
              'xS0_E':    '$x_{S0,\\alpha*}$ (mas)',
              'xS0_N':    '$x_{S0,\delta}$ (mas)',
              'break1':   '',
              'mL':       '$M_L$ ($\msun$)',
              'piL':      '$\pi_L$ (mas)',
              'piRel':    '$\pi_{rel}$ (mas)',
              'muL_E':    '$\mu_{L,\\alpha*}$ (mas/yr)',
              'muL_N':    '$\mu_{L,\delta}$ (mas/yr)',
              'muRel_E':  '$\mu_{rel,\\alpha*}$ (mas/yr)',
              'muRel_N':  '$\mu_{rel,\delta}$ (mas/yr)',
              'mag_src1': '$I_{src}$ (mag)',
              'mag_src2': '$Kp_{src}$ (mag)'
              # 'break2':   '',
              # 'gp_log_sigma1':      '$\log \sigma_{GP, I} (mag)$',
              # 'gp_rho1':            '$\\rho_{GP, I}$ (days)',
              # 'gp_log_omega04_S01': '$\log S_{0, GP, I} \omega_{0, GP, I}^4$ (mag$^2$ days$^{-2}$)',  
              # 'gp_log_omega01':     '$\log \omega_{0, GP, I}$ (days$^{-1}$)'
             }
        
    scale = {'t0':      1.0,
             'u0_amp':  1.0,
             'tE':      1.0,
             'piE_E':   1.0,
             'piE_N':   1.0,
             'b_sff1':  1.0,
             'mag_src1':1.0,
             'mag_base1':1.0,
             'add_err1':1e3,
             'thetaE':  1.0,
             'piS':     1.0,
             'muS_E':   1.0,
             'muS_N':   1.0,
             'xS0_E':   1e3,
             'xS0_N':   1e3,
             'b_sff2':  1.0,
             'mag_src2':1.0,
             'mag_base2':1.0,
             'add_err2':1e3,
             'mL':      1.0,
             'piL':     1.0,
             'piRel':   1.0,
             'muL_E':   1.0,
             'muL_N':   1.0,
             'muRel_E': 1.0,
             'muRel_N': 1.0,
             'gp_log_sigma1':      1.0,
             'gp_rho1':            1.0,
             'gp_log_omega04_S01': 1.0,
             'gp_log_omega01':     1.0
        }
    sig_digits = {'t0':       '0.2f',
                  'u0_amp':   '0.2f',
                  'tE':       '0.1f',
                  'piE_E':    '0.3f',
                  'piE_N':    '0.3f',
                  'b_sff1':   '0.3f',
                  'mag_src1': '0.3f',
                  'mag_base1': '0.3f',
                  'add_err1': '0.1f',
                  'thetaE':   '0.2f',
                  'piS':      '0.3f',
                  'muS_E':    '0.2f',
                  'muS_N':    '0.2f',
                  'xS0_E':    '0.2f',
                  'xS0_N':    '0.2f',
                  'b_sff2':   '0.2f',
                  'mag_src2': '0.2f',
                  'mag_base2': '0.2f',
                  'add_err2': '0.1f',
                  'mL':       '0.1f',
                  'piL':      '0.3f',
                  'piRel':    '0.3f',
                  'muL_E':    '0.2f',
                  'muL_N':    '0.2f',
                  'muRel_E':  '0.2f',
                  'muRel_N':  '0.2f',
                  'gp_log_sigma1':      '0.1f',
                  'gp_rho1':            '0.1f',
                  'gp_log_omega04_S01': '0.1f',
                  'gp_log_omega01':     '0.1f'
                  }

    ast_u0p_hi = 0  # hi muRel
    ast_u0p_lo = 1  # lo muRel
    ast_u0m_al = 2  # all neg u0

    tab_file = open(paper_dir + target + '_OGLE_phot_ast.txt', 'w')
    tab_file.write('log$\mathcal{L}$ '
                   + '& {0:.2f} & {1:.2f} & {2:.2f} & '.format(stats_ast['MaxLike_logL'][ast_u0p_hi],
                                                               stats_ast['MAP_logL'][ast_u0p_hi], 
                                                               stats_ast['Med_logL'][ast_u0p_hi])
                   + '& {0:.2f} & {1:.2f} & {2:.2f} & '.format(stats_ast['MaxLike_logL'][ast_u0p_lo],
                                                               stats_ast['MAP_logL'][ast_u0p_lo],
                                                               stats_ast['Med_logL'][ast_u0p_lo])
                   + '& {0:.2f} & {1:.2f} & {2:.2f} & '.format(stats_ast['MaxLike_logL'][ast_u0m_al],
                                                               stats_ast['MAP_logL'][ast_u0m_al],
                                                               stats_ast['Med_logL'][ast_u0m_al])
                   + ' \\\ \n')
    tab_file.write('$\chi^2_{dof}$ ' 
                   + '& {0:.2f} & {1:.2f} & {2:.2f} & '.format(stats_ast['MaxLike_rchi2'][ast_u0p_hi],
                                                               stats_ast['MAP_rchi2'][ast_u0p_hi],
                                                               stats_ast['Med_rchi2'][ast_u0p_hi])
                   + '& {0:.2f} & {1:.2f} & {2:.2f} & '.format(stats_ast['MaxLike_rchi2'][ast_u0p_lo],
                                                               stats_ast['MAP_rchi2'][ast_u0p_lo],
                                                               stats_ast['Med_rchi2'][ast_u0p_lo])
                   + '& {0:.2f} & {1:.2f} & {2:.2f} & '.format(stats_ast['MaxLike_rchi2'][ast_u0m_al],
                                                               stats_ast['MAP_rchi2'][ast_u0m_al],
                                                               stats_ast['Med_rchi2'][ast_u0m_al])
                   + ' \\\ \n')
    tab_file.write('log$\mathcal{Z}$ ' 
                   + '& & & {0:.1f} & '.format(stats_ast['logZ'][ast_u0p_hi])
                   + '& & & {0:.1f} & '.format(stats_ast['logZ'][ast_u0p_lo])
                   + '& & & {0:.1f} & '.format(stats_ast['logZ'][ast_u0m_al])
                   + ' \\\ \n')
    tab_file.write('$N_{dof}$ ' 
                   + '& & & {0:.0f} & '.format(stats_ast['N_dof'][ast_u0p_hi])
                   + '& & & {0:.0f} & '.format(stats_ast['N_dof'][ast_u0p_lo])
                   + '& & & {0:.0f} & '.format(stats_ast['N_dof'][ast_u0m_al])
                   + ' \\\ \n')
    prior_vol, logBF = calc_bayes_factor('ob150211')
    tab_file.write('$\pi-$volume ' 
                   + '& & & {0:.2f} & '.format(prior_vol[ast_u0p_hi])
                   + '& & & {0:.2f} & '.format(prior_vol[ast_u0p_lo])
                   + '& & & {0:.2f} & '.format(prior_vol[ast_u0m_al])
                   + ' \\\ \n'
                   + r'\hline ' + '\n')
    
    # Keep track of when we finish off the fitted parameters (vs. additional parameters).
    start_extra_params = False
    
    for key, label in labels.items():
        # We will have 4 solutions... each has a value and error bar.
        # Setup an easy way to walk through them (and rescale) as necessary.
        val_dict = [stats_ast, stats_ast, stats_ast]
        val_mode = [ast_u0p_hi, ast_u0p_lo, ast_u0m_al]

        if 'break' in key:
            tab_file.write('\\tableline\n')

        tab_file.write(label)
                           
        for ss in range(len(val_dict)):
            stats = val_dict[ss]
            
            if ('MaxLike_' + key in stats.colnames):
                fmt = ' & {0:' + sig_digits[key] + '} & {1:' + sig_digits[key] + '} '
                fmt += '& {2:' + sig_digits[key] + '} & [{3:' + sig_digits[key] + '}, {4:' + sig_digits[key] + '}] '
                
                val_mli = stats['MaxLike_' + key][val_mode[ss]]
                val_map = stats['MAP_' + key][val_mode[ss]]
                val_med = stats['Med_' + key][val_mode[ss]]
                elo = stats['lo68_'    + key][val_mode[ss]] - stats['Med_' + key][val_mode[ss]]
                ehi = stats['hi68_'    + key][val_mode[ss]] - stats['Med_' + key][val_mode[ss]]

                val_mli *= scale[key]
                val_map *= scale[key]
                val_med *= scale[key]
                elo *= scale[key]
                ehi *= scale[key]

                tab_file.write(fmt.format(val_mli, val_map, val_med, elo, ehi))
            else:
                fmt = ' & & & & '
                tab_file.write(fmt)

        tab_file.write(' \\\ \n')
    
    return


def results_best_params_all():
    targets = ['ob120169', 'ob140613', 'ob150029', 'ob150211']

    params = ['t0', 'u0_amp', 'tE', 'thetaE', 'mL',
              'piE_E', 'piE_N', 
              'muRel_E', 'muRel_N', 'muS_E', 'muS_N', 'muL_E', 'muL_N',
              'piS', 'piL', 'piRel',
              'xS0_E', 'xS0_N',
              'b_sff1', 'mag_src1',
              'b_sff2', 'mag_src2']
        # , 'add_err1', 'add_err2'
        
    fmt = '   {0:13s} = {1:12.4f} {2:12.4f} {3:12.4f}  [{4:12.4f} - {5:12.4f}]  [{6:12.4f} - {7:12.4f}]'

    def get_68CI(samples, weights):
        sumweights = np.sum(weights)
        tmp_weights = weights / sumweights
        
        sig1 = 0.682689
        sig1_lo = (1. - sig1) / 2.
        sig1_hi = 1. - sig1_lo

        tmp = model_fitter.weighted_quantile(samples, [sig1_lo, sig1_hi],
                                             sample_weight = tmp_weights)

        return tmp[0], tmp[1]

    def get_99CI(samples, weights):
        sumweights = np.sum(weights)
        tmp_weights = weights / sumweights
        
        sig1 = 0.9973
        sig1_lo = (1. - sig1) / 2.
        sig1_hi = 1. - sig1_lo

        tmp = model_fitter.weighted_quantile(samples, [sig1_lo, sig1_hi],
                                             sample_weight = tmp_weights)

        return tmp[0], tmp[1]
    
        

    for targ in targets:
        print('')
        print('**********')
        print('Best-Fit (Max L, MAP, Median, 68.3% CI, 99.7% CI) Parameters for ', targ)
        print('**********')
        stats_ast, data_ast, mod_ast = load_summary_statistics(pspl_ast_multiphot[targ])

        # Find the best-fit solution
        mdx = pspl_ast_multiphot_mode[targ]

        print('*  NOTE Best solution is idx = ', mdx)

        for par in params:
            if targ == 'ob140613' and par.startswith('add'):
                par = par.replace('add', 'mult')
                
            print(fmt.format(par, stats_ast['MaxLike_' + par][mdx],
                                  stats_ast['MAP_' + par][mdx],
                                  stats_ast['Med_' + par][mdx],
                                  stats_ast['lo68_' + par][mdx],
                                  stats_ast['hi68_' + par][mdx],
                                  stats_ast['lo99_' + par][mdx],
                                  stats_ast['hi99_' + par][mdx]))

        # Print a few more summary params
        muRel_mli = np.hypot(stats_ast['MaxLike_muRel_E'][mdx], stats_ast['MaxLike_muRel_N'][mdx])
        muRel_map = np.hypot(stats_ast['MAP_muRel_E'][mdx], stats_ast['MAP_muRel_N'][mdx])
        muRel_med = np.hypot(stats_ast['Med_muRel_E'][mdx], stats_ast['Med_muRel_N'][mdx])
        
        piE_mli = np.hypot(stats_ast['MaxLike_piE_E'][mdx], stats_ast['MaxLike_piE_N'][mdx])
        piE_map = np.hypot(stats_ast['MAP_piE_E'][mdx], stats_ast['MAP_piE_N'][mdx])
        piE_med = np.hypot(stats_ast['Med_piE_E'][mdx], stats_ast['Med_piE_N'][mdx])
        
        u0_mli = stats_ast['MaxLike_u0_amp'][mdx]
        u0_map = stats_ast['MAP_u0_amp'][mdx]
        u0_med = stats_ast['Med_u0_amp'][mdx]
        
        thetaE_mli = stats_ast['MaxLike_thetaE'][mdx]
        thetaE_map = stats_ast['MAP_thetaE'][mdx]
        thetaE_med = stats_ast['Med_thetaE'][mdx]
        
        Amp_mli = (u0_mli**2 + 2.0) / (np.abs(u0_mli) * np.sqrt(u0_mli**2 + 4.0))
        Amp_map = (u0_map**2 + 2.0) / (np.abs(u0_map) * np.sqrt(u0_map**2 + 4.0))
        Amp_med = (u0_med**2 + 2.0) / (np.abs(u0_med) * np.sqrt(u0_med**2 + 4.0))
        
        deltaC_mli = thetaE_mli * 2**0.5 / 4.0
        deltaC_map = thetaE_map * 2**0.5 / 4.0
        deltaC_med = thetaE_med * 2**0.5 / 4.0

        # Actually fetch the samples to calculate the errors on these.
        fitter, data = get_data_and_fitter(pspl_ast_multiphot[targ])
        samps_list = fitter.load_mnest_modes()
        samps = samps_list[mdx]
        
        muRel_samp = np.hypot(samps['muRel_E'], samps['muRel_N'])
        piE_samp = np.hypot(samps['piE_E'], samps['piE_N'])

        # Add thetaE (from log thetaE)
        if ('log10_thetaE' in samps.colnames) and ('thetaE' not in samps.colnames):
            thetaE_samp = 10**samps['log10_thetaE']
        else:
            thetaE_samp = samps['thetaE']
        u0 = samps['u0_amp']
        Amp_samp = (u0**2 + 2.0) / (np.abs(u0) * np.sqrt(u0**2 + 4.0))
        deltaC_samp = thetaE_samp * 2**0.5 / 4.0

        muRel_lo1, muRel_hi1 = get_68CI(muRel_samp, samps['weights'])
        piE_lo1, piE_hi1 = get_68CI(piE_samp, samps['weights'])
        Amp_lo1, Amp_hi1 = get_68CI(Amp_samp, samps['weights'])
        deltaC_lo1, deltaC_hi1 = get_68CI(deltaC_samp, samps['weights'])

        muRel_lo3, muRel_hi3 = get_99CI(muRel_samp, samps['weights'])
        piE_lo3, piE_hi3 = get_99CI(piE_samp, samps['weights'])
        Amp_lo3, Amp_hi3 = get_99CI(Amp_samp, samps['weights'])
        deltaC_lo3, deltaC_hi3 = get_99CI(deltaC_samp, samps['weights'])
        
        print(fmt.format('muRel', muRel_mli, muRel_map, muRel_med, muRel_lo1, muRel_hi1, muRel_lo3, muRel_hi3))
        print(fmt.format('piE', piE_mli, piE_map, piE_med, piE_lo1, piE_hi1, piE_lo3, piE_hi3))
        print(fmt.format('A', Amp_mli, Amp_map, Amp_med, Amp_lo1, Amp_hi1, Amp_lo3, Amp_hi3))
        print(fmt.format('deltaC_max', deltaC_mli, deltaC_map, deltaC_med, deltaC_lo1, deltaC_hi1, deltaC_lo3, deltaC_hi3))
    
    return

def plot_mass_posterior(target):
    fontsize1 = 18
    fontsize2 = 14

    fitter, data = get_data_and_fitter(pspl_ast_multiphot[target])
    stats = calc_summary_statistics(fitter)
    tab = fitter.load_mnest_modes()

    # Find the best-fit solution
    mdx = pspl_ast_multiphot_mode[target]
    tab = tab[mdx]
    stats = stats[mdx]
    
    plt.figure(1)
    plt.clf()
    plt.subplots_adjust(bottom = 0.15)
    
    bins = np.logspace(-3, 2, 100)

    n, b = np.histogram(tab['mL'], bins = bins,
                        weights = tab['weights'], density = False)

    # n = norm_kde(n, 10.)
    b0 = 0.5 * (b[1:] + b[:-1])

    # n, b, _ = plt.hist(b0, bins = b, weights = n)
    plt.plot(b0, n, drawstyle='steps-mid')

    plt.xlabel('Lens Mass $(M_\odot)$', fontsize=fontsize1, labelpad=10)
    plt.ylabel('Posterior Probability', fontsize=fontsize1, labelpad=10)
    plt.xticks(fontsize=fontsize2)
    plt.yticks(fontsize=fontsize2)

    plt.xscale('log')

    plt.axvline(stats['MaxLike_mL'], color='k', linestyle='-', lw = 2)
    plt.axvline(stats['lo68_mL'], color='k', linestyle='--', lw = 2)
    plt.axvline(stats['hi68_mL'], color='k', linestyle='--', lw = 2)
    plt.xlim(0.1, 60)
    plt.ylim(0, 1.1*n.max())
    plt.title(target)

    plt.savefig(paper_dir + target + '_mass_posterior.png')

    return

def get_CIs(samples, weights):
    sumweights = np.sum(weights)
    tmp_weights = weights / sumweights
    
    sig1 = 0.682689
    sig2 = 0.9545
    sig3 = 0.9973
    sig1_lo = (1. - sig1) / 2.
    sig2_lo = (1. - sig2) / 2.
    sig3_lo = (1. - sig3) / 2.
    sig1_hi = 1. - sig1_lo
    sig2_hi = 1. - sig2_lo
    sig3_hi = 1. - sig3_lo

    tmp = model_fitter.weighted_quantile(samples, [sig1_lo, sig1_hi, sig2_lo, sig2_hi, sig3_lo, sig3_hi],
                                         sample_weight = tmp_weights)

    return tmp



def plot_all_mass_posteriors(sol_mode='best'):
    fontsize1 = 18
    fontsize2 = 14
    
    targets = ['ob120169', 'ob140613', 'ob150029', 'ob150211']
    colors = {'ob120169': 'purple',
              'ob140613': 'red',
              'ob150029': 'darkorange',
              'ob150211': 'black'}
    kde_bw = {'ob120169': 0.1,
              'ob140613': 0.2,
              'ob150029': 0.1,
              'ob150211': 0.1}

    plt.close(1)
    plt.figure(1)
    plt.clf()
    plt.subplots_adjust(bottom = 0.15)

    for target in targets:
        fitter, data = get_data_and_fitter(pspl_ast_multiphot[target])
        stats = calc_summary_statistics(fitter)
        tab = fitter.load_mnest_modes()
        
        # Select out the best mode.
        mode = pspl_ast_multiphot_mode[target]
        stats = stats[mode]

        if sol_mode == 'best':
            tab = tab[mode]
        elif sol_mode == 'global':
            tab = fitter.load_mnest_results()

        # KDE
        kde = scipy.stats.gaussian_kde(np.log10(tab['mL']),
                                           weights=tab['weights'],
                                           bw_method=kde_bw[target])
        kde_mass = np.linspace(-2, 2, 500)
        # kde_mass = np.arange(0.0, 10, 0.01)
        kde_prob = kde(kde_mass)
        kde_bin_size = np.zeros(len(kde_prob), dtype=float)
        kde_bin_size[:-1] = np.diff(kde_mass)
        kde_bin_size[-1] = kde_bin_size[-2]
        kde_norm = (kde_prob * kde_bin_size).sum()
        kde_prob /= kde_norm

        # Normalize kde_prob
        # kde_prob /= kde_prob.max()
        plt.plot(kde_mass, kde_prob, color=colors[target], linestyle='-', label=target.upper())        

        # Bins for the histogram
        bins = np.linspace(-2, 2, 80)
        # bins = np.linspace(00, 10, 0.1)
        # bins = np.arange(0.0, 10, 0.1)

        n, b = np.histogram(np.log10(tab['mL']), bins = bins, 
                            weights = tab['weights'], density = True)
        b0 = 0.5 * (b[1:] + b[:-1])
        # n /= n.max()

        # plt.plot(b[:-1], n, drawstyle='steps-mid',
        #              color=colors[target], label=target)

        # plt.xscale('log')

        # plt.axvline(stats['MaxLike_mL'], color=colors[target], linestyle='--', lw = 2)
        plt.axvline(np.log10(stats['Med_mL']), color=colors[target], linestyle='--', lw = 2)
        # plt.axvline(np.log10(stats['MAP_mL']), color=colors[target], linestyle='--', lw = 2)
        # plt.axvline(np.log10(stats['MaxLike_mL']), color=colors[target], linestyle='-.', lw = 2)
        # plt.axvline(stats['lo68_mL'], color=colors[target], linestyle='--', lw = 2)
        # plt.axvline(stats['hi68_mL'], color=colors[target], linestyle='--', lw = 2)

        conf_int = get_CIs(tab['mL'], tab['weights'])
        print('Best-Fit Lens Mass MaxLike = {0:.2f} for {1:s}'.format(stats['MaxLike_mL'], target))
        print('Best-Fit Lens Mass MAP     = {0:.2f} for {1:s}'.format(stats['MAP_mL'], target))
        print('Best-Fit Lens Mass Median  = {0:.2f} for {1:s}'.format(stats['Med_mL'], target))
        print('Best-Fit Lens Mass Mean    = {0:.2f} for {1:s}'.format(stats['Mean_mL'], target))
        print('          68.3% CI = [{0:6.2f} - {1:6.2f}]'.format(conf_int[0], conf_int[1]))
        print('          95.5% CI = [{0:6.2f} - {1:6.2f}]'.format(conf_int[2], conf_int[3]))
        print('          99.7% CI = [{0:6.2f} - {1:6.2f}]'.format(conf_int[4], conf_int[5]))
        
    plt.xlabel('Lens Mass $(M_\odot)$', fontsize=fontsize1, labelpad=10)
    plt.ylabel('Posterior Probability', fontsize=fontsize1, labelpad=10)
    plt.xticks(fontsize=fontsize2)
    plt.yticks(fontsize=fontsize2)
    plt.xlim(-1.1, 2.0)
    plt.tick_params(axis='x', which='minor')
    # plt.ylim(0, 2)
    # plt.xlim(0, 5)
    plt.ylim(0, 5)
    plt.legend()

    def pow_10(x, pos):
        """The two args are the value and tick position.
        Label ticks with the product of the exponentiation"""
        return '%0.1f' % (10**x)

    formatter = plt.FuncFormatter(pow_10)
    plt.gca().xaxis.set_major_formatter(formatter)

    x_minor = matplotlib.ticker.FixedLocator([np.log10(mm) for mm in np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9,
                                                                               1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0,
                                                                               10, 20, 30, 40, 50, 60, 70, 80, 90, 100])])
    plt.gca().xaxis.set_minor_locator(x_minor)
    plt.gca().xaxis.set_minor_formatter(plt.NullFormatter())

    if sol_mode == 'global':
        plt.savefig(paper_dir + 'all_mass_posteriors_' + mode + '.png')
    else:
        plt.savefig(paper_dir + 'all_mass_posteriors.png')

    return

def plot_ob150211_mass_posterior_modes():
    fontsize1 = 18
    fontsize2 = 14
    
    target = 'ob150211'
    colors = ['black', 'purple', 'blue']
    kde_bw = {'ob150211': 0.1}

    plt.close(1)
    plt.figure(1)
    plt.clf()
    plt.subplots_adjust(bottom = 0.15)
    
    fitter, data = get_data_and_fitter(pspl_ast_multiphot[target])
    stats_list = calc_summary_statistics(fitter)
    tab_list = fitter.load_mnest_modes()

    for ii in range(len(stats_list)):
        stats = stats_list[ii]
        tab = tab_list[ii]

        # KDE
        kde = scipy.stats.gaussian_kde(np.log10(tab['mL']),
                                           weights=tab['weights'],
                                           bw_method=kde_bw[target])
        kde_mass = np.linspace(-2, 2, 500)
        # kde_mass = np.arange(0.0, 10, 0.01)
        kde_prob = kde(kde_mass)
        kde_bin_size = np.zeros(len(kde_prob), dtype=float)
        kde_bin_size[:-1] = np.diff(kde_mass)
        kde_bin_size[-1] = kde_bin_size[-2]
        kde_norm = (kde_prob * kde_bin_size).sum()
        kde_prob /= kde_norm

        # Normalize kde_prob
        # kde_prob /= kde_prob.max()
        leg_label = f'{target.upper()}, '
        if ii == 0:
            leg_label += '$u_0 > 0$, high $\mu_{{rel}}$'
        elif ii == 1:
            leg_label += '$u_0 > 0$, low $\mu_{{rel}}$'
        elif ii == 2:
            leg_label += '$u_0 <= 0$'
        plt.plot(kde_mass, kde_prob, color=colors[ii], linestyle='-', label=leg_label)

        # Bins for the histogram
        bins = np.linspace(-2, 2, 80)

        n, b = np.histogram(np.log10(tab['mL']), bins = bins, 
                            weights = tab['weights'], density = True)
        b0 = 0.5 * (b[1:] + b[:-1])


        plt.axvline(np.log10(stats['Med_mL']), color=colors[ii], linestyle='--', lw = 2)

        conf_int = get_CIs(tab['mL'], tab['weights'])
        print('Best-Fit Lens Mass MaxLike = {0:.2f} for {1:s}'.format(stats['MaxLike_mL'], target))
        print('Best-Fit Lens Mass MAP     = {0:.2f} for {1:s}'.format(stats['MAP_mL'], target))
        print('Best-Fit Lens Mass Median  = {0:.2f} for {1:s}'.format(stats['Med_mL'], target))
        print('Best-Fit Lens Mass Mean    = {0:.2f} for {1:s}'.format(stats['Mean_mL'], target))
        print('          68.3% CI = [{0:6.2f} - {1:6.2f}]'.format(conf_int[0], conf_int[1]))
        print('          95.5% CI = [{0:6.2f} - {1:6.2f}]'.format(conf_int[2], conf_int[3]))
        print('          99.7% CI = [{0:6.2f} - {1:6.2f}]'.format(conf_int[4], conf_int[5]))
        
    plt.xlabel('Lens Mass $(M_\odot)$', fontsize=fontsize1, labelpad=10)
    plt.ylabel('Posterior Probability', fontsize=fontsize1, labelpad=10)
    plt.xticks(fontsize=fontsize2)
    plt.yticks(fontsize=fontsize2)
    plt.xlim(-1.1, 2.0)
    plt.tick_params(axis='x', which='minor')
    # plt.ylim(0, 2)
    # plt.xlim(0, 5)
    plt.ylim(0, 5)
    plt.legend()

    def pow_10(x, pos):
        """The two args are the value and tick position.
        Label ticks with the product of the exponentiation"""
        return '%0.1f' % (10**x)

    formatter = plt.FuncFormatter(pow_10)
    plt.gca().xaxis.set_major_formatter(formatter)

    x_minor = matplotlib.ticker.FixedLocator([np.log10(mm) for mm in np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9,
                                                                               1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0,
                                                                               10, 20, 30, 40, 50, 60, 70, 80, 90, 100])])
    plt.gca().xaxis.set_minor_locator(x_minor)
    plt.gca().xaxis.set_minor_formatter(plt.NullFormatter())

    plt.savefig(paper_dir + 'ob150211_mass_posterior_modes.png')

    return


def plot_trace_corner(target):
    labels = {'t0':       '$t_0$ (MJD)',
              'u0_amp':   '$u_0$',
              'tE':       '$t_E$ (days)',
              'piE_E':    '$\pi_{E,E}$',
              'piE_N':    '$\pi_{E,N}$',
              'piE':      '$\pi_{E}$',
              'b_sff1':   '$b_{SFF,I}$',
              'mag_src1': '$I_{src}$ (mag)',
              'mag_base1': '$I_{base}$ (mag)',
              'mult_err1': '$\\varepsilon_{m,I}$',
              'add_err1':  '$\\varepsilon_{a,I}$ (mmag)',
              'thetaE_amp':'$\\theta_E$ (mas)',
              'thetaE':    '$\\theta_E$ (mas)',
              'log10_thetaE':'$\log_{10} [\\theta_E$ (mas)]',
              'piS':      '$\pi_S$ (mas)',
              'muS_E':    '$\mu_{S,\\alpha*}$ (mas/yr)',
              'muS_N':    '$\mu_{S,\delta}$ (mas/yr)',
              'xS0_E':    '$x_{S0,\\alpha*}$ (mas)',
              'xS0_N':    '$x_{S0,\delta}$ (mas)',
              'b_sff2':   '$b_{SFF,Kp}$',
              'mag_src2': '$Kp_{src}$ (mag)',
              'mag_base2': '$Kp_{base}$ (mag)',
              'mult_err2': '$\\varepsilon_{m,Kp}$',
              'add_err2': '$\\varepsilon_{a,Kp}$ (mmag)',
              'mL':       '$M_L (M_\odot)$',
              'piL':      '$\pi_L$ (mas)',
              'piRel':    '$\pi_{rel}$ (mas)',
              'muL_E':    '$\mu_{L,\\alpha*}$ (mas/yr)',
              'muL_N':    '$\mu_{L,\delta}$ (mas/yr)',
              'muRel_E':  '$\mu_{rel,\\alpha*}$ (mas/yr)',
              'muRel_N':  '$\mu_{rel,\delta}$ (mas/yr)',
              'muRel':    '$\mu_{rel}$ (mas/yr)',
              'gp_log_sigma1':      '$\log \sigma_{GP, I} (mag)$',
              'gp_rho1':            '$\\rho_{GP, I}$ (days)',
              'gp_log_omega04_S01': '$\log S_{0, GP, I} \omega_{0, GP, I}^4$ (mag$^2$ days$^{-2}$)',  
              'gp_log_omega01':     '$\log \omega_{0, GP, I}$ (days$^{-1}$)'
             }

        
    # Photometric and Astrometry Posteriors
    fitter, data = get_data_and_fitter(pspl_ast_multiphot[target])
    
    from dynesty import plotting as dyplot

    res = fitter.load_mnest_results_for_dynesty()
    res_m = fitter.load_mnest_modes_results_for_dynesty()
    params_global = fitter.get_best_fit(def_best='median')[0]
    fitter_params = copy.deepcopy(fitter.all_param_names)
    smy = calc_summary_statistics(fitter)
    fitter.all_param_names = fitter_params

    #
    # Add ampltidue of piE
    #
    idx_piEE = fitter.all_param_names.index('piE_E')
    idx_piEN = fitter.all_param_names.index('piE_N')
    
    piE = np.hypot(res['samples'][:, idx_piEE], res['samples'][:, idx_piEN])
    res['samples'] = np.append(res['samples'], np.array([piE]).T, axis=1)
    for mm in range(len(res_m)):
        piE_m = np.hypot(res_m[mm]['samples'][:, idx_piEE], res_m[mm]['samples'][:, idx_piEN])
        res_m[mm]['samples'] = np.append(res_m[mm]['samples'], np.array([piE_m]).T, axis=1)

    fitter.all_param_names.append('piE')
    smy['MaxLike_piE'] = np.hypot(smy['MaxLike_piE_E'], smy['MaxLike_piE_N'])
    smy['MAP_piE'] = np.hypot(smy['MAP_piE_E'], smy['MAP_piE_N'])
    smy['Med_piE'] = np.hypot(smy['Med_piE_E'], smy['Med_piE_N'])
    params_global['piE'] = np.hypot(params_global['piE_E'], params_global['piE_N'])

    #
    # Add ampltidue of muRel
    #
    idx_muRelE = fitter.all_param_names.index('muRel_E')
    idx_muRelN = fitter.all_param_names.index('muRel_N')
    
    muRel = np.hypot(res['samples'][:, idx_muRelE], res['samples'][:, idx_muRelN])
    res['samples'] = np.append(res['samples'], np.array([muRel]).T, axis=1)
    for mm in range(len(res_m)):
        muRel_m = np.hypot(res_m[mm]['samples'][:, idx_muRelE], res_m[mm]['samples'][:, idx_muRelN])
        res_m[mm]['samples'] = np.append(res_m[mm]['samples'], np.array([muRel_m]).T, axis=1)
    
    fitter.all_param_names.append('muRel')
    smy['MaxLike_muRel'] = np.hypot(smy['MaxLike_muRel_E'], smy['MaxLike_muRel_N'])
    smy['MAP_muRel'] = np.hypot(smy['MAP_muRel_E'], smy['MAP_muRel_N'])
    smy['Med_muRel'] = np.hypot(smy['Med_muRel_E'], smy['Med_muRel_N'])
    params_global['muRel'] = np.hypot(params_global['muRel_E'], params_global['muRel_N'])

    
    # # Trim down to just the primary fitting parameters.
    # N_fit = len(fitter.fitter_param_names)
    # samples = res['samples'][0:N_fit]

    #
    # Setup the truths arrays (choose Median)
    #
    best_sol = 'Med_'
    truths = np.zeros(len(fitter.all_param_names), dtype=float)
    truths_m = np.zeros((len(res_m), len(fitter.all_param_names)), dtype=float)
    ax_labels = []
    
    for pp in range(len(fitter.all_param_names)):
        param = fitter.all_param_names[pp]
        
        ax_labels.append(labels[param])

        # Global solution
        if param in params_global:
            # truths[pp] = smy[best_sol + param][0]  # global best fit.
            truths[pp] = params_global[param]  # global best fit.
        else:
            truths[pp] = None

        # Mode solutions
        for mm in range(len(res_m)):
            if best_sol + param in smy.colnames:
                truths_m[mm, pp] = smy[best_sol + param][mm]  # global best fit.
            else:
                truths_m[mm, pp] = None
            
    ax_labels = np.array(ax_labels)


    # plt.close('all')
    # dyplot.traceplot(res, labels=ax_labels,
    #                  show_titles=True, truths=truths)
    # plt.subplots_adjust(hspace=0.7)
    # plt.savefig(paper_dir + target + '_dy_trace.png')


    sigma_vals = np.array([0.682689])  # , 0.9545, 0.9973])
    credi_ints_lo = (1.0 - sigma_vals) / 2.0
    credi_ints_hi = (1.0 + sigma_vals) / 2.0
    credi_ints_med = np.array([0.5])
    quantiles = np.concatenate([credi_ints_med, credi_ints_lo, credi_ints_hi])

    ##########
    # Prep the figure
    ##########

    # First subset
    # fig1 = ['mL', 'piE', 'muRel']
    fig1 = ['mL', 'u0_amp', 'tE', 'piE', 'log10_thetaE', 'muRel', 'piRel', ]
    fig2 = ['mL', 'piS', 'piL', 'piE_E', 'piE_N', 'xS0_E', 'xS0_N', 't0']
    # if target == 'ob140613':
    #     fig3 = ['mL', 'b_sff1', 'mag_base1', 'mult_err1', 'b_sff2', 'mag_base2', 'mult_err2']
    # else:
    #     fig3 = ['mL', 'b_sff1', 'mag_base1', 'add_err1', 'b_sff2', 'mag_base2', 'add_err2']
    fig3 = ['mL', 'b_sff1', 'mag_base1', 'b_sff2', 'mag_base2']
    fig4 = ['mL', 'muS_E', 'muS_N', 'muL_E', 'muL_N', 'muRel_E', 'muRel_N']

    idx1 = [fitter.all_param_names.index(fig1_val) for fig1_val in fig1]
    idx2 = [fitter.all_param_names.index(fig2_val) for fig2_val in fig2]
    idx3 = [fitter.all_param_names.index(fig3_val) for fig3_val in fig3]
    idx4 = [fitter.all_param_names.index(fig4_val) for fig4_val in fig4]

    # all_idxs = [idx1, idx2, idx3, idx4]
    all_idxs = [idx1]

    for ii in range(len(all_idxs)):
        idx = all_idxs[ii]
        ndim = len(idx)

        # Set the axis limits. hard code the mass limit.
        span = [1.0 - 1e-6] * ndim
        mdx = np.where(np.array(fitter.all_param_names)[idx] == 'mL')[0]
        if len(mdx) > 0:
            if target == 'ob150211':
                span[mdx[0]] = [0, 100]
            else:
                span[mdx[0]] = [0, 10]
        
        smooth = 0.05
        
        fig, axes = plt.subplots(ndim, ndim, figsize=(20, 20))
        plt.subplots_adjust(left=0.3, bottom=0.3)
        dyplot.cornerplot(res,
                          dims=idx, labels=ax_labels[idx], truths=truths[idx],
                          show_titles=False, quantiles=quantiles,
                          fig=(fig, axes), span=span, smooth=smooth)
        ax = plt.gca()
        ax.tick_params(axis='both', which='major', labelsize=10)
        plt.savefig(paper_dir + target + '_dy_corner_' + str(ii) + '.png')
    
    plt.close('all')

    #####
    # Now plot the individual modes.
    #####
    if (len(smy) > 1):
        for mm in range(len(smy)):
            for ii in range(len(all_idxs)):
                idx = all_idxs[ii]
                ndim = len(idx)

                # Set the axis limits. hard code the mass limit.
                span = [1.0 - 1e-6] * ndim
                mdx = np.where(np.array(fitter.all_param_names)[idx] == 'mL')[0]
                if len(mdx) > 0:
                    if target == 'ob150211':
                        span[mdx[0]] = [0, 100]
                    else:
                        span[mdx[0]] = [0, 10]

                smooth = 0.05
        
                fig, axes = plt.subplots(ndim, ndim, figsize=(20, 20))
                plt.subplots_adjust(left=0.3, bottom=0.3)
                dyplot.cornerplot(res_m[mm],
                                      dims=idx, labels=ax_labels[idx], truths=truths_m[mm, idx],
                                      show_titles=False, quantiles=quantiles,
                                      fig=(fig, axes), span=span, smooth=smooth)
                ax = plt.gca()
                ax.tick_params(axis='both', which='major', labelsize=10)
                plt.savefig(paper_dir + target + '_dy_corner_' + 'mode' + str(mm) + '_' + str(ii) + '.png')
    
    return

def plot_ob150211_mass_piE_muRel_all_modes():
    target = 'ob150211'
    
    foo = get_data_fitter_params_models_samples(target, return_mode = 'all')
    fitter = foo[0]
    data = foo[1]
    stats_all = foo[2]
    params_all = foo[3]
    models_all = foo[4]
    sampls_all = foo[5]

    # plt.close('all')
    # plt.figure(1, figsize=(12, 6))
    # plt.clf()
    # plt.subplots_adjust(left=0.08, right=0.95, top=0.95)
    # ax1 = plt.subplot(1, 2, 1)
    # ax2 = plt.subplot(1, 2, 2)

    plt.figure(1, figsize=(6, 6))
    plt.clf()
    plt.subplots_adjust(left=0.15, right=0.95, top=0.95)
    ax1 = plt.subplot(1, 1, 1)
    # ax2 = plt.subplot(1, 2, 2)

    colors = ['black', 'purple', 'green']

    for ii in range(len(sampls_all)):
        sampls_all[ii]['muRel'] = np.hypot(sampls_all[ii]['muRel_E'], sampls_all[ii]['muRel_N'])
        sampls_all[ii]['piE'] = np.hypot(sampls_all[ii]['piE_E'], sampls_all[ii]['piE_N'])

        span_mL = 1.0 - 1e-3
        span_muRel = 1.0 - 1e-3
        span_piE = 1.0 - 1e-3

        smooth_mL = 0.025
        smooth_muRel = 0.05
        smooth_piE = 0.05

        plot_2d_contour(np.log10(sampls_all[ii]['mL']), sampls_all[ii]['muRel'],
                        sampls_all[ii]['weights'],
                        ax1, colors[ii],
                        span1=span_mL, span2=span_muRel,
                        smooth1=smooth_mL, smooth2=smooth_muRel)
        # plot_2d_contour(sampls_all[ii]['piE'], sampls_all[ii]['muRel'],
        #                 sampls_all[ii]['weights'],
        #                 ax2, colors[ii],
        #                 span1=span_piE, span2=span_muRel,       
        #                 smooth1=smooth_piE, smooth2=smooth_muRel)
        
    # ax1.set_xscale('log')
    ax1.set_xlabel('$\log M_L$ (M$_\odot$)')
    ax1.set_ylabel('$\mu_{rel}$ (mas/yr)')
    # ax1.set_xlim(0, 25)
    ax1.set_xlim(-2, 1.5)

    # ax2.set_xlabel('$\pi_E$')
    # ax2.set_ylabel('$\mu_{rel}$ (mas/yr)')
    # ax2.set_xlim(0, 0.15)

    ax1.text(-1.8, 4.3, 'Mode 1 - Preferred', color=colors[0])
    ax1.text(-1.8, 4.0, 'Mode 2', color=colors[1])
    ax1.text(-1.8, 3.7, 'Mode 3', color=colors[2])
    plt.savefig(paper_dir + 'ob150211_mL_piE_muRel_all_modes.png')


    # # Check against other corner plot code.
    # fitter.all_param_names = fitter.all_param_names[:-3]
    # res_m = fitter.load_mnest_modes_results_for_dynesty()
    
    # #
    # # Add ampltidue of piE and muRel
    # #
    # idx_piEE = fitter.all_param_names.index('piE_E')
    # idx_piEN = fitter.all_param_names.index('piE_N')
    # idx_muRelE = fitter.all_param_names.index('muRel_E')
    # idx_muRelN = fitter.all_param_names.index('muRel_N')
    
    # for mm in range(len(res_m)):
    #     piE_m = np.hypot(res_m[mm]['samples'][:, idx_piEE], res_m[mm]['samples'][:, idx_piEN])
    #     res_m[mm]['samples'] = np.append(res_m[mm]['samples'], np.array([piE_m]).T, axis=1)
        
    #     muRel_m = np.hypot(res_m[mm]['samples'][:, idx_muRelE], res_m[mm]['samples'][:, idx_muRelN])
    #     res_m[mm]['samples'] = np.append(res_m[mm]['samples'], np.array([muRel_m]).T, axis=1)
    # fitter.all_param_names += ['piE', 'muRel']
    

    # fig_params = ['mL', 'piE', 'muRel']
    # idx = [fitter.all_param_names.index(fig_par) for fig_par in fig_params]
    # ndim = len(idx)
    
        
    # fig, axes = plt.subplots(ndim, ndim, figsize=(10, 10))
    # plt.subplots_adjust(left=0.3, bottom=0.3)
    # for mm in range(len(sampls_all)):
    #     span = [1.0 - 1e-4] * ndim
    #     mdx = np.where(np.array(fig_params) == 'mL')[0]
    #     if len(mdx) > 0:
    #         span[mdx[0]] = [0, 100]
        
    #     dyplot.cornerplot(res_m[mm],
    #                       dims=idx, labels=fig_params, 
    #                       show_titles=False, color=colors[mm],
    #                       fig=(fig, axes), span=span, smooth=0.02)
    #                       # fig=(fig, axes), smooth=100)
    #     ax = plt.gca()
    #     ax.tick_params(axis='both', which='major', labelsize=10)

    # plt.savefig(paper_dir + 'ob150211_mL_piE_muRel_all_modes1.png')

    # plt.figure(3)
    # plt.hist(np.log10(res_m[0]['samples'][:, idx[0]]), bins=100, weights=res_m[0]['weights'])
    # plt.xlabel('log(mL)')
    # plt.ylabel('Probability')

    # plt.figure(4)
    # plt.hist(res_m[0]['samples'][:, idx[0]], bins=1000, weights=res_m[0]['weights'])
    # plt.xlabel('mL')
    # plt.ylabel('Probability')
    # plt.xlim(0, 60)
    
    
    return


def plot_2d_contour(samples1, samples2, weights, axes, color,
                    smooth1=0.02, smooth2=0.02,
                    span1=0.999, span2=0.999,
                    truths=None):
    # Plot the 2-D marginalized posteriors.
    
    # Generate distribution.
    fill_contours = False
    plot_contours = True

    levels_sigma = np.array([0.5, 1.0, 2.0])
    quantiles_2d = 1.0 - np.exp(-0.5 * levels_sigma**2)
    
    hist2d_kwargs = dict()
    hist2d_kwargs['levels'] = hist2d_kwargs.get('levels', quantiles_2d)
    hist2d_kwargs['fill_contours'] = hist2d_kwargs.get('fill_contours',
                                                       fill_contours)
    hist2d_kwargs['plot_contours'] = hist2d_kwargs.get('plot_contours',
                                                       plot_contours)
    hist2d_kwargs['alpha'] = 1.0
    hist2d_kwargs['no_fill_contours'] = True
    hist2d_kwargs['plot_density'] = False

     # This "color map" is the list of colors for the contour levels if the
    # contours are filled.
    from matplotlib.colors import LinearSegmentedColormap, colorConverter
    rgba_color = colorConverter.to_rgba(color)
    contour_cmap = [list(rgba_color) for l in quantiles_2d] + [rgba_color]
    for i, l in enumerate(quantiles_2d):
        contour_cmap[i][-1] *= float(i+1) / (len(quantiles_2d))

    contour_kwargs = dict()
    contour_kwargs['colors'] = contour_cmap

    dyplot._hist2d(samples1, samples2, ax=axes, 
                   weights=weights, color=color,
                   smooth=[smooth1, smooth2], span=[span1, span2],
                   contour_kwargs=contour_kwargs, **hist2d_kwargs)
    #span=[span[j], span[i]],
    
    # Add truth values
    if truths is not None:
        if truths[0] is not None:
            ax.axvline(truths[0], color=color) #, **truth_kwargs)
        if truths[1] is not None:
            ax.axhline(truths[1], color=color) # , **truth_kwargs)

    return
    

def get_data_and_fitter(mnest_base):
    info_file = open(mnest_base + 'params.yaml', 'r')
    info = yaml.full_load(info_file)

    pho_dsets = info['phot_data']
    if 'astrom_data' in info.keys():
        ast_dsets = info['astrom_data']
    else:
        ast_dsets = []

    my_model = getattr(model, info['model'])
    my_data = munge.getdata2(info['target'].lower(), 
                             phot_data=pho_dsets,
                             ast_data=ast_dsets)

    # Need this for backwards compatability.
    if 'use_phot_optional_params' in info:
        use_phot_opt = info['use_phot_optional_params']
    else:
        use_phot_opt = False

    # Load up the first fitter object to get the parameter names.
    fitter = model_fitter.PSPL_Solver(my_data, my_model,
                                      use_phot_optional_params = use_phot_opt,
                                      add_error_on_photometry = info['add_error_on_photometry'],
                                      multiply_error_on_photometry = info['multiply_error_on_photometry'],
                                      outputfiles_basename = mnest_base)

    return fitter, my_data

def get_data_fitter_params_models_samples(target, return_mode = 'best', def_best='median'):
    """
    Return the fitter, data, stats table, parameters array, model, and posterior samples for
    the desired target from the PSPL astrometry + multi-photometry fits. 

    If return_mode == 'best', then return the mode designated in the 
    pspl_ast_multiphot_mode array.

    If return_mode == 'all' (or anything else), then return all the modes.
    """
    # Load model_fitter and data objects
    fitter, data = get_data_and_fitter(pspl_ast_multiphot[target])

    # Load best-fit model objects for all modes.
    models_all = fitter.get_best_fit_modes_model(def_best = def_best)

    # Load best-fit parameters and statistics (median, mean, maxLike, etc.) for all modes.
    # Load posterior samples for all modes.
    stats_all, sampls_all = calc_summary_statistics(fitter, return_samples=True)

    # Select the best-fit parameters and make them into a useful table. 
    params_all = get_best_fit_from_stats(fitter, stats_all, def_best=def_best)
    
    mode = pspl_ast_multiphot_mode[target]

    # Sort all of the samples by logLike
    for mm in range(len(stats_all)):
        sdx = sampls_all[mm]['logLike'].argsort()
        sampls_all[mm] = sampls_all[mm][sdx]
        
    # Return just th designated mode.
    if return_mode == 'best':
        stats_all = stats_all[mode]
        params_all = params_all[mode]
        models_all = models_all[mode]
        sampls_all = sampls_all[mode]


    return (fitter, data, stats_all, params_all, models_all, sampls_all)


def load_summary_statistics(mnest_base, verbose=False):
    info_file = open(mnest_base + 'params.yaml', 'r')
    info = yaml.full_load(info_file)
    
    my_model = getattr(model, info['model'])
    my_data = munge.getdata2(info['target'].lower(), 
                             phot_data=info['phot_data'], 
                             ast_data=info['astrom_data'])

    if 'use_phot_optional_params' in info:
        use_phot_opt = info['use_phot_optional_params']
    else:
        use_phot_opt = False
    

    # Load up the first fitter object to get the parameter names.
    fitter = model_fitter.PSPL_Solver(my_data, my_model,
                                      use_phot_optional_params = use_phot_opt,
                                      add_error_on_photometry = info['add_error_on_photometry'],
                                      multiply_error_on_photometry = info['multiply_error_on_photometry'],
                                      outputfiles_basename = mnest_base)

    stats = calc_summary_statistics(fitter, verbose=verbose)

    return stats, my_data, my_model
    

def calc_summary_statistics(fitter, verbose=False, return_samples=False):
    # Get the number of modes.
    summ_tab = Table.read(fitter.outputfiles_basename + 'summary.fits')
    N_modes = len(summ_tab) - 1
    
    # Calculate the number of data points we have all together.
    N_data = get_num_data_points(fitter.data,
                                 astrometryFlag=fitter.model_class.astrometryFlag,
                                 verbose=verbose)
    
    N_params = len(fitter.fitter_param_names)
    N_dof = N_data - N_params

    if verbose:
        print('**** calc_summary_statistics: ****')
        print('Model = ', type(fitter.model_class).__name__)
        print('Data = ', fitter.data.keys())

    # First, we want the statistics for the following types of solutions.
    sol_types = ['maxl', 'mean', 'map', 'median']
    sol_prefix = {'maxl': 'MaxLike_',
                  'mean': 'Mean_',
                  'map': 'MAP_',
                  'median': 'Med_' }
    
    tab_list = fitter.load_mnest_modes()
    smy = fitter.load_mnest_summary()
    # remember smy has N+1 because the 0th entry is the global solution.
    # Trim this one out.
    smy = smy[1:]

    # Add in custom variables derived from other quantites.
    current_params = tab_list[0].colnames

    # Add thetaE (from log thetaE)
    if ('log10_thetaE' in current_params) and ('thetaE' not in current_params):
        for ii in range(len(tab_list)):
            tab_list[ii]['thetaE'] = 10**tab_list[ii]['log10_thetaE']

        smy['Mean_thetaE'] = 10**smy['Mean_log10_thetaE']
        smy['MAP_thetaE'] = 10**smy['MAP_log10_thetaE']
        smy['MaxLike_thetaE'] = 10**smy['MaxLike_log10_thetaE']
        #smy['Med_thetaE'] = 10**smy['Med_log10_thetaE']

        fitter.all_param_names.append('thetaE')

    # Add mag_src. 
    if ('mag_base1' in current_params) and ('mag_src1' not in current_params):
        for ii in range(len(tab_list)):
            tab_list[ii]['mag_src1'] = tab_list[ii]['mag_base1'] - 2.5 * np.log10(tab_list[ii]['b_sff1'])

        smy['Mean_mag_src1'] = smy['Mean_mag_base1'] - 2.5 * np.log10(smy['Mean_b_sff1'])
        smy['MAP_mag_src1'] = smy['MAP_mag_base1'] - 2.5 * np.log10(smy['MAP_b_sff1'])
        smy['MaxLike_mag_src1'] = smy['MaxLike_mag_base1'] - 2.5 * np.log10(smy['MaxLike_b_sff1'])
        #smy['Med_mag_src1'] = smy['Med_mag_base1'] - 2.5 * np.log10(smy['Med_b_sff1'])

        fitter.all_param_names.append('mag_src1')
        

    if ('mag_base2' in current_params) and ('mag_src2' not in current_params):
        for ii in range(len(tab_list)):
            tab_list[ii]['mag_src2'] = tab_list[ii]['mag_base2'] - 2.5 * np.log10(tab_list[ii]['b_sff2'])

        smy['Mean_mag_src2'] = smy['Mean_mag_base2'] - 2.5 * np.log10(smy['Mean_b_sff2'])
        smy['MAP_mag_src2'] = smy['MAP_mag_base2'] - 2.5 * np.log10(smy['MAP_b_sff2'])
        smy['MaxLike_mag_src2'] = smy['MaxLike_mag_base2'] - 2.5 * np.log10(smy['MaxLike_b_sff2'])
        #smy['Med_mag_src2'] = smy['Med_mag_base2'] - 2.5 * np.log10(smy['Med_b_sff2'])

        fitter.all_param_names.append('mag_src2')
        

    # Make a deepcopy of this table and set everything to zeros.
    # This will contain our final results.
    stats = copy.deepcopy(smy)
    for col in stats.colnames:
        stats[col] = np.nan

    # Loop through the different modes.
    for nn in range(N_modes):
            
        # Loop through different types of "solutions"
        for sol in sol_types:

            # Loop through the parameters and get the best fit values.
            foo = fitter.calc_best_fit(tab_list[nn], smy[[nn]], s_idx=0, def_best=sol)

            if sol == 'maxl' or sol == 'map':
                best_par = foo
            else:
                best_par = foo[0]
                best_parerr = foo[1]

            for param in fitter.all_param_names:
                if sol_prefix[sol] + param not in stats.colnames:
                    stats[sol_prefix[sol] + param] = 0.0
                stats[sol_prefix[sol] + param][nn] = best_par[param]

            # Add chi^2 to the table.
            chi2 = fitter.calc_chi2(best_par)
            if sol_prefix[sol] + 'chi2' not in stats.colnames:
                stats[sol_prefix[sol] + 'chi2'] = 0.0
            stats[sol_prefix[sol] + 'chi2'][nn] = chi2

            # Add reduced chi^2 to the table.
            rchi2 = chi2 / N_dof
            if sol_prefix[sol] + 'rchi2' not in stats.colnames:
                stats[sol_prefix[sol] + 'rchi2'] = 0.0
            stats[sol_prefix[sol] + 'rchi2'][nn] = rchi2

            # Add log-likelihood to the table. 
            logL = fitter.log_likely(best_par)
            if sol_prefix[sol] + 'logL' not in stats.colnames:
                stats[sol_prefix[sol] + 'logL'] = 0.0
            stats[sol_prefix[sol] + 'logL'][nn] = logL

            # BIC
            if sol_prefix[sol] + 'BIC' not in stats.colnames:
                stats[sol_prefix[sol] + 'BIC'] = 0.0
            stats[sol_prefix[sol] + 'BIC'][nn] = calc_BIC(N_data, N_params, stats[sol_prefix[sol] + 'logL'][nn])        
            
            # Next figure out the errors.
            # Only need to do this once.
            if sol == 'median':
                sigma_vals = np.array([0.682689, 0.9545, 0.9973])
                credi_ints_lo = (1.0 - sigma_vals) / 2.0
                credi_ints_hi = (1.0 + sigma_vals) / 2.0
                credi_ints_med = np.array([0.5])
                credi_ints = np.concatenate([credi_ints_med, credi_ints_lo, credi_ints_hi])

                sumweights = np.sum(tab_list[nn]['weights'])
                weights = tab_list[nn]['weights'] / sumweights
                
                for param in fitter.all_param_names:
                    # Calculate median, 1 sigma lo, and 1 sigma hi credible interval.
                    tmp = model_fitter.weighted_quantile(tab_list[nn][param], credi_ints, sample_weight=weights)
                    ci_med_val = tmp[0]
                    ci_lo = tmp[1:1+len(sigma_vals)]
                    ci_hi = tmp[1+len(sigma_vals):]

                    if 'lo68_' + param not in stats.colnames:
                        stats['lo68_' + param] = 0.0
                        stats['hi68_' + param] = 0.0
                    if 'lo95_' + param not in stats.colnames:
                        stats['lo95_' + param] = 0.0
                        stats['hi95_' + param] = 0.0
                    if 'lo99_' + param not in stats.colnames:
                        stats['lo99_' + param] = 0.0
                        stats['hi99_' + param] = 0.0
                        
                    # Add back in the median to get the actual value (not diff on something).
                    stats['lo68_' + param][nn] = ci_lo[0]
                    stats['hi68_' + param][nn] = ci_hi[0]
                    stats['lo95_' + param][nn] = ci_lo[1]
                    stats['hi95_' + param][nn] = ci_hi[1]
                    stats['lo99_' + param][nn] = ci_lo[2]
                    stats['hi99_' + param][nn] = ci_hi[2]
            
        # Get the evidence values out of the _stats.dat file.
        if 'logZ' not in stats.colnames:
            stats['logZ'] = 0.0
        stats['logZ'][nn] = smy['logZ'][nn]

    # Add number of degrees of freedom
    stats['N_dof'] = N_dof

    # # Sort such that the modes are in reverse order of evidence.
    # # Increasing logZ (nan's are at the end)
    # zdx = np.argsort(stats['logZ'])
    # non_nan = np.where(np.isfinite(stats['logZ'][zdx]))[0]
    # zdx = zdx[non_nan[::-1]]

    # stats = stats[zdx]

    if return_samples:
        return stats, tab_list
    else:
        return stats

def get_best_fit_from_stats(fitter, stats, def_best='median'):
    """
    Return a dictionary of all the best-fit parameters for
    the best-fit style of your choice.

    This will work on a single mode or all modes, 
    stats or summary tables. 
    """
    best_params_all = []

    sol_types = ['maxl', 'mean', 'map', 'median']
    sol_prefix = {'maxl': 'MaxLike_',
                  'mean': 'Mean_',
                  'map': 'MAP_',
                  'median': 'Med_' }

    # Loop through all solutions/modes.
    for mm in range(len(stats)):
        best_params = {}
        
        # Loop through all parameters.
        for pp in range(len(fitter.all_param_names)):
            param = fitter.all_param_names[pp]
            prefix = sol_prefix[def_best]

            if prefix + param in stats.colnames:
                best_params[param] = stats[prefix + param][mm]
            else:
                best_params[param] = None

        best_params_all.append(best_params)

    return best_params_all

    
def make_BIC_comparison_table():
    # Use the one with the highest likelihood solution.
    ob120169_none = load_summary_statistics(ogle_phot_all['ob120169_none'] + 'd7_')
    ob120169_add  = load_summary_statistics(ogle_phot_all['ob120169_add'] + 'c3_') 
    ob120169_mult = load_summary_statistics(ogle_phot_all['ob120169_mult'] + 'a0_')

    ob140613_none = load_summary_statistics(ogle_phot_all['ob140613_none'] + 'c2_') 
    ob140613_add  = load_summary_statistics(ogle_phot_all['ob140613_add']  + 'b3_') 
    ob140613_mult = load_summary_statistics(ogle_phot_all['ob140613_mult'] + 'c8_') 

    ob150029_none = load_summary_statistics(ogle_phot_all['ob150029_none'] + 'b5_') 
    ob150029_add  = load_summary_statistics(ogle_phot_all['ob150029_add']  + 'd8_')
    ob150029_mult = load_summary_statistics(ogle_phot_all['ob150029_mult'] + 'd2_') 

    ob150211_none = load_summary_statistics(ogle_phot_all['ob150211_none'] + 'a1_')
    ob150211_add  = load_summary_statistics(ogle_phot_all['ob150211_add']  + 'a4_')
    ob150211_mult = load_summary_statistics(ogle_phot_all['ob150211_mult'] + 'd5_')

    
    with open(paper_dir + 'BIC_comparison.txt', 'w+') as tab_file:
        tab_file.write('No error term' + ' & ' 
                       + '{0:.2f}'.format(ob120169_none['MaxLike_BIC']) + ' & ' 
                       + '{0:.2f}'.format(ob140613_none['MaxLike_BIC']) + ' & ' 
                       + '{0:.2f}'.format(ob150029_none['MaxLike_BIC']) + ' & ' 
                       + '{0:.2f}'.format(ob150211_none['MaxLike_BIC']) + r' \\ ' + '\n'
                       + 
                       'Multiplicative' + ' & ' 
                       + '{0:.2f}'.format(ob120169_mult['MaxLike_BIC']) + ' & ' 
                       + '{0:.2f}'.format(ob140613_mult['MaxLike_BIC']) + ' & ' 
                       + '{0:.2f}'.format(ob150029_mult['MaxLike_BIC']) + ' & ' 
                       + '{0:.2f}'.format(ob150211_mult['MaxLike_BIC']) + r' \\ ' + '\n'
                       +
                       'Additive' + ' & ' 
                       + '{0:.2f}'.format(ob120169_add['MaxLike_BIC']) + ' & ' 
                       + '{0:.2f}'.format(ob140613_add['MaxLike_BIC']) + ' & ' 
                       + '{0:.2f}'.format(ob150029_add['MaxLike_BIC']) + ' & ' 
                       + '{0:.2f}'.format(ob150211_add['MaxLike_BIC']) + r' \\ ' + '\n')

    return
                       
    
def calc_chi2(target):
    """
    Calculate the chi^2 with and without the additive (or multiplicative
    error and separetely for each astrometric and photometric data 
    set so we can report back in the paper. 
    """
    fitter1, data1 = get_data_and_fitter(pspl_phot[target])
    fitter2, data2 = get_data_and_fitter(pspl_ast_multiphot[target])

    model1 = fitter1.get_best_fit_model(def_best='maxl')
    model2 = fitter2.get_best_fit_model(def_best='maxl')

    params1 = fitter1.get_best_fit(def_best='maxl')
    params2 = fitter2.get_best_fit(def_best='maxl')

    # Check that these are actually the same u0 sign
    if np.sign(model1.u0_amp) != np.sign(model2.u0_amp):
        print(' *** Model mis-match!!! ***')

        # Assume that model2 (ast + phot) is what we want to go
        # with. Pick out the corresponding same-sign u0 for the
        # photometry fit as well.
        model1_all = fitter1.get_best_fit_modes_model(def_best='maxl')
        params1_all = fitter1.get_best_fit_modes(def_best='maxl')
        model1 = model1_all[1]
        params1 = params1_all[1]

    assert np.sign(model1.u0_amp) == np.sign(model2.u0_amp)

    def print_chi2_info(fitter, pspl, params):
        # Get likelihoods.
        lnL_phot = fitter.log_likely_photometry(pspl, params)
        lnL_ast = fitter.log_likely_astrometry(pspl)

        # Calculate constants needed to subtract from lnL to calculate chi2.
        if pspl.astrometryFlag:
            lnL_const_ast =  -0.5 * np.log(2.0 * math.pi * fitter.data['xpos_err1'] ** 2)
            lnL_const_ast += -0.5 * np.log(2.0 * math.pi * fitter.data['ypos_err1'] ** 2)
            lnL_const_ast = lnL_const_ast.sum()
        else:
            lnL_const_ast = 0

        # Lists to store lnL, chi2, and constants for each filter.
        chi2_phot_filts1 = []  # no error modification
        chi2_phot_filts2 = []  # error modification
        rchi2_phot_filts1 = []  # no error modification
        rchi2_phot_filts2 = []  # error modification
        lnL_const_phot_filts1 = []
        lnL_const_phot_filts2 = []
        ndata_phot = []
    
        for nn in range(fitter.n_phot_sets):
            t_phot =  fitter.data['t_phot' + str(nn + 1)]
            mag      = fitter.data['mag' + str(nn + 1)]
            mag_err1 = fitter.data['mag_err' + str(nn + 1)]            
            mag_err2 = fitter.get_modified_mag_err(params, nn)
        
            # Calculate the lnL for just a single filter.
            # Without additive/multiplicative error.
            lnL_phot_nn1 = pspl.likely_photometry(t_phot, mag, mag_err1, nn)
            lnL_phot_nn1 = lnL_phot_nn1.sum()
            # With additive/multiplicative error.
            lnL_phot_nn2 = pspl.likely_photometry(t_phot, mag, mag_err2, nn)
            lnL_phot_nn2 = lnL_phot_nn2.sum()

            # Calculate the chi2 and constants for just a single filter.
            lnL_const_phot_nn1 = -0.5 * np.log(2.0 * math.pi * mag_err1**2)
            lnL_const_phot_nn1 = lnL_const_phot_nn1.sum()
            lnL_const_phot_nn2 = -0.5 * np.log(2.0 * math.pi * mag_err2**2)
            lnL_const_phot_nn2 = lnL_const_phot_nn2.sum()
        
            chi2_phot_nn1 = (lnL_phot_nn1 - lnL_const_phot_nn1) / -0.5
            chi2_phot_nn2 = (lnL_phot_nn2 - lnL_const_phot_nn2) / -0.5

            N_dof = len(mag) - len(params)

            rchi2_phot_nn1 = chi2_phot_nn1 / N_dof
            rchi2_phot_nn2 = chi2_phot_nn2 / (N_dof + 1)  # discount the additional error term

            # Save to our lists
            chi2_phot_filts1.append(chi2_phot_nn1)
            chi2_phot_filts2.append(chi2_phot_nn2)
            rchi2_phot_filts1.append(rchi2_phot_nn1)
            rchi2_phot_filts2.append(rchi2_phot_nn2)
            lnL_const_phot_filts1.append(lnL_const_phot_nn1)
            lnL_const_phot_filts2.append(lnL_const_phot_nn2)
            ndata_phot.append(len(mag))

        # For the "all photometry" chi^2, we need to subtract off the total constant.
        lnL_const_phot = np.sum(lnL_const_phot_filts2)

        # Calculate chi2.
        chi2_ast = (lnL_ast - lnL_const_ast) / -0.5
        chi2_phot = (lnL_phot - lnL_const_phot) / -0.5
        chi2 = chi2_ast + chi2_phot

        if pspl.astrometryFlag:
            N_dof_ast = len(fitter.data['t_ast1']) - len(params)
            rchi2_ast = chi2_ast / N_dof_ast
            
        N_dof_phot = np.sum(ndata_phot) - len(params)
        N_dof = len(fitter.data['t_ast1']) + np.sum(ndata_phot) - len(params)
        rchi2_phot = chi2_phot / N_dof_phot
        rchi2 = chi2 / N_dof
        
        # Now print everything out. 
        fmt = '{0:13s} = {1:f} '
        for ff in range(fitter.n_phot_sets):
            print(fmt.format('chi2_phot' + str(ff + 1) + ', no error mod  ', chi2_phot_filts1[ff]))
            print(fmt.format('chi2_phot' + str(ff + 1) + ', with error mod', chi2_phot_filts2[ff]))
            print(fmt.format('rchi2_phot' + str(ff + 1) + ', no error mod  ', rchi2_phot_filts1[ff]))
            print(fmt.format('rchi2_phot' + str(ff + 1) + ', with error mod', rchi2_phot_filts2[ff]))
            
        if pspl.astrometryFlag:
            print(fmt.format('chi2_ast', chi2_ast))
        print(fmt.format('chi2_phot', chi2_phot))
        print(fmt.format('chi2', chi2))
        
        if pspl.astrometryFlag:
            print(fmt.format('rchi2_ast', rchi2_ast))
        print(fmt.format('rchi2_phot', rchi2_phot))
        print(fmt.format('rchi2', rchi2))

        # Calculate a total chi^2 with and without extra error terms.
        chi2_tot1 = np.sum(chi2_phot_filts1)
        chi2_tot2 = np.sum(chi2_phot_filts2)
        if pspl.astrometryFlag:
            chi2_tot1 += chi2_ast
            chi2_tot2 += chi2_ast
        rchi2_tot1 = chi2_tot1 / N_dof
        rchi2_tot2 = chi2_tot2 / N_dof
        print(fmt.format('chi2_tot, no error mod   ', chi2_tot1))
        print(fmt.format('chi2_tot, with error mod ', chi2_tot2))
        print(fmt.format('rchi2_tot, no error mod  ', rchi2_tot1))
        print(fmt.format('rchi2_tot, with error mod', rchi2_tot2))
        

        if pspl.astrometryFlag:
            print('Ndata_ast = {0:d}'.format(len(fitter.data['t_ast1'])))
        for ff in range(fitter.n_phot_sets):
            print('Ndata_phot{0:d} = {1:d}'.format(ff+1, ndata_phot[ff]))
        print('Ndata_phot all = {0:d}'.format(np.sum(ndata_phot)))
                      
        return

    print('\n**** OGLE-only Phot ****')
    print_chi2_info(fitter1, model1, params1)
    
    print('\n**** OGLE+Keck Phot+Ast ****')
    print_chi2_info(fitter2, model2, params2)
    
    return


def calc_BIC(n, k, maxlogL):
    """
    maxL = maximized value of the LOG likelihood function
    n = number of data points
    k = number of parameters in model
    """

    bic = np.log(n) * k - 2 * maxlogL

    return bic


####################################
### PopSyCLE Visualization Stuff ###
####################################
def murel_popsycle():
    t = Table.read(popsycle_events)

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
    t = Table.read(popsycle_events)

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
    t = Table.read(popsycle_events)

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
    t = Table.read(popsycle_events)

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

    plt.close('all')

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

    fitter = model_fitter.PSPL_parallax_Solver(data, outputfiles_basename=pspl_ast_multiphot[target])
    fitter.load_mnest_results()
    mymodel = fitter.get_best_fit_model(use_median=True)

    #####
    # Astrometry on the sky
    #####
    plt.figure(2)
    plt.clf()

    # Data
    plt.errorbar(data['xpos1']*1e3, data['ypos1']*1e3,
                     xerr=data['xpos_err1']*1e3, yerr=data['ypos_err1']*1e3, fmt='k.')

    # 1 day sampling over whole range
    t_mod = np.arange(data['t_ast1'].min(), data['t_ast1'].max(), 1)

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
    plt.errorbar(data['t_ast1'], data['xpos1']*1e3,
                     yerr=data['xpos_err1']*1e3, fmt='k.')
    plt.plot(t_mod, pos_out[:, 0]*1e3, 'r-')
    plt.xlabel('t - t0 (days)')
    plt.ylabel(r'$\Delta \alpha^*$ (mas)')
    plt.title('Input Data and Output Model')

    plt.figure(4)
    plt.clf()
    plt.errorbar(data['t_ast1'], data['ypos1']*1e3,
                     yerr=data['ypos_err1']*1e3, fmt='k.')
    plt.plot(t_mod, pos_out[:, 1]*1e3, 'r-')
    plt.xlabel('t - t0 (days)')
    plt.ylabel(r'$\Delta \delta$ (mas)')
    plt.title('Input Data and Output Model')

    #####
    # Remove the unlensed motion (proper motion)
    # astrometry vs. time
    #####
    # Make the model unlensed points.
    p_mod_unlens_tdat = mymodel.get_astrometry_unlensed(data['t_ast1'])
    x_mod_tdat = p_mod_unlens_tdat[:, 0]
    y_mod_tdat = p_mod_unlens_tdat[:, 1]
    x_no_pm = data['xpos1'] - x_mod_tdat
    y_no_pm = data['ypos1'] - y_mod_tdat

    # Make the dense sampled model for the same plot
    dp_tmod_unlens = mymodel.get_astrometry(t_mod) - mymodel.get_astrometry_unlensed(t_mod)
    x_mod_no_pm = dp_tmod_unlens[:, 0]
    y_mod_no_pm = dp_tmod_unlens[:, 1]

    # Prep some colorbar stuff
    cmap = plt.cm.viridis
    norm = plt.Normalize(vmin=data['t_ast1'].min(), vmax=data['t_ast1'].max())
    smap = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    smap.set_array([])

    # Average the data for the last three years.
    t_year = []
    x_year = []
    y_year = []
    xe_year = []
    ye_year = []

    t_dat_yr = multinest_plot.mmjd_to_year(data['t_ast1'])

    min_year = np.floor(t_dat_yr.min())
    max_year = np.ceil(t_dat_yr.max())

    years = np.arange(min_year, max_year + 1, 1)
    for tt in range(len(years)-1):
        ii = np.where((years[tt] < t_dat_yr) & (t_dat_yr <= years[tt+1]))[0]

        if tt == 100:
            t_year = data['t_ast1'][ii].data
            x_year = x_no_pm[ii]
            y_year = y_no_pm[ii]
            xe_year = data['xpos_err1'][ii]
            ye_year = data['ypos_err1'][ii]
        else:
            #wgts = np.hypot(data['xpos_err1'][ii], data['ypos_err1'][ii])
            wgts = (data['xpos_err1'][ii] + data['ypos_err1'][ii]) / 2.0
            t_avg, t_std = weighted_avg_and_std(data['t_ast1'][ii], wgts )
            x_avg, x_std = weighted_avg_and_std(x_no_pm[ii], wgts )
            y_avg, y_std = weighted_avg_and_std(y_no_pm[ii], wgts )

            t_avg = data['t_ast1'][ii].mean()
            t_std = data['t_ast1'][ii].std()
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
    plt.scatter(x_no_pm*1e3, y_no_pm*1e3, c=data['t_ast1'],
                cmap=cmap, norm=norm, s=10)
    plt.errorbar(x_no_pm*1e3, y_no_pm*1e3,
                 xerr=data['xpos_err1']*1e3, yerr=data['ypos_err1']*1e3,
                 fmt='none', ecolor=smap.to_rgba(data['t_ast1']))
    plt.scatter(x_mod_no_pm*1e3, y_mod_no_pm*1e3, c=t_mod, cmap=cmap, norm=norm, s=4)
    plt.gca().invert_xaxis()
    plt.axis('equal')
    plt.xlabel(r'$\Delta \alpha^*$ (mas)')
    plt.ylabel(r'$\Delta \delta$ (mas)')
    plt.colorbar()

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

    plt.close(1)
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

    plt.close(1)
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

    plt.close(2)
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
    t = Table.read(popsycle_events)

    mas_to_rad = 4.848 * 10**-9

    bh_idx = np.where(t['rem_id_L'] == 103)[0]
    ns_idx = np.where(t['rem_id_L'] == 102)[0]
    wd_idx = np.where((t['rem_id_L'] == 101) |
                      (t['rem_id_L'] == 6))[0]
    st_idx = np.where(t['rem_id_L'] == 0)[0]

    # start with a rectangular Figure
    plt.close(15)
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

    plt.close(1)
    fig_a = plt.figure(1, figsize=(6,6))
    fig_a.clf()

    f1 = fig_a.add_axes([0.2, 0.33, 0.75, 0.6])
    f2 = fig_a.add_axes([0.2, 0.13, 0.75, 0.2])

    # Amplification Curve
    f1.plot(t_mod, amp_mod, '-', label='model')
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

    plt.close(2)
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

def get_num_ref_stars():
    all_targets = epochs.keys()
    
    print('Number of reference stars for transformation:')
    
    for target in all_targets:
        ast_data_file = astrom_data[target]
        data5 = Table.read(ast_data_file)

        if target != 'ob150029':
            ast_data_file = ast_data_file.replace('p5', 'p4')
            data4 = Table.read(ast_data_file)
        else:
            data4 = data5

        print('{0:s} N={1:d} N_final={2:d}'.format(target,
                                                    data4['use_in_trans'].sum(),
                                                    data5['use_in_trans'].sum()))
        
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
        plt.close(1)
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
        
def get_num_data_points(data_dict, astrometryFlag=True, verbose=False):
    N_data = 0

    # Loop through photometry data
    for pp in range(len(data_dict['phot_files'])):
        N_phot_pp = len(data_dict['t_phot{0:d}'.format(pp+1)])
        if verbose: print('N_phot_pp = ', N_phot_pp)
        N_data += N_phot_pp

    # Loop through astrometry data
    if astrometryFlag:
        for aa in range(len(data_dict['ast_files'])):
            # Multiply astrometry by 2 to account for X and Y independent positions.
            if len(data_dict['ast_files']) > 1:
                N_ast_aa = 2 * len(data_dict['t_ast{0:d}'.format(aa+1)])
            else:
                N_ast_aa = 2 * len(data_dict['t_ast1'])
                
            if verbose: print('N_ast_aa = ', N_ast_aa)
            N_data += N_ast_aa

    if verbose: print('N_data = ', N_data)
    
    return N_data

