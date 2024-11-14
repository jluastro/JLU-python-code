import time
import numpy as np
import pylab as plt
from numpy import random
from astropy.table import Table
from astropy import table
from spisea import synthetic as syn
from spisea import atmospheres as atm
from spisea import evolution
from spisea import reddening
from spisea import ifmr
from spisea.imf import imf
from spisea.imf import multiplicity
from pysynphot import spectrum
from matplotlib.patches import FancyArrow
import pickle
from scipy.interpolate import interp1d
from astropy.io import fits
from astropy.visualization import SqrtStretch
from astropy.visualization import ZScaleInterval
from astropy.visualization import make_lupton_rgb
from astropy.wcs import WCS
from astropy.io import fits
from astropy import units as u
import pysynphot as S
from astropy.convolution import convolve_fft
from astropy.convolution import Gaussian1DKernel
import os
import pandeia.engine
import json
import copy
import time
import math
from pandeia.engine.perform_calculation import perform_calculation
from functools import reduce
from jlu.gc.gcwork import wobble

saturate = {'F070W': 14.81,
            'F090W': 15.35,
            'F115W': 15.47,
            'F140M': 14.81,
            'F150W': 15.63,
            'F150W2': 17.01,
            'F162M': 14.65,
            'F164N': 12.08,
            'F182M': 14.78,
            'F187N': 11.50,
            'F200W': 15.11,
            'F210M': 14.21,
            'F212N': 11.44,
            'F250M': 14.50,
            'F277W': 15.48,
            'F300M': 14.22,
            'F322W2': 15.80,
            'F323N': 11.13,
            'F335M': 13.92,
            'F356W': 14.76,
            'F360M': 13.72,
            'F405N': 10.62,
            'F410M': 13.17,
            'F430M': 12.20,
            'F444W': 13.82,
            'F460M': 11.80,
            'F466N': 9.58,
            'F470N': 9.69,
            'F480M': 12.00}

#work_dir = '/u/jlu/work/gc/jwst/2022_01_20/'
#work_dir = '/u/jlu/work/gc/jwst/2023_10_10/'
work_dir = '/u/jlu/work/gc/jwst/2024_10_11/'

def plot_sim_cluster():
    yng_file = work_dir + 'iso_gc_jwst/iso_6.61_2.70_08000_p00.fits'
    old_file = work_dir + 'iso_gc_jwst/iso_9.90_2.70_08000_p00.fits'
    out_dir = work_dir + 'plots/'

    yng_clust_file = work_dir + 'iso_gc_jwst/clust_6.60_2.10_08000.fits'
    med_clust_file = work_dir + 'iso_gc_jwst/clust_8.00_2.10_08000.fits'
    old_clust_file = work_dir + 'iso_gc_jwst/clust_9.90_2.10_08000.fits'

    # Isochrones
    yng_all = Table.read(yng_file)
    old_all = Table.read(old_file)

    # Clusters
    # yng_c = Table.read(yng_clust_file)
    # old_c = Table.read(old_clust_file)

    # Filter out compact object phases
    ygood = np.where(yng_all['phase'] < 7)[0]
    ogood = np.where(old_all['phase'] < 7)[0]
    yng = yng_all[ygood]
    old = old_all[ogood]

    print('Yng mean F182M - Kp = {0:.2f}'.format((yng['m_jwst_F182M'] - yng['m_nirc2_Kp']).mean()))
    print('Old mean F182M - Kp = {0:.2f}'.format((old['m_jwst_F182M'] - old['m_nirc2_Kp']).mean()))

    ## Mass-Luminosity Relationship at ~K-band
    plt.close('all')
    plt.figure(1, figsize=(5,5))
    plt.semilogx(yng['mass'], yng['m_jwst_F090W'], 'c.', label='F090W')
    plt.semilogx(yng['mass'], yng['m_jwst_F115W'], 'b.', label='F115W')
    plt.semilogx(yng['mass'], yng['m_jwst_F164N'], 'y.', label='F164N')
    plt.semilogx(yng['mass'], yng['m_jwst_F182M'], 'k.', label='F182M', color='orange')
    plt.semilogx(yng['mass'], yng['m_jwst_F212N'], 'g.', label='F212N')
    plt.semilogx(yng['mass'], yng['m_jwst_F323N'], 'r.', label='F323N')
    plt.semilogx(yng['mass'], yng['m_jwst_F470N'], 'm.', label='F470N')
    plt.axhline(saturate['F090W'], color='c', linestyle='--')
    plt.axhline(saturate['F115W'], color='b', linestyle='--')
    plt.axhline(saturate['F164N'], color='y', linestyle='--')
    plt.axhline(saturate['F182M'], color='orange', linestyle='--')
    plt.axhline(saturate['F212N'], color='g', linestyle='--')
    plt.axhline(saturate['F323N'], color='r', linestyle='--')
    plt.axhline(saturate['F470N'], color='m', linestyle='--')
    plt.gca().invert_yaxis()
    plt.ylim(30, 6)
    plt.xlabel('Mass (Msun)')
    plt.ylabel('JWST Magnitudes')
    plt.legend(loc='upper left', numpoints=1)
    plt.savefig(out_dir + 'mass_luminosity.png')

    ## CMD
    plt.figure(figsize=(5,5))
    plt.clf()
    plt.plot(yng['m_jwst_F090W'] - yng['m_jwst_F323N'], yng['m_jwst_F323N'], 'k.')
    plt.plot(old['m_jwst_F090W'] - old['m_jwst_F323N'], old['m_jwst_F323N'], 'r.')
    plt.ylim(25, 8)
    plt.xlabel('F090W - F323N (mag)')
    plt.ylabel('F323N (mag)')
    plt.savefig(out_dir + 'gc_cmd_F090W_F323N.png')

    plt.figure(figsize=(5,5))
    plt.clf()
    plt.plot(yng['m_jwst_F115W'] - yng['m_jwst_F323N'], yng['m_jwst_F323N'], 'k.')
    plt.plot(old['m_jwst_F115W'] - old['m_jwst_F323N'], old['m_jwst_F323N'], 'r.')
    plt.ylim(25, 8)
    plt.xlabel('F115W - F323N (mag)')
    plt.ylabel('F323N (mag)')
    plt.savefig(out_dir + 'gc_cmd_F115W_F323N.png')
    
    plt.figure(figsize=(5,5))
    plt.clf()
    plt.plot(yng['m_jwst_F090W'] - yng['m_jwst_F212N'], yng['m_jwst_F212N'], 'k.')
    plt.plot(old['m_jwst_F090W'] - old['m_jwst_F212N'], old['m_jwst_F212N'], 'r.')
    plt.ylim(25, 8)
    plt.xlabel('F090W - F212N (mag)')
    plt.ylabel('F212N (mag)')
    plt.savefig(out_dir + 'gc_cmd_F090W_F212N.png')

    plt.figure(figsize=(5,5))
    plt.clf()
    plt.plot(yng['m_jwst_F115W'] - yng['m_jwst_F212N'], yng['m_jwst_F212N'], 'k.')
    plt.plot(old['m_jwst_F115W'] - old['m_jwst_F212N'], old['m_jwst_F212N'], 'r.')
    plt.ylim(25, 8)
    plt.xlabel('F115W - F212N (mag)')
    plt.ylabel('F212N (mag)')
    plt.savefig(out_dir + 'gc_cmd_F115W_F212N.png')

    plt.figure(figsize=(5,5))
    plt.clf()
    plt.plot(yng['m_jwst_F090W'] - yng['m_jwst_F182M'], yng['m_jwst_F182M'], 'k.')
    plt.plot(old['m_jwst_F090W'] - old['m_jwst_F182M'], old['m_jwst_F182M'], 'r.')
    plt.ylim(25, 8)
    plt.xlabel('F090W - F182M (mag)')
    plt.ylabel('F182M (mag)')
    plt.savefig(out_dir + 'gc_cmd_F090W_F182M.png')

    plt.figure(figsize=(5,5))
    plt.clf()
    plt.plot(yng['m_jwst_F115W'] - yng['m_jwst_F182M'], yng['m_jwst_F182M'], 'k.')
    plt.plot(old['m_jwst_F115W'] - old['m_jwst_F182M'], old['m_jwst_F182M'], 'r.')
    plt.ylim(25, 8)
    plt.xlabel('F115W - F182M (mag)')
    plt.ylabel('F182M (mag)')
    plt.savefig(out_dir + 'gc_cmd_F115W_F182M.png')

    plt.figure(figsize=(5,5))
    plt.clf()
    plt.plot(yng['m_jwst_F212N'] - yng['m_jwst_F470N'], yng['m_jwst_F212N'], 'k.')
    plt.plot(old['m_jwst_F212N'] - old['m_jwst_F470N'], old['m_jwst_F212N'], 'r.')
    plt.ylim(25, 8)
    plt.xlabel('F212N - F470N (mag)')
    plt.ylabel('F212N (mag)')
    plt.savefig(out_dir + 'gc_cmd_F212N_F470N.png')

    plt.figure(figsize=(5,5))
    plt.clf()
    plt.plot(yng['m_jwst_F212N'] - yng['m_jwst_F323N'], yng['m_jwst_F212N'], 'k.')
    plt.plot(old['m_jwst_F212N'] - old['m_jwst_F323N'], old['m_jwst_F212N'], 'r.')
    plt.ylim(25, 8)
    plt.xlabel('F212N - F323N (mag)')
    plt.ylabel('F212N (mag)')
    plt.savefig(out_dir + 'gc_cmd_F212N_F323N.png')
    
    plt.figure(figsize=(5,5))
    plt.clf()
    plt.plot(yng['m_jwst_F090W'] - yng['m_jwst_F212N'], yng['m_jwst_F212N'] - yng['m_jwst_F323N'], 'k.')
    plt.plot(old['m_jwst_F090W'] - old['m_jwst_F212N'], old['m_jwst_F212N'] - old['m_jwst_F323N'], 'r.')
    plt.xlabel('F090W - F212N (mag)')
    plt.ylabel('F212N - F323N (mag)')
    plt.savefig(out_dir + 'gc_ccd_F090W_F212N_F323N.png')

    plt.figure(figsize=(5,5))
    plt.clf()
    plt.plot(yng['m_jwst_F115W'] - yng['m_jwst_F212N'], yng['m_jwst_F212N'] - yng['m_jwst_F323N'], 'k.')
    plt.plot(old['m_jwst_F115W'] - old['m_jwst_F212N'], old['m_jwst_F212N'] - old['m_jwst_F323N'], 'r.')
    plt.xlabel('F115W - F212N (mag)')
    plt.ylabel('F212N - F323N (mag)')
    plt.savefig(out_dir + 'gc_ccd_F115W_F212N_F323N.png')

    plt.figure(figsize=(5,5))
    plt.clf()
    plt.plot(yng['m_jwst_F115W'] - yng['m_jwst_F182M'], yng['m_jwst_F182M'] - yng['m_jwst_F323N'], 'k.')
    plt.plot(old['m_jwst_F115W'] - old['m_jwst_F182M'], old['m_jwst_F182M'] - old['m_jwst_F323N'], 'r.')
    plt.xlabel('F115W - F182M (mag)')
    plt.ylabel('F182M - F323N (mag)')
    plt.savefig(out_dir + 'gc_ccd_F115W_F182M_F323N.png')
    
    plt.figure(figsize=(5,5))
    plt.clf()
    plt.plot(yng['m_jwst_F090W'] - yng['m_jwst_F182M'], yng['m_jwst_F212N'], 'k.')
    plt.plot(old['m_jwst_F090W'] - old['m_jwst_F182M'], old['m_jwst_F212N'], 'r.')
    plt.ylim(30, 8)
    plt.xlabel('F090W - F182M (mag)')
    plt.ylabel('F212N (mag)')
    plt.savefig(out_dir + 'gc_cmd_F090W_F182M.png')

    return yng


def make_sim_cluster():
    ages = [4e6, 1e8, 8e9]
    cluster_mass = [1e4, 1e4, 1e7]
    AKs = 2.1
    deltaAKs = 0.05
    distance = 8000
    mass_sampling = 5

    isochrones = []
    clusters = []
    
    evo = evolution.MergedBaraffePisaEkstromParsec()
    atm_func = atm.get_merged_atmosphere
    red_law = reddening.RedLawHosek18b()
    multi = multiplicity.MultiplicityUnresolved()

    imf_mass_limits = np.array([0.07, 0.5, 1, np.inf])
    imf_powers_old = np.array([-1.3, -2.3, -2.3])
    imf_powers_yng = np.array([-1.3, -1.8, -1.8])
    my_imf_old = imf.IMF_broken_powerlaw(imf_mass_limits, imf_powers_old,
                                         multiplicity=multi)
    my_imf_yng = imf.IMF_broken_powerlaw(imf_mass_limits, imf_powers_yng,
                                         multiplicity=multi)

    # Test all filters
    filt_list = ['wfc3,ir,f127m', 'wfc3,ir,f139m', 'wfc3,ir,f153m', 'acs,wfc1,f814w',
                     'wfc3,ir,f125w', 'wfc3,ir,f160w', 
                     'jwst,F090W', 'jwst,F115W', 
                     'jwst,F164N', 'jwst,F187N', 'jwst,F212N', 'jwst,F323N',
                     'jwst,F405N', 'jwst,F466N', 'jwst,F470N',
                     'jwst,F140M', 'jwst,F162M', 'jwst,F182M', 'jwst,F210M', 'jwst,F250M',
                     'jwst,F300M', 'jwst,F335M', 'jwst,F360M', 'jwst,F410M', 'jwst,F430M',
                     'jwst,F460M', 'jwst,F480M',
                     'nirc2,J', 'nirc2,H', 'nirc2,Kp',
                     'nirc2,Lp', 'nirc2,Ms']
    
    startTime = time.time()
    for ii in range(len(ages)):
        logAge = np.log10(ages[ii])
        
        iso = syn.IsochronePhot(logAge, AKs, distance,
                                evo_model=evo, atm_func=atm_func,
                                red_law=red_law, filters=filt_list,
                                mass_sampling=mass_sampling, iso_dir=work_dir)

        time2 = time.time()
        print( 'Constructed isochrone: %d seconds' % (time2 - startTime))

        if ii < 2:
            imf_ii = my_imf_yng
        else:
            imf_ii = my_imf_old
            
        cluster = syn.ResolvedClusterDiffRedden(iso, imf_ii, cluster_mass[ii], deltaAKs)

        time3 = time.time()
        print( 'Constructed cluster: %d seconds' % (time3 - time2))

        # Save generated clusters to file.
        save_file_fmt = '{0}clust_{1:.2f}_{2:4.2f}_{3:4s}'
        save_file_txt = save_file_fmt.format(work_dir, logAge, AKs, str(distance).zfill(5))
        cluster.save_to_file(save_file_txt)        

    return
        
        
    
def plot_cycle1_fig1():
    iso_dir = work_dir + 'iso_gc_jwst/'
    out_dir = work_dir + 'plots/'

    import glob
    # iso_files = glob.glob(iso_dir + 'iso_*.fits')
    # cl_files = glob.glob(iso_dir + 'clu_r*.pkl')

    iso_files = ['iso_6.90_2.70_08000_p00.fits', 'iso_6.60_2.70_08000_p00.fits', 
                 'iso_9.90_2.70_08000_p00.fits', 'iso_9.90_2.70_08000_m10.fits',
                 'iso_9.00_2.70_08000_p00.fits', 'iso_10.00_2.70_08000_p00.fits']
    clu_files = [iso_f.replace('iso', 'clu').replace('.fits', '.pkl') for iso_f in iso_files]
    iso_files = [iso_dir + iso_f for iso_f in iso_files]
    clu_files = [iso_dir + 'with_diff_reddening_0.1/' + clu_f for clu_f in clu_files]
    # clu_files = [iso_dir + clu_f for clu_f in clu_files]

    colors = ['cyan', 'blue', 'red', 'salmon', 'darkorange', 'maroon']

    # Reverse order for plotting
    iso_files = iso_files[::-1]
    clu_files = clu_files[::-1]
    colors = colors[::-1]

    iso = []
    clu = []

    for ff in range(len(iso_files)):
        print(iso_files[ff])
        iso_tmp = Table.read(iso_files[ff])
        clu_tmp = pickle.load(open(clu_files[ff], 'rb'))

        good = np.where(iso_tmp['phase'] < 7)[0]
        iso.append(iso_tmp[good])

        # Cut the stars out below F212N > 24 (gets rid of a bunch).
        stars = clu_tmp.star_systems
        bdx = np.where(stars['m_jwst_F212N'] <= 24)[0]
        stars = stars[bdx]

        # Add some noise to the filters we will observe in.
        # snr is calculated in jwst_gc_etc.ipynb and is calibrated relative to a K-band magnitude.
        snr_f115w = 979.0  * 10**(-(stars['m_nirc2_Kp'] - 14.00) / 5)
        snr_f182m = 1228.0 * 10**(-(stars['m_nirc2_Kp'] - 14.00) / 5)
        snr_f212n = 1066.0 * 10**(-(stars['m_nirc2_Kp'] - 14.00) / 5)        
        snr_f323n = 1102.0 * 10**(-(stars['m_nirc2_Kp'] - 14.00) / 5)
        snr_f405n = 1025.0 * 10**(-(stars['m_nirc2_Kp'] - 14.00) / 5)

        stars['m_jwst_F115W'] += np.random.randn(len(stars)) / snr_f115w
        stars['m_jwst_F182M'] += np.random.randn(len(stars)) / snr_f182m
        stars['m_jwst_F212N'] += np.random.randn(len(stars)) / snr_f323n
        stars['m_jwst_F323N'] += np.random.randn(len(stars)) / snr_f323n
        # stars['m_jwst_F405N'] += np.random.randn(len(stars)) / snr_f405n

        # Also perturb based on extinction uncertainties.
        stars['m_jwst_F115W'] += np.random.randn(len(stars)) * 0.050
        stars['m_jwst_F182M'] += np.random.randn(len(stars)) * 0.020
        stars['m_jwst_F212N'] += np.random.randn(len(stars)) * 0.020
        stars['m_jwst_F323N'] += np.random.randn(len(stars)) * 0.015
        # stars['m_jwst_F405N'] += np.random.randn(len(stars)) * 0.010
        

        clu_tmp.star_systems = stars
        
        clu.append(clu_tmp)

    ## CMD
    plt.close(10)
    plt.figure(10, figsize=(16, 7))
    plt.clf()
    plt.subplots_adjust(left=0.12, wspace=0.25, top=0.65, right=0.95)

    plt.subplot(1, 3, 1)
    for ss in range(len(iso)):
        label = 'Age = {0:5.3f} Gyr, [Fe/H] = {1:3.1f}, d = {2:3.1f} kpc'
        label = label.format(10**iso[ss].meta['LOGAGE'] / 1e9,
                             iso[ss].meta['METAL_IN'],
                             iso[ss].meta['DISTANCE'] / 1e3)
        plt.plot(iso[ss]['m_jwst_F115W'] - iso[ss]['m_jwst_F212N'], iso[ss]['m_jwst_F212N'],
                     color=colors[ss], label=label)

    plt.ylim(23, 9)
    plt.xlim(6.5, 8.5)
    plt.xlabel('F115W - F212N (mag)')
    plt.ylabel('F212N (mag)')

    confusion = {'F090W': 32, 
                 'F115W': 27.2,
                 'F212N': 21.5,
                 'F323N': 19.3}
                 # 'F212N': 22.5,
        
    # Modified the F212N one to be a bit closer in.
    
    plt.axhline(saturate['F212N'], color='black', linestyle='--')
    plt.text(6.9, saturate['F212N']-0.2, 'JWST Saturation Limit', fontsize=10)
    
    plt.fill_between([6.5, 8.5], [confusion['F212N'], confusion['F212N']], [confusion['F212N']-1.5, confusion['F212N']-1.5],
                         color='black', alpha=0.2)
    plt.text(6.6, confusion['F212N']-0.2, 'JWST Confusion Limit', fontsize=10)
    
    plt.axhline(18, color='grey', linestyle='--')
    plt.text(8.0, 18-0.2, 'HST Imaging', fontsize=10)
    
    plt.axhline(19, color='grey', linestyle='--')
    plt.text(8.0, 19-0.2, 'Keck Imaging', fontsize=10)
    
    plt.axhline(15, color='grey', linestyle='--')
    plt.text(8.0, 15-0.2, 'Keck Spectra', fontsize=10)
    
    # plt.axhline(21.5, color='black', linestyle='--')
    # plt.text(8.0, 21.5-0.2, 'JWST', fontsize=10, color='black')
                    
    lg = plt.legend(fontsize=12, bbox_to_anchor=(1.12, 1.14), markerscale=3, loc='center', ncol=2)
    # for lh in lg.legendHandles: 
    #     lh._legmarker.set_alpha(1)

    def label_mass(iso, col_color, col_mag,
                   mass_val, mass_label, label_dcol=-1.0, color='blue'):
        idx = np.argmin(np.abs(iso['mass'] - mass_val))
        print(mass_val, idx, iso['mass'][idx])

        m_label = '{0:s} M$_\odot$'.format(mass_label)
        xy = (col_color[idx], col_mag[idx])
        xytext = (col_color[idx] + label_dcol, col_mag[idx])
        arrow_props = dict(facecolor=color, edgecolor=color,
                           shrink=0.05, width=1, headwidth=2)
        plt.annotate(m_label, xy=xy, xytext=xytext,
                     color=color,
                     arrowprops=arrow_props)

    # label_mass(yng, ycol, ymag, 10, '10')
    # label_mass(yng, ycol, ymag, 5, '5')
    # label_mass(yng, ycol, ymag, 2, '2')
    # label_mass(yng, ycol, ymag, 1, '1', label_dcol=0.4)
    # label_mass(yng, ycol, ymag, 0.5, '0.5', label_dcol=0.4)

    # label_mass(old, ocol, omag, 1, '1', color='red')
    # label_mass(old, ocol, omag, 2, '2', color='red')
    # label_mass(old, ocol, omag, 0.5, '0.5', label_dcol=0.4, color='red')
    
    # plt.legend(fontsize=12, loc='lower left')

    
    plt.subplot(1, 3, 2)
    for ss in range(len(clu)-1):
        label = 'Age = {0:5.3f} Gyr, [Fe/H] = {1:4.1f}, d = {2:3.1f} kpc'
        label = label.format(10**clu[ss].iso.points.meta['LOGAGE'] / 1e9,
                             clu[ss].iso.points.meta['METAL_IN'],
                             clu[ss].iso.points.meta['DISTANCE'] / 1e3)


        stars = clu[ss].star_systems

        # Pick out a random sample of "samp_size" for plotting.
        # Too many to plot all of them.
        if clu[ss].iso.points.meta['LOGAGE'] < 9.1:
            samp_size = int(2e4)
        else:
            samp_size = int(2e5)

        if clu[ss].iso.points.meta['METAL_IN'] < 0:
            samp_size = int(1e4)
        print(samp_size)
        
        sdx = np.random.randint(0, high=len(stars), size=samp_size)
        print(sdx)

        plt.plot(stars['m_jwst_F115W'][sdx] - stars['m_jwst_F212N'][sdx],
                     stars['m_jwst_F212N'][sdx], marker='.', linestyle='none',
                     label=label, alpha=0.1, markersize=3, color=colors[ss])
    
    plt.ylim(23, 9)
    plt.xlim(6.5, 8.5)
    plt.xlabel('F115W - F212N (mag)')
    plt.ylabel('F212N (mag)')
    
    # plt.axhline(saturate['F212N'], color='black', linestyle='--')
    # plt.text(6.7, saturate['F212N']-0.2, 'Saturation', fontsize=10)
    
    # plt.axhline(confusion['F212N'], color='black', linestyle='--')
    # plt.text(6.7, confusion['F212N']-0.2, 'Confusion', fontsize=10)
    
    plt.axhline(20, color='black', linestyle='--')
    plt.text(8.0, 20-0.2, 'JWST Imaging', fontsize=10, color='black')
    
    plt.axhline(18, color='black', linestyle='--')
    plt.text(8.0, 18-0.2, 'JWST Spectra', fontsize=10, color='black')
    
    plt.savefig(out_dir + 'cycle1_cmd.png')

    return

def plot_cycle1_fig1_f182m():
    iso_dir = work_dir + 'iso_gc_jwst/'
    out_dir = work_dir + 'plots/'

    import glob
    # iso_files = glob.glob(iso_dir + 'iso_*.fits')
    # cl_files = glob.glob(iso_dir + 'clu_r*.pkl')

    iso_files = ['iso_6.90_2.70_08000_p00.fits', 'iso_6.60_2.70_08000_p00.fits', 
                 'iso_9.90_2.70_08000_p00.fits', 'iso_9.90_2.70_08000_m10.fits',
                 'iso_9.00_2.70_08000_p00.fits', 'iso_10.00_2.70_08000_p00.fits']
    clu_files = [iso_f.replace('iso', 'clu').replace('.fits', '.pkl') for iso_f in iso_files]
    iso_files = [iso_dir + iso_f for iso_f in iso_files]
    # clu_files = [iso_dir + 'with_diff_reddening/' + clu_f for clu_f in clu_files]
    clu_files = [iso_dir + clu_f for clu_f in clu_files]

    colors = ['cyan', 'blue', 'red', 'salmon', 'darkorange', 'maroon']

    # Reverse order for plotting
    iso_files = iso_files[::-1]
    clu_files = clu_files[::-1]
    colors = colors[::-1]

    iso = []
    clu = []

    for ff in range(len(iso_files)):
        print(iso_files[ff])
        iso_tmp = Table.read(iso_files[ff])
        clu_tmp = pickle.load(open(clu_files[ff], 'rb'))

        good = np.where(iso_tmp['phase'] < 7)[0]
        iso.append(iso_tmp[good])

        # Cut the stars out below F212N > 24 (gets rid of a bunch).
        stars = clu_tmp.star_systems
        bdx = np.where(stars['m_jwst_F212N'] <= 24)[0]
        stars = stars[bdx]

        # Add some noise to the filters we will observe in.
        # snr is calculated in jwst_gc_etc.ipynb and is calibrated relative to a K-band magnitude.
        snr_f115w = 979.0  * 10**(-(stars['m_nirc2_Kp'] - 14.00) / 5)
        snr_f182m = 1228.0 * 10**(-(stars['m_nirc2_Kp'] - 14.00) / 5)
        snr_f323n = 1102.0 * 10**(-(stars['m_nirc2_Kp'] - 14.00) / 5)
        snr_f405n = 679.0  * 10**(-(stars['m_nirc2_Kp'] - 14.00) / 5)

        stars['m_jwst_F115W'] += np.random.randn(len(stars)) / snr_f115w
        stars['m_jwst_F182M'] += np.random.randn(len(stars)) / snr_f182m
        stars['m_jwst_F323N'] += np.random.randn(len(stars)) / snr_f323n
        # stars['m_jwst_F405N'] += np.random.randn(len(stars)) / snr_f405n

        # Also perturb based on extinction uncertainties.
        stars['m_jwst_F115W'] += np.random.randn(len(stars)) * 0.050
        stars['m_jwst_F182M'] += np.random.randn(len(stars)) * 0.020
        stars['m_jwst_F323N'] += np.random.randn(len(stars)) * 0.015
        # stars['m_jwst_F405N'] += np.random.randn(len(stars)) * 0.010
        

        clu_tmp.star_systems = stars
        
        clu.append(clu_tmp)

    ## CMD
    plt.close(10)
    plt.figure(10, figsize=(12, 7))
    plt.clf()
    plt.subplots_adjust(left=0.1, wspace=0.25, top=0.65, right=0.95)

    print(saturate['F090W'], saturate['F323N'], saturate['F090W']-saturate['F323N'])

    plt.subplot(1, 2, 1)
    for ss in range(len(iso)):
        label = 'Age = {0:5.3f} Gyr, [Fe/H] = {1:3.1f}, d = {2:3.1f} kpc'
        label = label.format(10**iso[ss].meta['LOGAGE'] / 1e9,
                             iso[ss].meta['METAL_IN'],
                             iso[ss].meta['DISTANCE'] / 1e3)
        plt.plot(iso[ss]['m_jwst_F115W'] - iso[ss]['m_jwst_F182M'], iso[ss]['m_jwst_F182M'],
                     color=colors[ss], label=label)

    plt.ylim(26, 10)
    plt.xlim(5.5, 7.5)
    plt.xlabel('F115W - F182M (mag)')
    plt.ylabel('F182MN (mag)')

    confusion = {'F090W': 32, 
                 'F115W': 27.2,
                 'F182M': 24.5,
                 'F212N': 22.5,
                 'F323N': 19.3}
    # Modified the F212N one to be a bit closer in.
    
    plt.axhline(saturate['F182M'], color='black', linestyle='--')
    plt.text(6.7, saturate['F182M']-0.2, 'Saturation Limit', fontsize=10)
    # plt.axhline(confusion['F212N'], color='black', linestyle='--')
    
    plt.fill_between([5.5, 7.5], [confusion['F182M'], confusion['F182M']], [confusion['F182M']-1.5, confusion['F182M']-1.5],
                         color='black', alpha=0.2)
    plt.text(5.7, confusion['F182M']-0.2, 'Confusion Limit', fontsize=10)
    
    plt.axhline(19, color='grey', linestyle='--')
    plt.text(7.0, 18-0.2, 'HST Imaging', fontsize=10)
    
    plt.axhline(20, color='grey', linestyle='--')
    plt.text(7.0, 20-0.2, 'Keck Imaging', fontsize=10)

    plt.axhline(15, color='grey', linestyle='--')
    plt.text(7.0, 15-0.2, 'Keck Spectra', fontsize=10)
    
    # plt.axhline(22.5, color='black', linestyle='--')
    # plt.text(7.0, 22.5-0.2, 'JWST', fontsize=10, color='black')

    lg = plt.legend(fontsize=12, bbox_to_anchor=(1.12, 1.14), markerscale=3, loc='center', ncol=2)
    # for lh in lg.legendHandles: 
    #     lh._legmarker.set_alpha(1)
                    

    def label_mass(iso, col_color, col_mag,
                   mass_val, mass_label, label_dcol=-1.0, color='blue'):
        idx = np.argmin(np.abs(iso['mass'] - mass_val))
        print(mass_val, idx, iso['mass'][idx])

        m_label = '{0:s} M$_\odot$'.format(mass_label)
        xy = (col_color[idx], col_mag[idx])
        xytext = (col_color[idx] + label_dcol, col_mag[idx])
        arrow_props = dict(facecolor=color, edgecolor=color,
                           shrink=0.05, width=1, headwidth=2)
        plt.annotate(m_label, xy=xy, xytext=xytext,
                     color=color,
                     arrowprops=arrow_props)

    # label_mass(yng, ycol, ymag, 10, '10')
    # label_mass(yng, ycol, ymag, 5, '5')
    # label_mass(yng, ycol, ymag, 2, '2')
    # label_mass(yng, ycol, ymag, 1, '1', label_dcol=0.4)
    # label_mass(yng, ycol, ymag, 0.5, '0.5', label_dcol=0.4)

    # label_mass(old, ocol, omag, 1, '1', color='red')
    # label_mass(old, ocol, omag, 2, '2', color='red')
    # label_mass(old, ocol, omag, 0.5, '0.5', label_dcol=0.4, color='red')
    
    # plt.legend(fontsize=12, loc='lower left')

    
    plt.subplot(1, 2, 2)
    for ss in range(len(clu)-1):
        label = 'Age = {0:5.3f} Gyr, [Fe/H] = {1:4.1f}, d = {2:3.1f} kpc'
        label = label.format(10**clu[ss].iso.points.meta['LOGAGE'] / 1e9,
                             clu[ss].iso.points.meta['METAL_IN'],
                             clu[ss].iso.points.meta['DISTANCE'] / 1e3)


        stars = clu[ss].star_systems

        # Pick out a random sample of "samp_size" for plotting.
        # Too many to plot all of them.
        if clu[ss].iso.points.meta['LOGAGE'] < 9.1:
            samp_size = int(2e4)
        else:
            samp_size = int(2e5)
        
        sdx = np.random.randint(0, high=len(stars), size=samp_size)
        print(sdx)

        plt.plot(stars['m_jwst_F115W'][sdx] - stars['m_jwst_F182M'][sdx],
                     stars['m_jwst_F182M'][sdx], marker='.', linestyle='none',
                     label=label, alpha=0.1, markersize=3, color=colors[ss])
    
    # plt.ylim(25, 10)
    # plt.xlim(5.5, 7.5)
    plt.ylim(21, 11)
    plt.xlim(0, 13)
    plt.xlabel('F115W - F182M (mag)')
    plt.ylabel('F182M (mag)')
    
    # plt.axhline(saturate['F182M'], color='black', linestyle='--')
    # plt.text(5.7, saturate['F182M']-0.2, 'Saturation', fontsize=10)
    
    # plt.axhline(confusion['F182M'], color='black', linestyle='--')
    # plt.text(5.7, confusion['F182M']-0.2, 'Confusion', fontsize=10)
    
    plt.axhline(22.5, color='black', linestyle='--')
    plt.text(7.0, 22.5-0.2, 'JWST Imaging', fontsize=10, color='black')

    plt.axhline(19, color='black', linestyle='--')
    plt.text(7.0, 19-0.2, 'JWST Spectra', fontsize=10, color='black')
    
    
    plt.savefig(out_dir + 'cycle1_cmd_f182m.png')

def plot_cycle2_fig1(iso=None, clu=None):
    iso_dir = work_dir + 'iso_gc_jwst/'
    out_dir = work_dir + 'plots/'

    import glob
    iso_files = ['iso_6.90_2.70_08000_p00.fits', 'iso_6.60_2.70_08000_p00.fits', 
                 'iso_9.90_2.70_08000_p00.fits', 'iso_9.90_2.70_08000_m10.fits',
                 'iso_9.00_2.70_08000_p00.fits', 'iso_10.00_2.70_08000_p00.fits']
    clu_files = [iso_f.replace('iso', 'clu').replace('.fits', '.pkl') for iso_f in iso_files]
    iso_files = [iso_dir + iso_f for iso_f in iso_files]
    clu_files = [iso_dir + 'with_diff_reddening_0.1/' + clu_f for clu_f in clu_files]
    # clu_files = [iso_dir + clu_f for clu_f in clu_files]

    colors = ['cyan', 'blue', 'red', 'salmon', 'darkorange', 'maroon']

    # Reverse order for plotting
    iso_files = iso_files[::-1]
    clu_files = clu_files[::-1]
    colors = colors[::-1]

    if iso == None and clu == None:
        iso = []
        clu = []

        for ff in range(len(iso_files)):
            print(iso_files[ff])
            iso_tmp = Table.read(iso_files[ff])
            clu_tmp = pickle.load(open(clu_files[ff], 'rb'))
            
            good = np.where(iso_tmp['phase'] < 7)[0]
            iso.append(iso_tmp[good])

            # Cut the stars out below F212N > 24 (gets rid of a bunch).
            stars = clu_tmp.star_systems
            bdx = np.where(stars['m_jwst_F212N'] <= 24)[0]
            stars = stars[bdx]

            # Add some noise to the filters we will observe in.
            # snr is calculated in jwst_gc_etc.ipynb and is calibrated relative to a K-band magnitude.
            snr_f115w = 979.0  * 10**(-(stars['m_nirc2_Kp'] - 14.00) / 5)
            snr_f182m = 1228.0 * 10**(-(stars['m_nirc2_Kp'] - 14.00) / 5)
            snr_f212n = 1066.0 * 10**(-(stars['m_nirc2_Kp'] - 14.00) / 5)        
            snr_f323n = 1102.0 * 10**(-(stars['m_nirc2_Kp'] - 14.00) / 5)
            snr_f405n = 1025.0 * 10**(-(stars['m_nirc2_Kp'] - 14.00) / 5)

            stars['m_jwst_F115W'] += np.random.randn(len(stars)) / snr_f115w
            stars['m_jwst_F182M'] += np.random.randn(len(stars)) / snr_f182m
            stars['m_jwst_F212N'] += np.random.randn(len(stars)) / snr_f323n
            stars['m_jwst_F323N'] += np.random.randn(len(stars)) / snr_f323n
            # stars['m_jwst_F405N'] += np.random.randn(len(stars)) / snr_f405n

            # Also perturb based on extinction uncertainties.
            stars['m_jwst_F115W'] += np.random.randn(len(stars)) * 0.050
            stars['m_jwst_F182M'] += np.random.randn(len(stars)) * 0.020
            stars['m_jwst_F212N'] += np.random.randn(len(stars)) * 0.010
            stars['m_jwst_F323N'] += np.random.randn(len(stars)) * 0.008
            # stars['m_jwst_F405N'] += np.random.randn(len(stars)) * 0.010

            clu_tmp.star_systems = stars
        
            clu.append(clu_tmp)


    ## CMD
    plt.close(10)
    plt.figure(10, figsize=(10, 5))
    plt.clf()
    plt.subplots_adjust(left=0.08, bottom=0.17, wspace=0.25, top=0.8, right=0.98)

    plt.subplot(1, 2, 1)
    for ss in range(len(iso)):
        label = 'Age = {0:5.3f} Gyr, [Fe/H] = {1:3.1f}, d = {2:3.1f} kpc'
        label = label.format(10**iso[ss].meta['LOGAGE'] / 1e9,
                             iso[ss].meta['METAL_IN'],
                             iso[ss].meta['DISTANCE'] / 1e3)
        plt.plot(iso[ss]['m_jwst_F212N'] - iso[ss]['m_jwst_F323N'], iso[ss]['m_jwst_F212N'],
                     color=colors[ss], label=label)

    plt.ylim(22, 9)
    plt.xlim(0.75, 1.3)
    plt.xlabel('F212N - F323N (mag)')
    plt.ylabel('F212N (mag)')

    confusion = {'F090W': 32, 
                 'F115W': 27.2,
                 'F212N': 21.5,
                 'F323N': 19.3}
                 # 'F212N': 22.5,
        
    # Modified the F212N one to be a bit closer in.
    
    plt.axhline(18, color='grey', linestyle='--')
    plt.text(1.15, 18-0.2, 'HST Imaging', fontsize=10)
    
    plt.axhline(19, color='grey', linestyle='--')
    plt.text(1.15, 19-0.2, 'AO Imaging', fontsize=10)
    
    plt.axhline(15, color='grey', linestyle='--')
    plt.text(1.11, 15-0.2, 'Narrow-Field\nAO Spectra', fontsize=10)

    plt.axhline(11.5, color='grey', linestyle='--')
    plt.text(1.11, 11.5-0.2, '50% Complete\nWide-Field Spectra', fontsize=10)
    
    
    # plt.axhline(21.5, color='black', linestyle='--')
    # plt.text(8.0, 21.5-0.2, 'JWST', fontsize=10, color='black')
                    
    lg = plt.legend(fontsize=12, bbox_to_anchor=(1.12, 1.14), markerscale=3, loc='center', ncol=2)
    # for lh in lg.legendHandles: 
    #     lh._legmarker.set_alpha(1)

    def label_mass(iso, col_color, col_mag,
                   mass_val, mass_label, label_dcol=-1.0, color='blue'):
        idx = np.argmin(np.abs(iso['mass'] - mass_val))
        # print(mass_val, idx, iso['mass'][idx])

        m_label = '{0:s} M$_\odot$'.format(mass_label)
        xy = (col_color[idx], col_mag[idx])
        xytext = (col_color[idx] + label_dcol, col_mag[idx])
        arrow_props = dict(facecolor=color, edgecolor=color,
                           shrink=0.05, width=1, headwidth=2)
        plt.annotate(m_label, xy=xy, xytext=xytext,
                     color=color,
                     arrowprops=arrow_props)

    # label_mass(yng, ycol, ymag, 10, '10')
    # label_mass(yng, ycol, ymag, 5, '5')
    # label_mass(yng, ycol, ymag, 2, '2')
    # label_mass(yng, ycol, ymag, 1, '1', label_dcol=0.4)
    # label_mass(yng, ycol, ymag, 0.5, '0.5', label_dcol=0.4)

    # label_mass(old, ocol, omag, 1, '1', color='red')
    # label_mass(old, ocol, omag, 2, '2', color='red')
    # label_mass(old, ocol, omag, 0.5, '0.5', label_dcol=0.4, color='red')
    
    # plt.legend(fontsize=12, loc='lower left')

    
    plt.subplot(1, 2, 2)
    for ss in range(len(clu)-1):
        label = 'Age = {0:5.3f} Gyr, [Fe/H] = {1:4.1f}, d = {2:3.1f} kpc'
        label = label.format(10**clu[ss].iso.points.meta['LOGAGE'] / 1e9,
                             clu[ss].iso.points.meta['METAL_IN'],
                             clu[ss].iso.points.meta['DISTANCE'] / 1e3)


        stars = clu[ss].star_systems

        # Pick out a random sample of "samp_size" for plotting.
        # Too many to plot all of them.
        if clu[ss].iso.points.meta['LOGAGE'] < 9.1:
            samp_size = int(2e4)
        else:
            samp_size = int(2e5)

        if clu[ss].iso.points.meta['METAL_IN'] < 0:
            samp_size = int(1e4)
        # print(samp_size)
        
        sdx = np.random.randint(0, high=len(stars), size=samp_size)
        # print(sdx)

        plt.plot(stars['m_jwst_F212N'][sdx] - stars['m_jwst_F323N'][sdx],
                     stars['m_jwst_F212N'][sdx], marker='.', linestyle='none',
                     label=label, alpha=0.1, markersize=3, color=colors[ss])
    
    plt.ylim(22, 9)
    plt.xlim(0.75, 1.3)
    plt.xlabel('F212N - F323N (mag)')
    plt.ylabel('F212N (mag)')
    
    plt.axhline(saturate['F212N'], color='black', linestyle='--')
    plt.text(0.76, saturate['F212N']-0.2, 'JWST Saturation Limit', fontsize=10)
    
    plt.fill_between([0.75, 1.5], [confusion['F212N'], confusion['F212N']],
                     [confusion['F212N']-1.5, confusion['F212N']-1.5],
                     color='black', alpha=0.2)
    plt.text(0.76, confusion['F212N']-0.2, 'JWST Confusion Limit', fontsize=10)
    
    # plt.axhline(saturate['F212N'], color='black', linestyle='--')
    # plt.text(6.7, saturate['F212N']-0.2, 'Saturation', fontsize=10)
    
    # plt.axhline(confusion['F212N'], color='black', linestyle='--')
    # plt.text(6.7, confusion['F212N']-0.2, 'Confusion', fontsize=10)
    
    plt.axhline(20, color='black', linestyle='--')
    plt.text(1.15, 20-0.2, 'JWST Imaging', fontsize=10, color='black')
    
    plt.axhline(16.5, color='black', linestyle='--')
    plt.text(1.15, 16.5-0.2, 'JWST Spectra', fontsize=10, color='black')

    # plt.axhline(18, color='black', linestyle='--')
    # plt.text(1.15, 18-0.2, 'JWST Spectra', fontsize=10, color='black')

    plt.savefig(out_dir + 'cycle2_cmd.png')

    return iso, clu

def plot_cycle4_fig1(iso=None, clu=None):
    iso_dir = work_dir + 'iso_gc_jwst/'
    out_dir = work_dir + 'plots/'

    import glob
    iso_files = ['iso_6.90_2.70_08000_p00.fits', 'iso_6.60_2.70_08000_p00.fits', 
                 'iso_9.90_2.70_08000_p00.fits', 'iso_9.90_2.70_08000_m10.fits',
                 'iso_9.00_2.70_08000_p00.fits', 'iso_10.00_2.70_08000_p00.fits']
    clu_files = [iso_f.replace('iso', 'clu').replace('.fits', '.pkl') for iso_f in iso_files]
    iso_files = [iso_dir + iso_f for iso_f in iso_files]
    clu_files = [iso_dir + 'with_diff_reddening_0.1/' + clu_f for clu_f in clu_files]
    # clu_files = [iso_dir + clu_f for clu_f in clu_files]

    colors = ['cyan', 'blue', 'red', 'salmon', 'darkorange', 'maroon']

    # Reverse order for plotting
    iso_files = iso_files[::-1]
    clu_files = clu_files[::-1]
    colors = colors[::-1]

    if iso == None and clu == None:
        iso = []
        clu = []

        for ff in range(len(iso_files)):
            print(iso_files[ff])
            iso_tmp = Table.read(iso_files[ff])
            clu_tmp = pickle.load(open(clu_files[ff], 'rb'))
            
            good = np.where(iso_tmp['phase'] < 7)[0]
            iso.append(iso_tmp[good])

            # Cut the stars out below F212N > 24 (gets rid of a bunch).
            stars = clu_tmp.star_systems
            bdx = np.where(stars['m_jwst_F212N'] <= 24)[0]
            stars = stars[bdx]

            # Add some noise to the filters we will observe in.
            # snr is calculated in jwst_gc_etc.ipynb and is calibrated relative to a K-band magnitude.
            snr_f115w = 979.0  * 10**(-(stars['m_nirc2_Kp'] - 14.00) / 5)
            snr_f182m = 1228.0 * 10**(-(stars['m_nirc2_Kp'] - 14.00) / 5)
            snr_f212n = 1066.0 * 10**(-(stars['m_nirc2_Kp'] - 14.00) / 5)        
            snr_f323n = 1102.0 * 10**(-(stars['m_nirc2_Kp'] - 14.00) / 5)
            snr_f405n = 1025.0 * 10**(-(stars['m_nirc2_Kp'] - 14.00) / 5)

            stars['m_jwst_F115W'] += np.random.randn(len(stars)) / snr_f115w
            stars['m_jwst_F182M'] += np.random.randn(len(stars)) / snr_f182m
            stars['m_jwst_F212N'] += np.random.randn(len(stars)) / snr_f323n
            stars['m_jwst_F323N'] += np.random.randn(len(stars)) / snr_f323n
            # stars['m_jwst_F405N'] += np.random.randn(len(stars)) / snr_f405n

            # Also perturb based on extinction uncertainties.
            stars['m_jwst_F115W'] += np.random.randn(len(stars)) * 0.050
            stars['m_jwst_F182M'] += np.random.randn(len(stars)) * 0.020
            stars['m_jwst_F212N'] += np.random.randn(len(stars)) * 0.010
            stars['m_jwst_F323N'] += np.random.randn(len(stars)) * 0.008
            # stars['m_jwst_F405N'] += np.random.randn(len(stars)) * 0.010

            clu_tmp.star_systems = stars
        
            clu.append(clu_tmp)


    ## CMD
    plt.close(10)
    fig = plt.figure(10, figsize=(12, 3.5))
    plt.clf()
    plt.subplots_adjust(left=0.07, bottom=0.2, wspace=0.27, top=0.90, right=0.75)

    plt.subplot(1, 2, 1)
    for ss in range(len(iso)):
        if 10**iso[ss].meta['LOGAGE'] >= 1e9:
            label = 'Age = {0:2.0f} Gyr, [Fe/H] = {1:3.1f}'  #, d = {2:3.1f} kpc'
            label = label.format(10**iso[ss].meta['LOGAGE'] / 1e9,
                                 iso[ss].meta['METAL_IN'])
        else:
            label = 'Age = {0:2.0f} Myr, [Fe/H] = {1:3.1f}'  #, d = {2:3.1f} kpc'
            label = label.format(10**iso[ss].meta['LOGAGE'] / 1e6,
                                 iso[ss].meta['METAL_IN'])
        plt.plot(iso[ss]['m_jwst_F212N'] - iso[ss]['m_jwst_F323N'], iso[ss]['m_jwst_F212N'],
                     color=colors[ss], label=label)

    plt.ylim(22, 9)
    plt.xlim(0.75, 1.3)
    plt.xlabel('F212N - F323N (mag)')
    plt.ylabel('F212N (mag)')
    plt.title('Spectral Limits')
    
    confusion = {'F090W': 32, 
                 'F115W': 27.2,
                 'F212N': 21.5,
                 'F323N': 19.3}
                 # 'F212N': 22.5,
        
    # Modified the F212N one to be a bit closer in.
    
    plt.axhline(15, color='grey', linestyle='--')
    plt.text(1.1, 15-0.2, 'Narrow-Field\nAO Spectra', fontsize=10)

    plt.axhline(11.5, color='grey', linestyle='--')
    plt.text(1.1, 11.5-0.2, 'Seeing-Limited\nWide-Field Spectra', fontsize=10)

    plt.axhline(16.5, color='black', linestyle='--', lw=2)
    plt.text(1.1, 16.5-0.2, 'JWST Spectra', fontsize=10, color='black')
    
    # plt.axhline(21.5, color='black', linestyle='--')
    # plt.text(8.0, 21.5-0.2, 'JWST', fontsize=10, color='black')
                    
    lg = plt.legend(fontsize=12, bbox_to_anchor=(0.88, 0.7),
                    bbox_transform=fig.transFigure,
                    markerscale=3, loc='center', ncol=1)
    # for lh in lg.legendHandles: 
    #     lh._legmarker.set_alpha(1)

    def label_mass(iso, col_color, col_mag,
                   mass_val, mass_label, label_dcol=-1.0, color='blue'):
        idx = np.argmin(np.abs(iso['mass'] - mass_val))
        # print(mass_val, idx, iso['mass'][idx])

        m_label = '{0:s} M$_\odot$'.format(mass_label)
        xy = (col_color[idx], col_mag[idx])
        xytext = (col_color[idx] + label_dcol, col_mag[idx])
        arrow_props = dict(facecolor=color, edgecolor=color,
                           shrink=0.05, width=1, headwidth=2)
        plt.annotate(m_label, xy=xy, xytext=xytext,
                     color=color,
                     arrowprops=arrow_props)

    # label_mass(yng, ycol, ymag, 10, '10')
    # label_mass(yng, ycol, ymag, 5, '5')
    # label_mass(yng, ycol, ymag, 2, '2')
    # label_mass(yng, ycol, ymag, 1, '1', label_dcol=0.4)
    # label_mass(yng, ycol, ymag, 0.5, '0.5', label_dcol=0.4)

    # label_mass(old, ocol, omag, 1, '1', color='red')
    # label_mass(old, ocol, omag, 2, '2', color='red')
    # label_mass(old, ocol, omag, 0.5, '0.5', label_dcol=0.4, color='red')
    
    # plt.legend(fontsize=12, loc='lower left')

    
    plt.subplot(1, 2, 2)
    for ss in range(len(clu)-1):
        label = 'Age = {0:5.3f} Gyr, [Fe/H] = {1:4.1f}, d = {2:3.1f} kpc'
        label = label.format(10**clu[ss].iso.points.meta['LOGAGE'] / 1e9,
                             clu[ss].iso.points.meta['METAL_IN'],
                             clu[ss].iso.points.meta['DISTANCE'] / 1e3)


        stars = clu[ss].star_systems

        # Pick out a random sample of "samp_size" for plotting.
        # Too many to plot all of them.
        if clu[ss].iso.points.meta['LOGAGE'] < 9.1:
            samp_size = int(2e4)
        else:
            samp_size = int(2e5)

        if clu[ss].iso.points.meta['METAL_IN'] < 0:
            samp_size = int(1e4)
        # print(samp_size)
        
        sdx = np.random.randint(0, high=len(stars), size=samp_size)
        # print(sdx)

        plt.plot(stars['m_jwst_F212N'][sdx] - stars['m_jwst_F323N'][sdx],
                     stars['m_jwst_F212N'][sdx], marker='.', linestyle='none',
                     label=label, alpha=0.1, markersize=3, color=colors[ss])
    
    plt.ylim(22, 9)
    plt.xlim(0.75, 1.3)
    plt.xlabel('F212N - F323N (mag)')
    plt.ylabel('F212N (mag)')
    plt.title('Imaging Limits')
    
    plt.axhline(saturate['F212N'], color='black', linestyle='--')
    plt.text(0.76, saturate['F212N']-0.2, 'JWST Saturation Limit', fontsize=10)
    
    plt.axhline(18, color='grey', linestyle='--')
    plt.text(1.15, 18-0.2, 'HST Imaging', fontsize=10)
    
    plt.axhline(19, color='grey', linestyle='--')
    plt.text(1.15, 19-0.2, 'AO Imaging', fontsize=10)
    
    plt.fill_between([0.75, 1.5], [confusion['F212N'], confusion['F212N']],
                     [confusion['F212N']-1.5, confusion['F212N']-1.5],
                     color='black', alpha=0.2)
    plt.text(0.76, confusion['F212N']-0.2, 'JWST Confusion Limit', fontsize=10)
    
    # plt.axhline(saturate['F212N'], color='black', linestyle='--')
    # plt.text(6.7, saturate['F212N']-0.2, 'Saturation', fontsize=10)
    
    # plt.axhline(confusion['F212N'], color='black', linestyle='--')
    # plt.text(6.7, confusion['F212N']-0.2, 'Confusion', fontsize=10)
    
    plt.axhline(20, color='black', linestyle='--', lw=2)
    plt.text(1.15, 20-0.2, 'JWST Imaging', fontsize=10, color='black', fontweight='bold')
    
    # plt.axhline(18, color='black', linestyle='--')
    # plt.text(1.15, 18-0.2, 'JWST Spectra', fontsize=10, color='black')

    plt.savefig(out_dir + 'cycle4_cmd.png')

    return iso, clu

def plot_image_spec_fov():
    gc_hst_img_root = '/Users/jlu/work/gc/hst/rgb/'
    out_dir = work_dir + 'plots/'

    fits_F153M = fits.open(gc_hst_img_root + 'hst_11671_03_wfc3_ir_f153m_drz.fits')
    fits_F139M = fits.open(gc_hst_img_root + 'hst_11671_06_wfc3_ir_f139m_drz.fits')
    fits_F127M = fits.open(gc_hst_img_root + 'hst_11671_06_wfc3_ir_f127m_drz.fits')

    img_F153M = fits_F153M[1].data
    img_F139M = fits_F139M[1].data
    img_F127M = fits_F127M[1].data

    wcs_F153M = WCS(fits_F153M[1].header)
    wcs_F139M = WCS(fits_F139M[1].header)
    wcs_F127M = WCS(fits_F127M[1].header)

    # Crop the F139M image
    img_F153M = img_F153M[0:img_F139M.shape[0], 0:img_F139M.shape[1]]
    print(img_F153M.shape, img_F139M.shape, img_F127M.shape)

    rimg = np.array(img_F153M, np.float_())
    gimg = np.array(img_F139M, np.float_())
    bimg = np.array(img_F127M, np.float_())

    stretch = SqrtStretch() + ZScaleInterval(krej=500, contrast=0.05)

    r = stretch(rimg)
    g = stretch(rimg)
    b = stretch(gimg)

    ### SAVING
    # https://docs.astropy.org/en/stable/api/astropy.visualization.make_lupton_rgb.html
    # astropy.visualization.make_lupton_rgb(image_r, image_g, image_b, minimum=0, stretch=5, Q=8, fil/ename=None)[source]
    # Return a Red/Green/Blue color image from up to 3 images using an asinh stretch.
    # The input images can be int or float, and in any range or bit-depth.

    # Get the value of lower and upper 0.5% of all pixels
    lo_val, up_val = np.percentile(np.hstack((r.flatten(), g.flatten(), b.flatten())), (0.5, 99.5))  
    
    stretch_val = up_val - lo_val

    rgb_default = make_lupton_rgb(r, g, b, minimum=lo_val, stretch=stretch_val, Q=0, filename="provafinale.png")

    # Cut the top rows - contains black pixels
    rgb_default = rgb_default[100:, :, :]

    print(r.flatten().mean(), r.min(), r.max())

    fig = plt.figure(figsize=(6,6))
    plt.subplot(projection=wcs_F153M, figure=fig)

    # plt.imshow(rgb_default, origin='lower')
    plt.imshow(r, origin='lower', vmin=0, vmax=1, cmap='afmhot')
    plt.axis('equal')

    # Plot the fields of view for NIRSpec
    sgra_ra = ( 17 + (45/60.) + (40.04/3600.) ) * 15.0
    sgra_dec = -29 - 28/3600.0
    fov_side = 3. / 3600.  # 3 arcsec in deg

    sgra_pix_x = 1025
    sgra_pix_y = 1032
    scale = -2.5e-5  # deg / pixel

    side_in_pix = 180
    xlim_lo = sgra_pix_x - (side_in_pix / 2.)
    xlim_hi = sgra_pix_x + (side_in_pix / 2.)
    ylim_lo = sgra_pix_y - (side_in_pix / 2.)
    ylim_hi = sgra_pix_y + (side_in_pix / 2.)
    plt.xlim(xlim_lo, xlim_hi)
    plt.ylim(ylim_lo, ylim_hi)
    
    leg_handles = []
    foo = plt.plot([sgra_pix_x + 6], [sgra_pix_y + 8], 'ko', mfc='none', mec='black', ms=10, label='Sgr A*')
    leg_handles.append(foo[0])
    
    from matplotlib.patches import Rectangle

    ax = plt.gca()
    # fov = Rectangle((sgra_ra, sgra_dec), fov_side, fov_side, angle=45, edgecolor='green', facecolor='none',
    #                   transform=ax.get_transform('fk5'))
    msc_step = 2.95 # arcsec
    msc_x = np.array([-msc_step, 0, msc_step])
    msc_y = np.array([-msc_step, 0, msc_step])
    
    angle = 0
    ang_rad = np.radians(angle)
    cosa = np.cos(ang_rad)
    sina = np.sin(ang_rad)

    img_size = 10.0 / (3600.0 * scale)
    xpix = sgra_pix_x - (img_size / 2.0)
    ypix = sgra_pix_y - (img_size / 2.0)
    img_fov = Rectangle((xpix, ypix),  img_size, img_size, angle=0,
                                edgecolor='cyan', facecolor='none', label='Keck FOV',
                            linewidth=2)

    
    ax.add_patch(img_fov)
    leg_handles.append(img_fov)

    
    for xx in range(len(msc_x)):
        for yy in range(len(msc_y)):
            fov_size = fov_side / scale
            xpix = sgra_pix_x - (fov_size / 2.0)
            ypix = sgra_pix_y - (fov_size / 2.0)
            
            dx =  msc_x[xx] * cosa + msc_y[yy] * sina
            dy = -msc_x[xx] * sina + msc_y[yy] * cosa
            xpix = xpix + (dx / (3600.0 * scale))
            ypix = ypix + (dy / (3600.0 * scale))
            
            fov = Rectangle((xpix, ypix), fov_side / scale, fov_side / scale, angle=angle,
                                edgecolor='green', facecolor='none', label='NIRSpec FOV',
                                linewidth=2)

            if xx == 0 and yy == 0:
                leg_handles.append(fov)
                
            ax.add_patch(fov)
            
    plt.legend(handles=leg_handles, ncol=2)
    ax.coords[0].set_axislabel('R.A.')
    ax.coords[1].set_axislabel('Dec.', minpad=-0.7)
    ax.coords[0].set_ticks(spacing=5. * u.arcsec)

    plt.savefig(out_dir + 'fov_nirspec_keck.png')
    
    return


def plot_spectra_yng_old(mag_f212n=18, snr=False, save_fits=False):
    """
    Plot the spectra for stars of the specified magnitude
    for three different isochrones:

    yng_file = work_dir + 'iso_gc_jwst/iso_6.61_2.70_08000_p00.fits'
    old_file = work_dir + 'iso_gc_jwst/iso_9.90_2.70_08000_p00.fits'
    old_lm_file = work_dir + 'iso_gc_jwst/iso_9.90_2.70_08000_m10.fits'

    If snr = False, then the spectra will be noise free. 
    If snr = ##, then artificial noise will be added. 

    Note that the spectra are plotted with the continuum divided out
    where the continuum is a 2nd order polynomial fit of the young star
    that is then applied to all three so we preserve the slight 
    changes in continuum slopes. 
    """
    yng_file = work_dir + 'iso_gc_jwst/iso_6.61_2.70_08000_p00.fits'
    old_file = work_dir + 'iso_gc_jwst/iso_9.90_2.70_08000_p00.fits'
    old_lm_file = work_dir + 'iso_gc_jwst/iso_9.90_2.70_08000_m10.fits'
    out_dir = work_dir + 'plots/'

    # Isochrones
    yng_all = Table.read(yng_file)
    old_all = Table.read(old_file)
    lom_all = Table.read(old_lm_file)

    # Read in the wavelength array for NIRSpec G235H
    jwst_wave = np.loadtxt('jwst_nirspec_g235h_wave.txt')

    # Filter out compact object phases
    ygood = np.where(yng_all['phase'] < 7)[0]
    ogood = np.where(old_all['phase'] < 7)[0]
    mgood = np.where(lom_all['phase'] < 7)[0]
    yng = yng_all[ygood]
    old = old_all[ogood]
    lom = lom_all[ogood]
    
    ## select a star with T&g from this isochrone
    ydx = np.argmin(np.abs(yng['m_jwst_F212N'] - mag_f212n))
    odx = np.argmin(np.abs(old['m_jwst_F212N'] - mag_f212n))
    mdx = np.argmin(np.abs(lom['m_jwst_F212N'] - mag_f212n))
    
    # stars = [yng[ydx], old[odx], lom[mdx]]
    stars = [yng[ydx], old[odx]]
    # stars = [old[odx]]
    stars_tab = table.vstack(stars) # destroys meta data. 
    red_law = reddening.RedLawHosek18b()
    AKs = 2.70
    
    ## make the plot
    plt.figure(figsize=(12, 4))
    plt.clf()
    plt.subplots_adjust(left=0.1, right=0.97, top=0.85, bottom=0.25)

    color = ['mediumblue', 'maroon', 'darkorange']
    names = ['yng', 'old_p00', 'old_m10']

    for ss in range(len(stars)):
        T = stars[ss]['Teff'][0]
        g = stars[ss]['logg'][0]
        R = stars[ss]['R'].to(u.pc).value[0]
        feh = stars[ss].meta['METAL_IN']
        mass = stars[ss]['mass'][0]
        distance = 8000 # pc
        
        ## get the spectrum
        star = atm.get_phoenixv16_atmosphere(temperature=T, gravity=g, metallicity=feh,
                                             rebin=False)
        
        # Convert into flux observed at Earth (unreddened)
        star *= (R / distance)**2  # in erg s^-1 cm^-2 A^-1

        # Redden the spectrum. This doesn't take much time at all.
        red = red_law.reddening(AKs).resample(star.wave) 
        star *= red

        # filt = syn.get_filter_info('jwst,F170LP')
        # star_in_filter = obs.Observation(star, filt, binset=filt.wave, force='taper')
        # import pdb
        # pdb.pm()
    
        # Convolve to desired spectral resolution
        star = smear(star, R=2700)

        # Trim down to F170LP (no filter transmission function though)
        star = spectrum.trimSpectrum(star, 16600, 31100)

        # Resample the spectra to be Nyquist sampled.
        star = star.resample(jwst_wave * 1e4)

        wave = star.wave
        flux = star.flux
        
        ## calculate the continuum to remove
        #  remove the continuum of only the blue star so
        # we preseve the slope differences.
        group_of_idxs = list(np.arange(len(wave)))   
        num_to_select = 10000
        rand_idx = random.choice(group_of_idxs, num_to_select)
        np.sort(rand_idx)

        if ss == 0:
            z = np.polyfit(wave[rand_idx], np.log10(flux[rand_idx]), 3)
            p = np.poly1d(z)

        # Add noise. (simply assume that f_err / f = 1 / snr
        # Must do this after we estimate the continuum; but before
        # we correct it.
        flux_rc = flux
        
        if snr != False:
            flux_rc_err = flux_rc / snr
            flux_rce = flux_rc + np.random.randn(len(flux)) * flux_rc_err

        # now remove the continuum
        flux_rc /= 10**(p(wave))
        if snr != False:
            flux_rce /= 10**(p(wave))
            flux_rc_err /= 10**(p(wave))

        norm_idx = np.where((wave > 21900) & (wave < 22000))[0]
        norm_flux = flux_rc[norm_idx].sum()
        if ss == 0:
            norm_flux0 = norm_flux
            
        flux_rc *= norm_flux0 / norm_flux
        if snr != False:
            flux_rce *= norm_flux0 / norm_flux
            flux_rc_err *= norm_flux0 / norm_flux

        # prep the legend label.
        label = 'Age = {0:5.3f} Gyr, M = {1:4.2f} M$_\odot$, T$_{{eff}}$ = {2:5.0f} K, log(g) = {3:3.1f}, [Fe/H] = {4:3.1f}'
        label = label.format(10**stars[ss].meta['LOGAGE'] / 1e9,
                             mass, T, g, stars[ss].meta['METAL_IN'])
        
        # Add noise. (simply assume that f_err / f = 1 / snr
        if snr != False:
            plt.plot(wave / 1e4, flux_rce, '-', alpha=0.2, color=color[ss])
            # markers, caps, bars = plt.errorbar(wave / 1e4, flux_rc, yerr=flux_rc_err,
            #                                        fmt='-',capsize=2, capthick=2, alpha=0.1, color=color[ss])
            # # loop through bars and caps and set the alpha value
            # [bar.set_alpha(0.1) for bar in bars]
            # [cap.set_alpha(0.1) for cap in caps]

        plt.plot(wave / 1e4, flux_rc, label=label, color=color[ss])

        if save_fits:
            if ss == 0:
                stars_tab = Table([wave / 1e4, flux_rc], names=['wave', names[ss]])
            else:
                stars_tab[names[ss]] = flux_rc
            
            if snr:
                stars_tab[names[ss] + '_noise'] = flux_rce
                stars_tab[names[ss] + '_err'] = flux_rc_err
        
    plt.legend(fontsize=12)
    plt.xlabel('Wavelength ($\mu$m)')
    # plt.ylabel('Flux (erg/s/cm$^2$/Ang)')
    plt.ylabel('Normalized Flux')
    plt.ylim(0.2, 1.2)
    #plt.xlim(2.1, 2.35)

    title = 'F212N = {0:.1f}'.format(mag_f212n)
    if snr != False:
        title += ', SNR = {0:.1f}'.format(snr)
        
    plt.title(title)

    # Save to output file.
    outfile = out_dir + 'spectra_yng_old_lowZ_F212N_{0:04.1f}'.format(mag_f212n)
    if snr != False:
        outfile += '_snr{0:03.0f}'.format(snr)
    outfile += '.png'
        
    plt.savefig(outfile)

    if save_fits:
        stars_tab.write(outfile.replace('.png', '_data.fits'), overwrite=True)

    return


def smear(sp, R, w_sample=1):
    '''
    Smears a model spectrum with a gaussian kernel to the given resolution, R.

    Parameters
    -----------

    sp: SourceSpectrum
        Pysynphot object that we willsmear

    R: int
        The resolution (dL/L) to smear to

    w_sample: int
        Oversampling factor for smoothing

    Returns
    -----------

    sp: PySynphot Source Spectrum
        The smeared spectrum
    '''

    # Save original wavelength grid and units
    w_grid = sp.wave
    w_units = sp.waveunits
    f_units = sp.fluxunits
    sp_name = sp.name

    # Generate logarithmic wavelength grid for smoothing
    w_logmin = np.log10(np.nanmin(w_grid))
    w_logmax = np.log10(np.nanmax(w_grid))
    n_w = np.size(w_grid)*w_sample
    w_log = np.logspace(w_logmin, w_logmax, num=n_w)

    # Find stddev of Gaussian kernel for smoothing
    R_grid = (w_log[1:-1]+w_log[0:-2])/(w_log[1:-1]-w_log[0:-2])/2
    sigma = np.median(R_grid)/R
    if sigma < 1:
        sigma = 1

    # Interpolate on logarithmic grid
    f_log = np.interp(w_log, w_grid, sp.flux)

    # Smooth convolving with Gaussian kernel
    gauss = Gaussian1DKernel(stddev=sigma)
    f_conv = convolve_fft(f_log, gauss)

    # Interpolate back on original wavelength grid
    f_sm = np.interp(w_grid, w_log, f_conv)

    # Write smoothed spectrum back into Spectrum object
    return S.ArraySpectrum(w_grid, f_sm, waveunits=w_units,
                            fluxunits=f_units, name=sp_name)


def etc_nircam_calc_snr_vs_mag(filt, ngroup=2, nint=2, nexp=5, readout='medium8',
                            mag_kp=np.arange(9, 24, 1.0),
                            recalc=False):
    """
    Run the ETC for the F090W filter over a range of magnitudes.

    The results will be saved in an output file. If the file
    already exists, it will be loaded (unless recalc=True).
    """
    # Note that this is an equivalent Kp magnitude for an early type star that 
    # is reddened at the Galactic Center extinction (and a steep law).
    n_calcs = len(mag_kp)

    # Configuration object
    with open(work_dir + 'etc_files/nircam/input.json', 'r') as inf:
        ci = json.loads(inf.read())

    # Setup the integration times.
    ci['configuration']['detector']['readout_pattern'] = readout
    ci['configuration']['detector']['ngroup'] = ngroup
    ci['configuration']['detector']['nint'] = nint
    ci['configuration']['detector']['nexp'] = nexp
    ci['configuration']['instrument']['filter'] = filt

    if filt in ['f250m', 'f335m', 'f360m', 'f430m', 'f323n', 'f405n', 'f466n', 'f470n']:
        ci['configuration']['instrument']['aperture'] = 'lw'
        ci['configuration']['instrument']['mode'] = 'lw_imaging'

    # Change our scene to just have a single star. 
    ci['scene'] = [ci['scene'][0]]
    ci['strategy']['target_xy'] = [0.0, 0.0]
    ci['scene'][0]['position']['x_offset'] = 0.0
    ci['scene'][0]['position']['y_offset'] = 0.0

    pkl_file = work_dir + 'etc_local/nircam/mag_test_ne{0:02d}_ni{1:02d}_ng{2:02d}_{3:s}_{4:s}.pkl'
    pkl_file = pkl_file.format(nexp, nint, ngroup, filt, readout)
                                   
    if recalc or not os.path.exists(pkl_file):
        _pkl = open(pkl_file, 'wb')

        t_start = time.time()
        t_ss = t_start
        for ss in range(n_calcs):
            ci['scene'][0]['spectrum']['normalization']['norm_flux'] = mag_kp[ss]

            rep = perform_calculation(ci)

            pickle.dump(rep, _pkl)

            t_tmp = time.time()
            print('m = {0:.1f} cumulative exec time = {1:.0f} sec'.format(mag_kp[ss], t_tmp - t_ss))
            t_ss = t_tmp

        t_stop = time.time()
        print('Execution time = {0:.0f} sec'.format(t_stop - t_start))

        _pkl.close()


    # The pickle file exists now. Load up our metrics.
    _pkl = open(pkl_file, 'rb')

    snr = np.zeros(n_calcs, dtype=float)
    f_sat = np.zeros(n_calcs, dtype=float)
    sat_ng = np.zeros(n_calcs, dtype=float)
    flux_ex = np.zeros(n_calcs, dtype=float)
    t_exp = np.zeros(n_calcs, dtype=float)

    for ss in range(n_calcs):
        rep = pickle.load(_pkl)
        
        snr[ss] = rep['scalar']['sn']
        f_sat[ss] = rep['scalar']['fraction_saturation']
        sat_ng[ss] = rep['scalar']['sat_ngroups']
        flux_ex[ss] = rep['scalar']['extracted_flux']
        t_exp[ss] = rep['scalar']['total_exposure_time']

    metrics = Table([mag_kp, snr, f_sat, sat_ng, flux_ex, t_exp],
                    names=['mag_kp', 'snr', 'f_sat', 'sat_ngroups', 'flux_ex', 'total_exp_time'])

    return metrics, rep['input']

def etc_nircam_plot_snr_vs_mag(metrics, input_json):
    mag = metrics['mag_kp']
    snr = metrics['snr']
    f_sat = metrics['f_sat']
    sat_ng = metrics['sat_ngroups']
    flux_ex = metrics['flux_ex']
    t_exp = metrics['total_exp_time']

    ci = input_json

    filt = ci['configuration']['instrument']['filter']

    plt.figure(figsize=(8, 5))
    plt.subplots_adjust(right=0.7)
    plt.semilogy(mag, snr, 'go', label='Good')
    plt.xlabel('K for Young Star (mag)')
    plt.ylabel('SNR in ' + filt)

    low = np.where(snr < 10)[0]
    plt.plot(mag[low], snr[low], 'ko', label='SNR<10')

    sat = np.where(f_sat > 1)[0]
    # plt.plot(mag[sat], snr[sat], 'ro', label='Saturated')
    plt.plot(mag[sat], np.repeat(snr[-1], len(sat)), 'ro', label='Saturated')

    plt.legend()

    # Label S0-2
    s02 = np.argmin(np.abs(mag - 14))
    plt.annotate('S0-2', 
                xy=(mag[s02], snr[s02]*1.1), 
                xytext=(mag[s02], snr[s02]*1.5),
                arrowprops=dict(facecolor='black', shrink=0.05))

    fig = plt.gcf()
    x_msg = 0.72
    y_msg = 0.85
    for key, val in ci['configuration']['detector'].items():
        msg = '{0:s} = {1}'.format(key, val)
        plt.text(x_msg, y_msg, msg, ha='left', va='top', 
                 fontsize=14,
                 transform=fig.transFigure)
        y_msg -= 0.05

    msg = 't_exp = {0:.1f} sec'.format(t_exp[0])
    plt.text(x_msg, y_msg, msg, ha='left', va='top', 
                 fontsize=14,
                 transform=fig.transFigure)

    # Make an analytic estimate of the scaling relation.
    mag_eq = np.arange(10, 20, 0.1)
    snr0 = snr[s02]
    mag0 = mag[s02]
    snr_eq =  snr0 * 10**(-(mag_eq - mag0) / 5.0)

    plt.plot(mag_eq, snr_eq, 'k--')
    print('SNR = {0:.0f} * 10**(-(mag - {1:.2f}) / 5)'.format(snr0, mag0))
    
    return


def etc_nirspec_calc_snr_vs_mag(ngroup=2, nint=2, nexp=5,
                                mag_kp=np.arange(9, 19, 1.0),
                                recalc=False):
    """
    Run the ETC for the F090W filter over a range of magnitudes.

    The results will be saved in an output file. If the file
    already exists, it will be loaded (unless recalc=True).
    """
    # Note that this is an equivalent Kp magnitude for an early type star that 
    # is reddened at the Galactic Center extinction (and a steep law).
    n_calcs = len(mag_kp)

    # Configuration object
    with open(work_dir + 'etc_files/nirspec/input.json', 'r') as inf:
        cs = json.loads(inf.read())

    inst = cs['configuration']['instrument']
    inst = {}
    inst['instrument'] = 'nirspec'
    inst['aperture'] = "ifu"
    inst['disperser'] = "g235h"
    inst['filter'] = "f170lp"
    inst['instrument'] = "nirspec"
    inst['mode'] = "ifu"
        
    det = cs['configuration']['detector']
    det['nexp'] = nexp
    det['ngroup'] = ngroup
    det['nint'] = nint
    det['readout_pattern'] = "nrsrapid"
    det['subarray'] = "full"

    # Change our scene to just have a single star. 
    cs['scene'] = [cs['scene'][0]]
    cs['strategy']['target_xy'] = [0.0, 0.0]
    cs['scene'][0]['position']['x_offset'] = 0.0
    cs['scene'][0]['position']['y_offset'] = 0.0
        
    pkl_file = work_dir + 'etc_local/nirspec/mag_test_ne{0:02d}_ni{1:02d}_ng{2:02d}.pkl'
    pkl_file = pkl_file.format(nexp, nint, ngroup)
                                   
    if recalc or not os.path.exists(pkl_file):
        _pkl = open(pkl_file, 'wb')
        pickle.dump(mag_kp, _pkl)

        t_start = time.time()
        t_ss = t_start
        for ss in range(n_calcs):
            cs['scene'][0]['spectrum']['normalization']['norm_flux'] = mag_kp[ss]

            rep = perform_calculation(cs)

            pickle.dump(rep, _pkl)

            t_tmp = time.time()
            print('m = {0:.1f} exec time = {1:.0f} sec'.format(mag_kp[ss], t_tmp - t_ss))
            t_ss = t_tmp

        t_stop = time.time()
        print('Total exec time = {0:.0f} sec'.format(t_stop - t_start))

        _pkl.close()


    # The pickle file exists now. Load up our metrics.
    _pkl = open(pkl_file, 'rb')
    mag_kp = pickle.load(_pkl)

    n_calcs = len(mag_kp)

    snr = np.zeros(n_calcs, dtype=float)
    f_sat = np.zeros(n_calcs, dtype=float)
    sat_ng = np.zeros(n_calcs, dtype=float)
    flux_ex = np.zeros(n_calcs, dtype=float)
    t_exp = np.zeros(n_calcs, dtype=float)

    
    for ss in range(n_calcs):
        rep = pickle.load(_pkl)
        
        snr[ss] = rep['scalar']['sn']
        f_sat[ss] = rep['scalar']['fraction_saturation']
        sat_ng[ss] = rep['scalar']['sat_ngroups']
        flux_ex[ss] = rep['scalar']['extracted_flux']
        t_exp[ss] = rep['scalar']['total_exposure_time']

    metrics = Table([mag_kp, snr, f_sat, sat_ng, flux_ex, t_exp],
                    names=['mag_kp', 'snr', 'f_sat', 'sat_ngroups', 'flux_ex', 'total_exp_time'])

    return metrics, rep['input']

def etc_nirspec_plot_snr_vs_mag(metrics, input_json):
    mag = metrics['mag_kp']
    snr = metrics['snr']
    f_sat = metrics['f_sat']
    sat_ng = metrics['sat_ngroups']
    flux_ex = metrics['flux_ex']
    t_exp = metrics['total_exp_time']

    cs = input_json

    plt.figure(figsize=(8, 5))
    plt.subplots_adjust(right=0.7)
    plt.plot(mag, snr, 'go', label='Good')
    plt.xlabel('K for Young Star (mag)')
    plt.ylabel('SNR at 2 microns at R=2700')

    low = np.where(snr < 10)[0]
    plt.plot(mag[low], snr[low], 'ko', label='SNR<10')

    sat = np.where(f_sat > 1)[0]
    plt.plot(mag[sat], snr[sat], 'ro', label='Saturated')

    plt.legend()

    # Label S0-2
    s02 = np.argmin(np.abs(mag - 14))
    plt.annotate('S0-2', 
                     xy=(mag[s02], snr[s02]*1.1), 
                     xytext=(mag[s02], snr[s02]*2),
                arrowprops=dict(facecolor='black', shrink=0.05))

    fig = plt.gcf()
    x_msg = 0.72
    y_msg = 0.85
    for key, val in cs['configuration']['detector'].items():
        msg = '{0:s} = {1}'.format(key, val)
        plt.text(x_msg, y_msg, msg, ha='left', va='top', 
                     fontsize=14,
                     transform=fig.transFigure)
        y_msg -= 0.05

    msg = 't_exp = {0:.1f} sec'.format(t_exp[0])
    plt.text(x_msg, y_msg, msg, ha='left', va='top', 
                 fontsize=14,
                 transform=fig.transFigure)

    # Make an analytic estimate of the scaling relation.
    mag_eq = np.arange(10, 20, 0.1)
    snr0 = snr[s02]
    mag0 = mag[s02]
    snr_eq =  snr0 * 10**(-(mag_eq - mag0) / 5.0)

    plt.plot(mag_eq, snr_eq, 'k--')

    print('SNR = {0:.0f} * 10**(-(mag - {1:.2f}) / 5)'.format(snr0, mag0))

    return


def etc_nirspec_snr_vs_ngroup(nframes=100, nexp=3,
                              mag_kp=np.array([12, 14, 18]),
                                recalc=False):
    """
    nframes = ngroup x nint

    We will explore all possible combinations and calculate SNR curves 
    for all of them. 

    Note this is a time-consuming process as this is 
    N_mags x N_factors

    where 
      N_mags is the length of the magnitude array
      N_factors is the number of integer factors in Nframes

    The outputs will be stored in pickle files, one for each
    magnitude with runs of different factors in each.
    """
    # Note that this is an equivalent Kp magnitude for an early type star that 
    # is reddened at the Galactic Center extinction (and a steep law).
    n_calcs = len(mag_kp)

    # Configuration object
    with open(work_dir + 'etc_files/nirspec/input.json', 'r') as inf:
        cs = json.loads(inf.read())

    inst = cs['configuration']['instrument']
    inst = {}
    inst['instrument'] = 'nirspec'
    inst['aperture'] = "ifu"
    inst['disperser'] = "g235h"
    inst['filter'] = "f170lp"
    inst['instrument'] = "nirspec"
    inst['mode'] = "ifu"

    det = cs['configuration']['detector']
    det['nexp'] = nexp
    det['readout_pattern'] = "nrsrapid"
    det['subarray'] = "full"

    # Keep the total number of frames at 100. 
    # Note (NRSrapid = 1 frame / group)
    # In other words, we can test
    #    100 integrations (nint) of 1 frames (ngroup) each
    #     50 integrations (nint) of 2 frames (ngroup) each
    # Better to have large ngroup up to the point of saturation.
    ngroup = np.array(factors(nframes))

    # Sort and pull out unique ones.
    ngroup = np.unique(ngroup)
    nint = ngroup[::-1]

    # Change our scene to just have a single star. 
    cs['scene'] = [cs['scene'][0]]
    cs['strategy']['target_xy'] = [0.0, 0.0]
    cs['scene'][0]['position']['x_offset'] = 0.0
    cs['scene'][0]['position']['y_offset'] = 0.0

    pkl_file_fmt = work_dir + 'etc_local/nirspec/ngroup_test_m{0:02d}_ne{1:02d}_ngni{2:03d}.pkl'

    t_start = time.time()
    t_ss = t_start

    for mm in range(len(mag_kp)):
        # open the pickle file and save our ngroups array.
        m_file = pkl_file_fmt.format(mag_kp[mm], nexp, nframes)
        
        if recalc or not os.path.exists(m_file):
            _out = open(m_file, 'wb')
            pickle.dump(ngroup, _out)
            pickle.dump(nint, _out)

            # Set the magnitude
            cs['scene'][0]['spectrum']['normalization']['norm_flux'] = mag_kp[mm]

            # loop through the number of groups for this mag.
            for gg in range(len(ngroup)):
                cs['configuration']['detector']['ngroup'] = ngroup[gg]
                cs['configuration']['detector']['nint'] = nint[gg]
        
                rep = perform_calculation(cs)

                pickle.dump(rep, _out)
        
                t_tmp = time.time()
                msg = 'm = {0:d}, ng = {1:d} exec time = {2:.0f} sec'
                print(msg.format(mag_kp[mm], ngroup[gg], t_tmp - t_ss))
                t_ss = t_tmp
        
            _out.close()
        
        t_stop = time.time()
        print('Total exec time = {0:.0f} sec'.format(t_stop - t_start))


    ng_snr = np.zeros((len(mag_kp), len(ngroup)), dtype=float)
    ng_f_sat = np.zeros((len(mag_kp), len(ngroup)), dtype=float)
    ng_sat_ng = np.zeros((len(mag_kp), len(ngroup)), dtype=int)
    ng_flux_ex = np.zeros((len(mag_kp), len(ngroup)), dtype=float)
    ng_texp = np.zeros((len(mag_kp), len(ngroup)), dtype=float)


    for mm in range(len(mag_kp)):
        # open the pickle file and save our ngroups array.
        m_file = pkl_file_fmt.format(mag_kp[mm], nexp, nframes)

        _out = open(m_file, 'rb')
        ngroup = pickle.load(_out)
        nint = pickle.load(_out)

        # loop through the number of groups for this mag_kp.
        for gg in range(len(ngroup)):
            rep = pickle.load(_out)

            ng_snr[mm][gg] = rep['scalar']['sn']
            ng_f_sat[mm][gg] = rep['scalar']['fraction_saturation']
            ng_sat_ng[mm][gg] = rep['scalar']['sat_ngroups']
            ng_flux_ex[mm][gg] = rep['scalar']['extracted_flux']
            ng_texp[mm][gg] = rep['scalar']['total_exposure_time']

            msg = 'mag_kp = {0:2d} ngroup = {1:2d} nint = {2:2d} texp = {3:4.0f} s'
            print(msg.format(mag_kp[mm], ngroup[gg], nint[gg], ng_texp[mm][gg]))

        _out.close()

    metrics = Table([mag_kp, ng_snr, ng_f_sat, ng_sat_ng, ng_flux_ex, ng_texp],
                    names=['mag_kp', 'snr', 'f_sat', 'sat_ngroups', 'flux_ex', 'total_exp_time'])
    metrics.meta['ngroup'] = ngroup
    metrics.meta['nint'] = nint
    metrics.meta['nframes'] = nframes
    
    return metrics, rep['input']
    
def etc_nirspec_plot_snr_vs_ngroup(metrics, input_json):
    mag = metrics['mag_kp']
    nframes = metrics.meta['nframes']
    ngroup = metrics.meta['ngroup']
    nint = metrics.meta['nint']
    ng_snr = metrics['snr']
    ng_f_sat = metrics['f_sat']
    ng_texp = metrics['total_exp_time']

    cs = input_json

    sym = ['o', 's', '>']

    plt.figure(figsize=(8, 5))
    plt.subplots_adjust(right=0.7)
    
    for mm in range(len(mag)):
        plt.semilogy(ngroup, ng_snr[mm], 'g' + sym[mm], label='Good m={0:d}'.format(mag[mm]))
    
        low = np.where(ng_snr[mm] < 10)[0]
        plt.plot(ngroup[low], ng_snr[mm][low], 'k' + sym[mm], label='SNR<10 m={0:d}'.format(mag[mm]))

        sat = np.where(ng_f_sat[mm] > 1)[0]
        plt.plot(ngroup[sat], ng_snr[mm][sat], 'r' + sym[mm], label='Saturated m={0:d}'.format(mag[mm]))


    plt.xlabel('N groups')
    plt.ylabel('SNR at 2 microns at R=2700')
    plt.title('Fixed nint x ngroup = {0:d}'.format(nframes))
    plt.legend(fontsize=10, ncol=2, bbox_to_anchor=(1.05, 0.5))

    fig = plt.gcf()
    x_msg = 0.72
    y_msg = 0.85
    for key, val in cs['configuration']['detector'].items():
        if key == 'ngroup':
            continue
        msg = '{0:s} = {1}'.format(key, val)
        plt.text(x_msg, y_msg, msg, ha='left', va='top', 
                     fontsize=14,
                     transform=fig.transFigure)
        y_msg -= 0.05

    print(ng_texp[0])
    print(ngroup)
    print(nint)

    return

def plot_prop_etc_snr_all_filt():
    pkl_file_fmt = work_dir + 'etc_local/nircam/mag_test_ne{0:02d}_ni{1:02d}_ng{2:02d}_{3:s}_{4:s}.pkl'

    filts = ['f115w', 'f212n', 'f323n', 'f405n']
    ngroups = [2, 2, 2, 2]
    nints = [2, 1, 2, 1]
    nexps = [12, 12, 12, 12]
    reads = ['shallow4', 'shallow4', 'shallow4', 'shallow4']

    met_all = []

    mag_kp = np.arange(9, 24, 1.0)
    n_calcs = len(mag_kp)
    
    for ff in range(len(filts)):
        pkl_file = pkl_file_fmt.format(nexps[ff], nints[ff], ngroups[ff], filts[ff], reads[ff])
        print(pkl_file)
        
        # The pickle file exists now. Load up our metrics.
        _pkl = open(pkl_file, 'rb')

        snr = np.zeros(n_calcs, dtype=float)
        f_sat = np.zeros(n_calcs, dtype=float)
        sat_ng = np.zeros(n_calcs, dtype=float)
        flux_ex = np.zeros(n_calcs, dtype=float)
        t_exp = np.zeros(n_calcs, dtype=float)

        for ss in range(n_calcs):
            rep = pickle.load(_pkl)
        
            snr[ss] = rep['scalar']['sn']
            f_sat[ss] = rep['scalar']['fraction_saturation']
            sat_ng[ss] = rep['scalar']['sat_ngroups']
            flux_ex[ss] = rep['scalar']['extracted_flux']
            t_exp[ss] = rep['scalar']['total_exposure_time']

        metrics = Table([mag_kp, snr, f_sat, sat_ng, flux_ex, t_exp],
                        names=['mag_kp', 'snr', 'f_sat', 'sat_ngroups', 'flux_ex', 'total_exp_time'])
        
        met_all.append(metrics)

        _pkl.close()
    

    colors = ['blue', 'green', 'orange', 'red']
    
    plt.figure(figsize=(8, 5))
    
    for ff in range(len(filts)):
        plt.semilogy(met_all[ff]['mag_kp'], met_all[ff]['snr'], 'k-',
                         color=colors[ff], label=filts[ff].upper())

    plt.xlabel('K (mag)')
    plt.ylabel('SNR')
    plt.xlim(12, 23)

    plt.legend()

    plt.savefig(work_dir + 'plots/snr_vs_mag_all_filts.png')

    return


def plot_klf_yng():
    """
    Plot the KLF for the young population and show current and future spectroscopic limits.
    Also show the mass ranges.
    """
    # Define isochrone parameters
    logAge = 6.60 # Age in log(years)
    AKs = 2.7 # Ks filter extinction in mags
    dist = 8000 # distance in parsecs
    metallicity = 0 # metallicity in [M/H]

    atm_func = atm.get_merged_atmosphere
    iso_dir = 'iso_gc_jwst/'

    # Define evolution models
    evo_mist = evolution.MISTv1(version=1.2) # Version 1.2 is the default
    redlaw = reddening.RedLawHosek18b()

    filt_list = ['nirc2,J', 'nirc2,Kp', 
         'jwst,F090W', 'jwst,F115W', 
         'jwst,F140M', 'jwst,F162M',
         'jwst,F164N', 'jwst,F212N',
         'jwst,F323N', 'jwst,F470N',
         'jwst,F182M', 'jwst,F360M']

    iso6 = syn.IsochronePhot(logAge, AKs, dist, metallicity=metallicity,
                             evo_model=evo_mist, atm_func=atm_func,
                             filters=filt_list, red_law=redlaw,
                             iso_dir=iso_dir, mass_sampling=3)        

    # Define a Kroupa IMF
    massLimits = np.array([0.1, 0.5, 120])
    powers = np.array([-1.3, -1.7])
    my_imf = imf.IMF_broken_powerlaw(massLimits, powers)

    total_mass = 1.6e4
    my_ifmr = ifmr.IFMR()

    cluster = syn.ResolvedCluster(iso6, my_imf, total_mass)

    stars = cluster.star_systems

    # Drop the compact objects
    idx = np.where(stars['phase'] < 10)
    stars = stars[idx]

    # Assign arbitrary magnitudes for WR stars. Uniformly distributed from Kp=9-11
    wdx = np.where(stars['isWR'] == True)[0]
    stars['m_nirc2_Kp'][wdx] = 9.0 + (np.random.rand(len(wdx))*2)
    print('N_WR = ', len(wdx))
    
    # Plot the mass luminosity relationship
    plt.close('all')
    plt.figure(1, figsize=(6, 6))
    plt.subplots_adjust(hspace=0.05)

    kbins = np.arange(9.0, 24, 0.5)
    
    ##########
    # Plot the KLF
    ##########
    ax1 = plt.subplot(2, 1, 1)
    klf, klfb, klfp = plt.hist(stars['m_nirc2_Kp'], bins=kbins,
                               histtype='step', linewidth=2, label='Multiples')
    plt.ylabel('N$_{{stars}}$')
    plt.title('Young Nuclear Cluster')
    plt.xlim(9, 18)
    plt.ylim(1, 250)
    rng = plt.axis()

    plt.setp(ax1.get_xticklabels(), visible=False)
    plt.xticks([])

    # Get the mass at the ground-based spectra/RV limit. This is for
    # wide-coverage IFU spectra, including seeing-limited and AO. 
    grnd_lim = 11.5 
    kdx_all = np.where(stars['m_nirc2_Kp'] < grnd_lim)[0]
    kdx_min = np.argmin(stars['mass'][kdx_all])
    kdx = kdx_all[kdx_min]
    keck_mass_label = '{0:.0f} M$_\odot$'.format(stars['mass'][kdx])
    
    plt.axvline(grnd_lim, color='grey', linestyle='--', linewidth=2)
    plt.text(grnd_lim - 0.7, rng[3] - 30, keck_mass_label,
                 fontsize=14,
                 horizontalalignment='center')
    ar1 = FancyArrow(grnd_lim, 100, -1, 0, width=8, color='black', head_length=0.3)
    plt.gca().add_patch(ar1)
    plt.text(grnd_lim - 0.2, 110, 'Ground\nSpectra',
                 fontsize=14,
                 horizontalalignment='right', verticalalignment='bottom')

    # Get the mass at the JWST spectra/RV limit
    jwst_lim = 16.5
    jdx = np.argmin(np.abs(stars['m_nirc2_Kp'] - jwst_lim))
    jwst_mass_label = '{0:.1f} M$_\odot$'.format(stars['mass'][jdx])
    
    plt.axvline(jwst_lim, color='black', linestyle='--', linewidth=2)
    plt.text(jwst_lim - 0.7, rng[3] - 30, jwst_mass_label,
                 fontsize=14,
                 horizontalalignment='center')
    ar1 = FancyArrow(jwst_lim, 100, -1, 0, width=8, color='black', head_length=0.3)
    plt.gca().add_patch(ar1)
    plt.text(jwst_lim - 0.2, 110, 'JWST\nSpectra',
                 fontsize=14,
                 horizontalalignment='right', verticalalignment='bottom')

    # pdx = np.where(stars['phase'] < 0)[0]
    # pms_mag = np.min(stars['m_nirc2_Kp'][pdx])
    # ar1 = FancyArrow(pms_mag + 0.1, 80, 0, 20, width=0.1, color='blue', head_length=10)
    # plt.gca().add_patch(ar1)
    # plt.text(pms_mag + 0.2, 72, 'Pre-MS\nTurn-On', color='blue',
    #          horizontalalignment='right', verticalalignment='top', fontsize=14)

    ##########
    # Completeness panel
    ##########
    # Our completeness
    plt.subplot(2, 1, 2)
    sjia_root = '/g/lu/scratch/siyao/work/3_young/'

    t = Table.read(sjia_root + 'completeness/completeness_info_r_0.0_30.0.txt', 
                       format='ascii.fixed_width')
    mag = t['Kp']
    comp = t['c_img'] * t['c_spec']
    comp_err = t['c_spec_err']

    # Append zeros at the end
    mag = np.append(mag, 20)
    comp = np.append(comp, 0)
    comp_err = np.append(comp_err, 1)
    
    workDir_old = '/u/jlu/work/gc/imf/klf/2012_05_01/' 
    img_file = workDir_old + 'image_completeness_r_0.0_30.0.txt'
    _img = Table.read(img_file, format='ascii')

    mag_img = _img['col1']
    comp_img = _img['col2']
    comp_ext_img = _img['col5'] 

    # interpolate
    f_img = interp1d(mag_img, comp_img, kind='slinear', bounds_error=False,
                         fill_value=(1,0))
    f_spec = interp1d(mag, comp, kind='slinear', bounds_error=False,
                         fill_value=(1,0))

    plt.plot(mag_img, comp_img, 'k-', label='$C_{{img}}$')
    plt.plot(mag, comp, 'r-', label='C$_{{spec,RV}}$')
    plt.xlabel('Kp')
    plt.ylabel('Completeness')
    plt.xlim(9, 18)
    plt.legend(loc='lower left', fontsize=12)

    plt.axvline(grnd_lim, color='grey', linestyle='--', linewidth=2)
    # plt.axvline(jwst_lim, color='black', linestyle='--', linewidth=2)

    # Get the mass at the JWST spectra/RV limit
    jwst_rv_lim = 16.5
    jwst_rv_label = '$\sigma_{RV}$~10 km/s'

    plt.axvline(jwst_rv_lim, color='black', linestyle='--', linewidth=2)
    # plt.text(jwst_rv_lim + 1.6, 0.5, jwst_rv_label,
    #              fontsize=14,
    #              horizontalalignment='center', verticalalignment='center')
    # ar1 = FancyArrow(jwst_rv_lim, 0.5, -1, 0, width=0.025, color='black', head_length=0.3)
    # plt.gca().add_patch(ar1)
    # plt.text(jwst_rv_lim - 0.18, 0.53, 'JWST\nSpectra',
    #              fontsize=14,
    #              horizontalalignment='right', verticalalignment='bottom')
    

    plt.savefig('plots/jwst_klf_spectral_sensitivity.png')

    # Estimate the number of young stars in the old and new samples.
    kbins_mid = kbins[:-1] + np.diff(kbins)
    ck_at_kbins = f_spec(kbins_mid)
    ck_at_kbins[kbins_mid<9] = 1.0
    ck_at_kbins[kbins_mid>17] = 0.0

    cj_at_kbins = f_img(kbins_mid)
    cj_at_kbins[kbins_mid<9] = 1.0
    cj_at_kbins[kbins_mid>=16.5] = 0.0
    
    N_tot = klf.sum()
    N_keck = (klf * ck_at_kbins).sum()
    N_jwst = (klf * cj_at_kbins).sum()

    for ii in range(len(kbins_mid)):
        fmt = '{0:4.1f} {1:4.2f} {2:4.0f} {3:4.0f}'
        print(fmt.format(kbins_mid[ii], cj_at_kbins[ii], klf[ii], klf[ii] * cj_at_kbins[ii]))

    print('N_tot  = {0:.0f}'.format(N_tot))
    print('N_keck = {0:.0f}'.format(N_keck))
    print('N_jwst = {0:.0f}'.format(N_jwst))

    foo = Table.read('/g/lu/scratch/siyao/work/3_young/data_rv/final.txt', format='ascii.fixed_width', delimiter='|')
    in_jwst = np.where((foo['x'] > -4.5) & (foo['x'] < 4.5) &
                       (foo['y'] > -4.5) & (foo['y'] < 4.5) &
                       (foo['kp'] <= 16.5))[0]
    print('N_known = {0:d}'.format( len(foo)) )
    print('N_known_in_jwst_field = {0:d}'.format( len(in_jwst)) )
    
    return
    

def factors(n): 
    return list(reduce(list.__add__, 
                ([i, n//i] for i in range(1, int(n**0.5) + 1) if n % i == 0)))


def plot_period_wobble():
    """
    Plot astrometric wobble vs. period for a range of BH binaries. 
    """
    m_star = 1.0 * u.Msun
    m_bh = np.array([1., 10., 100.]) * u.Msun
    a = np.array([3., 6., 10., 15.]) * u.AU
    P = np.array([1., 5., 10.]) * u.yr
    D = 8. * u.kpc

    # Default index when looping through arrays on other parameters. 
    ii_def = 1

    plt.close(1)
    plt.figure(1, figsize=(8,4))
    plt.clf()
    plt.subplots_adjust(left=0.13, bottom=0.2, top=0.90)

    # Loop through m_bh
    a_dense = np.geomspace(0.1, 1000., 500) * u.AU
    label_P_pos = [12, 11, 6]
    for ii in range(len(m_bh)):
        wob_ii, P_ii = wobble.astro_wobble(m_bh=m_bh[ii],
                                    m_star=m_star,
                                    distance=D,
                                    semi_major_axis=a_dense)

        plt.plot(P_ii, wob_ii, marker=None, ls='--', color='blue')

        mid = np.argmin(np.abs(P_ii.value - label_P_pos[ii]))
        #mid = int(len(wob_ii) / 2.) + 100
        
        dy = ((wob_ii[mid+1] - wob_ii[mid]) / u.mas).value
        dx = ((P_ii[mid+1] - P_ii[mid]) / u.yr).value

        angle = np.rad2deg(np.arctan2(dy, dx))

        # annotate with transform_rotates_text to align text and line
        plt.text(P_ii[mid].value, wob_ii[mid].value, 
                 f'M$_{{BH}}$={m_bh[ii].value:.0f} M$_\odot$',
                 ha='center', va='bottom',
                 transform_rotates_text=True,
                 rotation=angle, rotation_mode='anchor',
                 color='blue')

    # Loop through semi-major axis
    m_bh_dense = np.geomspace(0.01, 3e6, 500) * u.Msun
    label_P_pos = [2.5, 4.5, 6, 9]
    for ii in range(len(a)):
        wob_ii, P_ii = wobble.astro_wobble(m_bh=m_bh_dense,
                                    m_star=m_star,
                                    distance=D,
                                    semi_major_axis=a[ii])
        
        plt.plot(P_ii, wob_ii, marker=None, ls='-.', color='red')

        mid = np.argmin(np.abs(P_ii.value - label_P_pos[ii]))
        # mid = int(len(wob_ii) / 2.)# - 50
        
        dy = ((wob_ii[mid+1] - wob_ii[mid]) / u.mas).value
        dx = ((P_ii[mid+1] - P_ii[mid]) / u.yr).value

        angle = np.rad2deg(np.arctan2(dy, dx))
        if angle > 90 or angle < -90:
            angle += 180
        
        # annotate with transform_rotates_text to align text and line
        plt.text(P_ii[mid].value, wob_ii[mid].value, 
                 f'a={a[ii].value:.0f} AU',
                 ha='center', va='bottom',
                 transform_rotates_text=True,
                 rotation=angle, rotation_mode='anchor',
                 color='red')
        

    plt.xlabel('Period (yr)')
    plt.ylabel('Astrometric P2V\nWobble (mas)')

    plt.xlim(0, 14)
    plt.ylim(0, 5)
    
    plt.fill_between([0, 14], [0, 0], [0.3, 0.3], color='grey', alpha=0.1)

    plt.title("BH + Star (1M$_\odot$) Binaries at the GC")
    plt.savefig('gc_bh_astrom_wobble.png')
    
    return
