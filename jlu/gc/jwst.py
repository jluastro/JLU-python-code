import time
import numpy as np
import pylab as plt
from astropy.table import Table
from popstar import synthetic as syn
from popstar import atmospheres as atm
from popstar import evolution
from popstar import reddening
from popstar.imf import imf
from popstar.imf import multiplicity
import pickle

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


def plot_sim_cluster():
    yng_file = '/u/jlu/work/gc/jwst/2018_03_19/iso_6.60_2.10_08000.fits'
    old_file = '/u/jlu/work/gc/jwst/2018_03_19/iso_9.90_2.10_08000.fits'
    out_dir = '/u/jlu/work/gc/jwst/2018_03_19/'

    yng_clust_file = '/u/jlu/work/gc/jwst/2018_03_19/clust_6.60_2.10_08000.fits'
    med_clust_file = '/u/jlu/work/gc/jwst/2018_03_19/clust_8.00_2.10_08000.fits'
    old_clust_file = '/u/jlu/work/gc/jwst/2018_03_19/clust_9.90_2.10_08000.fits'

    # Isochrones
    yng = Table.read(yng_file)
    old = Table.read(old_file)

    # Clusters
    # yng_c = Table.read(yng_clust_file)
    # old_c = Table.read(old_clust_file)

    ## Mass-Luminosity Relationship at ~K-band
    plt.figure(1)
    plt.clf()
    plt.semilogx(yng['mass'], yng['m_jwst_F090W'], 'c.', label='F090W')
    plt.semilogx(yng['mass'], yng['m_jwst_F115W'], 'b.', label='F115W')
    plt.semilogx(yng['mass'], yng['m_jwst_F164N'], 'y.', label='F164N')
    plt.semilogx(yng['mass'], yng['m_jwst_F212N'], 'g.', label='F212N')
    plt.semilogx(yng['mass'], yng['m_jwst_F323N'], 'r.', label='F323N')
    plt.semilogx(yng['mass'], yng['m_jwst_F470N'], 'm.', label='F470N')
    plt.axhline(saturate['F090W'], color='c', linestyle='--')
    plt.axhline(saturate['F115W'], color='b', linestyle='--')
    plt.axhline(saturate['F164N'], color='y', linestyle='--')
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
    plt.figure(2)
    plt.clf()
    plt.plot(yng['m_jwst_F115W'] - yng['m_jwst_F323N'], yng['m_jwst_F323N'], 'k.')
    plt.plot(old['m_jwst_F115W'] - old['m_jwst_F323N'], old['m_jwst_F323N'], 'r.')
    plt.ylim(25, 8)
    plt.xlabel('F115W - F323N (mag)')
    plt.ylabel('F323N (mag)')
    plt.savefig(out_dir + 'gc_cmd_F115W_F323N.png')

    plt.figure(3)
    plt.clf()
    plt.plot(yng['m_jwst_F115W'] - yng['m_jwst_F212N'], yng['m_jwst_F212N'], 'k.')
    plt.plot(old['m_jwst_F115W'] - old['m_jwst_F212N'], old['m_jwst_F212N'], 'r.')
    plt.ylim(25, 8)
    plt.xlabel('F115W - F212N (mag)')
    plt.ylabel('F212N (mag)')
    plt.savefig(out_dir + 'gc_cmd_F115W_F212N.png')

    plt.figure(4)
    plt.clf()
    plt.plot(yng['m_jwst_F212N'] - yng['m_jwst_F470N'], yng['m_jwst_F212N'], 'k.')
    plt.plot(old['m_jwst_F212N'] - old['m_jwst_F470N'], old['m_jwst_F212N'], 'r.')
    plt.ylim(25, 8)
    plt.xlabel('F212N - F470N (mag)')
    plt.ylabel('F212N (mag)')
    plt.savefig(out_dir + 'gc_cmd_F212N_F470N.png')

    plt.figure(5)
    plt.clf()
    plt.plot(yng['m_jwst_F212N'] - yng['m_jwst_F323N'], yng['m_jwst_F212N'], 'k.')
    plt.plot(old['m_jwst_F212N'] - old['m_jwst_F323N'], old['m_jwst_F212N'], 'r.')
    plt.ylim(25, 8)
    plt.xlabel('F212N - F323N (mag)')
    plt.ylabel('F212N (mag)')
    plt.savefig(out_dir + 'gc_cmd_F212N_F323N.png')
    
    plt.figure(6)
    plt.clf()
    plt.plot(yng['m_jwst_F115W'] - yng['m_jwst_F212N'], yng['m_jwst_F212N'] - yng['m_jwst_F323N'], 'k.')
    plt.plot(old['m_jwst_F115W'] - old['m_jwst_F212N'], old['m_jwst_F212N'] - old['m_jwst_F323N'], 'r.')
    plt.xlabel('F115W - F212N (mag)')
    plt.ylabel('F212N - F323N (mag)')
    plt.savefig(out_dir + 'gc_ccd_F115W_F212N_F323N.png')

    
    return yng


def make_sim_cluster():
    work_dir = '/u/jlu/work/gc/jwst/2018_03_19/'
    
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
    red_law = reddening.RedLawHosek18_extended()
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
        
        
    
def plot_cycle1_fig1(yng_c=None, old_c=None):
    yng_file = '/u/jlu/work/gc/jwst/2018_03_19/iso_6.60_2.10_08000.fits'
    old_file = '/u/jlu/work/gc/jwst/2018_03_19/iso_9.90_2.10_08000.fits'
    out_dir = '/u/jlu/work/gc/jwst/2018_03_19/'

    yng_clust_file = '/u/jlu/work/gc/jwst/2018_03_19/clust_6.60_2.10_08000.fits'
    med_clust_file = '/u/jlu/work/gc/jwst/2018_03_19/clust_8.00_2.10_08000.fits'
    old_clust_file = '/u/jlu/work/gc/jwst/2018_03_19/clust_9.90_2.10_08000.fits'

    # Isochrones
    yng = Table.read(yng_file)
    old = Table.read(old_file)

    # Clusters
    if yng_c == None:
        yng_c = Table.read(yng_clust_file)
    if old_c == None:
        old_c = Table.read(old_clust_file)

    ## CMD
    plt.figure(10, figsize=(18, 6))
    plt.clf()
    plt.subplots_adjust(left=0.12, wspace=0.4)

    print(saturate['F115W'], saturate['F323N'], saturate['F115W']-saturate['F323N'])
    
    ymag = yng['m_jwst_F115W']
    ycol = yng['m_jwst_F115W'] - yng['m_jwst_F323N']
    omag = old['m_jwst_F115W']
    ocol = old['m_jwst_F115W'] - old['m_jwst_F323N']
    
    plt.subplot(1, 3, 1)
    plt.plot(ocol, omag, 'r-', color='red', label='6 Myr')
    plt.plot(ycol, ymag, 'k-', color='blue', label='8 Gyr')
    plt.ylim(30, 13)
    plt.xlim(6, 10)
    plt.xlabel('F115W - F323N (mag)')
    plt.ylabel('F115W (mag)')

    confusion = {'F115W': 27.2,
                'F323N': 19.3}
    
    plt.axhline(saturate['F115W'], color='black', linestyle='--')
    plt.text(6.2, saturate['F115W']-0.2, 'Saturation', fontsize=10)
    plt.axhline(confusion['F115W'], color='black', linestyle='--')
    plt.text(6.2, confusion['F115W']-0.2, 'Confusion', fontsize=10)
                    

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

    label_mass(yng, ycol, ymag, 10, '10')
    label_mass(yng, ycol, ymag, 5, '5')
    label_mass(yng, ycol, ymag, 2, '2')
    label_mass(yng, ycol, ymag, 1, '1', label_dcol=0.4)
    label_mass(yng, ycol, ymag, 0.5, '0.5', label_dcol=0.4)

    label_mass(old, ocol, omag, 1, '1', color='red')
    label_mass(old, ocol, omag, 2, '2', color='red')
    label_mass(old, ocol, omag, 0.5, '0.5', label_dcol=0.4, color='red')
    
    plt.legend(fontsize=12, loc='lower left')
    
    
    plt.subplot(1, 3, 2)
    plt.plot(old_c['m_jwst_F115W'] - old_c['m_jwst_F323N'], old_c['m_jwst_F115W'], 'r.', color='salmon', alpha=0.2, markersize=5)
    plt.plot(yng_c['m_jwst_F115W'] - yng_c['m_jwst_F323N'], yng_c['m_jwst_F115W'], 'k.', color='cornflowerblue', alpha=0.2, markersize=5)
    plt.plot(old['m_jwst_F115W'] - old['m_jwst_F323N'], old['m_jwst_F115W'], 'r-', color='red')
    plt.plot(yng['m_jwst_F115W'] - yng['m_jwst_F323N'], yng['m_jwst_F115W'], 'k-', color='blue')
    plt.ylim(30, 13)
    plt.xlim(6, 10)
    plt.xlabel('F115W - F323N (mag)')
    plt.ylabel('F115W (mag)')
    
    plt.axhline(saturate['F115W'], color='black', linestyle='--')
    plt.text(6.2, saturate['F115W']-0.2, 'Saturation', fontsize=10)
    plt.axhline(confusion['F115W'], color='black', linestyle='--')
    plt.text(6.2, confusion['F115W']-0.2, 'Confusion', fontsize=10)

    plt.subplot(1, 3, 3)
    plt.plot(old_c['m_jwst_F323N'] - old_c['m_jwst_F470N'],
             old_c['m_jwst_F115W'] - old_c['m_jwst_F212N'],
             'r.', color='salmon', alpha=0.2, markersize=5)
    plt.plot(yng_c['m_jwst_F323N'] - yng_c['m_jwst_F470N'],
             yng_c['m_jwst_F115W'] - yng_c['m_jwst_F212N'],
             'r.', color='cornflowerblue', alpha=0.2, markersize=5)
    plt.plot(old['m_jwst_F323N'] - old['m_jwst_F470N'],
             old['m_jwst_F115W'] - old['m_jwst_F212N'],
             'r-', color='red')
    plt.plot(yng['m_jwst_F323N'] - yng['m_jwst_F470N'],
             yng['m_jwst_F115W'] - yng['m_jwst_F212N'],
             'k-', color='blue')
    #plt.ylim(30, 13)
    #plt.xlim(6, 10)
    plt.ylabel('F115W - F212N (mag)')
    plt.xlabel('F323N - F470N (mag)')
    plt.ylim(5.8, 8.2)
    plt.savefig(out_dir + 'cycle1_cmd.png')

    

