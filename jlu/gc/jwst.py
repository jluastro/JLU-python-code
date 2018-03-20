import time
import numpy as np
import pylab as py
from astropy.table import Table
from popstar import synthetic as syn
from popstar import atmospheres as atm
from popstar import evolution
from popstar import reddening
from popstar.imf import imf
from popstar.imf import multiplicity
import pickle

def plot_sim_cluster():
    yng_file = '/Users/jlu/work/gc/jwst/iso_6.60_2.70_08000_JWST.fits'
    old_file = '/Users/jlu/work/gc/jwst/iso_9.70_2.70_08000_JWST.fits'
    out_dir = '/Users/jlu/work/gc/jwst/'

    yng = Table.read(yng_file)
    old = Table.read(old_file)

    saturate = {'F164N': 12.0,
                'F212N': 11.3,
                'F323N': 11.1,
                'F466N': 10.6,
                'F125W': 15.5,
                'F090W': 15.3}

    ## Mass-Luminosity Relationship at ~K-band
    py.figure(1)
    py.clf()
    py.semilogx(yng['mass'], yng['mag090w'], 'c.', label='F090W')
    py.semilogx(yng['mass'], yng['mag125w'], 'b.', label='F125W')
    py.semilogx(yng['mass'], yng['mag164n'], 'y.', label='F164N')
    py.semilogx(yng['mass'], yng['mag212n'], 'g.', label='F212N')
    py.semilogx(yng['mass'], yng['mag323n'], 'r.', label='F323N')
    py.semilogx(yng['mass'], yng['mag466n'], 'm.', label='F466N')
    py.axhline(saturate['F090W'], color='c', linestyle='--')
    py.axhline(saturate['F125W'], color='b', linestyle='--')
    py.axhline(saturate['F164N'], color='y', linestyle='--')
    py.axhline(saturate['F212N'], color='g', linestyle='--')
    py.axhline(saturate['F323N'], color='r', linestyle='--')
    py.axhline(saturate['F466N'], color='m', linestyle='--')
    py.gca().invert_yaxis()
    py.ylim(30, 10)
    py.xlabel('Mass (Msun)')
    py.ylabel('JWST Magnitudes')
    py.legend(loc='upper left', numpoints=1)
    py.savefig(out_dir + 'mass_luminosity.png')

    ## CMD
    py.figure(2)
    py.clf()
    py.plot(yng['mag125w'] - yng['mag323n'], yng['mag323n'], 'k.')
    py.plot(old['mag125w'] - old['mag323n'], old['mag323n'], 'r.')
    py.ylim(25, 8)
    py.xlabel('F125W - F323N (mag)')
    py.ylabel('F323N (mag)')
    py.savefig(out_dir + 'gc_cmd_F125W_F323N.png')

    py.figure(3)
    py.clf()
    py.plot(yng['mag125w'] - yng['mag212n'], yng['mag212n'], 'k.')
    py.plot(old['mag125w'] - old['mag212n'], old['mag212n'], 'r.')
    py.ylim(25, 8)
    py.xlabel('F125W - F212N (mag)')
    py.ylabel('F212N (mag)')
    py.savefig(out_dir + 'gc_cmd_F125W_F212N.png')

    py.figure(4)
    py.clf()
    py.plot(yng['mag212n'] - yng['mag466n'], yng['mag212n'], 'k.')
    py.plot(old['mag212n'] - old['mag466n'], old['mag212n'], 'r.')
    py.ylim(25, 8)
    py.xlabel('F212N - F466N (mag)')
    py.ylabel('F212N (mag)')
    py.savefig(out_dir + 'gc_cmd_F212N_F466N.png')

    py.figure(5)
    py.clf()
    py.plot(yng['mag125w'] - yng['mag212n'], yng['mag212n'] - yng['mag323n'], 'k.')
    py.plot(old['mag125w'] - old['mag212n'], old['mag212n'] - old['mag323n'], 'r.')
    py.xlabel('F125W - F212N (mag)')
    py.ylabel('F212N - F323N (mag)')
    py.savefig(out_dir + 'gc_ccd_F125W_F212N_F323N.png')
    
    return yng


def make_sim_cluster():
    work_dir = '/u/jlu/work/gc/jwst/2018_03_19/'
    
    ages = [4e6, 1e8, 8e9]
    cluster_mass = [1e4, 1e4, 1e7]
    AKs = 2.7
    deltaAKs = 1.0
    distance = 8000
    mass_sampling = 5

    isochrones = []
    clusters = []
    
    evo = evolution.MergedBaraffePisaEkstromParsec()
    atm_func = atm.get_merged_atmosphere
    red_law = reddening.RedLawHosek18()
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

        print( 'Constructed isochrone: %d seconds' % (time.time() - startTime))

        if ii < 2:
            imf_ii = my_imf_yng
        else:
            imf_ii = my_imf_old
            
        cluster = syn.ResolvedClusterDiffRedden(iso, imf_ii, cluster_mass[ii], deltaAKs)

        # Save generated clusters to file.
        save_file_fmt = '{0}/clust_{1:.2f}_{2:4.2f}_{3:4s}.fits'
        save_file_txt = save_file_fmt.format(work_dir, logAge, AKs, str(distance).zfill(5))
        save_file = open(save_file_txt, 'wb')
        pickle.dump( cluster, save_file )

    return
        
        
    

    return
