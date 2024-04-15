import numpy as np
import pylab as plt
import pandas
from astropy.io import fits

paper_dir = '/Users/jlu/Google Drive/My Drive/instruments/Keck/Keck Optical AO/Systems Engineering/Performance Models/maos_sims_paper_figures/'

def plot_turbulence_profile(atm_conf_file='/u/jlu/work/ao/keck/maos/keck/base/atm_mk13n50p_ground_detail.conf'):
    tab = pandas.read_csv(atm_conf_file,
                          delimiter = "\\s+=\\s+",
                          header=None,
                          names=['key','val'],
                          comment='#',
                          engine='python')

    tab.set_index('key', inplace=True)

    # Process the string arrays into numpy arrays.
    tab.at['atm.ht', 'val'] = process_maos_array(tab.at['atm.ht', 'val'])
    tab.at['atm.wt', 'val'] = process_maos_array(tab.at['atm.wt', 'val'])
    tab.at['atm.ws', 'val'] = process_maos_array(tab.at['atm.ws', 'val'])

    # Convert heights to km.
    tab.at['atm.ht', 'val'] /= 1e3
    
    # If there is a zero layer, modify it so we can plot on a x-log scale.
    idx = np.where(tab.at['atm.ht', 'val'] == 0)[0]
    tab.at['atm.ht', 'val'][idx] = 0.001

    # Convert weights to percentages
    tab.at['atm.wt','val'] *= 100
    
    plt.figure(1)
    plt.clf()
    
    ax1 = plt.subplot(2, 1, 1)
    plt.semilogy(tab.at['atm.wt','val'], tab.at['atm.ht','val'])
    plt.ylim(0.001, 30)
    plt.xlim(0, np.max(tab.at['atm.wt','val'])*1.1)
    plt.ylabel('Height (km)')
    plt.xlabel('Turbulence Strength (%)')
    if len(idx) > 0:
        plt.title('Note: Layer at 0 m moved to 1 m for plotting.', fontsize=10)

    ax2 = plt.subplot(2, 1, 2, sharex=ax1)
    plt.plot(tab.at['atm.wt','val'], tab.at['atm.ht','val'], 'kx-')
    plt.xlim(0, np.max(tab.at['atm.wt','val'])*1.1)
    plt.ylabel('Height (km)')
    plt.xlabel('Turbulence Strength (%)')
    
    plt.savefig(atm_conf_file.replace(".conf", "_profile.png"))

    return tab

def process_maos_array(str_array):
    str_entries = str_array.split()
    str_entries[0] = str_entries[0][1:]
    str_entries[-1] = str_entries[-1][0:-1]

    entries = np.array([float(str_entry) for str_entry in str_entries])

    return entries

def plot_keck_pupil(pupil_fits_file='/u/jlu/code/maos/config/maos/bin/KECK_gaps_spiders.fits'):
    img, hdr = fits.getdata(pupil_fits_file, header=True)

    img_x_lo = hdr['OX']
    img_y_lo = hdr['OY']
    img_x_hi = img_x_lo + img.shape[1] * hdr['DX']
    img_y_hi = img_y_lo + img.shape[1] * hdr['DY']

    plt.figure(figsize=(10,10))
    plt.imshow(img, cmap='Greys', extent=[img_x_lo, img_x_hi, img_y_lo, img_y_hi])
    plt.xlabel('meters')
    plt.title(f'Pupil sampled with {img.shape} pixels at {hdr["DX"]:.4f} m/pix',
              fontsize=12)
    plt.savefig(paper_dir + 'KECK_gaps_spiders.png')

    return
    
