import numpy as np
import pylab as plt
from astropy.table import Table

def plot_ogle(phot_file):
    dat = Table.read(phot_file, format='ascii')
    dat.rename_column('col1', 'mjd')
    dat.rename_column('col2', 'mag')
    dat.rename_column('col3', 'err')

    idx = np.where(dat['mjd'] <  2450000)[0]
    dat['mjd'][idx] += 2450000.0

    plt.clf()
    plt.errorbar(dat['mjd'], dat['mag'], yerr=dat['err'], fmt='k.')
    plt.gca().invert_yaxis()
    plt.xlim(2457000, 2458000)
    plt.title(phot_file)
    
    return
