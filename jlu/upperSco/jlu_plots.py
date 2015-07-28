import pylab as plt
from astropy.io import fits
import glob

def make_plots():
    work_dir = '/u/cmutnik/work/upperSco_copy/finished/'

    # Read in data
    files = glob.glob(work_dir + '*.fits')

    specs = []

    for ff in range(len(files)):
        spec = fits.getdata(files[ff])

        if ff == 0:
            tot0 = spec[1].sum()

        spec[1] *= tot0 / spec[1].sum()

        specs.append(spec)

    # Plot
    plt.clf()
    for ff in range(len(files)):
        legend = files[ff].split('/')[-1]
        plt.semilogy(specs[ff][0], specs[ff][1], label=legend)

    plt.legend(loc='lower left')
    plt.xlim(0.7, 2.55)

    return
