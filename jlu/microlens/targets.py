import numpy as np
import pylab as plt
from astropy.table import Table

def select():
    targets_file = '/Users/jlu/work/microlens/target_selection/ogle_2017_05_17.txt'

    targs = Table.read(targets_file, format='ascii')

    candidates = np.where((targs['tau'] > 100) & (targs['Amax'] > 8) & (targs['I0'] < 21) & (targs['Amax'] < 1000))[0]

    print('Found {0:d} long-duration candidates'.format(len(candidates)))
    print( targs[candidates]['Event','RA_J2000','Dec_J2000','Tmax_UT','tau','Umin','Amax','Dmag','fbl','Ibl','I0'] )
    
    tau_bins = np.arange(0, 400, 5)
    plt.figure(1)
    plt.clf()
    plt.hist(targs['tau'], bins=tau_bins, log=True)
    plt.xlabel('tau (days)')

    amax_bins = np.arange(1, 100, 1)
    plt.figure(2)
    plt.clf()
    plt.hist(targs['Amax'], bins=amax_bins, log=True, alpha=0.5)
    plt.hist(targs['Amax'][candidates], bins=amax_bins, log=True, alpha=0.5)
    plt.xlabel('Amax')
    
    
    return
