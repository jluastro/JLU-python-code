import numpy as np
import pylab as plt
from popstar import synthetic
from popstar import evolution
from popstar import reddening

def animate_ages():
    # Define isochrone parameters
    dist = 4000     # distance in parsecs
    AKs = 1.0       # Ks filter extinction in mags

    
    logAge = np.arange(6, 9, 0.05)   # Age in log(years)

    # Define extinction law and filters
    redlaw = reddening.RedLawCardelli(3.1) # Rv = 3.1
    evo_model = evolution.MISTv1()
    
    filt_list = ['nirc2,J', 'nirc2,Kp']

    plt.figure(1)

    for aa in range(len(logAge)):
        iso = synthetic.IsochronePhot(logAge[aa], AKs, dist,
                                      filters=filt_list, red_law=redlaw,
                                      evo_model=evo_model, mass_sampling=3)

        plt.clf()
        plt.plot(iso.points['m_nirc2_J'] - iso.points['m_nirc2_Kp'], 
                iso.points['m_nirc2_J'])
        plt.xlabel('J - Kp')
        plt.ylabel('J')
        plt.gca().invert_yaxis()
        plt.title('Age = 10^{0:.2f}'.format(logAge[aa]))

        plt.xlim(1, 3)
        plt.ylim(28, 6)
        plt.savefig('/u/jlu/doc/present/2020_06_ucb_lunch/iso_age_{0:.2f}.png'.format(logAge[aa]))

