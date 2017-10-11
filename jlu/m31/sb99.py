import numpy as np
import pylab as py
from astropy.io import fits as pyfits
from astropy.table import Table
import pickle
import time
import math, glob
import scipy
import pdb
import astropy
import pandas
import os
import matplotlib as mpl
import pysynphot as S
from nirc2 import synthetic as nirc2syn

sb99dir = '/Users/kel/Documents/Projects/M31/models/Starburst99/output/'

class Spec(object):
    def __init__(self,age=1.e8):
        self.age = age
        # read in Starburst99 spectrum, store as a class
        # age is the specified timestep to use
    
        inspecfile = sb99dir + '5000M_sun_300Myr.spectrum1'
        inspec = pandas.read_csv(inspecfile,delim_whitespace=True,header=None,skiprows=6,names=['t','wave','logflux','logstellar','lognebular'])
    
        # find the closest age to the one given as input
        tunique = np.unique(inspec.t)
        tuniquegood = np.nan_to_num(tunique)
        tdiff = tuniquegood - age
        tidx = np.argmin(np.abs(tdiff))
        tclose = tuniquegood[tidx]

        # get only the good values
        idx = np.where(inspec.t == tclose)[0]
        self.wave = np.array(inspec.wave[idx])
        self.logflux = np.array(inspec.logflux[idx])
        # at the distance of M31, ST mags
        self.mag = -2.5*np.log10((10.**self.logflux)/(4.*np.pi*(2.41e24**2)))-21.1
        self.logstellar = np.array(inspec.logstellar[idx])
        self.lognebular = np.array(inspec.lognebular[idx])

#def colors(spec=None):

def mags(age=None,mag435=None,plot=False):
    # uses the starburst99 spectrum, convolves with pre-defined filters to get the
    # color difference between F435W filter and others. Returns magnitudes in other
    # filters given the F435W magnitude.

    inspec = Spec(age=age)
    sp = S.ArraySpectrum(wave=inspec.wave,flux=inspec.mag,waveunits='angstrom',fluxunits='stmag') 

    # get the NIRC2 filters (HST filters are pre-defined)
    filtJ = nirc2syn.FilterNIRC2('J')
    filtH = nirc2syn.FilterNIRC2('H')
    filtK = nirc2syn.FilterNIRC2('Kp')
    
    # ACS
    obs330 = S.Observation(sp,S.ObsBandpass('acs,hrc,f330w'))
    obs435 = S.Observation(sp,S.ObsBandpass('acs,hrc,f435w'))
    # WFPC2
    obs555 = S.Observation(sp,S.ObsBandpass('wfpc2,1,f555w,a2d7,cont#49888'))
    obs814 = S.Observation(sp,S.ObsBandpass('wfpc2,1,f814w,a2d7,cont#49888'))
    obs1024 = S.Observation(sp,S.ObsBandpass('wfpc2,1,f1042M,a2d7,cont#49628'))
    # NIRC2
    obs1249 = S.Observation(sp,filtJ,binset=filtJ.wave)
    obs1633 = S.Observation(sp,filtH,binset=filtH.wave)
    obs2125 = S.Observation(sp,filtK,binset=filtK.wave)

    intmag = np.zeros(8,dtype=float)
    intmag[0] = obs330.effstim('stmag')
    intmag[1] = obs435.effstim('stmag')
    intmag[2] = obs555.effstim('stmag')
    intmag[3] = obs814.effstim('stmag')
    intmag[4] = obs1024.effstim('stmag')
    intmag[5] = obs1249.effstim('stmag')
    intmag[6] = obs1633.effstim('stmag')
    intmag[7] = obs2125.effstim('stmag')
    
    colors = np.zeros(8,dtype=float)
    calmag = np.zeros(8,dtype=float)
    # F435W - X
    for i in range(len(colors)):
        tmp = intmag[1] - intmag[i]
        colors[i] = tmp
        tmpmag = mag435 - tmp
        calmag[i] = tmpmag

    print('Filter: F435W-X color, mag')
    print('F330W: %5.2f, %5.2f' % (colors[0],calmag[0]))
    print('F435W: %5.2f, %5.2f' % (colors[1],calmag[1]))
    print('F555W: %5.2f, %5.2f' % (colors[2],calmag[2]))
    print('F814W: %5.2f, %5.2f' % (colors[3],calmag[3]))
    print('F1042M: %5.2f, %5.2f' % (colors[4],calmag[4]))
    print('J: %5.2f, %5.2f' % (colors[5],calmag[5]))
    print('H: %5.2f, %5.2f' % (colors[6],calmag[6]))
    print('Kp: %5.2f, %5.2f' % (colors[7],calmag[7]))

    magdiff = intmag[1] - mag435
    plfilt = [3300.,4350.,5550.,8140.,10420.,12490.,16330.,21250.]
    
    if plot:
        py.figure(2)
        py.clf()
        py.plot(inspec.wave,inspec.mag-magdiff)
        py.xlim(3000,22000)
        for i in range(len(plfilt)):
            py.plot([plfilt[i],plfilt[i]],[13,18],'k--')
        py.ylim(18,13)
        py.xlabel('Wavelength (angstrom)')
        py.ylabel('Mag')
        pltitle = 'P3 (5000 M$_{\odot}$, %4.1f Myr)' % (age/1.e6)
        py.title(pltitle)
        #pdb.set_trace()
