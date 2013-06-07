import pylab as py
import numpy as np
import atpy
import pdb

def ssp_spectrum(logAge):

    ##########
    # Stellar Evolution Models
    ##########
    iso = evolution.get_merged_isochrone(logAge=logAge)

    # Trim down to get rid of repeat masses or 
    # mass resolutions higher than 1/1000. We will just use the first
    # unique mass after rounding by the nearest 0.001.
    mass_rnd = np.round(evol.mass, decimals=3)
    tmp, idx = np.unique(mass_rnd, return_index=True)

    mass = evol.mass[idx]
    logT = evol.logT[idx]
    logg = evol.logg[idx]
    logL = evol.logL[idx]
    isWR = logT != evol.logT_WR[idx]

    temp = 10**logT

    # Convert luminosity to erg/s
    L_all = 10**(logL) * c.Lsun # luminsoity in erg/s

    # Calculate radius
    R_all = np.sqrt(L_all / (4.0 * math.pi * c.sigma * temp**4))
    R_all /= (c.cm_in_AU * c.AU_in_pc)


    ##########
    # IMF
    ##########
    mass, isMultiple, compMasses = sample_imf(clusterMass)


    ##########
    # Stellar Atmospheres
    ##########


                                              

def explore_miles():
    """
    Plot up some dwarf and giant spectra for different alpha abundances.
    """
    milesDir = '/u/jlu/work/cdbs/grid/miles/'
    t = atpy.Table(milesDir + 'Milone2011_all.dat', type='ascii')
    t.rename_column('col1', 'name')
    t.rename_column('col2', 'name2')
    t.rename_column('col3', 'Teff')
    t.rename_column('col4', 'logg')
    t.rename_column('col5', 'FeH')
    t.rename_column('col6', 'MgFe')
    t.rename_column('col7', 'e_MgFe')

    py.clf()
    py.plot(t.FeH, t.MgFe, 'k.')
    
    # Select out models near solar Fe/H and with enhanced Mg/Fe
    idx1 = np.where((t.FeH >= -0.13) & (t.FeH < 0.13) & (t.MgFe >= 0.1) & (t.MgFe < 0.15))[0]
    idx2 = np.where((t.FeH >= -0.13) & (t.FeH < 0.13) & (t.MgFe >= 0.15) & (t.MgFe < 0.2))[0]
    idx3 = np.where((t.FeH >= -0.13) & (t.FeH < 0.13) & (t.MgFe >= 0.2) & (t.MgFe < 0.25))[0]
    idx4 = np.where((t.FeH >= -0.13) & (t.FeH < 0.13) & (t.MgFe >= 0.25) & (t.MgFe < 0.3))[0]

    # Read in the spectra for each of these models.
    nwave = 4300
    spectra1 = np.zeros((len(idx1), nwave), dtype=float)
    spectra2 = np.zeros((len(idx2), nwave), dtype=float)
    spectra3 = np.zeros((len(idx3), nwave), dtype=float)
    spectra4 = np.zeros((len(idx4), nwave), dtype=float)

    wave = None

    print '0.10 <= [Mg/Fe] < 0.15 and -0.13 <= [Fe/H] < 0.13: ', t.name2[idx1]
    print '0.15 <= [Mg/Fe] < 0.20 and -0.13 <= [Fe/H] < 0.13: ', t.name2[idx2]
    print '0.20 <= [Mg/Fe] < 0.25 and -0.13 <= [Fe/H] < 0.13: ', t.name2[idx3]
    print '0.25 <= [Mg/Fe] < 0.30 and -0.13 <= [Fe/H] < 0.13: ', t.name2[idx4]

    print t.columns

    # for ii in range(len(idx1)):
    #     file_name = 'm' + t.name[idx1[ii]][0:4] + 'V'
    #     sp = atpy.Table(milesDir + file_name, type='ascii')
    #     spectra1[ii] = sp.col2
    #     if wave == None:
    #         wave = sp.col1

    # for ii in range(len(idx2)):
    #     file_name = 'm' + t.name[idx2[ii]][0:4] + 'V'
    #     sp = atpy.Table(milesDir + file_name, type='ascii')
    #     spectra2[ii] = sp.col2

    # for ii in range(len(idx3)):
    #     file_name = 'm' + t.name[idx3[ii]][0:4] + 'V'
    #     sp = atpy.Table(milesDir + file_name, type='ascii')
    #     spectra3[ii] = sp.col2

    # for ii in range(len(idx4)):
    #     file_name = 'm' + t.name[idx4[ii]][0:4] + 'V'
    #     sp = atpy.Table(milesDir + file_name, type='ascii')
    #     spectra4[ii] = sp.col2

    
    # py.clf()
    # py.plot(wave, spectra1[0])
    # pdb.set_trace()
