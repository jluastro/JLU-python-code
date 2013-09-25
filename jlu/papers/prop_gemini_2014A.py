from jlu.nirc2 import synthetic as syn
import os
import numpy as np
import pylab as py
import math

def plot_cluster_isochrones(redo_iso=False):
    """
    Plot isochrones and mass-luminosity functions for M17, Wd 2, Wd 1, and RSGC 1.
    """
    # Cluster Info
    name = ['M17', 'Wd 2', 'Wd 1', 'RSGC 1']
    dist = np.array([2100, 4160, 3600, 6000])
    age = np.array([1., 2., 5., 12.]) * 1.0e6
    AV = np.array([5., 6.5, 10., 23.])

    # Derived properties
    logage = np.log10(age)
    AKs = AV / 10.0

    iso_all = []

    # Loop through the clusters and make the isochrones.
    for ii in range(len(name)):
        pickleFile = 'syn_nir_d' + str(dist[ii]).zfill(5) + '_a' \
            + str(int(round(logage[ii]*100))).zfill(3) + '.dat'

        if (not os.path.exists(pickleFile)) or (redo_iso == True):
                AKsGrid = np.array([AKs[ii]])
                syn.nearIR(dist[ii], logage[ii], AKsGrid=AKsGrid)
                
        iso_all.append( syn.load_nearIR_dict(pickleFile) )


    ##########
    # Plot CMDs
    ##########
    py.figure(1)
    py.clf()
    py.subplots_adjust(left=0.15)
    colors = ['green', 'cyan', 'blue', 'red']
    for ii in range(len(iso_all)):
        iso = iso_all[ii]
        
        py.plot(iso['J'] - iso['K'], iso['K'], label=name[ii],
                linewidth=2, color=colors[ii])

        idx1 = np.argmin( np.abs(iso['mass'] - 1.0) )
        py.plot(iso['J'][idx1] - iso['K'][idx1], iso['K'][idx1], 'ks', 
                color=colors[ii], mew=0, ms=8)

    py.gca().invert_yaxis()
    py.xlabel("J - K color")
    py.ylabel("K magnitude")
    py.text(iso['J'][idx1] - iso['K'][idx1], iso['K'][idx1],
            r'1 M$_\odot$', color=colors[ii],
        horizontalalignment='right', verticalalignment='top')
    py.legend(loc="lower left")
    py.savefig('clusters_cmd_jk.png')

    ##########
    # Plot mass-luminosity relations for each of the filters.
    ##########
    py.figure(2, figsize=(12,4))
    py.clf()
    py.subplots_adjust(left=0.06, bottom=0.15, wspace=0.22, right=0.97)

    py.subplot(1, 3, 1)
    for ii in range(len(iso_all)):
        iso = iso_all[ii]
        py.plot(iso['mass'], iso['J'], linewidth=2,
                color=colors[ii], label=name[ii])
    py.xlabel(r'Stellar Mass (M$_\dot$)')
    py.ylabel('J magnitude')
    py.xlim(0, 5)
    py.ylim(26, 9)


    py.subplot(1, 3, 2)
    for ii in range(len(iso_all)):
        iso = iso_all[ii]
        py.plot(iso['mass'], iso['H'], linewidth=2,
                color=colors[ii], label=name[ii])
    py.legend(mode="expand", bbox_to_anchor=(-0.5, 0.99, 2.0, 0.08),
              loc=3, ncol=4, frameon=False)
    py.xlabel(r'Stellar Mass (M$_\dot$)')
    py.ylabel('H magnitude')
    py.xlim(0, 5)
    py.ylim(26, 9)

    py.subplot(1, 3, 3)
    for ii in range(len(iso_all)):
        iso = iso_all[ii]
        py.plot(iso['mass'], iso['K'], linewidth=2,
                color=colors[ii], label=name[ii])
    py.xlabel(r'Stellar Mass (M$_\dot$)')
    py.ylabel('K magnitude')
    py.xlim(0, 5)
    py.ylim(26, 9)

    py.savefig('clusters_mass_luminosity_jhk.png')


    ##########
    # Print out 0.1 Msun, 1 Msun, 10 Msun
    # photometry table.
    ##########

    for ii in range(len(iso_all)):
        iso = iso_all[ii]

        print('')
        print('Cluster: ' + name[ii])
        print('')

        hdr = '{0:6s} {1:6s} {2:6s} {3:6s}'
        dat = '{0:6.1f} {1:6.2f} {2:6.2f} {3:6.2f}'
        print(hdr.format('  Mass', '   J', '   H', '   K'))
        print(hdr.format('  ----', '  ---', '  ---', '  ---'))

        idx01 = np.argmin( np.abs(iso['mass'] - 0.1) )
        print(dat.format(iso['mass'][idx01], iso['J'][idx01][0],
                         iso['H'][idx01][0], iso['K'][idx01][0]))

        idx1 = np.argmin( np.abs(iso['mass'] - 1.0) )
        print(dat.format(iso['mass'][idx1], iso['J'][idx1][0],
                         iso['H'][idx1][0], iso['K'][idx1][0]))

        idx10 = np.argmin( np.abs(iso['mass'] - 10.0) )
        print(dat.format(iso['mass'][idx10], iso['J'][idx10][0],
                         iso['H'][idx10][0], iso['K'][idx10][0]))
print('')

              
    

def ast_err(tint_in_min, fwhm_in_mas, snr):
    ast_err = 0.180 * math.sqrt((5.4 / tint_in_min) + (fwhm_in_mas / 75.)**2 * (160.0/snr)**2)

    print ast_err, ' mas'
    return ast_err

def overheads(tint, coadds, dithers, nPerDither, readmode='bright'):
    """
    """
    readmode_dict = {'bright': 5.6,
                     'faint': 22.4,
                     'very faint': 44.8}

    readmode_list = readmode_dict.keys()
    if readmode not in readmode_list:
        print 'Incorrect read mode specified. Must be one of:'
        print readmode_list
        return

    read_overhead = readmode_dict[readmode]
    
    overhead = 21 + read_overhead * coadds + 6.5 * (coadds - 1)
    overhead *= dithers * nPerDither
    overhead += 30 * dithers

    integration = tint * coadds * dithers * nPerDither

    # Convert to minutes
    overhead /= 60.0
    integration /= 60.0

    totalTime = overhead + integration

    print 'Total Clock Time: {0:5.0f} min  or  {1:5.1f} hr'.format(totalTime, totalTime/60.)
    print 'Overheads:        {0:5.0f} min  or  {1:5.1f} hr'.format(overhead, overhead/60.)
    print 'Integration Time: {0:5.0f} min  or  {1:5.1f} hr'.format(integration, integration/60.)
