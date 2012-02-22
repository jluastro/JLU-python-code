from sqlite3 import dbapi2 as sqlite
import numpy as np
import pylab as py
import asciidata
import histNofill
import math
from mpl_toolkits.axes_grid import AxesGrid
import pyfits

def klfYoung():
    dbfile = '/u/jlu/data/gc/database/stars.sqlite'

    # Create a connection to the database file
    connection = sqlite.connect(dbfile)

    # Create a cursor object
    cur = connection.cursor()

    # Get info on the stars
    cur.execute('select * from stars where young = "T" and cuspPaper = "T"')
    
    rows = cur.fetchall()

    starCnt = len(rows)
    starName = []
    x = []
    y = []
    kp = []
    Ak = []
    starInField = []
    
    # Loop through each star and pull out the relevant information
    for ss in range(starCnt):
        record = rows[ss]
        
        name = record[0]

        # Check that we observed this star and what field it was in.
        cur.execute('select field from spectra where name = "%s"' % 
                    (name))

        row = cur.fetchone()
        if row != None:
            starInField.append(row[0])

            starName.append(name)
            kp.append(record[4])
            x.append(record[7])
            y.append(record[9])
            Ak.append(record[24])
        else:
            #print 'We do not have data on this star???'
            #starInField.append('C')
            continue

    starCnt = len(starName)
    starName = np.array(starName)
    starInField = np.array(starInField)
    x = np.array(x)
    y = np.array(y)
    kp = np.array(kp)
    Ak = np.array(Ak)

    # Now get the completeness corrections for each field.
    completenessFile = '/u/jlu/work/gc/imf/klf/2010_04_02/'
    completenessFile += 'completeness_extinct_correct.txt'
    completeness = asciidata.open(completenessFile)

    fields = ['C', 'E', 'SE', 'S', 'W', 'N', 'NE', 'SW', 'NW']

    compKp = completeness[0].tonumpy()

    comp = {}
    for ff in range(len(fields)):
        field = fields[ff]
        comp[field] = completeness[1+ff].tonumpy()


    # Load up the areas for each field
    masksDir = '/u/jlu/work/gc/imf/klf/2010_04_02/osiris_fov/masks/'
    area = {}
    area['C'] = pyfits.getdata(masksDir + 'central.fits').sum() * 0.01**2
    area['E'] = pyfits.getdata(masksDir + 'east.fits').sum() * 0.01**2
    area['SE'] = pyfits.getdata(masksDir + 'southeast.fits').sum() * 0.01**2
    area['S'] = pyfits.getdata(masksDir + 'south.fits').sum() * 0.01**2
    area['W'] = pyfits.getdata(masksDir + 'west.fits').sum() * 0.01**2
    area['N'] = pyfits.getdata(masksDir + 'north.fits').sum() * 0.01**2
    area['NE'] = pyfits.getdata(masksDir + 'northeast.fits').sum() * 0.01**2
    area['SW'] = pyfits.getdata(masksDir + 'southwest.fits').sum() * 0.01**2
    area['NW'] = pyfits.getdata(masksDir + 'northwest.fits').sum() * 0.01**2

    KLFs = np.zeros((len(fields), len(compKp)), dtype=float)
    KLFs_ext = np.zeros((len(fields), len(compKp)), dtype=float)
    KLFs_ext_cmp = np.zeros((len(fields), len(compKp)), dtype=float)
    eKLFs_ext_cmp = np.zeros((len(fields), len(compKp)), dtype=float)

    kp_ext = kp - Ak + 3.0

    for ff in range(len(fields)):
        field = fields[ff]

        # Pull out all the stars in the field
        sdx = np.where(starInField == field)[0]

        kpInField = kp[sdx]
        AkInField = Ak[sdx]
        nameInField = starName[sdx]

        # Set all stars to Ak = 3
        kpInField_ext = kpInField - AkInField + 3.0

        # Make a binned luminosity function
        binSizeKp = compKp[1] - compKp[0]

        perAsec2Mag = area[field] * binSizeKp

        for kk in range(len(compKp)):
            kp_lo = compKp[kk]
            kp_hi = compKp[kk] + binSizeKp

            idx1 = np.where((kpInField >= kp_lo) & (kpInField < kp_hi))[0]
            KLFs[ff,kk] = len(idx1) / perAsec2Mag

            idx2 = np.where((kpInField_ext >= kp_lo) & (kpInField_ext < kp_hi))[0]
            KLFs_ext[ff,kk] = len(idx2) / perAsec2Mag

            KLFs_ext_cmp[ff,kk] = KLFs_ext[ff,kk] / comp[field][kk]
            eKLFs_ext_cmp[ff,kk] = math.sqrt(len(idx2)) 
            eKLFs_ext_cmp[ff,kk] /= perAsec2Mag * comp[field][kk]

            idx = np.where(np.isnan(KLFs_ext_cmp[ff]) == True)[0]
            for ii in idx:
                KLFs_ext_cmp[ff,ii] = 0.0



    # ==========
    # Plot the KLF
    # ==========

    outputDir = '/u/jlu/work/gc/imf/klf/2010_04_02/plots/'

    py.clf()
    colors = ['blue', 'green', 'red', 'cyan', 'magenta', 'yellow',
              'purple', 'orange', 'brown']
    legendItems = []
    for ff in range(len(fields)):
        plt = py.plot(compKp, KLFs_ext_cmp[ff], 'k-', color=colors[ff])
        py.plot(compKp, KLFs_ext[ff], 'k--', color=colors[ff])
        #py.plot(compKp, KLFs[ff], 'k-.', color=colors[ff])
        legendItems.append(plt)

        print 'KLF for field ' + fields[ff]

    py.xlabel('Kp magnitude (A_Kp = 3)')
    py.ylabel('N_stars / (arcsec^2 mag)')
    py.legend(legendItems, fields, loc='upper left')
    py.savefig(outputDir + 'klf_young_fields_all.png')
    

    # Make a global KLF
    KLFall = KLFs_ext_cmp.sum(axis=0)
    KLFouter = KLFs_ext_cmp[1:,:].sum(axis=0)
    KLFinner = KLFs_ext_cmp[0,:]

    eKLFall = np.sqrt((eKLFs_ext_cmp**2).sum(axis=0))
    eKLFouter = np.sqrt((eKLFs_ext_cmp[1:,:]**2).sum(axis=0))
    eKLFinner = eKLFs_ext_cmp[0,:]


    p_compKp, p_KLFall = histNofill.convertForPlot(compKp, KLFall)
    p_compKp, p_KLFouter = histNofill.convertForPlot(compKp, KLFouter)
    p_compKp, p_KLFinner = histNofill.convertForPlot(compKp, KLFinner)

    
    py.clf()
    fig = py.figure(1)
    py.subplots_adjust(top=0.95, right=0.95)
    grid = AxesGrid(fig, 111, nrows_ncols=(3,1), axes_pad=0.12, add_all=True, 
                    share_all=True, aspect=False)
    grid[2].plot(p_compKp, p_KLFall, 'k-', linewidth=2)
    grid[2].errorbar(compKp+0.25, KLFall, yerr=eKLFall, fmt='k.')
    grid[2].text(9.25, 25, 'KLF all')
    grid[2].set_xlabel('Kp magnitude (A_Kp = 3)')

    grid[1].plot(p_compKp, p_KLFouter, 'k-', linewidth=2)
    grid[1].errorbar(compKp+0.25, KLFouter, yerr=eKLFouter, fmt='k.')
    grid[1].set_ylabel('N_stars / (arcsec^2 mag)')
    grid[1].text(9.25, 25, 'KLF outer')

    grid[0].plot(p_compKp, p_KLFinner, 'k-', linewidth=2)
    grid[0].errorbar(compKp+0.25, KLFinner, yerr=eKLFinner, fmt='k.')
    grid[0].text(9.25, 25, 'KLF central')

    grid[0].set_xlim(9, 16)
    grid[0].set_ylim(0, 12)
    grid[1].set_ylim(0, 12)
    grid[2].set_ylim(0, 12)
    
    py.savefig(outputDir + 'klf_young_fields.png')


    py.clf()
    py.subplots_adjust(top=0.95, right=0.95)

    p1 = py.plot(p_compKp, p_KLFinner, 'k-', linewidth=2)
    py.errorbar(compKp+0.25, KLFinner, yerr=eKLFinner, fmt='k.')
    p2 = py.plot(p_compKp, p_KLFouter, 'b-', linewidth=2)
    py.errorbar(compKp+0.25, KLFouter, yerr=eKLFouter, fmt='b.')
    py.xlabel('Kp magnitude (A_Kp = 3)')
    py.ylabel('N_stars / (arcsec^2 mag)')
    py.legend((p1, p2), ('Inner', 'Outer'))
    py.xlim(9, 16)
    py.ylim(0, 12)
    
    py.savefig(outputDir + 'klf_young_fields_inout.png')

    # ==========
    # Convert to masses
    # Assume solar metallicity, A_K=3, distance = 8 kpc, age = 6 Myr
    # ==========
    genevaFile = '/u/jlu/work/models/geneva/iso/020/c/'
    genevaFile += 'iso_c020_0680.UBVRIJHKLM'
    model = asciidata.open(genevaFile)
    modMass = model[1].tonumpy()
    modV = model[6].tonumpy()
    modVK = model[11].tonumpy()
    modHK = model[15].tonumpy()

    modK = modV - modVK
    modH = modK + modHK

    # Convert from K to Kp (Wainscoat and Cowie 1992)
    modKp = modK + 0.22 * (modH - modK)
    
    dist = 8000.0
    distMod = -5.0 + 5.0 * math.log10(dist)
    
    # Convert our observed magnitudes to absolute magnitudes.
    # Use the differential extinction corrected individual magnitudes.
    # Also use the diff. ex. corrected and completeness corrected KLF.
    absKp = kp_ext - 3.0 - distMod
    absKpKLF = compKp - 3.0 - distMod
    p_absKpKLF = p_compKp - 3.0 - distMod

    # First, calculate the indvidiual masses
    modMassStar = np.zeros(len(absKp), dtype=float)
    modKpStar = np.zeros(len(absKp), dtype=float)
    for ii in range(len(absKp)):
        idx = abs(absKp[ii] - modKp).argmin()

        modMassStar[ii] = modMass[idx]
        modKpStar[ii] = modKp[idx]

    # Calculate the mass function
    modMassIMF = np.zeros(len(absKpKLF), dtype=float)
    modKpIMF = np.zeros(len(absKpKLF), dtype=float)
    for ii in range(len(absKpKLF)):
        idx = abs(absKpKLF[ii] - modKp).argmin()

        modMassIMF[ii] = modMass[idx]
        modKpIMF[ii] = modKp[idx]

    # Calculate the mass function we can plot
    p_modMassIMF = np.zeros(len(p_absKpKLF), dtype=float)
    p_modKpIMF = np.zeros(len(p_absKpKLF), dtype=float)
    for ii in range(len(p_absKpKLF)):
        idx = abs(p_absKpKLF[ii] - modKp).argmin()

        p_modMassIMF[ii] = modMass[idx]
        p_modKpIMF[ii] = modKp[idx]


    # ==========
    # Plot the masses and mass functions
    # ==========
    py.clf()
    py.plot(modMassStar, absKp, 'rs')
    py.plot(modMassStar, modKpStar, 'b.')
    py.xlabel('Mass (Msun)')
    py.ylabel('Kp (mag)')
    py.legend(('Observed Absolute', 'Model Absolute'))
    py.savefig(outputDir + 'pdmf_young_indiv.png')

    py.clf()
    py.plot(modMassIMF, KLFall, 'ks')
    py.plot(p_modMassIMF, p_KLFall, 'k-')
    py.xlabel('Mass (Msun)')
    py.ylabel('N')


    # Completeness correction (polynomial fits)
    comp = {'C': [44.6449, -14.4162, 1.75114, -0.0923230, 0.00177092],
            'E': [228.967, -71.9702, 8.43935, -0.435115, 0.00830836],
            'SE': [527.679, -172.892, 21.1500, -1.14200, 0.0229464],
            'S': [-293.901, 90.8734, -10.4823, 0.537018, -0.0103230],
            'W': [2476.22, -795.154, 95.1204, -5.02096, 0.0986629],
            'N': [334.286, -107.820, 12.8997, -0.676552, 0.0131203],
            'NE': [-686.107, 224.631, -27.3240, 1.46633, -0.0293125],
            'all': [-142.858, 58.5277, -9.49619, 0.768180, -0.0309478, 0.000495177]}

    # Make a mass function 
    massEdges = 10**np.arange(1, 1.5, 0.05)
    massBins = massEdges[:-1] + ((massEdges[1:] - massEdges[:-1]) / 2.0)
    massHist = np.zeros(len(massBins), dtype=float)

    p_massBins = np.zeros(len(massBins)*2, dtype=float)
    p_massHist = np.zeros(len(massBins)*2, dtype=float)

    for mm in range(len(massBins)):
        m_lo = massEdges[mm]
        m_hi = massEdges[mm+1]

        idx = np.where((modMassStar > m_lo) & (modMassStar <= m_hi))[0]

        # Find the completeness factor for this mass bin
        kpInBin = kp_ext[idx].mean()
        cmpCorr = comp['all'][0] + comp['all'][1]*kpInBin + \
            comp['all'][2]*kpInBin**2 + comp['all'][3]*kpInBin**3 +\
            comp['all'][4]*kpInBin**4 + comp['all'][5]*kpInBin**5

        print m_lo, m_hi, cmpCorr, len(idx), kpInBin
        massHist[mm] = len(idx)# / cmpCorr

        p_massBins[2*mm] = m_lo
        p_massBins[2*mm + 1] = m_hi
        p_massHist[2*mm] = len(idx)# / cmpCorr
        p_massHist[2*mm+1] = len(idx)# / cmpCorr

    py.clf()
    py.plot(p_massBins, p_massHist, 'k-')
    py.plot(massBins, massHist, 'ks')
    py.show()
    
    


def massLuminosity(distance=8.0, AKp=3.0, 
                   isoFileName='iso_c020_0680.UBVRIJHKLM'):
    """
    Plot the mass luminosity relationship for the GC young stars. 
    Make two figures, one for
    absolute magnitudes (Kp) and one for apparent magnitudes at the 
    specified distance and AKp extinction.
    """
    # ==========
    # Convert to masses
    # Assume solar metallicity, A_K=3, distance = 8 kpc, age = 6 Myr
    # ==========
    genevaFile = '/u/jlu/work/models/geneva/iso/020/c/'
    genevaFile += isoFileName

    model = asciidata.open(genevaFile)
    modMass = model[1].tonumpy()
    modV = model[6].tonumpy()
    modVK = model[11].tonumpy()
    modHK = model[15].tonumpy()

    modK = modV - modVK
    modH = modK + modHK

    # Convert from K to Kp (Wainscoat and Cowie 1992)
    modKp = modK + 0.22 * (modH - modK)
    
    dist = 8000.0
    distMod = -5.0 + 5.0 * math.log10(dist)

    modKpGC = modKp + distMod + AKp

    outputDir = '/u/jlu/work/gc/imf/klf/2010_04_02/plots/'
    outputSuffix = '_%.1f_%.1f.png' % (distance, AKp)


    py.clf()
    py.plot(modMass, modKp, 'b.')
    py.xlabel('Mass (Msun)')
    py.ylabel('Absolute Kp (magnitude)')
    rng = py.axis()
    py.ylim(rng[3], rng[2])
    py.savefig(outputDir + 'mass_vs_absKp' + outputSuffix)
    py.show()

    py.clf()
    py.plot(modMass, modKpGC, 'b.')
    py.xlabel('Mass (Msun)')
    py.ylabel('Apparent Kp (magnitude)')
    rng = py.axis()
    py.ylim(rng[3], rng[2])
    py.title('Distance = %.1f kpc, A_Kp = %.1f' % (distance, AKp))
    py.savefig(outputDir + 'mass_vs_appKp' + outputSuffix)
    py.show()
    
    

