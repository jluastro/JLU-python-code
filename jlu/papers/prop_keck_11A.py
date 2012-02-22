import numpy as np
import pylab as py
import pyfits
import asciidata
import math
import pickle
from gcwork import starTables
from gcwork import objects
from sqlite3 import dbapi2 as sqlite
from scipy import interpolate
from matplotlib import nxutils
from jlu.starfinder import starPlant
from jlu.stellarModels import extinction
from jlu.nirc2 import synthetic
import pdb

dp_msc_images = ['mag06maylgs1_dp_msc_C_kp',
          'mag06maylgs1_dp_msc_C_SW_kp',
          'mag06maylgs1_dp_msc_C_NE_kp',
          'mag06maylgs1_dp_msc_C_SE_kp',
          'mag06maylgs1_dp_msc_C_NW_kp',
          'mag06maylgs1_dp_msc_NE_kp',
          'mag06maylgs1_dp_msc_SE_kp',
          'mag06maylgs1_dp_msc_NW_kp',
          'mag06maylgs1_dp_msc_SW_kp',
          'mag06maylgs1_dp_msc_E_kp',
          'mag06maylgs1_dp_msc_W_kp',
          'mag06maylgs1_dp_msc_N_kp',
          'mag06maylgs1_dp_msc_S_kp']

dp_msc_fields = []
for ii in range(len(dp_msc_images)):
    field = dp_msc_images[ii].replace('mag06maylgs1_dp_msc_', '').replace('_kp', '')
    dp_msc_fields.append(field)

cooStarDict = {'C': 'irs16C',
               'C_SW': 'irs2',
               'C_NE': 'irs16C',
               'C_SE': 'irs33E',
               'C_NW': 'irs34W',
               'NE': 'S11-6',
               'SE': 'S12-2',
               'NW': 'S8-3',
               'SW': 'idSW2',
               'E': 'S5-183',
               'W': 'S5-69',
               'N': 'S10-3',
               'S': 'irs14SW'}

def gcimf_completeness():
    """
    Make a completeness curve for every osiris spectroscopic field. Use
    the completeness analysis from mag06maylgs1_dp_msc.
    """
    # Keep a mapping of which OSIRIS field corresponds to which
    # NIRC2 field in the dp_msc.
    osiris2nirc2 = {'GC Central': 'C',
                    'GC East': 'C',
                    'GC South': 'C',
                    'GC Southeast': 'C', 
                    'GC Southwest': 'C',
                    'GC West': 'C',
                    'GC Northwest': 'C',
                    'GC North': 'C',
                    'GC Northeast': 'C',
                    'E2-1': 'C_NE',
                    'E2-2': 'C_SE',
                    'E2-3': 'C_SE',
                    'E3-1': 'E',
                    'E3-2': 'E',
                    'E3-3': 'E',
                    'E4-1': 'E',
                    'E4-2': 'E',
                    'E4-3': 'E'}

    # Load up the label.dat file and use the coordinates in there
    # to astrometrically calibrate each NIRC2 dp_msc field.
    label = starTables.Labels()

    # Load up the spectroscopic database to get the field-of-view definitions.
    dbfile = '/u/jlu/data/gc/database/stars.sqlite'

    connection = sqlite.connect(dbfile)
    cur = connection.cursor()
    cur.execute('select name, x_vertices, y_vertices from fields')
    rows = cur.fetchall()

    # Completeness Directory
    compDir = '/u/jlu/work/gc/dp_msc/2010_09_17/completeness'

    # NIRC2 Data directory
    nirc2Dir = '/u/ghezgroup/data/gc/06maylgs1/combo'

    for rr in range(len(rows)):
        fieldName = rows[rr][0]
        print 'Working on field %s' % fieldName

        xverts = np.array([float(ff) for ff in rows[rr][1].split(',')])
        yverts = np.array([float(ff) for ff in rows[rr][2].split(',')])

        xyverts = np.column_stack((xverts, yverts))

        # Load up the corresponding NIRC2 completeness starlists
        nircFieldName = osiris2nirc2[fieldName]
        alignDir = '%s/%s/kp/06maylgs1/align/' % \
            (compDir, nircFieldName)

        cData = starPlant.load_completeness(alignDir)
        
        # Load up information on the coo star for this field
        cooFile = '%s/mag06maylgs1_dp_msc_%s_kp.coo' % (nirc2Dir, nircFieldName)
        cooTmp = open(cooFile).readline().split()
        cooPix = [float(cooTmp[0]), float(cooTmp[1])]
        
        idx = np.where(label.name == cooStarDict[nircFieldName])[0][0]
        cooAsec = [label.x[idx], label.y[idx]]

        scale = 0.00995
        
        # Convert the completeness results to absolute coordinates
        cData.x_in = ((cData.x_in - cooPix[0]) * scale * -1) + cooAsec[0]
        cData.y_in = ((cData.y_in - cooPix[1]) * scale) + cooAsec[1]
        cData.x_out = ((cData.x_out - cooPix[0]) * scale * -1) + cooAsec[0]
        cData.y_out = ((cData.y_out - cooPix[1]) * scale) + cooAsec[1]
        cData.x = ((cData.x - cooPix[0]) * scale * -1) + cooAsec[0]
        cData.y = ((cData.y - cooPix[0]) * scale * -1) + cooAsec[0]

        # Now trim down to just those pixels that are within this
        # OSIRIS field of view
        xypoints = np.column_stack((cData.x_in, cData.y_in))
        inside = nxutils.points_inside_poly(xypoints, xyverts)
        inside = np.where(inside == True)[0]

        # Measure the completeness for this OSIRIS field. Also print it 
        # out into a file.
        outRoot = 'completeness_%s' % fieldName.replace(' ', '')
        completeness = np.zeros(len(cData.mag), dtype=float)
        _comp = open(outRoot + '.dat', 'w')
        for mm in range(len(cData.mag)):
            # All planted stars in this field
            planted = np.where(cData.m_in[inside] == cData.mag[mm])[0]

            # All detected stars in this field
            detected = np.where(cData.f_out[inside][planted] > 0)[0]

            completeness[mm] = float(len(detected)) / float(len(planted))

            _comp.write('%5.2f  %6.3f  %3d  %3d\n' % 
                        (cData.mag[mm], completeness[mm], 
                         len(planted), len(detected)))

        py.clf()
        py.plot(cData.mag, completeness, 'k.-')
        py.xlabel('Kp Magnitude')
        py.ylabel('Completeness')
        py.title('OSIRIS Field %s, NIRC2 Field %s' % (fieldName, nircFieldName))
        py.ylim(0, 1.1)
        py.xlim(9, 20)
        py.savefig(outRoot + '.png')

def klf():
    """
    Make a completeness curve for every osiris spectroscopic field. Use
    the completeness analysis from mag06maylgs1_dp_msc.
    """
    # Load up the spectroscopic database.
    dbfile = '/u/jlu/data/gc/database/stars.sqlite'

    connection = sqlite.connect(dbfile)
    cur = connection.cursor()

    # Select all the OSIRIS fields-of-view
    sql = 'select name, x_vertices, y_vertices from fields where '
    sql += 'name != "E2-1" and name != "E2-3" and '
    sql += 'name != "E3-1" and name != "E3-3" and '
    sql += 'name != "E4-1" and name != "E4-3"'
    cur.execute(sql)
    fields = cur.fetchall()

    # Select all the spectroscopically identified young stars.
    sql = 'select name, kp, x, y, AK_sch from stars where young="T" and '
    sql += '(cuspPaper="T" or edit != "")'
    cur.execute(sql)
    stars = cur.fetchall()

    numStars = len(stars)
    numFields = len(fields)

    # Make numpy arrays for the stars.
    starName = []
    starX = np.zeros(numStars, dtype=float)
    starY = np.zeros(numStars, dtype=float)
    starKp = np.zeros(numStars, dtype=float)
    starAKs = np.zeros(numStars, dtype=float)

    for ss in range(len(stars)):
        starName.append(stars[ss][0])
        starKp[ss] = stars[ss][1]
        starX[ss] = stars[ss][2]
        starY[ss] = stars[ss][3]
        starAKs[ss] = stars[ss][4]
    starName = np.array(starName)

    # Differential Extinction Correction... set all to AKs = 2.7
    # Apply correction in Ks not Kp.
    starKp_ext = np.zeros(numStars, dtype=float)
    synFile = '/u/jlu/work/gc/photometry/synthetic/syn_nir_d8000.0_a680.dat'

    # Will de-redden all stars to AKs = 2.7. Then these can be converted
    # using the ks_2_kp factor (assumes hot star atmospheres)
    theAKs = 2.7
    ks_2_kp = synthetic.get_Kp_Ks(theAKs, 30000.0, filename=synFile)

    for ii in range(numStars):
        kp_2_ks = synthetic.get_Kp_Ks(starAKs[ii], 30000.0, filename=synFile)

        # Switch to Ks, correct differential extinction, switch to Kp
        starKp_ext[ii] = starKp[ii] - kp_2_ks - starAKs[ii] + theAKs + ks_2_kp


    xypoints = np.column_stack((starX, starY))

    # Load up the extinction map from Schodel 2010
    schExtinctFile = '/u/jlu/work/gc/imf/extinction/2010schodel_AKs_fg6.fits'
    schExtinct = pyfits.getdata(schExtinctFile)

    # Load spectroscopic completeness data
    compSpecKp_ext, compSpecDict = load_osiris_spectral_comp()

    # Load the field areas 
    areaDict = load_osiris_field_areas()

    # We will make a luminosity and mass function for every field. We
    # will combine them afterwards (any way we want).
    magBins = np.arange(9, 18)
    massBins = 10**np.arange(1.0, 2.0, 0.1) # equally spaced in log(m)

    sizeKp = len(magBins)
    
    # Output arrays holding luminosity functions
    N = np.zeros((numFields, sizeKp), dtype=float)
    N_ext = np.zeros((numFields, sizeKp), dtype=float)
    KLF = np.zeros((numFields, sizeKp), dtype=float)
    KLF_ext = np.zeros((numFields, sizeKp), dtype=float)
    KLF_ext_cmp1 = np.zeros((numFields, sizeKp), dtype=float)
    KLF_ext_cmp2 = np.zeros((numFields, sizeKp), dtype=float)

    # errors
    eN = np.zeros((numFields, sizeKp), dtype=float)
    eN_ext = np.zeros((numFields, sizeKp), dtype=float)
    eKLF = np.zeros((numFields, sizeKp), dtype=float)
    eKLF_ext = np.zeros((numFields, sizeKp), dtype=float)
    eKLF_ext_cmp1 = np.zeros((numFields, sizeKp), dtype=float)
    eKLF_ext_cmp2 = np.zeros((numFields, sizeKp), dtype=float)

    comp_imag_ext = np.zeros((numFields, sizeKp), dtype=float)
    comp_spec_ext = np.zeros((numFields, sizeKp), dtype=float)
    comp_plant_ext = np.zeros((numFields, sizeKp), dtype=float)
    comp_found_ext = np.zeros((numFields, sizeKp), dtype=float)

    field_names = []

    for ff in range(numFields):
        fieldName = fields[ff][0]
        fieldSuffix = fieldName.replace(' ', '')
        print 'Working on field %s' % fieldName

        field_names.append(fieldSuffix)

        # Find the average extinction for this field
        fieldExtFile = '/u/jlu/work/gc/imf/gcows/extinct_mask_%s.fits' % \
            fieldSuffix
        fieldExtMask = pyfits.getdata(fieldExtFile)

        fieldExtMap = schExtinct * fieldExtMask
        idx = np.where(fieldExtMap != 0)
        fieldExtinctAvg = fieldExtMap[idx[0], idx[1]].mean()
        fieldExtinctStd = fieldExtMap[idx[0], idx[1]].std()

        # Load up the completeness curve for this field
        compFile = '/u/jlu/work/gc/imf/gcows/completeness_%s.dat' % \
            fieldSuffix
        _comp = asciidata.open(compFile)
        compKp = _comp[0].tonumpy()
        compImag = _comp[1].tonumpy()
        compPlant = _comp[2].tonumpy()
        compFound = _comp[3].tonumpy()
        compSpec_ext = compSpecDict[fieldSuffix]


        # Apply an average differential extinction correction to 
        # the completeness curve. Remember, this only corrects the
        # completeness curve to an extinction of AKs=2.7
        # Switch to Ks, correct differential extinction, switch to Kp
        kp_2_ks = synthetic.get_Kp_Ks(fieldExtinctAvg, 30000.0, 
                                      filename=synFile)
        compKp_ext = compKp - kp_2_ks - fieldExtinctAvg + theAKs + ks_2_kp

        # Resample the completeness curves at the magnitude bins of
        # the luminosity function.
        Kp_interp = interpolate.splrep(compKp_ext, compImag, k=1, s=0)
        compImag_ext = interpolate.splev(magBins, Kp_interp)


        Kp_interp = interpolate.splrep(compSpecKp_ext, compSpec_ext, k=1, s=0)
        compSpec_ext = interpolate.splev(magBins, Kp_interp)

        comp_imag_ext[ff] = compImag_ext
        comp_spec_ext[ff] = compSpec_ext
        comp_plant_ext[ff] = np.ones(len(magBins), dtype=int) * compPlant[0]
        Kp_interp = interpolate.splrep(compKp_ext, compFound, k=3, s=0)
        comp_found_ext[ff] = interpolate.splev(magBins, Kp_interp)

        # Find the stars inside this field.
        xverts = np.array([float(ii) for ii in fields[ff][1].split(',')])
        yverts = np.array([float(ii) for ii in fields[ff][2].split(',')])
        xyverts = np.column_stack((xverts, yverts))

        inside = nxutils.points_inside_poly(xypoints, xyverts)
        inside = np.where(inside == True)[0]

        kpInField = starKp[inside]
        kpInField_ext = starKp_ext[inside]

        # Make a binned luminosity function.
        binSizeKp = magBins[1] - magBins[0]
        perAsec2Mag = areaDict[fieldSuffix] * binSizeKp

        # Loop through each magnitude bin and get the surface density of stars.
        for kk in range(sizeKp):
            kp_lo = magBins[kk]
            kp_hi = magBins[kk] + binSizeKp

            # Kp luminosity function (no extinction correction)
            idx = np.where((kpInField >= kp_lo) & (kpInField < kp_hi))[0]
            N[ff, kk] = len(idx)
            eN[ff, kk] = math.sqrt(len(idx))
            KLF[ff, kk] = len(idx) / perAsec2Mag
            eKLF[ff, kk] = math.sqrt(len(idx)) / perAsec2Mag

            # Kp luminosity function (extinction corrected)
            idx = np.where((kpInField_ext >= kp_lo) & 
                           (kpInField_ext < kp_hi))[0]
            N_ext[ff,kk] = len(idx)
            eN_ext[ff, kk] = math.sqrt(len(idx))
            KLF_ext[ff, kk] = len(idx) / perAsec2Mag
            eKLF_ext[ff, kk] = math.sqrt(len(idx)) / perAsec2Mag

            # Correct for spectroscopic completeness
            KLF_ext_cmp1[ff, kk] = len(idx) / compSpec_ext[kk]
            KLF_ext_cmp1[ff,kk] /= perAsec2Mag
            eKLF_ext_cmp1[ff, kk] = math.sqrt(len(idx)) / compSpec_ext[kk]
            eKLF_ext_cmp1[ff, kk] /= perAsec2Mag

            # Correct for imaging completeness
            KLF_ext_cmp2[ff, kk] = KLF_ext_cmp1[ff, kk] / compImag_ext[kk]
            eKLF_ext_cmp2[ff, kk] = eKLF_ext_cmp1[ff, kk] / compImag_ext[kk]

            # Fix some stuff
            idx = np.where(np.isnan(KLF_ext_cmp1[ff]) == True)[0]
            for ii in idx:
                KLF_ext_cmp1[ff,ii] = 0.0

            idx = np.where(np.isnan(KLF_ext_cmp2[ff]) == True)[0]
            for ii in idx:
                KLF_ext_cmp2[ff,ii] = 0.0


    # Save to a pickle file
    pickleFile = '/u/jlu/work/gc/imf/gcows/klf.dat'

    _out = open(pickleFile, 'w')
    
    pickle.dump(field_names, _out)
    pickle.dump(magBins, _out)
    pickle.dump(N, _out)
    pickle.dump(eN, _out)
    pickle.dump(N_ext, _out)
    pickle.dump(eN_ext, _out)
    pickle.dump(KLF, _out)
    pickle.dump(eKLF, _out)
    pickle.dump(KLF_ext, _out)
    pickle.dump(eKLF_ext, _out)
    pickle.dump(KLF_ext_cmp1, _out)
    pickle.dump(eKLF_ext_cmp1, _out)
    pickle.dump(KLF_ext_cmp2, _out)
    pickle.dump(eKLF_ext_cmp2, _out)
    pickle.dump(comp_imag_ext, _out)
    pickle.dump(comp_spec_ext, _out)
    pickle.dump(comp_plant_ext, _out)
    pickle.dump(comp_found_ext, _out)

    _out.close()

def plot_klf():
    workDir = '/u/jlu/work/gc/imf/gcows/'
    
    d = load_klf()
    
    # Sum to get global KLFs
    numFields = d.KLF.shape[1]

    KLF = d.KLF.mean(axis=0)
    eKLF = np.sqrt((d.eKLF**2).sum(axis=0)) / numFields

    KLF_ext = d.KLF_ext.mean(axis=0)
    eKLF_ext = np.sqrt((d.eKLF_ext**2).sum(axis=0)) / numFields

    KLF_ext_cmp1 = d.KLF_ext_cmp1.mean(axis=0)
    eKLF_ext_cmp1 = np.sqrt((d.eKLF_ext_cmp1**2).sum(axis=0)) / numFields

    KLF_ext_cmp2 = d.KLF_ext_cmp2.mean(axis=0)
    eKLF_ext_cmp2 = np.sqrt((d.eKLF_ext_cmp2**2).sum(axis=0)) / numFields

    magBin = d.Kp[1] - d.Kp[0]

    idx1 = np.where(KLF != 0)[0]
    idx2 = np.where(KLF_ext != 0)[0]

    py.clf()
    py.errorbar(d.Kp[idx1], KLF[idx1], fmt='ko', xerr=magBin/2.0, capsize=0)
    py.errorbar(d.Kp[idx1], KLF[idx1], fmt='ko', yerr=eKLF[idx1])
    py.gca().set_yscale('log')
    py.xlabel('Kp magnitude')
    py.ylabel('Stars / (arcsec^2 mag)')
    py.savefig(workDir + 'plots/klf_obs.png')

    py.clf()
    py.errorbar(d.Kp[idx2], KLF_ext[idx2], fmt='ko', xerr=magBin/2.0, capsize=0)
    py.errorbar(d.Kp[idx2], KLF_ext[idx2], fmt='ko', yerr=eKLF_ext[idx2])
    py.gca().set_yscale('log')
    py.xlabel('Kp magnitude')
    py.ylabel('Stars / (arcsec^2 mag)')
    py.savefig(workDir + 'plots/klf_ext.png')

    py.clf()
    py.errorbar(d.Kp[idx2], KLF_ext_cmp1[idx2], 
                fmt='ko', xerr=magBin/2.0, capsize=0)
    py.errorbar(d.Kp[idx2], KLF_ext_cmp1[idx2], 
                fmt='ko', yerr=eKLF_ext_cmp1[idx2])
    py.gca().set_yscale('log')
    py.xlabel('Kp magnitude')
    py.ylabel('Stars / (arcsec^2 mag)')
    py.savefig(workDir + 'plots/klf_ext_cmp1.png')

    py.clf()
    py.errorbar(d.Kp[idx2], KLF_ext_cmp2[idx2], 
                fmt='ko', xerr=magBin/2.0, capsize=0)
    py.errorbar(d.Kp[idx2], KLF_ext_cmp2[idx2], 
                fmt='ko', yerr=eKLF_ext_cmp2[idx2])
    py.gca().set_yscale('log')
    py.xlabel('Kp magnitude')
    py.ylabel('Stars / (arcsec^2 mag)')
    py.savefig(workDir + 'plots/klf_ext_cmp2.png')

def plot_klf2():
    workDir = '/u/jlu/work/gc/imf/gcows/'
    
    d = load_klf()
    
    # Sum to get global KLFs
    numFields = d.KLF.shape[1]

    N = d.N.sum(axis=0)
    eN = np.sqrt(N)

    N_ext = d.N_ext.sum(axis=0)
    eN_ext = np.sqrt(N_ext)

    KLF = d.KLF.mean(axis=0)
    eKLF = np.sqrt((d.eKLF**2).sum(axis=0)) / numFields

    KLF_ext = d.KLF_ext.mean(axis=0)
    eKLF_ext = np.sqrt((d.eKLF_ext**2).sum(axis=0)) / numFields

    # Calculate the completeness curves for all the fields combined.
    # Use the weighted average completeness correction.
    weights = 1.0 / d.comp_plant_ext # plant is the same for all mag bins.
    comp_imag = (d.comp_imag_ext * weights).sum(axis=0) / weights.sum(axis=0)
    comp_spec = (d.comp_spec_ext * weights).sum(axis=0) / weights.sum(axis=0)

    KLF_ext_cmp1 = KLF_ext / comp_spec
    eKLF_ext_cmp1 = eKLF_ext / comp_spec

    KLF_ext_cmp2 = KLF_ext_cmp1 / comp_imag
    eKLF_ext_cmp2 = eKLF_ext_cmp1 / comp_imag

    N_ext_cmp1 = N_ext / comp_spec
    eN_ext_cmp1 = eN_ext / comp_spec

    N_ext_cmp2 = N_ext_cmp1 / comp_imag
    eN_ext_cmp2 = eN_ext_cmp1 / comp_imag

    magBin = d.Kp[1] - d.Kp[0]

    idx1 = np.where(KLF != 0)[0]
    idx2 = np.where(KLF_ext != 0)[0]
    idx3 = np.where(N != 0)[0]
    idx4 = np.where(N_ext != 0)[0]

    py.clf()
    py.errorbar(d.Kp[idx1], KLF[idx1], fmt='ko', xerr=magBin/2.0, capsize=0)
    py.errorbar(d.Kp[idx1], KLF[idx1], fmt='ko', yerr=eKLF[idx1])
    py.gca().set_yscale('log')
    py.xlabel('Kp magnitude')
    py.ylabel('Stars / (arcsec^2 mag)')
    py.savefig(workDir + 'plots/klf2_obs.png')

    py.clf()
    py.errorbar(d.Kp[idx2], KLF_ext[idx2], fmt='ko', xerr=magBin/2.0, capsize=0)
    py.errorbar(d.Kp[idx2], KLF_ext[idx2], fmt='ko', yerr=eKLF_ext[idx2])
    py.gca().set_yscale('log')
    py.xlabel('Kp magnitude')
    py.ylabel('Stars / (arcsec^2 mag)')
    py.savefig(workDir + 'plots/klf2_ext.png')

    py.clf()
    py.errorbar(d.Kp[idx2], KLF_ext_cmp1[idx2], 
                fmt='ko', xerr=magBin/2.0, capsize=0)
    py.errorbar(d.Kp[idx2], KLF_ext_cmp1[idx2], 
                fmt='ko', yerr=eKLF_ext_cmp1[idx2])
    py.gca().set_yscale('log')
    py.xlabel('Kp magnitude')
    py.ylabel('Stars / (arcsec^2 mag)')
    py.savefig(workDir + 'plots/klf2_ext_cmp1.png')

    py.clf()
    py.errorbar(d.Kp[idx2], KLF_ext_cmp2[idx2], 
                fmt='ko', xerr=magBin/2.0, capsize=0)
    py.errorbar(d.Kp[idx2], KLF_ext_cmp2[idx2], 
                fmt='ko', yerr=eKLF_ext_cmp2[idx2])
    py.gca().set_yscale('log')
    py.xlabel('Kp magnitude')
    py.ylabel('Stars / (arcsec^2 mag)')
    py.savefig(workDir + 'plots/klf2_ext_cmp2.png')

    py.clf()
    py.errorbar(d.Kp[idx1], N[idx3], fmt='ko', xerr=magBin/2.0, capsize=0)
    py.errorbar(d.Kp[idx1], N[idx3], fmt='ko', yerr=eN[idx3])
    py.gca().set_yscale('log')
    py.xlabel('Kp magnitude')
    py.ylabel('Stars')
    py.savefig(workDir + 'plots/klf2_count_obs.png')

    py.clf()
    py.errorbar(d.Kp[idx2], N_ext[idx4], fmt='ko', xerr=magBin/2.0, capsize=0)
    py.errorbar(d.Kp[idx2], N_ext[idx4], fmt='ko', yerr=eN_ext[idx4])
    py.gca().set_yscale('log')
    py.xlabel('Kp magnitude')
    py.ylabel('Stars')
    py.savefig(workDir + 'plots/klf2_count_ext.png')

    py.clf()
    py.errorbar(d.Kp[idx4], N_ext_cmp1[idx4], 
                fmt='ko', xerr=magBin/2.0, capsize=0)
    py.errorbar(d.Kp[idx4], N_ext_cmp1[idx4], 
                fmt='ko', yerr=eN_ext_cmp1[idx4])
    py.gca().set_yscale('log')
    py.xlabel('Kp magnitude')
    py.ylabel('Stars')
    py.savefig(workDir + 'plots/klf2_count_ext_cmp1.png')

    py.clf()
    py.errorbar(d.Kp[idx4], N_ext_cmp2[idx4], 
                fmt='ko', xerr=magBin/2.0, capsize=0)
    py.errorbar(d.Kp[idx4], N_ext_cmp2[idx4], 
                fmt='ko', yerr=eN_ext_cmp2[idx4])
    py.gca().set_yscale('log')
    py.xlabel('Kp magnitude')
    py.ylabel('Stars')
    py.savefig(workDir + 'plots/klf2_count_ext_cmp2.png')

    py.clf()
    py.plot(d.Kp, comp_imag, label='Imaging')
    py.plot(d.Kp, comp_spec, label='Spectra')
    py.legend()
    py.xlabel('Kp magnitude')
    py.ylabel('Completeness')
    py.savefig(workDir + 'plots/klf2_completeness.png')

def plot_klf_3radii():
    workDir = '/u/jlu/work/gc/imf/gcows/'
    
    d = load_klf()
    
    idx1 = []
    idx2 = []
    idx3 = []
    for ff in range(len(d.fields)):
        if d.fields[ff] == 'GCCentral':
            idx1.append(ff)
        else:
            if d.fields[ff].startswith('E'):
                idx3.append(ff)
            else:
                idx2.append(ff)

    # Calculate the completeness curves for all the fields combined.
    # Use the weighted average completeness correction.
    weights = 1.0 / d.comp_plant_ext # plant is the same for all mag bins.
    comp_fields = d.comp_imag_ext * d.comp_spec_ext

    comp1 = (comp_fields[idx1] * weights[idx1]).sum(axis=0)
    comp1 /= weights[idx1].sum(axis=0)
    comp2 = (comp_fields[idx2] * weights[idx2]).sum(axis=0)
    comp2 /= weights[idx2].sum(axis=0)
    comp3 = (comp_fields[idx3] * weights[idx3]).sum(axis=0)
    comp3 /= weights[idx3].sum(axis=0)

    N1 = d.N_ext[idx1].sum(axis=0) / comp1
    eN1 = np.sqrt(N1 * comp1) / comp1
    N2 = d.N_ext[idx2].sum(axis=0) / comp2
    eN2 = np.sqrt(N2 * comp2) / comp2
    N3 = d.N_ext[idx3].sum(axis=0) / comp3
    eN3 = np.sqrt(N3 * comp3) / comp3

    KLF1 = d.KLF_ext[idx1].mean(axis=0) / comp1
    eKLF1 = np.sqrt((d.eKLF_ext[idx1]**2).sum(axis=0)) / (len(idx1) * comp1)
    KLF2 = d.KLF_ext[idx2].mean(axis=0) / comp2
    eKLF2 = np.sqrt((d.eKLF_ext[idx2]**2).sum(axis=0)) / (len(idx2) * comp2)
    KLF3 = d.KLF_ext[idx3].mean(axis=0) / comp3
    eKLF3 = np.sqrt((d.eKLF_ext[idx3]**2).sum(axis=0)) / (len(idx3) * comp3)

    magBin = d.Kp[1] - d.Kp[0]

    # Repair for zeros since we are plotting in semi-log-y
    eN1 = np.ma.masked_where(N1 <= 0, eN1)
    eN2 = np.ma.masked_where(N2 <= 0, eN2)
    eN3 = np.ma.masked_where(N3 <= 0, eN3)

    N1 = np.ma.masked_where(N1 <= 0, N1)
    N2 = np.ma.masked_where(N2 <= 0, N2)
    N3 = np.ma.masked_where(N3 <= 0, N3)

    eKLF1 = np.ma.masked_where(KLF1 <= 0, eKLF1)
    eKLF2 = np.ma.masked_where(KLF2 <= 0, eKLF2)
    eKLF3 = np.ma.masked_where(KLF3 <= 0, eKLF3)

    KLF1 = np.ma.masked_where(KLF1 <= 0, KLF1)
    KLF2 = np.ma.masked_where(KLF2 <= 0, KLF2)
    KLF3 = np.ma.masked_where(KLF3 <= 0, KLF3)

    idx = np.where(d.Kp < 16)[0]

    py.clf()
    py.errorbar(d.Kp[idx], N1[idx], fmt='ko-', xerr=magBin/2.0, capsize=0)
    py.errorbar(d.Kp[idx], N1[idx], fmt='ko', yerr=eN1[idx], label='central')
    py.errorbar(d.Kp[idx], N2[idx], fmt='ro-', xerr=magBin/2.0, capsize=0)
    py.errorbar(d.Kp[idx], N2[idx], fmt='ro', yerr=eN2[idx], label='speckle - central')
    py.errorbar(d.Kp[idx], N3[idx], fmt='bo-', xerr=magBin/2.0, capsize=0)
    py.errorbar(d.Kp[idx], N3[idx], fmt='bo', yerr=eN3[idx], label='gcows')
    py.gca().set_yscale('log')
    py.legend(loc='upper left', numpoints=1)
    py.xlabel('Kp magnitude')
    py.ylabel('Stars')
    py.savefig(workDir + 'plots/klf_count_3radii.png')

    py.clf()
    py.errorbar(d.Kp[idx], KLF1[idx], fmt='ko-', xerr=magBin/2.0, capsize=0)
    py.errorbar(d.Kp[idx], KLF1[idx], fmt='ko', yerr=eKLF1[idx], label='central')
    py.errorbar(d.Kp[idx], KLF2[idx], fmt='ro-', xerr=magBin/2.0, capsize=0)
    py.errorbar(d.Kp[idx], KLF2[idx], fmt='ro', yerr=eKLF2[idx], label='speckle - central')
    py.errorbar(d.Kp[idx], KLF3[idx], fmt='bo-', xerr=magBin/2.0, capsize=0)
    py.errorbar(d.Kp[idx], KLF3[idx], fmt='bo', yerr=eKLF3[idx], label='gcows')
    py.gca().set_yscale('log')
    py.legend(loc='upper left', numpoints=1)
    py.xlabel('Kp magnitude')
    py.ylabel('Stars / (arcsec^2 mag)')
    py.savefig(workDir + 'plots/klf_3radii.png')

def plot_klf_2radii():
    workDir = '/u/jlu/work/gc/imf/gcows/'
    
    d = load_klf()
    
    idx1 = []
    idx2 = []
    for ff in range(len(d.fields)):
        if d.fields[ff] == 'GCCentral':
            idx1.append(ff)
        else:
            idx2.append(ff)

    # Calculate the completeness curves for all the fields combined.
    # Use the weighted average completeness correction.
    weights = 1.0 / d.comp_plant_ext # plant is the same for all mag bins.
    comp_fields = d.comp_imag_ext * d.comp_spec_ext

    comp1 = (comp_fields[idx1] * weights[idx1]).sum(axis=0)
    comp1 /= weights[idx1].sum(axis=0)
    comp2 = (comp_fields[idx2] * weights[idx2]).sum(axis=0)
    comp2 /= weights[idx2].sum(axis=0)

    N1 = d.N_ext[idx1].sum(axis=0) / comp1
    eN1 = np.sqrt(N1 * comp1) / comp1
    N2 = d.N_ext[idx2].sum(axis=0) / comp2
    eN2 = np.sqrt(N2 * comp2) / comp2

    KLF1 = d.KLF_ext[idx1].mean(axis=0) / comp1
    eKLF1 = np.sqrt((d.eKLF_ext[idx1]**2).sum(axis=0)) / (len(idx1) * comp1)
    KLF2 = d.KLF_ext[idx2].mean(axis=0) / comp2
    eKLF2 = np.sqrt((d.eKLF_ext[idx2]**2).sum(axis=0)) / (len(idx2) * comp2)

    magBin = d.Kp[1] - d.Kp[0]

    # Repair for zeros since we are plotting in semi-log-y
    eN1 = np.ma.masked_where(N1 <= 0, eN1)
    eN2 = np.ma.masked_where(N2 <= 0, eN2)

    N1 = np.ma.masked_where(N1 <= 0, N1)
    N2 = np.ma.masked_where(N2 <= 0, N2)

    eKLF1 = np.ma.masked_where(KLF1 <= 0, eKLF1)
    eKLF2 = np.ma.masked_where(KLF2 <= 0, eKLF2)

    KLF1 = np.ma.masked_where(KLF1 <= 0, KLF1)
    KLF2 = np.ma.masked_where(KLF2 <= 0, KLF2)

    idx = np.where(d.Kp < 16)[0]

    py.clf()
    py.errorbar(d.Kp[idx], N1[idx], fmt='ro-', xerr=magBin/2.0, capsize=0)
    py.errorbar(d.Kp[idx], N1[idx], fmt='ro', yerr=eN1[idx], label='central')
    py.errorbar(d.Kp[idx], N2[idx], fmt='bo-', xerr=magBin/2.0, capsize=0)
    py.errorbar(d.Kp[idx], N2[idx], fmt='bo', yerr=eN2[idx], label='outer')
    py.gca().set_yscale('log')
    py.legend(loc='upper left', numpoints=1)
    py.xlabel('Kp magnitude')
    py.ylabel('Stars')
    py.savefig(workDir + 'plots/klf_count_2radii.png')

    py.clf()
    py.errorbar(d.Kp[idx], KLF1[idx], fmt='ro-', xerr=magBin/2.0, capsize=0)
    py.errorbar(d.Kp[idx], KLF1[idx], fmt='ro', yerr=eKLF1[idx], label='central')
    py.errorbar(d.Kp[idx], KLF2[idx], fmt='bo-', xerr=magBin/2.0, capsize=0)
    py.errorbar(d.Kp[idx], KLF2[idx], fmt='bo', yerr=eKLF2[idx], label='outer')
    py.gca().set_yscale('log')
    py.legend(loc='upper left', numpoints=1)
    py.xlabel('Kp magnitude')
    py.ylabel('Stars / (arcsec^2 mag)')
    py.savefig(workDir + 'plots/klf_2radii.png')


def plot_klf_vs_bartko():
    workDir = '/u/jlu/work/gc/imf/gcows/'
    
    d = load_klf()
    
    idx1 = []
    idx2 = []
    for ff in range(len(d.fields)):
        if d.fields[ff] == 'GCCentral':
            idx1.append(ff)
        else:
            idx2.append(ff)

    # Calculate the completeness curves for all the fields combined.
    # Use the weighted average completeness correction.
    weights = 1.0 / d.comp_plant_ext # plant is the same for all mag bins.
    comp_fields = d.comp_imag_ext * d.comp_spec_ext

    comp1 = (comp_fields[idx1] * weights[idx1]).sum(axis=0)
    comp1 /= weights[idx1].sum(axis=0)
    comp2 = (comp_fields[idx2] * weights[idx2]).sum(axis=0)
    comp2 /= weights[idx2].sum(axis=0)

    N1 = d.N_ext[idx1].sum(axis=0) / comp1
    eN1 = np.sqrt(N1 * comp1) / comp1
    N2 = d.N_ext[idx2].sum(axis=0) / comp2
    eN2 = np.sqrt(N2 * comp2) / comp2

    KLF1 = d.KLF_ext[idx1].mean(axis=0) / comp1
    eKLF1 = np.sqrt((d.eKLF_ext[idx1]**2).sum(axis=0)) / (len(idx1) * comp1)
    KLF2 = d.KLF_ext[idx2].mean(axis=0) / comp2
    eKLF2 = np.sqrt((d.eKLF_ext[idx2]**2).sum(axis=0)) / (len(idx2) * comp2)

    magBin = d.Kp[1] - d.Kp[0]

    # Repair for zeros since we are plotting in semi-log-y
    eN1 = np.ma.masked_where(N1 <= 0, eN1)
    eN2 = np.ma.masked_where(N2 <= 0, eN2)

    N1 = np.ma.masked_where(N1 <= 0, N1)
    N2 = np.ma.masked_where(N2 <= 0, N2)

    eKLF1 = np.ma.masked_where(KLF1 <= 0, eKLF1)
    eKLF2 = np.ma.masked_where(KLF2 <= 0, eKLF2)

    KLF1 = np.ma.masked_where(KLF1 <= 0, KLF1)
    KLF2 = np.ma.masked_where(KLF2 <= 0, KLF2)

    idx = np.where(d.Kp < 16)[0]

    # Rescale Bartko to match our first two bins
    bartkoKLF = np.array([0.034, 0.08, 0.19, 0.20, 0.14, 0.18, 0.10])
    bartkoKp = np.array( [  9.5, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0])
    bartkoModel = np.array([0.0089, 0.02, 0.07, 0.24, 0.26, 0.4, 0.7])
    bartkoErrX = np.array([1, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5])

    scaleFactor = 1.0
    bartkoKLF = bartkoKLF * scaleFactor

    # Load up a model luminosity function
    modelStars = model_klf()
    scaleFactor = 4.7
    weights = np.ones(len(modelStars), dtype=float)
    weights *= scaleFactor / len(modelStars)

    py.clf()
    py.errorbar(d.Kp[idx], KLF2[idx], fmt='ro', xerr=magBin/2.0, capsize=0)
    py.errorbar(d.Kp[idx], KLF2[idx], fmt='ro', yerr=eKLF2[idx], 
                label='Keck (Lu et al. in prep.)')

    # Model
#     binsKp = np.arange(7.5, 19.5, 1)
#     (n, b, p) = py.hist(modelStars, bins=binsKp, align='mid',
#                         histtype='step', color='red', weights=weights)

    #     py.plot(bartkoKp, bartkoModel, 'r-')
    bartkoErr = np.zeros(len(bartkoKp), dtype=float) + 0.02
    py.errorbar(bartkoKp, bartkoKLF, fmt='bs', yerr=bartkoErr,
                label='VLT (Bartko et al. 2010)')
    py.errorbar(bartkoKp, bartkoKLF, fmt='bs', xerr=bartkoErrX, capsize=0)

    py.gca().set_yscale('log')
    py.legend(loc='upper left', numpoints=1)
    py.ylim(0.02, 2)
    py.xlim(8.5, 16.5)
    py.xlabel('Kp magnitude')
    py.ylabel('Stars / (arcsec^2 mag)')
    py.savefig(workDir + 'plots/klf_vs_bartko.png')


def load_osiris_spectral_comp(extinctCorrect=True):
    workDir = '/u/jlu/work/gc/imf/gcows/'
    # Get the completeness corrections for each field.
    if extinctCorrect == True:
        completenessFile = workDir + 'spec_completeness_extinct_correct.txt'
    else:
        completenessFile = workDir + 'spec_completeness.txt'

    completeness = asciidata.open(completenessFile)

    # Get the field names which are in the header
    fields = completeness.header.hdata[1].split()
    fields = fields[1:]

    # Get the completeness in dictionary by field name
    compKp = completeness[0].tonumpy()
    comp = {}

    # Fix the compKp because the values given are actually
    # the left side of the bin.
    kpBinSize = compKp[1] - compKp[0]

    compKp += kpBinSize / 2.0

    for ff in range(len(fields)):
        field = fields[ff]
        compTmp = completeness[ff+1].tonumpy()

        # We need to interpolate over empty stuff, but only where
        # completness is <= 1.
        # Get the "good" values.
        tmpKp = compKp[compTmp.mask == False]
        tmpComp = compTmp[compTmp.mask == False].data

        # Now find the last bin with completeness = 1 and inlcude 
        # only this bin plus all fainter magnitude bins.
        idx = np.where(tmpComp == 1)[0]
        tmpKp = tmpKp[idx[-1]:]
        tmpComp = tmpComp[idx[-1]:]
        
        c_interp = interpolate.splrep(tmpKp, tmpComp, s=0)
        compInField = interpolate.splev(compKp, c_interp)

        # Flatten to 1 at the bright end
        idx = np.where(compInField >= 1)[0]
        if len(idx) > 0:
            compInField[0:idx[-1]+1] = 1.0

        # Flatten to 0 at the faint end
        idx  = np.where(compInField <= 0)[0]
        if len(idx) > 0:
            compInField[idx[0]:] = 0.0

        comp[field] = compInField

    return compKp, comp

def load_osiris_field_areas():
    # Load up the spectroscopic database to get the field-of-view definitions.
    dbfile = '/u/jlu/data/gc/database/stars.sqlite'

    connection = sqlite.connect(dbfile)
    cur = connection.cursor()
    cur.execute('select name from fields')
    fields = cur.fetchall()
    connection.close()

    # Load up the areas for each field
    masksRoot = '/u/jlu/work/gc/imf/gcows/nirc2_mask_'

    area = {}
    
    for ff in range(len(fields)):
        field = fields[ff][0].replace(' ', '')

        mask = pyfits.getdata('%s%s.fits' % (masksRoot, field))
        area[field] = mask.sum() * 0.01**2

    return area


def load_klf():
    pickleFile = '/u/jlu/work/gc/imf/gcows/klf.dat'
    _in = open(pickleFile, 'r')
    
    d = objects.DataHolder()

    d.fields = pickle.load(_in)

    d.Kp = pickle.load(_in)
    
    d.N = pickle.load(_in)
    d.eN = pickle.load(_in)

    d.N_ext = pickle.load(_in)
    d.eN_ext = pickle.load(_in)

    d.KLF = pickle.load(_in)
    d.eKLF = pickle.load(_in)

    d.KLF_ext = pickle.load(_in)
    d.eKLF_ext = pickle.load(_in)

    d.KLF_ext_cmp1 = pickle.load(_in)
    d.eKLF_ext_cmp1 = pickle.load(_in)

    d.KLF_ext_cmp2 = pickle.load(_in)
    d.eKLF_ext_cmp2 = pickle.load(_in)

    d.comp_imag_ext = pickle.load(_in)
    d.comp_spec_ext = pickle.load(_in)
    d.comp_plant_ext = pickle.load(_in)
    d.comp_found_ext = pickle.load(_in)

    _in.close()

    return d


def plot_young_stars():
    # Load up the spectroscopic database to get the field-of-view definitions.
    dbfile = '/u/jlu/data/gc/database/stars.sqlite'

    connection = sqlite.connect(dbfile)
    cur = connection.cursor()
    cur.execute('select name, x, y, kp from stars where young="T"')
    yngRows = cur.fetchall()

    cur.execute('select name, x, y, kp from stars where old="T"')
    oldRows = cur.fetchall()

    cur.execute('select name, x_vertices, y_vertices from fields')
    regions = cur.fetchall()
    
    fieldNames = []
    xverts = []
    yverts = []
    for rr in range(len(regions)):
        fieldNames.append(regions[rr][0])
        xverts.append(np.array([float(ff) for ff in regions[rr][1].split(',')]))
        yverts.append(np.array([float(ff) for ff in regions[rr][2].split(',')]))

    # Load up the 06maylgs1 mosaic image
    imageRoot = '/u/jlu/work/gc/imf/gcows/'
    imageRoot += 'mag06maylgs1_dp_msc_kp'
    img = pyfits.getdata(imageRoot + '.fits')
    
    # Coo star is IRS 16C
    coords = open(imageRoot + '.coo').readline().split()
    cooPix = [float(coords[0]), float(coords[1])]

    # Load up the label.dat file to get the coordinates in arcsec
    label = starTables.Labels()
    idx = np.where(label.name == 'irs16C')[0]
    cooAsec = [label.x[idx[0]], label.y[idx[0]]]

    scale = 0.00995

    xmin = ((0 - cooPix[0]) * scale * -1.0) + cooAsec[0]
    xmax = ((img.shape[1] - cooPix[0]) * scale * -1.0) + cooAsec[0]
    ymin = ((0 - cooPix[1]) * scale) + cooAsec[1]
    ymax = ((img.shape[0] - cooPix[1]) * scale) + cooAsec[1]
    extent = [xmin, xmax, ymin, ymax]

    # Loop through the stars and make arrays
    numYng = len(yngRows)
    numOld = len(oldRows)

    yngName = np.zeros(numYng, dtype='S13')
    yngX = np.zeros(numYng, dtype=float)
    yngY = np.zeros(numYng, dtype=float)
    yngKp = np.zeros(numYng, dtype=float)

    oldName = np.zeros(numOld, dtype='S13')
    oldX = np.zeros(numOld, dtype=float)
    oldY = np.zeros(numOld, dtype=float)
    oldKp = np.zeros(numOld, dtype=float)

    for yy in range(len(yngRows)):
        yngName[yy] = yngRows[yy][0]
        yngX[yy] = yngRows[yy][1]
        yngY[yy] = yngRows[yy][2]
        yngKp[yy] = yngRows[yy][3]

    for oo in range(len(oldRows)):
        oldName[oo] = oldRows[oo][0]
        oldX[oo] = oldRows[oo][1]
        oldY[oo] = oldRows[oo][2]
        oldKp[oo] = oldRows[oo][3]

    print extent
    py.clf()
    py.imshow(img, extent=extent, cmap=py.cm.gray_r, vmin=0, vmax=10000)
    py.plot(yngX, yngY, 'bo', markerfacecolor='none', markersize=10, 
            markeredgecolor='blue', markeredgewidth=2)
    py.plot(oldX, oldY, 'ro', markerfacecolor='none', markersize=10,
            markeredgecolor='red', markeredgewidth=2)
    for yy in range(numYng):
        txt = '%s, Kp=%5.2f' % (yngName[yy], yngKp[yy])
#         txt = '%5.2f' % (yngKp[yy])
        py.text(yngX[yy], yngY[yy], txt, color='blue')
    py.axis('tight')
    py.axis('equal')

    for ff in range(len(fieldNames)):
        py.plot(np.append(xverts[ff], xverts[ff][0]), 
                np.append(yverts[ff], yverts[ff][0]), 
                color='black')



def model_klf():
    # Read in Geneva tracks
    genevaFile = '/u/jlu/work/models/geneva/iso/020/c/'
    genevaFile += 'iso_c020_0675.UBVRIJHKLM'
    model = asciidata.open(genevaFile)
    modMass = model[1].tonumpy()
    modV = model[6].tonumpy()
    modVK = model[11].tonumpy()
    modHK = model[15].tonumpy()
    modJLp = model[19].tonumpy()
    modJK = model[17].tonumpy()

#     genevaFile2 = '/u/jlu/work/models/geneva/iso/020/c/'
#     genevaFile2 += 'iso_c020_068.UBVRIJHKLM'
#     model = asciidata.open(genevaFile)
#     modMass = model[1].tonumpy()
#     modV = model[6].tonumpy()
#     modVK = model[11].tonumpy()
#     modHK = model[15].tonumpy()
#     modJLp = model[19].tonumpy()
#     modJK = model[17].tonumpy()

    

    # Reddening
    aV = 27.0
    RV = 2.9

#     # cardelli() returns A_L
#     aJ = aV * extinction.cardelli(1.248, RV)
#     aH = aV * extinction.cardelli(1.6330, RV)
#     aKp = aV * extinction.cardelli(2.1245, RV)
#     aK = aV * extinction.cardelli(2.196, RV)
#     aKs = aV * extinction.cardelli(2.146, RV)


    aKs = 2.7
    aJ = extinction.nishiyama09(1.248, aKs)
    aH = extinction.nishiyama09(1.6330, aKs)
    aKp = extinction.nishiyama09(2.1245, aKs)
    aK = extinction.nishiyama09(2.196, aKs)
    aKs = extinction.nishiyama09(2.146, aKs)

    modK = modV - modVK
    modH = modK + modHK
    modJ = modK + modJK
    modLp = modJ - modJLp
    modKs = modK + 0.002 + 0.026 * (modJK)
    modKp = modK + 0.22 * (modHK)

    dist = 8400.0
    distMod = -5.0 + 5.0 * math.log10(dist)

    modK_extinct = modK + aK + distMod
    modKp_extinct = modKp + aKp + distMod
    modKs_extinct = modKs + aKs + distMod

    return modKp_extinct
