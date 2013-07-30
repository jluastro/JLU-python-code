import atpy
import numpy as np
import pylab as py
import pdb
import pycurl
import re

def reformat_Johnson2013():
    dir = '/u/jlu/work/ellipIMF/galcen/'

    f_name = dir + 'Johnson2013.txt'

    t = atpy.Table(f_name, type='ascii')
    t.rename_column('col1', 'name')
    t.rename_column('col2', 'E_BV')
    t.rename_column('col3', 'J')
    t.rename_column('col4', 'H')
    t.rename_column('col5', 'Ks')
    t.rename_column('col6', 'Teff')
    t.rename_column('col7', 'logg')
    t.rename_column('col8', '[Fe/H]_1')
    t.rename_column('col9', 'V_t')
    t.rename_column('col10', 'RV')
    t.rename_column('col11', '[Fe/H]')
    t.rename_column('col12', '[O/Fe]')
    t.rename_column('col13', '[Si/Fe]')
    t.rename_column('col14', '[Ca/Fe]')

    ra = np.zeros(len(t), dtype=float)
    dec = np.zeros(len(t), dtype=float)

    for ii in range(len(t)):
        radec = t.name[ii].split('-')

        ra_hr = float(radec[0][0:2])
        ra_min = float(radec[0][2:4])
        ra_sec = float(radec[0][4:6] + '.' + radec[0][6:-1])
        ra[ii] = ra_hr + (ra_min / 60.0) + (ra_sec / 3600.0)
        ra[ii] *= 15.0

        dec_deg = float(radec[1][0:2])
        dec_min = float(radec[1][2:4])
        dec_sec = float(radec[1][4:6])
        dec[ii] = dec_deg + (dec_min / 60.0) + (dec_sec / 3600.0)
        dec[ii] *= -1.0

    t.add_column('RA', ra)
    t.add_column('DEC', dec)
        
    t.table_name = ''
    t.write(f_name.replace('.txt', '.fits'), overwrite=True)
    

def create_galcen_list():
    dir = '/u/jlu/work/ellipIMF/galcen/'

    f1_name = 'Zoccali2008.fits'
    f2_name = 'Gonzalez2011.fits'
    f3_name = 'Hill2011.fits'     # Baade's Window
    f4_name = 'Johnson2011.fits'
    f5_name = 'Johnson2013.fits'

    t1 = atpy.Table(f1_name)
    t2 = atpy.Table(f2_name)
    t3 = atpy.Table(f3_name)
    t4 = atpy.Table(f4_name)
    t5 = atpy.Table(f5_name)

    name = t1.OGLE
    ra = t1.RAJ2000
    dec = t1.DEJ2000
    teff = t1.Teff
    vmag = t1.Vmag
    ref = np.zeros(len(t1)) + 1

    fe_h = np.zeros(len(ra), dtype=float)
    mg_fe = np.zeros(len(ra), dtype=float)
    ca_fe = np.zeros(len(ra), dtype=float)
    ti_fe = np.zeros(len(ra), dtype=float)
    si_fe = np.zeros(len(ra), dtype=float)

    fnd = 0
    for ii in range(len(ra)):
        for jj in range(len(t2)):
            if t2.Name[jj].startswith(name[ii]):
                fe_h[ii] = t2.__Fe_H_[jj]
                mg_fe[ii] = t2.__Mg_Fe_[jj]
                ca_fe[ii] = t2.__Ca_Fe_[jj]
                ti_fe[ii] = t2.__Ti_Fe_[jj]
                si_fe[ii] = t2.__Si_Fe_[jj]
                fnd += 1
                continue

    print 'found ', fnd
    print len(fe_h), len(mg_fe), len(name)

    # Concatenate Hill 2011 table
    name = np.append(name, [str(ogle) for ogle in t3.OGLE])
    ra = np.append(ra, t3._RA)
    dec = np.append(dec, t3._DE)
    teff = np.append(teff, t3.Teff)
    vmag = np.append(vmag, t3.Vmag)
    ref = np.append(ref, np.zeros(len(t3)) + 3)
    fe_h = np.append(fe_h, t3.__Fe_H_)
    ca_fe = np.append(ca_fe, np.zeros(len(t3)))
    ti_fe = np.append(ti_fe, np.zeros(len(t3)))
    si_fe = np.append(si_fe, np.zeros(len(t3)))

    mg_fe_tmp = np.log10( 10 ** t3.__Mg_H_ / 10 ** t3.__Fe_H_ )
    mg_fe = np.append(mg_fe, mg_fe_tmp)

    print len(fe_h), len(mg_fe), len(name)

    # Append Johnson 2011
    name = np.append(name, t4._2MASS)
    ra = np.append(ra, t4._RA)
    dec = np.append(dec, t4._DE)
    teff = np.append(teff, t4.Teff)
    vmag = np.append(vmag, t4.Vmag)
    ref = np.append(ref, np.zeros(len(t4)) + 4)
    fe_h = np.append(fe_h, t4.__Fe_H_)
    ca_fe = np.append(ca_fe, t4.__Ca_Fe_)
    ti_fe = np.append(ti_fe, np.zeros(len(t4)))
    si_fe = np.append(si_fe, t4.__Si_Fe_)
    mg_fe = np.append(mg_fe, np.zeros(len(t4)))
    

    print len(fe_h), len(mg_fe), len(name)

    # Append Johnson 2013
    name = np.append(name, t5.name)
    ra = np.append(ra, t5.RA)
    dec = np.append(dec, t5.DEC)
    teff = np.append(teff, t5.Teff)
    vmag = np.append(vmag, np.zeros(len(t5)))
    fe_h = np.append(fe_h, t5['[Fe/H]'])
    ref = np.append(ref, np.zeros(len(t5)) + 5)
    ca_fe = np.append(ca_fe, t5['[Ca/Fe]'])
    ti_fe = np.append(ti_fe, np.zeros(len(t5)))
    si_fe = np.append(si_fe, t5['[Si/Fe]'])
    mg_fe = np.append(mg_fe, np.zeros(len(t5)))
                     

    print len(fe_h), len(mg_fe), len(name)

    t = atpy.Table()
    t.add_column('name', name)
    t.add_column('RA', ra)
    t.add_column('DEC', dec)
    t.add_column('Teff', teff)
    t.add_column('Vmag', vmag)
    t.add_column('ref', ref)
    t.add_column('fe_h', fe_h)
    t.add_column('ca_fe', ca_fe)
    t.add_column('ti_fe', ti_fe)
    t.add_column('si_fe', si_fe)
    t.add_column('mg_fe', mg_fe)
    t.table_name = ''
    t.write(dir + 'all.fits', overwrite=True)



def select_stars():
    dir = '/u/jlu/work/ellipIMF/galcen/'

    t = atpy.Table(dir + 'all.fits')
    
    # Trim down to field 1 stars.
    t = t.where((t.si_fe != 0) & (t.RA > 268) & (t.RA < 272) & (t.DEC > -31) & (t.DEC < -29))
    print len(t)

    # # Trim down to just those stars between -0.4 < [Fe/H] < 0.4
    t = t.where((t.fe_h > -0.4))
    print len(t)

    # Divide up into decrements of 0.1 in [Fe/H]. Calculate mean and dispersion
    # for each bin.
    feh_bin_size = 0.1
    feh_bins = np.arange(-0.35, 0.41, feh_bin_size)
    sife_avg = np.zeros(len(feh_bins), dtype=float)
    sife_std = np.zeros(len(feh_bins), dtype=float)

    alphaEnhance = np.zeros(len(t), dtype=float)

    for ii in range(len(feh_bins)):
        lo = feh_bins[ii] - (feh_bin_size / 2.0)
        hi = feh_bins[ii] + (feh_bin_size / 2.0)
        idx = np.where((t.fe_h > lo) & (t.fe_h <= hi))[0]

        sife_avg[ii] = t.si_fe[idx].mean()
        sife_std[ii] = t.si_fe[idx].std()

        alphaEnhance[idx] = (t.si_fe[idx] - sife_avg[ii]) / sife_std[ii]


    # Assign weights according to relative Si/Fe abundance
    deimosWeight = (alphaEnhance * 100) + 220

    py.clf()
    py.scatter(t.fe_h, t.si_fe, c=deimosWeight, s=100)
    py.plot(feh_bins, sife_avg, 'k--')
    py.colorbar()
    py.xlabel('[Fe/H]')
    py.ylabel('[Si/Fe]')
    py.title('Deimos Weights')
    py.savefig('plot_feh_sife.png')

    py.clf()
    py.scatter(t.fe_h, t.mg_fe, c=deimosWeight, s=100)
    py.colorbar()
    py.xlabel('[Fe/H]')
    py.ylabel('[Mg/Fe]')
    py.title('Deimos Weights')
    py.savefig('plot_feh_mgfe.png')

    py.clf()
    py.scatter(t.fe_h, t.si_fe, c=t.mg_fe, s=100)
    py.colorbar()
    py.xlabel('[Fe/H]')
    py.ylabel('[Si/Fe]')
    py.title('[Mg/Fe]')
    py.savefig('plot_feh_sife_color_mgfe.png')

    py.clf()
    py.scatter(t.RA, t.DEC, c=deimosWeight, s=100)
    py.colorbar()
    dxcen = 271.0
    dycen = -30.01
    corners_x = np.array([-1.8, -1.8, 1.8, 1.8, -1.8])
    corners_y = np.array([-7.5, 7.5, 7.5, -7.5, -7.5])

    corners_x = dxcen + (corners_x / 60.0)
    corners_y = dycen + (corners_y / 60.0)
    py.plot(corners_x, corners_y, 'b-')
    py.axis('equal')
    py.xlabel('RA (deg)')
    py.ylabel('DEC (deg)')
    py.title('Deimos Weight')
    py.savefig('plot_ra_dec.png')
    
    py.clf()
    py.scatter(t.fe_h, alphaEnhance, c=deimosWeight, s=100)
    py.colorbar()
    py.xlabel('[Fe/H]')
    py.ylabel('alpha abundance')
    py.title('Deimos Weights')
    py.savefig('plot_feh_alpha.png')


    py.clf()
    py.plot(t.fe_h, deimosWeight, 'k.', ms=2)
    py.xlabel('[Fe/H]')
    py.ylabel('Deimos Weight')
    py.savefig('plot_feh_weight.png')

    _out = open(dir + 'deimos_galcen.txt', 'w')
    
    name = t.name
    for ii in range(len(name)):
        _out.write('{0:10s}  {1:10.6f}  {2:10.6f}  2000  {3:5.2f}  V  {4:4d}\n'.format(t.name[ii], t.RA[ii], t.DEC[ii], t.Vmag[ii], int(deimosWeight[ii])))

    _out.close()

    t.table_name = ''
    t.write('galcen_observe.fits', overwrite=True)

def get_simbad_info():
    """
    Find the closest star in simbad to the specified location.
    Update the coordinates and fetch the I and J magnitude for
    the object.
    """

    t = atpy.Table('galcen_observe.fits')

    ra = t.RA
    dec = t.DEC

    # Write a query file to send to SIMBAD
    print 'Writing query'
    queryFile = 'simbad_query.txt'
    _query = open(queryFile, 'w')
    _query.write('output console=off\n')
    _query.write('output script=off\n')
    _query.write('output error=off\n')
    _query.write('set limit 1\n')
    _query.write('format object fmt1 "%30IDLIST(1), %5.3DIST, %13COO(A), %13COO(D), %13COO(d;A), %13COO(d;D), %5FLUXLIST(V,I,J; N=F, )"\n')
    _query.write('result full\n')

    for ii in range(len(ra)):
        _query.write('query coo {0:10.6f} {1:9.5f} radius=5s frame=FK5 epoch=J2000\n'.format(ra[ii], dec[ii]))

    _query.close()

    # Connect to simbad and submit the query. Save to a buffer.
    print 'Sending query'
    replyFile = 'simbad_results.txt'
    _reply = open(replyFile, 'w')
    curl = pycurl.Curl()
    curl.setopt(pycurl.POST, 1)
    curl.setopt(pycurl.URL, 'http://simbad.harvard.edu/simbad/sim-script')
    curl.setopt(pycurl.WRITEFUNCTION, _reply.write)

    curlform = [("scriptFile", (pycurl.FORM_FILE, queryFile))]
    curl.setopt(pycurl.HTTPPOST, curlform)

    curl.perform()
    _reply.close()

def process_simbad_reply():
    t = atpy.Table('galcen_observe.fits')

    ra = t.RA
    dec = t.DEC

    replyFile = 'simbad_results.txt'

    # Search reply for flux info.
    print 'Reading reply'
    _reply = open(replyFile)
    foo = _reply.readlines()

    # Find the start of the data
    startData = False
    startError = False
    notFound = []

    data_ii = 0

    nameNew = np.zeros(len(ra), dtype='S20')
    distNew = np.zeros(len(ra), dtype=float)
    raNewHex = np.zeros(len(ra), dtype='S12')
    decNewHex = np.zeros(len(ra), dtype='S12')
    raNew = np.zeros(len(ra), dtype=float)
    decNew = np.zeros(len(ra), dtype=float)
    vmagNew = np.zeros(len(ra), dtype=float)
    imagNew = np.zeros(len(ra), dtype=float)
    jmagNew = np.zeros(len(ra), dtype=float)
    
    for ff in range(len(foo)):
        if (foo[ff].startswith('::error')):
            startError = True
            continue

        if startError:
            if foo[ff].startswith('['):
                tmp = foo[ff][1:5].split(']')
                badIdx = int(tmp[0]) - 7  # There are 8 entry lines in the query file
                notFound.append(badIdx)
                print 'Star {0} not found'.format(badIdx)

        if foo[ff].startswith('::data'):
            startData = True
            startError = False
            continue


        if startData:
            if foo[ff].strip() != '':
                while data_ii in notFound:
                    print 'Star {0} being skipped (not found)'.format(data_ii)
                    data_ii += 1
                    
                tmp = foo[ff].split(',')

                nameNew[data_ii] = re.sub('\s+', '_', tmp[0])
                distNew[data_ii] = float(tmp[1].strip())

                if distNew[data_ii] > 1.0:
                    print 'Star {0} skipped (too far)'.format(data_ii)
                    continue
                
                print 'Star {0} found and loaded'.format(data_ii)
                raNewHex[data_ii] = tmp[2].strip()
                decNewHex[data_ii] = tmp[3].strip()
                raNew[data_ii] = float(tmp[4].strip())
                decNew[data_ii] = float(tmp[5].strip())

                for kk in range(6, len(tmp)-1):
                    filtTmp = tmp[kk].split("=")
                    band = filtTmp[0].strip()
                    mag = float(filtTmp[1])

                    if band == 'V':
                        vmagNew[data_ii] = mag
                    if band == 'I':
                        imagNew[data_ii] = mag
                    if band == 'J':
                        jmagNew[data_ii] = mag

                data_ii += 1

    for ii in range(len(t)):
        fmtStr = 'OGLE_{name} {ii:3d}  {ra1:10.6f} {ra2:10.6f}   '
        fmtStr += '{dec1:10.6f} {dec2:10.6f}   {v1:5.2f} {v2:5.2f}'
        print fmtStr.format(name=t.name[ii], ii=ii, ra1=t.RA[ii], ra2=raNew[ii],
                            dec1=t.DEC[ii], dec2=decNew[ii],
                            v1=t.Vmag[ii], v2=vmagNew[ii])
        
        if raNew[ii] > 0:
            dRA = t.RA[ii] - raNew[ii]
            dDEC = t.DEC[ii] - decNew[ii]
            dV = t.Vmag[ii] - vmagNew[ii]

            if np.abs(dRA) > 0.0003 or np.abs(dDEC) > 0.0003:
                print 'Problem with star {0}, dRA = {1}, dDEC = {2}'.format(ii, dRA, dDEC)
            else:
                t.RA[ii] = raNew[ii]
                t.DEC[ii] = decNew[ii]

            if np.abs(dV) > 0.3:
                print 'Problem with star {0} photometry, dV = {1}'.format(ii, dV)
            else:
                t.Vmag[ii] = vmagNew[ii]

    t.table_name = ''
    t.write('galcen_observe_simbad.fits', overwrite=True)
                

            
            
        
