import numpy as np
import pylab as py
import asciidata
import pyfits
import os
import math
import urllib
import pdb
import glob
from gcwork import objects
from matplotlib import mlab
from scipy import interpolate
import time
from gcreduce import gcutil
    
def get_geneva_isochrone(metallicity=0.02, logAge=8.0):
    """
    Metallicity in Z (def = solor of 0.02).
    """
    rootDir = '/u/jlu/work/models/geneva/iso/'
    
    metalPart = str(int(metallicity * 1000)).zfill(3)
    agePart = str(int(logAge * 100)).zfill(4)

    genevaFile = rootDir + metalPart + '/c/iso_c' + metalPart + '_' + \
        agePart + '.UBVRIJHKLM'

    
    if not os.path.exists(genevaFile):
        print 'Geneva isochrone file does not exist:'
        print '  ' + genevaFile
        return None

    table = asciidata.open(genevaFile)

    mass = table[1].tonumpy()
    logT = table[3].tonumpy()
    logg = table[4].tonumpy()
    logL = table[5].tonumpy()

    obj = objects.DataHolder()
    obj.mass = mass
    obj.logT = logT
    obj.logg = logg
    obj.logL = logL

    return obj

def get_meynetmaeder_isochrone(logAge, metallicity=0.02, rotation=True):
    """
    Load mass, effective temperature, log gravity, and log luminosity
    for the Meynet & Maeder Geneva evolutionary tracks (2003 for solar,
    2005 for other metallicities).

    Inputs:
    logAge - Logarithmic Age
    metallicity - in Z (def = solar of 0.02)
    """
    inputAge = 10**logAge # years

    rootDir = '/u/jlu/work/models/geneva/meynetMaeder/'

    metalPart = 'z' + str(int(metallicity * 1000)).zfill(2)
    rotPart = 'S0'
    if rotation:
        rotPart = 'S3'

    # First lets use isochrones if we have them instead of reconstructing
    # our own thing.
    isoFile = rootDir + 'iso_' + metalPart + rotPart + '/'
    isoFile += 'iso_%.2f.dat' % logAge
    if not os.path.exists(isoFile):
        # Make an isochrone file.
        print 'Generating new isochrone!!'
        make_isochrone_meynetmaeder(logAge, 
                                    metallicity=metallicity, 
                                    rotation=rotation)
        
    data = asciidata.open(isoFile)
    mass = data[1].tonumpy()
    logT = data[2].tonumpy()
    logg = data[7].tonumpy()
    logL = data[8].tonumpy()
    logT_WR = data[10].tonumpy()
    
    obj = objects.DataHolder()
    obj.mass = mass
    obj.logT = logT
    obj.logg = logg
    obj.logL = logL
    obj.logT_WR = logT_WR
    
    return obj

def make_isochrone_meynetmaeder(logAge, metallicity=0.02, rotation=True):
    rootDir = '/u/jlu/work/models/geneva/meynetMaeder/'

    metalPart = 'z' + str(int(metallicity * 1000)).zfill(2)
    rotPart = 'S0'
    if rotation:
        rotPart = 'S3'
    
    rootDir += 'iso_' + metalPart + rotPart + '/'
    isoFile = 'iso_%.2f.dat' % logAge 

    # We actually need to work in the root directory since that is 
    # what the Fortran code requires.
    origDir = os.getcwd()
    os.chdir(rootDir)

    f = open(rootDir + 'age.dat', 'w')
    f.write('%.2f\n' % logAge)
    f.close()

    os.system('iso')
    os.system('mv -f result %s' % isoFile)

    # Change back to the original directory
    os.chdir(origDir)



def fetch_meynet_maeder_2003():
    url1 = 'http://obswww.unige.ch/people/sylvia.ekstrom/evol/tables_WR/'
    _url = urllib.urlopen(url1)
    lines = _url.readlines()
    _url.close()

    # This is what we are going to search on
    href = 'href="'

    # Build up a list of files
    for line in lines:
        if ('Parent Directory' in line or
            'Last modified' in line):
            continue
        try:
            idx = line.index(href)
            line = line[idx+len(href):]

            idx = line.index('"')
            file = line[:idx]

            _file = urllib.urlopen(url1 + file)
            contents = _file.read()
            _file.close()
    
            print 'Saving %s' % file

            root = '/u/jlu/work/models/geneva/meynetMaeder2003/'
            outfile = open(root + file, 'w')
            outfile.write(contents)
            outfile.close()

        except ValueError:
            continue


def fetch_meynet_maeder_2005():
    url1 = 'http://obswww.unige.ch/people/sylvia.ekstrom/evol/tables_WR_nosolar/'
    _url = urllib.urlopen(url1)
    lines = _url.readlines()
    _url.close()

    # This is what we are going to search on
    href = 'href="'

    # Build up a list of files
    for line in lines:
        if ('Parent Directory' in line or
            'Last modified' in line):
            continue
        try:
            idx = line.index(href)
            line = line[idx+len(href):]

            idx = line.index('"')
            file = line[:idx]

            _file = urllib.urlopen(url1 + file)
            contents = _file.read()
            _file.close()
    
            print 'Saving %s' % file

            root = '/u/jlu/work/models/geneva/meynetMaeder2005/'
            outfile = open(root + file, 'w')
            outfile.write(contents)
            outfile.close()

        except ValueError:
            continue

    
def get_palla_stahler_isochrone(logAge):
    pms_isochrones = '/u/jlu/work/models/preMS/pallaStahler1999/' + \
        'pms_isochrones.txt'

    data = asciidata.open(pms_isochrones)

    mass = data[0].tonumpy()
    temp = data[1].tonumpy()
    lum = data[2].tonumpy()
    age = data[3].tonumpy()

    # Interpolate to get the isochrone at the proper age.
    inputAge = 10**logAge / 10**6

    # For each mass bin, find the closest ages on either side.
    umass = np.unique(mass)

def get_padova_isochrone(logAge, metallicity=0.02):
    mod_dir = '/u/jlu/work/models/padova/'

    metSuffix = 'z' + str(metallicity).split('.')[-1]

    if not os.path.exists(mod_dir + metSuffix):
        print 'Failed to find Padova models for metallicity = ' + metSuffix
        
    mod_dir += metSuffix + '/'
        
    # First lets use isochrones if we have them instead of reconstructing
    # our own thing.
    isoFile = '%siso_%.2f.dat' % (mod_dir, logAge)

    data = asciidata.open(isoFile)
    mass = data[1].tonumpy()
    logL = data[3].tonumpy()
    logT = data[4].tonumpy()
    logg = data[5].tonumpy()

    obj = objects.DataHolder()
    obj.mass = mass
    obj.logT = logT
    obj.logg = logg
    obj.logL = logL
    
    return obj

def get_siess_isochrone(logAge, metallicity=0.02):
    pms_dir = '/u/jlu/work/models/preMS/siess2000/'

    metSuffix = 'z' + str(metallicity).split('.')[-1]

    if not os.path.exists(pms_dir + metSuffix):
        print 'Failed to find Siess PMS models for metallicity = ' + metSuffix
        
    pms_dir += metSuffix + '/'
        
    # First lets use isochrones if we have them instead of reconstructing
    # our own thing.
    isoFile = pms_dir + 'iso/'
    isoFile += 'iso_%.2f.dat' % logAge
    if not os.path.exists(isoFile):
        # Make an isochrone file.
        print 'Generating new isochrone!!'
        make_isochrone_siess(logAge, metallicity=metallicity)

    data = asciidata.open(isoFile)
    mass = data[0].tonumpy()
    logT = data[1].tonumpy()
    logL = data[2].tonumpy()
    logg = data[3].tonumpy()

    obj = objects.DataHolder()
    obj.mass = mass
    obj.logT = logT
    obj.logg = logg
    obj.logL = logL
    
    return obj

def make_isochrone_siess(log_age, metallicity=0.02, 
                         tracks=None, test=False):
    """
    Read in a set of evolutionary tracks and generate an isochrone
    that is well sampled at the full range of masses. This code
    is a re-written version of iso.f provided by the Geneva models group.
    """
    age = 10**log_age

    rootDir = '/u/jlu/work/models/preMS/siess2000/'
    metSuffix = 'z' + str(metallicity).split('.')[-1]
    rootDir += metSuffix + '/'

    isoFile = rootDir + 'iso/iso_%.2f.dat' % log_age
    outSuffix = '_%.2f' % (log_age)

    if not os.path.exists(rootDir):
        print 'Failed to find Siess PMS models for metallicity = ' + metSuffix

    print '*** Generating SIESS isochrone for log t = %.2f and Z = %.2f' % \
        (log_age, metallicity)

    if tracks is None:
        print time.asctime(), 'Getting tracks.'
        t = get_siess_tracks()
    else:
        t = tracks

    numStars = len(t.masses)

    # Work with log masses
    log_M = np.log10( t.masses )

    # First thing we need to do is determine the highest
    # masses at this age. Do this by generating a curve of 
    # highest mass vs. time and pull out the new highest mass at the
    # specified age. This is only necessary if the age we are interested
    # in is larger than the lifetimes of the stars in this model.
    print time.asctime(), 'Getting highest mass.'
    log_M_hi = highest_mass_at_age(log_age, t, test=test)
    log_M_lo = log_M.min()
    M_lo = 10**log_M_lo
    M_hi = 10**log_M_hi
    
    # First we construct isochrones composed of only points
    # from the masses given in the tracks. Values along the 
    # track for each star are interpolated linearly with age.
    print time.asctime(), 'Interpolate tracks to different age.'
    iso = Isochrone(log_age)
    iso.M_init = t.masses
    iso.log_Teff = np.zeros(numStars, dtype=float)
    iso.log_L = np.zeros(numStars, dtype=float)
    iso.log_g = np.zeros(numStars, dtype=float)
    iso.Reff = np.zeros(numStars, dtype=float)
    iso.M = np.zeros(numStars, dtype=float)

    if test:
        py.clf()

    for ss in range(numStars):
        track = t.tracks[ss]
        age_on_track = 10**track.log_age

        if age <= age_on_track[-1]:
            f_log_Teff = interpolate.interp1d(age_on_track, track.log_Teff)
            f_log_L = interpolate.interp1d(age_on_track, track.log_L)
            f_log_g = interpolate.interp1d(age_on_track, track.log_g)
            f_Reff = interpolate.interp1d(age_on_track, track.Reff)
            f_M = interpolate.interp1d(age_on_track, track.M)

            iso.log_Teff[ss] = f_log_Teff(age)
            iso.log_L[ss] = f_log_L(age)
            iso.log_g[ss] = f_log_g(age)
            iso.Reff[ss] = f_Reff(age)
            iso.M[ss] = f_M(age)
        else:
            # We have hit the end of the road. Add the last data point
            # for this too-massive star just to prevent the next round
            # of interpolations from failing.
            print 'Adding too-massive track at %.2f' % t.masses[ss]
            iso.log_Teff[ss] = track.log_Teff[-1]
            iso.log_L[ss] = track.log_L[-1]
            iso.log_g[ss] = track.log_g[-1]
            iso.Reff[ss] = track.Reff[-1]
            iso.M[ss] = track.M[-1]
            
            break

        if test:
            py.plot(track.log_Teff, track.log_L, 'b-')
            py.plot([iso.log_Teff[ss]], [iso.log_L[ss]], 'ro')

    if test:
        rng = py.axis()
        py.xlim(rng[1], rng[0])
        py.xlabel('log Teff')
        py.ylabel('log L')
        py.title('Siess+ 2000 Tracks (red at log t = %.2f)' % log_age)
        py.savefig(outDir + 'plots/hr_tracks_at' + outSuffix + '.png')

    # clean up too-high masses
    idx = np.where(iso.log_Teff != 0)[0]
    log_M = log_M[idx]
    iso.log_Teff = iso.log_Teff[idx]
    iso.log_L = iso.log_L[idx]
    iso.log_g = iso.log_g[idx]
    iso.Reff = iso.Reff[idx]
    iso.M = iso.M[idx]

    # Now we take these isochrones and interpolate to a more finely
    # sampled mass distribution. Values are interpolated linearly in
    # log M.
    print time.asctime(), 'Interpolate across tracks to different masses.'
    f_log_Teff = interpolate.interp1d(log_M, iso.log_Teff, kind='linear')
    f_log_L = interpolate.interp1d(log_M, iso.log_L, kind='linear')
    f_log_g = interpolate.interp1d(log_M, iso.log_g, kind='linear')
    f_Reff = interpolate.interp1d(log_M, iso.Reff, kind='linear')
    f_M = interpolate.interp1d(log_M, iso.M, kind='linear')

    isoBig = Isochrone(log_age)
    isoBig.M_init = np.arange(M_lo, M_hi, 0.005)
    isoBig.log_M_init = np.log10(isoBig.M_init)
    isoBig.log_Teff = f_log_Teff(isoBig.log_M_init)
    isoBig.log_L = f_log_L(isoBig.log_M_init)
    isoBig.log_g = f_log_g(isoBig.log_M_init)
    isoBig.Reff = f_Reff(isoBig.log_M_init)
    isoBig.M = f_M(isoBig.log_M_init)
        
    if test:
        py.clf()
        py.plot(isoBig.log_Teff, isoBig.log_L, 'g-')
        py.plot(iso.log_Teff, iso.log_L, 'ro')
        rng = py.axis()
        py.xlim(rng[1], rng[0])
        py.xlabel('log Teff')
        py.ylabel('log L')
        py.title('Siess+ 2000 Isochrone at log t = %.2f' % log_age)
        py.savefig(outDir + 'plots/hr_isochrone_at' + outSuffix + '.png')

    print time.asctime(), 'Finished.'

    # Write output to file
    _out = open(isoFile, 'w')
    
    _out.write('%10s  %10s  %10s  %10s  %10s  %10s\n' % 
               ('# M_init', 'log Teff', 'log L', 'log g', 'Reff', 'Mass'))
    _out.write('%10s  %10s  %10s  %10s  %10s  %10s\n' % 
               ('# (Msun)', '(Kelvin)', '(Lsun)', '()', '(Rsun)', '(Msun)'))

    for ii in range(len(isoBig.M_init)):
        _out.write('%10.4f  %10.4f  %10.4f  %10.4f  %10.4f  %10.4f\n' %
                   (isoBig.M_init[ii], isoBig.log_Teff[ii], isoBig.log_L[ii],
                    isoBig.log_g[ii], isoBig.Reff[ii], isoBig.M[ii]))

    _out.close()

        
def test_merged_isochrones(logAge):
    geneva = get_geneva_isochrone(logAge=logAge)
    padova = get_padova_isochrone(logAge=logAge)
    siess = get_siess_isochrone(logAge)
    mm = get_meynetmaeder_isochrone(logAge)
    iso = get_merged_isochrone(logAge)

    # Find the Geneva mass that matches the highest Siess mass
    siessHighIdx = siess.mass.argmax()
    siessHighMass = siess.mass[siessHighIdx]
    dmass = siessHighMass - geneva.mass
    genevaLowIdx = np.abs(dmass).argsort()[0]

    # Find the Geneva mass that matches the lowest Meynet & Maeder mass
    mmLowIdx = mm.mass.argmin()
    mmLowMass = mm.mass[mmLowIdx]
    dmass = mmLowMass - geneva.mass
    genevaHighIdx = np.abs(dmass).argsort()[0]

    # Find the Padova mass that matches the highest Siess mass
    dmass = siessHighMass - padova.mass
    padovaLowIdx = np.abs(dmass).argsort()[0]

    import pylab as py

    # Plot all isochrones
    py.clf()
    py.plot(iso.logT, iso.logL, 'k-', label='Merged')
    py.plot(mm.logT, mm.logL, 'b-', label='Geneva (v=300 km/s)')
    py.plot(geneva.logT, geneva.logL, 'g-', label='Geneva (v=0 km/s)')
    py.plot(siess.logT, siess.logL, 'r-', label='Siess+ 2000')
    py.plot(padova.logT, padova.logL, 'y-', label='Padova')

    # Print out intersection points
    py.plot([siess.logT[siessHighIdx]], [siess.logL[siessHighIdx]], 'r*', ms=10)
    py.plot([mm.logT[mmLowIdx]], [mm.logL[mmLowIdx]], 'b*', ms=10)
    py.plot([geneva.logT[genevaLowIdx]], [geneva.logL[genevaLowIdx]], 'g*', ms=10)
    py.plot([geneva.logT[genevaHighIdx]], [geneva.logL[genevaHighIdx]], 'g*', ms=10)
    py.plot([padova.logT[padovaLowIdx]], [padova.logL[padovaLowIdx]], 'y*', ms=10)

    rng = py.axis()
    py.xlim(rng[1], rng[0])
    py.legend(numpoints=1, loc='lower left')
    py.xlabel('log[ Teff ]')
    py.ylabel('log[ L ]')

    py.savefig('/u/jlu/work/models/test/merged_iso_%.2f.png' % logAge)
    #py.savefig('/u/jlu/work/models/test/merged_iso_%.2f.eps' % logAge)

    # Print out information about the intersection points
    print '##########'
    print '# logAge = ' + str(logAge)
    print '##########'
    print 'Siess Intersection with Geneva'
    print '%10s  %10s  %10s' % ('', 'Geneva', 'Siess')
    print '%10s  %10.2f  %10.2f' % \
        ('Mass', geneva.mass[genevaLowIdx], siess.mass[siessHighIdx])
    print '%10s  %10.2f  %10.2f' % \
        ('logT', geneva.logT[genevaLowIdx], siess.logT[siessHighIdx])
    print '%10s  %10.2f  %10.2f' % \
        ('logg', geneva.logg[genevaLowIdx], siess.logg[siessHighIdx])
    print '%10s  %10.2f  %10.2f' % \
        ('logL', geneva.logL[genevaLowIdx], siess.logL[siessHighIdx])
    print ''

    print 'Meynet & Maeder Intersection with Geneva'
    print '%10s  %10s  %10s' % ('', 'Geneva', 'MeynetMaed')
    print '%10s  %10.2f  %10.2f' % \
        ('Mass', geneva.mass[genevaHighIdx], mm.mass[mmLowIdx])
    print '%10s  %10.2f  %10.2f' % \
        ('logT', geneva.logT[genevaHighIdx], mm.logT[mmLowIdx])
    print '%10s  %10.2f  %10.2f' % \
        ('logg', geneva.logg[genevaHighIdx], mm.logg[mmLowIdx])
    print '%10s  %10.2f  %10.2f' % \
        ('logL', geneva.logL[genevaHighIdx], mm.logL[mmLowIdx])
    print ''

    print 'Padova Intersection with Siess'
    print '%10s  %10s  %10s' % ('', 'Padova', 'Siess')
    print '%10s  %10.2f  %10.2f' % \
        ('Mass', padova.mass[padovaLowIdx], siess.mass[siessHighIdx])
    print '%10s  %10.2f  %10.2f' % \
        ('logT', padova.logT[padovaLowIdx], siess.logT[siessHighIdx])
    print '%10s  %10.2f  %10.2f' % \
        ('logg', padova.logg[padovaLowIdx], siess.logg[siessHighIdx])
    print '%10s  %10.2f  %10.2f' % \
        ('logL', padova.logL[padovaLowIdx], siess.logL[siessHighIdx])
    print ''


def merge_all_isochrones_siess_mm_padova(metallicity=0.02, rotation=True):
    """
    Make isochrone files containing a continuous distribution of 
    masses using the pre-MS Siess+ isochrones below 7 Msun and 
    the Geneva rotating Meynet & Maeder isochrones above 9 Msun. This code
    uses the already existing isochrones and then fills in the mass
    gap between the two using interpolation.

    For ages beyond logAge > 7.4, the MS turn-off mass moves to
    ~7 Msun and we need to resort to another set of models to fill in
    the post-main-sequence evolution for masses <= 7 Msun. We will use
    the Padova evolutionary tracks for all ages over logAge=7.4 (~25 Myr).
    """
    # Root data directory for Meynet & Maeder isocrhones
    rootDirMM = '/u/jlu/work/models/geneva/meynetMaeder/'
    metalPart = 'z' + str(int(metallicity * 1000)).zfill(2)
    rotPart = 'S0'
    if rotation:
        rotPart = 'S3'
    rootDirMM += 'iso_' + metalPart + rotPart + '/'

    # Root data directory for Siess+ isochrones
    rootDirSiess = '/u/jlu/work/models/preMS/siess2000/'
    metSuffix = 'z' + str(metallicity).split('.')[-1]
    rootDirSiess += metSuffix + '/iso/'

    # Root data directory for Geneva isochrones
    rootDirPadova = '/u/jlu/work/models/padova/'
    metalSuffix = 'z' + str(metallicity).split('.')[-1]
    rootDirPadova += metalSuffix + '/'

    # Search both directories for iso_*.dat files
    isoFilesMM = glob.glob(rootDirMM + 'iso_*.dat')
    isoFilesS = glob.glob(rootDirSiess + 'iso_*.dat')
    isoFilesP = glob.glob(rootDirPadova + 'iso*')

    # Output of merged isochrones
    outDir = '/u/jlu/work/models/merged/siess_meynetMaeder_padova/%s/' % (metSuffix)
    gcutil.mkdir(outDir)
    

    for ii in range(len(isoFilesMM)):
        isoFilesMM[ii] = isoFilesMM[ii].split('/')[-1]

    for ii in range(len(isoFilesS)):
        isoFilesS[ii] = isoFilesS[ii].split('/')[-1]

    for ii in range(len(isoFilesP)):
        isoFilesP[ii] = isoFilesP[ii].split('/')[-1]

    for ii in range(len(isoFilesS)):
        isoFileS = isoFilesS[ii]

        logAgeStr = isoFileS.replace('iso_', '').replace('.dat', '')
        logAge = float(logAgeStr)

        # Case where logAge <= 7.4, we merge with Meynet & Maeder
        if logAge <= 7.4:
            if isoFileS not in isoFilesMM:
                print 'Skipping isochrones from ', isoFileS
                continue

            print 'Merging isochrones Siess+Meynet from ', isoFileS
            iso = merge_isochrone_siess_mm(logAge, metallicity=metallicity,
                                           rotation=rotation)
        else:
            if isoFileS not in isoFilesP:
                print 'Skipping isochrones from ', isoFileS
                continue
                
            print 'Merging isochrones Siess+Padova from ', isoFileS
            iso = merge_isochrone_siess_padova(logAge, metallicity=metallicity)
            
        _out = open(outDir + isoFileS, 'w')

        _out.write('%12s  %10s  %10s  %10s  %10s  %-10s\n' % 
                   ('# M_init', 'log T', 'log L', 'log g', 'log T WR', 'Source'))
        _out.write('%12s  %10s  %10s  %10s  %10s  %-10s\n' % 
                   ('# (Msun)', '(Kelvin)', '(Lsun)', '()', '(Kelvin)', '()'))

        for kk in range(len(iso.mass)):
            _out.write('%12.6f  %10.4f  %10.4f  %10.4f  %10.4f  %-10s\n' %
                       (iso.mass[kk], iso.logT[kk], iso.logL[kk],
                        iso.logg[kk], iso.logT_WR[kk], iso.source[kk]))

        _out.close()
        

def merge_isochrone_siess_mm(logAge, metallicity=0.02, rotation=True):
    isoMM = get_meynetmaeder_isochrone(logAge, metallicity=metallicity,
                                       rotation=rotation)
    isoS = get_siess_isochrone(logAge, metallicity=metallicity)

    # Make arrays containing the source of each point
    isoS.source = np.array(['Siess']*len(isoS.mass))
    isoMM.source = np.array(['Meynet']*len(isoMM.mass))

    # Combine the arrays
    M = np.append(isoS.mass, isoMM.mass)
    logT = np.append(isoS.logT, isoMM.logT)
    logg = np.append(isoS.logg, isoMM.logg)
    logL = np.append(isoS.logL, isoMM.logL)
    logT_WR = np.append(isoS.logT, isoMM.logT_WR)
    source = np.append(isoS.source, isoMM.source)

    logM = np.log10(M)
    
    f_logT = interpolate.interp1d(logM, logT, kind='linear')
    f_logL = interpolate.interp1d(logM, logL, kind='linear')
    f_logg = interpolate.interp1d(logM, logg, kind='linear')
    f_logT_WR = interpolate.interp1d(logM, logT_WR, kind='linear')

    M_gap = np.arange(isoS.mass.max(), isoMM.mass.min(), 0.01)
    logM_gap = np.log10(M_gap)
    logT_gap = f_logT(logM_gap)
    logL_gap = f_logL(logM_gap)
    logg_gap = f_logg(logM_gap)
    logT_WR_gap = f_logT_WR(logM_gap)
    source_gap = np.array(['interp']*len(M_gap))

    # Combine all the arrays
    M = np.concatenate([isoS.mass, M_gap, isoMM.mass])
    logT = np.concatenate([isoS.logT, logT_gap, isoMM.logT])
    logg = np.concatenate([isoS.logg, logg_gap, isoMM.logg])
    logL = np.concatenate([isoS.logL, logL_gap, isoMM.logL])
    logT_WR = np.concatenate([isoS.logT, logT_WR_gap, isoMM.logT_WR])
    source = np.concatenate([isoS.source, source_gap, isoMM.source])
    
    iso = objects.DataHolder()
    iso.mass = M
    iso.logL = logL
    iso.logg = logg
    iso.logT = logT
    iso.logT_WR = logT_WR
    iso.source = source

    return iso

def merge_isochrone_siess_padova(logAge, metallicity=0.02):
    isoP = get_padova_isochrone(logAge, metallicity=metallicity)
    isoS = get_siess_isochrone(logAge, metallicity=metallicity)

    # Interpolate the Padvoa isochrone over a more finely sampled
    # mass distribution. Values are interpolated linearly in log M.
    isoP.logM = np.log10(isoP.mass)
    print time.asctime(), 'Interpolate across track to different masses.'
    f_logT = interpolate.interp1d(isoP.logM, isoP.logT, kind='linear')
    f_logL = interpolate.interp1d(isoP.logM, isoP.logL, kind='linear')
    f_logg = interpolate.interp1d(isoP.logM, isoP.logg, kind='linear')

    # Add in points from the raw Padova track as they contain even denser
    # sampling at key evolutionary stages (giants).
    isoBig = Isochrone(logAge)
    isoBig.mass = np.arange(isoP.mass.min(), isoP.mass.max(), 0.005)
    isoBig.mass = np.append(isoBig.mass, isoP.mass)
    isoBig.mass.sort()

    isoBig.logM = np.log10(isoBig.mass)
    isoBig.logT = f_logT(isoBig.logM)
    isoBig.logL = f_logL(isoBig.logM)
    isoBig.logg = f_logg(isoBig.logM)

    isoP = isoBig

    # Make arrays containing the source of each point
    isoS.source = np.array(['Siess']*len(isoS.mass))
    isoP.source = np.array(['Padova']*len(isoP.mass))

    isoP.logT_WR = isoP.logT
    
    # Use Siess model as high up as it goes... then fill
    # in with Padova after that.
    hiMassS = isoS.mass.max()
    idxP = np.where(isoP.mass > hiMassS)[0]

    if len(idxP) == 0:
        print 'merge_isochrone_siess_padova: No valid Padvoa masses (log_age = %.2f, Z = %.2f)' % \
            (logAge, metallicity)

    # Combine the arrays
    M = np.append(isoS.mass, isoP.mass[idxP])
    logT = np.append(isoS.logT, isoP.logT[idxP])
    logg = np.append(isoS.logg, isoP.logg[idxP])
    logL = np.append(isoS.logL, isoP.logL[idxP])
    logT_WR = np.append(isoS.logT, isoP.logT_WR[idxP])
    source = np.append(isoS.source, isoP.source[idxP])

    iso = objects.DataHolder()
    iso.mass = M
    iso.logL = logL
    iso.logg = logg
    iso.logT = logT
    iso.logT_WR = logT_WR
    iso.source = source

    return iso

def get_merged_isochrone(logAge, metallicity=0.02):
    # Pre-calculated merged isochrones
    metSuffix = 'z' + str(metallicity).split('.')[-1]
    isoFile = '/u/jlu/work/models/merged/siess_meynetMaeder_padova/%s/' % (metSuffix)
    isoFile += 'iso_%.2f.dat' % logAge

    data = asciidata.open(isoFile)
    
    iso = objects.DataHolder()
    iso.mass = data[0].tonumpy()
    iso.logT = data[1].tonumpy()
    iso.logL = data[2].tonumpy()
    iso.logg = data[3].tonumpy()
    iso.logT_WR = data[4].tonumpy()

    return iso
                     
class StarTrack(object):
    def __init__(self, mass_init):
	self.M_init = mass_init

class Isochrone(object):
    def __init__(self, log_age):
        self.log_age = log_age

def get_siess_tracks(metallicity=0.02):
    pms_dir = '/u/jlu/work/models/preMS/siess2000/'

    metSuffix = 'z' + str(metallicity).split('.')[-1]

    if not os.path.exists(pms_dir + metSuffix):
        print 'Failed to find Siess PMS models for metallicity = ' + metSuffix
        
    pms_dir += metSuffix + '/'
        
    files = glob.glob(pms_dir + '*.hrd')
    count = len(files)
    
    data = objects.DataHolder()

    data.tracks = []
    data.masses = []

    for ff in range(len(files)):
        d = asciidata.open(files[ff])

        track = StarTrack(d[9][0])

        track.phase = d[1].tonumpy()
        track.log_L = np.log10( d[2].tonumpy() )
        track.mag_bol = d[3].tonumpy()
        track.Reff = d[4].tonumpy()
        track.R_total = d[5].tonumpy()
        track.log_Teff = np.log10( d[6].tonumpy() )
        track.density_eff = d[7].tonumpy()
        track.log_g = d[8].tonumpy()
        track.M = d[9].tonumpy()
        track.log_age = np.log10( d[10].tonumpy() )

        data.tracks.append(track)
        data.masses.append(track.M_init)

    data.masses = np.array(data.masses)

    # We need to resort so that everything is in order of increasing mass.
    sdx = data.masses.argsort()
    data.masses = data.masses[sdx]
    data.tracks = [data.tracks[ss] for ss in sdx]

    return data


def highest_mass_at_age(log_age, t, test=False):
    """
    Calculate the highest mass star that still exists at a given age.
    Do this by generating a curve of highest mass vs. time and pull out 
    the new highest mass at the specified age. This is only necessary if 
    the age we are interested in is larger than the lifetimes of the stars
    in this model.    
    """
    numStars = len(t.masses)
    log_M = np.log10( t.masses )

    log_age_hi = np.zeros(numStars, dtype=float)
    for ii in range(numStars):
        # Get rid of the post-main-sequence evolution phsae.
        tmp = np.where(t.tracks[ii].phase <= 2)[0]
        log_age_hi[ii] = t.tracks[ii].log_age[tmp][-1]

    if log_age <= log_age_hi.min():
        log_M_hi = log_M.max()
    elif log_age >= log_age_hi.max():
        log_M_hi = log_M.min()
    else:
        # Linear interpolation
        sdx = log_age_hi.argsort()
        func_log_M = interpolate.interp1d(log_age_hi[sdx], log_M[sdx])
        log_M_hi = func_log_M(log_age)

        if test:
            py.clf()
            py.plot(log_age_hi, log_M, 'ko-')
            test_log_age = np.arange(log_age_hi.min(), log_age_hi.max(), 0.05)
            py.plot(test_log_age, func_log_M(test_log_age), 'b-')

    print 'Highest Mass at log(t) = %.2f is %.2f Msun' % (log_age, 10**log_M_hi)

    return log_M_hi



def old_meynetmaeder_code():
    # Get all the appropriate files
    files = glob.glob(rootDir + 'D*' + metalPart + rotPart + '*.dat')

    if len(files) == 0:
        print 'Failed to find Meynet & Maeder models for metallicity = ' + \
            metalPart
        return
        

    mainPartTmp = []
    suffixTmp = []
    for ff in range(len(files)):
        file = files[ff].split('/')[-1]
        mainPartTmp.append( file[0:8] )
        suffixTmp.append( file[8:-4] )
    mainPartTmp = np.array(mainPartTmp)
    suffixTmp = np.array(suffixTmp)

    # Clean up duplicate entries. Order preference of the various
    # models are (highest to lowest):
    #    'A': includes anisotropic winds
    #   'MZ': metallicity dependent winds
    #     '': default
    mainPart = np.unique( mainPartTmp )
    files = []

    for mm in range(len(mainPart)):
        idx = np.where(mainPartTmp == mainPart[mm])[0]

        if 'MZ' in suffixTmp[idx]:
            ii = np.where(suffixTmp[idx] == 'MZ')[0]
            suffix = 'MZ'
        elif 'Z' in suffixTmp[idx]:
            ii = np.where(suffixTmp[idx] == 'A')[0]
            suffix = 'A'
        else:
            suffix = suffixTmp[idx[0]]

        files.append( rootDir + mainPart[mm] + suffix + '.dat' )

    # Now loop through the files and pull out the data.
    solarT = 5777 # Kelvin
    g = 980.7 # cm/s^2
    solarg = 27.94  * g # cgs

    count = len(files)

    mass = np.zeros(count, dtype=float)
    logL = np.zeros(count, dtype=float)
    logg = np.zeros(count, dtype=float)
    logT = np.zeros(count, dtype=float)


    for ff in range(count):
        d = asciidata.open(files[ff])

        ageTmp = d[1].tonumpy()
        massTmp = d[2].tonumpy()
        logLTmp = d[3].tonumpy()
        logTTmp = d[4].tonumpy()

        tempTmp = 10**logTTmp
        lumTmp = 10**logLTmp

        # Convert gravity to log g (via cgs first)
        T = tempTmp / solarT
        loggTmp = np.log10(solarg * massTmp * T**4 / lumTmp)

        if inputAge > ageTmp[-1]:
            continue

        mass[ff] = round(massTmp[0])  # initial mass
        logL[ff] = np.interp(inputAge, ageTmp, logLTmp)
        logT[ff] = np.interp(inputAge, ageTmp, logTTmp)
        logg[ff] = np.interp(inputAge, ageTmp, loggTmp)

    # Clean up non-existent stars at this age
    idx = np.where(mass != 0)[0]
    mass = mass[idx]
    logT = logT[idx]
    logg = logg[idx]
    logL = logL[idx]

    obj = objects.DataHolder()
    obj.mass = mass
    obj.logT = logT
    obj.logg = logg
    obj.logL = logL

    return obj

