import numpy as np
import pylab as py
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
from astropy.table import Table, Column

models_dir = '/g/lu/models/evolution/'
    
def get_geneva_isochrone(metallicity=0.02, logAge=8.0):
    """
    Metallicity in Z (def = solor of 0.02).
    """
    rootDir = models_dir + 'geneva/iso/'
    
    metalPart = str(int(metallicity * 1000)).zfill(3)
    agePart = str(int(logAge * 100)).zfill(4)

    genevaFile = rootDir + metalPart + '/c/iso_c' + metalPart + '_' + \
        agePart + '.UBVRIJHKLM'

    
    if not os.path.exists(genevaFile):
        print 'Geneva isochrone file does not exist:'
        print '  ' + genevaFile
        return None

    table = Table.read(genevaFile, format='ascii')
    cols = table.keys()

    mass = table[cols[1]]
    logT = table[cols[3]]
    logg = table[cols[4]]
    logL = table[cols[5]]

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

    rootDir = models_dir + 'geneva/meynetMaeder/'

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
        
    data = Table.read(isoFile, format='ascii')
    cols = data.keys()
    mass = data[cols[1]]
    logT = data[cols[2]]
    logg = data[cols[7]]
    logL = data[cols[8]]
    logT_WR = data[cols[10]]
    
    obj = objects.DataHolder()
    obj.mass = mass
    obj.logT = logT
    obj.logg = logg
    obj.logL = logL
    obj.logT_WR = logT_WR
    
    return obj

def make_isochrone_meynetmaeder(logAge, metallicity=0.02, rotation=True):
    rootDir = models_dir + 'geneva/meynetMaeder/'

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

            root = models_dir + 'geneva/meynetMaeder2003/'
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

            root = models_dir + 'geneva/meynetMaeder2005/'
            outfile = open(root + file, 'w')
            outfile.write(contents)
            outfile.close()

        except ValueError:
            continue

    
def get_palla_stahler_isochrone(logAge):
    pms_isochrones = models_dir + 'preMS/pallaStahler1999/' + \
        'pms_isochrones.txt'

    data = Table.read(pms_isochrones, format='ascii')
    cols = data.keys()

    mass = data[cols[0]]
    temp = data[cols[1]]
    lum = data[cols[2]]
    age = data[cols[3]]

    # Interpolate to get the isochrone at the proper age.
    inputAge = 10**logAge / 10**6

    # For each mass bin, find the closest ages on either side.
    umass = np.unique(mass)

def get_padova_isochrone(logAge, metallicity=0.02):
    mod_dir = models_dir + 'padova/'

    metSuffix = 'z' + str(metallicity).split('.')[-1]

    if not os.path.exists(mod_dir + metSuffix):
        print 'Failed to find Padova models for metallicity = ' + metSuffix
        
    mod_dir += metSuffix + '/'
        
    # First lets use isochrones if we have them instead of reconstructing
    # our own thing.
    isoFile = '%siso_%.2f.dat' % (mod_dir, logAge)

    data = Table.read(isoFile, format='ascii')
    cols = data.keys()

    mass = data[cols[1]]
    logL = data[cols[3]]
    logT = data[cols[4]]
    logg = data[cols[5]]

    obj = objects.DataHolder()
    obj.mass = mass
    obj.logT = logT
    obj.logg = logg
    obj.logL = logL
    
    return obj

def get_siess_isochrone(logAge, metallicity=0.02):
    pms_dir = models_dir + 'preMS/siess2000/'

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

    data = Table.read(isoFile, format='ascii')
    cols = data.keys()
    
    mass = data[cols[0]]
    logT = data[cols[1]]
    logL = data[cols[2]]
    logg = data[cols[3]]

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

    rootDir = models_dir + 'preMS/siess2000/'
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
    py.plot(iso.logT, iso.logL, 'k-', color='orange', label='Interpolated', linewidth=1.5)
    py.plot(mm.logT, mm.logL, 'b-', label='Geneva (v=300 km/s)', linewidth=1.5)
    py.plot(geneva.logT, geneva.logL, 'g-', label='Geneva (v=0 km/s)', linewidth=1.5)
    py.plot(siess.logT, siess.logL, 'r-', label='Siess+ 2000', linewidth=1.5)
    #py.plot(padova.logT, padova.logL, 'y-', label='Padova', linewidth=1.5)

    # Print out intersection points
    py.plot([geneva.logT[genevaLowIdx]], [geneva.logL[genevaLowIdx]], 'gs', ms=7)
    py.plot([geneva.logT[genevaHighIdx]], [geneva.logL[genevaHighIdx]], 'g*', ms=10)
    py.plot([siess.logT[siessHighIdx]], [siess.logL[siessHighIdx]], 'rs', ms=7)
    py.plot([mm.logT[mmLowIdx]], [mm.logL[mmLowIdx]], 'b*', ms=10)
    #py.plot([padova.logT[padovaLowIdx]], [padova.logL[padovaLowIdx]], 'y*', ms=10)

    rng = py.axis()
    py.xlim(rng[1], rng[0])
    py.legend(numpoints=1, loc='lower left')
    py.xlabel('log Teff')
    py.ylabel('log L')

    py.savefig(models_dir + 'test/merged_iso_%.2f.png' % logAge)
    py.savefig(models_dir + 'test/merged_iso_%.2f.eps' % logAge)

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
    rootDirMM = models_dir + 'geneva/meynetMaeder/'
    metalPart = 'z' + str(int(metallicity * 1000)).zfill(2)
    rotPart = 'S0'
    if rotation:
        rotPart = 'S3'
    rootDirMM += 'iso_' + metalPart + rotPart + '/'

    # Root data directory for Siess+ isochrones
    rootDirSiess = models_dir + 'preMS/siess2000/'
    metSuffix = 'z' + str(metallicity).split('.')[-1]
    rootDirSiess += metSuffix + '/iso/'

    # Root data directory for Geneva isochrones
    rootDirPadova = models_dir + 'padova/'
    metalSuffix = 'z' + str(metallicity).split('.')[-1]
    rootDirPadova += metalSuffix + '/'

    # Search both directories for iso_*.dat files
    isoFilesMM = glob.glob(rootDirMM + 'iso_*.dat')
    isoFilesS = glob.glob(rootDirSiess + 'iso_*.dat')
    isoFilesP = glob.glob(rootDirPadova + 'iso*')

    # Output of merged isochrones
    outDir = models_dir + 'merged/siess_meynetMaeder_padova/%s/' % (metSuffix)
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
    isoFile = models_dir + 'merged/siess_meynetMaeder_padova/%s/' % (metSuffix)
    isoFile += 'iso_%.2f.dat' % logAge

    data = Table.read(isoFile, format='ascii')
    cols = data.keys()
    
    iso = objects.DataHolder()
    iso.mass = data[cols[0]]
    iso.logT = data[cols[1]]
    iso.logL = data[cols[2]]
    iso.logg = data[cols[3]]
    iso.logT_WR = data[cols[4]]

    return iso

def get_merged_isochrone_PEP(logAge, metallicity=0.015):
    """
    Get merged isochrones from Pisa, Ekstrom, and Parsec models.
    logAge must be between 6.0 - 8.0 in increments of 0.01
    """
    if metallicity != 0.015:
        print 'Non-solar metallicities not supported yet!'
        print 'Quitting'
        return

    # Pre-calculated merged isochrones
    metSuffix = 'z' + str(metallicity).split('.')[-1]
    isoFile = models_dir + 'merged/pisa_ekstrom_parsec/%s/' % (metSuffix)
    isoFile += 'iso_%.2f.dat' % logAge

    if not os.path.exists(isoFile):
        print 'Merged isochrone at logAge {0:3.2f} not created yet'.format(logAge)
        print 'ERROR!!!!!'
        return

    data = Table.read(isoFile, format='ascii')
    cols = data.keys()
    
    iso = objects.DataHolder()
    iso.mass = data[cols[0]]
    iso.logT = data[cols[1]]
    iso.logL = data[cols[2]]
    iso.logg = data[cols[3]]
    iso.logT_WR = data[cols[4]]

    return iso
                     
class StarTrack(object):
    def __init__(self, mass_init):
	self.M_init = mass_init

class Isochrone(object):
    def __init__(self, log_age):
        self.log_age = log_age

def get_siess_tracks(metallicity=0.02):
    pms_dir = models_dir + 'preMS/siess2000/'

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
        d = Table.read(files[ff], format='ascii')
        cols = d.keys()

        track = StarTrack(d[9][0])

        track.phase = d[cols[1]]
        track.log_L = np.log10( d[cols[2]] )
        track.mag_bol = d[cols[3]]
        track.Reff = d[cols[4]]
        track.R_total = d[cols[5]]
        track.log_Teff = np.log10( d[cols[6]] )
        track.density_eff = d[cols[7]]
        track.log_g = d[cols[8]]
        track.M = d[cols[9]]
        track.log_age = np.log10( d[cols[10]] )

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
        d = Table.read(files[ff], format='ascii')
        cols = d.keys()

        ageTmp = d[cols[1]]
        massTmp = d[cols[2]]
        logLTmp = d[cols[3]]
        logTTmp = d[cols[4]]

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

#--------------------------------------------------#
# Updated set of merged evolution models:
# Ekstrom+12 rotating + Pisa 2011 for logAge < 7.4,
# Ekstrom+12 rotating + PARSEC for logAge > 7.4
#--------------------------------------------------#
def get_pisa_tracks(metallicity=0.015):
    """
    Helper code to get pisa tracks at given metallicity
    """
    pms_dir = models_dir + 'Pisa2011/iso/tracks/'
    metSuffix = 'z' + str(metallicity).split('.')[-1]
    pms_dir += metSuffix + '/'

    if not os.path.exists(pms_dir):
        print 'Failed to find Siess PMS models for metallicity = ' + metSuffix

    # Collect the tracks 
    files = glob.glob(pms_dir + '*.DAT')
    count = len(files)
    
    data = objects.DataHolder()

    data.tracks = []
    data.masses = []

    # Extract useful params from tracks
    for ff in range(len(files)):
        d = Table.read(files[ff], format='ascii')
        cols = d.keys()

        # Extract initial mass from filename
        filename = files[ff].split('_')
        m_init = filename[1][1:] 
        track = StarTrack(float(m_init))

        # Calculate log g from T and L
        L_sun = 3.8 * 10**33 #cgs
        SB_sig = 5.67 * 10**-5 #cgs
        M_sun = 2 * 10**33 #cgs
        G_const = 6.67 * 10**-8 #cgs
        
        radius = np.sqrt( (10**d[cols[3]] * L_sun) /
                          (4 * np.pi * SB_sig *  (10**d[cols[4]])**4) )
        g = (G_const * float(m_init) * M_sun) / radius**2

        # Phase > 2 is post-main sequence; none for Pisa
        track.phase = np.ones(len(d[cols[1]])) 
        track.log_L = d[cols[3]]
        track.log_Teff = d[cols[4]]
        track.density_eff = d[cols[6]]
        track.log_g = np.log10(g)
        track.log_age = d[cols[1]]
        # Mass doesn't change in these tracks
        track.M = np.ones(len(d[cols[1]])) * float(m_init) 

        data.tracks.append(track)
        data.masses.append(track.M_init)

    data.masses = np.array(data.masses)

    # We need to resort so that everything is in order of increasing mass.
    sdx = data.masses.argsort()
    data.masses = data.masses[sdx]
    data.tracks = [data.tracks[ss] for ss in sdx]

    return data

def get_orig_pisa_isochrones(metallicity=0.015):
    """
    Helper code to get the original pisa isochrones at given metallicity.
    These are downloaded online
    """
    pms_dir = models_dir + '/Pisa2011/iso/iso_orig/'
    metSuffix = 'z' + str(metallicity).split('.')[-1]
    pms_dir += metSuffix + '/'

    if not os.path.exists(pms_dir):
        print 'Failed to find Siess PMS isochrones for metallicity = ' + metSuffix
        return
    
    # Collect the isochrones
    files = glob.glob(pms_dir + '*.dat')
    count = len(files)

    data = objects.DataHolder()

    data.isochrones = []
    data.log_ages = []
    
    # Extract useful params from isochrones
    for ff in range(len(files)):
        d = Table.read(files[ff], format='ascii')

        # Extract logAge from filename
        log_age = float(files[ff].split('_')[2][:-4])

        # Create an isochrone object   
        iso = Isochrone(log_age)
        iso.M = d['col3']
        iso.log_Teff = d['col2']
        iso.log_L = d['col1']

        # If a log g column exist, extract it. Otherwise, calculate
        # log g from T and L and add column at end
        if len(d.keys()) == 3:
            
            # Calculate log g from T and L
            L_sun = 3.8 * 10**33 #cgs
            SB_sig = 5.67 * 10**-5 #cgs
            M_sun = 2. * 10**33 #cgs
            G_const = 6.67 * 10**-8 #cgs
        
            radius = np.sqrt( (10**d['col1'] * L_sun) /
                          (4 * np.pi * SB_sig *  (10**d['col2'])**4) )
            g = (G_const * d['col3'] * M_sun) / radius**2


            iso.log_g = np.log10(g.astype(np.float))
        else:
            iso.log_g = d['col4']
        
        data.isochrones.append(iso)
        data.log_ages.append(log_age)

        # If it doesn't already exist, add a column with logg vals. This will
        # be appended at the end
        if len(d.keys()) == 3:
            logg_col = Column(iso.log_g, name = 'col4')
            d.add_column(logg_col, index=3)
            d.write(files[ff],format='ascii')
    data.log_ages = np.array(data.log_ages)

    # Resort so that everything is in order of increasing age
    sdx = data.log_ages.argsort()
    data.masses = data.log_ages[sdx]
    data.isochrones = [data.isochrones[ss] for ss in sdx]

    return data

def interpolate_iso_tempgrid(iso1, iso2):
    """
    Helper function that interpolates temps of iso2
    onto the temps of iso1. Both iso1 and iso2 are
    assumed to be Isochrone objects

    Returns iso1 and iso2 again, but iso2 with the same mass
    grid as iso1
    """
    # Extract mass grid of iso1. This is what we want to interpolate iso2 too.
    # Only interpolate values in temp range of iso2, however
    overlap = np.where( (iso1.log_Teff >= min(iso2.log_Teff)) &
                        (iso1.log_Teff <= max(iso2.log_Teff)) )
    interp_temp = iso1.log_Teff[overlap]

    # Build interpolation functions for iso2
    f_log_M = interpolate.interp1d(iso2.log_Teff, iso2.M, kind='linear')
    f_log_L = interpolate.interp1d(iso2.log_Teff, iso2.log_L, kind='linear')
    f_log_g = interpolate.interp1d(iso2.log_Teff, iso2.log_g, kind='linear')

    # Reset iso1, iso2 to values in overlap temps
    iso2.M = f_log_M(interp_temp)
    iso2.log_Teff = interp_temp
    iso2.log_L = f_log_L(interp_temp)
    iso2.log_g = f_log_g(interp_temp)

    iso1.M = iso1.M[overlap]
    iso1.log_Teff = iso1.log_Teff[overlap]
    iso1.log_L = iso1.log_L[overlap]
    iso1.log_g = iso1.log_g[overlap]

    return iso1, iso2

def make_isochrone_pisa_tracks(log_age, metallicity=0.015, 
                         tracks=None, test=False):
    """
    Read in a set of evolutionary tracks and generate an isochrone
    that is well sampled at the full range of masses. This code
    is a re-written version of iso.f provided by the Geneva models group.

    FOR PISA MODELS, THIS ISN'T AS ACCURATE AS make_isochrone_pisa_iso@@
    """
    # Check to make sure user really wants to interpolate using evolutionary
    # tracks
    cont = input('Warning: interpolating Pisa models via evolutionary tracks instead of'+
    ' isochrones is not as accurate. Do you wish to continue? (\'y\', \'n\', need quotes):')

    if cont == 'n':
        print 'Code stopped'
        return
    
    age = 10**log_age

    rootDir = models_dir + '/Pisa2011/iso/tracks/'
    metSuffix = 'z' + str(metallicity).split('.')[-1]
    rootDir += metSuffix + '/'

    isoFile = models_dir+'/Pisa2011/iso/'+metSuffix+'/interp_tests/'+'iso_%.2f_evoInt.dat' % log_age
    outSuffix = '_%.2f' % (log_age)

    if not os.path.exists(rootDir):
        print 'Failed to find Pisa PMS tracks for metallicity = ' + metSuffix
        return

    print '*** Generating Pisa isochrone for log t = %.2f and Z = %.3f' % \
        (log_age, metallicity)

    if tracks is None:
        print time.asctime(), 'Getting tracks.'
        t = get_pisa_tracks()
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
    iso.density_eff = np.zeros(numStars, dtype=float)
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
            f_density_eff = interpolate.interp1d(age_on_track, track.density_eff)
            f_M = interpolate.interp1d(age_on_track, track.M)

            iso.log_Teff[ss] = f_log_Teff(age)
            iso.log_L[ss] = f_log_L(age)
            iso.log_g[ss] = f_log_g(age)
            iso.density_eff[ss] = f_density_eff(age)
            iso.M[ss] = f_M(age)
        else:
            # We have hit the end of the road. Add the last data point
            # for this too-massive star just to prevent the next round
            # of interpolations from failing.
            print 'Adding too-massive track at %.2f' % t.masses[ss]
            iso.log_Teff[ss] = track.log_Teff[-1]
            iso.log_L[ss] = track.log_L[-1]
            iso.log_g[ss] = track.log_g[-1]
            iso.density_eff[ss] = track.density_eff[-1]
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
        py.title('Pisa 2011 Tracks (red at log t = %.2f)' % log_age)
        py.savefig(rootDir + 'plots/hr_tracks_at' + outSuffix + '.png')

    # clean up too-high masses
    idx = np.where(iso.log_Teff != 0)[0]
    log_M = log_M[idx]
    iso.log_Teff = iso.log_Teff[idx]
    iso.log_L = iso.log_L[idx]
    iso.log_g = iso.log_g[idx]
    iso.density_eff = iso.density_eff[idx]
    iso.M = iso.M[idx]

    # Now we take these isochrones and interpolate to a more finely
    # sampled mass distribution. Values are interpolated linearly in
    # log M.
    print time.asctime(), 'Interpolate across tracks to different masses.'
    f_log_Teff = interpolate.interp1d(log_M, iso.log_Teff, kind='linear')
    f_log_L = interpolate.interp1d(log_M, iso.log_L, kind='linear')
    f_log_g = interpolate.interp1d(log_M, iso.log_g, kind='linear')
    f_density_eff = interpolate.interp1d(log_M, iso.density_eff, kind='linear')
    f_M = interpolate.interp1d(log_M, iso.M, kind='linear')

    isoBig = Isochrone(log_age)
    isoBig.M_init = np.arange(M_lo, M_hi, 0.005)
    isoBig.log_M_init = np.log10(isoBig.M_init)
    isoBig.log_Teff = f_log_Teff(isoBig.log_M_init)
    isoBig.log_L = f_log_L(isoBig.log_M_init)
    isoBig.log_g = f_log_g(isoBig.log_M_init)
    isoBig.density_eff = f_density_eff(isoBig.log_M_init)
    isoBig.M = f_M(isoBig.log_M_init)
        
    if test:
        py.clf()
        py.plot(isoBig.log_Teff, isoBig.log_L, 'g-')
        py.plot(iso.log_Teff, iso.log_L, 'ro')
        rng = py.axis()
        py.xlim(rng[1], rng[0])
        py.xlabel('log Teff')
        py.ylabel('log L')
        py.title('Pisa 2011 Isochrone at log t = %.2f' % log_age)
        py.savefig(rootDir + 'plots/hr_isochrone_at' + outSuffix + '.png')

    print time.asctime(), 'Finished.'

    # Write output to file
    _out = open(isoFile, 'w')
    
    _out.write('%10s  %10s  %10s  %10s  %10s  %10s\n' % 
               ('# M_init', 'log Teff', 'log L', 'log g', 'Density', 'Mass'))
    _out.write('%10s  %10s  %10s  %10s  %10s  %10s\n' % 
               ('# (Msun)', '(Kelvin)', '(Lsun)', '()', '(g/cm^3)', '(Msun)'))

    for ii in range(len(isoBig.M_init)):
        _out.write('%10.4f  %10.4f  %10.4f  %10.4f  %10.4f  %10.4f\n' %
                   (isoBig.M_init[ii], isoBig.log_Teff[ii], isoBig.log_L[ii],
                    isoBig.log_g[ii], isoBig.density_eff[ii], isoBig.M[ii]))

    _out.close()

    return

def make_isochrone_pisa_iso(log_age, metallicity=0.015, 
                         tracks=None, test=False):
    """
    Read in a set of isochrones and generate an isochrone at log_age
    that is well sampled at the full range of masses.

    Puts isochrones is Pisa2011/iso/<metal>/
    """
    # If logage > 8.0, quit immediately...grid doesn't go that high
    if log_age > 8.0:
        print 'Age too high for Pisa grid (max logAge = 8.0)'
        return

    # Directory with where the isochrones will go (both downloaded and interpolated)
    rootDir = models_dir + '/Pisa2011/iso/'
    metSuffix = 'z' + str(metallicity).split('.')[-1]
    rootDir += metSuffix + '/'

    # Can we find the isochrone directory?
    if not os.path.exists(rootDir):
        print 'Failed to find Pisa PMS isochrones for metallicity = ' + metSuffix
        return

    # Check to see if isochrone at given age already exists. If so, quit
    if os.path.exists(rootDir+'iso_{0:3.2f}.dat'.format(log_age)):
        print 'Isochrone at logAge = {0:3.2f} already exists'.format(log_age)
        return
    
    # Name/directory for interpolated isochrone
    isoFile = rootDir+'iso_%3.2f.dat' % log_age
    outSuffix = '_%.2f' % (log_age)

    print '*** Generating Pisa isochrone for log t = %3.2f and Z = %.3f' % \
        (log_age, metallicity)

    print time.asctime(), 'Getting original Pisa isochrones.'
    iso = get_orig_pisa_isochrones()

    # First thing is to find the isochrones immediately above and below desired
    # age
    iso_log_ages = iso.log_ages
    tmp = np.append(iso_log_ages, log_age)

    # Find desired age in ordered sequence; isolate model younger and older
    tmp.sort()
    good = np.where(tmp == log_age)
    young_model_logage = tmp[good[0]-1]
    old_model_logage = tmp[good[0]+1]
    
    # Isolate younger/older isochrones
    young_ind = np.where(iso.log_ages == young_model_logage)
    old_ind = np.where(iso.log_ages == old_model_logage)

    young_iso = iso.isochrones[young_ind[0]]
    old_iso = iso.isochrones[old_ind[0]]

    # Need both younger and older model on same temperature grid for time
    # interpolation. Will adopt mass grid of whichever model is closer in time
    if abs(young_model_logage - log_age) <= abs(old_model_logage - log_age):
        # Use young model mass grid
        young_iso, old_iso = interpolate_iso_tempgrid(young_iso, old_iso)
        
    else:
        # Use old model mass grid
        old_iso, young_iso = interpolate_iso_tempgrid(old_iso, young_iso)

    # Now, can interpolate in time over the two models. Do this star by star.
    # Work in linear time here!!
    numStars = len(young_iso.M)
    
    interp_iso = Isochrone(log_age)
    interp_iso.log_Teff = np.zeros(numStars, dtype=float)
    interp_iso.log_L = np.zeros(numStars, dtype=float)
    interp_iso.log_g = np.zeros(numStars, dtype=float)
    interp_iso.M = young_iso.M # Since mass grids should already be matched
    
    for i in range(numStars):
        # Do interpolations in linear space
        model_ages = [10**young_model_logage[0], 10**old_model_logage[0]]
        target_age = 10**log_age
        #model_ages = [young_model_logage[0], old_model_logage[0]]
        #target_age = log_age
        
        # Build interpolation functions
        Teff_arr = [10**young_iso.log_Teff[i], 10**old_iso.log_Teff[i]]
        logL_arr = [10**young_iso.log_L[i], 10**old_iso.log_L[i]]
        logg_arr = [10**young_iso.log_g[i], 10**old_iso.log_g[i]]
        
        f_log_Teff = interpolate.interp1d(model_ages, Teff_arr, kind='linear')
        f_log_L = interpolate.interp1d(model_ages, logL_arr, kind='linear')
        f_log_g = interpolate.interp1d(model_ages, logg_arr, kind='linear')

        interp_iso.log_Teff[i] = np.log10(f_log_Teff(target_age))
        interp_iso.log_L[i] = np.log10(f_log_L(target_age))
        interp_iso.log_g[i] = np.log10(f_log_g(target_age))

    # If indicated, plot new isochrone along with originals it was interpolated
    # from
    if test:
        py.figure(1)
        py.clf()
        py.plot(interp_iso.log_Teff, interp_iso.log_L, 'k-', label = 'Interp')
        py.plot(young_iso.log_Teff, young_iso.log_L, 'b-',
                label = 'log Age = {0:3.2f}'.format(young_model_logage[0]))
        py.plot(old_iso.log_Teff, old_iso.log_L, 'r-',
                label = 'log Age = {0:3.2f}'.format(old_model_logage[0]))
        rng = py.axis()
        py.xlim(rng[1], rng[0])
        py.xlabel('log Teff')
        py.ylabel('log L')
        py.legend()
        py.title('Pisa 2011 Isochrone at log t = %.2f' % log_age)
        py.savefig(rootDir + 'plots/interp_isochrone_at' + outSuffix + '.png')
    
    print time.asctime(), 'Finished.'

    # Write output to file, MUST BE IN SAME ORDER AS ORIG FILES
    _out = open(isoFile, 'w')
    
    _out.write('%10s  %10s  %10s  %10s\n' % 
               ('# log L', 'log Teff', 'Mass', 'log g'))
    _out.write('%10s  %10s  %10s  %10s\n' % 
               ('# (Lsun)', '(Kelvin)', '(Msun)', '(cgs)'))

    for ii in range(len(interp_iso.M)):
        _out.write('%10.4f  %10.4f  %10.4f  %10.4f\n' %
                   (interp_iso.log_L[ii], interp_iso.log_Teff[ii], interp_iso.M[ii],
                    interp_iso.log_g[ii]))

    _out.close()

    return

def get_Ekstrom_isochrone(logAge, metallicity='solar', rotation=True):
    """
    Load mass, effective temperature, log gravity, and log luminosity
    for the Ekstrom isochrones at given logAge. Code will quit if that
    logAge value doesn't exist (can make some sort of interpolation thing
    later). Also interpolate model to finer mass grid

    Note: mass is currently initial mass, not instantaneous mass
    
    Inputs:
    logAge - Logarithmic Age
    metallicity - in Z (def = solar of 0.014)
    """
    rootDir = models_dir + 'Ekstrom2012/iso/'
    metSuffix = 'z014/'
    if metallicity != 'solar':
        print 'Non-solar Ekstrom+12 metallicities not supported yet'
        return
    rotSuffix = 'rot/'
    if not rotation:
        rotSuffix = 'norot'
        print 'Non-rotating Ekstrom+12 models not supported yet'
        return
    rootDir += metSuffix + rotSuffix

    # Check to see if isochrone exists
    isoFile = rootDir + 'iso_%.2f.dat' % logAge
    if not os.path.exists(isoFile):
        print 'Ekstrom isochrone for logAge = {0:3.2f} does\'t exist'.format(logAge)
        print 'Quitting'
        return
        
    data = Table.read(isoFile, format='ascii')
    cols = data.keys()
    mass = data[cols[2]] #Note: this is initial mass, in M_sun
    logT = data[cols[7]] # K
    logL = data[cols[6]] # L_sun
    logT_WR = data[cols[8]] # K; if this doesn't equal logT, we have a WR star

    # Need to calculate log g from mass and R
    R_sun = 7.*10**10 #cm
    M_sun = 2.*10**33 #g
    G_const = 6.67*10**-8 #cgs
    
    radius = data[cols[19]] #R_sun
    logg = np.log10( (G_const * np.array(mass).astype(np.float) * M_sun) /
                     (np.array(radius).astype(np.float) * R_sun)**2 )
    
    # Interpolate isochrone to finer mass grid on main-ish sequence
    # (1-60 M_sun, or the highest mass in the model); don't want to
    # completely redo all sampling, just this region
    if max(mass) > 60:
        new_masses = np.arange(1, 60+0.1, 0.5)
    else:
        new_masses = np.arange(1, max(mass), 0.5)
    mass_grid = np.append(new_masses, mass)
    mass_grid.sort() # Make sure grid is in proper order

    # Build interpolators in linear space
    f_logT = interpolate.interp1d(mass, 10**logT, kind='linear')
    f_logL = interpolate.interp1d(mass, 10**logL, kind='linear')
    f_logT_WR = interpolate.interp1d(mass, 10**logT_WR, kind='linear')
    f_logg = interpolate.interp1d(mass, 10**logg, kind='linear')

    # Do interpolation, convert back to logspace
    logT_interp = np.log10(f_logT(mass_grid))
    logL_interp = np.log10(f_logL(mass_grid))
    logT_WR_interp = np.log10(f_logT_WR(mass_grid))
    logg_interp = np.log10(f_logg(mass_grid))
    
    # Make isochrone
    obj = objects.DataHolder()
    obj.mass = mass_grid
    obj.logT = logT_interp
    obj.logg = logg_interp
    obj.logL = logL_interp
    obj.logT_WR = logT_WR_interp

    return obj

def get_parsec_isochrone(logAge, metallicity='solar'):
    """
    Load mass, effective temperature, log gravity, and log luminosity
    for the Parsec isochrones at given logAge. Code will quit if that
    logAge value doesn't exist (can make some sort of interpolation thing
    later).

    Note: mass is currently initial mass, not instantaneous mass
    
    Inputs:
    logAge - Logarithmic Age
    metallicity - in Z (def = solar of 0.014)
    """
    rootDir = models_dir + 'ParsecV1.2s/iso/'
    metSuffix = 'z015/'
    if metallicity != 'solar':
        print 'Non-solar Parsec 2011 metallicities not supported yet'
        return
    rootDir += metSuffix

    # Check to see if isochrone exists
    isoFile = rootDir + 'iso_%.2f.dat' % logAge
    if not os.path.exists(isoFile):
        print 'Parsec isochrone for logAge = {0:3.2f} does\'t exist'.format(logAge)
        print 'Quitting'
        return
        
    data = Table.read(isoFile, format='ascii')
    cols = data.keys()
    mass = data[cols[2]] #Note: this is initial mass, in M_sun
    logT = data[cols[5]] # K
    logL = data[cols[4]] # L_sun
    logg = data[cols[6]]
    
    obj = objects.DataHolder()
    obj.mass = mass
    obj.logT = logT
    obj.logg = logg
    obj.logL = logL
    
    return obj

def get_pisa_isochrone(logAge, metallicity='solar'):
    """
    Load mass, effective temperature, log gravity, and log luminosity
    for the Pisa isochrones at given logAge. Code will quit if that
    logAge value doesn't exist (can make some sort of interpolation thing
    later).

    Note: mass is currently initial mass, not instantaneous mass
    
    Inputs:
    logAge - Logarithmic Age
    metallicity - in Z (def = solar of 0.014)
    """
    rootDir = models_dir + 'Pisa2011/iso/'
    metSuffix = 'z015/'
    if metallicity != 'solar':
        print 'Non-solar Pisa 2011 metallicities not supported yet'
        return
    rootDir += metSuffix

    # Check to see if isochrone exists
    isoFile = rootDir + 'iso_%.2f.dat' % logAge
    if not os.path.exists(isoFile):
        print 'Pisa isochrone for logAge = {0:3.2f} does\'t exist'.format(logAge)
        print 'Quitting'
        return
        
    data = Table.read(isoFile, format='ascii')
    cols = data.keys()
    mass = data[cols[2]] #Note: this is initial mass, in M_sun
    logT = data[cols[1]] # K
    logL = data[cols[0]] # L_sun
    logg = data[cols[3]]
    
    obj = objects.DataHolder()
    obj.mass = mass
    obj.logT = logT
    obj.logg = logg
    obj.logL = logL

    return obj

def merge_isochrone_pisa_Ekstrom(logAge, metallicity='solar', rotation=True):
    """
    Function to merge Pisa 2011 and Ekstrom 2012 models. Solar metallicity is
    Z = 0.015 for Pisa 2011 and Z = 0.014 for Ekstrom+12.

    Will take Pisa models to highest available mass, then switch to Ekstrom

    Can only handle ages at which models already exist:
    logAge = 6.0 - 8.0, delta logAge = 0.01
    """
    if metallicity != 'solar':
        print 'Non-solar metallicity not supported yet'
        return
    
    # Get individual Ekstrom and Pisa isochrones at desired age
    isoEkstrom = get_Ekstrom_isochrone(logAge, metallicity=metallicity,
                                       rotation=rotation)
    
    isoPisa = get_pisa_isochrone(logAge, metallicity=metallicity)

    # Take Pisa isochrone as high up in mass as it goes, then switch to Ekstrom.
    # Will trim Ekstrom isochrone here
    max_Pisa = max(isoPisa.mass)
    good = np.where(isoEkstrom.mass > max_Pisa)
    isoEkstrom.mass = isoEkstrom.mass[good]
    isoEkstrom.logT = isoEkstrom.logT[good]
    isoEkstrom.logg = isoEkstrom.logg[good]
    isoEkstrom.logL = isoEkstrom.logL[good]
    isoEkstrom.logT_WR = isoEkstrom.logT_WR[good]
    
    # Make arrays containing the source of each point
    isoPisa.source = np.array(['Pisa']*len(isoPisa.mass))
    isoEkstrom.source = np.array(['Ekstrom']*len(isoEkstrom.mass))

    # Combine the arrays
    M = np.append(isoPisa.mass, isoEkstrom.mass)
    logT = np.append(isoPisa.logT, isoEkstrom.logT)
    logg = np.append(isoPisa.logg, isoEkstrom.logg)
    logL = np.append(isoPisa.logL, isoEkstrom.logL)
    logT_WR = np.append(isoPisa.logT, isoEkstrom.logT_WR)
    source = np.append(isoPisa.source, isoEkstrom.source)

    iso = objects.DataHolder()
    iso.mass = M
    iso.logL = logL
    iso.logg = logg
    iso.logT = logT
    iso.logT_WR = logT_WR
    iso.source = source

    return iso

def merge_isochrone_pisa_parsec(logAge, metallicity='solar'):
    """
    Function to merge Pisa 2011 and ParsecV1.2s models. Solar metallicity is
    Z = 0.015.

    Will take Pisa models to highest available mass, then switch to Parsec

    Can only handle ages at which both sets of models already exist:
    logAge = 6.6 - 8.0, delta logAge = 0.01
    """
    isoParsec = get_parsec_isochrone(logAge, metallicity=metallicity)
    isoPisa = get_pisa_isochrone(logAge, metallicity=metallicity)
    
    # Use Pisa model as high up as it goes, then switch to Parsec
    max_Pisa = max(isoPisa.mass)
    good = np.where(isoParsec.mass > max_Pisa)
    isoParsec.mass = isoParsec.mass[good]
    isoParsec.logT = isoParsec.logT[good]
    isoParsec.logg = isoParsec.logg[good]
    isoParsec.logL = isoParsec.logL[good]
    

    # Make arrays containing the source of each point
    isoParsec.source = np.array(['Parsec']*len(isoParsec.mass))
    isoPisa.source = np.array(['Pisa']*len(isoPisa.mass))

    # Combine the arrays
    M = np.append(isoPisa.mass, isoParsec.mass)
    logT = np.append(isoPisa.logT, isoParsec.logT)
    logg = np.append(isoPisa.logg, isoParsec.logg)
    logL = np.append(isoPisa.logL, isoParsec.logL)
    logT_WR = np.append(isoPisa.logT, isoParsec.logT)
    source = np.append(isoPisa.source, isoParsec.source)

    iso = objects.DataHolder()
    iso.mass = M
    iso.logL = logL
    iso.logg = logg
    iso.logT = logT
    iso.logT_WR = logT_WR
    iso.source = source

    return iso

def merge_all_isochrones_pisa_ekstrom_parsec(metallicity='solar', rotation=True, plot=False):
    """
    Make isochrone files containing a continuous distribution of 
    masses using the pre-MS Pisa 2011 isochrones up to top of mass
    limit (max ~7 M_sun), then switch over to Ekstom+12.

    For ages beyond logAge > 7.4 (~25 Myr), switch from Ekstrom+12 models
    to Parsec models. This is because Parsec models are better suited for
    older stars

    metallicity = 'solar' --> Ekstrom+12 z014, Pisa2011 z015, Parsec z015

    if plot = True, will make plots of merged isochrones in 'plots' directory,
    which must already exist
    
    Code is expected to be run in merged model working directory.
    """
    # Root data directory for Ekstrom+12 isochrones
    rootDirE = models_dir + 'Ekstrom2012/iso'
    metalPart = '/z014'
    if metallicity != 'solar':
        print 'Non-solar metallicities not supported yet'
        return
    rotPart = '/rot'
    if not rotation:
        rotPart = '/rot'
        print 'Ekstrom+12 non-rotation models not supported yet'
        return
    rootDirE += metalPart+rotPart+'/'

    # Root data directory for Pisa isochrones
    rootDirPisa = models_dir + 'Pisa2011/iso'
    metSuffix = '/z015'
    if metallicity != 'solar':
        print 'Non-solar metallicities not supported yet'
        return
    rootDirPisa += metSuffix + '/'

    # Root data directory for Parsec isochrones
    rootDirParsec = models_dir + 'ParsecV1.2s/iso'
    metalSuffix = '/z015'
    if metallicity != 'solar':
        print 'Non-solar metallicities not supported yet'
        return        
    rootDirParsec += metalSuffix + '/'

    # Search both directories for iso_*.dat files
    isoFilesE = glob.glob(rootDirE + 'iso_*.dat')
    isoFilesPi = glob.glob(rootDirPisa + 'iso_*.dat')
    isoFilesPa = glob.glob(rootDirParsec + 'iso*')

    # Output of merged isochrones
    outDir = models_dir + 'merged/pisa_ekstrom_parsec/%s/' % (metSuffix)
    gcutil.mkdir(outDir)

    # Isolate the iso*.dat file names
    for ii in range(len(isoFilesE)):
        isoFilesE[ii] = isoFilesE[ii].split('/')[-1]

    for ii in range(len(isoFilesPi)):
        isoFilesPi[ii] = isoFilesPi[ii].split('/')[-1]

    for ii in range(len(isoFilesPa)):
        isoFilesPa[ii] = isoFilesPa[ii].split('/')[-1]

    # Go through the Pisa isochrones, will merge with different models based on age
    for ii in range(len(isoFilesPi)):
        isoFilePi = isoFilesPi[ii]

        logAgeStr = isoFilePi.replace('iso_', '').replace('.dat', '')
        logAge = float(logAgeStr)

        # Case where logAge <= 7.4, we merge with Ekstrom. Otherwise, merge
        # which parsec
        if logAge <= 7.4:
            if isoFilePi not in isoFilesE:
                print 'Skipping isochrones from ', isoFilePi
                continue

            print 'Merging isochrones Pisa+Ekstrom from ', isoFilePi
            iso = merge_isochrone_pisa_Ekstrom(logAge, metallicity=metallicity,
                                               rotation=rotation)
        else:
            if isoFilePi not in isoFilesPa:
                print 'Skipping isochrones from ', isoFilePi
                continue
                
            print 'Merging isochrones Pisa+Parsec from ', isoFilePi
            iso = merge_isochrone_pisa_parsec(logAge, metallicity=metallicity)


        # Make test plot, if desired. These are put in plots directory
        if plot:
            # Make different models different colors
            pisa_ind = np.where(iso.source == 'Pisa')
            other_ind = np.where(iso.source != 'Pisa')
            #Extract age
            logAge = isoFilePi.split('_')[1][:-4]
            py.figure(1)
            py.clf()
            py.plot(iso.logT[pisa_ind], iso.logL[pisa_ind], 'r-', label = 'Pisa',
                    linewidth=2)
            py.plot(iso.logT[other_ind], iso.logL[other_ind], 'b-',
                    label = 'Ekstrom/Parsec', linewidth=2)
            py.xlabel('log Teff')
            py.ylabel('log L')
            py.title('Log Age = {0:s}'.format(logAge))
            py.legend(loc=3)
            py.axis([4.9, 3.2, -1.5, 8])
            py.savefig('plots/iso_'+logAge+'.png')
            
        _out = open(outDir + isoFilePi, 'w')

        _out.write('%12s  %10s  %10s  %10s %10s %-10s\n' % 
                   ('# M_init', 'log T', 'log L', 'log g', 'logT_WR', 'Source'))
        _out.write('%12s  %10s  %10s  %10s  %10s %-10s\n' % 
                   ('# (Msun)', '(Kelvin)', '(Lsun)', '(cgs)', '(Kelvin)', '()'))

        for kk in range(len(iso.mass)):
            _out.write('%12.6f  %10.4f  %10.4f  %10.4f  %10.4f %-10s\n' %
                       (iso.mass[kk], iso.logT[kk], iso.logL[kk],
                        iso.logg[kk], iso.logT_WR[kk], iso.source[kk]))

        _out.close()
    return

