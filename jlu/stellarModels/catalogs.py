import pyfits
import asciidata
import glob
import numpy as np
import pdb

def reformat_nextgen():
    """
    Reformat the NextGen model atmosphere catalog into an acceptable
    CDBS format. This includes a catalog.fits file 
    """
    nextgen_old_dir = '/u/jlu/work/models/nextgen/'
    nextgen_new_dir = '/u/jlu/work/cdbs/grid/nextgen/'

    # Read the names of all the NextGen spectra files and get their 
    # parameters (Teff, mH, logg, filename).
    files = glob.glob(nextgen_old_dir + 'lte*.spec')

    fname = []
    params = []
    newfile = []
    newfile2 = []

    for ff in files:
        filename = ff.replace(nextgen_old_dir, '')
        if not filename.startswith('lte'):
            continue

        paramStr = filename[3:].replace('.NextGen.spec', '')
        paramParts = paramStr.split('-')

        temp = float(paramParts[0]) * 100.0
        grav = float(paramParts[1])
        metal = float(paramParts[2]) * -1.0

        fname.append(ff)

        params.append( '%d,%.1f,%.1f' % (temp, metal, grav) )
        newfile.append( '%s/nextgen_%s.fits' % 
                        (nextgen_new_dir, paramStr) )
        newfile2.append( 'nextgen_%s.fits[flux]' % (paramStr) )

    # Make a catalog.fits file. Remember that everything with the same 
    # effective temperature and metallicity will go in the same FITS file.
    # We will model the sub-directories just like Kurucz (one sub-dir per
    # metallicity, one FITS file per temperature, one column per gravity).
    
    # The catalog.fits file will have two columns.
    #   Col 1: 'Teff, logg, mH'
    #   Col 2: 'filename'
    cat1 = pyfits.Column(name='INDEX', format='A20', array=params)
    cat2 = pyfits.Column(name='FILENAME', format='A100', array=newfile2)
    catCols = pyfits.ColDefs([cat1, cat2])
    catHDU = pyfits.new_table(catCols)
    catHDU.writeto('%s/catalog.fits' % (nextgen_new_dir), clobber=True)

    for ff in range(len(newfile)):
        # Make the new fits file.... remember this is trimmed down
        # sorted, and duplicates removed. (0.2 - 10 microns)
        wave, flux = load_nextgen_spectrum(fname[ff])
        
        col1 = pyfits.Column(name='wavelength', format='1E', unit='ANGSTROM',
                             array=wave)
        col2 = pyfits.Column(name='flux', format='1E', unit='FLAM',
                             array=flux)

        cols = pyfits.ColDefs([col1, col2])
        
        tbhdu = pyfits.new_table(cols)
        tbhdu.writeto(newfile[ff], clobber=True)


def load_nextgen_spectrum(filename):
    _spec = open(filename, 'r')

    # First two lines contain header info
    line1 = _spec.readline()
    line2 = _spec.readline()
    npoints = int(line2.split()[0])

    rest = _spec.read()
    rest = rest.split()

    wave = np.array(rest[0:npoints], dtype=float)
    flux = np.array(rest[npoints:(2*npoints)], dtype=float)

    _spec.close()

    # Get rid of duplicates
    diff = np.diff(wave)
    idx = np.where(diff != 0)[0]
    wave = wave[idx]
    flux = flux[idx]

    # Trim down to just 0.2 - 10 microns (wavelength is in Angstroms)
    idx = np.where((wave >= 2000) & (wave <= 10**5))[0]
    wave = wave[idx]
    flux = flux[idx]

    # Convert wavelength from erg/s/cm^2/cm to erg/s/cm^2/A
    flux /= 10**8

    # Sort
    sdx = wave.argsort()
    wave = wave[sdx]
    flux = flux[sdx]

    return wave, flux


def reformat_amesdusty():
    """
    Reformat the AMES-Dusty model atmosphere catalog into an acceptable
    CDBS format. This includes a catalog.fits file 
    """
    ames_old_dir = '/u/jlu/work/models/AMESdusty/SPECTRA/'
    ames_new_dir = '/u/jlu/work/cdbs/grid/AMESdusty/'

    # Read the names of all the NextGen spectra files and get their 
    # parameters (Teff, mH, logg, filename).
    files = glob.glob(ames_old_dir + 'lte*.AMES-dusty.7')

    fname = []
    params = []
    newfile = []
    newfile2 = []

    for ff in files:
        filename = ff.replace(ames_old_dir, '')
        if not filename.startswith('lte'):
            continue

        paramStr = filename[3:].replace('.AMES-dusty.7', '')
        paramParts = paramStr.split('-')

        temp = float(paramParts[0]) * 100.0
        grav = float(paramParts[1])
        metal = float(paramParts[2]) * -1.0

        fname.append(ff)

        params.append( '%d,%.1f,%.1f' % (temp, metal, grav) )
        newfile.append( '%s/ames_%s.fits' % 
                        (ames_new_dir, paramStr) )
        newfile2.append( 'ames_%s.fits[flux]' % (paramStr) )

    # Make a catalog.fits file. Remember that everything with the same 
    # effective temperature and metallicity will go in the same FITS file.
    # We will model the sub-directories just like Kurucz (one sub-dir per
    # metallicity, one FITS file per temperature, one column per gravity).
    
    # The catalog.fits file will have two columns.
    #   Col 1: 'Teff, logg, mH'
    #   Col 2: 'filename'
    cat1 = pyfits.Column(name='INDEX', format='A20', array=params)
    cat2 = pyfits.Column(name='FILENAME', format='A100', array=newfile2)
    cols = pyfits.ColDefs([cat1, cat2])
    catHDU = pyfits.new_table(cols)
    catHDU.writeto('%s/catalog.fits' % (ames_new_dir), clobber=True)

    for ff in range(len(newfile)):
        # Make the new fits file.... remember this is trimmed down
        # sorted, and duplicates removed. (0.2 - 10 microns)
        wave, flux = load_amesdusty_spectrum(fname[ff])
        
        col1 = pyfits.Column(name='wavelength', format='1E', unit='ANGSTROM',
                             array=wave)
        col2 = pyfits.Column(name='flux', format='1E', unit='FLAM',
                             array=flux)

        cols = pyfits.ColDefs([col1, col2])
        
        tbhdu = pyfits.new_table(cols)
        tbhdu.writeto(newfile[ff], clobber=True)


def load_amesdusty_spectrum(filename):
    _spec = open(filename, 'r')

    wave = []
    flux = []

    for line in _spec.readlines():
        if line[0] != ' ' or 'D ' in line:
            continue

        parts = line.split()

        if len(parts) <= 3:
            continue

        wave.append(parts[0])
        flux.append(parts[1])
        
    wave = np.array(wave)
    flux = np.array(flux)

    # Check that these are floats
    if type(wave[0]) == np.string_:
        wave = np.array([float(ww.replace('D', 'E')) for ww in wave])
        flux = np.array([float(ff.replace('D', 'E')) for ff in flux])

    # Trim down to just 0.2 - 10 microns
    idx = np.where((wave >= 2000) & (wave <= 10**5))[0]
    wave = wave[idx]
    flux = flux[idx]

    # Get rid of duplicates
    diff = np.diff(wave)
    idx = np.where(diff != 0)[0]
    wave = wave[idx]
    flux = flux[idx]

    # Sort
    sdx = wave.argsort()
    wave = wave[sdx]
    flux = flux[sdx]

    # Wavelength is in Angstroms
    # Convert flux to erg/s/cm^2/Angstrom
    flux = 10**(flux - 8.0)

    return wave, flux
        

def clean_ck04models():
    """
    Remove all entries from catalog.fits that have no positive
    flux values. If you don't do this, then interpolation between
    spectra produces incorrect results.
    """
    ck04dir = '/u/jlu/work/models/ck04models/'

    cat = pyfits.open(ck04dir + 'catalog_orig.fits')

    tab = cat[1].data

    good = []
    for ii in range(len(tab)):
        sp_file_all = tab[ii][1]
        
        parts = sp_file_all.split('[')
        sp_file = parts[0]
        sp_grav = parts[1][:-1]

        sp_tab = pyfits.getdata(ck04dir + sp_file)

        if sp_tab.field(sp_grav).max() > 0:
            good.append(ii)
        else:
            print 'No data found for ', tab[ii]

    print 'Original catalog has %d entries' % len(tab)
    print 'New catalog has %d entries' % len(good)
    
    cat[1].data = tab.take(good)
    cat.writeto('catalog.fits', clobber=True)
