import numpy as np
from astropy.table import Table, Column
from astropy.io import fits
import pylab as py
from pysynphot import observation, spectrum
from popstar import atmospheres
import pysynphot
import os
import glob
import pdb

# Define path to evolution model isochrone from Ekstrom+12 (w/rotation)
iso_path = '/g/lu/models/evolution/Ekstrom2012/iso/z014/rot/'

def get_mid_temp_params(log_age):
    """
    Get stellar params for for mid-temperature stars (specifically, between
    4000 - 7500 K) at the defined age

    Uses iso_path to evolution model isochrone to extract stellar
    parameters in temperature range of interest. CODE ASSUMES THIS IS THE
    Ekstrom+12 ROTATING MODELS

    log_age is the age of the isochrone. Isochrone filename format should be
    iso_<log_age>.dat (I set it up that way...).   
    """
    # Read in Ekstrom+12 isochrone, isolate temp region of interest
    filename = 'iso_{0:3.2f}.dat'.format(log_age)
    iso = Table.read(iso_path + filename, format='ascii')

    lowtemp = np.log10(4000)
    hightemp = np.log10(7500)
    good = np.where( (iso['col8'] > lowtemp) & (iso['col8'] < hightemp) )
    models = iso[good]

    # Extract temp (linear), log g
    temp = 10**models['col8'] # K
    # Need to calculate log g from luminosity, mass, temp
    lum = 10**models['col7'] # L_sun
    mass = models['col6'] # M_sun
    
    lum_sun = 3.846 * 10**33 # erg/s
    M_sun = 2 * 10**33 # g
    G_si = 6.67 * 10**(-8) # cgs
    sigma_si = 5.67 * 10**(-5) # cgs
       
    g = (G_si * mass * M_sun * 4 * np.pi * sigma_si * temp**4) / \
      (lum * lum_sun)

    g = np.array(list(g))
    logg = np.log10(np.array(g))

    logg_med = np.median(logg)
    logg_std = np.std(logg,ddof=1)
    temp_med = np.median(temp)
    temp_std = np.std(temp,ddof=1)
    
    print('**********************************************************')
    print('logg stats: {0:3.2f} +/- {1:3.2f}'.format(logg_med, logg_std))
    print('Temp stats: {0:3.2f} +/- {1:3.2f}'.format(temp_med, temp_std))
    print('**********************************************************')
    pdb.set_trace()
    return

def extract_atmospheres(temp_arr, logg_val, directory):
    """
    Helper function to extract atmospheres we want. Input is an ALREADY READ-IN
    catalog file, the array of temperatures we are interested in, logg val we are
    interested in (ONE ONLY!!), and the path to the relevant catalog.fits file.

    Output is a table: col1 is the wavelength points, while subsequent columns
    are the flux for different temps (labeled)
    """
    # Extract the models, consolidate into one table
    for i in range(len(temp_arr)):
        t = pysynphot.Icat(directory, temp_arr[i], 0.0, logg_val)

        # Initialize the table if first one
        if i == 0:
            wave = t.wave
            tmp = Column(wave, name = 'Wavelength')
            models = Table()
            models.add_column(tmp)
                
        # Extract fluxes, add to table
        flux = t.flux
        z = Column(flux, name = str(temp_arr[i]))
        models.add_column(z)
        
    return models

def rebin_spec(wave, specin, wavnew):
    """
    Helper routine to rebin spectra. TAKEN FROM ASTROBETTER BLOG FROM JESSICA:
    http://www.astrobetter.com/blog/2013/08/12/
    python-tip-re-sampling-spectra-with-pysynphot/
    """
    spec = spectrum.ArraySourceSpectrum(wave=wave, flux=specin)
    f = np.ones(len(wave))
    filt = spectrum.ArraySpectralElement(wave, f, waveunits='angstrom')
    obs = observation.Observation(spec, filt, binset=wavnew, force='taper')
 
    return obs.binflux

def test_mid_temp_atmospheres(ATLAS_dir, PHOENIX_dir):
    """
    Compare [normalized] atmosphere models for mid-temperature stars:
    specifically, between 4000 - 7500 K (steps of 500 K, hardcoded).
    Logg is assumed to be 4.5 (typical for this temp range, found using
    get_mid_temp_params)

    ATLAS_dir and PHOENIX_dir paths should lead from working directory into
    the model directory where the catalog.fits file is located
    """
    # Define temperature array of interest for our models
    temp_arr = np.arange(7000, 7000+1, 500)
    #temp_arr = np.array([9000])
    logg_val = 4.5
    
    # Extract ATLAS models
    catalog = Table.read(ATLAS_dir + 'catalog.fits', format='fits')
    atlas_mods = extract_atmospheres(catalog, temp_arr, logg_val, ATLAS_dir)
    print('Extracted ATLAS models')
    
    # Extract PHOENIX models
    catalog = Table.read(PHOENIX_dir + 'catalog.fits', format='fits')
    phoenix_mods = extract_atmospheres(catalog, temp_arr, logg_val, PHOENIX_dir)
    print('Extracted PHOENIX models')

    # Trim models down to wavelength region we want: JHK (1 - 2.4 microns)
    good = np.where( (atlas_mods['Wavelength'] > 10000) &
                     (atlas_mods['Wavelength'] < 24000) )
    atlas_mods = atlas_mods[good]
    good = np.where( (phoenix_mods['Wavelength'] > 10000) &
                     (phoenix_mods['Wavelength'] < 24000) )
    phoenix_mods = phoenix_mods[good]

    # Reduce resolution of phoenix models to match atlas models
    # Also, normalize flux to 1.10 microns. Want lambda * flux,
    # rather than just flux
    atlas_mods_norm = Table()
    phoenix_mods_norm = Table()
    wave = atlas_mods['Wavelength']
    col = Column(wave, name = 'Wavelength')
    atlas_mods_norm.add_column(col)
    phoenix_mods_norm.add_column(col)

    for i in temp_arr:
        # Rebinning phoenix spectra
        flux = phoenix_mods[str(i)]
        flux_new = rebin_spec(phoenix_mods['Wavelength'], flux, wave)

        # Normalizing phoenix flux
        ind = np.where( abs(wave - 11000) == min(abs(wave - 11000)) )
        n_flux = (wave * flux_new) / (wave[ind][0] * flux_new[ind][0])    
        tmp_col = Column(n_flux, name = str(i))
        phoenix_mods_norm.add_column(tmp_col)

        # Normalizing atlas flux
        flux = atlas_mods[str(i)]
        n_flux = (wave * flux) / (wave[ind][0] * flux[ind][0])    
        tmp_col = Column(n_flux, name = str(i))
        atlas_mods_norm.add_column(tmp_col)        

    print('Making plots')
    #----------PLOTTING MODELS: Normalized--------#
    # Want to plot 3 models per plot for scaling purposes
    subtemp = np.array_split(temp_arr, 2)
    py.figure(1, figsize=(10,10))
    py.clf()
    # For each plot, offset by n
    n = -0.8
    for i in subtemp[0]:
        n += 0.5
        # Only include the labels in the first set
        if i == min(subtemp[0]):
            py.plot(wave, atlas_mods_norm[str(i)] + n, 'k-',
                label='ATLAS (ck04)')
            py.plot(wave, phoenix_mods_norm[str(i)] + n, 'r-',
                label='PHOENIX (Husser+13)')
            py.annotate(str(i)+' K', xy=(11000, 0.6))
        else:
            py.plot(wave, atlas_mods_norm[str(i)] + n, 'k-')
            py.plot(wave, phoenix_mods_norm[str(i)] + n, 'r-')
            py.annotate(str(i)+' K', xy=(11000, 0.8+n))
    py.xlabel(r'$\lambda$ (Angstroms)')
    py.ylabel(r'$\lambda$f$_\lambda$ / $\lambda$f$_\lambda$ (1.10 $\mu$m) + const')
    py.title('ATLAS vs. PHOENIX, {0:4.0f} K - {1:4.0f} K'.format(min(subtemp[0]),
                                                             max(subtemp[0])))
    py.legend()
    py.axis([9000, 25000, 0, 2.5])
    py.savefig('ATLASvPHOENIX_comp3.png')

    py.figure(2, figsize=(10,10))
    py.clf()
    # For each plot, offset by n
    n = -0.7
    for i in subtemp[1]:
        n += 0.5
        # Only include the labels in the first set
        if i == min(subtemp[1]):
            py.plot(wave, atlas_mods_norm[str(i)] + n, 'k-',
                label='ATLAS (ck04)')
            py.plot(wave, phoenix_mods_norm[str(i)] + n, 'r-',
                label='PHOENIX (Husser+13)')
            py.annotate(str(i)+' K', xy=(11000, 0.6))
        else:
            py.plot(wave, atlas_mods_norm[str(i)] + n, 'k-')
            py.plot(wave, phoenix_mods_norm[str(i)] + n, 'r-')
            py.annotate(str(i)+' K', xy=(11000, 0.6+n))
    py.xlabel(r'$\lambda$ (Angstroms)')
    py.ylabel(r'$\lambda$f$_\lambda$ / $\lambda$f$_\lambda$ (1.10 $\mu$m) + const')
    py.title('ATLAS vs. PHOENIX, {0:4.0f} K - {1:4.0f} K'.format(min(subtemp[1]),
                                                             max(subtemp[1])))
    py.legend()
    py.axis([9000, 25000, 0, 2.2])
    py.savefig('ATLASvPHOENIX_comp4.png')    
    
    pdb.set_trace()
    return

def test_BTSettl_Phoenix(BTSettl_mod='BTSettl_2015_rebin',
                         PHOENIX_mod='phoenix_v16_rebin'):
    """
    Compare [normalized] atmosphere models for mid-temperature stars:
    specifically, between 4000 - 7500 K (steps of 500 K, hardcoded).
    Logg is assumed to be 4.5 (typical for this temp range, found using
    get_mid_temp_params)

    CODE ASSUMES WAVELENGTH ARRAYS ARE THE SAME FOR BOTH SETS OF MODELS
    
    BTSettl_mod and PHOENIX_mod paths should be names of the cdbs model
    directories we want to use
    """
    # Define temperature array of interest for our models
    temp_arr = np.arange(3000, 7000+1, 1000)
    logg_val = 4.5
    
    # Extract BTSettl models
    btsettl_mods = extract_atmospheres(temp_arr, logg_val, BTSettl_mod)
    print('Extracted BTSettl models')

    # Extract PHOENIX models
    phoenix_mods = extract_atmospheres(temp_arr, logg_val, PHOENIX_mod)
    print('Extracted PHOENIX models')

    # Trim models down to wavelength region we want: JHK (1 - 2.4 microns)
    good = np.where( (btsettl_mods['Wavelength'] > 10000) &
                     (btsettl_mods['Wavelength'] < 24000) )
    btsettl_mods = btsettl_mods[good]
    good = np.where( (phoenix_mods['Wavelength'] > 10000) &
                     (phoenix_mods['Wavelength'] < 24000) )
    phoenix_mods = phoenix_mods[good]

    # Extract wavelength array, since they are the same
    wave = btsettl_mods['Wavelength']
    idx = np.where( abs(wave - 11000) == min(abs(wave - 11000)) )[0][0]
    wavecol = Column(wave, name='Wavelength')
     
    btsettl_mods_norm = Table()
    phoenix_mods_norm = Table()
    btsettl_mods_norm.add_column(wavecol)
    phoenix_mods_norm.add_column(wavecol)
    
    for i in temp_arr:
        # Normalizing BTSettl flux
        flux = btsettl_mods[str(i)]
        norm_flux = (wave * flux) / (wave[idx] * flux[idx])    
        tmp_col = Column(norm_flux, name = str(i))
        btsettl_mods_norm.add_column(tmp_col)

        # Normalizing phoenix flux
        flux = phoenix_mods[str(i)]
        norm_flux = (wave * flux) / (wave[idx] * flux[idx])    
        tmp_col = Column(norm_flux, name = str(i))
        phoenix_mods_norm.add_column(tmp_col)        

    print('Making plots')
    #----------PLOTTING MODELS: Normalized--------#
    # Want to plot 3 models per plot for scaling purposes
    subtemp = np.array_split(temp_arr, 2)
    py.figure(1, figsize=(10,10))
    py.clf()
    # For each plot, offset by n
    n = -0.8
    for i in subtemp[0]:
        n += 0.5
        # Only include the labels in the first set
        if i == min(subtemp[0]):
            py.plot(wave, btsettl_mods_norm[str(i)] + n, 'k-',
                label='BTSettl_2015')
            py.plot(wave, phoenix_mods_norm[str(i)] + n, 'r-',
                label='PHOENIX (Husser+13)')
            py.annotate(str(i)+' K', xy=(11000, 0.6))
        else:
            py.plot(wave, btsettl_mods_norm[str(i)] + n, 'k-')
            py.plot(wave, phoenix_mods_norm[str(i)] + n, 'r-')
            py.annotate(str(i)+' K', xy=(11000, 0.8+n))
    py.xlabel(r'$\lambda$ (Angstroms)')
    py.ylabel(r'$\lambda$f$_\lambda$ / $\lambda$f$_\lambda$ (1.10 $\mu$m) + const')
    py.title('BTSettl vs. PHOENIX, {0:4.0f} K - {1:4.0f} K'.format(min(subtemp[0]),
                                                             max(subtemp[0])))
    py.legend()
    py.axis([9000, 25000, 0, 2.0])
    py.savefig('BTSettlvPHOENIX_comp.png')

    py.figure(2, figsize=(10,10))
    py.clf()
    # For each plot, offset by n
    n = -0.7
    for i in subtemp[1]:
        n += 0.5
        # Only include the labels in the first set
        if i == min(subtemp[1]):
            py.plot(wave, btsettl_mods_norm[str(i)] + n, 'k-',
                label='BTSettl_2015')
            py.plot(wave, phoenix_mods_norm[str(i)] + n, 'r-',
                label='PHOENIX (Husser+13)')
            py.annotate(str(i)+' K', xy=(11000, 0.6))
        else:
            py.plot(wave, btsettl_mods_norm[str(i)] + n, 'k-')
            py.plot(wave, phoenix_mods_norm[str(i)] + n, 'r-')
            py.annotate(str(i)+' K', xy=(11000, 0.6+n))
    py.xlabel(r'$\lambda$ (Angstroms)')
    py.ylabel(r'$\lambda$f$_\lambda$ / $\lambda$f$_\lambda$ (1.10 $\mu$m) + const')
    py.title('BTSettl vs. PHOENIX, {0:4.0f} K - {1:4.0f} K'.format(min(subtemp[1]),
                                                             max(subtemp[1])))
    py.legend()
    py.axis([9000, 25000, 0, 1.6])
    py.savefig('BTSettlvPHOENIX_comp2.png')    
    
    pdb.set_trace()
    return

def pysynphot_test(model, temp, metal, grav):
    """
    Extract model atmosphere from cdbs database using pysynphot.

    Inputs:
    model: name of model directory in cdbs directory (i.e. 'ck04models')
    metal: metallicity of model
    temp: temperature of model
    grav: gravity of model
    """
    # Extract atmosphere with pysynphot
    sp = pysynphot.Icat(model, temp, metal, grav)

    wave = sp.wave
    wave_units = str(sp.waveunits)
    flux = sp.flux
    flux_units = str(sp.fluxunits)

    # Trim spectrum to IR wavelengths (1 - 2.5 microns)
    if wave_units == 'angstrom':
        minLim = 10000
        maxLim = 25000
    elif wave_units == 'micron':
        minLim = 1.0
        maxLim = 2.5
    
    good = np.where( (wave > minLim) & (wave < maxLim))
    wave = wave[good]
    flux = flux[good]
    
    # Plot at IR wavelengths (1 - 2.5 microns)
    py.figure(1, figsize=(10,10))
    py.clf()
    py.plot(wave, flux, 'k-')
    py.xlabel('Wavelength ({0:s})'.format(wave_units))
    py.ylabel(r'Flux ({0:s})'.format(flux_units))
    py.title(model)
    py.savefig('test_{0:s}.png'.format(model))
    

    pdb.set_trace()
    return

def make_merged_catalog(prefix_arr, temp_arr, logg_arr, metallicity_arr):
    """
    Helper function that creates the catalog.fits file for the merged
    model atmospheres. NOTE: Merged atmosphere temp should be first in the
    temp_arr

    Assumes that each temperature + metallicity has a separate filename with
    name mergedm<metal>_<temp>.fits. Also returns filename_arr for convinience.
    """
    # Loop through temp, metallicity, then logg
    index_arr = []
    filename_arr = []
    for temp in temp_arr:
        for ii in range(len(metallicity_arr)):
            metal = metallicity_arr[ii]
            prefix = prefix_arr[ii]
            for logg in logg_arr:
                # Change logg_arr name for atlas
                if temp == 5500:
                    name = str(logg).split('.')
                    logg_s = 'g'+name[0]+name[1]
                    #filename = 'mergedm{0:2.1f}_{1:05.0f}.fits[{2}]'.format(metal,
                    #                                                         temp,
                    #                                                         logg_s)
                    filename = '{0}/{0}_{1:05.0f}.fits[{2}]'.format(prefix, temp, logg_s)
                    index = '{0:5.0f},{1:2.1f},{2:2.1f}'.format(temp, metal, logg)
                else:
                    #filename = 'mergedm{0:2.1f}_{1:05.0f}.fits[g{2:2.1f}]'.format(metal,
                    #                                                         temp,
                    #                                                         logg)
                    filename = '{0}/{0}_{1:05.0f}.fits[g{2:2.1f}]'.format(prefix, temp,
                                                                          logg)
                    index = '{0:5.0f},{1:2.1f},{2:2.1f}'.format(temp, metal, logg)

                filename_arr.append(filename)
                index_arr.append(index)

    catalog = Table([index_arr, filename_arr], names=('INDEX', 'FILENAME'))

    return catalog, filename_arr

def make_merged_header(tablehdu):
    """
    Helper function that creates table header for merged atmosphere. This
    is the phoenix table header with "TUNIT" keyword added.

    Assumes wavelengths are in angstroms, fluxees in pysynphot FLAM units
    """
    tablehdu.header['TUNIT1'] = 'ANGSTROM'
    tablehdu.header['TUNIT2'] = 'FLAM'
    tablehdu.header['TUNIT3'] = 'FLAM'
    tablehdu.header['TUNIT4'] = 'FLAM'
    tablehdu.header['TUNIT5'] = 'FLAM'
    tablehdu.header['TUNIT6'] = 'FLAM'
    tablehdu.header['TUNIT7'] = 'FLAM'
    tablehdu.header['TUNIT8'] = 'FLAM'
    tablehdu.header['TUNIT9'] = 'FLAM'
    tablehdu.header['TUNIT10'] = 'FLAM'
    tablehdu.header['TUNIT11'] = 'FLAM'
    tablehdu.header['TUNIT11'] = 'FLAM'
    tablehdu.header['TUNIT12'] = 'FLAM'
    
    
    return tablehdu

def create_merged_models(cdbs_path, plot=False):
    """
    From 5000 K - 5500 K, merge the ATLAS (ck04) and PHOENIX (v16) atmospheres.
    More like ATLAS near 5500 K, more like PHOENIX near 5000 K

    cdbs_path is path to cdbs directory (including cdbs). Will make new directory
    in cdbs/grid named "merged_atlas_phoenix" with with merged models. If plot = True, will plot
    the normalized merged model plus the original phoenix and atlas models

    Temp 5000 - 5500, steps of 250; logg 0 - 5, steps of 0.5, metallicity covering
    ATLAS range (-2.5 -- 0.5, in steps of 0.5). So, this is one model at 5250

    Note: metallicity directories created by hand
    
    Creates new directory "merged_atlas_phoenix" in cdbs/grid with new spectrum + catalog file.
    Also includes the atlas 5500K model and phoenix 5000K model, for interpolation purposes
    """
    #Setting logg sampling
    logg_arr = np.arange(0, 5+0.1, 0.5)

    # Setting metallicity sub-directories
    atlas_dir = ['ckm25', 'ckm20', 'ckm15', 'ckm10', 'ckm05', 'ckp00', 'ckp02', 'ckp05']
    phoenix_dir = ['phoenixm30', 'phoenixm20', 'phoenixm15', 'phoenixm10', 'phoenixm05', 'phoenixm00', 'phoenixp05', 'phoenixp05']
    output_dir = ['mergedm25', 'mergedm20', 'mergedm15', 'mergedm10', 'mergedm05', 'mergedp00', 'mergedp02', 'mergedp05']
    metal_arr = [-2.5, -2.0, -1.5, -1.0, -0.5, 0, 0.2, 0.5]
    
    assert len(atlas_dir) == len(phoenix_dir) == len(output_dir)

    # Make new cdbs merged directory, if it doesn't already exist
    newPath = '{0}/grid/merged_atlas_phoenix'.format(cdbs_path)
    if not os.path.exists(newPath):
        os.mkdir(newPath)

    # For each metallicity, create merged model
    for ii in range(len(output_dir)):
        # Make metallicity dir for merged model
        final_dir = '{0}/{1}'.format(newPath, output_dir[ii])
        if not os.path.exists(final_dir):
            os.mkdir(final_dir)
        
        # Extract the relevant ATLAS and PHEONIX models near 5250 K
        atlas_path = '{0}/grid/ck04models/{1}/'.format(cdbs_path, atlas_dir[ii])
        atlas_hdu = fits.open('{0}/{1}_5250.fits'.format(atlas_path, atlas_dir[ii]))
        atlas_5250 = atlas_hdu[1].data
        phoenix_path = '{0}/grid/phoenix_v16_rebin/{1}'.format(cdbs_path, phoenix_dir[ii])
        phoenix_hdu1 = fits.open('{0}/{1}_05200.fits'.format(phoenix_path, phoenix_dir[ii]))
        phoenix_hdu2 = fits.open('{0}/{1}_05300.fits'.format(phoenix_path, phoenix_dir[ii]))
        phoenix_5200 = phoenix_hdu1[1].data
        phoenix_5300 = phoenix_hdu2[1].data
        print('Done reading input models')

        # Trim both models to the wavelength region we want: VRIJHKL (0.25 - 4.2 mircons)
        good = np.where( (atlas_5250['Wavelength'] > 2500) &
                        (atlas_5250['Wavelength'] < 52000) )
        atlas_5250 = atlas_5250[good]
        good = np.where( (phoenix_5200['Wavelength'] > 2500) &
                        (phoenix_5200['Wavelength'] < 52000) )
        phoenix_5200 = phoenix_5200[good]
        phoenix_5300 = phoenix_5300[good]

        #----------------------------#
        # For each logg val, create phoenix spectrum for 5250 K,
        # degrade resolution to match atlas, then average with
        # 5250 K atlas model to get merged model at 5250 K
        #----------------------------#
        new_model = []
        phoenix_5250_f = []
        atlas_5250_f = []
        for logg in logg_arr:
            # Create phoenix models at 5250 K
            low = phoenix_5200['g{0:2.1f}'.format(logg)]
            high = phoenix_5300['g{0:2.1f}'.format(logg)]

            arr = np.transpose([low, high])
            phoenix_5250 = np.mean(arr, axis=1)

            phoenix_5250_rebin = rebin_spec(phoenix_5200['Wavelength'], phoenix_5250,
                                            atlas_5250['Wavelength'])

            # Store phoenix rebinned average spectrum
            phoenix_5250_f.append(phoenix_5250_rebin)

            # Store atlas spectrum for later
            grav = str(logg).split('.')        
            atlas_5250_f.append(atlas_5250['g'+grav[0]+grav[1]])
        
            # Now, final 5250 K model will be average of atlas and phoenix
            # models (since exactly inbetween 5000 K and 5500 K)
            arr = np.transpose([phoenix_5250_rebin, atlas_5250['g'+grav[0]+grav[1]]])
            final = np.mean(arr, axis=1)

            new_model.append(final)
            print('Done with logg {0:2.1f}'.format(logg))

        # Create fits table with new model. First column is wavelength, followed by
        # fluxes for different logg
        wave = atlas_5250['Wavelength'] # Still same wavelength array as before
        c0 = fits.Column(name='Wavelength', format='D', array=wave)
        c1 = fits.Column(name='g0.0', format='E', array=new_model[0])
        c2 = fits.Column(name='g0.5', format='E', array=new_model[1])
        c3 = fits.Column(name='g1.0', format='E', array=new_model[2])
        c4 = fits.Column(name='g1.5', format='E', array=new_model[3])
        c5 = fits.Column(name='g2.0', format='E', array=new_model[4])
        c6 = fits.Column(name='g2.5', format='E', array=new_model[5])
        c7 = fits.Column(name='g3.0', format='E', array=new_model[6])
        c8 = fits.Column(name='g3.5', format='E', array=new_model[7])
        c9 = fits.Column(name='g4.0', format='E', array=new_model[8])
        c10 = fits.Column(name='g4.5', format='E', array=new_model[9])
        c11 = fits.Column(name='g5.0', format='E', array=new_model[10])

        cols = fits.ColDefs([c0,c1,c2,c3,c4,c5,c6,c7,c8,c9,c10,c11])
        tbhdr = phoenix_hdu1[1].header
        prihdr = phoenix_hdu1[0].header 
        prihdu = fits.PrimaryHDU(header=prihdr) 
        tbhdu = fits.BinTableHDU.from_columns(cols, header=tbhdr)
        # Add TUNIT keysto tbhdu
        tbhdu = make_merged_header(tbhdu)

        # Make final hdu table, save it
        finalhdu = fits.HDUList([prihdu, tbhdu])
        finalhdu.writeto('{0}/{1}_05250.fits'.format(final_dir, output_dir[ii]), clobber=True)
        
        # Test plot, if desired
        if plot == True:
            py.figure(1, figsize=(10,10))
            py.clf()
            # Plot merged model
            py.semilogy(wave, wave * new_model[8], 'r-', label='Merged')
            # Plot atlas 5250 K model
            py.semilogy(wave, wave * atlas_5250_f[8], 'b-', label = 'Atlas')
            # Plot phoenix 5250 K model
            py.semilogy(wave, wave * phoenix_5250_f[8], 'g-', label = 'Phoenix')
            py.legend()
            py.xlabel('Wavelength (Angstrom)')
            py.ylabel(r'log ($\lambda$ F$_{\lambda}$)')
            py.axis([5000, 40000, 2*10**9, 4*10**10])
            py.title('Merged 5250 K Spectrum')
            py.savefig('Merged_atlas_phoenix_{0}.png'.format(output_dir[ii]))
            
            pdb.set_trace()
            py.close('all')

        # Copy the atlas 5500 K and phoenix 5000 K models into the
        # merged directory to accompany the merged model. This is for
        # interpolation purposes for pysynphot
        cmd = 'cp {0}/{1}_5500.fits {2}'.format(atlas_path, atlas_dir[ii], final_dir)
        cmd2 = 'cp {0}/{1}_05000.fits {2}'.format(phoenix_path, phoenix_dir[ii], final_dir)
        os.system(cmd)
        os.system(cmd2)

        cmd = 'mv {0}/{1}_5500.fits {0}/{2}_05500.fits'.format(final_dir, atlas_dir[ii], output_dir[ii])
        cmd2 = 'mv {0}/{1}_05000.fits {0}/{2}_05000.fits'.format(final_dir, phoenix_dir[ii], output_dir[ii])
        os.system(cmd)
        os.system(cmd2)
        
        # Close all open hdu
        atlas_hdu.close()
        phoenix_hdu1.close()
        phoenix_hdu2.close()

        print('Done {0}'.format(output_dir[ii]))
    
    # Make catalog.fits table for new merged model, plus the atlas and
    # phoenix models at the high and low extremes of the temp range. Note temp
    # of merged model goes first, for filename purposes
    catalog, filenames = make_merged_catalog(output_dir, [5250, 5000, 5500], logg_arr, metal_arr)
    catalog.write(cdbs_path+'/grid/merged_atlas_phoenix/catalog.fits', overwrite=True)

    return

def create_merged_models2(cdbs_path, plot=False):
    """
    From 3200 K - 3800 K, merge the BTSettl_CIFITS2011_2015 and PHOENIX (v16) atmospheres.
    Phoenix at 3800 K, BTSettl near 3200 K

    cdbs_path is path to cdbs directory (including cdbs). Will make new directory
    in cdbs/grid named "merged_btettl_phoenix" with with merged models. If plot = True, will plot
    the normalized merged model plus the original models

    Only 1 truely merged model at 3500 K, with log g from 2.5 - 5.5
    
    Creates new directory "merged_btsettl_phoenix" in cdbs/grid with new spectrum + catalog file.
    Also includes the atlas 5500K model and phoenix 5000K model, for interpolation purposes
    """
    # Set log g sampling
    logg_solar_arr = np.arange(2.5, 5.5+0.1, 0.5)
    logg_nonsolar_arr = np.array([4.5, 5.0, 5.5])

    # Setting metallicity sub-directories
    phoenix_dir = ['phoenixm30', 'phoenixm20', 'phoenixm15', 'phoenixm10', 'phoenixm05', 'phoenixm00', 'phoenixp05', 'phoenixp05']
    output_dir = ['mergedm25', 'mergedm20', 'mergedm15', 'mergedm10', 'mergedm05', 'mergedp00', 'mergedp02', 'mergedp05']
    metal_arr = [-2.5, -2.0, -1.5, -1.0, -0.5, 0, 0.2, 0.5]
    
    assert len(phoenix_dir) == len(output_dir)

    # Make new cdbs merged directory, if it doesn't already exist
    newPath = '{0}/grid/merged_BTSettl_phoenix'.format(cdbs_path)
    if not os.path.exists(newPath):
        os.mkdir(newPath)

   # For each metallicity, create merged model
    for ii in range(len(output_dir)):
        # Make metallicity dir for merged model
        final_dir = '{0}/{1}'.format(newPath, output_dir[ii])
        if not os.path.exists(final_dir):
            os.mkdir(final_dir)

        # For each gravity, extract BTSettl and phoenix models at 3500 K and
        # average them to get the merged model. Also write BTSettl models at
        # 3200 K and phoenix models at 3800 K
        if metal_arr[ii] == 0:
            logg_tmp = logg_solar_arr
            BT_func = atmospheres.get_BTSettl_2015_atmosphere
        else:
            BT_func = atmospheres.get_BTSettl_atmosphere
            logg_tmp = logg_nonsolar_arr

        for jj in logg_tmp:
            BT_atmo = BT_func(temperature=3500, gravity=jj, rebin=True)
            phoenix_atmo = atmospheres.get_phoenixv16_atmosphere(temperature=3500, gravity=jj, rebin=True)

            # Check to make sure wavelengths are the same between atmospheres
            diff = BT_atmo.wave - phoenix_atmo.wave
            if np.sum(diff > 0):
                print('Wavelength mismatch problem!')
                pdb.set_trace()

            merged_flux = np.mean(np.array([BT_atmo.flux, phoenix_atmo.flux]), axis=0)
            
            # Create fits file with new atmosphere
            c0 = fits.Column(name='Wavelength', format='D', array=BT_atmo.wave)
            c1 = fits.Column(name='Flux', format='E', array=merged_flux)

            cols = fits.ColDefs([c0, c1])
            tbhdu = fits.BinTableHDU.from_columns(cols)

            prihdu = fits.PrimaryHDU()
            tbhdu.header['TUNIT1'] = 'ANGSTROM'
            tbhdu.header['TUNIT2'] = 'FLAM'
            hdu_new = fits.HDUList([prihdu, tbhdu])

            # Save fits file to merged BTSettl-Phoenix direcotry
            hdu_new.writeto('{0}/{1}_03500_{2}.fits'.format(final_dir, output_dir[ii], jj), overwrite=True)

            # Make test plot, if desired
            if plot:
                py.figure(1, figsize=(10,10))
                py.clf()
                py.semilogy(BT_atmo.wave, BT_atmo.wave * merged_flux, 'r-', label='Merged')
                py.semilogy(BT_atmo.wave, BT_atmo.wave * BT_atmo.flux, 'b-', label = 'BTSettl')
                py.semilogy(phoenix_atmo.wave, phoenix_atmo.wave * phoenix_atmo.flux, 'g-', label = 'Phoenix')
                py.legend()
                py.xlabel('Wavelength (Angstrom)')
                py.ylabel(r'log ($\lambda$ F$_{\lambda}$)')
                py.xlim(5000, 40000)
                py.ylim(10**8, 10**10)
                py.title('Merged 3500 K Spectrum, Z = {1}, logg = {0}'.format(jj, metal_arr[ii]))
                py.savefig('Merged_BTSettl_phoenix_{1}_{0}.png'.format(jj, metal_arr[ii]))

            # Now to bring over the 3200 K BTSettl and 3800 K phoenix models
            BT_atmo = BT_func(temperature=3200, gravity=jj, rebin=True)
            phoenix_atmo = atmospheres.get_phoenixv16_atmosphere(temperature=3800, gravity=jj, rebin=True)

            # Create new fits files for these as well
            c0_b = fits.Column(name='Wavelength', format='D', array=BT_atmo.wave)
            c1_b = fits.Column(name='Flux', format='E', array=BT_atmo.flux)
            c0_p = fits.Column(name='Wavelength', format='D', array=phoenix_atmo.wave)
            c1_p = fits.Column(name='Flux', format='E', array=phoenix_atmo.flux)
        
            cols_b = fits.ColDefs([c0_b, c1_b])
            tbhdu_b = fits.BinTableHDU.from_columns(cols_b)
            cols_p = fits.ColDefs([c0_p, c1_p])
            tbhdu_p = fits.BinTableHDU.from_columns(cols_p)
            
            prihdu = fits.PrimaryHDU()
            tbhdu_b.header['TUNIT1'] = 'ANGSTROM'
            tbhdu_b.header['TUNIT2'] = 'FLAM'
            tbhdu_p.header['TUNIT1'] = 'ANGSTROM'
            tbhdu_p.header['TUNIT2'] = 'FLAM'
            
            hdu_newb = fits.HDUList([prihdu, tbhdu_b])
            hdu_newp = fits.HDUList([prihdu, tbhdu_p])
            
            # Save fits file to merged BTSettl-Phoenix direcotry
            hdu_newb.writeto('{0}/{1}_03200_{2}.fits'.format(final_dir, output_dir[ii], jj), clobber=True)
            hdu_newp.writeto('{0}/{1}_03800_{2}.fits'.format(final_dir, output_dir[ii], jj), clobber=True)
            hdu_new.close()
            hdu_newb.close()
            hdu_newp.close()
            
        print('Done {0}'.format(output_dir[ii]))
            
    return

def make_catalog_merged_models2(path='/g/lu/models/cdbs/grid/merged_BTSettl_phoenix/'):
    """
    Make cdbs catalog.fits file for merged BTSettl/phoenix direcotry. path should
    point to this directory.

    Writes catalog.fits file in the cdbs directory
    """
    output_dir = ['mergedm25', 'mergedm20', 'mergedm15', 'mergedm10', 'mergedm05', 'mergedp00', 'mergedp02', 'mergedp05']
    metal_arr = [-2.5, -2.0, -1.5, -1.0, -0.5, 0, 0.2, 0.5]

    index_str = []
    name_str = []
    for ii in range(len(output_dir)):
        files = glob.glob('{0}/{1}/*.fits'.format(path, output_dir[ii]))
    
        # Extract parameters for each atmosphere from the filename,
        # construct columns for catalog file
        for name in files:
            final = name.split('/')[-1]
            tmp = final.split('_')
            temp = float(tmp[1]) # In kelvin
            logg = float(tmp[2][:-5])

            index_str.append('{0},{1},{2:3.2f}'.format(int(temp), metal_arr[ii], logg))
            name_str.append('{0}/{1}[Flux]'.format(output_dir[ii], final))

    # Make catalog
    catalog = Table([index_str, name_str], names = ('INDEX', 'FILENAME'))

    # Create catalog.fits file in directory with the models
    catalog.write(path+'catalog.fits', format = 'fits', overwrite=True)
    
    return

def test_atlas_phoenix_atmospheres():
    """
    Test atlas and phoenix atmospheres; are they on the same flux scale?
    """
    # Get the atmospheres. By default, phoenix will be on atlas resolution
    ck04 = pysynphot.Icat('ck04models', 3500, 0, 3.5)
    phoenix_rebin = pysynphot.Icat('phoenix_v16_rebin', 3500, 0, 3.5)
    phoenix = pysynphot.Icat('phoenix_v16', 3500, 0, 3.5)

    # Trim to NIR region
    good_ck = np.where( (ck04.wave > 10000) & (ck04.wave < 25000) )
    good_ph_r = np.where( (phoenix_rebin.wave > 10000) & (phoenix_rebin.wave < 25000) )
    good_ph = np.where( (phoenix.wave > 10000) & (phoenix.wave < 25000) )

    wave_ck = ck04.wave[good_ck]
    flux_ck = ck04.flux[good_ck]
    wave_ph_r = phoenix_rebin.wave[good_ph_r]
    flux_ph_r = phoenix_rebin.flux[good_ph_r]
    wave_ph = phoenix.wave[good_ph]
    flux_ph = phoenix.flux[good_ph]    

    # Plot
    py.figure(1)
    py.clf()
    py.plot(wave_ph, flux_ph, 'g-', label = 'phoenix')
    py.plot(wave_ck, flux_ck, 'r-', label = 'ck04')
    py.plot(wave_ph_r, flux_ph_r, 'b-', label = 'phoenix_rebin')
    py.xlabel('Wavelength (Angstrom)')
    py.ylabel('Flux (FLAM)')
    py.savefig('test.png')

    pdb.set_trace()

    return

def read_index(catalog):
    """
    Short code to parse the index of a catalog.fits file in
    cdbs directory.
    """
    catalog = Table.read(catalog)
    
    index = catalog['INDEX']

    temp_arr = []
    metal_arr = []
    logg_arr = []
    for i in index:
        tmp = i.split(',')
        temp_arr.append(float(tmp[0]))
        metal_arr.append(float(tmp[1]))
        logg_arr.append(float(tmp[2]))

    temp_arr = np.array(temp_arr)
    metal_arr = np.array(metal_arr)
    logg_arr = np.array(logg_arr)
    pdb.set_trace()

    return

def test_atlas_cmfgen_atmospheres(path_to_cdbs):
    """
    Compare atlas and cmfgen atmosphere at T > 20,000 K. Will use
    cdbs directory to pull atmospheres; cmfgen_rot_rebin vs. nk04models.

    path_to_cdbs is path to cdbs directory, i.e. /g/lu/models/cdbs
    """
    # Atmosphere temps to test
    temps_to_test = np.arange(20000, 50000, 5000)
    
    # Extract paramters for CMFGEN grid from the catalog.fits file
    cmfgen_path = path_to_cdbs+'/grid/cmfgen_rot_rebin/'
    cat = fits.getdata(cmfgen_path + 'catalog.fits')
    
    files_all = [cat[ii][1].split('[')[0] for ii in range(len(cat))]
    temp_arr = np.zeros(len(files_all))
    metal_arr = np.zeros(len(files_all))
    grav_arr = np.zeros(len(files_all))
    
    for ff in range(len(files_all)):
        # Extract the temp, Z, logg
        vals = cat[ff][0].split(',')
        temp_arr[ff] = float(vals[0])
        metal_arr[ff] = float(vals[1])
        grav_arr[ff] = float(vals[2])
        
    # Find the nearest temps in the CMFGEN grid to the test temps
    idx = []
    for i in temps_to_test:
        good = np.where( abs(temp_arr - i) == min(abs(temp_arr - i)) )[0]
        idx.append(good[0])
    # Only want the unique indicies, no repeats
    idx = np.unique(np.array(idx))

    # Extract the cmfgen and atlas models for the desired temps
    sp_cmfgen = []
    sp_atlas = []
    for i in idx:
        temp_tmp = temp_arr[i]
        metal_tmp = metal_arr[i]
        grav_tmp = grav_arr[i]

        cmfgen = pysynphot.Icat('cmfgen_rot_rebin', temp_tmp, metal_tmp, grav_tmp)
        atlas = atmospheres.get_castelli_atmosphere(metallicity = metal_tmp,
                                                    temperature = temp_tmp,
                                                    gravity = grav_tmp)


        sp_cmfgen.append(cmfgen)
        sp_atlas.append(atlas)

    # Plot comparison between 1.1 - 2.4 microns
    good = np.where( (sp_cmfgen[0].wave > 11000) & (sp_cmfgen[0].wave < 24000) )

    for i in range(len(sp_cmfgen)):
        py.figure(1, figsize=(10,10))
        py.clf()
        # Want normalized fluxes
        wave_atlas = sp_atlas[i].wave[good]
        flux_atlas = wave_atlas * sp_atlas[i].flux[good] / \
          (wave_atlas[0] * sp_atlas[i].flux[good][0])
        wave_cmfgen = sp_cmfgen[i].wave[good]
        flux_cmfgen = wave_cmfgen * sp_cmfgen[i].flux[good] / \
          (wave_cmfgen[0] * sp_cmfgen[i].flux[good][0])
        py.plot(wave_atlas, flux_atlas, 'k-', label='Atlas')
        py.plot(wave_cmfgen, flux_cmfgen, 'r-', label='CMFGEN')
        py.xlabel('Wavelength (A)')
        py.ylabel(r'$\lambda$f$_{\lambda}$ / $\lambda$f$_{\lambda}$ (1.1 $\mu$m)')
        py.legend()
        py.title('T = {0:5.0f}, Z = {1:2.1f}, grav = {2:3.2f}'.format(temp_arr[idx[i]],
                                                                      metal_arr[idx[i]],
                                                                      grav_arr[idx[i]]))
        py.axis([11000, 24000,0,1.1])
        py.savefig('atlas_cmfgen_{0:5.0f}.png'.format(temp_arr[idx[i]]))

    pdb.set_trace()
    return
