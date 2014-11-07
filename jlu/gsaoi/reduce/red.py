from pyraf.iraf import gemini
from pyraf.iraf import gsaoi 
from . import struc
import os, shutil
from astropy.io import fits 

def doit(raw_path, clean_path, filters=None, dates=None):
    '''
    optional arguments for filters and dates
    
    '''


    if filters == None and dates == None:
        filters, dates = struc.mk_struc(directory=raw_path, ret=True)
    elif filters == None and dates != None:
        filters , dum = struc.mk_struc(directory=raw_path, ret=True)
    elif filters != None and dates == None:
        dum, dates = struc.mk_struc(directory=raw_path, ret=True)
    else:
       struc.mk_struc(directory=raw_path)
        
        
    os.chdir('reduce')
    cwd = os.getcwd()
    for i in dates:
        for j in filters:
            red_dir(cwd+'/'+i+'/'+j, clean_path)
            
            
    
    
    
def red_dir(directory,clean_dir, sky_key='sky', flat_key='Domeflat', sci_coadds=4):

    '''
    Note, must be ran from pyraf interavtive terminal
    perform reduction on directory given
    directory must be full path name
    sky_key is header keywaord for sky frames
    dome_key is header keyword for domes
    sci_coadds is the minimum number of coadds required for an image to be considered a science image
    '''

    os.chdir(directory)
    files = glob.glob(os.getcwd() + '/*.fits')

    #go through the fits files and make 3 lists, one of skies, one of domes one of science frames

    sky_s = ''
    flat_s = ''
    sci_s = ''
    sky = []
    flat = []
    sci = []
    for i in files:
        head = fits.getheader(i)
        if head['OBJ'] == sky_key:
            sky.append(i)
        elif head['OBJ'] == flat_key:
            flat.append(i)
        elif head['COADDS'] >= sci_coadds:
            sci.append(i)

            
    gsaoi.gaflat(flat_s, outimage='flat.fits')
    gsaoi.gaprepare(sky_s)
    gsaoi.gasky(sky_s, outimage='sky.fits', flatimg='flat.fits')
    gsaoi.gareduce(sci_s, fl_sky='yes', fl_flat='yes', flatimg='flat.fits', skyimg='sky.fits')

    for i in sci:
        shutil.copy(i, clean_dir)
    
    
        
    
    
    
