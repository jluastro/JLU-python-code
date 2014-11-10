from pyraf.iraf import gemini
from pyraf.iraf import gsaoi 
from . import struc
import os, shutil
from astropy.io import fits 
import numpy as np

def doit(epoch_dates ,clean_path,  log_file=None, filters=None, dates=None):
    '''
    optional arguments for filters and dates
    Must either give log file to base data on, or use both filters and dates arguements
    filters/dates must be in same form as those that created the directory structure

    Clean_path is the path to where you want the cleaned images to be saved
    epoch dates is a list of the dates the epoch dolders are based on
    
    '''


    if log_File !=None:
        if filters ==None and dates = None:
            frame,s obj ,filt1, ra, dec, date, exptime, coadds, mjd = util.read_log(log_file)
            filters  = np.unique(filt1)
            dates = np.unique(date)
        elif filters != None and dates = None:
            frame,s obj ,filt1, ra, dec, date, exptime, coadds, mjd = util.read_log(log_file)
            dates = np.unique(date)
        elif filters == None and dates != None:
            frame,s obj ,filt1, ra, dec, date, exptime, coadds, mjd = util.read_log(log_file)
            filters = np.unique(filt1)
    
        
        
    os.chdir('reduce')
    cwd = os.getcwd()
    for i in epoch_dates:
        util.mkdir(clean_path+'/'+i)
        for j in dates:
            util.mkdir(clean_path+'/'+i+'/'j
            for k in filters:
                util.mkdir(clean_path+'/'+i+'/'+j+'/'+k)
                red_dir(cwd+'/'+i+'/'+j+'/'k, clean_path+'/'+i+'/'+j+'/'+k)
            
            
    
    
    
def red_dir(directory,clean_dir, sky_key='sky', flat_key='Domeflat', sci_keys= ['Wd2pos1','Wd2pos2', 'Wd2pos3', 'Wd2pos4']):

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
        sci=False
        for j in sci_keys:
            if head['OBJ'] == j:
                sci=True
        if head['OBJ'] == sky_key:
            sky.append(i)
        elif head['OBJ'] == flat_key:
            flat.append(i)
        else:
            sci_b=False
            for j in sci_keys:
                if head['OBJ'] == j:
                    sci.append(i)

            
    gsaoi.gaflat(flat_s, outimage='flat.fits')
    gsaoi.gaprepare(sky_s)
    gsaoi.gasky(sky_s, outimage='sky.fits', flatimg='flat.fits')
    gsaoi.gareduce(sci_s, fl_sky='yes', fl_flat='yes', flatimg='flat.fits', skyimg='sky.fits')

    for i in sci:
        shutil.copy(i, clean_dir)
    
    
        
    
    
    
