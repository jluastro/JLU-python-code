from pyraf import iraf
#from pyraf.iraf import gsaoi
import struc
import os, shutil
from astropy.io import fits 
import numpy as np
import util


def doit(epoch_dates ,obj_path,  clean_path=None, log_file=None, filters=None, dates=None):
    '''
    optional arguments for filters and dates
    Must either give log file to base data on, or use both filters and dates arguements
    filters/dates must be in same form as those that created the directory structure

    Clean_path is the path to where you want the cleaned images to be saved
    epoch dates is a list of the dates the epoch dolders are based on

    obj_path is the path to the directory \object name, assumes strucute as shown below

      \object name
           \epoch date
              \clean  \reduce
                                 \filters
    
    '''

    
    if log_file !=None:
        if filters ==None and dates == None:
            frame_list, obj ,filt1, ra, dec, date, exptime, coadds, mjd = util.read_log(log_file)
            filters  = np.unique(filt1)
            dates = np.unique(date)
        elif filters != None and dates == None:
            frame_list, obj ,filt1, ra, dec, date, exptime, coadds, mjd = util.read_log(log_file)
            dates = np.unique(date)
        elif filters == None and dates != None:
            frame_list, obj ,filt1, ra, dec, date, exptime, coadds, mjd = util.read_log(log_file)
            filters = np.unique(filt1)
    else:
        frame_list=None

    
        
        
    cwd = os.getcwd()
    for i in epoch_dates:
        util.mkdir(obj_path+'/clean/'+i)
        for j in dates:
            util.mkdir(obj_path+'/clean/'+i+'/'+j)
            for k in filters:
                util.mkdir(obj_path+'/clean/'+i+'/'+j+'/'+k)
                if frame_list != None:
                    red_dir(cwd+'/'+i+'/reduce/'+k,obj_path+'/clean/'+i+'/'+j+'/'+k, frame_list=frame_list[filt1==k])
                else:
                    red_dir(cwd+'/'+i+'/reduce/'+k,obj_path+'/clean/'+i+'/'+j+'/'+k, frame_list=frame_list)
            
            
    
    
    
def red_dir(directory,clean_dir, sky_key='sky', flat_key='Domeflat', sci_keys= ['Wd 2 pos 1','Wd 2 pos 2', 'Wd 2 pos 3', 'Wd 2 pos 4'], frame_list = None):

    '''
    Note, must be ran from pyraf interavtive terminal
    perform reduction on directory given
    directory must be full path name
    sky_key is header keywaord for sky frames
    dome_key is header keyword for domes
    sci_coadds is the minimum number of coadds required for an image to be considered a science image
    '''

    os.chdir(directory)
    if frame_list == None:
        frame_list = glob.glob(util.getcwd() + '/*.fits')
  

    #go through the fits files and make 3 lists, one of skies, one of domes one of science frames

    
    sci_f = open('obj.lis', 'w')
    dome_f = open('flat.lis', 'w')
    dome_list = []
    sky_f = open('sky.lis','w')
    
    for i in frame_list:
        #import pdb; pdb.set_trace()
        
        head = fits.getheader(i+'.fits')
        if head['OBJECT'] == sky_key:
            print >> sky_f, i
        elif head['OBJECT']==flat_key:
            print >> dome_f, i
            dome_list.append(i)
        else:
            for j in sci_keys:
                if head['OBJECT'] == j:
                    print >> sci_f, i

    sky_f.close()
    sci_f.close()
    dome_f.close()

    
    iraf.gemini()
    iraf.gsaoi()
    gemini.unlearn()
    gsaoi.unlearn()

    #raw_dir = util.getcwd()
    raw_dir = './'
    prep_dir = raw_dir+'g'
    #print raw_dir
    gaprepare('*.fits', rawpath=raw_dir, outpref=prep_dir, fl_vardq='yes', logfile='gaprep.log')
    
    

    gsaoi.gaflat('g//@flat.lis', rawpath=raw_dir, gaprep_pref=prep_dir, outsufx='flat')
    flat_name= "g"+dome_list[0]+"_flat.fits"
    
    gsaoi.gareduce('g//@sky.lis', fl_flat='yes', flatimg=raw_dir+flat_name)
    gsaoi.gasky('g//@sky.lis', outimages='sky.fits', fl_vardq='yes', fl_dqprop='yes', flatimg=raw_dir+flat_name)
    
    gsaoi.gareduce('g//@sci.lis',fl_vardq='yes', fl_dqprop='yes', fl_dark='no', fl_sky='yes',skyimg=raw_dir+'sky.fits',  fl_flat='yes',flatimg=raw_dir+flat_name)

    

    #for i in sci:
    #    shutil.copy('g'+i+'.fits', clean_dir)
    
    
        
    
    
    
