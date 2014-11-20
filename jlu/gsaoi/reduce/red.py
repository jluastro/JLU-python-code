import struc
import os, shutil
from astropy.io import fits 
import numpy as np
import util
import glob


def doit(epoch_dates , clean_path=None, log_file=None, filters=None, dates=None):
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

    
        
        
    cwd = util.getcwd()
    for i in epoch_dates:
        util.mkdir(cwd+'/clean/'+i)
        for j in dates:
            util.mkdir(cwd+'/clean/'+i+'/'+j)
            for k in filters:
                util.mkdir(cwd+'/clean/'+i+'/'+j+'/'+k)
                #os.chdir(cwd+'/'+i+'/reduce/'+k)
                print 'Working in ' + cwd+'/'+i+'/reduce/'+k
                if frame_list != None:
                    red_dir(cwd+i+'/reduce/'+k,cwd+'/clean/'+i+'/'+j+'/'+k, frame_list=frame_list[filt1==k])
                else:
                    red_dir(cwd+i+'/reduce/'+k,cwd+'/clean/'+i+'/'+j+'/'+k, frame_list=frame_list)
            
            
    
    
    
def red_dir(directory,clean_dir, sky_key='sky', flat_key='Domeflat', sci_keys= ['Wd 2 pos 1','Wd 2 pos 2', 'Wd 2 pos 3', 'Wd 2 pos 4'], frame_list = None):

    '''
    Note, must be ran from pyraf interavtive terminal
    perform reduction on directory given
    directory must be full path name
    sky_key is header keywaord for sky frames
    dome_key is header keyword for domes
    sci_coadds is the minimum number of coadds required for an image to be considered a science image
    '''

    
    #os.chdir(directory)
    print 'current working directory is ' + util.getcwd()
    if frame_list == None:
        frame_list = glob.glob(util.getcwd()+'*.fits')
        for i in range(len(frame_list)):
            frame_list[i] = frame_list[i].replace('.fits','')
  
    
    #go through the fits files and make 3 lists, one of skies, one of domes one of science frames

    
    sci_f = open(directory+'obj.lis', 'w')
    dome_f = open(directory+'flat.lis', 'w')
    dome_list = []
    sky_f = open(directory+'sky.lis','w')
    
    
    for i in frame_list:
        #import pdb; pdb.set_trace()
        
        head = fits.getheader(directory+'/'+i+'.fits')
        if head['OBJECT'] == sky_key:
            print >> util.getcwd()+sky_f, i
        elif head['OBJECT']==flat_key:
            print >> util.getcwd()+dome_f, i
            dome_list.append(i)
        else:
            for j in sci_keys:
                if head['OBJECT'] == j:
                    print >> util.getcwd()+sci_f, i

    sky_f.close()
    sci_f.close()
    dome_f.close()

    

    from pyraf.iraf import gemini
    from pyraf.iraf import gsaoi
    
    
    #gemini.unlearn()
    #xsgsaoi.unlearn()

    #raw_dir = util.getcwd()
    raw_dir = './'
    #prep_dir = raw_dir+'g'
    #print raw_dir

    
    gsaoi.gaprepare('*.fits', fl_vardq='yes', rawpath=raw_dir, logfile='gaprep.log')
    
    

    gsaoi.gaflat(directory+'@flat.lis', outsufx='flat')
    flat_name= directory + 'g'+dome_list[0]+"_flat.fits"
    print flat_name
    
    gsaoi.gareduce(directory+'@sky.lis', rawpath=directory, gaprep_pref = directory+'g', fl_flat='yes', flatimg=flat_name)
    gsaoi.gasky(directory+'@sky.lis', outimages='sky.fits', fl_vardq='yes', fl_dqprop='yes', flatimg=flat_name)
    
    gsaoi.gareduce('g//@sci.lis',fl_vardq='yes', fl_dqprop='yes', fl_dark='no', fl_sky='yes',skyimg='sky.fits',  fl_flat='yes',flatimg=flat_name)

    #util.rmall(['obj.lis','sky.lis','flat.lis']]

    #for i in sci:
    #    shutil.copy('g'+i+'.fits', clean_dir)
    #print >> script, 'from pyraf.iraf import gemini'
    #print >> script, 'from pyraf.iraf import gsaoi'
    #print >> script, 'gsaoi.gareduce('+'"'+'*.fits'+'"'+', fl_vardq='+'"'+'yes'+'"'+')'

    #script.close()
    #import script
    
    
        
    
    
    
