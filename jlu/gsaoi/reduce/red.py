import struc
import os, shutil
from astropy.io import fits 
import numpy as np
import util
import glob


def doit(frame_file , clean_path=None):
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

    
    sky_bool, sci_bool, dome_bool, obs_breaks, epoch_bool_ars = util.mk_bool(frames, obj, filt1, ra, dec, date, exptime, coadds, mjd)
    
    
        
    cwd = util.getcwd()
    for index, i in enumerate(epoch_dates):
        util.mkdir(cwd+'clean/'+i)
        for j in np.unique(date[epoch_bool_ars[i]]):
            util.mkdir(cwd+'clean/'+i+'/'+j)
            for k in filters:
                util.mkdir(cwd+'clean/'+i+'/'+j+'/'+k)
                #os.chdir(cwd+'/'+i+'/reduce/'+k)
                print 'Working in  '+cwd+i+'/reduce/'+j+'/'+k+'/'
                if np.any((filt1==k)*(date==j)*(sci_bool)):
                    if frame_list != None:
                        #only give in list of frames that 
                        print np.logical_and(np.logical_and(np.logial_or(np.logical_and(np.logical_or(sci_bool,sky_bool),date==j),dome_bool),filt1==k),epoch_bool)
                        red_dir(cwd+i+'/reduce/'+j+'/'+k+'/',cwd+'/clean/'+i+'/'+j+'/'+k+'/', frame_list=frame_list[np.logical_and(np.logical_and(np.logial_or(np.logical_and(np.logical_or(sci_bool,sky_bool),date==j),dome_bool),filt1==k),epoch_bool)] )
                    else:
                        red_dir(cwd+i+'/reduce/'+j+'/'+k+'/',cwd+'/clean/'+i+'/'+j+'/'+k+'/', frame_list=frame_list)
                
            
            
            
    
    
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
    print 'Reduction being performed in ', directory
    
    if frame_list == None:
        frame_list = glob.glob(directory+'*.fits')
        dir_ap = ''
        for i in range(len(frame_list)):
            frame_list[i] = frame_list[i].replace('.fits','')
    else:
        dir_ap=directory
            
  
    
    #go through the fits files and make 3 lists, one of skies, one of domes one of science frames

    
    sci_f = open('obj.lis', 'w')
    dome_f = open('flat.lis', 'w')
    all_f = open('all.lis', 'w')
    sky_f = open('sky.lis','w')
    dome_list = []
    sci_l = []
    
    
    for i in frame_list:
        #import pdb; pdb.set_trace()
        print >> all_f, dir_ap+i+'.fits'
        head = fits.getheader(dir_ap+i+'.fits')
        if head['OBJECT'] == sky_key:
            print >> sky_f, dir_ap+'g'+i+'.fits'
        elif head['OBJECT']==flat_key:
            print >> dome_f, dir_ap+'g'+i+'.fits'
            dome_list.append(i)
        else:
            for j in sci_keys:
                if head['OBJECT'] == j:
                    print >> sci_f, dir_ap+'g'+i+'.fits'
                    sci_l.append(i+'.fits')

    sky_f.close()
    sci_f.close()
    dome_f.close()
    all_f.close()

    

    from pyraf.iraf import gemini
    from pyraf.iraf import gsaoi
    from pyraf.iraf import iraf 
    
    
    gemini.unlearn()
    gsaoi.unlearn()

    #raw_dir = util.getcwd()
    #prep_dir = raw_dir+'g'
    #print raw_dir

    util.rmall(['gaprep.log'])
    print 'Arguements for gaprepare', '@all.lis', directory+'g' 
    gsaoi.gaprepare('@all.lis',outpref=directory+'g',fl_vardq='yes', logfile='gaprep.log')
    
    

    gsaoi.gaflat('@flat.lis', outsufx='flat', fl_vardq='yes')
    flat_name=  'g'+dome_list[0]+"_flat.fits"
    shutil.move('g'+dome_list[0]+"_flat.fits", directory+'g'+dome_list[0]+"_flat.fits")
    
    
    #print flat_name
    
    #gsaoi.gareduce('@sky.lis', rawpath=directory, gaprep_pref = directory+'g',calpath=directory, fl_flat='yes', flatimg=flat_name)
    gsaoi.gasky('@sky.lis', outimages='sky.fits', fl_vardq='yes', fl_dqprop='yes', flatimg=directory+flat_name)
    shutil.move('sky.fits', directory+'sky.fits')
    
    gsaoi.gareduce('@obj.lis',fl_vardq='yes', fl_dqprop='yes', fl_dark='no',calpath=directory, fl_sky='yes',skyimg=directory+'sky.fits',  fl_flat='yes',flatimg=flat_name)

    util.rmall(['obj.lis','sky.lis','flat.lis'])

    for i in sci_l:
        for k in range(4):
            iraf.imcopy('rg'+i, 'rg'+i.replace('.fits',str(k)+'.fits'))
            shutil.move('rg'+i.replace('.fits',str(k)+'.fits'), clean_dir+'rg'+i.replace('.fits',str(k)+'.fits'))
    #print >> script, 'from pyraf.iraf import gemini'
    #print >> script, 'from pyraf.iraf import gsaoi'
    #print >> script, 'gsaoi.gareduce('+'"'+'*.fits'+'"'+', fl_vardq='+'"'+'yes'+'"'+')'

    #script.close()
    #import script
    
    
    
        
    
    
    
