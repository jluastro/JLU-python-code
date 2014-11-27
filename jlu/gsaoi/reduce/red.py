import struc
import os, shutil
from astropy.io import fits 
import numpy as np
import util
import glob


def full_reduc(frame_file,raw_dir,red_dir=None, mix_flats=True, sci_keys= ['Wd2pos1','Wd2pos2', 'Wd2pos3', 'Wd2pos4'], dome_key='Domeflat', sky_key='sky', mk_struc=True ):
    '''
    frame_file -- string, filename of log file (can be made with util.mk_log)
    raw_dir - string, pathname to directory containing raw fits files
    red_dir - keyword, path to reduction directory, if None assumes reduction directory is cwd.  Expected to be directory \object_name
    mix_flats - boolean, determines whether flats are expected to be per epoch or only one set for all observations
    '''
    if red_dir:
        os.chdir(red_dir)
    if mk_struc:
        struc.mk_struc(frame_file=frame_file, directory=raw_dir, sci_keys=sci_keys, dome_key=dome_key, sky_key=sky_key, mix_flats=mix_flats)

    doit(frame_file, mix_flats=mix_flats, sky_key=sky_key, flat_key=dome_key, sci_keys=sci_keys)
               
def doit(frame_file, epoch_start_in=0, mix_flats=True, sky_key='sky', flat_key='Domeflat', sci_keys= ['Wd 2 pos 1','Wd 2 pos 2', 'Wd 2 pos 3', 'Wd 2 pos 4']):
    '''
    optional arguments for filters and dates
    Must either give log file to base data on, or use both filters and dates arguements
    filters/dates must be in same form as those that created the directory structure

    Clean_path is the path to where you want the cleaned images to be saved
    epoch dates is a list of the dates the epoch dolders are based on

    obj_path is the path to the directory \object name, assumes structure as shown below

      \object name
           \epoch date
              \clean  \reduce
                                 \filters

    mix_flats -- Boolean, if true dome flats from all nights in the relevant filter will be used
    
    '''

    frames, obj, filt1, ra, dec, date, exptime, coadds, mjd = util.read_log(frame_file)
    sky_bool, sci_bool, dome_bool, epoch_dates, epoch_bool_ars = util.mk_bool(frames, obj, filt1, ra, dec, date, exptime, coadds, mjd)
    
    
        
    cwd = util.getcwd()
    for index, i in enumerate(epoch_dates[epoch_start_in:]):
        util.mkdir(cwd+'clean/'+i)
        for j in np.unique(date[epoch_bool_ars[index+epoch_start_in]]):
            filters = np.unique(filt1[epoch_bool_ars[index+epoch_start_in]])
            for k in filters:
                #os.chdir(cwd+'/'+i+'/reduce/'+k)
                util.mkdir(cwd+'clean/'+i+'/'+k)
                print 'Working in  '+cwd+i+'/reduce/'+j+'/'+k+'/'
                if np.any((filt1==k)*(date==j)*(sci_bool)):
                    if not np.any(sky_bool[(filt1==k)*(date==j)]):
                            #find skyies that are closest temporaly to the observations
                            arg = np.argmin(np.abs(mjd[(filt1==k)*(sky_bool)]-mjd[date==j][0]))
                            night_sky = date[(filt1==k)*(sky_bool)][arg]
                            ex_skies = frames[sky_bool*(filt1==k)*(date==night_sky)]
                            print 'extra sky frames added to  '+ i+k, ex_skies
                    else:
                        ex_skies=[]
                            
                    #only give in list of frames that 
                    #print np.logical_or((np.logical_or(sci_bool,sky_bool) * (date==j) ),dome_bool) * (filt1==k)
                    if mix_flats:
                        red_dir(cwd+i+'/reduce/'+j+'/'+k+'/',cwd+'/clean/'+i+'/'+k+'/', frame_list=np.concatenate((frames[np.logical_or((np.logical_or(sci_bool,sky_bool) * (date==j) ),dome_bool) * (filt1==k)],ex_skies)),sky_key=sky_key, flat_key=flat_key, sci_keys= sci_keys )
                    else:
                        red_dir(cwd+i+'/reduce/'+j+'/'+k+'/',cwd+i+'/clean/'+k+'/', frame_list=np.concatenate((frames[np.logical_or(np.logical_or(sci_bool, sky_bool),dome_bool)*(date==j)*(filt1==k)],ex_skies)) )
                np.logical_or(np.logical_or(sci_bool, sky_bool),dome_bool)*(date==j)*(filt1==k)
            
            
    
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
    from pyraf import iraf 
    
    
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
            
            iraf.imcopy('rg'+i+'['+str(k+1)+'][inherit+]' , 'rg'+i.replace('.fits',str(k+1)+'.fits'))
            shutil.move('rg'+i.replace('.fits',+str(k+1)+'.fits'), clean_dir+'rg'+i.replace('.fits','_'+str(k+1)+'.fits'))
            os.remove('rg'+i)
    
    
        
    
    
    
