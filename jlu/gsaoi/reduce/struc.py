import util
import os, shutil
from astropy.io import fits 
import glob 
import numpy as np
import math

def mk_struc(frame_file=None, directory=None, ret=False, day_diff=14, sci_keys= ['Wd2pos1','Wd2pos2', 'Wd2pos3', 'Wd2pos4'], dome_key='Domeflat', sky_key='sky'):
    '''
    Should be in directory called 'object name'
    creates structure as follows
    day_diff is number of days it groups images by
    ret decides if anything is returned
    frame_file is a file list that contains the information about the fits images needed to make the directory structure
    directory is the directory to look for fits files, is location that files will be copied from
    WARNING Assumes that the frames are in rank order of ascending date

    \object name
           \epoch date
              \clean  \reduce
                            \night_date 
                                 \filters
                                
    

    Expects that you are in the direcrtory 'object name' 
    '''

    if not frame_file:
        if not directory:
            '''
            Assumes that fits files are in current directory
            '''
            frame_file = util.mk_log(util.getcwd())
            directory = util.getcwd()
        else:
            frame_file = util.mk_log(directory)

    '''
    Find out how many dates there are
    and create date directories for each 2 week interval 
    Then find out how many filters are present 
    '''
    frames, obj, filt1, ra, dec, date, exptime, coadds, mjd = util.read_log(frame_file)

    sky_bool = obj == sky_key
    dome_bool = obj == dome_key
    sci_bool = np.zeros(len(frames), dtype=bool)
    for i in range(len(sci_keys)):
        sci_bool = sci_bool + (obj == sci_keys[i])
        #print sci_bool
    
    #find minimum of date
    obs_breaks = []
    obs_bool = np.ones(len(mjd), dtype=bool)
    epoch_bool_ars = []
    num_done = 1
    
    while np.any(obs_bool):
        \
        obs_bool = obs_bool * np.logical_not(mjd[obs_breaks[-1]]+14 > mjd)
        obs_breaks.append(np.argmin(mjd[obs_bool])+num_done-1)
        for i in range(len(mjd)):
            num_done += obs_bool[i]
        epoch_bool_ars.append((mjd[obs_breaks[-1]]+14 > mjd) * (mjd[obs_breaks[-1]] < mjd))
        #import pdb; pdb.set_trace()
    print obs_breaks
    print epoch_bool_ars
    print mjd
    import pdb ; pdb.set_trace()
        
        
         

    #get list of filters
    for i, ep_ind in enumerate(obs_breaks):
        print 'Making Directory '+date[ep_ind]
        util.mkdir(date[ep_ind])
        util.mkdir(date[ep_ind]+'/reduce')
        util.mkdir(date[ep_ind]+'/clean')
        #util.mkdir(date[ep_ind]+'/raw')
        #find nights in epoch, make directories for them
        uni_date = np.unique(date[epoch_bool_ars[i]])
        print uni_date
        for night in uni_date:
            uni_filt=np.unique(filt1[epoch_bool_ars[i]*(night==date)])
            print uni_filt
            for k in uni_filt:
                if np.any(sci_bool):
                    util.mkdir(date[ep_ind]+'/reduce/'+night+'/'+k)
                    for ii,frame in enumerate(frames[np.logical_and(np.logical_and(np.logical_or(sky_bool,sci_bool),filt1==k),date==night)]):
                        shutil.copy(directory+'/'+frame+'.fits', date[ep_ind]+'/reduce/'+night+'/'+k)
                    for ii, frame in enumerate(frames[np.logical_and(np.logical_and(dome_bool, filt1==k),epoch_bool_ars[i])]):
                        shutil.copy(directory+'/'+frame+'.fits', date[ep_ind]+'/reduce/'+night+'/'+k)
                        
                
                
    

    

    if ret:
        date_list  = []
        for i in obs_breaks:
            date_list.append(date[i])
            
        return filters_uni, date_list 
    
    
    
    

    
    
    
