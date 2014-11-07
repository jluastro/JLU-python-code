from . import util
import os, shutil
from astropy.io import fits 
import glob 

def mk_struc(frame_file=None, directory=None, ret=False, day_diff=14):
    '''
    Should be in directory called 'object name'
    creates structure as follows
    day_diff is number of days it groups images by
    ret decides if anything is returned
    frame_file is a file list that contains the information about the fits images needed to make the directory structure
    directory is the directory to look for fits files, only relevant if frame_file=None

    \object name
           \epoch date
            \clean \raw \reduce
                             \night date
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

    #find minimum of date
    obs_breaks = []
    obs_bool = np.ones(len(mjd), type=bool)
    while np.any(obs_bool):
        obs_break.append(np.argmin(mjd[obs_bool]))
        obs_bool[mjd < obs_break[-1]+day_diff]

    #get list of filters
    filters_uni = np.unique(filt1)
    util.mkdir('reduce')
    for i in obs_breaks:
        util.mkdir('reduce/'+date[i])
        util.mkdir('reduce/'+date[i])
        for j in filters_uni:
            util.mkdir('reduce/'+date[i]+'/'+j)
            for fn in frames[(mjd < mjd[i]) * (filt1 == j)]:
                shutil.copy(directory+'/'+fn, date[i]+'/'+j)

    

    if ret:
        date_list  = []
        for i in obs_break:
            date_list.append(date[i])
            
        return filters_uni, date_list 
    
    
    
    

    
    
    
