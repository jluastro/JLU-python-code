import os, errno, shutil
import pyfits
import pdb
import glob
import numpy as np
import math
import pyfits
from astropy.io import fits

def rmall(files):
    """Remove list of files without confirmation."""
    for file in files:
        if os.access(file, os.F_OK): os.remove(file)


def mkdir(dir):
    """Make directory if it doesn't already exist."""
    try: 
        os.makedirs(dir)
    except OSError, exc:
        if exc.errno == errno.EEXIST:
            pass
        else:
            raise

def getcwd():
    """
    IRAF doesn't like long file names. This reduces them.
    """
    
    curdir = os.getcwd()
    
    newdir1 = curdir.replace('/net/uni/Groups/ghez/ghezgroup', '/u/ghezgroup')
    newdir2 = newdir1.replace('/net/uni/Groups/ghez/jlu', '/u/jlu/work')
    newdir3 = newdir2.replace('/net/dione/data0/ghez', '/u/ghezgroup')
    newdir4 = newdir3.replace('/scr2/jlu/data', '/u/jlu/data')
    newdir5 = newdir4.replace('/net/calla.ifa.hawaii.edu/Volumes/indicium/g/lu/data/', '/g/lu/data/')
    newdir5 +=  '/'

    return newdir5

def trimdir(olddir):
    """
    IRAF doesn't like long file names. This reduces them.
    """
    newdir1 = olddir.replace('/net/uni/Groups/ghez/ghezgroup', '/u/ghezgroup')
    newdir2 = newdir1.replace('/net/uni/Groups/ghez/jlu', '/u/jlu/work')
    newdir3 = newdir2.replace('/net/dione/data0/ghez', '/u/ghezgroup')
    newdir4 = newdir3.replace('/scr2/jlu/data', '/u/jlu/data')
    newdir5 = newdir4.replace('/net/calla.ifa.hawaii.edu/Volumes/indicium/g/lu/data/', '/g/lu/data')
    return newdir5

def cp_change_prefix(arg1,arg2):
    """
    Takes files beginning with arg1 and replaces them with arg2
    Must be in the directory where files live
    """

    # Find files in this directory beginning with arg1
    files = os.listdir(".")
    # Ignore files beginning with '.'
    files=[filename for filename in files if filename[0] != '.']

    ln = len(arg1)

    for ff in range(len(files)):
        pre = files[ff][0:ln]
        if pre == arg1:
            suf = files[ff][len(arg1):]
            newFile = arg2 + suf
            shutil.copy(files[ff], newFile)


def cp_change_suffix(arg1,arg2):
    """
    Takes files ending with arg1 and replaces them with arg2
    Must be in the directory where files live
    """

    # Find files in this directory ending with arg1
    files = os.listdir(".")
    # Ignore files beginning with '.'
    files=[filename for filename in files if filename[0] != '.']

    ln = len(arg1)

    for ff in range(len(files)):
        suf = files[ff][len(files[ff])-len(arg1):]
        if suf == arg1:
            pre = files[ff][0:len(files[ff])-len(arg1)]
            newFile = pre + arg2 
            shutil.copy(files[ff], newFile)



def read_log(filename):
    f = open(filename)
    frames =[] 
    obj = []
    filt1 = []
    ra = []
    dec = []
    date = []
    exptime = []
    coadds = []
    mjd = []
    for lines in f:
        dum = lines.split()
        frames.append(dum[0])
        obj.append(dum[1])
        filt1.append(dum[2])
        ra.append(dum[3])
        dec.append(dum[4])
        date.append(dum[5])
        exptime.append(dum[6])
        coadds.append(int(dum[7]))
        mjd.append(float(dum[8]))
        
    return np.array(frames), np.array(obj), np.array(filt1), np.array(ra), np.array(dec), np.array(date), np.array(exptime), np.array(coadds), np.array(mjd)

def mk_lis(directory='.', obj='Domeflat'):
    '''
    returns list of fit files that have obj keywoed = obj
    '''

    files = glob.glob(directory+'/*.fits')
    file_lis = []
    
    for ff in files:
        hdr = fits.getheader(ff)

        dir, filename = os.path.split(ff)
        fileroot, fileext = os.path.splitext(filename)
       
        if hdr['OBJECT'] == obj:
            file_lis.append(fileroot)

    return file_lis
        

        
def mk_log(directory, output='gsaoi_log.txt'):
    """
    Read in all the fits files in the specified directory and print
    out a log containing the useful header information.

    directory - the directory to search for *.fits files
    output - the output file to print the information to
    """
    files = glob.glob(directory + '/*.fits')

    _out = open(directory + '/' +  output, 'w')

    for ff in files:
        hdr = fits.getheader(ff)

        line = ''

        dir, filename = os.path.split(ff)
        fileroot, fileext = os.path.splitext(filename)
        line += '{0:16} '.format(fileroot)
        
        line += '{0:15s} '.format(hdr['OBJECT'].replace(" ",""))
        line += '{0:15s} '.format(hdr['FILTER1'])

        ra = hdr['RA']
        raHour = math.floor(ra) * 24.0 / 360.0
        raMin = (ra % 1) * 60
        raSec = (raMin % 1) * 60
        line += '{0:2d}:{1:0>2d}:{2:0>5.2f} '.format(int(raHour), int(raMin), raSec)

        dec = hdr['DEC']
        decDeg = math.floor(dec)
        decMin = (dec % 1) * 60
        decSec = (dec % 1) * 60
        line += '{0:3d}:{1:0>2d}:{2:0>5.2f}  '.format(int(decDeg), int(decMin), decSec)

        line += '{0} '.format(hdr['DATE-OBS'])
        line += '{0:6.2f} '.format(hdr['EXPTIME'])
        line += '{0:3d} '.format(hdr['COADDS'])
        line += '{0:8.2f}'.format(hdr['MJD-OBS'])

        line += '\n'

        _out.write(line)

    _out.close()
    return directory + '/' +  output


def convertFile(fitsfile, outputDir=None, clobber=False):
    """Takes a full path to a fits file and converts it to simple fits format if it can.
    Names the converted extensions to be like
       
    j1570d10t_crj.fits ===> NEXTEND number of files named like 

          j1570d10t_crj_EXTNAME_EXTVER.fits

    Copied from
    http://apsis.googlecode.com/svn-history/r2/trunk/apsis/python/utils/fUtil.py

    """
    warningsList = []

    dir,fname = os.path.split(fitsfile)
    base,ext  = os.path.splitext(fname)
    
    fo = pyfits.open(fitsfile)

    if len(fo) == 1:
        warningsList.append("WARNING: File "+fname+" is not multi-extension. Not converted.")
        print "File "+fname+" is not multi-extension. Not converted."
        return None

    # first check that BSCALE and BZERO are *not* in the primary header.
    # remove them if they are and record a WARNING

    try:
        del fo[0].header.ascard['BSCALE']
        warningsList.append("WARNING: BSCALE keyword found in primary header: "+fname)
        print "WARNING: BSCALE keyword found in primary header: "+fname
    except:
        pass
    try:
        del fo[0].header.ascard['BZERO']
        warningsList.append("WARNING: BZERO keyword found in primary header: "+fname)
        print "WARNING: BZERO keyword found in primary header: "+fname
    except:
        pass    

    convertList = []
    for i in range(1,len(fo)):
        print "breaking out extension ",i," of file ",fname
        xname = fo[i].header['CCDNAME'].lower()
        try:
            xver = str(fo[i].header['EXTNAME']).strip().lower()
        except:
            xver = ''
        if xver:
            newfile = base+"_"+xname+"_"+xver + ext
        else:
            newfile = base+"_"+xname + ext

        if outputDir == None:
            outputDir = dir

        nfits = os.path.join(outputDir, newfile)
        convertList.append(nfits)

        f1 = pyfits.HDUList()
        f1.append(pyfits.PrimaryHDU())

        # Adjust header keywords for simple fits format.
        # Delete some keywords which must go away in simple fits format
        # First, get a copy of the Primary header into the new file

        f1[0].header = fo[0].header.copy()

        # These are primary header keywords that need to disappear.
        try:
            del f1[0].header['EXTEND']
            del f1[0].header['NEXTEND']
        except:
            passw

        # delete some of the extension-only keywords
        try:
            del fo[i].header['XTENSION']
            del fo[i].header['INHERIT']
            del fo[i].header['CCDNAME']
            del fo[i].header['EXTVER']
            del fo[i].header['PCOUNT']
            del fo[i].header['GCOUNT']
        except:
            pass

        # get the index of first HISTORY comment in the header we're updating
        # remember to increment this in the loop as we insert keys
        ihist = f1[0].header.get_history()
        if len(ihist) == 0:
            print 'Primary header has no HISTORY keyword.'
            print 'Appending extension keywords...'
            ihist = len(f1[0].header)

        # Now go through all keywords of extension and pass them
        # into the new simple fits header.
        # This is how I'd like to do it, but it doesn't work this way
        # for key,item in fo[i].header.ascard.keys(),fo[i].header.ascard:
        # keylist = fo[i].header.ascard.keys()

        # Below, an ascard object has attributes key,value,comment which allows 
        # direct access to these values.

        for item in fo[i].header.cards:
            key = item.key
            val = item.value
            
            if not key and not val:
                continue
            elif not key and val:
                f1[0].header.insert(ihist, item)
                ihist += 1
                continue

            #print "setting keyword,",key

            # if the 'key' is already in the header because it was copied
            # from the primary, we just want to update it with the value
            # of the keyword in the extension; no need to update ihist
            if (f1[0].header.has_key(key) and key != 'COMMENT' and key != 'HISTORY'):
                try:
                    f1[0].header.update(key,fo[i].header[key])
                except pyfits.FITS_SevereError,err:
                    warningsList.append("WARNING: FITS Error encountered in header,"+str(err))
                    print "FITS Error encountered in header:",err
                continue

            # if it's NAXIS1, insert after NAXIS

            if key == 'NAXIS1':
                try:
                    inaxis = f1[0].header.ascard.index_of('NAXIS')
                except KeyError,err:
                    raise pyfits.FITS_FatalError,"Cannot find NAXIS keyword!"

                # NOTE: dereferencing 'key' to get item at this point 
                # was no good.  For repeated 'keys', as for example with
                # 'COMMENT' lines, this just gave back the first one.
                # item = fo[i].header.ascard[key]

                f1[0].header.ascard.insert(inaxis+1,item)
                ihist += 1

            # if it's NAXIS2, insert after NAXIS1 if it's already there,
            # otherwise, insert after NAXIS

            elif key == 'NAXIS2':
                try:
                    inaxis2 = f1[0].header.ascard.index_of('NAXIS1') + 1
                except:
                    try:
                        inaxis2 = f1[0].header.ascard.index_of('NAXIS') + 1
                    except KeyError,err:
                        raise pyfits.FITS_FatalError,"Cannot find NAXIS keyword!"

                    f1[0].header.ascard.insert(inaxis2,item)
                    ihist += 1

            # if key is not NAXIS1 or NAXIS2, just insert before HISTORY
            else:
                f1[0].header.ascard.insert(ihist,item)
                ihist += 1

        # Fix the filename keyword
        f1[0].header.update('FILENAME',newfile)

        # And finally, get the data.
        f1[0].data = fo[i].data
        f1.writeto(nfits, clobber=clobber)

        del f1

    fo.close()
    
    return convertList, warningsList
def mk_bool(frames, obj, filt1, ra, dec, date, exptime, coadds, mjd, day_diff=14, sci_keys= ['Wd2pos1','Wd2pos2', 'Wd2pos3', 'Wd2pos4'], dome_key='Domeflat', sky_key='sky'):
    
    '''
    does boolean split up for given frame_file
    used by both re.py  and struc.py
    '''

    sky_bool = obj == sky_key
    dome_bool = obj == dome_key
    sci_bool = np.zeros(len(frames), dtype=bool)
    for i in range(len(sci_keys)):
        sci_bool = sci_bool + (obj == sci_keys[i])
        #print sci_bool
    
    #find minimum of date
    epoch_dates = []
    obs_bool = np.ones(len(mjd), dtype=bool)
    epoch_bool_ars = []
    num_done = 0
    
    
    limits = [np.min(mjd)]
    obs_bool = np.ones(len(mjd), dtype=bool)
    index = 1
    while np.any(obs_bool):
        
        epoch_dates.append(date[np.argmin(mjd[obs_bool])+num_done])
        limits.append(np.min(mjd[obs_bool])+day_diff)
        obs_bool = obs_bool - (mjd < limits[index]+.1)*(mjd > limits[index-1]-.1)
        epoch_bool_ars.append((mjd < limits[index]+.1)*(mjd > limits[index-1]-.1))
        num_done += np.sum(epoch_bool_ars[-1])
        index += 1  
        

    
    return sky_bool, sci_bool, dome_bool, epoch_dates, epoch_bool_ars
    
