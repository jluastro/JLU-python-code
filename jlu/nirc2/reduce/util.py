import asciidata
import os, errno, shutil
import pyfits
import pdb

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
    newdir4 +=  '/'

    return newdir4

def trimdir(olddir):
    """
    IRAF doesn't like long file names. This reduces them.
    """
    newdir1 = olddir.replace('/net/uni/Groups/ghez/ghezgroup', '/u/ghezgroup')
    newdir2 = newdir1.replace('/net/uni/Groups/ghez/jlu', '/u/jlu/work')
    newdir3 = newdir2.replace('/net/dione/data0/ghez', '/u/ghezgroup')
    newdir4 = newdir3.replace('/scr2/jlu/data', '/u/jlu/data')
    return newdir4

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



def update_header_coords(fileList):
    """
    Updates coordinates in the header for XREF, YREF
    and XSTREHL, and YSTREHL.
 
    fileList : list of files to update
    """

    _files = asciidata.open(fileList)
    files = _files[0].tonumpy()
    files = [files[ff].split('.')[0] for ff in range(len(files))]
    

    for ff in range(len(files)):
        # Open .coo file and read 16C's coordinates
        coo = asciidata.open(files[ff]+'.coo')
        xref = coo[0].tonumpy()
        yref = coo[1].tonumpy()

        # Open .coord file and read strehl source's coordinates
        coord = asciidata.open(files[ff]+'.coord')
        xstr = coord[0].tonumpy()
        ystr = coord[1].tonumpy()
 
        # Open image and write reference star x,y to fits header
        fits = pyfits.open(files[ff]+'.fits')

        fits[0].header.update('XREF', "%.3f" %xref,
                              'Cross Corr Reference Src x')
        fits[0].header.update('YREF', "%.3f" %yref,
                              'Cross Corr Reference Src y')
        fits[0].header.update('XSTREHL', "%.3f" %xstr,
                              'Strehl Reference Src x')
        fits[0].header.update('YSTREHL', "%.3f" %ystr,
                              'Strehl Reference Src y')

        # Output fits file
        _out = 'new_hdr/' + files[ff] + '.fits'
        fits[0].writeto(_out, output_verify='silentfix')

