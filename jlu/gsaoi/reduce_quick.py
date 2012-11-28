import numpy as np
from pyraf import iraf as ir
import glob
import pyfits
import math
import os
import ds9

def log(directory, output='gsaoi_log.txt'):
    """
    Read in all the fits files in the specified directory and print
    out a log containing the useful header information.

    directory - the directory to search for *.fits files
    output - the output file to print the information to
    """
    files = glob.glob(directory + '/*.fits')

    _out = open(directory + '/' +  output, 'w')

    for ff in files:
        hdr = pyfits.getheader(ff)

        line = ''

        dir, filename = os.path.split(ff)
        fileroot, fileext = os.path.splitext(filename)
        line += '{0:16} '.format(fileroot)
        
        line += '{0:15s} '.format(hdr['OBJECT'])

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
        line += '{0:5.1f} '.format(hdr['PA'])
        line += '{0:6.2f} '.format(hdr['EXPTIME'])
        line += '{0:3d} '.format(hdr['COADDS'])
        line += '{0:3d} '.format(hdr['LNRS'])

        line += '\n'

        _out.write(line)

    _out.close()


def convertExtension(fitsfile, extension=1, outputDir=None, clobber=False):
    """
    Takes a full path to a fits file, breaks out the passed extension into simple fits
    file and returns the full path name of this file to the caller.  This file will be written
    in the same dir as the passed fitsfile.  Returns None if the passed fitsfile does not have
    multiple extensions.

    Copied from
    http://apsis.googlecode.com/svn-history/r2/trunk/apsis/python/utils/fUtil.py
    """
    # check that the file name delivered is a full path

    warningsList = []

    oldfits = pyfits.open(fitsfile)
    if len(oldfits) == 1:
        warningsList.append("WARNING: File "+fitsfile+" is not multi-extension. No extension to convert.")
        print "File "+fname+" is not multi-extension. No extension to convert."
        return None
    dir,fname = os.path.split(fitsfile)
    base,ext  = os.path.splitext(fname)
    xname = oldfits[extension].header['CCDNAME'].upper()
    try:
        xver = str(oldfits[extension].header['EXTVER'])
    except:
        xver = ''
    if xver:
        newfile = base+"_"+xname+"_"+xver + ext
    else:
        newfile = base+"_"+xname + ext

    if outputDir == None:
        outputDir = dir

    tfile   = os.path.join(outputDir, newfile)
    newfits = pyfits.HDUList()
    newfits.append(pyfits.PrimaryHDU())
    newfits[0].header = oldfits[0].header.copy()
    try:
        del newfits[0].header.ascard['BSCALE']
        warningsList.append("WARNING: BSCALE keyword found in primary header: "+fname)
        print "WARNING: BSCALE keyword found in primary header: "+fname
    except: pass
    try:
        del newfits[0].header.ascard['BZERO']
        warningsList.append("WARNING: BZERO keyword found in primary header: "+fname)
        print "WARNING: BZERO keyword found in primary header: "+fname
    except: pass

    # These are primary header keywords that need to disappear.

    try:
        del newfits[0].header['EXTEND']
        del newfits[0].header['NEXTEND']
    except:
        pass

    for card in newfits[0].header.ascardlist():
        if card.key == "HISTORY":
            newfits[0].header.ascard.remove(card)
            continue
    # delete some of the extension-only keywords

    try:
        del oldfits[extension].header['XTENSION']
        del oldfits[extension].header['INHERIT']
        del oldfits[extension].header['CCDNAME']
        del oldfits[extension].header['EXTVER']
        del oldfits[extension].header['PCOUNT']
        del oldfits[extension].header['GCOUNT']
    except:
        pass

    print "breaking out extension ",extension," of file ",fname

    for card in oldfits[extension].header.ascard:
        key = card.key
        val = card.value
        if not key:
            continue
        if key == "HISTORY" or key == "COMMENT":
            continue

        # override a primary keyword value with the extension value

        if newfits[0].header.has_key(key):
            newfits[0].header.update(key,oldfits[extension].header[key])
            continue
        if key == 'NAXIS1':
            try:   inaxis = newfits[0].header.ascard.index_of('NAXIS')
            except KeyError,err:
                raise pyfits.FITS_FatalError,"Cannot find NAXIS keyword!"
            newfits[0].header.ascard.insert(inaxis+1,card)
            continue
        elif key == 'NAXIS2':
            try:       inaxis2 = newfits[0].header.ascard.index_of('NAXIS1') + 1
            except:
                try:   inaxis2 = newfits[0].header.ascard.index_of('NAXIS') + 1
                except KeyError,err:
                    raise pyfits.FITS_FatalError,"Cannot find NAXIS keyword!"
            newfits[0].header.ascard.insert(inaxis2,card)
            continue
        else:
            newfits[0].header.ascard.append(card)

    newfits[0].data = oldfits[extension].data
    newfits[0].header.update('FILENAME',newfile)
    newfits.writeto(tfile, clobber=clobber)
    oldfits.close()
    del oldfits,newfits
    return tfile,warningsList


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
        xname = fo[i].header['CCDNAME'].upper()
        try:
            xver = str(fo[i].header['EXTVER'])
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
            pass

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


def display_mosaic(fitsFile):
    fo = pyfits.open(fitsfile)

    q1 = fo[1].data
    q2 = fo[2].data
    q3 = fo[3].data
    q4 = fo[4].data

    d = ds9.ds9()

    d.set('frame delete all')
    d.set('frame frameno 1')

    
