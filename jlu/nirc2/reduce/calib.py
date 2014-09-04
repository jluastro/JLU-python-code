import os, sys
from . import util
import pyfits
from pyraf import iraf as ir
import numpy as np

module_dir = os.path.dirname(__file__)

def makedark(files, output):
    """
    Make dark image for NIRC2 data. Makes a calib/ directory
    and stores all output there. All output and temporary files
    will be created in a darks/ subdirectory. 

    files: integer list of the files. Does not require padded zeros.
    output: output file name. Include the .fits extension.
    """
    redDir = os.getcwd() + '/'  # Reduce directory.
    curDir = redDir + 'calib/'
    darkDir = util.trimdir(curDir + 'darks/')
    rawDir = util.trimdir(os.path.abspath(redDir + '../raw') + '/')

    util.mkdir(curDir)
    util.mkdir(darkDir)
    
    _out = darkDir + output
    _outlis = darkDir + 'dark.lis'
    util.rmall([_out, _outlis])

    darks = [rawDir + 'n' + str(i).zfill(4) + '.fits' for i in files]

    f_on = open(_outlis, 'w')
    f_on.write('\n'.join(darks) + '\n')
    f_on.close()
    
    ir.unlearn('imcombine')
    ir.imcombine.combine = 'median'
    ir.imcombine.reject = 'sigclip'
    ir.imcombine.nlow = 1
    ir.imcombine.nhigh = 1
    ir.imcombine('@' + _outlis, _out)
    

def makeflat(onFiles, offFiles, output, normalizeFirst=False):
    """
    Make flat field image for NIRC2 data. Makes a calib/ directory
    and stores all output there. All output and temporary files
    will be created in a flats/ subdirectory. 
    
    onFiles: integer list of lamps ON files. Does not require padded zeros.
    offFiles: integer list of lamps OFF files. Does not require padded zeros.

    If only twilight flats were taken (as in 05jullgs), use these flats as
    the onFiles, and use 0,0 for offFiles. So the reduce.py file should look
    something like this: onFiles = range(22, 26+1) and offFiles = range(0,0)
    The flat will then be made by doing a median combine using just the
    twilight flats.
    output: output file name. Include the .fits extension.
    """
    redDir = os.getcwd() + '/'
    curDir = redDir + 'calib/'
    flatDir = util.trimdir(curDir + 'flats/')
    rawDir = util.trimdir(os.path.abspath(redDir + '../raw') + '/')

    util.mkdir(curDir)
    util.mkdir(flatDir)

    _on = flatDir + 'lampsOn.fits'
    _off = flatDir + 'lampsOff.fits'
    _norm = flatDir + 'flatNotNorm.fits'
    _out = flatDir + output
    _onlis = flatDir + 'on.lis'
    _offlis = flatDir + 'off.lis'
    _onNormLis = flatDir + 'onNorm.lis'

    util.rmall([_on, _off, _norm, _out, _onlis, _offlis, _onNormLis])

    lampson = [rawDir + 'n' + str(i).zfill(4) + '.fits' for i in onFiles]
    lampsoff = [rawDir + 'n' + str(i).zfill(4) + '.fits' for i in offFiles]
    lampsonNorm = [flatDir + 'norm' + str(i).zfill(4) + '.fits' for i in onFiles]
    util.rmall(lampsonNorm)

    if (len(offFiles) != 0):
        f_on = open(_onlis, 'w')
        f_on.write('\n'.join(lampson) + '\n')
        f_on.close()
        f_on = open(_offlis, 'w')
        f_on.write('\n'.join(lampsoff) + '\n')
        f_on.close()
        f_onn = open(_onNormLis, 'w')
        f_onn.write('\n'.join(lampsonNorm) + '\n')
        f_onn.close()
    
        # Combine to make a lamps on and lamps off
        ir.unlearn('imcombine')
        ir.imcombine.combine = 'median'
        ir.imcombine.reject = 'sigclip'
        ir.imcombine.nlow = 1
        ir.imcombine.nhigh = 1
        ir.imcombine('@' + _offlis, _off)

        # Check if we should normalize individual flats first
        # such as in the case of twilight flats.
        if normalizeFirst:
            f_on = open(_offlis, 'w')
            f_on.write('\n'.join(lampsoff) + '\n')
            f_on.close()
            
            # Subtract "off" from individual frames
            ir.imarith('@'+_onlis, '-', _off, '@'+_onNormLis)
            
            # Scale them and combine
            ir.imcombine.scale = 'median'
            ir.imcombine('@' + _onNormLis, _norm)
        else:
            # Combine all "on" frames
            ir.imcombine('@' + _onlis, _on)

            # Now do lampsOn - lampsOff
            ir.imarith(_on, '-', _off, _norm)


        # Normalize the final flat
        ir.module.load('noao', doprint=0, hush=1)
        ir.module.load('imred', doprint=0, hush=1)
        ir.module.load('generic', doprint=0, hush=1)
        orig_img = pyfits.getdata(_norm)
        orig_size = (orig_img.shape)[0]
        if (orig_size >= 1024):
            flatRegion = '[100:900,513:950]'
        else:
            flatRegion = ''
        ir.normflat(_norm, _out, sample=flatRegion)

    else:
        f_on = open(_onlis, 'w')
        f_on.write('\n'.join(lampson) + '\n')
        f_on.close()

        # Combine twilight flats
        ir.unlearn('imcombine')
        ir.imcombine.combine = 'median'
        ir.imcombine.reject = 'sigclip'
        ir.imcombine.nlow = 1
        ir.imcombine.nhigh = 1
        if normalizeFirst:
            # Scale them
            ir.imcombine.scale = 'median'
        ir.imcombine('@' + _onlis, _norm)

        # Normalize the flat
        ir.module.load('noao', doprint=0, hush=1)
        ir.module.load('imred', doprint=0, hush=1)
        ir.module.load('generic', doprint=0, hush=1)
        flatRegion = '[100:900,513:950]'
        ir.normflat(_norm, _out, sample=flatRegion)

def makemask(dark, flat, output):
    """Make bad pixel mask for NIRC2 data. Makes a calib/ directory
    and stores all output there. All output and temporary files
    will be created in a masks/ subdirectory. 
    
    @param dark: The full relative path to a dark file. This is used to
        construct a hot pixel mask. Use a long (t>20sec) exposure dark.
    @type dark: str
    @param flat: The full relative path to a flat file. This is used to 
        construct a dead pixel mask. The flat should be normalized.
    @type flat: str
    @param output: output file name. This will be created in the masks/
        subdirectory.
    @type output: str
    """
    redDir = os.getcwd() + '/'
    calDir = redDir + 'calib/'
    maskDir = util.trimdir(calDir + 'masks/')
    flatDir = util.trimdir(calDir + 'flats/')
    darkDir = util.trimdir(calDir + 'darks/')
    rawDir = util.trimdir(os.path.abspath(redDir + '../raw') + '/')
    dataDir = util.trimdir(os.path.abspath(redDir + '../..') + '/')

    util.mkdir(calDir)
    util.mkdir(maskDir)

    _out = maskDir + output
    _dark = darkDir + dark
    _flat = flatDir + flat
    _nirc2mask = module_dir + '/masks/nirc2mask.fits'

    util.rmall([_out])

    # Make hot pixel mask
    whatDir = redDir + dark
    print(whatDir)

    text_output = ir.imstatistics(_dark, fields="mean,stddev", 
				  nclip=10, format=0, Stdout=1)
    print text_output
    values = text_output[0].split()
    hi = float(values[0]) + (10.0 * float(values[1]))

    img_dk = pyfits.getdata(_dark)
    hot = img_dk > hi

    # Make dead pixel mask
    text_output = ir.imstatistics(_flat, fields="mean,stddev", 
				  nclip=10, format=0, Stdout=1)
    values = text_output[0].split()
    #lo = float(values[0]) - (15.0 * float(values[1]))
    # If flat is normalized, then lo should be set to 0.5
    lo = 0.5
    hi = float(values[0]) + (15.0 * float(values[1]))

    img_fl = pyfits.getdata(_flat)
    dead = np.logical_or(img_fl > hi, img_fl < lo)
    
    # We also need the original NIRC2 mask (with cracks and such)
    nirc2mask = pyfits.getdata(_nirc2mask)

    # Combine into a final supermask. Use the flat file just as a template
    # to get the header from.
    ofile = pyfits.open(_flat)
    
    if ((hot.shape)[0] == (nirc2mask.shape)[0]):
        mask = hot + dead + nirc2mask
    else:
        mask = hot + dead
    mask = (mask != 0)
    unmask = (mask == 0)
    ofile[0].data[unmask] = 0
    ofile[0].data[mask] = 1
    ofile[0].writeto(_out, output_verify='silentfix')
    

def makeNirc2mask(dark, flat, outDir):
    """Make the static bad pixel mask for NIRC2. This only needs to be
    run once. This creates a file called nirc2mask.fits which is
    subsequently used throughout the pipeline. The dark should be a long
    integration dark.
    
    @param dark: The full absolute path to a medianed dark file. This is 
        used to construct a hot pixel mask (4 sigma detection thresh).
    @type dark: str
    @param flat: The full absolute path to a medianed flat file. This is
         used to construct a dead pixel mask.
    @type flat: str
    @param outDir: full path to output directory with '/' at the end.
    @type outDir: str
    """
    _out = outDir + 'nirc2mask.fits'
    _dark = dark
    _flat = flat

    util.rmall([_out])

    # Make hot pixel mask
    text_output = ir.imstatistics(_dark, fields="mean,stddev", 
				  nclip=10, format=0, Stdout=1)
    values = text_output[0].split()
    hi = float(values[0]) + (15.0 * float(values[1]))

    img_dk = pyfits.getdata(_dark)
    hot = img_dk > hi
    print 'Found %d hot pixels' % (hot.sum())

    # Make dead pixel mask
    text_output = ir.imstatistics(_flat, fields="mean,stddev", 
				  nclip=10, format=0, Stdout=1)
    values = text_output[0].split()

    # Assuming flat is normalized, we don't want pixels with less
    # than 0.5 sensitivity
    #lo = float(values[0]) - (15.0 * float(values[1]))
    lo = 0.5    #mask = hot

    hi = float(values[0]) + (15.0 * float(values[1]))

    img_fl = pyfits.getdata(_flat)
    dead = logical_or(img_fl > hi, img_fl < lo)
    print 'Found %d dead pixels' % (dead.sum())

    # Combine into a final supermask
    file = pyfits.open(_flat)

    mask = hot + dead
    mask = (mask != 0)
    unmask = (mask == 0)
    file[0].data[unmask] = 0
    file[0].data[mask] = 1
    file[0].writeto(_out, output_verify='silentfix')
    
def analyzeDarkCalib(firstFrame, skipcombo=False):
    """
    Reduce data from the dark_calib script that should be run once
    a summer in order to test the dark current and readnoise.

    This should be run in the reduce/calib/ directory for a particular
    run.
    """
    def printStats(frame, tint, sampmode, reads):
	files = range(frame, frame+3)
	
	fileName = 'dark_%ds_1ca_%d_%dsm.fits' % (tint, sampmode, reads)

	if (skipcombo == False):
	    makedark(files, fileName)

	text_output = ir.imstatistics('darks/'+fileName, 
				      fields="mean,stddev", 
				      nclip=10, format=0, Stdout=1)
	values = text_output[0].split()
	darkMean = float(values[0])
	darkStdv = float(values[1])

	return darkMean, darkStdv
	

    frame = firstFrame

    lenDarks = 11

    tints = np.zeros(lenDarks) + 12
    tints[-3] = 10
    tints[-2] = 50
    tints[-1] = 100

    reads = np.zeros(lenDarks)
    reads[0] = 1
    reads[1] = 2
    reads[2] = 4
    reads[3] = 8
    reads[4] = 16
    reads[5] = 32
    reads[6] = 64
    reads[7] = 92
    reads[-3:] = 16

    samps = np.zeros(lenDarks) + 3
    samps[0] = 2

    dMeans = np.zeros(lenDarks, dtype=float)
    dStdvs = np.zeros(lenDarks, dtype=float)

    for ii in range(lenDarks):
	(dMeans[ii], dStdvs[ii]) = printStats(frame, tints[ii], 
					      samps[ii], reads[ii])
	dStdvs[ii] *= np.sqrt(3)

	frame += 3

    # Calculate the readnoise
    rdnoise = dStdvs * 4.0 * np.sqrt(reads) / (np.sqrt(2.0))
    print 'READNOISE per read: ', rdnoise


    ##########
    # Print Stuff Out
    ##########
    outFile = 'darks/analyzeDarkCalib.out'
    util.rmall([outFile])
    _out = open(outFile,'w')
    hdr = '%8s  %5s  &9s  %9s  %4s  %6s'
    print 'Sampmode  Reads  Noise(DN)  Noise(e-)  Tint  Coadds'
    print '--------  -----  ---------  ---------  ----  ------'

    _out.write('Sampmode  Reads  Noise(DN)  Noise(e-)  Tint  Coadds\n')
    _out.write('--------  -----  ---------  ---------  ----  ------\n')

    for ii in range(lenDarks):
	print '%8d  %5d  %9.1f  %9.1f  %4d  1' % \
	    (samps[ii], reads[ii], dStdvs[ii], dStdvs[ii] * 4.0, tints[ii])

    for ii in range(lenDarks):
	_out.write('%8d  %5d  %9.1f  %9.1f  %4d  1\n' % \
	    (samps[ii], reads[ii], dStdvs[ii], dStdvs[ii] * 4.0, tints[ii]))

    _out.close()

