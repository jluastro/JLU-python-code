import os, sys
import pyfits, math
import user, asciidata
from pyraf import iraf as ir
import nirc2
import util
import time
import pdb
import numpy as np
import dar
import bfixpix

module_dir = os.path.dirname(__file__)

# Remember to change these if you are going to use the wide camera.
# You can change them in your reduce.py file after importing data.py
# Narrow Camera
distCoef = ''
distXgeoim = module_dir + '/distortion/nirc2_narrow_xgeoim.fits'
distYgeoim = module_dir + '/distortion/nirc2_narrow_ygeoim.fits'
# Wide Camera
#distCoef = module_dir + '/distortion/coeffs/nirc2_cubic_wide'
#distXgeoim = ''
#distYgeoim = ''

supermaskName = 'supermask.fits'
outputVerify = 'ignore'


def clean(files, nite, wave, refSrc, strSrc, badColumns=None, field=None,
          skyscale=0, skyfile=None, angOff=0.0, fixDAR=True):
    """
    Clean near infrared NIRC2 images.

    This program should be run from the reduce/ directory.
    Example directory structure is:
	calib/
	    flats/
		flat_kp.fits
		flat.fits (optional)
	    masks/
		supermask.fits
	kp/
	    sci_nite1/
	    sky_nite1/
		sky.fits

    All output files will be put into reduce/../clean/ in the 
    following structure:
	kp/
	    c*.fits
	    distort/
		cd*.fits
	    weight/
		wgt*.fits

    The clean directory may be optionally modified to be named
    <field_><wave> instead of just <wave>. So for instance, for Arches
    field #1 data reduction, you might call clean with: field='arch_f1'.

    @param files: a list of file numbers. Doesn't require zero pad.
    @type files: integer list
    @param nite: the nite suffix (e.g. nite1). This is only used inside
        the reduce sub-directories. For prefixes based on different fields,
        use the optional 'field' keyword.
    @type nite: string
    @param wave: the wavelength suffix (e.g.. kp).
    @type wave: string
    @kwparam field: Optional prefix for clean directory and final
        combining. All clean files will be put into <field_><wave>. You
        should also pass the same into combine(). If set to None (default)
        then only wavelength is used.
    @type field: string
    @kwparam skyscale: (def = 0) Turn on for scaling skies before subtraction.
    @type skyscale: 1/0
    @kwparam skyfile: (def = '') An optional file containing image/sky matches.
    @type skyfile: string
    @kwparam angOff: (def = 0) An optional absolute offset in the rotator
        mirror angle for cases (wave='lp') when sky subtraction is done with
        skies taken at matchin rotator mirror angles.
    @type angOff: float
    @kwparam badColumns: (def = None) An array specifying the bad columns (zero-based).
    			Assumes a repeating pattern every 8 columns.
    @type badColumns: int array
    """

    # Start out in something like '06maylgs1/reduce/kp/'
    waveDir = util.getcwd()
    redDir = os.path.abspath(waveDir + '../') + '/'
    rootDir = os.path.abspath(redDir + '../') + '/'

    sciDir = waveDir + '/sci_' + nite + '/'
    util.mkdir(sciDir)
    ir.cd(sciDir)

    # Set location of raw data
    rawDir = rootDir + 'raw/'

    # Setup the clean directory
    cleanRoot = rootDir + 'clean/'
    if (field != None):
        clean = cleanRoot + field + '_' + wave + '/'
    else:
        clean = cleanRoot + wave + '/'
    distort = clean + 'distort/'
    weight = clean + 'weight/'
    masks = clean + 'masks/'

    util.mkdir(cleanRoot)
    util.mkdir(clean)
    util.mkdir(distort)
    util.mkdir(weight)
    util.mkdir(masks)

    try:
        # Setup flat. Try wavelength specific, but if it doesn't
        # exist, then use a global one.
        flatDir = redDir + 'calib/flats/'
        flat = flatDir + 'flat_' + wave + '.fits'
        if not os.access(flat, os.F_OK): 
            flat = flatDir + 'flat.fits'

        # Bad pixel mask
        _supermask = redDir + 'calib/masks/' + supermaskName

        # Determine the reference coordinates for the first image.
        # This is the image for which refSrc is relevant.
        firstFile = rawDir + '/n' + str(files[0]).zfill(4) + '.fits'
        hdr1 = pyfits.getheader(firstFile,ignore_missing_end=True)
        radecRef = [float(hdr1['RA']), float(hdr1['DEC'])]
        aotsxyRef = nirc2.getAotsxy(hdr1)

        # Setup a Sky object that will figure out the sky subtraction
        skyDir = waveDir + 'sky_' + nite + '/'
        skyObj = Sky(sciDir, skyDir, wave, scale=skyscale,
        skyfile=skyfile, angleOffset=angOff)

        # Prep drizzle stuff
        # Get image size from header - this is just in case the image
        # isn't 1024x1024 (e.g., NIRC2 sub-arrays). Also, if it's
        # rectangular, choose the larger dimension and make it square
        imgsizeX = float(hdr1['NAXIS1'])
        imgsizeY = float(hdr1['NAXIS2'])
        if (imgsizeX >= imgsizeY):
            imgsize = imgsizeX
        else:
            imgsize = imgsizeY
        setup_drizzle(imgsize)

        ##########
        # Loop through the list of images
        ##########
        for f in files:
            root = str(f).zfill(4)

            # Define filenames
            _raw = rawDir + 'n' + root
            _n = 'n' + root + '.fits'
            _ss = 'ss' + root + '.fits'
            _ff = 'ff' + root + '.fits'
            _ff_f = 'ff' + root + '_f' + '.fits'
            _ff_s = 'ff' + root + '_s' + '.fits'
            _bp = 'bp' + root + '.fits'
            _cd = 'cd' + root + '.fits'
            _ce = 'ce' + root + '.fits'
            _cc = 'c' + root + '.fits'
            _wgt = 'wgt' + root + '.fits'
            _statmask = 'statmask' + root + '.fits'
            _crmask = 'crmask' + root + '.fits'
            _mask = 'mask' + root + '.fits'
            _pers = 'pers' + root + '.fits'
            _max = 'c' + root + '.max'
            _coo = 'c' + root + '.coo'
            _dlog = 'driz' + root + '.log'

            # Clean up if these files previously existed
            util.rmall([_n, _ss, _ff, _ff_f, _ff_s, _bp, _cd, _ce, _cc,
                        _wgt, _statmask, _crmask, _mask, _pers, _max, _coo, _dlog])

            ### Copy the raw file to local directory ###
            ir.imcopy(_raw, _n, verbose='no')

            ### Make perisistance mask ###
            # - Checked images, this doesn't appear to be a large effect.
            #clean_persistance(_n, _pers)

            ### Sky subtract ###
            # Get the proper sky for this science frame.
            # It might be scaled or there might be a specific one for L'.
            sky = skyObj.getSky(_n)

            ir.imarith(_n, '-', sky, _ss)

            ### Flat field ###
            ir.imarith(_ss, '/', flat, _ff)

            ### Make a static bad pixel mask ###
            # _statmask = supermask + bad columns
            clean_get_supermask(_statmask, _supermask, badColumns)

            ### Fix bad pixels ###
            # Produces _ff_f file
            bfixpix.bfixpix(_ff, _statmask)
            util.rmall([_ff_s])

            ### Fix cosmic rays and make cosmic ray mask. ###
            clean_cosmicrays(_ff_f, _crmask, wave)

            ### Combine static and cosmic ray mask ###
            # This will be used in combine later on.
            # Results are stored in _mask, _mask_static is deleted.
            clean_makemask(_mask, _crmask, _statmask, wave)

            ### Background Subtraction ###
            bkg = clean_bkgsubtract(_ff_f, _bp)

            ### Drizzle individual file ###
            clean_drizzle(_bp, _ce, _wgt, _dlog, fixDAR=fixDAR)

            ### Make .max file ###
            # Determine the non-linearity level. Raw data level of
            # non-linearity is 12,000 but we subtracted
            # off a sky which changed this level. The sky is 
            # scaled, so the level will be slightly different
            # for every frame.
            nonlinSky = skyObj.getNonlinearCorrection(sky)

            coadds_tmp = ir.hselect(_ss, "COADDS", "yes", Stdout=1)
            coadds = float(coadds_tmp[0])
            satLevel = (coadds*12000.0) - nonlinSky - bkg
            file(_max, 'w').write(str(satLevel))

            ### Rename and clean up files ###
            ir.imrename(_bp, _cd)
            util.rmall([_n, _ss, _ff, _ff_f])

            ### Make the *.coo file and update headers ###
            # First check if PA is not zero
            tmp = rootDir + 'raw/n' + root + '.fits'
            hdr = pyfits.getheader(tmp,ignore_missing_end=True)
            phi = nirc2.getPA(hdr)

            clean_makecoo(_ce, _cc, root, refSrc, strSrc, aotsxyRef, radecRef,
                          clean)

            ### Move to the clean directory ###
            util.rmall([clean + _cc, clean + _coo, 
                        distort + _cd, weight + _wgt,
                        clean + _ce, clean + _max,
                        masks + _mask, _ce])

            os.rename(_cc, clean + _cc)
            os.rename(_cd, distort + _cd)
            os.rename(_wgt, weight + _wgt)
            os.rename(_mask, masks + _mask)
            os.rename(_max, clean + _max)
            os.rename(_coo, clean + _coo)

            # This just closes out any sky logging files.
            skyObj.close()
    finally: 
        # Move back up to the original directory
        ir.cd('../')

def clean_get_supermask(_statmask, _supermask, badColumns):
    """
    Create temporary mask for each individual image that will contain the
    supermask plus the designated bad columns.

    _statmask -- output file containing supermask + bad columns
    """

    maskFits = pyfits.open(_supermask)

    # Check that we have some valid bad columns.
    if badColumns != None and len(badColumns) != 0:
        for cc in badColumns:
            if (cc < 0):
                continue

            # Make column index from 0-512 n steps of 8
            colIndex = arange(cc, 512, 8)
            maskFits[0].data[0:512,colIndex] = 1

    # Save to a temporary file.
    maskFits[0].writeto(_statmask, output_verify=outputVerify)

def clean_makemask(_mask, _mask_cosmic, _mask_static, wave):
    """
    _mask -- output name for final mask
    _mask_cosmic -- should contain cosmic ray mask
    _mask_static -- should contain supermask + bad columns

    Output:
    _mask is created to be supermask + bad columns + cosmic rays
    _mask will have 0=bad and 1=good pixels (as drizzle expects)
    _mask can be directly passed into drizzle
    """
    # Get the masks to combine
    staticMask = pyfits.getdata(_mask_static)
    cosmicMask = pyfits.getdata(_mask_cosmic)

    mask = staticMask + cosmicMask

    # check subarray
    if ('lp' in wave or 'ms' in wave) and (mask.shape[0] > 512): 
        _lpmask = '/u/ghezgroup/code/python/masks/nirc2_lp_edgemask.fits'
        lpmask = pyfits.getdata(_lpmask)
        mask += lpmask

    # Set to 0 or 1 -- note they are inverted
    weightone = (mask == 0)
    weightzero = (mask != 0)

    # Drizzle expects 0 = bad, 1 = good pixels.
    outMask = np.zeros(mask.shape)
    outMask[weightone] = 1
    outMask[weightzero] = 0

    # Trim 12 rows from top and bottom b/c the distortion solution
    # introduces a torque to the image.
    outMask[1012:1024,0:1024] = 0
    outMask[0:12,0:1024] = 0

    # Write out to file
    pyfits.writeto(_mask, outMask, output_verify=outputVerify)
    #outMask[0].writeto(_mask, output_verify=outputVerify)


def clean_lp(files, nite, wave, refSrc, strSrc, angOff, skyfile):
    """Only here for backwards compatability. You should use clean() instead."""
    clean(files, nite, wave, refSrc, strSrc,
          angOff=angOff, skyfile=skyfile)

def combine(files, wave, outroot, field=None, outSuffix=None,
            trim=0, weight=None, fwhm_max=0, submaps=0, fixDAR=True,
            mask=True):
    """Accepts a list of cleaned images and does a weighted combining after
    performing frame selection based on the Strehl and FWHM.

    Each image must have an associated *.coo file which gives the rough
    position of the reference source.

    @param files: List of integer file numbers to include in combine.
    @type files: list of int
    @param wave: Filter of observations (e.g. 'kp', 'lp', 'h')
    @type wave: string
    @param outroot: The output root name (e.g. '06jullgs'). The final combined
        file names will be <outroot>_<field>_<outSuffix>_<wave>.
        The <field> and <outSuffix> keywords are optional.

        Examples:
        06jullgs_kp for outroot='06jullgs' and wave='kp'
        06jullgs_arch_f1_kp for adding field='arch_f1'
    @type outroot: string
    @kwparam field: Optional field name used to get to clean directory and
        also affects the final output file name. 
    @type field: string
    @kwparam outSuffix: Optional suffix used to modify final output file name. 
    @type outSuffix: string
    @kwparam trim: Optional file trimming based on image quality. Default
        is 0. Set to 1 to turn trimming on.
    @type trim: 0 or 1
    @kwparam weight: Optional weighting. Set to 'strehl' to weight by Strehl,
        as found in strehl_source.txt file. OR set to a file name with the 
        first column being the file name (e.g., c0021.fits) and the second
        column being the weight. Weights will be renormalized to sum to 1.0.
        Default = None, no weighting.
    @type weight: string  
    @kwparam fwhm_max: The maximum allowed FWHM for keeping frames when
        trimming is turned on.
    @type fwhm_max: int
    @kwparam submaps: Set to the number of submaps to be made (def=0).
    @type submaps: int
    """
    # Start out in something like '06maylgs1/reduce/kp/'
    # Setup some files and directories
    waveDir = util.getcwd()
    redDir = util.trimdir( os.path.abspath(waveDir + '../') + '/')
    rootDir = util.trimdir( os.path.abspath(redDir + '../') + '/')
    if (field != None):
        cleanDir = util.trimdir( os.path.abspath(rootDir +
                                                   'clean/' +field+
                                                   '_' +wave) + '/')
        outroot += '_' + field
    else:
        cleanDir = util.trimdir( os.path.abspath(rootDir +
                                                   'clean/' + wave) + '/')
    if (outSuffix != None):
        outroot += '_' + outSuffix

    # This is the final output directory
    comboDir = rootDir + 'combo/'
    util.mkdir(comboDir)

    # Make strings out of all the filename roots.
    allroots = [str(f).zfill(4) for f in files]
    roots = allroots # This one will be modified by trimming

    # This is the output root filename
    _out = comboDir + 'mag' + outroot + '_' + wave
    _sub = comboDir + 'm' + outroot + '_' + wave


    ##########
    # Determine if we are going to trim and/or weight the files 
    # when combining. If so, then we need to determine the Strehl
    # and FWHM for each image. We check strehl source which shouldn't
    # be saturated. *** Hard coded to strehl source ***
    ##########
 
    # Load the strehl_source.txt file
    strehls, fwhm = loadStrehl(cleanDir, roots)

    # Default weights
    # Create an array with length equal to number of frames used,
    # and with all elements equal to 1/(# of files)
    weights = np.array( [1.0/len(roots)] * len(roots) )

    ##########
    # Trimming
    ##########
    if trim:
        roots, strehls, fwhm, weights = trim_on_fwhm(roots, strehls, fwhm, 
                                                     fwhm_max=fwhm_max)

    ##########
    # Weighting
    ##########
    if weight == 'strehl':
        weights = weight_by_strehl(roots, strehls)

    if ((weight != None) and (weight != 'strehl')):
        # Assume weight is set to a filename
        if not os.path.exists(weight):
            raise ValueError('Weights file does not exist, %s' % weight)

        print 'Weights file: ', weight
 
        weights = readWeightsFile(roots, weight)

    # Determine the reference image
    refImage = cleanDir + 'c' + roots[0] + '.fits'
    print 'combine: reference image - %s' % refImage

    ##########
    # Write out a log file. With a list of images in the
    # final combination.
    ##########
    combine_log(_out, roots, strehls, fwhm, weights)

    # See if all images are at same PA, if not, rotate all to PA = 0
    # temporarily. This needs to be done to get correct shifts.
    diffPA = combine_rotation(cleanDir, roots)

    # Make a table of coordinates for the reference source.
    # These serve as initial estimates for the shifts.
    #combine_ref(_out + '.coo', cleanDir, roots, diffPA)
    combine_coo(_out + '.coo', cleanDir, roots, diffPA) 

    # Keep record of files that went into this combine
    combine_lis(_out + '.lis', cleanDir, roots, diffPA)

    # Register images to get shifts.
    shiftsTab = combine_register(_out, refImage, diffPA)

    # Determine the size of the output image from max shifts
    xysize = combine_size(shiftsTab, refImage, _out, _sub, submaps)

    ##########
    # Sort frames -- recall that submaps assume sorted by FWHM.
    ##########
    roots, strehls, fwhm, weights, shiftsTab = sort_frames(roots, strehls, fwhm, weights, shiftsTab)

    # Combine all the images together.
    combine_drizzle(xysize, cleanDir, roots, _out, weights, shiftsTab,
                    wave, diffPA, fixDAR=fixDAR, mask=mask)

    # Now make submaps
    if (submaps > 0):
	combine_submaps(xysize, cleanDir, roots, _sub, weights, 
			shiftsTab, submaps, wave, diffPA, fixDAR=fixDAR, 
                        mask=mask)

    # Remove *.lis_r file & rotated rcoo files, if any - these
    # were just needed to get the proper shifts for xregister
    _lisr = _out + '.lis_r'
    util.rmall([_lisr])
    for i in range(len(allroots)):
        _rcoo = cleanDir + 'c' + str(allroots[i]) + '.rcoo'
        util.rmall([_rcoo])

def rot_img(root, phi, cleanDir):
    """Rotate images to PA=0 if they have a different PA from one
    another. If the entire data set is taken at a single PA, leave
    it as is. Do this only if set includes various PAs.
    """
    pa = str(phi)
    ir.unlearn('rotate')
    ir.rotate.verbose = 'no'
    ir.rotate.boundary = 'constant'
    ir.rotate.constant = 0
    ir.rotate.interpolant = 'spline3'

    inCln = cleanDir + 'c' + root + '.fits'
    outCln = cleanDir + 'r' + root + '.fits'

    util.rmall([outCln])

    if (phi != 0):
        print 'Rotating frame: ',root 
        ir.rotate(inCln, outCln, pa)
    else:
	ir.imcopy(inCln, outCln, verbose='no')

def gcSourceXY(name):
    """Queries label.dat for the xy offset from Sgr A* (in arcsec)
    for the star given as an input

    @param name: name of a star (e.g. 'irs16NE')
    @type name: string

    @returns pos: x and y offset from Sgr A* in arcsec
    @rtype pos: float list (2-elements)
    """
    
    # Read in label.dat
    _labels = '/u/jlu/gc/source_list/label.dat'
    table = asciidata.open(_labels)

    nameCol = (table['column1']).tonumpy().tolist()
    names = [n.strip() for n in nameCol]
    
    try:
	id = names.index(name)

	x = table['column3'][id]
	y = table['column4'][id]
    except ValueError, e:
	print 'Could not find source ' + name + ' in label.dat.'
	x = 0
	y = 0

    return [x,y]
    

def calcStrehl(files, wave, field=None):
    """Make Strehl and FWHM table on the strehl source for all 
    cleaned files.

    @param cleanDir The 'clean' directory.
    @type cleanDir string
    @param roots A list of string root names (e.g. 0001)
    @type roots list
    """
    waveDir = util.getcwd()
    redDir = util.trimdir( os.path.abspath(waveDir + '../') + '/')
    rootDir = util.trimdir( os.path.abspath(redDir + '../') + '/')
    if (field != None):
        cleanDir = util.trimdir( os.path.abspath(rootDir +
                                                   'clean/' +field+
                                                   '_' +wave) + '/')
    else:
        cleanDir = util.trimdir( os.path.abspath(rootDir +
                                                   'clean/' + wave) + '/')

    # Make a list of all the images
    clisFile = cleanDir + 'c.lis'
    _strehl = cleanDir + 'strehl_source.txt'
    _idl = 'idl.strehl.batch'
    tmpFiles = [clisFile, _strehl, _idl]
    util.rmall(tmpFiles)
    
    _clis = open(clisFile, 'w')
    roots = [str(f).zfill(4) for f in files]

    # Loop through all the files and determine the coordinates of the
    # strehl source from what is in the *.coo file.
    for root in roots:
        _coo = cleanDir + 'c' + root + '.coo'
        _fits = cleanDir + 'c' + root + '.fits'

        # Read xstrehl, ystrehl coordinates from header
        hdr = pyfits.getheader(_fits,ignore_missing_end=True)
        xystr = [float(hdr['XSTREHL']), float(hdr['YSTREHL'])]

        _coord = cleanDir + 'c' + root + '.coord'
        file(_coord, 'w').write('%7.2f  %7.2f\n' % (xystr[0], xystr[1]))

        _clis.write('%s\n' % _fits)

    _clis.close()

    # Now call Marcos's strehl widget in a command line format.
    batchFile = open(_idl, 'w')
    batchFile.write("cmd_strehl_widget, '" + clisFile + "', /list, ")
    batchFile.write("output='" + _strehl + "', aper=0.3, /silent\n")
    batchFile.write("exit\n")
    batchFile.close()

    os.system('idl < idl.strehl.batch > idl.strehl.batch.log')

    # Check that the number of lines in the resulting strehl file
    # matches the number of images we have. If not, some of the images
    # are bad and were dropped.
    strehlTable = asciidata.open(_strehl)
    if len(roots) != strehlTable.nrows:
        print len(roots), strehlTable.nrows
        # Figure out the dropped files.
        droppedFiles = []
        for rr in roots:
            foundIt = False
            for ss in strehlTable[0]._data:
                if rr in ss:
                    foundIt = True
                    continue
            if foundIt == False:
                droppedFiles.append(rr)

        raise RuntimeError('calcStrehl: Strehl widget lost files: ', 
                           droppedFiles)

def weight_by_strehl(roots, strehls):        
    """
    Calculate weights based on the strehl of each image. 
    This does some intelligent handling for REALLY bad data quality.
    """
    # Set negative Strehls to the lowest detected strehl.
    bidx = (np.where(strehls <= 0))[0]
    gidx = (np.where(strehls > 0))[0]
    if len(bidx) > 0: 
        badroots = [roots[i] for i in bidx]
        print 'Found files with incorrect Strehl. May be incorrectly'
        print 'weighted. Setting weights to minimum weight. '
        print '\t' + ','.join(badroots)

    strehl_min = strehls[gidx].min()
    strehls[bidx] = strehl_min
	    
    # Now determine a fractional weight
    wgt_tot = sum(strehls)
    weights = strehls / wgt_tot

    return weights


def trim_on_fwhm(roots, strehls, fwhm, fwhm_max=0):
    """
    Take a list of files and trim based on the FWHM. All files that have a 
    FWHM < 1.25 * FWHM.min()
    are kept. 

    The returned arrays contain only those files that pass the above criteria.
    """
    # Trim level (fwhm) can be passed in or determined
    # dynamically.
    if (fwhm_max == 0):
        # Determine the minimum FWHM
        idx = np.where(fwhm > 0)
        fwhm_min = fwhm[idx].min()
        
        # Maximum allowed FWHM to keep frame
        fwhm_max = 1.25 * fwhm_min
        
    # Pull out those we want to include in the combining
    keep = np.where((fwhm <= fwhm_max) & (fwhm > 0))[0]
    strehls = strehls[keep]
    fwhm = fwhm[keep]
    roots = [roots[i] for i in keep]
    weights = np.array( [1.0/len(roots)] * len(roots) )
	
    print 'combine: Keeping %d frames with FWHM < %4.1f' \
        % (len(roots), fwhm_max)

    return (roots, strehls, fwhm, weights)


def readWeightsFile(roots, weightFile):
    """
    Expects a file of the format:
    column1 = file name (e.g. c0001.fits).
    column2 = weights.
    """
    
    weightsTable = asciidata_open_trimmed(roots, weightFile)

    weights = weightsTable[1].tonumpy()

    # Renormalize so that weights add up to 1.0
    weights /= weights.sum()

    # Double check that we have the same number of 
    # lines in the weightsTable as files.
    if (len(weights) != len(roots)):
	print 'Wrong number of lines in  ' + weightFile

    return weights


def loadStrehl(cleanDir, roots):
    """
    Load Strehl and FWHM info. The file format will be 
    column1 = name of cleaned fits file (e.g. c0001.fits). 
              Expects single character before a 4 digit number.
    column2 = strehl
    column3 = RMS error (nm)
    column4 = FWHM (mas)
    column5 = MJD (UT)
    """
    _strehl = cleanDir + 'strehl_source.txt'

    # Read in file and get strehls and FWHMs
    strehlTable = asciidata_open_trimmed(roots, _strehl)
    strehls = strehlTable[1].tonumpy()
    fwhm = strehlTable[3].tonumpy()

    # Double check that we have the same number of 
    # lines in the strehlTable as files.
    if (len(strehls) != len(roots)):
	print 'Wrong number of lines in  ' + _strehl

    return (strehls, fwhm)

def asciidata_open_trimmed(outroots, tableFileName):
    """
    Takes a list of values (listed in tableFileName) and trim them down based on
    the desired output list of root files names (outroots).
    """
    table = asciidata.open(tableFileName)
    newtable = asciidata.create(table.ncols, len(outroots))
     
    for rr in range(len(outroots)):
        for ii in range(table.nrows):
            if outroots[rr] in table[0][ii]:
                for cc in range(table.ncols):
                    newtable[cc][rr] = table[cc][ii]
                table.delete(ii)
                break

    return newtable
 

def combine_drizzle(imgsize, cleanDir, roots, outroot, weights, shifts,
                    wave, diffPA, fixDAR=True, mask=True):
    _fits = outroot + '.fits'
    _tmpfits = outroot + '_tmp.fits'
    _wgt = outroot + '_sig.fits'
    _dlog = outroot + '_driz.log'
    _maxFile = outroot + '.max'

    util.rmall([_fits, _tmpfits, _wgt, _dlog])

    # Make directory for individually drizzled and pre-shifted images.
    #util.mkdir(cleanDir + 'shifted')

    # Prep drizzle stuff
    setup_drizzle(imgsize)

    # BUG: with context... when many files are drizzled
    # together, a new, bigger context file is created, but this 
    # fails with a Bus error.
    #ir.drizzle.outcont = _ctx
    ir.drizzle.outcont = ''
    ir.drizzle.fillval = 0.

    satLvl_combo = 0.0

    # Set a cleanDir variable in IRAF. This avoids the long-filename problem.
    ir.set(cleanDir=cleanDir)

    print 'combine: drizzling images together'
    f_dlog = open(_dlog, 'a')
    for i in range(len(roots)):
        # Cleaned image
        _c = cleanDir + 'c' + roots[i] + '.fits'
        _c_ir = _c.replace(cleanDir, 'cleanDir$')

        # Cleaned but distorted image
        _cd = cleanDir + 'distort/cd' + roots[i] + '.fits'
        _cdwt = cleanDir + 'weight/cdwt.fits'
        _cd_ir = _cd.replace(cleanDir, 'cleanDir$')
        _cdwt_ir = _cdwt.replace(cleanDir, 'cleanDir$')

        util.rmall([_cdwt])

        # Multiply each distorted image by it's weight
        ir.imarith(_cd_ir, '*', weights[i], _cdwt_ir)

        # Fix the ITIME header keyword so that it matches (weighted).
        # Drizzle will add all the ITIMEs together, just as it adds the flux.
        itime_tmp = ir.hselect(_cdwt_ir, "ITIME", "yes", Stdout=1)
        itime = float(itime_tmp[0]) * weights[i]
        ir.hedit(_cdwt_ir, 'ITIME', itime, verify='no', show='no')

        # Get pixel shifts
        xsh = shifts[1][i]
        ysh = shifts[2][i]

        # Read in PA of each file to feed into drizzle for rotation
        hdr = pyfits.getheader(_c,ignore_missing_end=True)
        phi = nirc2.getPA(hdr)
        if (diffPA == 1):
            ir.drizzle.rot = phi

        if (fixDAR == True):
            darRoot = _cdwt.replace('.fits', 'geo')
            (xgeoim, ygeoim) = dar.darPlusDistortion(_cdwt, darRoot,
                                                     xgeoim=distXgeoim,
                                                     ygeoim=distYgeoim)

            xgeoim = xgeoim.replace(cleanDir, 'cleanDir$')
            ygeoim = ygeoim.replace(cleanDir, 'cleanDir$')
            ir.drizzle.xgeoim = xgeoim
            ir.drizzle.ygeoim = ygeoim
        
        # Drizzle this file ontop of all previous ones.
        f_dlog.write(time.ctime())

        if (mask == True):
            _mask = 'cleanDir$masks/mask' + roots[i] + '.fits'
        else:
            _mask = ''
        ir.drizzle.in_mask = _mask
        ir.drizzle.outweig = _wgt
        ir.drizzle.xsh = xsh
        ir.drizzle.ysh = ysh

        ir.drizzle(_cdwt_ir, _tmpfits, Stdout=f_dlog)

        # Create .max file with saturation level for final combined image 
        # by weighting each individual satLevel and summing.
        # Read in each satLevel from individual .max files
        _max = cleanDir + 'c' + roots[i] + '.max'
        getsatLvl = asciidata.open(_max)
        satLvl = getsatLvl[0].tonumpy()
        satLvl_wt = satLvl * weights[i]
        satLvl_combo += satLvl_wt[0]

    f_dlog.close()

    print 'satLevel for combo image = ', satLvl_combo
    # Write the combo saturation level to a file
    _max = open(_maxFile, 'w')
    _max.write('%15.4f' % satLvl_combo)
    _max.close()

    # Clean up the drizzled image of any largely negative values.
    # Don't do this! See how starfinder handles really negative pixels,
    # and if all goes well...don't ever correct negative pixels to zero.
    text = ir.imstatistics(_tmpfits, fields='mean,stddev', nclip=5, 
                           lsigma=10, usigma=1, format=0, Stdout=1)
    vals = text[0].split()
    sci_mean = float(vals[0])
    sci_stddev = float(vals[1])

    fits = pyfits.open(_tmpfits)

    # Find and fix really bad pixels
    idx = np.where(fits[0].data < (sci_mean - 10*sci_stddev))
    fits[0].data[idx] = sci_mean - 10*sci_stddev

    # Set the ROTPOSN value for the combined image.
    if (diffPA == 1):
        phi = 0.7
        fits[0].header.update('ROTPOSN', "%.5f" % phi,
                              'rotator user position')

    # Add keyword with distortion image information
    fits[0].header.update('DISTCOEF', "%s" % distCoef,
                          'Distortion Coefficients File')
    fits[0].header.update('DISTORTX', "%s" % distXgeoim,
                          'X Distortion Image')
    fits[0].header.update('DISTORTY', "%s" % distYgeoim,
                          'Y Distortion Image')

    # Save to final fits file.
    fits[0].writeto(_fits, output_verify=outputVerify)
    util.rmall([_tmpfits, _cdwt])

def combine_submaps(imgsize, cleanDir, roots, outroot, weights, 
		    shifts, submaps, wave, diffPA, fixDAR=True, mask=True):
    """
    Assumes the list of roots are pre-sorted based on quality. Images are then
          divided up with every Nth image going into the Nth submap.

    mask: (def=True) Set to false for maser mosaics since they have only
          one image at each positions. Masking produces artifacts that
          Starfinder can't deal with.
    """

    extend = ['_1', '_2', '_3']
    _out = [outroot + end for end in extend]
    _fits = [o + '.fits' for o in _out]
    _tmp = [o + '_tmp.fits' for o in _out]
    _wgt = [o + '_sig.fits' for o in _out]
    _log = [o + '_driz.log' for o in _out]
    _max = [o + '.max' for o in _out]

    util.rmall(_fits + _tmp + _wgt + _log + _max)

    # Prep drizzle stuff
    setup_drizzle(imgsize)
    ir.drizzle.outcont = ''

    satLvl_tot = np.zeros(submaps, dtype=float)
    satLvl_sub = np.zeros(submaps, dtype=float)

    print 'combine: drizzling sub-images together'
    f_log = [open(log, 'a') for log in _log]

    # Final normalization factor	
    weightsTot = np.zeros(submaps, dtype=float)

    for i in range(len(roots)):
        # Cleaned image
        _c = cleanDir + 'c' + roots[i] + '.fits'

	# Cleaned but distorted image
        _cd = cleanDir + 'distort/cd' + roots[i] + '.fits'
        cdwt = cleanDir + 'weight/cdwt.fits'

        # Multiply each distorted image by it's weight
        util.rmall([cdwt])
        ir.imarith(_cd, '*', weights[i], cdwt)

        # Fix the ITIME header keyword so that it matches (weighted).
        # Drizzle will add all the ITIMEs together, just as it adds the flux.
        itime_tmp = ir.hselect(cdwt, "ITIME", "yes", Stdout=1)
        itime = float(itime_tmp[0]) * weights[i]
        ir.hedit(cdwt, 'ITIME', itime, verify='no', show='no')

	# Get pixel shifts
	xsh = shifts[1][i]
	ysh = shifts[2][i]

	# Determine which submap we should be drizzling to.
	sub = int(i % submaps)
	fits = _tmp[sub]
	wgt = _wgt[sub]
	log = f_log[sub]

        # Read in PA of each file to feed into drizzle for rotation
        hdr = pyfits.getheader(_c,ignore_missing_end=True)
        phi = nirc2.getPA(hdr)
        if (diffPA == 1):
            ir.drizzle.rot = phi

        # Calculate saturation level for submaps 
        # by weighting each individual satLevel and summing.
        # Read in each satLevel from individual .max files
        max_indiv = cleanDir + 'c' + roots[i] + '.max'
        getsatLvl = asciidata.open(max_indiv)
        satLvl = getsatLvl[0].tonumpy()
        satLvl_wt = satLvl * weights[i]
        satLvl_tot[sub] += satLvl_wt[0]

	# Add up the weights that go into each submap
	weightsTot[sub] += weights[i]

        satLvl_sub[sub] = satLvl_tot[sub] / weightsTot[sub]

        if (fixDAR == True):
            darRoot = cdwt.replace('.fits', 'geo')
            (xgeoim, ygeoim) = dar.darPlusDistortion(cdwt, darRoot,
                                                     xgeoim=distXgeoim,
                                                     ygeoim=distYgeoim)
            ir.drizzle.xgeoim = xgeoim
            ir.drizzle.ygeoim = ygeoim
        
        # Drizzle this file ontop of all previous ones.
        log.write(time.ctime())

        if (mask == True):
            _mask = cleanDir + 'masks/mask' + roots[i] + '.fits'
        else:
            _mask = ''
        ir.drizzle.in_mask = _mask
        ir.drizzle.outweig = wgt
        ir.drizzle.xsh = xsh
        ir.drizzle.ysh = ysh
        
        ir.drizzle(cdwt, fits, Stdout=log)

    for f in f_log:
	f.close()

    print 'satLevel for submaps = ', satLvl_sub
    # Write the saturation level for each submap to a file
    for l in range(submaps):
        _maxsub = open(_max[l], 'w')
        _maxsub.write('%15.4f' % satLvl_sub[l])
        _maxsub.close()

    for s in range(submaps):
        # Clean up the drizzled image of any largely negative values.
        # Don't do this! See how starfinder handles really negative pixels,
        # and if all goes well...don't ever correct negative pixels to zero.
        text = ir.imstatistics(_tmp[s], fields='mean,stddev', nclip=5, 
                               lsigma=10, usigma=1, format=0, Stdout=1)
        vals = text[0].split()
        sci_mean = float(vals[0])
        sci_stddev = float(vals[1])

	fits = pyfits.open(_tmp[s])

        # Find and fix really bad pixels
        idx = np.where(fits[0].data < (sci_mean - 10*sci_stddev))
        fits[0].data[idx] = 0.0

        # Normalize properly
	fits[0].data = fits[0].data / weightsTot[s]

        # Fix the ITIME header keyword so that it matches (weighted).
        itime = fits[0].header.get('ITIME')
        itime /= weightsTot[s]
        fits[0].header.update('ITIME', '%.5f' % itime)
        
        # Set the ROTPOSN value for the combined submaps. 
        if (diffPA == 1):
            phi = 0.7
            fits[0].header.update('ROTPOSN', "%.5f" % phi,
                                  'rotator user position')

        # Add keyword with distortion image information
        fits[0].header.update('DISTORTX', "%s" % distXgeoim,
                              'X Distortion Image')
        fits[0].header.update('DISTORTY', "%s" % distYgeoim,
                              'Y Distortion Image')

	fits[0].writeto(_fits[s], output_verify=outputVerify)


    util.rmall(_tmp)
    util.rmall([cdwt])


def combine_rotation(cleanDir, roots):
    """
    Determine if images are different PAs. If so, then
    temporarily rotate the images for xregister to use
    in order to get image shifts that are fed into drizzle.

    WARNING: If multiple PAs are found, then everything
    is rotated to PA = 0.
    """
    diffPA = 0
    firstFile = cleanDir + 'c' + roots[0] + '.fits'
    hdr = pyfits.getheader(firstFile,ignore_missing_end=True)
    phiRef = nirc2.getPA(hdr)

    for root in roots:
	_fits = cleanDir + 'c' + root + '.fits'
        hdr = pyfits.getheader(_fits,ignore_missing_end=True)
        phi = nirc2.getPA(hdr)
        diff = phi - phiRef

        if (diff != 0.0):
            print 'Different PAs found'
            diffPA = 1
            break

    if (diffPA == 1):
        for root in roots:
	    _fits = cleanDir + 'c' + root + '.fits'
            hdr = pyfits.getheader(_fits,ignore_missing_end=True)
            phi = nirc2.getPA(hdr)
            rot_img(root, phi, cleanDir)

    return (diffPA)

def sort_frames(roots, strehls, fwhm, weights, shiftsTab):
    sidx = np.argsort(fwhm)

    # Make sorted lists.
    strehls = strehls[sidx]
    fwhm = fwhm[sidx]
    weights = weights[sidx]
    roots = [roots[i] for i in sidx]
    shiftsX = shiftsTab[1].tonumpy()
    shiftsX = shiftsX[sidx]
    shiftsY = shiftsTab[2].tonumpy()
    shiftsY = shiftsY[sidx]
    
    # Move all the ones with fwhm = -1 to the end
    gidx = (np.where(fwhm > 0))[0]
    bidx = (np.where(fwhm <= 0))[0]
    goodroots = [roots[i] for i in gidx]
    badroots = [roots[i] for i in bidx]
    if len(bidx) > 0: 
	print 'Found files with incorrect FWHM. They may be rejected.'
	print '\t' + ','.join(badroots)
	
    strehls = np.concatenate([strehls[gidx], strehls[bidx]])
    fwhm = np.concatenate([fwhm[gidx], fwhm[bidx]])
    weights = np.concatenate([weights[gidx], weights[bidx]])
    shiftsX = np.concatenate([shiftsX[gidx], shiftsX[bidx]])
    shiftsY = np.concatenate([shiftsY[gidx], shiftsY[bidx]])
    roots = goodroots + badroots

    newShiftsTab = asciidata.create(shiftsTab.ncols, shiftsTab.nrows)
    for rr in range(newShiftsTab.nrows):
        newShiftsTab[0][rr] = roots[rr]
        newShiftsTab[1][rr] = shiftsX[rr]
        newShiftsTab[2][rr] = shiftsY[rr]

    return (roots, strehls, fwhm, weights, newShiftsTab)


def combine_ref(coofile, cleanDir, roots, diffPA):
    """
    Pulls reference star coordinates from image header keywords.
    """
    # Delete any previously existing file
    util.rmall([coofile])

    cFits = [cleanDir + 'c' + r + '.fits' for r in roots]

    _allCoo = open(coofile, 'w')

    # write reference source coordinates
    hdr = pyfits.getheader(cFits[0],ignore_missing_end=True)
    _allCoo.write(' ' + hdr['XREF'] + '   ' + hdr['YREF'] + '\n')

    # write all coordinates, including reference frame
    for i in range(len(roots)):
        hdr = pyfits.getheader(cFits[i],ignore_missing_end=True)
        _allCoo.write(' ' + hdr['XREF'] + '   ' + hdr['YREF'] + '\n')

    _allCoo.close()


def combine_coo(coofile, cleanDir, roots, diffPA):
    """
    Pulls reference star coordinates from *.coo files.
    """
    # Delete any previously existing file
    util.rmall([coofile])

    # If images were rotated because of differing PAs, make a
    # different input list
    if (diffPA == 1):
        cCoos = [cleanDir + 'c' + r + '.rcoo' for r in roots]
    else:
        cCoos = [cleanDir + 'c' + r + '.coo' for r in roots]

    # Need to make table of coordinates of a reference source. These
    # will be used as initial estimates of the shifts (they don't necessarily
    # need to be real sources).
    _allCoo = open(coofile, 'w')
    
    # First line must be the coordinates in the reference image
    _allCoo.write(open(cCoos[0], 'r').read())

    # Now loop through all files (including the reference) and print
    # coordinates of same reference source.
    for i in range(len(roots)):
	_allCoo.write(open(cCoos[i], 'r').read())

    _allCoo.close()


def combine_lis(outfile, cleanDir, roots, diffPA):
    # Delete previously existing file
    util.rmall([outfile])

    cFits = [cleanDir + 'c' + r + '.fits' for r in roots]

    # Write all the files to a list
    f_lis = open(outfile, 'w')
    f_lis.write('\n'.join(cFits) + '\n')
    f_lis.close()

    # If images were rotated because of differing PAs, make a
    # different input list for xregister (to get shifts)
    if (diffPA == 1):
        rFits = [cleanDir + 'r' + r + '.fits' for r in roots]
        out = outfile + '_r'
        f_lis = open(out, 'w')
        f_lis.write('\n'.join(rFits) + '\n')
        f_lis.close()

def combine_register(outroot, refImage, diffPA):
    shiftFile = outroot + '.shifts'
    util.rmall([shiftFile])

    # xregister parameters
    ir.immatch
    ir.unlearn('xregister')
    ir.xregister.coords = outroot + '.coo'
    ir.xregister.output = ''
    ir.xregister.append = 'no'
    ir.xregister.databasefmt = 'no'
    ir.xregister.verbose = 'no'

    print 'combine: registering images'
    if (diffPA == 1):
        input = '@' + outroot + '.lis_r'
    else:
        input = '@' + outroot + '.lis'

    regions = '[*,*]'
    # print 'input = ', input
    print 'refImage = ', refImage
    print 'regions = ', regions
    print 'shiftFile = ', shiftFile

    fileNames = asciidata.open(input[1:])
    coords = asciidata.open(outroot + '.coo')
    shiftsTable = asciidata.create(3, fileNames.nrows)
    
    for ii in range(fileNames.nrows):
        inFile = fileNames[0][ii]

        tmpCooFile = outroot + '_tmp.coo'
        _coo = open(tmpCooFile, 'w')
        _coo.write('%.2f  %.2f\n' % (coords[0][0], coords[1][0]))
        _coo.write('%.2f  %.2f\n' % (coords[0][ii+1], coords[1][ii+1]))
        _coo.close()

        util.rmall([shiftFile])
        print 'inFile = ', inFile
        ir.xregister.coords = tmpCooFile
        ir.xregister(inFile, refImage, regions, shiftFile)

        _shifts = asciidata.open(shiftFile)
        shiftsTable[0][ii] = _shifts[0][0]
        shiftsTable[1][ii] = _shifts[1][0]
        shiftsTable[2][ii] = _shifts[2][0]

    # # Read in the shifts file. Column format is:
    # # Filename.fits  xshift  yshift
    # shiftsTable = asciidata.open(shiftFile)

    util.rmall([shiftFile])
    shiftsTable.writeto(shiftFile)

    return (shiftsTable)


def combine_log(outroot, roots, strehls, fwhm, weights):
    _log = outroot + '.log'
    util.rmall([_log])
    
    f_log = open(_log, 'w')
    for i in range(len(roots)):
	f_log.write('c%s %6.2f %5.2f %6.3f\n' % 
		    (roots[i], fwhm[i], strehls[i], weights[i]))

    f_log.close()
	
def combine_size(shiftsTable, refImage, outroot, subroot, submaps):
    """Determine the final size of the fully combined image. Use the
    shifts stored in the shiftsTable.

    @param shiftsTable: Table with x and y shifts for each image
    @type shiftsTable: asciidata table
    @param refImage: The reference image from which the shifts are
        calculated from.
    @type refImage: string
    @param outroot: The name of the file for which shift information
        will be stored. The filename will be <outroot>.coo.
    @type outroot: string
    @param subroot: Same as outroot but for submaps
    @type subroot: string
    @param submaps: number of submaps
    @type sbumaps: int
    """
    x_allShifts = shiftsTable[1].tonumpy()
    y_allShifts = shiftsTable[2].tonumpy()

    xhi = abs(x_allShifts.max())
    xlo = abs(x_allShifts.min())
    yhi = abs(y_allShifts.max())
    ylo = abs(y_allShifts.min())

    # Make sure to include the edges of all images. 
    # Might require some extra padding on one side.
    maxoffset = max([xlo, xhi, ylo, yhi])

    orig_img = pyfits.getdata(refImage)
    orig_size = (orig_img.shape)[0]
    padd = 8.0

    # Read in 16C's position in the ref image and translate
    # it into the coordinates of the final main and sub maps.
    hdr = pyfits.getheader(refImage,ignore_missing_end=True)
    xrefSrc = float(hdr['XREF'])
    yrefSrc = float(hdr['YREF'])

    xrefSrc = xrefSrc + (maxoffset + padd)
    yrefSrc = yrefSrc + (maxoffset + padd)

    cooMain = [outroot + '.coo']
    cooSubs = ['%s_%d.coo' % (subroot, i) for i in range(submaps+1)]
    cooAll = cooMain + cooSubs

    util.rmall(cooAll)
    for coo in cooAll:
	_allCoo = open(coo, 'w')
	_allCoo.write('%9.3f %9.3f\n' % (xrefSrc, yrefSrc))
	_allCoo.close()
    
    xysize = float(orig_size) + ((maxoffset + padd) * 2.0)
    print 'combine: Size of output image is %d' % xysize

    return xysize

def setup_drizzle(imgsize):
    """Setup drizzle parameters for NIRC2 data.
    @param imgsize: The size (in pixels) of the final drizzle image.
    This assumes that the image will be square.
    @type imgsize: int
    @param mask: The name of the mask to use during
    drizzle.
    @param type: str
    """
    # Setup the drizzle parameters we will use
    ir.module.load('stsdas', doprint=0, hush=1)
    ir.module.load('analysis', doprint=0, hush=1)
    ir.module.load('dither', doprint=0, hush=1)
    ir.unlearn('drizzle')
    ir.drizzle.outweig = ''
    ir.drizzle.in_mask = ''
    ir.drizzle.wt_scl = 1
    ir.drizzle.outnx = imgsize
    ir.drizzle.outny = imgsize
    ir.drizzle.pixfrac = 1
    ir.drizzle.kernel = 'lanczos3'
    ir.drizzle.scale = 1
    ir.drizzle.coeffs = distCoef
    ir.drizzle.xgeoim = distXgeoim
    ir.drizzle.ygeoim = distYgeoim
    ir.drizzle.shft_un = 'input'
    ir.drizzle.shft_fr = 'output'
    ir.drizzle.align = 'center'
    ir.drizzle.expkey = 'ITIME'
    ir.drizzle.in_un = 'counts'
    ir.drizzle.out_un = 'counts'

def clean_drizzle(_bp, _cd, _wgt, _dlog, fixDAR=True):
    if (fixDAR == True):
        darRoot = _cd.replace('.fits', 'geo')

        # Future: add distortion xgeoim, ygeoim to inputs here.
        (xgeoim, ygeoim) = dar.darPlusDistortion(_bp, darRoot,
                                                 xgeoim=distXgeoim,
                                                 ygeoim=distYgeoim)

        ir.drizzle.xgeoim = xgeoim
        ir.drizzle.ygeoim = ygeoim

    ir.drizzle(_bp, _cd, outweig=_wgt, Stdout=_dlog)

def clean_cosmicrays(_ff, _mask, wave):
    """Clean the image of cosmicrays and make a mask containing the location
    of all the cosmicrays. The CR masks can later be used in combine() to
    keep cosmicrays from being included.

    @param _ff: Flat fielded file on which to fix cosmic rays. A new
        image will be created with the _f appended to it.
    @type _ff: string
    @param _mask: The filename used for the resulting mask.
    @type _mask: string
    @parram wave: The filter of the observations (e.g. 'kp', 'lp'). This
        is used to determine different thresholds for CR rejection.
    @type wave: string
    """
    # Determine the threshold at which we should start looking
    # for cosmicrays. Need to figure out the mean level of the
    # background.
    text_output = ir.imstatistics(_ff, fields='mean,stddev', 
                                  usigma=2, lsigma=5, nclip=5, 
                                  format=0, Stdout=1)
    values = text_output[0].split()
    mean = float(values[0])
    stddev = float(values[1])
    
    # CR candidates are those that exceed surrounding pixels by
    # this threshold amount.
    crthreshold = 5.0*stddev

    fluxray = 13.
    if 'h' in wave:
        fluxray = 10.
    if 'kp' in wave:
        fluxray = 13.
    if 'lp' in wave:
        fluxray = 10.0
    if 'ms' in wave:
        fluxray = 10.0

    ir.module.load('noao', doprint=0, hush=1)
    ir.module.load('imred', doprint=0, hush=1)
    ir.module.load('crutil', doprint=0, hush=1)
    ir.unlearn('cosmicrays')

    ir.cosmicrays(_ff, ' ', crmasks=_mask, thresho=crthreshold, 
                  fluxrat=fluxray, npasses=10., window=7, 
                  interac='no', train='no', answer='NO')

    ir.imcopy(_mask+'.pl', _mask, verbose='no')
    ir.delete(_mask+'.pl')

def clean_cosmicrays2(_ff, _ff_cr, _mask, wave):
    """Clean the image of cosmicrays and make a mask containing the location
    of all the cosmicrays. The CR masks can later be used in combine() to
    keep cosmicrays from being included.

    @param _ff: Flat fielded file on which to fix cosmic rays. A new
        image will be created with the _f appended to it.
    @type _ff: string
    @param _ff_cr: Output image with cosmicrays fixed.
    @type _ff_cr: string
    @param _mask: The filename used for the resulting mask.
    @type _mask: string
    @parram wave: The filter of the observations (e.g. 'kp', 'lp'). This
        is used to determine different thresholds for CR rejection.
    @type wave: string
    """
    # Determine the threshold at which we should start looking
    # for cosmicrays. Need to figure out the mean level of the
    # background.
    text_output = ir.imstatistics(_ff, fields='mean,stddev', 
                                  usigma=2, lsigma=5, nclip=5, 
                                  format=0, Stdout=1)
    values = text_output[0].split()
    mean = float(values[0])
    stddev = float(values[1])

    gain = 4.0
    tmp = ir.hselect(_ff, 'SAMPMODE,MULTISAM', 'yes', Stdout=1)
    tmp2 = tmp[0].split()
    sampmode = int(tmp2[0])
    multisam = int(tmp2[1])
    if sampmode == 2:
        readnoise = 60
    else:
        readnoise = 15.0 * (16.0 / multisam)**0.5

    
    from jlu.util import cosmics
    img, hdr = pyfits.getdata(_ff, header=True)
    c = cosmics.cosmicsimage(img, gain=gain, readnoise=readnoise,
                             sigclip=10, sigfrac=0.5, objlim=5.0)
    c.run(maxiter=3)
    pyfits.writeto(_ff_cr, c.cleanarray, hdr,
                   clobber=True, output_verify=outputVerify)
    pyfits.writeto(_mask, np.where(c.mask==True, 1, 0), hdr,
                   clobber=True, output_verify=outputVerify)

def clean_persistance(_n, _pers):
    """
    Make masks of the persistance to be used in combining the images
    later on.
    """
    # Read in image
    fits = pyfits.open(_n)
    img = fits[0].data
    
    # Define the high pixels
    persPixels = where(img > 12000)

    # Set saturated pixels to 0, good pixels to 1
    fits[0].data[persPixels] = 0
    fits[0].data = fits[0].data / fits[0].data

    # Save to an image
    fits[0].writeto(_pers, output_verify=outputVerify)


def clean_bkgsubtract(_ff_f, _bp):
    """Do additional background subtraction of any excess background
    flux. This isn't strictly necessary since it just removes a constant."""
    # Calculate mean and STD for science image
    text = ir.imstatistics(_ff_f, fields='mean,stddev', nclip=20, 
                           lsigma=10, usigma=1, format=0, Stdout=1)
    vals = text[0].split()
    sci_mean = float(vals[0])
    sci_stddev = float(vals[1])

    # Excess background flux at (mean - 2*std)
    bkg = sci_mean - (2.0 * sci_stddev)
    #print 'Bkg mean = %5d +/- %5d   bkg = %5d  Name = %s' % \
    #      (sci_mean, sci_stddev, bkg, _ff_f)
    
    # Open old, subtract BKG
    fits = pyfits.open(_ff_f)

    # Find really bad pixels
    idx = np.where(fits[0].data < (sci_mean - 10*sci_stddev))

    # Subtract background
    fits[0].data -= bkg

    # Fix really bad negative pixels.
    fits[0].data[idx] = 0.0

    # Write to new file
    fits[0].writeto(_bp, output_verify=outputVerify)

    # Return the background we subtracted off
    return bkg

def clean_makecoo(_ce, _cc, root, refSrc, strSrc, aotsxyRef, radecRef, clean):
    """Make the *.coo file for this science image. Use the difference
    between the AOTSX/Y keywords from a reference image and each science
    image to tell how the positions of the two frames are related.

    @param _ce: Name of the input cleaned file.
    @type _ce: string
    @param _cc: Name of the output header modified image.
    @type _cc: string
    @param root: Integer root of this file number
    @type root: int
    @param refSrc: Array with the X/Y positions of the reference source.
        This will be put into the image header and the *.coo file.
    @type refSrc: array of floats with length=2 [x, y]
    @param strSrc: Array with the X/Y positions of the strehl source.
        This will be put into the image header.
    @type strSrc: array of floats with length=2 [x, y]
    @param aotsxyRef: The AOTSX/Y header values from the reference image.
    @type aotsxyRef: array of floats with length=2 [x, y]
    @param radecRef: The RA/DEC header values from the reference image.
    @type radecRef: array of floats with length=2 [x, y]
    @param clean: The clean directory.
    @type clean: string
    """
    hdr = pyfits.getheader(_ce,ignore_missing_end=True)

    radec = [float(hdr['RA']), float(hdr['DEC'])]
    aotsxy = nirc2.getAotsxy(hdr)

    # Determine the image's PA and plate scale
    phi = nirc2.getPA(hdr)
    scale = nirc2.getScale(hdr)

    # Calculate the pixel offsets from the reference image
    # We've been using aotsxy2pix, but the keywords are wrong
    # for 07maylgs and 07junlgs
    #d_xy = nirc2.radec2pix(radec, phi, scale, radecRef)
    d_xy = nirc2.aotsxy2pix(aotsxy, scale, aotsxyRef)

    # In the new image, find the REF and STRL coords
    xref = refSrc[0] + d_xy[0]
    yref = refSrc[1] + d_xy[1]
    xstr = strSrc[0] + d_xy[0]
    ystr = strSrc[1] + d_xy[1]

    # re-center stars to get exact coordinates
    centBox = 12.0
    text = ir.imcntr(_ce, xref, yref, cbox=centBox, Stdout=1)
    values = text[0].split()
    xref = float(values[2])
    yref = float(values[4])

    text = ir.imcntr(_ce, xstr, ystr, cbox=centBox, Stdout=1)
    values = text[0].split()
    xstr = float(values[2])
    ystr = float(values[4])

    # write reference star x,y to fits header
    fits = pyfits.open(_ce)
    fits[0].header.update('XREF', "%.3f" %xref,
                          'Cross Corr Reference Src x')
    fits[0].header.update('YREF', "%.3f" %yref,
                          'Cross Corr Reference Src y')
    fits[0].header.update('XSTREHL', "%.3f" %xstr,
                          'Strehl Reference Src x')
    fits[0].header.update('YSTREHL', "%.3f" %ystr,
                          'Strehl Reference Src y')
    fits[0].writeto(_cc, output_verify=outputVerify)

    file('c'+root+'.coo', 'w').write('%7.2f  %7.2f\n' % (xref, yref))

    # Make a temporary rotated coo file, in case there are any data sets
    # with various PAs; needed for xregister; remove later
    xyRef_rot = nirc2.rotate_coo(xref, yref, phi)
    xref_r = xyRef_rot[0]
    yref_r = xyRef_rot[1]

    xyStr_rot = nirc2.rotate_coo(xstr, ystr, phi)
    xstr_r = xyStr_rot[0]
    ystr_r = xyStr_rot[1]

    file(clean+'c'+root+'.rcoo', 'w').write('%7.2f  %7.2f\n' % (xref_r, yref_r))

def mosaic_ref(outFile, cleanDir, roots, diffPA):
    """Calculate an initial guess at the offsets between mosaic frames.
    using the AOTSX/Y keywords from a reference image and each science
    image to tell how the positions of the two frames are related.

    @param cleanDir: Name of the input cleaned file.
    @type cleanDir: string
    @param roots: List of root filenames
    @type roots: list of strings
    @param diffPA: 1 = found different PAs so use rot images.
    @type difPA: int
    """
    if (diffPA == 1):
	fileNames = [cleanDir + 'r' + root + '.fits' for root in roots]
    else: 
	fileNames = [cleanDir + 'c' + root + '.fits' for root in roots]

    hdrRef = pyfits.getheader(fileNames[0],ignore_missing_end=True)
    aotsxyRef = nirc2.getAotsxy(hdrRef)

    # Determine the image's PA and plate scale
    phi = nirc2.getPA(hdrRef)
    scale = nirc2.getScale(hdrRef)

    _out = open(outFile, 'w')

    # First line of shifts file must be for a reference
    # image (assumed to be the first image).
    _out.write('%7.2f  %7.2f\n' % (0.0, 0.0))

    for rr in range(len(roots)):
	hdr = pyfits.getheader(fileNames[rr],ignore_missing_end=True)
	aotsxy = nirc2.getAotsxy(hdr)

	# Calculate the pixel offsets from the reference image
	# We've been using aotsxy2pix, but the keywords are wrong
	# for 07maylgs and 07junlgs
	d_xy = nirc2.aotsxy2pix(aotsxy, scale, aotsxyRef)

	_out.write('%7.2f  %7.2f\n' % (d_xy[0], d_xy[1]))
    
    _out.close()



class Sky(object):
    def __init__(self, sciDir, skyDir, wave, scale=1,
                 skyfile='', angleOffset=0.0):
        # Setup some variables we will need later on
        self.sciDir = sciDir
        self.skyDir = skyDir
        self.wave = wave
        self.skyFile = skyfile
        self.scale = scale
        self.angleOffset = angleOffset
        
        self.defaultSky = skyDir + 'sky_' + wave + '.fits'

        if (wave == 'lp' or wave == 'ms'):
            self.__initLp__()

        # This will be the final returned skyname
        self.skyName = skyDir + 'sky_scaled.fits'

    def __initLp__(self):
	print 'Initializing Lp Sky skyfile=%s' % (self.skyFile) 

        # Read skies from manual sky file (format: raw_science   sky)
        if (self.skyFile):
            skyTab = asciidata.open(self.skyDir + self.skyFile)
            self.images = skyTab[0].tonumpy()
            skies = skyTab[1].tonumpy()

            skyAng = np.zeros([len(skies)], Float64)
            for i in range(0,len(skies)):
                sky = skies[i].strip()
                hdr = pyfits.getheader(self.skyDir + sky,ignore_missing_end=True)
                skyAng[i] = float(hdr['ROTPPOSN'])

        else:
            # Read in the sky table. Determine the effective K-mirror
            # angle for each sky.
            skyTab = asciidata.open(self.skyDir + 'rotpposn.txt')
            skies = skyTab[0].tonumpy()
            skyAng = skyTab[1].tonumpy()

        # The optimal sky angle to use is skyAng = A + B*sciAng
        self.angFitA = self.angleOffset
        self.angFitB = 1.0

        # Open a log file that we will keep
        _skylog = self.sciDir + 'sci_sky_subtract.log'
        util.rmall([_skylog])
        f_skylog = open(_skylog, 'w')

        # Stuff we are keeping
        self.skyTab = skyTab
        self.skies = skies
        self.skyAng = skyAng
        self.f_skylog = f_skylog

    def getSky(self, _n):
        if (self.wave == 'lp' or self.wave == 'ms'):
            sky = self.getSkyLp(_n)
        else:
            sky = self.defaultSky

        # Edit the science image to contain the
        # original sky name that will be subtracted.
        skyOrigName = sky[sky.rfind('/')+1:]
        ir.hedit(_n, 'SKYSUB', skyOrigName, add='yes', show='no', verify='no')

        # Now scale the sky to the science image
        skyScale = self.scaleSky(_n, sky)

        return skyScale

    def scaleSky(self, _n, _sky):
        """Scale the mean level of the sky so that it matches the
        science image.
        
        @param _n: name of science frame
        @type _n: string
        @param _sky: name of sky frame
        @type _sky: string
        """
        util.rmall([self.skyName])
        
        # scale sky to science frame
        if self.scale:
            text = ir.imstat(_n, fields='mean,stddev', nclip=20, 
                             lsigma=10, usigma=1.0, format=0, Stdout=1)
            vals = text[0].split()
            sci_mean = float(vals[0])
            text = ir.imstat(_sky, fields='mean,stddev', nclip=5, 
                             lsigma=5, usigma=5, format=0, Stdout=1)
            vals = text[0].split()
            sky_mean = float(vals[0])
            fact = sci_mean/sky_mean
            #print 'scaleSky: factor = %5f  sci_mean = %5f  sky_mean = %5f' % \
            #      (fact, sci_mean, sky_mean)
            ir.imarith(_sky, '*', fact, self.skyName)
        else:
            ir.imcopy(_sky, self.skyName)

        return self.skyName


    def getSkyLp(self, _n):
        """Determine which sky we should use for L'. Does all the
        rotator mirror angle matching.

        @param _n: Name of science frame.
        @type _n: string
        @returns sky: name of sky file to use.
        @rtype sky: string
        """
        # Sky subtract
        # determine the best angle for sky or use manual file

        # -- Determine the rotpposn for this image
        sciAng_tmp = ir.hselect(_n, "ROTPPOSN", "yes", Stdout=1)
        sciAng = float(sciAng_tmp[0])

        # -- Determine the best sky rotpposn.
        skyBest = self.angFitA + (self.angFitB * sciAng)

        # -- Repair all angles to be between -180 and 180.
        if (skyBest > 180): skyBest -= 360.0
        if (skyBest < -180): skyBest += 360.0
        if (sciAng > 180): sciAng -= 360.0
        if (sciAng < -180): sciAng += 360.0

        if (self.skyFile):
            for i in range(0,len(self.images)):
                if (self.images[i] == _n):
                    skyidx = i
        else:
            # -- Determine which sky file to use
            diff = [abs(skyAngle - skyBest) for skyAngle in self.skyAng]
            skyidx = np.argmin(diff)

        sky = self.skyDir + self.skies[skyidx]

        print('Science = ', _n)
        print('Sky image = ', sky)

        foo = '%s - %s  %6.1f  %6.1f' % \
              (_n, self.skies[skyidx], sciAng, self.skyAng[skyidx])
            
        self.f_skylog.write( foo )

        return sky

    def getNonlinearCorrection(self, sky):
        """Determine the non-linearity level. Raw data level of
        non-linearity is 12,000 but we subtracted
        off a sky which changed this level. The sky is 
        scaled, so the level will be slightly different
        for every frame.

        @param sky: File name of the sky used.
        @type sky: string
        @returns (sky_mean + sky_stddev) which is the value that should
            be subtracted off of the saturation count level.
        @rtype float
        """
        text_output = ir.imstatistics(sky, fields='mean,stddev', 
                                      usigma=4, lsigma=4, nclip=4, 
                                      format=0, Stdout=1)
        values = text_output[0].split()
        sky_mean = float(values[0])
        sky_stddev = float(values[1])
        coadds_tmp = ir.hselect(sky, "COADDS", "yes", Stdout=1)
        coadds = float(coadds_tmp[0])

        # -- Log what we did
        if (self.wave == 'lp' or self.wave == 'ms'):
            foo = ' %7d %7d\n' % (sky_mean, sky_stddev)

            self.f_skylog.write( foo )
	    
        return sky_mean + sky_stddev

    def close(self):
        """Close log files opened at init."""
        if (self.wave == 'lp' or self.wave == 'ms'):
            self.f_skylog.close()


def mosaic(files, wave, outroot, field=None, outSuffix=None,
            trim=0, weight=0, fwhm_max=0, submaps=0, fixDAR=True, maskSubmap=False):
    """Accepts a list of cleaned images and does a weighted combining after
    performing frame selection based on the Strehl and FWHM.

    Each image must have an associated *.coo file which gives the rough
    position of the reference source.

    @param files: List of integer file numbers to include in combine.
    @type files: list of int
    @param wave: Filter of observations (e.g. 'kp', 'lp', 'h')
    @type wave: string
    @param outroot: The output root name (e.g. '06jullgs'). The final combined
        file names will be <outroot>_<field>_<wave>. The <field> keyword
        is optional.

        Examples:
        06jullgs_kp for outroot='06jullgs' and wave='kp'
        06jullgs_arch_f1_kp for adding field='arch_f1'
    @type outroot: string
    @kwparam field: Optional field name used to get to clean directory and
        also effects the final output file name.
    @type field: string
    @kwparam trim: Optional file trimming based on image quality. Default
        is 0. Set to 1 to turn trimming on.
    @kwparam outSuffix: Optional suffix used to modify final output file name. 
    @type outSuffix: string
    @type trim: 0 or 1
    @kwparam weight: Optional weighting based on Strehl. Set to 1 to
        to turn file weighting on (default is 0).
    @type weight: 0 or 1
    @kwparam fwhm_max: The maximum allowed FWHM for keeping frames when
        trimming is turned on.
    @type fwhm_max: int
    @kwparam submaps: Set to the number of submaps to be made (def=0).
    @type submaps: int
    @kwparam mask: Set to false for maser mosaics; 06maylgs1 is an exception
    @type mask: Boolean
    """
    # Start out in something like '06maylgs1/reduce/kp/'
    # Setup some files and directories
    waveDir = util.getcwd()
    redDir = util.trimdir( os.path.abspath(waveDir + '../') + '/')
    rootDir = util.trimdir( os.path.abspath(redDir + '../') + '/')
    if (field != None):
        cleanDir = util.trimdir( os.path.abspath(rootDir +
                                                   'clean/' +field+
                                                   '_' +wave) + '/')
        outroot += '_' + field
    else:
        cleanDir = util.trimdir( os.path.abspath(rootDir +
                                                   'clean/' + wave) + '/')

    if (outSuffix != None):
        outroot += outSuffix

    # This is the final output directory
    comboDir = rootDir + 'combo/'
    util.mkdir(comboDir)

    # Make strings out of all the filename roots.
    roots = [str(f).zfill(4) for f in files]

    # This is the output root filename
    _out = comboDir + 'mag' + outroot + '_' + wave
    _sub = comboDir + 'm' + outroot + '_' + wave

    ##########
    # Determine if we are going to trim and/or weight the files 
    # when combining. If so, then we need to determine the Strehl
    # and FWHM for each image. We check strehl source which shouldn't
    # be saturated. *** Hard coded to strehl source ***
    ##########
 
    # Load Strehls and FWHM for sorting and trimming
    strehls, fwhm = loadStrehl(cleanDir, roots)

    # Default weights
    # Create an array with length equal to number of frames used,
    # and with all elements equal to 1/(# of files)
    weights = np.array( [1.0/len(roots)] * len(roots) )

    ##########
    # Trimming
    ##########
    if trim:
        roots, strehls, fwhm, weights = trim_on_fwhm(roots, strehls, fwhm,
                                                     fwhm_max=fwhm_max)

    ##########
    # Weighting
    ##########
    if weight == 'strehl':
        weights = weight_by_strehl(roots, strehls)

    if ((weight != None) and (weight != 'strehl')):
        # Assume weight is set to a filename
        if not os.path.exists(weight):
            raise ValueError('Weights file does not exist, %s' % weight)

        weights = readWeightsFile(roots, weight)

    # Determine the reference image
    refImage = cleanDir + 'c' + roots[0] + '.fits'
    print 'combine: reference image - %s' % refImage

    ##########
    # Write out a log file. With a list of images in the
    # final combination.
    ##########
    combine_log(_out, roots, strehls, fwhm, weights)

    # See if all images are at same PA, if not, rotate all to PA = 0
    # temporarily. This needs to be done to get correct shifts.
    print 'Calling combine_rotation'
    diffPA = combine_rotation(cleanDir, roots)

    # Make a table of initial guesses for the shifts.
    # Use the header keywords AOTSX and AOTSY to get shifts.
    print 'Calling mosaic_ref'
    mosaic_ref(_out + '.init.shifts', cleanDir, roots, diffPA)

    # Keep record of files that went into this combine
    print 'Calling combine_lis'
    combine_lis(_out + '.lis', cleanDir, roots, diffPA)

    # Register images to get shifts.
    print 'Calling mosaic_register'
    shiftsTab = mosaic_register(_out, refImage, diffPA)

    # Determine the size of the output image from max shifts
    print 'Calling mosaic_size'
    xysize = mosaic_size(shiftsTab, refImage, _out, _sub, submaps)

    # Combine all the images together.
    print 'Calling mosaic_drizzle'
    combine_drizzle(xysize, cleanDir, roots, _out, weights, shiftsTab,
                    wave, diffPA, fixDAR=fixDAR)
    
    # Now make submaps
    if (submaps > 0):
    	combine_submaps(xysize, cleanDir, roots, _sub, weights, 
    			shiftsTab, submaps, wave, diffPA, 
                        fixDAR=fixDAR, mask=maskSubmap)

    # Remove *.lis_r file & rotated rcoo files, if any - these
    # were just needed to get the proper shifts for xregister
    _lisr = _out + '.lis_r'
    util.rmall([_lisr])
    for i in range(len(roots)):
        _rcoo = cleanDir + 'c' + str(roots[i]) + '.rcoo'
        util.rmall([_rcoo])

def mosiac_ref(coofile, cleanDir, roots, rootDir):
    """
    Determine the relative offsets of files in a mosaic using
    the AOTSX and AOTSY header keywords. Output is written to a
    coofile.

    @param coofile: The name of the file to write the offsets to.
    @type coofile: string
    @param cleanDir: The full path to the clean/ directory.
    @type cleanDir: string
    @param roots: List of file numbers for all files in this mosaic.
    @type roots: list of strings
    @param rootDir: The full path to the root directory.
    @type rootDir: string
    """

    # Read the AOTSX coords from the first file
    firstFile = rootDir + 'raw/n' + roots[0] + '.fits'
    hdr1 = pyfits.getheader(firstFile,ignore_missing_end=True)
    scaleRef = nirc2.getScale(hdr1)
    aotsxyRef = nirc2.getAotsxy(hdr1)

    d_xyRef = nirc2.aotsxy2pix(aotsxyRef, scaleRef, aotsxyRef)

    # Write combined *.coo file containing the offsets.
    _allCoo = file(coofile, 'w')
    _allCoo.write('%7.2f  %7.2f\n' % (d_xyRef[0], d_xyRef[1]))

    for root in roots:
        # Compute the relative offsets.
	_fits = cleanDir + 'c' + root + '.fits'
        hdr = pyfits.getheader(_fits,ignore_missing_end=True)
        aotsxy = nirc2.getAotsxy(hdr)

        phi = nirc2.getPA(hdr)
        scale = nirc2.getScale(hdr)

        # Calculate the pixel offsets from the reference image
        d_xy = nirc2.aotsxy2pix(aotsxy, scale, aotsxyRef)

        _allCoo.write('%7.2f  %7.2f\n' % (d_xy[0], d_xy[1]))

    _allCoo.close()

def mosaic_register(outroot, refImage, diffPA):
    """
    Register images for a mosaic. This only calculates the exact
    shifts between each image... it doesn't do the combining.

    @param outroot: The root for the output image. The resulting
    shifts will be written into a file called <outroot>.shifts
    @type outroot: string
    @param refImage: The name of the reference image.
    @type refImage: string
    """
    shiftFile = outroot + '.shifts'
    util.rmall([shiftFile])

    # xregister parameters
    ir.immatch
    ir.unlearn('xregister')
    ir.xregister.coords = outroot + '.init.shifts'
    ir.xregister.output = ''
    ir.xregister.append = 'no'
    ir.xregister.databasefmt = 'no'
    ir.xregister.verbose = 'yes'
    ir.xregister.correlation = 'fourier'
    ir.xregister.xwindow = '10'
    ir.xregister.ywindow = '10'

    print 'combine: registering images'
    if (diffPA == 1):
        input = '@' + outroot + '.lis_r'
    else:
        input = '@' + outroot + '.lis'

    regions = '[*,*]'
    ir.xregister(input, refImage, regions, shiftFile)

    # Read in the shifts file. Column format is:
    # Filename.fits  xshift  yshift
    shiftsTable = asciidata.open(shiftFile)

    return (shiftsTable)


def mosaic_size(shiftsTable, refImage, outroot, subroot, submaps):
    """
    Determine the final size for the completed mosaic.

    @params shiftsTable: Table from mosaic_register containing the
    shifts for all the images.
    @type shiftsTable: string
    @param refImage: The first image used as  reference.
    @type refImage: string
    @param outroot: The root name for the resulting output file.
    @type outroot: string
    @param subroot:
    @type subroot: string
    @param submaps:
    @type submaps:
    """
    x_allShifts = shiftsTable[1].tonumpy()
    y_allShifts = shiftsTable[2].tonumpy()

    xhi = abs(x_allShifts.max())
    xlo = abs(x_allShifts.min())
    yhi = abs(y_allShifts.max())
    ylo = abs(y_allShifts.min())

    # Make sure to include the edges of all images.
    # Might require some extra padding on one side.
    maxoffset = max([xlo, xhi, ylo, yhi])

    orig_img = pyfits.getdata(refImage)
    orig_size = (orig_img.shape)[0]
    padd = 8.0

    xref = x_allShifts[0]
    yref = y_allShifts[0]

    xref = xref + (maxoffset + padd)
    yref = yref + (maxoffset + padd)

    # Read in 16C's position in the ref image and translate
    # it into the coordinates of the final main and sub maps.
    hdr = pyfits.getheader(refImage,ignore_missing_end=True)
    xrefSrc = float(hdr['XREF'])
    yrefSrc = float(hdr['YREF'])

    xrefSrc = xrefSrc + (maxoffset + padd)
    yrefSrc = yrefSrc + (maxoffset + padd)

    cooMain = [outroot + '.coo']
    cooSubs = ['%s_%d.coo' % (subroot, i) for i in range(submaps+1)]
    cooAll = cooMain + cooSubs

    util.rmall(cooAll)
    for coo in cooAll:
        _allCoo = open(coo, 'w')
        _allCoo.write('%9.3f %9.3f\n' % (xrefSrc, yrefSrc))
        _allCoo.close()

    xysize = float(orig_size) + ((maxoffset + padd) * 2.0)
    print 'combine: Size of output image is %d' % xysize

    return xysize

