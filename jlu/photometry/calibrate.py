#!/usr/bin/env python
import optparse
import textwrap
import numpy as np
import pylab as py
import asciidata
import math
import sys
import pyfits
import pdb

# Map of the possible plate scales
all_scales = [[1.0, 'No scaling'],
              [0.0102, 'Speckle'],
              [0.0087, 'KCAM-AO'],
              [0.0085, 'SCAM-AO'],
              [0.00993, 'NIRC2-AO narrow'],
              [0.02000, 'NIRC2-AO medium'],
              [0.04000, 'NIRC2-AO wide'],
              [0.0170, 'SCAM-AO unmagnified'],
              [0.030, 'NIRC2-AO narrow, binned by 3'],
              [0.01998, 'GEMINI'],
              [0.107, 'MMT PISCES'],
              [0.200, 'UKIDSS'],
              [0.004, 'TMT/IRIS'],
              [0.0196, 'GSAOI'],
              [0.050, 'ACS-WFC']]

##################################################
# 
# Help formatter for command line arguments. 
# This is very generic... skip over for main code.
#
##################################################
class IndentedHelpFormatterWithNL(optparse.IndentedHelpFormatter):
    def format_description(self, description):
        if not description: return ""
        desc_width = self.width - self.current_indent
        indent = " "*self.current_indent
        # the above is still the same
        bits = description.split('\n')
        formatted_bits = [
            textwrap.fill(bit,
                          desc_width,
                          initial_indent=indent,
                          subsequent_indent=indent)
            for bit in bits]
        result = "\n".join(formatted_bits) + "\n"
        return result

    def format_option(self, option):
        # The help for each option consists of two parts:
        #   * the opt strings and metavars
        #   eg. ("-x", or "-fFILENAME, --file=FILENAME")
        #   * the user-supplied help string
        #   eg. ("turn on expert mode", "read data from FILENAME")
        #
        # If possible, we write both of these on the same line:
        #   -x    turn on expert mode
        #
        # But if the opt string list is too long, we put the help
        # string on a second line, indented to the same column it would
        # start in if it fit on the first line.
        #   -fFILENAME, --file=FILENAME
        #       read data from FILENAME
        result = []
        opts = self.option_strings[option]
        opt_width = self.help_position - self.current_indent - 2
        if len(opts) > opt_width:
            opts = "%*s%s\n" % (self.current_indent, "", opts)
            indent_first = self.help_position
        else: # start help on same line as opts
            opts = "%*s%-*s  " % (self.current_indent, "", opt_width, opts)
            indent_first = 0
        result.append(opts)
        if option.help:
            help_text = self.expand_default(option)
            # Everything is the same up through here
            help_lines = []
            for para in help_text.split("\n"):
                help_lines.extend(textwrap.wrap(para, self.help_width))
            # Everything is the same after here
            result.append("%*s%s\n" % (
                    indent_first, "", help_lines[0]))
            result.extend(["%*s%s\n" % (self.help_position, "", line)
                           for line in help_lines[1:]])
        elif opts[-1] != "\n":
            result.append("\n")
        return "".join(result)



##################################################
#
# Main body of calibrate code.
#
##################################################
def main(argv=None):
    if argv is None:
        argv = sys.argv[1:]

    # Read options and check for errors.
    options = read_command_line(argv)
    if (options == None):
        return

    # Read in the photometric calibrators
    calibs = read_photo_calib_file(options)
    
    # Read in the starlist
    stars = input_data(options)

    # Match up calibrator stars with stars in our starlist.
    calibs.index = find_cal_stars(calibs, stars, options)

    # Calculate the average zeropoint (and errors)
    zeropt, zeropt_err = calc_zeropt(calibs, stars, options)

    # Write out the output files
    output_new(zeropt, zeropt_err, calibs, stars, options)
  
    
def read_command_line(argv):
    p = optparse.OptionParser(usage='usage: %prog [options] [starlist]',
                              formatter=IndentedHelpFormatterWithNL())

    p.add_option('-f', dest='data_type', type=int, default=1, metavar='[#]',
                 help='Data type of the input starlist:\n'+
                 '[1] starfinder format without errors (default)\n'+
                 '[2] align_rms format with errors\n')

    scaleHelp = 'Camera type to set plate scale (default: %default):\n'
    for ss in range(len(all_scales)):
        scaleHelp += '[%d] %s (%7.5f asec/pix) \n' % \
            (ss, all_scales[ss][1], all_scales[ss][0])
        
        
    p.add_option('-c', dest='camera_type', type=int, default=1, metavar='[#]',
                 help=scaleHelp)
    p.add_option('-T', dest='theta', type=float, default=0.0, metavar='[ANGLE]',
                 help='Rotation of the image from North with positive angles '+
                 'in the clockwise direction (default: %default)')
    p.add_option('-I', dest='first_star', default='16C', metavar='[STAR]',
                 help='Name of the first star in the list. Any star from the '+
                 'input calibration list (see -N) can be used. The coo star'+
                 'should have a known magnitude (within 0.5 mag) in the '+
                 'calibration list in order to properly find the rest of the '
                 'calibrator stars. The coo star does not have to be used '+ 
                 'as a calibrator. (default: %default)')
    p.add_option('-N', dest='calib_file', metavar='[FILE]',
                 default='/u/ghezgroup/data/gc/source_list/photo_calib.dat',
                 help='A file containing the calibration stars and magnitudes '+
                 '(default: %default). The file format should have different '+
                 'columns for each photometric calibration reference. The '+
                 'default calibration sources for each column are stored in '+
                 'a comment at the head of the file. Choose which column to '+
                 'use with the -M flag.')
    p.add_option('-M', dest='calib_column', type=int, default=6, metavar='[#]',
                 help='Choose which column to use in the photo_calib.dat '+
                 'file (default: %default). See below for column choices.')
    p.add_option('-S', dest='calib_stars', metavar='[STAR1,STAR2]', default=None,
                 help='Specify the stars to be used as calibrators in a '+
                 'comma-separated list. The defaults are set in the '+
                 'photo_calib.dat file at the top (or see below).')
    p.add_option('-r', dest='outroot', metavar='[ROOT]',
                 help='Rename root name for output (default: [listname]_cal)')
    p.add_option('-V', dest='verbose', action='store_true', default=False,
                 help='Print extra diagnostics.')
    p.add_option('-R', dest='reorder', action='store_true', default=False,
                 help='Reorder the starlist so that the calibration sources'+
                 'are at the top. The coo star (first star) will remain first.')
    p.add_option('-s', '--snr', dest='snr_flag', default=0, metavar='[#]',
                 help='Use this flag to indicate the purpose of the SNR '+
                 'column in the input star list (default: %default). '+
                 'The choices are:\n'+
                 '[0] overwrite SNR column with calibration error (default)\n'+
                 '[1] add SNR in quadrature to calibration error\n'+
                 '[2] leave the SNR column alone')
    options, args = p.parse_args(argv)
    
    # Keep a copy of the original calling parameters
    options.originalCall = ' '.join(argv)

    # Read the input filename
    options.input_file = None
    if len(args) == 1:
        options.input_file = args[0]
    else:
        print ''
        p.print_help()
        read_photo_calib_file(options, verbose=True)
        return None

    # Set the output filenames
    if (options.outroot == '' or options.outroot == None):
        parts = options.input_file.split('.')
    else:
        parts = options.outroot.split('.')
    if (len(parts) > 1):
        root = '.'.join(parts[0:-1])
        extension = parts[-1]
    else:
        root = parts[0]
        extension = 'lis'
    options.outname = root + '_cal.' + extension
    options.zername = root + '_cal.zer'

    # Set plate scale
    options.plate_scale = all_scales[options.camera_type][0]

    # Parse calib stars 
    if (options.calib_stars != None):
        options.calib_stars = options.calib_stars.split(',')

    # Verbose mode printing
    if options.verbose:
        print 'VERBOSE mode on'
        print 'options.first_star = %s' % options.first_star
        print 'options.data_type = %d' % options.data_type
        print 'options.camera_type = %d' % options.camera_type
        print 'options.plate_scale = %7.2f' % options.plate_scale
        print 'options.outname = %s' % options.outname
        print 'options.calib_file = %s' % options.calib_file
        print 'options.calib_column = %d' % options.calib_column
        print 'options.theta = %6.1f' % options.theta
        if options.reorder:
            print 'Reordering lis file'
        else:
            print 'Not reordering lis file.'

    return options


def read_photo_calib_file(options, verbose=False):
    f_calib = open(options.calib_file, 'r')
    
    magInfo = []
    defaultStars = []

    if verbose:
        print ''
        print 'Photometric calibration information loaded from:'
        print '\t', options.calib_file
        print 'Specify a different file with the -N flag.'
        print 'Choose a calibration column with the -M flag.'
        print 'The column choices are listed by [#] below.'
        print ''
        print 'Bandpass and References:'

    # Loop through lines and parse them. Recall that 
    # comments are "##" and column headers are "# ".
    for line in f_calib:
        if line.startswith('##'):
            # Comments, skip
            continue

        if line.startswith('# '):
            # Column headers. The first four are hardcoded.
            # The rest tell us how many magnitude columns
            # we have and the associated references.
            
            fields = line.split('--')

            colnum = int( fields[0].replace('# ', '') )
            # Skip the first 4 columns
            if ((len(fields) >= 2) and (colnum > 4)):
                magInfo.append(fields[1])

                if len(fields) > 2:
                    defaultStars.append(fields[2])
                else:
                    defaultStars.append(None)

                if verbose:
                    print '[%d]\t %s' % (colnum-4, fields[1])

        else:
            # Found the first line of data after the
            # header. Finished reading the header.
            break

    f_calib.close()

    if verbose:
        print ''
        print 'Calibration Sources:'
        print '\t(* default if no -S flag)'
        print '\t(! stars determined to be variable)'
        print ''
    
    ##########
    #
    # Read in the data portion of the file.
    #
    ##########
    tab = asciidata.open(options.calib_file)

    name = tab[0].tonumpy()
    x = tab[1].tonumpy()
    y = tab[2].tonumpy()
    isVariable = (tab[3].tonumpy() == 1)

    magMatrix = np.zeros((len(magInfo), tab.nrows), dtype=float)
    isDefaultMatrix = np.zeros((len(magInfo), tab.nrows), dtype=bool)

    for i in range(len(magInfo)):
        magMatrix[i,:] = tab[i+4].tonumpy()

        # If no default stars were set, then assume
        # all stars with non-zero magnitudes are the defaults.
        if defaultStars[i] == None:
            idx = np.where(magMatrix[i,:] != 0)[0]
            isDefaultMatrix[i,idx] = True

        else:
            stars = defaultStars[i].split(',')
            stars = [stars[s].strip() for s in range(len(stars))]
            for s in range(tab.nrows):
                if (name[s] in stars):
                    isDefaultMatrix[i,s] = True
                else:
                    isDefaultMatrix[i,s] = False

    ##########
    # Print out 
    ##########
    if verbose:
        print ' %10s ' % 'Name',
        for i in range(len(magInfo)):
            print ' [%3d]  ' % (i+1),

        print '\n'

        for s in range(tab.nrows):
            varChar = '!' if isVariable[s] else ''
            print '%1s%13s ' % (varChar, name[s]),
            
            for i in range(len(magInfo)):
                defChar = '*' if isDefaultMatrix[i,s] else ''
                print ' %5.2f%1s ' % (magMatrix[i,s], defChar),

            print ''


    ##########
    # Catch case where this is called
    # just to print out the file (no starlist to calibrate)
    ##########
    if options.input_file == None:
        return None

    ##########
    # Get only the calibration magnitudes
    # that were asked for.
    ##########
    calibs = Starlist()
    calibs.name = name
    calibs.x = x
    calibs.y = y
    # Pick out the magnitude column
    calibs.mag = magMatrix[options.calib_column - 1,:]  
    calibs.magInfo = magInfo[options.calib_column-1] # String with source info.

    ##########
    #
    # Decide which stars will be included as
    # calibrators based on user input. This 
    # sets up calibs.include which is set to
    # True for those calibrators that should be
    # used in the photometric calibration.
    #
    ##########
    if (options.calib_stars == None):
        # Use defaults
        calibs.include = isDefaultMatrix[options.calib_column - 1,:]
    else:
        calibs.include = np.zeros(len(name), dtype=bool)

        for tt in range(len(options.calib_stars)):
            idx = np.where(name == options.calib_stars[tt])[0]

            if len(idx) == 0:
                msg = 'Failed to find user specified calibrator: %s' % \
                    options.calib_stars[tt]
                raise Exception(msg)
                
            calibs.include[idx] = True

            if options.verbose:
                print 'Found calibrator: ', name[idx], ' ', options.calib_stars[tt]

    return calibs
    
def input_data(options):
    """
    Read in a starlist and return a starlist object
    will the following hanging off:
      name
      mag
      epoch
      x
      y
      xerr (none for starfinder starlists)
      yerr (none for starfinder starlists)
      snr
      corr
      nframes
      fwhm
    """
    if options.verbose:
        print 'Opening starlist: ', options.input_file

    tab = asciidata.open(options.input_file)

    name = tab[0].tonumpy()
    name = np.array(name, dtype='S13')  # pre-allocate 13 characters.
    mag = tab[1].tonumpy()
    epoch = tab[2].tonumpy()
    x = tab[3].tonumpy()
    y = tab[4].tonumpy()
    
    
    if options.data_type == 2:
        xerr = tab[5].tonumpy()
        yerr = tab[6].tonumpy()
        snr = tab[7].tonumpy()
        corr = tab[8].tonumpy()
        nframes = tab[9].tonumpy()
        fwhm = tab[10].tonumpy()
    else:
        xerr = None
        yerr = None
        snr = tab[5].tonumpy()
        corr = tab[6].tonumpy()
        nframes = tab[7].tonumpy()
        fwhm = tab[8].tonumpy()

        
    # Trim out stars with errors in magnitudes
    idx = np.where(mag != float('Inf'))[0]

    if len(idx) > 0:
        name = name[idx]
        mag = mag[idx]
        epoch = epoch[idx]
        x = x[idx]
        y = y[idx]
        snr = snr[idx]
        corr = corr[idx]
        nframes = nframes[idx]
        fwhm = fwhm[idx]
        
        if (xerr != None):
            xerr = xerr[idx]
            yerr = yerr[idx]

    if options.verbose:
        print 'Read %d lines in the input file.' % (len(x) + len(idx))
        print 'Skipped %d lines in the input file.' % (len(idx))

    starlist = Starlist()
    starlist.name = name
    starlist.mag = mag
    starlist.epoch = epoch
    starlist.x = x
    starlist.y = y
    starlist.xerr = xerr
    starlist.yerr = yerr
    starlist.snr = snr
    starlist.corr = corr
    starlist.nframes = nframes
    starlist.fwhm = fwhm

    return starlist
    
def find_cal_stars(calibs, stars, options):
    """
    Returns an array of indices which holds the index of the 
    matching star in the starlist. Non-matches have an index of -1. 
    """
    # First we need to find out if the first star
    # in the star list is in our list of calibrators.
    fidx = np.where(calibs.name == options.first_star)[0]
    if (len(fidx) == 0):
        msg =  'Failed to find the first star in the calibrators:\n'
        msg += '  %s' % options.first_star
        raise Exception(msg)

    # Change the positional offsets to be relative to 
    # the reference source.
    calibs.x -= calibs.x[fidx]
    calibs.y -= calibs.y[fidx]

    # Determine the pixel positions for the calibrators
    cosScale = math.cos(math.radians(options.theta)) / options.plate_scale
    sinScale = math.sin(math.radians(options.theta)) / options.plate_scale
    calibs.xpix = stars.x[0] - (calibs.x * cosScale) + (calibs.y * sinScale)
    calibs.ypix = stars.y[0] + (calibs.x * sinScale) + (calibs.y * cosScale)
    if options.verbose:
        for c in range(len(calibs.xpix)):
            print 'Looking for %10s at (%.2f, %.2f)' % \
                (calibs.name[c], calibs.xpix[c], calibs.ypix[c])
            
    # Create an array of indices into the starlist.
    # Set to -1 for non-matches. 
    index = np.ones(len(calibs.name), dtype=int) * -1 

    # Loop through all the calibrators and find their match in the starlist
    # search radius = 0.25 arcsec for bright sources
    searchRadius = 0.25 / options.plate_scale   
    searchMag = 1.5
    
    magAdjust = stars.mag[0] - calibs.mag[fidx]
    if options.verbose:
        print 'Search dr = %d pixels, dm = %.2f' % (searchRadius, searchMag)
        print 'Adjusting input magnitudes by %.2f' % magAdjust

    for c in range(len(calibs.name)):
        dx = stars.x - calibs.xpix[c]
        dy = stars.y - calibs.ypix[c]
        dr = np.hypot(dx, dy)
        dm = abs(stars.mag - calibs.mag[c] - magAdjust)

        # Find the matches within our tolerance.
        if (calibs.mag[c] < 12):
            # For the bright stars we have the default search radius:
            idx = np.where((dr < searchRadius) & (dm < searchMag))[0]
        else:
            # For the fainter stars, use a smaller search radius:
            idx = np.where((dr < searchRadius/2) & (dm < searchMag))[0]
        
        # Default is not found
        index[c] = -1

        # But if we find one, change names, record index, etc.
        if (len(idx) > 0):
            # Record match index into the starlist
            index[c] = idx[0]
            
            # Rename
            origName = stars.name[index[c]]
            stars.name[index[c]] = calibs.name[c]

            # Print out
            if options.verbose:
                notUsed = '' if (calibs.include[c] == True) else '(not used)'
                print '%10s found at %.2f, %.2f as %s %s' % \
                    (calibs.name[c], 
                     stars.x[index[c]],
                     stars.y[index[c]],
                     origName,
                     notUsed)

        else:
            if options.verbose:
                print '%10s not found' % (calibs.name[c])


    return index

def calc_zeropt(calibs, stars, options):
    """
    Calculuate the average zeropoints from all the
    calibrator stars specified by the user (or defaults).
    Recall that not all calibrators in our list will
    be used for the calibration... onlyt those with
    both include = True and found in the starlist.
    """

    # Identify the calibrators we will use to calculate
    # the zeropoint.
    cidx = np.where((calibs.include == True) & (calibs.index != -1))[0]
    sidx = calibs.index[cidx]

    # Calculate the zero point as the difference between the calculated 
    # magnitude and the published magnitude for each calibration star.
    dm = calibs.mag[cidx] - stars.mag[sidx]
    all_zeropts = np.power(10, -dm/2.5)
    
    # Take mean as the value and standard deviation as the error.
    # Note, we are not using the error on the mean.
    zeropt = all_zeropts.mean()
    zeropt_err = all_zeropts.std(ddof=1)

    # Using the St.Dev propagate the errors into mag space
    zeropt_err *= 1.0857 / zeropt

    # Convert the flux ratio back to a magnitude difference
    zeropt = -2.5 * math.log10( zeropt )

    if options.verbose:
        print ''
        print 'Zero-point = %5.3f +/- %.3f' % (zeropt, zeropt_err)
        print ''
        for i in range(len(cidx)):
            c = cidx[i]
            s = sidx[i]
            print '%10s Published Mag: %6.3f  Calculate Mag: %6.3f  DIFF = %6.3f' % \
                (calibs.name[c], calibs.mag[c], stars.mag[s]+zeropt,
                 calibs.mag[c] - (stars.mag[s]+zeropt))

    return (zeropt, zeropt_err)


def output_new(zeropt, zeropt_err, calibs, stars, options):
    """
    Write out a calibrated starlist and a *.zer file with 
    the calculated zeropoints.
    """
    # Update the magnitudes of all the stars
    stars.mag += zeropt

    # Update the SNR for calibration error
    # 0 = use only zero point error
    # 1 = add in quadrature 1/SNR and zero point error
    # 2 = use original snr, ignore zero point error
    if (options.snr_flag == 1):
        orig_err = 1.0 / stars.snr # flux
        new_err = np.sqrt(orig_err**2 + (zeropt_err/1.0857)**2)
        stars.snr = 1.0 / new_err
    if (options.snr_flag == 0):
        stars.snr = np.zeros(len(stars.snr), dtype=float) + (1.0857/zeropt_err)
    # Fix infinite SNR
    idx = np.where(stars.snr == float('inf'))
    if (len(idx) > 0):
        stars.snr[idx] = 9999.0

    ##########
    # *.zer file
    ##########
    _zer = open(options.zername, 'w')
    
    # Get the number of calibrators used:
    cidx = np.where((calibs.include == True) & (calibs.index >= 0))[0]
    calibCnt = len(cidx)

    _zer.write('# Original Calling Parameters:\n')
    _zer.write('# %s\n' % options.originalCall)
    _zer.write('#\n')
    _zer.write('#ZeroPoint   Error   Ncal   Cal names - ')
    _zer.write(calibs.magInfo + '\n')
    _zer.write('%10.3f   ' % zeropt)
    _zer.write('%5.3f   ' % zeropt_err)
    _zer.write('%4d   ' % calibCnt)
    for c in cidx:
        _zer.write('%s  ' % calibs.name[c])
    _zer.write('\n')

    _zer.close()


    ##########
    # *_cal.lis file
    ##########
    # Get the output order for the new starlist
    # If re-ordering, the calibrators we used are
    # first, in order.
    idx = np.arange(len(stars.name))

    fdx = []  # This will hold the reordered stuff that goes to the top.
    if (options.reorder):
        # Put the coo star at the top
        cdx = np.where(calibs.name == options.first_star)[0]
        fdx.append(calibs.index[cdx[0]])

        if (options.calib_stars == None):
            # Used default calibraters
            cdx = np.where((calibs.include == True) & 
                           (calibs.index >= 0) &
                           (calibs.name != options.first_star))[0]
            fdx.extend(calibs.index[cdx])
        else:
            # Use in order specified by user.
            for c in range(len(options.calib_stars)):
                # This should always work here since any issues
                # with the calib stars should have been caught 
                # eralier.
                if options.calib_stars[c] == options.first_star:
                    continue

                ss = np.where(stars.name == options.calib_stars[c])[0]

                if len(ss) > 0:
                    fdx.append(ss[0])

        idx = np.delete(idx, fdx)
                
    # Now we have everything in order of fdx, idx
    _out = open(options.outname, 'w')
    
    for ff in fdx:
        _out.write('%13s  %6.3f  %8.3f  %9.3f %9.3f  ' % 
                   (stars.name[ff], 
                    stars.mag[ff],
                    stars.epoch[ff],
                    stars.x[ff], stars.y[ff]))

        if (options.data_type == 2):
            _out.write('%7.3f %7.3f  ' % 
                       (stars.xerr[ff], stars.yerr[ff]))

        _out.write('%10.2f  %9.2f  %8d  %13.3f\n' %
                   (stars.snr[ff], stars.corr[ff],
                    stars.nframes[ff], stars.fwhm[ff]))

    for ff in idx:
        _out.write('%13s  %6.3f  %8.3f  %9.3f %9.3f  ' % 
                   (stars.name[ff], 
                    stars.mag[ff],
                    stars.epoch[ff],
                    stars.x[ff], stars.y[ff]))

        if (options.data_type == 2):
            _out.write('%7.3f %7.3f  ' % 
                       (stars.xerr[ff], stars.yerr[ff]))

        _out.write('%10.2f  %9.2f  %8d  %13.3f\n' %
                   (stars.snr[ff], stars.corr[ff],
                    stars.nframes[ff], stars.fwhm[ff]))

    _out.close()
    
def get_camera_type(fitsfile):
    """
    Helper class to get the calibrate camera type from the
    FITS header.
    """
    # First check the instrument
    hdr = pyfits.getheader(fitsfile)
    instrument = hdr.get('CURRINST')
    if (instrument == None):
       # OLD SETUP
       instrument = hdr.get('INSTRUME')
       
    if (instrument == None):
       # OSIRIS
       instrument = hdr.get('INSTR')

       if ('imag' in instrument):
          instrument = 'OSIRIS'
    
    # Default is still NIRC2
    if (instrument == None): 
        instrument = 'NIRC2'

    # get rid of the whitespace
    instrument = instrument.strip()

    # Check NICMOS camera
    if instrument == 'NICMOS':
        camera = hdr.get('CAMERA')
        instrument += camera.strip()
        

    # Check NIRC2 camera
    if instrument == 'NIRC2':
        camera = hdr.get('CAMNAME')
        instrument += camera.strip()


    cameraInfo = {'NIRC-D79': 1,
                  'KCAM-AO': 2,
                  'SCAM-AO': 3,
                  'NIRC2narrow': 4, 
                  'NIRC2medium': 5, 
                  'NIRC2wide': 6,
                  'SCAM-AO unmagnified': 7,
                  'NIRC2-AO narrow, binned by 3': 8,
                  'Hokupaa+QUIRC': 9,
                  'MMT PISCES': 10,
                  'UKIDSS': 11,
                  'OSIRIS': 5,
                  'LGSAO': 4
                  }


    camera = cameraInfo.get(instrument)

    # Default value
    if camera == None:
        camera = 1

    return camera


class Starlist(object):
    """Use this to keep all the lists associated with
    our input starlist."""
    pass

if __name__ == '__main__':
    main()


