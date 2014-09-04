#!/usr/bin/env python
#
# Make an electronic log for NIRC2 data

import pyfits
import sys
import os
import glob
import asciidata
from numpy import *

def nirc2log(directory):
    """Make an electronic NIRC2 log for all files in the specified
    directory.

    Output is a file called nirc2.log."""
    if not os.access(directory, os.F_OK):
	print 'Cannot access directory ' + directory

    files = glob.glob(directory + '/*.fits')
    files.sort()
    f = open(directory + '/nirc2.log', 'w')
    
    for file in files:
	hdr = pyfits.getheader(file,ignore_missing_end=True)

	# First column is frame number
	frame = (hdr['filename'].strip())[0:5]
	f.write('%5s  ' % frame)
	
	# Second column is object name
	f.write('%-16s  ' % hdr['object'].replace(' ', ''))

	# Next is integration time, coadds, sampmode, multisam
	f.write('%8.3f  %3d  ' % (hdr['itime'], hdr['coadds']))
	f.write('%1d x %2d  ' % (hdr['sampmode'], hdr['multisam']))
	
	# Filter
	filter1 = hdr['fwiname']
	filter2 = hdr['fwoname']
        filter = filter1
	if (filter1.startswith('PK')): filter = filter2

	f.write('%-10s ' % filter)
	 
	# Camera name
	f.write('%-6s ' % hdr['camname'])
	
	# Shutter state
	f.write('%-6s ' % hdr['shrname'])
	
	# End of this line
	f.write('\n')

    f.close()
        

if __name__ == '__main__':
    _nargs = len(sys.argv)
    
    if _nargs != 2:
	print 'Usage: nirc2log directory'
    else:
	nirc2log(sys.argv[1])

def getAotsxy(hdr):
    # Note: 04jullgs does not have LSPROP or AOTSX keywords
    if (hdr.get('OUTDIR').strip() != '/sdata904/nirc2eng/2004jul26/'):
        if (hdr.get('LSPROP') == 'yes'):
            # LGS MODE
            aotsxy = [float(hdr['AOTSX']), float(hdr['AOTSY'])]

            # Another special case: 09may01 UT had a AOTSX/Y linear drift.
            # Found drift with 09maylgs/clean/kp/coord.py
            if ((hdr['OUTDIR']).strip() == '/sdata903/nirc3/2009may01/'):
                mjdobs = float(hdr['MJD-OBS'])
                # old distortion solution (pre-ship) gives:
                #aotsxy[0] -= (1.895701e5*0.727) + (-3.449709*0.727) * mjdobs 
                # new distortion solution (yelda et al. 2010)
                aotsxy[0] -= (1.903193e5*0.727) + (-3.463342*0.727) * mjdobs 

                print 'getAotsxy: modified from %8.3f %8.3f to %8.3f %8.3f' % \
                    (float(hdr['AOTSX']), float(hdr['AOTSY']), 
                     aotsxy[0], aotsxy[1])

            # Another special case: 10jul06 UT had a AOTSX/Y linear drift.
            # Found drift with 10jullgs1/clean/kp/coord.py
            if ((hdr['OUTDIR']).strip() == '/sdata903/nirc3/2010jul06/'):
                mjdobs = float(hdr['MJD-OBS'])
                # new distortion solution (yelda et al. 2010)
                aotsxy[0] -= (2.039106e5*0.727) + (-3.681807*0.727) * mjdobs 

                print 'getAotsxy: modified from %8.3f %8.3f to %8.3f %8.3f' % \
                    (float(hdr['AOTSX']), float(hdr['AOTSY']), 
                     aotsxy[0], aotsxy[1])
        else:
            # NGS MODE
            # Note: OBFMYIM refers to X, and vice versa!
            aotsxy = [float(hdr['OBFMYIM']), float(hdr['OBFMXIM'])]
    else:
        # 04jullgs
        # Assumes that 04jullgs has AOTSX/AOTSY keywords were added
        # by hand (see raw/fix_headers.py)
        # Note: OBFMYIM refers to X, and vice versa!
        #aotsxy = [float(hdr['OBFMYIM']), float(hdr['OBFMXIM'])]
        aotsxy = [float(hdr['AOTSX']), float(hdr['AOTSY'])]

    return aotsxy


def pix2radec():
    print 'Not done yet'

def radec2pix(radec, phi, scale, posRef):
    """Determine pixel shifts from true RA and Dec positions.

    @param radec: a 2-element list containing the RA and Dec in degrees.
    @type radec: float list
    @param phi: position angle (E of N) in degrees.
    @type phi: float
    @param scale: arcsec per pixel.
    @type scale: float
    @param posRef: 2-element list containing the ra, dec positions (in degrees)
            of a reference object.
    @type posRef: float list
    """
    # Expected in degrees
    ra = radec[0]
    dec = radec[1]
    
    # Difference in RA and Dec. Converted to arcsec.
    d_ra = math.radians(ra - posRef[0]) * 206265.0
    d_dec = math.radians(dec - posRef[1]) * 206265.0

    cos = math.cos(math.radians(phi))
    sin = math.sin(math.radians(phi))
    cosdec = math.cos(math.radians(dec))

    d_x = (d_ra * cosdec * cos) + (d_dec * sin)
    d_y = (d_ra * cosdec * sin) - (d_dec * cos)
    d_x = d_x * (1.0/scale)
    d_y = d_y * (1.0/scale)
    
    return [d_x, d_y]

def aotsxy2pix(aotsxy, scale, aotsxyRef):
    # Determine pixel shifts from AOTSX and AOTSY positions.
    
    x = aotsxy[0]
    y = aotsxy[1]

    # AOTSX,Y are in units of mm. Conversion is 0.727 mm/arcsec
    d_x = (x - aotsxyRef[0]) / 0.727
    d_y = (y - aotsxyRef[1]) / 0.727
    d_x = d_x * (1.0/scale)
    d_y = d_y * (1.0/scale)
    
    return [d_x, d_y]

def pix2xyarcsec(xypix, phi, scale, sgra):
    """Determine  E and N offsets from Sgr A* (in arcsec) from 
    pixel positions and the pixel position of Sgr A*.

    xypix: 2-element list containing the RA and Dec in degrees.
    phi: position angle (E of N) in degrees.
    scale: arcsec per pixel.
    sgra: 2-element list containing the pixel position of Sgr A*.
    """
    # Expected in arcseconds
    xpix = xypix[0] - sgra[0]
    ypix = xypix[1] - sgra[1]

    cos = math.cos(math.radians(phi))
    sin = math.sin(math.radians(phi))
    
    d_x = (xpix * cos) + (xpix * sin)
    d_y = -(xpix * sin) + (ypix * cos)
    d_x = d_x * -scale
    d_y = d_y * scale
    
    return [d_x, d_y]

def xyarcsec2pix(xyarcsec, phi, scale):
    """Determine pixel shifts from E and N offsets from Sgr A*.

    xyarcsec: 2-element list containing the RA and Dec in degrees.
    phi: position angle (E of N) in degrees.
    scale: arcsec per pixel.
    """
    # Expected in arcseconds
    xarc = xyarcsec[0]
    yarc = xyarcsec[1]

    cos = math.cos(math.radians(phi))
    sin = math.sin(math.radians(phi))

    d_x = (-xarc * cos) + (yarc * sin)
    d_y = (xarc * sin) + (yarc * cos)
    d_x = d_x * (1.0/scale)
    d_y = d_y * (1.0/scale)
    
    return [d_x, d_y]

def rotate_coo(x, y, phi):
    """Rotate the coordinates in the *.coo files for data sets
    containing images at different PAs.
    """
    # Rotate around center of image, and keep origin at center
    xin = 512.
    yin = 512.
    xout = 512.
    yout = 512.
  
    cos = math.cos(math.radians(phi))
    sin = math.sin(math.radians(phi))

    xrot = (x - xin) * cos - (y - yin) * sin + xout
    yrot = (x - xin) * sin + (y - yin) * cos + yout
   
    return [xrot, yrot]
    

def getScale(hdr):
    # Setup NIRC2 plate scales
    scales = {"narrow": 0.009942,
	      "medium": 0.019829,
	      "wide": 0.039686}

    return scales[hdr['CAMNAME']]    


def getPA(hdr):
    return float(hdr['ROTPOSN']) - float(hdr['INSTANGL'])

def getCentralWavelength(hdr):
    return float(hdr['CENWAVE'])

def calcOverhead(tint, coadds, ndithers, nframes, reads, tread=0.181):
    t = 0.0
    
    if (ndithers > 1):
        t += 6.0 * (ndithers + 1.0)
        
    t += 12.0 * nframes * ndithers
    t += ndithers * nframes * coadds * (tint + tread*(reads - 1.0))

    tmin = t / 60.0

    print '\tIntegration time: %.3f' % tint
    print '\tNumber of Coadds: %d' % coadds
    print '\tNumber of Dither Positions: %d' % ndithers
    print '\tFrames at each Position: %d' % nframes
    print '\tNumber of Reads: %d' % reads
    print 'Total elapsed observing time = %5d sec (or %5.1f min)' % \
          (t, tmin)

def plotKeyword(keyword1, keyword2, imgList):
    """
    Pass in a file containing a list of images. For each of these
    images, read out the values of the header keywords specified.
    Then plot each of the keywords against each other.
    """
    tab = asciidata.open(imgList)

    files = [tab[0][i].strip() for i in range(tab.nrows)]

    value1 = zeros(tab.nrows, dtype=float)
    value2 = zeros(tab.nrows, dtype=float)

    print keyword1, keyword2

    for ff in range(len(files)):
        hdr = pyfits.getheader(files[ff],ignore_missing_end=True)

        value1[ff] = hdr[keyword1]
        value2[ff] = hdr[keyword2]


    import pylab as py
    py.clf()

    py.plot(value1, value2, 'k.')
    py.xlabel(keyword1)
    py.ylabel(keyword2)

    return (value1, value2)
    

def getPlateScale():
    """
    Return the plate scale in mas/yr. This is the plate scale reported
    in Yelda et al. 2010.
    """
    return 0.009952
