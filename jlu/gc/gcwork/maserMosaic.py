import asciidata
import numpy as np
import pylab as py
import pyfits, math
import gcutil
from gcwork import starTables

def findOverlapStars(mosaicRoot, mosaicStars, 
                     cleanRoot1, cleanStars1,
                     cleanRoot2, cleanStars2):
    """
    Take two cleaned images from a maser mosaic and use the combined
    mosaic (with starfinder run on it) to identify the set of stars 
    that are in both cleaned images. These stars in the overlap region
    can be used to align the two individual cleaned frames.
    """

    # Read in the shifts file and extract shifts for the two frames
    shiftsFile = mosaicRoot + '.shifts'
    print(shiftsFile)

    shiftsTable = asciidata.open(shiftsFile)
    cleanFiles = shiftsTable[0]._data
    xshifts = shiftsTable[1].tonumpy()
    yshifts = shiftsTable[2].tonumpy()

    idx1 = -1
    idx2 = -1
    for cc in range(len(cleanFiles)):
        if cleanRoot1 in cleanFiles[cc]:
            idx1 = cc
        if cleanRoot2 in cleanFiles[cc]:
            idx2 = cc

    print(idx1, idx2, cleanFiles[idx1], cleanFiles[idx2])
    xshift1 = xshifts[idx1]
    xshift2 = xshifts[idx2]
    yshift1 = yshifts[idx1]
    yshift2 = yshifts[idx2]
    print('Shifts:')
    print(cleanFiles[idx1], xshift1, yshift1)
    print(cleanFiles[idx2], xshift2, yshift2)

    # Read in the mosaic fits file to get the size of the mosaic.
    # Then define coordinates for the mosaic
    fitsFile = mosaicRoot + '.fits'
    fitsImg = pyfits.getdata(fitsFile)
    mosaicSize = fitsImg.shape
    print('Mosaic Size: ', mosaicSize)

    (ym, xm) = np.meshgrid(np.arange(mosaicSize[0]), np.arange(mosaicSize[1]))
    mosaicHalfX = mosaicSize[1] / 2.0
    mosaicHalfY = mosaicSize[0] / 2.0

    # Define coordinates for the cleaned frames and shift accordingly.
    (y1orig, x1orig) = np.meshgrid(np.arange(1024), np.arange(1024))
    (y2orig, x2orig) = np.meshgrid(np.arange(1024), np.arange(1024))
    cleanHalf = 512.0

    x1 = ((x1orig - cleanHalf) + xshift1) + mosaicHalfX
    y1 = ((y1orig - cleanHalf) + yshift1) + mosaicHalfY
    x2 = ((x2orig - cleanHalf) + xshift2) + mosaicHalfX
    y2 = ((y2orig - cleanHalf) + yshift2) + mosaicHalfY
    box1x = np.array([x1.min(), x1.max(), x1.max(), x1.min(), x1.min()])
    box1y = np.array([y1.min(), y1.min(), y1.max(), y1.max(), y1.min()])
    box2x = np.array([x2.min(), x2.max(), x2.max(), x2.min(), x2.min()])
    box2y = np.array([y2.min(), y2.min(), y2.max(), y2.max(), y2.min()])

    overlap = np.where((x1 >= x2.min()) & (x1 <= x2.max()) & (y1 >= y2.min()) & (y1 <= y2.max()))
    xo = x1[overlap]
    yo = y1[overlap]

    xo_min = xo.min()
    xo_max = xo.max()
    yo_min = yo.min()
    yo_max = yo.max()

    print('Overlap Range:')
    print(xo_min, xo_max, yo_min, yo_max)
    box3x = np.array([xo_min, xo_max, xo_max, xo_min, xo_min])
    box3y = np.array([yo_min, yo_min, yo_max, yo_max, yo_min])

    # Read in the two cleaned star lists and the mosaic list
    stars1 = starTables.StarfinderList(cleanStars1)
    stars2 = starTables.StarfinderList(cleanStars2)
    starsM = starTables.StarfinderList(mosaicStars)

    # Extract stars from the mosaic list in the overlap region.
    idx = np.where((starsM.x >= xo_min) & (starsM.x <= xo_max) & 
                   (starsM.y >= yo_min) & (starsM.y <= yo_max))[0]
    
    # Trim out named sources
    idxNamed = []
    for ii in idx:
        if not starsM.name[ii].startswith('star'):
            idxNamed.append(ii)

    # Loop through and find matching sources (by name)
    idxIn1 = []
    idxIn2 = []
    for ii in idxNamed:
        try:
            match1 = stars1.name.index(starsM.name[ii])
            match2 = stars2.name.index(starsM.name[ii])
        except ValueError:
            continue

        idxIn1.append(match1)
        idxIn2.append(match2)

        print(stars1.name[match1], stars2.name[match2])

def mosaicDistortionErrors(rootDir, ep, mosaicRoot, mosaicStarlist,
                           addResid=True, distResid=0.11):
    """
    Read in a maser mosaic and it's associated *.shifts file and
    calculate the distortion error contribution to each pixel in the 
    maser mosaic. This is done by using the NIRC2 single frame distortion
    error maps (X and Y) located here:

    /u/ghezgroup/code/python/gcreduce/nirc2dist_xerr.fits
    /u/ghezgroup/code/python/gcreduce/nirc2dist_yerr.fits

    For each independent dither position (where independent excludes any
    frame within 30 pixels of another frame), the distortion error map
    is shifted to the appropriate location in the final moasic and 
    added in quadrature as sqrt( err1^2 + err2^2 + ... + errN^2 ) / sqrt(N).
    The resulting distortion error maps for the mosaics are saved to FITS
    files with the name:
    
    <mosaicRoot>_xdisterr.fits
    <mosaicRoot>_ydisterr.fits

    Finally, <mosaicStarlist> is the existing positional errors are
    added in quadrature to the distortion error at the nearest pixel for
    each star. The final starlist is saved off to the following file in 
    the same directory as the input list:

    <mosaicStarlist_root>_rms_dist.lis

    Inputs:
    mosaicRoot - The root name of the maser mosaic fits file
                 (e.g. 'mag08maylgs1_msr_kp')

    mosaicStarlist - The full name of a maser starlist with positional 
                     errors (e.g. 'starfinder/mag08maylgs1_msr_kp_0.8_rms.lis')
    addResid - Set to True to add residual distortion error (0.11 pix) in
               quadrature to the centroiding and distortion errors
    distResid - Value of residual distortion in pixels, to be added to centroid
                and distortion errors in quadrature (def=0.11 pix).

    """
    # Read in the shifts file
    shiftsFile = mosaicRoot + '.shifts'
    print(shiftsFile)

    shiftsTable = asciidata.open(shiftsFile)
    singleFiles = shiftsTable[0]._data
    xshifts = shiftsTable[1].tonumpy()
    yshifts = shiftsTable[2].tonumpy()

    # Read in the mosaic fits file to get the size of the mosaic.
    # Then define coordinates for the mosaic.
    fitsFile = mosaicRoot + '.fits'
    fitsImg, fitsHdr = pyfits.getdata(fitsFile, header=True)
    mosaicSize = fitsImg.shape

    # Make the final distortion error map.
    # Also keep the number of frames that contributes to each pixel.
    xdistErrMap = np.zeros((mosaicSize[0], mosaicSize[1]), dtype=float)
    ydistErrMap = np.zeros((mosaicSize[0], mosaicSize[1]), dtype=float)
    distErrCnt = np.zeros((mosaicSize[0], mosaicSize[1]), dtype=int)

    # Read in the distortion errors map
    errorRoot = '/u/ghezgroup/code/python/gcreduce/nirc2dist'
    xerrOrig = pyfits.getdata(errorRoot + '_xerr.fits')
    yerrOrig = pyfits.getdata(errorRoot + '_yerr.fits')

    # Get the half-way points for both the mosaic and the single frame.
    # For zero-shift, the center of the single frame lies at the center of
    # the mosaic.
    mosaicHalfX = mosaicSize[1] / 2.0
    mosaicHalfY = mosaicSize[0] / 2.0
    singleHalfX = xerrOrig.shape[1] / 2.0
    singleHalfY = xerrOrig.shape[0] / 2.0
    print('Mosaic Size: ', mosaicSize)
    print('Mosaic Half: ', mosaicHalfX, mosaicHalfY)
    print('Single Half: ', singleHalfX, singleHalfY)

    # Loop through the individual frames and add them in 
    # quadrature to the final distortion error map. Remember that we
    # can only do this for independent dither positions. We will 
    # consider independent dither positions to be those that are
    # more than 30 pixel distant from any other ones.
    xshiftsUsed = np.array([])
    yshiftsUsed = np.array([])
    allowedSep = 30.0  # pixels

    for ii in range(len(singleFiles)):
        # Check to see if we should count this single file as
        # a new "independent" position by comparing to all 
        # shifts used so far.
        sep = np.hypot(xshifts[ii] - xshiftsUsed, yshifts[ii] - yshiftsUsed)
        idx = np.where(sep < allowedSep)[0]
        
        # If used before, skip
        if (len(idx) > 0):
            continue

        # This is a new independent dither position
        xshiftsUsed = np.append(xshiftsUsed, xshifts[ii])
        yshiftsUsed = np.append(yshiftsUsed, yshifts[ii])
        
        # Calculate the stop and start indices in the maser mosaic
        xlo = round(mosaicHalfX + xshifts[ii] - singleHalfX)
        xhi = round(mosaicHalfX + xshifts[ii] + singleHalfX)
        ylo = round(mosaicHalfY + yshifts[ii] - singleHalfY)
        yhi = round(mosaicHalfY + yshifts[ii] + singleHalfY)

        print(singleFiles[ii], xshifts[ii], yshifts[ii])
        print('  xrange = [%4d:%4d]  yrange = [%4d:%4d]' % (xlo, xhi, ylo, yhi))

        xdistErrMap[ylo:yhi,xlo:xhi] += xerrOrig**2
        ydistErrMap[ylo:yhi,xlo:xhi] += yerrOrig**2
        distErrCnt[ylo:yhi,xlo:xhi] += 1.0

    idx = np.where(distErrCnt != 0)
    xdistErrMap[idx] = np.sqrt(xdistErrMap[idx] / distErrCnt[idx])
    ydistErrMap[idx] = np.sqrt(ydistErrMap[idx] / distErrCnt[idx])

    # Save these to a FITS file
    xfits = pyfits.PrimaryHDU(xdistErrMap)
    gcutil.rmall([mosaicRoot + '_xdisterr.fits', mosaicRoot + '_ydisterr.fits'])
    pyfits.writeto(mosaicRoot + '_xdisterr.fits', xdistErrMap)
    pyfits.writeto(mosaicRoot + '_ydisterr.fits', ydistErrMap)
    

    ##########
    #
    # Read in starlist and apply distortion errors.
    #
    ##########
    starlist = asciidata.open(mosaicStarlist)
    
    # Check that this starlist properly has error columns
    if starlist.nrows < 11:
        print('Starlist does not have error columns: %s' % mosaicStarlist)

    # Now for each star, add the distortion errors in quadrature
    for rr in range(starlist.nrows):
        xpix = round(starlist[3][rr])
        ypix = round(starlist[4][rr])

        if addResid:
            xerr = math.sqrt(starlist[5][rr]**2 + xdistErrMap[ypix,xpix]**2 + \
                             distResid**2)
            yerr = math.sqrt(starlist[6][rr]**2 + ydistErrMap[ypix,xpix]**2 + \
                             distResid**2)
        else:
            xerr = math.sqrt(starlist[5][rr]**2 + xdistErrMap[ypix,xpix]**2)
            yerr = math.sqrt(starlist[6][rr]**2 + ydistErrMap[ypix,xpix]**2)

        starlist[5][rr] = xerr
        starlist[6][rr] = yerr

    # Reformat the columns so they get printed out nicely
    starlist[0].reformat('%-13s ')
    starlist[1].reformat('%6.3f ')
    starlist[2].reformat('%8.3f')
    starlist[3].reformat('%9.3f ')
    starlist[4].reformat('%9.3f ')
    starlist[5].reformat('%6.3f')
    starlist[6].reformat('%6.3f')
    starlist[7].reformat('%11.4f')
    starlist[8].reformat('%5.2f')
    starlist[9].reformat('%5d')
    starlist[10].reformat('%8.4f')

    # Write out the new starlist
    outroot = mosaicStarlist.replace('rms', 'rms_dist')
    starlist.writeto(outroot)
    
