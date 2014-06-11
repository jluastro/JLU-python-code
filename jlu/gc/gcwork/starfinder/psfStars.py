import asciidata, math
import numpy as np
import pylab as py
import pyfits
import pdb

def find_psf_stars(labelList, centerStar='S1-5', psfRadiusCut=3.5,
                   psfMagCut=13.0, secMagCut=15.5, satLevel=11, outSuffix=''):
    """
    Using a label.dat file as an input list, go through and select all the PSF stars that
    are:

    1) relatively isolated
    2) brighter than K<13
    3) within a certain radius of S1-5 (a star close to the center)
    """
    
    tab = asciidata.open(labelList)

    name = tab[0]._data
    mag = tab[1].tonumpy()
    x = tab[2].tonumpy()
    y = tab[3].tonumpy()
    xerr = tab[4].tonumpy()
    yerr = tab[5].tonumpy()
    vx = tab[6].tonumpy()
    vy = tab[7].tonumpy()
    vxerr = tab[8].tonumpy()
    vyerr = tab[9].tonumpy()
    t0 = tab[10].tonumpy()

    cdx = name.index(centerStar)
    r = np.hypot( x - x[cdx], y - y[cdx] )

    name = np.array(name)

    isPsfStar = np.zeros(len(x), dtype=np.bool)
    isSecStar = np.zeros(len(x), dtype=np.bool)
    isSaturated = np.zeros(len(x), dtype=np.bool)

    ##########
    # 1. Apply secondary magnitude cut... nothing makes it past this.
    ##########
    secondaries = np.where(mag < secMagCut)[0]
    isSecStar[secondaries] = True

    ##########
    # 2. Apply PSF star's radius cut and magnitude cut
    ##########
    psfCandidates = np.where((mag < psfMagCut) & (r < psfRadiusCut))[0]
    
    print 'PSF Candidates: (%d)' % len(psfCandidates)
    for pp in psfCandidates:
        print '%13s  %5.2f  %5.2f' % (name[pp], mag[pp], r[pp])

    ##########
    # 3. Only allow relatively isolated stars to be PSF stars.
    #    Isolation is determined by several factors such as:
    #    -- no stars at all within 100 mas radius
    #    -- no stars at all with delta-mag < 
    #    -- no stars with delta-mag < 3 within 
    ##########

    # Define the envelope of acceptable deltaK vs. radius as vertices.
    # Anything below the envelope is allowed as a PSF star.
    radiusPoints = np.array([ 0.0, 0.08, 0.08,  0.4,  1.0])
    deltaKPoints = np.array([ 5.0,  5.0,  2.0, -3.0, -3.0])

    py.clf()
    py.plot(radiusPoints, deltaKPoints, linewidth=2)
    py.xlabel('Radius (arcsec)')
    py.ylabel('Delta-K (mag)')
    axrng = py.axis()
    py.ylim(axrng[3], axrng[2])
    py.savefig('stf_psf_profile' + outSuffix + '.png')

    for ii in psfCandidates:
        dr = np.hypot(x - x[ii], y - y[ii])
        dm = mag - mag[ii]  # Remember positive is fainter

        
        removeFromPsfList = False
        for rr in range(len(radiusPoints)-1):
            radiusRange = radiusPoints[rr+1] - radiusPoints[rr]
            deltaKRange = deltaKPoints[rr+1] - deltaKPoints[rr]
            
            if (radiusRange == 0):
                continue
            
            slope = deltaKRange / radiusRange
            inter = deltaKPoints[rr] - (slope * radiusPoints[rr])

            dmLimit = inter + slope*dr

            # Check that stars within this radius range aren't too bright
            idx = np.where((dr > radiusPoints[rr]) & (dr <= radiusPoints[rr+1]) & 
                           (dm < dmLimit))[0]
            if (len(idx) > 0):
                print 'Removing %s from cut # %d' % (name[ii], rr)
                for kk in idx:
                    print '  too near: %10s   dr = %5.2f  dm = %5.2f < dmLimit = %5.2f' % \
                        (name[kk], dr[kk], dm[kk], dmLimit[kk])
                print ''
                removeFromPsfList = True
                break

        if removeFromPsfList == True:
            continue

        # If we made it this far without continuing out of the loop, then this 
        # PSF candidate should become a PSF star.
        isPsfStar[ii] = True

    # Go through and mark all the possibly saturated stars as PSF stars.
    # These are the exceptions to all of the above rules... otherwise, they won't be
    # repaired for saturation.
    idx = np.where(mag < satLevel)[0]
    isPsfStar[idx] = True
    isSaturated[idx] = True


    # Print out names of PSF stars.
    idxPsfStars = np.where(isPsfStar)[0]
    print ''
    print 'PSF Stars: (%d, %d saturated)' % (len(idxPsfStars), len(idx))
    print '%13s  %5s  %5s  %5s  %5s  %5s' % ('Name', 'Mag', 'X', 'Y', 'R', 'isSat')
    for pp in idxPsfStars:
        print '%13s  %5.2f  %5.2f  %5.2f  %5.2f  %s' % \
            (name[pp], mag[pp], x[pp], y[pp], r[pp], isSaturated[pp])

        
    print ''
    print 'Secondary Stars (%d)' % (len(secondaries) - len(idxPsfStars))

    _out = open('stf_psf_positions_new' + outSuffix + '.dat', 'w')
    _out.write('%-13s  %5s  %7s  %7s  %6s  %6s  %-8s  %-10s  %-5s\n' %
               ('#Name', 'Mag', 'Xarc', 'Yarc', 'Vx', 'Vy', 't0', 'Filt', 'isPSF'))
    
    for ss in range(len(x)):
        if (isSecStar[ss] == False) and (isPsfStar[ss] == False):
            continue
        
        _out.write('%-13s  ' % name[ss])
        _out.write('%5.2f  ' % mag[ss])
        _out.write('%7.3f  %7.3f  ' % (x[ss], y[ss]))
        _out.write('%6.2f  %6.2f  ' % (vx[ss], vy[ss]))
        _out.write('%8.3f  ' % t0[ss])

        # There are some special filter exclusions
        filter = 'K'  # don't use for speckle

        # speckle PSF stars
        if (name[ss] == 'irs16C' or name[ss] == 'irs16NW' or 
            name[ss] == 'S1-1'   or name[ss] == 'S1-20' or name[ss] == 'S1-3'): 
            filter = '-'

        # Long-wavelength PSF stars or PSF exclusions
        if (name[ss] == 'irs16SW' or name[ss] == 'irs29N'):
            filter = 'K,LP,MS'

        if (name[ss] == 'S1-23'):
            filter = 'K,MS'

        _out.write('%-10s  ' % filter)

        # Is PSF Star
        isPsf = 0
        if (isPsfStar[ss] == True):
            isPsf = 1

        _out.write('%3d\n' % isPsf)

    _out.close()

    return

def plot_psf_stars(psfList, fitsFile='/u/ghezgroup/data/gc/09maylgs1/combo/mag09maylgs1_kp.fits', sgraX=576.875, sgraY=681.500):
    """
    Plot the PSf starlist over an image.
    """
    tab = asciidata.open(psfList)

    name = tab[0]._data
    mag = tab[1].tonumpy()
    x = tab[2].tonumpy()
    y = tab[3].tonumpy()
    vx = tab[4].tonumpy()
    vy = tab[5].tonumpy()
    t0 = tab[6].tonumpy()
    filt = tab[7]._data
    isPsf = (tab[8].tonumpy() == 1)

    # Check the PA of the image and rotate to North up if necessary
    hdr = pyfits.getheader(fitsFile)
    pa = float(hdr['ROTPOSN']) - 0.7
    if pa != 0:
        rotPos = rotate_pos(x, y, pa)
        x = rotPos[0]
        y = rotPos[1]

    # Remove PSF stars that should be rejected for the K' filter.
    kpReject = np.zeros(len(x), dtype=np.bool)
    for ff in range(len(filt)):
        filters = filt[ff].split(',')
        if 'KP' in filters:
            kpReject[ff] = True


    isPsf[kpReject == True] = False

    sgra = np.array([sgraX, sgraY])

    img = pyfits.getdata(fitsFile)

    xaxis = (np.arange(img.shape[1]) - sgra[0]) * 0.00995 * -1.0
    yaxis = (np.arange(img.shape[0]) - sgra[1]) * 0.00995
#    pdb.set_trace()

    py.close(2)
    py.figure(2, figsize=(12, 12))
    py.clf()
    py.imshow(np.log10(img), extent=[xaxis[0], xaxis[-1], yaxis[0], yaxis[-1]],
              vmin=1, vmax=math.log10(40000), cmap=py.cm.Greys, origin='lowerleft')
    py.plot(x[isPsf == False], y[isPsf == False], 'bx', mew=2)
    py.plot(x[isPsf == True],  y[isPsf == True],  'rx', mew=2)
    
    idxPsf = np.where(isPsf == True)[0]
    for pp in range(len(idxPsf)):
        py.text(x[idxPsf[pp]], y[idxPsf[pp]], name[idxPsf[pp]], color='red')

    py.axis([xaxis[0], xaxis[-1], yaxis[0], yaxis[-1]])
    py.title(psfList)
    py.xlabel('R.A. Offset from Sgr A* (arcsec)')
    py.ylabel('Dec. Offset from Sgr A* (arcsec)')
    py.savefig(psfList + '.png')
    

def rotate_pos(x, y, phi):
    # Rotate around center of image, and keep origin at center
    cos = math.cos(math.radians(phi))
    sin = math.sin(math.radians(phi))
    
    xrot = x * cos - y * sin 
    yrot = x * sin + y * cos
    
    return [xrot, yrot]

