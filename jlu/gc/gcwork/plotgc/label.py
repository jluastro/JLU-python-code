import math
import numpy as np
import asciidata
import Image, ImageDraw
import pylab as py
import pyfits
from gcwork import starTables
from gcwork import starset
from gcreduce import gcutil
import pdb

def plot(labels='/u/ghezgroup/data/gc/source_list/label.dat'):
    ## Read in star labels
    tab = asciidata.open(labels)
    name = [tab[0][ss].strip() for ss in range(tab.nrows)]
    mag = tab[1].tonumpy()
    x0 = tab[2].tonumpy()
    y0 = tab[3].tonumpy()
    vx = tab[6].tonumpy()
    vy = tab[7].tonumpy()
    vxe = tab[8].tonumpy()
    vye = tab[9].tonumpy()
    t0 = tab[10].tonumpy()

    t = 2005.580

    xarc = x0 + vx * (t - t0) / 10**3
    yarc = y0 + vy * (t - t0) / 10**3

    vxarc = vx
    vyarc = vy

    angle = math.radians(0.0)
    cosa = math.cos(angle)
    sina = math.sin(angle)

    x = xarc * cosa + yarc * sina
    y = -xarc * sina + yarc * cosa

    vx = vxarc * cosa + vyarc * sina
    vy = vxarc * sina - vyarc * cosa

    r = np.hypot(x, y)
    v = np.hypot(vx, vy)

    # Lets also make some coordinates for compass rose and scale bar
    xrose = np.array([-15.5, -15.5])
    yrose = np.array([15.5, 15.5])
    xroseLen = np.array([20.0, 0.0])
    yroseLen = np.array([0.0, 20.0])

    xr = xrose
    yr = yrose
    xrlen = xroseLen * cosa + yroseLen * sina
    yrlen = xroseLen * sina - yroseLen * cosa

    # Image
    #im = Image.open('/u/ghezgroup/public_html/gc/images/media/rgb05jullgs_hires_plain.png')
    #imgsize = (im.size)[0]
    im = pyfits.getdata('/u/ghezgroup/data/gc/06maylgs1/combo/mag06maylgs1_dp_msc_kp.fits')
    imgsize = (im.shape)[0]

    # pixel position (0,0) is at upper left
    xpix = np.arange(0, im.shape[0], dtype=float)
    ypix = np.arange(0, im.shape[1], dtype=float)

    print im.shape
    sgra = [1422.6, 1543.8]
    #scale_jpg = 0.005
    scale_jpg = 0.00994
    xim = (xpix - sgra[0]) * scale_jpg * -1.0
    yim = (ypix - sgra[1]) * scale_jpg
    
    py.clf()
    py.close(2)
    py.figure(2, figsize=(6,4.5))
    py.grid(True)
    py.subplots_adjust(left=0.05, right=0.95, bottom=0.05, top=0.95)
    py.imshow(np.log10(im), extent=[xim[0], xim[-1], yim[0], yim[-1]],
              aspect='equal', vmin=1.9, vmax=6.0, cmap=py.cm.gray)
    py.xlabel('X Offset from Sgr A* (arcsec)')
    py.ylabel('Y Offset from Sgr A* (arcsec)')
    py.title('UCLA/Keck Galactic Center Group', fontsize=20, fontweight='bold')
    thePlot = py.gca()
    #thePlot.get_xaxis().set_major_locator(py.MultipleLocator(1))

    py.axis([15, -15, -15, 15])
    
    idx = (np.where( (mag < 18) &
                     (x > xim.min()) & (x < xim.max()) &
                     (y > yim.min()) & (y < yim.max()) ))[0]

    py.plot(x[idx], y[idx], 'r+', color='orange')
    for ii in idx:
        py.text(x[ii], y[ii], name[ii], color='orange', fontsize=10)

    py.savefig('gcposter_10x10.png', dpi=300)

def poster():
    ## Read in star labels
    labels = '/u/ghezgroup/data/gc/source_list/label.dat'

    tab = asciidata.open(labels)
    name = [tab[0][ss].strip() for ss in range(tab.nrows)]
    mag = tab[1].tonumpy()
    x0 = tab[2].tonumpy()
    y0 = tab[3].tonumpy()
    vx = tab[6].tonumpy()
    vy = tab[7].tonumpy()
    vxe = tab[8].tonumpy()
    vye = tab[9].tonumpy()
    t0 = tab[10].tonumpy()

    t = 2005.580

    xarc = x0 + vx * (t - t0) / 10**3
    yarc = y0 + vy * (t - t0) / 10**3

    vxarc = vx
    vyarc = vy

    angle = math.radians(-10.0)
    cosa = math.cos(angle)
    sina = math.sin(angle)

    x = xarc * cosa + yarc * sina
    y = -xarc * sina + yarc * cosa

    vx = vxarc * cosa + vyarc * sina
    vy = vxarc * sina - vyarc * cosa

    r = np.hypot(x, y)
    v = np.hypot(vx, vy)

    # Lets also make some coordinates for compass rose and scale bar
    xrose = np.array([-4.5, -4.5])
    yrose = np.array([4.5, 4.5])
    xroseLen = np.array([20.0, 0.0])
    yroseLen = np.array([0.0, 20.0])

    xr = xrose
    yr = yrose
    xrlen = xroseLen * cosa + yroseLen * sina
    yrlen = xroseLen * sina - yroseLen * cosa

    # Image
    im = Image.open('/u/ghezgroup/public_html/gc/images/media/rgb05jullgs_hires_plain.png')
    imgsize = (im.size)[0]

    # pixel position (0,0) is at upper left
    xpix = np.arange(0, im.size[0], dtype=float)
    ypix = np.arange(0, im.size[1], dtype=float)

    print im.size
    sgra = [998.0, 755]
    scale_jpg = 0.005
    xim = (xpix - sgra[0]) * scale_jpg * -1.0
    yim = (ypix - sgra[1]) * scale_jpg
    
    py.clf()
    py.close(2)
    py.figure(2, figsize=(14,12.5))
    py.grid(True)
    py.subplots_adjust(left=0.05, right=0.95, bottom=0.05, top=0.95)
    py.imshow(im, extent=[xim[0], xim[-1], yim[0], yim[-1]],
              aspect='equal')
    py.xlabel('X Offset from Sgr A* (arcsec)')
    py.ylabel('Y Offset from Sgr A* (arcsec)')
    py.title('UCLA/Keck Galactic Center Group', fontsize=20, fontweight='bold')
    thePlot = py.gca()
    thePlot.get_xaxis().set_major_locator(py.MultipleLocator(1))
    

    # Draw velocity vectors
    arrScale = 1.0*10**2

    idx = (np.where( (abs(vx) > 0) &
                     (v < 20) &
                     (mag < 18) &
                     (r > 0.5) ))[0]
    idx2 = (np.where( (r[idx] <= 4) | (v[idx] < 5) ))[0]
    idx = idx[idx2]
    py.quiver(x[idx], y[idx], vx[idx], vy[idx], scale=arrScale, color='orange',
              units='x', width=0.01, headlength=3, headaxislength=3)

    py.quiver(xr, yr, xrlen, yrlen, scale=arrScale, color='w',
              units='x', width=0.02, headlength=3, headaxislength=3)
    py.text(-4.2, 4.5, 'E', color='w', fontsize=8)
    py.text(-4.52, 4.75, 'N', color='w', fontsize=8)
    py.text(-4.5, 4.35, 'Compass arrow', color='w', fontsize=8,
            horizontalalignment='center')
    py.text(-4.5, 4.25, 'is 20 mas/yr', color='w', fontsize=8,
            horizontalalignment='center')

    py.axis([4.9, -4.9, -3.75, 4.9])

    idx = (np.where( (mag < 18) &
                     (x > xim.min()) & (x < xim.max()) &
                     (y > yim.min()) & (y < yim.max()) ))[0]
    for ii in idx:
        py.text(x[ii], y[ii], name[ii], color='orange', fontsize=4)

    py.savefig('gcposter_10x10.png', dpi=300)
#    py.savefig('gcposter_10x10.eps')


def plotLabelAndAlign(alignRoot, polyRoot, magCut=25,
                      labelFile='label.dat', showStars=True):

    label = starTables.Labels(labelFile)
    label.take(np.where(label.mag < magCut)[0])

    t = 2005.580
    x_lab = label.x + label.vx * (t - label.t0) / 10**3
    y_lab = label.y + label.vy * (t - label.t0) / 10**3
    n_lab = label.ourName
    use = label.useToAlign

    # Now read in the align stuff
    s = starset.StarSet(alignRoot)
    s.loadPolyfit(polyRoot, accel=0)
    mag = s.getArray('mag')
    idx = np.where(mag < magCut)[0]
    s.stars = [s.stars[ii] for ii in idx]

    n_aln = np.array(s.getArray('name'))
    mag = s.getArray('mag')
    x0 = s.getArray('fitXv.p') * -1.0
    y0 = s.getArray('fitYv.p')
    x0err = s.getArray('fitXv.perr')
    y0err = s.getArray('fitYv.perr')
    vx = s.getArray('fitXv.v') * -1.0
    vy = s.getArray('fitYv.v')
    vxerr = s.getArray('fitXv.verr')
    vyerr = s.getArray('fitYv.verr')
    t0x = s.getArray('fitXv.t0')
    t0y = s.getArray('fitYv.t0')
    r = np.hypot(x0, y0)

    x_aln = x0 + vx * (t - t0x)
    y_aln = y0 + vy * (t - t0y)

    # Fix x0err, y0err to some minimum value
    idx = np.where(x0err < 0.0001)[0]
    x0err[idx] = 0.0001
    idx = np.where(y0err < 0.0001)[0]
    y0err[idx] = 0.0001

    # First lets update the 16 sources since they are used
    # to align label.dat to the reference epoch (crude I know)
    names16 = ['irs16C', 'irs16NW', 'irs16CC', 'irs16NE', 'irs16SW',
               'irs16SW-E', 'irs33N', 'irs33E']
    for nn in range(len(names16)):
        idx = np.where(n_aln == names16[nn])[0]

        for rr in idx:
            print '%-11s  %4.1f    %6.3f  %6.3f  %7.4f %7.4f  %8.3f %8.3f  %7.3f  %7.3f   %8.3f    1   %5.3f' % \
                  (n_aln[rr], mag[rr], x0[rr], y0[rr], x0err[rr],
                   y0err[rr], vx[rr]*10**3, vy[rr]*10**3,
                   vxerr[rr]*10**3, vyerr[rr]*10**3, t0x[rr], r[rr])

    # Image
    imageFile = '/u/ghezgroup/data/gc/'
    imageFile += '06maylgs1/combo/mag06maylgs1_dp_msc_kp.fits'
    im = pyfits.getdata(imageFile)
    imgsize = (im.shape)[0]

    # pixel position (0,0) is at upper left
    xpix = np.arange(0, im.shape[0], dtype=float)
    ypix = np.arange(0, im.shape[1], dtype=float)

    sgra = [1422.6, 1543.8]
    scale_jpg = 0.00994
    xim = (xpix - sgra[0]) * scale_jpg * -1.0
    yim = (ypix - sgra[1]) * scale_jpg
    
    # Lets also make some coordinates for compass rose and scale bar
    xrose = np.array([-15.5, -15.5])
    yrose = np.array([15.5, 15.5])
    xroseLen = np.array([20.0, 0.0])
    yroseLen = np.array([0.0, 20.0])

    py.clf()
    py.close(1)
    py.close(2)
    def drawImage(xlo, xhi, ylo, yhi):
        py.figure(2, figsize=(9, 9))
        py.grid(True)
        py.subplots_adjust(left=0.05, right=0.95, bottom=0.05, top=0.95)
        py.imshow(np.log10(im), extent=[xim[0], xim[-1], yim[0], yim[-1]],
                  aspect='equal', vmin=2.0, vmax=4.0, cmap=py.cm.gray)
        py.xlabel('X Offset from Sgr A* (arcsec)')
        py.ylabel('Y Offset from Sgr A* (arcsec)')
        py.title('UCLA/Keck Galactic Center Group',
                 fontsize=20, fontweight='bold')
        thePlot = py.gca()
    
        # Plot label.dat points
        py.plot(x_lab, y_lab, 'rs')
        for ii in range(len(x_lab)):
            py.text(x_lab[ii], y_lab[ii], n_lab[ii], color='red', fontsize=10)

        # Plot align points
        py.plot(x_aln, y_aln, 'bo')
        for ii in range(len(x_aln)):
            py.text(x_aln[ii], y_aln[ii], n_aln[ii], color='blue', fontsize=10)

        
        py.axis([xlo, xhi, ylo, yhi])

    # We can do some matching here to help us out.
    drawImage(15, -15, -15, 15)

    if showStars:
        fmt = '%-11s   %4.1f   %9.5f %9.5f   %8.5f  %8.5f  %8.3f %8.3f  %7.3f  %7.3f   %8.3f    %1d   %5.3f\n'
        gcutil.rmall(['label_updated.dat'])
        out = open('label_updated.dat','w')

        out.write('#Name           K         x       y           xerr      yerr       vx       vy    vxerr    vyerr         t0  use?    r2d\n')
        out.write('#()         (mag)      (asec)    (asec)     (asec)    (asec)  (mas/yr) (mas/yr) (mas/yr) (mas/yr)     (year)   ()  (asec)\n')
        foo = raw_input('Continue?')
        for ss in range(len(x_lab)):
            update = 0

            idx = np.where(n_aln == n_lab[ss])[0]
                
            # The rest of the code allows us to find stars that were not matched
            # between the old label and new absolute alignment.
            # This will write out the new info from absolute alignment to a new
            # label_updated.dat file for just those stars that weren't matched properly,
            # but still need to run updateLabelInfoWithAbsRefs() to update the info
            # for stars that were matched.
            if len(idx) == 0:
                dr = np.hypot(x_aln - x_lab[ss], y_aln - y_lab[ss])
                
                rdx = np.where(dr < 0.2)[0]

                if len(rdx) > 0:
                    xlo = x_lab[ss] + 0.2
                    xhi = x_lab[ss] - 0.2
                    ylo = y_lab[ss] - 0.2
                    yhi = y_lab[ss] + 0.2
                    py.axis([xlo, xhi, ylo, yhi])
                    
                    print 'Did not find a match for %s (K=%4.1f  x=%7.3f y = %7.3f):' % \
                          (n_lab[ss], label.mag[ss], x_lab[ss], y_lab[ss])

                    for rr in rdx:
                        print fmt % \
                              (n_aln[rr], mag[rr], x0[rr], y0[rr], x0err[rr],
                               y0err[rr], vx[rr]*10**3, vy[rr]*10**3,
                               vxerr[rr]*10**3, vyerr[rr]*10**3, t0x[rr], use[ss], r[rr])

                        # if the starname in align has the same ending as the name in
                        # label.dat, it's likely the same star (e.g. '1S9-87' and 'S9-87';
                        # Note that this is the naming used in mosaicked star list analysis)
                        if n_aln[rr].endswith(n_lab[ss]):
                            # do a check that these stars have similar magnitudes
                            if np.abs(label.mag[ss] - mag[rr]) <= 0.5:
                                update = 1
                                # replace the whole line in label_updated.dat
                                out.write(fmt % \
                                          (n_lab[ss], mag[rr], x0[rr], y0[rr], x0err[rr],
                                           y0err[rr], vx[rr]*10**3, vy[rr]*10**3,
                                           vxerr[rr]*10**3, vyerr[rr]*10**3, t0x[rr], use[ss], r[rr]))
                                continue
                            else:
                                foo = raw_input('Similar names, but magnitudes are off. Check star %s manually and decide whether or not to replace it in label_updated.dat when this program completes. Hit enter to continue.' % n_aln[rr])
                             

                        # Or the star just wasn't matched with any star in the old label.dat file
                        # and was given an arbitrary name
                        elif (('star' in n_aln[rr]) | ('ep' in n_aln[rr])):
                            print 'Manually check and update this star if needed:'
                            print 'Name in label: %s' % n_lab[ss]
                            print 'Name in new alignment: %s' % n_aln[rr]
                            if len(rdx) > 1:
                                print 'CAUTION: There are other nearby stars, including:'
                                for ii in range(len(rdx)):
                                    print n_aln[rdx[ii]]
                            update = raw_input('Update with %s velocity? (enter 1 for yes; 0 for no) ' % n_aln[rr])
                            update = int(update)
                            if update == 1:
                                out.write(fmt % \
                                          (n_lab[ss], mag[rr], x0[rr], y0[rr], x0err[rr],
                                           y0err[rr], vx[rr]*10**3, vy[rr]*10**3,
                                           vxerr[rr]*10**3, vyerr[rr]*10**3, t0x[rr], use[ss], r[rr]))
                            # check magnitude offset
                            #if np.abs(label.mag[ss] - mag[rr]) <= 0.5:
                            #    update = 1
                            #    # replace the whole line in label_updated.dat
                            #    out.write(fmt % \
                            #              (n_lab[ss], mag[rr], x0[rr], y0[rr], x0err[rr],
                            #               y0err[rr], vx[rr]*10**3, vy[rr]*10**3,
                            #               vxerr[rr]*10**3, vyerr[rr]*10**3, t0x[rr], 1, r[rr]))
                            
            if update == 0: 
                # just re-write what was in label.dat, and later run
                # updateLabelInfoWithAbsRefs() to update matched stars with the new
                # absolute alignment
                out.write(fmt % \
                          (label.ourName[ss], label.mag[ss], label.x[ss], label.y[ss],
                          label.xerr[ss], label.yerr[ss], label.vx[ss], label.vy[ss],
                          label.vxerr[ss], label.vyerr[ss], label.t0[ss], use[ss],
                          label.r[ss]))

        out.close()

def plotPhotoCalib(image, cooStar,
                   photoCalib='/u/ghezgroup/data/gc/source_list/photo_calib.dat'):
    """
    Plot the specified image and overlay the photo_calib.dat sources on top.
    Coordinates are converted from pixels to arcsec using the coo star and
    assuming that the angle of the image is 0.
    """
    # Load up the photometric calibraters table.
    _tab = asciidata.open(photoCalib)

    name = _tab[0].tonumpy()
    x = _tab[1].tonumpy()
    y = _tab[2].tonumpy()

    # Load up the image
    imageRoot = image.replace('.fits', '')
    im = pyfits.getdata(imageRoot + '.fits')

    # Coo star pixel coordinates
    _coo = open(imageRoot + '.coo', 'r')
    tmp = _coo.readline().split()
    cooPixel = [float(tmp[0]), float(tmp[1])]

    imgsize = (im.shape)[0]
    xpix = np.arange(0, im.shape[0], dtype=float)
    ypix = np.arange(0, im.shape[1], dtype=float)

    cooIdx = np.where(name == cooStar)[0]
    if len(cooIdx) == 0:
        print 'Failed to find the coo star %s in %s' % (cooStar, photoCalib)

    cooArcsec = [x[cooIdx[0]], y[cooIdx[0]]]

    scale = 0.00994
    xim = ((xpix - cooPixel[0]) * scale * -1.0) + cooArcsec[0]
    yim = ((ypix - cooPixel[1]) * scale) + cooArcsec[1]
    
    py.figure(1)
    py.clf()
    py.grid(True)
    py.subplots_adjust(left=0.05, right=0.95, bottom=0.05, top=0.95)
    py.imshow(np.log10(im), extent=[xim[0], xim[-1], yim[0], yim[-1]],
              aspect='equal', vmin=1.9, vmax=6.0, cmap=py.cm.gray)
    py.xlabel('X Offset from Sgr A* (arcsec)')
    py.ylabel('Y Offset from Sgr A* (arcsec)')
    py.title(imageRoot)

    thePlot = py.gca()
    
    idx = (np.where((x > xim.min()) & (x < xim.max()) &
                    (y > yim.min()) & (y < yim.max()) ))[0]

    py.plot(x[idx], y[idx], 'r+', color='orange')
    for ii in idx:
        py.text(x[ii], y[ii], name[ii], color='orange', fontsize=12)

def plotStarfinderList(starList, hasErrors=True, magCut=18,
                       cooStarList='16C', cooStarLabels='irs16C', scaleList=0.00995, scaleImg=0.00995,
                       image='/u/ghezgroup/data/gc/06maylgs1/combo/mag06maylgs1_dp_msc_kp.fits'):
    """
    Plot the specified image and overlay the photo_calib.dat sources on top.
    Coordinates are converted from pixels to arcsec using the coo star and
    assuming that the angle of the image is 0.

    This code assumes coordinates are NIRC2 narrow pixels coordaintes. You can modify this by 
    changing the scale. But +x must be to the west in the starlist.
    """

    # Load up the photometric calibraters table.
    lis = starTables.StarfinderList(starList, hasErrors=hasErrors)
    labels = starTables.Labels()

    # Find the coo star in the starlist and in the labels
    ii1 = np.where(lis.name == cooStarList)[0]
    ii2 = np.where(labels.name == cooStarLabels)[0]

    dt = lis.epoch[0] - labels.t0
    labels.x += (labels.vx / 10**3) * dt
    labels.y += (labels.vy / 10**3) * dt

    # Convert the pixels in the starlist into arcsec
    x = ((lis.x - lis.x[ii1]) * -scaleList) + labels.x[ii2]
    y = ((lis.y - lis.y[ii1]) * scaleList) + labels.y[ii2]

    im = pyfits.getdata(image)
    imgsize = (im.shape)[0]

    # pixel position (0,0) is at upper left
    xpix = np.arange(0, im.shape[0], dtype=float)
    ypix = np.arange(0, im.shape[1], dtype=float)

    # Read in the image coo file
    # Coo star pixel coordinates
    _coo = open(image.replace('.fits', '.coo'), 'r')
    coordsTmp = _coo.readline().split()
    coords = [float(coo) for coo in coordsTmp]
    print 'Coordinates for %s:' % cooStarLabels
    print '  [%10.4f, %10.4f] pixels' % (coords[0], coords[1])
    print '  [%10.4f, %10.4f] arcsec' % (labels.x[ii2], labels.y[ii2])

    sgrax = coords[0] - (labels.x[ii2] / -scaleImg)
    sgray = coords[1] - (labels.y[ii2] / scaleImg)
    sgra = [sgrax, sgray]

    # Image coordinates (in arcsec)
    xim = (xpix - sgra[0]) * -scaleImg
    yim = (ypix - sgra[1]) * scaleImg
    
    py.clf()
    py.close(2)
    py.figure(2, figsize=(6,4.5))
    py.grid(True)
    py.subplots_adjust(left=0.05, right=0.95, bottom=0.05, top=0.95)
    py.imshow(np.log10(im), extent=[xim[0], xim[-1], yim[0], yim[-1]],
              aspect='equal', vmin=1.9, vmax=6.0, cmap=py.cm.gray)
    py.xlabel('X Offset from Sgr A* (arcsec)')
    py.ylabel('Y Offset from Sgr A* (arcsec)')
    py.title('UCLA/Keck Galactic Center Group', fontsize=20, fontweight='bold')
    thePlot = py.gca()

    py.axis([15, -15, -15, 15])

    idx = (np.where( (lis.mag < magCut) &
                     (x > xim.min()) & (x < xim.max()) &
                     (y > yim.min()) & (y < yim.max()) ))[0]

    py.plot(x[idx], y[idx], 'r+', color='orange')
    for ii in idx:
        py.text(x[ii], y[ii], lis.name[ii], color='orange', fontsize=10)


def compareLabelLists(labelFile1, labelFile2, magCut=18):
    t = 2006.580

    ## Read in star labels
    tab1 = asciidata.open(labelFile1)
    name1 = [tab1[0][ss].strip() for ss in range(tab1.nrows)]
    mag1 = tab1[1].tonumpy()
    x01 = tab1[2].tonumpy()
    y01 = tab1[3].tonumpy()
    vx1 = tab1[6].tonumpy()
    vy1 = tab1[7].tonumpy()
    t01 = tab1[10].tonumpy()
    x1 = x01 + vx1 * (t - t01) / 10**3
    y1 = y01 + vy1 * (t - t01) / 10**3

    tab2 = asciidata.open(labelFile2)
    name2 = [tab2[0][ss].strip() for ss in range(tab2.nrows)]
    mag2 = tab2[1].tonumpy()
    x02 = tab2[2].tonumpy()
    y02 = tab2[3].tonumpy()
    vx2 = tab2[6].tonumpy()
    vy2 = tab2[7].tonumpy()
    t02 = tab2[10].tonumpy()
    x2 = x02 + vx2 * (t - t02) / 10**3
    y2 = y02 + vy2 * (t - t02) / 10**3

    # Image
    im = pyfits.getdata('/u/ghezgroup/data/gc/06maylgs1/combo/mag06maylgs1_dp_msc_kp.fits')
    imgsize = (im.shape)[0]

    # pixel position (0,0) is at upper left
    xpix = np.arange(0, im.shape[0], dtype=float)
    ypix = np.arange(0, im.shape[1], dtype=float)

    sgra = [1422.6, 1543.8]
    scale_jpg = 0.00995
    xim = (xpix - sgra[0]) * scale_jpg * -1.0
    yim = (ypix - sgra[1]) * scale_jpg
    
    py.clf()
    py.grid(True)
    py.subplots_adjust(left=0.05, right=0.95, bottom=0.05, top=0.95)
    py.imshow(np.log10(im), extent=[xim[0], xim[-1], yim[0], yim[-1]],
              aspect='equal', vmin=1.9, vmax=6.0, cmap=py.cm.gray)
    py.xlabel('X Offset from Sgr A* (arcsec)')
    py.ylabel('Y Offset from Sgr A* (arcsec)')
    py.title('UCLA/Keck Galactic Center Group', fontsize=20, fontweight='bold')
    thePlot = py.gca()

    py.axis([15, -15, -15, 15])
    
    idx2 = np.where(mag2 < magCut)[0]
    py.plot(x2[idx2], y2[idx2], 'ro', color='cyan', mfc='none', mec='cyan')
    for ii in idx2:
        py.text(x2[ii], y2[ii], name2[ii], color='cyan', fontsize=10)

    idx1 = np.where(mag1 < magCut)[0]
    py.plot(x1[idx1], y1[idx1], 'ro', color='orange', mfc='none', mec='orange')
    for ii in idx1:
        py.text(x1[ii], y1[ii], name1[ii], color='orange', fontsize=10)



def printLabelInfo(alignRoot, polyRoot, starName):
    # Now read in the align stuff
    s = starset.StarSet(alignRoot)
    s.loadPolyfit(polyRoot, accel=0)
    mag = s.getArray('mag')
    name = np.array(s.getArray('name'))
    x0 = s.getArray('fitXv.p') * -1.0
    y0 = s.getArray('fitYv.p')
    x0err = s.getArray('fitXv.perr')
    y0err = s.getArray('fitYv.perr')
    vx = s.getArray('fitXv.v') * -1.0
    vy = s.getArray('fitYv.v')
    vxerr = s.getArray('fitXv.verr')
    vyerr = s.getArray('fitYv.verr')
    t0x = s.getArray('fitXv.t0')
    t0y = s.getArray('fitYv.t0')
    r = np.hypot(x0, y0)

    # Fix x0err, y0err to some minimum value
    idx = np.where(x0err < 0.0001)[0]
    x0err[idx] = 0.0001
    idx = np.where(y0err < 0.0001)[0]
    y0err[idx] = 0.0001


    idx = np.where(name == starName)[0]

    if len(idx) > 0:
        rr = idx[0]
        print '%-11s  %4.1f    %6.3f  %6.3f  %7.4f %7.4f  %8.3f %8.3f  %7.3f  %7.3f   %8.3f    1   %5.3f' % \
              (name[rr], mag[rr], x0[rr], y0[rr], x0err[rr],
               y0err[rr], vx[rr]*10**3, vy[rr]*10**3,
               vxerr[rr]*10**3, vyerr[rr]*10**3, t0x[rr], r[rr])
    else:
        print 'Could not find star %s' % starName

