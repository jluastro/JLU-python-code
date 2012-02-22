import pylab as py
import numpy as np
import pyfits
import math

def w49_fields():
    """
    Map fields on the UKIDSS image.
    """
    img = pyfits.getdata('/u/jlu/data/w49/09jun26/raw/n1366.fits')
    # Rotate Image by -90 = 270 degrees (NTT image)
    #img = np.rot90(img, 1)
    #s0coords = np.array([478.0, 521.0])
    #scale = 0.29   # arcsec/pixel
    s0coords = np.array([591.0, 322.0])
    scale = 0.04   # arcsec/pixel
    
    xaxis = (np.arange(img.shape[0], dtype=float) - s0coords[0]) * scale * -1.0
    yaxis = (np.arange(img.shape[1], dtype=float) - s0coords[1]) * scale

    fieldCenters = []
    fieldCenters.append( {'name': 'f1', 'x': 4.5, 'y': -4.5} ) # f1
    fieldCenters.append( {'name': 'f2', 'x': 4.5, 'y': 4.5}  ) # f2
    fieldCenters.append( {'name': 'f3', 'x': -4.5, 'y': 4.5} ) # f3
    fieldCenters.append( {'name': 'f4', 'x': -4.5, 'y': -4.5}) # f4

    ##########
    # Plot
    ##########
    py.clf()
    py.imshow(img, extent=[xaxis[0], xaxis[-1], yaxis[0], yaxis[-1]],
              cmap=py.cm.YlOrBr_r, vmin=20, vmax=100)
    py.xlabel('R.A. Offset (arcsec)')
    py.ylabel('Dec. Offset (arcsec)')
    py.axis([10, -10, -10, 10])

    fig = py.gca()

    for f in fieldCenters:
        box = py.Rectangle([f['x']-5, f['y']-5], 10.0, 10.0, 
                           fill=False, edgecolor='lightgreen',
                                   linewidth=2)
        fig.add_patch(box)
        
        if (np.sign(f['x']) > 0):
            ha = 'left'
        else:
            ha = 'right'
            
        if (np.sign(f['y']) > 0):
            va = 'top'
        else:
            va = 'bottom'
               
        xText = f['x'] + (4.5 * np.sign(f['x']))
        yText = f['y'] + (4.5 * np.sign(f['y']))
        py.text(xText, yText, f['name'], fontsize=18, fontweight='bold', 
                color='lightgreen', horizontalalignment=ha, verticalalignment=va)
        py.savefig('/u/jlu/work/w49/maps/w49_fields.png')

def w49_f1_stars():
    """
    Plot Coo Star for W49, Field 1, SE

    W49 - Field 1

    """
    fitsfile = '/u/jlu/data/w49/09jun26/raw/n1370.fits'
    outfile = '/u/jlu/work/w49/maps/w49_f1_coo.png'
    coords = np.array([597.0, 241.0])   # s1 or f1_psf0
    title = 'W49\n Field 1 (f1), SE'

    map_stars(fitsfile, outfile, coords, title, vmin=250)

def w49_f1_psfstars():
    """
    Plot PSF Stars for W49 Field 1, SE

    W49 - Field 1

    """
    # Pixel positions pulled from combo image
    # Magnitudes pulled from UKIDSS table.

    fitsfile = '/u/jlu/data/w49/09jun26/combo/mag09jun26_w49_f1_kp.fits'
    outfile = '/u/jlu/work/w49/maps/w49_f1_psf.png'
    title = 'W49\n Field 1 (f1), SE'

    psfStars = []
    psfStars.append({'name':'f1_psf0', 'xpix':638.49, 'ypix':264.11, 'mag':0.00})
    psfStars.append({'name':'f1_psf1', 'xpix':132.11, 'ypix':842.85, 'mag':0.00})
    psfStars.append({'name':'f1_psf2', 'xpix':810.20, 'ypix':968.20, 'mag':0.00})
    psfStars.append({'name':'f1_psf3', 'xpix':443.81, 'ypix':608.27, 'mag':0.00})

    map_psf_stars(fitsfile, outfile, psfStars, title)

def w49_f2_stars():
    """
    Plot Coo Star for W49 Field 2, NE

    W49 - Field 2

    """
    fitsfile = '/u/jlu/data/w49/09jun26/raw/n1404.fits'
    outfile = '/u/jlu/work/w49/maps/w49_f2_coo.png'
    coords = np.array([765.0, 818.0]) # s4 or f2_psf0
    title = 'W49\n Field 2 (f2), NE'

    map_stars(fitsfile, outfile, coords, title, vmin=250)

def w49_f2_psfstars():
    """
    Plot PSF Stars for W49 Field 2, NE

    W49 - Field 2

    """
    # Pixel positions pulled from combo image
    # Magnitudes pulled from UKIDSS table.

    fitsfile = '/u/jlu/data/w49/09jun26/combo/mag09jun26_w49_f2_kp.fits'
    outfile = '/u/jlu/work/w49/maps/w49_f2_psf.png'
    title = 'W49\n Field 2 (f2), NE'

    psfStars = []
    psfStars.append({'name':'f2_psf0', 'xpix':787.84, 'ypix':840.55, 'mag':0.0})
    psfStars.append({'name':'f2_psf1', 'xpix':901.56, 'ypix':703.38, 'mag':0.0})
    psfStars.append({'name':'f2_psf2', 'xpix':595.67, 'ypix':348.18, 'mag':0.0})
    psfStars.append({'name':'f2_psf3', 'xpix':483.16, 'ypix':945.94, 'mag':0.0})
    #psfStars.append({'name':'sec1', 'xpix':549.58, 'ypix':272.30, 'mag':0.00})

    map_psf_stars(fitsfile, outfile, psfStars, title)

def w49_f3_stars():
    """
    Plot Coo Star for W49 Field 3, NW

    W49 - Field 3

    """
    fitsfile = '/u/jlu/data/w49/09jun26/raw/n1437.fits'
    outfile = '/u/jlu/work/w49/maps/w49_f3_coo.png'
    coords = np.array([345.0, 589.0]) # s8 or f3_psf0
    title = 'W49\n Field 3 (f3), NW'

    map_stars(fitsfile, outfile, coords, title, vmin=250)


def w49_f3_psfstars():
    """
    Plot PSF Star for W49 Field 3, NW

    W49 - Field 3

    """
    # Pixel positions pulled from combo image
    # Magnitudes pulled from UKIDSS table.

    fitsfile = '/u/jlu/data/w49/09jun26/combo/mag09jun26_w49_f3_kp.fits'
    outfile = '/u/jlu/work/w49/maps/w49_f3_psf.png'
    title = 'W49\n Field 3 (f3), NW'

    psfStars = []
    psfStars.append({'name':'f3_psf0', 'xpix':365.32, 'ypix':609.59, 'mag':0.0})
    psfStars.append({'name':'f3_psf1', 'xpix':150.20, 'ypix':646.08, 'mag':0.0})
    psfStars.append({'name':'f3_psf2', 'xpix':115.13, 'ypix':931.98, 'mag':0.0})
    psfStars.append({'name':'f3_psf3', 'xpix':547.10, 'ypix':418.16, 'mag':0.0})

    map_psf_stars(fitsfile, outfile, psfStars, title)


def w49_f4_stars():
    """
    Plot Coo Star for W49 Field 4, SW

    W49 - Field 4

    """
    fitsfile = '/u/jlu/data/w49/09jun26/raw/n1470.fits'
    outfile = '/u/jlu/work/w49/maps/w49_f4_coo.png'
    coords = np.array([398.0, 595.0]) # s8 or f4_psf0
    title = 'W49\n Field 4 (f4), NW'

    map_stars(fitsfile, outfile, coords, title, vmin=300)


def w49_f4_psfstars():
    """
    Plot PSF Star for W49 Field 4, SW

    W49 - Field 4

    """
    # Pixel positions pulled from combo image
    # Magnitudes pulled from UKIDSS table.

    fitsfile = '/u/jlu/data/w49/09jun26/combo/mag09jun26_w49_f4_kp.fits'
    outfile = '/u/jlu/work/w49/maps/w49_f4_psf.png'
    title = 'W49\n Field 4 (f4), NW'

    psfStars = []
    psfStars.append({'name':'f4_psf0', 'xpix':446.49, 'ypix':629.52, 'mag':0.0})
    psfStars.append({'name':'f4_psf1', 'xpix':883.17, 'ypix':859.17, 'mag':0.0})
    psfStars.append({'name':'f4_psf2', 'xpix':728.22, 'ypix':793.35, 'mag':0.0})

    map_psf_stars(fitsfile, outfile, psfStars, title)


def tables_w49():
    """
    Print out the photo_calib.dat and label.dat information for
    w49. Currently, we just use the above listed psfStars. I manually
    found a common source between the SE, NE, and NW fields, and included
    the pixel offsets below to put everything into a common coordinate
    system. This is very crude for now, but proabably more accurate than
    using the UKIDSS data.
    """
    stars = []

    # Put in the stars from field 1 (f1)
    stars.append({'name':'s0', 'xpix':871.50, 'ypix':876.00, 'mag':10.86})
    stars.append({'name':'s1', 'xpix':469.70, 'ypix':768.26, 'mag':13.65})
    stars.append({'name':'s2', 'xpix':518.92, 'ypix':463.84, 'mag':13.94})
    stars.append({'name':'s3', 'xpix':621.21, 'ypix':677.23, 'mag':13.98})

    # Put in the stars from field 2 (f2)
    tmpstars = []
    tmpstars.append({'name':'s4', 'xpix':326.95, 'ypix':435.76, 'mag':13.50})
    tmpstars.append({'name':'s5', 'xpix':621.13, 'ypix':638.77, 'mag':13.96})
    tmpstars.append({'name':'s6', 'xpix':783.02, 'ypix':369.85, 'mag':14.38})
    tmpstars.append({'name':'s7', 'xpix':579.09, 'ypix':266.52, 'mag':14.62})

    f2_dx = 660.01 - 919.92
    f2_dy = 992.29 - 275.83
    for tmp in tmpstars:
        tmp['xpix'] += f2_dx
        tmp['ypix'] += f2_dy

    stars += tmpstars

    # Put in the stars from field 3 (f3)
    tmpstars = []
    tmpstars.append({'name':'s8', 'xpix':375.62, 'ypix':509.89, 'mag':14.05})
    tmpstars.append({'name':'s9', 'xpix':613.46, 'ypix':566.33, 'mag':14.27})
    tmpstars.append({'name':'s10', 'xpix':939.34, 'ypix':384.77, 'mag':14.00})
    tmpstars.append({'name':'s11', 'xpix':339.19, 'ypix':244.12, 'mag':13.34})

    f3_dx = 660.01 - 84.71
    f3_dy = 992.29 - 38.00
    for tmp in tmpstars:
        tmp['xpix'] += f3_dx
        tmp['ypix'] += f3_dy
        print tmp['name'], tmp['xpix'], tmp['ypix']

    stars += tmpstars

    # Convert everything into arcsec offset from the BRIGHT star
    scale = 0.00995
    for star in stars:
        star['x'] = (star['xpix'] - stars[0]['xpix']) * scale * -1.0
        star['y'] = (star['ypix'] - stars[0]['ypix']) * scale

    # Print out photo_calib.dat file
    print '\n*** w49_photo_calib.dat ***'
    print ''
    print '## Columns: Format of this header is hardcoded for read in by calibrate.py'
    print '## Field separator in each column is "--". Default calibrators are listed after'
    print '## each magnitude column header entry.'
    print '# 1 -- Star Name'
    print '# 2 -- X position (arcsec, increasing to the East)'
    print '# 3 -- Y position (arcsec)'
    print '# 4 -- Variable? flag'
    print '# 5 -- K band (UKIDSS DR 3) -- %s' % (','.join([star['name'] for star in stars]))
    for star in stars:
        print '%-10s  %7.3f  %7.3f  0  %5.2f' % \
            (star['name'], star['x'], star['y'], star['mag'])

    # Print out label.dat file
    print '\n*** w49_label.dat ***'
    print ''
    print '%-10s  %5s  %7s  %7s  %6s  %6s  %8s  %8s  %8s  %8s  %8s  %4s  %6s' % \
        ('#Name', 'K', 'x', 'y', 'xerr', 'yerr', 
         'vx', 'vy', 'vxerr', 'vyerr',
         't0', 'use?', 'r2d')
    print '%-10s  %5s  %7s  %7s  %6s  %6s  %8s  %8s  %8s  %8s  %8s  %4s  %6s' % \
        ('#()', '(mag)', '(asec)', '(asec)', '(asec)', '(asec)', 
         '(mas/yr)', '(mas/yr)', '(mas/yr)', '(mas/yr)',
         '(year)', '()', '(asec)')

    for star in stars:
        print '%-10s  %5.2f  %7.3f  %7.3f  %6.3f  %6.3f  %8.3f  %8.3f  %8.3f  %8.3f  %8.3f  %4d  %6.3f' % \
            (star['name'], star['mag'], star['x'], star['y'], 0.0, 0.0,
             0.0, 0.0, 0.0, 0.0,
             2009.0, 1, math.hypot(star['x'], star['y']))



def map_stars(fitsfile, outfile, coords, title, vmin=100, vmax=1500):
    """
    Plot 'first' image for each field in W49 G48.9-0.3. 
    """
    datadir = '/u/jlu/data/w49/'
    outdir = '/u/jlu/work/w49/maps/'

    ##########
    # W49 - Field 1
    ##########
    f_fits = pyfits.getdata(fitsfile)
    f_coo = coords

    py.clf()
    py.imshow(np.log10(f_fits), cmap=py.cm.YlOrBr_r, 
              vmin=math.log10(vmin), vmax=math.log10(vmax))
    py.xlabel('X Axis (NIRC2 pixels)')
    py.ylabel('Y Axis (NIRC2 pixels)')

    fig = py.gca()

    py.text(512, 1024, title, weight='bold',
            horizontalalignment='center', verticalalignment='bottom')
    
    # Encircle and Label coo star
    cooRadius = 40
    cooLabel = '[%d, %d]' % (f_coo[0], f_coo[1])
    cooStar = py.Circle(f_coo, radius=cooRadius, fill=False,
                        ec='black', linewidth=3)
    fig.add_patch(cooStar)
    py.text(f_coo[0], f_coo[1]-(1.2*cooRadius), cooLabel,
            weight='bold', color='black',
            horizontalalignment='center', verticalalignment='top')


    # Draw East/North arrows
    arr0 = [170, 30]
    arrLen = 100
    arrColor = 'black'
    north = py.Arrow(arr0[0], arr0[1], 0, arrLen, width=0.3*arrLen, 
                     fill=True, fc=arrColor, ec=arrColor)
    east = py.Arrow(arr0[0], arr0[1], -arrLen, 0, width=0.3*arrLen, 
                    fill=True, fc=arrColor, ec=arrColor)
    
    fig.add_patch(north)
    fig.add_patch(east)

    py.text(arr0[0], arr0[1]+(1.1*arrLen), 'N', color=arrColor, weight='bold',
            horizontalalignment='center', verticalalignment='bottom')
    py.text(arr0[0]-(1.1*arrLen), arr0[1], 'E', color=arrColor, weight='bold',
            horizontalalignment='right', verticalalignment='center')
    
    # Save
    py.savefig(outfile)
    

def map_psf_stars(fitsfile, outfile, stars, title, vmin=10, vmax=1500):
    """
    Plot combo image for each field in W49 G48.9-0.3 and overlay the PSF
    stars.
    """
    datadir = '/u/jlu/data/w49/'
    outdir = '/u/jlu/work/w49/maps/'

    ##########
    # Load the fits file
    ##########
    img = pyfits.getdata(fitsfile)

    ##########
    # Calculate the coordinates in arcsec
    ##########
    scale = 0.00995

    # Image extent
    xaxis = (np.arange(img.shape[0]) - stars[0]['xpix']) * scale * -1.0
    yaxis = (np.arange(img.shape[1]) - stars[0]['ypix']) * scale

    for star in stars:
        # RA and Dec Offsets (in arcsec)
        star['x'] = (star['xpix'] - stars[0]['xpix']) * scale * -1.0
        star['y'] = (star['ypix'] - stars[0]['ypix']) * scale
    
    ##########
    # Plot Image
    ##########
    py.rc('font', **{'weight':'bold'})
    py.clf()
    py.imshow(np.log10(img), cmap=py.cm.YlOrBr_r, 
              vmin=math.log10(vmin), vmax=math.log10(vmax),
              extent=[xaxis[0], xaxis[-1], yaxis[0], yaxis[-1]])

    py.xlabel('R.A. Offset (arcsec)', fontweight='bold')
    py.ylabel('Dec. Offset (arcsec)', fontweight='bold')

    py.setp(py.getp(py.gca(), 'xticklabels'), fontweight='bold')
    py.setp(py.getp(py.gca(), 'yticklabels'), fontweight='bold')

    py.title(title, fontweight='bold')
    
    fig = py.gca()

    # Draw East/North arrows
    arr0 = np.array([xaxis[0]-1.7, yaxis[0]+0.3])
    arrLen = 1.0
    arrColor = 'black'
    north = py.Arrow(arr0[0], arr0[1], 0, arrLen, width=0.3*arrLen, 
                     fill=True, fc=arrColor, ec=arrColor)
    east = py.Arrow(arr0[0], arr0[1], arrLen, 0, width=0.3*arrLen, 
                    fill=True, fc=arrColor, ec=arrColor)
    
    fig.add_patch(north)
    fig.add_patch(east)

    py.text(arr0[0], arr0[1]+(1.1*arrLen), 'N', color=arrColor, weight='bold',
            horizontalalignment='center', verticalalignment='bottom')
    py.text(arr0[0]+(1.1*arrLen), arr0[1], 'E', color=arrColor, weight='bold',
            horizontalalignment='right', verticalalignment='center')

    ##########
    # Plot each PSF star.
    ##########
    for star in stars:
        # Encircle and Label coo star
        psfRadius = 0.2
        psfLabel = '%s\n[%6.3f, %6.3f]' % (star['name'], star['x'], star['y'])
        psfStar = py.Circle([star['x'], star['y']], 
                            radius=psfRadius, fill=False,
                            ec='black', linewidth=3)
        fig.add_patch(psfStar)
        py.text(star['x'], star['y']-(1.2*psfRadius), psfLabel,
                fontweight='bold', color='black', fontsize=10,
                horizontalalignment='center', verticalalignment='top')
    
    # Save
    py.savefig(outfile)

    ##########
    # Generate a PSF star list for Starfinder... only print out to the screen.
    ##########
    print '########'
    print '#'
    print '# PRINTING PSF starlist for input to Starfinder.'
    print '#  -- paste everything below to a file.'
    print '#  -- example file is /u/jlu/code/idl/w49/psftables/w49_f1_psf.list'
    print '#'
    print '########'
    print '\n'

    print '# This file should contain sources we absolutely know exist '
    print '# (with names). Additionally, all sources should have velocities '
    print '# so that they can be found across multiple epochs. The Filt column '
    print '# shows what filters this star should NOT be used in '
    print '# (K=speckle, Kp,Lp,Ms = NIRC2). The PSF column specifies whether '
    print '# this is a PSF star (1) or just a secondary source (0).'
    print '#%-12s  %-4s  %-7s  %-7s  %-7s  %-7s  %-8s  %-4s  %-4s' % \
        ('Name', 'Mag', 'Xarc', 'Yarc', 'Vx', 'Vy', 't0', 'Filt', 'PSF?')
    for star in stars:
        print '%-13s  %4.1f  %7.3f  %7.3f  %7.3f  %7.3f  %8.3f  %-4s  %1d' % \
            (star['name'], star['mag'], star['x'], star['y'],
             0.0, 0.0, 2009.0, '-', 1)

    
