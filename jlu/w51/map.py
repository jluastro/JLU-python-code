import pylab as py
import numpy as np
import pyfits
import math
import asciidata
from scipy.ndimage import interpolation as interp
from jlu.util import img_scale
from gcwork import starTables
from gcreduce import gcutil

def w51a_fields():
    """
    Map fields on the UKIDSS image.
    """
    img = pyfits.getdata('/u/jlu/data/w51/ukidss/g48.9-0.3/ukidss_rgb_r_k.fits')

    s0coords = np.array([570.0, 598.0])
    scale = 0.2   # arcsec/pixel
    
    xaxis = (np.arange(img.shape[0], dtype=float) - s0coords[0]) * scale * -1.0
    yaxis = (np.arange(img.shape[1], dtype=float) - s0coords[1]) * scale

    fieldCenters = []
    fieldCenters.append( {'name': 'f1', 'x': 3.341, 'y': -3.351} ) # f1
    fieldCenters.append( {'name': 'f2', 'x': 5.855, 'y': 3.693}  ) # f2
    fieldCenters.append( {'name': 'f3', 'x': -2.611, 'y': 6.240} ) # f3
    fieldCenters.append( {'name': 'f4', 'x': -7.000, 'y': -2.500}) # f4

    ##########
    # Plot
    ##########
    py.clf()
    py.imshow(img, extent=[xaxis[0], xaxis[-1], yaxis[0], yaxis[-1]],
              cmap=py.cm.YlOrBr_r, vmin=500, vmax=50000)
    py.xlabel('R.A. Offset (arcsec)')
    py.ylabel('Dec. Offset (arcsec)')
    py.axis([20, -20, -20, 20])

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
        py.savefig('/u/jlu/work/w51/maps/w51a_fields.png')

def w51a_f1_stars():
    """
    Plot Coo Star for W51 G48.9-0.3 South, Field 1, SE

    W51a - Field 1

    """
    fitsfile = '/u/jlu/data/w51/09jun10/raw/n0174.fits'
    outfile = '/u/jlu/work/w51/maps/w51a_f1_coo.png'
    coords = np.array([446.0, 741.0])   # s1 or f1_psf0
    title = 'W51 G48.9-0.3 South (a.k.a. W51a)\n Field 1 (f1), SE'

    map_stars(fitsfile, outfile, coords, title)

def w51a_f1_psfstars():
    """
    Plot PSF Stars for W51 G48.9-0.3 South, Field 1, SE

    W51a - Field 1

    """
    # Pixel positions pulled from combo image
    # Magnitudes pulled from UKIDSS table.

    fitsfile = '/u/jlu/data/w51/09jun10/combo/mag09jun10_w51a_f1_kp.fits'
    outfile = '/u/jlu/work/w51/maps/w51a_f1_psf.png'
    title = 'W51 G48.9-0.3 South (a.k.a. W51a)\n Field 1 (f1), SE'

    psfStars = []
    psfStars.append({'name':'psf0', 'xpix':469.70, 'ypix':768.26, 'mag':13.65})
    psfStars.append({'name':'psf1', 'xpix':871.50, 'ypix':876.00, 'mag':10.86})
    psfStars.append({'name':'psf2', 'xpix':518.92, 'ypix':463.84, 'mag':13.94})
    psfStars.append({'name':'psf3', 'xpix':621.21, 'ypix':677.23, 'mag':13.98})

    map_psf_stars(fitsfile, outfile, psfStars, title)

def w51a_f2_stars():
    """
    Plot Coo Star for W51 G48.9-0.3 South, Field 2, NE

    W51a - Field 2

    """
    fitsfile = '/u/jlu/data/w51/09jun10/raw/n0225.fits'
    outfile = '/u/jlu/work/w51/maps/w51a_f2_coo.png'
    coords = np.array([296.0, 417.0]) # s4 or f2_psf0
    title = 'W51 G48.9-0.3 South (a.k.a. W51a)\n Field 2 (f2), NE'

    map_stars(fitsfile, outfile, coords, title, vmin=150)

def w51a_f2_psfstars():
    """
    Plot PSF Stars for W51 G48.9-0.3 South, Field 2, NE

    W51a - Field 2

    """
    # Pixel positions pulled from combo image
    # Magnitudes pulled from UKIDSS table.

    fitsfile = '/u/jlu/data/w51/09jun10/combo/mag09jun10_w51a_f2_kp.fits'
    outfile = '/u/jlu/work/w51/maps/w51a_f2_psf.png'
    title = 'W51 G48.9-0.3 South (a.k.a. W51a)\n Field 2 (f2), NE'

    psfStars = []
    psfStars.append({'name':'psf0', 'xpix':326.95, 'ypix':435.76, 'mag':13.50})
    psfStars.append({'name':'psf1', 'xpix':621.13, 'ypix':638.77, 'mag':13.96})
    psfStars.append({'name':'psf2', 'xpix':783.02, 'ypix':369.85, 'mag':14.38})
    psfStars.append({'name':'psf3', 'xpix':579.09, 'ypix':266.52, 'mag':14.62})
    psfStars.append({'name':'sec1', 'xpix':549.58, 'ypix':272.30, 'mag': 0.00})

    map_psf_stars(fitsfile, outfile, psfStars, title)

def w51a_f3_stars():
    """
    Plot Coo Star for W51 G48.9-0.3 South, Field 3, NW

    W51a - Field 3

    """
    fitsfile = '/u/jlu/data/w51/09jun10/raw/n0275.fits'
    outfile = '/u/jlu/work/w51/maps/w51a_f3_coo.png'
    coords = np.array([329.0, 473.0]) # s8 or f3_psf0
    title = 'W51 G48.9-0.3 South (a.k.a. W51a)\n Field 3 (f3), NW'

    map_stars(fitsfile, outfile, coords, title, vmin=150)


def w51a_f3_psfstars():
    """
    Plot PSF Star for W51 G48.9-0.3 South, Field 3, NW

    W51a - Field 3

    """
    # Pixel positions pulled from combo image
    # Magnitudes pulled from UKIDSS table.

    fitsfile = '/u/jlu/data/w51/09jun10/combo/mag09jun10_w51a_f3_kp.fits'
    outfile = '/u/jlu/work/w51/maps/w51a_f3_psf.png'
    title = 'W51 G48.9-0.3 South (a.k.a. W51a)\n Field 3 (f3), NW'

    psfStars = []
    psfStars.append({'name':'psf0', 'xpix':375.62, 'ypix':509.89, 'mag':14.05})
    psfStars.append({'name':'psf1', 'xpix':613.46, 'ypix':566.33, 'mag':14.27})
    psfStars.append({'name':'psf2', 'xpix':939.34, 'ypix':384.77, 'mag':14.00})
    psfStars.append({'name':'psf3', 'xpix':339.19, 'ypix':244.12, 'mag':13.34})

    map_psf_stars(fitsfile, outfile, psfStars, title)


def w51a_f4_stars():
    """
    Plot Coo Star for W51 G48.9-0.3 South, Field 4, SW

    W51a - Field 4

    """
    fitsfile = '/u/jlu/data/w51/09jun26/raw/n0064.fits'
    outfile = '/u/jlu/work/w51/maps/w51a_f4_coo.png'
    coords = np.array([456., 798.]) # f4_psf0
    title = 'W51 G48.9-0.3 South (a.k.a. W51a)\n Field 4 (f4), SW'

    map_stars(fitsfile, outfile, coords, title, vmin=150)


def w51a_f4_psfstars():
    """
    Plot PSF Star for W51 G48.9-0.3 South, Field 4, SW

    W51a - Field 4

    """
    # Pixel positions pulled from combo image
    # Magnitudes pulled from UKIDSS table.

    fitsfile = '/u/jlu/data/w51/09jun26/combo/mag09jun26_w51a_f4_kp.fits'
    outfile = '/u/jlu/work/w51/maps/w51a_f4_psf.png'
    title = 'W51 G48.9-0.3 South (a.k.a. W51a)\n Field 4 (f4), SW'

    psfStars = []
    psfStars.append({'name':'psf0', 'xpix':478.35, 'ypix':824.13, 'mag':13.11})
    psfStars.append({'name':'psf1', 'xpix':777.35, 'ypix':836.95, 'mag':12.67})
    psfStars.append({'name':'psf2', 'xpix':323.90, 'ypix':352.98, 'mag':12.82})
    psfStars.append({'name':'psf3', 'xpix':151.01, 'ypix':549.15, 'mag':12.28})
    psfStars.append({'name':'psf4', 'xpix':183.56, 'ypix':347.95, 'mag':13.52})
    psfStars.append({'name':'psf5', 'xpix':676.77, 'ypix':747.14, 'mag':14.21})

    map_psf_stars(fitsfile, outfile, psfStars, title)


def tables_w51a():
    """
    Print out the photo_calib.dat and label.dat information for
    w51a. Currently, we just use the above listed psfStars. I manually
    found a common source between the SE, NE, and NW fields, and included
    the pixel offsets below to put everything into a common coordinate
    system. This is very crude for now, but proabably more accurate than
    using the UKIDSS data.
    """
    stars = []

    # Put in the stars from field 1 (f1)
    stars.append({'name':'f1_psf1', 'xpix':871.50, 'ypix':876.00, 'mag':10.86})
    stars.append({'name':'f1_psf0', 'xpix':469.70, 'ypix':768.26, 'mag':13.65})
    stars.append({'name':'f1_psf2', 'xpix':518.92, 'ypix':463.84, 'mag':13.94})
    stars.append({'name':'f1_psf3', 'xpix':621.21, 'ypix':677.23, 'mag':13.98})

    # Put in the stars from field 2 (f2)
    tstars = []
    tstars.append({'name':'f2_psf0', 'xpix':326.95, 'ypix':435.76, 'mag':13.50})
    tstars.append({'name':'f2_psf1', 'xpix':621.13, 'ypix':638.77, 'mag':13.96})
    tstars.append({'name':'f2_psf2', 'xpix':783.02, 'ypix':369.85, 'mag':14.38})
    tstars.append({'name':'f2_psf3', 'xpix':579.09, 'ypix':266.52, 'mag':14.62})

    f2_dx = 660.01 - 919.92
    f2_dy = 992.29 - 275.83
    for tmp in tstars:
        tmp['xpix'] += f2_dx
        tmp['ypix'] += f2_dy

    stars += tstars

    # Put in the stars from field 3 (f3)
    tstars = []
    tstars.append({'name':'f3_psf0', 'xpix':375.62, 'ypix':509.89, 'mag':14.05})
    tstars.append({'name':'f3_psf1', 'xpix':613.46, 'ypix':566.33, 'mag':14.27})
    tstars.append({'name':'f3_psf2', 'xpix':939.34, 'ypix':384.77, 'mag':14.00})
    tstars.append({'name':'f3_psf3', 'xpix':339.19, 'ypix':244.12, 'mag':13.34})

    f3_dx = 660.01 - 84.71
    f3_dy = 992.29 - 38.00
    for tmp in tstars:
        tmp['xpix'] += f3_dx
        tmp['ypix'] += f3_dy

    stars += tstars

    # Put in stars from field 4 (f4)
    tstars = []
    tstars.append({'name':'f4_psf0', 'xpix':478.35, 'ypix':824.13, 'mag':13.11})
    tstars.append({'name':'f4_psf1', 'xpix':777.35, 'ypix':836.95, 'mag':12.67})
    tstars.append({'name':'f4_psf2', 'xpix':323.90, 'ypix':352.98, 'mag':12.82})
    tstars.append({'name':'f4_psf3', 'xpix':151.01, 'ypix':549.15, 'mag':12.28})
    tstars.append({'name':'f4_psf4', 'xpix':183.56, 'ypix':347.95, 'mag':13.52})
    tstars.append({'name':'f4_psf5', 'xpix':676.77, 'ypix':747.14, 'mag':14.21})

    f4_dx = 871.50 + (156.0 * 4.02) - 478.35 # measured from wide camera image
    f4_dy = 876.00 + (8.0 * 4.02) - 824.13
    for tmp in tstars:
        tmp['xpix'] += f4_dx
        tmp['ypix'] += f4_dy
        print tmp['name'], tmp['xpix'], tmp['ypix']

    stars += tstars

    # Convert everything into arcsec offset from the BRIGHT star
    scale = 0.00995
    for star in stars:
        star['x'] = (star['xpix'] - stars[0]['xpix']) * scale * -1.0
        star['y'] = (star['ypix'] - stars[0]['ypix']) * scale

    # Print out photo_calib.dat file
    print '\n*** w51a_photo_calib.dat ***'
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
    print '\n*** w51_label.dat ***'
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
    Plot 'first' image for each field in W51 G48.9-0.3. 
    """
    datadir = '/u/jlu/data/w51/'
    outdir = '/u/jlu/work/w51/maps/'

    ##########
    # W51a - Field 1
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
    Plot combo image for each field in W51 G48.9-0.3 and overlay the PSF
    stars.
    """
    datadir = '/u/jlu/data/w51/'
    outdir = '/u/jlu/work/w51/maps/'

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
    print '#  -- example file is /u/jlu/code/idl/w51/psftables/w51a_f1_psf.list'
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

    

def plot_wide_labeled(labels='/u/jlu/data/w51/source_list/w51a_label.dat',
                      markCoordRefStars=False):
    lab = starTables.Labels(labelFile=labels)

    t = 2009.580

    xarc = lab.x + lab.vx * (t - lab.t0) / 10**3
    yarc = lab.y + lab.vy * (t - lab.t0) / 10**3

    vxarc = lab.vx
    vyarc = lab.vy

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
    im = pyfits.getdata('/u/jlu/data/w51/09jun26/combo/mag09jun26_w51a_wide_h.fits')
    imgsize = (im.shape)[0]

    # pixel position (0,0) is at upper left
    xpix = np.arange(0, im.shape[0], dtype=float)
    ypix = np.arange(0, im.shape[1], dtype=float)

    origin = [607.0, 749.0]
    scale_jpg = 0.04
    xim = (xpix - origin[0]) * scale_jpg * -1.0
    yim = (ypix - origin[1]) * scale_jpg
    
    py.clf()
    py.grid(True)
    py.subplots_adjust(left=0.05, right=0.95, bottom=0.05, top=0.95)
    py.imshow(np.log10(im), extent=[xim[0], xim[-1], yim[0], yim[-1]],
              aspect='equal', vmin=0.5, vmax=5, cmap=py.cm.gray)
    py.xlabel('X Offset (arcsec)')
    py.ylabel('Y Offset (arcsec)')
    py.title('J. R. Lu', fontsize=20, fontweight='bold')
    thePlot = py.gca()

    
    idx = (np.where( #(mag < 18) &
                     (x > xim.min()) & (x < xim.max()) &
                     (y > yim.min()) & (y < yim.max()) ))[0]

    py.plot(x[idx], y[idx], 'ro', color='orange', markerfacecolor='None',
            markeredgecolor='orange')

    if markCoordRefStars:
        mdx = (np.where((x > xim.min()) & (x < xim.max()) &
                        (y > yim.min()) & (y < yim.max()) &
                        (lab.useToAlign != '0')))[0]
        py.plot(x[mdx], y[mdx], 'ro', color='cyan', markerfacecolor='None',
                markeredgecolor='green')


    for ii in idx:
        if markCoordRefStars and ii in mdx:
            color = 'cyan'
        else:
            color = 'orange'
        py.text(x[ii], y[ii], lab.name[ii], color=color, fontsize=10)

    py.axis([15, -15, -15, 15])
    py.savefig('w51a_wide_labeled.png', dpi=300)


def mosaic3color():
    """
    Make a 3 color mosaic of our NIRC2 data on W51.
    """
    hepochs = ['09jun26', '09jun26', '09jun26', '09jun26']
    kepochs = ['09jun10', '09jun10', '09jun10', '09jun26']
    lepochs = ['09jun26', '09jun26', '09jun26', '09jun26']

    cooStarsH = ['f1_psf0', 'f2_psf0', 'f3_psf0', 'f4_psf0']
    cooStarsK = ['f1_psf0', 'f2_psf0', 'f3_psf0', 'f4_psf0']
    cooStarsL = ['f1_psf1', 'f2_psf0', 'f3_psf2', 'f4_psf1']

    cooStarsH = ['E4-1', 'E8-1', 'N5-1', 'W6-2']
    cooStarsK = ['E4-1', 'E8-1', 'N5-1', 'W6-2']
    cooStarsL = ['S0-1', 'E8-1', 'W7-1', 'W9-1']

    scaleMinH = [0, 0, 0, 0]
    scaleMinK = [0, 0, 0, 0]
    scaleMinL = [1000, 1100, 1200, 1250]

    scaleMaxH = [6000, 6000, 5000, 6000]
    scaleMaxK = [5500, 5500, 5500, 4500]
#     scaleMaxL = [1600, 1300, 1400, 1600]
    scaleMaxL = [2000, 2000, 2000, 2000]

    
    img = np.zeros((2400, 2400, 3), dtype=float)
    origin = np.array([1200.0, 1200.0])


    labelFile = '/u/jlu/data/w51/source_list/w51a_label.dat'
    labels = starTables.Labels(labelFile=labelFile)

    dataRoot = '/u/jlu/data/w51/'

    py.clf()
    foo = range(len(hepochs))
    for ii in foo[::-1]:
#     for ii in range(1):
        rootH = '%s/%s/combo/mag%s_w51a_f%d_h' % \
            (dataRoot, hepochs[ii], hepochs[ii], ii+1)
        rootK = '%s/%s/combo/mag%s_w51a_f%d_kp' % \
            (dataRoot, kepochs[ii], kepochs[ii], ii+1)
        rootL = '%s/%s/combo/mag%s_w51a_f%d_lp' % \
            (dataRoot, lepochs[ii], lepochs[ii], ii+1)
        
        # Load up the images
        h = pyfits.getdata(rootH + '.fits')
        k = pyfits.getdata(rootK + '.fits')
        l = pyfits.getdata(rootL + '.fits')

        # Make the arrays into the largest size.
        h_new = np.zeros((img.shape[0], img.shape[1]), dtype=float)
        k_new = np.zeros((img.shape[0], img.shape[1]), dtype=float)
        l_new = np.zeros((img.shape[0], img.shape[1]), dtype=float)

        h_new[0:h.shape[0], 0:h.shape[1]] = h
        k_new[0:k.shape[0], 0:k.shape[1]] = k
        l_new[0:l.shape[0], 0:l.shape[1]] = l

        # Load up the coo stars
        tmpH = open(rootH + '.coo').readline().split()
        cooH = np.array([float(tmpH[0]), float(tmpH[1])])
        
        tmpK = open(rootK + '.coo').readline().split()
        cooK = np.array([float(tmpK[0]), float(tmpK[1])])

        tmpL = open(rootL + '.coo').readline().split()
        cooL = np.array([float(tmpL[0]), float(tmpL[1])])
        
        # Get the coordinates of each coo star in arcsec.
        idxH = np.where(labels.name == cooStarsH[ii])[0][0]
        idxK = np.where(labels.name == cooStarsK[ii])[0][0]
        idxL = np.where(labels.name == cooStarsL[ii])[0][0]

        asecH = np.array([labels.x[idxH], labels.y[idxH]])
        asecK = np.array([labels.x[idxK], labels.y[idxK]])
        asecL = np.array([labels.x[idxL], labels.y[idxL]])

        scale = np.array([-0.00995, 0.00995])

        # Now figure out the necessary shifts
        originH = cooH - asecH/scale
        originK = cooK - asecK/scale
        originL = cooL - asecL/scale

        # Shift the J and H images to be lined up with K-band
        shiftL = origin - originL
        shiftK = origin - originK
        shiftH = origin - originH
        l = interp.shift(l_new, shiftL[::-1])
        k = interp.shift(k_new, shiftK[::-1])
        h = interp.shift(h_new, shiftH[::-1])
        print shiftH
        print shiftL

        xx, yy = np.meshgrid(np.arange(img.shape[0]), np.arange(img.shape[1]))

        idx = np.where((h >= 1) & (k >= 1) & (l >= 1))

        # Trim off the bottom 10 rows where there is data
        ymin = yy[idx[0], idx[1]].min()
        ydx = np.where(yy[idx[0],idx[1]] > (ymin+10))[0]
        idx = (idx[0][ydx], idx[1][ydx])

        img[idx[0],idx[1],0] = img_scale.sqrt(l[idx[0],idx[1]], 
                                              scale_min=scaleMinL[ii], 
                                              scale_max=scaleMaxL[ii])
        img[idx[0],idx[1],1] = img_scale.sqrt(k[idx[0], idx[1]], 
                                              scale_min=scaleMinK[ii], 
                                              scale_max=scaleMaxK[ii])
        img[idx[0],idx[1],2] = img_scale.sqrt(h[idx[0], idx[1]], 
                                              scale_min=scaleMinH[ii], 
                                              scale_max=scaleMaxH[ii])

        # Defin the axes
        xaxis = np.arange(-0.5, img.shape[1]+0.5, 1)
        xaxis = ((xaxis - origin[0]) * scale[0])
        yaxis = np.arange(-0.5, img.shape[0]+0.5, 1)
        yaxis = ((yaxis - origin[1]) * scale[1])
        extent = [xaxis[0], xaxis[-1], yaxis[0], yaxis[-1]]

        py.imshow(img, extent=extent)
        py.axis('equal')
        foo = raw_input('Continue?')

    py.axis([7, -7, -7, 7])
    py.savefig('/u/jlu/work/w51/maps/w51a_3color.png')

def mosaic3color_ukidss():
    """
    Make a 3 color mosaic of our NIRC2 data on W51.
    """
    root = '/u/jlu/data/w51/ukidss/g48.9-0.3/ukidss_rgb'
    j = pyfits.getdata(root + '_b_j.fits')
    h = pyfits.getdata(root + '_g_h.fits')
    k = pyfits.getdata(root + '_r_k.fits')

    scale = [-0.2, 0.2]

    img = np.zeros((j.shape[0], j.shape[1], 3), dtype=float)

    img[:,:,0] = img_scale.sqrt(k, scale_min=0, scale_max=10000)
    img[:,:,1] = img_scale.sqrt(h, scale_min=0, scale_max=10000) 
    img[:,:,2] = img_scale.sqrt(j, scale_min=0, scale_max=10000)
    print img[:,:,0].min(), img[:,:,0].max()
    print img[:,:,1].min(), img[:,:,1].max()
    print img[:,:,2].min(), img[:,:,2].max()

    origin = [j.shape[1]/2.0, j.shape[0]/2.0]
    print origin

    # Define the axes
    xaxis = np.arange(-0.5, img.shape[1]+0.5, 1)
    xaxis = ((xaxis - origin[0]) * scale[0])
    yaxis = np.arange(-0.5, img.shape[0]+0.5, 1)
    yaxis = ((yaxis - origin[1]) * scale[1])
    extent = [xaxis[0], xaxis[-1], yaxis[0], yaxis[-1]]

    py.clf()
    py.imshow(img, extent=extent)
    py.axis('equal')
    py.axis([60, -40, -30, 70])
    py.xlabel('R.A. Offset (arcsec)')
    py.ylabel('Dec. Offset (arcsec)')
    py.title('UKIDSS JHK')

    py.savefig('/u/jlu/work/w51/maps/w51a_3color_ukidss.png')

def mosaic_kp():
    """
    Make a mosaic of our NIRC2 data on W51, one per filter.
    """
    hepochs = ['09jun26', '09jun26', '09jun26', '09jun26']
    kepochs = ['09jun10', '09jun10', '09jun10', '09jun26']
    lepochs = ['09jun26', '09jun26', '09jun26', '09jun26']

    cooStarsH = ['f1_psf0', 'f2_psf0', 'f3_psf0', 'f4_psf0']
    cooStarsK = ['f1_psf0', 'f2_psf0', 'f3_psf0', 'f4_psf0']
    cooStarsL = ['f1_psf1', 'f2_psf0', 'f3_psf2', 'f4_psf1']

    cooStarsH = ['E4-1', 'E8-1', 'N5-1', 'W6-2']
    cooStarsK = ['E4-1', 'E8-1', 'N5-1', 'W6-2']
    cooStarsL = ['S0-1', 'E8-1', 'W7-1', 'W9-1']

    scaleMinH = [0, 0, 0, 0]
    scaleMinK = [0, 0, 0, 0]
    scaleMinL = [1000, 1100, 1200, 1250]

    scaleMaxH = [6000, 6000, 5000, 6000]
    scaleMaxK = [6500, 6500, 6500, 5500]
    scaleMaxL = [1600, 1300, 1400, 1600]

    
    img = np.zeros((2400, 2400, 3), dtype=float)
    imgH = np.zeros((2400, 2400), dtype=float)
    imgK = np.zeros((2400, 2400), dtype=float)
    imgL = np.zeros((2400, 2400), dtype=float)
    origin = np.array([1200.0, 1200.0])


    labelFile = '/u/jlu/data/w51/source_list/w51a_label.dat'
    labels = starTables.Labels(labelFile=labelFile)

    dataRoot = '/u/jlu/data/w51/'

    py.clf()
    foo = range(len(hepochs))
    for ii in foo[::-1]:
#     for ii in range(1):
        rootH = '%s/%s/combo/mag%s_w51a_f%d_h' % \
            (dataRoot, hepochs[ii], hepochs[ii], ii+1)
        rootK = '%s/%s/combo/mag%s_w51a_f%d_kp' % \
            (dataRoot, kepochs[ii], kepochs[ii], ii+1)
        rootL = '%s/%s/combo/mag%s_w51a_f%d_lp' % \
            (dataRoot, lepochs[ii], lepochs[ii], ii+1)
        
        # Load up the images
        h, hhdr = pyfits.getdata(rootH + '.fits', header=True)
        k, khdr = pyfits.getdata(rootK + '.fits', header=True)
        l, lhdr = pyfits.getdata(rootL + '.fits', header=True)

        hint = hhdr['ITIME'] * hhdr['COADDS']
        kint = khdr['ITIME'] * khdr['COADDS']
        lint = lhdr['ITIME'] * lhdr['COADDS']

        # Make the arrays into the largest size.
        h_new = np.zeros((img.shape[0], img.shape[1]), dtype=float)
        k_new = np.zeros((img.shape[0], img.shape[1]), dtype=float)
        l_new = np.zeros((img.shape[0], img.shape[1]), dtype=float)

        h_new[0:h.shape[0], 0:h.shape[1]] = h
        k_new[0:k.shape[0], 0:k.shape[1]] = k
        l_new[0:l.shape[0], 0:l.shape[1]] = l

        # Load up the coo stars
        tmpH = open(rootH + '.coo').readline().split()
        cooH = np.array([float(tmpH[0]), float(tmpH[1])])
        
        tmpK = open(rootK + '.coo').readline().split()
        cooK = np.array([float(tmpK[0]), float(tmpK[1])])

        tmpL = open(rootL + '.coo').readline().split()
        cooL = np.array([float(tmpL[0]), float(tmpL[1])])
        
        # Get the coordinates of each coo star in arcsec.
        idxH = np.where(labels.name == cooStarsH[ii])[0][0]
        idxK = np.where(labels.name == cooStarsK[ii])[0][0]
        idxL = np.where(labels.name == cooStarsL[ii])[0][0]

        asecH = np.array([labels.x[idxH], labels.y[idxH]])
        asecK = np.array([labels.x[idxK], labels.y[idxK]])
        asecL = np.array([labels.x[idxL], labels.y[idxL]])

        scale = np.array([-0.00995, 0.00995])

        # Now figure out the necessary shifts
        originH = cooH - asecH/scale
        originK = cooK - asecK/scale
        originL = cooL - asecL/scale

        # Shift the J and H images to be lined up with K-band
        shiftL = origin - originL
        shiftK = origin - originK
        shiftH = origin - originH
        l = interp.shift(l_new, shiftL[::-1])
        k = interp.shift(k_new, shiftK[::-1])
        h = interp.shift(h_new, shiftH[::-1])
        print shiftH
        print shiftL

        xx, yy = np.meshgrid(np.arange(img.shape[0]), np.arange(img.shape[1]))

        idx = np.where((h >= 1) & (k >= 1) & (l >= 1))

        # Trim off the bottom 10 rows where there is data
        ymin = yy[idx[0], idx[1]].min()
        ydx = np.where(yy[idx[0],idx[1]] > (ymin+10))[0]
        idx = (idx[0][ydx], idx[1][ydx])

        img[idx[0],idx[1],0] = img_scale.log(l[idx[0],idx[1]], 
                                             scale_min=scaleMinL[ii], 
                                             scale_max=scaleMaxL[ii])
        img[idx[0],idx[1],1] = img_scale.log(k[idx[0], idx[1]], 
                                             scale_min=scaleMinK[ii], 
                                             scale_max=scaleMaxK[ii])
        img[idx[0],idx[1],2] = img_scale.log(h[idx[0], idx[1]], 
                                             scale_min=scaleMinH[ii], 
                                             scale_max=scaleMaxH[ii])

        
        imgH[idx[0], idx[1]] = h[idx[0], idx[1]] / hint
        imgK[idx[0], idx[1]] = k[idx[0], idx[1]] / kint
        imgL[idx[0], idx[1]] = l[idx[0], idx[1]] / lint

        # Fix scaling of first image.
        if ii == 0:
            imgK[idx[0], idx[1]] -= 0.4



        # Save on memory
        l = None
        k = None
        h = None
        l_new = None
        k_new = None
        h_new = None

        # Define the axes
        xaxis = np.arange(-0.5, img.shape[1]+0.5, 1)
        xaxis = ((xaxis - origin[0]) * scale[0])
        yaxis = np.arange(-0.5, img.shape[0]+0.5, 1)
        yaxis = ((yaxis - origin[1]) * scale[1])
        extent = [xaxis[0], xaxis[-1], yaxis[0], yaxis[-1]]

#         py.figure(1)
#         py.imshow(img[:,:,0], extent=extent, cmap=py.cm.gray)
#         py.axis('equal')

        py.figure(2)
        py.imshow(img[:,:,1], extent=extent, cmap=py.cm.gray)
        py.axis('equal')

#         py.figure(3)
#         py.imshow(img[:,:,2], extent=extent, cmap=py.cm.gray)
#         py.axis('equal')

        foo = raw_input('Continue?')


    gcutil.rmall(['w51a_h_mosaic.fits', 
                  'w51a_k_mosaic.fits', 
                  'w51a_l_mosaic.fits'])

    pyfits.writeto('w51a_h_mosaic.fits', imgH)
    pyfits.writeto('w51a_k_mosaic.fits', imgK)
    pyfits.writeto('w51a_l_mosaic.fits', imgL)
