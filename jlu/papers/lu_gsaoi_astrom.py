import numpy as np
import pylab as py
import pyfits
import glob
import os
from jlu.util import img_scale
import atpy
import math
from jlu.util import statsIter
from scipy import interpolate as interp
from jlu.gc.gcwork import starset

ngc1851_data = '/u/jlu/data/gsaoi/commission/reduce/ngc1851/'

def guide_star_info():
    """
    Get the guide star information out of the image headers.
    """
    imgDir = ngc1851_data + 'starfinder/'

    imgs = glob.glob(imgDir + '*_G1_1_cr.fits')

    # Keep results in an output file
    f_out = open(ngc1851_data + 'guide_star_info.txt', 'w')
    f_fmt = '{0:8s} {1:15s} '

    for img in imgs:
        hdr = pyfits.getheader(img)

        f_out.write('{0:20s}  '.format(os.path.basename(img)))

        # There are 4 possible guide stars. Try to print them all out
        for ii in range(4):
            try:
                cfg = 'GWFS{0}CFG'.format(ii+1)
                obj = 'GWFS{0}OBJ'.format(ii+1)
                f_out.write(f_fmt.format(hdr[cfg], hdr[obj]))
            except KeyError:
                f_out.write(f_fmt.format('-', '-'))
                
        f_out.write('\n')

    f_out.close()

def image_info():
    """
    Get the exposure time and filter info for each image. Store it in
    an output file called image_info.txt.
    """
    imgDir = ngc1851_data + 'starfinder/'

    imgs = glob.glob(imgDir + '*_G1_1_cr.fits')

    # Keep results in an output file
    f_out = open(ngc1851_data + 'image_info.txt', 'w')
    f_fmt = '{img:15s}  {xoff:7.2f} {yoff:7.2f} {pa:5.1f}  {filt:12s}  '
    f_fmt += '{exp:6.2f} {coad:2d} {tot:7.2f}\n'

    for img in imgs:
        hdr = pyfits.getheader(img)

        xoff = hdr['XOFFSET']
        yoff = hdr['YOFFSET']
        pa = hdr['PA']
        filt = hdr['FILTER1']
        exp = hdr['EXPTIME']
        coadd = hdr['COADDS']

        f_out.write(f_fmt.format(img=os.path.basename(img), xoff=xoff, yoff=yoff,
                                 pa=pa, filt=filt, exp=exp, coad=coadd, tot=exp*coadd))

    f_out.close()

    
def ngc1851_image():
    """
    Plot an image of NGC 1851 for the paper.
    """
    image_root = '/u/jlu/data/gsaoi/commission/reduce/ngc1851/combo/ngc1851'

    # Load up the PSF stars file to get the coordinates.
    stars = atpy.Table(image_root + '_psf_stars_pixel.txt', type='ascii')

    stars.Xarc = stars.X * scale
    stars.Yarc = stars.Y * scale

    scale = 0.00995

    # gc = aplpy.FITSFigure(image_file)
    # gc.show_grayscale()

    img = pyfits.getdata(image_root + '.fits')
    img = img_scale.log(img, scale_min=0, scale_max=1e4)
    #img = img_scale.sqrt(img, scale_min=500, scale_max=5e4)
    # img = img_scale.linear(img, scale_min=500, scale_max=4e4)

    xmin = ((0 - cooPix[0]) * scale * -1.0) + cooAsec[0]
    xmax = ((img.shape[1] - cooPix[0]) * scale * -1.0) + cooAsec[0]
    ymin = ((0 - cooPix[1]) * scale) + cooAsec[1]
    ymax = ((img.shape[0] - cooPix[1]) * scale) + cooAsec[1]
    extent = [xmin, xmax, ymin, ymax]

    py.clf()
    py.imshow(img, extent=extent, cmap=py.cm.gray_r)
    
