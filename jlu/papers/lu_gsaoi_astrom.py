import numpy as np
import pylab as py
import pyfits
import glob

ngc1815_data = '/u/jlu/data/gsaoi/commission/reduce/ngc1815/'

def guide_star_info():
    """
    Get the guide star information out of the image headers.
    """
    imgDir = ngc1815_data + 'starfinder/'

    imgs = glob.glob(imgDir + '*_G1_1_cr.fits')

    for img in imgs:
        hdr = pyfits.getheader(img)

        print hdr['GWFS1CFG'], hdr['GWFS1OBJ'], \
            hdr['GWFS2CFG'], hdr['GWFS2OBJ'], \
            hdr['GWFS3CFG'], hdr['GWFS3OBJ'], \
            hdr['GWFS4CFG'], hdr['GWFS4OBJ']

