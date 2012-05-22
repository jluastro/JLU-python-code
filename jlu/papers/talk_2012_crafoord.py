import pyfits
import pylab as py
import numpy as np
from jlu.util import img_scale
from pyraf import iraf as ir

workdir = '/u/jlu/doc/present/2012_05_crafoord/'

def make_gc_images():
    make_speckle_image()
    make_deep_ao_image()
    make_big_ao_image()
    make_color_ao_image()
    
def rotate_2005_lgsao():
    # Need to rotate an image.
    ir.unlearn('rotate')
    ir.rotate.boundary = 'constant'
    ir.rotate.constant = 0
    ir.rotate.interpolant = 'spline3'
    ir.rotate.ncols = 1040
    ir.rotate.nlines = 1040
    
    ir.rotate.xin = 528
    ir.rotate.xout = 540
    ir.rotate.yin = 645
    ir.rotate.yout = 410
    ir.rotate(workdir + 'mag05jullgs_kp.fits', workdir + 'mag05jullgs_kp_rot.fits', 190)

    ir.rotate.xin = 535
    ir.rotate.xout = 540
    ir.rotate.yin = 655
    ir.rotate.yout = 410
    ir.rotate(workdir + 'mag05jullgs_h.fits', workdir + 'mag05jullgs_h_rot.fits', 190)

    ir.rotate.xin = 572
    ir.rotate.xout = 540
    ir.rotate.yin = 694
    ir.rotate.yout = 410
    ir.rotate(workdir + 'mag05jullgs_lp.fits', workdir + 'mag05jullgs_lp_rot.fits', 190)

    ir.rotate.xin = 483
    ir.rotate.xout = 483
    ir.rotate.yin = 612
    ir.rotate.yout = 612
    ir.rotate(workdir + 'mag04jul.fits', workdir + 'mag04jul_rot.fits', 1)


def make_speckle_image():
    speckle = pyfits.getdata(workdir + 'mag04jul_rot.fits')
    speckImg = img_scale.sqrt(speckle, scale_min=0, scale_max=150)
    
    sgra = np.array([483, 612])
    scale = 0.0102
    xextent = np.array([0, speckImg.shape[0]])
    yextent = np.array([0, speckImg.shape[0]])
    xextent = (xextent - sgra[0]) * -scale
    yextent = (yextent - sgra[1]) * scale
    extent = [xextent[0], xextent[-1], yextent[0], yextent[-1]]

    py.figure(1, figsize=(6, 6))
    py.clf()
    py.subplots_adjust(left=0, right=1, bottom=0, top=1)
    py.imshow(speckImg, cmap=py.cm.gray, extent=extent)

    ax = py.gca()
    py.setp(ax.get_xticklabels(), visible=False)
    py.setp(ax.get_yticklabels(), visible=False)

    py.axis('equal')
    py.xlim(3.7, -2.3)
    py.ylim(-2.8, 3.2)

    py.savefig(workdir + 'img_speckle.png')


def make_deep_ao_image():
    img = pyfits.getdata(workdir + 'mag05jullgs_kp_rot.fits')
    imgScl = img_scale.sqrt(img, scale_min=200, scale_max=15000)

    sgra = np.array([538, 400])
    scale = 0.00995
    xextent = np.array([0, img.shape[0]])
    yextent = np.array([0, img.shape[0]])
    xextent = (xextent - sgra[0]) * -scale
    yextent = (yextent - sgra[1]) * scale
    extent = [xextent[0], xextent[-1], yextent[0], yextent[-1]]

    py.figure(2, figsize=(6, 6))
    py.clf()
    py.subplots_adjust(left=0, right=1, bottom=0, top=1)
    py.imshow(imgScl, cmap=py.cm.gray, extent=extent)
    
    ax = py.gca()
    py.setp(ax.get_xticklabels(), visible=False)
    py.setp(ax.get_yticklabels(), visible=False)

    py.xlim(3.7, -2.3)
    py.ylim(-2.7, 3.3)

    py.savefig(workdir + 'img_lgsao_deep.png')


def make_big_ao_image():
    img = pyfits.getdata(workdir + 'mag05jullgs_kp_rot.fits')
    imgScl = img_scale.sqrt(img, scale_min=200, scale_max=15000)

    sgra = np.array([540, 410])
    scale = 0.00995
    xextent = np.array([0, img.shape[0]])
    yextent = np.array([0, img.shape[0]])
    xextent = (xextent - sgra[0]) * -scale
    yextent = (yextent - sgra[1]) * scale
    extent = [xextent[0], xextent[-1], yextent[0], yextent[-1]]

    py.figure(3, figsize=(10, 10))
    py.clf()
    py.subplots_adjust(left=0, right=1, bottom=0, top=1)
    py.imshow(imgScl, cmap=py.cm.gray, extent=extent)
    
    ax = py.gca()
    py.setp(ax.get_xticklabels(), visible=False)
    py.setp(ax.get_yticklabels(), visible=False)

    py.xlim(5, -5)
    py.ylim(-4, 6)

    py.savefig(workdir + 'img_lgsao_big.png')

def make_color_ao_image():
    h_img = pyfits.getdata(workdir + 'mag05jullgs_h_rot.fits')
    h_imgScl = img_scale.sqrt(h_img, scale_min=50, scale_max=5500)
    kp_img = pyfits.getdata(workdir + 'mag05jullgs_kp_rot.fits')
    kp_imgScl = img_scale.sqrt(kp_img, scale_min=10, scale_max=17000)
    lp_img = pyfits.getdata(workdir + 'mag05jullgs_lp_rot.fits')
    lp_imgScl = img_scale.sqrt(lp_img, scale_min=-100, scale_max=60000)

    sgra = np.array([540, 410])
    scale = 0.00995
    h_xextent = np.array([0, h_img.shape[0]])
    h_yextent = np.array([0, h_img.shape[0]])
    h_xextent = (h_xextent - sgra[0]) * -scale
    h_yextent = (h_yextent - sgra[1]) * scale
    h_extent = [h_xextent[0], h_xextent[-1], h_yextent[0], h_yextent[-1]]

    img = np.zeros((h_img.shape[0], h_img.shape[1], 3), dtype=float)
    img[:,:,0] = lp_imgScl
    img[:,:,1] = kp_imgScl
    img[:,:,2] = h_imgScl

    py.figure(4, figsize=(10, 10))
    py.clf()
    py.subplots_adjust(left=0, right=1, bottom=0, top=1)
    py.imshow(img, extent=h_extent)
    
    ax = py.gca()
    py.setp(ax.get_xticklabels(), visible=False)
    py.setp(ax.get_yticklabels(), visible=False)

    py.xlim(5, -5)
    py.ylim(-4, 6)

    py.savefig(workdir + 'img_lgsao_color.png')

