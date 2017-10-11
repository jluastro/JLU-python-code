import numpy as np
import pylab as py
import pyfits, math
from gcreduce import gcutil
from gcwork import objects
from jlu.util import img_scale


datadir = '/u/jlu/data/m31/08oct/081021/SPEC/reduce/m31/ss/'
workdir = '/u/jlu/work/m31/nucleus/ifu_09_02_24/'
cuberoot = 'm31_08oct_Kbb_050'

cc = objects.Constants()

uvInNIRC2 = np.array([700.8, 582.7])

def plotWithHST(rotateFITS=True):
    # Load up the OSIRIS image
    cubeFile = datadir + cuberoot + '_img.fits'
    cube, cubehdr = pyfits.getdata(cubeFile, header=True)
    paCube = cubehdr['PA_SPEC']
    scaleCube = 0.05

    # Load up the NIRC2 image
    kFile = '/u/jlu/data/m31/05jul/combo/m31_05jul_kp.fits'
#     kFile = '/u/jlu/data/m31/09sep/combo/mag09sep_m31_kp.fits'
    k, khdr = pyfits.getdata(kFile, header=True)
    m31K = np.array([751.986, 716.989])
    paK = 0.0
    scaleK = 0.00995
    
    f330Root = '/u/jlu/data/m31/hst/m31_acshrc_f330w_j9am01030'
    f435Root = '/u/jlu/data/m31/hst/m31_acshrc_f435w_j9am01010'
    f555Root = '/u/jlu/data/m31/hst/m31_wfpc2pc1_f555w_u2lg020at'
    f814Root = '/u/jlu/data/m31/hst/m31_wfpc2pc1_f814w_u2lg020bt'
    ##########
    # Rotate all the HST images to PA=0, scale=NIRC2
    ##########
    if rotateFITS == True:
        # Load up the HST ACS image (330 nm)
        f330File = f330Root + '.fits'
        f330FileRot = f330Root + '_rot.fits'
        f330, f330hdr  = pyfits.getdata(f330File, 1, header=True)
        m31F330 = np.array([547.0, 623.2])
        paF330 = f330hdr['ORIENTAT']
        scaleF330 = 0.025
        
        # Load up the HST ACS image (435 nm)
        f435File = f435Root + '.fits'
        f435FileRot = f435Root + '_rot.fits'
        f435, f435hdr  = pyfits.getdata(f435File, 1, header=True)
        m31F435 = np.array([546.9, 623.5])
        paF435 = f435hdr['ORIENTAT']
        scaleF435 = 0.025

        # Load up the HST ACS image (555 nm)
        f555File = f555Root + '.fits'
        f555FileRot = f555Root + '_rot.fits'
        f555, f555hdr  = pyfits.getdata(f555File, 1, header=True)
        m31F555 = np.array([973.0, 961.0])
        paF555 = f555hdr['ORIENTAT']
        scaleF555 = 0.1
        
        # Load up the HST ACS image (814 nm)
        f814File = f814Root + '.fits'
        f814FileRot = f814Root + '_rot.fits'
        f814, f814hdr  = pyfits.getdata(f814File, 1, header=True)
        m31F814 = np.array([975.0, 962.0])
        paF814 = f814hdr['ORIENTAT']
        scaleF814 = 0.1

        print('scaleK = ', scaleK)
        print('scaleF330 = ', scaleF330)
        print('scaleF435 = ', scaleF435)
        print('scaleF555 = ', scaleF555)
        print('scaleF814 = ', scaleF814)

        print('paK = ', paK)
        print('paF330 = ', paF330)
        print('paF435 = ', paF435)
        print('paF555 = ', paF555)
        print('paF814 = ', paF814)

        gcutil.rmall([f330FileRot, f435FileRot, f555FileRot, f814FileRot])

        from pyraf import iraf as ir
        ir.unlearn('imlintran')
        ir.imlintran.boundary = 'constant'
        ir.imlintran.constant = 0
        ir.imlintran.interpolant = 'spline3'
        ir.imlintran.fluxconserve = 'yes'

        # 330
        ir.imlintran.xin = m31F330[0]
        ir.imlintran.yin = m31F330[1]
        ir.imlintran.xout = m31K[0]
        ir.imlintran.yout = m31K[1]
        ir.imlintran.ncols = k.shape[1]
        ir.imlintran.nlines = k.shape[0]

        ir.imlintran(f330File+'[1]', f330FileRot, paF330, paF330, 
                     scaleK/scaleF330, scaleK/scaleF330)

        # 435
        ir.imlintran.xin = m31F435[0]
        ir.imlintran.yin = m31F435[1]
        ir.imlintran.xout = m31K[0]
        ir.imlintran.yout = m31K[1]
        ir.imlintran.ncols = k.shape[1]
        ir.imlintran.nlines = k.shape[0]

        ir.imlintran(f435File+'[1]', f435FileRot, paF435, paF435, 
                     scaleK/scaleF435, scaleK/scaleF435)

        # 555
        ir.imlintran.xin = m31F555[0]
        ir.imlintran.yin = m31F555[1]
        ir.imlintran.xout = m31K[0]
        ir.imlintran.yout = m31K[1]
        ir.imlintran.ncols = k.shape[1]
        ir.imlintran.nlines = k.shape[0]

        ir.imlintran(f555File+'[1]', f555FileRot, paF555, paF555, 
                     scaleK/scaleF555, scaleK/scaleF555)

        # 814
        ir.imlintran.xin = m31F814[0]
        ir.imlintran.yin = m31F814[1]
        ir.imlintran.xout = m31K[0]
        ir.imlintran.yout = m31K[1]
        ir.imlintran.ncols = k.shape[1]
        ir.imlintran.nlines = k.shape[0]

        ir.imlintran(f814File+'[1]', f814FileRot, paF814, paF814, 
                     scaleK/scaleF814, scaleK/scaleF814)
        
        
        
        
    f330 = pyfits.getdata(f330Root + '_rot.fits')
    f435 = pyfits.getdata(f435Root + '_rot.fits')

    img = np.zeros((k.shape[0], k.shape[1], 3), dtype=float)
    img[:,:,0] = img_scale.linear(k, scale_min=300, scale_max=2400)
    img[:,:,1] = img_scale.linear(f435, scale_min=0.45, scale_max=1.8)
    img[:,:,2] = img_scale.linear(f330, scale_min=0, scale_max=0.16)

    # Axes
    xaxis = (np.arange(img.shape[0], dtype=float) - m31K[0])*0.00995
    yaxis = (np.arange(img.shape[1], dtype=float) - m31K[1])*0.00995
    
    py.clf()
    py.imshow(img, aspect='equal', extent=[xaxis[0],xaxis[-1],yaxis[0],yaxis[-1]])
    #py.plot([m31K[0]], [m31K[1]], 'ko')
    #py.axis([625, 825, 650, 850])
    py.plot([0], [0], 'k+')
    py.axis([-1.5, 1.5, -1.5, 1.5])
    py.xlabel('X Offset from M31* (arcsec)')
    py.ylabel('Y Offset from M31* (arcsec)')
    py.title('Blue = F330W, Green = F435W, Red = Kband')
    py.savefig(workdir + 'plots/hst_nirc2_rgb.png')
    py.show()


def plotNewUV():
    # Load up the OSIRIS image
    cubeFile = datadir + cuberoot + '_img.fits'
    cube, cubehdr = pyfits.getdata(cubeFile, header=True)
    paCube = cubehdr['PA_SPEC']
    scaleCube = 0.05

    # Load up the NIRC2 image
    kFile = '/u/jlu/data/m31/05jul/combo/m31_05jul_kp.fits'
    k, khdr = pyfits.getdata(kFile, header=True)
    #m31K = np.array([747.63, 708.65])
    m31K = np.array([750.986, 717.989])
    paK = 0.0
    scaleK = 0.00995
    
    f330Root = '/u/jlu/data/m31/hst/m31_acshrc_f330w_j9am01030'
    f435Root = '/u/jlu/data/m31/hst/m31_acshrc_f435w_j9am01010'
    f555Root = '/u/jlu/data/m31/hst/m31_wfpc2pc1_f555w_u2lg020at'
    f814Root = '/u/jlu/data/m31/hst/m31_wfpc2pc1_f814w_u2lg020bt'
        
    f330 = pyfits.getdata(f330Root + '_rot.fits')
    f435 = pyfits.getdata(f435Root + '_rot.fits')

    img = np.zeros((k.shape[0], k.shape[1], 3), dtype=float)
    img[:,:,0] = img_scale.linear(k, scale_min=1600, scale_max=2400)
    img[:,:,1] = img_scale.linear(f435, scale_min=0.9, scale_max=2.0)
    img[:,:,2] = img_scale.linear(f330, scale_min=0.06, scale_max=0.18)

    # Axes
    xaxis = (np.arange(img.shape[0], dtype=float) - m31K[0])*0.00995
    yaxis = (np.arange(img.shape[1], dtype=float) - m31K[1])*0.00995
    
    py.close(2)
    py.figure(2, figsize=(18,6))
    py.clf()
    py.subplots_adjust(left=0.05, right=0.98, top=0.98, bottom=0.05)

    py.subplot(1, 3, 1)
    py.imshow(img[:,:,0], aspect='equal', cmap=py.cm.Oranges,
              extent=[xaxis[0],xaxis[-1],yaxis[0],yaxis[-1]])
    py.plot([0], [0], 'kx', mew=2, ms=20)
    py.axis([-0.2, 0.2, -0.2, 0.2])
    py.xlabel('X Offset from M31* (arcsec)')
    py.ylabel('Y Offset from M31* (arcsec)')
    py.title('Kband')

    py.subplot(1, 3, 2)
    py.imshow(img[:,:,1], aspect='equal', cmap=py.cm.Greens,
              extent=[xaxis[0],xaxis[-1],yaxis[0],yaxis[-1]])
    py.plot([0], [0], 'kx', mew=2, ms=20)
    py.axis([-0.2, 0.2, -0.2, 0.2])
    py.xlabel('X Offset from M31* (arcsec)')
    py.ylabel('Y Offset from M31* (arcsec)')
    py.title('F435W')

    py.subplot(1, 3, 3)
    py.imshow(img[:,:,2], aspect='equal', cmap=py.cm.Blues,
              extent=[xaxis[0],xaxis[-1],yaxis[0],yaxis[-1]])
    py.plot([0], [0], 'kx', mew=2, ms=20)
    py.axis([-0.2, 0.2, -0.2, 0.2])
    py.xlabel('X Offset from M31* (arcsec)')
    py.ylabel('Y Offset from M31* (arcsec)')
    py.title('F330W')

    
    py.savefig(workdir + 'plots/hst_nirc2_rgb_zoom_uv.png')
    py.show()
    
    

    
