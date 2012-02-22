import numpy as np
import pylab as py
import pyfits
from gcwork import starTables
from scipy.ndimage import interpolation as interp
from jlu.util import img_scale
from gcreduce import gcutil

def arches_figure():
    """
    Plot a 3 panel figure showing seeing-limited, HST, and AO data on the
    Arches cluster to illustrate the power of AO.
    """
    # ----------
    # NIRC2
    # ----------

    hroot = 'mag06maylgs2_arch_f1_h'
    kroot = 'mag06maylgs2_arch_f1_kp'
    lroot = 'mag06maylgs2_arch_f1_lp'

    cooStar = 'f1_psf0'

    scaleMinH = 1000
    scaleMinK = 800
    scaleMinL = 600

    scaleMaxH = 4500
    scaleMaxK = 8000
    scaleMaxL = 10000

    img = np.zeros((1500, 1500, 3), dtype=float)
    origin = np.array([750.0, 750.0])

    labelFile = '/u/ghezgroup/data/gc/source_list/label_arch.dat'
    labels = starTables.Labels(labelFile=labelFile)

    dataDir = '/u/ghezgroup/data/gc/06maylgs2/combo/'

    # Load up the images
    h = pyfits.getdata(dataDir + hroot + '.fits')
    k = pyfits.getdata(dataDir + kroot + '.fits')
    l = pyfits.getdata(dataDir + lroot + '.fits')

        # Make the arrays into the largest size.
    h_new = np.zeros((img.shape[0], img.shape[1]), dtype=float)
    k_new = np.zeros((img.shape[0], img.shape[1]), dtype=float)
    l_new = np.zeros((img.shape[0], img.shape[1]), dtype=float)

    h_new[0:h.shape[0], 0:h.shape[1]] = h
    k_new[0:k.shape[0], 0:k.shape[1]] = k
    l_new[0:l.shape[0], 0:l.shape[1]] = l

    # Load up the coo stars
    tmpH = open(dataDir + hroot + '.coo').readline().split()
    cooH = np.array([float(tmpH[0]), float(tmpH[1])])
        
    tmpK = open(dataDir + kroot + '.coo').readline().split()
    cooK = np.array([float(tmpK[0]), float(tmpK[1])])

    tmpL = open(dataDir + lroot + '.coo').readline().split()
    cooL = np.array([float(tmpL[0]), float(tmpL[1])])
        
    # Get the coordinates of each coo star in arcsec.
    idxH = np.where(labels.name == cooStar)[0][0]
    idxK = np.where(labels.name == cooStar)[0][0]
    idxL = np.where(labels.name == cooStar)[0][0]

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

#     gcutil.rmall(['arches_f1_h.fits', 'arches_f1_kp.fits', 'arches_f1_lp.fits'])
#     pyfits.writeto('arches_f1_h.fits', h)
#     pyfits.writeto('arches_f1_kp.fits', k)
#     pyfits.writeto('arches_f1_lp.fits', l)

    img[idx[0],idx[1],0] = img_scale.sqrt(l[idx[0],idx[1]], 
                                          scale_min=scaleMinL, 
                                          scale_max=scaleMaxL)
    img[idx[0],idx[1],1] = img_scale.sqrt(k[idx[0], idx[1]], 
                                          scale_min=scaleMinK, 
                                          scale_max=scaleMaxK)
    img[idx[0],idx[1],2] = img_scale.sqrt(h[idx[0], idx[1]], 
                                          scale_min=scaleMinH, 
                                          scale_max=scaleMaxH)

    # Define the axes
    xaxis = np.arange(-0.5, img.shape[1]+0.5, 1)
    xaxis = ((xaxis - origin[0]) * scale[0])
    yaxis = np.arange(-0.5, img.shape[0]+0.5, 1)
    yaxis = ((yaxis - origin[1]) * scale[1])
    extent = [xaxis[0], xaxis[-1], yaxis[0], yaxis[-1]]

    img_nirc2 = img
    ext_nirc2 = extent

    # ----------
    # UKIDSS
    # ----------
    scaleMinJ = 20
    scaleMinH = 400
    scaleMinK = 1000

    scaleMaxJ = 5000
    scaleMaxH = 35000
    scaleMaxK = 90000

    dataDir = '/u/jlu/data/arches/ukidss/'

    # Load up the images
    j = pyfits.getdata(dataDir + 'ukidss_arches_j.fits')
    h = pyfits.getdata(dataDir + 'ukidss_arches_h.fits')
    k = pyfits.getdata(dataDir + 'ukidss_arches_k.fits')

    img = np.zeros((j.shape[0], j.shape[1], 3), dtype=float)
    origin = [173, 198]
    scale = [-0.2, 0.2]

    xx, yy = np.meshgrid(np.arange(img.shape[0]), np.arange(img.shape[1]))

    img[:,:,0] = img_scale.sqrt(k,
                                scale_min=scaleMinK, 
                                scale_max=scaleMaxK)
    img[:,:,1] = img_scale.sqrt(h,
                                scale_min=scaleMinH, 
                                scale_max=scaleMaxH)
    img[:,:,2] = img_scale.sqrt(j,
                                scale_min=scaleMinJ, 
                                scale_max=scaleMaxJ)
    # Define the axes
    xaxis = np.arange(-0.5, img.shape[1]+0.5, 1)
    xaxis = ((xaxis - origin[0]) * scale[0])
    yaxis = np.arange(-0.5, img.shape[0]+0.5, 1)
    yaxis = ((yaxis - origin[1]) * scale[1])
    extent = [xaxis[0], xaxis[-1], yaxis[0], yaxis[-1]]


    img_ukidss = img
    ext_ukidss = extent


    py.figure(2, figsize=(6,12))
    py.clf()
    py.subplots_adjust(bottom=0.05, top=0.95, hspace=0.25)

    py.subplot(2, 1, 1)
    py.imshow(img_ukidss, extent=ext_ukidss)
    py.axis('equal')
    py.axis([4.5, -6.5, -6.5, 4.5])
    py.title('UKIDSS JHK')
    py.xlabel('R.A. Offset (arcsec)')
    py.ylabel('Dec. Offset (arcsec)')

    py.subplot(2, 1, 2)
    py.imshow(img_nirc2, extent=ext_nirc2)
    py.axis('equal')
    py.axis([4.5, -6.5, -6.5, 4.5])
    py.title("Keck AO HK'L'")
    py.xlabel('R.A. Offset (arcsec)')
    py.ylabel('Dec. Offset (arcsec)')

    py.savefig('arches_see_vs_ao.png')


    
