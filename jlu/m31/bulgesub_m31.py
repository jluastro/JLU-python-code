import math
import pyfits
import numpy as np
import pylab as py
import photutils as pu
import scipy
import asciidata
from scipy import signal
from pyraf import iraf as ir
from jlu.osiris import cube as cube_code
from jlu.osiris import spec
from jlu.m31 import ppxf_m31
import glob
import os
import pdb
import atpy

datadir = '/Users/kel/Documents/Projects/M31/analysis_new/ifu_11_11_30/'

cuberoot = 'm31_all_scalederr_cleanhdr'

bhpos = ppxf_m31.bhpos

def plotLitSersic(outSB=None,plot=True):

    # plots a variety of Sersic profiles with values from the literature
    # can return the profiles in an array, can specify central SB (mag/arcsec^2)
    # can turn off plotting if desired

    # literature values
    # [KB99, Courteau2011_E, Courteau2011_M, Dorman2013]
    # Sersic index
    n_b = np.array([2.19, 2.18, 1.83, 1.917])
    # half-light radius, in kpc
    R_e_kpc = np.array([1.0, 0.82, 0.74, 0.778])
    # in pc
    R_e = R_e_kpc*1000.
    # central surface brightness, in mag arcsec^-2
    # [V, 3.6 um, I, I] (last two are same data set)
    if outSB is None:
        mu_e = np.array([17.55, 15.77, 17.73, 17.849])
    else:
        mu_e = np.ones(4)*outSB
    # convert to central intensity
    I_e = 10.**(mu_e/(-2.5))
    # radius to calculate for (radius of data is 2" - going out slightly further)
    radarcsec = 60.
    radpc = radarcsec*3.73
    R = np.arange(radpc)
    #R = np.arange(radarcsec)

    I_b = np.zeros((len(n_b),len(R)),dtype=float)
    mu_b = np.zeros((len(n_b),len(R)),dtype=float)

    if plot:
        fig=py.figure(2)
        py.clf()

    for i in np.arange(len(n_b)):
        # from Cappaccioli 1989
        b_n = 1.9992*n_b[i] - 0.3271
        negb_n = -1. * b_n
        I_b[i,:] = I_e[i]*np.exp(negb_n*( ((R/R_e[i])**(1./n_b[i])) - 1.))
        mu_b[i,:] = -2.5*np.log10(I_b[i,:])
        if plot:
            py.semilogx(R[1:-1],mu_b[i,1:-1])

    if plot:
        py.legend(('KB99 (V band)','C11_E (3.6 $\mu$m)','C11_M (I band)','D13 (I band)'))
        py.ylim(mu_b.max()+0.5,mu_b.min()-0.5)
        py.ylabel('$\mu$ (mag arcsec$^{-2}$)')
        py.xlabel('Radius (pc)')
        
        ax1 = fig.add_subplot(111)
        ax2 = ax1.twiny()
        ax2.set_xlim(ax1.get_xlim())
        ax2.set_xscale('log')
        ax2.set_xticks(np.array([3.73,37.3,223.]))
        ax2.set_xticklabels(['1','10','60'])
        ax2.set_xlabel('Radius (arcsec)')

    return mu_b, I_b, R

def plotNIRC2profile(mask=True,wide=False):

    if wide:
        nirc2file = '/Users/kel/Documents/Projects/M31/data/nirc2_koa/KOA_3344/NIRC2/calibrated/N2.20071019.36147_drp.fits'
        # eyeballed in wide camera image - need to properly register if using this one
        bh_nirc2 = [539.,485.]
        # arcsec / pixel
        pixscale = 0.04
        # wide camera FOV is 40"x40"
        rad = 20.
        # PA of the wide NIRC2 data
        # 360 - PA of the frame - PA of the bulge (Dorman 2013) - 90 (to get to ref axis of photutils)
        pa = np.radians(360. - 151. - 6.632 - 90.)
        # radius in arcsec to match the fitted Sersic profile at
        #matchR = 10.
        matchR = 5.
        endR = 90
    else:
        nirc2file = '/Users/kel/Documents/Projects/M31/analysis_old/align/m31_2125nm_osiris_rot.fits'
        # from analysis.registerNIRC2toOSIRIS(), saved to align/bhpos.txt
        bh_nirc2 = [699.6, 584.3]
        pixscale = 0.01
        #NIRC2 FOV is 10"x10"
        rad = 6.
        # PA of the wide NIRC2 data
        # 360 - PA of the frame - PA of the bulge (Dorman 2013) - 90 (to get to ref axis of photutils)
        pa = np.radians(360. - 56. - 6.632 - 90.)
        # radius in arcsec to match the fitted Sersic profile at
        matchR = 5.
        endR = 25
    
    cubeimg = pyfits.getdata(datadir + cuberoot + '_img.fits')
    nirc2 = pyfits.getdata(nirc2file)

    # NIRC2 gain, from header - both narrow and wide cameras
    gain = 4.
    nirc2 *= gain

    #if mask:
        # 1 pixel = 0.01"
    #    distarcsec = 1.
    #    maskdistpix = distarcsec/0.01
    #else:
    #    maskdistpix = 0.

    # this is for circular apertures - using elliptical now
    #dist = dist_circle(nirc2.shape,bh_nirc2)
    #distas = dist*0.01
    # wide camera
    #distas = dist*0.04

    # set the inner edges of the apertures, width of 0.25"
    binin = np.arange(rad*4)/4.
    #binin = np.arange(rad)
    binsbflux = np.zeros(len(binin))
    binsbmag = np.zeros(len(binin))
    
    # ellipticity, Dorman 2013
    ell = 0.277
    # inclination angle of the large-scale disk
    thetai = np.radians(77.5)
    
    for i in np.arange(len(binin)-1):
        # this was for circular apertures
        #good = np.where((distas >= binin[i]) & (distas < binin[i+1]))
        #area = (np.pi*(binin[i+1]**2)) - (np.pi*(binin[i]**2))
        #tmp = np.sum(nirc2[good])/area
        #tmpmag = -2.5*np.log10(tmp)
        #binsbflux[i] = tmp
        #binsbmag[i] = tmpmag
        # annulus in pixels
        ap = pu.EllipticalAnnulus(bh_nirc2,binin[i]/pixscale,binin[i+1]/pixscale,(binin[i+1]/pixscale)*np.sqrt(1-(ell**2.)),pa)
        tmpflux = pu.aperture_photometry(nirc2,ap)
        # area in pixels
        area = ap.area()
        # area in arcsec
        area *= pixscale
        # deproject the area and flux (to match fitted SB profiles)
        dearea = area/np.cos(thetai)
        binsbflux[i] = tmpflux[0][0]/dearea
        # convert to mags
        tmpmag = -2.5*np.log10(tmpflux[0][0]/dearea)
        binsbmag[i] = tmpmag

    # fitted profiles
    # constants for KB99
    n_KB99 = 2.19
    # in pc
    Re_KB99 = 1.0*1000.
    b_n_KB99 = 1.9992*n_KB99 - 0.3271
    # constants for Courteau2011_E (3.6 um profile)
    n_C11e = 2.18
    Re_C11e = 0.82*1000.
    b_n_C11e = 1.9992*n_C11e - 0.3271
    # difference between the magnitude at the effective radius (Re) and that
    # at the given radius (matchR)
    diffmag_KB99 = -2.5*np.log10(np.exp(b_n_KB99*(((matchR*3.93/Re_KB99)**(1./n_KB99)) - 1.)))
    diffmag_C11e = -2.5*np.log10(np.exp(b_n_C11e*(((matchR*3.93/Re_C11e)**(1./n_C11e)) - 1.)))
    # scale by this difference
    matchidx = np.where(binin == matchR)
    profMu_KB99, profI_KB99, profR_KB99 = plotLitSersic(outSB=(binsbmag[matchidx]+diffmag_KB99),plot=False)
    profMu_C11e, profI_C11e, profR_C11e = plotLitSersic(outSB=(binsbmag[matchidx]+diffmag_C11e),plot=False)
        
    py.figure(3)
    py.clf()
    py.plot(binin,binsbflux,'ko')
    py.ylabel('$\mu$ (e$^{-1}$ arcsec$^{-2}$ [NOT calibrated])')
    py.xlabel('arcsec')
    pdb.set_trace()
    py.clf()
    py.plot(binin,binsbmag,'ko')
    if wide:
        py.ylim(-10.5,-13)
    else:
        py.ylim(-10.0,-14)
    py.ylabel('$\mu$ (mag arcsec$^{-2}$ [NOT calibrated])')
    py.xlabel('arcsec')
    pdb.set_trace()
    py.plot(profR_KB99[0:endR]/3.93, profMu_KB99[0,0:endR],'g-')
    py.plot(profR_C11e[0:endR]/3.93, profMu_C11e[1,0:endR],'r-')
    py.legend(('Observed (Kp)','KB99 bulge (V band)','C11_E bulge (3.6 $\mu$m)'))
    #pdb.set_trace()

def dist_circle(n=None,cent=None):
    # python rewrite of astrolib's dist_circle function
    # n: size of array, either a scale (for a square array) or [x,y]
    # cent: [x,y] to calculate distance from
    # output: returns array with distance from cent (in units of pixels)
    
    if n is None:
        print 'Please enter the size of the array.'
        return
    if cent is None:
        print 'Please enter the coordinates to calculate distance from.'
        return

    if len(n)==2:
        nx = n[0]
        ny = n[1]
    elif len(n)==1:
        nx = n
        ny = n
    else:
        print '2D arrays only'
        return

    xcen = cent[0]
    ycen = cent[1]

    # x distances
    x_2 = (np.arange(nx) - xcen)**2
    # y distances
    y_2 = (np.arange(ny) - ycen)**2
    #initialize output
    im = np.zeros((nx,ny),dtype=float)

    # loop over rows
    for i in range(ny):
        im[:,i] = np.sqrt(x_2 + y_2[i])

    return im

    
