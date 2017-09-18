import math
import pyfits
import numpy as np
import pylab as py
import photutils as pu
import scipy
import asciidata
from scipy import signal
from gcwork import objects
from pyraf import iraf as ir
from jlu.osiris import cube as cube_code
from jlu.osiris import spec
from jlu.m31 import ppxf_m31
from jlu.m31 import ifu
import glob
import os
import pdb
import atpy

datadir = '/Users/kel/Documents/Projects/M31/analysis_new/ifu_11_11_30/'

cuberoot = 'm31_all_scalederr_cleanhdr'

bhpos = ppxf_m31.bhpos
bhpos_hor = ppxf_m31.bhpos_hor
bhpos_pix = ppxf_m31.bhpos_pix

cc = objects.Constants()

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

def plotNIRC2profile(mask=True,wide=False,matchR=None,bulgeProfile='C11E',toplot=True,verbose=True):

    if wide:
        # NIRC2 gain, from header - both narrow and wide cameras
        gain = 4.
        nirc2file = '/Users/kel/Documents/Projects/M31/data/nirc2_koa/KOA_3344/NIRC2/calibrated/N2.20071019.36147_drp.fits'
        #nirc2file = '/Users/kel/Documents/Projects/M31/analysis_old/align/m31_0555nm.fits'
        cubeimg = pyfits.getdata(datadir + cuberoot + '_img.fits')
        nirc2 = pyfits.getdata(nirc2file)
        # eyeballed in wide camera image - need to properly register if using this one
        #bh_nirc2 = [539.,485.]
        # registered the wide FOV camera
        #bh_nirc2 = [ 538.0, 484.5]
        bh_nirc2 = [ 538.0, 484.5]
        #bh_nirc2 = [416.,427.]
        # arcsec / pixel
        pixscale = 0.04
        #pixscale = 0.0455
        # wide camera FOV is 40"x40"
        rad = 20.
        # PA of the wide NIRC2 data
        # 360 - PA of the frame - PA of the bulge (Dorman 2013) - 90 (to get to ref axis of photutils)
        #pa = np.radians(360. - 6.632 - 90.)
        #pa = np.radians(360. - 79.57 - 6.632 - 90.)
        pa = np.radians(360. - 151. - 6.632 - 90.)
        #pa = np.radians(100.)
        # radius in arcsec to match the fitted Sersic profile at
        #matchR = 10.
        if matchR is None:
            matchR = 5.
        endR = 90
        itime = 5.
        #itime = 500.
        #testing - subtracting a sky level
        nirc2 -= 455.
        #nirc2 -= 600.
        py.figure(3)
    else:
        # NIRC2 gain, from header - both narrow and wide cameras
        gain = 4.
        #nirc2file = '/Users/kel/Documents/Projects/M31/analysis_old/align/m31_2125nm_osiris_rot.fits'
        #bh_nirc2 = [699.6, 584.3]
        # 360 - PA of the frame - PA of the bulge (Dorman 2013) - 90 (to get to ref axis of photutils)
        pa = np.radians(360. - 56. - 6.632 - 90.) # org
        #pixscale = 0.01  
        #itime = 60.
        #nirc2file = '/Users/kel/Documents/Projects/M31/data/combo/m31_05jul_kp.fits'
        #bh_nirc2 = [755.,721.]
        #nirc2file = '/Users/kel/Documents/Projects/M31/analysis_old/align/m31_2125nm.fits'
        #bh_nirc2 = [712.,583.]
        #nirc2file = '/Users/kel/Documents/Projects/M31/data/nirc2_koa/KOA_3344/NIRC2/narrow/N2.20090910.42407.fits'
        #bh_nirc2 = [533.,531.]
        #pa = np.radians(360. - 6.632 - 90.)
        #nirc2file = '/Users/kel/Documents/Projects/M31/data/nirc2_koa/KOA_3344/NIRC2/narrow/calibrated/KOA_25476/NIRC2/calibrated/N2.20090910.42407_drp.fits.gz'
        #bh_nirc2 = [535.,532.]
        #nirc2file = '/Users/kel/Documents/Projects/M31/analysis_old/align/m31_2125nm_osiris_rot_scale.fits'
        nirc2file = '/Users/kel/Documents/Projects/M31/analysis_old/align/m31_2125nm_w_osiris_rot_scale_NIRC2_DTOTOFF_no100828.fits'
        bh_nirc2 = [bhpos_pix[1],bhpos_pix[0]]
        pixscale = 0.05
        itime=5.
        #nirc2file = '/Users/kel/Documents/Projects/M31/data/combo/mag09sep_m31_j.fits'
        #bh_nirc2 = [592.,589.]
        #itime = 120.
        #pa = np.radians(360. - 6.632 - 90.)
        #pixscale = 0.01
        #nirc2file = '/Users/kel/Documents/Projects/M31/data/combo/mag09sep_m31_h.fits'
        #bh_nirc2 = [602.,593.]
        #itime = 60.
        #nirc2file = '/Users/kel/Documents/Projects/M31/analysis_old/align/m31_0330nm.fits'
        #bh_nirc2 = [586.,634.]
        #itime = 1. # image in e-/s, not counts
        #gain=1.
        #pa = np.radians(360. - 81.837029 - 6.632 - 90.)
        #pixscale=0.025
        #nirc2file = '/Users/kel/Documents/Projects/M31/analysis_old/align/m31_0435nm.fits'
        #bh_nirc2 = [585.,634.]
        #gain=1.
        #itime=1.
        #pa = np.radians(360. - 81.837029 - 6.632 - 90.)
        #pixscale=0.025
        #nirc2file = '/Users/kel/Documents/Projects/M31/analysis_old/align/m31_0555nm.fits'
        #bh_nirc2 = [416.,427.]
        #gain=42.
        #itime=500.
        #pa = np.radians(360. - 79.570702 - 6.632 - 90.)
        #pixscale=0.04552
        #nirc2file = '/Users/kel/Documents/Projects/M31/analysis_old/align/m31_0814nm.fits'
        #bh_nirc2 = [416.,427.]
        #pa = np.radians(360. - 79.570702 - 6.632 - 90.)
        #itime=500.
        #gain=7.5
        #pixscale=0.04552
        #nirc2file = '/Users/kel/Documents/Projects/M31/analysis_old/align/m31_1024nm.fits'
        #bh_nirc2 = [445.,445.]
        #pa = np.radians(360. - 9.99 - 6.632 - 90.)
        #gain=7.
        #itime=700.
        #pixscale=0.04552
        #nirc2file = '/Users/kel/Documents/Projects/M31/analysis_old/align/m31_1249nm.fits'
        #bh_nirc2 = [595.,584.]
        #pa = np.radians(360. - 6.632 - 90.)
        #gain=4.
        #itime=120.
        #pixscale=0.01
        #nirc2file = '/Users/kel/Documents/Projects/M31/analysis_old/align/m31_1633nm.fits'
        #bh_nirc2 = [606.,590.]
        #pa = np.radians(360. - 6.632 - 90.)
        #gain=4.
        #itime=60.
        #pixscale=0.01
        cubeimg = pyfits.getdata(datadir + cuberoot + '_img.fits')
        nirc2 = pyfits.getdata(nirc2file)
        #nirc2 -= 316. # /N2.20090910.42407.fits
        nirc2 -= 568. # 455 value for wide, converted to 0.05 pixel scale    
        #NIRC2 FOV is 10"x10"
        rad = 6.
        #rad = 15.
        
        # radius in arcsec to match the fitted Sersic profile at
        if matchR is None:
            matchR = 5.5
        endR = 25
        #endR=70
        py.figure(2)
    
    
    # convert to e-
    nirc2 *= gain
    # convert to e-/s
    nirc2 /= itime

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
    binflux = np.zeros(len(binin))
    binsbflux = np.zeros(len(binin))
    binsbmag = np.zeros(len(binin))
    bindearea = np.zeros(len(binin))
    
    # ellipticity, Dorman 2013
    ell = 0.277
    #ell = 0.
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
        #ap = pu.EllipticalAnnulus(bh_nirc2,binin[i]/pixscale,binin[i+1]/pixscale,(binin[i+1]/pixscale)*np.sqrt(1-(ell**2.)),pa)
        # using definition of ellipticity from Dorman+2012 (not defined in Dorman+2013)
        ap = pu.EllipticalAnnulus(bh_nirc2,binin[i]/pixscale,binin[i+1]/pixscale,(binin[i+1]/pixscale)*(1.-ell),pa)
        tmpflux = pu.aperture_photometry(nirc2,ap)
        # area in pixels^2
        area = ap.area()
        # area in arcsec^2
        area *= (pixscale**2.)
        # deproject the area and flux (to match fitted SB profiles)
        dearea = area/np.cos(thetai)
        bindearea[i] = dearea
        #bindearea[i] = area
        binflux[i] = tmpflux[0][0]
        binsbflux[i] = tmpflux[0][0]/dearea
        #binsbflux[i] = tmpflux[0][0]/area
        # convert to mags
        # NIRC2 zeropoint for K band, Strehl=1, =24.63
        tmpmag = -2.5*np.log10(tmpflux[0][0]/dearea)+24.63
        #tmpmag = -2.5*np.log10(tmpflux[0][0])+24.63
        #tmpmag = -2.5*np.log10(tmpflux[0][0]/area)+24.63
        binsbmag[i] = tmpmag
        #pdb.set_trace()

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
    # constants for Courteau2011_M (I band)
    n_C11m = 1.83
    Re_C11m = 0.74*1000.
    b_n_C11m = 1.9992*n_C11m - 0.3271
    # constants for Dorman2013 (I band)
    n_D13 = 1.917
    Re_D13 = 0.778*1000.
    b_n_D13 = 1.9992*n_D13 - 0.3271
    # difference between the magnitude at the effective radius (Re) and that
    # at the given radius (matchR)
    diffmag_KB99 = -2.5*np.log10(np.exp(b_n_KB99*(((matchR*3.73/Re_KB99)**(1./n_KB99)) - 1.)))
    diffmag_C11e = -2.5*np.log10(np.exp(b_n_C11e*(((matchR*3.73/Re_C11e)**(1./n_C11e)) - 1.)))
    diffmag_C11m = -2.5*np.log10(np.exp(b_n_C11m*(((matchR*3.73/Re_C11m)**(1./n_C11m)) - 1.)))
    diffmag_D13 = -2.5*np.log10(np.exp(b_n_D13*(((matchR*3.73/Re_D13)**(1./n_D13)) - 1.)))
    # scale by this difference
    matchidx = np.where(binin == matchR)
    profMu_KB99, profI_KB99, profR_KB99 = plotLitSersic(outSB=(binsbmag[matchidx]+diffmag_KB99),plot=False)
    profMu_C11e, profI_C11e, profR_C11e = plotLitSersic(outSB=(binsbmag[matchidx]+diffmag_C11e),plot=False)
    #profMu_C11e, profI_C11e, profR_C11e = plotLitSersic(outSB=-8.61758645,plot=False)
    # outSB = binsbmag[matchidx]+diffmag_C11e (wide, flux cal) = 16.01241355 (not flux call = 16.01241355-24.63 = -8.61758645)
    profMu_C11m, profI_C11m, profR_C11m = plotLitSersic(outSB=(binsbmag[matchidx]+diffmag_C11m),plot=False)
    # outSB = binsbmag[matchidx]+diffmag_C11e (wide, not flux cal) = -9.02365025 (match R = 10); -8.96452571 (match R = 15)
    profMu_D13, profI_D13, profR_D13 = plotLitSersic(outSB=(binsbmag[matchidx]+diffmag_D13),plot=False)
    #pdb.set_trace()  
    if toplot:
        py.close(3)
        py.figure(3,figsize=(7,5))
        py.plot(binin,binsbflux,'ko')
        py.ylabel('$\mu_{K\'}$ (e$^{-}$ s$^{-1}$ arcsec$^{-2}$ (arbitrary units))')
        py.xlabel('Semimajor axis (arcsec)')
        py.plot(profR_C11e[0:endR]/3.73, profI_C11e[1,0:endR],'r-')
        pdb.set_trace()
        
        py.clf()
        py.plot(binin,binsbmag,'ko',label='Observed (K\')')
        if wide:
            #py.ylim(-14,-17)
            #py.ylim(-11,-15)
            py.ylim(13.3,10)
            py.xlim(0,18)
        else:
            py.ylim(-8,-12)
        py.ylabel('$\Sigma_{K\'}$ (mag arcsec$^{-2}$)')
        py.xlabel('Semimajor axis (arcsec)')
        #pdb.set_trace()
        
        #py.plot(profR_KB99[0:endR]/3.73, profMu_KB99[0,0:endR],'g-',label='KB99 bulge (V band)')
        #py.plot(profR_C11e[0:endR]/3.73, profMu_C11e[1,0:endR],'r-',label='C11_E bulge (3.6 $\mu$m)')
        py.plot(profR_C11m[0:endR]/3.73, profMu_C11m[2,0:endR],'b-',label='C11$_M$ bulge (I band)')
        #py.plot(profR_D13[0:endR]/3.73, profMu_D13[3,0:endR],'m-',label='D13 bulge (I band)')
        #py.legend(('Observed (Kp)','KB99 bulge (V band)','C11_E bulge (3.6 $\mu$m)', 'C11_M bulge (I band)', 'D13 bulge (I band)'))
        py.legend(loc=0)
        pdb.set_trace()

    magdiff = binsbmag[1] - profMu_C11e[1,1]
    fluxrat = 10.**(magdiff/(-2.5))

    if verbose:
        print("Flux ratio at 0.25 arcsec is ", fluxrat)
        print("Bulge percentage at 0.25 arcsec is ", 1./fluxrat)

    # get flux ratio (not SB ratio), integrated over center arcsec
    # observed flux, center arcsec
    influx = binflux[0:4].sum()
    # profI are already in SB, so convert each bin to total flux by multiplying by the area of the bin, then sum
    inKB99F = (profI_KB99[0,0:4]*bindearea[0:4]).sum()
    inC11eF = (profI_C11e[1,0:4]*bindearea[0:4]).sum()
    inC11mF = (profI_C11m[2,0:4]*bindearea[0:4]).sum()
    inD13F = (profI_D13[3,0:4]*bindearea[0:4]).sum()
    # take the flux ratio
    inrat_KB99 = influx/inKB99F
    inrat_C11e = influx/inC11eF
    inrat_C11m = influx/inC11mF
    inrat_D13 = influx/inD13F

    if verbose:
        print("Flux ratio (from C11_E) within the inner arcsec is ", inrat_C11e)
        print("Bulge percentage (KB99) within the inner arcsec is ", 1./inrat_KB99)
        print("Bulge percentage (C11_E) within the inner arcsec is ", 1./inrat_C11e)
        print("Bulge percentage (C11_M) within the inner arcsec is ", 1./inrat_C11m)
        print("Bulge percentage (D13) within the inner arcsec is ", 1./inrat_D13)

    # for testing and plotting
    return binin, binsbmag

    pdb.set_trace()
    if bulgeProfile=='KB99':
        return 1./inrat_KB99
    if bulgeProfile=='C11E':
        return 1./inrat_C11e
    if bulgeProfile=='C11M':
        return 1./inrat_C11m
    if bulgeProfile=='D13':
        return 1./inrat_D13

def dist_circle(n=None,cent=None):
    # python rewrite of astrolib's dist_circle function
    # n: size of array, either a scale (for a square array) or [x,y]
    # cent: [x,y] to calculate distance from
    # output: returns array with distance from cent (in units of pixels)
    
    if n is None:
        print('Please enter the size of the array.')
        return
    if cent is None:
        print('Please enter the coordinates to calculate distance from.')
        return

    if len(n)==2:
        nx = n[0]
        ny = n[1]
    elif len(n)==1:
        nx = n
        ny = n
    else:
        print('2D arrays only')
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

def a_ellipse(n=None,cent=None,ell=None,pa=None):
    # everything in [y,x]
    # given the parameters of an ellipse (semi-major axis length a,
    # ellipticity ell, position angle pa), plus info about the
    # array (shape n, and ellipse center coordinates cent), returns
    # an array of shape n that contains at each point the semi-major axis
    # length that describes the ellipse the given point falls on the edge of

    # ell = 1 - (b/a)
    # pa is the angle **in degrees** from the vertical to the minor
    # axis (rotated counterclockwise from the top)

    ny = n[0]
    nx = n[1]
    ycen = cent[0]
    xcen = cent[1]

    # L will contain the distance from the center
    L = np.zeros((ny,nx),dtype=float)
    # thetap contains the angle between the vector L and the vertical
    thetap = np.zeros((ny,nx),dtype=float)
    # outa is eventually the output
    outa = np.zeros((ny,nx),dtype=float)

    # setting up for the distance calc
    distx2 = (np.arange(nx) - xcen)**2
    disty2 = (np.arange(ny) - ycen)**2
    # setting up for the angle calc
    dx = np.arange(nx) - xcen
    dy = np.arange(ny) - ycen
    # loop over rows
    for i in range(int(ny)):
        L[i,:] = np.sqrt(distx2 + disty2[i])
        thetap[i,:] = np.arctan(dx/dy[i])
        #thetap[i,:] = (dy/dx[i])

    # account for the rotation of the ellipse - this is the angle
    # between the minor axis and the vector L
    theta = np.radians(pa) + thetap

    outa = L*np.sqrt((np.sin(theta)**2) + ((np.cos(theta)**2) / ((1.-ell)**2)))

    return outa

def getLitProfile2D(outSB=None,rpc2D=None,inprofile='C11E'):

    # put in a 2D array of radius values (in pc) and the scaling
    # SB (outSB), returns an array of the same size with the flux
    # of the literature profile at the given radii

    if inprofile=='KB99':
        n_b = 2.19
        R_e_kpc = 1.0
        if outSB is None:
            mu_e = 17.55
        else:
            mu_e = outSB
    if inprofile=='C11E':
        n_b = 2.18
        R_e_kpc = 0.82
        if outSB is None:
            mu_e = 15.77
        else:
            mu_e = outSB
    if inprofile=='C11M':
        n_b = 1.83
        R_e_kpc = 0.74
        if outSB is None:
            mu_e = 17.73
        else:
            mu_e = outSB
    if inprofile=='D13':
        n_b = 1.917
        R_e_kpc = 0.778
        if outSB is None:
            mu_e = 17.849
        else:
            mu_e = outSB
    
    # convert to central intensity
    I_e = 10.**(mu_e/(-2.5))
    R_e = R_e_kpc*1000.
    # 2D input radius should already be in pc
    R = rpc2D

    Rshape = R.shape

    I_b = np.zeros((Rshape[0],Rshape[1]),dtype=float)
    mu_b = np.zeros((Rshape[0],Rshape[1]),dtype=float)

    # from Cappaccioli 1989
    b_n = 1.9992*n_b - 0.3271
    negb_n = -1. * b_n
    I_b = I_e*np.exp(negb_n*( ((R/R_e)**(1./n_b)) - 1.))
    mu_b = -2.5*np.log10(I_b)

    return mu_b, I_b

def subtractBulge(inputFile=None,inputPPXF=None,bulgeProfile='C11E',matchR=5.):
    # routine to subtract the bulge spectrum from a reduced
    # data cube
    
    # pass in a reduced data cube (not tessellated yet) and
    # some parameters for the bulge profile (the profile to use
    # and matchR, the matching radius), and the routine returns
    # an output cube

    cubefits = pyfits.open(inputFile)
    
    cube = cubefits[0].data
    hdr = cubefits[0].header
    errors = cubefits[1].data
    quality = cubefits[2].data
    nframes = cubefits[3].data
    cubeimg = np.median(cube,axis=2)

    # trim spectra to match template lengths
    # Get the wavelength solution so we can specify range:
    w0 = hdr['CRVAL1']
    dw = hdr['CDELT1']
    wavelength = w0 + dw * np.arange(cube.shape[2], dtype=float)
    wavelength /= 1000.0   # Convert to microns
    # make the blue-end cut (no cut on red end)
    waveCut = 2.18
    idx = np.where(wavelength >= waveCut)[0]
    waveClip = wavelength[idx]
    cube = cube[:,:,idx]
    errors = errors[:,:,idx]
    quality = quality[:,:,idx]
    nframes = nframes[:,:,idx]
    # update header to match new blue end
    #pdb.set_trace()
    hdr['CRVAL1'] = 2180.0
    #pdb.set_trace()

    #nirc2scalefits = pyfits.open('/Users/kel/Documents/Projects/M31/analysis_old/align/m31_2125nm_osiris_rot_scale.fits')
    #nirc2scalefits = pyfits.open('/Users/kel/Documents/Projects/M31/analysis_old/align/m31_2125nm_w_osiris_rot_scale_maxpsf2.fits')
    #nirc2scalefits = pyfits.open('/Users/kel/Documents/Projects/M31/analysis_old/align/m31_2125nm_w_osiris_rot_scale_numclip15.fits')
    #nirc2scalefits = pyfits.open('/Users/kel/Documents/Projects/M31/analysis_old/align/m31_2125nm_w_osiris_rot_scale_allnewshifts.fits')
    #nirc2scalefits = pyfits.open('/Users/kel/Documents/Projects/M31/analysis_old/align/m31_2125nm_w_osiris_rot_scale_telshift.fits')
    #nirc2scalefits = pyfits.open('/Users/kel/Documents/Projects/M31/analysis_old/align/m31_2125nm_w_osiris_rot_scale_100815.fits')
    #nirc2scalefits = pyfits.open('/Users/kel/Documents/Projects/M31/analysis_old/align/m31_2125nm_w_osiris_rot_scale_100828.fits')
    #nirc2scalefits = pyfits.open('/Users/kel/Documents/Projects/M31/analysis_old/align/m31_2125nm_w_osiris_rot_scale_combshift.fits')
    #nirc2scalefits = pyfits.open('/Users/kel/Documents/Projects/M31/analysis_old/align/m31_2125nm_w_osiris_rot_scale_comb2shift.fits')
    #nirc2scalefits = pyfits.open('/Users/kel/Documents/Projects/M31/analysis_old/align/m31_2125nm_w_osiris_rot_scale_comb3shift.fits')
    #nirc2scalefits = pyfits.open('/Users/kel/Documents/Projects/M31/analysis_old/align/m31_2125nm_w_osiris_rot_scale_NIRC2_DTOTOFF.fits')
    #nirc2scalefits = pyfits.open('/Users/kel/Documents/Projects/M31/analysis_old/align/m31_2125nm_w_osiris_rot_scale_CC_mos_DTOTOFF.fits')
    #nirc2scalefits = pyfits.open('/Users/kel/Documents/Projects/M31/analysis_old/align/m31_2125nm_w_osiris_rot_scale_NIRC2_DTOTOFF_2.fits')
    #nirc2scalefits = pyfits.open('/Users/kel/Documents/Projects/M31/analysis_old/align/m31_2125nm_w_osiris_rot_scale_CC_mos_DTOTOFF_2.fits')
    nirc2scalefits = pyfits.open('/Users/kel/Documents/Projects/M31/analysis_old/align/m31_2125nm_w_osiris_rot_scale_NIRC2_DTOTOFF_no100828.fits')
    #nirc2scalefits = pyfits.open('/Users/kel/Documents/Projects/M31/analysis_old/align/m31_2125nm_w_osiris_rot_scale_NIRC2_DTOTOFF_no100828_29.fits')
    #nirc2scalefits = pyfits.open('/Users/kel/Documents/Projects/M31/analysis_old/align/m31_2125nm_w_osiris_rot_scale_CC_mos_DTOTOFF_no100828.fits')
    #nirc2scalefits = pyfits.open('/Users/kel/Documents/Projects/M31/analysis_old/align/m31_2125nm_w_osiris_rot_scale_CC_mos_DTOTOFF_no100828_100829.fits')
    
    nirc2scale = nirc2scalefits[0].data
    # convert to e-/s (wide: 5s, narrow: 60s)
    #nirc2flux = nirc2scale*4./60.
    nirc2flux = nirc2scale*4./5.
    # area of a spaxel, in spaxels (=1)
    area = 1.**2
    # in arcsec^2
    area *= (0.05**2)
    # deprojected area
    thetai = np.radians(77.5)
    dearea = area/np.cos(thetai)
    # convert to SB in e-/s/arcsec^2 (deproj)
    nirc2sb = nirc2flux/dearea
    # reorient for plotting
    nirc2sb = np.rot90(nirc2sb,3)

    cubeShape = (cube.shape[0],cube.shape[1],cube.shape[2])

    # get the bulge luminosity across the field
    # first, get the corresponding semimajor axis length for every point
    # parameters from Dorman+2013
    ell_C11=0.277
    # 360 - PA of the frame - PA of the bulge (Dorman 2013) - 90 (to convert from PA of semimajor to PA of semiminor)
    pa_C11 = 360. - 56. - 6.632 - 90.
    # returns semi-major axis length in pixels
    a_all = a_ellipse(n=[cubeShape[0],cubeShape[1]],cent=[bhpos_pix[1],bhpos_pix[0]],ell=ell_C11,pa=pa_C11)
    # OSIRIS: 1 pixel = 0.05"
    aarcsec = a_all*0.05
    # M31: 1" = 3.73 pc
    apc = aarcsec*3.73
    # then get the bulge contribution at each semimajor axis length
    # for now, hard-coding in outSB (from C11E, matchR=5.5)
    #outSB_C11E = -4.76470713
    # area is *not* deprojected
    #outSB_C11E = -4.8263725
    # using wide camera image, matchR=10., C11E (binsbmag[matchidx]+diffmag_C11e)
    outSB_C11E = -8.61758645
    outSB_C11M = -9.02365025 #(matchR = 10)
    #outSB_C11M = -8.96452571 # (matchR = 15)
    #outSB_C11E = 16.01241355 # can't use flux calibrated, as we're in intensities, not magitudes, here
    # returns mag, SB
    #litMu, litI = getLitProfile2D(outSB=outSB_C11E,rpc2D=apc,inprofile='C11E')
    litMu, litI = getLitProfile2D(outSB=outSB_C11M,rpc2D=apc,inprofile='C11M')
    # reorient lit maps
    litMu = np.rot90(litMu.T,3)
    litI = np.rot90(litI.T,3)
    
    # smooth the bulge and NIRC2 maps by the seeing
    PSFparams = ppxf_m31.readPSFparams('/Users/kel/Documents/Projects/M31/data/osiris_mosaics/drf/sigclip/all_NIRC2_CC_DTOTOFF_2/no100828/data/osiris_perf/sig_1.05/params.txt',twoGauss=False)
    PSFsig = PSFparams.sig1[0]
    gauss = ifu.gauss_kernel(PSFparams.sig1[0],PSFparams.amp1[0],half_box=PSFsig*5.)
    litIsm = signal.convolve2d(litI,gauss,mode='same',boundary='wrap')
    nirc2sbsm = signal.convolve2d(nirc2sb,gauss,mode='same',boundary='wrap')

    # take ratio of bulge_lit to observed to get SB ratio map
    sbratiomap = litIsm/nirc2sbsm

    #pdb.set_trace()
    # plot the SB ratio map
    py.close(2)
    py.figure(2,figsize=(8,3))
    xaxis = (np.arange(sbratiomap.shape[1], dtype=float) - bhpos_pix[0])*0.05
    yaxis = (np.arange(sbratiomap.shape[0], dtype=float) - bhpos_pix[1])*0.05
    py.imshow(sbratiomap,extent=[xaxis[0],xaxis[-1],yaxis[0],yaxis[-1]])
    py.plot([0],'kx',markeredgewidth=2)
    py.axis('image')
    cbar = py.colorbar(orientation='vertical',ticks=[.15,.3,.45])
    cbar.set_label('$\Sigma$ ratio')
    
    pdb.set_trace()
    
    # make the model bulge spectrum
    # first read in the ppxf files to get the template weights
    pIn=ppxf_m31.PPXFresults(inputPPXF,bestfit=True)
    tw = pIn.tweights
    tw = np.nan_to_num(tw)
    # changed bulge spaxel selection to be spaxels that are above a certain bulge ratio
    #bfrac = 0.5
    #bfrac = 0.45
    bfrac = 0.42
    bulgeidx = np.where((sbratiomap >= bfrac) & (cubeimg > 0) & (tw.sum(axis=2) != 0))
    nobulgeidx = np.where(sbratiomap < bfrac)
    # scale by the luminosity contribution
    # match the bulge spectrum median flux to that of the science spectrum
    medsciflux = np.median(cube,axis=2)
    # get the cube's model templates
    # normalize so the sum of the weights in each spaxel = 1
    logWaveSpec, modSpecCubeOut = ppxf_m31.create_model_templates(tw,norm=False,rebinWave=False)#,pWeights=pIn.pweights)#,normmask=medsciflux)
    # for plotting, later
    twnorm = np.zeros(tw.shape)
    twsum = tw.sum(axis=2)
    for i in range(tw.shape[2]):
        twnorm[:,:,i] = tw[:,:,i]/twsum
    # clip the red end to match the cube (same wavelength scale and
    # blue cut off, so can drop the wavelength vector)
    #modSpecCube = modSpecCube[:,:,0:cubeShape[2]]
    waveSpec = waveClip
    # interpolate templates onto the same wavelength grid
    modSpecCubeNoCont = np.zeros([modSpecCubeOut.shape[0],modSpecCubeOut.shape[1],cube.shape[2]])
    for i in np.arange(modSpecCubeOut.shape[0]):
        for j in np.arange(modSpecCubeOut.shape[1]):
            tck = scipy.interpolate.splrep(logWaveSpec, modSpecCubeOut[i,j,:], s=0)
            modSpecCubeNoCont[i,j,:] = scipy.interpolate.splev(waveClip, tck)
    # add the ppxf continuum to the templates (should about match the science spaxels now, except for the LOSVD)
    modSpecCube = np.zeros([modSpecCubeOut.shape[0],modSpecCubeOut.shape[1],cube.shape[2]])
    x = np.linspace(-1, 1, len(waveClip))
    for i in range(modSpecCubeOut.shape[0]):
        for j in range(modSpecCubeOut.shape[1]):
            apoly = np.polynomial.legendre.legval(x, pIn.pweights[i,j,:])
            modSpecCube[i,j,:] = modSpecCubeNoCont[i,j,:] + apoly
    # normalize, to compare line depths
    modSpecCubeNorm = np.zeros(modSpecCube.shape)
    for i in range(modSpecCube.shape[2]):
        modSpecCubeNorm[:,:,i] = modSpecCube[:,:,i] / np.median(modSpecCube,axis=2)
    #pdb.set_trace()
    modSpec = np.median(modSpecCubeNorm[bulgeidx[0],bulgeidx[1],:],axis=0)
        
    
    #pdb.set_trace()
    # set the velocity
    # have two choices - systemic velocity, or scaled w/ distance from SMBH (probably same, within errors)
    # going w/ systemic velocity for now
    # velocity in km/s
    #v = -308.
    v = -1.*ppxf_m31.vsys
    # convert to pixels - manually calc v/pixel
    vScale = cc.c*((waveSpec[401]-waveSpec[400])/waveSpec[400])
    vPix = v/vScale
    
    # set the dispersion
    # sticking with a single value for now
    # grabbing the value from the edge of the data cube (tessellated)
    # first in km/s
    disp = 110.
    #disp = 150.
    # now in pixels
    dispPix = disp/vScale

    # if using a single velocity/dispersion, can avoid for loops
    # create the bulge LOSVD kernel from the set velocity/dispersion (assuming Gaussian)
    # (partly adapted from ppxf.py)
    # create a window at least 5*sigma wide to avoid edge effects
    dx = int(np.ceil(np.max(abs(vPix)+5*dispPix)))
    nl = 2*dx+1
    x = np.linspace(-dx,dx,nl)
    # fill in the kernel with a Gaussian
    w = (x-vPix)/dispPix
    w2 = w**2
    gausstmp = np.exp(-0.5*w2)
    # normalize
    gausskern = gausstmp/gausstmp.sum()

    # also convolve by the diff between the OSIRIS and GNIRS resolution, before the LOSVD
    
    # convolve the model bulge spectrum by the LOSVD Gaussian
    newSpec = signal.convolve(modSpec, gausskern, mode='same')

    # flip the ratio map to match the orientation of the cube
    sbratiomap = np.rot90(sbratiomap.T,3)
    # combine the scaling factors
    totscale = medsciflux * sbratiomap
    # make a 3D array of the ratio map, with the same factor at each spectral channel
    #tmp1 = np.tile(sbratiomap,(len(waveSpec),1,1))
    tmp1 = np.tile(totscale,(len(waveSpec),1,1))
    # swap the axes around so it's the correct dimensions
    tmp2 = np.swapaxes(tmp1,0,2)
    ratiomap3d = np.swapaxes(tmp2,0,1)
    
    specScale = ratiomap3d*newSpec

    # subtract from the data cube
    newCube = cube - specScale

    # example plot for a single spaxel
    py.close(1)
    py.figure(1,figsize=(7,5))
    py.subplots_adjust(left=0.14, right=0.94, top=0.95,bottom=.15)
    py.plot(waveClip,cube[20,40,:],'b-',label='Original science spectrum')
    py.plot(waveClip,specScale[20,40,:],'g-',label='Scaled bulge spectrum')
    py.plot(waveClip,newCube[20,40,:],'r-',label='Bulge-subtracted science spectrum')
    py.xlim(waveClip[0],waveClip[-1])
    py.ylim(0,.65)
    py.xlabel('Wavelength ($\mu$m)')
    py.ylabel('Flux (DN s$^{-1}$)')
    py.legend(loc=0)
    
    pdb.set_trace()

    mask0 = np.where(cube[:,:,500] == 0.)
    newCube[mask0[0],mask0[1],:] = 0.

    #pdb.set_trace()
    # how to do the errors?
    outFile = inputFile.replace('.fits', '_bulgesub.fits')
    pyfits.writeto(outFile, newCube, header=hdr, clobber=True,output_verify='warn')
    pyfits.append(outFile,errors)
    pyfits.append(outFile,quality)
    pyfits.append(outFile,nframes)
    
    #pdb.set_trace()

def plotHM80():
    # data from Hoessel & Melnick 1980 Table 1

    g = np.array([17.04,17.54,17.98,18.33,18.6,18.84,19.07,19.26,19.41,19.59,19.73,19.86,19.97,20.09,20.16,20.23,20.38,20.46,20.52,20.58,20.47,19.86,20.,20.82,20.85,20.91,20.96,20.95,20.97,21.,20.93,20.92,21.06,21.1,21.13,21.18,21.2,21.23,21.28,21.3,21.34,21.42,21.42,21.45,21.5,21.54,21.57,21.78,21.88,21.89,21.84,21.83,21.79,21.8,21.9,21.89,22.02,22.04,22.06,21.89,21.92,22.07,22.19,22.21,22.13,22.12,22.16,22.16,22.11,22.1,22.13,22.13,22.14,22.05,22.06,22.12,22.15,22.17,22.25,22.34,22.39,22.33,27.08,22.16,22.19,22.18,22.03,21.81,21.57,21.43,21.51,21.61,21.76,21.68,21.81,21.98,22.32])

    gr=np.array([.56,.57,.55,.55,.56,.57,.56,.57,.57,.56,.56,.54,.54,.55,.52,.53,.55,.53,.52,.52,.39,.08,.24,.65,.53,.55,.56,.53,.53,.53,.45,.48,.53,.5,.49,.51,.5,.5,.52,.5,.52,.53,.49,.5,.5,.49,.51,.59,.56,.54,.47,.5,.47,.46,.53,.48,.57,.58,.54,.45,.56,.63,.63,.57,.5,.53,.58,.53,.48,.53,.54,.52,.51,.48,.53,.51,.51,.59,.59,.65,.61,.56,.44,.54,.49,.49,.34,.27,.17,.25,.33,.34,.39,.33,.43,.51,.57])

    r=g-gr
    # in arcsec
    rad=np.arange(13.1,2555.,26.3)

    py.plot(rad,g,'go')
    py.plot(rad,r,'ro')
    py.axvspan(5.5,300,facecolor='c',alpha=.5)
    py.legend(('G','R','KB99 bulge range'))
    
    py.ylim(23,16)
    py.xlim(0,2500)
    py.xlabel('arcsec')
    py.ylabel('$\mu$ (mag arcsec$^{-2}$)')
    py.title('Hoessel and Melnick 1980, Palomar obs')
    
