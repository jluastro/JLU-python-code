import os, sys, math
import numpy as np
import pylab as py

def rPix2Arc(xpix, ypix, trans, absolute=0, relErr=1):
    """Convert positional pixel value to positions in arcseconds
    offset from Sgr A*.

    @param xpix: X position in pixels
    @type xpix: float
    @param ypix: Y position in pixels
    @type ypix: float
    @param trans: The Transform.class object that holds coordinate 
        transformation information.
    @type trans: Transform
    @param absolute: Set to 1 in order to rotate coordinates
    @type absoulte: boolean or integer
    @param relErr: Set to 1 to only exclude absolute astrometric errors.
    @type relErr: boolean or integer
    """
    # Check that this epoch has valid data.
    # If invalid, then just return the same place holder values.
    if (xpix < -1000 and ypix < -1000):
        return (xpix, ypix)

    # Handle converting into Sgr A* = 0,0 frame
    x1 = xpix - trans.sgra[0]
    y1 = ypix - trans.sgra[1]

    # Now handle scaling
    x2 = x1 * -trans.scale
    y2 = y1 * trans.scale

    # Handle angles but only if requested by absolute=1 keyword
    if absolute == 1:
	cosa = math.cos(trans.angle)
	sina = math.sin(trans.angle)

	x3 = (x2 * cosa) + (y2 * sina)
	y3 = -(x2 * sina) + (y2 * cosa)
    else:
	x3 = x2
	y3 = y2

    # Get proper errors (take sqrt)
    x = x3
    y = y3

    return (x, y)

def rerrPix2Arc(xpix, ypix, xpixErr, ypixErr, trans, absolute=0, relErr=1):
    """Convert positional pixel values (and errors) to positions/errors
    in arcseconds offset from Sgr A*.

    @param xpix: X position in pixels
    @type xpix: float
    @param ypix: Y position in pixels
    @type ypix: float
    @param xpixErr: X positional error in pixels
    @type xpixErr: float
    @param ypixErr: Y positional error in pixels
    @type ypixErr: float
    @param trans: The Transform.class object that holds coordinate 
        transformation information.
    @type trans: Transform
    @param absolute: Set to 1 in order to rotate coordinates
    @type absoulte: boolean or integer
    @param relErr: Set to 1 to only exclude absolute astrometric errors.
    @type relErr: boolean or integer
    """
    # Check that this epoch has valid data.
    # If invalid, then just return the same place holder values.
    if (xpix <= -1000 and ypix <= -1000):
        return (xpix, ypix, xpixErr, ypixErr)
    
    # Handle converting into Sgr A* = 0,0 frame
    x1 = xpix - trans.sgra[0]
    y1 = ypix - trans.sgra[1]

    xerr1 = xpixErr**2
    yerr1 = ypixErr**2
    if relErr != 1:
	xerr1 += trans.sgraErr[0]**2
	yerr1 += trans.sgraErr[1]**2

    # Now handle scaling
    x2 = x1 * -trans.scale
    y2 = y1 * trans.scale

    xerr2 = xerr1 * trans.scale**2
    yerr2 = yerr1 * trans.scale**2
    if relErr != 1:
	xerr2 += (x1 * trans.scaleErr)**2
	yerr2 += (y1 * trans.scaleErr)**2
    
    # Handle angles but only if requested by absolute=1 keyword
    if absolute == 1:
	cosa = math.cos(trans.angle)
	sina = math.sin(trans.angle)

	x3 = (x2 * cosa) + (y2 * sina)
	y3 = -(x2 * sina) + (y2 * cosa)

	xerr3 = (xerr2 * cosa**2) + (yerr2 * sina**2)
	yerr3 = (xerr2 * sina**2) + (yerr2 * cosa**2)
	if relErr != 1:
	    xerr3 += (trans.angleErr * y2)**2
	    yerr3 += (trans.angleErr * x2)**2
    else:
	x3 = x2
	y3 = y2
	xerr3 = xerr2
	yerr3 = yerr2

    # Get proper errors (take sqrt)
    x = x3
    y = y3
    xerr = math.sqrt(xerr3)
    yerr = math.sqrt(yerr3)

    return (x, y, xerr, yerr)


def vPix2Arc(xpix, ypix, trans, absolute=0, relErr=1):
    """Convert positional pixel value to positions in arcseconds
    offset from Sgr A*"""
    # Setup
    x1 = xpix
    y1 = ypix

    # Now handle scaling
    x2 = x1 * -trans.scale
    y2 = y1 * trans.scale

    # Handle angles but only if requested by absolute=1 keyword
    if absolute == 1:
	cosa = math.cos(trans.angle)
	sina = math.sin(trans.angle)

	x3 = (x2 * cosa) + (y2 * sina)
	y3 = -(x2 * sina) + (y2 * cosa)
    else:
	x3 = x2
	y3 = y2

    # Get proper errors (take sqrt)
    x = x3
    y = y3

    return (x, y)

def verrPix2Arc(xpix, ypix, xpixErr, ypixErr, trans, absolute=0, relErr=1):
    # Setup
    x1 = xpix
    y1 = ypix
    xerr1 = xpixErr
    yerr1 = ypixErr

    # Now handle scaling
    x2 = x1 * -trans.scale
    y2 = y1 * trans.scale

    xerr2 = (xerr1 * trans.scale)**2
    yerr2 = (yerr1 * trans.scale)**2
    if relErr != 1:
	xerr2 += (x1 * trans.scaleErr)**2
	yerr2 += (y1 * trans.scaleErr)**2
    
    # Handle angles but only if requested by absolute=1 keyword
    if absolute == 1:
	cosa = math.cos(trans.angle)
	sina = math.sin(trans.angle)

	x3 = (x2 * cosa) + (y2 * sina)
	y3 = -(x2 * sina) + (y2 * cosa)

	xerr3 = (xerr2 * cosa**2) + (yerr2 * sina**2)
	yerr3 = (xerr2 * sina**2) + (yerr2 * cosa**2)
	if relErr != 1:
	    xerr3 += (trans.angleErr * y2)**2
	    yerr3 += (trans.angleErr * x2)**2
    else:
	x3 = x2
	y3 = y2
	xerr3 = xerr2
	yerr3 = yerr2

    # Get proper errors 
    x = x3
    y = y3
    xerr = math.sqrt(xerr3)
    yerr = math.sqrt(yerr3)

    return (x, y, xerr, yerr)

def aerrPix2Arc(xpix, ypix, xpixErr, ypixErr, trans, absolute=0, relErr=1):
    # Setup
    x1 = xpix
    y1 = ypix
    xerr1 = xpixErr
    yerr1 = ypixErr

    # Now handle scaling
    x2 = x1 * -trans.scale
    y2 = y1 * trans.scale

    xerr2 = (xerr1 * trans.scale)**2
    yerr2 = (yerr1 * trans.scale)**2
    if relErr != 1:
	xerr2 += (x1 * trans.scaleErr)**2
	yerr2 += (y1 * trans.scaleErr)**2
    
    # Handle angles but only if requested by absolute=1 keyword
    if absolute == 1:
	cosa = math.cos(trans.angle)
	sina = math.sin(trans.angle)

	x3 = (x2 * cosa) + (y2 * sina)
	y3 = -(x2 * sina) + (y2 * cosa)

	xerr3 = (xerr2 * cosa**2) + (yerr2 * sina**2)
	yerr3 = (xerr2 * sina**2) + (yerr2 * cosa**2)
	if relErr != 1:
	    xerr3 += (trans.angleErr * y2)**2
	    yerr3 += (trans.angleErr * x2)**2
    else:
	x3 = x2
	y3 = y2
	xerr3 = xerr2
	yerr3 = yerr2

    # Get proper errors 
    x = x3
    y = y3
    xerr = math.sqrt(xerr3)
    yerr = math.sqrt(yerr3)

    return (x, y, xerr, yerr)


def errPix2Arc(value, error, trans, relErr=1):
    newval = value * trans.scale
    newerr = (error * trans.scale)**2
    if relErr != 1:
	newerr += (value * trans.scaleErr)**2

    newerr = math.sqrt(newerr)

    return (newval, newerr)

def xy2circ(x, y, vx, vy):
    """Convert 2D cartesion coordinates to circular coordinates.

    @param pos Position vector
    @type pos Numarray array
    @param vec Vector at the position
    @type vec Numarray array
    """
    # Determine the magnitude of the position vector.
    magPos = np.sqrt(x**2 + y**2)

    # Radial component comes from the dot product.
    radial = (x*vx + y*vy) / magPos

    # Tangential component comes from the cross product with z.
    tangen = ((vx * y) - (vy * x)) / magPos

    return (radial, tangen)

def xy2circErr(x, y, vx, vy, xerr, yerr, vxerr, vyerr):
    """Convert 2D cartesion coordinates to circular coordinates.

    @param pos Position vector
    @type pos Numarray array
    @param vec Vector at the position
    @type vec Numarray array
    """
    # Determine the magnitude of the position vector.
    r = np.sqrt(x**2 + y**2)

    (vr, vt) = xy2circ(x, y, vx, vy)

    # Compute uncertainties
    vrerr =  (vxerr * x/r)**2 + (vyerr * y/r)**2
    vrerr += (y * xerr * vt/r**2)**2 + (x * yerr * vt/r**2)**2
    vrerr =  np.sqrt(vrerr)
    vterr =  (vxerr * y/r)**2 + (vyerr * x/r)**2
    vterr += (y * xerr * vr/r**2)**2 + (x * yerr * vr/r**2)**2
    vterr =  np.sqrt(vterr)

    return (vr, vt, vrerr, vterr)


def cross_product(a, b):
    c = np.arange(3, dtype=float)

    c[0] = (a[1] * b[2]) - (a[2] * b[1])
    c[1] = (a[2] * b[0]) - (a[0] * b[2])
    c[2] = (a[0] * b[1]) - (a[1] * b[0])
    
    return c

def usetexTrue():
    py.rc('text', usetex=True)
    #rc('font', family='serif', size=16)
    py.rc('font', **{'family':'sans-serif', 'size':16})
    py.rc('axes', titlesize=20, labelsize=20)
    py.rc('xtick', labelsize=16)
    py.rc('ytick', labelsize=16)

def usetexFalse():
    py.rc('text', usetex=False)
    py.rc('font', family='sans-serif', size=14)
    py.rc('axes', titlesize=16, labelsize=16)
    py.rc('xtick', labelsize=14)
    py.rc('ytick', labelsize=14)
