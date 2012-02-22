import os
import sys
#import casac
import taskutil
from math import *
import string
from odict import odict
#import pdb


#from mrange import * # Multidimensional iterators from sage.

def radplot(img=None,
            wc=None,
            maxrad=None,
            rs=None,
            azav=None,
            runit=None,
            title=None,
            rlab=None,
            azavlab=None):
    """
    Plots the azimuthal average of an image about a point.

    For now it assumes you are only interested in the first plane of img, and
    that it has square pixels.

    Keyword arguments:
    img  -- Name of image to made a radial plot of.
    wc   -- World coordinate to use as center.
    rs   -- An optional list or array of radii to use instead of calculating.
    azav -- An optional list or array of azimuthal averages to use instead of calculating.
    blunit -- Unit of the radii.  Derived from img if necessary.
    title -- Title for the plot.  Defaults to 'Azimuthal average of %s about %s' % (img, wc).
    rlab -- Label for the radial axis.  Defaults to 'Radius (runit)'.
    azavlab -- Label for the azimuthal average axis.  Defaults to 'Azimuthal average (azavunit)'
    """
    myf = taskutil.get_global_namespace()
    ia = myf['ia']
    pl = myf['pl']

    myf['taskname'] = 'radplot'

    #taskutil.update_myf(myf)
    
##     # Set all task parameters to their global value
##     img     = myf['img']   
##     wc      = myf['wc']    
##     maxrad  = myf['maxrad']
##     rs      = myf['rs']    
##     azav    = myf['azav']  
##     runit   = myf['runit'] 
##     title   = myf['title'] 
##     rlab    = myf['rlab']  
##     azavlab = myf['azavlab']

    casalog = myf['casalog'] # include the logger

##     #Add type/menu/range error checking here
##     #	if type(mask)==str: mask=[mask]
##     arg_desc = {}
##     #         Name        Variable     Allowed type(s)
##     arg_desc['img']     = (img,	       str)
##     arg_desc['wc']      = (wc,	       (str,list,dict))
##     arg_desc['maxrad']  = (maxrad,     (str, dict, list))
##    # arg_desc['rs']      = (rs,	       (list, array))
##    # arg_desc['azav']    = (azav,       (list, array))  
##     arg_desc['runit']   = (runit,      str)
##     arg_desc['title']   = (title,      str)
##     arg_desc['rlab']    = (rlab,       str)
##     arg_desc['azavlab'] = (azavlab,    str) 
##     if(taskutils.check_param_types(myf['taskname'], arg_desc)):
##         return

    taskutil.startlog(casalog, myf['taskname'])
    
    try:
        if rs == None or azav == None:
            rs, azav, wts, runit = radav(img, wc, maxrad)
        
            pl.clf()
            pl.ion()
            pl.plot(rs, azav)

            ia.open(img)
            imgsummary = ia.summary()
            ia.close()
            imgsummary = imgsummary['header']

            if azavlab == None:
                azavlab = 'Azimuthal average (%s)' % imgsummary['unit']
            pl.ylabel(azavlab)

            if rlab == None:
                rlab = 'Radius'
            if runit == None:
                runit = imgsummary['axisunits'][0]
            pl.xlabel('%s (%s)' % (rlab, runit))

            if title == None:
                title = 'Azimuthal average of %s about %s' % (img, wc)
            pl.title(title)
    except TypeError, e:
        print myf['taskname'], "-- TypeError: ", e
        taskutil.endlog(casalog, myf['taskname'])
        return
    except ValueError, e:
        print myf['taskname'], "-- OptionError: ", e
        taskutil.endlog(casalog, myf['taskname'])
        return
    except Exception, instance:
        print '***Error***', instance
        taskutil.endlog(casalog, myf['taskname'])
        return
    taskutil.endlog(casalog, myf['taskname'])

##     saveinputs = myf['saveinputs']
##     saveinputs(myf['taskname'], myf['taskname'] + '.last')

###End of radplot

def radplot_defaults(param = None):
    myf = taskutil.get_global_namespace()
    
    a = odict()
    a['img']     = None
    a['wc']      = None
    a['rs']      = None
    a['azav']    = None
    a['blunit']  = None
    a['title']   = None
    a['rlab']    = None
    a['azavlab'] = None

    if(param == None):
        myf['__set_default_parameters'](a)
    elif(param == 'paramkeys'):
        return a.keys()
    else:
        if(a.has_key(param)):
            return a[param]

def radplot_description(key='radplot', subkey = None):
    desc = {'radplot': 'Plots the azimuthal average of an image about a point.',
            'img':     'Name of image to made a radial plot of.',
            'wc':      "World coordinate to use as center.",
            'rs':      "An optional list or array of radii to use instead of calculating.",
            'azav':    "An optional list or array of azimuthal averages to use instead of calculating.",
            'blunit':  "Unit of the radii.  Derived from img if necessary.",
            'title':   "Plot title.  Defaults to 'Azimuthal average of %s about %s' % (img, wc).",
            'rlab':    "Radial axis label.  Defaults to 'Radius (runit)'.",
            'azavlab': "Azimuthal average axis label.  Defaults to 'Azimuthal average (azavunit)'"
            }
    if(desc.has_key(key)):
        return desc[key]
    return ''

def radplot_check_params(param=None, value=None):
    if(param=='img'):
        return ((type(value) == str) & os.path.exists(value))
    elif(param in ('wc', 'maxrad')):
        if(type(value) in (str, list, dict)):
            return True
        else:
            return False
    elif(param in ('rs', 'azav')):
        if(type(value) in (list, array)):
            return True
        else:
            return False
    elif(param in ('runit', 'title', 'rlab', 'azavlab')):
        return (type(value) == str)
    else:
        return True

def radav(img, wc, maxrad=None):
    """
    Returns the azimuthal average of image with name img about the world
    coordinate wc, and the weights and baseline unit used to calculate that
    average.  The baseline unit is taken from maxrad, or from img's axisunits
    if maxrad is not specified.

    For now it assumes you are only interested in the first plane of img, and
    that it has square pixels.
    """
    myf = taskutil.get_global_namespace()
    ia = myf['ia']
    pl = myf['pl']
    tb = myf['tb']

    ia.open(img)
    imgsummary = ia.summary()['header']
    imgshape   = imgsummary['shape']
    
    # For some unknown reason topixel returns a dictionary containing an array
    # of floats.
    centpix = ia.topixel(wc)['numeric'].tolist()

##     # Figure out the shape of the azimuthal average.

##     # Iterate over the non-xy axes.
##     atleastoneplane = imgshape[2:]
##     if len(atleastoneplane) < 1:      # It was a simple 2D image.
##         atleastoneplane
##     for plane in xmrange_iter([1], ):

    if maxrad == None:
        maxdelta = [max(abs(imgshape[i] - centpix[i]),
                        abs(centpix[i]))**2 for i in [0, 1]]
        maxdelta = int(pl.ceil(sqrt(sum(maxdelta)) + 1.5))
        runit = imgsummary['axisunits'][0]
    else:
        maxdelta, runit = maxrad.split()
        if runit != 'pixels':
            maxdelta = int(pl.ceil(abs(float(maxdelta)/imgsummary['incr'][1]) + 1.5))

    # We could use a lot of pixelvalue()s here, but that would be slow.  Close
    # up and reopen as a tb.
    ia.close()
    tb.open(img)
    imgarr = tb.getcol('map')
    tb.close()
            
    azav = pl.zeros([maxdelta], dtype=float)
    wts  = pl.zeros([maxdelta], dtype=float)

    cx, cy = centpix[0:2]
    firstplane = tuple(pl.zeros([len(imgshape[2:]) + 1]).tolist())

    # Set up some numbers for gridding the image pixels onto the radial plot.
    # Instead of calculating the angle between the radius vector and an x or y
    # axis, treat it as pi/8 on average.
    oosqrt2 = 1.0 / sqrt(2.0)
    secpio8 = 1.0 / sqrt(0.5 * (1.0 + oosqrt2))

    secpio8or2 = oosqrt2 * secpio8
    
    # cos(pi/8)/sqrt(2)
    ohs1p1or2 = 0.5 * sqrt(1 + oosqrt2)
    
    # sin(pi/8)/sqrt(2)
    ohs1m1or2 = 0.5 * sqrt(1 - oosqrt2)

    minx = int(pl.ceil(max(0, cx - maxdelta)))
    maxx = int(pl.floor(min(imgshape[0], cx + maxdelta)))
    Miny = int(pl.ceil(max(0, cy - maxdelta)))
    Maxy = int(pl.floor(min(imgshape[1], cy + maxdelta)))
    maxd2 = maxdelta**2
    maxdelta -= 1
    for i in xrange(minx, maxx):
        possxspread = sqrt(maxd2 - (i - cx)**2)
        miny = max(Miny, int(pl.ceil(cy - possxspread)))
        maxy = min(Maxy, int(pl.floor(cy + possxspread)))
        for j in xrange(miny, maxy):
            # I don't understand why a[(0, 0, 0)] != a[[0, 0, 0]], but it does.
            pv = imgarr[i, j][firstplane]
            rad = sqrt((i - cx)**2 + (j - cy)**2)
            if rad > 0.8 and rad < maxdelta:
                pi = int(pl.floor(rad))

                r = rad - pi
                if r < ohs1p1or2:
                    if r > ohs1m1or2:
                        wnear = sqrt(2) * (r - ohs1p1or2)**2
                    else:
                        wnear = 0.5 - secpio8 * r
                else:
                    wnear = 0.0
                if r > 1.0 - ohs1p1or2:
                    if r < 1.0 - ohs1m1or2:
                        wfar = sqrt(2) * (r + ohs1p1or2 - 1.0)**2
                    else:
                        wfar = 0.5 - secpio8 * (1.0 - r)
                else:
                    wfar = 0.0
                w0 = 1.0 - wnear - wfar

                #pi = int(pl.floor(rad + 0.5))
                try:
                    azav[pi - 1] += wnear * pv
                    wts[pi - 1]  += wnear
                    azav[pi]     += w0 * pv
                    wts[pi]      += w0
                    azav[pi + 1] += wfar * pv
                    wts[pi + 1]  += wfar
                except IndexError:
                    print "pi:", pi
                    print "rad:", rad
                    print "maxdelta:", maxdelta
            else:                                # This *is* the center pixel.
                azav[0] += pv
                wts[0]  += 1.0
                
    for i in xrange(0, len(wts)):
        azav[i] /= wts[i]

    rs = [i for i in xrange(0, len(wts))]

    if runit != 'pixels':
        pixscale = abs(imgsummary['incr'][1])
        for i in xrange(0, len(wts)):
            rs[i] *= pixscale
            
    return rs, azav, wts, runit
