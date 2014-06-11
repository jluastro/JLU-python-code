import math, os, shutil
import pylab, pyfits
import asciidata
import matplotlib as mlib
from matplotlib.colors import colorConverter
from gcwork import starTables
from gcwork import starset
from gcwork import orbits
from gcwork import objects
from gcwork.plotgc import gccolors 
from numarray import *
from numpy import *
import numarray as na
import matplotlib.colors
import pdb

def newSources(root='./', align='align/align_d_rms_t', poly='polyfit_d/fit',
               orbitFile='orbits_movie.dat',flatten=False,yrlyPts=False):
    s = getOrbitStars(orbitFile=orbitFile,
                      root=root, align=align, poly=poly, flatten=flatten)

    print s.t.scale, s.t.sgra

    # Determine the mass assuming a distance of 8.46 kpc
    star0orb = s.stars[0].orbit
    dist = 8460.0 # in parsec
    axis = (star0orb.a / 1000.0) * dist # in au
    mass = (axis)**3 / star0orb.p**2
    print 'Mass = %.2e Msun    Dist = %4d pc' % (mass, dist)

    # Make a color table
    #colors = pylab.cm.Dark2(na.arange(0, 1, 1.0/(len(s.stars))))
    colors = ['orange', 'red', 'yellowgreen', 'brown', 'turquoise', 'steelblue', 'plum',
              'purple', 'blue', 'navy', 'magenta', 'green', 'purple', 'mediumorchid',
              'deeppink', 'olive', 'tomato', 'salmon', 'rosybrown', 'sienna', 'cyan']*4

    names = s.getArray('name')

    ## Three sets of stars
    ##    -- constrains Ro today
    ##    -- constrains Ro/extended mass in future (radial velocities)
    ##    -- constrains extended mass in future (no radial velocities)
    # Assume the stars are in increasing order by name
    stars1 = ['S0-2', 'S0-16', 'S0-19', 'S0-1', 'S0-20']
    stars2 = []
    stars3 = []

    roundT = []
    # All stars3 come after S0-26
    foundS026 = False
    for name in names:
        try:
            idx = stars1.index(name)
            continue
        except ValueError:
            if foundS026 == False:
                stars2.append(name)
                if name == 'S0-26':
                    foundS026 = True
            else:
                stars3.append(name)

    # Keep a function that does the orbit plotting
    def plotStar(star, time='orbit', color='k', alpha=1.0):
        # Choices for time are
        #   'orbit' = plot the whole orbit
        #   'obs' = duration of observation
        #   'extend' = duration of observation plus out to 2017.5
        orb = star.orbit

        # Determine time steps
        x = star.getArrayAllEpochs('x')
        y = star.getArrayAllEpochs('y')

        if (time == 'orbit'):
            t = na.arange(orb.t0+0.01, orb.t0+orb.p+0.11, orb.p/200.0, type=na.Float)
        if (time == 'obs'):
            idx = ( na.where(x != -1000) )[0]
            t = na.arange(math.floor(star.years[idx[0]]),
                          math.ceil(star.years[idx[-1]]),
                          0.1, type=na.Float)
        if (time == 'extend'):
            idx = ( na.where(x != -1000) )[0]
            t = na.arange(math.floor(star.years[idx[0]]),
                          2017.5,
                          0.1, type=na.Float)

        (r, v, a) = orb.kep2xyz(t, mass=mass, dist=dist)
        
        pp = pylab.plot(r[:,0], r[:,1], color=color, alpha=alpha)
        # To plot no line and just the data points:
        #pp = pylab.plot([], [],color=color)

        ##
        ## Now plot the actual data points
        ##
        # Load from points files
        if yrlyPts:
            # Just take the points from 'r' array spaced about a year apart
            roundT = array([int(round(t[qq])) for qq in range(len(t))])
            firstT = roundT[0]
            tPts = [roundT.searchsorted(firstT+zz+1) for zz in range(roundT[-1]-firstT)]
            c = pylab.scatter(r[tPts,0], r[tPts,1], 20.0, color, marker='o', faceted=False)
        else:
            # Get the actual data
            c = pylab.scatter(x, y, 20.0, color, marker='o', faceted=False)

        c.set_alpha(alpha)

        return pp
        
        
    def layer(stars, alpha=1.0, time='obs'):
        lines = []

        for ii in range(len(stars)):
            idx = names.index(stars[ii])
            star = s.stars[idx]
            color = colors[idx]
            
            pp = plotStar(star, time=time, color=color, alpha=alpha)
            lines.append(pp)

        if (alpha == 1.0):
            lgd = pylab.legend(lines, stars, numpoints=1,loc=(0.9,0))
            ltext = lgd.get_texts()
            llines = lgd.get_lines()
            pylab.setp(ltext, fontsize=10)
            pylab.setp(llines, marker='o')


    ##########
    #
    # Plotting First Layer
    #
    ##########
    # Use LATEX for the axis labels
    usetexTrue()

    alphaOn = 1.0
    alphaOff = 0.1

    def newFigLayer():
        pylab.clf()
        pylab.figure(figsize=(8,8))
        pylab.axes([0.15, 0.15, 0.8, 0.81])
        pylab.axis([0.6, -0.4, -0.4, 0.6])
        pylab.xlabel(r'$\Delta$\textsf{RA from Sgr A* (arcsec)}')
        pylab.ylabel(r'$\Delta$\textsf{Dec. from Sgr A* (arcsec)}')

    def saveFigLayer(layerNum):
        pylab.axis([0.6, -0.4, -0.4, 0.6])
        outfile = 'plots/newSources/plot%s.png' % (str(layerNum).zfill(3))
        pylab.savefig(outfile)

    newFigLayer()
    layer(stars1, alpha=alphaOn, time='orbit')
    saveFigLayer(1)

    newFigLayer()
    layer(stars1, alpha=alphaOff, time='orbit')
    layer(stars2, alpha=alphaOn, time='obs')
    saveFigLayer(2)

    newFigLayer()
    layer(stars1, alpha=alphaOff, time='orbit')
    layer(stars2, alpha=alphaOn, time='extend')
    saveFigLayer(3)

    newFigLayer()
    layer(stars1, alpha=alphaOff, time='orbit')
    layer(stars2, alpha=alphaOff, time='obs')
    layer(stars3, alpha=alphaOn, time='obs')
    saveFigLayer(4)

    newFigLayer()
    layer(stars1, alpha=alphaOff, time='orbit')
    layer(stars2, alpha=alphaOff, time='obs')
    layer(stars3, alpha=alphaOn, time='extend')
    saveFigLayer(5)

    # Used LATEX for the axis labels 
    usetexFalse()

def usetexTrue():
    pylab.rc('text', usetex=True)
    pylab.rc('font', size=16)
    pylab.rc('axes', titlesize=20, labelsize=20)
    pylab.rc('xtick', labelsize=16)
    pylab.rc('ytick', labelsize=16)

def usetexFalse():
    pylab.rc('text', usetex=False)
    pylab.rc('font', size=14)
    pylab.rc('axes', titlesize=16, labelsize=16)
    pylab.rc('xtick', labelsize=14)
    pylab.rc('ytick', labelsize=14)


def orbitsAnimate(years=None, rad=0.5,
                      root='./',
                      align='align/align_d_rms_1000_abs_t',
                      poly='polyfit_d/fit',
                      orbitFile='orbits_movie.dat'):
    """
    Set rad to indicate the radial extent of orbits animation.
    Default is rad=0.5, which will show the central arcsecond orbits.
    """

    ##########
    #
    # START - Modify stuff in here only
    #
    ##########
    # Today's date
    today = 2011.7
    
    # Load up a starset of just those stars in orbits_movie.dat
    s = getOrbitStars(orbitFile=orbitFile,
                      root=root, align=align, poly=poly)
    tab = asciidata.open('/u/ghezgroup/data/gc/source_list/orbits_movie.dat')
    #tab = asciidata.open('/u/ghezgroup/data/gc/source_list/new_orbits_movie_keck_foundation_v2.dat')
    #tab = asciidata.open('/u/syelda/research/gc/aligndir/09_09_20/efit/new_orbits_movie.dat')

    ##########
    #
    # STOP - Modify stuff in here only
    #
    ##########

    name = s.getArray('name')
    mag = s.getArray('mag')

    # Get plotting properties from the orbits.dat file
    discovered = tab[9].tonumpy()  # Discovery date
    xshift1 = tab[10].tonumpy()    # Shifts for labels (in first frame)
    yshift1 = tab[11].tonumpy()
    xshift2 = tab[12].tonumpy()    # Shifts for labels (in last frame)
    yshift2 = tab[13].tonumpy()
    colors = [tab[14][ss].strip() for ss in range(tab.nrows)]

    # Determine the mass assuming a distance of 8.0 kpc
    star0orb = s.stars[0].orbit
    dist = 8000.0 # in parsec
    axis = (star0orb.a / 1000.0) * dist # in au
    mass = (axis)**3 / star0orb.p**2

    # Set the duration of the animation from the years keyword
    if (years == None):
        idx = name.index('S0-2')

        # Use S0-2's orbital period, rounded up to the nearest year
        years = math.ceil(s.stars[idx].orbit.p)

    # Array of time steps (0.1 yr steps)
    t = na.arange(1995.5, 1995.5+years, 0.2, type=na.Float)

    # For keck foundation presentation, make two versions:
    # Keck foundation version 1 -- extend to 2010.7
    #t = na.arange(1995.5, 2010.8, 0.2, type=na.Float)
    # Keck foundation version 2 -- extend to 2019.7
    #t = na.arange(1995.5, 2019.8, 0.2, type=na.Float)

    # Do a flux scaling so that all the stars look good in our image.
    #flux = 10.0**(mag/-3.0)
    #flux /= flux.max()
    flux = 10.0**(mag/-4.0) # Had to change to get 19th mag star to show up!
    flux /= flux.max()

    # Loop through all the stars and make an array of the X and Y positions
    # as a function of time. Store this on the star object as
    #   star.xanim -- array of X positions at each time step in t
    #   star.yanim -- array of Y positions at each time step in t
    for star in s.stars:
        (r, v, a) = star.orbit.kep2xyz(t, mass=mass, dist=dist)

        star.xanim = r[:,0].copy()
        star.yanim = r[:,1].copy()

    # Make an image 500x500 pixels (1" x 1")
    imgSize = 500 # pixels
    scale = (2.0*rad) / imgSize
    xaxis = (na.arange(imgSize, type=na.Float) - (imgSize/2.0)) # xextent
    xaxis *= -scale
    yaxis = (na.arange(imgSize, type=na.Float) - (imgSize/2.0)) # yextent
    yaxis *= scale

    # Make grids of X/Y value at each pixel
    xx, yy = pylab.meshgrid(xaxis, yaxis)

    ##########
    #
    # Create image with gaussian PSF for each star
    #
    ##########
    fwhm = 0.020   # Make 20 mas instead of 55 mas

    #for tt in range(1):
    #for tt in [len(t)-1]:
    for tt in range(len(t)):
        time = t[tt]
        img = na.zeros((imgSize, imgSize), type=na.Float)
        xorb = []
        yorb = []
        
        #for ss in range(1):
        for ss in range(len(s.stars)):
            star = s.stars[ss]

            xpos = star.xanim[tt]
            ypos = star.yanim[tt]

            # Make a 2D gaussian for this star
            psf = na.exp(-((xx - xpos)**2 + (yy - ypos)**2) / fwhm**2)

            img += flux[ss] * psf

        pylab.close(2)
        # For higher resolution, just increase figsize slightly
        pylab.figure(2, figsize=(5,5))
        pylab.clf()
        pylab.axes([0.0, 0.0, 1.0, 1.0])
        pylab.axis('off')
        cmap = gccolors.idl_rainbow()
        pylab.imshow(sqrt(img), origin='lowerleft', cmap=cmap,
                     extent=[xaxis[0], xaxis[-1], yaxis[0], yaxis[-1]],
                     vmin=sqrt(0.01), vmax=sqrt(1.0))

        # Plot the trails for each star 
        for ss in range(len(s.stars)):
            star = s.stars[ss]

            before = where((t < time) & (t < discovered[ss]))[0]
            during = where((t < time) & (t >= discovered[ss]) & (t <= today))[0]
            future = where((t < time) & (t > today))[0]

            # Dashed before discovery and in the future
            if (len(before) > 0):    
                pylab.plot(star.xanim[before], star.yanim[before], '--',
                           color=colors[ss], linewidth=2)
            if (len(during) > 0):    
                pylab.plot(star.xanim[during], star.yanim[during], '-',
                           color=colors[ss], linewidth=2)
            if (len(future) > 0):    
                pylab.plot(star.xanim[future], star.yanim[future], '--',
                           color=colors[ss], linewidth=2)
            # Label the stars in the first and last image
            if (tt == 0):
                pylab.text(star.xanim[tt]+xshift1[ss],
                           star.yanim[tt]+yshift1[ss],
                           name[ss],color='y', fontsize=10)
            if (tt == (len(t)-1)):
                pylab.text(star.xanim[tt]+xshift2[ss],
                           star.yanim[tt]+yshift2[ss],
                           name[ss],color='y', fontsize=10)
            # Label the first LGSAO image
            #diff = (abs(2005.5 - t).argsort())[0]
            #if (tt == diff):
            #    pylab.text(star.xanim[tt]+0.05,star.yanim[tt]+0.05,name[ss],color='y')

        # Draw an outline box
        #bx = 0.49
        bx = rad-(0.02*rad)
        pylab.plot([bx, -bx, -bx, bx, bx], [-bx, -bx, bx, bx, -bx],
                   color='white', linewidth=2)

        pylab.text(rad-(0.1*rad), rad-(0.2*rad), t[tt], color='white',
                   fontsize=16, fontweight='bold',
                   horizontalalignment='left', verticalalignment='bottom')
        pylab.text(rad-(0.4*rad), -rad+(0.2*rad), 'Keck/UCLA Galactic',
                   color='white', fontsize=10, fontweight='bold',
                   horizontalalignment='center', verticalalignment='top')
        pylab.text(rad-(0.4*rad), -rad+(0.12*rad), 'Center Group',
                   color='white', fontsize=10, fontweight='bold',
                   horizontalalignment='center', verticalalignment='top')

        # Plot a scale (make it slightly larger than 0.1", otherwise overlapping
        # arrows look funny
        arrowSize = rad/5.
        pylab.quiver([-rad+(0.1*rad)],[-rad+(0.3*rad)],[0],[arrowSize+0.005],
                       color='w',width=0.005,scale=1,units='x')
        pylab.quiver([-rad+(0.1*rad)],[-rad+(0.3*rad)-arrowSize+0.003],[0],
                       [-arrowSize-0.005], color='w',width=0.005,scale=1,units='x')
        pylab.text(-rad+(0.22*rad), -rad+(0.25*rad), str(arrowSize)+'\"',
                   color='white', fontsize=14, fontweight='bold',
                   horizontalalignment='center', verticalalignment='top')

        # Draw a star at the position of Sgr A* (large at first, then smaller)
        sgraColor = 'white'
        if (tt == 0):
            star = gccolors.Star(0,0,0.08)
            pylab.fill(star[0], star[1], fill=True,edgecolor=sgraColor,
                       linewidth=1.5,facecolor=sgraColor)
        if (tt == 1):
            star = gccolors.Star(0,0,0.07)
            pylab.fill(star[0], star[1], fill=True,edgecolor=sgraColor,
                       linewidth=1.5,facecolor=sgraColor)
        if (tt == 2):
            star = gccolors.Star(0,0,0.06)
            pylab.fill(star[0], star[1], fill=True,edgecolor=sgraColor,
                       linewidth=1.5,facecolor=sgraColor)
        if (tt == 3):
            star = gccolors.Star(0,0,0.05)
            pylab.fill(star[0], star[1], fill=True,edgecolor=sgraColor,
                       linewidth=1.5,facecolor=sgraColor)
        if (tt == 4):
            star = gccolors.Star(0,0,0.04)
            pylab.fill(star[0], star[1], fill=True,edgecolor=sgraColor,
                       linewidth=1.5,facecolor=sgraColor)
        if (tt == 5):
            star = gccolors.Star(0,0,0.04)
            pylab.fill(star[0], star[1], fill=True,edgecolor=sgraColor,
                       linewidth=1.5,facecolor=sgraColor)
        if (tt > 5):
            star = gccolors.Star(0,0,0.03)
            pylab.fill(star[0], star[1], fill=False,edgecolor=sgraColor,
                       linewidth=1.5)
        #pylab.axis([0.5, -0.5, -0.5, 0.5])
        pylab.axis([rad, -1.*rad, -1*rad, rad])
        # Save as png for the best animation image quality and smallest animation!!!!
        #pylab.savefig('/u/ghezgroup/public_html/gc/images/media/orbits_anim_2008/img_%s.png'
        pylab.savefig('/u/syelda/research/gc/anim/orbits/2011/img_%s.png'
                      % str(t[tt]), dpi=100)
        #pylab.savefig('/u/syelda/research/gc/aligndir/09_09_20/efit/orbits_anim_large/img_%s.png'
        #              % str(t[tt]), dpi=100)
        # Keck foundation version 1
        #pylab.savefig('/u/syelda/doc/prop/keck_foundation/site_visit/orbits_anim/v1/img_%s.png'
        #              % str(t[tt]), dpi=100)
        # Keck foundation version 2
        #pylab.savefig('/u/syelda/doc/prop/keck_foundation/site_visit/orbits_anim/v2/img_%s.png'
        #              % str(t[tt]), dpi=100)
            
        
def getOrbitStars(orbitFile='orbits.dat',
                  root='./',
                  align='align/align_d_rms_t',
                  poly='polyfit_d/fit',
                  flatten=False):

    orbs = starTables.Orbits(orbitFile=orbitFile)

    # Load up observed data
    s = starset.StarSet(root + align, relErr=True)
    s.loadPolyfit(root + poly, arcsec=True)
    s.loadPolyfit(root + poly, arcsec=True, accel=True)

    # Need to repair a couple of names
    for ii in range(len(s.stars)):
        star = s.stars[ii]

        #if (star.name == 'S0-39'):
        #    star.name = 'S0-40'
        #if (star.name == '29star_844'):
        #    star.name = 'S0-59'
        #if (star.name == '29star_1012'):
        #    star.name = 'S0-57'
        #if (star.name == '29star_2035'):
        #    star.name = 'S0-39'
        if (star.name == '32star_217'):
            star.name = 'S0-1'
        #if (star.name == '37star_551'):
        #    star.name = 'S0-20'
        if (star.name == 'star_1037'):
            star.name = 'S0-102'
        #if (star.name == 'star_866'):
        #    star.name = 'S0-103'
        if (star.name == 'star_700'):
            star.name = 'S0-104'
        #if (star.name == 'star_1906'):
        #    star.name = 'S0-105'

    # Trim out only the stars that are in orbits.dat
    stars = []
    allNames = [star.name for star in s.stars]
    for ii in range(len(orbs.name)):
        # Name in the orbits file
        name = orbs.name[ii]
        
        # Find this star in the align data
	try:
	    idx = allNames.index(name)

	    star = s.stars[idx]

            # Create an orbit object for this star. We will
            # use this to make the model orbits later on
            star.orbit = orbits.Orbit()
            star.orbit.a = orbs.a[ii]
            star.orbit.w = orbs.w[ii]
            star.orbit.o = orbs.o[ii]
            if (flatten):
                if (orbs.i[ii] > 90.0):
                    star.orbit.i = 180.0
                else:
                    star.orbit.i = 0.0
            else:
                star.orbit.i = orbs.i[ii]
            star.orbit.e = orbs.e[ii]
            star.orbit.p = orbs.p[ii]
            star.orbit.t0 = orbs.t0[ii]

            stars.append(star)

	except ValueError, e:
	    # Couldn't find the star in the align data
            print 'Could not find %s' % (name)
	    continue

    # Reset the starset's star list. Should only contain orbit stars now.
    s.stars = stars

    return s



def partiview(root='./', align='align/align_d_rms_t', poly='polyfit_d/fit'):
    s = getOrbitStars(orbitFile='orbits_movie.dat',
                      root=root, align=align, poly=poly)

    # Determine the mass assuming a distance of 8.46 kpc
    star0orb = s.stars[0].orbit
    dist = 8460.0 # in parsec
    axis = (star0orb.a / 1000.0) * dist # in au
    mass = (axis)**3 / star0orb.p**2
    print 'Mass = %.2e Msun    Dist = %4d pc' % (mass, dist)

    names = s.getArray('name')

    ## Three sets of stars
    ##    -- constrains Ro today
    ##    -- constrains Ro/extended mass in future (radial velocities)
    ##    -- constrains extended mass in future (no radial velocities)
    # Assume the stars are in increasing order by name
    stars1 = ['S0-2', 'S0-16', 'S0-19', 'S0-1', 'S0-20']
    stars2 = []
    stars3 = []

    yngstars = ['S0-1', 'S0-2', 'S0-3', 'S0-4', 'S0-5', 'S0-7', 'S0-8', 
		'S0-16', 'S0-19', 'S0-20', 'S0-26']
    oldstars = ['S0-6', 'S0-18']

    # All stars3 come after S0-26
    foundS026 = False
    for name in names:
        try:
            idx = stars1.index(name)
            continue
        except ValueError:
            if foundS026 == False:
                stars2.append(name)
                if name == 'S0-26':
                    foundS026 = True
            else:
                stars3.append(name)

    # Duration of the animation (30 years)
    t = na.arange(1995.5, 1995.5 + 300.0, 0.1, type=na.Float)

    xpos = na.zeros((len(s.stars), len(t)), type=na.Float)
    ypos = na.zeros((len(s.stars), len(t)), type=na.Float)
    zpos = na.zeros((len(s.stars), len(t)), type=na.Float)
    colors = []

    for ii in range(len(s.stars)):
        star = s.stars[ii]
        orb = star.orbit
        print 'Working on ', star.name
        
        # Determine time steps
        x = star.getArrayAllEpochs('x')
        y = star.getArrayAllEpochs('y')

        (r, v, a) = orb.kep2xyz(t, mass=mass, dist=dist)

        xpos[ii] = r[:,0]
        ypos[ii] = r[:,1]
        zpos[ii] = r[:,2]                  

	# Determine color
	if (star.name in yngstars):
	    colors.append('b')
	elif (star.name in oldstars):
	    colors.append('r')
	else:
	    colors.append('y')
	    
    _out = open('plots/partiview/partiview.dat', 'w')
    for ti in range(len(t)):
        _out.write('%9.4f %d\n' % (t[ti], ti))
        for si in range(len(s.stars)):
            _out.write('%7.3f %7.3f %7.3f %s\n' %
                       (xpos[si,ti], ypos[si,ti], zpos[si,ti], colors[si]))
            

    _out.close()

def partiviewOrbit(root='./', align='align/align_d_rms_t', 
		   poly='polyfit_d/fit'):
    s = getOrbitStars(orbitFile='orbits_movie.dat',
                      root=root, align=align, poly=poly)

    # Determine the mass assuming a distance of 8.46 kpc
    star0orb = s.stars[0].orbit
    dist = 8460.0 # in parsec
    axis = (star0orb.a / 1000.0) * dist # in au
    mass = (axis)**3 / star0orb.p**2
    print 'Mass = %.2e Msun    Dist = %4d pc' % (mass, dist)

    names = s.getArray('name')

    ## Three sets of stars
    ##    -- constrains Ro today
    ##    -- constrains Ro/extended mass in future (radial velocities)
    ##    -- constrains extended mass in future (no radial velocities)
    # Assume the stars are in increasing order by name
    stars1 = ['S0-2', 'S0-16', 'S0-19', 'S0-1', 'S0-20']
    stars2 = []
    stars3 = []

    yngstars = ['S0-1', 'S0-2', 'S0-3', 'S0-4', 'S0-5', 'S0-7', 'S0-8', 
		'S0-16', 'S0-19', 'S0-20', 'S0-26']
    oldstars = ['S0-6', 'S0-18']

    # All stars3 come after S0-26
    foundS026 = False
    for name in names:
        try:
            idx = stars1.index(name)
            continue
        except ValueError:
            if foundS026 == False:
                stars2.append(name)
                if name == 'S0-26':
                    foundS026 = True
            else:
                stars3.append(name)

    # Sample each star's full orbit 
    tlength = 201

    _out = open('plots/partiview/partiviewOrbits.dat', 'w')

    for ii in range(len(s.stars)):
        star = s.stars[ii]
        orb = star.orbit
        print 'Working on ', star.name
        
	tstep = orb.p / (tlength-1)
	t = na.arange(orb.t0, orb.t0 + (tlength*tstep), tstep, type=na.Float)

        # Determine time steps
        x = star.getArrayAllEpochs('x')
        y = star.getArrayAllEpochs('y')

        (r, v, a) = orb.kep2xyz(t, mass=mass, dist=dist)

        xpos = r[:,0]
        ypos = r[:,1]
        zpos = r[:,2]                  

	# Determine color
	if (star.name in yngstars):
	    color = 'b'
	elif (star.name in oldstars):
	    color ='r'
	else:
	    color = 'y'
	    
	_out.write('%-10s %5.2f %s\n' % (star.name, star.mag, color))
	for ti in range(len(t)):
            _out.write('%7.3f %7.3f %7.3f\n' %
                       (xpos[ti], ypos[ti], zpos[ti]))

    _out.close()

def lgsAnimate(redoAlign=False, redoImages=False):
    dataRoot = '/u/ghezgroup/data/gc/'

    imgRoots = ['05jullgs', '06maylgs1', '06junlgs', '06jullgs',
                '07maylgs', '07auglgs', '08maylgs1', '08jullgs',
                '09maylgs', '10maylgs', '10jullgs1', '10auglgs1',
                '11maylgs', '11auglgs']

    refEpoch = '08maylgs'

    workRoot = '/u/ghezgroup/public_html/gc/images/media/image_anim/'
    alignRoot = workRoot + 'align/align'

    if (redoAlign):
        # Copy over the starlists
        for epoch in imgRoots:
            lisFile = 'mag' + epoch + '_kp_rms.lis'
            copyFrom = dataRoot + epoch + '/combo/starfinder/' + lisFile
            copyTo = workRoot + 'lis/'
            shutil.copy(copyFrom, copyTo)
            lisFile = 'mag' + epoch + '_kp_rms_named.lis'
            copyFrom = dataRoot + epoch + '/combo/starfinder/' + lisFile
            copyTo = workRoot + 'lis/'
            shutil.copy(copyFrom, copyTo)


        # Align
        os.chdir(workRoot + 'align/')

        alignList = open(alignRoot + '.list', 'w')
        for epoch in imgRoots:
            lisFile = 'mag' + epoch + '_kp_rms.lis'

            alignList.write('../lis/' + lisFile + ' 9')
            if (epoch == refEpoch):
                alignList.write(' ref')
            alignList.write('\n')

        alignList.close()

        alignCmd = 'java align -a 2 -p '
        alignCmd += '-N /u/ghezgroup/data/gc/source_list/label.dat '
        #alignCmd += '-N /u/ghezgroup/data/gc/source_list/label_new.dat '
        alignCmd += '-o /u/ghezgroup/data/gc/source_list/orbits.dat '
        alignCmd += 'align.list'
        os.system(alignCmd)

        absCmd = 'java align_absolute '
        absCmd += '-abs /u/ghezgroup/data/gc/source_list/absolute_refs.dat '
        absCmd += 'align align_abs'
        os.system(absCmd)

        os.chdir(workRoot)

    s = starset.StarSet(alignRoot+'_abs')
    names = s.getArray('name')
    idx = names.index('SgrA')

    sgraxOrig = s.stars[idx].getArrayAllEpochs('xorig')
    sgrayOrig = s.stars[idx].getArrayAllEpochs('yorig')
    sgrax = s.stars[idx].getArrayAllEpochs('xpix')
    sgray = s.stars[idx].getArrayAllEpochs('ypix')
    print 'sgraxOrig = ', sgraxOrig
    print 'sgrax = ', sgrax
    print 'sgrayOrig = ', sgrayOrig
    print 'sgray = ', sgray

    # Tweak 05jullgs Sgr A* position, since it was not detected
    print
    print 'Tweaking 05jullgs Sgr A* position, since it was not named in the starlist'
    sgraxOrig[0] = 525.9
    sgrayOrig[0] = 643.9
    sgrax[0] = 0.0
    sgray[0] = 0.0


    print 'Update code with positions of Sgr A*'
    foo = raw_input('Continue? ')
    if (foo == 'q' or foo == 'Q'):
        return

    # Tweaks to Sgr A*'s positions (in arcseconds)
    # more negative goes to the right
    dx = array([-0.017, -0.002,  -0.003, -0.005,  0.001,  0.00,  0.002,  0.001,  0.002,  0.001,
                0.00, 0.00, 0.001, 0.00])
    dy = array([ 0.005,  0.00,  0.00,  0.00,  0.001,  0.00, 0.00, -0.003, 0.00, 0.00,
                0.000, 0.00, 0.000, 0.00])

    # Color Ranges for Images - Log Scale
    vmin = array([2.48, 2.51, 2.45, 2.48, 2.50, 2.55, 2.50, 2.50, 2.50, 2.55,
                  2.53, 2.53, 2.53, 2.45]) 
    vmax = array([3.45, 3.32, 3.38, 3.35, 3.40, 3.25, 3.3, 3.35, 3.3, 3.30,
                  3.30, 3.30, 3.30, 3.45]) 

    # Image plate scale
    scale = 0.00995

    if (len(imgRoots) != len(sgrax)):
        print 'Problem... images and align do not match. Compare to align.list'
        print imgRoots


    from gcreduce import gcutil
    import nirc2

    if (redoImages):
        # Need to rotate images
        from pyraf import iraf as ir
        ir.unlearn('rotate')
        ir.rotate.boundary = 'constant'
        ir.rotate.constant = 0
        ir.rotate.interpolant = 'spline3'

    # Need to plot images
    pylab.clf()
    for i in range(len(imgRoots)):
        imgIn = dataRoot + imgRoots[i] + '/combo/mag' + imgRoots[i] + '_kp.fits'
        imgOut = 'images/mag' + imgRoots[i] + '_rot.fits'


        hdr = pyfits.getheader(imgIn)
        phi = nirc2.getPA(hdr)

        if (redoImages):
            gcutil.rmall([imgOut])
            ir.rotate.xin = sgraxOrig[i]+1
            ir.rotate.yin = sgrayOrig[i]+1
            ir.rotate.xout = (sgrax[i]/scale)+512+1
            ir.rotate.yout = (sgray[i]/scale)+512+1
            ir.rotate(imgIn, imgOut, phi)

        img = pyfits.getdata(imgOut)

        xax = arange(0, img.shape[1])
        yax = arange(0, img.shape[0])

        xax = (xax - 512.) * -scale + dx[i]
        yax = (yax - 512.) *  scale + dy[i]
        
        # With labels and borders
        pylab.figure(1)
        pylab.clf()
        pylab.figure(figsize=(7,7))
        pylab.imshow(log10(img), extent=[xax[0], xax[-1], yax[0], yax[-1]],
                     vmin=vmin[i], vmax=vmax[i],origin='lowerleft')
        pylab.plot([0], [0], 'k+')
        #pylab.axis([0.5, -0.5, -0.5, 0.5])
        #pylab.axis([0.8, -0.8, -0.8, 0.8])
        #pylab.axis([3, -3, -3, 3])
        pylab.axis([1, -1, -1, 1])
        pylab.xlabel('RA Offset from Sgr A* (arcsec)')
        pylab.ylabel('Dec Offset from Sgr A* (arcsec)')
        pylab.title(imgRoots[i])

        pylab.savefig('frame%d.png' % i)
        pylab.close(1)

        
        # No labels and borders
        pylab.figure(2)
        pylab.clf()
        pylab.figure(figsize=(6,6))
        pylab.subplots_adjust(left=0, right=1, bottom=0, top=1)
        pylab.imshow(log10(img), extent=[xax[0], xax[-1], yax[0], yax[-1]],
                     vmin=vmin[i], vmax=vmax[i],origin='lowerleft')
        pylab.plot([0], [0], 'k+')
        pylab.axis([0.5, -0.5, -0.5, 0.5])
        pylab.axis('off')
        pylab.title(imgRoots[i])

        pylab.savefig('frame%d_noborder.png' % i)
        pylab.close(2)

    # See
    # http://www.astro.ucla.edu/~ghezgroup/gc/gc_doc/creating_lgsanim.shtml
    # for how to animate using photoshop.


def nightAnimation(epoch):
    cleanShifts = '/u/ghezgroup/data/gc/' + epoch 
    cleanShifts += '/combo/mag' + epoch + '_kp.shifts'

    table = asciidata.open(cleanShifts)

    cleanFiles = table[0]._data
    xshifts = table[1].tonumpy()
    yshifts = table[2].tonumpy()

    cleanDir = '/u/ghezgroup/data/gc/' + epoch + '/clean/kp/'
    for cc in range(len(cleanFiles)):
        print '%4d out of %4d' % (cc, len(cleanFiles))

        cleanImg = cleanDir + cleanFiles[cc]
        img = pyfits.getdata(cleanImg)

        xax = arange(0, img.shape[1])
        yax = arange(0, img.shape[0])

        xax -= 512.0 - xshifts[cc]
        yax -= 512.0 - yshifts[cc]

        pylab.clf()
        pylab.imshow(log10(img), extent=[xax[0], xax[-1], yax[0], yax[-1]],
                     vmin=1.5, vmax=3.5)
        pylab.plot([0], [0], 'k+')
        pylab.axis([-300, 300, -300, 300])
        pylab.xlabel('RA Offset (pixels)')
        pylab.ylabel('Dec Offset (pixels)')
        pylab.title(cleanFiles[cc])

        pylab.savefig(cleanFiles[cc].replace('.fits', '_shift.png'))
