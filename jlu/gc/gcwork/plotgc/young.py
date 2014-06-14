from gcwork import starTables

def partiview(root='./', align='align/align_d_rms_t', poly='polyfit_d/fit'):
    datfile = '/u/ghezgroup/data/gc/source_list/young_new.dat'
    yngNames = starTables.youngStarNames(datfile=datfile)
    cc = objects.Constants()

    # First look for efit orbital solutions (incorporate radial velocities)
    s = getOrbitStars(orbitFile='orbits_movie.dat', absolute=True,
                      root=root, align=align, poly=poly)

    # File contains analytic orbit solutions without acceleration limits
    pickleFile = root + 'analyticOrbits/' + star + '.results.dat'
    #oo = pickle.load(open(pickleFile))

    # File contains analytic orbit solutions with acceleration limits (MC)
    pdffile = '%s%s.mc.dat' % (pdfdir, star)
    pdf = pickle.load(open(pdffile))


    def dispHist(xx, yy, xxtheory, yytheory):
        fmt = 'k.'
        fmt2 = 'r--'
        pntsize = 2
        #cmap = cm.YlGnBu
        #cmap = cm.summer_r
        cmap = cm.hot_r

        # Make 2D histogram
        (probDist, b1, b2) = h2d.histogram2d(xx, yy, bins=(50, 50))

        # Need to convert the 2d histogram into floats
        probDist = array(probDist, type=Float)
        
        # Determine contour levels
        # Flatten and reverse sort our prob. distribution
        sid0 = probDist.flat.argsort()
        sid = sid0[::-1]
        pixSort = probDist.flat[sid]
        
        # Make a cumulative distribution function starting from the
        # highest pixel value. This way we can find the level above
        # which 68% of the trials will fall.
        cdf = cumsum(pixSort)
        
        # Determine point at which we reach 68% level
        #percents = array([0.6827, 0.9545, 0.9973]) * len(xx)
        percents = array([0.6827, 0.9545]) * len(xx)
        levels = zeros(len(percents), type=Float)
        for ii in range(len(levels)):
            # Get the index of the pixel at which the CDF
            # reaches this percentage (the first one found)
            idx = (where(cdf < percents[ii]))[0]

            # Now get the level of that pixel
            levels[ii] = pixSort[idx[-1]]
        #print levels
            
        # Mask out the parts where we don't have data.
        probDist = ma.masked_where(probDist == 0, log10(probDist))
        levels = log10(levels)

        imshow(probDist, extent=[b1[0], b1[-1], b2[0], b2[-1]],
               cmap=cmap, origin='lower', aspect='auto')
        contour(probDist, levels, origin=None, colors='black',
                extent=[b1[0], b1[-1], b2[0], b2[-1]])

        #plot(xxtheory, yytheory, fmt2)
        #plot(xx, yy, fmt, markersize=pntsize)

    def axisLabel(xtext, ytext, yrange=None):
        # Rescales fonts for ticks and labels
        thePlot = gca()
        rng = axis()

        # Incrememnt for ticks on the X axis
        tmp = abs(float(rng[1]) - float(rng[0])) / 5.0
        xinc = __builtin__.round(tmp, 1)

        if (xinc == 0):
            xinc = 0.05
        
        thePlot.get_xaxis().set_major_locator(MultipleLocator(xinc))
        setp( thePlot.get_xticklabels(), fontsize=tickSize )
        setp( thePlot.get_yticklabels(), fontsize=tickSize )

        # Add axis labels
        xlabel(xtext, labelFont)
        ylabel(ytext, labelFont)

        # Optional re-scale axes
        if (yrange != None):
            axis([rng[0], rng[1], yrange[0], yrange[1]])

    #xlab = r'a$_\rho$ (mas/yr$^2$)'
    xlab = r'{\bf z (pc)}'
    xdat = pdf.z * cc.dist / cc.au_in_pc
    #xdat2 = oo.z * cc.dist / cc.au_in_pc
    xdat2 = None

    # Plot Eccentricity
    subplot(2, 3, 1)
    #dispHist(xdat, pdf.e, xdat2, oo.e)
    dispHist(xdat, pdf.e, xdat2, None)
    axisLabel(xlab, r'${\bf {\it e}}$', yrange=[0, 1.0])






    
    s = getOrbitStars(orbitFile='orbits_movie.dat', absolute=True,
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
    s = getOrbitStars(orbitFile='orbits_movie.dat', absolute=True,
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
