import os, sys, math
import mpfit, asciidata
from pylab import *
from numpy import *
from gcwork import starset
from gcwork import objects
from gcwork import young
from gcwork import util
from gcwork import orbits

def fit(root=None, silent=False, label=False, outsuffix=''):
    """Fit the disk of young stars (clockwise) using our latest
    data and the indirect velocity method of Levin & Beloborodov (2003).
    
    @keyword root: The root directory for align stuff (default is
	'/u/jlu/work/gc/proper_motion/align/06_02_14/').
    @type root: string

    @keyword silent: Set to True for no printed output.
    @type: boolean
    """

    def fitfunc(n_in, fjac=None, yng=None):
	"""The fit function that is passed to mpfit."""

	# Normalize
	n = n_in / sqrt( vdot(n_in, n_in))

	vx = yng.getArray('vx')
	vy = yng.getArray('vy')
	vz = yng.getArray('vz')
	vxerr = yng.getArray('vxerr')
	vyerr = yng.getArray('vyerr')
	vzerr = yng.getArray('vzerr')
	r = yng.getArray('r2d')
	jz = yng.getArray('jz')

	# Chi-squared formula from Levin's paper
	num = len(yng.stars)
	devs = arange(num, dtype=float)
        bias = arange(num, dtype=float)

	for i in range(num):
	    v = array([vx[i], vy[i], vz[i]])
	    verr = array([vxerr[i], vyerr[i], vzerr[i]])

	    top = vdot(n, v)
	    bott = sqrt( ((n*verr)**2).sum() )

	    #devs[i] = (1.0 / num) * pow(top, 2) / pow(bott, 2)
	    devs[i] = top / bott
            bias[i] = 2.0 * log( bott )
            #print '  %2d  %7.3f  %7.3f' % (i, devs[i], bias[i])

        devsUnbias = sqrt(devs**2 + bias)

	status = 0
	return [status, devs / sqrt(num - 1)]



    ##########
    #
    # Load up young star sample
    #
    ##########

    if (root == None):
	root = '/u/jlu/work/gc/proper_motion/align/08_03_26/'

    yng = young.loadYoungStars(root)
    cc = objects.Constants()

    # Decide which young stars we are including in our fits.
    x = yng.getArray('x')
    y = yng.getArray('y')
    vX = yng.getArray('vx')
    vY = yng.getArray('vy')
    vZ = yng.getArray('vz')
    sigx = yng.getArray('vxerr')
    sigy = yng.getArray('vyerr')
    sigz = yng.getArray('vzerr')
    r = yng.getArray('r2d')
    jz = yng.getArray('jz')
    names = yng.getArray('name')

    # Read a table of sources + disk membership probability
    diskTabFile = root + 'tables/disk_membership_prob.dat'
    diskTab = asciidata.open(diskTabFile)
    diskNames = [diskTab[0][i].strip() for i in range(diskTab.nrows)]
    diskProbs = diskTab[1].tonumpy()

    indisk = zeros(len(names))
    indiskNames = []
    for i in range(len(names)):
        try:
            idx = diskNames.index(names[i])

            if (diskProbs[idx] > 2.7e-3):
                indisk[i] = 1
                indiskNames.append(names[i])
        except ValueError, e:
            print 'Failed'
            foo = 0

    # Filter that starlist in our starset so that it
    # only contains clockwise stars.
#     idx  = nonzero((vX != 0) & (vZ != 0) &
# 		   (r > 0.8) & (jz > 0.0) & (indisk == 1))[0]
    idx  = nonzero((vX != 0) & (vZ != 0) &
		   (r > 0.8) & (jz > 0.0))[0]
#     idx  = nonzero((vX != 0) & (vZ != 0) &
# 		   (r > 0.8) & (sigx < 10) & (jz > 0.6))[0]
#     idx  = nonzero((vX != 0) & (vZ != 0) &
# 		   (r > 0.8) & (sigx < 10) & (jz > 0.6) & (indisk == 1))[0]
    #cidx  = nonzero((vX != 0) & (vZ != 0) &
    #		    (r > 0.8) & (sigx < 10) & (jz > 0))[0]
    #print 'Throwing out stars:'
    #print [names[i] for i in idx]

    yng.stars = [yng.stars[i] for i in idx]

    # Now lets pull some variables
    x = yng.getArray('x')
    y = yng.getArray('y')
    vX = yng.getArray('vx')
    vY = yng.getArray('vy')
    vZ = yng.getArray('vz')
    sigx = yng.getArray('vxerr')
    sigy = yng.getArray('vyerr')
    sigz = yng.getArray('vzerr')
    r = yng.getArray('r2d')
    jz = yng.getArray('jz')
    names = yng.getArray('name')

    ##########
    #
    # Fitting
    #
    ##########

    # The initial guess for the normal vector
    n0 = array([-0.9, 0.01, -0.01])
    n0 = n0 / sqrt(vdot(n0, n0))

    # Three element list of parinfo objects
    parinfo = {'relStep':0.1, 'step':0.1, 'fixed':0, 'limits':[-1.0,1.0],
	      'limited':[1,1], 'mpside':2}
    pinfo = [parinfo.copy() for i in range(3)]

    # Put a constraint on the z-axis to break the degeneracy
    pinfo[2]['limits'] = [-1.0, -0.0001]
    pinfo[2]['limited'] = [1,1]
    functargs = {'yng': yng}

    m = mpfit.mpfit(fitfunc, n0, functkw=functargs, parinfo=pinfo, quiet=1)
    if (m.status <= 0): 
	print 'error message = ', m.errmsg

    n = m.params / sqrt( vdot(m.params, m.params) )
    chisq = m.fnorm

    # Use orbital reference plane: IRS 16SW-E
    #incl_orb = 101.0
    #bigOm_orb = 95.0
    #incl_orb = 109.47
    #bigOm_orb = 105.47

    # Paumard disk numbers
    #incl_orb = 127.0
    #bigOm_orb = 99.0

#     n_orb = zeros(3, dtype=float)
#     n_orb[2] = math.cos(math.radians(incl_orb))
#     n_orb[0] = -math.sqrt(1.0 - n_orb[2]**2) * math.cos(math.radians(bigOm_orb))
#     n_orb[1] = math.sqrt(1.0 - n_orb[2]**2) * math.sin(math.radians(bigOm_orb))

#     n = n_orb

    ##########
    #
    # Coordinate Conversion
    #
    ##########

    # Convert into plane coordinates: phat, qhat, that
    incl = math.acos(n[2])
    bigOm = math.acos( -n[0] / hypot(n[0], n[1]))
    incl_deg = math.degrees(incl)
    bigOm_deg = math.degrees(bigOm)

    that = array(n)
    tmp = hypot(n[0], n[1])
    phat = array([n[1], -n[0], 0.0]) / tmp
    qhat = array([n[1]*n[2], n[0]*n[2], -pow(tmp,2)]) / tmp

    # Coordinate rotation matrix
    coordRot = array([phat, qhat, that])

    # Project radius vectors along each direction
    z = -((n[0] * x) + (n[1] * y)) / n[2]

    # Number of stars
    numStars = len(x)

    # New position and velocity vectors
    pPos = zeros([numStars, 3], dtype=float)
    pVel = zeros([numStars, 3], dtype=float)
    pVelErr = zeros([numStars, 3], dtype=float)

    for i in range(numStars):
	r = array([x[i], y[i], z[i]])
	v = array([vX[i], vY[i], vZ[i]])
	verr = array([sigx[i], sigy[i], sigz[i]])
	
       	rnew = matrixmultiply(coordRot, r)
       	vnew = matrixmultiply(coordRot, v)
	verrnew = sqrt(matrixmultiply( pow(coordRot,2), pow(verr, 2) ))

	pPos[i] = rnew
	pVel[i] = vnew
	pVelErr[i] = verrnew

    ##########
    #
    # Plotting
    #
    ##########
    ylims = 300
    xlims = 4

    font = {'fontname' : 'Sans',
	    'fontsize' : 16,
	    'fontweight' : 'bold'}

    clf()
    subplot(2, 1, 1)
    errorbar(x, pVel[:,2], yerr=pVelErr[:,2], 
		   fmt='^', mfc='k', mec='k', ecolor='k')
    xlabel('X Offset from Sgr A*', font)
    ylabel('Total Vel (km/s)', font)
    
    if (label):
        for i in range(len(x)):
            text(x[i], pVel[i,2], names[i])
    plot([-1000,1000], [0,0], 'k--')
    xlim(xlims, -xlims)
    ylim(-ylims,ylims)

    subplot(2, 1, 2)
    errorbar(y, pVel[:,2], yerr=pVelErr[:,2], 
		   fmt='^', mfc='k', mec='k', ecolor='k')
    xlabel('Y Offset from Sgr A*', font)
    ylabel('Total Vel (km/s)', font)

    if (label):
        for i in range(len(x)):
            text(y[i], pVel[i,2], names[i])
    plot([-1000,1000], [0,0], 'k--')
    xlim(-xlims, xlims)
    ylim(-ylims, ylims)
    savefig('plots/yngdisk_resVel' + outsuffix + '.eps')


    clf()
    theta = arctan2(x, y) * 180.0 / math.pi
    errorbar(theta, pVel[:,2], pVelErr[:,2],
             fmt='^', mfc='k', mec='k', ecolor='k')
    xlabel('Position Angle from North (deg)', font)
    ylabel('Velocity Out-of-the-Plane (km/s)', font)

    if (label):
        for i in range(len(x)):
            text(theta[i], pVel[i,2], names[i])
    plot([-1000,1000], [0,0], 'k--')
    xlim(-200, 200)
    ylim(-ylims, ylims)

    thePlot = gca()
    setp( thePlot.get_xticklabels(), fontsize=14 )
    setp( thePlot.get_yticklabels(), fontsize=14 )

    savefig('plots/yngdisk_resVelPA' + outsuffix + '.eps')


#     # Now plot 2D map of tangential velocity component
#     # 2D ranges:
#     axlims = [10, -10, -10, 10]

#     clf()
#     plot([-10,10], [0,0], 'k--')
#     plot([0,0], [-10,10], 'k--')
#     scatter(pPos[:,0], pPos[:,1], zeros(numStars)+50.0, pVel[:,2], 
# 	    faceted=False)
#     #for i in range(len(x)):
#     #	text(pPos[i,0], pPos[i,1], names[i])
#     axis('equal')
#     axis(axlims)
#     xlabel('pHat in Plane (arcsec)')
#     ylabel('qHat in Plane (arcsec)')
#     title('Tangential Velocity (km/s)')
#     colorbar()
#     savefig('plots/yngdisk_resVelMap' + outsuffix + '.eps')

#     # Now plot 2D map of bigOmega difference
#     clf()
#     bigOmOrb = yng.getArray('efit.bigOmega')
#     bigOmDiff = bigOm_deg - bigOmOrb
    
#     scatter(pPos[:,0], pPos[:,1], zeros(numStars)+50.0, bigOmDiff, 
# 	    faceted=False)
#     #for i in range(len(x)):
#     #	text(pPos[i,0], pPos[i,1], names[i])
#     plot([-10,10], [0,0], 'k--')
#     plot([0,0], [-10,10], 'k--')
#     axis('equal')
#     axis(axlims)
#     xlabel('pHat in Plane (arcsec)')
#     ylabel('qHat in Plane (arcsec)')
#     title('Line of Nodes Difference')
#     colorbar()
#     savefig('plots/yngdisk_resBigOmMap' + outsuffix + '.eps')


#     # Now plot 2D map of inclination difference
#     clf()
#     inclOrb = yng.getArray('efit.incl')
#     inclDiff = incl_deg - inclOrb
    
#     scatter(pPos[:,0], pPos[:,1], zeros(numStars)+50.0, inclDiff, 
# 	    faceted=False)
#     #for i in range(len(x)):
#     #	text(pPos[i,0], pPos[i,1], names[i])
#     plot([-10,10], [0,0], 'k--')
#     plot([0,0], [-10,10], 'k--')
#     axis('equal')
#     axis(axlims)
#     xlabel('pHat in Plane (arcsec)')
#     ylabel('qHat in Plane (arcsec)')
#     title('Inclination Difference')
#     colorbar()
#     savefig('plots/yngdisk_resInclMap' + outsuffix + '.eps')

    
    ##########
    #
    # Determine accelerations
    #
    ##########
    asec_to_cm = cc.dist * cc.cm_in_au
    cm_to_asec = 1.0 / asec_to_cm
    
    rmag = sqrt(pow(x,2) + pow(y,2) + pow(z,2)) # arcsec
    rmag_cm = rmag * asec_to_cm

    ax = cc.G * cc.mass * cc.msun * (x*asec_to_cm) / pow(rmag_cm,3)
    ay = cc.G * cc.mass * cc.msun * (y*asec_to_cm) / pow(rmag_cm,3)
    az = cc.G * cc.mass * cc.msun * (z*asec_to_cm) / pow(rmag_cm,3)
    amag = sqrt(pow(ax,2) + pow(ay,2) + pow(az,2))
    ax = ax * cc.sec_in_yr / 1.0e5
    ay = ay * cc.sec_in_yr / 1.0e5
    az = az * cc.sec_in_yr / 1.0e5
    amag = amag * cc.sec_in_yr / 1.0e5

    if (silent == False):
	print 'MPFIT: Nstars = %d\tNiter=%d\tStatus = %d\t' % \
	    (len(names), m.niter, m.status)
        print "Normal vector:  [%5.2f, %5.2f, %5.2f]" % \
	    (n[0], n[1], n[2])
        print "Chi-Square:  %f" % (chisq / (len(x)-1.0))
        print 'Angle to Ascending Node: %f' % (bigOm_deg)
        print "Inclination:  %f" % (incl_deg)

	devs = (fitfunc(n, fjac=None, yng=yng))[1]
	newChi2 = (devs**2).sum() / (len(x) - 1.0)
        print "\nChi-Square of Orbit Disk:  %f" % (newChi2)

        print '\nDisk Residuals Mean = %4.1f, Stddev = %4.1f' % \
              (pVel[:,2].mean(), pVel[:,2].std())

#     clf()
#     hist(pVel[:,2], arange(-300, 300, 50))

    return n
    
