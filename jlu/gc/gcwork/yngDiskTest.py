from gcwork import analyticOrbits as aorb
import math
import numarray as na
import mpfit
import pylab
import pickle
from gcwork import util

def run():
    # Assume circular orbits with isotropic normal vectors on circular
    # orbits with circular orbital velocity of 500 km/s.
    ntrials = 10000
    nstars = int((1.0 / 3.0) * 100)
    gens = aorb.create_generators(7, ntrials*10)

    velTot = 500.0  # km/s -- velocity amplitude
    radTot = 2.04   # arcsec

    def fitfunc(p, fjac=None, vx=None, vy=None, vz=None, 
		vxerr=None, vyerr=None, vzerr=None):
        irad = math.radians(p[0])
        orad = math.radians(p[1])
	nx = math.sin(irad) * math.cos(orad)
	ny = -math.sin(irad) * math.sin(orad)
	nz = -math.cos(irad)

	top = (nx*vx + ny*vy + nz*vz)**2
	bot = (nx*vxerr + ny*vyerr + nz*vzerr)**2

	devs = (1.0 / (len(vx) - 1.0)) * top / bot
	status = 0

        return [status, devs.flat]

    # Keep track from every trial the incl, omeg, chi2, number of stars
    incl = na.zeros(ntrials, type=na.Float)
    omeg = na.zeros(ntrials, type=na.Float)
    chi2 = na.zeros(ntrials, type=na.Float)
    niter = na.zeros(ntrials)
    stars = na.zeros(ntrials)

    # Keep track of same stuff for spatial test
    inclPos = na.zeros(ntrials, type=na.Float)
    omegPos = na.zeros(ntrials, type=na.Float)
    chi2Pos = na.zeros(ntrials, type=na.Float)
    niterPos = na.zeros(ntrials)

    angleAvg = na.zeros(ntrials, type=na.Float)
    angleStd = na.zeros(ntrials, type=na.Float)

    for trial in range(ntrials):
	if ((trial % 100) == 0):
	    print 'Trial %d' % trial

	x = na.zeros(nstars, type=na.Float)
	y = na.zeros(nstars, type=na.Float)
	z = na.zeros(nstars, type=na.Float)

	vx = na.zeros(nstars, type=na.Float)
	vy = na.zeros(nstars, type=na.Float)
	vz = na.zeros(nstars, type=na.Float)

	rmag = na.zeros(nstars, type=na.Float)
	vmag = na.zeros(nstars, type=na.Float)
	nx = na.zeros(nstars, type=na.Float)
	ny = na.zeros(nstars, type=na.Float)
	nz = na.zeros(nstars, type=na.Float)
	

	for ss in range(nstars):
	    vx[ss] = gens[0].uniform(-velTot, velTot)
	
	    vyTot = math.sqrt(velTot**2 - vx[ss]**2)
	    vy[ss] = gens[1].uniform(-vyTot, vyTot)
	    
	    vz[ss] = math.sqrt(velTot**2 - vx[ss]**2 - vy[ss]**2)
	    vz[ss] *= gens[2].choice([-1.0, 1.0])
	    
	    x[ss] = gens[3].uniform(-radTot, radTot)

	    yTot = math.sqrt(radTot**2 - x[ss]**2)
	    y[ss] = gens[4].uniform(-yTot, yTot)

	    z[ss] = math.sqrt(radTot**2 - x[ss]**2 - y[ss]**2)
	    z[ss] *= gens[5].choice([-1.0, 1.0])

	    rmag[ss] = math.sqrt(x[ss]**2 + y[ss]**2 + z[ss]**2)
	    vmag[ss] = math.sqrt(vx[ss]**2 + vy[ss]**2 + vz[ss]**2)

	    rvec = [x[ss], y[ss], z[ss]]
	    vvec = [vx[ss], vy[ss], vz[ss]]
	    tmp = util.cross_product(rvec, vvec)
	    tmp /= rmag[ss] * vmag[ss]
	    nx[ss] = tmp[0]
	    ny[ss] = tmp[1]
	    nz[ss] = tmp[2]
	    
	r2d = na.hypot(x, y)
	v2d = na.hypot(vx, vy)
	top = (x * vy - y * vx)
	jz = (x * vy - y * vx) / (r2d * v2d)

	djzdx = (vy * r2d * v2d - (top * v2d * x / r2d)) / (r2d * v2d)**2
	djzdy = (-vx * r2d * v2d - (top * v2d * y / r2d)) / (r2d * v2d)**2
	djzdvx = (-y * r2d * v2d - (top * r2d * vx / v2d)) / (r2d * v2d)**2
	djzdvy = (x * r2d * v2d - (top * r2d * vy / v2d)) / (r2d * v2d)**2
	
	xerr = na.zeros(nstars, type=na.Float) + 0.001 # arcsec
	yerr = na.zeros(nstars, type=na.Float) + 0.001
	vxerr = na.zeros(nstars, type=na.Float) + 10.0  # km/s
	vyerr = na.zeros(nstars, type=na.Float) + 10.0
	vzerr = na.zeros(nstars, type=na.Float) + 30.0 # km/s
	
	jzerr = na.sqrt((djzdx*xerr)**2 + (djzdy*yerr)**2 + 
			(djzdvx*vxerr)**2 + (djzdvy*vyerr)**2)

	# Eliminate all stars with jz > 0 and jz/jzerr < 2
	# I think these are they cuts they are doing
	idx = (na.where((jz < 0) & (na.abs(jz/jzerr) > 2)))[0]
	#idx = (na.where(jz < 0))[0]
	#idx = range(len(jz))

	cotTheta = vz / na.sqrt(vx**2 + vy**2)
	phi = na.arctan2(vy, vx)

	# Initial guess:
	p0 = na.zeros(2, type=na.Float)
	p0[0] = gens[5].uniform(0.1, 90)     # deg -- inclination
	p0[1] = gens[6].uniform(0.1, 360)     # deg -- omega

	# Setup properties of each free parameter.
	parinfo = {'relStep':10.0, 'step':10.0, 'fixed':0, 
		   'limits':[0.0,360.0],
		   'limited':[1,1], 'mpside':1}
	pinfo = [parinfo.copy() for i in range(len(p0))]

	pinfo[0]['limits'] = [0.0, 180.0]
	pinfo[1]['limits'] = [0.0, 360.0]

	# Stuff to pass into the fit function
	functargs = {'vx': vx[idx], 'vy': vy[idx], 'vz': vz[idx],
		     'vxerr':vxerr[idx], 'vyerr':vyerr[idx], 'vzerr':vzerr[idx]}

	m = mpfit.mpfit(fitfunc, p0, functkw=functargs, parinfo=pinfo,
			quiet=1)
	if (m.status <= 0): 
	    print 'error message = ', m.errmsg

	p = m.params

	incl[trial] = p[0]
	omeg[trial] = p[1]
	stars[trial] = len(idx)
	chi2[trial] = m.fnorm / (stars[trial] - len(p0))
	niter[trial] = m.niter
	
	n = [math.sin(p[0]) * math.cos(p[1]),
	     -math.sin(p[0]) * math.sin(p[1]),
	     -math.cos(p[0])]

	# Now look at the angle between the best fit normal vector
	# from the velocity data and the true r cross v normal vector.
	# Take the dot product between n and nreal.
	angle = na.arccos(n[0]*nx + n[1]*ny + n[2]*nz)
	angle *= (180.0 / math.pi)

	# What is the average angle and std angle
	angleAvg[trial] = angle.mean()
	angleStd[trial] = angle.stddev()

# 	print chi2[trial], chi2Pos[trial], incl[trial], inclPos[trial], \
# 	    omeg[trial], omegPos[trial], niter[trial], niterPos[trial]
# 	print angleAvg[trial], angleStd[trial]
	

    # Plot up chi2 for v-fit vs. chi2 for x-fit
    pylab.clf()
    pylab.semilogx(chi2, angleAvg, 'k.')
    pylab.errorbar(chi2, angleAvg, fmt='k.', yerr=angleStd)
    pylab.xlabel('Chi^2')
    pylab.ylabel('Angle w.r.t. Best Fit')
    foo = raw_input('Contine?')

    # Probability of encountering solution with chi^2 < 2
    idx = (na.where(chi2 < 2.0))[0]
    print 'Prob(chi^2 < 2) = %5.3f ' % (len(idx) / float(ntrials))

    # Probability of encountering solution with chi^2 < 2 AND 
    # inclination = 20 - 30 and Omega = 160 - 170
    foo = (na.where((chi2 < 2.0) & (incl > 20) & (incl < 40)))[0]
    print 'Prob of chi2 and incl = %5.3f' % (len(foo) / float(ntrials))

    pylab.clf()
    pylab.subplot(2, 2, 1)
    pylab.hist(chi2, bins=na.arange(0, 10, 0.5))
    pylab.xlabel('Log Chi^2')

    pylab.subplot(2, 2, 2)
    pylab.hist(incl[idx])
    pylab.xlabel('Inclination for Chi^2 < 2')
    rng = pylab.axis()
    pylab.axis([0, 180, rng[2], rng[3]])

    pylab.subplot(2, 2, 3)
    pylab.hist(omeg[idx])
    pylab.xlabel('Omega for Chi^2 < 2')
    rng = pylab.axis()
    pylab.axis([0, 360, rng[2], rng[3]])

    pylab.subplot(2, 2, 4)
    pylab.hist(stars[idx])
    pylab.xlabel('Nstars for Chi^2 < 2')
    rng = pylab.axis()
    pylab.axis([0, 33, rng[2], rng[3]])

    pylab.savefig('diskTest.png')
    
    # Pickle everything
    foo = {'incl': incl, 'omeg': omeg, 'star': stars, 
	   'chi2': chi2, 'niter': niter}

    pickle.dump(foo, open('diskTestSave.pick', 'w'))
