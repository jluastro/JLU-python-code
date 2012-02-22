from matplotlib import pyplot as py
import numpy as np
import time
import healpy
import pickle
import pdb
import glob
from gcwork import young
from gcwork import objects
from gcwork import orbits
from gcwork import analyticOrbits as aorb
from jlu.gc.orbits import healpix
import random, math, scipy, scipy.stats, scipy.optimize

def generateData(root, outdir, N=10**7):
    """
    Remember to calculate at least 2 times more stars than you want
    because many are dropped from our observing window.
    """
    cc = objects.Constants()

    yng = young.loadAllYoungStars(root)
    names = yng.getArray('name')

    # Get out the distributions we will need for our simulation.
    # Each relevant parameter will be plotted as a function of 
    # radius, divided into the following radial bins:
    #    BIN #1:  r = 0.8" - 2.5"
    #    BIN #2:  r = 2.5" - 3.5"
    #    BIN #3:  r = 3.5" - 7.0"
    #    BIN #4:  r = 7.0" - 13.5"
    
    binsIn = np.array([0.8, 2.5, 3.5, 7.0, 13.5])
    colors = ['red', 'orange', 'green', 'blue']
    legend = ['0.8"-2.5"', '2.5"-3.5"', '3.5"-7.0"', '7.0"-13.5"']

    r2d = yng.getArray('r2d')

    py.figure(1)
    py.clf()
    py.hist(r2d, bins=binsIn, histtype='step', linewidth=2)
    py.xlabel('Projected Radius (arcsec)')
    py.ylabel('Number of Stars Observed')
    py.savefig(outdir + 'hist_r2d_obs.png')

    binCnt = len(binsIn) - 1
    
    ridx = []
    for bb in range(binCnt):
        idx = np.where((r2d >= binsIn[bb]) & (r2d < binsIn[bb+1]))[0] 
        ridx.append( idx )
        
    ##########
    #
    # For each star we will use error distributions from the 
    # data itself. Set these up.
    #
    ##########
    # Positional Uncertainties in arcsec
    xerrDist, yerrDist = distPositionalError(yng, binsIn)

    # Proper motion uncertainties in arcsec/yr
    vxerrDist, vyerrDist = distProperMotionError(yng, binsIn, r2d, ridx, 
                                                 colors, legend, outdir)

    # Radial velocity uncertainties in km/s
    vzerrDist = distRadialVelocityError(yng, binsIn, r2d, ridx, 
                                        colors, legend, outdir)

    # Acceleration uncertainties in arcsec/yr^2
    axerrDist, ayerrDist = distAccelerationError(yng, binsIn, r2d,
                                                 colors, legend, outdir)

    # We will need a random number distribution for each of these.
    # These are all 0-1, but we will convert them into indices into
    # the above distributions later on when we know the radius of each
    # of the simulated stars.
    xerrRand = scipy.rand(N)
    yerrRand = scipy.rand(N)
    vxerrRand = scipy.rand(N)
    vyerrRand = scipy.rand(N)
    vzerrRand = scipy.rand(N)
    axerrRand = scipy.rand(N)
    ayerrRand = scipy.rand(N)


    ##########
    #
    # Generate simulated stars' orbital parameters
    #
    ##########

    # Inclination and Omega (degrees)
    i, o = distNormalVector(N, outdir)

    # Angle to the ascending node (degrees)
    w = scipy.rand(N) * 360.0

    # Semi-major axis (pc) and period (years)
    a, p = distSemimajorAxisPeriod(N, outdir)

    # Eccentricity
    e = scipy.stats.powerlaw.rvs(2, 0, size=N)

    # Lets correct the eccentricities just to prevent the
    # infinite loops.
    edx = np.where(e > 0.98)[0]
    e[edx] = 0.98

    # Time of Periapse (yr)
    t0 = scipy.rand(N) * p

    
    ##########
    #
    # Determine x, y, vx, vy, vz, ax, ay
    # Assign errors accordingly.
    # Record star to file IF it falls within our projected
    # observing window.
    #
    ##########
    x = np.zeros(N, dtype=np.float32)
    y = np.zeros(N, dtype=np.float32)
    vx = np.zeros(N, dtype=np.float32)
    vy = np.zeros(N, dtype=np.float32)
    vz = np.zeros(N, dtype=np.float32)
    ax = np.zeros(N, dtype=np.float32)
    ay = np.zeros(N, dtype=np.float32)

    xerr = np.zeros(N, dtype=np.float32)
    yerr = np.zeros(N, dtype=np.float32)
    vxerr = np.zeros(N, dtype=np.float32)
    vyerr = np.zeros(N, dtype=np.float32)
    vzerr = np.zeros(N, dtype=np.float32)
    axerr = np.zeros(N, dtype=np.float32)
    ayerr = np.zeros(N, dtype=np.float32)

    skipped = np.zeros(N, dtype=np.int16)

    for nn in range(N):
        if ( ((nn % 10**4) == 0)):
            print 'Simulated Star %d: ' % nn, time.ctime(time.time())

        orb = orbits.Orbit()
        orb.w = w[nn]
        orb.o = o[nn]
        orb.i = i[nn]
        orb.e = e[nn]
        orb.p = p[nn]
        orb.t0 = t0[nn]

        # Remember r (arcsec), v (mas/yr), a (mas/yr^2)
        (pos, vel, acc) = orb.kep2xyz(np.array([0.0]), mass=4.0e6, dist=8.0e3)
        pos = pos[0]         # in arcsec
        vel = vel[0] / 10**3 # convert to arcsec/yr
        acc = acc[0] / 10**3 # convert to arcsec/yr^2

        r2d = math.sqrt(pos[0]**2 + pos[1]**2)

        # Impose our observing limits
        if (r2d < binsIn[0]) or (r2d > binsIn[-1]):
            skipped[nn] = 1
            continue

        tmp = np.where(binsIn > r2d)[0]
        errIdx = tmp[0] - 1

        # Set noisey positions
        xerr[nn] = xerrDist[errIdx]
        yerr[nn] = yerrDist[errIdx]
        x[nn] = np.random.normal(pos[0], xerr[nn])
        y[nn] = np.random.normal(pos[1], yerr[nn])
        
        # Set noisy velocities
        errCnt = len(vxerrDist)
        vxerr[nn] = vxerrDist[errIdx][ int(math.floor(vxerrRand[nn] * errCnt)) ]
        vyerr[nn] = vyerrDist[errIdx][ int(math.floor(vyerrRand[nn] * errCnt)) ]
        vzerr[nn] = vzerrDist[errIdx][ int(math.floor(vzerrRand[nn] * errCnt)) ]

        # Convert radial velocity into km/s
        vel[2] *= 8.0 * cc.cm_in_au / (1e5 * cc.sec_in_yr)

        vx[nn] = np.random.normal(vel[0], vxerr[nn])
        vy[nn] = np.random.normal(vel[1], vyerr[nn])
        vz[nn] = np.random.normal(vel[2], vzerr[nn])


        # Set noisy accelerations ONLY for close in stars

        if (errIdx <= 1):
            # Close in stars have acceleration constraints
            axerr[nn] = axerrDist[errIdx][ int(math.floor(axerrRand[nn] * errCnt)) ]
            ayerr[nn] = ayerrDist[errIdx][ int(math.floor(ayerrRand[nn] * errCnt)) ]
            ax[nn] = np.random.normal(acc[0], axerr[nn])
            ay[nn] = np.random.normal(acc[1], ayerr[nn])
        else:
            axerr[nn] = 0.0
            ayerr[nn] = 0.0
            ax[nn] = 0.0
            ay[nn] = 0.0

    useIdx = np.where(skipped == 0)[0]
    finalTotal = len(useIdx)
    skipCount = N - finalTotal
    
    print 'Skipped %d, Final Total %d' % (skipCount, finalTotal)

    x = x[useIdx]
    y = y[useIdx]
    vx = vx[useIdx]
    vy = vy[useIdx]
    vz = vz[useIdx]
    ax = ax[useIdx]
    ay = ay[useIdx]

    xerr = xerr[useIdx]
    yerr = yerr[useIdx]
    vxerr = vxerr[useIdx]
    vyerr = vyerr[useIdx]
    vzerr = vzerr[useIdx]
    axerr = axerr[useIdx]
    ayerr = ayerr[useIdx]

    #print 'x  = ', x[-1], xerr[-1]
    #print 'y  = ', y[-1], yerr[-1]
    #print 'vx = ', vx[-1], vxerr[-1]
    #print 'vy = ', vy[-1], vyerr[-1]
    #print 'vz = ', vz[-1], vzerr[-1]
    #print 'ax = ', ax[-1], axerr[-1]
    #print 'ay = ', ay[-1], ayerr[-1]

    # Verify that we get the expected 2D distribution back out again
    checkSurfaceDensity(x, y, outdir)

    ##########
    #
    # Write to output file
    # 
    ##########
    _out = open(outdir + '/isotropic_stars.dat', 'w')
    
    pickle.dump((x, xerr), _out)
    pickle.dump((y, yerr), _out)
    pickle.dump((vx, vxerr), _out)
    pickle.dump((vy, vyerr), _out)
    pickle.dump((vz, vzerr), _out)
    pickle.dump((ax, axerr), _out)
    pickle.dump((ay, ayerr), _out)

    _out.close()


def distPositionalError(yng, binsIn):
    """
    Return the distribution of positional errors in arcsec.
    """
    # Positional Errors are hard coded.
    binCnt = len(binsIn) - 1

    # Default value is 1 mas
    xerrDist = np.ones(binCnt)
    yerrDist = np.ones(binCnt)

    # Hard code innermost sample to 0.4 mas
    xerrDist[0] = 0.4
    yerrDist[0] = 0.4

    # Convert from mas to arcsec
    xerrDist /= 10**3
    yerrDist /= 10**3

    return (xerrDist, yerrDist)


def distProperMotionError(yng, binsIn, r2d, ridx, colors, legend, outdir):
    """
    Return the distribution of proper motion errors in arcsec/yr.
    """
    cc = objects.Constants()
    binCnt = len(binsIn) - 1

    # Velocity Errors are taken from the observed distribution
    vxerrData = yng.getArray('vxerr') / cc.asy_to_kms
    vyerrData = yng.getArray('vyerr') / cc.asy_to_kms

    vxerrDist = []
    vyerrDist = []
    py.clf()
    for bb in range(binCnt):
        vxerrDist.append( np.array(vxerrData[ridx[bb]]) )
        py.hist(vxerrDist[bb]*10**3, bins=5, histtype='step', 
                ec=colors[bb], linewidth=2)
    py.xlabel('X Proper Motion Errors (mas/yr)')
    py.ylabel('Number of Stars Observed')
    py.legend(legend)
    py.savefig(outdir + 'hist_vxerr_obs.png')

    py.clf()
    for bb in range(binCnt):
        vyerrDist.append( np.array(vyerrData[ridx[bb]]) )
        py.hist(vyerrDist[bb]*10**3, bins=5, histtype='step', 
                ec=colors[bb], linewidth=2)
    py.xlabel('Y Proper Motion Errors (mas/yr)')
    py.ylabel('Number of Stars Observed')
    py.legend(legend)
    py.savefig(outdir + 'hist_vyerr_obs.png')

    py.clf()
    for bb in range(binCnt):
        py.plot(r2d[ridx[bb]], vxerrDist[bb]*10**3, 
                marker='s', color=colors[bb], linestyle='')
        py.plot(r2d[ridx[bb]], vyerrDist[bb]*10**3, 
                marker='o', color=colors[bb], linestyle='')
    py.xlabel('Projected Radius (arcsec)')
    py.ylabel('Proper Motion Errors (mas/yr)')
    py.savefig(outdir + 'plot_vxyerr_vs_r2d_obs.png')

    return (vxerrDist, vyerrDist)
    
def distRadialVelocityError(yng, binsIn, r2d, ridx, colors, legend, outdir):
    """
    Return the distribution of radial velocity errors in km/s
    """
    binCnt = len(binsIn) - 1

    # Radial velocity distribution
    vzerrData = yng.getArray('vzerr')
    vzerrDist = []
    py.clf()
    for bb in range(binCnt):
        vzerrDist.append( np.array(vzerrData[ridx[bb]]) )
        py.hist(vzerrDist[bb], bins=5, histtype='step',
                ec=colors[bb], linewidth=2)
    py.xlabel('Radial Velocity Errors (km/s)')
    py.ylabel('Number of Stars Observed')
    py.legend(legend)
    py.savefig(outdir + 'hist_vzerr_obs.png')

    return vzerrDist

def distAccelerationError(yng, binsIn, r2d, colors, legend, outdir):
    """
    Return the distribution of acceleration errors in arcsec/yr^2
    """
    # Acceleration distribution
    binCnt = len(binsIn) - 1

    axerrDist = []
    ayerrDist = []
    for bb in range(binCnt):
        axerrDist.append( np.array([], dtype=float) )
        ayerrDist.append( np.array([], dtype=float) )

    for star in yng.stars:
        # All stars with observed accelerations fall only into the 
        # first two bins
        try:
            if star.r2d < 2.5:
                axerrDist[0] = np.append( axerrDist[0], star.fitXa.aerr )
            else:
                axerrDist[1] = np.append( axerrDist[1], star.fitXa.aerr )
        except AttributeError:
            # Not in our data set.. just skip
            pass

        try:
            if star.r2d < 2.5:
                ayerrDist[0] = np.append( ayerrDist[0], star.fitYa.aerr )
            else:
                ayerrDist[1] = np.append( ayerrDist[1], star.fitYa.aerr )
        except AttributeError:
            # Not in our data set.. just skip
            pass

    py.clf()
    py.hist(axerrDist[0]*10**3, bins=5, histtype='step',
            ec=colors[0], linewidth=2)
    py.hist(axerrDist[1]*10**3, bins=5, histtype='step',
            ec=colors[1], linewidth=2)
    py.xlabel('X Acceleration Errors (mas/yr^2)')
    py.ylabel('Number of Stars Observed')
    py.legend([legend[0], legend[1]])
    py.savefig(outdir + 'hist_axerr_obs.png')

    py.clf()
    py.hist(ayerrDist[0]*10**3, bins=5, histtype='step',
            ec=colors[0], linewidth=2)
    py.hist(ayerrDist[1]*10**3, bins=5, histtype='step',
            ec=colors[1], linewidth=2)
    py.xlabel('Y Acceleration Errors (mas/yr^2)')
    py.ylabel('Number of Stars Observed')
    py.legend([legend[0], legend[1]])
    py.savefig(outdir + 'hist_ayerr_obs.png')
        
    return (axerrDist, ayerrDist)

def distNormalVector(N, outdir):
    incl = np.arccos( 2.0 * scipy.rand(N) - 1.0 )
    omeg = scipy.rand(N) * 2.0 * math.pi
    
    # Double check for random orientation
    nside = 4
    npix = healpy.nside2npix(nside)
    hidx = healpy.ang2pix(nside, incl, omeg)

    pdf = np.zeros(npix)
    for hh in hidx:
        pdf[hh] += 1

    healpix.plot(pdf, unit='N stars')
    py.savefig(outdir + 'heal_io_sim.png')
    py.close()
    
    incl = np.degrees(incl)
    omeg = np.degrees(omeg)

    return (incl, omeg)

def distSemimajorAxisPeriod(N, outdir):
    """
    Generate semi-major axis in parsec and period in years.
    """
    cc = objects.Constants()
    
    # Stretch from 0.2" (1600 AU) out to 40" (1.6 pc)
    a = scipy.stats.reciprocal.rvs(0.008, 1.6, size=N)  # in pc

    # Generate period in years
    p = np.sqrt( (a * cc.au_in_pc)**3 / 4.0e6 )

    # Double check that this produces a volume density \propto 1/r^3
    py.clf()
    counts, bins, foo = py.hist(a, bins=100)

    xdata = bins[:-1] + ((bins[1:] - bins[:-1]) / 2.0)

    shellVolume = (4.0 / 3.0) * math.pi * (bins[1:]**3 - bins[:-1]**3)
    shellVolume /= 0.04**3
    ydata = np.array(counts, dtype=float) / shellVolume

    idx = np.where(counts <= 0)[0]
    if (len(idx) > 0):
        xdata = xdata[0:idx[0]]
        ydata = ydata[0:idx[0]]

    # Fit the data to check the powerlaw of the volume density.
    logx = np.log10(xdata)
    logy = np.log10(ydata)

    fitfunc = lambda p, x, y: y - (p[0] + p[1]*x)
    pinit = [1.0, -3.0]
    out = scipy.optimize.leastsq(fitfunc, pinit, args=(logx, logy), 
                                 full_output=1)
    pfinal = out[0]
    amp = 10.0**pfinal[0]
    index = pfinal[1]
    print 'Semimajor-Axis: Amplitude = %5.2f   Index = %5.2f' % (amp, index)

    powerlaw = lambda x, amp, index: amp * (x**index)

    py.clf()
    py.loglog(xdata, ydata, 'bo')
    py.loglog(xdata, powerlaw(xdata, amp, index), 'k-')
    py.ylabel('Number Density (stars/arcsec^3)')
    py.xlabel('Semi-major axis (pc)')
    py.savefig(outdir + '/semimajor_axis_profile.png')

    return a, p

def checkSurfaceDensity(x, y, outdir):
    r2d = np.hypot(x, y)

    # Double check that this produces a volume density \propto 1/r^3
    py.clf()
    counts, bins, foo = py.hist(r2d, bins=100)

    xdata = bins[:-1] + ((bins[1:] - bins[:-1]) / 2.0)

    # area in arcsec^2
    area = math.pi * (bins[1:]**2 - bins[:-1]**2)

    ydata = np.array(counts, dtype=float) / area

    idx = np.where(counts <= 0)[0]
    if (len(idx) > 0):
        xdata = xdata[0:idx[0]]
        ydata = ydata[0:idx[0]]

    # Fit the data to check the powerlaw of the volume density.
    logx = np.log10(xdata)
    logy = np.log10(ydata)

    fitfunc = lambda p, x, y: y - (p[0] + p[1]*x)
    pinit = [1.0, -2.0]
    out = scipy.optimize.leastsq(fitfunc, pinit, args=(logx, logy), 
                                 full_output=1)
    pfinal = out[0]
    amp = 10.0**pfinal[0]
    index = pfinal[1]
    print 'Simulated Surface Density: Amplitude = %5.2f   Index = %5.2f' % \
        (amp, index)

    powerlaw = lambda x, amp, index: amp * (x**index)

    py.clf()
    py.loglog(xdata, ydata, 'bo')
    py.loglog(xdata, powerlaw(xdata, amp, index), 'k-')
    py.ylabel('Surface Density (stars/arcsec^2)')
    py.xlabel('Projected Radius (arcsec)')
    py.savefig(outdir + '/projected_radius_profile.png')
    

def analyzeData(root, dir, suffix, radii=[0.8,13.5], 
                Nstars=73, Nsims=100, Npdf=10**3,
                Nside=32, selectionEffects=False):
    """
    Calculate the density of normal vectors for each simulation.
    Results are output files with 2D arrays of 
    [Nsims x density of normal vectors].
    """

    # Calculate density of normal vectors using a 6th nearest 
    # neighbor algorithm.
    neighborCount = 6

    # Number of pixels in our final sky map
    Npix = healpy.nside2npix(Nside)

    # Read in the simulated data
    isoDataFile = root + dir + 'isotropic_stars.dat' 
    _data = open(isoDataFile)
    print 'Loaded isotropic stars from %s' % isoDataFile

    # Read in the mass and Ro PDF
    mrDataFile = root + 'tables/massRo_efit.dat'
    mr_efit = pickle.load( open(mrDataFile) )
    print 'Loaded mass/Ro distribution from %s' % mrDataFile
    
    xsim, xerrsim = pickle.load(_data)
    ysim, yerrsim = pickle.load(_data)
    vxsim, vxerrsim = pickle.load(_data)
    vysim, vyerrsim = pickle.load(_data)
    vzsim, vzerrsim = pickle.load(_data)
    axsim, axerrsim = pickle.load(_data)
    aysim, ayerrsim = pickle.load(_data)
    
    r2dsim = np.hypot(xsim, ysim)

    # Trim down to only the radius range we are interested in.
    # We can also apply some selectione effects that are present in the
    # spectroscopic FOV (>= 7.0 only use North-South stars (|x| < 5.0))
    if selectionEffects and radii[0] >= 7.0: 
        idx = np.where((r2dsim >= radii[0]) & (r2dsim < radii[1]) &
                       (np.abs(xsim) < 5.0))[0]
        print 'Applying N-S (|x| < 5.0") selection effect'
    else:
        idx = np.where((r2dsim >= radii[0]) & (r2dsim < radii[1]))[0]
    print 'Selected %d isotropic stars for simulation.' % len(idx)

    xsim = xsim[idx]
    xerrsim = xerrsim[idx]
    ysim = ysim[idx]
    yerrsim = yerrsim[idx]
    vxsim = vxsim[idx]
    vxerrsim = vxerrsim[idx]
    vysim = vysim[idx]
    vyerrsim = vyerrsim[idx]
    vzsim = vzsim[idx]
    vzerrsim = vzerrsim[idx]
    axsim = axsim[idx]
    axerrsim = axerrsim[idx]
    aysim = aysim[idx]
    ayerrsim = ayerrsim[idx]
    r2dsim = r2dsim[idx]

    # Random number generator for selecting the stars
    starGen = random.Random()

    # Our final output
    avgDensityMap = np.zeros((Nsims, Npix), dtype=np.float32)
    stdDensityMap = np.zeros((Nsims, Npix), dtype=np.float32)

    # Some other useful variables
    pixIdx = np.arange(Npix, dtype=int)
    (ipix, opix) = healpy.pix2ang(Nside, pixIdx)
    sinip = np.sin(ipix)
    cosip = np.cos(ipix)
    
    # Identity Matrices for calculations later
    onesNpix = np.ones(Npix, dtype=float)
    onesNstars = np.ones(Nstars, dtype=float)
    sqdegInSky = 2.0 * math.pi * (180.0 / math.pi)**2

    # We will use time t0 = 0 for all stars
    t0 = 0.0

    for ii in range(Nsims):
        print 'Simulation %d: ' % ii, time.ctime(time.time())

        # pick out a random set of Nstars
        starIdx = np.random.randint(0, len(xsim), size=Nstars)


        # inclination and Omega for all these stars
        iAll = np.zeros((Nstars, Npdf), dtype=np.float32)
        oAll = np.zeros((Nstars, Npdf), dtype=np.float32)

        for ss in range(Nstars):
            idx = starIdx[ss]
            star = objects.Star('simstar_%d_%d' % (ii, ss))
            
            # Check for accelerations
            ax = axsim[idx]
            ay = aysim[idx]
            axerr = axerrsim[idx]
            ayerr = ayerrsim[idx]

            if (axerrsim[idx] == 0.0) and (ayerrsim[idx] == 0.0):
                zfrom = 'all'
            else:
                zfrom = 'acc'

            star.setFitpXa(t0, xsim[idx], xerrsim[idx], 
                           vxsim[idx], vxerrsim[idx],
                           ax, axerr)
            star.setFitpYa(t0, ysim[idx], yerrsim[idx], 
                           vysim[idx], vyerrsim[idx],
                           ay, ayerr)
            star.vz = vzsim[idx]
            star.vzerr = vzerrsim[idx]

            mc = aorb.StarOrbitsMC(star, ntrials=Npdf)
            mc.run(mcMassRo=mr_efit, zfrom=zfrom, verbose=False)
            
            # Record inclination and Omega in radians
            iAll[ss,:] = np.radians(mc.i)
            oAll[ss,:] = np.radians(mc.o)

        # Compute the nearest neighbor density
        neighborMap = np.zeros(Npix, dtype=float)
        neighborMapStd = np.zeros(Npix, dtype=float)

        for nn in range(Npdf):
            incl = iAll[:,ii]
            omeg = oAll[:,ii]
            sini = np.sin(incl)
            cosi = np.cos(incl)

            # Check for bad things
            idx = np.where((incl == float('nan')) | 
                           (omeg == float('nan')))[0]

            if (len(idx) > 0):
                print nn, idx

            # Find densities
            omegSq = np.outer(omeg, onesNpix)
            opixSq = np.outer(onesNstars, opix)
            cosodiff = np.cos(opixSq - omegSq)

            sinSq = np.outer(sini, sinip)
            cosSq = np.outer(cosi, cosip)

            # Angular offset from each pixel for all stars (radians)
            angOff = np.arccos( (sinSq * cosodiff) + cosSq )
            angOff.sort(axis=0)

            # Density per square degree from nearest neighbor
            # Solid angle is 2 * pi * (1 - cos theta)
            nth = neighborCount
            densityMap = nth / (sqdegInSky*(1.0 - np.cos(angOff[nth-1,:])))
            maxPix = densityMap.argmax()

            neighborMap += densityMap
            neighborMapStd += densityMap**2

        neighborMap /= Npdf
        neighborMapStd = np.sqrt( (neighborMapStd / Npdf) - neighborMap**2 )

        avgDensityMap[ii] = neighborMap
        stdDensityMap[ii] = neighborMapStd

    print 'End Simulations: ', time.ctime(time.time())

    # Record the output to files
    avgfile = '%s%s/iso_d_avg_%.1f_%.1f_%d_%d_%d_%s.dat' % \
        (root, dir, radii[0], radii[1], Nstars, Nsims, Npdf, suffix)
    stdfile = '%s%s/iso_d_std_%.1f_%.1f_%d_%d_%d_%s.dat' % \
        (root, dir, radii[0], radii[1], Nstars, Nsims, Npdf, suffix)

    pickle.dump(avgDensityMap, file(avgfile, 'w'))
    pickle.dump(stdDensityMap, file(stdfile, 'w'))

def isoDensityMap(isoroot):
    isoDensity, isoDensityStd = loadIsotropicResults(isoroot)

    # Calculate the average of the average density from the isotropic 
    # simulations. 
    avgIsoDensity = isoDensity.mean(axis=0)
    stdIsoDensity = isoDensity.std(axis=0)

    # Plot the maps
    healpix.plot(avgIsoDensity, showdisks=True, 
                 unit='Avg. Density [stars/deg^2]')
    py.savefig(isoroot + '_avg.png')

    healpix.plot(stdIsoDensity, showdisks=True, 
                 unit='Std. Density [stars/deg^2]')
    py.savefig(isoroot + '_std.png')
    
    


def probabilityFromMap(i, o, density, avgDensityMap, angle=0, verbose=True):
    """
    Calculate the probability of encountering the specified
    density or higher at given pixel (or range of pixels) from 
    the isotropic simulations. 

    i --  inclination of a single pixel or the central pixel when
          angle is not 0 (degrees).
    o --  Omega of a single pixel or the central pixel when 
          angle is not 0 (degrees).
    density -- The peak density in this range from the observations.
    avgDensityMap -- The Nsims x Npix matrix of the results from the 
                     isotropic simulations.
    angle -- Default is 0. Set to an angle about the central pixel
             to include in probability calculations (degrees).
    """
    Nsides = 32
    Npix = healpy.nside2npix(Nsides)

    # Convert the angles to radians
    i = math.radians(i)
    o = math.radians(o)
    angle = math.radians(angle)

    # Get the pixel (or pixels) that fall within the specified
    # inclination and Omega constraints.
    if angle > 0:
        allPix = np.arange(0, Npix)
        (ipix, opix) = healpy.pix2ang(Nsides, allPix)

        sini = math.sin(i)
        cosi = math.cos(i)
        sino = math.sin(o)
        coso = math.cos(o)

        anglePix = np.arccos(sini * coso * np.sin(ipix) * np.cos(opix) + 
                             sini * sino * np.sin(ipix) * np.sin(opix) + 
                             cosi * np.cos(ipix))

        pixels = np.where(anglePix <= angle)[0]
    else:
        # Only a single pixel will be used
        pixels = healpy.ang2pix(Nsides, i, o)
        
    myAvgDensity = avgDensityMap[:,pixels].flatten()

    idx = np.where(myAvgDensity >= density)[0]

    probability = float(len(idx)) / len(myAvgDensity)

    if verbose:
        print('The probability of obtaining a density of %.2e' % density)
        print('or higher from an isotropic distribution is:')
        print('    prob = %.2e' % probability)

    return probability
            
def probability(i, o, density, isoroot, angle=0):
    """
    Calculate the probability of encountering the specified
    density or higher at given pixel (or range of pixels) from 
    the isotropic simulations. 

    i --  inclination of a single pixel or the central pixel when
          angle is not 0 (degrees).
    o --  Omega of a single pixel or the central pixel when 
          angle is not 0 (degrees).
    density -- The peak density in this range from the observations.
    isoroot -- The file root (e.g. 'iso_test/iso_d_avg_0.8_13.5_73_200_1000')
    angle -- Default is 0. Set to an angle about the central pixel
             to include in probability calculations (degrees).
    """
    avgDensity, stdDensity = loadIsotropicResults(isoroot)
    
    prob = probabilityFromMap(i, o, density, avgDensity, angle=angle)

    return prob

def probabilityMap(obsdir, isoroot):
    """
    Read a healpix map of our observed density of normal vectors.
    For each pixel in the map, calculate the probability distribution
    for the corresponding pixel from isotropic simulations. This yields
    a "probability map" indicating the probability that the observed
    density of normal vectors could be the result of an isotropic 
    distribution of stars.

    Parameters:
    obsdir -- the directory to search for the disk.neighbor.dat file
    isoroot -- the directory and root file name for the isotropic simulations.
               an example would be 'iso_test/iso_d_avg_0.8_13.5_73_200_1000'
               and all files with this root name will be included in the
               isotropic simulations.
    """
    Nside = 32
    Npix = healpy.nside2npix(Nside)

    # Load the observations
    from papers import lu06yng as lu
    obsDensity, obsDensityStd = lu.loadDiskDensity(Npix, orbDir=obsdir)

    # Load results from the Isotropic Simulations
    isoDensity, isoDensityStd = loadIsotropicResults(isoroot)

    # Get the inclination and omegas for each pixel
    pixIdx = np.arange(0, Npix)
    (ipix, opix) = healpy.pix2ang(Nside, pixIdx)
    ipix = np.degrees(ipix)
    opix = np.degrees(opix)

    # Construct a probability map and a gaussian significance map
    probMap = np.zeros(Npix, dtype=float)
    gsigMap = np.zeros(Npix, dtype=float)

    # Calculate the values for each pixel.
    for jj in range(Npix):
        i = ipix[jj]
        o = opix[jj]
        density = obsDensity[jj]
        
        probMap[jj] = probabilityFromMap(i, o, density, isoDensity, 
                                         verbose=False)

        # Calculate the average of the average density from the isotropic 
        # simulations. 
        avgIsoDensity = isoDensity[:,jj].mean()
        stdIsoDensity = isoDensity[:,jj].std()

        # Use the same definition as Bartko et al.
        gsigMap[jj] = (density - avgIsoDensity) / stdIsoDensity
    
    # Find the peak in the observations
    peakIdx = obsDensity.argmax()
    peakIncl = ipix[peakIdx]
    peakOmeg = opix[peakIdx]
    peakDens = obsDensity[peakIdx]
    print 'Disk candidate: '
    print '    inclination = %6.2f deg' % (peakIncl)
    print '    Omega       = %6.2f deg' % (peakOmeg)
    print '    density     = %9.5f stars/deg^2' % (peakDens)
    print 'Probability of measuring this density or higher'
    print 'from an isotropic distribution is:'
    print '    Prob        = %.2e' % (probMap[peakIdx])

    probMapLog = np.ma.masked_where(probMap == 0, np.log10(probMap))
    healpix.plot(probMapLog, showdisks=True, unit='Log Probabilty')
    py.savefig(obsdir + 'iso_probability_map.png')

    healpix.plot(gsigMap, showdisks=True, unit='normal sigma')
    py.savefig(obsdir + 'iso_significance_map.png')

def loadIsotropicResults(isoroot):
    avgfiles = glob.glob(isoroot + '_*.dat')
    stdfiles = glob.glob(isoroot + '_*.dat')
    print('Found %d files matching %s_*.dat' % (
            len(avgfiles), isoroot))

    # These variables will hold all our results from all the files
    avgDensity = None
    stdDensity = None

    # Read the average density files
    for avgfile in avgfiles:
        avgDensitySingle = pickle.load(open(avgfile))

        if avgDensity is None:
            avgDensity = avgDensitySingle
        else:
            avgDensity = np.concatenate((avgDensity, avgDensitySingle), axis=0)

    # Read the standard deviation files
    for stdfile in stdfiles:
        stdDensitySingle = pickle.load(open(stdfile))

        if stdDensity is None:
            stdDensity = stdDensitySingle
        else:
            stdDensity = np.concatenate((stdDensity, stdDensitySingle), axis=0)

    return (avgDensity, stdDensity)
