from scipy import stats
from pylab import *
from numpy import *
import numpy as np
import pickle, time, random, pdb
import histogram2d as hist2d
import os, mpfit
from matplotlib.mlab import griddata
import scipy.integrate

from gcwork import objects
from gcwork import orbits
from gcwork import young
from gcwork import plot_disk_healpix
from gcwork import util
import healpy

class StarOrbits(object):
    def __init__(self, theStar, nodata=False, mscStar=False):
        """
        Pass in a gcwork.objects.Star and read off the following variables:
        x,y,xerr,yerr -- from fitXa and fitYa on the star object (arcsec)
        vx,vy,vz,vxerr,vyerr,vzerr -- from same named variables (km/s)
        ax,ay,axerr,ayerr -- from fitXa and fitYa (mas/yr^2)
        refTime -- the epoch at which these values are valid
        x_dat,y_dat,xerr_dat,yerr_dat -- from individual epochs of data
        date_dat -- the dates for the individual epochs of data
        """
        self.nodata = nodata
        cc = objects.Constants()

        self.name = theStar.name
        
        ##########
        #
        # Load up kinematics of this star.
        #
        ##########
        # Arcsec
        if (nodata == False) & (mscStar == False):
            self.x = theStar.fitXa.p
            self.y = theStar.fitYa.p
            self.xerr = theStar.fitXa.perr
            self.yerr = theStar.fitYa.perr
            self.refTime = theStar.fitXa.t0
        elif (nodata == False) & (mscStar == True):
            self.x = theStar.fitXv.p
            self.y = theStar.fitYv.p
            self.xerr = theStar.fitXv.perr
            self.yerr = theStar.fitYv.perr
            self.refTime = theStar.fitXv.t0
        else:
            self.x = theStar.x
            self.y = theStar.y
            self.xerr = theStar.xerr
            self.yerr = theStar.yerr
            self.refTime = theStar.fitXv.t0
            
        self.r2d = theStar.r2d

        # km/s
        self.vx = theStar.vx
        self.vy = theStar.vy
        self.vz = theStar.vz
        self.vxerr = theStar.vxerr
        self.vyerr = theStar.vyerr
        self.vzerr = theStar.vzerr
        self.v3d = sqrt(self.vx**2 + self.vy**2 + self.vz**2)
        self.vz_ref = theStar.rv_ref

        # mas/yr^2
        if (nodata == False) & (mscStar == False):
            self.ax = theStar.fitXa.a * 1000.0
            self.ay = theStar.fitYa.a * 1000.0
            self.axerr = theStar.fitXa.aerr * 1000.0
            self.ayerr = theStar.fitYa.aerr * 1000.0


        # Only do the following if in our dataset.
        if not self.nodata:
            years = theStar.getDates()

            # Get all the years of observations for this star.
            # Also build up an matrix of the measurements.
            self.x_dat = zeros(theStar.velCnt, dtype=float)
            self.y_dat = zeros(theStar.velCnt, dtype=float)
            self.xerr_dat = zeros(theStar.velCnt, dtype=float)
            self.yerr_dat = zeros(theStar.velCnt, dtype=float)
            self.date_dat = zeros(theStar.velCnt, dtype=float)

            trans = objects.Transform()
            trans.loadAbsolute()

            ii = 0
            for ee in range(len(theStar.years)):
                epoch = theStar.e[ee]
                xpix = epoch.x_pnt
                ypix = epoch.y_pnt
                xepix = epoch.xe_pnt
                yepix = epoch.ye_pnt
                date = epoch.t

                if (xepix >= 0):
                    (self.x_dat[ii], self.y_dat[ii],
                     self.xerr_dat[ii], self.yerr_dat[ii]) = \
                     util.rerrPix2Arc(xpix, ypix, xepix, yepix, trans)

                    self.date_dat[ii] = date

                    ii += 1


            self.avg_err = self.xerr_dat.sum() + self.yerr_dat.sum()
            self.avg_err /= len(self.xerr_dat) + len(self.yerr_dat)

            #(self.ar, self.at, self.arerr, self.aterr) = \
            #          util.xy2circErr(self.x, self.y, self.ax, self.ay,
            #                          0.0, 0.0, self.axerr, self.ayerr)
            # Feed in x and y errors as well.        
            (self.ar, self.at, self.arerr, self.aterr) = \
                      util.xy2circErr(self.x, self.y, self.ax, self.ay,
                                      self.xerr, self.yerr,
                                      self.axerr, self.ayerr)
        
            # Acceleration Limits from POLYFIT (mas/yr^2)
            self.sigma = 3.0
            self.alimX = self.ax - (self.sigma * self.axerr)
            self.alimY = self.ay - (self.sigma * self.ayerr)
            self.alimR = self.ar - (self.sigma * self.arerr)
            self.aLoLim = self.ar + (self.sigma * self.arerr)
            

        # Derive radial and tangential acceleration (and uncertainties)
        self.rAsec = hypot(self.x, self.y)

    def calc(self):
        """
        Use x, y, vx, vy, vz data to compute the full range of all
        possible orbital solutions. Save z + all orbital parameters
        to arrays.

        This StarOrbits object (and all results of the calculation)
        are stored in a pickle file in analyticOrbits/<star>.results.dat.
        You can re-load this pickle file using:

        import pickle
        oo = pickle.load(open('analyticOrbits/S1-2.results.dat', 'r'))

        Also use the acceleration limits to further constrain
        the range of orbital parameters. Orbital parameter limits are
        assigned to variables:
        (e.g. i_lo_p for inclination, lowerlimit, positive Z values)

        Variables include:
        zmax - Maximum allowed Z assuming bound orbit.
        z - array of all possible Z values.
        i, ie - array of Inclination and error 
        e, ee - array of Eccentricity and error
        o, oe - array of PA to Asc. Node and error
        w, we - array of Angle to Periapse and error
        p, pe - array of Period and error
        t0, t0e - array of Time of Periapse and error
        ph, phe - array of Orbital Phase and error

        r_mag, r_mag_cgs - array of radius (arcsec and cgs)
        v_mag, v_mag_cgs - array of velocity (mas/yr and cgs)
        ve_mag, ve_mag_cgs - array of vel. error
        a_mag, a2d_mag - 3D and plane of the sky accelerations (mas/yr^2)
        h_mag_cgs, he_mag_cgs - array of angular mom. and error
        h_rv, h_rve - array of |h| / |r||v| = sin(theta) and error

        vesc - array of escape velocities
        rms - array of RMS residuals from model orbit and data at each Z
        chi2, chi2red - array from model orbit and data at each Z

        a - array of 3D acceleration vectors at each Z (mas/yr^2)
        r - array of 3D position vectors at each Z (arcsec)
        v - array of 3D velocity vectors at each Z (mas/yr); all the same?
        ve - array of 3D vel. error vectors at each Z (mas/yr); all the same?

        idxp - array indices for acceptable orbital solutions (positive Z)
        idxn - array indices for acceptable orbital solutions (negative Z)
        idx - array indices for acceptable orbital solutions (all Z)

        zp_lo, zp_hi - lowest and highest positive Z values
        zn_lo, zn_hi - lowest and highest negative Z values

        i_lo_p, i_hi_p - inclination lower/upper limits for Z > 0
        i_lo_n, i_hi_n - inclination lower/upper limits for Z < 0
        i_lo, i_hi - inclination lower/upper limits for all Z
        ...
        ...

        """
        cc = objects.Constants()
        GM = cc.G * cc.msun * cc.mass
        
        pm = hypot(self.vx, self.vy)
        pmErr = sqrt((self.vx*self.vxerr)**2 + (self.vy*self.vyerr)**2) / pm
        vtot = sqrt(self.vx**2 + self.vy**2 + self.vz**2)
        vtotErr = sqrt((self.vx*self.vxerr)**2 +
                       (self.vy*self.vyerr)**2 +
                       (self.vz*self.vzerr)**2) / vtot

        print 'Radial velocity from: %s' % self.vz_ref
 
        # Assume z = 0 and zerr = +/- zmax
        # zmax is set by distance at which vTot = vEsc
        # Calculate the escape velocity at the 2D projected radius.
        # This will tell us if the energy<0 requirement is constraining.
        r2dtmp = sqrt(self.x**2 + self.y**2)
        r2d_cm = r2dtmp * cc.dist * cc.cm_in_au
        rmax = 2.0 * cc.G * cc.msun * cc.mass / (vtot * 1.0e5)**2
        rmax /= cc.cm_in_au * cc.dist
        self.zmax = sqrt(rmax**2 - r2dtmp**2)

        zerr = self.zmax

        # Make an array of all possible z values
        self.z = arange(-self.zmax+0.002, self.zmax-0.002,
                        (2.0 * self.zmax) / 100.0)

        zcnt = len(self.z)

        # Arrays of values that we would like to plot at some point.
        self.i = zeros(zcnt, dtype=float)
        self.ie = zeros(zcnt, dtype=float)
        self.e = zeros(zcnt, dtype=float)
        self.ee = zeros(zcnt, dtype=float)
        self.p = zeros(zcnt, dtype=float)
        self.pe = zeros(zcnt, dtype=float)
        self.t0 = zeros(zcnt, dtype=float)
        self.t0e = zeros(zcnt, dtype=float)
        self.w = zeros(zcnt, dtype=float)
        self.we = zeros(zcnt, dtype=float)
        self.o = zeros(zcnt, dtype=float)
        self.oe = zeros(zcnt, dtype=float)
        self.vesc = zeros(zcnt, dtype=float)
        self.h_rv = zeros(zcnt, dtype=float)
        self.h_rve = zeros(zcnt, dtype=float)
        self.ph = zeros(zcnt, dtype=float)
        self.phe = zeros(zcnt, dtype=float)

        self.rms = zeros(zcnt, dtype=float)
        self.chi2 = zeros(zcnt, dtype=float)
        self.chi2red = zeros(zcnt, dtype=float)

        self.r = zeros((zcnt, 3), dtype=float)
        self.v = zeros((zcnt, 3), dtype=float)
        self.a = zeros((zcnt, 3), dtype=float)
        self.ve = zeros((zcnt, 3), dtype=float)

        self.r_mag = zeros(zcnt, dtype=float)
        self.r_mag_cgs = zeros(zcnt, dtype=float)
        self.v_mag = zeros(zcnt, dtype=float)
        self.v_mag_cgs = zeros(zcnt, dtype=float)
        self.ve_mag = zeros(zcnt, dtype=float)
        self.ve_mag_cgs = zeros(zcnt, dtype=float)
        self.a_mag = zeros(zcnt, dtype=float)
        self.a2d_mag = zeros(zcnt, dtype=float)
        self.h_mag_cgs = zeros(zcnt, dtype=float)
        self.he_mag_cgs = zeros(zcnt, dtype=float)


        ##########
        #
        # Analytic Orbits:
        # 
        # Loop through all the possible z-values and determine the
        # orbital parameters.
        #
        ##########
        for zz in range(len(self.z)):
            # Input info
            # r in arcsec
            # v in km/s
            rvec = array([self.x, self.y, self.z[zz]])
            vvec = array([self.vx, self.vy, self.vz])
            revec = array([self.xerr, self.yerr, 0.0])
            vevec = array([self.vxerr, self.vyerr, self.vzerr])

            orb = orbits.Orbit()
            orb.xyz2kep(rvec, vvec, revec, vevec, self.refTime)
            (rtmp, vtmp, atmp) = orb.kep2xyz(array([self.refTime]))
            self.r[zz,:] = rtmp[0,:]
            self.v[zz,:] = vtmp[0,:]
            self.a[zz,:] = atmp[0,:]

            #if (z[zz] < 1.0 and z[zz] > -1.0):
            #    print '%5.1f %5.3f  %5.3f  %5.3f' % \
            #          (z[zz], a[zz,0], a[zz,1], a[zz,2])

            # Convert velocity to mas/yr which is same as orbit output
            self.ve[zz,:] = vevec * 1000.0 / cc.asy_to_kms

            self.i[zz] = orb.i
            self.ie[zz] = orb.ie
            self.e[zz] = orb.e
            self.ee[zz] = orb.ee
            self.p[zz] = orb.p
            self.pe[zz] = orb.pe
            self.t0[zz] = orb.t0
            self.t0e[zz] = orb.t0e
            self.w[zz] = orb.w
            self.we[zz] = orb.we
            self.o[zz] = orb.o
            self.oe[zz] = orb.oe
            self.ph[zz] = orb.ph
            self.phe[zz] = orb.phe

            self.r_mag[zz] = sqrt(( self.r[zz,:]**2 ).sum())
            self.r_mag_cgs[zz] = self.r_mag[zz] * cc.dist * cc.cm_in_au

            self.v_mag[zz] = sqrt(( self.v[zz,:]**2 ).sum())
            self.v_mag_cgs[zz] = self.v_mag[zz] * cc.asy_to_kms * 1.0e5 / 1000.0

            self.a_mag[zz] = -sqrt(( self.a[zz,:]**2 ).sum())
            self.a2d_mag[zz] = -sqrt(( self.a[zz,0]**2 ) + ( self.a[zz,1]**2 ))

            self.ve_mag[zz] = sqrt(( (self.v[zz,:]*self.ve[zz,:])**2 ).sum())
            self.ve_mag[zz] /= self.v_mag[zz]
            self.ve_mag_cgs[zz] = self.ve_mag[zz] * cc.asy_to_kms
            self.ve_mag_cgs[zz] *= 1.0e5 / 1000.0

            self.h_mag_cgs[zz] = sqrt(( orb.hvec**2 ).sum())
            self.he_mag_cgs[zz] = sqrt(( (orb.hevec * orb.hvec)**2 ).sum())
            self.he_mag_cgs[zz] /= self.h_mag_cgs[zz]

            # Put this in mas/yr
            self.vesc[zz] = sqrt(2.0 * GM / self.r_mag_cgs[zz])
            self.vesc[zz] *= 1000.0 / (1.0e5 * cc.asy_to_kms)

            self.h_rv[zz] = self.h_mag_cgs[zz]
            self.h_rv[zz] /= (self.r_mag_cgs[zz] * self.v_mag_cgs[zz])
            self.h_rve[zz] = (self.he_mag_cgs[zz] / self.h_mag_cgs[zz])**2 + \
                        (self.ve_mag_cgs[zz] / self.v_mag_cgs[zz])**2
            self.h_rve[zz] = sqrt( self.h_rve[zz] ) * self.h_rv[zz]

            # Calculate Chi-Squared and RMS for the orbit vs. data
            if not self.nodata:
                # Now compute the theoretical positions for all epochs
                # Units are in arcsec for positions
                (r_mod, v_mod, a_mod) = orb.kep2xyz(self.date_dat)

                x_mod = r_mod[:,0]
                y_mod = r_mod[:,1]

                diffX = self.x_dat - x_mod
                diffY = self.y_dat - y_mod
                diff = sqrt(diffX**2 + diffY**2)

                self.rms[zz] = sqrt((diff**2).sum() / (2.0 * len(diff)))
                chi2_tmp = (diffX / self.xerr_dat)**2 + \
                           (diffY / self.yerr_dat)**2
                self.chi2[zz] = chi2_tmp.sum()
                self.chi2red[zz] = self.chi2[zz] / ((2.0 * len(diff)) - 6.0)

        ##########
        #
        # Parameter Ranges:
        #
        # Find acceptable range of Z values based
        # on acceleration limits from polyfit.
        # Remember, accelerations are negative.
        #
        ##########
        if (self.nodata == True):
            self.alimR = -1000.0
            self.date_dat = [0]
            
        ## Only look for acceleration limits in stars with lots of data.
        ## We define this as existing in > 24 epochs.
        print 'ORBIT ANALYSIS: Polyfit vs. MinDist acc = %7.3f vs. %7.3f' % \
              (self.alimR, self.a2d_mag.min())

        if (self.alimR > self.a2d_mag.min()):
            # We have significant acceleration limits.
            # Z solutions are both positive and negative.
            self.idxp = (where((self.a2d_mag > self.alimR) & \
                               (self.z >= 0)))[0]
            self.idxn = (where((self.a2d_mag > self.alimR) & \
                                   (self.z < 0)))[0]

            if self.alimR < 0.0:
                print 'ORBIT ANALYSIS: Significant polyfit acceleration constraints.'
                # Check if this is significantly different from zero
                if (self.aLoLim < 0.0):
                    print 'ORBIT ANALYSIS: Acceleration Detection'
            else:
                print 'ORBIT ANALYSIS: Positive radial acceleration (unphysical).'
        else:
            print 'ORBIT ANALYSIS: No polyfit acceleration constraint.'
            self.idxp = (where(self.z >= 0))[0]
            self.idxn = (where(self.z < 0))[0]
        #else:
        #    # This star doesn't have enough epochs of data to look for
        #    # acceleration limits.
        #    print 'ORBIT ANALYSIS: Star not in enough epochs to test acceleration (N=%i).' % \
        #          len(self.date_dat)
        #    self.idxp = (where(self.z >= 0))[0]
        #    self.idxn = (where(self.z < 0))[0]

        # Lets also make an index for ALL acceptable solutions
        self.idx = concatenate([self.idxp, self.idxn])
        print 'number of acceptable solutions: %i' % len(self.idx)

        # Positive solutions:
        self.zp_lo = min(self.z[self.idxp])
        self.zp_hi = max(self.z[self.idxp])
        # Negative solutions:
        self.zn_lo = min(self.z[self.idxn])
        self.zn_hi = max(self.z[self.idxn])

        ### Now lets save all our results to various variables

        findRanges(self, 'i')    # Inclination
        findRanges(self, 'o')    # Big Omega
        findRanges(self, 'w')    # Little omega
        findRanges(self, 'e')    # Eccentricity
        findRanges(self, 'p')    # Period
        findRanges(self, 't0')   # t0
        findRanges(self, 'ph')   # Phase

        # Fix the bigOmega range
        if ((abs(self.o_lo_p - self.o_hi_p) > 180) or
            (abs(self.o_lo_n - self.o_hi_n) > 180)):
            idx = (where(self.o > 180))[0]

            self.o[idx] -= 360.0

        findRanges(self, 'o')    # Big Omega
            

    def saveToFile(self, savefile):
        """
        SAVE all the results to a binary file that python can
        later read in for further plotting/analysis. This is
        called "pickle" in python. To reload, simply call
        
             selfe = pickle.load(open(<file>, 'r'))
        """
        _pic = open(savefile, 'w')
        pickle.dump(self, _pic)





class StarOrbitsMC(object):
    def __init__(self, theStar, ntrials=100000,
                 outroot='analyticOrbits/', mscStar=False):
        """
        Pass in a gcwork.objects.Star and read off the following variables:
        x,y,xerr,yerr -- from fitXa and fitYa on the star object (arcsec)
        vx,vy,vz,vxerr,vyerr,vzerr -- from same named variables (km/s)
        ax,ay,axerr,ayerr -- from fitXa and fitYa (mas/yr^2)
        refTime -- the epoch at which these values are valid

        Other inputs:
        ntrials = the number of trials to run to compute the PDF
        outroot = the place to store the results and plots
        """
        self.ntrials = ntrials
        self.outroot = outroot

        self.name = theStar.name
        
        ##########
        #
        # Load up kinematics of this star.
        #
        ##########
        if mscStar == False:
            starXfit = theStar.fitpXa
            starYfit = theStar.fitpYa
        else:
            starXfit = theStar.fitpXv
            starYfit = theStar.fitpYv

        # pixels
        self.x_dat = starXfit.p
        self.y_dat = starYfit.p
        self.xerr_dat = starXfit.perr
        self.yerr_dat = starYfit.perr
        self.r2d = theStar.r2d

        # pix/yr or km/s (radial)
        self.vx_dat = starXfit.v
        self.vy_dat = starYfit.v
        self.vz_dat = theStar.vz
        self.vz_ref = theStar.rv_ref
        self.vxerr_dat = starXfit.verr
        self.vyerr_dat = starYfit.verr
        self.vzerr_dat = theStar.vzerr

        if mscStar == False:
            # pix/yr^2
            self.ax_dat = starXfit.a
            self.ay_dat = starYfit.a
            self.axerr_dat = starXfit.aerr
            self.ayerr_dat = starYfit.aerr

        # Year
        self.refTime = starXfit.t0

        # NEW
        self.sigma = 3.0
        # End NEW


    def run(self, mcMassRo=None, zfrom='acc', mExtended=None, verbose=True, makeplot=False):
        """
        Perform a Monte Carlo analysis on the data to determine the
        probability distribution for the orbital parameters and the
        possible plane-of-the-sky accelerations. Make and save contours
        of the resulting probability distribution so that it can be
        used in conjunction with other stars to derive the best fit
        plane.

        This StarOrbitsMC object (and all results of the Monte Carlo)
        are stored in a pickle file in analyticOrbits/<star>.mc.dat.
        You can re-load this pickle file using:

        import pickle
        oo = pickle.load(open('analyticOrbits/S1-2.mc.dat', 'r'))

        Input Options:
        mcMassRo -- (def=None) Set to file name containing pickled
                    bhPotential.BHprops() object.
        zfrom    -- Set to 'acc' (def), 'all', or 'plane':
                    acc: derive z from polyfit acceleration
                    all: derive z from all possible bound orbits (uniform acc)
                    all2: derive z from all possible bound orbits (uniform z)
                    plane: derive z from plane (hardcoded plane i/Omega)

        Available variables are:
        i,e,w,o,p,t0,ph -- Orbital parameters.
        x,y,z -- MC selected distances (arcsec)
        vx,vy,vz -- MC selected velocities (km/s)
        ar -- MC selected acceleration (mas/yr**2) in the plane of the sky
        m  -- MC selected mass (solar masses)
        r0 -- MC selected distance to the SBH (parsec)
        x0 -- MC selected X focus (pixels)
        y0 -- MC selected Y focus (pixels)
        name -- Star Name
        pdf -- 2D Probability Density Function for i and o
        """
        
        ##########
        #
        # First we need to create a collection of a-values whose
        # histogram is distributed according to the probability
        # prob(a) set by the measured polyfit a and sigma_a.
        # We can then select from this collection
        # when doing the monte carlo.
        #
        ##########

        if (verbose):
            print 'START: ', time.ctime(time.time())

        cc = objects.Constants()

        # Make all our random number generators. We will need
        # 7 all together (x, y, vx, vy, vz, ax, ay)
        gens = create_generators(7, self.ntrials*1000)
        xgen = gens[0]
        ygen = gens[1]
        vxgen = gens[2]
        vygen = gens[3]
        vzgen = gens[4]
        axgen = gens[5]
        aygen = gens[6]

        self._initVariables()
        self._initTransform()

        ##########
        #
        # Monte Carlo.
        #
        ##########
        for zz in range(int(self.ntrials/2.0)):
            if ( ((zz % 5000) == 0) & (verbose == True)):
                print 'PDF Trial %d: ' % (zz*2), \
                      time.ctime(time.time()), self.i[zz-1]
            
            #if (isnan(self.i[zz-1]) == 1):
            #    pdb.set_trace()

            # Set temp values for our while loop
            amin = -1.0
            amax = -2.0
            aminLoopCount = 0

            # We need to make sure we always satisfy the
            # bound condition. We do this by checking that
            # amin (set by r2d) < amax (set by v=vesc), where
            # all accels are negative.
            while(amin >= amax): # not bound; keep checking for bound condition
                aminLoopCount += 1

                def sample_gaussian(self):
                    # Sample from Gaussian centered on our acceleration measurement
                    ax = axgen.gauss(self.ax_dat, self.axerr_dat)  # arcsec/yr^2
                    ay = aygen.gauss(self.ay_dat, self.ayerr_dat)  # arcsec/yr^2
                    
                    # Convert into mas/yr^2  
                    (ax, ay) = util.vPix2Arc(ax, ay, self.trans)
                    ax *= 1000.0
                    ay *= 1000.0
                    
                    # Convert into radial and tangential
                    (ar, at) = util.xy2circ(x, y, ax, ay)

                    return (ar, at)

                # Sample our monte carlo variables
                x = xgen.gauss(self.x_dat, self.xerr_dat)     # asec (+x west)
                y = ygen.gauss(self.y_dat, self.yerr_dat)     # asec (+y north)
                vx = vxgen.gauss(self.vx_dat, self.vxerr_dat) # asec/yr
                vy = vygen.gauss(self.vy_dat, self.vyerr_dat) # asec/yr
                vz = vzgen.gauss(self.vz_dat, self.vzerr_dat) # km/s

                rad = sqrt(x**2 + y**2)
                
                if mExtended != None:
                    (mass, dist, x0, y0) = self._getPotential(mcMassRo, zz, mExtended, rad)
                else:
                    (mass, dist, x0, y0) = self._getPotential(mcMassRo, zz)

                GM = cc.G * mass * cc.msun        # cgs
                
                (x, y) = util.rPix2Arc(x, y, self.trans)     # asec
                (vx, vy) = util.vPix2Arc(vx, vy, self.trans) # asec/yr

                # Convert velocities in km/s
                asy_to_kms = dist * cc.cm_in_au / (1.0e5 * cc.sec_in_yr)
                vx *= asy_to_kms
                vy *= asy_to_kms

                # At this point:
                #   all positions are in arcsec and
                #   all velocities are in km/s

                # This is the difference between this new code and
                # that in analyticOrbits.py. We compute amin, then get z,
                # then check if it's bound using the 3D position.
                # Determine the minimum allowed acceleration (where a < 0)
                # set by the minimum radius = 2D projected radius.
                amin = self._calcMinAcc(x, y, GM, dist, cc)

                # Determine the maximum allowed acceleration (where a < 0)
                # set by assuming a bound orbit.
                amax = self._calcMaxAcc(x, y, vx, vy, vz, GM, dist, cc)

                # Now sample the acceleration between [amin, amax)
                # If we have an acceleration measurement, make sure
                # it falls within the bounds set by [amin, amax)
                if (zfrom == 'acc'):
                    ar, at = sample_gaussian(self)
                elif (zfrom == 'uni_acc'):
                    ar = axgen.uniform(amin, amax)

                if (ar >= amin and ar < amax):

                    # Now convert our acceleration into a z-value. We will 
                    # have to do both the positive and negative solutions.
                    z = acc2z(x, y, ar, dist, mass)

                    # Now check that the velocity we sampled is not greater than
                    # the escape velocity at this 3D distance
                    vtot = sqrt(vx**2 + vy**2 + vz**2)
                    r3d = sqrt(x**2 + y**2 + z**2)
                    vtotcgs = vtot * 1.0e5
                    r3dcgs = r3d * dist * cc.cm_in_au

                    #print 'Mass at r2d: %6.3e' % (mass * cc.msun)

                    # Recompute the mass for this 3D radius
                    if mExtended != None:
                        (mass, dist, x0, y0) = self._getPotential(mcMassRo, zz, mExtended, r3d)
                    #else:
                    #    (mass, dist, x0, y0) = self._getPotential(mcMassRo, zz)
                    GM = cc.G * mass * cc.msun        # cgs

                    #print 'Mass at r3d: %6.3e' % (mass * cc.msun)

                    if (vtotcgs**2 > (2.0 * GM / r3dcgs)):
                        if ((aminLoopCount % 100) == 0 and verbose):
                            print 'Random pos/vel break bound orbit criteria: zz=%d, %s' % \
                                  (zz, self.name)
                            print '\tamin = %7.3f  amax = %7.3f' % (amin, amax)
                            print '\tvel = %7.3f    escape = %7.3f' % \
                                   (vtot, sqrt(2.0 * GM / r3dcgs) / 1.0e5)
                            print '\tvx = %3d  vy = %3d  vz = %3d' % \
                                  (vx, vy, vz)
                            print '\tmass = %7.3e' % (mass * cc.msun)

                        amin = -1.0
                        amax = -2.0
                        continue
                else:
                    if ((aminLoopCount % 500) == 0 and verbose):
                        print 'Having problems getting enough accelerations within'
                        print 'the range for zz = %d (%7.3f - %7.3f) vs. %7.3f, Mass=%e ' \
                              % (zz, amin, amax, ar, mass)

                    amin = -1.0
                    amax = -2.0
                    continue

                #else:
                #    print 'Sampled acceleration does not fall between (amin, amax), %i' % zz 

                # Convert accelerations from mas/yr^2
                # into line of sight distances in arcsec
                #zmax = acc2z(x, y, amax, dist, mass)

                #if (isnan(zmax) == 1):
                #    print 'zmax = 0 for ', self.name
                #    zmax = 0.0001

            ##########
            # POSITIVE Z-value
            ##########
            zidx = zz
            try:
                self._runOrbit(zidx, x, y, z, vx, vy, vz, ar,
                               mass, dist, x0, y0)
            except ValueError, e:
                print 'Problem calculating orbits for POSITIVE %d, mass=%e' % (zz, mass)
                print e
                #pdb.set_trace()
                continue

            ##########
            # NEGATIVE Z-value
            ##########
            zidx = zz + (self.ntrials/2)
            try:
                self._runOrbit(zidx, x, y, -z, vx, vy, vz, ar,
                               mass, dist, x0, y0)
            except ValueError, e:
                print 'Problem calculating orbits for NEGATIVE %d, mass=%e' % (zz, mass)
                print e
                continue

        # Fix the bigOmega range
        #idxp = (where(zf >= 0))[0]
        #idxn = (where(zf < 0))[0]
        #o_lo_p = o[idxp].min()
        #o_hi_p = o[idxp].max()
        #o_lo_n = o[idxn].min()
        #o_hi_n = o[idxn].max()
        #if ((abs(o_lo_p - o_hi_p) > 180) or
        #    (abs(o_lo_n - o_hi_n) > 180)):
        #    idx = (where(o > 180))[0]
        #    o[idx] -= 360.0

        if (verbose):
            print 'FINISH: ', time.ctime(time.time())

        self.trans = None

    def runPlane(self, mcMassRo=None, verbose=True, makeplot=False):
        """
        Perform a Monte Carlo analysis on the data to determine the
        probability distribution for the orbital parameters assuming
        that the star lies on the specified orbital plane.

        This StarOrbitsMC object (and all results of the Monte Carlo)
        can be stored in a pickle file which is then reloaded using:

        import pickle
        oo = pickle.load(open('analyticOrbits/S1-2.mc.dat', 'r'))

        Input Options:
        mcMassRo -- (def=None) Set to file name containing pickled
                    bhPotential.BHprops() object.

        Available variables are:
        i,e,w,o,p,t0,ph -- Orbital parameters.
        x,y,z -- MC selected distances (arcsec)
        vx,vy,vz -- MC selected velocities (km/s)
        ar -- MC selected acceleration (mas/yr**2) in the plane of the sky
        m  -- MC selected mass (solar masses)
        r0 -- MC selected distance to the SBH (parsec)
        x0 -- MC selected X focus (pixels)
        y0 -- MC selected Y focus (pixels)
        name -- Star Name
        pdf -- 2D Probability Density Function for i and o
        """
        if (verbose):
            print 'START: ', time.ctime(time.time())

        cc = objects.Constants()

        # Make all our random number generators. We will need
        # 7 all together (x, y, vx, vy, vz, ax, ay)
        gens = create_generators(6, self.ntrials*1000)
        xgen = gens[0]
        ygen = gens[1]
        vxgen = gens[2]
        vygen = gens[3]
        vzgen = gens[4]
        axgen = gens[5]

        self._initVariables()
        self._initTransform()

        ##########
        #
        # Monte Carlo.
        #
        ##########
        for zz in range(int(self.ntrials)):
            if ((zz % 5000) == 0):
                print 'PDF Trial %d: ' % (zz), \
                      time.ctime(time.time()), self.i[zz-1]
            
            # Set temp values for our while loop
            amin = -1.0
            amax = -2.0
            aminLoopCount = 0

            # We need to make sure we always satisfy the
            # bound condition. We do this by checking that
            # amin (set by r2d) < amax (set by v=vesc). 
            while(amin > amax):
                aminLoopCount += 1
                
                # Occasionally we get cases where the randomly selected
                # velocity exceeds the escape velocity for the randomly
                # selected distance. We need to throw out these cases.
                if ((aminLoopCount % 100) == 0 and verbose):
                    print 'Random pos/vel break bound orbit criteria: zz=%d'% zz
                    print '\tamin = %7.3f  amax = %7.3f' % (amin, amax)
                    print '\tvel = %7.3f    escape = %7.3f' % \
                          (sqrt(vx**2 + vy**2 + vz**2),
                           sqrt(2.0 * GM / rcgs) / 1.0e5)
                    print '\tvx = %3d  vy = %3d  vz = %3d' % \
                          (vx, vy, vz)
                
                # Sample our monte carlo variables
                x = xgen.gauss(self.x_dat, self.xerr_dat)     # pixels
                y = ygen.gauss(self.y_dat, self.yerr_dat)     # pixels
                vx = vxgen.gauss(self.vx_dat, self.vxerr_dat) # pix/yr
                vy = vygen.gauss(self.vy_dat, self.vyerr_dat) # pix/yr
                vz = vzgen.gauss(self.vz_dat, self.vzerr_dat) # km/s

                rad = sqrt(x**2 + y**2)
                # Optional monte carlo of the mass and r0
                if mExtended != None:
                    (mass, dist, x0, y0) = self._getPotential(mcMassRo, zz, mExtended, rad)
                else:
                    (mass, dist, x0, y0) = self._getPotential(mcMassRo, zz)
                    
                GM = cc.G * mass * cc.msun        # cgs

                # Convert into positions relative to Sgr A*
                (x, y) = util.rPix2Arc(x, y, self.trans)     # asec
                (vx, vy) = util.vPix2Arc(vx, vy, self.trans) # asec/yr                
                # Convert velocities in km/s
                asy_to_kms = dist * cc.cm_in_au / (1.0e5 * cc.sec_in_yr)
                vx *= asy_to_kms
                vy *= asy_to_kms

                # At this point:
                #   all positions are in arcsec and
                #   all velocities are in km/s

                # Determine the maximum allowed acceleration (where a < 0)
                # set by assuming a bound orbit.
                amax = self._calcMaxAcc(x, y, vx, vy, vz, GM, dist, cc)

                # Determine the minimum allowed acceleration (where a < 0)
                # set by the minimum radius = 2D projected radius.
                amin = self._calcMinAcc(x, y, GM, dist, cc)

                # Choose z such that this star lies on the plane
                z = plane2z(x, y)

                # Determine the acceleration
                r2d = sqrt(x**2 + y**2) * dist * cc.cm_in_au
                r3d = sqrt(x**2 + y**2 + z**2) * dist * cc.cm_in_au

                arcgs = -GM * r2d / r3d**3
                ar = arcgs * 1000.0 * cc.sec_in_yr**2
                ar /= (cc.cm_in_au * dist)        # mas/yr^2

                # Check to make sure nothing really wierd happened.
                # We set the acceleration up here rather than in
                # the while loop down below.
                if (ar < amin or ar > amax):
                    if ((aminLoopCount % 100) == 0 and verbose):
                        print 'Orbit is not bound:  a = %7.3f' % ar
                        print '\tamin = %7.3f  amax = %7.3f' % (amin, amax)
                        print '\tx = %6.2f  y = %6.2f  z = %6.2f' % \
                              (x, y, z)
                    # Something wierd happened. Choose different
                    # x, y, vx, vy, vz.
                    amin = -1.0
                    amax = -2.0

            ##########
            # POSITIVE Z-value
            ##########
            try:
                self._runOrbit(zz, x, y, z, vx, vy, vz, ar,
                               mass, dist, x0, y0)
            except ValueError, e:
                print 'Problem calculating orbits for %d' % zidx
                print e
                continue

        if (verbose):
            print 'FINISH: ', time.ctime(time.time())

        self.trans = None

    def _initVariables(self):
        # These are all the variables that are spit out by
        # our monte carlo.
        self.i = zeros(self.ntrials, dtype=float)
        self.e = zeros(self.ntrials, dtype=float)
        self.evec = zeros((self.ntrials, 3), dtype=float)
        self.w = zeros(self.ntrials, dtype=float)
        self.o = zeros(self.ntrials, dtype=float)
        self.p = zeros(self.ntrials, dtype=float)
        self.t0 = zeros(self.ntrials, dtype=float)
        self.ph = zeros(self.ntrials, dtype=float)
        self.x = zeros(self.ntrials, dtype=float)
        self.y = zeros(self.ntrials, dtype=float)
        self.z = zeros(self.ntrials, dtype=float)
        self.vx = zeros(self.ntrials, dtype=float)
        self.vy = zeros(self.ntrials, dtype=float)
        self.vz = zeros(self.ntrials, dtype=float)
        self.ar = zeros(self.ntrials, dtype=float)
        self.m = zeros(self.ntrials, dtype=float)
        self.r0 = zeros(self.ntrials, dtype=float)
        self.x0 = zeros(self.ntrials, dtype=float)
        self.y0 = zeros(self.ntrials, dtype=float)

    def _initTransform(self):
        self.trans = objects.Transform()
        self.trans.scale = 1.0
        self.trans.scaleErr = 0.0
        self.trans.sgra = [-0.001, -0.005]
        self.trans.sgraErr = [0.0, 0.0]
        self.trans.angle = 0.0
        self.trans.angleErr = 0.0

    def _getPotential(self, mcMassRo, zz, mExtended, r2d=None):
        cc = objects.Constants()
        
        if (mcMassRo != None):
            if (mcMassRo == 'germ'):
                mass = 3.61e6          # solar masses
                dist = 7620.0          # pc

                x0 = -0.001
                y0 = -0.005
            else:
                mass = mcMassRo.m[zz]  # solar masses
                dist = mcMassRo.r0[zz] # pc
                x0 = mcMassRo.x0[zz]   # pixels
                y0 = mcMassRo.y0[zz]   # pixels

            if (mExtended != None):
                # Now sample the extended mass, if needed
                if (mExtended == 'trippe'):
                    # Include extended mass distribution from Trippe et al. (2008)
                    Mbh = mass 	# solar masses
                    rho0 = 2.1e6  	# solar masses/pc^3
                    Rb_as = 8.9	# break radius; arcsec
                    Rb = Rb_as * dist / 206265. # pc
                    #r2d = sqrt(x**2 + y**2) * dist / 206265. # pc
                    rad = r2d * dist / 206265. # pc
    
                    const = 4.0 * math.pi * rho0
                    Mext = scipy.integrate.quad(lambda r: r**2 / (1 + (r / Rb)**2), 0, rad)
                    Mext = const * Mext[0]
                    
                    mass = Mbh + Mext
    
                else:
                    # Include extended mass distribution from Schoedel et al. (2009)
                    Mbh = mass 	# solar masses
                    gamma = 1.0	# mass density power law
                    rm = 5.0 	# pc
                    #r2d = sqrt(x**2 + y**2) * dist / 206265. # pc
                    rad = r2d * dist / 206265. # pc
    
                    # Sample from rho0
                    rho0 = mExtended.rho0[zz]
    
                    const = 4.0 * math.pi * rho0
                    Mext = scipy.integrate.quad(lambda r: (r / rm)**(-gamma) * r**2, 0, rad)
                    Mext = const * Mext[0]
                    
                    mass = Mbh + Mext

        else:
            mass = 4.1e6           # solar masses
            massErr = 0.6e6        # solar masses
            dist = 7960.0          # pc
            distErr = 600.0        # pc
            x0 = -0.001            # arcsec (+ to West)
            y0 = -0.005            # arcsec (+ to North)

        self.trans.sgra = [x0, y0]

        return (mass, dist, x0, y0)

    def _calcMaxAcc(self, x, y, vx, vy, vz, GM, dist, cc):
        """
        Maximum acceleration (where v = escape velocity).
        This corresponds to the largest allowed line of sight
        distance. 
        """
        r = sqrt(x**2 + y**2)             # arcsec
        rcgs = r * dist * cc.cm_in_au
        vcgs = sqrt(vx**2 + vy**2 + vz**2) * 1.0e5
        amax_cgs = -rcgs * vcgs**6 / (8.0 * GM**2)
        amax = amax_cgs * 1000.0 * cc.sec_in_yr**2
        amax /= (cc.cm_in_au * dist)      # mas/yr^2

        return amax

    def _calcMinAcc(self, x, y, GM, dist, cc):
        r = sqrt(x**2 + y**2)             # arcsec
        rcgs = r * dist * cc.cm_in_au
        amin_cgs = -GM / rcgs**2
        amin = amin_cgs * 1000.0 * cc.sec_in_yr**2
        amin /= (cc.cm_in_au * dist)      # mas/yr^2

        return amin

    def _calcMinAccErr(self, x, y, GM, dist, cc):
        r = sqrt(x**2 + y**2)             # arcsec
        rcgs = r * dist * cc.cm_in_au
        amin_cgs = -GM / rcgs**2
        amin = amin_cgs * 1000.0 * cc.sec_in_yr**2
        amin /= (cc.cm_in_au * dist)      # mas/yr^2

        amin_err_cgs = cc.G / rcgs**2 * cc.massErr * cc.msun
        amin_err = amin_err_cgs * 1000.0 * cc.sec_in_yr**2
        amin_err /= (cc.cm_in_au * dist)      # mas/yr^2

        return amin, amin_err

    def _runOrbit(self, zidx, x, y, z, vx, vy, vz, ar, mass, dist, x0, y0):
        rvec = array([x, y, z])
        vvec = array([vx, vy, vz])
        revec = zeros(3, dtype=float)
        vevec = zeros(3, dtype=float)

        orb = orbits.Orbit()
        orb.xyz2kep(rvec, vvec, revec, vevec, self.refTime,
                    mass=mass, dist=dist)

        self.i[zidx] = orb.i
        self.e[zidx] = orb.e
        self.evec[zidx] = orb.evec
        self.w[zidx] = orb.w
        self.o[zidx] = orb.o
        self.p[zidx] = orb.p
        self.t0[zidx] = orb.t0
        self.ph[zidx] = orb.ph
        self.x[zidx] = x
        self.y[zidx] = y
        self.z[zidx] = z
        self.vx[zidx] = vx
        self.vy[zidx] = vy
        self.vz[zidx] = vz
        self.ar[zidx] = ar
        self.m[zidx] = mass
        self.r0[zidx] = dist
        self.x0[zidx] = x0
        self.y0[zidx] = y0



    def makePdfHealpix(self, nside=64, makeplot=False):
	"""
	Make a 2D histogram of the inclination and PA to the 
	ascending node. The points of the monte carlo are distributed
	on a HEALPix map of the sky.
	"""
        #npix = healpix.nside2npix(nside)
        npix = healpy.nside2npix(nside)

	# Determine which pixel in the map each of the
	# points goes (2D histogram)
	incl = self.i * math.pi / 180.0
	omeg = self.o * math.pi / 180.0

	#hidx = healpix.ang2pix_ring(nside, incl, omeg)
	hidx = healpy.ang2pix(nside, incl, omeg)

	# Star's PDF
	pdf = zeros(npix, dtype=float)
        for hh in hidx:
            pdf[hh] += 1.0
	pdf /= self.ntrials

        if (makeplot):
            mcFile = '%s%s_mc_heal.dat' % (self.outroot, self.name)
            pdf.tofile(mcFile)

            #icmd = "idl -e 'plot_disk_healpix, \"%s%s_mc_heal.dat\", %d, %d'" \
            #          % (self.outroot, self.name, npix, 1)
            #os.system(icmd)
            plot_disk_healpix.go(mcFile, npix, 1)
            
	return pdf

    def saveToFile(self, savefile):
        _f = open(savefile, 'w')
        pickle.dump(self, _f)
        _f.close()




class Disk(object):
    def __init__(self, root, mcdir, nside=64, outdir='', mscDir=None):
        self.nside = nside
        self.npix = healpy.nside2npix(self.nside)
        #self.npix = healpix.nside2npix(self.nside)
        self.mcdir = mcdir
        self.root = root
        self.outdir = outdir
        if mscDir != None:
            self.polyM = 'polyfit_1000/fit'
            self.pointsM = 'points_1000/'

        # Load names of young stars 
        if mscDir != None:
            # Load up mosaic data as well; select only stars at r>4, since
            # we don't want to add any info from mosaics if we have it in
            # the central 10" already
            yng1 = young.loadYoungStars(self.root, withRVonly=True)
            yng2 = young.loadYoungStars(mscDir,fit=self.polyM, withRVonly=True,
                                        points=self.pointsM, mosaic=True)
            # Merge this object with object from central 10" analysis
            yng = merge(yng1, yng2)

            names = yng.getArray('name')
            r2d = yng.getArray('r2d')
            xorig = yng.getArray('x')
            yorig = yng.getArray('y')
        else:
            yng = young.loadYoungStars(self.root, withRVonly=True)
            names = yng.getArray('name')
            r2d = yng.getArray('r2d')
            xorig = yng.getArray('x')
            yorig = yng.getArray('y')

        self.names = names
        self.r2d = r2d
        self.xorig = xorig
        self.yorig = yorig


    def run(self, makeplot=False, do_all=True, do_radial_bins=False,
            do_radial_bins_cntrl=False, lu09_sample=False):
        nstars = len(self.names)

        # Load up i, omega and projected radius for all stars, all trials
        #  -- iomap = sum of all PDFs
        #  -- ioCntMap = number of trials at each position
        iAll, oAll, r2d = self.loadStars(return_r2d=True)

        # Save the marginalized PDF to a file.
        self.iomap.tofile('%s/%s/disk.heal.dat' % \
                          (self.mcdir, self.outdir))
        self.ioCntMap.tofile('%s/%s/disk.iocnt.dat' % \
                             (self.mcdir, self.outdir))

        # Optional plotting
        if (makeplot):
            mcFile = '%s/%s/%s/disk.heal.dat' % (self.root, self.mcdir, self.outdir)
            plot_disk_healpix.go(mcFile, self.npix, 1)
            #icmd = "idl -e 'plot_disk_healpix, "
            #icmd += "\"%s/%s/%s/disk.heal.dat\", %d, 1'" % \
            #        (self.root, self.mcdir, self.outdir, self.npix)
            #os.system(icmd)

        if do_all == True:
            # Map out the PDF for the density of normal vectors
            (neigh, neighStd, peakD, peakI, peakO) = self.densityPDF(iAll, oAll)
            
            # Save the i/o density maps
            print 'Making density map'
            neigh.tofile('%s/%s/disk.neighbor.dat' % \
                         (self.mcdir, self.outdir))
            neighStd.tofile('%s/%s/disk.neighborStd.dat' % \
                            (self.mcdir, self.outdir))
            peakD.tofile('%s/%s/disk.peakDensity.dat' % \
                         (self.mcdir, self.outdir))
            peakI.tofile('%s/%s/disk.peakIncli.dat' % \
                         (self.mcdir, self.outdir))
            peakO.tofile('%s/%s/disk.peakOmega.dat' % \
                         (self.mcdir, self.outdir))

        if do_radial_bins == True:
            rad1 = where(r2d < 3.5)[0]
            i_inner = iAll[rad1,:]
            o_inner = oAll[rad1,:]

            rad2 = where((r2d >= 3.5) & (r2d < 7.0))[0]
            i_middle = iAll[rad2,:]
            o_middle = oAll[rad2,:]

            # TEMPORARY:
            # excluding stars due north in the outer radial bin
#            rad3 = where((r2d >= 7.0) & (self.yorig < 8.0))[0]
#            print [self.names[rr] for rr in rad3]
            # END TEMPORARY

            rad3 = where(r2d >= 7.0)[0]
            i_outer = iAll[rad3,:]
            o_outer = oAll[rad3,:]

            # Radial bin 1
            # Map out the PDF for the density of normal vectors
            (neigh, neighStd, peakD, peakI, peakO) = self.densityPDF(i_inner, o_inner)
            
            # Save the i/o density maps
            print 'Making density map for stars at r < 3.5 arcsec'
            neigh.tofile('%s/%s/inner_disk.neighbor.dat' % \
                         (self.mcdir, self.outdir))
            neighStd.tofile('%s/%s/inner_disk.neighborStd.dat' % \
                            (self.mcdir, self.outdir))
            peakD.tofile('%s/%s/inner_disk.peakDensity.dat' % \
                         (self.mcdir, self.outdir))
            peakI.tofile('%s/%s/inner_disk.peakIncli.dat' % \
                         (self.mcdir, self.outdir))
            peakO.tofile('%s/%s/inner_disk.peakOmega.dat' % \
                         (self.mcdir, self.outdir))

            # Radial bin 2
            # Map out the PDF for the density of normal vectors
            (neigh, neighStd, peakD, peakI, peakO) = self.densityPDF(i_middle, o_middle)
            
            ## Save the i/o density maps
            print 'Making density map for stars at 3.5 <= r < 7.0 arcsec'
            neigh.tofile('%s/%s/middle_disk.neighbor.dat' % \
                         (self.mcdir, self.outdir))
            neighStd.tofile('%s/%s/middle_disk.neighborStd.dat' % \
                            (self.mcdir, self.outdir))
            peakD.tofile('%s/%s/middle_disk.peakDensity.dat' % \
                         (self.mcdir, self.outdir))
            peakI.tofile('%s/%s/middle_disk.peakIncli.dat' % \
                         (self.mcdir, self.outdir))
            peakO.tofile('%s/%s/middle_disk.peakOmega.dat' % \
                         (self.mcdir, self.outdir))

            ####
            # Temporary! Removes the problematic stars with i=90 deg
            # until I create a criterion that gets rid of them!
            #####
            #i90stars = ['S1-1', 'S1-24', 'S2-66', 'S3-2', 'S5-34', 'S5-187', 'S5-235',
            #            'S5-236', 'S7-180', 'S8-4', 'S8-196', 'S9-9', 'S9-114', 'S9-283',
            #            'irs13E1', 'irs13E3b', 'irs29N']
            ## Just get names that are not in common with these stars
            #kp = []
            #for jj in range(len(self.names)):
            #    if self.names[jj] in i90stars:
            #        continue
            #    else:
            #        kp = np.concatenate([kp, [jj]])
            #kp = [int(ii) for ii in kp] # indices we want to keep
            ## Now we want just the ones in the outer radial bin that are not in i90stars
            #good = np.intersect1d(kp, rad3)
            #i_outer = iAll[good,:]
            #o_outer = oAll[good,:]
            #####
            # End temporary
            #####

            # Radial bin 3
            # Map out the PDF for the density of normal vectors
            (neigh, neighStd, peakD, peakI, peakO) = self.densityPDF(i_outer, o_outer)
            
            # Save the i/o density maps
            print 'Making density map for stars at r >= 7.0 arcsec'
            neigh.tofile('%s/%s/outer_disk.neighbor.dat' % \
                         (self.mcdir, self.outdir))
            neighStd.tofile('%s/%s/outer_disk.neighborStd.dat' % \
                            (self.mcdir, self.outdir))
            peakD.tofile('%s/%s/outer_disk.peakDensity.dat' % \
                         (self.mcdir, self.outdir))
            peakI.tofile('%s/%s/outer_disk.peakIncli.dat' % \
                         (self.mcdir, self.outdir))
            peakO.tofile('%s/%s/outer_disk.peakOmega.dat' % \
                         (self.mcdir, self.outdir))

        if do_radial_bins_cntrl == True:
            # Split up the data into 3 radial bins within the central 10 arcsec FOV
            rad1 = where(r2d < 2.266)[0]
            i_inner = iAll[rad1,:]
            o_inner = oAll[rad1,:]

            rad2 = where((r2d > 2.266) & (r2d <= 3.538))[0]
            i_middle = iAll[rad2,:]
            o_middle = oAll[rad2,:]

            rad3 = where(r2d >= 3.538)[0]
            i_outer = iAll[rad3,:]
            o_outer = oAll[rad3,:]

            # Radial bin 1
            # Map out the PDF for the density of normal vectors
            (neigh, neighStd, peakD, peakI, peakO) = self.densityPDF(i_inner, o_inner)
            
            # Save the i/o density maps
            print 'Making density map for stars at r < 2.266 arcsec'
            neigh.tofile('%s/%s/inner_disk_cntrl.neighbor.dat' % \
                         (self.mcdir, self.outdir))
            neighStd.tofile('%s/%s/inner_disk_cntrl.neighborStd.dat' % \
                            (self.mcdir, self.outdir))
            peakD.tofile('%s/%s/inner_disk_cntrl.peakDensity.dat' % \
                         (self.mcdir, self.outdir))
            peakI.tofile('%s/%s/inner_disk_cntrl.peakIncli.dat' % \
                         (self.mcdir, self.outdir))
            peakO.tofile('%s/%s/inner_disk_cntrl.peakOmega.dat' % \
                         (self.mcdir, self.outdir))

            # Radial bin 2
            # Map out the PDF for the density of normal vectors
            (neigh, neighStd, peakD, peakI, peakO) = self.densityPDF(i_middle, o_middle)
            
            ## Save the i/o density maps
            print 'Making density map for stars at 2.266 <= r < 3.538 arcsec'
            neigh.tofile('%s/%s/middle_disk_cntrl.neighbor.dat' % \
                         (self.mcdir, self.outdir))
            neighStd.tofile('%s/%s/middle_disk_cntrl.neighborStd.dat' % \
                            (self.mcdir, self.outdir))
            peakD.tofile('%s/%s/middle_disk_cntrl.peakDensity.dat' % \
                         (self.mcdir, self.outdir))
            peakI.tofile('%s/%s/middle_disk_cntrl.peakIncli.dat' % \
                         (self.mcdir, self.outdir))
            peakO.tofile('%s/%s/middle_disk_cntrl.peakOmega.dat' % \
                         (self.mcdir, self.outdir))

            # Radial bin 3
            # Map out the PDF for the density of normal vectors
            (neigh, neighStd, peakD, peakI, peakO) = self.densityPDF(i_outer, o_outer)
            
            # Save the i/o density maps
            print 'Making density map for stars at r >= 3.538 arcsec'
            neigh.tofile('%s/%s/outer_disk_cntrl.neighbor.dat' % \
                         (self.mcdir, self.outdir))
            neighStd.tofile('%s/%s/outer_disk_cntrl.neighborStd.dat' % \
                            (self.mcdir, self.outdir))
            peakD.tofile('%s/%s/outer_disk_cntrl.peakDensity.dat' % \
                         (self.mcdir, self.outdir))
            peakI.tofile('%s/%s/outer_disk_cntrl.peakIncli.dat' % \
                         (self.mcdir, self.outdir))
            peakO.tofile('%s/%s/outer_disk_cntrl.peakOmega.dat' % \
                         (self.mcdir, self.outdir))

        if lu09_sample == True:
            luNames = array(['S0-14', 'S0-15', 'S1-3', 'S1-2',
                             'S1-8', 'irs16NW', 'irs16C', 'S1-12',
                             'S1-14', 'irs16SW', 'S1-21', 'S1-22',
                             'S1-24', 'S2-4', 'irs16CC', 'S2-6',
                             'S2-7', ' irs29N', 'irs16SW-E', 'irs33N',
                             'S2-17', 'S2-16', 'S2-19', 'S2-66',
                             'S2-74', 'irs16NE', 'S3-5', 'irs33E',
                             'S3-19', 'S3-25', 'S3-30', 'S3-10'])
            lu09 = []
            for ll in luNames:
                l = np.where(np.array(self.names) == ll)[0]
                lu09 = concatenate([lu09, l])
            lu09 = [int(ll) for ll in lu09]
             
            i_lu09 = iAll[lu09,:]
            o_lu09 = oAll[lu09,:]

            # Map out the PDF for the density of normal vectors
            (neigh, neighStd, peakD, peakI, peakO) = self.densityPDF(i_lu09, o_lu09)
            
            # Save the i/o density maps
            print 'Making density map for stars at r < 3.5 arcsec'
            neigh.tofile('%s/%s/lu09rpt_disk.neighbor.dat' % \
                         (self.mcdir, self.outdir))
            neighStd.tofile('%s/%s/lu09rpt_disk.neighborStd.dat' % \
                            (self.mcdir, self.outdir))
            peakD.tofile('%s/%s/lu09rpt_disk.peakDensity.dat' % \
                         (self.mcdir, self.outdir))
            peakI.tofile('%s/%s/lu09rpt_disk.peakIncli.dat' % \
                         (self.mcdir, self.outdir))
            peakO.tofile('%s/%s/lu09rpt_disk.peakOmega.dat' % \
                         (self.mcdir, self.outdir))
            
    def sample_wr(self, samples, rbin):
        """
        Chooses k random elements (with replacement) from a population.
        Runs densityPDF() for each radial bin, where the stars selected
        for input into densityPDF() are sampled with replacement.
        Uses 6 nearest neighbors.

        Input:
        samples = Number of times to sample with replacement
        rbin    = String indicating the radial bin to run the analysis on.
         	  Options are 'r1', 'r2', 'r3', or, to do all stars in the
                  sample, use 'all'.
                  	--This allows for running each radial bin in parallel
                          by specifying which bin to run in separate python calls.
        """

        # Load up i, omega for all stars, all trials
        iAll, oAll, r2d = self.loadStars(return_r2d=True)

        # We want to sample with replacement WITHIN each radial bin.
        # So we need to send the indices for each radial bin
        if rbin == 'r1':
            rad = where(r2d <= 2.266)[0]
            suffix = rbin
            print 'Number of stars in inner radial bin: %i' % len(rad)
        elif rbin == 'r2':
            rad = where((r2d > 2.266) & (r2d <= 3.538))[0]
            suffix = rbin
            print 'Number of stars in middle radial bin: %i' % len(rad)
        elif rbin == 'r3':
            rad = where(r2d > 3.538)[0]
            suffix = rbin
            print 'Number of stars in outer radial bin: %i' % len(rad)
        elif rbin == 'all':
            rad = where(r2d > 0.8)[0]
            suffix = rbin
            print 'Number of stars in total sample: %i' % len(rad)
            
        # some stuff for randomly choosing stars with replacement
        nstars = len(rad)
        idStars = rad

        for tt in range(samples):
            _random, _int = random.random, int  # speed hack 
            result = [None] * nstars
            for i in xrange(nstars):
                j = _int(_random() * nstars)
                result[i] = idStars[j]  # contains indices for SWR

            strtt = str(tt).zfill(4)   # string version
            print 'Sample with replacement for %s trial %s' % \
                  (suffix, strtt)

            sampleNames = [self.names[n] for n in result]
            for ss in sampleNames:
                print ss

            iSample = iAll[result,:]
            oSample = oAll[result,:]

            # Since we are sampling with replacement, a star's PDF
            # can be sampled multiple times in the same trial. To prevent
            # correlated PDFs, we need to randomize the solutions for a
            # given star before making the nearest neighbor density map
            for rr in range(len(result)):
                # First shuffle the indices across the MC trials:
                ii = random.sample(arange(len(iSample[0])), len(iSample[0]))
                # Now re-order the trials for this star
                iSample[rr,:] = iSample[rr,ii]
                oSample[rr,:] = oSample[rr,ii]

            # Map out the PDF for the density of normal vectors
            (neigh, neighStd, peakD, peakI, peakO) = \
                    self.densityPDF(iSample, oSample, neighbors=[6],
                                    aperture=False, sampleWR=strtt)

            # Save the i/o density maps
            neigh.tofile('%ssample_wr/disk.neighbor_%s_%s.dat' % \
                         (self.mcdir, suffix, strtt))
            neighStd.tofile('%ssample_wr/disk.neighborStd_%s_%s.dat' % \
                            (self.mcdir, suffix, strtt))
            peakD.tofile('%ssample_wr/disk.peakDensity_%s_%s.dat' % \
                            (self.mcdir, suffix, strtt))
            peakI.tofile('%ssample_wr/disk.peakIncli_%s_%s.dat' % \
                            (self.mcdir, suffix, strtt))
            peakO.tofile('%ssample_wr/disk.peakOmega_%s_%s.dat' % \
                            (self.mcdir, suffix, strtt))

    
            
    def bootstrap(self, samples, suffix=''):
        """
        Half-sample bootstrap on disk analysis.
        Uses only 5th nearest neighbor density measurement.
        """
        # some stuff for randomly choosing half
        nstars = len(self.names)
        halfPoint = int(ceil(nstars / 2.0))
        idStars = arange(nstars)

        # Load up i, omega for all stars, all trials
        iAll, oAll = self.loadStars()

        # Loop through number of samples and calc density for each
        for tt in range(samples):
            strtt = str(tt).zfill(4)   # string version
            print 'Bootstrap %s for %s' % (strtt, suffix)

            # Randomly select half (no replacement)
            random.shuffle(idStars)
            idHalf = idStars[0:halfPoint]
            sampleNames = [self.names[n] for n in idHalf]
 
            iSample = iAll.take(idHalf)
            oSample = oAll.take(idHalf)
            
            # Map out the PDF for the density of normal vectors
            (neigh, neighStd, peakD, peakI, peakO) = \
                    self.densityPDF(iSample, oSample, neighbors=[6],
                                    aperture=False)
            
            # Save the i/o density maps
            neigh.tofile('%sbootstrap/disk.neighbor%s_%s.dat' % \
                         (self.mcdir, suffix, strtt))
            neighStd.tofile('%sbootstrap/disk.neighborStd%s_%s.dat' % \
                            (self.mcdir, suffix, strtt))
            peakD.tofile('%sbootstrap/disk.peakDensity%s_%s.dat' % \
                            (self.mcdir, suffix, strtt))
            peakI.tofile('%sbootstrap/disk.peakIncli%s_%s.dat' % \
                            (self.mcdir, suffix, strtt))
            peakO.tofile('%sbootstrap/disk.peakOmega%s_%s.dat' % \
                            (self.mcdir, suffix, strtt))


    def loadStars(self, return_r2d=False):
        self.iomap = zeros((1, self.npix), dtype=float)
        self.ioCntMap = zeros((1, self.npix), dtype=float)
        
        iAll = None
        oAll = None
        nstars = len(self.names)
        for ss in range(nstars):
            name = self.names[ss]

            try:
                # Read in the 6D Probability distribution of orbital
                # parameters for this star.
                _f = open('%s/%s.mc.dat' % (self.mcdir, name), 'r')
                mc = pickle.load(_f)
                _f.close()
                print 'Adding %15s (%d) to disk' % (name, len(mc.i))
    
                # Marginalize the PDF for this star onto just
                # incl (i) and PA to ascending node (o).
                pdf = mc.makePdfHealpix(nside=self.nside)
    
                # Store the monte carlo results as (i, o) pairs.
                if (iAll == None):
                    self.pdftrials = len(mc.i)
                    iAll = zeros((nstars, self.pdftrials), dtype=float)
                    oAll = zeros((nstars, self.pdftrials), dtype=float)
                     
                # This assumes that the Mass/r0/x0/y0 values are the
                # same for a single trial across all stars.
                iAll[ss,:] = mc.i
                oAll[ss,:] = mc.o

                self.iomap[0] += pdf
                self.ioCntMap[0] += (pdf / pdf.max())
            except IOError:
                print 'No MC file for %s; no RV measurement' % name

        # Get the (i,o) values for each pixel in the sky
        iAll *= math.pi / 180.0
        oAll *= math.pi / 180.0

        if return_r2d:
            return iAll, oAll, self.r2d
        else:
            return iAll, oAll

    def densityPDF(self, iAll, oAll, neighbors=[4,5,6,7], aperture=True, sampleWR=None):
        """
        Map out the PDF for the density of normal vectors
        """
        nstars = iAll.shape[0]
        print nstars, type(nstars)
        trials = 62000
        #trials = 100000
        npdfs = len(neighbors)

        pixIdx = arange(self.npix, dtype=int)
        (ipix, opix) = healpy.pix2ang(self.nside, pixIdx)
        #(ipix, opix) = healpix.pix2ang_ring(self.nside, pixIdx)
        sinip = sin(ipix)
        cosip = cos(ipix)

        siniAll = sin(iAll)
        cosiAll = cos(iAll)

        onesNpix = ones(self.npix, dtype=float)
        onesNstars = ones(nstars, dtype=float)
        factor = 2.0 * math.pi * (180.0 / math.pi)**2

        if (aperture == True):
            # We will be using nearest neighbor and aperture densities.
            # Pre-calc some stuff for the aperture densities.
            #angCut = 0.1745  # radians (10 deg )
            angCut = 0.1047  # radians (6 deg )
            angCutArea = factor * (1 - cos(angCut)) # area in deg^2
            
        if (trials > self.pdftrials):
            print 'Must have more PDF trials than disk trials'
        
        # Compute the PDF for the density at each pixel along
        # with the weighted average density at each pixel.
        neighborMap = zeros((npdfs, self.npix), dtype=float)
        neighborMapStd = zeros((npdfs, self.npix), dtype=float)

        # Keep track of the peak density and position for each trial
        peakDensity = zeros((npdfs, trials), dtype=float)
        peakIncli = zeros((npdfs, trials), dtype=float)
        peakOmega = zeros((npdfs, trials), dtype=float)

        if (aperture == True):
            # Also record maps from aperture density calculation
            apertureMap = zeros(self.npix, dtype=float)
            apertureMapStd = zeros(self.npix, dtype=float)
            
            # Keep track of the peak density and position for each trial
            peakDensityAp = zeros(trials, dtype=float)
            peakIncliAp = zeros(trials, dtype=float)
            peakOmegaAp = zeros(trials, dtype=float)

        if sampleWR != None:
            _out1 = open('%s/%s/sample_wr/disk_nn_results_%s.txt' % \
                         (self.mcdir, self.outdir, sampleWR), 'w')
            if aperture == True:
                _out2 = open('%s/%s/sample_wr/disk_ap_results_%s.txt' % \
                             (self.mcdir, self.outdir, sampleWR), 'w')
        else:
            _out1 = open('%s/%s/disk_nn_results.txt' % \
                         (self.mcdir, self.outdir), 'w')
            if aperture == True:
                _out2 = open('%s/%s/disk_ap_results.txt' % \
                             (self.mcdir, self.outdir), 'w')

        print 'Running MC to obtain density map.'

        # temp
        import Numeric
        # end temp
        
        for ii in range(trials):
            if ((ii % 100) == 0):
                print 'Trial %d' % ii, time.ctime(time.time())
            
            # Randomly select an (i,o) pair out of each star's
            # marginalized PDF.
            incl = iAll[:, ii]
            omeg = oAll[:, ii]
            sini = siniAll[:, ii]
            cosi = cosiAll[:, ii]
            
            # Check for bad things
            idx = (where((incl == float('nan')) |
                         (omeg == float('nan'))))[0]
            if (len(idx) > 0):
                print ii, idx
                    
            # Find densities
            #omegSq = Numeric.outerproduct(omeg, onesNpix)
            #opixSq = Numeric.outerproduct(onesNstars, opix)
            omegSq = np.outer(omeg, onesNpix)
            opixSq = np.outer(onesNstars, opix)
            cosodiff = cos(opixSq - omegSq)

            #sinSq = Numeric.outerproduct(sini, sinip)
            #cosSq = Numeric.outerproduct(cosi, cosip)
            sinSq = np.outer(sini, sinip)
            cosSq = np.outer(cosi, cosip)

            # Angular offset from each pixel for all stars (radians)
            angOff = arccos( (sinSq * cosodiff) + cosSq )
            angOff.sort(axis=0)

            # Density per square degree from nearest neighbor
            # Solid angle is 2 * pi * (1 - cos theta)
            for nn in range(npdfs):
                # Nearest neighbor algorithm
                nth = neighbors[nn]
                densityMap = nth / (factor*(1.0 - cos(angOff[nth-1,:])))
                maxPix = densityMap.argmax()

                neighborMap[nn,:] += densityMap
                neighborMapStd[nn,:] += densityMap**2
                peakDensity[nn,ii] = densityMap[maxPix]
                peakOmega[nn,ii] = opix[maxPix]
                peakIncli[nn,ii] = ipix[maxPix]

                # Save to an output file
                if (nn == 2):
                    #fubar = array(densityMap * 10**5, dtype=int16)
                    fubar = array(densityMap * 10**5, dtype=int64)
                    fubar.tofile(_out1)
                    fubar = None

                #pickle.dump(densityMap, nnFiles[nn])

            if (aperture == True):
                # Aperture density:
                densityMapAp = zeros(self.npix, dtype=float32)
                pidx = (where(angOff < angCut))[1]
                ifreq = (stats.itemfreq(pidx)).astype('int')
                for pix, cnt in ifreq:
                    densityMapAp[pix] = cnt / angCutArea

                maxPix = densityMapAp.argmax()

                apertureMap += densityMapAp
                apertureMapStd += densityMapAp**2
                peakDensityAp[ii] = apertureMap[maxPix]
                peakOmegaAp[ii] = opix[maxPix]
                peakIncliAp[ii] = ipix[maxPix]
                
                # Save to an output file
                #fubar = array(densityMapAp * 10**5, dtype=int16)
                fubar = array(densityMapAp * 10**5, dtype=int64)
                fubar.tofile(_out2)
                fubar = None

        neighborMap /= trials
        neighborMapStd = sqrt( (neighborMapStd / trials) - neighborMap**2 )

        if (aperture == True):
            apertureMap /= trials
            apertureMapStd = sqrt( (apertureMapStd / trials) - apertureMap**2 )

            # Save the i/o density maps
            print 'Making density map'
            apertureMap.tofile('%s/%s/disk.aperture.dat' % \
                               (self.mcdir, self.outdir))
            apertureMapStd.tofile('%s/%s/disk.apertureStd.dat' % \
                                  (self.mcdir, self.outdir))
            peakDensityAp.tofile('%s/%s/disk.peakDensityAp.dat' % \
                                 (self.mcdir, self.outdir))
            peakIncliAp.tofile('%s/%s/disk.peakIncliAp.dat' % \
                               (self.mcdir, self.outdir))
            peakOmegaAp.tofile('%s/%s/disk.peakOmegaAp.dat' % \
                               (self.mcdir, self.outdir))

        return (neighborMap, neighborMapStd, peakDensity, peakIncli, peakOmega)
        
    def densityPDFpp(self, iAll, oAll, neighbors=[4,5,6,7], aperture=True):
        """
        Parallel Processing Version:
        Map out the PDF for the density of normal vectors
        """
        import ipython1.kernel.api as kernel
        ipc = kernel.RemoteController(('127.0.0.1', 10105))
        #ipc.executeAll( 'from numarrray import *' )
        #ipc.executeAll( 'from scipy import stats' )
        nodeIDs = ipc.getIDs()
        
        nstars = iAll.shape[0]
        #trials = 100000
        trials = 100
        npdfs = len(neighbors)

        pixIdx = arange(self.npix, dtype=int)
        (ipix, opix) = healpy.pix2ang(self.nside, pixIdx)
        #(ipix, opix) = healpix.pix2ang_ring(self.nside, pixIdx)
        sinip = sin(ipix)
        cosip = cos(ipix)

        siniAll = sin(iAll)
        cosiAll = cos(iAll)

        onesNpix = ones(self.npix, dtype=float)
        onesNstars = ones(nstars, dtype=float)
        factor = 2.0 * math.pi * (180.0 / math.pi)**2

        if (aperture == True):
            # We will be using nearest neighbor and aperture densities.
            # Pre-calc some stuff for the aperture densities.
            #angCut = 0.1745  # radians (10 deg )
            angCut = 0.1047  # radians (6 deg )
            angCutArea = factor * (1 - cos(angCut)) # area in deg^2
            
        if (trials > self.pdftrials):
            print 'Must have more PDF trials than disk trials'
        
        # Compute the PDF for the density at each pixel along
        # with the weighted average density at each pixel.
        neighborMap = zeros((npdfs, self.npix), dtype=float)
        neighborMapStd = zeros((npdfs, self.npix), dtype=float)

        # Also record maps from aperture density calculation
        apertureMap = zeros(self.npix, dtype=float)
        apertureMapStd = zeros(self.npix, dtype=float)

        for ii in nodeIDs:
            ipc.push(ii, _out1 = open('%s/parallel/disk_nn_results_%d.txt' %
                                      (self.mcdir, ii), 'w'))
            ipc.push(ii, _out2 = open('%s/parallel/disk_ap_results_%d.txt' %
                                      (self.mcdir, ii), 'w'))

        ipc.pushAll( sinip )
        ipc.pushAll( cosip )
        ipc.pushAll( ipix )
        ipc.pushAll( opix )
        ipc.pushAll( factor )
        ipc.pushAll( npdfs )
        ipc.pushAll( neighbors )
        ipc.pushAll( angCut )
        ipc.pushAll( angCutArea )
        ipc.pushAll( iAll )
        ipc.pushAll( oAll )
        ipc.pushAll( siniAll )
        ipc.pushAll( cosiAll )
        ipc.pushAll( onesNpix )
        ipc.pushAll( oneNstars )
        ipc.pushAll( neighborMap )
        ipc.pushAll( neighborMapStd )
        ipc.pushAll( apertureMap )
        ipc.pushAll( apertureMapStd )

        rngTrials = range(trials)

        node_run = """
        trialCnt = len(rngTrials)

        # Keep track of the peak density and position for each trial
        peakDensity = zeros((npdfs, trialCnt), dtype=float)
        peakIncli = zeros((npdfs, trialCnt), dtype=float)
        peakOmega = zeros((npdfs, trialCnt), dtype=float)
        
        # Keep track of the peak density and position for each trial
        peakDensityAp = zeros(trialCnt, dtype=float)
        peakIncliAp = zeros(trialCnt, dtype=float)
        peakOmegaAp = zeros(trialCnt, dtype=float)

        for i in range(len(rngTrials)):
            ii = rngTrials[i]
            
            if ((ii % 100) == 0):
                print 'Trial %d' % ii, time.ctime(time.time())
            
            # Randomly select an (i,o) pair out of each star's
            # marginalized PDF.
            incl = iAll[:, ii]
            omeg = oAll[:, ii]
            sini = siniAll[:, ii]
            cosi = cosiAll[:, ii]
            
            # Check for bad things
            idx = (where((incl == float('nan')) |
                         (omeg == float('nan'))))[0]
            if (len(idx) > 0):
                print ii, idx
                    
            # Find densities
            #omegSq = outerproduct(omeg, onesNpix)
            #opixSq = outerproduct(onesNstars, opix)
            omegSq = outer(omeg, onesNpix)
            opixSq = outer(onesNstars, opix)
            cosodiff = cos(opixSq - omegSq)

            #sinSq = outerproduct(sini, sinip)
            #cosSq = outerproduct(cosi, cosip)
            sinSq = outer(sini, sinip)
            cosSq = outer(cosi, cosip)

            # Angular offset from each pixel for all stars (radians)
            angOff = arccos( (sinSq * cosodiff) + cosSq )
            angOff.sort(axis=0)

            # Density per square degree from nearest neighbor
            # Solid angle is 2 * pi * (1 - cos theta)
            for nn in range(npdfs):
                # Nearest neighbor algorithm
                nth = neighbors[nn]
                densityMap = nth / (factor*(1.0 - cos(angOff[nth-1,:])))
                maxPix = densityMap.argmax()

                neighborMap[nn,:] += densityMap
                neighborMapStd[nn,:] += densityMap**2
                peakDensity[nn,i] = densityMap[maxPix]
                peakOmega[nn,i] = opix[maxPix]
                peakIncli[nn,i] = ipix[maxPix]

                # Save to an output file
                if (nn == 2):
                    #fubar = array(densityMap * 10**5, dtype=int16)
                    fubar = array(densityMap * 10**5, dtype=int64)
                    fubar.tofile(_out1)
                    fubar = None

            # Aperture density:
            densityMapAp = zeros(len(densityMap, dtype=float32)
            pidx = (where(angOff < angCut))[1]
            ifreq = (stats.itemfreq(pidx)).astype('int')
            for pix, cnt in ifreq:
                densityMapAp[pix] = cnt / angCutArea

            maxPix = densityMapAp.argmax()

            apertureMap += densityMapAp
            apertureMapStd += densityMapAp**2
            peakDensityAp[ii] = apertureMap[maxPix]
            peakOmegaAp[ii] = opix[maxPix]
            peakIncliAp[ii] = ipix[maxPix]
            
            # Save to an output file
            #fubar = array(densityMapAp * 10**5, dtype=int16)
            fubar = array(densityMapAp * 10**5, dtype=int64)
            fubar.tofile(_out2)
            fubar = None
            """

        ipc.pushAll('node_run')
        ipc.scatterAll('rngTrials', rngTrials)

        print 'Running MC to obtain density map.'
        ipc.executeAll('exec node_run')

        all_neighborMap = ipc.gatherAll('neighborMap')
        all_neighborMapStd = ipc.gatherAll('neighborMapStd')
        all_peakDensity = ipc.gatherAll('peakDensity')
        all_peakIncli = ipc.gatherAll('peakIncli')
        all_peakOmega = ipc.gatherAll('peakOmega')

        all_apertureMap = ipc.gatherAll('apertureMap')
        all_apertureMapStd = ipc.gatherAll('apertureMapStd')
        all_peakDensityAp = ipc.gatherAll('peakDensityAp')
        all_peakIncliAp = ipc.gatherAll('peakIncliAp')
        all_peakOmegaAp = ipc.gatherAll('peakOmegaAp')

        print all_neighborMap.shape
        print all_peakDensity.shape

        neighborMap = total(all_neighborMap, axis=0)
        neighborMapStd = total(all_neighborMapStd, axis=0)
        apertureMap = total(all_apertureMap, axis=0)
        apertureMapStd = total(all_apertureMapStd, axis=0)

        peakDensity = concatenate(all_peakDensity)
        peakIncli = concatenate(all_peakIncli)
        peakOmega = concatenate(all_peakOmega)
        peakDensityAp = concatenate(all_peakDensityAp)
        peakIncliAp = concatenate(all_peakIncliAp)
        peakOmegaAp = concatenate(all_peakOmegaAp)

        neighborMap /= trials
        neighborMapStd = sqrt( (neighborMapStd / trials) - neighborMap**2 )

        apertureMap /= trials
        apertureMapStd = sqrt( (apertureMapStd / trials) - apertureMap**2 )
                     
        # Save the i/o density maps
        print 'Making density map'
        apertureMap.tofile('%s/parallel/disk.aperture.dat' % (self.mcdir))
        apertureMapStd.tofile('%s/parallel/disk.apertureStd.dat' % (self.mcdir))
        peakDensityAp.tofile('%s/parallel/disk.peakDensityAp.dat' % (self.mcdir))
        peakIncliAp.tofile('%s/parallel/disk.peakIncliAp.dat' % (self.mcdir))
        peakOmegaAp.tofile('%s/parallel/disk.peakOmegaAp.dat' % (self.mcdir))

        return (neighborMap, neighborMapStd, peakDensity, peakIncli, peakOmega)


class DiskPlane(object):
    def __init__(self, root, mcdir, makeplot=False, nside=64,
                 names=None):
        self.nside = nside

        self.npix = healpy.nside2npix(self.nside)
        #self.npix = healpix.nside2npix(self.nside)
        self.mcdir = mcdir
        
        if (names == None):
            yng = young.loadYoungStars(root)

            names = yng.getArray('name')
            names.sort()

        nstars = len(names)

        iAll = None
        oAll = None
        for ss in range(nstars):
            name = names[ss]
            
            # Read in the 6D Probability distribution of orbital
            # parameters for this star.
            _f = open('%s%s.mc.dat' % (mcdir, name), 'r')
            mc = pickle.load(_f)
            _f.close()
            print 'Adding %15s (%d) to disk' % (name, len(mc.i))

            # Store the monte carlo results as (i, o) pairs.
            if (iAll == None):
                pdftrials = len(mc.i)
                iAll = zeros((nstars, pdftrials), dtype=float)
                oAll = zeros((nstars, pdftrials), dtype=float)
                     
            # This assumes that the Mass/r0/x0/y0 values are the
            # same for a single trail across all stars.
            iAll[ss,:] = mc.i
            oAll[ss,:] = mc.o
            
        ##########
        # Map out the PDF for the density of normal vectors
        ##########
        npix = healpy.nside2npix(nside)
        #npix = healpix.nside2npix(nside)
        trials = 100000

        # Compute the PDF for the density at each pixel along
        # with the weighted average density at each pixel.
        neighborMap = zeros((4, npix), dtype=float)

        # Get the (i,o) values for each pixel in the sky
        # Switch into radians
        iAll *= math.pi / 180.0
        oAll *= math.pi / 180.0

        # Get the coordinates of each pixel in the HEALpix map
        pixIdx = arange(npix, dtype=int)
        (ipix, opix) = healpy.pix2ang(nside, pixIdx)
        #(ipix, opix) = healpix.pix2ang_ring(nside, pixIdx)
        sinip = sin(ipix)
        cosip = cos(ipix)

        siniAll = sin(iAll)
        cosiAll = cos(iAll)

        # Some useful stuff for future calculations
        onesNpix = ones(npix, dtype=float)
        onesNstars = ones(nstars, dtype=float)
        factor = 2.0 * math.pi * (180.0 / math.pi)**2

        if (trials > pdftrials):
            print 'Cannot run more trials than are available from the PDF.'
        
        print 'Running MC to obtain density map.'
        for ii in range(trials):
            if ((ii % 100) == 0):
                print 'Trial %d' % ii, time.ctime(time.time())
            
            # Randomly select an (i,o) pair out of each star's
            # marginalized PDF. Pairs need to match because we need
            # to compare the same mass/r0/x0/y0 for all stars.
            incl = iAll[:, ii]
            omeg = oAll[:, ii]
            sini = siniAll[:, ii]
            cosi = cosiAll[:, ii]

            idx = (where((incl == float('nan')) |
                         (omeg == float('nan'))))[0]
            if (len(idx) > 0):
                print ii, idx

            # Find densities
            #omegSq = outerproduct(omeg, onesNpix)
            #opixSq = outerproduct(onesNstars, opix)
            omegSq = outer(omeg, onesNpix)
            opixSq = outer(onesNstars, opix)
            cosodiff = cos(opixSq - omegSq)

            #sinSq = outerproduct(sini, sinip)
            #cosSq = outerproduct(cosi, cosip)
            sinSq = outer(sini, sinip)
            cosSq = outer(cosi, cosip)

            # Angular offset from each pixel for all stars (radians)
            angOff = arccos( (sinSq * cosodiff) + cosSq )
            angOff.sort(axis=0)

            # Density per steradian from nearest neighbor
            # Solid angle is 2 * pi * (1 - cos theta)
            neighborMap[0,:] += 3.0 / (factor*(1.0 - cos(angOff[3,:])))
            neighborMap[1,:] += 4.0 / (factor*(1.0 - cos(angOff[4,:])))
            neighborMap[2,:] += 5.0 / (factor*(1.0 - cos(angOff[5,:])))
            neighborMap[3,:] += 6.0 / (factor*(1.0 - cos(angOff[6,:])))

        neighborMap /= trials
        neighborMap.tofile('%sdisk.neighbor.dat' % (mcdir))


class IsotropicMC(object):
    def __init__(self, ntrials, pdftrials):
        # Setup data we are sampling from
        
        root = '/u/jlu/work/gc/proper_motion/align/08_03_26/'
        yng = young.loadYoungStars(root)

        yngNames = yng.getArray('name')
        yngNames.sort()

        self.ntrials = ntrials
        self.pdftrials = pdftrials
        self.numStars = len(yngNames)

        # Random number generator
        self.g = random.Random()

        # We want to keep the magnitudes of the
        # 2D radii and the 3D velocities fixed.
        # We will randomly select the directions 
        # of the position and velocity vectors. 

        # Load up all the stars information:
        self.stars = []
        for ii in range(len(yngNames)):
            star = yng.stars[ii]
            star.rmag = sqrt(star.x**2 + star.y**2)
            star.vmag = sqrt(star.vx**2 + star.vy**2 + star.vz**2)

            self.stars.append( star )


    def runmc(self, savefile, verbose=False, zfrom='all'):
        if (verbose):
            print 'runmc START: ', time.ctime(time.time())

        cc = objects.Constants()
        
        # The resolution of our PDF sky maps
        nside = 64
        npix = healpy.nside2npix(nside)
        #npix = healpix.nside2npix(nside)

        # Maps contain the probability distribution function
        # This is a HEALPix map for each trial.
        healmap = zeros((self.ntrials, npix), dtype=float)

        alltrials = []
        
        for ii in range(self.ntrials):
            print 'ISO Trial %d: ' % (ii), time.ctime(time.time())

            thistrial = []
            
            for ss in range(self.numStars):
                star = self.stars[ss]
                fitxv = star.getFitXv()
                fityv = star.getFitYv()

                # Randomly sample the direction of the position vector
                rmax = star.rmag
                y = self.g.uniform(-rmax, rmax)
                x = sqrt(rmax**2 - y**2)
                x *= self.g.choice([-1.0,1.0])  # positive/negative

                xerr = fitxv.perr
                yerr = fityv.perr
                
                # Randomly sample the direction of the velocitiy vector
                vxmax = star.vmag
                vx = self.g.uniform(-vxmax, vxmax)

                vymax = sqrt(star.vmag**2 - vx**2)
                vy = self.g.uniform(-vymax, vymax)

                vz = sqrt(star.vmag**2 - vx**2 - vy**2)
                vz *= self.g.choice([-1.0, 1.0]) # positive/negative
                #print '%-13s  %6.1f  %6.1f  %6.1f' % (star.name, vx, vy, vz)

                # convert proper motions back to arcsec
                vx /= cc.asy_to_kms
                vy /= cc.asy_to_kms
                
                vxerr = fitxv.verr # arcsec/yr
                vyerr = fityv.verr # arcsec/yr
                vzerr = star.vzerr  # km/s
                
                #print '   x  = %6.1f +/- %4.1f' % (x, xerr)
                #print '   y  = %6.1f +/- %4.1f' % (y, yerr)
                #print '   vx = %6.4f +/- %6.4f' % (vx, vxerr)
                #print '   vy = %6.4f +/- %6.4f' % (vy, vyerr)
                #print '   vy = %6.1f +/- %6.1f' % (vz, vzerr)

                star.setFitpXa(fitxv.t0, x, xerr, vx, vxerr, 0.0, 0.0)
                star.setFitpYa(fityv.t0, y, yerr, vy, vyerr, 0.0, 0.0)
                star.vz = vz
                star.vzerr = vzerr

                # Perform a MC to get the PDF for this star
                mc = StarOrbitsMC(star, ntrials=self.pdftrials)
                mc.run(verbose=False, zfrom=zfrom)

                pdf = mc.makePdfHealpix(nside=nside)

                healmap[ii] += pdf / pdf.max()

                thistrial.append(pdf)

            alltrials.append(thistrial)
                
        # Save the whole array of MC objects to a pickle file.
        #pickle.dump(alltrials, open(savefile + '_mc_all.pick', 'w'), protocol=1)

        # Save the results of the MC
        _info = open(savefile + '_info.txt', 'w')
        _info.write('NumberOfStars       %5d\n' % (self.numStars))
        _info.write('NumberOfPdfTrials   %5d\n' % (self.pdftrials))
        _info.write('NumberOfSims        %5d\n' % (self.ntrials))
        _info.close()

        # Save for each trial, the sum of the PDFs.
        healmap.tofile(savefile + '_maps.dat')

        if (verbose):
            print 'runmc FINISH: ', time.ctime(time.time())


def diskWidth2d(mapfile, n, nside=64):
    npix = healpy.nside2npix(nside)
    #npix = healpix.nside2npix(nside)
    neighborMap = fromfile(mapfile, dtype=float, shape=(4, npix))

    # This is a 1D HEALpix map of the sky
    planeMap = neighborMap[n]

    # Convert into a 2D sky map
    pixIdx = arange(len(planeMap))

    (i, o) = healpy.pix2ang(nside, pixIdx)
    #(i, o) = healpix.pix2ang_ring(nside, pixIdx)
    i *= 180.0 / math.pi
    o *= 180.0 / math.pi

    # Convert into regularly spaced grid
    print 'Makeing mesh'
    ii, oo = meshgrid(arange(0, 180, stride=2, dtype=float),
                      arange(0, 360, stride=2, dtype=float))
    print 'Gridding data'
    vals = griddata(i, o, planeMap, ii, oo)


    ##########
    # Fit a 2D gaussian
    ##########
    def fitfunc(p, fjac=None, incl=None, omeg=None, data=None):
        # Constant + 2D gaussian
        gg = bivariate_normal(incl, omeg, mux=p[2], muy=p[3],
                              sigmax=p[4], sigmay=p[5], sigmaxy=p[6])
        model = p[0] + (p[1] * gg)

        devs = (data - model) / sqrt(data)
        status = 0

        return [status, devs.flat]

    def printParams(p, msg):
        print msg
        print '  Inclination    = %6.2f +/- %6.2f' % (p[2], p[4])
        print '  PA to Asc Node = %6.2f +/- %6.2f' % (p[3], p[5])
        print '  Correlation    = %4.2f' % (p[6])
        print '  Amplitude      = %f' % (p[1])
        print '  Constant       = %f' % (p[0])

    # Fit hasa 7 free parameters.
    print 'Fitting 2D gaussian'
    
    # Initial guess for each:
    p0 = zeros(7, dtype=float)
    p0[0] = 8.4e-4                      # constant
    p0[1] = 100.0                       # gaussian amplitude
    p0[2] = ii.flat[vals.flat.argmax()] # gaussian origin in incl
    p0[3] = oo.flat[vals.flat.argmax()] # gaussian origin in omeg
    p0[4] = 20.0                        # gaussian width in incl
    p0[5] = 20.0                        # gaussian width in omeg
    p0[6] = 0.1                         # gaussian correlation in incl/omeg

    printParams(p0, 'Initial Guess')
    
    # Setup properties of each free parameter.
    parinfo = {'relStep':0.1, 'step':0.1, 'fixed':0, 'limits':[1e-7,360],
	      'limited':[1,1], 'mpside':1}
    pinfo = [parinfo.copy() for i in range(7)]

    pinfo[0]['limits'] = [1e-4, 1e-3]
    pinfo[1]['limits'] = [0.01, 1e5]
    pinfo[2]['limits'] = [90, 130]
    pinfo[3]['limits'] = [90, 130]
    pinfo[4]['limits'] = [6, 30]
    pinfo[5]['limits'] = [6, 30]
    pinfo[6]['limits'] = [0, 0.6]

    # Stuff to pass into the fit function
    functargs = {'incl': ii, 'omeg': oo, 'data': vals}

    m = mpfit.mpfit(fitfunc, p0, functkw=functargs, parinfo=pinfo,
                    quiet=0)
    if (m.status <= 0): 
	print 'error message = ', m.errmsg

    p = m.params
    printParams(p, 'Final Solution')
    
    gg = bivariate_normal(ii, oo, mux=p[2], muy=p[3],
                          sigmax=p[4], sigmay=p[5], sigmaxy=p[6])
    model = p[0] + (p[1] * gg)
    devs = (vals - model)

    print 'Plotting'
    figure(2, figsize=(14,4))
    subplot(1, 3, 1)
    contourf(ii, oo, vals, 50, cmap=cm.jet)
    colorbar()
    subplot(1, 3, 2)
    contourf(ii, oo, model, 50, cmap=cm.jet)
    colorbar()
    subplot(1, 3, 3)
    contourf(ii, oo, devs, 50, cmap=cm.jet)
    colorbar()
    show()

def diskWidthCDF(mapfile, n, nside=64):
    """
    Calculate the width of the disk from the cumulative distribution
    function and looking for the 68% contour level. Also use the
    half-max level as a double-check.

    mapfile -- A stack of 4 i/Omega maps (e.g. aorb100000/disk.neighbor.dat)
               produced from aorb.Disk.calc
    n -- Which of the maps to use from the stack (typically n=2 for
         the 5th nearest neighbor analysis.
    """
    npix = healpy.nside2npix(nside)
    #npix = healpix.nside2npix(nside)

    neighborMap = fromfile(mapfile, dtype=float)
    neighborMap = neighborMap.reshape((4, npix))

    # This is a 1D HEALpix map of the sky
    planeMap = neighborMap[n]

    # Convert into a 2D sky map
    pixIdx = arange(len(planeMap))

    (i, o) = healpy.pix2ang(nside, pixIdx)
    #(i, o) = healpix.pix2ang_ring(nside, pixIdx)
    i *= 180.0 / math.pi
    o *= 180.0 / math.pi

    # Sort pixel values
    sid0 = planeMap.argsort()
    sid = sid0[::-1]

    # Determine the half-max level
    pixSort = planeMap[sid]
    print 'Peak at i = %6.2f, o = %6.2f' % (i[sid[0]], o[sid[0]])
    print 'Total Probability = ', pixSort.sum()

    idHalf = (where(pixSort > pixSort[0] / 2.0))[0]
    iHalf = (i[sid])[idHalf]
    oHalf = (o[sid])[idHalf]
    print 'Half-Max Level:  i = [%6.2f - %6.2f]  O = [%6.2f - %6.2f]' % \
          (iHalf.min(), iHalf.max(), oHalf.min(), oHalf.max())
    print 'Half-Max Solid Angle: %6.2f steradians' % \
	(len(idHalf) * 4.0 * math.pi / npix)

    # Determine the 68% contours assuming a background of 0.00084
    # pixSort = planeMap[sid] - 0.00084

    # Only keep positive pixels
    pos = where(pixSort > 0)
    cdf = cumsum(pixSort[pos]) / pixSort[pos].sum()

    # Determine point at which we reach 68% level
    id68 = (where(cdf < 0.68))[0]

    # Now fetch incl and omeg for all these pixels
    i68 = ((i[sid])[pos])[id68]
    o68 = ((o[sid])[pos])[id68]
    print 'CDF 0.68 Level:  i = [%6.2f - %6.2f]  O = [%6.2f - %6.2f]' % \
          (i68.min(), i68.max(), o68.min(), o68.max())
    plot(cdf)
    show()


##################################################
#
#  Helper Functions
#
##################################################
def create_generators(num, delta, firstseed=None):
    """Return list of num distinct generators.
    Each generator has its own unique segment of delta elements
    from Random.random()'s full period.
    Seed the first generator with optional arg firstseed (default
    is None, to seed from current time).
    """

    g = random.Random(firstseed)
    result = [g]
    for i in range(num - 1):
        laststate = g.getstate()
        g = random.Random()
        g.setstate(laststate)
        g.jumpahead(delta)
        result.append(g)
    return result

    
def findRanges(oo, varName):
    exec('oo.%s_lo_p = oo.%s[oo.idxp].min()' % (varName, varName))
    exec('oo.%s_hi_p = oo.%s[oo.idxp].max()' % (varName, varName))
    exec('oo.%s_lo_n = oo.%s[oo.idxn].min()' % (varName, varName))
    exec('oo.%s_hi_n = oo.%s[oo.idxn].max()' % (varName, varName))
    exec('oo.%s_lo = oo.%s[oo.idx].min()' % (varName, varName))
    exec('oo.%s_hi = oo.%s[oo.idx].max()' % (varName, varName))

def plane2z(x, y, idisk=109.47, odisk=105.47):
    """
    Calculate the line of site distance for a star in the orbital
    plane with inclination <idisk> and PA to ascending node <odisk>
    using only the stars projected plane-of-the-sky positions.

    Input Parameters:
    x, y -- Positions in the plane of the sky for a star.

    Optional Input Parameters:
    idisk -- inclination of the planes normal vector (deg)
    odisk -- PA to asc. node of the planes normal vector (deg)

    Return:
    z -- Line of sight distance to the star (in the same units as x and y).
    """
    idisk = math.radians(idisk)
    odisk = math.radians(odisk)

    n = array([ sin(idisk) * cos(odisk),
               -sin(idisk) * sin(odisk),
               -cos(idisk)], dtype=float)

    z = -(n[0] * x + n[1] * y) / n[2]
    
    return z


def acc2z(x, y, ar, dist, mass):
    """
    Change acceleration (in the plane of the sky) from mas/yr^2 to
    line of sight distance in arcsec.

    Input:
    x - x position in arcsec
    y - y position in arcsec
    ar - plane of the sky acc in mas/yr^2
    dist - Ro in pc
    mass - in solar masses
    """
    cc = objects.Constants()
    GM = mass * cc.msun * cc.G

    # Convert acceleration into CGS
    arcgs = ar * dist * cc.cm_in_au / (cc.sec_in_yr**2 * 1000.0)

    # Convert into z-distance
    r = sqrt(x**2 + y**2)             # arcsec
    rcgs = r * dist * cc.cm_in_au
    tmp1 = (GM * rcgs / -arcgs)**(2.0/3.0)
    zcgs = sqrt(tmp1 - rcgs**2)
    z = zcgs / (cc.cm_in_au * dist)

    return z

def acc2zErr(x, y, ar, arerr, dist, mass):
    """
    Change acceleration (in the plane of the sky) from mas/yr^2 to
    line of sight distance in arcsec.

    Input:
    x - x position in arcsec
    y - y position in arcsec
    ar - plane of the sky acc in mas/yr^2
    arerr - error in ar
    dist - Ro in pc
    mass - in solar masses
    """
    cc = objects.Constants()
    GM = mass * cc.msun * cc.G

    # Convert acceleration into CGS
    arcgs = ar * dist * cc.cm_in_au / (cc.sec_in_yr**2 * 1000.0)
    arerrcgs = arerr * dist * cc.cm_in_au / (cc.sec_in_yr**2 * 1000.0)

    # Convert into z-distance
    r = sqrt(x**2 + y**2)             # arcsec
    rcgs = r * dist * cc.cm_in_au
    tmp1 = (GM * rcgs / -arcgs)**(2.0/3.0)
    zcgs = sqrt(tmp1 - rcgs**2)
    z = zcgs / (cc.cm_in_au * dist)

    zerrcgs = abs(arerrcgs * tmp1 / (3.0 * zcgs * arcgs))
    zerr = zerrcgs / (cc.cm_in_au * dist)

    return (z, zerr)

def z2acc(x, y, z, dist, mass):
    """
    Change line of sight distance in arcsec to
    acceleration (in the plane of the sky) in mas/yr^2.

    Input:
    x - x position in arcsec
    y - y position in arcsec
    z - z position in arcsec
    dist - Ro in pc
    mass - in solar masses
    """
    cc = objects.Constants()
    GM = mass * cc.msun * cc.G

    # Convert distance into CGS
    r = sqrt(x**2 + y**2)             # arcsec
    zcgs = z * dist * cc.cm_in_au
    rcgs = r * dist * cc.cm_in_au

    arcgs = -GM * rcgs / (rcgs**2 + zcgs**2)**(3.0/2.0)
    ar = arcgs * cc.sec_in_yr**2 * 1000.0 / (dist * cc.cm_in_au)

    return ar
    
def merge(ob1, ob2):
    """
    Merge two starset objects. Useful for merging the objects from
    the central 10 arcsec analysis with the deep mosaic analysis.
    """

    names = ob1.getArray('name')

    # Loop through the mosaic stars
    for ii in range(len(ob2.stars)):
        # If this mosaic star is already in central 10 asec, don't include it!
        if ob2.stars[ii].name in names:
            continue
        else:
            ob1.stars.append(ob2.stars[ii])

    return ob1

