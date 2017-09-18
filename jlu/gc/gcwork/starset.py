import os, sys
from astropy.table import Table # in place of asciidata
import math
import numpy as np
from . import objects
from . import util
from . import starTables
import time
import pdb

class StarSet(object):
    """Object containing align output.

    s = StarSet('align_d')
    print s.stars[0].name

    Properties include:
    stars - list of Star objects
    absolute - whether to rotate into North up coordinates
    verbose - Whether to print info
    relErr - whether to include absolute astrometric errors
    t - Transform object
    """

    def __init__(self, root, verbose=0, relErr=0, trans=None):
        self.root = root
        self.verbose = verbose
        self.relErr = relErr

        # Set up the default position of Sgr A* from align output.
        # Uses the *.sgra and *.scale files.
        if (trans == None):
            self.t = objects.Transform()
            self.t.loadFromAlign(root)
        else:
            self.t = trans

        # Load up tables
        _date = open(root + '.date', 'r')
        f_pos = open(root + '.pos', 'r')
        f_err = open(root + '.err', 'r')
        f_mag = open(root + '.mag', 'r')
        f_vel = open(root + '.vel', 'r')
        f_par = open(root + '.param', 'r')
        f_name = open(root + '.name', 'r')
        f_opos = open(root + '.origpos', 'r')
        f_b0 = open(root + '.b0', 'r')

        dates = _date.readline().split()
        numEpochs = len(dates)

        # Make an infinity float to compare against
        Inf = float('inf')

        # Read epochs
        years = [float(dates[i]) for i in range(numEpochs)]
        dates = ['%4d' % math.floor(year) for year in years]

        objects.Star.years = years

        self.years = np.array(years)
        self.dates = np.array(dates)

        self.stars = []
        for line in f_pos:
            _pos = line.split()
            _err = f_err.readline().split()
            _mag = f_mag.readline().split()
            _vel = f_vel.readline().split()
            _par = f_par.readline().split()
            _opos = f_opos.readline().split()
            _b0 = f_b0.readline().split()
            _name = f_name.readline().split()

            # Initialize a new star with the proper name. Append to the set.
            star = objects.Star(_vel[0])
            self.stars.append(star)

            # Parse velocity file
            star.velCnt = int(_vel[1])
            star.mag    = float(_mag[1])
            star.magerr = float(_mag[2])
            star.x      = float(_vel[3])
            star.y      = float(_vel[4])
            star.vx     = float(_vel[5])
            star.vxerr  = float(_vel[6])
            star.vy     = float(_vel[7])
            star.vyerr  = float(_vel[8])
            star.v2d    = float(_vel[9])
            star.v2derr = float(_vel[10])

            # Do conversions for these values
            (star.x, star.y) = \
                util.rPix2Arc(star.x, star.y, self.t, relErr=self.relErr)
            (star.vx, star.vy, star.vxerr, star.vyerr) = \
                util.verrPix2Arc(star.vx, star.vy, star.vxerr, star.vyerr,
                                 self.t, relErr=self.relErr)
            (star.v2d, star.v2derr) = \
                util.errPix2Arc(star.v2d, star.v2derr, self.t,
                                relErr=self.relErr)


            # Parse b0 file
            t0 = float(_b0[1])
            x0 = float(_b0[2])
            x0err = float(_b0[3])
            vx = float(_b0[4])
            vxerr = float(_b0[5])
            y0 = float(_b0[8])
            y0err = float(_b0[9])
            vy = float(_b0[10])
            vyerr = float(_b0[11])

            star.setFitpXalign(t0, x0, x0err, vx, vxerr)
            star.setFitpYalign(t0, y0, y0err, vy, vyerr)

            (x0, y0, x0err, y0err) = util.rerrPix2Arc(x0, y0, x0err, y0err,
                                                      self.t, relErr=self.relErr)
            (vx, vy, vxerr, vyerr) = util.verrPix2Arc(vx, vy, vxerr, vyerr,
                                                      self.t, relErr=self.relErr)
            star.setFitXalign(t0, x0, x0err, vx, vxerr)
            star.setFitYalign(t0, y0, y0err, vy, vyerr)

            # Parse err, mag, pos files for all epochs
            for ei in range(numEpochs):
                ep = star.e[ei]
                ep.name = _name[ei+1]
                ep.xpix = float(_pos[(ei*2)+1])
                ep.ypix = float(_pos[(ei*2)+2])

                ep.xorig = float(_opos[(ei*2)+1])
                ep.yorig = float(_opos[(ei*2)+2])

                if (len(_err) > (numEpochs*2 + 1)):
                    ep.xpixerr_p = float(_err[(ei*4)+1])
                    ep.ypixerr_p = float(_err[(ei*4)+2])
                    ep.xpixerr_a = float(_err[(ei*4)+3])
                    ep.ypixerr_a = float(_err[(ei*4)+4])

                    if (ep.xpixerr_p < 0): ep.xpixerr_p = 0.0
                    if (ep.ypixerr_p < 0): ep.ypixerr_p = 0.0
                else:
                    ep.xpixerr_p = 0.0
                    ep.ypixerr_p = 0.0
                    ep.xpixerr_a = float(_err[(ei*2)+1])
                    ep.ypixerr_a = float(_err[(ei*2)+2])

                    # Check for the infinity case
                    if (ep.xpixerr_a == Inf):
                        ep.xpixerr_a = 0.0
                        ep.ypixerr_a = 0.0

                ep.mag     = float(_mag[ei+4])
                ep.snr     = float(_par[(ei*4)+1])
                ep.corr    = float(_par[(ei*4)+2])
                ep.nframes = float(_par[(ei*4)+3])
                ep.fwhm    = float(_par[(ei*4)+4])

                # Convert stuff into arcseconds
                (ep.x, ep.y, ep.xerr_p, ep.yerr_p) = \
                    util.rerrPix2Arc(ep.xpix, ep.ypix, ep.xpixerr_p, ep.ypixerr_p,
                                     self.t, relErr=self.relErr)

                (ep.x, ep.y, ep.xerr_a, ep.yerr_a) = \
                    util.rerrPix2Arc(ep.xpix, ep.ypix, ep.xpixerr_a, ep.ypixerr_a,
                                     self.t, relErr=self.relErr)

    def loadList(self, suffix='.list'):
        _list = open(self.root + suffix, 'r')

        fileName = []
        dataType = []
        isRef = []

        for line in _list:
            parts = line.split()

            fileName.append(parts[0])
            dataType.append(int(parts[1]))

            if len(parts) > 2 and parts[-1] == 'ref':
                isRef.append(True)
            else:
                isRef.append(False)

        # Convert to numpy arrays
        fileName = np.array(fileName)
        dataType = np.array(dataType)
        isRef = np.array(isRef)

        # Check that we have a reference epoch, otherwise, it is the
        # first one.
        idx = np.where(isRef == True)[0]
        if len(idx) == 0:
            isRef[0] = True

        self.starlist = fileName
        self.datatype = dataType
        self.starlist_is_ref = isRef


    def loadStarsUsed(self):
        _used = open(self.root + '.starsUsed', 'r')

        names = self.getArray('name')

        # Make a "isUsed" variable on every star in every epoch.
        # Start with value = False.
        for star in self.stars:
            for ep in star.e:
                ep.isUsed = False

        for line in _used:
            parts = line.split()

            if len(parts) != len(self.years):
                print('Error reading starsUsed file.')
                return

            # This is a loop through the epochs.
            for ee in range(len(parts)):
                if parts[ee] == '---':
                    continue

                try:
                    ss = names.index(parts[ee])
                    self.stars[ss].e[ee].isUsed = True
                except ValueError:
                    pass

        return

    def loadPoints(self, pointsDir):
        """
        Loads .points and .phot files. The following variables are
        created for each star:

        star.pointsCnt - the number of rows in the points file
        star.e[ee].pnt_x
        star.e[ee].pnt_y
        star.e[ee].pnt_xe
        star.e[ee].pnt_ye

        star.e[ee].phot_r
        star.e[ee].phot_x
        star.e[ee].phot_y
        star.e[ee].phot_xe
        star.e[ee].phot_ye
        star.e[ee].phot_mag
        star.e[ee].phot_mage

        When there is no data for a certain epoch in the points or
        phot file, the values are set to -1000. Remember that the
        pointsCnt and velCnt may not be equal and a star's information
        in align may not be the same as in the points file as we
        have a number of codes that modify (or trim out) data from the
        points files.
        """
        self.pointsDir = pointsDir

        for ss in range(len(self.stars)):
            star = self.stars[ss]

            # Number of Epochs Detected should be corrected
            # for epochs trimmed out of the *.points files.
            pntsFile = '%s%s.points' % (pointsDir, star.name)
            _pnts = Table.read(pntsFile)

            photFile = '%s%s.phot' % (pointsDir, star.name)
            _phot = Table.read(photFile)

            star.pointsCnt = _pnts.nrows
            if star.pointsCnt == 0:
                pntDate = np.array([])
                pntX = np.array([])
                pntY = np.array([])
                pntXe = np.array([])
                pntYe = np.array([])
                photDate = np.array([])
                photR = np.array([])
                photX = np.array([])
                photY = np.array([])
                photXe = np.array([])
                photYe = np.array([])
                photM = np.array([])
                photMe = np.array([])

            else:
                pntDate = _pnts[0].tonumpy()
                pntX = _pnts[1].tonumpy()
                pntY = _pnts[2].tonumpy()
                pntXe = _pnts[3].tonumpy()
                pntYe = _pnts[4].tonumpy()

                photDate = _phot[0].tonumpy()
                photR = _phot[1].tonumpy()
                photX = _phot[2].tonumpy()
                photY = _phot[3].tonumpy()
                photXe = _phot[4].tonumpy()
                photYe = _phot[5].tonumpy()
                photM = _phot[6].tonumpy()
                photMe = _phot[7].tonumpy()


            # Load up data from the points files.
            for ee in range(len(star.years)):
                ttPnts = (np.where(abs(pntDate - star.years[ee]) < 0.001))[0]
                ttPhot = (np.where(abs(photDate - star.years[ee]) < 0.001))[0]


                if (len(ttPnts) == 0):
                    star.e[ee].pnt_x = -1000.0
                    star.e[ee].pnt_y = -1000.0
                    star.e[ee].pnt_xe = -1000.0
                    star.e[ee].pnt_ye = -1000.0
                else:
                    ttPnts = ttPnts[0]
                    star.e[ee].pnt_x = pntX[ttPnts]
                    star.e[ee].pnt_y = pntY[ttPnts]
                    star.e[ee].pnt_xe = pntXe[ttPnts]
                    star.e[ee].pnt_ye = pntYe[ttPnts]

                if (len(ttPhot) == 0):
                    star.e[ee].phot_r = -1000.0
                    star.e[ee].phot_x = -1000.0
                    star.e[ee].phot_y = -1000.0
                    star.e[ee].phot_xe = -1000.0
                    star.e[ee].phot_ye = -1000.0
                    star.e[ee].phot_mag = -1000.0
                    star.e[ee].phot_mage = -1000.0
                else:
                    ttPhot = ttPhot[0]
                    star.e[ee].phot_r = photR[ttPhot]
                    star.e[ee].phot_x = photX[ttPhot]
                    star.e[ee].phot_y = photY[ttPhot]
                    star.e[ee].phot_xe = photXe[ttPhot]
                    star.e[ee].phot_ye = photYe[ttPhot]
                    star.e[ee].phot_mag = photM[ttPhot]
                    star.e[ee].phot_mage = photMe[ttPhot]


    def loadPolyfit(self, fitRoot, accel=0, arcsec=0, trimUnfound=True, silent=False):
        fitFile = fitRoot + '.linearFormal'
        t0File = fitRoot + '.lt0'

        if (accel == 1):
            fitFile = fitRoot + '.accelFormal'
            t0File = fitRoot + '.t0'

        f_fit = open(fitFile, 'r')
        f_t0 = open(t0File, 'r')

        # Get the align names array, since we will index out of it a lot.
        names = np.array([star.name for star in self.stars])

        alignIdx = 0
        trimIdx = []

        for line in f_fit:
            _fit = line.split()
            _t0 = f_t0.readline().split()

            # First line should be a header line with "#" as the first letter
            if _fit[0].startswith("#"):
                # Determine the number of coefficients in this file.
                # Columns are 1 starname, 2*2 for chiSq and Q (x and y),
                # and 2*2*(N) for x and y coefficients and their
                # associated errors for each order of the polynomial
                # where N is the number of coefficients.
                numCols = len(_fit)
                numCoeffs = (numCols - 5) / 4
                continue

            # Assume that the stars loaded from align are in the same
            # order as the polyfit output.
            if names[alignIdx] != _fit[0]:
                if silent == False:
                    print(('%-13s: Mis-match in align and polyfit...' % _fit[0]))

                while names[alignIdx] != _fit[0]:
                    if trimUnfound:
                        msg = '\t trimming '
                        trimIdx.append(alignIdx)
                    else:
                        msg = '\t skipping '

                    if silent == False:
                        print(('%s %s' % (msg, names[alignIdx])))
                    alignIdx += 1


            star = self.stars[alignIdx]
            alignIdx += 1

            t0x = float(_t0[1])
            t0y = float(_t0[2])

            if (numCoeffs >= 1):
                x0 = float(_fit[1])
                x0err = float(_fit[ 1 + numCoeffs ])
                y0 = float(_fit[ (2*numCoeffs) + 3 ])
                y0err = float(_fit[ (3*numCoeffs) + 3 ])

            if (numCoeffs >= 2):
                vx = float(_fit[2])
                vxerr = float(_fit[ 2 + numCoeffs ])
                vy = float(_fit[ (2*numCoeffs) + 4 ])
                vyerr = float(_fit[ (3*numCoeffs) + 4 ])

            if (numCoeffs >= 3):
                ax = float(_fit[ 3 ])
                axerr = float(_fit[ 3 + numCoeffs ])
                ay = float(_fit[ (2*numCoeffs) + 5 ])
                ayerr = float(_fit[ (3*numCoeffs) + 5 ])

            if (accel == 1):
                # Save the polyfit (in pixels)
                star.setFitpXa(t0x, x0, x0err, vx, vxerr, ax, axerr)
                star.setFitpYa(t0y, y0, y0err, vy, vyerr, ay, ayerr)
                fitpx = star.fitpXa
                fitpy = star.fitpYa
                # Save the polyfit (in arcsec)
                star.setFitXa(t0x, x0, x0err, vx, vxerr, ax, axerr)
                star.setFitYa(t0y, y0, y0err, vy, vyerr, ay, ayerr)
                fitx = star.fitXa
                fity = star.fitYa
            else:
                # Save the polyfit (in pixels)
                star.setFitpXv(t0x, x0, x0err, vx, vxerr)
                star.setFitpYv(t0y, y0, y0err, vy, vyerr)
                fitpx = star.fitpXv
                fitpy = star.fitpYv
                # Save the polyfit (in arcsec)
                star.setFitXv(t0x, x0, x0err, vx, vxerr)
                star.setFitYv(t0y, y0, y0err, vy, vyerr)
                fitx = star.fitXv
                fity = star.fitYv

            fitx.chi2 = float(_fit[(2*numCoeffs) + 1])
            fitx.q = float(_fit[(2*numCoeffs) + 2])
            fitx.dof = star.velCnt - numCoeffs
            fitx.chi2red = fitx.chi2 / fitx.dof

            fity.chi2 = float(_fit[(4*numCoeffs) + 3])
            fity.q = float(_fit[(4*numCoeffs) + 4])
            fity.dof = star.velCnt - numCoeffs
            fity.chi2red = fity.chi2 / fity.dof

            fitpx.chi2 = float(_fit[(2*numCoeffs) + 1])
            fitpx.q = float(_fit[(2*numCoeffs) + 2])
            fitpx.dof = star.velCnt - numCoeffs
            fitpx.chi2red = fitpx.chi2 / fitpx.dof

            fitpy.chi2 = float(_fit[(4*numCoeffs) + 3])
            fitpy.q = float(_fit[(4*numCoeffs) + 4])
            fitpy.dof = star.velCnt - numCoeffs
            fitpy.chi2red = fitpy.chi2 / fitpy.dof

            if (arcsec):
                (fitx.p, fity.p, fitx.perr, fity.perr) = \
                         util.rerrPix2Arc(fitx.p, fity.p, fitx.perr, fity.perr,
                      self.t, relErr=self.relErr)


                (fitx.v, fity.v, fitx.verr, fity.verr) = \
                         util.verrPix2Arc(fitx.v, fity.v, fitx.verr, fity.verr,
                      self.t, relErr=self.relErr)

                if (accel == 1):
                    (fitx.a, fity.a, fitx.aerr, fity.aerr) = \
                             util.aerrPix2Arc(fitx.a, fity.a, fitx.aerr,
                          fity.aerr,
                          self.t, relErr = self.relErr)

        f_fit.close()
        f_t0.close()

        if trimUnfound and len(trimIdx) > 0:
            # Trim stars down to just those with polyfit information
            newStars = []
            for ss in range(len(self.stars)):
                if not ss in trimIdx:
                    newStars.append(self.stars[ss])

            self.stars = newStars


    def loadEfitResults(self, efitRoot, trimStars=0, orbitsOnly=0):
        accelFile = efitRoot + '.acclim'
        orbitsFile = efitRoot + '.orbits'

        # Load up acceleration limits for the young stars
        if (orbitsOnly == 1):
            efits = objects.Efit.loadOrbits(orbitsFile)
        else:
            efits = objects.Efit.loadAcclim(accelFile)
            efits = objects.Efit.loadOrbits(orbitsFile, efitSet=efits)

        efitNames = [efit.name for efit in efits]

        names = [star.name for star in self.stars]

        # Keep list of stars with efit matches
        newStars = []
        # Keep list of star names with multiple efit matches
        dupNames = []

        # Loop through efit objects and find name matches
        for efit in efits:
            try:
                idx = names.index(efit.name)
                star = self.stars[idx]
            except ValueError as e:

                print(('Failed to find efit data for %s' % efit.name))
                continue

            # Select for the first one.
            try:
                type(star.efit)
                dupNames.append(star.name)

                # We are interested in the highest acceleration limit.
                if (efit.at_hi > star.efit.at_hi):
                    star.efit.at_hi = efit.at_hi

                # Modify orbital elements' limits to take the highest
                # and lowest
                if (efit.omega_lo < star.efit.omega_lo):
                    star.efit.omega_lo = efit.omega_lo
                if (efit.omega_hi > star.efit.omega_hi):
                    star.efit.omega_hi = efit.omega_hi
                if (efit.incl_lo < star.efit.incl_lo):
                    star.efit.incl_lo = efit.incl_lo
                if (efit.incl_hi > star.efit.incl_hi):
                    star.efit.incl_hi = efit.incl_hi
                if (efit.ecc_lo < star.efit.ecc_lo):
                    star.efit.ecc_lo = efit.ecc_lo
                if (efit.ecc_hi > star.efit.ecc_hi):
                    star.efit.ecc_hi = efit.ecc_hi
                if (efit.period_lo < star.efit.period_lo):
                    star.efit.period_lo = efit.period_lo
                if (efit.period_hi > star.efit.period_hi):
                    star.efit.period_hi = efit.period_hi
                if (efit.t0_lo < star.efit.t0_lo):
                    star.efit.t0_lo = efit.t0_lo
                if (efit.t0_hi > star.efit.t0_hi):
                    star.efit.t0_hi = efit.t0_hi
                if (efit.phase_lo < star.efit.phase_lo):
                    star.efit.phase_lo = efit.phase_lo
                if (efit.phase_hi > star.efit.phase_hi):
                    star.efit.phase_hi = efit.phase_hi
            except AttributeError as e:
                star.efit = efit

            if (trimStars == 1):
                newStars.append(star)

        # Are we keeping only those stars with efit matches?
        if (trimStars == 1):
            self.stars = newStars


    def onlyYoungDisk(self):
        """
        Trim the starlist down to only those known young stars
        that are in the Paumard star list and that we have data for,
        and that are more than 0.8'' from the SBH.
        """
        names = self.getArray('name')

        stars = []

        yng = starTables.youngStarNames()
        paum = starTables.Paumard2006()

        for name in yng:
            # Find the star in our star lists
            try:
                idx = names.index(name)
                star = self.stars[idx]

                if (star.r2d >= 0.8):
                    stars.append(star)

                    # Make sure this star is in the Paumard star list
                    idx = paum.ourName.index(name)
            except ValueError as e:
                # Couldn't find the star in our lists
                continue

        # Set the starset's star list
        self.stars = stars

        print(('Found %d young stars.' % len(stars)))


    def getArray(self, varName):
        """Turn an attribute hanging off the list of stars into
        an array.

        @param varName A string of the requested variable name (e.g. 'x')
        @type varName String

        @return objArray
        @rtype Either list or Numarray array
        """
        objNames = varName.split('.')

        objList = self.stars
        for name in objNames:
            #objList = [obj.__getattribute__(name) for obj in objList]
            # returns NaN by default if the attribute doesn't exist
            objList = [getattr(obj,name,np.nan) for obj in objList]

            if (type(objList[0]) == type('')):
                # Strings shouldn't be numpy arrays
                objArray = objList
            else:
                objArray = np.array(objList)

        return objArray


    def getArrayFromEpoch(self, epochIndex, varName):
        """Turn an attribute hanging off the list of stars' epochs
        list into an array. Only select froma specific epoch.

        @type epochIndex integer
        @param epochIndex the zero-based index for the epoch to select from.
        @type varName String
        @param varName A string of the requested variable name (e.g. 'x')

        @return objArray
        @rtype Either list or Numarray array
        """
        objNames = varName.split('.')

        objList = [star.e[epochIndex] for star in self.stars]
        for name in objNames:
            objList = [obj.__getattribute__(name) for obj in objList]

            if (type(objList[0]) == type('')):
                # Strings shouldn't be numpy arrays
                objArray = objList
            else:
                objArray = np.array(objList)

        return objArray

    def getArrayFromAllEpochs(self, varName):
        """Turn an attribute hanging off the list of stars' epochs
        list into a 2D array with the first index being the epoch
        and the second index being the star.

        Example:
        xpix = s.getArrayFromAllEpochs('xpix')
        print xpix[epochIdx, starIdx]

        @type epochIndex integer
        @param epochIndex the zero-based index for the epoch to select from.
        @type varName String
        @param varName A string of the requested variable name (e.g. 'x')

        @return objArray
        @rtype Either list or Numarray array
        """

        objList = []
        for ee in range(len(self.years)):
            objList.append( self.getArrayFromEpoch(ee, varName) )

        objArray = np.array(objList)

        return objArray
