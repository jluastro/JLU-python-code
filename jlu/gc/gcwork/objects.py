import numpy as np
import math
import os
from astropy.table import Table


class Star(object):
    """Keep properties of an individual star. Properties include:

    name = name of this star
    years = float list of epochs
    dates = string list of epochs
    velCnt = number of epochs where this star is found
    mag = the average magnitude
    r = [x, y, z]: average positions (arcsec)
    v = [vx, vy, vz]: velocities
    rerr = [xerr, yerr, zerr]: average position error (arcsec)
    verr = [vxerr, vyerr, vzerr]: velocity error (arcsec/yr)
    e = list of Epoch objects
    r2d = magnitude of [x, y]
    v2d = magniutde of [vx, vy]
    fitx = a Fit object containing X linear fit parameters.
    fity = a Fit object containing X linear fit parameters.
    t = a GCTransform object containing coordinate transformation info.
    
    """
    years = []
    def __init__(self, name):
        self.name = name
        self.e = [Epoch(year) for year in Star.years]
        
    def getr2d(self):
        return math.hypot(self.x, self.y)
    r2d = property(fget=getr2d, doc="2D projected distance from Sgr A*") 

    def getr2dErr(self):
        tmp = np.sqrt((self.x * self.xerr)**2 + (self.y * self.yerr)**2)
        tmp /= self.getr2d()
        return tmp
    r2dErr = property(fget=getr2dErr, doc="Error in 2D distance from Sgr A*") 

    def getDates(self):
        return [str(math.floor(year)) for year in self.years]
    dates = property(fget=getDates, doc="String representation of years")

    # Linear Fit Properties from align
    def setFitXalign(self, t0, x0, x0err, vx, vxerr):
        self.__fitXalign = Fit(t0, x0, x0err, vx, vxerr)
    def getFitXalign(self):
        return self.__fitXalign
    fitXalign = property(fget=getFitXalign, fset=setFitXalign,
                         doc="X Fit from align")

    def setFitYalign(self, t0, y0, y0err, vy, vyerr):
        self.__fitYalign = Fit(t0, y0, y0err, vy, vyerr)
    def getFitYalign(self):
        return self.__fitYalign
    fitYalign = property(fget=getFitYalign, fset=setFitYalign,
                         doc="Y Fit from align")

    # Linear Fit Properties from align in pixels
    def setFitpXalign(self, t0, x0, x0err, vx, vxerr):
        self.__fitpXalign = Fit(t0, x0, x0err, vx, vxerr)
    def getFitpXalign(self):
        return self.__fitpXalign
    fitpXalign = property(fget=getFitpXalign, fset=setFitpXalign,
                         doc="X Fit from align in pixels")

    def setFitpYalign(self, t0, y0, y0err, vy, vyerr):
        self.__fitpYalign = Fit(t0, y0, y0err, vy, vyerr)
    def getFitpYalign(self):
        return self.__fitpYalign
    fitpYalign = property(fget=getFitpYalign, fset=setFitpYalign,
                         doc="Y Fit from align in pixels")

    # Linear Fit Properties from polyfit
    def setFitXv(self, t0, x0, x0err, vx, vxerr):
        self.__fitXv = Fit(t0, x0, x0err, vx, vxerr)
    def getFitXv(self):
        return self.__fitXv
    fitXv = property(fget=getFitXv, fset=setFitXv,
                     doc="X linear Fit from polyfit")

    def setFitYv(self, t0, y0, y0err, vy, vyerr):
        self.__fitYv = Fit(t0, y0, y0err, vy, vyerr)
    def getFitYv(self):
        return self.__fitYv
    fitYv = property(fget=getFitYv, fset=setFitYv,
                     doc="Y linear fit from polyfit")

    # Linear Fit Properties from polyfit in pixels
    def setFitpXv(self, t0, x0, x0err, vx, vxerr):
        self.__fitpXv = Fit(t0, x0, x0err, vx, vxerr)
    def getFitpXv(self):
        return self.__fitpXv
    fitpXv = property(fget=getFitpXv, fset=setFitpXv,
                     doc="X linear Fit from polyfit in pixels")

    def setFitpYv(self, t0, y0, y0err, vy, vyerr):
        self.__fitpYv = Fit(t0, y0, y0err, vy, vyerr)
    def getFitpYv(self):
        return self.__fitpYv
    fitpYv = property(fget=getFitpYv, fset=setFitpYv,
                     doc="Y linear fit from polyfit in pixels")

    # Acceleration Fit Properties from polyfit
    def setFitXa(self, t0, x0, x0err, vx, vxerr, ax, axerr):
        self.__fitXa = AccelFit(t0, x0, x0err, vx, vxerr, ax, axerr)
    def getFitXa(self):
        return self.__fitXa
    fitXa = property(fget=getFitXa, fset=setFitXa,
                     doc="X accel fit from polyfit")

    def setFitYa(self, t0, y0, y0err, vy, vyerr, ay, ayerr):
        self.__fitYa = AccelFit(t0, y0, y0err, vy, vyerr, ay, ayerr)
    def getFitYa(self):
        return self.__fitYa
    fitYa = property(fget=getFitYa, fset=setFitYa,
                     doc="Y accel fit from polyfit")

    # Acceleration Fit Properties from polyfit in pixels
    def setFitpXa(self, t0, x0, x0err, vx, vxerr, ax, axerr):
        self.__fitpXa = AccelFit(t0, x0, x0err, vx, vxerr, ax, axerr)
    def getFitpXa(self):
        return self.__fitpXa
    fitpXa = property(fget=getFitpXa, fset=setFitpXa,
                     doc="X accel fit from polyfit in pixels")

    def setFitpYa(self, t0, y0, y0err, vy, vyerr, ay, ayerr):
        self.__fitpYa = AccelFit(t0, y0, y0err, vy, vyerr, ay, ayerr)
    def getFitpYa(self):
        return self.__fitpYa
    fitpYa = property(fget=getFitpYa, fset=setFitpYa,
                     doc="Y accel fit from polyfit in pixels")

    def setR(self, r):
        self.x = r[0]
        self.y = r[1]
        self.z = r[2]
    def getR(self):
        return np.array([self.x, self.y, self.z])
    r = property(fget=getR, fset=setR, doc="3D positional vector")

    def setV(self, v):
        self.vx = v[0]
        self.vy = v[1]
        self.vz = v[2]
    def getV(self):
        return np.array([self.vx, self.vy, self.vz])
    v = property(fget=getV, fset=setV, doc="3D velocity vector")

    def setRerr(self, rerr):
        self.xerr = rerr[0]
        self.yerr = rerr[1]
        self.zerr = rerr[2]
    def getRerr(self):
        return np.array([self.xerr, self.yerr, self.zerr])
    rerr = property(fget=getRerr, fset=setRerr, doc="3D pos. error vector")

    def setVerr(self, verr):
        nself.vxerr = verr[0]
        self.vyerr = verr[1]
        self.vzerr = verr[2]
    def getVerr(self):
        return np.array([self.vxerr, self.vyerr, self.vzerr])
    verr = property(fget=getVerr, fset=setVerr, doc="3D vel. error vector")

    def getArrayAllEpochs(self, varName):
        """Turn an attribute hanging off the list of star's epochs
        into an array. Collect the same variable from all epochs.
        
        @type varName String
        @param varName A string of the requested variable name (e.g. 'x')

        @return objArray
        @rtype Either list or Numarray array
        """
        objNames = varName.split('.')

        epochCnt = len(self.years)
        objList = [self.e[ee] for ee in range(epochCnt)]
        for name in objNames:
            objList = [obj.__getattribute__(name) for obj in objList]

        try:
            objArray = np.array(objList)
        except TypeError as e:
            objArray = objList

        return objArray

class Fit(object):
    "Linear fit."
    def __init__(self, t0, p, perr, v, verr):
        self.t0 = t0
        self.p = p
        self.v = v
        self.perr = perr
        self.verr = verr

    def getPosition(self, t):
        return self.p + (self.v * (t - self.t0))

    def getPositionError(self, t):
        dt = t - self.t0
        pos = self.p + (self.v * dt)
        err = self.perr**2 + (dt**2 * self.verr**2)
        return (pos, math.sqrt(err))

class AccelFit(Fit):
    "Acceleration fit."
    def __init__(self, t0, p, perr, v, verr, a, aerr):
        Fit.__init__(self, t0, p, perr, v, verr)
        self.a = a
        self.aerr = aerr

    def getPosition(self, t):
        dt = (t - self.t0)
        pos = self.p + (self.v * dt) + (0.5 * self.a * dt**2)
        return pos

    def getPositionError(self, t):
        dt = t - self.t0
        pos = self.p + (self.v * dt) + (0.5 * self.a * dt**2)
        err = self.perr**2 
        err += (self.verr * dt)**2
        err += (self.aerr * dt**2 * 0.5)**2
        return (pos, math.sqrt(err))


class Epoch(object):
    """Data for one epoch of one star. Includes the following:

    r = [x, y, z]: the position in arcsec
    rpix = [xpix, ypix, zpix]" the position in pixels
    rorig = [xorig, yorig, zorig]" the position in pixels in the 
            original map.
    rerr_p = [xerr_p, yerr_p, zerr_p]: the positional error (arcsec)
    rerr_a = [xerr_a, yerr_a, zerr_a]: the alignment error (arcsec)
    mag = magnitude of star at this epoch
    snr = signal to noise ratio
    corr = correlation value for this star in this epoch
    nframes = the total number of frames for this star
    fwhm = the FWHM for this star
    name = original name for this stars
    """
    def __init__(self, t):
        self.t = t

    def setR(self, r):
        self.x = r[0]
        self.y = r[1]
        self.z = r[2]
    def getR(self):
        return [self.x, self.y, self.z]
    r = property(fget=getR, fset=setR, doc="3D position (arcsec)")

    def setRpix(self, r):
        self.xpix = r[0]
        self.ypix = r[1]
        self.zpix = r[2]
    def getRpix(self):
        return [self.xpix, self.ypix, self.zpix]
    rpix = property(fget=getRpix, fset=setRpix, doc="3D position (pixels)")

    def setRorig(self, r):
        self.xorig = r[0]
        self.yorig = r[1]
        self.zorig = r[2]
    def getRorig(self):
        return [self.xorig, self.yorig, self.zorig]
    rorig = property(fget=getRorig, fset=setRorig, doc="3D position (orig)")

    def setRerrPos(self, r):
        self.xerr_p = r[0]
        self.yerr_p = r[1]
        self.zerr_p = r[2]
    def getRerrPos(self):
        return [self.xerr_p, self.yerr_p, self.zerr_p]
    rerr_p = property(fget=getRerrPos, fset=setRerrPos, 
                      doc="3D positional error")

    def setRerrAlign(self, r):
        self.xerr_a = r[0]
        self.yerr_a = r[1]
        self.zerr_a = r[2]
    def getRerrAlign(self):
        return [self.xerr_a, self.yerr_a, self.zerr_a]
    rerr_a = property(fget=getRerrAlign, fset=setRerrAlign, 
                      doc="3D alignment error")


class Transform(object):
    "Information for coordinate transformations."
    
    def loadAbsolute(self):
        # Assume we loaded an align set in absolute coordinates
        self.scale = 1.0
        self.scaleErr = 0.000001
        self.angle = 0.0
        self.angleErr = 0.01 # deg

        # Use S0-2 orbit determination of Sgr A*s position
        # +x to the West, +y to the North
        self.sgra = [-0.001, -0.005]
        self.sgraErr = [0.001, 0.002]


    def loadFromAbsolute(self,
                         root='/u/jlu/work/gc/proper_motion/align/07_01_17/',
                         align='absolute/align1000.trans', idx=2):
        transTab = Table.read(root + align, format='ascii')

        self.numStars = transTab['NumStars'][idx]
        self.a = [transTab['a0'][idx], transTab['a1'][idx], transTab['a2'][idx]]
        self.b = [transTab['b0'][idx], transTab['b1'][idx], transTab['b2'][idx]]
        self.aerr = [transTab['a0err'][idx], transTab['a1err'][idx], transTab['a2err'][idx]]
        self.berr = [transTab['b0err'][idx], transTab['b1err'][idx], transTab['b2err'][idx]]

    def loadFromAlignTrans(self, root, idx=1):
        transTab = Table.read(root + '.trans', format='ascii')

        self.numStars = transTab['NumStars'][idx]
        self.a = [transTab['a0'][idx], transTab['a1'][idx], transTab['a2'][idx]]
        self.b = [transTab['b0'][idx], transTab['b1'][idx], transTab['b2'][idx]]
        self.aerr = [transTab['a0err'][idx], transTab['a1err'][idx], transTab['a2err'][idx]]
        self.berr = [transTab['b0err'][idx], transTab['b1err'][idx], transTab['b2err'][idx]]

    def loadFromAlign(self, root):
        if os.path.exists(root + '.sgra'):
            sgraTab = Table.read(root + '.sgra', format='ascii')
            self.sgra = [sgraTab['col1'][0], sgraTab['col2'][0]]
            self.sgraErr = [0.0, 0.0]
            self.angle = sgraTab['col3'][0]
            self.angleErr = sgraTab['col4'][0]
        else:
            self.sgra = [0.0, 0.0]
            self.sgraErr = [0.0, 0.0]
            self.angle = 0.0
            self.angleErr = 0.0
        
        scaleTab = open(root + '.scale', 'r')
        line = scaleTab.readline()
        self.scale = float(line)
        self.scaleErr = 0.0

    def linearToSpherical(self, silent=0, override=True):
        sgra = [0.0, 0.0]
        sgraErr = [0.0, 0.0]

        scale = np.sqrt(self.a[1]**2 + self.a[2]**2)
        scaleErr = (self.a[1] * self.aerr[1])**2 
        scaleErr += (self.a[2] * self.aerr[2])**2
        scaleErr = np.sqrt(scaleErr) / scale

        angle = math.atan2(self.a[2], self.a[1])
        angleErr = (self.a[1] * self.aerr[2])**2 
        angleErr += (self.a[2] * self.aerr[1])**2
        angleErr = np.sqrt(angleErr) / scale**2

        sgra[0] = -1 * (self.a[0] * self.a[1] - self.b[0] * self.a[2]) 
        sgra[0] /= scale**2
        sgra[1] = -1 * (self.a[0] * self.a[2] + self.b[0] * self.a[1]) 
        sgra[1] /= scale**2

        tmp = self.a[1]**2 - self.a[2]**2
        term1 = (self.a[0]*tmp - 2.0*self.a[1]*self.a[2]*self.b[0])**2
        term2 = (self.b[0]*tmp + 2.0*self.a[1]*self.a[2]*self.a[0])**2

        sgraErr[0] = (self.a[1] * self.aerr[0] * scale**2)**2 
        sgraErr[0] += (self.a[2] * self.berr[0] * scale**2)**2
        sgraErr[0] += self.aerr[1]**2 * term1
        sgraErr[0] += self.aerr[2]**2 * term2
        sgraErr[0] = np.sqrt(sgraErr[0]) / scale**4

#         print np.sqrt((self.a[1] * self.aerr[0] * scale**2)**2) / scale**4
#         print np.sqrt((self.a[2] * self.berr[0] * scale**2)**2) / scale**4
#         print np.sqrt(self.aerr[1]**2 * term1) / scale**4
#         print np.sqrt(self.aerr[2]**2 * term2) / scale**4
#         print np.sqrt((self.aerr[0]*self.a[1])**2 + (self.berr[0]*self.a[2])**2) / scale**2

        sgraErr[1] = (self.a[1]*self.berr[0]*scale**2)**2
        sgraErr[1] += (self.a[2]*self.aerr[0]*scale**2)**2
        sgraErr[1] +=  self.aerr[1]**2 * term2
        sgraErr[1] += self.aerr[2]**2 * term1
        sgraErr[1] = np.sqrt(sgraErr[1]) / scale**4


        if (override):
            # Numbers computed from Eisenhauer et al. (2005)
            # Computed S0-2 offset from Sgr A* at 2005.495 using
            # orbital parameters.
            #sgra = [728.935, 831.71]
            #sgraErr = [0.155, 0.155]
            # Numbers computed from the best-fit S0-2 focus position
            # of Sgr A*.
            sgra = [728.740, 831.183]
            sgraErr = [0.097, 0.16]
            #sgra = [728.81, 830.96]
            #sgraErr = [0.115, 0.114]
            scale = 0.00995025
            #angle = 0.0

        if (silent == 0):
            print('Absolute astrometry:')
            print('\tScale: %8.5f +/- %7.5f (mas/pixel)' % \
                (scale * 1000.0, scaleErr * 1000.0))
            print('\tAngle: %8.2f +/- %5.2f (degrees)' % \
                (angle * 180.0 / math.pi, angleErr * 180.0 / math.pi))
            print('\tSgr A*: (%9.3f +/- %6.3f, %9.3f +/- %6.3f)' % \
                (sgra[0], sgraErr[0], sgra[1], sgraErr[1]))
            
        self.sgra = sgra
        self.sgraErr = sgraErr
        self.angle = angle
        self.angleErr = angleErr
        self.scale = scale
        self.scaleErr = scaleErr

    def linearToSphericalNew(self, silent=0):
        sgra = [0.0, 0.0]
        sgraErr = [0.0, 0.0]

        scale = np.sqrt(self.a[1]**2 + self.a[2]**2)
        scaleErr = (self.a[1] * self.aerr[1])**2 
        scaleErr += (self.a[2] * self.aerr[2])**2
        scaleErr = np.sqrt(scaleErr) / scale

        angle = math.atan2(self.a[2], self.a[1])
        angleErr = (self.a[1] * self.aerr[2])**2 
        angleErr += (self.a[2] * self.aerr[1])**2
        angleErr = np.sqrt(angleErr) / scale**2

        sgra[0] = -1 * (self.a[0] * self.a[1] - self.b[0] * self.a[2]) 
        sgra[0] /= scale**2
        sgra[1] = -1 * (self.a[0] * self.a[2] + self.b[0] * self.a[1]) 
        sgra[1] /= scale**2

        sgraErr[0] = (self.a[1] * self.aerr[0])**2 
        sgraErr[0] += (self.a[2] * self.berr[0])**2
        sgraErr[0] = np.sqrt(sgraErr[0]) / scale**2

        sgraErr[1] = (self.a[1]*self.berr[0])**2
        sgraErr[1] += (self.a[2]*self.aerr[0])**2
        sgraErr[1] = np.sqrt(sgraErr[1]) / scale**2

        if (silent == 0):
            print('Absolute astrometry:')
            print('\tScale: %8.5f +/- %7.5f (mas/pixel)' % \
                (scale * 1000.0, scaleErr * 1000.0))
            print('\tAngle: %8.2f +/- %5.2f (degrees)' % \
                (angle * 180.0 / math.pi, angleErr * 180.0 / math.pi))
            print('\tSgr A*: (%9.3f +/- %6.3f, %9.3f +/- %6.3f)' % \
                (sgra[0], sgraErr[0], sgra[1], sgraErr[1]))
            
        self.sgra = sgra
        self.sgraErr = sgraErr
        self.angle = angle
        self.angleErr = angleErr
        self.scale = scale
        self.scaleErr = scaleErr

class Efit(object):
    def __init__(self, name):
        self.name = name
    
    def loadAcclim(cls, file):
        f_alim = open(file, 'r')

        set = []
        for line in f_alim:
            _alim = line.split()

            efit = Efit(_alim[0])
            set.append(efit)
            efit.sigma = float(_alim[1])
            efit.bigOmHi = float(_alim[2])
            efit.ax    = float(_alim[3])
            efit.ax_lo = float(_alim[4])
            efit.ax_hi = float(_alim[5])
            efit.ay    = float(_alim[6])
            efit.ay_lo = float(_alim[7])
            efit.ay_hi = float(_alim[8])
            efit.az    = float(_alim[9])
            efit.az_lo = float(_alim[10])
            efit.az_hi = float(_alim[11])
            efit.ap    = float(_alim[12])
            efit.ap_lo = float(_alim[13])
            efit.ap_hi = float(_alim[14])
            efit.at    = float(_alim[15])
            efit.at_lo = float(_alim[16])
            efit.at_hi = float(_alim[17])
            efit.zn    = float(_alim[18])
            efit.zn_lo = float(_alim[19])
            efit.zn_hi = float(_alim[20])
            efit.zp    = float(_alim[21])
            efit.zp_lo = float(_alim[22])
            efit.zp_hi = float(_alim[23])

        f_alim.close()
        return set

    loadAcclim = classmethod(loadAcclim)


    def loadOrbits(cls, file, efitSet=None):
        """Load the .orbits file which contains orbital elements and
        limits on those elements."""
        f_alim = open(file, 'r')
        print(file)

        if (efitSet == None):
            set = []
        else:
            set = efitSet

        cnt = 0
        for line in f_alim:
            _alim = line.split()

            if (efitSet == None):
                efit = Efit(_alim[0])
                set.append(efit)
            else:
                efit = set[cnt]

                # Double check that names agree
                if (efit.name != _alim[0]):
                    print('Efit.loadOrbits: Name mismatch')
    
            cnt += 1
            efit.sigma = float(_alim[1])
            efit.bigOmHi = float(_alim[2])
            efit.omega    = float(_alim[3])
            efit.omega_lo = float(_alim[4])
            efit.omega_hi = float(_alim[5])
            efit.bigOmega    = float(_alim[6])
            efit.bigOmega_lo = float(_alim[7])
            efit.bigOmega_hi = float(_alim[8])
            efit.incl    = float(_alim[9])
            efit.incl_lo = float(_alim[10])
            efit.incl_hi = float(_alim[11])
            efit.ecc    = float(_alim[12])
            efit.ecc_lo = float(_alim[13])
            efit.ecc_hi = float(_alim[14])
            efit.period    = float(_alim[15])
            efit.period_lo = float(_alim[16])
            efit.period_hi = float(_alim[17])
            efit.t0    = float(_alim[18])
            efit.t0_lo = float(_alim[19])
            efit.t0_hi = float(_alim[20])
            efit.phase    = float(_alim[21])
            efit.phase_lo = float(_alim[22])
            efit.phase_hi = float(_alim[23])
            efit.direction = _alim[24]

        f_alim.close()
        return set

    loadOrbits = classmethod(loadOrbits)
    

class Constants(object):
    def __init__(self):
        # Mass and Ro from S0-2 (our data)
        #self.mass = 4.8e6
        #self.dist = 8300.0
        self.mass = 4.07e6 # Ghez et al. (2008)
        self.massErr = 0.60e6 # Ghez et al. (2008)
        self.dist = 7960.1 # Ghez et al. (2008)
        self.distErr = 600. # Ghez et al. (2008)
        # Mass and Ro from Eisenhauer et al. (2005)
        #self.mass = 3.61e6
        #self.dist = 7620.0

        self.G = 6.6726e-8
        self.msun = 1.99e33
        self.sec_in_yr = 3.1557e7
        self.cm_in_au = 1.496e13
        self.cm_in_pc = 3.086e18
        self.au_in_pc = 206265.0
        self.asy_to_kms = self.dist * self.cm_in_au / (1e5 * self.sec_in_yr)

        self.c = 299792.458  # speed of light in km/s

class DataHolder(object):
    def __init__(self):
        return
