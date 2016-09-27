import numpy as np
import scipy as sp
import pylab as plt
import scipy.stats
import os, sys
import math
from astropy import constants as const
from astropy import units
import pdb
from astropy.time import Time
import time

# Define some constants & conversion factors
kappa = 4.0 * const.G * units.rad / (const.c**2 * units.au)
kappa = kappa.to(units.mas / units.solMass)

days_per_year = 365.25

class PSPL(object):
    """ 
    DESCRIPTION:
    Point Source Point Lens model for microlensing. This model includes
    proper motions of both the lens and source; but does not include any
    parallax.
    """

    def __init__(self, mL, t0, xS0, beta, muL, muS, dL, dS, imag_base):
        """
        INPUTS:
        ###############################################################################
        t0: Time of photometric peak, as seen from Earth (MJD.DDD)
        mL: Mass of the lens (Msun)
        xS0: vector [RA, Dec] Source position on sky at t = t0 (arcsec) in an arbitrary ref. frame.
        beta: Angular distance between the lens and source on the plane of the sky (mas). Can
              be positive (u0_hat cross thetaE_hat pointing away from us)
              or negative (u0_hat cross thetaE_hat pointing towards us).
        muL: vector [RA, Dec] Lens proper motion (mas/yr)
        muS: vector [RA, Dec] Source proper motion (mas/yr)
        dL: Distance from the observer to the lens (pc)
        dS: Distance from the observer to the source (pc)
        ###############################################################################
        """
        self.t0 = t0
        self.mL = mL
        self.xS0 = xS0
        self.beta = beta
        self.muL = muL
        self.muS = muS
        self.dL = dL
        self.dS = dS
        self.imag_base = imag_base

        # Calculate the relative parallax
        inv_dist_diff = (1.0 / (dL * units.pc)) - (1.0 / (dS * units.pc))
        piRel = units.rad * units.au * inv_dist_diff
        self.piRel = piRel.to('mas').value

        # Calculate the individual parallax
        piS = (1.0 / self.dS) * (units.rad * units.au / units.pc)
        piL = (1.0 / self.dL) * (units.rad * units.au / units.pc)
        self.piS = piS.to('mas').value
        self.piL = piL.to('mas').value

        # Calculate the relative velocity vector. Note that this will be in the
         # direction of theta_hat
        self.muRel = self.muS - self.muL
        self.muRel_amp = np.linalg.norm(self.muRel) # mas/yr
        
        # Calculate the Eintstein radius
        thetaE = units.rad * np.sqrt((4.0 * const.G * mL * units.M_sun / const.c**2) * inv_dist_diff)
        self.thetaE_amp = thetaE.to('mas').value  # mas
        self.thetaE_hat = self.muRel / self.muRel_amp
        self.thetaE = self.thetaE_amp * self.thetaE_hat

        # Calculate the microlensing parallax
        self.piE = (self.piRel / self.thetaE_amp) * self.thetaE_hat

        # Calculate the closest approach vector. Define beta sign convention as Andy Gould
        # does with beta > 0 means u0_x < 0 (pass lens to the right side as seen from earth)?
        # This is a crappy definition.
        self.u0_hat = np.zeros(2, dtype=float)
        if beta > 0:
            self.u0_hat[0] = -np.abs(self.thetaE_hat[1])

            if np.sign(self.thetaE_hat).prod() > 0:
                self.u0_hat[1] = np.abs(self.thetaE_hat[0])
            else:
                self.u0_hat[1] = -np.abs(self.thetaE_hat[0])
        else:
            self.u0_hat[0] = np.abs(self.thetaE_hat[1])

            if np.sign(self.thetaE_hat).prod() > 0:
                self.u0_hat[1] = -np.abs(self.thetaE_hat[0])
            else:
                self.u0_hat[1] = np.abs(self.thetaE_hat[0])
                

        self.u0_amp = self.beta / self.thetaE_amp  # in Einstein units
        self.u0 = np.abs(self.u0_amp) * self.u0_hat
        
        # Angular separation vector between source and lens (vector from lens to source)
        self.thetaS0 = self.u0 * self.thetaE_amp    # mas

        # Calculate the position of the lens on the sky at time, t0
        self.xL0 = self.xS0 - (self.thetaS0 * 1e-3)

        # Calculate the microlensing parallax
        self.piE_amp = self.piRel / self.thetaE_amp
        self.piE = self.piE_amp * self.thetaE_hat

        # Calculate the Einstein crossing time.
        self.tE = (self.thetaE_amp / self.muRel_amp) * days_per_year

        return

    def get_amplification(self, t):
        """Get the photometric amplification term at a set of times, t.

        Inputs
        ----------
        t: Array of times in MJD.DDD
        """

        tau = (t - self.t0) / self.tE

        # Convert to matrices for more efficient operations.
        # Matrix shapes below are:
        #  u0, thetaE_hat: [1, 2]
        #  tau:      [N_times, 1]
        u0 = self.u0.reshape(1, len(self.u0))
        thetaE_hat = self.thetaE_hat.reshape(1, len(self.thetaE_hat))
        tau = tau.reshape(len(tau), 1)

        # Shape of u: [N_times, 2]        
        u = u0 + tau * thetaE_hat

        # Shape of u_amp: [N_times]
        u_amp = np.linalg.norm(u, axis=1)

        A = (u_amp**2 + 2) / (u_amp * np.sqrt(u_amp**2 + 4))

        return A

    def get_centroid_shift(self, t):
        """Get the centroid shift (in mas) for a list of
        observation times (in MJD).
        """
        
        dt_in_years = (t - self.t0) / days_per_year
        tau = (t - self.t0) / self.tE

        # Shape of arrays:
        # thetaS: [N_times, 2]
        # u: [N_times, 2]
        # u_amp: [N_times]
        thetaS = self.thetaS0 + np.outer(dt_in_years, self.muRel) # mas
        u = thetaS / self.thetaE_amp
        u_amp = np.linalg.norm(u, axis=1)

        shift_norm_factor = u_amp**2 + 2.0

        shift = thetaS
        shift[:, 0] /= shift_norm_factor
        shift[:, 1] /= shift_norm_factor
                    
        return shift

    def get_photometry(self, t_obs):
        flux_base = 10**(self.imag_base / -2.5)
        flux_model = flux_base * self.get_amplification(t_obs)
        mag_model = -2.5 * np.log10(flux_model)

        return mag_model

    def get_astrometry(self, t_obs):
        srce_pos_model = self.xS0 + np.outer((t_obs - self.t0) / days_per_year, self.muS) * 1e-3
        pos_model = srce_pos_model + (self.get_centroid_shift(t_obs) * 1e-3)

        return pos_model
        

    def likely_photometry(self, t_obs, mag_obs, mag_err_obs):
        mag_model = self.get_photometry(t_obs)

        lnL = -0.5 * ((mag_obs - mag_model) / mag_err_obs)**2

        return lnL

    def likely_astrometry(self, t_obs, x_obs, y_obs, x_err_obs, y_err_obs):
        pos_model = self.get_astrometry(t_obs)
        
        lnL_x = ((x_obs - pos_model[:, 0]) / x_err_obs)**2
        lnL_y = ((y_obs - pos_model[:, 1]) / y_err_obs)**2

        lnL = -0.5 * (lnL_x + lnL_y)

        return lnL


class PSPL_parallax(PSPL):
    """ 
    DESCRIPTION:
    Point Source Point Lens model for microlensing. This model includes
    proper motions of both the lens and source AND parallax (both the
    microlensing parallax effects on the photometry and astrometry.
    """

    def __init__(self, raL, decL, mL, t0, xS0, beta, muL, muS, dL, dS, imag_base):
        """
        INPUTS:
        ###############################################################################
        t0: Time of photometric peak, as seen from Earth (MJD.DDD)
        mL: Mass of the lens (Msun)
        xS0: vector [RA, Dec] Source position on sky at t = t0 (arcsec) in an arbitrary ref. frame.
        beta: Angular distance between the lens and source on the plane of the sky (mas). Can
              be positive (u0_hat cross thetaE_hat pointing away from us)
              or negative (u0_hat cross thetaE_hat pointing towards us).
        muL: vector [RA, Dec] Lens proper motion (mas/yr)
        muS: vector [RA, Dec] Source proper motion (mas/yr)
        dL: Distance from the observer to the lens (pc)
        dS: Distance from the observer to the source (pc)
        ###############################################################################
        """
        self.raL = raL
        self.decL = decL

        super(PSPL_parallax, self).__init__(mL, t0, xS0, beta, muL, muS, dL, dS, imag_base)
        
        return

    def get_amplification(self, t):
        """Get the photometric amplification term at a set of times, t.

        Inputs
        ----------
        t: Array of times in MJD.DDD
        """
        # Get the parallax vector for each date.
        parallax_vec = parallax_in_direction(self.raL, self.decL, t)

        # Equation of relative motion (angular on sky) Eq. 16 from Hog+ 1995
        dt_in_years = (t - self.t0) / days_per_year
        thetaS = self.thetaS0 + np.outer(dt_in_years, self.muRel)  - (self.piRel * parallax_vec) # mas
        u = thetaS / self.thetaE_amp
        u_amp = np.linalg.norm(u, axis=1)
        
        A = (u_amp**2 + 2) / (u_amp * np.sqrt(u_amp**2 + 4))

        return A

    def get_centroid_shift(self, t):
        tau = (t - self.t0) / self.tE
        
        # Lens-induced astrometric shift of the sum of all source images (in mas)
        numer = (np.outer(tau, self.thetaE_hat) + self.u0) * self.thetaE_amp
        denom = (tau**2.0 + self.u0_amp**2.0 + 2.0).reshape(numer.shape[0], 1)
        shift =  numer / denom

        return shift

    def get_astrometry_unlensed(self, t_obs):
        # Get the parallax vector for each date.
        parallax_vec = parallax_in_direction(self.raL, self.decL, t_obs)

        # Equation of motion for just the background source.
        dt_in_years = (t_obs - self.t0) / days_per_year
        xS_unlensed = self.xS0 + np.outer(dt_in_years, self.muS) * 1e-3
        xS_unlensed += (self.piS * parallax_vec) * 1e-3 # arcsec

        return xS_unlensed

    def get_lens_astrometry(self, t_obs):
        # Get the parallax vector for each date.
        parallax_vec = parallax_in_direction(self.raL, self.decL, t_obs)

        # Equation of motion for just the background source.
        dt_in_years = (t_obs - self.t0) / days_per_year
        xL_unlensed = self.xL0 + np.outer(dt_in_years, self.muL) * 1e-3
        xL_unlensed += (self.piL * parallax_vec) * 1e-3 # arcsec

        return xL_unlensed
        
    def get_astrometry(self, t_obs):

        # Things we will need.
        dt_in_years = (t_obs - self.t0) / days_per_year
            
        # Get the parallax vector for each date.
        parallax_vec = parallax_in_direction(self.raL, self.decL, t_obs)

        # Equation of motion for just the background source.
        xS_unlensed = self.xS0 + np.outer(dt_in_years, self.muS) * 1e-3
        xS_unlensed += (self.piS * parallax_vec) * 1e-3 # arcsec
        
        # Equation of motion for the relative angular separation between the background source and lens.
        thetaS = self.thetaS0 + np.outer(dt_in_years, self.muRel)  # mas
        thetaS -= (self.piRel * parallax_vec)                      # mas
        u_vec = thetaS / self.thetaE_amp
        u_amp = np.linalg.norm(u_vec, axis=1)

        denom = u_amp**2 + 2.0
        
        shift = thetaS / denom.reshape((len(u_amp), 1)) # mas

        xS = xS_unlensed + (shift * 1e-3) # arcsec

        return xS
    
    
        
def parallax_in_direction(RA, Dec, mjd):
    """
    R.A. in degrees. (J2000)
    Dec. in degrees. (J2000)
    MJD
    """
    #### Ecliptic longitude and obliquity of the Sun
    # More Accurate but 100 times slower:
    # time1 = time.time()
    # t = Time(mjd, format='mjd')

    # sun = ephem.Sun()

    # l = np.zeros(len(t), dtype=float)
    # for ii in range(len(t)):
    #     print ii
    #     sun.compute(t.datetime[ii], epoch='2000')
    #     l[ii] = ephem.Ecliptic(sun).lon   # radians
    # time2 = time.time()
    
    # # Obliquity of the ecliptic: 23 deg, 27 min
    # epsilon = 23.0 + (27./60.)   # degrees
    # cose = np.cos(np.radians(epsilon))
    # sine = np.sin(np.radians(epsilon))
    # print 'First round took: ', time2 - time1

    # Less Accurate but 100 times faster. Also gives obliquity (epsilon).
    foo1, foo2, l, epsilon = sun_position(mjd, radians=True)
    # time3 = time.time()
    # print 'Second round took: ', time3 - time2
    epsilon = epsilon[0] # This is constant

    cose = np.cos(epsilon)
    sine = np.sin(epsilon)
        
    cosl = np.cos(l)
    sinl = np.sin(l)
    
    # Earth-Sun Distance -- simplistic for now.
    R_e_s = 1.0  # AU

    # Convert R.A. of target
    cosa = np.cos(np.radians(RA))
    sina = np.sin(np.radians(RA))

    # Convert Dec. of target
    cosd = np.cos(np.radians(Dec))
    sind = np.sin(np.radians(Dec))

    par_E = (cose*cosa*sinl - sina*cosl) * R_e_s
    par_N = ((sine*cosd - cose*sina*sind) * sinl - cosa*sind*cosl) * R_e_s

    parallax_vec = np.array([par_E, par_N]).T

    return parallax_vec

                    
def sun_position(mjd, radians=False):
    """    
    ;+
    ; NAME:
    ;       SUNPOS
    ; PURPOSE:
    ;       To compute the RA and Dec of the Sun at a given date.
    ;
    ; INPUTS:
    ;       mjd    - The modified Julian date of the day (and time), scalar or vector
    ;
    ; OUTPUTS:
    ;       ra    - The right ascension of the sun at that date in DEGREES
    ;               double precision, same number of elements as jd
    ;       dec   - The declination of the sun at that date in DEGREES
    ;       elong - Ecliptic longitude of the sun at that date in DEGREES.
    ;       obliquity - the obliquity of the ecliptic, in DEGREES
    ;
    ; OPTIONAL INPUT KEYWORD:
    ;       RADIAN [def=False] - If this keyword is set to True, then all output variables 
    ;               are given in Radians rather than Degrees
    ;
    ; NOTES:
    ;       Patrick Wallace (Rutherford Appleton Laboratory, UK) has tested the
    ;       accuracy of a C adaptation of the sunpos.pro code and found the 
    ;       following results.   From 1900-2100 SUNPOS  gave 7.3 arcsec maximum 
    ;       error, 2.6 arcsec RMS.  Over the shorter interval 1950-2050 the figures
    ;       were 6.4 arcsec max, 2.2 arcsec RMS.  
    ;
    ;       The returned RA and Dec are in the given date's equinox.
    ;
    ;       Procedure was extensively revised in May 1996, and the new calling
    ;       sequence is incompatible with the old one.
    ; METHOD:
    ;       Uses a truncated version of Newcomb's Sun.    Adapted from the IDL
    ;       routine SUN_POS by CD Pike, which was adapted from a FORTRAN routine
    ;       by B. Emerson (RGO).
    ; EXAMPLE:
    ;       (1) Find the apparent RA and Dec of the Sun on May 1, 1982
    ;       
    ;       IDL> jdcnv, 1982, 5, 1,0 ,jd      ;Find Julian date jd = 2445090.5   
    ;       IDL> sunpos, jd, ra, dec
    ;       IDL> print,adstring(ra,dec,2)
    ;                02 31 32.61  +14 54 34.9
    ;
    ;       The Astronomical Almanac gives 02 31 32.58 +14 54 34.9 so the error
    ;               in SUNPOS for this case is < 0.5".      
    ;
    ;       (2) Find the apparent RA and Dec of the Sun for every day in 1997
    ;
    ;       IDL> jdcnv, 1997,1,1,0, jd                ;Julian date on Jan 1, 1997
    ;       IDL> sunpos, jd+ dindgen(365), ra, dec    ;RA and Dec for each day 
    ;
    ; MODIFICATION HISTORY:
    ;       Written by Michael R. Greason, STX, 28 October 1988.
    ;       Accept vector arguments, W. Landsman     April,1989
    ;       Eliminated negative right ascensions.  MRG, Hughes STX, 6 May 1992.
    ;       Rewritten using the 1993 Almanac.  Keywords added.  MRG, HSTX, 
    ;               10 February 1994.
    ;       Major rewrite, improved accuracy, always return values in degrees
    ;       W. Landsman  May, 1996 
    ;       Added /RADIAN keyword,    W. Landsman       August, 1997
    ;       Converted to IDL V5.0   W. Landsman   September 1997
    ;       Converted to python     J. R. Lu    August 2016
    ;-
    """
    #  form time in Julian centuries from 1900.0
    t_obj = Time(mjd, format='mjd')
    t = (t_obj.jd - 2415020.0) / 36525.0

    #  form sun's mean longitude
    l = (279.696678 + ((36000.768925 * t) % 360.0)) * 3600.0

    #  allow for ellipticity of the orbit (equation of centre)
    #  using the Earth's mean anomaly ME
    me = 358.475844 + ((35999.049750 * t) % 360.0)
    ellcor  = (6910.1 - 17.2*t)*np.sin(np.radians(me)) + 72.3*np.sin(np.radians(2.0*me))
    l = l + ellcor

    # allow for the Venus perturbations using the mean anomaly of Venus MV
    mv = 212.603219 + ((58517.803875*t) % 360.0) 
    vencorr = 4.8 * np.cos(np.radians(299.1017 + mv - me)) + \
              5.5 * np.cos(np.radians(148.3133 +  2.0 * mv  -  2.0 * me)) + \
              2.5 * np.cos(np.radians(315.9433 +  2.0 * mv  -  3.0 * me)) + \
              1.6 * np.cos(np.radians(345.2533 +  3.0 * mv  -  4.0 * me)) + \
              1.0 * np.cos(np.radians(318.1500 +  3.0 * mv  -  5.0 * me))
    l += vencorr

    #  Allow for the Mars perturbations using the mean anomaly of Mars MM
    mm = 319.529425  +  (( 19139.858500 * t)  %  360.0 )
    marscorr = 2.0 * np.cos(np.radians(343.8883 - 2.0 * mm  +  2.0 * me)) + \
               1.8 * np.cos(np.radians(200.4017 - 2.0 * mm  + me))
    l += marscorr

    # Allow for the Jupiter perturbations using the mean anomaly of Jupiter MJ
    mj = 225.328328  +  (( 3034.6920239 * t)  %  360.0 )
    jupcorr = 7.2 * np.cos(np.radians(179.5317 - mj + me )) + \
              2.6 * np.cos(np.radians(263.2167 - mj )) + \
              2.7 * np.cos(np.radians( 87.1450 - 2.0 * mj + 2.0 * me )) + \
              1.6 * np.cos(np.radians(109.4933 - 2.0 * mj + me ))
    l += jupcorr

    # Allow for the Moons perturbations using the mean elongation of
    # the Moon from the Sun D
    d = 350.7376814  + (( 445267.11422 * t)  %  360.0 )
    mooncorr  = 6.5 * np.sin(np.radians(d))
    l += mooncorr

    # Allow for long period terms
    longterm  = + 6.4 * np.sin(np.radians( 231.19  +  20.20 * t ))
    l += longterm
    l = (l + 2592000.0) % 1296000.0
    longmed = l/3600.0

    # Allow for Aberration
    l -= 20.5

    # Allow for Nutation using the longitude of the Moons mean node OMEGA
    omega = 259.183275 - (( 1934.142008 * t ) % 360.0 )
    l -= 17.2 * np.sin(np.radians(omega))

    # Form the True Obliquity
    oblt  = 23.452294 - 0.0130125*t + (9.2*np.cos(np.radians(omega)))/3600.0

    # Form Right Ascension and Declination
    l = l/3600.0
    l_rad = np.radians(l)
    oblt_rad = np.radians(oblt)
    ra  = np.arctan2( np.sin(l_rad) * np.cos(oblt_rad), np.cos(l_rad))

    if (len(ra) > 1):
        neg = np.where(ra < 0.0)[0]
        ra[neg] = ra[neg] + 2.0*math.pi

    dec = np.arcsin(np.sin(l_rad) * np.sin(oblt_rad))
 
    if radians:
        oblt = oblt_rad
        longmed = np.radians(longmed)
    else:
        ra = np.degrees(ra)
        dec = np.degrees(dec)

    return ra, dec, longmed, oblt
    

