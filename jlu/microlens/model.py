import numpy as np
import scipy as sp
import pylab as plt
import scipy.stats
import os, sys
from astropy import constants as const
from astropy import units
import pdb

# Define some constants & conversion factors
kappa = 4.0 * const.G * units.rad / (const.c**2 * units.au)
kappa = kappa.to(units.mas / units.solMass)

days_per_year = 365.25

class PSPL(object):
    """ 
    DESCRIPTION:
    - This function computes the apparent motion, thetaS, of source in the sky plane,
      as it is lensed by a foreground object. It returns thetaS, the angular Einstein 
      radius (thetaE) and lens mass (M). It does NOT account for Earth's orbital motion, 
      but uses microlensing parallax (piE) to compute M. It computes the projected 
      separation and corresponding astrometric shift at each time of observation. 
      It computes thetaE from the given Einstein crossing time, and relative source-lens
      proper motion. M is calculated from piE and thetaE.   
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
        self.dL = dL
        self.dS = dS
        self.xS0 = xS0
        self.beta = beta
        self.muL = muL
        self.muS = muS
        self.imag_base = imag_base

        # Calculate the relative parallax
        inv_dist_diff = (1.0 / (dL * units.pc)) - (1.0 / (dS * units.pc))
        piRel = units.rad * units.au * inv_dist_diff
        self.piRel = piRel.to('mas').value

        # Calculate the relative velocity vector. Note that this will be in the
        # direction of theta_hat
        self.muRel = self.muS - self.muL
        self.muRel_amp = np.linalg.norm(self.muRel) # mas/yr
        
        # Calculate the Eintstein radius
        thetaE = units.rad * np.sqrt((4.0 * const.G * mL * units.M_sun / const.c**2) * inv_dist_diff)
        self.thetaE_amp = thetaE.to('mas').value  # mas
        self.thetaE_hat = self.muRel / self.muRel_amp
        self.thetaE = self.thetaE_amp * self.thetaE_hat

        # Calculate the closest approach vector.
        if beta < 0:
            self.u0_hat = np.array([-self.thetaE_hat[1], self.thetaE_hat[0]])
        else:
            self.u0_hat = np.array([self.thetaE_hat[1], -self.thetaE_hat[0]])

        self.u0_amp = self.beta / self.thetaE_amp   # units of Einstein radii
        self.u0 = self.u0_amp * self.u0_hat
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

        # Convert to matrices for more efficient operations
        u0 = self.u0.reshape(1, len(self.u0))
        thetaE_hat = self.thetaE_hat.reshape(1, len(self.thetaE_hat))
        tau = tau.reshape(len(tau), 1)
        
        u = u0 + tau * thetaE_hat
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


    def likely_photometry(self, t_obs, mag_obs, mag_err_obs):
        flux_base = 10**(self.imag_base / -2.5)
        flux_model = flux_base * self.get_amplification(t_obs)
        mag_model = -2.5 * np.log10(flux_model)

        lnL = ((mag_obs - mag_model) / mag_err_obs)**2

        return lnL

    def likely_astrometry(self, t_obs, x_obs, y_obs, x_err_obs, y_err_obs):
        srce_pos_model = self.xS0 + np.outer((t_obs - self.t0) / model.days_per_year, self.muS) * 1e-3
        pos_model = srce_pos_model + (self.get_centroid_shift(t_obs) * 1e-3)

        lnL_x = ((x_obs - pos_model[:, 0]) / x_err_obs)**2
        lnL_y = ((y_obs - pos_model[:, 0]) / y_err_obs)**2

        lnL = lnL_x + lnL_y

        return lnL

    
        
        


        
        
        
