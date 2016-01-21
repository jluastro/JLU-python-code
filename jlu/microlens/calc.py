import numpy as np
import pylab as py
import astropy.units as u
import astropy.constants as c
import datetime
import time
import pdb

def einstein_radius(lens_mass, lens_dist, source_dist):
    """
    Calculate the einstein radius for a source and lens
    given their distances and the mass of the lens.

    Parameters:
    lens_mass -- in units of solar masses
    lens_dist -- in units of pc
    source_dist -- in units of pc
    
    Return: Einstein radius in milli-arcseconds.
    """
    m = u.Quantity(lens_mass, unit=u.Msun)
    d_L = u.Quantity(lens_dist, unit=u.pc)
    d_S = u.Quantity(source_dist, unit=u.pc)

    coeff = 4 * c.G * m / c.c**2
    dist_inv = (1.0 / d_L) - (1.0 / d_S)  # almost pi_rel
    theta_E = np.sqrt(coeff * dist_inv)   # unit less... in radians

    theta_E *= 206265.0 * 1.0e3

    return theta_E.decompose().value

def parallax_correction(target, t0, tE, xS0, muS, muRel, beta, piE):
    if target == 'ob110022':
        RA_split  = [ 17, 53, 17.93]
        Dec_split = [-30,  2, 29.30]
        dates = [datetime.datetime(2011, 5, 25, 10, 0, 0),
                 datetime.datetime(2011, 7, 7, 10, 0, 0),
                 datetime.datetime(2012, 6, 23, 10, 0, 0),
                 datetime.datetime(2012, 7, 10, 10, 0, 0),
                 datetime.datetime(2013, 4, 30, 10, 0, 0),
                 datetime.datetime(2013, 7, 15, 10, 0, 0)]
    if target == 'ob110125':
        RA_split  = [ 18,  3, 32.95]
        Dec_split = [-29, 49, 43.00]
        dates = [datetime.datetime(2012, 5, 23, 10, 0, 0),
                 datetime.datetime(2012, 6, 23, 10, 0, 0),
                 datetime.datetime(2012, 7, 10, 10, 0, 0),
                 datetime.datetime(2013, 4, 30, 10, 0, 0),
                 datetime.datetime(2013, 7, 15, 10, 0, 0)]
    if target == 'ob120169':
        RA_split  = [ 17, 49, 51.38]
        Dec_split = [-35, 22, 28.00]
        dates = [datetime.datetime(2012, 5, 23, 10, 0, 0),
                 datetime.datetime(2012, 6, 23, 10, 0, 0),
                 datetime.datetime(2012, 7, 10, 10, 0, 0),
                 datetime.datetime(2013, 4, 30, 10, 0, 0),
                 datetime.datetime(2013, 7, 15, 10, 0, 0)]

    # Convert RA and Dec to degrees
    RA = (RA_split[0] + (RA_split[1] / 60.) + (RA_split[2] / 3600.)) * 360. / 24.
    Dec = Dec_split[0] + (Dec_split[1] / 60.) + (Dec_split[2] / 3600.)

    # Convert dates to fractional year.
    year = np.zeros(len(dates), dtype=float)
    day_in_year = np.zeros(len(dates), dtype=float)
    for dd in range(len(dates)):
        year[dd] = toYearFraction(dates[dd])
        day_in_year[dd] = 365.0 * (year[dd] % 1)
        
    # year = np.arange(2005, 2025, 0.1)
    # day_in_year = 360.0 * (year % 1)

    # Define some constants & conversion factors
    daysPerYr = 365.25 
    kappa = 8.1459 # 4.0*G/(c^2*AU) units of mas/Msun

    t = year
    
    # Compute the Einstein radius (amplitude) and piRel
    muRel_amp = np.linalg.norm(muRel)         # mas/yr
    thetaE_amp = muRel_amp * tE / daysPerYr   # mas
    piRel = np.linalg.norm(piE) * thetaE_amp  # mas
    
    # Rotate mu_rel by 90 degrees CCW to get beta unit vector
    muRel_hat = muRel / muRel_amp  # same as thetaE_hat
    beta_hat = np.array([-muRel_hat[1], muRel_hat[0]])  # same as u0_hat

    # This becomes an array with different times.
    tau = (t - t0) / (tE / daysPerYr)

    #Convert to matrices to for more efficient operations
    t = t.reshape(len(t), 1) 
    muRel = muRel.reshape(1, len(muRel))
    beta_hat = beta_hat.reshape(1, len(beta_hat))
    muS = muS.reshape(1, len(muS))
    tau = tau.reshape(len(tau), 1)
    
    # 
    #   1) no parallax. This is the way Evan's calculates it.
    #
    # Equation of relative motion in Einstein units (Eq 4 of Sinukoff+)
    u_nop = (beta * beta_hat) + (tau * muRel_hat)
    uamp_nop = np.sqrt(beta**2 + tau**2)
    
    # Astrometric shift (Eq. 8 of Sinkoff+)
    dc_nop = thetaE_amp * u_nop / (uamp_nop**2 + 2.0)

    # Observed motion (Eq. 12 of Sinukoff+)
    x_nop = xS0 + muS*(t - t0) + dc_nop

    # 
    #   2) parallax included. 
    #
    # Parallax vector (P(t) vector from Eq. 13 in Hog+ 1995
    parallax_vec = parallax_in_direction(RA, Dec, day_in_year)
    
    # Equation of relative motion (angular on sky) Eq. 16 from Hog+ 1995
    r_par = (beta * thetaE_amp * beta_hat) + muRel * (t - t0) - piRel * parallax_vec
        
    u_par = r_par / thetaE_amp
    uamp_par = np.linalg.norm(u_par, axis=1)
    
    # Astrometric shift
    dc_par = thetaE_amp * u_par
    dc_par[:, 0] /= (uamp_par**2 + 2.0)
    dc_par[:, 1] /= (uamp_par**2 + 2.0)
    
    # Observed motion
    piS = 1.0e3 / 8000.0  # in mas. assume source is at 8 kpc.
    x_par = xS0 + muS*(t - t0) + dc_par + (piS * parallax_vec) # missing source parallax

    py.figure(1)
    py.clf()
    py.plot(dc_nop[:,0], dc_nop[:, 1], 'ko', label='No Parallax')
    py.plot(dc_par[:,0], dc_par[:, 1], 'ro', label='Parallax')
    py.xlabel('x_E lensing shift (mas)')
    py.ylabel('x_N lensing shift (mas)')
    py.legend(numpoints=1)
    py.title(target)
    
    py.figure(2)
    py.clf()
    py.plot(x_nop[:,0], x_nop[:, 1], 'ko', label='No Parallax')
    py.plot(x_par[:,0], x_par[:, 1], 'ro', label='Parallax')
    py.xlabel('x_E position (mas)')
    py.ylabel('x_N position (mas)')
    py.legend(numpoints=1)
    py.title(target)
    
    py.figure(3)
    py.clf()
    py.plot(t - t0, x_par[:, 0] - x_nop[:,0], 'ko', label='x_E')
    py.plot(t - t0, x_par[:, 1] - x_nop[:,1], 'ro', label='x_N')
    py.xlabel('Time since t_0 (year)')
    py.ylabel('Parallax - No Parallax position (mas)')
    py.legend(numpoints=1)
    py.title(target)

    print 'RMS of Parallax - No Parallax (mas)'
    print 'X: ', (x_par[:, 0] - x_nop[:,0]).std()
    print 'Y: ', (x_par[:, 1] - x_nop[:,1]).std()
    
    return

def parallax_in_direction(RA, Dec, day_of_year):
    """
    R.A. in degrees.
    Dec. in degrees.
    day of year (in days since beginning of the year)
    """
    # Obliquity of the ecliptic: 23 deg, 27 min
    epsilon = 23.0 + (27./60.)   # degrees
    cose = np.cos(np.radians(epsilon))
    sine = np.sin(np.radians(epsilon))

    # Ecliptlic longitude of the Sun
    l = 360.0 * (day_of_year + 284.0) / 365.0   # degrees
    cosl = np.cos(np.radians(l))
    sinl = np.sin(np.radians(l))

    # Earth-Sun Distance -- simplistic for now.
    R_e_s = 1.0  # AU

    # Convert R.A.
    cosa = np.cos(np.radians(RA))
    sina = np.sin(np.radians(RA))

    # Convert Dec.
    cosd = np.cos(np.radians(Dec))
    sind = np.sin(np.radians(Dec))

    par_E = (cose*cosa*sinl - sina*cosl) * R_e_s
    par_N = ((sine*cosd - cose*sina*sind) * sinl - cosa*sind*cosl) * R_e_s

    parallax_vec = np.array([par_E, par_N]).T

    return parallax_vec
                    
    
def toYearFraction(date):
    def sinceEpoch(date): # returns seconds since epoch
        return time.mktime(date.timetuple())
    s = sinceEpoch

    year = date.year
    startOfThisYear = datetime.datetime(year=year, month=1, day=1)
    startOfNextYear = datetime.datetime(year=year+1, month=1, day=1)

    yearElapsed = s(date) - s(startOfThisYear)
    yearDuration = s(startOfNextYear) - s(startOfThisYear)
    fraction = yearElapsed/yearDuration

    return date.year + fraction
