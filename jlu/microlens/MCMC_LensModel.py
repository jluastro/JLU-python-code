import numpy as np
import os
import sys
import string
import pylab as py
import scipy.stats
import scipy as sp

def go(starNames, rootDir='./', align='align/align_t',
       poly='polyfit_d/fit', points='points_d/'):
    
    GetData(starNames=starNames, rootDir = rootDir, align = align, 
            poly=poly, points=points)
    #Choose Start vector
    

def GetData(starNames, rootDir='./', align='align/align_t',
            poly='polyfit_d/fit', points='points_d/'):
    
    
    s = starset.StarSet(root + align)
    s.loadPolyfit(root + poly, accel=0, arcsec=0)
    names = s.getArray('name')
    ii = names.index(StarName)
    star = s.stars[ii]
    
    pointsTab = Table.read(rootDir + points + StarName + '.points', format='ascii')
        
    time = pointsTab['col1']
    x = pointsTab['col2']
    y = pointsTab['col3']
    xerr = pointsTab['col4']
    yerr = pointsTab['col5']
        
    fitx = star.fitXv
    fity = star.fitYv
    dt = time - fitx.t0
    fitLineX = fitx.p + (fitx.v * dt)
    fitSigX = np.sqrt( fitx.perr**2 + (dt * fitx.verr)**2 )
  
    fitLineY = fity.p + (fity.v * dt)
    fitSigY = np.sqrt( fity.perr**2 + (dt * fity.verr)**2 )

    diffX = x - fitLineX
    diffY = y - fitLineY   
    
    nEpochs =  len(x)
    return x, y, xerr, yerr 



def LensModel_Simple(tobs, to, te, M, pi_E, beta, OutputUnits = 'pix', plateScale = None):
    
    """ 
    DESCRIPTION:
    - This function computes the astrometric shift of a background source caused by
      a lens of given mass and impact parameter using photometrically derived inputs.  
      Specifically, the time of the photometric peak, Einstein crossing time, and 
      microlens parallax must be known.  
    
    
    INPUTS:
    ###############################################################################
    tobs = times of observation (yrs, e.g. 2014.123, assuming 1 yr = 365.2425 days)
    to = time of photometric peak (same units as tobs)
    te = Einstein crossing time (days)
    M = Lens mass (Msun) 
    pi_E = Microlens parallax (mas)
    beta = Impact parameter, in units of Einstein radii (>= 0)
    OutputUnits = Desired units of lens model. Either 'mas' or 'pix' (Default)
                  If 'pix', must provide correct plateScale
    PlateScale = mas per pixel.  Only necessary if OutputUnits = 'pix'
                 9.952 for Keck NIRC2 obs   
    ###############################################################################
    
    
    OUTPUTS: 
    ###############################################################################
    Astrometric shift in X and Y at times t, with desired "OutputUnits"
    ############################################################################### 
    """   
    py.clf()
    # Compute times relative to 
    # photometric peak, units of te
    t = np.array(tobs)
    te_yrs = te/365.2425				         # Convert te to years    
    tau = (t-to)/te_yrs					   		   
    
    # Compute Einstein radius 
    # from M and pi_E
    Kappa = 1.91e-16                       		 # Kappa = 4G/c^2 = 1.91e-16 kpc/Msun
    AU = 4.848e-9                      	   		 # 1 AU = 4.848e-9 kpc
    masPerRad = 2.06265e8                  		  
    D = (pi_E * np.sqrt(Kappa * M) / AU)**2. # D = (1/dl)-(1/ds), units of kpc^(-1)    
    theta_E = np.sqrt(Kappa * M * D)  		   	 # Einstein radius 
    
    # Compute astrometric shifts at times t
    u = np.sqrt(tau**2. + beta**2.)  # Proj. separation, units of Einstein rad
    if np.min(t) < to:
        ind = np.where((t-to) < 0.0)[0]
        u[ind] = -u[ind]		   	    
    term1 = u**2. + 2.    
    dthetaX_rad = theta_E * tau / term1    
    dthetaX_mas = masPerRad * dthetaX_rad
    dthetaY_rad = theta_E * beta / term1
    dthetaY_mas = masPerRad * dthetaY_rad
    
    # Manage some possible input errors        
    if OutputUnits == 'pix':
        try:
            dthetaX = dthetaX_mas/plateScale
            dthetaY = dthetaY_mas/plateScale
        except TypeError:   #TypeError will occur if plateScale is default 'None' 
            print "ERROR: plateScale must be provided when OutputUnits='pix'"
            print 'ABORTING...'
            sys.exit()
    elif OutputUnits == 'mas': 
        dthetaX = dthetaX_mas
        dthetaY = dthetaY_mas
    else:
        print "ERROR: OutputUnits must be either 'mas' or 'pix'"
        print 'ABORTING...'
        sys.exit()    
	
	#Need mag_base as free parameter?
	# MOA Calibration: I = 27.55 - 2.5 log10(deltaF + 59941.63)) 
	A = term1/(np.abs(u)* np.sqrt(term1+2.))
	mag = mag_base - 2.5 *np.log10(A)
	
	
    #Plot
    py.plot(u, dthetaX)
    py.xlim(-3, 3)
    py.ylim(0,10**0.2)
    ax=py.gca() 
    ax.set_yscale('linear')
    py.savefig('/Volumes/data3/LensModel.pdf')
    py.clf()
    py.plot(t, dthetaX)
    py.plot(t, dthetaY, 'k')
    py.plot(t, np.sqrt(dthetaX**2.+dthetaY**2.), 'r')
    py.xlim(2011, 2016)
    py.ylim(-(10**0.2),10**0.2)
    py.ylabel = 'Astrometric shift (mas)'
    py.xlabel = 'Year'
#     py.annotate('beta = ' + str(beta))
    py.savefig('/Volumes/data3/sinukoff/LensModel_vs_t_beta15.pdf')
    	
#     py.plot(u, np.sqrt(dthetaX**2. + dthetaY**2.) )  
    
     
    py.plot(np.abs(u), dthetaX)
    py.plot(np.abs(u), dthetaX/np.sqrt(2.))
    py.plot(np.abs(u), dthetaX/np.sqrt(10.))
    py.xlim(10**-3,10**2)
    py.ylim(0,1.4)
    ax=py.gca() 
    ax.set_xscale('log')
    py.savefig('/Volumes/data3/sinukoff/LensModel_log.pdf')
    return dthetaX, dthetaY
	

def LensModel_Complex(tobs, t0, tE, srcCoo_t0, lensCoo_t0, src_mu, Ds,  
                      v_twidle, vl, vLSR):
    
    """ 
    DESCRIPTION:
    - This function derives the astrometric shift induced by a lens event. 
      It first computes the lens mass, distance, and angular Einstein radius   
      from the given Einstein crossing time, proper motion of the lens and source, 
      source distance, lens velocity, solar velocity, and 
      v_twidle (which is obtained from the microlens parallax fit). It then computes
      the projected separation and corresponding astrometric shift at each time tobs.  
    
    
    INPUTS:
    ###############################################################################
    tE = Einstein crossing time (days)
    t0 = reference time (yrs)
    srcCoo_t0 = vector [x,y] specifying position of source at t = t0
    lensCoo_t0 = vector [x,y] specifying position of lens at t = t0
    src_mu = source proper motion (mas/yr)
    Ds = Source distance
    v_twidle = transverse speed of lens projected to position of Sun. 
               This is obtained from the microlens parallax fit. (km/s)
    vl = lens velocity (km/s)
    vLSR = Velocity of the local standard of rest 
          (radial, azimuthal, vertical)
          (Upec, Vpec, Wpec) = (11.1, 12.24, 7.25), 
          uncertainties of (1.23, 2.05, 0.62) km/s (Schonrich, Binney, & Dehnen (2010))     
    
    OUTPUTS: 
    ###############################################################################
    Astrometric shift at each time tobs (vector [x,y])
    ############################################################################### 
    """   
    
    
    c_kpc_d = 8.395e-7 #speed of light kpc/day
    secPerYr = 3.15569
    src_mu_sec = src_mu * secPerYr 
    
    G = 4.302e-6 #Gravitational constant in (kpc/Msun)*(km/s)^2
    c1 = v_twidle + vLSR - vl
    t = tobs - t0
    
    M = ((v_twidle*tE*c_kpc_d)**2.0/(16.*G*Ds)) *  (vl-src_mu)/c1
    Dl = c1 /( ((v_twidle + vLSR)/Ds) - src_mu)
    D = (1./Dl)-(1./Ds)
    
    thetaE = np.sqrt((4.*G*M*D/c**2.))	
    
    #how to treat as vectors?
    src_mas =  (srcCoo_t0 + src_mu*Ds*t)/Ds 
    lens_mas =  (lensCoo_t0 + vl*t)/Dl 
    
    u = (src_mas - lens_mas)/thetaE #(mas)
    usq = (np.sum(u))**2.
    
    shift = (u/(usq+2.))*thetaE
    
    return shift 
    


def LensModel_parallax(RA, Dec, tobs, t0, tE, thetaS0, vS, dS, dL, beta,  
                      piE, vLSR):
    
    """ 
    DESCRIPTION:
    - This function derives the astrometric shift induced by a lens event. 
      It first computes the lens mass, distance, and angular Einstein radius   
      from the given Einstein crossing time, proper motion of the lens and source, 
      source distance, lens velocity, solar velocity, and 
      pi_E (which is obtained from the microlens parallax fit). It then computes
      the projected separation and corresponding astrometric shift at each time tobs.  
    
    
    INPUTS:
    ###############################################################################
    RA = RA of target (deg)
    Dec = Declination of target (deg)
    tobs = Times of observation (yrs)
    tE = Einstein crossing time (days)
    t0 = Time of photometric peak, as seen from Earth (yrs)
    thetaS0 = vector [RA,Dec] specifying position of source at t = t0
    beta = vector [x,y] specifying position of lens at t = t0 (in units of the Einstein radius)
    muS = source proper motion (mas/yr)
    vS = Source velocity (km/s)
    vL = Lens velocity (km/s)  (might be possible to solve for this in terms of 
    dS = Source distance (kpc)
    dL = Lens distance (kpc)
    piE = Microlensing parallax vector [piE_N, piE_E] (units of inverse Einstein radii)
    vLSR = Velocity of the local standard of rest (km/s) 
          (radial, azimuthal, vertical)
          (Upec, Vpec, Wpec) = (11.1, 12.24, 7.25), 
          uncertainties of (1.23, 2.05, 0.62) km/s (Schonrich, Binney, & Dehnen (2010)) 
    ###############################################################################
    
    
    OUTPUTS: 
    ###############################################################################
    Astrometric shift at each time tobs (vector [RA,Dec])
    ############################################################################### 
    """   
    
    tobs = np.array(tobs)
    thetaE = np.array(thetaE)
    vS = np.array(vS)
    vL = np.array(vL)
    beta = np.array(beta)
    piE = np.array(piE)
    vLSR = np.array(vLSR)
    
    
    #Define some constants & conversion factors
    G = 4.302e-6 #Gravitational constant in (kpc/Msun)*(km/s)^2
    c_kpc_d = 8.395e-7 #speed of light kpc/day
    radPermas = 4.8481368e-9
    secPerYr = 3.15576e7 
    secPerDay = 8.64e4
    kmPerAU = 1.49597871e8 
    kpcPerKm = 3.2407793e-17
    eps0_deg = 23.5
    eps0 = eps0_deg*3.141592654/180.0
    daysPerYr = 365.25 
    
    #Convert inputs to appropriate units
    muS_sec = muS * secPerYr 
    RA = RA*np.radians(1)
    Dec = Dec*np.radians(1)
    
    #Get vL from dL, dS, pi_E
    X = dL/dS 
    vTwidle = (kmPerAU/secPerDay)/(piE*tE)  #km/s...NEEDS TO BE VECTORIZED...Should AU be replaced by projected separation???
    vL = (1.0-X)*(vLSR + vTwidle) + X*vS # From Bennett+(2002) 
      
    mu_rel = (vS/dS - vL/dL)*kpcPerKm*secPerYr/radPermas # mas/yr
    mu_rel_amp = np.linalg.norm(mu_rel) # mas/yr
    thetaE = mu_rel_amp*tE/daysPerYr # mas
    piE_amp = np.linalg.norm(piE)
    pi_rel_amp = piE_amp*thetaE 
    P = np.array([baryX*np.sin(RA)/np.cos(Dec) - baryY*np.cos(RA)/np.cos(Dec), baryX*np.cos(RA)*np.sin(Dec) - baryY*np.tan(eps0)*np.cos(Dec) - np.sin(RA)*np.sin(Dec)])
        
         
    tau = (t-t0)/tE
    
    u = beta + (mu_rel/mu_rel_amp)*tau + piE_amp*P
    dtheta = u*thetaE  #mas
    
    u2 = np.sum(u**2.0)   
    
    shift = (u/(u2+2.))*thetaE
    
    thetaS = thetaS0 + mu_rel*(t-t0) + shift
    
    return thetaS 
    


def LensModel_Gould(tobs, t0, tE, thetaS0, vS, dS, dL, beta,  
                      piE, vLSR):
    
    """ 
    DESCRIPTION:
    - This function derives the astrometric shift induced by a lens event. 
      It first computes the lens mass, distance, and angular Einstein radius   
      from the given Einstein crossing time, proper motion of the lens and source, 
      source distance, lens velocity, solar velocity, and 
      pi_E (which is obtained from the microlens parallax fit). It then computes
      the projected separation and corresponding astrometric shift at each time tobs.  
    
    
    INPUTS:
    ###############################################################################
    RA = RA of target (deg)
    Dec = Declination of target (deg)
    tobs = Times of observation (yrs)
    tE = Einstein crossing time (days)
    t0 = Time of photometric peak, as seen from Earth (yrs)
    thetaS0 = vector [RA,Dec] specifying position of source at t = t0
    beta = vector [x,y] specifying position of lens at t = t0 (in units of the Einstein radius)
    muS = source proper motion (mas/yr)
    vS = Source velocity (km/s)
    vL = Lens velocity (km/s)  (might be possible to solve for this in terms of 
    dS = Source distance (kpc)
    dL = Lens distance (kpc)
    piE = Microlensing parallax vector [piE_N, piE_E] (units of inverse Einstein radii)
    vLSR = Velocity of the local standard of rest (km/s) 
          (radial, azimuthal, vertical)
          (Upec, Vpec, Wpec) = (11.1, 12.24, 7.25), 
          uncertainties of (1.23, 2.05, 0.62) km/s (Schonrich, Binney, & Dehnen (2010)) 
    ###############################################################################
    
    
    OUTPUTS: 
    ###############################################################################
    Astrometric shift at each time tobs (vector [RA,Dec])
    ############################################################################### 
    """   
    
    tobs = np.array(tobs)
    thetaE = np.array(thetaE)
    vS = np.array(vS)
    vL = np.array(vL)
    beta = np.array(beta)
    piE = np.array(piE)
    
    
    #Define some constants & conversion factors
    G = 4.302e-6 #Gravitational constant in (kpc/Msun)*(km/s)^2
    c_kpc_d = 8.395e-7 #speed of light kpc/day
    radPermas = 4.8481368e-9
    secPerYr = 3.15576e7 
    secPerDay = 8.64e4
    kmPerAU = 1.49597871e8 
    kpcPerKm = 3.2407793e-17
    daysPerYr = 365.25 
        
    #Get vL from dL, dS, pi_E
    X = dL/dS 
    vTwidle = (kmPerAU/secPerDay)/(piE*tE)  #km/s...NEEDS TO BE VECTORIZED...Should AU be replaced by projected separation???
    vL = (1.0-X)*(vLSR + vTwidle) + X*vS # From Bennett+(2002)   
    
    mu_rel = (vS/dS - vL/dL)*kpcPerKm*secPerYr/radPermas # mas/yr
    mu_rel_amp = np.linalg.norm(mu_rel) # mas/yr
    thetaE = mu_rel*tE/daysPerYr # mas
    thetaE_amp = np.linalg.norm(thetaE)
    piE_amp = np.linalg.norm(piE)
    pi_rel_amp = piE_amp*thetaE 
        
    #Rotate mu_rel by 90 degrees CW to get beta unit vector
    mu_rel_hat = mu_rel/mu_rel_amp
    beta_hat = np.array(mu_rel_hat[1], -mu_rel_hat[0])
    
    tau = (t-t0)/tE
    
    sn, se = Get_s_vals(t)  
     
    dtau =  sn*piE[0] + se*piE[1] 
    dbeta = -sn*piE[1] + se*piE[0]
    
    shift = ((tau + dtau)*(thetaE) + (beta+dbeta)*beta_hat*thetaE_amp)/((tau + dtau)**2.0 + beta**2.0 + 2.0)  
    thetaS = thetaS0 + muS*(t-t0) + shift    
    
    return thetaS 



def LensModel_Trial1(tobs, t0, tE, thetaS0, muS, muRel, beta, piE):
    
    """ 
    DESCRIPTION:
    - This function computes the apparent motion, thetaS, of source in the sky plane,
      as it is lensed by a foreground object. It returns thetaS, the angular Einstein 
      radius (thetaE) and lens mass (M). It does NOT account for Earth's orbital motion, 
      but uses microlensing parallax (piE) to compute M. It computes the projected 
      separation and corresponding astrometric shift at each time of observation. 
      It computes thetaE from the given Einstein crossing time, and relative source-lens
      proper motion. M is calculated from piE and thetaE.   
    
    
    INPUTS:
    ###############################################################################
    tobs = Times of observation (YYYY.YYY) 
    t0 = Time of photometric peak, as seen from Earth (yrs)
    tE = Einstein crossing time (days)
    thetaS0 = vector [RA,Dec] specifying position of source at t = t0, relative to 
              observed source position at time closest to t0 
    muS = vector [RA,Dec] Source proper motion (mas/yr)
    muRel = vector [RA,Dec] Source-lens relative proper motion (mas/yr)
    beta = impact parameter (scalar in units of Einstein radius)
    piE = vector [piE_N, piE_E] microlensing parallax 
    ###############################################################################
    
    
    OUTPUTS: 
    ###############################################################################
    1. Source position at each time tobs (vector [RA,Dec]) accounting for lensing 
    2. Angular Einstein radius 
    3. Lens mass 
    ############################################################################### 
    """   
    
    t = np.array(tobs)
    thetaS0 = np.array(thetaS0)
    muS = np.array(muS)
    muRel = np.array(muRel)
    beta = np.array(beta)
    piE = np.array(piE)
    
    
    #Define some constants & conversion factors
    daysPerYr = 365.25 
    kappa = 8.1459 # 4.0*G/(c^2*AU) units of mas/Msun
    
    #Compute Einstein radius (amplitude)
    muRel_amp = np.linalg.norm(muRel) # mas/yr
    thetaE_amp = muRel_amp*tE/daysPerYr # mas
        
    #Rotate mu_rel by 90 degrees CCW to get beta unit vector
    muRel_hat = muRel/muRel_amp
    beta_hat = np.array([-muRel_hat[1], muRel_hat[0]])
    
    tau = (t-t0)/(tE/daysPerYr)
    
    #Convert to matrices to for more efficient operations
    t=t.reshape(len(t),1) 
    muRel = muRel.reshape(1,len(muRel))
    beta_hat = beta_hat.reshape(1,len(beta_hat))
    muS = muS.reshape(1,len(muS))
    tau = tau.reshape(len(tau),1)
    
    #Lens-induced astrometric shift
    shift = (muRel*(t-t0) + beta*beta_hat*thetaE_amp)/(tau**2.0 + beta**2.0 + 2.0) 

    #Observed motion
    thetaS = thetaS0 + muS*(t-t0) + shift    
    #thetaS = shift
    
    #COMPUTE LENS MASS
    #Convert piE (units of thetaE) to absolute angular units
    piRel = np.linalg.norm(piE)*thetaE_amp  # mas
    
    #Lens Mass
    M = (thetaE_amp)**2.0/(kappa*piRel) # Msun
    #print muRel_hat
    #print beta_hat 
    thetaS_nolens = thetaS0 + muS*(t-t0)
    return thetaS, thetaE_amp, M , shift, thetaS_nolens  

	
def lnLikelihood(M, D, err):
    
    """ 
    DESCRIPTION:
    - This function computes the likelihood, P(M|D), of a model M based on the 
      chi square of its fit to a set of data points D. It returns the natural 
      log of the likelihood, ln(P(M|D).
    
    
    INPUTS:
    ###############################################################################
    M = List of vectors (tuples): each model point    
    D = List of vectors (tuples): each data point 
    err = List of vectors (tuples): errors on each vector element in D
    ###############################################################################
    
    OUTPUTS: 
    ###############################################################################
    Natural log of the likelihood, ln(P(M|D)
    ############################################################################### 
    """             
    
    lnlikelihood = -ChiSq(M, D, err)/2.
    
    return lnlikelihood
    
    
def ChiSq(M, D, err):

    """ 
    DESCRIPTION:
    - This function computes the chi square statistic for a model fitted to data in 
      a vector space of any dimensionality.
       
    INPUTS:
    ###############################################################################
    M = List of vectors (tuples): each model point    
    D = List of vectors (tuples): each data point  
    err = List of vectors (tuples): errors on each vector element in D
    ###############################################################################

    OUTPUTS: 
    ###############################################################################
    chi square statistic of model fit to data
    ############################################################################### 
    """           
    
    M = np.array(M)
    D = np.array(D) 
    Npts = len(M)
    if len(M) != len(D):
        print "ERROR: # of model points does not match # of data points"
        print 'ABORTING...'
        sys.exit()    
    
    # Compute difference between model vector and data vector 
    diff = np.array([np.sqrt(((M[i]-D[i])**2.).sum()) for i in range(Npts)]) 
    # Compute error on data vector
    Derr = np.array( [np.sqrt( ((D[i] * err[i])**2.).sum() / (D[i]**2.).sum() )  for i in range(Npts)] )
    # Compute chi square
    chisq = ((diff/Derr)**2.).sum()        
      
    return chisq    
