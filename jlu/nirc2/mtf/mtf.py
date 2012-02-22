import mmm
import numpy as np
import pdb
import math
from scipy import fftpack
from scipy import interpolate
from scipy import integrate
from scipy import special
from jlu.util import radialProfile
import pylab as py
from matplotlib import nxutils
from jlu.util import mpfit
import time

Fvalues = {'wide': 139.9, 'narrow': 557.0}

def get_mtf(image, params, sources):
    """
    image - 2D numpy array containing the image. Image must be square.
    params - a dictionary with the various parameters used to calculate
             the MTF.
    sources - (def=None) and optional 2D numpy array (same size as image)
              that contains the distribution of stars in the image as delta
              functions.
    """
    # Some image testing to catch unexpected inputs
    if len(image.shape) != 2:
        print 'Input image must be 2-dimensional'
        return
    if image.shape[0] != image.shape[1]:
        print 'Input image must be square'
        return

    if (sources != None):
        if (len(sources.shape) != 2) or (sources.shape[0] != sources.shape[1]):
            print 'Input sources must be of the same size as image'
            return

    # Pull out the necessary paramters
    D = params['D']              # telescope primary mirror diameter in meters
    wave = params['wave']        # observing wavelength in meters
    F = params['F']              # effective focal length at detector in meters
    Apix = params['Apix']        # pixel size of detector in meters
    platescale = Apix / F        # plate scale of detector in radians/pixel

    # Calculate the sky in the image.
    skyInfo = mmm.mmm(image)
    skyMode = skyInfo['mode']
    skySigma = skyInfo['sigma']

    # Apodize the image with a Hanning kernal to enforce periodicity
    szx = image.shape[0]
    szy = image.shape[1]
    han = hanning(szx, szy)
    img_skysub = image - skyMode

    fftim = fftpack.fft2(img_skysub * han) / (szx * szy)
    absim = np.real( fftim * fftim.conjugate() )
    absim[0,0] = np.nan   # don't count the DC component
    wrapim = fftpack.fftshift( absim )  # this is the 2D power spectrum
    ind = np.where( np.isfinite(wrapim) == False )
    xcen = ind[0][0]
    ycen = ind[1][0]

    tmp = radialProfile.azimuthalAverage(wrapim, center=[xcen,ycen], 
                                         ignoreNAN=True)
    pix = tmp[0]
    value = tmp[1]
    rms = tmp[2]
    npts = tmp[3]

    cut_d = 2.0 * platescale   # detector minimum angle in radians
    cut_t = wave / D           # telescope minimum angle in radians
    rat = cut_d / cut_t
    freq = pix / (0.5 * szx * rat)
    error = rms / np.sqrt(npts)

    # Ignore frequencies higher than the critical frequency
    keepind = np.where(freq <= 1)
    freq = freq[keepind]
    power = value[keepind]
    error = error[keepind]

    pspec_sources_2d = fftpack.fft2(sources * han) / (szx * szy)
    pspec_sources_2d = np.real(pspec_sources_2d * pspec_sources_2d.conjugate())
    pspec_sources_2d[0,0] = np.nan
    pspec_sources_2d = fftpack.fftshift( pspec_sources_2d )
    
    tmp = radialProfile.azimuthalAverage(pspec_sources_2d, center=[xcen, ycen],
                                         ignoreNAN=True)

    pspec_freq = tmp[0]
    pspec_sources = tmp[1]

    pspec_sources /= np.median(pspec_sources)
    pspec_sources = pspec_sources[keepind]

    return (freq, power, error, pspec_sources)
    

def hanning(xsize, ysize, invert=False):
    """
    Make a 2D hanning kernel from the seperable 1D hanning kernels. The
    default kernel peaks at 1 at the center and falls to zero at the edges.
    Use invert=True to make an inner mask that is 0 at the center and rises
    to 1 at the edges.
    
    """
    mask1D_x = np.hanning(xsize)
    mask1D_y = np.hanning(ysize)
    mask = np.outer(mask1D_x, mask1D_y)
    if invert:
        mask = -1.0 * mask + 1.0

    return mask

    

def mtffunc_keck(pp, nu=None, wave=None, F=None, D=None, Apix=None, 
                 pupil=None, tperf=None, sources=None, 
                 output='system', fjac=None):
    """
     NAME:
            MTFFUNC_KECK
    
     PURPOSE:
            Given the appropriate input parameters, returns the
            square of the 1-D Modulation Transfer Function (MTF)^2 of
            an optical system consisting of a 
            telescope + detector + atmosphere + Adaptive Optics (AO) system.
    
     CATEGORY:
            How should I know?
    
     CALLING SEQUENCE:
            RESULT=MTFFUNC_KECK(NU, PARAMS, [/PERF, /ATMOS_AO, $
                                SPDIST=SPDIST])
    
     INPUTS:
            NU:   The spatial frequencies at which the MTF will be
            computed, in normalized units (i.e. nu=1 at spatial frequency
            D/lambda for a circular aperture telescope), and need not be
            regularly gridded.
    
            PARAMS:   A structure of parameters defined as follows:
    
                      params.lambda  - wavelength in meters of the
                                       observation 
                      params.f       - effective focal length at detector
                                       in meters
                      params.D       - telescope pupil diameter in meters,
                                       defining the normalized spatial
                                       frequency NU which is in units of
                                       D/lambda; not necessarily 10
                                       meters for Keck! (see PROCEDURE
                                       below)   
                      params.A       - width of detector pixel in meters,
                                       27 microns for the NIRC2 narrow
                                       field.
                      params.pupil   - a string equal to the name of
                                       the pupil-stop of the NIRC2
                                       camera (see documentation for
                                       T_PERFECT_KECK for a list of
                                       available pupil-stop names) or for
                                       a circular pupil, the scalar
                                       floating point value of the
                                       pupil's central obscuration.  The
                                       data type of this parameter
                                       deterimines the form of the pupil
                                       MTF used (see PROCEDURE below).
                      params.L0      - the outer scale of turbluence, in
                                       meters, for a modified Kolmogorov
                                       spectrum
                      params.sigma   - the width of the AO system
                                       deformable mirror's (DM's)
                                       influence function as projected
                                       onto the pupil plane, in meters.
                      params.w       - scaling factor for the influence
                                       function's Fourier transform;
                                       mimics variable AO correction (see
                                       documentation for
                                       FT_INFL_FUNCTION) 
                      params.delta   - wavefront measurement error
                      params.cmult   - constant scaling factor of the output
                                       MTF
                      params.N       - additive constant to output MTF
                                       representing a noise floor
                      params.r0      - wavelength specific Fried
                                       parameter in meters (see PROCEDURE
                                       below). 
    
    
     KEYWORD PARAMETERS:
            SPDIST: Set this keyword to a vector the same size as nu, the
                    1-D power spectrum of the source distribution power
                    spectrum of an image for which you wish to fit an
                    MTF.  The output (MTF)^2 is multiplied by this vector
                    before output.
            /PERF:  If this keyword is set, RESULT is the square of the
                    diffraction limited pupil MTF. 
            /ATMOS_AO:   If this keyword is set, RESULT is the square of
                    the AO filtered atmospheric MTF.
            /PIX    Returns the MTF^2 of an ideal pixel.
            _EXTRA: Use this keyword to pass in a structure containing
                    constant parameters that need not be recomputed each
                    time MTFFUNC_KECK is called.  This is useful for
                    speeding up an iterative fitting procedure that uses
                    MTFFUNC_KECK.  The structure passed in via this
                    keyword must contain the following parameters/tags:
    
                     _EXTRA = {lambda:lambda, f:f, D:D, pupil:pupil, $
                               A:A, TPERF:TPERF, SPDIST:SPDIST}
    
                    where lambda, f, D, A, and pupil are definted exactly
                    as they are for input via PARAMS; TPERF is a vector
                    of size equal to NU and is the diffraction limited
                    pupil MTF (N.B., not the squared MTF) and is
                    equivalent to the square root of the output of
                    MTFFUNC_KECK with the /PERF keyword set; SPDIST is
                    the power spectrum of and image's source distribution
                    function, identical to SPDIST.  

                    If the keyword _EXTRA is set to this structure,
                    then PARAMS must not be a structure containing all of
                    the parameters defined in PARAMS above, but must be a
                    7 element vector defined as such:
    
                    PARAMS[0] = L0, PARAMS[1] = sigma, PARAMS[2] = w,
                     PARAMS[3] = delta, PARAMS[4] = cmult, PARAMS[5]= N,
                      PARAMS[6] = r0.
    
                    Setting the keyword SPDIST
                    overrides the source distribution power spectrum
                    passed in via the _EXTRA keyword.  If the _EXTRA
                    keyword is not set and a structure of parameters is
                    passed in via PARAMS, then SPDIST is the only way to
                    multiply the output of MTFFUNC_KECK by a vector
                    before returning.
                     
     OUTPUTS:
            RESULT:  The (MTF)^2, evaluated at the input spatial
            frequencies, NU.
    
     RESTRICTIONS:
            NU must be greater than zero.  
    
     PROCEDURE CALLS:
            T_ATMOS_AO(), T_PIX(), T_PERFECT_KECK(), T_PERFECT()
    
     PROCEDURE:
            If PARAMS.pupil is a string, it must specify the pupil stop
            name of the NIRC2 camera that was in place at the time of the
            observation.  See the documentation for NIRC2PUPIL for a list
            of acceptable pupil stop names.  In this case, the pupil MTF
            is numerically calculated as the autocorrelation function of
            the Keck pupil via the procedure T_PERFECT_KECK.
    
            If PARAMS.pupil is a floating point scalar, the pupil MTF is
            calculated analytically via the procedure T_PERFECT for a
            circular pupil with central obscuration PARAMS.pupil.  (See
            the documentation for T_PERFECT.)  This functionality is not
            intended for use with Keck AO data, but is included in the
            event this software is applied to data from other AO systems,
            such as the Lick 3-meter telescope.
            
            PARAMS.delta is untested, and for the time being should be
            left set to zero.
    
            In general PARAMS.D should not be set to the familiar 10
            meters, which is the effective diameter of the Keck pupil.
            Since this parameter defines the maximum spatial frequency
            D/lambda to which the telescope is sensitive, it should be
            equal to the diameter of the circle inscribing the Keck
            pupil.  This is because the lambda/D minimum angle changes
            depending on orientation in the image plane, and in certain
            orientations the diameter appropriate for this specification
            is D = 10.99 m.
    
            r0 is the wavelength specific Fried parameter.  Generally,
            the r0 specifying seeing conditions is quoted for a
            wavelength of 500 nm.  If one has reason to believe that the
            r0 for a set of observations is 20 cm, then the wavelength
            specific r0 is given as
                r0_lambda = r0 * (lambda/500 nm)^(6/5)
            and this is the r0 that should be specified in PARAMS.RO.
    
            The effective focal length at the detector, PARAMS.F, is
            related to the platescale of the detector by
                F = Apix / platescale
            where platescale is in radians / pixel.  If the platescale
            and the pixel size is accurately known, F should be
            calculated in this manner.
     
     EXAMPLE:
            Generate a model MTF for an H-band NIRC2 image at the .01
            arcsec/pixel platescale at normalized spatial fequencies in
            the image plane from 0 to 1:
            
            params={lambda:1.6e-6, F:557.0, D:10.99,  $
                    L0:30.0, sigma:0.56, w:1.5, delta:0.0, cmult:1.0, $
                    N:1e-4, r0:0.65, Apix:27e-6, pupil:'largehex'}
            nu = findgen(100)/99.
            tsys_squared = mtffunc_keck(nu,params) 
    
     MODIFICATION HISTORY:
            Written by:  Christopher D. Sheehy, January 2006.
    
    """
    if nu.min() < 0:
        print 'Input NU cannot be less than zero.'

    if wave != None:
        p = {'wave': wave, 'F': F, 'D': D, 'Apix': Apix, 'pupil': pupil}
        p.update( params_arr2dict(pp) )

        MTF_perf = tperf
        spdist_mult = sources
    else:
        p = pp
        spdist_mult = 1
        if type(p['pupil']) == str:
            MTF_perf = t_perfect_keck(nu, p, pupil=p['pupil'])
        else:
            MTF_perf = t_perfect(nu, p['pupil'])
            
    if sources != None:
        spdist_mult = sources
    else:
        spdist_mult = 1

    MTF_atmos_ao = t_atmos_ao(nu, p)
    MTF_pix = t_pix(nu, p)

    tsys = MTF_perf * MTF_atmos_ao * MTF_pix
    MTF_sys2 = (spdist_mult * p['cmult'] * tsys**2) + p['N']

    # We have several output options:
    all_mtfs = {}
    all_mtfs['perfect'] = MTF_perf**2
    all_mtfs['atmos_ao'] = MTF_atmos_ao**2
    all_mtfs['pixel'] = MTF_pix**2
    all_mtfs['system'] = MTF_sys2
    
    return all_mtfs[output]
    

def fitmtf_keck(nu, power, error, pspec_sources, 
                clip=None, startParams=None, relStep=0.2, quiet=False):
    """
    NAME:
         FITMTF_KECK
    
    PURPOSE:
         Uses MPFIT to perform a constrained Levenberg-Markwardt fit of
         a model MTF to data from a Keck adaptive optics image.  It is
         highly recommended that the user edit this procedure to suit
         his or her particular needs, i.e. changing the default initial
         guesses for the fit parameters and the step size to MPFIT.
    
         This procedure is meant as a guide to illustrate how to use
         MPFIT in conjunction with the AO MTF software.  Just because
         MPFIT returns best fit parameters does not mean that it has
         found an absolute minimum in Chi-squared.  As with most
         non-linear fit routines, some tinkering may be required to
         avoid local minima and to obtain accurate fits.  In other
         words, this procedure should not be applied blindly.
    
         In general, a good way to obtain accurate fits is to first
         perform the fits with decent starting guesses for the fit
         parameters and specifying a relatively large step size over
         which MPFIT calculates the numerical derivative of the fit
         function w.r.t. the fit parameters.  Once the best fit
         parameters are obtained from this iteration, the step
         size can be decreased and the best fit parameters from the
         first iteration should be used as the starting guesses for the
         parameters in the second iteration.  
    
         A good rule of thumb that seems to work in some (many?) cases
         is specifying the step size in the first iteration to be 20% of the
         parameters' respective values, and a step size of 2% in the
         second interation.  This can be accomplished by changing the
         values of PARINFO[*].relstep from 0.20 to 0.02 within the
         procedure or via the RELSTEP keyword defined at the main
         level. Different step sizes for different parameters can be
         specified by editing PARINFO[*].relstep within the procedure.
    
    CATEGORY:
         ???
    
    CALLING SEQUENCE:
         FITMTF_KECK, filename, params, perror, covar, chisq, $
                      [start=start, relstep=relstep, quiet=quiet]
    
    INPUTS:
         FILENAME: A scalar string specifying the filename of an IDL
                   save file containing the data to which the model MTF
                   is fit.  The command
                         restore, FILENAME
                   must restore the variables NU, POWER, ERROR, and
                   SPDIST, which are vectors of equal size.  These
                   variables are the outputs of the routine GETMTF.
    
    KEYWORD PARAMETERS:
         START:   Set this keyword equal to a structure containing
                  starting guesses for the fit parameters as defined in
                  the documentation for MTFFUNC_KECK.  If not set, they
                  the default values must be defined within the
                  procedure.  
    
         CLIP:    Set this keyword equal to a scalar value defining the
                  normalized spatial frequency (defined by NU) below
                  which the data restored from FILENAME will be ignored.
                  By default, this value is None (no clipping).  
                  Setting CLIP = 0.0 fits all of the data.  
                  However, this is generally not
                  recommended because imperfect sky subtraction of the
                  image from which NU and POWER were computed usually
                  contaminates the power spectrum at low spatial
                  frequencies. The recommended value is 0.02.
    
         RELSTEP: Defines the relative step size over which MPFIT
                  computes the numerical derivative of Chi-sqared
                  w.r.t. the fit parameters.  Sets the value
                  PARINFO[*].relstep, which is an input to MPFIT (see
                  the documentation for MPFIT).
      
         /QUIET : Set this keyword to supress the printed output of
                  MPFIT.  Generally, though, it is good practice to keep
                  tabs on what MPFIT is doing as it's proceding with the
                  fit.  
    
    OUTPUTS:
         PARAMS:  A structure of the best fit parameters as determined
                  by MPFIT; can be used as the input parameters to
                  MTFFUNC_KECK, MTF2PSF, MTF2EE, or any of
                  MTFFUNC_KECK's subsidiary routines.  
    
                  The parameters that are fit are L0, sigma, w,
                  delta, cmult, N, and r0.  The rest are only supplied
                  as information to MTFFUNC_KECK.  In addition, some of
                  these fit parameters may be held constant if they are
                  known or assumed (SIMGA and DELTA are held fixed by 
                  default when performing the fit).  Constant parameters
                  are specified by setting PARINFO[i].fixed to 1 within
                  the FITMTF_KECK procedure (see documentation for
                  MPFIT and MPFITFUN).
    
         PERROR:  A structure of formal error in the best fit
                  parameters, as determined by MPFIT.  Parameters that
                  are not included in the fit or are held fixed during
                  the fit return an error of 0.
    
         COVAR:   The parameter covariance matrix, of size N x N, where
                  N is the number of fit parameters supplied to MPFIT,
                  the values of which depend on the order of the vector
                  of input parameters supplied to MPFIT.  (See
                  documentation for MPFIT and order of FCNARGS as
                  defined within this procedure.)
    
         CHISQ:   The quantity (RESIDS/ERROR)^2, where 
                  RESIDS = POWER - BESTFIT, and POWER and ERROR are
                  the vectors restored from FILENAME (see above).
                  CHISQ is calculated ignoring data at NU < CLIP. 
    
         NITER:   Number of iterations performed by the fitting routine.
    
    
    
    PROCEDURE CALLS:
         MPFITFUN, MPFIT, T_PERFECT_KECK
    
    EXAMPLE:
         read in a fully reduced H-band NIRC2 image
         im = READFITS('myimage.fits')  
        
         calculate its power spectrum
         p = {lambda:1.65e-6, D:10.99, F:557.0, APIX:27e-6}
         GETMTF, im, p, nu, power, error, spdist
         SAVE, nu, power, error, spdist, filename='mtfdata.sav'
    
         fit an MTF to this data
         startp = {wave:1.65e-6, D:10.99, F:557.0, Apix:27e-6, $
                   pupil:'largehex', L0:30.0, sigma:0.56, $
                   w:1.3, delta:0.0, cmult:1.0, N:1e-5, r0:0.5}
         FITMTF_KECK, 'mtfdata.sav', bestfit_params, start=startp
    
         zero in on the best fit parameters by editing FITMTF_KECK to
         perform the fit using a smaller step size to MPFIT.  Change
         PARINFO[*].relstep from 0.20 to 0.02, recomplie FITMTF_KECK,
         and perform the fit again using the previous best fit
         parameters as the new starting parameters.
         FITMTF_KECK, 'mtfdata.sav', bestfit_params2, start=bestfit_params
    
         calculate the encircled energy for the PSF in image
         'myimage.fits' at a radius of 25 pixels (at the 0.01 arcsec/pix
         NIRC2 platescale). 
         MTF2EE, bestfit_params2, 25.0, EE
         print, EE
    
         compute the best fit power spectrum
         PSPEC = MTFFUNC_KECK(nu, bestfit_params2, spdist=spdist)
         plot, nu, power, /ylog, psym=4
         oplot, nu, pspec
    
         plot the best fit MTF
         T = sqrt(MTFFUNC_KECK(nu, bestfit_params2)
         plot, nu, T, /ylog
    
    MODIFICATION HISTORY:
         Written by Christopher D. Sheehy, January 2006.
         Added "niter" keyword, Nate McCrady, May 17, 2007.
    """

    if clip != None:
        idx = np.where(nu <= clip)[0]
        lo = idx.max() + 1
    else:
        lo = 0

    xx = nu[lo:]
    data = power[lo:]
    err = error[lo:]
    #err = power[lo:] * 0.01   # uniform 1 percent error for a test??
    spdist = pspec_sources[lo:]

    # Define starting guesses for parameters
    if startParams == None:
        wave = 1.65e-6       # wavelength in meters
        F = 557.0            # effective focal length in meters
        print 'Using effective focal length for the narrow camera.', F
        D = 10.99            # primary mirror diamter in meters
        pupil = 'largehex'   # NIRC2 pupil-stop
        Apix = 27e-6         # width of detector's pixel in meters
        L0 = 20.0            # outer scale of turbulence in meters
        sigma = 0.56         # IF width on primary in meters
        w = 1.3              # IF height
        delta = 0            # wavefront measurement error
        cmult = 10.0         # multiplicative constant
        N = 1e-5             # additive noise floor constant
        r0 = 0.5             # wavelength specific Fried parameter in meters
    else:
        wave = startParams['wave']
        F = startParams['F']
        D = startParams['D']
        pupil = startParams['pupil']
        Apix = startParams['Apix']
        L0 = startParams['L0']
        sigma = startParams['sigma']
        w = startParams['w']
        delta = startParams['delta']
        cmult = startParams['cmult']
        N = startParams['N']
        r0 = startParams['r0']
        
    # Declare the structure to pass parameters to mpfit
    parinfo = {'value': 0.0, 'fixed': 0, 'limited': [0, 0], 'limits': [0., 0.],
               'relstep': relStep, 'parname': ''}
    pinfo = [parinfo.copy() for i in range(7)]

    # additive constant for model MTF    
    pinfo[0]['parname'] = 'N'    
    pinfo[0]['value'] = N

    # Fried parameter in meters
    pinfo[1]['parname'] = 'r_0'   
    pinfo[1]['value'] = r0
    pinfo[1]['limits'] = [0.05, 2.0]

    # multiplicative constant for model MTF
    pinfo[2]['parname'] = 'cmult' 
    pinfo[2]['value'] = cmult
    pinfo[2]['limited'] = [1, 0]
    pinfo[2]['limits'] = [0, 0]

    # w of influence func's 1st gaussian
    pinfo[3]['parname'] = 'w'      
    pinfo[3]['value'] = w
    pinfo[3]['limited'] = [1, 1]
    pinfo[3]['limits'] = [0.0, 2.01]

    # outer turbulence scale, L_0, in meters
    pinfo[4]['parname'] = 'L0'
    pinfo[4]['value'] = L0   
    pinfo[4]['fixed'] = 0
    pinfo[4]['limited'] = [1, 1]
    pinfo[4]['limits'] = [0., 3500.]

    # sigma of influ. func's 1st gaussian (meters)
    pinfo[5]['parname'] = 'sigma'
    pinfo[5]['value'] = sigma
    pinfo[5]['fixed'] = 1

    # additive noise factor
    pinfo[6]['parname'] = 'delta'  
    pinfo[6]['value'] = delta
    pinfo[6]['fixed'] = 1
    pinfo[6]['limited'] = [1, 0]
    pinfo[6]['limits'] = [0, 0]

    startp = [pinfo[i]['value'] for i in range(len(pinfo))]
        
    # Pass in constant parameters using the keyword functargs
    if type(pupil) == str:
        # This is a keck NIRC2 pupil string
        tp = t_perfect_keck(xx, D, pupil=pupil)
    else:
        # Otherwise assume no specific telescope and pupil 
        # is just a secondary obscuration.
        tp = t_perfect(xx, pupil)

    fcnargs = {'nu': xx, 'obs': data, 'err': err,
               'wave': wave, 'F': F, 'D': D, 'Apix': Apix, 
               'pupil': pupil, 'tperf': tp, 'sources': spdist,
               }
    
    # Call the fitting routine MPFIT, passing it the function that returns
    # the MTF**2 at spatial frequency XX for given input parameters defined
    # in PARINFO and other information required to generate the MTF
    # supplied in FCNARGS.

    # changed 3/18/08 to fit the log of the data

    def residuals(params, nu=None, obs=None, err=None, 
                  wave=None, F=None, D=None, Apix=None, 
                  pupil=None, tperf=None, sources=None, 
                  fjac=None):
        fit = mtffunc_keck(params, nu=nu, wave=wave, F=F, D=D, Apix=Apix,
                           pupil=pupil, tperf=tperf, sources=sources,
                           output='system', fjac=fjac)

        # Check for appropriate error weighting. Otherwise, simply unweighted.
        res = (np.log(obs) - np.log(fit)) / np.log(err)
        #res = (obs - fit) / err
        
        param_dict = params_arr2dict(params)
        paramStr = ''
        for key, value in param_dict.items():
            paramStr += '%s=%.2g ' % (key, value)

        py.clf()
        py.semilogy(nu, obs, label='Observed')
        py.semilogy(nu, fit, label='Fit')
        py.legend()
        py.title(paramStr, fontsize=10)
        py.draw()

        return (0, res)

    print 'Start Fitting: ', time.ctime()
    fit = mpfit.mpfit(residuals, startp, nprint=1,
                      parinfo=pinfo, functkw=fcnargs, quiet=1)
    print 'Stop Fitting: ', time.ctime()

    params = {}
    params['wave'] = fcnargs['wave']
    params['F'] = fcnargs['F']
    params['D'] = fcnargs['D']
    params['Apix'] = fcnargs['Apix']
    params['pupil'] = fcnargs['pupil']
    params.update( params_arr2dict(fit.params) )

    perror = {}
    perror['wave'] = 0
    perror['F'] = 0
    perror['D'] = 0
    perror['Apix'] = 0
    perror['pupil'] = 0
    perror.update(params_arr2dict(fit.perror))
    
    output = DataHolder()
    output.obs_nu = nu
    output.obs_data = data
    output.obs_error = err
    output.obs_sources = spdist
    output.tperf = tp
    output.params = params
    output.perror = perror
    output.fit_params = fit.params
    output.fit_covar = fit.covar
    output.fit_stat = fit.fnorm

    return output

def params_arr2dict(param_array):
    param_dict = {}
    param_dict['N'] = param_array[0]
    param_dict['r0'] = param_array[1]
    param_dict['cmult'] = param_array[2]
    param_dict['w'] = param_array[3]
    param_dict['L0'] = param_array[4]
    param_dict['sigma'] = param_array[5]
    param_dict['delta'] = param_array[6]
    
    return param_dict
    
def params_dict2arr(param_dict):
    param_arr = []
    param_arr[0] = param_dict['N']
    param_arr[1] = param_dict['r0']
    param_arr[2] = param_dict['cmult']
    param_arr[3] = param_dict['w']
    param_arr[4] = param_dict['L0']
    param_arr[5] = param_dict['sigma']
    param_arr[6] = param_dict['delta']
    
    return param_arr
    

def t_perfect_keck(x, p, pupil=None):
    """
    NAME:
           T_PERFECT_KECK
    
    PURPOSE:
           Numerically computes a 1-D approximation to the 2-D
           diffraction limited MTF of the segmented Keck pupil,
           appropriate for the NIRC2 camera. 
    
    CATEGORY:
           ???
    
    CALLING SEQUENCE:
           RESULT = T_PERFECT_KECK(NU, D, [PUPIL=PUPIL])
    
    INPUTS:
           NU:  Normalized spatial frequency in the image plane, in
                units of D/lambda.
    
           D:   The diameter of the Keck pupil, in meters, to which the
                spatial frequencies are normalized, i.e. not necessarily
                the oft-quoted effective diameter of 10 meters.  
                If D is a scalar or a 1-element array, this value is
                used.  Otherwise D must be a structure with the tag "D"
                defined, as per the definition of PARAMS in
                MTFFUNC_KECK. 
    
    KEYWORD PARAMETERS:
           PUPIL:      Set this keyword to a string to select the
                       NIRC2 pupil-stop.  Available choices are
                          'OPEN'
                          'LARGEHEX'
                          'MEDIUMHEX'
                          'SMALLHEX'
                          'INCIRCLE'
                       If this keyword is not set, the default pupil is
                       'LARGEHEX'. (See documentation for NIRC2PUPIL.)
    
    OUTPUTS:
           RESULT:  The diffraction limited MTF (N.B. not the square of
           the MTF).
    
    PROCEDURE:
           1) Retrieves a pupil image from NIRC2PUPIL
           2) Computes the 2-D autocorrelation function of the pupil
           3) Radially bins and averages the 2-D autocorrelation
              function into a 1-D MTF.
           4) Interpoates the 1-D MTF onto the grid of normalized
              spatial frequencies, NU, defined by the user.
    
    PROCEDURE CALLS:
           NIRC2PUPIL(), CONVOLVE()
    
    MODIFICATION HISTORY:
           Written by Christopher D. Sheehy, January 2006.
           Commented by Nate McCrady, January 2008.
    """
    # Some image testing to catch unexpected inputs
    if x.min() < 0:
        print 'Input NU cannot be less than zeros.'
        return
    if type(p) == dict:
        D = p['D']
    else:
        D = p

    # Constants
    mperpix = 0.04
    npix = 550
    
    # Get the pupil image.
    pupim = nirc2pupil(npix=npix, du=mperpix, pmsname=pupil)  

    # Compute the 2D autocorr of the pupil image
    pupcor = autocorr2d(pupim)

    # normalize to max of 1.0
    pupcor /= pupcor.max()

    # identify the center of the autocorrelated pupil
    ind = pupcor.argmax()
    xcen = ind % npix
    ycen = ind / npix

    # Radially bin/average to produce a 1-D MTF
    tmp = radialProfile.azimuthalAverage(pupcor, center=[xcen,ycen], 
                                         ignoreNAN=True)
    pix = tmp[0]
    val = tmp[1]
    rms = tmp[2]
    npts = tmp[3]

    # Define cutoff frequency in pixel units
    maxpix = D / mperpix
    f = pix / maxpix
    # Radially binning doesn't give a value for r=0, which must be 1
    f = np.append([0], f)
    val = np.append([1], val)

    # The interpolation might return very small negative values that
    # by definition should be zero, so make them such
    idx = np.where(val < 0)
    val[idx] = 0

    interp = interpolate.splrep(f, val, k=1, s=0)
    newval = interpolate.splev(x, interp)

    return newval

def autocorr2d(d):
    """
    2D auto correlation.
    """
    npix = d.shape[0] * d.shape[1]
    d_fft = fftpack.fft2(d)

    tmp1 = d_fft * d_fft.conjugate()
    tmp2 = fftpack.ifft2(tmp1)
    tmp3 = np.real(tmp2)
    tmp4 = fftpack.fftshift(tmp3)

    return tmp4
    


def nirc2pupil(npix=256, du=None, pmsname='largehex', pmrangl=0.0):
    """
    NAME:
    	NIRC2PUPIL
    
    PURPOSE:
    	Calculate NIRC2 pupil image
    
    EXPLANATION:
    	Calculate pupil image for any pupil stop, pupil angle, and image
    	scale, for use by NIRC2PSF in determining theoretical PSF.
    
    CALLING SEQUENCE:
    	result = NIRC2PUPIL( [NPIX=, DU=, PMSNAME=, PMRANGL= ])
    
    INPUTS:
    	none.
    
    OUTPITS:
    	result = binary image of pupil
    
    OPTIONAL INPUT KEYWORDS:
    	NPIX = size of pupil image, in pixels
    
    	DU = platescale of pupil image, in m/pixel at the telescope primary
    
    	PMSNAME = pupil stop name, eg. 'largehex' (the default).
    
    	PMRANGL = pupil drive's angular position (for rotated pupil images).
    	  NOT TESTED.  There could be an offset and/or a sign flip needed!
    
    EXAMPLE:
    	pupil = NIRC2PUPIL(npix=512, du=0.05, PMSNAME='open')
    
    ERROR HANDLING:
    	none
    
    RESTRICTIONS:
    	none
    
    NOTES:
    	The dimentions are based on Keck KAON 253 and the NIRC2 pupil
    	  stop drawings.
    
    PROCEDURES USED:
    	none
    
    MODIFICATION HISTORY:
    	Original writen May 2004, A. Bouchez, W.M. Keck Observatory
    """
    if du == None:
        du = 2.124e-6 / (npix * 0.00995 / 206265.0)

    # 1. Define dimensions of pupil in inches based on engineering drawings.
    pmsstr = pmsname.strip().upper()

    pmsInfo = {}
    pmsInfo['OPEN'] =      np.array([0.4900, 0.4200, 0.3500, 0.2800, 0.0000, 0.0000], dtype=float)
    pmsInfo['LARGEHEX'] =  np.array([0.4790, 0.4090, 0.3390, 0.2690, 0.1170, 0.0020], dtype=float)
    pmsInfo['MEDIUMHEX'] = np.array([0.4710, 0.4010, 0.3310, 0.2610, 0.1250, 0.0030], dtype=float)
    pmsInfo['SMALLHEX'] =  np.array([0.4510, 0.3810, 0.3110, 0.2410, 0.1450, 0.0030], dtype=float)
    pmsInfo['INCIRCLE'] =  np.array([0.3920, 0.1325, 0.0030], dtype=float)
               
    d = pmsInfo[pmsstr]

    # m/inch derived in KAON 253
    pms_pscl = 0.0899

    pupil = np.zeros((npix, npix), dtype=bool)
    tmp = np.arange(npix)
    tmpy, tmpx = np.meshgrid(tmp, tmp)
    xypupil = np.array([tmpx.flatten(), tmpy.flatten()]).transpose()

    r = dist_circle(npix, center=(npix/2-0.5, npix/2-0.5))

    # 1. Create INCIRCLE pupil
    if pmsstr == 'INCIRCLE':
        w = np.where((r * du * pms_pscl < d[0]) & 
                     (r * du * pms_pscl > d[1]))
        if (len(w[0]) != 0):
            pupil[w] = True

        v = np.array([[-1*d[2], 0], 
                      [d[2], 0], 
                      [d[2], d[0]*1.1],
                      [-1*d[2], d[0]*1.1], 
                      [-1*d[2], 0]])
        ang = np.radians(60 * np.arange(6, dtype=float) + pmrangl)

        for ii in range(len(ang)):
            rmat = np.array([[-1*math.sin(ang[ii]), math.cos(ang[ii])],
                             [   math.cos(ang[ii]), math.sin(ang[ii])]])
            rv = npix/2 + (np.dot(v, rmat) / (du * pms_pscl))
            w = nxutils.points_inside_poly(xypupil, rv)
            w = w.reshape(pupil.shape)

            pupil[w] = False

    else:
        # 2. For others, compute vertices for one sextant (in mm)
        cosa = math.cos(math.radians(30.0))
        sina = math.sin(math.radians(30.0))
        s = (d[0] - d[1]) / cosa   # length of segment edge
        v0 = np.array([[d[5], d[4]/cosa - d[5]*sina],
                       [d[5], d[2]/cosa + d[5]*sina],
                       [s*cosa, d[2]/cosa + s*sina],
                       [2*s*cosa, d[2]/cosa],
                       [3*s*cosa, d[2]/cosa + s*sina],
                       [d[0]*sina, d[0]*cosa],
                       [d[4]*sina, d[4]*cosa],
                       [d[5], d[4]/cosa - d[5]*sina]])
        # mirror image across Y axis
        v1 = v0 * np.array([[-1, 1] for ii in range(8)])
    
        # Fill in pupil image (dimensions in pixels)
        ang = np.radians((60 * np.arange(6) + pmrangl))
        for i in range(6):
            rmat = np.array([[-1*math.sin(ang[i]), math.cos(ang[i])],
                             [   math.cos(ang[i]), math.sin(ang[i])]])
            rv0 = npix/2 + (np.dot(v0, rmat) / (du * pms_pscl))
            rv1 = npix/2 + (np.dot(v1, rmat) / (du * pms_pscl))

            rv0tmp = np.array([[rv0[i,1], rv0[i,0]] for i in range(len(rv0))])
            inpupil = nxutils.points_inside_poly(xypupil, rv0)
            inpupil2 = nxutils.points_inside_poly(xypupil, rv0tmp)
            inpupil = inpupil.reshape(pupil.shape)
            pupil[inpupil] = True

            inpupil = nxutils.points_inside_poly(xypupil, rv1)
            inpupil = inpupil.reshape(pupil.shape)
            pupil[inpupil] = True
        
        # Cut out circular secondary
        if pmsstr == 'OPEN':
            w = np.where(r * du < 1.30)
            pupil[w] = False

    return pupil

    
def dist_circle(size, center=None):
    """"
    NAME: 
         DIST_CIRCLE
    PURPOSE:      
         Form a square array where each value is its distance to a given center.
    EXPLANATION:
         Returns a square array in which the value of each element is its 
         distance to a specified center. Useful for circular aperture photometry.
    
    CALLING SEQUENCE:
         DIST_CIRCLE, IM, N, [ XCEN, YCEN,  /DOUBLE ]
    
    INPUTS:
         N = either  a scalar specifying the size of the N x N square output
                  array, or a 2 element vector specifying the size of the
                  N x M rectangular output array.
    
    OPTIONAL INPUTS:
         XCEN,YCEN = Scalars designating the X,Y pixel center.  These need
                  not be integers, and need not be located within the
                  output image.   If not supplied then the center of the output
                  image is used (XCEN = YCEN = (N-1)/2.).
    
    OUTPUTS:
          IM  - N by N (or M x N) floating array in which the value of each 
                  pixel is equal to its distance to XCEN,YCEN
    
    OPTIONAL INPUT KEYWORD:
          /DOUBLE - If this keyword is set and nonzero, the output array will
                  be of type DOUBLE rather than floating point.
    
    EXAMPLE:
          Total the flux in a circular aperture within 3' of a specified RA
          and DEC on an 512 x 512 image IM, with a header H.
    
                  IDL> adxy, H, RA, DEC, x, y       Convert RA and DEC to X,Y
          IDL> getrot, H, rot, cdelt        CDELT gives plate scale deg/pixel
          IDL> cdelt = cdelt*3600.          Convert to arc sec/pixel
          IDL> dist_circle, circle, 512, x, y  ;Create a distance circle image
          IDL> circle = circle*abs(cdelt[0])   ;Distances now given in arcseconds
          IDL> good = where(circle LT 180)  ;Within 3 arc minutes
          IDL> print,total( IM[good] )      Total pixel values within 3'
    
    RESTRICTIONS:
          The speed of DIST_CIRCLE decreases and the the demands on virtual
          increase as the square of the output dimensions.   Users should
          dimension the output array as small as possible, and re-use the
          array rather than re-calling DIST_CIRCLE
    
    MODIFICATION HISTORY:
          Adapted from DIST    W. Landsman            March 1991
          Allow a rectangular output array   W. Landsman     June 1994
          Converted to IDL V5.0   W. Landsman   September 1997
          Add /DOUBLE keyword, make XCEN,YCEN optional  W. Landsman Jun 1998
    """
    if type(size) == int:
        nx = size
        ny = size
    else:
        nx = size[0]
        ny = size[1]

    if center == None:
        xcen = (nx - 1) / 2.0
        ycen = (ny - 1) / 2.0
    else:
        xcen = center[0]
        ycen = center[1]

    x = np.arange(nx)
    y = np.arange(ny)
    
    yy, xx = np.meshgrid(y, x)
    xx -= xcen
    yy -= ycen

    r = np.hypot(xx, yy)

    return r
    

             
def t_perfect(x, ee):
    """
     NAME:
            T_PERFECT
    
     PURPOSE:
            Computes the analytic diffraction limited MTF of a
            circular pupil with a circular central obscuration.
    
     CATEGORY:
            What goes here?
    
     CALLING SEQUENCE:
            RESULT = T_PERFECT(NU, PUPIL)
    
     INPUTS:
            NU:  Normalized spatial frequency in the image plane, in
                 units of D/lambda where D is the pupil diameter and
                 lambda is the observing wavelength     need not be
                 regularly gridded
    
            PUPIL:  The pupil's central obscuration, defined as the
                 ratio of the obscuration's diameter to the pupil's
                 diameter.  If PUPIL is a scalar or a 1-element array,
                 this value is used.  Otherwise PUPIL must be a structure
                 with the tag "PUPIL" defined, as per the definition of
                 PARAMS in MTFFUNC_KECK. 
    
     OPTIONAL INPUTS:
            NONE
    
     KEYWORD PARAMETERS:
            NONE
    
     OUTPUTS:
            RESULT:  The diffraction limited pupil MTF (N.B. not the
            square of the MTF)
    
     OPTIONAL OUTPUTS:
            NONE
    
     MODIFICATION HISTORY:
            Written by Christopher D. Sheehy, January 2006.

    """
    if x.min() < 0:
        print 'Input NU cannot be less than zero.'
    
    if type(ee) == dict:
        e = ee['pupil']
    elif np.size(ee) == 8:
        e = ee[0]
    else:
        # Assume size = 1
        e = ee

    if e < 0 or e >= 1:
        print 'Central obscuration cannot be < 0 or >= 1'
        
    nx = len(x)
    ind0 = np.where(x <= 1)[0]
    A = np.zeros(nx, dtype=float)
    A[ind0] = np.arccos(x[ind0]) - x[ind0]*np.sqrt(1 - x[ind0]**2)

    ind0 = np.where(x <= e)[0]
    ind1 = np.where(x > e)[0]
    B = np.zeros(nx, dtype=float)
    if e != 0:
        tmp = x[ind0]/e
        B[ind0] = np.arccos(tmp) - tmp * np.sqrt(1 - tmp**2)
        B[ind0] *= e**2
    else:
        B[:] = 0.0
    B[ind1] = 0.0

    ind0 = np.where(x <= (1-e)/2.0)[0]
    ind1 = np.where(x >= (1+e)/2.0)[0]
    ind2 = np.where((x > (1-e)/2.0) & (x < (1+e)/2.0))[0]

    chi = np.zeros(nx, dtype=float)
    C = np.zeros(nx, dtype=float)

    tmp = np.arccos((1 + e**2 - 4*x[ind2]**2) / (2*e))
    chi[ind2] = tmp
    C[ind2] = -1 * math.pi * e**2
    C[ind2] += e * np.sin(tmp) + (tmp/2.0)*(1 + e**2)
    C[ind2] -= (1 - e**2) * np.arctan(((1+e) / (1-e)) * np.tan(tmp/2.))

    C[ind0] = -1 * math.pi * e**2
    C[ind1] = 0.0

    Tp = (2.0 / math.pi) * (A + B + C) / (1 - e**2)

    return Tp
    
def t_pix(x, p):
    """
     NAME:
            T_PIX
    
     PURPOSE:
            Analytically computes the MTF of an ideal square pixel,
            i.e. the detector MTF.
    
     CATEGORY:
            ???
    
     CALLING SEQUENCE:
            RESULT = T_PIX(NU, PARAMS)
    
     INPUTS:
            NU:  Normalized spatial frequency in the image plane, in
                 units of D/lambda.
    
            PARAMS:  Must be a structure containing the tags "lambda",
                 "D", "F", and "A" as defined in the documentation for
                 MTFFUNC_KECK.
    
     OPTIONAL INPUTS:
            NONE
    
     KEYWORD PARAMETERS:
            NONE
    
     OUTPUTS:
            The detector MTF (N.B. not the square of the MTF)
    
     PROCEDURE CALLS:
            SINC()
    
     MODIFICATION HISTORY:
            Written by Christopher Sheehy, January 2006.
    """
    if x.min() < 0:
        print 'Input NU cannot be less than zero.'
    
    d = p['D']
    wave = p['wave']
    a = p['Apix']
    f = p['F']

    # Effective focal ratio
    fratio = f / d
    
    if d <= 0 or wave <= 0 or a <= 0 or f <=0:
        print 'Input D, wave, Apix, and F must all be > 0.'

    delta = wave * fratio / a

    f = np.sinc(x / delta)

    return f


def t_atmos_ao(x, p):
    """
     NAME:
            T_ATMOS_AO
    
     PURPOSE:
            Computes the AO filtered atmospheric MTF assuming a modified
            Kolmogorov atmospheric phase error power spectrum with a
            finite outer scale of turbulence.
    
     CATEGORY:
            ????
    
     CALLING SEQUENCE:
            RESULT = T_ATMOS_AO(NU, PARAMS)
    
     INPUTS:
            NU:  Normalized spatial frequency in the image plane, in
                 units of D/lambda.
            PARAMS:  A structure of parameters, defined in the
                 documentation for MTFFUNC_KECK
    
     OPTIONAL INPUTS:
            NONE
    
     KEYWORD PARAMETERS:
            NONE
    
     OUTPUTS:
            RESULT:  The AO+atmosphere MTF     the same size as NU.
    
     RESTRICTIONS:
            NU must be greater than zero.
    
     PROCEDURE CALLS:
            STRUC_FUNC()
    
     PROCEDURE:
            Computes the structure function for the AO filtered
            Kolmogorov power spectrum using STRUC_FUNC and exponentiates
            to yield the atmosphere + AO MTF.
    
     MODIFICATION HISTORY:
            Written by Christopher D. Sheehy, January 2006.
    """
    if x.min() < 0:
        print 'Input NU cannot be less than zero.'
    if type(p) != dict:
        print 'Input PARAMS must be a dictionary.'

    wave = p['wave']
    flength = p['F']
    diam = p['D']
    r = diam * x

    # Comput the structure function. Can be time consuming.
    D = structure_function(r, p)

    mtf = np.exp(-0.5 * D)

    t_at_ao = mtf

    return t_at_ao

def phi_atmos_ao(x, p, unmod=False, tat=False):
    """
     NAME:
            PHI_ATMOS_AO
    
     PURPOSE:
            Calculates the AO filtered atmospheric power spectrum for
            spatial frequencies in the pupil plane, Phi_AO = Phi*(1-H)^2
            where Phi is the unfiltered atmospheric power spectrum and H
            is the Fourier transform of the deformable mirror's (DM's)
            influence function.  The default behavior is to use modified
            Kolmogorov power spectrum with finite outer scale and the
            influence function for the Keck AO system's DM.
    
     CATEGORY:
            ???
    
     CALLING SEQUENCE:
            Result = PHI_ATMOS_AO(KAPPA, PARAMS, [unmod=unmod, tat=tat])
    
     INPUTS:
            KAPPA:  A vector of spatial frequencies in the pupil plane,
                    in m^-1
            PARAMS: A structure of parameters as defined in the
                    documentation for MTFFUNC_KECK 
    
     KEYWORD PARAMETERS:
            /UNMOD: Set this keyword to use an unmodified Kolmogorov
                    atmospheric power spectrum, i.e. with an infinite
                    outer scale.
            /TAT:   Set this keyword to use a modified Tatarski power
                    spectrum, i.e. one with both a finite outer scale
                    and finite inner scale (WARNING: UNTESTED).  If this
                    keyword is invoked, the PARAMS structure must contain
                    an extra tag, "Inner:", that is the inner scale of
                    turbulence in meters.
    
     OUTPUTS:
            Result: The AO filtered atmospheric power spectrum, of
                    evaluated at and the same size as KAPPA.
    
     RESTRICTIONS:
            If the /UNMOD keyword is set, an input KAPPA of zero will
            cause a divide by 0 error, and RESULT will be Infinity,
            because the power at zero spatial frequency for an infinite
            outer scale is infinity.
    
            Works for the Keck DM's influence function.  If the user
            wishes to use a different influence function, replace the
            call to FT_INFL_FUNC within the routine with another
            function that returns the Fourier transform of the
            appropriate influence function.  This function should take as
            inputs KAPPA and PARAMS (the user may add other tags to the
            PARAMS structure as necessary) and return the FT of the
            influence function as projected onto the pupil plane,
            evaluated at the inpute KAPPA.
    
     PROCEDURE CALLS:
            PHI__ATMOS(), FT_INFL_FUNC()
    
     MODIFICATION HISTORY:
            Written by Christopher Sheehy, January 2006.
    """
    
    delta = p['delta']
    
    H = ft_infl_func(x, p)
    
    # Unmodified Kolmogorov power spectrum
    phi_at = phi_atmos(x, p, unmod=unmod, tat=tat)

    # AO filter power spectrum, H is already multiplied by p['w']
    phi_at_ao = ((1.0 - H)**2) * phi_at + delta * H**2

    return phi_at_ao


def phi_atmos(x, p, tat=False, unmod=False):
    """
     NAME:
            PHI_ATMOS
    
     PURPOSE:
            Calculates a modified Kolmogorov atmospheric phase error
            power spectrum with finite outer scale of turbulence.
    
     CATEGORY:
            ???
    
     CALLING SEQUENCE:
            RESULT = PHI_ATMOS(KAPPA, PARAMS, [unmod=unmod, tat=tat])
    
     INPUTS:
            KAPPA:  A vector of spatial frequencies in the pupil plane,
                    in m^-1, at which the power spectrum is evaluated
                    (need not be regularly gridded).
            PARAMS:  A structure of parameters as definted in the
                    documentation for MTFFUNC_KECK
    
     KEYWORD PARAMETERS:
            /UNMOD:  Set this keyword to return an unmodified Kolmogorov
                     power spectrum, i.e. one with an infinite outer
                     scale. 
            /TAT:    Set this keyword to return a modified Tatarski power
                     spectrum, i.e. one with both a finite outer scale
                     and finite inner scale (WARNING: UNTESTED).  If this
                     keyword is invoked, the PARAMS structure must contain
                     an extra tag, "Inner:", that is the inner scale of
                     turbulence in meters.
    
     OUTPUTS:
            RESULT:  The Kolmogorov atmospheric power spectrum, of same
                     size as input KAPPA.
    
     RESTRICTIONS:
            If the /UNMOD keyword is set, an input KAPPA of zero will
            cause a divide by 0 error, and RESULT will be Infinity,
            because the power at zero spatial frequency for an infinite
            outer scale is infinity.
    
     PROCEDURE:
            1) The Kolmogorov spectrum with inifinte outer scale is
               calculated.  If the /UNMOD keyword is set, this is
               immediately returned.
            2) The Kolmogorov spectrum with finite outer scale is
               calculated.  Since the power at spatial scales much less
               than the outer scale should be left unaffected, the
               spectrum is normalized to have the same power at a very
               high spatial frequency, 100 m^-1.
               
     MODIFICATION HISTORY:
            Written by Christopher Sheehy, January 2006
    """
    L = p['L0']
    r0 = p['r0']
    
    if unmod:
        phi_unmod = (0.0229 / (r0**(5.0/3.0))) * x**(-11.0/3.0)
        return phi_unmod

    if tat:
        l0 = p['inner']
        # UNTESTED! pretty sure this should be normalized in the 
        # same way as below
        phi = np.exp(-(x * l0 / (2*math.pi))**2) / ((1 + (x*L)**2)**(11.0/6.0))
        return phi
    
    phi = (0.0229 / (r0**(5.0/3.0))) * (1 + (x*L)**2)**(-11.0/6.0)
    a = ((0.0229 / (r0**(5.0/3.0))) * 100.0**(-11.0/3.0)) 
    a /= ((0.0229 / (r0**(5.0/3.0))) * (1 + (100.0*L)**2)**(-11.0/6.0))

    return a * phi

    
def ft_infl_func(x, p):
    """
    +
     NAME:
            FT_INFL_FUNC
    
     PURPOSE:
            Evaltuates the Fourier transform of the Keck AO system's
            deformable mirror at given input spatial frequency in the
            pupil plane.  Uses the functional form for the influence
            function given by van Dam, et al., 2004, Applied Optics, 43,
            5452.
    
     CATEGORY:
            ???
    
     CALLING SEQUENCE:
            Result = FT_INFL_FUNC(KAPPA, PARAMS)
    
     INPUTS:
            KAPPA:  Spatial frequency in the pupil plane of the telescope
                    in m^-1.
            PARAMS: A structure of parameters as defined in the
                    documentation for MTFFUNC_KECK.  Only SIGMA and W
                    need be defined. 
    
     OUTPUTS:
            Result:  The FT of the influence function, evaluated at and
                     of the same size as KAPPA.  The output is multiplied
                     by W before output.  W=2 gives perfect AO correction
                     at zero spatial frequency.
    
     PROCEDURE:
            The FT of the influence function has a set functional form,
            the difference of two Gaussians, and depends only on the
            parameter SIGMA (as defined in PARAMS) that is on the order
            of the separation between DM actuators.
    
            The result is multiplied by a constant scaling factor w, defined in
            PARAMS, before being output.
    
     MODIFICATION HISTORY:
            Written by Christopher Sheehy
    """
    inputIterable = hasattr(x, '__iter__')
    if not inputIterable:
        x = np.array([x], dtype=float)

    sig1 = float(p['sigma'])
    w1 = p['w']
    a = -0.5 * (x * sig1 * math.pi**2)**2
    b = -2.0 * (x * sig1 * math.pi   )**2
    
    nx = len(x)
    
    f1 = np.zeros(nx, dtype=float)
    f2 = np.zeros(nx, dtype=float)
    
    ind1 = np.where(a >= -40.)
    ind2 = np.where(b >= -40.)

    f1[ind1] = np.exp( a[ind1] ) * -0.5
    f2[ind2] = np.exp( b[ind2] )

    H = w1 * (f1 + f2)

    if not inputIterable:
        H = H[0]

    return H
                 

def structure_function(rr, pp, phi_call=phi_atmos_ao):
    """"
     NAME:
            STRUC_FUNC
    
     PURPOSE:
            Calculates the structure function, D(r), where r is the
            distance between two points in the pupil plane of the
            telescope in meter.  Default behavior is to calculate D(r)
            for an AO corrected Kolmogorov atmospheric power spectrum
            with finite outer scale.  However, the user can define any
            arbitrary atmospheric power spectra for which to calculate
            D(r).
    
     CATEGORY:
            ???
    
     CALLING SEQUENCE:
            RESULT = STRUC_FUNC(R, PARAMS, [phi_call=phi_call])
    
     INPUTS:
            R:  A vector of radii in the pupil plane, in meters
    
            PARAMS:  A structure of parameters as defined in the
            documentation for MTFFUNC_KECK
    
     OPTIONAL INPUTS:
            NONE
    
     KEYWORD PARAMETERS:
            PHI_CALL:  The default behavior of STRUC_FUNC, i.e. not
                       defning PHI_CALL, is to calculate the structure 
                       function assuming a modified Kolmogorov power
                       spectrum with a finite outer scale and corrected
                       by adaptive optics.  If the user wishes to find
                       D(r) for a different power spectrum, PHI_CALL
                       should be a string that, as executed, calls an IDL
                       function that returns the power spectrum for 
                       input spatial frequencies KAPPA (as defined in the
                       documentation for PHI_ATMOS/PHI_ATMOS_AO).
                       This function must be structured as follows:
    
                           FUNCTION MYFUNCT, KAPPA, PARAMS, KEYWORDS=...
                             (compute the atmospheric power spectrum PHI
                              at given X for input parameters PARAMS)
                              RETURN, PHI
                           END
    
                       The PARAMS input to the function is the same
                       structure as the PARAMS input into STRUC_FUNC.
    
                       PHI_CALL MUST HAVE 'k' AS THE SPATIAL FREQUENCY
                       VARIABLE NAME 'k' AND 'p' AS THE PARAMETER
                       STRUCTURE VARAIBLE NAME!
                       
                       Example: The function PHI_ATMOS returns the
                       uncorrected atmospheric power spectrum, and would
                       be called as such:
    
                           IDL> x=findgen(10)
                           IDL> phi=PHI_ATMOS(x,params)
    
                       where params has already been defined and is a
                       structure containing information for PHI_ATMOS to
                       be able to calculate the power spectrum.  In the
                       call to STRUC_FUNC, setting the keyword
    
                           PHI_CALL = 'PHI_ATMOS(k,p)'
    
                       forces STRUC_FUNC to use this power spectrum in
                       calculating D(r).  
    
                       PHI_ATMOS also accepts keywords     for instance
    
                           IDL> phi=PHI_ATMOS(x,params,/unmod)
    
                       returns an uncorrected power spectrum with an
                       infinite outer scale.  In this case, one would set
                       the keyword
    
                           PHI_CALL = 'PHI_ATMOS(k,p,/unmod)'
    
                       Leaving the keyword PHI_CALL undefined is
                       equivalent to setting
    
                           PHI_CALL = 'PHI_ATMOS_AO(k,p)
    
     OUTPUTS:
            RESULT:  A vector of same size as R, the structure function
            evaluated at R
    
     PROCEDURE CALLS:
            PHI_ATMOS_AO(), D_R(), QPINT1D()
    
     PROCEDURE:
            Utilizes the adaptive 1-d integration routine QPINT1D,
            written by Craig Markwardt and available from
            http://cow.physics.wisc.edu/~craigm/idl/idl.html
    
     MODIFICATION HISTORY:
            Written by Christopher Sheehy, January 2006.
    -
    """
    nr = len(rr)
    d_of_r = np.zeros(nr, dtype=float)

    # Make function is valid
    if not hasattr(phi_call, '__call__'):
        print 'Invalid phi_call in structure_function()'

    # Numerically compute the integral at each r necessary to construct
    # D(r). The integral's limits are actually 0 to infinity, but
    # integration 0 to 10 makes virtually no difference and is much faster.
    def d_r(x, r, p, command):
        phi = command(x, p)

        # R must be a scalar, use the J bessel function of zeroth order
        integrand = phi * (1 - special.jn(0, 2.0 * math.pi * x * r)) * x
        
        return integrand
        
    for ii in range(nr):
        args = (rr[ii], pp, phi_call)
        tmp = integrate.romberg(d_r, 0.0, 10.0, args, vec_func=True)
        d_of_r[ii] = tmp

    return 4.0 * math.pi * d_of_r


def mtf2psf(par, sz, perfect=False):
    """"
     NAME:
            MTF2PSF
    
     PURPOSE:
            Computes the adaptive optics PSF for the modulation
            transfer function (MTF) defined by a set of user supplied
            input parameters.  The PSF is circularly symmetric,
            i.e. azimuthally averaged.
    
     CATEGORY:
            ???
    
     CALLING SEQUENCE:
            MTF2PSF, Params, Size, Psf2D, PSF1D
    
     INPUTS:
            Params:   A structure of parameters defined as follows, and
                      which may contain more tags than listed here:
    
                      params.lambda  - observing wavelength in meters
                      params.f       - effective focal length at detector
                                       in meters
                      params.D -       telsscope pupil diameter in meters,
                                       i.e. the D/lambda at which normalized
                                       spatial frequency nu=1
                      params.Apix      - width of detector pixel in meters
                      params.pupil   - a string equal to the name of
                                       the pupil-stop of the NIRC2
                                       camera (see documentation for
                                       T_PERFECT_KECK for a list of
                                       available pupil-stop names) or for
                                       a circular pupil, the scalar
                                       floating point value of the
                                       pupil's central obscuration.
                      params.L0      - the outer scale of turbluence, in
                                       meters, for a modified Kolmogorov
                                       spectrum
                      params.sigma   - the width of the AO system
                                       deformable mirror's (DM's)
                                       influence function as projected
                                       onto the pupil plane, in meters.
                      params.w       - scaling factor for the influence
                                       function's Fourier transform    
                                       mimics variable AO correction
                      params.delta   - wavefront measurement error
                      params.cmult   - constant scaling factor of the output
                                       MTF
                      params.N       - additive constant to output MTF
                                       representing a noise floor
                      params.r0      - wavelength specific Fried
                                       parameter in meters
    
            Size:     A scalar, the radius of the output PSF in
                      arcseconds.  For instance, is Size = 1.0 and the
                      platescale at the detector is .01 arcsec/pixel, the
                      output PSF is a 200 x 200 element array.
    
     OPTIONAL INPUTS:
            NONE
    
     KEYWORD PARAMETERS:
            /PERF:   If this keyword is set, the PSF is calculated from
                     the diffraction limited pupil MTF.
    
     OUTPUTS:
            Psf2D:   A 2-dimensional array, the azimuthally averaged AO
                     PSF. 
    
     OPTIONAL OUTPUTS:
            Psf1D:   A 1-dimensional array, the 1-D PSF whose value at
                     each element corresponds to the value of the PSF at
                     the corresponding pixel.
    
     PROCEDURE CALLS:
            MTFFUNC_KECK(), INT_TABULATED()
    
     PROCEDURE:
            For the given input parameters Params, the 1-D MTF is
            computed via the procedure MTFFUNC_KECK.  The 1-D PSF is 
            then given by the integral
                      1/
              PSF(w) = | MTF(nu) * nu * BesselJ_0(2*PI*w*nu)* d nu,
                      0/
            where w is dimensionless angular distance from the PSF center
            in units of (lambda/D), and nu is normalized spatial
            frequency in the image plane.  The 1-D PSF is computed,
            5x oversampled w.r.t. the detector platescale.  (The
            platescale is determined from the input parameters, 
            Apix / f.)  From this oversampled 1-D PSF, the circularly
            symmetric 2-D PSF is constructed.
    
     EXAMPLE:
            Compute H-band NIRC2 PSF for r0 = 15 cm:
    
            r0 goes as (lambda)^(6/5), so the K-band r0 in meters is
            0.15 * (2.2 / 0.5)^(6./5.), since r0 = 15 cm is defined for a
            wavelength of 500 nm.
    
            Set up the input parameters:
    
            p    =   {wave:2.2e-6, F:557.0, D:10.99, $
                      Apix:27e-6, pupil:'largehex', L0:30.0, $
                      sigma:0.56, w:1.5, delta:0.0, cmult:1.0, N:1e-5, $
                      r0:0.888}        
            sz = 1.0      return a PSF with a radius of 1 arcsecond
            MTF2PSF, p, sz, psf2, psf1
            tvscl, psf2
            plot, psf1
    
            nu = findgen(100)/99.
            T = sqrt(MTFFUNC_KECK(nu, p))     compute the PSF's corresponding MTF
            plot, nu, T
    
                compute the PSF for identical seeing conditions but with no
                AO correction
            p.w = 0        no AO correction 
            MTF2PSF, p, sz, psf2_noAO, psf1_noAO  
    
                compute the diffraction limited PSF
            MTF2PSF, p, sz, psf2_perf, psf1_perf, /perf
    
            There is a subtle difference between setting the /perf
            keyword to MTF2PSF and setting r0 to some extremely large
            and unphysical value to mimic diffraction limited seeing.
            The former computes the PSF from the pupil MTF, while the
            latter uses the product of the puipl MTF, the atmospheric/AO
            MTF (which is essentially unity) and the detector MTF.  The
            detector MTF is a broad sinc function, and its resulting
            effect on the PSF is small.  In other words, setting the
            /perf keyword fails to take into account the effect of the
            detector on the PSF.       
    
     MODIFICATION HISTORY:
            Written by Christopher Sheehy, January 2006.
    """
    p = par.copy()

    wave = p['wave']
    D = p['D']
    flen = p['F']
    Apix = p['Apix']

    platescale = (Apix / flen) * 206265.0  # detector scale in arcsec
    xsamp = 5  # oversample PSF by a factor of 5
    maxpix = int((sz / platescale) * math.sqrt(2)) + 5.0

    nw = maxpix * xsamp 
    pix = np.arange(nw, dtype=float)
    pix *= maxpix / nw
    alpha = pix * platescale
    w = alpha / ((wave / D) * 206265.0)
    nf = 1000.0
    f = np.arange(nf, dtype=float)
    f /= (nf-1)
    pp = p
    pp['N'] = 0
    pp['cmult'] = 1
    
    if perfect:
        T = np.sqrt( mtffunc_keck(pp, nu=f, output='perfect') )
    else:
        T = np.sqrt( mtffunc_keck(pp, nu=f, output='system') )

    psf = np.zeros(nw, dtype=float)

    for i in range(nw):
        func = T * special.jn(0, 2.0 * math.pi * w[i] * f) * f
        integral = integrate.simps(func, x=f)
        psf[i] = integral

    pix = (w * (wave/D) * 206265.0) / platescale
    szpix = round(sz / platescale) * 2

    xx, yy = np.mgrid[0:szpix,0:szpix]
    a = np.zeros((szpix, szpix), dtype=int)
    cent = round((szpix - 1.0) / 2.0)
    r = np.hypot(xx - cent, yy - cent)

    rr = r.flatten()
    
    psfInterp = interpolate.splrep(pix, psf)
    psf2d_0 = interpolate.splev(rr, psfInterp)
    
    psf2d = psf2d_0.reshape((szpix, szpix))
    psf1d = psf[np.arange(sz/platescale, dtype=int)*xsamp]

    return psf2d, psf1d

def mtf2ee(par, pixx, perfect=False):
    """"
     NAME:
            MTF2EE
    
     PURPOSE:
            Calculates the AO PSF's encircled energy curve from the AO
            corrected modulation transfer function.
    
     CATEGORY:
            ???
    
     CALLING SEQUENCE:
            MTF2EE, Params, Pix, EE
    
     INPUTS:
            Params:  A structure of parameters, defined as in the
                     documentation for MTFFUNC_KECK.
            Pix:     A vector of radii, in pixels, at which the encircled
                     energy curve of growth is calculated.  The pixel
                     scale in pixels/arcsecond is calculated as
                     (Params.APIX / Params.F)*206265.
    
     OPTIONAL INPUTS:
            NONE
    
     KEYWORD PARAMETERS:
            /Perf:  If this keyword is set, the returned curve of growth
                    is for the diffraction limited PSF as calculated from
                    the pupil MTF.
    
     OUTPUTS:
            EE:  A vector of size equal to Pix     the curve of growth
                 evaluated at input Pix
    
     OPTIONAL OUTPUTS:
            NONE
    
     PROCEDURE CALLS:
            MTFFUNC_KECK(), INT_TABULATED()
    
     PROCEDURE:
            For the given input parameters Params, the 1-D MTF is
            computed via the procedure MTFFUNC_KECK.  The encirled energy
            curve of growth is then given by the integral
                              1/
              EE(w) = (2*Pi*w) | MTF(nu) * BesselJ_1(2*PI*w*nu)* d nu,
                              0/
            where w is dimensionless angular distance from the PSF center
            in units of (lambda/D), and nu is normalized spatial
            frequency in the image plane.  
    
     MODIFICATION HISTORY:
            Written by Christopher D. Sheehy, January 2006.
    """

    p = par.copy()    # prevent input params from begin altered at main level
    wave = p['wave']
    D = p['D']
    flen = p['F']
    apix = p['Apix']
    platescale = (apix / flen)  # detector platescale in radians/pixel
    nw = len(pixx)
    w = (pixx * platescale) / (wave / D)
    
    # set the frequency resolution of the MTF over which to integrate
    npts = 500  
    vn = np.arange(npts, dtype=float)
    vn /= npts - 1.0

    p['N'] = 0.0
    p['cmult'] = 1.0
    
    if perfect:
        T = np.sqrt( mtffunc_keck(p, nu=vn, output='perfect') )
    else:
        T = np.sqrt( mtffunc_keck(p, nu=vn, output='system') )

    ee = np.zeros(nw, dtype=float)

    for i in range(nw):
        func = T * special.jn(1, 2.0 * math.pi * w[i] * vn)
        integral = integrate.simps(func, x=vn)
        ee[i] = 2 * math.pi * w[i] * integral

    return ee



def strehl(par):
    """
     NAME:
            STREHL
    
     PURPOSE:
            Computes the Strehl ratio of an AO PSF from its corresponding
            modulation transfer function (MTF).
    
     CATEGORY:
            ???
    
     CALLING SEQUENCE:
            sr = STREHL(params)
    
     INPUTS:
            Params:  A structure of parameters used to compute the AO
                     MTF, as defined in the documentation for
                     MTFFUNC_KECK. 
    
     KEYWORD PARAMETERS:
            NONE
    
     OUTPUTS:
            The scalar strehl ratio for the given input parameters
    
     PROCEDURE CALLS:
            MTFFUNC_KECK(), INT_TABULATED()
    
     PROCEDURE:
            For the given input parameters, the MTF, T,  is calculated
            via MTFFUNC_KECK() along with the diffraction limited MTF,
            Tperf.  The strehl ratio, defined as the ratio of the height
            of the observed PSF to the diffraction limited PSF is
            calculated as (see documentation for MTF2PSF)
    
                   1/
                    | T(nu)*nu d nu
                   0/
            Sr = -------------------- .
                 1/
                  | Tperf(nu)*nu d nu
                 0/
    
     MODIFICATION HISTORY:
            Written by Christopher D. Sheehy, January 2006.
    """            

    npts = 500.0
    vn = np.arange(npts, dtype=float)
    vn /= (npts - 1.0)
    p = par.copy()
    p['cmult'] = 1.0
    p['N'] = 0.0

    # modified for fitting in log space
    T = np.sqrt( mtffunc_keck(p, nu=vn) )
    Tp = np.sqrt( mtffunc_keck(p, nu=vn, output='perfect') )

    sr = integrate.simps(T*vn, x=vn) / integrate.simps(Tp*vn, x=vn)

    return sr


class DataHolder(object):
    def __init__(self):
        return
