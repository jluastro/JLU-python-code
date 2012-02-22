import numpy as np
import math
import pdb

def mmm(sky_vector, highbad=None, readnoise=None, integer=False,
        debug=False, silent=False):
    """
    Estimate the sky background in a stellar contaminated field.

    MMM assumes that contaminated sky pixel values overwhelmingly display 
    POSITIVE departures from the true value.  Adapted from DAOPHOT 
    routine of the same name.
    
     CALLING SEQUENCE:
           MMM, sky, [ skymod, sigma, skew, HIGHBAD = , READNOISE=, /DEBUG, 
                      NSKY=, /INTEGER,/SILENT]
    
     INPUTS:
           SKY - Array or Vector containing sky values.  This version of
                   MMM does not require SKY to be sorted beforehand.  SKY
                   is unaltered by this program.
    
     OPTIONAL OUTPUTS:
           skymod - Scalar giving estimated mode of the sky values
           SIGMA -  Scalar giving standard deviation of the peak in the sky
                   histogram.  If for some reason it is impossible to derive
                   skymod, then SIGMA = -1.0
           SKEW -   Scalar giving skewness of the peak in the sky histogram
    
                   If no output variables are supplied or if /DEBUG is set
                   then the values of skymod, SIGMA and SKEW will be printed.
    
     OPTIONAL KEYWORD INPUTS:
           HIGHBAD - scalar value of the (lowest) "bad" pixel level (e.g. cosmic 
                    rays or saturated pixels) If not supplied, then there is 
                    assumed to be no high bad pixels.
           READNOISE - Scalar giving the read noise (or minimum noise for any 
                    pixel).     Normally, MMM determines the (robust) median by 
                    averaging the central 20% of the sky values.     In some cases
                    where the noise is low, and pixel values are quantized a
                    larger fraction may be needed.    By supplying the optional
                    read noise parameter, MMM is better able to adjust the
                    fraction of pixels used to determine the median.                
           /INTEGER - Set this keyword if the  input SKY vector only contains
                    discrete integer values.    This keyword is only needed if the
                    SKY vector is of type float or double precision, but contains 
                    only discrete integer values.     (Prior to July 2004, the
                    equivalent of /INTEGER was set for all data types)
           /DEBUG - If this keyword is set and non-zero, then additional 
                   information is displayed at the terminal.
           /SILENT - If set, then error messages will be suppressed when MMM
                    cannot compute a background.    Sigma will still be set to -1
     OPTIONAL OUTPUT KEYWORD:
          NSKY - Integer scalar giving the number of pixels actually used for the
                 sky computation (after outliers have been removed).
     NOTES:
           (1) Program assumes that low "bad" pixels (e.g. bad CCD columns) have
           already been deleted from the SKY vector.
           (2) MMM was updated in June 2004 to better match more recent versions
           of DAOPHOT.
           (3) Does not work well in the limit of low Poisson integer counts
           (4) MMM may fail for strongly skewed distributions.
     METHOD:
           The algorithm used by MMM consists of roughly two parts:
           (1) The average and sigma of the sky pixels is computed.   These values
           are used to eliminate outliers, i.e. values with a low probability
           given a Gaussian with specified average and sigma.   The average
           and sigma are then recomputed and the process repeated up to 20
           iterations:
           (2) The amount of contamination by stars is estimated by comparing the 
           mean and median of the remaining sky pixels.   If the mean is larger
           than the median then the true sky value is estimated by
           3*median - 2*mean
             
     REVISION HISTORY:
           Adapted from Python to IDL -- J. R. Lu 2010-05-30
    """
    
    # Set up some parameters
    mxiter = 30   # Maximum number of iterations allowed
    minsky = 20   # Minimum number of legal sky elements
    nsky = sky_vector.size   # Number of input sky elements

    if nsky < minsky:
        if not silent:
            print 'ERROR Input vector must contain at least %d elements' % \
                minsky
        return

    nlast = nsky - 1    # Subscript of last pixel in the sky array

    if debug:
        print 'Processing %d element array', nsky

    sz_sky = sky_vector.shape
    
    # Sort the sky in ascending values
    sky = np.array(sky_vector.flatten(), dtype=float)
    sky.sort()

    # Median value of all sky values
    skymid = 0.5 * sky[(nsky-1)/2] + 0.5 * sky[nsky/2]

    cut1 = np.min([skymid - sky[0], sky[nsky-1] - skymid])
    if highbad != None:
        cut1 = np.min([cut1, highbad - skymid])
    cut2 = skymid + cut1
    cut1 = skymid - cut1

    # Select the pixels between cut1 and cut2
    good = np.where( (sky <= cut2) & (sky >= cut1) )[0]
    
    if len(good) == 0:
        if not silent:
            print 'ERROR No sky values fall within %d and %d' % (cut1, cut2)
        return

    # Subtract the median to improve arithmetic accuracy
    delta = sky[good] - skymid
    sum = delta.sum()
    sumsq = (delta**2).sum()

    maximm = np.max(good)       # Highest value accepted at upper end of vector
    minimm = np.min(good) - 1  # Highest value reject at lower end of vector

    # Compute the mean and sigma (from the first pass)
    skymed = 0.5 * sky[(minimm+maximm+1)/2] + 0.5 * sky[(minimm+maximm)/2 + 1]
    skymn = sum / (maximm - minimm)
    sigma = math.sqrt(sumsq / (maximm - minimm) - skymn**2)
    skymn = skymn + skymid  # add median back in

    # If mean is less than mode, then contamination is slight, and the mean
    # value is what we really want
    if (skymed < skymn):
        skymode = (3. * skymed - 2. * skymn)  
    else:
        skymode = skymn

    # Rejection and computation Loop
    niter = 0
    clamp = 1.0
    old = 0

    redo = True
    while redo:
        niter += 1
        
        if niter > mxiter:
            if not silent:
                print 'ERROR Too many (%d) iterations, unable to compute sky' %\
                    niter
            return

        if (maximm - minimm < minsky):
            if not silent:
                print 'ERROR Too few (%d) valid sky elements, unable to compute sky' % (maximm - minimm)
            return

        # compute Chauvenet rejection criterion
        r = math.log10(float(maximm - minimm))
        r = np.max([2., ( -0.1042*r + 1.1695) * r + 0.8895])

        # compute rejection limits (symmetric about the current mode)
        cut = r * sigma + 0.5 * abs(skymn - skymode)
        if integer: 
            cut = cut if cut > 1.5 else 1.5
        cut1 = skymode - cut
        cut2 = skymode + cut

        # Recompute mean and sigma by adding and/or subtracting sky values
        # at both ends of the intervale of acceptable values.
        redo = False
        newmin = minimm
        
        # Is minimm+1 above current cut?
        tst_min = sky[newmin+1] >= cut1

        # Are we at first pixel of sky
        done = (newmin == -1) and tst_min
        if not done:
            done = (sky[newmin>0] <= cut1) and tst_min

        if not done:
            istep = 1 - 2 * int(tst_min)
            while not done:
                newmin = newmin + istep
                done = (newmin == -1) or (newmin == nlast)
                if not done:
                    done = (sky[newmin] <= cut1) and (sky[newmin+1] >= cut1)
            if tst_min:
                delta = sky[newmin+1:minimm+1] - skymid
            else:
                delta = sky[minimm+1:newmin+1] - skymid
    
            sum -= istep * delta.sum()
            sumsq -= istep * (delta**2).sum()
            redo = True
            minimm = newmin

        newmax = maximm
        
        # is current maximum below upper cut?
        tst_max = sky[maximm] <= cut2
        # are we at the last pixel of sky array?
        done = (maximm == nlast) and tst_max
        if not done:
            done = (tst_max) and (sky[min((maximm+1), nlast)] > cut2)

        if not done:
            istep = -1 + 2*int(tst_max)

            while not done:
                newmax = newmax + istep
                done = (newmax == nlast) or (newmax == -1)
                if not done:
                    done = (sky[newmax] <= cut2) and (sky[newmax+1] >= cut2)
            if tst_max:
                delta = sky[maximm+1:newmax+1] - skymid
            else:
                delta = sky[newmax+1:maximm+1] - skymid

            sum += istep * delta.sum()
            sumsq += istep * (delta**2).sum()
            redo = True
            maximm = newmax

        # Compute mean and sigma (from this pass)
        nsky = maximm - minimm
        if nsky < minsky:
            if not silent:
                print 'ERROR Outlier rejection left too few sky elements.'
            return

        skymn = sum / nsky
        tmp = sumsq/nsky - skymn**2
        if (tmp > 0):
            sigma = float( math.sqrt( tmp ) )
        else:
            sigma = 0
        skymn += skymid

        # Determine a more robust median by averaging the central 20% of
        # pixels. Estimate the median using the mean of the central 20 percent
        # of sky values. Be careful to include a perfectly symmetric sample
        # of pixels about the median, whether the total number is even or
        # odd within the acceptance interval.
        center = (minimm + 1 + maximm) / 2.0
        side = int(round(0.2 * (maximm - minimm))) / 2.0 + 0.25
        J = int(round(center - side))
        K = int(round(center + side))

        # In case the data has a large number of the same (quantized)
        # intensity, expand the range until both limiting values differ 
        # frm the central value by at least 0.25 times the read noise.
        if readnoise != None:
            L = int(round(center - 0.25))
            M = int(round(center + 0.25))
            R = 0.25 * readnoise

            while ((J > 0) and (K < nsky - 1) and
                   ( ((sky[L] - sky[J]) < R) or ((sky[K] - sky[M]) < R))):
                
                J = J - 1
                K = K + 1
        skymed = sky[J:K+1].sum() / (K - J + 1)

        # If the mean is less than the median, then the problem of contamination
        # is slight, and the mean is what we really want.
        if skymed < skymn:
            dmod = (3.0 * skymed - 2.0 * skymn - skymode)
        else:
            dmod = skymn - skymode

        # prevent oscillations by clamping down if the sky adjustments are
        # changing sign
        if dmod * old < 0: clamp = 0.5 * clamp
        skymode += clamp * dmod
        old = dmod

    skew = float( (skymn - skymode) / max([1.0, sigma]) )
    nsky = maximm - minimm
        
    if debug:
        print 'MMM: Number of unrejected sky elements: %d' % nsky
        print 'MMM: Number of iterations: %d' % niter
        print 'MMM: Mode, Sigma, Skew of sky vector: ', skymode, sigma, skew
        
    # outputs:
    # sky, skymod, sigma, skew, Nsky

    # Make our output object
    output = {}
    output['mode'] = skymode
    output['sigma'] = sigma
    output['skew'] = skew
    output['n'] = nsky

    return output
    
