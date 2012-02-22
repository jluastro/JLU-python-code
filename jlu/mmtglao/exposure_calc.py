import math

def signalToNoise(mag=9, tint=1, coadds=1, filter='Ks'):
    """
    Calculate the signal to noise for PISCES behind the
    MMT GLAO system.
    """
    # Gain (electrons/DN)
    g = 4.35

    # Readnoise (e-)
    R = 4.0 * g

    # Aperture (in pixels)
    apRadius = 4.0
    npix = math.pi * apRadius**2

    # Background (e-/sec)
    B = 0.0 * g ### **** DON'T KNOW *** ###

    # Zeropoint (for flux in DN/s)
    # numbers calculated from observations taken in October 2008
    # with PISCES behind the GLAO system observing 2MASS standards.
    # Zeropoint should be around 18.4 based on my by hand calculations.
    m0 = 9.974
    f0 = 143771.0 / 60.0  # integrated flux from tint=1, coadds=60 (in DN/s)
    f0peak = 4902.0 / 60.0 # DN/s

    ZP = m0 + 2.5 * math.log10(f0)

    # Calculate the signal
    fluxInDNperSec = 10.0**(-(mag - ZP)/2.5)
    S = fluxInDNperSec * g # e-/sec

    ##########
    # SNR
    ##########
    # Calculate the signal-to-noise
    totalSignal = S * coadds * tint # (e-)
    totalNoise = math.sqrt(S + npix * (S + B + R**2/tint))
    totalNoise *= math.sqrt(coadds * tint) # (e-) 
    snr = totalSignal / totalNoise

    ##########
    # Saturation
    ##########
    # Non-linear Level (DN)
    nonLinLevel = 12000.0
    # Saturation Level
    satLevel = 20000.0

    # Determine for what integration time this star would saturate.
    # Calculate the flux in the peak pixel (for saturation purposes)
    # This assumes the same Strehl performance as in 08oct data.
    peakFlux = f0peak * 10**(-(mag - m0) / 2.5) # DN/s
    nonLinearTime = nonLinLevel / peakFlux
    saturationTime = satLevel / peakFlux

    # Determine what magnitude star would saturate in this tint.
    fluxAtNonLin = nonLinLevel / tint # (DN/s)
    fluxAtSatur = satLevel / tint # (DN/s)

    magAtNonLin = m0 - 2.5 * math.log10(fluxAtNonLin/f0peak)
    magAtSatur = m0 - 2.5 * math.log10(fluxAtSatur/f0peak)

    ##########
    # Print out all the results
    ##########
    print 'Exposure Properties:'
    print '       tint = %5.1f sec' % (tint)
    print '     coadds = %5d' % (coadds)
    print '     filter = %5s' % (filter)
    print '  zeropoint = %5.1f mag' % (ZP)
    print ''
    print 'Results for a star of %s = %5.2f:' % (filter, mag)
    print '                    SNR = %8.1f' % (snr)
    print '           Total Signal = %8d DN' % (totalSignal / g)
    print '            Total Noise = %8d DN' % (totalNoise / g)
    print '          Aperture Area = %8.1f pix^2' % (npix)
    print '  Peak Signal (1 coadd) = %8.1f DN' % (peakFlux * tint)
    print ''
    print 'Saturation Properties:'
    print '    nonlinear regime = %5d DN' % (nonLinLevel)
    print '          satruation = %5d DN' % (satLevel)
    print '   nonlinear in tint = %.1f s for %2s = %.1f star' % \
        (nonLinearTime, filter, mag)
    print '  saturation in tint = %.1f s for %2s = %.1f star' % \
        (saturationTime, filter, mag)
    print '     nonlinear at %2s = %.1f in tint = %.1f s' % \
        (filter, magAtNonLin, tint)
    print '    sautration at %2s = %.1f in tint = %.1f s' % \
        (filter, magAtSatur, tint)
    
