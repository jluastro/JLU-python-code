import math

def imagerLimitingMag(tint):
    """
    Determine the limiting magnitude (SNR = 3.0) for a Kbb image of a given
    exposure time. This calculation assumes that the dominant source of
    noise is the sky background. Read noise is ignored (don't know the proper
    values at this moment, but it seems to be negligable at K).

    Input:
    tint -- total integration time in seconds
    """
    # Values pulled from i070720_a019001.fits
    snr = 3.0
    f0 = 56750.0   # numbers from S1-23
    m0 = 11.9
    sky = 500.0

    apertureRadius = 5.0
    apertureArea = math.pi * apertureRadius**2

    bkg = sky * apertureArea

    mag = m0 - 2.5 * math.log10( (snr/f0) * math.sqrt( bkg / tint))

    print('Limiting Magnitude ')
    print('    is K = %5.2f ' % mag)
    print('    in tint = %d seconds' % tint)
    print('    at SNR  = %d' % snr)

    
    
