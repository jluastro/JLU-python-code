import numpy as np
import pylab as py
from scipy import interpolate
import scipy

def schoedel09(wavelength, AKs):
    """
    Return the extinction based on the Schoedel et al. 2009 extinction
    law. Their extinction is calibrated to Ks-band observations.
    
    Inputs:
    wavelength -- in microns
    AKs -- the extinction for Ks band in magnitudes
    """
    pass

def schoedel09_HKs():
    return 2.21

def schoedel09_KsLp():
    return 1.34

def cardelli(wavelength, Rv):
    """
    Cardelli extinction law
    """
    x = 1.0 / wavelength
    
    # check for applicability
    if (x < 0.3):
        print 'wavelength is longer than applicable range for Cardelli law'
        return None

    if (x > 8.0):
        print 'wavelength is shorter than applicable range for Cardelli law'
        return None

    y = x - 1.82

    if (x <= 1.1):
        a =  0.574*x**1.61
        b = -0.527*x**1.61

    if (x > 1.1) and (x <= 3.3):
        a = 1 + 0.17699*y - 0.50447*y**2 - 0.02427*y**3 + \
            0.72085*y**4 + 0.01979*y**5 - 0.77530*y**6 + 0.32999*y**7
        b = 1.41338*y + 2.28305*y**2 + 1.07233*y**3 - \
            5.38434*y**4 - 0.62251*y**5 + 5.30260*y**6 - 2.09002*y**7

    if (x > 3.3):
        if (x < 5.9):
            Fa = 0.
            Fb = 0.
        else:
            Fa = -0.04473*(x-5.9)**2 - 0.009779*(x-5.9)**3
            Fb = 0.2130*(x-5.9)**2 + 0.1207*(x-5.9)**3

        a = 1.752 - 0.316*x -0.104/((x-4.67)**2.+0.341) + Fa
        b = -3.090 + 1.825*x + 1.206/((x-4.62)**2.+0.263) + Fb

    # A(lam)/A(V)
    extinction = a + b/Rv

    return extinction


def nishiyama09(wavelength, AKs, makePlot=False):
    # Data pulled from Nishiyama et al. 2009, Table 1

    filters = ['V', 'J', 'H', 'Ks', '[3.6]', '[4.5]', '[5.8]', '[8.0]']
    wave =      np.array([0.551, 1.25, 1.63, 2.14, 3.545, 4.442, 5.675, 7.760])
    A_AKs =     np.array([16.13, 3.02, 1.73, 1.00, 0.500, 0.390, 0.360, 0.430])
    A_AKs_err = np.array([0.04,  0.04, 0.03, 0.00, 0.010, 0.010, 0.010, 0.010])

    # Interpolate over the curve
    spline_interp = interpolate.splrep(wave, A_AKs, k=3, s=0)

    A_AKs_at_wave = interpolate.splev(wavelength, spline_interp)
    A_at_wave = AKs * A_AKs_at_wave

    if makePlot:
        py.clf()
        py.errorbar(wave, A_AKs, yerr=A_AKs_err, fmt='bo', 
                    markerfacecolor='none', markeredgecolor='blue',
                    markeredgewidth=2)
        
        # Make an interpolated curve.
        wavePlot = np.arange(wave.min(), wave.max(), 0.1)
        extPlot = interpolate.splev(wavePlot, spline_interp)
        py.loglog(wavePlot, extPlot, 'k-')

        # Plot a marker for the computed value.
        py.plot(wavelength, A_AKs_at_wave, 'rs',
                markerfacecolor='none', markeredgecolor='red',
                markeredgewidth=2)
        py.xlabel('Wavelength (microns)')
        py.ylabel('Extinction (magnitudes)')
        py.title('Nishiyama et al. 2009')

    
    return A_at_wave

def romanzuniga07(wavelength, AKs, makePlot=False):
    # Data pulled from Nishiyama et al. 2009, Table 1

    filters = ['J', 'H', 'Ks', '[3.6]', '[4.5]', '[5.8]', '[8.0]']
    wave =      np.array([1.240, 1.664, 2.164, 3.545, 4.442, 5.675, 7.760])
    A_AKs =     np.array([2.299, 1.550, 1.000, 0.618, 0.525, 0.462, 0.455])
    A_AKs_err = np.array([0.530, 0.080, 0.000, 0.077, 0.063, 0.055, 0.059])

    # Interpolate over the curve
    spline_interp = interpolate.splrep(wave, A_AKs, k=3, s=0)

    A_AKs_at_wave = interpolate.splev(wavelength, spline_interp)
    A_at_wave = AKs * A_AKs_at_wave

    if makePlot:
        py.clf()
        py.errorbar(wave, A_AKs, yerr=A_AKs_err, fmt='bo', 
                    markerfacecolor='none', markeredgecolor='blue',
                    markeredgewidth=2)
        
        # Make an interpolated curve.
        wavePlot = np.arange(wave.min(), wave.max(), 0.1)
        extPlot = interpolate.splev(wavePlot, spline_interp)
        py.loglog(wavePlot, extPlot, 'k-')

        # Plot a marker for the computed value.
        py.plot(wavelength, A_AKs_at_wave, 'rs',
                markerfacecolor='none', markeredgecolor='red',
                markeredgewidth=2)
        py.xlabel('Wavelength (microns)')
        py.ylabel('Extinction (magnitudes)')
        py.title('Roman Zuniga et al. 2007')

    
    return A_at_wave

