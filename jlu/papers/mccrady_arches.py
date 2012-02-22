import numpy as np
import pylab as py
import asciidata
import matplotlib
import os
from gcwork import objects
from scipy import interpolate
import pyfits
from jlu.nirc2 import synthetic
from scipy import optimize
from gcwork import starTables
from gcwork import starset

def test_kp_ks_k_conversion(makeSynthetic=False):
    """
    For Arches photometry, we need to convert between Kp, Ks, and K
    filters for extinction corrections and model comparisons. This
    code tests whether we can simply use a linear relationship between
    H-Kp and Kp-Ks or Kp-K for the conversion. I use synthetic atmospheres
    for a simulated stellar population at 8 kpc, 2.5 Myr, solar metallicity,
    for extinctions between AKs of 2-4 to get the range of colors.
    """
    dir = '/u/jlu/work/arches/photo_calib/filter_conversions/'
    os.chdir(dir)

    if makeSynthetic:
        synthetic.nearIR(8000, 6.4)
    
    # Load up the synthetic data generated at a distance of 
    # 8 kpc and for a population with an age of 10**9.35 years.
    synFile = dir + 'syn_nir_d08000_a640.dat'

    T, AKs, J, H, K, Kp, Ks, Lp, mass, logg, logL = synthetic.load_nearIR(synFile)

    # From Schoedel et al. 2010, the range of extinctions is given by a
    # gaussian with mean AKs = 2.74 and stddev = 0.3. Our synthetic
    # photometry samples AKs in steps of 0.1 so we will take the range 
    # 2.1 <= AKs <= 4.2 (see Espinoza et al. 2009).
    adx = np.where((AKs >= 2.1) & (AKs <= 4.2))[0]
    
    AKs = AKs[adx]
    J = J[:,adx]
    H = H[:,adx]
    K = K[:,adx]
    Kp = Kp[:,adx]
    Ks = Ks[:,adx]
    Lp = Lp[:,adx]
    mass = mass[:,adx]
    logg = logg[:,adx]
    logL = logL[:,adx]

    # Lets make a color scale for our range of AKs for plotting.
    colorNorm = matplotlib.colors.Normalize(AKs)

    # First lets plot the full range of temperatures and magnitudes
    # to show what matches to the calibrators at Ks. Recall that our
    # photometric calibrators have H < 16.5, Ks < 14.5, Lp < 13.5.
    py.clf()
    pltH = py.semilogx(T, H, color='b', label='H')
    pltK = py.semilogx(T, Kp, color='g', label='Ks')
    pltL = py.semilogx(T, Lp, color='r', label='Lp')
    py.legend((pltH[0], pltK[0], pltL[0]), ('H', 'Kp', 'Lp'))

    loc = matplotlib.ticker.MultipleLocator(10000)
    py.gca().xaxis.set_major_locator( loc )

    title_DA = 'd=8 kpc, 2.1<=AKs<=4.2'

    py.xlim(40000, 3000)
    py.ylim(25, 8)
    py.xlabel('Effective Temperature (K)')
    py.ylabel('Magnitude')
    py.title(title_DA)
    py.savefig(dir + 'temp_vs_mag.png')

    ##########
    # Temperature range for calibrators (inclusive)
    ##########
    T_range = [6000, 35000]

    title_AT = 'AKs=[2.1:4.2] T=[6000:35000]'

    # Lets trim down
    tdx = np.where((T >= T_range[0]) & (T <= T_range[1]))[0]
    T = T[tdx]
    J = J[tdx,:]
    H = H[tdx,:]
    K = K[tdx,:]
    Kp = Kp[tdx,:]
    Ks = Ks[tdx,:]
    Lp = Lp[tdx,:]

    # Plot up the color-magnitude diagrams and compare with 
    # Schoedel et al. (2010).
    AKs_2D = AKs.repeat(len(T)).reshape((len(AKs), len(T))).transpose()

    # H-Ks vs. Ks
    py.clf()
    py.scatter(H-Ks, Ks, c=AKs_2D.flatten(), cmap=py.cm.jet, edgecolor='none')
    cbar = py.colorbar(orientation='vertical', fraction=0.2)
    cbar.set_label('AKs')

    rng = py.axis()
    py.xlim(0, 4)
    py.ylim(25, 8)
    py.xlabel('H - Ks')
    py.ylabel('Ks')
    py.title(title_AT)
    py.savefig(dir + 'cmd_h_ks.png')

    # Ks-Lp vs. Lp
    py.clf()
    py.scatter(Ks-Lp, Lp, c=AKs_2D.flatten(), cmap=py.cm.jet, edgecolor='none')
    cbar = py.colorbar(orientation='vertical', fraction=0.2)
    cbar.set_label('AKs')

    rng = py.axis()
    py.xlim(0, 3)
    py.ylim(25, 8)
    py.xlabel('Ks - Lp')
    py.ylabel('Lp')
    py.title(title_AT)
    py.savefig(dir + 'cmd_ks_lp.png')

    
    # Plot up the relationship between H-Ks and Kp-Ks.. same for Ks-Lp.
    # In principle, we can use this relationship directly as long as any
    # scatter for this simulated stellar population is less than 1%
    # First, Lets fit a line to each relation
    hkp = (H - Kp).flatten()
    kpks = (Kp - Ks).flatten()
    kplp = (Kp - Lp).flatten()
    hkp_coeffs = np.polyfit(hkp, kpks, 1)
    kplp_coeffs = np.polyfit(kplp, kpks, 1)

    hkp_idx = hkp.argsort()
    kplp_idx = kplp.argsort()

    hkp_fit = np.polyval(hkp_coeffs, hkp[hkp_idx])
    kplp_fit = np.polyval(kplp_coeffs, kplp[kplp_idx])

    # H-Ks vs. Kp-Ks
    py.clf()
    py.scatter(H-Kp, Kp-Ks, c=AKs_2D.flatten(), cmap=py.cm.jet, 
               edgecolor='none')
    py.plot(hkp[hkp_idx], hkp_fit, 'k--')

    cbar = py.colorbar(orientation='vertical', fraction=0.2)
    cbar.set_label('AKs')
    py.xlabel('H - Kp')
    py.ylabel('Kp - Ks')
    py.title('Kp-Ks = %.4f + %.4f * H-Kp' % (hkp_coeffs[0], hkp_coeffs[1]),
             fontsize=14)
    py.savefig(dir+'color_HKp_KpKs.png')

    diff = hkp_fit - kpks[hkp_idx]
    print 'Best fit Kp-Ks = %.5f + %.5f * H-Ks' % \
        (hkp_coeffs[0], hkp_coeffs[1])
    print 'Residuals/Range from H-Kp vs. Kp-Ks fit: %.4f  [%.4f - %.4f]' % \
        (diff.std(), diff.min(), diff.max())
    print ''

    # Kp-Lp vs. Kp-Ks
    py.clf()
    py.scatter(Kp-Lp, Kp-Ks, c=AKs_2D.flatten(), cmap=py.cm.jet, 
               edgecolor='none')
    py.plot(kplp[kplp_idx], kplp_fit, 'k--')

    cbar = py.colorbar(orientation='vertical', fraction=0.2)
    cbar.set_label('AKs')
    py.xlabel('Kp - Lp')
    py.ylabel('Kp - Ks')
    py.title('Kp-Ks = %.4f + %.4f * Kp-Lp' % (kplp_coeffs[0], kplp_coeffs[1]),
             fontsize=14)
    py.savefig(dir+'color_KpLp_KpKs.png')

    diff = kplp_fit - kpks[kplp_idx]
    print 'Best fit Kp-Ks = %.5f + %.5f * Kp-Lp' % \
        (kplp_coeffs[0], kplp_coeffs[1])
    print 'Residuals/Range from Kp-Lp vs. Kp-Ks fit: %.4f  [%.4f - %.4f]' % \
        (diff.std(), diff.min(), diff.max())
    print ''

    # Lets do the same for Kp vs. K
    hkp = (H - Kp).flatten()
    kpk = (Kp - K).flatten()
    kplp = (Kp - Lp).flatten()
    hkp_coeffs = np.polyfit(hkp, kpk, 1)
    kplp_coeffs = np.polyfit(kplp, kpk, 1)

    hkp_idx = hkp.argsort()
    kplp_idx = kplp.argsort()

    hkp_fit = np.polyval(hkp_coeffs, hkp[hkp_idx])
    kplp_fit = np.polyval(kplp_coeffs, kplp[kplp_idx])

    # H-Kp vs. Kp-K
    py.clf()
    py.scatter(H-Kp, Kp-K, c=AKs_2D.flatten(), cmap=py.cm.jet, 
               edgecolor='none')
    py.plot(hkp[hkp_idx], hkp_fit, 'k--')

    cbar = py.colorbar(orientation='vertical', fraction=0.2)
    cbar.set_label('AKs')
    py.xlabel('H - Kp')
    py.ylabel('Kp - K')
    py.title('Kp-K = %.4f + %.4f * H-Kp' % (hkp_coeffs[0], hkp_coeffs[1]),
             fontsize=14)
    py.savefig(dir+'color_HKp_KpK.png')

    diff = hkp_fit - kpk[hkp_idx]
    print 'Best fit Kp-K = %.5f + %.5f * H-Kp' % \
        (hkp_coeffs[0], hkp_coeffs[1])
    print 'Residuals/Range from H-Kp vs. Kp-K fit: %.4f  [%.4f - %.4f]' % \
        (diff.std(), diff.min(), diff.max())
    print ''

    # Kp-Lp vs. Kp-K
    py.clf()
    py.scatter(Kp-Lp, Kp-K, c=AKs_2D.flatten(), cmap=py.cm.jet, 
               edgecolor='none')
    py.plot(kplp[kplp_idx], kplp_fit, 'k--')

    cbar = py.colorbar(orientation='vertical', fraction=0.2)
    cbar.set_label('AKs')
    py.xlabel('Kp - Lp')
    py.ylabel('Kp - K')
    py.title('Kp-K = %.4f + %.4f * Kp-Lp' % (kplp_coeffs[0], kplp_coeffs[1]),
             fontsize=14)
    py.savefig(dir+'color_KpLp_KpK.png')

    diff = kplp_fit - kpk[kplp_idx]
    print 'Best fit Kp-K = %.5f + %.5f * Kp-Lp' % \
        (kplp_coeffs[0], kplp_coeffs[1])
    print 'Residuals/Range from Kp-Lp vs. Kp-K fit: %.4f  [%.4f - %.4f]' % \
        (diff.std(), diff.min(), diff.max())

