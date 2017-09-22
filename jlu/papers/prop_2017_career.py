from jlu.microlens import model
import numpy as np
import pylab as plt

plot_dir = '/u/jlu/doc/proposals/nsf/career/2017/plots/'

def plot_astrometry_lens_ref():
    """
    Plot the astrometry as seen on the sky when in the
    rest frame of the lens.
    """
    # Scenario from Belokurov and Evans 2002 (Figure 1)
    raL = 17.5
    decL = -30.0
    mL = 10.0 # msun
    t0 = 57650.0
    xS0 = np.array([0.000, 0.003])
    beta = -3.0 # mas
    muS = np.array([-8.0, 0.0])
    muL = np.array([0.0, 0.0])
    dL = 3000.0
    dS = 6000.0
    imag = 19.0

    # With parallax
    pspl_p = model.PSPL_parallax(raL, decL, mL, t0, xS0, beta, muL, muS, dL, dS, imag)

    # In Days.
    t = np.arange(t0 - 5000, t0 + 5000, 10)
    dt = t - pspl_p.t0

    A_p = pspl_p.get_amplification(t)
    i_p = pspl_p.get_photometry(t)

    xS_p_unlens = pspl_p.get_astrometry_unlensed(t)
    xS_p_lensed = pspl_p.get_astrometry(t)
    xL_p = pspl_p.get_lens_astrometry(t)

    # Plot the amplification
    plt.close(1)
    fig = plt.figure(1, figsize=(18, 5.67))
    plt.clf()
    plt.subplots_adjust(left=0.065, right=0.98, bottom=0.16, wspace=0.34)
    ax1 = fig.add_subplot(131)
    ax2 = fig.add_subplot(132)
    ax3 = fig.add_subplot(133)
    ax1.invert_yaxis()
    ax1.plot(dt, i_p, 'r-')
    ax1.set_ylabel('I-band (mag)')
    ax1.set_xlabel(r'$t - t_0$ (days)')
    ax1.set_xlim(-1000, 1000)

    # Plot the positions of everything
    ax2.plot(xL_p[:, 0] * 1e3, xL_p[:, 1] * 1e3, 'k--',
                 mfc='none', mec='grey', label='Lens')
    ax2.plot(xS_p_unlens[:, 0] * 1e3, xS_p_unlens[:, 1] * 1e3, 'b--',
                 mfc='none', mec='blue', label='Src, unlensed')
    ax2.plot(xS_p_lensed[:, 0] * 1e3, xS_p_lensed[:, 1] * 1e3, 'r-', label='Src, lensed')
    ax2.legend(fontsize=10)
    ax2.invert_xaxis()
    ax2.set_xlabel('R.A. (mas)')
    ax2.set_ylabel('Dec. (mas)')
    ax2.axis('equal')
    lim = 18
    ax2.set_xlim(lim, -lim) # arcsec
    ax2.set_ylim(-lim, lim)

    # Check just the astrometric shift part.
    shift_p = (xS_p_lensed - xS_p_unlens) * 1e3 # mas
    shift_p_amp = np.linalg.norm(shift_p, axis=1)
    
    ax3.plot(shift_p[:, 0], shift_p[:, 1], 'r-')
    ax3.axhline(0, linestyle='--', color='grey')
    ax3.axvline(0, linestyle='--', color='grey')
    ax3.invert_xaxis()
    ax3.set_xlabel(r'$\Delta$R.A. (mas)')
    ax3.set_ylabel(r'$\Delta$Dec. (mas)')
    ax3.axis('equal')

    plt.savefig(plot_dir + 'phot_astrom.png')

    print('Maximum astrometric sift: {0:.2f} mas'.format(shift_p_amp.max()))

    return

