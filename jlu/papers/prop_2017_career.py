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
    xS0 = np.array([0.000, 0.000])
    beta = 3.0 # mas
    muS = np.array([-10.0, -0.0])
    muL = np.array([0.0, 0.0])
    dL = 4000.0
    dS = 8000.0
    imag = 19.0

    # No parallax
    pspl_n = model.PSPL(mL, t0, xS0, beta, muL, muS, dL, dS, imag)
    
    # With parallax
    pspl_p = model.PSPL_parallax(raL, decL, mL, t0, xS0, beta, muL, muS, dL, dS, imag)

    t = np.arange(t0 - 500, t0 + 500, 1)
    dt = t - pspl_n.t0

    A_n = pspl_n.get_amplification(t)
    A_p = pspl_p.get_amplification(t)

    xS_n = pspl_n.get_astrometry(t)
    xS_p_unlens = pspl_p.get_astrometry_unlensed(t)
    xS_p_lensed = pspl_p.get_astrometry(t)
    xL_p = pspl_p.get_lens_astrometry(t)

    # Plot the amplification
    fig1 = plt.figure(1)
    plt.clf()
    f1_1 = fig1.add_axes((0.20, 0.3, 0.75, 0.6))
    plt.plot(dt, 2.5 * np.log10(A_n), 'b-', label='No parallax')
    plt.plot(dt, 2.5 * np.log10(A_p), 'r-', label='Parallax')
    plt.legend(fontsize=10)
    plt.ylabel('2.5 * log(A)')
    f1_1.set_xticklabels([])

    # plt.savefig(outdir + 'lens_ref_amp_v_time.png')
    # print("save to " + outdir)
    
    
    # Plot the positions of everything
    fig2 = plt.figure(2)
    plt.clf()
    plt.plot(xS_n[:, 0] * 1e3, xS_n[:, 1] * 1e3, 'r--',
                 mfc='none', mec='red', label='Src, No parallax model')
    plt.plot(xS_p_unlens[:, 0] * 1e3, xS_p_unlens[:, 1] * 1e3, 'b--',
                 mfc='none', mec='blue',
             label='Src, Parallax model, unlensed')
    plt.plot(xL_p[:, 0] * 1e3, xL_p[:, 1] * 1e3, 'k--',
                 mfc='none', mec='grey', label='Lens')
    plt.plot(xS_p_lensed[:, 0] * 1e3, xS_p_lensed[:, 1] * 1e3, 'b-', label='Src, Parallax model, lensed')
    plt.legend(fontsize=10)
    plt.gca().invert_xaxis()
    plt.xlabel('R.A. (mas)')
    plt.ylabel('Dec. (mas)')
    plt.axis('equal')
    lim = 10
    print('LIM = ', lim)
    plt.xlim(lim, -lim) # arcsec
    plt.ylim(-lim, lim)
    plt.savefig(outdir + 'on_sky.png')

    # Check just the astrometric shift part.
    shift_n = pspl_n.get_centroid_shift(t) # mas
    shift_p = (xS_p_lensed - xS_p_unlens) * 1e3 # mas
    shift_n_amp = np.linalg.norm(shift_n, axis=1)
    shift_p_amp = np.linalg.norm(shift_p, axis=1)
    
    fig3 = plt.figure(3)
    plt.clf()
    f1_3 = fig3.add_axes((0.20, 0.3, 0.75, 0.6))
    plt.plot(dt, shift_n_amp, 'r--', label='No parallax model')
    plt.plot(dt, shift_p_amp, 'b--', label='Parallax model')
    plt.ylabel('Astrometric Shift (mas)')
    plt.legend(fontsize=10)
    f1_3.set_xticklabels([])

    f2_3 = fig3.add_axes((0.20, 0.1, 0.75, 0.2))
    plt.plot(dt, shift_p_amp - shift_n_amp, 'k-', label='Par - No par')
    plt.legend(fontsize=10)
    plt.axhline(0, linestyle='--', color='k')
    plt.ylabel('Diff (mas)')
    plt.xlabel('t - t0 (MJD)')
    
    plt.savefig(outdir + 'shift_amp_v_t.png')

   

    fig4 = plt.figure(4)
    plt.clf()
    plt.plot(shift_n[:, 0], shift_n[:, 1], 'r-', label='No parallax')
    plt.plot(shift_p[:, 0], shift_p[:, 1], 'b-', label='Parallax')
    plt.axhline(0, linestyle='--')
    plt.axvline(0, linestyle='--')
    plt.gca().invert_xaxis()
    plt.legend(fontsize=10)
    plt.xlabel('Shift RA (mas)')
    plt.ylabel('Shift Dec (mas)')
    plt.axis('equal')
    plt.savefig(outdir + 'shift_on_sky.png')
    

    print('Einstein radius: ', pspl_n.thetaE_amp, pspl_p.thetaE_amp)
    print('Einstein crossing time: ', pspl_n.tE, pspl_n.tE)

    return
    
