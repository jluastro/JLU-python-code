from jlu.microlens import model
from jlu.microlens import model_fitter
import numpy as np
import pylab as plt
import pdb
import os

def test_PSPL():
    # mL = 10.0 # msun
    # t0 = 57000.00
    # xS0 = np.array([0.000, 0.000])
    # # beta = -0.4 # mas
    # beta = 1.4 # mas
    # muS = np.array([8.0, 0.0])
    # # muL = np.array([-7.0, 0.00])
    # muL = np.array([0.00, 0.00])
    # dL = 4000.0
    # dS = 8000.0
    # imag = 19.0

    # Scenario from Belokurov and Evans 2002 (Figure 1)
    mL = 0.5 # msun
    t0 = 57000.00
    xS0 = np.array([0.000, 0.000])
    beta = 7.41 # mas
    muS = np.array([-7.0, 1.0])
    muL = np.array([-100.0, 0.00])
    dL = 150.0
    dS = 1500.0
    imag = 19.0
    
    
    pspl = model.PSPL(mL, t0, xS0, beta, muL, muS, dL, dS, imag)

    t = np.arange(54000, 60000, 1)
    dt = t - pspl.t0

    A = pspl.get_amplification(t)
    shift = pspl.get_centroid_shift(t)
    shift_amp = np.linalg.norm(shift, axis=1)

    # Plot the amplification
    plt.figure(1)
    plt.clf()
    plt.plot(dt, 2.5 * np.log10(A), 'k.')
    plt.xlabel('t - t0 (MJD)')
    plt.ylabel('2.5 * log(A)')

    # Plot the positions of everything
    lens_pos = pspl.xL0 + np.outer(dt / model.days_per_year, pspl.muL) * 1e-3
    srce_pos = pspl.xS0 + np.outer(dt / model.days_per_year, pspl.muS) * 1e-3
    imag_pos = srce_pos + (shift * 1e-3)

    plt.figure(2)
    plt.clf()
    plt.plot(lens_pos[:, 0], lens_pos[:, 1], 'r--', mfc='none', mec='red')
    plt.plot(srce_pos[:, 0], srce_pos[:, 1], 'b--', mfc='none', mec='blue')
    plt.plot(imag_pos[:, 0], imag_pos[:, 1], 'b-')
    lim = 0.005
    plt.xlim(lim, -lim) # arcsec
    plt.ylim(-lim, lim)
    plt.xlabel('dRA (arcsec)')
    plt.ylabel('dDec (arcsec)')
    plt.title('Zoomed-in')

    plt.figure(3)
    plt.clf()
    plt.plot(lens_pos[:, 0], lens_pos[:, 1], 'r--', mfc='none', mec='red')
    plt.plot(srce_pos[:, 0], srce_pos[:, 1], 'b--', mfc='none', mec='blue')
    plt.plot(imag_pos[:, 0], imag_pos[:, 1], 'b-')
    lim = 0.05
    plt.xlim(lim, -lim) # arcsec
    plt.ylim(-lim, lim)
    plt.xlabel('dRA (arcsec)')
    plt.ylabel('dDec (arcsec)')
    plt.title('Zoomed-out')
    
    plt.figure(4)
    plt.clf()
    plt.plot(dt, shift_amp)
    plt.xlabel('t - t0 (MJD)')
    plt.ylabel('Astrometric Shift (mas)')

    plt.figure(5)
    plt.clf()
    plt.plot(shift[:, 0], shift[:, 1])
    plt.gca().invert_xaxis()
    plt.xlabel('RA Shift (mas)')
    plt.ylabel('Dec Shift (mas)')
    
    plt.close(6)
    f, (ax1, ax2, ax3) = plt.subplots(3, sharex=True)
    f.subplots_adjust(hspace=0)
    ax1.plot(dt / pspl.tE, shift[:, 0] / pspl.thetaE_amp, 'k-')
    ax2.plot(dt / pspl.tE, shift[:, 1] / pspl.thetaE_amp, 'k-')
    ax3.plot(dt / pspl.tE, shift_amp / pspl.thetaE_amp, 'k-')
    ax3.set_xlabel('(t - t0) / tE)')
    ax1.set_ylabel(r'dX / $\theta_E$')
    ax2.set_ylabel(r'dY / $\theta_E$')
    ax3.set_ylabel(r'dT / $\theta_E$')
    ax1.set_ylim(-0.4, 0.4)
    if shift[0, 1] > 0:
        ax2.set_ylim(0, 0.4)
    else:
        ax2.set_ylim(-0.4, 0)
    ax3.set_ylim(0, 0.4)

    print 'Einstein radius: ', pspl.thetaE_amp
    print 'Einstein crossing time: ', pspl.tE

    return pspl    


def test_pspl_fit():
    data, p_in = fake_data1()

    model_fitter.multinest_pspl(data, n_live_points=300, saveto='./mnest_pspl/', runcode='aa')

    model_fitter.plot_posteriors('mnest_pspl/', 'aa')

    best = model_fitter.get_best_fit('mnest_pspl/', 'aa')

    
    pspl_out = model.PSPL(best['mL'], best['t0'], np.array([best['xS0_E'], best['xS0_N']]), best['beta'], 
                          np.array([best['muL_E'], best['muL_N']]), np.array([best['muS_E'], best['muS_N']]),
                          best['dL'], best['dS'], best['imag_base'])

    pspl_in = model.PSPL(p_in['mL'], p_in['t0'], np.array([p_in['xS0_E'], p_in['xS0_N']]), p_in['beta'], 
                          np.array([p_in['muL_E'], p_in['muL_N']]), np.array([p_in['muS_E'], p_in['muS_N']]),
                          p_in['dL'], p_in['dS'], p_in['imag_base'])
    
    imag_out = pspl_out.get_photometry(data['t_phot'])
    pos_out = pspl_out.get_astrometry(data['t_ast'])

    imag_in = pspl_in.get_photometry(data['t_phot'])
    pos_in = pspl_in.get_astrometry(data['t_ast'])
    
    lnL_phot_out = pspl_out.likely_photometry(data['t_phot'], data['imag'], data['imag_err'])
    lnL_ast_out = pspl_out.likely_astrometry(data['t_ast'], data['xpos'], data['ypos'],
                                             data['xpos_err'], data['ypos_err'])
    lnL_out = lnL_phot_out.mean() + lnL_ast_out.mean()

    lnL_phot_in = pspl_in.likely_photometry(data['t_phot'], data['imag'], data['imag_err'])
    lnL_ast_in = pspl_in.likely_astrometry(data['t_ast'], data['xpos'], data['ypos'],
                                             data['xpos_err'], data['ypos_err'])
    lnL_in = lnL_phot_in.mean() + lnL_ast_in.mean()
    
    print 'lnL for input: ', lnL_in
    print 'lnL for output: ', lnL_out
    pdb.set_trace()
    
    plt.figure(1)
    plt.clf()
    plt.errorbar(data['t_phot'], data['imag'], yerr=data['imag_err'], fmt='k.')
    plt.plot(data['t_phot'], imag_out, 'r-')
    plt.plot(data['t_phot'], imag_in, 'g-')
    plt.xlabel('t - t0 (days)')
    plt.ylabel('I (mag)')
    plt.title('Input Data and Output Model')

    plt.figure(2)
    plt.clf()
    plt.errorbar(data['xpos'], data['ypos'], xerr=data['xpos_err'], yerr=data['ypos_err'], fmt='k.')
    plt.plot(pos_out[:, 0], pos_out[:, 1], 'r-')
    plt.plot(pos_in[:, 0], pos_in[:, 1], 'g-')
    plt.gca().invert_xaxis()
    plt.xlabel('X Pos (")')
    plt.ylabel('Y Pos (")')
    plt.title('Input Data and Output Model')

    plt.figure(3)
    plt.clf()
    plt.errorbar(data['t_ast'], data['xpos'], yerr=data['xpos_err'], fmt='k.')
    plt.plot(data['t_ast'], pos_out[:, 0], 'r-')
    plt.plot(data['t_ast'], pos_in[:, 0], 'g-')
    plt.xlabel('t - t0 (days)')
    plt.ylabel('X Pos (")')
    plt.title('Input Data and Output Model')

    plt.figure(4)
    plt.clf()
    plt.errorbar(data['t_ast'], data['ypos'], yerr=data['ypos_err'], fmt='k.')
    plt.plot(data['t_ast'], pos_out[:, 1], 'r-')
    plt.plot(data['t_ast'], pos_in[:, 1], 'g-')
    plt.xlabel('t - t0 (days)')
    plt.ylabel('Y Pos (")')
    plt.title('Input Data and Output Model')
    
    return


def test_pspl_parallax():
    # raL = (17.0 + (45.0 / 60.) + (40.0 / 3600.0)) * 15.0   # degrees
    # decL = -29 + (1.0 / 60.0) + (28.0 / 3600.0)
    # mL = 10.0 # msun
    # t0 = 57000.00
    # xS0 = np.array([0.000, 0.000])
    # beta = -0.8 # mas
    # muS = np.array([1.5, -0.5])
    # muL = np.array([-7.0, 0.00])
    # dL = 4000.0
    # dS = 8000.0
    # imag = 19.0

    # Scenario from Belokurov and Evans 2002 (Figure 1)
    raL = 17.5
    decL = +65.0
    mL = 0.5 # msun
    t0 = 57000.00
    xS0 = np.array([0.000, 0.000])
    beta = -7.41 # mas
    muS = np.array([-2.0, 7.0])
    muL = np.array([95.4, 0.00])
    dL = 150.0
    dS = 1500.0
    imag = 19.0
    
    # No parallax
    pspl_n = model.PSPL(mL, t0, xS0, beta, muL, muS, dL, dS, imag)
    print 'pspl_n.u0', pspl_n.u0
    print 'pspl_n.muS', pspl_n.muS
    print 'pspl_n.u0_hat', pspl_n.u0_hat
    print 'pspl_n.thetaE_hat', pspl_n.thetaE_hat
    
    # With parallax
    pspl_p = model.PSPL_parallax(raL, decL, mL, t0, xS0, beta, muL, muS, dL, dS, imag)

    t = np.arange(56000, 58000, 1)
    dt = t - pspl_n.t0

    A_n = pspl_n.get_amplification(t)
    A_p = pspl_p.get_amplification(t)

    xS_n = pspl_n.get_astrometry(t)
    xS_p_unlens = pspl_p.get_astrometry_unlensed(t)
    xS_p_lensed = pspl_p.get_astrometry(t)

    # Plot the amplification
    fig1 = plt.figure(1)
    plt.clf()
    f1_1 = fig1.add_axes((0.1, 0.3, 0.8, 0.6))
    plt.plot(dt, 2.5 * np.log10(A_n), 'b-', label='No parallax')
    plt.plot(dt, 2.5 * np.log10(A_p), 'r-', label='Parallax')
    plt.legend()
    plt.ylabel('2.5 * log(A)')
    f1_1.set_xticklabels([])

    f2_1 = fig1.add_axes((0.1, 0.1, 0.8, 0.2))
    plt.plot(dt, 2.5 * (np.log10(A_p) - np.log10(A_n)), 'k-', label='Par - No par')
    plt.axhline(0, linestyle='--', color='k')
    plt.legend()
    plt.ylabel('Diff')
    plt.xlabel('t - t0 (MJD)')

    
    # Plot the positions of everything
    fig2 = plt.figure(2)
    plt.clf()
    plt.plot(xS_n[:, 0], xS_n[:, 1], 'r--', mfc='none', mec='red', label='No parallax model')
    plt.plot(xS_p_unlens[:, 0], xS_p_unlens[:, 1], 'b--', mfc='none', mec='blue',
             label='Parallax model, unlensed')
    plt.plot(xS_p_lensed[:, 0], xS_p_lensed[:, 1], 'b-', label='Parallax model, lensed')
    plt.legend()
    # plt.gca().invert_xaxis()
    # lim = 0.05
    # plt.xlim(lim, -lim) # arcsec
    # plt.ylim(-lim, lim)
    plt.xlim(0.006, -0.006) # arcsec
    plt.ylim(-0.02, 0.02)
    plt.xlabel('R.A. (")')
    plt.ylabel('Dec. (")')

    # Check just the astrometric shift part.
    shift_n = pspl_n.get_centroid_shift(t) # mas
    shift_p = (xS_p_lensed - xS_p_unlens) * 1e3 # mas
    shift_n_amp = np.linalg.norm(shift_n, axis=1)
    shift_p_amp = np.linalg.norm(shift_p, axis=1)
    
    fig3 = plt.figure(3)
    plt.clf()
    f1_3 = fig3.add_axes((0.15, 0.3, 0.8, 0.6))
    plt.plot(dt, shift_n_amp, 'r--', label='No parallax model')
    plt.plot(dt, shift_p_amp, 'b--', label='Parallax model')
    plt.legend()
    f1_3.set_xticklabels([])

    f2_3 = fig3.add_axes((0.15, 0.1, 0.8, 0.2))
    plt.plot(dt, shift_p_amp - shift_n_amp, 'k-', label='Par - No par')
    plt.legend()
    plt.axhline(0, linestyle='--', color='k')
    plt.xlabel('t - t0 (MJD)')
    plt.ylabel('Astrometric Shift (mas)')

    fig4 = plt.figure(4)
    plt.clf()
    plt.plot(shift_n[:, 0], shift_n[:, 1], 'r-', label='No parallax')
    plt.plot(shift_p[:, 0], shift_p[:, 1], 'b-', label='Parallax')
    plt.axhline(0, linestyle='--')
    plt.axvline(0, linestyle='--')
    plt.gca().invert_xaxis()
    plt.legend()
    plt.xlabel('Shift RA (mas)')
    plt.ylabel('Shift Dec (mas)')
    plt.axis('equal')

    print 'Einstein radius: ', pspl_n.thetaE_amp, pspl_p.thetaE_amp
    print 'Einstein crossing time: ', pspl_n.tE, pspl_n.tE

    return

def test_pspl_parallax_belokurov():
    # Scenario from Belokurov and Evans 2002 (Figure 1)
    raL = 17.5
    decL = +65.0
    mL = 0.5 # msun
    t0 = 57000.00
    xS0 = np.array([0.000, 0.000])
    beta = -7.41 # mas
    muS = np.array([-2.0, 7.0])
    muL = np.array([95.4, 0.00])
    dL = 150.0
    dS = 1500.0
    imag = 19.0
    
    # No parallax
    pspl_n = model.PSPL(mL, t0, xS0, beta, muL, muS, dL, dS, imag)
    print 'pspl_n.u0', pspl_n.u0
    print 'pspl_n.muS', pspl_n.muS
    print 'pspl_n.u0_hat', pspl_n.u0_hat
    print 'pspl_n.thetaE_hat', pspl_n.thetaE_hat
    
    # With parallax
    pspl_p = model.PSPL_parallax(raL, decL, mL, t0, xS0, beta, muL, muS, dL, dS, imag)

    t = np.arange(56000, 58000, 1)
    dt = t - pspl_n.t0

    A_n = pspl_n.get_amplification(t)
    A_p = pspl_p.get_amplification(t)

    xS_n = pspl_n.get_astrometry(t)
    xS_p_unlens = pspl_p.get_astrometry_unlensed(t)
    xS_p_lensed = pspl_p.get_astrometry(t)

    # Plot the amplification
    fig1 = plt.figure(1)
    plt.clf()
    f1_1 = fig1.add_axes((0.1, 0.3, 0.8, 0.6))
    plt.plot(dt, 2.5 * np.log10(A_n), 'b-', label='No parallax')
    plt.plot(dt, 2.5 * np.log10(A_p), 'r-', label='Parallax')
    plt.legend()
    plt.ylabel('2.5 * log(A)')
    f1_1.set_xticklabels([])

    f2_1 = fig1.add_axes((0.1, 0.1, 0.8, 0.2))
    plt.plot(dt, 2.5 * (np.log10(A_p) - np.log10(A_n)), 'k-', label='Par - No par')
    plt.axhline(0, linestyle='--', color='k')
    plt.legend()
    plt.ylabel('Diff')
    plt.xlabel('t - t0 (MJD)')

    
    # Plot the positions of everything
    fig2 = plt.figure(2)
    plt.clf()
    plt.plot(xS_n[:, 0], xS_n[:, 1], 'r--', mfc='none', mec='red', label='No parallax model')
    plt.plot(xS_p_unlens[:, 0], xS_p_unlens[:, 1], 'b--', mfc='none', mec='blue',
             label='Parallax model, unlensed')
    plt.plot(xS_p_lensed[:, 0], xS_p_lensed[:, 1], 'b-', label='Parallax model, lensed')
    plt.legend()
    # plt.gca().invert_xaxis()
    # lim = 0.05
    # plt.xlim(lim, -lim) # arcsec
    # plt.ylim(-lim, lim)
    plt.xlim(0.006, -0.006) # arcsec
    plt.ylim(-0.02, 0.02)
    plt.xlabel('R.A. (")')
    plt.ylabel('Dec. (")')

    # Check just the astrometric shift part.
    shift_n = pspl_n.get_centroid_shift(t) # mas
    shift_p = (xS_p_lensed - xS_p_unlens) * 1e3 # mas
    shift_n_amp = np.linalg.norm(shift_n, axis=1)
    shift_p_amp = np.linalg.norm(shift_p, axis=1)
    
    fig3 = plt.figure(3)
    plt.clf()
    f1_3 = fig3.add_axes((0.15, 0.3, 0.8, 0.6))
    plt.plot(dt, shift_n_amp, 'r--', label='No parallax model')
    plt.plot(dt, shift_p_amp, 'b--', label='Parallax model')
    plt.legend()
    f1_3.set_xticklabels([])

    f2_3 = fig3.add_axes((0.15, 0.1, 0.8, 0.2))
    plt.plot(dt, shift_p_amp - shift_n_amp, 'k-', label='Par - No par')
    plt.legend()
    plt.axhline(0, linestyle='--', color='k')
    plt.xlabel('t - t0 (MJD)')
    plt.ylabel('Astrometric Shift (mas)')

    fig4 = plt.figure(4)
    plt.clf()
    plt.plot(shift_n[:, 0], shift_n[:, 1], 'r-', label='No parallax')
    plt.plot(shift_p[:, 0], shift_p[:, 1], 'b-', label='Parallax')
    plt.axhline(0, linestyle='--')
    plt.axvline(0, linestyle='--')
    plt.gca().invert_xaxis()
    plt.legend()
    plt.xlabel('Shift RA (mas)')
    plt.ylabel('Shift Dec (mas)')
    plt.axis('equal')

    print 'Einstein radius: ', pspl_n.thetaE_amp, pspl_p.thetaE_amp
    print 'Einstein crossing time: ', pspl_n.tE, pspl_n.tE

    return

def test_pspl_parallax_paczynski1998(t0):
    """
    I can't quite get this one to match!!! Why not? Maybe they kept in the parallax of the source?
    i.e. just removed proper motions. 
    """
    # Scenarios from Paczynski 1998
    raL = 80.89375   # LMC R.A. 
    raL = 240.0   # LMC R.A. 
    # decL = -69.75611 # LMC Dec.
    decL = -71.74 # LMC Dec. This is the sin \beta = -0.99 where \beta = 
    mL = 0.3 # msun
    # t0 = 57000.00
    xS0 = np.array([0.000, 0.088e-3]) # arcsec
    beta = 0.088 # mas
    # muS = np.array([-3.18, -0.28])
    # muL = np.array([0.0, 0.0])
    muS = np.array([-4.18, -0.28])
    muL = np.array([0.0, 0.0])
    # muS = np.array([-2.4, -0.00000001])
    # muL = np.array([0.0, 0.0])
    dL = 10e3  # 10 kpc
    dS = 50e3  # 50 kpc in LMC
    imag = 19.0
    
    # No parallax
    pspl_n = model.PSPL(mL, t0, xS0, beta, muL, muS, dL, dS, imag)
    print 'pspl_n.u0', pspl_n.u0
    print 'pspl_n.muS', pspl_n.muS
    print 'pspl_n.u0_hat', pspl_n.u0_hat
    print 'pspl_n.thetaE_hat', pspl_n.thetaE_hat
    
    # With parallax
    pspl_p = model.PSPL_parallax(raL, decL, mL, t0, xS0, beta, muL, muS, dL, dS, imag)

    #t = np.arange(56000, 58000, 1)
    t = np.arange(t0 - 500, t0 + 500, 1)
    dt = t - pspl_n.t0

    A_n = pspl_n.get_amplification(t)
    A_p = pspl_p.get_amplification(t)

    xS_n = pspl_n.get_astrometry(t)
    xS_p_unlens = pspl_p.get_astrometry_unlensed(t)
    xS_p_lensed = pspl_p.get_astrometry(t)
    xL_p_unlens = pspl_p.get_lens_astrometry(t)

    thetaS = (xS_p_unlens - xL_p_unlens) * 1e3  # mas
    u = thetaS / pspl_p.tE
    thetaS_lensed = (xS_p_lensed - xL_p_unlens) * 1e3  # mas
    

    shift_n = pspl_n.get_centroid_shift(t) # mas
    shift_p = (xS_p_lensed - xS_p_unlens) * 1e3 # mas
    shift_n_amp = np.linalg.norm(shift_n, axis=1)
    shift_p_amp = np.linalg.norm(shift_p, axis=1)
    
    # Plot the amplification
    fig1 = plt.figure(1)
    plt.clf()
    f1_1 = fig1.add_axes((0.1, 0.3, 0.8, 0.6))
    plt.plot(dt, 2.5 * np.log10(A_n), 'b-', label='No parallax')
    plt.plot(dt, 2.5 * np.log10(A_p), 'r-', label='Parallax')
    plt.legend()
    plt.ylabel('2.5 * log(A)')
    f1_1.set_xticklabels([])

    f2_1 = fig1.add_axes((0.1, 0.1, 0.8, 0.2))
    plt.plot(dt, 2.5 * (np.log10(A_p) - np.log10(A_n)), 'k-', label='Par - No par')
    plt.axhline(0, linestyle='--', color='k')
    plt.legend()
    plt.ylabel('Diff')
    plt.xlabel('t - t0 (MJD)')

    idx = np.argmin(np.abs(t - t0))
    
    # Plot the positions of everything
    fig2 = plt.figure(2)
    plt.clf()
    plt.plot(xS_n[:, 0], xS_n[:, 1], 'r--', mfc='none', mec='red', label='No parallax model')
    plt.plot(xS_p_unlens[:, 0], xS_p_unlens[:, 1], 'b--', mfc='blue', mec='blue',
             label='Parallax model, unlensed')
    plt.plot(xS_p_lensed[:, 0], xS_p_lensed[:, 1], 'b-', label='Parallax model, lensed')
    plt.plot(xL_p_unlens[:, 0], xL_p_unlens[:, 1], 'g--', mfc='none', mec='green',
             label='Parallax model, Lens')
    plt.plot(xS_n[idx, 0], xS_n[idx, 1], 'rx')
    plt.plot(xS_p_unlens[idx, 0], xS_p_unlens[idx, 1], 'bx')
    plt.plot(xS_p_lensed[idx, 0], xS_p_lensed[idx, 1], 'bx')
    plt.plot(xL_p_unlens[idx, 0], xL_p_unlens[idx, 1], 'gx')
    plt.legend()
    plt.gca().invert_xaxis()
    # lim = 0.05
    # plt.xlim(lim, -lim) # arcsec
    # plt.ylim(-lim, lim)
    # plt.xlim(0.006, -0.006) # arcsec
    # plt.ylim(-0.02, 0.02)
    plt.xlabel('R.A. (")')
    plt.ylabel('Dec. (")')

    # Check just the astrometric shift part.
    fig3 = plt.figure(3)
    plt.clf()
    f1_3 = fig3.add_axes((0.2, 0.3, 0.7, 0.6))
    plt.plot(dt, shift_n_amp, 'r--', label='No parallax model')
    plt.plot(dt, shift_p_amp, 'b--', label='Parallax model')
    plt.legend(fontsize=10)
    plt.ylabel('Astrometric Shift (mas)')
    f1_3.set_xticklabels([])

    f2_3 = fig3.add_axes((0.2, 0.1, 0.7, 0.2))
    plt.plot(dt, shift_p_amp - shift_n_amp, 'k-', label='Par - No par')
    plt.legend()
    plt.axhline(0, linestyle='--', color='k')
    plt.xlabel('t - t0 (MJD)')
    plt.ylabel('Res.')

    fig4 = plt.figure(4)
    plt.clf()
    plt.plot(shift_n[:, 0], shift_n[:, 1], 'r-', label='No parallax')
    plt.plot(shift_p[:, 0], shift_p[:, 1], 'b-', label='Parallax')
    plt.axhline(0, linestyle='--')
    plt.axvline(0, linestyle='--')
    plt.gca().invert_xaxis()
    plt.legend(loc='upper left')
    plt.xlabel('Shift RA (mas)')
    plt.ylabel('Shift Dec (mas)')
    plt.axis('equal')

    plt.figure(5)
    plt.clf()
    plt.plot(thetaS[:, 0], shift_p[:, 0], 'r-', label='RA')
    plt.plot(thetaS[:, 1], shift_p[:, 1], 'b-', label='Dec')
    plt.xlabel('thetaS (")')
    plt.ylabel('Shift (mas)')

    plt.figure(6)
    plt.clf()
    plt.plot(thetaS[:, 0], thetaS[:, 1], 'r-', label='Unlensed')
    plt.plot(thetaS_lensed[:, 0], thetaS_lensed[:, 1], 'b-', label='Lensed')
    plt.axvline(0, linestyle='--', color='k')
    plt.legend()
    plt.xlabel('thetaS_E (")')
    plt.ylabel('thetaS_N (")')

    print 'Einstein radius: ', pspl_n.thetaE_amp, pspl_p.thetaE_amp
    print 'Einstein crossing time: ', pspl_n.tE, pspl_n.tE

    return


def test_pspl_parallax_boden1998(t0):
    """
    I can get this one to match Figure 6 of Boden et al. 1998.
    """
    
    # Scenarios from Paczynski 1998
    raL = 80.89375   # LMC R.A.
    decL = -71.74 # LMC Dec. This is the sin \beta = -0.99 where \beta = 
    mL = 0.1 # msun
    xS0 = np.array([0.000, 0.088e-3]) # arcsec
    beta = -0.16 # mas  same as p=0.4
    muS = np.array([-2.0, 1.5])
    muL = np.array([0.0, 0.0])
    dL = 8e3  # 10 kpc
    dS = 50e3  # 50 kpc in LMC
    imag = 19.0
    
    # No parallax
    pspl_n = model.PSPL(mL, t0, xS0, beta, muL, muS, dL, dS, imag)
    print 'pspl_n.u0', pspl_n.u0
    print 'pspl_n.muS', pspl_n.muS
    print 'pspl_n.u0_hat', pspl_n.u0_hat
    print 'pspl_n.thetaE_hat', pspl_n.thetaE_hat
    
    # With parallax
    pspl_p = model.PSPL_parallax(raL, decL, mL, t0, xS0, beta, muL, muS, dL, dS, imag)

    #t = np.arange(56000, 58000, 1)
    t = np.arange(t0 - 500, t0 + 500, 1)
    dt = t - pspl_n.t0

    A_n = pspl_n.get_amplification(t)
    A_p = pspl_p.get_amplification(t)

    xS_n = pspl_n.get_astrometry(t)
    xS_p_unlens = pspl_p.get_astrometry_unlensed(t)
    xS_p_lensed = pspl_p.get_astrometry(t)
    xL_p_unlens = pspl_p.get_lens_astrometry(t)

    thetaS = (xS_p_unlens - xL_p_unlens) * 1e3  # mas
    u = thetaS / pspl_p.tE
    thetaS_lensed = (xS_p_lensed - xL_p_unlens) * 1e3  # mas
    

    shift_n = pspl_n.get_centroid_shift(t) # mas
    shift_p = (xS_p_lensed - xS_p_unlens) * 1e3 # mas
    shift_n_amp = np.linalg.norm(shift_n, axis=1)
    shift_p_amp = np.linalg.norm(shift_p, axis=1)
    
    # Plot the amplification
    fig1 = plt.figure(1)
    plt.clf()
    f1_1 = fig1.add_axes((0.1, 0.3, 0.8, 0.6))
    plt.plot(dt, 2.5 * np.log10(A_n), 'b-', label='No parallax')
    plt.plot(dt, 2.5 * np.log10(A_p), 'r-', label='Parallax')
    plt.legend()
    plt.ylabel('2.5 * log(A)')
    f1_1.set_xticklabels([])

    f2_1 = fig1.add_axes((0.1, 0.1, 0.8, 0.2))
    plt.plot(dt, 2.5 * (np.log10(A_p) - np.log10(A_n)), 'k-', label='Par - No par')
    plt.axhline(0, linestyle='--', color='k')
    plt.legend()
    plt.ylabel('Diff')
    plt.xlabel('t - t0 (MJD)')

    idx = np.argmin(np.abs(t - t0))
    
    # Plot the positions of everything
    fig2 = plt.figure(2)
    plt.clf()
    plt.plot(xS_n[:, 0], xS_n[:, 1], 'r--', mfc='none', mec='red', label='No parallax model')
    plt.plot(xS_p_unlens[:, 0], xS_p_unlens[:, 1], 'b--', mfc='blue', mec='blue',
             label='Parallax model, unlensed')
    plt.plot(xS_p_lensed[:, 0], xS_p_lensed[:, 1], 'b-', label='Parallax model, lensed')
    plt.plot(xL_p_unlens[:, 0], xL_p_unlens[:, 1], 'g--', mfc='none', mec='green',
             label='Parallax model, Lens')
    plt.plot(xS_n[idx, 0], xS_n[idx, 1], 'rx')
    plt.plot(xS_p_unlens[idx, 0], xS_p_unlens[idx, 1], 'bx')
    plt.plot(xS_p_lensed[idx, 0], xS_p_lensed[idx, 1], 'bx')
    plt.plot(xL_p_unlens[idx, 0], xL_p_unlens[idx, 1], 'gx')
    plt.legend()
    plt.gca().invert_xaxis()
    # lim = 0.05
    # plt.xlim(lim, -lim) # arcsec
    # plt.ylim(-lim, lim)
    # plt.xlim(0.006, -0.006) # arcsec
    # plt.ylim(-0.02, 0.02)
    plt.xlabel('R.A. (")')
    plt.ylabel('Dec. (")')

    # Check just the astrometric shift part.
    fig3 = plt.figure(3)
    plt.clf()
    f1_3 = fig3.add_axes((0.2, 0.3, 0.7, 0.6))
    plt.plot(dt, shift_n_amp, 'r--', label='No parallax model')
    plt.plot(dt, shift_p_amp, 'b--', label='Parallax model')
    plt.legend(fontsize=10)
    plt.ylabel('Astrometric Shift (mas)')
    f1_3.set_xticklabels([])

    f2_3 = fig3.add_axes((0.2, 0.1, 0.7, 0.2))
    plt.plot(dt, shift_p_amp - shift_n_amp, 'k-', label='Par - No par')
    plt.legend()
    plt.axhline(0, linestyle='--', color='k')
    plt.xlabel('t - t0 (MJD)')
    plt.ylabel('Res.')

    fig4 = plt.figure(4)
    plt.clf()
    plt.plot(shift_n[:, 0], shift_n[:, 1], 'r-', label='No parallax')
    plt.plot(shift_p[:, 0], shift_p[:, 1], 'b-', label='Parallax')
    plt.axhline(0, linestyle='--')
    plt.axvline(0, linestyle='--')
    plt.gca().invert_xaxis()
    plt.legend(loc='upper left')
    plt.xlabel('Shift RA (mas)')
    plt.ylabel('Shift Dec (mas)')
    plt.axis('equal')

    plt.figure(5)
    plt.clf()
    plt.plot(thetaS[:, 0], shift_p[:, 0], 'r-', label='RA')
    plt.plot(thetaS[:, 1], shift_p[:, 1], 'b-', label='Dec')
    plt.xlabel('thetaS (")')
    plt.ylabel('Shift (mas)')

    plt.figure(6)
    plt.clf()
    plt.plot(thetaS[:, 0], thetaS[:, 1], 'r-', label='Unlensed')
    plt.plot(thetaS_lensed[:, 0], thetaS_lensed[:, 1], 'b-', label='Lensed')
    plt.axvline(0, linestyle='--', color='k')
    plt.legend()
    plt.xlabel('thetaS_E (")')
    plt.ylabel('thetaS_N (")')

    print 'Einstein radius: ', pspl_n.thetaE_amp, pspl_p.thetaE_amp
    print 'Einstein crossing time: ', pspl_n.tE, pspl_n.tE

    return


def test_pspl_parallax_fit():    
    data, p_in = fake_data_parallax()

#     model_fitter.multinest_pspl_parallax(data,
# test                                         n_live_points=300,
#                                          saveto='./mnest_pspl_par/',
#                                          runcode='aa')
# 
    model_fitter.plot_posteriors('mnest_pspl_par/', 'aa')

    best = model_fitter.get_best_fit('mnest_pspl_par/', 'aa')

    
    pspl_out = model.PSPL_parallax(p_in['raL'], p_in['decL'],
                                   best['mL'], best['t0'],
                                   np.array([best['xS0_E'], best['xS0_N']]),
                                   best['beta'], 
                                   np.array([best['muL_E'], best['muL_N']]),
                                   np.array([best['muS_E'], best['muS_N']]),
                                   best['dL'], best['dS'], best['imag_base'])

    pspl_in = model.PSPL_parallax(p_in['raL'], p_in['decL'],
                                  p_in['mL'], p_in['t0'],
                                  np.array([p_in['xS0_E'], p_in['xS0_N']]), p_in['beta'], 
                                  np.array([p_in['muL_E'], p_in['muL_N']]),
                                  np.array([p_in['muS_E'], p_in['muS_N']]),
                                  p_in['dL'], p_in['dS'], p_in['imag_base'])
    
    imag_out = pspl_out.get_photometry(data['t_phot'])
    pos_out = pspl_out.get_astrometry(data['t_ast'])

    imag_in = pspl_in.get_photometry(data['t_phot'])
    pos_in = pspl_in.get_astrometry(data['t_ast'])
    
    lnL_phot_out = pspl_out.likely_photometry(data['t_phot'], data['imag'], data['imag_err'])
    lnL_ast_out = pspl_out.likely_astrometry(data['t_ast'], data['xpos'], data['ypos'],
                                             data['xpos_err'], data['ypos_err'])
    lnL_out = lnL_phot_out.mean() + lnL_ast_out.mean()

    lnL_phot_in = pspl_in.likely_photometry(data['t_phot'], data['imag'], data['imag_err'])
    lnL_ast_in = pspl_in.likely_astrometry(data['t_ast'], data['xpos'], data['ypos'],
                                             data['xpos_err'], data['ypos_err'])
    lnL_in = lnL_phot_in.mean() + lnL_ast_in.mean()
    
    print 'lnL for input: ', lnL_in
    print 'lnL for output: ', lnL_out

    outroot = 'mnest_pspl_par/plots/aa'
    
    plt.figure(1)
    plt.clf()
    plt.errorbar(data['t_phot'], data['imag'], yerr=data['imag_err'], fmt='k.')
    plt.plot(data['t_phot'], imag_out, 'r-')
    plt.plot(data['t_phot'], imag_in, 'g-')
    plt.xlabel('t - t0 (days)')
    plt.ylabel('I (mag)')
    plt.title('Input Data and Output Model')
    plt.savefig(outroot + '_phot.png')

    plt.figure(2)
    plt.clf()
    plt.errorbar(data['xpos'], data['ypos'], xerr=data['xpos_err'], yerr=data['ypos_err'], fmt='k.')
    plt.plot(pos_out[:, 0], pos_out[:, 1], 'r-')
    plt.plot(pos_in[:, 0], pos_in[:, 1], 'g-')
    plt.gca().invert_xaxis()
    plt.xlabel('X Pos (")')
    plt.ylabel('Y Pos (")')
    plt.title('Input Data and Output Model')
    plt.savefig(outroot + '_ast.png')

    plt.figure(3)
    plt.clf()
    plt.errorbar(data['t_ast'], data['xpos'], yerr=data['xpos_err'], fmt='k.')
    plt.plot(data['t_ast'], pos_out[:, 0], 'r-')
    plt.plot(data['t_ast'], pos_in[:, 0], 'g-')
    plt.xlabel('t - t0 (days)')
    plt.ylabel('X Pos (")')
    plt.title('Input Data and Output Model')
    plt.savefig(outroot + '_t_vs_E.png')

    plt.figure(4)
    plt.clf()
    plt.errorbar(data['t_ast'], data['ypos'], yerr=data['ypos_err'], fmt='k.')
    plt.plot(data['t_ast'], pos_out[:, 1], 'r-')
    plt.plot(data['t_ast'], pos_in[:, 1], 'g-')
    plt.xlabel('t - t0 (days)')
    plt.ylabel('Y Pos (")')
    plt.title('Input Data and Output Model')
    plt.savefig(outroot + '_t_vs_N.png')
    
    return

def test_make_t0_gen():
    data = fake_data1()

    t0_gen = model_fitter.make_t0_gen(data['t_phot'], data['imag'])

    t0_rand = t0_gen.rvs(size=100)

    plt.clf()
    plt.plot(data['t_phot'], data['imag'], 'k.')
    plt.axvline(t0_rand.min())
    plt.axvline(t0_rand.max())
    print 't0 between: ', t0_rand.min(), t0_rand.max()

    assert t0_rand.min() < 56990
    assert t0_rand.max() > 57000

    return

def fake_data_parallax():
    raL_in = 80.89375   # LMC R.A.
    decL_in = -29.0 # LMC Dec. This is the sin \beta = -0.99 where \beta = ecliptic lat
    mL_in = 10.0 # msun
    t0_in = 57000.0
    xS0_in = np.array([0.000, 0.088e-3]) # arcsec
    beta_in = 2.0 # mas  same as p=0.4
    # muS_in = np.array([-2.0, 1.5])
    # muL_in = np.array([0.0, 0.0])
    muS_in = np.array([-5.0, 0.0])
    muL_in = np.array([0.0, 0.0])
    dL_in = 4000.0  # pc
    dS_in = 8000.0  # pc
    imag_in = 19.0

    pspl_in = model.PSPL_parallax(raL_in, decL_in, mL_in,
                                  t0_in, xS0_in, beta_in,
                                  muL_in, muS_in, dL_in, dS_in, imag_in)

    # Simulate
    # photometric observations every 1 day and
    # astrometric observations every 14 days
    # for the bulge observing window. Observations missed
    # for 125 days out of 365 days for photometry and missed
    # for 245 days out of 365 days for astrometry.
    t_phot = np.array([], dtype=float)
    t_ast = np.array([], dtype=float)
    for year_start in np.arange(56000, 58000, 365.25):
        phot_win = 240.0
        phot_start = (365.25 - phot_win) / 2.0
        t_phot_new = np.arange(year_start + phot_start, year_start + phot_start + phot_win, 1)
        t_phot = np.concatenate([t_phot, t_phot_new])

        ast_win = 120.0
        ast_start = (365.25 - ast_win) / 2.0
        t_ast_new = np.arange(year_start + ast_start, year_start + ast_start + ast_win, 14)
        t_ast = np.concatenate([t_ast, t_ast_new])

    
    # Make the photometric observations.
    # Assume 0.05 mag photoemtric errors at I=19.
    # This means Signal = 400 e- at I=19.
    flux0 = 4000.0
    imag0 = 19.0
    imag_obs = pspl_in.get_photometry(t_phot)
    flux_obs = flux0 * 10**((imag_obs - imag0) / -2.5)
    flux_obs_err = flux_obs**0.5
    flux_obs += np.random.randn(len(t_phot)) * flux_obs_err
    imag_obs = -2.5 * np.log10(flux_obs / flux0) + imag0
    imag_obs_err = 1.087 / flux_obs_err

    # Make the astrometric observations.
    # Assume 0.15 milli-arcsec astrometric errors in each direction at all epochs. 
    pos_obs_tmp = pspl_in.get_astrometry(t_ast)
    pos_obs_err = np.ones((len(t_ast), 2), dtype=float) * 0.01 * 1e-3
    pos_obs = pos_obs_tmp + pos_obs_err * np.random.randn(len(t_ast), 2)

    plt.figure(1)
    plt.clf()
    plt.errorbar(t_phot, imag_obs, yerr=imag_obs_err, fmt='k.')
    plt.xlabel('t - t0 (days)')
    plt.ylabel('I (mag)')
    plt.title('Input Data and Model')

    plt.figure(2)
    plt.clf()
    plt.errorbar(pos_obs[:,0], pos_obs[:,1], xerr=pos_obs_err[:, 0], yerr=pos_obs_err[:, 1], fmt='k.')
    plt.gca().invert_xaxis()
    plt.xlabel('X Pos (")')
    plt.ylabel('Y Pos (")')
    plt.plot(pos_obs_tmp[:, 0], pos_obs_tmp[:, 1], 'r--')
    plt.title('Input Data and Model')

    plt.figure(3)
    plt.clf()
    plt.errorbar(t_ast, pos_obs[:, 0], yerr=pos_obs_err[:, 0], fmt='k.')
    plt.plot(t_ast, pos_obs_tmp[:, 0], 'r--')
    plt.xlabel('t - t0 (days)')
    plt.ylabel('X Pos (")')
    plt.title('Input Data and Model')

    plt.figure(4)
    plt.clf()
    plt.errorbar(t_ast, pos_obs[:, 1], yerr=pos_obs_err[:, 1], fmt='k.')
    plt.plot(t_ast, pos_obs_tmp[:, 1], 'r--')
    plt.xlabel('t - t0 (days)')
    plt.ylabel('Y Pos (")')
    plt.title('Input Data and Model')

    data = {}
    data['t_phot'] = t_phot
    data['imag'] = imag_obs
    data['imag_err'] = imag_obs_err

    data['t_ast'] = t_ast
    data['xpos'] = pos_obs[:, 0]
    data['ypos'] = pos_obs[:, 1]
    data['xpos_err'] = pos_obs_err[:, 0]
    data['ypos_err'] = pos_obs_err[:, 1]
    data['raL'] = raL_in
    data['decL'] = decL_in

    params = {}
    params['raL'] = raL_in
    params['decL'] = decL_in
    params['mL'] = mL_in
    params['t0'] = t0_in
    params['xS0_E'] = xS0_in[0]
    params['xS0_N'] = xS0_in[1]
    params['beta'] = beta_in
    params['muS_E'] = muS_in[0]
    params['muS_N'] = muS_in[1]
    params['muL_E'] = muL_in[0]
    params['muL_N'] = muL_in[1]
    params['dL'] = dL_in
    params['dS'] = dS_in
    params['imag_base'] = imag_in
    
            
    return data, params
    

def fake_data1():
    
    # Input parameters
    mL_in = 10.0 # msun
    t0_in = 57000.00
    xS0_in = np.array([0.000, 0.000])
    beta_in = -0.4 # Einstein radii
    muL_in = np.array([-0.0, -7.0])  # Strong
    #muL_in = np.array([-7.0, 0.0])  # Weak
    muS_in = np.array([1.5, -0.5])  # mas/yr
    dL_in = 4000.0
    dS_in = 8000.0
    imag_in = 19.0
    
    pspl_in = model.PSPL(mL_in, t0_in, xS0_in, beta_in, muL_in, muS_in, dL_in, dS_in, imag_in)

    # Simulate
    # photometric observations every 1 day and
    # astrometric observations every 14 days
    # for the bulge observing window. Observations missed
    # for 125 days out of 365 days for photometry and missed
    # for 245 days out of 365 days for astrometry.
    t_phot = np.array([], dtype=float)
    t_ast = np.array([], dtype=float)
    for year_start in np.arange(54000, 60000, 365.25):
        phot_win = 240.0
        phot_start = (365.25 - phot_win) / 2.0
        t_phot_new = np.arange(year_start + phot_start, year_start + phot_start + phot_win, 1)
        t_phot = np.concatenate([t_phot, t_phot_new])

        ast_win = 120.0
        ast_start = (365.25 - ast_win) / 2.0
        t_ast_new = np.arange(year_start + ast_start, year_start + ast_start + ast_win, 14)
        t_ast = np.concatenate([t_ast, t_ast_new])

    
    A = pspl_in.get_amplification(t_phot)
    shift = pspl_in.get_centroid_shift(t_ast)

    dt_phot = t_phot - pspl_in.t0
    dt_ast = t_ast - pspl_in.t0

    # Make the photometric observations.
    # Assume 0.05 mag photoemtric errors at I=19.
    # This means Signal = 400 e- at I=19.
    flux0 = 400.0
    imag0 = 19.0
    # flux_in = flux0 * 10**((imag_in - imag0) / -2.5)
    # flux_obs = flux_in * A
    # flux_obs_err = flux_obs**0.5
    # flux_obs += np.random.randn(len(t_phot)) * flux_obs_err
    # imag_obs = -2.5 * np.log10(flux_obs / flux0) + imag0
    # imag_obs_err = 1.087 / flux_obs_err

    imag_obs = pspl_in.get_photometry(t_phot)
    flux_obs = flux0 * 10**((imag_obs - imag0) / -2.5)
    flux_obs_err = flux_obs**0.5
    flux_obs += np.random.randn(len(t_phot)) * flux_obs_err
    imag_obs = -2.5 * np.log10(flux_obs / flux0) + imag0
    imag_obs_err = 1.087 / flux_obs_err

    # Make the astrometric observations.
    # Assume 0.15 milli-arcsec astrometric errors in each direction at all epochs. 
    lens_pos_in = pspl_in.xL0 + np.outer(dt_ast / model.days_per_year, pspl_in.muL) * 1e-3
    srce_pos_in = pspl_in.xS0 + np.outer(dt_ast / model.days_per_year, pspl_in.muS) * 1e-3
    pos_obs_tmp = pspl_in.get_astrometry(t_ast)  #  srce_pos_in + (shift * 1e-3)
    pos_obs_err = np.ones((len(t_ast), 2), dtype=float) * 0.15 * 1e-3
    pos_obs = pos_obs_tmp + pos_obs_err * np.random.randn(len(t_ast), 2)

    plt.figure(1)
    plt.clf()
    plt.errorbar(t_phot, imag_obs, yerr=imag_obs_err, fmt='k.')
    plt.xlabel('t - t0 (days)')
    plt.ylabel('I (mag)')
    plt.title('Input Data and Model')

    plt.figure(2)
    plt.clf()
    plt.errorbar(pos_obs[:,0], pos_obs[:,1], xerr=pos_obs_err[:, 0], yerr=pos_obs_err[:, 1], fmt='k.')
    plt.gca().invert_xaxis()
    plt.xlabel('X Pos (")')
    plt.ylabel('Y Pos (")')
    plt.plot(srce_pos_in[:, 0], srce_pos_in[:, 1], 'k--')
    plt.plot(pos_obs_tmp[:, 0], pos_obs_tmp[:, 1], 'r--')
    plt.title('Input Data and Model')

    plt.figure(3)
    plt.clf()
    plt.errorbar(t_ast, pos_obs[:, 0], yerr=pos_obs_err[:, 0], fmt='k.')
    plt.plot(t_ast, srce_pos_in[:, 0], 'k--')
    plt.plot(t_ast, pos_obs_tmp[:, 0], 'r--')
    plt.xlabel('t - t0 (days)')
    plt.ylabel('X Pos (")')
    plt.title('Input Data and Model')

    plt.figure(4)
    plt.clf()
    plt.errorbar(t_ast, pos_obs[:, 1], yerr=pos_obs_err[:, 1], fmt='k.')
    plt.plot(t_ast, srce_pos_in[:, 1], 'k--')
    plt.plot(t_ast, pos_obs_tmp[:, 1], 'r--')
    plt.xlabel('t - t0 (days)')
    plt.ylabel('Y Pos (")')
    plt.title('Input Data and Model')

    data = {}
    data['t_phot'] = t_phot
    data['imag'] = imag_obs
    data['imag_err'] = imag_obs_err

    data['t_ast'] = t_ast
    data['xpos'] = pos_obs[:, 0]
    data['ypos'] = pos_obs[:, 1]
    data['xpos_err'] = pos_obs_err[:, 0]
    data['ypos_err'] = pos_obs_err[:, 1]

    params = {}
    params['mL'] = mL_in
    params['t0'] = t0_in
    params['xS0_E'] = xS0_in[0]
    params['xS0_N'] = xS0_in[1]
    params['beta'] = beta_in
    params['muS_E'] = muS_in[0]
    params['muS_N'] = muS_in[1]
    params['muL_E'] = muL_in[0]
    params['muL_N'] = muL_in[1]
    params['dL'] = dL_in
    params['dS'] = dS_in
    params['imag_base'] = imag_in
    
            
    return data, params
    
