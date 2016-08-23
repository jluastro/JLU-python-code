from jlu.microlens import model
from jlu.microlens import model_fitter
import numpy as np
import pylab as plt
import pdb

def test_PSPL():
    mL = 10.0 # msun
    t0 = 57000.00
    xS0 = np.array([0.000, 0.000])
    beta = -0.4 # mas
    muS = np.array([1.5, -0.5])
    muL = np.array([-7.0, 0.00])
    dL = 4000.0
    dS = 8000.0
    imag = 19.0
    
    pspl = model.PSPL(mL, t0, xS0, beta, muL, muS, dL, dS, imag)

    t = np.arange(54000, 60000, 1)
    dt = t - pspl.t0

    A = pspl.get_amplification(t)
    shift = pspl.get_centroid_shift(t)

    # Plot the amplification
    plt.figure(1)
    plt.clf()
    plt.plot(dt, 2.5 * np.log10(A), 'k.')
    plt.xlabel('t - t0 (MJD)')

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

    plt.figure(3)
    plt.clf()
    plt.plot(lens_pos[:, 0], lens_pos[:, 1], 'r--', mfc='none', mec='red')
    plt.plot(srce_pos[:, 0], srce_pos[:, 1], 'b--', mfc='none', mec='blue')
    plt.plot(imag_pos[:, 0], imag_pos[:, 1], 'b-')
    lim = 0.05
    plt.xlim(lim, -lim) # arcsec
    plt.ylim(-lim, lim)
    
    plt.figure(4)
    plt.clf()
    plt.plot(dt, np.linalg.norm(shift, axis=1))
    plt.xlabel('t - t0 (MJD)')

    print 'Einstein radius: ', pspl.thetaE_amp
    print 'Einstein crossing time: ', pspl.tE

    return pspl    


def test_pspl_fit():
    data = fake_data1()

    
    
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

def fake_data1():
    
    # Input parameters
    mL_in = 10.0 # msun
    t0_in = 57000.00
    xS0_in = np.array([0.000, 0.000])
    beta_in = -0.4 # mas
    muS_in = np.array([1.5, -0.5])
    muL_in = np.array([-0.0, -7.0])  # Strong
    #muL_in = np.array([-7.0, 0.0])  # Weak
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
    flux_in = flux0 * 10**((imag_in - imag0) / -2.5)
    flux_obs = flux_in * A
    flux_obs_err = flux_obs**0.5
    flux_obs += np.random.randn(len(t_phot)) * flux_obs_err
    imag_obs = -2.5 * np.log10(flux_obs / flux0) + imag0
    imag_obs_err = 1.087 / flux_obs_err

    # Make the astrometric observations.
    # Assume 0.15 milli-arcsec astrometric errors in each direction at all epochs. 
    lens_pos_in = pspl_in.xL0 + np.outer(dt_ast / model.days_per_year, pspl_in.muL) * 1e-3
    srce_pos_in = pspl_in.xS0 + np.outer(dt_ast / model.days_per_year, pspl_in.muS) * 1e-3
    pos_obs_tmp = srce_pos_in + (shift * 1e-3)
    pos_obs = pos_obs_tmp + np.random.randn(len(t_ast) * 2).reshape((len(t_ast), 2)) * 0.15 * 1e-3
    pos_obs_err = np.ones(pos_obs.shape) * 0.15 * 1e-3

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
            
    return data
    
