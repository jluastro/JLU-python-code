import atpy
import math
import numpy as np
import pylab as py

def plot_arches():
    catalog_file = '/Users/jlu/work/arches/mwhosek/osiris_5_14/catalog_key1_Aks0.0.fits'

    cat = atpy.Table(catalog_file)

    scale = 120.0
    nexposures = 21.0
    
    # Repair the positional uncertainties so that they are the error on the mean rather than
    # the standard deviation.
    xe = scale * cat['xe_2010_f153m'] / math.sqrt(nexposures - 1.0)
    ye = scale * cat['ye_2010_f153m'] / math.sqrt(nexposures - 1.0)
    m = cat['m_2010_f153m']

    pe = (xe + ye) / 2.0

    at_m = 18
    m_min = at_m - 0.3
    m_max = at_m + 0.3
    idx = np.where((m > m_min) & (m < m_max))[0]
    print len(idx)

    pe_at_m = pe[idx].mean()

    ve = (cat['fit_vxe'] + cat['fit_vye']) / 2.0
    ve_at_m = ve[idx].mean()

    t = np.array([2010.6043, 2010.615, 2010.615,  2011.6829, 2012.6156])
    # t = np.array([2010.6043, 2011.6829, 2012.6156])
    ve_predict = predict_pm_err(t, pe_at_m) 

    print 'Median Positional Error at F153M  = {0:d} mag: {1:.2f} mas'.format( at_m, pe_at_m )
    print 'Median Velocity Error at F153M    = {0:d} mag: {1:.2f} mas/yr'.format( at_m, ve_at_m )
    print 'Predicted velocity error at F153M = {0:d} mag: {1:.2f} mas/yr'.format( at_m, ve_predict )
    
    py.figure(1)
    py.clf()
    py.semilogy(m, pe, 'k.', ms=2)
    py.axhline(pe_at_m, linestyle='--', color='blue', linewidth=2)
    py.text(11, pe_at_m*1.1, 'Median at \nF153M={0:d} mag'.format(at_m), color='blue')
    py.plot(at_m, pe_at_m, 'rs', ms=15, color='blue')
    py.xlabel('WFC3-IR F153M Magnitude')
    py.ylabel('Positional Error (mas)')
    py.ylim(0.0025, 25)
    py.xlim(10, 21)
    py.savefig('/Users/jlu/doc/papers/proceed_2014_spie/wfc3ir_arches_pos_err.png')

    py.figure(2)
    py.clf()
    py.semilogy(m, ve, 'k.', ms=2)
    py.axhline(ve_predict, linestyle='--', color='blue', linewidth=2)
    py.text(11, ve_predict*1.1, 'Predicted based\non Pos. Err. at\nF153M={0:d} mag'.format(at_m), color='blue')
    py.plot(at_m, ve_at_m, 'rs', ms=15, color='yellow')
    py.xlabel('WFC3-IR F153M Magnitude')
    py.ylabel('Proper Motion Error (mas)')
    #py.ylim(0.00, 1.0)
    py.ylim(0.01, 1.0)
    py.xlim(10, 21)
    py.savefig('/Users/jlu/doc/papers/proceed_2014_spie/wfc3ir_arches_pm_err.png')
    

def predict_pm_err(t, pos_err):
    t0 = t.mean()
    nepoch = len(t)

    dt = t - t0
    pm_err = pos_err / math.sqrt((dt**2).sum())
    #pm_err *= math.sqrt(nepoch / (nepoch - 1.0))

    return pm_err

def scale_errors(pos_err, tint, mag, pos_err_floor, tint_final=15, mag_final=18, verbose=True):
    t_scale = (tint / tint_final)**0.5

    dm = mag - mag_final
    f_ratio = 10**(-dm / 2.5)
    f_scale = f_ratio**0.5

    pos_err_new = pos_err * t_scale * f_scale

    pos_err_new = (pos_err_new**2 + pos_err_floor**2)**0.5

    if verbose:
        print 'Final Pos Error: {0:.2f} at K={1:.1f} in tint={2:.1f}'.format(pos_err_new, 
                                                                             mag_final, 
                                                                             tint_final)

    return pos_err_new

def scale_errors_keck():
    pos_err = 240.
    mag = 18.0
    tint = 30.0
    pos_err_floor = 150.0

    print 'Keck AO err: '
    scale_errors(pos_err, tint, mag, pos_err_floor)
    
def scale_errors_gsaoi():
    pos_err = 460
    mag = 14.0
    tint = 3.0
    pos_err_floor = 385.0

    print 'Keck AO err: '
    scale_errors(pos_err, tint, mag, pos_err_floor)
    

def scale_errors_wfc3ir():
    pos_err = 260
    mag = 18.0
    tint = 200.0
    pos_err_floor = 150.0

    print 'Keck AO err: '
    scale_errors(pos_err, tint, mag, pos_err_floor)

def plot_scaled_errors():
    keck_pos_err = 240.
    keck_mag = 18.0
    keck_tint = 30.0
    keck_pos_err_floor = 150.0
        
    gsaoi_pos_err = 460
    gsaoi_mag = 14.0
    gsaoi_tint = 3.0
    gsaoi_pos_err_floor = 385.0

    wfc3ir_pos_err = 260
    wfc3ir_mag = 18.0
    wfc3ir_tint = 200.0
    wfc3ir_pos_err_floor = 150.0

    # Plot errors vs. t_int at K=18.
    tint_final = np.arange(5, 60, 1.0)
    mag_final = 18.0
    keck = scale_errors(keck_pos_err, keck_tint, keck_mag, keck_pos_err_floor,
                        tint_final=tint_final, mag_final=mag_final, verbose=False)
    gsaoi = scale_errors(gsaoi_pos_err, gsaoi_tint, gsaoi_mag, gsaoi_pos_err_floor,
                         tint_final=tint_final, mag_final=mag_final, verbose=False)
    wfc3ir = scale_errors(wfc3ir_pos_err, wfc3ir_tint, wfc3ir_mag, wfc3ir_pos_err_floor,
                          tint_final=tint_final, mag_final=mag_final, verbose=False)
    py.figure(1)
    py.clf()
    py.plot(tint_final, keck, label='Keck NIRC2')
    py.plot(tint_final, gsaoi, label='Gemini GSAOI')
    py.plot(tint_final, wfc3ir, label='HST WFC3IR')
    py.xlabel('Integration Time (min)')
    py.ylabel(r'Astrometric Error ($\mu$as)')
    py.legend()
    py.title('18th magnitude star')
    py.ylim(0, 2000)
    py.savefig('/Users/jlu/doc/papers/proceed_2014_spie/compare_ast_tint.png')
    
    # Plot errors vs. flux at tint=15
    tint_final = 15.0
    mag_final = np.arange(10, 22, 0.1)
    keck = scale_errors(keck_pos_err, keck_tint, keck_mag, keck_pos_err_floor,
                        tint_final=tint_final, mag_final=mag_final, verbose=False)
    gsaoi = scale_errors(gsaoi_pos_err, gsaoi_tint, gsaoi_mag, gsaoi_pos_err_floor,
                         tint_final=tint_final, mag_final=mag_final, verbose=False)
    wfc3ir = scale_errors(wfc3ir_pos_err, wfc3ir_tint, wfc3ir_mag, wfc3ir_pos_err_floor,
                          tint_final=tint_final, mag_final=mag_final, verbose=False)
    py.figure(2)
    py.clf()
    py.plot(mag_final, keck, label='Keck NIRC2')
    py.plot(mag_final, gsaoi, label='Gemini GSAOI')
    py.plot(mag_final, wfc3ir, label='HST WFC3IR')
    py.xlabel('Magnitude')
    py.ylabel(r'Astrometric Error ($\mu$as)')
    py.legend(loc='upper left')
    py.title('15 minute integration')
    py.ylim(0, 2000)
    py.savefig('/Users/jlu/doc/papers/proceed_2014_spie/compare_ast_mag.png')
    
