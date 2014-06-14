import numpy as np
import pylab as py
import pyfits
import glob
import os
from jlu.util import img_scale
import atpy
import math
from jlu.util import statsIter
from scipy import interpolate as interp
from jlu.gc.gcwork import starset

ngc1851_data = '/u/jlu/data/gsaoi/commission/reduce/ngc1851/'

def plot_spie_figure3():
    """
    Pass in a starlist (stack of starlists) and calculate the
    astrometric and photometric errors per star. Then plot them.
    """
    align_dir = '/g/lu/data/gsaoi/commission/reduce/ngc1851/align_2013_03_04_G2/'
    plot_dir = '/u/jlu/doc/papers/proceed_2014_spie/'
    
    from jlu.gsaoi import stars as stars_obj
    stars2 = stars_obj.Starlist(align_dir + 'align_a2')
    stars3 = stars_obj.Starlist(align_dir + 'align_a3')
    stars4 = stars_obj.Starlist(align_dir + 'align_a4')

    stars_all = [stars2, stars3, stars4]

    py.close('all')
    f, ax_all = py.subplots(1, 3, sharex=True, sharey=True, figsize=(12, 4))
    f.subplots_adjust(left=0.10, bottom=0.18, wspace=0.01)

    for ii in range(len(stars_all)):
        stars = stars_all[ii]
        ax = ax_all[ii]
        
        Nstars = stars.dxm.shape[1]
        Nimages = stars.dxm.shape[0]
    
        mavg = stars.mm.mean(axis=0)

        # Error on the mean
        xerr = stars.dxm.std(axis=0) / math.sqrt(Nimages)
        yerr = stars.dym.std(axis=0) / math.sqrt(Nimages)
        merr = stars.mm.std(axis=0) / math.sqrt(Nimages)

        # Convert to mas
        xerr *= 1e3
        yerr *= 1e3

        rerr = np.hypot(xerr, yerr)

        xerr_med = np.median(xerr)
        yerr_med = np.median(yerr)
        err_med = np.median([xerr_med, yerr_med])
        rerr_med = np.median(np.hypot(xerr, yerr))

        # Make a curve of the median error
        magBinCent = np.arange(10, 13.5, 0.05)
        magBinSize = 0.5
        rerrMed = np.zeros(len(magBinCent), dtype=float)
        rerrAvg = np.zeros(len(magBinCent), dtype=float)
        rerrStd = np.zeros(len(magBinCent), dtype=float)
        for ii in range(len(magBinCent)):
            magLo = magBinCent[ii] - (magBinSize / 2.0)
            magHi = magBinCent[ii] + (magBinSize / 2.0)
            idx = np.where((mavg > magLo) & (mavg <= magHi))[0]
            rerrAvg[ii] = statsIter.mean(rerr[idx], hsigma=2, lsigma=5, iter=10)
            rerrStd[ii] = statsIter.std(rerr[idx], hsigma=2, lsigma=5, iter=10)
            rerrMed[ii] = np.median(rerr[idx])

        # Smooth the average curve
        n_window = 20
        window = np.hanning(n_window)
        s = np.r_[rerrAvg[n_window-1:0:-1], rerrAvg, rerrAvg[-1:-n_window:-1]]
        rerrAvg_smooth = np.convolve(window/window.sum(), s, mode='valid')
        rerrAvg_smooth = rerrAvg_smooth[(n_window/2)-1:-(n_window/2)]
        
        ax.plot(mavg, rerr, 'k.', alpha=0.4)
        #ax.plot(magBinCent, rerrMed, 'k-')
        #ax.plot(magBinCent, rerrAvg, 'r-', linewidth=2)
        ax.plot(magBinCent, rerrAvg_smooth, 'r-', linewidth=2)
        # ax.plot(magBinCent, rerrAvg+rerrStd, 'r--')
        # ax.plot(magBinCent, rerrAvg-rerrStd, 'r--')
        ax.set_xlabel('K Magnitude')
        ax.set_ylim(0, 1)
        ax.set_xlim(9.1, 14)

    ax_all[0].set_ylabel('Astrometric Error (mas)')
    py.savefig(plot_dir + 'gems_ngc1851_pos_err.png')

    
def plot_gsaoi_compare_nights():
    alignDir = '/u/jlu/data/gsaoi/commission/reduce/ngc1851/'
    alignDir += 'align_G2_2_nodith/'

    alignRootAll = [alignDir + 'align_2_t', alignDir + 'align_4_t']

    py.close(1)
    py.figure(1, figsize=(12,6))
    py.subplots_adjust(left=0.1)
    py.axis('equal')
    
    for ii in range(len(alignRootAll)):
        alignRoot = alignRootAll[ii]
        
        s = starset.StarSet(alignRoot)

        x = s.getArrayFromAllEpochs('x')
        y = s.getArrayFromAllEpochs('y')
        m = s.getArrayFromAllEpochs('mag')

        x = x[1:,:]
        y = y[1:,:]
        m = m[1:,:]

        name = s.getArray('name')
        ndet = s.getArray('velCnt')

        # Set up masks to get rid of invalid values
        xm = np.ma.masked_where(x <= -1e5, x)
        ym = np.ma.masked_where(x <= -1e5, y)
        mm = np.ma.masked_where(x <= -1e5, m)

        # Read in the align.list file and figure out the dates.
        starlistsTab = atpy.Table(alignRoot + '.list', type='ascii',
                                  delimiter=' ', data_start=1)
        starlists = starlistsTab.col1

        xavg = xm.mean(axis=0)
        xstd = xm.std(axis=0) * 1e3
        xerr = xstd / math.sqrt(xm.shape[0])

        yavg = ym.mean(axis=0)
        ystd = ym.std(axis=0) * 1e3
        yerr = ystd / math.sqrt(ym.shape[0])

        dates = np.array([ss[1:9] for ss in starlists])
        uniqueDates = np.unique(dates)
        idx_n1 = np.where(dates == uniqueDates[0])[0]
        idx_n2 = np.where(dates == uniqueDates[1])[0]

        x_n1 = xm[idx_n1,:]
        y_n1 = ym[idx_n1,:]
        m_n1 = mm[idx_n1,:]

        x_n2 = xm[idx_n2,:]
        y_n2 = ym[idx_n2,:]
        m_n2 = mm[idx_n2,:]

        xavg_n1 = x_n1.mean(axis=0)
        yavg_n1 = y_n1.mean(axis=0)
        mavg_n1 = m_n1.mean(axis=0)

        xavg_n2 = x_n2.mean(axis=0)
        yavg_n2 = y_n2.mean(axis=0)
        mavg_n2 = m_n2.mean(axis=0)

        xstd_n1 = x_n1.std(axis=0) * 1e3
        ystd_n1 = y_n1.std(axis=0) * 1e3
        mstd_n1 = m_n1.std(axis=0) * 1e3

        xstd_n2 = x_n2.std(axis=0) * 1e3
        ystd_n2 = y_n2.std(axis=0) * 1e3
        mstd_n2 = m_n2.std(axis=0) * 1e3

        # Correct error to report (assuming random errors) is the error on the mean.
        xerr_n1 = xstd_n1 / math.sqrt(x_n1.shape[0])
        yerr_n1 = ystd_n1 / math.sqrt(x_n1.shape[0])
        merr_n1 = mstd_n1 / math.sqrt(x_n1.shape[0])

        xerr_n2 = xstd_n2 / math.sqrt(x_n2.shape[0])
        yerr_n2 = ystd_n2 / math.sqrt(x_n2.shape[0])
        merr_n2 = mstd_n2 / math.sqrt(x_n2.shape[0])

        dx = (xavg_n2 - xavg_n1) * 1.0e3
        dy = (yavg_n2 - yavg_n1) * 1.0e3

        py.subplot(1, 2, ii+1)
        py.errorbar(dx, dy, xerr=xerr_n1, yerr=yerr_n1, fmt='k.')
        py.plot([0], [0], 'rs', ms=8)
        py.xlabel('X Difference (mas)')
        py.ylabel('Y Difference (mas)')
        py.title('GSAOI Night 2 - Night 1')
        lim = 6
        py.axis([-lim, lim, -lim, lim])

    outFile = '/u/jlu/doc/papers/proceed_2014_spie/gems_ngc1851_comp_nights.png'
    py.savefig(outFile)


def plot_arches():
    catalog_file = '/u/jlu/work/arches/mwhosek/osiris_5_14/catalog_key1_Aks0.0.fits'

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

    py.close(1)
    py.figure(1)
    py.clf()
    py.semilogy(m, pe, 'k.', ms=2)
    py.axhline(pe_at_m, linestyle='--', color='blue', linewidth=2)
    py.text(11, pe_at_m*1.1, 'Median at \nF153M={0:d} mag'.format(at_m), color='blue')
    py.plot(at_m, pe_at_m, 'rs', ms=15, color='blue')
    py.xlabel('WFC3-IR F153M Magnitude')
    py.ylabel('Positional Error (mas)')
    py.ylim(0.0025, 25)
    py.ylim(0.01, 10)
    py.xlim(10, 21)
    py.savefig('/u/jlu/doc/papers/proceed_2014_spie/wfc3ir_arches_pos_err.png')

    py.close(2)
    py.figure(2)
    py.clf()
    py.semilogy(m, ve, 'k.', ms=2)
    py.axhline(ve_predict, linestyle='--', color='blue', linewidth=2)
    py.text(11, ve_predict*1.1, 'Predicted based\non Pos. Err. at\nF153M={0:d} mag'.format(at_m), color='blue')
    py.plot(at_m, ve_at_m, 'rs', ms=15, color='yellow')
    py.xlabel('WFC3-IR F153M Magnitude')
    py.ylabel('Proper Motion Error (mas/yr)')
    #py.ylim(0.00, 1.0)
    py.ylim(0.01, 1.0)
    py.xlim(10, 21)
    py.savefig('/u/jlu/doc/papers/proceed_2014_spie/wfc3ir_arches_pm_err.png')
    

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

    # Switch to milli-arcseconds
    keck /= 1e3
    gsaoi /= 1e3
    wfc3ir /= 1e3

    py.close(1)
    py.figure(1)
    py.clf()
    py.plot(tint_final, gsaoi, label='Gemini GSAOI')
    py.plot(tint_final, wfc3ir, label='HST WFC3IR')
    py.plot(tint_final, keck, label='Keck NIRC2')
    py.xlabel('Integration Time (min)')
    py.ylabel('Astrometric Error (mas)')
    py.legend()
    py.title('18th magnitude star')
    py.ylim(0, 2)
    py.savefig('/u/jlu/doc/papers/proceed_2014_spie/compare_ast_tint.png')
    
    # Plot errors vs. flux at tint=15
    tint_final = 15.0
    mag_final = np.arange(10, 22, 0.1)
    keck = scale_errors(keck_pos_err, keck_tint, keck_mag, keck_pos_err_floor,
                        tint_final=tint_final, mag_final=mag_final, verbose=False)
    gsaoi = scale_errors(gsaoi_pos_err, gsaoi_tint, gsaoi_mag, gsaoi_pos_err_floor,
                         tint_final=tint_final, mag_final=mag_final, verbose=False)
    wfc3ir = scale_errors(wfc3ir_pos_err, wfc3ir_tint, wfc3ir_mag, wfc3ir_pos_err_floor,
                          tint_final=tint_final, mag_final=mag_final, verbose=False)
    # Switch to milli-arcseconds
    keck /= 1e3
    gsaoi /= 1e3
    wfc3ir /= 1e3

    py.close(2)
    py.figure(2)
    py.clf()
    py.plot(mag_final, gsaoi, label='Gemini GSAOI')
    py.plot(mag_final, wfc3ir, label='HST WFC3IR')
    py.plot(mag_final, keck, label='Keck NIRC2')
    py.xlabel('Magnitude')
    py.ylabel('Astrometric Error (mas)')
    py.legend(loc='upper left')
    py.title('15 minute integration')
    py.ylim(0, 2)
    py.savefig('/u/jlu/doc/papers/proceed_2014_spie/compare_ast_mag.png')
    

    
                    
    
