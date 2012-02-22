import asciidata
from gcwork import starset
import massdimm
import numpy as np
import pylab as py
import pyfits
import pickle
from gcwork import objects

def make_all_plots():
    plot_ao_info('10may_nt1_ao_info.dat')
    plot_ao_info('10may_nt2_ao_info.dat')
    plot_ao_info('06may_ao_info.dat')
    plot_ao_info('06jun_ao_info.dat')
    plot_ao_info('06jul_ao_info.dat')
    plot_ao_info('07may_ao_info.dat')

def calc_10may():
    calc_ao_info(dataRoot='/u/ghezgroup/data/gc_new/10maylgs/clean/kp/',
                 alignFile='align_kp_0.7_named_nt1_t',     
                 mdimmRoot='/u/jlu/work/gc/ao_performance/massdimm/',
                 massFile='20100504.dimm.dat',
                 dimmFile='20100504.mass.dat',
                 pickleFile='10may_nt1_ao_info.dat')

    calc_ao_info(dataRoot='/u/ghezgroup/data/gc_new/10maylgs/clean/kp/',
                 alignFile='align_kp_0.7_named_nt2_t',     
                 mdimmRoot='/u/jlu/work/gc/ao_performance/massdimm/',
                 massFile='20100505.dimm.dat',
                 dimmFile='20100505.mass.dat',
                 pickleFile='10may_nt2_ao_info.dat')
    

def calc_06may():
    calc_ao_info(dataRoot='/u/ghezgroup/data/gc_new/06maylgs1/clean/kp/',
                 alignFile='align_kp_0.7_t',     
                 mdimmRoot='/u/jlu/work/gc/astrometry/massdimm/TMTmass_dimm/',
                 massFile='results.TMTDIMM.T6-Hawaii.20060502',
                 dimmFile='results.TMTMASS.T6-Hawaii.20060502',
                 pickleFile='06may_ao_info.dat')

def calc_06jun():
    calc_ao_info(dataRoot='/u/ghezgroup/data/gc_new/06junlgs/clean/kp/',
                 alignFile='align_kp_0.7_t',     
                 mdimmRoot='/u/jlu/work/gc/astrometry/massdimm/TMTmass_dimm/',
                 massFile='results.TMTDIMM.T6-Hawaii.20060620',
                 dimmFile='results.TMTMASS.T6-Hawaii.20060620',
                 pickleFile='06jun_ao_info.dat')

def calc_06jul():
    calc_ao_info(dataRoot='/u/ghezgroup/data/gc_new/06jullgs/clean/kp/',
                 alignFile='align_kp_0.7_t',     
                 mdimmRoot='/u/jlu/work/gc/astrometry/massdimm/TMTmass_dimm/',
                 massFile='results.TMTDIMM.T6-Hawaii.20060716',
                 dimmFile='results.TMTMASS.T6-Hawaii.20060716',
                 pickleFile='06jul_ao_info.dat')

def calc_07may():
    calc_ao_info(dataRoot='/u/ghezgroup/data/gc_new/07maylgs/clean/kp/',
                 alignFile='align_kp_0.7_t',     
                 mdimmRoot='/u/jlu/work/gc/astrometry/massdimm/TMTmass_dimm/',
                 massFile='results.TMTDIMM.T6-Hawaii.20070516',
                 dimmFile='results.TMTMASS.T6-Hawaii.20070516',
                 pickleFile='07may_ao_info.dat')

# WRONG TMT file, also two nights
# def calc_07aug():
#     calc_ao_info(dataRoot='/u/ghezgroup/data/gc_new/07auglgs/clean/kp/',
#                  alignFile='align_kp_0.7_t',     
#                  mdimmRoot='/u/jlu/work/gc/astrometry/massdimm/TMTmass_dimm/',
#                  massFile='results.TMTDIMM.T6-Hawaii.20070811',
#                  dimmFile='results.TMTMASS.T6-Hawaii.20070811',
#                  pickleFile='07aug_ao_info.dat')



def calc_ao_info(dataRoot='/u/ghezgroup/data/gc_new/06junlgs/clean/kp/',
                 alignFile='align_kp_0.7_t',     
                 mdimmRoot='/u/jlu/work/gc/astrometry/massdimm/TMTmass_dimm/',
                 massFile='results.TMTDIMM.T6-Hawaii.20060620',
                 dimmFile='results.TMTMASS.T6-Hawaii.20060620',
                 pickleFile='06jun_ao_info.dat'):

    # Load up align 
    print 'Loading align info'
    s = starset.StarSet(dataRoot + 'starfinder/align/' + alignFile)
    s.loadList()
    imgCount = len(s.years) - 1
    starCount = len(s.stars)

    # Save average x, y, r, and mag
    x = s.getArray('x')
    y = s.getArray('x')
    m = s.getArray('mag')
    r = np.hypot(x, y)

    afile = s.starlist[1:]
    for aa in range(imgCount):
        afile[aa] = afile[aa][3:8]

    # Load up image quality metrics
    print 'Loading Strehl/FWHM info'
    table = asciidata.open(dataRoot + 'irs33N.strehl')
    sfile = table[0].tonumpy()
    strehl = table[1].tonumpy()
    fwhm = table[3].tonumpy()
    mjd = table[4].tonumpy()

    for ss in range(len(sfile)):
        sfile[ss] = sfile[ss].replace('.fits', '')

    # Reorder the quality metrics to match the align order
    idx = []
    for aa in range(imgCount):
        ii = np.where(sfile == afile[aa])[0]

        if len(ii) == 0:
            print 'Failed to find ', afile[aa]
        idx.append(ii[0])

    sfile = sfile[idx]
    strehl = strehl[idx]
    fwhm = fwhm[idx]
    mjd = mjd[idx]
    
    # Load up the hour/minute/second from the image headers.
    # Also get the corrsponding MASS/DIMM data.
    hour = np.zeros(imgCount, dtype=int)
    minute = np.zeros(imgCount, dtype=int)
    second = np.zeros(imgCount, dtype=int)

    dimm = massdimm.DIMM(mdimmRoot + massFile)
    mass = massdimm.MASS(mdimmRoot + dimmFile)

    mass_seeing = np.zeros(imgCount, dtype=float)
    dimm_seeing = np.zeros(imgCount, dtype=float)
    iso_angle = np.zeros(imgCount, dtype=float)
    tau0 = np.zeros(imgCount, dtype=float)
    airmass = np.zeros(imgCount, dtype=float)
    r0 = np.zeros(imgCount, dtype=float)

    xrms = np.zeros((imgCount, starCount), dtype=float)
    yrms = np.zeros((imgCount, starCount), dtype=float)
    mrms = np.zeros((imgCount, starCount), dtype=float)

    print 'Loading MASS/DIMM and astrom/photom RMS info'
    for aa in range(imgCount):
        hdr = pyfits.getheader(dataRoot + afile[aa] + '.fits')
        
        utc = hdr['UTC'].split(':')
        utc_hour = int(utc[0])
        utc_min = int(utc[1])
        utc_sec = int(round(float(utc[2])))

        hour[aa] = utc_hour
        minute[aa] = utc_min
        second[aa] = utc_sec

        massIdx = mass.indexTime(hour[aa], minute[aa], second[aa])
        dimmIdx = dimm.indexTime(hour[aa], minute[aa], second[aa])

        mass_seeing[aa] = mass.free_seeing[massIdx]
        dimm_seeing[aa] = dimm.seeing[dimmIdx]
        iso_angle[aa] = mass.isoplanatic_angle[massIdx]
        tau0[aa] = mass.tau0[massIdx]
        airmass[aa] = dimm.airmass[dimmIdx]
        r0[aa] = dimm.r0[dimmIdx]
        
        # Also calculate the RMS error in astrometry and photometry
        # for the nearest 3 exposures.
        lo = aa-1+1   # additional 1 from skipping reference epoch
        hi = aa+2+1
        if (lo < 0):
            lo += 1
            hi += 1
        if (hi > imgCount):
            lo -= 1
            hi -= 1

        xtmp = np.zeros((3, starCount), dtype=float)
        ytmp = np.zeros((3, starCount), dtype=float)
        mtmp = np.zeros((3, starCount), dtype=float)

        xtmp[0] = s.getArrayFromEpoch(lo, 'x')
        xtmp[1] = s.getArrayFromEpoch(lo+1, 'x')
        xtmp[2] = s.getArrayFromEpoch(lo+2, 'x')

        ytmp[0] = s.getArrayFromEpoch(lo, 'y')
        ytmp[1] = s.getArrayFromEpoch(lo+1, 'y')
        ytmp[2] = s.getArrayFromEpoch(lo+2, 'y')

        mtmp[0] = s.getArrayFromEpoch(lo, 'mag')
        mtmp[1] = s.getArrayFromEpoch(lo+1, 'mag')
        mtmp[2] = s.getArrayFromEpoch(lo+2, 'mag')

        xrms[aa] = xtmp.std(axis=0)
        yrms[aa] = ytmp.std(axis=0)
        mrms[aa] = mtmp.std(axis=0)
    
    print 'Saving everything to a pickle file.'
    _out = open(pickleFile, 'w')

    pickle.dump(imgCount, _out)
    pickle.dump(starCount, _out)

    pickle.dump(x, _out)
    pickle.dump(y, _out)
    pickle.dump(m, _out)
    pickle.dump(r, _out)

    pickle.dump(sfile, _out)
    pickle.dump(strehl, _out)
    pickle.dump(fwhm, _out)
    pickle.dump(mjd, _out)

    pickle.dump(hour, _out)
    pickle.dump(minute, _out)
    pickle.dump(second, _out)

    pickle.dump(mass_seeing, _out)
    pickle.dump(dimm_seeing, _out)
    pickle.dump(iso_angle, _out)
    pickle.dump(tau0, _out)
    pickle.dump(airmass, _out)
    pickle.dump(r0, _out)

    pickle.dump(xrms, _out)
    pickle.dump(yrms, _out)
    pickle.dump(mrms, _out)
    
    _out.close()

    return pickleFile

def load_ao_info(pickleFile):
    # Hang everything off this object.
    o = objects.DataHolder()

    _out = open(pickleFile, 'r')

    o.imgCount = pickle.load(_out)
    o.starCount = pickle.load(_out)

    o.x = pickle.load(_out)
    o.y = pickle.load(_out)
    o.m = pickle.load(_out)
    o.r = pickle.load(_out)

    o.sfile = pickle.load(_out)
    o.strehl = pickle.load(_out)
    o.fwhm = pickle.load(_out)
    o.mjd = pickle.load(_out)

    o.hour = pickle.load(_out)
    o.minute = pickle.load(_out)
    o.second = pickle.load(_out)

    o.mass_seeing = pickle.load(_out)
    o.dimm_seeing = pickle.load(_out)
    o.iso_angle = pickle.load(_out)
    o.tau0 = pickle.load(_out)
    o.airmass = pickle.load(_out)
    o.r0 = pickle.load(_out)

    o.xrms = pickle.load(_out)
    o.yrms = pickle.load(_out)
    o.mrms = pickle.load(_out)
    
    _out.close()

    return o


def plot_ao_info(pickleFile, suffix=None):
    parts = pickleFile.split('/')

    if suffix == None:
        suffix = '_' + parts[-1].replace('_ao_info.dat', '')

    o = load_ao_info(pickleFile)

    # Convert xrms and yrms to mas
    o.xrms *= 10**3
    o.yrms *= 10**3

    # Figure our which stars we will use to average together
    # RMS errors in X, Y, and MAG.
    xrms_by_star = o.xrms.mean(axis=0)
    yrms_by_star = o.yrms.mean(axis=0)
    mrms_by_star = o.mrms.mean(axis=0)

#     print 'Astrometric and Photometric RMS errors for each star:'
#     for ss in range(o.starCount):
#         print 'ss = %2d  r = %.3f  m = %5.2f  xrms = %.2f  yrms = %.2f  mrms = %.2f' % \
#             (ss, o.r[ss], o.m[ss], 
#              xrms_by_star[ss], yrms_by_star[ss], mrms_by_star[ss])

    # Now average over the stars with xrms and yrms < 0.4 mas
    idx = np.where((xrms_by_star < 0.4) & (yrms_by_star < 0.4))[0]
    xrms_stars = o.xrms[:,idx].mean(axis=1)
    yrms_stars = o.yrms[:,idx].mean(axis=1)
    mrms_stars = o.mrms[:,idx].mean(axis=1)

    # ----------
    # Plotting
    # ----------

    # Some ranges for the various variables
    fwhm_range = [50, 120]
    mrms_range = [0.0, 0.05]
    seeing_range = [0.0, 1.2]
    xyrms_range = [0.0, 1.8]
    strehl_range = [0.0, 0.5]
    

    # Plot RMS error in X/Y positions vs. Strehl
    py.clf()
    py.plot(o.strehl, xrms_stars, 'r.', label='X')
    py.plot(o.strehl, yrms_stars, 'b.', label='Y')
    py.xlabel('Strehl')
    py.ylabel('RMS Error in Position (mas)')
    py.legend(numpoints=1)
    py.xlim(strehl_range[0], strehl_range[1])
    py.ylim(xyrms_range[0], xyrms_range[1])
    py.title(suffix[1:])
    py.savefig('plots/strehl_xyrms' + suffix + '.png')

    # Plot RMS error in X/Y positions vs. FWHM
    py.clf()
    py.plot(o.fwhm, xrms_stars, 'r.', label='X')
    py.plot(o.fwhm, yrms_stars, 'b.', label='Y')
    py.xlabel('FWHM (mas)')
    py.ylabel('RMS Error in Position (mas)')
    py.legend(numpoints=1)
    py.xlim(fwhm_range[0], fwhm_range[1])
    py.ylim(xyrms_range[0], xyrms_range[1])
    py.title(suffix[1:])
    py.savefig('plots/fwhm_xyrms' + suffix + '.png')
    

    # Plot RMS error in MAG positions vs. Strehl
    py.clf()
    py.plot(o.strehl, mrms_stars, 'r.')
    py.xlabel('Strehl')
    py.ylabel('RMS Error in Brightness (mag)')
    py.xlim(strehl_range[0], strehl_range[1])
    py.ylim(mrms_range[0], mrms_range[1])
    py.title(suffix[1:])
    py.savefig('plots/strehl_mrms' + suffix + '.png')

    # Plot RMS error in MAG positions vs. FWHM
    py.clf()
    py.plot(o.fwhm, mrms_stars, 'r.')
    py.xlabel('FWHM (mas)')
    py.ylabel('RMS Error in Brightness (mag)')
    py.xlim(fwhm_range[0], fwhm_range[1])
    py.ylim(mrms_range[0], mrms_range[1])
    py.title(suffix[1:])
    py.savefig('plots/fwhm_mrms' + suffix + '.png')


    # Plot Strehl vs. MASS/DIMM seeing
    py.clf()
    py.plot(o.strehl, o.mass_seeing, 'r.', label='High Seeing (MASS)')
    py.plot(o.strehl, o.dimm_seeing, 'b.', label='Total Seeing (DIMM)')
    py.xlabel('Strehl')
    py.ylabel('Seeing at 500 nm (arcsec)')
    py.legend(numpoints=1)
    py.xlim(strehl_range[0], strehl_range[1])
    py.ylim(seeing_range[0], seeing_range[1])
    py.title(suffix[1:])
    py.savefig('plots/strehl_seeing' + suffix + '.png')

    # Plot FWHM vs. MASS/DIMM seeing
    py.clf()
    py.plot(o.fwhm, o.mass_seeing, 'r.', label='High Seeing (MASS)')
    py.plot(o.fwhm, o.dimm_seeing, 'b.', label='Total Seeing (DIMM)')
    py.xlabel('FWHM (mas)')
    py.ylabel('Seeing at 500 nm (arcsec)')
    py.legend(numpoints=1)
    py.xlim(fwhm_range[0], fwhm_range[1])
    py.ylim(seeing_range[0], seeing_range[1])
    py.title(suffix[1:])
    py.savefig('plots/fwhm_seeing' + suffix + '.png')


    # Plot MASS/DIMM seeing vs. RMS error in XY positions
    prms_stars = (xrms_stars + yrms_stars)/2.0
    py.clf()
    py.plot(prms_stars, o.mass_seeing, 'r.', label='High Seeing (MASS)')
    py.plot(prms_stars, o.dimm_seeing, 'b.', label='Total Seeing (DIMM)')
    py.xlabel('RMS Error in Position (mas)')
    py.ylabel('Seeing at 500 nm (arcsec)')
    py.legend(numpoints=1)
    py.xlim(xyrms_range[0], xyrms_range[1])
    py.ylim(seeing_range[0], seeing_range[1])
    py.title(suffix[1:])
    py.savefig('plots/xyrms_seeing' + suffix + '.png')

    # Plot MASS/DIMM seeing vs. RMS error in MAG
    py.clf()
    py.plot(mrms_stars, o.mass_seeing, 'r.', label='High Seeing (MASS)')
    py.plot(mrms_stars, o.dimm_seeing, 'b.', label='Total Seeing (DIMM)')
    py.xlabel('RMS Error in Brightness (mag)')
    py.ylabel('Seeing at 500 nm (arcsec)')
    py.legend(numpoints=1)
    py.xlim(mrms_range[0], mrms_range[1])
    py.ylim(seeing_range[0], seeing_range[1])
    py.title(suffix[1:])
    py.savefig('plots/mrms_seeing' + suffix + '.png')
