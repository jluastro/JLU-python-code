import asciidata
import numpy as np

def calib_file_from_stolte_old():
    """
    *** THIS FUNCTION IS SUPERSEDED NOW***

    Make a photometric calibration file that is readable
    by calibrate.py based on the photometry published in Stolte et al. 2010.
    """
    workDir = '/u/jlu/work/arches/photo_calib/from_stolte/'

    info = []
    info.append({'name':'f1', 'star':'f1_psf0', 'xpix':435.53, 'ypix':722.87})
    info.append({'name':'f2', 'star':'f2_psf1', 'xpix':1138.20, 'ypix':1081.66})
    info.append({'name':'f3', 'star':'f3_psf0', 'xpix':944.65, 'ypix':423.45})
    info.append({'name':'f4', 'star':'f4_psf0', 'xpix':830.02, 'ypix':301.25})
#     info.append({'name':'f5', 'star':'f5_psf0', 'xpix':378.33, 'ypix':957.13})
    info.append({'name':'f6', 'star':'f6_psf0', 'xpix':261.22, 'ypix':723.89})

    # Read in the existing label.dat file which is already on a common
    # coordinate system.
    labels = asciidata.open('/u/ghezgroup/data/gc/source_list/label_arch.dat')
    names = labels[0].tonumpy()
    old_kp = labels[1].tonumpy()
    old_x = labels[2].tonumpy()
    old_y = labels[3].tonumpy()

    scale = 0.00995

    _out = open(workDir + 'arch_photo_calib_new.dat', 'w')
    _out.write('## Columns: Format of this header is hardcoded for read in by calibrate.py\n')
    _out.write('## Field separator in each column is "--". Default calibrators are listed after\n')
    _out.write('## each magnitude column header entry.\n')
    _out.write('# 1 -- Star Name\n')
    _out.write('# 2 -- X Position (arcsec, increasing east)\n')
    _out.write('# 3 -- Y Position (arcsec)\n')
    _out.write('# 4 -- isVariable? flag\n')
    _out.write('# 5 -- H band from Stolte et al. 2010 -- f1_psf0,f1_psf1,f1_psf2,f1_psf3\n')
    _out.write('# 6 -- Kp band from Stolte et al. 2010 -- f1_psf0,f1_psf1,f1_psf2,f1_psf3\n')
    _out.write('# 7 -- Lp band from Stolte et al. 2010 -- f1_psf0,f1_psf1,f1_psf2,f1_psf3\n')

    for ff in range(len(info)):
        # Read in the calibrated matched star list for this field
        stars = asciidata.open(workDir + info[ff]['name'] + '_hkplp.cal')
        
        xpix = stars[7].tonumpy()
        ypix = stars[8].tonumpy()
        m_h = stars[4].tonumpy()
        me_h = stars[5].tonumpy()
        m_kp = stars[11].tonumpy()
        me_kp = stars[12].tonumpy()
        m_lp = stars[18].tonumpy()
        me_lp = stars[19].tonumpy()
        
        # Find the reference star in the label.dat file
        idx = np.where(names == info[ff]['star'])
        
        dr = np.hypot(xpix - info[ff]['xpix'], ypix - info[ff]['ypix'])
        rdx = dr.argmin()

        # Sanity check on the magnitude difference.
        dm = m_kp[rdx] - old_kp[idx]
        if abs(dm) > 1.0:
            'Error: found wrong source in %s: ' % (info[ff]['name'])
            '   dr = %.3f  dm = %.2f'  % (dr[rdx], dm)

        xpixRef = xpix[rdx]
        ypixRef = ypix[rdx]
        xarcRef = old_x[idx]
        yarcRef = old_y[idx]

        xarc = ((xpix - xpixRef) * scale * -1.0) + xarcRef
        yarc = ((ypix - ypixRef) * scale) + yarcRef

        # Match Named Sources and get out their magnitudes for
        # our new arch_photo_calib.dat file. Lets also check
        # uncertainties as well just to make sure these are good
        # calibrators.
        for nn in range(len(names)):
            diff = np.hypot(xarc - old_x[nn], yarc - old_y[nn])
            rdx = diff.argmin()

            dr = diff[rdx]
            dm = m_kp[rdx] - old_kp[nn]

            # Call this a match
            if (dr < 0.2) and (dm < 1.0):
                _out.write('%-10s  ' % (names[nn]))
                _out.write('%8.3f  %8.3f  0    ' % (xarc[rdx], yarc[rdx]))
                _out.write('%6.3f  ' % (m_h[rdx]))
                _out.write('%6.3f  ' % (m_kp[rdx]))
                _out.write('%6.3f\n' % (m_lp[rdx]))
        

    _out.close()
