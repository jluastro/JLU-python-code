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

def guide_star_info():
    """
    Get the guide star information out of the image headers.
    """
    imgDir = ngc1851_data + 'starfinder/'

    imgs = glob.glob(imgDir + '*_G1_1_cr.fits')

    # Keep results in an output file
    f_out = open(ngc1851_data + 'guide_star_info.txt', 'w')
    f_fmt = '{0:8s} {1:15s} '

    for img in imgs:
        hdr = pyfits.getheader(img)

        f_out.write('{0:20s}  '.format(os.path.basename(img)))

        # There are 4 possible guide stars. Try to print them all out
        for ii in range(4):
            try:
                cfg = 'GWFS{0}CFG'.format(ii+1)
                obj = 'GWFS{0}OBJ'.format(ii+1)
                f_out.write(f_fmt.format(hdr[cfg], hdr[obj]))
            except KeyError:
                f_out.write(f_fmt.format('-', '-'))
                
        f_out.write('\n')

    f_out.close()

def image_info():
    """
    Get the exposure time and filter info for each image. Store it in
    an output file called image_info.txt.
    """
    imgDir = ngc1851_data + 'starfinder/'

    imgs = glob.glob(imgDir + '*_G1_1_cr.fits')

    # Keep results in an output file
    f_out = open(ngc1851_data + 'image_info.txt', 'w')
    f_fmt = '{img:15s}  {xoff:7.2f} {yoff:7.2f} {pa:5.1f}  {filt:12s}  '
    f_fmt += '{exp:6.2f} {coad:2d} {tot:7.2f}\n'

    for img in imgs:
        hdr = pyfits.getheader(img)

        xoff = hdr['XOFFSET']
        yoff = hdr['YOFFSET']
        pa = hdr['PA']
        filt = hdr['FILTER1']
        exp = hdr['EXPTIME']
        coadd = hdr['COADDS']

        f_out.write(f_fmt.format(img=os.path.basename(img), xoff=xoff, yoff=yoff,
                                 pa=pa, filt=filt, exp=exp, coad=coadd, tot=exp*coadd))

    f_out.close()

    
def ngc1851_image():
    """
    Plot an image of NGC 1851 for the paper.
    """
    image_root = '/u/jlu/data/gsaoi/commission/reduce/ngc1851/combo/ngc1851'

    # Load up the PSF stars file to get the coordinates.
    stars = atpy.Table(image_root + '_psf_stars_pixel.txt', type='ascii')

    stars.Xarc = stars.X * scale
    stars.Yarc = stars.Y * scale

    scale = 0.00995

    # gc = aplpy.FITSFigure(image_file)
    # gc.show_grayscale()

    img = pyfits.getdata(image_root + '.fits')
    img = img_scale.log(img, scale_min=0, scale_max=1e4)
    #img = img_scale.sqrt(img, scale_min=500, scale_max=5e4)
    # img = img_scale.linear(img, scale_min=500, scale_max=4e4)

    xmin = ((0 - cooPix[0]) * scale * -1.0) + cooAsec[0]
    xmax = ((img.shape[1] - cooPix[0]) * scale * -1.0) + cooAsec[0]
    ymin = ((0 - cooPix[1]) * scale) + cooAsec[1]
    ymax = ((img.shape[0] - cooPix[1]) * scale) + cooAsec[1]
    extent = [xmin, xmax, ymin, ymax]

    py.clf()
    py.imshow(img, extent=extent, cmap=py.cm.gray_r)
    
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

        py.subplot(2, 1, ii+1)
        py.clf()
        py.errorbar(dx, dy, xerr=xerr_n1, yerr=yerr_n1, fmt='k.')
        py.plot([0], [0], 'rs', ms=8)
        py.xlabel('X Difference (mas)')
        py.ylabel('Y Difference (mas)')
        py.title('GSAOI Night 2 - Night 1')
        lim = 6
        py.axis([-lim, lim, -lim, lim])

    outFile = '/u/jlu/doc/papers/proceed_2014_spie/gsaoi_compare_nights.png'
    py.savefig(outFile)



    
                    
    
