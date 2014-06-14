import asciidata, math
import pylab as p
import numarray as na
from gcwork import starset

def plotTrans(root):
    """
    Plot the fractional change in plate scale and PA over many different
    starlists that  have been aligned. You can either give align results
    for many different epochs or align results for many different cleaned
    frames in a single epoch.

    root - align output
    """
    tab = asciidata.open(root + '.trans')

    a0 =  tab[3].tonumarray()
    a0e = tab[4].tonumarray()
    a1 =  tab[5].tonumarray()
    a1e = tab[6].tonumarray()
    a2 =  tab[7].tonumarray()
    a2e = tab[8].tonumarray()
    b0 =  tab[9].tonumarray()
    b0e = tab[10].tonumarray()
    b1 =  tab[11].tonumarray()
    b1e = tab[12].tonumarray()
    b2 =  tab[13].tonumarray()
    b2e = tab[14].tonumarray()

    trans = []
    for ff in range(len(a0)):
        tt = objects.Transform()
        tt.a = [a0[ff], a1[ff], a2[ff]]
        tt.b = [b0[ff], b1[ff], b2[ff]]
        tt.aerr = [a0e[ff], a1e[ff], a2e[ff]]
        tt.berr = [b0e[ff], b1e[ff], b2e[ff]]
        tt.linearToSpherical(override=False)

        trans.append(tt)

    # Read epochs
    dateTab = asciidata.open(root + '.date')
    numEpochs = dateTab.ncols
    years = [dateTab[i][0] for i in range(numEpochs)]

    p.clf()
    p.subplot(211)
    p.plot(scale - 1.0, 'ko')
    p.ylabel('Fract. Plate Scale Difference')
    if (years[0] != years[1]):
        thePlot = p.gca()
        thePlot.get_xaxis().set_major_locator(p.MultipleLocator(0.1))
        thePlot.get_xaxis().set_major_formatter(p.FormatStrFormatter('%8.3f'))
    
    p.subplot(212)
    p.plot(angle, 'ko')
    p.ylabel('Position Angle')
    if (years[0] != years[1]):
        thePlot = p.gca()
        thePlot.get_xaxis().set_major_locator(p.MultipleLocator(0.1))
        thePlot.get_xaxis().set_major_formatter(p.FormatStrFormatter('%8.3f'))


def plotFitRms(root, polyroot, gcfitdir):
    s = starset.StarSet(root)
    s.loadPolyfit(polyroot, arcsec=1, accel=0)

    years = s.stars[0].years
    fitPx = s.getArray('fitXv.p')
    fitVx = s.getArray('fitXv.v')
    fitPy = s.getArray('fitYv.p')
    fitVy = s.getArray('fitYv.v')
    t0x = s.getArray('fitXv.t0')
    t0y = s.getArray('fitYv.t0')

    rmsX = na.zeros(len(s.stars), type=na.Float)
    rmsY = na.zeros(len(s.stars), type=na.Float)
    rms = na.zeros(len(s.stars), type=na.Float)
    cnt = na.zeros(len(s.stars), type=na.Int)

    for ee in range(len(years)):
        dtX = years[ee] - t0x
        dtY = years[ee] - t0y

        xfit = fitPx + (dtX * fitVx)
        yfit = fitPy + (dtY * fitVy)

        x = s.getArrayFromEpoch(ee, 'x')
        y = s.getArrayFromEpoch(ee, 'y')
        xpix = s.getArrayFromEpoch(ee, 'xpix')
        ypix = s.getArrayFromEpoch(ee, 'ypix')

        diffx = xfit - x
        diffy = yfit - y
        diff = na.sqrt(diffx**2 + diffy**2)

        idx = (na.where(xpix > -999))[0]

        rmsX[idx] += diffx**2
        rmsY[idx] += diffy**2
        rms[idx] += diff**2
        cnt[idx] += 1

    rmsX = na.sqrt(rmsX / cnt) * 1000.0
    rmsY = na.sqrt(rmsY / cnt) * 1000.0
    rms = na.sqrt(rms / cnt) * 1000.0

    mag = s.getArray('mag')
    x = s.getArray('x')
    y = s.getArray('y')
    r = na.sqrt(x**2 + y**2)

    idx = (na.where(mag < 15))[0]

    p.clf()
    p.semilogy(r[idx], rms[idx], 'k.')


def plotPosError(starlist, outsuffix, imgtype='png'):
    """
    Load up a starlist output by align_rms.
    """

    tab = asciidata.open(starlist)

    mag = tab[1].tonumarray()
    x = tab[3].tonumarray()
    y = tab[4].tonumarray()
    xerr = tab[5].tonumarray()
    yerr = tab[6].tonumarray()

    xdiff = x - 512.0
    ydiff = y - 512.0
    r = na.sqrt(xdiff**2 + ydiff**2)

    err = (xerr + yerr) / 2.0

    # Plot magnitude dependance
    p.clf()
    p.semilogy(mag, err*9.94, 'k.')

    idx = (na.where(mag < 14))[0]

    medianPix = na.linear_algebra.mlab.median(err[idx])
    medianArc = medianPix * 9.94
    p.text(9, 60, 'For (8 < K < 14)')
    p.text(9, 40, 'Median Error = %5.3f pix (%5.3f mas)' %
           (medianPix, medianArc))
    p.xlabel('Magnitude')
    p.ylabel('Positional Uncertainty (mas)''Positional Uncertainty (mas)')

    p.savefig('pos_error_mag_%s.%s' % (outsuffix, imgtype))


    # Plot radial dependance
    p.clf()
    p.semilogy(r[idx] * 0.00994, err[idx] * 9.94, 'k.')
    p.xlabel('Radius from Field Center (arcsec)')
    p.ylabel('Positional Uncertainty (mas)')
    p.savefig('pos_error_r_%s.%s' % (outsuffix, imgtype))
