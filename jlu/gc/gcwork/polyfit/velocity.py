from gcwork import objects
from gcwork import starset
from scipy import stats
import asciidata
import pylab as py
import numpy as np
import histNofill
import math

def compareAlignVels(alignRoot, polyRoot):
    """
    Compare the velocities that are derived in align and in polyfit.
    There must be the same number of stars (and in the same order)
    in the align and polyfit output.

    Input Parameters:
    -- the root name of the align output (e.g. 'align/align_d_rms_t'
    -- the root name of the polyfit output( e.g. 'polyfit_d/fit')
    """
    cc = objects.Constants()
    s = starset.StarSet(alignRoot, absolute=0)
    s.loadPolyfit(polyRoot, arcsec=0)

    pixyr2kms = 0.00995 * cc.asy_to_kms
    #pixyr2kms = cc.asy_to_kms

    # Need to get rid of the central arcsecond sources
    r = s.getArray('r2d')
    idx = (py.where(r > 0.5))[0]
    s.stars = [s.stars[ii] for ii in idx]
    r = s.getArray('r2d')

    # Align Fit in Pixels
    vxa = s.getArray('fitpXalign.v') * pixyr2kms
    vya = s.getArray('fitpYalign.v') * pixyr2kms
    vxa_err = s.getArray('fitpXalign.verr') * pixyr2kms
    vya_err = s.getArray('fitpYalign.verr') * pixyr2kms
    vxp = s.getArray('fitpXv.v') * pixyr2kms
    vyp = s.getArray('fitpYv.v') * pixyr2kms
    vxp_err = s.getArray('fitpXv.verr') * pixyr2kms
    vyp_err = s.getArray('fitpYv.verr') * pixyr2kms

    # Plot differences in Sigmas but can't combine (not independent)
    vxdiff = vxa - vxp
    vydiff = vya - vyp

    vxsig1 = vxdiff / vxa_err
    vysig1 = vydiff / vya_err
    vxsig2 = vxdiff / vxp_err
    vysig2 = vydiff / vyp_err

    #####
    # Crude velocity plot (absolute velocity differences)
    #####
    py.clf()
    py.subplot(2, 1, 1)
    py.plot(vxa, vxp, 'k.')
    py.xlabel('Align vx (km/s)')
    py.ylabel('Polyfit vx (km/s)')
    py.title('Align vs. Polyfit Vel')

    py.subplot(2, 1, 2)
    py.plot(vya, vyp, 'k.')
    py.xlabel('Align vy (km/s)')
    py.ylabel('Polyfit vy (km/s)')
    py.savefig('plots/align_vs_poly_v.png')


    #####
    # Compare velocity errors
    #####
    py.clf()
    py.plot(vxa_err, vxp_err, 'r.')
    py.plot(vya_err, vyp_err, 'b.')
    py.plot([0, 30], [0, 30], 'k--')
    py.axis('equal')
    py.axis([0, 30, 0, 30])
    py.xlabel('Align Vel Error (km/s)')
    py.ylabel('Poly Vel Error (km/s)')
    py.legend(('X', 'Y'))
    py.title('Align vs. Polyfit Vel. Errors')
    py.savefig('plots/align_vs_poly_verr.png')

    #####
    # Absolute velocity differences
    #####
    py.clf()
    py.plot(vxa, vxdiff, 'r.')
    py.plot(vya, vydiff, 'b.')
    py.legend(('X', 'Y'))
    py.xlabel('Align v (km/s)')
    py.ylabel('Align - Poly (km/s)')
    py.title('Align - Polyfit Vel.')
    py.savefig('plots/align_vs_poly_vdiff.png')

    #####
    # Velocity difference in sigmas
    #####
    py.clf()
    py.plot(vxa, vxsig1, 'r.')
    py.plot(vya, vysig1, 'b.')
    py.legend(('X', 'Y'))
    py.title('Diff over align error')
    py.xlabel('Align v (km/s)')
    py.ylabel('Align - Poly (sigma)')
    py.title('(Align - Polyfit) / Align Err')
    py.savefig('plots/align_vs_poly_vsig_alignerr.png')

    py.clf()
    py.plot(vxa, vxsig2, 'r.')
    py.plot(vya, vysig2, 'b.')
    py.legend(('X', 'Y'))
    py.title('Diff over poly error')
    py.xlabel('Align v (km/s)')
    py.ylabel('Align - Poly (sigma)')
    py.title('(Align - Polyfit) / Polyfit Err')
    py.savefig('plots/align_vs_poly_vsig_polyerr.png')

    #####
    # Histogram of Sigmas
    #####
    binsIn = py.arange(-6, 6, 0.5)
    (bins, vxhist1) = histNofill.hist(binsIn, vxsig1, normed=True)
    (bins, vxhist2) = histNofill.hist(binsIn, vxsig2, normed=True)
    (bins, vyhist1) = histNofill.hist(binsIn, vysig1, normed=True)
    (bins, vyhist2) = histNofill.hist(binsIn, vysig2, normed=True)

    # Make a gaussian for what is expected
    gg = stats.distributions.norm()
    gaussian = gg.pdf(bins)

    py.clf()
    py.plot(bins, vxhist1, 'r-')
    py.plot(bins, vyhist1, 'b-')
    py.plot(bins, gaussian, 'k--')
    py.legend(('X', 'Y'))
    py.xlabel('Align - Poly (sigma)')
    py.title('Vel Diff Significance (Align Err)')
    py.savefig('plots/align_vs_poly_hist_alignerr.png')

    py.clf()
    py.plot(bins, vxhist2, 'r-')
    py.plot(bins, vyhist2, 'b-')
    py.plot(bins, gaussian, 'k--')
    py.legend(('X', 'Y'))
    py.xlabel('Align - Poly (sigma)')
    py.title('Vel Diff Significance (Poly Err)')
    py.savefig('plots/align_vs_poly_hist_polyerr.png')


def plotVelocityMap(root='./', align='align/align_d_rms_1000_abs_t',
                    poly='polyfit_d/fit'):

    s = starset.StarSet(root + align)
    s.loadPolyfit(root + poly)

    x = s.getArray('fitXv.p')
    y = s.getArray('fitYv.p')
    vx = s.getArray('fitXv.v')
    vy = s.getArray('fitYv.v')

    py.clf()
    arrScale = 1.0
    py.quiver([x], [y], [vx], [vy], scale=0.35, headwidth=3, color='black')

        
def plotStar(star, root='./', align = 'align/align_d_rms_1000_abs_t',
             poly='polyfit_d/fit', points='points_d/',LGSonly=False):

    pointsFile = root + points + star + '.points'
    fitFile = root + poly + '.' + star + '.lfit'

    _tabPoints = asciidata.open(pointsFile)
    date = _tabPoints[0].tonumpy()
    x = _tabPoints[1].tonumpy() * -1.0
    y = _tabPoints[2].tonumpy()
    xerr = _tabPoints[3].tonumpy()
    yerr = _tabPoints[4].tonumpy()

    _fitPoints = asciidata.open(fitFile)
    date_f = _fitPoints[0].tonumpy()
    x_f = _fitPoints[1].tonumpy() * -1.0
    y_f = _fitPoints[2].tonumpy()
    xerr_f = _fitPoints[3].tonumpy()
    yerr_f = _fitPoints[4].tonumpy()

    # Find range of plot
    halfRange = max([ abs(x.max() - x.min()), abs(y.max() - y.min()) ]) / 2.0

    padd = 0.0005
    xmax = x.min() + ((x.max() - x.min())/2.0) + halfRange + padd
    ymax = y.min() + ((y.max() - y.min())/2.0) + halfRange + padd
    xmin = x.min() + ((x.max() - x.min())/2.0) - halfRange - 2*padd
    ymin = y.min() + ((y.max() - y.min())/2.0) - halfRange - padd

    if LGSonly:
        legend_items = ['2006', '2007', '2008']
        legend_colors = ['darkorange', 'orangered', 'red']
    else:
        legend_items = ['1995', '1996', '1997', '1998', '1999',
                        '2000', '2001', '2002', '2003', '2004',
                        '2005', '2006', '2007', '2008', '2009',
                        '2010']
        legend_colors = ['olive', 'brown', 'purple', 'darkslateblue',
                         'mediumblue', 'steelblue', 'teal', 'seagreen',
                         'green', 'lawngreen', 'goldenrod', 'greenyellow',
                         'gold', 'darkorange', 'orangered', 'red']

    # Assign color to each epoch
    year = [str(int(np.floor( d ))) for d in date]
    color_arr = []


    py.close(2)
    py.figure(2, figsize=(6.2,5.4))
    py.clf()
    py.plot(x_f, y_f, 'k-')
    py.plot(x_f + xerr_f, y_f + yerr_f, 'k--')
    py.plot(x_f - xerr_f, y_f - yerr_f, 'k--')
    
    for i in range(len(year)):
        # find out which year
        try:
            idx = legend_items.index( year[i] )
            color_arr.append( legend_colors[idx] )
        except ValueError:
            color_arr.append('black')

        py.errorbar([x[i]], [y[i]], fmt='ko', xerr=xerr[i], yerr=yerr[i],
                    ms=5,
                    color=color_arr[i], mec=color_arr[i], mfc=color_arr[i])

    # Set axis ranges
    py.title(star)
    ax = py.axis([xmax, xmin, ymin, ymax])
    py.gca().set_aspect('equal')
    py.xlabel('R.A. Offset from Sgr A* (arcsec)')
    py.ylabel('Dec. Offset from Sgr A* (arcsec)')

    # Draw legend
    py.legend(legend_items, numpoints=1, loc=2)
    ltext = py.gca().get_legend().get_texts()
    lgdLines = py.gca().get_legend().get_lines()
    for j in range(len(ltext)):
        py.setp(ltext[j], color=legend_colors[j])
        lgdLines[j].set_marker('o')
        lgdLines[j].set_mec(legend_colors[j])
        lgdLines[j].set_mfc(legend_colors[j])
        lgdLines[j].set_ms(5)


    # Retrieve information on the velocity fit from the .linearFormal file
    fitFile = root + poly + '.linearFormal'
    _fit = open(fitFile, 'r')

    for line in _fit:
        entries = line.split()

        if (entries[0] == star):
            chiSqX = float( entries[5] )
            Qx = float( entries[6] )
            chiSqY = float( entries[11] )
            Qy = float( entries[12] )
            break


    _fit.close()

    py.savefig(root+'plots/plotStar_'+star+'_orbit.eps')    
    py.savefig(root+'plots/plotStar_'+star+'_orbit.png')    


def velocityAverage(alignRoot, polyRoot, magCut=15):
    """
    Calculate the mean (and error on mean) of the velocities from
    a align/polyfit.

    Input Parameters:
    -- the root name of the align output (e.g. 'align/align_d_rms_t'
    -- the root name of the polyfit output( e.g. 'polyfit_d/fit')
    """
    # This should be an absolute aligned data set. 
    cc = objects.Constants()
    s = starset.StarSet(alignRoot)
    s.loadPolyfit(polyRoot, accel=0)

    vx = s.getArray('fitXv.v') * 10**3
    vy = s.getArray('fitYv.v') * 10**3
    vxerr = s.getArray('fitXv.verr') * 10**3
    vyerr = s.getArray('fitYv.verr') * 10**3

    mag = s.getArray('mag')
    idx = np.where(mag <= magCut)[0]

    py.clf()
    py.hist(vx[idx])
    py.hist(vy[idx])
    
    print 'Number of Stars: %d' % len(idx)
    print 'X Mean Velocity: %5.2f' % (vx[idx].mean())
    print 'X Error on Mean: %5.2f' % (vx[idx].std() / math.sqrt(len(vx)))
    print 'Y Mean Velocity: %5.2f' % (vy[idx].mean())
    print 'Y Error on Mean: %5.2f' % (vy[idx].std() / math.sqrt(len(vx)))


    # Plot distribution of velocity errors
    py.clf()
    binsIn = np.arange(0, max([max(vxerr), max(vyerr)]), 0.1)
    (bins,data)=histNofill.hist(binsIn,vxerr)
    py.plot(bins,data,'r',linewidth=2)
    (bins,data)=histNofill.hist(binsIn,vyerr)
    py.plot(bins,data,'b',linewidth=2)
    py.axis([0,10,0,600])
    py.xlabel('Velocity Errors (mas/yr)')
    py.ylabel('N')
    py.savefig('plots/histVelErr.png')

