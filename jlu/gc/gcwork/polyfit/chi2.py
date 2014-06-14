import asciidata, pyfits
import os, sys, math
import pylab as py
import numpy as np
from scipy import optimize
from gcwork import starset
from gcwork import objects
from gcwork import util
from gcwork import young
from gcwork import orbits
import scipy.stats

def histogram(root='./', align='align/align_d_rms_t',
              poly='polyfit_d_points/fit', accel=False, youngOnly=False,
              trimMaxEpochs=True):

    s = starset.StarSet(root + align)
    s.loadPolyfit(root + poly, arcsec=1, accel=accel)

    if (youngOnly):
        s.onlyYoungDisk()

    if (accel):
        chi2Xred = s.getArray('fitXa.chi2red')
        chi2Yred = s.getArray('fitYa.chi2red')
        chi2X = s.getArray('fitXa.chi2')
        chi2Y = s.getArray('fitYa.chi2')
        dof = s.getArray('fitXa.dof')
    else:
        chi2Xred = s.getArray('fitXv.chi2red')
        chi2Yred = s.getArray('fitYv.chi2red')
        chi2X = s.getArray('fitXv.chi2')
        chi2Y = s.getArray('fitYv.chi2')
        dof = s.getArray('fitXv.dof')

    cnt = s.getArray('velCnt')

    if trimMaxEpochs:
        idx = np.where(cnt == cnt.max())[0]
        chi2Xred = chi2Xred[idx]
        chi2Yred = chi2Yred[idx]
        chi2X = chi2X[idx]
        chi2Y = chi2Y[idx]
        dof = dof[idx]
        cnt = cnt[idx]

    # ==========
    # First histograms on the chi2 (not reduced)
    # ==========

    # Plot the unreduced distribution
    binSize = np.median(chi2X) / 10.0
    binsIn = np.arange(0, dof.max()*100, binSize)
    binScale = 100.0
    binsInModel = np.arange(0, dof.max()*100, binSize/binScale)
    chi2model = np.zeros(len(binsInModel), dtype=float)

    # Make a model distribution using the appropriate degrees of 
    # freedom for all the stars.
    maxEpochCnt = cnt.max()
    minEpochCnt = cnt.min()

    for ii in range(minEpochCnt, maxEpochCnt+1):
        idx = np.where(cnt == ii)[0]

        if len(idx) > 0:
            chi2 = scipy.stats.chi2( dof[idx[0]] )

            # Get PDF, and then convert to histogram with the
            # Same prob/bin as our data histogram
            tmp = chi2.pdf(binsInModel) * len(idx) * binSize

            # Add to our model
            chi2model += tmp

                

    # Plot up the results
    py.clf()

    # X
    py.subplot(2, 1, 1)
    (n, bins, p) = py.hist(chi2X, bins=binsIn, alpha=0.6)
    p1 = py.plot(binsInModel, chi2model)
    py.xlim(0, dof.max()*5)
    py.ylabel('Number of Stars')
    py.xlabel('X Velocity Fit Chi^2')
    py.title('%d stars in %d-%d Epochs' % (len(chi2Xred), cnt.min(), cnt.max()))
        
    # Y
    py.subplot(2, 1, 2)
    (n, bins, p) = py.hist(chi2Y, bins=binsIn, alpha=0.6)
    p1 = py.plot(binsInModel, chi2model)
    py.xlim(0, dof.max()*5)
    py.ylabel('Number of Stars')
    py.xlabel('Y Velocity Fit Chi^2')

    if trimMaxEpochs:
        outputFile = root + 'plots/poly_chi2_hist_max_dof.png'
    else:
        outputFile = root + 'plots/poly_chi2_hist_all_dof.png'

    py.savefig(outputFile)


    # ==========
    # Now histograms on the reduced-chi2
    # ==========
    # Plot the reduced distribution
    binSize = np.median(chi2Xred) / 10.0
    binsIn = np.arange(0, 100, binSize)
    binScale = 100.0
    binsInModel = np.arange(0, 100, binSize/binScale)
    chi2model = np.zeros(len(binsInModel), dtype=float)

    # Make a model distribution using the appropriate degrees of 
    # freedom for all the stars.
    maxEpochCnt = cnt.max()
    minEpochCnt = cnt.min()

    for ii in range(minEpochCnt, maxEpochCnt+1):
        idx = np.where(cnt == ii)[0]

        if len(idx) > 0:
            degreesOfFreedom = dof[idx[0]]
            chi2 = scipy.stats.chi2( degreesOfFreedom )

            # Get PDF, and then convert to histogram with the
            # Same prob/bin as our data histogram
            tmp = chi2.pdf(binsInModel*degreesOfFreedom)
            tmp *= len(idx) * binScale / tmp.sum()

            # Add to our model
            chi2model += tmp

    # Plot up the results
    py.clf()

    # X
    py.subplot(2, 1, 1)
    (n, bins, p) = py.hist(chi2Xred, bins=binsIn, alpha=0.6)
    p1 = py.plot(binsInModel, chi2model)
    py.xlim(0, 5)
    py.ylabel('Number of Stars')
    py.xlabel('X Velocity Fit Reduced Chi^2')
    py.title('%d stars in %d-%d Epochs' % (len(chi2Xred), cnt.min(), cnt.max()))
        
    # Y
    py.subplot(2, 1, 2)
    (n, bins, p) = py.hist(chi2Yred, bins=binsIn, alpha=0.6)
    p1 = py.plot(binsInModel, chi2model)
    py.xlim(0, 5)
    py.ylabel('Number of Stars')
    py.xlabel('Y Velocity Fit Reduced Chi^2')

    if trimMaxEpochs:
        outputFile = root + 'plots/poly_chi2red_hist_max_dof.png'
    else:
        outputFile = root + 'plots/poly_chi2red_hist_all_dof.png'

    py.savefig(outputFile)

def cdf(root='./', align='align/align_d_rms_t',
        poly='polyfit_d_points/fit', accel=False, youngOnly=False,
        trimMaxEpochs=True):

    s = starset.StarSet(root + align)
    s.loadPolyfit(root + poly, arcsec=1, accel=accel)

    if (youngOnly):
        s.onlyYoungDisk()

    if (accel):
        chi2Xred = s.getArray('fitXa.chi2red')
        chi2Yred = s.getArray('fitYa.chi2red')
        chi2X = s.getArray('fitXa.chi2')
        chi2Y = s.getArray('fitYa.chi2')
        dof = s.getArray('fitXa.dof')
    else:
        chi2Xred = s.getArray('fitXv.chi2red')
        chi2Yred = s.getArray('fitYv.chi2red')
        chi2X = s.getArray('fitXv.chi2')
        chi2Y = s.getArray('fitYv.chi2')
        dof = s.getArray('fitXv.dof')

    cnt = s.getArray('velCnt')

    if trimMaxEpochs:
        idx = np.where(cnt == cnt.max())[0]
        chi2Xred = chi2Xred[idx]
        chi2Yred = chi2Yred[idx]
        chi2X = chi2X[idx]
        chi2Y = chi2Y[idx]
        dof = dof[idx]
        cnt = cnt[idx]

    # Make a model distribution using the appropriate degrees of 
    # freedom for all the stars.
    maxEpochCnt = cnt.max()
    minEpochCnt = cnt.min()
    

    # ==========
    # First analyze chi^2 (not reduced)
    # ==========
    chi2Xsorted = chi2X.copy()
    chi2Xsorted.sort()

    chi2Ysorted = chi2Y.copy()
    chi2Ysorted.sort()

    cdfObserved = (np.arange(len(chi2X)) + 1.) / len(chi2X)

    def cdfModel(chi2data):
        chi2model = np.zeros(len(chi2data), dtype=float)

        for ii in range(minEpochCnt, maxEpochCnt+1):
            idx = np.where(cnt == ii)[0]
            
            if len(idx) > 0:
                degreesOfFreedom = dof[idx[0]]
                chi2 = scipy.stats.chi2( degreesOfFreedom )
                
                tmp = chi2.cdf(chi2data)
                tmp *= float(len(idx)) / len(chi2data)
                chi2model += tmp

        return chi2model

    chi2modelX = cdfModel(chi2Xsorted)
    chi2modelY = cdfModel(chi2Ysorted)

    foox = scipy.stats.kstest(chi2Xsorted, cdfModel)
    fooy = scipy.stats.kstest(chi2Ysorted, cdfModel)
    print foox, fooy
    print 'KS Test - Probability that observations follow expected distribution:'
    print '  X: D = %.2f   P = %e' % (foox[0], foox[1])
    print '  Y: D = %.2f   P = %e' % (fooy[0], fooy[1])
            
    # Plot up the results
    py.clf()

    # X
    py.subplot(2, 1, 1)
    py.plot(chi2Xsorted, cdfObserved, 'k-')
    py.plot(chi2Xsorted, chi2modelX, 'b-')
    py.xlim(0, dof[0]*5)
    py.ylabel('CDF for X Vel. Fit')
    py.title('%d stars in %d-%d epochs' % (len(chi2Xred), cnt.min(), cnt.max()))
    
    # Y
    py.subplot(2, 1, 2)
    py.plot(chi2Ysorted, cdfObserved, 'k-')
    py.plot(chi2Ysorted, chi2modelY, 'b-')
    py.xlim(0, dof[0]*5)
    py.ylabel('CDF for Y Vel. Fit')
    py.xlabel('Chi^2')
    py.legend(('Observed', 'Model'), loc='lower right')

    if trimMaxEpochs:
        outputFile = root + 'plots/poly_chi2_cdf_max_dof.png'
    else:
        outputFile = root + 'plots/poly_chi2_cdf_all_dof.png'

    py.savefig(outputFile)


    # ==========
    # Now analyze reduced chi^2
    # ==========
    chi2Xsorted = chi2Xred.copy()
    chi2Xsorted.sort()

    chi2Ysorted = chi2Yred.copy()
    chi2Ysorted.sort()

    cdfObserved = (np.arange(len(chi2Xred)) + 1.) / len(chi2Xred)

    def cdfModel(chi2data):
        # Here chi2data is a list of reduced chi2 values.
        chi2model = np.zeros(len(chi2data), dtype=float)

        for ii in range(minEpochCnt, maxEpochCnt+1):
            idx = np.where(cnt == ii)[0]
            
            if len(idx) > 0:
                degreesOfFreedom = dof[idx[0]]
                chi2 = scipy.stats.chi2( degreesOfFreedom )
                
                tmp = chi2.cdf(chi2data * degreesOfFreedom)
                tmp *= float(len(idx)) / len(chi2data)
                chi2model += tmp

        return chi2model

    chi2modelX = cdfModel(chi2Xsorted)
    chi2modelY = cdfModel(chi2Ysorted)

    foox = scipy.stats.kstest(chi2Xsorted, cdfModel)
    fooy = scipy.stats.kstest(chi2Ysorted, cdfModel)
    print 'KS Test - Probability that observations follow expected distribution:'
    print '  X: D = %.2f   P = %e' % (foox[0], foox[1])
    print '  Y: D = %.2f   P = %e' % (fooy[0], fooy[1])
            
    # Plot up the results
    py.clf()

    # X
    py.subplot(2, 1, 1)
    py.plot(chi2Xsorted, cdfObserved, 'k-')
    py.plot(chi2Xsorted, chi2modelX, 'b-')
    py.xlim(0, 5)
    py.ylabel('CDF for X Vel. Fit')
    py.title('%d stars in %d-%d epochs' % (len(chi2Xred), cnt.min(), cnt.max()))
    
    # Y
    py.subplot(2, 1, 2)
    py.plot(chi2Ysorted, cdfObserved, 'k-')
    py.plot(chi2Ysorted, chi2modelY, 'b-')
    py.xlim(0, 5)
    py.ylabel('CDF for Y Vel. Fit')
    py.xlabel('Reduced Chi^2')
    py.legend(('Observed', 'Model'), loc='lower right')

    if trimMaxEpochs:
        outputFile = root + 'plots/poly_chi2red_cdf_max_dof.png'
    else:
        outputFile = root + 'plots/poly_chi2red_cdf_all_dof.png'

    py.savefig(outputFile)

def plotDegreesOfFreedom(root='./', align='align/align_d_rms_t',
        poly='polyfit_d_points/fit', accel=False, youngOnly=False):

    s = starset.StarSet(root + align)
    s.loadPolyfit(root + poly, arcsec=1, accel=accel)

    if (youngOnly):
        s.onlyYoungDisk()

    if (accel):
        dof = s.getArray('fitXa.dof')
    else:
        dof = s.getArray('fitXv.dof')

    cnt = s.getArray('velCnt')

    dofBins = np.arange(0, cnt.max()+1)

    py.clf()
    py.hist(dof, bins=dofBins, alpha=0.6)
    py.xlabel('Degrees of Freedom')
    py.ylabel('Number of Stars')
    py.title('%d stars in %d-%d epochs' % (len(cnt), cnt.min(), cnt.max()))

    py.savefig(root + 'plots/hist_degrees_of_freedom.png')

def plotVsMag(root='./', align='align/align_d_rms_t',
              poly='polyfit_d_points/fit', youngOnly=False, showLabels=False):

    # Load up our starlist
    s = starset.StarSet(root + align)
    s.loadPolyfit(root + poly, accel=0, arcsec=1)

    if (youngOnly):
        s.onlyYoungDisk()

    chiXred = s.getArray('fitXv.chi2red')
    chiYred = s.getArray('fitYv.chi2red')
    mag = s.getArray('mag')
    names = s.getArray('name')

    py.clf()
    py.subplot(2, 1, 1)
    py.plot(mag, chiXred, 'k^')
    py.xlabel('Magniutde')
    py.ylabel('X Reduced Chi-Sq')
    if (showLabels):
        for ii in range(len(names)):
            py.text(mag[ii], chiXred[ii], names[ii])
    py.ylim(0, 100)

    py.subplot(2, 1, 2)
    py.plot(mag, chiYred, 'k^')
    py.xlabel('Magniutde')
    py.ylabel('Y Reduced Chi-Sq')
    if (showLabels):
        for ii in range(len(names)):
            py.text(mag[ii], chiYred[ii], names[ii])

    py.ylim(0, 100)


def medianError(root='./', align='align/align_d_rms_t',
                poly='polyfit_d_points/fit', points='points_d/',
                youngOnly=False, adjustPoints=False):
    """
    Determine the median error in the Y direction for all stars
    during all epochs.
    """
    # Load up our starlist
    s = starset.StarSet(root + align)

    if (youngOnly):
        s.onlyYoungDisk()

    
    xerrAll = np.arange(0, float)
    yerrAll = np.arange(0, float)

    # We also want to get median errors for varioius
    # magnitude bins.
    err9_10 = np.arange(0, float)
    err10_11 = np.arange(0, float)
    err11_12 = np.arange(0, float)
    err12_13 = np.arange(0, float)
    err13_14 = np.arange(0, float)
    err14_15 = np.arange(0, float)

    for star in s.stars:
        pointsFile = root + points + star.name + '.points'

        pointsTab = asciidata.open(pointsFile)

        # Observed Data
        t = pointsTab[0].tonumpy()
        x = pointsTab[1].tonumpy()
        y = pointsTab[2].tonumpy()
        xerr = pointsTab[3].tonumpy()
        yerr = pointsTab[4].tonumpy()

        xerrAll = np.concatenate((xerrAll, xerr))
        yerrAll = np.concatenate((yerrAll, yerr))

        if (star.mag >=9 and star.mag < 10):
            err9_10 = np.concatenate((err9_10, yerr))
        if (star.mag >=10 and star.mag < 11):
            err10_11 = np.concatenate((err10_11, yerr))
        if (star.mag >=11 and star.mag < 12):
            err11_12 = np.concatenate((err11_12, yerr))
        if (star.mag >=12 and star.mag < 13):
            err12_13 = np.concatenate((err12_13, yerr))
        if (star.mag >=13 and star.mag < 14):
            err13_14 = np.concatenate((err13_14, yerr))
        if (star.mag >=14 and star.mag < 15):
            err14_15 = np.concatenate((err14_15, yerr))

    print 'Median X err: %7.4f (pix)' % np.median(xerrAll)
    print 'Median Y err: %7.4f (pix)' % np.median(yerrAll)

    mag = arange(9.5, 15, 1.0)
    medianMag = array([np.median(err9_10), np.median(err10_11), 
                       np.median(err11_12), np.median(err12_13), 
                       np.median(err13_14), np.median(err14_15)])

    py.clf()
    py.plot(mag, medianMag, 'k^')
    py.xlabel('Magnitude')
    py.ylabel('Median Error (pix)')


    # Stars in K=11-12 bin seem to be well fit. So lets determine
    # the global scale parameter by adjusting the next lowest bin
    # up to that level
    sysErr = np.sqrt(medianMag[2]**2 - medianMag[1]**2) / 1.5
    #sysErr = 0.03

    print 'Systematic Error to be quad summed to Y: ', sysErr

    # --- The sysErr will be added in quadrature to everything.
    # --- Adjustment will only be made in the Y direction.

    if (adjustPoints):
        # Points adjustment must happen on ALL stars
        s = starset.StarSet(root + align)
        
        newPoints = points[:-1] + '_err/'
        print 'Adjusted points saved to ', newPoints
        
        for star in s.stars:
            oldPointsFile = root + points + star.name + '.points'
            newPointsFile = root + newPoints + star.name + '.points'

            pointsTab = asciidata.open(oldPointsFile)

            date = pointsTab[0].tonumpy()
            oldErrX = pointsTab[3].tonumpy()
            oldErrY = pointsTab[4].tonumpy()

            #newErrX = oldErrX
            newErrX = np.sqrt(oldErrX**2 + sysErr**2)
            newErrY = np.sqrt(oldErrY**2 + sysErr**2)

            for row in range(len(oldErrX)):
                pointsTab[3][row] = newErrX[row]
                pointsTab[4][row] = newErrY[row]

            # Temporarily delete AO epochs
            #pointsTab.delete(len(oldErrX) - 2)
            #pointsTab.delete(len(oldErrX) - 5)

            # Temporarily scale AO epochs
            #ao1 = len(oldErrX) - 2
            #pointsTab[3][ao1] = sqrt(oldErrX[ao1]**2 + 0.08**2)
            #pointsTab[4][ao1] = sqrt(oldErrY[ao1]**2 + 0.08**2)
            #ao2 = len(oldErrX) - 5
            #pointsTab[3][ao2] = sqrt(oldErrX[ao2]**2 + 0.08**2)
            #pointsTab[4][ao2] = sqrt(oldErrY[ao2]**2 + 0.08**2)

            pointsTab.writeto(newPointsFile)


def usetexTrue():
    py.rc('text', usetex=True)
    #py.rc('font', family='serif', size=16)
    py.rc('font', **{'family':'sans-serif', 'size':16})
    py.rc('axes', titlesize=20, labelsize=20)
    py.rc('xtick', labelsize=16)
    py.rc('ytick', labelsize=16)

def usetexFalse():
    py.rc('text', usetex=False)
    py.rc('font', family='sans-serif', size=14)
    py.rc('axes', titlesize=16, labelsize=16)
    py.rc('xtick', labelsize=14)
    py.rc('ytick', labelsize=14)
