##################################################
#
#  THIS MUST BE OPENED WITHOUT using
#     ipython -pylab
#  NO -pylab FLAG!!!! Otherwise it will just hang.
#
#
##################################################


from pyraf import iraf
import ephem
import asciidata, pyfits, pickle
import os, sys, math
import numpy as np
import pylab as py
from gcwork import starset
from gcwork import starTables
from gcwork import objects
from gcwork import util
from gcwork import young
from gcwork import orbits
#from gcwork import analyticOrbits
from gcreduce import gcutil
from scipy import stats
import refro, refco
import pyfits
from keckdar import dar
import datetime
import pdb
import histNofill

def astroIsokin(root, outsuffix):
    """
    Plot positional errors using pre-aligned positions for a variety of
    different distances (from tip-tilt star, from Sgr A*, from field center).
    
    INPUTS:
    root - Root of align output of a series of cleaned images.
    outsuffix - Usually choose as epoch name (e.g. '06maylgs1')
    
    OUTPUTS:
    astro_err_tts_<outsuffix>.eps -- Radial/Tangen PosErr vs. Dist. from TTS
    astro_err_sgra_<outsuffix>.eps -- Radial/Tangen PosErr vs. Dist. from SgrA*
    astro_err_cent_<outsuffix>.eps -- Radial/Tangen PosErr vs. Dist from Center
    """
    # Read in the align data
    s = starset.StarSet(root)

    # Need to convert all coordinates to system with origin
    # at the tip-tilt star's location.
    # Measured in the 2004 July LGS NIRC2-wide camera field.
    ttCoord = [9.24, 16.68]  # [E, N]

    # Also need coordinates at avg field center.
    s0 = s.stars[0].e[0]
    # Coords of s0 relative to field center
    s0Coo = [-(s0.xpix - 512.0) * s.t.scale,
              (s0.ypix - 512.0) * s.t.scale]
    # Coords of field center relative to Sgr A*
    cenCoo = [s.stars[0].x - s0Coo[0], s.stars[0].y - s0Coo[1]]
    print 'Coordinates of field center are (%6.2f, %6.2f)' % \
          (cenCoo[0], cenCoo[1])

    starCnt = len(s.stars)

    # Distances from Sgr A*, TT star, and field center.
    r = np.zeros(starCnt, float)
    rTT = np.zeros(starCnt, float)
    rPix = np.zeros(starCnt, float)
    
    # RMS radial/tangent from Sgr A*
    rRms = np.zeros(starCnt, float)
    tRms = np.zeros(starCnt, float)

    # RMS radial/tangent from Sgr A*
    rRmsTT = np.zeros(starCnt, float)
    tRmsTT = np.zeros(starCnt, float)

    # RMS radial/tangent from field center
    rRmsPix = np.zeros(starCnt, float)
    tRmsPix = np.zeros(starCnt, float)

    ecnt = np.zeros(starCnt, int)

    # Now for every epoch, compute the offset from the mean
    i = 0
    for star in s.stars:
        xAvgTT = star.x - ttCoord[0]
        yAvgTT = star.y - ttCoord[1]

        xAvgPix = star.x - cenCoo[0]
        yAvgPix = star.y - cenCoo[0]

        r[i] = np.sqrt(star.x**2 + star.y**2)
        rTT[i] = np.sqrt(xAvgTT**2 + yAvgTT**2)
        rPix[i] = np.sqrt(xAvgPix**2 + yAvgPix**2)

        for epoch in star.e:
            if (epoch.xpix < -999):
                continue
            
            ecnt[i] += 1
            
            xDiff = epoch.x - star.x
            yDiff = epoch.y - star.y

            rDiff = (xDiff * star.x + yDiff * star.y) / r[i]
            tDiff = (xDiff * star.y - yDiff * star.x) / r[i]
            rDiffTT = (xDiff * xAvgTT + yDiff * yAvgTT) / rTT[i]
            tDiffTT = (xDiff * yAvgTT - yDiff * xAvgTT) / rTT[i]
            rDiffPix = (xDiff * xAvgPix + yDiff * yAvgPix) / rPix[i]
            tDiffPix = (xDiff * yAvgPix - yDiff * xAvgPix) / rPix[i]

            rRms[i] += rDiff**2
            tRms[i] += tDiff**2
            rRmsTT[i] += rDiffTT**2
            tRmsTT[i] += tDiffTT**2
            rRmsPix[i] += rDiffPix**2
            tRmsPix[i] += tDiffPix**2

        rRms[i] = np.sqrt(rRms[i] / (ecnt[i] - 1))
        tRms[i] = np.sqrt(tRms[i] / (ecnt[i] - 1))
        rRmsTT[i] = np.sqrt(rRmsTT[i] / (ecnt[i] - 1))
        tRmsTT[i] = np.sqrt(tRmsTT[i] / (ecnt[i] - 1))
        rRmsPix[i] = np.sqrt(rRmsPix[i] / (ecnt[i] - 1))
        tRmsPix[i] = np.sqrt(tRmsPix[i] / (ecnt[i] - 1))

        i += 1

    # Convert RMS into milli-arcsec
    rRms *= 1000.0
    tRms *= 1000.0
    rRmsTT *= 1000.0
    tRmsTT *= 1000.0
    rRmsPix *= 1000.0
    tRmsPix *= 1000.0
    

    py.clf()
    py.subplot(2, 1, 1)
    py.semilogy(rTT, rRmsTT, 'k.')
    py.xlabel('Distance from TTS (arcsec)')
    py.ylabel('Radial Pos. Error (mas)')
    py.axis([12, 26, 0.1, 10])

    py.subplot(2, 1, 2)
    py.semilogy(rTT, tRmsTT, 'k.')
    py.xlabel('Distance from TTS (arcsec)')
    py.ylabel('Tangen. Pos. Error (mas)')
    py.axis([12, 26, 0.1, 10])
    py.savefig('astro_err_tts_%s.eps' % outsuffix)


    py.clf()
    py.subplot(2, 1, 1)
    py.semilogy(r, rRms, 'k.')
    py.xlabel('Distance from Sgr A* (arcsec)')
    py.ylabel('Radial Pos. Error (mas)')
    py.axis([0, 8, 0.1, 10])

    py.subplot(2, 1, 2)
    py.semilogy(r, tRms, 'k.')
    py.xlabel('Distance from Sgr A* (arcsec)')
    py.ylabel('Tangen. Pos. Error (mas)')
    py.axis([0, 8, 0.1, 10])
    py.savefig('astro_err_sgra_%s.eps' % outsuffix)

    py.clf()
    py.subplot(2, 1, 1)
    py.semilogy(r, rRmsPix, 'k.')
    py.xlabel('Distance from Center (arcsec)')
    py.ylabel('Radial Pos. Error (mas)')
    py.axis([0, 8, 0.1, 10])

    py.subplot(2, 1, 2)
    py.semilogy(r, tRmsPix, 'k.')
    py.xlabel('Distance from Center (arcsec)')
    py.ylabel('Tangen. Pos. Error (mas)')
    py.axis([0, 8, 0.1, 10])
    py.savefig('astro_err_cent_%s.eps' % outsuffix)


def plotFwhm(root, strehlFile, outsuffix):
    """
    Plot the FWHM of a series of individual exposures. Pass in the
    results of aligning all the individual exposures. Also pass in the
    irs33N.strehl file.
    """
    # Read in the align data
    s = starset.StarSet(root)

    # Read in the strehl file
    tab = asciidata.open(strehlFile)
    filesFits = tab[0]
    strehlIn = tab[1].tonumpy()
    nmWfeIn = tab[2].tonumpy()
    fwhmIn = tab[3].tonumpy()

    filesIn = [(ff.split('.'))[0] for ff in filesFits] 

    # Match the file names from the strehl file to those used
    # in align.
    tab2 = asciidata.open(root + '.list')
    filesLis = tab[0]
    filesAlign = [(ff.split('_'))[0] for ff in filesLis]

    idx = np.zeros(len(filesAlign), int)
    for ii in range(len(filesAlign)):
        idx[ii] = filesIn.index(filesAlign[ii])

    strehl = strehlIn[idx]
    fwhm = fwhmIn[idx]
    nmWfe = nmWfeIn[idx]

    

def pairIsokin(root, outsuffix):
    """
    Plot positional error from a pairwise analysis.
    """
    # Read in the align data
    s = starset.StarSet(root)

    # Need to convert all coordinates to system with origin
    # at the tip-tilt star's location.
    # Measured in the 2004 July LGS NIRC2-wide camera field.
    ttCoord = [9.24, 16.68]  # [E, N]

    # Also need coordinates at avg field center.
    s0 = s.stars[0].e[0]
    # Coords of s0 relative to field center
    s0Coo = [-(s0.xpix - 512.0) * s.t.scale,
              (s0.ypix - 512.0) * s.t.scale]
    # Coords of field center relative to Sgr A*
    cenCoo = [s.stars[0].x - s0Coo[0], s.stars[0].y - s0Coo[1]]
    print 'Coordinates of field center are (%6.2f, %6.2f)' % \
          (cenCoo[0], cenCoo[1])

    starCnt = len(s.stars)
    epochCnt = len(s.stars[0].e)

    # Need to find the pixel positions of Sgr A* and
    # the TTS in each image. Assume no rotation.
    sgraXpix = np.zeros(epochCnt, float)
    sgraYpix = np.zeros(epochCnt, float)
    ttsXpix = np.zeros(epochCnt, float)
    ttsYpix = np.zeros(epochCnt, float)

    s0 = s.stars[0]
    for ee in range(epochCnt):
        se0 = s0.e[ee]

        sgraXpix[ee] = se0.xorig + (s0.x / s.t.scale)
        sgraYpix[ee] = se0.yorig - (s0.y / s.t.scale)
        ttsXpix[ee] = sgraXpix[ee] - (ttCoord[0] / s.t.scale)
        ttsYpix[ee] = sgraYpix[ee] + (ttCoord[1] / s.t.scale)

    def factorial(n):
        if n:
            foo = n*factorial(n-1)
            print n, foo
            return foo
        else:
            return 1.0

    # Calculate the number of combinations we will have.
    # Combination of "starCnt" things taken 2 at a time
    #combCnt = factorial(starCnt) / (factorial(starCnt - 2) * factorial(2))
    combCnt = starCnt * (starCnt - 1) / 2
    print starCnt, combCnt
    sepX = np.zeros(combCnt, float)
    sepY = np.zeros(combCnt, float)
    sepRtts = np.zeros(combCnt, float)
    sepTtts = np.zeros(combCnt, float)
    sepRcen = np.zeros(combCnt, float)
    sepTcen = np.zeros(combCnt, float)
    rmsX = np.zeros(combCnt, float)
    rmsY = np.zeros(combCnt, float)
    rmsRtts = np.zeros(combCnt, float)
    rmsTtts = np.zeros(combCnt, float)
    rmsRcen = np.zeros(combCnt, float)
    rmsTcen = np.zeros(combCnt, float)
    rTT = np.zeros(combCnt, float)
    rCen = np.zeros(combCnt, float)
    
    cnt = np.zeros(combCnt, int)

    # Now for every epoch, compute the offset from the mean
    for ee in range(epochCnt):
        if ((ee % 20) == 0):
            print 'Working on epoch ', ee
            
        xpix = s.getArrayFromEpoch(ee, 'xorig')
        ypix = s.getArrayFromEpoch(ee, 'yorig')

        ii = 0
        for ss in range(starCnt):
            xdiff = xpix[ss] - xpix[ss+1:starCnt]
            ydiff = ypix[ss] - ypix[ss+1:starCnt]

            npairs = len(xdiff)

            x0 = xpix[ss] + (xdiff / 2.0)
            y0 = ypix[ss] + (ydiff / 2.0)
            #x0 = np.zeros(npairs, float) + xpix[ss]
            #y0 = np.zeros(npairs, float) + ypix[ss]

            distXtts = x0 - ttsXpix[ee]
            distYtts = y0 - ttsYpix[ee]
#             distXcen = x0 - 512.0
#             distYcen = y0 - 512.0
            distXcen = x0 - sgraXpix[ee]
            distYcen = y0 - sgraYpix[ee]

            distTT = np.sqrt(distXtts**2 + distYtts**2)
            distCen = np.sqrt(distXcen**2 + distYcen**2)

            rdiffTT = (xdiff * distXtts + ydiff * distYtts) / distTT
            tdiffTT = (xdiff * distYtts - ydiff * distXtts) / distTT
            rdiffCen = (xdiff * distXcen + ydiff * distYcen) / distCen
            tdiffCen = (xdiff * distYcen - ydiff * distXcen) / distCen

#             if (ss == 0):
#                 print xdiff[0], ydiff[0], \
#                       rdiffTT[0], tdiffTT[0], rdiffCen[0], tdiffCen[0]
                      

            # Build up the avg/rms across all epochs
            sepX[ii:ii+npairs] += xdiff
            sepY[ii:ii+npairs] += ydiff
            sepRtts[ii:ii+npairs] += rdiffTT
            sepTtts[ii:ii+npairs] += tdiffTT
            sepRcen[ii:ii+npairs] += rdiffCen
            sepTcen[ii:ii+npairs] += tdiffCen

            rmsX[ii:ii+npairs] += xdiff**2
            rmsY[ii:ii+npairs] += ydiff**2
            rmsRtts[ii:ii+npairs] += rdiffTT**2
            rmsTtts[ii:ii+npairs] += tdiffTT**2
            rmsRcen[ii:ii+npairs] += rdiffCen**2
            rmsTcen[ii:ii+npairs] += tdiffCen**2

            # Average distance from TTS and Cen
            rTT[ii:ii+npairs] += distTT
            rCen[ii:ii+npairs] += distCen
            
            cnt[ii:ii+npairs] += 1

            ii += npairs

    sepX /= cnt
    sepY /= cnt
    sepRtts /= cnt
    sepTtts /= cnt
    sepRcen /= cnt
    sepTcen /= cnt

    rTT /= cnt
    rCen /= cnt

    # Now compute the RMS
    rmsX /= (cnt - 1)
    rmsY /= (cnt - 1)
    rmsRtts /= (cnt - 1)
    rmsTtts /= (cnt - 1)
    rmsRcen /= (cnt - 1)
    rmsTcen /= (cnt - 1)

    rmsX -= (cnt / (cnt - 1.0)) * sepX**2
    rmsY -= (cnt / (cnt - 1.0)) * sepY**2
    rmsRtts -= (cnt / (cnt - 1.0)) * sepRtts**2
    rmsTtts -= (cnt / (cnt - 1.0)) * sepTtts**2
    rmsRcen -= (cnt / (cnt - 1.0)) * sepRcen**2
    rmsTcen -= (cnt / (cnt - 1.0)) * sepTcen**2

    rmsX = np.sqrt(rmsX)
    rmsY = np.sqrt(rmsY)
    rmsRtts = np.sqrt(rmsRtts)
    rmsTtts = np.sqrt(rmsTtts)
    rmsRcen = np.sqrt(rmsRcen)
    rmsTcen = np.sqrt(rmsTcen)

    # Convert RMS into milli-arcsec
    rmsX *= s.t.scale * 1000.0
    rmsY *= s.t.scale * 1000.0
    rmsRtts *= s.t.scale * 1000.0
    rmsTtts *= s.t.scale * 1000.0
    rmsRcen *= s.t.scale * 1000.0
    rmsTcen *= s.t.scale * 1000.0

    sepX *= s.t.scale
    sepY *= s.t.scale
    sepRtts *= s.t.scale
    sepTtts *= s.t.scale
    sepRcen *= s.t.scale
    sepTcen *= s.t.scale

    rTT *= s.t.scale
    rCen *= s.t.scale

    pntsize = 2

    py.clf()
    py.subplot(2, 1, 1)
    py.semilogy(rTT, rmsRtts, 'k.', markersize=pntsize)
    py.xlabel('Distance from TTS (arcsec)')
    py.ylabel('Radial Pos. Error (mas)')
    #py.axis([12, 26, 0.1, 10])

    py.subplot(2, 1, 2)
    py.semilogy(rTT, rmsTtts, 'k.', markersize=pntsize)
    py.xlabel('Distance from TTS (arcsec)')
    py.ylabel('Tangen. Pos. Error (mas)')
    #py.axis([12, 26, 0.1, 10])
    py.savefig('astro_err_tts_%s.eps' % outsuffix)

    py.clf()
    py.subplot(2, 1, 1)
    py.semilogy(rCen, rmsRcen, 'k.', markersize=pntsize)
    py.xlabel('Distance from Sgra (arcsec)')
    py.ylabel('Radial Pos. Error (mas)')
    #py.axis([12, 26, 0.1, 10])

    py.subplot(2, 1, 2)
    py.semilogy(rCen, rmsTcen, 'k.', markersize=pntsize)
    py.xlabel('Distance from Sgra (arcsec)')
    py.ylabel('Tangen. Pos. Error (mas)')
    #py.axis([12, 26, 0.1, 10])
    py.savefig('astro_err_cent_%s.eps' % outsuffix)

    py.clf()
    py.subplot(2, 1, 1)
    py.semilogy(abs(sepRtts), rmsRtts, 'k.', markersize=pntsize)
    py.xlabel('Seperation Along TT Radial (arcsec)')
    py.ylabel('Radial Pos. Error (mas)')
    #py.axis([12, 26, 0.1, 10])

    py.subplot(2, 1, 2)
    py.semilogy(abs(sepTtts), rmsTtts, 'k.', markersize=pntsize)
    py.xlabel('Seperation Along TT Tangential (arcsec)')
    py.ylabel('Tangen. Pos. Error (mas)')
    #py.axis([12, 26, 0.1, 10])
    py.savefig('astro_sep_tts_%s.eps' % outsuffix)

    py.clf()
    py.subplot(2, 1, 1)
    py.semilogy(abs(sepRcen), rmsRcen, 'k.', markersize=pntsize)
    py.xlabel('Seperation Along Cen Radial (arcsec)')
    py.ylabel('Radial Pos. Error (mas)')
    #py.axis([0, 8, 0.1, 10])

    py.subplot(2, 1, 2)
    py.semilogy(abs(sepTcen), rmsTcen, 'k.', markersize=pntsize)
    py.xlabel('Seperation Along Cen Tangential (arcsec)')
    py.ylabel('Tangen. Pos. Error (mas)')
    #py.axis([0, 8, 0.1, 10])
    py.savefig('astro_sep_cent_%s.eps' % outsuffix)

    return (rTT, sepRtts, sepTtts, rmsRtts, rmsTtts)

def isoplanatic(root, outsuffix, refSrc='S1-5'):
    # Read in the align data
    s = starset.StarSet(root)

    # Also need coordinates relative to a bright star near
    # the center of the FOV.
    names = s.getArray('name')
    rid = names.index(refSrc)

    starCnt = len(s.stars)
    epochCnt = len(s.stars[0].e)

    ttCoord = [9.24, 16.68]  # [E, N]

    # Positions and Errors to some reference source.
    x = np.zeros(starCnt, float)
    y = np.zeros(starCnt, float)
    rmsX = np.zeros(starCnt, float)
    rmsY = np.zeros(starCnt, float)
    rmsR = np.zeros(starCnt, float)
    rmsT = np.zeros(starCnt, float)
    rmsRarc = np.zeros(starCnt, float)
    rmsTarc = np.zeros(starCnt, float)
    r = np.zeros(starCnt, float)
    rarc = np.zeros(starCnt, float)

    # Positions and Errors relative to TTS
    avgXtt = np.zeros(starCnt, float)
    avgYtt = np.zeros(starCnt, float)
    rmsRtt = np.zeros(starCnt, float)
    rmsTtt = np.zeros(starCnt, float)
    rtt = np.zeros(starCnt, float)

    # Get the reference coordinates at each epoch
    refStar = s.stars[rid]
    refCooX = np.zeros(epochCnt, float)
    refCooY = np.zeros(epochCnt, float)
    for ee in range(epochCnt):
        refCooX[ee] = refStar.e[ee].xorig
        refCooY[ee] = refStar.e[ee].yorig

    # Now for every epoch, compute the offset from the mean
    py.clf()
    for ss in range(starCnt):
        star = s.stars[ss]

        # Original pixel positions
        xpix = star.getArrayAllEpochs('xorig')
        ypix = star.getArrayAllEpochs('yorig')

        xpos = (xpix - refCooX) * s.t.scale
        ypos = (ypix - refCooY) * s.t.scale

        idx = (np.where(xpix > -999))[0]
        xpos = xpos[idx]
        ypos = ypos[idx]

        avgX = xpos.mean()
        avgY = ypos.mean()
        x[ss] = avgX
        y[ss] = avgY
        r[ss] = np.sqrt(avgX**2 + avgY**2)
        rmsX[ss] = xpos.std() * 1000.0
        rmsY[ss] = ypos.std() * 1000.0

        xdiff = xpos - avgX
        ydiff = ypos - avgY

        rdiff = (xdiff * avgX + ydiff * avgY) / r[ss]
        tdiff = (xdiff * avgY - ydiff * avgX) / r[ss]

        rmsR[ss] = np.sqrt((rdiff**2).sum() / (len(rdiff) - 1)) * 1000.0
        rmsT[ss] = np.sqrt((tdiff**2).sum() / (len(tdiff) - 1)) * 1000.0

        # Tip-tilt analysis
        ttLen = np.sqrt(ttCoord[0]**2 + ttCoord[1]**2)
        rttdiff = (xdiff * ttCoord[0] + ydiff * ttCoord[1]) / ttLen
        tttdiff = (xdiff * ttCoord[1] + ydiff * ttCoord[0]) / ttLen

        avgXtt[ss] = avgX - -ttCoord[0]
        avgYtt[ss] = avgY - ttCoord[1]
        rmsRtt[ss] = np.sqrt((rttdiff**2).sum() / (len(rttdiff) - 1)) * 1000.0
        rmsTtt[ss] = np.sqrt((tttdiff**2).sum() / (len(tttdiff) - 1)) * 1000.0
        rtt[ss] = np.sqrt(avgXtt[ss]**2 + avgYtt[ss]**2)

        # Do the same thing with aligned data relative to Sgr A*
        xarc = star.getArrayAllEpochs('x')
        yarc = star.getArrayAllEpochs('y')
        xarc = xarc[idx]
        yarc = yarc[idx]
        
        rarc[ss] = np.sqrt(star.x**2 + star.y**2)

        xdiff = xarc - star.x
        ydiff = yarc - star.y

        rdiff = (xdiff * star.x + ydiff * star.y) / rarc[ss]
        tdiff = (xdiff * star.y - ydiff * star.x) / rarc[ss]

        rmsRarc[ss] = np.sqrt((rdiff**2).sum() / (len(rdiff) - 1)) * 1000.0
        rmsTarc[ss] = np.sqrt((tdiff**2).sum() / (len(tdiff) - 1)) * 1000.0

        
    pntsize = 6
    mag = s.getArray('mag')

    mid10 = (np.where(mag <= 10.0))[0]
    mid12 = (np.where((mag > 10.0) & (mag <= 12.0)))[0]
    mid14 = (np.where((mag > 12.0) & (mag <= 14.0)))[0]
    mid16 = (np.where((mag > 14.0) & (mag <= 16.0)))[0]

    #idx = (np.where(mag <= 15.0))[0]
    idx = (np.where((mag > 10.0) & (mag <= 14.5)))[0]
    idx = (np.where((mag > 10.5) & (mag <= 13)))[0]
    print len(idx), ' out of ', starCnt


    py.clf()
    py.subplot(2, 1, 1)
    py.semilogy(r[idx], rmsX[idx], 'k.', markersize=pntsize)
    py.xlabel('X Distance (arcsec)')
    py.ylabel('X Pos. Error (mas)')
    #py.axis([12, 26, 0.1, 10])

    py.subplot(2, 1, 2)
    py.semilogy(r[idx], rmsY[idx], 'k.', markersize=pntsize)
    py.xlabel('Y Distance (arcsec)')
    py.ylabel('Y Pos. Error (mas)')
    #py.axis([12, 26, 0.1, 10])
    py.savefig('astro_iso_xy_%s.eps' % outsuffix)
    py.savefig('astro_iso_xy_%s.png' % outsuffix)



    # Fit a line to the radial and tangential RMS errors to 
    # see if the difference in slope corresponds to the sqrt(3) that
    # is expected.
    rparams = np.polyfit(r[idx], rmsRarc[idx], 1)
    rfit = np.poly1d(rparams)
    rmsRfit = rfit(r[idx])

    tparams = np.polyfit(r[idx], rmsTarc[idx], 1)
    tfit = np.poly1d(tparams)
    rmsTfit = tfit(r[idx])

    print 'sqrt(3) = ', math.sqrt(3.)
    print (rmsTfit / rmsRfit).mean()
    print tparams / rparams

    print 'Radial RMS error line fit: ', rparams
    print 'Tangen RMS error line fit: ', tparams

    py.clf()
    py.subplot(1, 1, 1)
    py.plot(r[idx], rmsRarc[idx], 'r.', markersize=pntsize, label='Radial')
    py.plot(r[idx], rmsTarc[idx], 'b.', markersize=pntsize, label='Tangen')
    py.plot(r[idx], rmsRfit, 'r--')
    py.plot(r[idx], rmsTfit, 'b--')
    py.xlabel('Distance from %s (arcsec)' % (refSrc))
    py.ylabel('Positional Error (mas)')
    py.legend()
    py.savefig('astro_iso_rt_1panel_%s.png' % outsuffix)


    py.clf()
    py.subplot(2, 1, 1)
    py.plot(r[idx], rmsRarc[idx], 'k.', markersize=pntsize)
    py.plot(r[idx], rmsRfit, 'k--')
#     py.semilogy(r[mid16], rmsR[mid16], 'g.', markersize=pntsize)
#     py.semilogy(r[mid14], rmsR[mid14], 'm.', markersize=pntsize)
#     py.semilogy(r[mid12], rmsR[mid12], 'b.', markersize=pntsize)
#     py.semilogy(r[mid10], rmsR[mid10], 'r.', markersize=pntsize)
    py.xlabel('Distance from %s (arcsec)' % (refSrc))
    py.ylabel('Radial Pos. Error (mas)')

    py.subplot(2, 1, 2)
    py.plot(r[idx], rmsTarc[idx], 'k.', markersize=pntsize)
    py.plot(r[idx], rmsTfit, 'k--')
#     f1 = py.semilogy(r[mid16], rmsT[mid16], 'g.', markersize=pntsize)
#     f2 = py.semilogy(r[mid14], rmsT[mid14], 'm.', markersize=pntsize)
#     f3 = py.semilogy(r[mid12], rmsT[mid12], 'b.', markersize=pntsize)
#     f4 = py.semilogy(r[mid10], rmsT[mid10], 'r.', markersize=pntsize)
#     py.legend((f4, f3, f2, f1), \
#            ('K <= 10', '10 < K <= 12', '12 < K <= 14', '14 < K <= 16'), \
#            loc=4, numpoints=3, \
#            prop=matplotlib.font_manager.FontProperties('smaller'))
    py.xlabel('Distance from %s (arcsec)' % (refSrc))
    py.ylabel('Tangen. Pos. Error (mas)')
    py.savefig('astro_iso_rt_%s.eps' % outsuffix)
    py.savefig('astro_iso_rt_%s.png' % outsuffix)

#     py.clf()
#     py.subplot(2, 1, 1)
#     py.semilogy(rarc[idx], rmsRarc[idx], 'k.', markersize=pntsize)
#     py.xlabel('Distance from Laser (arcsec)')
#     py.ylabel('Radial Pos. Error (mas)')

#     py.subplot(2, 1, 2)
#     py.semilogy(rarc[idx], rmsTarc[idx], 'k.', markersize=pntsize)
#     py.xlabel('Distance from Laser (arcsec)')
#     py.ylabel('Tangen. Pos. Error (mas)')
#     py.savefig('astro_iso_rtarc_%s.eps' % outsuffix)


#     idx = (np.where((mag > 10.0) & (mag <= 14.0) & (r > 3) & (r < 4)))[0]
#     rmstt = np.sqrt(rmsRtt**2 + rmsTtt**2)
#     py.clf()
#     py.subplot(2, 1, 1)
#     py.semilogy(rtt[idx], rmsRtt[idx], 'k.', markersize=pntsize)
#     py.xlabel('Distance from TTS (arcsec)')
#     py.ylabel('Radial Pos. Error (mas)')

#     py.subplot(2, 1, 2)
#     py.semilogy(rtt[idx], rmsTtt[idx], 'k.', markersize=pntsize)
#     py.xlabel('Distance from TTS (arcsec)')
#     py.ylabel('Tangen. Pos. Error (mas)')
#     py.savefig('astro_iso_rt_tts_%s.eps' % outsuffix)



#     py.clf()
#     rmsAll = np.sqrt(rmsX**2 + rmsY**2)

#     for ii in idx:
#         msT = rmsTtt[ii]*10
#         msR = rmsRtt[ii]*10
#         msX = rmsX[ii]*10
#         msY = rmsY[ii]*10
#         plot([x[ii]], [y[ii]], 'b_', markersize=msX)
#         plot([x[ii]], [y[ii]], 'b|', markersize=msY)
    
#     py.scatter(x[idx], y[idx], (rmsT[idx]**2)*50, marker='o', edgecolor='b', \
#             facecolor='w')
#     py.scatter(x[idx], y[idx], (rmsR[idx]**2)*50, marker='s', edgecolor='r', \
#             facecolor='w')
#     py.scatter(x[idx], y[idx], log10(rmsAll[idx])*500, marker='o', edgecolor='b', \
#             facecolor='w')


#     ttdistX = x - (-ttCoord[0])
#     ttdistY = y - ttCoord[1]
#     ttdist = sqrt(ttdistX**2 + ttdistY**2)
#     rid = (np.where((r > 3) & (r < 4) & (mag > 11) & (mag < 13)))[0]
#     for rr in rid:
#         print '%13s %5.2f  %5.2f  %4.2f' % \
#               (names[rr], r[rr], ttdist[rr], mag[rr])

    #py.clf()
    #py.plot(ttdist[rid], rmstt[rid], 'b^')
    #py.plot(ttdist[rid], rmsX[rid], 'r+')
    #py.plot(ttdist[rid], rmsY[rid], 'bx')

def compareDetections(root1, root2):
    s1 = starset.StarSet(root1)
    s2 = starset.StarSet(root2)

    cnt1 = len(s1.stars)
    cnt2 = len(s2.stars)
    print cnt1, cnt2
    corrCnt1 = np.zeros(cnt1)
    corrCnt1_low = np.zeros(cnt1)
    corrCnt2 = np.zeros(cnt2)
    corrCnt2_low = np.zeros(cnt2)

    for ss in range(cnt1):
        star = s1.stars[ss]
        corr = np.array([star.e[ee].corr for ee in range(len(star.years))])

        idx = (np.where(corr > 0.8))[0]
        corrCnt1[ss] = len(idx)

        idx = (np.where((corr > 0) & (corr <= 0.8)))[0]
        if (len(idx) > 0):
            corrCnt1_low[ss] = len(idx)

    for ss in range(cnt2):
        star = s2.stars[ss]
        corr = np.array([star.e[ee].corr for ee in range(len(star.years))])

        idx = np.where((corr > 0.8))[0]
        corrCnt2[ss] = len(idx)

        idx = (np.where((corr > 0) & (corr < 0.8)))[0]
        if (len(idx) > 0):
            corrCnt2_low[ss] = len(idx)

    clf()
    (n1, b1, p1) = py.hist(corrCnt1, bins=np.arange(0,29))
    (n2, b2, p2) = py.hist(corrCnt2, bins=np.arange(0,29))
    (n1a, b1a, p1a) = py.hist(corrCnt1_low, bins=np.arange(0,29))
    (n2a, b2a, p2a) = py.hist(corrCnt2_low, bins=np.arange(0,29))

    py.clf()
    py.subplot(211)
    py.plot(b1, n1, 'bo-')
    py.plot(b2, n2, 'ro-')
    py.legend((root1, root2), loc='upper left')
    py.xlabel('Number of Epochs Detected (c > 0.8)')
    py.ylabel('Number of Stars')

    py.subplot(212)
    py.plot(b1a, n1a, 'bo-')
    py.plot(b2a, n2a, 'ro-')
    py.legend((root1, root2), loc='upper left')
    py.xlabel('Number of Epochs Detected (c < 0.8)')
    py.ylabel('Number of Stars')

    py.savefig('plots/compareDetections.png')

    print corrCnt2
    print corrCnt2_low

def plotPosError(starlist, raw=False, suffix='', radius=4, magCutOff=15.0,
                 title=True):
    """
    Make three standard figures that show the data quality 
    from a *_rms.lis file. 

    1. astrometric error as a function of magnitude.
    2. photometric error as a function of magnitude.
    3. histogram of number of stars vs. magnitude.

    Use raw=True to plot the individual stars in plots 1 and 2.
    """
    # Load up the starlist
    lis = asciidata.open(starlist)

    # Assume this is NIRC2 data.
    scale = 0.00995
    
    name = lis[0]._data
    mag = lis[1].tonumpy()
    x = lis[3].tonumpy()
    y = lis[4].tonumpy()
    xerr = lis[5].tonumpy()
    yerr = lis[6].tonumpy()
    snr = lis[7].tonumpy()
    corr = lis[8].tonumpy()

    merr = 1.086 / snr

    # Convert into arsec offset from field center
    # We determine the field center by assuming that stars
    # are detected all the way out the edge.
    xhalf = x.max() / 2.0
    yhalf = y.max() / 2.0
    x = (x - xhalf) * scale
    y = (y - yhalf) * scale
    xerr *= scale * 1000.0
    yerr *= scale * 1000.0

    r = np.hypot(x, y)
    err = (xerr + yerr) / 2.0

    magStep = 1.0
    radStep = 1.0
    magBins = np.arange(10.0, 20.0, magStep)
    radBins = np.arange(0.5, 9.5, radStep)
    
    errMag = np.zeros(len(magBins), float)
    errRad = np.zeros(len(radBins), float)
    merrMag = np.zeros(len(magBins), float)
    merrRad = np.zeros(len(radBins), float)

    ##########
    # Compute errors in magnitude bins
    ########## 
    #print '%4s  %s' % ('Mag', 'Err (mas)')
    for mm in range(len(magBins)):
        mMin = magBins[mm] - (magStep / 2.0)
        mMax = magBins[mm] + (magStep / 2.0)
        idx = (np.where((mag >= mMin) & (mag < mMax) & (r < radius)))[0]

        if (len(idx) > 0):
            errMag[mm] = np.median(err[idx])
            merrMag[mm] = np.median(merr[idx])
        
        #print '%4.1f  %5.2f' % (magBins[mm], errMag[mm])
        
                       
    ##########
    # Compute errors in radius bins
    ########## 
    for rr in range(len(radBins)):
        rMin = radBins[rr] - (radStep / 2.0)
        rMax = radBins[rr] + (radStep / 2.0)
        idx = (np.where((r >= rMin) & (r < rMax) & (mag < magCutOff)))[0]

        if (len(idx) > 0):
            errRad[rr] = np.median(err[idx])
            merrRad[rr] = np.median(err[idx])

    idx = (np.where((mag < magCutOff) & (r < radius)))[0]
    errMedian = np.median(err[idx])

    ##########
    #
    # Plot astrometry errors
    #
    ##########
 
    # Remove figures if they exist -- have to do this
    # b/c sometimes the file won't be overwritten and
    # the program crashes saying 'Permission denied'
    if os.path.exists('plotPosError%s.png' % suffix):
        os.remove('plotPosError%s.png' % suffix)
    if os.path.exists('plotMagError%s.png' % suffix):
        os.remove('plotMagError%s.png' % suffix)
    if os.path.exists('plotNumStars%s.png' % suffix):
        os.remove('plotNumStars%s.png' % suffix)

    if os.path.exists('plotPosError%s.eps' % suffix):
        os.remove('plotPosError%s.eps' % suffix)
    if os.path.exists('plotMagError%s.eps' % suffix):
        os.remove('plotMagError%s.eps' % suffix)
    if os.path.exists('plotNumStars%s.eps' % suffix):
        os.remove('plotNumStars%s.eps' % suffix)

    py.figure(figsize=(6,6))
    py.clf()
    if (raw == True):
        idx = (np.where(r < radius))[0]
        py.semilogy(mag[idx], err[idx], 'k.')
        
    py.semilogy(magBins, errMag, 'g.-')
    py.axis([8, 22, 5e-2, 30.0])
    py.xlabel('K Magnitude for r < %4.1f"' % radius, fontsize=16)
    py.ylabel('Positional Uncertainty (mas)', fontsize=16)
    if title == True:
        py.title(starlist)
    
    py.savefig('plotPosError%s.png' % suffix)
    #py.savefig('plotPosError%s.eps' % suffix)

    ##########
    #
    # Plot photometry errors
    #
    ##########
    py.clf()
    if (raw == True):
        idx = (np.where(r < radius))[0]
        py.plot(mag[idx], merr[idx], 'k.')
        
    py.plot(magBins, merrMag, 'g.-')
    py.axis([8, 22, 0, 0.15])
    py.xlabel('K Magnitude for r < %4.1f"' % radius)
    py.ylabel('Photo. Uncertainty (mag)')
    py.title(starlist)
    
    py.savefig('plotMagError%s.png' % suffix)
    py.savefig('plotMagError%s.eps' % suffix)

    ##########
    # 
    # Plot histogram of number of stars detected
    #
    ##########
    py.clf()
    idx = (np.where(r < radius))[0]
    (n, bb, pp) = py.hist(mag[idx], bins=np.arange(9, 22, 0.5))
    py.xlabel('K Magnitude for r < %4.1f"' % radius)
    py.ylabel('Number of Stars')

    py.savefig('plotNumStars%s.png' % suffix)
    py.savefig('plotNumStars%s.eps' % suffix)

    # Find the peak of the distribution
    maxHist = n.argmax()
    maxBin = bb[maxHist]


    ##########
    # 
    # Save relevant numbers to an output file.
    #
    ##########
    # Print out some summary information
    print 'Number of detections: %4d' % len(mag)
    print 'Median Pos Error (mas) for K < %2i, r < %4.1f:  %5.2f' % \
          (magCutOff, radius, errMedian)
    print 'Median Mag Error (mag) for K < %2i, r < %4.1f:  %5.2f' % \
          (magCutOff, radius, np.median(merr[idx]))
    print 'Turnover mag = %4.1f' % (maxBin)


    out = open('plotPosError%s.txt' % suffix, 'w')
    out.write('Number of detections: %4d\n' % len(mag))
    out.write('Median Pos Error (mas) for K < %2i, r < %4.1f:  %5.2f\n' % \
          (magCutOff, radius, errMedian))
    out.write('Median Mag Error (mag) for K < %2i, r < %4.1f:  %5.2f\n' % \
          (magCutOff, radius, np.median(merr[idx])))
    out.write('Turnover mag = %4.1f\n' % (maxBin))
    out.close()
    


def compDistort(root1, root2, refSrc='irs16SW-E', reqFrames=6, outsuffix=''):
    """
    Compare two different aligned data sets with different distortion
    solutions.

    Inputs:
    root1 -- The old pre-ship review distortion files, aligned and trimmed.
    root2 -- The new distortion files, aligned and trimmed.
    refSrc -- The name of the source to use as a reference. Must be in all
              frames.
    reqFrames -- The required number of frames that a source must be in.
              Typically you want a star to be in 2 or more pointings. This
              translates into numbers of frames based on how many frames were
              taken at each pointing.
    outsuffix - suffix to identify the distortion solutions being compared.
    """

    if (outsuffix != ''):
        outsuffix = '_' + outsuffix

    s1 = starset.StarSet(root1)
    s2 = starset.StarSet(root2)

    cnt1 = len(s1.stars)
    cnt2 = len(s2.stars)
    epochCnt = len(s2.stars[0].e)

    names1 = s1.getArray('name')
    names2 = s2.getArray('name')

    rid1 = names1.index(refSrc)
    rid2 = names2.index(refSrc)

    avgX1 = np.zeros(cnt1, float)
    avgY1 = np.zeros(cnt1, float)
    stdX1 = np.zeros(cnt1, float)
    stdY1 = np.zeros(cnt1, float)

    avgX2 = np.zeros(cnt1, float)
    avgY2 = np.zeros(cnt1, float)
    stdX2 = np.zeros(cnt1, float)
    stdY2 = np.zeros(cnt1, float)

    avgX1p = np.zeros(cnt1, float)
    avgY1p = np.zeros(cnt1, float)
    stdX1p = np.zeros(cnt1, float)
    stdY1p = np.zeros(cnt1, float)

    avgX2p = np.zeros(cnt1, float)
    avgY2p = np.zeros(cnt1, float)
    stdX2p = np.zeros(cnt1, float)
    stdY2p = np.zeros(cnt1, float)

    numVals = np.zeros(cnt1, float)
    numDthrs = np.zeros(cnt1, float)

    # Get the reference coordinates at each epoch
    refStar1 = s1.stars[rid1]
    refStar2 = s2.stars[rid2]
    refCooX1 = np.zeros(epochCnt, float)
    refCooY1 = np.zeros(epochCnt, float)
    refCooX2 = np.zeros(epochCnt, float)
    refCooY2 = np.zeros(epochCnt, float)
    for ee in range(epochCnt):
        refCooX1[ee] = refStar1.e[ee].xorig
        refCooY1[ee] = refStar1.e[ee].yorig
        refCooX2[ee] = refStar2.e[ee].xorig
        refCooY2[ee] = refStar2.e[ee].yorig

    xnew = s2.getArray('x')
    ynew = s2.getArray('y')
    xold = s1.getArray('x')
    yold = s1.getArray('y')

    #centErr = 0.67

    for j1 in range(cnt1):
        diff = np.sqrt((xnew - xold[j1])**2 + (ynew - yold[j1])**2)
        sdx = diff.argsort()

        j2 = sdx[0]

        if (diff[j2] > 0.03):
            stdX1[j1] = -1
            continue

        # For named sources, check that names match
        if (names1[j1].find("star") == -1):
            try:
                tmp = names2.index(names1[j1])
                if (tmp != j2):
                    print 'Mismatch occurred: %s (old) vs. %s (new) sep=%f' % \
                          (names1[j1], names2[j2], diff[j2])
            except ValueError:
                # Do nothing
                print 'This should never occur.', names1[j1], names2[j2], \
                      diff[j2]
            
                stdX1[j1] = -1

        star1 = s1.stars[j1]
        star2 = s2.stars[j2]

        # Aligned pixel coordinates:
        xpix1 = star1.getArrayAllEpochs('xpix')
        xpix2 = star2.getArrayAllEpochs('xpix')
        ypix1 = star1.getArrayAllEpochs('ypix')
        ypix2 = star2.getArrayAllEpochs('ypix')

        # These are in arcseconds, don't want these
        # (we don't do anything with them anyway):
        x1 = star1.getArrayAllEpochs('x')
        y1 = star1.getArrayAllEpochs('y')
        x2 = star2.getArrayAllEpochs('x')
        y2 = star2.getArrayAllEpochs('y')

        # Original pixel coordinates:
        xorig1 = star1.getArrayAllEpochs('xorig')
        yorig1 = star1.getArrayAllEpochs('yorig')
        xorig2 = star2.getArrayAllEpochs('xorig')
        yorig2 = star2.getArrayAllEpochs('yorig')

        # We want the deltas from 16SW-E in original detector coords
        xdiff1 = xorig1 - refCooX1
        ydiff1 = yorig1 - refCooY1
        xdiff2 = xorig2 - refCooX2
        ydiff2 = yorig2 - refCooY2

        idx = (np.where((xpix1 > -999) & (xpix2 > -999)))[0]

        # Require that the star be in at least two different
        # pointings.
        #if (len(idx) < reqFrames):
        #    stdX1[j1] = -1
        #    continue

        x1 = x1[idx]
        y1 = y1[idx]
        x2 = x2[idx]
        y2 = y2[idx]
        xpix1 = xpix1[idx]
        ypix1 = ypix1[idx]
        xpix2 = xpix2[idx]
        ypix2 = ypix2[idx]
        xorig1 = xorig1[idx]
        yorig1 = yorig1[idx]
        xorig2 = xorig2[idx]
        yorig2 = yorig2[idx]

        xdiff1 = xdiff1[idx]
        ydiff1 = ydiff1[idx]
        xdiff2 = xdiff2[idx]
        ydiff2 = ydiff2[idx]

        if (names1[j1] == 'irs16C'):
            print xdiff1
            print xdiff2

        numVals[j1] = len(xdiff2) 

        # Setup arrays for the averages
        x1a = []
        y1a = []
        x2a = []
        y2a = []
        xdiff1a = []
        ydiff1a = []
        xdiff2a = []
        ydiff2a = []
        xcentErr1 = []
        ycentErr1 = []
        xcentErr2 = []
        ycentErr2 = []
        dthrPos = []
        # Get the average delta at each *pointing* for this star
        # Only use the frames that have the star in all 3 exposures
        if (0 in idx and 1 in idx and 2 in idx):
            numDthrs[j1] += 1
            dthrPos = np.concatenate([dthrPos, [0]])
            x1a = np.concatenate([x1a,[x1[0:3].mean()]])
            y1a = np.concatenate([y1a,[y1[0:3].mean()]])
            x2a = np.concatenate([x2a,[x2[0:3].mean()]])
            y2a = np.concatenate([y2a,[y2[0:3].mean()]])
            xdiff1a = np.concatenate([xdiff1a,[xdiff1[0:3].mean()]])
            ydiff1a = np.concatenate([ydiff1a,[ydiff1[0:3].mean()]])
            xdiff2a = np.concatenate([xdiff2a,[xdiff2[0:3].mean()]])
            ydiff2a = np.concatenate([ydiff2a,[ydiff2[0:3].mean()]])
            #centErr = np.concatenate([centErr,[0.074]])
            xcentErrTmp1 = xpix1[0:3].std(ddof=1)
            ycentErrTmp1 = ypix1[0:3].std(ddof=1)
            xcentErr1 = np.concatenate([xcentErr1,[xcentErrTmp1]])
            ycentErr1 = np.concatenate([ycentErr1,[ycentErrTmp1]])
            xcentErrTmp2 = xpix2[0:3].std(ddof=1)
            ycentErrTmp2 = ypix2[0:3].std(ddof=1)
            xcentErr2 = np.concatenate([xcentErr2,[xcentErrTmp2]])
            ycentErr2 = np.concatenate([ycentErr2,[ycentErrTmp2]])
        if (3 in idx and 4 in idx and 5 in idx):
            numDthrs[j1] += 1
            dthrPos = np.concatenate([dthrPos, [1]])
            foo3 = np.where(idx == 3)[0]
            foo4 = np.where(idx == 4)[0]
            foo5 = np.where(idx == 5)[0]
            foo = np.array([foo3,foo4,foo5])
            x1a = np.concatenate([x1a,[x1[foo].mean()]])
            y1a = np.concatenate([y1a,[y1[foo].mean()]])
            x2a = np.concatenate([x2a,[x2[foo].mean()]])
            y2a = np.concatenate([y2a,[y2[foo].mean()]])
            xdiff1a = np.concatenate([xdiff1a,[xdiff1[foo].mean()]])
            ydiff1a = np.concatenate([ydiff1a,[ydiff1[foo].mean()]])
            xdiff2a = np.concatenate([xdiff2a,[xdiff2[foo].mean()]])
            ydiff2a = np.concatenate([ydiff2a,[ydiff2[foo].mean()]])
            xcentErrTmp1 = xpix1[foo].std(ddof=1)
            ycentErrTmp1 = ypix1[foo].std(ddof=1)
            xcentErr1 = np.concatenate([xcentErr1,[xcentErrTmp1]])
            ycentErr1 = np.concatenate([ycentErr1,[ycentErrTmp1]])
            xcentErrTmp2 = xpix2[foo].std(ddof=1)
            ycentErrTmp2 = ypix2[foo].std(ddof=1)
            xcentErr2 = np.concatenate([xcentErr2,[xcentErrTmp2]])
            ycentErr2 = np.concatenate([ycentErr2,[ycentErrTmp2]])
            #centErr = np.concatenate([centErr,[0.095]])
        if (6 in idx and 7 in idx and 8 in idx):
            numDthrs[j1] += 1
            dthrPos = np.concatenate([dthrPos, [2]])
            foo6 = np.where(idx == 6)[0]
            foo7 = np.where(idx == 7)[0]
            foo8 = np.where(idx == 8)[0]
            foo = np.array([foo6,foo7,foo8])
            x1a = np.concatenate([x1a,[x1[foo].mean()]])
            y1a = np.concatenate([y1a,[y1[foo].mean()]])
            x2a = np.concatenate([x2a,[x2[foo].mean()]])
            y2a = np.concatenate([y2a,[y2[foo].mean()]])
            xdiff1a = np.concatenate([xdiff1a,[xdiff1[foo].mean()]])
            ydiff1a = np.concatenate([ydiff1a,[ydiff1[foo].mean()]])
            xdiff2a = np.concatenate([xdiff2a,[xdiff2[foo].mean()]])
            ydiff2a = np.concatenate([ydiff2a,[ydiff2[foo].mean()]])
            xcentErrTmp1 = xpix1[foo].std(ddof=1)
            ycentErrTmp1 = ypix1[foo].std(ddof=1)
            xcentErr1 = np.concatenate([xcentErr1,[xcentErrTmp1]])
            ycentErr1 = np.concatenate([ycentErr1,[ycentErrTmp1]])
            xcentErrTmp2 = xpix2[foo].std(ddof=1)
            ycentErrTmp2 = ypix2[foo].std(ddof=1)
            xcentErr2 = np.concatenate([xcentErr2,[xcentErrTmp2]])
            ycentErr2 = np.concatenate([ycentErr2,[ycentErrTmp2]])
            #centErr = np.concatenate([centErr,[0.052]])
        if (9 in idx and 10 in idx and 11 in idx):
            numDthrs[j1] += 1
            dthrPos = np.concatenate([dthrPos, [3]])
            foo9 = np.where(idx == 9)[0]
            foo10 = np.where(idx == 10)[0]
            foo11 = np.where(idx == 11)[0]
            foo = np.array([foo9,foo10,foo11])
            x1a = np.concatenate([x1a,[x1[foo].mean()]])
            y1a = np.concatenate([y1a,[y1[foo].mean()]])
            x2a = np.concatenate([x2a,[x2[foo].mean()]])
            y2a = np.concatenate([y2a,[y2[foo].mean()]])
            xdiff1a = np.concatenate([xdiff1a,[xdiff1[foo].mean()]])
            ydiff1a = np.concatenate([ydiff1a,[ydiff1[foo].mean()]])
            xdiff2a = np.concatenate([xdiff2a,[xdiff2[foo].mean()]])
            ydiff2a = np.concatenate([ydiff2a,[ydiff2[foo].mean()]])
            xcentErrTmp1 = xpix1[foo].std(ddof=1)
            ycentErrTmp1 = ypix1[foo].std(ddof=1)
            xcentErr1 = np.concatenate([xcentErr1,[xcentErrTmp1]])
            ycentErr1 = np.concatenate([ycentErr1,[ycentErrTmp1]])
            xcentErrTmp2 = xpix2[foo].std(ddof=1)
            ycentErrTmp2 = ypix2[foo].std(ddof=1)
            xcentErr2 = np.concatenate([xcentErr2,[xcentErrTmp2]])
            ycentErr2 = np.concatenate([ycentErr2,[ycentErrTmp2]])
            #centErr = np.concatenate([centErr,[0.045]])

        # Save 16SWE's centroiding error
        if j1 == 0:
            xcentErr16swe1 = xcentErr1
            ycentErr16swe1 = ycentErr1
            xcentErr16swe2 = xcentErr2
            ycentErr16swe2 = ycentErr2
        xcentErr16sweP1 = np.array([xcentErr16swe1[ii] for ii in dthrPos])
        ycentErr16sweP1 = np.array([ycentErr16swe1[ii] for ii in dthrPos])
        xcentErr16sweP2 = np.array([xcentErr16swe2[ii] for ii in dthrPos])
        ycentErr16sweP2 = np.array([ycentErr16swe2[ii] for ii in dthrPos])

        # Require that the star be in at least 2 of the dither positions
        if (len(xdiff1a) < 2):
            stdX1[j1] = -1
            continue

        avgX1[j1] = x1a.mean()
        avgY1[j1] = y1a.mean()
        stdX1[j1] = np.sqrt(((x1a - avgX1[j1])**2).sum() / (len(x1a) - 1.))
        stdY1[j1] = np.sqrt(((y1a - avgY1[j1])**2).sum() / (len(y1a) - 1.))

        avgX2[j1] = x2a.mean()
        avgY2[j1] = y2a.mean()
        stdX2[j1] = np.sqrt(((x2a - avgX2[j1])**2).sum() / (len(x2a) - 1.))
        stdY2[j1] = np.sqrt(((y2a - avgY2[j1])**2).sum() / (len(y2a) - 1.))

        # xdiff1a contains the average delta at each pointing (dither)
        # avgX1p is the average over all pointings (dithers)
        avgX1p[j1] = xdiff1a.mean()
        avgY1p[j1] = ydiff1a.mean()
        stdX1p[j1] = np.sqrt(((xdiff1a - avgX1p[j1])**2).sum() / (2.*(numDthrs[j1] - 1.)) - \
                             ((xcentErr1)**2 + (xcentErr16sweP1)**2).sum() / (2.*2.*numDthrs[j1]))

        stdY1p[j1] = np.sqrt(((ydiff1a - avgY1p[j1])**2).sum() / (2.*(numDthrs[j1] - 1.)) - \
                              ((ycentErr1)**2 + (ycentErr16sweP1)**2).sum() / (2.*2.*numDthrs[j1]))

        avgX2p[j1] = xdiff2a.mean()
        avgY2p[j1] = ydiff2a.mean()
        stdX2p[j1] = np.sqrt(((xdiff2a - avgX2p[j1])**2).sum() / (2.*(numDthrs[j1] - 1.)) - \
                              ((xcentErr2)**2 + (xcentErr16sweP2)**2).sum() / (2.*2.*numDthrs[j1]))
                              #((xcentErr2/3.)**2 + (xcentErr16sweP2/3.)**2).sum() / 2.)
        stdY2p[j1] = np.sqrt(((ydiff2a - avgY2p[j1])**2).sum() / (2.*(numDthrs[j1] - 1.)) - \
                              ((ycentErr2)**2 + (ycentErr16sweP2)**2).sum() / (2.*2.*numDthrs[j1]))
                              #((ycentErr2/3.)**2 + (ycentErr16sweP2/3.)**2).sum() / 2.)

        # If the centroiding error is larger than the measured RMS delta, then these
        # stars are probably no good anyway. So throw these out (there are 1 or 2 stars
        # that fall into this category). This will automatically remove the first star,
        # which is itself 16SWE, which naturally has a centroiding error larger than the
        # delta from itself!
        if (np.isnan(stdX1p[j1]) | np.isnan(stdY1p[j1]) | np.isnan(stdX1p[j1]) | np.isnan(stdY1p[j1])):
            stdX1[j1] = -1
            continue
        if (np.isnan(stdX2p[j1]) | np.isnan(stdY2p[j1]) | np.isnan(stdX2p[j1]) | np.isnan(stdY2p[j1])):
            stdX1[j1] = -1
            continue

    idx = (np.where(stdX1 != -1))[0]
    names1 = [names1[ii] for ii in idx]
    avgX1 = avgX1[idx]
    avgY1 = avgY1[idx]
    stdX1 = stdX1[idx]
    stdY1 = stdY1[idx]

    avgX2 = avgX2[idx]
    avgY2 = avgY2[idx]
    stdX2 = stdX2[idx]
    stdY2 = stdY2[idx]

    numVals = numVals[idx]
    numDthrs = numDthrs[idx]

    avgX1p = avgX1p[idx] * 9.95
    avgY1p = avgY1p[idx] * 9.95
    stdX1p = stdX1p[idx] * 9.95 
    stdY1p = stdY1p[idx] * 9.95

    avgX2p = avgX2p[idx] * 9.95
    avgY2p = avgY2p[idx] * 9.95
    stdX2p = stdX2p[idx] * 9.95 
    stdY2p = stdY2p[idx] * 9.95

    avgR2p = np.hypot(avgX2p, avgY2p)

    py.figure(1)
    py.clf()
    py.plot(stdX1p, stdX2p, 'k.')
    py.plot([0, 10], [0, 10], 'k--')
    py.axis([0,10,0,10])
    py.title('X RMS of Distance from %s' % refSrc)
    py.xlabel('Pos. Error with Old (mas)')
    py.ylabel('Pos. Error with New (mas)')
    py.savefig('compDistort_X%s.eps' % outsuffix)
    py.savefig('compDistort_X%s.png' % outsuffix)

    py.figure(2)
    py.clf()
    py.plot(stdY1p, stdY2p, 'k.')
    py.plot([0, 10], [0, 10], 'k--')
    py.axis([0,10,0,10])
    py.title('Y RMS of Distance from %s' % refSrc)
    py.xlabel('Pos. Error with Old (mas)')
    py.ylabel('Pos. Error with New (mas)')
    py.savefig('compDistort_Y%s.eps' % outsuffix)
    py.savefig('compDistort_Y%s.png' % outsuffix)

    py.figure(3)
    py.clf()
    py.figure(figsize=(6,6))
    py.plot(stdX1p, stdX2p, 'rx',label='X',ms=7,mew=1.5)
    py.plot(stdY1p, stdY2p, 'b+',label='Y',ms=7,mew=1.5)
    py.plot([0, 8], [0, 8], 'k--')
    py.axis([0,8,0,8])
    #py.title('RMS of Distance from %s' % refSrc)
    py.xlabel('Positional Error with Old Solution (mas)', fontsize=16)
    py.ylabel('Positional Error with New Solution (mas)', fontsize=16)
    py.legend(numpoints=1)
    py.savefig('compDistort_XY%s.eps' % outsuffix)
    py.savefig('compDistort_XY%s.png' % outsuffix)

    # Also plot the pos error as a function of separation from ref star
    py.figure(4)
    py.clf()
    py.plot(avgR2p/1e3, stdX2p, 'r.',label='X')
    py.plot(avgR2p/1e3, stdY2p, 'b.',label='Y')
    py.axis([0, 20, 0, 10])
    py.title('New Solution - RMS of Distance from %s Vs. Distance' % refSrc)
    py.xlabel('Mean Separation from %s (arcsec)' % refSrc)
    py.ylabel('Pos. Error with New (mas)')
    py.legend()
    py.savefig('posErrVsSep%s.eps' % outsuffix)

    #for ii in range(len(stdX1p)):
    #    print '%15s  %4.2f   %4.2f' % (names1[ii], stdX1p[ii], stdX2p[ii])

    print 'Median X,Y pos error for %3i stars - Old: %4.2f, %4.2f mas' % \
          (len(stdX1p), np.median(stdX1p), np.median(stdY1p))
    print 'Median X,Y pos error for %3i stars - New: %4.2f, %4.2f mas' % \
          (len(stdX2p), np.median(stdX2p), np.median(stdY2p))

def stability(root, refSrc='S1-5', plotSrc='S1-17',
              imgRoot='/u/ghezgroup/data/gc/06maylgs1/clean/kp/',
              ylim=1.5):
    # Read in the align data
    s = starset.StarSet(root)

    # Also need coordinates relative to a bright star near
    # the center of the FOV.
    names = s.getArray('name')
    rid = names.index(refSrc)

    starCnt = len(s.stars)
    epochCnt = len(s.stars[0].e)

    # Read in the list of images used in the alignment
    listFile = open(root+'.list', 'r')
    files = []
    for line in listFile:
        _data = line.split()
        files.append(_data[0])

    # Get the Strehl, FWHM, and Wave front error for each image
    print "Reading Strehl File"
    strehlFile = open(imgRoot + 'irs33N.strehl', 'r')
    _frameno = []
    _strehl = []
    _wfe = []
    _fwhm = []
    for line in strehlFile:
        if (line.startswith("#")):
            continue

        _data = line.split()
        _frameno.append(_data[0])
        _strehl.append(float(_data[1]))
        _wfe.append(float(_data[2]))
        _fwhm.append(float(_data[3]))

    _strehl = np.array(_strehl)
    _wfe = np.array(_wfe)
    _fwhm = np.array(_fwhm)

    strehl = np.zeros(epochCnt, dtype=float)
    wfe = np.zeros(epochCnt, dtype=float)
    fwhm = np.zeros(epochCnt, dtype=float)
    elevation = np.zeros(epochCnt, dtype=float)
    darCoeff1 = np.zeros(epochCnt, dtype=float)
    darCoeff2 = np.zeros(epochCnt, dtype=float)
    airmass = np.zeros(epochCnt, dtype=float)
    parang = np.zeros(epochCnt, dtype=float)
    horizonX = np.zeros(epochCnt, dtype=float)
    horizonY = np.zeros(epochCnt, dtype=float)
    zenithX = np.zeros(epochCnt, dtype=float)
    zenithY = np.zeros(epochCnt, dtype=float)

    # Get the differential atmospheric refraction coefficients
    # assume all stars effective wavelengths are at 2.1 microns
    (refA, refB) = keckDARcoeffs(2.1)

    for ff in range(epochCnt):
        # Find the first instance of "/c"
        temp1 = files[ff].split('/')
        temp2 = temp1[-1].split('_')

        if (temp2[0].startswith('mag')):
            # This is a combo image. We don't have info about this file.
            # We should do this one last and adopt the average of all values.
            strehl[ff] = None
            wfe[ff] = None
            fwhm[ff] = None

            files[ff] = imgRoot + '../../combo/' + \
                        '_'.join(temp2[0:-1]) + '.fits'
        else:
            # Find this file
            idx = _frameno.index(temp2[0] + '.fits')
            strehl[ff] = _strehl[idx]
            wfe[ff] = _wfe[idx]
            fwhm[ff] = _fwhm[idx]
            
            files[ff] = imgRoot + temp2[0] + '.fits'

        # Get header info
        hdr = pyfits.getheader(files[ff])

        effWave = hdr['EFFWAVE']
        elevation[ff] = hdr['EL']
        lamda = hdr['CENWAVE']
        airmass[ff] = hdr['AIRMASS']
        parang[ff] = hdr['PARANG']
        
        date = hdr['DATE-OBS'].split('-')
        year = int(date[0])
        month = int(date[1])
        day = int(date[2])
        
        utc = hdr['UTC'].split(':')
        hour = int(utc[0])
        minute = int(utc[1])
        second = int(math.floor(float(utc[2])))

        utc = datetime.datetime(year, month, day, hour, minute, second)
        utc2hst = datetime.timedelta(hours=-10)
        hst = utc + utc2hst

        (refA, refB) = dar.keckDARcoeffs(effWave, hst.year, hst.month, hst.day,
                                         hst.hour, hst.minute)

        tanz = math.tan(math.radians(90.0 - elevation[ff]))
        tmp = 1 + tanz**2
        darCoeff1[ff] = tmp * (refA + 3.0 * refB * tanz**2)
        darCoeff2[ff] = -tmp * (refA*tanz +
                                3.0 * refB * (tanz + 2.0*tanz**3))

        # Lets determine the zenith and horizon unit vectors for
        # this image
        pa = math.radians(parang[ff] + float(hdr['ROTPOSN']) -
                          float(hdr['INSTANGL']))
        zenithX[ff] = -math.sin(pa)
        zenithY[ff] = math.cos(pa)
        horizonX[ff] = math.cos(pa)
        horizonY[ff] = math.sin(pa)

    posx = np.zeros((starCnt, epochCnt), float)
    posy = np.zeros((starCnt, epochCnt), float)
    diffx = np.zeros((starCnt, epochCnt), float)
    diffy = np.zeros((starCnt, epochCnt), float)
    diffr = np.zeros((starCnt, epochCnt), float)
    difft = np.zeros((starCnt, epochCnt), float)
    diffDR = np.zeros((starCnt, epochCnt), float)
    hasData = np.zeros((starCnt, epochCnt))

    # Positions and Errors to some reference source.
    x = np.zeros(starCnt, float)
    y = np.zeros(starCnt, float)
    rmsX = np.zeros(starCnt, float)
    rmsY = np.zeros(starCnt, float)
    rmsR = np.zeros(starCnt, float)
    rmsT = np.zeros(starCnt, float)
    rmsRarc = np.zeros(starCnt, float)
    rmsTarc = np.zeros(starCnt, float)
    r = np.zeros(starCnt, float)

    # Get the reference coordinates at each epoch
    refStar = s.stars[rid]
    refCooX = np.zeros(epochCnt, float)
    refCooY = np.zeros(epochCnt, float)
    for ee in range(epochCnt):
        refCooX[ee] = refStar.e[ee].xorig
        refCooY[ee] = refStar.e[ee].yorig

    # Now for every epoch, compute the offset from the mean
    py.clf()
    for ss in range(starCnt):
        star = s.stars[ss]

        # Original pixel positions
        xpix = star.getArrayAllEpochs('xorig')
        ypix = star.getArrayAllEpochs('yorig')

        #xarc = (xpix - refCooX) * s.t.scale
        #yarc = (ypix - refCooY) * s.t.scale
        xarc = (xpix - refCooX) * 0.00996
        yarc = (ypix - refCooY) * 0.00996

        # Filter out epochs with nodata
        idx = (np.where((xpix > -999) & (refCooX > -999)))[0]
        xpos = xarc[idx]
        ypos = yarc[idx]

        avgX = np.median(xpos)   #.mean()
        avgY = np.median(ypos)   #.mean()

        x[ss] = avgX
        y[ss] = avgY
        r[ss] = np.sqrt(avgX**2 + avgY**2)
        rmsX[ss] = xpos.std() * 1000.0
        rmsY[ss] = ypos.std() * 1000.0

        xdiff = xpos - avgX
        ydiff = ypos - avgY

        diffxtmp = (xarc - avgX) * 1000.0
        diffytmp = (yarc - avgY) * 1000.0

        avgR = np.sqrt(avgX**2 + avgY**2)
        
        # Convert into zenith (Y) and horizon(X) coordinates
        diffx[ss] = (diffxtmp * horizonX) + (diffytmp * horizonY)
        diffy[ss] = (diffxtmp * zenithX) + (diffytmp * zenithY)
        diffr[ss] = ((diffxtmp * avgX) + (diffytmp * avgY)) / avgR
        difft[ss] = ((diffxtmp * -avgY) + (diffytmp * avgX)) / avgR

        posx[ss] = xarc
        posy[ss] = yarc
        hasData[ss,idx] += 1
        hasData[ss, 0] = 0   # Turn off the first epoch

        # Compute the predicted differential atmospheric refraction
        # between this star and the reference star in each of the
        # observed images.
        deltaZ = ((xarc * zenithX) + (yarc * zenithY)) / 206265.0
        deltaR = darCoeff1 * deltaZ + darCoeff2 * deltaZ**2
        deltaR *= 206265.0

        avgDR = deltaR.mean()
        diffDR[ss] = -(deltaR - avgDR) * 1000.0


    pntsize = 6
    mag = s.getArray('mag')

    idx = (np.where((rmsX < 0.8) & (rmsY < 0.8) & (mag < 14)))[0]
    #print [names[ii] for ii in idx]
    for ii in idx:
        edx = (np.where(hasData[ii] == 1))[0]
 
        errx = diffx[ii,edx].std()
        erry = diffy[ii,edx].std()
        print 'pos=(%5.2f, %5.2f) asec   perr=(%5.2f, %5.2f) mas for %s (%4.1f)' % \
              (x[ii], y[ii], errx, erry, names[ii], mag[ii])


    #sdx = (np.where((mag > 10) & (mag < 14) & (r > 0) & (r < 0.55)))[0]
    #colors = cm.flag(linspace(0, 1, len(sdx)))
    #sdx = [names.index('S1-21')]
    #sdx = [names.index('S0-13')]
    #sdx = [names.index('S1-17')]
    sdx = [names.index(plotSrc)]
    colors = ['k']

    frameno = np.arange(0, epochCnt)

    py.clf()
    for ss in range(len(sdx)):
        source = names[sdx[ss]]
        idx = names.index(source)
        edx = (np.where(hasData[idx] == 1))[0]

        errx = diffx[idx,edx].std()
        erry = diffy[idx,edx].std()

        #if ((errx > 0.5) or (erry > 0.5) or (errx == 0) or (erry == 0)):
        #    continue

        print 'pos = (%5.2f, %5.2f) asec   perr = (%5.2f, %5.2f) mas for %s' % \
              (x[idx], y[idx], errx, erry, source)
        print 'Parallactic Angle range: %5.1f - %5.1f' % \
              (parang.min(), parang.max())
        print 'Airmass range:           %5.2f - %5.2f' % \
              (airmass.min(), airmass.max())
        print 'Elevation range:         %5.2f - %5.2f' % \
              (elevation.min(), elevation.max())

        c = colors[ss]

        def plotStuff(xdat, xlab, suffix):
            py.clf()
            py.subplots_adjust(hspace=0.12, top=0.93)

            #####
            # Plot positions
            #####
            py.subplot(2, 1, 1)
            py.plot(xdat, posx[idx, edx], 'k.', mfc=c, mec=c)
            rng = py.axis()
            py.plot([rng[0], rng[1]], [x[idx], x[idx]], 'k--')
            py.ylabel('Sep. (arcsec)')
            legX = rng[0] + (rng[1] - rng[0]) * 0.98
            legY = rng[2] + ((rng[3] - rng[2]) * 0.9)
            py.text(legX, legY, 'X', fontweight='bold',
                    horizontalalignment='right', verticalalignment='top')
            py.title('%s vs. %s dX = %4.1f  dY=%4.1f' % \
                  (plotSrc, refSrc, x[idx], y[idx]))

            py.subplot(2, 1, 2)
            py.plot(xdat, posy[idx, edx], 'k.', mfc=c, mec=c)
            rng = py.axis()
            py.plot([rng[0], rng[1]], [y[idx], y[idx]], 'k--')
            py.ylabel('Sep. (arcsec)')
            legX = rng[0] + (rng[1] - rng[0]) * 0.98
            legY = rng[2] + ((rng[3] - rng[2]) * 0.9)
            py.text(legX, legY, 'Y', fontweight='bold',
                    horizontalalignment='right', verticalalignment='top')
            
            py.savefig('stability_xy_%s_%s.png' % (suffix, source))
            py.savefig('stability_xy_%s_%s.eps' % (suffix, source))

            #####
            # Horizon & Zenith
            #####
            py.clf()
            py.subplot(2, 1, 1)
            py.plot(xdat, diffx[idx, edx], 'k.', mfc=c, mec=c)
            rng = py.axis()
            py.plot([rng[0], rng[1]], [0, 0], 'k--')
            py.axis([rng[0], rng[1], -ylim, ylim])
            py.ylabel('Delta-Sep. (mas)')
            legX = rng[0] + (rng[1] - rng[0]) * 0.98
            legY = 1.4
            py.text(legX, legY, 'Horizon', fontweight='bold',
                    horizontalalignment='right', verticalalignment='top')
            py.title('%s vs. %s dX = %4.1f  dY=%4.1f' % \
                  (plotSrc, refSrc, x[idx], y[idx]))

            py.subplot(2, 1, 2)
            py.plot(xdat, diffy[idx, edx], 'k.', mfc=c, mec=c)
            if (suffix == 'parang'):
                py.plot(xdat, diffDR[idx, edx], 'r-')
            else:
                py.plot([rng[0], rng[1]], [0, 0], 'k--')
            py.axis([rng[0], rng[1], -ylim, ylim])
            py.xlabel(xlab)
            py.ylabel('Delta-Sep. (mas)')
            py.text(legX, legY, 'Zenith', fontweight='bold',
                    horizontalalignment='right', verticalalignment='top')
            
            py.savefig('stability_zh_%s_%s.png' % (suffix, source))
            py.savefig('stability_zh_%s_%s.eps' % (suffix, source))

            #####
            # Radial & Tangential
            #####
            py.clf()
            py.subplot(2, 1, 1)
            py.plot(xdat, diffr[idx, edx], 'k.', mfc=c, mec=c)
            rng = py.axis()
            py.plot([rng[0], rng[1]], [0, 0], 'k--')
            py.axis([rng[0], rng[1], -ylim, ylim])
            py.ylabel('Delta-Sep. (mas)')
            legX = rng[0] + (rng[1] - rng[0]) * 0.98
            legY = 1.4
            py.text(legX, legY, 'Radial', fontweight='bold',
                    horizontalalignment='right', verticalalignment='top')
            py.title('%s vs. %s dX = %4.1f  dY=%4.1f' % \
                  (plotSrc, refSrc, x[idx], y[idx]))
            
            py.subplot(2, 1, 2)
            py.plot(xdat, difft[idx, edx], 'k.', mfc=c, mec=c)
            py.plot([rng[0], rng[1]], [0, 0], 'k--')
            py.axis([rng[0], rng[1], -ylim, ylim])
            py.xlabel(xlab)
            py.ylabel('Delta-Sep. (mas)')
            legX = rng[0] + (rng[1] - rng[0]) * 0.98
            legY = 1.4
            py.text(legX, legY, 'Tangential', fontweight='bold',
                    horizontalalignment='right', verticalalignment='top')
            
            py.savefig('stability_rt_%s_%s.png' % (suffix, source))
            py.savefig('stability_rt_%s_%s.eps' % (suffix, source))



        # Make some additional plots
        py.clf()
        py.plot(parang, fwhm, 'k.')
        py.xlabel('Parallactic Angle (deg)')
        py.ylabel('FWHM (mas)')
        py.savefig('stability_fwhm_vs_parang.png')
        py.savefig('stability_fwhm_vs_parang.eps')

        py.clf()
        py.plot(parang, strehl, 'k.')
        py.xlabel('Parallactic Angle (deg)')
        py.ylabel('Strehl')
        py.savefig('stability_strehl_vs_parang.png')
        py.savefig('stability_strehl_vs_parang.eps')

        py.clf()
        py.plot(parang, airmass, 'k.')
        py.xlabel('Parallactic Angle (deg)')
        py.ylabel('Airmass')
        py.savefig('stability_airmass_vs_parang.png')
        py.savefig('stability_airmass_vs_parang.eps')

        # Plot Strehl vs. Pos Diff
        plotStuff(strehl[edx], 'Strehl', 'strehl')

        # Plot FWHM vs. Pos Diff
        plotStuff(fwhm[edx], 'FWHM (mas)', 'fwhm')

        # Plot airmass vs. Pos Diff
        plotStuff(airmass[edx], 'Airmass', 'airmass')

        # Plot ParAng vs. Pos Diff
        plotStuff(parang[edx], 'Parallactic Angle (deg)', 'parang')


def plotTransVsTime(alignRoot, suffix='', plotErrors=False):
    """
    Plot transformation parameters vs. time in the scale/angle formats.

    Input:
    alignRoot - Root file name (including directory) of the align output.

    Optional Input:
    suffix - An optional suffix for the output plot files.
    plotErrors - Plot error bars. This requires the align bootstrap to have
        been run (default=False).

    Output:
    trans_stability<suffix>.png
    trans_stability<suffix>.eps
    """
    # Analysis of global plate scale

    # Pull out dates
    dateFile = open(alignRoot + '.date', 'r')
    datesStr = dateFile.readline().split()
    dates = np.array([float(dd) for dd in datesStr])
    numEpochs = len(dates)

    # Load scales/angles
    scale = np.zeros(numEpochs, float)
    angle = np.zeros(numEpochs, float)
    sgrax = np.zeros(numEpochs, float)
    sgray = np.zeros(numEpochs, float)
    imgPA = np.zeros(numEpochs, float)
    isAO = np.zeros(numEpochs, int)
    scaleErr = np.zeros(numEpochs, float)
    angleErr = np.zeros(numEpochs, float)
    sgraxErr = np.zeros(numEpochs, float)
    sgrayErr = np.zeros(numEpochs, float)

    # Load up the *.list file in order to get data types
    lisFile = open(alignRoot + '.list', 'r')
    ref = 0

    epochNames = []
    for e in range(numEpochs):
        fields = lisFile.readline().split()

        epochNames.append( (fields[0].split('/'))[-1])
        
        if (int(fields[1]) > 3):
            isAO[e] = 1

        if (len(fields) > 2 and fields[2] == 'ref'):
            ref = e

        trans = objects.Transform()
        trans.loadFromAbsolute(root='./', align=alignRoot + '.trans',
                               idx=e)
            
        trans.linearToSpherical(silent=1, override=False)
        scale[e] = trans.scale
        angle[e] = math.degrees(trans.angle)
        sgrax[e] = trans.sgra[0]
        sgray[e] = trans.sgra[1]

        scaleErr[e] = trans.scaleErr
        angleErr[e] = math.degrees(trans.angleErr)
        sgraxErr[e] = trans.sgraErr[0]
        sgrayErr[e] = trans.sgraErr[1]

        # All PAs are zero except for a few:
        if '04jullgs' in epochNames[e]:
            imgPA[e] = 200.0
        if '05jullgs' in epochNames[e]:
            imgPA[e] = 190.0
        if '07maylgs_tran4' in epochNames[e]:
            imgPA[e] = 200.0

    paDiff = imgPA - imgPA[ref] + angle
    idx = (np.where(paDiff < -180))[0]
    paDiff[idx] += 360.0
    idx = (np.where(paDiff > 180))[0]
    paDiff[idx] -= 360.0

    aoidx = (np.where(isAO == 1))[0]
    spidx = (np.where(isAO == 0))[0]

    if (len(spidx) > 0):
        minYear = 1994
    else:
        minYear = 2004

    maxYear = 2009 + 1

    py.clf()
    py.subplots_adjust(left=0.15, right=0.93)
    py.subplot(2, 1, 1)
    py.plot([minYear, maxYear], [1, 1], 'k--')
    if (len(spidx) > 0):
        py.plot([minYear, maxYear], [scale[spidx[-3]], scale[spidx[-3]]], 'k--')
        py.plot(dates[spidx], scale[spidx], 'ro', ms=4, mec='r')
        if plotErrors:
            py.errorbar(dates[spidx], scale[spidx], yerr=scaleErr[spidx],
                        fmt='ro', color='r')

    if (len(aoidx) > 0):
        py.plot(dates[aoidx], scale[aoidx], 'bo', ms=4, mec='b')
        if plotErrors:
            py.errorbar(dates[aoidx], scale[aoidx], yerr=scaleErr[aoidx],
                        fmt='bo', color='b')
        
    py.ylabel('Relative Plate Scale')
    py.title('Reference Epoch: ' + epochNames[ref])
    py.xlim(minYear, maxYear)

    if (len(spidx) > 0):
        py.ylim(0.996, 1.033)
    else:
        py.ylim(0.996, 1.004)

    py.gca().get_xaxis().set_major_formatter(py.FormatStrFormatter('%d'))

    py.subplot(2, 1, 2)
    py.plot([minYear, maxYear], [0, 0], 'k--')

    legItems = []
    legLabel = []

    if (len(spidx) > 0):
        n1 = py.plot(dates[spidx], paDiff[spidx], 'ro', ms=4, mec='r')
        if plotErrors:
            py.errorbar(dates[spidx], angle[spidx], yerr=angleErr[spidx],
                        fmt='ro', color='r')
        legItems.append(n1)
        legLabel.append('NIRC')

    if (len(aoidx) > 0):
        n2 = py.plot(dates[aoidx], paDiff[aoidx], 'bo', ms=4, mec='b')
        if plotErrors:
            py.errorbar(dates[aoidx], angle[aoidx], yerr=angleErr[aoidx],
                        fmt='bo', color='b')
        legItems.append(n2)
        legLabel.append('NIRC2')

    py.xlabel('Date of Observation (years)')
    py.ylabel('Absolute PA Offset (deg)')
    py.xlim(minYear, maxYear)

    if (len(spidx) > 0):
        py.ylim(-1.2, 1.2)
    else:
        py.ylim(-0.3, 0.3)

    py.gca().get_xaxis().set_major_formatter(py.FormatStrFormatter('%d'))

    py.legend(legItems, legLabel, numpoints=1)
    py.savefig('plots/trans_stability'+suffix+'.png')
    py.savefig('plots/trans_stability'+suffix+'.eps')

def plotTransVsTime2(alignRoot, suffix='', plotErrors=False):
    """
    Plot transformation parameters vs. time in their raw
    6 (or more) parameter format. Use plotTransVsTime() to plot
    in plate scale and angle format. This code can handle a variable
    number of transformation parameters, so it should work for
    all 1st and 2nd order transformations out of align.

    Input:
    alignRoot - Root file name (including directory) of the align output.

    Optional Input:
    suffix - An optional suffix for the output plot files.
    plotErrors - Plot error bars. This requires the align bootstrap to have
        been run (default=False).

    Output:
    trans_stability_ab#<suffix>.png
    trans_stability_ab#<suffix>.eps

    where # is the number of each transformation parameter. 
    """
    # Analysis of global plate scale

    # Pull out dates
    dateFile = open(alignRoot + '.date', 'r')
    datesStr = dateFile.readline().split()
    dates = np.array([float(dd) for dd in datesStr])
    numEpochs = len(dates)

    # Load up the *.list file in order to get data types
    lisFile = open(alignRoot + '.list', 'r')
    ref = 0

    # Load up the *.trans file
    transTab = asciidata.open(alignRoot + '.trans')

    # Ignore the first 3 columns, except the number of
    # parameters is useful.
    numParams = int(transTab[2][0])

    # Create variables for all the transformation parameters.
    # These will be 2D arrays with the first index giving the
    # parameter (a[0] is a_0 in the linear equation).
    a = np.zeros((numParams/2, numEpochs), dtype=float)
    b = np.zeros((numParams/2, numEpochs), dtype=float)
    ae = np.zeros((numParams/2, numEpochs), dtype=float)
    be = np.zeros((numParams/2, numEpochs), dtype=float)
    isAO = np.zeros(numEpochs, int)

    epochNames = []
    for e in range(numEpochs):
        fields = lisFile.readline().split()

        epochNames.append( (fields[0].split('/'))[-1])
        
        if (int(fields[1]) > 3):
            isAO[e] = 1

        if (len(fields) > 2 and fields[2] == 'ref'):
            ref = e

        for nn in range(numParams/2):
            # First 3 columns are ignored.
            # There are two columns for every parameter (value, error).
            # All X params (a0, a1,...) come first, then Y params (b0, b1,...)
            a[nn][e] = transTab[3+(2*nn)][e]
            b[nn][e] = transTab[3+numParams+(2*nn)][e]
            ae[nn][e] = transTab[3+(2*nn)+1][e]
            be[nn][e] = transTab[3+numParams+(2*nn)+1][e]

    aoidx = (np.where(isAO == 1))[0]
    spidx = (np.where(isAO == 0))[0]

    if (len(spidx) > 0):
        minYear = 1994
    else:
        minYear = 2004

    maxYear = 2009 + 1

    # Make plots for each parameter
    for nn in range(numParams/2):
        legItems = []
        legLabel = []

        ##########
        # X Parameter
        ##########
        py.clf()
        py.subplots_adjust(left=0.15, right=0.93)
        py.subplot(2, 1, 1)
        py.plot([minYear, maxYear], [a[nn].mean(), a[nn].mean()], 'k--')

        # Speckle
        if (len(spidx) > 0):
            n1 = py.plot(dates[spidx], a[nn][spidx], 'ro', ms=4, mec='r')
            if plotErrors:
                py.errorbar(dates[spidx], a[nn][spidx], yerr=ae[nn][spidx],
                            fmt='ro', color='r')
            legItems.append(n1)
            legLabel.append('NIRC')
            
        # AO
        if (len(aoidx) > 0):
            n2 = py.plot(dates[aoidx], a[nn][aoidx], 'bo', ms=4, mec='b')
            if plotErrors:
                py.errorbar(dates[aoidx], a[nn][aoidx], yerr=ae[nn][aoidx],
                            fmt='bo', color='b')
            legItems.append(n2)
            legLabel.append('NIRC2')
        
        py.ylabel('a%d' % nn)
        py.title('Reference Epoch: ' + epochNames[ref])
        py.xlim(minYear, maxYear)
        py.gca().get_xaxis().set_major_formatter(py.FormatStrFormatter('%d'))
        py.legend(legItems, legLabel, numpoints=1)


        ##########
        # Y Parameter
        ##########
        py.subplot(2, 1, 2)
        py.plot([minYear, maxYear], [b[nn].mean(), b[nn].mean()], 'k--')

        # Speckle
        if (len(spidx) > 0):
            py.plot(dates[spidx], b[nn][spidx], 'ro', ms=4, mec='r')
            if plotErrors:
                py.errorbar(dates[spidx], b[nn][spidx], yerr=be[nn][spidx],
                            fmt='ro', color='r')
        # AO
        if (len(aoidx) > 0):
            py.plot(dates[aoidx], b[nn][aoidx], 'bo', ms=4, mec='b')
            if plotErrors:
                py.errorbar(dates[aoidx], b[nn][aoidx], yerr=be[nn][aoidx],
                            fmt='bo', color='b')

        py.xlabel('Date of Observation (years)')
        py.ylabel('b%d' % nn)
        py.xlim(minYear, maxYear)
        py.gca().get_xaxis().set_major_formatter(py.FormatStrFormatter('%d'))

        py.savefig('plots/trans_stability_ab' + str(nn) + suffix + '.png')
        #py.savefig('plots/trans_stability_ab' + nn + suffix + '.eps')


def gcDAR():
    iraf.noao()
    obs = iraf.noao.observatory

    # Set up Keck observatory info
    obs(command="set", obsid="keck")

    ####################
    # Setup all the parameters for the atmospheric refraction
    # calculations. Typical values obtained from the Mauna Kea
    # weather pages and from the web.
    ####################
    #
    # Height above sea level (meters)
    hm = obs.altitude
    # Ambient Temperature (Kelvin)
    tdk = 272.0
    # Pressure at the observer (millibar)
    pmb = 617.0
    # Relative humidity (%)
    rh = 0.1
    # Latitude of the observer (radian)
    phi = math.radians(obs.latitude)
    # Temperature Lapse Rate (Kelvin/meter)
    tlr = 0.0065
    # Precision required to terminate the iteration (radian)
    eps = 1.0e-9

    ## Star specific values ##
    # Effective wavelength  (microns)
    lamda1 = 2.1
    lamda2 = 2.1


    ####################
    #
    # Get the curves for a specific night
    #
    ####################
    keck = ephem.Observer()
    keck.long = math.radians(-obs.longitude)
    keck.lat = math.radians(obs.latitude)
    keck.elev = obs.altitude
    keck.pressure = pmb
    keck.temp = tdk
    
    # Date of Observations: 06maylgs1
    keck.date = '2006/05/03 %d:%d:%f' % (obs.timezone, 28, 46.33)

    # Set up the galactic center target
    sgra = ephem.FixedBody()
    sgra._ra = ephem.hours("17:47:40")
    sgra._dec = ephem.degrees("-29:00:28")
    sgra._epoch = 2000
    sgra.compute()

    # Another star that is seperated to the north
    sepInput = np.array([1.0, 5.0, 10.0, 15.0])      # arcsec
    
    sgra.compute(keck)
    transit = sgra.transit_time.tuple()
    print 'Sgr A* Transit Time: ', sgra.transit_time
    
    lines = []
    colors = ['r', 'g', 'b', 'k']

    util.usetexTrue()

    py.clf()
    for ss in range(len(sepInput)):
        # Working with Seperation:
        sep = sepInput[ss]
        
        # Now step through the night at intervals of 10 minutes and 
        # calculate the DAR. Calc every 10 minutes for 4 hours centered
        # on the transit time.
        timeRange = range(6*4 + 1)
        hour = transit[3] - 2
        minute = transit[4]
        
        hourAngle = np.arange(len(timeRange), dtype=float)
        airmass = np.arange(len(timeRange), dtype=float)
        refAngle = np.arange(len(timeRange), dtype=float)
        deltaR = np.arange(len(timeRange), dtype=float)
        deltaZtrue = np.arange(len(timeRange), dtype=float)
        deltaZobs = np.arange(len(timeRange), dtype=float)
        
        for ii in timeRange:
            if (ii != 0):
                minute += 10
                
            if (minute > 60):
                minute -= 60
                hour += 1
                
            keck.date = '%d/%d/%d %d:%d:%f' % \
                        (transit[0], transit[1], transit[2],
                         hour, minute, transit[5])

            timetmp1 = keck.date.triple()
            timetmp2 = sgra.transit_time.triple()
            
            hourAngle[ii] = (timetmp1[2] - timetmp2[2]) * 24.0
            
            sgra.compute(keck)
            
            z1 = (math.pi/2.0) - sgra.alt
            z2 = z1 - (sep / 206265.0)

            airmass[ii] = 1.0 / math.cos(z1)
            
            R1 = refro.slrfro(z1, hm, tdk, pmb, rh, lamda1, phi, tlr, eps)
            R2 = refro.slrfro(z2, hm, tdk, pmb, rh, lamda2, phi, tlr, eps)
            
            # All in arcseconds
            deltaR[ii] = (R1 - R2) * 206265.0
            deltaZobs[ii] = (z1 - z2) * 206265.0
            deltaZtrue[ii] = deltaZobs[ii] + deltaR[ii]
            refAngle[ii] = R2 * 206265.0
            
            print '%6.2f  %4.2f %6.3f  dZobs = %4.2f  dZtrue = %4.2f  dR = %4.2f' % \
                  (hourAngle[ii], airmass[ii], refAngle[ii],
                   deltaZobs[ii]*1e3, deltaZtrue[ii]*1e3, deltaR[ii]*1e3)
            
        foo = py.semilogy(hourAngle, deltaR*1e3, colors[ss]+'.')
        lines.append(foo)

    py.xlabel('Hour Angle')
    py.ylabel('Differential Refraction (mas) = Sep[true] - Sep[obs]')
    py.ylabel(r'$\Delta$R (mas) = $\Delta$z$_0 - \Delta$z')
    py.title('2006-5-3: Galactic Center')
    labels = ['Sep = %2d"' % sep for sep in sepInput]
    py.legend(lines, labels, loc='upper left', numpoints=1)
    py.axis([-3, 3, 0.2, 40])
    #py.axis([-3, 3, 0.1, 4])

    py.savefig('gcDAR_log.eps')
    py.savefig('gcDAR_log.png')
    
    util.usetexFalse()

def keckDAR(lamda, elevation, separation):
    """
    Calculate the differential atmospheric refraction
    for two objects observed at Keck.

    Input:
    lamda -- Effective wavelength (microns) assumed to be the same for both
    elevation -- Elevation angle (degrees) of the observations
    separation -- Seperation (arcsec) along zenith axis for two objects.

    Output:
    deltaR -- Amount of DAR (mas) for the two input objects.
    """
    iraf.noao()
    obs = iraf.noao.observatory

    # Set up Keck observatory info
    foo = obs(command="set", obsid="keck", Stdout=1)

    ####################
    # Setup all the parameters for the atmospheric refraction
    # calculations. Typical values obtained from the Mauna Kea
    # weather pages and from the web.
    ####################
    #
    # Height above sea level (meters)
    hm = obs.altitude
    # Ambient Temperature (Kelvin)
    tdk = 272.0
    # Pressure at the observer (millibar)
    pmb = 617.0
    # Relative humidity (%)
    rh = 0.1
    # Latitude of the observer (radian)
    phi = math.radians(obs.latitude)
    # Temperature Lapse Rate (Kelvin/meter)
    tlr = 0.0065
    # Precision required to terminate the iteration (radian)
    eps = 1.0e-9

    z1 = math.radians(90.0 - elevation)
    z2 = z1 - (separation / 206265.0)
            
    R1 = refro.slrfro(z1, hm, tdk, pmb, rh, lamda, phi, tlr, eps)
    R2 = refro.slrfro(z2, hm, tdk, pmb, rh, lamda, phi, tlr, eps)

    deltaR = (R1 - R2) * 206265.0 * 1e3

    return deltaR

def keckDARcoeffs(lamda):
    """
    Calculate the differential atmospheric refraction
    for two objects observed at Keck.

    Input:
    lamda -- Effective wavelength (microns) assumed to be the same for both
    elevation -- Elevation angle (degrees) of the observations
    separation -- Seperation (arcsec) along zenith axis for two objects.

    Output:
    deltaR -- Amount of DAR (mas) for the two input objects.
    """
    iraf.noao()
    obs = iraf.noao.observatory

    # Set up Keck observatory info
    foo = obs(command="set", obsid="keck", Stdout=1)

    ####################
    # Setup all the parameters for the atmospheric refraction
    # calculations. Typical values obtained from the Mauna Kea
    # weather pages and from the web.
    ####################
    #
    # Height above sea level (meters)
    hm = obs.altitude
    # Ambient Temperature (Kelvin)
    tdk = 272.0
    # Pressure at the observer (millibar)
    pmb = 617.0
    # Relative humidity (%)
    rh = 0.1
    # Latitude of the observer (radian)
    phi = math.radians(obs.latitude)
    # Temperature Lapse Rate (Kelvin/meter)
    tlr = 0.0065
    # Precision required to terminate the iteration (radian)
    eps = 1.0e-9

            
    return refco.slrfco(hm, tdk, pmb, rh, lamda, phi, tlr, eps)


def plotPosDiff(align, epoch1, epoch2, noAlignErr=True):
    """
    Compare positional differences from two different epochs.
    If the epochs are close in time, then the differences should
    be consistent with the measurement error in the individual epochs.

    Problem: For the LGSAO epochs, the high precision means that
    significant proper motion is actually detectable on month time scales.
    """
    s = starset.StarSet(align)

    # Get the set of stars that are in both epochs of interest
    xorig1 = s.getArrayFromEpoch(epoch1, 'xorig')
    xorig2 = s.getArrayFromEpoch(epoch2, 'xorig')

    idx = (np.where((xorig1 > -999) & (xorig2 > -999)))[0]

    # Now get all the star positions and errors
    x1 = s.getArrayFromEpoch(epoch1, 'x')
    y1 = s.getArrayFromEpoch(epoch1, 'y')
    x2 = s.getArrayFromEpoch(epoch2, 'x')
    y2 = s.getArrayFromEpoch(epoch2, 'y')
    xerr_p1 = s.getArrayFromEpoch(epoch1, 'xerr_p')
    yerr_p1 = s.getArrayFromEpoch(epoch1, 'yerr_p')
    xerr_a1 = s.getArrayFromEpoch(epoch1, 'xerr_a')
    yerr_a1 = s.getArrayFromEpoch(epoch1, 'yerr_a')
    xerr_p2 = s.getArrayFromEpoch(epoch2, 'xerr_p')
    yerr_p2 = s.getArrayFromEpoch(epoch2, 'yerr_p')
    xerr_a2 = s.getArrayFromEpoch(epoch2, 'xerr_a')
    yerr_a2 = s.getArrayFromEpoch(epoch2, 'yerr_a')
    mag = s.getArray('mag')
    name = s.getArray('name')
    
    x1 = x1[idx]
    y1 = y1[idx]
    x2 = x2[idx]
    y2 = y2[idx]
    xerr_p1 = xerr_p1[idx]
    yerr_p1 = yerr_p1[idx]
    xerr_a1 = xerr_a1[idx]
    yerr_a1 = yerr_a1[idx]
    xerr_p2 = xerr_p2[idx]
    yerr_p2 = yerr_p2[idx]
    xerr_a2 = xerr_a2[idx]
    yerr_a2 = yerr_a2[idx]
    mag = mag[idx]
    name = [name[ii] for ii in idx]

    # Combine sources of positional error
    if (noAlignErr == True):
        xerr1 = xerr_p1
        yerr1 = yerr_p1
        xerr2 = xerr_p2
        yerr2 = yerr_p2
    else:
        xerr1 = np.sqrt(xerr_p1**2 + xerr_a1**2)
        yerr1 = np.sqrt(yerr_p1**2 + yerr_a1**2)
        xerr2 = np.sqrt(xerr_p2**2 + xerr_a2**2)
        yerr2 = np.sqrt(yerr_p2**2 + yerr_a2**2)

    # Now lets determine the positional difference from one
    # epoch to the next
    xdiff = (x2 - x1) * 1e3  # mas
    ydiff = (y2 - y1) * 1e3  # mas

    diff = np.sqrt(xdiff**2 + ydiff**2)

    xdiffErr = np.sqrt(xerr1**2 + xerr2**2) * 1e3
    ydiffErr = np.sqrt(yerr1**2 + yerr2**2) * 1e3

    diffErr = np.sqrt( (xdiff*xdiffErr)**2 + (ydiff*ydiffErr)**2 ) / diff

    sigma = diff / diffErr

    # Average the error for each magnitude bin
    magStep = 1.0
    magBins = np.arange(10.0, 19.0, magStep)
    errMag = np.zeros(len(magBins), float)

    for mm in range(len(magBins)):
        mMin = magBins[mm] - (magStep / 2.0)
        mMax = magBins[mm] + (magStep / 2.0)
        idx = (np.where((mag >= mMin) & (mag < mMax)))[0]

        if (len(idx) > 0):
            errMag[mm] = np.median(diffErr[idx])


    ##########
    # Compute errors in radius bins
    ########## 
    # Convert into arcsec offset from field center (roughly - just
	# to print out positional differences in each radial bin)
    x1 = (x1 - .512)
    y1 = (y1 - .512)
    r1 = np.hypot(x1, y1)

    radStep = 1.0
    radBins = np.arange(0.5, 9.5, radStep)
    errRad = np.zeros(len(radBins), float)

    py.clf()
    print '\n%4s  %s  %s' % ('Rad', 'Err (mas)', 'N Stars')
    for rr in range(len(radBins)):
        rMin = radBins[rr] - (radStep / 2.0)
        rMax = radBins[rr] + (radStep / 2.0)
        ridx = (np.where((r1 >= rMin) & (r1 < rMax) & (mag < 15)))[0]

        if (len(ridx) > 0):
            errRad[rr] = np.median(diff[ridx])

        print '%4.2f  %5.2f  %4i' % (radBins[rr], errRad[rr], len(ridx))


    py.clf()
    py.semilogy(mag, diff, 'k.')
    py.semilogy(magBins, errMag, 'b-')
    
    py.axis([8, 20, 5e-2, 10.0])
    py.xlabel('K Magnitude')
    py.ylabel('Pos. Uncertainty (mas)')
    py.title('Epoch %d vs. %d' % (epoch1, epoch2))

    py.savefig('posDiff_%s_e%d_vs_e%d.png' % (align, epoch1, epoch2))
    py.savefig('posDiff_%s_e%d_vs_e%d.eps' % (align, epoch1, epoch2))


def plotScaleAngle(align):
    tab = asciidata.open(align + '.trans')

    scale = np.zeros(tab.nrows, float)
    angle = np.zeros(tab.nrows, float)
    scaleErr = np.zeros(tab.nrows, float)
    angleErr = np.zeros(tab.nrows, float)

    for rr in range(tab.nrows):
        t = objects.Transform()

	t.a = [tab[3][rr], tab[5][rr], tab[7][rr]]
	t.b = [tab[9][rr], tab[11][rr], tab[13][rr]]

	t.aerr = [tab[4][rr],  tab[6][rr],  tab[8][rr]]
	t.berr = [tab[10][rr], tab[12][rr], tab[14][rr]]

        t.linearToSpherical(override=False)

        scale[rr] = t.scale                   
        angle[rr] = t.angle * 180.0 / math.pi # degrees
        scaleErr[rr] = t.scaleErr
        angleErr[rr] = t.angleErr * 180.0 / math.pi

    scale -= 1.0
    scale *= 100.0
    scaleErr *= 100.0

    py.clf()
    frameno = range(tab.nrows)

    py.subplot(2, 1, 1)
    py.errorbar(frameno, scale, yerr=scaleErr, fmt='k.')
    rng = py.axis()
    py.axis([-0.5, 3.5, rng[2], rng[3]])
    py.xticks(frameno, ['06may1', '06may2', '06jun', '06jul'])
    py.ylabel('% Change in Platescale')

    py.subplot(2, 1, 2)
    py.errorbar(frameno, angle, yerr=angleErr, fmt='k.')
    rng = py.axis()
    py.axis([-0.5, 3.5, rng[2], rng[3]])
    py.xticks(frameno, ['06may1', '06may2', '06jun', '06jul'])
    py.ylabel('Position Angle (deg)')

    py.savefig('scaleAngle.png')


def movieClean():
    root = '/u/ghezgroup/data/gc/06maylgs1/clean/kp/'

    # June
    #files = np.arange(1123, 1284, 10)
    # May
    files = np.arange(1073, 1189, 10)

    for ii in range(len(files)):
        ff = '%sc%d' % (root, files[ii])
        print ff
        gcutil.rmall([ff + '_tmp.fits'])

        _coo = asciidata.open(ff + '.coo')

        xref = _coo[0][0]
        yref = _coo[1][0]

        if (ii == 0):
            xref0 = xref
            yref0 = yref

        xdiff = xref0 - xref
        ydiff = yref0 - yref

        # Shift the image
        iraf.imshift(ff + '.fits', ff + '_tmp.fits', xdiff, ydiff,
                     interp_type="spline3", boundary="constant")

        img = pyfits.getdata(ff + '_tmp.fits')
        hdr = pyfits.getheader(ff + '_tmp.fits')

        parang = hdr['PARANG']

        py.cla()
        py.imshow(log10(img), aspect='equal', interpolation='bicubic',
               vmin=2, vmax=4.5, origin='lowerleft')
        py.title('Parallactic Angle = %3d' % parang)
        py.savefig('movie/c%d.png' % (files[ii]))

def posErrorFromAllStars(root):
    """
    Using an aligned set of clean frames, calculate the positional
    error by taking the RMS of all frames with data. This pretty much
    does the same as align_rms. Then determine the average RMS from
    all stars with K < 14.5.
    """

    s = starset.StarSet(root)

    names = s.getArray('name')
    mag = s.getArray('mag')

    starCnt = len(s.stars)

    rmsX = np.zeros(starCnt, float)
    rmsY = np.zeros(starCnt, float)
    corr = np.zeros(starCnt, float)

    sampleFactor = math.sqrt(len(rmsX) / (len(rmsX) - 1.0))

    # For each star take the RMS error of the positions
    for ss in range(starCnt):
        star = s.stars[ss]

        x = star.getArrayAllEpochs('xpix')
        y = star.getArrayAllEpochs('ypix')

        corrEpoch = star.getArrayAllEpochs('corr')

        #x = x[1:]
        #y = y[1:]

        idx = (np.where((x > -999) & (corrEpoch > 0.8)))[0]

        if (len(idx) < 4):
            mag[ss] = 100
            continue
        
        x = x[idx]
        y = y[idx]

        rmsX[ss] = x.std() 
        rmsX[ss] *= sampleFactor * 9.95
        rmsY[ss] = y.std() 
        rmsY[ss] *= sampleFactor * 9.95

        corr[ss] = corrEpoch[idx].mean()

    py.clf()
    py.plot(mag, rmsX, 'b.')
    py.plot(mag, rmsY, 'r.')
    py.axis([8,15,0,0.5])
    #py.xlim(8, 15)

    idx = (np.where((mag < 14.5) & (corr > 0.8)))[0]

    rmsXall = rmsX[idx].mean()
    rmsYall = rmsY[idx].mean()

    print 'Number of Stars: %d' % len(idx)
    print 'RMS error in X = %5.2f and Y = %5.2f' % (rmsXall, rmsYall)
    py.savefig('posErrors.eps')


def posErrorAlignVsCent(root, epoch):
    s = starset.StarSet(root)

    x = s.getArray('x')
    y = s.getArray('y')
    xerr_p = s.getArrayFromEpoch(epoch, 'xerr_p')
    yerr_p = s.getArrayFromEpoch(epoch, 'yerr_p')
    xerr_a = s.getArrayFromEpoch(epoch, 'xerr_a')
    yerr_a = s.getArrayFromEpoch(epoch, 'yerr_a')
    num_ep = s.getArray('velCnt')

    idx = (np.where(xerr_p > 0))[0]
    x = x[idx]
    y = y[idx]
    xerr_p = xerr_p[idx] * 1000.0
    yerr_p = yerr_p[idx] * 1000.0
    xerr_a = xerr_a[idx] * 1000.0
    yerr_a = yerr_a[idx] * 1000.0
    num_ep = num_ep[idx]

    r = np.hypot(x, y)

    py.clf()
    py.subplot(2, 1, 1)
    py.semilogy(r, xerr_p, 'bx')
    py.semilogy(r, yerr_p, 'r+')
    py.ylabel('Centroid (mas)')
    py.title('Errors for Epoch #%d:  %8.3f' % (epoch, s.stars[0].years[epoch]))
    py.axis([0, 5, 0.1, 15])

    py.subplot(2, 1, 2)
    py.semilogy(r, xerr_a, 'bx')
    py.semilogy(r, yerr_a, 'r+')
    py.xlabel('Radius (arcsec)')
    py.ylabel('Alignment (mas)')
    py.axis([0, 5, 0.1, 15])

    py.savefig('plots/posErrorAlignVsCent_'+str(epoch)+'.png')


def posErrorXvsY(starlist, radCut=1000, magCut=1000):
    """
    Read in a *_rms.lis file and compare the difference between
    the X and Y errors with several plots.
    """

    from matplotlib.collections import EllipseCollection

    lis = starTables.StarfinderList(starlist, hasErrors=True)

    fileName = (lis.file.split('/'))[-1]
    fileName = (fileName.split('.'))[0]

    xmid = lis.x.min() + (lis.x.max() - lis.x.min())/2.0
    ymid = lis.y.min() + (lis.y.max() - lis.y.min())/2.0
    r = np.hypot(lis.x - xmid, lis.y - ymid)

    idx = np.where((lis.mag < magCut) & (r < radCut))[0]

    x = lis.x[idx]
    y = lis.y[idx]
    r = r[idx]
    xerr = lis.xerr[idx]
    yerr = lis.yerr[idx]
    diff = xerr - yerr

    # ==========
    # Plot up the distribution of X and Y errors
    # ==========
    bins = np.arange(0.0001, 2, 0.005)
    xbins, xdata = histNofill.hist(bins, xerr)
    ybins, ydata = histNofill.hist(bins, yerr)

    py.clf()
    py.semilogx(xbins, xdata)
    py.semilogx(ybins, ydata)
    py.xlabel('Positional Errors (pix)')
    py.ylabel('Number of Stars in ' + fileName)
    py.xlim(0.001, 2)
    py.title(fileName)
    py.savefig('poserr_x_vs_y_hists_'+ fileName +'.png')

    # ==========
    # Plot a Xerr-Yerr vs. positions (X, Y, r)
    # ==========
    py.clf()
    py.subplots_adjust(left=0.2, right=0.95, bottom=0.1, top=0.97, 
                       hspace=0.35)
    ylimit = 0.05

    py.subplot(3, 1, 1)
    py.plot(r, diff, 'k.')
    rng = py.axis()
    py.plot([rng[0], rng[1]], [0,0], 'r--')
    py.ylim(-ylimit, ylimit)
    py.xlabel('Radius (pix)')
    py.ylabel('Xerr - Yerr (pix)')

    py.subplot(3, 1, 2)
    py.plot(x, diff, 'k.')
    rng = py.axis()
    py.plot([rng[0], rng[1]], [0,0], 'r--')
    py.ylim(-ylimit, ylimit)
    py.xlabel('X (pix)')
    py.ylabel('Xerr - Yerr (pix)')

    py.subplot(3, 1, 3)
    py.plot(y, diff, 'k.')
    rng = py.axis()
    py.plot([rng[0], rng[1]], [0,0], 'r--')
    py.ylim(-ylimit, ylimit)
    py.xlabel('Y (pix)')
    py.ylabel('Xerr - Yerr (pix)')

    py.savefig('poserr_x_vs_y_diffxy_pos_' + fileName + '.png')

    # ==========
    # Plot the difference between X and Y
    # ==========
    binsIn = np.arange(-0.1, 0.1, 0.003)
    bins, data = histNofill.hist(binsIn, diff)

    meanDiff = diff.mean()
    meanStd = diff.std()
    meanDiffErr = meanStd / math.sqrt(len(diff))
    meanMedian = np.median(diff)

    py.clf()
    py.plot(bins, data)
    rng = py.axis()
    py.plot([0, 0], [rng[2], rng[3]], 'k--')
    py.xlabel('Xerr - Yerr (pix)')
    py.ylabel('Number of Stars in ' + fileName)
    py.title('Mean = %.4f +/- %.4f (%.1f sigma)' % 
             (meanDiff, meanDiffErr, meanDiff/meanDiffErr))
    py.savefig('poserr_x_vs_y_diff_'+ fileName +'.png')

    # Print average X and Y errors
    idx = np.where(lis.mag < 15)[0]
    print 'Mean X error: %4.2f mas' % xerr[idx].mean()
    print 'Mean Y error: %4.2f mas' % yerr[idx].mean()
    print ''

    # Print out some stats about the difference distribution
    print 'Differene between X and Y positional errors'
    print '  mean   = %.4f +/- %.4f' % (meanDiff, meanDiffErr)
    print '  std    = %.4f' % (meanStd)
    print '  median = %.4f' % (meanMedian)




    # ==========
    # Then plot a radialErr-tangentErr vs. distance from tip-tilt star??
    # ==========
    xtt = xmid - (10.1/0.01) # Assume center is roughly where Sgr A* is.
    ytt = ymid + (16.9/0.01)
    rtt = np.hypot(x - xtt, y - ytt)

    py.clf()
    py.subplots_adjust(left=0.15, right=0.95, bottom=0.08, top=0.97, 
                       hspace=0.32)
    ylimit = 0.05

    py.subplot(3, 1, 1)
    py.plot(rtt, diff, 'k.')
    rng = py.axis()
    py.plot([rng[0], rng[1]], [0,0], 'r--')
    py.ylim(-ylimit, ylimit)
    py.xlabel('Radial Distance from TT (pix)')
    py.ylabel('Xerr - Yerr (pix)')

    py.subplot(3, 1, 2)
    py.plot(x-xtt, diff, 'k.')
    rng = py.axis()
    py.plot([rng[0], rng[1]], [0,0], 'r--')
    py.ylim(-ylimit, ylimit)
    py.xlabel('X Distance from TT (pix)')
    py.ylabel('Xerr - Yerr (pix)')

    py.subplot(3, 1, 3)
    py.plot(y-ytt, diff, 'k.')
    rng = py.axis()
    py.plot([rng[0], rng[1]], [0,0], 'r--')
    py.ylim(-ylimit, ylimit)
    py.xlabel('Y Distance from TT (pix)')
    py.ylabel('Xerr - Yerr (pix)')

    py.savefig('poserr_x_vs_y_diffxy_ttpos_' + fileName + '.png')

    # ==========
    # Plot error ellipsoids for every star vs. detector position.
    # ==========
    py.clf()
    py.subplots_adjust(top=0.94)
    ax = py.subplot(1, 1, 1)

    idx = np.where((xerr < 0.25) & (yerr < 0.25))[0]
    angles = np.zeros(len(x[idx]))
    xy = np.column_stack((x[idx], y[idx]))
    
    scale = 0.15
    ec = EllipseCollection(xerr[idx]*scale, yerr[idx]*scale, 
                           angles, units='width', offsets=xy,
                           transOffset=ax.transData, facecolors='none')
    ecRef = EllipseCollection(np.array([0.05])*scale, np.array([0.05])*scale, 
                              [0], units='width', offsets=[[1100, 1100]],
                              transOffset=ax.transData, 
                              facecolors='none', edgecolors='red')
    ax.add_collection(ec)
    ax.add_collection(ecRef)
    ax.autoscale_view()
    py.title('Trimmed to errors < 0.25 pix')
    py.xlabel('X (pix)')
    py.ylabel('Y (pix)')
    py.text(1100, 1130, '0.05 pix', color='red', 
            horizontalalignment='center', verticalalignment='bottom')

    py.savefig('poserr_x_vs_y_err_ellipses_' + fileName + '.png')
    

def testNumberOfSubmaps():
    """
    Do a quick simulation to test whether the number of submaps
    we choose to use makes a difference.
    """
    subCnts = [3, 4, 5, 6, 8, 9, 10, 12, 15, 20]#, 25, 50, 100, 150]
    pointsCount = 120

    # Run a simulation with different random number 
    # distributions in each trial.
    simCnt = 1000

    stdDev = np.zeros((simCnt, len(subCnts)), dtype=float)
    errOnMean = np.zeros((simCnt, len(subCnts)), dtype=float)
    
    # Loop through Monte Carlo trials
    for aa in range(simCnt):
        x = py.randn(pointsCount)

        # Calculate the main-map position for this trial
        mainMapPos = x.mean()

        # Loop through the different sub-maps configurations
        for ss in range(len(subCnts)):
            subMapPos = np.zeros(subCnts[ss], dtype=float)
            subMapLen = len(x) / subCnts[ss]

            # Loop through and make the sub-maps
            for ii in range(subCnts[ss]):
                loIdx = ii * subMapLen
                hiIdx = loIdx + subMapLen

                # Calculate the sub-map position
                subMapPos[ii] = x[loIdx:hiIdx].mean()

            # Calculate the positional error for this sub-map configuration.
            # There are two possible positional errors we can use:
            #    - RMS error (standard deviation)
            #    - error on the mean 
            stdDev[aa,ss] = ((subMapPos - mainMapPos)**2).sum() 
            stdDev[aa,ss] /= (subCnts[ss] - 1)
            stdDev[aa,ss] = math.sqrt( stdDev[aa,ss] )

            errOnMean[aa,ss] = stdDev[aa,ss] / math.sqrt(subCnts[ss])

    # Calculate means and spreads for the positional errors as a function
    # of sub-map configuration (subCnts).
    stdDev_mean = stdDev.mean(axis=0)
    stdDev_std = stdDev.std(axis=0)
    errOnMean_mean = errOnMean.mean(axis=0)
    errOnMean_std = errOnMean.std(axis=0)

    # Plot
    py.clf()
    py.plot(subCnts, stdDev_mean, 'r-')
    py.fill_between(subCnts, 
                    stdDev_mean + stdDev_std, 
                    stdDev_mean - stdDev_std,
                    color='red', alpha=0.1)
    py.plot(subCnts, errOnMean_mean, 'b-')
    py.fill_between(subCnts, 
                    errOnMean_mean + errOnMean_std, 
                    errOnMean_mean - errOnMean_std,
                    color='blue', alpha=0.1)
    py.legend(('Standard Dev.', 'Error on Mean'), loc='upper left')
    py.xlabel('Number of Subsets')
    py.ylabel('Error on Position')
    py.title('Gaussian (mu=0, sigma=1), sampled with %d points' % pointsCount)
    py.savefig('testNumberOfSubmaps.png')



def comparePos_combo2submaps(epoch='10maylgs'):
    """
    Compares the positions from a combo map to the mean of the
    positions from the 3 submaps.
    """

    root = '/u/ghezgroup/data/gc/' + epoch + '/combo/starfinder/align/align_kp_0.8'

    # Read in the align
    s = starset.StarSet(root)
    cx = s.getArrayFromEpoch(0,'xpix')
    cy = s.getArrayFromEpoch(0,'ypix')
    s1x = s.getArrayFromEpoch(1,'xpix')
    s1y = s.getArrayFromEpoch(1,'ypix')
    s2x = s.getArrayFromEpoch(2,'xpix')
    s2y = s.getArrayFromEpoch(2,'ypix')
    s3x = s.getArrayFromEpoch(3,'xpix')
    s3y = s.getArrayFromEpoch(3,'ypix')
    cnt = s.getArray('velCnt')

    # Remove anything not in all 3 submaps and combo map
    idx = np.where(cnt == 4)[0]
    cx = cx[idx]
    cy = cy[idx]
    s1x = s1x[idx]
    s1y = s1y[idx]
    s2x = s2x[idx]
    s2y = s2y[idx]
    s3x = s3x[idx]
    s3y = s3y[idx]
    
    # compute mean of the 3 submaps and combo map
    sx_ave = np.zeros((len(s1x)),dtype=float)
    sy_ave = np.zeros((len(s1y)),dtype=float)
    for ii in range(len(cx)):
        sx_ave[ii] = np.mean([s1x[ii],s2x[ii],s3x[ii]])
        sy_ave[ii] = np.mean([s1y[ii],s2y[ii],s3y[ii]])

    dx = sx_ave - cx
    dy = sy_ave - cy

    print 'Average offset between submaps and combo map:'
    print 'dx = %4.2f +- %4.2f pix' % (dx.mean(),dx.std(ddof=1))
    print 'dy = %4.2f +- %4.2f pix' % (dy.mean(),dy.std(ddof=1))

    # Plot a histogram of the differences
    py.figure(figsize=(7,7))
    py.subplots_adjust(left=0.1,right=0.98,top=0.95)
    py.clf()
    binsIn = py.arange(-2, 2, 0.05)
    (nx,bx,ptx) = py.hist(dx,binsIn,histtype='step',lw=1.5,label='x')
    (ny,by,pty) = py.hist(dy,binsIn,histtype='step',lw=1.5,label='y')
    py.xlabel('Submap Ave - Combo Pos (pix)')
    py.ylabel('N')
    py.legend(numpoints=1,fancybox=True)
    py.axis([-1,1,0,max(nx.max(),ny.max())+20])
    py.savefig(root + '_comparePos_comboVsSubmaps.png')


