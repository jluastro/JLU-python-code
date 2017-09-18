import pylab as py
import numpy as np
import pyfits, math, datetime
import pickle
import histNofill
from gcwork import starset

class Pairwise(object):
    """
    This is a class to hold pairwise analysis information on a
    set of cleaned images from a single night. 
    """
    def __init__(self, root, refSrc='S1-5', align='align/align_t',
                 imgRoot='/u/ghezgroup/data/gc/06maylgs1/clean/kp/'):

        self.root = root
        self.refSrc = refSrc
        self.align = align
        self.imgRoot = imgRoot

        # Read in the aligned cleaned images
        s = starset.StarSet(root + align)

        self.names = s.getArray('name')
        self.mag = s.getArray('mag')
        self.starCnt = len(s.stars)
        self.epochCnt = len(s.stars[0].e)

        # Also need coordinates relative to a bright star near
        # the center of the field (refSrc)
        rid = self.names.index(refSrc)

        # Read in the list of images used in the alignment
        listFile = open(root+align+'.list', 'r')
        self.files = []
        for line in listFile:
            _data = line.split()
            self.files.append(_data[0])

        # Get the Strehl, FWHM, and Wave front error for each image
        print("Reading Strehl File")
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
        
        self.strehl = np.zeros(self.epochCnt, dtype=float)
        self.wfe = np.zeros(self.epochCnt, dtype=float)
        self.fwhm = np.zeros(self.epochCnt, dtype=float)
        self.elevation = np.zeros(self.epochCnt, dtype=float)
        self.darCoeff1 = np.zeros(self.epochCnt, dtype=float)
        self.darCoeff2 = np.zeros(self.epochCnt, dtype=float)
        self.airmass = np.zeros(self.epochCnt, dtype=float)
        self.parang = np.zeros(self.epochCnt, dtype=float)
        self.horizonX = np.zeros(self.epochCnt, dtype=float)
        self.horizonY = np.zeros(self.epochCnt, dtype=float)
        self.zenithX = np.zeros(self.epochCnt, dtype=float)
        self.zenithY = np.zeros(self.epochCnt, dtype=float)
        
        # Get the differential atmospheric refraction coefficients
        # assume all stars effective wavelengths are at 2.1 microns
        from gcwork import astrometry as ast
        (refA, refB) = ast.keckDARcoeffs(2.1)
        ast = None
        
        for ff in range(self.epochCnt):
            # Find the first instance of "/c"
            temp1 = self.files[ff].split('/')
            temp2 = temp1[-1].split('_')
            
            if (temp2[0].startswith('mag')):
                # This is a combo image. We don't have info about this file.
                # We should do this one last and adopt the average of
                # all values.
                self.strehl[ff] = None
                self.wfe[ff] = None
                self.fwhm[ff] = None

                self.files[ff] = imgRoot + '../../combo/' + \
                            '_'.join(temp2[0:-1]) + '.fits'
            else:
                # Find this file
                idx = _frameno.index(temp2[0] + '.fits')
                self.strehl[ff] = _strehl[idx]
                self.wfe[ff] = _wfe[idx]
                self.fwhm[ff] = _fwhm[idx]
                
                self.files[ff] = imgRoot + temp2[0] + '.fits'

            # Get header info
            hdr = pyfits.getheader(self.files[ff])

            effWave = hdr['EFFWAVE']
            self.elevation[ff] = hdr['EL']
            lamda = hdr['CENWAVE']
            self.airmass[ff] = hdr['AIRMASS']
            self.parang[ff] = hdr['PARANG']

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
            
            from keckdar import dar
            (refA, refB) = dar.keckDARcoeffs(effWave, hst.year,
                                             hst.month, hst.day,
                                             hst.hour, hst.minute)
            dar = None

            tanz = math.tan(math.radians(90.0 - self.elevation[ff]))
            tmp = 1 + tanz**2
            self.darCoeff1[ff] = tmp * (refA + 3.0 * refB * tanz**2)
            self.darCoeff2[ff] = -tmp * (refA*tanz +
                                         3.0 * refB * (tanz + 2.0*tanz**3))
            
            # Lets determine the zenith and horizon unit vectors for
            # this image.
            pa = math.radians(self.parang[ff] + float(hdr['ROTPOSN']) -
                              float(hdr['INSTANGL']))
            self.zenithX[ff] = math.sin(pa)
            self.zenithY[ff] = -math.cos(pa)
            self.horizonX[ff] = math.cos(pa)
            self.horizonY[ff] = math.sin(pa)
            
        # Now for each star in each image, calculate the positional difference
        # between that star and the reference star.
        self.posx = np.zeros((self.starCnt, self.epochCnt), float)
        self.posy = np.zeros((self.starCnt, self.epochCnt), float)
        self.posh = np.zeros((self.starCnt, self.epochCnt), float)
        self.posz = np.zeros((self.starCnt, self.epochCnt), float)
        self.diffx = np.zeros((self.starCnt, self.epochCnt), float)
        self.diffy = np.zeros((self.starCnt, self.epochCnt), float)
        self.diffh = np.zeros((self.starCnt, self.epochCnt), float)
        self.diffz = np.zeros((self.starCnt, self.epochCnt), float)
        self.diffr = np.zeros((self.starCnt, self.epochCnt), float)
        self.difft = np.zeros((self.starCnt, self.epochCnt), float)
        self.dar = np.zeros((self.starCnt, self.epochCnt), dtype=float)
        self.diffDR = np.zeros((self.starCnt, self.epochCnt), float)
        self.hasData = np.zeros((self.starCnt, self.epochCnt))
        
        # Positions and Errors to some reference source.
        self.avgx = np.zeros(self.starCnt, float)
        self.avgy = np.zeros(self.starCnt, float)
        self.rmsx = np.zeros(self.starCnt, float)
        self.rmsy = np.zeros(self.starCnt, float)
        self.rmsr = np.zeros(self.starCnt, float)
        self.rmst = np.zeros(self.starCnt, float)
        self.rmsRarc = np.zeros(self.starCnt, float)
        self.rmsTarc = np.zeros(self.starCnt, float)
        self.r = np.zeros(self.starCnt, float)
        
        # Get the reference coordinates at each epoch
        refStar = s.stars[rid]
        refCooX = np.zeros(self.epochCnt, float)
        refCooY = np.zeros(self.epochCnt, float)
        for ee in range(self.epochCnt):
            refCooX[ee] = refStar.e[ee].xorig
            refCooY[ee] = refStar.e[ee].yorig
            
        # Now for every epoch, compute the offset from the mean
        for ss in range(self.starCnt):
            star = s.stars[ss]

            # Original pixel positions
            xpix = star.getArrayAllEpochs('xorig')
            ypix = star.getArrayAllEpochs('yorig')

            xarc = (xpix - refCooX) * 0.00996
            yarc = (ypix - refCooY) * 0.00996

            # Filter out epochs with nodata
            idx = (np.where((xpix > -999) & (refCooX > -999)))[0]
            xgood = xarc[idx]
            ygood = yarc[idx]

            # Calculate the values averaged over all epochs
            self.avgx[ss] = xgood.mean()
            self.avgy[ss] = ygood.mean()
            self.r[ss] = np.sqrt(self.avgx[ss]**2 + self.avgy[ss]**2)
            self.rmsx[ss] = xgood.std() * 1000.0
            self.rmsy[ss] = ygood.std() * 1000.0

            # Calculate the values independently for each epoch
            self.posx[ss] = xarc
            self.posy[ss] = yarc
            self.posh[ss] = (xarc * self.horizonX) + (yarc * self.horizonY)
            self.posz[ss] = (xarc * self.zenithX) + (yarc * self.zenithY)
            
            dx = (xarc - self.avgx[ss]) * 1000.0
            dy = (yarc - self.avgy[ss]) * 1000.0

            avgX = self.avgx[ss]
            avgY = self.avgy[ss]
            avgR = np.sqrt(avgX**2 + avgY**2)
        
            # Convert into zenith (Y) and horizon(X) coordinates
            self.diffx[ss] = dx
            self.diffy[ss] = dy
            self.diffh[ss] = (dx * self.horizonX) + (dy * self.horizonY)
            self.diffz[ss] = (dx * self.zenithX) + (dy * self.zenithY)
            self.diffr[ss] = ((dx * avgX) + (dy * avgY)) / avgR
            self.difft[ss] = ((dx * -avgY) + (dy * avgX)) / avgR

            self.hasData[ss,idx] += 1
            self.hasData[ss, 0] = 0   # Turn off the combo epoch

            # Compute the predicted differential atmospheric refraction
            # between this star and the reference star in each of the
            # observed images.
            deltaZ = self.posz[ss] / 206265.0
            deltaR = self.darCoeff1 * deltaZ
            deltaR += self.darCoeff2 * deltaZ * abs(deltaZ)
            deltaR *= 206265.0

            self.dar[ss] = deltaR

            avgDR = deltaR[idx].mean()
            self.diffDR[ss] = (deltaR - avgDR) * 1000.0


        savefile = '%s/tables/pairwise_wrt_%s.pickle' % \
                   (self.root, self.refSrc)
        
        _save = open(savefile, 'w')
        pickle.dump(self, _save)
        _save.close()

def plotRmsSep(pairwise):
    """
    Pass in either the filename to a pickle file containing a
    Pairwise object or pass in the Pairwise object itself.
    """
    if (type(pairwise) == type('')):
        pw = pickle.load(open(pairwise))
    else:
        pw = pairwise

    rms = (pw.rmsx + pw.rmsy) / 2.0

    x = pw.avgx
    y = pw.avgy

    avgz = np.zeros(pw.starCnt, dtype=float)
    for ss in range(pw.starCnt):
        idx = (np.where(pw.hasData[ss,:] == 1))[0]
        avgz[ss] = abs(pw.diffz[ss,idx]).mean()

    # 2D plot
    py.clf()
    py.scatter(x, y, rms*50, edgecolor='k', facecolor='w')
    py.xlabel('X Separation (arcsec)')
    py.ylabel('Y Separation (arcsec)')
    py.title('RMS Error on Separation')
    py.scatter([-5], [5.2], 1.0 * 50, 'r')
    py.text(-4.7, 5, 'RMS = 1 mas')
    py.savefig('plots/rmsSeparation2d_'+pw.refSrc+'.png')
    py.savefig('plots/rmsSeparation2d_'+pw.refSrc+'.eps')

    py.clf()
    py.plot(np.hypot(x, y), rms, 'k.')
    py.xlabel('Separation (arcsec)')
    py.ylabel('RMS Error on Separation (mas)')
    py.savefig('plots/rmsSeparation_'+pw.refSrc+'.png')
    py.savefig('plots/rmsSeparation_'+pw.refSrc+'.eps')

    py.clf()
    py.plot(avgz, rms, 'k.')
    py.xlabel('Avg. Zenith Sep. (arcsec)')
    py.ylabel('RMS Error on Separation (mas)')
    py.savefig('plots/rmsZenithSeparation_'+pw.refSrc+'.png')
    py.savefig('plots/rmsZenithSeparation_'+pw.refSrc+'.eps')

    print('All Sources with rmsX and rmsY < 0.7 mas')
    idx = np.where((pw.rmsx < 0.7) & (pw.rmsy < 0.7))[0]

    for i in idx:
        print('%13s  pos = [%6.3f, %6.3f]  r = %6.3f  avgZ = %6.3f  rmsX = %4.2f  rmsY = %4.2f' % \
              (pw.names[i], x[i], y[i], pw.r[i], avgz[i],
               pw.rmsx[i], pw.rmsy[i]))

def plotSepVsParang(pairwise, src):
    """
    Pass in either the filename to a pickle file containing a
    Pairwise object or pass in the Pairwise object itself.
    """
    if (type(pairwise) == type('')):
        pw = pickle.load(open(pairwise))
    else:
        pw = pairwise
        
    ndx = pw.names.index(src)
    idx = np.where(pw.hasData[ndx] == 1)[0]

    # Calculate predicted change in separation due to DAR for this star
    parang = pw.parang[idx]
    dzObs = pw.posz[ndx,idx]
    dhObs = pw.posh[ndx,idx]

    dR = pw.dar[ndx,idx]

    dzObsAvg = (pw.avgx[ndx] * pw.zenithX) + (pw.avgy[ndx] * pw.zenithY)
    dhObsAvg = (pw.avgx[ndx] * pw.horizonX) + (pw.avgy[ndx] * pw.horizonY)
    dzObsAvg = dzObsAvg[idx]
    dhObsAvg = dhObsAvg[idx]


    dzTrue = (np.abs(dzObs) + dR) * np.sign(dzObs)
    sepObs = np.hypot(dzObs, dhObs)
    sepTrue = np.hypot(dzTrue, dhObs)

    print(dzTrue.mean(), dzObs.mean(), dzObsAvg.mean(), dR.mean())


    py.clf()
    py.plot(parang, sepObs, 'k.')
    py.plot(parang, sepTrue, 'r-')
    #py.plot(parang, dzObsAvg - dzTrue, 'r-')
    #py.plot(parang, dzObsAvg - dR, 'r-')
    #py.plot(parang, sepObs, 'k.')
    #py.plot(parang, sepTrue, 'r-')

    print('Separation = %7.4f +/- %6.4f from %d points' % \
          (sepObs.mean(), sepObs.std(), len(sepObs)))


def plotSepVsStrehl(pairwise, src):
    """
    Pass in either the filename to a pickle file containing a
    Pairwise object or pass in the Pairwise object itself.
    """
    if (type(pairwise) == type('')):
        pw = pickle.load(open(pairwise))
    else:
        pw = pairwise
        
    ndx = pw.names.index(src)
    idx = np.where(pw.hasData[ndx] == 1)[0]

    # Calculate predicted change in separation due to DAR for this star
    strehl = pw.strehl[idx]
    dzObs = pw.posz[ndx,idx]
    dhObs = pw.posh[ndx,idx]
    sepObs = np.hypot(dzObs, dhObs)

    py.clf()
    py.plot(strehl, sepObs, 'k.')

    print('Separation = %7.4f +/- %6.4f from %d points' % \
          (sepObs.mean(), sepObs.std(), len(sepObs)))

def plotZHvsParang(pairwise, src):
    """
    Pass in either the filename to a pickle file containing a
    Pairwise object or pass in the Pairwise object itself.
    """
    if (type(pairwise) == type('')):
        pw = pickle.load(open(pairwise))
    else:
        pw = pairwise
        
    ndx = pw.names.index(src)
    idx = np.where(pw.hasData[ndx] == 1)[0]

    py.clf()
    py.subplot(2, 1, 1)
    py.plot(pw.parang[idx], pw.posx[ndx, idx], 'k.')

    py.subplot(2, 1, 2)
    py.plot(pw.parang[idx], pw.posy[ndx, idx], 'k.')

    py.show()
    
def plotNightlyProps(pairwise, outdir='./'):
    if (type(pairwise) == type('')):
        pw = pickle.load(open(pairwise))
    else:
        pw = pairwise
    
    # Make some additional plots
    py.clf()
    fwhmCut = 1.25 * pw.fwhm.min()
    py.plot(pw.parang, pw.fwhm, 'k.')
    rng = py.axis()
    py.plot([rng[0], rng[1]], [fwhmCut, fwhmCut], 'k--')
    py.ylim(45, 95)
    py.xlabel('Parallactic Angle (deg)')
    py.ylabel('FWHM (mas)')
    py.savefig(outdir + 'plots/fwhm_vs_parang.png')
    py.savefig(outdir + 'plots/fwhm_vs_parang.eps')

    py.clf()
    py.plot(pw.parang, pw.strehl, 'k.')
    py.xlabel('Parallactic Angle (deg)')
    py.ylabel('Strehl')
    py.ylim(0.1, 0.5)
    py.savefig(outdir + 'plots/strehl_vs_parang.png')
    py.savefig(outdir + 'plots/strehl_vs_parang.eps')
    
    py.clf()
    py.plot(pw.parang, pw.airmass, 'k.')
    py.xlabel('Parallactic Angle (deg)')
    py.ylabel('Airmass')
    py.savefig(outdir + 'plots/airmass_vs_parang.png')
    py.savefig(outdir + 'plots/airmass_vs_parang.eps')


def plotPairwise(pairwise, src, fixDarOn=True, ylim=1.5):
    if (type(pairwise) == type('')):
        pw = pickle.load(open(pairwise))
    else:
        pw = pairwise
        
    idx = pw.names.index(src)
    edx = np.where(pw.hasData[idx] == 1)[0]

    errx = pw.diffx[idx,edx].std()
    erry = pw.diffy[idx,edx].std()

    print('pos = (%5.2f, %5.2f) asec   perr = (%5.2f, %5.2f) mas for %s' % \
          (pw.avgx[idx], pw.avgy[idx], errx, erry, src))
    print('Parallactic Angle range: %5.1f - %5.1f' % \
          (pw.parang.min(), pw.parang.max()))
    print('Airmass range:           %5.2f - %5.2f' % \
          (pw.airmass.min(), pw.airmass.max()))
    print('Elevation range:         %5.2f - %5.2f' % \
          (pw.elevation.min(), pw.elevation.max()))

    c = 'k'

    def plotStuff(xdat, xlab, suffix):
        py.clf()
        py.subplots_adjust(hspace=0.12, top=0.93)

        #####
        # Plot positions
        #####
        py.subplot(2, 1, 1)
        py.plot(xdat, pw.posx[idx, edx], 'k.', mfc=c, mec=c)
        rng = py.axis()
        py.plot([rng[0], rng[1]], [pw.avgx[idx], pw.avgx[idx]], 'k--')
        py.ylabel('Sep. (arcsec)')
        legX = rng[0] + (rng[1] - rng[0]) * 0.98
        legY = rng[2] + ((rng[3] - rng[2]) * 0.9)
        py.text(legX, legY, 'X', fontweight='bold',
                horizontalalignment='right', verticalalignment='top')
        py.title('%s vs. %s dX = %4.1f  dY=%4.1f' % \
                 (src, pw.refSrc, pw.avgx[idx], pw.avgy[idx]))

        py.subplot(2, 1, 2)
        py.plot(xdat, pw.posy[idx, edx], 'k.', mfc=c, mec=c)
        rng = py.axis()
        py.plot([rng[0], rng[1]], [pw.avgy[idx], pw.avgy[idx]], 'k--')
        py.ylabel('Sep. (arcsec)')
        legX = rng[0] + (rng[1] - rng[0]) * 0.98
        legY = rng[2] + ((rng[3] - rng[2]) * 0.9)
        py.text(legX, legY, 'Y', fontweight='bold',
                horizontalalignment='right', verticalalignment='top')
            
        py.savefig('plots/pw_xy_%s_%s.png' % (suffix, src))
        py.savefig('plots/pw_xy_%s_%s.eps' % (suffix, src))

        #####
        # Horizon & Zenith
        #####
        py.clf()
        py.subplot(2, 1, 1)
        py.plot(xdat, pw.diffh[idx, edx], 'k.', mfc=c, mec=c)
        rng = py.axis()
        py.plot([rng[0], rng[1]], [0, 0], 'k--')
        py.axis([rng[0], rng[1], -ylim, ylim])
        py.ylabel('Delta-Sep. (mas)')
        legX = rng[0] + (rng[1] - rng[0]) * 0.98
        legY = 1.4
        py.text(legX, legY, 'Horizon', fontweight='bold',
                horizontalalignment='right', verticalalignment='top')
        py.title('%s vs. %s dX = %4.1f  dY=%4.1f' % \
                 (src, pw.refSrc, pw.avgx[idx], pw.avgy[idx]))

        py.subplot(2, 1, 2)
        py.plot(xdat, pw.diffz[idx, edx], 'k.', mfc=c, mec=c)
        if (suffix == 'parang' and fixDarOn == False):
            py.plot(xdat, -pw.diffDR[idx, edx], 'r-')
        py.plot([rng[0], rng[1]], [0, 0], 'k--')
        py.axis([rng[0], rng[1], -ylim, ylim])
        py.xlabel(xlab)
        py.ylabel('Delta-Sep. (mas)')
        py.text(legX, legY, 'Zenith', fontweight='bold',
                horizontalalignment='right', verticalalignment='top')
            
        py.savefig('plots/pw_zh_%s_%s.png' % (suffix, src))
        py.savefig('plots/pw_zh_%s_%s.eps' % (suffix, src))

        #####
        # Radial & Tangential
        #####
        py.clf()
        py.subplot(2, 1, 1)
        py.plot(xdat, pw.diffr[idx, edx], 'k.', mfc=c, mec=c)
        rng = py.axis()
        py.plot([rng[0], rng[1]], [0, 0], 'k--')
        py.axis([rng[0], rng[1], -ylim, ylim])
        py.ylabel('Delta-Sep. (mas)')
        legX = rng[0] + (rng[1] - rng[0]) * 0.98
        legY = 1.4
        py.text(legX, legY, 'Radial', fontweight='bold',
                horizontalalignment='right', verticalalignment='top')
        py.title('%s vs. %s dX = %4.1f  dY=%4.1f' % \
                 (src, pw.refSrc, pw.avgx[idx], pw.avgy[idx]))
            
        py.subplot(2, 1, 2)
        py.plot(xdat, pw.difft[idx, edx], 'k.', mfc=c, mec=c)
        py.plot([rng[0], rng[1]], [0, 0], 'k--')
        py.axis([rng[0], rng[1], -ylim, ylim])
        py.xlabel(xlab)
        py.ylabel('Delta-Sep. (mas)')
        legX = rng[0] + (rng[1] - rng[0]) * 0.98
        legY = 1.4
        py.text(legX, legY, 'Tangential', fontweight='bold',
                horizontalalignment='right', verticalalignment='top')
            
        py.savefig('plots/pw_rt_%s_%s.png' % (suffix, src))
        py.savefig('plots/pw_rt_%s_%s.eps' % (suffix, src))

    
    # Plot Strehl vs. Pos Diff
    plotStuff(pw.strehl[edx], 'Strehl', 'strehl')
    
    # Plot FWHM vs. Pos Diff
    plotStuff(pw.fwhm[edx], 'FWHM (mas)', 'fwhm')
    
    # Plot airmass vs. Pos Diff
    plotStuff(pw.airmass[edx], 'Airmass', 'airmass')
    
    # Plot ParAng vs. Pos Diff
    plotStuff(pw.parang[edx], 'Parallactic Angle (deg)', 'parang')


def compareDARstar(pwDARfixOn, pwDARfixOff, src, outdir='./'):
    """
    Pass in either the filename to a pickle file containing a
    Pairwise object or pass in the Pairwise object itself for both
    a DAR corrected and uncorrected data set. This way we can plot up
    both and compare them. Also plot the model DAR correction which
    is the average separation of the pre-DAR + model-DAR values
    for all frames with strehl > 0.30.
    """
    # Rename variables for brevity
    if (type(pwDARfixOn) == type('')):
        pw1 = pickle.load(open(pwDARfixOn))
    else:
        pw1 = pwDARfixOn

    if (type(pwDARfixOff) == type('')):
        pw2 = pickle.load(open(pwDARfixOff))
    else:
        pw2 = pwDARfixOff
        
    ndx1 = pw1.names.index(src)
    ndx2 = pw2.names.index(src)

    idx1 = np.where(pw1.hasData[ndx1] == 1)[0]
    idx2 = np.where(pw2.hasData[ndx2] == 1)[0]

    # Calculate predicted change in separation due to DAR for this star
    parang1 = pw1.parang[idx1]
    parang2 = pw2.parang[idx2]

    strehl1 = pw1.strehl[idx1]
    strehl2 = pw2.strehl[idx2]

    scale = 0.00996
    dzObs1 = pw1.posz[ndx1,idx1] / scale
    dhObs1 = pw1.posh[ndx1,idx1] / scale
    dR1 = pw1.dar[ndx1,idx1] / scale

    dzObs2 = pw2.posz[ndx2,idx2] / scale
    dhObs2 = pw2.posh[ndx2,idx2] / scale
    dR2 = pw2.dar[ndx2,idx2] / scale

    sepObs1 = np.hypot(dzObs1, dhObs1)
    sepObs2 = np.hypot(dzObs2, dhObs2)

    sepTrue1 = np.hypot(dzObs1, dhObs1)
    sepTrue2 = np.hypot(dzObs2 + dR2, dhObs2)

    # Figure out the highest strehl data
    sdx = np.where(strehl2 > 0.30)[0]
    sepTrueAvg = sepTrue2[sdx].mean()

    py.clf()
    py.subplots_adjust(left=0.15)
    # tick formatting
    yformatter = py.FormatStrFormatter('%5.1f')
    ax = py.gca()
    ax.yaxis.set_major_formatter(yformatter)

    # Plot pre-DAR corrected
    py.plot(parang2, sepObs2, 'r.')

    # Plot model DAR corrected
    rng = py.axis()
    py.plot([rng[0], rng[1]], [sepTrueAvg, sepTrueAvg], 'k-')

    # Plot post-DAR corrected
    py.plot(parang1, sepObs1, 'k.')

    py.title(src)
    py.xlabel('Parallactic Angle (deg)')
    py.ylabel('Distance from %s (pixels)' % (pw1.refSrc))
    py.legend(('Pre-DAR', 'Model DAR',
               'Post-DAR'), numpoints=3)

    py.savefig(outdir + 'plots/compare_dar_%s.eps' % src)
    py.savefig(outdir + 'plots/compare_dar_%s.png' % src)
    

def compareDAR(pwDARfixOn, pwDARfixOff, outdir='./'):
    """
    Pass in either the filename to a pickle file containing a
    Pairwise object or pass in the Pairwise object itself for both
    a DAR corrected and uncorrected data set. This way we can plot up
    both and compare them for all the stars.
    """
    # Rename variables for brevity
    if (type(pwDARfixOn) == type('')):
        pw1 = pickle.load(open(pwDARfixOn))
    else:
        pw1 = pwDARfixOn

    if (type(pwDARfixOff) == type('')):
        pw2 = pickle.load(open(pwDARfixOff))
    else:
        pw2 = pwDARfixOff

    # We can only compare stars that have matches in both.
    # Also restrict to just those stars with K<14.
    idx1 = []
    idx2 = []
    for j1 in range(pw1.starCnt):
        try:
            j2 = pw2.names.index(pw1.names[j1])

            if (pw1.mag[j1] > 14):
                idx1.append(j1)
                idx2.append(j2)
        except ValueError:
            # Skip
            continue
    
    # Calculate the average position of each star relative
    # to the reference star.
    starCnt = len(idx1)
    avgx1 = np.zeros(starCnt, dtype=float)
    avgy1 = np.zeros(starCnt, dtype=float)
    rmsx1 = np.zeros(starCnt, dtype=float)
    rmsy1 = np.zeros(starCnt, dtype=float)
    rmsz1 = np.zeros(starCnt, dtype=float)
    rmsh1 = np.zeros(starCnt, dtype=float)
    rmsr1 = np.zeros(starCnt, dtype=float)
    rmst1 = np.zeros(starCnt, dtype=float)

    avgx2 = np.zeros(starCnt, dtype=float)
    avgy2 = np.zeros(starCnt, dtype=float)
    rmsx2 = np.zeros(starCnt, dtype=float)
    rmsy2 = np.zeros(starCnt, dtype=float)
    rmsz2 = np.zeros(starCnt, dtype=float)
    rmsh2 = np.zeros(starCnt, dtype=float)
    rmsr2 = np.zeros(starCnt, dtype=float)
    rmst2 = np.zeros(starCnt, dtype=float)

    for ii in range(starCnt):
        j1 = idx1[ii]
        j2 = idx2[ii]

        # For this star, find the epochs with data.
        e1 = np.where(pw1.hasData[j1] == 1)[0]
        e2 = np.where(pw2.hasData[j2] == 1)[0]

        avgx1[ii] = pw1.posx[j1, e1].mean()
        avgy1[ii] = pw1.posy[j1, e1].mean()
        rmsx1[ii] = np.sqrt((pw1.diffx[j1, e1]**2).sum() / (starCnt - 1))
        rmsy1[ii] = np.sqrt((pw1.diffy[j1, e1]**2).sum() / (starCnt - 1))
        rmsh1[ii] = np.sqrt((pw1.diffh[j1, e1]**2).sum() / (starCnt - 1))
        rmsz1[ii] = np.sqrt((pw1.diffz[j1, e1]**2).sum() / (starCnt - 1))
        rmsr1[ii] = np.sqrt((pw1.diffr[j1, e1]**2).sum() / (starCnt - 1))
        rmst1[ii] = np.sqrt((pw1.difft[j1, e1]**2).sum() / (starCnt - 1))

        avgx2[ii] = pw2.posx[j2, e2].mean()
        avgy2[ii] = pw2.posy[j2, e2].mean()
        rmsx2[ii] = np.sqrt((pw2.diffx[j2, e2]**2).sum() / (starCnt - 1))
        rmsy2[ii] = np.sqrt((pw2.diffy[j2, e2]**2).sum() / (starCnt - 1))
        rmsh2[ii] = np.sqrt((pw2.diffh[j2, e2]**2).sum() / (starCnt - 1))
        rmsz2[ii] = np.sqrt((pw2.diffz[j2, e2]**2).sum() / (starCnt - 1))
        rmsr2[ii] = np.sqrt((pw2.diffr[j2, e2]**2).sum() / (starCnt - 1))
        rmst2[ii] = np.sqrt((pw2.difft[j2, e2]**2).sum() / (starCnt - 1))

    # Now calculate average separation
    avgr1 = np.hypot(avgx1, avgy1)
    avgr2 = np.hypot(avgx2, avgy2)

    py.clf()
    py.plot(avgr1, (avgr1-avgr2)*10**3, 'k.')
    rng = py.axis()
    py.plot([rng[0], rng[1]], [0, 0], 'k--')
    py.xlabel('DAR Corrected (arcsec)')
    py.ylabel('DAR Corrected - Uncorrected (mas)')
    py.title('Average Distance from S1-5')
    py.axis([0, 7, -0.5, 2.5])
    py.savefig(outdir + 'plots/compare_dar_sep.eps')
    py.savefig(outdir + 'plots/compare_dar_sep.png')

    # Same thing but convert back to pixels.
    py.clf()
    py.plot(avgr1 / 0.00996, (avgr1-avgr2)/0.00996, 'k.')
    rng = py.axis()
    py.plot([rng[0], rng[1]], [0, 0], 'k--')
    py.xlabel('DAR Corrected (pixels)')
    py.ylabel('DAR Corrected - Uncorrected (pixels)')
    py.title('Average Distance from S1-5')
    py.axis([0, 700, -0.05, 0.3])
    py.savefig(outdir + 'plots/compare_dar_sep_pix.eps')
    py.savefig(outdir + 'plots/compare_dar_sep_pix.png')

    py.clf()
    py.plot(rmsx2, rmsx2-rmsx1, 'r,')
    py.plot(rmsy2, rmsy2-rmsy1, 'b,')
    rng = py.axis()
    py.plot([rng[0], rng[1]], [0, 0], 'k--')
    py.xlabel('DAR Uncorrected (mas)')
    py.ylabel('DAR Uncorrected - Corrected (mas)')
    py.title('RMS Error in Distance from S1-5')
    leg = py.legend(('NIRC2 X', 'NIRC2 Y'), numpoints=1, markerscale=2)
    txts = leg.get_texts()
    txts[0].set_color('r')
    txts[1].set_color('b')
    py.ylim(-0.6, 0.6)
    py.savefig(outdir + 'plots/compare_dar_rmsxy.eps')
    py.savefig(outdir + 'plots/compare_dar_rmsxy.png')

    py.clf()
    py.plot(rmsh2, rmsh2-rmsh1, 'r,')
    py.plot(rmsz2, rmsz2-rmsz1, 'b,')
    rng = py.axis()
    py.plot([rng[0], rng[1]], [0, 0], 'k--')    
    py.xlabel('DAR Uncorrected (mas)')
    py.ylabel('DAR Uncorrected - Corrected (mas)')
    py.title('RMS Error in Distance from S1-5')
    leg = py.legend(('Horizon', 'Zenith'), numpoints=1, markerscale=2)
    txts = leg.get_texts()
    txts[0].set_color('r')
    txts[1].set_color('b')
    py.ylim(-0.6, 0.6)
    py.savefig(outdir + 'plots/compare_dar_rmshz.eps')
    py.savefig(outdir + 'plots/compare_dar_rmshz.png')

    py.clf()
    py.plot(avgr1, (rmsh2 - rmsh1)/rmsh2, 'r.')
    py.plot(avgr1, (rmsz2 - rmsz1)/rmsz2, 'b.')
    rng = py.axis()
    py.plot([rng[0], rng[1]], [0, 0], 'k--')
    py.xlabel('Distance from S1-5 (arcsec)')
    py.ylabel('Fractional Reduction in RMS Error')
    py.title('RMS Error in Distance from S1-5')
    leg = py.legend(('Horizon', 'Zenith'), numpoints=1, markerscale=2)
    txts = leg.get_texts()
    txts[0].set_color('r')
    txts[1].set_color('b')
    py.quiver([6.1, 6.1], [0, 0], [0, 0], [0.09, -0.09], scale=1.0)
    py.text(6.1, 0.1, 'Decreased\nErrors', fontsize=10,
            horizontalalignment='center', verticalalignment='bottom')
    py.text(6.1, -0.1, 'Increased\nErrors', fontsize=10,
            horizontalalignment='center', verticalalignment='top')
    py.ylim(-0.5, 0.5)
    py.savefig(outdir + 'plots/compare_dar_rmshz_sep.eps')
    py.savefig(outdir + 'plots/compare_dar_rmshz_sep.png')

    py.clf()
    py.plot(rmsh2, (rmsh2 - rmsh1)/rmsh2, 'r.')
    py.plot(rmsz2, (rmsz2 - rmsz1)/rmsz2, 'b.')
    py.xlabel('RMS Error in Distance (mas)')
    py.ylabel('Fractional Reduction in RMS Error')
    leg = py.legend(('Horizon', 'Zenith'), numpoints=1, markerscale=2)
    txts = leg.get_texts()
    txts[0].set_color('r')
    txts[1].set_color('b')
    py.quiver([4.1, 4.1], [0, 0], [0, 0], [0.09, -0.09], scale=1.0)
    py.text(4.1, 0.1, 'Decreased\nErrors', fontsize=10,
            horizontalalignment='center', verticalalignment='bottom')
    py.text(4.1, -0.1, 'Increased\nErrors', fontsize=10,
            horizontalalignment='center', verticalalignment='top')
    rng = py.axis()
    py.plot([rng[0], rng[1]], [0, 0], 'k--')
    py.ylim(-0.5, 0.5)
    py.savefig(outdir + 'plots/compare_dar_rmshz_rms.eps')
    py.savefig(outdir + 'plots/compare_dar_rmshz_rms.png')

    py.clf()
    py.plot(rmsr2, rmsr2-rmsr1, 'r,')
    py.plot(rmst2, rmst2-rmst1, 'b,')
    rng = py.axis()
    py.plot([rng[0], rng[1]], [0, 0], 'k--')
    py.xlabel('DAR Uncorrected (mas)')
    py.ylabel('DAR Uncorrected - Corrected (mas)')
    py.title('RMS Error in Distance from S1-5')
    leg = py.legend(('Radial', 'Tangential'), numpoints=1, markerscale=2)
    txts = leg.get_texts()
    txts[0].set_color('r')
    txts[1].set_color('b')
    py.ylim(-0.6, 0.6)
    py.savefig(outdir + 'plots/compare_dar_rmsrt.eps')
    py.savefig(outdir + 'plots/compare_dar_rmsrt.png')

    # Also plot histograms for each of the RMS values.
    binsIn = np.arange(-0.5, 0.5, 0.05)

    (binx, datx) = histNofill.hist(binsIn, rmsx2-rmsx1)
    (biny, daty) = histNofill.hist(binsIn, rmsy2-rmsy1)
    datx /= len(rmsx2)
    daty /= len(rmsx2)
    py.clf()
    py.plot(binx, datx, 'r-')
    py.plot(biny, daty, 'b-')
    py.quiver([-0.1, 0.1], [0.2, 0.2], [-0.2, 0.2], [0, 0], scale=1.1)
    py.text(0.2, 0.21, 'Decreased\nErrors', fontsize=10,
            horizontalalignment='left', verticalalignment='bottom')
    py.text(-0.2, 0.21, 'Increased\nErrors', fontsize=10,
            horizontalalignment='right', verticalalignment='bottom')
    rng = py.axis()
    py.plot([0, 0], [rng[2], rng[3]], 'k--')
    py.ylabel('Fraction of Stars')
    py.xlabel('DAR Uncorrected - Corrected (mas)')
    py.title('RMS Error in Distance from S1-5')
    leg = py.legend(('NIRC2 X', 'NIRC2 Y'), markerscale=2)
    txts = leg.get_texts()
    txts[0].set_color('r')
    txts[1].set_color('b')
    py.savefig(outdir + 'plots/compare_dar_rmsxy_hist.eps')
    py.savefig(outdir + 'plots/compare_dar_rmsxy_hist.png')

    (binh, dath) = histNofill.hist(binsIn, rmsh2-rmsh1)
    (binz, datz) = histNofill.hist(binsIn, rmsz2-rmsz1)
    dath /= len(rmsh2)
    datz /= len(rmsz2)
    py.clf()
    py.plot(binh, dath, 'r-')
    py.plot(binz, datz, 'b-')
    py.quiver([-0.1, 0.1], [0.2, 0.2], [-0.2, 0.2], [0, 0], scale=1.1)
    py.text(0.2, 0.21, 'Decreased\nErrors', fontsize=10,
            horizontalalignment='left', verticalalignment='bottom')
    py.text(-0.2, 0.21, 'Increased\nErrors', fontsize=10,
            horizontalalignment='right', verticalalignment='bottom')
    rng = py.axis()
    py.plot([0, 0], [rng[2], rng[3]], 'k--')
    py.ylabel('Number of Stars')
    py.xlabel('DAR Uncorrected - Corrected (mas)')
    py.title('RMS Error in Distance from S1-5')
    leg = py.legend(('Horizon', 'Zenith'), markerscale=2)
    txts = leg.get_texts()
    txts[0].set_color('r')
    txts[1].set_color('b')
    py.savefig(outdir + 'plots/compare_dar_rmshz_hist.eps')
    py.savefig(outdir + 'plots/compare_dar_rmshz_hist.png')

    (binr, datr) = histNofill.hist(binsIn, rmsr2-rmsr1)
    (bint, datt) = histNofill.hist(binsIn, rmst2-rmst1)
    datr /= len(rmsr2)
    datt /= len(rmst2)
    py.clf()
    py.plot(binr, datr, 'r-')
    py.plot(bint, datt, 'b-')
    py.quiver([-0.1, 0.1], [0.2, 0.2], [-0.2, 0.2], [0, 0], scale=1.1)
    py.text(0.2, 0.21, 'Decreased\nErrors', fontsize=10,
            horizontalalignment='left', verticalalignment='bottom')
    py.text(-0.2, 0.21, 'Increased\nErrors', fontsize=10,
            horizontalalignment='right', verticalalignment='bottom')
    rng = py.axis()
    py.plot([0, 0], [rng[2], rng[3]], 'k--')
    py.ylabel('Number of Stars')
    py.xlabel('DAR Uncorrected - Corrected (mas)')
    py.title('RMS Error in Distance from S1-5')
    leg = py.legend(('Radial', 'Tangential'), markerscale=2)
    txts = leg.get_texts()
    txts[0].set_color('r')
    txts[1].set_color('b')
    py.savefig(outdir + 'plots/compare_dar_rmsrt_hist.eps')
    py.savefig(outdir + 'plots/compare_dar_rmsrt_hist.png')


def vectorSeps(pairwise, strehlLim=0.30):
    """
    Make a vector plot (in NIRC2 coordinates) of the difference between
    each star's separation from S1-5 at high and low strehls. 
    """
    if (type(pairwise) == type('')):
        pw = pickle.load(open(pariwise))
    else:
        pw = pairwise

    xLo = np.zeros(pw.starCnt, dtype=float)
    yLo = np.zeros(pw.starCnt, dtype=float)
    sepLo = np.zeros(pw.starCnt, dtype=float)

    xHi = np.zeros(pw.starCnt, dtype=float)
    yHi = np.zeros(pw.starCnt, dtype=float)
    sepHi = np.zeros(pw.starCnt, dtype=float)

    for ss in range(pw.starCnt):
        idxLo = np.where((pw.hasData[ss] == 1) & (pw.strehl < strehlLim))[0]
        idxHi = np.where((pw.hasData[ss] == 1) & (pw.strehl >= strehlLim))[0]

        # Calculate average separation.
        xLo[ss] = pw.posx[ss,idxLo].mean()
        yLo[ss] = pw.posy[ss,idxLo].mean()
        sepLo[ss] = np.hypot(pw.posx[ss,idxLo], pw.posy[ss,idxLo]).mean()

        xHi[ss] = pw.posx[ss,idxHi].mean()
        yHi[ss] = pw.posy[ss,idxHi].mean()
        sepHi[ss] = np.hypot(pw.posx[ss,idxHi], pw.posy[ss,idxHi]).mean()

    xdiff = xHi - xLo
    ydiff = yHi - yLo

    print(xdiff.mean())
    print(ydiff.mean())

    py.clf()
    py.quiver(xHi, yHi, xdiff, ydiff, scale=0.04)
    py.quiver([4], [5], [0.005], [0], scale=0.04, color='r')
    py.text(4, 5.1, '5 mas', color='r')

    py.savefig('plots/vector_sep_lohi_strehl.png')
    py.savefig('plots/vector_sep_lohi_strehl.eps')

    py.clf()
    py.hist(xdiff)
    py.hist(ydiff)

def vectorSepsMovie(pairwise):
    """
    Make a vector plot (in NIRC2 coordinates) of the difference between
    each star's separation from S1-5 at high and low strehls. 
    """
    if (type(pairwise) == type('')):
        pw = pickle.load(open(pariwise))
    else:
        pw = pairwise

    # Calculate the average separation between each star and the
    # reference star for our highest strehl data.
    xAvg = np.zeros(pw.starCnt, dtype=float)
    yAvg = np.zeros(pw.starCnt, dtype=float)
    sepAvg = np.zeros(pw.starCnt, dtype=float)
    strehlLim = 0.30

    for ss in range(pw.starCnt):
        idx = np.where((pw.hasData[ss] == 1) & (pw.strehl >= strehlLim))[0]

        xAvg[ss] = pw.posx[ss,idx].mean()
        yAvg[ss] = pw.posy[ss,idx].mean()
        sepAvg[ss] = np.hypot(pw.posx[ss,idx], pw.posy[ss,idx]).mean()

    # Loop through each cleaned image, and make a vector plot of each
    # stars seperation vs. the average seperation.
#     print xAvg[0]
#     print yAvg[0]
#     print pw.posx[0,0:10]
#     print pw.posy[0,0:10]
    for ee in range(pw.epochCnt):
        idx = np.where(pw.hasData[:,ee] == 1)[0]

        xdiff = pw.posx[idx,ee] - xAvg[idx]
        ydiff = pw.posy[idx,ee] - yAvg[idx]

        py.clf()
        py.quiver(xAvg[idx], yAvg[idx], xdiff, ydiff, scale=0.04)

        # Plot a reference arrow in the direction of the zenith
        refx = 0.005 * math.sin(-math.radians(pw.parang[ee]))
        refy = 0.005 * math.cos(-math.radians(pw.parang[ee]))
        py.quiver([4], [5], [refx], [refy], scale=0.04, color='r')
        py.title('PA = %4d' % pw.parang[ee])

#         foo = raw_input('Elev. = %4.1f  Airmass = %4.2f  PA = %4d  Strehl = %4.2f' % \
#               (pw.elevation[ee], pw.airmass[ee], pw.parang[ee], pw.strehl[ee]))

        if (foo == 'q' or foo == 'Q'):
            return


def histZHrms(pairwise, plot=True, outdir='./'):
    """
    Make a histogram of the RMS error in the zenith and horizon components
    of all star's separation vectors.
    """
    if (type(pairwise) == type('')):
        pw = pickle.load(open(pariwise))
    else:
        pw = pairwise

    rmsh = np.zeros(pw.starCnt, dtype=float)
    rmsz = np.zeros(pw.starCnt, dtype=float)
    rmsx = np.zeros(pw.starCnt, dtype=float)
    rmsy = np.zeros(pw.starCnt, dtype=float)
    rmsx2 = np.zeros(pw.starCnt, dtype=float)
    rmsy2 = np.zeros(pw.starCnt, dtype=float)

    for ii in range(pw.starCnt):
        ee = np.where(pw.hasData[ii] == 1)[0]

        rmsh[ii] = math.sqrt( (pw.diffh[ii,ee]**2).sum() / (len(ee) - 1.0) )
        rmsz[ii] = math.sqrt( (pw.diffz[ii,ee]**2).sum() / (len(ee) - 1.0) )
        rmsx[ii] = math.sqrt( (pw.diffx[ii,ee]**2).sum() / (len(ee) - 1.0) )
        rmsy[ii] = math.sqrt( (pw.diffy[ii,ee]**2).sum() / (len(ee) - 1.0) )
        rmsx2[ii] = pw.posx[ii,ee].std() * 10**3
        rmsy2[ii] = pw.posy[ii,ee].std() * 10**3

    # We only care about those stars that are K<14 and within 4".
    # This helps us avoid issues in the corners and with faint stars.
    idx = np.where((pw.mag < 14) & (pw.r < 4))[0]
    rmsh = rmsh[idx]
    rmsz = rmsz[idx]
    
    binsIn = np.arange(0, 5, 0.2)
    (binsh, datah) = histNofill.hist(binsIn, rmsh)
    (binsz, dataz) = histNofill.hist(binsIn, rmsz)
    (binsx, datax) = histNofill.hist(binsIn, rmsx)
    (binsy, datay) = histNofill.hist(binsIn, rmsy)
    (binsx2, datax2) = histNofill.hist(binsIn, rmsx2)
    (binsy2, datay2) = histNofill.hist(binsIn, rmsy2)

    peakh = binsh[datah.argmax()]
    peakz = binsz[dataz.argmax()]
    peakx = binsx[datax.argmax()]
    peaky = binsy[datay.argmax()]
    peakx2 = binsx2[datax2.argmax()]
    peaky2 = binsy2[datay2.argmax()]

    print('RMS error in separation vector (mas): ')
    print('Horizon: mean = %4.2f  median = %4.2f  peak = %4.2f' % \
          (rmsh.mean(), np.median(rmsh), peakh))
    print('Zenith:  mean = %4.2f  median = %4.2f  peak = %4.2f' % \
          (rmsz.mean(), np.median(rmsz), peakz))
    print('x-axis:  mean = %4.2f  median = %4.2f  peak = %4.2f' % \
          (rmsx.mean(), np.median(rmsx), peakx))
    print('y-axis:  mean = %4.2f  median = %4.2f  peak = %4.2f' % \
          (rmsy.mean(), np.median(rmsy), peaky))
    print('x-axis2: mean = %4.2f  median = %4.2f  peak = %4.2f' % \
          (rmsx2.mean(), np.median(rmsx2), peakx2))
    print('y-axis2: mean = %4.2f  median = %4.2f  peak = %4.2f' % \
          (rmsy2.mean(), np.median(rmsy2), peaky2))
    
    if (plot == True):
        py.clf()
        py.plot(binsh, datah, 'r-')
        py.plot(binsz, dataz, 'b-')
        py.xlabel('RMS Error In Separation (mas)')
        py.ylabel('Number of Stars')
        
        py.legend(('Horizon Median = %4.2f mas' % np.median(rmsh),
                   'Zenith Median  = %4.2f mas' % np.median(rmsz)))
    
        py.savefig(outdir + '/plots/hist_zh_rms.eps')
        py.savefig(outdir + '/plots/hist_zh_rms.png')

    return (np.median(rmsh), np.median(rmsz))


def plotAstErrorvsTime(pairwise, outdir=''):
    """
    Find all the stars between K=11 and K=12 (not saturated) and
    within r < 2.5''. Then plot up positional uncertainty (as
    calculated from the RMS error on the position w.r.t S1-5) as
    a function of the total number of exposures that goes into
    the measurement. 
    """
    if (type(pairwise) == type('')):
        pw = pickle.load(open(pariwise))
    else:
        pw = pairwise


    # Find the stars between K=11 and K=12 and a distance of
    # less than 2.5" from S1-5. Only use stars detected in ALL
    # the epochs.
    idx = np.where((pw.mag >= 11) & (pw.mag < 12) &
                   (pw.r < 2) &
                   (pw.hasData.sum(axis=1) == (pw.epochCnt-1)))[0]    
    starCnt = len(idx)
    print('Number of Stars', starCnt)

    # Trim off the first epoch cuz it was the "main-map".
    # Might as well trim down to just the stars of interest also.
    posx = pw.posx[idx, 1:]
    posy = pw.posy[idx, 1:]
    diffx = pw.diffx[idx, 1:]
    diffy = pw.diffy[idx, 1:]

    imgCnt = len(posx[0, 1:])

    # Exposure time
    texp = 30.0     # 30 sec

    # We will try averaging over N exposures where we choose
    # a bunch of different N values. For each N exposure set,
    # we will compute the Allan variance. Then we will compute
    # the geometric mean of these Allan variances over all the
    # exposures in the set. This last thing is what we keep around.
    # We will do this for several stars.
    #navg = np.arange(3, math.floor(imgCnt/2))
    navg = np.arange(1, imgCnt/3)
    allan1 = np.zeros((starCnt, len(navg)), dtype=float)
    allan2 = np.zeros((starCnt, len(navg)), dtype=float)
    allan3 = np.zeros((starCnt, len(navg)), dtype=float)

    for n in range(len(navg)):
        # Number of exposure sets we will have
        nCnt = navg[n]
        setCnt = imgCnt / nCnt
        avgx = np.zeros((starCnt, setCnt), dtype=float)
        avgy = np.zeros((starCnt, setCnt), dtype=float)

        dx = np.zeros((starCnt, setCnt), dtype=float)
        dy = np.zeros((starCnt, setCnt), dtype=float)

        # For each set, calc the RMS error on the 
        # position for each star.
        for s in range(setCnt):
            startIdx = s*nCnt
            stopIdx = startIdx + nCnt
            #avgx[:,s] = diffx[:, startIdx:stopIdx].sum(axis=1) / nCnt
            #avgy[:,s] = diffy[:, startIdx:stopIdx].sum(axis=1) / nCnt

            avgx[:,s] = posx[:, startIdx:stopIdx].sum(axis=1) / nCnt
            avgy[:,s] = posy[:, startIdx:stopIdx].sum(axis=1) / nCnt

        allan3[:,n] = np.sqrt(avgx.std(axis=1) * avgy.std(axis=1))
        allan3[:,n] = avgx.std(axis=1)

        # Allan variance is:
        # SUM( SQR( x[i] - x[i-1] ) ) / (2 * (n-1))
        #
        # for each star.
        allanvarx = (avgx[:,1:] - avgx[:,0:-1] / nCnt)**2
        allanvary = (avgy[:,1:] - avgy[:,0:-1] / nCnt)**2
        allanvar = np.sqrt(allanvarx.sum(axis=1) * allanvary.sum(axis=1))
        allanvar /= ( 2.0 * (setCnt - 1))
        allan1[:,n] = np.sqrt(allanvar)

        foox = np.zeros(starCnt, dtype=float)
        fooy = np.zeros(starCnt, dtype=float)
        ee = setCnt*nCnt - 2*nCnt
        for ii in range(0, ee):
            for qq in range(0, nCnt-1):
                foox += diffx[:, ii+qq] - diffx[:, ii+qq+nCnt]
                fooy += diffy[:, ii+qq] - diffy[:, ii+qq+nCnt]
            foox = (foox / nCnt)**2
            fooy = (fooy / nCnt)**2
#             iin = ii + nCnt
#             ii2n = ii + 2*nCnt
#             foo += (posx[:, ii] -
#                     posx[:, iin]*2 +
#                     posx[:, ii2n] )**2
#             fooCnt += 1

        foox /= (2.0 * ee)
        foox = np.sqrt(foox)
        fooy /= (2.0 * ee)
        fooy = np.sqrt(fooy)

        allan2[:,n] = np.sqrt(foox * fooy)
        print('%2d  %8.2e  %8.2e  %8.2e' % \
              (nCnt, allan2[0,n]*10**3, allan1[0,n]*10**3, allan3[0,n]*10**3))

    colors = ['orange', 'red', 'blue', 'green',
              'purple', 'cyan', 'magenta', 'purple']
    py.clf()
    for ss in range(1):
        py.semilogy(navg, allan1[ss,:]*10**3, 'k-.', color=colors[ss])
        py.semilogy(navg, allan2[ss,:]*10**3, 'k--', color=colors[ss])
        py.semilogy(navg, allan3[ss,:]*10**3, 'k-', color=colors[ss])
    #py.semilogy(navg, allan1.mean(axis=0)*10**3, 'r-')
    #py.semilogy(navg, allan2.mean(axis=0)*10**3, 'g-')
    #py.semilogy(navg, allan3.mean(axis=0)*10**3, 'k-')
    
    #py.legend([pw.names[ii] for ii in idx])

    # Plot up a line that goes as 1/sqrt(N)
    theory = allan3[0,0] * 10**3 * math.sqrt(navg[0]) / np.sqrt(navg)
    py.loglog(navg, theory, 'k--')

    #py.xlabel('Averaging Time (sec)')
    py.xlabel('Number of Images Averaged Over')
    py.ylabel('Astrometric Error (mas)')

    print(pw.r[idx], pw.mag[idx])
            
    py.savefig(outdir + 'plots/allan_variance.eps')
    py.savefig(outdir + 'plots/allan_variance.png')
