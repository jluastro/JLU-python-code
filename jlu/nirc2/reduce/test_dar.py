from pyraf import iraf
import asciidata, glob, os
import numpy as np
import pylab as py
import math, datetime
import pyfits
from gcwork import objects
import dar

def diffDarOnOff(cleanDir1, cleanDir2):
    files1tmp = glob.glob(cleanDir1 + '/c????.fits')
    files2tmp = glob.glob(cleanDir2 + '/c????.fits')

    for f1 in files1tmp:
        cname1 = f1.split('/')[-1]

        for f2 in files2tmp:
            cname2 = f2.split('/')[-1]

            if (cname1 == cname2):
                outname = cname1.replace('c', 'diff')

                print 'IMARITH: %s - %s = %s' % (cname1, cname2, outname)
                if (os.path.exists(outname)):
                    iraf.imdelete(outname)
                iraf.imarith(f1, '-', f2, outname)
            
        
def plotScalePosangOverNight(alignRoot, imgDir):
    # Read in the list of images used in the alignment
    listFile = open(alignRoot+'.list', 'r')
    
    parang = []
    for line in listFile:
        _data = line.split()

        lisFile = _data[0].split('/')[-1]

        if (lisFile.startswith('mag')):
            continue
                
        fitsFile = imgDir + lisFile.split('_')[0] + '.fits'

        # Get header info
        hdr = pyfits.getheader( fitsFile )
        parang.append( hdr['PARANG'] )

    parang = np.array(parang)
    numEpochs = len(parang)

    # Load scales/angles
    scale = np.zeros(numEpochs, float)
    angle = np.zeros(numEpochs, float)
    sgrax = np.zeros(numEpochs, float)
    sgray = np.zeros(numEpochs, float)
    scaleErr = np.zeros(numEpochs, float)
    angleErr = np.zeros(numEpochs, float)
    sgraxErr = np.zeros(numEpochs, float)
    sgrayErr = np.zeros(numEpochs, float)
    imgPA = np.zeros(numEpochs, float)

    for e in range(numEpochs):
        trans = objects.Transform()
        trans.loadFromAbsolute(root='./', align=alignRoot + '.trans', idx=e+1)
            
        trans.linearToSpherical(silent=1, override=False)
        scale[e] = trans.scale
        angle[e] = math.degrees(trans.angle)

    scale *= 9.96

    py.clf()
    py.subplot(2, 1, 1)
    py.plot(parang, scale, 'k.')
    py.ylabel('Plate Scale (mas/pix)')
    py.xlabel('Parallactic Angle (deg)')
    py.title('Relative Transformation')

    py.subplot(2, 1, 2)
    py.plot(parang, angle, 'k.')
    py.ylabel('Position Angle (deg)')
    py.xlabel('Parallactic Angle (deg)')

    py.savefig('plots/scale_pa_vs_parang.png')


def plotDarCoeffsVsZenith():
    effWave = 2.12 # microns
    utc = datetime.datetime(2008, 6, 15, 0, 0, 0)
    utc2hst = datetime.timedelta(hours=-10)
    hst = utc + utc2hst

    (refA, refB) = dar.keckDARcoeffs(effWave, hst.year, hst.month, hst.day,
                                     hst.hour, hst.minute)

    elevation = np.arange(30.0, 90.0, 1.0)
    tanz = np.tan((90.0 - elevation) * math.pi / 180.0)
    tmp = 1.0 + tanz**2
    darCoeffL = tmp * (refA + 3.0 * refB * tanz**2)
    darCoeffQ = -tmp * (refA*tanz +
                            3.0 * refB * (tanz + 2.0*tanz**3))

    # Convert DAR coefficients for use with arcseconds
    darCoeffL *= 1.0
    darCoeffQ *= 1.0 / 206265.0

    # 1" sep
    linear1 = darCoeffL * 1.0 * 10**3      # in mas
    quadra1 = darCoeffQ * 1.0**2 * 10**3   # in mas

    # 10" sep
    linear2 = darCoeffL * 10.0 * 10**3     # in mas
    quadra2 = darCoeffQ * 10.0**2 * 10**3  # in mas

    # 60" sep
    linear3 = darCoeffL * 60.0 * 10**3     # in mas
    quadra3 = darCoeffQ * 60.0**2 * 10**3  # in mas
    
    print '            Linear(mas)    Quardatic(mas)'
    print '1" sep    %12.7f  %12.7f' % (linear1.mean(), quadra1.mean())
    print '10" sep   %12.7f  %12.7f' % (linear2.mean(), quadra2.mean())
    print '60" sep   %12.7f  %12.7f' % (linear3.mean(), quadra3.mean())

    py.clf()
    py.semilogy(elevation, linear1, 'r-')
    py.semilogy(elevation, -quadra1, 'r--')

    py.semilogy(elevation, linear2, 'b-')
    py.semilogy(elevation, -quadra2, 'b--')

    py.semilogy(elevation, linear3, 'g-')
    py.semilogy(elevation, -quadra3, 'g--')

    py.legend(('1" lin', '1" quad',
               '10" lin', '10" quad', '60" lin', '60" quad'), loc='lower left')

    py.xlabel('Elevation (deg)')
    py.ylabel('Delta-R (mas)')

    py.savefig('dar_linear_vs_quad_terms.png')
    py.savefig('dar_linear_vs_quad_terms.eps')

