import asciidata
import numpy as np
import pylab as py
import pyfits
import math
from jlu.util import radialProfile

def plot_trim_fake_count():
    """
    Plot up
    """
    rootDir = '/u/jlu/work/testStarfinder/10_05_12/'
    _file = open(rootDir + 'report_trimmed.txt', 'r')

    images = []
    beforeCuts = []
    afterCuts = []
    cutCount = []

    # These are temporary variables for holding 
    # info on an individual image.
    iBeforeCuts = None
    iAfterCuts = None
    iCutCount = None

    for line in _file:
        if 'Image' in line:
            # Now process the new image.
            fields = line.split()
            fileDirs = fields[2].split('/')
            fileName = fileDirs[-1]

            if 'mag' in fileName:
                # Found a new image, record the old info from 
                # the previous image.
                if iBeforeCuts != None:
                    beforeCuts.append(iBeforeCuts)
                    afterCuts.append(iAfterCuts)
                    cutCount.append(iCutCount)

                isMainMap = True
                images.append(fileName)
            
                iBeforeCuts = []
                iAfterCuts = []
                iCutCount = []
            else:
                isMainMap = False

        else:
            if 'Compiled' in line:
                continue

            if isMainMap == False:
                continue

            fields = line.split()
            if 'Total Number' in line:
                iBeforeCuts.append( int(fields[-1]) )
            if 'Cutting False' in line:
                iCutCount.append( int(fields[-1]) )
            if 'Keeping Real' in line:
                iAfterCuts.append( int(fields[-1]) )

    # Need to add the last data.
    beforeCuts.append(iBeforeCuts)
    afterCuts.append(iAfterCuts)
    cutCount.append(iCutCount)

    # Lets make everything into a numpy array for easy 
    # slicing and dicing.
    images = np.array(images)
    beforeCuts = np.array(beforeCuts)
    afterCuts = np.array(afterCuts)
    cutCount = np.array(cutCount)


    # Now we have all the data. Lets make some plots.
    # Only plot the final numbers. There were 3 rounds of
    # starfinder for each image. Take the 3rd pass results.
    py.clf()
    py.plot(beforeCuts[:,2], cutCount[:,2], 'k.')
    py.xlabel('Original Number of Stars')
    py.ylabel('Number Trimmed')
    py.savefig(rootDir + 'plots/number_trimmed.png')
    
    fracCut = cutCount[:,2] * 1.0 / beforeCuts[:,2]

    py.clf()
    py.plot(beforeCuts[:,2], fracCut, 'k.')
    py.xlabel('Original Number of Stars')
    py.ylabel('Fractional Number Trimmed')
    py.savefig(rootDir + 'plots/fraction_trimmed.png')


    # Print out a table of useful info:
    print '%-30s  %6s  %6s  %6s' % ('Filename', 'Before', 'After', 'Cut')
    for ii in range(len(images)):
        print '%-30s  %6d  %6d  %6d' % \
            (images[ii], beforeCuts[ii,2], afterCuts[ii,2], cutCount[ii,2])

    print '%-30s  %6s  %6s  %6s' % ('----------', '------', '------', '------')
    print '%-30s  %6d  %6d  %6d  (%.2f fractionally)' % \
        ('Median', np.median(beforeCuts[:,2]), np.median(afterCuts[:,2]),
         np.median(cutCount[:,2]), np.median(fracCut))


def plot_trim_fake_envelope():
    """
    Plot up the pairwise delta-mag vs. delta-sep plot and overplot
    the PSF envelope which we have used in trim_fakes.
    """
    rootDir = '/u/jlu/work/testStarfinder/09_06_01/stf_old/'
    origList = rootDir + 'img_old_0.8_stf_cal.lis'
    origPSF = rootDir + 'img_old_psf.fits'

    # ----------
    # Starlist
    # ----------
    # Read in the starlist
    tab = asciidata.open(origList)
    mag = tab[1].tonumpy()
    x = tab[3].tonumpy()
    y = tab[4].tonumpy()

    starCount = tab.nrows

    magMatrix = np.tile(mag, (starCount, 1))
    xMatrix = np.tile(x, (starCount, 1))
    yMatrix = np.tile(y, (starCount, 1))

    # Take only the upper matrix in order to not repeat.
    dm = np.triu(magMatrix.transpose() - magMatrix)
    dx = np.triu(xMatrix.transpose() - xMatrix)
    dy = np.triu(yMatrix.transpose() - yMatrix)
    dr = np.triu(np.hypot(dx, dy))

    # Flatten and take all non-zeros
    idx = np.where(dr.flatten() != 0)[0]
    dm = -np.abs(dm.flatten()[idx])
    dr = dr.flatten()[idx]

    # ----------
    # PSF
    # ----------
    psf2D = pyfits.getdata(origPSF)
    psf = radialProfile.azimuthalAverage(psf2D)
    psf = -2.5 * np.log10(psf[0] / psf)

    # Plot this up
    py.clf()
    py.plot(dr, dm, 'k.')
    py.plot(psf, 'b-')
    py.xlim(0, 100)
    
    
    
