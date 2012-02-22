import numpy as np
import pylab as py


def plotRadialProfile(psfImage, xcenter=None, ycenter=None, rbin=1):
    """
    Calculate and plot the radial profile. If the X and Y centers
    are not given, then it calculates from the center of the image.
    """

    xsize = psfImage.shape[1]
    ysize = psfImage.shape[0]

    xhalf = xsize / 2.0
    yhalf = ysize / 2.0
    
    x, y = np.indices((ysize, xsize), dtype=float)

    r = np.hypot(x - xhalf, y - yhalf)

    # Sort by radius
    ridx = np.argsort(r.flat)

    # Flatten images and sort by radius
    rFlat = r.flat[ridx]
    imFlat = psfImage.flat[ridx]

    radialBins = np.arange(0, r.max(), rbin) + (rbin / 2.0) # bins center point
    radialProfile = np.zeros(len(radialBins), dtype=float)

    for rr in range(len(radialBins)):
        rmin = radialBins[rr] - (rbin/2.0)
        rmax = radialBins[rr] + (rbin/2.0)

        idx = np.where((rFlat > rmin) & (rFlat <= rmax))

        radialProfile[rr] = imFlat[idx].mean()

    py.clf()
    py.semilogy(radialBins, radialProfile)

    return radialProfile
