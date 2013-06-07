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


def gaussian2d(Xsize, Ysize, Height, XCenter, YCenter, FWHMX, FWHMY):
    """
    2D Gaussian
    INPUTS:   Height  = Amplitude
              XCenter = Mean x
              FWHM    = The Full Width at Half Max = 2.3548*Sigma
    
    RETURNS:  Data    = The Gaussian
    """
    X, Y    = np.mgrid[0:Xsize,0:Ysize]
    SigmaX  = float(FWHMX)/2.3548
    SigmaY  = float(FWHMY)/2.3548
    Gauss2d = lambda x, y: Height*np.exp(-(((XCenter-x)/SigmaX)**2+((YCenter-y)/SigmaY)**2)/2)
    Data = Gauss2d(X,Y)
    return Data

def moments(Data, SubSize=1):
    """
    moments
    INPUTS:    Data    = 2-d Gaussian data
    KEYWORDS   SubSize = int, extract a centered subportion of the array of this size
    RETURNS:   Height  = Amplitude
	    MuX     = Mean X
	    MuY     = Mean Y
        FWHMX   = Full Width at Half Max in X direction
        FWHMY   = Full Width at Half Max in X direction
    """

    if SubSize != 1:
        MidData = (Data.shape)[0]
        Data    = Data[MidData-SubSize:MidData+SubSize, \
                       MidData-SubSize:MidData+SubSize] 

    Total  = Data.sum()
    X, Y   = np.indices(Data.shape)
    MuData = Data.min()
    MuX    = (X*(Data-MuData)).sum()/Total
    MuY    = (Y*(Data-MuData)).sum()/Total
    col    = Data[:, int(MuY)]-MuData
    YWidth = np.sqrt(abs((np.arange(col.size)-MuY)**2*col).sum()/col.sum())
    row    = Data[int(MuX), :]-MuData
    XWidth = np.sqrt(abs((np.arange(row.size)-MuX)**2*row).sum()/row.sum())
    Height = Data.max()
    FWHMX  = 2.3548*XWidth
    FWHMY  = 2.3548*YWidth
    FWHM   = (FWHMX+FWHMY)/2

    #Get Ellipticity
    MidX = X.mean()
    MidY = Y.mean()
    X    = X-MidX
    Y    = Y-MidY
    Ixx  = np.sum((Data-MuData)*X**2)/Total
    Ixy  = np.sum((Data-MuData)*X*Y )/Total
    Iyy  = np.sum((Data-MuData)*Y**2)/Total
    E1   = (Ixx-Iyy)/(Ixx+Iyy)
    E2   = 2*Ixy/(Ixx+Iyy)
    E    = (E1**2+E2**2)**.5
    EA   = np.arctan2(E2, E1)/2.

    return Height, MuX, MuY, FWHMX, FWHMY, FWHM, E, EA
