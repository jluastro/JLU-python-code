from pyraf import iraf
import asciidata, glob
import numpy as np
import pylab as py
import math
import pyfits
import datetime
import urllib
import os, sys
import nirc2
import util

module_dir = os.path.dirname(__file__)

# Setup SLALIB python/fortran modules
slalib_dir = module_dir + '/slalib_pyf'
if slalib_dir not in sys.path:
    sys.path.append(slalib_dir)
import refro, refco

def get_atm_conditions(year):
    """
    Retrieve atmospheric conditions from CFHT archive website,
    then calls dar.splitAtmosphereCFHT() to separate the data
    by months.
    """
    yearStr = str(year)

    _atm = urllib.urlopen("http://mkwc.ifa.hawaii.edu/archive/wx/cfht/cfht-wx.%s.dat" % yearStr)
    atm = _atm.read()
    _atm.close()
    
    root = module_dir + '/weather/'
    atmfile = open(root + 'cfht-wx.' + yearStr + '.dat','w')
    atmfile.write(atm)
    atmfile.close()

    splitAtmosphereCFHT(str(year))


def keckDARcoeffs(lamda, year, month, day, hour, minute):
    """
    Calculate the differential atmospheric refraction
    for two objects observed at Keck.

    Input:
    lamda -- Effective wavelength (microns) assumed to be the same for both
    year, month, day, hour, minute of observation (HST)

    Output:
    refA
    refB
    """
    iraf.noao()

    # Set up Keck observatory info
    foo = iraf.noao.observatory(command="set", obsid="keck", Stdout=1)
    obs = iraf.noao.observatory

    ####################
    # Setup all the parameters for the atmospheric refraction
    # calculations. Typical values obtained from the Mauna Kea
    # weather pages and from the web.
    ####################
    # 
    # Temperature Lapse Rate (Kelvin/meter)
    tlr = 0.0065
    # Precision required to terminate the iteration (radian)
    eps = 1.0e-9
    # Height above sea level (meters)
    hm = obs.altitude
    # Latitude of the observer (radian)
    phi = math.radians(obs.latitude)

    # Pull from atmosphere logs.
    logDir = module_dir + '/weather/'
    logFile = logDir +'cfht-wx.'+ str(year) +'.'+ str(month).zfill(2) +'.dat'
    
    _atm = asciidata.open(logFile)
    atmYear = _atm[0].tonumpy()
    atmMonth = _atm[1].tonumpy()
    atmDay = _atm[2].tonumpy()
    atmHour = _atm[3].tonumpy()
    atmMin = _atm[4].tonumpy()  # HST times
    atmTemp = _atm[7].tonumpy() # Celsius
    atmHumidity = _atm[8].tonumpy() # percent
    atmPressure = _atm[9].tonumpy() # mb pressure

    # Find the exact time match for year, month, day, hour
    idx = (np.where((atmYear == year) & (atmMonth == month) &
                    (atmDay == day) & (atmHour == hour)))[0]
    
    if (len(idx) == 0):
        print 'Could not find DAR data for %4d-%2d-%2d %2d:%2d in %s' % \
            (year, month, day, hour, minute, logFile)

    atmMin = atmMin[idx]
    atmTemp = atmTemp[idx]
    atmHumidity = atmHumidity[idx]
    atmPressure = atmPressure[idx]

    # Find the closest minute
    minDiff = abs(atmMin - minute)
    sdx = minDiff.argsort()

    # Select out the closest in time.
    # Ambient Temperature (Kelvin)
    # Should be around 274.0 Kelvin
    tdk = atmTemp[sdx[0]] + 272.15
    # Pressure at the observer (millibar)
    # Should be around 760.0 millibars
    pmb = atmPressure[sdx[0]]
    # Relative humidity (%)
    # Should be around 0.1 %
    rh = atmHumidity[sdx[0]]

    return refco.slrfco(hm, tdk, pmb, rh, lamda, phi, tlr, eps)

def nirc2dar(fitsFile):
    """
    Use the FITS header to extract date, time, wavelength,
    elevation, and image orientation information. This is everything
    that is necessary to calculate the differential atmospheric
    refraction. The differential atmospheric refraction 
    is applicable only along the zenith direction of an image.
    This code calculates the predicted DAR using archived CFHT
    atmospheric data and the elevation and wavelength of the observations.
    Then the DAR correction is transformed into image coefficients that
    can be applied in image coordinates. 
    """
    # Get header info
    hdr = pyfits.getheader(fitsFile)

    effWave = hdr['EFFWAVE']
    elevation = hdr['EL']
    lamda = hdr['CENWAVE']
    airmass = hdr['AIRMASS']
    parang = hdr['PARANG']

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

    (refA, refB) = keckDARcoeffs(effWave, hst.year, hst.month, hst.day,
                                 hst.hour, hst.minute)

    tanz = math.tan(math.radians(90.0 - elevation))
    tmp = 1.0 + tanz**2
    darCoeffL = tmp * (refA + 3.0 * refB * tanz**2)
    darCoeffQ = -tmp * (refA*tanz +
                            3.0 * refB * (tanz + 2.0*tanz**3))

    # Convert DAR coefficients for use with units of NIRC2 pixels
    scale = nirc2.getScale(hdr)
    darCoeffL *= 1.0
    darCoeffQ *= 1.0 * scale / 206265.0
    
    # Lets determine the zenith and horizon unit vectors for
    # this image.
    pa = math.radians(parang + float(hdr['ROTPOSN']) - float(hdr['INSTANGL']))
    zenithX = -math.sin(pa)
    zenithY = math.cos(pa)

    # Compute the predicted differential atmospheric refraction
    # over a 10'' seperation along the zenith direction.
    # Remember coeffecicents are only for deltaZ in pixels
    deltaZ = 10.0
    deltaR = darCoeffL * (deltaZ/scale) + darCoeffQ * (deltaZ/scale)**2
    deltaR *= scale # now in arcseconds

    magnification = (deltaZ + deltaR) / deltaZ
    print 'DAR FITS file = %s' % (fitsFile)
    print 'DAR over 10": Linear dR = %f"  Quad dR = %f"' % \
          (darCoeffL * deltaZ, darCoeffQ * deltaZ**2)
    print 'DAR Magnification = %f' % (magnification)
    print 'DAR Vertical Angle = %6.1f' % (math.degrees(pa))

    return (pa, darCoeffL, darCoeffQ)

def darPlusDistortion(inputFits, outputRoot, xgeoim=None, ygeoim=None):
    """
    Create lookup tables (stored as FITS files) that can be used
    to correct DAR. Optionally, the shifts due to DAR can be added
    to existing NIRC2 distortion lookup tables if the xgeoim/ygeoim
    input parameters are set.

    Inputs:
    inputFits - a NIRC2 image for which to determine the DAR correction
    outputRoot - the root name for the output. This will be used as the
        root name of two new images with names, <outputRoot>_x.fits and 
        <outputRoot>_y.fits.

    Optional Inputs:
    xgeoim/ygeoim - FITS images used in Drizzle distortion correction
        (lookup tables) will be modified to incorporate the DAR correction.
        The order of the correction is 1. distortion, 2. DAR.
        
    """
    # Get the size of the image and the half-points
    hdr = pyfits.getheader(inputFits)
    imgsizeX = float(hdr['NAXIS1'])
    imgsizeY = float(hdr['NAXIS2'])
    halfX = round(imgsizeX / 2.0)
    halfY = round(imgsizeY / 2.0)

    # First get the coefficients
    (pa, darCoeffL, darCoeffQ) = nirc2dar(inputFits)
    #(a, b) = nirc2darPoly(inputFits)

    # Create two 1024 arrays (or read in existing ones) for the
    # X and Y lookup tables
    if ((xgeoim == None) or (xgeoim == '')):
        x = np.zeros((imgsizeY, imgsizeX), dtype=float)
    else:
        x = pyfits.getdata(xgeoim)

    if ((ygeoim == None) or (ygeoim == '')):
        y = np.zeros((imgsizeY, imgsizeX), dtype=float)
    else:
        y = pyfits.getdata(ygeoim)

    # Get proper header info.
    fits = pyfits.open(inputFits)

    axisX = np.arange(imgsizeX, dtype=float) - halfX
    axisY = np.arange(imgsizeY, dtype=float) - halfY
    xcoo2d, ycoo2d = np.meshgrid(axisX, axisY)

    xnew1 = xcoo2d + x
    ynew1 = ycoo2d + y

    # Rotate coordinates clockwise by PA so that zenith is along +ynew2
    # PA = parallactic angle (angle from +y to zenith going CCW)
    sina = math.sin(pa)
    cosa = math.cos(pa)

    xnew2 = xnew1 * cosa + ynew1 * sina
    ynew2 = -xnew1 * sina + ynew1 * cosa

    # Apply DAR correction along the y axis
    xnew3 = xnew2
    ynew3 = ynew2*(1 + darCoeffL) + ynew2*np.abs(ynew2)*darCoeffQ

    # Rotate coordinates counter-clockwise by PA back to original
    xnew4 = xnew3 * cosa - ynew3 * sina
    ynew4 = xnew3 * sina + ynew3 * cosa

    #xnew2 = a[0] + a[1]*xnew1 + a[2]*ynew1 + \
    #        a[3]*xnew1**2 + a[4]*xnew1*ynew1 + a[5]*ynew1**2
    #ynew2 = b[0] + b[1]*xnew1 + b[2]*ynew1 + \
    #        b[3]*xnew1**2 + b[4]*xnew1*ynew1 + b[5]*ynew1**2

    x = xnew4 - xcoo2d
    y = ynew4 - ycoo2d

    xout = outputRoot + '_x.fits'
    yout = outputRoot + '_y.fits'
    util.rmall([xout, yout])
    fits[0].data = x
    fits[0].writeto(xout, output_verify='silentfix')
    fits[0].data = y
    fits[0].writeto(yout, output_verify='silentfix')

    return (xout, yout)


def applyDAR(fits, spaceStarlist, plot=False):
    """
    Input a starlist in x=RA (+x = west) and y=Dec (arcseconds) taken from
    space and introduce differential atmospheric refraction (DAR). The amount
    of DAR that is applied depends on the header information in the input fits
    file. The resulting output starlist should contain what was observed
    after the starlight passed through the atmosphere, but before the
    starlight passed through the telescope. Only achromatic DAR is 
    applied in this code.

    The output file has the name <fitsFile>_acs.lis and is saved to the
    current directory.
    """
    # Get header info
    hdr = pyfits.getheader(fits)

    effWave = hdr['EFFWAVE']
    elevation = hdr['EL']
    lamda = hdr['CENWAVE']
    airmass = hdr['AIRMASS']
    parang = hdr['PARANG']

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

    (refA, refB) = keckDARcoeffs(effWave, hst.year, hst.month, hst.day,
                                 hst.hour, hst.minute)

    tanz = math.tan(math.radians(90.0 - elevation))
    tmp = 1.0 + tanz**2
    darCoeffL = tmp * (refA + 3.0 * refB * tanz**2)
    darCoeffQ = -tmp * (refA*tanz +
                            3.0 * refB * (tanz + 2.0*tanz**3))

    # Convert DAR coefficients for use with arcseconds
    darCoeffL *= 1.0
    darCoeffQ *= 1.0 / 206265.0
    
    # Lets determine the zenith and horizon unit vectors for
    # this image. The angle we need is simply the parallactic
    # (or vertical) angle since ACS images are North Up already.
    pa = math.radians(parang)

    ##########
    #
    # Read in the starlist
    #
    ##########
    _list = asciidata.open(spaceStarlist)
    names = [_list[0][ss].strip() for ss in range(_list.nrows)]
    mag = _list[1].tonumpy()
    date = _list[2].tonumpy()
    x = _list[3].tonumpy() # RA in arcsec
    y = _list[4].tonumpy() # Dec. in arcsec
    xe = _list[5].tonumpy()
    ye = _list[6].tonumpy()
    f1 = _list[7].tonumpy()
    f2 = _list[7].tonumpy()
    f3 = _list[7].tonumpy()
    f4 = _list[7].tonumpy()

    # Magnify everything in the y (zenith) direction. Do it relative to
    # the first star. Even though dR depends on dzObs (ground observed dz),
    # it is a small mistake and results in less than a 10 micro-arcsec
    # change in dR.
    dx = x - x[0]
    dy = y - y[0]

    # Rotate coordinates CW so that the zenith angle is at +ynew
    sina = math.sin(pa)
    cosa = math.cos(pa)
    xnew1 = dx * cosa + dy * sina
    ynew1 = -dx * sina + dy * cosa

    # Apply DAR
    xnew2 = xnew1
    ynew2 = ynew1 * (1.0 - darCoeffL) - ynew1 * np.abs(ynew1) * darCoeffQ

    # Rotate coordinates CCW back to original angle
    xnew3 = xnew2 * cosa - ynew2 * sina
    ynew3 = xnew2 * sina + ynew2 * cosa

    xnew = xnew3 + x[0]
    ynew = ynew3 + y[0]


    ##########
    #
    # Write out the starlist
    #
    ##########
    # Save the current directory
    newFits = fits.replace('.fits', '').split('/')[-1]
    newList = newFits + '_acs.lis'
    print newList
    _new = open(newList, 'w')
    for i in range(len(names)):
        _new.write('%10s  %7.3f  %7.2f  %10.4f  %10.4f  0  0  10  1  1  8\n' % \
              (names[i], mag[i], date[i], xnew[i], ynew[i]))

    _new.close()

    if (plot==True):
        py.clf()
        py.quiver(x, y, xnew - x, ynew - y, scale=0.02)
        py.quiver([0], [0], [0.001], [0], color='r', scale=0.02)
        py.axis([-5, 5, -5, 5])
        py.show()
        


def splitAtmosphereCFHT(year):
    """
    Take an original archive file containing atmospheric parameters and
    split it up into seperate files for individual months. This makes
    later calls to calculate DAR parameters MUCH faster.
    """
    yearStr = str(year)
    logDir = module_dir + '/weather/'
    logFile = logDir + '/cfht-wx.' + yearStr + '.dat'

    _infile = open(logFile, 'r')

    outfiles = []
    for ii in range(1, 12+1):
        monthStr = str(ii).zfill(2)
        _month = open(logDir + '/cfht-wx.' +yearStr+ '.' +monthStr+ '.dat', 'w')
        outfiles.append( _month )

    for line in _infile:
        fields = line.split()

        month = int(fields[1])

        _outfile = outfiles[month-1]
        _outfile.write(line)

    for _month in outfiles:
        _month.close()
        
        
    
def test_darPlusDistortion():
    data_dir = module_dir + '/distortion/'
    file_geox_darunfix = data_dir + 'nirc2dist_xgeoim.fits'
    file_geoy_darunfix = data_dir + 'nirc2dist_ygeoim.fits'

    data_dir = '/u/ghezgroup/data/m92_test/08jul_new_on/'
    file_geox_darfix = data_dir + 'reduce/kp/gc_f1/ce0249geo_x.fits'
    file_geoy_darfix = data_dir + 'reduce/kp/gc_f1/ce0249geo_y.fits'

    xon = pyfits.getdata(file_geox_darfix)
    yon = pyfits.getdata(file_geoy_darfix)
    xoff = pyfits.getdata(file_geox_darunfix)
    yoff = pyfits.getdata(file_geoy_darunfix)

    # Make arrays with the coordinates for each 
    imgsize = 1024
    axisX = np.arange(imgsize, dtype=float)
    axisY = np.arange(imgsize, dtype=float)
    xcoo2d, ycoo2d = np.meshgrid(axisX, axisY)

    # Lets trim so that we only keep every 20th pixel
    idx = np.arange(25, imgsize, 50)
    xon = xon.take(idx, axis=0).take(idx, axis=1)
    yon = yon.take(idx, axis=0).take(idx, axis=1)
    xoff = xoff.take(idx, axis=0).take(idx, axis=1)
    yoff = yoff.take(idx, axis=0).take(idx, axis=1)
    xcoo2d = xcoo2d.take(idx, axis=0).take(idx, axis=1)
    ycoo2d = ycoo2d.take(idx, axis=0).take(idx, axis=1)

    # Calculate differences
    xdiff = xon - xoff
    ydiff = yon - yoff

    # Make vector plots
    py.clf()
    qvr = py.quiver2([xcoo2d], [ycoo2d], [xdiff], [ydiff],
                     units='width', scale=5, 
                     width=0.005, headwidth=3, headlength=3, 
                     headaxislength=3)
    py.quiverkey(qvr, 100, 1120, 1.0, '1 pixel', coordinates='data', color='r')
    py.xlabel('NIRC2 X (pixel)')
    py.ylabel('NIRC2 Y (pixel)')
    py.title('Arrows point to DAR Fix')
    #py.savefig('plots/vector_daroffon.png')
    py.show()


