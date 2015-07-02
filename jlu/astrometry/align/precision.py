import numpy as np
import pylab as py
import os
from astropy.table import Table

def plotPosError(starlist, raw=False, suffix='', radius=4, magCutOff=15.0,
                 title=True, scale=0.00995, header=False):
    """
    Make three standard figures that show the data quality 
    from a *_rms.lis file. 

    1. astrometric error as a function of magnitude.
    2. photometric error as a function of magnitude.
    3. histogram of number of stars vs. magnitude.

    Use raw=True to plot the individual stars in plots 1 and 2.
    """
    # Load up the starlist
    lis = Table.read(starlist, format='ascii')

    if header == True:
        name = lis['col1']
        mag = lis['col2']
        x = lis['col4']
        y = lis['col5']
        xerr = lis['col6']
        yerr = lis['col7']
        snr = lis['col8']
        corr = lis['col9']
    else:
        name = lis['name']
        mag = lis['mag']
        x = lis['x']
        y = lis['y']
        xerr = lis['xe']
        yerr = lis['ye']
        snr = lis['snr']
        corr = lis['corr']        

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
    magBins = np.arange(8.0, 20.0, magStep)
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
    #py.axis([8, 22, 5e-2, 30.0])
    py.xlabel('K Magnitude for r < %4.1f"' % radius, fontsize=16)
    py.ylabel('Positional Uncertainty (mas)', fontsize=16)
    if title == True:
        py.title(starlist)
    
    py.savefig('plotPosError%s.png' % suffix)

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
    #py.axis([8, 22, 0, 0.15])
    py.xlabel('K Magnitude for r < %4.1f"' % radius)
    py.ylabel('Photo. Uncertainty (mag)')
    py.title(starlist)
    
    py.savefig('plotMagError%s.png' % suffix)

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
    


def spitzerPlotPosError(starlist, raw=True, suffix='', radius=400, magCutOff=11.0,
                 title=False, scale=1.22, header=False):
    """
    Make three standard figures that show the data quality 
    from a *_rms.lis file. 

    1. astrometric error as a function of magnitude.
    2. photometric error as a function of magnitude.
    3. histogram of number of stars vs. magnitude.

    Use raw=True to plot the individual stars in plots 1 and 2.
    """
    # Load up the starlist
    lis = Table.read(starlist, format='ascii')

    if header == True:
        name = lis['col1']
        mag = lis['col2']
        x = lis['col4']
        y = lis['col5']
        xerr = lis['col6']
        yerr = lis['col7']
        snr = lis['col8']
        corr = lis['col9']
    else:
        name = lis['name']
        mag = lis['mag']
        x = lis['x']
        y = lis['y']
        xerr = lis['xe']
        yerr = lis['ye']
        snr = lis['snr']
        corr = lis['corr']        

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
    radStep = 20.0
    magBins = np.arange(7.0, 15.0, magStep)
    radBins = np.arange(10.0, 190.0, radStep)
    
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
    #py.axis([8, 22, 5e-2, 30.0])
    py.xlabel('3.6 Magnitude for r < %4.1f"' % radius, fontsize=16)
    py.ylabel('Positional Uncertainty (mas)', fontsize=16)
    if title == True:
        py.title(starlist)
    
    py.savefig('plotPosError%s.png' % suffix)

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
    #py.axis([8, 22, 0, 0.15])
    py.xlabel('3.6 Magnitude for r < %4.1f"' % radius)
    py.ylabel('Photo. Uncertainty (mag)')
    #py.title(starlist)
    
    py.savefig('plotMagError%s.png' % suffix)

    ##########
    # 
    # Plot histogram of number of stars detected
    #
    ##########
    py.clf()
    idx = (np.where(r < radius))[0]
    (n, bb, pp) = py.hist(mag[idx], bins=np.arange(7, 15, 0.5))
    py.xlabel('3.6 Magnitude for r < %4.1f"' % radius)
    py.ylabel('Number of Stars')

    py.savefig('plotNumStars%s.png' % suffix)

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
    print 'Median Pos Error (mas) for 3.6 mag < %2i, r < %4.1f:  %5.2f' % \
          (magCutOff, radius, errMedian)
    print 'Median Mag Error (mag) for 3.6 mag < %2i, r < %4.1f:  %5.2f' % \
          (magCutOff, radius, np.median(merr[idx]))
    print 'Turnover mag = %4.1f' % (maxBin)


    out = open('plotPosError%s.txt' % suffix, 'w')
    out.write('Number of detections: %4d\n' % len(mag))
    out.write('Median Pos Error (mas) for 3.6 mag < %2i, r < %4.1f:  %5.2f\n' % \
          (magCutOff, radius, errMedian))
    out.write('Median Mag Error (mag) for 3.6 mag < %2i, r < %4.1f:  %5.2f\n' % \
          (magCutOff, radius, np.median(merr[idx])))
    out.write('Turnover mag = %4.1f\n' % (maxBin))
    out.close()
    
