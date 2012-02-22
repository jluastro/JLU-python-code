import asciidata
import numpy as np
import pylab as py

def plotPosErrorAll():
    """
    Plot all the curves for the GC astrometric errors.
    """
    dataDir = '/u/ghezgroup/data/gc_new/'

    epochs = ['05jullgs', '06maylgs1', '06junlgs',
              '06jullgs', '07maylgs', '07auglgs', '08maylgs1',
              '08jullgs', '09jullgs']
    #'09maylgs'
    #'09seplgs'

    epochCnt = len(epochs)

    magStep = 1.0
    radStep = 1.0
    magBins = np.arange(10.0, 20.0, magStep)
    radBins = np.arange(0.5, 9.5, radStep)
    
    errMag = np.zeros((epochCnt, len(magBins)), float)
    errRad = np.zeros((epochCnt, len(radBins)), float)

    radius = 4.0
    magCutOff = 15.0

    # Assume this is NIRC2 data.
    scale = 0.00995
    
    for e in range(epochCnt):
        listFile = dataDir + epochs[e] + '/combo/starfinder/mag' + \
            epochs[e] + '_kp_rms.lis'
        lis = asciidata.open(listFile)

        mag = lis[1].tonumpy()
        x = lis[3].tonumpy()
        y = lis[4].tonumpy()
        xerr = lis[5].tonumpy()
        yerr = lis[6].tonumpy()

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
        
        ##########
        # Compute errors in magnitude bins
        ########## 
        for mm in range(len(magBins)):
            mMin = magBins[mm] - (magStep / 2.0)
            mMax = magBins[mm] + (magStep / 2.0)
            idx = (np.where((mag >= mMin) & (mag < mMax) & (r < radius)))[0]

            if (len(idx) > 0):
                errMag[e,mm] = np.median(err[idx])
        
        
                       
        ##########
        # Compute errors in radius bins
        ########## 
        for rr in range(len(radBins)):
            rMin = radBins[rr] - (radStep / 2.0)
            rMax = radBins[rr] + (radStep / 2.0)
            idx = (np.where((r >= rMin) & (r < rMax) & (mag < magCutOff)))[0]
            
            if (len(idx) > 0):
                errRad[e,rr] = np.median(err[idx])


        # Print out some summary information
        idx = (np.where((mag < magCutOff) & (r < radius)))[0]
        errMedian = np.median(err[idx])
        print 'Number of detections: %4d' % len(mag)
        print 'Median Pos Error (mas) for K < %2i, r < %4.1f:  %5.2f' % \
              (magCutOff, radius, errMedian)

    ##########
    #
    # Plot astrometry errors
    #
    ##########
    colors = ['red', 'yellow', 'green', 'blue', 
              'cyan', 'magenta', 'orange', 'purple',
              'brown', 'lime', 'plum', 'tan']
    py.clf()
    for e in range(epochCnt):
        py.semilogy(magBins, errMag[e], 'k-', color=colors[e])
    py.axis([10, 18, 3e-2, 3.0])
    py.xlabel('K Magnitude for r < %4.1f"' % radius)
    py.ylabel('Astrometric Uncertatinty (mas)')
    
    py.savefig('plotPosError_v_mag_all.png')


    py.clf()
    for e in range(epochCnt):
        py.semilogy(radBins, errRad[e], 'k-', color=colors[e])
    py.axis([0, 7, 3e-2, 3.0])
    py.xlabel('Radius for K < %2d (arcsec) ' % magCutOff)
    py.ylabel('Astrometric Uncertainty (mag)')
    
    py.savefig('plotPosError_v_rad_all.png')
