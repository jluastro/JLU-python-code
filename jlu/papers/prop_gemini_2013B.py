import pylab as py
import numpy as np
from gcwork import starset
import math

def allan_variance_clusters():
    alignDir = '/u/jlu/data/gsaoi/commission/reduce/ngc1815/align_G2_2_nodith/'
    alignRoot = 'align_4_t'

    allan_variance(alignDir, alignRoot)

def allan_variance_bdwarfs():
    alignDir = '/u/jlu/data/gsaoi/commission/reduce/ngc1815/align_G2_2_nodith/'
    alignRoot = 'align_2_t'

    allan_variance(alignDir, alignRoot)
    

def allan_variance(alignDir, alignRoot):
    s = starset.StarSet(alignDir + alignRoot)
    ndet = s.getArray('velCnt')

    # Trim down to only those stars detected in all the epochs.
    idx = np.where(ndet == ndet.max())[0]
    newstars = [s.stars[i] for i in idx]
    s.stars = newstars

    x = s.getArrayFromAllEpochs('x')
    y = s.getArrayFromAllEpochs('y')
    m = s.getArrayFromAllEpochs('mag')

    x = x[1:,:]
    y = y[1:,:]
    m = m[1:,:]

    name = s.getArray('name')
    ndet = s.getArray('velCnt')

    rmsErrCut = None
    if rmsErrCut != None:
        # Read in the stats file and get rid of bad epochs (based on the RMS error).
        t = atpy.Table(alignDir + alignRoot + '.stats', type='ascii')
        epochErr = t.RMS_weighted[1:]
        idx = np.where(epochErr < 3.2)[0]

        x = x[idx,:]
        y = y[idx,:]
        m = m[idx,:]

    nepoch = x.shape[0]
    nstars = x.shape[1]
    print 'N_epochs = {0}, N_stars = {1}'.format(nepoch, nstars)
    
    # Calculate the Allan variance.
    #navg = np.array([2,3,4,5,6,7,8,9,10,11,12,13,15,17,20,29])  # for clusters
    navg = np.array([2,3,4,5,6,7,8,9,10,11,12,13,15,17,20])  # for bdwarfs
    # navg = np.arange(2, nepoch/2)
    allan1 = np.zeros((len(navg), nstars), dtype=float)
    allan3 = np.zeros((len(navg), nstars), dtype=float)

    for n in range(len(navg)):
        # Number of exposure sets we will have
        nCnt = navg[n]
        setCnt = nepoch / nCnt
        avgx = np.zeros((setCnt, nstars), dtype=float)
        avgy = np.zeros((setCnt, nstars), dtype=float)
            
        # For each set, calc the RMS error on the 
        # position for each star.
        for s in range(setCnt):
            startIdx = s*nCnt
            stopIdx = startIdx + nCnt

            avgx[s,:] = x[startIdx:stopIdx, :].sum(axis=0) / nCnt
            avgy[s,:] = y[startIdx:stopIdx, :].sum(axis=0) / nCnt
            
            allan3[n,:] = np.sqrt(avgx.std(axis=0) * avgy.std(axis=0))
            # allan3[n,:] = avgx.std(axis=0)

            # Allan variance is:
            # SUM( SQR( x[i] - x[i-1] ) ) / (2 * (n-1))
            #
            # for each star.
            allanvarx = (avgx[1:,:] - avgx[0:-1,:])**2
            allanvary = (avgy[1:,:] - avgy[0:-1,:])**2
            allanvarx = allanvarx.sum(axis=0) / (2.0 * (setCnt-1))
            allanvary = allanvary.sum(axis=0) / (2.0 * (setCnt-1))
            allanvar = np.sqrt(allanvarx * allanvary)
            allan1[n,:] = np.sqrt(allanvar)


    time = navg * 5.5 / 60.0  # integration time in seconds
    
    ##########
    #
    # Plot the RMS error
    #
    ##########
    outDir = '/u/jlu/doc/proposals/gemini/2013B/'

    py.figure(1)
    py.clf()

    allan3mean = allan3.mean(axis=1) * 10**3
    py.semilogy(time, allan3mean, 'r.-', label='Obs')
        
    # Plot up a line that goes as 1/sqrt(N)
    theoryY0 = allan3mean[6]
    theory =  theoryY0 * math.sqrt(time[6]) / np.sqrt(time)
    py.loglog(time, theory, 'k-', label='Theory')

    py.legend()
    py.xlabel('Combined Integration Time (min)')
    py.ylabel('Astrometric Error (mas)')
    py.ylim(0.1, 10.0)
    py.savefig(outDir + 'allan_variance_all_rms_' + alignRoot + '.eps')
    py.savefig(outDir + 'allan_variance_all_rms_' + alignRoot + '.png')


    ##########
    #
    # Plot the true Allan Variance
    #
    ##########
    py.figure(2)
    py.clf()
    allan1mean = allan1.mean(axis=1) * 10**3
    py.loglog(time, allan1mean, 'r.-', label='Obs', linewidth=2)

    # Plot up a line that goes as 1/sqrt(N)
    theoryTime = np.arange(0.1, 10, 0.1)
    theoryY0 = allan1mean[5]
    theory =  theoryY0 * math.sqrt(time[5]) / np.sqrt(theoryTime)
    py.loglog(theoryTime, theory, 'k--', label='Theory', linewidth=2)

    py.legend()
    py.xlabel('Total Integration Time (min)')
    py.ylabel('Astrometric Precision (mas)')
    py.ylim(0.1, 10.0)
    py.savefig(outDir + 'allan_variance_all_' + alignRoot + '.eps')
    py.savefig(outDir + 'allan_variance_all_' + alignRoot + '.png')

