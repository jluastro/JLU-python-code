from jlu.imf import cimf
from jlu.imf import imf
from datetime import datetime
import numpy as np
import pylab as py

def time_test_cimf(massLimits, imfSlopes, nstars):
    # Turn the variables into a string
    pin = 'basic\n\n'
    pin += 'breakpoint '
    for mm in massLimits:
        pin += ' {0:.2f}'.format(mm)
    pin += '\n\n'
    for seg in imfSlopes:
        pin += 'segment\n'
        pin += 'type powerlaw\n'
        pin += 'alpha {0:.2f}\n'.format(seg)
    
    tstart = datetime.now()
    imfInst = cimf.IMF(pin)
    m = imfInst.draw_stars(int(nstars))
    tend = datetime.now()

    print 'N stars = {0:d}  Mass = {1:.0f} Msun'.format(len(m), m.sum())

    dt = tend - tstart
    print dt
    
    return m

def time_test_imf(massLimits, imfSlopes, totalMass):
    tstart = datetime.now()
    res = imf.sample_imf(massLimits, imfSlopes, totalMass, makeMultiples=False)
    m = res[0]
    tend = datetime.now()

    print 'N stars = {0:d}  Mass = {1:.0f} Msun'.format(len(m), np.sum(m))

    dt = tend - tstart
    print dt

    return m

def compare():
    print '####################'
    print 'Salpeter Comparison'
    print '####################'
    
    massLimits = np.array([1.0, 120.])
    imfSlopes = np.array([-2.3])
    totalMass = 1e7
    nstars = 3.023e6

    print 'Mass Limits: ', massLimits
    print 'IMF Slopes: ', imfSlopes
    print ''

    print 'IMF in Python'
    m1 = time_test_imf(massLimits, imfSlopes, totalMass)
    print 'IMF in C++'
    m2 = time_test_cimf(massLimits, imfSlopes, nstars)

    log_m1 = np.log10(m1)
    log_m2 = np.log10(m2)

    bins = np.arange(np.log10(1), np.log10(120), 0.01)
    py.clf()
    (n1, b1, p1) = py.hist(log_m1, bins, color='red', 
                           histtype='step', log=True)
    (n2, b2, p2) = py.hist(log_m2, bins, color='blue', 
                           histtype='step', log=True)

    minN = np.min([n1, n2])
    maxN = np.max([n1, n2])

    py.ylim(minN, maxN)
    

    print 
    print '####################'
    print 'Weidner and Kroupa 2004'
    print '####################'
    massLimits = np.array([0.01, 0.08, 0.5, 1, 120])
    imfSlopes = np.array([-0.3, -1.3, -2.3, -2.35])
    totalMass = 1e6
    nstars = 2.752e6

    print 'Mass Limits: ', massLimits
    print 'IMF Slopes: ', imfSlopes
    print ''

    print 'IMF in Python'
    m1 = time_test_imf(massLimits, imfSlopes, totalMass)
    print 'IMF in C++'
    m2 = time_test_cimf(massLimits, imfSlopes, nstars)

    log_m1 = np.log10(m1)
    log_m2 = np.log10(m2)

    bins = np.arange(np.log10(0.01), np.log10(120), 0.01)
    py.clf()
    (n1, b1, p1) = py.hist(log_m1, bins, color='red', 
                           histtype='step', log=True)
    (n2, b2, p2) = py.hist(log_m2, bins, color='blue', 
                           histtype='step', log=True)

    minN = np.min([n1, n2])
    maxN = np.max([n1, n2])

    py.ylim(minN, maxN)
    



