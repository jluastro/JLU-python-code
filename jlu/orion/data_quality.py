import asciidata
import numpy as np
import pylab as py

def strehl_2010oct():
    """
    Calculate the mean and standard deviation of the Strehl and 
    FWHM for the observations taken in 2010 October.
    """

    dataDir = '/u/jlu/data/orion/2010oct/clean/'
    plotDir = '/u/jlu/work/orion/bn/data_quality/plots/'
    fields = ['orion_t_kp', 'orion_t_lp', 'orion_t_ms',
              'orion_wide_kp', 'theta1A_ms', 'theta1B_ms']

    for ff in range(len(fields)):
        strehlFile = dataDir + fields[ff] + '/irs33N.strehl'

        _data = asciidata.open(strehlFile)
        
        strehl = _data[1].tonumpy()
        fwhm = _data[3].tonumpy()

        print 'Data Quality Info from %s in irs33N.strehl file' % fields[ff]
        print '   Strehl = %6.2f +/- %5.2f' % (strehl.mean(), strehl.std())
        print '     FWHM = %6.2f +/- %5.2f mas' % (fwhm.mean(), fwhm.std())

        py.clf()
        py.hist(strehl)
        py.xlabel('Strehl')
        py.ylabel('Number of Images')
        py.xlim(0, 1)
        py.savefig(plotDir + 'strehl_' + fields[ff] + '.png')

        py.clf()
        py.hist(fwhm)
        py.xlabel('FWHM (mas)')
        py.ylabel('Number of Images')
        py.xlim(40, 200)
        py.savefig(plotDir + 'fwhm_' + fields[ff] + '.png')
