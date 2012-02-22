import pylab as py
import numpy as np
import os
import asciidata
import datetime

def plotRo(year, month, day):
    homedir = os.path.expanduser('~')

    root = homedir + '/work/gc/astrometry/massdimm/TMTmass_dimm/'

    dateSuffix = str(year) + str(month).zfill(2) + str(day).zfill(2)

    dimmfile = root + 'results.TMTDIMM.T6-Hawaii.' + dateSuffix
    massfile = root + 'results.TMTMASS.T6-Hawaii.' + dateSuffix
    proffile = root + 'results.TMTMASS_profile.T6-Hawaii.' + dateSuffix

    dimm = DIMM(dimmfile)

    py.clf()
    py.plot(dimm.r0)

    

class DIMM(object):
    def __init__(self, dimmfile):
        self.file = dimmfile

        table = asciidata.open(dimmfile)

        # Date and time are in UT
        self.year = table[0].tonumpy()
        self.month = table[1].tonumpy()
        self.day = table[2].tonumpy()
    
        self.hour = table[3].tonumpy()
        self.minute = table[4].tonumpy()
        self.second = table[5].tonumpy()

        if '2010' in dimmfile:
            self.seeing = table[6].tonumpy()
            
            # No airmass in new file format
            self.airmass = np.zeros(len(self.hour))

            # Convert from HST to UT
            self.hour += 10
            
            idx = np.where(self.hour > 24)[0]
            self.day[idx] += 1
            self.hour[idx] -= 24
        else:
            self.airmass = table[8].tonumpy()
            self.seeing = table[9].tonumpy()

        self.r0 = 0.98 * 500e-7 * 206265.0 / self.seeing # in cm

        self.timeInHours = self.hour + (self.minute/60.0) + (self.second/3600.0)

    def indexTime(self, hour, minute, second):
        """
        Fetch the closest row of data for a specified time (UT).
        """
        inputTime = hour + (minute/60.0) + (second/3600.0)

        timeDiff = abs(self.timeInHours - inputTime)
        closestIndex = timeDiff.argmin()

        # Make sure we aren't off by more than an hour
        if timeDiff[closestIndex] > 1.0:
            print 'Could not find DIMM data close to ', hour, minute, second
            return None

        return closestIndex

class MASS(object):
    def __init__(self, massfile):
        self.file = massfile

        table = asciidata.open(massfile)
        
        # Date and time are in UT
        self.year = table[0].tonumpy()
        self.month = table[1].tonumpy()
        self.day = table[2].tonumpy()

        self.hour = table[3].tonumpy()
        self.minute = table[4].tonumpy()
        self.second = table[5].tonumpy()

        self.free_seeing = table[6].tonumpy()
        if '2010' in massfile:
            # Values Don't exist
            self.isoplanatic_angle = np.zeros(len(self.hour))
            self.tau0 = np.zeros(len(self.hour))

            # Convert from HST to UT
            self.hour += 10
            
            idx = np.where(self.hour > 24)[0]
            self.day[idx] += 1
            self.hour[idx] -= 24
        else:
            self.isoplanatic_angle = table[18].tonumpy()
            self.tau0 = table[22].tonumpy()  # in milli-sec

        self.timeInHours = self.hour + (self.minute/60.0) + (self.second/3600.0)


    def indexTime(self, hour, minute, second):
        """
        Fetch the closest row of data for a specified time (UT).
        """
        inputTime = hour + (minute/60.0) + (second/3600.0)

        timeDiff = abs(self.timeInHours - inputTime)
        closestIndex = timeDiff.argmin()

        # Make sure we aren't off by more than an hour
        if timeDiff[closestIndex] > 1.0:
            print 'Could not find MASS data close to ', hour, minute, second
            return None

        return closestIndex

def fetch_data(utDate, saveTo='/u/jlu/work/gc/ao_performance/massdimm/'):
    import urllib

    print 'Saving MASS/DIMM data to directory:'
    print saveTo

    urlRoot = 'http://mkwc.ifa.hawaii.edu/current/seeing/'
    
    # Save the MASS file
    massFile = utDate + '.mass.dat'
    url = urlRoot + 'mass/' + massFile
    urllib.urlretrieve(url, saveTo + massFile)

    # Save the DIMM file
    dimmFile = utDate + '.dimm.dat'
    url = urlRoot + 'dimm/' + dimmFile
    urllib.urlretrieve(url, saveTo + dimmFile)

    # Save the MASS profile
    massproFile = utDate + '.masspro.dat'
    url = urlRoot + 'masspro/' + massproFile
    urllib.urlretrieve(url, saveTo + massproFile)
