import asciidata
import numpy as np
import pylab as py

def avg_mauna_kea_weather(measured=None):
    """
    Plot the distribution of Mauna Kea weather valuse
    (humidity, temperature, pressure) during
    observing hours to estimate whether your particular
    observing conditions were average or not. Use data
    from 2008-2009 in May through August.

    Optional Inputs:
    measured - (def=None) A length-3 tuple containing the measured
               temperature, pressure, and humidity. These values
               will be overplotted on the histograms and the probabilities
               will be calculated of encountering these values to
               within the 1% bin width of the histograms.
    """
    years = [2008, 2009]
    months = np.arange(5, 9) # May through August
    
    logDir = '/u/ghezgroup/code/python/keckdar/'

    atmTemp = np.array([], dtype=float)
    atmHumidity = np.array([], dtype=float)
    atmPressure = np.array([], dtype=float)
    
    for year in years:
        for month in months:
            logFile = logDir + 'cfht-wx.' + str(year) + '.' + \
                str(month).zfill(2) + '.dat'

            atm = asciidata.open(logFile)

            hour = atm[3].tonumpy()
            temp = atm[7].tonumpy()
            humidity = atm[8].tonumpy()
            pressure = atm[9].tonumpy()
            
            # Assume observing hours are 8 pm to 6 am.
            idx = np.where((hour > 20) | (hour < 6))[0]

            atmTemp = np.append(atmTemp, temp[idx]) # Celsius
            atmHumidity = np.append(atmHumidity, humidity[idx]) # percent
            atmPressure = np.append(atmPressure, pressure[idx]) # mbars
            
    py.close(2)
    py.figure(2, figsize=(16,6))
    py.clf()
    py.subplots_adjust(left=0.05, right=0.97, wspace=0.25)

    # ----------
    # Temperature Plot
    # ----------
    py.subplot(1, 3, 1)
    (nT, binsT, patchesT) = py.hist(atmTemp, bins=25, 
                                    normed=1, histtype='step')
    py.xlabel('Temperature (Celsius)')
    py.ylabel('Probability Density')
    py.ylim(0, nT.max()*1.05)

    if measured != None:
        arr = py.Arrow(measured[0], nT.max()*1.05, 0, -nT.max()*0.1)
        py.gca().add_patch(arr)


    # ----------
    # Pressure Plot
    # ----------
    py.subplot(1, 3, 2)
    (nP, binsP, patchesP) = py.hist(atmPressure, bins=25,
                                    normed=1, histtype='step')
    py.xlabel('Pressure (milli-bars)')
    py.ylabel('Probability Density')
    py.ylim(0, nP.max()*1.05)
    py.title('Mauna Kea Weather Conditions in Months %d - %d of %d - %d\n' % \
                 (months[0], months[-1], years[0], years[-1]))

    if measured != None:
        arr = py.Arrow(measured[1], nP.max()*1.05, 0, -nP.max()*0.1)
        py.gca().add_patch(arr)


    # ----------
    # Relative Humdity Plot
    # ----------
    py.subplot(1, 3, 3)
    (nH, binsH, patchesH) = py.hist(atmHumidity, bins=25, range=[0,100],
                                    normed=1, histtype='step')
    py.xlabel('Relative Humidity (%)')
    py.ylabel('Probability Density')
    py.ylim(0, nH.max()*1.05)

    if measured != None:
        arr = py.Arrow(measured[2], nH.max()*1.05, 0, -nH.max()*0.1, width=5)
        py.gca().add_patch(arr)


    # Save the figure
    py.savefig('avg_mauna_kea_weather.png')
    

    # ----------
    # Print out some stats
    # ----------
    if measured != None:
        idx = abs(nT - measured[0]).argmin()
        probTemp = nT[idx] * (binsT[idx+1] - binsT[idx])

        idx = abs(nP - measured[1]).argmin()
        probPressure = nP[idx] * (binsP[idx+1] - binsP[idx])

        idx = abs(nH - measured[2]).argmin()
        probHumidity = nH[idx] * (binsH[idx+1] - binsH[idx])

    print 'Temperature (Celsius)'
    print '   Mean = %.1f +/- %.1f' % (atmTemp.mean(), atmTemp.std())
    print '   Median = %.1f' % (np.median(atmTemp))
    if measured != None:
        print '   Probility of Measured Value = %.2f' % probTemp
    print ''

    print 'Pressure (milli-bars)'
    print '   Mean = %.1f +/- %.1f' % (atmPressure.mean(), atmPressure.std())
    print '   Median = %.1f' % (np.median(atmPressure))
    if measured != None:
        print '   Probility of Measured Value = %.2f' % probPressure
    print ''

    print 'Relative Humidity (%)'
    print '   Mean = %.1f +/- %.1f' % (atmHumidity.mean(), atmHumidity.std())
    print '   Median = %.1f' % (np.median(atmHumidity))
    if measured != None:
        print '   Probility of Measured Value = %.2f' % probHumidity


    
def cfht_weather_data(year, month, day, hour, minute,
                      dir='/u/ghezgroup/code/python/keckdar/'):
    """
    Pull out archived weather information from the CFHT weather
    tower by specifying the year, month, day, hour, and minute. 
    The data files should have been downloaded and then split into
    monthly files using keckdar.splitAtmosphereCFHT().

    All dates and times should be specified in Hawaii Standard Time.
    
    Inputs: 
    All inputs should be arrays (even if they are length=1).

    Outputs:
    Temperature (celsius), 
    Pressure (milli-bars), 
    Humidity (% relative), 
    Wind Speed (km/h), 
    Wind Direction (degrees)
    """

    temperature = np.zeros(len(year), dtype=float)
    pressure = np.zeros(len(year), dtype=float)
    humidity = np.zeros(len(year), dtype=float)
    wind_speed = np.zeros(len(year), dtype=float)
    wind_dir = np.zeros(len(year), dtype=float)


    cfht_file = None

    for ii in range(len(year)):
        cfht_file_new = dir + 'cfht-wx.' + str(year[ii]) + '.' + \
            str(month[ii]).zfill(2) + '.dat'

        if (cfht_file != cfht_file_new):
            cfht_file = cfht_file_new
            cfht = asciidata.open(cfht_file)

            atmYear = cfht[0].tonumpy()
            atmMonth = cfht[1].tonumpy()
            atmDay = cfht[2].tonumpy()
            atmHour = cfht[3].tonumpy()
            atmMin = cfht[4].tonumpy()  # HST times
            atmWindSpeed = cfht[5].tonumpy() # km/h
            atmWindDir = cfht[6].tonumpy() # degrees
            atmTemp = cfht[7].tonumpy() # Celsius
            atmHumidity = cfht[8].tonumpy() # percent
            atmPressure = cfht[9].tonumpy() # mb pressure


        # Find the exact time match for year, month, day, hour
        idx = (np.where((atmDay == day[ii]) & (atmHour == hour[ii])))[0]
    
        if (len(idx) == 0):
            print 'Could not find DAR data for %4d-%2d-%2d %2d:%2d in %s' % \
                (year, month, day, hour, minute, logFile)

        # Find the closest minute
        mdx = abs(atmMin[idx] - minute[ii]).argmin()
        match = idx[ mdx ]

        # Ambient Temperature (Celsius)
        temperature[ii] = atmTemp[match]

        # Pressure at the observer (millibar)
        # Should be around 760.0 millibars
        pressure[ii] = atmPressure[match]

        # Relative humidity (%)
        # Should be around 0.1 %
        humidity[ii] = atmHumidity[match]

        # Wind speed (km/h)
        wind_speed[ii] = atmWindSpeed[match]

        # Wind direction (degrees)
        wind_dir[ii] = atmWindDir[match]

    return temperature, pressure, humidity, wind_speed, wind_dir

def cso_water_column(year, month, day, hour, minute):
    """
    Fetch CSO Tau (225 GHz) measurements and convert into a water
    column (in millimeters) for the specified dates.

    Inputs: 
    All inputs should be single int values. Dates and times should be
    UT.
    """
    import urllib, urllib2
    
    params = {}
    params['tboxbl1'] = str(month)
    params['tboxbl2'] = str(day)
    params['tboxbl3'] = str(year)
    params['tboxbl4'] = str(0)
    params['tboxbl5'] = str(0)
    params['tboxbl6'] = str(24)
    params['tboxbl7'] = str(0)
    params['Submit'] = 'Submit'

    params = urllib.urlencode(params)
    
    url = 'http://ulu.submm.caltech.edu/csotau/2tau.pl'

    request = urllib2.Request(url, params)
    response = urllib2.urlopen(request)
    
    html = response.read()

    # Parse the response. Hopefully the webpage format won't change.
    try:
        startIdx = html.index('<table')
        stopIdx = html.index('</table')
    except ValueError:
        noDataMsg = 'There is no Tau data avaliable for that time period'
        try:
            idx = html.index(noDataMsg)
            print 'No CSO water data for Year=%d Month=%d Day=%d Time=%d:%d' % \
                (year, month, day, hour, minute)
        except ValueError:
            print 'Invalid response from CSO for %d %d %d %d:%d' % \
                (year, month, day, hour, minute)
            print html
        return 0
        

    rows = html[startIdx:stopIdx].split('</tr><tr>')
    # rows 0, 1, -1 are all junk
    rows = rows[2:-1]

    utHour = np.zeros(len(rows), dtype=int)
    utMin = np.zeros(len(rows), dtype=int)
    tau225 = np.zeros(len(rows), dtype=float)

    for rr in range(len(rows)):
        row = rows[rr]
        
        row = row.replace('<td align=center>', '')
        row = row.replace(':', ' ')
        row = row.replace('<td>', ' ')
        fields = row.split()

        if len(fields) == 3:
            utHour[rr] = int(fields[0])
            utMin[rr] = int(fields[1])
            tau225[rr] = float(fields[2])
    
        
    # Trim out crappy columns
    idx = np.where(tau225 != 0)[0]
    utHour = utHour[idx]
    utMin = utMin[idx]
    tau225 = tau225[idx]

    # Now lets find the closest in time. We already know the year, month,
    # and day agree. Just check the hour and minute.
    utTimeInHours = utHour + (utMin/60.0)
    obsTimeInHours = hour + (minute/60.0)

    idx = abs(utTimeInHours - obsTimeInHours).argmin()

    water_column = tau225[idx] * 20.0

    print 'CSO tau_225GHz = %.3f at UT = %d %d %d %d:%d:00 for obs at %d:%d:00' % \
        (tau225[idx], year, month, day, utHour[idx], utMin[idx], hour, minute)

    # Return water column in mm of H2O
    return water_column 

def fetch_massdimm(year, month, day):
    """
    Pass in the UT year, month, and day and this saves three files
    (mass seeing, dimm seeing, mass Cn2 profile) to the current directory.
    """
    import urllib

    urlRoot = 'http://mkwc.ifa.hawaii.edu/current/seeing/'
    
    dateString = '%4d%2s%2s' % (year, str(month).zfill(2), str(day).zfill(2))

    fileMASS = dateString + '.mass.dat'
    fileDIMM = dateString + '.dimm.dat'
    filePROF = dateString + '.masspro.dat'

    urlMASS = urlRoot + 'mass/' + fileMASS
    urlDIMM = urlRoot + 'dimm/' + fileDIMM
    urlPROF = urlRoot + 'masspro/' + filePROF
 
    request = urllib.urlretrieve(urlMASS, fileMASS)
    request = urllib.urlretrieve(urlDIMM, fileDIMM)
    request = urllib.urlretrieve(urlPROF, filePROF)
    
    
