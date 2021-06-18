import datetime
import time
import numpy as np 

def toYearFraction(date):
    """
    Given a date time object, calculate the decimal form of the year.
    """
    def sinceEpoch(date): # returns seconds since epoch
        return time.mktime(date.timetuple())
    s = sinceEpoch

    year = date.year
    startOfThisYear = datetime.date(year=year, month=1, day=1)
    startOfNextYear = datetime.date(year=year+1, month=1, day=1)

    yearElapsed = s(date) - s(startOfThisYear)
    yearDuration = s(startOfNextYear) - s(startOfThisYear)
    fraction = yearElapsed/yearDuration

    return date.year + fraction

def toYYYYMMDD(yearfraction):
    if np.isnan(yearfraction):
        return 'YYYY-MM-DD'
    else:
        year = int(yearfraction)
        rem = yearfraction - year
        
        base = datetime.datetime(year, 1, 1)
        result = base + datetime.timedelta(seconds=(base.replace(year=base.year + 1) - base).total_seconds() * rem)
        yyyy = result.year
        mm = result.month
        dd = result.day
        yyyymmdd = '{0:0>4d}-{1:0>2d}-{2:0>2d}'.format(yyyy, mm, dd)
    
        return yyyymmdd
