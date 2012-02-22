import ephem
import numpy as np
import pylab as py

arches = ephem.FixedBody()
arches._ra = ephem.hours('17:45:50.5')
arches._dec = ephem.degrees('-28:49:28')
arches._epoch = 2000
arches.compute()

maser = ephem.FixedBody()
maser._ra = ephem.hours('17:45:51.339')
maser._dec = ephem.degrees('-28:48:6.96')
maser._epoch = 2000
maser.compute()

print ephem.separation(arches, maser), ' degrees'
