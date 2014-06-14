from pyraf import iraf
import ephem
import math

def gcDAR():
    iraf.noao()
    obs = iraf.noao.observatory

    # Set up Keck observatory info
    obs(command="set", obsid="keck")
    keck = ephem.Observer()
    keck.long = math.radians(-obs.longitude)
    keck.lat = math.radians(obs.latitude)
    keck.elev = obs.altitude
    keck.pressure = 800.0
    keck.temp = 278.0
    
    # Date of Observations: 06maylgs1
    keck.date = '2006/05/03 %d:%d:%f' % (obs.timezone+3, 28, 46.33)

    # Set up the galactic center target
    sgra = ephem.FixedBody()
    sgra._ra = ephem.hours("17:45:40")
    sgra._dec = ephem.degrees("-29:00:10")
    sgra._epoch = 2000
    sgra.compute()

    pos2 = ephem.FixedBody()
    pos2._ra = ephem.hours("17:45:40")
    pos2._dec = ephem.degrees("-29:00:00")
    pos2._epoch = 2000
    pos2.compute()

    print 'Date of Obs: ', keck.date
    sgra.compute(keck)
    pos2.compute(keck)

    print 'Azimuth of Objects:  %s  vs.  %s' % (sgra.az, pos2.az)

    for ii in range(15):
	keck.lat = math.radians(obs.latitude - (ii*2))

	sgra.compute(keck)
	pos2.compute(keck)

	angAbs = ephem.separation((sgra.ra, sgra.dec), (pos2.ra, pos2.dec))
	angAbs *= 206265.0
	angRel = ephem.separation((sgra.az, sgra.alt), (pos2.az, pos2.alt))
	angRel *= 206265.0
	angDiff = angAbs - angRel

	print 'Sgr A*:  %s   vs.  %s  deltaR = %5d (muas)' % \
	    (sgra.alt, pos2.alt, angDiff*1e6)
