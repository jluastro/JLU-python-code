import numarray as na
import pylab
import asciidata
from gcwork import objects

def plotRadial():
    # Constants take from Bender et al. (2005)
    cc = objects.Constants()
    m31mass = 1.4e8
    m31dist = 760000.0

    # Construct an array of radii out to 5 arcsec in steps of 0.05''
    r = (na.arange(12 * 5) * 0.05) + 0.05
    r_au = r * m31dist
    r_pc = r_au / cc.au_in_pc
    r_cm = r_au * cc.cm_in_au

    # Determine the theoretical amount for a vs. r
    a_cm_s2 = cc.G * m31mass * cc.msun / r_cm**2
    a_km_s_yr = a_cm_s2 * cc.sec_in_yr / 1.0e5
    a_mas_yr2 = a_cm_s2 * pow(cc.sec_in_yr, 2) * 1000.0 
    a_mas_yr2 /= (cc.cm_in_au * m31dist)

    # Plot circular velocity in both mas/yr and km/s
    v_cm_s = na.sqrt(cc.G * m31mass * cc.msun / r_cm)
    v_km_s = v_cm_s / 1.0e5
    v_mas_yr = v_cm_s * cc.sec_in_yr * 1000.0/ (cc.cm_in_au * m31dist)
    
    masyr_kms = (1.0/1000.0) * m31dist * cc.cm_in_au / (1.0e5 * cc.sec_in_yr)
    masyr2_kmsyr = (1.0/1000.0) * m31dist * cc.cm_in_au 
    masyr2_kmsyr /= 1.0e5 * pow(cc.sec_in_yr, 2)

    ##########
    #
    # Calculate some useful quantities for Keck/TMT
    #
    ##########
    dKeck = 10.0
    dTMT = 10.0

    resKeckK = 1.0e3 * 0.25 * 2.2 / dKeck   # K (GC) on Keck has similar Strehl
    resTMTZ = 1.0e3 * 0.25 * 1.035 / dTMT   # to obs with Z on TMT (mas).

    posErrKeck = 0.15   # mas
    ratioKeck = resKeckK / posErrKeck

    posErrTMT = resTMTZ / ratioKeck
    print 'Estimated positional error for TMT at Z-band: %5.3f' % (posErrTMT)

    # 1 years, 3 sigma
    velLo1 = 3.0 * posErrTMT
    velLoKms1 = velLo1 * masyr_kms

    # 3 years, 3 sigma
    velLo3 = posErrTMT
    velLoKms3 = velLo3 * masyr_kms

    print 'Lowest detectable velocities in:'
    print '\t 1 year, 3 sigma -- low vel = %4.2f mas/yr = %4d km/s' % \
          (velLo1, velLoKms1)
    print '\t 3 year, 3 sigma -- low vel = %4.2f mas/yr = %4d km/s' % \
          (velLo3, velLoKms3)

    ##########
    #
    # Velocity vs. Radius
    #
    ##########
    pylab.figure(2, figsize=(7, 7))
    pylab.clf()

#     pylab.plot(r, v_mas_yr, linewidth=2)
#     pylab.xlabel('Distance from Mbh (arcsec)')
#     pylab.ylabel('Circular Velocity (mas/yr)')
    pylab.plot(r, v_km_s, linewidth=2)
    pylab.xlabel('Distance from Mbh (arcsec)')
    pylab.ylabel('Circular Velocity (km/s)')

    # Detection limit
#     pylab.plot([0, 10], [velLo1, velLo1], 'k--')
#     pylab.plot([0, 10], [velLo3, velLo3], 'k--')
#     pylab.text(2.5, velLo1, '1 year')
#     pylab.text(2.5, velLo3, '3 years')
    pylab.plot([0, 10], [velLoKms1, velLoKms1], 'k--')
    pylab.plot([0, 10], [velLoKms3, velLoKms3], 'k--')
    pylab.plot([0, 10], [30.0, 30.0], 'k--')
    pylab.text(2.5, velLoKms1, '1 year')
    pylab.text(2.5, velLoKms3, '3 years')
    pylab.text(0.3, 35.0, 'Radial Vel.')

    arr1 = pylab.Arrow(2.4, velLoKms1, 0, 0.04*masyr_kms, width=0.09)
    arr3 = pylab.Arrow(2.4, velLoKms3, 0, 0.04*masyr_kms, width=0.09)
    arrRv = pylab.Arrow(0.2, 30.0, 0, 0.04*masyr_kms, width=0.09)
    fig = pylab.gca()
    fig.add_patch(arr1)
    fig.add_patch(arr3)
    fig.add_patch(arrRv)

    str = '0.1 mas/yr = %4d km/s' % (0.1 * masyr_kms)
    pylab.axis([0.0, 3, 0.0, 0.5*masyr_kms])
    pylab.text(1.3, 0.45*masyr_kms, str)
    pylab.savefig('m31theory_vel.png')
    pylab.savefig('m31theory_vel.eps')

    pylab.clf()
    pylab.plot(r, a_mas_yr2)
    pylab.xlabel('Distance from Mbh (arcsec)')
    pylab.ylabel('Acceleration (mas/yr^2)')
    str = '1 mas/yr^2 = %5.2f km/s/yr' % (1.0 * masyr2_kmsyr)
    pylab.text(1.0e-3, 1.5, str)
    pylab.savefig('m31theory_acc.eps')
    pylab.savefig('m31theory_acc.png')

def ttsRadius():
    file = '/u/jlu/work/m31/tts_close.txt'

    tab = asciidata.open(file)

    name = tab[1].tonumarray().tolist()
    ra_hr = tab[7].tonumarray()
    ra_min = tab[8].tonumarray()
    ra_sec = tab[9].tonumarray()
    dec_deg = tab[10].tonumarray()
    dec_min = tab[11].tonumarray()
    dec_sec = tab[12].tonumarray()

    m31_ra_hr = 0.0
    m31_ra_min = 42.0
    m31_ra_sec = 44.23
    m31_dec_deg = 41.0
    m31_dec_min = 16.0
    m31_dec_sec = 8.8

    # Convert into floats
    ra_tmp = ra_hr + (ra_min / 60.0) + (ra_sec / 3600.0)
    dec_tmp = dec_deg + (dec_min / 60.0) + (dec_sec / 3600.0)
    m31_ra_tmp = m31_ra_hr + (m31_ra_min / 60.0) + (m31_ra_sec / 3600.0)
    m31_dec_tmp = m31_dec_deg + (m31_dec_min / 60.0) + (m31_dec_sec / 3600.0)

    ra_diff = (ra_tmp - m31_ra_tmp) * math.cos(math.radians(m31_dec_tmp))
    dec_diff = dec_tmp - m31_dec_tmp

    # Convert to degrees
    ra_diff *= (360.0 / 24.0)

    diff = sqrt(pow(ra_diff, 2) + pow(dec_diff, 2))

    # Convert into arcsec
    ra_diff *= 3600.0
    dec_diff *= 3600.0
    diff *= 3600.0

    idx = diff.argsort()

    for i in idx:
	print '%10s   %2d %2d %4.1f  %2d %2d %4.1f  %4d  %4d %4d' % \
	    (name[i], ra_hr[i], ra_min[i], ra_sec[i], 
	     dec_deg[i], dec_min[i], dec_sec[i],
	     diff[i], ra_diff[i], dec_diff[i])

	if (diff[i] < 65.0):
	    os.system('grep %s tts_clust.txt' % name[i])
	    os.system('grep %s tts_stars.txt' % name[i])
