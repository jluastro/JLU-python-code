import os, sys
from numpy import *
import numpy as np
import asciidata
from . import starTables as tabs
from jlu.gc.gcwork import starset
from jlu.gc.gcwork import objects
import sqlite3 as sqlite
import pdb

def youngStarNames():
    """
    Loads list of known young stars from database, which is
    continuously updated (per T. Do).
    """
    dbfile = '/u/ghezgroup/data/gc/database/stars.sqlite'

    # Create a connection to the database file
    connection = sqlite.connect(dbfile)

    # Create a cursor object
    cur = connection.cursor()

    # First look in our stars table
    cur.execute('SELECT * FROM stars')
    yngNames = []
    for row in cur:
        yng = row[2]
        if (yng == 'T'):
            yngNames = np.concatenate([yngNames,[row[0]]])
        
    # Next look in Bartko et al. 2009 table
    # If they are in this table, they are young
    cur.execute('SELECT * FROM bartko2009')
    for row in cur:
        yng = row[1]
        yngNames = np.concatenate([yngNames,[yng]])

    # Next look in Paumard et al. 2006 table
    # If they are in this table, they are young
    cur.execute('SELECT * FROM paumard2006')
    for row in cur:
        yng = row[1]
        yngNames = np.concatenate([yngNames,[yng]])

    return np.unique(yngNames)


def loadYoungStars(root, align='align/align_d_rms_1000_abs_t',
                   fit='polyfit_c/fit', points='points_c/',
                   radiusCut=0.8, relErr=1, verbose=False,
                   withRVonly=False, silent=False, mosaic=False):


    if not os.path.exists(root + align + '.trans'):
        align = 'align/align_d_rms_100_abs_t'
        #align = 'align/align_d_rms1000_t'
        
    # Load up the absolute position of Sgr A* based on S0-2's orbit
    t = objects.Transform()
    t.loadAbsolute()

    # Load up position/velocity information
    s = starset.StarSet(root + align, relErr=relErr, trans=t)
    s.loadPolyfit(root + fit, arcsec=1, silent=silent)
    if mosaic == False:
        s.loadPolyfit(root + fit, arcsec=1, accel=1, silent=silent)

    yng = youngStarNames()
    # Tables in database w/ RV information
    ucla = tabs.UCLAstars()
    bart = tabs.Bartko2009()
    paum = tabs.Paumard2006()

    cc = objects.Constants()

    # Pull out from the set of stars those that are young
    # stars.
    stars = []
    names = [star.name for star in s.stars]

    for name in yng:
	# Find the star in our star lists
	try:
	    idx = names.index(name)
	    star = s.stars[idx]

	    if (star.r2d >= radiusCut):
		stars.append(star)

            # Number of Epochs Detected should be corrected
            # for epochs trimmed out of the *.points files.
            pntsFile = '%s%s%s.points' % (root, points, star.name)
            _pnts = asciidata.open(pntsFile)
            star.velCnt = _pnts.nrows

            pntDate = _pnts[0].tonumpy()
            pntX = _pnts[1].tonumpy()
            pntY = _pnts[2].tonumpy()
            pntXe = _pnts[3].tonumpy()
            pntYe = _pnts[4].tonumpy()

            # Load up data from the points files.
            for ee in range(len(star.years)):
                tt = (where(abs(pntDate - star.years[ee]) < 0.001))[0]

                if (len(tt) == 0):
                    star.e[ee].x_pnt = -1000.0
                    star.e[ee].y_pnt = -1000.0
                    star.e[ee].xe_pnt = -1000.0
                    star.e[ee].ye_pnt = -1000.0
                else:
                    tt = tt[0]
                    star.e[ee].x_pnt = pntX[tt]
                    star.e[ee].y_pnt = pntY[tt]
                    star.e[ee].xe_pnt = pntXe[tt]
                    star.e[ee].ye_pnt = pntYe[tt]

            if mosaic == True: # We only have 3 epochs (as of 2011)
	        star.fitXv.v *= cc.asy_to_kms
	        star.fitYv.v *= cc.asy_to_kms
	        star.fitXv.verr *= cc.asy_to_kms
	        star.fitYv.verr *= cc.asy_to_kms

	        star.vx = star.fitXv.v
	        star.vy = star.fitYv.v
	        star.vxerr = star.fitXv.verr
	        star.vyerr = star.fitYv.verr
            else:
	        star.fitXa.v *= cc.asy_to_kms
	        star.fitYa.v *= cc.asy_to_kms
	        star.fitXa.verr *= cc.asy_to_kms
	        star.fitYa.verr *= cc.asy_to_kms

	        star.vx = star.fitXa.v
	        star.vy = star.fitYa.v
	        star.vxerr = star.fitXa.verr
	        star.vyerr = star.fitYa.verr
                

            star.rv_ref = 'None'

            def other_RV_tables():
                # If not found in UCLA tables, then check Bartko+2009
                idx = np.where(bart.ourName == name)[0]
                if len(idx) > 0:
                    star.vz = bart.vz[idx][0] 
	            star.vzerr = bart.vzerr[idx][0]
	            star.vzt0 = bart.t0_spectra[idx][0] 
                    star.rv_ref = 'Bartko+2009'
                else:
                    # Next check Paumard+2006
                    idx = np.where(paum.ourName == name)[0]
                    if len(idx) > 0:
                        star.vz = paum.vz[idx][0] 
                        star.vzerr = paum.vzerr[idx][0]
                        star.vzt0 = paum.t0_spectra[idx][0]
                        star.altName = paum.name[idx][0]
                        star.rv_ref = 'Paumard+2006'
                    #else:
                    #    print 'Could not find radial velocity for %s' % name

	    # Find the radial velocity for each star
            # First look in OSIRIS data, then Bartko, then Paumard
            idx = np.where(ucla.ourName == name)[0]
            if len(idx) > 0:
	        star.vz = ucla.vz[idx][0]
	        star.vzerr = ucla.vzerr[idx][0]
	        star.vzt0 = ucla.t0_spectra[idx][0]
                star.rv_ref = 'UCLA'

                # A star could be in the stars table but still not have vz
                if star.vz == None: # then check other tables
                    star.rv_ref = None
                    other_RV_tables()
            else:
                other_RV_tables()

            if withRVonly == True:
                if star.rv_ref == None:
                    # remove this star
                    stars.remove(star)

            if (verbose == True):
                print('Matched %15s to %12s' % (name, star.rv_ref))

	    star.jz = (star.x * star.vy) - (star.y * star.vx)
	    star.jz /= (star.r2d * hypot(star.vx, star.vy))	    
	except ValueError as e:
	    # Couldn't find the star in our lists
	    continue

    # Set the starset's star list
    s.stars = stars

    print('Found %d young stars' % len(stars))
    return s
    

def loadAllYoungStars(root, radiusCut=0.8, withRVonly=False):
    cc = objects.Constants()
    
    # Use our data if available. Otherwise use Paumards.
    ours = loadYoungStars(root, radiusCut=radiusCut, withRVonly=withRVonly)

    # Now get all other young star info from Paumard2006
    paum = tabs.Paumard2006()

    # Paumard doesn't have positional uncertainties so we
    # need to figure out what to use. Do this by comparing
    # to stars that are in both.
    ourNames = ours.getArray('name')

    for ii in range(len(ourNames)):
        star = ours.stars[ii]

	# Find the star in our star lists
	try:
	    #idx = paum.ourName.index(ourNames[ii])
	    idx = np.where(paum.ourName == ourNames[ii])[0]
            
            star.paumDiffX = paum.x[idx] - star.fitXa.p
            star.paumDiffY = paum.y[idx] - star.fitYa.p
            star.paumDiff = sqrt(star.paumDiffX**2 + star.paumDiffY**2)
	except ValueError as e:
            continue
    
    # We now have young stars that are not in Paumard, so must account
    # for this
    diff = ours.getArray('paumDiff')
    mtch = np.where(diff > 0)[0] # This will find all but the NaN's
    avgDiff = diff[mtch].mean()
    print('The average error in Paumard distances is', avgDiff)

    # Loop through all the young stars in Paumard's list.
    for ii in range(len(paum.name)):
        name = paum.ourName[ii]
	# Find the star in our star lists
	try:
	    #idx = ourNames.index(name)
	    idx = np.where(ourNames == name)[0]
	    star = ours.stars[idx]
	except ValueError as e:
	    # Couldn't find the star in our lists. Use Paumard info.
            if (name == ' ' or name == ''):
                name = 'paum' + paum.name[ii]

            # Only keep if there is 3D velocity
            if (paum.vx[ii] == 0 and paum.vy[ii] == 0):
                continue

            if (sqrt(paum.x[ii]**2 + paum.y[ii]**2) < radiusCut):
                continue
                
	    star = objects.Star(name)
	    ours.stars.append(star)

            star.x = paum.x[ii]
            star.y = paum.y[ii]
            star.vx = paum.vx[ii]
            star.vy = paum.vy[ii]
            star.vz = paum.vz[ii]

            star.mag = paum.mag[ii]

            star.xerr = avgDiff
            star.yerr = avgDiff
            star.vxerr = paum.vxerr[ii]
            star.vyerr = paum.vyerr[ii]
            star.vzerr = paum.vzerr[ii]

            star.velCnt = 0

            # Proper motions need to be fixed for a different distance.
            # Paumard assumed 8kpc.
            star.vx *= cc.dist / 8000.0
            star.vy *= cc.dist / 8000.0
            star.vxerr *= cc.dist / 8000.0
            star.vyerr *= cc.dist / 8000.0

            # Need to set the positional errors to something.

            # We don't know the epoch of the Paumard measurements:
            t0 = 2005.495
            star.setFitXv(t0, star.x, star.xerr,
                          star.vx/cc.asy_to_kms, star.vxerr/cc.asy_to_kms)
            star.setFitYv(t0, star.y, star.yerr,
                          star.vy/cc.asy_to_kms, star.vyerr/cc.asy_to_kms)

	    star.jz = (star.x * star.vy) - (star.y * star.vx)
	    star.jz /= (star.r2d * hypot(star.vx, star.vy))
            
    # Return the starlist 
    return ours


def idNewYoung(alignRoot):
    s = starset.StarSet(alignRoot)
    
    young_new_dat = '/u/ghezgroup/data/gc/source_list/young_new.dat'
    yng = youngStarNames(datfile=young_new_dat)

    ourName = s.getArray('name')
    x = s.getArray('x')
    y = s.getArray('y')
    mag = s.getArray('mag')

    # Load up Paumard
    paum = tabs.Paumard2006()

    for i in range(len(paum.name)):
        dx = x - paum.x[i]
        dy = y - paum.y[i]
        dm = mag - paum.mag[i]

        dr = sqrt(dx**2 + dy**2)

        idx = (where((dr < 0.5) & (abs(dm) < 0.5)))[0]

        if (len(idx) > 0):
            print('Possible matches for:')
            print('  %-14s  %4.1f  %7.3f  %7.3f (%s)' % \
                  (paum.name[i], paum.mag[i], paum.x[i], paum.y[i],
                   paum.ourName[i]))
            for k in idx:
                print('  %-14s  %4.1f  %7.3f  %7.3f   %5.2f  %5.2f  %3.1f' % \
                      (ourName[k], mag[k], x[k], y[k], dx[k], dy[k], dm[k]))

        else:
            print('No match for: %s' % (paum.name[i]))


def makeYoungDat():
    """
    Loads list of known young stars from database, which is
    continuously updated (per T. Do).
    """
    dbfile = '/u/ghezgroup/data/gc/database/stars.sqlite'

    # Create a connection to the database file
    connection = sqlite.connect(dbfile)

    # Create a cursor object
    cur = connection.cursor()
    
    # Output File
    _out = open('/u/ghezgroup/data/gc/source_list/young_new.dat', 'w')

    _out.write('# List of Young Stars\n')
    _out.write('# -- created with gcwork.youngNames.makeYoungDat()\n')

#     cur.execute('SELECT ref_id,author,year FROM references')
    _out.write('# References:\n')
#     for row in cur:
#         _out.write('#  %s = %s (%s)\n' % (row[0], row[1], row[2]))

    _out.write('#  %s = %s (%s)\n' % ('1', 'Paumard et al.', '2006'))
    _out.write('#  %s = %s (%s)\n' % ('2', 'Do et al.', '2009'))
    _out.write('#  %s = %s (%s)\n' % ('3', 'Ghez et al.', '2008'))
    _out.write('#  %s = %s (%s)\n' % ('4', 'Genzel et al.', '2000'))
    _out.write('#  %s = %s (%s)\n' % ('5', 'Gillessen et al.', '2009'))

    _out.write('#%-14s  %-4s  %s\n' % ('OurName', 'Ref', 'AltNames'))

    # Write out the young stars.
    cur.execute('SELECT name,young,type_ref FROM stars')

    for row in cur:
        name = row[0]
        isYoung = row[1]
        reference = row[2]
        if (isYoung == 'T'):
            _out.write('%-15s  %-4s  %s\n' % (name, reference, '-'))
            
    _out.close()
