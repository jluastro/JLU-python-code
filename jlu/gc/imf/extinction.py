import numpy as np
import pylab as py
import pyfits
import math
from sqlite3 import dbapi2 as sqlite
import pdb
from gcwork import starTables
import ds9
from matplotlib import nxutils

def calc_osiris_field_vertices():
    """
    Input a ds9 region file and calculate the vertices from
    all the box regions.
    """
    regionFile = '/u/jlu/work/gc/imf/gcows/mag06maylgs1_wide_kp_GCOWS4.reg'

    # This region file is overlayed on top of the 06maylgs1 deep mosaic.
    fitsFile = '/u/jlu/work/gc/imf/gcows/mag06maylgs1_dp_msc_kp.fits'

    # We need to get a reference coordinate for this file. Use the *.coo file
    # which contains the coordinates of irs16C.
    cooFile = '/u/jlu/work/gc/imf/gcows/mag06maylgs1_dp_msc_kp.coo'
    coo = open(cooFile).readline().split()
    pix16C = [float(coo[0]), float(coo[1])]
    
    labels = starTables.Labels()
    idx = np.where(labels.name == 'irs16C')[0]
    asec16C = [labels.x[idx[0]], labels.y[idx[0]]]
    print 'Identified 16C at '
    print '   pixel position =  [%7.2f, %7.2f]' % (pix16C[0], pix16C[1])
    print '   arcsec positino = [%7.3f, %7.3f]' % (asec16C[0], asec16C[1])
    
    scale = 0.00995

    # Now read the region file and pull out all the box regions
    _reg = open(regionFile, 'r')
    
    ii = 0

    xvertsAll = []
    yvertsAll = []
    namesAll = []

    py.clf()
    for line in _reg:
        if not line.startswith('box'):
            continue

        startBox = line.index('(') + 1
        stopBox = line.index(')')

        boxNumbers = line[startBox:stopBox].split(',')
        
        x = float(boxNumbers[0])
        y = float(boxNumbers[1])
        width = float(boxNumbers[2])
        height = float(boxNumbers[3])
        angle = float(boxNumbers[4])

        # Convert into vertices (corners)
        dxHalf = width / 2.0
        dyHalf = height / 2.0

        # Box centered on origin, not yet rotated.
        dx_tmp = np.array([-dxHalf, -dxHalf, dxHalf, dxHalf])
        dy_tmp = np.array([-dyHalf, dyHalf, dyHalf, -dyHalf])

        # Rotate box
        sina = math.sin(math.radians(-angle))
        cosa = math.cos(math.radians(-angle))

        xverts = dx_tmp * cosa + dy_tmp * sina
        yverts = dy_tmp * cosa - dx_tmp * sina

        xverts += x
        yverts += y

        # Now convert the coordinates into absolute coordinates
        xverts = (xverts - pix16C[0]) * scale * -1.0
        yverts = (yverts - pix16C[1]) * scale
        xverts += asec16C[0]
        yverts += asec16C[1]

        py.plot(np.append(xverts, xverts[0]), 
                np.append(yverts, yverts[0]), 
                color='red')
        rng = py.axis('equal')
        py.xlim(rng[1], rng[0])

        xvertsAll.append(xverts)
        yvertsAll.append(yverts)

    # Now loop through again and highlight each box one at a time
    # and print out the coordinates. Also ask if we want to load it 
    # into the database.
    dbfile = '/u/jlu/data/gc/database/stars.sqlite'

    connection = sqlite.connect(dbfile)
    cur = connection.cursor()

    for ii in range(len(xvertsAll)):
        py.plot(np.append(xvertsAll[ii], xvertsAll[ii][0]), 
                np.append(yvertsAll[ii], yvertsAll[ii][0]), 
                color='blue')
        
        xstr = ','.join([str('%.3f' % xx) for xx in xvertsAll[ii]]) 
        ystr = ','.join([str('%.3f' % yy) for yy in yvertsAll[ii]]) 
        
        # Print out the vertices information for loading into the
        # database.
        print ''
        print 'Box #%d' % ii
        print xstr
        print ystr
        print ''
        print 'Specify the field name in order to enter these vertices'
        print 'into the database. Or just hit return to skip entering it.'

        foo = raw_input('   Field Name: ')

        if len(foo) > 0:
            tuple = (xstr, ystr, foo)

            sql = "UPDATE fields SET x_vertices=?, y_vertices=? "
            sql += "WHERE name=?"

            cur.execute(sql, tuple)
            print ''
            print 'SQL:'
            print sql, tuple
            print ''
            print 'Modified vertices in database for "%s"' % foo
            
    connection.commit()
    connection.close()

def plot_osiris_regions():

    # Verify that we can read it in from the database
    dbfile = '/u/jlu/data/gc/database/stars.sqlite'

    connection = sqlite.connect(dbfile)
    cur = connection.cursor()
    cur.execute('select x_vertices, y_vertices from fields')
    rows = cur.fetchall()
    
    py.clf()
    for rr in range(len(rows)):
        #pdb.set_trace()
        xverts = np.array([float(ff) for ff in rows[rr][0].split(',')])
        yverts = np.array([float(ff) for ff in rows[rr][1].split(',')])

        py.plot(np.append(xverts, xverts[0]), 
                np.append(yverts, yverts[0]), 
                color='red')
        
    rng = py.axis('equal')
    py.xlim(rng[1], rng[0])
    
    py.xlabel('R.A. Offset (arcsec)')
    py.ylabel('Dec. Offset (arcsec)')
    py.title('OSIRIS Fields')
    py.savefig('osiris_fields.png')


def osiris_masks():
    """
    Create extinction masks map for each OSIRIS field.
    Also create image masks for our NIRC2 06maylgs1 deep mosaic.
    """
    # ==========
    # Load up the extinction map
    # ==========
    schodel2010ext = '/u/jlu/work/gc/imf/extinction/2010schodel_AKs_fg6.fits'
    fitsObj = pyfits.open(schodel2010ext)
    map = fitsObj[0].data
    hdr = fitsObj[0].header

    map = np.array(map, dtype=float)

    scaleX = hdr['CD1_1'] * 3600.0
    scaleY = hdr['CD2_2'] * 3600.0
    sgraX = hdr['CRPIX1']
    sgraY = hdr['CRPIX2']

    mapX = np.arange(map.shape[1], dtype=float)
    mapY = np.arange(map.shape[0], dtype=float)

    mapX = (mapX - sgraX) * scaleX
    mapY = (mapY - sgraY) * scaleY

    # 2D versions
    mapXX, mapYY = np.meshgrid(mapX, mapY)

    # Paired version for input into points_inside_poly()
    xypoints = np.column_stack((mapXX.flatten(), mapYY.flatten()))


    # ==========
    # Lets also make a map in the NIRC2 pixel scale. 
    # ==========
    nircScale = 0.01    # arcsec/pixel
    nircSize = 3000.0   # image size in pixels
    
    nirc = np.zeros((nircSize, nircSize), dtype=int)
    nircOrigin = [nircSize/2.0, nircSize/2.0]

    nircHdr = hdr.copy()
    nircHdr['CRPIX1'] = nircOrigin[0]
    nircHdr['CRPIX2'] = nircOrigin[1]
    nircHdr['CD1_1'] = nircScale * -1.0 / 3600.0
    nircHdr['CD2_2'] = nircScale / 3600.0

    nircX = np.arange(nirc.shape[1], dtype=float)
    nircY = np.arange(nirc.shape[0], dtype=float)
    
    nircX = (nircX - nircOrigin[0]) * nircScale * -1.0
    nircY = (nircY - nircOrigin[1]) * nircScale

    nircXX, nircYY = np.meshgrid(nircX, nircY)

    xypointsNirc = np.column_stack((nircXX.flatten(), nircYY.flatten()))
    

    # ==========
    # The database contains, for each OSIRIS field, the vertices of 
    # the field of view. Use these to pull out the extinction in
    # each field.
    # ==========
    dbfile = '/u/jlu/data/gc/database/stars.sqlite'

    connection = sqlite.connect(dbfile)
    cur = connection.cursor()
    cur.execute('select name, x_vertices, y_vertices from fields where name not like "%Imaging%"')
    rows = cur.fetchall()

#     print 'Starting ds9... select from possible target windows:'
#     ds9_choices = ds9.ds9_targets()
#     for ii in range(len(ds9_choices)):
#         print '(%d) %s' % (ii, ds9_choices[ii])
#     foo = raw_input('   DS9 Target Choice: ')
#     ds9target = (ds9_choices[int(foo)].split())[1]
#     d = ds9.ds9(ds9target)

    # ==========
    # Keep the total masks for the extinction and NIRC2 maps
    # ==========
    total_mask = np.zeros(map.shape, dtype=np.uint8)
    total_mask_nirc = np.zeros(nirc.shape, dtype=np.uint8)

    # For each OSIRIS field, figure out which points are inside the polygon.
    for rr in range(len(rows)):
        fieldName = rows[rr][0]
        print 'Working on field %s' % fieldName

        xverts = np.array([float(ff) for ff in rows[rr][1].split(',')])
        yverts = np.array([float(ff) for ff in rows[rr][2].split(',')])

        xyverts = np.column_stack((xverts, yverts))

        # ==========
        # Mask for extinction
        # ==========
        mask = nxutils.points_inside_poly(xypoints, xyverts)
        mask = mask.reshape(map.shape)
        masked_map = map.copy()
        
        field_mask = np.zeros(map.shape, dtype=np.uint8)

        idx = np.where(mask == False)
        masked_map[idx[0], idx[1]] = 0

        idx = np.where(mask == True)
        field_mask[idx[0], idx[1]] = 1
        total_mask[idx[0], idx[1]] = 1

#         d.set('frame 1')
#         d.set_np2arr(map)
#         d.set('zoom to fit')
        
#         d.set('frame 2')
#         d.set_np2arr(total_mask)
#         d.set('zoom to fit')
        
        # ==========
        # Mask for NIRC2
        # ==========
        maskNirc = nxutils.points_inside_poly(xypointsNirc, xyverts)
        maskNirc = maskNirc.reshape(nirc.shape)
        field_mask_nirc = np.zeros(nirc.shape, dtype=np.uint8)

        idx = np.where(maskNirc == True)
        field_mask_nirc[idx[0], idx[1]] = 1
        total_mask_nirc[idx[0], idx[1]] = 1

#         foo = raw_input('Continue? (yes=return, no=q)')
        
#         if (foo == 'q'):
#             return

        # Save the individual field masks
        fitsObj[0].header = hdr
        fitsObj[0].data = field_mask
        fitsObj.writeto('extinct_mask_%s.fits' % fieldName.replace(' ', ''),
                        clobber=True)

        fitsObj[0].header = nircHdr
        fitsObj[0].data = field_mask_nirc
        fitsObj.writeto('nirc2_mask_%s.fits' % fieldName.replace(' ', ''),
                        clobber=True)
        

    # Save the total field mask
    fitsObj[0].header = hdr
    fitsObj[0].data = total_mask
    fitsObj.writeto('extinct_mask_all.fits', clobber=True)

    fitsObj[0].header = nircHdr
    fitsObj[0].data = total_mask_nirc
    fitsObj.writeto('nirc2_mask_all.fits', clobber=True)



