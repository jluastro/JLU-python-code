import numpy as np
from pysqlite2 import dbapi2 as sqlite
from gcwork import starTables
import asciidata
import atpy
import pdb

dbfile = '/u/jlu/work/w51/database/w51a.sqlite'

def update_labels(labelFile):
    # Read in the label.dat file
    lab = starTables.Labels(labelFile=labelFile)

    # Create a connection to the database
    connection = sqlite.connect(dbfile)

    # Create a cursor object
    cur = connection.cursor()

    # Loop through all the stars and insert or replace them
    for ss in range(len(lab.name)):
        sql = 'insert or replace into stars '
        sql += '(name, kp, x, y, xerr, yerr, t0, '
        sql += 'vx, vy, vxerr, vyerr, useToAlign) '
        sql += 'values (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)'

        cur.execute(sql, (lab.name[ss], lab.mag[ss], lab.x[ss], lab.y[ss],
                          lab.xerr[ss], lab.yerr[ss], lab.t0[ss],
                          lab.vx[ss], lab.vy[ss], lab.vxerr[ss], lab.vyerr[ss],
                          lab.useToAlign[ss]))

    connection.commit()

def update_stars(astromTable):
    """
    Update the database with the positions, photometry, and velocities
    from our final results.
    """
    foo = asciidata.open(astromTable)
    
    name = foo[0].tonumpy()
    x = foo[1].tonumpy()
    xerr = foo[3].tonumpy()
    y = foo[4].tonumpy()
    yerr = foo[6].tonumpy()
    h = foo[7].tonumpy()
    herr = foo[9].tonumpy()
    kp = foo[10].tonumpy()
    kperr = foo[12].tonumpy()
    lp = foo[13].tonumpy()
    lperr = foo[15].tonumpy()
    x0 = foo[16].tonumpy()
    x0err = foo[18].tonumpy()
    y0 = foo[19].tonumpy()
    y0err = foo[21].tonumpy()
    vx = foo[22].tonumpy()
    vxerr = foo[24].tonumpy()
    vy = foo[25].tonumpy()
    vyerr = foo[27].tonumpy()
    vy = foo[25].tonumpy()
    vyerr = foo[27].tonumpy()
    t0 = foo[28].tonumpy()
    velField = foo[29].tonumpy()


    # Create a connection to the database
    connection = sqlite.connect(dbfile)

    # Create a cursor object
    cur = connection.cursor()

    for ss in range(len(name)):
        sql = 'update stars '
        sql += 'set x=?, xerr=?, y=?, yerr=?, vx=?, vxerr=?, vy=?, vyerr=?, '
        sql += 'h=?, herr=?, kp=?, kperr=?, lp=?, lperr=?, '
        sql += 't0=?, velField=? where name=?'

        if x0[ss] == 0:
            x0[ss] = x[ss]
        if y0[ss] == 0:
            y0[ss] = y[ss]
        
        cur.execute(sql, (x0[ss], x0err[ss], y0[ss], y0err[ss],
                          vx[ss], vxerr[ss], vy[ss], vyerr[ss],
                          h[ss], herr[ss], kp[ss], kperr[ss], lp[ss], lperr[ss],
                          t0[ss], velField[ss], name[ss]))
        
    connection.commit()
        


def make_psf_starlists():
    psfdir = '/u/jlu/code/idl/w51/psfstars'
    
    # Load up the PSF stars
    psfStars = atpy.Table('sqlite', dbfile, table='psfstars')
    fields = np.unique(psfStars['field'])

    # Load up all the rest of the stars
    sql = 'select name, kp, x, y, vx, vy, t0 from stars'
    allStars = atpy.Table('sqlite', dbfile, table='stars', query=sql)
        
    for ff in range(len(fields)):
        _out = open('%s/w51a_%s_psf.list' % (psfdir, fields[ff]), 'w')
        _out.write('%-13s  %5s  %7s  %7s  %7s  %7s  %8s  %-10s  %4s\n' %
                   ('#Name', 'Kp', 'Xarc', 'Yarc', 'Vx', 'Vy', 't0', 
                    'Filt', 'PSF?'))
        
        for ss in range(len(allStars)):
            # Figure out if this star is a PSF star
            idx = np.where((psfStars['field'] == fields[ff]) & 
                           (psfStars['star'] == allStars['name'][ss]))[0]

            if len(idx) > 0:
                # This is a PSF star
                isPsfStar = 1
                exFilt = psfStars['filtersExcluded'][idx[0]]
                
                if exFilt == '' or exFilt == 'None':
                    exFilt = '-'

            else:
                isPsfStar = 0
                exFilt = '-'

            # Now print it out
            _out.write('%-13s  %5.2f  %7.3f  %7.3f  %7.3f  %7.3f  ' %
                       (allStars['name'][ss], allStars['kp'][ss],
                        allStars['x'][ss], allStars['y'][ss], 
                        allStars['vx'][ss], allStars['vy'][ss]))
            _out.write('%8.3f  %-10s  %4d\n' % 
                       (allStars['t0'][ss], exFilt, isPsfStar))

                
        _out.close()
                           
        
    # Make a wide-field star list that contains ALL the PSF stars.
    _out = open('%s/w51a_wide_psf.list' % (psfdir), 'w')
    _out.write('%-13s  %5s  %7s  %7s  %7s  %7s  %8s  %-10s  %4s\n' %
               ('#Name', 'Kp', 'Xarc', 'Yarc', 'Vx', 'Vy', 't0', 
                'Filt', 'PSF?'))
    
    for ss in range(len(allStars)):
        # Figure out if this star is a PSF star
        idx = np.where(psfStars['star'] == allStars['name'][ss])[0]

        if len(idx) > 0:
            # This is a PSF star
            isPsfStar = 1
            exFilt = psfStars['filtersExcluded'][idx[0]]
            
            if exFilt == '' or exFilt == 'None':
                exFilt = '-'

        else:
            isPsfStar = 0
            exFilt = '-'

        # Now print it out
        _out.write('%-13s  %5.2f  %7.3f  %7.3f  %7.3f  %7.3f  ' %
                   (allStars['name'][ss], allStars['kp'][ss],
                    allStars['x'][ss], allStars['y'][ss], 
                    allStars['vx'][ss], allStars['vy'][ss]))
        _out.write('%8.3f  %-10s  %4d\n' % 
                   (allStars['t0'][ss], exFilt, isPsfStar))

                
    _out.close()
                           
        


