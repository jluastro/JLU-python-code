import healpy
import numpy as np
import pylab as py
import math
from matplotlib import cm, rcParams

def test():
    testfile = '/home/phobos/jlu/work/gc/proper_motion/align/08_03_26/aorb_efit/'
    testfile += 'disk.neighbor.dat'

    nside = 32
    npix = healpy.nside2npix(nside)
    pixIdx = np.arange(npix)
    
    i, o = healpy.pix2ang(nside, pixIdx)

    foo = np.fromfile(testfile, dtype=float)
    foo2 = foo.reshape((4, npix))
    data = foo2[2]

    plot(data, showdisks=True)


def plot(data, showdisks=False, unit='stars/deg^2'):
    _fontsize = rcParams['font.size']
    _fontweight = rcParams['font.weight']

    rcParams['font.size'] = 14
    rcParams['font.weight'] = 'normal'

    # Draw the map
    healpy.mollview(map=data, rot=[0, 180],
                    cmap=cm.gist_stern_r, title='', 
                    unit=unit)
    healpy.graticule(dpar=30.0, dmer=45.0, local=True)

    # Handle colorbar labels
    try:
        fig = py.gcf()
        cbar = fig.axes[-1]
        
        cbarLabels = cbar.xaxis.get_ticklabels()
        for cc in range(len(cbarLabels)):
            cbarLabels[cc] = '%.5f' % float(cbarLabels[cc].get_text())
    
        cbar.xaxis.set_ticklabels(cbarLabels, fontweight='bold', fontsize=14)
    except UnicodeEncodeError:
        pass
    
    # Draw contours for the old disk positions
    if showdisks:
        incl = [124.0, 30.0]
        incl_err = [2.0, 4.0]
        incl_thick = [7.0, 9.5]
        Om = [100.0, 167.0]
        Om_err = [3.0, 9.0]
        Om_thick = [7.0, 9.5]

        for jj in range(len(incl)):
            x1, y1 = ellipseOutline(incl[jj], Om[jj], 
                                  incl_err[jj], Om_err[jj])
            x2, y2 = ellipseOutline(incl[jj], Om[jj], 
                                  incl_thick[jj], Om_thick[jj])
            healpy.projplot(x1, y1, lonlat=True, color='k', 
                            linestyle='-', linewidth=2)
            healpy.projplot(x2, y2, lonlat=True, color='k',
                            linestyle='--', linewidth=2)


    # Flip so that the axes go in the right direction
    foo = py.axis()
    py.xlim(2.01, -2.01)

    # Make axis labels.
    healpy.projtext(178, 90, '0', lonlat=True)
    healpy.projtext(178, 60, '30', lonlat=True)
    healpy.projtext(178, 30, '60', lonlat=True)
    healpy.projtext(178, 0, '90', lonlat=True)
    healpy.projtext(178, -30, '120', lonlat=True)
    healpy.projtext(178, -60, '150', lonlat=True)
    healpy.projtext(178, -90, 'i = 180', lonlat=True,
                    horizontalalignment='center')

    healpy.projtext(92, 1, 'S', lonlat=True,
                    horizontalalignment='right', verticalalignment='top')
    healpy.projtext(182, 1, 'W', lonlat=True, 
                    horizontalalignment='right', verticalalignment='top')
    healpy.projtext(272, 1, 'N', lonlat=True, 
                    horizontalalignment='right', verticalalignment='top')
    healpy.projtext(362, 1, 'E', lonlat=True, 
                    horizontalalignment='right', verticalalignment='top')

    rcParams['font.size'] = _fontsize
    rcParams['font.weight'] = _fontweight

                    

def ellipseOutline(i, o, irad, orad):
    # Go through 360.0 degrees and find the coordinates for each angle
    binsize = 10
    bincnt = 360 / binsize
    
    angle = np.arange(bincnt + 1, dtype=float)
    angle *= binsize * math.pi / 180.0
    
    x = o + (orad * np.sin(angle))
    y = i + (irad * np.cos(angle))

    y = 90.0 - y

    return (x, y)
