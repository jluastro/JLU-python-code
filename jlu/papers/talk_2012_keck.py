import numpy as np
import pylab as py
import atpy

def plot_wd1_pm():
    """
    Plot the proper motion vectors for Wd 1.
    """
    dataFile = '/u/jlu/data/Wd1/hst/from_jay/EXPORT_WEST1.2012.02.04/wd1_catalog.fits'

    d = atpy.Table(dataFile)
    d = d.where((d.x2005_e < 0.05) & (d.y2005_e < 0.05) &
                (d.x2010_e < 0.05) & (d.y2010_e < 0.05) &
                (d.mag125_e < 0.03))

    f_pmX = 1.8 # mas/yr
    f_pmY = 1.3 # mas/yr
    # f_pmX = 0. # mas/yr
    # f_pmY = 0. # mas/yr

    pmX = (d.dx / 5.0) - f_pmX
    pmY = (d.dy / 5.0) - f_pmY
    
    py.clf()
    py.subplots_adjust(left=0.02, right=0.98, bottom=0.02, top=0.98)
    Q = py.quiver(d.x2005 * 0.06, d.y2005 * 0.06, pmX, pmY, scale=100)
    py.setp(py.gca().get_xticklabels(), visible=False)
    py.setp(py.gca().get_yticklabels(), visible=False)
    py.xlim(30, 200)
    py.ylim(30, 200)
    

    # py.quiverkey(Q, 0.5, 0.94, 10**5, '30 km/s', coordinates='figure',
    #              color='red', labelpos='E')
    py.savefig('wd1_pm.png')

    return

    
def plot_arches_pm():
    """
    Plot the proper motion vectors for the Arches (Clarkson+ 2012).
    """
    dataFile = '/u/jlu/data/arches/clarkson2012_table5.txt'

    d = atpy.Table(dataFile, type='mrt')

    # Put into a "field" reference frame.
    idx = np.where(d.FMem > 0.9)[0]
    f_pmX = d.pmX[idx].mean()
    f_pmY = d.pmY[idx].mean()

    d.pmX -= f_pmX
    d.pmY -= f_pmY

    py.clf()
    Q = py.quiver(d.DelX, d.DelY, d.pmX, d.pmY)
    py.quiverkey(Q, 95, 95, 1, '40 km/s', coordinates='figure')
