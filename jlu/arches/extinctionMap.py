import numpy as np
import pylab as py
import asciidata

stolteCatalog = '/u/jlu/work/arches/photo_calib/from_stolte/stolte_catalog.dat'

def color_map_hkp():
    """
    Plot a map of the H-Kp colors..
    """
    tab = asciidata.open(stolteCatalog)

    member = tab[14].tonumpy()
    idx = np.where(member == 1)[0]

    id = (tab[0].tonumpy())[idx]
    field = (tab[1].tonumpy())[idx]

    x = (tab[16].tonumpy())[idx]
    y = (tab[17].tonumpy())[idx]
    xe = (tab[18].tonumpy())[idx]
    ye = (tab[19].tonumpy())[idx]
    
    h = (tab[6].tonumpy())[idx]
    he = (tab[7].tonumpy())[idx]
    kp = (tab[8].tonumpy())[idx]
    kpe = (tab[9].tonumpy())[idx]
    lp = (tab[10].tonumpy())[idx]
    lpe = (tab[11].tonumpy())[idx]
    

    h_kp = h - kp
    h_kp_err = np.hypot(he, kpe)

    print h_kp.min(), h_kp.max()
    scale = 10.0
    
    py.close(1)
    py.figure(1, figsize=(18, 12))
    py.clf()
    py.scatter(x, y, scale**h_kp)
    
    py.axes().set_aspect('equal')
    py.xlim(-15, 25)
    py.ylim(-10, 18)
