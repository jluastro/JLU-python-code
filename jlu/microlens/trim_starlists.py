import numpy as np
from astropy.table import Table
from astropy.io import fits
import os

def trim_in_radius(Readpath = '/Users/jlu/work/microlens/OB120169/analysis_ob120169_2014_03_22/lis/',   
                   TargetName = 'ob120169',
                   epochs = ['12may', '12jun', '12jul', '13apr', '13jul', '15may', '15jun07'],
                   radius_cut_in_mas=4000.0):
    
    Nepochs = len(epochs)
    for i in range(Nepochs):
        # Read in coords from PSF list 
        Readfile = 'mag' + epochs[i]+ '_' + TargetName +'_kp_rms_named.lis' 
        Starlist = Readpath + Readfile 
        PSFtab = Table.read(Starlist, format='ascii')
        names = PSFtab[PSFtab.colnames[0]]
        mags = PSFtab[PSFtab.colnames[1]]
        Xpix = PSFtab[PSFtab.colnames[3]]
        Ypix = PSFtab[PSFtab.colnames[4]]

        fitsPath = '/g/lu/data/microlens/' + epochs[i] + '/combo/'
        fitsFile = fitsPath + 'mag' + epochs[i] + '_' + TargetName.lower() + '_kp.fits'
        fitsData = fits.getdata(fitsFile)
        center = np.array(fitsData.shape) / 2.0
        center_x = center[0]
        center_y = center[1]
        
        Nstars = len(Xpix)
        
        rad_pix = np.hypot(Xpix - center_x, Ypix - center_y)
        rad_mas = rad_pix * 10.0

        indgood = np.where((rad_mas <= radius_cut_in_mas))[0]
        names = names[indgood]
        mags = mags[indgood]
        Xpix = Xpix[indgood]
        Ypix = Ypix[indgood]
                
        f = open(Starlist, 'r')

        lines = []
        for line in f:
            li = line.strip()
            if not li.startswith('#'):
                lines.append(line)
        f.close()

        msg = 'Keeping {0} of {1} stars within 4" in {2}'
        print msg.format(len(indgood), Nstars, Starlist)
        
        str1, str2 = Starlist.split('.lis')
        newName = str1 + '_radTrim.lis'
        f = open(newName, 'w')
        for k in range(len(indgood)):
        	f.write(lines[indgood[k]])
        f.close()

    return
