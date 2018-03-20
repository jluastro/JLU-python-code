import numpy as np
import pylab as py
import pdb
import math
import os
from jlu.observe import skycalc
#from jlu.microlens import residuals
import shutil, os, sys
import scipy
import scipy.stats
from gcwork import starset
from gcwork import starTables
from astropy.table import Table
from jlu.util import fileUtil

def plot_airmass_moon():
    # coordinates to center of OGLE field BLG502 (2017 objects)
    ra = "18:00:00"
    dec = "-30:00:00"
    months = np.array([8, 8])
    days = np.array([1, 30])
    outdir = '/Users/fatima/Desktop/keck_prop_18B_mlens/'
    
    # Keck 2
    skycalc.plot_airmass(ra, dec, 2018, months, days, 'keck2', outfile=outdir + 'microlens_airmass_keck2.png', date_idx=-1)
    skycalc.plot_moon(ra, dec, 2018, months, outfile=outdir + 'microlens_moon.png')

    # Keck 1
    #skycalc.plot_airmass(ra, dec, 2018, months, days, 'keck1', outfile=outdir + 'microlens_airmass_keck1.png', date_idx=-1)
    
    return


def plot_3_targs():
    
    #name three objects
    targNames = ['ob140613', 'OB150211', 'ob150029']

    #object alignment directories
    an_dirs = ['/u/nijaid/work/OB140613/a_2017_09_21/prop/',
                 '/Users/jlu/work/microlens/OB150211/a_2017_09_19/prop/',
                 '/Users/jlu/work/microlens/OB150029/a_2017_09_21/prop/']
    align_dirs = ['align/align_t', 'align/align_t', 'align/align_t']
    points_dirs = ['points_d/', 'points_a/', 'points_d/']
    poly_dirs = ['polyfit_d/fit', 'polyfit_a/fit', 'polyfit_d/fit']

    xlim = [1.0, 2.0, 1.5]
    ylim = [1.0, 4.0, 1.5]

    #Output file
    filename = '/Users/jlu/doc/proposals/keck/uc/18A/plot_3_targs.png'
    figsize = (15, 4.5)
    
    ps = 9.92


    py.close(1)
    py.figure(1, figsize=figsize)
    
    Ntarg = len(targNames)
    for i in range(Ntarg):
        rootDir = an_dirs[i]
        starName = targNames[i]
        align = align_dirs[i]
        poly = poly_dirs[i]
        point = points_dirs[i]
    
        s = starset.StarSet(rootDir + align)
        s.loadPolyfit(rootDir + poly, accel=0, arcsec=0)

        names = s.getArray('name')
        mag = s.getArray('mag')
        x = s.getArray('x') 
        y = s.getArray('y') 

        ii = names.index(starName)
        star = s.stars[ii]

        pointsTab = Table.read(rootDir + point + starName + '.points', format='ascii')

        time = pointsTab[pointsTab.colnames[0]]
        x = pointsTab[pointsTab.colnames[1]]
        y = pointsTab[pointsTab.colnames[2]]
        xerr = pointsTab[pointsTab.colnames[3]]
        yerr = pointsTab[pointsTab.colnames[4]]

        if i == 1:
            print('Doing MJD')
            idx_2015 = np.where(time <= 57387)
            idx_2016 = np.where((time > 57387) & (time <= 57753))
            idx_2017 = np.where((time > 57753) & (time <= 58118))
        else:
            idx_2015 = np.where(time < 2016)
            idx_2016 = np.where((time >= 2016) & (time < 2017))
            idx_2017 = np.where((time >= 2017) & (time < 2018))

        fitx = star.fitXv
        fity = star.fitYv
        dt = time - fitx.t0
        fitLineX = fitx.p + (fitx.v * dt)
        fitSigX = np.sqrt( fitx.perr**2 + (dt * fitx.verr)**2 )

        fitLineY = fity.p + (fity.v * dt)
        fitSigY = np.sqrt( fity.perr**2 + (dt * fity.verr)**2 )

        from matplotlib.ticker import FormatStrFormatter
        fmtX = FormatStrFormatter('%5i')
        fmtY = FormatStrFormatter('%6.2f')
        fontsize1 = 16
        
        # Convert everything into relative coordinates:
        x -= fitx.p
        y -= fity.p
        fitLineX -= fitx.p
        fitLineY -= fity.p

        # Change plate scale.
        x = x * ps * -1.0
        y = y * ps
        xerr *= ps
        yerr *= ps
        fitLineX = fitLineX * ps * -1.0
        fitLineY = fitLineY * ps
        fitSigX *= ps
        fitSigY *= ps

        paxes = py.subplot(1, 3, i+1)
        py.errorbar(x[idx_2015], y[idx_2015], xerr=xerr[idx_2015], yerr=yerr[idx_2015], fmt='r.', label='2015')  
        py.errorbar(x[idx_2016], y[idx_2016], xerr=xerr[idx_2016], yerr=yerr[idx_2016], fmt='g.', label='2016')  
        py.errorbar(x[idx_2017], y[idx_2017], xerr=xerr[idx_2017], yerr=yerr[idx_2017], fmt='b.', label='2017')  

        # if i==1:
        #     py.yticks(np.arange(np.min(y-yerr-0.1*ps), np.max(y+yerr+0.1*ps), 0.3*ps))
        #     py.xticks(np.arange(np.min(x-xerr-0.1*ps), np.max(x+xerr+0.1*ps), 0.25*ps))
        # else:
        #     py.yticks(np.arange(np.min(y-yerr-0.1*ps), np.max(y+yerr+0.1*ps), 0.15*ps))
        #     py.xticks(np.arange(np.min(x-xerr-0.1*ps), np.max(x+xerr+0.1*ps), 0.15*ps))
        paxes.tick_params(axis='both', which='major', labelsize=fontsize1)
        paxes.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
        paxes.xaxis.set_major_formatter(FormatStrFormatter('%.2f'))
        py.xlabel('X offset (mas)', fontsize=fontsize1)
        py.ylabel('Y offset (mas)', fontsize=fontsize1)
        py.plot(fitLineX, fitLineY, 'k-', label='_nolegend_')    
        py.plot(fitLineX + fitSigX, fitLineY + fitSigY, 'k--', label='_nolegend_')
        py.plot(fitLineX - fitSigX, fitLineY - fitSigY, 'k--',label='_nolegend_')

        # Plot lines between observed point and the best fit value along the model line.
        for ee in range(len(time)):
            if ee in idx_2015[0].tolist():
                color_line = 'red'
            if ee in idx_2016[0].tolist():
                color_line = 'green'
            if ee in idx_2017[0].tolist():
                color_line = 'blue'
                
            py.plot([fitLineX[ee], x[ee]], [fitLineY[ee], y[ee]], color=color_line, linestyle='dashed', alpha=0.8)
        
        py.axis([xlim[i], -xlim[i], -ylim[i], ylim[i]])
        
        py.title(starName.upper())
        if i==0:
            py.legend(loc=1)


    
    py.subplots_adjust(wspace=0.6, hspace=0.6, left = 0.08, bottom = 0.05, right=0.95, top=0.90)
    py.tight_layout()
    py.show()
    py.savefig(filename)


    return
