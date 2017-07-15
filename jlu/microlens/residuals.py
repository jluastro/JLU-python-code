import shutil, os, sys
import pylab as py
import numpy as np
import scipy
import scipy.stats
# from jlu.gc.gcwork import starset
# from jlu.gc.gcwork import starTables
from gcwork import starset
from gcwork import starTables
from astropy.table import Table
from jlu.util import fileUtil
import sys
import pdb
import scipy.stats

def plotStar(starNames, rootDir='./', align='align/align_d_rms_1000_abs_t',
             poly='polyfit_d/fit', points='points_d/', radial=False, NcolMax=3, figsize=(15,15)):

    print( 'Creating residuals plots for star(s):' )
    print( starNames )
    
    s = starset.StarSet(rootDir + align)
    s.loadPolyfit(rootDir + poly, accel=0, arcsec=0)
    Nstars = len(starNames)
    Ncols = 2 * np.min([Nstars, NcolMax])

    if Nstars <= Ncols/2:
        Nrows = 3
    else:
        Nrows = (Nstars // (Ncols / 2)) * 3

    py.close('all')
    py.figure(2, figsize=figsize)
    names = s.getArray('name')
    mag = s.getArray('mag')
    x = s.getArray('x')
    y = s.getArray('y')
    r = np.hypot(x,y)
    
    for i in range(Nstars):
    
        starName = starNames[i]
        
        try:
            
            ii = names.index(starName)
            star = s.stars[ii]
            pdb.set_trace()
       
            pointsTab = Table.read(rootDir + points + starName + '.points', format='ascii')
        
            time = pointsTab[pointsTab.colnames[0]]
            x = pointsTab[pointsTab.colnames[1]]
            y = pointsTab[pointsTab.colnames[2]]
            xerr = pointsTab[pointsTab.colnames[3]]
            yerr = pointsTab[pointsTab.colnames[4]]
        
            fitx = star.fitXv
            fity = star.fitYv
            dt = time - fitx.t0
            fitLineX = fitx.p + (fitx.v * dt)
            fitSigX = np.sqrt( fitx.perr**2 + (dt * fitx.verr)**2 )
        
            fitLineY = fity.p + (fity.v * dt)
            fitSigY = np.sqrt( fity.perr**2 + (dt * fity.verr)**2 )

        
            if (radial == True):
                # Lets also do radial/tangential
                x0 = fitx.p
                y0 = fity.p
                vx = fitx.v
                vy = fity.v
                x0e = fitx.perr
                y0e = fity.perr
                vxe = fitx.verr
                vye = fity.verr
                
                r0 = np.sqrt(x0**2 + y0**2)
        
                vr = ((vx*x0) + (vy*y0)) / r0
                vt = ((vx*y0) - (vy*x0)) / r0
                vre =  (vxe*x0/r0)**2 + (vye*y0/r0)**2
                vre += (y0*x0e*vt/r0**2)**2 + (x0*y0e*vt/r0**2)**2
                vre =  np.sqrt(vre)
                vte =  (vxe*y0/r0)**2 + (vye*x0/r0)**2
                vte += (y0*x0e*vr/r0**2)**2 + (x0*y0e*vr/r0**2)**2
                vte =  np.sqrt(vte)
        
                r = ((x*x0) + (y*y0)) / r0
                t = ((x*y0) - (y*x0)) / r0
                rerr = (xerr*x0/r0)**2 + (yerr*y0/r0)**2
                rerr += (y0*x0e*t/r0**2)**2 + (x0*y0e*t/r0**2)**2
                rerr =  np.sqrt(rerr)
                terr =  (xerr*y0/r0)**2 + (yerr*x0/r0)**2
                terr += (y0*x0e*r/r0**2)**2 + (x0*y0e*r/r0**2)**2
                terr =  np.sqrt(terr)
        
                fitLineR = ((fitLineX*x0) + (fitLineY*y0)) / r0
                fitLineT = ((fitLineX*y0) - (fitLineY*x0)) / r0
                fitSigR = ((fitSigX*x0) + (fitSigY*y0)) / r0
                fitSigT = ((fitSigX*y0) - (fitSigY*x0)) / r0
        
                diffR = r - fitLineR
                diffT = t - fitLineT
                sigR = diffR / rerr
                sigT = diffT / terr
        
                idxR = np.where(abs(sigR) > 4)
                idxT = np.where(abs(sigT) > 4)
                
        
            diffX = x - fitLineX
            diffY = y - fitLineY
            diff = np.hypot(diffX, diffY)
            rerr = np.sqrt((diffX*xerr)**2 + (diffY*yerr)**2) / diff
            sigX = diffX / xerr
            sigY = diffY / yerr
            sig = diff / rerr
        
            
            # Determine if there are points that are more than 5 sigma off
            idxX = np.where(abs(sigX) > 4)
            idxY = np.where(abs(sigY) > 4)
            idx = np.where(abs(sig) > 4)
        
            print( 'Star:        ', starName )
            print( '\tX Chi^2 = %5.2f (%6.2f for %2d dof)' % 
                  (fitx.chi2red, fitx.chi2, fitx.chi2/fitx.chi2red))
            print( '\tY Chi^2 = %5.2f (%6.2f for %2d dof)' % 
                  (fity.chi2red, fity.chi2, fity.chi2/fity.chi2red))
            # print( 'X  Outliers: ', time[idxX] )
            # print( 'Y  Outliers: ', time[idxY] )
            # if (radial):
            #     print( 'R  Outliers: ', time[idxX] )
            #     print( 'T  Outliers: ', time[idxY] )
            # print( 'XY Outliers: ', time[idx] )
        
            # close(2)
            #             figure(2, figsize=(7, 8))
            #             clf()
            
            dateTicLoc = py.MultipleLocator(3)
            dateTicRng = [2006, 2017]
            dateTics = np.array([2011, 2012, 2013, 2014, 2015, 2016, 2017])
            DateTicsLabel = dateTics-2000

            # See if we are using MJD instead.
            if time[0] > 50000:
                dateTicLoc = py.MultipleLocator(1000)
                dateTicRng = [56000, 58000]
                dateTics = np.arange(dateTicRng[0], dateTicRng[-1]+1, 1000)
                DateTicsLabel = dateTics
                
    	    
            maxErr = np.array([xerr, yerr]).max()
            resTicRng = [-1.1*maxErr, 1.1*maxErr]
        
            from matplotlib.ticker import FormatStrFormatter
            fmtX = FormatStrFormatter('%5i')
            fmtY = FormatStrFormatter('%6.2f')
            fontsize1 = 10
	        
#             paxes = py.subplot2grid((3,2*Nstars),(0, 2*i))
            
            
            if i < (Ncols/2):
                col = (2*i)+1
                row = 1
            else:
                col = 1 + 2*(i % (Ncols/2))
                row = 1 + 3*(i//(Ncols/2)) 
            
            ind = (row-1)*Ncols + col
            
            paxes = py.subplot(Nrows, Ncols, ind)
            py.plot(time, fitLineX, 'b-')
            py.plot(time, fitLineX + fitSigX, 'b--')
            py.plot(time, fitLineX - fitSigX, 'b--')
            py.errorbar(time, x, yerr=xerr, fmt='k.')
            rng = py.axis()
            py.ylim(np.min(x-xerr-0.1),np.max(x+xerr+0.1)) 
            py.xlabel('Date - 2000 (yrs)', fontsize=fontsize1)
            if time[0] > 50000:
                py.xlabel('Date (MJD)', fontsize=fontsize1)
            py.ylabel('X (pix)', fontsize=fontsize1)
            paxes.xaxis.set_major_formatter(fmtX)
            paxes.get_xaxis().set_major_locator(dateTicLoc)
            paxes.yaxis.set_major_formatter(fmtY)
            paxes.tick_params(axis='both', which='major', labelsize=fontsize1)
            py.yticks(np.arange(np.min(x-xerr-0.1), np.max(x+xerr+0.1), 0.2))
            py.xticks(dateTics, DateTicsLabel)
            py.xlim(np.min(dateTics), np.max(dateTics))
            py.annotate(starName,xy=(1.0,1.1), xycoords='axes fraction', fontsize=12, color='red')
            
            
#             show()
    
            col = col + 1
            ind = (row-1)*Ncols + col
            
#             paxes = subplot2grid((3,2*Nstars),(0, 2*i+1))
            paxes = py.subplot(Nrows, Ncols, ind)
            py.plot(time, fitLineY, 'b-')
            py.plot(time, fitLineY + fitSigY, 'b--')
            py.plot(time, fitLineY - fitSigY, 'b--')
            py.errorbar(time, y, yerr=yerr, fmt='k.')
            rng = py.axis()
            py.axis(dateTicRng + [rng[2], rng[3]], fontsize=fontsize1)
            py.xlabel('Date - 2000 (yrs)', fontsize=fontsize1)
            if time[0] > 50000:
                py.xlabel('Date (MJD)', fontsize=fontsize1)
            py.ylabel('Y (pix)', fontsize=fontsize1)
            #paxes.get_xaxis().set_major_locator(dateTicLoc)
            paxes.xaxis.set_major_formatter(fmtX)
            paxes.get_xaxis().set_major_locator(dateTicLoc)
            paxes.yaxis.set_major_formatter(fmtY)
            paxes.tick_params(axis='both', which='major', labelsize=12)
            py.ylim(np.min(y-yerr-0.1),np.max(y+yerr+0.1))
            py.yticks(np.arange(np.min(y-yerr-0.1), np.max(y+yerr+0.1), 0.2))
            py.xticks(dateTics, DateTicsLabel)
            py.xlim(np.min(dateTics), np.max(dateTics))
#             show()
            
            row = row + 1
            col = col - 1
            ind = (row-1)*Ncols + col
    
#             paxes = subplot2grid((3,2*Nstars),(1, 2*i))
            paxes = py.subplot(Nrows, Ncols, ind)
            py.plot(time, np.zeros(len(time)), 'b-')
            py.plot(time, fitSigX, 'b--')
            py.plot(time, -fitSigX, 'b--')
            py.errorbar(time, x - fitLineX, yerr=xerr, fmt='k.')
            py.axis(dateTicRng + resTicRng, fontsize=fontsize1)
            py.xlabel('Date - 2000 (yrs)', fontsize=fontsize1)
            if time[0] > 50000:
                py.xlabel('Date (MJD)', fontsize=fontsize1)
            py.ylabel('X Residuals (pix)', fontsize=fontsize1)
            paxes.get_xaxis().set_major_locator(dateTicLoc)
            paxes.xaxis.set_major_formatter(fmtX)
            paxes.tick_params(axis='both', which='major', labelsize=fontsize1)
            py.xticks(dateTics, DateTicsLabel)
            py.xlim(np.min(dateTics), np.max(dateTics))
#             show()
	        
            col = col + 1
            ind = (row-1)*Ncols + col
    
#             paxes = subplot2grid((3,2*Nstars),(1, 2*i+1))
            paxes = py.subplot(Nrows, Ncols, ind)
#             paxes = subplot(3, 2, 4)
            py.plot(time, np.zeros(len(time)), 'b-')
            py.plot(time, fitSigY, 'b--')
            py.plot(time, -fitSigY, 'b--')
            py.errorbar(time, y - fitLineY, yerr=yerr, fmt='k.')
            py.axis(dateTicRng + resTicRng, fontsize=fontsize1)
            py.xlabel('Date -2000 (yrs)', fontsize=fontsize1)
            if time[0] > 50000:
                py.xlabel('Date (MJD)', fontsize=fontsize1)
            py.ylabel('Y Residuals (pix)', fontsize=fontsize1)
            paxes.get_xaxis().set_major_locator(dateTicLoc)
            paxes.xaxis.set_major_formatter(fmtX)
            paxes.tick_params(axis='both', which='major', labelsize=fontsize1)
            py.xticks(dateTics, DateTicsLabel)
            py.xlim(np.min(dateTics), np.max(dateTics))
#             show()
            
            row = row + 1
            col = col - 1
            ind = (row-1)*Ncols + col
    
           
            paxes = py.subplot(Nrows, Ncols, ind)
            py.errorbar(x,y, xerr=xerr, yerr=yerr, fmt='k.')
            py.yticks(np.arange(np.min(y-yerr-0.1), np.max(y+yerr+0.1), 0.2))
            py.xticks(np.arange(np.min(x-xerr-0.1), np.max(x+xerr+0.1), 0.2), rotation = 270)
            py.axis('equal')
            paxes.tick_params(axis='both', which='major', labelsize=fontsize1)
            paxes.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
            paxes.xaxis.set_major_formatter(FormatStrFormatter('%.2f'))
            py.xlabel('X (pix)', fontsize=fontsize1)
            py.ylabel('Y (pix)', fontsize=fontsize1)
            py.plot(fitLineX, fitLineY, 'b-')    
            
            col = col + 1
            ind = (row-1)*Ncols + col

            bins = np.arange(-7, 7, 1)
            paxes = py.subplot(Nrows, Ncols, ind)
            id = np.where(diffY < 0)[0]
            sig[id] = -1.*sig[id] 
            (n, b, p) = py.hist(sigX, bins, histtype='stepfilled', color='b')
            py.setp(p, 'facecolor', 'b')
            (n, b, p) = py.hist(sigY, bins, histtype='step', color='r')
            py.axis([-7, 7, 0, 8], fontsize=10)
            py.xlabel('X Residuals (sigma)', fontsize=fontsize1)
            py.ylabel('Number of Epochs', fontsize=fontsize1)

            ##########
            #
            # Also plot radial/tangential
            #
            ##########
            if (radial == True):
                py.clf()
        
                dateTicLoc = py.MultipleLocator(3)
                
                maxErr = np.array([rerr, terr]).max()
                resTicRng = [-3*maxErr, 3*maxErr]
                
                from matplotlib.ticker import FormatStrFormatter
                fmtX = FormatStrFormatter('%5i')
                fmtY = FormatStrFormatter('%6.2f')
                
                paxes = py.subplot(3,2,1)
                py.plot(time, fitLineR, 'b-')
                py.plot(time, fitLineR + fitSigR, 'b--')
                py.plot(time, fitLineR - fitSigR, 'b--')
                py.errorbar(time, r, yerr=rerr, fmt='k.')
                rng = py.axis()
                py.axis(dateTicRng + [rng[2], rng[3]])
                py.xlabel('Date (yrs)')
                py.ylabel('R (pix)')
                paxes.xaxis.set_major_formatter(fmtX)
                paxes.get_xaxis().set_major_locator(dateTicLoc)
                paxes.yaxis.set_major_formatter(fmtY)
                
                paxes = py.subplot(3, 2, 2)
                py.plot(time, fitLineT, 'b-')
                py.plot(time, fitLineT + fitSigT, 'b--')
                py.plot(time, fitLineT - fitSigT, 'b--')
                py.errorbar(time, t, yerr=terr, fmt='k.')
                rng = py.axis()
                py.axis(dateTicRng + [rng[2], rng[3]])
                py.xlabel('Date (yrs)')
                py.ylabel('T (pix)')
                paxes.xaxis.set_major_formatter(fmtX)
                paxes.get_xaxis().set_major_locator(dateTicLoc)
                paxes.yaxis.set_major_formatter(fmtY)
                
                paxes = py.subplot(3, 2, 3)
                py.plot(time, np.zeros(len(time)), 'b-')
                py.plot(time, fitSigR, 'b--')
                py.plot(time, -fitSigR, 'b--')
                py.errorbar(time, r - fitLineR, yerr=rerr, fmt='k.')
                py.axis(dateTicRng + resTicRng)
                py.xlabel('Date (yrs)')
                py.ylabel('R Residuals (pix)')
                paxes.get_xaxis().set_major_locator(dateTicLoc)
                
                paxes = py.subplot(3, 2, 4)
                py.plot(time, np.zeros(len(time)), 'b-')
                py.plot(time, fitSigT, 'b--')
                py.plot(time, -fitSigT, 'b--')
                py.errorbar(time, t - fitLineT, yerr=terr, fmt='k.')
                py.axis(dateTicRng + resTicRng)
                py.xlabel('Date (yrs)')
                py.ylabel('T Residuals (pix)')
                paxes.get_xaxis().set_major_locator(dateTicLoc)
                
                bins = np.arange(-7, 7, 1)
                py.subplot(3, 2, 5)
                (n, b, p) = py.hist(sigR, bins)
                py.setp(p, 'facecolor', 'k')
                py.axis([-5, 5, 0, 20])
                py.xlabel('T Residuals (sigma)')
                py.ylabel('Number of Epochs')
                
                py.subplot(3, 2, 6)
                (n, b, p) = py.hist(sigT, bins)
                py.axis([-5, 5, 0, 20])
                py.setp(p, 'facecolor', 'k')
                py.xlabel('Y Residuals (sigma)')
                py.ylabel('Number of Epochs')
                
                py.subplots_adjust(wspace=0.4, hspace=0.4, right=0.95, top=0.95)
                py.show()
                py.savefig(rootDir+'plots/plotStarRadial_' + starName + '.png')
    
        except Exception as e:
            print( 'Star ' + starName + ' not in list' )
            print( e )
           
    title = rootDir.split('/')[-2]
    py.suptitle(title, x=0.5, y=0.97)

    if Nstars == 1:
        py.subplots_adjust(wspace=0.4, hspace=0.4, left = 0.15, bottom = 0.1, right=0.9, top=0.9) 
        py.savefig(rootDir+'plots/plotStar_' + starName + '.png')
    else:
        py.subplots_adjust(wspace=0.6, hspace=0.6, left = 0.08, bottom = 0.05, right=0.95, top=0.90)
        py.show()
        py.savefig(rootDir+'plots/plotStar_all.png')

    py.show()
        
        
        
def CompareTarget(TargetName, root='./', align='align/align_d_rms_1000_abs_t',
                 poly='polyfit_d/fit', useAccFits=False):
    
    s = starset.StarSet(rootDir + align)
    s.loadPolyfit(rootDir + poly, accel=0, arcsec=0)             
    names = s.getArray('name')
    indTarg = names.index(TargetName)
    star = s.stars[indTarg]
    
    pointsFile = root + points + TargetName + '.points'
    if os.path.exists(pointsFile + '.orig'):
        pointsTab = Table.read(pointsFile + '.orig', format='ascii')
    else:
        pointsTab = Table.read(pointsFile, format='ascii')

    # Observed Data
    time = pointsTab[pointsTab.colnames[0]]
    x = pointsTab[pointsTab.colnames[1]]
    y = pointsTab[pointsTab.colnames[2]]
    xerr = pointsTab[pointsTab.colnames[3]]
    yerr = pointsTab[pointsTab.colnames[4]]
    
    
    fitx = star.fitXv
    fity = star.fitYv
    dt = time - fitx.t0
    fitLineX = fitx.p + (fitx.v * dt)
    fitSigX = np.sqrt( fitx.perr**2 + (dt * fitx.verr)**2 )

    fitLineY = fity.p + (fity.v * dt)
    fitSigY = np.sqrt( fity.perr**2 + (dt * fity.verr)**2 )

    resTargX = x - fitLineX
    resTargY = y - fitliney
	
    diffEpX, diffEpY = sigmaVsEpoch(root=root, align=align, poly=poly, useAccFits=useAccFits)


def ResVectorPlot(root='./', align='align/align_t',
                 poly='polyfit_d/fit', points='points_d/', useAccFits=False,
                 TargetName='OB120169',
                 radCut_pix = 10000, magCut = 22):
	
	
    print( 'Creating quiver plot of residuals...' )

    s = starset.StarSet(root + align)
    s.loadStarsUsed()
    s.loadPolyfit(root + poly, accel=0, arcsec=0)

    numEpochs = len(s.years)
    
    try: 
        pointsFile = root + points + TargetName + '.points'
        if os.path.exists(pointsFile + '.orig'):
            pointsTab = Table.read(pointsFile + '.orig', format='ascii')
        else:
            pointsTab = Table.read(pointsFile, format='ascii')
            
        times = pointsTab[pointsTab.colnames[0]]
    except:
        print( 'Star ' + TargetName + ' not in list' )
    
    py.clf()
    py.close(1)
    py.figure(1, figsize=(10, 10))
    for ee in range(numEpochs):
        # Observed data
        x = s.getArrayFromEpoch(ee, 'xpix')
        y = s.getArrayFromEpoch(ee, 'ypix')
        m = s.getArrayFromEpoch(ee, 'mag')
        isUsed = s.getArrayFromEpoch(ee, 'isUsed')
        rad = np.hypot(x - 512, y - 512)

        good = np.where(isUsed == True)
        stars = s.stars
        
        # good = (rad < radCut_pix) & (m < magCut)
        # idx = np.where(good)[0]
        # stars = [s.stars[i] for i in idx]
        # x = x[idx]
        # y = y[idx]
        # rad = rad[idx]
              
    	
        Nstars = len(x)
        x_fit = np.zeros(Nstars, dtype=float)
        y_fit = np.zeros(Nstars, dtype=float)
        residsX = np.zeros(Nstars, dtype=float)
        residsY = np.zeros(Nstars, dtype=float)
        idx2 = []
        for i in range(Nstars):
            fitx = stars[i].fitXv
            fity = stars[i].fitYv 
            StarName = stars[i].name

            dt = times[ee] - fitx.t0
            fitLineX = fitx.p + (fitx.v * dt)
            
            fitSigX = np.sqrt( fitx.perr**2 + (dt * fitx.verr)**2 )

            fitLineY = fity.p + (fity.v * dt)
            fitSigY = np.sqrt( fity.perr**2 + (dt * fity.verr)**2 )

            x_fit[i] = fitLineX
            y_fit[i] = fitLineY
            residsX[i] = x[i] - fitLineX
            residsY[i] = y[i] - fitLineY
        
        idx = np.where((np.abs(residsX) < 10.0) & (np.abs(residsY) < 10.0))[0]
        print ("Trimmed {0:d} stars with too-large residuals (>10 pix)".format(len(idx)))
        py.subplot(3, 3, ee+1)
        py.ylim(0, 1100)
        py.xlim(0, 1100)
        py.yticks(fontsize=10)
        py.xticks([200,400,600,800,1000], fontsize=10)
        # q = py.quiver(x_fit[idx], y_fit[idx], residsX[idx], residsY[idx], scale_units='width', scale=0.5)
        q = py.quiver(x_fit, y_fit, residsX, residsY, scale_units='width', scale=0.5, color='gray')
        q = py.quiver(x_fit[idx], y_fit[idx], residsX[idx], residsY[idx], scale_units='width', scale=0.5, color='black')
        q = py.quiver(x_fit[good], y_fit[good], residsX[good], residsY[good], scale_units='width', scale=0.5, color='red')
        py.quiver([850, 0], [100, 0], [0.05, 0.05], [0, 0], color='red', scale=0.5, scale_units='width')
        py.text(600, 100, '0.5 mas', color='red', fontsize=8)
        # py.quiverkey(q, 0.85, 0.1, 0.02, '0.2 mas', color='red', fontsize=6)

   
    fname = 'quiverplot_all.png'
    py.subplots_adjust(bottom=0.1, right=0.97, top=0.97, left=0.05)	
    if os.path.exists(root + 'plots/' + fname):
        os.remove(root + 'plots/' + fname)
    py.show()
    py.savefig(root + 'plots/' + fname)
            
    return

def check_alignment_fit(root_dir='./', align_root='align/align_t', poly_root='polyfit_d/fit'):
	
    s = starset.StarSet(root_dir + align_root)
    s.loadPolyfit(root_dir + poly_root, accel=0, arcsec=0)
    s.loadStarsUsed()

    x = s.getArrayFromAllEpochs('xpix')
    y = s.getArrayFromAllEpochs('ypix')
    xe_p = s.getArrayFromAllEpochs('xpixerr_p')
    ye_p = s.getArrayFromAllEpochs('ypixerr_p')
    xe_a = s.getArrayFromAllEpochs('xpixerr_a')
    ye_a = s.getArrayFromAllEpochs('ypixerr_a')
    isUsed = s.getArrayFromAllEpochs('isUsed')

    # x0 = s.getArray('fitpXalign.p')
    # vx = s.getArray('fitpXalign.v')
    # t0x = s.getArray('fitpXalign.t0')
    # y0 = s.getArray('fitpYalign.p')
    # vy = s.getArray('fitpYalign.v')
    # t0y = s.getArray('fitpYalign.t0')

    x0 = s.getArray('fitpXv.p')
    vx = s.getArray('fitpXv.v')
    t0x = s.getArray('fitpXv.t0')
    y0 = s.getArray('fitpYv.p')
    vy = s.getArray('fitpYv.v')
    t0y = s.getArray('fitpYv.t0')

    m = s.getArray('mag')
    cnt = s.getArray('velCnt')

    N_epochs = x.shape[0]
    N_stars = x.shape[1]

    # Setup arrays.
    xresid_rms_all = np.zeros(N_epochs, dtype=float)
    yresid_rms_all = np.zeros(N_epochs, dtype=float)
    
    xresid_rms_used = np.zeros(N_epochs, dtype=float)
    yresid_rms_used = np.zeros(N_epochs, dtype=float)
    
    xresid_err_a_all = np.zeros(N_epochs, dtype=float)
    yresid_err_a_all = np.zeros(N_epochs, dtype=float)
    
    xresid_err_a_used = np.zeros(N_epochs, dtype=float)
    yresid_err_a_used = np.zeros(N_epochs, dtype=float)

    xresid_err_p_all = np.zeros(N_epochs, dtype=float)
    yresid_err_p_all = np.zeros(N_epochs, dtype=float)
    
    xresid_err_p_used = np.zeros(N_epochs, dtype=float)
    yresid_err_p_used = np.zeros(N_epochs, dtype=float)
        
    chi2x_all = np.zeros(N_epochs, dtype=float)
    chi2y_all = np.zeros(N_epochs, dtype=float)
    
    chi2x_used = np.zeros(N_epochs, dtype=float)
    chi2y_used = np.zeros(N_epochs, dtype=float)
    
    N_stars_all = np.zeros(N_epochs, dtype=float)
    N_stars_used = np.zeros(N_epochs, dtype=float)
    
    year = np.zeros(N_epochs, dtype=float)
    
    idx = np.where(cnt > 2)[0]
    N_stars_3ep = len(idx)

    for ee in range(N_epochs):
        idx = np.where((cnt > 2) & (x[ee, :] > -1000) & (xe_p[ee, :] > 0))[0]
        used = np.where(isUsed[ee, idx] == True)

        # Everything below should be arrays sub-indexed by "idx"
        dt_x = s.years[ee] - t0x[idx]
        dt_y = s.years[ee] - t0y[idx]

        x_fit = x0[idx] + (vx[idx] * dt_x)
        y_fit = y0[idx] + (vy[idx] * dt_y)

        xresid = x[ee, idx] - x_fit
        yresid = y[ee, idx] - y_fit
        
        N_stars_all[ee] = len(xresid)
        N_stars_used[ee] = len(xresid[used])

        # Note this chi^2 only includes positional errors.
        chi2x_terms = xresid**2 / xe_p[ee, idx]**2
        chi2y_terms = yresid**2 / ye_p[ee, idx]**2

        xresid_rms_all[ee] = np.sqrt(np.mean(xresid**2))
        yresid_rms_all[ee] = np.sqrt(np.mean(yresid**2))
        
        xresid_rms_used[ee] = np.sqrt(np.mean(xresid[used]**2))
        yresid_rms_used[ee] = np.sqrt(np.mean(yresid[used]**2))

        xresid_err_p_all[ee] = xe_p[ee, idx].mean() / N_stars_all[ee]**0.5
        yresid_err_p_all[ee] = ye_p[ee, idx].mean() / N_stars_all[ee]**0.5

        xresid_err_p_used[ee] = xe_p[ee, idx][used].mean() / N_stars_used[ee]**0.5
        yresid_err_p_used[ee] = ye_p[ee, idx][used].mean() / N_stars_used[ee]**0.5
        
        xresid_err_a_all[ee] = xe_a[ee, idx].mean() / N_stars_all[ee]**0.5
        yresid_err_a_all[ee] = ye_a[ee, idx].mean() / N_stars_all[ee]**0.5

        xresid_err_a_used[ee] = xe_a[ee, idx][used].mean() / N_stars_used[ee]**0.5
        yresid_err_a_used[ee] = ye_a[ee, idx][used].mean() / N_stars_used[ee]**0.5
        
        chi2x_all[ee] = chi2x_terms.sum()
        chi2y_all[ee] = chi2y_terms.sum()

        chi2x_used[ee] = chi2x_terms[used].sum()
        chi2y_used[ee] = chi2y_terms[used].sum()

        year[ee] = s.years[ee]


    data = {'xres_rms_all': xresid_rms_all, 'yres_rms_all': yresid_rms_all,
            'xres_rms_used': xresid_rms_used, 'yres_rms_used': yresid_rms_used,
            'xres_err_p_all': xresid_err_p_all, 'yres_err_p_all': yresid_err_p_all, 
            'xres_err_p_used': xresid_err_p_used, 'yres_err_p_used': yresid_err_p_used, 
            'xres_err_a_all': xresid_err_a_all, 'yres_err_a_all': yresid_err_a_all, 
            'xres_err_a_used': xresid_err_a_used, 'yres_err_a_used': yresid_err_a_used, 
            'chi2x_all': chi2x_all, 'chi2y_all': chi2y_all,
            'chi2x_used': chi2x_used, 'chi2y_used': chi2y_used,
            'N_stars_all': N_stars_all, 'N_stars_used': N_stars_used,
            'year': year, 'N_stars_3ep': N_stars_3ep}
        
    return data

def sum_all_stars(root='./', align='align/align_t',
                    poly='polyfit_d/fit', points='points_d/',
                    youngOnly=False, trimOutliers=False, trimSigma=4,
                    useAccFits=False, magCut=None, radCut=None, target = 'ob110022'):
    """Analyze the distribution of points relative to their best
    fit velocities. Optionally trim the largest outliers in each
    stars *.points file.  Optionally make a magnitude cut with
    magCut flag and/or a radius cut with radCut flag."""
    
    s = starset.StarSet(root + align)
    s.loadPolyfit(root + poly, accel=0, arcsec=0)

    #####
    # 
    # Make Cuts to Sample of Stars
    #
    #####
    # Set default (none) cuts.
    if magCut == None:
        magCut = 100   
    if radCut == None:
        radCut = 100

    # Get som parameters we will cut on.
    mag = s.getArray('mag')
    x = s.getArray('x')
    y = s.getArray('y')
    r = np.hypot(x, y)

    # Trim out stars.
    idx = np.where((mag < magCut) & (r < radCut))[0]
    newstars = []
    for i in idx:
        newstars.append(s.stars[i])
    s.stars = newstars

    # Get arrays we want to plot    
    mag = s.getArray('mag')
    x = s.getArray('x')
    y = s.getArray('y')
    r = np.hypot(x, y)
    chi2red_x = s.getArray('fitXv.chi2red')
    chi2red_y = s.getArray('fitYv.chi2red')
    chi2_x = s.getArray('fitXv.chi2')
    chi2_y = s.getArray('fitYv.chi2')
    ve_x = s.getArray('fitXv.verr')
    ve_y = s.getArray('fitYv.verr')

    scale = 9.952  # mas/pixel
    
    ve_x_mas = ve_x * scale
    ve_y_mas = ve_y * scale
    
    #####
    # Plot Velocity Errors vs. Kp
    #####    
    markersize = 3.0 
    xmin = 12
    xmax = 22
    ymin = 0.06
    ymax = 1.0

    py.close(1)
    fig1 = py.figure(1)
    py.semilogy(mag, ve_x_mas, 'ro', markersize=markersize, label='X')
    py.semilogy(mag, ve_y_mas, 'bo', markersize=markersize, label='Y')
    py.xlim(xmin, xmax)
    py.ylim(ymin, ymax)
    py.ylabel('Proper Motion Uncertinaty [mas/yr]', fontsize=16)
    py.xlabel('K magnitude', fontsize=16)
    py.legend(loc='upper left')
    py.yticks(fontsize=16)
    py.xticks(fontsize=16)
    py.show()
    py.savefig(root + 'plots/velErr_vs_mag_xy.png')

    py.clf()
    py.semilogy(mag, (ve_x_mas + ve_y_mas) / 2.0, 'ro', markersize=markersize)
    py.xlim(xmin, xmax)
    py.ylim(ymin, ymax)
    py.ylabel('Proper Motion Uncertainty [mas/yr]', fontsize=16)
    py.xlabel('K magnitude', fontsize=16)
    py.yticks(fontsize=16)
    py.xticks(fontsize=16)
    py.show()
    py.savefig(root + 'plots/velErr_vs_mag.png')
	
    #####
    # Plot Velocity Errors vs. radius
    #####    
    xmin = 0
    xmax= 5.0

    py.clf()    
    py.semilogy(r, ve_x_mas, 'ro', markersize=markersize, label='X')
    py.semilogy(r, ve_y_mas, 'bo', markersize=markersize, label='Y')
    py.ylim(ymin, ymax)
    py.xlim(xmin, xmax)
    py.ylabel('Proper Motion Uncertainty [mas/yr]', fontsize=16)
    py.xlabel('Radius (")', fontsize=16)
    py.yticks(fontsize=16)
    py.xticks(fontsize=16)
    py.legend(loc='upper left')
    py.show()
    py.savefig(root+'plots/velErr_vs_rad.png')


    #####	
	# Plot histogram of reduced chi square values 
    #####	
    py.clf()
    bins = np.arange(0, 30, 1.0)
    (nx, bx, pntx) = py.hist(chi2red_x, bins, label='X', histtype='step')
    (ny, by, pnty) = py.hist(chi2red_y, bins, label='Y', histtype='step')
    py.xlabel(r'$\chi^2_{reduced}$')
    py.ylabel('Number of Stars')
    py.legend(loc='upper left')
    py.show()
    py.savefig(root+'plots/chi2red_xy_hist.png')

    #####	
	# Plot histogram of reduced chi square values 
    #####	
    py.clf()
    bins = np.arange(0, np.max([chi2_y, chi2_x]), 4.0)
    (nx, bx, px) = py.hist(chi2_x, bins, label='X', histtype='step')
    (nx, bx, px) = py.hist(chi2_y, bins, label='Y', histtype='step')
    py.xlabel('$\chi^2$')
    py.ylabel('Number of Stars')
    xarr = np.linspace(0, 100, 1000)
    df = 4
    dist = scipy.stats.chi2(df, 0)
    chiplot = scipy.stats.chi2.pdf(xarr, dist.pdf(xarr))
    chiplot = 7 * chiplot / np.max(chiplot)
    py.plot(xarr, chiplot, '--', label='Model')
    py.legend()
    py.show()
    py.savefig(root+'plots/chi2_xy_hist.png')

    #####
    # Plot histogram of chi square values, x & y combined 
    #####
    py.clf()
    bins = np.arange(0, np.max(chi2_y + chi2_x), 5.0)
    (nx, bx, px) = py.hist(chi2_x + chi2_y, bins, histtype='step')
    py.xlabel('Total $\chi^2$', fontsize=16)
    df = 30
    xarr = np.linspace(0.1, 100, 1000)
    dist = scipy.stats.chi2(df, 0)
    chiplot = scipy.stats.chi2.pdf(xarr, dist.pdf(xarr))
    chiplot = 10 * chiplot / np.max(chiplot)
    py.plot(xarr, chiplot, '--r')
    py.ylabel('Number of Stars', fontsize=16)
    py.xlim(0, np.round(np.max(chi2_y) * 1.2))
    py.ylim(0, np.max(nx) + 3)
    py.show()
    py.savefig(root+'plots/chi2_hist.png')


    ##########
    #
    # Loop through all the stars and combine their residuals.
    #
    ##########
	
    # Make some empty arrays to hold all our results.
    sigmaX = np.arange(0, dtype=float)
    sigmaY = np.arange(0, dtype=float)
    sigma  = np.arange(0, dtype=float)
    diffX_all = np.arange(0, dtype=float)
    diffY_all = np.arange(0, dtype=float)
    xerr_all = np.arange(0, dtype=float)
    yerr_all = np.arange(0, dtype=float)
    chisq_all = np.arange(0, dtype=float)
    
    for star in s.stars:
        starName = star.name
        
        pointsFile = root + points + starName + '.points'
        if os.path.exists(pointsFile + '.orig'):
            pointsTab = np.genfromtxt(pointsFile + '.orig')
        else:
            pointsTab = np.genfromtxt(pointsFile)

        # Observed Data
        t = pointsTab[:, 0]
        x = pointsTab[:, 1]
        y = pointsTab[:, 2]
        xerr = pointsTab[:, 3]
        yerr = pointsTab[:, 4]

        # Best fit velocity model
        fitx = star.fitXv
        fity = star.fitYv

        dt = t - fitx.t0
        fitLineX = fitx.p + (fitx.v * dt)
        fitSigX = np.sqrt( fitx.perr**2 + (dt * fitx.verr)**2 )

        fitLineY = fity.p + (fity.v * dt)
        fitSigY = np.sqrt( fity.perr**2 + (dt * fity.verr)**2 )

        # Residuals
        diffX = x - fitLineX
        diffY = y - fitLineY
        diff = np.hypot(diffX, diffY)
        rerr = np.sqrt((diffX*xerr)**2 + (diffY*yerr)**2) / diff
        sigX = diffX / xerr
        sigY = diffY / yerr
        sig = diff / rerr

        idxX = (np.where(abs(sigX) > trimSigma))[0]
        idxY = (np.where(abs(sigY) > trimSigma))[0]
        idx  = (np.where(abs(sig) > trimSigma))[0]
         
        if ((trimOutliers == True) and (len(idx) > 0)):
            if not os.path.exists(pointsFile + '.orig'):
                shutil.copyfile(pointsFile, pointsFile + '.orig')

            for ii in idx[::-1]:
                pointsTab.delete(ii)

            pointsTab.writeto(pointsFile)

        # Combine this stars information with all other stars.
        sigmaX = np.concatenate((sigmaX, sigX))
        sigmaY = np.concatenate((sigmaY, sigY))
        sigma = np.concatenate((sigma, sig))
        diffX_all = np.concatenate((diffX_all,diffX))
        diffY_all = np.concatenate((diffY_all,diffY))
        xerr_all = np.concatenate((xerr_all,xerr))
        yerr_all = np.concatenate((yerr_all,yerr))
        chisq_all = np.concatenate((chisq_all, np.array([sum((diffX/xerr)**2.) + np.sum((diffY/yerr)**2.)])))

    # for i in range(len(chisq_all)):
    #     print( chisq_all[i] )

    diff_all = np.concatenate((diffX_all, diffY_all))
    err_all = np.concatenate((xerr_all, yerr_all))
    sigma_all = np.concatenate((sigmaX, sigmaY))   
    py.errorbar(range(len(diff_all)), diff_all, err_all, fmt='o')
    py.show()
    py.savefig(root+'plots/TEST_resids.pdf', bbox_inches='tight', pad_inches=0.1)
    py.clf()
    py.plot(range(len(diff_all)), sigma_all, 'o')
    py.show()
    py.savefig(root+'plots/TEST_resids_sigma.pdf', bbox_inches='tight', pad_inches=0.1)
    py.clf()
    py.plot(range(len(chisq_all)), chisq_all, 'o')
    py.show()
    py.savefig(root+'plots/TEST_resids_chisq.pdf', bbox_inches='tight', pad_inches=0.1)
    py.clf()
    

    rmsDiffXY = (diffX_all.std() + diffY_all.std()) / 2.0 * 1000.0
    aveDiffR = np.sqrt(diffX_all**2 + diffY_all**2).mean()
    medDiffR = np.median(np.sqrt(diffX_all**2 + diffY_all**2))

    print('mean(diffX) = {0:7.3f} mas    mean(diffY) = {1:7.3f} mas'.format(diffX_all.mean()*1e3, diffY_all.mean()*1e3) )
    print('stdv(diffX) = {0:7.3f} mas    stdv(diffY) = {1:7.3f} mas'.format(diffX_all.std()*1e3, diffY_all.std()*1e3) )
    print(' RMS diffXY = {0:7.3f} mas      AVE diffR = {1:7.3f} mas     MED diffR = {2:7.3f} mas'.format(rmsDiffXY * 1e3, aveDiffR*1e3, medDiffR*1e3 ))
    print('   MED xerr = {0:7.3f} mas       MED yerr = {1:7.3f} mas'.format( np.median(xerr_all)*1e3, np.median(yerr_all)*1e3 ))

    # Residuals should have a gaussian probability distribution
    # with a mean of 0 and a sigma of 1. Overplot this to be sure.
    ggx = np.arange(-7, 7, 0.25)
    ggy = scipy.stats.norm.pdf(ggx, 0, 1)

    print( 'Mean   RMS residual: %5.2f sigma' % (sigma.mean()) )
    print( 'Stddev RMS residual: %5.2f sigma' % (sigma.std()) )
    print( 'Median RMS residual: %5.2f sigma' % (np.median(sigma)) )
    print( '' )
    print( 'Mean X centroiding error: %5.4f mas (median %5.4f mas)' % 
            ((xerr_all*1000.0).mean(), np.median(xerr_all)*10**3))
    print( 'Mean Y centroiding error: %5.4f mas (median %5.4f mas)' % 
            ((yerr_all*1000.0).mean(), np.median(yerr_all)*10**3))
    print( 'Mean distance from velocity fit: %5.4f mas (median %5.4f mas)' % 
            ((aveDiffR*10**3, medDiffR*10**3)))

    ##########
    # Plot
    ##########
    bins = np.arange(-7, 7, 1.0)
    figsize = (10, 10)
    fig4 = py.figure(figsize=figsize)
    ax = fig4.add_subplot(3, 1, 1)
    (nx, bx, px) = ax.hist(sigmaX, bins)
    ggamp = ((np.sort(nx))[-2:]).sum() / (2.0 * ggy.max())
    ax.plot(ggx, ggy*ggamp, 'k-')
    py.xlabel('X Residuals (sigma)')

    ax2 = fig4.add_subplot(3, 1, 2)
    (ny, by, pnty) = ax2.hist(sigmaY, bins)
    ggamp = ((np.sort(ny))[-2:]).sum() / (2.0 * ggy.max())
    ax2.plot(ggx, ggy*ggamp, 'k-')
    py.xlabel('Y Residuals (sigma)')

    ax3 = fig4.add_subplot(3, 1, 3)
    (ny, by, pnty) = ax3.hist(sigma, np.arange(0, 7, 0.5))
    py.xlabel('Total Residuals (sigma)')

    py.subplots_adjust(wspace=0.34, hspace=0.33, right=0.95, top=0.97)
    py.show()
    py.savefig(root+'plots/residualsDistribution_pub.pdf')
    py.savefig(root+'plots/residualsDistribution_pub.png')
    py.clf()
    
    # Put all residuals together in one histogram
    fig5 = py.figure(figsize=[6,6])
    ax = fig5.add_subplot(1,1,1)
    sigmaA = []
    for ss in range(len(sigmaX)):
        sigmaA = np.concatenate([sigmaA,[sigmaX[ss]]])
        sigmaA = np.concatenate([sigmaA,[sigmaY[ss]]])
    (na, ba, pa) = ax.hist(sigmaA, bins, color='b')
    ggamp = ((np.sort(na))[-2:]).sum() / (2.0 * ggy.max())
    ax.plot(ggx, ggy*ggamp, 'k-')
    py.xlabel('Residuals ($\sigma$)')
    py.ylabel('Frequency')
    py.xlim(-6,6)
    py.show()
    py.savefig(root+'plots/residualsAll_pub.png')
    print( 'Saved ' + root + 'plots/residualsAll_pub.png')
    py.clf()


def chi2_dist_all_epochs(align_root, root_dir='./', poly_root='polyfit_d/fit',
                         points_dir = 'points_d/', only_stars_in_fit=False):
    """
    Plot the complete chi^2 distribution of all stars
    positions at all epochs. Restrict down to just those
    stars detected in all epochs. Only divide by the
    positional error. This is the metric for how good
    our alignment + velocity fitting was and whether
    we used the right sample and right order.
    """

    # Load up the list of stars used in the transformation.	
    s = starset.StarSet(root_dir + align_root)
    s.loadPolyfit(root_dir + poly_root, accel=0, arcsec=0)
    s.loadPoints(root_dir + points_dir)
    s.loadStarsUsed()

    trans = Table.read(root_dir + align_root + '.trans', format='ascii')
    N_par = trans['NumParams'][0]
    
    # Keep only stars detected in all epochs.
    cnt = s.getArray('velCnt')
    used = s.getArray('isUsed')
    N_epochs = cnt.max()

    idx = np.where(cnt == N_epochs)[0]
    msg = 'Keeping {0:d} stars in all epochs'

    if only_stars_in_fit:
        isUsed = s.getArrayFromAllEpochs('isUsed')
        cnt_used = isUsed.sum(axis=0)
        idx = np.where(cnt_used == cnt_used.max())[0]

        msg += ' and used'
        
    newstars = [s.stars[i] for i in idx]
    s.stars = newstars
    print( msg.format(len(idx)) )

    # Now that we have are final list of stars, fetch all the
    # relevant variables.     
    cnt = s.getArray('velCnt')

    x = s.getArrayFromAllEpochs('xpix')
    y = s.getArrayFromAllEpochs('ypix')
    xe_p = s.getArrayFromAllEpochs('xpixerr_p')
    ye_p = s.getArrayFromAllEpochs('ypixerr_p')
    xe_a = s.getArrayFromAllEpochs('xpixerr_a')
    ye_a = s.getArrayFromAllEpochs('ypixerr_a')
    isUsed = s.getArrayFromAllEpochs('isUsed')

    # x02 = s.getArray('fitpXalign.p')
    # vx2 = s.getArray('fitpXalign.v')
    # t0x2 = s.getArray('fitpXalign.t0')
    # y02 = s.getArray('fitpYalign.p')
    # vy2 = s.getArray('fitpYalign.v')
    # t0y2 = s.getArray('fitpYalign.t0')

    x0 = s.getArray('fitpXv.p')
    vx = s.getArray('fitpXv.v')
    t0x = s.getArray('fitpXv.t0')
    y0 = s.getArray('fitpYv.p')
    vy = s.getArray('fitpYv.v')
    t0y = s.getArray('fitpYv.t0')

    m = s.getArray('mag')

    N_epochs = x.shape[0]
    N_stars = x.shape[1]
    year = np.zeros(N_epochs, dtype=float)

    # Make some output variables with the right shape/size.
    chi2x = np.zeros((N_epochs, N_stars), dtype=float)
    chi2y = np.zeros((N_epochs, N_stars), dtype=float)
    chi2 = np.zeros((N_epochs, N_stars), dtype=float)

    xresid = np.zeros((N_epochs, N_stars), dtype=float)
    yresid = np.zeros((N_epochs, N_stars), dtype=float)
    resid = np.zeros((N_epochs, N_stars), dtype=float)

    for ee in range(N_epochs):
        # Everything below should be arrays sub-indexed by "idx"
        dt_x = s.years[ee] - t0x
        dt_y = s.years[ee] - t0y

        x_fit = x0 + (vx * dt_x)
        y_fit = y0 + (vy * dt_y)

        xresid[ee, :] = x[ee, :] - x_fit
        yresid[ee, :] = y[ee, :] - y_fit

    # Note this chi^2 only includes positional errors.
    chi2x = ((xresid / xe_p)**2).sum(axis=0)
    chi2y = ((yresid / ye_p)**2).sum(axis=0)
    chi2 = chi2x + chi2y

    # Total residuals for each star.
    resid = np.hypot(xresid, yresid)

    # Total error for each star.
    xye_p = np.hypot(xresid * xe_p, yresid * ye_p) / resid
    xye_a = np.hypot(xresid * xe_a, yresid * ye_a) / resid

    # Figure out the number of degrees of freedom expected
    # for this chi^2 distribution.
    # N_data = N_stars * N_epochs * 2.0   # times 2 for X and Y measurements
    # N_free = N_par * N_epochs * 2.0
    N_data = N_epochs * 2.0   # times 2 for X and Y measurements
    N_free = 4.0
    N_dof = N_data - N_free
    N_dof_1 = N_dof / 2.0
    print( 'N_data = {0:.0f}  N_free = {1:.0f}  N_dof = {2:.0f}'.format(N_data, N_free, N_dof))

    # Setup some bins for making chi2 and residuals histograms
    chi2_lim = int(np.ceil(chi2[np.isfinite(chi2)].max()))
    if chi2_lim > (N_dof * 20):
        chi2_lim = N_dof * 20
    chi2_bin = chi2_lim / 20.0
    chi2_bins = np.arange(0, chi2_lim, chi2_bin)
    chi2_mod_bin = 0.25
    chi2_mod_bins = np.arange(0, chi2_lim, chi2_mod_bin)

    res_lim = 1.1 * resid.max()
    if res_lim > 1:
        res_lim = 1
    res_bin = 2.0 * res_lim / 25.0
    res_bins = np.arange(-res_lim, res_lim + res_bin, res_bin)

    sig_lim = 1.1 * (resid / xye_p).max()
    if sig_lim > 6:
        sig_lim = 6
    sig_bin = 2.0 * sig_lim / 20.0
    sig_bins = np.arange(-sig_lim, sig_lim + sig_bin, sig_bin)
    sig_mod_bin = sig_bin / 10.0
    sig_mod_bins = np.arange(-sig_lim, sig_lim + sig_mod_bin, sig_mod_bin)

    # Setup theoretical chi^2 distributions for X, Y, total.
    chi2_dist_a = scipy.stats.chi2(N_dof)
    chi2_dist_1 = scipy.stats.chi2(N_dof_1)
    chi2_plot_a = chi2_dist_a.pdf(chi2_mod_bins)
    chi2_plot_1 = chi2_dist_1.pdf(chi2_mod_bins)
    chi2_plot_a *= N_stars / chi2_mod_bin
    chi2_plot_1 *= N_stars / chi2_mod_bin

    # Setup theoretical normalized residuals distribution for X and Y
    sig_plot_1 = scipy.stats.norm.pdf(sig_mod_bins)
    sig_plot_1 *= N_stars * N_epochs / (sig_plot_1 * sig_mod_bin).sum()

    ##########
    # Plot Chi^2 Distribution
    ##########
    py.clf()
    fig = py.gcf()
    fig.set_size_inches(12, 12, forward=True)
    py.subplots_adjust(bottom=0.08, left=0.08, top=0.95, wspace=0.28, hspace=0.32)

    ax_chi2 = py.subplot(3, 3, 1)
    py.hist(chi2x, bins=chi2_bins, color='blue')
    py.plot(chi2_mod_bins, chi2_plot_1, 'k--')
    py.xlabel(r'$\chi^2$')
    py.ylabel(r'N$_{obs}$')
    py.title('X')

    py.subplot(3, 3, 2, sharex=ax_chi2, sharey=ax_chi2)
    py.hist(chi2y, bins=chi2_bins, color='green')
    py.plot(chi2_mod_bins, chi2_plot_1, 'k--')
    py.xlabel(r'$\chi^2$')
    py.title('Y')

    py.subplot(3, 3, 3, sharex=ax_chi2, sharey=ax_chi2)
    py.hist(chi2.flatten(), bins=chi2_bins, color='red')
    py.plot(chi2_mod_bins, chi2_plot_a, 'k--')
    py.xlabel(r'$\chi^2$')
    py.title('X and Y')

    ax_res = py.subplot(3, 3, 4)
    py.hist(xresid.flatten(), bins=res_bins, color='blue')
    py.xlabel('X Residuals (mas)')
    py.ylabel(r'N$_{obs}$')

    py.subplot(3, 3, 5, sharex=ax_res, sharey=ax_res)
    py.hist(yresid.flatten(), bins=res_bins, color='green')
    py.xlabel('Y Residuals (mas)')

    py.subplot(3, 3, 6, sharey=ax_res)
    py.hist(resid.flatten(), bins=res_bins, color='red')
    py.xlim(0, res_lim)
    py.xlabel('Total Residuals (mas)')

    ax_nres = py.subplot(3, 3, 7)
    py.hist((xresid / xe_p).flatten(), bins=sig_bins, color='blue')
    py.plot(sig_mod_bins, sig_plot_1, 'k--')
    py.xlabel('Normalized X Res.')

    py.subplot(3, 3, 8, sharex=ax_nres, sharey=ax_nres)
    py.hist((yresid / ye_p).flatten(), bins=sig_bins, color='green')
    py.plot(sig_mod_bins, sig_plot_1, 'k--')
    py.xlabel('Normalized Y Res.')

    py.subplot(3, 3, 9, sharey=ax_nres)
    py.hist((resid / xye_p).flatten(), bins=sig_bins, color='red')
    py.xlim(0, sig_lim)
    py.xlabel('Normalized Total Res.')

    fileUtil.mkdir(root_dir + 'plots/')

    outfile = root_dir + 'plots/'
    outfile += 'chi2_dist_all_epochs'
    if only_stars_in_fit:
        outfile += '_infit'
    outfile += '.png'
    print( outfile)

    py.show()
    py.savefig(outfile)


    return_val = {'chi2x': chi2x, 'chi2y': chi2y, 'chi2': chi2,
                  'xres': xresid, 'yres': yresid, 'res': resid,
                  'xe_p':   xe_p, 'ye_p':   ye_p, 'xye_p': xye_p,
                  'xe_a':   xe_a, 'ye_a':   ye_a, 'xye_a': xye_a,
                  'Ndof_x': N_dof_1, 'Ndof_y': N_dof_1, 'Ndof': N_dof}
    
    return return_val

    
