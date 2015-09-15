import shutil, os, sys
import pylab as py
import numpy as np
import scipy
import scipy.stats
from jlu.gc.gcwork import starset
from jlu.gc.gcwork import starTables
from astropy.table import Table
import sys
import pdb

def plotStar(starNames, rootDir='./', align='align/align_d_rms_1000_abs_t',
             poly='polyfit_d/fit', points='points_d/', radial=False, NcolMax=3, figsize=(15,15)):

    print 'Creating residuals plots for star(s):'
    print starNames
    
    s = starset.StarSet(rootDir + align)
    s.loadPolyfit(rootDir + poly, accel=0, arcsec=0)
    Nstars = len(starNames)
    Ncols = 2 * np.min([Nstars, NcolMax])

    if Nstars <= Ncols/2:
        Nrows = 3
    else:
        Nrows = (Nstars // (Ncols / 2)) * 3
            
    py.clf()
    py.close(2)
    py.figure(2, figsize=figsize)
    names = s.getArray('name')
    mag = s.getArray('mag')
    x = s.getArray('x')
    y = s.getArray('y')
    r = np.hypot(x,y)
#     idx = np.where((mag < 22.5) & (r <= 4.0))[0]
#     newstars = []
#     for i in idx:
#         newstars.append(s.stars[i])
#     s.stars = newstars
    
    for i in range(Nstars):
    
        starName = starNames[i]
        
        try:
            
            ii = names.index(starName)
            star = s.stars[ii]
       
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
        
    #         print 'Star:        ', starName
#             print '\tX Chi^2 = %5.2f (%6.2f for %2d dof)' % \
#                   (fitx.chi2red, fitx.chi2, fitx.chi2/fitx.chi2red)
#             print '\tY Chi^2 = %5.2f (%6.2f for %2d dof)' % \
#                   (fity.chi2red, fity.chi2, fity.chi2/fity.chi2red)
#             print 'X  Outliers: ', time[idxX]
#             print 'Y  Outliers: ', time[idxY]
#             if (radial):
#                 print 'R  Outliers: ', time[idxX]
#                 print 'T  Outliers: ', time[idxY]
#             print 'XY Outliers: ', time[idx]
        
            # close(2)
#             figure(2, figsize=(7, 8))
#             clf()
            
            dateTicLoc = py.MultipleLocator(3)
            dateTicRng = [2006, 2010]
            dateTics = np.array([2011, 2012, 2013, 2014, 2015, 2016])
    	    DateTicsLabel = dateTics-2000
    	    
            maxErr = np.array([xerr, yerr]).max()
            resTicRng = [-3*maxErr, 3*maxErr]
        
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
            py.ylabel('X (pix)', fontsize=fontsize1)
            #paxes.get_xaxis().set_major_locator(dateTicLoc)
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
            paxes.tick_params(axis='both', which='major', labelsize=fontsize1)
            paxes.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
            paxes.xaxis.set_major_formatter(FormatStrFormatter('%.2f'))
            py.xlabel('X (pix)', fontsize=fontsize1)
            py.ylabel('Y (pix)', fontsize=fontsize1)
           #  names_fit = np.genfromtxt('polyfit_d/fit.linearFormal', 
#                                                            usecols=(0), comments=None, 
#                                                            unpack=True, dtype=None, skip_header=1) 
#             Xint = np.genfromtxt('polyfit_d/fit.linearFormal', 
#                                                            usecols=(1), comments=None, 
#                                                            unpack=True, dtype=None, skip_header=1)
#             mX = np.genfromtxt('polyfit_d/fit.linearFormal', 
#                                                            usecols=(2), comments=None, 
#                                                            unpack=True, dtype=None, skip_header=1)
#             Yint = np.genfromtxt('polyfit_d/fit.linearFormal', 
#                                                            usecols=(7), comments=None, 
#                                                            unpack=True, dtype=None, skip_header=1)
#             mY = np.genfromtxt('polyfit_d/fit.linearFormal', 
#                                                            usecols=(8), comments=None, 
#                                                            unpack=True, dtype=None, skip_header=1)                                               
#            
#             t0X = np.genfromtxt('polyfit_d/fit.lt0', 
#                                                            usecols=(1), comments=None, 
#                                                            unpack=True, dtype=None, skip_header=1)
#             t0Y = np.genfromtxt('polyfit_d/fit.lt0', 
#                                                            usecols=(2), comments=None, 
#                                                            unpack=True, dtype=None, skip_header=1)
#                                                             
            
           #  ind = np.where(names_fit == starName)[0][0]
    
 #            xyfit_X = [] 
#             xyfit_Y = []
#             for n in range(len(x)):
#                 xyfit_X.append(Xint[ind] + mX[ind]*(time[n]-t0X[ind]))
#                 xyfit_Y.append(Yint[ind] + mY[ind]*(time[n]-t0Y[ind]))
            py.plot(fitLineX, fitLineY, 'b-')    
            
            col = col + 1
    	    ind = (row-1)*Ncols + col

            bins = np.arange(-7, 7, 1)
#             py.subplot2grid((3,2*Nstars),(2, 2*i))
            paxes = py.subplot(Nrows, Ncols, ind)
#             subplot(3, 2, 5)
            id = np.where(diffY < 0)[0]
            sig[id] = -1.*sig[id] 
            (n, b, p) = py.hist(sigX, bins, histtype='stepfilled', color='b')
            py.setp(p, 'facecolor', 'b')
            (n, b, p) = py.hist(sigY, bins, histtype='step', color='r')
            py.axis([-7, 7, 0, 8], fontsize=10)
            py.xlabel('X Residuals (sigma)', fontsize=fontsize1)
            py.ylabel('Number of Epochs', fontsize=fontsize1)


#             bins = np.arange(-7, 7, 1)
# #             subplot2grid((3,2*Nstars),(2, 2*i))
#             paxes = subplot(Nrows, Ncols, ind)
# #             subplot(3, 2, 5)
#             (n, b, p) = hist(sigX, bins)
#             setp(p, 'facecolor', 'k')
#             axis([-5, 5, 0, 20], fontsize=10)
#             xlabel('X Residuals (sigma)', fontsize=fontsize1)
#             ylabel('Number of Epochs', fontsize=fontsize1)
# #         	show()
#     	    
#     	    col = col + 1
#     	    ind = row*np.min([NcolMax,i+1]) + col
#     	    ind = (row-1)*Ncols + col
#     
# #         	subplot2grid((3,2*Nstars),(2, 2*i+1))
#             paxes = subplot(Nrows, Ncols, ind)
# #             subplot(3, 2, 6)
#             (n, b, p) = hist(sigY, bins)
#             axis([-5, 5, 0, 20], fontsize=10)
#             setp(p, 'facecolor', 'k')
#             xlabel('Y Residuals (sigma)', fontsize=fontsize1)
#             ylabel('Number of Epochs', fontsize=fontsize1)
# #             show()
        
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
                py.savefig(rootDir+'plots/plotStarRadial_' + starName + '.eps')
                py.savefig(rootDir+'plots/plotStarRadial_' + starName + '.png')
    
            # suptitle(starNames, fontsize=12)
#             subplots_adjust(wspace=0.4, hspace=0.4, right=0.95, top=0.95)
#             savefig(rootDir+'plots/plotStar_' + 'all2' + '.eps')
#             savefig(rootDir+'plots/plotStar_' + 'all2' + '.png')
        except Exception as e:
            print 'Star ' + starName + ' not in list'
            print e
           
#     suptitle(starNames, fontsize=12)
    if Nstars == 1:
        py.subplots_adjust(wspace=0.4, hspace=0.4, left = 0.15, bottom = 0.1, right=0.9, top=0.9) 
        py.savefig(rootDir+'plots/plotStar_' + starName + '.pdf')
        py.savefig(rootDir+'plots/plotStar_' + starName + '.png')
    else:
        py.subplots_adjust(wspace=0.6, hspace=0.6, left = 0.08, bottom = 0.05, right=0.95, top=0.95)
        py.savefig(rootDir+'plots/plotStar_' + 'all' + '.pdf')
        #py.savefig(rootDir+'plots/plotStar_' + 'all' + '.png')
        
        
        
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
                 poly='polyfit_d/fit', points='points_d/', useAccFits=False, numEpochs=5, TargetName='ob120169_R',
                 radCut_pix = 10000, magCut = 22):
	
	
    print 'Creating quiver plot of residuals...'

    s = starset.StarSet(root + align)
    s.loadPolyfit(root + poly, accel=0, arcsec=0)             
#    names = s.getArray('name')
#    indTarg = names.index(TargetName)
    
    try: 
        pointsFile = root + points + TargetName + '.points'
        if os.path.exists(pointsFile + '.orig'):
            pointsTab = Table.read(pointsFile + '.orig', format='ascii')
        else:
            pointsTab = Table.read(pointsFile, format='ascii')
            
        times = pointsTab[pointsTab.colnames[0]]
    except:
        print 'Star ' + TargetName + ' not in list'
    
    py.clf()
    # for ee in range(1):
    py.close(1)
    py.figure(1, figsize=(10,7))
    for ee in range(numEpochs):
        
        # Observed data
        x = s.getArrayFromEpoch(ee, 'xpix')
        y = s.getArrayFromEpoch(ee, 'ypix')
        m = s.getArrayFromEpoch(ee, 'mag')
    	rad = np.hypot(x - 512,y - 512)

        good = (rad < radCut_pix) & (m < magCut)
        idx = np.where(good)[0]
        stars = [s.stars[i] for i in idx]
        x = x[idx]
        y = y[idx]
        rad = rad[idx]
              
    	
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
            if ((StarName == 'star_151') or (StarName == 'star_160') or (StarName == 'star_164')
                or (StarName == 'star_166') or (StarName == 'star_168') or (StarName == 'star_181')):
                idx2.append(i)
        # idx = np.where((np.abs(residsX) < 0.3) & (np.abs(residsY) < 0.3))[0]
#         subplot(2,3,ee+1)
#         print 'hwkjekjwjkweew'
#         q = quiver(x_fit[idx], y_fit[idx], residsX[idx], residsY[idx])
#         quiverkey(q, 0.9, 0.9, 0.1, '0.1 pix', color='red')
        
        idx = np.where((np.abs(residsX) < 10.0) & (np.abs(residsY) < 10.0))[0]
#         print idx2
        py.subplot(2,3,ee+1)
        py.ylim(0,1100)
        py.xlim(0,1100)
        py.yticks(fontsize=10)
        py.xticks([200,400,600,800,1000],fontsize=10)
        q = py.quiver(x_fit[idx], y_fit[idx], residsX[idx], residsY[idx], scale_units='width', scale =0.5)
#         q = py.quiver(x_fit[idx2], y_fit[idx2], residsX[idx2], residsY[idx2], scale_units='width', scale =0.5, color ='orange')
        py.quiverkey(q, 0.85, 0.1, 0.02, '0.2 mas', color='red', fontproperties={'size': 6})
   
    
    fname = 'quiverplot_all.pdf'
    py.subplots_adjust(bottom=0.1, right=0.97, top=0.97, left=0.05)	
    if os.path.exists(root + 'plots/' + fname):
        os.remove(root + 'plots/' + fname)
    py.savefig(root + 'plots/' + fname)
            
    return
