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

    py.close('all')
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
        
            print 'Star:        ', starName
            print '\tX Chi^2 = %5.2f (%6.2f for %2d dof)' % \
                  (fitx.chi2red, fitx.chi2, fitx.chi2/fitx.chi2red)
            print '\tY Chi^2 = %5.2f (%6.2f for %2d dof)' % \
                  (fity.chi2red, fity.chi2, fity.chi2/fity.chi2red)
            # print 'X  Outliers: ', time[idxX]
            # print 'Y  Outliers: ', time[idxY]
            # if (radial):
            #     print 'R  Outliers: ', time[idxX]
            #     print 'T  Outliers: ', time[idxY]
            # print 'XY Outliers: ', time[idx]
        
            # close(2)
#             figure(2, figsize=(7, 8))
#             clf()
            
            dateTicLoc = py.MultipleLocator(3)
            dateTicRng = [2006, 2010]
            dateTics = np.array([2011, 2012, 2013, 2014, 2015, 2016])
    	    DateTicsLabel = dateTics-2000
    	    
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
            py.axis('equal')
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
                py.savefig(rootDir+'plots/plotStarRadial_' + starName + '.png')
    
            # suptitle(starNames, fontsize=12)
#             subplots_adjust(wspace=0.4, hspace=0.4, right=0.95, top=0.95)
#             savefig(rootDir+'plots/plotStar_' + 'all2' + '.eps')
#             savefig(rootDir+'plots/plotStar_' + 'all2' + '.png')
        except Exception as e:
            print 'Star ' + starName + ' not in list'
            print e
           
#     suptitle(starNames, fontsize=12)
    title = rootDir.split('/')[-2]
    py.suptitle(title, x=0.5, y=0.97)

    if Nstars == 1:
        py.subplots_adjust(wspace=0.4, hspace=0.4, left = 0.15, bottom = 0.1, right=0.9, top=0.9) 
        py.savefig(rootDir+'plots/plotStar_' + starName + '.png')
    else:
        py.subplots_adjust(wspace=0.6, hspace=0.6, left = 0.08, bottom = 0.05, right=0.95, top=0.90)
        py.savefig(rootDir+'plots/plotStar_' + 'all' + '.png')
        
        
        
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
                 numEpochs=7, TargetName='OB120169_R',
                 radCut_pix = 10000, magCut = 22):
	
	
    print 'Creating quiver plot of residuals...'

    s = starset.StarSet(root + align)
    s.loadPolyfit(root + poly, accel=0, arcsec=0)             
    
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
    py.figure(1, figsize=(10, 10))
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
        
        idx = np.where((np.abs(residsX) < 10.0) & (np.abs(residsY) < 10.0))[0]
        py.subplot(3, 3, ee+1)
        py.ylim(0, 1100)
        py.xlim(0, 1100)
        py.yticks(fontsize=10)
        py.xticks([200,400,600,800,1000], fontsize=10)
        q = py.quiver(x_fit[idx], y_fit[idx], residsX[idx], residsY[idx], scale_units='width', scale=0.5)
        py.quiver([850, 0], [100, 0], [0.05, 0.05], [0, 0], color='red', scale=0.5, scale_units='width')
        py.text(600, 100, '0.5 mas', color='red', fontsize=8)
        # py.quiverkey(q, 0.85, 0.1, 0.02, '0.2 mas', color='red', fontsize=6)

   
    fname = 'quiverplot_all.png'
    py.subplots_adjust(bottom=0.1, right=0.97, top=0.97, left=0.05)	
    if os.path.exists(root + 'plots/' + fname):
        os.remove(root + 'plots/' + fname)
    py.savefig(root + 'plots/' + fname)
            
    return

def check_alignment_fit(align_root, root_dir='./'):
	
    s = starset.StarSet(root_dir + align_root)
    s.loadStarsUsed()

    x = s.getArrayFromAllEpochs('xpix')
    y = s.getArrayFromAllEpochs('ypix')
    xe_p = s.getArrayFromAllEpochs('xpixerr_p')
    ye_p = s.getArrayFromAllEpochs('ypixerr_p')
    xe_a = s.getArrayFromAllEpochs('xpixerr_a')
    ye_a = s.getArrayFromAllEpochs('ypixerr_a')
    isUsed = s.getArrayFromAllEpochs('isUsed')

    x0 = s.getArray('fitpXalign.p')
    vx = s.getArray('fitpXalign.v')
    t0x = s.getArray('fitpXalign.t0')
    y0 = s.getArray('fitpYalign.p')
    vy = s.getArray('fitpYalign.v')
    t0y = s.getArray('fitpYalign.t0')

    m = s.getArray('mag')
    cnt = s.getArray('velCnt')

    N_epochs = x.shape[0]
    N_stars = x.shape[1]

    xresid_rms_all = np.zeros(N_epochs, dtype=float)
    yresid_rms_all = np.zeros(N_epochs, dtype=float)
    
    xresid_rms_used = np.zeros(N_epochs, dtype=float)
    yresid_rms_used = np.zeros(N_epochs, dtype=float)
    
    xresid_err_a_all = np.zeros(N_epochs, dtype=float)
    yresid_err_a_all = np.zeros(N_epochs, dtype=float)
    
    xresid_err_p_used = np.zeros(N_epochs, dtype=float)
    yresid_err_p_used = np.zeros(N_epochs, dtype=float)

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
            'xres_err_all': xresid_err_all, 'yres_err_all': yresid_err_all, 
            'xres_err_used': xresid_err_used, 'yres_err_used': yresid_err_used, 
            'chi2x_all': chi2x_all, 'chi2y_all': chi2y_all,
            'chi2x_used': chi2x_used, 'chi2y_used': chi2y_used,
            'N_stars_all': N_stars_all, 'N_stars_used': N_stars_used,
            'year': year}
        
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
    r = hypot(x, y)

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
    r = hypot(x, y)
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
    ymin = 6e-2
    ymax = 1e0

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
    py.savefig(root + 'plots/velErr_vs_mag_xy.eps')
    py.savefig(root + 'plots/velErr_vs_mag_xy.png')

    py.clf()
    py.semilogy(mag, (ve_x_mas + ve_y_mas) / 2.0, 'ro', markersize=markersize)
    py.xlim(xmin, xmax)
    py.ylim(ymin, ymax)
    py.ylabel('Proper Motion Uncertainty [mas/yr]', fontsize=16)
    py.xlabel('K magnitude', fontsize=16)
    py.yticks(fontsize=16)
    py.xticks(fontsize=16)
    py.savefig(root + 'plots/velErr_vs_mag.eps')
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
    py.savefig(root+'plots/velErr_vs_rad.eps')
    py.savefig(root+'plots/velErr_vs_rad.png')


    #####	
	# Plot histogram of reduced chi square values 
    #####	
    py.clf()
    bins = np.arange(0, 30, 1.0)
    (nx, bx, px) = py.hist(chi2red_x, bins, label='X', histtype='step')
    (ny, by, py) = py.hist(chi2red_y, bins, label='Y', histtype='step')
    py.xlabel(r'$\chi^2_{reduced}$')
    py.ylabel('Number of Stars')
    py.legend(loc='upper left')
    py.savefig(root+'plots/chi2red_xy_hist.eps')
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
    chiplot = chi2.pdf(xarr, dist.pdf(xarr))
    chiplot = 7 * chiplot / np.max(chiplot)
    py.plot(xarr, chiplot, '--', label='Model')
    py.legend()
    py.savefig(root+'plots/chi2_xy_hist.eps')
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
    chiplot = chi2.pdf(xarr, dist.pdf(xarr))
    chiplot = 10 * chiplot / np.max(chiplot)
    py.plot(xarr, chiplot, '--r')
    py.ylabel('Number of Stars', fontsize=16)
    py.xlim(0, np.round(np.max(chi2_y) * 1.2))
    py.ylim(0, np.max(nx) + 3)
    py.savefig(root+'plots/chi2_hist.eps')
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
            pointsTab = asciidata.open(pointsFile + '.orig')
        else:
            pointsTab = asciidata.open(pointsFile)

        # Observed Data
        t = pointsTab[0].tonumpy()
        x = pointsTab[1].tonumpy()
        y = pointsTab[2].tonumpy()
        xerr = pointsTab[3].tonumpy()
        yerr = pointsTab[4].tonumpy()

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
        diff = hypot(diffX, diffY)
        rerr = sqrt((diffX*xerr)**2 + (diffY*yerr)**2) / diff
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
        sigmaX = concatenate((sigmaX, sigX))
        sigmaY = concatenate((sigmaY, sigY))
        sigma = concatenate((sigma, sig))
        diffX_all = concatenate((diffX_all,diffX))
        diffY_all = concatenate((diffY_all,diffY))
        xerr_all = concatenate((xerr_all,xerr))
        yerr_all = concatenate((yerr_all,yerr))
        chisq_all = concatenate((chisq_all, np.array([sum((diffX/xerr)**2.) + sum((diffY/yerr)**2.)])))

    for i in range(len(chisq_all)):
        print mag_all[i], chisq_all[i]

    diff_all = concatenate((diffX_all, diffY_all))
    err_all = concatenate((xerr_all, yerr_all))
    sigma_all = concatenate((sigmaX, sigmaY))   
    errorbar(range(len(diff_all)), diff_all, err_all, fmt='o')
    savefig(root+'plots/TEST_resids.pdf', bbox_inches='tight', pad_inches=0.1)
    clf()
    plot(range(len(diff_all)), sigma_all, 'o')
    savefig(root+'plots/TEST_resids_sigma.pdf', bbox_inches='tight', pad_inches=0.1)
    clf()
    plot(range(len(chisq_all)), chisq_all, 'o')
    savefig(root+'plots/TEST_resids_chisq.pdf', bbox_inches='tight', pad_inches=0.1)
    clf()
    

    rmsDiffXY = (diffX_all.std() + diffY_all.std()) / 2.0 * 1000.0
    aveDiffR = np.sqrt(diffX_all**2 + diffY_all**2).mean()
    medDiffR = np.median(np.sqrt(diffX_all**2 + diffY_all**2))

    print diffX_all.mean(), diffY_all.mean()
    print diffX_all.std(), diffY_all.std()
    print rmsDiffXY, aveDiffR, medDiffR
    print np.median(xerr_all)

    # Residuals should have a gaussian probability distribution
    # with a mean of 0 and a sigma of 1. Overplot this to be sure.
    ggx = np.arange(-7, 7, 0.25)
    ggy = normpdf(ggx, 0, 1)

    print 'Mean   RMS residual: %5.2f sigma' % (sigma.mean())
    print 'Stddev RMS residual: %5.2f sigma' % (sigma.std())
    print 'Median RMS residual: %5.2f sigma' % (median(sigma))
    print
    print 'Mean X centroiding error: %5.4f mas (median %5.4f mas)' % \
        ((xerr_all*1000.0).mean(), np.median(xerr_all)*10**3)
    print 'Mean Y centroiding error: %5.4f mas (median %5.4f mas)' % \
        ((yerr_all*1000.0).mean(), np.median(yerr_all)*10**3)
    print 'Mean distance from velocity fit: %5.4f mas (median %5.4f mas)' % \
        (aveDiffR*10**3, medDiffR*10**3)

    ##########
    # Plot
    ##########
    bins = np.arange(-7, 7, 1.0)
    fig4 = figure(figsize=figsize)
    ax = fig4.add_subplot(3, 1, 1)
    (nx, bx, px) = ax.hist(sigmaX, bins)
    ggamp = ((sort(nx))[-2:]).sum() / (2.0 * ggy.max())
    ax.plot(ggx, ggy*ggamp, 'k-')
    xlabel('X Residuals (sigma)')

    ax2 = fig4.add_subplot(3, 1, 2)
    (ny, by, py) = ax2.hist(sigmaY, bins)
    ggamp = ((sort(ny))[-2:]).sum() / (2.0 * ggy.max())
    ax2.plot(ggx, ggy*ggamp, 'k-')
    xlabel('Y Residuals (sigma)')

    ax3 = fig4.add_subplot(3, 1, 3)
    (ny, by, py) = ax3.hist(sigma, np.arange(0, 7, 0.5))
    xlabel('Total Residuals (sigma)')

    subplots_adjust(wspace=0.34, hspace=0.33, right=0.95, top=0.97)
    savefig(root+'plots/residualsDistribution_pub.pdf')
    savefig(root+'plots/residualsDistribution_pub.png')
    clf()
    
    # Put all residuals together in one histogram
    fig5 = figure(figsize=[6,6])
    ax = fig5.add_subplot(1,1,1)
    sigmaA = []
    for ss in range(len(sigmaX)):
        sigmaA = np.concatenate([sigmaA,[sigmaX[ss]]])
        sigmaA = np.concatenate([sigmaA,[sigmaY[ss]]])
    (na, ba, pa) = ax.hist(sigmaA, bins, color='b')
    ggamp = ((sort(na))[-2:]).sum() / (2.0 * ggy.max())
    ax.plot(ggx, ggy*ggamp, 'k-')
    xlabel('Residuals ($\sigma$)')
    ylabel('Frequency')
    xlim(-6,6)
    print 'hi'
    savefig(root+'plots/residualsAll_pub.eps')
    savefig(root+'plots/residualsAll_pub.png')
    clf()
