import numpy as np
import pylab as py
from astropy.io import fits as pyfits
import pickle
import time
import math, glob
import scipy
import pdb
import pp
import itertools
import pandas
import astropy
from scipy.optimize import curve_fit
import os
from gcwork import objects
import warnings
import matplotlib as mpl
import colormaps as cmaps
import datetime
from jlu.m31 import ppxf_m31,voronoi_m31

workdir = '/Users/kel/Documents/Projects/M31/analysis_new/ifu_11_11_30/'
datadir = workdir
mctmpdir = workdir+'tmp_mc/'
modeldir = '/Users/kel/Documents/Projects/M31/models/Peiris/2003/'
contmpdir = workdir+'tmp_convert/'
modelworkdir = '/Users/kel/Documents/Projects/M31/analysis_new/modeling/'

#cuberoot = 'm31_all_semerr'
#cuberoot = 'm31_all'
#cuberoot = 'm31_all_halforgerr'
#cuberoot = 'm31_all_seventherr'
#cuberoot = 'm31_all_scalederr'
#cuberoot = 'm31_all_scalederr_cleanhdr'
cuberoot = 'm31_all_scalederr_cleanhdr_bulgesub_vorcube_20160825'

cc = objects.Constants()

# BH positions in arcsec
bhpos = ppxf_m31.bhpos
bhpos_pix = bhpos/0.05
bhpos_hor = ppxf_m31.bhpos_hor

def modelConvertBin(testAngles=None,testSMBH=None,l98bin=False,verbose=True,tess=True,pathstem=modeldir+'nonaligned_grid_rotate/',inVorTxt=None,PSFfile=None,precess=None,outPrecessFile=None):
    # results similar to running all three of ppxf_m31.modelConvertCoordinates,
    # ppxf_m31.modelOSIRISrotation, and ppxf_m31.modelBin. In that set of routines,
    # model stellar particles with positions and velocities in disk-plane coordinates
    # are (1) converted to sky-plane coordinates, (2) rotated to match the OSIRIS
    # observations, and (3) binned to match the OSIRIS observations, LOS velocity
    # histograms fit, and the Gauss-Hermite expansion fit to the histogram to get
    # the LOSVD ensemble kinematics.

    # In this implementation, all three routines are condensed to one in order to speed
    # up computation time. Coordinate transformations are done using the tensordot function
    # without using tmp files. Both coordinate transformations are done here, along
    # with the binning and LOSVD fitting. Test angle rotations can be passed in as a keyword,
    # in the format testAngles=[thetaL, thetaI, thetaA, thetaPA].

    # binning is automatically done to match OSIRIS pixel scale (=0.05") but set keyword
    # l98bin=True to bin to match Lauer 1998 data (=0.0228")

    # smoothing has been added before binning, so no need to do it after

    # added option to add a precession speed (in units of km/s/pc) to the disk-plane velocities (x and y)

    t1 = time.time()
    
    # read in nonaligned model file; units are in pc, BH at origin
    inputFile = modeldir + 'nonaligned_model.dat'
    model = pandas.read_csv(inputFile,delim_whitespace=True,header=None,names=['x','y','z','v_x','v_y','v_z'])

    if verbose:
        print "Starting rotations"
    if testAngles:
        thetaL = np.radians(testAngles[0])
        thetaI = np.radians(testAngles[1])
        thetaA = np.radians(testAngles[2])
        thetaCPA = np.radians(testAngles[3])
        if verbose:
            print "Rotation angles are theta_L = %.1f, theta_I =  %.1f, theta_A =  %.1f, theta_CPA =  %.1f" % (testAngles[0],testAngles[1],testAngles[2],testAngles[3])
    else:
        # using the nonaligned angles from Peiris & Tremaine 2003
        thetaL = np.radians(-42.8)
        thetaI = np.radians(54.1)
        thetaA = np.radians(-34.5)
        thetaCPA = np.radians(-56.)

    # stack positions to begin the coordinate transform
    xyz = np.column_stack((model.x,model.y,model.z))
    
    # add precession to v_x and v_y, if requested
    if precess is not None:
        vxp, vyp = velPrecess(model.x, model.y, model.v_x, model.v_y, precess)
        # stack velocities, using the modified v_x and v_y
        vxyz = np.column_stack((vxp, vyp, model.v_z))
        if outPrecessFile is not None:
            np.savetxt(outPrecessFile, np.c_[model.x, model.y, model.z, vxp, vyp, model.z],fmt=('%8.6f','%8.6f','%8.6f','%8.6f','%8.6f','%8.6f'),delimiter='\t')
    else:
        # if no precession, stack velocities 
        vxyz = np.column_stack((model.v_x,model.v_y,model.v_z))

    # matrices to rotate from disk plane to sky plane (P&T 2003, eq 4)
    matL = np.matrix([[np.cos(thetaL),-np.sin(thetaL),0.],[np.sin(thetaL),np.cos(thetaL),0.],[0.,0.,1.]])
    matI = np.matrix([[1.,0.,0.],[0.,np.cos(thetaI),-np.sin(thetaI)],[0.,np.sin(thetaI),np.cos(thetaI)]])
    matA = np.matrix([[np.cos(thetaA),-np.sin(thetaA),0.],[np.sin(thetaA),np.cos(thetaA),0.],[0.,0.,1.]])

    # these are rotation matrices, so can do the constant part of the dot product separately
    matLIA = matL.dot(matI).dot(matA)

    # transform coordinates from disk plane to sky plane
    bigXYZ = np.tensordot(matLIA, xyz, axes=([1], [1]))
    bigV_XYZ = np.tensordot(matLIA, vxyz, axes=([1], [1]))

    # transpose to right shape
    bigXYZ = bigXYZ.T
    bigV_XYZ = bigV_XYZ.T

    # matrix to rotate in sky plane to OSIRIS position angle
    rotMat = np.matrix([[np.cos(thetaCPA),-np.sin(thetaCPA)],[np.sin(thetaCPA),np.cos(thetaCPA)]])

    # only need to rotate X and Y (Z remains along LOS)
    XY = bigXYZ[:,0:2]
    V_XY = bigV_XYZ[:,0:2]

    outXY = np.tensordot(rotMat, XY, axes=([1], [1]))
    outV_XY = np.tensordot(rotMat, V_XY, axes=([1], [1]))

    outXY = outXY.T
    outV_XY = outV_XY.T

    # strip everything into individual arrays
    X = outXY[:,0]
    Y = outXY[:,1]
    Z = bigXYZ[:,2]
    V_X = outV_XY[:,0]
    V_Y = outV_XY[:,1]
    outV_Z = bigV_XYZ[:,2]
    # by definition, LOS velocity has the opposite sign of V_Z
    losV = -1.*outV_Z

    if testSMBH:
        # if offsets have been set for the SMBH position, add them to the X and Y positions
        # thus the "real" position of the SMBH is offset by offx and offy from the position
        # that will be used as the SMBH position
        # offsets are in units of pixels, but X and Y are in units of pc
        offx = testSMBH[0]*0.05*3.73
        offy = testSMBH[1]*0.05*3.73
        X += offx
        Y += offy

    # smooth in OSIRIS space (X and Y only, though V_X and V_Y could be
    # treated similarly) by adding a bit of noise (random additive in with
    # st dev of the PSF of the seeing halo)
    nPart = len(X)
    if PSFfile is None:
        #PSFfile = workdir+'plots/osir_perf_m31_all_scalederr_cleanhdr_params.txt'
        PSFfile = '/Users/kel/Documents/Projects/M31/data/osiris_mosaics/drf/sigclip/all_NIRC2_CC_DTOTOFF_2/data/osiris_perf/osir_perf_m31_2125nm_w_osiris_rot_scale_params.txt_params.txt'
    PSFparams = ppxf_m31.readPSFparams(inputFile=PSFfile,twoGauss=True)
    # grab sigma of the PSF in units of pixels and convert to arcsec
    PSFsig = PSFparams.sig1[0]*0.05
    # convert to units of pc
    pcsig = PSFsig * 3.73
    # get the randomized offsets from the original coordinates
    dx=np.random.normal(0.,pcsig,nPart)
    dy=np.random.normal(0.,pcsig,nPart)
    # apply the random offsets to the positions
    smX=X+dx
    smY=Y+dy
    
    #pdb.set_trace()
    
    if verbose:
        print "Starting binning"

    if tess:
        # performs the tessellated binning on the models *before* doing the LOSVD fits
        # this function needs to read in a file from disk so first writing out the
        # transformed coordinates so it can grab those
        # writing out V_Z, *not* LOS V as the sign flip is taken care of in the tessellation routine
        if verbose:
            print "Tessellating and binning models"
        # use the smoothed version of X and Y as inputs to the binning
        posvel = np.column_stack((smX,smY,Z,V_X,V_Y,outV_Z))
        outFile = pathstem+'nonaligned_OSIRIScoords_fit_full_smooth_tess.dat'
        if testAngles:
            if precess is None:
                PP = 0.0
            else:
                PP = precess
            outFile=pathstem+'nonaligned_OSIRIScoords_fit_full_smooth_%.1f_%.1f_%.1f_%.1f_%.1f.dat' % (testAngles[0],testAngles[1],testAngles[2],testAngles[3],PP)
        if testSMBH:
            outFile=pathstem+'nonaligned_OSIRIScoords_fit_full_smooth_SMBH_x%+.1f_y%+.1f.dat' % (testSMBH[0],testSMBH[1])
        if inVorTxt is None:
            voronoi_m31.tessModels(inputModelArr=posvel,inputVoronoiFile=datadir+'voronoi_2d_binning_output_20160825.txt',l98bin=l98bin,outFile=outFile)
        else:
            voronoi_m31.tessModels(inputModelArr=posvel,inputVoronoiFile=inVorTxt,l98bin=l98bin,outFile=outFile)

    else:
        # preparing to do the model binning
        # OSIRIS bin size = 0.05" = 0.1865 pc
        # if matching L98 photometry, use 0.0228" = 0.08504 pc
        if l98bin:
            binpc = 0.08504
            binas = 0.0228
        else:
            binpc = 0.1865
            binas = 0.05

        # Setting the BH pixel phase to match that of the data
        xfrac = bhpos_pix[0]-np.floor(bhpos_pix[0])
        yfrac = bhpos_pix[1]-np.floor(bhpos_pix[1])
        # if L98 bin size, divide by the ratio of the two pixel scales and
        # calc the new pixel phase
        if l98bin:
            xfrac = (xfrac*(.05/.0228))-np.floor(xfrac*(0.05/0.0228))
            yfrac = (yfrac*(.05/.0228))-np.floor(yfrac*(0.05/0.0228))
        # reposition BH (originally at origin in model) to the correct pixel phase
        X += (binpc*xfrac)
        Y += (binpc*yfrac)
    
        # get the full size of the binned array, but making sure to leave bin boundaries on the axes
        # positive and negative extent of the x axis
        posxbin = np.ceil(np.max(X)/binpc)
        negxbin = np.ceil(np.abs(np.min(X)/binpc))
        nxbin = posxbin+negxbin
        # and y axis
        posybin = np.ceil(np.max(Y)/binpc)
        negybin = np.ceil(np.abs(np.min(Y)/binpc))
        nybin = posybin + negybin

        # new BH position: (0,0) + (xfrac,yfrac)
        modbhpos = [negxbin+xfrac,negybin+yfrac]
        if verbose:
            print "Model BH is at ", modbhpos
        #pdb.set_trace()

        # trim to only cover the OSIRIS FOV
        newnegxbin = 0. - np.floor(bhpos_pix[0])
        newnegybin = 0. - np.floor(bhpos_pix[1])

        xlen = 41
        ylen = 82
        goodTrim = np.where((X/binpc >= newnegxbin) & (X/binpc <= (newnegxbin + xlen)) & (Y/binpc >= newnegybin) & (Y/binpc <= (newnegybin + ylen)))

        xClip = X[goodTrim[0]] - newnegxbin*binpc
        yClip = Y[goodTrim[0]] - newnegybin*binpc
        zClip = Z[goodTrim[0]]
        vxClip = V_X[goodTrim[0]]
        vyClip = V_Y[goodTrim[0]]
        losVClip = losV[goodTrim[0]]

        # convert x,y positions to bin numbers
        xBin = np.floor(xClip/binpc)
        yBin = np.floor(yClip/binpc)
        xClippc = xClip/binpc
        yClippc = yClip/binpc

        # initializing flux and kinematic arrays
        modShape = (xlen,ylen)

        newNstar = np.zeros(modShape, dtype=float)
        fitlosv = np.zeros(modShape, dtype=float)
        sigma = np.zeros(modShape, dtype=float)
        h3 = np.zeros(modShape, dtype=float)
        h4 = np.zeros(modShape, dtype=float)

        xbins = np.arange(xlen+1)
        ybins = np.arange(ylen+1)

        # binning LOS velocity in bins of 5 km/s, with cuts at +/- 1000 km/s
        vzbins = np.arange(-1000., 1005., 5.)
    
        # create the histogram and count up the number of particles in each velocity bin
        bins = (xbins, ybins, vzbins)
        losvd, bins_new, bin_num = scipy.stats.binned_statistic_dd((xClippc, yClippc, losVClip),
                                                                losVClip,
                                                                statistic='count',
                                                                bins=bins)

        # create the histogram and count up the number of particles in each spatial bin
        nstar, bins_new2, bin_num2 = scipy.stats.binned_statistic_dd((xClippc, yClippc),
                                                                    losVClip,
                                                                    statistic='count',
                                                                    bins=(xbins, ybins))

        # this loop does the LOSVD fits for each bin, to get the kinematics
        for i in range(int(xlen)):
            #print "Starting column ", i
            for j in range(int(ylen)):
                # if there are no particles, set everything to zero
                if nstar[i,j] == 0:
                    fitlosv[i,j] = 0.
                    sigma[i,j] = 0.
                    h3[i,j] = 0.
                    h4[i,j] = 0.
                # if there are too few particles to perform a fit, hack something together 
                elif losvd[i,j,:].max() <= 6:
                    fitlosv[i,j] = (losvd[i,j,:]*bins_new[2][0:-1]).sum()/nstar[i,j]
                    sigma[i,j] = 0.
                    h3[i,j] = 0.
                    h4[i,j] = 0.
                # for everything else, fit the LOSVD histogram
                else:
                    # set the initial values
                    gamma0 = nstar[i,j]
                    # approximate the average velocity of the particles
                    v0 = (losvd[i,j,:]*bins_new[2][0:-1]).sum()/nstar[i,j]
                    # pull out the standard deviation directly from the histogram
                    s0 = np.sqrt(((bins_new[2][0:-1]-v0)**2.*losvd[i,j,:]).sum()/nstar[i,j])
                    h3_0 = 0.
                    h4_0 = 0.
                    guess = [gamma0, v0, s0, h3_0, h4_0]
                    try:
                        popt, pcov = curve_fit(ppxf_m31.gaussHermite, bins_new[2][0:-1], losvd[i,j,:], p0=guess)
                        # popt = [gamma, v, sigma, h3, h4]
                        fitlosv[i,j] = popt[1]
                        sigma[i,j] = popt[2]
                        h3[i,j] = popt[3]
                        h4[i,j] = popt[4]
                    except (RuntimeError):
                        # if no fit is possible, take the mean
                        fitlosv[i,j] = v0
                        sigma[i,j] = 0.
                        h3[i,j] = 0.
                        h4[i,j] = 0.    
    
        # smooth
        # this section hasn't been tested...
        fitStack = [nstar,fitlosv,sigma,h3,h4]
        smoothMod = ppxf_m31.smoothModels(inModArr=fitStack)
        nstar = smoothMod[0][:][:]
        fitlosv = smoothMod[1][:][:]
        sigma = smoothMod[2][:][:]
        h3 = smoothMod[3][:][:]
        h4 = smoothMod[4][:][:]

        #pdb.set_trace()
                
        if testAngles:
            outFile = pathstem + 'nonaligned_OSIRIScoords_fit_full_smooth_%.1f_%.1f_%.1f_%.1f.dat' % (testAngles[0],testAngles[1],testAngles[2],testAngles[3])
        else:
            outFile = pathstem + 'nonaligned_OSIRIScoords_fits_full_smooth_test.dat'
            
        output = open(outFile, 'w')
        pickle.dump(nstar, output)
        pickle.dump(fitlosv, output)
        pickle.dump(sigma, output)
        pickle.dump(h3, output)
        pickle.dump(h4, output)
        output.close()

    if verbose:
        print "Time elapsed: ", time.time() - t1, "s"

def velPrecess(xpos=None, ypos=None, invx=None, invy=None, precess=None):
    # takes an input precession speed (in km/s/pc), disk plane positions,
    # and disk plane velocities (x and y) and adds the precession to the
    # velocities. Outputs the modified velocities
    
    # xpos and ypos in pc, invx and invy in km/s, rotation axis at the origin
    # positive precession assumed to be counterclockwise (right handed)

    # find the angle of the vector orthogonal to the position vector
    # to take the right handed orthogonal vector, x' = -y, y' = x
    theta = np.abs(np.arctan((-1.*ypos)/xpos))

    # get the precession at each radius
    # first get the radius (in pc)
    r = np.sqrt(xpos**2 + ypos**2)
    # then the precession at the radius (km/s) (=tangential velocity)
    vtan = precess*r

    # find the x and y components of the tangential velocity
    # make sure the sign matches that of the original vector, e.g. vtanx has the same sign as -ypos
    # and vtany has the same sign as +xpos. Leave 0 as 0
    vtanx = vtan * np.sin(theta) * ((-1.*((-1.*ypos)<0)) + (1.*((-1.*ypos)>0)))
    vtany = vtan * np.cos(theta) * ((-1*(xpos<0)) + (1.*(xpos>0)))

    # add to the original velocity
    outvx = invx + vtanx
    outvy = invy + vtany

    #pdb.set_trace()

    return outvx, outvy

def modelCompRes(inModel=modeldir+'nonaligned_OSIRIScoords_test_fits_trim_tess.dat',toPlot=False,inData=None,incubeimg=None):
    # routine to compare model outputs to data
    # uses existing routines to read in fitted models
    # and takes the residuals. Plotting is done if requested

    #ppxf_m31.smoothModels(inModel)
    #inSmooth = inModel.replace('.dat','_smooth.dat')
    if inData is None:
        inData = '/Users/kel/Documents/Projects/M31/analysis_new/ifu_11_11_30/ppxf_tess_bs_2_20160825.dat'
    ppxf_m31.modelResiduals(inModel,inData,trimTess=True,incubeimg=incubeimg)

    inRes = inModel.replace('.dat','_residuals.dat')

    if toPlot:
        outtmp = inModel.replace('nonaligned_model_OSIRIScoords_fits_trim_tess','residuals')
        outplotfile = outtmp.replace('.dat','')
        ppxf_m31.plotDataModelResiduals(inData,inModel,inRes,trimTess=True,outStem=outplotfile)
    
        
def modelMorphFitGrid(inFolder=None,plotOnly=False,incubeimg=None,inVorTxt=None,inDataFile=None):
    # routine to conduct a grid search over the coordinate transformation angles
    # (angles in Peiris & Tremaine 2003 eq 4)

    if incubeimg is None:
        cubeimg = pyfits.getdata(datadir + cuberoot + '_img.fits')
    else:
        cubeimg = pyfits.getdata(incubeimg)
    xaxis = np.arange(cubeimg.shape[0]) * 0.05
    yaxis = np.arange(cubeimg.shape[1]) * 0.05

    # original angles
    Lorg = -42.8
    Iorg = 54.1
    Aorg = -34.5
    # not changing CPA
    CPA = -56.

    # angle step size, in degrees
    dAng = 15.
    # number of steps away from the original angle in each direction (-/+)
    # halfN = 1 gives 3 total steps, =2 gives 5 steps
    halfN = 1.
    N = (halfN*2)+1

    Lgrid = np.arange(Lorg-(halfN*dAng),Lorg+(halfN*dAng)+1,dAng)
    Igrid = np.arange(Iorg-(halfN*dAng),Iorg+(halfN*dAng)+1,dAng)
    Agrid = np.arange(Aorg-(halfN*dAng),Aorg+(halfN*dAng)+1,dAng)

    # precession grid
    #Pgrid = np.array([-100.,-30.,0.,30.,100.])
    #Pgrid = np.array([-30.,0.,30.])
    #Pgrid = np.array([-20.,-10.,10.,20.])
    #Pgrid = np.array([-5.,5.])
    Pgrid = np.array([0.])
        
    if not plotOnly:
        # create the new folder
        FORMAT = '%Y%m%d_%H-%M-%S'
        now = datetime.datetime.now().strftime(FORMAT)
        newfolder = modeldir+'nonaligned_grid_rotate_'+now+'/'
        os.mkdir(newfolder)
        #newfolder='/Users/kel/Documents/Projects/M31/models/Peiris/2003/nonaligned_grid_rotate_20160919_17-09-05/'

        for LL in Lgrid:
            for II in Igrid:
                for AA in Agrid:
                    for PP in Pgrid:
                        testAng = [LL,II,AA,CPA]
                        modelConvertBin(testAngles=testAng,l98bin=False,verbose=False,tess=True,pathstem=newfolder,inVorTxt=inVorTxt,precess=PP)
                        modelCompRes(inModel=newfolder+'nonaligned_OSIRIScoords_fit_full_smooth_%.1f_%.1f_%.1f_%.1f_%.1f.dat' % (testAng[0],testAng[1],testAng[2],testAng[3],PP),toPlot=False,inData=inDataFile,incubeimg=incubeimg)
                        modelChiSq(inCubeImg=incubeimg,inData=inDataFile,inModel=newfolder+'nonaligned_OSIRIScoords_fit_full_smooth_%.1f_%.1f_%.1f_%.1f_%.1f.dat' % (testAng[0],testAng[1],testAng[2],testAng[3],PP))

    # plotting
    py.close(2)
    py.figure(2)
    py.subplots_adjust(left=0.05, right=0.94, top=0.95)
    nPlot = N**2
    # flux plots
    for AA in Agrid:
        for PP in Pgrid:
            py.clf()
            plLL = np.fliplr([Lgrid])[0]
            pI,pL = np.meshgrid(Igrid,plLL)
            pI = pI.flatten()
            pL = pL.flatten()
            for pp in np.arange(nPlot):
                py.subplot(N,N,pp+1)
                #inRes = newfolder+'nonaligned_OSIRIScoords_fit_full_smooth_%.1f_%.1f_%.1f_%.1f.dat' % (pL[pp],pI[pp],AA,CPA)
                if inFolder:
                    inRes = inFolder + 'nonaligned_OSIRIScoords_fit_full_smooth_%.1f_%.1f_%.1f_%.1f_%.1f_residuals.dat' % (pL[pp],pI[pp],AA,CPA,PP)
                else:
                    inRes = newfolder+'nonaligned_OSIRIScoords_fit_full_smooth_%.1f_%.1f_%.1f_%.1f_%.1f_residuals.dat' % (pL[pp],pI[pp],AA,CPA,PP)
                res = ppxf_m31.modelFitResults(inRes)
                py.imshow(np.rot90(py.ma.masked_where(cubeimg==0.,res.nstar.T),3),vmin=-0.25,vmax=0.25,
                        extent=[xaxis[0], xaxis[-1], yaxis[0], yaxis[-1]])
                pltitle = '$\\theta_l$ = %.1f, $\\theta_i$ = %.1f' % (pL[pp],pI[pp])
                py.title(pltitle)
                py.xticks([])
                py.yticks([])
            cbar = py.colorbar(orientation='horizontal',ticks=[-.25,0.,.25])
            clab = 'Flux residuals, $\\theta_a$ = %.1f, v$_{precess}$ = %.1f' % (AA,PP)
            cbar.set_label(clab)
            if inFolder:
                figname = inFolder + 'residuals_flux_thetaA_%.1f_%.1f.png' %(AA,PP)
            else:
                figname=newfolder + 'residuals_flux_thetaA_%.1f_%.1f.png' %(AA,PP)
            py.savefig(figname)

            # velocity plots
            py.clf()
            for pp in np.arange(nPlot):
                py.subplot(N,N,pp+1)
                #inRes = newfolder+'nonaligned_OSIRIScoords_fit_full_smooth_%.1f_%.1f_%.1f_%.1f.dat' % (pL[pp],pI[pp],AA,CPA)
                if inFolder:
                    inRes = inFolder + 'nonaligned_OSIRIScoords_fit_full_smooth_%.1f_%.1f_%.1f_%.1f_%.1f_residuals.dat' % (pL[pp],pI[pp],AA,CPA,PP)
                else:
                    inRes = newfolder+'nonaligned_OSIRIScoords_fit_full_smooth_%.1f_%.1f_%.1f_%.1f_%.1f_residuals.dat' % (pL[pp],pI[pp],AA,CPA,PP)
                res = ppxf_m31.modelFitResults(inRes)
                py.imshow(np.rot90(py.ma.masked_where(cubeimg==0.,res.velocity.T),3),vmin=-100.,vmax=200.,
                        extent=[xaxis[0], xaxis[-1], yaxis[0], yaxis[-1]])
                pltitle = '$\\theta_l$ = %.1f, $\\theta_i$ = %.1f' % (pL[pp],pI[pp])
                py.title(pltitle)
                py.xticks([])
                py.yticks([])
            cbar = py.colorbar(orientation='horizontal',ticks=[-100.,0.,200.])
            clab = 'Velocity residuals, $\\theta_a$ = %.1f, v$_{precess}$ = %.1f' % (AA,PP)
            cbar.set_label(clab)
            if inFolder:
                figname = inFolder + 'residuals_velocity_thetaA_%.1f_%.1f.png' %(AA,PP)
            else:
                figname=newfolder + 'residuals_velocity_thetaA_%.1f_%.1f.png' %(AA,PP)
            py.savefig(figname)

            # sigma plots
            py.clf()
            for pp in np.arange(nPlot):
                py.subplot(N,N,pp+1)
                #inRes = newfolder+'nonaligned_OSIRIScoords_fit_full_smooth_%.1f_%.1f_%.1f_%.1f.dat' % (pL[pp],pI[pp],AA,CPA)
                if inFolder:
                    inRes = inFolder + 'nonaligned_OSIRIScoords_fit_full_smooth_%.1f_%.1f_%.1f_%.1f_%.1f_residuals.dat' % (pL[pp],pI[pp],AA,CPA,PP)
                else:
                    inRes = newfolder+'nonaligned_OSIRIScoords_fit_full_smooth_%.1f_%.1f_%.1f_%.1f_%.1f_residuals.dat' % (pL[pp],pI[pp],AA,CPA,PP)
                res = ppxf_m31.modelFitResults(inRes)
                py.imshow(np.rot90(py.ma.masked_where(cubeimg==0.,res.sigma.T),3),vmin=-100.,vmax=100.,
                        extent=[xaxis[0], xaxis[-1], yaxis[0], yaxis[-1]])
                pltitle = '$\\theta_l$ = %.1f, $\\theta_i$ = %.1f' % (pL[pp],pI[pp])
                py.title(pltitle)
                py.xticks([])
                py.yticks([])
            cbar = py.colorbar(orientation='horizontal',ticks=[-100.,0.,100.])
            clab = 'Sigma residuals, $\\theta_a$ = %.1f, v$_{precess}$ = %.1f' % (AA,PP)
            cbar.set_label(clab)
            if inFolder:
                figname = inFolder + 'residuals_sigma_thetaA_%.1f_%.1f.png' %(AA,PP)
            else:
                figname=newfolder + 'residuals_sigma_thetaA_%.1f_%.1f.png' %(AA,PP)
            py.savefig(figname)

def modelBHFitGrid(inFolder=None,plotOnly=False,incubeimg=None):
    # routine to conduct a grid search over SMBH positions

    if incubeimg is None:
        cubeimg = pyfits.getdata(datadir + cuberoot + '_img.fits')
    else:
        cubeimg = pyfits.getdata(incubeimg)
    xaxis = np.arange(cubeimg.shape[0]) * 0.05
    yaxis = np.arange(cubeimg.shape[1]) * 0.05

    # original angles
    Lorg = -42.8
    Iorg = 54.1
    Aorg = -34.5
    CPA = -56.

    # BH step size, in pixels
    dPix = 1.
    # number of steps away from the original angle in each direction (-/+)
    # halfN = 1 gives 3 total steps, =2 gives 5 steps
    halfN = 2.
    N = (halfN*2)+1
    startpos = [0.,0.]

    posgridx = np.arange(startpos[0]-(halfN*dPix),startpos[0]+(halfN*dPix)+0.5,dPix)
    posgridy = np.arange(startpos[1]-(halfN*dPix),startpos[1]+(halfN*dPix)+0.5,dPix)
        
    if not plotOnly:
        # create the new folder
        FORMAT = '%Y%m%d_%H-%M-%S'
        now = datetime.datetime.now().strftime(FORMAT)
        newfolder = modeldir+'nonaligned_SMBH_grid_'+now+'/'
        os.mkdir(newfolder)
        #newfolder='/Users/kel/Documents/Projects/M31/models/Peiris/2003/nonaligned_grid_rotate_20160919_17-09-05/'

        for pX in posgridx:
            for pY in posgridy:
                testpos = [pX,pY]
                modelConvertBin(testSMBH=testpos,l98bin=False,verbose=False,tess=True,pathstem=newfolder)
                modelCompRes(inModel=newfolder+'nonaligned_OSIRIScoords_fit_full_smooth_SMBH_x%+.1f_y%+.1f.dat' % (testpos[0],testpos[1]),toPlot=False)

    # plotting
    py.close(2)
    py.figure(2, figsize=(17,12))
    py.subplots_adjust(left=0.05, right=0.94, top=0.95)
    nPlot = N**2
    # flux plots
    py.clf()
    plYY = np.fliplr([posgridy])[0]
    plXX = np.fliplr([posgridx])[0]
    pX,pY = np.meshgrid(plXX,plYY)
    pX = pX.flatten()
    pY = pY.flatten()
    for pp in np.arange(nPlot):
        py.subplot(N,N,pp+1)
        #inRes = newfolder+'nonaligned_OSIRIScoords_fit_full_smooth_%.1f_%.1f_%.1f_%.1f.dat' % (pL[pp],pI[pp],AA,CPA)
        if inFolder:
            inRes = inFolder+'nonaligned_OSIRIScoords_fit_full_smooth_SMBH_x%+.1f_y%+.1f_residuals.dat' % (pX[pp],pY[pp])
        else:
            inRes = newfolder+'nonaligned_OSIRIScoords_fit_full_smooth_SMBH_x%+.1f_y%+.1f_residuals.dat' % (pX[pp],pY[pp])
        res = ppxf_m31.modelFitResults(inRes)
        py.imshow(np.rot90(py.ma.masked_where(cubeimg==0.,res.nstar.T),3),vmin=-0.25,vmax=0.25,
                extent=[xaxis[0], xaxis[-1], yaxis[0], yaxis[-1]])
        # with the rotation+transposition of the residual image, the sign of dx is flipped (dy stays the same)
        pltitle = '%.1f, %.1f' % (-1.*pX[pp],pY[pp])
        py.title(pltitle)
        py.xticks([])
        py.yticks([])
    cbar = py.colorbar(orientation='horizontal',ticks=[-.25,0.,.25])
    clab = 'Flux res., $\\bullet_{model}$ at dx,dy'
    cbar.set_label(clab)
    if inFolder:
        figname = inFolder + 'residuals_flux_SMBH.png'
    else:
        figname=newfolder + 'residuals_flux_SMBH.png'
    py.savefig(figname)

    # velocity plots - can probably integrate with flux plotting
    py.clf()
    for pp in np.arange(nPlot):
        py.subplot(N,N,pp+1)
        #inRes = newfolder+'nonaligned_OSIRIScoords_fit_full_smooth_%.1f_%.1f_%.1f_%.1f.dat' % (pL[pp],pI[pp],AA,CPA)
        if inFolder:
            inRes = inFolder+'nonaligned_OSIRIScoords_fit_full_smooth_SMBH_x%+.1f_y%+.1f_residuals.dat' % (pX[pp],pY[pp])
        else:
            inRes = newfolder+'nonaligned_OSIRIScoords_fit_full_smooth_SMBH_x%+.1f_y%+.1f_residuals.dat' % (pX[pp],pY[pp])
        res = ppxf_m31.modelFitResults(inRes)
        py.imshow(np.rot90(py.ma.masked_where(cubeimg==0.,res.velocity.T),3),vmin=-100.,vmax=200.,
                extent=[xaxis[0], xaxis[-1], yaxis[0], yaxis[-1]])
        pltitle = '%.1f, %.1f' % (-1.*pX[pp],pY[pp])
        py.title(pltitle)
        py.xticks([])
        py.yticks([])
    cbar = py.colorbar(orientation='horizontal',ticks=[-100,0.,200])
    clab = 'Velocity res., $\\bullet_{model}$ at dx,dy'
    cbar.set_label(clab)
    if inFolder:
        figname = inFolder + 'residuals_velocity_SMBH.png'
    else:
        figname=newfolder + 'residuals_velocity_SMBH.png'
    py.savefig(figname)

    # sigma plots
    py.clf()
    for pp in np.arange(nPlot):
        py.subplot(N,N,pp+1)
        #inRes = newfolder+'nonaligned_OSIRIScoords_fit_full_smooth_%.1f_%.1f_%.1f_%.1f.dat' % (pL[pp],pI[pp],AA,CPA)
        if inFolder:
            inRes = inFolder+'nonaligned_OSIRIScoords_fit_full_smooth_SMBH_x%+.1f_y%+.1f_residuals.dat' % (pX[pp],pY[pp])
        else:
            inRes = newfolder+'nonaligned_OSIRIScoords_fit_full_smooth_SMBH_x%+.1f_y%+.1f_residuals.dat' % (pX[pp],pY[pp])
        res = ppxf_m31.modelFitResults(inRes)
        py.imshow(np.rot90(py.ma.masked_where(cubeimg==0.,res.sigma.T),3),vmin=-100.,vmax=100.,
                extent=[xaxis[0], xaxis[-1], yaxis[0], yaxis[-1]])
        pltitle = '%.1f, %.1f' % (-1.*pX[pp],pY[pp])
        py.title(pltitle)
        py.xticks([])
        py.yticks([])
    cbar = py.colorbar(orientation='horizontal',ticks=[-100,0.,100])
    clab = 'Sigma res., $\\bullet_{model}$ at dx,dy'
    cbar.set_label(clab)
    if inFolder:
        figname = inFolder + 'residuals_sigma_SMBH.png'
    else:
        figname=newfolder + 'residuals_sigma_SMBH.png'
    py.savefig(figname)

def calcModChiSq(inCubeImg=None,inData=None,inModelFolder=None,verbose=False,mask=False):
    # wrapper for modelChiSq - feed in the cube image and ppxf data fits to pass
    # through, and also the folder where the model .dat files are located, and
    # txt files with chi-squared values are saved to the same folder
    # verbose: sets the verbose keyword for modelChiSq and also gives you the model
    # file name

    files = glob.glob(inModelFolder + '*.0.dat')
    for ff in files:
        if verbose:
            print os.path.basename(ff)
        modelChiSq(inCubeImg=inCubeImg,inData=inData,inModel=ff,verbose=verbose,mask=mask)
        
def modelChiSq(inCubeImg=None,inData=None,inModel=None,verbose=False,mask=False):
    # input collapsed cube image and ppxf kinematic data, plus the tessellated model
    # outputs the chi-squared comparison for each moment (flux through h4)

    img = pyfits.getdata(inCubeImg)
    img = img.T
    if mask:
        # mask region outside of 1.3", per pg. 246 of P&T 2003
        xx,yy = np.meshgrid(np.arange(img.shape[1]),np.arange(img.shape[0]))
        dist = np.sqrt(((xx*0.05) - ppxf_m31.bhpos[0])**2 + ((yy*0.05) - ppxf_m31.bhpos[1])**2)
        idxmask = np.where(dist > 1.3)
        img[idxmask] = 0.

    #pdb.set_trace()
    data = ppxf_m31.PPXFresults(inputFile=inData)
    model = ppxf_m31.modelFitResults(inputFile=inModel)

    fluxres = (((img/img.sum()) - (model.nstar/model.nstar.sum()))**2 / (model.nstar/model.nstar.sum()))
    compvel = data.velocity + ppxf_m31.vsys
    velres = ((np.ma.masked_where(img==0,compvel) - np.ma.masked_where(img==0,model.velocity))**2 / np.ma.masked_where(img==0,model.velocity))
    sigres = ((np.ma.masked_where(img==0,data.sigma) - np.ma.masked_where(img==0,model.sigma))**2 / np.ma.masked_where(img==0,model.sigma))
    h3res = ((np.ma.masked_where(img==0,data.h3) - np.ma.masked_where(img==0,model.h3))**2 / np.ma.masked_where(img==0,model.h3))
    h4res = ((np.ma.masked_where(img==0,data.h4) - np.ma.masked_where(img==0,model.h4))**2 / np.ma.masked_where(img==0,model.h4))

    fluxCS = np.ma.masked_invalid(fluxres).sum()
    velCS = np.ma.masked_invalid(velres).sum()
    sigCS = np.ma.masked_invalid(sigres).sum()
    h3CS = np.ma.masked_invalid(h3res).sum()
    h4CS = np.ma.masked_invalid(h4res).sum()

    if verbose:
        print 'Chi-squared: flux: %.5f, velocity: %.5f, sigma: %.5f, h3: %.5f, h4: %.5f' % (fluxCS,velCS,sigCS,h3CS,h4CS)
    
    outfile = inModel.replace('.dat','_chisq.txt')
    _out = open(outfile,'w')
    _out.write('flux: %.5f, velocity: %.5f, sigma: %.5f, h3: %.5f, h4: %.5f' % (fluxCS,velCS,sigCS,h3CS,h4CS))
    _out.close
