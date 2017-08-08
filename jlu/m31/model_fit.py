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

        xlen = nxbin#41
        ylen = nybin#82
        goodTrim = np.where((X/binpc >= newnegxbin) & (X/binpc <= (newnegxbin + xlen)) & (Y/binpc >= newnegybin) & (Y/binpc <= (newnegybin + ylen)))

        xClip = X + negxbin*binpc#[goodTrim[0]] - newnegxbin*binpc
        yClip = Y + negybin*binpc#[goodTrim[0]] - newnegybin*binpc
        zClip = Z#[goodTrim[0]]
        vxClip = V_X#[goodTrim[0]]
        vyClip = V_Y#[goodTrim[0]]
        losVClip = losV#[goodTrim[0]]

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

        pdb.set_trace()
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
    
        
def modelMorphFitGrid(incube=None,inVorTxt=None,inDataFile=None,inErr=None,incubeimg=None,norm=False,inFolder=None,plotOnly=False):
    # routine to conduct a grid search over the coordinate transformation angles
    # (angles in Peiris & Tremaine 2003 eq 4)

    # inFolder: input folder if models have already been run (e.g. for plotOnly)
    # plotOnly: doesn't rerun models, just does plots
    # incube: full data cube, generally tessellated, with flux errors
    # inVorTxt: tessellates models using the same scheme as the data
    # inDataFile: ppxf fits that go with incube
    # inErr: kinematic MC errors that go with inDataFile
    # incubeimg: collapsed data cube from incube, for calculating chi2 for flux map
    # norm: set to return reduced chi2 instead of standard

    # to do a fresh run, modelMorphFitGrid(incube=,inVorTxt=,inDataFile=,inErr=,incubeimg=,norm=)

    if incube is None:
        cubeimg = pyfits.getdata(datadir + cuberoot + '_img.fits')
    else:
        cube = pyfits.getdata(incube)
        cubeimg = cube.mean(axis=2)
    xaxis = np.arange(cubeimg.shape[0]) * 0.05
    yaxis = np.arange(cubeimg.shape[1]) * 0.05

    # original angles
    Lorg = -42.8
    Iorg = 54.1
    Aorg = -34.5
    # not changing CPA
    CPA = -56.

    # angle step size, in degrees
    #dAng = 15.
    dAng = 5.
    #dAng=1.
    # number of steps away from the original angle in each direction (-/+)
    # halfN = 1 gives 3 total steps, =2 gives 5 steps
    halfN = 3.
    #halfN = 0
    N = (halfN*2)+1

    #Lgrid = np.arange(Lorg-(halfN*dAng),Lorg+(halfN*dAng)+1,dAng)
    #Lgrid = np.array([-22.8,-17.8])
    Lgrid = np.array([-17.8,-22.8,-27.8,-32.8,-37.8,-42.8,-47.8,-52.8,-57.8])
    #Igrid = np.arange(Iorg-(halfN*dAng),Iorg+(halfN*dAng)+1,dAng)
    #Igrid = np.array([34.1,29.1])
    Igrid = np.array([29.1,34.1,39.1,44.1,49.1,54.1,59.1,64.1,69.1])
    #Agrid = np.arange(Aorg-(halfN*dAng),Aorg+(halfN*dAng)+1,dAng)
    Agrid = np.array([-9.5])
    #Agrid = np.array([-14.5,-19.5,-24.5,-29.5,-34.5,-39.5,-44.5,-49.5])
    #Lgrid = np.array([-32.8])
    #Igrid = np.array([44.1])
    #Agrid = np.array([-14.5])

    # precession grid
    #Pgrid = np.array([-100.,-30.,0.,30.,100.])
    #Pgrid = np.array([-30.,0.,30.])
    #Pgrid = np.array([-20.,-10.,10.,20.])
    #Pgrid = np.array([-5.,5.])
    #Pgrid = np.array([0.])
    Pgrid = np.array([30.,10.,5.,0.,-5.,-10.,-15.,-20.,-30.])
    #Pgrid = np.array([30.,10.])
    #Pgrid = np.array([-5.,-10.,-20.,-30.])
    #Pgrid = np.array([-15.])
    #Pgrid = np.array([5.])
    #Pgrid = np.array([62.2])
        
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
                        modelChiSq(inCube=incube,inData=inDataFile,inErr=inErr,inModel=newfolder+'nonaligned_OSIRIScoords_fit_full_smooth_%.1f_%.1f_%.1f_%.1f_%.1f.dat' % (testAng[0],testAng[1],testAng[2],testAng[3],PP),mask=True,norm=norm)

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

def gridChiSqMap(inFolder=None):
    # reads in chi sq txts from inFolder, assembles data into a map,
    # saves the output to a fits file (one per moment)

    # read in all chi sq txts from a given folder
    filelist = glob.glob(inFolder + '*_chisq.txt')

    # extract chi sq values into an array
    nfiles = len(filelist)
    chisq = np.zeros((nfiles,5))
    # also extract angles and precession values
    # angle order is theta_L, theta_I, theta_A
    theta = np.zeros((nfiles,3))
    precessall = np.zeros((nfiles))
    for i in np.arange(nfiles):
        tmp = pandas.read_csv(filelist[i],header=None,names=['f','v','s','h3','h4'])
        chisq[i,0] = float(str.split(tmp['f'][0])[1])
        chisq[i,1] = float(str.split(tmp['v'][0])[1])
        chisq[i,2] = float(str.split(tmp['s'][0])[1])
        chisq[i,3] = float(str.split(tmp['h3'][0])[1])
        chisq[i,4] = float(str.split(tmp['h4'][0])[1])

        # read the angles and precession from the file name
        tmpf = filelist[i].split('_')
        theta[i,0] = float(tmpf[-6])
        theta[i,1] = float(tmpf[-5])
        theta[i,2] = float(tmpf[-4])

        precessall[i] = float(tmpf[-2])

    # get the range of angles and precession
    thetaL = np.unique(theta[:,0])
    thetaI = np.unique(theta[:,1])
    thetaA = np.unique(theta[:,2])
    precess = np.unique(precessall)

    nL = len(thetaL)
    nI = len(thetaI)
    nA = len(thetaA)
    nP = len(precess)

    # start filling in a big array
    # small box: nI x nL
    # big box: nA x nP
    grid = np.zeros((nI*nA,nL*nP,5))
    anglemap = np.zeros((nI*nA,nL*nP,4))
    for i in np.arange(nfiles):
        L = np.where(theta[i,0] == thetaL)
        I = np.where(theta[i,1] == thetaI)
        A = np.where(theta[i,2] == thetaA)
        P = np.where(precessall[i] == precess)

        coord1 = (nI * A[0]) + I[0]
        coord2 = (nL * P[0]) + L[0]
        grid[coord1,coord2,0] = chisq[i,0]
        grid[coord1,coord2,1] = chisq[i,1]
        grid[coord1,coord2,2] = chisq[i,2]
        grid[coord1,coord2,3] = chisq[i,3]
        grid[coord1,coord2,4] = chisq[i,4]

        anglemap[coord1,coord2,:] = [theta[i,0],theta[i,1],theta[i,2],precessall[i]]
    
    pdb.set_trace()
    outFile = inFolder+'chisq_map.fits'
    pyfits.writeto(outFile,grid)
    pyfits.append(outFile,anglemap)

def plotChiSqMap(inMap=None,sum=False,save=False):
    # reads in output from gridChiSqMap() or weightChi2(), plots it
    # for output from weightChi2(), set sum=True
    
    chiSq = pyfits.getdata(inMap)
    angleMap = pyfits.getdata(inMap,1)

    # check the possible values of angles/precession
    thetaL = np.unique(angleMap[:,:,0])
    thetaI = np.unique(angleMap[:,:,1])
    thetaA = np.unique(angleMap[:,:,2])
    precess = np.unique(angleMap[:,:,3])

    nL = len(thetaL)
    nI = len(thetaI)
    nA = len(thetaA)
    nP = len(precess)
    
    # break the main map up into subplots, for plotting
    py.close(2)
    #py.figure(2,figsize=(11,7))
    #py.figure(2,figsize=(3,9))
    py.figure(2,figsize=(13,9))
    py.subplots_adjust(left=0.1, right=0.85, top=0.95)
    if sum:
        nM = 1
    else:
        nM = 5
    for m in np.arange(nM):
        for i in np.arange(nA):
            for j in np.arange(nP):
                nPlot = ((nA-(i+1))*nP) + (j+1)
                ax = py.subplot(nA,nP,nPlot)
                #if (m == 1) or (m == 3) or (m == 4):
                #    lo = np.percentile(-1.*chiSq[:,:,m],90)
                #    lo *= -1.
                #    hi = np.percentile(chiSq[:,:,m],90)
                #else:
                if sum:
                    lo = np.min(chiSq[:,:])
                    hitmp = np.max(chiSq[:,:])
                    hi = (hitmp-lo)/4 + lo
                    #hi = np.max(chiSq[:,:])
                    py.imshow(chiSq[(nI*i):(nI*(i+1)),(nL*j):(nL*(j+1))],vmin=lo,vmax=hi)
                else:
                    lo = np.min(chiSq[:,:,m])
                    hitmp = np.max(chiSq[:,:,m])
                    hi = (hitmp-lo)/4 + lo
                    #hi = np.max(chiSq[:,:,m])
                    py.imshow(chiSq[(nI*i):(nI*(i+1)),(nL*j):(nL*(j+1)),m],vmin=lo,vmax=hi)
                py.xticks([])
                py.yticks([])
                # label the leftmost subplots with the correct thetaA values and the yticks
                if (nPlot-1)%nP == 0:
                    if nPlot  > (nP*(nA-1)):
                        tmplab = '$\Theta_A$=%.1f' % thetaA[i]
                    else:
                        tmplab = '%.1f' % thetaA[i]
                    py.ylabel(tmplab)
                    py.yticks([0,nI/2,nI-1],[thetaI[0],thetaI[nI/2],thetaI[nI-1]])
                # label the topmost subplots with the correct precession values
                if nPlot <= nP:
                    if nPlot == 1:
                        tmplab = '$\Omega_P$=%.1f' % precess[j]
                    else:
                        tmplab = '%.1f' % precess[j]
                    py.title(tmplab)
                # label the bottommost subplots with the xticks
                if nPlot > (nP*(nA-1)):
                    py.xticks([0,nL/2,nL-1],[thetaL[0],thetaL[nL/2],thetaL[nL-1]],rotation='vertical')
                    py.xlabel('$\Theta_L$')
                # add the colorbar to the bottom right subplot
                if nPlot == nA*nP:
                    box = ax.get_position()
                    axColor = py.axes([box.x0 + box.width * 1.05, box.y0, 0.01, box.height])
                    #axColor = py.axes([box.x0 + box.width * .7, box.y0, 0.01, box.height])
                    cbar=py.colorbar(cax = axColor,orientation='vertical',ticks=[lo,((hi-lo)/2.)+lo,hi])
                    cbar.set_label('$\chi^2$')
        # print the angles of the chi-squared minimum for each moment
        if sum:
            tmpminidx = np.argmin(chiSq[:,:])
            tmpminarr = np.unravel_index(tmpminidx,chiSq[:,:].shape)
            tmpang = angleMap[tmpminarr[0],tmpminarr[1],:]
            print 'Chi-sq min for the weighted sum at theta_l = %.1f, theta_i = %.1f, theta_a = %.1f, precess = %.1f' % (tmpang[0],tmpang[1],tmpang[2],tmpang[3])
            if save:
                outfile = inMap.replace('.fits','_angles.txt')
                _out = open(outfile,'w')
                _out.write('Chi-sq min for the weighted sum at theta_l = %.1f, theta_i = %.1f, theta_a = %.1f, precess = %.1f' % (tmpang[0],tmpang[1],tmpang[2],tmpang[3]))
                _out.close
        else:
            tmpminidx = np.argmin(chiSq[:,:,m])
            tmpminarr = np.unravel_index(tmpminidx,chiSq[:,:,m].shape)
            tmpang = angleMap[tmpminarr[0],tmpminarr[1],:]
            print 'Chi-sq min for moment %.0f at theta_l = %.1f, theta_i = %.1f, theta_a = %.1f, precess = %.1f' % (m,tmpang[0],tmpang[1],tmpang[2],tmpang[3])
            if save:
                outfile = inMap.replace('.fits','_mom%d_angles.txt' % (m))
                _out = open(outfile,'w')
                _out.write('Chi-sq min for moment %.0f at theta_l = %.1f, theta_i = %.1f, theta_a = %.1f, precess = %.1f' % (m,tmpang[0],tmpang[1],tmpang[2],tmpang[3]))
                _out.close

        if save:
            if sum:
                outfile = inMap.replace('.fits','.png')
            else:
                outfile = inMap.replace('.fits','_mom%d.png' % (m))
            py.savefig(outfile)
        else:
            pdb.set_trace()

        py.clf()   

def weightChi2(inMap=None):
    # reads in a chi^2 map, weights the flux, velocity, sigma chi^2 by
    # dividing them by the minimum chi^2 values for each moment

    # outputs the results to a fits file
    # use plotChiSqMap() to plot

    # reads in output from gridChiSqMap()
    chiSq = pyfits.getdata(inMap)
    angleMap = pyfits.getdata(inMap,1)

    # grab the index of the nonaligned angles + 0 precession
    #orgidx = np.where((angleMap[:,:,0] == -42.8) & (angleMap[:,:,1] == 54.1) & (angleMap[:,:,2] == -34.5) & (angleMap[:,:,3] == 0.))

    # get the chi^2 values at this index
    #orgchi2 = chiSq[orgidx[0][0],orgidx[1][0],:]

    # weight by the minimum values instead

    # weight all chi^2 values by this chi^2
    weightChi2 = chiSq
    weightChi2[:,:,0] /= np.min(weightChi2[:,:,0])
    weightChi2[:,:,1] /= np.min(weightChi2[:,:,1])
    weightChi2[:,:,2] /= np.min(weightChi2[:,:,2])
    weightChi2[:,:,3] /= np.min(weightChi2[:,:,3])
    weightChi2[:,:,4] /= np.min(weightChi2[:,:,4])

    # sum the weighted chi^2 values for only flux, velocity, sigma
    totChi2 = weightChi2[:,:,0:3].sum(axis=2)

    outFile = inMap.replace('.fits','_weightsum.fits')
    pyfits.writeto(outFile,totChi2)
    pyfits.append(outFile,angleMap)

def plot1Dchi2(inMap=None):
    # plots chi2 vs. [parameter] for all angles and precessions
    # (all but the plotted one are held at their best fit values)

    # inMap: should be chisq_map_weightsum.fits 
    chiSq = pyfits.getdata(inMap)
    angleMap = pyfits.getdata(inMap,1)

    # check the possible values of angles/precession
    thetaL = np.unique(angleMap[:,:,0])
    thetaI = np.unique(angleMap[:,:,1])
    thetaA = np.unique(angleMap[:,:,2])
    precess = np.unique(angleMap[:,:,3])

    nL = len(thetaL)
    nI = len(thetaI)
    nA = len(thetaA)
    nP = len(precess)

    # find the best-fit angles/precession
    minidx = np.argmin(chiSq[:,:])
    minarr = np.unravel_index(minidx,chiSq[:,:].shape)
    minang = angleMap[minarr[0],minarr[1],:]

    # get the indices where everything is held constant at best-fit except for one parameter
    idxL = np.where((angleMap[:,:,1] == minang[1]) & (angleMap[:,:,2] == minang[2]) & (angleMap[:,:,3] == minang[3]))
    idxI = np.where((angleMap[:,:,0] == minang[0]) & (angleMap[:,:,2] == minang[2]) & (angleMap[:,:,3] == minang[3]))
    idxA = np.where((angleMap[:,:,0] == minang[0]) & (angleMap[:,:,1] == minang[1]) & (angleMap[:,:,3] == minang[3]))
    idxP = np.where((angleMap[:,:,0] == minang[0]) & (angleMap[:,:,1] == minang[1]) & (angleMap[:,:,2] == minang[2]))

    # plot
    # st. dev on weighted chi2 is 0.01, from model_fit.chi2MCerr (7x7x7 grid)
    ytickLoc = py.MultipleLocator(1.0)
    err = 0.01
    err3sig = 3.*err
    py.close(2)
    py.figure(2,figsize=(6,14))
    py.subplots_adjust(left=0.18, right=0.94, top=0.95,bottom=0.1,hspace=.45)

    py.subplot(411)
    py.plot(angleMap[idxL[0],idxL[1],:][:,0],chiSq[idxL[0],idxL[1]],'ko')
    py.axhline(np.min(chiSq[idxL[0],idxL[1]]) + err,color='green',linestyle='--',label='1$\sigma$ error')
    py.axhline(np.min(chiSq[idxL[0],idxL[1]]) + err3sig,color='blue',linestyle=':',label='3$\sigma$ error')
    py.xlabel('$\\theta_l$ ($^{\circ}$)')
    py.ylabel('$\~{\chi}^2$')
    py.yticks([3,4,5,6])
    py.legend(loc=0)
    #pdb.set_trace()

    #py.clf()
    py.subplot(412)
    py.plot(angleMap[idxI[0],idxI[1],:][:,1],chiSq[idxI[0],idxI[1]],'ko')
    py.axhline(np.min(chiSq[idxI[0],idxI[1]]) + err,color='green',linestyle='--',label='1$\sigma$ error')
    py.axhline(np.min(chiSq[idxI[0],idxI[1]]) + err3sig,color='blue',linestyle=':',label='3$\sigma$ error')
    py.xlabel('$\\theta_i$ ($^{\circ}$)')
    py.ylabel('$\~{\chi}^2$')
    py.yticks([3,4,5,6])
    #pdb.set_trace()

    #py.clf()
    py.subplot(413)
    py.plot(angleMap[idxA[0],idxA[1],:][:,2],chiSq[idxA[0],idxA[1]],'ko')
    py.axhline(np.min(chiSq[idxA[0],idxA[1]]) + err,color='green',linestyle='--',label='1$\sigma$ error')
    py.axhline(np.min(chiSq[idxA[0],idxA[1]]) + err3sig,color='blue',linestyle=':',label='3$\sigma$ error')
    py.xlabel('$\\theta_a$ ($^{\circ}$)')
    py.ylabel('$\~{\chi}^2$')
    py.yticks([3.2,3.4,3.6,3.8])
    #pdb.set_trace()

    #py.clf()
    py.subplot(414)
    py.plot(angleMap[idxP[0],idxP[1],:][:,3],chiSq[idxP[0],idxP[1]],'ko')
    py.axhline(np.min(chiSq[idxP[0],idxP[1]]) + err,color='green',linestyle='--',label='1$\sigma$ error')
    py.axhline(np.min(chiSq[idxP[0],idxP[1]]) + err3sig,color='blue',linestyle=':',label='3$\sigma$ error')
    py.xlabel('$\Omega_P$ (km s$^{-1}$ pc$^{-1}$)')
    py.ylabel('$\~{\chi}^2$')
    py.yticks([3,4,5,6,7])
    py.xlim(-35,35)
    #pdb.set_trace()
            
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

def calcModChiSq(inCube=None,inData=None,inErr=None,inModelFolder=None,verbose=False,mask=False,norm=False,scaled=False):
    # wrapper for modelChiSq - feed in the cube image and ppxf data fits to pass
    # through, and also the folder where the model .dat files are located, and
    # txt files with chi-squared values are saved to the same folder
    # verbose: sets the verbose keyword for modelChiSq and also gives you the model
    # file name

    files = glob.glob(inModelFolder + '*.0.dat')
    for ff in files:
        if verbose:
            print os.path.basename(ff)
        modelChiSq(inCube=inCube,inData=inData,inModel=ff,inErr=inErr,verbose=verbose,mask=mask,norm=norm,scaled=scaled)
        
def modelChiSq(inCube=None,inData=None,inModel=None,inErr=None,verbose=False,mask=False,norm=False,scaled=False,outFolder=None):
    # input cube (w/ error array), MC errors, and ppxf kinematic data, plus the tessellated model
    # outputs the chi-squared comparison for each moment (flux through h4)

    # set norm to return the reduced chi-sq

    cubefits = pyfits.open(inCube)
    img = np.median(cubefits[0].data,axis=2)
    imgscl = img/img.sum()
    #img = img.T
    ferr = np.median(cubefits[1].data,axis=2)
    #ferr = ferr.T
    #snr = img / ferr
    #ferr_scl = imgscl / snr
    ferr_scl = ferr/img.sum()

    if mask:
        # mask region outside of 1.3", per pg. 246 of P&T 2003
        xx,yy = np.meshgrid(np.arange(img.shape[1]),np.arange(img.shape[0]))
        dist = np.sqrt(((xx*0.05) - ppxf_m31.bhpos[0])**2 + ((yy*0.05) - ppxf_m31.bhpos[1])**2)
        idxmask = np.where(dist > 1.3)
        img[idxmask] = 0.
        imgscl[idxmask] = 0.
        ferr_scl[idxmask] = 0.

        goodidx = np.where((dist <= 1.3) & (img != 0.))
        numgood = len(goodidx[0])

    #pdb.set_trace()
    data = ppxf_m31.PPXFresults(inputFile=inData)
    err = ppxf_m31.PPXFresults(inputFile=inErr)
    model = ppxf_m31.modelFitResults(inputFile=inModel)

    if scaled:
        # hard coded scale factors, based on the minimum reduced chi-sq for each moment
        scl_f = 98.1
        scl_v = 12.1
        scl_s = 3.1
        scl_h3 = 8.2
        scl_h4 = 6.5
    else:
        # set the scale factors to 1 if not using
        scl_f = 1
        scl_v = 1
        scl_s = 1
        scl_h3 = 1
        scl_h4 = 1


    compvel = data.velocity + ppxf_m31.vsys
    # bad chi-sq
    #fluxres = (((img/img.sum()) - (model.nstar/model.nstar.sum()))**2 / (model.nstar/model.nstar.sum()))
    #velres = ((np.ma.masked_where(img==0,compvel) - np.ma.masked_where(img==0,model.velocity))**2 / np.ma.masked_where(img==0,model.velocity))
    #sigres = ((np.ma.masked_where(img==0,data.sigma) - np.ma.masked_where(img==0,model.sigma))**2 / np.ma.masked_where(img==0,model.sigma))
    #h3res = ((np.ma.masked_where(img==0,data.h3) - np.ma.masked_where(img==0,model.h3))**2 / np.ma.masked_where(img==0,model.h3))
    #h4res = ((np.ma.masked_where(img==0,data.h4) - np.ma.masked_where(img==0,model.h4))**2 / np.ma.masked_where(img==0,model.h4))
    # no scaling
    #fluxres = ( imgscl - (model.nstar/model.nstar.sum()) )**2 / (ferr_scl**2)
    #velres = (np.ma.masked_where(img==0,compvel) - np.ma.masked_where(img==0,model.velocity))**2 / np.ma.masked_where(img==0,err.velocity)**2
    #sigres = (np.ma.masked_where(img==0,data.sigma) - np.ma.masked_where(img==0,model.sigma))**2 / np.ma.masked_where(img==0,err.sigma)**2
    #h3res = (np.ma.masked_where(img==0,data.h3) - np.ma.masked_where(img==0,model.h3))**2 / np.ma.masked_where(img==0,err.h3)**2
    #h4res = (np.ma.masked_where(img==0,data.h4) - np.ma.masked_where(img==0,model.h4))**2 / np.ma.masked_where(img==0,err.h4)**2
    # scaling by scaling the errors
    fluxres = ( imgscl - (model.nstar/model.nstar.sum()) )**2 / ( (ferr_scl**2) * scl_f)
    velres = (np.ma.masked_where(img==0,compvel) - np.ma.masked_where(img==0,model.velocity))**2 / ( np.ma.masked_where(img==0,err.velocity)**2 * scl_v)
    sigres = (np.ma.masked_where(img==0,data.sigma) - np.ma.masked_where(img==0,model.sigma))**2 / ( np.ma.masked_where(img==0,err.sigma)**2 * scl_s)
    h3res = (np.ma.masked_where(img==0,data.h3) - np.ma.masked_where(img==0,model.h3))**2 / ( np.ma.masked_where(img==0,err.h3)**2 * scl_h3)
    h4res = (np.ma.masked_where(img==0,data.h4) - np.ma.masked_where(img==0,model.h4))**2 / ( np.ma.masked_where(img==0,err.h4)**2 * scl_h4)
    # scaling by adding a scale factor in quadrature
    #fluxres = ( imgscl - (model.nstar/model.nstar.sum()) )**2 / np.sqrt( (ferr_scl**2) + ( (scl_f*(ferr_scl**2))**2 - (ferr_scl**2) ) )
    #velres = (np.ma.masked_where(img==0,compvel) - np.ma.masked_where(img==0,model.velocity))**2 / np.sqrt( np.ma.masked_where(img==0,err.velocity)**2 + scl_v)
    #sigres = (np.ma.masked_where(img==0,data.sigma) - np.ma.masked_where(img==0,model.sigma))**2 / ( np.ma.masked_where(img==0,err.sigma)**2 + scl_s)
    #h3res = (np.ma.masked_where(img==0,data.h3) - np.ma.masked_where(img==0,model.h3))**2 / ( np.ma.masked_where(img==0,err.h3)**2 + scl_h3)
    #h4res = (np.ma.masked_where(img==0,data.h4) - np.ma.masked_where(img==0,model.h4))**2 / ( np.ma.masked_where(img==0,err.h4)**2 + scl_h4)

    fluxCS = np.ma.masked_invalid(fluxres).sum()
    velCS = np.ma.masked_invalid(velres).sum()
    sigCS = np.ma.masked_invalid(sigres).sum()
    h3CS = np.ma.masked_invalid(h3res).sum()
    h4CS = np.ma.masked_invalid(h4res).sum()

    if norm:
        # divide by the number of spaxels minus the degrees of freedom
        degfree = 5
        fluxCS /= (numgood - degfree)
        velCS /= (numgood - degfree)
        sigCS /= (numgood - degfree)
        h3CS /= (numgood - degfree)
        h4CS /= (numgood - degfree)

    if verbose:
        print 'Chi-squared: flux: %.5f, velocity: %.5f, sigma: %.5f, h3: %.5f, h4: %.5f' % (fluxCS,velCS,sigCS,h3CS,h4CS)

    #pdb.set_trace()
    if outFolder:
        base = os.path.basename(inModel)
        outfile = os.path.dirname(inModel) + '/' + outFolder + '/'+ base.replace('.dat','.chisq.txt')
    else:
        outfile = inModel.replace('.dat','_chisq.txt')
    _out = open(outfile,'w')
    _out.write('flux: %.5f, velocity: %.5f, sigma: %.5f, h3: %.5f, h4: %.5f' % (fluxCS,velCS,sigCS,h3CS,h4CS))
    _out.close

def chi2Errors(inCube=None,inData=None,inErr=None,inModelFolder=None,mask=True,norm=True,rerun=True):
    # wrapper to calculate chi2 errors
    # reads in a data cube (contains flux errors), the tessellation file, the ppxf data file, and the MC errors on the ppxf fit
    # also reads in the model folder
    # code tweaks the value in each bin in each moment map (flux--h4) within the errors and calculates the reduced chi2
    #    w.r.t. each model in the given folder
    # chi^2 files are output to separate folders
    # iterates 100 times

    cubefits = pyfits.open(inCube)
    data = ppxf_m31.PPXFresults(inputFile=inData)
    err = ppxf_m31.PPXFresults(inputFile=inErr)
      
    img = np.median(cubefits[0].data,axis=2)
    imgscl = img/img.sum()
    ferr = np.median(cubefits[1].data,axis=2)
    ferr_scl = ferr/img.sum()

    if mask:
        # mask region outside of 1.3", per pg. 246 of P&T 2003
        xx,yy = np.meshgrid(np.arange(img.shape[1]),np.arange(img.shape[0]))
        dist = np.sqrt(((xx*0.05) - ppxf_m31.bhpos[0])**2 + ((yy*0.05) - ppxf_m31.bhpos[1])**2)
        idxmask = np.where(dist > 1.3)
        img[idxmask] = 0.
        imgscl[idxmask] = 0.
        ferr_scl[idxmask] = 0.

        goodidx = np.where((dist <= 1.3) & (img != 0.))
        numgood = len(goodidx[0])

    numiter = np.arange(100)
    #pdb.set_trace()

    t1 = time.time()
    #job_server = pp.Server()
    #print "Starting pp with", job_server.get_ncpus(), "workers"
    #jobs = [(i,job_server.submit(run_once_chi_mc, (imgscl,data,ferr_scl,err,numgood,inModelFolder,i), (), ('numpy as np','ppxf_m31','os','glob'))) for i in numiter]
    #job_server.wait()

    #run_once_chi_mc(imgscl,data,ferr_scl,err,numgood,inModelFolder,2)
    if rerun:
        for i in numiter:
            run_once_chi_mc(imgscl,data,ferr_scl,err,numgood,inModelFolder,i)

        for i in numiter:
            gridChiSqMap(inFolder=inModelFolder + '/chi2_MCerr/iter_' + str(i) + '/')
            weightChi2(inMap=inModelFolder + '/chi2_MCerr/iter_' + str(i) +'/chisq_map.fits')

    for i in numiter:
        plotChiSqMap(inMap=inModelFolder + '/chi2_MCerr/iter_' + str(i) +'/chisq_map_weightsum.fits',sum=True,save=True)
        
    print "Time elapsed: ", time.time() - t1, "s"
    
def run_once_chi_mc(img,data,ferr,err,numgood,inModelFolder,numiter):

    os.mkdir(inModelFolder + '/chi2_MCerr/iter_' + str(numiter))
    
    # grab the moments
    vel = data.velocity + ppxf_m31.vsys
    sig = data.sigma
    h3 = data.h3
    h4 = data.h4

    # tweak input moment maps
    idx = np.where(img == 0)
    img = img + (np.random.randn(img.shape[0],img.shape[1]) * ferr)
    img[idx] = 0.
    vel = vel + (np.random.randn(vel.shape[0],vel.shape[1]) * err.velocity)
    sig = sig + (np.random.randn(sig.shape[0],sig.shape[1]) * err.sigma)
    h3 = h3 + (np.random.randn(h3.shape[0],h3.shape[1]) * err.h3)
    h4 = h4 + (np.random.randn(h4.shape[0],h4.shape[1]) * err.h4)
    
    # compare the tweaked moment maps to the models
    files = glob.glob(inModelFolder + '/*.0.dat')
    for ff in files:
        model = ppxf_m31.modelFitResults(inputFile=ff)
        # no scaling
        fluxres = ( img - (model.nstar/model.nstar.sum()) )**2 / (ferr**2)
        velres = (np.ma.masked_where(img==0,vel) - np.ma.masked_where(img==0,model.velocity))**2 / np.ma.masked_where(img==0,err.velocity)**2
        sigres = (np.ma.masked_where(img==0,sig) - np.ma.masked_where(img==0,model.sigma))**2 / np.ma.masked_where(img==0,err.sigma)**2
        h3res = (np.ma.masked_where(img==0,h3) - np.ma.masked_where(img==0,model.h3))**2 / np.ma.masked_where(img==0,err.h3)**2
        h4res = (np.ma.masked_where(img==0,h4) - np.ma.masked_where(img==0,model.h4))**2 / np.ma.masked_where(img==0,err.h4)**2

        fluxCS = np.ma.masked_invalid(fluxres).sum()
        velCS = np.ma.masked_invalid(velres).sum()
        sigCS = np.ma.masked_invalid(sigres).sum()
        h3CS = np.ma.masked_invalid(h3res).sum()
        h4CS = np.ma.masked_invalid(h4res).sum()

        #if norm:
        # divide by the number of spaxels minus the degrees of freedom
        degfree = 5
        fluxCS /= (numgood - degfree)
        velCS /= (numgood - degfree)
        sigCS /= (numgood - degfree)
        h3CS /= (numgood - degfree)
        h4CS /= (numgood - degfree)

        #print 'Chi-squared: flux: %.5f, velocity: %.5f, sigma: %.5f, h3: %.5f, h4: %.5f' % (fluxCS,velCS,sigCS,h3CS,h4CS)
        # write out the chi2 files for each iteration to a separate subfolder
        base = os.path.basename(ff)
        outfile = inModelFolder + '/chi2_MCerr/iter_' + str(numiter) + '/'+ base.replace('.dat','_chisq.txt')
        _out = open(outfile,'w')
        _out.write('flux: %.5f, velocity: %.5f, sigma: %.5f, h3: %.5f, h4: %.5f' % (fluxCS,velCS,sigCS,h3CS,h4CS))
        _out.close

def chi2MCerr(inFolder=None):
    # reads in the chi2maps from the previous run of MC errors
    # finds the minimum chi2 for each moment from each run and takes
    # the standard deviation. Prints this, as an error on the minimum chi2

    # hard coded, same as number of MC run iterations
    niter = 100
    minchi2 = np.zeros((niter,5))
    wminchi2 = np.zeros((niter))
    for i in np.arange(niter):
        infile = pyfits.getdata(inFolder + '/iter_' + str(i) + '/' + 'chisq_map.fits')
        infilew = pyfits.getdata(inFolder + '/iter_' + str(i) + '/' + 'chisq_map_weightsum.fits')
        for j in np.arange(5):
            minchi2[i,j] = np.min(infile[:,:,j])
        wminchi2[i] = np.min(infilew)

    chi2err = np.std(minchi2,axis=0)
    wchi2err = np.std(wminchi2)

    print 'St. dev. of min. chi2 for each moment is: flux: %.2f, velocity: %.2f, dispersion: %.2f, h3: %.2f, h4: %.2f' %(chi2err[0],chi2err[1],chi2err[2],chi2err[3],chi2err[4])
    print 'St. dev. of weighted min. chi2 is: %.2f' % (wchi2err)

    pdb.set_trace()

def plotTWcal():
    # plotting the TW precession vs. input precession for a bunch of models, plus predicting the
    # precession for the data from the TW precession

    xs = np.array([-30.,-20.,-15.,-10.,-5.,0.,5.,10.,30.])
    # model TW precessions for best-fit orientation
    #ys = np.array([-13.8,-12.,-10.5,-9.8,-8.9,-7.8,-6.4,-5.8,-1.3])
    # model TW precessions for best-fit orientation, code as of 8/7/2017
    # (matches HP's test for B01 data now)
    #ys = np.array([47.18,41.27,35.89,34.13,30.94,27.23,22.06,20.06,4.71])
    # new best-fit orientation, as of 8/7/2017pm
    ys = np.array([49.2,41.48,37.58,37.13,30.94,27.50,24.74,20.31,4.71])
    # model TW precessions for nonaligned orientation
    #ysn = np.array([-16.2,-14.6,-13.8,-11.4,-10.9,-12.4,-8.9,-9.3,-5.9])
    # model TW precessions for nonaligned orientation, code as of 8/7/2017
    # (matches HP's test for B01 data now)
    ysn = np.array([77.43,70.15,66.08,54.52,52.48,59.98,42.78,44.79,28.46])

    # sin i for best-fit theta_i
    sini = np.sin(np.radians(44.1))

    ysi = ys/sini
    ysni = ysn/sini

    # fit a line to the inputs
    xall = np.arange(2000)/10. - 50
    coeff = scipy.polyfit(xs,ysi,1)
    line = scipy.polyval(coeff,xall)

    coeffn = scipy.polyfit(xs,ysni,1)
    linen = scipy.polyval(coeffn,xall)

    #dataTW = 7.4
    #eTWi = 2.3
    # as of 8/7/2017
    dataTW=-18.13
    eTWi = 5.4
    dataTWi = dataTW/sini

    idx = np.argmin(np.abs(line-dataTWi))
    idxlo = np.argmin(np.abs(line-dataTWi-eTWi))

    idxn = np.argmin(np.abs(linen-dataTWi))
    idxlon = np.argmin(np.abs(linen-dataTWi-eTWi))

    dataPrec = xall[idx]
    ePrec = np.abs(xall[idx]-xall[idxlo])

    print 'TW data precession from best-fit orietnation = %6.2f +/- %6.2f' % (dataPrec, ePrec)

    dataPrecn = xall[idxn]
    ePrecn = np.abs(xall[idxn]-xall[idxlon])

    print 'TW data precession from nonaligned orietnation = %6.2f +/- %6.2f' % (dataPrecn, ePrecn)

    py.plot(xs,ysi,'ko',label='Model')
    py.plot(xall,line,'r-',label='Linear fit to models')
    py.plot([dataPrec],[dataTWi],'bx',markeredgewidth=3,label='Data')
    py.xlim(-40,80)
    py.ylim(-40,80)
    #py.axis('equal')
    py.xlabel('Input model precession')
    py.ylabel('TW output precession')
    py.title('Precessions for best-fit orientation')
    py.legend(loc=0)

    pdb.set_trace()

    py.clf()
    py.plot(xs,ysni,'ko',label='Model')
    py.plot(xall,linen,'r-',label='Linear fit to models')
    py.plot([dataPrecn],[dataTWi],'bx',markeredgewidth=3,label='Data')
    py.xlim(-40,110)
    py.ylim(-40,110)
    #py.axis('equal')
    py.xlabel('Input model precession')
    py.ylabel('TW output precession')
    py.title('Precessions for nonaligned orientation')
    py.legend(loc=0)

    pdb.set_trace()

def statsys(inCube=None,inData=None,B01=False):
    # implements Hiranya's stationary system test

    if B01:
        N = 79
        cubeimg = np.zeros((N,N))
        v = np.zeros((N,N))
        bflux = pandas.read_csv('/Users/kel/Documents/Projects/M31/HP_pattern_speed/B01_analysis/TIGERTeam/oasis_flux_m8.out',delim_whitespace=True,header=None,names=['y','x','f'])
        bvel = pandas.read_csv('/Users/kel/Documents/Projects/M31/HP_pattern_speed/B01_analysis/TIGERTeam/oasis_vel_m8.out',delim_whitespace=True,header=None,names=['y','x','v'])
        for i in np.arange(len(bflux['f'])):
            tx = int(bflux['x'][i])
            ty = int(bflux['y'][i])
            cubeimg[tx-1,ty-1] = float(bflux['f'][i])
            v[tx-1,ty-1] = float(bvel['v'][i])

        idx = np.where(cubeimg > 1000)
        cubeimg[idx] = 0.
        astep = 0.051
        #bh = [N/2+1,N/2+1]
        bh = [N/2,N/2]
        pdb.set_trace()
    else:
        cube = pyfits.getdata(inCube)
        cubeimg = cube.sum(axis=2)
        cubeimg = np.rot90(cubeimg.T,3)
        p = ppxf_m31.PPXFresults(inputFile=inData)

        v = p.velocity + ppxf_m31.vsys
        v = np.rot90(v.T,3)

        astep = 0.05

        bh = ppxf_m31.bhpos_pix
        #pdb.set_trace()

    npsi = 60.
    res = np.zeros(npsi)
    psi = np.arange(npsi,dtype='int16')*(2*np.pi/npsi)
    ecc = 0.

    for ip in np.arange(npsi):
        psi_in = psi[ip]
        rmin = 0.
        rmax = .65
        AB_lim = [rmin,rmax]
        PQ_lim = [0.,2.*np.pi]

        def int_2d(integral=None,xlim=None,ylim=None,step=None):
            rlin = np.linspace(xlim[0], xlim[1], step)
            plin = np.linspace(ylim[0], ylim[1], step)
            R,P = np.meshgrid(rlin, plin)
            
            if integral == 'int1':
                angle = P+psi_in
                xx = R*np.cos(angle)
                yy = R*np.sin(angle)
                ii = bh[0] + np.round(xx/astep)
                jj = bh[1] + np.round(yy/astep)
                xx2 = R*np.cos(P)
                yy2 = R*np.sin(P)
                ii2 = bh[0] + np.round(xx2/astep)
                jj2 = bh[1] + np.round(yy2/astep)
                out = np.zeros((step,step))
                for xi in np.arange(step):
                    for xj in np.arange(step):
                        out[xi,xj] = R[xi,xj]*v[jj2[xi,xj],ii2[xi,xj]]*cubeimg[jj[xi,xj],ii[xi,xj]]

                #pdb.set_trace()
            if integral == 'int2':
                angle = P
                xx = R*np.cos(angle)
                yy = R*np.sin(angle)
                ii = bh[0] + np.round(xx/astep)
                jj = bh[1] + np.round(yy/astep)
                out = np.zeros((step,step))
                for xi in np.arange(step):
                    for xj in np.arange(step):
                        out[xi,xj] = R[xi,xj]*cubeimg[jj[xi,xj],ii[xi,xj]]
                #pdb.set_trace()

            I = np.zeros(step)
            for i in range(step):
                I[i] = np.trapz(out[i,:],rlin)

            F = np.trapz(I,plin)

            return F
        
        #tmp1 = int_2d('int1',AB_lim,PQ_lim,30)
        #tmp2 = int_2d('int2',AB_lim,PQ_lim,30)

        #pdb.set_trace()
        res[ip] = int_2d('int1',AB_lim,PQ_lim,30) / int_2d('int2',AB_lim,PQ_lim,30)

    #py.close(1)
    py.figure(1)
    py.plot(np.degrees(psi),res)#,'g-')
    #pdb.set_trace()
    if B01:
        py.ylim(-20,20)
        py.xlim(0,360)
        py.xlabel('$\psi$')
        py.ylabel('h($\psi$) (km/s)')
        py.title('B01 OASIS data')
    else:
        py.ylim(-50,20)
        py.xlim(0,360)
        py.xlabel('$\psi$')
        py.ylabel('h($\psi$) (km/s)')
        #py.title('OSIRIS data, rmax=0.85 arcsec')

def litPrecess():
    # last number of year published 200X
    year = np.array([0,1,1,1,2,4])
    # in km/s/pc
    # some literature values are given as omega * sin 77/sin i
    # inputting my value of sin i
    sin77sini = np.sin(np.radians(77.))/np.sin(np.radians(44.))
    precess = np.array([34.*sin77sini,3.,16.,13.6,16.,36.5])
    # [Sambhus+Sridhar00, Bacon01, Jacobs+01, Salow+Statler01, Sambhus+02, Salow+04]
    # 0=HST/FOC, 1=OASIS, 2=CFHT/SIS, 3=STIS, 4=combo
    data = np.array([0,1,2,0,1,4])
    # theory (1)/data (0) papers
    theory = np.array([1,0,1,1,1,1])
    err = [5,0,0,0,0,4.2,5]
    lituplims = [1,0,0,0,0,0,0]

    idx0 = np.where(data == 0)
    idx1 = np.where(data == 1)
    idx2 = np.where(data == 2)
    idx4 = np.where(data == 4)

    py.close(3)
    py.figure(3,figsize=(9,7))
    py.subplots_adjust(left=0.1, right=0.94, top=0.95)
    py.errorbar([0,1,1,1,2,4,7],[34.*sin77sini,3.,16.,13.6,16.,36.5,0],yerr=err,ecolor='black',fmt='none',uplims=lituplims)
    py.plot(year[idx0],precess[idx0],'ko',label='HST/FOC',markersize=14)
    py.plot(year[idx2],precess[idx2],'m^',label='CFHT/SIS',markersize=14)
    py.plot(year[idx4],precess[idx4],'bD',label='CFHT/SIS + HST/STIS',markersize=14)
    py.plot(year[idx1],precess[idx1],'gs',label='OASIS IFS',markersize=14)
    py.plot(7,0+.4,'r*',label='OSIRIS IFS',markersize=20)
    
    

    py.xticks([0,1,2,3,4,5,6,7],['2000','2001','2002','2003','2004',' ',' ','2017'])
    py.xlabel('Year published')
    py.ylabel('Precession (km s$^{-1}$ pc$^{-1}$)')
    py.axhspan(0,10, alpha=0.15, color='gray')
    py.xlim(-1,8)
    py.ylim(0,50)
    py.legend(loc=0)
    
