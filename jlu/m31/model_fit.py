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

def modelConvertBin(testAngles=None,l98bin=False,verbose=True,tess=True,pathstem=modeldir+'nonaligned_grid_rotate/'):
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
    else:
        # using the nonaligned angles from Peiris & Tremaine 2003
        thetaL = np.radians(-42.8)
        thetaI = np.radians(54.1)
        thetaA = np.radians(-34.5)
        thetaCPA = np.radians(-56.)

    # matrices to rotate from disk plane to sky plane (P&T 2003, eq 4)
    matL = np.matrix([[np.cos(thetaL),-np.sin(thetaL),0.],[np.sin(thetaL),np.cos(thetaL),0.],[0.,0.,1.]])
    matI = np.matrix([[1.,0.,0.],[0.,np.cos(thetaI),-np.sin(thetaI)],[0.,np.sin(thetaI),np.cos(thetaI)]])
    matA = np.matrix([[np.cos(thetaA),-np.sin(thetaA),0.],[np.sin(thetaA),np.cos(thetaA),0.],[0.,0.,1.]])

    # these are rotation matrices, so can do the constant part of the dot product separately
    matLIA = matL.dot(matI).dot(matA)

    # stack positions, velocities to begin the coordinate transform
    xyz = np.column_stack((model.x,model.y,model.z))
    vxyz = np.column_stack((model.v_x,model.v_y,model.v_z))

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

    if verbose:
        print "Starting binning"

    if tess:
        # performs the tessellated binning on the models *before* doing the LOSVD fits
        # this function needs to read in a file from disk so first writing out the
        # transformed coordinates so it can grab those
        # writing out V_Z, *not* LOS V as the sign flip is taken care of in the tessellation routine
        if verbose:
            print "Tessellating and binning models"
        posvel = np.column_stack((X,Y,Z,V_X,V_Y,outV_Z))
        #if testAngles:
        #    outCoordFile = modeldir + 'nonaligned_model_OSIRIScoords_%.1f_%.1f_%.1f_%.1f.dat' % (testAngles[0],testAngles[1],testAngles[2],testAngles[3])
        #else:
        #    outCoordFile = modeldir + 'nonaligned_OSIRIScoords_test.dat'
        #np.savetxt(outCoordFile,np.c_[X,Y,Z,V_X,V_Y,outV_Z],fmt='%8.6f',delimiter='\t')
        voronoi_m31.tessModels(inputModelArr=posvel,inputVoronoiFile=datadir+'voronoi_2d_binning_output_20160825.txt',l98bin=l98bin,outFile=pathstem+'nonaligned_OSIRIScoords_fit_full_%.1f_%.1f_%.1f_%.1f.dat' % (testAngles[0],testAngles[1],testAngles[2],testAngles[3]))
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

        # initializing flux and kinematic arrays
        fitlosv = np.zeros((nxbin,nybin))
        sigma = np.zeros((nxbin,nybin))
        h3 = np.zeros((nxbin,nybin))
        h4 = np.zeros((nxbin,nybin))
    
        # left/bottom edges of each spaxel (in units of pc)
        leftxbound = np.arange(-1.*negxbin*binpc, (posxbin*binpc) + (0.5*binpc), binpc)
        bottomybound = np.arange(-1*negybin*binpc, (posybin*binpc) + (0.5*binpc), binpc)
        # binning LOS velocity in bins of 5 km/s, with cuts at +/- 1000 km/s
        vzbins = np.arange(-1000., 1005., 5.)

        # create the histogram and count up the number of particles in each velocity bin
        bins = (leftxbound, bottomybound, vzbins)
        losvd, bins_new, bin_num = scipy.stats.binned_statistic_dd((X, Y, losV),
                                                                losV,
                                                                statistic='count',
                                                                bins=bins)

        # create the histogram and count up the number of particles in each spatial bin
        nstar, bins_new2, bin_num2 = scipy.stats.binned_statistic_dd((X, Y),
                                                                    losV,
                                                                    statistic='count',
                                                                    bins=(leftxbound, bottomybound))

        # this loop does the LOSVD fits for each bin, to get the kinematics
        for i in range(int(nxbin)):
            #print "Starting column ", i
            for j in range(int(nybin)):
                # if there are no particles, set everything to zero
                if nstar[i,j] == 0:
                    fitlosv[i,j] = 0.
                    sigma[i,j] = 0.
                    h3[i,j] = 0.
                    h4[i,j] = 0.
                # if there are too few particles to perform a fit, hack something together 
                elif losvd[i,j,:].max() <= 5:
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
                    popt, pcov = curve_fit(ppxf_m31.gaussHermite, bins_new[2][0:-1], losvd[i,j,:], p0=guess)
                    # popt = [gamma, v, sigma, h3, h4]
                    fitlosv[i,j] = popt[1]
                    sigma[i,j] = popt[2]
                    h3[i,j] = popt[3]
                    h4[i,j] = popt[4]                                    

        # trim models to 100 x 100, with BH at center (+ xfrac, + yfrac)
        xlo = np.floor(modbhpos[0]-50)
        xhi = np.ceil(modbhpos[0]+50)
        ylo = np.floor(modbhpos[1]-50)
        yhi = np.ceil(modbhpos[1]+50)
        nstarOld = nstar
        nstar = nstar[xlo:xhi,ylo:yhi]
        fitlosv = fitlosv[xlo:xhi,ylo:yhi]
        sigma = sigma[xlo:xhi,ylo:yhi]
        h3 = h3[xlo:xhi,ylo:yhi]
        h4 = h4[xlo:xhi,ylo:yhi]

        pdb.set_trace()
                
        if testAngles:
            outFile = pathstem + 'nonaligned_OSIRIScoords_fit_full_%.1f_%.1f_%.1f_%.1f.dat' % (testAngles[0],testAngles[1],testAngles[2],testAngles[3])
        else:
            outFile = modeldir + 'nonaligned_OSIRIScoords_fits_full_test.dat'
            
        output = open(outFile, 'w')
        pickle.dump(nstar, output)
        pickle.dump(fitlosv, output)
        pickle.dump(sigma, output)
        pickle.dump(h3, output)
        pickle.dump(h4, output)
        output.close()

    if verbose:
        print "Time elapsed: ", time.time() - t1, "s"

def modelCompRes(inModel=modeldir+'nonaligned_OSIRIScoords_test_fits_trim_tess.dat',toPlot=False):
    # routine to compare model outputs to data
    # uses existing routines to read in fitted models, smooths them,
    # and takes the residuals. Plotting is done if requested

    #ppxf_m31.smoothModels(inModel)
    inSmooth = inModel.replace('.dat','_smooth.dat')
    inData = '/Users/kel/Documents/Projects/M31/analysis_new/ifu_11_11_30/ppxf_tess_bs_2_20160825.dat'
    ppxf_m31.modelResiduals(inModel,inData,trimTess=True)

    inRes = inModel.replace('.dat','_residuals.dat')

    if toPlot:
        outtmp = inModel.replace('nonaligned_model_OSIRIScoords_fits_trim_tess','residuals')
        outplotfile = outtmp.replace('.dat','')
        ppxf_m31.plotDataModelResiduals(inData,inModel,inRes,trimTess=True,outStem=outplotfile)
    
        
def modelMorphFitGrid():
    # routine to conduct a grid search over the coordinate transformation angles
    # (angles in Peiris & Tremaine 2003 eq 4)

    # create the new folder
    FORMAT = '%Y%m%d_%H-%M-%S'
    now = datetime.datetime.now().strftime(FORMAT)
    newfolder = modeldir+'nonaligned_grid_rotate_'+now+'/'
    os.mkdir(newfolder)

    # original angles
    Lorg = -42.8
    Iorg = 54.1
    Aorg = -34.5
    # not changing CPA
    CPA = -56.

    # angle step size, in degrees
    dAng = 3.
    # number of steps away from the original angle in each direction (-/+)
    # halfN = 1 gives 3 total steps, =2 gives 5 steps
    halfN = 1.
    Lgrid = np.arange(Lorg-(halfN*dAng),Lorg+(halfN*dAng)+1,dAng)
    Igrid = np.arange(Iorg-(halfN*dAng),Iorg+(halfN*dAng)+1,dAng)
    Agrid = np.arange(Aorg-(halfN*dAng),Aorg+(halfN*dAng)+1,dAng)

    for LL in Lgrid:
        for II in Igrid:
            for AA in Agrid:
                testAng = [LL,II,AA,CPA]
                modelConvertBin(testAngles=testAng,l98bin=False,verbose=False,tess=True,pathstem=newfolder)
                modelCompRes(inModel=newfolder+'nonaligned_OSIRIScoords_fit_full_%.1f_%.1f_%.1f_%.1f.dat' % (testAng[0],testAng[1],testAng[2],testAng[3]),toPlot=True)
                
