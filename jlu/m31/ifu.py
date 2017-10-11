import math
from astropy.io import fits as pyfits
import numpy as np
import pylab as py
import scipy
# import asciidata
from scipy import signal
# from pyraf import iraf as ir
from jlu.osiris import cube as cube_code
from jlu.osiris import spec
from jlu.m31 import ppxf_m31
import glob
import os
import pdb
import atpy

#datadir = '/u/jlu/data/m31/08oct/081021/SPEC/reduce/m31/ss/'
datadir = '/Users/kel/Documents/Projects/M31/analysis_new/ifu_11_11_30/'
datadir2010 = '/Users/kel/Documents/Projects/M31/analysis_new/ifu_11_11_30/'
#datadir2010 = '/u/jlu/data/m31/10aug/mosaic/'

#workdir = '/u/jlu/work/m31/nucleus/ifu_09_02_24/'
#workdir = '/u/jlu/work/m31/nucleus/ifu_11_11_30/'
workdir = '/Users/kel/Documents/Projects/M31/analysis_new/ifu_11_11_30/'

#cuberoot = 'm31_08oct_Kbb_050'
#cuberoot = 'm31_Kbb_050'
#cuberoot = 'm31_all_scalederr_cleanhdr'
#cuberoot = 'm31_all_scalederr_cleanhdr_vorcube'
#cuberoot = 'm31_all_scalederr_cleanhdr_bulgesub'
cuberoot = 'm31_all_scalederr_cleanhdr_bulgesub_vorcube_20160825'

#m31pos = np.array([9.0, 38.0])
#m31pos = np.array([22.5, 37.5])
m31pos = ppxf_m31.bhpos/0.05
paSpec = 56.0

lineList = {2.107: 'Mg I',
            2.117: 'Al I', 
            2.146: 'Mg I', 
            2.166: 'Br-gamma',
            2.208: 'Na I', 
            2.226: 'Fe I', 
            2.239: 'Fe I', 
            2.263: 'Ca I',
            2.281: 'Mg I', 
            2.294: '12CO 2-0', 
            2.323: '12CO 3-1', 
            2.345: '13CO 2-0',
            2.352: '12CO 4-2'}

def get_all_cubes(year):
    if year == 2008:
        cubes = ['a019001',
                 'a020001', 'a020002', 'a020003', 'a020004',
                 'a021001', 'a021002', 'a021003', 'a021004', 
                 'a022001', 'a022002', 'a022003', 'a022004']
    
        for cc in range(len(cubes)):
            cubes[cc] = 's081021_' + cubes[cc] + '_tlc_Kbb_050.fits'

    if year == 2010:
        cubes = glob.glob(datadir2010 + '*_tlc_Kbb_050.fits')

        for cc in range(len(cubes)):
            cubes[cc] = cubes[cc].replace(datadir2010, '')
        
    return cubes


def osiris_performance_all(year):
    cubes = get_all_cubes(year)

    for cube in cubes:
        print('Calling osiris_peformance for ' + cube)
        osiris_performance(cube)

def make_cube_image(cubefile, rootdir=datadir, clobber=False):
    if clobber or not os.path.isfile(rootdir+cubefile.replace('.fits', '_img.fits')):
        return cube_code.collapse_to_image(rootdir + cubefile)

def tweakShifts(rootdir=datadir, frameimgdir=datadir, inCubeImg=None, frameShift=True, plot=False):
    """
    Default (frameShift=True): Given an input shifts file (shifts of each OSIRIS frame w.r.t. the NIRC2 frame),
    first shift the frame by the given amount then perform small shifts to
    tweak the shifts. Performing a least squares optimization for now.

    frameShift=False: Instead takes the final OSIRIS mosaic and finds the shift between
    it and the NIRC2 frame and writes it out to a file
    """
    
    # Read in the cube... assume it is K-band
    if inCubeImg is None:
        cube, cubehdr = pyfits.getdata(rootdir + cuberoot + '.fits', header=True)
    else:
        cube, cubehdr = pyfits.getdata(inCubeImg, header=True)

    # Read in the NIRC2 image (scale = 10 mas/pixel)
    nirc2file = '/Users/kel/Documents/Projects/M31/data/combo/m31_05jul_kp.fits'
    imgorg, imghdr = pyfits.getdata(nirc2file, header=True)

    # Get the PA of the OSIRIS spectrograph image
    specPA = cubehdr['PA_SPEC']
    
    # Get the PA of the NIRC2 image
    imagPA = imghdr['ROTPOSN'] - imghdr['INSTANGL']

    # Rotate the NIRC2 image
    angle = specPA - imagPA
    imgrot = scipy.misc.imrotate(imgorg, angle, interp='bicubic')

    # Get the shifts constructed manually
    xshift = 0
    yshift = 0

    def shift2nirc2(osirisimg,nirc2imgorg,nirc2imgrot,inxshift,inyshift,plot=plot):
        # NIRC2 image
        img = imgorg
        # Trim down the NIRC2 image to the same size as the OSIRIS image.
        # Still bigger than OSIRIS FOV
        # the -3/+1 offsets are to account for the difference between the 0th OSIRIS
        # frame and the NIRC2 frame - all other OSIRIS frame offsets are w.r.t. the 0th frame
        ycent = int(round((img.shape[0] / 2.0)  - (3.*5) + (inyshift*5.0) ))
        xcent = int(round((img.shape[1] / 2.0)  + (1.*5) + (inxshift*5.0) ))
        print('')
        print('Comparing OSIRIS with NIRC2 Image:')
        print('  NIRC2 xcent = ', xcent, '  ycent = ', ycent)

        yhalf = (osirisimg.shape[0] / 2.) * 5 # Make the image the same size
        xhalf = (osirisimg.shape[1] / 2.) * 5 # as the OSIRIS cube image

        img = imgrot[ycent-yhalf:ycent+yhalf, xcent-xhalf:xcent+xhalf]
        img = img.astype(float)

        # Rebin the NIRC2 image to the same 50 mas plate scale as the
        # OSIRIS image.
        img = scipy.misc.imresize(img, 0.2) # rebin to 50 mas/pixel.
        img = img.astype(float)
        
        # Save the modified NIRC2 image.
        #nirc2_file = datadir + 'data/osiris_perf/nirc2_ref_'+str(rr)+'.fits'
        #ir.imdelete(nirc2_file)
        #pyfits.writeto(nirc2_file, img, header=imghdr, output_verify='silentfix')

        # Clean up the cube image to get rid of very very low flux values 
        # (on the edges).
        cidx = np.where(osirisimg < osirisimg.max()*0.05)
        osirisimg[cidx] = 0

        # mask out the edges of the frame
        mask = maskOSIRIS(osirisimg)
        midx = np.where(mask == 0)
        osirisimg[midx] = 0

        osirisimg_norm = osirisimg / osirisimg.sum()
        img_norm = img / img.sum()

        img_norm[cidx] = 0
        img_norm[midx] = 0

        # get a refererence correlation
        testcorr_img = scipy.signal.fftconvolve(osirisimg_norm, osirisimg_norm[::-1,::-1], mode='same')
        testcorr = np.unravel_index(np.argmax(testcorr_img), testcorr_img.shape)
        # get the real correlation
        corr_img = scipy.signal.fftconvolve(osirisimg_norm, img_norm[::-1,::-1], mode='same')
        corr = np.unravel_index(np.argmax(corr_img), corr_img.shape)

        # take the difference to get the shift
        dx = testcorr[1] - corr[1]
        dy = testcorr[0] - corr[0]
        
        
        print("dx = ", dx, end=' ')
        print("dy = ", dy)

        # testing the new shift
        ycentnew = int(round((imgorg.shape[0] / 2.0)  - (3.*5) + ((yshift+dy)*5.0) ))
        xcentnew = int(round((imgorg.shape[1] / 2.0)  + (1.*5) + ((xshift+dx)*5.0) ))
        newimg = imgrot[ycentnew-yhalf:ycentnew+yhalf, xcentnew-xhalf:xcentnew+xhalf]
        newimg = scipy.misc.imresize(newimg, 0.2) # rebin to 50 mas/pixel.
        newimg = newimg.astype(float)
        newimg[cidx] = 0
        newimg[midx] = 0
        
        if plot:
                py.figure(1)
                py.clf()
                py.subplots_adjust(left=0.05, right=0.95, bottom=0.05, top=0.95)
            
                py.subplot(1, 3, 1)
                py.imshow(osirisimg_norm)
                py.title('OSIRIS')
                
                py.subplot(1, 3, 2)
                py.imshow(img_norm)
                py.title('NIRC2+')
            
                py.subplot(1, 3, 3)
                py.imshow(newimg)
                py.title('Test new shifts')

                #py.savefig(datadir + 'data/osiris_shift/osir_shift_' + 
                #        filerr.replace('.fits', '.png'))
                py.show()

        #pdb.set_trace()

        return dx,dy
    
    if frameShift:
        # get the preliminary shifts of each frame
        shiftsTable = asciidata.open('/Users/kel/Documents/Projects/M31/data/osiris_mosaics/shifts_all_wfilename.txt')

        xshiftnew = []
        yshiftnew = []
        filelist = []
    
        for rr in range(shiftsTable.nrows):
            if (rr == 0):
                xshift0 = float(shiftsTable[1][rr])
                yshift0 = float(shiftsTable[0][rr])
                xshift = float(shiftsTable[1][rr])
                yshift = float(shiftsTable[0][rr])
                filerr = str(shiftsTable[2][rr]).strip()+'.fits'
            else:
                xshift = float(shiftsTable[1][rr])
                yshift = float(shiftsTable[0][rr])
                filerr = str(shiftsTable[2][rr]).strip()+'.fits'

            filelist.append(shiftsTable[2][rr])
            frameimg = pyfits.getdata(frameimgdir + filerr.replace(".fits", '_img.fits'))

            dx, dy = shift2nirc2(frameimg,imgorg,imgrot,xshift,yshift)

            totx = xshift+dx
            toty = yshift+dy

            xshiftnew.append(totx)
            yshiftnew.append(toty)

            _out = open(datadir + 'data/osiris_shift/osir_shift_' + filerr.replace('.fits', '_params.txt'), 'w')
            _out.write('dx: %4.1f dy: %4.1f ' % (dx, dy))
            _out.write('totx: %4.1f toty: %4.1f ' % (totx, toty))
            _out.close()

        np.savetxt(datadir + 'data/osiris_shift/shifts_all_wfilename.txt', np.c_[yshiftnew, xshiftnew, filelist],
                fmt='%s %s %s',delimiter='\t')
    else:
        dx, dy = shift2nirc2(cube,imgorg,imgrot,xshift,yshift)

        _out = open(inCubeImg.replace('.fits','_shift2nirc2.txt'), 'w')
        _out.write('%4.1f %4.1f %s ' % (dy, dx, os.path.basename(inCubeImg.replace('.fits',''))))
        _out.close()
        
def getDtotoffShifts(inFolder=None,inFile=None,outPath=None):
    if inFolder is not None:
        files = glob.glob(inFolder + '/*_Kbb_050.fits')
    else:
        files = inFile

    dtotoff1 = []
    dtotoff2 = []
    outfiles = []
    for ff in files:
        cube, hdr = pyfits.getdata(ff,header=True)
        dtotoff1.append(hdr['DTOTOFF1'])
        dtotoff2.append(hdr['DTOTOFF2'])
        outfiles.append(ff.split('/')[-1])

    np.savetxt(outPath+'/shifts_DTOTOFF_wfilename.txt', np.c_[dtotoff1,dtotoff2,outfiles],
                fmt='%s %s %s',delimiter='\t')
        
def osiris_performance(cubefile, rootdir=datadir, plotdir=workdir, framedir=datadir, frameimgdir=datadir, shiftFile=None, frameQuality=False, twoGauss=False):
    """
    Determine the spatial resolution of an OSIRIS data cube image by
    comparing with a NIRC2 image. The NIRC2 image is convolved with a
    gaussian, rebinned to the OSIRIS plate scale and differenced until
    the optimal gaussian width is determined. 
    """
    
    # Read in the cube... assume it is K-band
    #cube, cubehdr = pyfits.getdata(rootdir + cubefile, header=True)
    cube, cubehdr = pyfits.getdata(cubefile, header=True)

    #cubeimg = make_cube_image(cubefile, rootdir=rootdir)
    #if cubeimg == None:
    print('Opening previosly existing cube image.')
    #cubeimg = pyfits.getdata(rootdir + cubefile.replace(".fits", '_img.fits'))
    cubeimg, cubeimghdr = pyfits.getdata(cubefile.replace(".fits", '_img.fits'), header=True)
        #cubeimg = pyfits.getdata(cubeimgfile)
    
    
    ### Register the NIRC2 image to the OSIRIS image.
    # Read in the NIRC2 image (scale = 10 mas/pixel)
    #nirc2file = '/u/jlu/data/m31/05jul/combo/m31_05jul_kp.fits'
    nirc2file = '/Users/kel/Documents/Projects/M31/data/combo/m31_05jul_kp.fits'
    imgorg, imghdr = pyfits.getdata(nirc2file, header=True)

    # Get the PA of the OSIRIS spectrograph image
    specPA = cubehdr['PA_SPEC']
    
    # Get the PA of the NIRC2 image
    imagPA = imghdr['ROTPOSN'] - imghdr['INSTANGL']

    # Rotate the NIRC2 image
    angle = specPA - imagPA
    imgrot = scipy.misc.imrotate(imgorg, angle, interp='bicubic')

    # Get the shifts constructed manually
    xshift = 0
    yshift = 0
    if frameQuality is False:
        #shiftsTable = asciidata.open(rootdir + 'shifts.txt')
        #shiftsTable = asciidata.open('/Users/kel/Documents/Projects/M31/analysis_old/ifu_11_11_30/data/shifts.txt')
        #shiftsTable = asciidata.open('/Users/kel/Documents/Projects/M31/data/osiris_mosaics/shifts_081021_cc.txt')
        if shiftFile is None:
            shiftsTable = asciidata.open('/Users/kel/Documents/Projects/M31/data/osiris_mosaics/shifts_m31mos_to_nirc2_cc.txt')
        else:
            shiftsTable = asciidata.open(shiftFile)
    else:
        #shiftsTable = asciidata.open('/Users/kel/Documents/Projects/M31/data/osiris_mosaics/shifts_all_tonirc2_wfilename.txt')
        #shiftsTable = asciidata.open('/Users/kel/Documents/Projects/M31/data/osiris_mosaics/shifts_all_wfilename.txt')
        # tweaked shifts
        if shiftFile is None:
            shiftsTable = asciidata.open('/Users/kel/Documents/Projects/M31/analysis_new/ifu_11_11_30/data/osiris_shift/shifts_all_wfilename.txt')
        else:
            shiftsTable = asciidata.open(shiftFile)
        
    for rr in range(shiftsTable.nrows):
        if (rr == 0):
    #        xshift0 = float(shiftsTable[1][rr])
    #        yshift0 = float(shiftsTable[2][rr])
            xshift0 = float(shiftsTable[1][rr])
            yshift0 = float(shiftsTable[0][rr])
            xshift = float(shiftsTable[1][rr])
            yshift = float(shiftsTable[0][rr])
            #filerr = str(shiftsTable[2][rr]).strip()+'.fits'
            filerr = str(shiftsTable[2][rr]).strip()
        else:
            xshift = float(shiftsTable[1][rr])
            yshift = float(shiftsTable[0][rr])
            #filerr = str(shiftsTable[2][rr]).strip()+'.fits'
            filerr = str(shiftsTable[2][rr]).strip()
        
        #    if shiftsTable[0][rr] == cubefile:
        #        xshift = float(shiftsTable[1][rr])
        #        yshift = float(shiftsTable[2][rr])
        #        print 'Shifting image: '
        #        print shiftsTable[0][rr], shiftsTable[1][rr], shiftsTable[2][rr]
        #        print ''
        #        break

        # Calculate the SNR for the OSIRIS data between 2.160 and 2.175 microns.
        # This is done for the same position in each of the combo + subset mosaics.
        # We need to remove the continuum first before we calculate the SNR.
        #xpixSNR = 11 + (xshift - xshift0)
        #ypixSNR = 47 + (yshift - yshift0)

        if frameQuality:
            frame, framehdr = pyfits.getdata(framedir + filerr, header=True)

            frameimg = make_cube_image(filerr, rootdir=frameimgdir)
            if frameimg == None:
                print('Opening previosly existing cube image.')
                frameimg = pyfits.getdata(frameimgdir + filerr.replace(".fits", '_img.fits'))
        else:
            frameimg = cubeimg
            framehdr = cubeimghdr
            frame = cube
                
        # if (xpixSNR < 0 or xpixSNR >= cube.shape[0] or
        #     ypixSNR < 0 or ypixSNR >= cube.shape[1]):
        tmp = np.unravel_index(frameimg.argmax(), frameimg.shape)
        xpixSNR = tmp[1]
        ypixSNR = tmp[0]
    
        # Get the wavelength solution so we can specify range:
        w0 = framehdr['CRVAL1']
        dw = framehdr['CDELT1']
        wavelength = w0 + dw * np.arange(frame.shape[2], dtype=float)
        wavelength /= 1000.0   # Convert to microns

        widx = np.where((wavelength > 2.160) & (wavelength < 2.175))
        waveCrop = wavelength[widx]
        specCrop = frame[xpixSNR, ypixSNR, widx].flatten()
        coeffs = scipy.polyfit(waveCrop, specCrop, 1)
        residuals = specCrop / scipy.polyval(coeffs, waveCrop)
        specSNR = (residuals.mean() / residuals.std())

        print('OSIRIS Signal-to-Noise:')
        print('   X = %d  Y = %d' % (xpixSNR, ypixSNR))
        print('   wavelength = [%5.3f - %5.3f]' % (2.160, 2.175))
        print('   SNR = %f' % specSNR)

        img = imgorg
        # Trim down the NIRC2 image to the same size as the OSIRIS image.
        # Still bigger than OSIRIS FOV
        #ycent = int(round((img.shape[0] / 2.0)  + (yshift*5.0) ))
        #xcent = int(round((img.shape[1] / 2.0)  + (xshift*5.0) ))
        # the -3/+1 offsets are to account for the difference between the 0th OSIRIS
        # frame and the NIRC2 frame - all other OSIRIS frame offsets are w.r.t. the 0th frame
        ycent = int(round((img.shape[0] / 2.0)  - (3.*5) + (yshift*5.0) ))
        xcent = int(round((img.shape[1] / 2.0)  + (1.*5) + (xshift*5.0) ))
        print('')
        print('Comparing OSIRIS with NIRC2 Image:')
        print('  NIRC2 xcent = ', xcent, '  ycent = ', ycent)

        #yhalf = int(frameimg.shape[0] / 2.) * 5 # Make the image the same size
        #xhalf = int(frameimg.shape[1] / 2.) * 5 # as the OSIRIS cube image.
        #yhalf = int((frameimg.shape[0] / 2.) * 5) # Make the image the same size
        #xhalf = int((frameimg.shape[1] / 2.) * 5) # as the OSIRIS cube image.
        yhalf = (frameimg.shape[0] / 2.) * 5 # Make the image the same size
        xhalf = (frameimg.shape[1] / 2.) * 5 # as the OSIRIS cube image.

        img = imgrot[ycent-yhalf:ycent+yhalf, xcent-xhalf:xcent+xhalf]

        # Rebin the NIRC2 image to the same 50 mas plate scale as the
        # OSIRIS image.
        img = scipy.misc.imresize(img, 0.2) # rebin to 50 mas/pixel.

        #pdb.set_trace()
        
        # Save the modified NIRC2 image.
        nirc2_file = rootdir + 'data/osiris_perf/nirc2_ref_'+str(rr)+'.fits'
        #ir.imdelete(nirc2_file)
        pyfits.writeto(nirc2_file, img, header=imghdr, output_verify='silentfix')

        # Clean up the cube image to get rid of very very low flux values 
        # (on the edges).
        cidx = np.where(frameimg < frameimg.max()*0.05)
        frameimg[cidx] = 0

        # mask out the bad edges of the frame
        mask = maskOSIRIS(frameimg)
        midx = np.where(mask == 0)
        frameimg[midx] = 0

        def fitfunction(params, plot=False, verbose=True, twoGauss=twoGauss):
            if twoGauss:
                amp1 = abs(params[0])
                # Actually amp1 should be fixed to 1.0
                amp1 = 1.0
                amp2 = abs(params[1])
                sigma1 = abs(params[2])
                sigma2 = abs(params[3])
                psf = twogauss_kernel(sigma1, sigma2, amp1, amp2, half_box=50)
            else:
                amp1 = abs(params[0])
                # Actually amp1 should be fixed to 1.0
                amp1 = 1.0
                sigma1 = abs(params[1])
                psf = gauss_kernel(sigma1, amp1, half_box=50)
                
            

            # Convolve the NIRC2 image with a gaussian
            boxsize = min(frameimg.shape) / 2
            newimg = signal.convolve(img, psf, mode='same')

            img2 = img.astype(float)
            if verbose:
                print('fitfunc shapes:', end=' ')
                print(' cubeimg = ', frameimg.shape, end=' ') 
                print(' psf = ', psf.shape, end=' ')
                print(' newimg = ', newimg.shape)
                print(' img = ', img.shape)

            newimg[cidx] = 0
            newimg[midx] = 0
            img2[cidx] = 0
            img2[midx] = 0
        
            frameimg_norm = frameimg / frameimg.sum()
            newimg_norm = newimg / newimg.sum()
            #frameimg_norm = frameimg / frameimg.max()
            #newimg_norm = newimg / newimg.max()
        
            residuals = (frameimg_norm - newimg_norm) / np.sqrt(frameimg_norm)
            residuals[cidx] = 0
            residuals[midx] = 0

            if verbose:
                if twoGauss:
                    print('Parameters: sig1 = %5.2f  sig2 = %5.2f ' % (sigma1, sigma2), end=' ')
                    print(' amp1 = %9.2e  amp2 = %9.2e' % (amp1, amp2))
                else:
                    print('Parameters: sig1 = %5.2f ' % (sigma1), end=' ')
                    print(' amp1 = %9.2e ' % (amp1))
                print('Residuals:  ', math.sqrt((residuals*residuals).sum()))
                print('') 

            if plot:
                py.close(1)
                py.figure(1,figsize=(7,10))
                py.clf()
                py.subplots_adjust(left=0.05, right=0.95, bottom=0.08, top=0.95, hspace=.4)

                xaxis = (np.arange(frameimg_norm.shape[0]) - ppxf_m31.bhpos_pix[0]) * 0.05
                yaxis = (np.arange(frameimg_norm.shape[1]) - ppxf_m31.bhpos_pix[1]) * 0.05

                xtickLoc = py.MultipleLocator(1.0)
        
                py.subplot(3, 1, 1)
                py.imshow(np.rot90(frameimg_norm,3),extent=[xaxis[0], xaxis[-1], yaxis[0], yaxis[-1]])
                #cbar1 = py.colorbar(orientation='vertical',format='%5.4f')
                #cbar1.set_label('Flux (norm)')
                #cbar1.set_ticks([0,frameimg_norm.max()/2.,frameimg_norm.max()])
                py.gca().get_xaxis().set_major_locator(xtickLoc)
                py.title('OSIRIS (FWHM: 124 mas)')
                py.ylabel('Y (arcsec)')
                
                py.subplot(3, 1, 2)
                py.imshow(np.rot90(newimg_norm,3),extent=[xaxis[0], xaxis[-1], yaxis[0], yaxis[-1]])
                #cbar2 = py.colorbar(orientation='vertical',format='%5.4f')
                #cbar2.set_label('Flux (norm)')
                #cbar2.set_ticks([0,newimg_norm.max()/2.,newimg_norm.max()])
                py.gca().get_xaxis().set_major_locator(xtickLoc)
                py.title('NIRC2 (FWHM: 124 mas)')
                py.ylabel('Y (arcsec)')
            
                py.subplot(3, 1, 3)
                #py.imshow(residuals)
                #cbar3 = py.colorbar(orientation='vertical',format='%5.4f')
                #cbar3.set_label('Residuals')
                #cbar3.set_ticks([residuals.min(),0,residuals.max()])
                #py.title('Residuals')
                py.imshow(np.rot90(img2 / img2.sum(),3),extent=[xaxis[0], xaxis[-1], yaxis[0], yaxis[-1]])
                #cbar3.set_label('Flux (norm)')
                testmax=(img2/img2.sum()).max()*.95
                #cbar3.set_ticks([0,(img2/img2.sum()).max()/2.,testmax])
                py.gca().get_xaxis().set_major_locator(xtickLoc)
                py.title('NIRC2 (FWHM: 60 mas)')
                #py.xlabel('Distance from SMBH (arcsec)')
                py.xlabel('X (arcsec)')
                py.ylabel('Y (arcsec)')
                
                py.savefig(rootdir + 'data/osiris_perf/osir_perf_' + 
                        filerr.replace('.fits', '.png'))
                py.savefig(rootdir + 'data/osiris_perf/osir_perf_' + 
                        filerr.replace('.fits', '.eps'))
                py.show()

                pdb.set_trace()
            
            return residuals.flatten()

        if twoGauss:
            params = np.zeros(4, dtype=float)
            params[0] = 1.0 # amp1
            params[1] = 0.000001 # amp2
            params[2] = 0.6 # sigma1 (near-diffraction-limit)
            params[3] = 0.9 # sigma2 (seeing halo)
        else:
            params = np.zeros(2, dtype=float)
            params[0] = 1.0 # amp1
            #params[1] = 0.6 # sigma1 (near-diffraction-limit)
            # testing
            params[1] = 1.05
       

        print('Fitting PSF: ')
        print('')
        print('Initial: ')
        fitfunction(params, plot=True, verbose=True)

        p, cov, infodict, errmsg, success = scipy.optimize.leastsq(fitfunction, 
                                                               params, 
                                                               full_output=1,
                                                               maxfev=100)
        print('')
        print('Results:')
        residuals = fitfunction(p, plot=True, verbose=True)

        if twoGauss:
            amp1 = abs(p[0])
            amp2 = abs(p[1])
            sigma1 = abs(p[2])
            sigma2 = abs(p[3])
        else:
            amp1 = abs(p[0])
            sigma1 = abs(p[1])
    
        #pdb.set_trace()
    
        _out = open(rootdir + 'data/osiris_perf/osir_perf_' + filerr.replace('.fits', '_params.txt'), 'w')
        if twoGauss:
            _out.write('sig1: %5.2f  sig2: %5.2f  amp1: %9.2e  amp2: %9.2e  res: %7.5f  ' %
                        (sigma1, sigma2, amp1, amp2, math.sqrt((residuals**2).sum())))
        else:
            _out.write('sig1: %5.2f  amp1: %9.2e  res: %7.5f  ' %
                        (sigma1, amp1, math.sqrt((residuals**2).sum())))
        _out.write('xpixSNR: %2d  ypixSNR: %2d  SNR: %5.1f\n' % (xpixSNR, ypixSNR, specSNR))
        _out.close()

def clipFramesMosaic(inPSFpath=datadir+'data/osiris_perf/',twoGauss=False,numFrameCut=None,maxPSF=None,outpath=datadir):
    # reads in PSF parameter files, flags bad frames based on either
    # a maximum allowable PSF (in pixels, maxPSF) or flags a given number of frames with
    # highest PSF (numFrameCut)
    # creates a shifts file (txt and fits) using only the good frames
    # creates a DRF with only the good frames

    # input shifts file - from the tweaked shifts
    inShiftsFile = '/Users/kel/Documents/Projects/M31/analysis_new/ifu_11_11_30/data/osiris_shift/shifts_all_wfilename.txt'

    shiftsTable = asciidata.open(inShiftsFile)
    xshift = []
    yshift = []
    stemlist = []
    filelist = []
    drflist = []
    sig1list = []
    for rr in range(shiftsTable.nrows):
        # read in the input shifts
        xshiftrr = float(shiftsTable[1][rr])
        yshiftrr = float(shiftsTable[0][rr])
        stemrr = str(shiftsTable[2][rr]).strip()
        filerr = stemrr + '.fits'
        xshift.append(xshiftrr)
        yshift.append(yshiftrr)
        stemlist.append(stemrr)
        filelist.append(filerr)
        drflist.append('    <fits FileName="'+filerr+'" />')
        # grab the PSF size 
        psffile = 'osir_perf_' + stemrr + '_params.txt'
        tmppsf = ppxf_m31.readPSFparams(inPSFpath+psffile,twoGauss=twoGauss)
        sig1list.append(tmppsf.sig1[0])

    sig1list = np.array(sig1list)
    xshift = np.array(xshift)
    yshift = np.array(yshift)
    filelist = np.array(filelist)
    drflist = np.array(drflist)

    if numFrameCut is not None:
        sortpsf = np.sort(sig1list)
        sortidx = -1*numFrameCut
        tmpmax = sortpsf[sortidx]
        #pdb.set_trace()
        idx = np.where(sig1list <= tmpmax)
        outxshift = xshift[idx]
        outyshift = yshift[idx]
        outfilelist = filelist[idx]
        outdrflist = drflist[idx]
        outpsf = sig1list[idx]
        
    if maxPSF is not None:
        idx = np.where(sig1list <= maxPSF)
        outxshift = xshift[idx]
        outyshift = yshift[idx]
        outfilelist = filelist[idx]
        outdrflist = drflist[idx]
        outpsf = sig1list[idx]

    outyx = np.array([outyshift,outxshift])
    outyx = outyx.T
    
    np.savetxt(datadir + 'data/shifts_clip_wfilename.txt', np.c_[outyshift, outxshift, outfilelist],
               fmt='%s %s %s',delimiter='\t')
    
    pyfits.writeto(datadir + 'data/shifts_clip_wfilename.fits', outyx)

    np.savetxt(datadir + 'data/filelist.txt',np.c_[outdrflist],
               fmt = '%s')


def compPSF(inPSFpath=datadir+'data/osiris_perf/',twoGauss=False,toTxt=False,numclip=None):
    # compare individual frame PSFs from fitting routine

    PSFfiles = glob.glob(inPSFpath + '*_params.txt')

    filename = []
    sigall = []
    sigas = []
    fwhm = []
    for ff in range(len(PSFfiles)):
        filename.append(os.path.basename(PSFfiles[ff]))
        tmp = ppxf_m31.readPSFparams(PSFfiles[ff],twoGauss=twoGauss)
        sigall.append(tmp.sig1[0])
        sigas.append(0.05*tmp.sig1[0])
        fwhm.append(2.35*0.05*tmp.sig1[0])

    if numclip is not None:
        sigall = np.array(sigall)
        sigas = np.array(sigas)
        fwhm = np.array(fwhm)
        sortpsf = np.sort(sigall)
        sortidx = -1*numclip
        tmpmax = sortpsf[sortidx]
        #pdb.set_trace()
        idx = np.where(sigall <= tmpmax)
        sigas = sigas[idx]
        fwhm = fwhm[idx]
        
    fwhmmed = np.median(fwhm)
    print('Median FWHM is ', fwhmmed)

    if toTxt:
        np.savetxt(datadir + 'data/osiris_perf/sig1_all_wfilename.txt', np.c_[sigall, filename],
               fmt='%s %s',delimiter='\t')
    
    # files are in units of pixels, convert to arcsec
    py.close(3)
    py.figure(3)
    py.hist(fwhm,bins=10)
    py.xlabel('PSF FWHM (arcsec)')
    py.ylabel('Number')
    py.annotate('Median FWHM = '+str(fwhmmed),xy=(.6,.9), xycoords='axes fraction')

    pdb.set_trace()

def twogauss_kernel(sigma1, sigma2, amplitude1, amplitude2, half_box=50):
    """
    This is a specialized two-gaussian PSF kernel. For simplicity,
    we assume circular symmetry.
    """
    if (half_box < 3*sigma1):
        print('PSF width is too big (%5.2f pixels) for the ' % sigma1)
        print('box size (%3d pixels). Change sigma1.'  % (2*half_box))
        return

    # Create a 2D grid of X and Y positions in our PSF
    x, y = scipy.mgrid[-half_box:half_box+1, -half_box:half_box+1]

    # Create the two gaussians
    g1 = amplitude1 * np.exp( -( (x/sigma1)**2 + (y/sigma1)**2 ) )
    g2 = amplitude2 * np.exp( -( (x/sigma2)**2 + (y/sigma2)**2 ) )

    # Sum and normalize the gaussians
    psf = g1 + g2
    psf /= psf.sum()

    return psf

def gauss_kernel(sigma1, amplitude1, half_box=50):

    if (half_box < 3*sigma1):
        print('PSF width is too big (%5.2f pixels) for the ' % sigma1)
        print('box size (%3d pixels). Change sigma1.'  % (2*half_box))
        return

    # Create a 2D grid of X and Y positions in our PSF
    x, y = scipy.mgrid[-half_box:half_box+1, -half_box:half_box+1]

    # Create the  gaussian
    g1 = amplitude1 * np.exp( -( (x/sigma1)**2 + (y/sigma1)**2 ) )

    # Sum and normalize the gaussians
    psf = g1
    psf /= psf.sum()

    return psf

def gauss_kernel1D(sigma1, amplitude1, half_box=50):

    if (half_box < 3*sigma1):
        print('PSF width is too big (%5.2f pixels) for the ' % sigma1)
        print('box size (%3d pixels). Change sigma1.'  % (2*half_box))
        return

    # Create an x array
    #dx = sigma1/10.
    x = np.arange(-half_box,half_box+1,1)

    # Create the  gaussian
    g1 = amplitude1 * np.exp( -(x**2)/(2.*(sigma1**2)))

    # Sum and normalize the gaussians
    psf = g1
    psf /= psf.sum()

    #pdb.set_trace()
    return psf

def maskOSIRIS(img=None):
    # returns a mask for a single OSIRIS frame
    # default method is to just clip off some columns on either side
    # to be added: more sophisticated method that clips along the slanted sides

    imgshape = img.shape
    mask = np.ones(imgshape,dtype=int)

    mask[0:4,:] = 0
    mask[-1,:] = 0
    mask[-2,:] = 0
    mask[-3,:] = 0
    mask[-4,:] = 0
    mask[:,0:4] = 0
    mask[:,-1] = 0
    mask[:,-2] = 0
    mask[:,-3] = 0
    mask[:,-4] = 0

    return mask

def integrated_spectrum():
    """
    Make an integrated spectrum of the whole cube. Restrict to only 
    those pixels that 
    """
    cube, hdr = pyfits.getdata(datadir + cuberoot + '.fits', header=True)
    cubeimg = pyfits.getdata(datadir + cuberoot + '_img.fits')
    snrimg = pyfits.getdata(datadir + cuberoot + '_snr.fits')

    # Lets clean up the cube of all the edge effect stuff
    cidx = np.where(cubeimg < cubeimg.max()*0.3)
    cube[cidx[1],cidx[0],:] = 0
    
    # We can also trim based on SNR
    sidx = np.where(snrimg > 40)

    # Trim off the edges of the cube
    spectrum = cube[sidx[1],sidx[0],:].sum(axis=0)
    #spectrum = cube[5:17,20:65,:].sum(axis=1).sum(axis=0)
    #spectrum = cube.sum(axis=1).sum(axis=0)

    # Now get the wavelength array
    w0 = hdr['CRVAL1']
    dw = hdr['CDELT1']
    
    wavelength = w0 + dw * np.arange(len(spectrum), dtype=float)
    wavelength *= 1e-3  # convert to microns

    ##########
    # Plot region used in the integrated spectrum
    ##########
    py.figure(1)
    py.clf()
    py.subplot(1, 2, 1)
    imshow_cube_image(cubeimg, header=hdr, blackhole=True, compass=True)

    py.subplot(1, 2, 2)
    usedPixels = np.zeros(cubeimg.shape)
    usedPixels[cidx] = 1
    imshow_cube_image(usedPixels, blackhole=True, compass=False)

    py.savefig(workdir + 'plots/integrated_spectrum_img.png')


    ##########
    # Plot spectrum
    ##########
    py.figure(2, figsize=(10,6))
    py.clf()
    py.plot(wavelength, spectrum)
    py.xlabel('Wavelength (microns)')
    py.ylabel('Arbitrary Flux')
    limits = py.axis()
    
    # Label lines
    lineWave = list(lineList.keys())
    lineName = list(lineList.values())
    for ll in range(len(lineWave)):
        xhash = [lineWave[ll], lineWave[ll]]
        yhash = [limits[3]*0.82, limits[3]*0.85]
        py.plot(xhash, yhash, 'k-')
        py.text(xhash[1], yhash[1], lineName[ll], rotation='vertical',
                horizontalalignment='center', fontsize=12)

    py.savefig(workdir + 'plots/integrated_spectrum.png')


def map_snr(inCubeRoot=None,waveLo=2.160, waveHi=2.175, datadir=datadir, errext=False):
    """
    Make SNR maps for the combo and sub-maps.

    Uses an empirical estimate of the errors (e.g. from the noisiness of the continuum)
    unless otherwise specified by setting errext=True, in which case S/N is calculated
    by taking the ratio of the median signal to the median noise, taken as extension 1
    of the input cube.

    bulgesub: 2.270 - 2.280
    """

    if inCubeRoot is None:
        fitsfiles = [datadir+cuberoot]
    else:
        fitsfiles = [inCubeRoot]
    #[datadir + cuberoot,
                 #datadir + cuberoot + '_1', 
                 #datadir + cuberoot + '_2',
                 #datadir + cuberoot + '_3']
    
    for ff in range(len(fitsfiles)):
        # Read in cube and cube image. 
        #   -  cube is indexed as [y, x, lambda]
        #   -  cube iamge is indexed as [x, y]
        cubefits = pyfits.open(fitsfiles[ff] + '.fits')
        cube = cubefits[0].data
        hdr = cubefits[0].header
        err = cubefits[1].data

        cubeimg, imghdr = pyfits.getdata(fitsfiles[ff] + '_img.fits', 
                                         header=True) 

        if errext is False:
            # Calculate the SNR for the OSIRIS data between 2.160 and 2.175 microns.
            # Get the wavelength solution so we can specify range:
            w0 = hdr['CRVAL1']
            dw = hdr['CDELT1']
            wavelength = w0 + dw * np.arange(cube.shape[2], dtype=float)
            wavelength /= 1000.0   # Convert to microns
        
            # We need to remove the continuum first before we calculate the SNR.
            #widx = np.where((wavelength > 2.160) & (wavelength < 2.175))
            widx = np.where((wavelength > waveLo) & (wavelength < waveHi))
            waveCrop = wavelength[widx]

            snrMap = np.zeros(cubeimg.shape, dtype=float)

            # Loop through each spaxel and determine the SNR:
            for xx in range(cubeimg.shape[0]):
                for yy in range(cubeimg.shape[1]):
                    specCrop = cube[yy, xx, widx].flatten()
                    #specCrop = cube[xx, yy, widx].flatten()
                    coeffs = scipy.polyfit(waveCrop, specCrop, 1)
                    residuals = specCrop / scipy.polyval(coeffs, waveCrop)
                    snrMap[xx, yy] = residuals.mean() / residuals.std()

            snrFile = fitsfiles[ff] + '_snr.fits'
            ir.imdelete(snrFile)
            pyfits.writeto(snrFile, snrMap, header=imghdr, output_verify='ignore')

        else:
            # take the median along the wavelength direction (will be the longest dimension)
            waveaxis = np.argmax(cube.shape)
            sig = np.median(cube,axis=waveaxis)
            noise = np.median(err,axis=waveaxis)
            
            snrMap = sig/noise

            badidx = np.where(np.isnan == 1)
            snrMap[badidx] = 0.

            # put into the more usual shape
            snrMap = snrMap.T
            
            snrFile = fitsfiles[ff] + '_errext_snr.fits'
            ir.imdelete(snrFile)
            pyfits.writeto(snrFile, snrMap, header=imghdr, output_verify='ignore')

def plot_map_snr(datadir=workdir):
    """
    Use after calling ifu.map_snr(). Plots the cube image on the left panel
    and the SNR map on the right panel.
    """
    # Read in the M31 flux and SNR maps
    cubeimg, hdr = pyfits.getdata(datadir + cuberoot + '_img.fits', header=True)
    snrimg = pyfits.getdata(datadir + cuberoot + '_snr.fits')

    # Mask out bad pixels (where SNR couldn't be calculated)
    snrMasked = np.ma.masked_invalid(snrimg)
    cubeMasked = np.ma.array(cubeimg, mask=snrMasked.mask)

    # Setup axis info (x is the 2nd axis)
    xcube = (np.arange(cubeimg.shape[1]) - m31pos[0]) * 0.05
    ycube = (np.arange(cubeimg.shape[0]) - m31pos[1]) * 0.05
    xtickLoc = py.MultipleLocator(0.4)

    # Get the spectrograph position angle for compass rose
    pa = hdr['PA_SPEC']

    py.clf()
    py.subplots_adjust(left=0.08, top=0.95, wspace=0.1)

    ##########
    # Plot flux
    ##########
    py.subplot(1, 2, 1)
    imshow_cube_image(cubeMasked, header=hdr, cmap=py.cm.hot,
                      compass=True, blackhole=True)
    
    ##########
    # Plot SNR
    ##########
    py.subplot(1, 2, 2)
    imshow_cube_image(snrMasked, cmap=py.cm.jet,
                      compass=False, blackhole=False, vmin=0.,vmax=65.)
    cbar = py.colorbar(orientation='vertical')
    cbar.set_label('Signal-to-Noise')

    py.savefig(datadir + 'plots/map_snr.png')
    py.show()

def map_co_eqw():
    # Read in cube and cube image. 
    #   -  cube is indexed as [y, x, lambda]
    #   -  cube iamge is indexed as [x, y]
    cubefits = pyfits.open(datadir + cuberoot + '.fits')
    cube = cubefits[0].data
    cubehdr = cubefits[0].header
    cubeNoise = cubefits[1].data

    img, imghdr = pyfits.getdata(datadir + cuberoot + '_img.fits', header=True)

    # Get the wavelength solution.
    w0 = cubehdr['CRVAL1']
    dw = cubehdr['CDELT1']
    wave = w0 + dw * np.arange(cube.shape[2], dtype=float)
    wave /= 1000.0   # Convert to microns
        
    ##########
    # Using the CO index from Marmol-Queralto et al. 2008.
    # Wiki Notes:
    #   http://131.215.103.243/groups/jlu/wiki/65bbf/M31__Line_Maps.html
    ##########
    # Define the wavelength regions for continuum #1, continuum #2, absorption
    waveReg = np.array([[2.2460, 2.2550], 
                        [2.2710, 2.2770], 
                        [2.2880, 2.3010]])

    # Redshift the regions for M31's systemic velocity
    m31rv = -300.99   # km/s
    waveReg *= (1.0 + m31rv / 2.99792e5)

    # Continuum region #1
    idxC1 = np.where((wave >= waveReg[0,0]) & (wave <= waveReg[0,1]))[0]
    waveC1 = wave[idxC1]

    # Continuum region #2
    idxC2 = np.where((wave >= waveReg[1,0]) & (wave <= waveReg[1,1]))[0]
    waveC2 = wave[idxC2]

    # CO region
    idxA1 = np.where((wave >= waveReg[2,0]) & (wave <= waveReg[2,1]))[0]
    waveA1 = wave[idxA1]

    # Loop through each pixel and calculate the CO index.
    coMap = np.zeros(img.shape, dtype=float)
    coMapErr = np.zeros(img.shape, dtype=float)

    contMap = np.zeros(img.shape, dtype=float)

    py.clf()
    for yy in range(img.shape[0]):
        for xx in range(img.shape[1]):
            fluxC1 = cube[xx, yy, idxC1].sum()
            fluxC2 = cube[xx, yy, idxC2].sum()
            fluxA1 = cube[xx, yy, idxA1].sum()

            if yy == 46 and xx == 10:
                py.plot(waveC1, cube[xx, yy, idxC1], 'b-')
                py.plot(waveC2, cube[xx, yy, idxC2], 'b-')
                py.semilogy(waveA1, cube[xx, yy, idxA1], 'b-')
                py.title('xx=10, yy=46')
            if yy == 35 and xx == 15:
                py.plot(waveC1, cube[xx, yy, idxC1], 'g-')
                py.plot(waveC2, cube[xx, yy, idxC2], 'g-')
                py.plot(waveA1, cube[xx, yy, idxA1], 'g-')
                py.title('xx=15, yy=35')

            # Need to filter out bad pixel issues in noise map... 
            # just use the median error instead.
            #varC1 = (cube[xx, yy, idxC1].std())**2
            varC1 = np.median(cubeNoise[xx, yy, idxC1])**2
            varC1 *= float(len(idxC1))

            #varC2 = (cube[xx, yy, idxC2].std())**2
            varC2 = np.median(cubeNoise[xx, yy, idxC2])**2 
            varC2 *= float(len(idxC2))

            #varA1 = (cube[xx, yy, idxC2].std())**2  # note same as C2
            varA1 = np.median(cubeNoise[xx, yy, idxA1])**2 
            varA1 *= float(len(idxA1))

            dwaveC1 = waveReg[0,1] - waveReg[0,0]
            dwaveC2 = waveReg[1,1] - waveReg[1,0]
            dwaveA1 = waveReg[2,1] - waveReg[2,0]

            contFlux = (fluxC1 + fluxC2) / ( dwaveC1 + dwaveC2)
            absoFlux = fluxA1 / dwaveA1

            contFluxVar = (varC1 + varC2) / (dwaveC1 + dwaveC2)**2
            absoFluxVar = varA1 / dwaveA1**2

            coMap[yy, xx] = contFlux / absoFlux
            coMapErr[yy, xx] = math.sqrt(((contFlux**2 * absoFluxVar) + 
                                          (absoFlux**2 * contFluxVar)) 
                                         / absoFlux**4)

            contMap[yy, xx] = (fluxC1 / dwaveC1) / (fluxC2 / dwaveC2)
    py.show()

    coFile = workdir + 'maps/co_map.fits'
    coErrFile = workdir + 'maps/co_err_map.fits'
    contFile = workdir + 'maps/cont_ratio_map.fits'

    ir.imdelete(coFile)
    ir.imdelete(coErrFile)
    ir.imdelete(contFile)

    pyfits.writeto(coFile, coMap, header=imghdr)
    pyfits.writeto(coErrFile, coMapErr, header=imghdr)
    pyfits.writeto(contFile, contMap, header=imghdr)

    # Lets plot the CO index in various ways
    x1d = np.arange(img.shape[1])
    y1d = np.arange(img.shape[0])

    y2d, x2d = scipy.mgrid[0:img.shape[0], 0:img.shape[1]]

    py.clf()
    py.errorbar(x2d.flatten(), coMap.flatten(), yerr=coMapErr.flatten(), 
                fmt='.')
    #py.show()

    py.clf()
    py.errorbar(y2d.flatten(), coMap.flatten(), yerr=coMapErr.flatten(),
                fmt='.')
    #py.show()

    # Calculate the mean CO index for the whole map. Then make
    # a map of the significance of the excess CO at each pixel
    idx = np.where(coMap > 0) # good pixels

    coAvg = coMap[idx[0],idx[1]].mean()
    coSigma = (coMap - coAvg) / coMapErr

    coSigmaMasked = np.ma.masked_invalid(coSigma)

    coSigFile = workdir + 'maps/co_sigma_map.fits'
    ir.imdelete(coSigFile)
    pyfits.writeto(coSigFile, coSigma, header=imghdr)

    py.clf()
    py.imshow(coSigmaMasked)
    py.colorbar(orientation='vertical')
    py.show()


def extract_spec():
    cubefits = pyfits.open(datadir + cuberoot + '.fits')
    cube = cubefits[0].data
    cubehdr = cubefits[0].header

    spec = cube[11, 47, :]

    fitsFile = workdir + 'maps/test_spec_11_47.fits'
    ir.imdelete(fitsFile)
    pyfits.writeto(fitsFile, spec, header=cubehdr)

def extract_spec2():
    infile = workdir + '/maps/standards/HD221246_K3III.fits'
    specInfo, hdr = pyfits.getdata(infile, header=True)
    wave = specInfo[0,:] * 1e3  # in nm
    spec = specInfo[1,:]
    print(wave[0:10])
    print(wave[-10:])
    print(spec)

    crpix1 = 1
    crval1 = wave[0]
    cdelt1 = wave[1] - wave[0]
    cunit1 = 'nm'

    tmp = np.arange(len(spec), dtype=float)
    tmp = tmp*cdelt1 + crval1
    print(tmp[0:10])
    print(tmp[-10:])


    hdr.update('CRPIX1', crpix1)
    hdr.update('CRVAL1', crval1)
    hdr.update('CDELT1', cdelt1)
    hdr.update('CUNIT1', cunit1)

    fitsFile = workdir + 'maps/test_spec_standard.fits'
    ir.imdelete(fitsFile)
    pyfits.writeto(fitsFile, spec, header=hdr)


def imshow_cube_image(image, header=None, compass=True, blackhole=True, 
                      cmap=None, vmin=None, vmax=None):
    """
    Call imshow() to plot a cube image. Make sure it is already 
    masked. Also pass in the header to do the compass rose calculations.
    """
    # Setup axis info (x is the 2nd axis)
    xcube = (np.arange(image.shape[1]) - m31pos[0]) * 0.05
    ycube = (np.arange(image.shape[0]) - m31pos[1]) * 0.05
    xtickLoc = py.MultipleLocator(0.4)

    if cmap is None:
        cmap = py.cm.jet

    # Plot the image.
    py.imshow(image, 
              extent=[xcube[0], xcube[-1], ycube[0], ycube[-1]],
              cmap=cmap, vmin=vmin, vmax=vmax)
    py.gca().get_xaxis().set_major_locator(xtickLoc)
    
    py.xlabel('X (arcsec)')
    py.ylabel('Y (arcsec)')

    # Get the spectrograph position angle for compass rose
    if compass is True:
        if header is None:
            pa = 90.-paSpec
        else:
            pa = 90.-header['PA_SPEC']
        #pdb.set_trace()

        # Make a compass rose
        cosSin = np.array([ math.cos(math.radians(pa)), 
                            math.sin(math.radians(pa)) ])
        arr_base = np.array([ xcube[-1]-0.5, ycube[-1]-0.6 ])
        arr_n = cosSin * 0.2
        arr_w = cosSin[::-1] * 0.2
        py.arrow(arr_base[0], arr_base[1], arr_n[0], arr_n[1],
                 edgecolor='w', facecolor='w', width=0.03, head_width=0.08)
        py.arrow(arr_base[0], arr_base[1], -arr_w[0], arr_w[1],
                 edgecolor='w', facecolor='w', width=0.03, head_width=0.08)
        py.text(arr_base[0]+arr_n[0]+0.1, arr_base[1]+arr_n[1]+0.1, 'N', 
                color='white', 
                horizontalalignment='left', verticalalignment='bottom')
        py.text(arr_base[0]-arr_w[0]-0.15, arr_base[1]+arr_w[1]+0.1, 'E', 
                color='white',
                horizontalalignment='right', verticalalignment='center')

    if blackhole is True:
        py.plot([0], [0], 'ko')

    py.axis('image')

    #py.xlim([-0.5, 0.6])
    #py.ylim([-2.0, 1.8])


    

def makeResolutionMapCO():
    """
    Use a specially created M31 mosaic where the sky lines have not been
    removed. Determine the spectral resolution at each spaxal from 3
    sky lines around 2.2 microns. This should be the same resolution
    for all wavelengths > 2.2 microns based on some figures in the OSIRIS
    manual.
    """
    # Here are the 3 OH sky lines we will be using. Wavelengths were 
    # extractged from the ohlines.dat file included with IRAF:
    # /Applications/scisoft/all/Packages/iraf/iraf/noao/lib/linelists/
    lines = np.array([2.19556, 2.21255, 2.23127])

    # Old 2008 analysis
    #cubefile = datadir + '../noss/' + cuberoot + '.fits'
    # New 2010 analysis
    cubefile = datadir2010 + '../100829/SPEC/reduce/sky/sky_900s_cube_nodrk_tlc.fits'

    # Read in the cube... assume it is K-band
    cube, cubehdr = pyfits.getdata(cubefile, header=True)

    # Get the wavelength solution so we can specify range:
    w0 = cubehdr['CRVAL1']
    dw = cubehdr['CDELT1']
    wavelength = w0 + dw * np.arange(cube.shape[2], dtype=float)
    wavelength /= 1000.0   # Convert to microns

    # Gaussian line fits
    halfWindow = dw * 7.0 / 1000.0

    def gaussian(p, x):
        amp, sigma, center, constant = p
        y = amp * np.exp(-(x-center)**2 / (2*sigma**2))
        y += constant
        return y
        
    def fitfunc(p, wavelength, flux, line):
        #p[2] = line
        model = gaussian(p, wavelength)
        residuals = flux - model
        return residuals

    # Recall that the images are transposed relative to the cubes.
    resolutionMap = np.zeros((cube.shape[1], cube.shape[0]), dtype=float)
    ampAll = np.zeros((cube.shape[1], cube.shape[0], 3), dtype=float)
    resAll = np.zeros((cube.shape[1], cube.shape[0], 3), dtype=float)
    constAll = np.zeros((cube.shape[1], cube.shape[0], 3), dtype=float)
    centAll = np.zeros((cube.shape[1], cube.shape[0], 3), dtype=float)

    # Loop through spatial pixels and fit lines
    for xx in range(cube.shape[0]):
        for yy in range(cube.shape[1]):
            lineCount = 0.0

            for ii in range(len(lines)):
                # Get spectrum right around each line
                idx = np.where((wavelength > lines[ii]-halfWindow) &
                               (wavelength < lines[ii]+halfWindow))[0]
                spec = cube[xx, yy, idx]
                wave = wavelength[idx]

                pinit = np.zeros(4, dtype=float)
                pinit[0] = spec.max() - spec.min()
                pinit[1] = dw * 1.25 / 1000.0
                pinit[2] = wave[spec.argmax()]
                pinit[3] = spec.min()

                out = scipy.optimize.leastsq(fitfunc, pinit, 
                                             args=(wave, spec, lines[ii]))
                amp, sigma, center, constant = out[0]

                # if xx == 2 and yy == 3:
                #     pdb.set_trace()

                if (sigma > 0.0008):
                    print('Invalid for xx = %2d, yy = %2d, niter = %d' % \
                        (xx, yy, out[1]))
                    print('    ', out[0])
                    continue
                
                lineCount += 1.0
                resolutionMap[yy, xx] += lines[ii] / (2.35 * sigma)

                ampAll[yy, xx, ii] = amp
                resAll[yy, xx, ii] = lines[ii] / (2.35 * sigma)
                centAll[yy, xx, ii] = center
                constAll[yy, xx, ii] = constant

            resolutionMap[yy, xx] /= lineCount

    idx1 = np.where(centAll[:,:,0] != 0)
    idx2 = np.where(centAll[:,:,1] != 0)
    idx3 = np.where(centAll[:,:,2] != 0)

    print('Line Information: ')
    print('      Center:  %8.5f  %8.5f  %8.5f' % \
        (centAll[idx1[0],idx1[1],0].mean(), 
         centAll[idx2[0],idx2[1],1].mean(), 
         centAll[idx3[0],idx3[1],2].mean()))
    print('            :  %8.5f  %8.5f  %8.5f' % \
        (centAll[idx1[0],idx1[1],0].std(), 
         centAll[idx2[0],idx2[1],1].std(), 
         centAll[idx3[0],idx3[1],2].std()))
    print('  Amplitudes:  %8.2e  %8.2e  %8.2e' % \
        (ampAll[idx1[0],idx1[1],0].mean(), 
         ampAll[idx2[0],idx2[1],1].mean(), 
         ampAll[idx3[0],idx3[1],2].mean()))
    print('            :  %8.2e  %8.2e  %8.2e' % \
        (ampAll[idx1[0],idx1[1],0].std(), 
         ampAll[idx2[0],idx2[1],1].std(), 
         ampAll[idx3[0],idx3[1],2].std()))
    print('  Resolution:  %8d  %8d  %8d' % \
        (resAll[idx1[0],idx1[1],0].mean(), 
         resAll[idx2[0],idx2[1],1].mean(), 
         resAll[idx3[0],idx3[1],2].mean()))
    print('            :  %8d  %8d  %8d' % \
        (resAll[idx1[0],idx1[1],0].std(), 
         resAll[idx2[0],idx2[1],1].std(), 
         resAll[idx3[0],idx3[1],2].std()))
    print('    Constant:  %8.2e  %8.2e  %8.2e' % \
        (constAll[idx1[0],idx1[1],0].mean(), 
         constAll[idx2[0],idx2[1],1].mean(), 
         constAll[idx3[0],idx3[1],2].mean()))
    print('            :  %8.2e  %8.2e  %8.2e' % \
        (constAll[idx1[0],idx1[1],0].std(), 
         constAll[idx2[0],idx2[1],1].std(), 
         constAll[idx3[0],idx3[1],2].std()))

    py.clf()
    py.subplot(1, 2, 1)
    py.imshow(cube.sum(axis=2).transpose())
    py.title('Flux')
            
    py.subplot(1, 2, 2)
    py.imshow(resolutionMap)
    py.colorbar()
    py.title('Resolution')

    py.savefig(workdir + 'plots/map_resolution.png')

    # Save the resolution map to a file
    mapfile = workdir + 'maps/resolution_map.fits'
    ir.imdelete(mapfile)
    pyfits.writeto(mapfile, resolutionMap, header=cubehdr)


def plot_resolution_map():
    mapfile = workdir + 'maps/resolution_map.fits'

    res_map = pyfits.getdata(mapfile)

    py.clf()
    py.subplots_adjust(left=0.05, right=0.9)
    py.subplot(1, 2, 1)


##################################################
#
# Analysis of the 2010 data.
#
##################################################
def make_cube_images_2010():
    cubes = get_all_cubes(2010)

    for cc in cubes:
        make_cube_image(cc, rootdir=datadir2010)


def compare_with_nirc2(cube_ii, xshift, yshift,
                       datadir=datadir2010, plotdir=datadir2010+'plots/', year=2010):
    """
    Offsets are in arcseconds.

    +x shifts move the NIRC2 image left w.r.t. the OSIRIS image
    +y shifts move the NIRC2 image down w.r.t. the OSIRIS image 
    """
    cubes = get_all_cubes(year)
    cubefile = cubes[cube_ii].replace('.fits', '_img.fits')

    # Read in the OSIRIS image
    cube, cubehdr = pyfits.getdata(datadir + cubefile, header=True)

    # Read in the NIRC2 image (scale = 10 mas/pixel)
    nirc2file = '/u/jlu/data/m31/05jul/combo/m31_05jul_kp.fits'
    img, imghdr = pyfits.getdata(nirc2file, header=True)

    # Get the PA of the OSIRIS spectrograph image
    specPA = cubehdr['PA_SPEC']
    
    # Get the PA of the NIRC2 image
    imagPA = imghdr['ROTPOSN'] - imghdr['INSTANGL']

    # Rotate the NIRC2 image
    angle = specPA - imagPA
    img_rot = scipy.misc.imrotate(img, angle, interp='bicubic')

    # Get the center of the image. We will line up the centers of
    # the NIRC2 and OSIRIS image and apply shifts from there.
    imag_center = np.floor(np.array(img.shape) / 2.0)
    spec_size = np.array(cube.shape, dtype=float)
    spec_center = np.floor(np.array(cube.shape) / 2.0)

    # Define the cut-out region for the NIRC2 image. Remember we haven't
    # rescaled yet so the scale is still 10 mas/pixel.
    imag_scale = 0.01
    spec_scale = 0.05
    spec2imag = spec_scale / imag_scale

    imag_new_size = spec_size * spec2imag

    xlo = np.int(np.round(imag_center[1] - (imag_new_size[1] / 2.0) - (xshift / imag_scale)))
    xhi = np.int(xlo + imag_new_size[1])
    ylo = np.int(np.round(imag_center[0] - (imag_new_size[0] / 2.0) + (yshift / imag_scale)))
    yhi = np.int(ylo + imag_new_size[0])
    print((yhi-ylo)/spec2imag, (xhi-xlo)/spec2imag, cube.shape)
    
    # Resize the NIRC2 image to get rid of a useless regions.
    # This is also where we apply the shifts
    img_trim = img_rot[ylo:yhi, xlo:xhi]

    # Rebin to 50 mas pixels (assumes NIRC2 = 10 mas pixels).
    # This isn't perfect, but it is close enough.
    newSize = np.array(img_trim.shape)/5
    img_bin = rebin(img_trim, newSize)

    # Take the difference between the two images after normalizing them to have the
    # same total flux.
    #flux_scale = cube.flatten().sum() / img_bin.flatten().sum()
    flux_scale = cube.max() / img_bin.max()
    img_scale = img_bin * flux_scale
    diff = cube - img_scale

    #py.close('all')
    py.figure(2, figsize=(10,12))
    py.clf()
    py.subplots_adjust(left=0.05, bottom=0.05, top=0.95, right=0.95)

    py.subplot(1, 3, 1)
    py.imshow(img_scale)
    py.colorbar(orientation='horizontal', fraction=0.03)
    py.title("NIRC2")

    py.subplot(1, 3, 2)
    py.imshow(cube)
    py.colorbar(orientation='horizontal', fraction=0.03)
    py.title('OSIRIS')

    py.subplot(1, 3, 3)
    py.imshow(diff)
    py.colorbar(orientation='horizontal', fraction=0.03)
    py.title('Diff')
    py.savefig(plotdir + 'compare_with_nirc2_' + cubefile.replace('_img.fits', '.png'))

    py.figure(1)
    py.clf()
    py.hist(diff.flatten())
    py.xlabel('Flux Difference')


    diff_flat = diff.flatten()

    print('############################')
    print('')
    print('Differenced Image Properites')
    print('   STD = %.1f' % (diff_flat.std()))
    print('   AVG = %.1f' % (diff_flat.mean()))
    print('   MED = %.1f' % (np.median(diff_flat)))
    print('')
    print('%s  %5.1f  %5.1f' % (cubefile.replace('_img', ''), xshift/spec_scale, yshift/spec_scale))
    print('############################')

def recompare_with_nirc2_2010():
    shifts = atpy.Table(datadir2010 + 'shifts.txt', type='ascii')
    cubes = get_all_cubes(2010)
    
    for ii in range(len(shifts)):
        cube_name = shifts.col1[ii]
        idx = cubes.index(cube_name)
        print(idx)
        
        compare_with_nirc2(idx, shifts.col2[ii]*-0.05, shifts.col3[ii]*0.05)

def convert_pixel_shifts_for_mosaic(origShiftsFile, newShiftsFile, refFile):
    """
    Take a shifts file as created from the above compare_with_nirc2() function
    and convert the shifts into RA and Dec. offsets suitable for passing into the
    OSIRIS pipeline's mosaic module in a shifts file.
    """
    shifts = atpy.Table(origShiftsFile, type='ascii')
    shifts.rename_column('col1', 'name')
    shifts.rename_column('col2', 'X')
    shifts.rename_column('col3', 'Y')

    # First lets figure out the reference file and shift it to the top
    idx = np.where(shifts.name == refFile)[0]
    if len(idx) == 0:
        ref = 0
        print('Could not find specified reference file, using the first file in the list:')
    else:
        ref = idx[0]
        print('Using the following reference file:')
    print('    %s' % shifts.name[ref])

    # Modify shifts to be relative to the first frame.
    x0 = shifts.X[ref]
    y0 = shifts.Y[ref]

    # Modify the shifts
    shifts.X -= x0
    shifts.Y -= y0

    # Do a spot check
    ii = 2
    orig_dx = shifts.X[ii]
    orig_dy = shifts.Y[ii]

    # Convert to the proper direction
    shifts.X *= -1.0

    # More spot checking
    print('Check Coords for %s' % shifts.name[ii])
    print('Ref Coords (pixels):  %5.1f  %5.1f' % (x0, y0))
    print('Orig Shifts (pixels): %5.1f  %5.1f' % (orig_dx, orig_dy))
    print('New Shifts (pixels):  %5.1f  %5.1f' % (shifts.X[ii], shifts.Y[ii]))
    print('   Sum Check: %5.2f  %5.2f' % (math.hypot(shifts.X[ii], shifts.Y[ii]),
                                          math.hypot(orig_dx, orig_dy)))

    refImg, refHdr = pyfits.getdata(refFile.replace('.fits', '_img.fits'), header=True)
    testImg = pyfits.getdata(shifts.name[ii].replace('.fits', '_img.fits'))
    py.clf()
    py.subplot(1, 2, 1)
    py.imshow(refImg)
    py.plot(refImg.shape[1]/2.0, refImg.shape[0]/2.0, 'k+', ms=10, mew=2)

    py.subplot(1, 2, 2)
    py.imshow(testImg)
    py.plot(testImg.shape[1]/2.0, testImg.shape[0]/2.0, 'k+', ms=10, mew=2)

    # Shift the reference image to the top
    indices = np.arange(len(shifts))
    indices = np.delete(indices, ref)
    indices = np.concatenate([[ref], indices])

    # Save to a new file... re-order to put the ref file first.
    _out = open(newShiftsFile, 'w')
    for ii in indices:
        _out.write('%-20s  %6.3f  %6.3f\n' % (shifts.name[ii], shifts.X[ii], shifts.Y[ii]))
    _out.close()

    # Blah Testing
    #indices = indices[0:31]

    # Note that OSIRIS Mosaic Cubes module expects a FITS image that has
    # dimensions of 2 x N data sets. The ordering is backwards as well, Y is first
    # and X is second and the array needs to be transposed. This was all figured
    # out by trial and error.
    shiftsArr = np.array([shifts.Y[indices], shifts.X[indices]])
    shiftsArr = shiftsArr.transpose()
    pyfits.writeto(newShiftsFile.replace('.txt', '.fits'), shiftsArr, clobber=True)


def osiris_performance_2010():
    cubes = get_all_cubes(2010)

    for cc in cubes:
        osiris_performance(cc, rootdir=datadir2010, plotdir=datadir2010 + '/plots/')


def imshow_cube_image_2010(image, header=None, compass=True, blackhole=True, 
                           cmap=None):
    """
    Call imshow() to plot a cube image. Make sure it is already 
    masked. Also pass in the header to do the compass rose calculations.
    """
    # Setup axis info (x is the 2nd axis)
    xcube = (np.arange(image.shape[1]) - m31pos[0]) * 0.05
    ycube = (np.arange(image.shape[0]) - m31pos[1]) * 0.05
    xtickLoc = py.MultipleLocator(0.4)

    if cmap is None:
        cmap = py.cm.jet

    # Plot the image.
    py.imshow(image, 
              extent=[xcube[0], xcube[-1], ycube[0], ycube[-1]],
              cmap=cmap)
    py.gca().get_xaxis().set_major_locator(xtickLoc)
    
    py.xlabel('X (arcsec)')
    py.ylabel('Y (arcsec)')

    # Get the spectrograph position angle for compass rose
    if compass is True:
        if header is None:
            pa = paSpec
        else:
            pa = header['PA_SPEC']


        # Make a compass rose
        cosSin = np.array([ math.cos(math.radians(pa)), 
                            math.sin(math.radians(pa)) ])
        arr_base = np.array([ xcube[-1]-0.6, ycube[-1]-0.8 ])
        arr_n = cosSin * 0.2
        arr_w = cosSin[::-1] * 0.2
        py.arrow(arr_base[0], arr_base[1], arr_n[0], arr_n[1],
                 edgecolor='w', facecolor='w', width=0.03, head_width=0.08)
        py.arrow(arr_base[0], arr_base[1], -arr_w[0], arr_w[1],
                 edgecolor='w', facecolor='w', width=0.03, head_width=0.08)
        py.text(arr_base[0]+arr_n[0]+0.1, arr_base[1]+arr_n[1]+0.1, 'N', 
                color='white', 
                horizontalalignment='left', verticalalignment='bottom')
        py.text(arr_base[0]-arr_w[0]-0.15, arr_base[1]+arr_w[1]+0.1, 'E', 
                color='white',
                horizontalalignment='right', verticalalignment='center')

    if blackhole is True:
        py.plot([0], [0], 'ko')


    py.xlim([-1.0, 1.0])
    py.ylim([-1.7, 1.8])


def plot_map_snr_2010():
    """
    Use after calling ifu.map_snr(). Plots the cube image on the left panel
    and the SNR map on the right panel.
    """
    # Read in the M31 flux and SNR maps
    cubeimg, hdr = pyfits.getdata(workdir + cuberoot + '_img.fits', header=True)
    snrimg = pyfits.getdata(workdir + cuberoot + '_snr.fits')

    # Mask out bad pixels (where SNR couldn't be calculated)
    snrMasked = np.ma.masked_invalid(snrimg)
    cubeMasked = np.ma.array(cubeimg, mask=snrMasked.mask)

    # Setup axis info (x is the 2nd axis)
    xcube = (np.arange(cubeimg.shape[1]) - m31pos[0]) * 0.05
    ycube = (np.arange(cubeimg.shape[0]) - m31pos[1]) * 0.05
    xtickLoc = py.MultipleLocator(0.4)

    # Get the spectrograph position angle for compass rose
    pa = hdr['PA_SPEC']

    py.close(2)
    py.figure(2, figsize=(9,6))
    py.clf()
    py.subplots_adjust(left=0.08, right=0.95, top=0.95, wspace=0)

    ##########
    # Plot flux
    ##########
    py.subplot(1, 2, 1)
    imshow_cube_image_2010(cubeMasked, header=hdr, cmap=py.cm.hot,
                           compass=True, blackhole=True)
    
    ##########
    # Plot SNR
    ##########
    py.subplot(1, 2, 2)
    imshow_cube_image_2010(snrMasked, cmap=py.cm.jet,
                           compass=False, blackhole=False)
    cbar = py.colorbar(orientation='vertical')
    cbar.set_label('Signal-to-Noise')

    py.savefig(workdir + 'plots/map_snr.png')
    py.show()



def calc_shifts_2010():
    """
    !!! BROKEN !!!  I tried to calculate automated shifts but it just didn't work.
    Instead, I had to calculate the shifts manually using the compare_with_nirc2
    routine above and judge the goodness of fit by eye. All the results were then
    pasted into the manual_shifts.txt file (and copied to shifts.txt). 
    """
    cubes = get_all_cubes(2010)

    # Read in the first cube... assume it is K-band and all are at the same PA
    cube0, cubehdr = pyfits.getdata(datadir2010 + cubes[0], header=True)
    cubeimg0 = pyfits.getdata(datadir2010 + cubes[0].replace('.fits', '_img.fits'))

    ### Register the NIRC2 image to the OSIRIS image.
    # Read in the NIRC2 image (scale = 10 mas/pixel)
    nirc2file = '/u/jlu/data/m31/05jul/combo/m31_05jul_kp.fits'
    img, imghdr = pyfits.getdata(nirc2file, header=True)

    # Get the PA of the OSIRIS spectrograph image
    specPA = cubehdr['PA_SPEC']
    
    # Get the PA of the NIRC2 image
    imagPA = imghdr['ROTPOSN'] - imghdr['INSTANGL']

    # Rotate the NIRC2 image
    angle = specPA - imagPA
    img_rot = scipy.misc.imrotate(img, angle, interp='bicubic')

    # Resize the NIRC2 image to get rid of a useless regions
    img_trim = img_rot[475:975, 475:975]

    # Rebin to 50 mas pixels (assumes NIRC2 = 10 mas pixels).
    # This isn't perfect, but it is close enough.
    newSize = np.array(img_trim.shape)/5
    img_bin = rebin(img_trim, newSize)
    py.clf()
    py.imshow(img_bin)

    # Write out the NIRC2 fits file we are going to use
    # in xregister. Modify the image header first.
    # pyfits.writeto(datadir2010 + 'nirc2_ref.fits', img_bin, imghdr,
    #                clobber=True)

    # ir.images()
    # ir.immatch()
    # ir.unlearn('xregister')
    # ir.xregister.coords = ''
    # ir.xregister.output = ''
    # ir.xregister.append = 'no'
    # ir.xregister.databasefmt = 'no'
    # ir.xregister.verbose = 'no'
    # ir.xregister.apodize = '0.2'
    # ir.xregister.correlation = 'difference'

    # # Make lists for xregister
    # xreg_in_file = datadir2010 + 'xreg_input.lis' 
    # _input = open(xreg_in_file, 'w')
    # for cc in cubes:
    #     cube_img_file = datadir2010 + cc.replace('.fits', '_img.fits')
    #     _input.write('%s\n' % cube_img_file)
    # _input.close()

    # inFile = '@' + xreg_in_file
    # #refFile = datadir2010 + 'nirc2_ref.fits'
    # idx = cubes.index('s100829_a010001_tlc_Kbb_050.fits')
    # refFile = datadir2010 + cubes[idx].replace('.fits', '_img.fits')
    # print refFile
    # regions = '[*,*]'
    # shiftFile = datadir2010 + 'xreg_shifts.txt'

    # ir.delete(shiftFile)
    # ir.xregister(inFile, refFile, regions, shiftFile)


def rebin(a, newshape):
    """
    rebin ndarray data into a smaller ndarray of the same rank whose dimensions
    are factors of the original dimensions. eg. An array with 6 columns and 4 rows
    can be reduced to have 6,3,2 or 1 columns and 4,2 or 1 rows.
    example usages:
    >>> a=rand(6,4); b=rebin(a,3,2)
    >>> a=rand(6); b=rebin(a,2)
    """
    shape = a.shape
    lenShape = len(shape)
    factor = np.array(shape) / np.array(newshape)

    evList = ['a.reshape('] + \
        ['newshape[%d],factor[%d],'%(i,i) for i in range(lenShape)] + \
        [')'] + ['.sum(%d)'%(i+1) for i in range(lenShape)]# + \
        #['/factor[%d]'%i for i in range(lenShape)]
    print(''.join(evList))

    return eval(''.join(evList))
