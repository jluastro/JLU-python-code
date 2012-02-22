import numpy as np
import pylab as py
import pidly
import pyfits, pickle
import time
import math, glob
import scipy
import scipy.interpolate
from gcwork import objects
import pdb

# datadir = '/u/jlu/data/m31/08oct/081021/SPEC/reduce/m31/ss/'
# workdir = '/u/jlu/work/m31/nucleus/ifu_09_02_24/'
# cuberoot = 'm31_08oct_Kbb_050'

workdir = '/u/jlu/work/m31/nucleus/ifu_11_11_30/'
datadir = workdir
cuberoot = 'm31_Kbb_050'

# Start an IDL session
idl = pidly.IDL(long_delay=0.1)

cc = objects.Constants()

def run():
    """
    Run the PPXF analysis the M31 OSIRIS data cube.
    """
    # Read in the data cube.
    cubefits = pyfits.open(datadir + cuberoot + '.fits')
    
    cube = cubefits[0].data
    hdr = cubefits[0].header
    errors = cubefits[1].data
    quality = cubefits[2].data
    nframes = cubefits[3].data

    # Convert the errors to the error on the mean vs. the stddev
    # nframes is wrong (too low by 1)... Why?
    # This produces errors that are still a little large compared to what
    # I get if I look at the noise in a line-less region of spectra.
    # Different is nearly a factor of 2 or so.
    errors /= (np.sqrt(nframes+1) * 2.0)

    # Get the wavelength solution so we can specify range:
    w0 = hdr['CRVAL1']
    dw = hdr['CDELT1']
    wavelength = w0 + dw * np.arange(cube.shape[2], dtype=float)
    wavelength /= 1000.0   # Convert to microns
    
    # Test #2
    logWaveCube, newCube, velScale = log_rebin_cube(wavelength, cube)

    # Load templates
    logWaveTemps, templates = load_templates(velScale)

    # Trim down galaxy spectrum to 2.18 microns and greater.
    # This is necessary for the ppxf routine which demands the templates
    # have broader wavelength coverage than the galaxy spectrum.
    waveCut = 2.185
    print 'blue wavelength cutoff = %.2f microns' % waveCut
    idx = np.where(np.exp(logWaveCube) > waveCut)[0]
    newCube = newCube[:,:,idx]
    logWaveCube = logWaveCube[idx]

    print 'logWaveCube.shape = ', logWaveCube.shape
    print 'logWaveTemps.shape = ', logWaveTemps.shape
    #print 'logWaveCube = ', logWaveCube
    #print 'logWaveTemps = ', logWaveTemps

    py.clf()
    py.plot(np.log(wavelength), cube[10,30,:], 'r-')
    py.plot(logWaveCube, newCube[10,30,:], 'b-')
    py.plot(logWaveTemps, templates[0,:], 'g-')
    py.show()

    # Mark good pixels as those with wavelength > 2.18.
    # Below this, atmosphere (and templates) get icky.
    # We have already made the waveCut, so all pixels in
    # the galaxy spectrum are good.
    #goodPixels = np.where(np.exp(logWaveCube) > waveCut)[0]
    #print 'goodPixels size = ', goodPixels.shape
    #goodPixels = np.array(goodPixels, dtype=np.int32)

    # Run ppxf
    print 'IDL PPXF:'
    print '  setting templates'
    idl.templates = templates
    print '  setting velocity scale = ', velScale
    idl.velScale = velScale
    print '  setting start velocity'
    idl.start = np.array([-300.0, 180.0])
    print '  setting good pixels'
    #idl.goodPixels = goodPixels
    idl('goodPixels = indgen(%d)' % (len(logWaveCube)))
    print '  setting vsyst'
    idl.dv = (logWaveTemps[0] - logWaveCube[0]) * cc.c
    idl('loadct, 12')

    # Do the whole cube
    imgShape = (newCube.shape[0], newCube.shape[1])
    velocity = np.zeros(imgShape, dtype=float)
    sigma = np.zeros(imgShape, dtype=float)
    h3 = np.zeros(imgShape, dtype=float)
    h4 = np.zeros(imgShape, dtype=float)
    h5 = np.zeros(imgShape, dtype=float)
    h6 = np.zeros(imgShape, dtype=float)
    chi2red = np.zeros(imgShape, dtype=float)

    pweights = np.zeros((newCube.shape[0], newCube.shape[1], 5), dtype=float)
    tweights = np.zeros((newCube.shape[0], newCube.shape[1], templates.shape[0]), dtype=float)


    for xx in range(4, newCube.shape[0]-4):
        for yy in range(10, newCube.shape[1]-10):
            print 'STARTING ppxf  ', time.ctime(time.time())

            print ''
            print '  PIXEL: xx = %d, yy = %d' % (xx, yy)
            print '    setting galaxy spectra'
            tmp = newCube[xx,yy,:]
            tmperr = np.zeros(len(tmp), dtype=float) + \
                np.median(errors[xx,yy,:])

            # Add a constant to the galaxy spectrum to make it above zero
            minFlux = tmp.mean() - (3.0 * tmp.std())
            tmp -= minFlux
            
            idl.galaxy = tmp
            print '    setting errors'
            idl.error = tmperr
            print '    calling ppxf'

            idl('ppxf, templates, galaxy, error, velScale, start, sol, GOODPIXELS=goodPixels, /plot, moments=4, degree=4, vsyst=dv, polyweights=polyweights, weights=weights')

            solution = idl.sol
            velocity[xx, yy] = solution[0]
            sigma[xx, yy] = solution[1]
            h3[xx, yy] = solution[2]
            h4[xx, yy] = solution[3]
            h5[xx, yy] = solution[4]
            h6[xx, yy] = solution[5]
            chi2red[xx, yy] = solution[6]

            pweights[xx, yy, :] = idl.polyweights
            tweights[xx, yy, :] = idl.weights

            print '  SOLUTION:'
            print '    velocity = %6.1f km/s' % velocity[xx, yy]
            print '    sigma    = %5.1f km/s' % sigma[xx, yy]
            print '    h3       = %6.3f' % h3[xx, yy]
            print '    h4       = %6.3f' % h4[xx, yy]
            print '    h5       = %6.3f' % h5[xx, yy]
            print '    h6       = %6.3f' % h6[xx, yy]
            print ''
            print '    Reduced Chi^2 = %6.3f' % chi2red[xx, yy]

    output = open(workdir + '/ppxf.dat', 'w')
    pickle.dump(velocity, output)
    pickle.dump(sigma, output)
    pickle.dump(h3, output)
    pickle.dump(h4, output)
    pickle.dump(h5, output)
    pickle.dump(h6, output)
    pickle.dump(chi2red, output)
    pickle.dump(pweights, output)
    pickle.dump(tweights, output)
    output.close()
    

def runErrorMC(suffix='_1'):
    """
    Run the PPXF analysis the M31 OSIRIS data cube.
    """
    # Read in the data cube.
    cubefits = pyfits.open(datadir + cuberoot + '.fits')
    
    cube = cubefits[0].data
    hdr = cubefits[0].header
    errors = cubefits[1].data
    quality = cubefits[2].data
    nframes = cubefits[3].data

    # Convert the errors to the error on the mean vs. the stddev
    # nframes is wrong (too low by 1)... Why?
    # This produces errors that are still a little large compared to what
    # I get if I look at the noise in a line-less region of spectra.
    # Different is nearly a factor of 2 or so.
    errors /= (np.sqrt(nframes+1) * 2.0)

    # Get the wavelength solution so we can specify range:
    w0 = hdr['CRVAL1']
    dw = hdr['CDELT1']
    wavelength = w0 + dw * np.arange(cube.shape[2], dtype=float)
    wavelength /= 1000.0   # Convert to microns
    
    # Test #2
    logWaveCube, newCube, velScale = log_rebin_cube(wavelength, cube)

    # Load templates
    logWaveTemps, templates = load_templates(velScale)

    # Trim down galaxy spectrum to 2.1 microns and greater.
    # This is necessary for the ppxf routine which demands the templates
    # have broader wavelength coverage than the galaxy spectrum.
    idx = np.where(np.exp(logWaveCube) > 2.1)[0]
    newCube = newCube[:,:,idx]
    logWaveCube = logWaveCube[idx]

    print 'logWaveCube.shape = ', logWaveCube.shape
    print 'logWaveTemps.shape = ', logWaveTemps.shape
    print 'logWaveCube = ', logWaveCube
    print 'logWaveTemps = ', logWaveTemps

    #py.clf()
    #py.plot(np.log(wavelength), cube[10,30,:], 'r-')
    #py.plot(logWaveCube, newCube[10,30,:], 'b-')
    #py.plot(logWaveTemps, templates[0,:], 'g-')
    #py.show()

    # Mark good pixels as those with wavelength > 2.1.
    # Below this, atmosphere (and templates) get icky
    goodPixels = np.where(np.exp(logWaveCube) > 2.1)[0]
    print 'goodPixels = ', goodPixels

    # Run ppxf
    print 'IDL PPXF:'
    print '  setting templates'
    idl.templates = templates
    print '  setting velocity scale = ', velScale
    idl.velScale = velScale
    print '  setting start velocity'
    idl.start = np.array([-300.0, 180.0])
    print '  setting good pixels'
    idl.goodPixels = goodPixels
    print '  setting vsyst'
    idl.dv = (logWaveTemps[0] - logWaveCube[0]) * cc.c
    idl('loadct, 12')

    # Error distributions
    nsim = 10
    imgShape = (newCube.shape[0], newCube.shape[1], nsim)
    velocityErr = np.zeros(imgShape, dtype=float)
    sigmaErr = np.zeros(imgShape, dtype=float)
    h3Err = np.zeros(imgShape, dtype=float)
    h4Err = np.zeros(imgShape, dtype=float)
    h5Err = np.zeros(imgShape, dtype=float)
    h6Err = np.zeros(imgShape, dtype=float)
    chi2redErr = np.zeros(imgShape, dtype=float)

    pweightsErr = np.zeros((newCube.shape[0], newCube.shape[1], 5, nsim), 
                           dtype=float)
    tweightsErr = np.zeros((newCube.shape[0], newCube.shape[1], 
                            templates.shape[0], nsim), dtype=float)

    for xx in range(4, 16):
    #for xx in range(4, newCube.shape[0]-4):
        print 'STARTING ppxf for xx = %d ' % xx, time.ctime(time.time())
        #for yy in range(10, newCube.shape[1]-10):
        for yy in range(26, 61):
            print ''
            print '  PIXEL: xx = %d, yy = %d' % (xx, yy)

            tmp = newCube[xx,yy,:]
            tmperr = np.zeros(len(tmp), dtype=float) + \
                np.median(errors[xx,yy,:])

            # Add a constant to the galaxy spectrum to make it above zero
            minFlux = tmp.mean() - (3.0 * tmp.std())
            tmp -= minFlux

            
            for nn in range(nsim):
                specSim = tmp + (np.random.randn(len(tmp)) * tmperr)
                idl.galaxy = specSim
                idl.error = tmperr

                idl('ppxf, templates, galaxy, error, velScale, start, sol, GOODPIXELS=goodPixels, /plot, moments=4, degree=4, vsyst=dv, polyweights=polyweights, weights=weights')

                solution = idl.sol
                velocityErr[xx, yy, nn] = solution[0]
                sigmaErr[xx, yy, nn] = solution[1]
                h3Err[xx, yy, nn] = solution[2]
                h4Err[xx, yy, nn] = solution[3]
                h5Err[xx, yy, nn] = solution[4]
                h6Err[xx, yy, nn] = solution[5]
                chi2redErr[xx, yy, nn] = solution[6]

                pweightsErr[xx, yy, :, nn] = idl.polyweights
                tweightsErr[xx, yy, :, nn] = idl.weights

    output = open(workdir + '/ppxf_errors'+suffix+'.dat', 'w')
    pickle.dump(velocityErr, output)
    pickle.dump(sigmaErr, output)
    pickle.dump(h3Err, output)
    pickle.dump(h4Err, output)
    pickle.dump(h5Err, output)
    pickle.dump(h6Err, output)
    pickle.dump(chi2redErr, output)
    pickle.dump(pweightsErr, output)
    pickle.dump(tweightsErr, output)
    output.close()
            
class PPXFresults(object):
    def __init__(self, inputFile=workdir+'ppxf.dat'):
        self.inputFile = inputFile
        
        input = open(inputFile, 'r')
        self.velocity = pickle.load(input)
        self.sigma = pickle.load(input)
        self.h3 = pickle.load(input)
        self.h4 = pickle.load(input)
        self.h5 = pickle.load(input)
        self.h6 = pickle.load(input)
        self.chi2red = pickle.load(input)
        self.pweights = pickle.load(input)
        self.tweights = pickle.load(input)


# Modified black hole position from 2009 sep alignment 
# analysis between NIRC2 and HST. This is the position
# in the osiris cubes.
#bhpos = np.array([8.7, 39.1]) * 0.05 # python coords, not ds9
bhpos = np.array([22.5, 37.5]) * 0.05 # guessed offset for new M31 mosaic

def plotResults():
    cubeimg = pyfits.getdata(datadir + cuberoot + '_img.fits')
    snrimg = pyfits.getdata(datadir + cuberoot + '_snr.fits')

    p = PPXFresults()

    print p.velocity.shape
    print cubeimg.shape
    print bhpos
    xaxis = np.arange(p.velocity.shape[0]) * 0.05
    yaxis = np.arange(p.velocity.shape[1]) * 0.05
    
    xtickLoc = py.MultipleLocator(0.5)

    py.figure(2, figsize=(12,5))
    py.subplots_adjust(left=0.01, right=0.94, top=0.95)
    py.clf()

    #bhpos = np.array([9.0, 38.0]) * 0.05
    ##########
    # Plot Cube Image
    ##########
    py.subplot(1, 4, 1)
    py.imshow(py.ma.masked_where(cubeimg<5, cubeimg), 
              extent=[xaxis[0], xaxis[-1], yaxis[0], yaxis[-1]],
              cmap=py.cm.hot)
    py.plot([bhpos[0]], [bhpos[1]], 'kx')
    py.ylabel('Y (arcsec)')
    py.xlabel('X (arcsec)')
    py.gca().get_xaxis().set_major_locator(xtickLoc)
    cbar = py.colorbar(orientation='vertical')
    cbar.set_label('Flux (cts/sec)')

    # Make a compass rose
    pa = 56.0
    cosSin = np.array([ math.cos(math.radians(pa)), 
                        math.sin(math.radians(pa)) ])
    arr_base = np.array([ xaxis[-1]-0.5, yaxis[-1]-0.6 ])
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
    py.title('K Image')


    ##########
    # Plot SNR Image
    ##########
    print snrimg[30,10]
    py.subplot(1, 4, 2)
    py.imshow(py.ma.masked_invalid(snrimg), 
              extent=[xaxis[0], xaxis[-1], yaxis[0], yaxis[-1]],
              cmap=py.cm.jet)
    py.plot([bhpos[0]], [bhpos[1]], 'kx')
    py.xlabel('X (arcsec)')
    py.gca().get_xaxis().set_major_locator(xtickLoc)
    cbar = py.colorbar(orientation='vertical')
    cbar.set_label('SNR')
    py.title('SNR')

    ##########
    # Plot Velocity
    ##########
    py.subplot(1, 4, 3)
    velimg = p.velocity.transpose()+308.0
    py.imshow(py.ma.masked_where(velimg>250, velimg), 
              extent=[xaxis[0], xaxis[-1], yaxis[0], yaxis[-1]])
    py.plot([bhpos[0]], [bhpos[1]], 'kx')
    py.xlabel('X (arcsec)')
    py.gca().get_xaxis().set_major_locator(xtickLoc)
    cbar = py.colorbar(orientation='vertical')
    cbar.set_label('Velocity (km/s)')
    py.title('Velocity')

    ##########
    # Plot Dispersion
    ##########
    py.subplot(1, 4, 4)
    sigimg = p.sigma.transpose()
    py.imshow(py.ma.masked_where(sigimg<=0, sigimg), 
              extent=[xaxis[0], xaxis[-1], yaxis[0], yaxis[-1]],
              cmap=py.cm.jet)
    py.plot([bhpos[0]], [bhpos[1]], 'kx')
    py.xlabel('X (arcsec)')
    py.gca().get_xaxis().set_major_locator(xtickLoc)
    cbar = py.colorbar(orientation='vertical')
    cbar.set_label('Dispersion (km/s)')
    py.title('Dispersion')

#     for xx in range(4, velocity.shape[0]-5):
#         for yy in range(10, velocity.shape[1]-11):
#             print 'xx=%2d, yy=%2d, velocity: mean=%6.1f  std = %6.1f' % \
#                 (xx, yy, 
#                  p.velocity[xx:xx+2,yy:yy+2].mean(), 
#                  p.velocity[xx:xx+2,yy:yy+2].std())

    py.savefig(workdir + 'plots/kinematic_maps.png')
    py.savefig(workdir + 'plots/kinematic_maps.eps')
    py.show()



def plotResults2():
    cubeimg = pyfits.getdata(datadir + cuberoot + '_img.fits')
    snrimg = pyfits.getdata(datadir + cuberoot + '_snr.fits')

    p = PPXFresults()

    xaxis = np.arange(p.velocity.shape[0]) * 0.05
    yaxis = np.arange(p.velocity.shape[1]) * 0.05
    
    xtickLoc = py.MultipleLocator(0.5)

    py.close(2)
    py.figure(2, figsize=(22,8))
    py.subplots_adjust(left=0.05, right=0.94, top=0.95)
    py.clf()

    #bhpos = np.array([9.0, 38.0]) * 0.05
    ##########
    # Plot Cube Image
    ##########
    py.subplot(1, 3, 1)
    py.imshow(py.ma.masked_where(cubeimg<-10000, cubeimg), 
              extent=[xaxis[0], xaxis[-1], yaxis[0], yaxis[-1]],
              cmap=py.cm.hot)
    py.plot([bhpos[0]], [bhpos[1]], 'kx', markeredgewidth=3)
    py.ylabel('Y (arcsec)')
    py.xlabel('X (arcsec)')
    py.gca().get_xaxis().set_major_locator(xtickLoc)
    cbar = py.colorbar(orientation='vertical')
    cbar.set_label('Flux (cts/sec)')

    # Make a compass rose
    pa = 56.0
    cosSin = np.array([ math.cos(math.radians(pa)), 
                        math.sin(math.radians(pa)) ])
    arr_base = np.array([ xaxis[-1]-0.5, yaxis[-1]-0.6 ])
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

    ##########
    # Plot Velocity
    ##########
    py.subplot(1, 3, 2)
    velimg = p.velocity.transpose()+308.0
    py.imshow(py.ma.masked_where(velimg>250, velimg), 
              extent=[xaxis[0], xaxis[-1], yaxis[0], yaxis[-1]])
    py.plot([bhpos[0]], [bhpos[1]], 'kx', markeredgewidth=3)
    py.xlabel('X (arcsec)')
    py.gca().get_xaxis().set_major_locator(xtickLoc)
    cbar = py.colorbar(orientation='vertical')
    cbar.set_label('Velocity (km/s)')

    ##########
    # Plot Dispersion
    ##########
    py.subplot(1, 3, 3)
    sigimg = p.sigma.transpose()
    py.imshow(py.ma.masked_where(sigimg<=0, sigimg), 
              extent=[xaxis[0], xaxis[-1], yaxis[0], yaxis[-1]],
              cmap=py.cm.jet)
    py.plot([bhpos[0]], [bhpos[1]], 'kx', markeredgewidth=3)
    py.xlabel('X (arcsec)')
    py.gca().get_xaxis().set_major_locator(xtickLoc)
    cbar = py.colorbar(orientation='vertical')
    cbar.set_label('Dispersion (km/s)')

#     for xx in range(4, velocity.shape[0]-5):
#         for yy in range(10, velocity.shape[1]-11):
#             print 'xx=%2d, yy=%2d, velocity: mean=%6.1f  std = %6.1f' % \
#                 (xx, yy, 
#                  p.velocity[xx:xx+2,yy:yy+2].mean(), 
#                  p.velocity[xx:xx+2,yy:yy+2].std())

    py.savefig(workdir + 'plots/kinematic_maps2.png')
    py.savefig(workdir + 'plots/kinematic_maps2.eps')
    py.show()



def precessionSpeed():
    cubeimg = pyfits.getdata(datadir + cuberoot + '_img.fits')
    snrimg = pyfits.getdata(datadir + cuberoot + '_snr.fits')

    p = PPXFresults()

    #bhpos = np.array([9.0, 38.0])
    xaxis = (np.arange(p.velocity.shape[0]) - bhpos[0]) * 0.05
    yaxis = (np.arange(p.velocity.shape[1]) - bhpos[1]) * 0.05
    
    # Pull out a profile along the line-of-nodes (Y-axis)
    binSize = 0.05 # * 3.77
    # Convert Surface Brightness from cts/s to mag/arcsec^2
    cubeimg += cubeimg.mean() - (cubeimg.std() * 3.0)
    fluxCube = cubeimg / binSize**2
    magCube = -2.5 * np.log10(cubeimg) + 23.7

    light = fluxCube[:,bhpos[0]]
    velocity = p.velocity[bhpos[0],:] + 308.0
    dispersion = p.sigma[bhpos[0],:]

    py.figure(1)
    py.clf()
    py.subplots_adjust(left=0.13)
    py.plot(yaxis, light)
    limits = py.axis()
    py.plot([0, 0], [limits[2], limits[3]], 'k--')
    py.xlabel('Distance Along Line-of-Nodes (")')
    py.ylabel('Surface Brightness (arbitrary flux/square arcsec)')
    py.xlim(-1.3, 1.3)
    py.savefig(workdir + 'plots/line_of_nodes_flux.png')
    #py.show()

    py.clf()
    py.subplots_adjust(left=0.13)
    py.plot(yaxis, velocity)
    py.xlabel('Distance Along Line-of-Nodes (")')
    py.ylabel('Velocity (km/s)')
    limits = py.axis()
    py.plot([0, 0], [limits[2], limits[3]], 'k--')
    py.plot([limits[0], limits[1]], [0, 0], 'k--')
    py.ylim(-300, 300)
    py.xlim(-1.3, 1.3)
    py.savefig(workdir + 'plots/line_of_nodes_velocity.png')
    #py.show()

    py.clf()
    py.subplots_adjust(left=0.13)
    py.plot(yaxis, dispersion)
    py.xlabel('Distance Along Line-of-Nodes (")')
    py.ylabel('Velocity Dispersion (km/s)')
    limits = py.axis()
    py.plot([0, 0], [limits[2], limits[3]], 'k--')
    py.plot([limits[0], limits[1]], [0, 0], 'k--')
    py.xlim(-1.3, 1.3)
    py.savefig(workdir + 'plots/line_of_nodes_dispersion.png')
    #py.show()

    # Calculate the Sigma_P * sin(i) (precession speed)
    numerator = light * velocity
    denominator = light * (yaxis * 3.77)

    # Sum it all up, but only over a fixed radial extent
    idx = np.where(np.abs(yaxis) < 1.2)[0]
    
    sigPsinI = numerator[idx].sum() / denominator[idx].sum()
    sigP = sigPsinI / math.sin(math.radians(55.0))
    print 'Precession Speed * sin(i) = %6.2f km/s/pc' % sigPsinI
    print 'Precession Speed (i=55 d) = %6.2f km/s/pc' % sigP
    
    # Calculate for several lines
    numAll = np.zeros(fluxCube.shape[0], dtype=float)
    denomAll = np.zeros(fluxCube.shape[0], dtype=float)
    sigPsinIall = np.zeros(fluxCube.shape[0], dtype=float)
    for xx in range(4, 17):
        lightTmp = fluxCube[idx,xx]
        velocityTmp = p.velocity[xx,idx] + 308.0
        numAll[xx] = (lightTmp * velocityTmp).sum() / lightTmp.sum()
        denomAll[xx] = (lightTmp * yaxis[idx] * 3.77).sum() / lightTmp.sum()
        sigPsinIall[xx] = numAll[xx] / denomAll[xx]
    print sigPsinIall
    foo = np.where(numAll != 0)[0]
    py.clf()
    py.plot(denomAll[foo]/3.77, numAll[foo], 'k.')
    py.show()

    # Calculate for a bunch of different velocity errors
    verrs = np.arange(3.0, 30.0, 1.0)

    denomSum = denominator[idx].sum()

    sigPErrs = np.zeros(len(verrs), dtype=float)

    for ii in range(len(verrs)):
        sigPsinIErr = np.sqrt((light[idx]**2).sum())
        sigPsinIErr *= verrs[ii] * binSize / denomSum
        sigPErr = sigPsinIErr / abs(math.sin(math.radians(55.0)))

        sigPErrs[ii] = sigPErr

#         print '   verr = %4.1f, error in precession speed = %6.2f km/s/pc' % \
#             (verrs[ii], sigPErr)

    py.clf()
    py.plot(verrs, sigPErrs)
    #py.show()


def load_templates(velScale, resolution=3241):
    templateDir = '/u/jlu/work/atlasSpectra/gemini/gnirs/'

    files = glob.glob(templateDir + '*.fits')
    print 'Using %d templates' % len(files)

    templates = None
    for ff in range(len(files)):
        newWave, newSpec = load_spectrum_gnirs(files[ff], velScale, resolution)

        if templates == None:
            templates = np.zeros((len(files), len(newSpec)), dtype=float)

        templates[ff,:] = newSpec

    return (newWave, templates)

def load_templates_old(velScale):
    templateDir = '/u/jlu/work/atlasSpectra/medresIR/K_band'

    files = glob.glob(templateDir + '/spec.*')
    specfiles = ['HR6406', 'KY_Cyg', 'HR8316', 'HR8726', 'HR8089',
                 'HR8465', 'HR2473', 'HR969', 'HR8313', 'HR8626', 
                 'HR8232', 'HR1017', 'HR7387', 'HR382', 'HR7924',
                 'HR1713', 'HR6714', 'HR7977', 'HR1903', 'HR8469',
                 'RX_Boo', 'HR7886', 'HR8621', 'HR7635',
                 'HR3275', 'HR7806', 'HR6703', 'HR3212', 'HR3323',
                 'HR4255', 'HR4883', 'HR8905', 'HR7495', 'HR5017',
                 'HR4031', 'HR1412', 'HR5291', 'HR1791', 'HR2294',
                 'HR6165', 'HR3982', 'HR4295', 'HR7557', 'HR4931',
                 'HR6927']
    files = []
    for ss in range(len(specfiles)):
        files.append( templateDir + '/spec.' + specfiles[ss] )

    print 'Using %d templates' % len(files)

    templates = None
    for ff in range(len(files)):
        newWave, newSpec = load_spectrum_medresIR(files[ff], velScale)

        if templates == None:
            templates = np.zeros((len(files), len(newSpec)), dtype=float)

        templates[ff,:] = newSpec

    return (newWave, templates)

def load_spectrum_gnirs(file, velScale, resolution):
    """
    Load up a spectrum from the Gemini GNIRS library.
    """
    spec, hdr = pyfits.getdata(file, header=True)

    pixScale = hdr['CD1_1'] * 1.0e-4    # microns

    wavelength = np.arange(len(spec), dtype=float)
    wavelength -= (hdr['CRPIX1']-1.0)   # get into pixels relative to the reference pix
    wavelength *= hdr['CD1_1']          # convert to the proper wavelength scale (Ang)
    wavelength += hdr['CRVAL1']         # shift to the wavelength zeropoint

    # Convert from Angstroms to microns
    wavelength *= 1.0e-4

    deltaWave = 2.21344 / resolution         # microns
    resInPixels = deltaWave / pixScale       # pixels
    sigmaInPixels = resInPixels / 2.355
    psfBins = np.arange(-4*math.ceil(resInPixels), 4*math.ceil(resInPixels))
    psf = py.normpdf(psfBins, 0, sigmaInPixels)
    specLowRes = np.convolve(spec, psf, mode='same')

    # Rebin into equals logarithmic steps in wavelength
    logWave, specNew, vel = log_rebin2(wavelength, specLowRes,
                                       inMicrons=True, velScale=velScale)

    return logWave, specNew


def load_spectrum_medresIR(file, velScale):
    """
    Load spectrum from the R~3000 Wallace & Hinkle library.
    Return:
        logWave -- log(wavelength in microns)
        specNew -- the spectral flux
    """
    f_spec = open(file, 'r')
    
    lineCount = 0
    waveNumber = []
    flux = []
    
    for line in f_spec:
        lineCount += 1
        if lineCount <= 6:
            continue
        
        fields = line.split()
        waveNumber.append( float(fields[0]) )
        flux.append( float(fields[1]) )
        
    waveNumber = np.array(waveNumber)
    flux = np.array(flux)
    
    print file
    logWave, specNew, vel = log_rebin2(waveNumber, flux, 
                                       inMicrons=False, velScale=velScale)

    return logWave, specNew

def log_rebin_cube(wavelength, cube):
    """
    Read in data cube and re-bin the spectrum at each spaxel
    to be sampled uniformly in log(wavelength).
    """
    newCube = np.zeros(cube.shape, dtype=float)
    for ii in range(cube.shape[0]):
        for jj in range(cube.shape[1]):
            logWave, spec, vel = log_rebin2(wavelength, cube[ii,jj,:])
            newCube[ii, jj, :] = spec

    return logWave, newCube, vel

def log_rebin2(xaxis, spectrum, inMicrons=True, velScale=None, plot=False):
    
    if inMicrons:
        # xaxis is currently expressed in wavelength with units of microns
        # We will assume that it is linear in wavelength
        wavelength = xaxis
    else:
        # Otherwise we assume it is wavenumbers with units of cm^-1
        # We will assume that it is linear in wavenumber.
        # Convert into wavelength in microns
        frequency = xaxis[::-1] # Reverse
        spectrum = spectrum[::-1]
        wavelength = 1.0e4 / frequency

    pixCnt = len(spectrum)

    tck = scipy.interpolate.splrep(wavelength, spectrum, s=0)
    
    # Define new wavelength scale with even steps in log(wavelength)
    if velScale == None:
        logDw = (np.log(wavelength[-1]) - np.log(wavelength[0])) 
        logDw /= (pixCnt - 1)
        velScale = logDw * cc.c
    else:
        logDw = velScale/cc.c
        pixCnt = (np.log(wavelength[-1]) - np.log(wavelength[0])) 
        pixCnt /= logDw

    logWave = np.log(wavelength[0]) + np.arange(pixCnt)*logDw

    waveNew = np.exp(logWave)
    specNew = scipy.interpolate.splev(waveNew, tck)

    if plot:
        py.clf()
        py.plot(wavelength, spectrum, 'r-')
        py.plot(waveNew, specNew, 'b-')
        py.show()

    return (logWave, specNew, velScale)


def log_rebin(wavelength, spectrum):

    cnt = len(wavelength)
    print 'cnt = ', cnt

    ###
    tck = scipy.interpolate.splrep(wavelength, spectrum, s=0)
    
    # Define new wavelength scale with even steps in log(wavelength)
    logDw = (np.log(wavelength[-1]) - np.log(wavelength[0])) 
    logDw /= (cnt - 1)

    logWave = np.log(wavelength[0]) + np.arange(cnt)*logDw

    waveNew = np.exp(logWave)
    specNew = scipy.interpolate.splev(waveNew, tck)
    print 'waveNew: ', waveNew
    print 'logWave: ', logWave
    print 'specNew: ', specNew
    print 'total flux = ', specNew.sum()
    py.clf()
    py.plot(waveNew, specNew, 'b-')
    ###

    waveRange = wavelength[[0,-1]]
    print 'waveRange = ', waveRange

    # Get the wavelength bin size (or scale)
    dw = wavelength[1] - wavelength[0]
    print 'dw = ', wavelength[0] - wavelength[1]
    print 'dw = ', wavelength[1] - wavelength[2]
    print 'dw = ', wavelength[2] - wavelength[3]

    # Wavelength in units of delta-lambda (dw)
    limits = (waveRange / dw) + np.array([-0.5, 0.5])
    print 'limits = ', limits

    dwWave = wavelength / dw
    borders = np.append(dwWave - 0.5, dwWave[-1] + 0.5)
    print 'borders = ', borders[0:5], borders[-1]

    # Get the natural log
    logLimits = np.log(limits)
    print 'logLimits = ', logLimits

    velScale = cc.c * (logLimits[1] - logLimits[0]) / cnt
    print 'velScale = ', velScale
    
    logDw = (logLimits[1]-logLimits[0]) / (cnt)
    newBorders = np.exp(logLimits[0] + np.arange(cnt+1) * logDw)
    print 'newBorders.shape = ', newBorders.shape
    print 'newBorders = ', newBorders[0:5], newBorders[-1]
    
    newBordersTmp = newBorders[0:-1]

    k = np.floor(newBorders - limits[0])
    k = np.where(k > 0, k, 0)
    k = np.where(k < cnt-1, k, cnt-1)

    print 'cnt-1 = ', cnt-1
    print 'k[0:15] = ', k[0:15] 
    print 'k[-5:] = ', k[-5:]
                        
    logWave = np.log(np.sqrt(newBorders[1:] * newBordersTmp) * dw)
    print 'logWave = ', logWave[0:5], logWave[-1]

    specNew = np.zeros(spectrum.shape, dtype=float)
    qualityNew = np.zeros(spectrum.shape, dtype=float)
    
    for ii in range(cnt):
        a = newBorders[ii] - borders[k[ii]]
        b = borders[k[ii+1]+1] - newBorders[ii+1]

        if len(spectrum.shape) == 3: 
            specNew[:,:,ii] = spectrum[:,:,k[ii]:k[ii+1]+1].sum(axis=2)
            specNew[:,:,ii] -= a*spectrum[:,:,k[ii]] 
            specNew[:,:,ii] -= b*spectrum[:,:,k[ii+1]]
        else:
            specNew[ii] = spectrum[k[ii]:k[ii+1]+1].sum()
            specNew[ii] -= a*spectrum[k[ii]] 
            specNew[ii] -= b*spectrum[k[ii+1]]

    py.plot(np.exp(logWave), specNew, 'g-')
    py.plot(wavelength, spectrum, 'r-')
    py.show()
    print 'total flux = ', specNew.sum()
    return (logWave, specNew)


def play(frequency, spectrum):

    # Reverse.
    # Remember that these are the center values of each spectral channel.
    frequency = frequency[::-1]
    spectrum = spectrum[::-1]
    wavelength = 1.0e4 / frequency

    cnt = len(wavelength)

    # Get the borders
    # assume uniform sampling in frequency
    dFreq = frequency[1] - frequency[0]

    freq1 = frequency[0] - dFreq
    freq2 = frequency[-1] + dFreq

    # Extra center values
    freqTmp = np.concatenate(([freq1], frequency, [freq2]))
    waveTmp = 1.0e4 / freqTmp

    print 'frequency.shape = ', frequency.shape
    print 'freqTmp.shape = ', freqTmp.shape
    print 'wavelength.shape = ', wavelength.shape
    print 'waveTmp.shape = ', waveTmp.shape

    # Now calculate delta-Lambda for each central value, by taking
    # finding delta Lambda from the pixel to the left and right and
    # averaging the deltaLambda.
    dWave1 = waveTmp[1:-1] - waveTmp[0:-2] # length same as original
    dWave2 = waveTmp[2:] - waveTmp[1:-1]
    dWave = (dWave1 + dWave2) / 2.0

    print 'dWave1.shape = ', dWave1.shape
    print 'dWave2.shape = ', dWave2.shape
    print 'dWave.shape = ', dWave.shape
    print ''
    print 'dWave = ', dWave
    
    # Construct the current borders
    borders = np.append(wavelength - (dWave/2.0), 
                        wavelength[-1] + (dWave[-1]/2.0))

    # We will be working with log(wave) so lets convert
    logWave = np.log(wavelength)
    logBorders = np.log(borders)
                             
    # Calculate the new dlogWave by assuming same range and same number
    # of pixels as input spectrum.
    dlogWave = (logBorders[-1] - logBorders[0]) / cnt

    # Calculate new borders and wavelengths
    newLogBorders = logBorders[0] + dlogWave * np.arange(cnt+1)
    newLogWave = newLogBorders[0:-1] + (dlogWave/2.0)
    newBorders = np.exp(newLogBorders)
    newWave = np.exp(newLogWave)

    print 'wavelength = ', wavelength
    print ''
    print 'borders = ', borders
    print 'newBorders = ', newBorders
    print ''
    print 'logWave = ', logWave
    print 'newLogWave = ', newLogWave
    print ''
    print 'logBorders = ', logBorders
    print 'newLogBorders = ', newLogBorders

    # Calculate the new flux
    specNew = np.zeros(spectrum.shape, dtype=float)
    
    residual = 0.0
    for ii in range(cnt):
        idxLo = ((np.where(borders <= newBorders[ii]))[0]).max()
        idxHi = ((np.where(borders >= newBorders[ii+1]))[0]).min()

        specNew[ii] = spectrum[idxLo:idxHi].sum()
        
        binSizeLo = borders[idxLo+1] - borders[idxLo]
        binSizeHi = borders[idxHi] - borders[idxHi-1]

        a = (newBorders[ii] - borders[idxLo]) / binSizeLo
        b = (borders[idxHi] - newBorders[ii+1]) / binSizeHi
        
        print 'ii = ', ii, ' idxLo = ', idxLo, ' idxHi = ', idxHi
        print '    a = ', a, ' b = ', b
        print '    specNew = ', specNew[ii]
        specNew[ii] -= a * spectrum[idxLo]
        print '    specNew = ', specNew[ii]
        specNew[ii] -= b * spectrum[idxHi-1]
        print '    specNew = ', specNew[ii]
    
    py.clf()
    py.plot(logWave, spectrum, 'b-')
    py.plot(newLogWave, specNew, 'g-')
    py.show()

    return newLogWave, specNew


def test():
    cubefits = pyfits.open(datadir + cuberoot + '.fits')
    
    cube = cubefits[0].data
    hdr = cubefits[0].header
    errors = cubefits[1].data
    quality = cubefits[2].data

    # Get the wavelength solution so we can specify range:
    w0 = hdr['CRVAL1']
    dw = hdr['CDELT1']
    wavelength = w0 + dw * np.arange(cube.shape[2], dtype=float)
    wavelength /= 1000.0   # Convert to microns
    
    # Lets temporary run a single spectrum
    spectrum = cube[10,30,:]

#     print 'IDL version:'
#     idl.specNew1 = np.zeros(cube.shape[2], dtype=float)
#     idl.logWave1 = np.zeros(len(wavelength), dtype=float)
#     idl.waveRange = [wavelength[0], wavelength[-1]]
#     idl.spec = spectrum
#     idl('log_rebin, waveRange, spec, specNew1, logWave1')

#     print 'PYTHON VERSION:'
#     (logWave2, specNew2) = log_rebin(wavelength, cube[10,30,:])
#     print 'logWave2 = ', logWave2
#     print 'specNew2 = ', specNew2
#     (logWave3, specNew3) = log_rebin(wavelength, cube)

#     print idl.logWave1
#     print logWave2
#     print idl.specNew1
#     print specNew2
#     print specNew3[10,30,:]

#     logWave = idl.logWave1
#     logSpec = idl.specNew1

    # Load templates
    #templates = load_templates()
    print 'Templates:'
    load_spectrum('/u/jlu/work/atlasSpectra/medresIR/K_band/spec.HR21')

    # Test #2
    logWave3, specNew3 = log_rebin2(wavelength, cube[10,30,:])


