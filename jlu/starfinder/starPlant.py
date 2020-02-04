from gcwork import starset
from gcwork import starTables
from gcwork import objects
import numpy as np
import pylab as py
#import pyfits
from gcreduce import gcutil
import os, shutil
import glob
import pickle
import math
import pdb



class Simulation(object):
    def __init__(self, imageRoot, starlist, corr=0.8):
        self.imageRoot = imageRoot
        self.starlist = starlist
        self.corr = corr
                 
    def makeData(self, remakeImages=True, stepPerRun=2, stepFinal=0.5,
                 magMin=None, magMax=None, magStep=0.25, magBins=None,
                 xyinit=None):
        """
        Make data with artificial stars added. Copy over all the
        necessary files to be able to re-run starfinder.

        remakeImages (def=True) - remake all the fits files and 
            starlists. If False, then just ste some variables.
        psfBoxSize - The size of the PSF in arcsec.
        stepPerRun - The seperation between planted stars per 
            Starfinder run in arcseconds. The default is 2 which is the
            PSF box size (so no overlap).
        stepFinal - The final seperation between artificial stars
            when considering the sum of all grids (arcsec).

        magMin - The minimum magnitude of stars to simulate. If None, then
            use the minimum magnitude of stars in the data itself.
        magMax - The maximum magnitude of stars to simulate.If None, then
            use the maximum magnitude of stars in the data itself.
        magStep - The step size of the magnitude bins. Default = 0.25.
        """
        self.fitsFile = self.imageRoot + '.fits'
        self.psfFile = self.imageRoot + '_psf.fits'
        self.backFile = self.imageRoot + '_back.fits'
        self.maxFile = self.imageRoot + '.max'
        self.cooFile = self.imageRoot + '.coo'

        # Generate a list of sources to be planted in the image. This
        # should be done one magnitude bin at a time.
        pixelScale = 0.01  # arcsec/pixel

        stepPerRun /= pixelScale
        stepFinal /= pixelScale

        # Number of iterations per magnitude bin (in each direction)
        stepCountPerSide = round(stepPerRun / stepFinal)

        # Read in the image and starlist
        img, hdr = pyfits.getdata(self.fitsFile, header=True)
        psf = pyfits.getdata(self.psfFile)
        stars = starTables.StarfinderList(self.starlist, hasErrors=False)

        # Get the mag to flux scale factor (e.g. zeropoints). 
        flux0 = stars.counts[0] / 10**(-0.4 * stars.mag[0])

        xsize = img.shape[1]
        ysize = img.shape[0]

        if magMin == None:
            magMin = stars.mag.min()+0.1
        if magMax == None:
            magMax = stars.mag.max()

        self.magMin = magMin
        self.magMax = magMax
        self.magStep = magStep

        magBins = np.arange(magMin, magMax, magStep)

        if xyinit == None:
            xyinit = np.array([stepFinal, stepFinal])

        self.xyinit = xyinit

        newStarCount = 0

        newFitsFiles = []
        newStarlists = []

        for mag in magBins:
            flux = flux0 * 10**(-0.4 * mag)

            for ii in range(stepCountPerSide):
                xinit_this_run = xyinit[0] + ii*stepFinal
                x1d = np.arange(xinit_this_run, xsize, stepPerRun)

                for kk in range(stepCountPerSide):
                    yinit_this_run = xyinit[1] + kk*stepFinal
                    y1d = np.arange(yinit_this_run, ysize, stepPerRun)

                    # For each "run" we are going to make a new
                    # fits file, a new starlist (copy of the old one but
                    # with artificial stars added), and copy over 
                    # the PSF, background, coo, and max files for
                    # running starfinder.
                    newImageRoot = 'sim_img_%.2f_%d_%d' % (mag, ii, kk)
                    newFitsFile = newImageRoot + '.fits' 
                    newStarlist = 'sim_orig_%.2f_%d_%d.lis' % (mag, ii, kk)

                    if remakeImages:
                        gcutil.rmall([newFitsFile, newStarlist])

                        newImage = np.zeros((ysize, xsize), dtype=float)
                        newStars = starTables.StarfinderList(self.starlist, 
                                                             hasErrors=False)

                        for x in x1d:
                            for y in y1d:
                                x = int(round(x))
                                y = int(round(y))
                                newName = 'sim_%d' % newStarCount
                                newStarCount += 1
                                
                                addStar(newImage, psf, x, y, flux)
                                newStars.append(newName, mag, x-1, y-1, counts=flux)

                        newImage += img

                        # Save off the image
                        pyfits.writeto(newFitsFile, newImage, hdr, 
                                       output_verify='silentfix')

                        # Copy over the PSF and background files
                        shutil.copyfile(self.psfFile,
                                        newImageRoot + '_psf.fits')
                        shutil.copyfile(self.backFile,
                                        newImageRoot + '_back.fits')
                        shutil.copyfile(self.maxFile, newImageRoot + '.max')
                        shutil.copyfile(self.cooFile, newImageRoot + '.coo')
                        
                        # Save off the starlist
                        newStars.saveToFile(newStarlist)

                        print('Made simulated image: ', newFitsFile)

                    newFitsFiles.append(newFitsFile)
                    newStarlists.append(newStarlist)
                    
        self.newFitsFiles = newFitsFiles
        self.newStarlists = newStarlists

    def makeStarfinderBatchFiles(self, cooStar, psfStars):
        self.cooStar = cooStar
        self.psfStars = psfStars

        print('Making IDL batch files')
        for ii in range(len(self.newFitsFiles)):
            # Write an IDL batch file
            fileIDLbatch = 'idl_' + self.newFitsFiles[ii].replace('.fits', '.batch')
            _batch = open(fileIDLbatch, 'w')
            _batch.write("find_stf, ")
            _batch.write("'" + self.newFitsFiles[ii] + "', ")
            _batch.write("%.1f, "  % self.corr)
            _batch.write("cooStar='" + self.cooStar + "', ")
            _batch.write("/trimfake, ")
            _batch.write("starlist='" + self.psfStars + "'")
            _batch.write("\n")
            _batch.write("exit\n")
            _batch.close()

    def runStarfinder(self):
        for ii in range(len(self.newFitsFiles)):
            print('Running starfinder on ', self.newFitsFiles[ii])
            # Write an IDL batch file
            fileIDLbatch = 'idl_' + self.newFitsFiles[ii].replace('.fits', '.batch')
            fileIDLlog = fileIDLbatch.replace('.batch', '.log')

            cmd = 'idl < ' + fileIDLbatch + ' >& ' + fileIDLlog
            os.system(cmd)

    def runAlign(self):
        for ii in range(len(self.newStarlists)):
            root = self.newFitsFiles[ii].replace('.fits', '')
            inputList = self.newStarlists[ii]
            outputList = '%s_%.1f_stf.lis' % (root, self.corr)

            if ((os.path.exists(inputList) == False) or 
                (os.path.exists(outputList) == False)):
                continue
            
            alignRoot = 'align/' + root.replace('sim', 'aln')
            
            _alignList = open(alignRoot + '.list', 'w')
            _alignList.write(inputList + ' 8 ref\n')
            _alignList.write(outputList + ' 8\n')
            _alignList.close()

            cmd = 'java align -a 0 -v -p -r ' + alignRoot  + ' '
            cmd += alignRoot + '.list'
            os.system(cmd)

            

def addStar(img, psf, xpos, ypos, flux):
    """
    Add an artificial star to an existing image. The PSF will be properly
    trimmed to fit within the pre-existing image boundaries.

    img - a numpy 2D array with img[y,x] coordinate system
    psf - a numpy 2D arra with psf[y,x] coordinate system. PSF should have an
          integrated flux of 1.0. 
    xpos - the X pixel position in the image where the star will be planted.
           Only integer pixels will be used to avoid interpolation.
    ypos - the Y pixel position in the image where the star will be planted.
           Only integer pixels will be used to avoid interpolation.
    flux - the total flux of the star to be planted. The PSF will be 
           multiplied by this value before adding to the image.

    The PSF is copied before any trimming is done, but the IMG is modified
    in place.
    """
    # Figure out the box within the image where the PSF will be added.
    # Code only allows integer pixel shifts, so no interpolation is done.
    xlo = int(round(  xpos - (psf.shape[1] / 2)  ))
    xhi = xlo + psf.shape[1]

    ylo = int(round(  ypos - (psf.shape[0] / 2)  ))
    yhi = ylo + psf.shape[0]

    # Handle boundaries by trimming down the PSF.
    psfTrim = psf.copy() # full PSF

    if ylo < 0:
        psfTrim = psfTrim[-ylo:,:]
        ylo = 0
    if xlo < 0:
        psfTrim = psfTrim[:,-xlo:]
        xlo = 0
    if yhi > img.shape[0]:
        diff = img.shape[0] - yhi
        psfTrim = psfTrim[:diff,:]
        yhi = img.shape[0]
    if xhi > img.shape[1]:
        diff = img.shape[1] - xhi
        psfTrim = psfTrim[:,:diff]
        xhi = img.shape[1]

    img[ylo:yhi,xlo:xhi] += psfTrim * flux


def gather_results(alignDir, mag0=None, flux0=None):
    print('Gathering results from ')
    print('   %s' % alignDir)

    # Get all the align output files... these should correspond to each
    # simulated image.
    velFiles = glob.glob(alignDir + 'aln_img*.vel')
    runFiles = glob.glob(alignDir + 'aln_img*.run')

    if len(runFiles) != len(velFiles):
        '**** !!!! Missing align files !!!! ****'

    # Make 1D huge lists with the input/output X and Y positions
    # and fluxes for all the simulated stars. At this point, we won't
    # do anything with the other stars.
    x_in = []
    y_in = []
    m_in = []
    f_in = []

    x_out = []
    y_out = []
    m_out = []
    f_out = []

    # Might as well build up our list of magnitudes and which 
    # items belong to which mag bin now.
    #   midx -- will be 2D (total length the same  as xin) where the 
    #           first dimension matches munique and the second dimension
    #           will have indices into xin/yin for all stars of a given n
    #           magnitude.
    #   munique -- will contain the unique magnitudes.
    midx = []  
    munique = []

    currentMag = 100.0

    # Loop through each of the align runs and gather data.
    for ii in range(len(velFiles)):
        print('Adding data from ', velFiles[ii])
        s = starset.StarSet(velFiles[ii].replace('.vel', ''))

        # Get flux zeropoint
        if mag0 == None:
            mag0 = s.stars[0].e[0].mag
        if flux0 == None:
            flux0 = s.stars[0].e[0].fwhm 

        # Find all the planted sources
        simStars = []
        origStars = []
        for star in s.stars:
            if star.name.startswith('sim'):
                simStars.append(star)
            elif not star.name.startswith('1'):
                origStars.append(star)

        # Trim out simulated stars that are in the "padded" regions.
        s.stars = origStars
        x_orig = s.getArrayFromEpoch(0, 'xorig')
        y_orig = s.getArrayFromEpoch(0, 'yorig')
        
        s.stars = simStars
        x_sim = s.getArrayFromEpoch(0, 'xorig')
        y_sim = s.getArrayFromEpoch(0, 'yorig')

        good = np.where((x_sim >= x_orig.min()) & (x_sim <= x_orig.max()) & 
                        (y_sim >= y_orig.min()) & (y_sim <= y_orig.max()))[0]
        print('Trimming out %d of %d stars in padded regions.' % \
            (len(simStars) - len(good), len(simStars)))
        s.stars = [simStars[gg] for gg in good]
        
        # Original info is what we planted
        x_in_ii = s.getArrayFromEpoch(0, 'xorig')
        y_in_ii = s.getArrayFromEpoch(0, 'yorig')
        f_in_ii = s.getArrayFromEpoch(0, 'fwhm')
        # Crude calibration
        #m_in_ii = s.getArrayFromEpoch(0, 'mag')
        m_in_ii = -2.5 * np.log10( f_in_ii / flux0 ) + mag0 

        # Resulting info is what Starfinder recovered
        x_out_ii = s.getArrayFromEpoch(1, 'xorig')
        y_out_ii = s.getArrayFromEpoch(1, 'yorig')
        f_out_ii = s.getArrayFromEpoch(1, 'fwhm')
        # Crude calibration
        m_out_ii = -2.5 * np.log10( f_out_ii / flux0 ) + mag0 
 
        # Figure out which magnitude bin this data goes into.
        # If we don't have a magnitude bin for these sources yet,
        # then create one.
        if m_in_ii[0] != currentMag:
            currentMag = m_in_ii[0]
            midx.append([])
            munique.append(currentMag)

        midx[-1].extend(list(range(len(x_in), len(x_in)+len(x_in_ii), 1)))

        x_in.extend(x_in_ii)
        y_in.extend(y_in_ii)
        m_in.extend(m_in_ii)
        f_in.extend(f_in_ii)

        x_out.extend(x_out_ii)
        y_out.extend(y_out_ii)
        m_out.extend(m_out_ii)
        f_out.extend(f_out_ii)

    # Now lets make them numpy arrays for calculations
    x_in = np.array(x_in)
    y_in = np.array(y_in)
    m_in = np.array(m_in)
    f_in = np.array(f_in)
    
    x_out = np.array(x_out)
    y_out = np.array(y_out)
    m_out = np.array(m_out)
    f_out = np.array(f_out)

    munique = np.array(munique)
    idx = munique.argsort()
    midx_new = []
    for ii in idx:
        midx_new.append(midx[ii])
    midx = midx_new
    munique = munique[idx]
    
    # Also build up a list of independent pixels and indices for 
    # those entries that belong to each pixel.
    xunique1d = np.unique(x_in)
    yunique1d = np.unique(y_in)
    xunique = []
    yunique = []
    xy_idx = []

    countAtEachPixel = None

    for xx in xunique1d:
        for yy in yunique1d:
            idx = np.where((x_in == xx) & (y_in == yy))[0]
            if len(idx) > 0:
                xunique.append(xx)
                yunique.append(yy)
                xy_idx.append(idx)

            if countAtEachPixel == None:
                countAtEachPixel = len(idx)

            if len(idx) != countAtEachPixel:
                print('Different number of stars at this pixel. ')

    xunique = np.array(xunique)
    yunique = np.array(yunique)

    # Calculate the completeness as a function of magnitude
    completeness = np.zeros(len(munique), dtype=float)
    countPlanted = np.zeros(len(munique), dtype=int)
    countFound = np.zeros(len(munique), dtype=int)

    for mm in range(len(munique)):
        countPlanted[mm] = len(midx[mm])

        idx = np.where(x_out[midx[mm]] > 0)[0]
        countFound[mm] = len(idx)

        completeness[mm] = float(countFound[mm]) / float(countPlanted[mm])
        
        print('Mag = %.2f   Completeness = %.2f   (%4d planted, %4d found)' % \
            (munique[mm], completeness[mm], countPlanted[mm], countFound[mm]))


    # Save Everything to Pickle Files
    _out = open(alignDir + 'all_results.dat', 'w')
    pickle.dump(x_in, _out)
    pickle.dump(y_in, _out)
    pickle.dump(m_in, _out)
    pickle.dump(f_in, _out)

    pickle.dump(x_out, _out)
    pickle.dump(y_out, _out)
    pickle.dump(m_out, _out)
    pickle.dump(f_out, _out)

    pickle.dump(munique, _out)
    pickle.dump(midx, _out)

    pickle.dump(countPlanted, _out)
    pickle.dump(countFound, _out)
    pickle.dump(completeness, _out)

    pickle.dump(xunique, _out)
    pickle.dump(yunique, _out)
    pickle.dump(xy_idx, _out)

    _out.close()

def plot_results(alignDir):
    _pickleFile = open(alignDir + 'all_results.dat', 'r')
 
    x_in = pickle.load(_pickleFile)
    y_in = pickle.load(_pickleFile)
    m_in = pickle.load(_pickleFile)
    f_in = pickle.load(_pickleFile)

    x_out = pickle.load(_pickleFile)
    y_out = pickle.load(_pickleFile)
    m_out = pickle.load(_pickleFile)
    f_out = pickle.load(_pickleFile)

    munique = pickle.load(_pickleFile)
    midx = pickle.load(_pickleFile)

    countPlanted = pickle.load(_pickleFile)
    countFound = pickle.load(_pickleFile)
    completeness = pickle.load(_pickleFile)

    # Also, for all the detected sources, lets calculate the 
    # difference in positions and fluxes.
    idx = np.where(x_out > 0)

    xdiff = x_out[idx] - x_in[idx]
    ydiff = y_out[idx] - y_in[idx]
    fratio = f_in[idx] / f_out[idx]
    mdiff = -2.5 * np.log10(fratio)

    xybins = np.arange(-1.0, 1.0, 0.01)
    mbins = np.arange(-0.1, 0.1, 0.005)

    py.figure(3)
    py.clf()
    py.plot(munique, completeness, 'ko-')
    py.xlabel('Kp Magnitude')
    py.ylabel('Completeness')
    py.savefig(alignDir + 'star_plant_completeness2.png')
    
    py.figure(1)
    py.clf()
    py.hist(xdiff, label='X', histtype='step', bins=xybins)
    py.hist(ydiff, label='Y', histtype='step', bins=xybins)
    py.legend()
    py.xlabel('Positional Offsets (pix)')
    py.ylabel('Number of Simulated Stars')
    py.savefig(alignDir + 'star_plant_xydiff_hist.png')

    py.figure(2)
    py.clf()
    py.hist(mdiff, histtype='step', bins=mbins)
    py.xlabel('Photometric Offset (mag)')
    py.ylabel('Number of Simulated Stars')
    py.savefig(alignDir + 'star_plant_mdiff_hist.png')
    
    print('xdiff = %5.2f +/- %4.2f pix' % (xdiff.mean(), xdiff.std()))
    print('ydiff = %5.2f +/- %4.2f pix' % (ydiff.mean(), ydiff.std()))
    print('mdiff = %5.2f +/- %4.2f mag' % (mdiff.mean(), mdiff.std()))

    print('median(abs( xdiff )) = %5.3f pix' % (np.median(np.abs(xdiff))))
    print('median(abs( ydiff )) = %5.3f pix' % (np.median(np.abs(ydiff))))
    print('median(abs( mdiff )) = %5.3f mag' % (np.median(np.abs(mdiff))))


def plot_completeness(alignDir):
    _pickleFile = open(alignDir + 'all_results.dat', 'r')
 
    x_in = pickle.load(_pickleFile)
    y_in = pickle.load(_pickleFile)
    m_in = pickle.load(_pickleFile)
    f_in = pickle.load(_pickleFile)

    x_out = pickle.load(_pickleFile)
    y_out = pickle.load(_pickleFile)
    m_out = pickle.load(_pickleFile)
    f_out = pickle.load(_pickleFile)

    munique = pickle.load(_pickleFile)
    midx = pickle.load(_pickleFile)

    countPlanted = pickle.load(_pickleFile)
    countFound = pickle.load(_pickleFile)
    completeness = pickle.load(_pickleFile)

    # Print out the completeness results.
    # Also save to a file.
    _out = open(alignDir + 'completeness.dat', 'w')
    _out.write('#%12s  %13s  %13s  %13s\n' %
               ('K_mag', 'Completness', 'N_planted', 'N_found'))
    for mm in range(len(munique)):
        print('Mag = %.2f   Completeness = %.2f   (%4d planted, %4d found)' % \
            (munique[mm], completeness[mm], countPlanted[mm], countFound[mm]))
        _out.write('%13.2f  %13.2f  %13d  %13d\n' %
                   (munique[mm], completeness[mm], 
                    countPlanted[mm], countFound[mm]))

    _out.close()

    # Plot the total completeness curve
    py.figure(3)
    py.clf()
    py.plot(munique, completeness, 'ko-')
    py.xlabel('Kp Magnitude')
    py.ylabel('Completeness')
    py.ylim(0, 1.1)
    py.savefig(alignDir + 'star_plant_completeness.png')

    # Plot completeness vs. radius from center of the image
    rbins = np.arange(0, 8, 1) / 0.01  # convert to pixels
    rbins = np.array([0, 1, 3, 5, 7]) / 0.01  # convert to pixels

    xmid = x_in.min() + ((x_in.max() - x_in.min()) / 2.0)
    ymid = y_in.min() + ((y_in.max() - y_in.min()) / 2.0)
    r = np.hypot(x_in - xmid, y_in - ymid)

    py.clf()
    for rr in range(len(rbins)-1):
        r_completeness = np.zeros(len(munique), dtype=float)
        r_comp_err = np.zeros(len(munique), dtype=float)

        for mm in range(len(munique)):
            x_at_mag_out = x_out[midx[mm]]
            x_at_mag = x_in[midx[mm]] 
            y_at_mag = y_in[midx[mm]]
            r_at_mag = np.hypot(x_at_mag - xmid, y_at_mag - ymid)

            # Get stars in this radius bin and that are detected by
            # starfinder
            idx = np.where((x_at_mag_out > 0) & 
                           (r_at_mag > rbins[rr]) & 
                           (r_at_mag <= rbins[rr+1]))[0]
            idx2 = np.where((r_at_mag > rbins[rr]) & 
                            (r_at_mag <= rbins[rr+1]))[0]

            r_countFound = len(idx)
            r_countPlanted = len(idx2)
            
            r_completeness[mm] = float(r_countFound) / float(r_countPlanted)
            if (r_countFound != 0):
                r_comp_err[mm] = math.sqrt((1./r_countFound) + (1./r_countPlanted))
                r_comp_err[mm] *= r_completeness[mm]
            
        label = '%d"<r<= %d" (%d stars per bin)' % (rbins[rr]*0.01, 
                                                       rbins[rr+1]*0.01, 
                                                       r_countPlanted)
        py.plot(munique, r_completeness, label=label, linewidth=2)
#         py.fill_between(munique, r_completeness-r_comp_err, 
#                         r_completeness+r_comp_err, label='_none_',
#                         color='grey', alpha=0.3)
        
    py.plot(munique, completeness, 'k-', linewidth=2)
    py.legend(loc='lower left', prop={'size':12})
    py.xlabel('Kp Magnitude')
    py.ylabel('Completeness')
    py.ylim(0, 1.1)
    py.savefig(alignDir + 'star_plant_completeness_r.png')


def load_completeness(alignDir):
    print('Loading star planting data from:')
    print('   %s' % alignDir + 'all_results.dat')

    _pickleFile = open(alignDir + 'all_results.dat', 'r')
    
    data = objects.DataHolder()

    # Individual measurements for every planted star
    data.x_in = pickle.load(_pickleFile)
    data.y_in = pickle.load(_pickleFile)
    data.m_in = pickle.load(_pickleFile)
    data.f_in = pickle.load(_pickleFile)

    data.x_out = pickle.load(_pickleFile)
    data.y_out = pickle.load(_pickleFile)
    data.m_out = pickle.load(_pickleFile)
    data.f_out = pickle.load(_pickleFile)

    # Combined for unique magnitude bins.
    data.mag = pickle.load(_pickleFile)
    data.midx = pickle.load(_pickleFile)

    data.countPlanted = pickle.load(_pickleFile)
    data.countFound = pickle.load(_pickleFile)
    data.completeness = pickle.load(_pickleFile)

    # Combined for unique positions on the detector.
    data.x = pickle.load(_pickleFile)
    data.y = pickle.load(_pickleFile)
    data.xy_idx = pickle.load(_pickleFile)

    return data

