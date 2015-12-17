from __future__ import absolute_import
import pyfits
from gcutil import statsWeighted
from gcwork import starTables
from gcwork import objects
from gcreduce import gcutil
from gcwork import starset
from gcreduce import dataUtil
import asciidata
import pylab as py
import numpy as np
import math
import shutil, os
import pdb


class Mosaic(object):
    """
    Object that will mosaic starlists from individual images in a mosaic
    sequence together to produce a final starlist (*_rms.lis) file.

    Here is an example of how to run this script from an <epoch>/reduce/ 
    directory under /u/ghezgroup/data/gc/

    from gcreduce import gcmosaic_starlists as gcmosaic
    msc = gcmosaic.MosaicDeep('08maylgs2')
    msc.makeMosaic()

    This should work by default; but for epochs where the fields are 
    different then you could do something like:
    
    mscFields = ['C', 'E', 'W', 'N', 'S', 'NE', 'SE', 'NW', 'SW']
    msc = gcmosaic.MosaicDeep('10junlgs', fieldsToMosaic=mscFields)
    msc.makeMosaic()

    """
    def __init__(self, epoch, mosaicName, filter, 
                 fieldsToMosaic, refStarsDict, 
                 firstRefList=0, 
                 rootDir='/g/lu/data/orion/', 
                 epochDirSuffix=None,
                 scale=0.00995, alignOrder=0):
        """
        epoch - The epoch to work on. This is used in the 
                directory structure. Example: '06maylgs'
                
        mosaicName - The name of the mosaic. Example: 'dp_msc' or 'msr'

        filter - Specify the filter information (e.g. 'kp'). This gives
                mag<epoch>_<mosaicName>_<fieldsToMosaic[i]>_<filt> 
                as the rootname of the starlists.

        fieldsToMosaic - A list of field names to mosaic together. Order
                matters. Example: ['C', 'C_NW', 'C_SW', 'C_SE', 'C_NE']

        refStarsDict - A Dictionary mapping the field name string for
                each position in the mosaic to a 2 element array that
                contains the names of two reference stars used to do
                the initial alignment. 
                
                This is the key element to mosaicing starlists together and 
                if you change the order that the starlists are mosaiced, then 
                you may need to change which reference stars get used. This 
                is particularly important if firstRefList=0 instead of some
                large FOV list like label.dat.

                Overriding classes should usually specify a default 
                value for this parameter.

        firstRefList - (default = 0). Specify the starlist to be used
                as the reference list in the very first alignment. This
                can be an integer that gives the index into fieldsToMosaic.
                Or it can be a string pointing to any starlist. You can also
                set it to a special value ending with 'label.dat' and 
                the label.dat file will be read in and reformated for 
                use in align. 

        rootDir - The root directory of the data.
        
        epochDirSuffix - Useful for testing. For example, 10junlgs_old/ would
                have epochDirSuffix='old'.
        """
        ##########
        # Static variables. Modify at your own risk. Values extracted from
        # Yelda et al. 2010.
        ##########
        # Residual distortion error
        self.distResid = 0.1 # pix

        # Average uncertainty in distortion solution
        self.distErr_X = 0.05 # pix
        self.distErr_Y = 0.04 # pix

        self.addErrX = math.sqrt(self.distErr_X**2 + self.distResid**2)
        self.addErrY = math.sqrt(self.distErr_Y**2 + self.distResid**2)

        ##########
        # Variables passed in on init.
        ##########
        self.rootDir = rootDir
        self.epochDirSuffix = epochDirSuffix
        self.epoch = epoch
        self.mosaic = mosaicName
        self.filter = filter
        self.fields = fieldsToMosaic
        self.refStars = refStarsDict
        self.alignOrder = alignOrder

        # Setup the roots of the images
        self.images = []
        for field in self.fields:
            imgRoot = 'mag' + self.epoch + '_' + self.mosaic + '_'
            imgRoot += field + '_' + self.filter
            self.images.append(imgRoot)

        # Setup directories
        self.dataDir = self.rootDir + self.epoch
        if self.epochDirSuffix:
            self.dataDir += self.epochDirSuffix
        self.dataDir += '/combo/starfinder/'

        self.mosaicDir = self.dataDir + 'mosaic_' + self.mosaic + '/'
        self.lisDir = self.mosaicDir + 'lis/'
        self.tableDir = self.mosaicDir + 'tables/'
        self.alignDir = self.mosaicDir + 'align/'

        # Make these directories
        gcutil.mkdir(self.mosaicDir)
        gcutil.mkdir(self.lisDir)
        gcutil.mkdir(self.tableDir)
        gcutil.mkdir(self.alignDir)

        if type(firstRefList) is str:
            if firstRefList.endswith('label.dat'):
                self.makeFirstRefList(firstRefList, scale=scale)
                self.firstRefList = 'label_rms.lis'
            else:
                # Copy this starlist into our lis/ directory
                shutil.copyfile(firstRefList, self.lisDir)
                # Use the copied version... we just need the root name.
                self.firstRefList = firstRefList.split('/')[-1]
        elif type(firstRefList) is int:
            # Use the *_rms.lis file for the specified field.
            self.firstRefList = self.images[firstRefList]
            self.firstRefList += '_rms_dist.lis'

    def makeFirstRefList(self, labelFile, scale=0.00995):
        """
        Read in a label.dat file specificed in <labelFile> and convert
        it into a starfinder *_rms.lis file. Coordinates in label.dat 
        are assumed to be in arcseconds and they will be converted to
        NIRC2 narrow pixels using <scale>. Sgr A* will be centered at
        x=2000, y=2000 in the output list. 

        labelFile - (string) The name of the label.dat file.
        scale - (float) The pixels scale in arcsec/pixel.
        """
        
        labels = starTables.Labels(labelFile=labelFile)
        starlist = starTables.StarfinderList(None, hasErrors=True)

        # We need to pull out the date of this epoch. Pull this from the
        # first image starlist in the mosaic.
        firstList = self.dataDir + self.images[0] + '_rms.lis'
        tmpList = starTables.StarfinderList(firstList)
        date = tmpList.epoch[0]

        # Trim label list to only stars with use? != 0
        idx = np.where(labels.useToAlign == 1)[0]
        labels.take(idx)

        # Convert label velocities to arcsec/yr instead of mas/yr
        labels.vx /= 1000.0
        labels.vy /= 1000.0
        labels.vxerr /= 1000.0
        labels.vyerr /= 1000.0

        starlist.name = labels.name
        starlist.mag = labels.mag
        starlist.epoch = np.zeros(len(labels.name), dtype=float)
        starlist.epoch += date
        
        dt = date - labels.t0
        starlist.x = ((labels.x + labels.vx * dt) / -scale) + 2000.0
        starlist.y = ((labels.y + labels.vy * dt) / scale) + 2000.0
        starlist.xerr = np.sqrt(labels.xerr**2 + (labels.vxerr * dt)**2)
        starlist.xerr /= scale
        starlist.yerr = np.sqrt(labels.yerr**2 + (labels.vyerr * dt)**2)
        starlist.yerr /= scale
        starlist.snr = np.ones(len(labels.name))
        starlist.corr = np.ones(len(labels.name))
        starlist.nframes = np.ones(len(labels.name))
        starlist.counts = np.ones(len(labels.name))

        outFile = self.lisDir + labelFile.split('/')[-1]
        outFile = outFile.replace('.dat', '_rms.lis')
        print 'Saving to %s' % outFile

        starlist.saveToFile(outFile)
            
    def makeMosaic(self):
        """
        Take the input list of images and slowly align them one by one.
        We do this by aligning to the first image, then we align to the 
        resulting master list after this.
        """
        fields = []
        for field in self.fields:
            fields.append('_%s_%s_%s' % (self.mosaic, field, self.filter))

        # Copy the starlists from original data directory. This includes
        # copying the reference star list if it is different.
        self.copy_starlists()

        # Add the residual distortion (0.1 pix) and the distortion uncertainty
        # to the centroiding errors in each star list
        self.add_distErr_to_lis()

        # Loop through and align the images to a master list that
        # is slowly being built up starting from the firstRefList.
        self.alignToMakeMaster()

        # Now align each of the individual fields to the final master list
        self.alignToMaster()

        # Extract all the data from each of the fields alignments to 
        # the master list
        self.extractAstrometry()

        # Make a named starlist
        self.makeNamedStarlist()

    def copy_starlists(self):
        """
        Copy the original *_rms.lis into a new location.
        """
        for image in self.images:
            shutil.copyfile(self.dataDir + image + '_rms.lis', 
                            self.lisDir + image + '_rms.lis')


    def add_distErr_to_lis(self):
        """
        For every single *_rms.lis file, modify the error columns
        to include distortion errors and distortion residuals.
        """
        for ii in range(len(self.images)):
            # read in this field's star list
            starlist = self.lisDir + self.images[ii] + '_rms.lis'

            lis = starTables.StarfinderList(starlist, hasErrors=True)

            lis.xerr = np.hypot(lis.xerr, self.addErrX)
            lis.yerr = np.hypot(lis.yerr, self.addErrY)

            outlist = starlist.replace('rms', 'rms_dist')
            lis.saveToFile(outlist)


    def alignToMakeMaster(self):
        """
        Loop through and align the images to the master list that
        is slowly being built up.
        Include the alignment of central to central field in order to
        include the distortion error in that image and to include those
        positions when computing the average
        """
        # Define a running variable that contains the refList. This
        # will be updated after each loop.
        refList = self.firstRefList


        for ii in range(0, len(self.images)):
            alignRoot = self.alignDir + 'align_'
            alignRoot += str(ii) + '_' + self.fields[ii]

            alignRootBoot = self.alignDir + 'align_boot_'
            alignRootBoot += str(ii) + '_' + self.fields[ii]

            starList = self.images[ii] + '_rms_dist.lis'

            # ==========
            # Re-order refList to shift common source to the top.
            # ==========
            newRefList = self.lisDir + 'ref_%d_%s' % (ii, refList)
            newStarList = self.lisDir + 'nirc2_%d_%s' % (ii, starList)

            # Ref List
            shiftToTopOfList(self.lisDir + refList, newRefList, 
                             self.refStars[self.fields[ii]])

            # Star List
            shiftToTopOfList(self.lisDir + starList, newStarList, 
                             self.refStars[self.fields[ii]])

            # Get the align data type
            fitsFile = self.dataDir + '../%s.fits' % (self.images[ii])
            alignType = dataUtil.get_align_type(fitsFile, errors=True)

            # Get the angle
            posAngle = dataUtil.get_pos_angle(fitsFile)
            if posAngle != 0 and self.alignOrder == 0:
                print 'You must allow for rotation with alignOrder > 0'

            # ==========
            # Align
            # ==========
            _list = open(alignRoot + '.list', 'w')
            _list.write('%s %s ref\n' % (newRefList, alignType))
            _list.write('%s %d\n' % (newStarList, alignType))
            _list.close()

            print '\n*** Aligning %s ***' % starList
            cmd = 'java align -v -p -a %d -r %s %s.list' % (self.alignOrder, alignRoot, alignRoot)
            os.system(cmd)

            # Bootstrap
            print '\n*** Aligning %s (bootstrap) ***' % starList
            ntrials = 100
            cmd = 'java align -v -p -a %d -n %d -r %s %s.list' % \
                (self.alignOrder, ntrials, alignRootBoot, alignRoot)
            os.system(cmd)

            # ==========
            # Make a new reference list out of the previous align results
            # This builds up a list after each alignment
            # ==========
            refList = 'aligned_%d_%s' % (ii, starList)
            starPrefix = 'ep%d' % ii

            makeNewRefList(alignRootBoot, self.lisDir + refList, 
                           starPrefix=starPrefix)

        shutil.copyfile(self.lisDir + refList, self.lisDir + 'master.lis')


    def alignToMaster(self):
        refList = 'master.lis'

        for ii in range(len(self.images)):
            alignRoot = self.alignDir + 'master_' + str(ii)
            alignRoot += '_' + self.fields[ii]
            
            alignRootBoot = self.alignDir + 'master_boot_' + str(ii)
            alignRootBoot += '_' + self.fields[ii]

            starList = self.images[ii] + '_rms_dist.lis'

            print '*** Aligning to master:'
            print '*** Working on %s ***' % starList
            print

            # ==========
            # Re-order refList to shift common source to the top.
            # ==========
            newRefList = self.lisDir + 'master_%d_%s' % (ii, refList) # Always the same ref list, but with
            							      # different stars at the top
            newStarList = self.lisDir + 'nirc2_%d_%s' % (ii, starList)

            # Ref List
            shiftToTopOfList(self.lisDir + refList, newRefList, 
                             self.refStars[self.fields[ii]])

            # Star List
            shiftToTopOfList(self.lisDir + starList, newStarList, 
                             self.refStars[self.fields[ii]])

            # Get the align data type
            fitsFile = self.dataDir + '../%s.fits' % (self.images[ii])
            alignType = dataUtil.get_align_type(fitsFile, errors=True)

            # Get the angle
            posAngle = dataUtil.get_pos_angle(fitsFile)
            if posAngle != 0 and self.alignOrder == 0:
                print 'You must allow for rotation with alignOrder > 0'

            # ==========
            # Align
            # ==========
            _list = open(alignRoot + '.list', 'w')
            _list.write('%s %d ref\n' % (newRefList, alignType))
            _list.write('%s %d\n' % (newStarList, alignType))
            _list.close()

            print '\n*** Aligning %s ***' % starList
            cmd = 'java align -v -p -a %d -r %s %s.list' % (self.alignOrder, alignRoot, alignRoot)
            os.system(cmd)

            # Bootstrap
            print '\n*** Aligning %s (bootstrap) ***' % starList
            ntrials = 100
            cmd = 'java align -v -p -a %d -n %d -r %s %s.list' % \
                (self.alignOrder, ntrials, alignRootBoot, alignRoot)
            os.system(cmd)

        
    def extractAstrometry(self):
        """
        Extracts all the astrometry from after each field was aligned to 
        the master star list created in alignToMaster().
        """
        # Load up the master list, purely for naming.
        master = asciidata.open(self.lisDir + 'master.lis')
        m_name = master[0].tonumpy()
        numStars = len(m_name)
        numImages = len(self.images)

        x = np.zeros((numImages, numStars), dtype=float)
        xe = np.zeros((numImages, numStars), dtype=float)
        xed = np.zeros((numImages, numStars), dtype=float)
        xea = np.zeros((numImages, numStars), dtype=float)

        y = np.zeros((numImages, numStars), dtype=float)
        ye = np.zeros((numImages, numStars), dtype=float)
        yed = np.zeros((numImages, numStars), dtype=float)
        yea = np.zeros((numImages, numStars), dtype=float)

        m = np.zeros((numImages, numStars), dtype=float)
        me = np.zeros((numImages, numStars), dtype=float)

        f = np.zeros((numImages, numStars), dtype=float)

        c = np.zeros((numImages, numStars), dtype=float)

        xshift = np.zeros(numImages, dtype=float)
        yshift = np.zeros(numImages, dtype=float)

        epochCount = np.zeros(numStars, dtype=int)

        for ii in range(numImages):
            alignRoot = self.alignDir + 'master_boot_' + str(ii)
            alignRoot += '_' + self.fields[ii]
            starList = self.images[ii] + '_rms_dist.lis'

            # Load the starset
            s = starset.StarSet(alignRoot)
            epoch = s.years[-1]

            # Load the *.trans file and transformation parameters
            trans = objects.Transform()
            trans.loadFromAlignTrans(alignRoot, idx=1)

            # Trim down to only stars detected in the aligned epoch
            velCnt = s.getArray('velCnt')
            idx = np.where(velCnt == 2)[0]
            newStars = [s.stars[jj] for jj in idx]
            s.stars = newStars

            # Match by name in the reference master list
            refName = np.array(s.getArrayFromEpoch(0, 'name'))

            # Pull the data from the aligned epoch
            xpix = s.getArrayFromEpoch(1, 'xpix')
            ypix = s.getArrayFromEpoch(1, 'ypix')

            # centroid+distortion+residual errors from each field
            xpixerr_p = s.getArrayFromEpoch(1, 'xpixerr_p') 
            ypixerr_p = s.getArrayFromEpoch(1, 'ypixerr_p')
            
            # alignment errors
            xpixerr_a = s.getArrayFromEpoch(1, 'xpixerr_a') 
            ypixerr_a = s.getArrayFromEpoch(1, 'ypixerr_a')

            mag = s.getArrayFromEpoch(1, 'mag')
            flux = s.getArrayFromEpoch(1, 'fwhm')
            corr = s.getArrayFromEpoch(1, 'corr')
            snr = s.getArrayFromEpoch(1, 'snr')

            magerr = 1.0857 / snr

            xshift[ii] = trans.a[0]
            yshift[ii] = trans.b[0]

            for ss in range(numStars):
                idx = np.where(refName == m_name[ss])[0]

                if len(idx) > 0:
                    x[ii,ss] = xpix[idx[0]]
                    y[ii,ss] = ypix[idx[0]]
                    xe[ii,ss] = np.sqrt(xpixerr_p[idx[0]]**2 - self.addErrX**2)
                    ye[ii,ss] = np.sqrt(ypixerr_p[idx[0]]**2 - self.addErrY**2)

                    # Check for NAN and replace with a very small number
                    if xpixerr_p[idx[0]] < self.addErrX:
                        xe[ii,ss] = 0.001
                    if ypixerr_p[idx[0]] < self.addErrY:
                        ye[ii,ss] = 0.001

                    if np.isnan(xe[ii, ss]) or np.isnan(ye[ii, ss]):
                        print 'Problem with ii=%d, ss=%d (%s)' % \
                            (ii, ss, m_name[ss])
                        print '  xpixerr_p = %9.5f is less than %9.5f' % \
                            (xpixerr_p[idx[0]], self.addErrX)
                        print '  ypixerr_p = %9.5f is less than %9.5f' % \
                            (ypixerr_p[idx[0]], self.addErrY)
                    xed[ii,ss] = xpixerr_p[idx[0]] # includes distortion
                    yed[ii,ss] = ypixerr_p[idx[0]]
                    xea[ii,ss] = xpixerr_a[idx[0]]
                    yea[ii,ss] = ypixerr_a[idx[0]]
                    m[ii, ss] = mag[idx[0]]
                    me[ii, ss] = magerr[idx[0]]
                    f[ii, ss] = flux[idx[0]]
                    c[ii, ss] = corr[idx[0]]

                    epochCount[ss] += 1

        # Trim out stragglers that somehow weren't re-matched
        idx = np.where(epochCount > 0)[0]
        m_name = m_name[idx]
        x = x[:, idx]
        y = y[:, idx]
        xe = xe[:, idx]
        ye = ye[:, idx]
        xed = xed[:, idx]
        yed = yed[:, idx]
        xea = xea[:, idx]
        yea = yea[:, idx]
        m = m[:, idx]
        me = me[:, idx]
        f = f[:, idx]
        c = c[:, idx]

        epochCount = epochCount[idx]
        numStars = len(idx)

        # All the data are loaded, now lets calculate some average values.
        # Remember that values of 0 indicate no data at that epoch.
        x_avg = np.zeros(numStars, dtype=float)
        y_avg = np.zeros(numStars, dtype=float)

        xe_avg = np.zeros(numStars, dtype=float) # no distortion errors
        ye_avg = np.zeros(numStars, dtype=float)

        xed_avg = np.zeros(numStars, dtype=float) # with distortion errors
        yed_avg = np.zeros(numStars, dtype=float)

        m_avg = np.zeros(numStars, dtype=float)
        me_avg = np.zeros(numStars, dtype=float)

        f_avg = np.zeros(numStars, dtype=float)

        c_avg = np.zeros(numStars, dtype=float)

        outRoot = self.tableDir + 'mag' + self.epoch + '_' + self.mosaic
        outRoot += '_' + self.filter

        _images = open(outRoot + '.images', 'w')
        _shifts = open(outRoot + '.shifts', 'w')
        _avg = open(outRoot + '_rms.lis', 'w')  # mosaicked starlist
        _avg_ep = open(outRoot + '_rms_ep.lis', 'w')  # with ep#_# names
        _avg_dist = open(outRoot + '_rms_dist.lis', 'w')  # with distortion

        _x = open(outRoot + '.x', 'w')
        _y = open(outRoot + '.y', 'w')

        # centroid+align+distortion+residual errors
        _xed = open(outRoot + '.xerr_dist', 'w') 
        _yed = open(outRoot + '.yerr_dist', 'w')

        # centroid+align errors
        _xe = open(outRoot + '.xerr', 'w') 
        _ye = open(outRoot + '.yerr', 'w')

        _m = open(outRoot + '.mag', 'w')
        _me = open(outRoot + '.magerr', 'w')
        _f = open(outRoot + '.flux', 'w')
        _c = open(outRoot + '.corr', 'w')

        for ii in range(len(self.images)):
            _images.write('%s\n' % self.images[ii])
            _shifts.write('%s  %10.4f  %10.4f\n' % 
                          (self.images[ii], xshift[ii], yshift[ii]))

        _images.close()
        _shifts.close()

        for ss in range(numStars):
            # Position data
            idx = np.where(x[:,ss] != 0)[0]

            # Add alignment errors to this star's other errors, in quadrature
            xe[:,ss] = np.sqrt(xe[:,ss]**2 + xea[:,ss]**2)
            ye[:,ss] = np.sqrt(ye[:,ss]**2 + yea[:,ss]**2)
            xed[:,ss] = np.sqrt(xed[:,ss]**2 + xea[:,ss]**2)
            yed[:,ss] = np.sqrt(yed[:,ss]**2 + yea[:,ss]**2)

            # For positions, get weighted average if a star was found more
            # than once (in an overlapping region). Weight by combo of centroid,
            # distortion, residual distortion and alignment errors
            x_avg[ss], xe_avg[ss] = getAveragePosition(x[:,ss], xe[:,ss], idx) 
            y_avg[ss], ye_avg[ss] = getAveragePosition(y[:,ss], ye[:,ss], idx)

            # For positional errors, we'll just take the average of the errors
            # in each dither position (so not using the xe_avg above)
            # remember to exclude the zeros in the array!
            # Remember -- errs include centroid, align, 
            #             distortion, & residual distortion
            xed_avg[ss] = xed[np.where(xed[:,ss]!=0)[0],ss].mean() 
            yed_avg[ss] = yed[np.where(yed[:,ss]!=0)[0],ss].mean() 

            xe_avg[ss] = xe[np.where(xe[:,ss]!=0)[0],ss].mean() 
            ye_avg[ss] = ye[np.where(ye[:,ss]!=0)[0],ss].mean() 

            # Brightness data
            idx = np.where(m[:,ss] != 0)[0]
            m_avg[ss], me_avg[ss] = getAverageMagnitude(m[:,ss], me[:,ss], idx)
            f_avg[ss] = getAverage(f[:, ss], idx)

            # Correlation data
            c_avg[ss] = getAverage(c[:, ss], idx)

            # Write everything out
            _x.write('%-13s' % m_name[ss])
            _y.write('%-13s' % m_name[ss])
            _xe.write('%-13s' % m_name[ss])
            _ye.write('%-13s' % m_name[ss])
            _xed.write('%-13s' % m_name[ss])
            _yed.write('%-13s' % m_name[ss])
            _m.write('%-13s' % m_name[ss])
            _me.write('%-13s' % m_name[ss])
            _f.write('%-13s' % m_name[ss])
            _c.write('%-13s' % m_name[ss])

            for ii in range(numImages):
                _x.write('  %9.4f' % x[ii, ss])
                _y.write('  %9.4f' % y[ii, ss])
                _xe.write('  %9.4f' % xe[ii, ss])
                _ye.write('  %9.4f' % ye[ii, ss])
                _xed.write('  %9.4f' % xed[ii, ss])
                _yed.write('  %9.4f' % yed[ii, ss])
                _m.write('  %8.3f' % m[ii, ss])
                _me.write('  %8.3f' % me[ii, ss])
                _f.write('  %14d' % f[ii, ss])
                _c.write('  %14d' % c[ii, ss])

            _x.write('\n')
            _y.write('\n')
            _xe.write('\n')
            _ye.write('\n')
            _xed.write('\n')
            _yed.write('\n')
            _m.write('\n')
            _me.write('\n')
            _f.write('\n')
            _c.write('\n')

            # Print output
            finalName = m_name[ss]
            if m_name[ss].startswith('ep'):
                finalName = 'star_%d' % ss

            _avg_ep.write('%-13s  %6.3f  %9.4f  %11.5f %11.5f  ' %
                       (m_name[ss], m_avg[ss], epoch, x_avg[ss], y_avg[ss]))
            _avg_ep.write('%9.5f %9.5f  %15.2f  %4.2f  %3d  %15.1f\n' %
                       (xe_avg[ss], ye_avg[ss], (1.0857/me_avg[ss]), c_avg[ss],
                        epochCount[ss], f_avg[ss]))

            _avg.write('%-13s  %6.3f  %9.4f  %11.5f %11.5f  ' %
                       (finalName, m_avg[ss], epoch, x_avg[ss], y_avg[ss]))
            _avg.write('%9.5f %9.5f  %15.2f  %4.2f  %3d  %15.1f\n' %
                       (xe_avg[ss], ye_avg[ss], (1.0857/me_avg[ss]), c_avg[ss],
                        epochCount[ss], f_avg[ss]))

            _avg_dist.write('%-13s  %6.3f  %9.4f  %11.5f %11.5f  ' %
                            (finalName, m_avg[ss], epoch, 
                             x_avg[ss], y_avg[ss]))
            _avg_dist.write('%9.5f %9.5f  %15.2f  %4.2f  %3d  %15.1f\n' %
                            (xed_avg[ss], yed_avg[ss], 
                             (1.0857/me_avg[ss]), c_avg[ss],
                             epochCount[ss], f_avg[ss]))


        _avg.close()
        _avg_dist.close()
        _avg_ep.close()
        _x.close()
        _y.close()
        _xe.close()
        _ye.close()
        _xed.close()
        _yed.close()
        _m.close()
        _me.close()
        _f.close()
        _c.close()

        outRootFinal = self.dataDir + 'mag' + self.epoch + '_' + self.mosaic
        outRootFinal += '_' + self.filter
        
        shutil.copyfile(outRoot + '_rms.lis', outRootFinal + '_rms.lis')
        shutil.copyfile(outRoot + '_rms_dist.lis', outRootFinal + '_rms_dist.lis')

    def makeNamedStarlist(self):
        """
        We would like to make a named starlist. We do this with a trick...
        just align a dummy starlist to our *rms.lis and pass in 
        label.dat to align.
        """
        mosaicRoot = 'mag' + self.epoch +'_'+ self.mosaic +'_'+ self.filter

        starlist = self.tableDir + mosaicRoot + '_rms_dist.lis'
        dumblist = self.lisDir + self.images[0] + '_rms_dist.lis' # This is shorter

        newList1 = self.lisDir + mosaicRoot + '_align_label.lis'
        newList2 = self.lisDir + self.images[0] + '_align_label.lis'

        # Match sources at the top
        shiftToTopOfList(starlist, newList1, self.refStars[self.fields[0]])
        shiftToTopOfList(dumblist, newList2, self.refStars[self.fields[0]])

        # Get the align data type
        fitsFile = self.dataDir + '../%s.fits' % (self.images[0])
        alignType = dataUtil.get_align_type(fitsFile, errors=True)

        alignRoot = self.alignDir + 'align_label'

        _list = open(alignRoot + '.list', 'w')
        _list.write('%s %d ref\n' % (newList1, alignType))
        _list.write('%s %d\n' % (newList2, alignType))
        _list.close()

        cmd = 'java align -v -p -a 2 -R 1 '
        cmd += '-N /g/lu/data/orion/source_list/label.dat '
        cmd += '-r %s %s.list' % (alignRoot, alignRoot)
        print cmd
        os.system(cmd)

        # This will be our new starlist
        list = starTables.StarfinderList(None, hasErrors=True)

        # Make a new starlist out of the resulting align output
        s = starset.StarSet(alignRoot)

        # Trim out all stars only detected in the dummy epoch.
        mag = s.getArrayFromEpoch(0, 'mag')
        idx = np.where(mag != 0)[0]
        newStars = []
        for ii in idx:
            newStars.append(s.stars[ii])
        s.stars = newStars

        # Pull the data from the aligned epoch
        list.name = np.array(s.getArray('name'))
        list.epoch = np.zeros(len(s.stars), dtype=float) + s.years[0]
        list.mag = s.getArrayFromEpoch(0, 'mag')
        list.x = s.getArrayFromEpoch(0, 'xpix')
        list.y = s.getArrayFromEpoch(0, 'ypix')
        list.xerr = s.getArrayFromEpoch(0, 'xpixerr_p')
        list.yerr = s.getArrayFromEpoch(0, 'ypixerr_p')
        list.snr = s.getArrayFromEpoch(0, 'snr')
        list.corr = s.getArrayFromEpoch(0, 'corr')
        list.nframes = s.getArrayFromEpoch(0, 'nframes')
        list.counts = s.getArrayFromEpoch(0, 'fwhm')

        list.saveToFile(starlist.replace('rms_dist', 'rms_named'))

        shutil.copyfile(self.tableDir + mosaicRoot + '_rms_named.lis', 
                        self.dataDir + mosaicRoot +  '_rms_named.lis')


    def convertToAbsolute(self):
        """
        Make a sham starlist in arcseconds with +x to the west. This is
        for Tuan's data analysis.
        """
        mosaicRoot = 'mag' + self.epoch +'_'+ self.mosaic +'_'+ self.filter
        starlist = self.tableDir + mosaicRoot + '_rms_named.lis'

        lis = starTables.StarfinderList(starlist, hasErrors=True)
        labels = starTables.Labels()

        # Convert the coordinates to arcseconds
        xpix = lis.x
        ypix = lis.y
        xpixerr = lis.xerr
        ypixerr = lis.yerr

        # Find 16C in both lists
        lis16C = np.where(lis.name == 'irs16C')
        lab16C = np.where(labels.name == 'irs16C')

        scale = 0.00995
        x = (xpix - xpix[lis16C]) * scale * -1.0
        x += labels.x[lab16C]
        x *= -1.0

        y = (ypix - ypix[lis16C]) * scale
        y += labels.y[lab16C]

        xe = xpixerr * scale
        ye = ypixerr * scale

        lis.x = x
        lis.y = y
        lis.xerr = xe
        lis.yerr = ye

        lis.saveToFile(starlist.replace('.lis', '_abs_xwest.lis'))

        shutil.copyfile(self.tableDir + mosaicRoot + '_rms_named_abs_xwest.lis', 
                        self.dataDir + mosaicRoot +  '_rms_named_abs_xwest.lis')

    def name_new_stars(self, oldNames):
        mosaicRoot = 'mag' + self.epoch +'_'+ self.mosaic +'_'+ self.filter
        starlist = self.tableDir + mosaicRoot + '_rms_named_abs_xwest.lis'
        labelNewFile = self.tableDir+'/label_new.dat'

        labels = starTables.Labels(labelFile=labelNewFile)
        list = starTables.StarfinderList(starlist, hasErrors=True)

        # Go through the existing named sources and figure out the highest
        # star index number.
        highestIndex = {}
        for ii in range(len(labels.name)):
            name = labels.name[ii]

            if name.startswith('S') and '-' in name:
                parts = name.split('-')

                radius = int(parts[0].replace('S', ''))
                index = int(parts[1])

                if highestIndex.has_key(radius):
                    if index > highestIndex[radius]:
                        highestIndex[radius] = index
                else:
                    highestIndex[radius] = index

        # Work with the starlist
        list.x *= -1.0
        list.r = np.hypot(list.x, list.y)

        for ii in range(len(oldNames)):
            idx = np.where(list.name == oldNames[ii])[0]

            rBin = int(math.floor( list.r[idx] ))

            highestIndex[rBin] += 1
            newName = 'S%d-%d' % (rBin, highestIndex[rBin])

            print '%-11s  %4.1f   %8.4f  %8.4f   %8.4f  %8.4f      0.000    0.000    0.000    0.000   %8.3f    0  %6.3f' % \
                (newName, list.mag[idx], list.x[idx], list.y[idx], 
                 list.xerr[idx], list.yerr[idx], list.epoch[idx], list.r[idx])

        

# ==============================
# Helper methods
# ==============================
def shiftToTopOfList(oldList, newList, starnames, withErr=True):
    print 'Shifting to top', starnames
    print '  old: %s' % (oldList)
    print '  new: %s' % (newList)

    # Star List
    _ref = starTables.StarfinderList(oldList, hasErrors=withErr)

    first = []
    rest = np.ones(len(_ref.name), dtype=bool)
    
    for ss in range(len(starnames)):
        idx = np.where(_ref.name == starnames[ss])[0]
        
        if len(idx) is 0:
            print 'Failed to find %s in %s' % (starnames[ss], oldList)
            print 'HERE2'
        first.append(idx[0])
            
        rest[idx[0]] = False

    rest = np.where(rest == True)[0]

    indices = np.append(first, rest)
    _ref.name = _ref.name[indices]
    _ref.mag = _ref.mag[indices]
    _ref.epoch = _ref.epoch[indices]
    _ref.x = _ref.x[indices]
    _ref.y = _ref.y[indices]
    _ref.snr = _ref.snr[indices]
    _ref.corr = _ref.corr[indices]
    _ref.nframes = _ref.nframes[indices]
    _ref.counts = _ref.counts[indices]

    if withErr:
        _ref.xerr = _ref.xerr[indices]
        _ref.yerr = _ref.yerr[indices]

    _ref.saveToFile(newList)


def makeNewRefList(alignRoot, refList, starPrefix='ep0'):
    print 'Making new reference list', refList
    s = starset.StarSet(alignRoot)

    # Get the zeropoint offset to convert to 
    magRef = s.getArrayFromEpoch(0, 'mag')
    magLis = s.getArrayFromEpoch(1, 'mag')

    idx = np.where((magRef != 0) & (magLis != 0))[0]
    # Only use the first 10 stars
    if len(idx) > 10:
        idx = idx[0:10]

    zp = (magRef[idx] - magLis[idx]).mean()
    print 'Zeropoint for %s is %.2f' % (starPrefix, zp)

    _out = open(refList, 'w')

    for star in s.stars:
        inRef = False
        inLis = False
        
        if star.e[0].x > -999: 
            inRef = True
        if star.e[1].x > -999: 
            inLis = True

        if inRef and inLis:

            name = star.e[0].name

            # Rename new stars from the mosaiced images.
            # Don't rename reference list sources, so we can reject 
            # them later on.
            if name.startswith('star') and starPrefix != None:
                name = star.e[1].name.replace('star', starPrefix)

            # get errors and combine them
            # centroid, distortion, and residual distortion
            xe0 = star.e[0].xpixerr_p 
            ye0 = star.e[0].ypixerr_p
            # centroid, distortion, and residual distortion
            xe1 = star.e[1].xpixerr_p 
            ye1 = star.e[1].ypixerr_p
            # align error
            xea = star.e[1].xpixerr_a 
            yea = star.e[1].ypixerr_a

            # add the alignment error in quadrature to the frame's other errors
            xe1_tot = np.hypot(xe1, xea)
            ye1_tot = np.hypot(ye1, yea)

            m = star.e[1].mag + zp

            # get original and aligned positions, then take weighted ave
            # weight by each field's errors (which contain centroid,
            # distortion, and residual distortion errors already)
            x0 = star.e[0].xpix
            y0 = star.e[0].ypix
            x1 = star.e[1].xpix
            y1 = star.e[1].ypix
            xwt0 = 1. / xe0**2
            ywt0 = 1. / ye0**2
            xwt1 = 1. / xe1_tot**2
            ywt1 = 1. / ye1_tot**2

            x = statsWeighted.mean_wgt(np.array([x0,x1]), np.array([xwt0,xwt1]))
            y = statsWeighted.mean_wgt(np.array([y0,y1]), np.array([ywt0,ywt1]))

            # For the next reference list's new positional error, we'll
            # take the average error of the ref and lis errors
            xerr = (xe0 + xe1_tot) / 2.0
            yerr = (ye0 + ye1_tot) / 2.0
        else:
            if inRef:
                name = star.e[0].name
                x = star.e[0].xpix
                y = star.e[0].ypix
                xe = star.e[0].xpixerr_p
                ye = star.e[0].ypixerr_p
                m = star.e[0].mag
                # just maintain the frame's (centroid+distortion+residual) 
                # errors
                xerr = xe 
                yerr = ye

            if inLis:
                name = star.e[1].name

                # Rename new stars from the narrow camera stuff.
                # Don't rename wide camera sources, so we can reject 
                # them later on.
                if name.startswith('star') and starPrefix != None:
                    name = name.replace('star', starPrefix)

                x = star.e[1].xpix
                y = star.e[1].ypix
                xe = star.e[1].xpixerr_p
                ye = star.e[1].ypixerr_p
                xea = star.e[1].xpixerr_a
                yea = star.e[1].ypixerr_a
                m = star.e[1].mag + zp
                # just maintain the frame's (centroid+distortion+residual) 
                # errors.
                xerr = xe 
                yerr = ye


        date = star.years[0]
        
        _out.write('%-13s %6.3f  %9.4f  %11.5f %11.5f  %8.5f %8.5f  1 1 1 1\n' %
                      (name, m, date, x, y, xerr, yerr))

    _out.close()


def getAveragePosition(p, pe, idx):
    if (len(idx) > 1):
        p_good = p[idx]
        p_weight = 1.0 / pe[idx]**2

        # Handle a rare case where pe == 0 due to not enough decimals in
        # our files.
        zeros = np.where(pe[idx] == 0)[0]
        non_zeros = np.where(pe[idx] != 0)[0]
        if len(zeros) > 0:
            min_err = pe[idx[non_zeros]].min()
            p_weight[zeros] = 1.0 / (min_err * 0.5)


        # for positions, get the weighted average
        # for errors, get the straight average
        p_avg = statsWeighted.mean_wgt(p_good, p_weight)
        #pe_avg = statsWeighted.std_wgt(p_good, p_weight)
        pe_avg = pe[idx].mean()
        
        #  -------------------
        # JLu had this next part originally, but we are already using
        # the average error, so not needed...for now

        ## Use the input magnitude errors if they are larger than
        ## the spread in the magnitude measurements. This is because
        ## we have so few measurements for some stars, there is a 
        ## reasonable chance that the photometric errors are underestimated.
        ## This is the conservative thing to do I think.
        #poserr_avg = pe[idx].mean()
        #if poserr_avg > pe_avg:
        #    pe_avg = poserr_avg
        #  -------------------


    if len(idx) == 1:
        p_avg = p[idx[0]]
        pe_avg = pe[idx[0]]

    if len(idx) == 0:
        p_avg = 0.0
        pe_avg = 0.0

    return p_avg, pe_avg

def getAverageMagnitude(m, me, idx):
    if len(idx) > 1:
        m_good = m[idx]
        m_weight = 1.0 / me[idx]**2

        flux_tmp = 10**(m_good/-2.5)
        flux_avg = statsWeighted.mean_wgt(flux_tmp, m_weight)
        flux_std = statsWeighted.std_wgt(flux_tmp, m_weight)

        m_avg = -2.5 * math.log10(flux_avg)
        me_avg = 2.5 * math.log10(math.e) * flux_std / flux_avg

        # Use the input magnitude errors if they are larger than
        # the spread in the magnitude measurements. This is because
        # we have so few measurements for some stars, there is a 
        # reasonable change that the photometric errors are underestimated.
        # This is the conservative thing to do I think.
        magerr_avg = me[idx].mean()
        if magerr_avg > me_avg:
            me_avg = magerr_avg
        
    if len(idx) == 1:
        m_avg = m[idx[0]]
        me_avg = me[idx[0]]

    if len(idx) == 0:
        m_avg = 0.0
        me_avg = 0.0

    return m_avg, me_avg

def getAverage(c, idx):
    if len(idx) > 1:
        c_avg = c[idx].mean()

    if len(idx) == 1:
        c_avg = c[idx[0]]

    if len(idx) == 0:
        c_avg = 0.0

    return c_avg



class MosaicMasers(Mosaic):
    defaultRefStars = {'C': ['16C', '16NW'],
                       'C_SW': ['33E', '33N'],
                       'C_NE': ['16C', '16NW'],
                       'C_SE': ['33E', '33N'],
                       'C_NW': ['16C', '16NW'],
                       'E': ['1NE', '1SE'],
                       'W': ['16C', '16NW'],
                       'N': ['10EE', '10W'],
                       'S': ['33E', '1SE'],
                       'NE': ['10EE', 'S12-1'],
                       'SE': ['1SE', 'S13-1'],
                       'NW': ['S7-9', 'S9-5'],
                       'SW': ['33N', '33E']
                       }


    defaultFields = ['C', 'E', 'W', 'N', 'S', 'NE', 'SE', 'NW', 'SW']
    
    def __init__(self, epoch, fieldsToMosaic=None, firstRefList=0,
                 rootDir='/g/lu/data/orion/', epochDirSuffix=None):
        """
        See help on Mosaic.

        Default fields and reference stars are specified. Override fields
        here in init. Override reference star dictionary by setting
        
            msc.refStars = refStarDict
        """
        if fieldsToMosaic is None:
            fieldsToMosaic = self.__class__.defaultFields

        if epoch == '12maylgs':
            self.__class__.defaultRefStars['S'] = ['33N', '1SE']
        
        Mosaic.__init__(self, epoch, 'msr', 'kp', 
                 fieldsToMosaic, self.__class__.defaultRefStars, 
                 firstRefList=firstRefList, 
                 rootDir='/g/lu/data/orion/', 
                 epochDirSuffix=epochDirSuffix)


class MosaicDeep(Mosaic):
    # Specify two stars per image that are in common with any of the prior images. 
    # These two stars will be shifted the top of modified starlists before
    # aligning. Remember if you change the order of the images, you will
    # need to change the order of the stars.

    defaultRefStars = {'C': ['irs16C', 'irs16NW'],
                       'C_SW': ['irs33E', 'irs33N'],
                       'C_NE': ['irs16C', 'irs16NW'],
                       'C_SE': ['irs33E', 'irs33N'],
                       'C_NW': ['irs16C', 'irs16NW'],
                       'E': ['irs1NE', 'S6-81'],
                       'W': ['irs2', 'S5-69'],
                       'N': ['S10-3', 'S12-4'],
                       'S': ['irs12N', 'irs14NE'],
                       'NE': ['irs10EE', 'S6-89'],
                       'SE': ['S8-8', 'S9-9'],
                       'NW': ['S8-3', 'S7-9'],
                       'SW': ['S9-114', 'S9-121']
                       }


    defaultFields = ['C', 'C_SW', 'C_NE', 'C_SE', 'C_NW', 
                     'E', 'W', 'N', 'S', 'NE', 'SE', 'NW', 'SW']
    
    def __init__(self, epoch, fieldsToMosaic=None,
                 firstRefList='/g/lu/data/orion/source_list/label.dat',
                 rootDir='/g/lu/data/orion/', epochDirSuffix=None):
        """
        See help on Mosaic.

        Default fields and reference stars are specified. Override fields
        here in init. Override reference star dictionary by setting
        
            msc.refStars = refStarDict
        """
        if fieldsToMosaic is None:
            fieldsToMosaic = self.__class__.defaultFields

        Mosaic.__init__(self, epoch, 'dp_msc', 'kp', 
                        fieldsToMosaic, self.__class__.defaultRefStars, 
                        firstRefList=firstRefList, 
                        rootDir='/g/lu/data/orion/', 
                        epochDirSuffix=epochDirSuffix,
                        alignOrder=4)

class MosaicOrionWide(Mosaic):
    # Specify two stars per image that are in common with any of the prior images. 
    # These two stars will be shifted the top of modified starlists before
    # aligning. Remember if you change the order of the images, you will
    # need to change the order of the stars.

    defaultRefStars = {'A': ['ir1491', 'Pare1839'],
                       'B': ['ir1491', 'Pare1839'],
                       'C': ['ir1491', 'Pare1839'],
                       'D': ['ir1491', 'Pare1839'],
                       'E': ['ir1491', 'Pare1839'],
                       'F': ['toller_11', 'toller_13'],
                       'G': ['ir1491', 'Pare1839'],
                       'H': ['IRn', 'Pare1839'],
                       'I': ['IRn', 'ir1374'],
                       'J': ['ir1491', 'Pare1839'],
                       'K': ['ir1491', 'Pare1839'],
                       }


    defaultFields = ['a', 'b', 'c', 'd', 'e', 
                     'f', 'g', 'h', 'i']
    

    def __init__(self, epoch, fieldsToMosaic=None,
                 firstRefList='/g/lu/data/orion/source_list/label.dat',
                 rootDir='/g/lu/data/orion/', epochDirSuffix=None,
                 alignOrder=2):
        """
        See help on Mosaic.

        Default fields and reference stars are specified. Override fields
        here in init. Override reference star dictionary by setting
        
            msc.refStars = refStarDict
        """
        if fieldsToMosaic is None:
            fieldsToMosaic = self.__class__.defaultFields
       
        Mosaic.__init__(self, epoch, 'BN_Mosaic', 'kp', 
                        fieldsToMosaic, self.__class__.defaultRefStars, 
                        firstRefList=firstRefList, 
                        rootDir='/g/lu/data/orion/', 
                        epochDirSuffix=epochDirSuffix,
                        scale=0.04, alignOrder=alignOrder)
