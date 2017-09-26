from nirc2.reduce import analysis

#-----------------------------------------------------#
# Analysis class specifically for the Arches cluster.
# Copied over from gcreduce --> gcanalysis.py
# 12/2/15, Matt Hosek
#-----------------------------------------------------#
        
class Arches(analysis.Analysis):
    def __init__(self, epoch, field, filt, rootDir='/g/lu/data/arches/nirc2/', 
                 epochDirSuffix=None, useDistorted=False, cleanList='c.lis'):
        """
        For Arches reduction:

        epoch -- '06maylgs' for example
        field -- 'f1', 'f2', etc. This sets the PSF star list to use.
        filt -- 'kp', 'lp', or 'h'

        Path to psf starlist, calFile, and labellist is hardcoded
        """
        # Initialize the Analysis object
        Analysis.__init__(self, epoch, filt=field + '_' + filt,
                          rootDir=rootDir, epochDirSuffix=epochDirSuffix,
                          useDistorted=useDistorted, cleanList=cleanList)

        # Setup some Arches specific parameters
        self.mapFilter2Cal = {'kp': 1, 'h': 2, 'lp': 3, 'J':4}
        self.mapField2CooStar = {'arch_f1': 'f1_psf0',
                                 'arch_f2': 'f2_psf8',
                                 'arch_f3': 'f3_psf0',
                                  'arch_f4': 'f4_psf0',
                                 'arch_f5': 'f5_psf1',
                                 'arch_f6': 'f6_psf7'}
        self.mapField2CalStars = {'arch_f1': ['f1_psf0', 'f1_psf1', 'f1_psf2', 'f1_psf3', 'f1_psf4', 'f1_psf5', 'f1_psf6', 'f1_psf7','f1_psf9'],
                                  'arch_f2': ['f2_psf0', 'f2_psf2', 'f2_psf3', 'f2_psf4', 'f2_psf6', 'f2_psf7', 'f2_psf8', 'f2_psf9', 'f2_psf10', 'f2_psf11', 'f2_psf12', 'f2_psf13'],
                                  'arch_f3': ['f3_psf0', 'f3_psf1', 'f3_psf5'],
                                  'arch_f4': ['f4_psf0', 'f4_psf2'],
                                  'arch_f5': ['f5_psf0', 'f5_psf1', 'f5_psf2', 'f5_psf3', 'f5_psf4', 'f5_psf5', 'f5_psf12'],
                                  'arch_f6': ['f6_psf2', 'f6_psf3'],
                                  }
                             

        # Use the field to set the psf starlist
        self.starlist = '/g/lu/code/idl/ucla_idl/'
        self.starlist += 'arches/psfstars/%s_%s_psf_all.list' % \
            (field, filt)

        ##########
        # Setup the appropriate calibration stuff.
        ##########
        # By default use the first 10 PSF stars.
        self.calStars = self.mapField2CalStars[field]

        # Choose the column based on the filter
        self.calColumn = self.mapFilter2Cal[filt]

        # Set the coo star
        self.cooStar = self.mapField2CooStar[field]
        self.calCooStar = self.cooStar

        # Override some of the default parameters
#         self.calFlags = '-f 1 -c 4 -R '
        self.calFlags = '-f 1 -R '
        self.calFile = '/g/lu/data/gc/source_list/photo_calib_arch.dat'
        
        self.labellist = '/g/lu/data/gc/source_list/label_arch.dat'
        self.orbitlist = None

        
