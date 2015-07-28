from nirc2.reduce.analysis import Analysis

class Quintuplet(Analysis):
    def __init__(self, epoch, field, filt, rootDir='/u/ghezgroup/data/gc/', 
                 epochDirSuffix=None, useDistorted=False, cleanList='c.lis'):
        """
        For Quintuplet reduction:

        epoch -- '06maylgs' for example
        field -- 'f1', etc. This sets the PSF star list to use.
        filt -- 'kp'
        """
        # Setup some Arches specific parameters
        self.mapFilter2Cal = {'kp': 1}
        self.mapField2CooStar = {'quint_f1': 'psf0'}

        # Initialize the Analysis object
        Analysis.__init__(self, epoch, filt=field + '_' + filt,
                          rootDir=rootDir, epochDirSuffix=epochDirSuffix,
                          useDistorted=useDistorted, cleanList=cleanList)

        # Use the field to set the psf starlist
        self.starlist = '/u/ghezgroup/code/idl/'
        self.starlist += 'quintuplet/psfstars/%s_%s_psf.list' % \
            (field, filt)

        # Setup the appropriate calibration stuff.
        # By default use the first 10 PSF stars.
        self.calStars = ['psf0', 'psf1', 'psf2', 'psf4']

        # Choose the column based on the filter
        self.calColumn = self.mapFilter2Cal[filt]

        # Set the coo star
        self.cooStar = self.mapField2CooStar[field]
        self.calCooStar = self.cooStar

        # Override some of the default parameters
#         self.calFlags = '-f 1 -c 4 -R '
        self.calFlags = '-f 1 -R '
        self.calFile = '/u/ghezgroup/data/gc/source_list/photo_calib_quint.dat'
        
        self.labellist = '/u/ghezgroup/data/gc/source_list/label_quint.dat'
        self.orbitlist = None
        
