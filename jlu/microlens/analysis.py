from gcreduce import gcanalysis

class OB110061(gcanalysis.Analysis):
    def __init__(self, epoch, filt, rootDir='/u/jlu/data/microlens/', 
                 epochDirSuffix=None, cleanList='c.lis'):
        """
        For OB110061 reduction:

        epoch -- '11may' for example
        filt -- 'kp', 'lp', or 'h'
        """
        # Setup some W51a specific parameters
        self.mapFilter2Cal = {'kp': 1, 'h': 2, 'j': 3}

        filt_field = 'ob110061_' + filt

        # Initialize the Analysis object
        gcanalysis.Analysis.__init__(self, epoch, filt=filt_field,
                                     rootDir=rootDir, 
                                     epochDirSuffix=epochDirSuffix,
                                     cleanList=cleanList)

        # Use the field to set the psf starlist
        self.starlist = '/u/jlu/code/idl/'
        self.starlist += 'microlens/psfstars/ob110061_psf.list'

        ##########
        # Setup the appropriate calibration stuff.
        ##########
        # Use the default stars
        self.calStars = None

        # Choose the column based on the filter
        self.calColumn = self.mapFilter2Cal[filt]

        # Set the coo star
        self.cooStar = 'psf_000'
        self.calCooStar = self.cooStar

        # Override some of the default parameters
        self.calFlags = '-f 1 -R -s 1 '
        self.calFile = '/u/jlu/data/microlens/source_list/ob110061_photo.dat'
        
        self.labellist = '/u/jlu/data/microlens/source_list/ob110061_label.dat'
        self.orbitlist = None

        # Fix align flags for all the W51 fields.
        # Otherwise, align is using too many faint stars.
        self.alignFlags += ' -m 18 '
        

class OB110022(gcanalysis.Analysis):
    def __init__(self, epoch, filt, rootDir='/u/jlu/data/microlens/', 
                 epochDirSuffix=None, cleanList='c.lis'):
        """
        For OB110022 reduction:

        epoch -- '11may' for example
        filt -- 'kp', 'lp', or 'h'
        """
        # Setup some W51a specific parameters
        self.mapFilter2Cal = {'kp': 1, 'h': 2, 'j': 3}

        filt_field = 'ob110022_' + filt

        # Initialize the Analysis object
        gcanalysis.Analysis.__init__(self, epoch, filt=filt_field,
                                     rootDir=rootDir, 
                                     epochDirSuffix=epochDirSuffix,
                                     cleanList=cleanList)

        # Use the field to set the psf starlist
        self.starlist = '/u/jlu/code/idl/'
        self.starlist += 'microlens/psfstars/ob110022_psf.list'

        ##########
        # Setup the appropriate calibration stuff.
        ##########
        # Use the default stars
        self.calStars = None

        # Choose the column based on the filter
        self.calColumn = self.mapFilter2Cal[filt]

        # Set the coo star
        self.cooStar = 'ob110022'
        self.calCooStar = self.cooStar

        # Override some of the default parameters
        self.calFlags = '-f 1 -R -s 1 '
        self.calFile = '/u/jlu/data/microlens/source_list/ob110022_photo.dat'
        
        self.labellist = '/u/jlu/data/microlens/source_list/ob110022_label.dat'
        self.orbitlist = None

        # Fix align flags for all the W51 fields.
        # Otherwise, align is using too many faint stars.
        self.alignFlags += ' -m 18 '
                

class OB110125(gcanalysis.Analysis):
    def __init__(self, epoch, filt, rootDir='/u/jlu/data/microlens/', 
                 epochDirSuffix=None, cleanList='c.lis'):
        # self.epoch='13jul'
#         self.filt='kp'
        """
        For OB110125 reduction:

        epoch -- '11may' for example
        filt -- 'kp', 'lp', or 'h'
        """
        # Setup some W51a specific parameters
        self.mapFilter2Cal = {'kp': 1, 'h': 2, 'j': 3}

        filt_field = 'ob110125_' + filt

        # Initialize the Analysis object
        gcanalysis.Analysis.__init__(self, epoch, filt=filt_field,
                                     rootDir=rootDir, 
                                     epochDirSuffix=epochDirSuffix,
                                     cleanList=cleanList)
        
        # Use the field to set the psf starlist
        self.starlist = '/u/jlu/code/idl/'
        self.starlist += 'microlens/psfstars/ob110125_psf.list'

        ##########
        # Setup the appropriate calibration stuff.
        ##########
        # Use the default stars
        self.calStars = None

        # Choose the column based on the filter
        self.calColumn = self.mapFilter2Cal[filt]

        # Set the coo star
        self.cooStar = 'ob110125'
        self.calCooStar = self.cooStar

        # Override some of the default parameters
        self.calFlags = '-f 1 -R -s 1 '
        self.calFile = '/u/jlu/data/microlens/source_list/ob110125_photo.dat'
        
        self.labellist = '/u/jlu/data/microlens/source_list/ob110125_label.dat'
        self.orbitlist = None

        # Fix align flags for all the W51 fields.
        # Otherwise, align is using too many faint stars.
        self.alignFlags += ' -m 18 '
        
class OB120169(gcanalysis.Analysis):
    def __init__(self, epoch, filt, rootDir='/u/jlu/data/microlens/', 
                 epochDirSuffix=None, cleanList='c.lis'):
        # self.epoch='13jul'
#         self.filt='kp'
        """
        For OB120169 reduction:

        epoch -- '11may' for example
        filt -- 'kp', 'lp', or 'h'
        """
        # Setup some W51a specific parameters
        self.mapFilter2Cal = {'kp': 1, 'h': 2, 'j': 3}

        filt_field = 'ob120169_' + filt

        # Initialize the Analysis object
        gcanalysis.Analysis.__init__(self, epoch, filt=filt_field,
                                     rootDir=rootDir, 
                                     epochDirSuffix=epochDirSuffix,
                                     cleanList=cleanList)
        
        # Use the field to set the psf starlist
        self.starlist = '/u/jlu/code/idl/'
        self.starlist += 'microlens/psfstars/ob120169_psf.list'

        ##########
        # Setup the appropriate calibration stuff.
        ##########
        # Use the default stars
        self.calStars = None

        # Choose the column based on the filter
        self.calColumn = self.mapFilter2Cal[filt]

        # Set the coo star
        self.cooStar = 'psf_000'
        self.calCooStar = self.cooStar

        # Override some of the default parameters
        self.calFlags = '-f 1 -R -s 1 '
        self.calFile = '/u/jlu/data/microlens/source_list/ob120169_photo.dat'
        
        self.labellist = '/u/jlu/data/microlens/source_list/ob120169_label.dat'
        self.orbitlist = None

        # Fix align flags for all the W51 fields.
        # Otherwise, align is using too many faint stars.
        self.alignFlags += ' -m 20 '
        
                