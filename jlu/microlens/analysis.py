from nirc2.reduce import analysis

class OB140613(analysis.Analysis):
    def __init__(self, epoch, filt, rootDir='/u/jlu/data/microlens/',
                 epochDirSuffix=None, cleanList='c.lis', alignMagCut=' -m 20 '):
        # Setup some W51a specific parameters
        self.mapFilter2Cal = {'kp': 1, 'h': 2, 'j': 3}

        filt_field = 'ob140613_' + filt

        # Initialize the Analysis object
        analysis.Analysis.__init__(self, epoch, filt=filt_field,
                                     rootDir=rootDir,
                                     epochDirSuffix=epochDirSuffix,
                                     cleanList=cleanList)

        # Use the field to set the psf starlist
        self.starlist = rootDir + 'source_list/ob140613_psf.list'

        ##########
        # Setup the appropriate calibration stuff.
        ##########
        # Use the default stars
        self.calStars = None

        # Choose the column based on the filter
        self.calColumn = self.mapFilter2Cal[filt]

        # Set the coo star
        self.cooStar = 'S20_11_5.3'
        self.calCooStar = self.cooStar

        # Override some of the default parameters
        self.calFlags = '-f 1 -R -s 1 '
        self.calFile = rootDir + 'source_list/ob140613_photo.dat'

        self.labellist = rootDir + 'source_list/ob140613_label.dat'
        self.orbitlist = None

        # Fix align flags for all the W51 fields. Otherwise, align is using too many faint stars.
        self.alignFlags = '-R 3 -v -p -a 2 ' + alignMagCut

        self.plotPosMagCut = 17.0

        return


class OB150211(analysis.Analysis):
    def __init__(self, epoch, filt, rootDir='/u/jlu/data/microlens/',
                 epochDirSuffix=None, cleanList='c.lis', alignMagCut=' -m 20 '):
        # Setup some W51a specific parameters
        self.mapFilter2Cal = {'kp': 1, 'h': 2, 'j': 3}

        filt_field = 'ob150211_' + filt

        # Initialize the Analysis object
        analysis.Analysis.__init__(self, epoch, filt=filt_field,
                                     rootDir=rootDir,
                                     epochDirSuffix=epochDirSuffix,
                                     cleanList=cleanList)

        # Use the field to set the psf starlist
        self.starlist = rootDir + 'source_list/ob150211_psf.list'

        ##########
        # Setup the appropriate calibration stuff.
        ##########
        # Use the default stars
        self.calStars = None

        # Choose the column based on the filter
        self.calColumn = self.mapFilter2Cal[filt]

        # Set the coo star
        self.cooStar = 'ob150211'
        self.calCooStar = self.cooStar

        # Override some of the default parameters
        self.calFlags = '-f 1 -R -s 1 '
        self.calFile = rootDir + 'source_list/ob150211_photo.dat'

        self.labellist = rootDir + 'source_list/ob150211_label.dat'
        self.orbitlist = None

        # Fix align flags for all the W51 fields. Otherwise, align is using too many faint stars.
        self.alignFlags = '-R 3 -v -p -a 2 ' + alignMagCut

        self.plotPosMagCut = 17.0

        return


class OB150029(analysis.Analysis):
    def __init__(self, epoch, filt, rootDir='/u/jlu/data/microlens/',
                 epochDirSuffix=None, cleanList='c.lis', alignMagCut=' -m 20 '):
        # Setup some W51a specific parameters
        self.mapFilter2Cal = {'kp': 1, 'h': 2, 'j': 3}

        filt_field = 'ob150029_' + filt

        # Initialize the Analysis object
        analysis.Analysis.__init__(self, epoch, filt=filt_field,
                                     rootDir=rootDir,
                                     epochDirSuffix=epochDirSuffix,
                                     cleanList=cleanList)

        # Use the field to set the psf starlist
        self.starlist = rootDir + 'source_list/ob150029_psf.list'

        ##########
        # Setup the appropriate calibration stuff.
        ##########
        # Use the default stars
        self.calStars = None

        # Choose the column based on the filter
        self.calColumn = self.mapFilter2Cal[filt]

        # Set the coo star
        self.cooStar = 'ob150029'
        self.calCooStar = self.cooStar

        # Override some of the default parameters
        self.calFlags = '-f 1 -R -s 1 '
        self.calFile = rootDir + 'source_list/ob150029_photo.dat'

        self.labellist = rootDir + 'source_list/ob150029_label.dat'
        self.orbitlist = None

        # Fix align flags for all the W51 fields. Otherwise, align is using too many faint stars.
        self.alignFlags = '-R 3 -v -p -a 2 ' + alignMagCut

        self.plotPosMagCut = 17.0

        return

class OB110061(analysis.Analysis):
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
        analysis.Analysis.__init__(self, epoch, filt=filt_field,
                                     rootDir=rootDir,
                                     epochDirSuffix=epochDirSuffix,
                                     cleanList=cleanList)

        # Use the field to set the psf starlist
        self.starlist = rootDir + 'source_list/ob110061_psf.list'

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


class OB110022(analysis.Analysis):
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
        analysis.Analysis.__init__(self, epoch, filt=filt_field,
                                     rootDir=rootDir,
                                     epochDirSuffix=epochDirSuffix,
                                     cleanList=cleanList)

        # Use the field to set the psf starlist
        self.starlist = rootDir + 'source_list/ob110022_psf.list'

        ##########
        # Setup the appropriate calibration stuff.
        ##########
        # Use the default stars
        self.calStars = None

        # Choose the column based on the filter
        self.calColumn = self.mapFilter2Cal[filt]

        # Set the coo star
        self.cooStar = 'OB110022'
        self.calCooStar = self.cooStar

        # Override some of the default parameters
        self.calFlags = '-f 1 -R -s 1 '
        self.calFile = '/u/jlu/data/microlens/source_list/ob110022_photo.dat'

        self.labellist = '/u/jlu/data/microlens/source_list/ob110022_label.dat'
        self.orbitlist = None

        # Fix align flags for all the W51 fields.
        # Otherwise, align is using too many faint stars.
        self.alignFlags += ' -m 18 '


class OB110125(analysis.Analysis):
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
        analysis.Analysis.__init__(self, epoch, filt=filt_field,
                                     rootDir=rootDir,
                                     epochDirSuffix=epochDirSuffix,
                                     cleanList=cleanList)

        # Use the field to set the psf starlist
        self.starlist = rootDir + 'source_list/ob110125_psf.list'

        ##########
        # Setup the appropriate calibration stuff.
        ##########
        # Use the default stars
        self.calStars = None

        # Choose the column based on the filter
        self.calColumn = self.mapFilter2Cal[filt]

        # Set the coo star
        self.cooStar = 'OB110125'
        self.calCooStar = self.cooStar

        # Override some of the default parameters
        self.calFlags = '-f 1 -R -s 1 '
        self.calFile = '/u/jlu/data/microlens/source_list/ob110125_photo.dat'

        self.labellist = '/u/jlu/data/microlens/source_list/ob110125_label.dat'
        self.orbitlist = None

        # Fix align flags for all the W51 fields.
        # Otherwise, align is using too many faint stars.
        self.alignFlags += ' -m 18 '

class OB120169(analysis.Analysis):
    def __init__(self, epoch, filt, rootDir='/u/jlu/data/microlens/',
                 epochDirSuffix=None, cleanList='c.lis', alignMagCut=' -m 20 '):
        """
        For OB120169 reduction:

        epoch -- '11may' for example
        filt -- 'kp', 'lp', or 'h'
        """
        # Setup some W51a specific parameters
        self.mapFilter2Cal = {'kp': 1, 'h': 2, 'j': 3}

        filt_field = 'ob120169_' + filt

        # Initialize the Analysis object
        analysis.Analysis.__init__(self, epoch, filt=filt_field,
                                     rootDir=rootDir,
                                     epochDirSuffix=epochDirSuffix,
                                     cleanList=cleanList)

        # Use the field to set the psf starlist
        self.starlist = self.rootDir + 'source_list/ob120169_psf.list'

        # Set up some extra starfinder keywords to optimize PSF handling.
        self.stf_extra_args = ', psfSize=2.0, trimfake=0' # 2 arcsec
        self.corrMain = 0.7
        self.corrSub = 0.5

        ##########
        # Setup the appropriate calibration stuff.
        ##########
        # Use the default stars
        self.calStars = None

        # Choose the column based on the filter
        self.calColumn = self.mapFilter2Cal[filt]

        # Set the coo star
        self.cooStar = 'p000_16_3.6'
        self.calCooStar = self.cooStar

        # Override some of the default parameters
        self.calFlags = '-f 1 -R -s 1 '
        self.calFile = self.rootDir + 'source_list/ob120169_photo.dat'

        self.labellist = self.rootDir + 'source_list/ob120169_label.dat'
        self.orbitlist = None

        # Fix align flags. Otherwise, align is using too many faint stars.
        self.alignFlags = '-R 3 -v -p -a 2 ' + alignMagCut

        self.plotPosMagCut = 17.0

        return

class MB980006(analysis.Analysis): # Made using OB120169 as reference
    def __init__(self, epoch, filt, rootDir='/u/jlu/data/microlens/',
                 epochDirSuffix=None, cleanList='c.lis', alignMagCut=' -m 20 '):
        """
        For MB980006 reduction:

        epoch -- '11may' for example
        filt -- 'kp', 'lp', or 'h'
        """
        # Setup some W51a specific parameters
        self.mapFilter2Cal = {'kp': 1, 'h': 2, 'j': 3}

        filt_field = 'mb980006_' + filt

        # Initialize the Analysis object
        analysis.Analysis.__init__(self, epoch, filt=filt_field,
                                     rootDir=rootDir,
                                     epochDirSuffix=epochDirSuffix,
                                     cleanList=cleanList)

        # Use the field to set the psf starlist
        self.starlist = self.rootDir + 'source_list/mb980006_psf.list'

        ##########
        # Setup the appropriate calibration stuff.
        ##########
        # Use the default stars
        self.calStars = None

        # Choose the column based on the filter
        self.calColumn = self.mapFilter2Cal[filt]

        # Set the coo star
        self.cooStar = 'MB980006'
        self.calCooStar = self.cooStar

        # Override some of the default parameters
        self.calFlags = '-f 1 -R -s 1 '
        self.calFile = self.rootDir + 'source_list/mb980006_photo.dat'

        self.labellist = self.rootDir + 'source_list/mb980006_label.dat'
        self.orbitlist = None

        # Fix align flags. Otherwise, align is using too many faint stars.
        self.alignFlags = '-R 3 -v -p -a 2 ' + alignMagCut

        self.plotPosMagCut = 17.0

        return

class MB960005(analysis.Analysis): # Made using OB120169 as reference
    def __init__(self, epoch, filt, rootDir='/u/jlu/data/microlens/',
                 epochDirSuffix=None, cleanList='c.lis', alignMagCut=' -m 20 '):
        """
        For MB980006 reduction:

        epoch -- '11may' for example
        filt -- 'kp', 'lp', or 'h'
        """
        # Setup some W51a specific parameters
        self.mapFilter2Cal = {'kp': 1, 'h': 2, 'j': 3}

        filt_field = 'mb960005_' + filt

        # Initialize the Analysis object
        analysis.Analysis.__init__(self, epoch, filt=filt_field,
                                     rootDir=rootDir,
                                     epochDirSuffix=epochDirSuffix,
                                     cleanList=cleanList)

        # Use the field to set the psf starlist
        self.starlist = self.rootDir + 'source_list/mb960005_psf.list'

        ##########
        # Setup the appropriate calibration stuff.
        ##########
        # Use the default stars
        self.calStars = None

        # Choose the column based on the filter
        self.calColumn = self.mapFilter2Cal[filt]

        # Set the coo star
        self.cooStar = 'MB960005'
        self.calCooStar = self.cooStar

        # Override some of the default parameters
        self.calFlags = '-f 1 -R -s 1 '
        self.calFile = self.rootDir + 'source_list/mb960005_photo.dat'

        self.labellist = self.rootDir + 'source_list/mb960005_label.dat'
        self.orbitlist = None

        # Fix align flags. Otherwise, align is using too many faint stars.
        self.alignFlags = '-R 3 -v -p -a 2 ' + alignMagCut

        self.plotPosMagCut = 17.0

        return
