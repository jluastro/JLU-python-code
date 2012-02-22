from gcreduce import gcanalysis

class W51aWide(gcanalysis.Analysis):
    def __init__(self, epoch, field, filt, rootDir='/u/jlu/data/w51/', 
                 epochDirSuffix=None, cleanList='c.lis'):
        """
        For W51a reduction:

        epoch -- '09jun26' for example
        field -- 'f1', 'f2', etc. This sets the PSF star list to use.
        filt -- 'kp', 'lp', or 'h'
        """
        # Setup some W51a specific parameters
        self.mapFilter2Cal = {'kp': 5, 'h': 4, 'lp': 6}

        # Initialize the Analysis object
        gcanalysis.Analysis.__init__(self, epoch, filt=field + '_' + filt,
                                     rootDir=rootDir, 
                                     epochDirSuffix=epochDirSuffix,
                                     cleanList=cleanList)

        # Use the field to set the psf starlist
        self.starlist = '/u/jlu/code/idl/'
        self.starlist += 'w51/psfstars/w51a_wide_psf.list'

        ##########
        # Setup the appropriate calibration stuff.
        ##########
        # Use the default stars
        self.calStars = None

        # Choose the column based on the filter
        self.calColumn = self.mapFilter2Cal[filt]

        # Set the coo star
        self.cooStar = 'f3_psf3'
        self.calCooStar = self.cooStar

        # Override some of the default parameters
        self.calFlags = '-f 1 -R -s 1 '
        self.calFile = '/u/jlu/data/w51/source_list/w51a_photo.dat'
        
        self.labellist = '/u/jlu/data/w51/source_list/w51a_label.dat'
        self.orbitlist = None

        # Fix align flags for all the W51 fields.
        # Otherwise, align is using too many faint stars.
        self.alignFlags += ' -m 18 '
        

class W51a(gcanalysis.Analysis):
    def __init__(self, epoch, field, filt, rootDir='/u/jlu/data/w51/', 
                 epochDirSuffix=None, cleanList='c.lis'):
        """
        For W51a reduction:

        epoch -- '09jun26' for example
        field -- 'f1', 'f2', etc. This sets the PSF star list to use.
        filt -- 'kp', 'lp', or 'h'
        """
        # Setup some W51a specific parameters
        self.mapFilter2Cal = {'kp': 5, 'h': 4, 'lp': 6}

        self.mapField2CooStar = {'w51a_f1': 'f1_psf0',
                                 'w51a_f2': 'f2_psf0',
                                 'w51a_f3': 'f3_psf0',
                                 'w51a_f4': 'f4_psf0',
                                 'w51a_f2_n2': 'f2_psf0'}

        self.mapField2CalStars = {'w51a_f1': ['f1_psf0', 'f1_psf2', 'f1_psf3', 
                                              'f2_psf3', 'phot1', 'phot2', 
                                              'phot3', 'phot7', 'phot8', 
                                              'phot10', 'phot12', 'phot15', 
                                              'phot16', 'phot17', 'phot18', 
                                              'phot20', 'phot23', 'phot24', 
                                              'phot30', 'phot31', 'phot33'],
                                  'w51a_f2': ['f2_psf0', 'f2_psf1', 'f2_psf2',
                                              'f2_psf3', 'phot21', 'phot27',
                                              'phot28', 'phot30', 'phot32',
                                              'phot33'],
                                  'w51a_f3': ['f3_psf0', 'f3_psf1', 'f3_psf2',
                                              'f3_psf3', 'phot4', 'phot11',
                                              'phot13'],
                                  'w51a_f4': ['f4_psf0', 'f4_psf1', 'f4_psf2',
                                              'f4_psf3', 'phot6', 'phot9',
                                              'phot22', 'phot29', 'phot34', 
                                              'phot35']
                                  }
                             

        # Initialize the Analysis object
        gcanalysis.Analysis.__init__(self, epoch, filt=field + '_' + filt,
                                     rootDir=rootDir, 
                                     epochDirSuffix=epochDirSuffix,
                                     cleanList=cleanList)

        # Use the field to set the psf starlist
        self.starlist = '/u/jlu/code/idl/'
        self.starlist += 'w51/psfstars/%s_psf.list' % \
            (field)

        ##########
        # Setup the appropriate calibration stuff.
        ##########
        # Use the default stars
        if field in self.mapField2CalStars:
            self.calStars = self.mapField2CalStars[field]
        else:
            self.calStars = None

        # Choose the column based on the filter
        self.calColumn = self.mapFilter2Cal[filt]

        # Set the coo star
        self.cooStar = self.mapField2CooStar[field]
        self.calCooStar = self.cooStar

        # Override some of the default parameters
        self.calFlags = '-f 1 -R -s 1 '
        self.calFile = '/u/jlu/data/w51/source_list/w51a_photo.dat'
        
        self.labellist = '/u/jlu/data/w51/source_list/w51a_label.dat'
        self.orbitlist = None

        # Fix align flags for all the W51 fields.
        # Otherwise, align is using too many faint stars.
        self.alignFlags += ' -m 18 '
        
