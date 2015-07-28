from nirc2.reduce.analysis import Analysis



class Speckle(Analysis):
    def __init__(self, epoch, rootDir='/u/ghezgroup/data/gc/'): 
        """
        For Speckle reduction:

        epoch -- '02apr' for example
        filt -- must use '' because the filt is not specified in image names
        """
        # Initialize the Analysis object
        Analysis.__init__(self, epoch, filt='', rootDir=rootDir, cleanList=None)


        # Set the psf starlist
        self.starlist = '/u/ghezgroup/code/idl/'
        self.starlist += 'gc/starfinder/psfstars/psf_central.dat' 

        # Override some of the default parameters
        self.type = 'speckle'
#         self.calFlags = '-f 1 -c 1 -R '
        self.calFlags = '-f 1 -R '



class DeepMosaic(Analysis):
    def __init__(self, epoch, filt, rootDir='/u/ghezgroup/data/gc/', 
                 epochDirSuffix=None, useDistorted=False, cleanList='c.lis'):
        """
        For reduction of deep wide mosaics:

        epoch -- '06maylgs1' for example
        filt -- 'dp_msc_C_kp', 'dp_msc_NE_kp', 'dp_msc_SE_kp', and so on...
        """

        # Create dictionary which maps the PSF stars to the field
        self.mapFilter2CalStars = {'dp_msc_C_kp': ['16C', '16NW', '16NE', '16CC', 'S2-16', '33N', 'S1-23', '33E', 'S2-22'],
                                  'dp_msc_NE_kp': ['10EE', '10W', 'b100', 'S12-1', 'S11-6', 'irs17'],
                                  'dp_msc_SE_kp': ['irs28', 'S12-2'],
                                  'dp_msc_NW_kp': ['S8-3', 'S7-9', 'S10-2', 'S12-8', 'S12-4'],
                                  'dp_msc_SW_kp': ['idSW2', 'S7-18'],
                                  'dp_msc_E_kp': ['1SE', '1NE', 'S10-1', 'S5-183', 'irs28'],
                                  'dp_msc_W_kp': ['S5-69', 'S7-18', 'irs2'],
                                  'dp_msc_N_kp': ['irs7', 'S9-5', '15NE', 'S10-3', 'S11-6', 'S12-4'],
                                  'dp_msc_S_kp': ['14NE', 'irs14SW', '12N'],
                                  'dp_msc_C_SW_kp': ['irs2', 'S7-18', '12N', 'S1-23', '33N', 'irs14SW'],
                                  'dp_msc_C_NE_kp': ['16C', '16NW', '16NE', '16CC', 'S2-22', '1NE', '1SE', '10W'],
                                  'dp_msc_C_SE_kp': ['33E', '33N', 'S5-183', '14NE'],
                                  'dp_msc_C_NW_kp': ['irs34W', 'S2-16', 'S5-69', 'S7-9', 'S8-3', '16NW', 'S1-23']
                                  }

        self.mapFilter2Coo = {'dp_msc_C_kp': 'irs16C',
                                  'dp_msc_NE_kp': 'S11-6',
                                  'dp_msc_SE_kp': 'S12-2',
                                  'dp_msc_NW_kp': 'S8-3',
                                  'dp_msc_SW_kp': 'S13-61',
                                  'dp_msc_E_kp': 'S10-1',
                                  'dp_msc_W_kp': 'S5-69',
                                  'dp_msc_N_kp': 'S10-3',
                                  'dp_msc_S_kp': 'irs14SW',
                                  'dp_msc_C_SW_kp': 'irs2',
                                  'dp_msc_C_NE_kp': 'irs16C',
                                  'dp_msc_C_SE_kp': 'irs33E',
                                  'dp_msc_C_NW_kp': 'irs34W'
                                  }
        self.mapFilter2CalCoo = {'dp_msc_C_kp': 'irs16C',
                                  'dp_msc_NE_kp': 'S11-6',
                                  'dp_msc_SE_kp': 'S12-2',
                                  'dp_msc_NW_kp': 'S8-3',
                                  'dp_msc_SW_kp': 'S13-61',
                                  'dp_msc_E_kp': 'S10-1',
                                  'dp_msc_W_kp': 'S5-69',
                                  'dp_msc_N_kp': 'S10-3',
                                  'dp_msc_S_kp': 'irs14SW',
                                  'dp_msc_C_SW_kp': 'irs2',
                                  'dp_msc_C_NE_kp': 'irs16C',
                                  'dp_msc_C_SE_kp': 'irs33E',
                                  'dp_msc_C_NW_kp': 'irs34W'
                                  }
                             

        # Initialize the Analysis object
        Analysis.__init__(self, epoch, filt=filt,
                          rootDir=rootDir, epochDirSuffix=epochDirSuffix,
                          useDistorted=useDistorted, cleanList=cleanList)

        ##########
        # Setup the appropriate calibration stuff.
        ##########
        self.calFile = '/u/ghezgroup/data/gc/source_list/'
        self.calFile += 'photo_calib_schoedel2010.dat'

        #self.calStars = self.mapFilter2CalStars[filt]
        self.calStars = None  # use defaults in photo_calib file

        # Choose the column based on the filter
        self.calColumn = 2

        # Set the coo star
        self.cooStar = self.mapFilter2Coo[filt]
        self.calCooStar = self.mapFilter2CalCoo[filt]

        # Set the psf starlist
        self.starlist = '/u/ghezgroup/code/idl/'
        self.starlist += 'gc/starfinder/psfstars/psf_mosaic.dat' 
        


class MaserMosaic(Analysis):
    def __init__(self, epoch, filt, rootDir='/u/ghezgroup/data/gc/', 
                 epochDirSuffix=None, useDistorted=False, cleanList='c.lis'):
        """
        For reduction of (shallow) maser mosaics:

        epoch -- '06maylgs1' for example
        filt -- 'msc_C_kp', 'msr_NE_kp', 'msr_SE_kp', and so on...
        """

        # Create dictionary which maps the PSF stars to the field
        self.mapFilter2CalStars = {'msr_C_kp': ['16C', '16NW', '16NE', '10W', '10EE', '1NE', '1SE', 'S2-22'],
                                  'msr_NE_kp': ['10EE', '10W', 'S12-1', 'irs17', 'b100'],
                                  'msr_SE_kp': ['irs28', '1SE', 'S13-1'],
                                  'msr_NW_kp': ['irs7', 'S9-5', 'S10-3', 'S12-4', 'S10-2', '15NE', '15SW'],
                                  'msr_SW_kp': ['33E', '33N', 'irs2', '14NE', 'S1-23'],
                                  'msr_E_kp': ['S12-1', 'S10-1', '1NE', '10EE', '1SE', 'S13-1'],
                                  'msr_W_kp': ['16C', '16NW', '16NE', '16CC', 'S2-16', 'S1-23', '33N'],
                                  'msr_N_kp': ['10W', '10EE', 'b100', 'S9-5'],
                                  'msr_S_kp': ['33E', '33N', '1SE', '14NE', 'S2-22']
                                  }

        self.mapFilter2Coo = {'msr_C_kp': ['irs16C'],
                                  'msr_NE_kp': ['irs10EE'],
                                  'msr_SE_kp': ['irs28'],
                                  'msr_NW_kp': ['irs7'],
                                  'msr_SW_kp': ['irs33N'],
                                  'msr_E_kp': ['irs10EE'],
                                  'msr_W_kp': ['irs16C'],
                                  'msr_N_kp': ['S9-5'],
                                  'msr_S_kp': ['irs33N']
                                  }
        self.mapFilter2CalCoo = {'msr_C_kp': ['16C'],
                                  'msr_NE_kp': ['10EE'],
                                  'msr_SE_kp': ['irs28'],
                                  'msr_NW_kp': ['irs7'],
                                  'msr_SW_kp': ['33N'],
                                  'msr_E_kp': ['10EE'],
                                  'msr_W_kp': ['16C'],
                                  'msr_N_kp': ['S9-5'],
                                  'msr_S_kp': ['33N']
                                  }
                             

        # Initialize the Analysis object
        Analysis.__init__(self, epoch, filt=filt,
                          rootDir=rootDir, epochDirSuffix=epochDirSuffix,
                          useDistorted=useDistorted, cleanList=cleanList)

        ##########
        # Setup the appropriate calibration stuff.
        ##########
        self.calStars = self.mapFilter2CalStars[filt]

        # Choose the column based on the filter
        self.calColumn = 2

        # Set the coo star
        self.cooStar = self.mapFilter2Coo[filt]
        self.calCooStar = self.mapFilter2CalCoo[filt]

        # Set the psf starlist
        self.starlist = '/u/ghezgroup/code/idl/'
        self.starlist += 'gc/starfinder/psfstars/psf_mosaic.dat'


class LpLargeMosaic(Analysis):
    def __init__(self, epoch, filt, rootDir='/u/ghezgroup/data/gc/', 
                 epochDirSuffix=None, useDistorted=False, cleanList='c.lis'):

    # Create dictionary which maps the PSF stars to the field:
        self.mapFilter2CalStars = {'h_lp': ['16C', '16NW', '16NE', '10W', '10EE', '1NE', '1SE', 'S2-22'],
                                  'f_lp': ['10EE', '10W', 'S12-1', 'irs17', 'b100'],
                                  'd_lp': ['irs28', '1SE', 'S13-1'],
                                  'l_lp': ['irs7', 'S9-5', 'S10-3', 'S12-4', 'S10-2', '15NE', '15SW'],
                                  'j_lp': ['33E', '33N', 'irs2', '14NE', 'S1-23'],
                                  'e_lp': ['S12-1', 'S10-1', '1NE', '10EE', '1SE', 'S13-1'],
                                  'k_lp': ['16C', '16NW', '16NE', '16CC', 'S2-16', 'S1-23', '33N'],
                                  'r_lp': ['10W', '10EE', 'b100', 'S9-5'],
                                  'i_lp': ['33E', '33N', '1SE', '14NE', 'S2-22'],
                                  'c_lp': ['irc1', 'irc2'],
                                  'b_lp': ['irb1', 'irb2', 'irb3'],
                                  'a_lp': ['ira1', 'ira2', 'ira3'],
                                  'p_lp': ['irp1', 'irp2', 'irp3'],
                                  'q_lp': ['irq1', 'irq2'],
                                  'r_lp': ['irr1', 'irr2'],
                                  's_lp': ['irsf1', 'irsf2', 'irsf3'],
                                  't_lp': ['irt1', 'irt2', 'irt3', 'irt4'],
                                  'm_lp': ['irm1', 'irm2', 'im3'],
                                  'n_lp': ['irn1'],
                                  'tot_o_lp': ['iro1', 'iro2', 'iro3', 'iro4']
                                  }

        self.mapFilter2Coo = {'a_lp': ['ira_coo'],
                                  'b_lp': ['irb_coo'],
                                  'c_lp': ['irc_coo'],
                                  'd_lp': ['irs28'],
                                  'e_lp': ['S10-1'],
                                  'f_lp': ['irs10EE'],
                                  'g_lp': ['irs15NE'],
                                  'h_lp': ['irs16C'],
                                  'i_lp': ['irs14NE'],
                                  'j_lp': ['S9-114'],
                                  'k_lp': ['S5-69'],
                                  'l_lp': ['S10-2'],
                                  'm_lp': ['irm_coo'],
                                  'n_lp': ['irn1'],
                                  'o_lp': ['iro_coo'],
                                  'p_lp': ['irp_coo'],
                                  'q_lp': ['irq_coo'],
                                  'r_lp': ['irr_coo'],
                                  's_lp': ['irsf2'],
                                  't_lp': ['irt_coo']
                                  }
        self.mapFilter2CalCoo = {'a_lp': ['ira_coo'],
                                  'b_lp': ['irb_coo'],
                                  'c_lp': ['irc_coo'],
                                  'd_lp': ['irs28'],
                                  'e_lp': ['S10-1'],
                                  'f_lp': ['irs10EE'],
                                  'g_lp': ['irs15NE'],
                                  'h_lp': ['irs16C'],
                                  'i_lp': ['irs14NE'],
                                  'j_lp': ['S9-114'],
                                  'k_lp': ['S5-69'],
                                  'l_lp': ['S10-2'],
                                  'm_lp': ['irm_coo'],
                                  'n_lp': ['irn1'],
                                  'o_lp': ['iro_coo'],
                                  'p_lp': ['irp_coo'],
                                  'q_lp': ['irq_coo'],
                                  'r_lp': ['irr_coo'],
                                  's_lp': ['irsf2'],
                                  't_lp': ['irt_coo']
                                  }
        
        # Initialize the Analysis object
        Analysis.__init__(self, epoch, filt=filt,
                          rootDir=rootDir, epochDirSuffix=epochDirSuffix,
                          useDistorted=useDistorted, cleanList=cleanList, imgSuffix = '2')

        ##########
        # Setup the appropriate calibration stuff.
        ##########
        self.calStars = self.mapFilter2CalStars[filt]

        # Choose the column based on the filter
        self.calColumn = 3

        # Set the coo star
        self.cooStar = self.mapFilter2Coo[filt]
        self.calCooStar = self.mapFilter2CalCoo[filt]

        self.cooStar = self.mapFilter2Coo[filt]
        self.calCooStar = self.mapFilter2CalCoo[filt]

        # Set the psf starlist
        self.starlist = '/u/ghezgroup/code/idl/'
        self.starlist += 'gc/starfinder/psfstars/psf_mosaic.dat'


       # pdb.set_trace()
