from nirc2.reduce.analysis import Analysis

class OrionWideMosaic(Analysis):
    def __init__(self, epoch, filt, rootDir='/g/lu/data/orion/2010oct/', 
                 epochDirSuffix=None, useDistorted=False, cleanList='c.lis'):
        """
        For reduction of deep wide mosaics:

        epoch -- '06maylgs1' for example
        filt -- 'dp_msc_C_kp', 'dp_msc_NE_kp', 'dp_msc_SE_kp', and so on...
        """
        
        # Create dictionary which maps the PSF stars to the field
        self.mapFilter2CalStars = {'orion_msc_a_kp': ['IRn', 'Pare1839', 'ir1491', 'ir1374'],
                                  'orion_msc_b_kp': ['IRn', 'Pare1839', 'ir1491', 'ir1366'],
                                  'orion_msc_c_kp': ['IRn', 'Pare1839', 'ir1491', 'ir1366'],
                                  'orion_msc_d_kp': ['IRn', 'Pare1839', 'ir1491', 'ir1374', 'ir1366', 'ir1486'],
                                  'orion_msc_e_kp': ['IRn', 'Pare1839', 'ir1491', 'ir1374', 'ir1366'],
                                  'orion_msc_f_kp': ['IRn', 'Pare1839', 'ir1491', 'ir1374', 'ir1366'],
                                  'orion_msc_g_kp': ['IRn', 'Pare1839', 'ir1491', 'ir1374', 'ir1366'],
                                  'orion_msc_h_kp': ['IRn', 'Pare1839', 'ir1491', 'ir1374', 'ir1366'],
                                  'orion_msc_i_kp': ['IRn', 'Pare1839', 'ir1491', 'ir1374', 'ir1366'],
                                  'orion_msc_j_kp': ['IRn', 'Pare1839', 'ir1491', 'ir1374', 'ir1366'],
                                  'BN_Mosaic_A_kp': ['IRn', 'Pare1839', 'ir1491', 'ir1374', 'ir1352', 'ir1486', 'ir1519', 'toller_5', 'toller_6', 'toller_7', 'toller_8', 'toller_9', 'toller_10', 'CXOONC'],
                                  'BN_Mosaic_B_kp': ['IRn', 'Pare1839', 'ir1491', 'ir1374', 'ir1352', 'ir1486', 'ir1519', 'toller_5', 'toller_6', 'toller_7', 'toller_8', 'toller_9', 'toller_10', 'CXOONC'],
                                  'BN_Mosaic_C_kp': ['IRn', 'Pare1839', 'ir1491', 'ir1374', 'ir1352', 'ir1486', 'toller_6', 'toller_7', 'toller_8', 'toller_9', 'CXOONC', 'toller_10', 'toller_5', 'ir1519'],
                                  'BN_Mosaic_D_kp': ['IRn', 'Pare1839', 'ir1491', 'ir1374', 'ir1486', 'toller_5', 'toller_6', 'toller_7', 'toller_8', 'toller_9', 'CXOONC', 'toller_10', 'toller_3'],
                                  'BN_Mosaic_E_kp': ['Pare1839', 'ir1491', 'ir1486', 'toller_8', 'toller_9', 'toller_3', 'toller_2', 'toller_4', 'toller_11'],
                                  'BN_Mosaic_F_kp': ['toller_4', 'toller_2', 'toller_3', 'toller_11'],
                                  'BN_Mosaic_F_kp': ['IRn', 'Pare1839', 'ir1491', 'ir1374', 'ir1366'],
                                  'BN_Mosaic_G_kp': ['IRn', 'Pare1839', 'ir1491', 'ir1374', 'ir1486', 'toller_5', 'toller_6', 'toller_7', 'toller_8', 'toller_9', 'toller_2', 'toller_3', 'CXOONC'],
                                  'BN_Mosaic_H_kp': ['IRn', 'Pare1839', 'ir1374', 'ir1352', 'ir1486', 'toller_8', 'CXOONC', 'toller_10'],
                                  'BN_Mosaic_I_kp': ['IRn', 'Pare1839', 'ir1491', 'ir1374', 'ir1352', 'CXOONC', 'toller_5', 'toller_6', 'toller_7', 'toller_9'],
                                  'BN_Mosaic_J_kp': ['IRn', 'Pare1839', 'ir1491', 'ir1374', 'ir1352', 'ir1519', 'ir1486', 'CXOONC', 'toller_5', 'toller_6', 'toller_7', 'toller_8', 'toller_9', 'toller_10'],
                                  'BN_Mosaic_K_kp': ['IRn', 'Pare1839', 'ir1491', 'ir1374', 'ir1352', 'ir1519', 'ir1486', 'CXOONC', 'toller_5', 'toller_6', 'toller_7', 'toller_8', 'toller_9', 'toller_10'],
                                  }

        self.mapFilter2Coo = {'orion_msc_a_kp': 'IRn',
                                  'orion_msc_b_kp': 'IRn',
                                  'orion_msc_c_kp': 'IRn',
                                  'orion_msc_d_kp': 'IRn',
                                  'orion_msc_e_kp': 'IRn',
                                  'orion_msc_f_kp': 'IRn',
                                  'orion_msc_g_kp': 'IRn',
                                  'orion_msc_h_kp': 'IRn',
                                  'orion_msc_i_kp': 'IRn',
                                  'orion_msc_j_kp': 'IRn',
                                  'BN_Mosaic_A_kp': 'Pare1839',
                                  'BN_Mosaic_B_kp': 'Pare1839',
                                  'BN_Mosaic_C_kp': 'Pare1839',
                                  'BN_Mosaic_D_kp': 'Pare1839',
                                  'BN_Mosaic_E_kp': 'ir1491',
                                  'BN_Mosaic_F_kp': 'toller_4',
                                  'BN_Mosaic_G_kp': 'Pare1839',
                                  'BN_Mosaic_H_kp': 'Pare1839',
                                  'BN_Mosaic_I_kp': 'ir1374',
                                  'BN_Mosaic_J_kp': 'Pare1839',
                                  'BN_Mosaic_K_kp': 'Pare1839',
                                  }
        self.mapFilter2CalCoo = {'orion_msc_a_kp': 'IRn',
                                  'orion_msc_b_kp': 'IRn',
                                  'orion_msc_c_kp': 'ir1374',
                                  'orion_msc_d_kp': 'IRn',
                                  'orion_msc_e_kp': 'IRn',
                                  'orion_msc_f_kp': 'IRn',
                                  'orion_msc_g_kp': 'IRn',
                                  'orion_msc_h_kp': 'IRn',
                                  'orion_msc_i_kp': 'IRn',
                                  'orion_msc_j_kp': 'IRn',
                                  'BN_Mosaic_A_kp': 'Pare1839',
                                  'BN_Mosaic_B_kp': 'Pare1839',
                                  'BN_Mosaic_C_kp': 'Pare1839',
                                  'BN_Mosaic_D_kp': 'Pare1839',
                                  'BN_Mosaic_E_kp': 'ir1491',
                                  'BN_Mosaic_F_kp': 'toller_4',
                                  'BN_Mosaic_G_kp': 'Pare1839',
                                  'BN_Mosaic_H_kp': 'Pare1839',
                                  'BN_Mosaic_I_kp': 'ir1374',
                                  'BN_Mosaic_J_kp': 'Pare1839',
                                  'BN_Mosaic_K_kp': 'Pare1839',
                                  }
                             

        # Initialize the Analysis object
        Analysis.__init__(self, epoch, filt=filt,
                          rootDir=rootDir, epochDirSuffix=epochDirSuffix,
                          useDistorted=useDistorted, cleanList=cleanList)

        ##########
        # Setup the appropriate calibration stuff.
        ##########
        self.calFile = '/g/lu/data/orion/source_list/'   # old file: '/u/ghezgroup/data/orion/source_list/'
        self.calFile += 'photo_calib.dat'    # old file: 'photo_calib.dat'

        self.calStars = self.mapFilter2CalStars[filt]
        #self.calStars = None  # use defaults in photo_calib file

        # Choose the column based on the filter
        self.calColumn = 3

        # Set the coo star
        self.cooStar = self.mapFilter2Coo[filt]
        self.calCooStar = self.mapFilter2CalCoo[filt]

        # Set the psf starlist
        self.starlist = '/g/lu/code/idl/ucla_idl/'
        self.starlist += 'orion/psfstars/psf_mosaic.dat'

        self.labellist = '/g/lu/data/orion/source_list/label.dat'
        self.orbitlist = None

        #Set some default parameters
        self.type = 'ao'
        self.corrMain = 0.7
        self.corrSub = 0.6
        self.corrClean = 0.7
        
        self.deblend = None

        self.alignFlags = '-R 3 -v -p -a 0'
        

