from jlu.imf import imf
import numpy as np
import numpy.testing as npt
import pdb

def test_single_value():
    #####
    # First set of IMF slopes -- similar to Kroupa
    #####
    massLimits = np.array([0.3, 1.2, 100])
    imfSlopes = np.array([-1.3, -2.35])
    totalMass = 1.0e4

    test_imf = imf.IMF_broken_powerlaw(massLimits, imfSlopes)
    test_imf.imf_norm_cl(totalMass)

    # Generate a mass from the lower segment
    mass = test_imf.imf_dice_star_cl(0.2)
    npt.assert_almost_equal(mass, 0.42199238)

    # Generate a mass from the upper segment
    mass = test_imf.imf_dice_star_cl(0.95)
    npt.assert_almost_equal(mass, 4.48895482)

    #####
    # Second set of IMF slopes -- excercise power = -1 special case
    #####
    massLimits = np.array([0.3, 1.2, 100])
    imfSlopes = np.array([-1.0, -2.35])
    totalMass = 1.0e4

    test_imf = imf.IMF_broken_powerlaw(massLimits, imfSlopes)
    test_imf.imf_norm_cl(totalMass)

    # Generate a mass from the lower segment
    mass = test_imf.imf_dice_star_cl(0.2)
    npt.assert_almost_equal(mass, 0.45889040)

    # Generate a mass from the upper segment
    mass = test_imf.imf_dice_star_cl(0.95)
    npt.assert_almost_equal(mass, 4.99091954)

def test_array_value():
    #####
    # First set of IMF slopes -- similar to Kroupa
    #####
    massLimits = np.array([0.3, 1.2, 100])
    imfSlopes = np.array([-1.3, -2.35])
    totalMass = 1.0e4

    test_imf = imf.IMF_broken_powerlaw(massLimits, imfSlopes)
    test_imf.imf_norm_cl(totalMass)

    rand_num = np.arange(0.001, 0.999, 0.05)
    mass_array = test_imf.imf_dice_star_cl(rand_num)

    for rr in range(len(rand_num)):
        mass_single = test_imf.imf_dice_star_cl(rand_num[rr])
        npt.assert_almost_equal(mass_single, mass_array[rr])


    #####
    # Second set of IMF slopes -- excercise power = -1 special case
    #####
    massLimits = np.array([0.3, 1.2, 100])
    imfSlopes = np.array([-1.0, -2.35])
    totalMass = 1.0e4

    test_imf = imf.IMF_broken_powerlaw(massLimits, imfSlopes)
    test_imf.imf_norm_cl(totalMass)

    rand_num = np.arange(0.001, 0.999, 0.05)
    mass_array = test_imf.imf_dice_star_cl(rand_num)

    for rr in range(len(rand_num)):
        mass_single = test_imf.imf_dice_star_cl(rand_num[rr])

        msg = 'Error for random number {0}'.format(rand_num[rr])
        npt.assert_almost_equal(mass_single, mass_array[rr], err_msg=msg)


def test_broken_dice():
    ###
    # 2013-07-26 Test that breaks dice cluster
    # 
    # I know that this random number, input to dice cluster,
    # breaks things.
    ###
    r = np.arange(0, 1, 0.05)
    # r = 0.051

    # Setup an IMF object
    massLimits = np.array([0.3, 1.2, 100])
    imfSlopes = np.array([-1.0, -2.0])
    totalMass = 1.0e4

    test_imf = imf.IMF_broken_powerlaw(massLimits, imfSlopes)
    test_imf.imf_norm_cl(totalMass)


    m = test_imf.imf_dice_star_cl(r)
    m_check = np.array([ 0.30000000, 0.33781459, 0.38039566, 0.42834401, 0.48233618,
                         0.54313400, 0.61159530, 0.68868605, 0.77549398, 0.87324393,
                         0.98331514, 1.10726066, 1.24776735, 1.42348243, 1.65679880,
                         1.98159236, 2.46477998, 3.25959271, 4.81097989, 9.18029750])

    npt.assert_almost_equal(m, m_check)

def test_boundary_double_count():
    """
    Test randomly selected masses right at the mass boundaries.
    """
    # Setup an IMF object
    massLimits = np.array([0.1, 1.0, 10.0, 100.0])
    imfSlopes = np.array([-1.0, -2.0, -1.5])
    totalMass = 1.0e3

    test_imf = imf.IMF_broken_powerlaw(massLimits, imfSlopes)
    test_imf.imf_norm_cl(totalMass)

    # Test where some random number (used to generate masses) exactly
    # equals the lamda values.
    lamda = test_imf.lamda
    rin = lamda / lamda[-1]

    m = test_imf.imf_dice_star_cl(rin)
    npt.assert_almost_equal(m, massLimits)
    
    
    

    
