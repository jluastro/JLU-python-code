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
    r = np.array([0.001, 0.051])
    # r = 0.051

    # Setup an IMF object
    massLimits = np.array([0.3, 1.2, 100])
    imfSlopes = np.array([-1.0, -2.35])
    totalMass = 1.0e4

    test_imf = imf.IMF_broken_powerlaw(massLimits, imfSlopes)
    test_imf.imf_norm_cl(totalMass)


    # Go through the steps in dice cluster
    r = np.atleast_1d(r)  # Make sure it is an array
    
    x = r * test_imf.lamda[-1]
    y = np.zeros(len(r), dtype=float)
    z = np.ones(len(r), dtype=float)

    for i in range(test_imf.nterms):
        aux = x - test_imf.lamda[i]

        # Only continue for those entries that are in later segments
        idx = np.where(aux >= 0)[0]

        # Maybe we are all done?
        if len(idx) == 0:
            break

        x_tmp = x[idx]
        aux_tmp = aux[idx]

        # len(idx) entries
        t1 = aux_tmp / (test_imf.coeffs[i] * test_imf.k)
        t1 += imf.imf_prim_power(test_imf.mLimitsLow[i], test_imf.powers[i])
        y_i = imf.gamma_closed(x_tmp, test_imf.lamda[i], test_imf.lamda[i+1])
        y_i *= imf.imf_inv_prim_power(t1, test_imf.powers[i])

        # Save results into the y array
        y[idx] += y_i

        z *= imf.delta(x - test_imf.lamda[i])




    

    
