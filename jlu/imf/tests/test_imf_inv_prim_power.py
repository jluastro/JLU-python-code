from jlu.imf import imf
import numpy as np
import numpy.testing as npt
import unittest

def test_single_inputs():
    """
    Test imf.imf_inv_prim_power() with single values (not arrays)
    as input parameters. Make sure the algorithm is properly
    returning:

    ((1 + power) * x) ** (1.0 / (1 + power))
    """

    ###
    # Valid: Negative x and power < -1
    ###
    x = -2.1
    power = -1.3

    val = imf.imf_inv_prim_power(x, power)
    check = ((1.0 + power) * x) ** (1.0 / (1.0 + power))

    assert val == check


    ###
    # Valid: Postive x and power > -1
    ###
    x = 2.1
    power = -0.3

    val = imf.imf_inv_prim_power(x, power)
    check = ((1.0 + power) * x) ** (1.0 / (1.0 + power))

    assert val == check

    ###
    # Invalid: Postive x and power < -1
    # Should return NAN
    ###
    x = 2.1
    power = -1.3

    val = imf.imf_inv_prim_power(x, power)
    npt.assert_equal(val, np.nan)


    ###
    # Valid: Negative x and power < -1
    # Repeat with power = -1 special case
    ###
    x = -2.1
    power = -1.0

    val = imf.imf_inv_prim_power(x, power)
    npt.assert_almost_equal(val, 0.12245643)


    ###
    # Valid: Postive x and power > -1
    # Repeat with power = -1 special case
    ###
    x = 2.1
    power = -1.0

    val = imf.imf_inv_prim_power(x, power)
    npt.assert_almost_equal(val, 8.16616991)


def test_array_inputs():
    """
    Test imf.imf_inv_prim_power() with array input values.
    """

    ###
    # Valid: Negative x and power < -1
    # x is array
    # power is single
    ###
    x = np.array([-2.1])
    power = -1.3

    val = imf.imf_inv_prim_power(x, power)
    npt.assert_almost_equal(val, np.array([4.66514165]))

    ###
    # Valid: Negative x and power < -1
    # x is single
    # power is array
    ###
    x = -2.1
    power = np.array([-1.3])

    val = imf.imf_inv_prim_power(x, power)
    npt.assert_almost_equal(val, np.array([4.66514165]))


    ###
    # Valid: Negative x and power < -1
    # x is array
    # power is array
    ###
    x = np.array([-2.1])
    power = np.array([-1.3])

    val = imf.imf_inv_prim_power(x, power)
    npt.assert_almost_equal(val, np.array([4.66514165]))



    ###
    # Valid: Negative x and power < -1
    # Repeat with power = -1 special case
    # x is array
    # power is single
    ###
    x = np.array([-2.1, -1.9])
    power = -1.0

    val = imf.imf_inv_prim_power(x, power)
    npt.assert_almost_equal(val, np.array([0.12245643, 0.14956862]))


    ###
    # Valid: Postive x and power > -1
    # Repeat with power = -1 special case
    # power is array
    # x is single
    ###
    x = -2.1
    power = np.array([-1.0, -2.3])

    val = imf.imf_inv_prim_power(x, power)
    npt.assert_almost_equal(val, np.array([0.12245643, 0.46183865]))

    ###
    # Valid: Postive x and power > -1
    # Repeat with power = -1 special case
    # power is array
    # x is array
    ###
    x = np.array([-2.1, -1.9])
    power = np.array([-1.0, -2.3])

    val = imf.imf_inv_prim_power(x, power)
    npt.assert_almost_equal(val, np.array([0.12245643, 0.49879882]))

    ###
    # INVALID: Postive x and power > -1
    # Repeat with power = -1 special case
    # power is array
    # x is array
    ###
    # x = np.array([-2.1, -1.9, -2.2])
    # power = np.array([-1.0, -2.3])

    # val = imf.imf_inv_prim_power(x, power)
    # unittest.assertRaises(
    # npt.assert_almost_equal(val, np.array([0.12245643]))

    return imf
