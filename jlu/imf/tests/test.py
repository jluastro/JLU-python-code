from jlu.imf import imf
import numpy as np
import numpy.testing as npt


def test_theta_open():
    # Single values
    a = imf.theta_open(-1)
    assert a == 0

    a = imf.theta_open(0)
    assert a == 0

    a = imf.theta_open(1)
    assert a == 1

    # Array values
    xin = np.array([-1.0, 0, 1.0])
    xout = np.array([0, 0, 1])
    aout = imf.theta_open(xin)
    npt.assert_equal(xout, aout)

def test_theta_closed():
    # Single values
    a = imf.theta_closed(-1)
    assert a == 0

    a = imf.theta_closed(0)
    assert a == 1

    a = imf.theta_closed(1)
    assert a == 1

    # Array values
    xin = np.array([-1.0, 0, 1.0])
    xout = np.array([0, 1, 1])
    aout = imf.theta_closed(xin)
    npt.assert_equal(xout, aout)

def test_gamma_closed():
    ###
    # Single Values
    ###
    # edge cases
    a = imf.gamma_closed(1.0, 0.0, 1.0)
    assert a == 1

    a = imf.gamma_closed(0.0, 0.0, 1.0)
    assert a == 1

    # outside cases
    a = imf.gamma_closed(-0.2, 0.0, 1.0)
    assert a == 0

    a = imf.gamma_closed(1.2, 0.0, 1.0)
    assert a == 0

    # normal case
    a = imf.gamma_closed(0.5, 0.0, 1.0)
    assert a == 1

    ###
    # Array Values
    ###
    xin = np.array([-0.2, 0.0, 0.5, 1.0, 1.2])
    xout = np.array([0.0, 1.0, 1.0, 1.0, 0.0])
    aout = imf.gamma_closed(xin, 0.0, 1.0)
    npt.assert_equal(xout, aout)

    

    
    
    
