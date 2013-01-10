from matplotlib import transforms
from scipy import optimize
import numpy as np
import math

def fit_affine2d_noshear(points_in, points_out,
                         err_points_in=None, err_points_out=None):
    """
    Find the optimal rotation, scale, and translation given a set of points
    (and optional error bars). 

    The resulting transformation is:

    points_out = S * (R * points_in) + trans
    
    where R is the rotation matrix and is applied first, S is the scale matrix
    and is applied second, and trans is the translation vector applied last. The
    appropriate code to transform points using the output solution would be:

    t = matplotlib.transforms.Affine2D()
    t.rotate(theta)
    t.scale(scale)
    t.transform(tx, ty)

    point_out = t.transform_point(point_in)

    The format of the input should be:
    points_in -- numpy array of shape N x 2, where N is the number of points
    points_out -- numpy array of shape N x 2
    err_points_in -- numpy array of shape N x 2 (optional)
    err_points_out -- numpy array of shape N x 2 (optional)
    """
    def fit_func(params, points_in, points_out, err_points_in, err_points_out):
        theta = params[0]
        scale = params[1]
        tx = params[2]
        ty = params[3]

        # Transform the input data points
        t = transforms.Affine2D()
        t.rotate_deg(theta)
        t.scale(scale)
        t.translate(tx, ty)

        points_test = t.transform(points_in)

        # Transform the input error bars
        if err_points_in != None:
            t_err = transforms.Affine2D()
            t_err.rotate(theta)
            t_err.scale(scale)

            err_points_test = t.transform(err_points_in)

        # Compute the deltas (squared)
        diffXY = points_out - points_test

        # Deal with optional errors.
        if err_points_in != None:
            if err_points_out != None:
                errXY = np.hypot(err_points_in, err_points_out)
            else:
                errXY = err_points_in
        else:
            if err_points_out != None:
                errXY = err_points_out
            else:
                errXY = np.ones(diffXY.shape, dtype=float)


        # Turn XY deltas into R deltas
        diffR = np.hypot(diffXY[:,0], diffXY[:,1])
        errR = np.hypot(diffXY[:,0] * errXY[:,0], diffXY[:,1] * errXY[:,1]) / diffR
        diffR /= errR

        return diffR

    params0 = np.array([90., 1.0, 0., 0.])
    data = (points_in, points_out, err_points_in, err_points_out)

    out = optimize.leastsq(fit_func, params0, args=data, full_output=1)

    pfinal = out[0]
    covar = out[1]

    params = {'angle': pfinal[0],
              'scale': pfinal[1],
              'transX': pfinal[2],
              'transY': pfinal[3]}

    perrors = {'angle': math.sqrt(covar[0][0]),
               'scale': math.sqrt(covar[1][1]),
               'transX': math.sqrt(covar[2][2]),
               'transY': math.sqrt(covar[3][3])}

    return params, perrors
    
