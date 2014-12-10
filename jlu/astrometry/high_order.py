import numpy as np
from scipy.optimize import curve_fit
from scipy import interpolate




class transform:
    
    def __init__(self, x1, y1, x_ref, y_ref, order=3, num_knots=5, smooth=False, smooth_fac=None, weights=None ):
        '''
        Wrapper to find high order fit to delta distances
    
        args
        ----------------------------
        x1,y1 - 1d array of rereference coordinates
        xref, yref - 1d array of master coordiantes that correspond to x1 and y1

        keywords
        -------------------------------
        order  - int, order of polynomial used
        num_knots - int, number of knots used by spline fit
        smooth_fac - float, optional parameter that sets smoothing factor used in LSQBivariateSpline
        smooth -  bool, if True spline fit uses smooth version (interpolate.SmoothBivariateSpline) note that num_knots is then unused

        weights - array of weights for the input data points, must be same length
        x_new, y_new - transformed coordinates, correspondds to reference frame
        spline_x, spline_y - spline transformation objects (see scipy.interpolate.BivariateSpline)
        coeff_x,xoeff_y - polynomial coefficients for the polynomial fit
        '''
        order_dict = {1:3,2:6,3:10,4:15,5:21,6:28}
        num_poly_param = order_dict[order]
    
        #x1, y1, x_ref, y_ref = match_simple(x1in, x2in, y1in, y2in, m1in, m_ref)

    
        '''
        With basic match, fit deltax , deltay with high order polynomial 
        '''

        coeff_x, coeff_y = fit_poly(x1, y1, x_ref, y_ref, num_poly_param)
        x_poly = poly(np.array([x1,y1]), coeff_x)
        y_poly = poly(np.array([x1, y1]), coeff_y)


        '''
        Now do spline fit on residual
        '''

        x_new , spline_x =  fit_spline(x_poly, y_poly, x_ref, num_knots=num_knots, smooth=smooth, smooth_fac=smooth_fac, weights=weights)
        y_new , spline_y =  fit_spline(x_poly, y_poly, y_ref, num_knots=num_knots, smooth=smooth, smooth_fac=smooth_fac, weights=weights)

        self.spline_x = spline_x
        self.spline_y = spline_y
        self.coeff_x = coeff_x
        self.coeff_y = coeff_y
    

    
    def evaluate(self, x,y):
        x_poly = poly(np.array([x,y]), self.coeff_x)
        y_poly = poly(np.array([x, y]), self.coeff_y)

        x_prime = self.spline_x.ev(x_poly, y_poly)
        y_prime = self.spline_y.ev(x_poly, y_poly)

        return x_prime, y_prime



def fit_poly(x1, y1, xref, yref, num_free_param):
    '''
    Assumes input is 2 matched starlists (x1, y1, x2, y2)
    free_param is number of free parameters to be used in the fit
    returns coefficients for best fit polynomial in both x and y
    '''


    c_x, cov_x = curve_fit(poly, np.array([x1,y1]), xref, p0=np.zeros(num_free_param))
    c_y, cov_y = curve_fit(poly, np.array([x1,y1]), yref, p0=np.zeros(num_free_param))

    return c_x, c_y  
    

def poly(data, *param_vec):
    '''
    Performs Polynomial Transformation up to number of coefficients it is given
    '''


    if len(param_vec) == 1:
        param_vec = param_vec[0]

        
    x = data[0,:]
    y = data[1,:]

    exp1 = 1
    exp2 = 0
    level= 1
    x_new = np.zeros(len(x)) + param_vec[0]
    i = 1
    cont = True 
    while i < len(param_vec):
        x_new = x_new + param_vec[i] * x**exp1 * y**exp2 
        i +=1 
            

        if exp2 == level:
            level += 1
            exp1 = level
            exp2 = 0 
        else:
            exp1 = exp1 - 1 
            exp2 = exp2 + 1
            
    return x_new 

def fit_spline( x, y, x_ref, knot_x=None, knot_y=None, num_knots=5, smooth=False, smooth_fac=None, weights=None, order=3):
        '''
        performs spline fit of the form dx = f(x,y)
        knot_x/knot_y are 1-d arrays that are the  x and y coordinates of knot locations
        '''
        if knot_x == None:
            knot_x = np.linspace(np.min(x), np.max(x), num=num_knots)
        if knot_y == None:
            knot_y = np.linspace(np.min(y), np.max(y), num=num_knots)
    

        if not smooth:
            if smooth_fac == None:
                spline = interpolate.LSQBivariateSpline(x, y, x_ref, knot_x, knot_y, kx=order, ky=order, w=weights)
                x_new = spline.ev(x, y)
            else:
                spline = interpolate.LSQBivariateSpline(x, y, x_ref, knot_x, knot_y, kx=order, ky=order,w=weights, s=smooth_fac)
                x_new = spline.ev(x, y)
        else:
            spline = interpolate.SmoothBivariateSpline(x, y, x_ref)
            x_new = spline.ev(x, y)
        return x_new, spline 
