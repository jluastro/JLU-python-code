import numpy as np
from scipy.optimize import curve_fit, minimize
from scipy import interpolate




class transform:
    
    def __init__(self, x1, y1, x_ref, y_ref, order_poly=3,order_spline=2 ,num_knots=2, mag = None ,smooth=False, smooth_fac=None, weights=None, fit_spline_b=True, mag_lim=None, min_points=2, dif_filt=True  ):
        '''
        
    
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
        coeff_x,coeff_y - polynomial coefficients for the polynomial fit
        weights is array of weights used in fits -- same length as x,y
        mag -- array of magntiudes for x,y -- if none then there will be no filtering in brightness
        mag_filt -- int or arraylike, if int it is brightness cut for a point to be used -- if list mag_filt[0] 
        
        '''
        
        order_dict = {0:1,1:3,2:6,3:10,4:15,5:21,6:28,7:36}
        num_poly_param = order_dict[order_poly]
    
        

        '''
        Fit polynomial to requested order
        '''


        self.fit_spline_b=fit_spline_b
        self.x1 = x1
        self.y1 = y1 
        self.y_ref = y_ref 
        self.x_ref = x_ref
        self.mag = mag
        self.num_knots = num_knots
        self.bool_ar = np.ones(self.x1.shape, dtype='bool')
        self.mag_lim = mag_lim
        self.mag = mag

        self.id_num = np.zeros(len(self.x1))
        for i in range(len(self.x1)):
            self.id_num[i] = i
            
        if dif_filt:
            self.filter(min_points, num_knots)
            
        if order_poly < 3:
            coeff_x, coeff_y = fit_poly(x1, y1, x_ref, y_ref, num_poly_param, weights=weights)
            x_poly = poly(np.array([x1,y1]), coeff_x)
            y_poly = poly(np.array([x1, y1]), coeff_y)

        else:

            
            cur_or = 1
            p0x = np.zeros(order_dict[cur_or])
            p0y = np.zeros(order_dict[cur_or])
            
            while cur_or < order_poly + 1:
                coeff_x, coeff_y = fit_poly(x1, y1, x_ref, y_ref, order_dict[cur_or], p0x=p0x, p0y = p0y, weights=weights)
                cur_or += 1
                p0x = np.zeros(order_dict[cur_or])
                p0y = np.zeros(order_dict[cur_or])

                for i in range(len(coeff_x)):
                    p0x[i] = coeff_x[i]
                    p0y[i] = coeff_y[i]

            x_poly = poly(np.array([x1,y1]), coeff_x)
            y_poly = poly(np.array([x1, y1]), coeff_y)
                
                 


        '''
        Now do spline fit on residual
        '''

       
        dx , spline_x =  fit_spline(x_poly, y_poly, x_ref-x_poly, num_knots=num_knots, smooth=smooth, smooth_fac=smooth_fac, weights=weights, order=order_spline)
        dy , spline_y =  fit_spline(x_poly, y_poly, y_ref-y_poly, num_knots=num_knots, smooth=smooth, smooth_fac=smooth_fac, weights=weights, order=order_spline)
        x_new = dx + x_poly
        y_new = dy + y_poly
                


        self.spline_x = spline_x
        self.spline_y = spline_y
        self.coeff_x = coeff_x
        self.coeff_y = coeff_y
    

    
    def evaluate(self, x,y):
        x_poly = poly(np.array([x,y]), self.coeff_x)
        y_poly = poly(np.array([x,y]), self.coeff_y)

        x_prime = self.spline_x.ev(x_poly, y_poly) + x_poly
        y_prime = self.spline_y.ev(x_poly, y_poly) + y_poly

        if self.fit_spline_b:
            return x_prime, y_prime
        else:
            return x_poly, y_poly

    def filter(self, min_points, num_knots):
    
        xknots = np.linspace(np.min(self.x1), np.max(self.x1), num=num_knots)
        yknots = np.linspace(np.min(self.y1), np.max(self.y1), num=num_knots)

        #start by filtering out everything too dim
        bool_ar = self.mag  < self.mag_lim

        #now add some of those stars back if we want them to have enough ampling for the spline 
        for i in range(num_knots-1):
            for j in range(num_knots-1):
                box_bool = np.zeros(self.x1.shape, dtype='bool')
                box_bool = (self.x1 > xknots[i])* (self.x1 < xknots[i+1]) * (self.y1 > yknots[j]) * (self.y1 < yknots[j+1])
                box_bool_mag = (self.x1 > xknots[i])* (self.x1 < xknots[i+1])* (self.y1 > yknots[j])* (self.y1 < yknots[j+1])* (self.mag < self.mag_lim)
                if np.sum(bool_ar[box_bool_mag]) < min_points:
                    #not enough points, now find next brightest points to get up to minimum number of points
                    #first check total numbe rof points in the correct range
                    if np.sum(box_bool) > min_points:
                        in_add = []
                        for k in range(min_points - np.sum(box_bool_mag)):
                            m = 100000
                            for ind in range(np.sum(box_bool)):
                                if  self.mag[box_bool][ind] > self.mag_lim and self.mag[box_bool][ind] < m and ind not in in_add:
                                    m = self.mag[box_bool][ind]
                                    new_ind = ind
                            in_add.append(new_ind)

                        for kk in in_add:
                            bool_ar[self.id_num[box_bool][kk]] = True
                    else:
                        print 'Only ',np.sum(box_bool),' points in range x ', xknots[i],xknots[i+1], '  y: ', yknots[j], yknots[j+1] 
                                   
                        

        self.bool_ar = bool_ar 
                    
                        
                
   
        

def fit_poly(x1, y1, xref, yref, num_free_param , weights=None, p0x = None, p0y=None, minim=False):
    '''
    Assumes input is 2 matched starlists (x1, y1, x2, y2)
    free_param is number of free parameters to be used in the fit
    returns coefficients for best fit polynomial in both x and y
    '''

    if p0x ==None:
        p0x = np.ones(num_free_param)
    if p0y == None:
        p0y = np.ones(num_free_param)


    if not minim:
        c_x, cov_x = curve_fit(poly, np.array([x1,y1]), xref, p0=p0x,sigma = weights)
        c_y, cov_y = curve_fit(poly, np.array([x1,y1]), yref, p0=p0y,sigma = weights)
    else:
        for i in range(len(p0x)):
            p0x[i] = p0x[i]# + i * .01
            p0y[i] = p0y[i]# + i * .01
        resx = minimize(poly_min,p0x, args=(np.array([x1,y1]), xref))
        resy = minimize(poly_min,p0y, args=(np.array([x1,y1]), yref))
        c_x = resx.x
        c_y = resy.x
    return c_x, c_y  




def poly_min(param_vec, data, ref ):
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
            
    return np.sum(np.abs(x_new - ref))

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
            

        #print exp1, exp2
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
            spline = interpolate.SmoothBivariateSpline(x, y, x_ref, w=weights)
            x_new = spline.ev(x, y)
        return x_new, spline 
