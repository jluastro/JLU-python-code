
from astropy.modeling import models, fitting
import numpy as np
from scipy.interpolate import LSQBivariateSpline as spline

class Transform:


    def __init__():
        '''
        Thought I would want a genreal Tranform class, but it wasn't clear there was enough general cgharacteristics to be useful
        I leave it here now as a reminder to keep the idea in mind
        '''
        pass


class PolyTransform:


    '''
    defines a 2d polynomial transform between x,y -> xref,yref
    tranforms are independent for x and y, of the form
    x' = c0_0 + c_0_1 *x + c1_0*y + ....
    y' = d0_0 + d_0_1 *x + d1_0*y + ....
    currently only supports initial guess of the linear terms
    '''
    def __init__(self,x, y, xref, yref, degree, init_gx=None,init_gy=None, weights=None ):

       
        init_gx = check_initial_guess(init_gx)
        init_gy = check_initial_guess(init_gy)
        
        
        self.degree = degree
        p_init_x = models.Polynomial2D(degree=degree, c0_0 =init_gx[0], c1_0=init_gx[1], c0_1=init_gx[2] )
        p_init_y = models.Polynomial2D(degree=degree, c0_0 =init_gy[0], c1_0=init_gy[1], c0_1=init_gy[2] )
        if self.degree == 1:
            fit_p  = fitting.LinearLSQFitter()
        else:
            fit_p = fitting.LevMarLSQFitter()

        self.px = fit_p(p_init_x, x, y, xref, weights=weights)
        self.py = fit_p(p_init_y,x,y,yref, weights=weights)
        

    def evaluate(self, x,y):
        return self.px(x,y), self.py(x,y)
    
class LegTransform:

    def __init__(self, x, y, xref, yref, degree, init_gx=None,init_gy=None, weights=None ):
        '''
        defines a 2d polnyomial tranformation fomr x,y -> xref,yref using Legnedre polynomials as the basis
        transforms are independent for x and y, of the form
        x' = c0_0 + c1_0 * L_1(x) + c0_1*L_1(y) + ....
        y' = d0_0 + d1_0 * L_1(x) + d0_1*L_1(y) + ....
        Note that all input coorindates will be renomalized to be on the interval of [-1:1] for fitting
        The evaulate function will use the same renomralization procedure
        '''

        init_gx = check_initial_guess(init_gx)
        init_gy = check_initial_guess(init_gy)
        
        self.x_nc , x_norm= self.norm0(x)
        self.x_ncr, x_norm_ref = self.norm0(xref)
        self.y_nc , y_norm = self.norm0(y)
        self.y_ncr , y_norm_ref = self.norm0(yref)
        self.degree = degree
        
        p_init_x = models.Legendre2D(degree, degree, c0_0 =init_gx[0], c1_0=init_gx[1], c0_1=init_gx[2])
        p_init_y = models.Legendre2D(degree, degree, c0_0 =init_gy[0], c1_0=init_gy[1], c0_1=init_gy[2])
        fit_p = fitting.LevMarLSQFitter()

        self.px = fit_p(p_init_x, x_norm, y_norm, x_norm_ref, weights=weights)
        self.py = fit_p(p_init_y, x_norm, y_norm, y_norm_ref, weights=weights)

    def evaluate(self, x, y):
        xnew = self.rnorm(self.px(self.norm(x, self.x_nc), self.norm(y, self.y_nc)),self.x_ncr )
        ynew = self.rnorm(self.py(self.norm(x,self.x_nc), self.norm(y,self.y_nc)),self.y_ncr)
        return xnew, ynew

        


    def norm0(self, x):

        
        xmin = np.min(x)
        xdiv = np.max(x- xmin)/2.0
        n_param = np.array([xmin, xdiv])
        return n_param, self.norm(x, n_param)
        
    def norm(self, x, n_param):
        '''
        x is vector to be normalized, n_param is an array of [offset to subtract, value to divide]
        '''
        return (x - n_param[0]) / n_param[1] - 1.0
       
        
    def rnorm(self,x,n_param):
        '''
        reverses normalization process 
        '''
        
        return (x+1.0)   * n_param[1] + n_param[0]

class ClipTransform:

    def __init__(self,x , y , xref, yref, degree, niter=3, sig_clip =3 , weights=None):
        self.s_bool = np.ones(x.shape, dtype='bool')

        c_x, y_x = four_param(x, y, xref, yref)
        for i in range(niter+1):
            t = PolyTransform(x, y, xref, yref, degree, init_gx=c_x, init_gy=c_y, weights=weights)
            #reset the initial guesses based on the previous tranforamtion
            #it is not clear to me that using these values is better than recalculating an intial guess from a 4 parameter tranform
            c_x[0] = t.px.c0_0
            c_x[1] = t.px.c1_0
            c_x[2] = t.px.c0_1

            c_y[0] = t.py.c0_0
            c_y[1] = t.py.c1_0
            c_y[2] = t.py.c0_1

            xev, yev = t.evaluate(x, y)
            dx = xref - xev
            dy = yref - yev
            mx = np.mean(dx[self.s_bool])
            my = np.mean(dy[self.s_bool])
            
            sigx = np.std(dx[self.s_bool])
            sigy = np.std(dy[self.s_bool])

            if i != niter :
                #do not update the star boolean if we have performed the final tranformation
                self.s_bool = self.s_bool - (dx > mx + sig_clip * sigx) - (dx < mx - sig_clip * sigx) - (dy > my + sig_clip * sigy) - (dy < my - sig_clip * sigy)

        self.t = t

    def evaluate(self,x,y):
        return self.t.evaluate(x,y)
            
        
class PolySplineTransform:

    def __init__(self, x, y, xref, yref, degree,weights ):

        '''
        '''
        self.poly = PolyTransform(x, y, xref, yref, degree,weights)
        xev, yev = self.poly.evaluate(x, y)
        self.spline = SplineTransform(xev, yev, xref, yref,weights)

    def evaluate(self, x, y):
        xev, yev = self.poly.evaluate(x, y)
        return self.spline.evaluate(xev, yev)
        

        

class SplineTransform:


    def __init__(self, x, y, xref, yref,weights=None, kx=None,ky=None):
        if weights==None:
            weights = np.ones(x.shape)
        if kx == None:
            kx = np.linspace(x.min(), x.max())
        if ky == None:
            ky = np.linspace(y.min(), y.max())

        self.spline_x = spline(x,y,xref, tx=kx,ty=ky,w=weights)
        self.spline_y = spline(x,y,yref, tx=kx,ty=ky,w=weights)

    def evaluate(self,x,y):
        return self.spline_x.ev(x,y), self.spline_y.ev(x,y)




        
def check_initial_guess(initial_param):
    '''
    Checks initial guesses for polynomial (and LEgendre) tranformations
    '''
    if initial_param==None:
            return  np.zeros(3)
    assert len(initial_param) == 3
    return  initial_param
      
def four_param(x,y,x_ref,y_ref):
    '''
    calulates the 4 parameter tranfrom between the inputs
    does not weight the fit

    returns two vecotors, with correct fromat to be the intitial guesses
    want to solve for four parameters
    equations
    x' = a0 + a1*x + a2 * y
    y' = b0 + -a2 * x + a1 * y

    Add in matrix notation of exactly what is going in here
    x'_0      x_0  y_0  1  0        a1
    y'_0      y_0  -x_0 0  1        a2 
    x'_1      x_1  y_1  1  0     *  a0
    y'_1   =  y_1  x_1  0  1        b0

    Above is the first 4 line of the matrix equation, the LHS and matrix with the coordiantes set the pattern that contines through te entire list of coordinates
    To solve, I take the psuedo inverse of  the coordinate matrix, and then take the dor product of coo_mat^-1 * LHS to give the tranformation coefficients

    As a final step, I recalucate the translation terms based on fixed value of a1 and a2, as the translatoin term is prone to numerical error
    '''

    mat_ref = []
    mat_coo = []
    for i in range(len(x_ref)):
        mat_ref.append(x_ref[i])
        mat_ref.append(y_ref[i])

        mat_coo.append([x[i],y[i],1,0])
        mat_coo.append([y[i],-1.0*x[i],0,1])


    imat = np.linalg.pinv(mat_coo)

    trans = np.dot(imat, mat_ref)
    #print 'trans all', trans
    #print 'first guess x', np.array([trans[2],trans[0],trans[1]])
    #print 'first guess y', np.array([trans[3],trans[1],trans[0]])

    #before returning, recalculate the offsets (a0,b0), as they are susceptible to numberical error
    #we take the linear terms as constant, and compute the average offset
    #a0 = x' - a1*x - a2*y
    #b0 = y' +a2 * x - a1 *y
    a0 = np.mean(x_ref - trans[0] * x - trans[1]*y)
    b0 = np.mean(y_ref + trans[1] *x - trans[0] *y)
    
    #returning the reaulting coeficient to match fitting
    #[x0,a1,a2], [y0,-a2,a1], should be applied ot x,y vector
    
    return np.array([a0,trans[0],trans[1]]), np.array([b0,-1.0*trans[1],trans[0]])
