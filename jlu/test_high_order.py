import numpy as np
import high_order
import matplotlib.pyplot as plt
from matplotlib import transforms



def test_sanity():

    '''
    Simple sanity check, puts in same star list, all coeff should be zero
    '''

    x1 = np.linspace(0,2000,num=10000) - 1000
    xref = x1 

    y1 = np.linspace(0,2000,num=10000) - 1000
    yref = y1

    cx , cy = high_order.match_high(x1, x1, y1,  y1, 6)

    print cx, cy 

    


def test_transform(num_free_param=10,co_ran=.1):


    
    #x1 =  np.linspace(0,2000,num=1000) - 1000
    #x1 = np.concatenate([x1, x1, x1, x1, x1, x1, x1, x1, x1, x1])
    #y1 = np.zeros(x1.shape)
    #for i in range(10):
    #    for j in range(1000):
    #        y1[i*1000 + j] = i*200.0-1000

    #print y1

    x1 = np.random.rand(10000)*4096 - 2048 
    y1 = np.random.rand(10000)*4096 - 2048 


    coeff_trans_x = np.random.randn(num_free_param)*co_ran
    coeff_trans_y = np.random.randn(num_free_param)*co_ran

    del_x = high_order.poly(np.array([x1,y1]), coeff_trans_x) 
    del_y = high_order.poly(np.array([x1,y1]), coeff_trans_y)

    xref = x1 + del_x
    yref = y1 + del_y

    c_x, c_y = high_order.fit_poly(x1, y1, del_x,  del_y , len(coeff_trans_x))

    print 'Transform Coefficents (given)' 
    print  coeff_trans_x
    print coeff_trans_y
    print 'Calculated Coefficients'
    print c_x
    print c_y 
    return x1, y1, xref, yref 


def test_spline(pix_sig=3, order=3, num_knots=5, ret_diff=False, amp=1, ret_u=False):


    x1 = (np.random.rand(1000) -.5) * 4096 
    y1 = (np.random.rand(1000) -.5) * 4096

    x_u = (np.random.rand(500)-.5)*4096
    y_u = (np.random.rand(500)-.5)*4096

    
    knots_x = np.linspace(-2048,2048,num=num_knots)
    knots_y = np.linspace(-2048,2048,num=num_knots)

    '''
    Now give random errors to xref, yref to see if spline fitter gets same measured error
    Also use Egg Carton A**2 * sin(2pi X/L1) cos(2pi Y/L2)
    '''

    A = amp
    L1 = 40
    L2 = 17

    Zx = A**2 * np.cos(2 * np.pi * x1 / L1) * np.cos(2 * np.pi * y1 / L2)
    Zx_u = A**2 * np.cos(2 * np.pi * x_u / L1) * np.cos(2 * np.pi * y_u / L2)

    L1 = 5
    L2 = 70

    Zy = A**2 * np.cos(2 * np.pi * x1 / L1) * np.cos(2 * np.pi * y1 / L2)
    Zy_u =  A**2 * np.cos(2 * np.pi * x_u / L1) * np.cos(2 * np.pi * y_u / L2)
    

    xref = x1 + pix_sig * np.random.randn(len(x1)) + Zx 
    yref = y1 + pix_sig * np.random.randn(len(x1)) + Zy 

    x_prime, spline_x = high_order.fit_spline(x1,y1,xref,order=order, num_knots=num_knots )
    y_prime, spline_y = high_order.fit_spline(y1,x1,yref, order=order, num_knots=num_knots)

    d_x = xref - x1
    d_y = yref - y1
    d_x_m = x_prime - x1 
    d_y_m = y_prime - y1
    
    if ret_diff:
        return x_prime, y_prime, x1, y1, xref, yref, d_x, d_y, d_x_m, d_y_m
    elif ret_u:
        return spline_x.ev(x_u, y_u), spline_y.ev(x_u, y_u), x_u, y_u, x_u +Zx_u, y_u +Zy_u, 0 ,0 ,0, 0

    return x_prime, y_prime, x1, y1, xref, yref

def periodic(pix_sig=3, num_poly_param=10, num_knots=5, ret_diff=False, smooth=False, amp=.5):


    x1 = (np.random.rand(10000) -.5 ) * 2048.0
    y1 = (np.random.rand(10000) -.5) * 2048.0
    

    knots_x = np.linspace(np.min(x1),np.max(x1),num=num_knots)
    knots_y = np.linspace(np.min(y1),np.max(y1),num=num_knots)

    '''
    Now give random errors to xref, yref to see if spline fitter gets same measured error
    Also use Egg Carton A**2 * sin(2pi X/L1) cos(2pi Y/L2)
    '''

    tot_dat_ran = np.max(x1) - np.min(x1)
    L1 = tot_dat_ran / 2.0
    L2 = tot_dat_ran / 2.0

    Zx = amp**2 * np.cos(2 * np.pi * x1 / L1) * np.cos(2 * np.pi * y1 / L2)

   
    L1 = tot_dat_ran / 2.0
    L2 =  tot_dat_ran / 3.0

    Zy = amp**2 * np.cos(2 * np.pi * x1 / L1) * np.cos(2 * np.pi * y1 / L2)
    

    xref = x1 + pix_sig * np.random.randn(len(x1)) + Zx 
    yref = y1 + pix_sig * np.random.randn(len(x1)) + Zy 


    #x_prime, y_prime, spline_x, spline_y, co_x, co_y = high_order.find_fit(x1, y1, xref, yref, num_knots=num_knots, num_poly_param=num_poly_param)
    dx_sp, dx_spline = high_order.fit_spline(x1, y1, xref-x1, num_knots=num_knots, smooth=smooth)
    dy_sp, dy_spline = high_order.fit_spline(x1, y1, yref-y1, num_knots=num_knots, smooth=smooth)
    

    x_prime = x1 + dx_sp
    y_prime = y1 + dy_sp
    d_x_true = xref - x1
    d_y_true = yref - y1

    d_x_m = dx_sp 
    d_y_m = dy_sp

    if ret_diff:
        return x_prime, y_prime, x1, y1, xref, yref, d_x_true, d_y_true, d_x_m, d_y_m
    
    return x_prime, y_prime, x1, y1, xref, yref


def test_total(num_poly=10, num_knots=4, noise=0.1, smooth=False):
    '''
    Function that performs transformation between x, y and xref yref then fits that difference using polynomial fit followed by spline
    num_poly is the number of polynomial terms used, 10 gets all quadratic terms
    num_knots is th enumber of knots in the slpine along each axis, that is 4 means there is a total of 16 knots 
    noise is the sigma (in pixels) used to 
    '''


    xref = np.random.rand(10000) * 4096 - 2048
    yref = np.random.rand(10000) * 4096 - 2048 


    '''
    these star lists will be put through both the known tranformation (rotation and translation) and also put through the derived polynomial and spline fits as a check
    '''
    x_dim_ref = np.random.rand(1000) * 4096 - 2048
    y_dim_ref = np.random.rand(1000) * 4096 - 2048 

     
    
    trans = transforms.Affine2D()
    trans.rotate_deg(75.0)
    trans.translate(184, -45)
    
    cooref = np.array([xref, yref]).T
    
    coo1 = trans.transform(cooref) 
    coo_dim = trans.transform(np.array([x_dim_ref,y_dim_ref]).T)
    
    x_dim1 = coo_dim[:,0] + noise*np.random.randn(len(x_dim_ref))
    y_dim1 = coo_dim[:,1] + noise*np.random.randn(len(y_dim_ref))
    
    x1 = coo1[:,0] + noise*np.random.randn(len(xref))
    y1 = coo1[:,1] + noise*np.random.randn(len(yref))

    c_x, c_y = high_order.fit_poly(x1, y1, xref,  yref, num_poly)

    x_poly = high_order.poly(np.array([x1,y1]), c_x)
    y_poly = high_order.poly(np.array([x1,y1]), c_y)

    #if np.sum(np.abs(x_poly-xref)) < 1 and  np.sum(np.abs(y_poly-yref)) < len(xref):
    #    print 'Polynomial Fit was sufficient'
    #    return  c_x, c_y

    
    '''
    Now do spline fit between the polynomial fit and reference, to get rid of residual
    '''

    dx_sp, spline_dx = high_order.fit_spline(x_poly, y_poly, xref-x_poly, num_knots=num_knots, smooth=smooth)
    dy_sp, spline_dy = high_order.fit_spline(x_poly, y_poly, yref-y_poly, num_knots=num_knots, smooth=smooth)

    x_sp, spline_x = high_order.fit_spline(x_poly, y_poly, xref, num_knots=num_knots, smooth=smooth)
    y_sp, spline_y = high_order.fit_spline(x_poly, y_poly, yref, num_knots=num_knots, smooth=smooth)

    
    assert np.sum(np.abs(x_sp-(x_poly+dx_sp)))/len(x_poly) < noise
    assert np.sum(np.abs(y_sp-(y_poly+dy_sp)))/len(y_poly) < noise

    assert np.sum(np.abs(x_sp - xref))/len(x_sp) < noise
    assert np.sum(np.abs(y_sp - yref))/len(y_sp) < noise

    x_dim_poly = high_order.poly(np.array([x_dim1,y_dim1]), c_x)
    y_dim_poly = high_order.poly(np.array([x_dim1,y_dim1]), c_y)

    assert np.sum(np.abs(x_dim_poly - x_dim_ref)) / len(x_dim_ref) < noise
    assert np.sum(np.abs(y_dim_poly - y_dim_ref)) / len(y_dim_ref) < noise

    x_dim_sp = spline_x.ev(x_dim_poly, y_dim_poly)
    y_dim_sp = spline_y.ev(x_dim_poly,y_dim_poly)

    assert np.sum(np.abs(x_dim_sp - x_dim_ref)) / len(x_dim_ref) < noise
    assert np.sum(np.abs(y_dim_sp - y_dim_ref)) / len(y_dim_ref) < noise

    
    
    
    return x_sp, y_sp, xref, yref, spline_x, spline_y , c_x, c_y, x1, y1


def test_noise(num_knots=5, order=3, smooth=False, amp=.5, show=True):

    sig = np.linspace(0,.1,num=25)

    errx = []
    erry = []
    for i in range(len(sig)):
        x_prime, y_prime, x1, y1, xref, yref, dx_true, dy_true, dx_m, dy_m = periodic(pix_sig=sig[i], num_knots=num_knots, num_poly_param=order, ret_diff=True, amp=amp)
        errx.append(np.sum(np.abs(x_prime - xref)) / len(x_prime))
        erry.append(np.sum(np.abs(y_prime - yref)) / len(y_prime))

    plt.xlabel('Noise (pixels)')
    plt.ylabel('Average of Absolute Value of error (pixels)')
    plt.plot(sig, errx, '-o', label=str(num_knots)+' knots in x')
    plt.plot(sig,erry, '-o', label=str(num_knots)+' knots in y')
    plt.legend(loc='lower right')
    if show:
        plt.show()
        
def test_knots(knots=None, sig = .05, order=3, smooth=False, amp=.5):

    if not knots:
        knots = np.linspace(2,8, num=7)

    errx = []
    erry = []
    for i in range(len(knots)):
        x_prime, y_prime, x1, y1, xref, yref, dx_true, dy_true, dx_m, dy_m = periodic(pix_sig=sig, num_knots=knots[i], num_poly_param=order, ret_diff=True, smooth=smooth, amp=amp)
        errx.append(np.sum(np.abs(dx_true - dx_m)) / len(dx_m))
        erry.append(np.sum(np.abs(dy_true - dy_m)) / len(dy_m))

    plt.xlabel('Noise (pixels)')
    plt.ylabel('Average of Absolute Value of error (pixels)')
    plt.plot(knots, errx, '-o', label='Error in x')
    plt.plot(knots,erry, '-o', label='Error in y')
    plt.legend(loc='lower right')
    plt.show()
