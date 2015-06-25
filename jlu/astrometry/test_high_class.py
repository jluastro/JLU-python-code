from jlu.astrometry import high_order_class as high_order
import numpy as np
from astropy.table import Table
from jlu.gc.gcwork import starset
import pytest
import matplotlib.pyplot as plt





def test_a2():
    '''
    takes starset that is assumed to be linear transfomration
    align a2 uses a four parameter tranform, so for fair comparrison, we use the four parmeter tranform that is implemented as an intitial guess
    then we can compare both output coordiantes of aligna and of high_order
    '''

    
    dat = Table.read('test/data.txt', format='ascii.fixed_width')
    trans = Table.read('test/a2.trans', format='ascii')
    x = dat['xorig']
    y = dat['yorig']
    xref = dat['xref']
    yref = dat['yref']
    xa2 = dat['xa2']
    ya2 = dat['ya2']
    
    c_x, c_y = high_order.four_param(x,y,xref,yref)
    xev = c_x[0] + c_x[1] * x + c_x[2] * y
    yev = c_y[0] + c_y[1] * x + c_y[2] * y
    
    
    #test that the coefficients agree to within 1% for the large linear term, and 2% for the smaller linear term (the linear y term for the s fit)
    assert (c_x[0] - trans['a0']) /c_x[0] < .01
    assert (c_x[1] - trans['a1']) /c_x[0] < .01
    assert (c_x[2] - trans['a2']) /c_x[0] < .01

    assert (c_y[0] - trans['b0']) /c_x[0] < .01
    assert (c_y[1] - trans['b2']) /c_x[0] < .01
    assert (c_y[2] - trans['b1']) /c_x[0] < .01
    

    #demand that the output coordinates agree to within .2 pixels
    assert np.mean(np.abs(xev-xa2)) < .2, np.mean(np.abs(xev-xa2))
    assert np.mean(np.abs(yev-ya2)) < .2,  np.mean(np.abs(yev-ya2)) 

def test_a3():
    '''
    align a3 uses independent linear fits, equivalent to align
    then we can compare both output coordiantes and tranformation coefficients
    '''

    
    dat = Table.read('test/data.txt', format='ascii.fixed_width')
    trans = Table.read('test/a3.trans', format='ascii')
    x = dat['xorig']
    y = dat['yorig']
    xref = dat['xref']
    yref = dat['yref']
    xa3 = dat['xa3']
    ya3 = dat['ya3']

    #independent linear fit, with no initial guess
    t = high_order.PolyTransform(x,y,xref,yref,1)
    xev, yev = t.evaluate(x,y)
   
    
    
    #test that the coefficients agree to within 1% for the large linear term, and 2% for the smaller linear term (the linear y term for the s fit)
    assert (t.px.c0_0 - trans['a0']) /t.px.c0_0 < .01
    assert (t.px.c1_0 - trans['a1']) /t.px.c1_0 < .01
    assert (t.px.c0_1 - trans['a2']) /t.px.c0_1 < .02

    assert (t.py.c0_0 - trans['b0']) /t.py.c0_0 < .01
    assert (t.py.c1_0 - trans['b2']) /t.py.c1_0 < .02
    assert (t.py.c0_1 - trans['b1']) /t.py.c0_1 < .01
    

    #demand that the output coordinates agree to within .1 pixels
    assert np.mean(np.abs(xev-xa3)) < .1, np.mean(np.abs(xev-xa3))
    assert np.mean(np.abs(yev-ya3)) < .1,  np.mean(np.abs(yev-ya3)) 

def test_a4():
    '''
    compares align results to my code, using equivalent tranformations
    test data is from alignment of one frame of gsaoi to HST
    align a4 uses a 2nd order independent polynomial tranformation
    '''

    dat = Table.read('test/data.txt', format='ascii.fixed_width')
    trans = Table.read('test/a4.trans', format='ascii')
    x = dat['xorig']
    y = dat['yorig']
    xref = dat['xref']
    yref = dat['yref']
    xa3 = dat['xa4']
    ya3 = dat['ya4']

    t = high_order.PolyTransform(x, y, xref, yref, 2)
    xev, yev = t.evaluate(x, y)

    #only comparing low orderterms because I am unsure as to which terms the align coefficient refer to
    
    assert (t.px.c0_0 - trans['a0']) /t.px.c0_0 < .01
    assert (t.px.c1_0 - trans['a1']) /t.px.c1_0 < .01
    assert (t.px.c0_1 - trans['a2']) /t.px.c0_1 < .02

    assert (t.py.c0_0 - trans['b0']) /t.py.c0_0 < .01
    assert (t.py.c1_0 - trans['b2']) /t.py.c1_0 < .02
    assert (t.py.c0_1 - trans['b1']) /t.py.c0_1 < .01

    assert np.mean(np.abs(xev-xa3)) < .1, np.mean(np.abs(xev-xa3))
    assert np.mean(np.abs(yev-ya3)) < .1, np.mean(np.abs(yev-ya3))
    


def test_3_param(plot=False):
    num_free_param = 3
    
    x1 = np.random.rand(10000)*4096 - 2048 
    y1 = np.random.rand(10000)*4096 - 2048 


   
    c_x = np.array([10.0 ,.95 ,-.05])
    c_y = np.array([-20.0 ,.05 ,.95])
    
    x_p = c_x[0] + c_x[1]*x1 + c_x[2]*y1
    y_p = c_y[0] + c_y[1]*x1+c_y[2]*y1

    t = high_order.PolyTransform(x1,y1, x_p,y_p,1)

    xn , yn = t.evaluate(x1,y1)
    

    #print 'Transform Coefficents (given)' 
    #print  coeff_trans_x
    #print  coeff_trans_y
    #print 'Calculated Coefficients'
    #print t.coeff_x
    #print t.coeff_y

    if plot:
        plt.figure(1)
        plt.clf()
        plt.hist(xn-x_p, histtype='step', color='red', label='x')
        plt.hist(yn-y_p, histtype='step', color='blue', label='y')
        plt.legend(loc='upper left')
        plt.show()
    
    assert np.mean(np.abs(xn-x_p))<0.1,np.mean(np.abs(xn-x_p))
    assert np.mean(np.abs(yn-y_p))<0.1, np.mean(np.abs(yn-y_p))
    
def test_leg_known(plot=False):
    '''
    test the legnedre tranform, using the actual transformation as the initial guess
    This checks that the normalization prcoedure is not causing issues, it should trivially pass unless something is very broken
    '''
    
    x1 = np.random.rand(10000)*4096 - 2048 
    y1 = np.random.rand(10000)*4096 - 2048 


   
    c_x = np.array([0 ,.95 ,-.05])
    c_y = np.array([0 ,.05 ,.95])


    x_p = c_x[0] + c_x[1]*x1 + c_x[2]*y1
    y_p = c_y[0] + c_y[1]*x1+c_y[2]*y1


    t = high_order.LegTransform(x1,y1, x_p,y_p,1, init_gx=c_x, init_gy=c_y)

    xn , yn = t.evaluate(x1,y1)
    

    #print 'Transform Coefficents (given)' 
    #print  coeff_trans_x
    #print  coeff_trans_y
    #print 'Calculated Coefficients'
    #print t.coeff_x
    #print t.coeff_y

    if plot:
        plt.figure(1)
        plt.clf()
        plt.hist(xn-x_p, histtype='step', color='red', label='x')
        plt.hist(yn-y_p, histtype='step', color='blue', label='y')
        plt.legend(loc='upper left')
        plt.show()
    print np.mean(np.abs(xn-x_p))
    print np.mean(np.abs(yn-y_p))
    assert np.mean(np.abs(xn-x_p))<0.1,np.mean(np.abs(xn-x_p))
    assert np.mean(np.abs(yn-y_p))<0.1, np.mean(np.abs(yn-y_p))
    
def test_leg_4_param(plot=False):
    '''
    test the legnedre tranform,using a simple translation and rotation and plate scale tranformation
    The fitting code is given no initial guess
    '''
    
    x1 = np.random.rand(10000)*4096 - 2048 
    y1 = np.random.rand(10000)*4096 - 2048 


   
    c_x = np.array([30 ,.95 ,-.05])
    c_y = np.array([-20 ,.05 ,.95])


    x_p = c_x[0] + c_x[1]*x1 + c_x[2]*y1
    y_p = c_y[0] + c_y[1]*x1+c_y[2]*y1


    t = high_order.LegTransform(x1,y1, x_p,y_p,1 )

    xn , yn = t.evaluate(x1,y1)
    

    #print 'Transform Coefficents (given)' 
    #print  coeff_trans_x
    #print  coeff_trans_y
    #print 'Calculated Coefficients'
    #print t.coeff_x
    #print t.coeff_y

    if plot:
        plt.figure(1)
        plt.clf()
        plt.hist(xn-x_p, histtype='step', color='red', label='x')
        plt.hist(yn-y_p, histtype='step', color='blue', label='y')
        plt.legend(loc='upper left')
        plt.show()
    print np.mean(np.abs(xn-x_p))
    print np.mean(np.abs(yn-y_p))
    assert np.mean(np.abs(xn-x_p))<0.1,np.mean(np.abs(xn-x_p))
    assert np.mean(np.abs(yn-y_p))<0.1, np.mean(np.abs(yn-y_p))



def test_periodic(pix_sig=0.0,  amp=2):


    #tol is the average difference in pixels between the fitted coordinates and the calculated ones
    tol = .1
    x1 = (np.random.rand(10000) -.5 ) * 2048.0
    y1 = (np.random.rand(10000) -.5) * 2048.0
    
    '''
    use Egg Carton A**2 * sin(2pi X/L1) cos(2pi Y/L2)
    '''

    tot_dat_ran = np.max(x1) - np.min(x1)
    L1 = tot_dat_ran / 5
    L2 = tot_dat_ran / 4

    Zx = amp**2 * np.cos(2 * np.pi * x1 / L1) * np.cos(2 * np.pi * y1 / L2)

   
    L1 = tot_dat_ran / 2.0
    L2 =  tot_dat_ran / 3.0

    Zy = amp**2 * np.cos(2 * np.pi * x1 / L1) * np.cos(2 * np.pi * y1 / L2)
    

    #import pdb; pdb.set_trace()
    xref = x1 + pix_sig * np.random.randn(len(x1)) + Zx 
    yref = y1 + pix_sig * np.random.randn(len(x1)) + Zy

    plt.figure(1)
    plt.clf()
    plt.hist(x1-xref, histtype='step',bins=100, color='red')
    plt.hist(y1-yref, histtype='step',bins=100, color='blue')
    
    

    t = high_order.PolyTransform(x1,y1,xref,yref, 1)

    xev , yev = t.evaluate(x1,y1)

    assert not np.mean(np.abs(xev-xref)) < tol
    assert not np.mean(np.abs(yev-yref)) < tol
    print np.std(x1-xref)
    print np.std(y1-yref)
    print np.std(xev-xref)
    print np.std(yev-yref)
    print np.sum(xev-xref)/len(xref)
    print np.sum(y1-yref)/len(yref)
    print np.sum(yev-yref)/len(yref)

def test_four_param():
    '''
    tests the intial guess, which fits for rotation, scale , and translation terms
    '''

    x = np.random.rand(1000)
    y = np.random.rand(1000)


    #now create translated coordiantes with known translation

    c_x = np.array([30,1,.1])
    c_y = np.array([-40,-.1,1])

    xp = c_x[0] + c_x[1]*x + c_x[2]*y
    yp = c_y[0] + c_y[1]*x + c_y[2]*y


    c_xn, c_yn = high_order.four_param(x,y,xp,yp)

        

    print c_x, c_xn
    print c_y, c_yn
    assert np.sum(np.abs(c_xn - c_x)) < .001
    assert np.sum(np.abs(c_yn - c_y)) < .001
    


def test_spline():
    '''
    Simple test of pline transformation on scale, rotation and translatoin tranformation
    THis ensures that it is at least functional, even though this functional form is not ideal for fitting
    '''

    x = np.random.rand(1000)
    y = np.random.rand(1000)


    c_x = np.array([5,1,.1])
    c_y = np.array([-35,-.1,1])

    xp = c_x[0] + c_x[1]*x + c_x[2]*y
    yp = c_y[0] + c_y[1]*x + c_y[2]*y

    t = high_order.SplineTransform(x, y, xp, yp)
    xev, yev, = t.evaluate(x,y)

    x_resid = np.mean(np.abs(xev-xp))
    y_resid = np.mean(np.abs(yev-yp))
    print 'x resid :', x_resid
    print 'y resid :', y_resid
    assert x_resid < .01
    assert y_resid < .01
