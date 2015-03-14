import numpy as np
from astropy.io import fits 



def find_star(coo_n, coo_x, coo_y, psf_file, im_size=2024, scale=0.01998):
    '''
    takes coordinate star, sets reference frame
    returns x,y,,name of psf stars in the frame
    '''
    

    names, xa, ya = read_psf(psf_file, scale)
    xpix = xa + coo_x
    ypix = ya + coo_y

    found  = False
    while found = False:


    return name, xpix, ypix
        
    
def read_psf(psf_file, scale=.01998)
    '''
    takes name of a file of psf stars and returns the starname, x and y coordinates
    '''

    name, xpix, ypix = np.loadtxt(psf_file, unpack = True)
    scale_inverse = 1.0/scale
    xpix = xpix**scale_inverse
    ypix = ypix**scale_inverse
    
    
    return name, xpix, ypix
    pass



def read_coo(coo_file):
    '''
    reads in coo file, retunrs name, x,y, of coo star
    '''
    coo_x,coo_y,coo_n = np.loadtxt(coo_file)

    returne coo_n, coo_x, coo_y
    pass

def fwhm(x,y,im,box_size=20):

    '''
    makes images cut out of size box_size, centered at x, y
    returns FWHM
    '''

    pass


def  fwhm_list(frame_lis, psf_file):
    '''
    goes through list of frames and
    '''

    pass

def read_image(imagefile)
    '''
    reads in a fits image, returns array with data values
    '''

    fitsfile = fits.open(imagefile)
    
    image_array = fitsfile[1].data

    fitsfile.close()

    return image_array

def plot_stars(imagefile, psf_file, coo_file, scale = 0.01998):
    '''
    plots an image with the psf stars circled
    '''

    image_array = read_image(imagefile)
    coo_x, coo_y, coo_n = read_coo(coo_file)
    name, xpix, ypix = find_star(coo_n, coo_x, coo_y, psf_file, im_size, scale)

    
