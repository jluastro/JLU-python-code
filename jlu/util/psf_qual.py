import numpy as np
from astropy.io import fits
import matplotlib.pyplot as plt
from astropy.table import Table 




def read_psf(psf_file, trim_psf=True):
    '''                                                                                                                                                                      
    takes name of a file of psf stars and returns the starname, x and y coordinates                                                                                          
    '''

    psf = Table.read(psf_file, format='ascii')

    if trim_psf:
        return psf['Name'][psf['PSF?']<1.5], psf['Xarc'][psf['PSF?']<1.5], psf['Yarc'][psf['PSF?']<1.5]
    else:
        return psf['Name'], psf['Xarc'], psf['Yarc']



def read_coo(coo_file):
    '''                                                                                                                                                                      
    reads in coo file, retunrs name, x,y, of coo star                                                                                                                        
    '''

    f = open(coo_file)
    ll = f.read()
    l = ll.split('  ')
    f.close()

    coo_x = float(l[0])
    coo_y = float(l[1])
    coo_n = l[2].replace('\n','')


    return  coo_n, coo_x, coo_y
    

def find_stars(imagefile, psf_file, coo_file, scale=0.01998, trim_psf=True):
    '''
    takes iamge, psf list, coo file and returns list of x, y of psf stars in image coordinates
    '''

    im = read_image(imagefile)
    xl = im.shape[1]
    yl = im.shape[0]

    name, xpsf, ypsf = read_psf(psf_file, trim_psf=trim_psf)
    nc, xcoo , ycoo = read_coo(coo_file)

    for i,n in enumerate(name):
        if n == nc:
            xcpsf = xpsf[i]
            ycpsf = ypsf[i]
            
    
    xpsfim = (xpsf-xcpsf) / scale  + xcoo
    ypsfim = (ypsf-ycpsf) / scale + ycoo

    imbool = (xpsfim > 0) * (xpsfim < xl) * (ypsfim > 0) * (ypsfim < yl) 
    xim = xpsfim[imbool]
    yim = ypsfim[imbool]
    nim = name[imbool]


    return xim, yim, nim

def box(ar, x,y, box_size =200):
    '''
    returns slice of input array centered on x,y with sixe of boxsize
    '''
    
    yl = y -box_size/2.0
    yu = y + box_size/2.0
    xl = x - box_size/2.0
    xu = x + box_size/2.0
    if yl  < 0:
        yl = 0
    if yu > ar.shape[0]:
        yu = ar.shape[0]
    if xl < 0:
        xl = 0
    if xu < ar.shape[1]:
        xu = ar.shape[0]
    
    return ar[yl:yu,xl:xu]
   
    
def plot_cut(imagefile, psf_file, coo_file):

    xim, yim, nim = find_stars(imagefile, psf_file, coo_file)
    im = read_image(imagefile)
    
    
    for i in range(len(xim)):
        cut = box(im, xim[i], yim[i])
        plt.set_cmap('gray_r')
        plt.imshow(np.log10(cut), vmin=0,vmax=np.log10(20000))
        plt.colorbar()
        plt.show()
        

def plot_stars(imagefile, psf_file, coo_file, trim_psf=True):

    xim, yim, nim = find_stars(imagefile, psf_file, coo_file, trim_psf=trim_psf)
    im = read_image(imagefile)

    plt.set_cmap('gray_r')
    plt.imshow(np.log10(im), vmin=0,vmax=np.log10(10000))
    plt.colorbar()

    plt.title(imagefile)
    plt.scatter(xim,yim, s=80, facecolors='none', edgecolors='g')
    for i in range(len(xim)):
        plt.text(xim[i],yim[i], nim[i])

    plt.show()

def plot_all_stars(imlist, psf_file):

    for i in imlist:
        plot_stars(i,psf_file, i.replace('.fits','.coo'))
    

def plot_all_cut(imlist, psf_file):

    for i in imlist:
        plot_cut(i,psf_file, i.replace('.fits','.coo'))
    
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

def read_image(imagefile):
    '''                                                                                                                                                                      
    reads in a fits image, returns array with data values                                                                                                                    
    '''

    fitsfile = fits.open(imagefile)

    image_array = fitsfile[0].data

    fitsfile.close()

    return image_array

