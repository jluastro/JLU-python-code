import math
from scipy import fftpack
import pyfits
import numpy as np
import pylab as py
from jlu.util import radialProfile

def run():
    imageFile = '/u/jlu/data/w51/09jun26/combo/mag09jun26_w51a_f4_kp.fits'

    img = pyfits.getdata(imageFile)

    F = fftpack.fftshift( fftpack.fft2(img) )
    psd2D = np.abs(F)**2
    psd1D = radialProfile.azimuthalAverage(psd2D)
    
    py.figure(1)
    py.clf()
    py.imshow(np.log10(img), cmap=py.cm.Greys, 
              vmin=math.log10(20), vmax=math.log10(3500))
    py.title('Image')
    py.savefig('image_fft_examp_image.png')
    
    py.figure(2)
    py.clf()
    py.imshow(np.log10(psd2D), cmap=py.cm.jet)
    py.title('2D Power Spectrum')
    py.savefig('image_fft_examp_psd2d.png')
    
    py.figure(3)
    py.clf()
    py.semilogy(psd1D)
    py.title('1D Power Spectrum')
    py.xlabel('Spatial Frequency')
    py.ylabel('Power Spectrum')
    py.savefig('image_fft_examp_psd1d.png')
    
    py.show()

