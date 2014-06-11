import asciidata
import pyfits
import pylab
import numpy

def stackAnimation():
    """
    Take the individaul exposures from a speckle stack and make one image
    of each one. This way we can see the speckles in motion.
    """
    #workDir = '/u/ghezgroup/public_html/gc/images/media/speckle_anim/'
    workDir = ''
    fitsFile = workDir + 'data/sr02036.fits'

    images = pyfits.getdata(fitsFile)

    pylab.figure(2, figsize=(3,3), facecolor='black')
    pylab.clf()
    pylab.subplots_adjust(left=0, right=1, bottom=0, top=1)
    
    for ii in range(images.shape[0]):
        print '%4d out of %4d' % (ii, images.shape[0])

        img = images[ii,:,:]

        pylab.clf()
        pylab.subplot(111, axisbg='k')
        pylab.imshow(img, cmap=pylab.cm.hot, 
                     vmin=(img.min()*0.8), vmax=(img.max()*0.4))
        pylab.gca().set_axis_bgcolor('black')
        pylab.axis([75, 275, 100, 300])
        pylab.savefig(workDir + 'images/image' + str(ii).zfill(4) + '.png')
