import numpy as np
import pylab as py
import scipy.ndimage as snd

def read_pinholes():
    img = np.loadtxt('/Users/jlu/work/ao/imaka/20140507_imaka_UH_x0y3_78amin.txt')

    threshold = img.mean() + 2.0 * img.std()
    labels, num = snd.label(img > threshold, np.ones((3,3)))
    centers = snd.center_of_mass(img, labels, range(1, num+1))

    x = np.array(centers)[:,0]
    y = np.array(centers)[:,1]

    print x
    print y

    py.clf()
    py.imshow(img)
    py.plot(x, y, 'rx', markersize=10)
    py.xlim(-0.5, 20)
    py.ylim(-0.5, 20)
    
