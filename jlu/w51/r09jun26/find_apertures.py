from pyraf import iraf as ir
import pyfits
import math
import asciidata
import numpy as np
import pylab as py
import pickle, glob
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from jlu.nirc2.photometry import run_phot

# These are the final values computed through the
# various analyses that are in this python code below.
# Other codes will call find_apertures and use the
# final resulting values shown below.
sky_annulus_start = {'h': 200, 'kp': 170, 'lp2': 100}
sky_annulus_width = {'h': 50, 'kp': 50, 'lp2': 200}
standard_aperture = {'h': 160, 'kp': 130, 'lp2': 20}


def all_curve_of_growth():
    stars = ['fs140', 'fs147', 'fs148']
    filters = ['h', 'kp', 'lp2']

    for star in stars:
        for filt in filters:
            curve_of_growth(star, filt)

def curve_of_growth(star, filter):
    """
    Make a plot of the radius of growth for each of the stars.
    Plot all the associated images for that star on top of each other.
    """
    dataDir = '/u/jlu/data/w51/09jun26/clean/'
    dir = star + '_' + filter

    images = glob.glob(dataDir + dir + '/c????.fits')
    imageRoots = [ii.replace('.fits', '') for ii in images]

    aperture = range(10, 201, 5)

    if filter == 'h':
        sky_a = 200
        sky_da = 50

    if filter == 'kp':
        sky_a = 170
        sky_da = 75

    if (filter == 'lp2'):
        aperture = range(10, 101, 5)
        sky_a = 100
        sky_da = 200

    mag = np.zeros((len(aperture), len(imageRoots)), dtype=float)
    merr = np.zeros((len(aperture), len(imageRoots)), dtype=float)
    diff = np.zeros((len(aperture), len(imageRoots)), dtype=float)

    for ii in range(len(imageRoots)):
        (r, f, m, me) = run_phot(imageRoots[ii], silent=True,
                                 apertures=aperture,
                                 sky_annulus=sky_a, sky_dannulus=sky_da)

        mag[:,ii] = m
        merr[:,ii] = me
        diff[1:,ii] = mag[1:,ii] - mag[:-1,ii]

    # Clean up arrays
    idx = np.where(mag == 0)
    mag[idx] = np.nan
    merr[idx] = np.nan
    diff[idx] = np.nan
        
    py.figure(2, figsize=(6,8))
    py.clf()
    py.subplots_adjust(left=0.15, bottom=0.08, top=0.95, right=0.95)

    py.subplot(2, 1, 1)
    for ii in range(len(imageRoots)):
        py.plot(aperture, mag[:,ii], '--')
    py.plot(aperture, mag.mean(axis=1), 'k-', linewidth=2)
    py.xlabel('Aperture (pix)')
    py.ylabel('Magnitude')
    py.title('Star = %s, Filter = %s, Sky = %d + %d pix' % 
             (star, filter, sky_a, sky_da))
    py.xlim(aperture[0], aperture[-1])

    py.subplot(2, 1, 2)
    for ii in range(len(imageRoots)):
        py.plot(aperture[1:], diff[1:,ii], '--')
    py.plot(aperture[1:], diff[1:,:].mean(axis=1), 'k-', linewidth=2)
    lims = py.axis()
    py.plot([lims[0], lims[1]], [0, 0], 'b-')
    py.xlabel('Aperture (pix)')
    py.ylabel('Delta-Magnitude')
    py.xlim(aperture[0], aperture[-1])

    py.savefig('curve_of_growth_' + star + '_' + filter + '.png')
    
        

def test_sky(dataDir, suffix):
    images = glob.glob(dataDir + 'c????.fits')
    imageRoots = [ii.replace('.fits', '') for ii in images]

    # Use a fixed aperture and vary the sky annuli
    aperture = range(10, 201, 5)
    sky_a = range(150, 251, 20)
    sky_da = range(25, 101, 25)

    mag = np.zeros((len(aperture), len(sky_a), len(sky_da)), dtype=float)
    merr1 = np.zeros((len(aperture), len(sky_a), len(sky_da)), dtype=float)
    merr2 = np.zeros((len(aperture), len(sky_a), len(sky_da)), dtype=float)
    flux = np.zeros((len(aperture), len(sky_a), len(sky_da)), dtype=float)
    ferr = np.zeros((len(aperture), len(sky_a), len(sky_da)), dtype=float)

    aper = np.zeros((len(aperture), len(sky_a), len(sky_da)), dtype=int)
    skya = np.zeros((len(aperture), len(sky_a), len(sky_da)), dtype=int)
    skyda = np.zeros((len(aperture), len(sky_a), len(sky_da)), dtype=int)

    imcnt = np.zeros((len(aperture), len(sky_a), len(sky_da)), dtype=int)

    for sa in range(len(sky_a)):
        for sd in range(len(sky_da)):
            print 'Sky Annulus = %d + %d' % (sky_a[sa], sky_da[sd])

            iflux = np.zeros((len(imageRoots), len(aperture)), dtype=float)
            imag = np.zeros((len(imageRoots), len(aperture)), dtype=float)
            imerr = np.zeros((len(imageRoots), len(aperture)), dtype=float)
            
            # Loop through all the images and get the mean magnitude
            # and the standard deviation of the magnitude.
            for ii in range(len(imageRoots)):
                (r, f, m, me) = run_phot(imageRoots[ii], silent=True,
                                         apertures=aperture,
                                         sky_annulus=sky_a[sa], 
                                         sky_dannulus=sky_da[sd])
                iflux[ii] = f
                imag[ii] = m
                imerr[ii] = me

            for aa in range(len(aperture)):
                idx = np.where(iflux[:,aa] != 0)[0]

                flux[aa, sa, sd] = iflux[idx,aa].mean()
                ferr[aa, sa, sd] = iflux[idx,aa].std()
                mag[aa, sa, sd] = imag[idx,aa].mean()
                merr1[aa, sa, sd] = imerr[idx,aa].mean()
                merr2[aa, sa, sd] = 1.0857 * ferr[aa,sa,sd] / flux[aa,sa,sd]

                aper[aa, sa, sd] = aperture[aa]
                skya[aa, sa, sd] = sky_a[sa]
                skyda[aa, sa, sd] = sky_da[sd]

                imcnt[aa, sa, sd] = len(imageRoots)

                
    pfile = open('test_sky_' + suffix + '.dat', 'w')
    pickle.dump(flux, pfile)
    pickle.dump(ferr, pfile)
    pickle.dump(mag, pfile)
    pickle.dump(merr1, pfile)
    pickle.dump(merr2, pfile)
    pickle.dump(aper, pfile)
    pickle.dump(skya, pfile)
    pickle.dump(skyda, pfile)
    pickle.dump(imcnt, pfile)


def plot_test_sky(suffix):
    pfile = open('test_sky_' + suffix + '.dat', 'r')
    flux = pickle.load(pfile)
    ferr = pickle.load(pfile)
    mag = pickle.load(pfile)
    merr1 = pickle.load(pfile)
    merr2 = pickle.load(pfile)
    aper = pickle.load(pfile)
    skya = pickle.load(pfile)
    skyda = pickle.load(pfile)
    imcnt = pickle.load(pfile)

    # Figure out the point with the lowest RMS error.
    idx = merr2.argmin()
    print 'Minimum RMS error in magnitude: %5.2f' % merr2.ravel()[idx]
    print '   Aperture Size     = %d pix' % aper.ravel()[idx]
    print '   Sky Annulus Start = %d pix' % skya.ravel()[idx]
    print '   Sky Annulus Width = %d pix' % skyda.ravel()[idx]

    fig = py.figure()
    ax = Axes3D(fig)
    
    colors = ['red', 'yellow', 'green', 'blue']
    legendItems = []

    medianMerr2 = np.median(merr2)

    for ii in range(skyda.shape[2]):
        a = aper[:,:,ii]
        sa = skya[:,:,ii]
        me = merr2[:,:,ii]
        
        idx = np.where(me > 3.0*medianMerr2)
        me[idx] = np.nan

        ax.plot_wireframe(a, sa, me,
                          rstride=1, cstride=1, color=colors[ii])
        legendItems.append('Skyda = %d' % skyda[0,0,ii])

    ax.set_xlabel('Aperture (pix)')
    ax.set_ylabel('Sky Annulus (pix)')
    ax.set_zlabel('RMS Mag Error')
    ax.view_init(10, 70)
    ax.legend(legendItems)
    
    py.savefig('plot_test_sky_' + suffix + '.png')
    py.show()

def all_test_sky():
    dataDir = '/u/jlu/data/w51/09jun26/clean/'
    
    stars = ['fs140', 'fs147', 'fs148']
    filters = ['h', 'kp', 'lp2']

    for star in stars:
        for filt in filters:
            print 'Running ' + star + '_' + filt
            test_sky(dataDir + star + '_' + filt + '/', star + '_' + filt)

def optimal_aperture_params():
    dataDir = '/u/jlu/data/w51/09jun26/clean/'
    
    stars = ['fs140', 'fs147', 'fs148']
    filters = ['h', 'kp', 'lp2']


    for ff in range(len(filters)):
        # combined RMS err in magnitude for all standards
        merrCombo = None
        weightSum = None

        for ss in range(len(stars)):
            star = stars[ss]
            filt = filters[ff]

            pfile = open('test_sky_' + star + '_' + filt + '.dat', 'r')
            flux = pickle.load(pfile)
            ferr = pickle.load(pfile)
            mag = pickle.load(pfile)
            merr1 = pickle.load(pfile)
            merr2 = pickle.load(pfile)
            aper = pickle.load(pfile)
            skya = pickle.load(pfile)
            skyda = pickle.load(pfile)
            imcnt = pickle.load(pfile)

            weight = 1.0 / imcnt

            if merrCombo == None:
                merrCombo = merr2**2 * weight
                weightSum = weight
            else:
                merrCombo += merr2**2 * weight
                weightSum += weight

        merrCombo = np.sqrt(merrCombo / weightSum)

        idx = merrCombo.argmin()
        print '*** %s ***' % filt
        print 'Minimum RMS error in magnitude: %6.3f' % merrCombo.ravel()[idx]
        print '   Aperture Size     = %d pix' % aper.ravel()[idx]
        print '   Sky Annulus Start = %d pix' % skya.ravel()[idx]
        print '   Sky Annulus Width = %d pix' % skyda.ravel()[idx]
    

def calc_aperture_corrections():
    """
    Calculate aperture correction curves from all the standard
    stars. Store the curves (mean + errors) so that they can be 
    read in later for getting the specific aperture correction for
    a given inner and outer aperture.
    """
    stars = ['fs140', 'fs147', 'fs148']
    filters = ['h', 'kp', 'lp2']
    sky_ann = sky_annulus_start
    sky_da = sky_annulus_width
    big_ap = standard_aperture

    # Calculate Curve of Growth for each filter
    for filter in filters:
        skya = sky_ann[filter]
        skyda = sky_da[filter]
        #outer = big_ap[filter]
        outer = sky_ann[filter]

        aperture = range(5, outer)
        growthCurves = []

        # Loop through all the stars and gather up the growth-curves
        for star in stars:
            dataDir = '/u/jlu/data/w51/09jun26/clean/'
            dir = star + '_' + filter

            images = glob.glob(dataDir + dir + '/c????.fits')
            imageRoots = [ii.replace('.fits', '') for ii in images]

            apCnt = len(aperture)
            imCnt = len(imageRoots)

            mag = np.zeros((apCnt, imCnt), dtype=float)
            merr = np.zeros((apCnt, imCnt), dtype=float)
            diff = np.zeros((apCnt, imCnt), dtype=float)

            for ii in range(len(imageRoots)):
                # run_phot can only handle 100 apertures at a time
                for aa in range(int(math.ceil(apCnt / 100.))):
                    apStart = aa*100
                    apStop = (aa+1)*100

                    if (apStop > apCnt):
                        apStop = apCnt

                    (r, f, m, me) = run_phot(imageRoots[ii], silent=True,
                                             apertures=aperture[apStart:apStop],
                                             sky_annulus=skya, 
                                             sky_dannulus=skyda)
                    mag[apStart:apStop,ii] = m
                    merr[apStart:apStop,ii] = me
                    
                diff[1:,ii] = mag[1:,ii] - mag[:-1,ii]


                # Cut it out if it is really really discrepant.
                # In our case, discrepant means a delta-mag in the first
                # differenced bin of < -0.3
                if (diff[:,ii].min() < -0.25):
                    continue
                
                # Clean up arrays
                idx = np.where(m == 0)
                mag[idx,ii] = np.nan
                merr[idx,ii] = np.nan
                diff[idx,ii] = np.nan

                growthCurves.append(diff[:,ii])
            
        ####################
        # All the growth curves for this filter (combine the stars)
        ####################
        growthCurves = np.array(growthCurves)
        aperture = np.array(aperture)

        py.clf()
        for ii in range(len(growthCurves)):
            py.plot(aperture[1:], growthCurves[ii,1:] , '--')
        py.plot(aperture[1:], growthCurves[:,1:].mean(axis=0), 'k-', 
                linewidth=2)
        lims = py.axis()
        py.plot([lims[0], lims[1]], [0, 0], 'b-')
        py.xlabel('Aperture (pix)')
        py.ylabel('Delta-Magnitude')
        py.xlim(aperture[0], aperture[-1])
        py.ylim(-0.25, 0.01)
        py.savefig('curve_of_growth_' + filter + '.png')
        
        ##########
        # Save off to a file for later retrieval
        ##########
        pfile = open('curve_of_growth_' + filter + '.dat', 'w')
        pickle.dump(aperture, pfile)
        pickle.dump(growthCurves, pfile)
        pfile.close()
        
def getGrowthCurve(filter, dir='./'):
    pfile = open(dir + 'curve_of_growth_' + filter + '.dat', 'r')
    apertures = pickle.load(pfile)
    growthCurves = pickle.load(pfile)

    return (apertures, growthCurves)
