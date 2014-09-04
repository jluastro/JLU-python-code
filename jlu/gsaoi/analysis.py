import pyfits
import numpy as np
import pylab as py
import os, glob, shutil
import scipy.ndimage
from jlu.util import img_scale
from jlu.util import psf

def get_sub_image(image, boxsize, x0, y0):
    boxHalfSize = int(round(boxsize / 2.0))

    xLo = x0 - boxHalfSize
    xHi = x0 + boxHalfSize
    yLo = y0 - boxHalfSize
    yHi = y0 + boxHalfSize

    imgSub = image[yLo:yHi, xLo:xHi]

    return imgSub, xLo, yLo

def key_press(event):
    if event.key == 'p' or event.key == 'P':
        psfName = raw_input('Name of PSF star you will select:')

        event.canvas.figure.gca().set_xlabel('{0}'.format(psfName))
        
    return

def coo_max(files, psfName, manual=True):
    py.close(1)
    py.figure(1, figsize=(10,10))
    py.subplots_adjust(left=0.1, bottom=0.08, right=0.95, top=0.93)

    firstFramePos = None

    for _file in files:
        fileDir, fileName = os.path.split(_file)
        fileRoot, fileExt = os.path.splitext(fileName)

        # Make a max file
        _max = open(fileDir + fileRoot + '.max', 'w')
        _max.write('35000\n')
        _max.close()

        # Read in the image for display to select a coo star
        img, hdr = pyfits.getdata(_file, header=True)
        imgRescale = img_scale.sqrt(img, scale_min=400, scale_max=5000)

        fig = py.figure(1)
        foo = fig.gca().get_xlabel()
        if foo != '':
            psfName = foo
            
        print 'Getting from old plot:', psfName
        py.clf()
        py.imshow(imgRescale, aspect='equal', cmap=py.cm.gray)
        py.title('{0} PA={1:.0f}'.format(fileRoot, hdr['PA']))
        fig.gca().set_xlabel('{0}'.format(psfName))
        print 'Plotting: ', psfName
        py.ylabel('Press "p" for new star name.')
        fig.canvas.mpl_connect('key_press_event', key_press)

        starFound = False

        while not starFound:
            psfName = fig.gca().get_xlabel()
            print 'In while: ', psfName

            # Get the user selected PSF stars
            if firstFramePos == None or manual == True:

                pts = py.ginput(1, timeout=0)
                xinit = int(round(pts[0][0]))
                yinit = int(round(pts[0][1]))
            else:
                xinit = firstFramePos[0]
                yinit = firstFramePos[1]

            boxSize = 50

            # Get an initial sub-image
            imgSub, xLo, yLo = get_sub_image(img, boxSize, xinit, yinit)

            # Find the maximum pixel value within this box...
            # assume this is the star and re-center up on it
            maxIdx = np.where(imgSub == imgSub.max())
            xinit = xLo + maxIdx[1][0]
            yinit = yLo + maxIdx[0][0]

            # Get an sub-image centered on brightest pixel
            imgSub, xLo, yLo = get_sub_image(img, boxSize, xinit, yinit)

            yc, xc = scipy.ndimage.center_of_mass(imgSub)
            xc += xLo
            yc += yLo
            print '%s  Centroid:  x=%.2f  y=%.2f' % (fileRoot, xc, yc)

            py.figure(2)
            py.clf()
            py.imshow(imgRescale, aspect='equal', cmap=py.cm.gray)
            py.plot([xc], [yc], 'kx', ms=10)
            py.xlim(xc - 30, xc + 30)
            py.ylim(yc - 30, yc + 30)
            py.figure(1)

            gaussInfo = psf.moments(imgSub, SubSize=5)
            g_height = gaussInfo[0]
            g_muX = gaussInfo[1]
            g_muY = gaussInfo[2]
            g_FWHMX = gaussInfo[3]
            g_FWHMY = gaussInfo[4]
            g_FWHM = gaussInfo[5]
            g_Ellip = gaussInfo[6]
            g_angle = gaussInfo[7]

            hdrout = '{0:5s} {1:5s} {2:5s} {3:5s} {4:5s} '
            hdrout += '{5:5s} {6:4s} {7}\n'
            strout = '{0:5.2f} {1:5.2f} {2:5.2f} {3:5.2f} {4:5.2f} '
            strout += '{5:5.2f} {6:4.2f} {7:.1f}\n'
            
            _gauss = open(fileDir + fileRoot + '.metrics', 'w')
            _gauss.write(hdrout.format('muX', 'muY', 'FWHMX', 'FWHMY', 'FWHM',
                                       'Ellip', 'Angle', 'Height'))
            _gauss.write(strout.format(g_muX, g_muY, g_FWHMX, g_FWHMY, g_FWHM,
                                       g_Ellip, g_angle, g_height))
            _gauss.close()


            # Make a max file
            _coo = open(fileDir + fileRoot + '.coo', 'w')
            _coo.write('{0:.2f}  {1:.2f}  {2}\n'.format(xc, yc, psfName))
            _coo.close()

            if firstFramePos == None:
                firstFramePos = [xc, yc]

            starFound = True

    return

def stf(date, field):
    """
    Date should be the root of the file name (e.g. '20111215' or '20111218')
    Field should be the field name (e.g. 'G1_1')
    """
    stfDir = 'starfinder/'

    files = glob.glob(stfDir + 'S' + date + '*_' + field + '_cr.fits')

    _idl = open('idlbatch_' + date + '_' + field, 'w')
    _idl.write('.r psf_extract\n')

    for _file in files:
        fileDir, fileName = os.path.split(_file)
        fileRoot, fileExt = os.path.splitext(fileName)

        # Get max file to pass in
        maxFile = fileDir + '/' + fileRoot + '.max'

        # Get coo star name to pass in
        cooFile = fileDir + '/' + fileRoot + '.coo'
        _coo = open(cooFile, 'r')
        cooLine = _coo.readline()
        cooStuff = cooLine.split()
        cooName = cooStuff[2]

        cmd = 'find_stf_gsaoi, "{0}", 0.9, year=2012.95, '.format(_file)
        cmd += 'trimfake=0, /makePsf, /makeRes, /makeStars, /makeRep, /quick, '
        cmd += 'cooStar="{0}", '.format(cooName)
        cmd += 'starlist="psf_stars.dat", '
        cmd += 'psfSize=1.0, psfSearchBox=1.0, secSearchBox=1.0\n'

        print cmd
        _idl.write(cmd)

    _idl.close()

    print '***********'
    print ' Run with:'
    print ''
    print 'idl < idlbatch_{0}_{1} >& idlbatch_{0}_{1}.log'.format(date, field)
    print ''
    print '***********'

def photo_calib(date, field):
    stfDir = 'starfinder/'
    

    files = glob.glob('{0}/S{1}*_{2}_cr_0.9_stf.lis'.format(stfDir, date, field))

    # Calibrate flags
    flagStr = '-f 1 -c 13 -s 2 '
    flagStr += '-N ngc1815_photo_calib.dat -M 1 -R '

    for _file in files:
        # Fetch the name of the PSF star
        cooFile = _file.replace('_0.9_stf.lis', '.coo')
        _coo = open(cooFile, 'r')
        cooLine = _coo.readline()
        psfStar = cooLine.split()[-1]

        flagStrNow = flagStr + '-I ' + psfStar

        # Fetch the angle of the image.
        fitsFile = _file.replace('_0.9_stf.lis', '.fits')
        hdr = pyfits.getheader(fitsFile)
        angle = hdr['PA']
        angle = 360 - angle

        flagStrNow += ' -T {0:.2f}'.format(angle)

        flagStrNow += ' ' + _file

        calibrate.main(argv=flagStrNow.split())

def align(field, nite=None, pa=None):
    alignDir = 'align/'
    stfDir = 'starfinder/'
    if not os.path.exists(alignDir):
        os.mkdir(alignDir)

    niteSuff = ''
    if nite == '20121230':
        niteSuff = '_n1'
    if nite == '20121231':
        niteSuff = '_n2'

    paSuff = ''
    if pa != None:
        pa = '_pa%d' % pa

    if nite == None and pa == None:
        files = glob.glob('{0}S2012*_{1}_cr_0.9_stf_cal.lis'.format(stfDir, field))
    else:
        # Broken for now
        files = glob.glob('{0}S{1}*_{2}_cr_0.9_stf_cal.lis'.format(stfDir, nite, field))
    print files

    # Make the align.list file
    alignRoot = 'align_{0}{1}'.format(field, niteSuff)
    _list = open(alignDir + alignRoot + '.list', 'w')

    for i in range(len(files)):
        _list.write(files[i] + ' 34')
        if i == 0:
            _list.write(' ref\n')
        else:
            _list.write('\n')

    _list.close()

    # Run align
    cmd = 'java align '
    cmd += '-N ngc1815_label.dat '
    cmd += '-R 15 -a 2 -p -v '
    cmd += '-r {0}{1} {0}{1}.list'.format(alignDir, alignRoot)

    print '**********'
    print '   Run align with:'
    print ''
    print cmd
    print ''
    print '   Consider manually trimming the *.list file first to get rid of bad frames.'
    print '**********'

def align_all():
    alignDir = 'align_all/'
    stfDir = 'starfinder/'
    if not os.path.exists(alignDir):
        os.mkdir(alignDir)

    files = glob.glob('starfinder/*0.9_stf_cal.lis')

    # TRim down to starlists that were properly calibrated.
    namedCnt = check_named_stars_from_list(files, verbose=False)
    idx = np.where(namedCnt > 1)[0]
    files = [files[ii] for ii in idx]
    namedCnt = namedCnt[idx]
    

    # Make the align.list file
    alignRoot = 'align'
    _list = open(alignDir + alignRoot + '.list', 'w')

    _list.write('combo/ngc1815_stf_east_right_cal.lis 34 ref\n')

    for i in range(len(files)):
        _list.write(files[i] + ' 34\n')

    _list.close()

    # Run align
    cmd = 'java align '
    cmd += '-N ngc1815_label.dat -initGuessNamed '
    cmd += '-R 15 -a 2 -p -v '
    cmd += '-r {0}{1} {0}{1}.list'.format(alignDir, alignRoot)

    print '**********'
    print '   Run align with:'
    print ''
    print cmd
    print ''
    print '   Consider manually trimming the *.list file first to get rid of bad frames.'
    print '**********'

def align_files(fileRoots, alignDir, alignRoot):
    stfDir = 'starfinder'
    if not os.path.exists(alignDir):
        os.mkdir(alignDir)

    # Make the align.list file
    _list = open(alignDir + alignRoot + '.list', 'w')
    _list.write('combo/ngc1815_stf_east_right_cal.lis 34 ref\n')

    for i in range(len(fileRoots)):
        ff = '{0}/S{1}_cr_0.9_stf_cal.lis 34\n'.format(stfDir, fileRoots[i])
        _list.write(ff)

    _list.close()

    # Run align
    cmd = 'java align '
    cmd += '-N ngc1815_label.dat '
    cmd += '-R 15 -a 2 -p -v '
    cmd += '-r {0}{1} {0}{1}.list'.format(alignDir, alignRoot)

    print '**********'
    print '   Run align with:'
    print ''
    print cmd
    print ''
    print '   Consider manually trimming the *.list file first to get rid of bad frames.'
    print '**********'

def check_named_stars(fileSearchString='*_cal.lis', verbose=True):
    calFiles = glob.glob(fileSearchString)

    namedCnt = check_named_stars_from_list(calFiles, verbose=verbose)

    return namedCnt
    

def check_named_stars_from_list(calFiles, verbose=True):
    namedCnt = np.zeros(len(calFiles), dtype=int)

    for ii in range(len(calFiles)):
        cal = calFiles[ii]
        
        t = atpy.Table(cal, type='ascii')
        cnt = 0
        
        for tt in range(len(t)):
            if t['col1'][tt].startswith('psf'):
                cnt += 1

        if verbose:
            print cal, cnt

        namedCnt[ii] = cnt

    return namedCnt

            
    
    
