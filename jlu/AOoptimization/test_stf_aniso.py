import pylab as py
import numpy as np
from gcwork import starset
import pdb
import atpy
import pyfits
import os
import shutil
import glob
import math


workDir = '/u/jlu/work/ao/ao_optimization/test_stf_aniso/'

def organize_psf_grid():
    """
    Organize the raw PSF files that Gunther provided into a single psf_cube_20_20.fits
    file and a svpart_20_20.txt file with the box boundaries.

    In the process of making a cube, we will trim the PSFs down to 100 pixel size
    to save space and speed up Starfinder runs for this test.
    """
    psfs = glob.glob(workDir + 'from_gunther/PSF/*.fits')

    xcoords = np.zeros(len(psfs), dtype=int)
    ycoords = np.zeros(len(psfs), dtype=int)

    psf_size = 100
    psf_cube = None

    # Read in all the PSF files and find their pixel positions.
    for ii in range(len(psfs)):
        psfFile = psfs[ii]
        fileRoot = os.path.splitext(os.path.basename(psfFile))[0]
        fileParts = fileRoot.split('_')

        xcoords[ii] = int(fileParts[1])
        ycoords[ii] = int(fileParts[2])

    # Use the pixel positions to determine box boundaries... pixel positions
    # should just be in the middle of the box. All boxes should be the same size.
    xuni = np.unique(xcoords)
    yuni = np.unique(ycoords)
    xuni.sort()
    yuni.sort()
    
    dx = np.diff(xuni)
    dy = np.diff(yuni)

    # Check that all the deltas are the same (square, regular grid)
    dd = np.append(dx, dy)
    idx = np.where(dd != dx[0])[0]
    if len(idx) != 0:
        print('Problem with grid spacing:')
        print(dx)
        print(dy)
        return

    psf_sep = dx[0]
    psf_sep_half = int(math.floor(psf_sep / 2))
    npsf_side = len(xuni)

    # Recall that the shape of the PSF cube is just a stack. The ordering
    # is given by:
    #     YY * NumXboxes + XX
    # where XX is the grid-box index in the X dimension and
    # where YY is the grid-box index in the Y dimension.
    # Grid-box boundaries are defined in an sv_par text file.

    # arrays for new boundaries
    lx = np.zeros(npsf_side, dtype=int)
    ux = np.zeros(npsf_side, dtype=int)
    ly = np.zeros(npsf_side, dtype=int)
    uy = np.zeros(npsf_side, dtype=int)

    # Set the box boundaries. Assumes X and Y have equal number of grid points
    # Had to do this carefully to make sure all pixels are encountered only once.
    lx = xuni - psf_sep_half
    ly = yuni - psf_sep_half
    ux[:-1] = lx[1:]
    uy[:-1] = ly[1:]
    ux[-1] = xuni[-1] + psf_sep_half
    uy[-1] = yuni[-1] + psf_sep_half

    # Fix anything less than 0 to be 0.
    lx[lx < 0] = 0
    ux[ux < 0] = 0
    ly[ly < 0] = 0
    uy[uy < 0] = 0

    # Print out the svpar_61_61.txt file.
    _txt = open(workDir + 'svpar_61_61.txt', 'w')
    fmtStr = '{lx:5d} {ux:5d} {ly:5d} {uy:5d}\n'
    for ii in range(len(lx)):
        _txt.write(fmtStr.format(lx=lx[ii], ux=ux[ii], ly=ly[ii], uy=uy[ii]))


    # Stack up all the PSFs into a cube.
    psf_cube = np.zeros((len(lx)*len(ly), psf_size, psf_size), dtype=float)
        
    # The inner loop should be X and the outer loop Y.
    for yy in range(len(ly)):
        for xx in range(len(lx)):
            pp = yy*len(lx) + xx

            psf = pyfits.getdata(psfs[ii])
            psfLo = (psf.shape[1]/2) - (psf_size/2)
            psfHi = psfLo + psf_size

            psf_cube[pp,:,:] = psf[psfLo:psfHi, psfLo:psfHi]

    pyfits.writeto('psf_cube_61_61.fits', psf_cube, clobber=True)
    

def make_psf_cube(undersample):
    """
    Read in the original 61x61 PSF cube and make a coarser grid.
    Just drop the extra PSFs. This is to test Starfinder's ability to
    recover astrometry/photometry with a coarse PSF grid when the image has a
    continuously variable PSF.

    Input:
    undersample -- 
    """
    # Read in the original boundaries
    t_orig = atpy.Table('svpar_61_61.txt', type='ascii')
    t_orig.rename_column('col1', 'lx')
    t_orig.rename_column('col2', 'ux')
    t_orig.rename_column('col3', 'ly')
    t_orig.rename_column('col4', 'uy')

    dx = t_orig.ux - t_orig.lx
    dy = t_orig.uy - t_orig.ly

    xcen = t_orig.lx + (dx / 2.0)
    ycen = t_orig.ly + (dy / 2.0)

    # Read in the original PSF cube
    psf_orig = pyfits.getdata('psf_cube_61_61.fits')
    psf_size = psf_orig.shape[1]

    # Recall that the shape of the PSF cube is just a stack. The ordering
    # is given by:
    #     YY * NumXboxes + XX
    # where XX is the grid-box index in the X dimension and
    # where YY is the grid-box index in the Y dimension.
    # Grid-box boundaries are defined in an sv_par text file.

    # Dimensions of new grid
    npsf_side = 61 / undersample
    suffix = '%d_%d' % (npsf_side, npsf_side)

    # arrays for new boundaries
    lx = np.zeros(npsf_side, dtype=int)
    ux = np.zeros(npsf_side, dtype=int)
    ly = np.zeros(npsf_side, dtype=int)
    uy = np.zeros(npsf_side, dtype=int)

    # array for new PSFs ... we will trim down each PSF to 100x100 pixels
    psf = np.zeros((npsf_side**2, psf_size, psf_size), dtype=float)

    # Set the new boundaries. Assumes X and Y have equal number of grid points
    # Had to do this carefully to make sure all pixels are encountered only once.
    for yy in range(len(npsf_side)):
        # Select the original grid boxes that are within this new box.
        yy_orig_lo = yy*undersample
        yy_orig_hi = yy_orig_lo + undersample
        yy_orig_use = yy_orig_lo + (yy_orig_hi - yy_orig_lo)/2)

        if yy_orig_hi > 61:
            yy_orig_hi = 61
        if yy_orig_use >= 61
            yy_orig_use = 61 - 1

        for xx in range(len(npsf_side)):
            # Select the original grid boxes that are within this new box.
            xx_orig_lo = xx*undersample
            xx_orig_hi = xx_orig_lo + undersample
            xx_orig_use = xx_orig_lo + (xx_orig_hi - xx_orig_lo)/2)

            if xx_orig_hi > 61:
                xx_orig_hi = 61
            if xx_orig_use >= 61
                xx_orig_use = 61 - 1

            pp = yy*npsf_side + xx
            pp_orig_use = yy_orig_use*61 + xx_orig_use
            
    # The inner loop should be X and the outer loop Y.
    for yy in range(len(t_orig.ly)):
        for xx in range(len(t_orig.lx)):
            pp = yy*len(t_orig.lx) + xx

            # Oversample indices
            for ys in range(oversample):
                for xs in range(oversample):
                    # New indices into the final output arrays
                    xi = (xx*oversample) + xs
                    yi = (yy*oversample) + ys
                    pi = yi*npsf_side + xi

        
        
    for ii in range(len(t_orig.lx)):
        dx_orig = t_orig.ux[ii] - t_orig.lx[ii]
        dy_orig = t_orig.uy[ii] - t_orig.ly[ii]
        dx_new = dx_orig / oversample
        dy_new = dy_orig / oversample

        for ss in range(oversample):
            i2 = ii*oversample + ss
            lx[i2] = t_orig.lx[ii] + ss*dx_new
            ly[i2] = t_orig.ly[ii] + ss*dy_new

    for ii in range(len(ux)-1):
        ux[ii] = lx[ii+1]
        uy[ii] = ly[ii+1]
    ux[-1] = t_orig.ux[-1]
    uy[-1] = t_orig.uy[-1]


    # The inner loop should be X and the outer loop Y.
    for yy in range(len(t_orig.ly)):
        for xx in range(len(t_orig.lx)):
            pp = yy*len(t_orig.lx) + xx

            # Oversample indices
            for ys in range(oversample):
                for xs in range(oversample):
                    # New indices into the final output arrays
                    xi = (xx*oversample) + xs
                    yi = (yy*oversample) + ys
                    pi = yi*npsf_side + xi

                    psf[pi,:,:] = psf_orig[pp,:,:]

    pyfits.writeto('psf_cube_'+suffix+'.fits', psf, clobber=True)

    _out = open('svpar_'+suffix+'.txt', 'w')
    for ii in range(len(lx)):
        _out.write('%4d %4d %4d %4d\n' % (lx[ii], ux[ii], ly[ii], uy[ii]))
    _out.close()

def run_starfinder(oversample):
    # Dimensions of new grid
    npsf_side = 8 * oversample
    suffix = '%d_%d' % (npsf_side, npsf_side)

    try:
        os.makedirs('grid_' + suffix)
    except OSError:
        pass

    idlRoot = 'grid_' + suffix + '/idlrun_'+suffix

    _idl = open(idlRoot, 'w')

    _idl.write('.r find_stf_tmt_grid\n')
    cmd = 'find_stf_tmt_grid, "image.fits", "psf_cube_%s.fits", "svpar_%s.txt", 0.9\n' % \
        (suffix, suffix)
    _idl.write(cmd)
    _idl.close()

    cmd = 'idl < %s >& %s.log' % (idlRoot, idlRoot)
    os.system(cmd)

    # Copy the output starlist
    shutil.copy('image_0.9_stf.lis', 'grid_' + suffix + '/image_'+suffix+'_0.9_stf.lis')

def run_calibrate(oversample):
    # Dimensions of new grid
    npsf_side = 8 * oversample
    suffix = '%d_%d' % (npsf_side, npsf_side)

    cmd = 'calibrate_new -N /u/jlu/data/gc/source_list/photo_calib.dat -c 12 -R -M 1 '
    cmd += 'grid_' + suffix + '/image_'+suffix+'_0.9_stf.lis'

    os.system(cmd)

def run_align(oversample):
    # Dimensions of new grid
    npsf_side = 8 * oversample
    suffix = '%d_%d' % (npsf_side, npsf_side)

    _list = open('grid_' + suffix + '/align_' + suffix + '.list', 'w')
    _list.write('image_input_positions.lis 30\n')
    _list.write('grid_' + suffix + '/image_' + suffix + '_0.9_stf_cal.lis 30 ref\n')
    _list.close()

    cmd = 'java align -v -p -a 2 -R 15 -N /u/jlu/data/gc/source_list/label.dat'
    cmd += ' -r grid_' + suffix + '/align_' + suffix
    cmd += ' grid_' + suffix + '/align_' + suffix + '.list '

    os.system(cmd)

def run_all(oversample):
    """
    Run starfinder, align resulting list with input positions, make plot
    for image.fits with an oversampled PSF. You can always have ovserample=1.
    """
    print 'Making PSF cube'
    make_psf_cube(oversample)

    print 'Running Starfinder'
    run_starfinder(oversample)

    print 'Running calibrate'
    run_calibrate(oversample)

    print 'Running align'
    run_align(oversample)

    print 'Plotting'
    plot_vector_diff(oversample)



def plot_vector_diff(oversample):
    """
    Make a vector plot of the differences between the input positions
    and the output positions after running starfinder.
    """
    # Dimensions of new grid
    npsf_side = 8 * oversample
    suffix = '%d_%d' % (npsf_side, npsf_side)

    s = starset.StarSet('grid_' + suffix + '/align_'+suffix)

    cnt = s.getArray('velCnt')
    mag = s.getArray('mag')
    idx = np.where((cnt == 2) & (mag < 16))[0]

    newStars = [s.stars[ii] for ii in idx]
    s.stars = newStars
    
    x0 = s.getArrayFromEpoch(0, 'xorig')
    y0 = s.getArrayFromEpoch(0, 'yorig')
    x1 = s.getArrayFromEpoch(1, 'xorig')
    y1 = s.getArrayFromEpoch(1, 'yorig')

    dx = x1 - x0
    dy = y1 - y0
    
    # Boundaries
    lx = np.array([  0, 512,1024,1536,2048,2560,3072,3584])
    ux = np.array([512,1024,1536,2048,2560,3072,3584,4096])
    ly = np.array([  0, 512,1024,1536,2048,2560,3072,3584])
    uy = np.array([512,1024,1536,2048,2560,3072,3584,4096])
    xedges = np.unique( np.append(lx, ux) )
    yedges = np.unique( np.append(ly, uy) )

    py.clf()
    q = py.quiver(x0, y0, dx, dy, scale=0.1)
    py.quiverkey(q, 0.1, 0.95, 0.1, '0.1 pixel', color='red', coordinates='axes')

    print('Mean Delta-X: {0:9.5f} +/- {1:9.5f} pixels'.format(dx.mean(), dx.std()))
    print('Mean Delta-Y: {0:9.5f} +/- {1:9.5f} pixels'.format(dy.mean(), dy.std()))

    for xx in xedges:
        py.axvline(xx, linestyle='--')
    for yy in yedges:
        py.axhline(yy, linestyle='--')
    
    py.savefig('plots/vec_diff_'+suffix+'.png')


