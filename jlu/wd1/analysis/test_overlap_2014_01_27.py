import numpy as np
import pylab as py
from hst_flystar import reduce as red
import os
import shutil
from jlu.hst import starlists
from astropy.io import ascii
import pickle
from jlu.util import dataUtil
import pdb

imgs = {}
imgs['2010'] = {}
imgs['2013'] = {}

imgs['2010']['F125W'] = {}
imgs['2010']['F160W'] = {}
imgs['2013']['F160W'] = {}

imgs['2010']['F125W']['pos1'] = ['ib5w04ssq', 'ib5w04t3q', 'ib5w04tcq', 'ib5w05uoq', 
                                 'ib5w05uyq', 'ib5w05v8q', 'ib5w05wgq']
imgs['2010']['F125W']['pos2'] = ['ib5w02msq', 'ib5w02n2q', 'ib5w02ncq', 'ib5w02npq', 
                                 'ib5w05utq', 'ib5w05v4q', 'ib5w05vgq']
imgs['2010']['F125W']['pos3'] = ['ib5w02mxq', 'ib5w02n8q', 'ib5w02nhq', 'ib5w03pdq', 
                                 'ib5w03pnq', 'ib5w03pxq', 'ib5w03q7q'] 
imgs['2010']['F125W']['pos4'] = ['ib5w03piq', 'ib5w03ptq', 'ib5w03q2q', 'ib5w04snq', 
                                 'ib5w04sxq', 'ib5w04t7q', 'ib5w04thq']

imgs['2010']['F160W']['pos1'] = ['ib5w04svq', 'ib5w04t5q', 'ib5w04teq', 'ib5w05urq', 
                                 'ib5w05v1q', 'ib5w05vbq', 'ib5w05wjq']
imgs['2010']['F160W']['pos2'] = ['ib5w02mvq', 'ib5w02n5q', 'ib5w02nfq', 'ib5w02nsq', 
                                 'ib5w05uwq', 'ib5w05v6q', 'ib5w05viq']
imgs['2010']['F160W']['pos3'] = ['ib5w02n0q', 'ib5w02naq', 'ib5w02njq', 'ib5w03pgq', 
                                 'ib5w03pqq', 'ib5w03q0q', 'ib5w03qaq'] 
imgs['2010']['F160W']['pos4'] = ['ib5w03plq', 'ib5w03pvq', 'ib5w03q4q', 'ib5w04sqq', 
                                 'ib5w04t0q', 'ib5w04taq', 'ib5w04tkq']

imgs['2013']['F160W']['pos1'] = ['ibyv02moq', 'ibyv02mqq', 'ibyv02msq', 'ibyv02muq', 
                                 'ibyv02mwq', 'ibyv02myq', 'ibyv02n0q', 'ibyv02n2q', 
                                 'ibyv02n4q', 'ibyv02n6q', 'ibyv02n8q', 'ibyv02naq', 
                                 'ibyv02ncq', 'ibyv02neq']
imgs['2013']['F160W']['pos2'] = ['ibyv01kbq', 'ibyv01kdq', 'ibyv01kfq', 'ibyv01khq', 
                                 'ibyv01kjq', 'ibyv01klq', 'ibyv01knq', 'ibyv01kpq', 
                                 'ibyv01krq', 'ibyv01ktq', 'ibyv01kvq', 'ibyv01kxq', 
                                 'ibyv01kzq', 'ibyv01l1q']
imgs['2013']['F160W']['pos3'] = ['ibyv01jjq', 'ibyv01jlq', 'ibyv01jnq', 'ibyv01jpq', 
                                 'ibyv01jrq', 'ibyv01jtq', 'ibyv01jvq', 'ibyv01jxq', 
                                 'ibyv01jzq', 'ibyv01k1q', 'ibyv01k3q', 'ibyv01k5q', 
                                 'ibyv01k7q', 'ibyv01k9q']
imgs['2013']['F160W']['pos4'] = ['ibyv02ngq', 'ibyv02niq', 'ibyv02nkq', 'ibyv02nmq', 
                                 'ibyv02noq', 'ibyv02nqq', 'ibyv02nsq', 'ibyv02nuq', 
                                 'ibyv02nwq', 'ibyv02nyq', 'ibyv02o0q', 'ibyv02o2q', 
                                 'ibyv02o4q', 'ibyv02o6q']

# This should live in the main work directory -- wfc3ir_overlap
ref_init = 'MATCHUP.XYMEEE.f814w'

def organize_files():
    """
    Initially organize directories and files. This only needs to be run once.
    """
    all_years = ['2010', '2010', '2013']
    all_filts = ['F125W', 'F160W', 'F160W']
    all_pos = ['pos1', 'pos2', 'pos3', 'pos4']

    for ii in range(len(all_years)):
        year = all_years[ii]
        filt = all_filts[ii]
        
        for pos in all_pos:
            dir_root = year + '_' + filt + '_' + pos
            os.makedirs(dir_root)

            dir_xym = dir_root + '/01.XYM'
            os.makedirs(dir_xym)

            dir_xym_old = '../' + year + '_' + filt + '/01.XYM/'

            images = imgs[year][filt][pos]
            for ii in range(len(images)):
                shutil.copy(dir_xym_old + images[ii] + '_flt.xym', dir_xym)

    return
    

def align_year_pos(year, filt, pos):
    """
    Align a set of starlists for the specified year and mosaic position into
    a common reference frame. Do this iteratively until we get good positions 
    and uncertainties.
    """
    year = str(year)

    # All codes will be run in the work directory. 
    # The data directory is relative to the work directory.
    suffix = '_' + pos

    images = imgs[year][filt][pos]
    N_images = len(images)

    
    # First pass runs with the reference simply being the first image.
    red.xym2mat('ref0', year, filt, dir_suffix=suffix,
                mag='m-11,-9', clobber=True)
    red.xym2bar('ref0', year, filt, dir_suffix=suffix,
                Nepochs=N_images, clobber=True)
    
    # Second pass uses the matchup file as the reference.
    red.xym2mat('ref1', year, filt, dir_suffix=suffix,
                ref='MATCHUP.XYMEEE.ref0', ref_camera='c0', 
                ref_mag='m-11.5,-6', mag='m-11.5,-6', radius_key=[22, 24])
    red.xym2bar('ref1', year, filt, dir_suffix=suffix,
                Nepochs=N_images, make_lnk=True)

    

    return
        

def align_year(year, filt, recopy=False):
    """
    Align all the MATCHUP starlists (produced from align_year_pos()) for a given
    epoch. Do this iteratively. In the first pass, align to an F814W image. On the 
    second and third pass, align to the results from the first pass.
    """
    year = str(year)

    dir_work = year + '_' + filt
    dir_xym = year + '_' + filt + '/01.XYM/'

    all_pos = ['pos1', 'pos2', 'pos3', 'pos4']

    # Make directories
    if not os.path.exists(dir_work):
        os.makedirs(dir_work)
        os.makedirs(dir_xym)

    # Copy over files.
    if recopy:
        dir_xym_old = '../' + year + '_' + filt + '/01.XYM/'
        
        for pos in all_pos:
            for image in imgs[year][filt][pos]:
                shutil.copy(dir_xym_old + image + '_flt.xym', dir_xym)

        # Copy over the F814 file
        shutil.copy('MATCHUP.XYMEEE.f814w', dir_xym)


    # Stars must be detected in all the frames in one pointing.
    N_images = len(imgs[year][filt]['pos1'])

    # First pass runs with the reference simply being the first image.
    red.xym2mat('ref0', year, filt, mag='m-99,-13', radius_key=[10], 
                ref='MATCHUP.XYMEEE.f814w', ref_camera='f5 c0', ref_mag='m-99,-17',
                clobber=True)
    red.xym2mat('ref1', year, filt, mag='m-99,-13', radius_key=[10], 
                ref='MATCHUP.XYMEEE.f814w', ref_camera='f5 c0', ref_mag='m-99,-17')
    red.xym2mat('ref2', year, filt, mag='m-99,-10', radius_key=[10], 
                ref='MATCHUP.XYMEEE.f814w', ref_camera='f5 c0', ref_mag='m-99,-13')
    red.xym2mat('ref3', year, filt, mag='m-99,-08', radius_key=[10, 17], 
                ref='MATCHUP.XYMEEE.f814w', ref_camera='f5 c0', ref_mag='m-99,-10')

    red.xym2bar('ref3', year, filt, Nepochs=N_images, clobber=True)
    
    # Next-to-last pass uses the matchup file as the reference.
    red.xym2mat('ref4', year, filt, ref='MATCHUP.XYMEEE.ref3', ref_camera='c0', 
                ref_mag='m-11.5,-6', mag='m-11.5,-6', radius_key=[22, 24])
    red.xym2bar('ref4', year, filt, Nepochs=N_images, make_lnk=True)
    
    return


def get_file_indices(year, filt, pos):
    """
    For a specific year, filter, and position, read in 
    the IN.xym2mat file from <year>_<filt>/01.XYM/ and 
    parse which file name goes with which file index for the 
    specified position. 

    Return a list of indices that match the file name array
    imgs[year][filt][pos].

    Note that we assume that IN.xym2mat was written in the 
    same order as the python array file names. I don't check
    to see that this is the case.
    """
    year = str(year)

    dir_xym = year + '_' + filt + '/01.XYM/'

    # Loop through the input file for xym2mat and get the 
    # index number for each file.
    pos_idx = []

    _in = open(dir_xym + 'IN.xym2mat', 'r')

    for line in _in:
        parts = line.split()

        fileName = parts[1][1:-9]
        fileIndex = int(parts[0])

        if fileName in imgs[year][filt][pos]:
            pos_idx.append(fileIndex)
            
    _in.close()

    pos_idx = np.array(pos_idx)

    return pos_idx
    
def make_residuals_table_year_2pos(year, filt, pos1, pos2):
    year = str(year)

    dir_xym = year + '_' + filt + '/01.XYM/'

    
    # Read in the matchup file with the final positions and errors.
    stars = starlists.read_matchup(dir_xym + 'MATCHUP.XYMEEE.ref4')

    # For each image, read in the raw pixel coordinates, the transformed 
    # coordinates, and the residual offsets.
    N_images = len(imgs[year][filt][pos1])
    N_stars = len(stars)

    pos1_idx = get_file_indices(year, filt, pos1)
    pos2_idx = get_file_indices(year, filt, pos2)
    pos_idx = np.array([pos1_idx, pos2_idx])

    xraw = np.zeros([2, N_images, N_stars])
    yraw = np.zeros([2, N_images, N_stars])
    mraw = np.zeros([2, N_images, N_stars])

    xt = np.zeros([2, N_images, N_stars])
    yt = np.zeros([2, N_images, N_stars])
    mt = np.zeros([2, N_images, N_stars])

    dx = np.zeros([2, N_images, N_stars])
    dy = np.zeros([2, N_images, N_stars])

    detected = np.zeros([2, N_images, N_stars], dtype=bool)

    # Loop through images at each position
    for nn in range(N_images):
        # Loop through 2 different positions.
        for ii in range(2):
            print('nn = ', nn, 'ii = ', ii)

            dat = ascii.read('{0:s}LNK.{1:03d}'.format(dir_xym, pos_idx[ii, nn]))

            xt[ii, nn, :] = dat['col1'].data
            yt[ii, nn, :] = dat['col2'].data
            mt[ii, nn, :] = dat['col3'].data

            xraw[ii, nn, :] = dat['col4'].data
            yraw[ii, nn, :] = dat['col5'].data
            mraw[ii, nn, :] = dat['col6'].data

            dx[ii, nn, :] = dat['col7'].data
            dy[ii, nn, :] = dat['col8'].data

            idx = np.where(dat['col3'].data != 0)[0]
            detected[ii, nn, idx] = True
        
    # Trim the data down to those stars detected in ALL 
    # the images (the overlaps) at these two positions.
    det_Nimg = detected.sum(axis=1).sum(axis=0)
    idx = np.where(det_Nimg == (2*N_images))[0]
    print('Trim 1', len(idx))

    xt = xt[:, :, idx]
    yt = yt[:, :, idx]
    mt = mt[:, :, idx]
    xraw = xraw[:, :, idx]
    yraw = yraw[:, :, idx]
    mraw = mraw[:, :, idx]
    dx = dx[:, :, idx]
    dy = dy[:, :, idx]
    detected = detected[:, :, idx]
    stars = stars[idx]

    # Also trim on some quality metrics.
    idx2 = np.where((stars['m']>-11) & (stars['m']<-6) & 
                   (stars['xe']<0.05) & (stars['ye']<0.05) & 
                   (stars['me']<0.1))[0]
    print('Trim 1', len(idx2))

    stars = stars[idx2]
    xraw = xraw[:, :, idx2]
    yraw = yraw[:, :, idx2]
    mraw = mraw[:, :, idx2]
    xt = xt[:, :, idx2]
    yt = yt[:, :, idx2]
    mt = mt[:, :, idx2]
    dx = dx[:, :, idx2]
    dy = dy[:, :, idx2]
    xmean_p = xt.mean(axis=1)
    ymean_p = yt.mean(axis=1)
    mmean_p = mt.mean(axis=1)
    xstd_p = xt.std(axis=1)
    ystd_p = yt.std(axis=1)
    mstd_p = mt.std(axis=1)
    xmean = xmean_p.mean(axis=0)
    ymean = ymean_p.mean(axis=0)
    mmean = mmean_p.mean(axis=0)

    _out = open(dir_xym + 'resid_' + pos1 + '_' + pos2 + '.pickle', 'w')

    pickle.dump(stars, _out)
    pickle.dump(xmean, _out)
    pickle.dump(ymean, _out)
    pickle.dump(mmean, _out)
    pickle.dump(xmean_p, _out)
    pickle.dump(ymean_p, _out)
    pickle.dump(mmean_p, _out)
    pickle.dump(xstd_p, _out)
    pickle.dump(ystd_p, _out)
    pickle.dump(mstd_p, _out)
    pickle.dump(xraw, _out)
    pickle.dump(yraw, _out)
    pickle.dump(mraw, _out)
    pickle.dump(xt, _out)
    pickle.dump(yt, _out)
    pickle.dump(mt, _out)
    pickle.dump(dx, _out)
    pickle.dump(dy, _out)

    _out.close()


def make_residuals_table_year_pos(year, filt, pos):
    year = str(year) 
    
    dir_xym = year + '_' + filt + '_' + pos + '/01.XYM/'

    # Read in the matchup file with the final positions and errors.
    stars = starlists.read_matchup(dir_xym + 'MATCHUP.XYMEEE.ref1')

    # For each image, read in the raw pixel coordinates, the transformed 
    # coordinates, and the residual offsets.
    N_images = len(imgs[year][filt][pos])
    N_stars = len(stars)

    xraw = np.zeros([N_images, N_stars])
    yraw = np.zeros([N_images, N_stars])
    mraw = np.zeros([N_images, N_stars])

    xt = np.zeros([N_images, N_stars])
    yt = np.zeros([N_images, N_stars])
    mt = np.zeros([N_images, N_stars])

    dx = np.zeros([N_images, N_stars])
    dy = np.zeros([N_images, N_stars])

    
    for nn in range(N_images):
        print('nn = ', nn)
        dat = ascii.read('{0:s}LNK.{1:03d}'.format(dir_xym, nn+1))

        xt[nn, :] = dat['col1']
        yt[nn, :] = dat['col2']
        mt[nn, :] = dat['col3']

        xraw[nn, :] = dat['col4']
        yraw[nn, :] = dat['col5']
        mraw[nn, :] = dat['col6']

        dx[nn, :] = dat['col7']
        dy[nn, :] = dat['col8']
        

    idx = np.where((stars['m']>-11) & (stars['m']<-6) & 
                   (stars['xe']<0.05) & (stars['ye']<0.05) & 
                   (stars['me']<0.1))[0]

    stars = stars[idx]
    xraw = xraw[:, idx]
    yraw = yraw[:, idx]
    mraw = mraw[:, idx]
    xt = xt[:, idx]
    yt = yt[:, idx]
    mt = mt[:, idx]
    dx = dx[:, idx]
    dy = dy[:, idx]
    xmean = xt.mean(axis=0)
    ymean = yt.mean(axis=0)
    mmean = mt.mean(axis=0)
    xstd = xt.std(axis=0)
    ystd = yt.std(axis=0)
    mstd = mt.std(axis=0)

    _out = open(dir_xym + 'resid_all_images.pickle', 'w')

    pickle.dump(stars, _out)
    pickle.dump(xmean, _out)
    pickle.dump(ymean, _out)
    pickle.dump(mmean, _out)
    pickle.dump(xstd, _out)
    pickle.dump(ystd, _out)
    pickle.dump(mstd, _out)
    pickle.dump(xraw, _out)
    pickle.dump(yraw, _out)
    pickle.dump(mraw, _out)
    pickle.dump(xt, _out)
    pickle.dump(yt, _out)
    pickle.dump(mt, _out)
    pickle.dump(dx, _out)
    pickle.dump(dy, _out)

    _out.close()
    
def read_residuals_table_year_pos(year, filt, pos):
    year = str(year) 
    
    dir_xym = year + '_' + filt + '_' + pos + '/01.XYM/'

    _in = open(dir_xym + 'resid_all_images.pickle', 'r')

    data = dataUtil.DataHolder()

    data.stars = pickle.load(_in)
    data.xmean = pickle.load(_in)
    data.ymean = pickle.load(_in)
    data.mmean = pickle.load(_in)
    data.xstd = pickle.load(_in)
    data.ystd = pickle.load(_in)
    data.mstd = pickle.load(_in)
    data.xraw = pickle.load(_in)
    data.yraw = pickle.load(_in)
    data.mraw = pickle.load(_in)
    data.xt = pickle.load(_in)
    data.yt = pickle.load(_in)
    data.mt = pickle.load(_in)
    data.dx = pickle.load(_in)
    data.dy = pickle.load(_in)

    
    _in.close()

    return data
    

def read_residuals_table_year_2pos(year, filt, pos1, pos2):
    year = str(year)

    dir_xym = year + '_' + filt + '/01.XYM/'

    _in = open(dir_xym + 'resid_' + pos1 + '_' + pos2 + '.pickle', 'r')

    data = dataUtil.DataHolder()

    data.stars = pickle.load(_in)
    data.xmean = pickle.load(_in)
    data.ymean = pickle.load(_in)
    data.mmean = pickle.load(_in)
    data.xmean_p = pickle.load(_in)
    data.ymean_p = pickle.load(_in)
    data.mmean_p = pickle.load(_in)
    data.xstd_p = pickle.load(_in)
    data.ystd_p = pickle.load(_in)
    data.mstd_p = pickle.load(_in)
    data.xraw = pickle.load(_in)
    data.yraw = pickle.load(_in)
    data.mraw = pickle.load(_in)
    data.xt = pickle.load(_in)
    data.yt = pickle.load(_in)
    data.mt = pickle.load(_in)
    data.dx = pickle.load(_in)
    data.dy = pickle.load(_in)

    _in.close()

    return data

def plot_residuals_year_2pos(year, filt, pos1, pos2, refresh=False):
    year = str(year) 
    
    dir_xym = year + '_' + filt + '/01.XYM/'

    # Try to read from a FITS table of compiled data if possible.
    if refresh == True:
        make_residuals_table_year_2pos(year, filt, pos1, pos2)

    d = read_residuals_table_year_2pos(year, filt, pos1, pos2)

    print(d.xstd_p.min(), d.xstd_p.max())

    # Remember, this was all aligned to F814W in ACS. So there is a
    # relative scale change between the transformed and the raw pixels.
    # We need to put evertyhing back on the same scale.
    # The plate scale was 0.411095 (pulled from TRANS.xym2mat.ref4)
    scale = 0.411095

    dx1 = d.xmean_p[0, :] - d.stars['x']
    dx2 = d.xmean_p[1, :] - d.stars['x']
    dy1 = d.ymean_p[0, :] - d.stars['y']
    dy2 = d.ymean_p[1, :] - d.stars['y']
    lim = 0.06
    idx = np.where((np.abs(dx1) < lim) & (np.abs(dy1) < lim) & 
                   (np.abs(dx2) < lim) & (np.abs(dy2) < lim))[0]

    # Plot the offsets for each position on a common coordinate system.
    py.clf()
    q1 = py.quiver(d.xmean[idx], d.ymean[idx], dx1[idx], dy1[idx], 
                   color='red', scale=1.8)
    q1 = py.quiver(d.xmean[idx], d.ymean[idx], dx2[idx], dy2[idx], 
                   color='blue', scale=1.8)
    py.quiverkey(q1, -0.1, -0.12, 0.02/scale, '0.02 WFC3IR pixels')
    py.axis('equal')
    py.xlabel("X (pixels)")
    py.ylabel("Y (pixels)")
    py.title(year + ',' + filt + ',' + pos1 + ',' + pos2)
    py.savefig('plots/resid_final_{0}_{1}_{2}_{3}.png'.format(year, filt, pos1, pos2))

    # Plot the offsets for each position on the raw coordinate system.
    py.clf()
    xraw1 = d.xraw[0,:,:].mean(axis=0)
    yraw1 = d.yraw[0,:,:].mean(axis=0)
    xraw2 = d.xraw[1,:,:].mean(axis=0)
    yraw2 = d.yraw[1,:,:].mean(axis=0)
    
    q1 = py.quiver(xraw1[idx], yraw1[idx], dx1[idx], dy1[idx], 
                   color='red', scale=1.8)
    q1 = py.quiver(xraw2[idx], yraw2[idx], dx2[idx], dy2[idx], 
                   color='blue', scale=1.8)
    py.quiverkey(q1, -0.1, -0.12, 0.02/scale, '0.02 WFC3IR pixels')
    py.axis('equal')
    py.xlabel("X (pixels)")
    py.ylabel("Y (pixels)")
    py.title(year + ',' + filt + ',' + pos1 + ',' + pos2)
    py.savefig('plots/resid_raw_{0}_{1}_{2}_{3}.png'.format(year, filt, pos1, pos2))

def plot_residuals_year_pos(year, filt, pos, refresh=False):
    year = str(year) 
    
    dir_xym = year + '_' + filt + '_' + pos + '/01.XYM/'

    # Try to read from a FITS table of compiled data if possible.
    if refresh == True:
        make_residuals_table_year_pos(year, filt, pos)

    d = read_residuals_table_year_pos(year, filt, pos)

    print(d.xstd.min(), d.xstd.max())
    
    # Look at the variance in the raw X pixels.
    py.clf()
    q = py.quiver(d.xmean, d.ymean, d.xstd, d.ystd, scale=0.7)
    py.quiverkey(q, -0.1, -0.12, 0.02, '0.02 pixels', color='red')
    py.xlim(0, 1050)
    py.xlabel("X (pixels)")
    py.ylabel("Y (pixels)")
    py.title(year + ',' + filt + ',' + pos)
    
    py.savefig('plots/resid_errors_{0}_{1}_{2}.png'.format(year, filt, pos))
    
