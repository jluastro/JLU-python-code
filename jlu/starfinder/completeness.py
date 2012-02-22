import numpy as np
import pylab as py
import pyfits
from gcwork import starTables
from gcwork import objects
from jlu.util import utilities as util
from jlu.util import radialProfile
import math
import os
import pdb
import pickle
from scipy import ndimage
from scipy import interpolate
import time

def plot_pairwise_envelope(imageRoot, comboDir, magCut=100, radCut=10000, 
                           drBinSize=5, dmBinSize=0.5, calcArea=False):
    """
    For all pairs of stars in a starlist, plot up the separation and 
    delta-magnitude. Overplot the Starfinder PSF on the plots.

    imageRoot = e.g. mag06maylgs2_arch_f1_kp
    comboDir = e.g. /u/ghezgroup/data/gc/06maylgs2/combo/
    magCut = Only plot pairs where the primary (brighter star) is brighter
             than a certain magnitude (default=100).
    radCut = Only plot pairs where the radius of the primary (brighter star)
             is less than a certain radius in pixels (default=10,000).
    """
    image_file = comboDir + imageRoot + '.fits'
    psf_file = comboDir + imageRoot + '_psf.fits'
    #psf_file = imageRoot + '_psf2d_2asec.fits'
    lis_file = comboDir + 'starfinder/' + imageRoot + '_rms.lis'

    outSuffix = '%s_m%g_r%d' % (imageRoot.replace('mag', ''), magCut, radCut)

    lis = starTables.StarfinderList(lis_file, hasErrors=True)
    img = pyfits.getdata(image_file)
    psf2D = pyfits.getdata(psf_file)

    # Sort by brightness
    sidx = lis.mag.argsort()
    mag = lis.mag[sidx]
    x = lis.x[sidx]
    y = lis.y[sidx]
    xe = lis.xerr[sidx]
    ye = lis.yerr[sidx]
    asterr = (xe + ye) / 2.0

#     # Make a cut based on the astrometric at a given magnitude.
#     starsPerBin = 30
#     numBins = int( math.ceil(len(x) / starsPerBin) )
#     keep = np.array([], dtype=int)
#     for ii in range(numBins):
#         lo = ii * starsPerBin
#         hi = (ii+1) * starsPerBin

#         asterr_median = np.median( asterr[lo:hi] )
#         asterr_stddev = asterr[lo:hi].std()

#         # Keep only those below median + 3*stddev
#         cutoff = asterr_median + (2.0 * asterr_stddev)
#         tmp = np.where(asterr[lo:hi] < cutoff)[0]
#         tmp += lo
#         keep = np.append(keep, tmp)
#         print 'Stars from %.2f - %.2f have median(asterr) = %.4f   std(asterr) = %.4f' % (mag[lo:hi].min(), mag[lo:hi].max(), asterr_median, asterr_stddev)

#     print 'Throwing out %d of %d stars with outlying astrometric errors' % \
#         (len(x) - len(keep), len(x))

#     mag = mag[keep]
#     x = x[keep]
#     y = y[keep]

    starCount = len(mag)

    magMatrix = np.tile(mag, (starCount, 1))
    xMatrix = np.tile(x, (starCount, 1))
    yMatrix = np.tile(y, (starCount, 1))
    
    # Do the matrix calculation, then take the upper triangule
    dm = util.triu2flat(magMatrix.transpose() - magMatrix)
    dx = util.triu2flat(xMatrix.transpose() - xMatrix)
    dy = util.triu2flat(yMatrix.transpose() - yMatrix)
    dr = np.hypot(dx, dy)

    # Also pull out the x and y positions (and magnitudes) of the
    # primary star in each pair.
    x = util.triu2flat(xMatrix.transpose())
    y = util.triu2flat(yMatrix.transpose())
    m = util.triu2flat(magMatrix.transpose())

    xmid = (x.max() - x.min()) / 2.0
    ymid = (y.max() - y.min()) / 2.0
    r = np.hypot(x - xmid, y - ymid)

    # Calculate the edges of the detectors
    edge_xlo = x.min()
    edge_xhi = x.max()
    edge_ylo = y.min()
    edge_yhi = y.max()

    # Azimuthally average the PSF
    psf_r, psf_f, psf_std, psf_n = radialProfile.azimuthalAverage(psf2D)
    psf = -2.5 * np.log10(psf_f[0] / psf_f)  # convert to magnitudes

    # Trim down to a manageable size
    idx = np.where((dr > 0) & (dr < 500) & (m < magCut) & (r < radCut))[0]
    dm = dm[idx]
    dx = dx[idx]
    dy = dy[idx]
    dr = dr[idx]
    x = x[idx]
    y = y[idx]
    r = r[idx]
    m = m[idx]

    py.close(2)
    py.figure(2, figsize=(8, 6))

    # ##########
    # Plot 2D scatter of positions
    # ##########
    py.clf()
    py.scatter(dx, dy, s=2, c=dm, marker='o', edgecolor='none')
    py.xlabel('X (pixels)')
    py.ylabel('Y (pixels)')
    py.title('MagCut < %g, RadCut < %d' % (magCut, radCut))
    cbar = py.colorbar()
    cbar.set_label('Delta Magnitude')
    print 'Saving plots/pairs_xy_%s.png' % outSuffix
    py.savefig('plots/pairs_xy_%s.png' % outSuffix)

    # ##########
    # Plot 2D histogram of positions
    # ##########
    (xy, x_edges, y_edges) = py.histogram2d(dx, dy, bins=[25, 25])

    dx_bin_size = (x_edges[1] - x_edges[0]) * 0.00995
    dy_bin_size = (y_edges[1] - y_edges[0]) * 0.00995
    xy /= dx_bin_size * dy_bin_size

    py.clf()
    py.imshow(xy.transpose(), 
              extent=[x_edges[0], x_edges[-1], y_edges[0], y_edges[-1]],
              cmap=py.cm.gray_r)
    py.axis('equal')
    py.xlabel('X (pixel)')
    py.ylabel('Y (pixel)')
    py.title('MagCut < %g, RadCut < %d' % (magCut, radCut))
    cbar = py.colorbar()
    cbar.set_label('stars / arcsec^2')
    print 'Saving plots/pairs_xy_hist_%s.png' % outSuffix
    py.savefig('plots/pairs_xy_hist_%s.png' % outSuffix)

    # ##########
    # Grid up into magnitude and distance 2D bins
    # ##########
    dr_bins = np.arange(0, 501, drBinSize)
    dm_bins = np.arange(-11, 0.1, dmBinSize)
    (h, r_edges, m_edges) = py.histogram2d(dr, dm, bins=[dr_bins,dm_bins])


    # Calculate the area covered for each star, taking into account
    # edge limits. Use a dummy grid to calculate this for each star.
    grid_pix_size = 1.0
    grid_side = np.arange(-500, 501, grid_pix_size)
    grid_y, grid_x = np.meshgrid(grid_side, grid_side)
    grid_r = np.hypot(grid_x, grid_y)

    areaFile = 'area_%s_dr%d.dat' % (outSuffix, drBinSize)
    if os.path.isfile(areaFile) and not calcArea:
        print 'Loading area file: %s' % areaFile
        _area = open(areaFile, 'r')
        area_at_r = pickle.load(_area)
        _area.close()

        if len(area_at_r) != len(dr_bins)-1:
            print 'Problem with area_at_r in %s' % areaFile
            print '   len(area_at_r) = %d' % len(area_at_r)
            print '   len(dr_bins)-1 = %d' % (len(dr_bins) - 1)
            
    else:
        print 'Creating area file: %s' % areaFile

        area_at_r = np.zeros(len(dr_bins)-1, dtype=float)
        for rr in range(len(dr_bins)-1):
            idx = np.where((grid_r > r_edges[rr]) & (grid_r <= r_edges[rr+1]))
        
            rgrid_x = grid_x[idx[0], idx[1]]
            rgrid_y = grid_y[idx[0], idx[1]]

            for ss in range(len(dr)):
                sgrid_x = rgrid_x + x[ss]
                sgrid_y = rgrid_y + y[ss]
            
                sdx = np.where((sgrid_x > edge_xlo) & (sgrid_x < edge_xhi) &
                               (sgrid_y > edge_ylo) & (sgrid_y < edge_yhi))[0]

                area_at_r[rr] += len(sdx)
            area_at_r[rr] *= 0.00995**2 * grid_pix_size**2 / len(dr)
        
            area_formula = (r_edges[rr+1]**2 - r_edges[rr]**2)
            area_formula *= math.pi * 0.00995**2
            print '%3d < r <= %3d   %.5f vs. %.5f' % \
                (r_edges[rr], r_edges[rr+1], area_at_r[rr], area_formula)

        _area = open(areaFile, 'w')
        pickle.dump(area_at_r, _area)
        _area.close()


    for ii in range(h.shape[0]):
        h[ii] /= area_at_r[ii] * dmBinSize

    # Calculate a 2nd histogram for a sliding window in magnitude space. This
    # means we have to code our own 2D histogram.
    dr_bins2 = dr_bins
    dm_bins2 = np.arange(-11, 0.1, dmBinSize/5.0)
    h2 = np.zeros((len(dr_bins2)-1, len(dm_bins2)), dtype=float)
    
    for rr in range(len(dr_bins2)-1):
        for mm in range(len(dm_bins2)):
            dm_lo = dm_bins2[mm] - (dmBinSize / 2.0)
            dm_hi = dm_bins2[mm] + (dmBinSize / 2.0)

            idx = np.where((dr >= dr_bins2[rr]) & (dr < dr_bins2[rr+1]) &
                           (dm >= dm_lo) & (dm < dm_hi))[0]

            h2[rr, mm] = len(idx) / (area_at_r[rr] * dmBinSize)

    #m_edges = dm_bins2
    #h = h2
    #r_edges = dr_bins2

    # Now calculate the envelope by taking the average from 
    # outside 2"
    critDist = np.zeros(h.shape[1], dtype=float)
    idx = np.where(r_edges[:-1] > 2.0/0.00995)[0]
    h_avg_hi_r = np.median(h[idx[0]:, :], axis=0)
    h_std_hi_r = h[idx[0]:, :].std(axis=0)
    r_center = r_edges[0:-1] + (drBinSize / 2.0)
    m_center = m_edges[0:-1] + (dmBinSize / 2.0)
    #m_center = m_edges

    for mm in range(len(m_edges)-1):
        # Walk from the inside out until we get to a pixel 
        # that has a density of at least avgDensity / 3.0
        idx = np.where(h[:, mm] > (h_avg_hi_r[mm] / 3.0))[0]
        
        if len(idx) > 0:
            critDist[mm] = r_edges[idx[0]]

        print mm, m_edges[mm], critDist[mm], h_avg_hi_r[mm]

    # ##########
    # Make a 2D version of the envelope (smoothed)
    #   - from 0 > dm > -8, use the PSF
    #   - from dm < -8, use a smoothed version of the envelope.
    # ##########
    # Trim the envelope down to dm < -8
    idx = np.where((m_center < -8) & (critDist > 0))[0]
    int_dist = critDist[idx]
    int_mag = m_center[idx]
    foo = int_dist.argsort()
    int_dist = int_dist[foo]
    int_mag = int_mag[foo]
    tck = interpolate.splrep(int_dist, int_mag, s=0, k=1)
    env_r = np.arange(400)
    env_m = interpolate.splev(env_r, tck)

    # Graft just the tail onto the existing PSF. (past dm < -8)
    npsf_side = np.arange(-200, 201, 1)
    npsf_y, npsf_x = np.meshgrid(npsf_side, npsf_side)
    npsf_r = np.hypot(npsf_x, npsf_y)
    npsf2D = np.zeros(npsf_r.shape, dtype=float)
    npsf1D = np.zeros(npsf_r.shape, dtype=float)
    
    # Set to the 1D azimuthally averaged PSF.
    tck_psf = interpolate.splrep(psf_r, psf_f)
    idx = np.where(npsf_r < 100)
    npsf1D[idx[0], idx[1]] = interpolate.splev(npsf_r[idx[0], idx[1]], tck_psf)

    npsf2D[100:300, 100:300] = psf2D

    # Add in the halo.
    idx = np.where(env_m < -8.25)[0]
    startingDistance = env_r[idx].min()

    idx = np.where(npsf_r >= startingDistance)

    halo = interpolate.splev(npsf_r[idx[0], idx[1]], tck)
    halo = psf_f[0] / 10**(-0.4 * halo)

    npsf1D[idx[0], idx[1]] = np.hypot(npsf1D[idx[0], idx[1]], halo)
    npsf2D[idx[0], idx[1]] = np.hypot(npsf2D[idx[0], idx[1]], halo)


    # Possible choices of the envelope include:
    # azimuthally averaged version
    env_file = imageRoot + '_env1D.fits'
    if os.access(env_file, os.F_OK): os.remove(env_file)
    pyfits.writeto(env_file, npsf1D)

    # full 2D structure inside ~1"
    env_file = imageRoot + '_env2D.fits'
    if os.access(env_file, os.F_OK): os.remove(env_file)
    pyfits.writeto(env_file, npsf2D)

    # the original PSF
    env_file = imageRoot + '_psf2D.fits'
    if os.access(env_file, os.F_OK): os.remove(env_file)
    pyfits.writeto(env_file, psf2D)

    # For plotting purposes get the 1D evelope back out again.
    env_r, env_f, env_std, env_n = radialProfile.azimuthalAverage(npsf2D)
    env_m = -2.5 * np.log10(env_f[0] / env_f)  # convert to magnitudes

    # ##########
    # Plot points of dr vs. dm (unbinned)
    # ##########
    py.clf()
    py.plot(dr, dm, 'k.', ms=2)
    py.plot(psf_r, psf, 'b-')
    py.axis([0, 500, -11, 0])
    py.xlabel('X (pixel)')
    py.ylabel('Y (pixel)')
    py.title('MagCut < %g, RadCut < %d' % (magCut, radCut))
    print 'Saving plots/pairs_rm_%s.png' % outSuffix
    py.savefig('plots/pairs_rm_%s.png' % outSuffix)


    # ##########
    # Plot dr vs. dm in a 2D histogram
    # ##########
    hmask = np.ma.masked_where(h.transpose() == 0, h.transpose())
    r_edges = r_edges * 0.00995

    py.clf()
    py.imshow(hmask,
              extent=[r_edges[0], r_edges[-1], m_edges[0], m_edges[-1]],
              cmap=py.cm.spectral_r)
    py.plot(psf_r * 0.00995, psf, 'b-')
    py.plot(critDist * 0.00995, m_center, 'r.-')
    py.plot(env_r * 0.00995, env_m, 'g.-')
    py.axis('tight')
    py.axis([0, 5.0, -11, 0])
    py.xlabel('Radius (arcsec)')
    py.ylabel('Delta Magnitude')
    py.title('MagCut < %g, RadCut < %d' % (magCut, radCut))
    cbar = py.colorbar()
    cbar.set_label('stars / (mag * arcsec^2)')
    print 'Saving plots/pairs_rm_hist_%s.png' % outSuffix
    py.savefig('plots/pairs_rm_hist_%s.png' % outSuffix)
    

def map_detection_threshold(imageRoot, comboDir, suffix='env'):
    env_file = '%s_%s.fits' % (imageRoot, suffix)
    lis_file = comboDir + 'starfinder/' + imageRoot + '_rms.lis'

    lis = starTables.StarfinderList(lis_file, hasErrors=True)
    mag = lis.mag
    x = lis.x
    y = lis.y
    starCount = len(mag)

    # Faintest Star (and user as our zeropoint)
    # Take the last 10 stars and average their magnitudes to get the faintest
    #faintestMag = mag[-1]
    faintestMag = mag[-10:-1].mean()
    faintestFlux = 1.0
    print 'Faintest star detected = %.2f' % faintestMag

    # Make a 2D map that will eventually hold our completeness limits
    ximg = np.arange(math.ceil(x.max())+1)
    yimg = np.arange(math.ceil(y.max())+1)

    x2d, y2d = np.meshgrid(ximg, yimg)
    mapThresh = np.zeros((len(yimg), len(ximg)), dtype=float) + faintestMag
    mapDistance = np.zeros((len(yimg), len(ximg)), dtype=float) - 1.0
    mapMagnitude = np.zeros((len(yimg), len(ximg)), dtype=float)

    # Envelope
    env_small = pyfits.getdata(env_file)
    env_min = env_small.min()

    env = np.zeros(mapThresh.shape, dtype=float)
    env[:env_small.shape[0],:env_small.shape[1]] = env_small

    # Get the peak position and flux of the ENV.
    idx = np.unravel_index(env_small.argmax(), env_small.shape)
    env_x = np.arange(env_small.shape[1])
    env_y = np.arange(env_small.shape[0])
    env_peak_f = env_small.max()
    env_peak_x = env_x[idx[1]]
    env_peak_y = env_y[idx[0]]

    # Calculate the detection threshold map. Use the ENV + an absolute
    # detection floor set from the minimum magnitude stellar detection.

    # Loop through all the stars and build up our map
    for ii in range(starCount):
        if (ii % 100) == 0:
           print 'Completed %d stars' % ii

        xshift = x[ii] % 1
        yshift = y[ii] % 1
        
        tmp = ndimage.shift(env_small, (yshift, xshift))

        # Cut out values < 0 at the edges of the ENV field of view
        idx1 = np.where(tmp <= 0.0)
        tmp[idx1[0], idx1[1]] = 0.0
        
        # Convert to magnitudes relative to this star
        env_m = -2.5 * np.log10(tmp / tmp.max()) + mag[ii]

        # Select out just the sub-part of the map to compare against
        map_xlo = int( math.floor(x[ii] - env_peak_x) )
        map_xhi = int( map_xlo + env_m.shape[1] )
        map_ylo = int( math.floor(y[ii] - env_peak_y) )
        map_yhi = int( map_ylo + env_m.shape[0] )
        
        env_xlo = 0
        env_xhi = env_m.shape[1]
        env_ylo = 0
        env_yhi = env_m.shape[0]

        if map_xlo < 0:
            env_xlo -= map_xlo
            map_xlo -= map_xlo
        if map_ylo < 0:
            env_ylo -= map_ylo
            map_ylo -= map_ylo
        if map_xhi > mapThresh.shape[1]:
            diff = map_xhi - mapThresh.shape[1]
            env_xhi -= diff
            map_xhi -= diff
        if map_yhi > mapThresh.shape[0]:
            diff = map_yhi - mapThresh.shape[0]
            env_yhi -= diff
            map_yhi -= diff

        envSub = env_m[env_ylo:env_yhi, env_xlo:env_xhi]
        mapSub = mapThresh[map_ylo:map_yhi, map_xlo:map_xhi]
        mapMagSub = mapMagnitude[map_ylo:map_yhi, map_xlo:map_xhi]
        mapDistSub = mapDistance[map_ylo:map_yhi, map_xlo:map_xhi]
        x2dSub = x2d[map_ylo:map_yhi, map_xlo:map_xhi]
        y2dSub = y2d[map_ylo:map_yhi, map_xlo:map_xhi]

        # Find where this star is dominating the detection threshold.
        idx = np.where(envSub < mapSub)

        mapSub[idx[0], idx[1]] = envSub[idx[0], idx[1]]

        # Preserve the magnitude of the star that is dominating
        mapMagSub[idx[0], idx[1]] = mag[ii]

        # Calculate the distance from this star for each pixel
        # where this star dominates.
        mapDistSub[idx[0], idx[1]] = np.hypot(y2dSub[idx[0], idx[1]] - y[ii],
                                              x2dSub[idx[0], idx[1]] - x[ii])

    output_file = 'map_detect_thresh_%s_%s.dat' % (imageRoot, suffix)
    _out = open(output_file, 'w')
    pickle.dump(mapThresh, _out)
    pickle.dump(mapMagnitude, _out)
    pickle.dump(mapDistance, _out)
    _out.close()

    return output_file

def plot_map(imageRoot, suffix='env'):
    _file = open('map_detect_thresh_%s_%s.dat' % (imageRoot, suffix), 'r')
    mapThresh = pickle.load(_file)
    mapMagnitude = pickle.load(_file)
    mapDistance = pickle.load(_file)
    _file.close()

    py.clf()
    py.imshow(mapThresh)
    py.colorbar()

def completeness(imageRoot, magStepSize=0.25, dir='./', suffix='env1D'):
    
    filename = dir + 'map_detect_thresh_%s_%s.dat' % (imageRoot, suffix)
    print 'Loading evelope completeness from'
    print '  %s' % filename

    _map = open(filename, 'r')
    
    mapThresh = pickle.load(_map)
    mapMag = pickle.load(_map)
    mapDist = pickle.load(_map)

    pixelCount = mapThresh.size
    print 'Total size of map = %d' % pixelCount

    minMag = mapThresh.min()
    maxMag = mapThresh.max()

    mag = np.arange(minMag, maxMag+magStepSize, magStepSize)
    completeness = np.zeros(len(mag), dtype=float)
    print minMag, maxMag

    for mm in range(len(mag)):
        idx = np.where(mapThresh >= mag[mm])[0]

        completeness[mm] = float(len(idx)) / float(pixelCount)

    data = objects.DataHolder()
    data.mag = mag
    data.completeness = completeness
    data.mapThreshold = mapThresh
    data.mapMagnitude = mapMag
    data.mapDistance = mapDist

    return data


        
        
    
    
    
    
