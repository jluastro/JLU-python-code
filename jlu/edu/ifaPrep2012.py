import pylab as py
import numpy as np
import scipy.stats
import atpy
import math
import pdb
import pyfits
import glob
from jlu.util import statsIter

name = []
radius = []   # pc
dist = []     # kpc
velocity = [] # km/s
velDispersion = []  # km/s
density = []  # stars/arcmin^2
metallicity = []
metalDispersion = []
xPos = []
yPos = []

# # Numbers from Martinez+ 2011
# name.append('Segue 1')
# radius.append(28.0) # pc
# dist.append(23.0)   # kpc
# velocity.append(209.0)     # km/s
# velDispersion.append(3.5)  # km/s
# density.append(1.65 / 2440.) # total mass density / mass-to-light ratio
# metallicity.append(-2.7)  # Norris+ arxiv: 1008.0137
# #metalDispersion.append(0.4)
# metalDispersion.append(0.1)
# xPos.append(10)   # X Position in the FOV in arcmin
# yPos.append(10)   # X Position in the FOV in arcmin

# Originated from Segue 1 above, but heavily modified.
# Medium density
name.append('gal1')
radius.append(14.0) # pc
dist.append(23.0)   # kpc
velocity.append(209.0)     # km/s
velDispersion.append(50.0)  # km/s
density.append(1.0) # stars per sq. arcmin
metallicity.append(-2.7)  # Norris+ arxiv: 1008.0137
metalDispersion.append(0.3)
xPos.append(10)   # X Position in the FOV in arcmin
yPos.append(10)   # X Position in the FOV in arcmin

# Loweset density
name.append('gal2')
radius.append(20.0) # pc
dist.append(23.0)   # kpc
velocity.append(300.0)     # km/s
velDispersion.append(50.0)  # km/s
density.append(0.5) # stars per sq. arcmin
metallicity.append(-3.2)  # Norris+ arxiv: 1008.0137
metalDispersion.append(0.2)
xPos.append(13)   # X Position in the FOV in arcmin
yPos.append(13)   # X Position in the FOV in arcmin

# Medium density
name.append('gal3')
radius.append(15.0) # pc
dist.append(23.0)   # kpc
velocity.append(400.0)     # km/s
velDispersion.append(13.0)  # km/s
density.append(1.0) # stars per sq. arcmin
metallicity.append(-2.0)  # Norris+ arxiv: 1008.0137
metalDispersion.append(0.2)
xPos.append(6)   # X Position in the FOV in arcmin
yPos.append(13)   # X Position in the FOV in arcmin

# High density
name.append('gal4')
radius.append(17.0) # pc
dist.append(23.0)   # kpc
velocity.append(250.0)     # km/s
velDispersion.append(50.0)  # km/s
density.append(1.5) # stars per sq. arcmin
metallicity.append(-2.0)  # Norris+ arxiv: 1008.0137
metalDispersion.append(0.6)
xPos.append(13)   # X Position in the FOV in arcmin
yPos.append(7)   # X Position in the FOV in arcmin

# No galaxy (or almost none)
name.append('gal5')
radius.append(30.0) # pc
dist.append(23.0)   # kpc
velocity.append(209.0)     # km/s
velDispersion.append(100.0)  # km/s
density.append(0.1) # stars per sq. arcmin
metallicity.append(0)  # Norris+ arxiv: 1008.0137
metalDispersion.append(1.5)
xPos.append(10)   # X Position in the FOV in arcmin
yPos.append(10)   # X Position in the FOV in arcmin

name = np.array(name)
radius = np.array(radius)
dist = np.array(dist)
velocity = np.array(velocity)
velDispersion = np.array(velDispersion)
density = np.array(density)
metallicity = np.array(metallicity)
metalDispersion = np.array(metalDispersion)

scale = 0.40          # arcsec per pixel
fov = 20.0            # arcmin
fieldDensity = 1.37   # stars / arcmin^2

workDir = '/u/jlu/classes/ifa_prep_2012/'

def make_field_population():
    """
    Generate a randomly distributed set of field stars with the specified field
    density. Also set their brightnesses based on a
    """
    # Load up the field table and use to draw magnitude information.
    field_sdss = atpy.Table(workDir + 'sdss_field_near_segue1.csv', type='ascii')

    # Load up field velocity information from Simon+ 2011 study of Segue 1.
    tmp = atpy.Table(workDir + 'simon_2011_table_3.txt', type='ascii')
    field_spec = tmp.where((tmp.BProb > -10) & (tmp.BProb < -9))

    # This will be our final field ATpy table
    field = atpy.Table()

    # Figure out the total number of stars in the FOV
    num_field_exp = fieldDensity * fov**2
    num_field = scipy.stats.poisson.rvs(num_field_exp)

    # Generate some metallicities
    metal = scipy.stats.uniform.rvs(size=num_field, loc=-6, scale=7.0)
    metalErr = np.ones(num_field) * 0.1

    # Figure out the brightness information for the stars.
    # Randomly select from the SDSS field sample.
    field_idx_sdss = np.random.randint(0, len(field_sdss), size=num_field)
    field_idx_spec = np.random.randint(0, len(field_spec), size=num_field)

    vel = scipy.stats.uniform.rvs(size=num_field, loc=-500, scale=1300)
    velErr = np.ones(num_field) * 10.0
    #vel = field_spec.Vel[field_idx_spec]
    #velErr = field_spec.e_Vel[field_idx_spec]

    # Figure out the positions of the stars
    field.add_column('x', scipy.stats.uniform.rvs(size=num_field) * fov)
    field.add_column('y', scipy.stats.uniform.rvs(size=num_field) * fov)


    field.add_column('g', field_sdss.g[field_idx_sdss])
    field.add_column('r', field_sdss.r[field_idx_sdss])
    field.add_column('i', field_sdss.i[field_idx_sdss])
    field.add_column('Err_g', field_sdss.Err_g[field_idx_sdss])
    field.add_column('Err_r', field_sdss.Err_r[field_idx_sdss])
    field.add_column('Err_i', field_sdss.Err_i[field_idx_sdss])
    field.add_column('vel', vel)
    field.add_column('Err_vel', velErr)
    field.add_column('FeH', metal)
    field.add_column('Err_FeH', metalErr)
    field.add_column('member', np.zeros(num_field))
    
    return field


def make_galaxy_population(galaxyName):
    global density
    
    idx = np.where(name == galaxyName)[0]
    if len(idx) == 0:
        print 'Could not find galaxy %s' % (galaxyName)
        return

    galVel = velocity[idx]
    galVelDispersion = velDispersion[idx]
    galFeH = metallicity[idx]
    galFeHDispersion = metalDispersion[idx]

    # This will be our final cluster ATpy table
    cluster = atpy.Table()

    # Re-express radius and density in arcmin
    pc2amin = 206265.0 / (dist[idx] * 1e3 * 60.)
    a = radius[idx] * pc2amin

    # dens = density[idx] / (pc2amin**3)  # pretend that this can just be stars/sq. arcmin

    # Density is already in stars / sq. arcmin
    dens = density[idx]
    print "Cluster radius: %.2f'" % (a)
    print "Cluster density: %.2f stars / sq. arcmin" % (dens)
    print "Cluster velocity: %4d +/- %3d km/s" % (galVel, galVelDispersion)
    print "Cluster [Fe/H]: %4.1f +/- %3.1f " % (galFeH, galFeHDispersion)

    # Get the number of random stars to generate
    numStars = dens * 4.0 * math.pi * a**2

    tmp = scipy.stats.uniform.rvs(size=numStars)
    py.clf()
    py.hist(tmp)
    r = a * (tmp**(-2.0/3.0) - 1)**(-0.5)

    rbins = np.arange(0, 100, 2)
    rbin_cent = rbins[:-1] + np.diff(rbins)/2.0
    py.figure(1)
    py.clf()
    (n, b, p) = py.hist(r, bins=rbins)

    volume = (4.0 / 3.0) * math.pi * (rbins[1:]**3 - rbins[:-1]**3)
    stellarDensity = n / volume

    py.figure(2)
    py.clf()
    py.plot(rbin_cent, stellarDensity, 'k-', label='Random')
    plummerDensity = (1.0 + (rbin_cent/a)**2)**(-5.0/2.0) / a**3.0
    plummerDensity *= (numStars**0.85)
    py.semilogx(rbin_cent, plummerDensity, 'r-', label='Model')
    
    tmp2 = scipy.stats.uniform.rvs(size=numStars)
    tmp3 = scipy.stats.uniform.rvs(size=numStars)

    z = (2.0 * r * tmp2) - r
    theta = 2.0 * math.pi * tmp3
    x = np.sqrt(r**2 - z**2) * np.cos( theta )
    y = np.sqrt(r**2 - z**2) * np.sin( theta )

    x += xPos[idx]
    y += yPos[idx]

    # Trim down to just those stars within the FOV.
    idx = np.where((x > 0) & (x < fov) & (y > 0) & (y < fov))[0]
    numStars = len(idx)
    x = x[idx]
    y = y[idx]

    py.clf()
    py.plot(x, y, 'k.')
    py.axis('equal')
    
    # Randomly generate velocities for these stars
    vel = scipy.stats.norm.rvs(loc=galVel, scale=galVelDispersion, size=numStars)
    velErr = np.ones(numStars) * galVelDispersion / 5.0

    # Randomly generate metallicities for these stars.
    feh = scipy.stats.norm.rvs(loc=galFeH, scale=galFeHDispersion, size=numStars)
    fehErr = np.ones(numStars) * galFeHDispersion / 5.0

    # Use the magnitude information for members from Simon+ paper
    tmp = atpy.Table(workDir + 'simon_2011_table_3.txt', type='ascii')
    cluster_spec = tmp.where((tmp.Mem == 1))

    # Figure out the brightness information for the stars.
    # Randomly select from the SDSS field sample.
    idx_spec = np.random.randint(0, len(cluster_spec), size=numStars)

    # Add the positions of the stars
    cluster.add_column('x', x)
    cluster.add_column('y', y)

    cluster.add_column('g', cluster_spec.g[idx_spec])
    cluster.add_column('r', cluster_spec.r[idx_spec])
    cluster.add_column('i', cluster_spec.i[idx_spec])
    cluster.add_column('Err_g', np.ones(numStars) * 0.1)
    cluster.add_column('Err_r', np.ones(numStars) * 0.1)
    cluster.add_column('Err_i', np.ones(numStars) * 0.1)
    cluster.add_column('vel', vel)
    cluster.add_column('Err_vel', velErr)
    cluster.add_column('FeH', feh)
    cluster.add_column('Err_FeH', fehErr)
    cluster.add_column('member', np.ones(numStars))

    return cluster


def simulate_observation(galaxyName):
    stars = make_field_population()
    stars.append(make_galaxy_population(galaxyName))

    # Convert X and Y from arcmin to pixels
    stars.x *= 60.0 / scale
    stars.y *= 60.0 / scale

    # Trim out all stars that are too close.
    tooClose = []
    for ii in range(len(stars)):
        for jj in range(ii+1, len(stars)):
            # If we have already "deleted" this star in a previous epoch,
            # just skip it now.
            if jj in tooClose:
                continue

            # See if it is too close.
            r = np.hypot(stars.x[ii] - stars.x[jj], stars.y[ii] - stars.y[jj])

            if r < 6.0:
                tooClose.append(jj)

    good = np.arange(len(stars))
    good = np.delete(good, tooClose)
    stars = stars.rows(good)
    print 'Deleted %d stars that were closer than 6.0 pixels' % (len(tooClose))

    idx = np.where(stars.member == 1)[0]

    py.figure(1)
    py.clf()
    py.plot(stars.x, stars.y, 'k.')
    py.plot(stars.x[idx], stars.y[idx], 'r.')
    py.axis('equal')
    py.xlabel('X (pixels)')
    py.ylabel('Y (pixels)')
    py.title("FOV = %d'" % fov)
    py.savefig('%s_plot_xy.png' % galaxyName)
    
    py.figure(2)
    py.clf()
    py.plot(stars.g - stars.i, stars.r, 'k.')
    py.plot(stars.g[idx] - stars.i[idx], stars.r[idx], 'r.')
    py.gca().set_ylim(py.gca().get_ylim()[::-1])
    py.xlabel('g - i')
    py.ylabel('r')
    py.title('Field (black) + Galaxy (red)')
    py.savefig('%s_plot_cmd.png' % galaxyName)

    py.figure(3)
    py.clf()
    py.hist(stars.vel, histtype='step')
    py.xlabel('Velocity (km/s)')
    py.ylabel('Number of Stars')
    py.savefig('%s_hist_vel.png' % galaxyName)

    py.figure(4)
    py.clf()
    py.hist(stars.FeH, histtype='step')
    py.xlabel('[Fe/H]')
    py.ylabel('Number of Stars')
    py.savefig('%s_hist_feh.png' % galaxyName)

    # Make an image with the specificed FOV and plate scale
    npix = int(round(fov * 60.0 / scale))
    bkgLevel = 50.0
    bkgNoise = 7.0
    psfStd = 1.10 # pixels -- gives ~ 2.5 pixels across PSF FWHM
    zeropoint = 28.0

    #bkgImg = scipy.stats.norm.rvs(loc=bkgLevel, scale=bkgNoise, size=(npix, npix))
    bkgImg = np.ones((npix, npix)) * 50.0
    starsImg = np.zeros((npix, npix), dtype=float)

    xedges = np.arange(npix+1)
    yedges = np.arange(npix+1)

    fluxes = 10**((stars.r - zeropoint) / -2.5)

    psfDiam = 20

    for ii in range(len(stars)):
        # Decide bounds of PSF
        xmin = int(math.floor(stars.x[ii] - (psfDiam / 2.0)))
        ymin = int(math.floor(stars.y[ii] - (psfDiam / 2.0)))

        xmax = xmin + psfDiam + 1
        ymax = ymin + psfDiam + 1

        # Make sure everything is inside the image.
        if xmin < 0:
            xmin = 0
        if ymin < 0:
            ymin = 0
        if xmax > xedges[-1]:
            xmax = xedges[-1]
        if ymax > yedges[-1]:
            ymax = yedges[-1]

        xedgesPsf = xedges[xmin:xmax]
        yedgesPsf = yedges[ymin:ymax]

        # Generate PSF at the specified location
        cdf_x = scipy.stats.norm.cdf(xedgesPsf, loc=stars.x[ii], scale=psfStd)
        cdf_y = scipy.stats.norm.cdf(yedgesPsf, loc=stars.y[ii], scale=psfStd)

        psf_x = np.diff(cdf_x)
        psf_y = np.diff(cdf_y)

        psf = np.outer(psf_x, psf_y)
        psf *= fluxes[ii]

        # print 'Working on star %6d (R=%4.2f   peakFlux = %d' % \
        #     (ii, stars.r[ii], psf.max())
        

        starsImg[ymin:ymax-1, xmin:xmax-1] += psf.transpose()


    img = starsImg + bkgImg

    # Generate noise.
    print 'Generating noise'
    img = scipy.stats.poisson.rvs(img)

    # Make a histogram of the flux (star finding)
    py.figure(6)
    py.clf()
    py.hist(img.flatten(), bins=np.arange(0, 1000, 5), histtype='step', log=True)
    py.xlabel('Flux (cts)')
    py.savefig('%s_hist_flux.png' % galaxyName)

    # Save image to a FITS file.
    fitsHdr = pyfits.Header()
    fitsHdr.update(key='scale', value=scale, comment='Plate Scale (arcsec/pix)')
    fits = pyfits.PrimaryHDU(data=img, header=fitsHdr)
    fits.writeto('%s_img.fits' % galaxyName, clobber=True)

    # Calculate the densities and add them to the master star list
    stars = calcDensity(stars, plotRoot=('%s_master' % galaxyName))

    # Write a master star list
    stars.write('%s_master.txt' % galaxyName, type='ascii', overwrite=True, delimiter=',')

    # Make a region file
    _reg = open('%s_img.reg' % galaxyName, 'w')
    _reg.write('global color=yellow dashlist=8 3 width=1 select=1 fixed=0\n')
    _reg.write('physical\n')
    for ii in range(len(stars)):
        _reg.write('circle(%.1f,%.1f,10)\n' % (stars.x[ii], stars.y[ii]))
    _reg.close()

    return stars
    
def examine_sdss_field():
    # This is a 10' radius circle from SDSS at coordinates of:
    #    RA: 10:00:00
    #   DEC: 17:00:00
    # This is near Segue 1; but we will use it for our entire field sample
    tab = atpy.Table(workDir + 'sdss_field_near_segue1.csv', type='ascii')

    # Examine spatial density
    py.clf()
    py.plot(tab.ra, tab.dec, 'k.')
    py.xlabel('RA')
    py.ylabel('Dec.')
    py.title('Field Stars at 10:00:00 +17:00:00')
    py.savefig(workDir + 'field_density.png')

    # Examine color-magnitude diagram
    py.clf()
    py.plot(tab.g - tab.i, tab.r, 'k.', ms=3)  # filters used by Simon+ Segue 1 paper
    py.xlabel('SDSS g - i')
    py.ylabel('SDSS r')
    py.xlim(-2, 4)
    py.ylim(24, 12)
    py.title('Field Stars at 10:00:00 +17:00:00')
    py.savefig(workDir + 'field_cmd.png')

    idx = np.where(tab.r < 23)[0]
    
    # Mean spatial density
    volume = 4.0 * math.pi * 10.0**2
    surfaceDensity = len(idx) / volume
    print 'Field surface density for r<23 is: %.2f stars / arcmin^2' % surfaceDensity

    
def examine_sdss_cluster():
    # This is a 5' radius circle from SDSS at Segue 1 coordinates.
    tab = atpy.Table(workDir + 'sdss_segue1.csv', type='ascii')

    # Examine spatial density
    py.clf()
    py.plot(tab.ra, tab.dec, 'k.')
    py.xlabel('RA')
    py.ylabel('Dec.')
    py.title('Segue 1')
    py.savefig(workDir + 'segue1_density.png')

    # Examine color-magnitude diagram
    py.clf()
    py.plot(tab.g - tab.i, tab.r, 'k.', ms=3)  # filters used by Simon+ Segue 1 paper
    py.xlabel('SDSS g - i')
    py.ylabel('SDSS r')
    py.xlim(-2, 4)
    py.ylim(24, 12)
    py.title('Segue 1')
    py.savefig(workDir + 'segue1_cmd.png')

    idx = np.where(tab.r < 23)[0]
    
    # Mean spatial density
    volume = 4.0 * math.pi * 10.0**2
    surfaceDensity = len(idx) / volume
    print 'Segue 1 surface density for r<23 is: %.2f stars / arcmin^2' % surfaceDensity

def calcDensityFromFile(starList):
    outroot = starList.replace('.txt', '')
    
    stars = atpy.Table(starList, type='ascii')

    stars = calcDensity(stars)

    return stars

def calcDensity(stars, plotRoot=None):
    """
    Make a density map by stepping at every pixel in the image and calculating the
    density within a circle of radius 150 pixels.
    """
    npix = int(round(fov * 60.0 / scale))

    stars_x = stars[stars.keys()[0]]
    stars_y = stars[stars.keys()[1]]

    # Zip up X and Y positions into a (2, N) dimension array
    stars_xy = zip(stars_x, stars_y)

    # Build a K.D. Tree for nearnest neighbor calculation
    kdtree = scipy.spatial.kdtree.KDTree(stars_xy)

    # Use the 15th nearest neighbor
    nnStar = 15

    # Evaluate the nearest neighbor distances
    pts = kdtree.query(stars_xy, k=nnStar)

    # Get distances for all K nearest neighbors (sorted)
    dist_all_nn = np.array(pts[0])

    # Get distance for the K'th nearest neighbors
    dist_k = dist_all_nn[:,-1]

    # Convert distances to arcmins
    dist_k *= scale / 60.0

    # Calculate densities in stars / sq.arcmin
    dens = nnStar / (math.pi * dist_k**2)

    # Do the same as above, but for just the background
    # population. Obviously we only do this for master star lists
    # where we have membership information.
    if 'member' in stars.columns:
        bkg = stars.where(stars.member == 0)
        bkg_xy = zip(bkg.x, bkg.y)
        kdtree_bkg = scipy.spatial.kdtree.KDTree(bkg_xy)
        pts_bkg = kdtree_bkg.query(bkg_xy, k=nnStar)
        dist_all_nn_bkg = np.array(pts_bkg[0])
        dist_k_bkg = dist_all_nn_bkg[:,-1]
        dist_k_bkg *= scale / 60.0
        dens_bkg = nnStar / (math.pi * dist_k_bkg**2)

    # Check to see if we should do some plotting
    if plotRoot != None:
        # Plot histogram of distances
        histBins = np.arange(0, 5, 0.15)
        py.figure(1)
        py.clf()
        py.hist(dist_k, bins=histBins, histtype='step', label='All', normed=True)
        if "member" in stars.columns:
            py.hist(dist_k_bkg, bins=histBins, histtype='step', label='Bkg', normed=True)
            py.legend()
        py.xlabel('NN Distance k=%d (arcmin)' % (nnStar))
        py.savefig('%s_hist_distance.png' % plotRoot)

        # Plot histogram of densities
        histBins = np.arange(0, 10, 0.25)
        py.figure(2)
        py.clf()
        py.hist(dens, bins=histBins, histtype='step', label='All', normed=True)
        if "member" in stars.columns:
            py.hist(dens_bkg, bins=histBins, histtype='step', label='Bkg', normed=True)
            py.legend()
        py.xlabel('NN Density k=%d (stars / sq. arcmin)' % (nnStar))
        py.savefig('%s_hist_density.png' % plotRoot)
    
        # Plot 2d Map of densities
        py.figure(4)
        py.clf()
        py.scatter(stars_x, stars_y, s=30, c=dens,
                   edgecolor='none', cmap=py.cm.gist_stern_r)
        py.colorbar()
        py.axis('equal')
        py.savefig('%s_density_2d.png' % plotRoot)

    # Add density information to the output table.
    stars.add_column('density', dens)

    return stars


def make_snr_thinking_tool():
    """
    Make some figures that are useful as thinking tools for signal-to-noise concept.
    """
    # Vary the signal
    v = np.arange(1000)
    bkg1 = scipy.stats.norm.rvs(loc=1, scale=0.1, size=len(v))
    signal1 = 0.01 * scipy.stats.norm.pdf(v, loc=500, scale=10)

    bkg2 = scipy.stats.norm.rvs(loc=3, scale=0.1, size=len(v))
    signal2 = 10. * scipy.stats.norm.pdf(v, loc=500, scale=10)

    bkg3 = scipy.stats.norm.rvs(loc=5, scale=0.1, size=len(v))
    signal3 = 30. * scipy.stats.norm.pdf(v, loc=500, scale=10)

    py.close(1)
    py.figure(1, figsize=(10,6))
    py.clf()
    py.subplots_adjust(left=0.1)
    py.plot(v, bkg1 + signal1, 'k.')
    py.plot(v, bkg2 + signal2, 'k.')
    py.plot(v, bkg3 + signal3, 'k.')

    ax = py.gca()
    ax.grid(True, which='both')
    py.setp(ax.get_xticklabels(), visible=False)
    py.setp(ax.get_yticklabels(), visible=False)
    ax.xaxis.set_minor_locator(py.MultipleLocator(25))
    ax.yaxis.set_minor_locator(py.MultipleLocator(0.25))
    ax.xaxis.grid(True, 'minor')
    ax.yaxis.grid(True, 'minor')

    py.text(810, 5.5, 'Strong Signal')
    py.text(810, 3.5, 'Weak Signal')
    py.text(810, 1.5, 'No Signal')
    py.ylim(0, 7)

    py.savefig('thinking_tool_snr_1.png')

    # Vary the noise
    bkg1 = np.ones(len(v))
    signal1 = 30. * scipy.stats.norm.pdf(v, loc=500, scale=10)

    bkg2 = scipy.stats.norm.rvs(loc=3, scale=0.1, size=len(v))
    signal2 = 30. * scipy.stats.norm.pdf(v, loc=500, scale=10)

    bkg3 = scipy.stats.norm.rvs(loc=5, scale=0.3, size=len(v))
    signal3 = 30. * scipy.stats.norm.pdf(v, loc=500, scale=10)
    
    py.clf()
    py.subplots_adjust(left=0.1)
    py.plot(v, bkg1 + signal1, 'k.')
    py.plot(v, bkg2 + signal2, 'k.')
    py.plot(v, bkg3 + signal3, 'k.')

    ax = py.gca()
    ax.grid(True, which='both')
    py.setp(ax.get_xticklabels(), visible=False)
    py.setp(ax.get_yticklabels(), visible=False)
    ax.xaxis.set_minor_locator(py.MultipleLocator(25))
    ax.yaxis.set_minor_locator(py.MultipleLocator(0.25))
    ax.xaxis.grid(True, 'minor')
    ax.yaxis.grid(True, 'minor')

    py.text(810, 5.8, 'Strong Noise')
    py.text(810, 3.5, 'Weak Noise')
    py.text(810, 1.2, 'No Noise')
    py.ylim(0, 7)

    py.savefig('thinking_tool_snr_2.png')
    

def plot_final_synthesis():
    data = atpy.Table(workDir + 'gal1_master.txt', type='ascii')
    img = pyfits.getdata(workDir + 'gal1_img.fits')

    imgBkg = statsIter.mean(img.flatten(), hsigma=5, iter=1, verbose=True)
    imgStd = statsIter.std(img.flatten(), hsigma=5, iter=1, verbose=True)
    print 'Image mean = %5.1f  std = %5.1f' % (imgBkg, imgStd)

    fluxBins = np.arange(0, 500, 8)
    velBins = np.arange(-500, 801, 50)
    n_bins = np.arange(10, 60, 2)

    npix, fluxEdges = np.histogram(img.flatten(), bins=fluxBins)
    nstars, binEdges = np.histogram(data.vel, bins=velBins)
    n, b = np.histogram(nstars, bins=n_bins)

    bkg = np.mean(nstars)
    noise = np.std(nstars)
    print 'First Iteration:  mean = %5.1f, std = %5.1f' % (bkg, noise)

    # Do one round of rejection
    idx = np.where(nstars < (bkg + (3.0 * noise)))[0]
    bkg = np.mean(nstars[idx])
    noise = np.std(nstars[idx])
    print 'Second Iteration: mean = %5.1f, std = %5.1f' % (bkg, noise)
    
    gaussianCDF = scipy.stats.norm.cdf(b, loc=bkg, scale=noise)
    gaussianPDF = np.diff(gaussianCDF) * n.sum()
    gaussianBinCenters = n_bins[0:-1] + np.diff(n_bins) / 2.0

    fluxBinsModel = np.arange(0, 500, 1)
    fluxNormCDF = scipy.stats.norm.cdf(fluxBinsModel, loc=imgBkg, scale=imgStd)
    fluxNormPDF = np.diff(fluxNormCDF)
    fluxNormPDF *= npix.max() / fluxNormPDF.max()
    fluxBinCenters = fluxBinsModel[0:-1] + np.diff(fluxBinsModel) / 2.0

    # Images
    py.figure(1)
    py.clf()
    py.subplots_adjust(left=0.22)
    py.subplot(2, 1, 1)
    py.hist(img.flatten(), histtype='step', bins=fluxBins)
    py.plot(fluxBinCenters, fluxNormPDF, 'g-')
    py.title('Field #1')
    py.xlim(0, 200)

    py.subplot(2, 1, 2)
    py.hist(img.flatten(), histtype='step', bins=fluxBins)
    py.plot(fluxBinCenters, fluxNormPDF, 'g-')
    py.ylim(0, 1000)
    py.xlim(0, 200)
    py.xlabel('Flux')
    py.savefig(workDir + 'final_syn_gal1_flux_hist.png')

    py.figure(1)
    py.subplots_adjust(left=0.15)
    py.clf()
    py.hist(data.vel, bins=velBins, histtype='step')
    py.xlim(-500, 800)
    py.xlabel('Velocity (km/s)')
    py.ylabel('Number of Stars')
    py.savefig(workDir + 'final_syn_gal1_vel_hist_no_lines.png')

    py.axhline(bkg, linestyle='-', linewidth=3, color='black')
    py.axhline(bkg+noise, linestyle='--', linewidth=3, color='black')
    py.axhline(bkg-noise, linestyle='--', linewidth=3, color='black')
    py.savefig(workDir + 'final_syn_gal1_vel_hist.png')
    
    py.figure(2)
    py.clf()
    py.subplots_adjust(left=0.15)
    (n, b, p) = py.hist(nstars, histtype='step', bins=n_bins, label='Data')
    py.ylim(0, n.max()*1.3)
    py.xlim(b.min(), b.max())
    py.xlabel('Number of Stars per Velocity Bin')
    py.ylabel('Number of Velocity Bins')


    print gaussianPDF.max(), n.max()
    
    py.plot(gaussianBinCenters, gaussianPDF, label='Gaussian Fit')
    axLim = py.axis()
    py.axvline(bkg, linestyle='-', color='black', linewidth=3)
    py.arrow(bkg+1*noise, axLim[3], 0, -0.5,
             color='black', linewidth=3, head_width=1, head_length=0.1)
    py.arrow(bkg+2*noise, axLim[3], 0, -0.5,
             color='black', linewidth=3, head_width=1, head_length=0.1)
    py.arrow(bkg+3*noise, axLim[3], 0, -0.5,
             color='black', linewidth=3, head_width=1, head_length=0.1)

    py.text(bkg+1*noise, axLim[3]+0.15, r'$1\sigma$',
            horizontalalignment='center', verticalalignment='bottom')
    py.text(bkg+2*noise, axLim[3]+0.15, r'$2\sigma$',
            horizontalalignment='center', verticalalignment='bottom')
    py.text(bkg+3*noise, axLim[3]+0.15, r'$3\sigma$',
            horizontalalignment='center', verticalalignment='bottom')
    
    py.savefig(workDir + 'final_syn_gal1_vel_hist_hist.png')
    

    
    
