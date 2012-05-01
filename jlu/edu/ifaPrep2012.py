import pylab as py
import numpy as np
import scipy.stats
import atpy
import math
import pdb

name = []
radius = []   # pc
dist = []     # kpc
velocity = [] # km/s
velDispersion = []  # km/s
density = []  # stars/arcmin^2
metallicity = []
metalDispersion = []

# Numbers from Martinez+ 2011
name.append('Segue 1')
radius.append(28.0) # pc
dist.append(23.0)   # kpc
velocity.append(209.0)     # km/s
velDispersion.append(3.5)  # km/s
density.append(1.65 / 2440.) # total mass density / mass-to-light ratio
metallicity.append(-2.7)  # Norris+ arxiv: 1008.0137
metalDispersion.append(0.4)

name = np.array(name)
radius = np.array(radius)
dist = np.array(dist)
velocity = np.array(velocity)
velDispersion = np.array(velDispersion)
density = np.array(density)
metallicity = np.array(metallicity)
metalDispersion = np.array(metalDispersion)

scale = 0.20          # arcsec per pixel
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

    # Figure out the positions of the stars
    field.add_column('x', scipy.stats.uniform.rvs(size=num_field) * fov)
    field.add_column('y', scipy.stats.uniform.rvs(size=num_field) * fov)

    # Figure out the brightness information for the stars.
    # Randomly select from the SDSS field sample.
    field_idx_sdss = np.random.randint(0, len(field_sdss), size=num_field)
    field_idx_spec = np.random.randint(0, len(field_spec), size=num_field)

    field.add_column('u', field_sdss.u[field_idx_sdss])
    field.add_column('g', field_sdss.g[field_idx_sdss])
    field.add_column('r', field_sdss.r[field_idx_sdss])
    field.add_column('i', field_sdss.i[field_idx_sdss])
    field.add_column('z', field_sdss.z[field_idx_sdss])
    field.add_column('Err_u', field_sdss.Err_u[field_idx_sdss])
    field.add_column('Err_g', field_sdss.Err_g[field_idx_sdss])
    field.add_column('Err_r', field_sdss.Err_r[field_idx_sdss])
    field.add_column('Err_i', field_sdss.Err_i[field_idx_sdss])
    field.add_column('Err_z', field_sdss.Err_z[field_idx_sdss])
    field.add_column('vel', field_spec.Vel[field_idx_spec])
    field.add_column('Err_vel', field_spec.e_Vel[field_idx_spec])
    
    return field


def make_galaxy_population(galaxyName):
    global density
    
    idx = np.where(name == galaxyName)[0]
    if len(idx) == 0:
        print 'Could not find galaxy %s' % (galaxyName)
        return
    print density

    # Re-express radius and density in arcmin
    pc2amin = 206265.0 / (dist[idx] * 1e3 * 60.)
    a = radius[idx] * pc2amin
    dens = density[idx] / (pc2amin**3)  # pretend that this can just be '^2

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
    density = n / volume

    py.figure(2)
    py.clf()
    py.plot(rbin_cent, density, 'k-', label='Random')
    plummerDensity = (1.0 + (rbin_cent/a)**2)**(-5.0/2.0) / a**3.0
    plummerDensity *= (numStars**0.85)
    py.semilogx(rbin_cent, plummerDensity, 'r-', label='Model')
    
    tmp2 = scipy.stats.uniform.rvs(size=numStars)
    tmp3 = scipy.stats.uniform.rvs(size=numStars)

    z = (2.0 * r * tmp2) - r
    theta = 2.0 * math.pi * tmp3
    x = np.sqrt(r**2 - z**2) * np.cos( theta )
    y = np.sqrt(r**2 - z**2) * np.sin( theta )

    py.clf()
    py.plot(x, y, 'k.')


    # Randomly generate velocities for these stars
    scipy.stats.norm.rvs(mu=galVel, sigma=galDispersion, size=numStars)
    
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
    py.plot(tab.g - tab.i, tab.r, 'k.')  # filters used by Simon+ Segue 1 paper
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

    
