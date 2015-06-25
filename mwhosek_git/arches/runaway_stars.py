import numpy as np
from astropy.table import Table
import pylab as py
import pdb

def deredden_mags(catalog):
    """
    Helper function that applies reddening observed mags. Given
    catalog (already read-in), returns the de-reddened mags.
    """
    A127 = np.array(catalog['A1.27'])
    A139 = np.array(catalog['A1.39'])
    A153 = np.array(catalog['A1.53'])

    mag127 = np.array(catalog['m_2010_f127m'])
    mag139 = np.array(catalog['m_2010_f139m'])
    mag153 = np.median([np.array(catalog['m_2010_f153m']), 
                        np.array(catalog['m_2011_f153m']), 
                        np.array(catalog['m_2012_f153m'])], 
                        axis = 0)
                        
    mag127_dered = mag127 - A127
    mag139_dered = mag139 - A139
    mag153_dered = mag153 - A153

    return mag127_dered, mag139_dered, mag153_dered

def plot_kinematic_members(catalogfile, prob):
    """
    Plot the CMDs of the kinematically-selected cluster members, defined
    as having probabilities higher than <prob>

    catalogfile should include calculations of extinction
    (e.g. catalog_key1_Aks2.4.fits). Cluster membership probabilities are
    calculated solely on kinematics

    This will give insight to photometric cuts on full catalog data
    """
    # Read differentially-dereddened catalogfile, apply
    # reddening correction
    d = Table.read(catalogfile)
    mag127_dered, mag139_dered, mag153_dered = deredden_mags(d)
    
    # Adopting P > prob as members
    members = np.where(d['Membership'] > prob)

    #-----CMD of cluster members----#
    # F127M vs. F127M - F153M
    py.figure(1, figsize=(10,10))
    py.clf()
    py.plot(mag127_dered[members] - mag153_dered[members],
             mag127_dered[members], 'k.')
    py.axis([0, 4, 23, 13])
    py.xlabel('F127M - F153M')
    py.ylabel('F127M')
    py.title('Kinematically-Selected Members, De-reddened')
    py.savefig('Kin_members_127_153.png')

    # F139M vs. F139M - F153M
    py.figure(2, figsize=(10,10))
    py.clf()
    py.plot(mag139_dered[members] - mag153_dered[members],
            mag139_dered[members], 'k.')
    py.axis([-1.0, 3, 23, 13])
    py.xlabel('F139M - F153M')
    py.ylabel('F139M')
    py.title('Kinematically-Selected Members, De-reddened')
    py.savefig('Kin_members_139_153.png')

    # Color-Color diagram
    py.figure(3, figsize=(10,10))
    py.clf()
    py.plot(mag139_dered[members] - mag153_dered[members],
            mag127_dered[members] - mag139_dered[members], 'k.')
    py.xlabel('F139M - F153M')
    py.ylabel('F127M - F139M')
    py.title('Kinematically-Selected Members, De-reddened')
    py.axis([0, 1.8, -1, 3])
    py.savefig('Kin_members_cc.png')
    
    return

def identify_photometric_candidates(catalogfile, prob, faintLim):
    """
    Go through catalogfile and identify possible cluster members photometrically
    rather than kinematically. For now, will adopt by-eye photometric cuts from
    distribution of kinematically-selected population (P > prob), as presented in
    tidal radius paper.

    catalogfile should include calculations of extinction
    (e.g. catalog_key1_Aks2.4.fits).

    Will only consider stars observed mags brighter than faintLim in F153M
    
    Photometric membership cuts determined by eye, hardcoded.
    THROUGHOUT CODE, F127M - F153M is color1, F139M - F153M is color2

    Returns catalog of the non-kinematic members that fulfill the photometric
    criteria
    """
    # Read in catalog, differentially de-redden mags. Calculate colors
    d = Table.read(catalogfile, format='fits')
    mag127_dered, mag139_dered, mag153_dered = deredden_mags(d)

    color1 = mag127_dered - mag153_dered
    color2 = mag139_dered - mag153_dered

    # Identify cluster members (P > prob), as well as non-members
    # brighter than faintLim
    members = np.where(d['Membership'] > prob)
    non_members = np.where( (d['Membership'] < prob) &
                            (d['m_2010_f153m'] < faintLim) )

    # Apply photometric cuts to non-kinematic members, see what's left
    color1_cut = [2.0, 2.7]
    color2_cut = [1.0, 1.5]

    possible = np.where( (color1[non_members] > color1_cut[0]) &
                         (color1[non_members] < color1_cut[1]) &
                         (color2[non_members] > color2_cut[0]) &
                         (color2[non_members] < color1_cut[1]) )
    
    print 'After photometric cuts, {0:4.0f} stars left'.format(len(possible[0]))

    # Write catalog of just photometric candidates
    candidates = d[possible]
    candidates.write('candidates.fits', format='fits')
    
    return

def explore_photo_candidates(catalogfile, photocatalog, prob):
    """
    Explore color-color diagram and kinematics of photometric
    candidates.

    catalogfile is of all stars and should include calculations of
    extinction (e.g. catalog_key1_Aks2.4.fits). Identify kinematic
    members as those stars with P > prob

    photocatalog is the catalog of just photometric candidates, output
    from identify_photometric_candidates
    """
    # Read full catalog and photo catalog, de-redden mags
    d = Table.read(catalogfile, format='fits')
    photo = Table.read(photocatalog, format='fits')

    mag127_dered, mag139_dered, mag153_dered = deredden_mags(d)
    photo127_dered, photo139_dered, photo153_dered = deredden_mags(photo)

    # Identify kinematic cluster members
    members = np.where(d['Membership'] > prob)

    # Plot color-color diagram of kinematic members vs. photo candidates
    #-------------PLOT------------#
    py.figure(1, figsize=(10,10))
    py.clf()
    py.plot(mag139_dered[members] - mag153_dered[members],
            mag127_dered[members] - mag139_dered[members], 'k.',
            label = 'Kinematic Members')
    py.plot(photo139_dered - photo153_dered, photo127_dered - photo139_dered,
            'r.', label = 'Photometric Candidates', alpha = 0.7)
    py.legend()
    py.xlabel('F139M - F153M')
    py.ylabel('F127M - F139M')
    py.title('Kinematically-Selected Members, De-reddened')
    py.axis([0, 2.3, -1, 3])
    py.savefig('Photo_cc.png')
    #------------------------------#
    
    # Look at quiverplot of photo candidates
    # Need to convert positions from mas to pixels
    pscale = 121.625
    xpos = photo['x_2010_f153m'] / pscale
    ypos = photo['y_2010_f153m'] / pscale
    xvel = photo['fit_vx']
    yvel = photo['fit_vy']

    # Put the proper motions and positions in the cluster reference
    # frame (cluster at (0,0), with 0 vel)
    xpos_clust = np.median(d['x_2010_f153m'][members]) / pscale
    ypos_clust = np.median(d['y_2010_f153m'][members]) / pscale
    xvel_clust = np.median(d['fit_vx'][members])
    yvel_clust = np.median(d['fit_vy'][members])
    xpos_ref = xpos - xpos_clust
    ypos_ref = ypos - ypos_clust
    xvel_ref = xvel - xvel_clust
    yvel_ref = yvel - yvel_clust

    # Cut out high proper motions which would be unlikely large
    # velocities (> 100 km/s) at the Arches distance
    lim = 7.67 # mas/yr
    good = np.where ( (xvel_ref < lim) & (yvel_ref < lim) )
    xpos_ref = xpos_ref[good]
    ypos_ref = ypos_ref[good]
    xvel_ref = xvel_ref[good]
    yvel_ref = yvel_ref[good]

    #-------PLOT--------#
    py.figure(2, figsize=(10,10))
    py.clf()
    q = py.quiver(xpos_ref, ypos_ref, xvel_ref, yvel_ref, scale=10)
    py.quiverkey(q, 0.2, 0.92, 1, '1 mas/yr', coordinates='figure', color='black')
    py.xlabel('X Position (pix)')
    py.ylabel('Y Position (pix)')
    py.title('Photometric Candidates, Velocity < {0:4.0f} km/s'.format(lim * 39.11))
    py.savefig('photo_quiver_all.png')    
    #-------------------#
    
    # When xpos_ref < 0, want xvel < 0. When xpos_ref > 0, want xvel > 0
    # When ypos_ref < 0, want yvel < 0. When ypos_ref > 0, want yvel > 0
    ind = []
    for i in range(len(xpos_ref)):
        if (xpos_ref[i] < 0) & (xvel_ref[i] < 0):
            if (ypos_ref[i] < 0) & (yvel_ref[i] < 0):
                ind.append(i)
            elif (ypos_ref[i] > 0) & (yvel_ref[i] > 0):
                ind.append(i)
        elif (xpos_ref[i] > 0) & (xvel_ref[i] > 0):
            if (ypos_ref[i] < 0) & (yvel_ref[i] < 0):
                ind.append(i)
            elif (ypos_ref[i] > 0) & (yvel_ref[i] > 0):
                ind.append(i)

    # Plot remaining candidates
    py.figure(3, figsize=(10,10))
    py.clf()
    q = py.quiver(xpos_ref[ind], ypos_ref[ind], xvel_ref[ind], yvel_ref[ind],
                  scale=40)
    py.quiverkey(q, 0.2, 0.92, 1, '1 mas/yr', coordinates='figure', color='black')
    py.xlabel('X Position (pix)')
    py.ylabel('Y Position (pix)')
    py.title('Velocity < {0:4.0f} km/s, Outward Radial Direction'.format(lim * 39.11))
    py.axis([-650, 650, -650, 650])
    py.savefig('photo_quiver_rad.png')
    
    return
