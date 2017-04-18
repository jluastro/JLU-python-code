import numpy as np
import pylab as plt
from popstar import synthetic
from astropy.table import Table
from flystar import starlists, align
from flystar import match, transforms
import pdb

def make_unreddened_isochrone():
    log_age = 7.0
    AKs = 0.0
    d = 10.0
    
    new_iso = synthetic.IsochronePhot(log_age, AKs, d)

    return


def scenarios():
    pass


def make_cmd():
    acs_table = '/Users/jlu/work/microlens/MB980006/hst_10198_0k_acs_hrc_total/hst_10198_0k_acs_hrc_total_daophot_trm.cat'
    wfpc2_table = '/Users/jlu/work/microlens/MB980006/hst_08654_03_wfpc2_total_wf/hst_08654_03_wfpc2_total_wf_sexphot_trm.cat'
    nirc2_table = '/Users/jlu/data/microlens/16jul14/combo/starfinder/mag16jul14_mb980006_kp_rms_named.lis'

    acs_id = 1445
    wfpc2_id = 1280
    nirc2_id = 'MB980006'

    nirc2 = starlists.read_starlist(nirc2_table, error=True)
    ndx = np.where(nirc2['name'] == nirc2_id)[0][0]
    xarc = (nirc2['x'] - nirc2['x'][ndx]) * 0.01
    yarc = (nirc2['y'] - nirc2['y'][ndx]) * 0.01
    nirc2.rename_column('x', 'xpix')
    nirc2.rename_column('y', 'ypix')
    nirc2['x'] = xarc
    nirc2['y'] = yarc
    nirc2['m_kp'] = nirc2['m'].copy()
    
    acs_orig = Table.read(acs_table, format='ascii')
    acs_orig.rename_column('col1', 'id')
    acs_orig.rename_column('col2', 'xpix')
    acs_orig.rename_column('col3', 'ypix')
    acs_orig.rename_column('col4', 'ra')
    acs_orig.rename_column('col5', 'dec')
    acs_orig.rename_column('col6', 'm_814')

    # Find the source and trim out everything else outside of
    # 500 pix (10") radius.
    adx = np.where(acs_orig['id'] == acs_id)[0][0]
    r = np.hypot(acs_orig['xpix'] - acs_orig['xpix'][adx],
                acs_orig['ypix'] - acs_orig['ypix'][adx])
    idx = np.where(r < 200)[0]
    acs = acs_orig[idx]
    
    adx = np.where(acs['id'] == acs_id)[0][0]
    xarc = (acs['xpix'] - acs['xpix'][adx]) * 0.025
    yarc = (acs['ypix'] - acs['ypix'][adx]) * 0.025
    acs['x'] = xarc
    acs['y'] = yarc
    acs['xe'] = 0.01
    acs['ye'] = 0.01

    # Read WFPC2 table.
    wfpc2 = Table.read(wfpc2_table, format='ascii')
    wfpc2.rename_column('col1', 'id')
    wfpc2.rename_column('col2', 'xpix')
    wfpc2.rename_column('col3', 'ypix')
    wfpc2.rename_column('col4', 'ra')
    wfpc2.rename_column('col5', 'dec')
    wfpc2.rename_column('col6', 'm_439')
    wfpc2.rename_column('col7', 'm_555')
    wfpc2.rename_column('col8', 'm_675')
    wfpc2.rename_column('col9', 'm_814')

    # Find the source and trim out everything else outside of
    # 500 pix (10") radius.
    wdx = np.where(wfpc2['id'] == wfpc2_id)[0][0]
    r = np.hypot(wfpc2['xpix'] - wfpc2['xpix'][wdx],
                 wfpc2['ypix'] - wfpc2['ypix'][wdx])
    idx = np.where(r < 60)[0]
    wfpc2 = wfpc2[idx]
    
    wdx = np.where(wfpc2['id'] == wfpc2_id)[0][0]
    xarc = (wfpc2['xpix'] - wfpc2['xpix'][wdx]) * 0.1
    yarc = (wfpc2['ypix'] - wfpc2['ypix'][wdx]) * 0.1
    wfpc2['x'] = xarc
    wfpc2['y'] = yarc
    wfpc2['xe'] = 0.01
    wfpc2['ye'] = 0.01
    wfpc2['m'] = wfpc2['m_814'].copy()


    ##########
    # Align NIRC2 to ACS
    ##########
    
    # Calculate a quick zeropoint offset.
    color_offset = nirc2['m_kp'][ndx] - acs['m_814'][adx]
    acs['m'] = acs['m_814'] + color_offset
    
    # Align the two starlists
    trans = align.initial_align(nirc2, acs)
    nirc2_trans = align.transform_from_object(nirc2, trans)

    transModel = transforms.PolyTransform
    for i in range(5):
        idx_nirc2, idx_acs1 = align.transform_and_match(nirc2, acs, trans,
                                                       dr_tol=0.1,
                                                       dm_tol=2.0)
        # Test transform: apply to label.dat, make diagnostic plots
        nirc2_trans = align.transform_from_object(nirc2, trans)
        
        trans, N_trans = align.find_transform(nirc2[idx_nirc2],
                                              nirc2_trans[idx_nirc2],
                                              acs[idx_acs1],
                                              transModel=transModel,
                                              order=3,
                                              weights=None)
    
    # Test transform: apply to label.dat, make diagnostic plots
    nirc2_trans = align.transform_from_object(nirc2, trans)

    ##########
    # Align WFPC2 to ACS
    ##########
    # Calculate a quick zeropoint offset.
    color_offset = wfpc2['m_814'][wdx] - acs['m_814'][adx]
    acs['m'] = acs['m_814'] + color_offset
    
    # Align the two starlists
    trans = align.initial_align(wfpc2, acs, briteN=30)
    wfpc2_trans = align.transform_from_object(wfpc2, trans)

    transModel = transforms.PolyTransform
    for i in range(1):
        idx_wfpc2, idx_acs2 = align.transform_and_match(wfpc2, acs, trans,
                                                       dr_tol=0.4,
                                                       dm_tol=1)
        # Test transform: apply to label.dat, make diagnostic plots
        wfpc2_trans = align.transform_from_object(wfpc2, trans)
        
        trans, N_trans = align.find_transform(wfpc2[idx_wfpc2],
                                              wfpc2_trans[idx_wfpc2],
                                              acs[idx_acs2],
                                              transModel=transModel,
                                              order=1,
                                              weights=None)
    
    # Test transform: apply to label.dat, make diagnostic plots
    wfpc2_trans = align.transform_from_object(wfpc2, trans)

    ####
    # Plotting
    ####
    py.figure(1)
    plt.clf()
    plt.plot(nirc2_trans['x'], nirc2_trans['y'], 'rx')
    plt.plot(wfpc2_trans['x'], wfpc2_trans['y'], 'gs', mfc='none', mec='green')
    plt.plot(acs['x'], acs['y'], 'b+')
    plt.axis('equal')
    plt.xlabel('X (arcsec)')
    plt.ylabel('Y (arcsec)')

    nirc2_match = nirc2_trans[idx_nirc2]
    acs_t_match = acs[idx_acs1]

    dt = 2016.5 - 2004.5
    vx = (nirc2_match['x'] - acs_t_match['x']) * 1.0e3 / dt
    vy = (nirc2_match['y'] - acs_t_match['y']) * 1.0e3 / dt
    r = np.hypot(nirc2_match['x'], nirc2_match['y'])
    
    plt.figure(2)
    plt.clf()
    idx = np.where(r < 1)[0]
    ndx = np.where(nirc2_match['name'] == nirc2_id)[0]
    plt.plot(vx, vy, 'ko')
    plt.plot(vx[idx], vy[idx], 'ro')
    plt.plot(vx[ndx], vy[ndx], 'y*', ms=20)
    plt.axis('equal')
    plt.xlabel('vx (mas/yr)')
    plt.ylabel('vy (mas/yr)')

    plt.figure(3)
    plt.clf()
    plt.quiver(nirc2_match['x'], nirc2_match['y'], vx, vy, scale=50)
    plt.quiver(nirc2_match['x'][idx], nirc2_match['y'][idx],
                   vx[idx], vy[idx], color='red', scale=50)
    plt.quiver(nirc2_match['x'][ndx], nirc2_match['y'][ndx],
                   vx[ndx], vy[ndx], color='yellow', scale=50)
    plt.xlabel('X (arcsec)')
    plt.ylabel('Y (arcsec)')
    plt.axis('equal')

    plt.figure(4)
    plt.clf()
    color = acs_t_match['m_814'] - nirc2_match['m_kp']
    mag = nirc2_match['m_kp']
    plt.plot(color, mag, 'k.')
    plt.plot(color[idx], mag[idx], 'r.')
    plt.plot(color[ndx], mag[ndx], 'y*', ms=20)
    plt.gca().invert_yaxis()
    plt.xlabel('m_814 - m_kp (mag)')
    plt.ylabel('m_kp (mag)')


    wfpc2_match = wfpc2_trans[idx_wfpc2]
    acs_t_match = acs[idx_acs2]
    r = np.hypot(wfpc2_match['x'], wfpc2_match['y'])
    
    idx = np.where(r < 1)[0]
    wdx = np.where(wfpc2_match['id'] == wfpc2_id)[0]

    plt.figure(5)
    plt.clf()
    color = wfpc2_match['m_555'] - wfpc2_match['m_675']
    mag = wfpc2_match['m_555']
    plt.plot(color, mag, 'k.')
    plt.plot(color[idx], mag[idx], 'r.')
    plt.plot(color[wdx], mag[wdx], 'y*', ms=20)
    plt.gca().invert_yaxis()
    plt.xlabel('m_555 - m_675 (mag)')
    plt.ylabel('m_555 (mag)')
    plt.xlim(0, 2)

    print(acs_t_match[wdx]['m_814'])
    print(wfpc2_match[wdx]['m_439','m_555','m_675','m_814'])
    print(nirc2_match[ndx]['m_kp'])
    
    return
    
