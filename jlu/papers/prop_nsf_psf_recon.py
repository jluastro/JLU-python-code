from jlu.papers import lu_gc_imf
import numpy as np
import pylab as py

def plot_spec_pairwise():
    """
    Calculate spectroscopic KLF and completeness curves.
    """
    # Load spectral identifications
    s = lu_gc_imf.load_spec_id_all_stars(flatten=True)

    m_spec = np.append(s.kp_yng, s.kp_old)
    x_spec = np.append(s.x_yng, s.x_old)
    y_spec = np.append(s.y_yng, s.y_old)

    # Sort by brightness so the plots look good.
    sdx = m_spec.argsort()
    m_spec = m_spec[sdx]
    x_spec = x_spec[sdx]
    y_spec = y_spec[sdx]

    sdx = s.kp_nirc2.argsort()
    m_nirc2 = s.kp_nirc2[sdx]
    x_nirc2 = s.x_nirc2[sdx]
    y_nirc2 = s.y_nirc2[sdx]

    # Calculate pairwise delta-pos and delta-mag for all NIRC2
    # sources.
    x2d = np.repeat([x_nirc2], len(x_nirc2), axis=0)
    y2d = np.repeat([y_nirc2], len(y_nirc2), axis=0)
    m2d = np.repeat([m_nirc2], len(m_nirc2), axis=0)

    dx2d = x2d.transpose() - x2d 
    dy2d = y2d.transpose() - y2d
    dm2d = m2d.transpose() - m2d  # brighter star - fainter star
    dr2d = np.hypot(dx2d, dy2d)

    dr_nirc2 = dr2d[np.triu_indices_from(dr2d, 1)]
    dm_nirc2 = dm2d[np.triu_indices_from(dm2d, 1)]

    # Calculate pairwise delta-pos and delta-mag for all
    # NIRC2 vs. spectral type combos.
    dr_spec = np.array([], dtype=float)
    dm_spec = np.array([], dtype=float)

    for ii in range(len(x_spec)):
        dr = np.hypot(x_spec[ii] - x_nirc2, y_spec[ii] - y_nirc2)
        dm = m_nirc2 - m_spec[ii]  # We care about m_nirc2 < m_spec

        dr_spec = np.append(dr_spec, dr)
        dm_spec = np.append(dm_spec, dm)

    # Trim out everything beyond 2"
    idx1 = np.where((dr_nirc2 < 2))[0]
    dr_nirc2 = dr_nirc2[idx1]
    dm_nirc2 = dm_nirc2[idx1]

    idx2 = np.where((dr_spec < 2) & (dm_spec <= 0))[0]
    dr_spec = dr_spec[idx2]
    dm_spec = dm_spec[idx2]

    # Plotting
    py.clf()
    py.plot(dr_nirc2, dm_nirc2, 'k.', color='grey', label='Imaging')
    py.plot(dr_spec, dm_spec, 'b.', label='Spectra')
    py.xlabel('Separation (arcsec)')
    py.ylabel('Contrast (mag)')
    py.legend(loc='lower left', numpoints=1)
    py.savefig('keck_ao_contrast_spec.png')

    print m_nirc2.max(), m_spec.max()
    
    
