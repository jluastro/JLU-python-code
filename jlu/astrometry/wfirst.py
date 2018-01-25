import numpy as np
import pylab as plt

def gc_red_clump():
    """
    Calculate distance to RC stars we could measure parallax and 
    proper motions for with WFIRST. All kinds of assumptions in here.
    """
    A_H_Ks_ratio = 2.09

    F153M_obs_ref = 19.13  # mag
    F153M_RC = -1.3032 # mag
    A_F153M_ref = 5.9182
    A_Ks_ref = 2.54 # mag
    A_F153M_Ks = 2.33
    DM_ref = F153M_obs_ref - A_F153M_ref - F153M_RC
    d_ref = 10**((DM_ref + 5.0) / 5.0)
    print('GC Values: Distance Modulus = ', str(DM_ref), " Distance = ", str(d_ref) + " pc")

    # Recalculate new distance for a F153M = 21 mag star with different A_Ks
    F153M_wfirst = 21.0
    A_Ks_wfirst = np.array([1.25, 2.5, 5.0])
    A_F153M_wfirst = A_Ks_wfirst * A_F153M_Ks

    DM_wfirst = F153M_wfirst - A_F153M_wfirst - F153M_RC
    d_wfirst = 10**((DM_wfirst + 5.0) / 5.0)
    print(d_wfirst)
    

    

