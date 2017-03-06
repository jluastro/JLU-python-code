import pylab as plt
from flystar import starlists
from flystar import match

def plot_15may():
    """
    We have three different versions of starlists for 15 may. 
    I am looking into the astrometry for all 3.
    """

    # Starlist 1: Old version of starfinder (v1.6)
    list1 = '/Volumes/indicium/g/lu/data/microlens/15may/combo_pre_2016_09_14/starfinder/mag15may_ob120169_kp_rms_named.lis'

    # Starlist 2: AIROPA in legacy mode, refined PSF starlist.
    list2 = '/Users/jlu/data/microlens/15may/combo/starfinder/mag15may_ob120169_kp_rms_named.lis'

    # Starlist 3: AIROPA in single-PSF mode.
    list3 = '/Users/jlu/data/microlens/15may/combo_2016_10_10/starfinder/mag15may_ob120169_kp_rms_named.lis'

    stars1 = starlists.read_starlist(list1)
    stars2 = starlists.read_starlist(list2)
    stars3 = starlists.read_starlist(list3)

    idx1_2, idx2_1, dr12, dm12 = match.match(stars1['x'], stars1['y'], stars1['mag'],
                                             stars2['x'], stars2['y'], stars2['mag'], dr_tol=2)

    idx1_3, idx3_1, dr13, dm13 = match.match(stars1['x'], stars1['y'], stars1['mag'],
                                             stars3['x'], stars3['y'], stars3['mag'], dr_tol=2)


    plt.quiver(stars1['x'][idx1_2], stars1['y'][idx1_2], stars1['x'][idx1_2] - stars2['x'][idx2_1],
                   stars1['y'][idx1_2] - stars2['y'][idx2_1], color='green')
    
