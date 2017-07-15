import numpy as np
import pylab as py
from astropy.table import Table
from gcwork import starset
import pdb

def make_new_labels(align_root, target='ob150029'):
    s = starset.StarSet(align_root)

    # Get name and positions. No velocities yet.
    name = s.getArray('name')
    x = s.getArray('x')
    y = s.getArray('y')
    m = s.getArray('m')
    pdb.set_trace()

    # Find the target of interest
    tdx = name.index(target)

    dx = x - x[tdx]
    dy = y - y[tdx]

    xarc = -dx * 0.00995
    yarc = dy * 0.00995
    rarc = np.hypot(xarc, yarc)

    # Assign new names to all the stars.
    new_name = []
    new_name_fmt = 'S{0:02d}_{1:02.0f}_{2:3.1f}'

    for ss in range(len(name)):
        new_name_ss = new_name_fmt.format(ss, m[ss], rarc[ss])
        new_name.append(new_name_ss)

    new_name = np.array(new_name)

    print(new_name)

    
