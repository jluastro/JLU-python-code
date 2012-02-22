import numpy as np

def azimuthalAverage(image, center=None, ignoreNAN=False):
    """
    Calculate the azimuthally averaged radial profile.

    image - The 2D image
    center - The [x,y] pixel coordinates used as the center. The default is 
             None, which then uses the center of the image (including 
             fracitonal pixels).
    

    Output:
    radii, radial_prof, stddev_prof, nr

    radii - array of radii
    radial_prof - array of average values for each radius bin
    stddev_prof - array of stddev value for each radius bin
    nr - the number of pixels that went into each radius bin
    """
    # Calculate the indices from the image
    y, x = np.indices(image.shape)

    if not center:
        center = np.array([(x.max()-x.min())/2.0, (x.max()-x.min())/2.0])

    r = np.hypot(x - center[0], y - center[1])

    # Get sorted radii
    ind = np.argsort(r.flat)
    r_sorted = r.flat[ind]
    i_sorted = image.flat[ind]

    # Get the integer part of the radii (bin size = 1)
    r_int = r_sorted.astype(int)

    # Find all pixels that fall within each radial bin.
    deltar = r_int[1:] - r_int[:-1]  # Assumes all radii represented
    rind = np.where(deltar)[0]       # location of changed radius

    radii = np.arange(len(rind)-1, dtype=float) + 1
    radii *= deltar[rind[:-1]]
    nr = rind[1:] - rind[:-1]        # number of pixels in radius bin
    radial_prof = np.zeros(len(nr), dtype=float)
    stddev_prof = np.zeros(len(nr), dtype=float)
    
    for rr in range(len(rind)-1):
        if not ignoreNAN:
            radial_prof[rr] = i_sorted[rind[rr]:rind[rr+1]].mean()
            stddev_prof[rr] = i_sorted[rind[rr]:rind[rr+1]].std()
        else:
            all_val = i_sorted[rind[rr]:rind[rr+1]]
            good = np.isfinite(all_val)

            radial_prof[rr] = all_val[good].mean()
            stddev_prof[rr] = all_val[good].std()
            nr[rr] = len(good)
            

    return radii, radial_prof, stddev_prof, nr
