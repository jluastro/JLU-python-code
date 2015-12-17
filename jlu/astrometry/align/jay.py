from jlu.astrometry.align import ord_brite
import numpy as np
import itertools
import pylab as py
import scipy.signal
import pdb
from skimage import transform
from scipy.spatial import cKDTree as KDT
from scipy.spatial import distance
from collections import Counter
from skimage import transform
import math

_NMATMAX_ = 199

search_radii = [99.00, 50.00, 25.00, 15.00, 10.00,
                 8.00,  5.00,  3.50,  2.00,  1.50,
                 1.20,  1.00,  0.80,  0.60,  0.45,
                 0.35,  0.25,  0.20,  0.16,  0.12,
                 0.09,  0.07,  0.05,  0.04,  0.03]
search_indices = np.arange(len(search_radii))


def miracle_match(xin1, yin1, min1, xin2, yin2, min2,
                  x1b, y1b, m1b, x2b, y2b, m2b,
                  NITREJ):

    # Check input parameters
    if (NITREJ < 1 or NITREJ > 25):
        msg = "miracle_match NITREJ must be between 1 and 25 (NITREJ = {0})"
        raise ValueError(msg.format(NITREJ))

    NIN1 = len(xin1)
    NIN2 = len(xin2)
    NBUOY = len(x1b)
    
    print 'MM:  '
    print 'MM: ENTER miracle_match()'
    print 'MM:  '
    print 'MM: MM --->  length of list 1: ', NIN1
    print 'MM: MM --->  length of list 2: ', NIN2
    print 'MM: MM --->  N matches: ', NBUOY
    print 'MM:   '
    print 'MM: INPUT PROPS:  '
    print 'MM:   '
    fmt =  ' {0:11s}  list1 = {1:11.2f}  list2 = {2:11.2f}'
    print fmt.format('MM:  xmin: ', xin1.min(), xin2.min())
    print fmt.format('MM:  xmax: ', xin1.max(), xin2.max())
    print fmt.format('MM:  ymin: ', yin1.min(), yin2.min())
    print fmt.format('MM:  ymax: ', yin1.max(), yin2.max())
    print fmt.format('MM:  mmin: ', min1.min(), min2.min())
    print fmt.format('MM:  mmax: ', min1.max(), min2.max())
    print 'MM:   '
    print 'MM:   '

    # If we already have a few matches, print them out to the screen.
    if (NBUOY > 0) and (NBUOY < 10):
        hdr = 'MM: {:4s} {:11s} {:11s} {:11s} {:11s}'
        fmt = 'MM: {:4d} {:11.4f} {:11.4f} {:11.4f} {:11.4f}'

        print hdr.format('ii', 'X1', 'Y1', 'X2', 'Y2')
        for ii in range(NBUOY):
            print fmt.format(ii, x1b[ii], y1b[ii], x2b[ii], y2b[ii])

    if (NBUOY == 0):
        print ' '
        print 'SEE IF THE STANDARD FILE "MIRMAT.HELP" IS AVAILABLE'
        print ' '

        try:
            # Note the file number matches to Jay's code.
            t77 = Table.read('MIRMAT.HELP', format='ascii')
            x1b = t77.columns[0].data
            y1b = t77.columns[1].data
            x2b = t77.columns[2].data
            y2b = t77.columns[3].data
            NBUOY = len(t77)
            
        except IOError:
            # Do nothing if the file doesn't exist... no initial guess needed.
            pass

    # A while loop to find some preliminary matches. Breakout once we have enough.
    while True:
        if (NBUOY > 0):
            print '  '
            print 'STARTING MATCHES ARE AVAILABLE FROM MIRMAT.HELP'
            print '(search for matches with this starting general'
            print ' transformation) '
            print '  '
            print '    NBUOY: ', NBUOY
            print '  '

            # Jay has found that trying to assume the orientation is
            # the same and only an offset is different can help substantially.
            # Note that this only gets done for very very small starlists.
            if (NBUOY < 100) and (NIN1 < 100):
                x2b, y2b = assume_orient(xin2, yin2, min2, x2b, y2b, m2b)

                # Break out of the while loop
                break
        else:
            print ' '
            print 'NO STARTING MATCHES AVAILABLE FROM MIRMAT.HELP'
            print ' '


        ##################################################
        # Test #1
        # Check to see if the two starlists are related
        # by a simple offset.
        ##################################################
        matmeth = 1
        print 'MM:  '
        print 'MM: MM TEST1: call find_offset...'

        NBUOY, OX, OY, x1b, y1b, m1b, x2b, y2b, m2b = find_offset(xin1, yin1, min1,
                                                                  xin2, yin2, min2)

        print 'MM:  '
        print 'MM: MM TEST1: found:  ', NBUOY
        print 'MM:  '
        print 'MM:                      '
        print 'MM:           -----------'
        print 'MM:           OC: OX: ',OX
        print 'MM:           OC: OY: ',OY
        print 'MM:           -----------'
        print 'MM:                      '
        print 'MM:  '

        if ((NBUOY >= (0.50 * len(xin1)) or (NBUOY >= (0.50 * len(xin2)))) and
            (NBUOY >= 5)):
            break   # good! we have found enough matches


        ##################################################
        # Test #2
        # Try the standard matching procedure.
        ##################################################
        matmeth = 2
        print 'MM:  '
        print 'MM: MM TEST2: call MM-3D...'
        print 'MM:  '

        NBUOY, x1b, y1b, m1b, x2b, y2b, m2b = miracle_match_3d(xin1, yin1, min1,
                                                               xin2, yin2, min2)

        print 'MM:  '
        print 'MM: MM TEST2: found:  ',NBUOY
        print 'MM:  '
        print 'MM:  '
        print 'MM:  '

        if (NBUOY >= 10):
            break

        ##################################################
        # Test #3
        # Use only the brightest 25 stars to find the match.
        ##################################################
        matmeth = 3
        print 'MM:  '
        print 'MM: MM TEST3:  call MM25 (use brightest 25)...'
        print 'MM:  '

        NBUOY, x1b, y1b, m1b, x2b, y2b, m2b = miracle_match_briteN(xin1, yin1, min1,
                                                                   xin2, yin2, min2, 25)

        print 'MM:   '
        print 'MM: MM TEST3:  found:  ',NBUOY
        print 'MM:   '

        if (NBUOY >= 10):
            break


        ##################################################
        # Test #4
        # Use local triangles for the brightest stars.
        ##################################################
        matmeth = 4
        print 'MM:  '
        print 'MM: MM TEST4:  call MM99 (local triangles)...'
        print 'MM:  '

        x1b, y1b, m1b, x2b, y2b, m2b = miracle_match_99(xin1, yin1, min1,
                                                        xin2, yin2, min2)

        NBUOY = len(x1b)

        print 'MM:   '
        print 'MM: MM TEST3:  found:  ',NBUOY
        print 'MM:   '

        if (NBUOY >= 10):
            break


        ##################################################
        # Test #5
        # Do an intensive check, assuming the scale of the
        # two images is the same.
        ##################################################
        matmeth = 5
        print 'MM:  '
        print 'MM: MM TEST5:  call MM5C (intense scale-fixed)...'
        print 'MM:  '

        NBUOY, x1b, y1b, m1b, x2b, y2b, m2b = miracle_match_5C(xin1, yin1, min1,
                                                               xin2, yin2, min2)

        print 'MM:   '
        print 'MM: MM TEST3:  found:  ',NBUOY
        print 'MM:   '

        if (NBUOY >= 5):
            break

        print 'MM: MM ---> '
        print 'MM: MM ---> NOT ENOUGH BUOY STARS FOUND'
        print 'MM: MM ---> '
        print 'MM: MM ---> DIED IN MMIRMATM.F '
        print 'MM: MM ---> '
        print 'MM: MM ---> RETURN NO MATCHES:   NBUOY = 0'

        NBUOY = 0

        return


    # WE MADE IT -- SUCCESSFULLY matched stars
    print 'MM:  '
    print 'MM:  '
    print 'MM: MM: COPY THE MATCHES TO A TEMPORARY ARRAY...   ', NBUOY
    print 'MM:  '

    x1bt = x1b.copy()
    y1bt = y1b.copy()
    m1bt = m1b.copy()
    x2bt = x2b.copy()
    y2bt = y2b.copy()
    m2bt = m2b.copy()
    NBT = len(x1bt)

    #-------------------------------------------------
    #
    # this is the starting "search" radius; disc=3 is a decent number, but if you
    # have some crude measurements or are comparing ground-based against space
    # based where the scales are very different, you may need to increase this
    #
    # the routine below will take the two raw star lists and the implied
    # transormation and will search for matches within the specified radius
    # "disc"
    #
    NITNET = 7
    if (NITREJ < NITNET):
        NITNET = NITREJ - 1
    if (NITNET < 0):
        NITNET = 0
    disc = search_radius[NITNET]
 
    print 'MM:            '
    print 'MM:            '
    print 'MM: PURIFY (remove inconsistent matches) '
    print 'MM:            '
    print 'MM:            '
    #discu = max(5.0, disc)
    discu = max(2.0, disc)

    print 'MM: purify     NBT: ', len(x1bt), discu
    trans, x1bt, y1bt, x2bt, y2bt = purify_buoy(x1bt, y1bt, x2bt, y2bt, discu)
    print 'MM: purified   NBT: ', len(x1bt), discu


    
    print 'MM: '
    print 'MM: '
    print 'MM: '
    print 'MM: call incorp  12: ',NIN1,NIN2
    print 'MM:             NBT: ',NBT
    print 'MM:          NITREJ: ',NITREJ
    print 'MM:          NITNET: ',NITNET
    print 'MM:            disc: ',DISC
    print 'MM: '
 
    incorp_buoy(x1bt,y1bt,x2bt,y2bt,NBT,
                xin1,yin1,min1,NIN1,
                xin2,yin2,min2,NIN2,
                x1b,y1b,m1b,
                x2b,y2b,m2b,NBUOY,DISC)
    print 'MM:  '
    print 'MM: exit incorp  12: '
    print 'MM:  '
    print 'MM:  NBUOY: ',NBUOY
    print 'MM: NITREJ: ',NITREJ
    print 'MM:  '
 
 
    # ----------------------------------------------
    # 
    #  this routine will remove the stray matches, allowing in
    #  the end only those that lie within the specified finding
    #  radius---as specified by darray(NITREJ)
    # 
    print 'MM:  '
    print 'MM: call purge_buoy: '
    print 'MM:  '
    for NIT in range(NITREJ):
        purge_buoy(x1b,y1b,m1b,
                   x2b,y2b,m2b,NBUOY,NIT)
    print 'MM:  '
    print 'MM: exit purge_buoy '
    print 'MM:  '
 
    print 'MM:  '
    print 'MM: NBUOY: ',NBUOY
    print 'MM:  '
    # 193  format(1x,a11,2f11.2)
    # if (NBUOY < 10):
    #     if (matmeth.eq.1) goto 112
    #     if (matmeth.eq.2) goto 113
    #     if (matmeth.eq.3) goto 114
    #     if (matmeth.eq.4) goto 115

    # print 'MM:  '
    # print 'MM:  '
    #   write( 193) 'MM:  xmin: ',minarr(x1b,NBUOY),minarr(x2b,NBUOY)
    #   write( 193) 'MM:  xmax: ',maxarr(x1b,NBUOY),maxarr(x2b,NBUOY)
    #   write( 193) 'MM:  ymin: ',minarr(y1b,NBUOY),minarr(y2b,NBUOY)
    #   write( 193) 'MM:  ymax: ',maxarr(y1b,NBUOY),maxarr(y2b,NBUOY)
    #   write( 193) 'MM:  mmin: ',minarr(m1b,NBUOY),minarr(m2b,NBUOY)
    #   write( 193) 'MM:  mmax: ',maxarr(m1b,NBUOY),maxarr(m2b,NBUOY)
    #   print 'MM:  '
    #   print 'MM:  '
    #   print 'MM: EXIT MIRMATM.F...'
    #   print 'MM:  '
    #   print 'MM:  '
    #   print 'MM:  '
 
     

    return
    

def assume_orient(x1, y1, x2, y2):
    """
    Calculate all possible delta-x and delta-y shifts between all combinations
    of stars in the two lists. For the true shifts between the two lists, there
    should be an over-abundance of the same delta-x and delta-y values.

    Algorithm: 
    -- Calculate a 2D histogram of delta-x and delta-y between all combinations of
    stars in the two lists.
    -- Convolve the histogram with a weighted window function (5x5 with weights
    decreasing with increasing distance from the center of the window). This
    smooths out any noise due to positional errors or proper motions.
    -- Find the point in the convolved histogram where there is a peak.

    Note that this code is extremely slow for arrays larger than 100 stars.
    """
    N1 = len(x1)
    N2 = len(x2)

    # Calculating positional differences
    x_pairs = cartesian_product((x1, x2))
    y_pairs = cartesian_product((y1, y2))
    dx = x_pairs[:,0] - x_pairs[:,1]
    dy = y_pairs[:,0] - y_pairs[:,1]

    #dxy_max = np.max( np.abs( np.concatenate( (dx, dy) ) ) )
    #bin_size = 2.0 * dxy_max / 401.0
    #Nbins = np.arange(-dxy_max, dxy_max, bin_size)
    bin_size = 1
    Nbins = np.arange(-200, 200+1)

    print 'Making 2D histogram of dx and dy'
    dxy_hist, dx_edges, dy_edges = np.histogram2d(dx, dy, bins=Nbins)

    # We are going to do a 2D windowed summation where the
    # window is 5 x 5 and outer pixels are de-weighted with
    # respect to the inner pixels.
    win_tmp = np.abs( np.mgrid[-2:2+1, -2:2+1] )  # supplies 2 x (5x5) arrays
    window = np.sum(win_tmp, axis=0)
    window_wgt = 1.0 - (window / 10.0)

    print 'Convolving dx x dy histogram by 5 x 5 window.'
    dxy_hist_cnvlv = scipy.signal.convolve2d(dxy_hist, window_wgt, mode='same')

    print 'Finding peak'
    amax = dxy_hist_cnvlv.max()
    idx_dx, idx_dy = np.where(dxy_hist_cnvlv == amax)

    # Now fetch the dx and dy value where we have the most matches. 
    dx_good = dx_edges[idx_dx[0]] + bin_size / 2.0
    dy_good = dy_edges[idx_dy[0]] + bin_size / 2.0

    print '    '
    print '  assume_orient:---> Nstars at peak of dx, dy histogram = ', amax
    print '  assume_orient:---> best dx = ', dx_good
    print '  assume_orient:---> best dy = ', dy_good
    print '    '
 
    # Apply dx and dy fix to the second star list.
    x2_new = x2 + dx_good
    y2_new = y2 + dy_good

    return x2_new, y2_new


def find_offset(x1, y1, m1, x2, y2, m2):
    print 'ENTER...   offset check FASTER... (slower)'
    print '     ---   designed for large offsets, '
    print '           and low precision...'

    Ns = len(x1)
    NNs = len(x2)
    Nu = Ns
    NNu = NNs

    xmin = x2.min()
    xmax = x2.max()
    ymin = y2.min()
    ymax = y2.max()

    print ''
    print '  find_offset: Nu  : ', Nu
    print '  find_offset: NNu  : ', NNu
    print '  find_offset: xmin: ', xmin
    print '  find_offset: xmax: ', xmax
    print '  find_offset: ymin: ', ymin
    print '  find_offset: ymax: ', ymax
    print ''

    # Build a hash map
    Nhash = 501
    Nhash_half = Nhash / 2

    # Calculating positional differences
    x_pairs = cartesian_product((x1, x2))
    y_pairs = cartesian_product((y1, y2))
    m_pairs = cartesian_product((m1, m2))
    dx = x_pairs[:,1] - x_pairs[:,0]
    dy = y_pairs[:,1] - y_pairs[:,0]

    dx_demag = (dx / 2.0) + 0.5
    dy_demag = (dy / 2.0) + 0.5

    bins = np.arange(-Nhash_half, Nhash_half + 1)
    dxy_hist_demag, dxy_xedge, dxy_yedge = np.histogram2d(dx_demag, dy_demag, bins=bins)

    # Sum up over a 2x2 window.
    dxy_hist_cnvlv = np.array(scipy.ndimage.uniform_filter(dxy_hist_demag, 2) * 4, dtype=np.int)

    # Search for the peak of the convolved histogram.
    # Something odd... we take the peak over the whole histogram. Jay skips the last
    # row and column... why?
    idx_max = np.unravel_index(dxy_hist_cnvlv.argmax(), dxy_hist_cnvlv.shape)
    dxy_hist_cnvlv_max = dxy_hist_cnvlv[idx_max[0], idx_max[1]]

    # Now search for a second peak in smaller subset of the convolved histogram.
    # I don't really understand why this is only searching the first 100 rows/columns;
    # but this is what Jay's code is doing.

    # --- First mask out a 10 x 10 window around the first peak.
    dxy_hist_tmp = dxy_hist_cnvlv.copy()
    mask_xlo = idx_max[0] - 5
    mask_xhi = idx_max[0] + 5 + 1
    mask_ylo = idx_max[1] - 5
    mask_yhi = idx_max[1] + 5 + 1
    dxy_hist_tmp[mask_xlo : mask_xhi, mask_ylo : mask_yhi] = 0

    # --- Select the first 100 x 100 pixels, don't know why
    dxy_hist_tmp = dxy_hist_cnvlv[0:100, 0:100]

    # --- calculate the max in this region.
    idx_max2 = np.unravel_index(dxy_hist_tmp.argmax(), dxy_hist_tmp.shape)
    dxy_hist_cnvlv_max2 = dxy_hist_tmp[idx_max2[0], idx_max2[1]]


    # Print out a representation of the dx/dy histogram
    print_box(dxy_hist_demag, bins, 15)

    print ''
    print '  find_offset: dx/dy hist max at ix = ', idx_max[0]
    print '  find_offset: dx/dy hist max at iy = ', idx_max[1]
    print '  find_offset: dx/dy hist       max = ', dxy_hist_cnvlv_max
    print '  find_offset: dx/dy hist   2nd max = ', dxy_hist_cnvlv_max2
    print '' 
    print '  find_offset: ABC '

    print_box(dxy_hist_demag, bins, 10, center=idx_max)

    # Use the peak pixel as the best offset.
    # Remember, this offset is still in units of 2.0 pixels
    OX = bins[idx_max[0]] + 0.5
    OY = bins[idx_max[1]] + 0.5
    print ''
    print '  find_offset: Nu: ', Nu
    print '  find_offset: X offset (in 2 pix units): ', OX
    print '  find_offset: Y offset (in 2 pix units): ', OY

    # Convert offsets back to normal pixel units.
    # This is still the peak-derived offset.
    OX *= 2
    OY *= 2
    print ''
    print '  find_offset: ADJ-----'
    print '  find_offset: X offset of peak: ', OX
    print '  find_offset: Y offset of peak: ', OY
 
    print '  find_offset:  '
    print '  find_offset:   Nu: ', Nu
    print '  find_offset:  NNu: ', NNu
    print '  find_offset:  '

    # Find the first _NMATMAX_ stars with offsets less than 5 pixels
    # (after correcting for the global offset). Then compute the median
    # of these offsets (original coordinates) and take that as the new offset.
    dx_off = -dx + OX
    dy_off = -dy + OY
    dd_off = np.hypot(dx_off, dy_off)

    idx = np.where(dd_off < 5)
    idx_keep = idx[0:_NMATMAX_]

    # Coordinates of low-offset stars.
    x1b = x_pairs[idx_keep, 0]
    y1b = y_pairs[idx_keep, 0]
    m1b = m_pairs[idx_keep, 0]
    x2b = x_pairs[idx_keep, 1]
    y2b = y_pairs[idx_keep, 1]
    m2b = m_pairs[idx_keep, 1]
    
    # Final offsets. Note the sign flip from up above.
    OX = (-dx[idx_keep]).mean()
    OY = (-dy[idx_keep]).mean()

    print '  find_offset: NEW -----'
    print '  find_offset: Number of matches = ', len(idx_keep)
    print '  find_offset: X Offset final = ', OX
    print '  find_offset: Y Offset final = ', OY
    print '  find_offset:'

    return (len(x1b), OX, OY, x1b, y1b, m1b, x2b, y2b, m2b)
    
        
    

def print_box(dxy_hist, dxy_bins, box_half, center=None):
    """
    Print a string representation of the cells
    within <box_half> of the center point. 
    """
    # This is the real center point of the input array.
    mid = np.array(dxy_hist.shape, dtype=np.int16) / 2

    # Decide the point where we want to print the box.
    if center == None:
        prt_point = mid
    else:
        prt_point = np.array(center, dtype=np.int16)

    ##########
    # We will print a box around the specified center. There will
    # be a row and column header with the offsets from the true
    # center of the array.
    ##########
    # make a generic box of the specified size.
    ii = np.arange(-box_half, box_half + 1, dtype=np.int16)
    # indices into the input array
    ix = ii + prt_point[0]
    iy = ii + prt_point[1]

    # Trim the indices down to valid values (if the array is too small).
    ix = ix[(ix >= 0) & (ix < dxy_hist.shape[0])]
    iy = iy[(iy >= 0) & (iy < dxy_hist.shape[1])]
    
    # the subset of the array we want to print
    dxy_hist_print = dxy_hist[ix[0]:ix[-1]+1, iy[0]:iy[-1]+1]

    # Print the column headers.
    prt_str = '{:4d}   ' + ('{:4.0f} ' * dxy_hist_print.shape[1])
    print prt_str.format(0, *(dxy_bins[iy]))

    # Loop through and print the row headers and the data values.
    for xx in range(len(ix)):
        print prt_str.format( dxy_bins[ix[xx]], *dxy_hist_print[xx,:] )

    return

    
    

def miracle_match_3d(xin1, yin1, min1, xin2, yin2, min2, mag_limit=-3.5):
    """
    Try ot match two star lists by taking the brightest 50 stars in each list
    and trying every permutation among them to see if we can find a combination
    of them that works.
    """
    # Check to see the lists are long enough.
    Nin1 = len(xin1)
    Nin2 = len(xin2)
    if (Nin1 < 50) or (Nin2 < 50):
        print ''
        print '  MM3D: You need at least 50 to '
        print '  MM3D: find the matches...'
        print '  MM3D: NIN1: ', Nin1
        print '  MM3D: NIN2: ', Nin2
        print ''
        return

    # Put the input coordinates into 2D formats that we will use later.
    coo1_in = np.array([xin1, yin1]).T
    coo2_in = np.array([xin2, yin2]).T

    Nbrite = 50
    
    # Take the Nbrite brightest stars from each list and order by brightness.
    print '  MM3D: '
    print '  MM3D: ORD_BRITE: '
    print '  MM3D: '
    x1, y1, m1 = ord_brite.order_by_brite(xin1, yin1, min1, Nbrite, verbose=True)
    x2, y2, m2 = ord_brite.order_by_brite(xin2, yin2, min2, Nbrite, verbose=True)

    # Put these coordinates into 2D formats that we will use later.
    coo1 = np.array([x1, y1]).T
    coo2 = np.array([x2, y2]).T

    print '  MM3D:  '
    print '  MM3D: TRY PERMUTATIONS, VOTE FOR GOOD ONES'
    print '  MM3D:  '
    tri_lengths1, tri_indices1 = calc_triangles_max_side(x1, y1)

    print '  MM3D:  '
    print '  MM3D: TALLY UP THE GOOD VOTES...'
    print '  MM3D:  '
    tri_lengths2, tri_indices2 = calc_triangles_max_side(x2, y2, N_first_stars=10)

    # Compare each triangle in Frame #2 to all possible triangles in Frame #1
    for ii in range(tri_lengths2.shape[0]):

        # Calculate the quad. sum of all the sides.
        # Find the most compact triangle
        dd = np.sqrt( ((tri_lengths1 - tri_lengths2[ii,:])**2).sum(axis=1) )
        nmin = dd.argmin()
        dmin = dd[nmin]

        if dmin < 2.0:
            # Pull out the X and Y positions for the 3 stars that make up
            # this minimum distance triangle.
            coo1_tri = coo1[tri_indices1[nmin], :]
            coo2_tri = coo2[tri_indices2[ii], :]

            # Calculate the mean positions -- this helps keep the coordinate
            # transformation accurate w.r.t. numerical limits.
            coo1_mean = coo1_tri.mean(axis=0)  # mean X and Y in Frame 1
            coo2_mean = coo2_tri.mean(axis=0)  # mean X and Y in Frame 2

            # Find the best-fit affine transformation
            trans = transform.estimate_transform('affine', coo1_tri - coo1_mean, coo2_tri - coo2_mean)

            # Transform the 50 brightest stars from Frame 1 into Frame 2 coordinates.
            t_coo2 = trans(coo1 - coo1_mean) + coo2_mean

            # Match the Frame 1 (transformed) and Frame 2 bright star lists.
            idx1_50, idx2_50, dr, dm = match(t_coo2[:,0], t_coo2[:,1], m1,
                                             coo2[:,0], coo2[:,1], m2, dr_tol=2.5, dm_tol=None)

            # The number of successful matches:
            N_match_50 = len(idx1_50)

            # Record the matches
            fmt =  ' Tri_2 at [{0:5d}, {1:5d}, {2:5d}] matches Tri_1 #{3:5d} with'
            fmt += ' dlength = {4:4.1f}, sides = [{5:5.1f}, {6:5.1f}, {7:5.1f}], Nmatch_50 = {8:5d}'
            print fmt.format(tri_indices2[ii,0], tri_indices2[ii,1], tri_indices2[ii,2],
                             nmin, dmin,
                             tri_lengths1[nmin, 0], tri_lengths1[nmin, 1], tri_lengths1[nmin, 2],
                             N_match_50)

            # If we have found more than 10 matches (out of 50) matches,
            # then we are in good shape. Re-derive the transformation with
            # these matching stars.
            if N_match_50 >= 10:
                # Coordinates of the matched stars.
                coo1_match = coo1[idx1_50, :]
                coo2_match = coo2[idx2_50, :]

                # Calculate the mean positions-- this helps keep the coordinate
                # transformation accurate w.r.t. numerical limits.
                coo1_match_mean = coo1_match.mean(axis=0) # mean X and Y in frame 1
                coo2_match_mean = coo2_match.mean(axis=0) # mean X and Y in frame 2

                # Find the best-fit affine transformation
                trans = transform.estimate_transform('similarity', coo1_match - coo1_match_mean,
                                                     coo2_match - coo2_match_mean)

                # Transform the entire input list brighter than a magnitude of -3.5.
                mdx1 = np.where(min1 < mag_limit)[0]
                if len(mdx1) == 0:
                    print '  MM3D:  '
                    print '  MM3D: FAILED - no stars brighter than {0:.1f}'.format(mag_limit)
                    print '  MM3D:  '
                    return (0, None, None, None, None, None, None)
                
                coo1_in = coo1_in[mdx1] # Trim down the input coordinate list.
                t_coo2_in = trans(coo1_in - coo1_match_mean) + coo2_match_mean
                
                # Match the Frame 1 (transformed) and Frame 2 input star lists.
                idx1, idx2, dr, dm = match(t_coo2_in[:, 0], t_coo2_in[:, 1], min1,
                                           coo2_in[:, 0], coo2_in[:, 1], min2, dr_tol=1.5, dm_tol=None)
                x1_match = xin1[mdx1][idx1]
                y1_match = yin1[mdx1][idx1]
                m1_match = min1[mdx1][idx1]
                x2_match = xin2[idx2]
                y2_match = yin2[idx2]
                m2_match = min2[idx2]
                
                # The number of successful matches:
                N_match = len(idx1)

                # Return the matched, not-transformed starlists.
                return (N_match, x1_match, y1_match, m1_match, x2_match, y2_match, m2_match)

    return (0, None, None, None, None, None, None)
    

def miracle_match_briteN(xin1, yin1, min1, xin2, yin2, min2, Nbrite,
                         Nbins_vmax=200, Nbins_angle=360,verbose=False):
    """
    Take two input starlists and select the <Nbrite> brightest stars from
    each. Then performa a triangle matching algorithm along the lines of
    Groth 1986.

    For every possible triangle (combination of 3 stars) in a starlist,
    compute the ratio of two sides and the angle between those sides.
    These quantities are invariant under scale and rotation transformations.
    Use a histogram of these quantities to vote for possible matching
    triangles between the two star lists. Then take the vote winners
    as actual matches.

    There may be some sensitivity to the bin sizes used in the histogram
    for vmax (ratio of two sides) and the angles. The larger the positional
    and brightness uncertainties, the more bigger the bin sizes should really
    be. But this isn't well tested.
    """
    print ''
    print '  miracle_match_briteN: mirmat50: use brightest 50'
    print '  miracle_match_briteN:  '
    print '  miracle_match_briteN:  '

    # Get/check the lengths of the two starlists
    nin1 = len(xin1)
    nin2 = len(xin2)

    if (nin1 < Nbrite) or (nin2 < Nbrite):
        print 'You need at least {0} to '.format(Nbrite)
        print 'find the matches...'
        print 'NIN1: ', nin1
        print 'NIN2: ', nin2
        return (0, None, None, None, None, None, None)

    # Take the Nbrite brightest stars from each list and order by brightness.
    print '  miracle_match_briteN: '
    print '  miracle_match_briteN: ORD_BRITE: '
    print '  miracle_match_briteN: '
    x1, y1, m1 = ord_brite.order_by_brite(xin1, yin1, min1, Nbrite, verbose=verbose)
    x2, y2, m2 = ord_brite.order_by_brite(xin2, yin2, min2, Nbrite, verbose=verbose)
    
    ####################
    #
    # Triangle Matching
    #
    ####################
    print '  miracle_match_briteN: '
    print '  miracle_match_briteN: DO Matching Triangles search...'
    print '  miracle_match_briteN: '

    # These are the bins for the 2D (vmax, angle) array we will be making later.
    bins_vmax = np.arange(-1.0, 1.01, 2.0 / Nbins_vmax)
    bins_angle = np.arange(0, 360+1, 360.0 / Nbins_angle)


    ##########
    # List 1
    ##########
    # Make triangles for all combinations within the first starlist.
    stars_in_tri1, vmax1, angle1 = calc_triangles_vmax_angle(x1, y1)

    # Over 2D (vmax, angle) space, decide where everything goes.
    # We only care about the first instance in any bin.
    idx1_vmax_hist = np.digitize(vmax1, bins_vmax) - 1  # indices into the 2D arraya
    idx1_angl_hist = np.digitize(angle1, bins_angle) - 1

    # Make a 2D array and every position is a 3 element vector containing the indicies
    # for the points in this triangle. At each 2D position, we only have a single
    # triangle recorded. We want the first insance. So we just go in reverse order
    # of the triangles and the last updates are the first entries in the original
    # array.
    stars1_at_hist = np.ones((len(bins_vmax)-1, len(bins_angle)-1, 3), dtype=np.int16) * -1
    stars1_at_hist[idx1_vmax_hist[::-1], idx1_angl_hist[::-1], :] = stars_in_tri1[::-1]

    ##########
    # List 2
    ##########
    # Make triangles for all combinations within the second starlist.
    stars_in_tri2, vmax2, angle2 = calc_triangles_vmax_angle(x2, y2)

    # Over 2D (vmax, angle) space, decide where everything goes.
    # We only care about the first instance in any bin.
    idx2_vmax_hist = np.digitize(vmax2, bins_vmax) - 1  # indices into the 2D arraya
    idx2_angl_hist = np.digitize(angle2, bins_angle) - 1

    ##########
    # Possible Matches
    ##########
    # Find the triangles that have the same vmax and angle in list 1 and list 2.
    stars_in1_matches2 = stars1_at_hist[idx2_vmax_hist, idx2_angl_hist, :]

    ##########
    # Tally Votes
    ##########
    # Now vote for all stars in the triangles that have possible matches (same vmax, angle)
    # between the first and second lists.
    votes = np.zeros((Nbrite, Nbrite))
    
    matches = np.where(stars_in1_matches2[:,0] >= 0)[0]
    match_stars1 = stars_in1_matches2[matches,:]
    match_stars2 = stars_in_tri2[matches,:]
    # Ideally I would like to do:
    #votes[match_stars1[:,0], match_stars2[:,0]] += 1   # vote for 1st star in triangle
    #votes[match_stars1[:,1], match_stars2[:,1]] += 1   # vote for 2nd star in triangle
    #votes[match_stars1[:,2], match_stars2[:,2]] += 1   # vote for 3rd star in triangle
    # But python doesn't handle this properly... repeat occurences don't respond to +1

    add_votes(votes, match_stars1[:,0], match_stars2[:,0])
    add_votes(votes, match_stars1[:,1], match_stars2[:,1])
    add_votes(votes, match_stars1[:,2], match_stars2[:,2])
    
    ##########
    # Find matching triangles with most votes (and that pass threshold)
    ##########
    # Reverse sort along the columns. Each column is a star in list #2.
    # For each star in list #2, sort the votes over all the different stars in list #1.
    votes_sdx = votes.argsort(axis=0)[::-1]
    tmp = votes[votes_sdx, range(votes.shape[1])]

    # For each star in list #2, figure out if the number of matches exceeds our threshold.
    # The threshold is that for each star in list #2, the highest voted list #1 stars has
    # votes that are 2 * higher than the second highest voted list #1 star.
    good = np.where(tmp[0, :] > (2 * tmp[1, :]))[0]  # good #2 stars

    ##########
    # Return the good matches
    ##########
    print '  miracle_match_briteN: '
    print '  miracle_match_briteN: found {0} matches '.format(len(good))
    print '  miracle_match_briteN: '

    x2_mat = x2[good]
    y2_mat = y2[good]
    m2_mat = m2[good]
    x1_mat = x1[votes_sdx[0, good]]
    y1_mat = y1[votes_sdx[0, good]]
    m1_mat = m1[votes_sdx[0, good]]


    return len(x1_mat), x1_mat, y1_mat, m1_mat, x2_mat, y2_mat, m2_mat


def miracle_match_99(xin1, yin1, min1, xin2, yin2, min2):
    """
    Do a local-based search for similar triangles.

    Originally miracle_match99() in Jay's fortran code.
    """
    _NVOTE_ = 500

    # Set the number of stars to use.
    NUSEA = min([len(xin1), _NVOTE_, _NMATMAX_])
    NUSEB = min([len(xin2), _NVOTE_, _NMATMAX_])

    # 
    # Step 0: copy the brightest _NMATMAX_ into arrays
    #
    x1, y1, m1 = ord_brite.order_by_brite(xin1, yin1, min1, _NMATMAX_, verbose=False)
    x2, y2, m2 = ord_brite.order_by_brite(xin2, yin2, min2, _NMATMAX_, verbose=False)
    
    #
    # Step 1: Print the _NMATMAX_ brightest in each list
    #
    print 'MM99: show the {0:d} brightest...'.format(_NMATMAX_)
    fmt = 'MM99:      {:5d} ' + 6*'{:8.2f} '
    hdr = 'MM99:      {:5s} ' + 6*'___{:2s}___ '
    print hdr.format('N', 'M1', 'M2', 'X1', 'X2', 'Y1', 'Y2')

    shortest = min([len(x1), len(x2)])
    for ii in range(0, min([shortest, 9]), 1):
        print fmt.format(ii, m1[ii], m2[ii], x1[ii], x2[ii], y1[ii], y2[ii])

    for ii in range(10, min([shortest, 90]), 10):
        print fmt.format(ii, m1[ii], m2[ii], x1[ii], x2[ii], y1[ii], y2[ii])

    for ii in range(100, min([shortest, 500]), 100):
        print fmt.format(ii, m1[ii], m2[ii], x1[ii], x2[ii], y1[ii], y2[ii])

    for ii in range(1000, min([shortest, 50000]), 1000):
        print fmt.format(ii, m1[ii], m2[ii], x1[ii], x2[ii], y1[ii], y2[ii])

    print fmt.format(shortest-1, m1[-1], m2[-1], x1[-1], x2[-1], y1[-1], y2[-1])
    
    #
    # Step 2: For each of these, find the 25 nearest neighbors and fill the
    # hash table with vital info.
    #
    NBRS = 25
    print 'MM99: fill hash...'
    print 'MM99:     NBRS: ', NBRS
    print 'MM99:    NUSEA: ', NUSEA
    print 'MM99:    NUSEB: ', NUSEB


    # Build a KD Tree for searching for nearest neighbors.
    # This is equivalent to, but faster than just doing np.array([x1, y1])
    coords1 = np.empty((x1.size, 2))
    coords1[:, 0] = x1
    coords1[:, 1] = y1
    kdt1 = KDT(coords1)

    neighbors1 = kdt1.query(coords1[0:NUSEA, :], NBRS)
    neigh1_sep = neighbors1[0]
    neigh1_idx = neighbors1[1]
    
    # We will be making a hash table of the values i1 and i2 (see below).
    i12_bins = np.arange(0, 100+1)
    i12_size = len(i12_bins) - 1
    i12_hist = np.zeros((i12_size, i12_size), dtype=np.int16)

    # This will contain the incides to triangles. This should be a ragged array, but
    # it is more efficient to just create this and fill in as necessary.
    i12_stars = np.ones((i12_size, i12_size, 20, 3), dtype=np.int16) * -1

    for n in range(NUSEA):
        if (n % 250) == 0:
            print 'M99:   n: ', n

        # The first nearest-neighbor is itself. Calculate all possible
        # triangles that include this first star and the 25 nearest neighbors.
        xtmp = x1[neigh1_idx[n]]
        ytmp = y1[neigh1_idx[n]]
        tri_lengths, tri_indices_tmp = calc_triangles_max_side(xtmp, ytmp,
                                                               max_side=None, N_first_stars=1)
        tri_indices = neigh1_idx[n][tri_indices_tmp]

        # Calc the ratios of the sides. Remember, for each triangle, the sides
        # have been sorted from least to greatest. Use the ratios:
        #   d1 = side #1 / side #3
        #   d2 = side #2 / side #3
        d1 = tri_lengths[:, 0] / tri_lengths[:, 2]
        d2 = tri_lengths[:, 1] / tri_lengths[:, 2]

        # Calculate indices into a 2D hash-table with 100 x 100 bins.
        # Need to figure out from Jay what is happening here.
        i2 = np.array(100 * d2 * (2 * d2 - 1), dtype=np.int)
        i1 = np.array(100 * (d1 - (1 - d2)) / (d2 - (1 - d2)), dtype=np.int)

        # Decide where everything goes into a 100x100 hash table.
        # Only keep the objects within our hash-table. All others get ignored.
        idx = np.where((i1 >= i12_bins[0]) & (i1 < i12_bins[-1]) &
                       (i2 >= i12_bins[0]) & (i2 < i12_bins[-1]))[0]
        i1 = i1[idx]
        i2 = i2[idx]
        tri_indices = tri_indices[idx]
        tri_lengths = tri_lengths[idx]

        # Drop all but the first triangle that falls within a specific bin.
        # We do this using "unique".
        a = np.array([i1, i2]).T
        b = np.ascontiguousarray(a).view(np.dtype((np.void, a.dtype.itemsize * a.shape[1])))
        foo, good = np.unique(b, return_index=True)

        i1 = i1[good]
        i2 = i2[good]
        tri_indices = tri_indices[good]
        tri_lengths = tri_lengths[good]

        # Make sure we haven't reached our limit of 20 triangles per bin.
        not_too_high = np.where(i12_hist[i1, i2] <= 19)
        i1 = i1[not_too_high]
        i2 = i2[not_too_high]
        tri_indices = tri_indices[not_too_high]
        tri_lengths = tri_lengths[not_too_high]

        # At each 2D position, record the indices of the first triangle found.
        # Save the indices into the next available slot of the 20 that are available at each bin.
        i12_stars[i1, i2, i12_hist[i1, i2], :] = tri_indices

        # Add 1 to our i12_hist everywhere we have a triangle that falls in the bin.
        i12_hist[i1, i2] += 1


    #
    # Step 2: For each of these, find the 25 nearest neighbors and fill the
    # hash table with vital info.
    #
    print 'MM99: unpack hash...'
    print 'MM99:     NBRS: ',NBRS
    print 'MM99:    NUSEA: ',NUSEA
    print 'MM99:    NUSEB: ',NUSEB

    votes = np.zeros((_NVOTE_, _NVOTE_), dtype=np.int)

    coords2 = np.empty((x2.size, 2))
    coords2[:, 0] = x2
    coords2[:, 1] = y2
    kdt2 = KDT(coords2)

    neighbors2 = kdt2.query(coords2[0:NUSEB, :], NBRS)
    neigh2_sep = neighbors2[0]
    neigh2_idx = neighbors2[1]

    for n in range(NUSEB):
        if (n % 250) == 0:
            print 'M99:   n: ', n

        # The first nearest-neighbor is itself. Calculate all possible
        # triangles that include this first star.
        xtmp = x2[neigh2_idx[n]]
        ytmp = y2[neigh2_idx[n]]
        tri_lengths, tri_indices_tmp = calc_triangles_max_side(xtmp, ytmp,
                                                               max_side=None, N_first_stars=1)
        tri_indices = neigh2_idx[n][tri_indices_tmp]

        # Calc the ratios of the sides. Remember, for each triangle, the sides
        # have been sorted from least to greatest. Use the ratios:
        #   d1 = side #1 / side #3
        #   d2 = side #2 / side #3
        d1 = tri_lengths[:, 0] / tri_lengths[:, 2]
        d2 = tri_lengths[:, 1] / tri_lengths[:, 2]

        # Calculate indices into a 2D hash-table with 100 x 100 bins.
        # Need to figure out from Jay what is happening here.
        i2 = np.array(100 * d2 * (2 * d2 - 1), dtype=np.int)
        i1 = np.array(100 * (d1 - (1 - d2)) / (d2 - (1 - d2)), dtype=np.int)

        # Decide where everything goes into a 100x100 hash table.
        # Only keep the objects within our hash-table. All others get ignored.
        idx = np.where((i1 >= i12_bins[0]) & (i1 < i12_bins[-1]) &
                       (i2 >= i12_bins[0]) & (i2 < i12_bins[-1]))[0]
        i1 = i1[idx]
        i2 = i2[idx]
        tri_indices[idx]

        ##########
        # Possible Matches
        ##########
        # Find the triangles that have the same i1 and i2 in list 1 and list 2.
        stars_in1_matches2 = i12_stars[i1, i2, :, :]
        Nstars_in1_matches2 = i12_hist[i1, i2]

        ##########
        # Tally Votes
        ##########
        # Now vote for all stars in the triangles that have possible matches (same i1, i2)
        # between the first and second lists.
        matches = np.where(stars_in1_matches2[:,:,0] >= 0)
        uniq_idx = np.unique(matches[0])
        match_stars1 = stars_in1_matches2[matches[0], matches[1], :]
        match_stars2 = np.repeat(tri_indices[uniq_idx, :], Nstars_in1_matches2[uniq_idx], axis=0)
        if (match_stars1.shape != match_stars2.shape):
            print 'PROBLEM in MM99... '

        add_votes(votes, match_stars1[:,0], match_stars2[:,0])
        add_votes(votes, match_stars1[:,1], match_stars2[:,1])
        add_votes(votes, match_stars1[:,2], match_stars2[:,2])


    ##########
    # Find matching triangles with most votes (and that pass threshold)
    ##########
    print 'MM99: construct BUOY array...'

    # Reverse sort along the columns (2nd index). Each column is a star in list #2.
    # For each star in list #2, sort the votes over all the different stars in list #1.
    votes_sdx = votes.argsort(axis=0)[::-1]
    tmp = votes[votes_sdx, range(votes.shape[1])]

    # For each star in list #2, figure out if the number of matches exceeds our threshold.
    # The threshold is that for each star in list #2, the highest voted list #1 stars has
    # votes that are 2 * higher than the second highest voted list #1 star.
    good = np.where(tmp[0, :] > (2 * tmp[1, :]))[0]  # good #2 stars

    ##########
    # Return the good matches
    ##########
    print '  MM99: '
    print '  MM99: found {0} matches '.format(len(good))
    print '  MM99: '

    x2_mat = x2[good]
    y2_mat = y2[good]
    m2_mat = m2[good]
    x1_mat = x1[votes_sdx[0, good]]
    y1_mat = y1[votes_sdx[0, good]]
    m1_mat = m1[votes_sdx[0, good]]


    return len(x1_mat), x1_mat, y1_mat, m1_mat, x2_mat, y2_mat, m2_mat


def miracle_match_5C(xin1, yin1, min1, xin2, yin2, min2):
    """
    This routine will assume that the scale between the
    two star lists is the same and will search for a match
    subject to that constraint; this removes one degree
    of freedom from the general similar triangles approaches.
    """
    NUSEA = 100
    NUSEB = 200

    NIN1 = len(xin1)
    NIN2 = len(xin2)
 
    print 'MM5C: '
    print 'MM5C:  use brightest stars'
    print 'MM5C:  '
    print 'MM5C:   NIN1: ', NIN1, NUSEA
    print 'MM5C:   NIN2: ', NIN2, NUSEB
    print 'MM5C:  '
 
    if (NIN1 < NUSEA or NIN2 < NUSEB):
        print 'MM5C: '
        print 'MM5C: You need at least a minimum number '
        print 'MM5C: of stars to find the matches...'
        print 'MM5C: NIN1: ', NIN1, '  (need at least ', NUSEA, ') '
        print 'MM5C: NIN2: ', NIN2, '  (need at least ', NUSEB, ') '
        print 'MM5C: '
        print 'MM5C: (return) '
        return (0, None, None, None, None, None, None)

    # 
    # Step 0: copy the brightest NUSE into arrays
    #
    x1, y1, m1 = ord_brite.order_by_brite(xin1, yin1, min1, NUSEA, verbose=False)
    x2, y2, m2 = ord_brite.order_by_brite(xin2, yin2, min2, NUSEB, verbose=False)

    #
    # Step 1: Calculate the distance between the brightest 250 stars and all possible
    # combinations of the other stars in List #1.
    #
    mask = np.ones((800, 800), dtype=np.int) * -1

    dx1 = np.subtract.outer(x1, xin1)
    dy1 = np.subtract.outer(y1, yin1)
    dd1 = np.hypot(dx1, dy1)

    # Indices into the A1 list (x1, bright stars) and the B1 list (xin1, all input).
    dd1_idx_a, dd1_idx_b = np.indices(dd1.shape)  

    # Trim down to those pairs with distances between 20 and 200 pixels.
    idx1 = np.where((dd1 > 20) & (dd1 < 195))
    ia1 = np.array(400 + dx1[idx1], dtype=int)
    ib1 = np.array(400 + dy1[idx1], dtype=int)
    dd1_idx_a = dd1_idx_a[idx1]
    dd1_idx_b = dd1_idx_b[idx1]

    # Count up the unique rows (combos of ia, ib).
    foo, good1 = remove_duplicate_rows(np.array([ia1, ib1]).T, return_index=True)

    dd1_idx_a_good = dd1_idx_a[good1]

    # Record the index of star from the the bright star list.
    # This is what gest used in Step #2.
    # mask[ia1[good1], ib1[good1]] = dd1_idx_a_good

    # KDT instead?
    iab1 = np.array([ia1[good1], ib1[good1]]).T
    indices1 = dd1_idx_a_good
    kdt1 = KDT(iab1)

    # For printing:
    all_a = np.bincount(dd1_idx_a, minlength=NUSEA)
    good_a = np.bincount(dd1_idx_a_good, minlength=NUSEA)

    hdr = '{0:5s} {1:5s} {2:5s} {3:8s} {4:8s} {5:8s}'
    fmt = '{0:5d} {1:5d} {2:5d} {3:8.2f} {4:8.2f} {5:8.2f}'
    print hdr.format(' a ', 'in_rad', 'uniq', 'x1', 'y1', 'm1')
    for aa in range(NUSEA):
        print fmt.format(aa, all_a[aa], good_a[aa], x1[aa], y1[aa], m1[aa])
    

    #
    # Process List 2
    #

    # Initiate arrays. These will be used to contain the number of votes
    # for each possible angle that list 2 should be rotated to match list 1.
    lvote = np.zeros(360, dtype=int)
    lsave = np.ones_like(lvote) * -1
    msave = lsave.copy()

    # For all possible combos of A2 and B2, calculate dx, dy, dd.
    dx2 = np.subtract.outer(x2, xin2)
    dy2 = np.subtract.outer(y2, yin2)
    dd2 = np.hypot(dx2, dy2)

    # Only keep stars with separations between 20 and 195 pixels. Only the first 1000.
    good2 = (dd2 > 20) & (dd2 < 195)
    NB_in_annulus = good2.cumsum(axis=1)
    idx_too_many = np.where(NB_in_annulus > 1000)
    good2[idx_too_many] = False

    # We are going to be checking different rotation angles.
    # We will need these arrays later.
    ang = np.arange(0, 360, 0.1)
    angr = np.radians(ang)
    cosa = np.cos(angr)
    sina = np.sin(angr)

    # Loop through the A2 brite stars
    for aa in range(dd2.shape[0]):  # NUSEB

        dx2_forA = dx2[aa, good2[aa, :]]
        dy2_forA = dy2[aa, good2[aa, :]]

        # 
        # Check all possible rotation angles
        #
        dx2_rot = np.outer( dx2_forA, cosa) + np.outer(dy2_forA, sina) # shape = [NB, ang]
        dy2_rot = np.outer(-dx2_forA, sina) + np.outer(dy2_forA, cosa) 

        ia2 = np.array(400 + dx2_rot, dtype=np.int) # shape = [NB, ang]
        ib2 = np.array(400 + dy2_rot, dtype=np.int)

        #
        # Count the number of votes for each A1 (brite) star from the mask.
        # nvote will be for each angle.
        #
        nvote = np.ones((NUSEA, len(ang)), dtype=np.int16) * -1

        # shape = [NB, ang, 2]
        iab2 = np.array([ia2.T, ib2.T]).T
        search_radius = np.hypot(1, 1)
        nn_dist, nn_idx1 = kdt1.query(iab2, k=9, distance_upper_bound=search_radius)
        pdb.set_trace()

        # Loop through each angle. We have to do this because of the raggedness
        # of the arrays once we apply all our conditions.
        for oo in range(len(ang)):
            # mask stars that aren't in our 0-800 hash mask from above (plus padding)
            good = np.where( (np.abs(dx2_rot[:, oo]) < 395) & (np.abs(dy2_rot[:, oo]) < 395) )[0]
            ia2_oo = ia2[good, oo]
            ib2_oo = ib2[good, oo]

            # Get the 9 nearest neighbor pixels and make sure they are
            # within sqrt(2) * 1.0 pixels.
            iab2 = np.array([ia2_oo, ib2_oo]).T
            search_radius = np.hypot(1.0, 1.0)
            nn_dist, nn_in_tree = kdt1.query(iab2, k=9, distance_upper_bound=search_radius)
            # shape = [NB_good, 9]   for nn_in_tree

            nn_idx_1 = indices1[nn_in_tree]
            nvote[:, oo] = np.bincount(nn_idx_1, minlength=NUSEA)


            # # From the mask, fetch the 9 pixels around each point (3x3 box)
            # m_m1_m1 = mask[ia2_oo-1, ib2_oo-1]
            # m_m1_p0 = mask[ia2_oo-1, ib2_oo  ]
            # m_m1_p1 = mask[ia2_oo-1, ib2_oo+1]
            # m_p0_m1 = mask[ia2_oo  , ib2_oo-1]
            # m_p0_p0 = mask[ia2_oo  , ib2_oo  ]
            # m_p0_p1 = mask[ia2_oo  , ib2_oo+1]
            # m_p1_m1 = mask[ia2_oo+1, ib2_oo-1]
            # m_p1_p0 = mask[ia2_oo+1, ib2_oo  ]
            # m_p1_p1 = mask[ia2_oo+1, ib2_oo+1]

            # # Shape = [NB * 9]
            # mask_iab = np.concatenate([m_m1_p1, m_p0_p1, m_p1_p1,
            #                            m_m1_p0, m_p0_p0, m_p1_p0,
            #                            m_m1_m1, m_p0_m1, m_p1_m1])


            # # shape = [NUSEA]
            # idx = np.where(mask_iab >= 0)
            # # Store the results for this angle in nvote. shape = [NUSEA, ang]
            # nvote[:, oo] = np.bincount(mask_iab[idx], minlength=NUSEA)


        # Find the maximum number of votes over all angles, all B2 stars, and the 3x3 window:
        # nvote_max_ang_idx = the index of the highest voted angle for each A2 star.
        # mvote = how many votes each A1 star got (for this A2 star).
        # shape = [NUSEA]
        nvote_max_ang_idx = nvote.argmax(axis=1)
        mvote = nvote[range(nvote.shape[0]), nvote_max_ang_idx]

        # Sort the A1 stars in reverse order according to their vote counts:
        rdx = mvote.argsort()[::-1]
        mvote = mvote[rdx]

        # highest voted angle for the highest voted A1 stars.
        ang_max = ang[nvote_max_ang_idx[rdx[0]]]
        ang_max = ang_max % 360.0      # Validate that the angle is between 0 and 360.

        # highest voted A1 star.
        a1_max = rdx[0]

        # 
        # For this A2 star, add a vote for the best angle to the lvote variable, which
        # contains 1 bin per 1 degree of angle.
        # Only vote if the highest voted star is significantly above the 2nd best choice.
        # We will split the vote in the two adjacent lvote bins.
        #
        iang = int(ang_max)
        if mvote[0] > (2*mvote[1]):
            lo_bin = int(math.floor(ang_max - 0.5)) % 360
            hi_bin = int(math.floor(ang_max + 0.5)) % 360
            lvote[lo_bin] += 1
            lvote[hi_bin] += 1
        
        # The best angle (int) for this A2 star. Also, the index into the lvote array.
        lsave[aa] = iang

        # Index of A1 star
        msave[aa] = a1_max

        #
        # EXIT OUT if we have succeded.
        #
        if lvote[iang] >= 5:
            print 'MM5C: ---> WE HAVE 5+ VOTES FOR MATCHES!'
            print 'MM5C: ---> iang: ', iang
            print 'MM5C: ---> lvot: ', lvote[iang]
            
            # Only keep those matches that fall near the most recent angle.
            # At least within +/- 1 degree.
            diff_lim = 360 / 2.
            ang_diff = lsave[:aa+1] - lsave[aa]
            ang_diff[ang_diff < -diff_lim] += 360
            ang_diff[ang_diff >  diff_lim] -= 360
            
            idx = np.where(np.abs(ang_diff) <= 1)[0]
            x1b = x1[msave[idx]]
            y1b = y1[msave[idx]]
            m1b = m1[msave[idx]]

            x2b = x2[idx]
            y2b = y2[idx]
            m2b = m2[idx]

            return len(x1b), x1b, y1b, m1b, x2b, y2b, m2b

    return (0, None, None, None, None, None, None)
    
    
def purify_buoy(x1, y1, x2, y2, DMAX):
    """
    This routine will take a list of matched stars and
    will throw out the ones (one at a time) that are not
    consistent to within DMAX.
    """
    N_stars = len(x1)
    
    print 'PB:  '
    print 'PB: PURIFYB: '
    print 'PB:   NBUOY: ', N_stars
    print 'PB:    DMAX: ', DMAX

    # Put the coordinates into NSTARS x 2 arrays
    coo1 = np.array([x1, y1]).T
    coo2 = np.array([x2, y2]).T

    N_keep = 0

    while N_keep < N_stars:
        # Update the number of stars in our current lists.
        N_stars = coo1.shape[0]
        
        # Calculate transformation x2, y2 = T(x1, y1)
        # Actually more like:
        #    x2_new = T_x(x1 - <x1>, y1 - <y1>) + <x2>
        #    y2_new = T_y(x1 - <x1>, y1 - <y1>) + <y2>
        trans, coo1_mean, coo2_mean = glob_fit6nrDP(coo1, coo2)

        # Transform the stars from Frame 1 into Frame 2 coordinates.
        coo2_t = trans(coo1 - coo1_mean) + coo2_mean

        # Calculate the positional residuals for each star sqrt(dx**2 + dy**2)
        dx = coo2[:, 0] - coo2_t[:, 0]
        dy = coo2[:, 1] - coo2_t[:, 1]
        dd = np.hypot(dx, dy)
        residuals = np.sqrt( (dd**2).sum() / len(dd) )

        # Some printing
        if len(x1) < 10:
            for ii in range(x1):
                print '{0:2d} {1:6.3f} {2:6.3f}'.format(ii, dx[ii], dy[ii])

        # Find the star with the maximum residual.
        dd_max_idx = dd.argmax()
        dd_max = dd[dd_max_idx]

        # Set the maximum allowed positional difference to either
        #    90% of maximum observed residual, or
        #    DMAX input variable
        # whichever one is larger.
        DLIM = max(0.9 * dd_max, DMAX)

        idx = np.where(dd < DLIM)[0]

        coo1 = coo1[idx, :]
        coo2 = coo2[idx, :]

        N_keep = coo1.shape[0]

        print 'PB: POSTREJ: limit = {0:.2f} gives {1:d} ---> {2:d}'.format(DLIM, N_stars, N_keep)

    x1_keep = coo1[:, 0]
    y1_keep = coo1[:, 1]
    x2_keep = coo2[:, 0]
    y2_keep = coo2[:, 1]
    
    return x1_keep, y1_keep, x2_keep, y2_keep
    

def calc_triangles_vmax_angle(x, y):
    idx = np.arange(len(x), dtype=np.int16)

    # Option 1 -- this takes 0.217 seconds for 50 objects
    # t1 = time.time()
    # combo_iter1 = itertools.combinations(idx1, 3)
    # combo_idx1_1 = np.array(list(combo_iter1), dtype=np.int16)
    # t2 = time.time()
    # print 'Finished Option 1: ', t2 - t1
    # print combo_idx1_1.shape
    # print combo_idx1_1

    # Option 2 -- this takes 0.016 seconds for 50 objects
    combo_iter = itertools.combinations(idx, 3)
    combo_dt = np.dtype('i2,i2,i2')
    combo_idx_tmp = np.fromiter(combo_iter, dtype=combo_dt)
    combo_idx = combo_idx_tmp.view(np.int16).reshape(-1, 3)

    ii0 = combo_idx[:,0]
    ii1 = combo_idx[:,1]
    ii2 = combo_idx[:,2]

    dxab = x[ii1] - x[ii0]
    dyab = y[ii1] - y[ii0]
    dxac = x[ii2] - x[ii0]
    dyac = y[ii2] - y[ii0]

    dab = np.hypot(dxab, dyab)
    dac = np.hypot(dxac, dyac)

    dmax = np.max([dab, dac], axis=0)
    dmin = np.min([dab, dac], axis=0)
    
    vmax = dmin ** 2 / dmax ** 2
    vmax[dab < dac] *= -1

    vdprod = dxab * dxac + dyab * dyac
    vcprod = dxab * dyac - dyab * dxac
    
    angle = np.degrees( np.arctan2( vdprod, vcprod) )
    angle[angle < 0] += 360.0
    angle[angle > 360] -= 360.0

    return combo_idx, vmax, angle


def calc_triangles_max_side(x, y, max_side=2500, N_first_stars=None):
    """
    Find all 3-point permutations (e.g. triangles)
    """
    # Indices array
    idx = np.arange(len(x), dtype=np.int16)

    # Make all combinations of 3 from the indices array.
    # This is somewhat kludgy in order to keep the array
    # sizes small. Eventually it produces a N x 3 array.
    combo_iter = itertools.combinations(idx, 3)
    combo_dt = np.dtype('i2,i2,i2')
    combo_idx_tmp = np.fromiter(combo_iter, dtype=combo_dt)
    combo_idx = combo_idx_tmp.view(np.int16).reshape(-1, 3)

    # In some cases, we only want to look at the combinations that
    # include the first N stars at one vertex + all possible other
    # stars in the list for the other two vertices.
    if N_first_stars != None:
        foo = np.where(combo_idx[:, 0] < N_first_stars)
        combo_idx = combo_idx[foo]

    ii0 = combo_idx[:,0]
    ii1 = combo_idx[:,1]
    ii2 = combo_idx[:,2]

    # Calculate the lengths of all three sides (ab, ac, bc) for each triangle.
    dxab = x[ii1] - x[ii0]
    dyab = y[ii1] - y[ii0]
    dxac = x[ii2] - x[ii0]
    dyac = y[ii2] - y[ii0]
    dxbc = x[ii2] - x[ii1]
    dybc = y[ii2] - y[ii1]

    dab = np.hypot(dxab, dyab)
    dac = np.hypot(dxac, dyac)
    dbc = np.hypot(dxbc, dybc)

    # Put back into a [N_perm x 3] array
    tri_lengths_tmp = np.array([dab, dac, dbc]).T

    # Make a similar array containing the indices of the point across
    # from the triangle side. Note we reverse the order to match up with 
    # the lengths array.
    tri_indices_tmp = combo_idx[:, ::-1]

    # Trim out any triangles with a side longer than the specified
    # maximum side length.
    if max_side != None:
        idx = np.where((dab < max_side) & (dac < max_side) & (dbc < max_side))[0]
        tri_lengths_tmp = tri_lengths_tmp[idx,:]
        tri_indices_tmp = tri_indices_tmp[idx,:]

    # Sort in increasing order of side length
    axis = 1
    tri_shape = tri_lengths_tmp.shape
    sdx = np.ogrid[:tri_shape[0], :tri_shape[1]]
    sdx[axis] = tri_lengths_tmp.argsort(axis=axis)
    tri_lengths = tri_lengths_tmp[sdx]
    tri_indices = tri_indices_tmp[sdx]

    return tri_lengths, tri_indices



def add_votes(votes, match1, match2):
    # Construct a histogram of how often a bin is matched... then add the delta
    flat_idx = np.ravel_multi_index((match1, match2), dims=votes.shape)

    # extract the unique indices and their position
    unique_idx, idx_idx = np.unique(flat_idx, return_inverse=True)

    # aggregate the repeated indices
    deltas = np.bincount(idx_idx)

    # Sum them to the array
    votes.flat[unique_idx] += deltas

    return


def cartesian_product(arrays):
    la = len(arrays)
    arr = np.empty([len(a) for a in arrays] + [la])
    for i, a in enumerate(np.ix_(*arrays)):
        arr[...,i] = a
    return arr.reshape(-1, la)


def match(x1, y1, m1, x2, y2, m2, dr_tol, dm_tol=None):
    """
    Finds matches between two different catalogs. No transformations are done and it
    is assumed that the two catalogs are already on the same coordinate system
    and magnitude system.

    For two stars to be matched, they must be within a specified radius (dr_tol) and
    delta-magnitude (dm_tol). For stars with more than neighbor (within the tolerances),
    if one is found that is the best match in both brightness and positional offsets
    (closest in both), then the match is made. Otherwise,
    their is a conflict and no match is returned for the star.
    
 
    Parameters
    x1 : array-like
        X coordinate in the first catalog
    y1 : array-like
        Y coordinate in the first catalog (shape of array must match `x1`)
    m1 : array-like
        Magnitude in the first catalog. Must have the same shape as x1.
    x2 : array-like
        X coordinate in the second catalog
    y2 : array-like
        Y coordinate in the second catalog (shape of array must match `x2`)
    m2 : array-like
        Magnitude in the second catalog. Must have the same shape as x2.
    dr_tol : float
        How close (in units of the first catalog) a match has to be to count as a match.
        For stars with more than one nearest neighbor, the delta-magnitude is checked
        and the closest in delta-mag is chosen.
    dm_tol : float or None, optional
        How close in delta-magnitude a match has to be to count as a match.
        If None, then any delta-magnitude is allowed.
 
    Returns
    -------
    idx1 : int array
        Indicies into the first catalog of the matches. Will never be
        larger than `x1`/`y1`.
    idx2 : int array
        Indicies into the second catalog of the matches. Will never be
        larger than `x1`/`y1`.
    dr : float array
        Distance between the matches.
    dm : float array
        Delta-mag between the matches. (m1 - m2)
 
    """
 
    x1 = np.array(x1, copy=False)
    y1 = np.array(y1, copy=False)
    m1 = np.array(m1, copy=False)
    x2 = np.array(x2, copy=False)
    y2 = np.array(y2, copy=False)
    m2 = np.array(m2, copy=False)
 
    if x1.shape != y1.shape:
        raise ValueError('x1 and y1 do not match!')
    if x2.shape != y2.shape:
        raise ValueError('x2 and y2 do not match!')
 
    # Setup coords1 pairs and coords 2 pairs
    # this is equivalent to, but faster than just doing np.array([x1, y1])
    coords1 = np.empty((x1.size, 2))
    coords1[:, 0] = x1
    coords1[:, 1] = y1
 
    # this is equivalent to, but faster than just doing np.array([x1, y1])
    coords2 = np.empty((x2.size, 2))
    coords2[:, 0] = x2
    coords2[:, 1] = y2

    # Utimately we will generate arrays of indices.
    # idxs1 is the indices for matches into catalog 1. This
    # is just a place holder for which stars actually
    # have matches.
    idxs1 = np.ones(x1.size, dtype=int) * -1
    idxs2 = np.ones(x1.size, dtype=int) * -1

    # The matchingn will be done using a KDTree.
    kdt = KDT(coords2)

    # This returns the number of neighbors within the specified
    # radius. We will use this to find those stars that have no or one
    # match and deal with them easily. The more complicated conflict
    # cases will be dealt with afterward.
    i2_match = kdt.query_ball_point(coords1, dr_tol)
    Nmatch = np.array([len(idxs) for idxs in i2_match])

    # What is the largest number of matches we have for a given star?
    Nmatch_max = Nmatch.max()

    # Loop through and handle all the different numbers of matches.
    # This turns out to be the most efficient so we can use numpy
    # array operations. Remember, skip the Nmatch=0 objects... they
    # already have indices set to -1.
    for nn in range(1, Nmatch_max+1):
        i1_nn = np.where(Nmatch == nn)[0]

        if nn == 1:
            i2_nn = np.array([i2_match[mm][0] for mm in i1_nn])
            if dm_tol != None:
                dm = np.abs(m1[i1_nn] - m2[i2_nn])
                keep = dm < dm_tol
                idxs1[i1_nn[keep]] = i1_nn[keep]
                idxs2[i1_nn[keep]] = i2_nn[keep]
            else:
                idxs1[i1_nn] = i1_nn
                idxs2[i1_nn] = i2_nn
        else:
            i2_tmp = np.array([i2_match[mm] for mm in i1_nn])

            # Repeat star list 1 positions and magnitudes
            # for nn times (tile then transpose) 
            x1_nn = np.tile(x1[i1_nn], (nn, 1)).T
            y1_nn = np.tile(y1[i1_nn], (nn, 1)).T
            m1_nn = np.tile(m1[i1_nn], (nn, 1)).T

            # Get out star list 2 positions and magnitudes
            x2_nn = x2[i2_tmp]
            y2_nn = y2[i2_tmp]
            m2_nn = m2[i2_tmp]
            dr = np.abs(x1_nn - x2_nn, y1_nn - y2_nn)
            dm = np.abs(m1_nn - m2_nn)

            if dm_tol != None:
                # Don't even consider stars that exceed our
                # delta-mag threshold. 
                dr_msk = np.ma.masked_where(dm > dm_tol, dr)
                dm_msk = np.ma.masked_where(dm > dm_tol, dm)

                # Remember that argmin on masked arrays can find
                # one of the masked array elements if ALL are masked.
                # But our subsequent "keep" check should get rid of all
                # of these.
                dm_min = dm_msk.argmin(axis=1)
                dr_min = dr_msk.argmin(axis=1)

                # Double check that "min" choice is still within our
                # detla-mag tolerence.
                dm_tmp = np.choose(dm_min, dm)

                keep = (dm_min == dr_min) & (dm_tmp < dm_tol)
            else:
                dm_min = dm.argmin(axis=1)
                dr_min = dr.argmin(axis=1)

                keep = (dm_min == dr_min)

            i2_keep_2D = i2_tmp[keep]
            dr_keep = dr_min[keep]  # which i2 star for a given i1 star
            ii_keep = np.arange(len(dr_keep))  # a running index for the i2 keeper stars.

            idxs1[i1_nn[keep]] = i1_nn[keep]
            idxs2[i1_nn[keep]] = i2_keep_2D[ii_keep, dr_keep]
                

    idxs1 = idxs1[idxs1 >= 0]
    idxs2 = idxs2[idxs2 >= 0]        

    dr = np.hypot(x1[idxs1] - x2[idxs2], y1[idxs1] - y2[idxs2])
    dm = m1[idxs1] - m2[idxs2]

    ##########
    # Deal with duplicates.
    ##########
    # We only have to deal with duplicates in List #2 -- we queried around List #1 coords.
    duplicates = [item for item, count in Counter(idxs2).iteritems() if count > 1]

    if len(duplicates) > 0:
        print '  MATCH: Found {0:d} duplicates'.format(len(duplicates))
        keep = np.ones(len(idxs1), dtpye=bool)

        for dd in range(len(duplicates)):
            dups = np.where(idxs2 == duplicates[dd])[0]
            keep[dups] = False   # right now, these are confused.

            # If the star in list #2 matches best in both position
            # and brightness to the same star, then choose that one
            # as the "correct" match.
            dr_min_idx = dr[dups].argmin()
            dm_min_idx = np.abs(dm[dups]).argmin()

            if dr_min_idx == dm_min_idx:
                keep[dups[dr_min_idx]] = True

            # Otherwise, considered this a case of confusion and drop it
            # as a match.

        # Remove the duplicates that couldn't be resolved.
        idxs1 = idxs1[keep]
        idxs2 = idxs2[keep]
        dr = dr[keep]
        dm = dm[keep]

    return idxs1, idxs2, dr, dm
 
 
    
def glob_fit6nrDP(coo1, coo2):
    """
    Routine that takes a list of stellar
    positions in two different frames and generates a
    least-squares linear transformation between them.
    
    It doesn't do any rejection of stars at all, so it's
    important to make sure that the stars all have "consistent"
    positions.  This means you should look at the residuals
    and make sure that none of the stars transform badly.
    In other words, you want to make sure that input x2 position
    and the transformed x2 position (based on the x1,y1 position)
    are not in disagreement.
    
    One way to do this is to do a transformation with all
    the stars, then look at all the residuals.  Reject the
    stars that have large residuals and regenerate the
    transformations.  You may have to do this several times
    to get reasonable residuals.  Of course you do not want to
    reject *too* many stars.  Often the residuals will have
    some dependence on magnitude; the bright stars can be better
    measured than the faint ones.
    
    All of these considerations should help you choose
    the best stars to use in the transformations.
    
    this routine does a least squares solution (with no
    data rejection or weighting) for the 6-param linear
    fit for:
    
    
    x2 = A*(x1-x1o) + B*(y1-y1o) + x2o
    y2 = C*(x1-x1o) + D*(y1-y1o) + y2o
    
    it may look like there are 8 params, but two of the
    offsets are arbitrary and are just set to the centroid
    of the distribution.
    
    ----
    The inverse transformations are:
    
    x1 = AA*(x2-x2o) + BB*(y2-y2o) + x1o
    y1 = CC*(x2-x2o) + DD*(y2-y2o) + y1o
    
    where:
    
    AA =  D/(A*D-B*C)
    BB = -B/(A*D-B*C)
    CC = -C/(A*D-B*C)
    DD =  A/(A*D-B*C)
    
    
    --------------------------------------------------------------
    
    """

    if coo1.shape[0] < 3:
        return None

    # Calculate the mean positions -- this helps keep the coordinate
    # transformation accurate w.r.t. numerical limits.
    coo1_mean = coo1.mean(axis=0)  # mean X and Y in Frame 1
    coo2_mean = coo2.mean(axis=0)  # mean X and Y in Frame 2

    # Find the best-fit affine transformation
    trans = transform.estimate_transform('affine', coo1 - coo1_mean, coo2 - coo2_mean)

    return trans, coo1_mean, coo2_mean
 
 
 
def incorp_buoy(x1bt, y1bt, x2bt, y2bt,
                x1, y1, m1, x2, y2, m2, DISC):
    """
    Once we have an initial match, this will use that match
    to incoroporate more stars (bright and faint) into a
    more comprehensive list of matches
    """
    NBT = len(x1bt)
    NIN1 = len(x1)
    NIN2 = len(x2)
 
    NCONFL = 0
 
    print 'IB: '
    print 'IB:   ENTER INCORPBM.F: '
    print 'IB:    NBT: ',NBT
    print 'IB:   NIN1: ',NIN1
    print 'IB:   NIN2: ',NIN2
    print 'IB:   DISC: ',DISC
    print 'IB: '
    print 'IB: '

    # Shape = [NIN1, 2]
    coo1 = np.array([x1, y1]).T
    coo2 = np.array([x2, y2]).T

    # shape = [NBT, 2]
    coo1bt = np.array([x1bt, y1bt]).T
    coo2bt = np.array([x2bt, y2bt]).T

    trans, coo1bt_mean, coo2bt_mean = glob_fit6nrDP(coo1bt, coo2bt)
    
    AG = trans._matrix[0, 0]
    BG = trans._matrix[0, 1]
    CG = trans._matrix[1, 0]
    DG = trans._matrix[1, 1]
    Wo = trans._matrix[0, 2] + coo2bt_mean[0]
    Zo = trans._matrix[1, 2] + coo2bt_mean[1]
    Xo = coo1bt_mean[0]
    Yo = coo1bt_mean[1]
    
    fmt = ' IB:  ' + (4 * '{:10.6f} ') + (2 * '{:9.3f} ') + ' {:6d}'
    print fmt.format(AG, BG, CG, DG, Xo - Wo, Yo - Zo, len(x1bt))
    fmt = ' IB:  ' + (4 * '{:9.3f} ')
    print fmt.format(Xo, Yo, Wo, Zo)

    xlo = x2.min()
    xhi = x2.max()
    ylo = y2.min()
    yhi = y2.max()

    print 'IB:  RANGES:'
    print 'IB:   x: ', x1.min(), x1.max()
    print 'IB:   y: ', y1.min(), y1.max()
    print 'IB:   m: ', m1.min(), m1.max()
    print 'IB:  xx: ', xlo, xhi
    print 'IB:  yy: ', ylo, yhi
    print 'IB:  mm: ', m2.min(), m2.max()

    #
    # Create a hash map with 2000 x 2000 bins and place one star
    # per bin for all the stars in list 2 (x2).
    # Only the brightest star will go into the hash map (or KD Tree).
    #
    # Make sure that we have at least 2000 bins to play with.
    if xhi < (xlo + 2000):
        xhi = xlo + 2000
    if yhi < (ylo + 2000):
        yhi = ylo + 2000
 
    # shape = [NIN2]
    ix = np.array(2000 * ((x2 + 0.5) - xlo) / (xhi - xlo), dtype=int)
    iy = np.array(2000 * ((y2 + 0.5) - ylo) / (yhi - ylo), dtype=int)

    # 
    # Find duplicate (ix, iy) entries and select the brightest star to put in the hash map.
    #
    # Sort in order of 1: ix   2: iy  3: m2
    # Duplicate (ix, iy) pairs will be adjacent and the brightest will be first.
    sdx = np.lexsort([m2, iy, ix])    # shape = [NIN2]
    
    # Find the unique (ix, iy) pairs. This will return the first instance, which should be
    # the brightest star of all the duplicates for a given (ix, iy).
    foo, sdx_uni = remove_duplicate_rows( np.array([ix[sdx], iy[sdx]]).T, return_index=True )

    # # Record the appropriate star's index (original index into x2, y2) into the hash map.
    # hash_table[ix[sdx[sdx_uni]], iy[sdx[sdx_uni]]] = sdx[sdx_uni]

    # original indices into ix, iy, x2, y2, m2
    points2_idx = sdx[sdx_uni]   

    # Make a KD Tree of these points (ix, iy) -- this is what we will query later.
    points2 = np.array([ix[points2_idx], iy[points2_idx]]).T
    kdt2 = KDT(points2)

    # Number of conflicts (duplicates) we encountered.
    NCONFL = len(ix) - len(sdx_uni)

    NBUOY = 0

    print 'IB: INCORPBM.F'
    print 'IB:    NBT: ', NBT
    print 'IB:   NCON: ', NCONFL
    print 'IB:   DISC: ', DISC
    print 'IB:   ABCD: ', AG, BG, CG, DG

    #-----
    # Transform the x1 coordinates
    #-----
    # shape = [NIN1, 2]
    coo2g = trans(coo1 - coo1bt_mean) + coo2bt_mean
    x2g = coo2g[:, 0]
    y2g = coo2g[:, 1]

    # shape = [NIN1]
    ixg = np.array(2000 * (x2g - xlo) / (xhi - xlo), dtype=int)
    iyg = np.array(2000 * (y2g - ylo) / (yhi - ylo), dtype=int)

    # shape = [NIN1, 2]
    points1_g = np.array([ixg, iyg]).T

    # Find the nearest 9 neighbor pixels in the hash map.
    # This is more efficient than query_ball_points() because query() returns
    # a square numpy array of ints and we can do further numpy calculations.
    # The alternative query_ball_points() return ragged arrays (array of objects).
    foo, neighbors = kdt2.query(points1_g, k=9)  # shape = [NIN1, 9]

    # Get the positional differences between the transformed coo1 coordinates ("g").
    # and the nearest bright stars from list 2.
    # shape = [NIN1, 9]
    nn_idx2 = points2_idx[neighbors]   
    nn_dx = x2g.reshape((len(x2g), -1)) - x2[nn_idx2]
    nn_dy = y2g.reshape((len(y2g), -1)) - y2[nn_idx2]
    nn_dd = np.abs(nn_dx) + np.abs(nn_dy)   # BEWARE -- not geometric distance.

    # Select the (x2, y2) star that has the smallest positional offset (dx and y average) 
    # for each of the (x1, y1) stars. Keep track of the indices into the (x2, y2) list.
    # shape = NIN1
    nn_min_idx1 = np.arange(nn_dd.shape[0])
    dd_min_idx = nn_dd.argmin(axis=1)
    dd_min = nn_dd[nn_min_idx1, dd_min_idx]
    nn_min_idx2 = nn_idx2[nn_min_idx1, dd_min_idx]

    # Make sure that these "nearest" stars are within our search radius.
    disc_use = max(DISC, 2)
    hash_search_radius = np.hypot(disc_use, disc_use)

    idx_close = np.where(dd_min < disc_use)[0]
    
    x1b = x1[nn_min_idx1[idx_close]]
    y1b = y1[nn_min_idx1[idx_close]]
    m1b = m1[nn_min_idx1[idx_close]]
    x2b = x2[nn_min_idx2[idx_close]]
    y2b = y2[nn_min_idx2[idx_close]]
    m2b = m2[nn_min_idx2[idx_close]]

    print 'IB: '
    print 'IB:   x: ', x1b.min(), x1b.max()
    print 'IB:   y: ', y1b.min(), y1b.max()
    print 'IB:   m: ', m1b.min(), m1b.max()
    print 'IB:  xx: ', x2b.min(), x2b.max()
    print 'IB:  yy: ', y2b.min(), y2b.max()
    print 'IB:  mm: ', m2b.min(), m2b.max()
    print 'IB: '
    print 'IB: ---> NBUOY: ',len(x1b)
    print 'IB: ---> exit IBT...'
    print 'IB: '
 
    return x1b, y1b, m1b, x2b, y2b, m2b
 
 
def remove_duplicate_rows(array_data, return_index=False, return_inverse=False):
    """
    Removes duplicate rows of a multi-dimensional array. Returns the
    array with the duplicates removed. If return_index is True, also
    returns the indices of array_data that result in the unique array.
    If return_inverse is True, also returns the indices of the unique
    array that can be used to reconstruct array_data.
    """

    array_type = np.dtype((np.void, array_data.dtype.itemsize * array_data.shape[1]))

    unique_array_data, index_map, inverse_map = np.unique(
        np.ascontiguousarray(array_data).view(array_type),
        return_index=True, return_inverse=True)
    
    unique_array_data = unique_array_data.view(
            array_data.dtype).reshape(-1, array_data.shape[1])
 
    # unique returns as int64, so cast back
    index_map = np.cast['uint32'](index_map)
    inverse_map = np.cast['uint32'](inverse_map)
    
    if return_index and return_inverse:
        return unique_array_data, index_map, inverse_map
    elif return_index:
        return unique_array_data, index_map
    elif return_inverse:
         return unique_array_data, inverse_map
    
    return unique_array_data
