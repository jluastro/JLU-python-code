from gcwork import starset
from pylab import *
import math

def nepochsOsiris(s=None, root=None):
    """Using a starset, make a scatter plot of the number of epochs.

    @type root: String
    @keyword root: Name of the root file to be passed into a new starset.
    A new starset is created using the default keywords, so if you want
    something special, you should create your own.
    @type s: starset.StarSet
    @keyword s: The StarSet to use for plotting. If both root and the
    starset are passed in, then the starset is re-created using root.
    
    """

    if (root != None):
	s = starset.StarSet(root)

    nepochs = s.getArray('velCnt')
    x = s.getArray('x')
    y = s.getArray('y')

    # Define the color of the points based on the number of 
    # epochs.
    maxN = max(nepochs)
    clf()

    # Cyan
    idx = where(nepochs < (maxN-3))
    scatter(x[idx], y[idx], c='#33ff99', marker='o', faceted=False)

    # light green
    idx = where(nepochs == (maxN-3))
    scatter(x[idx], y[idx], c='#99ff33', marker='o', faceted=False)

    # Yellow
    idx = where(nepochs == (maxN-2))
    scatter(x[idx], y[idx], c='#ffff00', marker='o', faceted=False)

    # Orange
    idx = where(nepochs == (maxN-1))
    scatter(x[idx], y[idx], c='#ff9900', marker='o', faceted=False)

    # Red
    idx = where(nepochs == maxN)
    scatter(x[idx], y[idx], c='#ff0000', marker='o', faceted=False)
    
    # For reference, just plot black crosses on the irs 16 sources
    names = s.getArray('name')
    
    idx = []
    for i in range(len(names)):
	if ((names[i] == 'irs16C') or (names[i] == 'irs16CC') or
	    (names[i] == 'irs16NE') or (names[i] == 'irs16NW') or
	    (names[i] == 'irs16SW')):
	    
	    idx.append(i)

    scatter(x[idx], y[idx], s=30.0, c='#ff0000')
    
    # Now we need to plot the OSIRIS field of view on top. 
    # The smallest narrow-K filter is 1.12x2.24... lets make a box this size.
    # Kn5
    #width = 1.12
    #height = 2.24
    # Kn3
    width = 1.68
    height = 2.24

    ptsX = array([-width/2.0, width/2.0, width/2.0, -width/2.0])
    ptsY = array([-height/2.0, -height/2.0, height/2.0, height/2.0])

    ##########
    # rotate
    ##########

    # Desired PA
    pa = 90.0

    # OSIRIS offset
    print 'If you want PA=%6.1f on SPEC, set the OSIRIS PA=%6.1f' % \
	(pa, pa+10.6)
    
    pa = math.radians(pa)

    cosa = math.cos(pa)
    sina = math.sin(pa)
    
    newX = (ptsX * cosa) + (ptsY * sina)
    newY = -(ptsX * sina) + (ptsY * cosa)

    # Make several boxes
    initOffX = 0.0
    initOffY = 0.0
    boxOffX = 1.0/2.0
    boxOffY = 0.6/2.0

    bx0 = newX + initOffX
    by0 = newY + initOffY
    bx1 = bx0 + boxOffX
    by1 = by0 + boxOffY
    bx2 = bx0 - boxOffX
    by2 = by0 + boxOffY
    bx3 = bx0 + boxOffX
    by3 = by0 - boxOffY
    bx4 = bx0 - boxOffX
    by4 = by0 - boxOffY
    
    fill(bx0, by0, '0.9', alpha=0.2)
    fill(bx1, by1, 'b', alpha=0.2)
    fill(bx2, by2, 'r', alpha=0.2)
    fill(bx3, by3, 'c', alpha=0.2)
    fill(bx4, by4, 'm', alpha=0.2)

    nDx = 0.0
    nDy = 0.5
    eDx = 0.5
    eDy = 0.0

    # X is -East, Y is North... then rotate
    xDx = (-eDx*cosa) + (eDy*sina)
    xDy = -(-eDx*sina) + (eDy*cosa)

    yDx = (nDx*cosa) + (nDy*sina)
    yDy = -(nDx*sina) + (nDy*cosa)

    # Make some arrows to indicate directions:
    arrWidth = 0.1
    north = Arrow(-2, 2, nDx, nDy, facecolor='black', width=arrWidth)
    east = Arrow(-2, 2, eDx, eDy, facecolor='black', width=arrWidth)
    xdir = Arrow(2, 2, xDx, xDy, facecolor='black', width=arrWidth)
    ydir = Arrow(2, 2, yDx, yDy, facecolor='black', width=arrWidth)

    # Get the subplot that we are currently working on
    ax = gca()

    # Now add the arrow
    ax.add_patch(north)
    ax.add_patch(east)
    ax.add_patch(xdir)
    ax.add_patch(ydir)

    # Add arrow labels
    text(-2+nDx, 2+nDy, 'N')
    text(-2+eDx, 2+eDy, 'E')
    text(2+xDx, 2+xDy, 'X')
    text(2+yDx, 2+yDy, 'Y')

    rng = axis('scaled')
    #axis('auto')
    axis([rng[1]-1, rng[0]+1, rng[2]+1, rng[3]-1])

    savefig('scatterNepochsOsiris.jpg')
