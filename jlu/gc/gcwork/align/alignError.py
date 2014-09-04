from gcwork import starset
import pylab as p
import numpy as np
import asciidata

def plot2d(root, epoch):
    s = starset.StarSet(root)

    x = s.getArray('x')
    y = s.getArray('y')
    xerr = s.getArrayFromEpoch(epoch, 'xerr_a')
    yerr = s.getArrayFromEpoch(epoch, 'yerr_a')

    idx = (np.where(xerr > 0))[0]
    x = x[idx]
    y = y[idx]
    xerr = xerr[idx] * 1000.0
    yerr = yerr[idx] * 1000.0

    p.clf()
    #p.quiver([x], [y], [xerr], [yerr], 0.4)
    scale = 3.0e2
    p.scatter(x, y, xerr*scale, c='b', marker='o', alpha=0.5)
    p.scatter(x, y, yerr*scale, c='r', marker='s', alpha=0.5)
    p.scatter([-3.5], [-3.5], 0.5 * scale, c='k', marker='s')
    p.text(-3.1, -3.6, '0.5 mas')

    p.xlabel('X')
    p.ylabel('Y')
    p.title('Epoch #%d:  %8.3f' % (epoch, s.stars[0].years[epoch]))
    p.savefig('plot/salignErr2D.png')

def errVsRadius(root, epoch):
    # Get the star set info
    s = starset.StarSet(root)

    x = s.getArray('x')
    y = s.getArray('y')

    xerr = s.getArrayFromEpoch(epoch, 'xerr_a')
    yerr = s.getArrayFromEpoch(epoch, 'yerr_a')

    idx = (np.where(xerr > 0))[0]

    x = x[idx]
    y = y[idx]
    xerr = xerr[idx] * 1000.0
    yerr = yerr[idx] * 1000.0

    r = np.hypot(x, y)

    p.clf()
    p.semilogy(r, xerr, 'bx')
    p.semilogy(r, yerr, 'r+')
    p.axis([0, 5, 0.05, 5])

    p.xlabel('Radius (arcsec)')
    p.ylabel('Alignment Error (mas)')

    p.title('Epoch #%d:  %8.3f' % (epoch, s.stars[0].years[epoch]))
    p.savefig('plots/alignErrVsRadius.png')


def errVsMag(root, epoch):
    # Get the star set info
    s = starset.StarSet(root)

    xerr = s.getArrayFromEpoch(epoch, 'xerr_a')
    yerr = s.getArrayFromEpoch(epoch, 'yerr_a')
    mag = s.getArray('mag')

    idx = (np.where(xerr > 0))[0]
    mag = mag[idx]
    xerr = xerr[idx] * 1000.0
    yerr = yerr[idx] * 1000.0

    p.clf()
    p.semilogy(mag, xerr, 'bx')
    p.semilogy(mag, yerr, 'r+')
    p.axis([8, 17, 0.05, 5])

    p.xlabel('Magnitude')
    p.ylabel('Alignment Error (mas)')

    p.title('Epoch #%d:  %8.3f' % (epoch, s.stars[0].years[epoch]))
    p.savefig('plots/alignErrVsMag.png')

def compare_a3(root1, root2, epoch):
    s1 = starset.StarSet(root1)
    s2 = starset.StarSet(root2)

    x1 = s1.getArrayFromEpoch(epoch, 'x')
    y1 = s1.getArrayFromEpoch(epoch, 'y')
    xerr_p1 = s1.getArrayFromEpoch(epoch, 'xerr_p')
    yerr_p1 = s1.getArrayFromEpoch(epoch, 'yerr_p')
    xerr_a1 = s1.getArrayFromEpoch(epoch, 'xerr_a')
    yerr_a1 = s1.getArrayFromEpoch(epoch, 'yerr_a')
    name1 = s1.getArray('name')

    x2 = s2.getArrayFromEpoch(epoch, 'x')
    y2 = s2.getArrayFromEpoch(epoch, 'y')
    xerr_p2 = s2.getArrayFromEpoch(epoch, 'xerr_p')
    yerr_p2 = s2.getArrayFromEpoch(epoch, 'yerr_p')
    xerr_a2 = s2.getArrayFromEpoch(epoch, 'xerr_a')
    yerr_a2 = s2.getArrayFromEpoch(epoch, 'yerr_a')
    name2 = s2.getArray('name')

    idx1 = []
    idx2 = []
    for id1 in range(len(name1)):
        # Skip if this star wasn't detected in this epoch for set #1
        if (xerr_p1[id1] <= 0):
            continue
        
        xdiff = x1[id1] - x2
        ydiff = y1[id1] - y2
        diff = p.sqrt(xdiff**2 + ydiff**2)

        id2 = (diff.argsort())[0]

        # Skip if this star wasn't detected in this epoch for set #2
        if (xerr_p2[id2] <= 0):
            continue

        idx1.append(int(id1))
        idx2.append(int(id2))

    x1 = x1[idx1]
    y1 = y1[idx1]
    xerr_p1 = xerr_p1[idx1]
    yerr_p1 = yerr_p1[idx1]
    xerr_a1 = xerr_a1[idx1]
    yerr_a1 = yerr_a1[idx1]

    x2 = x2[idx2]
    y2 = y2[idx2]
    xerr_p2 = xerr_p2[idx2]
    yerr_p2 = yerr_p2[idx2]
    xerr_a2 = xerr_a2[idx2]
    yerr_a2 = yerr_a2[idx2]

    ##########
    #
    #  Plots
    #
    ##########
    p.clf()
    p.quiver([x1], [y1], [x1 - x2], [y1 - y2], 0.6)
    p.quiver([[-7]], [[-5]], [[0.001]], [[0]], 0.6)
    p.text(-6, -5.2, '1 mas')

    p.xlabel('X')
    p.ylabel('Y')
    p.title('Compare -a 2 vs. -a 3')

    
