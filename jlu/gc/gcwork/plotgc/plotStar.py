import pylab as py
import numarray as na
import numpy as np
import asciidata
from gcwork import starset
from gcwork import objects
import histNofill
import pdb


def go(star,root, align = 'align/align_d_rms_1000_abs_t', poly='polyfit_1000/fit',
             points='points_1000/',fit='none',suffix='',LGSonly=False,legend=True):
    """
    Plot positions and best fit velocity and/or acceleration for star. Possible
    arguments for fit are: linear, accel, both, none.
    """

    py.figure(figsize=(6,6))
    py.clf()
    py.subplots_adjust(left=0.1,bottom=0.1,right=0.98,top=0.95,
                       wspace=0.3,hspace=0.2)

    pointsFile = root + points + star + '.points'
    _tabPoints = asciidata.open(pointsFile)
    date = _tabPoints[0].tonumpy()
    x = _tabPoints[1].tonumpy() * -1.0
    y = _tabPoints[2].tonumpy()
    xerr = _tabPoints[3].tonumpy()
    yerr = _tabPoints[4].tonumpy()

    if fit != 'none':
        if fit == 'linear':
            fitFile = root + poly + '.' + star + '.lfit'
        elif fit == 'accel':
            fitFile = root + poly + '.' + star + '.pfit'
        elif fit =='both':
            fitFile1 = root + poly + '.' + star + '.lfit'
            fitFile2 = root + poly + '.' + star + '.pfit'

    if ((fit != 'both') and (fit != 'none')):
        _fitPoints = asciidata.open(fitFile)
        date_f = _fitPoints[0].tonumpy()
        x_f = _fitPoints[1].tonumpy() * -1.0
        y_f = _fitPoints[2].tonumpy()
        xerr_f = _fitPoints[3].tonumpy()
        yerr_f = _fitPoints[4].tonumpy()
        py.figure(2, figsize=(6.2,5.4))
        py.clf()
        py.plot(x_f, y_f, 'k-')
        py.plot(x_f + xerr_f, y_f + yerr_f, 'k--')
        py.plot(x_f - xerr_f, y_f - yerr_f, 'k--')
    elif ((fit == 'both') and (fit != 'none')):
        _fitPoints = asciidata.open(fitFile1)
        date_f1 = _fitPoints[0].tonumpy()
        x_f1 = _fitPoints[1].tonumpy() * -1.0
        y_f1 = _fitPoints[2].tonumpy()
        xerr_f1 = _fitPoints[3].tonumpy()
        yerr_f1 = _fitPoints[4].tonumpy()
        py.figure(2, figsize=(6.2,5.4))
        py.clf()
        py.plot(x_f1, y_f1, 'k-')
        py.plot(x_f1 + xerr_f1, y_f1 + yerr_f1, 'k--')
        py.plot(x_f1 - xerr_f1, y_f1 - yerr_f1, 'k--')

        _fitPoints = asciidata.open(fitFile2)
        date_f2 = _fitPoints[0].tonumpy()
        x_f2 = _fitPoints[1].tonumpy() * -1.0
        y_f2 = _fitPoints[2].tonumpy()
        xerr_f2 = _fitPoints[3].tonumpy()
        yerr_f2 = _fitPoints[4].tonumpy()
        py.plot(x_f2, y_f2, 'k-')
        py.plot(x_f2 + xerr_f2, y_f2 + yerr_f2, 'k--')
        py.plot(x_f2 - xerr_f2, y_f2 - yerr_f2, 'k--')

    # Find range of plot
    halfRange = max([ abs(x.max() - x.min()), abs(y.max() - y.min()) ]) / 2.0

    padd = 0.001
    xmax = x.min() + ((x.max() - x.min())/2.0) + halfRange + padd
    ymax = y.min() + ((y.max() - y.min())/2.0) + halfRange + padd
    xmin = x.min() + ((x.max() - x.min())/2.0) - halfRange - 2*padd
    ymin = y.min() + ((y.max() - y.min())/2.0) - halfRange - padd

    if LGSonly:
        legend_items = ['2004', '2005', '2006', '2007', '2008', '2009','2010','2011']
        legend_colors = ['gold','lightsalmon','orange','darkorange', 'orangered',
                         'red','darkred', 'maroon']
    else:
        legend_items = ['1995', '1996', '1997', '1998', '1999',
                        '2000', '2001', '2002', '2003', '2004',
                        '2005', '2006', '2007', '2008', '2009',
                        '2010', '2011']
        legend_colors = ['olive', 'brown', 'purple', 'darkslateblue',
                         'mediumblue', 'steelblue', 'teal', 'green',
                         'greenyellow', 'gold', 'lightsalmon', 'orange',
                         'darkorange','orangered', 'red', 'darkred', 'maroon']

    # Assign color to each epoch
    year = [str(int(na.floor( d ))) for d in date]
    color_arr = []


    for i in range(len(year)):
        # find out which year
        try:
            idx = legend_items.index( year[i] )
            color_arr.append( legend_colors[idx] )
        except ValueError:
            color_arr.append('black')

        py.errorbar([x[i]], [y[i]], fmt='ko', xerr=xerr[i], yerr=yerr[i],
                    ms=5,
                    color=color_arr[i], mec=color_arr[i], mfc=color_arr[i])

 
    # Set axis ranges
    py.title(star)
    ax = py.axis([xmax+0.02, xmin-0.02, ymin-0.02, ymax+0.02])
    py.gca().set_aspect('equal')
    py.xlabel('X (arcsec)')
    py.ylabel('Y (arcsec)')

    if legend != False:
        # Draw legend
        import matplotlib.font_manager
        prop = matplotlib.font_manager.FontProperties(size=8)
        py.legend(legend_items, numpoints=1, loc=4, prop=prop)
        ltext = py.gca().get_legend().get_texts()
        lgdLines = py.gca().get_legend().get_lines()
        for j in range(len(ltext)):
            py.setp(ltext[j], color=legend_colors[j])
            lgdLines[j].set_marker('o')
            lgdLines[j].set_mec(legend_colors[j])
            lgdLines[j].set_mfc(legend_colors[j])
            lgdLines[j].set_ms(5)


    # Retrieve information on the acceleration fit from the .accelFormal file
    #fitFile = root + poly + '.accelFormal'
    #_fit = open(fitFile, 'r')

    #_vel = asciidata.open(root + align + '.vel')
    #name = _vel[0]._data
    #vcnt = _vel[1].tonumpy()
    #idx = name.index(star)
    
    #for line in _fit:
    #    entries = line.split()
#
#        if (entries[0] == star):
#            chiSqX = float( entries[7] )
#            chiSqY = float( entries[15] )
#            chiSqXr = chiSqX / (vcnt[idx] - 3.0)
#            chiSqYr = chiSqY / (vcnt[idx] - 3.0)
#            break
#
#    print 'Reduced Chi^2 (X): %6.3f' % chiSqXr
#    print 'Reduced Chi^2 (Y): %6.3f' % chiSqYr

#    _fit.close()

    py.savefig(root+'plots/plotStar_'+star+'_orbit'+suffix+'.png')    
    py.close(2)


def plot_multi_aligns(root='/u/syelda/research/gc/aligndir/'):
    """
    Plot positions of a star as found in many different alignments.
    Requires user input to specify the align directories to be plotted.
    The stars chosen for plotting are hardcoded.
    """
    numAlns = raw_input("How many aligns would you like to plot? ")
    numAlns = int(numAlns)

    # Central arcsecond stars (and the foreground source S0-32)
    stars1 = ['S0-32','S0-2','S0-4','32star_217']
    out1 = 'compare_central_stars_multi_aligns.png'
    # More distant stars from different parts of field
    stars2 = ['irs16C','S0-15','S1-4','S1-12']
    out2 = 'compare_outer_stars_multi_aligns.png'
    # Linear-moving stars from different parts of field
    stars3 = ['S2-80','irs29N','S2-6','S3-19']
    out3 = 'compare_linear_stars_multi_aligns.png'

    sets = [stars1, stars2, stars3]
    out = [out1, out2, out3]

    fmt = ['r.','b.','g.','m.','c.'] # allows up to 5 aligns
    clrs = ['red','blue','green','magenta','cyan']
    padd = 0.001

    fig = py.figure(figsize=(10,10))
    fig.subplots_adjust(left=0.1,right=0.95,top=0.95,hspace=0.2,wspace=0.25)
    fig.clf()
    import matplotlib.font_manager
    prop = matplotlib.font_manager.FontProperties(size=9)

    legend_items = []
    legend_colors = []
    for aa in range(numAlns):
        _dir = raw_input("Enter align directory # %i to be plotted (e.g., 10_11_05/): " % (aa+1))
        alnDir = root + _dir + '/'
        # Set up legend
        legend_items = np.concatenate([legend_items, [_dir]])
        legend_colors = np.concatenate([legend_colors, [clrs[aa]]])
        for ii in range(3): # each set of stars (central, outer, linear)
            stars = sets[ii]
            for ss in range(len(stars)):
                pointsFile = alnDir + 'points_c/' + stars[ss] + '.points'
                _tabPoints = asciidata.open(pointsFile)
                date = _tabPoints[0].tonumpy()
                x = _tabPoints[1].tonumpy() * -1.0
                y = _tabPoints[2].tonumpy()
                xerr = _tabPoints[3].tonumpy()
                yerr = _tabPoints[4].tonumpy()
    
                ax = fig.add_subplot(2,2,ss+1)
                ax.errorbar(x,y,xerr=xerr,yerr=yerr,fmt=fmt[aa])
                ax.set_xlabel('RA (arcsec)')
                ax.set_ylabel('Dec (arcsec)')
    
                if aa == numAlns-1:
                    # Find range of plot
                    halfRange = max([ abs(x.max() - x.min()), abs(y.max() - y.min()) ]) / 2.0
                    xmax = x.min() + ((x.max() - x.min())/2.0) + halfRange + padd
                    ymax = y.min() + ((y.max() - y.min())/2.0) + halfRange + padd
                    xmin = x.min() + ((x.max() - x.min())/2.0) - halfRange - 2*padd
                    ymin = y.min() + ((y.max() - y.min())/2.0) - halfRange - padd
                    ax.axis([xmax+0.03, xmin-0.03, ymin-0.03, ymax+0.03])
                    if stars[ss] == '32star_217':
                        stars[ss] = 'S0-1'
                    ax.set_title(stars[ss])
                
                    if ss == len(stars)-1:
                        ax.legend(legend_items, numpoints=1, loc=4, prop=prop)
                        ltext = ax.get_legend().get_texts()
                        lgdLines = ax.get_legend().get_lines()
                        for j in range(len(ltext)):
                            py.setp(ltext[j], color=legend_colors[j])
                            py.setp(lgdLines[j], visible=False)
    
            # Save off the plots after every 4 stars are plotted
            fig.savefig(root + out[ii])


def plot_s02_multi_aligns(root='/u/syelda/research/gc/aligndir/'):
    """
    This was made for the Keck 11B proposal, and a copy of this function
    is in /u/ghezgroup/code/python/papers/prop11B.py.
    Plot positions of S0-2 as found in many different alignments, with orbital 
    fits over plotted.
    Requires user input to specify the align directories to be plotted.
    The stars chosen for plotting are hardcoded.
    """
    numAlns = raw_input("How many aligns would you like to plot? ")
    numAlns = int(numAlns)

    # Central arcsecond stars (and the foreground source S0-32)
    star = 'S0-2'
    out = 'compare_S0-2_multi_aligns.png'

    fmt = ['r.','b.','g.','m.','c.'] # allows up to 5 aligns
    clrs = ['red','blue','green','magenta','cyan']
    padd = 0.001

    fig = py.figure(figsize=(6,6))
    fig.subplots_adjust(left=0.15,right=0.95,top=0.9,hspace=0.2,wspace=0.25)
    fig.clf()
    import matplotlib.font_manager
    prop = matplotlib.font_manager.FontProperties(size=9)

    legend_items = []
    legend_colors = []
    for aa in range(numAlns):
        _dir = raw_input("Enter align directory # %i to be plotted (e.g., 10_11_05/): " % (aa+1))
        alnDir = root + _dir + '/'
        # Set up legend
        legend_items = np.concatenate([legend_items, [_dir]])
        legend_colors = np.concatenate([legend_colors, [clrs[aa]]])

         # Read in points file
        pointsFile = alnDir + 'points_1000/' + star + '.points'
        _tabPoints = asciidata.open(pointsFile)
        date = _tabPoints[0].tonumpy()
        x = _tabPoints[1].tonumpy() * -1.0
        y = _tabPoints[2].tonumpy()
        xerr = _tabPoints[3].tonumpy()
        yerr = _tabPoints[4].tonumpy()

         # Read in orbit file
        orbitFile = alnDir + 'efit/orbit.S0-2.model'
        _tabOrbit = asciidata.open(orbitFile)
        dateO = _tabOrbit[0].tonumpy()
        xO = _tabOrbit[1].tonumpy() * -1.0
        yO = _tabOrbit[2].tonumpy()

        ax = fig.add_subplot(1,1,1)
        ax.errorbar(x,y,xerr=xerr,yerr=yerr,fmt=fmt[aa])
        ax.plot(xO,yO,color=clrs[aa])
        ax.set_xlabel('RA (arcsec)')
        ax.set_ylabel('Dec (arcsec)')

        if aa == numAlns-1:
            # Find range of plot
            halfRange = max([ abs(x.max() - x.min()), abs(y.max() - y.min()) ]) / 2.0
            xmax = x.min() + ((x.max() - x.min())/2.0) + halfRange + padd
            ymax = y.min() + ((y.max() - y.min())/2.0) + halfRange + padd
            xmin = x.min() + ((x.max() - x.min())/2.0) - halfRange - 2*padd
            ymin = y.min() + ((y.max() - y.min())/2.0) - halfRange - padd
            ax.axis([xmax+0.03, xmin-0.03, ymin-0.03, ymax+0.03])
            ax.set_title(star)
        
            ax.legend(legend_items, numpoints=1, loc=4, prop=prop)
            ltext = ax.get_legend().get_texts()
            lgdLines = ax.get_legend().get_lines()
            for j in range(len(ltext)):
                py.setp(ltext[j], color=legend_colors[j])
                py.setp(lgdLines[j], visible=False)

    fig.savefig(root + out)

def plot_star_orbit(root='/u/syelda/research/gc/aligndir/'):
    """
    Plot positions of a star, with orbital 
    fits over plotted.
    Requires user input to specify the align directories to be plotted.
    The star chosen for plotting are hardcoded.
    """
    
    numAlns = 1

    # Central arcsecond stars (and the foreground source S0-32)
    star = 'S0-38'
    out = 'orbit_S0-38_fixedfocus.png'

    fmt = ['r.'] 
    clrs = ['red']
    padd = 0.001

    fig = py.figure(figsize=(6,6))
    fig.subplots_adjust(left=0.15,right=0.95,top=0.9,hspace=0.2,wspace=0.25)
    fig.clf()
    import matplotlib.font_manager
    prop = matplotlib.font_manager.FontProperties(size=9)

    legend_items = []
    legend_colors = []
    for aa in range(numAlns):
        _dir = raw_input("Enter align directory # %i to be plotted (e.g., 10_11_05/): " % (aa+1))
        alnDir = root + _dir + '/'
        # Set up legend
        legend_items = np.concatenate([legend_items, [_dir]])
        legend_colors = np.concatenate([legend_colors, [clrs[aa]]])

         # Read in points file
        pointsFile = alnDir + 'efit/' + star + '.points'
        _tabPoints = asciidata.open(pointsFile)
        date = _tabPoints[0].tonumpy()
        x = _tabPoints[1].tonumpy() * -1.0
        y = _tabPoints[2].tonumpy()
        xerr = _tabPoints[3].tonumpy()
        yerr = _tabPoints[4].tonumpy()

         # Read in orbit file
        orbitFile = alnDir + 'efit/orbit.S0-38.model'
        _tabOrbit = asciidata.open(orbitFile)
        dateO = _tabOrbit[0].tonumpy()
        xO = _tabOrbit[1].tonumpy() * -1.0
        yO = _tabOrbit[2].tonumpy()

        ax = fig.add_subplot(1,1,1)
        ax.errorbar(x,y,xerr=xerr,yerr=yerr,fmt=fmt[aa])
        ax.plot(xO,yO,color=clrs[aa])
        ax.set_xlabel('RA (arcsec)')
        ax.set_ylabel('Dec (arcsec)')

        if aa == numAlns-1:
            # Find range of plot
            halfRange = max([ abs(x.max() - x.min()), abs(y.max() - y.min()) ]) / 2.0
            xmax = x.min() + ((x.max() - x.min())/2.0) + halfRange + padd
            ymax = y.min() + ((y.max() - y.min())/2.0) + halfRange + padd
            xmin = x.min() + ((x.max() - x.min())/2.0) - halfRange - 2*padd
            ymin = y.min() + ((y.max() - y.min())/2.0) - halfRange - padd
            ax.axis([xmax+0.03, xmin-0.03, ymin-0.03, ymax+0.03])
            ax.set_title(star)
        
            ax.legend(legend_items, numpoints=1, loc=4, prop=prop)
            ltext = ax.get_legend().get_texts()
            lgdLines = ax.get_legend().get_lines()
            for j in range(len(ltext)):
                py.setp(ltext[j], color=legend_colors[j])
                py.setp(lgdLines[j], visible=False)

    fig.savefig(root + out)


