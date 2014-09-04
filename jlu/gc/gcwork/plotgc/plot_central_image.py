import asciidata
import numpy as np
import pylab as py
import pyfits
from gcwork.plotgc import gccolors 
from gcwork import objects
from gcwork import starset
import young
from pylab import *
from matplotlib.colors import colorConverter
import colorsys
import pdb

def go(modelDir, outdir, nodata=True, nolabel=False, suffix=''):
    """
    Plots central arcsecond image with orbits overplotted
    (Originally written in IDL; converted to Python by S. Yelda)
    """

    outroot = 'plot_central_image' + suffix
    #outroot = 'plot_central_image'
    if nolabel == True:
        outroot += '_nolabel'
    

    # Stars to be plotted
    stars = ['S0-1','S0-2','S0-4','S0-5','S0-16','S0-19','S0-20']
    clr = ['magenta', 'red', 'yellow', 'blue', 'steelblue', 'green', 'orange']
    # If the star was discovered during speckle, use solid
    # If discovered during AO, use dashed
    linestyle = ['-', '-', '-', '-', '-', '-', '-']
    discover = [1995.5, 1995.5, 1995.5, 1995.5, 1995.5, 1995.5, 1995.5]

    rng = 0.6
    xcen = 0.0
    ycen = 0.0
    delta = rng

    # Set up the background image
    angle = 0.0
    scale = 0.00995
    sgra = [628.4, 734.0]
    imgFile = '/u/ghezgroup/data/gc/11maylgs/combo/mag11maylgs_kp.fits'
    img = pyfits.getdata(imgFile)
    imgsize = (img.shape)[0]
    # Make axes for images in arcsec
    pixL = np.arange(0,imgsize)
    xL = [-1*(xpos - sgra[0])*scale for xpos in pixL]
    yL = [(ypos - sgra[1])*scale for ypos in pixL]

    # Plot
    py.figure(1)
    py.clf()
    py.figure(figsize=(6,6))
    #py.figure(figsize=(7,5.5))
    py.subplots_adjust(left=0.02,right=0.98,top=0.98,bottom=0.02)
    py.imshow(np.log10(img), aspect='equal', interpolation='bicubic',
              extent=[max(xL), min(xL), min(yL), max(yL)],vmin=2.4,vmax=4.0,
              origin='lowerleft')

    # Loop through the stars and get the model fits
    for ss in range(len(stars)):
        #mod = asciidata.open(modelDir + 'orbit.' + stars[ss] + '_fixed.model')
        # Temporary -- use our latest orbit for S0-2
        if stars[ss] == 'S0-2':
            mod = asciidata.open(modelDir + 'orbit.' + stars[ss] + '.fixed.model')
        elif stars[ss] == 'S0-4':
            mod = asciidata.open('/u/syelda/research/gc/aligndir/11_08_29/efit/orbit.' + stars[ss] + '.model')
        else:
            mod = asciidata.open(modelDir + 'orbit.' + stars[ss] + '.model')
        mt = mod[0].tonumpy()
        mx = mod[1].tonumpy() * -1.0
        my = mod[2].tonumpy()

        idx = np.where((mt >= discover[ss]) & (mt < 2012.0))[0]
        emt = mt[idx]
        emx = mx[idx]
        emy = my[idx]

        # Plot the orbit model
        py.plot(mx, my, color=clr[ss], linestyle=linestyle[ss], lw=1.5)

        # Plot the data points over the model
        epochs = np.arange(discover[ss], discover[ss] + 17, 1)
        numE = len(epochs)
        starname = stars[ss]
        # Repair some star names
        #if stars[ss] == '34star_205':
        #    starname = 'S0-1'
        #if stars[ss] == '37star_551':
        #    starname = 'S0-20'

        # Get the colors for each star
        rgb = asarray(colorConverter.to_rgb(clr[ss]))
        hue = colorsys.rgb_to_hsv(rgb[0],rgb[1],rgb[2])

        for ee in range(numE):
            if nodata == True:
                # Plot the model
                tdiff = np.abs(emt - epochs[ee])
                t = np.where(tdiff == tdiff.min())[0]
                x = emx[t]
                y = emy[t]

                # Circles for data points
                # Make the colors fade
                rgb = colorsys.hsv_to_rgb(hue[0],hue[1]/((numE-ee)/8.0+1.0),hue[2])
                py.plot(x, y, color=rgb, marker='o', ms=7)
                # Black edge around data points
                py.plot(x, y, mfc='none', mec='k', ms=7)

        lx = -0.45
        ly = 0.1
        if nolabel == False:
            py.text(lx,ly-(0.05*ss),starname,weight='bold',fontsize=14, color='white')
            py.plot(lx+0.03, ly+0.01-(0.05*ss), color=clr[ss], marker='o', ms=9)
            py.plot(lx+0.03, ly+0.01-(0.05*ss), mfc='none', mec='k', ms=9)

    if nolabel == False:
        # Add some text to plot
        py.text(rng-0.28, -rng+0.11, 'Keck/UCLA', weight='bold', color='white', fontsize=17)
        py.text(rng-0.12, -rng+0.06, 'Galactic Center Group', weight='bold', color='white', fontsize=17)
        py.text(rng-0.87, -rng+0.07, '1995-2011', weight='bold', color='white', fontsize=17)

        # Add scale
        py.plot([0.45, 0.35], [0.55, 0.55], 'w-', lw=3)
        py.text(0.45,0.5,'0.1"',fontsize=16, weight='bold', color='white')

        # Add compass
        qvr = py.quiver([-0.48], [0.426], [0], [0.1], color='white', units='width', scale=1)
        py.text(-0.43,0.53,'N',fontsize=18, weight='bold', color='white')
        qvr = py.quiver([-0.4802], [0.43], [-0.1], [0], color='white', units='width', scale=1)
        py.text(-0.35,0.45,'E',fontsize=18, weight='bold', color='white')

    py.axis([rng,-rng,-rng,rng])
    thePlot = py.gca()
    py.setp(thePlot.set_xticks([]))
    py.setp(thePlot.set_yticks([]))
    py.setp(thePlot.get_xticklabels(),visible=False)
    py.setp(thePlot.get_yticklabels(),visible=False)
    
    py.savefig(outdir + outroot + '_hires.png', dpi=400)
    #py.savefig(outdir + outroot + '_hires.png', dpi=300)
    py.savefig(outdir + outroot + '.png')
