from jlu.gc.orbits import isotropic as iso
from gcwork import young
from gcwork import analyticOrbits as aorb
import os, shutil
import numpy as np
import pylab as py

root = '/u/jlu/work/gc/proper_motion/align/08_03_26/'

def obsVsIsotropic():
    # Analyze Observations
    calcRadialDiskDensity()
    calcRadialDiskDensity3()

    dir = 'iso_test/'
    r = [0.8, 13.5]
    iso.analyzeData('./', dir, '1', radii=r, Nstars=73, Nsims=200, Npdf=1000)
    iso.analyzeData('./', dir, '2', radii=r, Nstars=73, Nsims=200, Npdf=1000)
    iso.analyzeData('./', dir, '3', radii=r, Nstars=73, Nsims=200, Npdf=1000)
    iso.analyzeData('./', dir, '4', radii=r, Nstars=73, Nsims=200, Npdf=1000)
    iso.analyzeData('./', dir, '5', radii=r, Nstars=73, Nsims=200, Npdf=1000)

    ##########
    # 3 radial bins
    ##########
    r = [0.8, 3.5]
    iso.analyzeData('./', dir, '1', radii=r, Nstars=35, Nsims=500, Npdf=1000)
    iso.analyzeData('./', dir, '2', radii=r, Nstars=35, Nsims=500, Npdf=1000)

    r = [3.5, 7.0]
    iso.analyzeData('./', dir, '1', radii=r, Nstars=23, Nsims=500, Npdf=1000)
    iso.analyzeData('./', dir, '2', radii=r, Nstars=23, Nsims=500, Npdf=1000)

    r = [7.0, 13.5]
    iso.analyzeData('./', dir, '1', radii=r, Nstars=15, Nsims=500, Npdf=1000)
    iso.analyzeData('./', dir, '2', radii=r, Nstars=15, Nsims=500, Npdf=1000)

def calcRadialDiskDensity():
    yng = young.loadAllYoungStars('./')
    r2d = yng.getArray('r2d')
    names = yng.getArray('name')
    
    sidx = r2d.argsort()
    r2d = r2d[sidx]
    names = [names[ss] for ss in sidx]
    
    halfway = len(r2d) / 2
    names_inner = names[0:halfway]
    names_outer = names[halfway:]

    # copy over the *.mc.dat files
    for name in names_inner:
        if not os.path.exists('aorb_efit_all_inner/%s.mc.dat' % name):
            shutil.copy('aorb_efit_all/%s.mc.dat' % name, 
                        'aorb_efit_all_inner/%s.mc.dat' % name)

    for name in names_outer:
        if not os.path.exists('aorb_efit_all_outer/%s.mc.dat' % name):
            shutil.copy('aorb_efit_all/%s.mc.dat' % name, 
                        'aorb_efit_all_outer/%s.mc.dat' % name)



    #diski = aorb.Disk('./', 'aorb_efit_all_inner/')
    #diski.names = names_inner
    #diski.run(makeplot=True)

    disko = aorb.Disk('./', 'aorb_efit_all_outer/')
    disko.names = names_outer
    disko.run(makeplot=True)

def calcRadialDiskDensity3():
    yng = young.loadAllYoungStars('./')
    r2d = yng.getArray('r2d')
    names = yng.getArray('name')
    
    sidx = r2d.argsort()
    r2d = r2d[sidx]
    names = [names[ss] for ss in sidx]
    
    cut1 = 3.5
    cut2 = 7.0

    names1 = []
    names2 = []
    names3 = []

    for rr in range(len(r2d)):
        if (r2d[rr] <= cut1):
            names1.append(names[rr])
        if ((r2d[rr] > cut1) & (r2d[rr] <= cut2)):
            names2.append(names[rr])
        if (r2d[rr] > cut2):
            names3.append(names[rr])

    # copy over the *.mc.dat files
    for name in names1:
        if not os.path.exists('aorb_efit_all_r1/%s.mc.dat' % name):
            shutil.copy('aorb_efit_all/%s.mc.dat' % name, 
                        'aorb_efit_all_r1/%s.mc.dat' % name)

    for name in names2:
        if not os.path.exists('aorb_efit_all_r2/%s.mc.dat' % name):
            shutil.copy('aorb_efit_all/%s.mc.dat' % name, 
                        'aorb_efit_all_r2/%s.mc.dat' % name)

    for name in names3:
        if not os.path.exists('aorb_efit_all_r3/%s.mc.dat' % name):
            shutil.copy('aorb_efit_all/%s.mc.dat' % name, 
                        'aorb_efit_all_r3/%s.mc.dat' % name)



    disk1 = aorb.Disk('./', 'aorb_efit_all_r1/')
    disk1.names = names1
    disk1.run(makeplot=True)

    disk2 = aorb.Disk('./', 'aorb_efit_all_r2/')
    disk2.names = names2
    disk2.run(makeplot=True)

    disk3 = aorb.Disk('./', 'aorb_efit_all_r3/')
    disk3.names = names3
    disk3.run(makeplot=True)

    
