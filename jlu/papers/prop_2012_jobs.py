import pylab as py
import numpy as np
import atpy
import math
from jlu.wd1 import synthetic as syn

dataTable = "/Volumes/data1/dhuang/codeOld/4784P.fits"
AKs = 0.75
distance = 4000
logAge = 6.80
age = 10**logAge

scale = 61.0

outdir = '/u/jlu/work/wd1/plot_results/'

def make_plots():
    ##########
    # Data
    ##########
    d = atpy.Table(dataTable)

    dt = 2010.6521 - 2005.4846
    pmx = d.dx / dt
    pmy = d.dy / dt

    Pkin_cut = 0.90

    idx = np.where(d.Pkin >= Pkin_cut)[0]
    dx_mean = d.dx[idx].mean()
    dy_mean = d.dy[idx].mean()
    dr_cluster = d.dr[idx].max() / dt
    print dr_cluster

    ##########
    # Load up model isochrone
    ##########
    iso = load_isochrone()

    # Pull out a few key masses along the isochrone
    idxM1 = np.abs(iso.M - 1.0).argmin()
    idxM01 = np.abs(iso.M - 0.1).argmin()

    colM1 = iso.mag814w[idxM1] - iso.mag125w[idxM1]
    magM1 = iso.mag814w[idxM1]
    colM01 = iso.mag814w[idxM01] - iso.mag125w[idxM01]
    magM01 = iso.mag814w[idxM01]


    # Plot VPD
    py.figure(1)
    py.clf()
    py.plot(pmx, pmy, 'k.', ms=2)
    py.xlim(-3, 3)
    py.ylim(-3, 3)
    py.xlabel('X Proper Motion (mas/yr)')
    py.ylabel('Y Proper Motion (mas/yr)')
    py.title('Wd 1')
    py.savefig(outdir + 'wd1_vpd.png')

    # circ = py.Circle([dx_mean/dt, dy_mean/dt], radius=dr_cluster/dt,
    #                  fc='none', ec='yellow', linewidth=4, zorder=10)
    # py.gca().add_patch(circ)

    
    # py.scatter(pmx, pmy, c=d.Pkin, s=5, edgecolor='none')
    # py.colorbar()

    d2 = d.where(d.Pkin > Pkin_cut)

    # Plot CMD
    py.figure(2)
    py.clf()
    py.plot(d.mag814-d.mag125, d.mag814, 'k.', ms=2)
    py.ylim(25, 13)
    py.xlim(0, 7)
    py.xlabel('F814W - F125W (mag)')
    py.ylabel('F814W (mag)')
    py.savefig(outdir + 'wd1_cmd_all.png')

    py.figure(3)
    py.clf()
    py.plot(d2.mag814-d2.mag125, d2.mag814, 'k.', ms=2)
    py.ylim(25, 13)
    py.xlim(0, 7)
    py.xlabel('F814W - F125W (mag)')
    py.ylabel('F814W (mag)')
    py.savefig(outdir + 'wd1_cmd_members.png')

    py.clf()
    py.plot(d2.mag814-d2.mag125, d2.mag814, 'k.', ms=2)
    py.ylim(25, 13)
    py.xlim(0, 7)
    py.xlabel('F814W - F125W (mag)')
    py.ylabel('F814W (mag)')
    py.plot(iso.mag814w - iso.mag125w, iso.mag814w, 'r-', ms=2,
            color='red', linewidth=2)
    py.plot([colM1], [magM1], 'r*', ms=25)
    py.text(colM1+0.1, magM1, r'1 M$_\odot$', color='red', 
            fontweight='normal', fontsize=28)
    py.savefig(outdir + 'wd1_cmd_members_model.png')

    py.clf()
    py.plot(d.mag814 - d.mag125, d.mag814, 'k.', ms=2)
    py.plot(d2.mag814 - d2.mag125, d2.mag814, 'y.', ms=3)
    py.ylim(25, 13)
    py.xlim(0, 7)
    py.xlabel('F814W - F125W (mag)')
    py.ylabel('F814W (mag)')
    py.plot(iso.mag814w - iso.mag125w, iso.mag814w, 'r-', ms=2,
            color='red', linewidth=2)
    py.plot([colM1], [magM1], 'r*', ms=25)
    py.text(colM1+0.1, magM1, r'1 M$_\odot$', color='red', 
            fontweight='normal', fontsize=28)
    py.savefig(outdir + 'wd1_cmd_all_model.png')


    # Load up data for NIR only
    wd1_data = '/u/jlu/data/Wd1/hst/from_jay/EXPORT_WEST1.2012.02.04/wd1_catalog.fits'
    data = atpy.Table(wd1_data)

    colM1 = iso.mag125w[idxM1] - iso.mag160w[idxM1]
    magM1 = iso.mag125w[idxM1]
    colM01 = iso.mag125w[idxM01] - iso.mag160w[idxM01]
    magM01 = iso.mag125w[idxM01]

    py.clf()
    py.plot(data.mag125 - data.mag160, data.mag125, 'k.', ms=2)
    py.plot(d2.mag125 - d2.mag160, d2.mag125, 'y.', ms=3)
    py.plot(iso.mag125w - iso.mag160w, iso.mag125w, 'r-', ms=2,
            color='red', linewidth=2)
    py.plot([colM1], [magM1], 'r*', ms=25)
    py.plot([colM01], [magM01],'r*', ms=25)
    py.text(colM1+0.1, magM1, r'1 M$_\odot$', color='red', 
            fontweight='normal', fontsize=28)
    py.text(colM01+0.1, magM01, r'0.1 M$_\odot$', color='red',
            fontweight='normal', fontsize=28)
    py.xlim(0, 2)
    py.ylim(23, 12)
    py.xlabel(r'm$_{1.25 \mu m} - $m$_{1.60 \mu m}$ (mag)')
    py.ylabel(r'm$_{1.25 \mu m}$ (mag)')
    py.savefig(outdir + 'wd1_cmd_ir.png')



def load_isochrone(logAge=logAge, AKs=AKs, distance=distance):
    # Load up model isochrone
    iso = syn.load_isochrone(logAge=logAge, AKs=AKs, distance=4000)

    deltaDM = 5.0 * math.log10(distance / 4000.0)

    iso.mag814w += deltaDM
    iso.mag125w += deltaDM
    iso.mag139m += deltaDM
    iso.mag160w += deltaDM

    return iso
