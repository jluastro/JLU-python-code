import math
import pylab as py
import numpy as np
from pyraf import iraf as ir
import atpy
from mass_dimm import get_mass_dimm

workdir = '/u/jlu/work/m31/nucleus/ifu_11_11_30/'

def integrated_spectrum():
    pass

def get_ao_performance_data():
    # Retrieve information from image headers
    outfile = workdir + 'data/ao_perform.txt'

    ir.images()
    ir.imutil()
    ir.hselect('*_img.fits', 'DATAFILE,DATE-OBS,UTC,AOLBFWHM,LGRMSWF', 'yes', Stdout=outfile)

    # Retrieve MASS/DIMM data files
    # get_mass_dimm.go('20081021')   # this one doesn't exist
    get_mass_dimm.go('20100815')
    get_mass_dimm.go('20100828')
    get_mass_dimm.go('20100829')

def plot_ao_performance():
    outfile = workdir + 'data/ao_perform.txt'

    tab = atpy.Table(outfile, type='ascii')

    print calc_strehl(150)
    tab.rename_column('col2', 'date')
    tab.rename_column('col3', 'time')
    tab.rename_column('col5', 'wfe')
    strehl = calc_strehl(tab.wfe, inst_wfe=300.0)
    tab.add_column('strehl', strehl)
    
    idx1 = np.where(tab.date == '2008-10-21')[0]
    idx2 = np.where(tab.date == '2010-08-15')[0]
    idx3 = np.where(tab.date == '2010-08-28')[0]
    idx4 = np.where(tab.date == '2010-08-29')[0]
    
    index = np.arange(len(tab))
    break1 = index[idx1[-1]] + 0.5
    break2 = index[idx2[-1]] + 0.5
    break3 = index[idx3[-1]] + 0.5

    #dimm1 = get_mass_dimm.load_dimm('20081021')
    dimm2 = get_mass_dimm.load_dimm('20100815')
    dimm3 = get_mass_dimm.load_dimm('20100828')
    dimm4 = get_mass_dimm.load_dimm('20100829')

    #mass1 = get_mass_dimm.load_mass('20081021')
    mass2 = get_mass_dimm.load_mass('20100815')
    mass3 = get_mass_dimm.load_mass('20100828')
    mass4 = get_mass_dimm.load_mass('20100829')

    def plot_var(xVar, xLabel, outroot=None):
        py.close(2)
        py.figure(2, figsize=(10,6))
        py.clf()
        py.subplots_adjust(left=0.12)
        py.plot(index[idx1], xVar[idx1], 'r.', label='2008-10-21', ms=10)
        py.plot(index[idx2], xVar[idx2], 'b.', label='2010-08-15', ms=10)
        py.plot(index[idx3], xVar[idx3], 'g.', label='2010-08-28', ms=10)
        py.plot(index[idx4], xVar[idx4], 'c.', label='2010-08-29', ms=10)

        py.axvline(break1, c='k', linestyle='-')
        py.axvline(break2, c='k', linestyle='-')
        py.axvline(break3, c='k', linestyle='-')
        
        ax = py.gca()
        ax.set_xticklabels([])
        for t in ax.xaxis.get_ticklines():
            t.set_visible(False)

        rng = py.axis()
        ypos = 1.02 * rng[3]
        py.text(idx1.mean(), ypos, '2008-10-21', horizontalalignment='center')
        py.text(idx2.mean(), ypos, '2010-08-15', horizontalalignment='center')
        py.text(idx3.mean(), ypos, '2010-08-28', horizontalalignment='center')
        py.text(idx4.mean(), ypos, '2010-08-29', horizontalalignment='center')

        py.xlim(-1, len(tab))
        py.ylabel(xLabel)

        if outroot != None:
            py.savefig(workdir + 'data/' + outroot + '.png')

    plot_var(tab.strehl, 'Strehl', 'strehl_vs_time')
    plot_var(tab.wfe, 'RMS WFE (nm)', 'wfe_vs_time')

    ##################################################
    # Make the nightly plots for 2010 data
    ##################################################
    obsHourUT = np.zeros(len(tab), dtype=float)
    for ii in range(len(tab)):
        tmp = tab.time[ii].split(':')
        obsHourUT[ii] = float(tmp[0]) + (float(tmp[1])/60.0) + (float(tmp[2])/3600.0)

    def calc_dimmHourUT(dimmTab):
        tmp = dimmTab.hourUT
        tmp += dimmTab.min / 60.0
        tmp += dimmTab.sec / 3600.0

        return tmp

    # 2010-08-15
    dimmHourUT = calc_dimmHourUT(dimm2)
    massHourUT = calc_dimmHourUT(mass2)
    py.close(1)
    py.figure(1, figsize=(8, 8))
    py.clf()

    py.subplot(2, 1, 1)
    py.plot(obsHourUT[idx2], tab.wfe[idx2], 'k.')
    rng = py.axis()
    py.ylabel('RMS WFE (nm)')

    py.subplot(2, 1, 2)
    py.plot(dimmHourUT, dimm2.seeing, 'r.', label='DIMM')
    py.plot(massHourUT, mass2.seeing, 'b.', label='MASS')
    py.xlim(rng[0], rng[1])
    py.ylabel('Seeing (arcsec)')
    py.legend(loc='upper right', numpoints=1)
    py.xlabel('Time in Hours UT')

    

def calc_strehl(wfe, inst_wfe=30.0, inst_wave=2120.0):
    """
    Use the Marachel approximation to estimate the Strehl ratio
    from a RMS WFE given in nm. The default instrumental static WFE
    and wavelength can also be adjusted. The defaults are set for
    K-band in OSIRIS.
    """
    wfe_frac = np.hypot(inst_wfe, wfe) / inst_wave
    strehl = math.e**(-(2.0 * math.pi * wfe_frac)**2)
    
    return strehl


    

