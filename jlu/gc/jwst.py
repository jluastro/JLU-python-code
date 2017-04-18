import numpy as np
import pylab as py
from astropy.table import Table

def plot_sim_cluster():
    yng_file = '/Users/jlu/work/gc/jwst/iso_6.60_2.70_08000_JWST.fits'
    old_file = '/Users/jlu/work/gc/jwst/iso_9.70_2.70_08000_JWST.fits'
    out_dir = '/Users/jlu/work/gc/jwst/'

    yng = Table.read(yng_file)
    old = Table.read(old_file)

    saturate = {'F164N': 12.0,
                'F212N': 11.3,
                'F323N': 11.1,
                'F466N': 10.6,
                'F125W': 15.5,
                'F090W': 15.3}

    ## Mass-Luminosity Relationship at ~K-band
    py.figure(1)
    py.clf()
    py.semilogx(yng['mass'], yng['mag090w'], 'c.', label='F090W')
    py.semilogx(yng['mass'], yng['mag125w'], 'b.', label='F125W')
    py.semilogx(yng['mass'], yng['mag164n'], 'y.', label='F164N')
    py.semilogx(yng['mass'], yng['mag212n'], 'g.', label='F212N')
    py.semilogx(yng['mass'], yng['mag323n'], 'r.', label='F323N')
    py.semilogx(yng['mass'], yng['mag466n'], 'm.', label='F466N')
    py.axhline(saturate['F090W'], color='c', linestyle='--')
    py.axhline(saturate['F125W'], color='b', linestyle='--')
    py.axhline(saturate['F164N'], color='y', linestyle='--')
    py.axhline(saturate['F212N'], color='g', linestyle='--')
    py.axhline(saturate['F323N'], color='r', linestyle='--')
    py.axhline(saturate['F466N'], color='m', linestyle='--')
    py.gca().invert_yaxis()
    py.ylim(30, 10)
    py.xlabel('Mass (Msun)')
    py.ylabel('JWST Magnitudes')
    py.legend(loc='upper left', numpoints=1)
    py.savefig(out_dir + 'mass_luminosity.png')

    ## CMD
    py.figure(2)
    py.clf()
    py.plot(yng['mag125w'] - yng['mag323n'], yng['mag323n'], 'k.')
    py.plot(old['mag125w'] - old['mag323n'], old['mag323n'], 'r.')
    py.ylim(25, 8)
    py.xlabel('F125W - F323N (mag)')
    py.ylabel('F323N (mag)')
    py.savefig(out_dir + 'gc_cmd_F125W_F323N.png')

    py.figure(3)
    py.clf()
    py.plot(yng['mag125w'] - yng['mag212n'], yng['mag212n'], 'k.')
    py.plot(old['mag125w'] - old['mag212n'], old['mag212n'], 'r.')
    py.ylim(25, 8)
    py.xlabel('F125W - F212N (mag)')
    py.ylabel('F212N (mag)')
    py.savefig(out_dir + 'gc_cmd_F125W_F212N.png')

    py.figure(4)
    py.clf()
    py.plot(yng['mag212n'] - yng['mag466n'], yng['mag212n'], 'k.')
    py.plot(old['mag212n'] - old['mag466n'], old['mag212n'], 'r.')
    py.ylim(25, 8)
    py.xlabel('F212N - F466N (mag)')
    py.ylabel('F212N (mag)')
    py.savefig(out_dir + 'gc_cmd_F212N_F466N.png')

    py.figure(5)
    py.clf()
    py.plot(yng['mag125w'] - yng['mag212n'], yng['mag212n'] - yng['mag323n'], 'k.')
    py.plot(old['mag125w'] - old['mag212n'], old['mag212n'] - old['mag323n'], 'r.')
    py.xlabel('F125W - F212N (mag)')
    py.ylabel('F212N - F323N (mag)')
    py.savefig(out_dir + 'gc_ccd_F125W_F212N_F323N.png')
    
    return yng
