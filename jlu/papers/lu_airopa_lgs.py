import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import subprocess
import datetime
import os
from astropy.io import fits
from astropy.table import Table
from jlu.util import fileUtil
import pdb

paper_dir = '/Users/jlu/doc/papers/ao_opt_lgs_model/'

def plot_ngs_perfect():
    """
    Plot the perfect Keck NGS PSF and OTF.
    """
    # Make an IDL batch file to produce the perfect Keck PSF and OTF.
    work_dir = '/Users/jlu/work/ao/ao_opt/papers/lgs_theory/'
    plot_dir = work_dir + 'plot_ngs_perfect/'
    fileUtil.mkdir(plot_dir)

    _idl = open(plot_dir + 'plot_ngs_perfect.idl', 'w')
    _idl.write('print, "test"\n')
    _idl.write('.r lu_airopa_lgs.pro\n')
    _idl.write('plot_ngs_perfect\n')
    _idl.write('exit')
    _idl.close()

    cmd = 'airopa_idl < ' + plot_dir + 'plot_ngs_perfect.idl >& '+ plot_dir + 'plot_ngs_perfect.log'

    print('### Runing IDL:')
    print(cmd)
    # subprocess.call(cmd, shell=True)
    pdb.set_trace()

    psf_on = fits.getdata(plot_dir + 'ngs_perfect_psf.fits')
    otf_on = fits.getdata(plot_dir + 'ngs_perfect_otf.fits')

    psf_off = fits.getdata(plot_dir + 'ngs_psf_0867_0867.fits')
    otf_off = fits.getdata(plot_dir + 'ngs_otf_0867_0867.fits')
    
    # Colormap Normalization
    norm = colors.PowerNorm(gamma=1./4.)
    extent_rng = (psf_on.shape[0] - (psf_on.shape[0]/2.0)) * 0.00996
    extent = [extent_rng, -extent_rng, -extent_rng, extent_rng]
    
    plt.close(1)
    plt.figure(1, figsize=(10, 10))
    plt.clf()
    plt.subplots_adjust(left=0.15, bottom=0.0, right=1.0, hspace=0, wspace=0)
    
    plt.subplot(221)
    plt.imshow(psf_on, cmap="gray", norm=norm, extent=extent)
    plt.setp(plt.gca().get_yticklabels(), visible=False)
    plt.gca().get_xaxis().set_visible(False)
    plt.title('PSF')
    plt.ylabel('On Axis')

    plt.subplot(222)
    plt.imshow(otf_on, cmap="plasma", extent=extent)
    plt.gca().get_xaxis().set_visible(False)
    plt.gca().get_yaxis().set_visible(False)
    plt.title('OTF')

    plt.subplot(223)
    plt.imshow(psf_off, cmap="gray", norm=norm, extent=extent)
    plt.setp(plt.gca().get_yticklabels(), visible=False)
    plt.gca().get_xaxis().set_visible(False)
    plt.ylabel('Off Axis')

    plt.subplot(224)
    plt.imshow(otf_off, cmap="plasma", extent=extent)
    plt.gca().get_xaxis().set_visible(False)
    plt.gca().get_yaxis().set_visible(False)
    
    plt.tight_layout()

    fig_dir = paper_dir + 'figures/ngs_perfect/'
    fileUtil.mkdir(fig_dir)
    
    fig_out = fig_dir + 'ngs_perfect.png'
    plt.savefig(fig_out)
    
    return
    
    
def plot_lgs_atm_inst():
    """
    Plot PSFs and OTFs assuming stationary AO and instrumental structure function; 
    but variable anisoplanatism.
    """
    # Make an IDL batch file to produce the perfect Keck PSF and OTF.
    work_dir = '/Users/jlu/work/ao/ao_opt/papers/lgs_theory/'
    plot_dir = work_dir + 'plot_lgs_atm_inst/'
    fileUtil.mkdir(plot_dir)

    _idl = open(plot_dir + 'plot_lgs_atm_inst.idl', 'w')
    _idl.write('print, "test"\n')
    _idl.write('.r lu_airopa_lgs.pro\n')
    _idl.write('plot_lgs_atm_inst\n')
    _idl.write('exit')
    _idl.close()

    cmd = 'airopa_idl < ' + plot_dir + 'plot_lgs_atm_inst.idl >& '+ plot_dir + 'plot_lgs_atm_inst.log'

    print('### Runing IDL:')
    print(cmd)
    # subprocess.call(cmd, shell=True)
    pdb.set_trace()

    psf_on = fits.getdata(plot_dir + 'lgs_psf_0512_0512.fits')
    otf_on = fits.getdata(plot_dir + 'lgs_otf_0512_0512.fits')
    psf_off_atm = fits.getdata(plot_dir + 'lgs_psf_atm_0867_0867.fits')
    otf_off_atm = fits.getdata(plot_dir + 'lgs_otf_atm_0867_0867.fits')
    psf_off_both = fits.getdata(plot_dir + 'lgs_psf_both_0867_0867.fits')
    otf_off_both = fits.getdata(plot_dir + 'lgs_otf_both_0867_0867.fits')
    
    # Colormap Normalization
    norm = colors.PowerNorm(gamma=1./4.)
    extent_rng = (psf_on.shape[0] - (psf_on.shape[0]/2.0)) * 0.00996
    extent = [extent_rng, -extent_rng, -extent_rng, extent_rng]

    lim = 0.75
    
    plt.close(1)
    plt.figure(1, figsize=(15, 10))
    plt.clf()
    plt.subplots_adjust(left=0.2, bottom=0.0, right=1.0, hspace=0, wspace=0)
    
    plt.subplot(231)
    plt.imshow(psf_on, cmap="gray", norm=norm, extent=extent)
    plt.setp(plt.gca().get_yticklabels(), visible=False)
    plt.setp(plt.gca().get_xticklabels(), visible=False)
    plt.gca().get_xaxis().set_visible(False)
    plt.ylabel('PSF')
    plt.title('On Axis')
    plt.xlim(lim, -lim)
    plt.ylim(-lim, lim)

    plt.subplot(234)
    plt.imshow(otf_on, cmap="plasma", extent=extent)
    plt.ylabel('OTF')
    plt.setp(plt.gca().get_yticklabels(), visible=False)
    plt.setp(plt.gca().get_xticklabels(), visible=False)
    plt.gca().get_xaxis().set_visible(False)
    # plt.gca().get_yaxis().set_visible(False)
    plt.xlim(lim, -lim)
    plt.ylim(-lim, lim)
    plt.gca().set_yticks([])
    plt.gca().set_xticks([])

    plt.subplot(232)
    plt.imshow(psf_off_atm, cmap="gray", norm=norm, extent=extent)
    plt.gca().get_xaxis().set_visible(False)
    plt.gca().get_yaxis().set_visible(False)
    plt.title('Off Axis: Atm')
    plt.xlim(lim, -lim)
    plt.ylim(-lim, lim)

    plt.subplot(235)
    plt.imshow(otf_off_atm, cmap="plasma", extent=extent)
    plt.gca().get_xaxis().set_visible(False)
    plt.gca().get_yaxis().set_visible(False)
    plt.xlim(lim, -lim)
    plt.ylim(-lim, lim)

    plt.subplot(233)
    plt.imshow(psf_off_both, cmap="gray", norm=norm, extent=extent)
    plt.gca().get_xaxis().set_visible(False)
    plt.gca().get_yaxis().set_visible(False)
    plt.title('Off Axis: Atm + Inst')
    plt.xlim(lim, -lim)
    plt.ylim(-lim, lim)

    plt.subplot(236)
    plt.imshow(otf_off_both, cmap="plasma", extent=extent)
    plt.gca().get_xaxis().set_visible(False)
    plt.gca().get_yaxis().set_visible(False)
    plt.xlim(lim, -lim)
    plt.ylim(-lim, lim)
    
    plt.tight_layout()

    fig_dir = paper_dir + 'figures/lgs_atm_inst/'
    fileUtil.mkdir(fig_dir)
    
    fig_out = fig_dir + 'lgs_atm_inst.png'
    plt.savefig(fig_out)

    return
    
    
def plot_mass_dimm():
    path = os.environ['AIROPA_DATA_PATH']
    mass_file = path + '/20120725.masspro.dat'
    dimm_file = path + '/20120725.dimm.dat.txt'

    
    mass = Table.read(mass_file, format='ascii')
    dimm = Table.read(dimm_file, format='ascii')

    dt_mass = get_mass_dimm_datetimes(mass)
    dt_dimm = get_mass_dimm_datetimes(dimm)
    
    # Find the time that we use in our plots for the paper. 
    # 2012-07-26 11:00:00 UT
    # 2012-07-26 01:00:00 HST
    dt_used = datetime.datetime.strptime('2012-07-26 1:00:00', '%Y-%m-%d %H:%M:%S')

    idx_mass = np.argmin(np.abs(dt_mass - dt_used))
    idx_dimm = np.argmin(np.abs(dt_dimm - dt_used))
    
    mass_heights = np.array([0.5, 1.0, 2.0, 4.0, 8.0, 16.0])
    mass = mass[idx_mass]
    dimm = dimm[idx_dimm]

    mass_vals = np.array([mass['col7'], mass['col8'], mass['col9'],
                          mass['col10'],mass['col11'],mass['col12']])

    plt.close(1)
    plt.figure(1)
    plt.plot(mass_vals, mass_heights, 'ks-')
    plt.ylim(0, 17)
    plt.ylabel(r'Height (km)')
    plt.xlabel(r'Cn2 (m$^{-2/3}$)')

    return

def get_mass_dimm_datetimes(table):
    dt_all = []

    for ii in range(len(table)):
        dt_str_fmt = '{0:d}-{1:d}-{2:d} {3:d}:{4:.0f}:{5:.0f}'
        dt_str = dt_str_fmt.format(table['col1'][ii], table['col2'][ii], table['col3'][ii],
                                   table['col4'][ii], table['col5'][ii], table['col6'][ii])

        dt = datetime.datetime.strptime(dt_str, '%Y-%m-%d %H:%M:%S')
        dt_hst = datetime.datetime.strptime(dt_str, '%Y-%m-%d %H:%M:%S')
        dt_all.append(dt_hst)

    return np.array(dt_all)
    
