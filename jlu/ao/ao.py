import numpy as np
import pylab as py
import math
from astropy.io import fits
from astropy.table import Table
from imaka.reduce import reduce_fli


def plot_strehl_vs_angle():
    """
    Plot up the effect of anisoplanatism for Drew Newman.
    """

    # Median conditions at Mauna Kea pulled from
    # http://www.oir.caltech.edu/twiki_oir/pub/Keck/NGAO/NewKAONs/KAON_503_Mauna_Kea_Ridge_Turbulence_Models_v1.1.pdf
    
    theta0_opt = 2.7  # arcsec at 0.5 micron
    wave0_opt = 0.5   # micron

    wave0 = 2.12      # K-band
    theta0 = theta0_opt * (wave0 / wave0_opt)**1.2

    # Compare a perfect on-axis PSF (Strehl = 1) to an off-axis
    # PSF at different distances and see what anisoplanatism has done.
    theta = np.arange(0, 2.1, 0.1) * theta0
    wfeSq = (theta / theta0)**(5.0/3.0)
    strehl = math.e**(-wfeSq)   # Maraschel approximation

    py.clf()
    py.plot(theta, strehl)
    py.xlabel('Angle Offset from Laser (")')
    py.ylabel('Strehl Off-Axis / Strehl On-Axis')
    py.title('Isoplanatic Angle = %.1f"' % theta0)
    py.xlim(0, 2*theta0)
    py.savefig('strehl_vs_angle_anisoplanatism.png')
    py.savefig('strehl_vs_angle_anisoplanatism.jpg')

    
def test_for_jitter(img_nums, outfile='test_for_jitter.png'):
    """
    Read in a stack of starlists from images that weren't dithered.
    The image list should be produced by imaka.reduce.reduce_fli.find_stars()
    """
    star_lists = ['n{0:04d}_stars.txt'.format(ii) for ii in img_nums]
    
    shift_trans = reduce_fli.get_transforms_from_starlists(star_lists)
    
    shiftx = [st.px[0] for st in shift_trans]
    shifty = [st.py[0] for st in shift_trans]

    py.clf()
    py.plot(img_nums, shiftx, 'r.', label='X')
    py.plot(img_nums, shifty, 'b.', label='Y')
    py.legend(loc='upper right')
    py.xlim(img_nums.min() - 1, img_nums.max() + 1)
    py.ylim(-10, 10)
    py.xlabel('Image Number')
    py.ylabel('Shift (pixels)')
    py.savefig(outfile)
    
    return
