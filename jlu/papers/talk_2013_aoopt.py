import numpy as np
import pylab as py

out_dir = '/u/jlu/doc/present/2013_06_aoopt/'

def plot_stf_time_test():
    """
    Make a plot showing the times it takes to run Starfinder with different
    PSF grid sizes. The raw numbers were taken from my notes in DevonThink:

    Starfinder Test: Gridded Works? Yes

    The image was a 4096 x 4096 image with a 0.004 pixel scale (TMT image).
    The PSFs used for planting were 4" in diameter.
    The PSFs used for extracting were 2" diameter.
    The run times should scale with image size and PSF size; but I don't know
    exactly how... so leave it for now. Relative should be fine.
    """

    npsf_side = np.array([1, 16, 24, 32])
    stf_time = np.array([3, 5, 13, 34])

    py.clf()
    py.plot(npsf_side, stf_time, 'ro', ms=10, label='Data')
    py.xlabel('Number of PSFs per Side')
    py.ylabel('Time to run STF (min)')

    n = np.arange(0, 35)
    t = 2. * np.exp((n - 16) / 5.7)  + 3.0

    py.plot(n, t, 'k--', label='Model')

    py.legend(loc='upper left')
    py.ylim(0, 40)
    
    
    py.savefig(out_dir + 'plot_stf_time_test.png')

    
