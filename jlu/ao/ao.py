import numpy as np
import pylab as py
import math

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

    
    
