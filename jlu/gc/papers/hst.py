import numpy as np
import pylab as py
import atpy
import subprocess
import os

dataDir = '/u/jlu/work/gc/hst/'
workDir = dataDir + 'pm_2012_05_24/'

def run_xym1mat():
    c17 = atpy.Table(dataDir + 'ks2_17/LOGR.FIND_AVG_UV1_F.fits')
    c18 = atpy.Table(dataDir + 'ks2_18/LOGR.FIND_AVG_UV1_F.fits')

    c17_mag = -2.5 * np.log10(c17.f_1)
    c18_mag = -2.5 * np.log10(c18.f_1)

    idx = np.where(np.isnan(c17_mag) == True)[0]
    c17_mag[idx] = 99.0
    idx = np.where(np.isnan(c18_mag) == True)[0]
    c18_mag[idx] = 99.0

    # Make two files that contain the X, Y, Mag for F153M cycle 17 and 18.
    c17_xym = open(workDir + 'c17_ks2_uv1.xym', 'w')
    c17_xym.write('#%9s  %10s  %10s\n' % ('X', 'Y', 'm153M'))
    c17_xym.write('#%9s  %10s  %10s\n' % ('(pix)', '(pix)', '(mag)'))
    for ii in range(len(c17)):
        c17_xym.write('%10.4f  %10.4f  %10.3f\n' % (c17.x_1[ii], c17.y_1[ii], c17_mag[ii]))
    c17_xym.close()

    c18_xym = open(workDir + 'c18_ks2_uv1.xym', 'w')
    c18_xym.write('%10s  %10s  %10s\n' % ('# X', 'Y', 'm153M'))
    c18_xym.write('%10s  %10s  %10s\n' % ('# (pix)', '(pix)', '(mag)'))
    for ii in range(len(c18)):
        c18_xym.write('%10.4f  %10.4f  %10.3f\n' % (c18.x_1[ii], c18.y_1[ii], c18_mag[ii]))
    c18_xym.close()

    # Now call xym1mat
    os.chdir(workDir)
    subprocess.call(['xym1mat', '"c17_ks2_uv1.xym(*,*,-99:-5)"',
                     '"c18_ks2_uv1.xym(*,*,-99:-5)"', 'a', 'b', 'c', '22', '0.99'])

    
    

    
