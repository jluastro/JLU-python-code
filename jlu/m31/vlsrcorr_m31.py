import pyfits
import math
import pylab as py
import numpy as np
import glob
from jlu.osiris import cube as ocube
import pdb

def vlsrCorr(inFolder=None,night=None):
    # routine to correct cubes for V_LSR
    # feed in a folder that contains reduced cubes, and pass in a
    # night number. All cubes in the folder will be corrected for
    # the appropriate V_LSR

    # night numbers:
    # 20081021: night=1
    # 20100815: night=2
    # 20100828: night=3
    # 20100829: night=4

    filelist = glob.glob(inFolder+'/*.fits')

    # get the V_LSR correction for each night
    # this is the opposite sign of the V_LSR
    if night == 1:
        dv = 3.0
    if night == 2:
        dv = 26.5
    if night == 3:
        dv = 23.5
    if night == 4:
        dv = 23.2

    for file in filelist:
        ocube.cubeDopplerShift(cubeFile=file,dv=dv)

        #pdb.set_trace()
