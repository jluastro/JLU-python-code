import pyfits
import math
import pylab as py
import numpy as np
import glob
from jlu.osiris import cube as ocube
import pdb

def edgeMask(inFolder=None):
    # routine to mask edges of individual frames using the bad pixel mask
    # so these will be ignored during mosaicking

    filelist = glob.glob(inFolder+'/*.fits')

    for file in filelist:
        ocube.cubeEdgeMask(cubeFile=file)
