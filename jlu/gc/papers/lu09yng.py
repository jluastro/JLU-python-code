import asciidata, pyfits, pickle
import os, sys, math, time, mpfit
import pylab as py
import numpy as np
from gcwork import analyticOrbits as aorb

def newYoungOrbits():
    """
    Create orbit probability-density-functions for all of the new young
    stars identified in Do et al. 2009.
    """
    
    # Load up the newly identified young stars.
    
