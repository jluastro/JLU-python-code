import pyfits
import math
import pylab as py
import numpy as np
import glob
import pdb
import pandas

def plotWaveDiff(inAsc=None):

    # plots the difference between the input wavelength solution and the molecfit fitted wavelength solution
    # inAsc: pass in the .asc file output from molecfit

    # read in molecfit outputs - have to manually pass in column names because the first column is unnamed in the output file
    mf=pandas.read_csv(inAsc,delim_whitespace=True,header=None,skiprows=2,names=['row','chip','lambda','flux','weight','mrange','mlambda','mscal','mflux','mweight','dev','mtrans'])

    # difference between the original wavelength solution and the fitted solution
    diff = mf['lambda'] - mf['mlambda']

    py.figure(3)
    py.clf()
    
    py.plot(diff)
    py.xlabel('Spectral channel')
    py.ylabel('Wavelength difference ($\mu$m)')
    py.title('Difference between original wavelengths and molecfit fit')
