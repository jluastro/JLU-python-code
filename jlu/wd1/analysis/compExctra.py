import numpy as np
import csv
import pdb
import os
import sys

def import_text(filename, separator):
    for line in csv.reader(open(filename), delimiter=separator, 
                           skipinitialspace=True):
        if line:
            yield line

def make_Input(filename, band, outputname='input',saveto=''):
    """
    Created on May 08 2013
    filename: usually should be "LOGA.INPUT" 
    band: the name of the filter, which determine which columns should be read.
    However, the columns may vary depend on Jessica's code.
    """
    t=import_text(filename, ' ')
    x=[]
    y=[]
    name=[]
    mag=[]

    if ((band=='F160W')|(band=='F814W')):
        n=17
    elif band=='F139M':
        n=21
    elif band=='F125W':
        n=25
    else:
        return 'Please indicate either IR or V band.'
    for data in t:
        if len(data)>10:
            if data[n]=='************':
                data[n]='0.01'
            if float(data[n])<=0.01:
                data[n]=0.01
            x.append(float(data[0]))
            y.append(float(data[1]))
            name.append(data[16])
            mag.append(float(data[n]))
    mag=-2.5*np.log10(mag)
#    pdb.set_trace()
    text=open(saveto+band+'/'+outputname+'.xym','w')
    for i in range(len(mag)):
#        print str(x[i]),str(y[i]),str(mag[i]),str(name[i])
        text.write('{0:15}'.format(str(x[i])))
        text.write('{0:15}'.format(str(y[i])))
        text.write('{0:20}'.format(str(mag[i])))
        text.write('{0:15} \n'.format(str(name[i])))
    text.close()

