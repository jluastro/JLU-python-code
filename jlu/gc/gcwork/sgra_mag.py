# Reads in data from align files and creates an output file containing
# frame #, magnitudes, and fluxes of Sgr A*. It will also create
# two additional files, one for Sgr A* bright and one for dim

import asciidata
from numarray import *
from gcwork import starset

def go(epoch, limMag=15):
    root = '/u/ghezgroup/data/gc/'+epoch+'/clean/kp/starfinder/align/'

    aln_list = root+'align_kp_0.0.list'
    #frameList = asciidata.open(aln_list)
    f_list = open(aln_list)
    files = []

    for line in f_list:
        _line = line.split()
        fileParts = _line[0].split('/')
        files.append(fileParts[-1])
    
    files = files[1:]
    #frames = frameList[0].tonumarray()
    s=starset.StarSet(root+'align_kp_0.0')

    numstars = asciidata.open(root+'align_kp_0.0.mag').nrows
    numepochs = asciidata.open(root+'align_kp_0.0.mag').ncols - 5

    fluxFile = root+'/sgra_all.mag'
    brtFile = root+'/sgra_brt.mag'
    dimFile = root+'/sgra_dim.mag'

    _sgraAll = open(fluxFile, 'w')
    _sgraBrt = open(brtFile, 'w')
    _sgraDim = open(dimFile, 'w')

    _sgraAll.write('#Frame' + '                   Mag ' + '    Flux(mJy) ' + '\n')
    _sgraBrt.write('#Frame' + '                   Mag ' + '    Flux(mJy) ' + '\n')
    _sgraDim.write('#Frame' + '                   Mag ' + '    Flux(mJy) ' + '\n')

    # Find index for Sgr A* in the mag file
    for x in range(numstars):
        if (s.stars[x].name == 'SgrA'):
            sgra_idx = x

    # Loop through epochs and print frame, mags, & fluxes (in mJy)
    for i in range(numepochs):
        mag = s.stars[sgra_idx].e[i].mag
        flux = 655000. * 10 ** (-0.4*mag)

        #frame = frames[i]
        frame = files[i]

        _sgraAll.write('%5s % 5.3f %7.2f\n' % (frame, mag, flux))
    
        if (mag < limMag):
            _sgraBrt.write('%5s % 5.3f %7.2f\n' % (frame, mag, flux))
        else:
            _sgraDim.write('%5s % 5.3f %7.2f\n' % (frame, mag, flux))

    _sgraAll.close()
    _sgraBrt.close()
    _sgraDim.close()
