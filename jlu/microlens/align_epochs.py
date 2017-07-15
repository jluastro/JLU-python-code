import numpy as np
import os
import pdb
import sys
import string
from jlu.microlens import residuals
from astropy.table import Table
from astropy.modeling import models, fitting
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import scipy.stats
from flystar import align
from flystar import match
from flystar import starlists
from flystar import transforms
import glob

def make_align_list(root='/Users/jlu/work/microlens/OB120169/',
                    prefix='analysis', date='2016_06_22', target='ob120169',
                    refEpoch='13apr'):
    an_dir = root + prefix + '_' + date + '/'
    lis_dir = an_dir + prefix + '_' + target + '_' + date + '/lis/'
    starlists = glob.glob(lis_dir + '*radTrim.lis')
    dates = np.zeros(len(starlists), dtype=float)

    # Open each starlist and get the first date entry.
    # Store in the dates array.
    for ss in range(len(starlists)):
        _lis = open(starlists[ss], 'r')

        for line in _lis:
            fields = line.split()

            if fields[0].startswith('#'):
                continue

            dates[ss] = float(fields[2])
            break
            
    # Sort the dates
    sdx = dates.argsort()

    # Write out the InputAlign.list files
    align_dir = an_dir + prefix + '_' + target + '_' + date + '/align/'
    _align = open(align_dir + 'InputAlign.list', 'w')

    fmt = '../lis/{0:s} {1:d} {2:s}\n'
    for ss in sdx:
        starlist_filename = starlists[ss].split('/')[-1]
        if refEpoch in starlists[ss]:
            _align.write(fmt.format(starlist_filename, 9, 'ref'))
        else:
            _align.write(fmt.format(starlist_filename, 9, ''))

    _align.close()
    

def align_loop(root='/Users/jlu/work/microlens/OB120169/', prefix='analysis', target='ob120169', 
               stars=['OB120169', 'OB120169_L', 'S24_18_0.8', 'S10_17_1.4', 'S19_18_1.6', 'S7_17_2.3'], date='2016_06_22', 
               transforms=[3,4,5], magCuts=[22], weightings=[4],
               Nepochs='8', overwrite=False, nMC=100,
               makePlots=False, DoAlign=False, letterStart=0,
               restrict=False):
    """
    root -- The main directory for both the data and analysis sub-directories.
    prefix -- First part of new sub directory name.
    target -- Name of target used in sub directory and also assumed to have
              <target>_label.txt for label file input to align.
    stars -- Names of PSF stars, including the target's name, that are passed into the plotting routine.
    date -- String representation of the date, added to sub-dir.
    transforms -- The align -a flags to iterate over.
    magCuts -- The align -m flags to iterate over.
    weightings -- The align -w flags to iterate over.
    restrict -- (def=False) If True, use the -restrict flag in align.

    Code originally written by Evan Sinukoff. Modified by J.R. Lu
    """
    
    # Read in the label.dat file. Target must be in first row of label.dat file.
    analysisDir = prefix + '_' + date + '/'
    root = root + analysisDir
    sourceDir = prefix + '_' + target + '_' + date + '/'
    labelFile = target + '_label.dat'
    data = Table.read(root + sourceDir + 'source_list/'+ labelFile, format='ascii')
    transposed = zip(*data)

    if restrict:
        restrict = '-restrict'
    else:
        restrict = ''

    # Decide which sources we are going to be omitting for the align bootstrap.
    Nlines = len(data)
    if nMC > 1:
        Nomit = 0     #Do MC w/ half-sampling
    else:
        Nomit = Nlines  # Do jackknife
    
    a, m, w, o, Ntrials = TrialPars(Nomit, transforms, magCuts, weightings)
    DirNames = GetDirNames(a, m, w, o, Ntrials, target, date, prefix, nMC)

    labFmt = { 'name': '%-13s',    'kp': '%4.2f', 'xarc': '%7.3f', 'yarc': '%7.3f',
               'xerr': '%6.3f',  'yerr': '%6.3f',   'vx': '%7.3f',   'vy': '%7.3f',
              'vxerr': '%6.3f', 'vyerr': '%6.3f',   't0': '%8.3f',  'use': '%3d',
                'r2d': '%7.3f'}

    if DoAlign == True:
        n = -1

        for i in range(Ntrials):
            os.chdir(root)
            n = n + 1
            
            NewDirPath = root + DirNames[n]
            if (os.path.isdir(NewDirPath) == 0 or overwrite == True):    
                try:
                    os.mkdir(DirNames[n])
                except: 
                    os.system('rm -r ' + DirNames[n])
                    os.mkdir(DirNames[n])
                print( 'Creating new directory: ' + NewDirPath)

                os.chdir(DirNames[n])
                os.system('mkdir align lis plots points_d polyfit_d scripts source_list')
                os.system('cp ../' + sourceDir + '/lis/*radTrim* lis')

                if o[i] == None:
                    data.write('source_list/' + labelFile, format='ascii.fixed_width_no_header', delimiter=' ', formats=labFmt)
                else:
                    data_tmp = data.copy(copy_data=True)
                    data_tmp['use'][o[i]] = 0
                    data_tmp.write('source_list/' + labelFile, format='ascii.fixed_width_no_header', delimiter=' ', formats=labFmt)
                    
                os.system('cp ../' + sourceDir + '/align/*Input* align/align.list') 
                os.chdir('align')
                os.system('java align ' + restrict + ' -i -a ' + str(a[n]) + ' -m ' + str(m[n]) + ' -w ' + str(w[n]) + ' -n ' +
                           str(nMC) + ' -p -N ../source_list/' + target + '_label.dat align.list')
                os.system('java trim_align -e ' + Nepochs + ' -r align_t -N ../source_list/' + target +
                          '_label.dat -f ../points_d/ -p align')
                os.system('polyfit -d 1 -linear -i align_t -points ../points_d/ -o ../polyfit_d/fit')

            else:       
                if os.path.isdir(NewDirPath) == 1: 
                    print( 'Directory ' + NewDirPath + ' already exists.  Did not overwrite.' )

    os.chdir(root)
    if makePlots == True:
        print( 'Plotting...')
        AlignPlot(root=root, DirNames=DirNames, NjackKnifes=Nomit, magCut=m, target=target, stars=stars)

    return
    


def AlignPlot(root, DirNames, NjackKnifes, magCut, target, stars): 
	   
    Ndirs = len(DirNames)
    n = -1
    for i in range(int(Ndirs / (NjackKnifes + 1))):
        velXarr = []
        velYarr = []
        
        for j in range(NjackKnifes + 1):
            n += 1
            print( DirNames[n] )
            os.chdir(root + DirNames[n])
            makeResPlot(stars=stars, root=root, DirName=DirNames[n])  # Comment these out as needed
            makeVectorPlot(root=root, DirName=DirNames[n], magCut=magCut[n], target=target) # Comment these out as needed
            residuals.chi2_dist_all_epochs('align/align_t', root_dir=root + DirNames[n] + '/')
            residuals.sum_all_stars(root=root + DirNames[n] + '/', target=target)
            
            if j != 0:
                velX, velY = GetVel(root, DirNames[n])
                velXarr.append(velX)
                velYarr.append(velY)
                
        # If we are done with our jacknifes, plot up results.
        if NjackKnifes > 1:
            NewDirPath = root+DirNames[n]
            str1,str2 = NewDirPath.split('MCj_o') 
            PlotVelSig(str1 + 'MCj_o00/plots', velXarr, velYarr)   

    # velXlist = []
    # velYlist = []
    # velSigXlist = []
    # velSigYlist = []
    # for i in range(Ndirs/2):
    #     print( DirNames[2*i+1] )
    #     velXarr, velYarr = GetVel(root, DirNames[2*i+1])
    #     print( len(velXarr) )
    #     velSigXarr, velSigYarr, Nstars = GetVelSigDist(root, DirNames[2*i+1])
    #     velXlist.append(velXarr)
    #     velYlist.append(velYarr)
    #     velSigXlist.append(velSigXarr)
    #     velSigYlist.append(velSigYarr)
    # relErrX = []
    # relErrY = []
    # # print( len(velXlist), Nstars )
    # # print( velXlist )
    # for i in range(Ndirs/2):
    #     for j in range(Ndirs/2):
    #         for k in range (Nstars):
    #             print( i, j, k )
    #             if i != j:
    #                 deltaVx = velXlist[i][k] - velXlist[j][k]
    #                 sigVx = np.sqrt(velSigXlist[i][k]**2. + velSigXlist[j][k]**2.)
    #                 relErrX.append(deltaVx/sigVx)
    #                 deltaVy = velYlist[i][k] - velYlist[j][k]
    #                 sigVy = np.sqrt(velSigYlist[i][k]**2. + velSigYlist[j][k]**2.)
    #                 relErrY.append(deltaVy/sigVy)
    # PlotVelSigDist(root + DirNames[0], relErrX, relErrY)
    # print( 'plotting Velocity error distribution (sigma) in directory: ' + root + DirNames[0] )


def TrialPars(Nomit, transforms, magCuts, weightings):
    Ntrans = len(transforms)
    Nmags = len(magCuts)
    Nweights = len(weightings)
    
    Ntrials = Ntrans*Nmags*Nweights*(Nomit + 1)
    a = []
    m = []
    w = []
    o = []
    
    for i in range(Ntrans):   
        for j in range(Nmags):
            for k in range(Nweights):
                # The run with no omissions:
                a.append(transforms[i])
                m.append(magCuts[j])
                w.append(weightings[k])
                o.append(None)

                # The runs with omissions (only if jacknife)
                for l in range(Nomit):
                   a.append(transforms[i])
                   m.append(magCuts[j])
                   w.append(weightings[k])
                   o.append(l)
    
    return a, m, w, o, Ntrials     



def GetDirNames(a, m, w, o, Ntrials, target, date, prefix, nMC):
    DirNames = []

    for i in range(Ntrials):
        a_part = '_a{0:d}'.format(a[i])
        m_part = '_m{0:d}'.format(m[i])
        w_part = '_w{0:d}'.format(w[i])

        if nMC > 1:
            # In this case, half-sample bootstarp and Nomit = 1 (which means it doesn't matter)
            mc_part = '_MC{0:03d}'.format(nMC)
            o_part = ''
        else:
            if o[i] == None:
                # Jacknife, but we have to do everything together first.
                mc_part = '_MCj'
                o_part = '_oAll'
            else:
                mc_part = '_MCj'
                # Jacknife, now omit one star.
                o_part = '_o{0:02d}'.format(o[i])

        new_dir = prefix + '_' + target + '_' + date
        new_dir += a_part + m_part + w_part + mc_part + o_part

        DirNames.append(new_dir)
            
    return DirNames

        

def makeVectorPlot(root, DirName, magCut, target):  
    residuals.ResVectorPlot(root=root + DirName + '/', align='align/align_t',poly='polyfit_d/fit', useAccFits=False, TargetName=target, magCut=magCut)
    plt.show()


def makeResPlot(stars, root, DirName):
#       residuals.plotStar(['ob110022', 'p001_14_1.8', 'p002_16_1.0', 's000_16_1.1', 'p003_16_2.3', 's002_17_1.5'], 
#                         rootDir=root + DirName + '/', align='align/align_t', poly='polyfit_d/fit', points='/points_d/', 
#                         radial=False, NcolMax=3, figsize=(15,15))
#     residuals.plotStar(['ob110125', 'S1_16_3.9', 'S6_17_3.8', 'S13_18_1.7', 'S14_18_2.7', 'p004_18_3.0'], 
#                        rootDir=root + DirName + '/', align='align/align_t', poly='polyfit_d/fit', points='/points_d/', 
#                        radial=False, NcolMax=3, figsize=(15,15))     
    # residuals.plotStar(['OB120169', 'p005_15_3.5', 'S2_16_2.5', 'p000_16_3.6', 'S6_17_2.4', 'OB120169_L'], 
    #                    rootDir=root + DirName + '/', align='align/align_t', poly='polyfit_d/fit', points='/points_d/', 
    #                    radial=False, NcolMax=3, figsize=(15,15))
    numStars = int(len(stars))
    residuals.plotStar(starNames=stars, 
                       rootDir=root + DirName + '/', align='align/align_t', poly='polyfit_d/fit', points='/points_d/', 
                       radial=False, NcolMax=numStars, figsize=(15,numStars*10))
    plt.show()


    
def GetVel(root, DirName):

    velX = np.genfromtxt(root + DirName + '/polyfit_d/fit.linearFormal', usecols=(2), comments=None, 
                        unpack=True, dtype=None, skip_header=1)
    velY = np.genfromtxt(root + DirName + '/polyfit_d/fit.linearFormal', 
                        usecols=(8), comments=None, 
                        unpack=True, dtype=None, skip_header=1)
    return velX, velY                    



def PlotVelSig(plotDir, velXarr, velYarr):

    Njackknifes = len(velXarr)
    Nstars=[]
    for i in range(Njackknifes):
        Nstars.append(len(velXarr[i]))    
    
    for i in range(np.max(Nstars)):
        velXstar = []
        velYstar = []
        for j in range(Njackknifes):
            if i < Nstars[j]:
                velXstar.append(velXarr[j][i]*10.0)
                velYstar.append(velYarr[j][i]*10.0)
        
        rmsX = np.std(np.array(velXstar))
        rmsY = np.std(np.array(velYstar))
        maxX = np.max(np.array(velXstar))
        minX = np.min(np.array(velXstar))
        maxY = np.max(np.array(velYstar))
        minY = np.min(np.array(velYstar))
        medX = np.median(np.array(velXstar))
        medY = np.median(np.array(velYstar))

        plt.paxes = plt.subplot(1, 2, 1)
        plt.xlim(0,np.max(Nstars)+1)
#        plt.plot(np.zeros(len(velXstar))+(i+1), velXstar, 'k*-', markersize=3)
        plt.errorbar(i+1, medX, yerr=[[medX-minX],[maxX-medX]], color='k', markersize=0)
        rwidth = 0.5
        rect = plt.Rectangle((i+0.5+rwidth/2., medX-rmsX), rwidth, 2*rmsX,  ec='red', fc='red', zorder=10)
        plt.gca().add_patch(rect)
        plt.xlabel('Star #')
        plt.ylabel('X velocity [mas/yr]')

                      
        if i < Njackknifes:
            plt.plot(np.zeros(1)+(i+1), velXstar[i], 'b*',markersize=3, markeredgecolor='b',zorder=10)
		
        plt.paxes = plt.subplot(1, 2, 2)
        plt.xlim(0,np.max(Nstars)+1)
        plt.errorbar(i+1, medY, yerr=[[medY-minY],[maxY-medY]], color='k', markersize=0)
        rect = plt.Rectangle((i+0.5+rwidth/2., medY-rmsY), rwidth, 2*rmsY,  ec='red', fc='red', zorder=10)
        plt.gca().add_patch(rect)
#        plt.plot(np.zeros(len(velYstar))+(i+1), velYstar, 'k*-')
        plt.xlabel('Star #')
        plt.ylabel('Y velocity [mas/yr]')
        if i < Njackknifes:
            plt.plot(np.zeros(1)+(i+1), velYstar[i], 'b*',markersize=3, markeredgecolor='b',zorder=10)

    
    plt.subplots_adjust(left = 0.15, wspace=0.4, hspace=0.33, right=0.9, top=0.9)
    
    plt.savefig(plotDir + '/Vel_JackKnife.png')
    
    plt.clf()



def GetVelSigDist(root, DirName):

    sigVelX = np.genfromtxt(root + DirName + '/polyfit_d/fit.linearFormal', usecols=(4), comments=None,
                     unpack=True, dtype=None, skip_header=1)
    sigVelY = np.genfromtxt(root + DirName + '/polyfit_d/fit.linearFormal',
                         usecols=(10), comments=None,
                         unpack=True, dtype=None, skip_header=1)
    
    Nstars = len(sigVelX)

    return sigVelX, sigVelY, Nstars



def PlotVelSigDist(plotDir, relErrX, relErrY):
    
    plt.clf()
    fontsize1 = 10
    bins = np.arange(-7, 7, 1)
    print( relErrX )
    paxes = plt.subplot(1, 2, 1)
    (n, b, p) = plt.hist(relErrX, bins, color='b')
    plt.axis([-5, 5, 0, 180], fontsize=10)
    plt.xlabel('X Residuals (sigma)', fontsize=fontsize1)
    plt.ylabel('Number of Trias', fontsize=fontsize1)
    ggx = np.arange(-7, 7, 0.25)
    ggy = plt.normpdf(ggx, 0, 1)
    ggamp = ((plt.sort(n))[-2:]).sum() / (2.0 * ggy.max())
    plt.plot(ggx, ggy*ggamp, 'r-', linewidth=2)

    paxes = plt.subplot(1, 2, 2)
    
    #             subplot(3, 2, 5)
    (n, b, p) = plt.hist(relErrY, bins, color='b')
    plt.axis([-5, 5, 0, 180], fontsize=10)
    plt.xlabel('Y Residuals (sigma)', fontsize=fontsize1)
    plt.ylabel('Number of Trials', fontsize=fontsize1)
    ggx = np.arange(-7, 7, 0.25)
    ggy = plt.normpdf(ggx, 0, 1)
    ggamp = ((plt.sort(n))[-2:]).sum() / (2.0 * ggy.max())
    plt.plot(ggx, ggy*ggamp, 'r-', linewidth=2)
    plt.savefig(plotDir + '/velDist_a3_m20_w1-4.png')
    plt.clf()


def make_matched_catalog(root='/Users/jlu/work/microlens/OB110022/', prefix='analysis', target='ob110022', 
                        date='2016_03_18', sourceDir='analysis_ob110022_2016_03_18', output_name='stars_matched.fits'):

    # Find the starlists.
    lis_dir = root + prefix + '_' + date + '/' + sourceDir + '/lis/'
    print( lis_dir )
    star_lists = glob.glob(lis_dir + 'mag*_radTrim.lis')

    # Use the middle starlist as a reference... this will be arbitrary in the end.
    ref = int(len(star_lists) / 2)
    print( ref )
    print( star_lists )
    ref_list = starlists.read_starlist(star_lists[ref], error=True)

    # Calculate N_epochs and N_stars, where N_stars starts as the number in the
    # reference epoch. This will grow as we find new stars in the new epochs.
    N_epochs = len(star_lists)
    N_stars = len(ref_list)

    N_loops = 2

    # For each starlist, align and match to the reference. Since these aren't rotated and angled,
    # we should be able to blind match pretty easily.
    ref_list.meta['L_REF'] = ref
    ref_list.meta['N_epochs'] = N_epochs
    
    for ii in range(len(star_lists)):
        if ii == ref:
            suffix = '_{0:d}'.format(ii)
            ref_list.meta['L' + suffix] = os.path.split(star_lists[ii])[-1]

            ref_list['name' + suffix] = ref_list['name']
            ref_list['t' + suffix] = ref_list['t']
            ref_list['x' + suffix] = ref_list['x']
            ref_list['y' + suffix] = ref_list['y']
            ref_list['m' + suffix] = ref_list['m']
            ref_list['xe' + suffix] = ref_list['xe']
            ref_list['ye' + suffix] = ref_list['ye']
            
            continue

        # Read in the starlist into an astropy table.
        star_list = starlists.read_starlist(star_lists[ii], error=True)

        # Preliminary match and calculate a 1st order transformation.
        trans = align.initial_align(star_list, ref_list, briteN=50,
                                    transformModel=transforms.PolyTransform, order=1)

        # Repeat transform + match several times.
        for nn in range(N_loops):
            # Apply the transformation to the starlist.
            star_list_T = Table.copy(star_list)
            star_list_T = align.transform_from_object(star_list_T, trans)

            # Match stars between the lists.
            idx1, idx2, dm, dr = match.match(star_list_T['x'], star_list_T['y'], star_list_T['m'],
                                            ref_list['x'], ref_list['y'], ref_list['m'],
                                            dr_tol=5, dm_tol=2)
            print( 'In Loop ', nn, ' found ', len(idx1), ' matches' )
            
    
            # Calculate transform based on the matched stars    
            trans = transforms.PolyTransform(star_list['x'][idx1], star_list['y'][idx1],
                                             ref_list['x'][idx2], ref_list['y'][idx2],
                                             order=1)

        
        # The point is matching... lets do one final match.
        star_list_T = Table.copy(star_list)
        star_list_T = align.transform_from_object(star_list_T, trans)
        idx1, idx2, dm, dr = match.match(star_list_T['x'], star_list_T['y'], star_list_T['m'],
                                            ref_list['x'], ref_list['y'], ref_list['m'],
                                            dr_tol=3, dm_tol=2)

        # Add the matched stars to the reference list table. These will be un-transformed positions.
        suffix = '_{0:d}'.format(ii)
        ref_list['name' + suffix] = np.zeros(len(ref_list), dtype='S15')
        ref_list['t' + suffix] = np.zeros(len(ref_list), dtype=float)
        ref_list['x' + suffix] = np.zeros(len(ref_list), dtype=float)
        ref_list['y' + suffix] = np.zeros(len(ref_list), dtype=float)
        ref_list['m' + suffix] = np.zeros(len(ref_list), dtype=float)
        ref_list['xe' + suffix] = np.zeros(len(ref_list), dtype=float)
        ref_list['ye' + suffix] = np.zeros(len(ref_list), dtype=float)

        ref_list['name' + suffix][idx2] = star_list['name'][idx1]
        ref_list['t' + suffix][idx2] = star_list['t'][idx1]
        ref_list['x' + suffix][idx2] = star_list['x'][idx1]
        ref_list['y' + suffix][idx2] = star_list['y'][idx1]
        ref_list['m' + suffix][idx2] = star_list['m'][idx1]
        ref_list['xe' + suffix][idx2] = star_list['xe'][idx1]
        ref_list['ye' + suffix][idx2] = star_list['ye'][idx1]

        ref_list.meta['L' + suffix] = os.path.split(star_lists[ii])[-1]


    # Clean up th extra copy of the reference star list:
    ref_list.remove_columns(['name', 't', 'x', 'y', 'm', 'xe', 'ye'])
    ref_list.remove_columns(['snr', 'corr', 'N_frames', 'flux'])

    ref_list.write(lis_dir + output_name, overwrite=True)

    return ref_list


def run_align_iter(catalog, trans_order=1, poly_deg=1, ref_mag_lim=19, ref_radius_lim=300):
    # Load up data with matched stars.
    d = Table.read(catalog)

    # Determine how many epochs there are.
    N_epochs = len([n for n, c in enumerate(d.colnames) if c.startswith('name')])

    # Determine how many stars there are. 
    N_stars = len(d)

    # Determine the reference epoch
    ref = d.meta['L_REF']

    # Figure out the number of free parameters for the specified
    # poly2d order.
    poly2d = models.Polynomial2D(trans_order)
    N_par_trans_per_epoch = 2.0 * poly2d.get_num_coeff(2)  # one poly2d for each dimension (X, Y)
    N_par_trans = N_par_trans_per_epoch * N_epochs

    ##########
    # First iteration -- align everything to REF epoch with zero velocities. 
    ##########
    print('ALIGN_EPOCHS: run_align_iter() -- PASS 1')
    ee_ref = d.meta['L_REF']

    target_name = 'OB120169'

    trans1, used1 = calc_transform_ref_epoch(d, target_name, ee_ref, ref_mag_lim, ref_radius_lim)

    ##########
    # Derive the velocity of each stars using the round 1 transforms. 
    ##########
    calc_polyfit_all_stars(d, poly_deg, init_fig_idx=0)

    calc_mag_avg_all_stars(d)
    
    tdx = np.where((d['name_0'] == 'OB120169') | (d['name_0'] == 'OB120169_L'))[0]
    print(d[tdx]['name_0', 't0', 'mag', 'x0', 'vx', 'x0e', 'vxe', 'chi2x', 'y0', 'vy', 'y0e', 'vye', 'chi2y', 'dof'])

    ##########
    # Second iteration -- align everything to reference positions derived from iteration 1
    ##########
    print('ALIGN_EPOCHS: run_align_iter() -- PASS 2')
    target_name = 'OB120169'

    trans2, used2 = calc_transform_ref_poly(d, target_name, poly_deg, ref_mag_lim, ref_radius_lim)
    
    ##########
    # Derive the velocity of each stars using the round 1 transforms. 
    ##########
    calc_polyfit_all_stars(d, poly_deg, init_fig_idx=4)

    ##########
    # Save output
    ##########
    d.write(catalog.replace('.fits', '_aln.fits'), overwrite=True)
    
    return

def calc_transform_ref_epoch(d, target_name, ee_ref, ref_mag_lim, ref_radius_lim):
    # Determine how many epochs there are.
    N_epochs = len([n for n, c in enumerate(d.colnames) if c.startswith('name')])

    # output array
    trans = []
    used = []

    # Find the target
    tdx = np.where(d['name_0'] == 'OB120169')[0][0]

    # Reference values
    t_ref = d['t_{0:d}'.format(ee_ref)]
    m_ref = d['m_{0:d}'.format(ee_ref)]
    x_ref = d['x_{0:d}'.format(ee_ref)]
    y_ref = d['y_{0:d}'.format(ee_ref)]
    xe_ref = d['xe_{0:d}'.format(ee_ref)]
    ye_ref = d['ye_{0:d}'.format(ee_ref)]
    
    # Calculate some quanitites we use for selecting reference stars.
    r_ref = np.hypot(x_ref - x_ref[tdx], y_ref - y_ref[tdx])

    # Loop through and align each epoch to the reference epoch.
    for ee in range(N_epochs):
        # Pull out the X, Y positions (and errors) for the two
        # starlists we are going to align.
        x_epo = d['x_{0:d}'.format(ee)]
        y_epo = d['y_{0:d}'.format(ee)]
        t_epo = d['t_{0:d}'.format(ee)]
        xe_epo = d['xe_{0:d}'.format(ee)]
        ye_epo = d['ye_{0:d}'.format(ee)]

        # Figure out the set of stars detected in both epochs.
        idx = np.where((t_ref != 0) & (t_epo != 0) & (xe_ref != 0) & (xe_epo != 0))[0]

        # Find those in both epochs AND reference stars. This is [idx][rdx]
        rdx = np.where((r_ref[idx] < ref_radius_lim) & (m_ref[idx] < ref_mag_lim))[0]
        
        # Average the positional errors together to get one weight per star.
        xye_ref = (xe_ref + ye_ref) / 2.0
        xye_epo = (xe_epo + ye_epo) / 2.0
        xye_wgt = (xye_ref**2 + xye_epo**2)**0.5
        
        # Calculate transform based on the matched stars    
        trans_tmp = transforms.PolyTransform(x_epo[idx][rdx], y_epo[idx][rdx], x_ref[idx][rdx], y_ref[idx][rdx],
                                                 weights=xye_wgt[idx][rdx], order=2)

        trans.append(trans_tmp)


        # Apply thte transformation to the stars positions and errors:
        xt_epo = np.zeros(len(d), dtype=float)
        yt_epo = np.zeros(len(d), dtype=float)
        xet_epo = np.zeros(len(d), dtype=float)
        yet_epo = np.zeros(len(d), dtype=float)
        
        xt_epo[idx], xet_epo[idx], yt_epo[idx], yet_epo[idx] = trans_tmp.evaluate_errors(x_epo[idx], xe_epo[idx],
                                                                                         y_epo[idx], ye_epo[idx],
                                                                                         nsim=100)

        d['xt_{0:d}'.format(ee)] = xt_epo
        d['yt_{0:d}'.format(ee)] = yt_epo
        d['xet_{0:d}'.format(ee)] = xet_epo
        d['yet_{0:d}'.format(ee)] = yet_epo

        # Record which stars we used in the transform.
        used_tmp = np.zeros(len(d), dtype=bool)
        used_tmp[idx[rdx]] = True

        used.append(used_tmp)

        if True:
            plot_quiver_residuals(xt_epo, yt_epo, x_ref, y_ref, idx, rdx, 'Epoch: ' + str(ee))
            
    used = np.array(used)
    
    return trans, used


def calc_transform_ref_poly(d, target_name, poly_deg, ref_mag_lim, ref_radius_lim):
    # Determine how many epochs there are.
    N_epochs = len([n for n, c in enumerate(d.colnames) if c.startswith('name')])

    # output array
    trans = []
    used = []

    # Find the target
    tdx = np.where(d['name_0'] == 'OB120169')[0][0]

    # Temporary Reference values
    t_ref = d['t0']
    m_ref = d['mag']
    x_ref = d['x0']
    y_ref = d['y0']
    xe_ref = d['x0e']
    ye_ref = d['y0e']    
    
    # Calculate some quanitites we use for selecting reference stars.
    r_ref = np.hypot(x_ref - x_ref[tdx], y_ref - y_ref[tdx])

    for ee in range(N_epochs):
        # Pull out the X, Y positions (and errors) for the two
        # starlists we are going to align.
        x_epo = d['x_{0:d}'.format(ee)]
        y_epo = d['y_{0:d}'.format(ee)]
        t_epo = d['t_{0:d}'.format(ee)]
        xe_epo = d['xe_{0:d}'.format(ee)]
        ye_epo = d['ye_{0:d}'.format(ee)]

        # Shift the reference position by the polyfit for each star.
        dt = t_epo - t_ref
        if poly_deg >= 0:
            x_ref_ee = x_ref
            y_ref_ee = y_ref
            xe_ref_ee = x_ref
            ye_ref_ee = y_ref
            
        if poly_deg >= 1:
            x_ref_ee += d['vx'] * dt
            y_ref_ee += d['vy'] * dt
            xe_ref_ee = np.hypot(xe_ref_ee, d['vxe'] * dt)
            ye_ref_ee = np.hypot(ye_ref_ee, d['vye'] * dt)

        if poly_deg >= 2:
            x_ref_ee += d['ax'] * dt
            y_ref_ee += d['ay'] * dt
            xe_ref_ee = np.hypot(xe_ref_ee, d['axe'] * dt)
            ye_ref_ee = np.hypot(ye_ref_ee, d['aye'] * dt)
            
        # Figure out the set of stars detected in both.
        idx = np.where((t_ref != 0) & (t_epo != 0) & (xe_ref != 0) & (xe_epo != 0))[0]

        # Find those in both AND reference stars. This is [idx][rdx]
        rdx = np.where((r_ref[idx] < ref_radius_lim) & (m_ref[idx] < ref_mag_lim))[0]
        
        # Average the positional errors together to get one weight per star.
        xye_ref = (xe_ref_ee + ye_ref_ee) / 2.0
        xye_epo = (xe_epo + ye_epo) / 2.0
        xye_wgt = (xye_ref**2 + xye_epo**2)**0.5
        
        # Calculate transform based on the matched stars    
        trans_tmp = transforms.PolyTransform(x_epo[idx][rdx], y_epo[idx][rdx], x_ref_ee[idx][rdx], y_ref_ee[idx][rdx],
                                                 weights=xye_wgt[idx][rdx], order=2)
        trans.append(trans_tmp)

        # Apply thte transformation to the stars positions and errors:
        xt_epo = np.zeros(len(d), dtype=float)
        yt_epo = np.zeros(len(d), dtype=float)
        xet_epo = np.zeros(len(d), dtype=float)
        yet_epo = np.zeros(len(d), dtype=float)
        
        xt_epo[idx], xet_epo[idx], yt_epo[idx], yet_epo[idx] = trans_tmp.evaluate_errors(x_epo[idx], xe_epo[idx],
                                                                                         y_epo[idx], ye_epo[idx],
                                                                                         nsim=100)
        pdb.set_trace()

        d['xt_{0:d}'.format(ee)] = xt_epo
        d['yt_{0:d}'.format(ee)] = yt_epo
        d['xet_{0:d}'.format(ee)] = xet_epo
        d['yet_{0:d}'.format(ee)] = yet_epo

        # Record which stars we used in the transform.
        used_tmp = np.zeros(len(d), dtype=bool)
        used_tmp[idx[rdx]] = True

        used.append(used_tmp)

        if True:
            plot_quiver_residuals(xt_epo, yt_epo, x_ref_ee, y_ref_ee, idx, rdx, 'Epoch: ' + str(ee))

    used = np.array(used)
    
    return trans, used

def calc_polyfit_all_stars(d, poly_deg, init_fig_idx=0):
    # Determine how many stars there are. 
    N_stars = len(d)

    # Determine how many epochs there are.
    N_epochs = len([n for n, c in enumerate(d.colnames) if c.startswith('name')])
    
    # Setup some variables to save the results
    t0_all = []
    px_all = []
    py_all = []
    pxe_all = []
    pye_all = []
    chi2x_all = []
    chi2y_all = []
    dof_all = []

    # Get the time array, which is the same for all stars.
    # Also, sort the time indices.
    t = np.array([d['t_{0:d}'.format(ee)][0] for ee in range(N_epochs)])
    tdx = t.argsort()
    t_sorted = t[tdx]
    
    # Run polyfit on each star.
    for ss in range(N_stars):
        # Get the x, y, xe, ye, and t arrays for this star.
        xt = np.array([d['xt_{0:d}'.format(ee)][ss] for ee in range(N_epochs)])
        yt = np.array([d['yt_{0:d}'.format(ee)][ss] for ee in range(N_epochs)])
        xet = np.array([d['xet_{0:d}'.format(ee)][ss] for ee in range(N_epochs)])
        yet = np.array([d['yet_{0:d}'.format(ee)][ss] for ee in range(N_epochs)])
        t_tmp = np.array([d['t_{0:d}'.format(ee)][ss] for ee in range(N_epochs)])

        # Sort these arrays.
        xt_sorted = xt[tdx]
        yt_sorted = yt[tdx]
        xet_sorted = xet[tdx]
        yet_sorted = yet[tdx]
        t_tmp_sorted = t_tmp[tdx]

        # Get only the detected epochs.
        edx = np.where(t_tmp_sorted != 0)[0]

        # Calculate the weighted t0 (using the transformed errors).
        weight_for_t0 = 1.0 / np.hypot(xet_sorted, yet_sorted)
        t0 = np.average(t_sorted[edx], weights=weight_for_t0[edx])

        # for ee in edx:
        #     print('{0:8.3f}  {1:10.5f}  {2:10.5f}  {3:8.5f}  {4:8.5f}'.format(t[ee], xt[ee], yt[ee], xet[ee], yet[ee]))
        # pdb.set_trace()

        # Run polyfit
        dt = t_sorted - t0
        px, covx = np.polyfit(dt[edx], xt_sorted[edx], poly_deg, w=1./xet_sorted[edx], cov=True)
        py, covy = np.polyfit(dt[edx], yt_sorted[edx], poly_deg, w=1./yet_sorted[edx], cov=True)

        pxe = np.sqrt(np.diag(covx))
        pye = np.sqrt(np.diag(covy))


        x_mod = np.polyval(px, dt[edx])
        y_mod = np.polyval(py, dt[edx])
        chi2x = np.sum( ((x_mod - xt_sorted[edx]) / xet_sorted[edx])**2 )
        chi2y = np.sum( ((y_mod - yt_sorted[edx]) / yet_sorted[edx])**2 )
        dof = len(edx) - (poly_deg + 1)

        # Save results:
        t0_all.append(t0)
        px_all.append(px)
        py_all.append(py)
        pxe_all.append(pxe)
        pye_all.append(pye)
        chi2x_all.append(chi2x)
        chi2y_all.append(chi2y)
        dof_all.append(dof)

        if d[ss]['name_0'] in ['OB120169', 'OB120169_L']:
            gs = GridSpec(3, 2) # 3 rows, 1 column
            fig = plt.figure(ss + 1 + init_fig_idx, figsize=(12, 8))
            a0 = fig.add_subplot(gs[0:2, 0])
            a1 = fig.add_subplot(gs[2, 0])
            a2 = fig.add_subplot(gs[0:2, 1])
            a3 = fig.add_subplot(gs[2, 1])
            
            a0.errorbar(t_sorted[edx], xt_sorted[edx], yerr=xet_sorted[edx], fmt='ro')
            a0.plot(t_sorted[edx], x_mod, 'k-')
            a0.set_title(d[ss]['name_0'] + ' X')
            a1.errorbar(t_sorted[edx], xt_sorted[edx] - x_mod, yerr=xet_sorted[edx], fmt='ro')
            a1.axhline(0, linestyle='--')
            a1.set_xlabel('Time (yrs)')
            a2.errorbar(t_sorted[edx], yt_sorted[edx], yerr=yet_sorted[edx], fmt='ro')
            a2.plot(t_sorted[edx], y_mod, 'k-')
            a2.set_title(d[ss]['name_0'] + ' Y')
            a3.errorbar(t_sorted[edx], yt_sorted[edx] - y_mod, yerr=yet_sorted[edx], fmt='ro')
            a3.axhline(0, linestyle='--')
            a3.set_xlabel('Time (yrs)')

            

    t0_all = np.array(t0_all)
    px_all = np.array(px_all)
    py_all = np.array(py_all)
    pxe_all = np.array(pxe_all)
    pye_all = np.array(pye_all)
    chi2x_all = np.array(chi2x_all)
    chi2y_all = np.array(chi2y_all)
    dof_all = np.array(dof_all)
        
    # Done with all the stars... recast as numpy arrays and save to output table.
    d['t0'] = t0_all
    d['chi2x'] = chi2x_all
    d['chi2y'] = chi2y_all
    d['dof'] = dof_all
    if poly_deg >= 0:
        d['x0'] = px_all[:, -1]
        d['y0'] = py_all[:, -1]
        d['x0e'] = pxe_all[:, -1]
        d['y0e'] = pye_all[:, -1]
        
    if poly_deg >= 1:
        d['vx'] = px_all[:, -2]
        d['vy'] = py_all[:, -2]
        d['vxe'] = pxe_all[:, -2]
        d['vye'] = pye_all[:, -2]

    if poly_deg >= 2:
        d['ax'] = px_all[:, -3]
        d['ay'] = py_all[:, -3]
        d['axe'] = pxe_all[:, -3]
        d['aye'] = pye_all[:, -3]

    pdb.set_trace()
        
    return

def calc_mag_avg_all_stars(d):
    # Determine how many stars there are. 
    N_stars = len(d)

    # Determine how many epochs there are.
    N_epochs = len([n for n, c in enumerate(d.colnames) if c.startswith('name')])

    # 2D mag array
    mag_all = np.zeros((N_epochs, N_stars), dtype=float)

    for ee in range(N_epochs):
        mag_all[ee, :] = d['m_{0:d}'.format(ee)]

    mag_all_masked = np.ma.masked_equal(mag_all, 0)
    flux_all_masked = 10**(-mag_all_masked / 2.5)

    flux_avg = flux_all_masked.mean(axis=0)
    mag_avg = -2.5 * np.log10(flux_avg)

    d['mag'] = mag_avg

    return

    

def run_align_multinest(table_stars_matched):
    return
    
def get_epochs_array(stars_table, var_name):
    """
    Fetch a 'multi-epoch' array for the specified variable name. 
    The resulting array that is returned will be 2D with dimensions 
        N_stars x N_epochs
    where N_stars = number of rows in stars_table.

    Parameters
    ----------
    stars_table : astropy.table.Table
        An astropy table with column names in the format 't_0', 't_1', etc.
    var_name : str
        The string name of the column prefix that you want to fetch.

    Outputs
    ----------
    Return a 2D numpy array that contains the requested variables. 
    The dimensions of the array are (N_stars, N_epoch).
    """
    col_name_list = []
    
    for ee in range(stars_table.meta['N_EPOCHS']):
        col_name_list.append('{0:s}_{1:d}'.format(var_name, ee))

    # Get the data type of the first object .. assume it is the same/similar for all.
    dtype = stars_table[col_name_list[0]].dtype

    if dtype.kind not in set('buifc'):
        var_array_tmp = []
        for col in col_name_list:
            var_array_tmp.append(stars_table[col].data)
        var_array =  np.array(var_array_tmp).T
    else:
        var_array_tmp = []
        for col in col_name_list:
            var_array_tmp.append(stars_table[col].data)
        var_array =  np.vstack(var_array_tmp).T
        # var_array = stars_table[col_name_list].as_array()
        # var_array = var_array.view(dtype).reshape((len(var_array.dtype.names), len(stars_table)))

    return var_array

def set_epochs_array(stars_table, var_name, var_array):
    """
    Set a 'multi-epoch' array for the specified variable name. 
    The 2D array (var_array) should have the shape:
        N_stars x N_epochs
    where N_stars = number of rows in stars_table and
    the array will be split into Columns to be set
    on the stars_table as [var_name + '_' + epoch] (i.e. 't_0').

    Parameters
    ----------
    stars_table : astropy.table.Table
        An astropy table with column names in the format 't_0', 't_1', etc.
    var_name : str
        The string name of the column prefix that you want to fetch.
    var_array : numpy
        2D numpy array with shape (N_stars, N_epochs)

    Outputs
    ----------
    Modify the stars_table to reset the columns to the 
    values in var_array.
    """
    for ee in range(stars_table.meta['N_EPOCHS']):
        col_name = '{0:s}_{1:d}'.format(var_name, ee)

        stars_table[col_name] = var_array[:, ee]
        
    return

def plot_quiver_residuals(x_t, y_t, x_ref, y_ref, good_idx, ref_idx, title):
    dx = x_t - x_ref
    dy = y_t - y_ref

    plt.figure(1)
    plt.clf()
    plt.quiver(x_ref[good_idx], y_ref[good_idx], dx[good_idx], dy[good_idx],
                   color='black', scale=10)
    plt.quiver(x_ref[good_idx][ref_idx], y_ref[good_idx][ref_idx], dx[good_idx][ref_idx], dy[good_idx][ref_idx],
                   color='red', scale=10)
    # plt.quiverkey(q, 100, 100, 0.1, '0.1 pix')
    plt.quiver([50], [50], [0.1], [0.0], color='green')
    plt.text(50, 70, '0.1 pix', color='green')
    plt.xlabel('X (ref pix)')
    plt.ylabel('Y (ref pix)')
    plt.title(title)
    plt.axis('equal')
    plt.show()
    plt.pause(1)

    str_fmt = 'Residuals (mean, std): dx = {0:7.3f} +/- {1:7.3f}  dy = {2:7.3f} +/- {3:7.3f}'
    print(str_fmt.format(dx[good_idx][ref_idx].mean(), dx[good_idx][ref_idx].std(),
                         dy[good_idx][ref_idx].mean(), dy[good_idx][ref_idx].std()))


    return
    
