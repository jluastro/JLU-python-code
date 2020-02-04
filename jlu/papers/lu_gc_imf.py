from sqlite3 import dbapi2 as sqlite
import pandas as pd
import pymysql as mdb
import numpy as np
np.seterr(all='warn')
import pylab as py
import matplotlib
from matplotlib.path import Path
#import asciidata
#import atpy
#import pyfits
import math
import os
import scipy
import itertools
from matplotlib import gridspec
from gcwork import objects
from scipy import interpolate
from astropy.io import fits
import pdb
import pickle
from popstar import synthetic, evolution, atmospheres, reddening, evolution
from scipy import optimize
from gcwork import starTables
from gcwork import starset
#from jlu.util import plfit
#from jlu.util import CatalogFinder
from jlu.starfinder import starPlant
#from jlu.stellarModels import extinction
#from jlu.stellarModels import atmospheres
#from jlu.stellarModels import evolution
from matplotlib import path
from scipy.optimize import leastsq
from jlu.util import constants
from pysynphot import spectrum
from scipy import stats
#from jlu.util import img_scale
from gcreduce import gcutil
import pymultinest
import subprocess
from matplotlib.patches import FancyArrow
from mpl_toolkits.axes_grid1 import make_axes_locatable
from astropy.table import Table

workDir_old = '/u/jlu/work/gc/imf/klf/2012_05_01/'
workDir = '/g/lu/scratch/siyao/work/3_young/imf/'
theAKs = 2.7
distance = 8000.0
database = '/u/jlu/data/gc/database/stars.sqlite'
klf_mag_bins = np.arange(9.0, 18, 1.0)  # Bin Center Points

pwFile='/g/lu/scratch/siyao/other/pw.txt'
pw = open(pwFile).read().split()[0]


# fitAlpha = 1.71
# fitAge = 3.92e6
# fitMcl = 9.35e3
# fitDist = 7930
fitAlpha = 1.7
fitAge = 3.9e6
fitMcl = 9.2e3
fitDist = 7900
fitLogAge = math.log10(fitAge)



def analysis_by_radius(rmin=0, rmax=30):
    # image_completeness_in_osiris()
    # image_completeness_by_radius(rmin=rmin, rmax=rmax)
    calc_spec_id_all_stars()
    spec_completeness_by_radius(rmin=rmin, rmax=rmax)
    merge_completeness_by_radius(rmin=rmin, rmax=rmax)
    klf_by_radius(rmin=rmin, rmax=rmax)

    plot_klf_by_radius(rmin=rmin, rmax=rmax)
    plot_klf_by_radius_noWR(rmin=rmin, rmax=rmax)
    plot_klf_by_radius_WR_v_noWR(rmin=rmin, rmax=rmax)
    plot_klf_vs_model_by_radius(rmin=rmin, rmax=rmax)
    plot_klf_vs_bartko_by_radius(rmin=rmin, rmax=rmax)
    plot_klf_vs_bartko_by_radius_12A(rmin=rmin, rmax=rmax)

    # Do the Bayesian analysis (MultiNest) - see notebook to run with parallel processing.
    #fit_with_models(multiples=False)
    #fit_with_models(multiples=True)
    #table_best_fit_params(multiples=True)

def make_paper_plots():
    # Plots for paper (in Tuan's or just useful)
    # plot_image_completeness_vs_radius()
    # plot_klf_progression()
    # plot_klf_vs_bartko_by_radius_paper()
    # plot_WR_vs_age()
    # plot_sim_results()

    # Figures (in order)
    plot_klf_vs_old_models()  # fig 1
    evolution.test_merged_isochrones(6.8)    # fig 2
    plot_WR_OB_ratio_vs_age() # fig 3
    plot_WR_OB_ratio_vs_age() # fig 3   
    sim_age_monte_carlo() # fig 4
    plot_sim_clusters() # fig 5
    plot_sims_vs_multiplicity() # fig 6, fig 7
    plot_sims_vs_imf_slope() # fig 8
    plot_fit_posteriors_1d(multiples=True) # fig 9
    plot_fit_posteriors_2d() # fig 107475
    table_best_fit_params(multiples=True, rmin=0.0) # fig 11
    plot_model_vs_data_MC_for_paper() # fig 11
    compute_masses() # fig 11
    plot_masses_in_imf() # fig 11
    plot_fit_posteriors_1d(multiples=False, rmin=0.0) # fig 12
    plot_fit_posteriors_1d(multiples=True, rmin=0.8) # fig 13
    plot_klf_with_tmt()  # fig 14
    plot_binary_properties() # fig 15
    

def old_analysis():
    img_completeness()
    calc_spec_id_all_stars()
    spec_completeness()
    completeness_to_one_AKs()
    klf()
    plot_klf()
    plot_klf2()
    plot_klf_2radii()
    plot_klf_3radii()
    plot_klf_vs_bartko()


def load_yng_data_by_radius(rmin=0, rmax=30, magCut=None, trimWR=False):
    """
    Load up the list of young star

    magnitudes
    magnitude errors
    probability(young)

    This is generated in the spec_completeness_by_radius()
    function and is used by the bayesian analysis.

    Note magnitudes are extinction corrected.
    """
    tmp = load_yng_catalog_by_radius(rmin=rmin, rmax=rmax, magCut=magCut, trimWR=trimWR)

    # Note extinction corrected magnitude.
    yng = objects.DataHolder()
    yng.kp = tmp.kp_ext      
    yng.kp_err = tmp.kp_err
    yng.prob = tmp.prob
    yng.N_WR = tmp.N_WR

    return yng

def load_yng_catalog_by_radius(rmin=0, rmax=30, magCut=None, trimWR=False):
    yng = objects.DataHolder()

    outRoot = '%sspec_completeness_r_%.1f_%.1f' % (workDir+'completeness/', rmin, rmax)

    _pick = open(outRoot + '_yng_info.pickle', 'rb')
    yng.name = pickle.load(_pick)
    yng.x = pickle.load(_pick)
    yng.y = pickle.load(_pick)
    yng.kp = pickle.load(_pick)
    yng.kp_ext = pickle.load(_pick)
    yng.kp_err = pickle.load(_pick)
    yng.isWR = pickle.load(_pick)
    yng.prob = pickle.load(_pick)
    yng.prob_err = pickle.load(_pick)
    _pick.close()

    foo = np.where(yng.isWR == True)[0]
    yng.N_WR = len(foo)

    # Optional magnitude cut
    if magCut != None:
        print(('Cutting out young stars with Kp <= %.2f' % magCut))
        idx = np.where(yng.kp_ext <= magCut)[0]
        yng.name = yng.name[idx]
        yng.x = yng.x[idx]
        yng.y = yng.y[idx]
        yng.kp = yng.kp[idx]
        yng.kp_ext = yng.kp_ext[idx]
        yng.kp_err = yng.kp_err[idx]
        yng.prob = yng.prob[idx]
        yng.prob_err = yng.prob_err[idx]
        yng.isWR = yng.isWR[idx]

    if trimWR == True:
        print('Cutting out WR stars')
        idx = np.where(yng.isWR == False)[0]
        yng.name = yng.name[idx]
        yng.x = yng.x[idx]
        yng.y = yng.y[idx]
        yng.kp = yng.kp[idx]
        yng.kp_ext = yng.kp_ext[idx]
        yng.kp_err = yng.kp_err[idx]
        yng.prob = yng.prob[idx]
        yng.prob_err = yng.prob_err[idx]
        yng.isWR = yng.isWR[idx]
        
    return yng

def load_all_catalog_by_radius(rmin=0, rmax=30, magCut=None):
    all = objects.DataHolder()

    outRoot = '%sspec_completeness_r_%.1f_%.1f' % (workDir+'completeness/', rmin, rmax)

    _pick = open(outRoot + '_all_info.pickle', 'rb')
    all.name = pickle.load(_pick)
    all.x = pickle.load(_pick)
    all.y = pickle.load(_pick)
    all.kp = pickle.load(_pick)
    all.kp_ext = pickle.load(_pick)
    all.kp_err = pickle.load(_pick)
    all.isWR = pickle.load(_pick)
    all.prob = pickle.load(_pick)
    all.prob_err = pickle.load(_pick)
    _pick.close()

    foo = np.where(all.isWR == True)[0]
    all.N_WR = len(foo)

    # Optional magnitude cut
    if magCut != None:
        print(('Cutting out young stars with Kp <= %.2f' % magCut))
        idx = np.where(all.kp_ext <= magCut)[0]
        all.name = all.name[idx]
        all.x = all.x[idx]
        all.y = all.y[idx]
        all.kp = all.kp[idx]
        all.kp_ext = all.kp_ext[idx]
        all.kp_err = all.kp_err[idx]
        all.prob = all.prob[idx]
        all.prob_err = all.prob_err[idx]
        all.isWR = all.isWR[idx]

    return all

def load_image_completeness_by_radius(rmin=0, rmax=30):
    d = objects.DataHolder()

    img_file = '%simage_completeness_r_%.1f_%.1f.txt' % (workDir_old, rmin, rmax)
    _img = asciidata.open(img_file)

    d.mag = _img[0].tonumpy()
    d.comp_no_ext = _img[1].tonumpy()
    d.comp = _img[4].tonumpy()

    return d

def gcimf_completeness():
    """
    Make a completeness curve for every osiris spectroscopic field. Use
    the completeness analysis from mag06maylgs1_dp_msc.
    """
    # Keep a mapping of which OSIRIS field corresponds to which
    # NIRC2 field in the dp_msc.
    osiris2nirc2 = {'GC Central': 'C',
                    'GC East': 'C',
                    'GC South': 'C',
                    'GC Southeast': 'C', 
                    'GC Southwest': 'C',
                    'GC West': 'C',
                    'GC Northwest': 'C',
                    'GC North': 'C',
                    'GC Northeast': 'C',
                    'E2-1': 'C_NE',
                    'E2-2': 'C_SE',
                    'E2-3': 'C_SE',
                    'E3-1': 'E',
                    'E3-2': 'E',
                    'E3-3': 'E',
                    'E4-1': 'E',
                    'E4-2': 'E',
                    'E4-3': 'E',
                    'S2-1': 'C'}

    cooStarDict = {'C': 'irs16C',
                   'C_SW': 'irs2',
                   'C_NE': 'irs16C',
                   'C_SE': 'irs33E',
                   'C_NW': 'irs34W',
                   'NE': 'S11-6',
                   'SE': 'S12-2',
                   'NW': 'S8-3',
                   'SW': 'S13-61',
                   'E': 'S10-1',
                   'W': 'S5-69',
                   'N': 'S10-3',
                   'S': 'irs14SW'}

    # Load up the label.dat file and use the coordinates in there
    # to astrometrically calibrate each NIRC2 dp_msc field.
    label = starTables.Labels('')

    # Load up the spectroscopic database to get the field-of-view definitions.
    fields = getOsirisFields()

    # Completeness Directory
    compDir = '/u/jlu/work/gc/dp_msc/2010_11_08/completeness'

    # NIRC2 Data directory
    nirc2Dir = '/u/ghezgroup/data/gc/06maylgs1/combo'

    for rr in range(len(fields.name)):
        fieldName = fields.name[rr]
        print(('Working on field %s' % fields.name[rr]))

        # Load up the corresponding NIRC2 completeness starlists
        nircFieldName = osiris2nirc2[fieldName]
        alignDir = '%s/%s/kp/align_in_out/' % \
            (compDir, nircFieldName)

        cData = starPlant.load_completeness(alignDir)
        
        # Load up information on the coo star for this field
        cooFile = '%s/mag06maylgs1_dp_msc_%s_kp.coo' % (nirc2Dir, nircFieldName)
        cooTmp = open(cooFile).readline().split()
        cooPix = [float(cooTmp[0]), float(cooTmp[1])]
        
        idx = np.where(label.name == cooStarDict[nircFieldName])[0][0]
        cooAsec = [label.x[idx], label.y[idx]]

        scale = 0.00995
        
        # Convert the completeness results to absolute coordinates
        cData.x_in = ((cData.x_in - cooPix[0]) * scale * -1) + cooAsec[0]
        cData.y_in = ((cData.y_in - cooPix[1]) * scale) + cooAsec[1]
        cData.x_out = ((cData.x_out - cooPix[0]) * scale * -1) + cooAsec[0]
        cData.y_out = ((cData.y_out - cooPix[1]) * scale) + cooAsec[1]
        cData.x = ((cData.x - cooPix[0]) * scale * -1) + cooAsec[0]
        cData.y = ((cData.y - cooPix[0]) * scale * -1) + cooAsec[0]

        # Now trim down to just those pixels that are within this
        # OSIRIS field of view
        xypoints = np.column_stack((cData.x_in, cData.y_in))
        fov_path = path.Path(xypoints, closed=True)
        inside = fov_path.contains_points(xypoints)
        # inside = nxutils.points_inside_poly(xypoints, fields.xyverts[rr])
        inside = np.where(inside == True)[0]

        # Measure the completeness for this OSIRIS field. Also print it 
        # out into a file.
        outRoot = '%s/completeness_%s' % (workDir_old, fieldName.replace(' ', ''))
        completeness = np.zeros(len(cData.mag), dtype=float)
        _comp = open(outRoot + '.dat', 'w')
        for mm in range(len(cData.mag)):
            # All planted stars in this field
            planted = np.where(cData.m_in[inside] == cData.mag[mm])[0]

            # All detected stars in this field
            detected = np.where(cData.f_out[inside][planted] > 0)[0]

            completeness[mm] = float(len(detected)) / float(len(planted))

            _comp.write('%5.2f  %6.3f  %3d  %3d\n' % 
                        (cData.mag[mm], completeness[mm], 
                         len(planted), len(detected)))

        _comp.close()

        py.clf()
        py.plot(cData.mag, completeness, 'k.-')
        py.xlabel('Kp Magnitude')
        py.ylabel('Completeness')
        py.title('OSIRIS Field %s, NIRC2 Field %s' % (fieldName, nircFieldName))
        py.ylim(0, 1.1)
        py.xlim(9, 20)
        py.savefig(outRoot + '.png')
        print(('Saving %s' % (outRoot + '.png')))

def img_completeness():
    # Load up the completeness mosaic. Remember, it is aligned to the
    # mag06maylgs1_dp_msc_C_kp image coordinates and IRS 16C is the coo star.
    compDir = '/u/jlu/work/gc/dp_msc/2010_11_08/completeness/'
    cData = load_completeness_mosaic(compDir)

    x_in = cData.x_in
    y_in = cData.y_in
    m_in = cData.m_in

    x_out = cData.x_out
    y_out = cData.y_out
    m_out = cData.m_out

    # Load up the spectroscopic database to get the field-of-view definitions.
    fields.getOsirisFields()

    magStep = 0.25
    magBins = np.arange(8, 20, magStep)

    # Also keep track of the integrated completeness
    planted_all = np.zeros(len(magBins), dtype=float)
    detected_all = np.zeros(len(magBins), dtype=float)
    
    # Loop through each OSIRIS field and get the completeness curve.
    for rr in range(len(fields.name)):
        fieldName = fields.name[rr]
        print(('Working on field %s' % fieldName))

        # Now trim down to just those pixels that are within this
        # OSIRIS field of view
        xypoints = np.column_stack((x_in, y_in))
        inside = nxutils.points_inside_poly(xypoints, fields.xyverts[rr])
        inside = np.where(inside == True)[0]

        # Measure the completeness for this OSIRIS field. Also print it 
        # out into a file.
        outRoot = '%s/img_completeness_%s' % \
            (workDir_old, fieldName.replace(' ', ''))
        completeness = np.zeros(len(magBins), dtype=float)
        _comp = open(outRoot + '.dat', 'w')

        for mm in range(len(magBins)):
            # All planted stars in this field
            planted = np.where((m_in[inside] >= magBins[mm]) & 
                               (m_in[inside] < (magBins[mm]+magStep)))[0]

            # All detected stars in this field
            detected = np.where(np.isnan(m_out[inside][planted]) == False)[0]

            if len(planted) != 0:
                completeness[mm] = float(len(detected)) / float(len(planted))

                planted_all[mm] += len(planted)
                detected_all[mm] += len(detected)
            else:
                completeness[mm] = np.NAN
                
            _comp.write('%5.2f  %6.3f  %3d  %3d\n' % 
                        (magBins[mm], completeness[mm], 
                         len(planted), len(detected)))

        _comp.close()

        # Get rid of the NAN for plotting
        ok = np.isnan(completeness) == False

        py.clf()
        py.plot(magBins[ok], completeness[ok], 'k.-')
        py.xlabel('Kp Magnitude')
        py.ylabel('Completeness')
        py.title('OSIRIS Field %s' % (fieldName))
        py.ylim(0, 1.1)
        py.xlim(9, 20)
        py.savefig(outRoot + '.png')
        print(('Saving %s' % (outRoot + '.png')))

    # Write up the total completeness
    outRoot = '%s/img_completeness_all' % (workDir_old)
    completeness = detected_all / planted_all
    _comp = open(outRoot + '.dat', 'w')

    for mm in range(len(magBins)): 
        _comp.write('%5.2f  %6.3f  %3d  %3d\n' % 
                    (magBins[mm], completeness[mm], 
                     planted_all[mm], detected_all[mm]))
        
    _comp.close()

    py.clf()
    py.plot(magBins[ok], completeness[ok], 'k.-')
    py.xlabel('Kp Magnitude')
    py.ylabel('Completeness')
    py.title('All OSIRIS Fields')
    py.ylim(0, 1.1)
    py.xlim(9, 20)
    py.savefig(outRoot + '.png')
    print(('Saving %s' % (outRoot + '.png')))


def compare_img_completeness_methods():
    """
    Compare the two versions of imaging completeness curves:
    (1) using individual NIRC2 dp_msc fields
    (2) using combined NIRC2 dp_msc
    """
    # Load up the spectroscopic database to get the field-of-view definitions.
    fields = getOsirisFields()

    # Loop through each OSIRIS field and get the completeness curve.
    for rr in range(len(fields.name)):
        fieldName = fields.name[rr]
        print(('Working on field %s' % fieldName))

        compFile1 = '%s/completeness_%s.dat' % \
            (workDir_old, fieldName.replace(' ', ''))
        compFile2 = '%s/img_completeness_%s.dat' % \
            (workDir_old, fieldName.replace(' ', ''))

        table1 = asciidata.open(compFile1)
        table2 = asciidata.open(compFile2)

        mag1 = table1[0].tonumpy()
        mag2 = table2[0].tonumpy()
        comp1 = table1[1].tonumpy()
        comp2 = table2[1].tonumpy()

        py.clf()
        py.plot(mag1, comp1, 'r-', label='Single NIRC2')
        py.plot(mag2, comp2, 'b-', label='Mosaic NIRC2')
        py.xlabel('Kp Magnitude')
        py.ylabel('Completeness')
        py.title('OSIRIS Field %s' % (fieldName))
        py.legend()
        py.ylim(0, 1.1)
        py.xlim(9, 20)
        py.savefig('%simg_completeness_compare_%s.png' % \
                       (workDir_old, fieldName.replace(' ', '')))

    
def compare_img_completeness_submaps():
    """
    Compare completeness curves for the star planting analysis with
    and without submap selection.
    """
    oldDir = '/u/jlu/work/gc/dp_msc/2010_09_17/completeness/'
    newDir = '/u/jlu/work/gc/dp_msc/2010_11_08/completeness/'

    oldAlign = '/kp/06maylgs1/align/'
    newAlign = '/kp/align_in_out/'
    
    fields = ['C', 'C_NE', 'C_SE', 'E']

    outDir = '/u/jlu/work/gc/dp_msc/2010_11_08/completeness/compare_with_old/'

    for ff in range(len(fields)):
        oldFile = oldDir + fields[ff] + oldAlign + 'completeness.dat'
        newFile = newDir + fields[ff] + newAlign + 'completeness.dat'

        old = asciidata.open(oldFile)
        new = asciidata.open(newFile)

        py.clf()
        py.plot(old[0].tonumpy(), old[1].tonumpy(), 'r-',
                label='No Submap Selection')
        py.plot(new[0].tonumpy(), new[1].tonumpy(), 'b-',
                label='With Submap Selection')
        py.xlabel('Kp Magnitude')
        py.ylabel('Completeness')
        py.title('NIRC2 Field %s' % (fields[ff]))
        py.legend(loc='lower left')
        py.ylim(0, 1.1)
        py.xlim(9, 20)
        py.savefig(outDir + 'compare_completeness_' + fields[ff] + '.png')

def calc_spec_id_all_stars():
    """
    Tabulate up all of the young, old, and NIRC2 stars in a matched
    starlist. The photometry for all the stars is extracted from the
    dp_msc starlist. Extinction corrections are pulled for all the stars
    from the Schoedel et al. (2010) extinction map.
    """
    yng_age = 5e6
    old_age = 2e9

    yng_temp = 30000
    old_temp = 4500
    
    # Load up the names of known Wolf-Rayet stars.
    wolfRayetNames = get_wolf_rayet_stars()

    # Use an align of all the dp_msc starlists to 
    # estimate magnitudes.
    dp_msc_root = '/u/jlu/work/gc/dp_msc/2011_05_29/tables/'
    dp_msc_root += 'photo_catalog_dp_msc.dat'

    dp_msc = Table.read(dp_msc_root, format='ascii')
    nirc2 = objects.DataHolder()
    nirc2.name = np.array(dp_msc['Name'])
    nirc2.x = np.array(dp_msc['Xarc'])
    nirc2.y = np.array(dp_msc['Yarc'])
    nirc2.kp = np.array(dp_msc['Kp'])
    nirc2.kperr = np.array(dp_msc['Kperr'])

    # Impose magnitude error lower limit of 1\%
    idx = np.where(nirc2.kperr < 0.02)[0]
    nirc2.kperr[idx] = 0.02

    # Select all the OSIRIS fields-of-view
    fields = getOsirisFields()

    # Load up young and old stars
    yng = load_young()
    old = load_old()

    # Load up Tuan's completeness simulations for the unknown stars.
    specSims = load_unknown_sims()

    # Load up extinction values from the Schoedel extinction map
    get_extinctions_for_stars(nirc2)

    # Define some output arrays
    yng.kp_ext = np.zeros(len(yng.x), dtype=float)
    yng.kperr = np.zeros(len(yng.x), dtype=float)
    old.kp_ext = np.zeros(len(old.x), dtype=float)
    old.kperr = np.zeros(len(old.x), dtype=float)
    nirc2.kp_ext = np.zeros(len(nirc2.x), dtype=float)

    # Loop through young stars and update coords and brightness
    # to the NIRC2 values.
    update_star_by_name(nirc2, yng, comment='young')

    # Loop through old stars and update coords and brightness
    # to the NIRC2 values.
    update_star_by_name(nirc2, old, comment='old')

    # Will de-redden all stars to AKs = 2.7. Then these can be converted
    # using the ks_2_kp factor (assumes hot star atmospheres)
    ks_2_kp_yng = get_Kp_Ks(yng_age, theAKs, distance, yng_temp)
    ks_2_kp_old = get_Kp_Ks(old_age, theAKs, distance, old_temp)

    # Keep track of everything in the end
    kp_yng = []
    kp_old = []
    kp_nirc2 = []
    kperr_yng = []
    kperr_old = []
    kperr_nirc2 = []
    kp_ext_yng = []
    kp_ext_old = []
    kp_ext_nirc2 = []
    name_yng = []
    name_old = []
    name_nirc2 = []
    x_yng = []
    x_old = []
    x_nirc2 = []
    y_yng = []
    y_old = []
    y_nirc2 = []
    isWR_yng = []
    field_yng = []
    field_old = []
    field_nirc2 = []

    xypoints_yng = np.column_stack((yng.x, yng.y))
    xypoints_old = np.column_stack((old.x, old.y))
    xypoints_nirc2 = np.column_stack((nirc2.x, nirc2.y))

    def is_name_in_list(name, name_list_list):
        for ff2 in range(len(name_list_list)):
            for ii2 in range(len(name_list_list[ff2])):
                if name_list_list[ff2][ii2] == name:
                    return (True, ff2)
        return (False, -1)

    # Loop through each OSIRIS field
    for ff in range(len(fields.name)):
        fieldName = fields.name[ff]
        fieldSuffix = fieldName.replace(' ', '')
        print(('Working on field %s' % fieldName))

        # Find the stars inside this field.
        path = matplotlib.path.Path(fields.xyverts[ff])
        inside_yng = np.arange(len(xypoints_yng))[path.contains_points(xypoints_yng)]
        inside_old = np.arange(len(xypoints_old))[path.contains_points(xypoints_old)]
        inside_nirc2 = np.arange(len(xypoints_nirc2))[path.contains_points(xypoints_nirc2)]
        inside_nirc2_orig = inside_nirc2.copy()

        isWR_yng_field = []
        field_yng_tmp = []

        # Young Stars
        for ii in inside_yng:
            # Stars can only be claimed by ONE field. Loop through all
            # previous star names and make sure this star hasn't been
            # identified already.
            isNameInList, inField = is_name_in_list(yng.name[ii], name_yng)
            if isNameInList:
                print('Skip %s in field %s (yng star found in field %s).' % \
                    (yng.name[ii], fieldName, fields.name[inField]))

                # Remove it from our inside_yng list.
                inside_yng = inside_yng[inside_yng != ii]
                continue
            
            # Find the star in the deep mosaic
            ndx = np.where(nirc2.name[inside_nirc2] == yng.name[ii])[0]
            if len(ndx) == 0:
                # Small coordinate differences can put a star "in" 
                # in OSIRIS and "out" in the dp_msc list. Just include
                # them in this case.
                idx = np.where(nirc2.name == yng.name[ii])[0]
                nn = idx[0]
                print(('Confused about field of star: %s' % yng.name[ii]))
            else:
                nn = inside_nirc2[ndx[0]]
                inside_nirc2 = np.delete(inside_nirc2, ndx[0])


            #if np.abs(nirc2.AKs[nn] - yng.AKs[ii]) > 0.1:
            #    print("problem!!!")
            
            # Calculate extinction corrected photometry

            kp_2_ks_yng = get_Kp_Ks(yng_age, nirc2.AKs[nn], distance, yng_temp) 
            nirc2.kp_ext[nn] = nirc2.kp[nn] - kp_2_ks_yng - nirc2.AKs[nn] \
                + theAKs + ks_2_kp_yng

            # Use the photometry from the deep mosaic
            yng.kp[ii] = nirc2.kp[nn]
            yng.kperr[ii] = nirc2.kperr[nn]
            yng.kp_ext[ii] = nirc2.kp_ext[nn]

            wrIdx = np.where(wolfRayetNames == yng.name[ii])[0]
            if len(wrIdx) > 0:
                isWR_yng_field.append(True)
                print((yng.name[ii], ' is a Wolf-Rayet Star'))
            else:
                isWR_yng_field.append(False)

            field_yng_tmp.append(fieldName)

        isWR_yng_field = np.array(isWR_yng_field)

        # Old Stars
        field_old_tmp = []
        for ii in inside_old:
            # Stars can only be claimed by ONE field. Loop through all
            # previous star names and make sure this star hasn't been
            # identified already.
            isNameInList, inField = is_name_in_list(old.name[ii], name_old)
            if isNameInList:
                print('Skip %s in field %s (old star found in field %s).' % \
                    (old.name[ii], fieldName, fields.name[inField]))

                # Remove it from our inside_old list.
                inside_old = inside_old[inside_old != ii]
                continue

            # Find the star in the deep mosaic
            ndx = np.where(nirc2.name[inside_nirc2] == old.name[ii])[0]
            if len(ndx) == 0:
                # Small coordinate differences can put a star "in" 
                # in OSIRIS and "out" in the dp_msc list. Just include
                # them in this case.
                idx = np.where(nirc2.name == old.name[ii])[0]
                nn = idx[0]
                print(('Confused about field of star: %s' % old.name[ii]))
            else:
                nn = inside_nirc2[ndx[0]]
                inside_nirc2 = np.delete(inside_nirc2, ndx[0])
            
            # Calculate extinction corrected photometry
            kp_2_ks_old = get_Kp_Ks(old_age, nirc2.AKs[nn], distance, old_temp) 

            nirc2.kp_ext[nn] = nirc2.kp[nn] - kp_2_ks_old - nirc2.AKs[nn] \
                + theAKs + ks_2_kp_old

            # Use the photometry from the deep mosaic
            old.kp[ii] = nirc2.kp[nn]
            old.kperr[ii] = nirc2.kperr[nn]
            old.kp_ext[ii] = nirc2.kp_ext[nn]

            field_old_tmp.append(fieldName)

        # Un-typed stars
        field_nirc2_tmp = []
        py.clf()
        fov = matplotlib.patches.Polygon(fields.xyverts[ff], fill=False)
        py.gca().add_patch(fov)
        py.title(fields.name[ff])

        for ii in inside_nirc2:
            # Stars can only be claimed by ONE field. Loop through all
            # previous star names and make sure this star hasn't been
            # identified already.
            isNameInList, inField = is_name_in_list(nirc2.name[ii], name_nirc2)
            if isNameInList:
                print('Skip %s in field %s (nirc2 star found in field %s).' % \
                    (nirc2.name[ii], fieldName, fields.name[inField]))

                # Remove it from our inside_nirc2_orig list.
                inside_nirc2_orig = inside_nirc2_orig[inside_nirc2_orig != ii]

                continue

            # Calculate extinction corrected photometry
            kp_2_ks_old = get_Kp_Ks(old_age, nirc2.AKs[nn], distance, old_temp) 

            nirc2.kp_ext[ii] = nirc2.kp[ii] - kp_2_ks_old - nirc2.AKs[ii] \
                + theAKs + ks_2_kp_old

            # Check to see if we have bright stars that aren't reported in
            # Tuan's star planting experiment. If this is the case, then we
            # probably have a field of view issue. Just drop these stars for now.
            # **** TODO **** Remove this and figure out the root issue.
            if (nirc2.kp_ext[ii] <= 15.5) and (nirc2.name[ii] not in specSims['name']):
                print(('Removing bright un-typed %s in field %s' % \
                    (nirc2.name[ii], fieldName)))
                print(('    Kp = %5.2f (%5.2f ext-corr)  x = %8.3f  y = %8.3f' % \
                    (nirc2.kp[ii], nirc2.kp_ext[ii], nirc2.x[ii], nirc2.y[ii])))

                # Remove it from our inside_nirc2_orig list.
                inside_nirc2_orig = inside_nirc2_orig[inside_nirc2_orig != ii]

                py.plot(nirc2.x[ii], nirc2.y[ii], 'ko')
                py.text(nirc2.x[ii], nirc2.y[ii], nirc2.name[ii])


        py.xlabel('X Offset (arcsec)')
        py.ylabel('Y Offset (arcsec)')
        py.title(fieldName)
        py.savefig(workDir + 'missing_stars_' + fieldName + '.png')
        
        for ii in inside_nirc2_orig:
            field_nirc2_tmp.append(fieldName)

        kp_yng.append( yng.kp[inside_yng] )
        kperr_yng.append( yng.kperr[inside_yng] )
        kp_ext_yng.append( yng.kp_ext[inside_yng] )
        kp_old.append( old.kp[inside_old] )
        kperr_old.append( old.kperr[inside_old] )
        kp_ext_old.append( old.kp_ext[inside_old] )
        kp_nirc2.append( nirc2.kp[inside_nirc2_orig] )
        kperr_nirc2.append( nirc2.kperr[inside_nirc2_orig] )
        kp_ext_nirc2.append( nirc2.kp_ext[inside_nirc2_orig] )
        name_yng.append( yng.name[inside_yng] )
        name_old.append( old.name[inside_old] )
        name_nirc2.append( nirc2.name[inside_nirc2_orig] )
        x_yng.append( yng.x[inside_yng] )
        x_old.append( old.x[inside_old] )
        x_nirc2.append( nirc2.x[inside_nirc2_orig] )
        y_yng.append( yng.y[inside_yng] )
        y_old.append( old.y[inside_old] )
        y_nirc2.append( nirc2.y[inside_nirc2_orig] )
        isWR_yng.append(isWR_yng_field)
        field_yng.append(field_yng_tmp)
        field_old.append(field_old_tmp)
        field_nirc2.append(field_nirc2_tmp)
        
    _out = open(workDir + 'spec_id_all_stars.dat', 'wb')
    pickle.dump(fields.name, _out)
    pickle.dump(kp_yng, _out)
    pickle.dump(kp_old, _out)
    pickle.dump(kp_nirc2, _out)
    pickle.dump(kperr_yng, _out)
    pickle.dump(kperr_old, _out)
    pickle.dump(kperr_nirc2, _out)
    pickle.dump(kp_ext_yng, _out)
    pickle.dump(kp_ext_old, _out)
    pickle.dump(kp_ext_nirc2, _out)
    pickle.dump(name_yng, _out)
    pickle.dump(name_old, _out)
    pickle.dump(name_nirc2, _out)
    pickle.dump(x_yng, _out)
    pickle.dump(x_old, _out)
    pickle.dump(x_nirc2, _out)
    pickle.dump(y_yng, _out)
    pickle.dump(y_old, _out)
    pickle.dump(y_nirc2, _out)
    pickle.dump(isWR_yng, _out)
    pickle.dump(field_yng, _out)
    pickle.dump(field_old, _out)
    pickle.dump(field_nirc2, _out)
    _out.close()

def load_spec_id_all_stars(flatten=False):
    _in = open(workDir + 'spec_id_all_stars.dat', 'rb')

    d = objects.DataHolder()
    d.fields = pickle.load(_in)
    d.kp_yng = pickle.load(_in)
    d.kp_old = pickle.load(_in)
    d.kp_nirc2 = pickle.load(_in)
    d.kperr_yng = pickle.load(_in)
    d.kperr_old = pickle.load(_in)
    d.kperr_nirc2 = pickle.load(_in)
    d.kp_ext_yng = pickle.load(_in)
    d.kp_ext_old = pickle.load(_in)
    d.kp_ext_nirc2 = pickle.load(_in)
    d.name_yng = pickle.load(_in)
    d.name_old = pickle.load(_in)
    d.name_nirc2 = pickle.load(_in)
    d.x_yng = pickle.load(_in)
    d.x_old = pickle.load(_in)
    d.x_nirc2 = pickle.load(_in)
    d.y_yng = pickle.load(_in)
    d.y_old = pickle.load(_in)
    d.y_nirc2 = pickle.load(_in)
    d.isWR_yng = pickle.load(_in)
    d.field_yng = pickle.load(_in)
    d.field_old = pickle.load(_in)
    d.field_nirc2 = pickle.load(_in)

    if flatten:
        # Flatten the arrays so that they aren't broken up by field
        d.kp_yng = np.array(list(itertools.chain(*d.kp_yng)))
        d.kp_old = np.array(list(itertools.chain(*d.kp_old)))
        d.kp_nirc2 = np.array(list(itertools.chain(*d.kp_nirc2)))
        d.kperr_yng = np.array(list(itertools.chain(*d.kperr_yng)))
        d.kperr_old = np.array(list(itertools.chain(*d.kperr_old)))
        d.kperr_nirc2 = np.array(list(itertools.chain(*d.kperr_nirc2)))
        d.kp_ext_yng = np.array(list(itertools.chain(*d.kp_ext_yng)))
        d.kp_ext_old = np.array(list(itertools.chain(*d.kp_ext_old)))
        d.kp_ext_nirc2 = np.array(list(itertools.chain(*d.kp_ext_nirc2)))
        d.name_yng = np.array(list(itertools.chain(*d.name_yng)))
        d.name_old = np.array(list(itertools.chain(*d.name_old)))
        d.name_nirc2 = np.array(list(itertools.chain(*d.name_nirc2)))
        d.x_yng = np.array(list(itertools.chain(*d.x_yng)))
        d.y_yng = np.array(list(itertools.chain(*d.y_yng)))
        d.x_old = np.array(list(itertools.chain(*d.x_old)))
        d.y_old = np.array(list(itertools.chain(*d.y_old)))
        d.x_nirc2 = np.array(list(itertools.chain(*d.x_nirc2)))
        d.y_nirc2 = np.array(list(itertools.chain(*d.y_nirc2)))
        d.isWR_yng = np.array(list(itertools.chain(*d.isWR_yng)))
        d.field_yng = np.array(list(itertools.chain(*d.field_yng)))
        d.field_old = np.array(list(itertools.chain(*d.field_old)))
        d.field_nirc2 = np.array(list(itertools.chain(*d.field_nirc2)))

    return d

def update_star_by_name(fromList, toList, comment=''):
    """
    Update the starlist <toList> with the photometry and astrometry
    from the <fromList> starlist. The update is done by matching names.
    The updated arrays include: kp, kperr, x, y
    """
    for ii in range(len(toList.x)):
        # Match each star by name
        idx = np.where(toList.name[ii] == fromList.name)[0]

        if len(idx) == 0:
            print(('Could not find %s star in NIRC2: %s' % \
                (comment, toList.name[ii])))
            print(('    Kp = %5.2f  x = %8.3f  y = %8.3f' % \
                (toList.kp[ii], toList.x[ii], toList.y[ii])))
        else:
            toList.x[ii] = fromList.x[idx]
            toList.y[ii] = fromList.y[idx]
            toList.kp[ii] = fromList.kp[idx]
            toList.kperr[ii] = fromList.kperr[idx]
            

def our_yng_catalog():
    # Select all the OSIRIS fields-of-view
    fields = getOsirisFields()

    d = load_spec_id_all_stars()
    
    name = []
    kp = []
    kperr = []
    kp_ext = []
    x = []
    y = []
    isWR = []
    field = []

    for ff in range(len(fields.name)):
        for ii in range(len(d.x_yng[ff])):
            name.append(d.name_yng[ff][ii])
            kp.append(d.kp_yng[ff][ii])
            kperr.append(d.kperr_yng[ff][ii])
            kp_ext.append(d.kp_ext_yng[ff][ii])
            x.append(d.x_yng[ff][ii])
            y.append(d.y_yng[ff][ii])
            isWR.append(d.isWR_yng[ff][ii])
            field.append(fields.name[ff])

    name = np.array(name)
    kp = np.array(kp)
    kperr = np.array(kperr)
    kp_ext = np.array(kp_ext)
    x = np.array(x)
    y = np.array(y)
    isWR = np.array(isWR)
    field = np.array(field)

    # Sort
    sdx = name.argsort()
    name = name[sdx]
    kp = kp[sdx]
    kperr = kperr[sdx]
    kp_ext = kp_ext[sdx]
    x = x[sdx]
    y = y[sdx]
    isWR = isWR[sdx]
    field = field[sdx]

    print(('%-13s  %5s  %5s  %5s  %7s  %7s  %5s  %13s' % \
        ('Name', 'Kp', 'Kperr', 'Kpext', 'X', 'Y', 'isWR', 'field')))

    for ii in range(len(name)):
        print(('%-13s  %5.2f  %5.2f  %5.2f  %7.3f  %7.3f  %5d  %13s' % \
            (name[ii], kp[ii], kperr[ii], kp_ext[ii], x[ii], y[ii],
             isWR[ii], field[ii])))

    isWR = np.array(isWR)

    print(('Total number of young stars: %d (total unique: %d)' % \
        (len(name), len(np.unique(name)))))
    print(('    %d WR stars and %d non-WR stars' % \
        (isWR.sum(), len(isWR) - isWR.sum())))

def our_osiris_catalog():
    # Select all the OSIRIS fields-of-view
    fields = getOsirisFields()

    d = load_spec_id_all_stars(flatten=True)

    _out = open(workDir + 'osiris_catalog.txt', 'w')

    def print_sample(name, kp, kpext, kperr, x, y, isWR, field, type, printHead=False):
        sdx = name.argsort()
        name = name[sdx]
        kp = kp[sdx]
        kpext = kpext[sdx]
        kperr = kperr[sdx]
        x = x[sdx]
        y = y[sdx]
        field = field[sdx]
        if isWR != None:
            isWR = isWR[sdx]

        if printHead:
            _out.write('%-13s  %6s  %6s  %5s  %7s  %7s  %-5s  %-8s  %-13s\n' % \
                ('Name', 'Kp_obs', 'Kp_ext', 'Kperr', 'X', 'Y', 'isWR', 'type', 'field'))
                
        for ii in range(len(name)):
            _out.write('%-13s  %6.2f  %6.2f  %5.2f  %7.3f  %7.3f  ' %
                       (name[ii], kp[ii], kpext[ii], kperr[ii], x[ii], y[ii]))

            if (isWR != None) and (isWR[ii] == True):
                _out.write('%-5s  ' % 'True')
            else:
                _out.write('%-5s  ' % 'False')

            _out.write('%-8s  %-13s\n' % (type, field[ii].replace(' ', '')))
                   
        
    print_sample(d.name_yng, d.kp_yng, d.kp_ext_yng, d.kperr_yng,
                  d.x_yng, d.y_yng, d.isWR_yng, d.field_yng, 'early', printHead=True)
    print_sample(d.name_old, d.kp_old, d.kp_ext_old, d.kperr_old,
                  d.x_old, d.y_old, None, d.field_old, 'late')
    print_sample(d.name_nirc2, d.kp_nirc2, d.kp_ext_nirc2, d.kperr_nirc2,
                  d.x_nirc2, d.y_nirc2, None, d.field_nirc2, 'unknown')

    _out.close()

    
def spec_completeness():
    # Select all the OSIRIS fields-of-view
    fields = getOsirisFields()

    # Load spectral identifications
    s = load_spec_id_all_stars()

    # Now make completeness curves
    magStep = klf_mag_bins[1] - klf_mag_bins[0]
    magHalfStep = magStep / 2.0
    magBins = klf_mag_bins

    Nfields = len(fields.name)
    Nbins = len(magBins)

    cnt_yng = np.zeros((Nfields, Nbins), dtype=int)
    cnt_old = np.zeros((Nfields, Nbins), dtype=int)
    cnt_nirc2 = np.zeros((Nfields, Nbins), dtype=int)
    cnt_ext_yng = np.zeros((Nfields, Nbins), dtype=int)
    cnt_ext_old = np.zeros((Nfields, Nbins), dtype=int)
    cnt_ext_nirc2 = np.zeros((Nfields, Nbins), dtype=int)

    comp = np.zeros((Nfields, Nbins), dtype=float)
    comp_ext = np.zeros((Nfields, Nbins), dtype=float)
    comp_fix = np.zeros((Nfields, Nbins), dtype=float)
    comp_ext_fix = np.zeros((Nfields, Nbins), dtype=float)

    fieldSuffix = []

    for ff in range(Nfields):
        fieldSuffix.append( fields.name[ff].replace(' ', '') )

        for mm in range(Nbins):
            # Before extinction correction
            yy = np.where((s.kp_yng[ff] >= (magBins[mm]-magHalfStep)) & 
                          (s.kp_yng[ff] < (magBins[mm]+magHalfStep)))[0]
            oo = np.where((s.kp_old[ff] >= (magBins[mm]-magHalfStep)) & 
                          (s.kp_old[ff] < (magBins[mm]+magHalfStep)))[0]
            nn = np.where((s.kp_nirc2[ff] >= (magBins[mm]-magHalfStep)) & 
                          (s.kp_nirc2[ff] < (magBins[mm]+magHalfStep)))[0]

            cnt_yng[ff, mm] = len(yy)
            cnt_old[ff, mm] = len(oo)
            cnt_nirc2[ff, mm] = len(nn)

            if len(nn) == 0:
                comp[ff, mm] = float('nan')
            else:
                comp[ff, mm] = float(len(yy) + len(oo)) / float(len(nn))

            # After extinction correction
            yy = np.where((s.kp_yng[ff] >= (magBins[mm]-magHalfStep)) & 
                          (s.kp_yng[ff] < (magBins[mm]+magHalfStep)))[0]
            oo = np.where((s.kp_old[ff] >= (magBins[mm]-magHalfStep)) & 
                          (s.kp_old[ff] < (magBins[mm]+magHalfStep)))[0]
            nn = np.where((s.kp_nirc2[ff] >= (magBins[mm]-magHalfStep)) & 
                          (s.kp_nirc2[ff] < (magBins[mm]+magHalfStep)))[0]

            cnt_ext_yng[ff, mm] = len(yy)
            cnt_ext_old[ff, mm] = len(oo)
            cnt_ext_nirc2[ff, mm] = len(nn)

            if len(nn) == 0:
                comp_ext[ff, mm] = float('nan')
            else:
                cnt_id = len(yy) + len(oo)
                comp_ext[ff, mm] = float(cnt_id) / float(len(nn))
                
                if len(nn) < cnt_id:
                    comp_ext[ff, mm] = 1.0

        comp_fix[ff] = fix_osiris_spectral_comp(magBins, comp[ff])
        comp_ext_fix[ff] = fix_osiris_spectral_comp(magBins, comp_ext[ff])

        py.clf()
        py.plot(magBins, comp_ext_fix[ff], 'ks-', label='Ext. Correct, Fixed')
        py.plot(magBins, comp_ext[ff], 'ks--', label='Ext. Correct')
        py.plot(magBins, comp[ff], 'r^--', label='Raw')
        py.title(fields[ff][0])
        py.ylim(-0.05, 1.05)

    _pick = open(workDir + 'spec_completeness.pickle', 'w')
    pickle.dump(magBins, _pick)
    pickle.dump(np.array(fieldSuffix), _pick)
    pickle.dump(cnt_yng, _pick)
    pickle.dump(cnt_old, _pick)
    pickle.dump(cnt_nirc2, _pick)
    pickle.dump(cnt_ext_yng, _pick)
    pickle.dump(cnt_ext_old, _pick)
    pickle.dump(cnt_ext_nirc2, _pick)
    pickle.dump(comp, _pick)
    pickle.dump(comp_ext, _pick)
    pickle.dump(comp_fix, _pick)
    pickle.dump(comp_ext_fix, _pick)
    _pick.close()

    _out = open(workDir + 'spec_completeness_ext_jlu.txt', 'w')
    _out2 = open(workDir + 'spec_completeness_jlu.txt', 'w')

    # Header line
    _out.write('%-5s  ' % ('#mag'))
    _out2.write('%-5s  ' % ('#mag'))
    for ff in range(len(fieldSuffix)):
        _out.write('%11s  ' % (fieldSuffix[ff]))
        _out2.write('%11s  ' % (fieldSuffix[ff]))
    _out.write('\n')
    _out2.write('\n')

    # One line per magnitude bin
    for mm in range(Nbins):
        _out.write('%5.2f  ' % (magBins[mm]))
        _out2.write('%5.2f  ' % (magBins[mm]))

        for ff in range(len(fieldSuffix)):
            _out.write('%11.2f  ' % (comp_ext[ff, mm]))
            _out2.write('%11.2f  ' % (comp[ff, mm]))

        _out.write('\n')
        _out2.write('\n')
    _out.close()

def load_spec_completeness_pickle():
    _pick = open(workDir + 'spec_completeness.pickle')

    d = objects.DataHolder()

    d.magBins = pickle.load(_pick)
    d.fieldSuffix = pickle.load(_pick)
    d.cnt_yng = pickle.load(_pick)
    d.cnt_old = pickle.load(_pick)
    d.cnt_nirc2 = pickle.load(_pick)
    d.cnt_ext_yng = pickle.load(_pick)
    d.cnt_ext_old = pickle.load(_pick)
    d.cnt_ext_nirc2 = pickle.load(_pick)
    d.comp = pickle.load(_pick)
    d.comp_ext = pickle.load(_pick)
    d.comp_fix = pickle.load(_pick)
    d.comp_ext_fix = pickle.load(_pick)
    
    return d

def fix_osiris_spectral_comp(compKp, comp):
    """
    Take a completeness (comp) vs. magnitude (compKp) curve and clean
    it up for:
    -- repair bright magnitude bins with NaN to set completeness = 1
    -- repair faint magnitude bins with NaN to set completeness = 0
    -- interpolate over NaN values using adjacent (good) bins
    """
    compTmp = comp.copy()

    # We need to interpolate over empty stuff, but only where
    # completness is < 1. First, lets repair anything at the bright
    # end that has NaN if the first good bin = 1. Do the same
    # at the faint end.
    good = np.where(np.isfinite(compTmp) == True)[0]
    bad = np.where(np.isfinite(compTmp) == False)[0]
        
    # Find the lowest mag bin with a 1.0... all NaN before are set to 1.0
    idx = np.where(compTmp[good] == 1)[0]
    if len(idx) > 0:
        foo = np.where(bad < good[idx[-1]])[0]
        if len(foo) > 0:
            compTmp[bad[foo]] = 1.0
    if compTmp[good[-1]] == 0:
        compTmp[good[-1]+1:] = 0.0

    # Get the "bad" values and average the two adjacent bins
    bad = np.where(np.isfinite(compTmp) == False)[0]
    good = np.where(np.isfinite(compTmp) == True)[0]
    if len(bad) > 0:
        for bb in range(len(bad)):
            # Find the closest good point below and above our bad point.
            mdx1 = np.where(compKp[good] < compKp[bad[bb]])[0]
            mdx2 = np.where(compKp[good] > compKp[bad[bb]])[0]

            m1 = compKp[good][mdx1[-1]]
            m2 = compKp[good][mdx2[0]]
            c1 = compTmp[good][mdx1[-1]]
            c2 = compTmp[good][mdx2[0]]

            slope = (c2 - c1) / (m2 - m1)
            inter = c2 - slope * m2

            compTmp[bad[bb]] = inter + slope * compKp[bad[bb]]

            if compTmp[bad[bb]] < 0:
                compTmp[bad[bb]] = 0.0
            if compTmp[bad[bb]] > 1:
                compTmp[bad[bb]] = 1.0

    return compTmp

def load_young():
    # Create a connection to the database file and create a cursor
    pwFile='/g/lu/scratch/siyao/other/pw.txt'
    pw = open(pwFile).read().split()[0]
    con = mdb.connect(host='galaxy1.astro.ucla.edu',user='dbread',passwd=pw,db='gcg')
    
    # Get info on the stars
    rows = pd.read_sql_query('SELECT name, kp, x, y, AK_sch FROM stars WHERE young="T"', con)

    print(('Found %d young stars in database stars' % len(rows)))

    starName = []
    x = []
    y = []
    kp = []
    AKs = []
    starInField = []

    
    # Loop through each star and pull out the relevant information
    for ss in range(len(rows)):
        record = rows.iloc[ss]
        name = record['name']

        # Check that we observed this star and what field it was in.
        row = pd.read_sql_query('SELECT field FROM spectra WHERE name = "%s"' % (name), con)

        if not row.empty:
            starInField.append(row.iloc[0]['field'])

            starName.append(str(name))
            kp.append(record['kp'])
            x.append(record['x'])
            y.append(record['y'])
            AKs.append(record['AK_sch'])

            if AKs[-1] == None:
                print(('%-13s  %5.2f  %7.3f  %7.3f  None' % \
                    (starName[-1], kp[-1], x[-1], y[-1])))
            else:
                print(('%-13s  %5.2f  %7.3f  %7.3f  %4.2f' % \
                    (starName[-1], kp[-1], x[-1], y[-1], AKs[-1])))
        else:
            print(('We do not have data on this star???', name))
            #starInField.append('C')
            continue
    

    starCnt = len(starName)
    starName = np.array(starName)
    starInField = np.array(starInField)
    x = np.array(x)
    y = np.array(y)
    kp = np.array(kp)
    AKs = np.array(AKs)

    d = objects.DataHolder()
    d.starCnt = starCnt
    d.name = starName
    d.x = x
    d.y = y
    d.kp = kp
    d.AKs = AKs

    print(('Found %d young stars in database spectra' % starCnt))

    return d

def load_old():
    # Create a connection to the database file and create a cursor
    pwFile='/g/lu/scratch/siyao/other/pw.txt'
    pw = open(pwFile).read().split()[0]
    con = mdb.connect(host='galaxy1.astro.ucla.edu',user='dbread',passwd=pw,db='gcg')
    
    # Get info on the stars
    rows = pd.read_sql_query('SELECT name, kp, x, y, AK_sch FROM stars WHERE young="F"', con)

    print(('Found %d old stars in database stars' % len(rows)))

    starName = []
    x = []
    y = []
    kp = []
    AKs = []
    starInField = []
    
    # Loop through each star and pull out the relevant information
    for ss in range(len(rows)):
        record = rows.iloc[ss]
        name = record['name']

        # Check that we observed this star and what field it was in.
        row = pd.read_sql_query('SELECT field FROM spectra WHERE name = "%s"' % (name), con)

        if not row.empty:
            starInField.append(row.iloc[0]['field'])

            starName.append(str(name))
            kp.append(record['kp'])
            x.append(record['x'])
            y.append(record['y'])
            AKs.append(record['AK_sch'])

            if AKs[-1] == None:
                print(('%-13s  %5.2f  %7.3f  %7.3f  None' % \
                    (starName[-1], kp[-1], x[-1], y[-1])))
            else:
                print(('%-13s  %5.2f  %7.3f  %7.3f  %4.2f' % \
                    (starName[-1], kp[-1], x[-1], y[-1], AKs[-1])))
        else:
            print(('We do not have data on this star???', name))
            #starInField.append('C')
            continue
    

    starCnt = len(starName)
    starName = np.array(starName)
    starInField = np.array(starInField)
    x = np.array(x)
    y = np.array(y)
    kp = np.array(kp)
    AKs = np.array(AKs)

    d = objects.DataHolder()
    d.starCnt = starCnt
    d.name = starName
    d.x = x
    d.y = y
    d.kp = kp
    d.AKs = AKs

    print(('Found %d old stars in database spectra' % starCnt))

    return d
    
    
def completeness_to_one_AKs(magBins=None):
    """
    Take the NIRC2 completness correction curves and modify them
    so that they are in extinction corrected magnitudes. Everything
    is corrected to an extinction of AKs = 2.7.
    """
    if magBins == None:
        magBins = klf_mag_bins

    # Select all the OSIRIS fields-of-view
    fields = getOsirisFields()
    numFields = len(fields.name)

    # Load up the extinction map from Schodel 2010
    schExtinctFile = '/u/jlu/work/gc/imf/extinction/2010schodel_AKs_fg6.fits'
    schExtinct = pyfits.getdata(schExtinctFile)

    # Will de-redden all stars to AKs = 2.7. Then these can be converted
    # using the ks_2_kp factor (assumes hot star atmospheres)
    ks_2_kp = synthetic.get_Kp_Ks(theAKs, 30000.0, filename=synFile)

    # Load spectroscopic completeness data
    sc = load_spec_completeness_pickle()
    compSpecKp_ext = sc.magBins
    compSpecList = sc.comp_ext_fix

    comp_spec_ext = {}
    cnt_nirc2_ext = {}
    comp_imag_ext = {}
    comp_plant_ext = {}
    comp_found_ext = {}

    for ff in range(numFields):
        fieldName = fields.name[ff]
        fieldSuffix = fieldName.replace(' ', '')
        print(('Working on field %s' % fieldName))

        # Find the average extinction for this field
        fieldExtFile = '/u/jlu/work/gc/imf/extinction/extinct_mask_%s.fits' % \
            fieldSuffix
        fieldExtMask = pyfits.getdata(fieldExtFile)

        fieldExtMap = schExtinct * fieldExtMask
        idx = np.where(fieldExtMap != 0)
        fieldExtinctAvg = fieldExtMap[idx[0], idx[1]].mean()
        fieldExtinctStd = fieldExtMap[idx[0], idx[1]].std()

        # Load up the completeness curve for this field
        compFile = '%s/img_completeness_%s.dat' % (workDir, fieldSuffix)
        _comp = asciidata.open(compFile)
        compKp = _comp[0].tonumpy()
        compImag = _comp[1].tonumpy()
        compPlant = _comp[2].tonumpy()
        compFound = _comp[3].tonumpy()

        ff = np.where(sc.fieldSuffix == fieldSuffix)[0]
        compSpec_ext = compSpecList[ff[0]]
        cntNIRC2_ext = sc.cnt_ext_nirc2[ff[0]]

        # Apply an average differential extinction correction to 
        # the completeness curve. Remember, this only corrects the
        # completeness curve to an extinction of AKs=2.7
        # Switch to Ks, correct differential extinction, switch to Kp
        kp_2_ks = synthetic.get_Kp_Ks(fieldExtinctAvg, 30000.0, 
                                      filename=synFile)
        compKp_ext = compKp - kp_2_ks - fieldExtinctAvg + theAKs + ks_2_kp

        # Resample the completeness curves at the magnitude bins of
        # the luminosity function. Filter out NAN values.
        good = np.isnan(compImag) == False
        Kp_interp = interpolate.splrep(compKp_ext[good], compImag[good], k=1, s=0)
        compImag_ext = interpolate.splev(magBins, Kp_interp)

        Kp_interp = interpolate.splrep(compSpecKp_ext, compSpec_ext, k=1, s=0)
        compSpec_ext = interpolate.splev(magBins, Kp_interp)

        comp_imag_ext[fieldName] = compImag_ext
        comp_spec_ext[fieldName] = compSpec_ext
        comp_plant_ext[fieldName] = np.ones(len(magBins), dtype=int) * compPlant[0]
        cnt_nirc2_ext[fieldName] = cntNIRC2_ext

        Kp_interp = interpolate.splrep(compKp_ext, compFound, k=3, s=0)
        comp_found_ext[fieldName] = interpolate.splev(magBins, Kp_interp)

        py.clf()
        py.plot(magBins, comp_imag_ext[fieldName], label='Images')
        py.plot(magBins, comp_spec_ext[fieldName], label='Spectra')
        py.legend(loc='lower left')
        py.xlabel('Kp Magnitudes')
        py.ylabel('Extinction-Corrected Completeness')
        py.title(fieldName)
        py.savefig(workDir + 'completeness_spec_imag_' + fieldSuffix + '.png')

    _out = open(workDir + 'completeness_info.pickle', 'w')
    pickle.dump(magBins, _out)
    pickle.dump(comp_imag_ext, _out)
    pickle.dump(comp_spec_ext, _out)
    pickle.dump(comp_plant_ext, _out)
    pickle.dump(comp_found_ext, _out)
    pickle.dump(cnt_nirc2_ext, _out)
    _out.close()

def load_completeness_info():
    _in = open(workDir + 'completeness_info.pickle')

    c = objects.DataHolder()
    c.mag_bins = pickle.load(_in)
    c.comp_imag_ext = pickle.load(_in)
    c.comp_spec_ext = pickle.load(_in)
    c.comp_plant_ext = pickle.load(_in)
    c.comp_found_ext = pickle.load(_in)
    c.cnt_nirc2_ext = pickle.load(_in)

    return c

def klf():
    """
    Make a completeness curve for every osiris spectroscopic field. Use
    the completeness analysis from mag06maylgs1_dp_msc.
    """
    # Select all the OSIRIS fields-of-view
    fields = getOsirisFields()

    s = load_young()

    numStars = s.starCnt
    numFields = len(fields.name)

    # Differential Extinction Correction... set all to AKs = 2.7
    # Apply correction in Ks not Kp.
    s.kp_ext = np.zeros(numStars, dtype=float)

    # Will de-redden all stars to AKs = 2.7. Then these can be converted
    # using the ks_2_kp factor (assumes hot star atmospheres)
    ks_2_kp = synthetic.get_Kp_Ks(theAKs, 30000.0, filename=synFile)

    for ii in range(numStars):
        kp_2_ks = synthetic.get_Kp_Ks(s.AKs[ii], 30000.0, filename=synFile)

        # Switch to Ks, correct differential extinction, switch to Kp
        s.kp_ext[ii] = s.kp[ii] - kp_2_ks - s.AKs[ii] + theAKs + ks_2_kp


    xypoints = np.column_stack((s.x, s.y))

    # Load extinction corrected imaging and spectroscopic 
    # completeness
    c = load_completeness_info() 

    # Load up the extinction map from Schodel 2010
    schExtinctFile = '/u/jlu/work/gc/imf/extinction/2010schodel_AKs_fg6.fits'
    schExtinct = pyfits.getdata(schExtinctFile)

    # Load the field areas 
    areaDict = load_osiris_field_areas()

    # We will make a luminosity and mass function for every field. We
    # will combine them afterwards (any way we want).
    magBins = c.mag_bins
    massBins = 10**np.arange(1.0, 2.0, 0.1) # equally spaced in log(m)

    sizeKp = len(magBins)
    
    # Output arrays holding luminosity functions
    N = np.zeros((numFields, sizeKp), dtype=float)
    N_ext = np.zeros((numFields, sizeKp), dtype=float)
    KLF = np.zeros((numFields, sizeKp), dtype=float)
    KLF_ext = np.zeros((numFields, sizeKp), dtype=float)
    KLF_ext_cmp1 = np.zeros((numFields, sizeKp), dtype=float)
    KLF_ext_cmp2 = np.zeros((numFields, sizeKp), dtype=float)

    # errors
    eN = np.zeros((numFields, sizeKp), dtype=float)
    eN_ext = np.zeros((numFields, sizeKp), dtype=float)
    eKLF = np.zeros((numFields, sizeKp), dtype=float)
    eKLF_ext = np.zeros((numFields, sizeKp), dtype=float)
    eKLF_ext_cmp1 = np.zeros((numFields, sizeKp), dtype=float)
    eKLF_ext_cmp2 = np.zeros((numFields, sizeKp), dtype=float)

    field_names = []

    comp_spec = np.zeros((numFields, sizeKp), dtype=float)
    comp_imag = np.zeros((numFields, sizeKp), dtype=float)
    comp_imag_plant = np.zeros((numFields, sizeKp), dtype=float)
    comp_imag_found = np.zeros((numFields, sizeKp), dtype=float)
    cnt_nirc2 = np.zeros((numFields, sizeKp), dtype=float)

    for ff in range(numFields):
        fieldName = fields.name[ff]
        fieldSuffix = fieldName.replace(' ', '')
        print(('Working on field %s' % fieldName))

        field_names.append(fieldSuffix)

        # Find the average extinction for this field
        fieldExtFile = '/u/jlu/work/gc/imf/extinction/extinct_mask_%s.fits' % \
            fieldSuffix
        fieldExtMask = pyfits.getdata(fieldExtFile)

        fieldExtMap = schExtinct * fieldExtMask
        idx = np.where(fieldExtMap != 0)
        fieldExtinctAvg = fieldExtMap[idx[0], idx[1]].mean()
        fieldExtinctStd = fieldExtMap[idx[0], idx[1]].std()

        # Find the stars inside this field.
        inside = nxutils.points_inside_poly(xypoints, fields.xyverts[ff])
        inside = np.where(inside == True)[0]

        kpInField = s.kp[inside]
        kpInField_ext = s.kp_ext[inside]

        nameInField = s.name[inside]

        # Make a binned luminosity function.
        binSizeKp = magBins[1] - magBins[0]
        perAsec2Mag = areaDict[fieldName] * binSizeKp

        comp_spec[ff] = c.comp_spec_ext[fieldName]
        comp_imag[ff] = c.comp_imag_ext[fieldName]
        comp_imag_plant[ff] = c.comp_plant_ext[fieldName]
        comp_imag_found[ff] = c.comp_found_ext[fieldName]
        cnt_nirc2[ff] = c.cnt_nirc2_ext[fieldName]

        # Loop through each magnitude bin and get the surface density of stars.
        for kk in range(sizeKp):
            kp_lo = magBins[kk] - (binSizeKp/2.0)
            kp_hi = magBins[kk] + (binSizeKp/2.0)

            # Kp luminosity function (no extinction correction)
            idx = np.where((kpInField >= kp_lo) & (kpInField < kp_hi))[0]
            N[ff, kk] = len(idx)

            if magBins[kk] == 11:
                print((nameInField[idx]))
                print((kpInField[idx]))

            errN = math.sqrt(len(idx))
            if len(idx) == 0:
                errN = 1.0

            eN[ff, kk] = errN
            KLF[ff, kk] = len(idx) / perAsec2Mag
            eKLF[ff, kk] = errN / perAsec2Mag

            # Kp luminosity function (extinction corrected)
            idx = np.where((kpInField_ext >= kp_lo) & 
                           (kpInField_ext < kp_hi))[0]
            N_ext[ff,kk] = len(idx)
            
            errN = math.sqrt(len(idx))
#             if len(idx) == 0:
#                 errN = 1.0
            eN_ext[ff, kk] = errN
            KLF_ext[ff, kk] = len(idx) / perAsec2Mag
            eKLF_ext[ff, kk] = errN / perAsec2Mag

            # Correct for spectroscopic completeness
            KLF_ext_cmp1[ff, kk] = len(idx) / comp_spec[ff][kk]
            KLF_ext_cmp1[ff,kk] /= perAsec2Mag
            eKLF_ext_cmp1[ff, kk] = errN / comp_spec[ff][kk]
            eKLF_ext_cmp1[ff, kk] /= perAsec2Mag

            # Correct for imaging completeness
            KLF_ext_cmp2[ff, kk] = KLF_ext_cmp1[ff, kk] / comp_imag[ff][kk]
            eKLF_ext_cmp2[ff, kk] = eKLF_ext_cmp1[ff, kk] / comp_imag[ff][kk]

            # Fix some stuff
            idx = np.where(np.isnan(KLF_ext_cmp1[ff]) == True)[0]
            for ii in idx:
                KLF_ext_cmp1[ff,ii] = 0.0

            idx = np.where(np.isnan(KLF_ext_cmp2[ff]) == True)[0]
            for ii in idx:
                KLF_ext_cmp2[ff,ii] = 0.0


    # Save to a pickle file
    pickleFile = workDir + 'klf.dat'

    _out = open(pickleFile, 'w')
    
    pickle.dump(field_names, _out)
    pickle.dump(magBins, _out)
    pickle.dump(N, _out)
    pickle.dump(eN, _out)
    pickle.dump(N_ext, _out)
    pickle.dump(eN_ext, _out)
    pickle.dump(KLF, _out)
    pickle.dump(eKLF, _out)
    pickle.dump(KLF_ext, _out)
    pickle.dump(eKLF_ext, _out)
    pickle.dump(KLF_ext_cmp1, _out)
    pickle.dump(eKLF_ext_cmp1, _out)
    pickle.dump(KLF_ext_cmp2, _out)
    pickle.dump(eKLF_ext_cmp2, _out)
    pickle.dump(comp_spec, _out)
    pickle.dump(comp_imag, _out)
    pickle.dump(comp_imag_plant, _out)
    pickle.dump(comp_imag_found, _out)
    pickle.dump(cnt_nirc2, _out)

    _out.close()

def load_klf():
    pickleFile = workDir + 'klf.dat'
    _in = open(pickleFile, 'rb')
    
    d = objects.DataHolder()

    d.fields = pickle.load(_in)

    d.Kp = pickle.load(_in)
    
    d.N = pickle.load(_in)
    d.eN = pickle.load(_in)

    d.N_ext = pickle.load(_in)
    d.eN_ext = pickle.load(_in)

    d.KLF = pickle.load(_in)
    d.eKLF = pickle.load(_in)

    d.KLF_ext = pickle.load(_in)
    d.eKLF_ext = pickle.load(_in)

    d.KLF_ext_cmp1 = pickle.load(_in)
    d.eKLF_ext_cmp1 = pickle.load(_in)

    d.KLF_ext_cmp2 = pickle.load(_in)
    d.eKLF_ext_cmp2 = pickle.load(_in)

    d.comp_spec_ext = pickle.load(_in)
    d.comp_imag_ext = pickle.load(_in)
    d.comp_plant_ext = pickle.load(_in)
    d.comp_found_ext = pickle.load(_in)
    d.cnt_nirc2 = pickle.load(_in)

    _in.close()

    return d

def combine_klf(KLF_field, eKLF_field, comp_spec_field, comp_imag_field, 
                cnt_nirc2_field):

    KLF = KLF_per_field.mean(axis=0)

def plot_klf():
    d = load_klf()
    
    # Sum to get global KLFs
    numFields = d.KLF.shape[1]

    KLF = d.KLF.mean(axis=0)
    eKLF = np.sqrt((d.eKLF**2).sum(axis=0)) / numFields

    KLF_ext = d.KLF_ext.mean(axis=0)
    eKLF_ext = np.sqrt((d.eKLF_ext**2).sum(axis=0)) / numFields

    KLF_ext_cmp1 = d.KLF_ext_cmp1.mean(axis=0)
    eKLF_ext_cmp1 = np.sqrt((d.eKLF_ext_cmp1**2).sum(axis=0)) / numFields

    KLF_ext_cmp2 = d.KLF_ext_cmp2.mean(axis=0)
    eKLF_ext_cmp2 = np.sqrt((d.eKLF_ext_cmp2**2).sum(axis=0)) / numFields

    magBin = d.Kp[1] - d.Kp[0]

    idx1 = np.where(KLF != 0)[0]
    idx2 = np.where(KLF_ext != 0)[0]

    py.clf()
    py.errorbar(d.Kp[idx1], KLF[idx1], fmt='ko', xerr=magBin/2.0, capsize=0)
    py.errorbar(d.Kp[idx1], KLF[idx1], fmt='ko', yerr=eKLF[idx1])
    py.gca().set_yscale('log')
    py.xlabel('Kp magnitude')
    py.ylabel('Stars / (arcsec^2 mag)')
    py.savefig(workDir + 'plots/klf_obs.png')

    py.clf()
    py.errorbar(d.Kp[idx2], KLF_ext[idx2], fmt='ko', xerr=magBin/2.0, capsize=0)
    py.errorbar(d.Kp[idx2], KLF_ext[idx2], fmt='ko', yerr=eKLF_ext[idx2])
    py.gca().set_yscale('log')
    py.xlabel('Kp magnitude')
    py.ylabel('Stars / (arcsec^2 mag)')
    py.savefig(workDir + 'plots/klf_ext.png')

    py.clf()
    py.errorbar(d.Kp[idx2], KLF_ext_cmp1[idx2], 
                fmt='ko', xerr=magBin/2.0, capsize=0)
    py.errorbar(d.Kp[idx2], KLF_ext_cmp1[idx2], 
                fmt='ko', yerr=eKLF_ext_cmp1[idx2])
    py.gca().set_yscale('log')
    py.xlabel('Kp magnitude')
    py.ylabel('Stars / (arcsec^2 mag)')
    py.savefig(workDir + 'plots/klf_ext_cmp1.png')

    py.clf()
    py.errorbar(d.Kp[idx2], KLF_ext_cmp2[idx2], 
                fmt='ko', xerr=magBin/2.0, capsize=0)
    py.errorbar(d.Kp[idx2], KLF_ext_cmp2[idx2], 
                fmt='ko', yerr=eKLF_ext_cmp2[idx2])
    py.gca().set_yscale('log')
    py.xlabel('Kp magnitude')
    py.ylabel('Stars / (arcsec^2 mag)')
    py.savefig(workDir + 'plots/klf_ext_cmp2.png')

def plot_klf2():
    d = load_klf()
    
    # Sum to get global KLFs
    numFields = d.KLF.shape[1]

    N = d.N.sum(axis=0)
    eN = np.sqrt(N)

    N_ext = d.N_ext.sum(axis=0)
    eN_ext = np.sqrt(N_ext)

    KLF = d.KLF.mean(axis=0)
    eKLF = np.sqrt((d.eKLF**2).sum(axis=0)) / numFields

    KLF_ext = d.KLF_ext.mean(axis=0)
    eKLF_ext = np.sqrt((d.eKLF_ext**2).sum(axis=0)) / numFields

    # Calculate the completeness curves for all the fields combined.
    # Use the weighted average completeness correction.
    #weights = 1.0 / d.comp_plant_ext # plant is the same for all mag bins.
    weights =  d.cnt_nirc2 # plant is the same for all mag bins.
    comp_imag = (d.comp_imag_ext * weights).sum(axis=0) / weights.sum(axis=0)
    comp_spec = (d.comp_spec_ext * weights).sum(axis=0) / weights.sum(axis=0)

    KLF_ext_cmp1 = KLF_ext / comp_spec
    eKLF_ext_cmp1 = eKLF_ext / comp_spec

    KLF_ext_cmp2 = KLF_ext_cmp1 / comp_imag
    eKLF_ext_cmp2 = eKLF_ext_cmp1 / comp_imag

    N_ext_cmp1 = N_ext / comp_spec
    eN_ext_cmp1 = eN_ext / comp_spec

    N_ext_cmp2 = N_ext_cmp1 / comp_imag
    eN_ext_cmp2 = eN_ext_cmp1 / comp_imag

    magBin = d.Kp[1] - d.Kp[0]

    idx1 = np.where(KLF != 0)[0]
    idx2 = np.where(KLF_ext != 0)[0]
    idx3 = np.where(N != 0)[0]
    idx4 = np.where(N_ext != 0)[0]

    py.clf()
    py.errorbar(d.Kp[idx1], KLF[idx1], fmt='ko', xerr=magBin/2.0, capsize=0)
    py.errorbar(d.Kp[idx1], KLF[idx1], fmt='ko', yerr=eKLF[idx1])
    py.gca().set_yscale('log')
    py.xlabel('Kp magnitude')
    py.ylabel('Stars / (arcsec^2 mag)')
    py.savefig(workDir + 'plots/klf2_obs.png')

    py.clf()
    py.errorbar(d.Kp[idx2], KLF_ext[idx2], fmt='ko', xerr=magBin/2.0, capsize=0)
    py.errorbar(d.Kp[idx2], KLF_ext[idx2], fmt='ko', yerr=eKLF_ext[idx2])
    py.gca().set_yscale('log')
    py.xlabel('Kp magnitude')
    py.ylabel('Stars / (arcsec^2 mag)')
    py.savefig(workDir + 'plots/klf2_ext.png')

    py.clf()
    py.errorbar(d.Kp[idx2], KLF_ext_cmp1[idx2], 
                fmt='ko', xerr=magBin/2.0, capsize=0)
    py.errorbar(d.Kp[idx2], KLF_ext_cmp1[idx2], 
                fmt='ko', yerr=eKLF_ext_cmp1[idx2])
    py.gca().set_yscale('log')
    py.xlabel('Kp magnitude')
    py.ylabel('Stars / (arcsec^2 mag)')
    py.savefig(workDir + 'plots/klf2_ext_cmp1.png')

    py.clf()
    py.errorbar(d.Kp[idx2], KLF_ext_cmp2[idx2], 
                fmt='ko', xerr=magBin/2.0, capsize=0)
    py.errorbar(d.Kp[idx2], KLF_ext_cmp2[idx2], 
                fmt='ko', yerr=eKLF_ext_cmp2[idx2])
    py.gca().set_yscale('log')
    py.xlabel('Kp magnitude')
    py.ylabel('Stars / (arcsec^2 mag)')
    py.savefig(workDir + 'plots/klf2_ext_cmp2.png')

    py.clf()
    py.errorbar(d.Kp[idx1], N[idx3], fmt='ko', xerr=magBin/2.0, capsize=0)
    py.errorbar(d.Kp[idx1], N[idx3], fmt='ko', yerr=eN[idx3])
    py.gca().set_yscale('log')
    py.xlabel('Kp magnitude')
    py.ylabel('Stars')
    py.savefig(workDir + 'plots/klf2_count_obs.png')

    mag_tmp = d.Kp[idx1]
    N_tmp = N[idx3]
    eN_tmp = eN[idx3]

    sdx = mag_tmp.argsort()

    print(('%5s  %5s  %5s' % ('Kp', 'N', 'errN')))
    for ii in sdx:
        print(('%5.2f  %5d  %5d' % (mag_tmp[ii], N_tmp[ii], eN_tmp[ii])))

    py.clf()
    py.errorbar(d.Kp[idx2], N_ext[idx4], fmt='ko', xerr=magBin/2.0, capsize=0)
    py.errorbar(d.Kp[idx2], N_ext[idx4], fmt='ko', yerr=eN_ext[idx4])
    py.gca().set_yscale('log')
    py.xlabel('Kp magnitude')
    py.ylabel('Stars')
    py.savefig(workDir + 'plots/klf2_count_ext.png')

    py.clf()
    py.errorbar(d.Kp[idx4], N_ext_cmp1[idx4], 
                fmt='ko', xerr=magBin/2.0, capsize=0)
    py.errorbar(d.Kp[idx4], N_ext_cmp1[idx4], 
                fmt='ko', yerr=eN_ext_cmp1[idx4])
    py.gca().set_yscale('log')
    py.xlabel('Kp magnitude')
    py.ylabel('Stars')
    py.savefig(workDir + 'plots/klf2_count_ext_cmp1.png')

    py.clf()
    py.errorbar(d.Kp[idx4], N_ext_cmp2[idx4], 
                fmt='ko', xerr=magBin/2.0, capsize=0)
    py.errorbar(d.Kp[idx4], N_ext_cmp2[idx4], 
                fmt='ko', yerr=eN_ext_cmp2[idx4])
    py.gca().set_yscale('log')
    py.xlabel('Kp magnitude')
    py.ylabel('Stars')
    py.savefig(workDir + 'plots/klf2_count_ext_cmp2.png')

    py.clf()
    py.plot(d.Kp, comp_imag, label='Imaging')
    py.plot(d.Kp, comp_spec, label='Spectra')
    py.legend()
    py.xlabel('Kp magnitude')
    py.ylabel('Completeness')
    py.savefig(workDir + 'plots/klf2_completeness.png')

def plot_klf_3radii():
    d = load_klf()
    
    idx1 = []
    idx2 = []
    idx3 = []
    for ff in range(len(d.fields)):
        if d.fields[ff] == 'GCCentral':
            idx1.append(ff)
        else:
            if d.fields[ff].startswith('E'):
                idx3.append(ff)
            else:
                idx2.append(ff)

    # Calculate the completeness curves for all the fields combined.
    # Use the weighted average completeness correction.
    #weights = 1.0 / d.comp_plant_ext # plant is the same for all mag bins.
    weights = d.cnt_nirc2 # plant is the same for all mag bins.
    comp_fields = d.comp_imag_ext * d.comp_spec_ext

    comp1 = (comp_fields[idx1] * weights[idx1]).sum(axis=0)
    comp1 /= weights[idx1].sum(axis=0)
    comp2 = (comp_fields[idx2] * weights[idx2]).sum(axis=0)
    comp2 /= weights[idx2].sum(axis=0)
    comp3 = (comp_fields[idx3] * weights[idx3]).sum(axis=0)
    comp3 /= weights[idx3].sum(axis=0)

    N1 = d.N_ext[idx1].sum(axis=0) / comp1
    eN1 = np.sqrt(N1 * comp1) / comp1
    N2 = d.N_ext[idx2].sum(axis=0) / comp2
    eN2 = np.sqrt(N2 * comp2) / comp2
    N3 = d.N_ext[idx3].sum(axis=0) / comp3
    eN3 = np.sqrt(N3 * comp3) / comp3

    KLF1 = d.KLF_ext[idx1].mean(axis=0) / comp1
    eKLF1 = np.sqrt((d.eKLF_ext[idx1]**2).sum(axis=0)) / (len(idx1) * comp1)
    KLF2 = d.KLF_ext[idx2].mean(axis=0) / comp2
    eKLF2 = np.sqrt((d.eKLF_ext[idx2]**2).sum(axis=0)) / (len(idx2) * comp2)
    KLF3 = d.KLF_ext[idx3].mean(axis=0) / comp3
    eKLF3 = np.sqrt((d.eKLF_ext[idx3]**2).sum(axis=0)) / (len(idx3) * comp3)

    magBin = d.Kp[1] - d.Kp[0]

    # Repair for zeros since we are plotting in semi-log-y
    eN1 = np.ma.masked_where(N1 <= 0, eN1)
    eN2 = np.ma.masked_where(N2 <= 0, eN2)
    eN3 = np.ma.masked_where(N3 <= 0, eN3)

    N1 = np.ma.masked_where(N1 <= 0, N1)
    N2 = np.ma.masked_where(N2 <= 0, N2)
    N3 = np.ma.masked_where(N3 <= 0, N3)

    eKLF1 = np.ma.masked_where(KLF1 <= 0, eKLF1)
    eKLF2 = np.ma.masked_where(KLF2 <= 0, eKLF2)
    eKLF3 = np.ma.masked_where(KLF3 <= 0, eKLF3)

    KLF1 = np.ma.masked_where(KLF1 <= 0, KLF1)
    KLF2 = np.ma.masked_where(KLF2 <= 0, KLF2)
    KLF3 = np.ma.masked_where(KLF3 <= 0, KLF3)

    idx = np.where(d.Kp < 16)[0]

    py.clf()
    py.errorbar(d.Kp[idx], N1[idx], fmt='ko-', xerr=magBin/2.0, capsize=0)
    py.errorbar(d.Kp[idx], N1[idx], fmt='ko', yerr=eN1[idx], label='central')
    py.errorbar(d.Kp[idx], N2[idx], fmt='ro-', xerr=magBin/2.0, capsize=0)
    py.errorbar(d.Kp[idx], N2[idx], fmt='ro', yerr=eN2[idx], label='speckle - central')
    py.errorbar(d.Kp[idx], N3[idx], fmt='bo-', xerr=magBin/2.0, capsize=0)
    py.errorbar(d.Kp[idx], N3[idx], fmt='bo', yerr=eN3[idx], label='gcows')
    py.gca().set_yscale('log')
    py.legend(loc='upper left', numpoints=1)
    py.xlabel('Kp magnitude')
    py.ylabel('Stars')
    py.savefig(workDir + 'plots/klf_count_3radii.png')

    py.clf()
    py.errorbar(d.Kp[idx], KLF1[idx], fmt='ko-', xerr=magBin/2.0, capsize=0)
    py.errorbar(d.Kp[idx], KLF1[idx], fmt='ko', yerr=eKLF1[idx], label='central')
    py.errorbar(d.Kp[idx], KLF2[idx], fmt='ro-', xerr=magBin/2.0, capsize=0)
    py.errorbar(d.Kp[idx], KLF2[idx], fmt='ro', yerr=eKLF2[idx], label='speckle - central')
    py.errorbar(d.Kp[idx], KLF3[idx], fmt='bo-', xerr=magBin/2.0, capsize=0)
    py.errorbar(d.Kp[idx], KLF3[idx], fmt='bo', yerr=eKLF3[idx], label='gcows')
    py.gca().set_yscale('log')
    py.legend(loc='upper left', numpoints=1)
    py.xlabel('Kp magnitude')
    py.ylabel('Stars / (arcsec^2 mag)')
    py.savefig(workDir + 'plots/klf_3radii.png')

def plot_klf_2radii():
    d = load_klf()
    
    idx1 = []
    idx2 = []
    for ff in range(len(d.fields)):
        if d.fields[ff] == 'GCCentral':
            idx1.append(ff)
        else:
            idx2.append(ff)

    # Calculate the completeness curves for all the fields combined.
    # Use the weighted average completeness correction.
    #weights = 1.0 / d.comp_plant_ext # plant is the same for all mag bins.
    weights = d.cnt_nirc2 # plant is the same for all mag bins.
    comp_fields = d.comp_imag_ext * d.comp_spec_ext

    comp1 = (comp_fields[idx1] * weights[idx1]).sum(axis=0)
    comp1 /= weights[idx1].sum(axis=0)
    comp2 = (comp_fields[idx2] * weights[idx2]).sum(axis=0)
    comp2 /= weights[idx2].sum(axis=0)

    N1 = d.N_ext[idx1].sum(axis=0) / comp1
    eN1 = np.sqrt(N1 * comp1) / comp1
    N2 = d.N_ext[idx2].sum(axis=0) / comp2
    eN2 = np.sqrt(N2 * comp2) / comp2

    KLF1 = d.KLF_ext[idx1].mean(axis=0) / comp1
    eKLF1 = np.sqrt((d.eKLF_ext[idx1]**2).sum(axis=0)) / (len(idx1) * comp1)
    KLF2 = d.KLF_ext[idx2].mean(axis=0) / comp2
    eKLF2 = np.sqrt((d.eKLF_ext[idx2]**2).sum(axis=0)) / (len(idx2) * comp2)

    magBin = d.Kp[1] - d.Kp[0]

    # Repair for zeros since we are plotting in semi-log-y
    eN1 = np.ma.masked_where(N1 <= 0, eN1)
    eN2 = np.ma.masked_where(N2 <= 0, eN2)

    N1 = np.ma.masked_where(N1 <= 0, N1)
    N2 = np.ma.masked_where(N2 <= 0, N2)

    eKLF1 = np.ma.masked_where(KLF1 <= 0, eKLF1)
    eKLF2 = np.ma.masked_where(KLF2 <= 0, eKLF2)

    KLF1 = np.ma.masked_where(KLF1 <= 0, KLF1)
    KLF2 = np.ma.masked_where(KLF2 <= 0, KLF2)

    idx = np.where(d.Kp < 16)[0]

    py.clf()
    py.errorbar(d.Kp[idx], N1[idx], fmt='ro-', xerr=magBin/2.0, capsize=0)
    py.errorbar(d.Kp[idx], N1[idx], fmt='ro', yerr=eN1[idx], label='central')
    py.errorbar(d.Kp[idx], N2[idx], fmt='bo-', xerr=magBin/2.0, capsize=0)
    py.errorbar(d.Kp[idx], N2[idx], fmt='bo', yerr=eN2[idx], label='outer')
    py.gca().set_yscale('log')
    py.legend(loc='upper left', numpoints=1)
    py.xlabel('Kp magnitude')
    py.ylabel('Stars')
    py.savefig(workDir + 'plots/klf_count_2radii.png')

    py.clf()
    py.errorbar(d.Kp[idx], KLF1[idx], fmt='ro-', xerr=magBin/2.0, capsize=0)
    py.errorbar(d.Kp[idx], KLF1[idx], fmt='ro', yerr=eKLF1[idx], label='central')
    py.errorbar(d.Kp[idx], KLF2[idx], fmt='bo-', xerr=magBin/2.0, capsize=0)
    py.errorbar(d.Kp[idx], KLF2[idx], fmt='bo', yerr=eKLF2[idx], label='outer')
    py.gca().set_yscale('log')
    py.legend(loc='upper left', numpoints=1)
    py.xlabel('Kp magnitude')
    py.ylabel('Stars / (arcsec^2 mag)')
    py.savefig(workDir + 'plots/klf_2radii.png')


def plot_klf_vs_bartko(logAge=6.78, imfSlope=1.35):
    d = load_klf()
    
    idx1 = []
    idx2 = []
    for ff in range(len(d.fields)):
#         if d.fields[ff] == 'GCCentral':
#             idx1.append(ff)
#         else:
        idx2.append(ff)

    # Calculate the completeness curves for all the fields combined.
    # Use the weighted average completeness correction.
#     weights = 1.0 / d.comp_plant_ext # plant is the same for all mag bins.
    weights = d.cnt_nirc2 # plant is the same for all mag bins.
    comp_fields = d.comp_imag_ext * d.comp_spec_ext

    comp1 = (comp_fields[idx1] * weights[idx1]).sum(axis=0)
    comp1 /= weights[idx1].sum(axis=0)
    comp2 = (comp_fields[idx2] * weights[idx2]).sum(axis=0)
    comp2 /= weights[idx2].sum(axis=0)

    N1 = d.N_ext[idx1].sum(axis=0) / comp1
    eN1 = np.sqrt(N1 * comp1) / comp1
    N2 = d.N_ext[idx2].sum(axis=0) / comp2
    eN2 = np.sqrt(N2 * comp2) / comp2

    KLF1 = d.KLF_ext[idx1].mean(axis=0) / comp1
    eKLF1 = np.sqrt((d.eKLF_ext[idx1]**2).sum(axis=0)) / (len(idx1) * comp1)
    KLF2 = d.KLF_ext[idx2].mean(axis=0) / comp2
    eKLF2 = np.sqrt((d.eKLF_ext[idx2]**2).sum(axis=0)) / (len(idx2) * comp2)

    magBin = d.Kp[1] - d.Kp[0]

    # Repair for zeros since we are plotting in semi-log-y
    eN1 = np.ma.masked_where(N1 <= 0, eN1)
    eN2 = np.ma.masked_where(N2 <= 0, eN2)

    N1 = np.ma.masked_where(N1 <= 0, N1)
    N2 = np.ma.masked_where(N2 <= 0, N2)

    eKLF1 = np.ma.masked_where(KLF1 <= 0, eKLF1)
    eKLF2 = np.ma.masked_where(KLF2 <= 0, eKLF2)

    KLF1 = np.ma.masked_where(KLF1 <= 0, KLF1)
    KLF2 = np.ma.masked_where(KLF2 <= 0, KLF2)

    idx = np.where(d.Kp <= 16)[0]

    # Rescale Bartko to match our first two bins
    bartkoKLF = np.array([0.034, 0.08, 0.19, 0.20, 0.14, 0.18, 0.10])
    bartkoKp = np.array( [  9.5, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0])
    bartkoModel = np.array([0.0089, 0.02, 0.07, 0.24, 0.26, 0.4, 0.7])
    bartkoErrX = np.array([1, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5])

    scaleFactor = 1.0
    bartkoKLF = bartkoKLF * scaleFactor

    # Load up a model luminosity function
    modelStars = model_klf(logAge=logAge, imfSlope=imfSlope)
    scaleFactor = 700.
    weights = np.ones(len(modelStars), dtype=float)
    weights *= scaleFactor / len(modelStars)

    # Model
    binsKp = klf_mag_bins
    binEdges = binsKp[0:-1] + (binsKp[1:] - binsKp[0:-1]) / 2.0
    py.clf()
    (n, b, p) = py.hist(modelStars, bins=binEdges, histtype='step', 
                        color='green', weights=weights, 
                        label='Salpeter Model', align='mid')
    py.gca().set_yscale('log')


    #     py.plot(bartkoKp, bartkoModel, 'r-')
    bartkoErr = np.zeros(len(bartkoKp), dtype=float) + 0.02
    py.errorbar(bartkoKp, bartkoKLF, fmt='bs', yerr=bartkoErr,
                label='VLT (Bartko et al. 2010)')
    py.errorbar(bartkoKp, bartkoKLF, fmt='bs', xerr=bartkoErrX, capsize=0)

    py.errorbar(d.Kp[idx], KLF2[idx], fmt='ro-', xerr=magBin/2.0, capsize=0)
    py.errorbar(d.Kp[idx], KLF2[idx], fmt='ro-', yerr=eKLF2[idx], 
                label='Keck (Lu et al. in prep.)')


    py.gca().set_yscale('log')
    py.legend(loc='upper left', numpoints=1)
    py.ylim(0.01, 2)
    py.xlim(8.5, 16.5)
    py.xlabel('Kp magnitude')
    py.ylabel('Stars / (arcsec^2 mag)')
    py.title('logAge=%.2f, IMF slope=%.2f' % (logAge, imfSlope))
    py.savefig(workDir + 'plots/klf_vs_bartko.png')

def model_klf(logAge=6.78, imfSlope=2.35, clusterMass=10**4,
              filterName='Kp', makeMultiples=True,
              includeWR=False, output=None):
    from jlu.gc.imf import bayesian as b

    cluster = b.model_young_cluster(logAge, filterName=filterName,
                                    AKs=theAKs, distance=distance,
                                    imfSlope=imfSlope, clusterMass=clusterMass,
                                    makeMultiples=makeMultiples)

    logAgeString = '0%d' % (logAge * 100)

    # Now sample an IMF randomly from 1-30 Msun
    # Salpeter
    #minMass = 1.0
    #randMass = plfit.plexp_inv(np.random.rand(10**5), minMass, imfSlope)

    # # Kroupa+ 1993
    # randUni = np.random.rand(10**5)
    # randMass = 0.01 + ((0.19*(randUni**1.55) + 0.05*(randUni**0.6)) / ((1 - randUni)**0.58))

    outSuffix = '_logt%4.2f_imf%4.2f_%s.png' % (logAge, imfSlope, filterName)
    
    randMass = cluster.mass
    randMag = cluster.mag
    randTeff = cluster.Teff
    randIsWR = cluster.isWR

    # Now sort by mass
    sdx = randMass.argsort()
    randMass = randMass[sdx]
    randTeff = randTeff[sdx]
    randMag = randMag[sdx]
    randIsWR = randIsWR[sdx]

    # Save an output file for Tuan:
    # _out = open('for_tuan_sim_klf_t%.2f.txt' % logAge, 'w')
    # _out.write('%8s  %6s  %5s  %5s\n' % ('Mass', 'Kp', 'Teff', 'isWR'))
    # for ii in range(len(randMass)):
    #     _out.write('%8.4f  %6.3f  %5d  %s\n' %
    #                (randMass[ii], randMag[ii], randTeff[ii], randIsWR[ii]))
    # _out.close()

    # Plot the model mass luminosity relationship
    py.clf()
    py.semilogx(randMass, randMag, 'k.-', ms=1)
    py.semilogx(randMass[randIsWR], randMag[randIsWR], 'rs-')
    py.xlabel("Stellar Mass (Msun)")
    py.ylabel("%s (AKs = %.2f, d=8 kpc)" % (filterName, theAKs))
    py.title("logAge=%.2f, IMF slope=%.2f" % (logAge, imfSlope))
    yrange = py.gca().get_ylim()
    py.plot([0.08, 0.08], yrange, 'k--') # Brown Dwarf Limit
    py.xlim(0.1, 150)
    py.gca().set_ylim(yrange[::-1])
    py.savefig(workDir + 'plots/mass_luminosity' + outSuffix)

    # Plot the mass function
    bins = np.arange(-2, 2, 0.1)
    bins = 10**bins
    py.clf()
    py.hist(randMass, histtype='step', bins=bins, log=True)
    py.hist(randMass[randIsWR], histtype='step', bins=bins, log=True, 
            color='red')
    py.gca().set_xscale('log')
    py.title("logAge=%.2f, IMF slope=%.2f" % (logAge, imfSlope))
    py.xlabel('Stellar Mass (Msun)')
    py.ylabel('Number of Stars')
    py.xlim(randMass.min(), 100)
    py.savefig(workDir + 'plots/model_imf' + outSuffix)

    py.clf()
    py.semilogx(randTeff, randMag, 'k.-', ms=1)
    py.semilogx(randTeff[randIsWR], randMag[randIsWR], 'rs')
    py.xlabel('Temperature (K)')
    py.ylabel("%s (AKs = %.2f, d=8 kpc)" % (filterName, theAKs))
    rng = py.axis()
    py.gca().set_xlim(py.gca().get_xlim()[::-1])
    py.gca().set_ylim(py.gca().get_ylim()[::-1])
    py.title("logAge=%.2f, IMF slope=%.2f" % (logAge, imfSlope))
    py.savefig(workDir + 'plots/model_cmd' + outSuffix)

    bins = np.arange(6, 24, 0.5)
    py.clf()
    py.hist(randMag, histtype='step', bins=bins)
    py.hist(randMag[randIsWR], histtype='step', bins=bins, color='red')
    py.gca().set_yscale('log')
    py.xlabel("%s (AKs = %.2f, d=8 kpc)" % (filterName, theAKs))
    py.ylabel('Number of Stars')
    py.title("logAge=%.2f, IMF slope=%.2f" % (logAge, imfSlope))
    py.savefig(workDir + 'plots/model_klf' + outSuffix)

    print(('Age = %.2f Myr' % (10**(logAge-6.0))))
    
    # Trim out WR stars
    if not includeWR:
        idx = np.where(randIsWR == False)[0]
        randMag = randMag[idx]
        
    return randMag

def calc_mf():
    d = load_klf()

    idx1 = []
    idx2 = []
    for ff in range(len(d.fields)):
        if d.fields[ff] == 'GCCentral':
            idx1.append(ff)
        else:
            idx2.append(ff)

    # Calculate the completeness curves for all the fields combined.
    # Use the weighted average completeness correction.
    #weights = 1.0 / d.comp_plant_ext # plant is the same for all mag bins.
    weights = d.cnt_nirc2 # plant is the same for all mag bins.
    comp_fields = d.comp_imag_ext * d.comp_spec_ext

    comp1 = (comp_fields[idx1] * weights[idx1]).sum(axis=0)
    comp1 /= weights[idx1].sum(axis=0)
    comp2 = (comp_fields[idx2] * weights[idx2]).sum(axis=0)
    comp2 /= weights[idx2].sum(axis=0)

    N1 = d.N_ext[idx1].sum(axis=0) / comp1
    eN1 = np.sqrt(N1 * comp1) / comp1
    N2 = d.N_ext[idx2].sum(axis=0) / comp2
    eN2 = np.sqrt(N2 * comp2) / comp2

    KLF1 = d.KLF_ext[idx1].mean(axis=0) / comp1
    eKLF1 = np.sqrt((d.eKLF_ext[idx1]**2).sum(axis=0)) / (len(idx1) * comp1)
    KLF2 = d.KLF_ext[idx2].mean(axis=0) / comp2
    eKLF2 = np.sqrt((d.eKLF_ext[idx2]**2).sum(axis=0)) / (len(idx2) * comp2)

    # ==========
    # Convert to masses
    # Assume solar metallicity, A_K=3.3, distance = 8 kpc, age = 6 Myr
    # ==========
    iso = load_isochrone()
    modMass = iso.M
    modT = iso.T
    modLogg = iso.logg
    modLogL = iso.logL
    modKp = iso.mag
    isWR = iso.isWR
    
    # Calculate the mass function
    imfMass = np.zeros(len(d.Kp), dtype=float)
    imfKp = np.zeros(len(d.Kp), dtype=float)
    imfN = np.zeros(len(d.Kp), dtype=float)
    imfNerr = np.zeros(len(d.Kp), dtype=float)

    magBinSize = d.Kp[1] - d.Kp[0]

    print(('%5s   %5s   %15s   %5s' % ('Kp', 'Mass', 'N/(arcsec^ mag)', 'Teff')))
    for ii in range(len(d.Kp)):
        idx = abs(d.Kp[ii] - modKp).argmin()
        imfMass[ii] = modMass[idx]
        imfKp[ii] = modKp[idx]

        # we had dN/d(mag), but we want dN/d(mass)
        lo_Kp = d.Kp[ii] - magBinSize/2.0
        hi_Kp = d.Kp[ii] + magBinSize/2.0

        lo_idx = abs(lo_Kp - modKp).argmin()
        hi_idx = abs(hi_Kp - modKp).argmin()
        dmass = modMass[lo_idx] - modMass[hi_idx]
        
        imfN[ii] = KLF2[ii] / dmass
        imfNerr[ii] = eKLF2[ii] / dmass

        print(('%5.2f   %5.2f   %5.3f +/- %5.3f' % \
            (imfKp[ii], imfMass[ii], imfN[ii], imfNerr[ii])))


    idx = np.where(KLF2 != 0)[0]

    # Plot the luminosity to mass relation for this isochrone to look
    # for degeneracies.
    py.clf()
    py.semilogy(modKp, modMass, 'k.')
    py.xlabel('Kp Magnitude')
    py.ylabel('Stellar Mass (Msun)')
    py.xlim(6, 18)
    py.ylim(2, 40)
    py.savefig(workDir + 'plots/mass_luminosity.png')

    py.clf()
    py.errorbar(imfMass[idx], KLF2[idx], 
                fmt='ko', yerr=eKLF2[idx])
    py.gca().set_yscale('log')
    py.gca().set_xscale('log')
    py.xlabel('Stellar Mass (Msun)')
    py.ylabel('Stars / (arcsec^2 mag)')
    py.savefig(workDir + 'plots/mf_ext_cmp2.png')


    py.clf()
    py.errorbar(imfMass[idx], imfN[idx], fmt='ko', yerr=imfNerr[idx])
    py.gca().set_yscale('log')
    py.gca().set_xscale('log')
    py.xlabel('Stellar Mass (Msun)')
    py.ylabel('Stars / (arcsec^2 Msun)')
    py.savefig(workDir + 'plots/mass_function.png')

    # Fit a slope in log N/log M to just the non-WR stars (K>=13)
    idx = np.where((imfKp >= 13) & (imfKp < 17.5))[0]
    py.clf()
    py.errorbar(imfMass[idx], imfN[idx], fmt='ko', yerr=imfNerr[idx])
    py.gca().set_yscale('log')
    py.gca().set_xscale('log')
    py.xlabel('Stellar Mass (Msun)')
    py.ylabel('Stars / (arcsec^2 Msun)')
    py.savefig(workDir + 'plots/mf_ext_cmp2_noWR.png')

    fitOut = fit_imf_slope(imfMass[idx], KLF2[idx], eKLF2[idx])
    fitLine = fitOut[0]
    fitSlope = fitOut[1]
    fitSlopeErr = fitOut[2]
    fitAmp = fitOut[3]
    fitAmpErr = fitOut[4]

    print('Best Fit Powerlaw:')
    print(('  slope = %5.2f +/- %5.2f' % (fitSlope, fitSlopeErr)))
    print(('  amp   = %5.2f +/- %5.2f' % (fitAmp, fitAmpErr)))

    py.clf()
    py.errorbar(imfMass[idx], imfN[idx], fmt='ko', yerr=imfNerr[idx], 
                label='Data')
    py.plot(imfMass[idx], fitLine, 'k--', label='Best Fit')
    py.gca().set_yscale('log')
    py.gca().set_xscale('log')
    py.xlabel('Stellar Mass (Msun)')
    py.ylabel('Stars / (arcsec^2 Msun)')
    py.title('Powerlaw slope = %5.2f +/- %5.2f' % (fitSlope, fitSlopeErr))
    py.legend()
    py.savefig(workDir + 'plots/mass_function_noWR.png')
    

    # Add up all the points at the high mass end, since they
    # are all the same mass.
    idx = np.where(imfMass > 25)[0]
    lastBinMass = imfMass[idx].mean()
    lastBinCount = KLF2[idx].sum()
    lastBinErr = np.sqrt((eKLF2[idx]**2).sum())

    imfMass = np.concatenate(([lastBinMass], imfMass[idx[-1]+1:]))
    imfCount = np.concatenate(([lastBinCount], KLF2[idx[-1]+1:]))
    imfError = np.concatenate(([lastBinErr], eKLF2[idx[-1]+1:]))

    indices = list(range(len(imfMass)))
    indices.reverse()
    print('')
    for ii in indices:
        print(('%5.2f  %6.4f +/- %6.4f' % \
            (imfMass[ii], imfCount[ii], imfError[ii])))


    py.clf()
    py.errorbar(imfMass, imfCount, fmt='ko', yerr=imfError)


def fit_imf_slope(mass, dNdM, dNdMerr):
    # Convert into log-log space

    logMass = np.log10(mass)
    logN = np.log10(dNdM)
    error = dNdMerr / dNdM

    # Works on input data
    powerlaw = lambda x, amp, index: amp * (x**index)

    # Works on log of input data
    fitfunc = lambda p, x: p[0] + p[1] * x
    errfunc = lambda p, x, y, err: (y - fitfunc(p, x)) / err


    pinit = [1.0, -1.0]
    out = optimize.leastsq(errfunc, pinit,
                           args=(logMass, logN, error), full_output=1)
    pfinal = out[0]
    covar = out[1]

    index = pfinal[1]
    amp = 10.0**pfinal[0]

    indexErr = math.sqrt( covar[1][1] )
    ampErr = math.sqrt( covar[0][0] ) * amp * math.log(10.0)

    print((pfinal, covar))

    fit = powerlaw(mass, amp, index)

    return fit, index, indexErr, amp, ampErr


def load_isochrone(logAge=6.78, filterName='Kp', AKs=theAKs, distance=int(distance)):
    inFile = '/u/jlu/work/gc/imf/klf/models/iso/'
    inFile += 'iso_%.2f_%s_%4.2f_%4s.pickle' % (logAge, filterName, AKs,
                                                 str(distance).zfill(4))
    print(inFile)

    if not os.path.exists(inFile):
        make_observed_isochrone(logAge=logAge, filterName=filterName,
                                AKs=AKs, distance=distance)

    _in = open(inFile, 'r')
    iso = objects.DataHolder()
    iso.M = pickle.load(_in)
    iso.T = pickle.load(_in)
    iso.logg = pickle.load(_in)
    iso.logL = pickle.load(_in)
    iso.mag = pickle.load(_in)
    iso.isWR = pickle.load(_in)
    _in.close()

    return iso

def make_observed_isochrone(logAge=6.78, filterName='Kp', AKs=theAKs, distance=int(distance)):
    outFile = '/u/jlu/work/gc/imf/klf/models/iso/'
    outFile += 'iso_%.2f_%s_%4.2f_%4s.pickle' % (logAge, filterName, AKs,
                                                 str(distance).zfill(4))

    c = constants

    # Get solar mettalicity models for a population at a specific age.
    evol = evolution.get_merged_isochrone(logAge=logAge)

    # Lets do some trimming down to get rid of repeat masses or 
    # mass resolutions higher than 1/1000. We will just use the first
    # unique mass after rounding by the nearest 0.001.
    mass_rnd = np.round(evol.mass, decimals=2)
    tmp, idx = np.unique(mass_rnd, return_index=True)

    mass = evol.mass[idx]
    logT = evol.logT[idx]
    logg = evol.logg[idx]
    logL = evol.logL[idx]
    isWR = logT != evol.logT_WR[idx]

    temp = 10**logT

    # Output magnitudes for each temperature and extinction value.
    mag = np.zeros(len(temp), dtype=float)

    filter = synthetic.filters[filterName]
    flux0 = synthetic.filter_flux0[filterName]
    mag0 = synthetic.filter_mag0[filterName]

    # Make reddening
    red = synthetic.redlaw.reddening(AKs).resample(filter.wave)
    
    # For each temperature extract the synthetic photometry.
    for ii in range(len(temp)):
        gravity = logg[ii]
        L = 10**(logL[ii]) * c.Lsun # luminosity in erg/s
        T = temp[ii]  # in Kelvin
        # Get the radius
        R = math.sqrt(L / (4.0 * math.pi * c.sigma * T**4))   # in cm
        R /= (c.cm_in_AU * c.AU_in_pc)   # in pc

        # Get the atmosphere model now. Wavelength is in Angstroms
        star = atmospheres.get_merged_atmosphere(temperature=T, 
                                                  gravity=gravity)

        # Trim wavelength range down to JHKL range (1.0 - 4.25 microns)
        star = spectrum.trimSpectrum(star, 10000, 42500)

        # Convert into flux observed at Earth (unreddened)
        star *= (R / distance)**2  # in erg s^-1 cm^-2 A^-1

        # ----------
        # Now to the filter integrations
        # ----------
        mag[ii] = synthetic.mag_in_filter(star, filter, red, flux0, mag0)

        print(('M = %7.3f Msun   T = %5d K   R = %2.1f Rsun   logg = %4.2f   mag = %4.2f' % \
            (mass[ii], T, R * c.AU_in_pc / c.Rsun, logg[ii], mag[ii])))


    iso = objects.DataHolder()
    iso.M = mass
    iso.T = temp
    iso.logg = logg
    iso.logL = logL
    iso.mag = mag
    iso.isWR = isWR
    
    _out = open(outFile, 'wb')
    pickle.dump(iso.M, _out)
    pickle.dump(iso.T, _out)
    pickle.dump(iso.logg, _out)
    pickle.dump(iso.logL, _out)
    pickle.dump(iso.mag, _out)
    pickle.dump(iso.isWR, _out)
    _out.close()


def load_osiris_field_areas():
    # Load up the areas for each field
    maskDir = '/u/jlu/work/gc/imf/extinction/'

    # Load up the spectroscopic database to get the field-of-view definitions.
    connection = sqlite.connect(database)
    cur = connection.cursor()

    # Select all the OSIRIS fields-of-view
    sql = 'select name from fields where short_name !=""'
    cur.execute(sql)
    fields = cur.fetchall()
    numFields = len(fields)

    area = {}

    for ff in range(numFields):
        fieldName = str(fields[ff][0])
        if 'Imaging' in fieldName:
            continue

        fieldSuffix = fieldName.replace(' ', '')
        maskFile = maskDir + 'nirc2_mask_' + fieldSuffix + '.fits'
        
        area[fieldName] = pyfits.getdata(maskFile).sum() * 0.01**2

    return area
    

def test_kp_ks_conversion(logAge=9.70):
    """
    For photometric calibration of the GC data, I am going to use the 
    Schoedel et al. 2010 starlist. However, his work uses the Ks filter while
    we use the Kp filter. I need to convert between the two. Rather
    than take a fixed temperature and extinction, I can make a map
    of H-Ks vs. Kp-Ks for the range of extinctions and temperatures
    I expect to be in our calibrator list. This code creates some
    plots in support of this decision.
    """
    dir = '/u/jlu/work/gc/imf/photometry/'

    ageStr = '%.2f' % logAge
    ageStr2 = '%3d' % (int(round(logAge*100)))
    oldSynFile = dir + 'syn_nir_d08000_a' + ageStr2 + '.dat'
    
    # Load up the synthetic data generated at a distance of 
    # 8 kpc and for a population with an age of 10**9.35 years.
    T, AKs, J, H, K, Kp, Ks, Lp, m, g, l = synthetic.load_nearIR(oldSynFile)

    # From Schoedel et al. 2010, the range of extinctions is given by a
    # gaussian with mean AKs = 2.74 and stddev = 0.3. Our synthetic
    # photometry samples AKs in steps of 0.1 so we will take the range 
    # 2.4 <= AKs <= 3.0.
    adx = np.where((AKs >= 2.4) & (AKs <= 3.0))[0]
    
    AKs = AKs[adx]
    J = J[:,adx]
    H = H[:,adx]
    K = K[:,adx]
    Kp = Kp[:,adx]
    Ks = Ks[:,adx]
    Lp = Lp[:,adx]

    # Lets make a color scale for our range of AKs for plotting.
    colorNorm = matplotlib.colors.Normalize(AKs)

    # First lets plot the full range of temperatures and magnitudes
    # to show what matches to the calibrators at Ks. Recall that our
    # photometric calibrators have H < 16.5, Ks < 14.5, Lp < 13.5.
    py.clf()
    pltH = py.semilogx(T, H, color='b', label='H')
    pltK = py.semilogx(T, Ks, color='g', label='Ks')
    pltL = py.semilogx(T, Lp, color='r', label='Lp')
    py.legend((pltH[0], pltK[0], pltL[0]), ('H', 'Ks', 'Lp'), loc='lower left')

    # loc = matplotlib.ticker.MultipleLocator(2000)
    # py.gca().xaxis.set_major_locator( loc )

    # Plot the limits for the photometric calibrators
    py.axhline(y=18, color='blue', linestyle='--')
    py.axhline(y=16, color='green', linestyle='--')
    py.axhline(y=15, color='red', linestyle='--')

    py.xlim(35000, 2500)
    py.ylim(21, 9)
    py.xlabel('Effective Temperature (K)')
    py.ylabel('Magnitude')
    py.title('d=8 kpc, 2.4<=AKs<=3.0')
    py.savefig(dir + 'test_kp_ks/temp_vs_mag_' + ageStr + '.png')

    # Plot Kp - Ks vs. Teff
    # AKs_2D = AKs.repeat(len(T)).reshape((len(AKs), len(T))).transpose()
    # T_2D = AKs.repeat(len(AKs)).reshape((len(AKs), len(T))).transpose()
    py.clf()
    py.semilogx(T, Kp-Ks, marker='.', linestyle='none')
    py.xlim(35000, 2500)
    py.xlabel("Effective Temperature (K)")
    py.ylabel("Kp - Ks (mag)")
    py.savefig(dir + 'test_kp_ks/temp_vs_kp_ks_' + ageStr + '.png')

    ##########
    # Temperature range for calibrators (inclusive)
    ##########
    T_range = [2500, 55000]
    AKs_idx = np.where(AKs == 2.7)[0]

    # Lets trim down
    # tdx = np.where((T >= T_range[0]) & (T <= T_range[1]) & (Ks[:, AKs_idx] < 20))[0]
    tdx = np.where((T >= T_range[0]) & (T <= T_range[1]) &
                   (Ks[:, AKs_idx] > 14) & (Ks[:, AKs_idx] < 17))[0]
    T = T[tdx]
    J = J[tdx,:]
    H = H[tdx,:]
    K = K[tdx,:]
    Kp = Kp[tdx,:]
    Ks = Ks[tdx,:]
    Lp = Lp[tdx,:]

    # Plot up the color-magnitude diagrams and compare with 
    # Schoedel et al. (2010).
    AKs_2D = AKs.repeat(len(T)).reshape((len(AKs), len(T))).transpose()

    # H-Ks vs. Ks
    py.clf()
    py.scatter(H-Ks, Ks, c=AKs_2D.flatten(), cmap=py.cm.jet, edgecolor='none')
    cbar = py.colorbar(orientation='vertical', fraction=0.2)
    cbar.set_label(r'A$_{Ks}$')

    rng = py.axis()
    py.xlim(0, 3)
    py.ylim(rng[3], rng[2])
    py.xlabel('H - Ks')
    py.ylabel('Ks')
    py.title('AKs=[2.4:3.0] T=[%d:%d]' % (T_range[0], T_range[1]))
    py.savefig(dir + 'test_kp_ks/cmd_h_ks_' + ageStr + '.png')

    # Ks-Lp vs. Lp
    py.clf()
    py.scatter(Ks-Lp, Lp, c=AKs_2D.flatten(), cmap=py.cm.jet, edgecolor='none')
    cbar = py.colorbar(orientation='vertical', fraction=0.2)
    cbar.set_label(r'A$_{Ks}$')

    rng = py.axis()
    py.xlim(0, 2)
    py.ylim(rng[3], rng[2])
    py.xlabel('Ks - Lp')
    py.ylabel('Lp')
    py.title('AKs=[2.4:3.0] T=[%d:%d]' % (T_range[0], T_range[1]))
    py.savefig(dir + 'test_kp_ks/cmd_ks_lp_' + ageStr + '.png')

    
    # Plot up the relationship between H-Ks and Kp-Ks.. same for Ks-Lp.
    # In principle, we can use this relationship directly as long as any
    # scatter for this simulated stellar population is less than 1%
    # First, Lets fit a line to each relation
    hks = (H - Ks).flatten()
    kpks = (Kp - Ks).flatten()
    kslp = (Ks - Lp).flatten()
    hks_coeffs = np.polyfit(hks, kpks, 1)
    kslp_coeffs = np.polyfit(kslp, kpks, 1)

    hks_idx = hks.argsort()
    kslp_idx = kslp.argsort()

    hks_fit = np.polyval(hks_coeffs, hks[hks_idx])
    kslp_fit = np.polyval(kslp_coeffs, kslp[kslp_idx])

    # H-Ks vs. Kp-Ks
    py.clf()
    py.scatter(H-Ks, Kp-Ks, c=AKs_2D.flatten(), cmap=py.cm.jet, 
               edgecolor='none')
    py.plot(hks[hks_idx], hks_fit, 'k--')

    cbar = py.colorbar(orientation='vertical', fraction=0.2)
    cbar.set_label(r'A$_{Ks}$')
    py.xlabel('H - Ks')
    py.ylabel('Kp - Ks')
    py.title('Kp-Ks = %.4f + %.4f * H-Ks' % (hks_coeffs[1], hks_coeffs[0]),
             fontsize=14)
    py.savefig(dir+'test_kp_ks/color_HKs_KpKs_' + ageStr + '.png')
    py.savefig(dir+'test_kp_ks/color_HKs_KpKs_' + ageStr + '.eps')

    print(('Best fit Kp-Ks = %.5f + %.5f * H-Ks' % \
        (hks_coeffs[1], hks_coeffs[0])))
    print(('Residuals from H-Ks vs. Kp-Ks fit: %.4f' % \
        (hks_fit - kpks[hks_idx]).std()))
    print('')

    # Ks-Lp vs. Kp-Ks
    py.clf()
    py.scatter(Ks-Lp, Kp-Ks, c=AKs_2D.flatten(), cmap=py.cm.jet, 
               edgecolor='none')
    py.plot(kslp[kslp_idx], kslp_fit, 'k--')

    cbar = py.colorbar(orientation='vertical', fraction=0.2)
    cbar.set_label(r'A$_{Ks}$')
    py.xlabel('Ks - Lp')
    py.ylabel('Kp - Ks')
    py.title('Kp-Ks = %.4f + %.4f * Ks-Lp' % (kslp_coeffs[1], kslp_coeffs[0]),
             fontsize=14)
    py.savefig(dir+'test_kp_ks/color_KsLp_KpKs_' + ageStr + '.png')

    print(('Best fit Kp-Ks = %.5f + %.5f * Ks-Lp' % \
        (kslp_coeffs[1], kslp_coeffs[0])))
    print(('Residuals from Ks-Lp vs. Kp-Ks fit: %.4f' % \
        (kslp_fit - kpks[kslp_idx]).std()))
    print('')

    # Lets do the same for Kp vs. K
    hkp = (H - Kp).flatten()
    kpk = (Kp - K).flatten()
    kplp = (Kp - Lp).flatten()
    hkp_coeffs = np.polyfit(hkp, kpk, 1)
    kplp_coeffs = np.polyfit(kplp, kpk, 1)

    hkp_idx = hkp.argsort()
    kplp_idx = kplp.argsort()

    hkp_fit = np.polyval(hkp_coeffs, hkp[hkp_idx])
    kplp_fit = np.polyval(kplp_coeffs, kplp[kplp_idx])

    # H-Kp vs. Kp-K
    py.clf()
    py.scatter(H-Kp, Kp-K, c=AKs_2D.flatten(), cmap=py.cm.jet, 
               edgecolor='none')
    py.plot(hkp[hkp_idx], hkp_fit, 'k--')

    cbar = py.colorbar(orientation='vertical', fraction=0.2)
    cbar.set_label(r'A$_{Ks}$')
    py.xlabel('H - Kp')
    py.ylabel('Kp - K')
    py.title('Kp-K = %.4f + %.4f * H-Kp' % (hkp_coeffs[1], hkp_coeffs[0]),
             fontsize=14)
    py.savefig(dir+'test_kp_ks/color_HKp_KpK_' + ageStr + '.png')

    print(('Best fit Kp-K = %.5f + %.5f * H-Kp' % \
        (hkp_coeffs[1], hkp_coeffs[0])))
    print(('Residuals from H-Kp vs. Kp-K fit: %.4f' % \
        (hkp_fit - kpk[hkp_idx]).std()))
    print('')

    # Kp-Lp vs. Kp-K
    py.clf()
    py.scatter(Kp-Lp, Kp-K, c=AKs_2D.flatten(), cmap=py.cm.jet, 
               edgecolor='none')
    py.plot(kplp[kplp_idx], kplp_fit, 'k--')

    cbar = py.colorbar(orientation='vertical', fraction=0.2)
    cbar.set_label('AKs')
    py.xlabel('Kp - Lp')
    py.ylabel('Kp - K')
    py.title('Kp-K = %.4f + %.4f * Kp-Lp' % (kplp_coeffs[1], kplp_coeffs[0]),
             fontsize=14)
    py.savefig(dir+'test_kp_ks/color_KpLp_KpK_' + ageStr + '.png')

    print(('Best fit Kp-K = %.5f + %.5f * Kp-Lp' % \
        (kplp_coeffs[1], kplp_coeffs[0])))
    print(('Residuals from Kp-Lp vs. Kp-K fit: %.4f' % \
        (kplp_fit - kpk[kplp_idx]).std()))

    
def make_calib_schoedel2010():
    sch = atpy.Table('/u/ghezgroup/data/gc/source_list/schoedel2010_extinction.vot')
    print((len(sch), len(sch.columns)))

    ##################################################
    # Now pull out only those sources with the BEST photometric data.
    # Decided limits based on figure 3 from Schoedel et al. 2010.
    ##################################################
    h_limit = 18
    ks_limit = 16
    lp_limit = 15
    he_limit = 0.03
    kse_limit = 0.02
    lpe_limit = 0.04

    trim = sch.where((sch['Hmag'] <= h_limit) & 
                     (sch['Hmag'] > 0) & 
                     (sch['e_Hmag'] <= he_limit) & 
                     (sch['Ksmag'] <= ks_limit) & 
                     (sch['Ksmag'] > 0) & 
                     (sch['e_Ksmag'] <= kse_limit) &
                     (sch['Lpmag'] <= lp_limit) & 
                     (sch['Lpmag'] > 0) & 
                     (sch['e_Lpmag'] <= lpe_limit))

    print((len(trim), len(trim.columns)))

    # Lets also make a nearest neighbor cut. Throw out stars that have
    # any star 
    nrows = len(trim)
    x2d = trim['oRA'].reshape((nrows, 1)).repeat(nrows, 1)
    y2d = trim['oDE'].reshape((nrows, 1)).repeat(nrows, 1)
    m2d = trim['Ksmag'].reshape((nrows, 1)).repeat(nrows, 1)

    dx = x2d - x2d.transpose()
    dy = y2d - y2d.transpose()
    dm = m2d - m2d.transpose()

    dr = np.hypot(dx, dy)

    # Limits for dr and dm
    dr_limit = 0.3
    dm_limit = -1.5

    use = []
    for ii in range(nrows):
        idx = np.where((dr[:,ii] != 0) & 
                       (dr[:,ii] <= dr_limit) & 
                       (dm >= dm_limit))[0]

        if len(idx) == 0:
            if trim['Ksmag'][ii] <= (ks_limit + dm_limit):
                use.append(ii)

    final = trim.rows(use)
    print(('Final Number of Calibrators: %d (out of %d)' % (len(final), len(sch))))

    ##################################################
    # We need to convert from Ks to Kp. Conversions calculated in
    #   /u/jlu/code/python/jlu/papers/jlu_gc_imf.py
    #   test_kp_ks_conversion().
    # The conversion equation uses H-Ks to determine a Kp-Ks:
    #   (Kp - Ks) = 0.02414 + 0.00482 * (H - Ks)
    ##################################################
    Kp = convert_Ks_Kp_calib(final['Ksmag'], H=final['Hmag'])

    final.add_column('Kpmag', Kp, unit='mag', format='%.2f')

    final.write('/u/jlu/work/gc/imf/photometry/schoedel2010_calibrators.vot')

    return final

def plot_calibrators():
    final = atpy.Table('/u/jlu/work/gc/imf/photometry/schoedel2010_calibrators.vot')

    # Plot up the distribution of sources.
    sizes = 2*10**6 * (1.0 / final['Ksmag'])**4
    
    mlegend = np.array([9, 12, 14, 16])
    xlegend = np.array([-20, -20, -20, -20])
    ylegend = np.array([25, 20, 15, 10])
    tlegend = np.array(['Ks=9', 'Ks=13', 'Ks=16', 'Ks=19'])
    slegend = 2*10**6 * (1.0 / mlegend)**4
    
    py.clf()
    py.scatter(final['oRA'], final['oDE'], s=sizes, facecolors='none')
    py.scatter(xlegend, ylegend, s=slegend, edgecolors='red', facecolors='none')
    for ii in range(len(mlegend)):
        py.text(xlegend[ii]-1, ylegend[ii], tlegend[ii], color='red')
    rng = py.axis()
    py.xlim(rng[1], rng[0])
    py.savefig('schoedel_calibs_xy.png')

    bins = np.arange(7, 17, 1)
    py.clf()
    py.hist(final['Hmag'], color='blue', alpha=0.5, bins=bins)
    py.hist(final['Ksmag'], color='red', alpha=0.5, bins=bins)
    py.hist(final['Hmag'], color='blue', alpha=0.5, bins=bins)
    py.xlabel('Ks (mag)')
    py.ylabel('H (mag)')
    py.savefig('/u/jlu/work/gc/imf/photometry/schoedel_calibs_hist_hkl.png')

def name_calibrators():
    photoDir = '/u/jlu/work/gc/imf/photometry/'
    final = atpy.Table(photoDir + 'schoedel2010_calibrators.vot')

    ##################################################
    # Match with our names. We do this by creating two new
    # starfinder-formatted starlists and then aligning them. This
    # should match up sources and we can then extract names.
    ##################################################
    label = starTables.Labels()
    
    labelList = starTables.StarfinderList(None, hasErrors=False)
    schodList = starTables.StarfinderList(None, hasErrors=False)

    t0 = 2006.33

    labelList.name = label.name
    labelList.mag = label.mag
    labelList.epoch = np.zeros(len(label.x), dtype=float) + t0
    labelList.x = (label.x + label.vx * (label.t0 - t0) / 10**3) * -1.0
    labelList.y = (label.y + label.vy * (label.t0 - t0) / 10**3)
    labelList.snr = np.ones(len(label.x))
    labelList.corr = np.ones(len(label.x))
    labelList.nframes = np.ones(len(label.x))
    labelList.counts = np.ones(len(label.x))
    
    schoedelNames = []
    for ii in range(len(final)):
        schoedelNames.append('star_sch%s' % str(final['Seq'][ii]).zfill(4))
    schoedelNames = np.array(schoedelNames)

    schodList.name = schoedelNames
    schodList.mag = final['Kpmag']
    schodList.h = final['Hmag']
    schodList.lp = final['Lpmag']
    schodList.x = final['oRA'] * -1.0 # convert to +xwest
    schodList.y = final['oDE']
    schodList.epoch = np.zeros(len(schodList.x), dtype=float) + t0
    schodList.snr = np.ones(len(schodList.x))
    schodList.corr = np.ones(len(schodList.x))
    schodList.nframes = np.ones(len(schodList.x))
    schodList.counts = np.ones(len(schodList.x))

    # We will need to match two stars at the top of the list. Find
    # the mannually matched stars by mag and position.
    first2 = ['irs16C', 'irs16NW', 'irs16NE', 'irs16CC', 'S2-25', 'S2-17']
    first2 = first2[::-1]

    # Find the stars in both lists and shift them to the top.
    for ii in range(len(first2)):
        ldx = np.where(labelList.name == first2[ii])[0]
        dr = np.hypot(schodList.x - labelList.x[ldx],
                      schodList.y - labelList.y[ldx])
        sdx = dr.argsort()[0]
        schodList.name[sdx] = first2[ii]

        shiftToTop(labelList, ldx, alsoHLp=False)
        shiftToTop(schodList, sdx, alsoHLp=True)
    
    labelList.saveToFile(photoDir + 'name_schoedel/label.lis')
    schodList.saveToFile(photoDir + 'name_schoedel/schoedel.lis')

    # Now align the two starlists
    alignRoot = photoDir + 'name_schoedel/align'
    _align = open(alignRoot + '.list', 'w')
    _align.write(photoDir + 'name_schoedel/label.lis 20 ref\n')
    _align.write(photoDir + 'name_schoedel/schoedel.lis 20\n')
    _align.close()

    cmd = 'java align -r %s -p -a 0 -R 0.12 %s.list' % (alignRoot, alignRoot)
    os.system(cmd)

    s = starset.StarSet(alignRoot)
    ourName = np.array(s.getArrayFromEpoch(0, 'name'))
    schName = np.array(s.getArrayFromEpoch(1, 'name'))
    ourx = s.getArrayFromEpoch(0, 'xpix')
    oury = s.getArrayFromEpoch(0, 'ypix')
    schx = s.getArrayFromEpoch(1, 'xpix')
    schy = s.getArrayFromEpoch(1, 'ypix')
    cnt = s.getArray('velCnt')

    for ii in range(len(schodList.x)):
        idx = np.where(schodList.name[ii] == schName)[0][0]
        
        if cnt[idx] == 2:
            schodList.name[ii] = ourName[idx]
            schodList.x[ii] = ourx[idx]
            schodList.y[ii] = oury[idx]
        else:
            schodList.x[ii] = schx[idx]
            schodList.y[ii] = schy[idx]
            

    ##########
    # Write a photo_calib.dat file
    ##########
    _photo = open(photoDir + 'photo_calib_schoedel2010.dat', 'w')
    
    # First thing first, we need to setup the headers.
    _photo.write('## Columns: Format of this header is hardcoded for read in')
    _photo.write('by calibrate.py\n')
    _photo.write('## Field separator in each column is "--". ')
    _photo.write('Default calibrators are listed after\n')
    _photo.write('## each magnitude column header entry.\n')
    _photo.write('# 1 -- Star Name\n')
    _photo.write('# 2 -- X position (arcsec, increasing to the East\n')
    _photo.write('# 3 -- Y position (arcsec)\n')
    _photo.write('# 4 -- Variable? flag\n')
    _photo.write('# 5 -- H band (Schoedel et al. 2010)\n')
    _photo.write('# 6 -- Kp band (Schoedel et al. 2010)\n')
    _photo.write('# 7 -- Lp band (Schoedel et al. 2010)\n')

    for ii in range(len(schodList.x)):
        _photo.write('%-13s %7.3f %7.3f  0   %5.2f  %5.2f  %5.2f\n' % 
                     (schodList.name[ii], schodList.x[ii]*-1.0, schodList.y[ii],
                      schodList.h[ii], schodList.mag[ii], schodList.lp[ii]))
    _photo.close()

def shiftToTop(list, idx, hasErrors=False, alsoHLp=True):
    name = list.name[idx]
    mag = list.mag[idx]
    x = list.x[idx]
    y = list.y[idx]
    epoch = list.epoch[idx]
    snr = list.snr[idx]
    corr = list.corr[idx]
    nframes = list.nframes[idx]
    counts = list.counts[idx]
                      
    if hasErrors:
        xerr = list.xerr[idx]
        yerr = list.yerr[idx]
        
    if alsoHLp:
        h = list.h[idx]
        lp = list.lp[idx]
        
    list.name = np.delete(list.name, idx)
    list.mag = np.delete(list.mag, idx)
    list.x = np.delete(list.x, idx)
    list.y = np.delete(list.y, idx)
    list.epoch = np.delete(list.epoch, idx)
    list.snr = np.delete(list.snr, idx)
    list.corr = np.delete(list.corr, idx)
    list.nframes = np.delete(list.nframes, idx)
    list.counts = np.delete(list.counts, idx)
    if hasErrors:
        list.xerr = np.delete(list.xerr, idx)
        list.yerr = np.delete(list.yerr, idx)
        
    if alsoHLp:
        list.h = np.delete(list.h, idx)
        list.lp = np.delete(list.lp, idx)
        
    list.name = np.insert(list.name, 0, name)
    list.mag = np.insert(list.mag, 0, mag)
    list.x = np.insert(list.x, 0, x)
    list.y = np.insert(list.y, 0, y)
    list.epoch = np.insert(list.epoch, 0, epoch)
    list.snr = np.insert(list.snr, 0, snr)
    list.corr = np.insert(list.corr, 0, corr)
    list.nframes = np.insert(list.nframes, 0, nframes)
    list.counts = np.insert(list.counts, 0, counts)
    if hasErrors:
        list.xerr = np.insert(list.xerr, 0, xerr)
        list.yerr = np.insert(list.yerr, 0, yerr)
    if alsoHLp:
        list.h = np.insert(list.h, 0, h)
        list.lp = np.insert(list.lp, 0, lp)


def convert_Ks_Kp_calib(Ks, H=None, Lp=None):
    if H == None and Lp == None:
        print('Must specify either H or Lp to convert Ks to Kp')

    if H != None:
        Kp = 0.00639 + 0.01056 * (H - Ks) + Ks
        #Kp = 0.02414 + 0.00482 * (H - Ks) + Ks
    else:
        Kp = 0.00800 + 0.01374 * (Ks - Lp) + Ks
        #Kp = 0.03126 + 0.00799 * (Ks - Lp) + Ks

    return Kp

def convert_Kp_K_calib(Kp, H=None, Lp=None):
    if H == None and Lp == None:
        print('Must specify either H or Lp to convert Ks to Kp')

    if H != None:
        K = Kp - (-0.00048 + 0.06600 * (H - Kp))
        #K = Kp - (0.06966 + -0.01069 * (H - Kp))
    else:
        K = Kp - (0.00289 + 0.08747 * (Kp - Kp))
        #K = Kp - (0.08350 + 0.00007 * (Kp - Lp))

    return K

    

def compare_calibrations_blum1996():
    """
    Compare the magnitudes of the Schoedel et al. calibrators
    with some previous photometric calibrations.
    """
    photoDir = '/u/jlu/work/gc/imf/photometry/'

    ##################################################
    # Compare Schoedel et al. (2010) photometric 
    # calibrators from our Blum list in photo_calib.dat
    ##################################################
    photoOldFile = '/u/ghezgroup/data/gc/source_list/photo_calib.dat'
    photoNewFile = '/u/jlu/work/gc/imf/photometry/photo_calib_schoedel2010.dat'

    photoOld = asciidata.open(photoOldFile)
    photoNew = asciidata.open(photoNewFile)

    nameOld = photoOld[0].tonumpy()
    nameNew = photoNew[0].tonumpy()

    k_old = photoOld[4].tonumpy()
    kp_new = photoNew[5].tonumpy()
    h_new = photoNew[4].tonumpy()

    k_new = convert_Kp_K_calib(kp_new, h_new)

    idxNew = []
    idxOld = []
    for ii in range(len(nameOld)):
        # Skip stars without photometry in Blum et al. 1996
        if k_old[ii] == 0:
            continue

        # match by name
        idx = np.where(nameNew == nameOld[ii])[0]

        # If we didn't match, add 'irs' and try again.
        if (len(idx) == 0 and 
            (nameOld[ii].startswith('irs') == False) and 
            (nameOld[ii].startswith('S') == False)):

            idx = np.where(nameNew == 'irs'+nameOld[ii])[0]

        if len(idx) == 0:
            print(('Could not find a match for Blum source: %s' % nameOld[ii]))
            continue

        idxOld.append(ii)
        idxNew.append(idx[0])


    k_old = k_old[idxOld]
    k_new = k_new[idxNew]
    kp_new = kp_new[idxNew]
    nameOld = nameOld[idxOld]
    nameNew = nameNew[idxNew]

    py.clf()
    py.plot(k_new, k_old - k_new, 'k.')
    py.xlabel('K from Schoedel (mag)')
    py.ylabel('K Blum - Schoedel (mag)')
    for ii in range(len(k_new)):
        py.text(k_new[ii]+0.05, k_old[ii] - k_new[ii], nameNew[ii],
                fontsize=10)
    rng = py.axis()
    py.plot([rng[0], rng[1]], [0, 0], 'k--')
    catFind = CatalogFinder.CatalogFinder(k_new, k_old - k_new, nameNew)
    py.connect('button_release_event', catFind)
    py.savefig(photoDir + 'name_schoedel/compare_blum1996.png')
        

def compare_calibrations_rafelski2007():
    """
    Compare the magnitudes of the Schoedel et al. calibrators
    with some previous photometric calibrations.
    """
    photoDir = '/u/jlu/work/gc/imf/photometry/'

    ##################################################
    # Compare Schoedel et al. (2010) photometric 
    # calibrators from our Blum list in photo_calib.dat
    ##################################################
    photoOldFile = '/u/ghezgroup/data/gc/source_list/photo_calib.dat'
    photoNewFile = '/u/jlu/work/gc/imf/photometry/photo_calib_schoedel2010.dat'

    photoOld = asciidata.open(photoOldFile)
    photoNew = asciidata.open(photoNewFile)

    nameOld = photoOld[0].tonumpy()
    nameNew = photoNew[0].tonumpy()

    k_old = photoOld[9].tonumpy()
    kp_new = photoNew[5].tonumpy()
    h_new = photoNew[4].tonumpy()

    k_new = convert_Kp_K_calib(kp_new, h_new)

    idxNew = []
    idxOld = []
    for ii in range(len(nameOld)):
        # Skip stars without photometry in Rafelski et al. 2007
        if k_old[ii] == 0:
            continue

        # match by name
        idx = np.where(nameNew == nameOld[ii])[0]

        # If we didn't match, add 'irs' and try again.
        if (len(idx) == 0 and 
            (nameOld[ii].startswith('irs') == False) and 
            (nameOld[ii].startswith('S') == False)):

            idx = np.where(nameNew == 'irs'+nameOld[ii])[0]

        if len(idx) == 0:
            print(('Could not find a match for Rafelski source: %s' % nameOld[ii]))
            continue

        idxOld.append(ii)
        idxNew.append(idx[0])


    k_old = k_old[idxOld]
    k_new = k_new[idxNew]
    kp_new = kp_new[idxNew]
    nameOld = nameOld[idxOld]
    nameNew = nameNew[idxNew]

    py.clf()
    py.subplots_adjust(left=0.18)
    py.plot(k_new, k_old - k_new, 'k.')
    py.xlabel('K from Schoedel (mag)')
    py.ylabel('K Rafelski - Schoedel (mag)')
    for ii in range(len(k_new)):
        py.text(k_new[ii]+0.05, k_old[ii] - k_new[ii], nameNew[ii],
                fontsize=10)
    rng = py.axis()
    py.plot([rng[0], rng[1]], [0, 0], 'k--')
    catFind = CatalogFinder.CatalogFinder(k_new, k_old - k_new, nameNew)
    py.connect('button_release_event', catFind)
    py.savefig(photoDir + 'name_schoedel/compare_rafelski2007.png')

def compare_calibrations_oldKeck():
    """
    Compare the magnitudes of the Schoedel et al. calibrators
    with some previous photometric calibrations.
    """
    photoDir = '/u/jlu/work/gc/imf/photometry/'

    ##################################################
    # Compare Schoedel et al. (2010) photometric 
    # calibrators from our Rafelski extrapolated list in photo_calib.dat
    ##################################################
    photoOldFile = '/u/ghezgroup/data/gc/source_list/photo_calib.dat'
    photoNewFile = '/u/jlu/work/gc/imf/photometry/photo_calib_schoedel2010.dat'

    photoOld = asciidata.open(photoOldFile)
    photoNew = asciidata.open(photoNewFile)

    nameOld = photoOld[0].tonumpy()
    nameNew = photoNew[0].tonumpy()

    kp_old = photoOld[5].tonumpy()
    kp_new = photoNew[5].tonumpy()

    idxNew = []
    idxOld = []
    for ii in range(len(nameOld)):
        # Skip stars without photometry in Rafelski et al. 2007
        if kp_old[ii] == 0:
            continue

        # match by name
        idx = np.where(nameNew == nameOld[ii])[0]

        # If we didn't match, add 'irs' and try again.
        if (len(idx) == 0 and 
            (nameOld[ii].startswith('irs') == False) and 
            (nameOld[ii].startswith('S') == False)):

            idx = np.where(nameNew == 'irs'+nameOld[ii])[0]

        if len(idx) == 0:
            print(('Could not find a match for Rafelski source: %s' % nameOld[ii]))
            continue

        idxOld.append(ii)
        idxNew.append(idx[0])


    kp_old = kp_old[idxOld]
    kp_new = kp_new[idxNew]
    nameOld = nameOld[idxOld]
    nameNew = nameNew[idxNew]

    py.clf()
    py.subplots_adjust(left=0.18)
    py.plot(kp_new, kp_old - kp_new, 'k.')
    py.xlabel('Kp from Schoedel (mag)')
    py.ylabel('Kp Rafelski 2nd - Schoedel (mag)')
    for ii in range(len(kp_new)):
        py.text(kp_new[ii]+0.05, kp_old[ii] - kp_new[ii], nameNew[ii],
                fontsize=10)
    rng = py.axis()
    py.plot([rng[0], rng[1]], [0, 0], 'k--')
    catFind = CatalogFinder.CatalogFinder(kp_new, kp_old - kp_new, nameNew)
    py.connect('button_release_event', catFind)
    py.savefig(photoDir + 'name_schoedel/compare_rafelski_secondary.png')
        

def plot_model_klf_100_400_Myr():
    logAgeList = [8.00, 8.05, 8.10, 8.14, 8.19, 8.25, 8.30, 8.35,
                  8.39, 8.44, 8.50, 8.55, 8.60]
    plot_model_klf(logAgeList=logAgeList, suffix='_100_400_Myr')

def plot_model_klf_2_10_Myr():
    logAgeList = [6.30, 6.34, 
                  6.40, 6.44, 6.50, 6.55, 6.59, 6.65, 6.69, 6.75,
                  6.80, 6.84, 6.90, 6.94, 7.00]
    plot_model_klf(logAgeList=logAgeList, suffix='_2_10_Myr')

def plot_model_klf(logAgeList=[6.80], suffix=None):
    #imfSlope=1.35
    imfSlope=1.7

    """
    Make a plot of several model KLFs all together.
    """
    # Reddening
    aV = 27.0
    RV = 2.9

    aKs = theAKs
    aJ = extinction.nishiyama09(1.248, aKs)
    aH = extinction.nishiyama09(1.6330, aKs)
    aKp = extinction.nishiyama09(2.1245, aKs)
    aK = extinction.nishiyama09(2.196, aKs)
    aKs = extinction.nishiyama09(2.146, aKs)

    dist = 8000.0
    distMod = -5.0 + 5.0 * math.log10(dist)

    py.figure(1)
    py.clf()
    
    py.figure(2)
    py.clf()

    for aa in range(len(logAgeList)):
        print(('Working on logAge = %.2f' % logAgeList[aa]))

        logAgeString = '0%d' % (logAgeList[aa] * 100)

        # Read in Geneva tracks
        genevaFile = '/u/jlu/work/models/geneva/iso/020/c/'
        genevaFile += 'iso_c020_' + logAgeString + '.UBVRIJHKLM'
        model = asciidata.open(genevaFile)
        modMass = model[1].tonumpy()
        modTeff = model[3].tonumpy()
        modV = model[6].tonumpy()
        modVK = model[11].tonumpy()
        modHK = model[15].tonumpy()
        modJLp = model[19].tonumpy()
        modJK = model[17].tonumpy()

        modK = modV - modVK
        modH = modK + modHK
        modJ = modK + modJK
        modLp = modJ - modJLp
        modKs = modK + 0.002 + 0.026 * (modJK)
        modKp = modK + 0.22 * (modHK)

        modK_extinct = modK + aK + distMod
        modKp_extinct = modKp + aKp + distMod
        modKs_extinct = modKs + aKs + distMod

        # Sample an IMF randomly from 2-30 Msun
        minMass = 3.0
        randMass = plfit.plexp_inv(np.random.rand(10**5), minMass, imfSlope)
        randKp = np.zeros(len(randMass), dtype=float)
        randTeff = np.zeros(len(randMass), dtype=float)
    
        for ii in range(len(randMass)):
            dm = np.abs(modMass - randMass[ii])
            mdx = dm.argmin()

            # Get rid of stuff that we don't have models for
            if dm[mdx] > 100:
                continue

            randKp[ii] = modKp_extinct[mdx]
            randTeff[ii] = modTeff[mdx]

        # Get rid of the bad ones
        idx = np.where(randKp != 0)[0]
        cdx = np.where(randKp == 0)[0]
    
        randMass = randMass[idx]
        randKp = randKp[idx]
        randTeff = randTeff[idx]
        randTeff = 10**randTeff

        # Plot the model mass luminosity relationship
        logAgeLegend = ('%.2f' % logAgeList[aa])

        py.figure(1)
        py.plot(modKp_extinct, modMass, '.', label=logAgeLegend)

        # Plot the HR diagram
        py.figure(2)
        py.semilogx(randTeff, randKp, '.', label=logAgeLegend)
        
    py.figure(1)
    py.legend(title='Log(Age)', numpoints=1, loc='lower left',
              prop={'size':8})
    py.xlabel("Kp Magnitude (AKs = %.2f)" % theAKs)
    py.ylabel("Stellar Mass (Msun)")
    py.title("IMF slope=%.2f" % (imfSlope))
    py.xlim(8, 22)
    py.savefig('/u/jlu/work/gc/imf/klf/models/gc_mass_luminosity'+suffix+'.png')

    py.figure(2)
    py.legend(title='Log(Age)', numpoints=1, loc='lower left',
              prop={'size':8})
    py.xlabel('Temperature (K)')
    py.ylabel('Kp (AKs=%.2f)' % theAKs)
    rng = py.axis()
    py.xlim(50000, 3000)
    py.ylim(22, 8)
    py.title("IMF slope=%.2f" % (imfSlope))
    py.savefig('/u/jlu/work/gc/imf/klf/models/gc_HR_diagram'+suffix+'.png')

def plot_img_completeness():
    """
    Plot the completeness curves for each part of the dp_msc to see 
    how they compare.
    """
    dir = '/u/jlu/work/gc/dp_msc/2010_11_08/completeness/'

    fields = ['C', 'C_NE', 'C_NW', 'C_SE', 'C_SW', 'E', 'N', 'S', 'W',
              'NE', 'NW', 'SE', 'SW']

    colors = ['black', 'brown', 'salmon', 'red', 'orange', 
              'gold', 'greenyellow', 'green', 'cyan', 
              'blue', 'navy', 'purple', 'magenta']

    py.clf()

    for ff in range(len(fields)):
        field = fields[ff] 

        compFile = dir + field + '/kp/align_in_out/completeness.dat'

        tab = asciidata.open(compFile)

        mag = tab[0].tonumpy()
        comp = tab[1].tonumpy()

        py.plot(mag, comp, linestyle='-', label=field, color=colors[ff])

    py.xlabel('Kp Magnitude')
    py.ylabel('Completeness')
    py.ylim(0, 1.1)
    py.legend(loc='lower left', ncol=2, prop={'size':14})
    py.savefig(dir + 'plots/completeness_by_field.png')
    print(('Saving %s' % (dir + 'plots/completeness_by_field.png')))

def envelope_from_star_plant(field):
    """
    Examine the detection thresholds vs. the seperation and brightness
    difference between each real star and each planted star.
    """
    dir = '/u/jlu/work/gc/dp_msc/2010_11_08/completeness/'

    alignDir = dir + field + '/kp/align_in_out/'

    d = starPlant.load_completeness(alignDir)

    # Read in the original starlist for this epoch and get rid of any
    # simulated stars.
    lisFile = '/u/ghezgroup/data/gc/06maylgs1/combo/starfinder/' + \
        'mag06maylgs1_dp_msc_' + field + '_kp_rms.lis'
    lis = starTables.StarfinderList(lisFile, hasErrors=True)

    m_star = lis.mag
    x_star = lis.x
    y_star = lis.y

    # Create some output arrays.
    starsCount = len(m_star)
    plantCount = len(d.x)

    dr = np.zeros((plantCount, starsCount), dtype=float)
    dm_in = np.zeros((plantCount, starsCount), dtype=float)
    dm_out = np.zeros((plantCount, starsCount), dtype=float)
    print(('%d Planted, %d Real, %d pairs' % (plantCount, starsCount, 
                                             plantCount * starsCount)))

    # Loop through the planted star positions
    for ii in range(len(d.x)):
        x_plant = d.x[ii]
        y_plant = d.y[ii]

        # Find the faintest star detected at this position
        m_plant_out_all = d.m_out[d.xy_idx[ii]]
        m_plant_in_all = d.m_in[d.xy_idx[ii]]
        foo = np.isnan(m_plant_out_all) == False
        m_plant_out = m_plant_out_all[foo].max()
        m_plant_in = m_plant_in_all[foo].max()

        # Calculate the seperation between this planted star
        # and every real star.
        dr[ii, :] = np.hypot(x_plant - x_star, y_plant - y_star)
        dm_in[ii, :] = m_star - m_plant_in
        dm_out[ii, :] = m_star - m_plant_out

    # Trim out everything beyond 200 pixels (2")
    idx = np.where((dr < 100) & (dm_in < 0) & (dm_out < 0))
    dr = dr[idx[0], idx[1]]
    dm_in = dm_in[idx[0], idx[1]]
    dm_out = dm_out[idx[0], idx[1]]
        
    # Plotting
    py.clf()
    py.plot(dr, dm_in, 'k.', label='Planted Kp', ms=2)
    py.plot(dr, dm_out, 'r.', label='Recovered Kp', ms=2)
    py.xlabel('Separation (pixels)')
    py.ylabel('Kp (real - planted stars)')
    py.title('Field: %s' % field)
    py.legend(loc='lower left', numpoints=1)
    py.savefig(dir + field + '/kp/envelope_dr_vs_dm.png')
    
    
def mosaic_detection_threshold_map():
    dir = '/u/jlu/work/gc/dp_msc/2010_11_08/completeness/'
    d = load_completeness_mosaic(dir)

    # Calculate the detection threshold for each pixel where
    # we planted a star.
    numPixels = len(d.x)

    detectThresh = np.zeros(numPixels, dtype=float)
    magDeviant = np.zeros(numPixels, dtype=float)
    
    for ii in range(numPixels):
        found = np.where(np.isnan(d.m_out[d.xy_idx[ii]]) == False)[0]

        if len(found) == 0:
            continue
        
        dm = d.m_out[d.xy_idx[ii]][found] - d.m_in[d.xy_idx[ii]][found]

        detectThresh[ii] = d.m_in[d.xy_idx[ii]][found].max()
        magDeviant[ii] = dm[abs(dm).argmax()]

    # ##########
    # Plot a 2D map of the detection thresholds
    # ##########
    py.close(1)
    py.figure(1, figsize=(15,14))
    py.subplots_adjust(left=0.1, right=0.98)
    py.clf()
    py.scatter(d.x, d.y, c=detectThresh, marker='s', 
               cmap=py.cm.gist_stern, edgecolors='none', vmin=9, s=5)
    py.xlabel('X (pixel)')
    py.ylabel('Y (pixel)')
    cbar = py.colorbar()
    cbar.set_label('Detection Threshold (mag)')
    py.axis('equal')
    py.axis([-900, 2000, -900, 2000])
    py.savefig('%s/plots/map_detect_threshold.png' % dir)
    print(('Saving %s/plots/map_detect_threshold.png' % dir))

    # ##########
    # Plot a 2D map of the magnitude deviations
    # ##########
    py.clf()
    py.scatter(d.x, d.y, c=magDeviant, marker='s', 
               cmap=py.cm.gist_stern, edgecolors='none',
               vmin=-0.5, vmax=0.5)
    py.xlabel('X (pixel)')
    py.ylabel('Y (pixel)')
    cbar = py.colorbar()
    cbar.set_label('Max |Output - Input| (mag)')
    py.axis('equal')
    py.axis([-900, 2000, -900, 2000])
    py.savefig('%s/plots/map_dm.png' % dir)
    print(('Saving %s/plots/map_dm.png' % dir))
    
    print('Magnitude Deviations:')
    print(('   max = %.2f' % magDeviant.max()))
    print(('   min = %.2f' % magDeviant.min()))

    # ##########
    # Plot the 1D histograms of magnitude deviations
    # ##########
    py.clf()
    py.subplots_adjust(right=0.95)
    py.hist(magDeviant, histtype='step', align='mid', 
            bins=np.arange(-1.475, 1.0, 0.05))
    py.xlabel('Maximum Magnitude Deviation (mag)')
    py.ylabel('Number of Recovered Stars')
    py.savefig('%s/plots/hist_dm.png' % dir)
    print(('Saving %s/plots/hist_dm.png' % dir))

    

def load_completeness_mosaic(dir='/u/jlu/work/gc/dp_msc/2010_11_08/completeness/'):
    print('Loading star planting data from: mosaic_results.dat')
    _pickleFile = open(dir + 'mosaic_results.dat', 'r')
    
    data = objects.DataHolder()

    # Individual measurements for every planted star
    data.x_in_pix = pickle.load(_pickleFile)
    data.y_in_pix = pickle.load(_pickleFile)
    data.m_in = pickle.load(_pickleFile)
    data.f_in = pickle.load(_pickleFile)

    data.x_out_pix = pickle.load(_pickleFile)
    data.y_out_pix = pickle.load(_pickleFile)
    data.m_out = pickle.load(_pickleFile)
    data.f_out = pickle.load(_pickleFile)

    # Combined for unique magnitude bins.
    data.mag = pickle.load(_pickleFile)
    data.midx = pickle.load(_pickleFile)

    data.countPlanted = pickle.load(_pickleFile)
    data.countFound = pickle.load(_pickleFile)
    data.completeness = pickle.load(_pickleFile)

    # Combined for unique positions on the detector.
    data.x_pix = pickle.load(_pickleFile)
    data.y_pix = pickle.load(_pickleFile)
    data.xy_idx = pickle.load(_pickleFile)

    data.field = pickle.load(_pickleFile)

    _pickleFile.close()
    
    # Load up the label.dat file and use the coordinates in there
    # to astrometrically calibrate dp_msc mosaic.
    label = starTables.Labels()

    # Load up information on the coo star for this field
    nirc2Dir = '/u/ghezgroup/data/gc/06maylgs1/combo'
    cooFile = '%s/mag06maylgs1_dp_msc_C_kp.coo' % (nirc2Dir)
    cooTmp = open(cooFile).readline().split()
    cooPix = [float(cooTmp[0]), float(cooTmp[1])]
    
    idx = np.where(label.name == 'irs16C')[0][0]
    cooAsec = [label.x[idx], label.y[idx]]
    
    scale = 0.00995
        
    # Convert the completeness results to absolute coordinates
    data.x_in = ((data.x_in_pix - cooPix[0]) * -scale) + cooAsec[0]
    data.y_in = ((data.y_in_pix - cooPix[1]) * scale) + cooAsec[1]

    data.x_out = ((data.x_out_pix - cooPix[0]) * -scale) + cooAsec[0]
    data.y_out = ((data.y_out_pix - cooPix[1]) * scale) + cooAsec[1]

    data.x = ((data.x_pix - cooPix[0]) * -scale) + cooAsec[0]
    data.y = ((data.y_pix - cooPix[1]) * scale) + cooAsec[1]

    return data

def image_completeness_in_osiris():
    """
    Star planting was performed on the 06maylgs1_dp_msc images. This
    routine loads up the star planting results and selects out the
    planted and detected stars within each OSIRIS field of view. The final
    results are stored in the file

    image_completness_in_osiris.dat

    and containt the following variables which are 1D arrays (not
    segregated by field).

    x_in_os - the input x positions for the planted stars.
    y_in_os - the input y positions
    kp_in_os - the input kp magnitudes (observed)
    kp_ext_in_os - the input kp magnitudes (extinction corrected)
    x_out_os - the output x positions, NaN for not detected
    y_out_os - the output y positions, NaN for not detected
    kp_out_os - the output kp (obs), NaN for not detected
    kp_ext_out_os - the output kp (extinction corr.), NaN for not detected
    """
    # Load up the completeness mosaic. Remember, it is aligned to the
    # mag06maylgs1_dp_msc_C_kp image coordinates and IRS 16C is the coo star.
    compDir = '/u/jlu/work/gc/dp_msc/2010_11_08/completeness/'
    cData = load_completeness_mosaic(compDir)

    x_in = cData.x_in
    y_in = cData.y_in
    kp_in = cData.m_in

    x_out = cData.x_out
    y_out = cData.y_out
    kp_out = cData.m_out

    # Load up the spectroscopic database to get field-of-view definitions.
    fields = getOsirisFields()

    # Final variables to keep track of planted stars
    # only in the OSIRIS fields of view.
    x_in_os = np.array([], dtype=float)
    y_in_os = np.array([], dtype=float)
    kp_in_os = np.array([], dtype=float)
    kp_ext_in_os = np.array([], dtype=float)
    x_out_os = np.array([], dtype=float)
    y_out_os = np.array([], dtype=float)
    kp_out_os = np.array([], dtype=float)
    kp_ext_out_os = np.array([], dtype=float)

    # Load up extinction maps
    # Load up the extinction map from Schodel 2010
    schExtinctFile = '/u/jlu/work/gc/imf/extinction/2010schodel_AKs_fg6.fits'
    schExtinct = pyfits.open(schExtinctFile)

    # Calculate AKs for all the stars in the dp_mosaic
    ext_map = schExtinct[0].data
    ext_hdr = schExtinct[0].header
    ext_map = np.array(ext_map, dtype=float)

    ext_scaleX = ext_hdr['CD1_1'] * 3600.0
    ext_scaleY = ext_hdr['CD2_2'] * 3600.0
    ext_sgraX = ext_hdr['CRPIX1']
    ext_sgraY = ext_hdr['CRPIX2']

    ext_mapX = np.arange(ext_map.shape[1], dtype=float)
    ext_mapY = np.arange(ext_map.shape[0], dtype=float)

    ext_mapX = (ext_mapX - ext_sgraX) * ext_scaleX
    ext_mapY = (ext_mapY - ext_sgraY) * ext_scaleY

    ks_2_kp_yng = synthetic.get_Kp_Ks(theAKs, 30000.0, filename=synFile)

    # Loop through each OSIRIS field and get the completeness curve.
    for rr in range(len(fields.name)):
        fieldName = fields.name[rr]
        print(('Working on field %s' % fieldName))

        # Trim down to just those pixels that are within this
        # OSIRIS field of view
        xypoints = np.column_stack((x_in, y_in))
        inside = nxutils.points_inside_poly(xypoints, fields.xyverts[rr])
        inside = np.where(inside == True)[0]

        x_in_os = np.append(x_in_os, x_in[inside])
        y_in_os = np.append(y_in_os, y_in[inside])
        kp_in_os = np.append(kp_in_os, kp_in[inside])
        x_out_os = np.append(kp_out_os, x_out[inside])
        y_out_os = np.append(y_out_os, y_out[inside])
        kp_out_os = np.append(kp_out_os, kp_out[inside])

        kp_ext_in_tmp = np.zeros(len(inside), dtype=float)
        kp_ext_out_tmp = np.zeros(len(inside), dtype=float)

        # Figure out the extinction corrected magnitude
        # for each planted star.
        for kk in range(len(inside)):
            dx = np.abs(ext_mapX - x_in[inside[kk]])
            dy = np.abs(ext_mapY - y_in[inside[kk]])

            xid = dx.argmin()
            yid = dy.argmin()

            AKs = ext_map[yid, xid]
            kp_2_ks_yng = synthetic.get_Kp_Ks(AKs, 30000.0, 
                                          filename=synFile)

            kp_ext_in_tmp[kk] = kp_in[inside[kk]] - kp_2_ks_yng - AKs \
                + theAKs + ks_2_kp_yng

            if np.isnan(kp_out[inside[kk]]) == False:
                kp_ext_out_tmp[kk] = kp_out[inside[kk]] - kp_2_ks_yng - AKs \
                    + theAKs + ks_2_kp_yng
            else:
                # Set to NaN
                kp_ext_out_tmp[kk] = kp_out[inside[kk]]

        kp_ext_in_os = np.append(kp_ext_in_os, kp_ext_in_tmp)
        kp_ext_out_os = np.append(kp_ext_out_os, kp_ext_out_tmp)

        # Remove these stars so that duplicates can't be added.
        x_in = np.delete(x_in, inside)
        y_in = np.delete(y_in, inside)
        kp_in = np.delete(kp_in, inside)
        x_out = np.delete(x_out, inside)
        y_out = np.delete(y_out, inside)
        kp_out = np.delete(kp_out, inside)
    
    # Save the results
    out_file = workDir + 'image_completeness_in_osiris.dat'

    _out = open(out_file, 'w')
    pickle.dump(x_in_os, _out)
    pickle.dump(y_in_os, _out)
    pickle.dump(kp_in_os, _out)
    pickle.dump(kp_ext_in_os, _out)
    pickle.dump(x_out_os, _out)
    pickle.dump(y_out_os, _out)
    pickle.dump(kp_out_os, _out)
    pickle.dump(kp_ext_out_os, _out)

    _out.close()

def image_completeness_by_radius(rmin=0, rmax=30, plot=True):
    """
    Calculate the imaging completeness curves.
    """
    _planted = open(workDir + 'image_completeness_in_osiris.dat', 'r')
    x_in = pickle.load(_planted)
    y_in = pickle.load(_planted)
    kp_in = pickle.load(_planted)
    kp_ext_in = pickle.load(_planted)

    x_out = pickle.load(_planted)
    y_out = pickle.load(_planted)
    kp_out = pickle.load(_planted)
    kp_ext_out = pickle.load(_planted)

    r_in = np.hypot(x_in, y_in)

    # Trim down to just the specified radius range
    idx = np.where((r_in > rmin) & (r_in <= rmax))[0]
    if len(idx) == 0:
        print('calc_image_completeness(): No planted stars found.')

    x_in = x_in[idx]
    y_in = y_in[idx]
    kp_in = kp_in[idx]
    kp_ext_in = kp_ext_in[idx]

    x_out = x_out[idx]
    y_out = y_out[idx]
    kp_out = kp_out[idx]
    kp_ext_out = kp_ext_out[idx]

    # Define final completeness array
    magStep = 0.25
    magHalfStep = magStep / 2.0
    magBins = np.arange(8, 20, magStep)
    completeness = np.zeros(len(magBins), dtype=float)
    completeness_ext = np.zeros(len(magBins), dtype=float)

    outRoot = '%simage_completeness_r_%.1f_%.1f' % (workDir, rmin, rmax)
    _comp = open(outRoot + '.txt', 'w')

    for mm in range(len(magBins)):
        # All planted stars in this field
        planted = np.where((kp_in >= (magBins[mm]-magHalfStep)) & 
                           (kp_in < (magBins[mm]+magHalfStep)))[0]
        planted_ext = np.where((kp_ext_in >= (magBins[mm]-magHalfStep)) & 
                               (kp_ext_in < (magBins[mm]+magHalfStep)))[0]

        # All detected stars in this field
        detected = np.where(np.isnan(kp_out[planted]) == False)[0]
        detected_ext = np.where(np.isnan(kp_ext_out[planted_ext]) == False)[0]


        if len(planted) != 0:
            completeness[mm] = float(len(detected)) / float(len(planted))
        else:
            completeness[mm] = np.NAN

        if len(planted_ext) != 0:
            completeness_ext[mm] = float(len(detected_ext)) 
            completeness_ext[mm] /= float(len(planted_ext))
        else:
            completeness[mm] = np.NAN
            completeness_ext[mm] = np.NAN

        _comp.write('%5.2f  %6.3f  %3d  %3d  %6.3f  %3d  %3d\n' % 
                    (magBins[mm], 
                     completeness[mm], len(planted), len(detected),
                     completeness_ext[mm], len(planted_ext), len(detected_ext))
                    )

    _comp.close()

    if plot:
        # Get rid of the NAN for plotting
        ok = np.isnan(completeness) == False
        ok_ext = np.isnan(completeness_ext) == False

        py.clf()
        py.plot(magBins[ok], completeness[ok], 'r.-', label='Raw')
        py.plot(magBins[ok], completeness_ext[ok], 'b.-', 
                label='Extinction Corrected')
        py.xlabel('Kp Magnitude')
        py.ylabel('Completeness')
        py.legend(loc='lower left')
        py.title('%.1f < r <= %.1f' % (rmin, rmax))
        py.ylim(0, 1.1)
        py.xlim(9, 20)
        py.savefig(outRoot + '.png')
        print(('Saving %s' % (outRoot + '.png')))

def image_completeness_by_radius_for_tuan():
    """
    Calculate the imaging completeness curves.
    """
    radial_bins = [0.001, 0.500, 1.000, 1.500, 2.000, 2.500, 3.125, 3.906,
                   4.883, 6.104, 7.629, 9.537, 11.921, 14.901]

    t = atpy.Table()
    for rr in range(len(radial_bins) - 1):
        rmin = radial_bins[rr]
        rmax = radial_bins[rr+1]
        image_completeness_by_radius(rmin=rmin, rmax=rmax)
        d = load_image_completeness_by_radius(rmin=rmin, rmax=rmax)

        if rr == 0:
            t.add_column('Kp_ext_corr', d.mag, format='11.2f', dtype=np.float32)

        colName = 'r%.3f-%.3f' % (rmin, rmax)
        t.add_column(colName, d.comp, format='14.3f', dtype=np.float32)

    t.write(workDir + 'image_completeness_for_tuan.fits', overwrite=True, type='fits')

def plot_image_completeness_vs_radius():
    """
    Calculate the imaging completeness curves at different radii for plotting
    purposes in the paper.
    """
    radial_bins = [0.0, 2.0, 4.0, 8.0, 16.0]

    py.clf()
    for rr in range(len(radial_bins) - 1):
        rmin = radial_bins[rr]
        rmax = radial_bins[rr+1]
        image_completeness_by_radius(rmin=rmin, rmax=rmax, plot=False)
        d = load_image_completeness_by_radius(rmin=rmin, rmax=rmax)

        legLabel = '%d" - %d"' % (rmin, rmax)
        py.plot(d.mag, d.comp_no_ext, label=legLabel, linewidth=2)

    py.legend(loc='lower left')
    py.xlabel('Kp Magnitude')
    py.ylabel('Imaging Completeness')
    py.ylim(0, 1.05)
    py.savefig(workDir + 'plots/image_completeness_vs_radius.png')
    py.savefig(workDir + 'plots/image_completeness_vs_radius.eps')
        
def spec_completeness_by_radius(rmin=0, rmax=30, plot=True):
    """
    Calculate spectroscopic KLF and completeness curves.
    """
    # Load spectral identifications
    s = load_spec_id_all_stars(flatten=True)

    # Compute radii
    s.r_yng = np.hypot(s.x_yng, s.y_yng)
    s.r_old = np.hypot(s.x_old, s.y_old)
    s.r_nirc2 = np.hypot(s.x_nirc2, s.y_nirc2)

    # Load up Tuan's completeness simulations for the unknown stars.
    specSims = load_unknown_sims()

    # Define magnitude bins in which to calculate the completeness
    magStep = klf_mag_bins[1] - klf_mag_bins[0]
    magHalfStep = magStep / 2.0
    magBins = klf_mag_bins

    cnt_yng = np.zeros(len(magBins), dtype=float)
    cnt_old = np.zeros(len(magBins), dtype=float)
    cnt_unk = np.zeros(len(magBins), dtype=float)
    cnt_unk_yng = np.zeros(len(magBins), dtype=float)
    cnt_unk_old = np.zeros(len(magBins), dtype=float)
    cnt_yng_noWR = np.zeros(len(magBins), dtype=float)

    cnt_ext_yng = np.zeros(len(magBins), dtype=float)
    cnt_ext_old = np.zeros(len(magBins), dtype=float)
    cnt_ext_unk = np.zeros(len(magBins), dtype=float)
    cnt_ext_unk_yng = np.zeros(len(magBins), dtype=float)
    cnt_ext_unk_old = np.zeros(len(magBins), dtype=float)
    cnt_ext_yng_noWR = np.zeros(len(magBins), dtype=float)

    adderr_unk_yng = np.zeros(len(magBins), dtype=float)
    adderr_unk_old = np.zeros(len(magBins), dtype=float)
    adderr_ext_unk_yng = np.zeros(len(magBins), dtype=float)
    adderr_ext_unk_old = np.zeros(len(magBins), dtype=float)

    # Lets keep a copy of all possible young stars
    # (including those with low probabilities) since
    # this will feed into the bayesian analysis.
    # Only keep the extinction corrected magnitudes.
    # Also, don't include the WR stars.
    yng_kp = np.array([], dtype=float)
    yng_kp_err = np.array([], dtype=float)
    yng_prob = np.array([], dtype=float)

    # Also Make a complete table of all possible young stars
    # (including the WR stars). This may be printed in the table.
    allyng_name = np.array([], dtype='S15')
    allyng_x = np.array([], dtype=float)
    allyng_y = np.array([], dtype=float)
    allyng_kp = np.array([], dtype=float)
    allyng_kp_ext = np.array([], dtype=float)
    allyng_kp_err = np.array([], dtype=float)
    allyng_isWR = np.array([], dtype=bool)
    allyng_prob = np.array([], dtype=float)
    allyng_prob_err = np.array([], dtype=float)

    # Make a complete table of all stars
    # (including the WR stars and old stars).
    all_name = np.array([], dtype='S15')
    all_x = np.array([], dtype=float)
    all_y = np.array([], dtype=float)
    all_kp = np.array([], dtype=float)
    all_kp_ext = np.array([], dtype=float)
    all_kp_err = np.array([], dtype=float)
    all_isWR = np.array([], dtype=bool)
    all_prob = np.array([], dtype=float)
    all_prob_err = np.array([], dtype=float)

    # Also keep track of the number of WR stars
    idx = np.where(s.isWR_yng == True)[0]
    yng_N_WR = len(idx)

    for mm in range(len(magBins)):
        magLo = magBins[mm] - magHalfStep
        magHi = magBins[mm] + magHalfStep

        print(('Working on mag bin %.2f - %.2f' % (magLo, magHi)))

        # Before extinction correction, figure out which stars
        # fall into the magnitude bin and radius range.
        yy = np.where((s.kp_yng >= magLo) & (s.kp_yng < magHi) &
                      (s.r_yng > rmin) & (s.r_yng <= rmax))[0]
        oo = np.where((s.kp_old >= magLo) & (s.kp_old < magHi) & 
                      (s.r_old > rmin) & (s.r_old <= rmax))[0]
        nn = np.where((s.kp_nirc2 >= magLo) & (s.kp_nirc2 < magHi) & 
                      (s.r_nirc2 > rmin) & (s.r_nirc2 <= rmax))[0]
        yy_noWR = np.where((s.kp_yng >= magLo) & (s.kp_yng < magHi) &
                           (s.r_yng > rmin) & (s.r_yng <= rmax) &
                           (s.isWR_yng == False))[0]

        cnt_yng[mm] = len(yy)
        cnt_old[mm] = len(oo)
        cnt_unk[mm] = len(nn) - len(yy) - len(oo)
        cnt_yng_noWR[mm] = len(yy_noWR)

        # Gather up all the unknown sources and find them in the
        # Monte Carlo simulations table (specSims)
        unk_probY = []
        unk_probO = []
        unk_probYerr = []
        unk_probOerr = []
        unk_sdx = []

        for ni in nn:
            name = s.name_nirc2[ni]
            if (name in s.name_yng) or (name in s.name_old):
                continue

            unk_sdx.append(ni)

            # Find this unkonwn source in the spectral star planting results
            idx = np.where(name == specSims['name'])[0]
            if len(idx) == 0:
                print(('    Unknown source %s (Kp = %5.2f, field = %s) not in MC sim. Resorting to priors.' % \
                    (name, s.kp_nirc2[ni], s.field_nirc2[ni])))
                priorYngRad, priorOldRad = priorsAtRadius(s.r_nirc2[ni])
                unk_probY.append( priorYngRad )
                unk_probO.append( priorOldRad )
                unk_probYerr.append( 0.1 )
                unk_probOerr.append( 0.1 )
            else:
                unk_probY.append( specSims['probYng'][idx[0]] )
                unk_probO.append( specSims['probOld'][idx[0]] )
                unk_probYerr.append( specSims['probYngErr'][idx[0]] )
                unk_probOerr.append( specSims['probOldErr'][idx[0]] )

        unk_probY = np.array(unk_probY)
        unk_probO = np.array(unk_probO)
        unk_probYerr = np.array(unk_probYerr)
        unk_probOerr = np.array(unk_probOerr)
        unk_sdx = np.array(unk_sdx)

        if len(unk_probY) != cnt_unk[mm]:
            print('Incorrect number of unknown sources:')
            print((mm, len(unk_probY), cnt_unk[mm], len(nn), len(yy), len(oo)))
            pdb.set_trace()
            
        # For the unknown sources, figure out how many are likely
        # to be young and how many are likely to be old.
        cnt_unk_yng[mm] = unk_probY.sum()
        cnt_unk_old[mm] = unk_probO.sum()
        adderr_unk_yng[mm] = math.sqrt((unk_probYerr**2).sum())
        adderr_unk_old[mm] = math.sqrt((unk_probOerr**2).sum())

        print(('  %3d yng, %3d old, %3d unkown (%4.1f yng, %4.1f old), %3d noWR (observed)' % \
            (cnt_yng[mm], cnt_old[mm], cnt_unk[mm],
             cnt_unk_yng[mm], cnt_unk_old[mm], cnt_yng_noWR[mm])))

        ##########
        # After extinction correction
        ##########
        yy = np.where((s.kp_ext_yng >= magLo) & (s.kp_ext_yng < magHi) & 
                      (s.r_yng > rmin) & (s.r_yng <= rmax))[0]
        oo = np.where((s.kp_ext_old >= magLo) & (s.kp_ext_old < magHi) &
                      (s.r_old > rmin) & (s.r_old <= rmax))[0]
        nn = np.where((s.kp_ext_nirc2 >= magLo) & (s.kp_ext_nirc2 < magHi) &
                      (s.r_nirc2 > rmin) & (s.r_nirc2 <= rmax))[0]
        yy_noWR = np.where((s.kp_ext_yng >= magLo) & (s.kp_ext_yng < magHi) & 
                           (s.r_yng > rmin) & (s.r_yng <= rmax) &
                           (s.isWR_yng == False))[0]

        cnt_ext_yng[mm] = len(yy)
        cnt_ext_old[mm] = len(oo)
        cnt_ext_unk[mm] = len(nn) - len(yy) - len(oo)
        cnt_ext_yng_noWR[mm] = len(yy_noWR)

        yng_kp = np.append(yng_kp, s.kp_ext_yng[yy_noWR])
        yng_kp_err = np.append(yng_kp_err, s.kperr_yng[yy_noWR])
        yng_prob = np.append(yng_prob, np.ones(len(yy_noWR)))

        allyng_name = np.append(allyng_name, s.name_yng[yy])
        allyng_x = np.append(allyng_x, s.x_yng[yy])
        allyng_y = np.append(allyng_y, s.y_yng[yy])
        allyng_kp = np.append(allyng_kp, s.kp_yng[yy])
        allyng_kp_err = np.append(allyng_kp_err, s.kperr_yng[yy])
        allyng_kp_ext = np.append(allyng_kp_ext, s.kp_ext_yng[yy])
        allyng_prob = np.append(allyng_prob, np.ones(len(yy)))
        allyng_isWR = np.append(allyng_isWR, s.isWR_yng[yy])

        # Append the known young stars to the "all" list. 
        all_name = np.append(all_name, s.name_yng[yy])
        all_x = np.append(all_x, s.x_yng[yy])
        all_y = np.append(all_y, s.y_yng[yy])
        all_kp = np.append(all_kp, s.kp_yng[yy])
        all_kp_err = np.append(all_kp_err, s.kperr_yng[yy])
        all_kp_ext = np.append(all_kp_ext, s.kp_ext_yng[yy])
        all_prob = np.append(all_prob, np.ones(len(yy)))
        all_prob_err = np.append(all_prob_err, np.zeros(len(yy)))
        all_isWR = np.append(all_isWR, s.isWR_yng[yy])

        # Append the known old stars to the "all" list. 
        all_name = np.append(all_name, s.name_old[oo])
        all_x = np.append(all_x, s.x_old[oo])
        all_y = np.append(all_y, s.y_old[oo])
        all_kp = np.append(all_kp, s.kp_old[oo])
        all_kp_err = np.append(all_kp_err, s.kperr_old[oo])
        all_kp_ext = np.append(all_kp_ext, s.kp_ext_old[oo])
        all_prob = np.append(all_prob, np.zeros(len(oo), dtype=float))
        all_prob_err = np.append(all_prob_err, np.zeros(len(oo), dtype=float))
        all_isWR = np.append(all_isWR, np.zeros(len(oo), dtype=bool))

        # Gather up all the unknown sources and find them in the
        # Monte Carlo simulations table (specSims)
        unk_probY = []
        unk_probO = []
        unk_probYerr = []
        unk_probOerr = []
        unk_sdx = []

        for ni in nn:
            name = s.name_nirc2[ni]
            if (name in s.name_yng) or (name in s.name_old):
                continue

            unk_sdx.append(ni)
            
            idx = np.where(name == specSims['name'])[0]
            if len(idx) == 0:
                print(('    Unknown source %s (Kp = %5.2f, field = %s) not in MC sim. Resorting to priors.' % \
                    (name, s.kp_nirc2[ni], s.field_nirc2[ni])))

                priorYngRad, priorOldRad = priorsAtRadius(s.r_nirc2[ni])
                unk_probY.append( priorYngRad )
                unk_probO.append( priorOldRad )
                unk_probYerr.append( 0.1 )
                unk_probOerr.append( 0.1 )
            else:
                unk_probY.append( specSims['probYng'][idx[0]] )
                unk_probO.append( specSims['probOld'][idx[0]] )
                unk_probYerr.append( specSims['probYngErr'][idx[0]] )
                unk_probOerr.append( specSims['probOldErr'][idx[0]] )

        unk_probY = np.array(unk_probY)
        unk_probO = np.array(unk_probO)
        unk_probYerr = np.array(unk_probYerr)
        unk_probOerr = np.array(unk_probOerr)
        unk_sdx = np.array(unk_sdx)

        if len(unk_probY) != cnt_ext_unk[mm]:
            print('Incorrect number of unknown sources:')
            print((mm, len(unk_probY), cnt_ext_unk[mm], len(nn), len(yy), len(oo)))

        # For the unknown sources, figure out how many are likely
        # to be young and how many are likely to be old.
        cnt_ext_unk_yng[mm] = unk_probY.sum()
        cnt_ext_unk_old[mm] = unk_probO.sum()
        adderr_ext_unk_yng[mm] = math.sqrt((unk_probYerr**2).sum())
        adderr_ext_unk_old[mm] = math.sqrt((unk_probOerr**2).sum())

        print(('  %3d yng, %3d old, %3d unkown (%4.1f yng, %4.1f old), %3d no WR (extinction corrected)' % \
            (cnt_ext_yng[mm], cnt_ext_old[mm], cnt_ext_unk[mm],
             cnt_ext_unk_yng[mm], cnt_ext_unk_old[mm], cnt_ext_yng_noWR[mm])))

        # Save the unknown sources (that are young) to our lists.
        ndx = np.where(unk_probY != 0)[0]
        if (len(ndx) != 0):
            yng_kp = np.append(yng_kp, s.kp_ext_nirc2[unk_sdx[ndx]])
            yng_kp_err = np.append(yng_kp_err, s.kperr_nirc2[unk_sdx[ndx]])
            yng_prob = np.append(yng_prob, unk_probY[ndx])

            allyng_name = np.append(allyng_name, s.name_nirc2[unk_sdx[ndx]])
            allyng_x = np.append(allyng_x, s.x_nirc2[unk_sdx[ndx]])
            allyng_y = np.append(allyng_y, s.y_nirc2[unk_sdx[ndx]])
            allyng_kp = np.append(allyng_kp, s.kp_nirc2[unk_sdx[ndx]])
            allyng_kp_err = np.append(allyng_kp_err, s.kperr_nirc2[unk_sdx[ndx]])
            allyng_kp_ext = np.append(allyng_kp_ext, s.kp_ext_nirc2[unk_sdx[ndx]])
            allyng_prob = np.append(allyng_prob, unk_probY[ndx])
            allyng_prob_err = np.append(allyng_prob_err, unk_probYerr[ndx])
            allyng_isWR = np.append(allyng_isWR, np.zeros(len(ndx), dtype=bool))

        # Append all the unkown stars to the list.
        if len(unk_sdx) > 0:
            all_name = np.append(all_name, s.name_nirc2[unk_sdx])
            all_x = np.append(all_x, s.x_nirc2[unk_sdx])
            all_y = np.append(all_y, s.y_nirc2[unk_sdx])
            all_kp = np.append(all_kp, s.kp_nirc2[unk_sdx])
            all_kp_err = np.append(all_kp_err, s.kperr_nirc2[unk_sdx])
            all_kp_ext = np.append(all_kp_ext, s.kp_ext_nirc2[unk_sdx])
            all_prob = np.append(all_prob, unk_probY)
            all_prob_err = np.append(all_prob_err, unk_probYerr)
            all_isWR = np.append(all_isWR, np.zeros(len(unk_sdx), dtype=bool))


    
    comp = (cnt_yng + cnt_old) / (cnt_yng + cnt_old + cnt_unk)
    comp_ext = (cnt_ext_yng + cnt_ext_old) / (cnt_ext_yng + cnt_ext_old + cnt_ext_unk)
    comp_fix = fix_osiris_spectral_comp(magBins, comp)
    comp_ext_fix = fix_osiris_spectral_comp(magBins, comp_ext)

    comp_yng = cnt_yng / (cnt_yng + cnt_unk_yng)
    comp_ext_yng = cnt_ext_yng / (cnt_ext_yng + cnt_ext_unk_yng)
    comp_fix_yng = fix_osiris_spectral_comp(magBins, comp_yng)
    comp_ext_fix_yng = fix_osiris_spectral_comp(magBins, comp_ext_yng)

    comp_old = cnt_old / (cnt_old + cnt_unk_old)
    comp_ext_old = cnt_ext_old / (cnt_ext_old + cnt_ext_unk_old)
    comp_fix_old = fix_osiris_spectral_comp(magBins, comp_old)
    comp_ext_fix_old = fix_osiris_spectral_comp(magBins, comp_ext_old)

    outRoot = '%sspec_completeness_r_%.1f_%.1f' % (workDir, rmin, rmax)

    if plot == True:
        # Total completeness
        py.clf()
        py.plot(magBins, comp_ext_fix, 'ks-', label='Ext. Correct, Fixed')
        py.plot(magBins, comp_ext, 'ks--', label='Ext. Correct')
        py.plot(magBins, comp, 'r^--', label='Raw')
        py.xlabel('Kp Magnitude')
        py.ylabel('Completeness')
        py.title('%.1f < r <= %.1f' % (rmin, rmax))
        py.ylim(0, 1.1)
        py.xlim(9, 18)
        py.legend(loc='lower left')
        py.savefig(outRoot + '.png')

        # Completeness to young stars
        py.clf()
        py.plot(magBins, comp_ext_fix_yng, 'ks-', label='Ext. Correct, Fixed')
        py.plot(magBins, comp_ext_yng, 'ks--', label='Ext. Correct')
        py.plot(magBins, comp_yng, 'r^--', label='Raw')
        py.xlabel('Kp Magnitude')
        py.ylabel('Completeness')
        py.title('Young Stars: %.1f < r <= %.1f' % (rmin, rmax))
        py.ylim(0, 1.1)
        py.xlim(9, 18)
        py.legend(loc='lower left')
        py.savefig(outRoot + '_yng.png')

        # Completeness to old stars 
        py.clf()
        py.plot(magBins, comp_ext_fix_old, 'ks-', label='Ext. Correct, Fixed')
        py.plot(magBins, comp_ext_old, 'ks--', label='Ext. Correct')
        py.plot(magBins, comp_old, 'r^--', label='Raw')
        py.xlabel('Kp Magnitude')
        py.ylabel('Completeness')
        py.title('Old Stars: %.1f < r <= %.1f' % (rmin, rmax))
        py.ylim(0, 1.1)
        py.xlim(9, 18)
        py.legend(loc='lower left')
        py.savefig(outRoot + '_old.png')

        # Completeness compared
        py.clf()
        py.plot(magBins, comp_ext_fix, 'ks-', label='Total Spec. Comp.')
        py.plot(magBins, comp_ext_fix_yng, 'bo-', label='Young Spec. Comp.')
        py.plot(magBins, comp_ext_fix_old, 'r^-', label='Old Spec. Comp.')
        py.xlabel('Kp Magnitude')
        py.ylabel('Completeness')
        py.title('%.1f < r <= %.1f' % (rmin, rmax))
        py.ylim(0, 1.1)
        py.xlim(9, 18)
        py.legend(loc='lower left')
        py.savefig(outRoot + '_yng_v_old.png')

    # Save the original list of young star magnitudes
    _pick2 = open(outRoot + '_yng_kp.pickle', 'wb')
    pickle.dump(yng_kp, _pick2)
    pickle.dump(yng_kp_err, _pick2)
    pickle.dump(yng_prob, _pick2)
    pickle.dump(yng_N_WR, _pick2)
    _pick2.close()

    # Save the original list of young star -- complete info
    _pick3 = open(outRoot + '_yng_info.pickle', 'wb')
    pickle.dump(allyng_name, _pick3)
    pickle.dump(allyng_x, _pick3)
    pickle.dump(allyng_y, _pick3)
    pickle.dump(allyng_kp, _pick3)
    pickle.dump(allyng_kp_ext, _pick3)
    pickle.dump(allyng_kp_err, _pick3)
    pickle.dump(allyng_isWR, _pick3)
    pickle.dump(allyng_prob, _pick3)
    pickle.dump(allyng_prob_err, _pick3)
    _pick3.close()

    # Save the original list of all stars -- complete info
    _pick3 = open(outRoot + '_all_info.pickle', 'wb')
    pickle.dump(all_name, _pick3)
    pickle.dump(all_x, _pick3)
    pickle.dump(all_y, _pick3)
    pickle.dump(all_kp, _pick3)
    pickle.dump(all_kp_ext, _pick3)
    pickle.dump(all_kp_err, _pick3)
    pickle.dump(all_isWR, _pick3)
    pickle.dump(all_prob, _pick3)
    pickle.dump(all_prob_err, _pick3)
    _pick3.close()

    # Save the completeness and KLF curves
    _pick = open(outRoot + '.pickle', 'wb')
    pickle.dump(magBins, _pick)
    pickle.dump(cnt_yng, _pick)
    pickle.dump(cnt_old, _pick)
    pickle.dump(cnt_unk, _pick)
    pickle.dump(cnt_yng_noWR, _pick)
    pickle.dump(cnt_unk_yng, _pick)
    pickle.dump(cnt_unk_old, _pick)
    pickle.dump(adderr_unk_yng, _pick)
    pickle.dump(adderr_unk_old, _pick)
    pickle.dump(cnt_ext_yng, _pick)
    pickle.dump(cnt_ext_old, _pick)
    pickle.dump(cnt_ext_unk, _pick)
    pickle.dump(cnt_ext_yng_noWR, _pick)
    pickle.dump(cnt_ext_unk_yng, _pick)
    pickle.dump(cnt_ext_unk_old, _pick)
    pickle.dump(adderr_ext_unk_yng, _pick)
    pickle.dump(adderr_ext_unk_old, _pick)
    pickle.dump(comp, _pick)
    pickle.dump(comp_ext, _pick)
    pickle.dump(comp_fix, _pick)
    pickle.dump(comp_ext_fix, _pick)
    pickle.dump(comp_yng, _pick)
    pickle.dump(comp_ext_yng, _pick)
    pickle.dump(comp_fix_yng, _pick)
    pickle.dump(comp_ext_fix_yng, _pick)
    pickle.dump(comp_old, _pick)
    pickle.dump(comp_ext_old, _pick)
    pickle.dump(comp_fix_old, _pick)
    pickle.dump(comp_ext_fix_old, _pick)
    _pick.close()

    # Text file
    _out = open(outRoot + '.txt', 'w')

    _out.write('%-5s  %15s  %15s\n' % ('#mag', 'comp_yng', 'comp_ext_yng'))
    for mm in range(len(magBins)):
        _out.write('%5.2f  %15.3f  %15.3f\n' % 
                   (magBins[mm], comp_fix_yng[mm], comp_ext_fix_yng[mm]))
    _out.close()

    # Text file with numbers
    t_out = Table()
    t_out['magBins'] = magBins
    t_out['cnt_ext_yng'] = cnt_ext_yng
    t_out['cnt_ext_unk_yng'] = cnt_ext_unk_yng
    t_out['comp_ext_yng'] = comp_ext_yng
    t_out.write(outRoot + '_num.txt', format='ascii.fixed_width')


def load_spec_completeness_by_radius(rmin=0, rmax=30):
    """
    Load up the results from estimating completness (and
    star counts) on our spectral-typed and un-typed
    stars.
    """
    d = objects.DataHolder()

    outRoot = '%sspec_completeness_r_%.1f_%.1f' % (workDir, rmin, rmax)

    _pick = open(outRoot + '.pickle', 'rb')
    d.magBins = pickle.load(_pick)
    d.cnt_yng = pickle.load(_pick)
    d.cnt_old = pickle.load(_pick)
    d.cnt_unk = pickle.load(_pick)
    d.cnt_yng_noWR = pickle.load(_pick)
    d.cnt_unk_yng = pickle.load(_pick)
    d.cnt_unk_old = pickle.load(_pick)
    d.adderr_unk_yng = pickle.load(_pick)
    d.adderr_unk_old = pickle.load(_pick)
    d.cnt_ext_yng = pickle.load(_pick)
    d.cnt_ext_old = pickle.load(_pick)
    d.cnt_ext_unk = pickle.load(_pick)
    d.cnt_ext_yng_noWR = pickle.load(_pick)
    d.cnt_ext_unk_yng = pickle.load(_pick)
    d.cnt_ext_unk_old = pickle.load(_pick)
    d.adderr_ext_unk_yng = pickle.load(_pick)
    d.adderr_ext_unk_old = pickle.load(_pick)
    d.comp = pickle.load(_pick)
    d.comp_ext = pickle.load(_pick)
    d.comp_fix = pickle.load(_pick)
    d.comp_ext_fix = pickle.load(_pick)
    d.comp_yng = pickle.load(_pick)
    d.comp_ext_yng = pickle.load(_pick)
    d.comp_fix_yng = pickle.load(_pick)
    d.comp_ext_fix_yng = pickle.load(_pick)
    d.comp_old = pickle.load(_pick)
    d.comp_ext_old = pickle.load(_pick)
    d.comp_fix_old = pickle.load(_pick)
    d.comp_ext_fix_old = pickle.load(_pick)

    _pick.close()

    return d

def merge_completeness_by_radius(rmin=0, rmax=30):
    """
    Merge imaging and spectroscopic completeness curves. In reality we
    won't use the spectroscopic completeness curve and will instead simply
    use the un-typed star planting simulations that assign a probability
    of young/old to each un-typed star with Kp < 16.

    This routine does generate a useful plot though.
    """
    magBins = klf_mag_bins

    # Read in the image completeness table
    #img_file = '%simage_completeness_r_%.1f_%.1f.txt' % (workDir_old, rmin, rmax)
    img_file = '%simage_completeness_r_%.1f_%.1f.txt' % (workDir_old, 0, 30)
    _img = Table.read(img_file, format='ascii')

    mag_img = _img['col1']
    comp_img = _img['col2']
    comp_ext_img = _img['col5'] 

    # Read in the spectroscopic completeness table
    spec_file = '%sspec_completeness_r_%.1f_%.1f.txt' % (workDir, rmin, rmax)
    _spec = Table.read(spec_file, format='ascii')
    
    mag_spec = _spec['mag']
    comp_spec = _spec['comp_yng']
    comp_ext_spec = _spec['comp_ext_yng']

    py.clf()
    py.subplots_adjust(left=0.15)
    py.plot(mag_img, comp_ext_img, label='Imaging')
    py.plot(mag_spec, comp_ext_spec, label='Spectroscopy')
    py.legend(loc='lower left')
    py.xlabel('Extinction-Corrected Kp Magnitudes')
    py.ylabel('Completeness')
    py.ylim(0, 1.05)
    py.savefig('%scompleteness_ext_spec_imag_r_%.1f_%.1f.png' % 
               (workDir, rmin, rmax))
    #py.savefig('%scompleteness_ext_spec_imag_r_%.1f_%.1f.eps' % 
    #           (workDir, rmin, rmax))

    py.clf()
    py.subplots_adjust(left=0.15)
    py.plot(mag_img, comp_img, label='Imaging')
    py.plot(mag_spec, comp_spec, label='Spectroscopy')
    py.legend(loc='lower left')
    py.xlabel('Kp Magnitudes')
    py.ylabel('Non-Extinction-Corrected Completeness')
    py.ylim(0, 1.05)
    py.savefig('%scompleteness_spec_imag_r_%.1f_%.1f.png' % 
               (workDir, rmin, rmax))

    # Put everything on a common magnitude grid (same as the data)
    good = np.isnan(comp_img) == False
    Kp_interp = interpolate.splrep(mag_img[good], comp_img[good], k=1, s=0)
    comp_img_new = interpolate.splev(klf_mag_bins, Kp_interp)

    good = np.isnan(comp_ext_img) == False
    Kp_interp = interpolate.splrep(mag_img[good], comp_ext_img[good], k=1, s=0)
    comp_ext_img_new = interpolate.splev(klf_mag_bins, Kp_interp)

    good = np.isnan(comp_spec) == False
    Kp_interp = interpolate.splrep(mag_spec[good], comp_spec[good], k=1, s=0)
    comp_spec_new = interpolate.splev(klf_mag_bins, Kp_interp)

    good = np.isnan(comp_ext_spec) == False
    Kp_interp = interpolate.splrep(mag_spec[good], comp_ext_spec[good], k=1, s=0)
    comp_ext_spec_new = interpolate.splev(klf_mag_bins, Kp_interp)

    outRoot = '%scompleteness_info_r_%.1f_%.1f' % (workDir, rmin, rmax)

    _out = open(outRoot + '.pickle', 'wb')
    pickle.dump(klf_mag_bins, _out)
    pickle.dump(comp_ext_img_new, _out)
    pickle.dump(comp_ext_spec_new, _out)
    pickle.dump(comp_img_new, _out)
    pickle.dump(comp_spec_new, _out)
    _out.close()

    _out = open(outRoot + '.txt', 'w')
    _out.write('%15s  %15s  %15s  %15s  %15s\n' % 
               ('#Kp', 'c_ext_img', 'c_ext_spec', 'c_img', 'c_spec'))
                                                   
    for mm in range(len(klf_mag_bins)):
        _out.write('%15.2f  %15.2f  %15.2f  %15.2f  %15.2f\n' % 
                   (klf_mag_bins[mm], comp_ext_img_new[mm],
                    comp_ext_spec_new[mm], comp_img_new[mm], comp_spec_new[mm]))
    _out.close()

def klf_by_radius(rmin=0, rmax=30):
    # Load imaging completeness info
    comp_file = '%scompleteness_info_r_%.1f_%.1f.txt' % (workDir, rmin, rmax)
    _comp = Table.read(comp_file, format='ascii')
    comp_mag = _comp['Kp']
    comp_imag = _comp['c_ext_img']
    comp_spec = _comp['c_ext_spec']
    
    # Also load up the spectral completeness info that contains results from
    # star planting around un-typed stars.
    spec_info = load_spec_completeness_by_radius(rmin=rmin, rmax=rmax)

    if ((len(comp_mag) != len(klf_mag_bins)) or
        (len(spec_info.magBins) != len(klf_mag_bins))):
        print('klf_by_radius(): Mismatch between completeness and KLF mag bins')

    # Load up the area maps for the OSIRIS fields and calculate the
    # total area.
    areaFile = '/u/jlu/work/gc/imf/extinction/nirc2_mask_all.fits'
    areaFITS = fits.open(areaFile)
    area_map = areaFITS[0].data
    area_hdr = areaFITS[0].header
    area_map = np.array(area_map, dtype=float)

    area_scaleX = area_hdr['CD1_1'] * 3600.0
    area_scaleY = area_hdr['CD2_2'] * 3600.0
    area_sgraX = area_hdr['CRPIX1']
    area_sgraY = area_hdr['CRPIX2']

    area_mapX = np.arange(area_map.shape[1], dtype=float)
    area_mapY = np.arange(area_map.shape[0], dtype=float)

    area_mapX = (area_mapX - area_sgraX) * area_scaleX
    area_mapY = (area_mapY - area_sgraY) * area_scaleY

    mapXX, mapYY = np.meshgrid(area_mapX, area_mapY)
    mapRR = np.hypot(mapXX, mapYY)

    # Calculate the total area.
    rdx = np.where((mapRR > rmin) & (mapRR <= rmax))

    foo1 = np.where(mapRR <= rmin)
    area_map[foo1[0], foo1[1]] = 0
    foo2 = np.where(mapRR > rmax)
    area_map[foo2[0], foo2[1]] = 0

    totalArea = area_map[rdx[0], rdx[1]].sum()
    totalArea *= abs(area_scaleX * area_scaleY)

    py.clf()
    py.imshow(area_map, extent=[area_mapX[0], area_mapX[-1], 
                                area_mapY[0], area_mapY[-1]])
    py.title('Total Area = %.1f sq. arcsec' % totalArea)
    py.xlabel('R.A. Offset from Sgr A*')
    py.ylabel('Dec. Offset from Sgr A*')
    py.savefig('%sosiris_area_map_r_%.1f_%.1f.png' % (workDir, rmin, rmax))

    # Make a binned luminosity function.
    sizeKp = len(spec_info.magBins)
    binSizeKp = spec_info.magBins[1] - spec_info.magBins[0]
    perAsec2Mag = totalArea * binSizeKp


    # Make KLFs
    N = spec_info.cnt_yng
    eN = poisson_error(N)
    KLF = N / perAsec2Mag
    eKLF = eN / perAsec2Mag

    N_ext = spec_info.cnt_ext_yng
    eN_ext = poisson_error(N_ext)
    KLF_ext = N_ext / perAsec2Mag
    eKLF_ext = eN_ext / perAsec2Mag

    N_ext_cmp_sp = spec_info.cnt_ext_yng + spec_info.cnt_ext_unk_yng
    #eN_ext_cmp_sp = np.sqrt(N_ext_cmp_sp + spec_info.adderr_ext_unk_yng**2)
    eN_ext_cmp_sp = poisson_error(N_ext_cmp_sp)
    KLF_ext_cmp_sp = N_ext_cmp_sp / perAsec2Mag
    eKLF_ext_cmp_sp = eN_ext_cmp_sp / perAsec2Mag

    N_ext_cmp_sp_im = N_ext_cmp_sp / comp_imag
    eN_ext_cmp_sp_im = eN_ext_cmp_sp / comp_imag
    KLF_ext_cmp_sp_im = KLF_ext_cmp_sp / comp_imag
    eKLF_ext_cmp_sp_im = eKLF_ext_cmp_sp / comp_imag

    # Construct sub-set without WR stars.
    N_noWR = spec_info.cnt_yng_noWR
    eN_noWR = poisson_error(N_noWR)
    KLF_noWR = N_noWR / perAsec2Mag
    eKLF_noWR = eN_noWR / perAsec2Mag

    N_ext_noWR = spec_info.cnt_ext_yng_noWR
    eN_ext_noWR = poisson_error(N_ext_noWR)
    KLF_ext_noWR = N_ext_noWR / perAsec2Mag
    eKLF_ext_noWR = eN_ext_noWR / perAsec2Mag

    N_ext_cmp_sp_noWR = spec_info.cnt_ext_yng_noWR + spec_info.cnt_ext_unk_yng
    #eN_ext_cmp_sp_noWR = np.sqrt(N_ext_cmp_sp_noWR + spec_info.adderr_ext_unk_yng**2)
    eN_ext_cmp_sp_noWR = poisson_error(N_ext_cmp_sp_noWR)
    KLF_ext_cmp_sp_noWR = N_ext_cmp_sp_noWR / perAsec2Mag
    eKLF_ext_cmp_sp_noWR = eN_ext_cmp_sp_noWR / perAsec2Mag

    N_ext_cmp_sp_im_noWR = N_ext_cmp_sp_noWR / comp_imag
    eN_ext_cmp_sp_im_noWR = eN_ext_cmp_sp_noWR / comp_imag
    KLF_ext_cmp_sp_im_noWR = KLF_ext_cmp_sp_noWR / comp_imag
    eKLF_ext_cmp_sp_im_noWR = eKLF_ext_cmp_sp_noWR / comp_imag

    # Fix some stuff
    idx = np.where(np.isnan(KLF_ext_cmp_sp) == True)[0]
    for ii in idx:
        N_ext_cmp_sp[ii] = 0.0
        KLF_ext_cmp_sp[ii] = 0.0

    idx = np.where(np.isnan(KLF_ext_cmp_sp_im) == True)[0]
    for ii in idx:
        N_ext_cmp_sp_im[ii] = 0.0
        KLF_ext_cmp_sp_im[ii] = 0.0

    idx = np.where(np.isnan(KLF_ext_cmp_sp_noWR) == True)[0]
    for ii in idx:
        N_ext_cmp_sp_noWR[ii] = 0.0
        KLF_ext_cmp_sp_noWR[ii] = 0.0

    idx = np.where(np.isnan(KLF_ext_cmp_sp_im_noWR) == True)[0]
    for ii in idx:
        N_ext_cmp_sp_im_noWR[ii] = 0.0
        KLF_ext_cmp_sp_im_noWR[ii] = 0.0


    # Save to a pickle file
    pickleFile = '%sklf_r_%.1f_%.1f.dat' % (workDir, rmin, rmax)

    _out = open(pickleFile, 'wb')
    
    pickle.dump(spec_info.magBins, _out)
    pickle.dump(N, _out)
    pickle.dump(eN, _out)
    pickle.dump(N_ext, _out)
    pickle.dump(eN_ext, _out)
    pickle.dump(N_ext_cmp_sp, _out)
    pickle.dump(eN_ext_cmp_sp, _out)
    pickle.dump(N_ext_cmp_sp_im, _out)
    pickle.dump(eN_ext_cmp_sp_im, _out)
    pickle.dump(KLF, _out)
    pickle.dump(eKLF, _out)
    pickle.dump(KLF_ext, _out)
    pickle.dump(eKLF_ext, _out)
    pickle.dump(KLF_ext_cmp_sp, _out)
    pickle.dump(eKLF_ext_cmp_sp, _out)
    pickle.dump(KLF_ext_cmp_sp_im, _out)
    pickle.dump(eKLF_ext_cmp_sp_im, _out)
    pickle.dump(N_noWR, _out)
    pickle.dump(eN_noWR, _out)
    pickle.dump(N_ext_noWR, _out)
    pickle.dump(eN_ext_noWR, _out)
    pickle.dump(N_ext_cmp_sp_noWR, _out)
    pickle.dump(eN_ext_cmp_sp_noWR, _out)
    pickle.dump(N_ext_cmp_sp_im_noWR, _out)
    pickle.dump(eN_ext_cmp_sp_im_noWR, _out)
    pickle.dump(KLF_noWR, _out)
    pickle.dump(eKLF_noWR, _out)
    pickle.dump(KLF_ext_noWR, _out)
    pickle.dump(eKLF_ext_noWR, _out)
    pickle.dump(KLF_ext_cmp_sp_noWR, _out)
    pickle.dump(eKLF_ext_cmp_sp_noWR, _out)
    pickle.dump(KLF_ext_cmp_sp_im_noWR, _out)
    pickle.dump(eKLF_ext_cmp_sp_im_noWR, _out)
    pickle.dump(comp_spec, _out)
    pickle.dump(comp_imag, _out)

    _out.close()

def load_klf_by_radius(rmin=0, rmax=30, mask_for_log=False):
    pickleFile = '%sklf_r_%.1f_%.1f.dat' % (workDir, rmin, rmax)
    _in = open(pickleFile, 'rb')
    
    d = objects.DataHolder()

    d.Kp = pickle.load(_in)
    
    d.N = pickle.load(_in)
    d.eN = pickle.load(_in)

    d.N_ext = pickle.load(_in)
    d.eN_ext = pickle.load(_in)

    d.N_ext_cmp_sp = pickle.load(_in)
    d.eN_ext_cmp_sp = pickle.load(_in)

    d.N_ext_cmp_sp_im = pickle.load(_in)
    d.eN_ext_cmp_sp_im = pickle.load(_in)

    d.KLF = pickle.load(_in)
    d.eKLF = pickle.load(_in)

    d.KLF_ext = pickle.load(_in)
    d.eKLF_ext = pickle.load(_in)

    d.KLF_ext_cmp_sp = pickle.load(_in)
    d.eKLF_ext_cmp_sp = pickle.load(_in)

    d.KLF_ext_cmp_sp_im = pickle.load(_in)
    d.eKLF_ext_cmp_sp_im = pickle.load(_in)

    d.N_noWR = pickle.load(_in)
    d.eN_noWR = pickle.load(_in)

    d.N_ext_noWR = pickle.load(_in)
    d.eN_ext_noWR = pickle.load(_in)

    d.N_ext_cmp_sp_noWR = pickle.load(_in)
    d.eN_ext_cmp_sp_noWR = pickle.load(_in)

    d.N_ext_cmp_sp_im_noWR = pickle.load(_in)
    d.eN_ext_cmp_sp_im_noWR = pickle.load(_in)

    d.KLF_noWR = pickle.load(_in)
    d.eKLF_noWR = pickle.load(_in)

    d.KLF_ext_noWR = pickle.load(_in)
    d.eKLF_ext_noWR = pickle.load(_in)

    d.KLF_ext_cmp_sp_noWR = pickle.load(_in)
    d.eKLF_ext_cmp_sp_noWR = pickle.load(_in)

    d.KLF_ext_cmp_sp_im_noWR = pickle.load(_in)
    d.eKLF_ext_cmp_sp_im_noWR = pickle.load(_in)

    d.comp_spec_ext = pickle.load(_in)
    d.comp_imag_ext = pickle.load(_in)

    if mask_for_log:
        # Repair for zeros since we are plotting in semi-log-y
        d.eN = np.ma.masked_where(d.N <= 0, d.eN)
        d.N = np.ma.masked_where(d.N <= 0, d.N)

        d.eN_ext = np.ma.masked_where(d.N_ext <= 0, d.eN_ext)
        d.N_ext = np.ma.masked_where(d.N_ext <= 0, d.N_ext)

        d.eN_ext_cmp_sp = np.ma.masked_where(d.N_ext_cmp_sp <= 0, 
                                             d.eN_ext_cmp_sp)
        d.N_ext_cmp_sp = np.ma.masked_where(d.N_ext_cmp_sp <= 0, 
                                            d.N_ext_cmp_sp)

        d.eN_ext_cmp_sp_im = np.ma.masked_where(d.N_ext_cmp_sp_im <= 0, 
                                                d.eN_ext_cmp_sp_im)
        d.N_ext_cmp_sp_im = np.ma.masked_where(d.N_ext_cmp_sp_im <= 0, 
                                               d.N_ext_cmp_sp_im)
        
        d.eKLF = np.ma.masked_where(d.KLF <= 0, d.eKLF)
        d.KLF = np.ma.masked_where(d.KLF <= 0, d.KLF)
        
        d.eKLF_ext = np.ma.masked_where(d.KLF_ext <= 0, d.eKLF_ext)
        d.KLF_ext = np.ma.masked_where(d.KLF_ext <= 0, d.KLF_ext)
        
        d.eKLF_ext_cmp_sp = np.ma.masked_where(d.KLF_ext_cmp_sp <= 0, 
                                               d.eKLF_ext_cmp_sp)
        d.KLF_ext_cmp_sp = np.ma.masked_where(d.KLF_ext_cmp_sp <= 0, 
                                              d.KLF_ext_cmp_sp)
        
        d.eKLF_ext_cmp_sp_im = np.ma.masked_where(d.KLF_ext_cmp_sp_im <= 0, 
                                                  d.eKLF_ext_cmp_sp_im)
        d.KLF_ext_cmp_sp_im = np.ma.masked_where(d.KLF_ext_cmp_sp_im <= 0, 
                                                 d.KLF_ext_cmp_sp_im)


    _in.close()

    return d

def plot_klf_vs_bartko_by_radius(rmin=0, rmax=30, logAge=6.78, imfSlope=1.35):
    d = load_klf_by_radius(rmin, rmax, mask_for_log=True)

    magBin = d.Kp[1] - d.Kp[0]

    idx = np.where(d.Kp <= 16)[0]

    # Rescale Bartko to match our first two bins
    bartkoKLF = np.array([0.034, 0.08, 0.19, 0.20, 0.14, 0.18, 0.10])
    bartkoKp = np.array( [  9.5, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0])
    bartkoModel = np.array([0.0089, 0.02, 0.07, 0.24, 0.26, 0.4, 0.7])
    bartkoErrX = np.array([1, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5])

    scaleFactor = 1.0
    bartkoKLF = bartkoKLF * scaleFactor

    # Load up a model luminosity function
    modelStars = model_klf(logAge=logAge, imfSlope=imfSlope)
    scaleFactor = 700.0  # equivalent to 5000 stars / pi * 15" * 15"
    weights = np.ones(len(modelStars), dtype=float)
    weights *= scaleFactor / len(modelStars)

    # Model
    binsKp = klf_mag_bins
    binEdges = binsKp[0:-1] + (binsKp[1:] - binsKp[0:-1]) / 2.0
    binEdges = binsKp[0:-1] - np.diff(binsKp) / 2.0

    # Plotting
    py.clf()
    (n, b, p) = py.hist(modelStars, bins=binEdges, histtype='step', 
                        color='green', weights=weights, 
                        label='Salpeter Model', align='mid')
    py.gca().set_yscale('log')


    #     py.plot(bartkoKp, bartkoModel, 'r-')
    bartkoErr = np.zeros(len(bartkoKp), dtype=float) + 0.02
    py.errorbar(bartkoKp, bartkoKLF, fmt='bs', yerr=bartkoErr,
                label='VLT (Bartko et al. 2010)')
    py.errorbar(bartkoKp, bartkoKLF, fmt='bs', xerr=bartkoErrX, capsize=0)

    py.errorbar(d.Kp[idx], d.KLF_ext_cmp_sp_im[idx], fmt='ro-', 
                xerr=magBin/2.0, capsize=0)
    py.errorbar(d.Kp[idx], d.KLF_ext_cmp_sp_im[idx], fmt='ro-', 
                yerr=d.eKLF_ext_cmp_sp_im[idx], 
                label='Keck (Lu et al. in prep.)')


    py.gca().set_yscale('log')
    py.legend(loc='upper left', numpoints=1)
    py.ylim(0.01, 2)
    py.xlim(8.5, 16.5)
    py.xlabel('Kp magnitude')
    py.ylabel('Stars / (arcsec^2 mag)')
    py.title('logAge=%.2f, IMF slope=%.2f, %.1f<r<=%.1f' % 
             (logAge, imfSlope, rmin, rmax), fontsize=14)
    py.savefig('%splots/klf_vs_bartko_r_%.1f_%.1f.png' % 
               (workDir, rmin, rmax))

    # Plot a version without the model
    py.clf()
    py.errorbar(bartkoKp, bartkoKLF, fmt='bs', yerr=bartkoErr,
                label='VLT (Bartko et al. 2010)')
    py.errorbar(bartkoKp, bartkoKLF, fmt='bs', xerr=bartkoErrX, capsize=0)

    py.errorbar(d.Kp[idx], d.KLF_ext_cmp_sp_im[idx], fmt='ro-', 
                xerr=magBin/2.0, capsize=0)
    py.errorbar(d.Kp[idx], d.KLF_ext_cmp_sp_im[idx], fmt='ro-', 
                yerr=d.eKLF_ext_cmp_sp_im[idx], 
                label='Keck (Lu et al. in prep.)')


    py.gca().set_yscale('log')
    py.legend(loc='upper left', numpoints=1)
    py.ylim(0.01, 2)
    py.xlim(8.5, 16.5)
    py.xlabel('Kp magnitude')
    py.ylabel('Stars / (arcsec^2 mag)')
    py.title('logAge=%.2f, IMF slope=%.2f, %.1f<r<=%.1f' % 
             (logAge, imfSlope, rmin, rmax), fontsize=14)
    py.savefig('%splots/klf_vs_bartko_nomod_r_%.1f_%.1f.png' % 
               (workDir, rmin, rmax))
    py.savefig('%splots/klf_vs_bartko_nomod_r_%.1f_%.1f.ps' % 
               (workDir, rmin, rmax))


    # Plotting non-extinction corrected for comparison
    py.clf()
    bartkoErr = np.zeros(len(bartkoKp), dtype=float) + 0.02
    py.errorbar(bartkoKp, bartkoKLF, fmt='bs', yerr=bartkoErr,
                label='VLT (Bartko et al. 2010)')
    py.errorbar(bartkoKp, bartkoKLF, fmt='bs', xerr=bartkoErrX, capsize=0)
    py.gca().set_yscale('log')

    py.errorbar(d.Kp[idx], d.KLF_ext[idx], fmt='ro-', 
                xerr=magBin/2.0, capsize=0)
    py.errorbar(d.Kp[idx], d.KLF_ext[idx], fmt='ro-', 
                yerr=d.eKLF_ext[idx], 
                label='Keck Observed')

    py.errorbar(d.Kp[idx], d.KLF_ext_cmp_sp_im[idx], fmt='go-', 
                xerr=magBin/2.0, capsize=0)
    py.errorbar(d.Kp[idx], d.KLF_ext_cmp_sp_im[idx], fmt='go-', 
                yerr=d.eKLF_ext_cmp_sp_im[idx], 
                label='Keck Completeness Corr.')

    py.gca().set_yscale('log')
    py.legend(loc='upper left', numpoints=1)
    py.ylim(0.01, 2)
    py.xlim(8.5, 16.5)
    py.xlabel('Kp magnitude')
    py.ylabel('Stars / (arcsec^2 mag)')
    py.title('logAge=%.2f, IMF slope=%.2f, %.1f<r<=%.1f' % 
             (logAge, imfSlope, rmin, rmax), fontsize=14)
    py.savefig('%splots/klf_vs_bartko_cmp_r_%.1f_%.1f.png' % 
               (workDir, rmin, rmax))

def plot_klf_vs_bartko_by_radius_12A(rmin=0, rmax=30, logAge=6.78, imfSlope=1.35):
    d = load_klf_by_radius(rmin, rmax, mask_for_log=True)

    magBin = d.Kp[1] - d.Kp[0]

    idx = np.where(d.Kp <= 16)[0]

    # Rescale Bartko to match our first two bins
    bartkoKLF = np.array([0.034, 0.08, 0.19, 0.20, 0.14, 0.18, 0.10])
    bartkoKp = np.array( [  9.5, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0])
    bartkoModel = np.array([0.0089, 0.02, 0.07, 0.24, 0.26, 0.4, 0.7])
    bartkoErrX = np.array([1, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5])
    bartkoErr = np.zeros(len(bartkoKp), dtype=float) + 0.02

    scaleFactor = 1.0
    bartkoKLF = bartkoKLF * scaleFactor

    # Load up a model luminosity function
    modelStars = model_klf(logAge=logAge, imfSlope=imfSlope)
    scaleFactor = 700.0  # equivalent to 5000 stars / pi * 15" * 15"
    weights = np.ones(len(modelStars), dtype=float)
    weights *= scaleFactor / len(modelStars)

    # Model
    binsKp = klf_mag_bins
    binSize = binsKp[1] - binsKp[0]
    binEdges = np.append(binsKp - binSize/2.0, binsKp[-1] + binSize/2.0)
    print(binEdges)
    #binEdges = binsKp[0:-1] + (binsKp[1:] - binsKp[0:-1]) / 2.0

    # Plot a version without the model
    py.clf()
    py.errorbar(d.Kp[idx], d.KLF_ext_cmp_sp_im[idx], fmt='ro-', 
                xerr=magBin/2.0, capsize=0)
    py.errorbar(d.Kp[idx], d.KLF_ext_cmp_sp_im[idx], fmt='ro-', 
                yerr=d.eKLF_ext_cmp_sp_im[idx], 
                label='Keck')

    py.errorbar(bartkoKp, bartkoKLF, fmt='bs', yerr=bartkoErr,
                label='VLT')
    py.errorbar(bartkoKp, bartkoKLF, fmt='bs', xerr=bartkoErrX, capsize=0)


    py.gca().set_yscale('log')
    py.legend(loc='upper left', numpoints=1)
    py.ylim(0.01, 2)
    py.xlim(8.5, 16.5)
    py.xlabel('Kp magnitude')
    py.ylabel('Stars / (arcsec^2 mag)')
    py.title('logAge=%.2f, IMF slope=%.2f, %.1f<r<=%.1f' % 
             (logAge, imfSlope, rmin, rmax), fontsize=14)
    py.savefig('%splots/klf_vs_bartko_nomod_r_%.1f_%.1f_12A.png' % 
               (workDir, rmin, rmax))
    py.savefig('%splots/klf_vs_bartko_nomod_r_%.1f_%.1f_12A.ps' % 
               (workDir, rmin, rmax))

    (n, b, p) = py.hist(modelStars, bins=binEdges, histtype='step', 
                        color='green', weights=weights, 
                        label='Salpeter Model', align='mid')

    py.legend(loc='upper left', numpoints=1)
    py.savefig('%splots/klf_vs_bartko_r%.1f_%.1f_12A.png' % 
               (workDir, rmin, rmax))
    py.savefig('%splots/klf_vs_bartko_r%.1f_%.1f_12A.ps' % 
               (workDir, rmin, rmax))

def plot_klf_vs_bartko_by_radius_paper(rmin=0, rmax=30):
    d = load_klf_by_radius(rmin, rmax, mask_for_log=True)

    magBin = d.Kp[1] - d.Kp[0]

    idx = np.where(d.Kp < 16)[0]

    # Rescale Bartko to match our first two bins
    bartkoKLF = np.array([0.034, 0.08, 0.19, 0.20, 0.14, 0.18, 0.10])
    bartkoKp = np.array( [  9.5, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0])
    bartkoModel = np.array([0.0089, 0.02, 0.07, 0.24, 0.26, 0.4, 0.7])
    bartkoErrX = np.array([1, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5])
    bartkoErr = np.zeros(len(bartkoKp), dtype=float) + 0.02

    scaleFactor = 1.0
    bartkoKLF = bartkoKLF * scaleFactor

    # # Load up a model luminosity function
    # modelStars = model_klf(logAge=logAge, imfSlope=imfSlope)
    # scaleFactor = 700.0  # equivalent to 5000 stars / pi * 15" * 15"
    # weights = np.ones(len(modelStars), dtype=float)
    # weights *= scaleFactor / len(modelStars)

    # Model
    binsKp = klf_mag_bins
    binSize = binsKp[1] - binsKp[0]
    binEdges = np.append(binsKp - binSize/2.0, binsKp[-1] + binSize/2.0)

    # Plot a version without the model
    py.clf()
    py.subplots_adjust(left=0.2)
    py.errorbar(d.Kp[idx], d.KLF_ext_cmp_sp_im[idx], fmt='ro-', 
                xerr=magBin/2.0, capsize=0, label='This Work')
    py.errorbar(d.Kp[idx], d.KLF_ext_cmp_sp_im[idx], fmt='ro-', 
                yerr=d.eKLF_ext_cmp_sp_im[idx])

    py.errorbar(bartkoKp, bartkoKLF, fmt='bs', yerr=bartkoErr,
                label='Bartko et al. 2010')
    py.errorbar(bartkoKp, bartkoKLF, fmt='bs', xerr=bartkoErrX, capsize=0)


    #py.gca().set_yscale('log')
    py.legend(loc='upper left', numpoints=1)
    py.ylim(0.0, 0.35)
    py.xlim(8.5, 16.5)
    py.xlabel('Kp magnitude')
    py.ylabel(r'Stars / (asec$^2$ mag)')
    # py.title("%.1f'' < r <= %.1f''" % 
    #          (rmin, rmax), fontsize=14)
    py.savefig('%splots/klf_vs_bartko_r_%.1f_%.1f_paper.png' % 
               (workDir, rmin, rmax))
    py.savefig('%splots/klf_vs_bartko_r_%.1f_%.1f_paper.eps' % 
               (workDir, rmin, rmax))

    # (n, b, p) = py.hist(modelStars, bins=binEdges, histtype='step', 
    #                     color='green', weights=weights, 
    #                     label='Salpeter Model', align='mid')

    # py.legend(loc='upper left', numpoints=1)
    # py.savefig('%splots/klf_vs_bartko_r%.1f_%.1f_12A.png' % 
    #            (workDir, rmin, rmax))
    # py.savefig('%splots/klf_vs_bartko_r%.1f_%.1f_12A.ps' % 
    #            (workDir, rmin, rmax))

def plot_klf_by_radius(rmin=0, rmax=30):
    d = load_klf_by_radius(rmin, rmax, mask_for_log=True)

    magBin = d.Kp[1] - d.Kp[0]

    idx = np.where(d.Kp <= 16)[0]

    # By Number Plot
    py.clf()
    py.errorbar(d.Kp[idx], d.N[idx], fmt='bo-', 
                xerr=magBin/2.0, capsize=0)
    py.errorbar(d.Kp[idx], d.N[idx], fmt='bo-', 
                yerr=d.eN[idx], 
                label='Raw')

    py.errorbar(d.Kp[idx], d.N_ext[idx], fmt='go-', 
                xerr=magBin/2.0, capsize=0)
    py.errorbar(d.Kp[idx], d.N_ext[idx], fmt='go-', 
                yerr=d.eN_ext[idx], 
                label='Ext')

    py.errorbar(d.Kp[idx], d.N_ext_cmp_sp[idx], fmt='r-', 
                xerr=magBin/2.0, capsize=0)
    py.errorbar(d.Kp[idx], d.N_ext_cmp_sp[idx], fmt='ro-', 
                yerr=d.eN_ext_cmp_sp[idx], 
                label='Ext + SpCmp')

    py.errorbar(d.Kp[idx], d.N_ext_cmp_sp_im[idx], fmt='k-', 
                xerr=magBin/2.0, capsize=0)
    py.errorbar(d.Kp[idx], d.N_ext_cmp_sp_im[idx], fmt='ko-', 
                yerr=d.eN_ext_cmp_sp_im[idx], 
                label='Ext + SpCmp + ImCmp')

    py.gca().set_yscale('log')
    py.legend(loc='upper left', numpoints=1)
    py.xlim(8.5, 16.5)
    py.ylim(0.5, 500)
    py.xlabel('Kp magnitude')
    py.ylabel('Stars')
    py.title('%.1f<r<=%.1f' % (rmin, rmax), fontsize=14)
    py.savefig('%splots/klf_counts_r_%.1f_%.1f.png' % 
               (workDir, rmin, rmax))


    # By Density Plot
    py.clf()
    py.errorbar(d.Kp[idx], d.KLF[idx], fmt='bo-', 
                xerr=magBin/2.0, capsize=0)
    py.errorbar(d.Kp[idx], d.KLF[idx], fmt='bo-', 
                yerr=d.eKLF[idx], 
                label='Raw')

    py.errorbar(d.Kp[idx], d.KLF_ext[idx], fmt='go-', 
                xerr=magBin/2.0, capsize=0)
    py.errorbar(d.Kp[idx], d.KLF_ext[idx], fmt='go-', 
                yerr=d.eKLF_ext[idx], 
                label='Ext')

    py.errorbar(d.Kp[idx], d.KLF_ext_cmp_sp[idx], fmt='r-', 
                xerr=magBin/2.0, capsize=0)
    py.errorbar(d.Kp[idx], d.KLF_ext_cmp_sp[idx], fmt='ro-', 
                yerr=d.eKLF_ext_cmp_sp[idx], 
                label='Ext + SpCmp')

    py.errorbar(d.Kp[idx], d.KLF_ext_cmp_sp_im[idx], fmt='k-', 
                xerr=magBin/2.0, capsize=0)
    py.errorbar(d.Kp[idx], d.KLF_ext_cmp_sp_im[idx], fmt='ko-', 
                yerr=d.eKLF_ext_cmp_sp_im[idx], 
                label='Ext + SpCmp + ImCmp')

    py.gca().set_yscale('log')
    py.legend(loc='upper left', numpoints=1)
    py.xlim(8.5, 16.5)
    py.ylim(0.005, 5)
    py.xlabel('Kp magnitude')
    py.ylabel('Stars / [asec^2 mag]')
    py.title('%.1f<r<=%.1f' % (rmin, rmax), fontsize=14)
    py.savefig('%splots/klf_r_%.1f_%.1f.png' % 
               (workDir, rmin, rmax))


def plot_klf_by_radius_noWR(rmin=0, rmax=30):
    d = load_klf_by_radius(rmin, rmax, mask_for_log=True)

    magBin = d.Kp[1] - d.Kp[0]

    idx = np.where(d.Kp <= 16)[0]

    # By Number Plot
    py.clf()
    py.errorbar(d.Kp[idx], d.N_noWR[idx], fmt='bo-', 
                xerr=magBin/2.0, capsize=0)
    py.errorbar(d.Kp[idx], d.N_noWR[idx], fmt='bo-', 
                yerr=d.eN_noWR[idx], 
                label='Raw')

    py.errorbar(d.Kp[idx], d.N_ext_noWR[idx], fmt='go-', 
                xerr=magBin/2.0, capsize=0)
    py.errorbar(d.Kp[idx], d.N_ext_noWR[idx], fmt='go-', 
                yerr=d.eN_ext_noWR[idx], 
                label='Ext')

    py.errorbar(d.Kp[idx], d.N_ext_cmp_sp_noWR[idx], fmt='r-', 
                xerr=magBin/2.0, capsize=0)
    py.errorbar(d.Kp[idx], d.N_ext_cmp_sp_noWR[idx], fmt='ro-', 
                yerr=d.eN_ext_cmp_sp_noWR[idx], 
                label='Ext + SpCmp')

    py.errorbar(d.Kp[idx], d.N_ext_cmp_sp_im_noWR[idx], fmt='k-', 
                xerr=magBin/2.0, capsize=0)
    py.errorbar(d.Kp[idx], d.N_ext_cmp_sp_im_noWR[idx], fmt='ko-', 
                yerr=d.eN_ext_cmp_sp_im_noWR[idx], 
                label='Ext + SpCmp + ImCmp')

    py.gca().set_yscale('log')
    py.legend(loc='upper left', numpoints=1)
    py.xlim(8.5, 16.5)
    py.ylim(0.5, 500)
    py.xlabel('Kp magnitude')
    py.ylabel('Stars')
    py.title('%.1f<r<=%.1f (no WR)' % (rmin, rmax), fontsize=14)
    py.savefig('%splots/klf_counts_r_%.1f_%.1f_noWR.png' % 
               (workDir, rmin, rmax))


    # By Density Plot
    py.clf()
    py.errorbar(d.Kp[idx], d.KLF_noWR[idx], fmt='bo-', 
                xerr=magBin/2.0, capsize=0)
    py.errorbar(d.Kp[idx], d.KLF_noWR[idx], fmt='bo-', 
                yerr=d.eKLF_noWR[idx], 
                label='Raw')

    py.errorbar(d.Kp[idx], d.KLF_ext_noWR[idx], fmt='go-', 
                xerr=magBin/2.0, capsize=0)
    py.errorbar(d.Kp[idx], d.KLF_ext_noWR[idx], fmt='go-', 
                yerr=d.eKLF_ext_noWR[idx], 
                label='Ext')

    py.errorbar(d.Kp[idx], d.KLF_ext_cmp_sp_noWR[idx], fmt='r-', 
                xerr=magBin/2.0, capsize=0)
    py.errorbar(d.Kp[idx], d.KLF_ext_cmp_sp_noWR[idx], fmt='ro-', 
                yerr=d.eKLF_ext_cmp_sp_noWR[idx], 
                label='Ext + SpCmp')

    py.errorbar(d.Kp[idx], d.KLF_ext_cmp_sp_im_noWR[idx], fmt='k-', 
                xerr=magBin/2.0, capsize=0)
    py.errorbar(d.Kp[idx], d.KLF_ext_cmp_sp_im_noWR[idx], fmt='ko-', 
                yerr=d.eKLF_ext_cmp_sp_im_noWR[idx], 
                label='Ext + SpCmp + ImCmp')

    py.gca().set_yscale('log')
    py.legend(loc='upper left', numpoints=1)
    py.xlim(8.5, 16.5)
    py.ylim(0.005, 5)
    py.xlabel('Kp magnitude')
    py.ylabel('Stars / [asec^2 mag]')
    py.title('%.1f<r<=%.1f (no WR)' % (rmin, rmax), fontsize=14)
    py.savefig('%splots/klf_r_%.1f_%.1f_noWR.png' % 
               (workDir, rmin, rmax))

def plot_klf_by_radius_WR_v_noWR(rmin=0, rmax=30):
    d = load_klf_by_radius(rmin, rmax, mask_for_log=True)

    magBin = d.Kp[1] - d.Kp[0]

    idx = np.where(d.Kp <= 16)[0]

    # By Number Plot
    py.clf()
    py.errorbar(d.Kp[idx], d.N_ext_cmp_sp_im[idx], fmt='b-', 
                xerr=magBin/2.0, capsize=0)
    py.errorbar(d.Kp[idx], d.N_ext_cmp_sp_im[idx], fmt='bo-', 
                yerr=d.eN_ext_cmp_sp_im[idx], 
                label='With WR')

    py.errorbar(d.Kp[idx], d.N_ext_cmp_sp_im_noWR[idx], fmt='r-', 
                xerr=magBin/2.0, capsize=0)
    py.errorbar(d.Kp[idx], d.N_ext_cmp_sp_im_noWR[idx], fmt='ro-', 
                yerr=d.eN_ext_cmp_sp_im_noWR[idx], 
                label='No WR')

    py.gca().set_yscale('log')
    py.legend(loc='upper left', numpoints=1)
    py.xlim(8.5, 16.5)
    py.ylim(0.5, 500)
    py.xlabel('Kp magnitude')
    py.ylabel('Stars')
    py.title('%.1f<r<=%.1f (no WR)' % (rmin, rmax), fontsize=14)
    py.savefig('%splots/klf_counts_r_%.1f_%.1f_WR_v_noWR.png' % 
               (workDir, rmin, rmax))

    # By Density Plot
    py.clf()
    py.errorbar(d.Kp[idx], d.KLF_ext_cmp_sp_im[idx], fmt='b-', 
                xerr=magBin/2.0, capsize=0)
    py.errorbar(d.Kp[idx], d.KLF_ext_cmp_sp_im[idx], fmt='bo-', 
                yerr=d.eKLF_ext_cmp_sp_im_noWR[idx], 
                label='With WR')

    py.errorbar(d.Kp[idx], d.KLF_ext_cmp_sp_im_noWR[idx], fmt='r-', 
                xerr=magBin/2.0, capsize=0)
    py.errorbar(d.Kp[idx], d.KLF_ext_cmp_sp_im_noWR[idx], fmt='ro-', 
                yerr=d.eKLF_ext_cmp_sp_im_noWR[idx], 
                label='No WR')

    py.gca().set_yscale('log')
    py.legend(loc='upper left', numpoints=1)
    py.xlim(8.5, 16.5)
    py.ylim(0.005, 5)
    py.xlabel('Kp magnitude')
    py.ylabel('Stars / [asec^2 mag]')
    py.title('%.1f<r<=%.1f (no WR)' % (rmin, rmax), fontsize=14)
    py.savefig('%splots/klf_r_%.1f_%.1f_WR_v_noWR.png' % 
               (workDir, rmin, rmax))

    
def plot_klf_vs_model_by_radius(rmin=0, rmax=30, logAge=6.78, imfSlope=2.35,
                                clusterMass=10**4, makeMultiples=True):
    d = load_klf_by_radius(rmin, rmax, mask_for_log=True)

    magBin = d.Kp[1] - d.Kp[0]

    idx = np.where(d.Kp < 16)[0]

    # Load up a model luminosity function
    modelStars = model_klf(logAge=logAge, imfSlope=imfSlope, clusterMass=10**5,
                           makeMultiples=makeMultiples)
    area = 116.098  # arcsec^2
    scaleFactor = (clusterMass / 1.0e5) / area
    weights = np.ones(len(modelStars), dtype=float)
    weights *= scaleFactor


    # Plot the model... remember, no WR stars.
    binsKp = klf_mag_bins
    binEdges = binsKp[0:-1] + (binsKp[1:] - binsKp[0:-1]) / 2.0
    py.clf()
    (n, b, p) = py.hist(modelStars, bins=binEdges, histtype='step', weights=weights,
                        color='green', label='Salpeter Model', align='mid')
    print(n)
    py.gca().set_yscale('log')

    # Plot the observations
    py.errorbar(d.Kp[idx], d.KLF_ext_cmp_sp_im_noWR[idx], fmt='ro-', 
                xerr=magBin/2.0, capsize=0)
    py.errorbar(d.Kp[idx], d.KLF_ext_cmp_sp_im_noWR[idx], fmt='ro-', 
                yerr=d.eKLF_ext_cmp_sp_im_noWR[idx], 
                label='Observed')

    py.gca().set_yscale('log')
    py.legend(loc='upper left', numpoints=1)
    py.ylim(0.01, 1)
    py.xlim(8.5, 15.5)
    py.xlabel('Kp magnitude')
    py.ylabel('Stars / (arcsec^2 mag)')
    py.title('logAge=%.2f, IMF slope=%.2f, %.1f<r<=%.1f' % 
             (logAge, imfSlope, rmin, rmax), fontsize=14)
    py.savefig('%splots/klf_vs_model_r_%.1f_%.1f_noWR.png' % 
               (workDir, rmin, rmax))

def plot_klf_vs_old_models(rmin=0, rmax=30):
    makeMultiples = True
    logAge = 6.78
    
    # Salpeter
    clusterMassSP = 1.3e4
    imfSlopeSP = 2.35
    
    # Top-Heavy
    clusterMassTH = 2.7e4
    imfSlopeTH = 0.45
                           
    d = load_klf_by_radius(rmin, rmax, mask_for_log=True)

    magBin = d.Kp[1] - d.Kp[0]

    idx = np.where(d.Kp < 16)[0]

    # Load up a model luminosity functions
    modelStarsSP = model_klf(logAge=logAge, imfSlope=imfSlopeSP, clusterMass=1e6,
                             makeMultiples=makeMultiples)
    modelStarsTH = model_klf(logAge=logAge, imfSlope=imfSlopeTH, clusterMass=1e6,
                             makeMultiples=makeMultiples)
    area = 116.098  # arcsec^2
    scaleFactorSP = (clusterMassSP / 1e6) / area
    scaleFactorTH = (clusterMassTH / 1e6) / area
    weightsSP = np.ones(len(modelStarsSP), dtype=float)
    weightsTH = np.ones(len(modelStarsTH), dtype=float)
    weightsSP *= scaleFactorSP
    weightsTH *= scaleFactorTH

    # Plot the model... remember, no WR stars.
    binsKp = klf_mag_bins
    binEdges = binsKp[0:-1] + (binsKp[1:] - binsKp[0:-1]) / 2.0
    py.clf()
    (n1, b1, p1) = py.hist(modelStarsSP, bins=binEdges, histtype='step',
                           weights=weightsSP,
                           color='green', label='Salpeter Model', align='mid',
                           linewidth=1.5)
    (n1, b1, p1) = py.hist(modelStarsTH, bins=binEdges, histtype='step',
                           weights=weightsTH,
                           color='blue', label='Top Heavy Model', align='mid',
                           linewidth=1.5, linestyle='dashed')

    # Plot the observations
    py.errorbar(d.Kp[idx], d.KLF_ext_cmp_sp_im_noWR[idx], fmt='ro-', 
                xerr=magBin/2.0, capsize=0, linewidth=2)
    py.errorbar(d.Kp[idx], d.KLF_ext_cmp_sp_im_noWR[idx], fmt='ro-', 
                yerr=d.eKLF_ext_cmp_sp_im_noWR[idx], linewidth=2,
                label='Observed')

    py.legend(loc='upper left', numpoints=1)
    py.ylim(0, 0.5)
    py.xlim(8.5, 15.5)
    py.xlabel('Kp magnitude')
    py.ylabel('Stars / (arcsec^2 mag)')
    py.title('Age = %d Myr' % (10**(logAge-6)), fontsize=14)

    outRoot = '%splots/klf_vs_old_models' % workDir
    py.savefig('{0}.png'.format(outRoot))
    py.savefig('{0}_color.eps'.format(outRoot))
    convert = 'convert {0}.png -colorspace gray {0}.eps'.format(outRoot)
    os.system(convert)

    # Save the models and data to a text file for later re-use.
    tSP = Table([modelStarsSP, weightsSP], names=['Kp', 'weight'],
                meta={'IMFSLOPE': imfSlopeSP,
                      'MCLUST': clusterMassSP,
                      'AREA': area,
                      'LOGAGE': logAge,
                      'MULTI': makeMultiples})
    tTH = Table([modelStarsTH, weightsTH], names=['Kp', 'weight'],
                meta={'IMFSLOPE': imfSlopeTH,
                      'MCLUST': clusterMassTH,
                      'AREA': area,
                      'LOGAGE': logAge,
                      'MULTI': makeMultiples})

    tSP.write(outRoot + '_SP_model.fits', overwrite=True)
    tTH.write(outRoot + '_TH_model.fits', overwrite=True)

    return



def test_wr_transformed_radius():
    """
    We have to calculate a transformed radius for WR stars in order
    to flux calibrate synthetic spectra. This depends on some parameters
    of the WR wind and atmosphere. Lets test whether we can assume
    a uniform terminal wind velocity.
    """
    radius = np.arange(5, 65, 1.0)
    massloss = np.arange(-5.5, -4.0, 0.1)
    vterm = np.arange(500, 2500, 500.0)
    
    def r_transform(v_terminal, dm, clumpFactor=3):
        return

    for mm in range(len(massloss)):
        for vv in range(len(vterm)):
            pass
    
def getOsirisFields(dbCursor=None):
    # Connect to the database if a cursort objects isn't already
    # passed in.
    if dbCursor is None:
        con = mdb.connect(host='galaxy1.astro.ucla.edu',user='dbread',passwd=pw,db='gcg')
        df_fields = pd.read_sql_query("SELECT name,x_vertices,y_vertices FROM fields WHERE name NOT LIKE 'Verification%' AND nstars IS NOT NULL", con)
        rows = Table.from_pandas(df_fields)   
        con.close()
    

    names = []
    xverts = []
    yverts = []
    xyverts = []

    for row in rows:
        fieldName = row['name']
        if 'Imaging' in fieldName:
            continue

        names.append(str(fieldName))
        
        xvertsTmp = np.array([float(ff) for ff in row['x_vertices'].split(',')])
        yvertsTmp = np.array([float(ff) for ff in row['y_vertices'].split(',')])
        xverts.append( xvertsTmp )
        yverts.append( yvertsTmp )
        xyverts.append( np.column_stack((xvertsTmp, yvertsTmp)) )
        
    fieldInfo = objects.DataHolder()
    fieldInfo.name = names
    fieldInfo.xverts = xverts
    fieldInfo.yverts = yverts
    fieldInfo.xyverts = xyverts

    return fieldInfo


def load_unknown_sims():
    """
    Load the unknownSims table from the database so that we can 
    estimate the spectroscopic completeness for the young stars.
    """
    query = 'select name, field, '
    query += 'probYngSimPrior, probYngSimPriorErr, '
    query += 'probOldSimPrior, probOldSimPriorErr '
    query += 'from unknownSims'

    # Create a connection to the database file and create a cursor
    pwFile='/g/lu/scratch/siyao/other/pw.txt'
    pw = open(pwFile).read().split()[0]
    con = mdb.connect(host='galaxy1.astro.ucla.edu',user='dbread',passwd=pw,db='gcg')
    rows = pd.read_sql_query(query, con)

    t = Table.from_pandas(rows)


    # Rename columns
    t.rename_column('probYngSimPrior', 'probYng')
    t.rename_column('probOldSimPrior', 'probOld')
    t.rename_column('probYngSimPriorErr', 'probYngErr')
    t.rename_column('probOldSimPriorErr', 'probOldErr')

    # Set S0-32 values to 0
    idx = np.where(t['name'] == 'S0-32')[0]
    t['probYng'][idx] = 0
    t['probOld'][idx] = 0

    # There are a couple of wierd cases where the denominator is 0.
    # In this case, resort to priors.
    idx = np.where(np.isnan(t['probYngErr']) == True)[0]
    if len(idx) > 0:
        t['probYngErr'][idx] = 0
        t['probOldErr'][idx] = 0

    return t

# We will rescale these by our radial prior as well determined
# from the observed ratio of young to old stars at a given radius
def priorsAtRadius(radius):
    """
    Function will take original priors for probability

    of early-type (yng) and late-type (old) classification
    and rescale them based on what we know about the
    observed fraction of young/old as a function of radius.
    """
    priorYng = 0.6
    priorOld = 0.4
    
    #g = 1.10 * radius**-0.63
    g = 1.039 * radius**-0.86

    denom = priorYng * g + priorOld

    priorYngRad = priorYng * g / denom
    priorOldRad = priorOld / denom

    return (priorYngRad, priorOldRad)

    
def get_extinctions_for_stars(nirc2):
    # Create a new array to store extinction values
    nirc2.AKs = np.zeros(len(nirc2.x), dtype=float)

    # Load up the extinction map from Schodel 2010
    #schExtinctFile = '/u/jlu/work/gc/imf/extinction/2010schodel_AKs_fg6.fits'
    schExtinctFile = '/g/lu/scratch/siyao/work/3_young/imf/2018Lara_AKs_fg28.fits'
    schExtinct = fits.open(schExtinctFile)

    # Calculate AKs for all the stars in the dp_mosaic
    ext_map = schExtinct[0].data
    ext_hdr = schExtinct[0].header
    ext_map = np.array(ext_map, dtype=float)

    ext_scaleX = ext_hdr['CD1_1'] * 3600.0
    ext_scaleY = ext_hdr['CD2_2'] * 3600.0
    ext_sgraX = ext_hdr['CRPIX1']
    ext_sgraY = ext_hdr['CRPIX2']

    ext_mapX = np.arange(ext_map.shape[1], dtype=float)
    ext_mapY = np.arange(ext_map.shape[0], dtype=float)

    ext_mapX = (ext_mapX - ext_sgraX) * ext_scaleX
    ext_mapY = (ext_mapY - ext_sgraY) * ext_scaleY

    for ss in range(len(nirc2.x)):
        # Find the closest pixel to each star.
        dx = np.abs(ext_mapX - nirc2.x[ss])
        dy = np.abs(ext_mapY - nirc2.y[ss])

        xid = dx.argmin()
        yid = dy.argmin()

        nirc2.AKs[ss] = ext_map[yid, xid]


def get_wolf_rayet_stars():
    """
    Load up a list of star names for all known
    Wolf-Rayet stars from our database.
    """
    pwFile='/g/lu/scratch/siyao/other/pw.txt'
    pw = open(pwFile).read().split()[0]
    con = mdb.connect(host='galaxy1.astro.ucla.edu',user='dbread',passwd=pw,db='gcg')
    df_wr_stars = pd.read_sql_query("SELECT name FROM stars WHERE spec_type='WR'", con)
    wr_stars = df_wr_stars.name.values

    return wr_stars


def poisson_error(n):
    if type(n) != np.ndarray:
        n = np.array([n])

    err = np.sqrt(n)
    err[err == 0] = 1

    if len(n) == 1:
        return err[0]
    else:
        return err

    
def kp_error_distribution(plot=False):
    """
    Examine the observed Kp error distribution and find an appropriate
    analytic form (if possible). See if there are any magnitude
    dependencies.
    """
    # Load spectral identifications
    s = load_spec_id_all_stars(flatten=True)

    # Get rid of the Wolf-Rayet Stars
    idx = np.where(s.isWR_yng == False)[0]

    kp = s.kp_yng[idx]
    kperr = s.kperr_yng[idx]

    if plot:
        # Plot errors vs. brightness
        py.clf()
        py.plot(kp, kperr, 'k.')
        py.xlabel('Kp (mag)')
        py.ylabel('Kp Uncertainty')
        py.title('Young Stars')
        py.savefig('%splots/kp_error_vs_kp.png' % (workDir))

        # Plot kp error distribution
        bins = np.arange(0, 0.5, 0.025)
        gamma = stats.gamma

        py.clf()
        n, bins, patches = py.hist(kperr, bins=bins)

        gamma.fit(n)
        
        # py.plot(bins, poisson.pdf(bins) * n.sum())
        py.xlabel('Kp Uncertainty')
        py.ylabel('Number of Young Stars')
        py.savefig('%splots/kp_errors_hist.png' % (workDir))
    
    return kperr
        
def plot_pdf_sgra_dist():
    """
    Plot up the output of efit to see if the PDF(distance)
    can be approximated with a gaussian.
    """
    sgraDir = '/u/jlu/work/gc/imf/klf/sgra/'
    efitFile = sgraDir + 'sgra_mass_dist_pdf.log'

    sgra = atpy.Table(efitFile, type='ascii')
    dist = sgra['col1']

    # Fit a gaussian:
    dist_mean = dist.mean()
    dist_std = dist.std()

    py.clf()
    n, bins, patches = py.hist(dist, histtype='step',
                               bins=100, normed=True,
                               label='Observed')

    binCenters = bins[:-1] + (bins[1:] - bins[:-1])/2.0
    normValues = stats.norm.pdf(binCenters, loc=dist_mean, scale=dist_std)
    py.plot(binCenters, normValues, 'k--', label='Normal Fit')
    py.xlabel('Distance (pc)')
    py.ylabel('Probability Density Function')
    leg = py.legend()
    py.setp(leg.get_texts(), fontsize=12)
    py.title('Model as d = %d +/- %d pc' % (dist_mean, dist_std))
    py.savefig(sgraDir + 'dist_pdf.png')
    
    # Since we are going to sample from the actual
    # distribution of distances, lets save it off in
    # a convenient pickle file.
    _out = open(sgraDir + 'dist_pdf.cpickle', 'wb')
    pickle.dump(dist, _out)
    _out.close()
    
def plot_binary_properties():
    """
    Plot up the multiplicty fraction and companion star fraction vs.
    primary star mass. I use this to determine how the MF and CSF scale
    with primary mass. The notes for this code are in
    research -> GC IMF -> Bayesian -> Binary Stars
    """
    data = atpy.Table()

    data.add_empty_column('mass', float, shape=7)
    data.add_empty_column('MF', float)
    data.add_empty_column('CSF', float)
    data.add_empty_column('Ref', 'S50')

    data['mass'][0] = 0.175
    data['MF'][0] = 0.16
    data['CSF'][0] = 0.16
    data['Ref'][0] = 'Lafreniere+ 2008'

    data['mass'][1] = 0.39
    data['MF'][1] = 0.33
    data['CSF'][1] = 0.37
    data['Ref'][1] = 'Lafreniere+ 2008'

    data['mass'][2] = 0.915
    data['MF'][2] = 0.38
    data['CSF'][2] = 0.50
    data['Ref'][2] = 'Lafreniere+ 2008'

    data['mass'][3] = 2.14
    data['MF'][3] = 0.63
    data['CSF'][3] = 0.75
    data['Ref'][3] = 'Lafreniere+ 2008'

    data['mass'][4] = 12.7
    data['MF'][4] = 1.0
    data['CSF'][4] = 1.5
    data['Ref'][4] = 'Preibisch+1999'

    data['mass'][5] = 16.8  # Mean of Kiminki+ 2007 sample
    data['MF'][5] = 1.0
    data['CSF'][5] = None
    data['Ref'][5] = 'Kobulnicky+ 2008'

    data['mass'][6] = 3.4   # Mean of Rizzuto+ 2011 sample
    data['MF'][6] = 0.85    # 2 sigma limit to binary fraction in Kouwenhaven+ 2007
    data['CSF'][6] = None
    data['Ref'][6] = 'Kouwenhoven+ 2007'

    # Exclude... solar neighborhood (older)
    # Dynamical processing is an issue.
    # data['mass'][7] = 1.0
    # data['MF'][7] = 0.46
    # data['CSF'][7] = 0.33
    # data['Ref'][7] = 'Raghavan+ 2010'

    data.sort('mass')

    py.clf()
    py.semilogx(data['mass'], data['MF'], 'bs', label='MF')
    idx_MF = np.where(np.isnan(data['MF']) == False)[0]
    idx = np.where(np.isnan(data['CSF']) == False)[0]
    py.plot(data['mass'][idx], data['CSF'][idx], 'ro', label='CSF')
    py.xlabel('Mass')
    py.ylabel('Fraction')
    py.legend(loc='upper left', numpoints=1)
    py.ylim(0, 2)
    py.savefig(workDir + 'plots/binary_properties.png')

    # Fit a functional form to the two curves. They are powerlaws with truncation
    # above f = 1.0 for the multiplicity fraction.
    # Multiplicity Fraction
    log_m = np.log10(data['mass'])
    log_mf = np.log10(data['MF'])
    log_csf = np.log10(data['CSF'])
    # log_mf_err = np.log10(data['MF'] * 0.2)
    # log_csf_err = np.log10(data['CSF'] * 0.2)
    log_mf_err = np.log10(np.zeros(len(data['CSF']), dtype=float) + 0.1)
    log_csf_err = np.log10(np.zeros(len(data['CSF']), dtype=float) + 0.1)

    def fitfunc_mf(p, x):
        val = p[0] + (x * p[1])
        val[val > 0] = 0
        return val

    def fitfunc_csf(p, x):
        val = p[0] + (x * p[1])
        return val

    errfunc_mf = lambda p, x, y, err: (y - fitfunc_mf(p, x)) / err
    errfunc_csf = lambda p, x, y, err: (y - fitfunc_csf(p, x)) / err

    pinit = [0.0, 1.0]
    out_mf = optimize.leastsq(errfunc_mf, pinit,
                              args=(log_m[idx_MF], log_mf[idx_MF], log_mf_err[idx_MF]),
                              full_output=1)
    out_csf = optimize.leastsq(errfunc_csf, pinit,
                               args=(log_m[idx], log_csf[idx], log_csf_err[idx]),
                               full_output=1)

    mfParams = out_mf[0]
    mfCovar = out_mf[1]
    mfAmp = 10.0**mfParams[0]
    mfIndex = mfParams[1]
    mfAmpErr = math.sqrt( mfCovar[0][0] ) * mfAmp
    mfIndexErr = math.sqrt( mfCovar[1][1] )

    csfParams = out_csf[0]
    csfCovar = out_csf[1]
    csfAmp = 10.0**csfParams[0]
    csfIndex = csfParams[1]
    csfAmpErr = math.sqrt( csfCovar[0][0] ) * csfAmp
    csfIndexErr = math.sqrt( csfCovar[1][1] )

    print(('MF  power-law amp = %.1e +/- %.1e and index = %.2f +/- %.2f' % \
        (mfAmp, mfAmpErr, mfIndex, mfIndexErr)))
    print(('CSF power-law amp = %.1e +/- %.1e and index = %.2f +/- %.2f' % \
        (csfAmp, csfAmpErr, csfIndex, csfIndexErr)))

    def plaw_mf(x, amp, index):
        val = amp * x**index
        val[val > 1] = 1
        return val

    def plaw_csf(x, amp, index):
        val = amp * x**index
        val[val > 3] = 3
        return val
    
    modelMass = np.arange(0.1, 100, 0.1)
    
    py.clf()
    py.loglog(data['mass'], data['MF'], 'bs', label='MF Data', ms=10)
    py.plot(data['mass'][idx], data['CSF'][idx], 'ro', label='CSF Data', ms=9)
    py.plot(modelMass, plaw_mf(modelMass, mfAmp, mfIndex), 'b--', label='MF Fit')
    py.plot(modelMass, plaw_csf(modelMass, csfAmp, csfIndex), 'r-', label='CSF Fit')
    py.xlabel('Mass')
    py.ylabel('Fraction')
    py.legend(loc='upper left', numpoints=1)
    py.ylim(0, 3.1)

    outRoot = '{0}plots/binary_properties_fit'.format(workDir)
    py.savefig('{0}.png'.format(outRoot))
    py.savefig('{0}_color.eps'.format(outRoot))
    convert = ' convert {0}.png -colorspace Gray {0}.eps'.format(outRoot)
    os.system(convert)
    
    return data
    
def kiminki2007():
    """
    Read in the table of data from Kiminki+ 2007 on Cyg OB2 to
    extract the mean mass of the objects reported.
    """
    tableFile = '/u/jlu/work/gc/imf/literature/'
    tableFile += 'kiminki_etal2007_table5.txt'
    foo = atpy.Table(tableFile, type='ascii', quotechar='#')

    meanMass = foo['M0'].mean()
    medianMass = np.median(foo['M0'])

    print('Kiminki+ 2007 Sample Info:')
    print(('     Mean mass = %6.2f' % (meanMass)))
    print(('   Median mass = %6.2f' % (medianMass)))
    
def rizzuto2011():
    """
    Read in the Rizzuto et al. 2011 membership sample for
    Upper Sco. This is approximately the sample used by
    Kouwenhoven+ 2007 to estimate multiplicity fractions.
    """
    rootDir = '/u/jlu/doc/proposals/irtf/2012A/'

    # Read in a reference table for converting between
    # spectral types and effective temperatures.
    ref = atpy.Table(rootDir + 'Teff_SpT_table.txt', type='ascii')

    sp_type = np.array([ii[0] for ii in ref.col1])
    sp_class = np.array([float(ii[1:4]) for ii in ref.col1])
    sp_teff = ref.col2
    
    # Read in the upper sco table
    us = atpy.Table(rootDir + 'rizzuto_simbad_UBVRIJHK.txt', type='ascii', quotechar='#')

    us_sp_type = np.array([ii[0] for ii in us.spectype])
    us_sp_class = np.zeros(len(us_sp_type), dtype=int)
    us_sp_combo = np.zeros(len(us_sp_type), dtype='S2')
    us_sp_teff = np.zeros(len(us_sp_type), dtype=int)

    for ii in range(len(us_sp_class)):
        if (us_sp_type[ii] == "~"):
            us_sp_class[ii] = -1
        else:
            if ((len(us.spectype[ii]) < 2) or
              (us.spectype[ii][1].isdigit() == False)):
                us_sp_class[ii] = 5  # Arbitrarily assigned
            else:
                us_sp_class[ii] = us.spectype[ii][1]

            # Assign effective temperature
            idx = np.where(us_sp_type[ii] == sp_type)[0]
            tdx = np.abs(us_sp_class[ii] - sp_class[idx]).argmin()
            us_sp_teff[ii] = sp_teff[idx[tdx]]

            us_sp_combo[ii] = us_sp_type[ii] + str(us_sp_class[ii])

    # # Trim out the ones that don't have spectral types
    # idx = np.where((us_sp_type != "~") & (us.K != "~") & (us.J != "~"))[0]
    # print 'Keeping %d of %d with spectral types and K mags.' % \
    #     (len(idx), len(us_sp_type))

    us.add_column('sp_type', us_sp_type)
    us.add_column('sp_class', us_sp_class)
    us.add_column('sp_combo', us_sp_combo)
    us.add_column('sp_teff', us_sp_teff)

    # us = us.rows([idx])

    # Get the unique spectral classes and count how many of each
    # we have in the sample.
    sp_type_uniq = ['O'+str(ii) for ii in range(10)]
    sp_type_uniq = ['B'+str(ii) for ii in range(10)]
    sp_type_uniq += ['A'+str(ii) for ii in range(10)]
    sp_type_uniq += ['F'+str(ii) for ii in range(10)]
    sp_type_uniq += ['G'+str(ii) for ii in range(5)]
    sp_type_count = np.zeros(len(sp_type_uniq), dtype=int)
    sp_type_idx = []


    for ii in range(len(sp_type_uniq)):
        idx = np.where(us.sp_combo == sp_type_uniq[ii])[0]
        if len(idx) > 0:
            sp_type_count[ii] = len(idx)

        sp_type_idx.append(idx)

    # Plot up the distribution of spectral types
    xloc = np.arange(len(sp_type_uniq)) + 1
    py.clf()
    py.bar(xloc, sp_type_count, width=0.5)
    py.xticks(xloc+0.25, sp_type_uniq)
    py.xlim(0.5, xloc.max()+0.5)
    py.xlabel('Spectral Type')
    py.ylabel('Upper Sco Sample')
    py.savefig(rootDir + 'USco_spec_type_hist.png')

    foo = np.arange(len(sp_type_count), dtype=int)
    meanIdx = np.average(foo, weights=sp_type_count)
    print(('Mean Spectral Type is %s' % (sp_type_uniq[int(round(meanIdx))])))

    # Read in spectral type - mass mapping
    lit_root = '/u/jlu/work/gc/imf/literature/'
    spt_mass_file = lit_root + 'allens_table15.8.txt'
    sptMass = atpy.Table(spt_mass_file, type='ascii')

    # Also read in the mapping of spectral type to effective temperature
    sptTeff = atpy.Table(lit_root + 'Teff_SpT_table.txt', type='ascii')
    sptTeff.add_column('spt_type', sptTeff['col1'])
    sptTeff.add_column('spt_subtype', sptTeff['col1'])
    for ii in range(len(sptTeff)):
        sptTeff['spt_type'][ii] = sptTeff['spt_type'][ii][0]
        sptTeff['spt_subtype'][ii] = sptTeff['spt_subtype'][ii][1:4]

    # Get the effective temperature for everthing in the Allan's table
    sptMass.add_column('teff', np.zeros(len(sptMass), dtype=float))
    for ii in range(len(sptMass)):
        spt_type = sptMass['spectype'][ii][0]
        spt_subtype = sptMass['spectype'][ii][1] + '.0'

        idx = np.where((spt_type == sptTeff['spt_type']) &
                       (spt_subtype == sptTeff['spt_subtype']))[0]
        if len(idx) > 0:
            sptMass['teff'][ii] = sptTeff['col2'][idx[0]]


    # Get the effective temperature for everything in the sp_type_uniq table.
    sp_type_teff = np.zeros(len(sp_type_uniq), dtype=int)
    for ii in range(len(sp_type_uniq)):
        tmp = sp_type_uniq[ii] + '.0V'
        idx = np.where(tmp == sptTeff['col1'])[0]

        sp_type_teff[ii] = sptTeff['col2'][idx[0]]

    # Now we have a table with spectype, mass, and teff. We can interpolate
    # to get the masses for all the spectral types.
    sptMass.sort('teff')
    idx = np.where(sptMass['teff'] != 0)[0]
    mass_interp = interpolate.splrep(sptMass['teff'][idx], sptMass['mass'][idx], k=1, s=0)
    sp_type_mass = interpolate.splev(sp_type_teff, mass_interp)

    # Plot up the distribution of spectral types
    py.clf()
    py.plot(sp_type_mass, sp_type_count, 'ks')
    py.xlabel('Mass')
    py.ylabel('Upper Sco Sample')
    #py.savefig(rootDir + 'USco_spec_type_hist.png')

    sp_type_mass_all = np.repeat(sp_type_mass, sp_type_count)
    print(('Mean Mass is   %.1f Msun' % (sp_type_mass_all.mean())))
    print(('Median Mass is %.1f Msun' % (np.median(sp_type_mass))))
    

def plot_mcmc_diagnostics(mcmc_file, chain=0, old_params=False):
    """
    Load up several MCMC traces and examine them for convergence, burn-in, and thinning
    parameters. 
    """
    import pymc
    from jlu.gc.imf import bayesian as b

    if old_params:
        params = ['Mcl', 'age', 'alpha', 'dist']
    else:
        params = ['Mcl', 'logAgeCont', 'alpha', 'dist']
    plot_dir = workDir + 'plots/'

    db = pymc.database.hdf5.load(workDir + mcmc_file)

    # Analyze each variable in the first chain
    suffix = '_' + mcmc_file
    for pp in params:
        print()
        print('********************')
        print(('   ' + pp))
        print('********************')

        traceObj = db.trace(pp, chain=chain)
        
        pymc.Matplot.plot(traceObj, path=plot_dir, suffix=suffix)

        # Geweke Tests
        scores = pymc.geweke(traceObj[:], intervals=20)
        pymc.Matplot.geweke_plot(scores, name='geweke_' + pp, path=plot_dir, suffix=suffix)

        # Raftery-Lewis Tests
        rl_results = pymc.raftery_lewis(traceObj[:], q=0.025, r=0.01)
        
        
    # Look at the traces in 2D parameter plots
    mc_mod = b.pymc_model4()
    mc = pymc.MCMC(mc_mod, db=db)
    nodes = []
    for pp in params:
        node = mc.get_node(pp)
        node.trace = db.trace(pp, chain=chain)
        nodes.append(node)
        
        
    pymc.Matplot.pair_posterior(nodes, path=plot_dir, suffix=suffix)

    # Now compare several

def compare_mcmc_chains(files):
    import pymc
    from jlu.gc.imf import bayesian as b

    params = ['Mcl', 'age', 'alpha', 'dist']
    plot_dir = workDir + 'plots/'

    if type(files) is not list:
        files = [files]

    # Load up all the databases.
    db = []
    for ff in files:
        db.append( pymc.database.hdf5.load(workDir + ff) )

    colors = ['red', 'green', 'blue', 'purple', 'yellow', 'cyan', 'magenta']

    # Analyze each variable in the first chain
    suffix = '_' + files[0]
    for pp in params:
        print()
        print('********************')
        print(('   ' + pp))
        print('********************')

        py.clf()
        minSize = np.Inf
        for ii in range(len(db)):
            traceObj = db[ii].trace(pp)
            py.plot(traceObj[:], 'k-', label=files[ii], color=colors[ii], mec=colors[ii])
            if len(traceObj[:]) < minSize:
                minSize = len(traceObj[:])
                
        py.xlabel('Position in Chain')
        py.ylabel(pp)
        py.legend()
        py.xlim(0, minSize)
        py.savefig('%s/mcmc_compare_chains_%s.png' % (plot_dir, pp))


def plot_model_vs_data(logAge, AKs, distance, imfSlope, clusterMass, yngData=None,
                       outSuffix='', outDir=workDir, legendLabels=None):
    """
    Compare a specific synthetic population to the observed KLF and Number of WR stars.
    """
    from jlu.gc.imf import bayesian as b

    binsKp = klf_mag_bins
    binEdges = binsKp[0:-1] + (binsKp[1:] - binsKp[0:-1]) / 2.0

    if yngData == None:
        yngData = load_yng_data_by_radius(magCut=15.5, trimWR=True)

    if type(logAge) not in [list, np.ndarray]:
        logAge = [logAge]
        AKs = [AKs]
        distance = [distance]
        imfSlope = [imfSlope]
        clusterMass = [clusterMass]

    # Setup 4 different figures
    #    Figure 1 -- with multiples, complete    
    #    Figure 2 -- no multiples, complete    
    #    Figure 3 -- with multiples, incomplete    
    #    Figure 4 -- no multiples, incomplete    
    #py.close('all')
    fig1 = py.figure(1)
    fig2 = py.figure(2)
    fig3 = py.figure(3)
    fig4 = py.figure(4)
    fig1.clf()
    fig2.clf()
    fig3.clf()
    fig4.clf()
    f1 = fig1.gca()
    f2 = fig2.gca()
    f3 = fig3.gca()
    f4 = fig4.gca()

    cm = py.get_cmap('gist_rainbow')
    for i in range(len(logAge)):
        if legendLabels == None:
            if i == 0:
                legLabel = 'Sim'
            else:
                legLabel = None
        else:
            legLabel = legendLabels[i]

        color = cm(1.0 * i / len(logAge))

        c1 = b.model_young_cluster(logAge[i], AKs=AKs[i], distance=distance[i],
                                   imfSlope=imfSlope[i], clusterMass=clusterMass[i],
                                   makeMultiples=True)
        c2 = b.model_young_cluster(logAge[i], AKs=AKs[i], distance=distance[i],
                                   imfSlope=imfSlope[i], clusterMass=clusterMass[i],
                                   makeMultiples=False)

        o1 = b.sim_to_obs_klf(c1, magCut=15.5, withErrors=False)
        o2 = b.sim_to_obs_klf(c2, magCut=15.5, withErrors=False)
        
        f1.hist(c1.mag_noWR, bins=binEdges, histtype='step', color=color,
                label=legLabel, linewidth=2)
        f2.hist(c2.mag_noWR, bins=binEdges, histtype='step', color=color,
                label=legLabel, linewidth=2)

        f3.hist(o1.kp, bins=binEdges, histtype='step', color=color,
                label=legLabel, linewidth=2)
        f4.hist(o2.kp, bins=binEdges, histtype='step', color=color,
                label=legLabel, linewidth=2)

        f1.plot([8.5], c1.num_WR, 'bx', color=color, ms=10)
        f2.plot([8.5], c2.num_WR, 'bx', color=color, ms=10)
        f3.plot([8.5], o1.N_WR, 'bx', color=color, ms=10)
        f4.plot([8.5], o2.N_WR, 'bx', color=color, ms=10)

    # Now overplot the observed histograms
    titles = ['With Multiples, Complete',
              'No Multiples, Complete',
              'With Multiples, Incomplete',
              'No Multiples, Incomplete']

    outSuffixArr = ['with_multi_comp'+outSuffix,
                    'no_multi_comp'+outSuffix,
                    'with_multi_incomp'+outSuffix,
                    'no_multi_incomp'+outSuffix]
    gcutil.mkdir(outDir + 'plots/')
        
    f_arr = [f1, f2, f3, f4]
    for fidx in range(len(f_arr)):
        f = f_arr[fidx]
        f.hist(yngData.kp, bins=binEdges, histtype='step', color='black', label='Obs',
               weights=yngData.prob)
        f.plot([8.5], yngData.N_WR, 'ko', color='black', ms=10)
        f.legend(loc='upper left')
        #f.set_yscale('log')
        rng = py.axis()
        f.set_xlabel('Kp Magnitude')
        f.set_ylabel('Number of Stars')
        f.set_title(titles[fidx])
        f.set_ylim(0, rng[3])

        outFile = outDir + 'plots/klf_model_vs_data_' + outSuffixArr[fidx] + '.png'
        f.get_figure().savefig(outFile)

def plot_model_vs_data_MC(logAge, AKs, distance, imfSlope, clusterMass, yngData=None,
                          outSuffix='', outDir=workDir, numMC=50, makeEPS=False):
    """
    Compare a specific synthetic population to the observed KLF and Number of WR stars.
    Run the synthetic population numMC times and plot a shaded region that represents
    the possiblities.
    """
    from jlu.gc.imf import bayesian as b

    binsKp = klf_mag_bins
    binEdges = binsKp[0:-1] + (binsKp[1:] - binsKp[0:-1]) / 2.0

    if yngData == None:
        yngData = load_yng_data_by_radius(magCut=15.5, trimWR=True)

    # Setup 4 different figures
    #    Figure 1 -- with multiples, complete    
    #    Figure 2 -- no multiples, complete    
    #    Figure 3 -- with multiples, incomplete    
    #    Figure 4 -- no multiples, incomplete    
    py.close('all')
    fig1 = py.figure(1)
    fig2 = py.figure(2)
    fig3 = py.figure(3)
    fig4 = py.figure(4)
    fig1.clf()
    fig2.clf()
    fig3.clf()
    fig4.clf()
    f1 = fig1.gca()
    f2 = fig2.gca()
    f3 = fig3.gca()
    f4 = fig4.gca()

    color = 'red'
    for i in range(numMC):
        if i == 0:
            legLabel = 'Sim'
        else:
            legLabel = None
            
        c1 = b.model_young_cluster(logAge, AKs=AKs, distance=8000,
                                   imfSlope=imfSlope, clusterMass=clusterMass,
                                   makeMultiples=True)
        c2 = b.model_young_cluster(logAge, AKs=AKs, distance=8000,
                                   imfSlope=imfSlope, clusterMass=clusterMass,
                                   makeMultiples=False)

        c1.mag_noWR += 5.0 * np.log10(distance / 8000.0)
        c2.mag_noWR += 5.0 * np.log10(distance / 8000.0)


        o1 = b.sim_to_obs_klf(c1, magCut=15.5, withErrors=False, yng_orig=yngData)
        o2 = b.sim_to_obs_klf(c2, magCut=15.5, withErrors=False, yng_orig=yngData)
        
        f1.hist(c1.mag_noWR, bins=binEdges, histtype='step', color=color,
                label=legLabel, linewidth=2, alpha=0.2)
        f2.hist(c2.mag_noWR, bins=binEdges, histtype='step', color=color,
                label=legLabel, linewidth=2, alpha=0.2)

        f3.hist(o1.kp, bins=binEdges, histtype='step', color=color,
                label=legLabel, linewidth=2, alpha=0.2)
        f4.hist(o2.kp, bins=binEdges, histtype='step', color=color,
                label=legLabel, linewidth=2, alpha=0.2)

        f1.plot([8.5], c1.num_WR, 'bx', color=color, ms=10)
        f2.plot([8.5], c2.num_WR, 'bx', color=color, ms=10)
        f3.plot([8.5], o1.N_WR, 'bx', color=color, ms=10)
        f4.plot([8.5], o2.N_WR, 'bx', color=color, ms=10)

    # Now overplot the observed histograms
    titles = ['With Multiples, Complete',
              'No Multiples, Complete',
              'With Multiples, Incomplete',
              'No Multiples, Incomplete']

    outSuffixArr = ['with_multi_comp'+outSuffix,
                    'no_multi_comp'+outSuffix,
                    'with_multi_incomp'+outSuffix,
                    'no_multi_incomp'+outSuffix]
    gcutil.mkdir(outDir + 'plots/')
        
    f_arr = [f1, f2, f3, f4]
    for fidx in range(len(f_arr)):
        f = f_arr[fidx]

        try:
            idx = np.where(yngData.isWR == False)[0]
            plotData = yngData.kp[idx]
            plotWeights = yngData.prob[idx]
            # idx = np.where(yngData.isYoung == True)[0]
            # plotData = yngData.kp[idx]
            # plotWeights = None
        except AttributeError:
            plotData = yngData.kp
            plotWeights = yngData.prob
            
        (n, b, p) = f.hist(plotData, bins=binEdges, histtype='step', color='black', label='Obs',
                           linewidth=4, weights=plotWeights)
        f.plot([8.5], yngData.N_WR, 'ko', color='black', ms=10)
        f.legend(loc='upper left')

        maxLevel = n.max() * 2
        if maxLevel < 20:
            maxLevel = 20

        f.set_ylim((0, maxLevel))
        f.set_xlabel('Kp Magnitude')
        f.set_ylabel('Number of Stars')
        f.set_title(titles[fidx])

        outFile = outDir + 'plots/klf_model_vs_data_MC_' + outSuffixArr[fidx]
        f.get_figure().savefig(outFile + '.png')

        if makeEPS:
            tmp = 'convert %s.png %s.eps' % (outFile, outFile)
            subprocess.call(tmp, shell=True)

def plot_model_vs_data_tests():
    """
    Plot up the data vs. a set of model parameters to see how much variation there is.
    The data is actually a simulated cluster.
    """
    data_logt = 6.6
    data_AKs = 2.70
    data_dist = 8000
    data_Mcl = 1.1e4
    data_alpha = 2.35

    # sim_file = 'test_yng_sim3_t%.2f_AKs%.1f_d%d_a%.2f_m%d.pickle' % \
    #     (data_logt, data_AKs, data_dist, data_alpha, data_Mcl)
    # print sim_file
    # tmp = open(workDir + sim_file, 'r')
    
    # yng_sim = pickle.load(tmp)
    yng_sim = None

    # Vary the cluster age
    logAge = np.array([6.30, 6.60, 6.78, 6.90, 7.00])
    AKs = np.ones(len(logAge)) * data_AKs
    dist = np.ones(len(logAge)) * data_dist
    Mcl = np.ones(len(logAge)) * data_Mcl
    alpha = np.ones(len(logAge)) * data_alpha
    labels = ['log(t)=%.2f' % obj for obj in logAge]

    plot_model_vs_data(logAge, AKs, dist, alpha, Mcl, yngData=yng_sim,
                       outSuffix='_sim_vary_logAge', legendLabels=labels)

    # # Vary the distance
    # dist = np.array([7000, 7250, 7500, 7750, 8000, 8250, 8500, 8750, 9000])
    # logAge = np.ones(len(dist)) * data_logt
    # AKs = np.ones(len(dist)) * data_AKs
    # Mcl = np.ones(len(dist)) * data_Mcl
    # alpha = np.ones(len(dist)) * data_alpha

    # plot_model_vs_data(logAge, AKs, dist, alpha, Mcl, yngData=yng_sim,
    #                    outSuffix='_sim_vary_distance')

    # Vary the cluster mass
    Mcl = np.array([5e3, 1e4, 4e4, 8e4])
    dist = np.ones(len(Mcl)) * data_dist
    logAge = np.ones(len(Mcl)) * data_logt
    AKs = np.ones(len(Mcl)) * data_AKs
    alpha = np.ones(len(Mcl)) * data_alpha
    labels = ['Mcl=%.1e' % obj for obj in Mcl]

    plot_model_vs_data(logAge, AKs, dist, alpha, Mcl, yngData=yng_sim,
                       outSuffix='_sim_vary_cluster_mass', legendLabels=labels)

    # Vary the imfSlope
    alpha = np.array([0.35, 1.35, 2.35, 3.35])
    Mcl = np.ones(len(alpha)) * data_Mcl
    dist = np.ones(len(alpha)) * data_dist
    logAge = np.ones(len(alpha)) * data_logt
    AKs = np.ones(len(alpha)) * data_AKs
    labels = [r'$\alpha=%.2f$' % obj for obj in alpha]

    plot_model_vs_data(logAge, AKs, dist, alpha, Mcl, yngData=yng_sim,
                       outSuffix='_sim_vary_imf_slope', legendLabels=labels)


    
def plot_model_vary_imf():
    """
    Plot up several synthetic clusters with different IMFs. Do this for clusters
    with and without multiplicity.
    """
    from jlu.gc.imf import bayesian as b

    binsKp = klf_mag_bins
    binEdges = binsKp[0:-1] + (binsKp[1:] - binsKp[0:-1]) / 2.0

    logAge = 6.78
    AKs = 2.7
    dist = 8000
    clusterMass = 4e4

    imfSlopes = np.array([0.35, 1.35, 2.35, 3.35])

    colors=['red', 'yellow', 'green', 'blue']

    # Plot without completeness correction
    py.close(1)
    fig = py.figure(1, figsize=(12, 6))
    fig.subplots_adjust(left=0.08, right=0.95)
    p1 = fig.add_subplot(1, 2, 1)
    p2 = fig.add_subplot(1, 2, 2)

    for i in range(len(imfSlopes)):
        color = colors[i]
        model1 = b.model_young_cluster(logAge, AKs=AKs, distance=dist,
                                       imfSlope=imfSlopes[i], clusterMass=clusterMass,
                                       makeMultiples=True)
        model2 = b.model_young_cluster(logAge, AKs=AKs, distance=dist,
                                       imfSlope=imfSlopes[i], clusterMass=clusterMass,
                                       makeMultiples=False)

        legLabel = r'$\alpha=%.2f$' % imfSlopes[i]

        p1.hist(model1.mag_noWR, bins=binEdges, histtype='step', color=color, linewidth=2,
                label=legLabel)
        p1.plot([8.5], model1.num_WR, 'bx', color=color, ms=10)

        p2.hist(model2.mag_noWR, bins=binEdges, histtype='step', color=color, linewidth=2,
                label=legLabel)
        p2.plot([8.5], model2.num_WR, 'b+', color=color, ms=10)

        print(('IMF Slope = %.2f' % imfSlopes[i]))
        print(('   N_WR = %3d and %3d' % (model1.num_WR, model2.num_WR)))

    p1.set_xlabel('Kp Magnitude')
    p1.set_ylabel('Number of Stars')
    p1.set_title('With Multiples')

    p2.set_xlabel('Kp Magnitude')
    p2.set_title('No Multiples')
    p2.legend(loc='upper left')
    
    rng1 = p1.axis()
    rng2 = p2.axis()
    
    maxY = max([rng1[3], rng2[3]])
    p1.set_ylim(0, maxY)
    p2.set_ylim(0, maxY)
    py.savefig(workDir + 'plots/klf_vary_imf_complete.png')


    # Plot with completeness correction
    py.close(2)
    fig = py.figure(2, figsize=(12, 6))
    fig.subplots_adjust(left=0.08, right=0.95)
    p1 = fig.add_subplot(1, 2, 1)
    p2 = fig.add_subplot(1, 2, 2)

    for i in range(len(imfSlopes)):
        color = colors[i]
        model1 = b.simulated_klf(logAge, AKs, dist, imfSlopes[i], clusterMass,
                                 makeMultiples=True, magCut=15.5, withErrors=False)
        model2 = b.simulated_klf(logAge, AKs, dist, imfSlopes[i], clusterMass,
                                 makeMultiples=False, magCut=15.5, withErrors=False)

        legLabel = r'$\alpha=%.2f$' % imfSlopes[i]

        p1.hist(model1.kp, bins=binEdges, histtype='step', color=color, linewidth=2,
                label=legLabel)
        p1.plot([8.5], model1.N_WR, 'bx', color=color, ms=10)

        p2.hist(model2.kp, bins=binEdges, histtype='step', color=color, linewidth=2,
                label=legLabel)
        p2.plot([8.5], model2.N_WR, 'b+', color=color, ms=10)

        print(('IMF Slope = %.2f' % imfSlopes[i]))
        print(('   N_WR = %3d and %3d' % (model1.N_WR, model2.N_WR)))

    p1.set_xlabel('Kp Magnitude')
    p1.set_ylabel('Number of Stars')
    p1.set_title('With Multiples')

    p2.set_xlabel('Kp Magnitude')
    p2.set_title('No Multiples')
    p2.legend(loc='upper left')
    
    rng1 = p1.axis()
    rng2 = p2.axis()
    
    maxY = max([rng1[3], rng2[3]])
    p1.set_ylim(0, maxY)
    p2.set_ylim(0, maxY)
    py.savefig(workDir + 'plots/klf_vary_imf_incomplete.png')

    py.close(1)
    py.close(2)
    

def plot_sim_clusters():
    """
    Make a series of plots of simulated clusters just for illustration and testing purposes.
    """
    from jlu.gc.imf import bayesian as b

    def q_pdf(q, beta, qMin):
        tmp = (1 + beta) / (1 - qMin**(1+beta))
        return tmp * q**beta

    def m_pdf(m, alpha, mMin, mMax):
        tmp1 = 1 - alpha
        tmp2 = tmp1 / (mMax**tmp1 - mMin**tmp1)
        return tmp2 * m**(-alpha)
    
    logAge = 6.6
    AKs = 2.7
    dist = 8000
    Mcl = 1e6
    imfSlope = 2.35
    minMass = 1.0
    maxMass = 150.0
    qMin = 0.01
    qSlope = -0.4

    # First lest compare with and without multiples
    c1 = b.model_young_cluster(logAge, AKs=AKs, distance=dist, clusterMass=Mcl,
                               imfSlope=imfSlope, minMass=minMass, maxMass=maxMass,
                               makeMultiples=False, qMin=qMin, qIndex=qSlope)
    c2 = b.model_young_cluster(logAge, AKs=AKs, distance=dist, clusterMass=Mcl,
                               imfSlope=imfSlope, minMass=minMass, maxMass=maxMass,
                               makeMultiples=True, qMin=qMin, qIndex=qSlope)

    # Plot up the mass functions
    bins = np.arange(minMass, maxMass, 1)
    bin_center = bins[:-1] + (np.diff(bins)/2.0)
    py.clf()
    py.hist(c2.mass, bins=bins, histtype='step', label='With Multiples',
            log=True, color='red', linewidth=1.5)
    py.hist(c1.mass, bins=bins, histtype='step', label='No Multiples',
            log=True, color='green', linewidth=1.5)
    legLabel = r'$N(m) \propto m^{-%.2f}$' % imfSlope
    tmp = m_pdf(bin_center, imfSlope, minMass, maxMass)
    tmp *= len(c1.mass)
    py.plot(bin_center, tmp, label=legLabel)
    py.xlabel('Primary Mass (Msun)')
    py.ylabel('Number of Systems')
    py.xlim(0, c1.mass.max())
    py.legend()
    py.title('No Multi: M = %.1e N = %d' % (c1.mass.sum(), len(c1.mass)))
    py.savefig(workDir + 'plots/sim_clusters_imf_multiples.png')

    # Plot a mass-magnitude diagram
    py.clf()
    py.plot(c2.mass[c2.idx_noWR], c2.mag_noWR, 'r+', ms=3, mec='red',
            label='With Multiples')
    py.plot(c1.mass[c1.idx_noWR], c1.mag_noWR, 'gx', ms=3, mec='green',
            label='No Multiples')
    ax = py.axis()
    py.ylim(ax[3], ax[2])
    py.xlabel('Mass (Msun)')
    py.ylabel('Kp Magnitude')
    py.legend(loc='lower right', numpoints=1, markerscale=3)
    outFile = workDir + 'plots/sim_clusters_m_vs_kp_multiples'
    py.savefig(outFile + '.png')
    tmp = 'convert %s.png %s.eps' % (outFile, outFile)
    subprocess.call(tmp, shell=True)

    # Check out the companion information
    idx = np.where((c2.isMultiple == True) & (c2.mass > 10))[0]
    q = np.array([], dtype=float)
    for ii in idx:
        q_new = c2.compMasses[ii] / c2.mass[ii]
        q = np.append(q, q_new)

    # Plot the mass-ratio distribution for companions
    py.clf()
    (n, bins, patches) = py.hist(q, histtype='step', normed=True)
    bin_center = bins[:-1] + (np.diff(bins)/2.0)
    legLabel = r'$N(q) \propto q^{%.2f}$' % qSlope
    py.plot(bin_center, q_pdf(bin_center, qSlope, qMin), label=legLabel)
    py.xlabel('Mass Ratio q')
    py.ylabel('PDF(q)')
    py.legend()
    py.savefig(workDir + 'plots/sim_clusters_q_hist.png')

    
    # Plot KLFs
    py.clf()
    bins = np.arange(9, 21, 0.25)
    tmp2 = py.hist(c2.mag_noWR, bins=bins, histtype='step', label='With Multiples',
                   color='red', linewidth=1.5)
    tmp1 = py.hist(c1.mag_noWR, bins=bins, histtype='step', label='No Multiples',
                   color='green', linewidth=1.5)
    py.gca().set_yscale('log')
    py.legend(loc='upper left')
    py.xlabel('Kp Magnitude')
    py.ylabel('Number of Stars')
    py.savefig(workDir + 'plots/sim_clusters_klf_multiples.png')
    py.savefig(workDir + 'plots/sim_clusters_klf_multiples.eps')

    return

def plot_yng_osiris_fov():
    """
    Plot up a deep field mosaic along with the identified young stars
    and the OSIRIS FOV. Still deciding whether to also plot known young
    stars outside our FOV.
    """
    # Load up our young stars
    yng_ours = load_yng_catalog_by_radius()

    # Load up the spectroscopic database to get the field-of-view definitions.
    dbfile = '/u/jlu/data/gc/database/stars.sqlite'

    connection = sqlite.connect(dbfile)
    cur = connection.cursor()
    cur.execute('select name, x, y, kp from stars where young="T"')
    yngRows = cur.fetchall()

    cur.execute('select name, x_vertices, y_vertices from fields')
    regions = cur.fetchall()
    
    fieldNames = []
    xverts = []
    yverts = []
    for rr in range(len(regions)):
        fieldNames.append(regions[rr][0])
        xverts.append(np.array([float(ff) for ff in regions[rr][1].split(',')]))
        yverts.append(np.array([float(ff) for ff in regions[rr][2].split(',')]))

    # Load up the 06maylgs1 mosaic image
    imageRoot = '/u/jlu/work/gc/dp_msc/mosaic_image/2008/'
    imageRoot += 'mag08maylgs2_dp_msc_kp'
    img = pyfits.getdata(imageRoot + '.fits')

    #img = img_scale.log(img, scale_min=10, scale_max=1e7)
    #img = img_scale.sqrt(img, scale_min=500, scale_max=5e4)
    img = img_scale.linear(img, scale_min=500, scale_max=4e4)
    
    # Coo star is IRS 16C
    coords = open(imageRoot + '.coo').readline().split()
    cooPix = [float(coords[0])-1.0, float(coords[1])-1.0]

    # Load up the label.dat file to get the coordinates in arcsec
    label = starTables.Labels()
    idx = np.where(label.name == 'irs16C')[0]
    cooAsec = [label.x[idx[0]], label.y[idx[0]]]

    scale = 0.00995

    xmin = ((0 - cooPix[0]) * scale * -1.0) + cooAsec[0]
    xmax = ((img.shape[1] - cooPix[0]) * scale * -1.0) + cooAsec[0]
    ymin = ((0 - cooPix[1]) * scale) + cooAsec[1]
    ymax = ((img.shape[0] - cooPix[1]) * scale) + cooAsec[1]
    extent = [xmin, xmax, ymin, ymax]

    # Loop through the stars and make arrays
    numYng = len(yngRows)

    yngName = np.zeros(numYng, dtype='S13')
    yngX = np.zeros(numYng, dtype=float)
    yngY = np.zeros(numYng, dtype=float)
    yngKp = np.zeros(numYng, dtype=float)

    for yy in range(len(yngRows)):
        yngName[yy] = yngRows[yy][0]
        yngX[yy] = yngRows[yy][1]
        yngY[yy] = yngRows[yy][2]
        yngKp[yy] = yngRows[yy][3]

    # Trim down to those young stars NOT in our data set
    idx = []
    for yy in range(len(yngName)):
        if (yngName[yy] not in yng_ours.name) and (yngKp[yy] <= 15.5):
            idx.append(yy)
            
    yngName = yngName[idx]
    yngX = yngX[idx]
    yngY = yngY[idx]
    yngKp = yngKp[idx]

    # Trim down our sample to those brighter than 15.5
    idx = np.where(yng_ours.kp_ext <= 15.5)[0]
    yng_ours.x = yng_ours.x[idx]
    yng_ours.y = yng_ours.y[idx]
    yng_ours.name = yng_ours.name[idx]
    yng_ours.kp = yng_ours.kp[idx]
    yng_ours.kp_ext = yng_ours.kp_ext[idx]
    yng_ours.kp_err = yng_ours.kp_err[idx]
    yng_ours.isWR = yng_ours.isWR[idx]
    yng_ours.prob = yng_ours.prob[idx]

    # Plot image
    py.close(1)
    py.figure(1, figsize=(12, 10))
    py.subplots_adjust(left=0.1, right=0.95, bottom=0.1, top=0.95)
    py.clf()
    py.imshow(img, extent=extent, cmap=py.cm.gray_r)

    # Plot all known young stars
    py.plot(yngX, yngY, 'bs',
            markerfacecolor='none', markersize=8, 
            markeredgecolor='blue', markeredgewidth=2)

    # Plot all WR stars
    wolfRayetNames = get_wolf_rayet_stars()
    wdx = []
    for wr in wolfRayetNames:
        idx = np.where(yngName == wr)[0]
        if len(idx) == 0:
            continue
        wdx.append(idx[0])

    py.plot(yngX[wdx], yngY[wdx], 'cs',
            markerfacecolor='none', markersize=8,
            markeredgecolor='magenta', markeredgewidth=2)

    # Plot all WR stars in our FOV
    idx = np.where(yng_ours.isWR == True)[0]
    py.plot(yng_ours.x[idx], yng_ours.y[idx], 'co', label='Wolf-Rayet',
            markerfacecolor='none', markersize=8,
            markeredgecolor='magenta', markeredgewidth=2)

    # Plot all the non-WR stars (but P(yng) = 1) in our sample
    idx = np.where((yng_ours.isWR == False) & (yng_ours.prob == 1))[0]
    py.plot(yng_ours.x[idx], yng_ours.y[idx], 'bo', label='Young I, II, V',
            markerfacecolor='none', markersize=8,
            markeredgecolor='blue', markeredgewidth=2)

    # Plot all the non-WR stars (but P(yng) < 1) in our sample
    idx = np.where((yng_ours.isWR == False) & (yng_ours.prob != 1))[0]
    py.plot(yng_ours.x[idx], yng_ours.y[idx], 'go', label='0 < P(yng) < 1',
            markerfacecolor='none', markersize=8,
            markeredgecolor='limegreen', markeredgewidth=2)

    py.axis('tight')
    py.axis('equal')
    py.axis([13, -10, -8.5, 12])
    py.legend(loc='upper left', numpoints=1)
    py.xlabel('R.A. Offset from Sgr A* (arcsec)')
    py.ylabel('Dec. Offset from Sgr A* (arcsec)')

    # Overplot OSIRIS fields
    for ff in range(len(fieldNames)):
        if 'Imaging' in fieldNames[ff]:
            continue

        py.plot(np.append(xverts[ff], xverts[ff][0]), 
                np.append(yverts[ff], yverts[ff][0]), 
                'k--', color='black')

    
    py.savefig('/u/jlu/doc/papers/gcimf/young_stars_osiris_fov.png')
    py.savefig('/u/jlu/doc/papers/gcimf/young_stars_osiris_fov.eps')


def plot_klf_progression(rmin=0, rmax=30):
    d = load_klf_by_radius(rmin, rmax, mask_for_log=True)

    magBin = d.Kp[1] - d.Kp[0]

    idx = np.where(d.Kp < 16)[0]

    legFont = matplotlib.font_manager.FontProperties(size=12)

    # By Number Plot
    py.clf()

    py.plot(d.Kp[idx], d.N[idx], 'bo-', label='Observed')

    py.plot(d.Kp[idx], d.N_ext[idx], 'go-', label='Ext Corrected')

    py.plot(d.Kp[idx], d.N_ext_cmp_sp[idx], 'ro-', label='Ext + SpCmp')

    py.errorbar(d.Kp[idx], d.N_ext_cmp_sp_im[idx], fmt='k-', 
                xerr=magBin/2.0, capsize=0)
    py.errorbar(d.Kp[idx], d.N_ext_cmp_sp_im[idx], fmt='ko-', 
                yerr=d.eN_ext_cmp_sp_im[idx], 
                label='Ext + SpCmp + ImCmp')

    for ii in idx:
        print(('Kp = %5.2f  N = %5.2f  density = %5.3f' % \
            (d.Kp[ii], d.N_ext_cmp_sp_im[ii], d.KLF_ext_cmp_sp_im[ii])))

    #py.gca().set_yscale('log')
    py.legend(loc='upper left', numpoints=1, prop=legFont)
    py.xlim(8.5, 15.5)
    py.ylim(0, 40)
    py.xlabel('Kp magnitude')
    py.ylabel('Stars')
    py.title("%.1f'' < r <= %.1f''" % (rmin, rmax), fontsize=14)
    py.savefig('%splots/klf_progression_counts_r_%.1f_%.1f.png' % 
               (workDir, rmin, rmax))
    py.savefig('%splots/klf_progression_counts_r_%.1f_%.1f.eps' % 
               (workDir, rmin, rmax))


    # By Density Plot
    py.clf()
    py.plot(d.Kp[idx], d.KLF[idx], 'bo-', label='Observed')

    py.plot(d.Kp[idx], d.KLF_ext[idx], 'go-', label='Ext Corrected')

    py.plot(d.Kp[idx], d.KLF_ext_cmp_sp[idx], 'ro-', label='Ext + SpCmp')

    py.errorbar(d.Kp[idx], d.KLF_ext_cmp_sp_im[idx], fmt='k-', 
                xerr=magBin/2.0, capsize=0)
    py.errorbar(d.Kp[idx], d.KLF_ext_cmp_sp_im[idx], fmt='ko-', 
                yerr=d.eKLF_ext_cmp_sp_im[idx], 
                label='Ext + SpCmp + ImCmp')

    #py.gca().set_yscale('log')
    py.legend(loc='upper left', numpoints=1, prop=legFont)
    py.xlim(8.5, 15.5)
    py.ylim(0.0, 0.35)
    py.xlabel('Kp magnitude')
    py.ylabel(r'Stars / [asec$^2$ mag]')
    py.title("%.1f'' < r <= %.1f''" % (rmin, rmax), fontsize=14)
    py.savefig('%splots/klf_progression_r_%.1f_%.1f.png' % 
               (workDir, rmin, rmax))
    py.savefig('%splots/klf_progression_r_%.1f_%.1f.eps' % 
               (workDir, rmin, rmax))
    
    
def plot_sim_results(rootdir, index=0, sim=True, rmin=0, rmax=30):
    """
    Load up the results of a MultiNest run for on of the simulated clusters. Then
    overplot the best fitting model (or the one specified by <index> which is the index
    into the logLikelihood sorted array).

    index -- 0 for best fit, 1 for next best fit, etc.
    """
    # Convert index into a real array index
    idx = (index + 1) * -1

    # Load up the simulation results
    from jlu.gc.imf import multinest
    fit = multinest.load_results(rootdir)

    if sim:
        # Load up the original data that was fit.
        foo = rootdir.split('_')
        tmp = foo.index('sim')
        data_root = 'cluster_' + '_'.join(foo[tmp:]).replace('/', '') + '.pickle'

        tmp = open(data_root, 'r')
        data = pickle.load(tmp)
        tmp.close()

        idx_nonWR = np.where(data.kp < 15.5)[0]
    else:
        data = load_yng_catalog_by_radius(rmin=rmin, rmax=rmax, magCut=15.5, trimWR=True)
        data.kp = data.kp_ext

    out_suffix = '_best_fit_%d' % index

    print(('Plotting Solution %d' % index))
    print(('   log(Likelihood): %.2f' % fit['logLike'][idx]))
    print(('   log(age):        %.2f' % fit['logAge'][idx]))
    print(('   distance (pc):   %d' % (fit['distance'][idx]*10**3)))
    print(('   alpha:           %.2f' % fit['alpha'][idx]))
    print(('   Mcl (Msun)       %d' % (fit['Mcl'][idx]*10**3)))
    # print '   N_old:           %d' % (fit['N_old'][idx])
    # print '   gamma:           %.2f' % (fit['gamma'][idx])
    # print '   rcMean:          %.2f' % (fit['rcMean'][idx])
    # print '   rcSigma:         %.2f' % (fit['rcSigma'][idx])
    print(('   N(WR):           %d in data vs. %d in sim' % (data.N_WR, fit['N_WR_sim'][idx])))
    print(('   N(yng):          %d in data vs. %d in sim' % \
        (data.prob.sum(), fit['N_yng_obs'][idx])))

    plot_model_vs_data_MC(fit['logAge'][idx], 2.7, int(round(fit['distance'][idx]*10**3, 0)),
                          fit['alpha'][idx], int(round(fit['Mcl'][idx]*10**3, 0)), yngData=data,
                          outDir=rootdir+'/', outSuffix=out_suffix)

    # #####
    # # Plot up the old population
    # #####
    # binsKp = klf_mag_bins
    # binEdges = binsKp[0:-1] + (binsKp[1:] - binsKp[0:-1]) / 2.0
    # py.close(5)
    # fig = py.figure(5)
    # f = fig.gca()

    # # Load up imaging completness curve
    # completeness = load_image_completeness_by_radius()
    # Kp_interp = interpolate.splrep(completeness.mag, completeness.comp, k=1, s=0)

    # # This is the magnitude range over which the mixture model weights
    # # are determined.
    # k_min = 8.0  # K' magnitude limit for old KLF powerlaw
    # k_max = 18.0 # K' magnitude limit for old KLF powerlaw

    # # Generate old stars
    # pl_loc = math.e**k_min
    # pl_scale = math.e**k_max - math.e**k_min
    # pl_index = fit['gamma'][idx] * math.log(10.0)
    # powerlaw = scipy.stats.powerlaw(pl_index, loc=pl_loc, scale=pl_scale)
    # gaussian = scipy.stats.norm(loc=fit['rcMean'][idx], scale=fit['rcSigma'][idx])

    # for ii in range(50):
    #     # Fix the relative fraction of stars in the Red Clump
    #     fracInRC = 0.12
    #     kp_PLAW = np.log( powerlaw.rvs((1.0 - fracInRC)*(fit['N_old'][idx])) )
    #     kp_NORM = gaussian.rvs(fracInRC*fit['N_old'][idx])
    #     old_sim_kp = np.concatenate([kp_PLAW, kp_NORM])

    #     comp_for_stars = interpolate.splev(old_sim_kp, Kp_interp)
    #     comp_for_stars[comp_for_stars > 1] = 1.0
    #     comp_for_stars[comp_for_stars < 0] = 0.0

    #     if ii == 0:
    #         legLabel = 'Sim'
    #     else:
    #         legLabel = None
    #     f.hist(old_sim_kp, bins=binEdges, histtype='step', color='red', label=legLabel,
    #            linewidth=2, alpha=0.2, weights=comp_for_stars)


    # try:
    #     idx = np.where(data.isYoung == False)[0]
    #     plotData = data.kp[idx]
    #     plotWeights = None
    # except AttributeError:
    #     plotData = data.kp
    #     plotWeights = 1.0 - data.prob

    # plotData = data.kp
    # plotWeights = 1.0 - data.prob

    # (n, b, p) = f.hist(plotData, bins=binEdges, histtype='step', color='black', label='Obs',
    #                    linewidth=4, weights=plotWeights)

    # f.legend(loc='upper left')
    # f.set_ylim((0, n.max()*1.5))
    # f.set_xlabel('Kp Magnitude')
    # f.set_ylabel('Number of Old Stars')

    # outFile = rootdir + 'plots/klf_model_vs_data_MC_old_' + out_suffix + '.png'
    # f.get_figure().savefig(outFile)

def fit_with_models(multiples=True, rmin=0, rmax=30):
    """
    Run MultiNest on the observed data to find the best fit cluster properties
    """
    if multiples:
        out_dir = '%s/multinest/obs_multi_r_%.1f_%.1f/' % (workDir, rmin, rmax)
    else:
        out_dir = '%s/multinest/obs_single_r_%.1f_%.1f/' % (workDir, rmin, rmax)

    from jlu.gc.imf import multinest as m

    m.run(out_dir, n_live_points=300, multiples=multiples)
    m.plot_posteriors(out_dir)
    m.plot_posteriors_1D(out_dir, sim=False)
    plot_sim_results(out_dir, sim=False)

def plot_WR_vs_age():
    """
    Plot up the number of WR stars vs. age for a very massive star cluster.
    """
    from jlu.gc.imf import bayesian as b
    
    logAge = np.arange(6.2, 7.0, 0.01)
    #alpha = np.array([0.5, 1.35, 1.85, 2.35])
    alpha = np.array([0.5, 1.1, 1.7, 2.35])
    AKs = 2.7
    Mcl = 1e4
    distance = 8000
    minMass = 1.0
    maxMass = 150.0

    num_WR = np.zeros((len(logAge), len(alpha)), dtype=int)
    num_OB = np.zeros((len(logAge), len(alpha)), dtype=int)

    completeness = load_image_completeness_by_radius()
    magCut = 15.5

    for aa in range(len(alpha)):
        for tt in range(len(logAge)):
            tmp_logAge = round(logAge[tt], 2)
            tmp_alpha = round(alpha[aa], 2)

            print(('log(age) = %.2f  alpha = %.2f' % (tmp_logAge, tmp_alpha)))

            tmp = b.fetch_model_from_sims(tmp_logAge, AKs, distance,
                                          tmp_alpha, Mcl, minMass, maxMass)
            sim_N_WR = tmp[0]
            sim_k_bins = tmp[1]
            sim_k_pdf = tmp[2]
            sim_k_pdf_norm = tmp[3]

            sim_k_bin_widths = np.diff(sim_k_bins)
            sim_k_bin_center = sim_k_bins[:-1] + (sim_k_bin_widths/2.0)

            # Multiply by the completeness curve (after re-sampling). And renormalize.
            Kp_interp = interpolate.splrep(completeness.mag, completeness.comp, k=1, s=0)
            comp_resamp = interpolate.splev(sim_k_bin_center, Kp_interp)
            comp_resamp[comp_resamp < 0] = 0.0
            comp_resamp[comp_resamp > 1] = 1.0

            sim_k_pdf *= comp_resamp
            sim_k_pdf_norm *= comp_resamp
            sim_k_pdf_norm /= (sim_k_pdf_norm * sim_k_bin_widths).sum()

            # Trim down to magCut
            idx = np.where(sim_k_bins <= magCut)[0]

            sim_k_bins = np.append(sim_k_bins[idx], sim_k_bins[idx[-1]+1])
            sim_k_bin_center = sim_k_bin_center[idx]
            sim_k_bin_widths = sim_k_bin_widths[idx]
            sim_k_pdf = sim_k_pdf[idx]
            sim_k_pdf_norm = sim_k_pdf_norm[idx]

            sim_k_pdf_norm /= (sim_k_pdf_norm * sim_k_bin_widths).sum()

            sim_N_OB = sim_k_pdf.sum()

            num_WR[tt, aa] = sim_N_WR
            num_OB[tt, aa] = sim_N_OB
            
    
    legLabels = ['%.2f' % aa for aa in alpha]

    yngData = load_yng_data_by_radius(magCut=15.5, trimWR=True)
    obs_N_WR = yngData.N_WR
    obs_N_OB = yngData.prob.sum()

    # Logarithmically in time
    py.clf()
    py.plot(logAge, num_WR)
    py.legend(legLabels, title='IMF Slopes')
    py.xlabel('log(age [yr])')
    py.ylabel('Number of Wolf-Rayet Stars')
    py.title('Cluster Mass = %d Msun' % Mcl)
    py.axhline(obs_N_WR, linestyle='--')
    py.savefig(workDir + 'plots/num_WR_vs_logage.png')

    py.clf()
    py.plot(logAge, num_OB)
    py.legend(legLabels, title='IMF Slopes')
    py.xlabel('log(age [yr])')
    py.ylabel('Number of OB Stars (Kp <= 15.5)')
    py.title('Cluster Mass = %d Msun' % Mcl)
    py.axhline(obs_N_OB, linestyle='--')
    py.savefig(workDir + 'plots/num_OB_vs_logage.png')

    py.clf()
    py.plot(logAge, np.array(num_WR, dtype=float)/num_OB)
    py.legend(legLabels, title='IMF Slopes')
    py.xlabel('log(age [yr])')
    py.ylabel('WR / OB')
    py.title('Cluster Mass = %d Msun' % Mcl)
    py.axhline(obs_N_WR/obs_N_OB, linestyle='--')
    py.savefig(workDir + 'plots/num_WR_OB_ratio_vs_logage.png')


    # Linearly in time
    py.clf()
    py.plot(10**(logAge-6.0), num_WR)
    py.legend(legLabels, title='IMF Slopes')
    py.xlabel('Age [Myr]')
    py.ylabel('Number of Wolf-Rayet Stars')
    py.title('Cluster Mass = %d Msun' % Mcl)
    py.axhline(obs_N_WR, linestyle='--')
    py.savefig(workDir + 'plots/num_WR_vs_age.png')
    py.savefig(workDir + 'plots/num_WR_vs_age.eps')

    py.clf()
    py.plot(10**(logAge-6.0), num_OB)
    py.legend(legLabels, title='IMF Slopes')
    py.xlabel('Age [Myr]')
    py.ylabel('Number of OB Stars (Kp <= 15.5)')
    py.title('Cluster Mass = %d Msun' % Mcl)
    py.axhline(obs_N_OB, linestyle='--')
    py.savefig(workDir + 'plots/num_OB_vs_age.png')
    py.savefig(workDir + 'plots/num_OB_vs_age.eps')

    py.clf()
    py.plot(10**(logAge-6.0), np.array(num_WR, dtype=float)/num_OB)
    py.legend(legLabels, title='IMF Slopes')
    py.xlabel('age [Myr]')
    py.ylabel('WR / OB')
    py.title('Cluster Mass = %d Msun' % Mcl)
    py.axhline(obs_N_WR/obs_N_OB, linestyle='--')
    py.savefig(workDir + 'plots/num_WR_OB_ratio_vs_age.png')
    py.savefig(workDir + 'plots/num_WR_OB_ratio_vs_age.eps')

def calc_WR_OB_ratio_vs_age():
    """
    Plot up the ratio of WR to OB stars vs. age for a bunch of clusters.
    """
    from jlu.gc.imf import bayesian as b
    
    age = np.arange(1.5e6, 10e6, 0.1e6)
    logAge = np.log10(age)
    alpha = np.array([0.45, 1.05, 1.7, 2.35])
    AKs = 2.7
    Mcl = 1e4
    Mcl_sim = 1e7
    distance = 8000
    multi = True

    num_WR = np.zeros((len(logAge), len(alpha)), dtype=float)
    num_OB = np.zeros((len(logAge), len(alpha)), dtype=float)
    std_WR = np.zeros((len(logAge), len(alpha)), dtype=float)
    std_OB = np.zeros((len(logAge), len(alpha)), dtype=float)

    completeness = load_image_completeness_by_radius()
    magCut = 15.5

    for aa in range(len(alpha)):
        for tt in range(len(logAge)):
            tmp_logAge = round(logAge[tt], 2)
            tmp_alpha = round(alpha[aa], 2)

            print(('log(age) = %.2f  alpha = %.2f' % (tmp_logAge, tmp_alpha)))

            cluster = b.model_young_cluster(tmp_logAge, imfSlope=tmp_alpha, 
                                            AKs=AKs, distance=distance, 
                                            clusterMass=Mcl_sim,
                                            makeMultiples=multi)
            #####
            # Number of WR stars
            #####
            N_WR = cluster.num_WR

            # Rescale down to 1e4 mass
            N_WR *= float(Mcl) / float(Mcl_sim)

            #####
            # Number of OB stars
            #####
            kp = cluster.mag_noWR

            # Trim down to magCut
            idx = np.where(kp <= magCut)[0]
            kp = kp[idx]

            # Multiply by the completeness curve (after re-sampling). And renormalize.
            Kp_interp = interpolate.splrep(completeness.mag, completeness.comp, k=1, s=0)
            comp_at_kp = interpolate.splev(kp, Kp_interp)
            comp_at_kp[comp_at_kp < 0] = 0.0
            comp_at_kp[comp_at_kp > 1] = 1.0

            # Generate a random number for each star to decide if it was detected.
            # If the random number is less than the completeness, call it detected.
            rand = np.random.uniform(size=len(kp))
            isDetected = np.where(rand < comp_at_kp)[0]

            N_OB = len(isDetected)

            # Rescale down to 1e4 mass
            N_OB *= float(Mcl) / float(Mcl_sim)

            num_WR[tt, aa] = N_WR
            num_OB[tt, aa] = N_OB
            std_WR[tt, aa] = np.sqrt(N_WR)
            std_OB[tt, aa] = np.sqrt(N_OB)
            
    
    ratio = num_WR / num_OB
    ratio_err = ratio * np.hypot( std_WR / num_WR, std_OB / num_OB )

    _out = open('%s/age_vs_WR_OB.dat' % workDir, 'w')

    pickle.dump(age, _out)
    pickle.dump(alpha, _out)
    pickle.dump(num_WR, _out)
    pickle.dump(std_WR, _out)
    pickle.dump(num_OB, _out)
    pickle.dump(std_OB, _out)
    pickle.dump(ratio, _out)
    pickle.dump(ratio_err, _out)

    _out.close()


def plot_WR_OB_ratio_vs_age():
    _in = open('%s/age_vs_WR_OB.dat' % workDir, 'r')

    age = pickle.load(_in)
    age /= 1e6
    alpha = pickle.load(_in)
    num_WR = pickle.load(_in)
    std_WR = pickle.load(_in)
    num_OB = pickle.load(_in)
    std_OB = pickle.load(_in)
    ratio = pickle.load(_in)
    ratio_err = pickle.load(_in)

    _in.close()

    legLabels = [r'$\alpha = $%.2f' % aa for aa in alpha]

    yngData = load_yng_data_by_radius(magCut=15.5, trimWR=True)
    obs_N_WR = yngData.N_WR
    obs_N_OB = yngData.prob.sum()

    colors = ['red', 'green', 'blue', 'cyan']

    # Plot all range of IMF slopes
    py.figure(1)
    py.clf()
    for aa in range(len(alpha)):
        val = ratio[:,aa]
        valHi = ratio[:,aa] + ratio_err[:,aa]
        valLo = ratio[:,aa] - ratio_err[:,aa]

        py.plot(age, val, color=colors[aa])
        py.fill_between(age, valLo, valHi, color=colors[aa], alpha=0.2)
    py.legend(legLabels)
    py.xlabel('Age [Myr]')
    py.ylabel(r'N$_{WR}$ / N$_{OB}$', fontsize=22)
    py.axhline(obs_N_WR/obs_N_OB, linestyle='--', linewidth=2, color='black')

    py.xlim(1.5, 10)
    py.ylim(0, 1.0)

    outRoot = workDir + 'plots/num_WR_OB_ratio_vs_age'
    py.savefig(outRoot + '.png')
    tmp = 'convert %s.png %s.eps' % (outRoot, outRoot)
    print(tmp)
    subprocess.call(tmp, shell=True)

    # Plot a single IMF slope
    py.figure(2)
    py.clf()
    aa = 2
    val = ratio[:,aa]
    valHi = ratio[:,aa] + ratio_err[:,aa]
    valLo = ratio[:,aa] - ratio_err[:,aa]

    obsRatio = obs_N_WR / obs_N_OB

    py.plot(age, val, color=colors[aa])
    py.fill_between(age, valLo, valHi, color=colors[aa], alpha=0.2)
    py.xlabel('Age [Myr]')
    py.ylabel(r'N$_{WR}$ / N$_{OB}$', fontsize=22)
    py.text(4, 0.182, r'$\alpha = 1.7$, M$_{cl,obs} = 10^4$', fontsize=20, color='blue')

    py.axhline(obsRatio, linestyle='--', color='black', linewidth=2)
    py.text(8, obsRatio+0.003, 'Observed')
    idx = np.where(np.abs(ratio[:,aa] - obsRatio) < 0.01)[0]
    print((idx, age[idx], val[idx]))
    #py.plot(age[idx[:2]], val[idx[:2]], 'ks')
    py.plot([2.9, 3.8], [obsRatio, obsRatio], 'ks', ms=10)

    py.xlim(1.5, 10)
    py.ylim(0, 0.2)

    outRoot = workDir + 'plots/num_WR_OB_ratio_vs_age_1.7'
    py.savefig(outRoot + '.png')
    tmp = 'convert %s.png %s.eps' % (outRoot, outRoot)
    print(tmp)
    subprocess.call(tmp, shell=True)

def plot_fit_posteriors_2d():
    from jlu.gc.imf import multinest as m

    fit_dir = workDir + 'multinest/obs_multi_r_0.0_30.0/'
    fit = m.load_results(fit_dir)

    # Make a column that contains Myr instead of log(yr)
    fit.add_column('age', 10**(fit['logAge']-6.0), unit='Myr')

    fontsize = 16

    py.close(1)
    py.figure(1, figsize = (8,10))

    gs = gridspec.GridSpec(3, 2)
    gs.update(left=0.1, right=0.95, bottom=0.07, top=0.95,
              wspace=0.35, hspace=0.4)
    ax1 = py.subplot(gs[0, 0])
    ax2 = py.subplot(gs[0, 1])
    ax3 = py.subplot(gs[1, 0])
    ax4 = py.subplot(gs[1, 1])
    ax5 = py.subplot(gs[2, 0])
    ax6 = py.subplot(gs[2, 1])

    bins = {'alpha': 50, 'age': 50,
            'distance': np.arange(6.5, 9.5, 0.1),
            'Mcl': np.arange(1, 30, 1)}

    defSmooth = 1.2
    plot_fit_posterior_2d(ax1, 'alpha', 'age', fit=fit,
                          bins1=bins['alpha'], bins2=bins['age'])
    plot_fit_posterior_2d(ax2, 'distance', 'age', fit=fit,
                          bins1=bins['distance'], bins2=bins['age'])
    plot_fit_posterior_2d(ax3, 'alpha', 'Mcl', fit=fit,
                          bins1=bins['alpha'], bins2=bins['Mcl'])
    plot_fit_posterior_2d(ax4, 'age', 'Mcl', fit=fit,
                          bins1=bins['age'], bins2=bins['Mcl'])
    plot_fit_posterior_2d(ax5, 'alpha', 'distance', fit=fit,
                          bins1=bins['alpha'], bins2=bins['distance'])
    plot_fit_posterior_2d(ax6, 'Mcl', 'distance', fit=fit,
                          bins1=bins['Mcl'], bins2=bins['distance'])

    ## Set some custom axes limits
    #  age
    ax1.set_ylim(1, 8)
    ax2.set_ylim(1, 8)
    ax4.set_xlim(1, 8)

    # # distance
    # ax2.set_xlim(7, 9)
    # ax5.set_ylim(7, 9)
    # ax6.set_ylim(7, 9)

    # Align axis labels
    ax1.yaxis.set_label_coords(-0.16, 0.5)
    ax2.yaxis.set_label_coords(-0.16, 0.5)
    ax3.yaxis.set_label_coords(-0.12, 0.5)
    ax4.yaxis.set_label_coords(-0.12, 0.5)
    ax5.yaxis.set_label_coords(-0.16, 0.5)
    ax6.yaxis.set_label_coords(-0.16, 0.5)
    
    py.savefig(workDir + 'plots/plot_fit_posteriors_2D.png')
    py.savefig(workDir + 'plots/plot_fit_posteriors_2D.eps')

def plot_fit_posterior_2d(axes, param1, param2, fit=None,
                          bins1=None, bins2=None,
                          smooth1=1.0, smooth2=1.0):
    """
    Plot 2D posterior PDFs for the bayesian inference analysis on the
    observed data. The fit was done allowing for multiples.

    Possible parameters include:
    distance
    logAge
    alpha
    Mcl
    N_yng_obs
    N_WR_sim
    N_yng_sim
    """
    from jlu.gc.imf import multinest as m

    if fit == None:
        fit_dir = workDir + 'multinest/obs_multi_r_0.0_30.0/'
        #fit_dir = workDir + 'multinest/fit_multi_sim_t6.78_AKs2.7_d8000_a2.35_m10000_multi/'
        fit = m.load_results(fit_dir)

    labels = {'alpha': 'IMF Slope',
              'Mcl': r'Mass (10$^3$ M$_\odot$)',
              'distance': 'Distance (kpc)',
              'age': 'Age (Myr)',
              'logAge': 'Log( Age (Myr) )',
              'N_yng_obs': r'N$_{OB}$',
              'N_WR_sim': r'N$_{WR}$'}

    if bins1 == None:
        bins1 = 25
    if bins2 == None:
        bins2 = 25
    

    import scipy.ndimage as ndimage

    (H, x, y) = np.histogram2d(fit[param1], fit[param2],
                               weights=fit['weights'], bins=[bins1,bins2])
    xcenter = x[:-1] + (np.diff(x) / 2.0)
    ycenter = y[:-1] + (np.diff(y) / 2.0)

    extent = [x[0], x[-1], y[0], y[-1]]
    axes.imshow(H.T, extent=extent, interpolation='nearest', cmap=py.cm.binary)
    axes.axis('tight')
    py.setp(axes.get_xticklabels(), fontsize=16)
    py.setp(axes.get_yticklabels(), fontsize=16)
    axes.set_xlabel(labels[param1])
    axes.set_ylabel(labels[param2])

    # figure out the contour levels for credible intervals
    prob1d = H.flatten()
    prob1d.sort()
    prob1d = prob1d[::-1]
    cumProb1d = prob1d.cumsum()
    idx_68 = np.where(cumProb1d >= 0.68)[0]
    idx_95 = np.where(cumProb1d >= 0.95)[0]
    idx_99 = np.where(cumProb1d >= 0.99)[0]

    val_68 = prob1d[idx_68[0]]
    val_95 = prob1d[idx_95[0]]
    val_99 = prob1d[idx_99[0]]

    if smooth1 != None or smooth2 != None:
        if smooth1 == None or smooth2 == None:
            print('Both smoothing paramters must be set to smooth.')
        else:
            sigma = [smooth1, smooth2]
            H = ndimage.gaussian_filter(H, sigma, order=0)

    axes.contour(xcenter, ycenter, H.T, levels=[val_68, val_95, val_99], colors='black',
                 antialiased=True)

def plot_fit_posteriors_1d(multiples=True, rmin=0.0):
    from jlu.gc.imf import multinest as m

    if multiples:
        fit_dir = '%smultinest/obs_multi_r_%.1f_30.0/' % (workDir, rmin)
    else:
        fit_dir = '%smultinest/obs_single_r_%.1f_30.0/' % (workDir, rmin)
        
    fit = m.load_results(fit_dir)

    # Make a column that contains Myr instead of log(yr)
    fit.add_column('age', 10**(fit['logAge']-6.0), unit='Myr')

    yngData = load_yng_data_by_radius(magCut=15.5, trimWR=True)

    fontsize = 16

    def plot_PDF(ax, paramName, counter=False, label=None):
        bins = {'alpha': 50,
                'age': np.arange(1, 8, 0.1),
                'Mcl': np.arange(0, 20, 0.5),
                'distance': 40,
                'N_WR_sim': np.arange(0, round(fit[paramName].max())),
                'N_yng_obs': np.arange(0, round(fit[paramName].max()))}

        if label == None:
            label = paramName
            
        n, bins, patch = ax.hist(fit[paramName], normed=True, histtype='step',
                                 weights=fit['weights'], bins=bins[paramName],
                                 linewidth=1.5, color='black')
        py.setp(ax.get_xticklabels(), fontsize=fontsize)
        py.setp(ax.get_yticklabels(), fontsize=fontsize)
        ax.set_xlabel(label, size=fontsize+2)
        ax.set_ylim(0, n.max()*1.1)

        return bins


    py.close(1)
    py.figure(1, figsize = (9,8))

    gs = gridspec.GridSpec(2, 2)
    gs.update(left=0.15, right=0.95, bottom=0.1, top=0.95,
              wspace=0.3, hspace=0.3)
    ax1 = py.subplot(gs[0, 0])
    ax2 = py.subplot(gs[0, 1])
    ax3 = py.subplot(gs[1, 0])
    ax4 = py.subplot(gs[1, 1])

    bins_alpha = plot_PDF(ax1, 'alpha', label='IMF slope')
    bins_age = plot_PDF(ax2, 'age', label='Age (Myr))')
    bins_Mcl = plot_PDF(ax3, 'Mcl', label=r'Mass (10$^3$ M$_\odot$)', counter=True)
    bins_dist = plot_PDF(ax4, 'distance', label='Distance (kpc)')

    ax1.set_ylabel('Probability Density', size=fontsize+2)
    ax3.set_ylabel('Probability Density', size=fontsize+2)

    ax1.yaxis.set_label_coords(-0.25, 0.5)
    ax3.yaxis.set_label_coords(-0.25, 0.5)

    # Over plot priors.
    # Distance to cluster
    dist_mean = 8.096  # kpc
    dist_std = 0.483   # kpc
    dist_min = 6.793   # kpc
    dist_max = 9.510   # kpc
    dist_a = (dist_min - dist_mean) / dist_std
    dist_b = (dist_max - dist_mean) / dist_std
    prob_dist = scipy.stats.truncnorm.pdf(bins_dist, dist_a, dist_b,
                                          loc=dist_mean, scale=dist_std)
    ax4.plot(bins_dist, prob_dist, 'k--')

    # Log Age of the cluster
    log_age_min = 6.20
    log_age_max = 7.20
    log_age_diff = log_age_max - log_age_min
    tmp_bins = np.arange(6.15, 7.25, 0.01)
    prob_log_age_cont = scipy.stats.uniform.pdf(tmp_bins,
                                                loc=log_age_min, scale=log_age_diff)
    prob_log_age_cont /= 5.0
    ax2.plot(10**(tmp_bins-6.0), prob_log_age_cont, 'k--')
    print((10**(tmp_bins-6.0)))
    print(prob_log_age_cont)

    # Slope of the IMF
    alpha_min = 0.10
    alpha_max = 3.35
    alpha_diff = alpha_max - alpha_min
    prob_alpha = scipy.stats.uniform.pdf(bins_alpha, loc=alpha_min, scale=alpha_diff)
    ax1.plot(bins_alpha, prob_alpha, 'k--')

    # Total Cluster Mass
    Mcl_min = 1
    Mcl_max = 100
    Mcl_diff = Mcl_max - Mcl_min
    tmp_bins = np.arange(0, 20, 1)
    prob_Mcl = scipy.stats.uniform.pdf(tmp_bins, loc=Mcl_min, scale=Mcl_diff)
    ax3.plot(tmp_bins, prob_Mcl*8, 'k--')

    ax1.set_xlim(0, 3)
    ax2.set_xlim(1, 8)
    ax3.set_xlim(0, 20)
    ax4.set_xlim(7, 9)

    ax1.text(0.85, 0.9, '(a)', transform=ax1.transAxes)
    ax2.text(0.85, 0.9, '(b)', transform=ax2.transAxes)
    ax3.text(0.85, 0.9, '(c)', transform=ax3.transAxes)
    ax4.text(0.85, 0.9, '(d)', transform=ax4.transAxes)

    outFig = '%s/plots/plot_fit_posteriors_1D' % workDir
    if not multiples:
        outFig += '_single'
    if rmin > 0:
        outFig += '_%.1f' % rmin

    py.savefig(outFig + '.png')
    py.savefig(outFig + '.eps')
        


    # Now plot the resulting distributions for the number of WR and OB stars.
    py.close(2)
    py.figure(2, figsize = (10,5))

    gs = gridspec.GridSpec(1, 2)
    gs.update(left=0.15, right=0.95, bottom=0.15, top=0.91,
               wspace=0.3, hspace=0.3)
    ax1 = py.subplot(gs[0, 0])
    ax2 = py.subplot(gs[0, 1])

    plot_PDF(ax1, 'N_WR_sim', counter=True, label='N(WR) simulated')
    plot_PDF(ax2, 'N_yng_obs', counter=True, label='N(OB) simulated')

    # Overplot observed number of WR stars
    ax1.axvline(yngData.N_WR, linestyle='-', color='black', linewidth=2)
    ax2.axvline(yngData.prob.sum(), linestyle='-', color='black', linewidth=2)

    # Fix N(OB) tick labels
    ax2.set_xticks(np.arange(50, 120, 15))

    ax1.set_xlim(0, 20)
    ax2.set_xlim(45, 125)
    ax2.set_ylim(0, 0.08)

    ax1.set_ylabel('Probability Density', size=fontsize+2)
    ax1.yaxis.set_label_coords(-0.25, 0.5)


    outFig = '%s/plots/plot_fit_N_posteriors_1D' % workDir
    if not multiples:
        outFig += '_single'
    if rmin > 0:
        outFig += '_%.1f' % rmin

    py.savefig(outFig + '.png')
    py.savefig(outFig + '.eps')

        
def get_best_fit(param, weights, paramName):
    # The 1-sigma (68.6%) low and high bounds on a CDF
    cdf_lo = (1.0 - 0.686)/2.0
    cdf_hi = (1.0 + 0.686)/2.0
    cdf_lo2 = (1.0 - 0.9545)/2.0
    cdf_hi2 = (1.0 + 0.9545)/2.0
    cdf_lo3 = (1.0 - 0.9973)/2.0
    cdf_hi3 = (1.0 + 0.9973)/2.0

    weightSum = weights.sum()
    exp_param = (param*weights).sum() / weightSum
    sdx_param = param.argsort()
    param_sort = param[sdx_param]
    weights_sort = weights[sdx_param]
    cdf_param = weights_sort.cumsum() / weightSum
    cdf_lo_idx = np.where(cdf_param >= cdf_lo)[0][0]
    cdf_hi_idx = np.where(cdf_param <= cdf_hi)[0][-1]
    cdf_lo2_idx = np.where(cdf_param >= cdf_lo2)[0][0]
    cdf_hi2_idx = np.where(cdf_param <= cdf_hi2)[0][-1]
    cdf_lo3_idx = np.where(cdf_param >= cdf_lo3)[0][0]
    cdf_hi3_idx = np.where(cdf_param <= cdf_hi3)[0][-1]

    # Do a special check for alpha from Bartko.
    if paramName == 'alpha':
        bartkoIdx = np.where(param_sort <= 0.45)[0]
        print(('P(alpha <= 0.45) = %.2e' % cdf_param[bartkoIdx[-1]]))

        salpeterIdx = np.where(param_sort >= 2.35)[0]
        print(('P(alpha < 2.35) = %.2e' % (1.0 - cdf_param[salpeterIdx[0]])))

    param_mean = np.average(param, weights=weights)
    param_std = math.sqrt( np.dot(weights, (param - param_mean)**2) / weightSum)

    print('')
    print(('***** %s *****' % paramName))
    print(('Expectation Value:     %5.2f' % (exp_param)))
    print(('Most Probable:         %5.2f' % (param[-1])))
    print(('Standard Deviation:    %5.2f' % (param_std)))
    print(('Standard Dev. Range:   %5.2f - %5.2f' % \
        (param_mean-param_std, param_mean+param_std)))
    print(('68%% confidence range: %5.2f - %5.2f' % \
        (param_sort[cdf_lo_idx], param_sort[cdf_hi_idx])))
    print(('95%% confidence range: %5.2f - %5.2f' % \
        (param_sort[cdf_lo2_idx], param_sort[cdf_hi2_idx])))
    print(('99%% confidence range: %5.2f - %5.2f' % \
        (param_sort[cdf_lo3_idx], param_sort[cdf_hi3_idx])))
    print('Top 10 most probable:  ')
    print((param[-20:]))
    
    return param_mean, param_std

def table_best_fit_params(multiples=True):
    from jlu.gc.imf import multinest as m

    if multiples:
        fit_dir = workDir + 'multinest/obs_multi_r_0.0_30.0/'
    else:
        fit_dir = workDir + 'multinest/obs_single_r_0.0_30.0/'

    fit = m.load_results(fit_dir)

    # Make a column that contains Myr instead of log(yr)
    fit.add_column('age', 10**(fit['logAge']-6.0), unit='Myr')

    alpha_mean, alpha_std = get_best_fit(fit['alpha'], fit['weights'], 'alpha')
    mass_mean, mass_std = get_best_fit(fit['Mcl'], fit['weights'], 'Mcl')
    dist_mean, dist_std = get_best_fit(fit['distance'], fit['weights'], 'distance')
    age_mean, age_std = get_best_fit(fit['age'], fit['weights'], 'age')

    # Do something special with the age if there are multiple
    # peaks. This is only true for the multiples=True solution.
    if multiples == True:
        # Divide into three peaks.
        idx1 = np.where(fit['age'] <= 3.3)[0]
        idx2 = np.where(fit['age'] > 3.3)[0]

        print('') 
        print('**** Age <= 3.3 Myr *****')
        print('') 
        get_best_fit(fit['age'][idx1], fit['weights'][idx1], 'age1')
        get_best_fit(fit['alpha'][idx1], fit['weights'][idx1], 'alpha')
        get_best_fit(fit['Mcl'][idx1], fit['weights'][idx1], 'Mcl')
        get_best_fit(fit['distance'][idx1], fit['weights'][idx1], 'distance')

        print('') 
        print('**** Age > 3.3 Myr *****')
        print('') 
        age_mean, age_std = get_best_fit(fit['age'][idx2], fit['weights'][idx2], 'age2')
        alpha_mean, alpha_std = get_best_fit(fit['alpha'][idx2], fit['weights'][idx2], 'alpha')
        mass_mean, mass_std = get_best_fit(fit['Mcl'][idx2], fit['weights'][idx2], 'Mcl')
        dist_mean, dist_std = get_best_fit(fit['distance'][idx2], fit['weights'][idx2], 'distance')
        # get_best_fit(fit['age'][idx2], fit['weights'][idx2], 'age2')
        # get_best_fit(fit['alpha'][idx2], fit['weights'][idx2], 'alpha')
        # get_best_fit(fit['Mcl'][idx2], fit['weights'][idx2], 'Mcl')
        # get_best_fit(fit['distance'][idx2], fit['weights'][idx2], 'distance')


    logage_mean = math.log10(age_mean*1e6)

    print('')
    print('***** Plotted Solution *****')
    print(('%-20s  %5.2f' % ('IMF Slope', alpha_mean)))
    print(('%-20s  %5.2f' % ('Age (Myr)', age_mean)))
    print(('%-20s  %5.2f' % ('log[Age (Myr)]', logage_mean)))
    print(('%-20s  %5.2f' % ('Mass (x1000 Msun)', mass_mean)))
    print(('%-20s  %5.2f' % ('Distance (kpc)', dist_mean)))


    if multiples:
        out_suffix = '_fit_multi_means'
    else:
        out_suffix = '_fit_single_means'

    plot_model_vs_data_MC(logage_mean, 2.7, int(round(dist_mean*10**3, 0)),
                          alpha_mean, int(round(mass_mean*10**3, 0)),
        outDir=workDir, outSuffix=out_suffix, makeEPS=True, numMC=100)


def test_num_WR_vs_multiples():
    """
    I saw that for the same set of cluster properties, the number of WR stars was
    larger in a single star simulation vs. in a multiple simulation.

    I finally realized that if stars are only single, then you can add more systems
    (and hence have larger number of WR stars) to achieve the same total mass.
    If you add mass into the cluster via companions, it increases mass without
    increasing the number of WR stars.
    """
    
    from jlu.gc.imf import bayesian as b

    logAge = 6.59
    alpha = 2.33
    mass = 10900

    nIter = 100
    numWR_m = np.zeros(nIter)
    numWR_s = np.zeros(nIter)

    for ii in range(nIter):
        print(('Sim ', ii))
        tmp1 = b.model_young_cluster(logAge, imfSlope=alpha, clusterMass=mass,
                                     makeMultiples=True)
        tmp2 = b.model_young_cluster(logAge, imfSlope=alpha, clusterMass=mass,
                                     makeMultiples=False)

        numWR_m[ii] = tmp1.num_WR
        numWR_s[ii] = tmp2.num_WR

    py.clf()
    py.hist(numWR_m, histtype='step', label='Multiples')
    py.hist(numWR_s, histtype='step', label='Single')
    py.legend()
    py.xlabel('Number of WR stars')


def cluster_mass():
    """
    Calculate conversion factors for the total cluster mass. We haven't observed
    the entire cluster, so we have to correct for our coverage limitations.
    """
    ourYng = load_yng_catalog_by_radius(magCut=15.5)
    ourNames = np.array(ourYng.name)

    # Create a connection to the database file and create a cursor
    connection = sqlite.connect(database)
    cur = connection.cursor()

    # Get info on the stars
    sql = 'select name, kp, x, y, AK_sch from stars where young="T"'
    cur.execute(sql)
    
    rows = cur.fetchall()
    starCnt = len(rows)

    otherNames = []
    otherKp_ext = []

    # Sort out how many Kp<13 stars are outside our field of view
    for ii in range(starCnt):
        record = rows[ii]
        name = str(record[0])

        idx = np.where(ourNames == name)[0]
        if len(idx) == 0:
            
        
        # Calculate extinction corrected photometry
            kp_2_ks_yng = synthetic.get_Kp_Ks(nirc2.AKs[nn], 30000.0, 
                                              filename=synFile)
            nirc2.kp_ext[nn] = nirc2.kp[nn] - kp_2_ks_yng - nirc2.AKs[nn] \
                + theAKs + ks_2_kp_yng

    return

def plot_sims_vs_multiplicity():
    """
    Plot the results of different synthetic clusters with and without multiple
    systems in the simulated clusters and in the fitting.
    """
    from jlu.gc.imf import multinest as m

    logAge = 6.60
    age = 10**(logAge - 6.0)
    alpha = 2.35
    Mcl = 10
    distance = 8
    
    multiDir = workDir + '/multinest/'
    dataPart = '_sim_t%.2f_AKs2.7_d%4d_a%.2f_m%d_o1500_' % \
        (logAge, distance*1e3, alpha, Mcl*1e3)

    # S = single, M = multiples
    root_simS_fitS = multiDir + 'fit_single' + dataPart + 'single/'
    root_simS_fitM = multiDir + 'fit_multi' + dataPart + 'single/'
    root_simM_fitS = multiDir + 'fit_single' + dataPart + 'multi/'
    root_simM_fitM = multiDir + 'fit_multi' + dataPart + 'multi/'
    
    m_simS_fitS = m.load_results(root_simS_fitS)
    m_simS_fitM = m.load_results(root_simS_fitM)
    m_simM_fitS = m.load_results(root_simM_fitS)
    m_simM_fitM = m.load_results(root_simM_fitM)

    m_simS_fitS.add_column('age', 10**(m_simS_fitS.logAge-6.0))
    m_simS_fitM.add_column('age', 10**(m_simS_fitM.logAge-6.0))
    m_simM_fitS.add_column('age', 10**(m_simM_fitS.logAge-6.0))
    m_simM_fitM.add_column('age', 10**(m_simM_fitM.logAge-6.0))

    fontsize = 12

    def prep_plot():
        py.close(1)
        py.figure(1, figsize = (7,11))
        py.subplots_adjust(left=0.1, right=0.95, bottom=0.06, top=0.92,
                           wspace=0.3, hspace=0.3)

        ax11 = py.subplot2grid((4, 2), (0, 0))
        ax12 = py.subplot2grid((4, 2), (1, 0))
        ax13 = py.subplot2grid((4, 2), (2, 0))
        ax14 = py.subplot2grid((4, 2), (3, 0))
        ax21 = py.subplot2grid((4, 2), (0, 1))
        ax22 = py.subplot2grid((4, 2), (1, 1))
        ax23 = py.subplot2grid((4, 2), (2, 1))
        ax24 = py.subplot2grid((4, 2), (3, 1))

        axes = [[ax11, ax12, ax13, ax14], [ax21, ax22, ax23, ax24]]

        return axes

    def plot_PDF(tab, ax, paramName, counter=False):
        bins = {'alpha': 40,
                'age': np.arange(1, 10, 0.20),
                'Mcl': np.arange(0, 20, 0.5),
                'distance': 40}

        xlabels = {'alpha': 'IMF slope',
                   'age': 'Age (Myr))',
                   'Mcl': r'Cluster Mass (M$_\odot$)',
                   'distance': 'Distance (kpc)'}
        
        n, bins, patch = ax.hist(tab[paramName], normed=True, histtype='step',
                                 weights=tab['weights'], bins=bins[paramName], color='black')
        py.setp(ax.get_xticklabels(), fontsize=fontsize)
        py.setp(ax.get_yticklabels(), fontsize=fontsize)
        ax.set_xlabel(xlabels[paramName], size=fontsize+2)
        ax.set_ylim(0, n.max()*1.1)

        if paramName == 'Mcl':
            ax.set_xlim(0, 30)

        return

    # Plot compares the no-multiples cluster with different fits.
    axes = prep_plot()
    plot_PDF(m_simS_fitS, axes[0][0], 'alpha')
    plot_PDF(m_simS_fitS, axes[0][1], 'age')
    plot_PDF(m_simS_fitS, axes[0][2], 'Mcl')
    plot_PDF(m_simS_fitS, axes[0][3], 'distance')
    plot_PDF(m_simS_fitM, axes[1][0], 'alpha')
    plot_PDF(m_simS_fitM, axes[1][1], 'age')
    plot_PDF(m_simS_fitM, axes[1][2], 'Mcl')
    plot_PDF(m_simS_fitM, axes[1][3], 'distance')

    axes[0][0].set_title('Input Singles\nFit Singles')
    axes[1][0].set_title('Input Singles\nFit Multiples')

    axes[0][0].set_ylabel('PDF', size=fontsize+2)
    axes[0][1].set_ylabel('PDF', size=fontsize+2)
    axes[0][2].set_ylabel('PDF', size=fontsize+2)
    axes[0][3].set_ylabel('PDF', size=fontsize+2)

    axes[0][0].axvline(alpha, color='red', linewidth=2)
    axes[0][1].axvline(age, color='red', linewidth=2)
    axes[0][2].axvline(Mcl, color='red', linewidth=2)
    axes[0][3].axvline(distance, color='red', linewidth=2)
    axes[1][0].axvline(alpha, color='red', linewidth=2)
    axes[1][1].axvline(age, color='red', linewidth=2)
    axes[1][2].axvline(Mcl, color='red', linewidth=2)
    axes[1][3].axvline(distance, color='red', linewidth=2)

    outRoot = '{0}plots/plot_sims_vs_multi_simS'.format(multiDir)
    py.savefig('{0}.png'.format(outRoot))
    py.savefig('{0}_color.eps'.format(outRoot))
    convert = 'convert {0}.png -colorspace Gray {0}.eps'.format(outRoot)
    os.system(convert)


    # Plot compares the multiples cluster with different fits.
    axes = prep_plot()
    plot_PDF(m_simM_fitS, axes[0][0], 'alpha')
    plot_PDF(m_simM_fitS, axes[0][1], 'age')
    plot_PDF(m_simM_fitS, axes[0][2], 'Mcl')
    plot_PDF(m_simM_fitS, axes[0][3], 'distance')
    plot_PDF(m_simM_fitM, axes[1][0], 'alpha')
    plot_PDF(m_simM_fitM, axes[1][1], 'age')
    plot_PDF(m_simM_fitM, axes[1][2], 'Mcl')
    plot_PDF(m_simM_fitM, axes[1][3], 'distance')

    axes[0][0].set_title('Input Multiples\nFit Singles')
    axes[1][0].set_title('Input Multiples\nFit Multiples')

    axes[0][0].set_ylabel('PDF', size=fontsize+2)
    axes[0][1].set_ylabel('PDF', size=fontsize+2)
    axes[0][2].set_ylabel('PDF', size=fontsize+2)
    axes[0][3].set_ylabel('PDF', size=fontsize+2)

    axes[0][0].axvline(alpha, color='red', linewidth=2)
    axes[0][1].axvline(age, color='red', linewidth=2)
    axes[0][2].axvline(Mcl, color='red', linewidth=2)
    axes[0][3].axvline(distance, color='red', linewidth=2)
    axes[1][0].axvline(alpha, color='red', linewidth=2)
    axes[1][1].axvline(age, color='red', linewidth=2)
    axes[1][2].axvline(Mcl, color='red', linewidth=2)
    axes[1][3].axvline(distance, color='red', linewidth=2)

    outRoot = '{0}plots/plot_sims_vs_multi_simM'.format(multiDir)
    py.savefig('{0}.png'.format(outRoot))
    py.savefig('{0}_color.eps'.format(outRoot))
    convert = 'convert {0}.png -colorspace Gray {0}.eps'.format(outRoot)
    os.system(convert)

    
def plot_sims_vs_imf_slope():
    """
    Plot the results of different synthetic clusters with Salpeter and top-heavy
    IMF slopes and show that there are no strong biases.
    """
    from jlu.gc.imf import multinest as m

    logAge = 6.78
    distance = 8

    alpha1 = 2.35
    Mcl1 = 10

    alpha2 = 0.45
    Mcl2 = 40

    multiDir = workDir + '/multinest/'
    dataPart1 = '_sim_t%.2f_AKs2.7_d%4d_a%.2f_m%d_o1500_' % \
        (logAge, distance*1e3, alpha1, Mcl1*1e3)
    dataPart2 = '_sim_t%.2f_AKs2.7_d%4d_a%.2f_m%d_o1500_' % \
        (logAge, distance*1e3, alpha2, Mcl2*1e3)

    # SP = Salpeter, TH = top heavy
    root_simSP = multiDir + 'fit_multi' + dataPart1 + 'multi/'
    root_simTH = multiDir + 'fit_multi' + dataPart2 + 'multi/'
    
    m_simSP = m.load_results(root_simSP)
    m_simTH = m.load_results(root_simTH)

    fontsize = 16

    def prep_plot():
        py.close(1)
        #py.figure(1, figsize = (7,11))
        # py.subplots_adjust(left=0.08, right=0.95, bottom=0.06, top=0.95,
        #                    wspace=0.3, hspace=0.3)
        py.figure(1, figsize = (12,6))
        py.subplots_adjust(left=0.08, right=0.95, bottom=0.1, top=0.93,
                           wspace=0.2, hspace=0.2)

        # ax11 = py.subplot2grid((3, 2), (0, 0))
        # ax12 = py.subplot2grid((3, 2), (1, 0))
        # ax13 = py.subplot2grid((3, 2), (2, 0))
        # ax21 = py.subplot2grid((3, 2), (0, 1))
        # ax22 = py.subplot2grid((3, 2), (1, 1))
        # ax23 = py.subplot2grid((3, 2), (2, 1))

        # axes = [[ax11, ax12, ax13], [ax21, ax22, ax23]]

        ax11 = py.subplot2grid((1, 2), (0, 0))
        ax21 = py.subplot2grid((1, 2), (0, 1))

        axes = [[ax11], [ax21]]

        return axes

    def plot_PDF(tab, ax, paramName, counter=False):
        bins = 50

        xlabels = {'alpha': 'IMF slope',
                   'logAge': 'log(t [Myr])',
                   'Mcl': r'Cluster Mass (M$_\odot$)',
                   'distance': 'Distance (kpc)'}
        
        n, bins, patch = ax.hist(tab[paramName], normed=True, histtype='step',
                                 weights=tab['weights'], bins=bins, color='black',
                                 linewidth=1.5)
        py.setp(ax.get_xticklabels(), fontsize=fontsize)
        py.setp(ax.get_yticklabels(), fontsize=fontsize)
        ax.set_xlabel(xlabels[paramName], size=fontsize+2)
        ax.set_ylim(0, n.max()*1.1)

        binInt = bins[1] - bins[0]
        binCent = bins[:-1] + binInt/2.0

        # Determine the peak in the probability distribution
        peakIdx = n.argmax()

        # Walk away from the peak, one on the left, one on the right,
        # until you get an integrated probability of 0.68.
        distFromPeak = np.abs(binCent - binCent[peakIdx])
        sidx = distFromPeak.argsort()

        # Integrated probability
        cdfFromPeak = np.cumsum(n[sidx]) * binInt
        #idx = np.where(cdfFromPeak <= 0.9545)[0]
        idx = np.where(cdfFromPeak <= 0.6827)[0]

        # Pull out the bins and values that fall within this region
        idx_1sigma = np.sort(sidx[idx])
        n_1sigma = n[idx_1sigma]
        b_1sigma = bins[idx_1sigma.min():idx_1sigma.max()+2]

        # Duplicate so we can do a step plot.
        n_1sig_step = np.repeat(n_1sigma, 2)
        b_1sig_step = np.repeat(b_1sigma, 2)[1:-1]

        # Plot filled region
        ax.fill_between(b_1sig_step, n_1sig_step, color='grey', alpha=0.2)
        

        return

    # Plot shows the Salpeter and the Top Heavy posterior PDFs
    axes = prep_plot()
    plot_PDF(m_simSP, axes[0][0], 'alpha')
    # plot_PDF(m_simSP, axes[0][1], 'logAge')
    # plot_PDF(m_simSP, axes[0][2], 'Mcl')
    plot_PDF(m_simTH, axes[1][0], 'alpha')
    # plot_PDF(m_simTH, axes[1][1], 'logAge')
    # plot_PDF(m_simTH, axes[1][2], 'Mcl')

    axes[0][0].set_title('Salpeter')
    axes[1][0].set_title('Top-Heavy')
    axes[0][0].set_ylabel('PDF', size=fontsize+2)

    axes[0][0].axvline(alpha1, color='red', linewidth=2)
    # axes[0][1].axvline(logAge, color='red', linewidth=2)
    # axes[0][2].axvline(Mcl1, color='red', linewidth=2)
    axes[1][0].axvline(alpha2, color='red', linewidth=2)
    # axes[1][1].axvline(logAge, color='red', linewidth=2)
    # axes[1][2].axvline(Mcl2, color='red', linewidth=2)

    outRoot = multiDir + 'plots/plot_sims_vs_alpha'
    py.savefig('{0}.png'.format(outRoot))
    convert = 'convert {0}.png {0}_color.eps'.format(outRoot)
    os.system(convert)
    convert = 'convert {0}.png -colorspace Gray {0}.eps'.format(outRoot)
    os.system(convert)

def calc_total_mass():
    """
    Take the best-fit mass and extrapolate to include the uncovered field of
    view. We do this in different ways.
    """
    ##########
    # Method #1. ASSume spherical symmetry and radial profile from Do et al.
    ##########
    
    # Load up the area maps for the OSIRIS fields
    areaFile = '/u/jlu/work/gc/imf/extinction/nirc2_mask_all.fits'
    areaFITS = pyfits.open(areaFile)
    area_map = areaFITS[0].data
    area_hdr = areaFITS[0].header
    area_map = np.array(area_map, dtype=float)

    area_scaleX = area_hdr['CD1_1'] * 3600.0
    area_scaleY = area_hdr['CD2_2'] * 3600.0
    area_sgraX = area_hdr['CRPIX1']
    area_sgraY = area_hdr['CRPIX2']

    area_mapX = np.arange(area_map.shape[1], dtype=float)
    area_mapY = np.arange(area_map.shape[0], dtype=float)

    area_mapX = (area_mapX - area_sgraX) * area_scaleX
    area_mapY = (area_mapY - area_sgraY) * area_scaleY

    mapXX, mapYY = np.meshgrid(area_mapX, area_mapY)
    mapRR = np.hypot(mapXX, mapYY)

    # Find the maximum radius at which we have data.
    maxRadius = mapRR[area_map > 0].max()
    print(('Maximum Radius Observed: %.2f"' % (maxRadius)))

    # Surface Density profile: flat inner, power law outer.
    # Surface Density Slope and inner Cutoff. Don't care about
    # normalization.
    gamma = 0.9
    innerCut = 0.2
    innerSD = 1.0 / (innerCut**(-gamma))

    # Pixels are small, just make the simplifying assumption that the number
    # of stars in each pixels = surface density(r) * pixel area. Remember
    # the normalization doesn't matter. We are going to eventually take the ratio
    # of observed to un-observed areas, weighted by the surface density.
    mapNstars = mapRR**(-gamma) * np.abs(area_scaleX * area_scaleY)
    mapNstars[mapRR <= 0.2] = innerSD * np.abs(area_scaleX * area_scaleY)
    # Renormalize to 500 stars
    mapNstars *= 500 / mapNstars.sum()

    # Identify observed pixels
    obs = np.where(area_map > 0)
    noobs = np.where(area_map == 0)
    
    total = mapNstars.sum()
    inFOV = mapNstars[obs].sum()
    outFOV = mapNstars[noobs].sum()

    inFraction = inFOV / total
    outFraction = outFOV / total

    print(('mapNstars.sum() = ', total))
    print(('mapNstars[obs].sum() = ', inFOV))
    print(('mapNstars[noobs].sum() = ', outFOV))
    print('')
    print(('Fraction Observed = %.2f' % inFraction))
    print(('Fraction Unobserved = %.2f' % outFraction))

    print('')
    print(('Total Cluster Mass = %.2f * Best-Fit Cluster Mass' % \
        (1.0 / inFraction)))
    print('')
    

    ##########
    # Method #2
    ##########
    # Compare the number of young stars with Kp<13 we detect vs. Paumard.
    connection = sqlite.connect(database)
    cur = connection.cursor()
    sqlCmd = "select ucla from paumard2006 where Kmag < 13"
    cur.execute(sqlCmd)

    rows = cur.fetchall()
    
    ours = load_yng_catalog_by_radius()
    idx = np.where((ours.kp < 13) & (ours.prob == 1))[0]

    Nobs = len(idx)
    Ntot = len(rows)
    inFraction = float(Nobs) / float(Ntot)

    print('')
    print(('N_yng for Kp < 13 total     : %d' % Ntot))
    print(('N_yng for Kp < 13 in our FOV: %d' % Nobs))
    print(('Fraction Observed = %.2f' % inFraction))
    
    print('')
    print(('Total Cluster Mass = %.2f * Best-Fit Cluster Mass' % \
        (1.0 / inFraction)))
    print('')

def compute_masses(rmin=0, rmax=30, magCut=15.5):
    """
    Use the best-fit cluster isochrone to fit system masses for all the
    observed systems. Do this using a Monte Carlo technique. Then combine
    the PDFs of mass together (add them) weighted by their probability of youth.
    All of this is just to make a nice figure of the mass function.
    """
    from jlu.gc.imf import bayesian as b

    # Load up data on young stars.
    d = load_yng_catalog_by_radius(rmin=rmin, rmax=rmax, magCut=magCut)
    noWR = np.where(d.isWR == False)[0]

    # Generate some fake data to compensate for incompleteness.
    binsKp = klf_mag_bins
    binHalfStep = np.diff(binsKp) / 2.0
    binHalfStep = np.append(binHalfStep, binHalfStep[-1])
    binEdges = np.append(binsKp - binHalfStep, binsKp[-1] + binHalfStep[-1])

    # Load up imaging completness curve
    completeness = load_image_completeness_by_radius()

    # Make the KLF.
    klf, bins = np.histogram(d.kp_ext[noWR], bins=binEdges, weights=d.prob[noWR])

    # Make a completeness corrected version.
    Kp_interp = interpolate.splrep(completeness.mag, completeness.comp, k=1, s=0)
    comp_for_klf = interpolate.splev(binsKp, Kp_interp)
    comp_for_klf[comp_for_klf > 1] = 1.0
    comp_for_klf[comp_for_klf < 0] = 0.0

    klf_comp = klf / comp_for_klf
    
    # Determine number of "fake" stars we need to generate in each magnitude bin.
    # Assign them a random magnitude from that bin.
    N_to_comp = np.round(klf_comp - klf)
    print(N_to_comp)

    #### ACTUALLY IGNORING INCOMPLETENESS AS IT DOESN'T EFFECT ANYTHING ####


    # Plot the KLF.
    py.clf()
    py.plot(binsKp, klf, drawstyle='steps-mid', label='Incomplete')
    py.plot(binsKp, klf_comp, drawstyle='steps-mid', label='Complete')
    py.legend(loc='upper left')
    py.xlabel('Kp Magnitude')
    py.ylabel('Number of Stars')

    ##########
    # Generate masses for each star.
    ##########
    imfSlope = fitAlpha
    Mcl = fitMcl
    logAge = fitLogAge
    distance = fitDist
    AKs = theAKs
    filterName = 'Kp'

    iso = b.model_young_cluster(logAge=logAge, imfSlope=imfSlope,
                                clusterMass=1e6, makeMultiples=True)
                                
    iso.mag = iso.mag + 5.0 * np.log10(float(distance) / 8000)

    idx = np.where(iso.isWR == False)
    iso.mass = iso.mass[idx]
    iso.Teff = iso.Teff[idx]
    iso.isWR = iso.isWR[idx]
    iso.mag = iso.mag[idx]
    iso.isMultiple = iso.isMultiple[idx]

    # For each star, Monte Carlo the Kp magnitude and generate a mass PDF.
    # Save these off as useful information.
    numMC = 100

    all_kp = np.zeros((len(noWR), numMC), dtype=float)
    all_m = np.zeros((len(noWR), numMC), dtype=float)
    all_isMulti = np.zeros((len(noWR), numMC), dtype=bool)

    for ss in range(len(noWR)):
        print(('Monte Carlo for star %d' % ss))
        sdx = noWR[ss]
        all_kp[ss,:] = scipy.stats.norm.rvs(size=numMC, loc=d.kp_ext[sdx], scale=d.kp_err[sdx])

        for ii in range(numMC):
            dk = np.abs(all_kp[ss,ii] - iso.mag)
            idx = dk.argmin()

            if dk[idx] > 0.1:
                print(('Bad Mass match for this magnitude: ', all_kp[ss,ii], iso.mag[idx]))

            all_m[ss, ii] = iso.mass[idx]
            all_isMulti[ss, ii] = iso.isMultiple[idx]

    _out = open(workDir + 'best_fit_mass_pdfs.dat', 'w')
    pickle.dump(all_kp, _out)
    pickle.dump(all_m, _out)
    pickle.dump(all_isMulti, _out)
    pickle.dump(iso, _out)
    pickle.dump(d, _out)
    _out.close()

    return

def plot_masses_in_imf():
    """
    Plot the results form the compute_masses() function.
    """
    # Recall that these variables are 2D (starIdx, monteCarloIdx).
    # We ran 100 MC for each star to sample the Kp + error. Then
    # used the best fit isochrone to generate a mass PDF.
    _in = open(workDir + 'best_fit_mass_pdfs.dat', 'r')
    kp = pickle.load(_in)
    mass = pickle.load(_in)
    isMulti = pickle.load(_in)
    iso = pickle.load(_in)
    data = pickle.load(_in)
    _in.close()

    noWR = np.where((data.prob > 0) & (data.isWR == False))[0]
    
    weights = np.repeat([data.prob[noWR]], kp.shape[1], axis=0)
    weights = weights.T
    
    bins = np.arange(0.5, 2.0, 0.1)  # in log mass
    binCenter = bins[:-1] + np.diff(bins)/2.0

    # Make the histogram of the data.
    (n_all, b, p) = py.hist(np.log10(mass), bins=bins, weights=weights)
    n_all = np.array(n_all)
    n = n_all.mean(axis=0)
    n_err = n_all.std(axis=0)

    # Make the same histogram for simulated cluster. Rescale to line up with
    # the observations.
    n_sim, b_sim = np.histogram(np.log10(iso.mass), bins=bins, normed=True)

    idx = n.argmax()
    n_sim *= n[idx:].sum() / n_sim[idx:].sum()

    py.clf()
    py.errorbar(binCenter, n, yerr=n_err, drawstyle='steps-mid', label='Observed',
                linewidth=1.5, color='black')
    # py.plot(binCenter, n_sim, label='Model', linewidth=1.5, color='red')
    # py.text(0.9, 21, r'$\alpha$=1.7', color='red', fontsize=18)
    # py.gca().set_yscale('log')
    py.xlabel(r'log[ Mass (M$_\odot$) ]')
    py.ylabel('Number of Stars')
    py.xlim(1.0, 1.9)
    py.ylim(0, 20)
    # py.legend()
    py.xticks([1.0, 1.2, 1.4, 1.6, 1.8])
    py.savefig(workDir + 'plots/model_vs_data_mass_function.png')
    py.savefig(workDir + 'plots/model_vs_data_mass_function.eps')
    
    # Also print out the individual masses for the stars.
    mass_mean = mass.mean(axis=1)
    mass_std = mass.std(axis=1)
    print((len(mass_mean)))
    print((mass.shape))
    sdx = mass_mean.argsort()[::-1]

    for ii in sdx:
        dii = noWR[ii]
        print(('%-15s  Kp = %5.2f +/- %4.2f   M = %4.1f +/- %3.1f  prob = %.2f' % \
            (data.name[dii], data.kp_ext[dii], data.kp_err[dii],
             mass_mean[ii], mass_std[ii], data.prob[dii])))


def completeness_from_catalog(rmin=0, rmax=30, mmin=8, mmax=15.5):
    """
    Calculate both the imaging and spectroscopic completeness in the
    specified radial and brightness range. This is more time consuming than
    some of the custom codes above. But it is the most useful for very specific
    brightness and radii ranges.

    Completeness is calculated using extinction corrected magnitudes.

    All ranges are fully inclusive.

    Output: (image_comp, spec_comp)
    """
    # Spectroscopic completeness
    d = load_all_catalog_by_radius()
    d.r = np.hypot(d.x, d.y)

    idx = np.where((d.r >= rmin) & (d.r <=rmax) & (d.kp_ext >= mmin) & (d.kp_ext <= mmax))[0]
    n_total = len(idx)
    tdx = np.where(d.prob_err[idx] == 0)[0]
    n_typed = len(tdx)

    if n_total != 0:
        comp_spec = float(n_typed) / float(n_total)
    else:
        comp_spec = np.nan

    print(('r: %4.1f - %4.1f   m: %4.1f - %4.1f  n: %4d out of %4d' % \
        (rmin, rmax, mmin, mmax, n_typed, n_total)))

    # Imaging completeness
    _planted = open(workDir + 'image_completeness_in_osiris.dat', 'r')
    x_in = pickle.load(_planted)
    y_in = pickle.load(_planted)
    kp_in = pickle.load(_planted)
    kp_ext_in = pickle.load(_planted)
    x_out = pickle.load(_planted)
    y_out = pickle.load(_planted)
    kp_out = pickle.load(_planted)
    kp_ext_out = pickle.load(_planted)

    r_in = np.hypot(x_in, y_in)

    idx = np.where((r_in >= rmin) & (r_in <= rmax) & (kp_ext_in >= mmin) & (kp_ext_in <= mmax))[0]
    n_planted = len(idx)
    ddx = np.where(np.isnan(kp_ext_out[idx]) == False)[0]
    n_detected = len(ddx)

    comp_imag = float(n_detected) / float(n_planted)

    return comp_imag, comp_spec



def plot_completeness_for_tdo():
    """
    Make a two-panel plot for completeness vs. Kp and completeness vs. radius.
    """
    # Setup plots
    py.close(1)
    py.figure(1, figsize=(12, 6))
    py.clf()
    py.subplots_adjust(left=0.1)

    ##########
    # First the plots for completeness vs. Kp
    ##########
    magEdges = np.arange(8.5, 16.6, 0.5)
    rmin = 0
    rmax = 30

    comp_imag_kp = np.zeros(len(magEdges)-1, dtype=float)
    comp_spec_kp = np.zeros(len(magEdges)-1, dtype=float)

    for kk in range(len(magEdges)-1):
        mmin = magEdges[kk]
        mmax = magEdges[kk+1]

        comp_imag_kp[kk], comp_spec_kp[kk] = completeness_from_catalog(rmin, rmax,
                                                                       mmin, mmax)

    mags = np.repeat(magEdges, 2)[1:-1]
    comp_imag_kp = np.repeat(comp_imag_kp, 2)
    comp_spec_kp = np.repeat(comp_spec_kp, 2)

    py.subplot(1, 2, 1)
    py.plot(mags, comp_imag_kp, 'b--', label='Imaging', linewidth=2)
    py.plot(mags, comp_spec_kp, 'g-', label='Spectroscopy', linewidth=2)
    py.xlabel('Kp Magnitudes')
    py.ylabel('Completeness')
    py.ylim(0, 1.05)
    py.xlim(9, 15.5)
    py.title(r'%d" $\leq$ R $\leq$ %d"' % (rmin, rmax))
    py.legend(loc='lower left')

    ##########
    # Plots for completeness vs. radius
    ##########
    radiiEdges = np.array([0.0, 2.0, 4.0, 6.0, 8.0, 10.0, 16.0])
    mmin = 14.5
    mmax = 15.5

    comp_imag_r = np.zeros(len(radiiEdges)-1, dtype=float)
    comp_spec_r = np.zeros(len(radiiEdges)-1, dtype=float)
    
    for rr in range(len(radiiEdges)-1):
        rmin = radiiEdges[rr]
        rmax = radiiEdges[rr+1]

        comp_imag_r[rr], comp_spec_r[rr] = completeness_from_catalog(rmin, rmax,
                                                                     mmin, mmax)

    radii = np.repeat(radiiEdges, 2)[1:-1]
    comp_imag_r = np.repeat(comp_imag_r, 2)
    comp_spec_r = np.repeat(comp_spec_r, 2)
    
    py.subplot(1, 2, 2)
    py.plot(radii, comp_imag_r, 'b--', label='Imaging', linewidth=2)
    py.plot(radii, comp_spec_r, 'g-', label='Spectroscopy', linewidth=2)
    py.xlabel('Radius (arcsec)')
    py.ylabel('Completeness')
    py.ylim(0, 1.05)
    py.title(r'%.1f $\leq$ Kp $\leq$ %.1f' % (mmin, mmax))

    py.savefig('%scompleteness_ext_spec_imag_for_tdo.png' % (workDir))
    py.savefig('%scompleteness_ext_spec_imag_for_tdo.eps' % (workDir))


def calc_eccentricity_inner():
    """
    Read in Gillessen's paper and calculate the mean eccentricity for stars
    in the inner 0.03 pc.
    """
    filename = '/u/jlu/work/gc/published/gillessen2009_table7.dat'
    tab = atpy.Table(filename, type='ascii')

    semia = tab.col2
    ecc = tab.col5
    eccErr = tab.col7

    idx = np.where(semia <= 0.75)[0]

    print((ecc[idx].mean()))
    print((np.average(ecc[idx], weights=eccErr[idx])))

    return tab

def print_catalog():
    """
    Print out the entire catalog of GCOWS sources.
    """
    d = load_all_catalog_by_radius(magCut=16.0)
    _gcows = open('%s/gcows_catalog.txt' % workDir, 'w')

    # Sort the catalog
    r = np.hypot(d.x, d.y)
    sdx = r.argsort()

    d.name = d.name[sdx]
    d.x = d.x[sdx] 
    d.y = d.y[sdx] 
    d.kp = d.kp[sdx] 
    d.kp_ext = d.kp_ext[sdx] 
    d.kp_err = d.kp_err[sdx] 
    d.isWR = d.isWR[sdx] 
    d.prob = d.prob[sdx] 
    d.prob_err = d.prob_err[sdx]
    r = r[sdx]

    # Print
    _gcows.write('# %-11s  %6s  %7s  %7s  %5s  %5s  %5s  %4s  %7s  %7s\n' %
                 ('Name', 'R(")', 'X(")', 'Y(")', 'Kp_dA', 'eKp', 'Kp',
                  'isWR', 'P(yng)', 'eP(yng)'))
    for ii in range(len(d.name)):
        _gcows.write('%-13s  %6.3f  %7.3f  %7.3f  ' %
                     (str.upper(d.name[ii]), r[ii], d.x[ii], d.y[ii]))
        _gcows.write('%5.2f  %5.2f  %5.2f  ' %
                     (d.kp_ext[ii], d.kp_err[ii], d.kp[ii]))
        _gcows.write('%4d  %7.3f  %7.3f\n' %
                     (d.isWR[ii], d.prob[ii], d.prob_err[ii]))

    _gcows.close()

    # Do a quick comparison with Tuan's tables.
    old = atpy.Table('/u/jlu/doc/papers/gcows_paper/old_stars_table.tex',
                     type='ascii', delimiter='&', data_start=0)
    yng = atpy.Table('/u/jlu/doc/papers/gcows_paper/young_stars_table.tex',
                     type='ascii', delimiter='&', data_start=0)
    unk = atpy.Table('/u/jlu/doc/papers/gcows_paper/unknownSimTable.tex',
                     type='ascii', delimiter='&', data_start=0)

    names = np.concatenate([old.col1, yng.col1, unk.col1])
    print(('%d in Do list, %d in Lu list' % (len(names), len(d.name))))

    odx = np.where((d.prob_err == 0) & (d.prob == 0))[0]
    ydx = np.where((d.prob_err == 0) & (d.prob == 1))[0]
    udx = np.where((d.prob_err > 0))[0]

    print((len(odx), len(ydx), len(udx)))
    print((len(old.col1), len(yng.col1), len(unk.col1)))

    py.clf()
    py.plot(d.x[odx], d.y[odx], 'rx', ms=10)
    py.plot(old.col3, old.col4, 'r+', ms=10)
    py.plot(d.x[ydx], d.y[ydx], 'bx', ms=10)
    py.plot(yng.col3, yng.col4, 'b+', ms=10)
    py.plot(d.x[udx], d.y[udx], 'gx', ms=10)
    py.plot(unk.col3, unk.col4, 'g+', ms=10)
    py.axis('equal')

    # Loop through and figure out why Tuan's list is so much longer.
    print('')
    print('OLD: In Do not in Lu')
    for ii in range(len(old.col1)):
        if old.col1[ii] not in d.name:
            print((old[ii]))

    print('')
    print('YNG: In Do not in Lu')
    for ii in range(len(yng.col1)):
        if yng.col1[ii] not in d.name:
            print((yng[ii]))

    print('')
    print('UNK: In Do not in Lu')
    for ii in range(len(unk.col1)):
        if unk.col1[ii] not in d.name:
            print((unk[ii]))

    print('')
    print('ALL: In Lu not in Do')
    for ii in range(len(d.name)):
        if (d.name[ii] not in names) and (str.upper(d.name[ii]) not in names):
            print((d.name[ii], d.kp_ext[ii], d.kp[ii], d.x[ii], d.y[ii], d.prob[ii], d.prob_err[ii]))


def sim_age_monte_carlo():
    """
    I ran a Monte Carlo of 10 simulated clusters, all with the same parameters. I want
    to see if the input age is recovered in all the simulations and at what probability
    in the 1D marginalized PDFs.
    """
    outRoot = workDir + 'multinest/' + \
        'fit_single_sim_t6.60_AKs2.7_d8000_a1.70_m10000_o1500_single_v'

    outIdx = np.arange(1, 10+1)

    # Load up the simulation results
    from jlu.gc.imf import multinest

    # Set age bins (in Myr)
    bins = np.arange(1.5, 10, 0.2)
    binSize = bins[1] - bins[0]

    # Find the bin that will contain 3.98 Myr (log(t) = 6.60)
    ageIdx = np.where((bins - 3.98) <= 0)[0][-1]

    from mpl_toolkits.axes_grid1 import AxesGrid

    py.close(1)
    fig = py.figure(1, figsize=(8,12))
    py.clf()
    grid = AxesGrid(fig, [0.1, 0.07, 0.85, 0.9], aspect=False,
                    nrows_ncols=(5, 2), axes_pad=0, share_all=True)
    grid[0].set_ylim(0, 1.51)
    grid[0].set_xlim(1, 10.1)
    grid[8].set_xlabel('Age (Myr)', fontsize=20)
    grid[9].set_xlabel('Age (Myr)', fontsize=20)
    grid[0].set_ylabel('PDF')
    grid[2].set_ylabel('PDF')
    grid[4].set_ylabel('PDF')
    grid[6].set_ylabel('PDF')
    grid[8].set_ylabel('PDF')
    grid[0].set_yticks([0, 0.5, 1.0])
    grid[8].tick_params(axis='x', which='major', labelsize=20)
    grid[9].tick_params(axis='x', which='major', labelsize=20)
    
    for ii in range(len(outIdx)):
        fit = multinest.load_results(outRoot + str(outIdx[ii]) + '/')
        fit.age = 10**(fit.logAge - 6.0)

        
        n, bins, patch = grid[ii].hist(fit.age, normed=True, histtype='step', linewidth=1.5,
                                       weights=fit.weights, bins=bins, color='black',)

        # Sort the bins from highest probability to lowest probability
        sdx = n.argsort()[::-1]

        # Find the point when we reach the true input age
        idx = np.where(sdx == ageIdx)[0]

        grid[ii].axvline(bins[ageIdx] + binSize/2.0, color='red', linestyle='-',
                         linewidth=2)

        confidence = n[sdx[0:idx]].sum() * binSize

        # Sum up all the probability until you reach the true input age and
        # see at what confidence the true age is found.
        print(('Confidence that age = 4 Myr: %.3f' % \
            (confidence)))

        grid[ii].text(8.5, 1.3, '%2d%%' % (confidence*100))

        # print 'Maximum probability peak: %.3f' % \
        #     (n.max())

    outRoot = '{0}plots/sim_age_monte_carlo'.format(workDir)
    py.savefig('{0}.png'.format(outRoot))
    py.savefig('{0}_color.eps'.format(outRoot))
    convert = 'convert {0}.png -colorspace gray {0}.eps'.format(outRoot)
    os.system(convert)

def plot_model_vs_data_MC_for_paper():
    """
    Plot the final KLF observed vs. model comparison plot that goes into the paper.
    """
    age = fitAge
    logAge = fitLogAge
    alpha = fitAlpha
    Mcl = fitMcl
    distance = fitDist
    AKs = theAKs

    numMC = 100

    from jlu.gc.imf import bayesian as b

    binsKp = klf_mag_bins
    binSize = binsKp[1] - binsKp[0]
    binEdges = binsKp - binSize/2.0
    binEdges = np.append(binEdges, binsKp[-1] + binSize/2.0)

    yngData = load_yng_data_by_radius(magCut=15.5, trimWR=True)

    py.close(1)
    py.figure(1)
    py.clf()

    # Plot simulated data
    colorSim = 'red'
    colorObs = 'black'

    klfSim = np.zeros((numMC, len(binEdges)-1), dtype=int)
    nWRSim = np.zeros(numMC, dtype=int)

    for i in range(numMC):
        if i == 0:
            legLabel = 'Sim'
        else:
            legLabel = None
            
        c1 = b.model_young_cluster(logAge, AKs=AKs, distance=8000,
                                   imfSlope=alpha, clusterMass=Mcl,
                                   makeMultiples=True)

        c1.mag_noWR += 5.0 * np.log10(distance / 8000.0)
        o1 = b.sim_to_obs_klf(c1, magCut=15.5, withErrors=False, yng_orig=yngData)

        (n, foo1, foo2) = py.hist(o1.kp, bins=binEdges, histtype='step', color=colorSim,
                                  label=legLabel, linewidth=2, alpha=0.2)
        klfSim[i,:] = n
        nWRSim[i] = o1.N_WR
        py.plot([8.5], o1.N_WR, 'bx', color=colorSim, ms=10)

    klf_mean = np.median(klfSim, axis=0)
    klf_std = klfSim.std(axis=0)
    nWR_mean = np.median(nWRSim)
    nWR_std = nWRSim.std()

    py.clf()
    py.errorbar(binsKp, klf_mean, yerr=klf_std, drawstyle='steps-mid', label='Model',
                linewidth=2.0, color=colorSim)
    py.errorbar([9.0], [nWR_mean], yerr=[nWR_std], fmt='ks', color=colorSim, ms=8)

    # Plot observed data
    (n, b, p) = py.hist(yngData.kp, bins=binEdges, histtype='step', color=colorObs,
                        label='Obs', linewidth=2, weights=yngData.prob)
    py.plot([9.0], yngData.N_WR, 'ko', color=colorObs, ms=8)
    py.legend(loc='upper left')

    py.xlim(8.5, 15.5)
    py.ylim((0, 45))
    py.xlabel('Kp Magnitude')
    py.ylabel('Number of Stars')

    outFile = workDir + 'plots/klf_model_vs_data_MC_multi_for_paper'
    py.savefig(outFile + '.png')
    tmp = 'convert %s.png %s_color.eps' % (outFile, outFile)
    subprocess.call(tmp, shell=True)
    tmp = 'convert %s.png -colorspace Gray %s.eps' % (outFile, outFile)
    subprocess.call(tmp, shell=True)

def plot_klf_with_tmt():
    """
    Plot up a simulated KLF for the young star cluster at the Galactic Center
    and show spectroscopic and photometric detections today with Keck and in the
    future with TMT.
    """

    from jlu.gc.imf import bayesian as b
    
    # Use our best fit IMF for 0.5 - 150 Msun
    # Use Weidner_Kroupa_2004 IMF between 0.1 - 0.5 Msun
    #     Stop at 0.1 Msun as we don't have evolution models below that mass.
    massLimits = np.array([0.1, 0.5, 150])
    powers = np.array([-1.3, -fitAlpha])

    Mcl1 = 3.4e4
    Mcl2 = 3.0e4

    cluster1 = b.model_young_cluster_new(fitLogAge, massLimits=massLimits, imfSlopes=powers,
                                         clusterMass=1e6, makeMultiples=True,
                                         AKs=theAKs, distance=distance)
    cluster2 = b.model_young_cluster_new(fitLogAge, massLimits=massLimits, imfSlopes=powers,
                                         clusterMass=1e6, makeMultiples=False,
                                         AKs=theAKs, distance=distance)

    weights1 = np.ones(len(cluster1.mass))
    weights2 = np.ones(len(cluster1.mass))

    weights1 *= Mcl1 / 1.0e6
    weights2 *= Mcl2 / 1.0e6

    Mcl1_1_150 = (cluster1.systemMasses[cluster1.mass >= 1] * weights1).sum()
    Mcl2_1_150 = (cluster2.systemMasses[cluster2.mass >= 1] * weights2).sum()
    Mcl1_01_1 = (cluster1.systemMasses[cluster1.mass < 1] * weights1).sum()
    Mcl2_01_1 = (cluster2.systemMasses[cluster2.mass < 1] * weights2).sum()

    print('Cluster with multiples:')
    print(('     Total Cluster Mass = %d' % Mcl1))
    print(('    Mass [1 - 150] Msun = %d' % Mcl1_1_150))
    print(('    Mass [0.1 - 1] Msun = %d' % Mcl1_01_1))
    print('Cluster with singles:')
    print(('     Total Cluster Mass = %d' % Mcl2))
    print(('    Mass [1 - 150] Msun = %d' % Mcl2_1_150))
    print(('    Mass [0.1 - 1] Msun = %d' % Mcl2_01_1))

    # Assign arbitrary magnitudes for WR stars. Uniformly distributed from Kp=9-11
    idx = np.where(cluster1.isWR == True)[0]
    cluster1.mag[idx] = 9.0 + (np.random.rand(len(idx))*2)
    idx = np.where(cluster2.isWR == True)[0]
    cluster2.mag[idx] = 9.0 + (np.random.rand(len(idx))*2)

    # Plot the mass luminosity relationship
    py.clf()
    py.semilogy(cluster1.mag, cluster1.mass, 'k.', ms=2, label="Multiples")
    py.semilogy(cluster2.mag, cluster2.mass, 'b.', ms=2, label='Single')
    py.xlabel('Kp Magnitude')
    py.ylabel('Stellar Mass (Msun)')
    py.savefig(workDir + 'plots/tmt_mass_luminosity.png')

    kbins = np.arange(9.0, 24, 0.5)

    # Plot the KLF
    py.clf()
    py.hist(cluster1.mag, bins=kbins, histtype='step', linewidth=2,
            weights=weights1, label='Multiples')
    # py.hist(cluster2.mag, bins=kbins, histtype='step', linewidth=2,
    #         weights=weights2, label='Singles')
    py.gca().set_yscale('log')
    py.xlabel('Kp Magnitude')
    py.ylabel('Number of Stars')
    # py.legend(loc='upper left')
    py.title('Galactic Center Young Cluster')
    py.xlim(9, 22)
    py.ylim(1, 1e4)
    rng = py.axis()

    py.axvline(15.5, color='black', linestyle='--', linewidth=2)
    py.text(15.6, 6e3, r'13 M$_\odot$', horizontalalignment='center')
    ar1 = FancyArrow(15.5, 1e3, -1, 0, width=1e2, color='black', head_length=0.3)
    py.gca().add_patch(ar1)
    py.text(15.3, 7e2, 'Keck\nSpectra', horizontalalignment='right', verticalalignment='top')


    py.axvline(21, color='black', linestyle='-', linewidth=2)
    py.text(21.0, 6e3, r'0.4 M$_\odot$', horizontalalignment='center')
    ar1 = FancyArrow(21, 4e3, -1, 0, width=4e2, color='black', head_length=0.3)
    py.gca().add_patch(ar1)
    py.text(20.8, 3e3, 'TMT\nSpectra', horizontalalignment='right', verticalalignment='top')

    ar1 = FancyArrow(17.8, 10, 0, 13, width=0.1, color='blue', head_length=10)
    py.gca().add_patch(ar1)
    py.text(18, 9, 'Pre-MS\nTurn-On', color='blue',
            horizontalalignment='center', verticalalignment='top')


    outRoot = '{0}plots/tmt_klf_spectral_sensitivity'.format(workDir)
    py.savefig('{0}.png'.format(outRoot))
    py.savefig('{0}_color.eps'.format(outRoot))
    convert = 'convert {0}.png -colorspace Gray {0}.eps'.format(outRoot)
    os.system(convert)

    # Save to output to a file.
    
    return    
    
def plot_imf_vs_salpeter():
    """
    Plot the results form the compute_masses() function.
    Overplot a Salpeter mass function to compare.
    """
    # Recall that these variables are 2D (starIdx, monteCarloIdx).
    # We ran 100 MC for each star to sample the Kp + error. Then
    # used the best fit isochrone to generate a mass PDF.
    _in = open(workDir + 'best_fit_mass_pdfs.dat', 'r')
    kp = pickle.load(_in)
    mass = pickle.load(_in)
    isMulti = pickle.load(_in)
    iso = pickle.load(_in)
    data = pickle.load(_in)
    _in.close()

    noWR = np.where((data.prob > 0) & (data.isWR == False))[0]
    
    weights = np.repeat([data.prob[noWR]], kp.shape[1], axis=0)
    weights = weights.T
    
    bins = np.arange(0.5, 2.0, 0.1)  # in log mass
    binCenter = bins[:-1] + np.diff(bins)/2.0

    # Make the histogram of the data.
    (n_all, b, p) = py.hist(np.log10(mass), bins=bins, weights=weights)
    n_all = np.array(n_all)
    n = n_all.mean(axis=0)
    n_err = n_all.std(axis=0)

    # Make the same histogram for simulated cluster. Rescale to line up with
    # the observations.
    n_sim, b_sim = np.histogram(np.log10(iso.mass), bins=bins, normed=True)

    idx = n.argmax()
    n_sim *= n[idx:].sum() / n_sim[idx:].sum()

    # Make a Salpeter comparison version.
    salpeter = (10**binCenter)**(1.0 - 2.35)
    salpeter *= n[-4] / salpeter[-4]

    py.clf()
    py.errorbar(10**binCenter, n, yerr=n_err, drawstyle='steps-mid', label='Observed',
                linewidth=1.5, color='black')
    py.loglog(10**binCenter, salpeter, 'r--')
    py.xlabel(r'Mass (M$_\odot$)')
    py.ylabel('Number of Stars')
    py.xlim(10, 100)
    py.ylim(1, 50)
    # py.legend()
    py.xticks([10, 20, 30, 40, 50, 60, 70, 80, 90, 100])
    py.savefig(workDir + 'plots/model_vs_data_mass_function_salpeter.png')


def plot_isochrones():
    """
    Examine the possible age range for stars at r < 0.8" by looking at the
    set of stars with T > 6000 and Kp < 15.5.
    """

    ages = np.array([2, 4, 6, 8, 10, 15, 20, 25, 30, 40, 50], dtype=float)
    ages *= 1.0e6
    logAges = np.log10(ages)

    py.figure(1)
    py.clf()

    py.figure(2)
    py.clf()
    
    for aa in range(len(ages)):
        iso = load_isochrone(logAge=logAges[aa])

        idx = np.where(iso.isWR == False)[0]

        # Kp vs. Teff
        py.figure(1)
        # py.scatter(np.log10(iso.T[idx]), iso.mag[idx], c=iso.M[idx],
        #           s=10, vmin=0.1, vmax=50,
        py.scatter(np.log10(iso.T[idx]), iso.mag[idx], c=iso.logg[idx],
                  s=10, vmin=3.0, vmax=4.5,
                  cmap=py.cm.spectral_r, edgecolor='none')
        py.plot(np.log10(iso.T[idx]), iso.mag[idx], 'k-', color='gray', alpha=0.5)

        # L vs. Teff
        py.figure(2)
        # py.scatter(np.log10(iso.T), iso.logL, c=iso.M,
        #           s=10, vmin=0.1, vmax=50,
        py.scatter(np.log10(iso.T), iso.logL, c=iso.logg,
                  s=10, vmin=3.0, vmax=4.5,
                  cmap=py.cm.spectral_r, edgecolor='none')
        py.plot(np.log10(iso.T), iso.logL, 'k-', color='gray', alpha=0.5)

    #####
    # Kp vs. Teff
    #####
    py.figure(1)
    
    # Make a shaded region where we observe things
    xobs = np.log10( np.array([1e5, 1e4, 1e4, 1e5]) )
    yobs = np.array([15.5, 15.5, 5, 5])

    # py.fill(xobs, yobs, 'gray', alpha=0.1)
    cbar = py.colorbar()
    py.gca().invert_xaxis()
    py.gca().invert_yaxis()

    py.xlim(np.log10(50000), np.log10(3000))
    py.ylim(15.5, 6)

    # Plot some of the known stars.
    # S0-2
    s02_Tlo = 19000
    s02_Thi = 30000
    s02_Teff_log = np.log10(s02_Tlo + (s02_Thi - s02_Tlo) / 2.0)
    s02_Tlo_err_log = s02_Teff_log - np.log10(s02_Tlo)
    s02_Thi_err_log = np.log10(s02_Thi) - s02_Teff_log
    s02_Teff_err_log = [[s02_Tlo_err_log],[s02_Thi_err_log]]
    py.errorbar([s02_Teff_log], [14.39], xerr=s02_Teff_err_log, yerr=[0.07])

    py.xlabel(r'log T$_{eff}$ (K)')
    py.ylabel("Kp Magnitude (A$_{Ks}$ = 2.7)")
    cbar.set_label(r'Mass (M$_\odot$)')

    py.savefig(workDir + 'plots/hr_kp.png')


    #####
    # L vs. Teff
    #####
    py.figure(2)
    cbar = py.colorbar()
    py.gca().invert_xaxis()

    py.xlim(np.log10(50000), np.log10(3000))
    py.ylim(2.5, 6.5)

    py.xlabel(r'log T$_{eff}$ (K)')
    py.ylabel(r'log (L/L$_\odot$)')
    cbar.set_label(r'Mass (M$_\odot$)')
    
    
    #####
    # Zoom in and plot all the non-WR stars magnitudes (no Teff info).
    #####
    py.figure(3, figsize=(8, 6))
    py.clf()
    py.subplots_adjust(left=0.11)
    axCenter = py.subplot(111)

    cmap = py.get_cmap('spectral')
    cNorm = matplotlib.colors.Normalize(vmin=0, vmax=len(ages))
    scalarMap = matplotlib.cm.ScalarMappable(norm=cNorm, cmap=cmap)

    legPosX = [4.65, 4.56, 4.4, 4.37, 4.355, 4.32, 4.29, 4.26, 4.23, 4.2, 4.18]
    legPosY = [11.0, 11.3, 11.0, 11.65, 12.2, 12.9, 13.3, 13.7, 14.0, 14.4, 14.8]
    
    for aa in range(len(ages)):
        iso = load_isochrone(logAge=logAges[aa])

        idx = np.where(iso.isWR == False)[0]

        legLabel = '%d Myr' % (ages[aa]/1e6)
        color = scalarMap.to_rgba(aa)

        # Kp vs. Teff
        py.plot(np.log10(iso.T[idx]), iso.mag[idx], 'k.',
                color=color, ms=3, label=legLabel)
        #py.plot(np.log10(iso.T[idx]), iso.mag[idx], 'k-', color='gray', alpha=0.5)

        py.text(legPosX[aa], legPosY[aa], legLabel, color=color)

    py.gca().invert_xaxis()
    py.gca().invert_yaxis()
    py.xlabel(r'log T$_{eff}$ (K)')
    py.ylabel(r'Kp Magnitude (A$_{Ks}$ = 2.7)')
    #py.legend(numpoints=1, ncol=3)
    py.title('Model Isochrones')

    # Make left and top panels.
    divider = make_axes_locatable(axCenter)
    axHist = divider.append_axes('right', size=1.5, pad=0.1, sharey=axCenter)

    axCenter.set_xlim(np.log10(50000), np.log10(10000))
    axCenter.set_ylim(15.5, 8.5)

    # Make the histogram
    d = load_klf_by_radius()
    binHalf = (d.Kp[1] - d.Kp[0]) / 2.0

    xvals = np.zeros(len(d.Kp) * 2)
    yvals = np.zeros(len(d.Kp) * 2)
    for ii in range(len(d.Kp)):
        xvals[ii*2] = d.KLF_ext_cmp_sp_im_noWR[ii]
        xvals[ii*2+1] = d.KLF_ext_cmp_sp_im_noWR[ii]
        yvals[ii*2] = d.Kp[ii] - binHalf
        yvals[ii*2+1] = d.Kp[ii] + binHalf
    
    axHist.plot(xvals, yvals)
    py.setp(axHist.get_yticklabels(), visible=False)
    axHist.set_xticklabels([])
    axHist.set_xlim(0, 0.4)
    axHist.set_title('Observed')

    # py.savefig(workDir + 'plots/hr_kp.png')


def nstasr_vs_age_mass():
    """
    Plot up the number of stars as function of age and total cluster mass
    to explain why the best fit solutions for flatter IMF slopes have higher
    over all cluster masses. I think this is because it is a correlation with
    the age of the cluster.
    """
    from jlu.imf import imf

    alpha = np.arange(0.75, 2.5, 0.25)
    clMass = np.arange(5000, 25000, 2500.)
    maxMass = np.arange(20, 150, 5.0)  # this is a proxy for the cluster's age.
    minMass = 7.0  # Assume that the minimum observable mass is 8 Msun

    nstars = np.zeros((len(alpha), len(clMass), len(maxMass)), dtype=float)
    alpha3d = np.zeros((len(alpha), len(clMass), len(maxMass)), dtype=float)
    clMass3d = np.zeros((len(alpha), len(clMass), len(maxMass)), dtype=float)
    maxMass3d = np.zeros((len(alpha), len(clMass), len(maxMass)), dtype=float)

    imf_mass_range = np.array([1.0, 150.0])

    for a in range(len(alpha)):
        f = imf.IMF_broken_powerlaw(imf_mass_range, np.array([alpha[a]]))
        
        for c in range(len(clMass)):
            f.imf_norm_cl(clMass[c])

            for m in range(len(maxMass)):
                nstars[a, c, m] = f.imf_int_xi(minMass, maxMass[m])
                alpha3d[a, c, m] = alpha[a]
                clMass3d[a, c, m] = clMass[c]
                maxMass3d[a, c, m] = maxMass[m]

    idx = np.where((nstars > 70) & (nstars < 85))
    py.clf()
    py.scatter(maxMass3d[idx], clMass3d[idx], c=alpha3d[idx], edgecolors='none')
    cbar = py.colorbar()
    cbar.set_label('IMF slope')
    py.xlabel('Maximum Stellar Mass (age proxy)')
    py.ylabel('Total Cluster Mass (Msun)')
    py.axvspan(130, 140, color='grey', alpha=0.3)
    py.axvspan(80, 90, color='grey', alpha=0.3)
    py.savefig(workDir + 'plots/nstars_vs_age_mass.png')
    
                
    

def get_Kp_Ks(age, AKs, dist, temp):
    # Define isochrone parameters
    logAge = np.log10(age) # Age in log(years)
    
    # Define evolution/atmosphere models and extinction law
    evo_model = evolution.MergedBaraffePisaEkstromParsec()
    atm_func = atmospheres.get_merged_atmosphere
    red_law = reddening.RedLawHosek18b()
    
    # Also specify filters for synthetic photometry (optional). Here we use
    # the HST WFC3-IR F127M, F139M, and F153M filters
    filt_list = ['nirc2,Kp', '2mass,Ks']

    # Make Isochrone object. Note that is calculation will take a few minutes, unless the
    # isochrone has been generated previously.
    my_iso = synthetic.IsochronePhot(logAge, AKs, dist, metallicity=0,
                                evo_model=evo_model, atm_func=atm_func,
                                red_law=red_law, filters=filt_list,
                                iso_dir='/g/lu/scratch/siyao/work/3_young/imf/isochrone/')

    idx = np.abs(my_iso.points['Teff'] - temp).argmin()
    return my_iso.points[idx]['m_nirc2_Kp'] -  my_iso.points[idx]['m_2mass_Ks']

def spec_completeness_by_fields(plot=True):
    """
    Calculate spectroscopic KLF and completeness curves for each field
    """
    # Load spectral identifications at each filed
    s = load_spec_id_all_stars(flatten=False)
    fields = s.fields

    # Define magnitude bins in which to calculate the completeness
    magStep = klf_mag_bins[1] - klf_mag_bins[0]
    magHalfStep = magStep / 2.0
    magBins = klf_mag_bins

    # Load up Tuan's completeness simulations for the unknown stars.
    specSims = load_unknown_sims()

    # loop through each field
    for i,field in enumerate(fields):

        # count the number of young, old, unknown stars with obs mag
        cnt_yng = np.zeros(len(magBins), dtype=float)
        cnt_old = np.zeros(len(magBins), dtype=float)
        cnt_unk = np.zeros(len(magBins), dtype=float)
        cnt_unk_yng = np.zeros(len(magBins), dtype=float)
        cnt_unk_old = np.zeros(len(magBins), dtype=float)
        cnt_yng_noWR = np.zeros(len(magBins), dtype=float)

        # count the number of young, old, unknown stars with extinction corrected mag
        cnt_ext_yng = np.zeros(len(magBins), dtype=float)
        cnt_ext_old = np.zeros(len(magBins), dtype=float)
        cnt_ext_unk = np.zeros(len(magBins), dtype=float)
        cnt_ext_unk_yng = np.zeros(len(magBins), dtype=float)
        cnt_ext_unk_old = np.zeros(len(magBins), dtype=float)
        cnt_ext_yng_noWR = np.zeros(len(magBins), dtype=float)

        # the error on the number of unknown stars 
        adderr_unk_yng = np.zeros(len(magBins), dtype=float)
        adderr_unk_old = np.zeros(len(magBins), dtype=float)
        adderr_ext_unk_yng = np.zeros(len(magBins), dtype=float)
        adderr_ext_unk_old = np.zeros(len(magBins), dtype=float)

        # Lets keep a copy of all possible young stars
        # (including those with low probabilities) since
        # this will feed into the bayesian analysis.
        # Only keep the extinction corrected magnitudes.
        # Also, don't include the WR stars.
        yng_kp = np.array([], dtype=float)
        yng_kp_err = np.array([], dtype=float)
        yng_prob = np.array([], dtype=float)

        # Also Make a complete table of all possible young stars
        # (including the WR stars). This may be printed in the table.
        allyng_name = np.array([], dtype='S15')
        allyng_x = np.array([], dtype=float)
        allyng_y = np.array([], dtype=float)
        allyng_kp = np.array([], dtype=float)
        allyng_kp_ext = np.array([], dtype=float)
        allyng_kp_err = np.array([], dtype=float)
        allyng_isWR = np.array([], dtype=bool)
        allyng_prob = np.array([], dtype=float)
        allyng_prob_err = np.array([], dtype=float)

        # Make a complete table of all stars
        # (including the WR stars and old stars).
        all_name = np.array([], dtype='S15')
        all_x = np.array([], dtype=float)
        all_y = np.array([], dtype=float)
        all_kp = np.array([], dtype=float)
        all_kp_ext = np.array([], dtype=float)
        all_kp_err = np.array([], dtype=float)
        all_isWR = np.array([], dtype=bool)
        all_prob = np.array([], dtype=float)
        all_prob_err = np.array([], dtype=float)

        # Also keep track of the number of WR stars
        idx = np.where(s.isWR_yng == True)[0]
        yng_N_WR = len(idx)

        for mm in range(len(magBins)):
            magLo = magBins[mm] - magHalfStep
            magHi = magBins[mm] + magHalfStep

            print(('Working on mag bin %.2f - %.2f' % (magLo, magHi)))

            # Before extinction correction, figure out which stars
            # fall into the magnitude bin and radius range.
            yy = np.where((s.kp_yng[i] >= magLo) & (s.kp_yng[i] < magHi))[0]
            oo = np.where((s.kp_old[i] >= magLo) & (s.kp_old[i] < magHi)) [0] 
            nn = np.where((s.kp_nirc2[i] >= magLo) & (s.kp_nirc2[i] < magHi))[0]
            yy_noWR = np.where((s.kp_yng[i] >= magLo) & (s.kp_yng[i] < magHi) &
                               (s.isWR_yng[i] == False))[0]

            cnt_yng[mm] = len(yy)
            cnt_old[mm] = len(oo)
            cnt_unk[mm] = len(nn) - len(yy) - len(oo)
            cnt_yng_noWR[mm] = len(yy_noWR)

            # Gather up all the unknown sources and find them in the
            # Monte Carlo simulations table (specSims)
            unk_probY = []
            unk_probO = []
            unk_probYerr = []
            unk_probOerr = []
            unk_sdx = []

            for ni in nn:
                name = s.name_nirc2[i][ni]
                if (name in s.name_yng[i]) or (name in s.name_old[i]):
                    continue

                unk_sdx.append(ni)

                # Find this unkonwn source in the spectral star planting results
                idx = np.where(name == specSims['name'])[0]
                if len(idx) == 0:
                    print(('    Unknown source %s (Kp = %5.2f, field = %s) not in MC sim. Resorting to priors.' % \
                        (name, s.kp_nirc2[i][ni], s.field_nirc2[i][ni])))
                    priorYngRad, priorOldRad = priorsAtRadius(np.hypot(s.x_nirc2[i][ni], s.y_nirc2[i][ni]))
                    unk_probY.append( priorYngRad )
                    unk_probO.append( priorOldRad )
                    unk_probYerr.append( 0.1 )
                    unk_probOerr.append( 0.1 )
                else:
                    unk_probY.append( specSims['probYng'][idx[0]] )
                    unk_probO.append( specSims['probOld'][idx[0]] )
                    unk_probYerr.append( specSims['probYngErr'][idx[0]] )
                    unk_probOerr.append( specSims['probOldErr'][idx[0]] )

            unk_probY = np.array(unk_probY)
            unk_probO = np.array(unk_probO)
            unk_probYerr = np.array(unk_probYerr)
            unk_probOerr = np.array(unk_probOerr)
            unk_sdx = np.array(unk_sdx)

            if len(unk_probY) != cnt_unk[mm]:
                print('Incorrect number of unknown sources:')
                print((mm, len(unk_probY), cnt_unk[mm], len(nn), len(yy), len(oo)))
                pdb.set_trace()
                
            # For the unknown sources, figure out how many are likely
            # to be young and how many are likely to be old.
            cnt_unk_yng[mm] = unk_probY.sum()
            cnt_unk_old[mm] = unk_probO.sum()
            adderr_unk_yng[mm] = math.sqrt((unk_probYerr**2).sum())
            adderr_unk_old[mm] = math.sqrt((unk_probOerr**2).sum())

            print(('  %3d yng, %3d old, %3d unkown (%4.1f yng, %4.1f old), %3d noWR (observed)' % \
                (cnt_yng[mm], cnt_old[mm], cnt_unk[mm],
                 cnt_unk_yng[mm], cnt_unk_old[mm], cnt_yng_noWR[mm])))

            ##########
            # After extinction correction
            ##########
            yy = np.where((s.kp_ext_yng[i] >= magLo) & (s.kp_ext_yng[i] < magHi))[0]
            oo = np.where((s.kp_ext_old[i] >= magLo) & (s.kp_ext_old[i] < magHi))[0]
            nn = np.where((s.kp_ext_nirc2[i] >= magLo) & (s.kp_ext_nirc2[i] < magHi))[0]
            yy_noWR = np.where((s.kp_ext_yng[i] >= magLo) & (s.kp_ext_yng[i] < magHi) & 
                               (s.isWR_yng[i] == False))[0]

            cnt_ext_yng[mm] = len(yy)
            cnt_ext_old[mm] = len(oo)
            cnt_ext_unk[mm] = len(nn) - len(yy) - len(oo)
            cnt_ext_yng_noWR[mm] = len(yy_noWR)

            yng_kp = np.append(yng_kp, s.kp_ext_yng[i][yy_noWR])
            yng_kp_err = np.append(yng_kp_err, s.kperr_yng[i][yy_noWR])
            yng_prob = np.append(yng_prob, np.ones(len(yy_noWR)))

            # Append the known young stars to the "allyng" list. 
            allyng_name = np.append(allyng_name, s.name_yng[i][yy])
            allyng_x = np.append(allyng_x, s.x_yng[i][yy])
            allyng_y = np.append(allyng_y, s.y_yng[i][yy])
            allyng_kp = np.append(allyng_kp, s.kp_yng[i][yy])
            allyng_kp_err = np.append(allyng_kp_err, s.kperr_yng[i][yy])
            allyng_kp_ext = np.append(allyng_kp_ext, s.kp_ext_yng[i][yy])
            allyng_prob = np.append(allyng_prob, np.ones(len(yy)))
            allyng_isWR = np.append(allyng_isWR, s.isWR_yng[i][yy])

            # Append the known young stars to the "all" list. 
            all_name = np.append(all_name, s.name_yng[i][yy])
            all_x = np.append(all_x, s.x_yng[i][yy])
            all_y = np.append(all_y, s.y_yng[i][yy])
            all_kp = np.append(all_kp, s.kp_yng[i][yy])
            all_kp_err = np.append(all_kp_err, s.kperr_yng[i][yy])
            all_kp_ext = np.append(all_kp_ext, s.kp_ext_yng[i][yy])
            all_prob = np.append(all_prob, np.ones(len(yy)))
            all_prob_err = np.append(all_prob_err, np.zeros(len(yy)))
            all_isWR = np.append(all_isWR, s.isWR_yng[i][yy])

            # Append the known old stars to the "all" list. 
            all_name = np.append(all_name, s.name_old[i][oo])
            all_x = np.append(all_x, s.x_old[i][oo])
            all_y = np.append(all_y, s.y_old[i][oo])
            all_kp = np.append(all_kp, s.kp_old[i][oo])
            all_kp_err = np.append(all_kp_err, s.kperr_old[i][oo])
            all_kp_ext = np.append(all_kp_ext, s.kp_ext_old[i][oo])
            all_prob = np.append(all_prob, np.zeros(len(oo), dtype=float))
            all_prob_err = np.append(all_prob_err, np.zeros(len(oo), dtype=float))
            all_isWR = np.append(all_isWR, np.zeros(len(oo), dtype=bool))

            # Gather up all the unknown sources and find them in the
            # Monte Carlo simulations table (specSims)
            unk_probY = []
            unk_probO = []
            unk_probYerr = []
            unk_probOerr = []
            unk_sdx = []

            for ni in nn:
                name = s.name_nirc2[i][ni]
                if (name in s.name_yng[i]) or (name in s.name_old[i]):
                    continue

                unk_sdx.append(ni)
                
                idx = np.where(name == specSims['name'])[0]
                if len(idx) == 0:
                    print(('    Unknown source %s (Kp = %5.2f, field = %s) not in MC sim. Resorting to priors.' % \
                        (name, s.kp_nirc2[i][ni], s.field_nirc2[i][ni])))

                    priorYngRad, priorOldRad = priorsAtRadius(np.hypot(s.x_nirc2[i][ni],s.y_nirc2[i][ni]))
                    unk_probY.append( priorYngRad )
                    unk_probO.append( priorOldRad )
                    unk_probYerr.append( 0.1 )
                    unk_probOerr.append( 0.1 )
                else:
                    unk_probY.append( specSims['probYng'][idx[0]] )
                    unk_probO.append( specSims['probOld'][idx[0]] )
                    unk_probYerr.append( specSims['probYngErr'][idx[0]] )
                    unk_probOerr.append( specSims['probOldErr'][idx[0]] )

            unk_probY = np.array(unk_probY)
            unk_probO = np.array(unk_probO)
            unk_probYerr = np.array(unk_probYerr)
            unk_probOerr = np.array(unk_probOerr)
            unk_sdx = np.array(unk_sdx)

            if len(unk_probY) != cnt_ext_unk[mm]:
                print('Incorrect number of unknown sources:')
                print((mm, len(unk_probY), cnt_ext_unk[mm], len(nn), len(yy), len(oo)))

            # For the unknown sources, figure out how many are likely
            # to be young and how many are likely to be old.
            cnt_ext_unk_yng[mm] = unk_probY.sum()
            cnt_ext_unk_old[mm] = unk_probO.sum()
            adderr_ext_unk_yng[mm] = math.sqrt((unk_probYerr**2).sum())
            adderr_ext_unk_old[mm] = math.sqrt((unk_probOerr**2).sum())

            print(('  %3d yng, %3d old, %3d unkown (%4.1f yng, %4.1f old), %3d no WR (extinction corrected)' % \
                (cnt_ext_yng[mm], cnt_ext_old[mm], cnt_ext_unk[mm],
                 cnt_ext_unk_yng[mm], cnt_ext_unk_old[mm], cnt_ext_yng_noWR[mm])))

            # Save the unknown sources (that are young) to our lists.
            ndx = np.where(unk_probY != 0)[0]
            if (len(ndx) != 0):
                yng_kp = np.append(yng_kp, s.kp_ext_nirc2[i][unk_sdx[ndx]])
                yng_kp_err = np.append(yng_kp_err, s.kperr_nirc2[i][unk_sdx[ndx]])
                yng_prob = np.append(yng_prob, unk_probY[ndx])

                allyng_name = np.append(allyng_name, s.name_nirc2[i][unk_sdx[ndx]])
                allyng_x = np.append(allyng_x, s.x_nirc2[i][unk_sdx[ndx]])
                allyng_y = np.append(allyng_y, s.y_nirc2[i][unk_sdx[ndx]])
                allyng_kp = np.append(allyng_kp, s.kp_nirc2[i][unk_sdx[ndx]])
                allyng_kp_err = np.append(allyng_kp_err, s.kperr_nirc2[i][unk_sdx[ndx]])
                allyng_kp_ext = np.append(allyng_kp_ext, s.kp_ext_nirc2[i][unk_sdx[ndx]])
                allyng_prob = np.append(allyng_prob, unk_probY[ndx])
                allyng_prob_err = np.append(allyng_prob_err, unk_probYerr[ndx])
                allyng_isWR = np.append(allyng_isWR, np.zeros(len(ndx), dtype=bool))

            # Append all the unkown stars to the list.
            if len(unk_sdx) > 0:
                all_name = np.append(all_name, s.name_nirc2[i][unk_sdx])
                all_x = np.append(all_x, s.x_nirc2[i][unk_sdx])
                all_y = np.append(all_y, s.y_nirc2[i][unk_sdx])
                all_kp = np.append(all_kp, s.kp_nirc2[i][unk_sdx])
                all_kp_err = np.append(all_kp_err, s.kperr_nirc2[i][unk_sdx])
                all_kp_ext = np.append(all_kp_ext, s.kp_ext_nirc2[i][unk_sdx])
                all_prob = np.append(all_prob, unk_probY)
                all_prob_err = np.append(all_prob_err, unk_probYerr)
                all_isWR = np.append(all_isWR, np.zeros(len(unk_sdx), dtype=bool))


        
        comp = (cnt_yng + cnt_old) / (cnt_yng + cnt_old + cnt_unk)
        comp_ext = (cnt_ext_yng + cnt_ext_old) / (cnt_ext_yng + cnt_ext_old + cnt_ext_unk)
        comp_fix = fix_osiris_spectral_comp(magBins, comp)
        comp_ext_fix = fix_osiris_spectral_comp(magBins, comp_ext)

        comp_yng = cnt_yng / (cnt_yng + cnt_unk_yng)
        comp_ext_yng = cnt_ext_yng / (cnt_ext_yng + cnt_ext_unk_yng)
        comp_fix_yng = fix_osiris_spectral_comp(magBins, comp_yng)
        comp_ext_fix_yng = fix_osiris_spectral_comp(magBins, comp_ext_yng)

        comp_old = cnt_old / (cnt_old + cnt_unk_old)
        comp_ext_old = cnt_ext_old / (cnt_ext_old + cnt_ext_unk_old)
        comp_fix_old = fix_osiris_spectral_comp(magBins, comp_old)
        comp_ext_fix_old = fix_osiris_spectral_comp(magBins, comp_ext_old)

        outRoot = '%sspec_completeness_field_%s' % (workDir, field)

        if plot == True:
            # Total completeness
            py.clf()
            py.plot(magBins, comp_ext_fix, 'ks-', label='Ext. Correct, Fixed')
            py.plot(magBins, comp_ext, 'ks--', label='Ext. Correct')
            py.plot(magBins, comp, 'r^--', label='Raw')
            py.xlabel('Kp Magnitude')
            py.ylabel('Completeness')
            py.title('field %s' %(field))
            py.ylim(0, 1.1)
            py.xlim(9, 18)
            py.legend(loc='lower left')
            py.savefig(outRoot + '.png')

            # Completeness to young stars
            py.clf()
            py.plot(magBins, comp_ext_fix_yng, 'ks-', label='Ext. Correct, Fixed')
            py.plot(magBins, comp_ext_yng, 'ks--', label='Ext. Correct')
            py.plot(magBins, comp_yng, 'r^--', label='Raw')
            py.xlabel('Kp Magnitude')
            py.ylabel('Completeness')
            py.title('field %s' % (field))
            py.ylim(0, 1.1)
            py.xlim(9, 18)
            py.legend(loc='lower left')
            py.savefig(outRoot + '_yng.png')

            # Completeness to old stars 
            py.clf()
            py.plot(magBins, comp_ext_fix_old, 'ks-', label='Ext. Correct, Fixed')
            py.plot(magBins, comp_ext_old, 'ks--', label='Ext. Correct')
            py.plot(magBins, comp_old, 'r^--', label='Raw')
            py.xlabel('Kp Magnitude')
            py.ylabel('Completeness')
            py.title('field %s' % (field))
            py.ylim(0, 1.1)
            py.ylim(0, 1.1)
            py.xlim(9, 18)
            py.legend(loc='lower left')
            py.savefig(outRoot + '_old.png')

            # Completeness compared
            py.clf()
            py.plot(magBins, comp_ext_fix, 'ks-', label='Total Spec. Comp.')
            py.plot(magBins, comp_ext_fix_yng, 'bo-', label='Young Spec. Comp.')
            py.plot(magBins, comp_ext_fix_old, 'r^-', label='Old Spec. Comp.')
            py.xlabel('Kp Magnitude')
            py.ylabel('Completeness')
            py.title('field %s' % (field))
            py.ylim(0, 1.1)
            py.xlim(9, 18)
            py.legend(loc='lower left')
            py.savefig(outRoot + '_yng_v_old.png')

        # Save the original list of young star magnitudes
        _pick2 = open(outRoot + '_yng_kp.pickle', 'wb')
        pickle.dump(yng_kp, _pick2)
        pickle.dump(yng_kp_err, _pick2)
        pickle.dump(yng_prob, _pick2)
        pickle.dump(yng_N_WR, _pick2)
        _pick2.close()

        # Save the original list of young star -- complete info
        _pick3 = open(outRoot + '_yng_info.pickle', 'wb')
        pickle.dump(allyng_name, _pick3)
        pickle.dump(allyng_x, _pick3)
        pickle.dump(allyng_y, _pick3)
        pickle.dump(allyng_kp, _pick3)
        pickle.dump(allyng_kp_ext, _pick3)
        pickle.dump(allyng_kp_err, _pick3)
        pickle.dump(allyng_isWR, _pick3)
        pickle.dump(allyng_prob, _pick3)
        pickle.dump(allyng_prob_err, _pick3)
        _pick3.close()

        # Save the original list of all stars -- complete info
        _pick3 = open(outRoot + '_all_info.pickle', 'wb')
        pickle.dump(all_name, _pick3)
        pickle.dump(all_x, _pick3)
        pickle.dump(all_y, _pick3)
        pickle.dump(all_kp, _pick3)
        pickle.dump(all_kp_ext, _pick3)
        pickle.dump(all_kp_err, _pick3)
        pickle.dump(all_isWR, _pick3)
        pickle.dump(all_prob, _pick3)
        pickle.dump(all_prob_err, _pick3)
        _pick3.close()

        # Save the completeness and KLF curves
        _pick = open(outRoot + '.pickle', 'wb')
        pickle.dump(magBins, _pick)
        pickle.dump(cnt_yng, _pick)
        pickle.dump(cnt_old, _pick)
        pickle.dump(cnt_unk, _pick)
        pickle.dump(cnt_yng_noWR, _pick)
        pickle.dump(cnt_unk_yng, _pick)
        pickle.dump(cnt_unk_old, _pick)
        pickle.dump(adderr_unk_yng, _pick)
        pickle.dump(adderr_unk_old, _pick)
        pickle.dump(cnt_ext_yng, _pick)
        pickle.dump(cnt_ext_old, _pick)
        pickle.dump(cnt_ext_unk, _pick)
        pickle.dump(cnt_ext_yng_noWR, _pick)
        pickle.dump(cnt_ext_unk_yng, _pick)
        pickle.dump(cnt_ext_unk_old, _pick)
        pickle.dump(adderr_ext_unk_yng, _pick)
        pickle.dump(adderr_ext_unk_old, _pick)
        pickle.dump(comp, _pick)
        pickle.dump(comp_ext, _pick)
        pickle.dump(comp_fix, _pick)
        pickle.dump(comp_ext_fix, _pick)
        pickle.dump(comp_yng, _pick)
        pickle.dump(comp_ext_yng, _pick)
        pickle.dump(comp_fix_yng, _pick)
        pickle.dump(comp_ext_fix_yng, _pick)
        pickle.dump(comp_old, _pick)
        pickle.dump(comp_ext_old, _pick)
        pickle.dump(comp_fix_old, _pick)
        pickle.dump(comp_ext_fix_old, _pick)
        _pick.close()

        # Text file
        t = Table()
        t['mag'] = magBins
        t['comp_yng'] = comp_fix_yng
        t['comp_ext_yng'] = comp_ext_fix_yng
        t['cnt_ext_yng'] = cnt_ext_yng
        t['cnt_ext_yng_noWR'] = cnt_ext_yng_noWR
        t['cnt_ext_old'] = cnt_ext_old
        t['cnt_ext_unk'] = cnt_ext_unk
        t['cnt_ext_unk_yng'] = cnt_ext_unk_yng
        t['cnt_ext_unk_old'] = cnt_ext_unk_old

        t['comp_yng'].format = '5.2f'
        t['comp_ext_yng'].format = '5.2f'
        t['cnt_ext_unk_yng'].format = '6.1f'
        t['cnt_ext_unk_old'].format = '6.1f'
        t.write(outRoot+'.txt', format='ascii.fixed_width', delimiter='|')

        pdb.set_trace()


