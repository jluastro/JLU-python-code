"""
Photometric Calibration Codes for the 2013-03-12 HST reduction.
"""

from jlu.hst import starlists
import atpy
import math
import os
import numpy as np
import pylab as py
from jlu.util import statsIter


work_dir = '/u/jlu/data/Wd1/hst/reduce_2013_03_12/'
phot_dir = work_dir + 'aper_phot/'

images = {'F125W': ['ib5w02010_drz', 'ib5w03010_drz', 'ib5w04010_drz', 'ib5w05010_drz'],
          'F139M': ['ib5w02020_drz', 'ib5w03020_drz', 'ib5w04020_drz', 'ib5w05020_drz'],
          'F160W': ['ib5w02030_drz', 'ib5w03030_drz', 'ib5w04030_drz', 'ib5w05030_drz'],
          'F814W': ['j8z201010_drz']
          }


def run_all_images():
    # process_ks2_output()
    # process_brite_output()

    ##########
    # Infrared
    ##########

    # for ii in range(len(images['F125W'])):
    #     # run_daophot_wfc3ir(images['F125W'][ii])
    #     # process_daophot_output(images['F125W'][ii])
    #     matchup_lists(images['F125W'][ii], 'F125W')
    #     matchup_lists_brite(images['F125W'][ii], 'F125W')

    # for ii in range(len(images['F139M'])):
    #     # run_daophot_wfc3ir(images['F139M'][ii])
    #     # process_daophot_output(images['F139M'][ii])
    #     matchup_lists(images['F139M'][ii], 'F139M')
    #     matchup_lists_brite(images['F139M'][ii], 'F139M')

    # for ii in range(len(images['F160W'])):
    #     # run_daophot_wfc3ir(images['F160W'][ii])
    #     # process_daophot_output(images['F160W'][ii])
    #     matchup_lists(images['F160W'][ii], 'F160W')
    #     matchup_lists_brite(images['F160W'][ii], 'F160W')


    # combine_matched_lists('F125W', brite=False)
    # combine_matched_lists('F125W', brite=True)

    # combine_matched_lists('F139M', brite=False)
    # combine_matched_lists('F139M', brite=True)

    # combine_matched_lists('F160W', brite=False)
    # combine_matched_lists('F160W', brite=True)

    mag_range = [-11.0, -9.5]

    plot_mag_offsets('matched_ks2_apphot_F125W.fits', mag_range=mag_range)
    plot_mag_offsets('matched_ks2_apphot_F125W_brite.fits', mag_range=mag_range)

    plot_mag_offsets('matched_ks2_apphot_F139M.fits', mag_range=mag_range)
    plot_mag_offsets('matched_ks2_apphot_F139M_brite.fits', mag_range=mag_range)

    plot_mag_offsets('matched_ks2_apphot_F160W.fits', mag_range=mag_range)
    plot_mag_offsets('matched_ks2_apphot_F160W_brite.fits', mag_range=mag_range)
    
    ##########
    # Optical F814W
    ##########
    # for ii in range(len(images['F814W'])):
        # run_daophot_wfcacs(images['F814W'][ii])
        # process_daophot_output(images['F814W'][ii])
        # matchup_lists(images['F814W'][ii], 'F814W')
        # matchup_lists_brite(images['F814W'][ii], 'F814W')

    # combine_matched_lists('F814W', brite=False)
    # combine_matched_lists('F814W', brite=True)

    mag_range = [-13.0, -10.5]

    plot_mag_offsets('matched_ks2_apphot_F814W.fits', mag_range=mag_range)
    # plot_mag_offsets('matched_ks2_apphot_F814W_brite.fits', mag_range=mag_range)


def process_ks2_output():
    """
    Pre-process the ks2 output to make a *.XYM file for all three filters.
    """
    stars = atpy.Table(work_dir + '20.KS2_PMA/wd1_catalog.fits')

    _out1 = open(phot_dir + 'ks2_F125W.xym', 'w')
    _out2 = open(phot_dir + 'ks2_F139M.xym', 'w')
    _out3 = open(phot_dir + 'ks2_F160W.xym', 'w')
    _out4 = open(phot_dir + 'ks2_F814W.xym', 'w')
    
    # KS2 for F814W output is in Electrons (not e-/sec) per exposure.
    # Lets convert this to e-/sec instrumental magnitudes to match drizzle.
    tint_814 = 2407.0
    nexp_814 = 3.0
    # stars.m_814 -= -2.5 * math.log10(tint_814 / nexp_814)

    for ii in range(len(stars)):
        if (stars.me_125[ii] > 0) and (stars.me_125[ii] < 8):
            _out1.write('%8.2f  %8.2f  %6.2f\n' %
                        (stars.x_125[ii], stars.y_125[ii], stars.m_125[ii]))
        if (stars.me_139[ii] > 0) and (stars.me_139[ii] < 8):
            _out2.write('%8.2f  %8.2f  %6.2f\n' %
                        (stars.x_139[ii], stars.y_139[ii], stars.m_139[ii]))
        if (stars.me_160[ii] > 0) and (stars.me_160[ii] < 8):
            _out3.write('%8.2f  %8.2f  %6.2f\n' %
                        (stars.x_160[ii], stars.y_160[ii], stars.m_160[ii]))
        if (stars.me_814[ii] > 0) and (stars.me_814[ii] < 8):
            _out4.write('%8.2f  %8.2f  %6.2f\n' %
                        (stars.x_814[ii], stars.y_814[ii], stars.m_814[ii]))

    _out1.close()
    _out2.close()
    _out3.close()
    _out4.close()

def process_brite_output():
    """
    Make xym files for the brite stars.
    """
    stars1 = atpy.Table(work_dir + '12.KS2_2010/BRITE.XYMMMM', type='ascii')
    stars2 = atpy.Table(work_dir + '13.KS2_2005/BRITE.XYM', type='ascii')

    _out1 = open(phot_dir + 'brite_F125W.xym', 'w')
    _out2 = open(phot_dir + 'brite_F139M.xym', 'w')
    _out3 = open(phot_dir + 'brite_F160W.xym', 'w')
    _out4 = open(phot_dir + 'brite_F814W.xym', 'w')
    
    for ii in range(len(stars1)):
        _out1.write('%8.2f  %8.2f  %6.2f\n' %
                    (stars1.col1[ii], stars1.col2[ii], stars1.col5[ii]))
        _out2.write('%8.2f  %8.2f  %6.2f\n' %
                    (stars1.col1[ii], stars1.col2[ii], stars1.col4[ii]))
        _out3.write('%8.2f  %8.2f  %6.2f\n' %
                    (stars1.col1[ii], stars1.col2[ii], stars1.col3[ii]))


    for ii in range(len(stars2)):
        _out4.write('%8.2f  %8.2f  %6.2f\n' %
                    (stars2.col1[ii], stars2.col2[ii], stars2.col3[ii]))

    _out1.close()
    _out2.close()
    _out3.close()
    _out4.close()


def run_daophot_wfc3ir(imgRoot):
    """
    Run daofind and phot (aperture photometry) on a WFC3-IR DRZ image.
    The output files are *.coo for the coordinates from daofind and *.mag
    for the phot output. An aperture of 0.4" is used (3.12 pixels).
    """
    scale = 0.1283   # arcsec / pixel
    aperture = 0.4   # arcsec

    run_daophot(imgRoot, aperture / scale)

def run_daophot_acswfc(imgRoot):
    """
    Run daofind and phot (aperture photometry) on a ACS WFC DRZ image.
    The output files are *.coo for the coordinates from daofind and *.mag
    for the phot output. An aperture of 0.5" is used.
    """
    scale = 0.049   # arcsec / pixel
    aperture = 0.5   # arcsec

    run_daophot(imgRoot, aperture / scale)

def run_daophot(imgRoot, aperture):
    """
    Inputs
    ---------
    imgRoot : string
        Name of a drizzled fits file
        
    aperture : float
        Aperture radius in pixels
    """
    drzImage = imgRoot + '.fits[1]'

    ir.digiphot()
    ir.daophot()
    ir.unlearn('daophot')

    # Setup parameters
    ir.datapars.sigma = 1
    ir.findpars.threshold = 50
    ir.photpars.aperture = aperture
    
    # Find the sources
    ir.daofind(drzImage, output=imgRoot+'.coo', interactive='no', verify='no')
    ir.phot(drzImage, coords=imgRoot+'.coo', output=imgRoot+'.mag',
            interactive='no', verify='no')

    return

def process_daophot_output(imgRoot):
    """
    Process a daophot output file into something more manageable.
    """
    def skip_bad_lines(self, str_vals, ncols):
        """Ignore bad lines when reading the DAOPHOT file."""
        return None

    reader = atpy.asciitables.asciitable.DaophotReader
    reader.inconsistent_handler = skip_bad_lines

    # Make a *.XYM file for the aperture photometry.
    stars = atpy.Table(imgRoot + '.mag', type='daophot', guess=False)
                       
    stars = stars.where(stars['FLUX'] > 0)
    mag = -2.5 * np.log10(stars['FLUX'])

    _out = open(imgRoot + '.xym', 'w')

    for ii in range(len(stars)):
        _out.write('%8.2f  %8.2f  %6.2f\n' % (stars['XCENTER'][ii], stars['YCENTER'][ii], mag[ii]))

    _out.close()

def matchup_lists(imgRoot, filt):
    """
    Run xym1mat on two starlists.
    """
    cmd = 'xym1mat '
    cmd += '"ks2_%s.xym(*,*,-99:-5)" ' % filt
    cmd += '"%s/%s.xym(*,*,-99:-5)" ' % (filt, imgRoot)
    cmd += '{0}_{1}_a.txt {0}_{1}_b.txt {0}_{1}_c.txt '.format(imgRoot, filt)
    cmd += '12 0.99'

    print cmd
    os.system(cmd)

def matchup_lists_brite(imgRoot, filt):
    """
    Run xym1mat on two starlists.
    """
    cmd = 'xym1mat '
    cmd += '"brite_%s.xym(*,*,-99:-5)" ' % filt
    cmd += '"%s/%s.xym(*,*,-99:-5)" ' % (filt, imgRoot)
    cmd += '{0}_{1}_a_brite.txt {0}_{1}_b_brite.txt {0}_{1}_c_brite.txt '.format(imgRoot, filt)
    cmd += '12 0.99'

    print cmd
    os.system(cmd)

def combine_matched_lists(filt, brite=False):
    finalTab = None

    for ii in range(len(images[filt])):
        inFile = images[filt][ii] + '_' + filt + '_a'
        if brite:
            inFile += '_brite'
        inFile += '.txt'

        starsTab = atpy.Table(inFile, type='ascii')
        img_id = np.ones(len(starsTab)) + ii + 1
        starsTab.add_column('img_id', img_id)

        # concatenate all the results
        if finalTab == None:
            finalTab = starsTab
        else:
            finalTab.append(starsTab)

    outFile = 'matched_ks2_apphot_' + filt
    if brite:
        outFile += '_brite'
    outFile += '.fits'

    finalTab.rename_column('col1', 'x_ks2')
    finalTab.rename_column('col2', 'y_ks2')
    finalTab.rename_column('col3', 'x_ap')
    finalTab.rename_column('col4', 'y_ap')
    finalTab.rename_column('col5', 'm_ks2')
    finalTab.rename_column('col6', 'm_ap')

    finalTab.table_name = ''
    finalTab.write(outFile, type='fits', overwrite=True)

    return


def plot_mag_offsets(matchup_file, mag_range=None):
    """
    Read in the results of an xym1mat run
    """
    if matchup_file.endswith('.txt'):
        type = 'ascii'
    else:
        type = 'fits'
        
    stars = atpy.Table(matchup_file, type=type)

    if type != 'fits':
        stars.rename_column('col1', 'x_ks2')
        stars.rename_column('col2', 'y_ks2')
        stars.rename_column('col3', 'x_ap')
        stars.rename_column('col4', 'y_ap')
        stars.rename_column('col5', 'm_ks2')
        stars.rename_column('col6', 'm_ap')
        
    stars.add_column('f_ks2', 10**(-stars.m_ks2/2.5))
    stars.add_column('f_ap', 10**(-stars.m_ap/2.5))

    if mag_range != None:
        stars = stars.where((stars.m_ks2 > mag_range[0]) & (stars.m_ks2 < mag_range[1]))

    dm = stars.m_ks2 - stars.m_ap
    df = stars.f_ks2 - stars.f_ap
    fluxRatio = stars.f_ap / stars.f_ks2

    # Find a list of stars with no other stars around it. Isolated.
    iso = []
    for ii in range(len(stars)):
        dx = stars.x_ks2 - stars.x_ks2[ii]
        dy = stars.y_ks2 - stars.y_ks2[ii]
        dr = np.hypot(dx, dy)
            
        # Use stars with nothing else within 5 pixels.
        idx = np.where((dr != 0) & (dr < 20))[0]
        if len(idx) == 0:
            iso.append(ii)

    iso = np.array(iso)
    print len(stars), len(iso)

    out_root = matchup_file.split('.')[0]
    out_file = out_root + '_stats.txt'

    stats = ZP_stats(fluxRatio)
    print stats
    stats.save_to_file(out_file, mode='w')
    
    stats_iso = ZP_stats(fluxRatio[iso], note='Isolated')
    print stats_iso
    stats_iso.save_to_file(out_file, mode='a')

    py.figure(4)
    py.clf()
    py.hist(fluxRatio, bins=np.arange(0, 3, 0.025), histtype='step')
    py.axvline(stats.mean_FR, linestyle='--', color='black')

    py.figure(1)
    py.clf()
    py.plot(stars.m_ks2, fluxRatio, 'k.', ms=2)
    py.axhline(stats.mean_FR, linestyle='--')
    py.ylim(0, 2)
    py.xlabel('KS2 magnitude')
    py.ylabel('Flux Ratio [APPHOT / KS2]')
    py.title('All Stars')
    py.savefig('plots/' + out_root + '_mag_vs_FR.png')

    py.figure(2)
    py.clf()
    py.plot(stars.m_ks2[iso], fluxRatio[iso], 'k.', ms=2)
    py.axhline(stats_iso.mean_FR, linestyle='--')
    py.ylim(0, 2)
    py.xlabel('KS2 magnitude')
    py.ylabel('Flux Ratio [APPHOT / KS2]')
    py.title('Isolated')
    py.savefig('plots/' + out_root + '_mag_vs_FR_iso.png')

    colors = ['red', 'blue', 'green', 'cyan']
    tmp = np.unique(stars.img_id)
    py.figure(3)
    py.clf()

    ZP_pos = np.zeros(len(tmp), dtype=float)
    ZP_err_pos = np.zeros(len(tmp), dtype=float)

    ZP_pos_iso = np.zeros(len(tmp), dtype=float)
    ZP_err_pos_iso = np.zeros(len(tmp), dtype=float)

    for ii in range(len(tmp)):
        idx = np.where(stars.img_id == tmp[ii])[0]
        idx_iso = np.where(stars.img_id[iso] == tmp[ii])[0]

        stats_ii = ZP_stats(fluxRatio[idx], note='Image {0}'.format(ii+1))
        print stats_ii
        stats_ii.save_to_file(out_file, mode='a')

        stats_ii_iso = ZP_stats(fluxRatio[iso][idx_iso], note='Image {0} Isolated'.format(ii+1))
        print stats_ii_iso
        stats_ii.save_to_file(out_file, mode='a')

        py.plot(stars.m_ks2[idx], fluxRatio[idx], 'k.', ms=2, color=colors[ii])
        py.axhline(stats_ii.mean_FR, linestyle='--', color=colors[ii])

        ZP_pos[ii] = stats_ii.ZP
        ZP_err_pos[ii] = stats_ii.ZP_err

        ZP_pos_iso[ii] = stats_ii_iso.ZP
        ZP_err_pos_iso[ii] = stats_ii_iso.ZP_err

    py.ylim(0, 2)
    py.xlabel('KS2 magnitude')
    py.ylabel('Flux Ratio [APPHOT / KS2]')
    py.title('All Stars in Each Image')
    py.savefig('plots/' + out_root + '_mag_vs_FR_image.png')

    # Final Value
    final = '*****\n'
    final += 'Final: m(DRZ) = m(KS2) + {0:6.3f} +/- {1:6.3f}\n'.format(ZP_pos.mean(), ZP_pos.std())
    final += '*****\n'

    final_iso = '*****\n'
    final_iso += 'Final Iso: m(DRZ) = m(KS2) + {0:6.3f} +/- {1:6.3f}\n'.format(ZP_pos_iso.mean(), ZP_pos_iso.std())
    final_iso += '*****\n'

    print final
    print final_iso

    _out = open(out_file, 'a')
    _out.write(final)
    _out.write(final_iso)
    _out.close()
    
class ZP_stats(object):
    def __init__(self, flux_ratio, note=''):
        self.median_FR = np.median(flux_ratio)
        m, s, n = statsIter.mean_std_clip(flux_ratio, clipsig=2.5, maxiter=10,
                                          converge_num=0.01, verbose=False,
                                          return_nclip=True)
        self.mean_FR = m
        self.std_FR = s
        self.n_stars = n
        self.mean_err_FR = self.std_FR / np.sqrt(self.n_stars)

        self.ZP = -2.5 * math.log10(self.mean_FR)
        self.ZP_err = 1.0857 * self.mean_err_FR / self.mean_FR

        self.note = note

        out =  ''
        out += 'Flux Ratio (APPHOT / KS2) {0}: \n'.format(note)
        out += '       N_stars: %6d\n' % self.n_stars
        out += '          Mean: %8.5f (iterative)\n' % self.mean_FR
        out += '   Err on Mean: %8.5f (iterative)\n' % self.mean_err_FR
        out += '\n'
        out += '        Median: %8.5f\n' % self.median_FR
        out += '        StdDev: %8.5f (iterative)\n' % self.std_FR
        out += '\n'
        out += 'Mean Mag Diff: {0:8.5f} +/- {1:8.5f}\n'.format(self.ZP, self.ZP_err)
        out += '\n'

        self.out = out

    def __str__(self):
        return self.out
            

    def save_to_file(self, outfile, mode='w'):
        _out = open(outfile, mode)
        _out.write(self.__str__())
        _out.close()
            
