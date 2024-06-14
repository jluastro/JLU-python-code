import numpy as np
import pylab as plt
from jlu.ao.keck import kola
from astropy import table
from astropy import units as u

outdir = '/Users/jlu/doc/papers/2024_spie_kola/figures/'

# Dictionary of angular resolution vs. wavelength
ang_res = {}

#JWST
# wave, ang res (")
ang_res['jwst'] = np.array([[0.6150882191637174, 0.06096813379546762],
                            [0.6847258746571921, 0.05034995600224686],
                            [0.8260960920542535, 0.04298808757681049],
                            [0.9446147673668059, 0.039556334000279865],
                            [1.0947156078110876, 0.043709369599795066],
                            [1.2857907753486013, 0.05119476022987374],
                            [1.4901050447023707, 0.05897257152690221],
                            [1.7384999098106393, 0.06907184173786375],
                            [2.055676997540564, 0.07956564517718195],
                            [2.366411392455907, 0.09475518351689512],
                            [2.7424380555099965, 0.11098235790405625],
                            [7.596970951421151, 0.30370520341827834],
                            [18.652728035375677, 0.7458968988201465],
                            [23.42736452991857, 0.9337603823024639]])

ang_res['keck'] = np.array([[0.40321000092027387, 0.8106080105786589],
                            [0.8542531884302459, 0.8106080105786589],
                            [1.294438775134021, 0.803891954595756],
                            [1.421799731728458, 0.7458968988201465],
                            [1.4801498205895072, 0.6159918911705092],
                            [1.5102168144631545, 0.4271631697831025],
                            [1.5616918435820313, 0.2794603142689441],
                            [1.703888014593333, 0.21955184093015348],
                            [1.7619642353892697, 0.15480404562189568],
                            [2.111541925227522, 0.05517526072484411],
                            [3.094130120123976, 0.08023037048544114],
                            [3.965092060495943, 0.10212259872103778]])

ang_res['eelt'] = np.array([[0.3951824697000148, 0.6529312546451438],
                            [0.4996776867506684, 0.6529312546451438],
                            [0.6150882191637174, 0.6475215833836249],
                            [0.7321984766002337, 0.5621208529099093],
                            [0.8372458123123961, 0.45277898683811474],
                            [0.9013143251904917, 0.364705934428315],
                            [0.9899954253349506, 0.2508120665307376],
                            [1.058632613575993, 0.16823424209107787],
                            [1.1244655190769715, 0.10470356878387858],
                            [1.1706131314974373, 0.06516412580350552],
                            [1.226851077344493, 0.039886804472220184],
                            [1.2857907753486013, 0.02322579889092003],
                            [1.3296163485828525, 0.010363092894532034],
                            [1.5306000303775826, 0.010536971590437956],
                            [1.7501927506174333, 0.012037246457952379],
                            [2.1400411543197233, 0.014697523656984388],
                            [2.9921442906624742, 0.020672154189709024],
                            [3.708012534833107, 0.02461858707452773]])

ang_res['hst'] = np.array([[0.10910526437053988, 0.09960525023807865],
                           [0.1334080236725459, 0.09014128052346396],
                           [0.17678796344187966, 0.08504156826181883],
                           [0.22204130442474534, 0.07956564517718195],
                           [0.2678845077261668, 0.0744423229939835],
                           [0.3951824697000148, 0.07200573388253859],
                           [0.5908403682054318, 0.06736920296650838],
                           [1.023738953322875, 0.08791927463427197],
                           [1.1169530914871202, 0.11961147011371337],
                           [1.4604384572156706, 0.1560973445917794],
                           [2.168925034102089, 0.23466199294798376]])

labels = {}
labels['jwst'] = 'JWST'
labels['keck'] = 'Keck'
labels['eelt'] = 'E-ELT'
labels['hst'] = 'HST'

labels_ang_res_pos = {}
labels_ang_res_pos['jwst'] = [2.0, 0.1]
labels_ang_res_pos['keck'] = [0.6, 0.86]
labels_ang_res_pos['eelt'] = [0.38, 0.48]
labels_ang_res_pos['hst']  = [0.38, 0.08]

labels_ang_res_rot = {}
labels_ang_res_rot['jwst'] = 25
labels_ang_res_rot['keck'] = 0
labels_ang_res_rot['eelt'] = 0
labels_ang_res_rot['hst']  = 0


colors = {}
colors['jwst'] = 'orange'
colors['keck'] = 'red'
colors['eelt'] = 'seagreen'
colors['hst']  = 'steelblue'

def plot_angres_sensit_wave():
    plt.figure(1)
    plt.clf()
    for key in ang_res:
        plt.loglog(ang_res[key][:,0], ang_res[key][:,1], label=key,
                   color=colors[key], lw=3)
        plt.text(labels_ang_res_pos[key][0], labels_ang_res_pos[key][1],
                 labels[key], rotation=labels_ang_res_rot[key],
                 color=colors[key])

    plt.xlim(0.35, 3)
    plt.ylim(0.01, 1.2)
    plt.xlabel('Wavelength ($\mu$m)')
    plt.ylabel('Angular Resolution (")')

    # Plot the diffraction-limited angular resolution of Keck vs. wavelength.
    wave = np.logspace(-1, 1, 100) # micron
    keck_diam = 10 # m
    res = 0.25 * wave / keck_diam  # arcsec

    plt.plot(wave, res, 'k--')
    plt.text(1.5, 0.03, 'Keck DL', color='black', rotation=25)

    # Make the box for KOLA.
    from matplotlib.patches import Polygon
    kola_poly_coords = np.array([[0.35, 0.015], [0.6, 0.015],
                                 [2.3, 0.057], [0.35, 0.025]])
    
    kola_poly = Polygon(kola_poly_coords, facecolor='firebrick')
    plt.gca().add_patch(kola_poly)

    plt.text(0.4, 0.018, 'KOLA', color='white',
             rotation=0, weight='bold', size=24)

    # Pretty
    import matplotlib.ticker as mticker
    plt.gca().xaxis.set_major_formatter(mticker.ScalarFormatter())
    plt.gca().xaxis.set_minor_formatter(mticker.ScalarFormatter())
    plt.gca().yaxis.set_major_formatter(mticker.ScalarFormatter())
    plt.gca().tick_params(axis="x", which="both", rotation=90)
    plt.subplots_adjust(top=0.95, bottom=0.2)

    plt.savefig(outdir + 'kola_angular_res_compare.png')

    return

def plot_kebs_metrics():
    kebs_out = '/Users/jlu/Google Drive/My Drive/instruments/'
    kebs_out += 'Keck/KOLA Keck Optical LGS AO/Systems Engineering/'
    kebs_out += 'Performance Models/kola_sims_ebtool/'

    tab = table.Table.read(kebs_out + 'kola_grid_2024_05_30.fits')

    # Plot 1
    wfs_rate = 1600 # Hz
    lgs_pow = 50    # W
    filt = "r'"
    rad_ee = 50
    axis1 = 'act_cnt'
    axis2 = 'lgs_cnt'

    kola.plot_metrics3_any_pair(table=tab, axis1=axis1, axis2=axis2,
                                wfs_rate=wfs_rate, lgs_pow=lgs_pow,
                                filter=filt, r_ensqE=rad_ee, 
                                interpolate=True, contour_levels=[0.05, 0.10])
    plt.savefig(outdir + f'kebs_actcnt_lgscnt_wfs{wfs_rate}_lgspow{lgs_pow}_r.png')

    # Plot 2
    wfs_rate = 1600   # Hz
    lgs_pow_tot = 300 # W
    filt = "r'"
    rad_ee = 50
    axis1 = 'act_cnt'
    axis2 = 'lgs_cnt'
    kola.plot_metrics3_any_pair(table=tab, axis1=axis1, axis2=axis2,
                                wfs_rate=wfs_rate, lgs_pow_tot=lgs_pow_tot,
                                filter=filt, r_ensqE=rad_ee, 
                                interpolate=True, contour_levels=[0.05, 0.10])
    plt.savefig(outdir + f'kebs_actcnt_lgscnt_wfs{wfs_rate}_lgspowtot{lgs_pow_tot}_r.png')

    # Plot 2
    lgs_pow = 50 # W
    lgs_cnt = 8
    filt = "r'"
    rad_ee = 50
    axis1 = 'act_cnt'
    axis2 = 'wfs_rate'    
    kola.plot_metrics3_any_pair(table=tab, axis1=axis1, axis2=axis2,
                                lgs_cnt=lgs_cnt, lgs_pow=lgs_pow,
                                filter=filt, r_ensqE=rad_ee, 
                                interpolate=True, contour_levels=[0.05, 0.10])
    plt.savefig(outdir + f'kebs_actcnt_wfsrate_lgscnt{lgs_cnt}_lgspow{lgs_pow}_r.png')
    

    return

def planet_resolutions():
    # Data source is https://www.jpl.nasa.gov/edu/pdfs/scaless_reference.pdf
    diam_neptune = 49528. * u.km
    diam_uranus  = 51118. * u.km
    diam_mars    =  6792. * u.km
    diam_venus   = 12104. * u.km

    # Note these are distances from sun.
    dist_neptune = 4495100000 * u.km
    dist_uranus  = 2872500000 * u.km
    dist_mars    =  227900000 * u.km
    dist_venus   =  108200000 * u.km

    # approximate, since distance from sun.
    # scale is 1 km = YYY mas
    scale_neptune = (u.km / dist_neptune).to(u.mas, equivalencies=u.dimensionless_angles())
    scale_uranus = (u.km / dist_uranus).to(u.mas, equivalencies=u.dimensionless_angles())
    scale_mars = (u.km / dist_mars).to(u.mas, equivalencies=u.dimensionless_angles())
    scale_venus = (u.km / dist_venus).to(u.mas, equivalencies=u.dimensionless_angles())

    # Desired spatial resolutions
    # Requirements laid out in VIPER report.
    # https://docs.google.com/document/d/1pujkxTtxGYrlgPNSKKD3ON4ZKsfp6CORlx2iVpIf188/edit
    res_req_neptune = 500. # km
    res_req_uranus  = 500.
    res_req_mars    = 50.
    res_req_venus   = 50.

    ares_neptune = res_req_neptune * scale_neptune # mas
    ares_uranus = res_req_uranus * scale_uranus
    ares_mars = res_req_mars * scale_mars
    ares_venus = res_req_venus * scale_venus

    print(f'Neptune: Required resolution of {res_req_neptune:.0f} km is {ares_neptune:.3f}')
    print(f'Uranus:  Required resolution of {res_req_uranus:.0f} km is {ares_uranus:.3f}')
    print(f'Mars:    Required resolution of {res_req_mars:.0f} km is {ares_mars:.3f}')
    print(f'Venus:   Required resolution of {res_req_venus:.0f} km is {ares_venus:.3f}')

    return

    
    
