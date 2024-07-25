import numpy as np
import pylab as plt
from jlu.ao.keck import kola
from astropy import table
from astropy import units as u
from astropy.io import fits
import scipy
from astropy.convolution import convolve_fft
import matplotlib.animation as animation

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


def plot_kola_gs_layout():
    from matplotlib.patches import Circle, Rectangle

    plt.figure(1, figsize=(7.5,7.5))
    plt.subplots_adjust(left=0.13, bottom=0.10, top=0.95, right=0.95)
    plt.clf()

    # Science Field of View
    sci_fov = Rectangle([-30, -30], 60, 60, fc='lightgrey', ec='none',
                        label='Science FOV')
    plt.gca().add_patch(sci_fov)

    # Notional Science Targets
    sci_x = scipy.stats.truncnorm.rvs(-30, 30, scale=10, size=50)
    sci_y = scipy.stats.truncnorm.rvs(-30, 30, scale=10, size=50)
    plt.plot(sci_x, sci_y, ls='none', marker='o',
             mec='none', mfc='firebrick', ms=4,
             label='Science Target')

    # TT Field of Regard
    tt_fov = Circle([0,0], 60, ec='steelblue', fc='none', lw=2,
                     label='NGS TT Field of Regard')
    plt.gca().add_patch(tt_fov)
    
    # LGS FOV
    lgs_fov = Circle([0,0], 30, fc='none', ec='limegreen', lw=1)
    plt.gca().add_patch(lgs_fov)

    # LGS Stars
    lgs_thetax = np.array([30.0, 21.21, 0.0, -21.21,
                           -30.0, -21.21, -0.0, 21.21])
    lgs_thetay = np.array([ 0.0, 21.21, 30.0, 21.21,
                            0.0, -21.21, -30.0, -21.21])
    plt.plot(lgs_thetax, lgs_thetay,
             ls='none', marker='*', ms=30, mfc='yellowgreen', mec='green',
             label="LGS Beacons")
    
    # NGS Stars
    ngs_thetax = np.array([15,   -30, 15])
    ngs_thetay = np.array([25.98,  0, -25.98])
    plt.plot(ngs_thetax, ngs_thetay,
             ls='none', marker='*', ms=15, mfc='steelblue', mec='mediumblue',
             label='TT Stars')

    plt.xlabel('(arcsec)')
    plt.ylabel('(arcsec)')
    plt.legend(fontsize=14, ncol=2, loc='upper center')
    plt.axis('equal')
    plt.ylim(-100, 100)
    plt.xlim(-70, 70)
    plt.show()
    plt.savefig('kola_asterism.png')

    return
    
    
def plot_microlens_event(psf_wave_idx=5):
    from bagle import model
    from matplotlib import patches
    from skimage.draw import polygon

    mL = 20.0 # msun
    t0 = 60000 # MJD
    beta = 7.0 # mas
    dL = 1000.0 # pc
    dL_dS = dL / 8000.0 # pc
    xS0_E = 0.0     # arcsec
    xS0_N = beta/1e3 # arcsec
    muL_E = 0.0 # mas/yr
    muL_N = 0.0 # mas/yr
    muS_E = 40.0 # mas/yr
    muS_N = 0.0 # mas/yr
    radiusS = 4.0 # mas
    b_sff = 1.0
    mag_src = 20.0
    n_outline = 30
    raL = 17.75 * 15.0
    decL = -29.0

    fspl = model.FSPL_PhotAstrom_noPar_Param1(mL, t0, beta, dL, dL_dS,
                                            xS0_E, xS0_N,
                                            muL_E, muL_N, muS_E, muS_N,
                                            radiusS,
                                            b_sff, mag_src, n_outline,
                                            raL=raL, decL=decL)

    # dt = 1
    # dt_max = 1
    dt = 20
    dt_max = 400
    t_obs = np.arange(60000-dt_max, 60001+dt_max, dt)
    xS_images = fspl.get_resolved_astrometry_outline(t_obs) * 1e3 # mas
    xS = fspl.get_astrometry_unlensed(t_obs) * 1e3 # mas
    xL = fspl.get_lens_astrometry(t_obs) * 1e3 # mas
    A = fspl.get_resolved_amplification(t_obs).T
    thetaE = fspl.thetaE_amp # mas

    psf_file = '/Users/jlu/work/ao/keck/maos/keck_maos/vismcao/A_keck_scao_lgs/'
    psf_file += 'evlpsfcl_1_x0_y0.fits'

    psf_fits = fits.open(psf_file)
    psf = psf_fits[psf_wave_idx].data
    psf_hdr = psf_fits[psf_wave_idx].header
    psf_scale = psf_hdr['DP'] * 1e3 # mas/pixel
    psf_wave = psf_hdr['wvl'] * 1e9 # nm

    # Make an empty image.
    img_scale = 2 # mas/pix sampling
    img_size = 100   # pixels (about 200 mas)
    rescale = psf_scale / img_scale
    print(f'image scale = {img_scale:.2f} mas/pix')
    print(f'orig psf scale = {psf_scale:.2f} mas/pix')
    print(f'rescale = {rescale:.2f}')
    print(f'image size = {img_size} pix')
    print(f'thetaE = {thetaE:.2f} mas')
    print(f'tE = {fspl.tE:.2f} days')
    
    img = np.zeros((len(t_obs), img_size, img_size), dtype=float)
    img_c = np.zeros((len(t_obs), img_size, img_size), dtype=float)

    # # Rescale the PSF, cutof first 2 pixels to recenter.
    psf = scipy.ndimage.zoom(psf, zoom=rescale, order=3)
    trim_ii = int(rescale)
    psf = psf[trim_ii:, trim_ii+1:-1]  # why is it different in x and y?

    for tt in range(len(t_obs)):
        # Make the image polygons
        poly_p_verts = np.append(xS_images[tt, :, 0, :],
                                 [xS_images[tt, 0, 0, :]], axis=0)
        poly_n_verts = np.append(xS_images[tt, :, 1, :],
                                 [xS_images[tt, 0, 1, :]], axis=0)
        poly_p_verts *= 1.0 / img_scale
        poly_n_verts *= 1.0 / img_scale
        poly_p_verts += img_size/2.0
        poly_n_verts += img_size/2.0
        
        rr_p, cc_p = polygon(poly_p_verts[:, 1], poly_p_verts[:, 0], img[tt].shape)
        rr_n, cc_n = polygon(poly_n_verts[:, 1], poly_n_verts[:, 0], img[tt].shape)
        img[tt, rr_p, cc_p] = A[tt, 0] / len(rr_p)
        img[tt, rr_n, cc_n] = A[tt, 1] / len(rr_n)
        print(f'image {tt} flux after = {img[tt].sum()}, pix_cnt = {len(rr_p)}')

        #Convolve our image with the PSF.
        img_c[tt, :, :] = convolve_fft(img[tt, :, :], psf, boundary='wrap')

    
    ##########
    # Plot
    ##########
    ##
    ## Plot schematic.
    ##
    plt.close('all')
    fig1 = plt.figure(1)
    plt.clf()

    plt.plot([0], [0], 'k.')

    # Plot Enstein radius in mas and the black hole.
    f1_thetaE = plt.gca().add_patch(patches.Circle(xL[0], thetaE,
                                       ec='purple', fc='none'))
    f1_xL = plt.gca().add_patch(patches.Circle(xL[0], thetaE/20.0,
                                       ec='none', fc='black'))
    f1_xS = plt.gca().add_patch(patches.Circle(xS[0], thetaE/20.0,
                                       ec='red', fc='none'))

    f1_img_p = patches.Polygon(xS_images[0, :, 0, :], fc='orange', ec='darkorange')
    f1_img_n = patches.Polygon(xS_images[0, :, 1, :], fc='orange', ec='darkorange')
    plt.gca().add_patch(f1_img_p)
    plt.gca().add_patch(f1_img_n)

    plt.axis('equal')
    lim = thetaE*2
    plt.xlim(-lim, lim)
    plt.ylim(-lim, lim)
    plt.xlabel('(mas)')
    plt.ylabel('(mas)')
    plt.text(0.05, 0.95, f'M$_{{BH}}$ = {mL:.0f} M$_\odot$', color='black',
             transform = fig1.gca().transAxes, fontsize=14)
    plt.text(0.05, 0.90, f't$_{{E}}$ = {fspl.tE:.0f} days', color='black',
             transform = fig1.gca().transAxes, fontsize=14)
    plt.text(0.05, 0.85, f'$\\theta_{{E}}$ = {thetaE:.0f} mas', color='black',
             transform = fig1.gca().transAxes, fontsize=14)
    plt.text(0.05, 0.80, f'$\lambda_{{PSF}}$ = {psf_hdr["wvl"]*1e3:.0f} nm', color='black',
             transform = fig1.gca().transAxes, fontsize=14)
    

    ##
    ## Plot the intrinsic image.
    ##
    fig2 = plt.figure(2)
    plt.clf()
    img_ext = np.array([-img[0].shape[0], img[0].shape[0]]) * img_scale / 2.0
    img_ext = np.append(img_ext, img_ext)
    f2_img = plt.imshow(img[0, :, :], cmap='binary_r', extent=img_ext)
    
    # Plot Enstein radius in mas, the black hole, and the true source position.
    f2_thetaE = plt.gca().add_patch(patches.Circle(xL[0], thetaE,
                                       ec='cyan', fc='none', ls='--'))
    f2_xL = plt.gca().add_patch(patches.Circle(xL[0], thetaE/20.0,
                                       ec='lightgrey', fc='black'))
    f2_xS = plt.gca().add_patch(patches.Circle(xS[0], thetaE/20.0,
                                       ec='yellow', fc='yellow'))
    plt.xlabel('(mas)')
    plt.ylabel('(mas)')
    plt.title('Intrinsic Flux')

    ##
    ## Plot the PSF
    ##
    fig3 = plt.figure(3)
    plt.clf()
    psf_ext = np.array([-psf.shape[0], psf.shape[0]]) * img_scale / 2.0
    psf_ext = np.append(psf_ext, psf_ext)
    plt.imshow(psf, cmap='binary_r', extent=psf_ext)
    plt.xlim(img_ext[0], img_ext[1])
    plt.ylim(img_ext[0], img_ext[1])
    plt.xlabel('(mas)')
    plt.ylabel('(mas)')
    plt.title('PSF')
    
    ##
    ## Plot the convolved image.
    ##
    fig4 = plt.figure(4)
    plt.clf()
    f4_img = plt.imshow(img_c[0, :, :], cmap='binary_r', extent=img_ext)
    
    # Plot Enstein radius in mas, the black hole, and the true source position.
    f4_thetaE = plt.gca().add_patch(patches.Circle(xL[0], thetaE,
                                       ec='cyan', fc='none', ls='--'))
    f4_xL = plt.gca().add_patch(patches.Circle(xL[0], thetaE/20.0,
                                       ec='lightgrey', fc='black'))
    f4_xS = plt.gca().add_patch(patches.Circle(xS[0], thetaE/20.0,
                                       ec='yellow', fc='yellow'))
    plt.xlabel('(mas)')
    plt.ylabel('(mas)')
    plt.title('Observed Flux')
    plt.text(0.05, 0.95, f'M$_{{BH}}$ = {mL:.0f} M$_\odot$', color='white',
             transform = fig4.gca().transAxes, fontsize=14)
    plt.text(0.05, 0.90, f't$_{{E}}$ = {fspl.tE:.0f} days', color='white',
             transform = fig4.gca().transAxes, fontsize=14)
    plt.text(0.05, 0.85, f'$\\theta_{{E}}$ = {thetaE:.0f} mas', color='white',
             transform = fig4.gca().transAxes, fontsize=14)
    plt.text(0.05, 0.80, f'$\lambda_{{PSF}}$ = {psf_hdr["wvl"]*1e6:.0f} nm', color='white',
             transform = fig4.gca().transAxes, fontsize=14)

    ##
    ## Plot 2 panel
    ##
    fig5 = plt.figure(5, figsize=(12,6))
    plt.clf()
    f5a1 = plt.subplot(1, 2, 1)
    f5a2 = plt.subplot(1, 2, 2)
    plt.subplots_adjust(left=0.1, right=0.95)

    # Left Panel - Schematic
    f5a1.plot([0], [0], 'k.')

    # Plot Enstein radius in mas and the black hole.
    f5a1_thetaE = f5a1.add_patch(patches.Circle(xL[0], thetaE,
                                       ec='purple', fc='none'))
    f5a1_xL = f5a1.add_patch(patches.Circle(xL[0], thetaE/20.0,
                                       ec='none', fc='black'))
    f5a1_xS = f5a1.add_patch(patches.Circle(xS[0], thetaE/20.0,
                                       ec='red', fc='none'))

    f5a1_img_p = patches.Polygon(xS_images[0, :, 0, :], fc='orange', ec='darkorange')
    f5a1_img_n = patches.Polygon(xS_images[0, :, 1, :], fc='orange', ec='darkorange')
    f5a1.add_patch(f5a1_img_p)
    f5a1.add_patch(f5a1_img_n)

    lim = thetaE*2
    f5a1.set_xlabel('(mas)')
    f5a1.set_ylabel('(mas)')
    f5a1.set_title('Schematic')
    f5a1.text(0.05, 0.95, f'M$_{{BH}}$ = {mL:.0f} M$_\odot$', color='black',
             transform = f5a1.transAxes, fontsize=14)
    f5a1.text(0.05, 0.90, f't$_{{E}}$ = {fspl.tE:.0f} days', color='black',
             transform = f5a1.transAxes, fontsize=14)
    f5a1.text(0.05, 0.85, f'$\\theta_{{E}}$ = {thetaE:.0f} mas', color='black',
             transform = f5a1.transAxes, fontsize=14)
    f5a1.text(0.05, 0.80, f'$\lambda_{{PSF}}$ = {psf_hdr["wvl"]*1e3:.0f} nm',
              color='black',
              transform = f5a1.transAxes, fontsize=14)

                        

    # Right Panel
    f5a2_img = f5a2.imshow(img_c[0, :, :], cmap='binary_r', extent=img_ext)
    
    # Plot Enstein radius in mas, the black hole, and the true source position.
    f5a2_thetaE = f5a2.add_patch(patches.Circle(xL[0], thetaE,
                                       ec='cyan', fc='none', ls='--'))
    f5a2_xL = f5a2.add_patch(patches.Circle(xL[0], thetaE/20.0,
                                       ec='lightgrey', fc='black'))
    f5a2_xS = f5a2.add_patch(patches.Circle(xS[0], thetaE/20.0,
                                       ec='yellow', fc='yellow'))
    f5a2.set_xlabel('(mas)')
    f5a2.set_title('Observed')

    f5a1.axis('equal')
    f5a1.set_xlim(img_ext[0], img_ext[1])
    f5a1.set_ylim(img_ext[0], img_ext[1])
    f5a2.axis('equal')
    f5a2.set_xlim(img_ext[0], img_ext[1])
    f5a2.set_ylim(img_ext[0], img_ext[1])
    
    
    ##########
    # Animate
    ##########
    
    plt_objs1 = [f1_thetaE, f1_xL, f1_xS, f1_img_p, f1_img_n]
    plt_objs2 = [f2_thetaE, f2_xL, f2_xS, f2_img]
    plt_objs4 = [f4_thetaE, f4_xL, f4_xS, f4_img]
    plt_objs5 = [f5a1_thetaE, f5a1_xL, f5a1_xS, f5a1_img_p, f5a1_img_n,
                 f5a2_thetaE, f5a2_xL, f5a2_xS, f5a2_img]

    
    def f1_update(t, xL, xS, p_outline, n_outline, plt_objs1):
        f1_thetaE, f1_xL, f1_xS, f1_img_p, f1_img_n = plt_objs1
        f1_thetaE.center = xL[t]
        f1_xL.center = xL[t]
        f1_xS.center = xS[t]
        f1_img_p.xy = p_outline[t]
        f1_img_n.xy = n_outline[t]
        return plt_objs1
        
    def f2_update(t, xL, xS, img, plt_objs2):
        f2_thetaE, f2_xL, f2_xS, f2_img = plt_objs2
        f2_thetaE.center = xL[t]
        f2_xL.center = xL[t]
        f2_xS.center = xS[t]
        f2_img.set_array(img[t])
        return plt_objs2

    def f4_update(t, xL, xS, img_c, plt_objs4):
        f4_thetaE, f4_xL, f4_xS, f4_img = plt_objs4
        f4_thetaE.center = xL[t]
        f4_xL.center = xL[t]
        f4_xS.center = xS[t]
        f4_img.set_array(img_c[t])
        return plt_objs4

    def f5_update(t, xL, xS, p_outline, n_outline, img_c, plt_objs5):
        f5a1_thetaE, f5a1_xL, f5a1_xS, f5a1_img_p, f5a1_img_n, f5a2_thetaE, f5a2_xL, f5a2_xS, f5a2_img = plt_objs5
        
        f5a1_thetaE.center = xL[t]
        f5a1_xL.center = xL[t]
        f5a1_xS.center = xS[t]
        f5a1_img_p.xy = p_outline[t]
        f5a1_img_n.xy = n_outline[t]
        
        f5a2_thetaE.center = xL[t]
        f5a2_xL.center = xL[t]
        f5a2_xS.center = xS[t]
        f5a2_img.set_array(img_c[t])
        
        return plt_objs5
    
    p_outline = xS_images[:, :, 0, :]
    n_outline = xS_images[:, :, 1, :]
    frame_time = 100 # ms
    
    ani1 = animation.FuncAnimation(fig1, f1_update, len(t_obs),
                                  fargs=[xL, xS, p_outline, n_outline, plt_objs1],
                                  blit=True, interval=frame_time)
    ani1.save(f'fspl_schematic_{psf_wave:04.0f}nm.gif')
    

    ani2 = animation.FuncAnimation(fig2, f2_update, len(t_obs),
                                  fargs=[xL, xS, img, plt_objs2],
                                  blit=True, interval=frame_time)
    ani2.save(f'fspl_image_raw_{psf_wave:04.0f}nm.gif')

    
    ani4 = animation.FuncAnimation(fig4, f4_update, len(t_obs),
                                  fargs=[xL, xS, img_c, plt_objs4],
                                  blit=True, interval=frame_time)
    ani4.save(f'fspl_image_conv_{psf_wave:04.0f}nm.gif')

    ani5 = animation.FuncAnimation(fig5, f5_update, len(t_obs),
                                  fargs=[xL, xS, p_outline, n_outline, img_c, plt_objs5],
                                  blit=True, interval=frame_time)
    ani5.save(f'fspl_2panel_{psf_wave:04.0f}nm.gif')
    
    return fspl

    
