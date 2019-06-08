import numpy as np
import pylab as plt
import math
from astropy.io import fits
from os import environ, path
import pdb
from scipy import signal
from scipy import ndimage
from matplotlib import colors

test_dir = '/u/jlu/work/ao/airopa/test_defocus/'

def defocus_keck_ao(focus_coeff, wavelength=2.12, tel_diam=10.0, f_num=15):
    """
    Parameters
    ---------------
    focus_coeff : float (nm)
        Coefficient of the defocs term assuming that the optical path difference
        from defocus is given by:
            c * sqrt(3) * (2 \rho^2  - 1)

    Optional Parameters
    ---------------
    wavelength : float (microns)
    tel_diam : float (m)
    f_num : float
    """
    pix_scale = 206265. / (tel_diam * f_num)   # arcsec / m
    pix_scale *= 1.0e3 / 1e6   # mas / micron
    print("Pixel Scale: {0:.2f} mas / micron".format(pix_scale))

    fwhm_z0_asec = 0.21 * (wavelength / tel_diam)  # arcsec
    fwhm_z0_micron = 1.02 * wavelength * f_num     # microns
    str_fmt = "FWHM at z=0 mm: {0:.1f} mas = {1:.1f} micron"
    print(str_fmt.format(fwhm_z0_asec*1e3, fwhm_z0_micron))

    # Handy quantities:
    sqrt_2ln2 = math.sqrt(2.0 * math.log(2.0))
    
    # Gaussian Beam Approximation:
    # The beam radius at focus:
    w0 = fwhm_z0_micron / sqrt_2ln2  # micron
    print("Beam Radius at z=0 mm: w0 = {0:.1f} micron".format(w0))

    # Delta-focus length set by rayliegh criteria
    z_R = math.pi * w0**2 * 1e-3 / wavelength  # mm
    print("Focus shift at RC: z_R = {0:.2f} mm".format(z_R))
    print()
    
    # Peak-to-valley of defocused wavefront
    p2v = 2.0 * math.sqrt(3) * focus_coeff  # nm
    print("Peak-to-valley for c = {0:.0f}: {1:.1f} nm".format(focus_coeff, p2v))

    # The focal shift distance (along z)
    z = 8.0 * p2v * f_num**2   # nm
    z *= 1e-6 # mm
    print("Focus length for   c = {0:.0f}: {1:.2f} mm".format(focus_coeff, z))

    # Defocused beam radius
    w = w0 * math.sqrt(1.0 + (z / z_R)**2)   # micron
    print("Beam radius for    c = {0:.0f}: {1:.2f} micron".format(focus_coeff, w))

    # Defocused beam FWHM
    fwhm_z_micron = w * sqrt_2ln2  # micron
    fwhm_z_mas = fwhm_z_micron * pix_scale     # mas
    str_fmt = "Beam FWHM for      c = {0:.0f}: {1:.2f} micron = {2:.2f} mas"    
    print(str_fmt.format(focus_coeff, fwhm_z_micron, fwhm_z_mas))

    print("Fractional Change in Beam FWHM: {0:.2f}".format(fwhm_z_micron / fwhm_z0_micron))

    return


def make_defocus_phase_map(rms_wfe, circular_pupil=False):
    # We will use some files from AIROPA
    dir_airopa = environ['AIROPA_DATA_PATH']
    
    # Read in the default NIRC2 pupil.
    pupil_file = path.join(dir_airopa, 'phase_maps', 'defocus', 'pupil.fits')
    pupil = fits.getdata(pupil_file)

    clear_idx = np.where(pupil == 1)

    # Calculate the pupil plane coordinates (normalized over the clear aperture).
    pupil_u0 = np.median(clear_idx[0])
    pupil_v0 = np.median(clear_idx[1])
    pupil_u = np.arange(pupil.shape[0], dtype=float) - pupil_u0  # note this isn't quite right; but 
    pupil_v = np.arange(pupil.shape[1], dtype=float) - pupil_v0  # it appears to be the convention Gunther used.
    pupil_u_2d, pupil_v_2d = np.meshgrid(pupil_u, pupil_v, indexing='ij')
    pupil_rho_2d = np.hypot(1.0 * pupil_u_2d, pupil_v_2d)
    rho_max_clear = pupil_rho_2d[clear_idx].max()

    if circular_pupil:
        pupil[:,:] = 0
        idx = np.where(pupil_rho_2d <= rho_max_clear)
        pupil[idx] = 1

    pupil_u /= rho_max_clear
    pupil_v /= rho_max_clear
    pupil_u_2d /= rho_max_clear
    pupil_v_2d /= rho_max_clear
    pupil_rho_2d /= rho_max_clear
    clear_idx = np.where(pupil == 1)

    # Plot the clear pupil.
    plt.figure(1, figsize=(8,6))
    plt.clf()
    plt.subplots_adjust(right=0.9)
    plt.imshow(pupil, extent=[pupil_u[0], pupil_u[-1], pupil_v[0], pupil_v[-1]])
    plt.colorbar()
    plt.axis('equal')
    plt.xlabel('Pupil u')
    plt.ylabel('Pupil v')
    plt.title('Pupil')

    # Add a defocus term
    phase_map = rms_wfe * math.sqrt(3.0) * (2.0 * pupil_rho_2d**2 - 1.0)
    print('RMS WFE before pupil applied: {0:.1f} nm'.format(phase_map[clear_idx].std()))
    phase_map *= pupil
    print('RMS WFE after pupil applied: {0:.1f} nm'.format(phase_map[clear_idx].std()))
    rms = np.sqrt((phase_map**2).sum()) / np.size(phase_map[clear_idx])**0.5
    print('RMS WFE after pupil applied: {0:.1f} nm'.format( rms))

    # Plot the phase map
    plt.figure(2, figsize=(8, 6))
    plt.clf()
    plt.subplots_adjust(right=0.9)
    plt.imshow(phase_map, extent=[pupil_u[0], pupil_u[-1], pupil_v[0], pupil_v[-1]])
    plt.colorbar(label='RMS WFE (nm)')
    plt.axis('equal')
    plt.xlabel('Pupil u')
    plt.ylabel('Pupil v')
    plt.title('Defocus: {0:.0f} nm'.format(rms_wfe))

    out_file = test_dir + 'phase_map_defocus_{0:0.0f}'.format(rms_wfe)
    if circular_pupil:
        out_file += '_circ'
    out_file += '.fits'
    hdu_phase = fits.PrimaryHDU(phase_map)
    hdu_amp = fits.ImageHDU(pupil)
    hdu_u_2d = fits.ImageHDU(pupil_u_2d)
    hdu_v_2d = fits.ImageHDU(pupil_v_2d)

    hdu_list = fits.HDUList([hdu_phase, hdu_amp, hdu_u_2d, hdu_v_2d])
    
    hdu_list.writeto(out_file, overwrite=True)
    
    return

    

def test_otf_to_psf(rms_wfe, wavelength=2120.0, pad_factor=1, circular_pupil=False):
    """
    wavelength : float
        Wavelength in nm.
    """
    in_file = test_dir + 'phase_map_defocus_{0:0.0f}'.format(rms_wfe)
    if circular_pupil:
        in_file += '_circ'
    in_file += '.fits'
    hdu_list = fits.open(in_file)

    phase_map = hdu_list[0].data # nm
    amp_map = hdu_list[1].data   # unitless
    phase_map_u2d = hdu_list[2].data # normalized to 1 at full aperture radius.
    phase_map_v2d = hdu_list[3].data
    print('Shape of phase map: ', phase_map.shape)
    print('Shape of amplitude map: ', amp_map.shape)

    # We must pad the pupil image with 0s so that
    # it doesn't ring in our FFTs. We will pad it to be
    # 3 times as big as what comes in.
    print('Padding')
    pad_phase = np.pad(phase_map, phase_map.shape[0]*pad_factor, 'constant')
    pad_ampli = np.pad(amp_map, phase_map.shape[0]*pad_factor, 'constant')
    print('Shape of new phase map: ', pad_phase.shape)
    print('Shape of new amplitude map: ', pad_ampli.shape)

    # Convert the phase to OPD in radians and make the complex aperture
    opd = (2.0 * math.pi / wavelength) * pad_phase
    complex_aperture = pad_ampli * np.exp(1.j * opd)
    print('Shape of complex aperture: ', complex_aperture.shape)

    # # Compute the OTF
    # otf = signal.correlate2d(complex_aperture, complex_aperture, mode='same')
    # mtf = otf * np.conj(otf)

    # Compute the PSF
    print('ifftshift')
    cap_origin_corner = np.fft.ifftshift(complex_aperture)
    print('fft2')
    asf_origin_corner = np.fft.fft2(cap_origin_corner)
    print('fftshift')
    asf = np.fft.fftshift(asf_origin_corner)
    print('calc PSF')
    psf = np.abs(asf)
    psf /= psf.sum()
    print('calc MTF')
    mtf = np.abs(np.fft.fft2(psf))

    # Calculate the Strehl
    idx = np.where(pad_ampli > 0)
    print(np.size(idx[0]))
    strehl = np.abs((complex_aperture[idx].sum() / np.size(idx[0])))**2
    print('Strehl: {0:.2f}'.format(strehl))
    print('Peak Pixel Value: ', psf.max())

    # Unpad the PSF:
    # For helpful display, calculate the PSF center
    psf_center = np.array(psf.shape) / 2.0
    psf_pad = int(phase_map.shape[0] * pad_factor / 2.0)
    print('psf_pad: ', psf_pad)
    psf = psf[psf_pad:-psf_pad,psf_pad:-psf_pad]
             
    psf_center = np.array(psf.shape) / 2.0
    
    psf_file = test_dir + 'psf_defocus_{0:.0f}'.format(rms_wfe)
    if circular_pupil:
        psf_file += '_circ'
    psf_file += '.fits'
    
    fits.writeto(psf_file, psf, overwrite=True)

    plot_psf(rms_wfe)
    
    return


def plot_psf(rms_wfe):    
    psf_file = test_dir + 'psf_defocus_{0:.0f}.fits'.format(rms_wfe)
    psf = fits.getdata(psf_file)
    
    psf_center = np.array(psf.shape) / 2.0
    
    plt.figure(1)
    plt.clf()
    plt.imshow(psf)
    plt.axis('equal')
    plt.xlim(psf_center[0] - 50, psf_center[0] + 50)
    plt.ylim(psf_center[1] - 50, psf_center[1] + 50)

    # plt.figure(2)
    # plt.clf()
    # norm = colors.LogNorm(vmin=mtf.max()*1e-9, vmax=mtf.max())
    # plt.imshow(mtf, norm=norm)

    # Plot the radial profile
    rprof = radial_profile(psf)
    plt.figure(3)
    # plt.clf()
    plt.plot(rprof)
    plt.xlim(0, 50)

    # Calculate the FWHM empirically:
    print('FWHM = {0:.1f} pix'.format(calc_fwhm(psf)))

    # pdb.set_trace()
    return
    
def radial_profile(data):
    # center = np.array(data.shape) / 2.0
    max_pix = data.max()
    center = np.where(data == max_pix)
    print(center)
    center = np.array([center[0][0], center[1][0]])
    print('center: ', center)
    
    x, y = np.indices((data.shape))
    r = np.sqrt((x - center[0])**2 + (y - center[1])**2)
    r = r.astype(np.int)

    tbin = np.bincount(r.ravel(), data.ravel())
    nr = np.bincount(r.ravel())
    radialprofile = tbin / nr

    # Normalize.
    # radial_profile /= radial_profile.sum()
    
    return radialprofile

def calc_fwhm(data):
    # center = np.array(data.shape) / 2.0
    # print('old center: ', center)
    
    max_pix = data.max()
    center = np.where(data == max_pix)
    print(center)
    center = np.array([center[0][0], center[1][0]])
    print('center: ', center)
    
    x, y = np.indices((data.shape))
    r = np.sqrt((x - center[0])**2 + (y - center[1])**2)

    idx = np.where(data >= (max_pix / 2.0))
    fwhm = r[idx].max()
    # print('FWHM 1: ', fwhm)

    # # Now chop down the box to 10* FWHM and recalc.
    # idx = np.where((data >= (max_pix / 2.0)) & (r < 10*fwhm))
    # fwhm = r[idx].max()
    # print('FWHM 2: ', fwhm)

    return fwhm
    
