import numpy as np
import pylab as plt
from astropy.io import fits
from astropy import stats

def stripes_sky():
    
    ### First investigate variance in sky files. 
    sky_files  = ['/g/lu/data/microlens/20may25os/raw/i200525_a041{0:03d}_xflip.fits'.format(ii) for ii in range(2, 4+1)]

    sky_images = np.zeros((2048, 2048, len(sky_files)), dtype=float)

    for ss in range(len(sky_files)):
        img = fits.getdata(sky_files[ss])

        sky_images[:, :, ss] = img

    foo = stats.sigma_clipped_stats(sky_images, sigma=3)
    sky_mean = foo[0]    # DN/coadd
    sky_stddev = foo[2]

    # Calculate the expected photon noise.
    gain = 2.15
    sky_noise = (sky_mean * gain)**0.5 / gain


    print('sky mean   = ' + sky_mean)
    print('sky stddev = ' + sky_stddev)
    print('sky photon noise = ' + sky_noise)

    return
    

def stripes_dark():
    ### First investigate variance in sky files. 
    drk_files  = ['/g/lu/data/microlens/20may26os/raw/i200526_s006{0:03d}_xflip.fits'.format(ii) for ii in range(113, 118+1)]

    drk_images = np.zeros((2048, 2048, len(drk_files)), dtype=float)

    for ss in range(len(drk_files)):
        img = fits.getdata(drk_files[ss])

        drk_images[:, :, ss] = img

    foo = stats.sigma_clipped_stats(drk_images, sigma=3)
    drk_mean = foo[0]    # DN/coadd
    drk_stddev = foo[2]

    # Calculate the expected photon noise.
    gain = 2.15
    drk_noise = (drk_mean * gain)**0.5 / gain


    print('drk mean   = ' + drk_mean)
    print('drk stddev = ' + drk_stddev)
    print('drk photon noise = ' + drk_noise)

    return
    
def stripes_ref_pix():
    """
    Analyze the sky and the reference pixels, collapsed along x-axis to 
    see the horizontal striping patterns.
    """
    outdir = '/u/jlu/work/ao/keck/osiris/'
    sky_files  = ['/g/lu/data/microlens/20may25os/raw/i200525_a041{0:03d}.fits'.format(ii) for ii in range(2, 4+1)]
    sky_images = np.zeros((2048, 2048, len(sky_files)), dtype=float)

    for ss in range(len(sky_files)):
        img = fits.getdata(sky_files[ss])

        sky_images[:, :, ss] = img

    # Sigma clip to clean out bad and dead pixels.
    sky_im_clipped = stats.sigma_clip(sky_images, sigma_lower=10, sigma_upper=3)

    ##########
    # X-collapse analysis.
    ##########
    # Grab a big middle chunk of the sky and collapse along x
    y_sky = sky_im_clipped[0:2048, 500:1500, :].mean(axis=1)
    y_ref = sky_im_clipped[0:2048, 2044:2048, :].mean(axis=1)
    y_dif = y_sky - y_ref

    snr_sky = y_sky[900:1200, 0].mean() / y_sky[900:1200, 0].std()
    snr_dif = y_dif[900:1200, 0].mean() / y_dif[900:1200, 0].std()
    print('X collapse analysis:')
    print('   Sky SNR = {0:.1f}'.format(snr_sky))
    print('   Dif SNR = {0:.1f}'.format(snr_dif))

    plt.figure(1, figsize=(10, 6))
    plt.clf()
    plt.subplots_adjust(left=0.1)
    plt.plot(y_sky[:, 0], label='Sky Collapsed along x=[300:800]')
    plt.plot(y_ref[:, 0], label='Ref Collapsed along x=[2044:2048]')
    plt.plot(y_dif[:, 0], label='Difference')
    plt.xlabel('Y pixel')
    plt.ylabel('Mean (ADU/coadd)')
    plt.title('Sky tint=5.901 x 3 coadd')
    plt.legend(loc='center')
    plt.savefig(outdir + 'sky_x_collapse.png')



    ##########
    # Y-collapse analysis.
    ##########
    # Grab a big middle chunk of the sky and collapse along x
    x_sky = sky_im_clipped[500:1500, 0:2048, :].mean(axis=0)
    x_ref = sky_im_clipped[2044:2048, 0:2048, :].mean(axis=0)
    x_dif = x_sky - x_ref

    snr_sky = x_sky[900:1200, 0].mean() / x_sky[900:1200, 0].std()
    snr_dif = x_dif[900:1200, 0].mean() / x_dif[900:1200, 0].std()
    print('Y collapse analysis:')
    print('   Sky SNR = {0:.1f}'.format(snr_sky))
    print('   Dif SNR = {0:.1f}'.format(snr_dif))
    
    plt.figure(2, figsize=(10, 6))
    plt.clf()
    plt.subplots_adjust(left=0.1)
    plt.plot(x_sky[:, 0], label='Sky Collapsed along y=[500:1500]')
    plt.plot(x_ref[:, 0], label='Ref Collapsed along y=[2044:2048]')
    plt.plot(x_dif[:, 0], label='Difference')
    plt.xlabel('X pixel')
    plt.ylabel('Mean (ADU/coadd)')
    plt.title('Sky tint=5.901 x 3 coadd')
    plt.legend(loc='center')
    plt.savefig(outdir + 'sky_y_collapse.png')

    ##########
    # Subtract both X and Y reference pixels.
    ##########
    # Grab a big middle chunk of the sky and collapse along x
    x_ref2 = sky_images[2044:2048, :, :].mean(axis=0)
    y_ref2 = sky_images[:, 2044:2048, :].mean(axis=1)

    x_ref2 = x_ref2.reshape((1, 2048, len(sky_files)))
    y_ref2 = y_ref2.reshape((2048, 1, len(sky_files)))

    sky_im_refsub = sky_im_clipped - x_ref2
    sky_im_refsub -= y_ref2

    x_dif2 = sky_im_refsub[500:1500, 0:2048, :].mean(axis=0)
    y_dif2 = sky_im_refsub[0:2048, 500:1500, :].mean(axis=1)

    snr_x_dif2 = x_dif2[900:1200, 0].mean() / x_dif2[900:1200, 0].std()
    snr_y_dif2 = y_dif2[900:1200, 0].mean() / y_dif2[900:1200, 0].std()
    snr_x_sky = x_sky[900:1200, 0].mean() / x_sky[900:1200, 0].std()
    snr_y_sky = y_sky[900:1200, 0].mean() / y_sky[900:1200, 0].std()
    
    print('Reference pixel subtraction:')
    print('   X collapse: Dif SNR = {0:.1f}'.format(snr_y_dif2))
    print('   Y collapse: Dif SNR = {0:.1f}'.format(snr_x_dif2))
    print('   X collapse: Raw SNR = {0:.1f}'.format(snr_y_sky))
    print('   Y collapse: Raw SNR = {0:.1f}'.format(snr_x_sky))
    
    plt.figure(3, figsize=(10, 6))
    plt.clf()
    plt.subplots_adjust(left=0.1)
    plt.plot(x_sky[:, 0], label='Original Sky Collapsed along y=[500:1500]', color='cyan')
    plt.plot(y_sky[:, 0], label='Original Sky Collapsed along x=[500:1500]', color='magenta')
    plt.plot(x_dif2[:, 0], label='Corrected Sky Collapsed along y=[500:1500]', color='blue')
    plt.plot(y_dif2[:, 0], label='Corrected Sky Collapsed along x=[500:1500]', color='red')
    plt.xlabel('pixel')
    plt.ylabel('Mean (ADU/coadd)')
    plt.title('Sky tint=5.901 x 3 coadd')
    plt.legend(loc='center')
    plt.savefig(outdir + 'sky_xy_collapse_ref_correct.png')

    return

    
    
    
