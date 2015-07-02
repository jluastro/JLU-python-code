import numpy as np
import argparse
from jlu.gc.gcwork import starset
import pdb

def run(args=None):
    """
    align_rms main routine.
    """
    options = parse_options(args)

    # Read in the align output. Determine the number of
    # individual starlists are in the stack.
    s = starset.StarSet(options.root_name)
    
    N_lists = len(s.years)

    if options.idx_min == None:
        options.idx_min = N_lists
        
    if options.idx_max == None:
        options.idx_max = N_lists


    # Trim down the starlist to just those that are
    # in the desired number of epochs and are detected
    # in the reference epoch.
    s.stars = trim_stars(s, options)

    # Fetch the data off the starset
    name = s.getArray('name')
    x = s.getArrayFromAllEpochs('xpix')
    y = s.getArrayFromAllEpochs('ypix')
    m = s.getArrayFromAllEpochs('mag')
    f = 10**(m/-2.5)

    flux_dn = s.getArrayFromAllEpochs('fwhm')
    corr = s.getArrayFromAllEpochs('corr')
    nimg = s.getArrayFromAllEpochs('nframes')
    snr = s.getArrayFromAllEpochs('snr')

    # Identify where we have measurements and where are non-detections.
    good = ((x > -1000) & (y > -1000) & (m != 0))

    # Calculate the number of epochs the stars are detected in.
    cnt = good[options.idx_min : options.idx_max, :].sum(axis=0)

    # Mask the bad data.    
    x_msk = np.ma.masked_where(good == False, x, copy=True)
    y_msk = np.ma.masked_where(good == False, y, copy=True)
    f_msk = np.ma.masked_where(good == False, f, copy=True)
    m_msk = np.ma.masked_where(good == False, m, copy=True)

    flux_dn = np.ma.masked_where(good == False, flux_dn, copy=True)
    corr_msk = np.ma.masked_where(good == False, corr, copy=True)
    nimg_msk = np.ma.masked_where(good == False, nimg, copy=True)
    snr_msk = np.ma.masked_where(good == False, snr, copy=True)
    

    # Calculate the average x, y, m, f
    if options.idx_ref != None:
        print 'Using epoch {0} as average pos/flux'.format(options.idx_ref)
        x_avg = x[options.idx_ref, :]
        y_avg = y[options.idx_ref, :]
        m_avg = m[options.idx_ref, :]
        f_avg = f[options.idx_ref, :]
        
        year = s.dates[options.idx_ref]

        corr_avg = corr[options.idx_ref, :]
        flux_dn_avg = flux_dn[options.idx_ref, :]
        nimg_avg = nimg[options.idx_ref, :]
        snr_orig = snr[options.idx_ref, :]
    else:
        print 'Calculate average pos/flux '
        print 'from epochs {0} - {1}'.format(options.idx_min, options.idx_max)
        x_avg = x_msk[options.idx_min : options.idx_max, :].mean(axis=0)
        y_avg = y_msk[options.idx_min : options.idx_max, :].mean(axis=0)
        f_avg = f_msk[options.idx_min : options.idx_max, :].mean(axis=0)
        m_avg = -2.5 * np.log10(f_avg)
        
        year = s.years[options.idx_min : options.idx_max].mean()

        corr_avg = corr_msk[options.idx_min : options.idx_max, :].mean(axis=0)
        flux_dn_avg = flux_dn[options.idx_min : options.idx_max, :].mean(axis=0)
        nimg_avg = cnt
        snr_orig = None
        
    # Calculate the error on x, y, m, f
    x_std = calc_error(x_msk, x_avg, cnt, options)
    y_std = calc_error(y_msk, y_avg, cnt, options)
    f_std = calc_error(f_msk, f_avg, cnt, options)
    m_std = f_std / f_avg

    # Estimate a new signal to noise ratio
    new_snr = f_avg / f_std

    if (options.calc_rel_err == False) and (snr_orig != None):
        new_snr = 1.0 / np.hypot(1.0/new_snr, 1.0/snr_orig)

    # Fix up any infinities in the SNR. Set them to 0.
    new_snr[np.isinf(new_snr)] = 0.0
    new_snr[new_snr.mask] = 0.0

    _out = open(options.out_root + '.lis', 'w')

    hdr = '{name:13s}  {mag:>6s}  {year:>8s}  '
    hdr += '{x:>9s}  {y:>9s}  {xe:>9s}  {ye:>9s}  '
    hdr += '{snr:>20s}  {corr:>6s}  {nimg:>8s}  {flux:>20s}\n'

    _out.write(hdr.format(name='# name', mag='mag', year='year',
                          x='x', y='y', xe='xe', ye='ye',
                          snr='snr', corr='corr', nimg='nimg', flux='flux'))
    

    
    fmt = '{name:13s}  {mag:6.3f}  {year:8.3f}  '
    fmt += '{x:9.3f}  {y:9.3f}  {xe:9.3f}  {ye:9.3f}  '
    fmt += '{snr:20.4f}  {corr:6.2f}  {nimg:8f}  {flux:20.0f}\n'
    
    for ss in range(len(x_avg)):
        _out.write(fmt.format(name=name[ss], mag=m_avg[ss], year=float(year),
                              x=x_avg[ss], y=y_avg[ss], xe=x_std[ss], ye=y_std[ss],
                              snr=new_snr[ss], corr=corr_avg[ss], nimg=nimg_avg[ss],
                              flux=flux_dn_avg[ss]))

    _out.close()
    
    return s

def trim_stars(s, options):
    # Get the relevant variables off the starset
    x = s.getArrayFromAllEpochs('xpix')
    y = s.getArrayFromAllEpochs('ypix')
    m = s.getArrayFromAllEpochs('mag')

    # Mask where we have null detections and
    # ID good stars where we have detections.
    good = ((x > -1000) & (y > -1000) & (m != 0))

    # Figure out the number of epochs each star is detected in.
    # Trim out the stars that don't have enough detections.
    cnt = good[options.idx_min : options.idx_max, :].sum(axis=0)

    # Figure out if a stars is not detected in the reference epoch
    # and set it for trimming (cnt = 0).
    if options.idx_ref != None:
        cnt[good[options.idx_ref, :] == False] = 0

    # Trim our arrays to only the "good" stars that meet our number
    # of epochs criteria.
    idx = np.where(cnt >= options.N_required)[0]
    
    new_stars = [s.stars[ii] for ii in idx]

    return new_stars

def calc_error(v_msk, v_avg, cnt, options):
    v_tmp = (v_msk[options.idx_min : options.idx_max, :] - v_avg)**2
    v_tmp = v_tmp.sum(axis=0)
    v_tmp /= cnt - 1.0
    v_std = np.sqrt(v_tmp)

    idx = np.where(cnt <= 1)[0]
    v_std[idx] = 0.0
    
    return v_std
    
def parse_options(args):
    purpose = 'Combine a stack of aligned starlists to produce a new "average"\n'
    purpose += 'starlist with astrometric and photometric errors estimated from\n'
    purpose += 'the RMS error (or error on the mean) of the stack.\n'
    purpose += '\n'
    purpose += ''

    ##########
    # Setup a Parser
    ##########    
    parser = argparse.ArgumentParser(description=purpose)

    # root_name
    help_str = 'The root of the align output files.'
    parser.add_argument('root_name', type=str, help=help_str)

    # N_required
    help_str = 'The minimum number of starlists a star must be in to be included '
    help_str += 'in the output list.'
    parser.add_argument('N_required', type=int, help=help_str)

    # out_root
    help_str = 'Output root of files. Default is the input align '
    help_str += 'root_name + "_rms".'
    parser.add_argument('-o', '--outroot', dest='out_root', help=help_str)
    
    # calc_err_mean
    help_str = 'Include to calculate the error on the mean rather than the '
    help_str += 'RMS error.'
    parser.add_argument('--errOnMean', '-e', dest='calc_err_mean',
                        action='store_true', help=help_str)

    # calc_rel_err    
    help_str = 'Include to calculate the relative pohtometric error '
    help_str += '(no zero-point error).'
    parser.add_argument('--relPhotErr', '-r', dest='calc_rel_err',
                        action='store_true', help=help_str)

    # stack_min
    help_str = 'The index of the starlist that starts the stack over which '
    help_str += 'to calculate averages and errors (def=1). Note that the '
    help_str += 'default use is that the first image (idx=0) contains the'
    help_str += '"average" star positions/fluxes and that the first stack'
    help_str += 'image is at --stackMin=1.'
    parser.add_argument('--stackMin', dest='idx_min', help=help_str,
                        action='store', default=1, type=int)
    
    # stack_max
    help_str = 'The index of the starlist that ends the stack over which '
    help_str += 'to calculate averages and errors (def=last). Note that the '
    help_str += 'default use is that the first image (idx=0) contains the'
    help_str += '"average" star positions/fluxes and that the last stack'
    help_str += 'image is at --stackMax=max.'
    parser.add_argument('--stackMax', dest='idx_max', help=help_str, type=int)
    
    # reference_list
    help_str = 'Specify the index number (0-based) of the starlist that should be '
    help_str += 'adopted as the reference. If a reference list is specified, the average '
    help_str += 'positions and fluxes will come from this list and only the astrometric '
    help_str += 'and photometric errors will be calculated from the remaining stack '
    help_str += 'of images. If None is specified, then the average positions and fluxes '
    help_str += 'will come from stacking the remaining set of images.'
    parser.add_argument('--refList', dest='idx_ref', help=help_str,
                        action='store', default=0)
    
    options = parser.parse_args(args)

    # Define the root file name for the output files.
    if options.out_root == None:
        options.out_root = options.root_name + '_rms'

    # Fix the reference epoch.
    if options.idx_ref == 'None':
        options.idx_ref = None
        
    return options

if __name__ == "__main__":
    run()



