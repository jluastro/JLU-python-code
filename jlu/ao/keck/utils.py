from astropy.io import fits
import glob

def get_obrt(glob_str):
    files = glob.glob(glob_str)

    for ii in range(len(files)):
        hdr = fits.getheader(files[ii], ignore_missing_end=True, option='ignore')
        print(hdr['filename'], hdr['obrt'])

    return
