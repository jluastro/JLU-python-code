import astropy.io.fits as fits

"""
Test WFC3 Distortion with Overlap Region
in both the 2010 and 2013 F160W data.
"""
data_sets = {'2010_1': [],
             '2010_2': [],
             '2013_1': [],
             '2013_2': []}    

def get_shifts(drz_root):
    """
    From a drizzle file, get the shifts that were applied.
    """
    hdr = fits.getheader(drz_root + '_drz.fits')
    keys = hdr.keys()
    vals = hdr.values()

    file_name = []
    xshift = []
    yshift = []

    for ii in range(len(keys)):
        key = keys[ii]
        val = vals[ii]

        if key.startswith('D') and key.endswith('DATA'):
            file_name.append(val)
        if key.startswith('D') and key.endswith('XSH'):
            xshift.append(val)
        if key.startswith('D') and key.endswith('YSH'):
            yshift.append(val)

    for ii in range(len(file_name)):
        print file_name[ii], xshift[ii], yshift[ii]
        
