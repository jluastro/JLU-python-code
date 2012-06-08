import numpy as np
import asciidata

def find_stars(file_in, aperture, file_out=None):
    """
    PURPOSE:
    
       Given array of pixel positions that have significant flux levels,
       return an array of pixel positions containing only the brightest
       pixels within an aperture.  Useful for determining the positions
       of stars in a field.  
    
    CALLING SEQUENCE:
       x_stars, y_stars = find_stars(file_in, aperture, file_out='mystars.txt')
    
    INPUTS:
      
       FILE_IN - Filename of table containing the following 2 columns: 
           X_PIX : X-coordinates of pixels with significant flux 
                   levels.  
           Y_PIX : Y-coordinates of pixels with significant flux
                   levels. 
           FLUX : Flux (in arbitrary units) of each pixel
    
       APER - Radius of (circular) aperture used to determine whether 
                 a pixel represents a star or not. In units of pixels.  
    
    KEYWORD PARAMETERS:
    
       FILE_OUT - Set to pathname of table where output will be written.
                  If left unset, will not save a table.
    
    OUTPUTS:
       X_STARS - Array of pixels representing the x-positions of the
                 actual stars in field 
    
       Y_STARS - Array of pixels representing the y-positions of the
                 actual stars in the field
    """
    # Read input table
    dataTab = asciidata.open(file_in)

    x = dataTab[0].tonumpy()
    y = dataTab[1].tonumpy()
    f = dataTab[2].tonumpy()

    badCount = 0
    
    # Loop over all the sources
    for ii in range(len(x)):
        # Select the pixel of interest
        x_src = x[ii]
        y_src = y[ii]
        f_src = f[ii]
    
        # Calculate distance to all other significant pixels
        dist = np.hypot(x - x_src, y - y_src)

        # Where is the distance closer than the aperture?
        idx_close = np.where((dist > 0) & (dist <= aperture))[0]

        # Check to see if our source is the brightest in the aperture
        if len(idx_close) > 0:
            idx_bright = np.where(f[idx_close] > f_src)[0]

            # If there are brighter sources then set values for our star to -99. The
            # distance will be large and will not show up in future comparisons,
            # and the flux will also never be large. Basically set this source to
            # be dropped.
            if len(idx_bright) > 0:
                x[ii] = -99
                y[ii] = -99
                f[ii] = -99
                badCount += 1
                print 'badCount = ', badCount

    idx_new = np.where(f > 0)[0]
    x_new = x[idx_new]
    y_new = y[idx_new]
    f_new = f[idx_new]

    if file_out != None:
        # Save to output table in ASCII and Region File in DS9 format.
        _out = open(file_out, 'w')
        _reg = open(file_out + '.reg', 'w')

        _out.write('#%7s  %8s  %6s\n' % ('X', 'Y', 'Flux'))
        for ii in range(len(x_new)):
            _out.write('%8.2f  %8.2f  %6d\n' %
                       (x_new[ii], y_new[ii], f_new[ii]))
            _reg.write('circle(%4d,%4d,%2d) # color = green\n' %
                       (x_new[ii], y_new[ii], aperture))
        _out.close()
        _reg.close()

    return (x_new, y_new)

