import numpy as np
from scipy import interpolate


class ShiftTransformer(object):
    def __init__(self, image, interp_type='F', spline_degree=3):
        """
        An object that can interpolate a 2D image in order to perform
        a fractional shift of the origin.

        Inputs:
        image - 2D array to be shifted

        Keyword arguments:
        interp_type - The type of interpolation to perform. This is over-ridden by
                      the data structure, which should have an interp_type variable.
                      'F': interpolation by Fourier transform shift (default)
                      'S': interpolation by spline functions
                      'I': interpolation by the scipy interp2d function

        Output:
        ShiftTransformer object that can then be called to transform() the image.

        Restrictions:
        1) Interpolation is suited to well-sampled data. For undersampled data, use
           other techniques such as the stars.pro module in Starfinder.
        2) This routine performs fractional shifts. When the shift exceeds .5 pixels,
           strong edge effects might occur.

        References:
        This was adapted from the image_shift.pro routine included with Starfinder
        (Diolatti et al. 1999) and modified by Gunther Witzel (2013).
        """
        # Check that interpolation type is set properly
        self.interp_type = interp_type.upper()
        self.spline_degree = spline_degree

        if self.interp_type not in ['F', 'S', 'I']:
            self.interp_type = 'F'

        if self.interp_type == 'S':
            if (self.spline_degree <= 0) or (self.spline_degree >= 6):
                # Override invalid choice spline degree
                print 'Invalide spline_degree, resetting to degree = 3.'
                self.spline_degree = 3


        # Pad image with 0s to prevent edge effects. Skip for FFT-baesd shifting.
        if self.interp_type != 'F':
            img_pad = np.pad(image, 1, mode='constant')
            offset = [1, 1]
        else:
            img_pad = image
            offset = [0, 0]

        # Spline interpolation
        if self.interp_type == 'S':
            x = np.arange(img_pad.shape[0])
            y = np.arange(img_pad.shape[1])
            interp_obj = interpolate.RectBivariateSpline(x, y, img_pad,
                                                         kx = self.spline_degree,
                                                         ky = self.spline_degree)
            self.interp_obj = interp_obj
        
        

def pad_array(image, new_size):

        
def shift(image, x_shift, y_shift, interp_type='F', data=None):
    """
    Interpolate a 2D image in order to perform a fractional shift of the origin.

    Inputs:
    image - 2D array to be shifted
    x_shift - shift along the first axis (index) (e.g. img[x, y])
    y_shift - shift along the second axis
    
    Keyword arguments:
    interp_type - The type of interpolation to perform. This is over-ridden by
                  the data structure, which should have an interp_type variable.
                  'F': interpolation by Fourier transform shift (default)
                  'S': interpolation by spline functions
                  'I': interpolation by the scipy interp2d function

    data - A structure created by a previous call to this function.

    Output:
    new_image - the shifted image.
    data - A structure to hold the interpolation information for future calls.

    Restrictions:
    1) Interpolation is suited to well-sampled data. For undersampled data, use
       other techniques such as the stars.pro module in Starfinder.
    2) This routine performs fractional shifts. When the shift exceeds .5 pixels,
       strong edge effects might occur.

    References:
    This was adapted from the image_shift.pro routine included with Starfinder
    (Diolatti et al. 1999) and modified by Gunther Witzel (2013).
        
    """
    # Catch the zero-shift case.
    if (x_shift == 0) and (y_shift == 0):
        return image, data

    # Check if we have a DATA structure passed in.
    
