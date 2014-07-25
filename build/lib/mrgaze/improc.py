#!/opt/local/bin/python
"""
Image processing support functions

This file is part of mrgaze.

    mrgaze is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    mrgaze is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with mrgaze.  If not, see <http://www.gnu.org/licenses/>.

Copyright 2014 California Institute of Technology.
"""

import cv2
import numpy as np
from skimage import exposure
from mrgaze import utils

def EstimateBias(fr):
    '''
    Estimate illumination bias field
    
    Arguments
    ----
    fr : 2D numpy uint8 array
        Uncorrected image with biased illumination
    
    Returns
    ----
    bias_field : 2D numpy float array
        Estimated bias multiplier field
    '''

    # Target downsampled matrix size
    nd = 32;

    # Get image dimensions
    ny, nx = fr.shape
    
    # Target maximum dimension is 32
    # Apect ratio preserved approximately
    if nx > ny:
        nxd = nd
        nyd = int(nx/32.0 * ny)
    else:
        nxd = int(ny/32.0 * nx)
        nyd = nd
        
    # Downsample frame
    fr_d = cv2.resize(fr, (nxd, nyd))
    
    # 2D baseline estimation
    # Use large kernel relative to image size
    k = utils._forceodd(nd/2)
    bias_field_d = cv2.medianBlur(fr_d, k)
    
    # Upsample biasfield to same size as original frame
    bias_field = cv2.resize(bias_field_d, (nx, ny))
    
    return bias_field

    
def Unbias(fr, bias_field):

    # Cast frame to floats    
    frf = fr.astype(float)

    # Save image mean (add back after high pass filter)
    fr_mean = np.mean(frf)
    
    # Apply bias correction multiplier
    fr_unbias = np.uint8(frf / bias_field * fr_mean)
    
    return fr_unbias

    
def RobustRescale(gray, perc_range=(5, 95)):
    """
    Robust image intensity rescaling
    
    Arguments
    ----
    gray : numpy uint8 array
        Original grayscale image.
    perc_range : two element tuple of floats in range [0,100]
        Percentile scaling range
    
    Returns
    ----
    gray_rescale : numpy uint8 array
        Percentile rescaled image.
    """
    
    pA, pB = np.percentile(gray, perc_range)
    gray_rescale = exposure.rescale_intensity(gray, in_range=(pA, pB))
    
    return gray_rescale

