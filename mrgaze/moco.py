#!/opt/local/bin/python
#
# Motion correction for gaze tracking
# - assume rigid body displacement of eye within frame only
# - use mean calibration video image as reference
# - use large downsampling factors for speed and robustness
#
# AUTHOR : Mike Tyszka
# PLACE  : Caltech
# DATES  : 2014-05-23 JMT From scratch
#
# This file is part of mrgaze.
#
#    mrgaze is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.
#
#    mrgaze is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.
#
#    You should have received a copy of the GNU General Public License
#   along with mrgaze.  If not, see <http://www.gnu.org/licenses/>.
#
# Copyright 2014 California Institute of Technology.

import cv2
import numpy as np
from scipy.ndimage.measurements import center_of_mass


def PseudoGlint(frame):
    '''
    Estimate pseudo-glint location in frame
    
    Use robust center of mass of horizontal edge detection as a proxy
    for a missing glint. Use horizontal gradient to improve robustness
    to horizontal scanline artifacts from MRI.
    
    Arguments
    ----
    frame : 2D numpy uint8 array
    
    Returns
    ----
    px, py : float tuple
        Pseudo-glint location in frame
    '''
    
    # Sobel horizontal intensity gradient
    gx = np.abs(cv2.Sobel(frame, cv2.CV_32F, 1, 0))
    
    # Robust center of mass of x-gradient image
    py, px = center_of_mass(gx)
    
    return px, py


def SobelX(img):
    '''
    Sobel x-gradient of image : dI/dx
    '''
    
    # Sobel kernel size
    k = 5
    
    # Sobel edges in video scanline direction (x)
    xedge = cv2.Sobel(img, cv2.CV_32F, 1, 0, ksize=k)

    # Force positive   
    xedge = np.abs(xedge)
    
    # Rescale image to [0,255] 32F
    imax, imin = np.amax(xedge), np.amin(xedge)
    xgrad = ((xedge - imin) / (imax - imin) * 255).astype(np.float32)

    return xgrad