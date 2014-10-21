#!/opt/local/bin/python
"""
Utility functions, primarily for I/O

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
from mrgaze import mrclean as mrc
from mrgaze import improc as ip
from skimage.transform import rotate


def LoadVideoFrame(v_in, cfg):
    """
    Load and preprocess a single frame from a video stream
    
    Parameters
    ----------
    v_in : opencv video stream
        video input stream
    scale : float
        downsampling scale factor [1.0]
    border : int
        pixel border width to strip
    rotate : int
        rotation in
        
    Returns
    ----
    status : boolean
        Completion status.
    fr : numpy uint8 array
        Preprocessed video frame.
    art_power : float
        Artifact power in frame.
    """
    
    # Extract config parameters
    downsampling = cfg.getfloat('VIDEO', 'downsampling')
    border       = cfg.getint('VIDEO', 'border')
    rotate       = cfg.getint('VIDEO', 'rotate')
    do_mrclean   = cfg.getboolean('ARTIFACTS', 'mrclean')
    z_thresh     = cfg.getfloat('ARTIFACTS', 'zthresh')
    
    # Preprocesing flags
    gauss_sd = 0.0
    perc_range = (1, 99)
    bias_correct = False
    
    # Init returned artifact power
    art_power = 0.0    
    
    # Read one frame from stream    
    status, fr = v_in.read()
    
    if status:
        
        # Convert to grayscale
        fr = cv2.cvtColor(fr, cv2.COLOR_RGB2GRAY)
        
        # Trim border first
        fr = TrimBorder(fr, border)
        
        # Apply optional MR artifact suppression
        if do_mrclean:
            fr, art_power = mrc.MRClean(fr, z_thresh)

        if downsampling > 1:
            fr = Downsample(fr, downsampling)
        
        # Gaussian blur
        if gauss_sd > 0.0:
            fr = cv2.GaussianBlur(fr, (3,3), gauss_sd)
            
        # Correct for illumination bias
        if bias_correct:
            bias_field = ip.EstimateBias(fr)
            fr = ip.Unbias(fr, bias_field)

        # Robust rescale to [0,50] percentile
        # Emphasize darker areas such as pupil
        fr = ip.RobustRescale(fr, perc_range)
        
        # Finally rotate frame
        fr = RotateFrame(fr, rotate)
    
    return status, fr, art_power


def Downsample(frame, factor):
    # Get trimmed frame size
    nx, ny = frame.shape[1], frame.shape[0]
    
    # Calculate downsampled matrix
    nxd, nyd = int(nx/factor), int(ny/factor)
    
    # Downsample
    frame = cv2.resize(frame, (nxd, nyd))

    return frame
    

def LoadImage(image_file, border=0):
    """
    Load an image from a file and strip the border.

    Parameters
    ----------
    image_file : string
        File name of image
    border : integer
        Pixel width of border to strip [0].
        

    Returns
    -------
    frame : 2D numpy array
        Grayscale image array.

    Examples
    --------
    >>> img = LoadImage('test.png', 5)
    """

    # Initialize frame
    frame = np.array([])

    # load test frame image
    try:
        frame = cv2.imread(image_file)
    except:
        print('Problem opening %s to read' % image_file)
        return frame
        
    # Convert to grayscale image if necessary
    if frame.shape[2] == 3:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Trim border (if requested)
    frame = TrimBorder(frame, border)
    
    return frame


def TrimBorder(frame, border = 0):
    """
    Trim video frame border introduced by frame capture
    
    Parameters
    ----------
    frame : numpy uint8 array
        video frame
    border : integer
        border width in pixels to strip [0]
        
    Returns
    -------
    frame : numpy unit8 array
        video frame without border
    """
    
    if border > 0:
        
        # Get image dimension
        nx, ny = frame.shape[1], frame.shape[0]
        
        # Set bounding box
        x0 = border
        y0 = border
        x1 = nx - border
        y1 = ny - border
        
        # Make sure bounds are inside image
        x0 = x0 if x0 > 0 else 0
        x1 = x1 if x1 < nx else nx-1
        y0 = y0 if y0 > 0 else 0
        y1 = y1 if y1 < ny else ny-1
        
        # Crop and return
        return frame[y0:y1, x0:x1]
        
    else:
        
        return frame


def RotateFrame(frame, theta_deg):
    """
    Rotate frame in multiples of 90 degrees.
    
    Arguments
    ----
    frame : numpy uint8 array
        Video frame to rotate.
    theta_deg : integer
        CCW rotation angle in degrees (math convention)
        Integer multiples of 90 degrees are handled quickly.
        Arbitrary rotation angles are slower.
        
    Returns
    ----
    new_frame : numpy uint8 array
        rotated frame
        
    Example
    ----
    >>> frame_rot = RotateFrame(frame, 90)
    """
    
    if theta_deg == 0:
        
        # Do nothing
        new_frame = frame.copy()
    
    elif theta_deg == 90:
    
        # Rotate CCW 90
        new_frame = cv2.transpose(frame)
        new_frame = cv2.flip(new_frame, flipCode = 0)

    elif theta_deg == 270:
    
        # Rotate CCW 270 (CW 90)
        new_frame = cv2.transpose(frame)
        new_frame = cv2.flip(new_frame, flipCode = 1)
        
    elif theta_deg == 180:
    
        # Rotate by 180
        new_frame = cv2.flip(frame, flipCode = 0)
        new_frame = cv2.flip(new_frame, flipCode = 1)
    
    else: # Arbitrary rotation
    
        new_frame = rotate(frame, theta_deg, resize=True)
        
        # Scale and recast to uint8
        new_frame = np.uint8(new_frame * 255.0)
        
    return new_frame
