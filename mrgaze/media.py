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
import mrgaze.mrclean as mrc
from skimage import exposure


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

        # Get trimmed frame size
        nx, ny = fr.shape[1], fr.shape[0]
        
        # Calculate downsampled matrix
        nxd, nyd = int(nx/downsampling), int(ny/downsampling)
        
        # Downsample
        fr = cv2.resize(fr, (nxd, nyd))
        
        # Gaussian blur
        fr = cv2.GaussianBlur(fr, (3,3), 1.0)

        # Robust rescale to 5th, 95th percentile
        fr = RobustRescale(fr, (1,99))
        
        # Finally rotate frame
        fr = RotateFrame(fr, rotate)
    
    return status, fr, art_power



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


def RotateFrame(frame, rot):
    """
    Rotate frame in multiples of 90 degrees.
    
    Arguments
    ----
    frame : numpy uint8 array
        video frame to rotate
    rot : integer
        rotation angle in degrees (0, 90, 180 or 270)
        
    Returns
    ----
    frame : numpy uint8 array
        rotated frame
        
    Example
    ----
    >>> frame_rot = RotateFrame(frame, 180)
    """
    
    if rot == 270: # Rotate CCW 90
        frame = cv2.transpose(frame)
        frame = cv2.flip(frame, flipCode = 0)

    elif rot == 90: # Rotate CW 90
        frame = cv2.transpose(frame)
        frame = cv2.flip(frame, flipCode = 1)
        
    elif rot == 180: # Rotate by 180
        frame = cv2.flip(frame, flipCode = 0)
        frame = cv2.flip(frame, flipCode = 1)
    
    else: # Do nothing
        pass
        
    return frame
