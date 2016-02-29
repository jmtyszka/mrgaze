#!/usr/bin/env python
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
from mrgaze import improc, mrclean
from skimage.transform import rotate


def find_cameras(maxdevno=10):

    cameraList = []

    for devno in range(0,maxdevno):

        print('Device %d ... ' % devno, end='')
        vin = cv2.VideoCapture(devno)

        print('checking ... ', end='')
        if vin.isOpened():
            print('good')
        else:
            print('bad')

        vin.release()

    return cameraList


def load_video_frame(v_in, cfg):
    """ Load and preprocess a single frame from video stream

    Parameters
    ----------
    v_in : opencv video stream
        video input stream
    cfg : border/rotate/mrclean/zthresh/downsampling

    Returns
    ----
    status : boolean
        Completion status.
    fr : numpy uint8 array
        Preprocessed video frame.
    art_power : float
        Artifact power in frame.
    """

    # Load one frame
    status, fr = v_in.read()

#    # If frame loaded successfully, preprocess
#    if status:
#        fr, art_power = preproc(fr, cfg)
#    else:
#        art_power = 0.0

    return status, fr


def preproc(fr, cfg):
    """
    Preprocess a single frame

    Parameters
    ----------
    fr : numpy uint8 array
        raw video frame.
    cfg : border/rotate/mrclean/zthresh/downsampling

    Returns
    ----
    fr : numpy uint8 array
        Preprocessed video frame.
    art_power : float
        Artifact power in frame.
    """

    # Extract video processing parameters
    downsampling = cfg.getfloat('VIDEO', 'downsampling')
    border = cfg.getint('VIDEO', 'border')
    rotate = cfg.getint('VIDEO', 'rotate')
    do_mrclean = cfg.getboolean('ARTIFACTS', 'mrclean')
    z_thresh = cfg.getfloat('ARTIFACTS', 'zthresh')
    perclow = cfg.getfloat('PREPROC', 'perclow')
    perchigh = cfg.getfloat('PREPROC', 'perchigh')

    # Preprocessing flags
    perc_range = (perclow, perchigh)
    bias_correct = False
    bias_correct = True

    # Init returned artifact power
    art_power = 0.0

    # Convert to grayscale
    fr = cv2.cvtColor(fr, cv2.COLOR_RGB2GRAY)

    # Trim border first
    fr = trim_border(fr, border)

    # Apply optional MR artifact suppression
    if do_mrclean:
        fr, art_power = mrclean.MRClean(fr, z_thresh)

    # downsample
    if downsampling > 1:
        fr = downsample(fr, downsampling)

    # Correct for illumination bias
    if bias_correct:
        bias_field = improc.EstimateBias(fr)
        fr = improc.Unbias(fr, bias_field)

    # Robust rescale to [0,50] percentile
    # Emphasize darker areas such as pupil
    fr = improc.RobustRescale(fr, perc_range)

    # Finally rotate frame
    fr = rotate_frame(fr, rotate)

    return fr, art_power


def downsample(frame, factor):
    # Get trimmed frame size
    nx, ny = frame.shape[1], frame.shape[0]

    # Calculate downsampled matrix
    nxd, nyd = int(nx/factor), int(ny/factor)

    # downsample with area averaging
    frame = cv2.resize(frame, (nxd, nyd), interpolation=cv2.INTER_AREA)

    return frame


def load_image(image_file, cfg):
    """
    Load an image from a file and strip the border.

    Parameters
    ----------
    image_file : string
        File name of image
    cfg :


    Returns
    -------
    frame : 2D numpy array
        Grayscale image array.

    Examples
    --------
    >>> img = load_image('test.png', 5)
    """

    # Initialize frame
    frame = np.array([])

    # load test frame image
    frame = cv2.imread(image_file)

    if frame.size == 0:
        print('Problem opening %s to read' % image_file)
        return frame

    # Preprocess frame
    frame, _ = preproc(frame, cfg)

    return frame


def trim_border(frame, border = 0):
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


def rotate_frame(frame, theta_deg):
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
    >>> frame_rot = rotate_frame(frame, 90)
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
