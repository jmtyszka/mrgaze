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
# This file is part of geetee.
#
#    geetee is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.
#
#    geetee is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.
#
#    You should have received a copy of the GNU General Public License
#   along with geetee.  If not, see <http://www.gnu.org/licenses/>.
#
# Copyright 2014 California Institute of Technology.

import sys
import cv2
import gtIO
import numpy as np
from scipy import signal
from matplotlib import pyplot as plt

def MotionCorrect(cal_video, gaze_video):
    
    # Downsampling scale factor and frame border in pixels
    scale = 16
    border = 16    
    
    # Create template from calibration video
    template = CreateTemplate(cal_video, scale, border)

    # Cross correlate
    dx, dy = CrossCorrelate(gaze_video, template, scale, border)
    

def CreateTemplate(v_file, scale = 16, border = 16):
    
    # Open video
    try:
        v_in = cv2.VideoCapture(v_file)
    except:
        sys.exit(1)
        
    if not v_in.isOpened():
        sys.exit(1)
        
    # Downsample and average video

    # Init continuation flag
    keep_going = True
    
    # Read first frame for info
    keep_going, frame = v_in.read()
    frame = gtIO.TrimBorder(frame, border)
    
    # Convert frame to grayscale
    I = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)

    # Determine downsampling factor    
    nx, ny = I.shape[1], I.shape[0]
    nxd, nyd = int(nx / scale), int(ny / scale)
    
    # Frame counter
    fc = 0
        
    while keep_going:
            
        # Downsample 
        I = (cv2.resize(I, (nxd, nyd))).astype(float)
            
        # Add current downsampled frame to running total
        if fc < 1:
            Is  = I
            Iss = I*I
        else:
            Is += I
            Iss += I*I

        # Read next frame (if available)
        keep_going, frame = v_in.read()
        
        if keep_going:
            frame = gtIO.TrimBorder(frame, 16)
            I = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        
        # Increment frame counter
        fc += 1
    
    # Calculate temporal mean
    template = Is / float(fc)
    
    # Calculate temporal SD for use as a weighting function
    # Isd   = Iss / float(fc) - Imean * Imean

    return template

    
def CrossCorrelate(v_file, template, scale, border):
    
    # Open video of moving eye
    try:
        v_in = cv2.VideoCapture(v_file)
    except:
        sys.exit(1)
        
    if not v_in.isOpened():
        sys.exit(1)
        
    # Init continuation flag
    keep_going = True
    
    # Read first frame for info
    keep_going, frame = v_in.read()
        
    # Convert frame to grayscale
    I = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    
    # Trim border
    frame = gtIO.TrimBorder(frame, border)

    # Determine downsampling factor    
    nx, ny = I.shape[1], I.shape[0]
    nxd, nyd = int(nx / scale), int(ny / scale)
    
    while keep_going:
            
        # Downsample current frame
        I = (cv2.resize(I, (nxd, nyd))).astype(float)
            
        # Cross correlate with template
        xc2 = signal.correlate2d(I, template)

        # Find maximum cross correlation
        ij = np.unravel_index(np.argmax(xc2), xc2.shape)
        x, y = ij[::-1]
        
        print('Maximum xcorr at (%f,%f)' % (x, y))

        # Read next frame (if available)
        keep_going, frame = v_in.read()
        
        if keep_going:
    
            # Convert frame to grayscale
            I = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    
            # Trim border
            frame = gtIO.TrimBorder(frame, border)

    return template