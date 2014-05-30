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

import os
import sys
import cv2
import gtIO
import numpy as np

def MotionCorrect(cal_video, gaze_video):
    
    # Downsampling scale factor and frame border in pixels
    scale = 4
    border = 16    
    
    # Create template from calibration video
    print('Creating phase correlation template')
    template = CreateTemplate(cal_video, scale, border)

    # Cross correlate
    print('Phase correlating video')
    dx_t, dy_t = PhaseCorrelate(gaze_video, template, scale, border)
    
    # Optional: generate motion corrected video
    # print('Motion correcting video')
    # MotionCorrectVideo(gaze_video, dx_t, dy_t)
    
    # Clean up
    print('Done')

#
# Create phase correlation template from mean calibration video
#
def CreateTemplate(v_in_file, scale = 1, border = 0):
    
    # Open video
    try:
        v_in = cv2.VideoCapture(v_in_file)
    except:
        sys.exit(1)
        
    if not v_in.isOpened():
        sys.exit(1)

    # Frame counter
    fc = 0
            
    # Init continuation flag
    keep_going = True
        
    while keep_going:
            
        keep_going, I = gtIO.LoadVideoFrame(v_in, scale, border)
        
        if keep_going:
        
            # Caste to float32
            I = np.float32(I)
            
            # Add current downsampled frame to running total
            if fc < 1:
                Is  = I
                Iss = I*I
            else:
                Is += I
                Iss += I*I

            # Increment frame counter
            fc += 1
                            
    # Close video stream
    v_in.release()

    # Calculate temporal mean
    template = Is / float(fc)
    
    # Calculate temporal SD for use as a weighting function
    # Isd   = Iss / float(fc) - Imean * Imean
    
    # Save template to input video directory
    fstub, fext = os.path.splitext(v_in_file)
    out_file = fstub + 'moco_template.png'
    cv2.imwrite(out_file, np.uint8(template))

    return template

#
# FFT-based phase correlation of video frames with template
# Use OpenCV implementation
#   
def PhaseCorrelate(v_file, template, scale = 1, border = 0):
    
    # Open video of moving eye
    try:
        v_in = cv2.VideoCapture(v_file)
    except:
        sys.exit(1)
        
    if not v_in.isOpened():
        sys.exit(1)
        
    # Get frame total for video
    n_frames = v_in.get(cv2.cv.CV_CAP_PROP_FRAME_COUNT)
    
    # Create results arrays
    dx_t = np.zeros((n_frames,1))
    dy_t = np.zeros((n_frames,1))
        
    # Frame counter
    fc = 0
    
    # Init continuation flag
    keep_going = True
    
    while keep_going:
            
        # Downsample current frame
        keep_going, I = gtIO.LoadVideoFrame(v_in, scale, border)
        
        if keep_going:

            # Cast frame to float32
            I = np.float32(I)
            
            # Cross correlate with template
            dx, dy = cv2.phaseCorrelate(template, I)
            
            # Rescale to original pixel size
            dx_t[fc], dy_t[fc] = dx * scale, dy * scale
            
            # Construct rigid body transform matrix
            M = np.float32([[1,0,dx], [0,1,dy]]) 

            # Apply rigide body transform to frame
            I_moco = cv2.warpAffine(I, M, (I.shape[1], I.shape[0]), flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)
            
            # Display original and corrected frame
            im = np.hstack((np.uint8(I), np.uint8(I_moco), np.uint8(template)))
            cv2.imshow('MOCO', im)
            cv2.waitKey(5)
            
            # Increment frame counter
            fc += 1
            
    # Close video stream
    v_in.release()

    return dx_t, dy_t
    
#
# Weighted normalized cross correlation of two images
# Ported from Matlab code by Andy Diamond
# http://www.mathworks.com/matlabcentral/fileexchange/33340-wncc-weighted-normalized-cross-correlation
# BSD license on original Matlab source
#
def WNCC(im, ref, w):
    
    # Weighted average
    Ew(x,w) = sum_i(xi.wi)
    
    # Weighted covariance
    covw(x,y,w) = Ew( (x - Ew(x,w)) * (y - Ew(y,w)), w )
    
    # WNCC
    wncc(x,y,w) = covw(x,y,w) / sqrt(covw(x,x,w) * covw(y,y,w))