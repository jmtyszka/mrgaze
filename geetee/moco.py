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
import numpy as np

import io

def MotionCorrect(cal_video, gaze_video):
    
    # Downsampling scale factor and frame border in pixels
    scale = 4
    border = 16    
    
    # Create template from calibration video
    print('Creating phase correlation template')
    template = CreateTemplate(cal_video, scale, border)

    # Phase correlate
    print('Phase correlating video')
    dx_t, dy_t = PhaseCorrelate(cal_video, template, scale, border)
    
    # Optional: generate motion corrected video
    # print('Motion correcting video')
    # MotionCorrectVideo(gaze_video, dx_t, dy_t)
    
    # Clean up
    print('Done')

#
# Create phase correlation template from mean calibration video
#
def CreateTemplate(v_in_file, scale = 1, border = 0):
    
    # Template creation method
    method = 'last_frame'    
    
    # Open video
    try:
        v_in = cv2.VideoCapture(v_in_file)
    except:
        sys.exit(1)
        
    if not v_in.isOpened():
        sys.exit(1)
        
    # Get number of frames
    n_frames = int(v_in.get(cv2.cv.CV_CAP_PROP_FRAME_COUNT))
    
    if method == 'average':

        # Frame counter
        fc = 0
                
        # Init continuation flag
        keep_going = True
            
        while keep_going:
                
            keep_going, I, artifact = io.LoadVideoFrame(v_in, scale, border)
            
            if keep_going:
            
                # Caste to float32
                I = np.float32(I)
                
                # Add current downsampled frame to running total
                if fc < 1:
                    Is  = I
                else:
                    Is += I
    
                # Increment frame counter
                fc += 1
                                
        # Close video stream
        v_in.release()
    
        # Calculate temporal mean
        Im = Is / float(fc)
        
    elif method == 'last_frame':
        
        v_in.set(cv2.cv.CV_CAP_PROP_POS_FRAMES, n_frames-2)
        keep_going, I, artifact = io.LoadVideoFrame(v_in, scale, border)
        Im = np.float32(I)
    
    # Sobel edges in video scanline direction (x)
    xedge = SobelX(Im)
    
    # Rescale image to [0,255] 32F
    imax, imin = np.amax(xedge), np.amin(xedge)
    template = ((xedge - imin) / (imax - imin) * 255).astype(np.float32)
            
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
        keep_going, I, artifact = io.LoadVideoFrame(v_in, scale, border)
        
        if keep_going:
            
            # Horizontal Sobel edges
            xedge = SobelX(I)
            
            # Cross correlate with template
            dx, dy = cv2.phaseCorrelate(template, xedge)
            
            # Rescale to original pixel size
            dx_t[fc], dy_t[fc] = dx * scale, dy * scale
            
            # Construct rigid body transform matrix
            M = np.float32([[1,0,dx], [0,1,dy]]) 

            # Apply rigide body transform to frame
            I_moco = cv2.warpAffine(I, M, (I.shape[1], I.shape[0]), flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)
            
            # Display original and corrected frame
            im = np.hstack((np.uint8(template), np.uint8(xedge), np.uint8(I_moco)))
            cv2.imshow('MOCO', im)
            if cv2.waitKey(5) > 0:
                break
            
            # Increment frame counter
            fc += 1
            
    # Close video stream
    v_in.release()

    return dx_t, dy_t
    

def SobelX(img):
    
    # Sobel kernel size
    k = 5
    
    # Sobel edges in video scanline direction (x)
    xedge = cv2.Sobel(img, cv2.CV_32F, 1, 0, ksize=k)

    # Force positive   
    xedge = np.abs(xedge)
    
    # Rescale image to [0,255] 32F
    imax, imin = np.amax(xedge), np.amin(xedge)
    xedge = ((xedge - imin) / (imax - imin) * 255).astype(np.float32)

    return xedge