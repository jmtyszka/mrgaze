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

import os
import sys
import cv2
import numpy as np
from mrgaze import media

def MotionCorrect(cal_video, gaze_video, cfg):

    # Create template from calibration video
    print('Creating phase correlation template')
    template = CreateTemplate(cal_video, cfg)

    # Phase correlate
    print('Phase correlating video')
    dx_t, dy_t = PhaseCorrelate(cal_video, template, cfg)
    
    # Optional: generate motion corrected video
    # print('Motion correcting video')
    # MotionCorrectVideo(gaze_video, dx_t, dy_t)
    
    # Clean up
    print('Done')

#
# Create phase correlation template from mean calibration video
#
def CreateTemplate(v_in_file, cfg):
    
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
                
            keep_going, I, artifact = media.LoadVideoFrame(v_in, cfg)
            
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
        keep_going, I, artifact = media.LoadVideoFrame(v_in, cfg)
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

  
def PhaseCorrelate(src, template):
    '''
    Estimate frame to template displacement by phase correlation
    
    Arguments
    ----
    src : 2D numpy uint8 array
        Displaced frame
    template : 2D numpy uint8 array
        Undisplaced template frame (Sobel x-gradient)
    
    Returns
    ----
    dest : 2D numpy uint8 array
        Corrected frame
    dx, dy : floats
        X and Y displacement of img from template
    '''    

    # Affine warp flags
    aff_flags = cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP

    # Source frame size (x = columns)
    src_dims = src.shape[1], src.shape[0]

    # Horizontal Sobel edges
    xgrad = SobelX(src)
            
    # Cross correlate with template
    dx, dy = cv2.phaseCorrelate(template, xgrad)
    
    # Construct rigid body transform matrix
    M = np.float32([[1,0,dx], [0,1,dy]]) 

    # Apply rigid body transform to frame
    dest = cv2.warpAffine(src, M, src_dims, flags=aff_flags)

    return dest, dx, dy
    

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