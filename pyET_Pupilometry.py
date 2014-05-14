#!/opt/local/bin/python
#
# Video pupilometry functions
# - takes calibration and gaze video filenames as input
# - controls calibration and gaze estimation workflow
#
# USAGE : pyET.py <Calibration Video> <Gaze Video>
#
# AUTHOR : Mike Tyszka
# PLACE  : Caltech
# DATES  : 2014-05-07 JMT From scratch
#
# This file is part of pyET.
#
#    pyET is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.
#
#    pyET is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.
#
#    You should have received a copy of the GNU General Public License
#   along with pyET.  If not, see <http://www.gnu.org/licenses/>.
#
# Copyright 2014 California Institute of Technology.

import os
import sys
import time
import cv2
import numpy as np
from skimage import exposure
import pyET_FitEllipse as etf
    
def VideoPupilometry(v_file, rot = 0):
    
    # Output flags
    do_graphic = True
    verbose    = False    
    
    # Resampling scalefactor
    sf = 4;
    
    #%% Input video
    
    print('Opening input video stream')
    try:
        vin_stream = cv2.VideoCapture(v_file)
    except:
        sys.exit(1)
        
    if not vin_stream.isOpened():
        sys.exit(1)
    
    fps = vin_stream.get(cv2.cv.CV_CAP_PROP_FPS)
    
    if verbose: print('Input video FPS     : %0.1f' % fps)
    
    # Set up LBP cascade classifier
    cascade = cv2.CascadeClassifier('Cascade/cascade.xml')

    # Init frame count
    frame_count = 0
    
    # Init timer (for processing FPS)
    t0 = time.time()
    
    # Read first interlaced frame from stream
    keep_going, frame = vin_stream.read()
     
    # Apply rotation and get new size
    frame_rot = RotateFrame(frame, rot)
    nx, ny = frame_rot.shape[1], frame_rot.shape[0]
    
    #%% Output video
    
    # Downsample original NTSC video by 4 in both axes
    nxd, nyd = int(nx / sf), int(ny / sf)
    
    if verbose: print('Output video size   : %d x %d' % (nxd, nyd))    
    
    # Output video codec (MP4V - poor quality compression)
    # TODO : Find a better multiplatform codec
    fourcc = cv2.cv.CV_FOURCC('m','p','4','v')
    
    try:
        vout_stream = cv2.VideoWriter('tracking.mov', fourcc, 30, (nxd, nyd), True)
    except:
        print('Problem creating output video stream')
        raise
        
    if not vout_stream.isOpened():
        print('Output video not opened')
        raise 
        
    #%% Output pupilometry data
    
    # Modify video file name to get pupilometry text file
    fstub, fext = os.path.splitext(v_file)
    pout_name   = fstub + '_pupils.txt'
    
    # Open pupilometry text file to write
    try:
        pout_stream = open(pout_name, 'w')
    except:
        print('Problem opening pupilometry file : %s' % pout_name)
        return False

    while keep_going:
        
        # Convert current frame to single channel gray
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        
        # Rotate frame
        frame = RotateFrame(frame, rot)
        
        # Downsample frame
        frd = cv2.resize(frame, (nxd, nyd))
        
        # Find pupils in frame
        pupils = cascade.detectMultiScale(frd, minNeighbors = 40)
        
        # Count detected pupil candidates
        n_pupils = len(pupils)

        # TODO : adaptively adjust minNeighbors to return one pupil
        
        if n_pupils > 0:
            
            # Take first detected pupil ROI
            x, y, w, h = pupils[0,:]
            x0, x1, y0, y1 = x, x+w, y, y+h
            
            # Extract pupil ROI (note row,col indexing of image array)
            roi = frd[y0:y1,x0:x1]
            
            # Run pupil fitting within ROI
            el_roi, thresh = FitPupil(roi)
            
            # Add ROI offset
            el = (el_roi[0][0]+x0, el_roi[0][1]+y0), el_roi[1], el_roi[2]
            
            # Write data line to pupils file
            WritePupilometry(pout_stream, frame_count, el)
            
            # Display fitted pupil
            if do_graphic:

                # RGB version of downsampled frame
                frd_rgb = cv2.cvtColor(frd, cv2.COLOR_GRAY2RGB)

                # Overlay ROI bounds on downsampled frame
                cv2.rectangle(frd_rgb, (x0, y0), (x1, y1), (0,255,0), 1)
                    
                # Overlay fitted ellipse on frame
                cv2.ellipse(frd_rgb, el, (128,255,255), 1)
                
                cv2.imshow('frameWindow', frd_rgb)
                if cv2.waitKey(1) > 0:
                    break
            
        else:
            
            if verbose: print('*** Blink *** : %d' % frame_count)
        
        # Write output video frame
        vout_stream.write(frd)

        # Read next frame
        keep_going, frame = vin_stream.read()
        
        # Increment frame counter
        frame_count = frame_count + 1
        
        # Report processing FPS
        if verbose:
            pfps = frame_count / (time.time() - t0)  
            print('Processing FPS : %0.1f' % pfps)
    
    # Clean up
    if verbose: print('Cleaning up')
    cv2.destroyAllWindows()
    vin_stream.release()
    vout_stream.release()
    pout_stream.close()

    # Clean exit
    return True

#   
# Find and fit pupil boundary ala Swirski
#
def FitPupil(roi, thresh = -1):

    # Intensity rescale to emphasize pupil
    # - assumes pupil is one of the darkest regions
    # - assumes pupil area is < 50% of frame
    pA, pB = np.percentile(roi, (1, 50))
    roi = exposure.rescale_intensity(roi, in_range  = (pA, pB))
    
    # Segment pupil in grayscale image and update threshold
    roi_bw, thresh = SegmentPupil(roi, thresh)
    
    # Remove small features
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
    roi_bw = cv2.morphologyEx(roi_bw, cv2.MORPH_OPEN, kernel)
    
    # Identify edge pixels using Canny filter
    roi_edges = cv2.Canny(roi_bw, 0, 1)
    
    # Find all nonzero point coordinates
    pnts = np.transpose(np.nonzero(roi_edges))
    
    # Swap columns - pnts are (row, col) and need to be (x,y)
    pnts[:,[0,1]] = pnts[:,[1,0]]
    
    # RANSAC ellipse fitting to edge points
    ellipse = etf.FitEllipse_RANSAC(pnts, roi)
    
    return ellipse, thresh


def SegmentPupil(roi, thresh = -1):
        
    # Follow up with Ostu thresholding
    thresh, pupil_bw = cv2.threshold(roi, 128, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)

    return pupil_bw, thresh
        
#
# Overlay fitted pupil ellipse on original frame
#    
def DisplayPupilEllipse(frame, ellipse):

    # Ellipse color and line thickness
    color = (0,255,0)
    thickness = 1    
    
    # Overlay ellipse
    cv2.ellipse(frame, ellipse, color, thickness)
    
    # Display frame
    cv2.imshow('frameWindow', frame)
    
    # Wait for key press
    cv2.waitKey()
    
#
# Robust percentil intensity scaling
#
def RobustIntensity(gray, perc_range = (5, 95)):
    
    pA, pB = np.percentile(gray, perc_range)

#
# Rotate frame by multiples of 90 degrees
#
def RotateFrame(img, rot):
    
    if rot == 270: # Rotate CCW 90
        img = cv2.transpose(img)
        img = cv2.flip(img, flipCode = 0)

    elif rot == 90: # Rotate CW 90
        img = cv2.transpose(img)
        img = cv2.flip(img, flipCode = 1)
        
    elif rot == 180: # Rotate by 180
        img = cv2.flip(img, flipCode = 0)
        img = cv2.flip(img, flipCode = 1)
    
    else: # Do nothing
        pass
        
    return img

#
# Write pupilometry data line to file
#
def WritePupilometry(pupil_out, t, ellipse):
    
    # Unpack ellipse tuple
    (x0, y0), (bb, aa), phi_b_deg = ellipse
    
    # Pupil area
    area = etf.EllipseArea(ellipse)
    
    # Write pupilometry line to file
    pupil_out.write('%d,%0.1f,%0.1f,%0.1f,%0.1f,%0.1f,%0.1f,\n' % (t, area, x0, y0, bb, aa, phi_b_deg))