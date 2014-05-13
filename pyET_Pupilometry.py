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

import sys
import cv2
import numpy as np
from skimage import exposure
import pyET_FitEllipse as etf
    
def VideoPupilometry(v_file, rot = 0):
    
    # Input video
    print('Opening input video stream')
    try:
        vin_stream = cv2.VideoCapture(v_file)
    except:
        sys.exit(1)
        
    if not vin_stream.isOpened():
        sys.exit(1)
    
    fps = vin_stream.get(cv2.cv.CV_CAP_PROP_FPS)

    # Output video properties (where different from input)
    fourcc = cv2.cv.CV_FOURCC('m','p','4','v')
    
    print('Input video FPS     : %0.1f' % fps)
    
    # Set up LBP cascade classifier
    cascade = cv2.CascadeClassifier('Cascade/cascade.xml')
    
    # Read first interlaced frame from stream
    ret,frame = vin_stream.read()
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    
    # Apply rotation (if any)
    frame = RotateFrame(frame, rot)

    # Find rotated frame size
    nx, ny = frame.shape[1], frame.shape[0]
    
    # Downsample by 4
    nxd, nyd = int(nx/4), int(ny/4)
    
    print('Output video size   : %d x %d' % (nxd, nyd))
    
    # Output video
    print('Opening output video stream')
    try:
        vout_stream = cv2.VideoWriter('tracking.mov', fourcc, 30, (nxd, nyd), True)
    except:
        print('Problem creating output video stream')
        raise
        
    if not vout_stream.isOpened():
        print('Output video not opened')
        raise    
    
    
    while ret:
        
        # Rotate frame
        frame = RotateFrame(frame, rot)
        
        # Downsample frame
        frd = cv2.resize(frame, (nxd,nyd))
        
        # Find pupils in frame
        rects = cascade.detectMultiScale(frd, minNeighbors = 32)
        
        if len(rects) > 0:
            
            # Take first detected pupil ROI
            # TODO : adaptively adjust minNeighbors to return one pupil
            x, y, w, h = rects[0,:]
            x0, x1, y0, y1 = x, x+w, y, y+h
        
            # Overlay ROI bounds on frame
            cv2.rectangle(frd, (x0, y0), (x1, y1), (0,255,0), 1)
            
            # Extract pupil ROI
            roi = frd[y0:x0,x0:x1]
            
            # Run pupil fitting within ROI
            center, axes, angle, thresh = FitPupil(roi)
            
            print center
            
        else:
            
            print('Blink')

        cv2.imshow('frameWindow', frd)
        key = cv2.waitKey(int(1/fps*100))
        
        # Write output video frame
        vout_stream.write(frd)
        
        # Break on keypress
        if key > 0:
            break
        
        ret,frame = vin_stream.read()
    
    # Clean up
    print('Cleaning up')
    cv2.destroyAllWindows()
    vin_stream.release()
    vout_stream.release()

#   
# Find and fit pupil boundary ala Swirski
#
def FitPupil(roi, thresh = -1):

    # Intensity rescale to emphasize pupil
    # - assumes pupil is one of the darkest regions
    # - assumes pupil area is < 15% of frame
    pA, pB = np.percentile(roi, (1, 15))
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
    center, axes, angle = etf.FitEllipse_RANSAC(pnts, roi)
    
    return center, axes, angle, thresh

def SegmentPupil(gray, thresh = -1):
        
    # Follow up with Ostu thresholding
    thresh, pupil_bw = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)

    return pupil_bw, thresh
        
#
# Overlay fitted pupil ellipse on original frame
#    
def DisplayPupilEllipse(frame, center, axes, angle):

    # Ellipse color and line thickness
    color = (255,255,255)
    thickness = 1    
    
    # Overlay ellipse
    cv2.ellipse(frame, center, axes, angle, 0, 360, color, thickness)
    
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
    