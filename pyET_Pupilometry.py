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
    
def RunPupilometry(v_file):
    
    func_name = 'RunPupilometry'    
    
    try:
        vin_stream = cv2.VideoCapture(v_file)
    except:
        print('%s : problem opening %s' % (func_name, v_file))
        sys.exit(1)
        
    if not vin_stream.isOpened():
        print('%s : video stream not open' % func_name)
        sys.exit(1)

    nFrames = int(vin_stream.get(cv2.cv.CV_CAP_PROP_FRAME_COUNT))
    print('%s : %d frames detected' % (func_name, nFrames))

    fps = vin_stream.get(cv2.cv.CV_CAP_PROP_FPS)
    print('%s : %0.1f frames per second' % (func_name, fps))    
    
    # Read first interlaced frame from stream
    ret,frame = vin_stream.read()
    
    while ret:
        cv2.imshow('frameWindow', frame)
        cv2.waitKey(int(1/fps*1000))
        ret,frame = vin_stream.read()
        
    # Close stream
    vin_stream.release()

#   
# Find and fit pupil boundary ala Swirski
#
def DetectPupil(gray, thresh = -1):

    # Downsample image
    h,w = gray.shape
    gray = cv2.resize(gray, (w/2, h/2))
    
    # Intensity rescale to emphasize pupil
    # - assumes pupil is one of the darkest regions
    # - assumes pupil area is < 15% of frame
    pA, pB = np.percentile(gray, (1, 15))
    gray = exposure.rescale_intensity(gray, in_range  = (pA, pB))
    
    # Crop to eye area
    # TODO : Add LBP classifier detection of pupil-iris region
    eye_gray = gray

    # Segment pupil in grayscale image and update threshold
    eye_bw, thresh = SegmentPupil(eye_gray, thresh)
    
    # Remove small features
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
    eye_bw = cv2.morphologyEx(eye_bw, cv2.MORPH_OPEN, kernel)
    
    # Identify edge pixels using Canny filter
    eye_edges = cv2.Canny(eye_bw, 0, 1)
    
    # Find all nonzero point coordinates
    pnts = np.transpose(np.nonzero(eye_edges))
    
    # Swap columns - pnts are (row, col) and need to be (x,y)
    pnts[:,[0,1]] = pnts[:,[1,0]]
    
    # Display all edge points
    cv2.imshow('Edge Points', eye_edges)
    cv2.waitKey(0)
    
    # RANSAC ellipse fitting to edge points
    center, axes, angle = etf.FitEllipse_RANSAC(pnts, eye_gray)
    
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
    