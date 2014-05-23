#!/opt/local/bin/python
#
# Video pupilometry functions
# - takes calibration and gaze video filenames as input
# - controls calibration and gaze estimation workflow
#
# USAGE : geetee.py <Calibration Video> <Gaze Video>
#
# AUTHOR : Mike Tyszka
# PLACE  : Caltech
# DATES  : 2014-05-07 JMT From scratch
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
import time
import cv2
import scipy.ndimage as spi
import numpy as np
import gtFitEllipse as fe
from skimage import exposure

def VideoPupilometry(v_file, rot = 0):
    
    # Output flags
    do_graphic = True
    verbose    = False    
    
    # Resampling scalefactor
    sf = 4;
    
    # Set up LBP cascade classifier
    cascade = cv2.CascadeClassifier('Cascade/cascade.xml')
    
    if cascade.empty():
        print('LBP cascade is empty - check Cascade directory exists')
        sys.exit(1)
    
    #
    # Input video
    #
    if verbose: print('Opening input video stream')
    try:
        vin_stream = cv2.VideoCapture(v_file)
    except:
        sys.exit(1)
        
    if not vin_stream.isOpened():
        sys.exit(1)
    
    # Get FPS from video file
    fps = vin_stream.get(cv2.cv.CV_CAP_PROP_FPS)
    
    # Read first interlaced frame from stream
    keep_going, frame = vin_stream.read()
     
    # Apply rotation and get new size
    frame_rot = RotateFrame(frame, rot)
    nx, ny = frame_rot.shape[1], frame_rot.shape[0]
        
    # Downsample original NTSC video by 4 in both axes
    nxd, nyd = int(nx / sf), int(ny / sf)
    
    #
    # Output video
    #
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

    # 
    # Output pupilometry data
    #

    # Modify video file name to get pupilometry text file
    fstub, fext = os.path.splitext(v_file)
    pout_name   = fstub + '_pupils.csv'
    
    # Open pupilometry text file to write
    try:
        pout_stream = open(pout_name, 'w')
    except:
        print('Problem opening pupilometry file : %s' % pout_name)
        return False

    #
    # Main Video Frame Loop
    #

    # Init frame counter
    fc = 0
    
    # Init processing timer
    t0 = time.time()

    while keep_going:
        
        # Current video time in seconds
        t = fc / fps
        
        # Convert current frame to single channel gray
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        
        # Rotate frame
        frame = RotateFrame(frame, rot)
        
        # Downsample frame
        frd = cv2.resize(frame, (nxd, nyd))
        
        # RGB version of downsampled frame for output video
        frd_rgb = cv2.cvtColor(frd, cv2.COLOR_GRAY2RGB)
        
        # -------------------------------------
        # Pass this frame to pupilometry engine
        # -------------------------------------
        ellipse, roi_rect, blink = PupilometryEngine(frd, cascade)
                
        # Write data line to pupils file
        WritePupilometry(pout_stream, t, ellipse, blink)
            
        if do_graphic and not blink:

            # Overlay ROI and pupil ellipse on RGB frame
            p1, p2 = roi_rect                      
            cv2.rectangle(frd_rgb, p1, p2, (0,255,0), 1)
            cv2.ellipse(frd_rgb, ellipse, (128,255,255), 1)
            cv2.imshow('frameWindow', frd_rgb)

            if cv2.waitKey(1) > 0:
                break
        
        # Write output video frame
        vout_stream.write(frd_rgb)

        # Read next frame (if available)
        keep_going, frame = vin_stream.read()
        
        # Increment frame counter
        fc = fc + 1
        
        # Report processing FPS
        if verbose:
            pfps = fc / (time.time() - t0)  
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
# Pupil find and fit engine
#
def PupilometryEngine(img, cascade):
    
    # Find pupils in frame
    pupils = cascade.detectMultiScale(img, minNeighbors = 40)
        
    # Count detected pupil candidates
    n_pupils = len(pupils)

    # TODO : adaptively adjust minNeighbors to return one pupil
        
    if n_pupils > 0:
        
        # Unset blink flag
        blink = False
            
        # Take first detected pupil ROI
        x, y, w, h = pupils[0,:]
        x0, x1, y0, y1 = x, x+w, y, y+h
        roi_rect = (x0,y0),(x1,y1)
            
        # Extract pupil ROI (note row,col indexing of image array)
        pupil_roi = img[y0:y1,x0:x1]
        
        # Segment pupil intelligently
        pupil_bw = SegmentPupil(pupil_roi)
            
        # Fit ellipse to pupil boundary
        el_roi = FitPupil(pupil_bw, pupil_roi)
            
        # Add ROI offset
        el = (el_roi[0][0]+x0, el_roi[0][1]+y0), el_roi[1], el_roi[2]
            
    else:
            
        # Set blink flag
        blink = True
        el = ((np.nan, np.nan), (np.nan, np.nan), np.nan)
        roi_rect = (np.nan, np.nan), (np.nan, np.nan)

    return el, roi_rect, blink            

#   
# Segment pupil region
# Use relative area and circularity to score candidate regions
#
def SegmentPupil(roi):

    # Intensity rescale to emphasize pupil
    # - assumes pupil is one of the darkest regions
    # - assumes pupil occupies between 5% and 50% of frame area
    pA, pB = np.percentile(roi, (5, 50))
    roi = exposure.rescale_intensity(roi, in_range  = (pA, pB))
        
    # Segment pupil in contrast stretched roi and update threshold
    thresh, blobs = cv2.threshold(roi, 128, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
        
    # Morphological opening (circle 5 pixels diameter)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
    blobs = cv2.morphologyEx(blobs, cv2.MORPH_OPEN, kernel)
        
    # Label connected components - one should be the pupil
    labels, n_labels = spi.measurements.label(blobs)
    
    # Measure blob areas
    areas = spi.sum(blobs, labels, range(n_labels+1))
        
    # Find maximum area blob
    pupil_label = np.where(areas == areas.max())[0][0]
    
    # Extract blob with largest area
    pupil_bw = np.uint8(labels == pupil_label)
        
    return pupil_bw


def FitPupil(bw, roi):
     
    # Identify edge pixels using Canny filter
    roi_edges = cv2.Canny(bw, 0, 1)
    
    # Find all edge point coordinates
    pnts = np.transpose(np.nonzero(roi_edges))
    
    # Swap columns - pnts are (row, col) and need to be (x,y)
    pnts[:,[0,1]] = pnts[:,[1,0]]
    
    # RANSAC ellipse fitting to edge points
    ellipse = fe.FitEllipse_RANSAC(pnts, roi)
    
    return ellipse

        
#
# Overlay fitted pupil ellipse on original frame
#    
def DisplayPupilEllipse(frame, ellipse):

    # Ellipse color and line thickness
    color = (0,255,0)
    thickness = 1
    
    # Convert frame to RGB color
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
    
    # Overlay ellipse
    cv2.ellipse(frame_rgb, ellipse, color, thickness)
    
    # Display frame
    cv2.imshow('PupilFit', frame_rgb)
    
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
def WritePupilometry(pupil_out, t, ellipse, blink):
    
    # Unpack ellipse tuple
    (x0, y0), (bb, aa), phi_b_deg = ellipse
    
    # Pupil area corrected for viewing angle
    area = PupilArea(ellipse)
    
    # Write pupilometry line to file
    pupil_out.write('%0.3f,%0.1f,%0.1f,%0.1f,%d,\n' % (t, area, x0, y0, blink))
    
#
# Pupil area corrected for viewing angle
#
def PupilArea(ellipse):
    
    # Unpack ellipse tuple
    (x0,y0), (b,a), phi_b_deg = ellipse

    # Ellipse area assuming semi major axis is actual pupil radius
    return np.pi * a**2
