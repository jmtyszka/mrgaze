#!/opt/local/bin/python
#
# Video pupilometry functions
# - takes calibration and gaze video filenames as input
# - controls calibration and gaze estimation workflow
#
# USAGE : mrgaze.py <Calibration Video> <Gaze Video>
#
# AUTHOR : Mike Tyszka
# PLACE  : Caltech
# DATES  : 2014-05-07 JMT From scratch
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
import time
import cv2
import scipy.ndimage as spi
import numpy as np
import mrgaze.fitellipse as mrf
import mrgaze.utils as mru

def VideoPupilometry(data_dir, subj_sess, v_stub, config):
    """
    Perform pupil boundary ellipse fitting on entire video
    
    Arguments
    ----
    vin_path : string
        Video file name. Any format supported by ffmpeg.
    res_dir : string
        Pupilometry results directory
    config : 
        Analysis configuration parameters
    
    Returns
    ----
    status : boolean
        Completion status (True = successful)
    """
    
    # Output flags
    verbose = config.getboolean('OUTPUT', 'verbose')
    graphics = config.getboolean('OUTPUT', 'graphics')
    
    # Input video extension
    vin_ext = config.get('VIDEO', 'inputextension')
    vout_ext = config.get('VIDEO' ,'outputextension')    
    
    # Full video file paths
    ss_dir = os.path.join(data_dir, subj_sess)
    vin_path = os.path.join(ss_dir, 'videos', v_stub + vin_ext)
    vout_path = os.path.join(ss_dir, 'results', v_stub + '_pupils' + vout_ext)
    
    # Pupilometry CSV file output path
    pout_path = os.path.join(ss_dir, 'results', v_stub + '_pupils.csv')
    
    # Check that input video file exists
    if not os.path.isfile(vin_path):
        print('* %s does not exist - returning' % vin_path)
        return False
    
    # Set up the LBP cascade classifier
    mrclean_root = mru._package_root()
    LBP_path = os.path.join(mrclean_root, 'Cascade/cascade.xml')
    
    print('  Loading LBP cascade')
    cascade = cv2.CascadeClassifier(LBP_path)
    
    if cascade.empty():
        print('* LBP cascade is empty - mrgaze installation problem')
        return False
    
    #
    # Input video
    #
    print('  Opening input video stream')
        
    try:
        vin_stream = cv2.VideoCapture(vin_path)
    except:
        print('* Problem opening input video stream - skipping pupilometry')        
        return False
        
    if not vin_stream.isOpened():
        print('* Video input stream not opened - skipping pupilometry')
        return False
    
    # Get FPS from video file
    fps = vin_stream.get(cv2.cv.CV_CAP_PROP_FPS)
    
    # Total number of frames in video file
    nf = vin_stream.get(cv2.cv.CV_CAP_PROP_FRAME_COUNT)
    
    # Read preprocessed video frame from stream
    keep_going, frame, artifact = mru.LoadVideoFrame(vin_stream, config)
     
    # Get size of preprocessed frame for output video setup
    nx, ny = frame.shape[1], frame.shape[0]
    
    #
    # Output video
    #
    print('  Opening output video stream')
        
    # Output video codec (MP4V - poor quality compression)
    # TODO : Find a better multiplatform codec
    fourcc = cv2.cv.CV_FOURCC('m','p','4','v')

    try:
        vout_stream = cv2.VideoWriter(vout_path, fourcc, 30, (nx, ny), True)
    except:
        print('* Problem creating output video stream - skipping pupilometry')
        return False
        
    if not vout_stream.isOpened():
        print('* Output video not opened - skipping pupilometry')
        return False 

    # Open pupilometry CSV file to write
    try:
        pout_stream = open(pout_path, 'w')
    except:
        print('* Problem opening pupilometry CSV file - skipping pupilometry')
        return False

    #
    # Main Video Frame Loop
    #

    # Print verbose column headers
    if verbose:
        print('')
        print('  %10s %10s %10s %10s %10s %10s' % (
            'Time (s)',
            '% Done',
            'Area',
            'Blink',
            'Artifact',
            'FPS'))

    # Init frame counter
    fc = 0
    
    # Init processing timer
    t0 = time.time()
    
    while keep_going:
        
        # Current video time in seconds
        t = fc / fps
        
        # RGB version of preprocessed frame for output video
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
        
        # -------------------------------------
        # Pass this frame to pupilometry engine
        # -------------------------------------
        ellipse, roi_rect, blink = PupilometryEngine(frame, cascade)
                
        # Write data line to pupilometry CSV file
        area = WritePupilometry(pout_stream, t, ellipse, blink, artifact)
            
        if graphics:

            # Overlay ROI and pupil ellipse on RGB frame
            if not blink:
                p1, p2 = roi_rect                      
                cv2.rectangle(frame_rgb, p1, p2, (0,255,0), 1)
                cv2.ellipse(frame_rgb, ellipse, (128,255,255), 1)
            
            cv2.imshow('Tracking', frame_rgb)

            if cv2.waitKey(5) > 0:
                break
        
        # Write output video frame
        vout_stream.write(frame_rgb)

        # Read next frame (if available)
        keep_going, frame, artifact = mru.LoadVideoFrame(vin_stream, config)
        
        # Increment frame counter
        fc = fc + 1
        
        # Report processing FPS
        if verbose:
            if fc % 100 == 0:
                perc_done = fc / float(nf) * 100.0
                pfps = fc / (time.time() - t0)  
                print('  %10.1f %10.1f %10.1f %10d %10d %10.1f' % (
                    t, perc_done, area, blink, artifact, pfps))
    

    print('  Cleaning up')
    
    # Clean up
    cv2.destroyAllWindows()
    vin_stream.release()
    vout_stream.release()
    pout_stream.close()

    # Clean exit
    return True


def PupilometryEngine(img, cascade):
    """
    RANSAC ellipse fitting of pupil boundary with image support
    """
    
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


def SegmentPupil(roi):
    """
    Segment pupil within pupil-iris ROI'
    """

    # Intensity rescale to emphasize pupil
    # - assumes pupil is one of the darkest regions
    # - assumes pupil occupies between 5% and 50% of frame area
    roi = mru.RobustRescale(roi, (5,50))
            
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
    ellipse = mrf.FitEllipse_RANSAC(pnts, roi)
    
    return ellipse
        
#
# Overlay fitted pupil ellipse on original frame
#    
def DisplayPupilEllipse(frame, ellipse, roi_rect):

    # Ellipse color and line thickness
    thickness = 1
    
    # Convert frame to RGB color
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
    
    # Overlay ellipse
    cv2.ellipse(frame_rgb, ellipse, (0,255,0), thickness)
    
    # Overlay ROI rectangle
    cv2.rectangle(frame_rgb, roi_rect[0], roi_rect[1], (255,255,0), thickness)
    
    # Display frame
    cv2.imshow('PupilFit', frame_rgb)
    
    # Wait for key press
    cv2.waitKey()


def WritePupilometry(pupil_out, t, ellipse, blink, artifact):
    """
    Write pupilometry data line to file
    
    Arguments
    ----
    pupil_out : file stream
        Output pupilometry stream
    t : float
        Time from video start in seconds
    ellipse : ellipse tuple
        Ellipse parameters ((x0,y0),(a,b),theta)
    blink : boolean
        Blink flag
    artifact : boolean
        Artifact flag
    
    Returns
    ----
    area : float
        Corrected pupil area (AU)
    """
    
    # Unpack ellipse tuple
    (x0, y0), (bb, aa), phi_b_deg = ellipse
    
    # Pupil area corrected for viewing angle
    area = PupilArea(ellipse)
    
    # Write pupilometry line to file
    pupil_out.write('%0.3f,%0.1f,%0.1f,%0.1f,%d,%d,\n' % (t, area, x0, y0, blink, artifact))
    
    # Return corrected area
    return area
    

def PupilArea(ellipse):
    """
    Pupil area corrected for viewing angle
    """
    
    # Unpack ellipse tuple
    (x0,y0), (b,a), phi_b_deg = ellipse

    # Ellipse area assuming semi major axis is actual pupil radius
    return np.pi * a**2
