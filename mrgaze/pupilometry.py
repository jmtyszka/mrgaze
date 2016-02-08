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
import getpass
import cv2
import json
import numpy as np
from skimage import measure, morphology
from mrgaze import media, utils, fitellipse, improc
import scipy.ndimage as spi
from skimage.measure import label, regionprops
from mrgaze import media, utils, fitellipse, improc, config, calibrate, report

def LivePupilometry(data_dir):
    """
    Perform pupil boundary ellipse fitting on camera feed
    
    Arguments
    ----
    data_dir : string
        Root data directory path.
    cfg : 
        Analysis configuration parameters
    
    Returns
    ----
    pupils : boolean
        Completion status (True = successful)
    """

    # Load Configuration
    cfg = config.LoadConfig(data_dir)
    cfg_ts = time.time()
    
    # Output flags
    verbose   = cfg.getboolean('OUTPUT', 'verbose')
    overwrite = cfg.getboolean('OUTPUT','overwrite')
    
    # Video information
    # vin_ext = cfg.get('VIDEO', 'inputextension')
    vout_ext = cfg.get('VIDEO' ,'outputextension')
    # vin_fps = cfg.getfloat('VIDEO', 'inputfps')
    
    # Full video file paths
    hostname = os.uname()[1]
    username = getpass.getuser()
    ss_dir = os.path.join(data_dir, "%s_%s_%s" % (hostname, username, int(time.time())))
    vid_dir = os.path.join(ss_dir, 'videos')
    res_dir = os.path.join(ss_dir, 'results')
    # vin_path = os.path.join(vid_dir, v_stub + vin_ext)
    vout_path = os.path.join(vid_dir, 'gaze_pupils' + vout_ext)
    cal_vout_path = os.path.join(vid_dir, 'cal_pupils' + vout_ext)
    
    # Raw and filtered pupilometry CSV file paths
    cal_pupils_csv = os.path.join(res_dir, 'cal_pupils.csv')
    pupils_csv = os.path.join(res_dir, 'gaze_pupils.csv')
    
    
    # Check that output directory exists
    if not os.path.isdir(res_dir):
        os.makedirs(res_dir)
        print('* %s does not exist - creating' % res_dir)
    if not os.path.isdir(vid_dir):
        os.makedirs(vid_dir)
        print('* %s does not exist - creating' % vid_dir)

    
    # Set up the LBP cascade classifier
    LBP_path = os.path.join(utils._package_root(), 'Cascade/cascade.xml')
    
    print('  Loading LBP cascade')
    cascade = cv2.CascadeClassifier(LBP_path)
    
    if cascade.empty():
        print('* LBP cascade is empty - mrgaze installation problem')
        return False
        
    # Check for output CSV existance and overwrite flag
    if os.path.isfile(pupils_csv):
        print('+ Pupilometry output already exists - checking overwrite flag')
        if overwrite:
            print('+ Overwrite allowed - continuing')
        else:
            print('+ Overwrite forbidden - skipping pupilometry')
            return True
    
    #
    # Camera Input
    #
    print('  Opening camera stream')
        
    try:
        vin_stream = cv2.VideoCapture(0)
    except:
        print('* Problem opening input video stream - skipping pupilometry')    
        return False

    while not vin_stream.isOpened():
        key = cv2.waitKey(500)
        if key == 27:
            print "User Abort."
            break

    if not vin_stream.isOpened():
        print('* Video input stream not opened - skipping pupilometry')
        return False
    
    # Video FPS from metadata
    # TODO: may not work with Quicktime videos
    # fps = vin_stream.get(cv2.cv.CV_CAP_PROP_FPS)
    fps = cfg.getfloat('CAMERA', 'fps')

    # desired time between frames in milliseconds
    time_bw_frames = 1000.0 / fps
    
    vin_stream.set(cv2.cv.CV_CAP_PROP_FRAME_WIDTH, 320)
    vin_stream.set(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT, 240)
    vin_stream.set(cv2.cv.CV_CAP_PROP_FPS, 30)
    
    # Total number of frames in video file
    # nf = vin_stream.get(cv2.cv.CV_CAP_PROP_FRAME_COUNT)
    
    # print('  Video has %d frames at %0.3f fps' % (nf, vin_fps))
    
    # Read first preprocessed video frame from stream
    keep_going, frame, art_power = media.LoadVideoFrame(vin_stream, cfg)
     
    # Get size of preprocessed frame for output video setup
    nx, ny = frame.shape[1], frame.shape[0]
    
    # By default we start in non-calibration mode
    # switch between gaze/cal modes by pressing key "c"
    do_cal = False

    while keep_going:
        if do_cal == False:
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
                pupils_stream = open(pupils_csv, 'w')
            except:
                print('* Problem opening pupilometry CSV file - skipping pupilometry')
                return False

            #
            # Main Video Frame Loop
            #
            
            # Print verbose column headers
            if verbose:
                print('')
                print('  %10s %10s %10s %10s %10s' % (
                    'Time (s)', 'Area', 'Blink', 'Artifact', 'FPS'))

            # Init frame counter
            fc = 0
    
            # Init processing timer
            t0 = time.time()
            t = t0
    
            while keep_going:
                # check whether config file has been updated, reload of that is the case
                if fc % 30 == 0:
                    cfg_mtime = os.path.getmtime(os.path.join(data_dir, 'mrgaze.cfg'))
                    if cfg_mtime > cfg_ts:
                        print "Updating Configuration"
                        cfg = config.LoadConfig(data_dir)
                        cfg_ts = time.time()

                # Current video time in seconds
                t = time.time()
        
                # -------------------------------------
                # Pass this frame to pupilometry engine
                # -------------------------------------
                # b4_engine = time.time()
                pupil_ellipse, roi_rect, blink, glint, frame_rgb = PupilometryEngine(frame, cascade, cfg)
                # print "Enging took %s ms" % (time.time() - b4_engine)
        
                # Derive pupilometry parameters
                px, py, area = PupilometryPars(pupil_ellipse, glint, cfg)
        
                # Write data line to pupilometry CSV file
                pupils_stream.write(
                    '%0.4f,%0.3f,%0.3f,%0.3f,%d,%0.3f,\n' %
                    (t, area, px, py, blink, art_power)
                )
                                
                # Write output video frame
                vout_stream.write(cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB))

                # Read next frame (if available)
                # if verbose:
                #     b4_frame = time.time()
                keep_going, frame, art_power = media.LoadVideoFrame(vin_stream, cfg)
                #if verbose:
                # print "Time to load frame: %s" % (time.time() - b4_frame)
        
                # Increment frame counter
                fc = fc + 1

                # Report processing FPS
                if verbose:
                    if fc % 100 == 0:
                        pfps = fc / (time.time() - t0)  
                        print('  %10.1f %10.1f %10d %10.3f %10.1f' % (
                            t, area, blink, art_power, pfps))
                        t0 = time.time()
                        fc = 0

                # wait whether user pressed esc to exit the experiment
                key = cv2.waitKey(1)
                if key == 27 or key == 1048603:
                    # Clean up
                    vout_stream.release()
                    pupils_stream.close()
                    keep_going = False
                elif key == 99:
                    # Clean up
                    vout_stream.release()
                    pupils_stream.close()
                    do_cal = True
                    print "Starting calibration."
                    break
        else: # do calibration
            #
            # Output video
            #
            print('  Opening output video stream')
        
            # Output video codec (MP4V - poor quality compression)
            # TODO : Find a better multiplatform codec
            fourcc = cv2.cv.CV_FOURCC('m','p','4','v')

            try:
                cal_vout_stream = cv2.VideoWriter(cal_vout_path, fourcc, 30, (nx, ny), True)
            except:
                print('* Problem creating output video stream - skipping pupilometry')
                return False
        
            if not cal_vout_stream.isOpened():
                print('* Output video not opened - skipping pupilometry')
                return False 

            # Open pupilometry CSV file to write
            try:
                cal_pupils_stream = open(cal_pupils_csv, 'w')
            except:
                print('* Problem opening pupilometry CSV file - skipping pupilometry')
                return False

            #
            # Main Video Frame Loop
            #
            
            # Print verbose column headers
            if verbose:
                print('')
                print('  %10s %10s %10s %10s %10s' % (
                    'Time (s)', 'Area', 'Blink', 'Artifact', 'FPS'))

            # Init frame counter
            fc = 0
    
            # Init processing timer
            t0 = time.time()
            t = t0
            while keep_going:
                # check whether config file has been updated, reload of that is the case
                if fc % 30 == 0:
                    cfg_mtime = os.path.getmtime(os.path.join(data_dir, 'mrgaze.cfg'))
                    if cfg_mtime > cfg_ts:
                        print "Updating Configuration"
                        cfg = config.LoadConfig(data_dir)
                        cfg_ts = time.time()

                # Current video time in seconds
                t = time.time()
        
                # -------------------------------------
                # Pass this frame to pupilometry engine
                # -------------------------------------
                # b4_engine = time.time()
                pupil_ellipse, roi_rect, blink, glint, frame_rgb = PupilometryEngine(frame, cascade, cfg)
                # print "Engine took %s ms" % (time.time() - b4_engine)
        
                # Derive pupilometry parameters
                px, py, area = PupilometryPars(pupil_ellipse, glint, cfg)
        
                # Write data line to pupilometry CSV file
                cal_pupils_stream.write(
                    '%0.4f,%0.3f,%0.3f,%0.3f,%d,%0.3f,\n' %
                    (t, area, px, py, blink, art_power)
                )
                                
                # Write output video frame
                cal_vout_stream.write(cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB))

                # Read next frame (if available)
                # if verbose:
                #     b4_frame = time.time()
                keep_going, frame, art_power = media.LoadVideoFrame(vin_stream, cfg)
                #if verbose:
                # print "Time to load frame: %s" % (time.time() - b4_frame)
        
                # Increment frame counter
                fc = fc + 1

                # Report processing FPS
                if verbose:
                    if fc % 100 == 0:
                        pfps = fc / (time.time() - t0)  
                        print('  %10.1f %10.1f %10d %10.3f %10.1f' % (
                            t, area, blink, art_power, pfps))
                        t0 = time.time()
                        fc = 0

                # wait whether user pressed esc to exit the experiment
                key = cv2.waitKey(1)
                if key == 27 or key == 1048603:
                    keep_going = False
                    # Clean up
                    cal_vout_stream.release()
                    cal_pupils_stream.close()
                elif key == 118:
                    do_cal = False
                    print "Stopping calibration."
                    # Clean up
                    cal_vout_stream.release()
                    cal_pupils_stream.close()
                    break

            print('  Create calibration model')
            C, central_fix = calibrate.AutoCalibrate(res_dir, cfg)
            
            if not C.any():
                print('* Empty calibration matrix detected - skipping')
    try:
        print('  Calibrate pupilometry')
        calibrate.ApplyCalibration(ss_dir, C, central_fix, cfg)
    except UnboundLocalError:
        print('  No calibration data found')

    cv2.destroyAllWindows()
    vin_stream.release()

    print('')
    print('  Generate Report')
    print('  ---------------')
    report.WriteReport(ss_dir, cfg)

    # Return pupilometry timeseries
    return t, px, py, area, blink, art_power

def VideoPupilometry(data_dir, subj_sess, v_stub, cfg):
    """
    Perform pupil boundary ellipse fitting on entire video
    
    Arguments
    ----
    data_dir : string
        Root data directory path.
    subj_sess : string
        Subject/Session name used for subdirectory within data_dir
    v_stub : string
        Video filename stub, eg 'cal' or 'gaze'
    cfg : 
        Analysis configuration parameters
    
    Returns
    ----
    pupils : boolean
        Completion status (True = successful)
    """
    
    # Output flags
    verbose   = cfg.getboolean('OUTPUT', 'verbose')
    overwrite = cfg.getboolean('OUTPUT','overwrite')
    
    # Video information
    vin_ext = cfg.get('VIDEO', 'inputextension')
    vout_ext = cfg.get('VIDEO' ,'outputextension')
    vin_fps = cfg.getfloat('VIDEO', 'inputfps')
    
    # Full video file paths
    ss_dir = os.path.join(data_dir, subj_sess)
    vid_dir = os.path.join(ss_dir, 'videos')
    res_dir = os.path.join(ss_dir, 'results')
    vin_path = os.path.join(vid_dir, v_stub + vin_ext)
    vout_path = os.path.join(res_dir, v_stub + '_pupils' + vout_ext)
    
    # Raw and filtered pupilometry CSV file paths
    pupils_csv = os.path.join(res_dir, v_stub + '_pupils.csv')
    
    # Check that input video file exists
    if not os.path.isfile(vin_path):
        print('* %s does not exist - returning' % vin_path)
        return False
    
    # Set up the LBP cascade classifier
    LBP_path = os.path.join(utils._package_root(), 'Cascade/cascade.xml')
    
    print('  Loading LBP cascade')
    cascade = cv2.CascadeClassifier(LBP_path)
    
    if cascade.empty():
        print('* LBP cascade is empty - mrgaze installation problem')
        return False
        
    # Check for output CSV existance and overwrite flag
    if os.path.isfile(pupils_csv):
        print('+ Pupilometry output already exists - checking overwrite flag')
        if overwrite:
            print('+ Overwrite allowed - continuing')
        else:
            print('+ Overwrite forbidden - skipping pupilometry')
            return True
    
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
    
    # Video FPS from metadata
    # TODO: may not work with Quicktime videos
    # fps = vin_stream.get(cv2.cv.CV_CAP_PROP_FPS)
    
    # Total number of frames in video file
    nf = vin_stream.get(cv2.CAP_PROP_FRAME_COUNT)
    
    print('  Video has %d frames at %0.3f fps' % (nf, vin_fps))
    
    # Read first preprocessed video frame from stream
    keep_going, frame, art_power = media.LoadVideoFrame(vin_stream, cfg)
     
    # Get size of preprocessed frame for output video setup
    nx, ny = frame.shape[1], frame.shape[0]

    #
    # Output video
    #
    print('  Opening output video stream')
        
    # Output video codec (MP4V - poor quality compression)
    fourcc = cv2.VideoWriter_fourcc('m','p','4','v')

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
        pupils_stream = open(pupils_csv, 'w')
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
            'Time (s)', '% Done', 'Area', 'Blink', 'Artifact', 'FPS'))

    # Init frame counter
    fc = 0
    
    # Init processing timer
    t0 = time.time()
    
    while keep_going:
        
        # Current video time in seconds
        t = fc / vin_fps
        
        # -------------------------------------
        # Pass this frame to pupilometry engine
        # -------------------------------------
        pupil_ellipse, roi_rect, blink, glint, frame_rgb = PupilometryEngine(frame, cascade, cfg)
        
        # Derive pupilometry parameters
        px, py, area = PupilometryPars(pupil_ellipse, glint, cfg)
        
        # Write data line to pupilometry CSV file
        pupils_stream.write(
            '%0.3f,%0.3f,%0.3f,%0.3f,%d,%0.3f,\n' %
            (t, area, px, py, blink, art_power)
        )
            
        # Write output video frame
        vout_stream.write(frame_rgb)

        # Read next frame (if available)
        keep_going, frame, art_power = media.LoadVideoFrame(vin_stream, cfg)
        
        # Increment frame counter
        fc = fc + 1
        
        # Report processing FPS
        if verbose:
            if fc % 100 == 0:
                perc_done = fc / float(nf) * 100.0
                pfps = fc / (time.time() - t0)  
                print('  %10.1f %10.1f %10.1f %10d %10.3f %10.1f' % (
                    t, perc_done, area, blink, art_power, pfps))


    # Clean up
    cv2.destroyAllWindows()
    vin_stream.release()
    vout_stream.release()
    pupils_stream.close()

    # Return pupilometry timeseries
    return t, px, py, area, blink, art_power


def PupilometryEngine(frame, cascade, cfg):
    """
    RANSAC ellipse fitting of pupil boundary with image support
    
    Arguments
    ----
    frame : 2D numpy uint8 array
        Video frame        
    cascade : opencv LBP cascade object
        Pupil classifier cascade
    cfg : configuration object
        Analysis pipeline configuration parameters
        
    Returns
    ----
    pupil_ellipse : float tuple
        Ellipse parameters (x0, y0), (a, b), phi
    roi_rect : float tuple
        Pupil ROI rectangle (x0, y0), (x1, y1)
    blink : boolean
        Blink flag (no pupil detected)
    """
    
    # Unset blink flag
    blink = False
    
    # Frame width and height in pixels
    frw, frh = frame.shape[1], frame.shape[0]
    
    # RGB version of preprocessed frame for later use
    frame_rgb =  cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
    
    # Init ROI to whole frame
    # Note (row, col) = (y, x) for shape
    x, y, w, h = 0, 0, frw, frh

    # Shall we use the classifier at all, or whole frame?
    if cfg.getboolean('PUPILDETECT', 'enabled'):
        
        min_neighbors = cfg.getint('PUPILDETECT','specificity')
        scale_factor  = cfg.getfloat('PUPILDETECT','scalefactor')
        
        # Find pupils in frame
        pupils = cascade.detectMultiScale(image=frame,
                                      scaleFactor=scale_factor,
                                      minNeighbors=min_neighbors)
        
        # Count detected pupil candidates
        n_pupils = len(pupils)
    
        if n_pupils > 0:
                
            # Use largest pupil candidate (works most of the time)
            # TODO: refine this selection somehow
            sizes = np.sqrt(pupils[:,2] * pupils[:,3])
            best_pupil = sizes.argmax()
        
            # Get ROI info for largest pupil
            x, y, w, h = pupils[best_pupil,:]

        else:

            x, y, w, h = 0, 0, 0, 0            
            blink = True
            
    else:
            
        # LBP pupil detection is off
        
        # Load manual ROI center and width (normalized units)
        xn, yn, wn = json.loads(cfg.get('PUPILDETECT', 'manualroi'))
        
        # Check for non-zero manual ROI definition
        if wn > 0.0:
            blink = False
        else:
            blink = True
            
        # Convert from normalized units to pixels at current resampled size
        # Note: ROI rectangle origin is upper left corner
        x, y, w, h = int((xn - 0.5*wn) * frw), int((yn-0.5*wn) * frh), int(wn * frw), int(wn * frw)
        
            
    # Init pupil and glint parameters
    pupil_ellipse = ((np.nan, np.nan), (np.nan, np.nan), np.nan)
    roi_rect = (np.nan, np.nan), (np.nan, np.nan)
    glint_center = (np.nan, np.nan)
            
    if not blink:

        # Define ROI rect
        roi_rect = (x,y), (x+w,y+h)
        
        # Extract pupil ROI (note row,col indexing of image array)
        roi = frame[y:y+h, x:x+w]
        
        ###################
        # BEGIN ENGINE CORE
    
        # Find and remove primary glint in ROI (assumes single illumination source)
        glint, glint_mask, roi_noglint = FindRemoveGlint(roi, cfg)
        
        if np.isnan(glint[0]):
            blink = True

        # Segment pupil within ROI
        pupil_bw, pupil_labels, roi_rescaled = SegmentPupil(roi_noglint, cfg)
        
        if pupil_bw.sum() > 0:
     
            # Fit ellipse to pupil boundary - returns ellipse parameter tuple
            ell = FitPupil(pupil_bw, roi, cfg)
        
            # Add ROI offset to ellipse center and glint
            pupil_ellipse = (x + ell[0][0], y + ell[0][1]),ell[1], ell[2]
            glint_center = (x + glint[0], y + glint[1])
            
        else:
            blink = True
    
        # END ENGINE CORE
        ##################
        
        # Check for unusually high eccentricity
        if fitellipse.Eccentricity(pupil_ellipse) > 0.5:
            blink = True
            
        
    if not blink:

        # Overlay ROI, pupil ellipse and pseudo-glint on background RGB frame
        frame_rgb = OverlayPupil(frame_rgb, pupil_ellipse, roi_rect, glint_center)
        

    if cfg.getboolean('OUTPUT', 'graphics'):
        
            if not blink:
                
                # Rescale and cast label images to uint8/ubyte
                pupil_labels = utils._touint8(pupil_labels)
                glint_mask = utils._touint8(glint_mask)
                
                # Create quad montage preprocessing stages in pupil/glint detection
                A = np.hstack( (roi, roi_rescaled) )
                B = np.hstack( (pupil_labels, glint_mask) )
                
                # Apply colormaps
                A_rgb = cv2.applyColorMap(A, cv2.COLORMAP_BONE)
                B_rgb = cv2.applyColorMap(B, cv2.COLORMAP_JET)
                
                quad_rgb = np.vstack( (A_rgb, B_rgb) )

                # Resample montage to 256 rows
                ny, nx, nc = quad_rgb.shape
                new_ny, new_nx = 256, int(256.0 / ny * nx)
                quad_up_rgb = cv2.resize(quad_rgb, dsize=(new_nx, new_ny),
                                         interpolation=cv2.INTER_NEAREST)

            else:
                
                # Create blank 256 x 256 RGB image
                quad_up_rgb = np.zeros((256,256,3), dtype="uint8")
            
            # Resample overlay image to 256 rows
            ny, nx, nc = frame_rgb.shape
            new_ny, new_nx = 256, int(256.0 / ny * nx)
            frame_up_rgb = cv2.resize(frame_rgb, dsize=(new_nx, new_ny), interpolation=cv2.INTER_NEAREST)

            # Montage preprocessing and final overlay into single RGB image
            montage_rgb = np.hstack( (quad_up_rgb, frame_up_rgb) )

            cv2.imshow('Pupilometry', montage_rgb)
            cv2.waitKey(5)

    return pupil_ellipse, roi_rect, blink, glint_center, frame_rgb      


def SegmentPupil(roi, cfg):
    """
    Segment pupil within pupil-iris ROI
    ROI should have been rescaled 
    
    Arguments
    ----
    roi : 2D numpy uint8 array
        Grayscale image of pupil-iris region
    cfg : configuration object
        Analysis configuration parameters
    
    Returns
    ----
    pupil_bw : 2D numpy uint8 array
        Binary thresholded version of ROI image
    roi_rescaled : 2D numpy uint8 array
        Rescaled version of ROI with bright regions saturated
    """

    # Get segmentation parameters
    method = cfg.get('PUPILSEG','method')
        
    # Estimate pupil diameter in pixels
    pupil_d = int(cfg.getfloat('PUPILSEG','pupildiameterperc') * roi.shape[0] / 100.0)
    
    # Estimate pupil area in pixels
    pupil_A = np.pi * (pupil_d/2.0)**2
    
    # Pupil area lower and upper bounds (/3, *3)
    A_min, A_max = pupil_A / 3.0, pupil_A * 3.0
    
    # Saturate image pixels above upper limit on pupil area
    roi_rescaled = improc.RobustRescale(roi, (0.0, 25.0))
    
    if method == 'manual':

        # Convert percent threshold to pixel intensity threshold
        thresh = int(cfg.getfloat('PUPILSEG','pupilthresholdperc') / 100.0 * 255.0)
        
        # Manual thresholding - ideal for real time ET with UI thresh control
        _, blobs = cv2.threshold(roi_rescaled, thresh, 255, cv2.THRESH_BINARY_INV) 
    
    elif method == 'otsu':
        
        # Binary threshold of rescaled ROI using Ostu's method
        thresh, blobs = cv2.threshold(roi_rescaled, 128, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    
    # Morphological opening removes small objects
    # kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
    # blobs = cv2.morphologyEx(blobs, cv2.MORPH_OPEN, kernel)
    
    # Label connected regions
    pupil_labels = measure.label(blobs, background=0)

    # Region properties
    pupil_props = measure.regionprops(pupil_labels)
    
    # Init maximum circularity
    C_max = 0.0
    
    # Init best pupil label
    pupil_label = -1
    
    for props in pupil_props:
        
        # Extract region properties
        L, A, P = props.label, props.area, props.perimeter

        # Circularity C = 4 pi A / P**2
        if P > 0.0:
            C = 4.0 * np.pi * A / (P**2)
        else:
            C = 0.0

        if A > A_min and A < A_max:

            if C > C_max:
                C_max = C
                pupil_label = L

    # Extract most pupil-like blob
    if pupil_label > 0:

        pupil_bw = np.uint8(pupil_labels == pupil_label)

        # Replace pupil blob with filled convex hull
        pupil_bw = np.uint8(morphology.convex_hull_image(pupil_bw))

    else:

        pupil_bw = np.zeros_like(blobs)
        
    return pupil_bw, pupil_labels, roi_rescaled
    

def FindRemoveGlint(roi, cfg):
    '''
    Locate small bright region roughly centered in ROI
    This function should be called before any major preprocessing of the frame.
    The ROI should be unscaled and unblurred, since we assume glints
    are saturated and have maximum intensity (255 in uint8)
    
    Arguments
    ----
    roi : 2D numpy uint8 array
        Pupil/iris ROI image
    cfg : configuration object
        Configuration parameters including fractional glint diameter estimate
    
    Returns
    ----
    glint : float array
        N x 2 array of glint centroids
    glint_mask : 2D numpy uint8 array
        Bright region mask used by glint removal
    roi_noglint : 2D numpy uint8 array
        Pupil/iris ROI without small bright areas
    '''

    # ROI dimensions and center
    ny, nx = roi.shape
    roi_cx, roi_cy = nx/2.0, ny/2.0

    # Estimated glint diameter in pixels
    glint_d = int(cfg.getfloat('PUPILSEG','glintdiameterperc') * nx / 100.0)

    # Glint diameter should be >= 1 pixel
    if glint_d < 1:
        glint_d = 1
    
    # Reasonable upper and lower bounds on glint area (x3, /3)
    glint_A = np.pi * (glint_d / 2.0)**2
    A_min, A_max = glint_A / 3.0, glint_A * 3.0

    # Find bright pixels in full scale uint8 image (ie value > 250)
    bright = np.uint8(roi > 250)
    
    # Label connected regions (blobs)
    bright_labels = measure.label(bright)
    
    # Get region properties for all bright blobs in mask
    bright_props = measure.regionprops(bright_labels)
        
    # Init glint parameters
    r_min = np.Inf
    glint_label = -1
    glint = (np.nan, np.nan)
    glint_mask = np.zeros_like(roi, dtype="uint8")
    roi_noglint = roi.copy()
        
    # Find closest blob to ROI center within glint area range
    for props in bright_props:
    
        # Blob area in pixels^2    
        A = props.area
        
        # Only accept blobs with area in glint range
        if A > A_min and A < A_max:

            # Check distance from ROI center            
            cy, cx = props.centroid  # (row, col)
            r = np.sqrt((cx-roi_cx)**2 + (cy-roi_cy)**2)
            
            if r < r_min:
                r_min = r
                glint_label = props.label
                glint = (cx, cy)
    
    
    if glint_label > 0:
    
        # Construct glint mask
        glint_mask = np.uint8(bright_labels == glint_label)
            
        # Dilate glint mask
        k = glint_d * 3;
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k,k))
        glint_mask = cv2.morphologyEx(glint_mask, cv2.MORPH_DILATE, kernel)
    
        # Inpaint dilated glint in ROI
        roi_noglint = cv2.inpaint(roi, glint_mask, 3, cv2.INPAINT_TELEA)
        
    
    return glint, glint_mask, roi_noglint


def FitPupil(bw, roi, cfg):
    '''
    Fit ellipse to pupil-iris boundary in segmented ROI
    
    Arguments
    ----
    bw : 2D numpy uint8 array
        Binary thresholded version of pupil ROI (from SegmentPupil)
    roi : 2D scalar array
        Grayscale image of pupil-iris region
    cfg : configuration object
    
    Returns
    ----
    ellipse : tuple of tuples
        Fitted ellipse parameters ((x0, y0), (a,b), theta)
    '''
     
    # Identify edge pixels using Canny filter
    roi_edges = cv2.Canny(bw, 0, 1)
    
    # Find all edge point coordinates
    pnts = np.transpose(np.nonzero(roi_edges))
    
    # Swap columns - pnts are (row, col) and need to be (x,y)
    pnts[:,[0,1]] = pnts[:,[1,0]]
    
    # Ellipse fitting to edge points

    # Methods supported:
    # 1. RANSAC with image support (requires grascale ROI)
    # 2. RANSAC without image support
    # 3. Robust least-squares
    # 4. Least-squares (requires clean segmentation)

    # Extract ellipse fitting parameters
    method = cfg.get('PUPILFIT','method')
    max_itts = cfg.getint('PUPILFIT','maxiterations')
    max_refines = cfg.getint('PUPILFIT','maxrefinements')
    max_perc_inliers = cfg.getfloat('PUPILFIT','maxinlierperc')

    if method == 'RANSAC_SUPPORT':
        ellipse = fitellipse.FitEllipse_RANSAC_Support(pnts, roi, cfg, max_itts, max_refines, max_perc_inliers)        
    
    elif method == 'RANSAC':
        ellipse = fitellipse.FitEllipse_RANSAC(pnts, roi, cfg, max_itts, max_refines, max_perc_inliers)

    elif method == 'ROBUST_LSQ':
        ellipse = fitellipse.FitEllipse_RobustLSQ(pnts, roi, cfg, max_refines, max_perc_inliers)
        
    elif method == 'LSQ':
        ellipse = fitellipse.FitEllipse_LeastSquares(pnts, roi, cfg)                

    else:
        print('* Unknown ellipse fitting method: %s' % method)
        ellipse = ((0,0),(0,0),0)
    
    return ellipse
        

def OverlayPupil(frame_rgb, ellipse, roi_rect, glint):
    """
    Overlay fitted pupil ellipse and ROI on original frame
    """

    # line thickness
    thickness = 1

    # Ellipse color
    ellipse_color = (0,255,0)
    
    # ROI rectangle color
    roi_color = (255,255,0)
    
    # Glint marker color
    glint_color = (0,0,255)

    # Pupil center cross hair
    (x0, y0), (a, b), phi = ellipse
    
    # Cross hair size proportional to mean ellipse axis
    r = (a + b) / 20.0
    
    # Endpoints of major and minor axes
    a0 = int(x0 + r), int(y0)
    a1 = int(x0 - r), int(y0)
    b0 = int(x0), int(y0 + r)
    b1 = int(x0), int(y0 - r)
    
    # Overlay ellipse
    cv2.ellipse(frame_rgb, ellipse, ellipse_color, thickness)
    
    # Overlay ellipse axes
    cv2.line(frame_rgb, a0, a1, ellipse_color, thickness)
    cv2.line(frame_rgb, b0, b1, ellipse_color, thickness)
    
    # Overlay ROI rectangle
    cv2.rectangle(frame_rgb, roi_rect[0], roi_rect[1], roi_color, thickness)
    
    # Overlay glint centroid
    px, py = int(glint[0]), int(glint[1])
    cv2.circle(frame_rgb, (px, py), 3, glint_color)

    return frame_rgb
    

def ReadPupilometry(pupils_csv):
    '''
    Read text pupilometry results from CSV file
    
    Returns
    ----
    p : 2D numpy float array
        Timeseries in columns. Column order is:
        0 : Time (s)
        1 : Corrected pupil area (AU)
        2 : Pupil center in x (pixels)
        3 : Pupil center in y (pixels)
        4 : Blink flag (pupil not found)
        5 : MR artifact power
    '''
    
    # Read time series in rows
    return np.genfromtxt(pupils_csv, delimiter=',')


def PupilometryPars(ellipse, glint, cfg):
    """
    Extract pupil center and corrected area. Pupil center is reported
    in voxels relative to top left of frame (video origin) or the glint
    centroid (glint origin)
    
    Arguments
    ----
    ellipse : tuple
        Ellipse parameter tuple
    glint : tuple
        Glint center in video frame coordinates
    
    Returns
    ----
    px, py : floats
        Pupil center in video or glint frame of reference
    area : float
        Pupil area (sq voxels) corrected for viewing angle
    """
    
    # Unpack ellipse tuple
    (px, py), (bb, aa), phi_b_deg = ellipse
    
    # Adjust pupil center for glint location
    if cfg.get('ARTIFACTS','motioncorr') == 'glint':
        gx, gy = glint
        px = px - gx
        py = py - gy
    
    # Pupil area corrected for viewing angle
    # Assumes semi-major axis is actual pupil radius
    area = np.pi * aa**2
    
    return px, py, area

  
def FilterPupilometry(pupils_csv, pupils_filt_csv):
    '''
    DEPRECATED: Temporally filter all pupilometry timeseries
    '''

    if not os.path.isfile(pupils_csv):
        print('* Raw pupilometry CSV file missing - returning')
        return False
        
    # Read raw pupilometry data
    p = ReadPupilometry(pupils_csv)
    
    # Sampling time (s)
    dt = p[1,0] - p[0,0]
    
    # Kernel widths for each metric
    k_area  = utils._forceodd(0.25 / dt)
    k_pupil = 3
    k_blink = utils._forceodd(0.25 / dt)
    k_art   = utils._forceodd(1.0 / dt)
    
    # Moving median filter
    pf = p.copy()
    pf[:,1] = utils._nanmedfilt(p[:,1], k_area)
    pf[:,2] = utils._nanmedfilt(p[:,2], k_pupil) # Pupil x
    pf[:,3] = utils._nanmedfilt(p[:,3], k_pupil) # Pupil y
    
    # Blink filter
    pf[:,4] = utils._nanmedfilt(p[:,8], k_blink)
    
    # Artifact power
    pf[:,5] = utils._nanmedfilt(pf[:,9], k_art)
    
    # Write filtered timeseries to new CSV file in results directory
    np.savetxt(pupils_filt_csv, pf, fmt='%.6f', delimiter=',')
    
    # Clean return
    return True
