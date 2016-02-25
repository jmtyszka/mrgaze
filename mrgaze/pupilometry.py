#!/usr/bin/env python
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
#          2016-02-22 JMT Update print for python3. Remove unused vars, imports
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
from mrgaze import media, utils, config, calibrate, report, engine


def LivePupilometry(data_dir, live_eyetracking=False):
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

    # If user did not provide a root data directory, we use HOME/mrgaze/<hostname>_<username>_<timestamp>
    if data_dir == '':
        data_dir = os.path.join(os.getenv("HOME"), 'mrgaze')

        # Full video file paths
        hostname = os.uname()[1]
        username = getpass.getuser()

        ss_dir = os.path.join(data_dir, "%s_%s_%s" % (hostname, username, utils.mktimestamp()))
    else:
        ss_dir = data_dir

    # Load Configuration
    cfg = config.LoadConfig(data_dir)
    cfg_ts = time.time()

    # Output flags
    verbose   = cfg.getboolean('OUTPUT', 'verbose')
    overwrite = cfg.getboolean('OUTPUT', 'overwrite')

    # Video information
    vin_ext = cfg.get('VIDEO', 'inputextension')
    vout_ext = cfg.get('VIDEO' ,'outputextension')
    # vin_fps = cfg.getfloat('VIDEO', 'inputfps')

    # Flag for freeze frame
    freeze_frame = False

    vid_dir = os.path.join(ss_dir, 'videos')
    res_dir = os.path.join(ss_dir, 'results')

    vout_path = os.path.join(res_dir, 'gaze_pupils' + vout_ext)
    cal_vout_path = os.path.join(res_dir, 'cal_pupils' + vout_ext)

    raw_vout_path = os.path.join(vid_dir, 'gaze' + vout_ext)
    raw_cal_vout_path = os.path.join(vid_dir, 'cal' + vout_ext)

    # if we do not dolive eye-tracking, read in output of previous live eye-tracking
    if not live_eyetracking:
        vin_path = raw_vout_path
        cal_vin_path = raw_cal_vout_path
    else:
        vin_path = 0

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
        if not live_eyetracking:
            vin_stream = cv2.VideoCapture(vin_path)
            cal_vin_stream = cv2.VideoCapture(cal_vin_path)
        else:
            vin_stream = cv2.VideoCapture(vin_path)
            cal_vin_stream = vin_stream

    except:
        print('* Problem opening input video stream - skipping pupilometry')
        return False


    while not vin_stream.isOpened():
        print("Waiting for Camera.")
        key = utils._waitKey(500)
        if key == 'ESC':
            print("User Abort.")
            break

    if not vin_stream.isOpened():
        print('* Video input stream not opened - skipping pupilometry')
        return False

    if not cal_vin_stream.isOpened():
        print('* Calibration video input stream not opened - skipping pupilometry')
        return False

    # Video FPS from metadata
    # TODO: may not work with Quicktime videos
    # fps = vin_stream.get(cv2.cv.CV_CAP_PROP_FPS)
    # fps = cfg.getfloat('CAMERA', 'fps')

    # Desired time between frames in milliseconds
    # time_bw_frames = 1000.0 / fps

    vin_stream.set(cv2.cv.CV_CAP_PROP_FRAME_WIDTH, 320)
    vin_stream.set(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT, 240)
    vin_stream.set(cv2.cv.CV_CAP_PROP_FPS, 30)

    # Total number of frames in video file
    # nf = vin_stream.get(cv2.cv.CV_CAP_PROP_FRAME_COUNT)

    # print('  Video has %d frames at %0.3f fps' % (nf, vin_fps))

    # Read first preprocessed video frame from stream
    keep_going, frame_orig = media.LoadVideoFrame(vin_stream, cfg)
    if keep_going:
        frame, art_power = media.Preproc(frame_orig, cfg)
    else:
        art_power = 0.0

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
            if live_eyetracking:
                print('  Opening output video stream')

                # Output video codec (MP4V - poor quality compression)
                # TODO : Find a better multiplatform codec
                fourcc = cv2.cv.CV_FOURCC('m','p','4','v')


                try:
                    vout_stream = cv2.VideoWriter(vout_path, fourcc, 30, (nx, ny), True)
                except:
                    print('* Problem creating output video stream - skipping pupilometry')
                    return False

                try:
                    raw_vout_stream = cv2.VideoWriter(raw_vout_path, fourcc, 30, (nx, ny), True)
                except:
                    print('* Problem creating raw output video stream - skipping pupilometry')
                    return False

                if not vout_stream.isOpened():
                    print('* Output video not opened - skipping pupilometry')
                    return False

                if not raw_vout_stream.isOpened():
                    print('* Raw output video not opened - skipping pupilometry')
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
                        print("Updating Configuration")
                        cfg = config.LoadConfig(data_dir)
                        cfg_ts = time.time()

                # Current video time in seconds
                t = time.time()

                # -------------------------------------
                # Pass this frame to pupilometry engine
                # -------------------------------------
                # b4_engine = time.time()
                pupil_ellipse, roi_rect, blink, glint, frame_rgb = engine.PupilometryEngine(frame, cascade, cfg)
                # print "Enging took %s ms" % (time.time() - b4_engine)

                # Derive pupilometry parameters
                px, py, area = engine.PupilometryPars(pupil_ellipse, glint, cfg)

                # Write data line to pupilometry CSV file
                pupils_stream.write(
                    '%0.4f,%0.3f,%0.3f,%0.3f,%d,%0.3f,\n' %
                    (t, area, px, py, blink, art_power)
                )

                # Write output video frame
                vout_stream.write(frame_rgb)

                # Write raw output video frame
                if live_eyetracking:
                    raw_vout_stream.write(frame_orig)

                # Read next frame, unless we want to figure out the correct settings for this frame
                if not freeze_frame:
                    keep_going, frame_orig = media.LoadVideoFrame(vin_stream, cfg)
                if keep_going:
                    frame, art_power = media.Preproc(frame_orig, cfg)
                else:
                    art_power = 0.0

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
                key = utils._waitKey(1)
                if key == 'ESC':
                    # Clean up
                    if live_eyetracking:
                        raw_vout_stream.release()
                    vout_stream.release()
                    pupils_stream.close()
                    keep_going = False
                elif key == 'c':
                    # Clean up
                    if live_eyetracking:
                        raw_vout_stream.release()
                    vout_stream.release()
                    pupils_stream.close()
                    do_cal = True
                    print("Starting calibration.")
                    break
                elif key == 'f':
                    freeze_frame = not freeze_frame
        else: # do calibration
            #
            # Output video
            #
            if live_eyetracking:
                print('  Opening output video stream')

                # Output video codec (MP4V - poor quality compression)
                # TODO : Find a better multiplatform codec
                fourcc = cv2.cv.CV_FOURCC('m','p','4','v')

                try:
                    cal_vout_stream = cv2.VideoWriter(cal_vout_path, fourcc, 30, (nx, ny), True)
                except:
                    print('* Problem creating output video stream - skipping pupilometry')
                    return False

                try:
                    raw_cal_vout_stream = cv2.VideoWriter(raw_cal_vout_path, fourcc, 30, (nx, ny), True)
                except:
                    print('* Problem creating output video stream - skipping pupilometry')
                    return False

                if not cal_vout_stream.isOpened():
                    print('* Output video not opened - skipping pupilometry')
                    return False

                if not raw_cal_vout_stream.isOpened():
                    print('* Raw output video not opened - skipping pupilometry')
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
                        print("Updating Configuration")
                        cfg = config.LoadConfig(data_dir)
                        cfg_ts = time.time()

                # Current video time in seconds
                t = time.time()

                # -------------------------------------
                # Pass this frame to pupilometry engine
                # -------------------------------------
                # b4_engine = time.time()
                pupil_ellipse, roi_rect, blink, glint, frame_rgb = engine.PupilometryEngine(frame, cascade, cfg)
                # print "Engine took %s ms" % (time.time() - b4_engine)

                # Derive pupilometry parameters
                px, py, area = engine.PupilometryPars(pupil_ellipse, glint, cfg)

                # Write data line to pupilometry CSV file
                cal_pupils_stream.write(
                    '%0.4f,%0.3f,%0.3f,%0.3f,%d,%0.3f,\n' %
                    (t, area, px, py, blink, art_power)
                )

                # Write output video frame
                cal_vout_stream.write(frame_rgb)

                # Write output video frame
                if live_eyetracking:
                    raw_cal_vout_stream.write(frame_orig)

                # Read next frame (if available)
                # if verbose:
                #     b4_frame = time.time()
                keep_going, frame_orig = media.LoadVideoFrame(vin_stream, cfg)
                if keep_going:
                    frame, art_power = media.Preproc(frame_orig, cfg)
                else:
                    art_power = 0.0

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
                key = utils._waitKey(1)
                if key == 'ESC':
                    keep_going = False
                    # Clean up
                    if live_eyetracking:
                        raw_cal_vout_stream.release()
                    cal_vout_stream.release()
                    cal_pupils_stream.close()
                elif key == 'v' or not keep_going:
                    do_cal = False
                    print("Stopping calibration.")
                    # Clean up
                    if live_eyetracking:
                        raw_cal_vout_stream.release()
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
    keep_going, frame_orig = media.LoadVideoFrame(vin_stream, cfg)
    if keep_going:
        frame, art_power = media.Preproc(frame_orig, cfg)
    else:
        art_power = 0.0

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
        pupil_ellipse, roi_rect, blink, glint, frame_rgb = engine.PupilometryEngine(frame, cascade, cfg)

        # Derive pupilometry parameters
        px, py, area = engine.PupilometryPars(pupil_ellipse, glint, cfg)

        # Write data line to pupilometry CSV file
        pupils_stream.write(
            '%0.3f,%0.3f,%0.3f,%0.3f,%d,%0.3f,\n' %
            (t, area, px, py, blink, art_power)
        )

        # Write output video frame
        vout_stream.write(frame_rgb)

        # Read next frame (if available)
        keep_going, frame_orig = media.LoadVideoFrame(vin_stream, cfg)
        if keep_going:
            frame, art_power = media.Preproc(frame_orig, cfg)
        else:
            art_power = 0.0

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
