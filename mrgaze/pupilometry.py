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
import numpy as np
import scipy.ndimage as spi
from scipy.cluster.vq import kmeans, whiten, vq
from mrgaze import media, utils, fitellipse
from mrgaze import improc as ip

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
    mrclean_root = utils._package_root()
    LBP_path = os.path.join(mrclean_root, 'Cascade/cascade.xml')
    
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
    nf = vin_stream.get(cv2.cv.CV_CAP_PROP_FRAME_COUNT)
    
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
        px, py, area = PupilometryPars(pupil_ellipse, glint)
        
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
    
    # Init ROI to whole frame
    # Note (col, row) = (x, y) for shape
    x0, x1, y0, y1 = 0, frame.shape[1], 0, frame.shape[0]

    # Shall we use the classifier at all, or whole frame?
    if cfg.getboolean('LBP', 'enabled'):
        
        min_neighbors = cfg.getint('LBP','minneighbors')
        scale_factor  = cfg.getfloat('LBP','scalefactor')
        
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
            x0, x1, y0, y1 = x, x+w, y, y+h

        else:
            
            blink = True

    # Define ROI rect
    roi_rect = (x0,y0),(x1,y1)
        
    # Extract pupil ROI (note row,col indexing of image array)
    pupil_roi = frame[y0:y1,x0:x1]
    
    # Segment pupil intelligently - also return glint mask
    pupil_bw, glint_mask, roi_rescaled = SegmentPupil(pupil_roi, cfg)
            
    # Fit ellipse to pupil boundary - returns ellipse ROI
    eroi = FitPupil(pupil_bw, pupil_roi, cfg)
            
    # Add ROI offset to ellipse center
    pupil_ellipse = (eroi[0][0]+x0, eroi[0][1]+y0),eroi[1], eroi[2]
        
    # Check for unusually high eccentricity
    if fitellipse.Eccentricity(pupil_ellipse) > 0.8:
        blink = True  # Convert to blink
        
    # Fill return values with NaNs if blink detected
    if blink:
        pupil_ellipse = ((np.nan, np.nan), (np.nan, np.nan), np.nan)
        roi_rect = (np.nan, np.nan), (np.nan, np.nan)
        glint = (np.nan, np.nan)
        
    # TODO: find best glint candidate in glint mask
    glint = FindBestGlint(glint_mask, pupil_ellipse)
    
    # RGB version of preprocessed frame for output video
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
            
    # Overlay ROI, pupil ellipse and pseudo-glint on RGB frame
    if not blink:

        # Create RGB overlay of pupilometry on ROI
        frame_rgb = OverlayPupil(frame_rgb, pupil_ellipse, roi_rect, glint)

        if cfg.getboolean('OUTPUT', 'graphics'):
        
            # Create composite image of various stages of pupil detection
            seg_gray = np.hstack((pupil_roi, pupil_bw * 255, glint_mask * 255, roi_rescaled))
            cv2.imshow('Segmentation', seg_gray)
            cv2.imshow('Pupilometry', frame_rgb)
            cv2.waitKey(5)

    return pupil_ellipse, roi_rect, blink, glint, frame_rgb      


def SegmentPupil(roi, cfg):
    """
    Segment pupil within pupil-iris ROI
    
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
    glint_mask : 2D numpy uint8 array
        Bright region mask used by glint removal
    """

    # Get config parameters
    method = cfg.get('PUPILSEG','method')

    # Remove glints if present - helps RANSAC stability
    roi, glint_mask = RemoveGlint(roi, cfg)
    
    # Set the rescale percentile threshold about 25% larger than maximum percent
    # of ROI estimated to be occupied by pupil
    rescale_thresh = min(100.0, cfg.getfloat('PUPILSEG','pupil_percmax') * 1.25)

    # perform histrogram equalization if desired
    if cfg.getboolean('PUPILSEG','histogram_equalization'):
        roi = cv2.equalizeHist(roi)

    # Clamp bright regions to emphasize pupil
    roi = ip.RobustRescale(roi, (0, rescale_thresh))

    # perform another gaussian blur if rescaling and histogram equalization caused too much noise
    gauss_sd = cfg.getint('PUPILSEG','gauss_sd')
    if gauss_sd > 0:
        roi = cv2.GaussianBlur(roi, (3,3), gauss_sd)

    if method == 'manual':
        
        # Manual thresholding - ideal for real time ET with UI thresh control
        
        thresh = cfg.getint('PUPILSEG','pupil_threshold')
        _, blobs = cv2.threshold(roi, thresh, 255, cv2.THRESH_BINARY_INV) 
    
    elif method == 'otsu':

        # Binary threshold of ROI using Ostu's method
        thresh, blobs = cv2.threshold(roi, 128, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    
    elif method == 'kmeans':
        
        # Number of clusters to shoot for
        K = 4
        
        # Flatten and whiten ROI pixels
        data = whiten(roi.flatten())
        
        # 1D k-means clustering of intensity values
        centroids, _ = kmeans(data, K)
        
        # Assume minimum valued centroid represents pupil pixels
        pupil_idx = np.argmin(centroids)
        
        # Assign all pixels to a cluster
        idx, _ = vq(data, centroids)
        
        idx = np.reshape(idx, roi.shape)
        
        # Create mask for minimum centroid pixels
        blobs = np.uint8(idx == pupil_idx) * 255
        
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
        
    return pupil_bw, glint_mask, roi
    

def RemoveGlint(roi, cfg):
    '''
    Remove small bright areas from pupil/iris ROI
    
    Arguments
    ----
    roi : 2D numpy uint8 array
        Pupil/iris ROI image
    percmax : float
        Maximum ROI area occupied by glints in percent
    k : odd integer
        Kernel size for mask dilation and inpainting
    
    Returns
    ----
    roi_noglint : 2D numpy uint8 array
        Pupil/iris ROI without small bright areas
    glint_mask : 2D numpy uint8 array
        Bright region mask used by glint removal
    '''

    # Flag for glint removal by inpainting
    inpaint = True
    
    percmax = cfg.getfloat('PUPILSEG','glint_percmax')

    # Kernel sizes
    k_dil = cfg.getint('PUPILSEG','k_dil')
    k_inpaint = cfg.getint('PUPILSEG','k_inpaint')

    # Determine lower threshold for brightest pixels
    glint_thresh = np.percentile(roi, percmax)
    
    # Inpainting mask
    mask = np.uint8(roi >= glint_thresh)
    
    # Dilate
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(k_dil,k_dil))
    glint_mask = cv2.morphologyEx(mask, cv2.MORPH_DILATE, kernel)
    
    # Remove bright regions
    if inpaint:
        
        # Inpaint bright regions - higher quality glint removal
        roi_noglint = cv2.inpaint(roi, glint_mask, k_inpaint, cv2.INPAINT_NS)

    else:

        # Zero out bright regions
        # Note: this will introduce edge artifacts around glints
        roi_noglint = roi * (1 - glint_mask)
        
    return roi_noglint, glint_mask


def FindBestGlint(glint_mask, pupil_ellipse):
    '''
    TODO: Find closest glint to pupil center
    '''    
    
    glint = (0.0, 0.0)
    
    return glint


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
    
    # RANSAC ellipse fitting to edge points
    ellipse = fitellipse.FitEllipse_RANSAC(pnts, roi, cfg)
    
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
    
    # Pseudo-glint marker color
    glint_color = (0,255,255)

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
    
    # Overlay pseudoglint
    px, py = int(glint[0]), int(glint[1])
    cv2.circle(frame_rgb, (px, py), 2, glint_color)

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


def PupilometryPars(ellipse, glint):
    """
    Extract pupil center and corrected area
    """
    
    # Unpack ellipse tuple
    (px, py), (bb, aa), phi_b_deg = ellipse
    
    # Adjust pupil center for glint location
    # If glint correction is off, gx = gy = 0.0
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
