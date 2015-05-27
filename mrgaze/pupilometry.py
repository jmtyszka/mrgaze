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
from skimage.measure import label, regionprops
from mrgaze import media, utils, fitellipse, improc

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
    
    # RGB version of preprocessed frame for later use
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
    
    # Init ROI to whole frame
    # Note (col, row) = (x, y) for shape
    x0, x1, y0, y1 = 0, frame.shape[1], 0, frame.shape[0]

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
            x0, x1, y0, y1 = x, x+w, y, y+h

        else:
            
            blink = True
            
    if not blink:

        # Define ROI rect
        roi_rect = (x0,y0), (x1,y1)
        
        # Extract pupil ROI (note row,col indexing of image array)
        roi = frame[y0:y1,x0:x1]
    
        ###################
        # BEGIN ENGINE CORE
    
        # Find glint(s) in ROI
        glints, glints_mask, roi_noglints = FindGlints(roi, cfg)

        # Segment pupil within ROI
        pupil_bw, roi_rescaled = SegmentPupil(roi, cfg)
     
        # Fit ellipse to pupil boundary - returns ellipse parameter tuple
        ell = FitPupil(pupil_bw, roi, cfg)
        
        # Identify glint closest to pupil center.
        gl = FindBestGlint(glints, ell[0])
            
        # Add ROI offset to ellipse center and glint
        pupil_ellipse = (ell[0][0]+x0, ell[0][1]+y0),ell[1], ell[2]
        glint_center = (gl[0]+x0, gl[1]+y0)
    
        # END ENGINE CORE
        ##################
        
        # Check for unusually high eccentricity
        if fitellipse.Eccentricity(pupil_ellipse) > 0.8:
        
            # Convert to blink
            blink = True
        
    else:
        
        # Fill return values with NaNs if blink detected
        pupil_ellipse = ((np.nan, np.nan), (np.nan, np.nan), np.nan)
        roi_rect = (np.nan, np.nan), (np.nan, np.nan)
        glint_center = (np.nan, np.nan)


    if not blink:

        # Overlay ROI, pupil ellipse and pseudo-glint on background RGB frame
        frame_rgb = OverlayPupil(frame_rgb, pupil_ellipse, roi_rect, glint_center)
        

    if cfg.getboolean('OUTPUT', 'graphics'):
        
            if not blink:
                # Create composite image of various stages of pupil detection
                seg_gray = np.hstack((roi, pupil_bw * 255, glints_mask * 255, roi_rescaled))
                cv2.imshow('Segmentation', seg_gray)

            cv2.imshow('Pupilometry', frame_rgb)
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
    sigma = cfg.getfloat('PUPILSEG','sigma')
    
    # Apply Gaussian smoothing
    if sigma > 0.0:
        roi = cv2.GaussianBlur(roi, (0,0), sigma, sigma)
    
    # Estimate pupil diameter in pixels
    pupil_d = int(cfg.getfloat('PUPILSEG','pupildiameterperc') * roi.shape[0] / 100.0)
    
    # Estimate pupil bounding box area in pixels
    pupil_bb_area_perc = pupil_d * pupil_d / float(roi.size) * 100.0
    
    # Saturate all but pupil bounding box area
    roi_rescaled = improc.RobustRescale(roi, (0, pupil_bb_area_perc))
    
    if method == 'manual':

        # Convert percent threshold to pixel intensity threshold
        thresh = int(cfg.getfloat('PUPILSEG','pupilthresholdperc') / 100.0 * 255.0)
        
        # Manual thresholding - ideal for real time ET with UI thresh control
        _, blobs = cv2.threshold(roi_rescaled, thresh, 255, cv2.THRESH_BINARY_INV) 
    
    elif method == 'otsu':
        
        # Binary threshold of rescaled ROI using Ostu's method
        thresh, blobs = cv2.threshold(roi_rescaled, 128, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
        
    # Morphological opening (circle 5 pixels diameter)
    # kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
    # blobs = cv2.morphologyEx(blobs, cv2.MORPH_OPEN, kernel)
        
    # Label connected components - one should be the pupil
    labels, n_labels = spi.measurements.label(blobs)
    
    # Measure blob areas
    areas = spi.sum(blobs, labels, range(n_labels+1))
        
    # Find maximum area blob
    pupil_label = np.where(areas == areas.max())[0][0]
    
    # Extract blob with largest area
    pupil_bw = np.uint8(labels == pupil_label)
        
    return pupil_bw, roi_rescaled
    

def FindGlints(roi, cfg):
    '''
    Identify small bright objects in ROI
    ROI should be unscaled and unblurred, since we assume glints
    are saturated and have maximum intensity (255 in uint8)
    
    Arguments
    ----
    roi : 2D numpy uint8 array
        Pupil/iris ROI image
    cfg : configuration object
        Configuration parameters including fractional glint diameter estimate
    
    Returns
    ----
    glints : float array
        N x 2 array of glint centroids
    glints_mask : 2D numpy uint8 array
        Bright region mask used by glint removal
    roi_noglints : 2D numpy uint8 array
        Pupil/iris ROI without small bright areas
    '''

    # Estimated glint diameters in pixels
    glint_d = int(cfg.getfloat('PUPILSEG','glintdiameterperc') * roi.shape[0] / 100.0)

    # Glint should at least be 1 pixel
    if glint_d < 1: glint_d = 1

    # Binarize maximum intensity pixels
    glints_mask = np.uint8(roi == 255)

    # Get region properties for all glints in mask
    glint_props = np.array(regionprops(label(glints_mask)))
    
    # Array for storing x,y coord of glints
    tmp_glints = [] # Only contains the glints which are large enough
    glints = np.zeros_like(glint_props) # 

    # Loop over all glints, finding minimum distance to pupil center
    for i, props in enumerate(glint_props):
        gy, gx = props.centroid
        glints[i] = gx, gy
        # Only include glints that are about the right size
        if np.sqrt(props.area / np.pi) > glint_d / 2 and np.sqrt(props.area / np.pi) < glint_d * 2:
            tmp_glints.append((gx,gy))

    # If no glints were large enough, return them anyways
    if len(glints) > 0:
        glints = tmp_glints
    
    # Remove glints from ROI
    roi_noglints = roi * (1 - glints_mask)
    
    return glints, glints_mask, roi_noglints


def FindBestGlint(glints, pupil_center):
    '''
    Find best/closest glint to pupil center
    
    Arguments
    ----
    glint_mask : 2D numpy int array
        Mask for potential glints in frame
    pupil_center : tuple
        Fitted pupil center (px, py)
    '''    
    
    # Unpack pupil center
    px, py = pupil_center
    
    # Init best glint
    min_d = np.Inf
    best_glint = (0.0, 0.0)
    
    # Loop over all glints
    for glint in glints:
        
        # Unpack glint coordinate
        gx,gy = glint
    
        # Glint pupil distance
        d = np.sqrt((px-gx)**2 + (py-gy)**2)
        
        if d < min_d:
            min_d = d
            best_glint = (gx, gy)
    
    return best_glint


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
