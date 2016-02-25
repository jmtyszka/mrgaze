#!/usr/bin/env python
'''
 Pupilometry engine functions.

 AUTHOR : Mike Tyszka
 PLACE  : Caltech
 DATES  : 2016-02-22 JMT Spin off from pupilometry.py to solve cross import issue

 This file is part of mrgaze.

    mrgaze is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    mrgaze is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
   along with mrgaze.  If not, see <http://www.gnu.org/licenses/>.

 Copyright 2014-2016 California Institute of Technology.
'''

import os
import cv2
import json
import numpy as np
from skimage import measure, morphology
from mrgaze import utils, fitellipse, improc

def PupilometryEngine(frame, cascade, cfg):
    """
    Detection and ellipse fitting of pupil boundary

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
            sizes = np.sqrt(pupils[:,2] * pupils[:,3])
            best_pupil = sizes.argmax()

            # Get ROI info for largest pupil
            x, y, w, h = pupils[best_pupil,:]

        else:

            # Dummy ROI during blink
            x, y, w, h = 0, 0, 1, 1
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
        roi_pix = int(wn * np.min([frw, frh]))
        roi_half_pix = int(roi_pix / 2.0)
        xc_pix, yc_pix = int(xn * frw), int(yn * frh)
        x, y, w, h = xc_pix - roi_half_pix, yc_pix - roi_half_pix, roi_pix, roi_pix


    # Init pupil and glint parameters
    pupil_ellipse = ((np.nan, np.nan), (np.nan, np.nan), np.nan)
    glint_center = (np.nan, np.nan)

    # Catch zero-sized ROI
    if w < 1 | h < 1:
        x, y, w, h = 0, 0, 1, 1

    # Extract pupil ROI (note row,col indexing of image array)
    roi = frame[y:y+h, x:x+w]

    # Create black standard image, to display in case of a blink
    pupil_labels = np.zeros_like(roi)
    glint_mask = np.zeros_like(roi)
    roi_rescaled = np.zeros_like(roi)

    # Define ROI rect
    roi_rect = (x,y), (x+w,y+h)

    if not blink:

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
        if fitellipse.Eccentricity(pupil_ellipse) > 0.95:
            blink = True

    if not blink:

        # Overlay ROI, pupil ellipse and pseudo-glint on background RGB frame
        frame_rgb = OverlayPupil(frame_rgb, pupil_ellipse, roi_rect, glint_center)


    if cfg.getboolean('OUTPUT', 'graphics'):

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
    pupil_labels : 2D numpy uint8 array
        Labeled pupil candidates
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
    pupil_bw : 2D numpy unit8 array
        Black and white pupil segmentation

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

    print ("%s, %s" % (roi_cx, roi_cy))

    # Estimated glint diameter in pixels
    glint_d = int(cfg.getfloat('PUPILSEG','glintdiameterperc') * nx / 100.0)

    # Glint diameter should be >= 1 pixel
    if glint_d < 1:
        glint_d = 1

    # Reasonable upper and lower bounds on glint area (x3, /3)
    glint_A = np.pi * (glint_d / 2.0)**2
    A_min, A_max = glint_A / 3.0, glint_A * 9.0
    
    # print
    # print A_min
    # print A_max

    # Find bright pixels in full scale uint8 image (ie value > 250)
    bright = np.uint8(roi > 254)

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
            #            print A
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
