#!/usr/bin/env python3
'''
 Pupilometry engine functions.

 AUTHOR : Mike Tyszka
 PLACE  : Caltech
 DATES  : 2016-02-22 JMT Spin off from pupilometry.py to solve cross import issue
          2016-03-10 JMT Adapt for MrGaze UI. Drop classifier for manual ROI

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

 Copyright 2016 California Institute of Technology.
'''

import cv2
import numpy as np
from skimage import measure, morphology

from mrgaze import fitellipse


def pupilometry_engine(frame_gray, pupil_thresh, glint_thresh, qtroi):
    """
    Detection and ellipse fitting of pupil boundary

    Arguments
    ----
    frame_gray : 2D numpy uint8 array
        Video frame
    pupil_thresh : float
        Percent of full intensity range to use as pupil upper threshold
    glint_thresh : float
        Percent of full intensity range to use as glint lower threshold
    qtroi : QRect object
        ROI details

    Returns
    ----
    pupil_ellipse : float tuple
        Ellipse parameters (x0, y0), (a, b), phi
    roi_rect : float tuple
        Pupil ROI rectangle (x0, y0), (x1, y1)
    blink : boolean
        Blink flag (no pupil detected)
    roi_rgb : RGB image
        ROI contents with pupilometry overlays
    """

    DEBUG = True

    if DEBUG:
        print("Entering pupilometry engine")

    # Unset blink flag
    blink = False

    # Init pupil and glint parameters
    pupil_ellipse = ((1, 1), (1, 1), 0)
    glint = (0, 0)

    # Convert thresholds from percent full range to 8-bit level
    pupil_thresh = np.int(pupil_thresh * 2.55)
    glint_thresh = np.int(glint_thresh * 2.55)

    # Get ROI corner coordinates using getCoords method of QRect
    if qtroi.isEmpty():
        x1, y1, x2, y2 = 0, 0, frame_gray.shape[1], frame_gray.shape[0]
    else:
        x1, y1, x2, y2 = qtroi.getCoords()

    if DEBUG:
        print("ROI : (%d, %d, %d, %d)" % (x1, y2, x2, y2))

    # Extract ROI subimage (note row,col indexing of image array)
    roi_gray = frame_gray[y1:y2, x1:x2]

    # Create bright and dark pixel masks by simple thresholding
    # These masks contain candidate pupil and glint blobs

    _, dark_mask = cv2.threshold(roi_gray, pupil_thresh, maxval=255, type=cv2.THRESH_BINARY_INV)
    _, bright_mask = cv2.threshold(roi_gray, glint_thresh, maxval=255, type=cv2.THRESH_BINARY)

    if not blink:

        ###################
        # BEGIN ENGINE CORE

        if DEBUG:
            print("Locating pupil in dark mask")

        # Locate pupil blob within pupil mask
        # pupil_mask = find_pupil(dark_mask)
        pupil_mask = np.array([])

        if np.any(pupil_mask):

            if DEBUG:
                print("Fitting ellipse to pupil boundary")

            # Fit ellipse to pupil blob boundary
            pupil_ellipse = fit_pupil(pupil_mask, roi_gray)

            if DEBUG:
                print("Locating candidate glints")

            # Find best glint candidate in bright mask
            glints = find_glints(bright_mask)

            if DEBUG:
                print("Choosing best glint")

            # Select best glint candidate
            glint = find_best_glint(glints, pupil_ellipse)

            # Add ROI offset to ellipse center and glint
            (px, py), (pa, pb), ptheta = pupil_ellipse
            pupil_ellipse = (px + x1, py + y1), (pa, pb), ptheta
            gx, gy = glint
            glint = (gx + x1, gy + y1)

        else:
            blink = True

        # END ENGINE CORE
        ##################

    # Overlay ROI, pupil ellipse and pseudo-glint on ROI contents
    roi_rgb = overlay_pupil(roi_gray, dark_mask, bright_mask, pupil_ellipse, glint)

    return pupil_ellipse, blink, glint, roi_rgb


def find_pupil(dark_mask):
    """
    Locate pupil blob within dark pixel mask

    Arguments
    ----
    dark_mask : 2D numpy uint8 array
        Dark pixel mask - all pixels < pupil_thresh

    Returns
    ----
    pupil_mask : 2D numpy uint8 array
        Pupil pixel mask

    """

    DEBUG = True

    # Morphological opening removes small objects
    # kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
    # blobs = cv2.morphologyEx(blobs, cv2.MORPH_OPEN, kernel)

    # Label connected regions
    pupil_labels = measure.label(dark_mask, background=0) + 1

    # Region properties
    pupil_props = measure.regionprops(pupil_labels)

    # Init maximum circularity
    cmax = 0.0

    # Init best pupil label
    pupil_label = -1

    for props in pupil_props:

        # Extract region properties
        n, a, p = props.label, props.area, props.perimeter

        # Circularity c = 4 pi A / P**2
        if p > 0.0:
            c = 4.0 * np.pi * a / (p ** 2)
        else:
            c = 0.0

        if c > cmax:
            cmax = c
            pupil_label = n

    if DEBUG:
        print('Pupil label : %d' % pupil_label)

    # Extract most pupil-like blob
    if pupil_label > 0:

        pupil_mask = np.uint8(pupil_labels == pupil_label)

        # Replace pupil blob with filled convex hull
        pupil_mask = np.uint8(morphology.convex_hull_image(pupil_mask))

    else:

        pupil_mask = np.zeros_like(dark_mask)

    return pupil_mask


def find_glints(bright_mask):
    """
    Locate small bright region roughly centered in ROI
    This function should be called before any major preprocessing of the frame.
    The ROI should be unscaled and unblurred, since we assume glints
    are saturated and have maximum intensity (255 in uint8)

    Arguments
    ----
    bright_mask : 2D numpy unit8 array
        Bright pixel mask - all pixels > glint_threshold

    Returns
    ----
    bright_props : list of region properties (Scikit Image)
        Region property list for bright blobs in mask
    """

    # Label connected regions (blobs)
    bright_labels = measure.label(bright_mask)

    # Get region properties for all bright blobs in mask
    bright_props = measure.regionprops(bright_labels)

    return bright_props


def find_best_glint(bright_props, pupil_ellipse):

    # Init glint parameters
    dr_min = np.Inf
    glint = (0, 0)

    # Parse pupil ellipse values
    (px, py), (pa, pb), ptheta = pupil_ellipse

    # Find closest blob to ROI center within glint area range
    for props in bright_props:

        gy, gx = props.centroid  # (row, col)
        dr = np.sqrt((gx - px) ** 2 + (gy - py) ** 2)

        if dr < dr_min:
            dr_min = dr
            glint = (gx, gy)

    return glint


def fit_pupil(pupil_mask, roi_gray):
    """
    Fit ellipse to pupil-iris boundary in segmented ROI

    Arguments
    ----
    img_bw : 2D numpy uint8 array
        Binary thresholded version of pupil ROI (from FindPupil)
    img_gray : 2D scalar array
        Grayscale image of pupil-iris region

    Returns
    ----
    ellipse : tuple of tuples
        Fitted ellipse parameters ((x0, y0), (a,b), theta)
    """

    # Identify edge pixels using Canny filter
    pupil_edges = cv2.Canny(pupil_mask, 0, 1)

    # Find all edge point coordinates
    pnts = np.transpose(np.nonzero(pupil_edges))

    # Swap columns - pnts are (row, col) and need to be (x,y)
    pnts[:,[0,1]] = pnts[:,[1,0]]

    # Robust LSQ fitting parameters
    max_refines = 10
    max_inliers_perc = 95

    # Fit ellipse to boundary point cloud
    ellipse = fitellipse.FitEllipse_RobustLSQ(pnts, roi_gray, max_refines, max_inliers_perc)

    return ellipse


def pupilometry_pars(ellipse, glint):
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
    gx, gy = glint
    px = px - gx
    py = py - gy

    # Pupil area corrected for viewing angle
    # Assumes semi-major axis is actual pupil radius
    area_corr = np.pi * aa ** 2

    return px, py, area_corr


def overlay_pupil(roi_gray, dark_mask, bright_mask, ellipse, glint):
    """
    Overlay fitted pupil, glint and masks on ROI contents

    Parameters
    ----------
    frame_gray
    dark_mask
    bright_mask
    ellipse
    glint
    qtroi

    Returns
    -------
    roi_rgb
    """

    # line thickness
    thickness = 1

    # Mask blending parameters
    alpha = 0.5

    # Overlay colors
    ellipse_color = (0,255,0)
    glint_color = (0,0,255)

    # Parse pupil ellipse values
    (x0, y0), (a, b), phi = ellipse

    # Cross hair size proportional to mean ellipse axis
    r = (a + b) / 20.0

    # Endpoints of major and minor axes
    a0 = int(x0 + r), int(y0)
    a1 = int(x0 - r), int(y0)
    b0 = int(x0), int(y0 + r)
    b1 = int(x0), int(y0 - r)

    # RGB version of roi contents
    roi_rgb = cv2.cvtColor(roi_gray, cv2.COLOR_GRAY2RGB)
    dark_rgb = cv2.cvtColor(dark_mask, cv2.COLOR_GRAY2RGB)
    bright_rgb = cv2.cvtColor(bright_mask, cv2.COLOR_GRAY2RGB)

    # Colorize dark and bright masks
    dark_rgb = cv2.applyColorMap(dark_rgb, cv2.COLORMAP_WINTER)
    bright_rgb = cv2.applyColorMap(bright_rgb, cv2.COLORMAP_COOL)

    # Transparent dark and bright mask overlays
    cv2.addWeighted(roi_rgb, 1 - alpha, dark_rgb, alpha, 0)
    cv2.addWeighted(roi_rgb, 1 - alpha, bright_rgb, alpha, 0)

    # Overlay ellipse
    cv2.ellipse(roi_rgb, ellipse, ellipse_color, thickness)

    # Overlay ellipse axes
    cv2.line(roi_rgb, a0, a1, ellipse_color, thickness)
    cv2.line(roi_rgb, b0, b1, ellipse_color, thickness)

    # Overlay glint centroid
    px, py = int(glint[0]), int(glint[1])
    cv2.circle(roi_rgb, (px, py), 3, glint_color)

    return roi_rgb
