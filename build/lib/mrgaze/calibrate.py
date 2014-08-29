#!/opt/local/bin/python
#
# Gaze tracking calibration
# - use calibration video heatmap and priors
#
# AUTHOR : Mike Tyszka
# PLACE  : Caltech
# DATES  : 2014-05-15 JMT From scratch
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
import cv2
import json
import numpy as np
import pylab as plt
from skimage import filter, exposure
from scipy import ndimage
from mrgaze import pupilometry


def AutoCalibrate(ss_res_dir, cfg):
    '''
    Automatic calibration transform from pupil center timeseries
    '''
    
    # Get target coordinates
    targetx = json.loads(cfg.get('CALIBRATION', 'targetx'))
    targety = json.loads(cfg.get('CALIBRATION', 'targety'))
    targets = np.array([targetx, targety])
    
    # Calibration pupilometry file
    cal_pupils_csv = os.path.join(ss_res_dir,'cal_pupils_raw.csv')
    
    if not os.path.isfile(cal_pupils_csv):
        print('* Calibration pupilometry not found - returning')
        return False
        
    # Read raw pupilometry data
    p = pupilometry.ReadPupilometry(cal_pupils_csv)
    
    # Extract useful timeseries
    t      = p[:,0]
    pgx    = p[:,6]
    pgy    = p[:,7]
    
    # Remove NaNs (blinks, etc) from t, x and y
    ok = np.isfinite(pgx)
    t, x, y = t[ok], pgx[ok], pgy[ok]
    
    # Find spatial fixations and sort temporally
    # Returns heatmap with axes
    fixations, hmap, xedges, yedges = FindFixations(x, y)
    
    # Temporally sort fixations - required for matching to targets
    fixations, fix_order, t_fix_sorted = SortFixations(t, x, y, fixations)
    
    # Plot labeled calibration heatmap to results directory
    PlotCalibration(ss_res_dir, hmap, xedges, yedges, fixations)   

    # Check for autocalibration problems
    n_targets = targets.shape[1]
    n_fixations = fixations.shape[0]

    if n_targets != n_fixations:
        print('* Number of detected fixations (%d) and targets (%d) differ - exiting' % (n_fixations, n_targets))
        return np.array([])
    
    # Search for optimal fixation order and associated calibration model coefficients
    C = CalibrationModel(fixations, targets)
   
    return C


def FindFixations(x, y):
    '''
    Find fixations by blob location in pupil center heat map
    
    Fixations returned are not time ordered
    '''
    
    # Gaussian blur sigma for heatmap
    sigma = 2.0
    
    # Find robust ranges
    xmin, xmax = np.percentile(x, (1, 99))
    ymin, ymax = np.percentile(y, (1, 99))
    
    # Expand bounding box by 20%
    hx, hy = (xmax - xmin) * 0.6, (ymax - ymin) * 0.6
    cx, cy = (xmin + xmax) * 0.5, (ymin + ymax) * 0.5
    xmin, xmax = cx - hx, cx + hx
    ymin, ymax = cy - hy, cy + hy
    
    # Compute calibration video heatmap
    hmap, xedges, yedges = HeatMap(x, y, (xmin, xmax), (ymin, ymax), sigma)
    
    # Heatmap dimensions
    # *** Note y/row, x/col ordering

    ny, nx = hmap.shape
        
    # Determine blob threshold for heatmap
    # Need to accommodate hotspots from longer fixations
    # particularly at center.
    # A single fixation blob shouldn't exceed 1% of total frame
    # area so clamp heatmap to 99th percentile
    pA, pB = np.percentile(hmap, (0, 99))
    hmap = exposure.rescale_intensity(hmap, in_range  = (pA, pB))
    
    # Otsu threshold clamped heatmap
    th = filter.threshold_otsu(hmap)
    blobs = np.array(hmap > th, dtype = int)
        
    # Label connected components
    labels, n_labels = ndimage.label(blobs)
    
    # Find blob centroids
    # Transpose before assigning to x and y arrays
    pnts = np.array(ndimage.measurements.center_of_mass(hmap, labels, range(1, n_labels+1)))
    
    # Parse x and y coordinates
    # *** Note y/row, x/col ordering
    fix_x, fix_y  = pnts[:,1], pnts[:,0]
    
    # Map blob centroids to video pixel space using xedges and yedges
    # of histogram2d bins (heatmap pixels). Note that pixels are centered
    # on their coordinates when rendered by imshow. So a pixel at (1,2) is
    # rendered as a rectangle with corners at (0.5,1.5) and (1.5, 2.5)
    fix_xi = np.interp(fix_x, np.linspace(-0.5, nx-0.5, nx+1), xedges)
    fix_yi = np.interp(fix_y, np.linspace(-0.5, ny-0.5, ny+1), yedges)
    
    # Fixation centroids (n x 2)
    fixations = np.array((fix_xi, fix_yi)).T
    
    return fixations, hmap, xedges, yedges


def SortFixations(t, x, y, fixations):
    '''
    Temporally sort detected spatial fixations
    
    Arguments
    ----
    t : float vector
        Sample time points in seconds
    x : float vector
        Pupil center x coordinate timeseries
    y : float vector
        Pupil center y coordinate timeseries
    fixations : n x 2 float array
        Detected spatial fixation coordinates
        
    Returns
    ----
    fixations_sorted : 2 x n float array
        Spatial fixations sorted temporally
    fix_order : integer vector
        Order mapping original to sorted fixations
    t_fix_sorted : float vector
        Sorted median times in seconds of each fixation
    '''

    # Count number of fixations and timepoints
    nt = x.shape[0]
    nf = fixations.shape[0]

    # Put coordinate timeseries in columns
    X = np.zeros([nt,2])
    X[:,0] = x
    X[:,1] = y

    # Map each pupil center to nearest fixation
    idx = NearestFixation(X, fixations)

    # Median time of each fixation
    t_fix = np.zeros(nf)
    for fc in np.arange(0,nf):
        t_fix[fc] = np.median(t[idx==fc])
    
    # Temporally sort fixations
    fix_order = np.argsort(t_fix)

    t_fix_sorted = t_fix[fix_order]

    fixations_sorted = fixations[fix_order,:]
    
    return fixations_sorted, fix_order, t_fix_sorted


def NearestFixation(X, fixations):
    '''
    Map pupil centers to index of nearest fixation
    '''

    # Number of time points and fixations
    nt = X.shape[0]
    nf = fixations.shape[0]

    # Distance array
    dist2fix = np.zeros((nt, nf))

    # Fill distance array (nt x nfix)
    for (fix_i, fix) in enumerate(fixations):
        dx, dy = X[:,0] - fix[0], X[:,1] - fix[1]
        dist2fix[:, fix_i] = np.sqrt(dx**2 + dy**2)
        
    # Find index of minimum distance fixation for each timepoint
    return np.argmin(dist2fix, axis=1)
    

def CalibrationModel(fixations, targets):
    '''
    Construct biquadratic transform from video space to gaze space
    
    BIQUADRATIC CALIBRATION MODEL
    ----
    
    We need to solve the matrix equation C * R = R0 where

    C = biquadratic coefficients (2 x 6),

    R = binomial coordinate matrix (6 x 9) - one column per centroid and
    
    R0 = corrected screen coordinates (2 x 9)
    
    R has rows x2_i, xy_i, y2_i, x_i, y_i, 1 (i = 1..9)

    x0_j = R_1j = C11 * x_j^2 + C12 * x_j * y_j + ... + C16

    y0_j = R_2j = C21 * x_j^2 + C22 * x_j * y_j + ... + C26    
    
    
    Arguments
    ----
    fixations : n x 2 float array
        Fixation coordinates in video space. n >= 6
    targets : 2 x 9 float array
        Fixation targets in normalized gazed space
    
    Returns
    ----
    C : 2 x 6 float array
        Biquadratic video-gaze transform matrix
    '''

    # Init biquadratic coefficient array
    C = np.zeros((2,6))
        
    # Need at least 6 points for biquadratic mapping
    if fixations.shape[0] < 6:
        print('Too few fixations for biquadratic video to gaze mapping')
        return C

    # Extract fixation coordinates
    fx, fy = fixations[:,0], fixations[:,1]    
    
    # Additional binomial coordinates
    fx2 = fx * fx
    fy2 = fy * fy
    fxy = fx * fy;
    
    # Construct R (6 x 9)
    R = np.array((fx2, fxy, fy2, fx, fy, np.ones_like(fx)))
    
    # Moore-Penrose pseudoinverse of R (9 x 6)
    Rinv = np.linalg.pinv(R)
    
    # R0 is the target coordinate array (2 x 9)
    R0 = targets
    
    # Solve C.R = R0 by postmultiplying R0 by Rinv
    # C.R.Rinv = C = R0.Rinv
    C = R0.dot(Rinv)

    return C


def HeatMap(x, y, xlims, ylims, sigma=2):
    '''
    Convert pupil center timeseries to 2D heatmap
    '''
    
    # Eliminate NaNs in x, y (from blinks)
    x = x[np.isfinite(x)]
    y = y[np.isfinite(y)]

    # Parse out limits
    xmin, xmax = xlims
    ymin, ymax = ylims
    
    #---
    # NOTE: heatmap dimensions are y (1st) then x (2nd)
    # corresponding to rows then columns.
    # All coordinate orderings are adjusted accordingly
    #--- 
    
    # Composite histogram axis ranges
    # Make bin count different for x and y for debugging
    # *** Note y/row, x/col ordering
    hbins = [np.linspace(ymin, ymax, 50), np.linspace(xmin, xmax, 51)]

    # Construct histogram 
    # *** Note y/row, x/col ordering
    hmap, yedges, xedges = np.histogram2d(y, x, bins=hbins)
    
    # Gaussian blur
    if sigma > 0:
        hmap = cv2.GaussianBlur(hmap, (5,5), sigma)
    
    return hmap, xedges, yedges


def ApplyCalibration(ss_res_dir, C):
    '''
    Apply calibration transform to gaze pupil center timeseries
    
    Save to text file in results directory
    '''
    
    print('  Calibrating pupilometry timeseries')

    # Uncalibrated gaze pupilometry file
    gaze_uncal_csv = os.path.join(ss_res_dir,'gaze_pupils_filt.csv')
    
    if not os.path.isfile(gaze_uncal_csv):
        print('* Uncalibrated gaze pupilometry not found - returning')
        return False
        
    # Read raw pupilometry data
    p = pupilometry.ReadPupilometry(gaze_uncal_csv)
    
    # Extract useful timeseries
    t      = p[:,0]
    x      = p[:,6] # Pupil-glint x
    y      = p[:,7] # Pupil-glint y
    
    # Additional binomial coordinates
    x2 = x * x
    y2 = y * y
    xy = x * y;
    
    # Construct R
    R = np.array((x2, xy, y2, x, y, np.ones_like(x))) 
    
    # Apply calibration transform to pupil-glint vector timeseries
    # (2 x n) = (2 x 6) x (6 x n)
    gaze = C.dot(R)
    
    # Write calibrated gaze to CSV file in results directory
    gaze_csv = os.path.join(ss_res_dir,'gaze_calibrated.csv')
    WriteGaze(gaze_csv, t, gaze[0,:], gaze[1,:])
    
    return True


def WriteGaze(gaze_csv, t, gaze_x, gaze_y):
    '''
    Write calibrated gaze to CSV file
    '''    
        
    # Open calibrated gaze CSV file to write
    try:
        gaze_stream = open(gaze_csv, 'w')
    except:
        print('* Problem opening gaze CSV file to write - skipping')
        return False
    
    '''
    Write gaze line to file
        Timeseries in columns. Column order is:
        0 : Time (s)
        1 : Calibrated gaze x
        2 : Calibrated gaze y
    '''
    
    for (tc,tt) in enumerate(t):
        gaze_stream.write('%0.3f,%0.3f,%0.3f,\n' % (tt, gaze_x[tc], gaze_y[tc]))

    # Close gaze CSV file
    gaze_stream.close()
    
    return True


def ReadGaze(gaze_csv):
    '''
    Read calibrated gaze timerseries from CSV file
    '''
    
    # Read time series in rows
    gt = np.genfromtxt(gaze_csv, delimiter=',')

    # Parse out array
    t, gaze_x, gaze_y = gt[:,0], gt[:,1], gt[:,2]
    
    return t, gaze_x, gaze_y
    

def PlotCalibration(res_dir, hmap, xedges, yedges, fixations):
    '''
    Plot the calibration heatmap and temporally sorted fixation labels
    '''

    # Create a new figure
    fig = plt.figure()

    # Plot spatial heatmap with fixation centroids
    plt.imshow(hmap, interpolation='nearest', aspect='equal',
               extent=[xedges[0], xedges[-1], yedges[-1], yedges[0]])

    # Fixation coordinate vectors
    fx, fy = fixations[:,0], fixations[:,1]
    
    # Overlay fixation centroids with temporal order labels
    plt.scatter(fx, fy, c='w', s=40)
    alignment = {'horizontalalignment':'center', 'verticalalignment':'center'}
    for fc in np.arange(0,fx.shape[0]):
        plt.text(fx[fc], fy[fc], '%d' % fc, backgroundcolor='w', color='k', **alignment)
    
    # Save figure without displaying
    plt.savefig(os.path.join(res_dir, 'cal_fix_space.png'), dpi=150, bbox_inches='tight')

    # Close figure without showing it
    plt.close(fig)