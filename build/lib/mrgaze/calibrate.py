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
try:
    from mrgaze import moco, pupilometry
except:
    import pupilometry



def AutoCalibrate(ss_res_dir, cfg):
    '''
    Automatic calibration transform from pupil center timeseries
    '''
    
    # Get fixation heatmap percentile limits and Gaussian blur sigma
    pmin = cfg.getfloat('CALIBRATION', 'heatpercmin')    
    pmax = cfg.getfloat('CALIBRATION', 'heatpercmax')
    plims = (pmin, pmax)
    sigma = cfg.getfloat('CALIBRATION', 'heatsigma')
    
    # Get target coordinates
    targetx = json.loads(cfg.get('CALIBRATION', 'targetx'))
    targety = json.loads(cfg.get('CALIBRATION', 'targety'))
    
    # Gaze space target coordinates (n x 2)
    targets = np.array([targetx, targety]).transpose()
    
    # Calibration pupilometry file
    cal_pupils_csv = os.path.join(ss_res_dir,'cal_pupils.csv')
    
    if not os.path.isfile(cal_pupils_csv):
        print('* Calibration pupilometry not found - returning')
        return False
        
    # Read raw pupilometry data
    p = pupilometry.ReadPupilometry(cal_pupils_csv)
    
    # Extract useful timeseries
    t  = p[:,0] # Video soft timestamp
    px = p[:,2] # Video pupil center, x
    py = p[:,3] # Video pupil center, y
    
    # Remove NaNs (blinks, etc) from t, x and y
    ok = np.isfinite(px)
    t, x, y = t[ok], px[ok], py[ok]
    
    # Find spatial fixations and sort temporally
    # Returns heatmap with axes
    fixations, hmap, xedges, yedges = FindFixations(x, y, plims, sigma)
    
    # Temporally sort fixations - required for matching to targets
    fixations = SortFixations(t, x, y, fixations)
    
    # Plot labeled calibration heatmap to results directory
    PlotCalibration(ss_res_dir, hmap, xedges, yedges, fixations)   

    # Check for autocalibration problems
    n_targets = targets.shape[0]
    n_fixations = fixations.shape[0]

    if n_targets == n_fixations:
        
        # Compute calibration mapping video to gaze space
        C = CalibrationModel(fixations, targets)
    
        # Determine central fixation coordinate in video space
        central_fix = CentralFixation(fixations, targets)
    
        # Write calibration results to CSV files in the results subdir
        WriteCalibration(ss_res_dir, fixations, C, central_fix)

    else:
        
        print('* Number of detected fixations (%d) and targets (%d) differ - exiting' % (n_fixations, n_targets))

        # Return empty/dummy values
        C = np.array([])
        central_fix = 0.0, 0.0
    
    return C, central_fix


def FindFixations(x, y, plims=(5,95), sigma=2.0):
    '''
    Find fixations by blob location in pupil center heat map
    
    Fixations returned are not time ordered
    '''
    
    # Find robust ranges
    xmin, xmax = np.percentile(x, plims)
    ymin, ymax = np.percentile(y, plims)
    
    # Expand bounding box by 30%
    sf = 1.30
    hx, hy = (xmax - xmin) * sf * 0.5, (ymax - ymin) * sf * 0.5
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
    blobs = np.array(hmap > th, np.uint8)
    
    # Morphological opening (circle 2 pixels diameter)
    # kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
    # blobs = cv2.morphologyEx(blobs, cv2.MORPH_OPEN, kernel)
        
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
    central_fix : float tuple
        Pupil center in video space for central fixation
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
    fixations_sorted = fixations[fix_order,:]
    
    
    return fixations_sorted


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

    C = biquadratic transform matrix (2 x 6) (rank 2, full row rank)

    R = fixation matrix (6 x n) in video space (rank 6)
    
    R0 = fixation targets (2 x n) in gaze space (rank 2)
    
    R has rows xx, xy, yy, x, y, 1
    
    Arguments
    ----
    fixations : n x 2 float array
        Fixation coordinates in video space. n >= 6
    targets : n x 2 float array
        Fixation targets in normalized gazed space
    
    Returns
    ----
    C : 2 x 6 float array
        Biquadratic video-gaze post-multiply transform matrix
    '''

    # Init biquadratic coefficient array
    C = np.zeros((2,6))
        
    # Need at least 6 points for biquadratic mapping
    if fixations.shape[0] < 6:
        print('Too few fixations for biquadratic video to gaze mapping')
        return C

    # Create fixation biquadratic matrix, R
    R = MakeR(fixations)
    
    # R0t is the transposed target coordinate array (n x 2)
    R0 = targets.transpose()
    
    # Compute C by pseudoinverse of R (R+)
    # C.R = R0
    # C.R.R+ = R0.R+ = C
    Rplus = np.linalg.pinv(R)
    C = R0.dot(Rplus)
    
    # Check that C maps correctly
    # print(C.dot(R).transpose())
    # print(R0.transpose())

    return C


def MakeR(points):
    
    # Extract coordinates from n x 2 points matrix
    x, y = points[:,0], points[:,1]    
    
    # Additional binomial coordinates
    xx = x * x
    yy = y * y
    xy = x * y;
    
    # Construct R (n x 6)
    R = np.array((xx, xy, yy, x, y, np.ones_like(x)))
    
    return R


def HeatMap(x, y, xlims, ylims, sigma=1.0):
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
    hbins = [np.linspace(ymin, ymax, 64), np.linspace(xmin, xmax, 65)]

    # Construct histogram 
    # *** Note y/row, x/col ordering
    hmap, yedges, xedges = np.histogram2d(y, x, bins=hbins)
    
    # Gaussian blur
    if sigma > 0:
        hmap = cv2.GaussianBlur(hmap, (0,0), sigma, sigma)
    
    return hmap, xedges, yedges


def ApplyCalibration(ss_dir, C, central_fix, cfg):
    '''
    Apply calibration transform to gaze pupil center timeseries
    
    - apply motion correction if requested (highpass or known fixations)
    - Save calibrated gaze to text file in results directory
    
    Arguments
    ----
    
    Returns
    ----
    '''
    
    print('  Calibrating pupilometry timeseries')
    
    # Uncalibrated gaze pupilometry file
    gaze_uncal_csv = os.path.join(ss_dir,'results','gaze_pupils.csv')
    
    # Known central fixations file
    fixations_txt = os.path.join(ss_dir,'videos','fixations.txt')
    
    if not os.path.isfile(gaze_uncal_csv):
        print('* Uncalibrated gaze pupilometry not found - returning')
        return False
        
    # Read raw pupilometry data
    p = pupilometry.ReadPupilometry(gaze_uncal_csv)
    
    # Extract useful timeseries
    t      = p[:,0] # Video soft timestamp
    x      = p[:,2] # Pupil x
    y      = p[:,3] # Pupil y
    
    # Retrospective motion correction - only use when consistent glint is unavailable
    motioncorr = cfg.get('ARTIFACTS','motioncorr')
    mocokernel = cfg.getint('ARTIFACTS','mocokernel')
    
    if motioncorr == 'knownfixations':
        
        print('  Motion correction using known fixations')
        print('  Central fixation at (%0.3f, %0.3f)' % (central_fix[0], central_fix[1]))
        
        x, y, bx, by = moco.KnownFixations(t, x, y, fixations_txt, central_fix)
        
    elif motioncorr == 'highpass':
        
        print('  Motion correction by high pass filtering (%d sample kernel)' % mocokernel)
        print('  Central fixation at (%0.3f, %0.3f)' % (central_fix[0], central_fix[1]))
        
        x, y, bx, by = moco.HighPassFilter(t, x, y, mocokernel, central_fix)

    else:
        
        print('* Unknown motion correction requested (%s) - skipping' % (motioncorr))
        
        # Return dummy x and y baseline estimates
        bx, by = np.zeros_like(x), np.zeros_like(y)
    
    # Additional binomial coordinates
    xx = x * x
    yy = y * y
    xy = x * y;
    
    # Construct R
    R = np.array((xx, xy, yy, x, y, np.ones_like(x))) 
    
    # Apply calibration transform to pupil-glint vector timeseries
    # (2 x n) = (2 x 6) x (6 x n)
    gaze = C.dot(R)
    
    # Write calibrated gaze to CSV file in results directory
    gaze_csv = os.path.join(ss_dir,'results','gaze_calibrated.csv')
    WriteGaze(gaze_csv, t, gaze[0,:], gaze[1,:], bx, by)
    
    return True
    

def CentralFixation(fixations, targets):
    '''
    Find video coordinate corresponding to gaze fixation at (0.5, 0.5)
    '''
        
    idx = -1
    central_fix = np.array([np.NaN, np.NaN])
    
    for ii in range(targets.shape[0]):
        if targets[ii,0] == 0.5 and targets[ii,1] == 0.5:
            idx = ii
            central_fix = fixations[idx,:]
            
    if idx < 0:
        print('* Central fixation target not found')
        central_fix = np.array([np.NaN, np.NaN])

    return central_fix


def WriteGaze(gaze_csv, t, gaze_x, gaze_y, bline_x, bline_y):
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
        gaze_stream.write('%0.3f,%0.3f,%0.3f,%0.3f,%0.3f\n' %
            (tt, gaze_x[tc], gaze_y[tc], bline_x[tc], bline_y[tc]))

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
    fig = plt.figure(figsize = (6,6))

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
    

def WriteCalibration(ss_res_dir, fixations, C, central_fix):
    '''
    Write calibration matrix and fixations to CSV files in results subdirectory
    '''
    
    # Write calibration matrix to text file in results subdir
    calmat_csv = os.path.join(ss_res_dir, 'calibration_matrix.csv')
    
    # Write calibration matrix to CSV file
    try:
        np.savetxt(calmat_csv, C, delimiter=",")
    except:
        print('* Problem saving calibration matrix to CSV file - skipping')
        return False
 
    # Write calibration fixations in video space to results subdir
    calfix_csv = os.path.join(ss_res_dir, 'calibration_fixations.csv')
    
    # Write calibration fixations to CSV file
    try:
        np.savetxt(calfix_csv, fixations, delimiter=",")
    except:
        print('* Problem saving calibration fixations to CSV file - skipping')
        return False
        
    # Write central fixation in video space to results subdir
    ctrfix_csv = os.path.join(ss_res_dir, 'central_fixation.csv')
    
    # Write calibration fixations to CSV file
    try:
        np.savetxt(ctrfix_csv, central_fix, delimiter=",")
    except:
        print('* Problem saving central fixation to CSV file - skipping')
        return False
        
    return True
