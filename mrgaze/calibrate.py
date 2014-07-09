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
import numpy as np
from skimage import filter, exposure
from scipy import ndimage
from scipy.ndimage import filters
from scipy.signal import medfilt, gaussian
from matplotlib import pyplot as plt

import io

def AutoCalibrate(subjsess_dir, targets):

    # Calibration pupilometry file
    cal_pupils_csv = os.path.join(subjsess_dir,'cal_pupils.csv')
    
    if not os.path.isfile(cal_pupils_csv):
        print('Calibration pupilometry not found - returning')
        return False
        
    # Load table from CSV file
    t,area,x,y,blink,dummy = io.ReadPupilometry(cal_pupils_csv)
    
    # Remove NaNs (blinks, etc) from t, x and y
    ok = np.isfinite(x)
    t, x, y = t[ok], x[ok], y[ok]
    
    # Use kmeans clustering to find calibration fixations
    fixations = FindFixations(t, x, y)
    
    # Search for optimal fixation order and associated calibration model coefficients
    C = CalibrationModel(fixations, targets)    
   
    return C

#
# Heatmap fixation finder with post hoc temporal sorting
#
def FindFixations(t, x, y):
    
    # Display flag
    graphic = False
    
    # Locate hotspots in pupil center heatmap
    fixations_space = FindFixations_Space(x, y, graphic)
    
    # Find stationary periods in pupil center timeseries
    fixations_time, fix_t = FindFixations_Time(t, x, y, graphic)    
    
    # Estimate median times for each fixation
    # Assumes only one fixation per target in calibration video
    fixations = SortFixations(fixations_space, fixations_time)
    
    return fixations

#
# Find fixations by blob location in pupil center heat map
# Fixations returned are not time ordered
#
def FindFixations_Space(x, y, graphic = False):
    
    # Gaussian blur sigma for heatmap
    sigma = 2.0
    
    # Compute calibration video heatmap
    hmap, xedges, yedges = HeatMap(x, y, sigma)
    
    # hmap dimensions
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
    fix_x, fix_y  = pnts[:,1], pnts[:,0]
    
    # Map blob centroids to video pixel space using xedges and yedges
    # of histogram2d bins (heatmap pixels). Note that pixels are centered
    # on their coordinates when rendered by imshow. So a pixel at (1,2) is
    # rendered as a rectangle with corners at (0.5,1.5) and (1.5, 2.5)
    fix_xi = np.interp(fix_x, np.linspace(-0.5, nx-0.5, nx+1), xedges)
    fix_yi = np.interp(fix_y, np.linspace(-0.5, ny-0.5, ny+1), yedges)
    
    # Fixation centroids (n x 2)
    fixations = np.array((fix_xi, fix_yi)).T
    
    if graphic:

        # Plot heatmap with fixation centroids
        plt.imshow(hmap, interpolation='nearest',
                   extent=[xedges[0], xedges[-1], yedges[-1], yedges[0]])
        plt.scatter(x = fixations[:,0], y = fixations[:,1], c = 'w', s = 40)

    return fixations

#
# Locate fixations in pupil center timecourse
# Fixations returned are intrinsically time ordered
#   
def FindFixations_Time(t, x, y, graphic = False):
    
    # Frame duration in seconds
    dt = t[1] - t[0]

    # Set 500 ms kernel width
    k = int(0.5 / dt)

    # Moving median filter (500 ms kernel)
    x = medfilt(x, k)
    y = medfilt(y, k)
    
    # Gaussian low pass filter
    H = gaussian(k, k/4)
    H = H / H.sum()
    x = filters.convolve1d(x, H)
    y = filters.convolve1d(y, H)
    
    # Calculate pixel velocity of pupil
    vx = np.gradient(x)
    vy = np.gradient(y)
    
    # Pupil voxel speed
    ps = np.sqrt(vx**2 + vy**2)

    # Find possible fixations (ps < 0.1 pixels/frame)
    eye_fix = (ps < 0.1).astype(int)

    # Force final fixation to end with the video
    eye_fix[-1] = 0

    # Assume moving before video - insert leading zero
    eye_fix = np.insert(eye_fix, 0, 0)

    # Fixation state changes - forward difference results loss of last element
    eye_change = eye_fix[1:] - eye_fix[:-1]

    # Find indices of start and end of fixations
    fix_start = np.where(eye_change > 0)[0]
    fix_end   = np.where(eye_change < 0)[0]

    # Fixations durations
    fix_dur = fix_end - fix_start

    # Eliminate fixations shorter than 500 ms
    min_fix_dur = int(0.5 / dt)
    good_fix    = fix_dur > min_fix_dur
    n_good_fix  = np.sum(good_fix)

    # Extract start, end and duration (in frames) of good fixations
    fix_start_good = fix_start[good_fix]
    fix_end_good   = fix_end[good_fix]

    # Find fixation centroids

    fixations = np.zeros((n_good_fix, 2))
    fix_midtime = np.zeros((n_good_fix))

    for fc in range(n_good_fix):
  
        # Frame range of fixation
        f0 = fix_start_good[fc]
        f1 = fix_end_good[fc] - 1
        
        # Mean time of fixation
        fix_midtime[fc] = (f0+f1)/2 * dt
  
        # Mean pupil position during fixation
        fixations[fc,0] = np.mean(x[f0:f1])
        fixations[fc,1] = np.mean(y[f0:f1])
    
    return fixations, fix_midtime


#
# Use temporal fixations to reorder spatial
# fixations
#
def SortFixations(fixations_space, fixations_time):

    # Count fixations
    n_fix_space = fixations_space.shape[0]
    n_fix_time  = fixations_time.shape[0]
    
    if n_fix_time < n_fix_space:
        print('Fewer temporal fixations than spatial fixations : %d < %d'
        % (n_fix_time, n_fix_space))
    
    # Init fixation ordering and sorted fixation arrays
    fixations = np.zeros((n_fix_space, 2))
    
    # Loop over spatial fixations
    for i, fix in enumerate(fixations_space):
        
        dx = fix[0] - fixations_time[:,0]
        dy = fix[1] - fixations_time[:,1]
        dr = np.sqrt(dx**2 + dy**2)
        
        # Find index of temporal fixation closest to current spatial fixation
        temp_idx = np.argmin(dr)        
        
        # Place spatial fixation in correct position
        fixations[temp_idx,:] = fix
    
    return fixations

#
# Permute fixation point order vs targets to determine
# best biquadratic mapping from video (x,y) to gaze (x',y') space.
#
# fixations : 9 x 2 array (n >= 6)
# R0        : fixation target coordinates in gaze space (9 x 2)
#
def CalibrationModel(fixations, targets):
    
    # Init biquadratic coefficient array
    C = np.zeros((2,6))
        
    # Need at least 6 points for biquadratic mapping
    if fixations.shape[0] < 6:
        print('Too few fixations for biquadratic video to gaze mapping')
        return C

    # BIQUADRATIC CALIBRATION MODEL
    #
    # We need to solve the matrix equation C * R = R0
    # C = biquadratic coefficients (2 x 6)
    # R = binomial coordinate matrix (6 x 9) - one column per centroid
    # R0 = corrected screen coordinates (2 x 9)
    #
    # Twelve biquadratic coefficients: x2, xy, y2, xy, x, y, 1
    #
    # R has rows x2_i, xy_i, y2_i, x_i, y_i, 1 (i = 1..9)
    # x0_j = R_1j = C11 * x_j^2 + C12 * x_j * y_j + ... + C16
    # y0_j = R_2j = C21 * x_j^2 + C22 * x_j * y_j + ... + C26
    
    # Extract fixation coordinates
    fx, fy = fixations[:,0], fixations[:,1]    
    
    # Additional binomial coordinates
    fx2 = fx * fx
    fy2 = fy * fy
    fxy = fx * fy;
    
    # Construct R
    R = np.array((fx2, fxy, fy2, fx, fy, np.ones_like(fx)))
    
    # Moore-Penrose pseudoinverse of R
    Rinv = np.linalg.pinv(R)
    
    # R0 is the target coordinate array
    R0 = targets
    
    # Solve C.R = R0 by postmultiplying R0 by Rinv
    # C.R.Rinv = C = R0.Rinv
    C = R0.dot(Rinv)
    
    print C.dot(R)
    print R0
    print R0 - C.dot(R)

    return C
     
#
# Generate the 2D histogram (heatmap) from pupil center timeseries
# with optional 2D Gaussian blur (sigma)
#
def HeatMap(x, y, sigma = 2):
    
    # Eliminate NaNs in x, y (from blinks)
    x = x[np.isfinite(x)]
    y = y[np.isfinite(y)]
    
    # Find robust ranges
    xmin, xmax = np.percentile(x, (1, 99))
    ymin, ymax = np.percentile(y, (1, 99))
    
    # Expand boundaries
    xmin, xmax = xmin*0.9, xmax*1.1
    ymin, ymax = ymin*0.9, ymax*1.1
    
    # Composite histogram axis ranges
    hrng = [[xmin, xmax], [ymin,ymax]]

    # Compute histogram
    # Bin edges arrays will be one larger than hmap matrix sizes
    hmap, xedges, yedges = np.histogram2d(x, y, bins = 50, range = hrng)
    
    # Gaussian blur
    if sigma > 0:
        hmap = cv2.GaussianBlur(hmap, (5,5), sigma)
    
    return hmap, xedges, yedges

