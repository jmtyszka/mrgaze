#!/usr/bin/env python
#
# Motion and drift corrections for gaze tracking
# - robust high-pass filtering
# - known fixation correction
# - pseudo-glint estimation
#
# AUTHOR : Mike Tyszka
# PLACE  : Caltech
# DATES  : 2014-05-23 JMT From scratch
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

import cv2
import numpy as np
from scipy.ndimage.measurements import center_of_mass
from scipy.signal import medfilt
from mrgaze import utils

def HighPassFilter(t, px, py, moco_kernel, central_fix):
    '''
    Slow drift correction by robust high-pass filtering

    Effectively forces long-term average pupil fixation to be centrally
    fixated in video space.

    Arguments
    ----
    t : 1D float array
        Video soft timestamps in seconds
    px : 1D float array
        Video space pupil center x
    py : 1D float array
        Video space pupil center y
    moco_kernel : integer
        Temporal kernel width in samples [31]
    central_fix : float tuple
        (x,y) coordinate in video space of central fixation

    Returns
    ----
    px_filt : 1D float array
        Drift corrected video space pupil center x
    py_filt : 1D float array
        Drift corrected video space pupil center y
    '''

    # Force odd-valued kernel width
    moco_kernel = utils._forceodd(moco_kernel)

    print('  Highpass filtering with %d sample kernel' % moco_kernel)

    # Infill NaN regions
    # Replace NaNs with unreasonable but finite value for median filtering
    nan_idx = np.isnan(px)
    px[nan_idx] = -1e9
    py[nan_idx] = -1e9

    # Moving median filter to estimate baseline
    px_bline = medfilt(px, moco_kernel)
    py_bline = medfilt(py, moco_kernel)

    # Restore NaNs to vectors
    px_bline[nan_idx] = np.nan
    py_bline[nan_idx] = np.nan

    # Subtract baseline and add central fixation offset
    px_filt = px - px_bline + central_fix[0]
    py_filt = py - py_bline + central_fix[1]

    return px_filt, py_filt, px_bline, py_bline


def KnownFixations(t, px, py, fixations_txt, central_fix):
    '''
    t : 1D float array

    px : 1D float array
        Uncorrected pupil x in video space
    py : 1D float array
        Uncorrected pupil y in video space
    fixations_txt : string
        CSV file containing known fixation times and durations
    central_fix : float tuple
        (x,y) coordinate in video space of central fixation

    Returns
    ----
    px_filt : 1D float array
        Drift corrected video space pupil center x
    py_filt : 1D float array
        Drift corrected video space pupil center y
    '''

    print('*** Motion correction by known fixations not yet implemented ***')

    # Load known central fixations from space-delimited text file into array
    # Columns : fixation number, start time (s), duration (s)
    # fix = np.genfromtxt(fixations_txt)

    # Parse fixation timing array
    # t0 = fix[:,1]
    # t1 = t0 + fix[:,2]

    # TODO: Create central fixation mask for pupilometry
    # for (tc, _) in enumerate(t0):
    #    print('  Fixation from %0.3f to %0.3f' % (t0[tc], t1[tc]))

    # TODO: Estimate linear trend during fixation

    # TODO: Interpolate gaps between fixation periods

    # TODO: Correct pupil center timeseries with estimated trends

    # Return unchanged pupil vectors and zero baseline
    return px.copy(), py.copy(), np.zeros_like(px), np.zeros_like(py)


def PseudoGlint(frame, roi_rect):
    '''
    Estimate pseudo-glint location in frame

    Use robust center of mass of horizontal edge detection as a proxy
    for a missing glint. Use horizontal gradient to improve robustness
    to horizontal scanline artifacts from MRI.

    Arguments
    ----
    frame : 2D numpy uint8 array
        Current video frame
    roi_rect : tuple
        Pupil-iris square ROI ((p0x, p0y), (p1x, p1y))

    Returns
    ----
    px, py : float tuple
        Pseudo-glint location in frame
    '''

    # Debugging display flag
    debug = False

    # Sobel edge detection
    edges = Sobel(frame, gx=True, gy=True)

    # Mask out interior of ROI
    # p0, p1 = roi_rect
    # gxy[p0[1]:p1[1], p0[0]:p1[0]] = 0.0

    # Center of mass of  Sobel edge image
    py, px = center_of_mass(edges)

    if debug:
        rgb = cv2.cvtColor(np.uint8(edges), cv2.COLOR_GRAY2RGB)
        cv2.circle(rgb, (int(px), int(py)), 1, (255,255,0))
        cv2.imshow('PseudoGlint', rgb)
        cv2.waitKey(5)

    return px, py


def Sobel(img, gx=True, gy=False):
    '''
    Sobel edge detection image

    Arguments
    ---
    img : 2D uint8 array
        Grayscale video frame
    gx : boolean
        X-gradient flag
    gy : boolean
        Y-gradient flag
    '''

    # Sobel kernel size
    k = 3

    # Sobel edges in x dimension
    if gx:
        xedge = np.abs(cv2.Sobel(img, cv2.CV_32F, 1, 0, ksize=k))
    else:
        xedge = np.zeros_like(img)

    if gy:
        yedge = np.abs(cv2.Sobel(img, cv2.CV_32F, 0, 1, ksize=k))
    else:
        yedge = np.zeros_like(img)

    edges = xedge + yedge

    # Rescale image to [0,255] 32F
    imax, imin = np.amax(edges), np.amin(edges)
    edges = ((edges - imin) / (imax - imin) * 255).astype(np.float32)

    return edges