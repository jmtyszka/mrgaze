#!/opt/local/bin/python
#
# Gaze tracking calibration
# - use calibration video heatmap and priors
#
# AUTHOR : Mike Tyszka
# PLACE  : Caltech
# DATES  : 2014-05-15 JMT From scratch
#
# This file is part of pyET.
#
#    pyET is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.
#
#    pyET is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.
#
#    You should have received a copy of the GNU General Public License
#   along with pyET.  If not, see <http://www.gnu.org/licenses/>.
#
# Copyright 2014 California Institute of Technology.

import os
import cv2
import gtIO
import numpy as np

def AutoCalibrate(subjsess_dir):

    # Calibration pupilometry file
    cal_pupils_csv = os.path.join(subjsess_dir,'cal_pupils.csv')
    
    if not os.path.isfile(cal_pupils_csv):
        print('Calibration pupilometry not found - returning')
        return False
        
    # Load table from CSV file
    t,area,x,y,blink,dummy = gtIO.ReadPupilometry(cal_pupils_csv)
    
    # Compute calibration video heatmap
    hmap, xedges, yedges = HeatMap(x,y)
    
    # Find fixations in heatmap
    fixations = FindFixations(hmap, xedges, yedges)
    
    # Find optimal biquadratic fixation to target mapping
    cal = CalibrationModel(fixations, targets)
    
    return cal
    

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
    hmap, xedges, yedges = np.histogram2d(x, y, bins = 50, range = hrng)
    
    # Gaussian blur
    if sigma > 0:
        hmap = cv2.GaussianBlur(hmap, (5,5), sigma)
    
    return hmap, xedges, yedges
    
def FindFixations(hmap, xedges, yedges):
    
    pass

def CalibrationModel(fixations, targets):
    
    pass



