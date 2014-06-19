#!/opt/local/bin/python
#
# Test head motion correction
#
# USAGE : testmoco.py
#
# AUTHOR : Mike Tyszka
# PLACE  : Caltech
# DATES  : 2014-05-15 JMT From scratch
#
# This file is part of geetee.
#
#    geetee is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.
#
#    geetee is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.
#
#    You should have received a copy of the GNU General Public License
#   along with geetee.  If not, see <http://www.gnu.org/licenses/>.
#
# Copyright 2014 California Institute of Technology.

import sys
import numpy as np
import cv2
import gtIO

def main():
    
    # Border and scaling
    scale = 4
    border = 16
    
    # params for Shi-Tomasi corner detection
    feature_params = dict(maxCorners = 25,
                          qualityLevel = 0.01,
                          minDistance = 10,
                          blockSize = 10)

    # Parameters for lucas-Kanade optical flow
    lk_params = dict(winSize  = (15,15),
                     maxLevel = 2,
                     criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
    
    # Example calibration video
    # v_in_file = '/Users/jmt/Data/Eye_Tracking/Groups/Jaron/videos/04axa_cal1_choice1/cal.mov'
    v_in_file = '/Users/jmt/Data/Eye_Tracking/Groups/Jaron/videos/04axa_cal1_choice1/gaze.mov'
    
    # Open video
    try:
        v_in = cv2.VideoCapture(v_in_file)
    except:
        sys.exit(1)
        
    if not v_in.isOpened():
        sys.exit(1)

    # Load initial frame and find corners
    keep_going, I_old = gtIO.LoadVideoFrame(v_in, scale, border)
    
    if keep_going:
        p_old = cv2.goodFeaturesToTrack(I_old, mask = None, **feature_params)
        
    while keep_going:
            
        keep_going, I_new = gtIO.LoadVideoFrame(v_in, scale, border)
        
        if keep_going:

            # Calculate sparse optical flow from previous frame
            p_new, st, err = cv2.calcOpticalFlowPyrLK(I_old, I_new, p_old, None, **lk_params)

            # Select good points
            good_new = p_new[st == 1]
            good_old = p_old[st == 1]

            # Mean displacement
            dx, dy = np.mean(good_new - good_old, axis = 0)
            
            print('%6.1f, %6.1f' % (dx, dy))

            # Now update the previous frame and previous points
            I_old = I_new.copy()
            p_old = good_new.reshape(-1,1,2)

            
    # Close video stream
    v_in.release()
    
# This is the standard boilerplate that calls the main() function.
if __name__ == '__main__':
    main()