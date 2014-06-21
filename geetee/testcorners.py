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
    
    # Graphic output
    do_graphics = True
    
    # Border and scaling
    scale = 4
    border = 16
    
    # params for Shi-Tomasi corner detection
    feature_params = dict(maxCorners = 100,
                          qualityLevel = 0.01,
                          minDistance = 20,
                          blockSize = 20)

    # Parameters for lucas-Kanade optical flow
    lk_params = dict(winSize  = (20,20),
                     maxLevel = 4,
                     criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
    
    # Cal-Gaze video pair
    # Final frame of calibration file is motion correction template
    # cal_file = '/Users/jmt/Data/Eye_Tracking/Groups/Jaron/videos/04axa_cal1_choice1/cal.mov'
    gaze_file = '/Users/jmt/Data/Eye_Tracking/Groups/Jaron/videos/04axa_cal1_choice1/gaze.mov'
    
    # Open gaze video stream
    try:
        gaze_stream = cv2.VideoCapture(gaze_file)
    except:
        sys.exit(1)
        
    if not gaze_stream.isOpened():
        sys.exit(1)

    # Load initial frame and find corners
    keep_going, I_old = gtIO.LoadVideoFrame(gaze_stream, scale, border)
    
    if keep_going:
        p_old = cv2.goodFeaturesToTrack(I_old, mask = None, **feature_params)
        
    while keep_going:
            
        keep_going, I_new = gtIO.LoadVideoFrame(gaze_stream, scale, border)
        
        if keep_going:

            # Calculate sparse optical flow around detected features
            # from previous frame
            p_new, st, err = cv2.calcOpticalFlowPyrLK(I_old, I_new, p_old, None, **lk_params)

            good_idx = [st == 1]

            # Select good points
            p_new_good = p_new[good_idx]
            p_old_good = p_old[good_idx]

            # Mean displacement over all features
            dx, dy = np.mean(p_new_good - p_old_good, axis = 0)
            
            print('%6.1f, %6.1f' % (dx, dy))

            # Now update the previous frame and previous points
            I_old = I_new.copy()
            p_old = p_new_good.reshape(-1,1,2)
            
            if do_graphics:
                
                img = cv2.cvtColor(I_old, cv2.COLOR_GRAY2RGB)
                
                for new in p_new_good:
                    x0,y0 = new.astype(int)
                    cv2.circle(img, (x0,y0), 1, (0,255,0))

                cv2.imshow('Corners', img)
                if cv2.waitKey(5) > 0:
                    break
                
    # Close gaze video stream
    gaze_stream.release()
    
# This is the standard boilerplate that calls the main() function.
if __name__ == '__main__':
    main()