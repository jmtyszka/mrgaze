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

import sys
import numpy as np
import cv2
import ConfigParser
import mrgaze.io as mrio

def main():
    
    # Default configuration
    config = ConfigParser.ConfigParser()
    config = mrio.InitConfig(config)
    
    # Sobel edge detection kernel size
    k = 7

    # Cal-Gaze video pair
    # gaze_file = '/Users/jmt/Data/Eye_Tracking/Groups/Jaron/videos/04axa_cal1_choice1/cal.mov'
    gaze_file = '/Users/jmt/Data/Eye_Tracking/Groups/Jaron/videos/04axa_cal1_choice1/gaze.mov'
    
    # Open gaze video stream
    try:
        gaze_stream = cv2.VideoCapture(gaze_file)
    except:
        sys.exit(1)
        
    if not gaze_stream.isOpened():
        sys.exit(1)

    keep_going = True

    while keep_going:
            
        keep_going, I_new, artifact = mrio.LoadVideoFrame(gaze_stream, config)
    
        if keep_going:

            # Sobel edge detect (both directions)
            xedge = cv2.Sobel(I_new, cv2.CV_32F, 1, 0, ksize=k)
            
            # Rescale image to [0,255] unsigned bytes
            imax, imin = np.amax(xedge), np.amin(xedge)
            xedge = ((xedge - imin) / (imax - imin) * 255).astype(np.uint8)
                        
            
            cv2.imshow('Edges', xedge)
            if cv2.waitKey(5) > 0:
                break
                
    # Close gaze video stream
    gaze_stream.release()
    
    print('Done')
    
# This is the standard boilerplate that calls the main() function.
if __name__ == '__main__':
    main()