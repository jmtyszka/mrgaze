#!/opt/local/bin/python
#
# Single frame geetee test
# - finds pupil
#
# USAGE : testframe.py <Test Frame Image>
#
# AUTHOR : Mike Tyszka
# PLACE  : Caltech
# DATES  : 2014-05-07 JMT From scratch
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
import cv2
import gtIO
import gtPupilometry as p

def main():
    
    # Test frame image passed as command line arg
    if len(sys.argv) > 1:
        test_img = sys.argv[1]
    else:
        # test_frame_image = 'RealPupil.png'
        test_img = '../Data/BiasedPupil.png'

    # Resampling scalefactor
    sf = 4;
    
    # Set up LBP cascade classifier
    cascade = cv2.CascadeClassifier('Cascade/cascade.xml')
    
    if cascade.empty():
        print('LBP cascade is empty - check Cascade directory exists')
        sys.exit(1)

    # Load single frame
    frame = gtIO.LoadImage(test_img)
    
    # Downsample original NTSC video by 4 in both axes
    nx, ny = frame.shape[1], frame.shape[0]
    nxd, nyd = int(nx / sf), int(ny / sf)

    # Downsample frame
    frd = cv2.resize(frame, (nxd, nyd))
        
    # Call pupilometry engine
    el, roi_rect, blink = p.PupilometryEngine(frd, cascade)
    
    # Display fitted pupil ellipse over original image
    p.DisplayPupilEllipse(frd, el, roi_rect)
    
    
# This is the standard boilerplate that calls the main() function.
if __name__ == '__main__':
    main()
