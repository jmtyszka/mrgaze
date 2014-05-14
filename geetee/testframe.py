#!/opt/local/bin/python
#
# Single frame pyET test
# - finds pupil
#
# USAGE : pyET_TestFrame.py <Test Frame Image>
#
# AUTHOR : Mike Tyszka
# PLACE  : Caltech
# DATES  : 2014-05-07 JMT From scratch
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

import sys
import pyET_Pupilometry as p
import pyET_IO as io

def main():
    
    # Test frame image passed as command line arg
    if len(sys.argv) > 1:
        test_frame_image = sys.argv[1]
    else:
        # test_frame_image = 'RealPupil.png'
        test_frame_image = 'IdealPupil.png'

    # Load single frame
    frame = io.LoadImage(test_frame_image, 16)
        
    # Run pupilometry on single test frame
    center, axes, angle, thresh = p.FitPupil(frame)
    
    # Display fitted pupil ellipse over original image
    # et.DisplayPupilEllipse(frame, center, axes, angle)
    
    
# This is the standard boilerplate that calls the main() function.
if __name__ == '__main__':
    main()
