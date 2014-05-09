#!/opt/local/bin/python
#
# Main python eyetracking wrapper
# - takes calibration and gaze video filenames as input
# - controls calibration and gaze estimation workflow
#
# USAGE : pyET.py <Calibration Video> <Gaze Video>
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
import os
import string
import numpy as np
import scipy as sp
import cv2

import pyET_Pupilometry as et

def main():
    
    # Get calibration and gaze video filenames from command line
    if len(sys.argv) >= 3:
        cal_video = sys.argv[1]
        gaze_video = sys.argv[2]
    else:
        print('USAGE : pyET.py <Calibration Video> <Gaze Video>')
        sys.exit(1)
    
    print('  Calibration video : ' + cal_video)
    print('  Gaze video        : ' + gaze_video)

    # Analyze calibration video
    et.RunPupilometry(cal_video)

    # Analyze gaze video
    et.RunPupilometry(gaze_video)

    # Clean up and exit
    sys.exit(0)   
    
    
# This is the standard boilerplate that calls the main() function.
if __name__ == '__main__':
    main()
