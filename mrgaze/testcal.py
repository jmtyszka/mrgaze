#!/opt/local/bin/python
#
# Calibration test from pupilometry results
#
# USAGE : testcal.py
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

import numpy as np
import mrigaze.calibrate as mrc

def main():
    
    # Setup target coordinates
    # C, TL, TR, BL, BR, TC, LC, RC, BC
    # 2 x 9 array
    targets = np.array( ((0.5, 0.1, 0.9, 0.1, 0.1, 0.5, 0.1, 0.9, 0.5),
                         (0.5, 0.9, 0.9, 0.1, 0.9, 0.9, 0.5, 0.5, 0.1)) )
    
    # Subject/Session directory
    subjsess_dir = '/Users/jmt/Data/Eye_Tracking/Groups/Jaron/videos/02txw_cal2_choice1'
    # subjsess_dir = '/Users/jmt/Data/Eye_Tracking/Groups/Jaron/videos/26mxk_cal2_choice1'
    
    # LAURA
    # subjsess_dir = '/Volumes/Data/laura/ET_Sandbox/RA0546_Gaze1_JedLive'        
    
    # Autocalibrate using calibration video pupilometry results
    C = mrc.AutoCalibrate(subjsess_dir, targets)
    
    print C
    
# This is the standard boilerplate that calls the main() function.
if __name__ == '__main__':
    main()
