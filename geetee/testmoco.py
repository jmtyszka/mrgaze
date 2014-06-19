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

import gtMoCo

def main():
    
    # Large shift between cal and gaze videos
    cal_file = '/Users/jmt/Data/Eye_Tracking/Groups/Jaron/videos/04axa_cal1_choice1/cal.mov'
    gaze_file = '/Users/jmt/Data/Eye_Tracking/Groups/Jaron/videos/04axa_cal1_choice1/gaze.mov'

    gtMoCo.MotionCorrect(cal_file, gaze_file)
    
    
# This is the standard boilerplate that calls the main() function.
if __name__ == '__main__':
    main()