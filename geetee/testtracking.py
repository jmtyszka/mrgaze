#!/opt/local/bin/python
#
# Single frame geetee test
# - finds pupil
#
# USAGE : geetee_TestFrame.py <Test Frame Image>
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

import gtPupilometry as p

def main():
    
    # JARON
    # v_file = '/Users/jmt/Data/Eye_Tracking/Groups/Jaron/videos/26mxk_cal2_choice1/cal.mov'   
    # v_file = '/Users/jmt/Data/Eye_Tracking/Groups/Jaron/videos/02txw_cal2_choice1/cal.mov'   
    
    # LAURA
    v_file = '/Volumes/Data/laura/ET_Sandbox/RA0546_Gaze1_JedLive/RA0546_Gaze1_JedLive_Gaze.mpg'    
    
    p.VideoPupilometry(v_file, rot = 0)

# This is the standard boilerplate that calls the main() function.
if __name__ == '__main__':
    main()