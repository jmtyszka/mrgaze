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

import pyET_Pupilometry as p

def main():
    
    # JARON
    v_file = '/Users/jmt/Data/Eye_Tracking/Groups/Jaron/02txw/02txw_cal2_choice1_Cal.mov'   
    # v_file = '/Users/jmt/Data/Eye_Tracking/Groups/Jaron/13axg/13axg_cal4_choice5_Cal.mov'
    # v_file = '/Users/jmt/Data/Eye_Tracking/Groups/Jaron/02txw/02txw_cal2_choice1_Gaze.mov' 
    # v_file = '/Users/jmt/Data/Eye_Tracking/Groups/Jaron/03axs/03axs_cal1_choice1_Cal.mov' 

    # LAURA
    # v_file = '/Volumes/Data/laura/ET_Sandbox/RA0077_Gaze3/RA0077_Gaze3_JedRecorded_Cal.mpg'
    # v_file = '/Volumes/Data/laura/ET_Sandbox/RA0546_Gaze1/RA0546_Gaze1_JedLive_Cal.mpg'
    
    p.VideoPupilometry(v_file, rot = 0)

# This is the standard boilerplate that calls the main() function.
if __name__ == '__main__':
    main()