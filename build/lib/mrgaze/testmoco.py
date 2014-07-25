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

import os
from mrgaze import moco, config

def main():

    data_dir = '/Users/jmt/Data/Eye_Tracking/Groups/Jaron/data'
    subj_sess = '04axa_cal1_choice1'
    ssv_dir = os.path.join(data_dir, subj_sess, 'videos')
    
    cal_file = os.path.join(ssv_dir, 'cal.mov')
    gaze_file = os.path.join(ssv_dir, 'gaze.mov')

    # Load config
    cfg = config.LoadConfig(data_dir, subj_sess)
    
    if not cfg:
        print('* Configuration file missing - returning')
        return False

    moco.MotionCorrect(cal_file, gaze_file, cfg)
    
    
# This is the standard boilerplate that calls the main() function.
if __name__ == '__main__':
    main()