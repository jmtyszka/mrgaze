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

import os
import sys
import gtPupilometry as p
import gtIO

def main():
    
    # Get data directory
    if len(sys.argv) > 1:
        root_dir = sys.argv[1]
    else:
        # data_dir = os.getcwd()
        root_dir = '/Users/jmt/Data/Eye_Tracking/Groups/Jaron'
    
    # Load configuration from file
    # If no config file exists, write a default one
    print ('Loading configuration from geetee.cfg')
    config = gtIO.LoadConfig(root_dir)
    
    videos_root = os.path.join(root_dir,'videos')
    results_root = os.path.join(root_dir,'results')    
        
    # Loop over all subject subdirectories of the data directory
    for subj in os.walk(videos_root).next()[1]:
        
        video_subj_dir = os.path.join(videos_root, subj)        
        
        if os.path.isdir(video_subj_dir):
            
            print('')
            print('Analysing Subject/Session %s' % subj)
            
            print('  Calibration Pupilometry')
            
            cal_video  = os.path.join(video_subj_dir, 'cal.mov')
            # print cal_video
            p.VideoPupilometry(cal_video)
            
            print('  Gaze Pupilometry')

            gaze_video = os.path.join(video_subj_dir, 'gaze.mov')
            # print gaze_video
            #p.VideoPupilometry(gaze_video)

    # Clean exit
    sys.exit(0) 
    
# This is the standard boilerplate that calls the main() function.
if __name__ == '__main__':
    main()
