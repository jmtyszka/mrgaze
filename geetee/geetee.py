#!/opt/local/Library/Frameworks/Python.framework/Versions/2.7/Resources/Python.app/Contents/MacOS/Python
"""
Perform calibrated gaze estimation on a single cal-gaze video pair
- takes calibration and gaze video filenames as input
- controls calibration and gaze estimation workflow

USAGE : geetee.py <Calibration Video> <Gaze Video>

AUTHOR : Mike Tyszka
PLACE  : Caltech
DATES  : 2014-05-07 JMT From scratch

This file is part of geetee.

    geetee is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    geetee is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.
    You should have received a copy of the GNU General Public License
    along with geetee.  If not, see <http://www.gnu.org/licenses/>.

Copyright 2014 California Institute of Technology.
"""

import os
import sys
import geetee as gt

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
    config = gt.io.LoadConfig(root_dir)
    
    # Video and results root directories
    videos_root = os.path.join(root_dir,'videos')
    results_root = os.path.join(root_dir,'results')
    
    # Check directories exist
    if not os.path.isdir(videos_root):
        print('Video root directory does not exist - exiting')
        sys.exit(1)
    
    # Safely create results directory
    gt.io._mkdir(results_root)
        
    # Loop over all subject subdirectories of the data directory
    for subjsess in os.walk(videos_root).next()[1]:
        
        # Video and results directory names for this subject/session
        subjsess_video_dir = os.path.join(videos_root, subjsess)
        subjsess_results_dir = os.path.join(results_root, subjsess)
        
        if os.path.isdir(subjsess_video_dir):
            
            print('')
            print('Analysing Subject/Session %s' % subjsess)
            
            print('  Calibration Pupilometry')

            # Create results subj/sess dir
            gt.io._mkdir(subjsess_results_dir)
            
            cal_video  = os.path.join(subjsess_video_dir, 'cal.mov')
            gt.pupilometry.VideoPupilometry(cal_video, config)
            
            print('  Gaze Pupilometry')

            gaze_video = os.path.join(subjsess_video_dir, 'gaze.mov')
            gt.pupilometry.VideoPupilometry(gaze_video, config)
            
            print('  Create calibration model')
            C = gt.calibrate.AutoCalibrate(subjsess_video_dir, config)

            print('  Calibrate pupilometry')
            gt.calibrate.ApplyCalibration(subjsess_video_dir, C)
            
            print('  Write report')
            gt.report.WriteReport(subjsess_results_dir)

        else:
            
            print('%s does not exist - skipping' % subjsess_video_dir)

    # Clean exit
    sys.exit(0) 
    
# This is the standard boilerplate that calls the main() function.
if __name__ == '__main__':
    main()
