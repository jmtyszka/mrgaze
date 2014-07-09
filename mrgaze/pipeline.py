#!/opt/local/Library/Frameworks/Python.framework/Versions/2.7/Resources/Python.app/Contents/MacOS/Python
"""
Main python eyetracking wrapper

 - takes calibration and gaze video filenames as input
 - controls calibration and gaze estimation workflow

 Example
 ----
 >>> mrgaze_batch.py <Calibration Video> <Gaze Video>

 AUTHOR : Mike Tyszka
 PLACE  : Caltech
 DATES  : 2014-05-07 JMT From scratch

 This file is part of mrgaze.

    mrgaze is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    mrgaze is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
   along with mrgaze.  If not, see <http://www.gnu.org/licenses/>.

Copyright 2014 California Institute of Technology.
"""

import os
import sys

import mrgaze.utils as mru
import mrgaze.pupilometry as mrp
import mrgaze.calibrate as mrc
import mrgaze.report as mrr

def RunBatch(data_dir=[]):
    """
    Run the gaze tracking pipeline over all sessions within a data directory
    """

    # Default data directory
    if not data_dir:
        print('* No data directory provided - exiting')
        return False

    # Check for missing directories
    if not os.path.isdir(data_dir):
        print('* Data directory does not exist - exiting')
        sys.exit(1)
        
    # Loop over all subject subdirectories of the data directory
    for subj_sess in os.walk(data_dir).next()[1]:
     
        # Run single-session pipeline
        RunSingle(data_dir, subj_sess)
        
    # Clean exit
    return True 
    
    
def RunSingle(data_dir, subj_sess):
    """
    Run the gaze tracking pipeline on a single gaze tracking session
    """

    print('')
    print('Running single-session pipeline : ' + subj_sess)
    
    if not data_dir or not subj_sess:
        print('* Data or subject/session directory not provided - returning')
        return False
    
    # Subject/session directory name
    ss_dir = os.path.join(data_dir, subj_sess)     

    # Video and results directory names for this subject/session
    ss_vid_dir = os.path.join(ss_dir, 'videos')
    ss_res_dir = os.path.join(ss_dir, 'results')

    # Load configuration from root directory or subj/sess video dir
    # If no config file exists, a default root config is created
    config = mru.LoadConfig(data_dir, subj_sess)
    
    if not config:
        print('* Configuration file missing - returning')
        return False
        
    # Extract operational flags from config
    do_cal = config.getboolean('CALIBRATION', 'calibrate')
    
    # Run pipeline if video directory present
    if os.path.isdir(ss_vid_dir):
        
        # Create results subj/sess dir
        mru._mkdir(ss_res_dir)
        
        print('')
        print('  Calibration Pupilometry')
        print('  -----------------------')
     
        mrp.VideoPupilometry(data_dir, subj_sess, 'cal', config)

        print('')            
        print('  Gaze Pupilometry')
        print('  -----------------------')
        
        mrp.VideoPupilometry(data_dir, subj_sess, 'gaze', config)
            
        if do_cal:
            
            print('  Create calibration model')
            C = mrc.AutoCalibrate(ss_vid_dir, config)

            print('  Calibrate pupilometry')
            mrc.ApplyCalibration(ss_vid_dir, C)
            
        print('  Write report')
        mrr.WriteReport(ss_res_dir)

    else:
            
        print('%s does not exist - skipping' % ss_vid_dir)
        
    print('Completed single-session pipeline')
        
    return True