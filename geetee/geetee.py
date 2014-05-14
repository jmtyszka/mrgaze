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

import pyET_Pupilometry as p

def main():
    
    # Get data directory
    if len(sys.argv) > 0:
        data_dir = sys.argv[1]
    else:
        data_dir = os.path
    
    # Load configuration from file
    # If no config file exists or there's a problem with the existing file
    # it'll be overwritten
    config = LoadConfig(cfg_file)
    
    # Create output directory tree
    
    # Start gaze estimation workflow
    for subject in config.subjectList:
        
        # Calibration and gaze video filenames
        cal_video = os.path.join(config.dataDir,subject + '_Cal' + 

        # Calibration pupilometry
        ok = p.RunPupilometry(cal_video)
        if not ok: break

        # Gaze pupilometry
        # et.RunPupilometry(gaze_video)

        # Autocalibration

        # Apply calibration to calibration and gaze videos

        # Create report

    # Clean exit
    
    sys.exit(0) 
    
# This is the standard boilerplate that calls the main() function.
if __name__ == '__main__':
    main()
