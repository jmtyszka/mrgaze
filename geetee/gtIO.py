#!/opt/local/bin/python
#
# Load a single video frame from an image file
# - optional border trim border argument
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

import os
import cv2
import numpy as np
import ConfigParser

def LoadImage(image_file, border = 0):

    # Initialize frame
    frame = np.array([])

    # load test frame image
    try:
        frame = cv2.imread(image_file)
    except:
        print('Problem opening %s to read' % image_file)
        return frame
        
    # Convert to grayscale image
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Trim border (if requested)
    frame = TrimBorder(frame, border)
    
    return frame

#
# Trim border (typically introduced by video conversion)
#
def TrimBorder(frame, border = 0):
    
    if border > 0:
        
        # Get image dimension
        nx, ny = frame.shape[1], frame.shape[0]
        
        # Set bounding box
        x0 = border
        y0 = border
        x1 = nx - border
        y1 = ny - border
        
        # Make sure bounds are inside image
        x0 = x0 if x0 > 0 else 0
        x1 = x1 if x1 < nx else nx-1
        y0 = y0 if y0 > 0 else 0
        y1 = y1 if y1 < ny else ny-1
        
        # Crop and return
        return frame[y0:y1, x0:x1]
        
    else:
        
        return frame

#
# Load setup file for complete ET pipeline
#
def LoadConfig(root_dir):
    
    # Config filename
    cfg_file = os.path.join(root_dir,'geetee.cfg')    
    
    # Create a new parser
    config = ConfigParser.ConfigParser()

    if os.path.isfile(cfg_file):
        
        # Load existing config file
        config.read(cfg_file)
        
    else:

        # Write a new default config file
        config = InitConfig(config)
        with open(cfg_file,'wb') as cfg_stream:
            config.write(cfg_stream)
            cfg_stream.close()
       
    return config
   

def InitConfig(config):
    
    # Add video defaults
    config.add_section('VIDEO')
    config.set('VIDEO','inputextension','.mov')
    config.set('VIDEO','outputextension','.mov')
    
    config.add_section('RANSAC')
    config.set('RANSAC','maxiterations','5')
    config.set('RANSAC','maxrefinements','3')
    config.set('RANSAC','maxinlierperc','95')
    
    config.add_section('LBP')
    config.set('LBP','strictness','40')
    
    config.add_section('CALIBRATION')
    config.set('CALIBRATION','targetx','0.5, 0.1, 0.9, 0.1, 0.1, 0.5, 0.1, 0.9, 0.5')
    config.set('CALIBRATION','targety','0.5, 0.9, 0.9, 0.1, 0.9, 0.9, 0.5, 0.5, 0.1')
    
    return config
    
#
# Pupilometry CSV IO
#
def ReadPupilometry(pupils_csv):
    
    return np.genfromtxt(pupils_csv, delimiter = ',', unpack = True)
    
#
# Safe mkdir - from http://code.activestate.com/recipes/82465-a-friendly-mkdir/
#
def _mkdir(newdir):
    """works the way a good mkdir should :)
        - already exists, silently complete
        - regular file in the way, raise an exception
        - parent directory(ies) does not exist, make them as well
    """
    if os.path.isdir(newdir):
        pass
    elif os.path.isfile(newdir):
        raise OSError("a file with same name as desired dir ('%s') already exists." % newdir)
    else:
        head, tail = os.path.split(newdir)
        if head and not os.path.isdir(head):
            _mkdir(head)
        if tail:
            os.mkdir(newdir)
