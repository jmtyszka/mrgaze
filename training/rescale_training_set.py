#!/usr/bin/env python

"""

This script can be used for rescaling the training set, so that the
gray levels in the training set images are more similar to those the
classifier will have to deal with during pupil detection during
eye-tracking

Example
-------

>>> rescale_training_set.py <in_path> <out_path>

Author
------

    Wolfgang M. Pauli, Caltech

License
-------

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

Copyright
---------

    2016 California Institute of Technology.
"""

import sys
sys.path.insert(0,'../mrgaze')
import improc

import cv2
import glob
import os

if len(sys.argv) < 3:
    print("Usage: python %s <in_path> <out_path>" % sys.argv[0])
    exit(1)

in_path = sys.argv[1]
out_path = sys.argv[2]

if not os.path.isdir(in_path):
    print "Please provide the directory in which to look for the RAW training set images!"
    exit(1)

if not os.path.isdir(out_path):
    print("Please provide the path to an existing directory for the output")
    exit(1)

in_files = glob.glob(os.path.join(in_path,'*jpg'))

for filename in in_files:
    img = cv2.imread(filename)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    in_path, filename = os.path.split(filename)
    cv2.imshow('raw', img)

    cv2.waitKey(10)

    img_rs = improc.RobustRescale(img, perc_range=(0.5,70))
    
    cv2.imshow('raw', img_rs)

    cv2.waitKey(10)
    
    out_file = os.path.join(out_path, filename)

    cv2.imwrite(out_file, img_rs)

cv2.destroyAllWindows()
