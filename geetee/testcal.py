#!/opt/local/bin/python
#
# Calibration test from pupilometry results
#
# USAGE : testcal.py
#
# AUTHOR : Mike Tyszka
# PLACE  : Caltech
# DATES  : 2014-05-15 JMT From scratch
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

import numpy as np
import gtIO
import gtCalibrate as cal
from matplotlib import pyplot as plt

def main():
    
    cal_pupils_csv = '/Users/jmt/Data/Eye_Tracking/Groups/Jaron/videos/26mxk_cal2_choice1/cal_pupils.csv'
    
    # Read pupilometry values from file
    t,area,x,y,blink,dummy =  gtIO.ReadPupilometry(cal_pupils_csv)

    # Generate heatmap
    hmap, xedges, yedges = cal.HeatMap(x, y)
    
    plt.imshow(hmap, interpolation='nearest', origin='low',extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]])
    plt.show()
    
# This is the standard boilerplate that calls the main() function.
if __name__ == '__main__':
    main()
