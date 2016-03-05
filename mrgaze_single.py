#!/usr/bin/env python
"""
Run gaze tracking pipeline on a single subject/session

Example
----
% mrgaze_single.py /Users/jmt/Data/Eye_Tracking/iSight_Test/test

Author
----
Mike Tyszka, Caltech Brain Imaging Center

Dates
----
2014-05-07 JMT From scratch

License
----
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
----
2014-2016 California Institute of Technology.
"""

__version__ = '0.8.0'

import argparse
import datetime as dt
import os

from mrgaze import pipeline


def main():

    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Analyze single session eye tracking video')
    parser.add_argument('-d', '--ss_dir', required=False, help="Single session directory with videos subdirectory")

    # Parse command line arguments
    args = parser.parse_args()

    # Get single session directory from command line
    if args.ss_dir:
        ss_dir = args.ss_dir
    else:
        ss_dir = os.path.join(os.getenv("HOME"), 'mrgaze')

    # Split subj/session directory path into data_dir and subj/sess name
    data_dir, subj_sess = os.path.split(os.path.abspath(ss_dir))

    # Text splash
    print('')
    print('--------------------------------------------------')
    print('mrgaze Single Session Gaze Tracking Video Analysis')
    print('--------------------------------------------------')
    print('Version   : %s' % __version__)
    print('Date      : %s' % dt.datetime.now())
    print('Data dir  : %s' % data_dir)
    print('Subj/Sess : %s' % subj_sess)

    # Run single-session pipeline
    pipeline.RunSingle(data_dir, subj_sess)


# This is the standard boilerplate that calls the main() function.
if __name__ == '__main__':
    main()
