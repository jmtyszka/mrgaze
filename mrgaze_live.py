#!/usr/bin/env python

"""
Run gaze tracking on live camera feed

Example
----
% mrgaze_live.py [output_directory]

Authors
----
Mike Tyszka, Caltech Brain Imaging Center
Wolfgang Pauli, HSS, Caltech

Dates
----
2014-05-07 JMT From scratch
2016-01-28 WMP Adpat mrgaze_single.py for live video
2016-02-24 JMT Update command line args with -d

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
2016 California Institute of Technology.
"""

__version__ = '0.7.2'

import argparse
import datetime as dt

from mrgaze import pupilometry


def main():

    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Do real time Eye Tracking.')
    parser.add_argument('-d', '--ss_dir', required=False,
                        help="Single session directory. This is either the location of previously recorded data, or the destination directory for live eye-tracking.")
    parser.add_argument('-p', action="store_true",
                        help="Do preliminary post-processing of previously recorded data, rather than live eye-tracking.")

    # Parse command line arguments
    args = parser.parse_args()

    # Get single session directory from command line
    if args.ss_dir:
        data_dir = args.ss_dir
    else:
        data_dir = ''

    # Text splash
    print('')
    print('--------------------------------------------------')
    print('mrgaze Single Session Gaze Tracking Video Analysis')
    print('--------------------------------------------------')
    print('Version   : %s' % __version__)
    print('Date      : %s' % dt.datetime.now())
    print('Data dir  : %s' % data_dir)

    # Run single-session pipeline
    pupilometry.LivePupilometry(data_dir, not args.p)


# This is the standard boilerplate that calls the main() function.
if __name__ == '__main__':
    main()
