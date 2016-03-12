#!/usr/bin/env python3
"""
Start MrGaze application with Qt5 GUI

Example
----
% mrgaze.py

Author
----
Mike Tyszka, Caltech Brain Imaging Center

Dates
----
2016-05-26 JMT From scratch

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

import sys

from PyQt5 import QtWidgets, QtCore

from mrgaze.capture import CaptureVideo
from mrgaze.mainwindow import MainWindow

__version__ = '0.8.0'

if __name__ == '__main__':

    # Create an application object
    app = QtWidgets.QApplication(sys.argv)

    # Create a video capture thread
    thread = QtCore.QThread()
    thread.start()

    # Create a video capture object and add it to the thread
    vidcap = CaptureVideo()
    vidcap.moveToThread(thread)

    # Create the MrGaze application UI
    mainwin = MainWindow()

    # Connect video capture object to pupilometry widget
    vidcap.VideoSignal.connect(mainwin.ui.pupilometryView.receive_image)

    # Connect play button to capture start method
    mainwin.ui.playButton.clicked.connect(vidcap.start_capture)

    # Reveal the UI
    mainwin.show()

    sys.exit(app.exec_())
