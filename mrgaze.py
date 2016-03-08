#!/usr/bin/env python
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

from mrgaze.qtclasses import MrGazeApp, CaptureVideo, AnalyseFrame, ShowVideoFrame

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

    # Create an image analysis object
    analyse = AnalyseFrame()

    # Create an image viewer object
    showvid = ShowVideoFrame()

    # Connect the output from the video capture to the analysis slot
    vidcap.VideoSignal.connect(analyse.receive_image)
    analyse.AnalysisSignal.connect(showvid.set_image)

    # Create the MrGaze application UI
    myapp = MrGazeApp()

    # Connect the play button to the video capture start method
    myapp.ui.playButton.clicked.connect(vidcap.start_video)

    # Add the image viewer to the UI
    myapp.ui.videoLayout.addWidget(showvid)

    # Reveal the UI
    myapp.show()

    sys.exit(app.exec_())
