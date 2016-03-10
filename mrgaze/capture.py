#!/usr/bin/env python3
'''
 Video capture class

 AUTHOR : Mike Tyszka
 PLACE  : Caltech
 DATES  : 2016-03-09 JMT Separate video capture class module

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

 Copyright 2016 California Institute of Technology.
'''

import cv2
import numpy as np
from PyQt5 import QtCore


class CaptureVideo(QtCore.QObject):
    '''
    Video capture class with OpenCV 3 and Qt5 support
    '''

    # Open the xIQ camera (Ximea API)
    camera_port = cv2.CAP_XIAPI
    camera = cv2.VideoCapture(camera_port)
    VideoSignal = QtCore.pyqtSignal(np.ndarray)

    def __init__(self, parent=None):
        super(CaptureVideo, self).__init__(parent)
        self.do_capture = False

    @QtCore.pyqtSlot()
    def start_video(self):

        # Start eternal capture loop
        while True:

            # Grab image from camera stream
            ret, cv_image = self.camera.read()

            # Detect camera channels
            ch = cv_image.ndim

            # Convert from BGR or BGRA to grayscale if necessary
            if ch == 3:
                cv_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
            elif ch == 4:
                cv_image = cv2.cvtColor(cv_image, cv2.COLOR_BGRA2GRAY)

            self.VideoSignal.emit(cv_image)
