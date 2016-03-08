#!/usr/bin/env python
'''
 Qt5 GUI classes for MrGaze

 AUTHOR : Mike Tyszka
 PLACE  : Caltech
 DATES  : 2016-03-07 JMT Put all GUI classes in a single module

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

import configparser
import os

import cv2
import numpy as np
from PyQt5 import QtCore, QtWidgets, QtGui

from mrgaze import config, engine, utils, qtui


class MrGazeApp(QtWidgets.QMainWindow, qtui.Ui_MrGaze):
    """
    Main application GUI class for MrGaze
    """

    def __init__(self):
        # Init GUI
        super().__init__()
        self.ui = qtui.Ui_MrGaze()
        self.ui.setupUi(self)

    def update_intensity_lb(self):
        # Show the lower bound dial value as a label
        lb = self.ui.Intensity_LB_Dial.value()
        self.ui.Intensity_LB_Label.setText(str(lb))

    def update_intensity_ub(self):
        # Show the upper bound dial value as a label
        ub = self.ui.Intensity_UB_Dial.value()
        self.ui.Intensity_UB_Label.setText(str(ub))


class CaptureVideo(QtCore.QObject):
    """
    OpenCV class captures frame from video feed and passes it to the ShowImage class thread
    """

    # Open the xIQ camera (Ximea API)
    camera_port = cv2.CAP_XIAPI
    camera = cv2.VideoCapture(camera_port)
    VideoSignal = QtCore.pyqtSignal(np.ndarray)

    def __init__(self, parent=None):
        super(CaptureVideo, self).__init__(parent)

    # Decorate start video method as a Qt slot
    @QtCore.pyqtSlot()
    def start_video(self):

        run_video = True

        while run_video:

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


class AnalyseFrame(QtCore.QObject):
    """
    Receives QImage from CaptureVideo, analyses it and emits result to ShowVideoFrame
    """

    AnalysisSignal = QtCore.pyqtSignal(QtGui.QImage)

    def __init__(self, parent=None):
        super(AnalyseFrame, self).__init__(parent)

        # Set up the LBP cascade classifier
        LBP_path = os.path.join(utils._package_root(), ('Cascade_%s/cascade.xml' % 'thorlabs'))
        self.cascade = cv2.CascadeClassifier(LBP_path)

        # Setup configuration
        self.cfg = config.InitConfig(configparser.ConfigParser())

    # Decorate receive image method as a slot
    @QtCore.pyqtSlot(np.ndarray)
    def receive_image(self, cv_image):
        results_rgb = self.analyze_image(cv_image)

        # Convert from opencv color image array to QImage object
        qt_image = QtGui.QImage(results_rgb.data,
                                results_rgb.shape[1],
                                results_rgb.shape[0],
                                results_rgb.strides[0],
                                QtGui.QImage.Format_RGB888)

        self.emit_image(qt_image)

    def analyze_image(self, cv_image):
        # Pass incoming video frame to MrGaze engine
        pupil_ellipse, roi_rect, blink, glint, frame_rgb = engine.PupilometryEngine(cv_image, self.cascade, self.cfg)

        return frame_rgb

    # Decorate emit_image method as a signal
    @QtCore.pyqtSlot()
    def emit_image(self, qt_image):
        self.AnalysisSignal.emit(qt_image)


class ShowVideoFrame(QtWidgets.QWidget):
    """
    Qt widget subclass displaying received video frame in the UI
    """

    def __init__(self, parent=None):
        super(ShowVideoFrame, self).__init__(parent)
        self.qt_image = QtGui.QImage()
        self.setAttribute(QtCore.Qt.WA_OpaquePaintEvent)

    # Override inherited method paintEvent
    def paintEvent(self, event):
        # Paint image data set by set_image method
        painter = QtGui.QPainter(self)
        painter.drawImage(0, 0, self.qt_image)

        # Reset image data
        self.qt_image = QtGui.QImage()

    # Decorate set image method as a receiver slot
    @QtCore.pyqtSlot(QtGui.QImage)
    def set_image(self, img):
        if img.isNull():
            print("Viewer Dropped frame!")

        # Resize incoming img to match widget
        img = img.scaled(128, 128)

        # Set qt_image data for paintEvent method
        self.qt_image = img

        self.update()
