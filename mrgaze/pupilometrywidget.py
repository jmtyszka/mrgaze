#!/usr/bin/env python3
'''
 PupilometryWidget subclass of QWidget for MrGaze UI

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
from PyQt5 import QtCore, QtWidgets, QtGui

from mrgaze import utils, config


class PupilometryWidget(QtWidgets.QWidget):
    def __init__(self, parent=None):

        super(PupilometryWidget, self).__init__(parent)
        self.qt_image = QtGui.QImage()
        self.setAttribute(QtCore.Qt.WA_OpaquePaintEvent)

        # Set up the LBP cascade classifier
        LBP_path = os.path.join(utils._package_root(), ('Cascade_%s/cascade.xml' % 'thorlabs'))
        self.cascade = cv2.CascadeClassifier(LBP_path)

        # Init configuration
        self.cfg = config.InitConfig(configparser.ConfigParser())

        # Other inits
        self.paused = True
        self.last_image = []

    def paintEvent(self, event):
        """
        Paint image data set by set_image method
        Override inherited method paintEvent

        Parameters
        ----------
        event

        Returns
        -------

        """

        painter = QtGui.QPainter(self)
        painter.drawImage(0, 0, self.qt_image)

        # Reset image data
        self.qt_image = QtGui.QImage()

    def analyze_image(self, cv_image):

        # Pass incoming video frame to MrGaze engine
        # pupil_ellipse, roi_rect, blink, glint, frame_rgb = engine.PupilometryEngine(cv_image, self.cascade, self.cfg)

        frame_rgb = cv2.cvtColor(cv_image, cv2.COLOR_GRAY2RGB)

        return frame_rgb

    def set_image(self, qimg):
        """
        Display image in widget

        Parameters
        ----------
        qimg

        Returns
        -------

        """

        if qimg.isNull():

            print("*** Dropped Frame ***")

        else:

            # Get widget dimensions
            w, h = self.geometry().width(), self.geometry().height()

            print(qimg.size())

            # Resize incoming qimg to match widget width
            qimg = qimg.scaled(w, h, QtCore.Qt.KeepAspectRatio)

            # Set qt_image data for paintEvent method
            self.qt_image = qimg

            self.update()

    def change_pupil_thresh(self, thr):
        print('New pupil threshold : %d' % thr)
        pass

    def change_glint_thresh(self, thr):
        print('New glint threshold : %d' % thr)
        pass

    def toggle_pause(self):
        self.paused = not self.paused
        if self.paused:
            print('Paused')
        else:
            print('Running')

    def receive_image(self, cv_image_in):

        print('Received Frame')

        # if self.paused:
        #    cv_image_in = self.last_image
        # else:
        #    self.last_image = cv_image_in

        # Run engine on video frame
        results_rgb = self.analyze_image(cv_image_in)

        # Convert from opencv color image array to QImage object
        qt_image = QtGui.QImage(results_rgb.data,
                                results_rgb.shape[1],
                                results_rgb.shape[0],
                                results_rgb.strides[0],
                                QtGui.QImage.Format_RGB888)

        # Pass the final RGB QImage to the eye widget
        self.set_image(qt_image)
