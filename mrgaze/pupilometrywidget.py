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
import numpy as np
from PyQt5 import QtCore, QtWidgets, QtGui
from skimage import exposure

from mrgaze import engine
from mrgaze import utils, config


class PupilometryWidget(QtWidgets.QWidget):

    def __init__(self, parent=None):

        super(PupilometryWidget, self).__init__(parent)
        self.qt_image = QtGui.QImage()
        self.setAttribute(QtCore.Qt.WA_OpaquePaintEvent)

        # Set up the LBP cascade classifier
        cascade_path = os.path.join(utils._package_root(), ('Cascade_%s/cascade.xml' % 'thorlabs'))
        self.cascade = cv2.CascadeClassifier(cascade_path)

        # Init configuration
        self.cfg = config.InitConfig(configparser.ConfigParser())

        # Initial thresholds
        self.pupil_thresh = 0.0
        self.glint_thresh = 100.0

        # Rubber band ROI selection inits
        self.roi = QtCore.QRect()
        self.rubberband = QtWidgets.QRubberBand(QtWidgets.QRubberBand.Rectangle, self)

    def receive_image(self, cv_image):
        """

        Parameters
        ----------
        cv_image_in : 2D numpy array

        Returns
        -------

        """

        # Immediately resample to 400 x 320
        cv_image = cv2.resize(cv_image, (400, 320))

        # Robust range normalization
        plow, phigh = np.uint8(np.percentile(cv_image, (1, 99)))
        cv_image = exposure.rescale_intensity(cv_image, in_range=(plow, phigh), out_range=(0, 255))

        # Pass incoming frame to MrGaze pupilometry engine
        pupil_ellipse, blink, glint, frame_rgb = engine.pupilometry_engine(cv_image,
                                                                           self.pupil_thresh,
                                                                           self.glint_thresh,
                                                                           self.roi)

        # Convert from OpenCV color image array to QImage object
        self.qt_image = QtGui.QImage(frame_rgb.data,
                                     frame_rgb.shape[1],
                                     frame_rgb.shape[0],
                                     frame_rgb.strides[0],
                                     QtGui.QImage.Format_RGB888)

        if self.qt_image.isNull():
            print("*** Dropped Frame ***")

        # Update widget (calls paintEvent)
        self.update()

    def paintEvent(self, event):
        """
        Override inherited method paintEvent

        Parameters
        ----------
        event

        Returns
        -------

        """

        if not self.qt_image.isNull():
            painter = QtGui.QPainter(self)

            # Get widget dimensions
            w, h = self.width(), self.height()

            # Isotropic enlarge qt_image and center
            my_img = self.qt_image.scaled(w, h, QtCore.Qt.KeepAspectRatio)

            # Center image within widget
            my_rect = my_img.rect()
            my_rect.moveCenter(self.rect().center())

            painter.drawImage(my_rect.topLeft(), my_img)

            # Reset image data
            # self.qt_image = QtGui.QImage()

    #
    # Mouse event handlers
    #

    def mousePressEvent(self, event):

        # Protect existing ROI
        if self.roi.isNull():
            self.origin = event.pos()
            self.rubberband.setGeometry(
                QtCore.QRect(self.origin, QtCore.QSize()))
            self.rubberband.show()

        QtWidgets.QWidget.mousePressEvent(self, event)

    def mouseMoveEvent(self, event):

        if self.rubberband.isVisible():
            self.rubberband.setGeometry(
                QtCore.QRect(self.origin, event.pos()).normalized())
        QtWidgets.QWidget.mouseMoveEvent(self, event)

    def mouseReleaseEvent(self, event):

        if self.rubberband.isVisible():
            self.rubberband.hide()
            self.roi = self.rubberband.geometry()
        QtWidgets.QWidget.mouseReleaseEvent(self, event)

    def mouseDoubleClickEvent(self, event):

        # Clear ROI
        self.roi = QtCore.QRect()

    #
    # Threshold handlers
    #

    def change_pupil_thresh(self, thr):
        self.pupil_thresh = thr
        pass

    def change_glint_thresh(self, thr):
        self.glint_thresh = thr
        pass

    def toggle_pause(self):
        self.paused = not self.paused
        if self.paused:
            print('Paused')
        else:
            print('Running')
