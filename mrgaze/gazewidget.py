#!/usr/bin/env python
'''
 GazeWidget subclass of QWidget for MrGaze UI

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

from PyQt5 import QtCore, QtWidgets, QtGui


class GazeWidget(QtWidgets.QWidget):
    def __init__(self, parent=None):
        super(GazeWidget, self).__init__(parent)
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
    def set_image(self, qimg):

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
