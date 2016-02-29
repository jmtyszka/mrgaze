#!/usr/bin/env python
"""
Open MrGaze GUI

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

from PyQt5.QtWidgets import QApplication, QMainWindow
from mrgaze.gui import Ui_MrGaze
from mrgaze import media


class MrGazeApp(QMainWindow, Ui_MrGaze):

    def __init__(self):

        # Init GUI
        super().__init__()
        self.ui = Ui_MrGaze()
        self.ui.setupUi(self)

        self.detect_devices()

    def update_intensity_lb(self):
        lb = self.ui.Intensity_LB_Dial.value()
        self.ui.Intensity_LB_Label.setText(str(lb))

    def update_intensity_ub(self):
        ub = self.ui.Intensity_UB_Dial.value()
        self.ui.Intensity_UB_Label.setText(str(ub))

    def detect_devices(self):
        camera_list = media.find_cameras(maxdevno=10)


if __name__ == '__main__':
    app = QApplication(sys.argv)

    myapp = MrGazeApp()
    myapp.show()

    sys.exit(app.exec_())
