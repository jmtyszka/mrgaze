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

from mrgaze.gui import Ui_MainWindow


class MrGazeApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)


if __name__ == '__main__':
    app = QApplication(sys.argv)

    myapp = MrGazeApp()
    myapp.show()

    sys.exit(app.exec_())
