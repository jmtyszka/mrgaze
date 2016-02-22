#!/opt/local/bin/python
"""
Utility functions, primarily for I/O

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

Copyright 2014 California Institute of Technology.
"""

import os
import sys
import numpy as np
from scipy.ndimage import generic_filter
from scipy.stats import nanmedian


def _mkdir(newdir):
    """
    Safe mkdir accounting for existing filenames, exits cleanly

    Source
    ----
    http://code.activestate.com/recipes/82465-a-friendly-mkdir/
    """
    if os.path.isdir(newdir):
        pass
    elif os.path.isfile(newdir):
        raise OSError("a file with same name as desired dir ('%s') already exists." % newdir)
    else:
        head, tail = os.path.split(newdir)
        if head and not os.path.isdir(head):
            _mkdir(head)
        if tail:
            os.mkdir(newdir)


def _package_root():
    """
    Safely determine absolute path to mrclean install directory

    Source
    ----
    https://wiki.python.org/moin/Distutils/Tutorial
    """

    try:
        root = __file__
        if os.path.islink (root):
            root = os.path.realpath(root)
        return os.path.dirname (os.path.abspath(root))
    except:
        print("I'm sorry, but something is wrong.")
        print("There is no __file__ variable. Please contact the author.")
        sys.exit(1)


def _clamp(x, x_min, x_max):
    """
    Clamp value to range

    Arguments
    ----
    x : scalar
        Value to be clamped
    x_min : scalar
        Minimum allowable value
    x_max : scalar
        Maximum allowable value

    Returns
    ----
    x clamped to range [x_min, x_max]

    Example
    ----
    >>> clamp(11, 0, 10)
    >>> 10
    """

    return np.max((x_min, np.min((x, x_max))))


def _forceodd(x):
    '''
    Force float array values to nearest odd integer
    '''

    return int(x/2.0)*2 + 1


def _rms(x):
    '''
    Root mean square of flattened array
    '''

    return np.sqrt(np.mean(x.flatten()**2))


def _mad(x):
    '''
    Median absolute deviation from the median
    '''

    return np.median(np.abs(x.flatten()))


def _nanmedfilt(x, k):
    '''
    1D moving median filter with NaN masking
    '''

    return generic_filter(x, nanmedian, k)


def _touint8(x):
    '''
    Rescale and cast arbitrary number x to uint8
    '''

    x = np.float32(x)

    x_min, x_max = x.min(), x.max()

    y = (x - x_min) / (x_max - x_min) * 255.0

    return np.uint8(y)