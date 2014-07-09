#!/opt/local/bin/python
"""
Video pupilometry functions

- takes calibration and gaze video filenames as input
- controls calibration and gaze estimation workflow

Example
-------

>>> mrgaze.py <Calibration Video> <Gaze Video>


Author
------

    Mike Tyszka, Caltech Brain Imaging Center

License
-------

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
---------

    2014 California Institute of Technology.
"""

import numpy as np
from scipy.signal import medfilt
from scipy.ndimage.morphology import binary_dilation
import matplotlib.pyplot as plt

def MRClean(frame, verbose=False):
    """
    Attempt to repair scan lines corrupted by MRI RF or gradient pulses. 
    
    Parameters
    ----------
    frame : numpy integer array
        Original corrupted, interlaced video frame
    verbose : boolean
        Verbose output flag [False]

    Returns
    -------
    frame_clean : numpy integer array
        Repaired interlaced frame

    Example
    -------
    >>>

    """
    
    # Init repaired frame
    frame_clean = frame.copy()
    
    # Init artifact flag
    artifact = False
    
    # Split frame into even and odd lines
    fr_even = frame[0::2,:]
    fr_odd  = frame[1::2,:]
    
    # Odd - even difference
    fr_diff = fr_odd.astype(float) - fr_even.astype(float)
    
    # Absolute row median of frame difference
    med = np.abs(np.median(fr_diff, axis=1))

    # Find scanlines with median row difference > 3.0
    bad_rows = med > 3.0
        
    # Median smooth the bad rows mask then dilate by 3 lines (kernel 2*3+1 = 7)
    bad_rows = medfilt(bad_rows)
    bad_rows = binary_dilation(bad_rows, structure=np.ones((7,)))
    
    # Cast bad rows to integers
    bad_rows = bad_rows.astype(int)
    
    if bad_rows.sum() > 0:
        
        # Set artifact present flag
        artifact = True
        
        # Zero pad bad_rows by 1 at start and end
        bad_rows_pad = np.append(0, np.append(bad_rows, 0))
    
        # Find bad row block start and end indices by forward differencing
        # Add leading and trailing zeros to avoid unterminated blocks
        # Remember this later when determining correct row indices
        dbad = np.diff(bad_rows_pad)
        
        # Bad row block start and end indices
        # bad_on indicates row indices immediately prior to block starts
        # bad_off indicates row indices immediate after block ends
        bad_on = (np.where(dbad > 0))[0] - 1
        bad_off = (np.where(dbad < 0))[0]
        
        if bad_on.size != bad_off.size:
            print('Block start and end arrays differ in size - returning')
            return frame_clean, artifact
        
        # Init cleaned half frames
        fr_odd_clean = fr_odd.copy()
        fr_even_clean = fr_even.copy()
        
        # Recurse over each bad row block
        for i, r0 in enumerate(bad_on):
            
            r1 = bad_off[i]
            
            fr_odd_clean = InpaintRows(fr_odd_clean, r0, r1)
            fr_even_clean = InpaintRows(fr_even_clean, r0, r1)
            
        # Reinterlace cleaned frame
        frame_clean[0::2,:] = fr_odd_clean
        frame_clean[1::2,:] = fr_even_clean
            
        # Display results
        if verbose:
    
            ny = bad_rows.shape[0]
            y = np.arange(0,ny)           
                
            plt.figure(1)
            plt.set_cmap('jet')
                
            plt.subplot(321)
            plt.imshow(fr_odd)
            plt.title('Odd')
            
            plt.subplot(322)
            plt.imshow(fr_even)
            plt.title('Even')
            
            plt.subplot(323)
            plt.imshow(fr_odd_clean)
            plt.title('Odd Repaired')
            
            plt.subplot(324)
            plt.imshow(fr_even_clean)
            plt.title('Even Repaired')
            
            plt.subplot(325)
            plt.imshow(fr_diff)
            plt.title('Odd - Even')
            
            plt.subplot(326)
            plt.plot(y, med / np.max(med), y, bad_rows)
            plt.title('Bad Row Mask')
            
            plt.show()

    return frame_clean, artifact


def InpaintRows(src, r0, r1):
    """
    Repair bad row blocks by vertical linear interpolation
    
    Parameters
    ----
    src : 2D numpy uint8 array
        Original image to be repaired
    r0 : integer
        Row index immediately before start of corrupted rows
    r1 : integer
        Row index immediately after end of corrupted rows
        
    Returns
    ----
    dest : 2D numpy uint8 array
        Repaired image
        
    Example:
    ----
    >>> img_repaired = InpaintRows(img, 5, 25)
    """

    # Init repaired image
    dest = src.copy()
    
    # Protect against overrange rows
    nr = src.shape[0]
    r0 = clamp(r0, 0, nr-1)
    r1 = clamp(r1, 0, nr-1)
    
    # Extract start and end row values
    I0 = (src[r0, :].astype(float)).reshape(1,-1)
    I1 = (src[r1, :].astype(float)).reshape(1,-1)

    # Create row vector for interpolation
    # These rows define the bad row block
    ri = np.arange(1, r1 - r0).reshape(-1,1)
    dr = r1-r0
    
    # Intensity difference between good rows bounding the bad row block
    dI = I1 - I0
    
    # Linear interpolation over bad row block
    Ii = ri.dot(dI / dr) + I0
    
    # Replace bad row block with interpolated values
    dest[(r0+1):r1,:] = np.round(Ii).astype(int)
    
    return dest


def clamp(x, x_min, x_max):
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