#!/opt/local/bin/python
"""
Video pupilometry functions

- takes calibration and gaze video filenames as input
- controls calibration and gaze estimation workflow

Example
-------

>>> geetee.py <Calibration Video> <Gaze Video>


Author
------

    Mike Tyszka, Caltech Brain Imaging Center

License
-------

This file is part of geetee.

    geetee is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    geetee is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with geetee.  If not, see <http://www.gnu.org/licenses/>.

Copyright
---------

    2014 California Institute of Technology.
"""

import numpy as np
from scipy.signal import medfilt
from scipy.ndimage.morphology import binary_dilation
import matplotlib.pyplot as plt

def MRClean(frame):
    """
    Attempt to repair scan lines corrupted by MRI RF or gradient pulses. 
    
    Parameters
    ----------
    frame : numpy integer array
        Original corrupted, interlaced video frame

    Returns
    -------
    frame_clean : numpy integer array
        Repaired interlaced frame

    Example
    -------
    >>>

    """
    
    # Verbose output flag
    verbose = False
    
    # Init returned frame
    frame_clean = frame.copy()
    
    # Init artifact flag
    artifact = False
    
    # Split frame into even and odd lines
    fr_even = frame[0::2,:].astype(float)
    fr_odd  = frame[1::2,:].astype(float)
    
    # Odd - even difference
    fr_diff = fr_odd - fr_even
    
    # Row median and IQR
    med = np.median(fr_diff, axis=1)

    # Robust background SD assuming zero median
    # Artifacts are assumed to occupy a minority of pixels
    fr_sd = np.median(np.abs(med)) * 1.48
    
    # Only repair frame if difference sd > 0
    if fr_sd > 0:

        # Z-scores for row medians
        z = med / fr_sd
    
        # Find all rows with |z| > z_crit
        bad_rows = np.abs(z) > 2.0
        
        # Median smooth the bad rows mask then dilate by 3 lines
        # bad_rows = medfilt(bad_rows)
        # bad_rows = binary_dilation(bad_rows, structure=np.ones((7,)))
        
        if bad_rows.sum() > 0:
            
            artifact = True
    
            # Artifacts generally have low variance within a scan line
            # Find which rows have lower SD in odd frame
            sd_even = np.std(fr_even, axis=1)
            sd_odd  = np.std(fr_odd,  axis=1)
            odd_low_sd = sd_odd < sd_even
            
            # Mask for row swaps from even to odd and vise versa
            even_to_odd = odd_low_sd & bad_rows
            odd_to_even = ~odd_low_sd & bad_rows
            
            even_to_odd = medfilt(even_to_odd,13).astype(bool)
            odd_to_even = medfilt(odd_to_even,13).astype(bool)
        
            # Repair
            fr_odd_clean = fr_odd.copy()
            fr_even_clean = fr_even.copy()
            fr_odd_clean[even_to_odd,:] = fr_even[even_to_odd,:]
            fr_even_clean[odd_to_even,:] = fr_odd[odd_to_even,:]
        
            # Reinterleave
            frame_clean[0::2,:] = fr_even_clean
            frame_clean[1::2,:] = fr_odd_clean
            
            # Display results
            if verbose:

                ny = bad_rows.shape[0]
                y = np.arange(0,ny)           
            
                plt.figure(1)
                plt.set_cmap('jet')
            
                plt.subplot(331)
                plt.imshow(fr_odd)
    
                plt.subplot(332)
                plt.imshow(fr_even)

                plt.subplot(333)
                plt.imshow(fr_diff)
            
                plt.subplot(334)
                plt.plot(y, z)
                plt.ylabel('z score')

                plt.subplot(335)
                plt.plot(y, bad_rows)
                plt.axis([0,250,-0.25,1.25])

                plt.subplot(336)
                plt.imshow(frame)

                plt.subplot(337)
                plt.plot(y, sd_odd, y, sd_even)

                plt.subplot(338)
                plt.plot(y, even_to_odd, y, odd_to_even + 1.5)
                plt.axis([0,250,-0.25,2.75])
                plt.legend(('even to odd', 'odd to even'))

                plt.subplot(339)
                plt.imshow(frame_clean)

                plt.show()
    
    return frame_clean, artifact