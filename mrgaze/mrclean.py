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
import pywt
import matplotlib.pyplot as plt
from scipy.signal import medfilt
from scipy.ndimage.morphology import binary_dilation
from mrgaze import utils

def MRClean(frame, z_thresh=8.0):
    """
    Attempt to repair scan lines corrupted by MRI RF or gradient pulses. 
    
    Parameters
    ----------
    frame : numpy integer array
        Original corrupted, interlaced video frame
    cfg : configuration object
        Pipeline configuration parameters

    Returns
    -------
    frame_clean : numpy integer array
        Repaired interlaced frame
    art_power : float
        Artifact power in frame

    Example
    -------
    >>>

    """
    
    # Internal debug flag
    DEBUG = False
    
    # Init repaired frame
    frame_clean = frame.copy()
    
    # Split frame into even and odd lines
    fr_even = frame[0::2,:]
    fr_odd  = frame[1::2,:]
    
    # Odd - even frame difference
    df = fr_odd.astype(float) - fr_even.astype(float)

    # Row mean of frame difference
    df_row_mean = np.mean(df, axis=1)
    
    # Artifact power - mean square of row means
    art_power = np.mean(df_row_mean**2)

    # Robust estimate of noise SD in row projection
    sd_n = WaveletNoiseSD(df_row_mean)
    
    # Frame difference projection z-scores
    z = df_row_mean / sd_n

    # Find scanlines with |z| > z_thresh
    bad_rows = np.abs(z) > z_thresh
        
    # Median smooth the bad rows mask then dilate by 3 lines (kernel 2*3+1 = 7)
    bad_rows = medfilt(bad_rows)
    bad_rows = binary_dilation(bad_rows, structure=np.ones((7,)))

    # If an artifact is present    
    if np.sum(bad_rows) > 0:
        
        # Add leading and trailing zero to bad rows flag array
        # This lets forward difference work correctly below
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
            return frame_clean, art_power
        
        # Init cleaned half frames
        fr_odd_clean = fr_odd.copy()
        fr_even_clean = fr_even.copy()
        
        # Loop over last good rows before bad blocks
        for i, r0 in enumerate(bad_on):
            
            # First good row after bad block
            r1 = bad_off[i]
            
            # Protect against overrange rows
            # This reduces artifact cleanup effectiveness if bad rows
            # Are adjacent to the top or bottom of the frame
            nr = fr_odd.shape[0]
            r0 = utils._clamp(r0, 0, nr-1)
            r1 = utils._clamp(r1, 0, nr-1)
            
            # Linear interp between leading and trailing good rows
            odd_interp = InterpRows(fr_odd, r0, r1)
            even_interp = InterpRows(fr_even, r0, r1)
                
            # Extract equivalent rows from original odd and even frames
            odd_orig = fr_odd[r0:r1+1,:]
            even_orig = fr_even[r0:r1+1,:]
            
            # Calculate RMS difference between interp and orig
            odd_rms_diff = utils._rms(odd_orig - odd_interp)
            even_rms_diff = utils._rms(even_orig - even_interp)
            
            # If RMS diff for odd < even, consider odd rows, clean
            # and vise versa
            if odd_rms_diff < even_rms_diff:
                fr_even_clean[r0:r1+1,:] = odd_orig
            else:
                fr_odd_clean[r0:r1+1,:] = even_orig
            
        # Reinterlace cleaned frame
        frame_clean[0::2,:] = fr_odd_clean
        frame_clean[1::2,:] = fr_even_clean
            
        # Display results
        if DEBUG:
    
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
            plt.imshow(df)
            plt.title('Odd - Even')
            
            plt.subplot(326)
            plt.plot(y, z, y, bad_rows * z.max() * 0.9)
            plt.title('Z-score and Bad Row Mask')
            
            plt.show()

    return frame_clean, art_power


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
    
    # Linear interpolation over bad row block
    Ii = InterpRows(src, r0, r1)
    
    # Replace bad row block with interpolated values
    dest[r0:r1+1,:] = np.round(Ii).astype(int)
    
    return dest
    

def InterpRows(src, r0, r1):
    '''
    Create a linear interpolation block between two rows (inclusive)
    
    Arguments
    ----
    src : 2D numpy float array
        Source image for start and end rows. 
    r0 : integer
        Starting row index within src.
    r1 : integer
        Ending row index within src.
        
    Returns
    ----
    row_block : 2D numpy float array
        Interpolation between rows r0 and r1 inclusive.
        
    '''
        

    
    # Extract image rows r0 and r1
    I0 = (src[r0, :].astype(float)).reshape(1,-1)
    I1 = (src[r1, :].astype(float)).reshape(1,-1)

    # Create vector of row indices for interpolation
    # NOTE : np.arange(0,n) generates [0,1,2,...,n-1]
    # so we need to +1 the number of elements for inclusion
    # of the trailing row.
    f = (np.arange(0, r1 - r0 + 1).reshape(-1,1)) / float(r1-r0)
    
    # Linear interpolation over bad row block
    row_block = f.dot(I1-I0) + I0
    
    return row_block


def NoiseSD(x):
    '''
    Robust background noise SD estimation
    '''
    
    return np.median(np.abs(x.flatten())) * 1.48
    

def WaveletNoiseSD(x):
    '''
    Estimate noise SD from wavelet detail coefficients
    '''
    
    # Wavelet decomposition
    cA, cD = pywt.dwt(x.flatten(), 'db1')    
    
    # Estimate sd_n from MAD of detail coefficients
    sd_n = np.median(np.abs(cD)) * 1.48
    
    return sd_n