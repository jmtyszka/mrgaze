#!/opt/local/bin/python
#
# Gaze tracking report generator
# - collects info from subject/session results directory
# - report placed in report subdirectory
#
# AUTHOR : Mike Tyszka
# PLACE  : Caltech
# DATES  : 2014-05-15 JMT From scratch
#
# This file is part of mrgaze.
#
#    mrgaze is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.
#
#    mrgaze is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.
#
#    You should have received a copy of the GNU General Public License
#   along with mrgaze.  If not, see <http://www.gnu.org/licenses/>.
#
# Copyright 2014 California Institute of Technology.

import os
import string
import pylab as plt
import numpy as np
from mrgaze import pupilometry


# Define template
TEMPLATE_FORMAT = """
<html>

<head>

<STYLE TYPE="text/css">
BODY {
  font-family    : sans-serif;
}
td {
  padding-left   : 10px;
  padding-right  : 10px;
  padding-top    : 0px;
  padding-bottom : 0px;
  vertical-align : top;
}
</STYLE>

</head>

<body>

<h1 style="background-color:#E0E0FF">MRGAZE Report</h1>

<!-- Subject/Session Information -->
<p>
<table>
  <tr><td><h2>Session Information</h2><td></tr>
  <tr><td><b>Subject/Session</b> <td>$subj_sess</tr>
  <tr><td><b>Artifact Start Time</b> <td>$art_t0 seconds</tr>
</table>

<!-- Plotted timeseries -->
<p>
<table>

  <tr><td><h2>Raw Calibration Pupilometry</h2></tr>
  <tr><td valign="top"><img src=cal_pupils_raw.png /></tr>

  <tr><td><h2>Filtered Calibration Pupilometry</h2></tr>
  <tr><td valign="top"><img src=cal_pupils_filt.png /></tr>

  <tr><td><h2>Raw Gaze Pupilometry (Downsampled by 1000)</h2></tr>
  <tr><td valign="top"><img src=gaze_pupils_raw.png /></tr>

  <tr><td><h2>Filtered Gaze Pupilometry (Downsampled by 1000)</h2></tr>
  <tr><td valign="top"><img src=gaze_pupils_filt.png /></tr>

</table>

</body>

</html>
"""

from mrgaze import utils

# Main function
def WriteReport(ss_dir):
    
    # Results subdirectoy
    ss_res_dir = os.path.join(ss_dir, 'results')
    
    # Check that results directory exists
    if not os.path.isdir(ss_res_dir):
        print('* Results directory does not exist - returning')
        return False
        
    # Extract subj/sess name
    subj_sess = os.path.basename(ss_dir)
    
    # Create timeseries plots
    print('  Plot raw calibration video timeseries')
    cal_raw_csv = os.path.join(ss_res_dir, 'cal_pupils_raw.csv')
    cal_raw_png = os.path.join(ss_res_dir, 'cal_pupils_raw.png')
    PlotTimeseries(cal_raw_csv, cal_raw_png)
    
    print('  Plot filtered calibration video timeseries')
    cal_filt_csv = os.path.join(ss_res_dir, 'cal_pupils_filt.csv')
    cal_filt_png = os.path.join(ss_res_dir, 'cal_pupils_filt.png')
    PlotTimeseries(cal_filt_csv, cal_filt_png)
    
    print('  Plot raw gaze video timeseries')
    gaze_raw_csv = os.path.join(ss_res_dir, 'gaze_pupils_raw.csv')
    gaze_raw_png = os.path.join(ss_res_dir, 'gaze_pupils_raw.png')
    PlotTimeseries(gaze_raw_csv, gaze_raw_png)
    
    print('  Plot filtered gaze video timeseries')
    gaze_filt_csv = os.path.join(ss_res_dir, 'gaze_pupils_filt.csv')
    gaze_filt_png = os.path.join(ss_res_dir, 'gaze_pupils_filt.png')
    PlotTimeseries(gaze_filt_csv, gaze_filt_png)
    
    # Estimate time of first artifact
    print('  Locating artifact start time')
    art_t0 = ArtifactStartTime(gaze_raw_csv)
    
    #
    # HTML report generation
    #

    print('  Generating HTML report')

    # Create substitution dictionary for HTML report
    qa_dict = dict([
        ('subj_sess',  "%s"    % (subj_sess)),
        ('art_t0',     "%0.1f" % (art_t0))        
    ])

    # Generate HTML report from template (see above)
    TEMPLATE = string.Template(TEMPLATE_FORMAT)
    html_data = TEMPLATE.safe_substitute(qa_dict)
    
    # Write HTML report page
    report_index = os.path.join(ss_res_dir, 'index.html')
    open(report_index, "w").write(html_data)


def PlotTimeseries(csv_file, plot_png):
    
    if not os.path.isfile(csv_file):
        print('* Pupilometry file not found - returning')
        return False
    
    # Load pupilometry data from CSV file
    p = pupilometry.ReadPupilometry(csv_file)
    
    # Downsample if total samples > 1000
    nt = p.shape[0]
    if nt > 1000:
        dd = int(nt / 1000.0)
        inds = np.arange(0, nt, dd)
        p = p[inds,:]
    
    # Create figure, plot all timeseries in subplots
    fig = plt.figure(figsize = (6,6))
    
    # Extract time vector
    t = p[:,0]

    ax = fig.add_subplot(411)
    ax.plot(t, p[:,1])
    ax.set_title('Corrected Pupil Area', x = 0.5, y = 0.8, fontsize=8)
    ax.tick_params(axis='both', labelsize=8)
    
    ax = fig.add_subplot(412)
    ax.plot(t, p[:,2], p[:,0], p[:,3])
    ax.set_title('Pupil Center', x = 0.5, y = 0.8, fontsize=8)
    ax.tick_params(axis='both', labelsize=8)
        
    ax = fig.add_subplot(413)
    ax.plot(t, p[:,4])
    ax.set_ylim([-0.5, 1.5])
    ax.set_title('Blink', x = 0.5, y = 0.8, fontsize=10)
    ax.tick_params(axis='both', labelsize=8)
        
    ax = fig.add_subplot(414)
    ax.plot(t, p[:,5])
    ax.set_title('Artifact Power', x = 0.5, y = 0.8, fontsize=10)
    ax.tick_params(axis='both', labelsize=8)
    
    # Pack all subplots and labels tightly
    fig.subplots_adjust(hspace = 0.2)

    # Save figure to file in same directory as CSV file
    plt.savefig(plot_png, dpi = 150, bbox_inches = 'tight')    
    

def ArtifactStartTime(csv_file):
    '''
    Estimate the time of the first artifact
    '''

    if not os.path.isfile(csv_file):
        print('* Pupilometry file not found - returning')
        return False
    
    # Load pupilometry data from CSV file
    p = pupilometry.ReadPupilometry(csv_file)
    
    # Frame times in seconds
    t = p[:,0]
    dt = t[1]-t[0]
    
    # Artifact power in each frame (temporally median filtered)
    art = p[:,5]
    
    # Get MAD of first 1.0s of artifact power
    clean_mad = utils._mad(art[0:int(1.0/dt)])
    
    # Threshold at 100 times this value
    art_thresh = 100.0 * clean_mad
    art_on = art > art_thresh
    
    # Time in seconds corresponding to first detected artifact
    art_t0 = t[np.argmax(art_on)]
    
    return art_t0
    