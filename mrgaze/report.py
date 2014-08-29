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
from mrgaze import pupilometry, calibrate


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

  <tr><td><h2>Calibrated Gaze Results</h2></tr>
  <tr><td valign="top"><img src=gaze_calibrated.png /></tr>

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
    PlotPupilometry(cal_raw_csv, cal_raw_png)
    
    print('  Plot filtered calibration video timeseries')
    cal_filt_csv = os.path.join(ss_res_dir, 'cal_pupils_filt.csv')
    cal_filt_png = os.path.join(ss_res_dir, 'cal_pupils_filt.png')
    PlotPupilometry(cal_filt_csv, cal_filt_png)
    
    print('  Plot raw gaze video timeseries')
    gaze_pupils_raw_csv = os.path.join(ss_res_dir, 'gaze_pupils_raw.csv')
    gaze_pupils_raw_png = os.path.join(ss_res_dir, 'gaze_pupils_raw.png')
    PlotPupilometry(gaze_pupils_raw_csv, gaze_pupils_raw_png)
    
    print('  Plot filtered gaze video timeseries')
    gaze_pupils_filt_csv = os.path.join(ss_res_dir, 'gaze_pupils_filt.csv')
    gaze_pupils_filt_png = os.path.join(ss_res_dir, 'gaze_pupils_filt.png')
    PlotPupilometry(gaze_pupils_filt_csv, gaze_pupils_filt_png)
    
    print('  Plot calibrated gaze results')
    gaze_csv = os.path.join(ss_res_dir, 'gaze_calibrated.csv')
    gaze_png = os.path.join(ss_res_dir, 'gaze_calibrated.png')
    PlotGaze(gaze_csv, gaze_png)
    
    # Estimate time of first artifact
    print('  Locating artifact start time')
    art_t0 = ArtifactStartTime(gaze_pupils_filt_csv)
    
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


def PlotPupilometry(csv_file, plot_png):
    '''
    Read pupilometry CSV and plot timeseries
    '''
    
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
        
    # Extract timeseries
    t        = p[:,0]
    area     = p[:,1]
    px, py   = p[:,2], p[:,3]
    gx, gy   = p[:,4], p[:,5]
    pgx, pgy = p[:,6], p[:,7]
    blink    = p[:,8]
    art      = p[:,9]
        
    # Create figure, plot all timeseries in subplots
    fig = plt.figure(figsize = (6,8))

    ax = fig.add_subplot(611)
    ax.plot(t, area)
    ax.set_title('Corrected Pupil Area', y=1.1, fontsize=8)
    ax.tick_params(axis='both', labelsize=8)
    
    ax = fig.add_subplot(612)
    ax.plot(t, pgx, t, pgy)
    ax.set_title('Corrected Pupil Center', y=1.1, fontsize=8)
    ax.tick_params(axis='both', labelsize=8)
    
    ax = fig.add_subplot(613)
    ax.plot(t, px, t, py)
    ax.set_title('Raw Pupil Center', y=1.1, fontsize=8)
    ax.tick_params(axis='both', labelsize=8)    
    
    ax = fig.add_subplot(614)
    ax.plot(t, gx, t, gy)
    ax.set_title('Pseudoglint Center', y=1.1, fontsize=8)
    ax.tick_params(axis='both', labelsize=8)    
    
    ax = fig.add_subplot(615)
    ax.plot(t, blink)
    ax.set_ylim([-0.1, 1.1])
    ax.set_title('Blink', y=1.1, fontsize=8)
    ax.tick_params(axis='both', labelsize=8)
        
    ax = fig.add_subplot(616)
    ax.plot(t, art)
    ax.set_title('Artifact Power', y=1.1, fontsize=8)
    ax.tick_params(axis='both', labelsize=8)
    
    # Pack all subplots and labels tightly
    fig.subplots_adjust(hspace=0.6)

    # Save figure to file in same directory as CSV file
    plt.savefig(plot_png, dpi = 150, bbox_inches = 'tight')    
    
    # Close figure without showing it
    plt.close(fig)

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
    
    
def PlotGaze(csv_file, plot_png):
    '''
    Plot calibrated gaze results in a single figure
    '''
    
    if not os.path.isfile(csv_file):
        print('* Calibrated gaze file not found - returning')
        return False
    
    # Load calibrated gaze timeseries from CSV file
    t, gaze_x, gaze_y = calibrate.ReadGaze(csv_file)
    
    # Create heatmap of calibrated gaze
    hmap, xedges, yedges = calibrate.HeatMap(gaze_x, gaze_y, (0.0, 1.0), (0.0,1.0), sigma=1.0)
    
    # Create figure, plot timeseries and heatmaps in subplots
    fig = plt.figure(figsize = (6,6))

    ax = fig.add_subplot(211)
    ax.plot(t, gaze_x, t, gaze_y)
    ax.set_title('Calibrated Normalized Gaze Position', fontsize=8)
    ax.tick_params(axis='both', labelsize=8)
    ax.set_ylim((-1.0, 2.0))
    
    ax = fig.add_subplot(223)
    ax.scatter(gaze_x, gaze_y, s=1)
    ax.tick_params(axis='both', labelsize=8)
    ax.set_xlim((-0.1, 1.1))
    ax.set_ylim((-0.1, 1.1))
        
    ax = fig.add_subplot(224)
    ax.imshow(hmap, interpolation='bicubic', aspect='equal', origin='lower',
                   extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]])
    ax.tick_params(axis='both', labelsize=8)
    ax.set_xlim((-0.1, 1.1))
    ax.set_ylim((-0.1, 1.1))
    
    # Pack all subplots and labels tightly
    fig.subplots_adjust(hspace = 0.2)

    # Save figure to file in same directory as CSV file
    plt.savefig(plot_png, dpi = 150, bbox_inches = 'tight')    
    
    # Close figure without showinging it
    plt.close(fig)