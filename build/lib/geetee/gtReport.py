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
# This file is part of geetee.
#
#    geetee is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.
#
#    geetee is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.
#
#    You should have received a copy of the GNU General Public License
#   along with geetee.  If not, see <http://www.gnu.org/licenses/>.
#
# Copyright 2014 California Institute of Technology.

import os
import string
import gtIO

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

<h1 style="background-color:#E0E0FF">CBIC Daily QA Report</h1>

<table>
<tr>

<!-- Scanner and acquisition info -->
<td>
  <table>
  <tr> <td> <b>Acquisition Date</b> <td bgcolor="#E0FFE0"> <b>$acq_date</b> </tr>
  <tr> <td> <b>Scanner ID</b>       <td> $scanner_serno </tr>
  <tr> <td> <b>Frequency</b>        <td> $scanner_freq MHz </tr>
  <tr> <td> <b>TR</b>               <td> $TR_ms ms </tr>
  <tr> <td> <b>Volumes</b>          <td> $num_volumes </tr>
  </table>
</td>

<!-- SNR and absolute signal info -->
<td>
  <table>
  <tr> <td> <b>SNR Phantom</b>      <td bgcolor="#E0FFE0"> <b>$phantom_snr</b> </tr>
  <tr> <td> <b>SNR Nyquist</b>      <td> $nyquist_snr </tr>
  <tr> <td> <b>Mean Phantom</b>     <td> $phantom_mean
  <tr> <td> <b>Mean Nyquist</b>     <td> $nyquist_mean
  <tr> <td> <b>Mean Noise</b>       <td> $noise_mean
  </table>
</td>

<!-- Spikes and drift -->
<td>
  <table>
  <tr> <td> <b>Phantom Spikes</b>   <td> $phantom_spikes </tr>
  <tr> <td> <b>Nyquist Spikes</b>   <td> $nyquist_spikes </tr>
  <tr> <td> <b>Noise Spikes</b>     <td> $noise_spikes </tr>
  <tr> <td> <b>Phantom Drift</b>   <td> $phantom_drift % </tr>
  <tr> <td> <b>Nyquist Drift</b>   <td> $nyquist_drift % </tr>
  <tr> <td> <b>Noise Drift</b>     <td> $noise_drift % </tr>
  </table>
</td>

<!-- Center of mass and apparent motion -->
<td>
  <table>
  <tr> <td> <b>CofM (mm)</b>        <td> ($com_x, $com_y, $com_z)
  <tr> <td> <b>Max Disp (um)</b>    <td> ($max_adx, $max_ady, $max_adz)
  </table>
</td>

<br><br>

<!-- Plotted timeseries -->
<table>

<tr>
<td> <h3>Signal, Drift and Noise</h3>
<td> <h3>Temporal Summary Images and Masks</h3>
</tr>

<tr>
<td valign="top"><img src=qa_timeseries.png />
<td valign="top">
<b>tMean</b><br> <img src=qa_mean_ortho.png /><br><br>
<b>tSD</b><br> <img src=qa_sd_ortho.png /><br><br>
<b>Region Mask</b><br> <img src=qa_mask_ortho.png /><br><br>
</tr>

</table>

"""

# Main function
def WriteReport(subjsess_results_dir):
    
    # Report directory
    report_dir = os.path.join(subjsess_results_dir,'report')
    
    # Safely create report dir
    gtIO._mkdir(report_dir)
    
    #
    # HTML report generation
    #

    # Create substitution dictionary for HTML report
    qa_dict = dict([
      ('scanner_serno',  "%d"    % (5555)),
    ])

    # Generate HTML report from template (see above)
    TEMPLATE = string.Template(TEMPLATE_FORMAT)
    html_data = TEMPLATE.safe_substitute(qa_dict)
    
    # Write HTML report page
    report_index = os.path.join(report_dir, 'index.html')
    open(report_index, "w").write(html_data)