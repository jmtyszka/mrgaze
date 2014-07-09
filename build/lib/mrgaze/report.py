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

<h1 style="background-color:#E0E0FF">MRGAZE Gaze Tracking Report</h1>

<table>
<tr>

<!-- Subject/Session Information -->
<td>
  <table>
    <tr>
      <td><b>Subject/Session</b>
      <td><b>$subj_sess</b>
    </tr>
  </table>
</td>

<!-- Plotted timeseries -->
<table>
  <tr>
    <td valign="top"><img src=pupilometry_timeseries.png />
  </tr>
</table>

</body>

</html>
"""

# Main function
def WriteReport(ss_res_dir):
    
    # Check that results directory exists
    if not os.path.isdir(ss_res_dir):
        print('* Results directory does not exist - returning')
        return False
        
    # Extract subj/sess name
    subj_sess = os.path.basename(ss_res_dir)
    
    # Create timeseries plots
    PlotTimeseries(ss_res_dir)
    
    # Create gaze heatmaps
    PlotHeatmaps(ss_res_dir)
    
    #
    # HTML report generation
    #

    # Create substitution dictionary for HTML report
    qa_dict = dict([
      ('subj_sess',  "%s"    % (subj_sess))
    ])

    # Generate HTML report from template (see above)
    TEMPLATE = string.Template(TEMPLATE_FORMAT)
    html_data = TEMPLATE.safe_substitute(qa_dict)
    
    # Write HTML report page
    report_index = os.path.join(ss_res_dir, 'index.html')
    open(report_index, "w").write(html_data)


def PlotTimeseries(ss_res_dir):
    
    print('Plotting timeseries')
    

def PlotHeatmaps(ss_res_dir):
    
    print('Plotting heatmaps')