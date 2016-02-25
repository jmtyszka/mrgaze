#!/usr/bin/env python
"""
Configuration file supoort

AUTHOR : Mike Tyszka
PLACE  : Caltech
DATES  : 2016-02-22 JMT Update for python3 and built-in configparser
                        Moved InitConfig to top of module

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

Copyright 2014-2016 California Institute of Technology.
"""

import os
import configparser


def InitConfig(config):
    '''
    All the configuration parameters for the MrGaze engine
    used in the config file, with defaults
    '''

    # Add video defaults
    config.add_section('VIDEO')
    config.set('VIDEO','inputextension','.avi')
    config.set('VIDEO','outputextension','.avi')
    config.set('VIDEO','inputfps','29.97')
    config.set('VIDEO','downsampling','1')
    config.set('VIDEO','border','0')
    config.set('VIDEO','rotate','0')

    config.add_section('PREPROC')
    config.set('PREPROC','perclow','0.0')
    config.set('PREPROC','perchigh','50.0')

    config.add_section('PUPILDETECT')
    config.set('PUPILDETECT','enabled','True')
    config.set('PUPILDETECT','specificity','10')
    config.set('PUPILDETECT','scalefactor','1.05')
    config.set('PUPILDETECT','manualroi','[0.5, 0.5, 0.5]')

    config.add_section('PUPILSEG')
    config.set('PUPILSEG','method','manual')
    config.set('PUPILSEG','pupildiameterperc','25.0')
    config.set('PUPILSEG','glintdiameterperc','2.0')
    config.set('PUPILSEG','pupilthresholdperc','50.0')

    config.add_section('PUPILFIT')
    config.set('PUPILFIT','method','ROBUST_LSQ')
    config.set('PUPILFIT','maxiterations','5')
    config.set('PUPILFIT','maxrefinements','5')
    config.set('PUPILFIT','maxinlierperc','95.0')

    config.add_section('ARTIFACTS')
    config.set('ARTIFACTS','mrclean','True')
    config.set('ARTIFACTS','zthresh','8.0')
    config.set('ARTIFACTS','motioncorr','none')
    config.set('ARTIFACTS','mocokernel','151')

    config.add_section('CALIBRATION')
    config.set('CALIBRATION','calibrate','False')
    config.set('CALIBRATION','targetx','[0.5, 0.1, 0.9, 0.1, 0.1, 0.5, 0.1, 0.9, 0.5]')
    config.set('CALIBRATION','targety','[0.5, 0.9, 0.9, 0.1, 0.9, 0.9, 0.5, 0.5, 0.1]')
    config.set('CALIBRATION','heatpercmin','5.0')
    config.set('CALIBRATION','heatpercmax','95')
    config.set('CALIBRATION','heatsigma','2.0')

    config.add_section('OUTPUT')
    config.set('OUTPUT','verbose','True')
    config.set('OUTPUT','graphics','True')
    config.set('OUTPUT','overwrite','True')

    config.add_section('CAMERA')
    config.set('CAMERA','fps','30')

    return config


def LoadConfig(data_dir, subjsess=''):
    """
    Load ET pipeline configuration parameters

    Check first for a global configuration in the root directory,
    then for a specific configuration in the subject/session directory.
    The subj/sess config has precedence.

    Arguments
    ----
    root_dir : string
        Root directory containing videos subdir.
    subjsess_dir : string
        Subject/Session subdirectory.

    Returns
    ----
    config : config object (see ConfigParser package)
        Configuration object.
    """

    # Root config filename
    root_cfg_file = os.path.join(data_dir, 'mrgaze.cfg')

    # Subject/Session config filename
    ss_dir = os.path.join(data_dir, subjsess)
    ss_cfg_file = os.path.join(ss_dir, 'mrgaze.cfg')

    # Create a new parser
    config = configparser.ConfigParser()

    # Check first for subject/session config
    if os.path.isfile(ss_cfg_file):

        # Load existing subj/sess config file
        config.read(ss_cfg_file)

    elif os.path.isfile(root_cfg_file):

        # Load existing root config file
        config.read(root_cfg_file)
    else:

        # Write a new default root config file
        config = InitConfig(config)
        with open(root_cfg_file,'w') as cfg_stream:
            config.write(cfg_stream)

    return config


def SaveConfig(config, data_dir):
    """ Save configuration

    Arguments
    ----
    config : Configuration settings
    data_dir : Directory to store the settings in

    """

    # Root config filename
    root_cfg_file = os.path.join(data_dir, 'mrgaze.cfg')

    with open(root_cfg_file,'w') as cfg_stream:
        config.write(cfg_stream)
