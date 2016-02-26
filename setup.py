from setuptools import setup, find_packages

setup ( name = 'mrgaze',
        version = '0.7.2',
        description = 'Video pupilometry and gaze tracking library',
        author = ['Mike Tyszka & Wolfgang Pauli'],
        author_email = ['jmt@caltech.edu'],
        url = ['https://github.com/jmtyszka/mrgaze'],
        license = 'LICENSE.txt',
        packages = find_packages(),
        package_data = {'mrgaze': ['Cascade_*/*']},
        scripts = ['mrgaze_single.py','mrgaze_batch.py','mrgaze_live.py'],
      )
