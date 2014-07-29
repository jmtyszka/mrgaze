from setuptools import setup, find_packages

setup ( name = 'mrgaze',
        version = '0.3.2',
        description = 'Offline video pupilometry and gaze tracking',
        author = ['Mike Tyszka'],
        author_email = ['jmt@caltech.edu'],
        url = ['https://github.com/jmtyszka/geetee'],
        license = 'LICENSE.txt',
        packages = find_packages(),
        scripts = ['mrgaze_batch.py', 'mrgaze_single.py'],
        package_data = {'mrgaze': ['Cascade/*']},
      )
