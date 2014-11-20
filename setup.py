from setuptools import setup, find_packages

setup ( name = 'mrgaze',
        version = '0.1.1',
        description = 'Real-time pupilometry module'
        author = ['Mike Tyszka'],
        author_email = ['jmt@caltech.edu'],
        url = ['https://github.com/jmtyszka/mrgaze'],
        license = 'LICENSE.txt',
        packages = find_packages(),
        scripts = ['mrgaze_batch.py', 'mrgaze_single.py'],
        package_data = {'mrgaze': ['Cascade/*']},
      )
