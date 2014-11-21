from setuptools import setup, find_packages

setup ( name = 'mrgaze',
        version = '0.2',
        description = 'Real-time pupilometry module',
        author = ['Mike Tyszka'],
        author_email = ['jmt@caltech.edu'],
        url = ['https://github.com/jmtyszka/mrgaze'],
        license = 'LICENSE.txt',
        packages = find_packages(),
        package_data = {'mrgaze': ['Cascade/*']},
      )
