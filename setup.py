from setuptools import setup, find_packages

setup ( name = 'mrgaze',
<<<<<<< HEAD
        version = '0.5.3',
        description = 'Offline video pupilometry and gaze tracking',
=======
        version = '0.2',
        description = 'Real-time pupilometry module',
>>>>>>> real-time
        author = ['Mike Tyszka'],
        author_email = ['jmt@caltech.edu'],
        url = ['https://github.com/jmtyszka/mrgaze'],
        license = 'LICENSE.txt',
        packages = find_packages(),
        package_data = {'mrgaze': ['Cascade/*']},
      )
