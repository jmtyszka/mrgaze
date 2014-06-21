from distutils.core import setup
setup ( name = 'geetee',
        version = '0.1',
        packages = ['geetee'],
        scripts=['scripts/geetee.py','scripts/geetee.py'],
        data_files=[('/usr/local/bin',['scripts/geetee.py','scripts/geetee_batch.py'])],
        author=['Mike Tyszka'],
        author_email=['jmt@caltech.edu'],
        url=['https://github.com/jmtyszka/geetee']
      )