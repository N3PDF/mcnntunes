from __future__ import print_function
import sys
from setuptools import setup, find_packages

with open('README.md') as f:
    long_desc = f.read()

setup(name= "mcnn",
      version = '0.1',
      description = "MC Neural Network Tunes",
      author = "S. Carrazza & S. Alioli",
      author_email = "stefano.carrazza@cern.ch",
      long_description = long_desc,
      entry_points = {'console_scripts':
                    ['mcnntemplate = scripts.mcnntemplate:main']},
      package_dir = {'': 'src'},
      packages = find_packages('src'),
      package_data = {
       },
      install_requires=[
          'os', 'argparse', 'yaml', 'itertools', 'jinja2'
      ],
      zip_safe = False,
      classifiers=[
            'Operating System :: Unix',
            'Programming Language :: Python',
            'Topic :: Scientific/Engineering',
            'Topic :: Scientific/Engineering :: Physics',
            ],
     )
