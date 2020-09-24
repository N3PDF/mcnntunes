from __future__ import print_function
import sys
from setuptools import setup, find_packages

with open('README.md') as f:
    long_desc = f.read()

# Read requirements.txt
requirements = open('requirements.txt').readlines()
requirements = [r.strip() for r in requirements]

setup(name= "mcnntune",
      version = '0.1',
      description = "MC Neural Network Tunes",
      author = "S. Carrazza & S. Alioli",
      author_email = "stefano.carrazza@cern.ch",
      long_description = long_desc,
      entry_points = {'console_scripts':
                    ['mcnntemplate = mcnntune.scripts.mcnntemplate:main',
                     'mcnntune = mcnntune.scripts.mcnntune:main' ]},
      package_dir = {'': 'src'},
      packages = find_packages('src'),
      package_data = {'':['*.html']
       },
      install_requires=requirements,
      zip_safe = False,
      classifiers=[
            'Operating System :: Unix',
            'Programming Language :: Python',
            'Topic :: Scientific/Engineering',
            'Topic :: Scientific/Engineering :: Physics',
            ],
     )
