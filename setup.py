import os
import re
from setuptools import setup, find_packages

PACKAGE = "mcnntunes"

# Returns the mcnntunes version
def get_version():
    """ Gets the version from the package's __init__ file
    if there is some problem, let it happily fail """
    VERSIONFILE = os.path.join("src", PACKAGE, "__init__.py")
    initfile_lines = open(VERSIONFILE, "rt").readlines()
    VSRE = r"^__version__ = ['\"]([^'\"]*)['\"]"
    for line in initfile_lines:
        mo = re.search(VSRE, line, re.M)
        if mo:
            return mo.group(1)

# Read requirements.txt
requirements = open('requirements.txt').readlines()
requirements = [r.strip() for r in requirements]

# load long description from README
this_directory = os.path.abspath(os.path.dirname(__file__))
with open(os.path.join(this_directory, "README.md"), encoding="utf-8") as f:
    long_description = f.read()

setup(
      name="mcnntunes",
      version=get_version(),
      description = "MC Neural Network Tunes",
      author = "M. Lazzarin, S. Alioli, S. Carrazza",
      author_email = "",
      url="https://github.com/N3PDF/mcnntunes",
      entry_points = {'console_scripts':
                    ['mcnntemplate = mcnntunes.scripts.mcnntemplate:main',
                     'mcnntunes = mcnntunes.scripts.mcnntunes:main',
                     'mcnntunes-buildruns = mcnntunes.scripts.mcnntunes_buildruns:main']},
      package_dir = {'': 'src'},
      packages = find_packages('src'),
      package_data = {'':['*.html']
       },
      zip_safe = False,
      classifiers=[
            'Topic :: Scientific/Engineering :: Physics',
            ],
      install_requires=requirements,
      extra_require={
            "docs": ["sphinx", "sphinx_rtd_theme", "recommonmark", "sphinxcontrib-bibtex", "sphinx_markdown_tables"]
      },
      python_requires=">=3.6.0",
      long_description=long_description,
      long_description_content_type='text/markdown',
)
