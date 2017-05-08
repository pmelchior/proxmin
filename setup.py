from setuptools import setup
import os

packages = []
for root, dirs, files in os.walk('.'):
    if not root.startswith('./build') and '__init__.py' in files:
        packages.append(root[2:])

print('Packages:', packages)

setup(
  name = 'proxmin',
  packages = packages,
  version = '0.0',
  long_description = 'Implementation of the General Linearization Method of Multipliers to solve minimization problems with multiple constraints and variables',
  author = 'Fred Moolekamp and Peter Melchior',
  author_email = 'fred.moolekamp@gmail.com',
  url = 'https://github.com/fred3m/proxmin',
  #download_url = 'https://github.com/fred3m/popper/tarball/0.1',
  keywords = ['proximal', 'minimization', 'data analysis', 'constraint'],
  include_package_data=True
  #classifiers = [],
)