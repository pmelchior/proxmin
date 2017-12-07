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
  version = '0.4.3',
  long_description = 'Proximal methods for constrained optimization, including NMF',
  author = 'Peter Melchior and Fred Moolekamp',
  author_email = 'peter.m.melchior@gmail.com',
  url = 'https://github.com/pmelchior/proxmin',
  keywords = ['optimization', 'constrained optimization', 'proximal algorithms', 'data analysis', 'non-negative matrix factorization'],
  include_package_data=False
)
