from setuptools import setup
import os

packages = []
for root, dirs, files in os.walk('.'):
    if not root.startswith('./build') and '__init__.py' in files:
        packages.append(root[2:])

long_description = open('README.md').read()

setup(
    name = 'proxmin',
    description = 'Proximal methods for constrained optimization',
    long_description = long_description,
    long_description_content_type='text/markdown',
    packages = packages,
    include_package_data=False,
    version = '0.5.4',
    license='MIT',
    author = 'Peter Melchior, Fred Moolekamp',
    author_email = 'peter.m.melchior@gmail.com',
    url = 'https://github.com/pmelchior/proxmin',
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "License :: OSI Approved :: MIT License",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Operating System :: OS Independent",
        "Programming Language :: Python",
        "Topic :: Scientific/Engineering :: Information Analysis"
    ],
    keywords = ['optimization', 'constrained optimization', 'proximal algorithms', 'data analysis', 'non-negative matrix factorization'],
    requires=['numpy','scipy']
)
