from setuptools import setup, find_packages
import os

long_description = open('README.md').read()

setup(
    name = 'proxmin',
    description = 'Proximal methods for constrained optimization',
    long_description = long_description,
    long_description_content_type='text/markdown',
    packages=find_packages(),
    include_package_data=False,
    use_scm_version=True,
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
    setup_requires=['setuptools_scm', 'setuptools_scm_git_archive'],
    install_requires=['numpy','scipy'],
)
