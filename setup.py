#!/usr/bin/env python3
import setuptools


setuptools.setup(
    version="0.1.0",
    install_requires=[],
    python_requires='>=3.7, <4',
    package_dir={'': 'src'},
    packages=setuptools.find_packages(where='src')
)
