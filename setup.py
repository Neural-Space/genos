#!/usr/bin/env python3
import setuptools

from genos.version import VERSION

setuptools.setup(
    version=VERSION,
    install_requires=[],
    python_requires='>=3.7, <4',
    package_dir={'': 'src'},
    packages=setuptools.find_packages(where='src')
)
