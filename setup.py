#!/usr/bin/env python3
import setuptools

from genos.version import VERSION

setuptools.setup(
    name='genos',
    version=VERSION,
    description='Instantiate objects and call functions using dictionary configs in Python using Genos.',
    url='https://github.com/Neural-Space/genos',
    author='Kushal Jain',
    author_email='kushal@neuralspace.ai',
    keywords='instantiation, objects, recursive instantiation',
    # install_requires=[],
    python_requires='>=3.7, <4',
    package_dir={'': 'src'},
    packages=setuptools.find_packages(where='src')
)
