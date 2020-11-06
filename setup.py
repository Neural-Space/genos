#!/usr/bin/env python3
import pathlib
import setuptools

from genos.version import VERSION


here = pathlib.Path(__file__).parent.resolve()

# Get the long description from the README file
long_description = (here / 'README.md').read_text(encoding='utf-8')

setuptools.setup(
    name='genos',
    version=VERSION,
    description='Instantiate objects and call functions using dictionary configs in Python using Genos.',
    long_description=long_description,
    url='https://github.com/Neural-Space/genos',
    author='Kushal Jain',
    author_email='kushal@neuralspace.ai',
    keywords='instantiation, objects, recursive instantiation, functio call, config instantiate',
    install_requires=[
        "omegaconf~=2.0.4"
    ],
    python_requires='>=3.7, <4',
    package_dir={'': 'src'},
    packages=setuptools.find_packages(where='src')
)
