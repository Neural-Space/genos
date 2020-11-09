#!/usr/bin/env python3
import pathlib
import sys

import setuptools
from setuptools.command.install import install
import os
from genos.version import VERSION


here = pathlib.Path(__file__).parent.resolve()

# Get the long description from the README file
long_description = (here / 'README.md').read_text(encoding='utf-8')

class VerifyVersionCommand(install):
    """Custom command to verify that the git tag matches our version"""
    description = 'verify that the git tag matches our version'

    def run(self):
        tag = os.getenv('CIRCLE_TAG')

        if tag != VERSION:
            info = "Git tag: {0} does not match the version of this app: {1}. " \
                   "Update version in src/genos/version.py".format(
                tag, VERSION
            )
            sys.exit(info)


setuptools.setup(
    name='genos',
    version=VERSION,
    description='Instantiate objects and call functions using dictionary configs in Python using Genos.',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/Neural-Space/genos',
    author='Kushal Jain',
    author_email='kushal@neuralspace.ai',
    keywords='instantiation, objects, recursive instantiation, functio call, config instantiate',
    license='MIT',
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Topic :: Software Development :: Build Tools",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Topic :: Internet",
        "Programming Language :: Python :: 3.7",
    ],
    install_requires=[
        "omegaconf~=2.0.4"
    ],
    python_requires='>=3.7, <4',
    package_dir={'': 'src'},
    packages=setuptools.find_packages(where='src'),
    cmdclass={
        'verify': VerifyVersionCommand,
    }
)
