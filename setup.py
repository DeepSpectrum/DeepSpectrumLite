#!/usr/bin/env python
import re
import sys
import warnings

warnings.filterwarnings('ignore', category=DeprecationWarning)
warnings.filterwarnings('ignore', category=FutureWarning)

from setuptools import setup, find_packages
from subprocess import CalledProcessError, check_output

PROJECT = "DeepSpectrumLite"
VERSION = "1.0.1"
LICENSE = "GPLv3+"
AUTHOR = "Tobias Hübner"
AUTHOR_EMAIL = "tobias.huebner@informatik.uni-augsburg.de"
URL = 'https://github.com/DeepSpectrum/DeepSpectrumLite'

with open("DESCRIPTION.md", "r") as fh:
    LONG_DESCRIPTION = fh.read()

install_requires = [
    "librosa",
    "numba",
    "pillow",
    "pandas",
    "scikit-learn",
    "click",
    "tensorflow==2.4.1",
    "tensorboard==2.4.1",
    "keras-applications"
]

tests_require = ['pytest>=4.4.1', 'pytest-cov>=2.7.1']
needs_pytest = {'pytest', 'test', 'ptr'}.intersection(sys.argv)
setup_requires = ['pytest-runner'] if needs_pytest else []
packages = find_packages('src')

setup(
    name=PROJECT,
    version=VERSION,
    license=LICENSE,
    author=AUTHOR,
    author_email=AUTHOR_EMAIL,
    long_description=LONG_DESCRIPTION,
    long_description_content_type="text/markdown",
    descrption="DeepSpectrumLite is a Python toolkit for training light-weight CNN networks targeted at embedded devices.",
    platforms=["Any"],
    scripts=[],
    provides=[],
    python_requires="~=3.8.0",
    install_requires=install_requires,
    setup_requires=setup_requires,
    tests_require=tests_require,
    namespace_packages=[],
    packages=packages,
    package_dir={'': 'src'},
    include_package_data=True,
    entry_points={
        "console_scripts": [
            "deepspectrumlite = deepspectrumlite.__main__:cli",
        ]
    },
    classifiers=[
        # How mature is this project? Common values are
        #   3 - Alpha
        #   4 - Beta
        #   5 - Production/Stable
        'Development Status :: 4 - Beta',

        'Environment :: GPU :: NVIDIA CUDA :: 11.0',
        # Indicate who your project is intended for
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Intended Audience :: Science/Research',

        # Pick your license as you wish (should match "license" above)
        'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',

        'Programming Language :: Python :: 3.8',
    ],
    keywords='machine-learning audio-analysis science research',
    project_urls={
        'Source': 'https://github.com/DeepSpectrum/DeepSpectrumLite',
        'Tracker': 'https://github.com/DeepSpectrum/DeepSpectrumLite/issues',
    },
    url=URL,
    zip_safe=False,
)