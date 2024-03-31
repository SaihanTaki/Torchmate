#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Note: To use the 'upload' functionality of this file, you must:
#   $ pip install twine

import io
import os
import shutil
import sys

from setuptools import Command, find_packages, setup

# Package meta-data.
NAME = "torchmate"
DESCRIPTION = "Torchmate: A High level PyTorch Training Library"
AUTHOR = "Abdullah Saihan Taki"
EMAIL = "saihan0176@gmail.com"
URL = "https://github.com/SaihanTaki/torchmate"
URL_DOC = "https://torchmate.readthedocs.io/en/latest/"
REQUIRES_PYTHON = ">=3.9.0"
VERSION = None


CLASSIFIERS = [
    # Full list: https://pypi.python.org/pypi?%3Aaction=list_classifiers
    "Development Status :: 3 - Alpha",
    "Intended Audience :: Developers",
    "License :: OSI Approved :: MIT License",
    "Operating System :: OS Independent",
    "Programming Language :: Python :: 3",
    "Programming Language :: Python :: 3.9",
    "Programming Language :: Python :: 3.10",
    "Topic :: Scientific/Engineering :: Artificial Intelligence",
]

REQUIRES_STYLE = [
    "pre-commit>=3.6.2",
    "black>=24.1.1",
    "flake8>=7.0.0",
    "isort>=5.13.2",
]

REQUIRES_TESTS = [
    "pytest>=8.0.1",
    "pytest-mock>=3.12.0",
    "coverage>=7.4.1",
    "pytest-cov>=5.0.0",
    "six>=1.16.0",
]

REQUIRES_DOCS = [
    "Sphinx==7.2.6",
    "sphinx-copybutton==0.5.2",
    "sphinx-notfound-page==1.0.0",
    "sphinx-rtd-theme==2.0.0",
    "sphinxcontrib-applehelp==1.0.8",
    "sphinxcontrib-devhelp==1.0.6",
    "sphinxcontrib-htmlhelp==2.0.5",
    "sphinxcontrib-jquery==4.1",
    "sphinxcontrib-jsmath==1.0.1",
    "sphinxcontrib-qthelp==1.0.7",
    "sphinxcontrib-serializinghtml==1.1.10",
    "autodocsumm==0.2.12",
    "nbsphinx==0.9.3",
]

here = os.path.abspath(os.path.dirname(__file__))

# What packages are required for this module to be executed?
try:
    with open(os.path.join(here, "requirements.txt"), encoding="utf-8") as f:
        REQUIRED = f.read().split("\n")
except:
    REQUIRED = []

# Optional Packages
EXTRAS = dict(
    style=REQUIRES_STYLE,
    tests=REQUIRES_TESTS,
    docs=REQUIRES_DOCS,
    dev=REQUIRES_TESTS + REQUIRES_STYLE,
    all=REQUIRES_TESTS + REQUIRES_STYLE + REQUIRES_DOCS,
)

# Import the README and use it as the long-description.
try:
    with io.open(os.path.join(here, "README.md"), encoding="utf-8") as f:
        LONG_DESCRIPTION = "\n" + f.read()
except FileNotFoundError:
    LONG_DESCRIPTION = DESCRIPTION

# Load the package's __version__.py module as a dictionary.
about = {}
if not VERSION:
    with open(os.path.join(here, NAME, "__version__.py")) as f:
        exec(f.read(), about)
else:
    about["__version__"] = VERSION


# https://github.com/qubvel/segmentation_models.pytorch/blob/master/setup.py
class UploadCommand(Command):
    """Support setup.py upload."""

    description = "Build and publish the package."
    user_options = []

    @staticmethod
    def status(s):
        """Prints things in color."""
        BG = "\033[46m"
        FG = "\033[30m"
        RESET = "\033[0m"
        print(FG + BG + s + RESET)

    def initialize_options(self):
        pass

    def finalize_options(self):
        pass

    def run(self):
        try:
            self.status("Removing previous builds...")
            print("\n")
            shutil.rmtree(os.path.join(here, "dist"))
        except OSError:
            pass

        self.status("Building Source and Wheel (universal) distribution...")
        os.system(f"{sys.executable} setup.py sdist bdist_wheel --universal")

        # updated before uploading to main pypi
        self.status("Uploading the package to PyPI via Twine...")
        os.system(f"twine upload dist/*")

        # self.status("Pushing git tags...")
        # os.system("git tag v{0}".format(about["__version__"]))
        # os.system("git push --tags")

        sys.exit()


# Where the magic happens:
setup(
    name=NAME,
    version=about["__version__"],
    description=DESCRIPTION,
    long_description=LONG_DESCRIPTION,
    long_description_content_type="text/markdown",
    author=AUTHOR,
    author_email=EMAIL,
    python_requires=REQUIRES_PYTHON,
    packages=find_packages(exclude=("tests", "docs", "images")),
    install_requires=REQUIRED,
    extras_require=EXTRAS,
    include_package_data=True,
    license="MIT",
    classifiers=CLASSIFIERS,
    project_urls=dict(
        Homepage=URL,
        Documentation=URL_DOC,
    ),
    cmdclass={  # $ setup.py publish support.
        "upload": UploadCommand,
    },
)
