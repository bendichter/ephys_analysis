# -*- coding: utf-8 -*-

from setuptools import setup, find_packages

with open('README.md', 'r') as fp:
    readme = fp.read()

setup_args = {
    'name': 'ephys_analysis',
    'version': '0.0.1',
    'description': 'ephys analysis of NWB data',
    'long_description': readme,
    'long_description_content_type': 'text/x-rst; charset=UTF-8',
    'author': 'Ben Dichter',
    'author_email': 'ben.dichter@gmail.com',
    'url': 'https://github.com/ben.dichter/ephys_analysis',
    'license': "BSD",
    'packages': find_packages(),
    'install_requires': ['numpy', 'scipy', 'tqdm', 'matplotlib', 'seaborn','python-snippets'],
    'classifiers': [
        "Programming Language :: Python",
        "Programming Language :: Python :: 2.7",
        "Programming Language :: Python :: 3.5",
        "Programming Language :: Python :: 3.6",
        "License :: OSI Approved :: BSD License",
        "Development Status :: 2 - Pre-Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Operating System :: Microsoft :: Windows",
        "Operating System :: MacOS",
        "Operating System :: Unix",
        "Topic :: Scientific/Engineering :: Medical Science Apps."
    ],
    'keywords': 'Neuroscience '
                'python '
                'HDF '
                'HDF5 '
                'cross-platform '
                'open-data '
                'data-format '
                'open-source '
                'open-science '
                'reproducible-research '
                'PyNWB '
                'NWB '
                'NWB:N '
                'NeurodataWithoutBorders',
}

#if __name__ == '__main__':
setup(**setup_args)
