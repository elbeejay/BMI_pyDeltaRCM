#! /usr/bin/env python
from setuptools import setup, find_packages

from BMI_pyDeltaRCM import utils

setup(name='BMI_pyDeltaRCM',
      version=utils._get_version(),
      author='The DeltaRCM Team',
      license='MIT',
      description="Basic Modeling Interface for pyDeltaRCM",
      long_description=open('README.rst').read(),
      packages=find_packages(exclude=['*.tests']),
      include_package_data=True,
      url='https://github.com/DeltaRCM/BMI_pyDeltaRCM',
      install_requires=['matplotlib', 'netCDF4', 'scipy', 'numpy', 'pyyaml',
                        'numba', 'pyDeltaRCM', 'bmipy']
      )
