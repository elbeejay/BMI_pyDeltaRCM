**************
BMI_pyDeltaRCM
**************

.. image:: https://github.com/DeltaRCM/BMI_pyDeltaRCM/workflows/build/badge.svg
    :target: https://github.com/DeltaRCM/BMI_pyDeltaRCM/actions

.. image:: https://codecov.io/gh/DeltaRCM/BMI_pyDeltaRCM/branch/develop/graph/badge.svg
  :target: https://codecov.io/gh/DeltaRCM/BMI_pyDeltaRCM

Basic Modeling Interface (BMI) wrapper to the pyDeltaRCM model.

**NOTE:** this project is in alpha testing. It is incomplete, unstable, and untested.


Documentation
#############

`Find the full documentation here <https://deltarcm.org/BMI_pyDeltaRCM/index.html>`_.

Installation
############

**While project is in alpha:** You will need to install `pyDeltaRCM` first, visit that repository and follow installation instructions there.

To install this package into an existing Python 3.x environment, download or clone the repository and run:

.. code:: bash

    $ python setup.py install

Or for a developer installation run:

.. code:: bash

    $ pip install -e .


Executing the model
###################

The below code provides the simplest method for initializing and running
the pyDeltaRCM model using the BMI interface.

.. code:: python

    from BMI_pyDeltaRCM.bmidelta import BmiDelta
    delta = BmiDelta()  # instantiate model
    delta.initialize()  # initialize model
    delta.update()  # update model
