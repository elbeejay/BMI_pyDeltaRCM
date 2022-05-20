**************
BMI_pyDeltaRCM
**************

.. image:: https://github.com/DeltaRCM/BMI_pyDeltaRCM/workflows/build/badge.svg
    :target: https://github.com/DeltaRCM/BMI_pyDeltaRCM/actions

.. image:: https://codecov.io/gh/DeltaRCM/BMI_pyDeltaRCM/branch/develop/graph/badge.svg
  :target: https://codecov.io/gh/DeltaRCM/BMI_pyDeltaRCM

Basic Modeling Interface (BMI) wrapper to the pyDeltaRCM model.


Documentation
#############

`Find the full documentation here <https://deltarcm.org/BMI_pyDeltaRCM/index.html>`_.

Installation
############

**Note:**

    If you intend to manipulate the underlying *pyDeltaRCM* code in any way, be sure to follow the `Developer Installation instructions <https://deltarcm.org/pyDeltaRCM/meta/installing.html#developer-installation>`_ from that project before installing the BMI wrapper.

To install this package into an existing Python 3.x environment, download or clone the repository and run:

.. code:: bash

    $ pip install -r requirements.txt
    $ python setup.py install

Or for a developer installation run:

.. code:: bash

    $ pip install -r requirements.txt
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

See the `BMI_pyDeltaRCM User Guide <https://deltarcm.org/BMI_pyDeltaRCM/userguide.html>`_.
