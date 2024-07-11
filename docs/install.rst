Installation
------------
Using pip
~~~~~~~~~~~~

Requirements:

* Linux operating system on x86-64 architecture (for other architectures build from source)
* Python >= 3.7

Additional requirements for GPU functionality:

* NVIDIA GPU: In theory GPU devices with Compute Capability >= 2.x should be supported, but the code is not tested using devices with Compute Capability < 6.0.
* NVIDIA Driver >= 450.80.02

Install via:

.. code-block:: bash

    pip install cubnm
    

From source
~~~~~~~~~~~~~~~~~

This might be needed in case prebuilt wheels are not available for your machine (e.g. on arm64 machines).

Requirements:

* Linux operating system
* Python >= 3.7
* GCC: Pre-built wheels were compiled using version 10.2.1 but alternative versions can be used.
* `GSL 2.7 <https://www.gnu.org/software/gsl/>`_: If GSL is not found (in ``"/usr/lib"``, ``"/lib"``, ``"/usr/local/lib"``, ``"~/miniconda/lib"``, ``$LIBRARY_PATH``, ``$LD_LIBRARY_PATH``) it will be installed and built by the package in ``~/.cubnm/gsl``, but this takes a rather long time (5+ minutes). If you have GSL 2.7 installed find the location of its libraries ``libgsl.a`` and ``libgslcblas.a`` and add the directory to ``$LIBRARY_PATH``. In this case note that GSL must have been built with ``--enable-shared`` option.

Additional requirements for GPU functionality:

* NVIDIA GPU: In theory GPU devices with Compute Capability >= 2.x should be supported, but the code is not tested using devices with Compute Capability < 6.0.
* CUDA Toolkit: Pre-built wheels were compiled using version 11.8 but alternative versions can be used.

The package can be installed from source using:

.. code-block:: bash

    pip install git+https://github.com/amnsbr/cubnm.git -vvv

.. install-end