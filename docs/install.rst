Installation
------------
The package requires Python version 3.7 or higher and works on Linux with GCC 12. 
For GPU functionality, Nvidia GPUs are required.

The easiest way to install is from pip using:

.. code-block:: bash

    pip install cuBNM

In some cases, the package should be installed from source:

* To enable parallel simulations on multiple CPU threads using  OpenMP (this feature is disabled in the PyPi release due to  manylinux constraints)
* If the installation using pip fails or with ``import cuBNM``  you get an ``ImportError`` with the error message reporting an "undefined symbol"


To install from source:

.. code-block:: bash

    git clone https://github.com/amnsbr/cuBNM.git
    cd cuBNM && pip install .

The installation from srouce also requires `GSL 2.7 <https://www.gnu.org/software/gsl/>`_. 
If GSL is not found (in ``"/usr/lib"``, ``"/lib"``, ``"/usr/local/lib"``, ``"~/miniconda/lib"``
, $LIBRARY_PATH, $LD_LIBRARY_PATH``) it will be installed and built by the package in 
``~/.cuBNM/gsl``, but this takes a rather long time (5+ minutes). If you have GSL 2.7 
installed find the location of its libraries ``libgsl.a`` and ``libgslcblas.a`` and 
add the directory to ``$LIBRARY_PATH``.
