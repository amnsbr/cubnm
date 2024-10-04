Installation
------------
Using pip
~~~~~~~~~~~~

Requirements:

* Linux operating system on x86-64 architecture (for other architectures build from source)
* Python >= 3.7

Additional requirements for GPU functionality:

* NVIDIA GPU: In theory GPU devices with Compute Capability >= 2.x should be supported, but the code is not tested using devices with Compute Capability < 6.0.
* NVIDIA Driver >= 520.61.05

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
* `GSL 2.7 <https://www.gnu.org/software/gsl/>`_: If GSL is not found (in ``"/usr/lib"``, ``"/lib"``, ``"/usr/local/lib"``, ``"~/miniconda/lib"``, ``$LIBRARY_PATH``, ``$LD_LIBRARY_PATH``) it will be installed and built by the package in ``~/.cubnm/gsl``, but this takes a rather long time (5+ minutes). A faster and recommended alternative would be to use a conda environment and do ``conda install -y --no-deps -c conda-forge gsl=2.7``, then add conda environment's ``lib`` directory to ``$LIBRARY_PATH``, and then install the package from source. If you have GSL 2.7 installed elsewhere find the location of its libraries ``libgsl.a`` and ``libgslcblas.a`` and add the directory to ``$LIBRARY_PATH``. In this case note that GSL must have been built with ``--enable-shared`` option.

Additional requirements for GPU functionality:

* NVIDIA GPU: In theory GPU devices with Compute Capability >= 2.x should be supported, but the code is not tested using devices with Compute Capability < 6.0.
* CUDA Toolkit: Pre-built wheels were compiled using version 11.8 but alternative versions can be used.

The package can be installed from source using:

.. code-block:: bash

    pip install git+https://github.com/amnsbr/cubnm.git -vvv

Using Docker or Singularity/Apptainer
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Docker images are available for the development and stable versions (except v0.0.1).

.. image:: https://img.shields.io/badge/docker-amnsbr/cubnm-blue.svg?logo=docker
  :target: https://hub.docker.com/r/amnsbr/cubnm

* Stable (``amnsbr/cubnm:v*``): These are more lightweight and smaller in size. The output simulations given the same input data and random seed should be reproducible across platforms.  
* Development (``amnsbr/cubnm:dev``): This includes latest changes of the code but is not updated after each commit. The output of identical simulations with the same random seed may be different across platforms, and therefore, tests of expected simulations may fail.

Pull the container via ``docker pull amnsbr/cubnm:<version>`` or ``singularity pull /path/to/cubnm-<version>.sif docker://amnsbr/cubnm:<version>``. 

The containers can be used in two modes:

* Interactively: ``docker run -it --entrypoint /bin/bash amnsbr/cubnm:<version>`` or ``singularity shell /path/to/cubnm-<version>.sif``. ``cubnm`` is installed and can be imported in ``python3.10`` (stable versions) or ``/opt/miniconda/bin/python`` (development version).
* Using :doc:`command line interface </cli>`: ``docker run amnsbr/cubnm:*`` or ``singularity run /path/to/cubnm-<version>.sif``. Command line interface is not available in ``v0.0.2``.

Remember to bind your input and output directories to the container via ``-v`` in Docker and ``-B`` in Singularity.
    
To use GPUs add the flag ``--gpus all`` to Docker (before container name) and ``--nv`` to Singularity (before path to image). For more details on prerequisites for using GPUs inside the containers see `Docker <https://docs.docker.com/config/containers/resource_constraints/#gpu>`_ and `Singularity <https://docs.sylabs.io/guides/3.5/user-guide/gpu.html>`_ documentations.

.. install-end