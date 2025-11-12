Installation
------------

Quick Start
~~~~~~~~~~~

For most users with a standard Linux system and NVIDIA GPU:

.. code-block:: bash

    pip install cubnm

We recommend using a virtual environment (``venv`` or ``conda``).

Requirements
~~~~~~~~~~~~

**Required:**

* Linux operating system on x86-64 architecture (for other architectures see :ref:`from-source`)
* Python >= 3.7

**Optional (for GPU functionality):**

* NVIDIA GPU with Compute Capability >= 6.0 (older devices >= 2.x may work but are untested)
* NVIDIA Driver >= 520.61.05 (required for CUDA 11.8 support)

To check your CUDA version:

.. code-block:: bash

    nvcc --version  # If CUDA Toolkit is installed
    nvidia-smi      # Shows compatible CUDA version

Installation Options
~~~~~~~~~~~~~~~~~~~

Using pip (Recommended)
^^^^^^^^^^^^^^^^^^^^^^^

**Basic installation:**

.. code-block:: bash

    pip install cubnm

This provides GPU-accelerated simulations and FC/FCD calculations.

**With optional GPU-accelerated goodness-of-fit calculations:**

For GPU-accelerated computation of empirical-to-simulated FC correlation and FCD 
Kolmogorov-Smirnov distance (especially useful for very large parameter grids), 
install with CUDA-specific extras:

.. code-block:: bash

    pip install cubnm[cu11]  # For CUDA 11.x
    pip install cubnm[cu12]  # For CUDA 12.x
    pip install cubnm[cu13]  # For CUDA 13.x

This installs additional dependencies: ``cupy``, ``numba-cuda``, and ``nvidia-cublas``.

.. note::
    If you don't know your CUDA version, use ``nvcc --version`` or ``nvidia-smi`` 
    to check before installing.

.. _from-source:

From source
^^^^^^^^^^^

Building from source may be necessary for:

* Non-x86-64 architectures (e.g., ARM64)
* Custom CUDA versions
* Development purposes

**Requirements:**

* Linux operating system
* Python >= 3.7
* GCC (pre-built wheels use version 10.2.1, but other versions should work)
* `GSL 2.7 <https://www.gnu.org/software/gsl/>`_ (see installation options below)

**For GPU functionality:**

* NVIDIA GPU with Compute Capability >= 6.0
* CUDA Toolkit (pre-built wheels use version 11.8, but other versions should work)

.. warning::
    **AMD GPU/APU Support (Experimental):** AMD devices are experimentally supported 
    when building from source. ROCm toolkit and ``hipcc`` must be available in ``$PATH``. 
    AMD support is not regularly maintained or tested, and compatibility is not guaranteed.

**GSL 2.7 Installation:**

GSL is required for building from source. The package will search for GSL in standard 
locations (``/usr/lib``, ``/lib``, ``/usr/local/lib``, ``~/miniconda/lib``, 
``$LIBRARY_PATH``, ``$LD_LIBRARY_PATH``).

**Option 1 (Recommended):** Install via conda:

.. code-block:: bash

    conda install -y --no-deps -c conda-forge gsl=2.7
    export LIBRARY_PATH=$CONDA_PREFIX/lib:$LIBRARY_PATH

**Option 2:** If GSL is not found, the package will automatically download and build 
it in ``~/.cubnm/gsl``. This takes 5+ minutes.

**Option 3:** If you have GSL 2.7 installed elsewhere, locate ``libgsl.a`` and 
``libgslcblas.a`` and add their directory to ``$LIBRARY_PATH``. 

.. note::
    GSL must have been built with the ``--enable-shared`` option.

**Install from source:**

.. code-block:: bash

    pip install git+https://github.com/amnsbr/cubnm.git -vvv

Using Docker or Singularity/Apptainer
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. note::
    Container support is experimental. We recommend installing via pip for most use cases.

Docker images are available for stable versions (except v0.0.1).

.. image:: https://img.shields.io/badge/docker-amnsbr/cubnm-blue.svg?logo=docker
  :target: https://hub.docker.com/r/amnsbr/cubnm

**Pull the container:**

.. code-block:: bash

    # Docker
    docker pull amnsbr/cubnm:v<version>
    
    # Singularity/Apptainer
    singularity pull /path/to/cubnm-<version>.sif docker://amnsbr/cubnm:v<version>

**Usage modes:**

*Interactive mode:*

.. code-block:: bash

    # Docker
    docker run -it --entrypoint /bin/bash amnsbr/cubnm:v<version>
    
    # Singularity
    singularity shell /path/to/cubnm-<version>.sif

In interactive mode, ``cubnm`` is installed and can be imported in ``python3.10``.

*Command line interface:*

.. code-block:: bash

    # Docker
    docker run amnsbr/cubnm:v<version>
    
    # Singularity
    singularity run /path/to/cubnm-<version>.sif

See :doc:`command line interface </cli>` for details.

**Mounting directories:**

Remember to bind your input and output directories:

.. code-block:: bash

    # Docker
    docker run -v /host/path:/container/path amnsbr/cubnm:v<version>
    
    # Singularity
    singularity run -B /host/path:/container/path /path/to/cubnm-<version>.sif

**Using GPUs in containers:**

For Docker, add ``--gpus all`` before the container name:

.. code-block:: bash

    docker run --gpus all amnsbr/cubnm:v<version>

For Singularity, add ``--nv`` before the image path:

.. code-block:: bash

    singularity run --nv /path/to/cubnm-<version>.sif

**Prerequisites for GPU support in containers:**

* Docker: `NVIDIA Container Toolkit <https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html>`_ 
  must be installed. See `Docker GPU documentation <https://docs.docker.com/config/containers/resource_constraints/#gpu>`_.
* Singularity: NVIDIA drivers must be available on the host. See `Singularity GPU documentation <https://docs.sylabs.io/guides/3.5/user-guide/gpu.html>`_.

Verification
~~~~~~~~~~~~

After installation, verify that cuBNM is working:

.. code-block:: python

    import cubnm
    print(cubnm.__version__)
    
    # Check if GPU is available
    from cubnm.utils import avail_gpus
    print(f"Available GPUs: {avail_gpus()}")

You can also run a quick test simulation to ensure everything is working properly:

.. code-block:: python

    from cubnm import datasets, sim
    import numpy as np
    
    sc = datasets.load_sc('strength', 'schaefer-100', 'group-train706')
    emp_bold = datasets.load_bold('schaefer-100')
    
    sim_group = sim.rWWSimGroup(
        duration=60,
        TR=1,
        window_size=10,
        window_step=2,
        sc=sc,
        sim_verbose=True,
    )
    sim_group.N = 1
    sim_group.param_lists['G'] = np.repeat(0.5, sim_group.N)
    sim_group._set_default_params(missing=True)
    sim_group.run()
    sim_group.score(emp_bold=emp_bold)

.. install-end