.. raw:: html

    <div align="center">
    <img src="https://raw.githubusercontent.com/amnsbr/cuBNM/main/docs/_static/logo_text.png" style="width:300px; margin:auto; padding-bottom:10px;" alt="cuBNM logo">
    </div>


The cuBNM toolbox is designed for efficient simulation and optimization of biophysical network models (BNM) of the brain on GPUs.

Overview
--------
The toolbox supports simulation of network nodes activity based on the reduced Wong-Wang model, 
with analytical-numerical feedback inhibition control. The Balloon-Windkessel model is utilized 
for the calculation of simulated BOLD signals. The toolbox calculates the goodness of fit of 
the simulated BOLD to the empirical BOLD based on functional connectivity (FC) and functional 
connectivity dynamics (FCD) matrices. The model local parameters (connectivity weight of 
excitatory and inhibitory neurons) can be homogeneous, or can vary across nodes based on a 
parameterized combination of fixed maps or using independent free parameters for each node 
/ group of nodes.

Parameter optimization of the model can be performed using grid search or evolutionary optimizers. 
Parallelization of the entire grid or each iteration of evolutionary optimizers is done at 
two levels:

#. Simulations (across the GPU "blocks")
#. Nodes (across each block’s "threads")

The toolbox also supports running the simulations on single- or multi-core CPUs, which will 
be used if no GPUs are detected or when requested by the user. However, the primary focus 
of the toolbox is on GPU usage.

Below is a simplified flowchart of the different components of the program, which is 
written in Python, C++, and CUDA:

.. image:: https://raw.githubusercontent.com/amnsbr/cuBNM/main/docs/_static/flowchart_extended.png
    :alt: Flowchart of the cuBNM program

.. overview-end

Documentation
-------------
Please find the documentation at https://cubnm.readthedocs.io.

.. include:: ./docs/install.rst
.. include:: ./docs/quickstart.rst

Installation
------------
The package requires Python version 3.7 or higher and works on Linux with GCC 12. 
For GPU functionality, Nvidia GPUs are required.

The easiest way to install is from pip using:

.. code-block:: bash

    pip install cuBNM

In some cases, the package should be installed from source:

* To enable parallel simulations on multiple CPU threads using OpenMP (this feature is disabled in the PyPi release due to  manylinux constraints)
* If the installation using pip fails or with ``import cuBNM``  you get an ``ImportError`` with the error message reporting an  "undefined symbol"


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

Quickstart
-------------
Evolutionary optimization
~~~~~~~~~~~~~~~~~~~~~~~~~
Run a CMAES optimization for 10 iterations with a population size of 20:

.. code-block:: python

    from cuBNM import datasets, optimize

    problem = optimize.RWWProblem(
        params = {
            'G': (0.5, 2.5),
            'wEE': (0.05, 0.75),
            'wEI': 0.15,
        },
        emp_fc_tril = datasets.load_functional('FC', 'schaefer-100', exc_interhemispheric=True),
        emp_fcd_tril = datasets.load_functional('FCD', 'schaefer-100', exc_interhemispheric=True),
        duration = 60,
        TR = 1,
        sc_path = datasets.load_sc('strength', 'schaefer-100', return_path=True),
    )
    cmaes = optimize.CMAESOptimizer(popsize=20, n_iter=10, seed=1)
    cmaes.setup_problem(problem)
    cmaes.optimize()

Grid search
~~~~~~~~~~~
Run a 10x10 grid search of G, wEE with fixed wEI:

.. code-block:: python

    from cuBNM import datasets, optimize

    gs = optimize.GridSearch(
        params = {
            'G': (0.5, 2.5, 10),
            'wEE': (0.05, 0.75, 10),
            'wEI': 0.21
        },
        duration = 60,
        TR = 1,
        sc_path = datasets.load_sc('strength', 'schaefer-100', return_path=True)
    )
    emp_fc_tril = datasets.load_functional('FC', 'schaefer-100', exc_interhemispheric=True)
    emp_fcd_tril = datasets.load_functional('FCD', 'schaefer-100', exc_interhemispheric=True)
    scores = gs.evaluate(emp_fc_tril, emp_fcd_tril)