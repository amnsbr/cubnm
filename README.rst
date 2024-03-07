.. raw:: html

    <div align="center">
    <img src="https://raw.githubusercontent.com/amnsbr/cuBNM/main/docs/_static/logo_text.png" style="width:300px; margin:auto; padding-bottom:10px;" alt="cuBNM logo">
    </div>


The cuBNM toolbox is designed for efficient biophysical network modeling of the brain on GPUs.

Overview
--------
The toolbox simulates neuronal activity of network nodes based on biophysical network modeling. 
Currently two types of reduced Wong-Wang model based on 
`Deco et al. 2014 <https://doi.org/10.1523/JNEUROSCI.5068-13.2014>`_ and
`Deco et al. 2013 <https://doi.org/10.1523/JNEUROSCI.1091-13.2013>`_ are implemented, 
but the modular design of the code makes it possible to  include additional models in the future. 
The simulated activity of model neurons is fed into the Balloon-Windkessel model to calculate 
simulated BOLD signal. GPU code is equipped with functions to calculate the functional connectivity
(FC) and functional connectivity dynamics (FCD) from the simulated BOLD signal, which can be 
compared to FC and FCD matrices derived from empirical BOLD signals to assess similarity 
(goodness-of-fit) of the simulated to empirical BOLD signal. The models
may include global or regional free parameters which are fit to the empirical data. 
Regional parameters can be set to be homogeneous or can vary across nodes based on a 
parameterized combination of fixed maps or using independent free parameters for each node 
/ group of nodes.

Parameter optimization of the model can be performed via various optimization algorithms, including 
grid (brute-force) search and evolutionary optimizers provided by the ``pymoo`` package. 
Indepdendent simulations (i.e., within the grid or each iteration of the evolutionary optimizers)
are parallelized across the GPU "blocks" (one block per simualtion) and "threads" (one thread
per node).

The toolbox also supports running the simulations on single- or multi-core CPUs, which will 
be used if no GPUs are detected or when requested by the user. However, the primary focus 
of the toolbox is on GPU usage.

Below is a simplified flowchart of the different components of the program, which is 
written in Python, C/C++, and CUDA:

.. image:: https://raw.githubusercontent.com/amnsbr/cuBNM/main/docs/_static/flowchart_extended.png
    :alt: Flowchart of the cuBNM program

.. overview-end

Documentation
-------------
Please find the documentation at https://cubnm.readthedocs.io.

.. include:: ./docs/install.rst
.. include:: ./docs/quickstart.rst

.. install-start
Installation
------------
The package requires Python version 3.7 or higher and works on Linux with GCC 12. 
For GPU functionality, Nvidia GPUs and a working installation of CUDA Toolkit are required.

Installation from source is currently the recommended way to install the package. 
Other than CUDA Toolkit, installation from srouce requires `GSL 2.7 <https://www.gnu.org/software/gsl/>`_. 
If GSL is not found (in ``"/usr/lib"``, ``"/lib"``, ``"/usr/local/lib"``, ``"~/miniconda/lib"``
, ``$LIBRARY_PATH``, ``$LD_LIBRARY_PATH``) it will be installed and built by the package in 
``~/.cuBNM/gsl``, but this takes a rather long time (5+ minutes). If you have GSL 2.7 
installed find the location of its libraries ``libgsl.a`` and ``libgslcblas.a`` and 
add the directory to ``$LIBRARY_PATH``. When dependencies are installed, the
package can be installed from source using:

.. code-block:: bash

    pip install git+https://github.com/amnsbr/cuBNM.git

Alternatively, pre-built wheels of the initial version are available on PyPi and 
can be installed using:

.. code-block:: bash

    pip install cuBNM

.. install-end

Quickstart
-------------
Evolutionary optimization
~~~~~~~~~~~~~~~~~~~~~~~~~
Run a CMAES optimization for 10 iterations with a population size of 20:

.. code-block:: python

    from cuBNM import datasets, optimize

    problem = optimize.BNMProblem(
        model = 'rWW',
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
        mode = 'rWW',
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