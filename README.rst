.. raw:: html

    <div align="center">
    <img src="https://raw.githubusercontent.com/amnsbr/cubnm/main/docs/_static/logo_text.png" style="width:300px; margin:auto; padding-bottom:10px;" alt="cuBNM logo">
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

.. image:: https://raw.githubusercontent.com/amnsbr/cubnm/main/docs/_static/flowchart_extended.png
    :alt: Flowchart of the cuBNM program

.. overview-end

Documentation
-------------
Please find the documentation at https://cubnm.readthedocs.io.

.. install-start

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

.. quickstart-start

Quickstart
-------------
Evolutionary optimization
~~~~~~~~~~~~~~~~~~~~~~~~~
Run a CMAES optimization of reduced Wong Wang model with G and wEE as free parameters:

.. image:: https://kaggle.com/static/images/open-in-kaggle.svg 
   :target: https://www.kaggle.com/code/aminsaberi/cubnm-0-0-2-demo-cmaes-homogeneous

.. code-block:: python

    from cubnm import datasets, optimize

    problem = optimize.BNMProblem(
        model = 'rWW',
        params = {
            'G': (0.5, 2.5),
            'wEE': (0.05, 0.75),
            'wEI': 0.15,
        },
        emp_fc_tril = datasets.load_functional('FC', 'schaefer-100'),
        emp_fcd_tril = datasets.load_functional('FCD', 'schaefer-100'),
        duration = 60,
        TR = 1,
        sc_path = datasets.load_sc('strength', 'schaefer-100'),
        states_ts = True,
        out_dir = './cmaes_homo',
    )
    cmaes = optimize.CMAESOptimizer(popsize=20, n_iter=10, seed=1)
    cmaes.setup_problem(problem)
    cmaes.optimize()
    cmaes.save()

Using command line interface:

.. code-block:: bash

    cubnm \
        --model rWW --sc example --emp_fc_tril example --emp_fcd_tril example \
        --out_dir ./cmaes_homo_cli \
        --TR 1 --duration 60 --states_ts \
        --params G=0.5:2.5,wEE=0.05:0.75,wEI=0.15 \
        optimize \
        --optimizer CMAES --optimizer_seed 0 --n_iter 10 --popsize 20

Run a CMAES optimization of reduced Wong Wang model with G as a global free parameter and wEE and wEI as
regional free parameters that are regionally heterogeneous based on a weighted combination of two fixed
maps (HCP T1w/T2w, HCP FC G1):

.. image:: https://kaggle.com/static/images/open-in-kaggle.svg
   :target: https://www.kaggle.com/code/aminsaberi/cubnm-0-0-2-demo-cmaes-heterogeneous

.. code-block:: python

    from cubnm import datasets, optimize

    problem = optimize.BNMProblem(
        model = 'rWW',
        params = {
            'G': (0.5, 2.5),
            'wEE': (0.05, 0.75),
            'wEI': (0.05, 0.75),
        },
        het_params = ['wEE', 'wEI'],
        maps_path = datasets.load_maps(['myelinmap', 'fcgradient01'], 'schaefer-100', norm='zscore'),
        emp_fc_tril = datasets.load_functional('FC', 'schaefer-100'),
        emp_fcd_tril = datasets.load_functional('FCD', 'schaefer-100'),
        duration = 60,
        TR = 1,
        sc_path = datasets.load_sc('strength', 'schaefer-100'),
        states_ts = True,
        out_dir = './cmaes_hetero',
    )
    cmaes = optimize.CMAESOptimizer(popsize=30, n_iter=10, seed=1)
    cmaes.setup_problem(problem)
    cmaes.optimize()
    cmaes.save()

Using command line interface:

.. code-block:: bash

    cubnm \
        --model rWW --sc example --emp_fc_tril example --emp_fcd_tril example \
        --out_dir ./cmaes_hetero_cli \
        --TR 1 --duration 60 --states_ts \
        --params G=0.5:2.5,wEE=0.05:0.75,wEI=0.15 \
        optimize \
        --optimizer CMAES --optimizer_seed 0 --n_iter 10 --popsize 30 \
        --het_params wEE wEI --maps example

Grid search
~~~~~~~~~~~
Run a 10x10 grid search of reduced Wong Wang model with G and wEE as free parameters:

.. image:: https://kaggle.com/static/images/open-in-kaggle.svg
   :target: https://www.kaggle.com/code/aminsaberi/cubnm-0-0-2-demo-grid

.. code-block:: python

    from cubnm import datasets, optimize

    gs = optimize.GridSearch(
        model = 'rWW',
        params = {
            'G': (0.5, 2.5, 10),
            'wEE': (0.05, 0.75, 10),
            'wEI': 0.21
        },
        duration = 60,
        TR = 1,
        sc_path = datasets.load_sc('strength', 'schaefer-100'),
        states_ts = True,
        out_dir = './grid',
        sim_verbose = True
    )
    emp_fc_tril = datasets.load_functional('FC', 'schaefer-100')
    emp_fcd_tril = datasets.load_functional('FCD', 'schaefer-100')
    scores = gs.evaluate(emp_fc_tril, emp_fcd_tril)

Using command line interface:

.. code-block:: bash

    cubnm \
        --model rWW --sc example --emp_fc_tril example --emp_fcd_tril example \
        --out_dir ./grid_cli \
        --TR 1 --duration 60 --states_ts \
        --params G=0.5:2.5:10,wEE=0.05:0.75:10,wEI=0.21 --sim_verbose \
        grid

.. quickstart-end
