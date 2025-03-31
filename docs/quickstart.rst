Quickstart
-------------
Evolutionary optimization
~~~~~~~~~~~~~~~~~~~~~~~~~
Run a CMAES optimization of reduced Wong Wang model with G and wEE as free parameters:

.. image:: https://kaggle.com/static/images/open-in-kaggle.svg 
   :target: https://www.kaggle.com/code/aminsaberi/cubnm-demo-cmaes-homogeneous

.. code-block:: python

    from cubnm import datasets, optimize

    problem = optimize.BNMProblem(
        model = 'rWW',
        params = {
            'G': (0.5, 2.5),
            'wEE': (0.05, 0.75),
            'wEI': 0.15,
        },
        sc = datasets.load_sc('strength', 'schaefer-100'),
        emp_bold = datasets.load_bold('schaefer-100'),
        duration = 60,
        TR = 1,
        window_size = 10,
        window_step = 2,
        states_ts = True,
        out_dir = './cmaes_homo',
    )
    cmaes = optimize.CMAESOptimizer(popsize=20, n_iter=10, seed=1)
    cmaes.setup_problem(problem)
    cmaes.optimize()
    cmaes.save()

Using command line interface:

.. code-block:: bash

    cubnm optimize \
        --model rWW --sc example --emp_bold example \
        --out_dir ./cmaes_homo_cli \
        --TR 1 --duration 60 --window_size 10 --window_step 2 --states_ts \
        --params G=0.5:2.5,wEE=0.05:0.75,wEI=0.15 \
        --optimizer CMAES --optimizer_seed 0 --n_iter 10 --popsize 20

Run a CMAES optimization of reduced Wong Wang model with G as a global free parameter and wEE and wEI as
regional free parameters that are regionally heterogeneous based on a weighted combination of two fixed
maps (HCP T1w/T2w, HCP FC G1):

.. image:: https://kaggle.com/static/images/open-in-kaggle.svg
   :target: https://www.kaggle.com/code/aminsaberi/cubnm-demo-cmaes-heterogeneous

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
        maps = datasets.load_maps(['myelinmap', 'fcgradient01'], 'schaefer-100', norm='zscore'),
        sc = datasets.load_sc('strength', 'schaefer-100'),
        emp_bold = datasets.load_bold('schaefer-100'),
        duration = 60,
        TR = 1,
        window_size = 10,
        window_step = 2,
        states_ts = True,
        out_dir = './cmaes_hetero',
    )
    cmaes = optimize.CMAESOptimizer(popsize=30, n_iter=10, seed=1)
    cmaes.setup_problem(problem)
    cmaes.optimize()
    cmaes.save()

Using command line interface:

.. code-block:: bash

    cubnm optimize \
        --model rWW --sc example --emp_fc_tril example --emp_fcd_tril example \
        --out_dir ./cmaes_hetero_cli \
        --TR 1 --duration 60 --window_size 10 --window_step 2 --states_ts \
        --params G=0.5:2.5,wEE=0.05:0.75,wEI=0.15 \
        --optimizer CMAES --optimizer_seed 0 --n_iter 10 --popsize 30 \
        --het_params wEE wEI --maps example

Grid search
~~~~~~~~~~~
Run a 10x10 grid search of reduced Wong Wang model with G and wEE as free parameters:

.. image:: https://kaggle.com/static/images/open-in-kaggle.svg
   :target: https://www.kaggle.com/code/aminsaberi/cubnm-demo-grid

.. code-block:: python

    from cubnm import datasets, optimize

    problem = optimize.BNMProblem(
        model = 'rWW',
        params = {
            'G': (0.5, 2.5),
            'wEE': (0.05, 0.75),
            'wEI': 0.21,
        },
        sc = datasets.load_sc('strength', 'schaefer-100'),
        emp_bold = datasets.load_bold('schaefer-100'),
        duration = 60,
        TR = 1,
        window_size = 10,
        window_step = 2,
        states_ts = True,
        sim_verbose = True,
        out_dir = './grid',
    )
    grid = optimize.GridOptimizer()
    grid.optimize(problem, grid_shape={'G': 10, 'wEE': 10})
    grid.save()

Using command line interface:

.. code-block:: bash

    cubnm grid \
        --model rWW --sc example --emp_bold example \
        --out_dir ./grid_cli \
        --TR 1 --duration 60 --window_size 10 --window_step 2 --states_ts \
        --params G=0.5:2.5,wEE=0.05:0.75,wEI=0.21 --grid_shape G=10,wEI=10 --sim_verbose