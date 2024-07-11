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

    cubnm optimize \
        --model rWW --sc example --emp_fc_tril example --emp_fcd_tril example \
        --out_dir ./cmaes_homo_cli \
        --TR 1 --duration 60 --states_ts \
        --params G=0.5:2.5,wEE=0.05:0.75,wEI=0.15 \
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

    cubnm optimize \
        --model rWW --sc example --emp_fc_tril example --emp_fcd_tril example \
        --out_dir ./cmaes_hetero_cli \
        --TR 1 --duration 60 --states_ts \
        --params G=0.5:2.5,wEE=0.05:0.75,wEI=0.15 \
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

    cubnm grid \
        --model rWW --sc example --emp_fc_tril example --emp_fcd_tril example \
        --out_dir ./grid_cli \
        --TR 1 --duration 60 --states_ts \
        --params G=0.5:2.5:10,wEE=0.05:0.75:10,wEI=0.21 --sim_verbose