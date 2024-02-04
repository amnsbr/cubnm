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
        model = 'rWW',
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