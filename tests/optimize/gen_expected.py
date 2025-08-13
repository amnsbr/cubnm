"""
Generates expected optimization outputs used to test stability of optimization through
different versions.
"""
import os
import sys
import pandas as pd
from cubnm import optimize, datasets

test_data_dir = os.path.join(os.path.dirname(__file__), '..', 'expected', 'optimize')

# define problem arguments
emp_bold = datasets.load_bold('schaefer-100')
problem_args = dict(
    model = 'rWW',
    params = {
        'G': (0.001, 10.0),
        'w_p': (0, 2.0),
        'J_N': (0.001, 0.5),
    },
    emp_bold = emp_bold,
    het_params = ['w_p', 'J_N'],
    maps = datasets.load_maps(
        ['myelinmap', 'fcgradient01'],
        'schaefer-100', norm='minmax'
    ),
    maps_coef_range = (-3, 3),
    duration = 60,
    TR = 1,
    window_size = 10,
    window_step = 2,
    sc = datasets.load_sc('strength', 'schaefer-100'),
)
# for grid search use a smaller number of free parameters
grid_problem_args = problem_args.copy()
grid_problem_args.update({
    'maps': datasets.load_maps(
        ['myelinmap'],
        'schaefer-100', norm='minmax'
    ),
    'het_params': ['J_N']
})


def gen_expected(optimizer_name):
    # get optimizer class
    Optimizer = getattr(optimize, f'{optimizer_name}Optimizer')
    if optimizer_name == 'Grid':
        p_args = grid_problem_args
    else:
        p_args = problem_args.copy()
        # initialize problem
        if Optimizer.max_obj>1:
            p_args['multiobj'] = True
    problem = optimize.BNMProblem(**p_args)
    if optimizer_name == 'Grid':
        optimizer = Optimizer()
        optimizer.optimize(problem, grid_shape={'G': 3, 'w_p': 2, 'J_N': 2, 'J_Nscale0': 2})
    else:
        # initialize optimizer and register problem
        optimizer = Optimizer(popsize=10, n_iter=2, seed=1)
        optimizer.setup_problem(problem)
        # run optimization
        optimizer.optimize()
    # warn if history has changed
    if os.path.exists(os.path.join(test_data_dir, f'{optimizer_name}.csv')):
        prev_hist = pd.read_csv(os.path.join(test_data_dir, f'{optimizer_name}.csv'))
        if not prev_hist.equals(optimizer.history):
            print(f"Warning: {optimizer_name} history has changed from the previous version")
    # save history
    optimizer.history.to_csv(os.path.join(test_data_dir, f'{optimizer_name}.csv'))

if __name__ == "__main__":
    optimizer_name = sys.argv[1]
    os.makedirs(test_data_dir, exist_ok=True)
    gen_expected(optimizer_name)