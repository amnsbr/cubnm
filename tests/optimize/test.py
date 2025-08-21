"""
Testing consistency of rJR model simulations
Since stability of simulations is checked elsewhere, this test only concerns
the optimization outputs (therefore there is no checks for CPU-GPU identity, etc.),
and is run on a specific problem of rWW model on GPUs and if not available, on CPUs.
"""
import pytest
import os
import tempfile
import copy
import pandas as pd
import numpy as np
from cubnm import optimize, datasets

test_data_dir = os.path.join(os.path.dirname(__file__), '..', 'expected', 'optimize')

# define problem arguments
emp_bold = datasets.load_bold('schaefer-100')
problem_args = dict(
    model = 'rWW',
    params = {
        'G': (0.001, 5.0),
        'wEE': (0.001, 3.0),
        'wEI': (0.001, 3.0),
    },
    emp_bold = emp_bold,
    het_params = ['wEE', 'wEI'],
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
    'het_params': ['wEE']
})

def get_test_params(for_batch=False):
    """
    Get all possible test parameters for all the optimizers

    Parameters
    ----------
    for_batch: (bool)
        if True, returns parameters for batch optimization
        i.e., will exclude non-Pymoo optimizers
    
    Returns
    -------
    test_params: (list)
        list of pytest test parameters
    """
    test_params = []
    # get all optimizer names
    optimizer_names = [m.replace('Optimizer', '') for m in dir(optimize) if 'Optimizer' in m]
    # exclude base classes
    for exc_name in ['', 'Pymoo']:
        optimizer_names.remove(exc_name)
    # exclude non-Pymoo optimizers if for batch
    if for_batch:
        optimizer_names.remove('Grid')
    # get all possible combinations
    for optimizer_name in optimizer_names:
        test_param = pytest.param(optimizer_name)
        test_params.append(test_param)
    return test_params

@pytest.mark.parametrize(
    "optimizer_name", 
    get_test_params()
)
def test_opt(optimizer_name):
    """
    Tests if history of the optimizer matches the expected 
    (saved from previous versions).

    Parameters
    -------
    optimizer_name: (str)
    """
    # get optimizer class
    Optimizer = getattr(optimize, f'{optimizer_name}Optimizer')
    # load expected history
    hist_expected = pd.read_csv(os.path.join(test_data_dir, f'{optimizer_name}.csv'), index_col=0)
    # make a copy of problem args for current test
    if optimizer_name == 'Grid':
        p_args = grid_problem_args.copy()
    else:
        p_args = problem_args.copy()
        if Optimizer.max_obj>1:
            p_args['multiobj'] = True
    # initialize problem
    p_args['out_dir'] = tempfile.mkdtemp() # to test saving in a temporary directory
    problem = optimize.BNMProblem(**p_args)
    # run optimization
    if optimizer_name == 'Grid':
        optimizer = Optimizer()
        optimizer.optimize(problem, grid_shape={'G': 3, 'wEE': 2, 'wEI': 2, 'wEEscale0': 2})
    else:    
        optimizer = Optimizer(popsize=10, n_iter=2, seed=1)
        optimizer.setup_problem(problem)
        optimizer.optimize()
    # compare history
    assert np.isclose(hist_expected.values, optimizer.history.values, atol=1e-6).all()
    # test saving
    optimizer.save()

@pytest.mark.parametrize(
    "optimizer_name", 
    get_test_params(for_batch=True)
)
def test_batch(optimizer_name):
    """
    Tests if batch and independent optimization results are consistent.

    Parameters
    -------
    optimizer_name: (str)
    """
    # make a copy of problem args for current test
    p_args = copy.deepcopy(problem_args)
    # get optimizer class
    Optimizer = getattr(optimize, f'{optimizer_name}Optimizer')
    # initialize problem
    if Optimizer.max_obj>1:
        p_args['multiobj'] = True
    p_args['out_dir'] = tempfile.mkdtemp() # to test saving in a temporary directory
    problem = optimize.BNMProblem(**p_args)
    # initialize optimizer
    o1 = Optimizer(popsize=10, n_iter=2, seed=1)
    o2 = Optimizer(popsize=10, n_iter=2, seed=2)
    # run batch optimization
    batch_optimizers = optimize.batch_optimize([o1, o1, o2, o2], problem)
    # run independent optimizations
    o1.setup_problem(copy.deepcopy(problem))
    o1.optimize()
    o2.setup_problem(copy.deepcopy(problem))
    o2.optimize()
    # compare histories
    assert np.isclose(batch_optimizers[0].history.values, batch_optimizers[1].history.values, atol=1e-6).all()
    assert np.isclose(batch_optimizers[2].history.values, batch_optimizers[3].history.values, atol=1e-6).all()
    assert not np.isclose(batch_optimizers[0].history.values, batch_optimizers[2].history.values, atol=1e-6).all()
    assert np.isclose(batch_optimizers[0].history.values, o1.history.values, atol=1e-6).all()
    assert np.isclose(batch_optimizers[2].history.values, o2.history.values, atol=1e-6).all()

@pytest.mark.parametrize(
    "optimizer_name", 
    get_test_params(for_batch=True)
)
def test_batch_variable_sc(optimizer_name):
    """
    Tests if batch and independent optimization results are consistent
    when different SCs are used.

    Parameters
    -------
    optimizer_name: (str)
    """
    # make a copy of problem args for current test
    p_args = copy.deepcopy(problem_args)
    # get optimizer class
    Optimizer = getattr(optimize, f'{optimizer_name}Optimizer')
    # initialize problem
    if Optimizer.max_obj>1:
        p_args['multiobj'] = True
    p_args['out_dir'] = tempfile.mkdtemp() # to test saving in a temporary directory
    p1 = optimize.BNMProblem(**p_args)
    p_args['sc'] = p_args['sc'][::-1, ::-1].copy() # reverse SC
    p2 = optimize.BNMProblem(**p_args)
    # initialize optimizer
    o = Optimizer(popsize=10, n_iter=2, seed=1)
    # run batch optimization
    batch_optimizers = optimize.batch_optimize(o, [p1, p2])
    # run independent optimizations
    o1 = Optimizer(popsize=10, n_iter=2, seed=1)
    o1.setup_problem(copy.deepcopy(p1))
    o1.optimize()
    o2 = Optimizer(popsize=10, n_iter=2, seed=1)
    o2.setup_problem(copy.deepcopy(p2))
    o2.optimize()
    # compare histories
    assert not np.isclose(batch_optimizers[0].history.values, batch_optimizers[1].history.values, atol=1e-6).all()
    assert np.isclose(batch_optimizers[0].history.values, o1.history.values, atol=1e-6).all()
    assert np.isclose(batch_optimizers[1].history.values, o2.history.values, atol=1e-6).all()