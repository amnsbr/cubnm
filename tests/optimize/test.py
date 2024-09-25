"""
Testing consistency of rJR model simulations
Since stability of simulations is checked elsewhere, this test only concerns
the optimization outputs (therefore there is no checks for CPU-GPU identity, etc.),
and is run on a specific problem of rWW model on GPUs and if not available, on CPUs.
"""
import pytest
import os
import tempfile
import pandas as pd
import numpy as np
from cubnm import optimize, datasets

test_data_dir = os.path.join(os.path.dirname(__file__), '..', 'expected', 'optimize')

# define problem arguments
problem_args = dict(
    model = 'rWW',
    params = {
        'G': (1.0, 3.0),
        'wEE': (0.05, 0.5),
        'wEI': 0.15,
    },
    emp_fc_tril = datasets.load_functional('FC', 'schaefer-100'),
    emp_fcd_tril = datasets.load_functional('FCD', 'schaefer-100'),
    het_params = ['wEE', 'wEI'],
    maps = datasets.load_maps(
        ['myelinmap', 'fcgradient01'],
        'schaefer-100', norm='minmax'
    ),
    duration = 60,
    TR = 1,
    sc = datasets.load_sc('strength', 'schaefer-100'),
)

def get_test_params():
    """
    Get all possible test parameters for all the optimizers
    
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
    # load expected history
    hist_expected = pd.read_csv(os.path.join(test_data_dir, f'{optimizer_name}.csv'), index_col=0)
    # get optimizer class
    Optimizer = getattr(optimize, f'{optimizer_name}Optimizer')
    # initialize problem
    if Optimizer.max_obj>1:
        problem_args['multiobj'] = True
    problem_args['out_dir'] = tempfile.mkdtemp() # to test saving in a temporary directory
    problem = optimize.BNMProblem(**problem_args)
    # initialize optimizer and register problem
    optimizer = Optimizer(popsize=10, n_iter=2, seed=1)
    optimizer.setup_problem(problem)
    # run optimization
    optimizer.optimize()
    # compare history
    assert np.isclose(hist_expected.values, optimizer.history.values).all()
    # test saving
    optimizer.save()