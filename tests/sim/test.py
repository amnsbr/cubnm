"""
Testing consistency of rJR model simulations
"""
# TODO: consider testing all models in a single test file
import pytest
from cubnm import sim
from cubnm.utils import avail_gpus
import numpy as np
import os
import pickle
import gzip
import itertools
import copy

test_data_dir = os.path.join(os.path.dirname(__file__), '..', 'expected', 'sim')
sel_state_var = 'theta'

def no_gpu():
    # to skip GPU-dependent tests
    return avail_gpus()==0

def get_opts(cpu_gpu_identity=False):
    test_params = []
    # get all model names
    model_names = [m.replace('SimGroup', '') for m in dir(sim) if 'SimGroup' in m]
    model_names.remove('')
    # get all possible combinations
    for model in model_names:
        simgroup_cls = getattr(sim, f'{model}SimGroup')
        model_configs = simgroup_cls._get_test_configs(cpu_gpu_identity=cpu_gpu_identity) 
        combs = itertools.product(*[model_configs[i] for i in model_configs])
        for comb in combs:
            curr_opt = ''
            for i, key in enumerate(model_configs):
                curr_opt += f'{key}:{comb[i]},'
            curr_opt = curr_opt[:-1]
            if 'force_cpu:0' in curr_opt:
                curr_marks = pytest.mark.skipif(no_gpu(), reason="No GPU available")
            else:
                curr_marks = []
            test_param = pytest.param(model, curr_opt, marks=curr_marks)
            test_params.append(test_param)
    return test_params

@pytest.mark.parametrize(
    "model, opts_str", 
    get_opts()
)
def test_single_sim(model, opts_str):
    """
    Tests if BOLD, FC and FCD and selected state variable of a specific simulation
    matches pre-calculated results (based on the same code)
    Note that the expected results are not ground truth (as
    there is no ground truth), and this is simply a test
    of consistency of calculations throughout different versions
    of the code and platforms.

    Parameters
    -------
    model: (str)
        model name
    opts_str: (str)
        with format 'k1:v1,k2:v2,...'
    """
    # These tests are going to fail if code is compiled with alternative
    # toolchains, and therefore are only expected to work with binary
    # wheels or source-compiled library on GCC 12. This is likely due
    # to differences in the random arrays (TODO: test it)
    # TODO: skip tests if binaries are built differently (or check if first [few] element[s] of
    # noise arrays are the same)
    # parse opts into a dictionary
    opts = {i.split(':')[0]:int(i.split(':')[1]) for i in opts_str.split(',')}
    # load expected
    with gzip.open(os.path.join(test_data_dir, f'{model}.pkl.gz'), 'rb') as f:
        expected = pickle.load(f)[opts_str]
    # get test simgroup
    simgroup_cls = getattr(sim, f'{model}SimGroup')
    sg = simgroup_cls._get_test_instance(opts)
    # run simulation
    sg.N = 1
    sg._set_default_params()
    sg.run()
    # compare results and expected
    assert np.isclose(sg.sim_bold, expected['sim_bold'], atol=1e-12).all()
    assert np.isclose(sg.sim_fc_trils, expected['sim_fc_trils'], atol=1e-12).all()
    assert np.isclose(sg.sim_fcd_trils, expected['sim_fcd_trils'], atol=1e-12).all()
    assert np.isclose(
        sg.sim_states[simgroup_cls.sel_state_var], 
        expected['sim_sel_state'], 
        atol=1e-12
    ).all() # TODO: use all state variables

@pytest.mark.parametrize(
    "model, opts_str", 
    get_opts()
)
def test_identical_sims(model, opts_str):
    """
    Tests if two identical simulations run on parallel (GPU/multi-CPU)
    or serially (single thread) result in identical BOLD, FC and FCD,
    and selected state.

    Parameters
    -------
    model: (str)
        model name
    opts_str: (str)
        with format 'k1:v1,k2:v2,...'
    """
    opts = {i.split(':')[0]:int(i.split(':')[1]) for i in opts_str.split(',')}
    # get test simgroup
    simgroup_cls = getattr(sim, f'{model}SimGroup')
    sg = simgroup_cls._get_test_instance(opts)
    sg.N = 2
    sg._set_default_params()
    sg.run()
    # compare results
    assert np.isclose(sg.sim_bold[0], sg.sim_bold[1], atol=1e-12).all()
    assert np.isclose(sg.sim_fc_trils[0], sg.sim_fc_trils[1], atol=1e-12).all()
    assert np.isclose(sg.sim_fcd_trils[0], sg.sim_fcd_trils[1], atol=1e-12).all()
    assert np.isclose(
        sg.sim_states[simgroup_cls.sel_state_var][0], 
        sg.sim_states[simgroup_cls.sel_state_var][1], 
        atol=1e-12
    ).all() # TODO: use all state variables

@pytest.mark.skipif(no_gpu(), reason="No GPU available")
@pytest.mark.parametrize(
    "model, opts_str", 
    get_opts(cpu_gpu_identity=True)
)
def test_identical_cpu_gpu(model, opts_str):
    """
    Tests if two identical simulations run on CPU vs GPU
    result in identical BOLD, FC, FCD and selected state variable.

    Parameters
    -------
    model: (str)
        model name
    opts_str: (str)
        with format 'k1:v1,k2:v2,...'
    """
    opts = {i.split(':')[0]:int(i.split(':')[1]) for i in opts_str.split(',')}
    simgroup_cls = getattr(sim, f'{model}SimGroup')
    sg = simgroup_cls._get_test_instance(opts)
    sg.N = 1
    sg._set_default_params()
    sim_bolds = {}
    sim_fc_trils = {}
    sim_fcd_trils = {}
    sim_sel_states = {}
    for force_cpu in [True, False]:
        sg.force_cpu = force_cpu
        sg.run()
        sim_bolds[force_cpu] = sg.sim_bold.copy()
        sim_fc_trils[force_cpu] = sg.sim_fc_trils.copy()
        sim_fcd_trils[force_cpu] = sg.sim_fcd_trils.copy()
        sim_sel_states[force_cpu] = sg.sim_states[simgroup_cls.sel_state_var].copy() # TODO: use all state variables
    # compare results
    assert np.isclose(sim_bolds[False], sim_bolds[True], atol=1e-12).all()
    assert np.isclose(sim_fc_trils[False], sim_fc_trils[True], atol=1e-12).all()
    assert np.isclose(sim_fcd_trils[False], sim_fcd_trils[True], atol=1e-12).all()
    assert np.isclose(sim_sel_states[False], sim_sel_states[True], atol=1e-12).all()
