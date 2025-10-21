"""
Testing consistency of simulations across different versions of the code,
between identical simulations on the same platform, and between identical
simulations run on CPU vs GPU.
"""
import pytest
from cubnm import sim
from cubnm.utils import avail_gpus
from cubnm._setup_opts import noise_segment_flag
import numpy as np
import os
import pickle
import gzip
import itertools

test_data_dir = os.path.join(os.path.dirname(__file__), '..', 'expected', 'sim')

def no_gpu():
    # to skip GPU-dependent tests
    return avail_gpus()==0

def get_test_params(cpu_gpu_identity=False):
    """
    Get all possible test parameters for all the models

    Parameters
    -------
    cpu_gpu_identity: (bool)
        whether cpu-gpu identity is being tested, in which
        case force_cpu will not be a variable
    
    Returns
    -------
    test_params: (list)
        list of pytest test parameters
    """
    test_params = []
    # get all model names
    model_names = [m.replace('SimGroup', '') for m in dir(sim) if m.endswith('SimGroup')]
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
            curr_marks = []
            if 'force_cpu:0' in curr_opt:
                curr_marks.append(pytest.mark.skipif(no_gpu(), reason="No GPU available"))
            if (model == 'Kuramoto') and ('do_delay:1' in curr_opt) and cpu_gpu_identity:
                # mark it as xfail due to known CPU-GPU differences
                curr_marks.append(pytest.mark.xfail(reason=
                    "Known issue #24: Kuramoto model with delay may "
                    "result in different outcomes on CPU vs GPU"
                ))
            test_param = pytest.param(model, curr_opt, marks=curr_marks)
            test_params.append(test_param)
    return test_params

@pytest.mark.parametrize(
    "model, opts_str", 
    get_test_params()
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
    sg.param_lists['G'] = np.array([0.5])
    sg._set_default_params(missing=True)
    sg.run()
    # compare results and expected
    assert np.isclose(sg.sim_fc_trils, expected['sim_fc_trils'], atol=1e-6).all()
    assert np.isclose(sg.sim_fcd_trils, expected['sim_fcd_trils'], atol=1e-6).all()
    assert np.isclose(sg.sim_bold, expected['sim_bold'], atol=1e-6).all()
    assert np.isclose(
        sg.sim_states[simgroup_cls.sel_state_var], 
        expected['sim_sel_state'], 
        atol=1e-6
    ).all() # TODO: use all state variables

@pytest.mark.parametrize(
    "model, opts_str", 
    get_test_params()
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
    sg.param_lists['G'] = np.array([0.5, 0.5])
    sg._set_default_params(missing=True)
    sg.run()
    # compare results
    assert np.isclose(sg.sim_fc_trils[0], sg.sim_fc_trils[1], atol=1e-6).all()
    assert np.isclose(sg.sim_fcd_trils[0], sg.sim_fcd_trils[1], atol=1e-6).all()
    assert np.isclose(sg.sim_bold[0], sg.sim_bold[1], atol=1e-6).all()
    assert np.isclose(
        sg.sim_states[simgroup_cls.sel_state_var][0], 
        sg.sim_states[simgroup_cls.sel_state_var][1], 
        atol=1e-6
    ).all() # TODO: use all state variables

@pytest.mark.skipif(no_gpu(), reason="No GPU available")
@pytest.mark.parametrize(
    "model, opts_str", 
    get_test_params(cpu_gpu_identity=True)
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
    sg.param_lists['G'] = np.array([0.5])
    sg._set_default_params(missing=True)
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
    assert np.isclose(sim_fc_trils[False], sim_fc_trils[True], atol=1e-6).all()
    assert np.isclose(sim_fcd_trils[False], sim_fcd_trils[True], atol=1e-6).all()
    assert np.isclose(sim_bolds[False], sim_bolds[True], atol=1e-6).all()
    assert np.isclose(sim_sel_states[False], sim_sel_states[True], atol=1e-6).all()

@pytest.mark.skipif(not noise_segment_flag, reason="Only indicated for segmented noise")
def test_get_noise():
    """
    Tests if the noise returned by `SimGroup.get_noise` is consistent
    with the full noise array used in the simulation, by mimicking how
    it is used in the simulation.
    """
    from decimal import Decimal
    from tqdm import tqdm
    # create a test simgroup with 10 fully connected nodes
    nodes = 10
    sc = np.ones((nodes, nodes), dtype=float)
    np.fill_diagonal(sc, 0)
    sg = sim.rWWSimGroup(
        duration = 8,
        noise_segment_length=3,
        TR = 0.1,
        dt = '0.2',
        bw_dt = '0.8',
        sc = sc,
        do_fc = False,
        do_fcd = False,
        gof_terms = [],
        force_cpu = True,
        noise_out = True
    )
    # run simulation
    sg.N = 1
    sg.param_lists['G'] = np.array([0.5])
    sg._set_default_params(missing=True)
    sg.run()
    # get noise array via get_noise
    noise_recon = sg.get_noise()
    # reconstruct noise using a loop mimicking
    # the loop in the core code
    noise_mimick = np.empty_like(noise_recon)

    # preparations
    bw_dt_s = sg.bw_dt / 1000 # in C++ bw_dt is in seconds
    noise_bw_it = int((Decimal(sg.noise_segment_length*1000) / 1000) / bw_dt_s)
    noise_repeats = int(np.ceil(sg.bw_it / noise_bw_it))
    assert noise_repeats == sg._shuffled_nodes.shape[0]
    # shuffled ts and shuffled nodes are flattened in C++
    shuffled_ts = sg._shuffled_ts.flatten()
    shuffled_nodes = sg._shuffled_nodes.flatten()

    # main loop mimicking the C++/CUDA code
    curr_noise_repeat = 0
    for bw_i in tqdm(range(sg.bw_it)):
        sh_ts_noise = shuffled_ts[
            (bw_i % noise_bw_it)
            +(curr_noise_repeat*noise_bw_it)
        ]
        for inner_i in range(sg.inner_it):
            for j in range(sg.nodes):
                sh_j = shuffled_nodes[curr_noise_repeat*sg.nodes + j]
                noise_idx = (
                    ((sh_ts_noise * sg.inner_it + inner_i)
                        * sg.nodes * sg.n_noise)
                    + (sh_j * sg.n_noise)
                )
                noise_mimick[0, j, bw_i, inner_i] = sg._noise[noise_idx]
                noise_mimick[1, j, bw_i, inner_i] = sg._noise[noise_idx+1]
        # reset noise segment time 
        # and shuffle nodes if the segment
        # has reached to the end
        if ((bw_i+1) % noise_bw_it == 0):
            if (bw_i+1 < sg.bw_it):
                # at the last time point don't do this
                # to avoid going over the extent of shuffled_nodes
                curr_noise_repeat += 1
    assert np.isclose(noise_recon, noise_mimick).all()