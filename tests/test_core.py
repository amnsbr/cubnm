"""
Tests for the core run_simulations function
"""
import pytest
from cuBNM._core import run_simulations
from cuBNM import sim
from cuBNM import datasets
from cuBNM.utils import avail_gpus
import numpy as np
import os

# constant configs used in all simulations
NODES = 100
TIME_STEPS = 60 * 1000
BOLD_TR = 1 * 1000
WINDOW_SIZE = 10
WINDOW_STEP = 2
RAND_SEED = 410
EXTENDED_OUTPUT = True
EXTENDED_OUTPUT_TS = True
# it is important to force reinit across tests because N_SIMS might be different
FORCE_REINIT = True
SC = datasets.load_sc('strength', 'schaefer-100').flatten()

def no_gpu():
    # to skip GPU-dependent tests
    return avail_gpus()==0

# TODO: include one of the extended output in the tests

@pytest.mark.parametrize(
    "opts, expected", 
    [
        pytest.param('use_gpu:1,do_delay:0,do_fic:1', {'bold': 0.25229340320262333, 'fc': 0.15228818488965185, 'fcd': 0.053617605632305304}, marks=pytest.mark.skipif(no_gpu(), reason="No GPU available")),
        pytest.param('use_gpu:1,do_delay:0,do_fic:0', {'bold': 0.5584541639446075, 'fc': 0.13219002265346044, 'fcd': 0.06140882888975842}, marks=pytest.mark.skipif(no_gpu(), reason="No GPU available")),
        pytest.param('use_gpu:1,do_delay:1,do_fic:1', {'bold': 0.2532300911753404, 'fc': 0.15932360699106757, 'fcd': 0.05324493950679656}, marks=pytest.mark.skipif(no_gpu(), reason="No GPU available")),
        pytest.param('use_gpu:1,do_delay:1,do_fic:0', {'bold': 0.5607041128108943, 'fc': 0.12980554063007843, 'fcd': 0.06004315998040354}, marks=pytest.mark.skipif(no_gpu(), reason="No GPU available")),
        pytest.param('use_gpu:0,do_delay:0,do_fic:1', {'bold': 0.25229340320262333, 'fc': 0.1522881848896672, 'fcd': 0.05361760563230998}, marks=pytest.mark.skip(reason="not implemented")),
        pytest.param('use_gpu:0,do_delay:0,do_fic:0', {'bold': 0.5584541639446068, 'fc': 0.13219002265345683, 'fcd': 0.06140882888975444}, marks=pytest.mark.skip(reason="not implemented")),
        pytest.param('use_gpu:0,do_delay:1,do_fic:1', {}, marks=pytest.mark.skip(reason="not implemented")),
        pytest.param('use_gpu:0,do_delay:1,do_fic:0', {}, marks=pytest.mark.skip(reason="not implemented")),
    ]) # the expected values based on tests run on gpu1.htc.inm7.de using commit fa4250f29dd6546480382ae5db2429a62634eda4 with noise segmenting
def test_single_sim(opts, expected):
    """
    Tests if BOLD, FC and FCD of a specific simulation
    matches pre-calculated values (based on the same code)
    Note that the values used here are not ground truth (as
    there is no ground truth), and this is simply a test
    of consistency of calculations throughout different versions
    of the code and platforms.

    Parameters
    -------
    opts: (str)
        with format 'use_gpu:*,do_delay:*,do_fic:*'
    expected: (dict)
        with keys 'bold' (expected bold[500]), 'fc' (expected fc[30]), 'fcd' (expected fcd[30])
    """
    # These tests are going to fail if code is compiled with alternative
    # toolchains, and therefore are only expected to work with binary
    # wheels or source-compiled library on GCC 12. This is likely due
    # to differences in the random arrays (TODO: test it)
    # TODO: skip tests if binaries are built otherwise
    opts = {i.split(':')[0]:int(i.split(':')[1]) for i in opts.split(',')}
    N_SIMS = 1
    G_list = np.repeat(0.5, N_SIMS)
    w_EE_list = np.repeat(0.21, NODES*N_SIMS)
    w_EI_list = np.repeat(0.15, NODES*N_SIMS)
    if opts['do_fic']:
        w_IE_list = np.repeat(0.0, NODES*N_SIMS)
    else:
        w_IE_list = np.repeat(1.0, NODES*N_SIMS)
    model_config = {
        'do_fic': str(int(opts['do_fic'])),
        'max_fic_trials': '5',
    }
    if opts['do_delay']:
        v_list = np.repeat(0.5, N_SIMS)
        SC_dist = datasets.load_sc('length', 'schaefer-100').flatten()
        model_config["sync_msec"] = "1"
    else:
        v_list = np.repeat(0.0, N_SIMS) # doesn't matter what it is!
        SC_dist = np.zeros(NODES*NODES, dtype=float) # doesn't matter what it is!
    global_params = G_list[np.newaxis, :]
    regional_params = np.vstack([w_EE_list, w_EI_list, w_IE_list])
    out = run_simulations(
        'rWW', SC, SC_dist, global_params, regional_params, v_list,
        model_config, EXTENDED_OUTPUT, EXTENDED_OUTPUT_TS,
        opts['do_delay'], FORCE_REINIT, (not opts['use_gpu']),
        N_SIMS, NODES, TIME_STEPS, BOLD_TR,
        WINDOW_SIZE, WINDOW_STEP, RAND_SEED
    )
    sim_bolds , sim_fc_trils, sim_fcd_trils = out[:3]
    # TODO: consider comparing the entire arrays
    assert np.isclose(sim_bolds[0, 500], expected['bold'], atol=1e-12)
    assert np.isclose(sim_fc_trils[0, 30], expected['fc'], atol=1e-12)
    assert np.isclose(sim_fcd_trils[0, 30], expected['fcd'], atol=1e-12)

@pytest.mark.parametrize(
    "opts", 
    [
        pytest.param('use_gpu:1,do_delay:0,do_fic:1', marks=pytest.mark.skipif(no_gpu(), reason="No GPU available")),
        pytest.param('use_gpu:1,do_delay:0,do_fic:0', marks=pytest.mark.skipif(no_gpu(), reason="No GPU available")),
        pytest.param('use_gpu:1,do_delay:1,do_fic:1', marks=pytest.mark.skipif(no_gpu(), reason="No GPU available")),
        pytest.param('use_gpu:1,do_delay:1,do_fic:0', marks=pytest.mark.skipif(no_gpu(), reason="No GPU available")),
        pytest.param('use_gpu:0,do_delay:0,do_fic:1', marks=pytest.mark.skip(reason="not implemented")),
        pytest.param('use_gpu:0,do_delay:0,do_fic:0', marks=pytest.mark.skip(reason="not implemented")),
        pytest.param('use_gpu:0,do_delay:1,do_fic:1', marks=pytest.mark.skip(reason="not implemented")),
        pytest.param('use_gpu:0,do_delay:1,do_fic:0', marks=pytest.mark.skip(reason="not implemented")),
    ])
def test_identical_sims(opts):
    """
    Tests if two identical simulations run on parallel (GPU/multi-CPU)
    or serially (single thread) result in identical BOLD, FC and FCD

    Parameters
    -------
    opts: (str)
        with format 'use_gpu:*,do_delay:*,do_fic:*'
    """
    opts = {i.split(':')[0]:int(i.split(':')[1]) for i in opts.split(',')}
    N_SIMS = 2
    G_list = np.repeat(0.5, N_SIMS)
    w_EE_list = np.repeat(0.21, NODES*N_SIMS)
    w_EI_list = np.repeat(0.15, NODES*N_SIMS)
    if opts['do_fic']:
        w_IE_list = np.repeat(0.0, NODES*N_SIMS)
    else:
        w_IE_list = np.repeat(1.0, NODES*N_SIMS)
    model_config = {
        'do_fic': str(int(opts['do_fic'])),
        'max_fic_trials': '5',
    }
    if opts['do_delay']:
        v_list = np.repeat(1.0, N_SIMS)
        SC_dist = datasets.load_sc('length', 'schaefer-100').flatten()
        model_config['sync_msec'] = '1'
    else:
        v_list = np.repeat(0.0, N_SIMS) # doesn't matter what it is!
        SC_dist = np.zeros(NODES*NODES, dtype=float) # doesn't matter what it is!
    global_params = G_list[np.newaxis, :]
    regional_params = np.vstack([w_EE_list, w_EI_list, w_IE_list])
    out = run_simulations(
        'rWW', SC, SC_dist, global_params, regional_params, v_list,
        model_config, EXTENDED_OUTPUT, EXTENDED_OUTPUT_TS,
        opts['do_delay'], FORCE_REINIT, (not opts['use_gpu']),
        N_SIMS, NODES, TIME_STEPS, BOLD_TR,
        WINDOW_SIZE, WINDOW_STEP, RAND_SEED
    )
    sim_bolds , sim_fc_trils, sim_fcd_trils = out[:3]

    assert (sim_bolds[0] == sim_bolds[1]).all() # consider using np.isclose
    assert (sim_fc_trils[0] == sim_fc_trils[1]).all()
    assert (sim_fcd_trils[0] == sim_fcd_trils[1]).all()

# @pytest.mark.skipif(no_gpu(), reason="No GPU available")
@pytest.mark.skip(reason="CPU not implemented")
def test_gpu_cpu_identical_fic_no_delay():
    # check if GPU and CPU outputs with identical inputs
    # are similar (up to the precision allowed with doubles)
    # TODO: see if it's needed to parameterize this test as well across the 4 conditions
    N_SIMS = 1
    G_list = np.repeat(0.5, N_SIMS)
    w_EE_list = np.repeat(0.21, NODES*N_SIMS)
    w_EI_list = np.repeat(0.15, NODES*N_SIMS)
    w_IE_list = np.repeat(0.0, NODES*N_SIMS)
    do_fic = True
    do_delay = False
    v_list = np.repeat(0.0, N_SIMS)
    SC_dist = np.zeros(NODES*NODES, dtype=float)

    # first run on CPU
    force_cpu = True
    out = run_simulations(
        SC, SC_dist, G_list, w_EE_list, w_EI_list, w_IE_list, v_list,
        do_fic, EXTENDED_OUTPUT, EXTENDED_OUTPUT_TS,
        do_delay, FORCE_REINIT, force_cpu,
        N_SIMS, NODES, TIME_STEPS, BOLD_TR,
        WINDOW_SIZE, WINDOW_STEP, RAND_SEED
    )
    cpu_sim_bolds , cpu_sim_fc_trils, cpu_sim_fcd_trils = out[:3]

    # then run on GPU
    force_cpu = False
    out = run_simulations(
        SC, SC_dist, G_list, w_EE_list, w_EI_list, w_IE_list, v_list,
        do_fic, EXTENDED_OUTPUT, EXTENDED_OUTPUT_TS,
        do_delay, FORCE_REINIT, force_cpu,
        N_SIMS, NODES, TIME_STEPS, BOLD_TR,
        WINDOW_SIZE, WINDOW_STEP, RAND_SEED
    )
    gpu_sim_bolds , gpu_sim_fc_trils, gpu_sim_fcd_trils = out[:3]
    assert np.isclose(cpu_sim_bolds[0], gpu_sim_bolds[0], atol=1e-12).all()
    # assert (cpu_sim_bolds[0]==gpu_sim_bolds[0]).all() # this might fail especially after running the previous tests
    assert np.isclose(cpu_sim_fc_trils[0], gpu_sim_fc_trils[0], atol=1e-12).all()
    assert np.isclose(cpu_sim_fcd_trils[0], gpu_sim_fcd_trils[0], atol=1e-12).all()