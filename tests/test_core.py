"""
Tests for the core run_simulations function
"""
import pytest
from cuBNM.core import run_simulations
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
# it is important to force reinit across tests because N_SIMS might be different
FORCE_REINIT = True
SC = datasets.load_sc('strength', 'schaefer-100').flatten()

def no_gpu():
    # to skip GPU-dependent tests
    return avail_gpus()==0

@pytest.mark.parametrize(
    "opts, expected", 
    [
        pytest.param('use_gpu:1,do_delay:0,do_fic:1', {'bold': 0.7420394211841863, 'fc': 0.1522881886810883, 'fcd': 0.05361760521411114}, marks=pytest.mark.skipif(no_gpu(), reason="No GPU available")),
        pytest.param('use_gpu:1,do_delay:0,do_fic:0', {'bold': 1.6425122468959044, 'fc': 0.13219002758619677, 'fcd': 0.06140883019133118}, marks=pytest.mark.skipif(no_gpu(), reason="No GPU available")),
        pytest.param('use_gpu:1,do_delay:1,do_fic:1', {'bold': 0.7447943858098247, 'fc': 0.15932362134671932, 'fcd': 0.05324494338900664}, marks=pytest.mark.skipif(no_gpu(), reason="No GPU available")),
        pytest.param('use_gpu:1,do_delay:1,do_fic:0', {'bold': 1.6491297435614538, 'fc': 0.1298055398225281, 'fcd': 0.0600431676259622}, marks=pytest.mark.skipif(no_gpu(), reason="No GPU available")),
        pytest.param('use_gpu:0,do_delay:0,do_fic:1', {'bold': 0.7420394211841863, 'fc': 0.15228818488966733, 'fcd': 0.053617605632309974}),
        pytest.param('use_gpu:0,do_delay:0,do_fic:0', {}, marks=pytest.mark.skip(reason="bug causing segmentation fault")),
        pytest.param('use_gpu:0,do_delay:1,do_fic:1', {}, marks=pytest.mark.skip(reason="not implemented")),
        pytest.param('use_gpu:0,do_delay:1,do_fic:0', {}, marks=pytest.mark.skip(reason="not implemented")),
    ]) # the expected values based on tests run on gpu1.htc.inm7.de using commit 70f164667a275bca75f373234344b96f495aa781
def test_single_sim(opts, expected):
    """
    Tests if BOLD, FC and FCD of a specific simulation
    matches pre-calculated values (based on the same code)
    Note that the values used here are not ground truth (as
    there is no ground truth), and this is simply a test
    of consistency of calculations throughout different versions
    of the code and platforms

    Parameters
    -------
    opts: (str)
        with format 'use_gpu:*,do_delay:*,do_fic:*'
    expected: (dict)
        with keys 'bold' (expected bold[500]), 'fc' (expected fc[30]), 'fcd' (expected fcd[30])
    """
    opts = {i.split(':')[0]:int(i.split(':')[1]) for i in opts.split(',')}
    N_SIMS = 1
    G_list = np.repeat(0.5, N_SIMS)
    w_EE_list = np.repeat(0.21, NODES*N_SIMS)
    w_EI_list = np.repeat(0.15, NODES*N_SIMS)
    if opts['do_fic']:
        w_IE_list = np.repeat(0.0, NODES*N_SIMS)
    else:
        w_IE_list = np.repeat(1.0, NODES*N_SIMS)
    if opts['do_delay']:
        v_list = np.linspace(0.5, 4.0, N_SIMS)
        SC_dist = datasets.load_sc('length', 'schaefer-100').flatten()
        os.environ['BNM_SYNC_MSEC'] = '1'
    else:
        v_list = np.repeat(0.0, N_SIMS) # doesn't matter what it is!
        SC_dist = np.zeros(NODES*NODES, dtype=float) # doesn't matter what it is!
    out = run_simulations(
        SC, SC_dist, G_list, w_EE_list, w_EI_list, w_IE_list, v_list,
        opts['do_fic'], EXTENDED_OUTPUT, opts['do_delay'], FORCE_REINIT, (not opts['use_gpu']),
        N_SIMS, NODES, TIME_STEPS, BOLD_TR,
        WINDOW_SIZE, WINDOW_STEP, RAND_SEED
    )
    sim_bolds , sim_fc_trils, sim_fcd_trils = out[:3]
    if opts['do_delay']:
        del os.environ['BNM_SYNC_MSEC'] # remove it so it's not done for next tests
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
        pytest.param('use_gpu:0,do_delay:0,do_fic:1'),
        pytest.param('use_gpu:0,do_delay:0,do_fic:0', marks=pytest.mark.skip(reason="there is a bug causing segmentation fault")),
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
    if opts['do_delay']:
        v_list = np.repeat(1.0, N_SIMS)
        SC_dist = datasets.load_sc('length', 'schaefer-100').flatten()
        os.environ['BNM_SYNC_MSEC'] = '1'
    else:
        v_list = np.repeat(0.0, N_SIMS) # doesn't matter what it is!
        SC_dist = np.zeros(NODES*NODES, dtype=float) # doesn't matter what it is!
    out = run_simulations(
        SC, SC_dist, G_list, w_EE_list, w_EI_list, w_IE_list, v_list,
        opts['do_fic'], EXTENDED_OUTPUT, opts['do_delay'], FORCE_REINIT, (not opts['use_gpu']),
        N_SIMS, NODES, TIME_STEPS, BOLD_TR,
        WINDOW_SIZE, WINDOW_STEP, RAND_SEED
    )
    sim_bolds , sim_fc_trils, sim_fcd_trils = out[:3]
    if opts['do_delay']:
        del os.environ['BNM_SYNC_MSEC'] # remove it so it's not done for next tests

    assert (sim_bolds[0] == sim_bolds[1]).all() # consider using np.isclose
    assert (sim_fc_trils[0] == sim_fc_trils[1]).all()
    assert (sim_fcd_trils[0] == sim_fcd_trils[1]).all()

@pytest.mark.skipif(no_gpu(), reason="No GPU available")
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
        do_fic, EXTENDED_OUTPUT, do_delay, FORCE_REINIT, force_cpu,
        N_SIMS, NODES, TIME_STEPS, BOLD_TR,
        WINDOW_SIZE, WINDOW_STEP, RAND_SEED
    )
    cpu_sim_bolds , cpu_sim_fc_trils, cpu_sim_fcd_trils = out[:3]

    # then run on GPU
    force_cpu = False
    out = run_simulations(
        SC, SC_dist, G_list, w_EE_list, w_EI_list, w_IE_list, v_list,
        do_fic, EXTENDED_OUTPUT, do_delay, FORCE_REINIT, force_cpu,
        N_SIMS, NODES, TIME_STEPS, BOLD_TR,
        WINDOW_SIZE, WINDOW_STEP, RAND_SEED
    )
    gpu_sim_bolds , gpu_sim_fc_trils, gpu_sim_fcd_trils = out[:3]
    assert np.isclose(cpu_sim_bolds[0], gpu_sim_bolds[0], atol=1e-12).all()
    # assert (cpu_sim_bolds[0]==gpu_sim_bolds[0]).all() # this might fail especially after running the previous tests
    assert np.isclose(cpu_sim_fc_trils[0], gpu_sim_fc_trils[0], atol=1e-12).all()
    assert np.isclose(cpu_sim_fcd_trils[0], gpu_sim_fcd_trils[0], atol=1e-12).all()