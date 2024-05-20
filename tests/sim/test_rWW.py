"""
Testing consistency of rWW simulations
"""
# TODO: consider testing all models in a single test file
import pytest
from cubnm import sim
from cubnm import datasets
from cubnm.utils import avail_gpus
import numpy as np
import os
import pickle
import gzip

test_data_dir = os.path.join(os.path.dirname(__file__), '..', 'expected', 'sim')
sel_state_var = 'r_E'

def no_gpu():
    # to skip GPU-dependent tests
    return avail_gpus()==0

@pytest.mark.parametrize(
    "opts", 
    [
        pytest.param('force_cpu:0,do_delay:0,do_fic:1', marks=pytest.mark.skipif(no_gpu(), reason="No GPU available")),
        pytest.param('force_cpu:0,do_delay:0,do_fic:0', marks=pytest.mark.skipif(no_gpu(), reason="No GPU available")),
        pytest.param('force_cpu:0,do_delay:1,do_fic:1', marks=pytest.mark.skipif(no_gpu(), reason="No GPU available")),
        pytest.param('force_cpu:0,do_delay:1,do_fic:0', marks=pytest.mark.skipif(no_gpu(), reason="No GPU available")),
        pytest.param('force_cpu:1,do_delay:0,do_fic:1'),
        pytest.param('force_cpu:1,do_delay:0,do_fic:0'),
        pytest.param('force_cpu:1,do_delay:1,do_fic:1'),
        pytest.param('force_cpu:1,do_delay:1,do_fic:0'),
    ])
def test_single_sim(opts):
    """
    Tests if BOLD, FC and FCD and r_E of a specific simulation
    matches pre-calculated results (based on the same code)
    Note that the expected results are not ground truth (as
    there is no ground truth), and this is simply a test
    of consistency of calculations throughout different versions
    of the code and platforms.

    Parameters
    -------
    opts: (str)
        with format 'force_cpu:*,do_delay:*,do_fic:*'
    """
    # These tests are going to fail if code is compiled with alternative
    # toolchains, and therefore are only expected to work with binary
    # wheels or source-compiled library on GCC 12. This is likely due
    # to differences in the random arrays (TODO: test it)
    # TODO: skip tests if binaries are built differently
    # load expected
    with gzip.open(os.path.join(test_data_dir, 'rWW.pkl.gz'), 'rb') as f:
        expected = pickle.load(f)[opts]
    # run simulation
    opts = {i.split(':')[0]:int(i.split(':')[1]) for i in opts.split(',')}
    if opts['do_delay']:
        sc_dist_path = datasets.load_sc('length', 'schaefer-100', return_path=True)
    else:
        sc_dist_path = None
    sg = sim.rWWSimGroup(
        duration=60,
        TR=1,
        sc_path=datasets.load_sc('strength', 'schaefer-100', return_path=True),
        sc_dist_path=sc_dist_path,
        sim_verbose=False,
        force_cpu=opts['force_cpu'],
        do_fic=opts['do_fic'],
    )
    sg.N = 1
    sg._set_default_params()
    sg.run()
    # compare results and expected
    assert np.isclose(sg.sim_bold, expected['sim_bold'], atol=1e-12).all()
    assert np.isclose(sg.sim_fc_trils, expected['sim_fc_trils'], atol=1e-12).all()
    assert np.isclose(sg.sim_fcd_trils, expected['sim_fcd_trils'], atol=1e-12).all()
    assert np.isclose(sg.sim_states[sel_state_var], expected['sim_sel_state'], atol=1e-12).all()

@pytest.mark.parametrize(
    "opts", 
    [
        pytest.param('force_cpu:0,do_delay:0,do_fic:1', marks=pytest.mark.skipif(no_gpu(), reason="No GPU available")),
        pytest.param('force_cpu:0,do_delay:0,do_fic:0', marks=pytest.mark.skipif(no_gpu(), reason="No GPU available")),
        pytest.param('force_cpu:0,do_delay:1,do_fic:1', marks=pytest.mark.skipif(no_gpu(), reason="No GPU available")),
        pytest.param('force_cpu:0,do_delay:1,do_fic:0', marks=pytest.mark.skipif(no_gpu(), reason="No GPU available")),
        pytest.param('force_cpu:1,do_delay:0,do_fic:1'),
        pytest.param('force_cpu:1,do_delay:0,do_fic:0'),
        pytest.param('force_cpu:1,do_delay:1,do_fic:1'),
        pytest.param('force_cpu:1,do_delay:1,do_fic:0'),
    ])
def test_identical_sims(opts):
    """
    Tests if two identical simulations run on parallel (GPU/multi-CPU)
    or serially (single thread) result in identical BOLD, FC and FCD,
    and r_E.

    Parameters
    -------
    opts: (str)
        with format 'force_cpu:*,do_delay:*,do_fic:*'
    """
    opts = {i.split(':')[0]:int(i.split(':')[1]) for i in opts.split(',')}
    if opts['do_delay']:
        sc_dist_path = datasets.load_sc('length', 'schaefer-100', return_path=True)
    else:
        sc_dist_path = None
    sg = sim.rWWSimGroup(
        duration=60,
        TR=1,
        sc_path=datasets.load_sc('strength', 'schaefer-100', return_path=True),
        sc_dist_path=sc_dist_path,
        sim_verbose=False,
        force_cpu=opts['force_cpu'],
        do_fic=opts['do_fic'],
    )
    sg.N = 2
    sg._set_default_params()
    sg.run()
    # compare results
    assert np.isclose(sg.sim_bold[0], sg.sim_bold[1], atol=1e-12).all()
    assert np.isclose(sg.sim_fc_trils[0], sg.sim_fc_trils[1], atol=1e-12).all()
    assert np.isclose(sg.sim_fcd_trils[0], sg.sim_fcd_trils[1], atol=1e-12).all()
    assert np.isclose(sg.sim_states[sel_state_var][0], sg.sim_states[sel_state_var][1], atol=1e-12).all()

@pytest.mark.skipif(no_gpu(), reason="No GPU available")
@pytest.mark.parametrize(
    "opts", 
    [
        pytest.param('do_delay:0,do_fic:1'),
        pytest.param('do_delay:0,do_fic:0'),
        pytest.param('do_delay:1,do_fic:1'),
        pytest.param('do_delay:1,do_fic:0'),
    ])
def test_identical_cpu_gpu(opts):
    """
    Tests if two identical simulations run on CPU vs GPU
    result in identical BOLD, FC, FCD and r_E

    Parameters
    -------
    opts: (str)
        with format 'force_cpu:*,do_delay:*,do_fic:*'
    """
    opts = {i.split(':')[0]:int(i.split(':')[1]) for i in opts.split(',')}
    sc_dist_path = None
    sgs = {}
    for force_cpu in [True, False]:
        sg = sim.rWWSimGroup(
            duration=60,
            TR=1,
            sc_path=datasets.load_sc('strength', 'schaefer-100', return_path=True),
            sc_dist_path=sc_dist_path,
            sim_verbose=False,
            force_cpu=force_cpu,
            do_fic=opts['do_fic'],
        )
        sg.N = 1
        sg._set_default_params()
        sg.run()
        sgs[force_cpu] = sg
    # compare results
    assert np.isclose(sgs[False].sim_bold, sgs[True].sim_bold, atol=1e-12).all()
    assert np.isclose(sgs[False].sim_fc_trils, sgs[True].sim_fc_trils, atol=1e-12).all()
    assert np.isclose(sgs[False].sim_fcd_trils, sgs[True].sim_fcd_trils, atol=1e-12).all()
    assert np.isclose(sgs[False].sim_states[sel_state_var], sgs[True].sim_states[sel_state_var], atol=1e-12).all()