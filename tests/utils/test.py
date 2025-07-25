"""
Testing utility functions
"""
import pytest
import numpy as np
import scipy.stats
from cubnm import sim, utils

def no_gpu():
    # to skip GPU-dependent tests
    return utils.avail_gpus()==0

@pytest.mark.skipif(no_gpu(), reason="No GPU available")
def test_fc_corr_device():
    """
    Tests calculation of simulated and empirical FC correlation
    on GPU in comparison to scipy results on CPU
    """
    # TODO: also test using a larger number of simulations to test batch processing
    # generate random FC trils uniformly distributed between -1 and 1
    np.random.seed(0)
    n_simulations = 100
    n_pairs = (100 * 99) // 2
    sim_fc_trils = np.random.rand(n_simulations, n_pairs) * 2 - 1
    emp_fc_tril = np.random.rand(n_pairs) * 2 - 1
    # calculate FC correlation on CPU
    cpu_res = []
    for sim_idx in range(n_simulations):
        cpu_res.append(scipy.stats.pearsonr(sim_fc_trils[sim_idx], emp_fc_tril).statistic)
    cpu_res = np.array(cpu_res)
    # calculate FC correlation on GPU
    gpu_res = utils.fc_corr_device(sim_fc_trils, emp_fc_tril)
    # calculate FC correlation on limited GPU memory (1 MiB) to test batch processing
    gpu_res_batch = utils.fc_corr_device(sim_fc_trils, emp_fc_tril, usable_mem=1024**2)
    # compare results
    assert np.isclose(cpu_res, gpu_res).all()
    assert np.isclose(cpu_res, gpu_res_batch).all()

@pytest.mark.skipif(no_gpu(), reason="No GPU available")
def test_fcd_ks_device():
    """
    Tests calculation of simulated and empirical FCD KS statistic
    on GPU in comparison to scipy results on CPU
    """
    # generate random FCD trils uniformly distributed between -1 and 1
    np.random.seed(0)
    n_simulations = 1000
    # using different number of elements for simulated and empirical FCD trils
    m1 = 300
    m2 = 500
    sim_fcd_trils = np.random.rand(n_simulations, m1) * 2 - 1
    emp_fcd_tril = np.random.rand(m2) * 2 - 1
    # calculate FCD KS statistic on CPU
    cpu_res = []
    for sim_idx in range(n_simulations):
        cpu_res.append(scipy.stats.ks_2samp(sim_fcd_trils[sim_idx], emp_fcd_tril).statistic)
    cpu_res = np.array(cpu_res)
    # calculate FCD KS statistic on GPU with all available GPU memory
    gpu_res = utils.fcd_ks_device(sim_fcd_trils, emp_fcd_tril)
    # calculate FCD KS statistic on limited GPU memory (1 MiB) to test batch processing
    gpu_res_batch = utils.fcd_ks_device(sim_fcd_trils, emp_fcd_tril, usable_mem=1024**2)
    # compare results
    assert np.isclose(cpu_res, gpu_res).all()
    assert np.isclose(cpu_res, gpu_res_batch).all()
    