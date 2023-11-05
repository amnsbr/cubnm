import numpy as np
import bnm 
import os

def test(N_SIMS=2):
    # run identical simulations and check if BOLD is the same
    nodes = 100
    # time_steps = 60000
    time_steps = 450000
    # BOLD_TR = 1000
    BOLD_TR = 3000
    window_size = 10
    window_step = 2
    rand_seed = 410
    extended_output = True

    np.random.seed(0)

    SC = np.loadtxt('/data/project/ei_development/tools/cuBNM/sample_input/ctx_parc-schaefer-100_approach-median_mean001_desc-strength.txt').flatten()
    # SC = np.random.randn(nodes*nodes)
    SC_dist = np.loadtxt('/data/project/ei_development/tools/cuBNM/sample_input/ctx_parc-schaefer-100_approach-median_mean001_desc-length.txt').flatten()
    G_list = np.array([0.5]*N_SIMS)
    w_EE_list = np.repeat(0.21, nodes*N_SIMS)
    w_EI_list = np.repeat(0.15, nodes*N_SIMS)
    w_IE_list = np.repeat(0.0, nodes*N_SIMS)
    do_fic = True
    # w_IE_list = np.repeat(1.0, nodes*N_SIMS)
    # do_fic = False
    v_list = np.linspace(0.5, 4, N_SIMS)
    # v_list = np.repeat(-1.0, N_SIMS)
    has_delay = not(-1 in v_list)
    if has_delay:
        # with delay it is recommended to do
        # the syncing of nodes every 1 msec instead
        # of every 0.1 msec. Otherwise it'll be very slow
        os.environ['BNM_SYNC_MSEC'] = '1'
        # TODO: add a function to do bnm.set_conf('sync_msec', True)

    sim_bolds, sim_fc_trils, sim_fcd_trils = bnm.run_simulations(
        SC, SC_dist, G_list, w_EE_list, w_EI_list, w_IE_list, v_list,
        do_fic, extended_output, N_SIMS, nodes, time_steps, BOLD_TR,
        window_size, window_step, rand_seed
    )

    for sim_idx in range(N_SIMS):
        print(f"BOLD Python {sim_idx}: shape {sim_bolds.shape}, idx 500 {sim_bolds[sim_idx, 500]}")
        print(f"fc_trils Python {sim_idx}: shape {sim_fc_trils.shape}, idx 30 {sim_fc_trils[sim_idx, 30]}")
        print(f"fcd_trils Python {sim_idx}: shape {sim_fcd_trils.shape}, idx 30 {sim_fcd_trils[sim_idx, 30]}")

if __name__ == '__main__':
    test(2)