import numpy as np
import bnm

N_SIMS = 1
nodes = 100
time_steps = 60000
BOLD_TR = 1000
window_size = 10
window_step = 2
rand_seed = 410

np.random.seed(0)

SC = np.random.randn(nodes*nodes)
G_list = np.array([1.5])
w_EE_list = np.repeat(0.21, nodes)
w_EI_list = np.repeat(0.15, nodes)
w_IE_list = np.repeat(0.0, nodes)
do_fic = True

sim_bold = bnm.run_simulations(
    SC, G_list, w_EE_list, w_EI_list, w_IE_list,
    do_fic, N_SIMS, nodes, time_steps, BOLD_TR,
    window_size, window_step, rand_seed
)
print("BOLD Python:", sim_bold[0, 500])