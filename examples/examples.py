import numpy as np
import cuBNM
from cuBNM._core import run_simulations, set_conf, set_const
from cuBNM import optimize, sim, utils, datasets
from cuBNM._setup_flags import many_nodes_flag, gpu_enabled_flag

import os
from pymoo.algorithms.soo.nonconvex.cmaes import CMAES
from pymoo.optimize import minimize
from pymoo.termination.default import DefaultSingleObjectiveTermination
from pymoo.core.termination import Termination
from pymoo.termination import get_termination
import cma

def run_sims(N_SIMS=2, v=0.5, force_cpu=False, rand_seed=410, force_reinit=False):
    # run identical simulations and check if BOLD is the same
    nodes = 100
    time_steps = 60000
    # time_steps = 450000
    BOLD_TR = 1000
    # BOLD_TR = 3000
    window_size = 10
    window_step = 2
    extended_output = True

    np.random.seed(0)

    SC = datasets.load_sc('strength', 'schaefer-100').flatten()
    SC_dist = datasets.load_sc('length', 'schaefer-100').flatten()
    G_list = np.repeat(0.5, N_SIMS)
    w_EE_list = np.repeat(0.21, nodes*N_SIMS)
    w_EI_list = np.repeat(0.15, nodes*N_SIMS)
    w_IE_list = np.repeat(0.0, nodes*N_SIMS)
    do_fic = True
    # w_IE_list = np.repeat(1.0, nodes*N_SIMS)
    # do_fic = False

    do_delay = False
    # do_delay = True
    if do_delay:
        # v_list = np.linspace(0.5, 4.0, N_SIMS)
        v_list = np.repeat(v, N_SIMS)
        # v_list = np.repeat(-1.0, N_SIMS)

        # with delay it is recommended to do
        # the syncing of nodes every 1 msec instead
        # of every 0.1 msec. Otherwise it'll be very slow
        set_conf('sync_msec', True)
        # TODO: add a function to do bnm.set_conf('sync_msec', True)
    else:
        v_list = np.repeat(0.0, N_SIMS) # doesn't matter what it is!
        SC_dist = np.zeros(nodes*nodes, dtype=float) # doesn't matter what it is!
    force_cpu = force_cpu | (not gpu_enabled_flag) | (utils.avail_gpus()==0)
    # make sure all the input arrays are of type float/double
    out = run_simulations(
        SC, SC_dist, G_list, w_EE_list, w_EI_list, w_IE_list, v_list,
        do_fic, extended_output, do_delay, force_reinit, force_cpu,
        N_SIMS, nodes, time_steps, BOLD_TR,
        window_size, window_step, rand_seed
    )
    sim_bolds , sim_fc_trils, sim_fcd_trils = out[:3]

    for sim_idx in range(N_SIMS):
        print(f"BOLD Python {sim_idx}: shape {sim_bolds.shape}, idx 500 {sim_bolds[sim_idx, 500]}")
        print(f"fc_trils Python {sim_idx}: shape {sim_fc_trils.shape}, idx 30 {sim_fc_trils[sim_idx, 30]}")
        print(f"fcd_trils Python {sim_idx}: shape {sim_fcd_trils.shape}, idx 30 {sim_fcd_trils[sim_idx, 30]}")

def run_grid():
    gs = optimize.GridSearch(
        params = {
            'G': (0.5, 2.5, 4),
            'wEE': 0.15,
            'wEI': 0.21
        },
        # duration = 450,
        # TR = 3,
        duration = 60,
        TR = 1,
        sc_path = datasets.load_sc('strength', 'schaefer-100', return_path=True)
    )
    emp_fc_tril = datasets.load_functional('FC', 'schaefer-100', exc_interhemispheric=True)
    emp_fcd_tril = datasets.load_functional('FCD', 'schaefer-100', exc_interhemispheric=True)
    scores = gs.evaluate(emp_fc_tril, emp_fcd_tril)
    gs.sim_group.save()
    return gs, scores

def run_grid_no_fic():
    gs = optimize.GridSearch(
        params = {
            'G': 0.5,
            'wEE': (0.05, 1, 2),
            'wEI': (0.07, 0.75, 2),
            'wIE': (1.5, 0.75, 2),
        },
        duration = 60,
        TR = 1,
        sc_path = datasets.load_sc('strength', 'schaefer-100', return_path=True),
        do_fic = False
    )
    emp_fc_tril = datasets.load_functional('FC', 'schaefer-100', exc_interhemispheric=True)
    emp_fcd_tril = datasets.load_functional('FCD', 'schaefer-100', exc_interhemispheric=True)
    scores = gs.evaluate(emp_fc_tril, emp_fcd_tril)
    return gs, scores

def run_grid_delay():
    gs = optimize.GridSearch(
        params = {
            'G': 0.5,
            'wEE': (0.05, 1, 2),
            'wEI': (0.07, 0.75, 2),
            'v': (1, 5, 2)
        },
        duration = 60,
        TR = 1,
        sc_path = datasets.load_sc('strength', 'schaefer-100', return_path=True),
        sc_dist_path = datasets.load_sc('length', 'schaefer-100', return_path=True),
    )
    emp_fc_tril = datasets.load_functional('FC', 'schaefer-100', exc_interhemispheric=True)
    emp_fcd_tril = datasets.load_functional('FCD', 'schaefer-100', exc_interhemispheric=True)
    scores = gs.evaluate(emp_fc_tril, emp_fcd_tril)
    return gs, scores

def run_problem():
    emp_fc_tril = datasets.load_functional('FC', 'schaefer-100', exc_interhemispheric=True)
    emp_fcd_tril = datasets.load_functional('FCD', 'schaefer-100', exc_interhemispheric=True)
    problem = optimize.RWWProblem(
        params = {
            'G': (1.0, 3.0),
            'wEE': (0.05, 0.5),
            'wEI': 0.15,
        },
        emp_fc_tril = emp_fc_tril,
        emp_fcd_tril = emp_fcd_tril,
        duration = 60,
        TR = 1,
        sc_path = datasets.load_sc('strength', 'schaefer-100', return_path=True),
    )
    # assume that optimizer is using 10 particles
    problem.sim_group.N = 10
    # synthesize example X with dims (particles, free_params)
    X = np.vstack([np.linspace(1.0, 3.0, 10), np.linspace(0.05, 0.5, 10)]).T
    # synthesize out dictionary
    out = {'F': None, 'G': None}
    problem._evaluate(X, out)
    return problem, out

def run_cmaes_optimizer():
    emp_fc_tril = datasets.load_functional('FC', 'schaefer-100', exc_interhemispheric=True)
    emp_fcd_tril = datasets.load_functional('FCD', 'schaefer-100', exc_interhemispheric=True)
    problem = optimize.RWWProblem(
        params = {
            'G': (1.0, 3.0),
            'wEE': (0.05, 0.5),
            'wEI': 0.15,
        },
        emp_fc_tril = emp_fc_tril,
        emp_fcd_tril = emp_fcd_tril,
        duration = 60,
        TR = 1,
        sc_path = datasets.load_sc('strength', 'schaefer-100', return_path=True),
    )
    cmaes = optimize.CMAESOptimizer(popsize=10, n_iter=2)
    cmaes.setup_problem(problem, seed=1)
    cmaes.optimize()
    return cmaes

def run_bayes_optimizer():
    emp_fc_tril = datasets.load_functional('FC', 'schaefer-100', exc_interhemispheric=True)
    emp_fcd_tril = datasets.load_functional('FCD', 'schaefer-100', exc_interhemispheric=True)
    problem = optimize.RWWProblem(
        params = {
            'G': (1.0, 3.0),
            'wEE': (0.05, 0.5),
            'wEI': 0.15,
        },
        emp_fc_tril = emp_fc_tril,
        emp_fcd_tril = emp_fcd_tril,
        duration = 60,
        TR = 1,
        sc_path = datasets.load_sc('strength', 'schaefer-100', return_path=True),
    )
    bo = optimize.BayesOptimizer(popsize=10, n_iter=7)
    bo.setup_problem(problem, seed=1)
    bo.optimize()
    return bo

def run_cmaes_optimizer_het(force_cpu=False, use_bound_penalty=False):
    emp_fc_tril = datasets.load_functional('FC', 'schaefer-100', exc_interhemispheric=True)
    emp_fcd_tril = datasets.load_functional('FCD', 'schaefer-100', exc_interhemispheric=True)
    problem = optimize.RWWProblem(
        params = {
            'G': (1.0, 3.0),
            'wEE': (0.05, 0.5),
            'wEI': 0.15,
            'v': (0.5, 8.0)
        },
        emp_fc_tril = emp_fc_tril,
        emp_fcd_tril = emp_fcd_tril,
        het_params = ['wEE', 'wEI'],
        maps_path = datasets.load_maps('6maps', 'schaefer-100', norm='minmax', return_path=True),
        duration = 60,
        TR = 1,
        sc_path = datasets.load_sc('strength', 'schaefer-100', return_path=True),
        sc_dist_path = datasets.load_sc('length', 'schaefer-100', return_path=True),
        force_cpu = force_cpu 
    )
    cmaes = optimize.CMAESOptimizer(popsize=10, n_iter=2, seed=1, 
                                    use_bound_penalty=use_bound_penalty,
                                    algorithm_kws=dict(tolfun=5e-3))
    cmaes.setup_problem(problem)
    cmaes.optimize()
    cmaes.save()
    return cmaes

def run_cmaes_optimizer_het_nofic():
    emp_fc_tril = datasets.load_functional('FC', 'schaefer-100', exc_interhemispheric=True)
    emp_fcd_tril = datasets.load_functional('FCD', 'schaefer-100', exc_interhemispheric=True)
    problem = optimize.RWWProblem(
        params = {
            'G': (1.0, 3.0),
            'wEE': (0.05, 0.5),
            'wEI': 0.15,
            'wIE': (1.0, 4.0)
        },
        emp_fc_tril = emp_fc_tril,
        emp_fcd_tril = emp_fcd_tril,
        het_params = ['wEE', 'wIE'],
        maps_path = datasets.load_maps('6maps', 'schaefer-100', norm='minmax', return_path=True),
        duration = 60,
        TR = 1,
        sc_path = datasets.load_sc('strength', 'schaefer-100', return_path=True),
        do_fic = False
    )
    cmaes = optimize.CMAESOptimizer(popsize=10, n_iter=2, seed=1)
    cmaes.setup_problem(problem)
    cmaes.optimize()
    cmaes.save()
    return cmaes

def run_cmaes_optimizer_regional(node_grouping='sym'):
    if node_grouping == 'yeo':
        node_grouping = datasets.load_maps('yeo7', 'schaefer-100', norm=None, return_path=True)
    emp_fc_tril = datasets.load_functional('FC', 'schaefer-100', exc_interhemispheric=True)
    emp_fcd_tril = datasets.load_functional('FCD', 'schaefer-100', exc_interhemispheric=True)
    problem = optimize.RWWProblem(
        params = {
            'G': (1.0, 3.0),
            'wEE': (0.05, 0.5),
            'wEI': (0.05, 0.75),
        },
        emp_fc_tril = emp_fc_tril,
        emp_fcd_tril = emp_fcd_tril,
        het_params = ['wEE', 'wEI'],
        node_grouping = node_grouping,
        duration = 60,
        TR = 1,
        sc_path = datasets.load_sc('strength', 'schaefer-100', return_path=True),
    )
    cmaes = optimize.CMAESOptimizer(popsize=10, n_iter=2, seed=1)
    cmaes.setup_problem(problem)
    cmaes.optimize()
    cmaes.save()
    return cmaes

def run_nsga2_optimizer_het(force_cpu=False):
    emp_fc_tril = datasets.load_functional('FC', 'schaefer-100', exc_interhemispheric=True)
    emp_fcd_tril = datasets.load_functional('FCD', 'schaefer-100', exc_interhemispheric=True)
    problem = optimize.RWWProblem(
        params = {
            'G': (1.0, 3.0),
            'wEE': (0.05, 0.5),
            'wEI': 0.15,
            'v': (0.5, 8.0)
        },
        emp_fc_tril = emp_fc_tril,
        emp_fcd_tril = emp_fcd_tril,
        het_params = ['wEE', 'wEI'],
        maps_path = datasets.load_maps('6maps', 'schaefer-100', norm='minmax', return_path=True),
        duration = 60,
        TR = 1,
        sc_path = datasets.load_sc('strength', 'schaefer-100', return_path=True),
        sc_dist_path = datasets.load_sc('length', 'schaefer-100', return_path=True),
        force_cpu = force_cpu,
        gof_terms = ['-fc_normec', '-fcd_ks'],
        multiobj = True
    )
    optimizer = optimize.NSGA2Optimizer(popsize=10, n_iter=3, seed=1)
    optimizer.setup_problem(problem)
    optimizer.optimize()
    optimizer.save()
    return optimizer

if __name__ == '__main__':
    run_sims()
    # gs, scores = run_grid()
    # problem, out = run_problem()
    # cmaes = run_cmaes_optimizer_het()
    # run_grid_many_nodes()
    # nsga2 = run_nsga2_optimizer_het()