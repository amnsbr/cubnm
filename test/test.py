import numpy as np
import cuBNM
from cuBNM.core import run_simulations
from cuBNM import optimize, sim
from cuBNM._flags import many_nodes_flag, gpu_enabled_flag

import os
from pymoo.algorithms.soo.nonconvex.cmaes import CMAES
from pymoo.optimize import minimize
from pymoo.termination.default import DefaultSingleObjectiveTermination
from pymoo.core.termination import Termination
from pymoo.termination import get_termination
import cma

def run_sims(N_SIMS=1, v=0.5, force_cpu=False):
    # os.environ['BNM_MAX_FIC_TRIALS_CMAES'] = '0'
    # run identical simulations and check if BOLD is the same
    nodes = 100
    time_steps = 60000
    # time_steps = 450000
    BOLD_TR = 1000
    # BOLD_TR = 3000
    window_size = 10
    window_step = 2
    rand_seed = 410
    extended_output = True
    force_reinit = False

    np.random.seed(0)

    SC = np.loadtxt('/data/project/ei_development/tools/cuBNM/sample_input/ctx_parc-schaefer-100_approach-median_mean001_desc-strength.txt').flatten()
    # SC = np.random.randn(nodes*nodes) => This will lead to unstable FIC error
    SC_dist = np.loadtxt('/data/project/ei_development/tools/cuBNM/sample_input/ctx_parc-schaefer-100_approach-median_mean001_desc-length.txt').flatten()
    G_list = np.repeat(0.5, N_SIMS)
    w_EE_list = np.repeat(0.21, nodes*N_SIMS)
    w_EI_list = np.repeat(0.15, nodes*N_SIMS)
    w_IE_list = np.repeat(0.0, nodes*N_SIMS)
    do_fic = True
    # w_IE_list = np.repeat(1.0, nodes*N_SIMS)
    # do_fic = False

    do_delay = False
    if do_delay:
        v_list = np.linspace(0.5, 4.0, N_SIMS)
        # v_list = np.repeat(v, N_SIMS)
        # v_list = np.repeat(-1.0, N_SIMS)

        # with delay it is recommended to do
        # the syncing of nodes every 1 msec instead
        # of every 0.1 msec. Otherwise it'll be very slow
        os.environ['BNM_SYNC_MSEC'] = '1'
        # TODO: add a function to do bnm.set_conf('sync_msec', True)
    else:
        v_list = np.repeat(0.0, N_SIMS) # doesn't matter what it is!
        SC_dist = np.zeros(nodes*nodes, dtype=float) # doesn't matter what it is!
    # make sure all the input arrays are of type float/double
    out = run_simulations(
        SC, SC_dist, G_list, w_EE_list, w_EI_list, w_IE_list, v_list,
        do_fic, extended_output, do_delay, force_reinit, ((not gpu_enabled_flag) | force_cpu),
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
        sc_path = '/data/project/ei_development/tools/cuBNM/sample_input/ctx_parc-schaefer-100_approach-median_mean001_desc-strength.txt'
    )
    emp_fc_tril = np.loadtxt('/data/project/ei_development/tools/cuBNM/sample_input/ctx_parc-schaefer-100_hemi-LR_exc-inter_desc-FCtril.txt')
    emp_fcd_tril = np.loadtxt('/data/project/ei_development/tools/cuBNM/sample_input/ctx_parc-schaefer-100_hemi-LR_exc-inter_desc-FCDtril.txt')
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
        sc_path = '/data/project/ei_development/tools/cuBNM/sample_input/ctx_parc-schaefer-100_approach-median_mean001_desc-strength.txt',
        do_fic = False
    )
    emp_fc_tril = np.loadtxt('/data/project/ei_development/tools/cuBNM/sample_input/ctx_parc-schaefer-100_hemi-LR_exc-inter_desc-FCtril.txt')
    emp_fcd_tril = np.loadtxt('/data/project/ei_development/tools/cuBNM/sample_input/ctx_parc-schaefer-100_hemi-LR_exc-inter_desc-FCDtril.txt')
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
        sc_path = '/data/project/ei_development/tools/cuBNM/sample_input/ctx_parc-schaefer-100_approach-median_mean001_desc-strength.txt',
        sc_dist_path = '/data/project/ei_development/tools/cuBNM/sample_input/ctx_parc-schaefer-100_approach-median_mean001_desc-length.txt',
    )
    emp_fc_tril = np.loadtxt('/data/project/ei_development/tools/cuBNM/sample_input/ctx_parc-schaefer-100_hemi-LR_exc-inter_desc-FCtril.txt')
    emp_fcd_tril = np.loadtxt('/data/project/ei_development/tools/cuBNM/sample_input/ctx_parc-schaefer-100_hemi-LR_exc-inter_desc-FCDtril.txt')
    scores = gs.evaluate(emp_fc_tril, emp_fcd_tril)
    return gs, scores

def test_problem():
    emp_fc_tril = np.loadtxt('/data/project/ei_development/tools/cuBNM/sample_input/ctx_parc-schaefer-100_hemi-LR_exc-inter_desc-FCtril.txt')
    emp_fcd_tril = np.loadtxt('/data/project/ei_development/tools/cuBNM/sample_input/ctx_parc-schaefer-100_hemi-LR_exc-inter_desc-FCDtril.txt')
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
        sc_path = '/data/project/ei_development/tools/cuBNM/sample_input/ctx_parc-schaefer-100_approach-median_mean001_desc-strength.txt',    
    )
    # assume that optimizer is using 10 particles
    problem.sim_group.N = 10
    # synthesize example X with dims (particles, free_params)
    X = np.vstack([np.linspace(1.0, 3.0, 10), np.linspace(0.05, 0.5, 10)]).T
    # synthesize out dictionary
    out = {'F': None, 'G': None}
    problem._evaluate(X, out)
    return problem, out

def test_cmaes_simple():
    emp_fc_tril = np.loadtxt('/data/project/ei_development/tools/cuBNM/sample_input/ctx_parc-schaefer-100_hemi-LR_exc-inter_desc-FCtril.txt')
    emp_fcd_tril = np.loadtxt('/data/project/ei_development/tools/cuBNM/sample_input/ctx_parc-schaefer-100_hemi-LR_exc-inter_desc-FCDtril.txt')
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
        sc_path = '/data/project/ei_development/tools/cuBNM/sample_input/ctx_parc-schaefer-100_approach-median_mean001_desc-strength.txt',    
    )
    # bound_penalty = cma.constraints_handler.BoundPenalty([0, 1])
    algorithm = CMAES(
        x0=np.array([0.5, 0.5]), # initial guess in the middle; note that X is 0-1 normalized
        sigma=0.5,
        maxiter=1,
        popsize=20, 
        eval_initial_x=False,
    )
    termination = get_termination('n_iter', 3)
    res = minimize(problem,
                algorithm,
                termination,
                seed=1,
                verbose=True,
                save_history=True
                )

    return problem, algorithm, termination, res

def test_cmaes_ask_tell():
    emp_fc_tril = np.loadtxt('/data/project/ei_development/tools/cuBNM/sample_input/ctx_parc-schaefer-100_hemi-LR_exc-inter_desc-FCtril.txt')
    emp_fcd_tril = np.loadtxt('/data/project/ei_development/tools/cuBNM/sample_input/ctx_parc-schaefer-100_hemi-LR_exc-inter_desc-FCDtril.txt')
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
        sc_path = '/data/project/ei_development/tools/cuBNM/sample_input/ctx_parc-schaefer-100_approach-median_mean001_desc-strength.txt',    
    )
    # initialize algorithm with bound penalty
    bound_penalty = cma.constraints_handler.BoundPenalty([0, 1])
    algorithm = CMAES(
        # x0=np.array([0.5, 0.5]), # initial guess in the middle; note that X is 0-1 normalized
        x0=None, # will estimate the initial guess based on 20 random samples
        sigma=0.5,
        maxiter=1, # this is overwritten by the termination rules
        popsize=10, 
        eval_initial_x=False,
        BoundaryHandler=bound_penalty
    )
    # set up algorithm with the problem and termination rules
    termination = get_termination('n_iter', 2)
    algorithm.setup(problem, termination=termination, seed=1, verbose=True, save_history=True)
    if algorithm.options.get('BoundaryHandler') is not None:
        # the following is to avoid an error caused by pymoo interfering with cma
        # after problem is registered with the algorithm
        # the bounds will be enforced by bound_penalty
        algorithm.options['bounds'] = None 
    while algorithm.has_next():
        # ask the algorithm for the next solution to be evaluated
        pop = algorithm.ask()
        # evaluate the individuals using the algorithm's evaluator (necessary to count evaluations for termination)
        algorithm.evaluator.eval(problem, pop)
        # returned the evaluated individuals which have been evaluated or even modified
        algorithm.tell(infills=pop)
        # do same more things, printing, logging, storing or even modifying the algorithm object
        print(algorithm.n_gen, algorithm.evaluator.n_eval)

    return problem, algorithm, termination

def test_cmaes_optimizer():
    emp_fc_tril = np.loadtxt('/data/project/ei_development/tools/cuBNM/sample_input/ctx_parc-schaefer-100_hemi-LR_exc-inter_desc-FCtril.txt')
    emp_fcd_tril = np.loadtxt('/data/project/ei_development/tools/cuBNM/sample_input/ctx_parc-schaefer-100_hemi-LR_exc-inter_desc-FCDtril.txt')
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
        sc_path = '/data/project/ei_development/tools/cuBNM/sample_input/ctx_parc-schaefer-100_approach-median_mean001_desc-strength.txt',    
    )
    cmaes = optimize.CMAESOptimizer(popsize=10, n_iter=2)
    cmaes.setup_problem(problem, seed=1)
    cmaes.optimize()
    return cmaes

def test_bayes_optimizer():
    emp_fc_tril = np.loadtxt('/data/project/ei_development/tools/cuBNM/sample_input/ctx_parc-schaefer-100_hemi-LR_exc-inter_desc-FCtril.txt')
    emp_fcd_tril = np.loadtxt('/data/project/ei_development/tools/cuBNM/sample_input/ctx_parc-schaefer-100_hemi-LR_exc-inter_desc-FCDtril.txt')
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
        sc_path = '/data/project/ei_development/tools/cuBNM/sample_input/ctx_parc-schaefer-100_approach-median_mean001_desc-strength.txt',    
    )
    bo = optimize.BayesOptimizer(popsize=10, n_iter=7)
    bo.setup_problem(problem, seed=1)
    bo.optimize()
    return bo

def test_cmaes_optimizer_het():
    emp_fc_tril = np.loadtxt('/data/project/ei_development/tools/cuBNM/sample_input/ctx_parc-schaefer-100_hemi-LR_exc-inter_desc-FCtril.txt')
    emp_fcd_tril = np.loadtxt('/data/project/ei_development/tools/cuBNM/sample_input/ctx_parc-schaefer-100_hemi-LR_exc-inter_desc-FCDtril.txt')
    problem = optimize.RWWProblem(
        params = {
            'G': (1.0, 3.0),
            'wEE': (0.05, 0.5),
            'wEI': 0.15,
        },
        emp_fc_tril = emp_fc_tril,
        emp_fcd_tril = emp_fcd_tril,
        het_params = ['wEE', 'wEI'],
        maps_path = '/data/project/ei_development/tools/cuBNM/sample_input/ctx_parc-schaefer-100_desc-6maps_zscore.txt',
        # maps_path = '/data/project/ei_development/tools/cuBNM/sample_input/ctx_parc-schaefer-100_desc-6maps_minmax.txt',
        # duration = 60,
        # TR = 1,
        duration = 450,
        TR = 1,
        sc_path = '/data/project/ei_development/tools/cuBNM/sample_input/ctx_parc-schaefer-100_approach-median_mean001_desc-strength.txt',  
    )
    cmaes = optimize.CMAESOptimizer(popsize=10, n_iter=2)
    cmaes.setup_problem(problem, seed=1)
    cmaes.optimize()
    return cmaes

def test_cmaes_optimizer_het_nofic():
    emp_fc_tril = np.loadtxt('/data/project/ei_development/tools/cuBNM/sample_input/ctx_parc-schaefer-100_hemi-LR_exc-inter_desc-FCtril.txt')
    emp_fcd_tril = np.loadtxt('/data/project/ei_development/tools/cuBNM/sample_input/ctx_parc-schaefer-100_hemi-LR_exc-inter_desc-FCDtril.txt')
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
        maps_path = '/data/project/ei_development/tools/cuBNM/sample_input/ctx_parc-schaefer-100_desc-6maps_zscore.txt',
        # maps_path = '/data/project/ei_development/tools/cuBNM/sample_input/ctx_parc-schaefer-100_desc-6maps_minmax.txt',
        duration = 60,
        TR = 1,
        sc_path = '/data/project/ei_development/tools/cuBNM/sample_input/ctx_parc-schaefer-100_approach-median_mean001_desc-strength.txt',  
        do_fic=False
    )
    cmaes = optimize.CMAESOptimizer(popsize=10, n_iter=2)
    cmaes.setup_problem(problem, seed=1)
    cmaes.optimize()
    return cmaes

def test_cmaes_optimizer_regional(node_grouping='sym'):
    if node_grouping == 'yeo':
        node_grouping = '/data/project/ei_development/tools/cuBNM/sample_input/yeo7_schaefer-100.txt'
    emp_fc_tril = np.loadtxt('/data/project/ei_development/tools/cuBNM/sample_input/ctx_parc-schaefer-100_hemi-LR_exc-inter_desc-FCtril.txt')
    emp_fcd_tril = np.loadtxt('/data/project/ei_development/tools/cuBNM/sample_input/ctx_parc-schaefer-100_hemi-LR_exc-inter_desc-FCDtril.txt')
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
        sc_path = '/data/project/ei_development/tools/cuBNM/sample_input/ctx_parc-schaefer-100_approach-median_mean001_desc-strength.txt',  
    )
    cmaes = optimize.CMAESOptimizer(popsize=10, n_iter=2)
    cmaes.setup_problem(problem, seed=1)
    cmaes.optimize()
    return cmaes

def run_grid_many_nodes():
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
        sc_path = '/data/project/ei_development/tools/cuBNM/sample_input/ctx_parc-schaefer-400_approach-median_mean001_desc-strength.txt'
    )
    gs.sim_group.run()
    # emp_fc_tril = np.loadtxt('/data/project/ei_development/tools/cuBNM/sample_input/ctx_parc-schaefer-100_hemi-LR_exc-inter_desc-FCtril.txt')
    # emp_fcd_tril = np.loadtxt('/data/project/ei_development/tools/cuBNM/sample_input/ctx_parc-schaefer-100_hemi-LR_exc-inter_desc-FCDtril.txt')
    # scores = gs.evaluate(emp_fc_tril, emp_fcd_tril)
    gs.sim_group.save()
    return gs, scores

if __name__ == '__main__':
    run_sims()
    # gs, scores = run_grid()
    # problem, out = test_problem()
    # cmaes = test_cmaes_optimizer_het()
    # run_grid_many_nodes()