import numpy as np
import cubnm
from cubnm._core import run_simulations, set_const
from cubnm import optimize, sim, utils, datasets
from cubnm._setup_opts import many_nodes_flag, gpu_enabled_flag

import os
from pymoo.algorithms.soo.nonconvex.cmaes import CMAES
from pymoo.optimize import minimize
from pymoo.termination.default import DefaultSingleObjectiveTermination
from pymoo.core.termination import Termination
from pymoo.termination import get_termination
import cma

def run_sims(N_SIMS=2, v=0.1, force_cpu=False, rand_seed=410, force_reinit=False, serial=False):
    # run identical simulations and check if BOLD is the same
    nodes = 100
    time_steps = 60000
    BOLD_TR = 1000
    # time_steps = 450000
    # BOLD_TR = 3000
    window_size = 10
    window_step = 2
    ext_out = True
    states_ts = True
    states_sampling = BOLD_TR
    noise_out = True

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

    model_config = {
        'do_fic': str(int(do_fic)),
        'max_fic_trials': '5',
        'verbose': '1',
        # 'noise_time_steps': '30001',
        # 'noise_time_steps': str(time_steps) # whole-noise
        # 'fic_verbose': '0',
    }
    if serial:
        model_config['serial'] = '1'
        if do_fic:
            model_config['max_fic_trials'] = '0'
    do_delay = False
    # do_delay = True
    if do_delay:
        # v_list = np.linspace(0.5, 4.0, N_SIMS)
        v_list = np.repeat(v, N_SIMS)
        # v_list = np.repeat(-1.0, N_SIMS)

        # with delay it is recommended to do
        # the syncing of nodes every 1 msec instead
        # of every 0.1 msec. Otherwise it'll be very slow
        model_config['sync_msec'] = '1'
    else:
        v_list = np.repeat(0.0, N_SIMS) # doesn't matter what it is!
        SC_dist = np.zeros(nodes*nodes, dtype=float) # doesn't matter what it is!
    force_cpu = force_cpu | (not gpu_enabled_flag) | (utils.avail_gpus()==0)
    global_params = G_list[np.newaxis, :]
    regional_params = np.vstack([w_EE_list, w_EI_list, w_IE_list])
    # make sure all the input arrays are of type float/double
    out = run_simulations(
        'rWW', SC[None, :], np.repeat(0, N_SIMS),
        SC_dist, global_params, regional_params, v_list,
        model_config,
        ext_out, states_ts, noise_out, do_delay, force_reinit, force_cpu,
        N_SIMS, nodes, time_steps, BOLD_TR, states_sampling,
        window_size, window_step, rand_seed
    )
    sim_bolds , sim_fc_trils, sim_fcd_trils = out[:3]

    for sim_idx in range(N_SIMS):
        print(f"BOLD Python {sim_idx}: shape {sim_bolds.shape}, idx 500 {sim_bolds[sim_idx, 500]}")
        print(f"fc_trils Python {sim_idx}: shape {sim_fc_trils.shape}, idx 30 {sim_fc_trils[sim_idx, 30]}")
        print(f"fcd_trils Python {sim_idx}: shape {sim_fcd_trils.shape}, idx 30 {sim_fcd_trils[sim_idx, 30]}")
    return sim_bolds, sim_fc_trils, sim_fcd_trils

def run_sim_group(force_cpu=False):
    nodes = 100
    N_SIMS = 2
    sim_group = sim.rWWSimGroup(
        duration=60,
        TR=1,
        sc=datasets.load_sc('strength', 'schaefer-100'),
        out_dir='./rWW',
        sim_verbose=True,
        force_cpu=force_cpu,
        states_ts=False,
        states_sampling=1,
    )
    sim_group.N = N_SIMS
    sim_group.param_lists['G'] = np.repeat(0.5, N_SIMS)
    sim_group.param_lists['wEE'] = np.full((N_SIMS, nodes), 0.21)
    sim_group.param_lists['wEI'] = np.full((N_SIMS, nodes), 0.15)
    sim_group.run()
    emp_fc_tril = datasets.load_functional('FC', 'schaefer-100', exc_interhemispheric=True)
    emp_fcd_tril = datasets.load_functional('FCD', 'schaefer-100', exc_interhemispheric=True)
    sim_group.score(emp_fc_tril, emp_fcd_tril)
    sim_group.save()
    return sim_group

def run_sim_group_400(force_cpu=False):
    nodes = 400
    N_SIMS = 1
    sim_group = sim.rWWSimGroup(
        duration=60,
        TR=1,
        sc=datasets.load_sc('strength', 'schaefer-400'),
        out_dir='./rWW',
        sim_verbose=True,
        force_cpu=force_cpu,
        progress_interval=2000,
        # do_fic=False
    )
    sim_group.N = N_SIMS
    sim_group.param_lists['G'] = np.repeat(0.5, N_SIMS)
    sim_group.param_lists['wEE'] = np.full((N_SIMS, nodes), 0.21)
    sim_group.param_lists['wEI'] = np.full((N_SIMS, nodes), 0.15)
    # sim_group.param_lists['wIE'] = np.full((N_SIMS, nodes), 1.0)
    sim_group.sync_msec = True # otherwise it'll be very slow
    # sim_group.sync_msec = False
    sim_group.run()
    # emp_fc_tril = datasets.load_functional('FC', 'schaefer-100', exc_interhemispheric=True)
    # emp_fcd_tril = datasets.load_functional('FCD', 'schaefer-100', exc_interhemispheric=True)
    # sim_group.score(emp_fc_tril, emp_fcd_tril)
    # sim_group.save()
    return sim_group

def run_sim_group_rWWEx(force_cpu=False):
    nodes = 100
    N_SIMS = 2
    sim_group = sim.rWWExSimGroup(
        duration=60,
        TR=1,
        sc=datasets.load_sc('strength', 'schaefer-100'),
        out_dir='./rWWEx',
        sim_verbose=True,
        force_cpu=force_cpu,
        ext_out=False
    )
    sim_group.N = N_SIMS
    sim_group.param_lists['G'] = np.repeat(0.5, N_SIMS)
    sim_group.param_lists['w'] = np.full((N_SIMS, nodes), 0.9)
    sim_group.param_lists['I0'] = np.full((N_SIMS, nodes), 0.3)
    sim_group.param_lists['sigma'] = np.full((N_SIMS, nodes), 0.001)
    sim_group.run()
    emp_fc_tril = datasets.load_functional('FC', 'schaefer-100', exc_interhemispheric=True)
    emp_fcd_tril = datasets.load_functional('FCD', 'schaefer-100', exc_interhemispheric=True)
    sim_group.score(emp_fc_tril, emp_fcd_tril)
    sim_group.save()
    return sim_group

def run_sim_group_kuramoto(N_SIMS=2, force_cpu=False):
    nodes = 100
    sim_group = sim.KuramotoSimGroup(
        duration=30,
        TR=0.5,
        sc=datasets.load_sc('strength', 'schaefer-100'),
        out_dir='./Kuramoto',
        sim_verbose=True,
        force_cpu=force_cpu,
        states_ts=True,
        bold_remove_s=0,
    )
    sim_group.N = N_SIMS
    sim_group.param_lists['G'] = np.repeat(0.5, N_SIMS)
    sim_group.param_lists['omega'] = np.full((N_SIMS, nodes), np.pi)
    sim_group.param_lists['sigma'] = np.full((N_SIMS, nodes), 0.17)
    sim_group.run()
    emp_fc_tril = datasets.load_functional('FC', 'schaefer-100', exc_interhemispheric=True)
    emp_fcd_tril = datasets.load_functional('FCD', 'schaefer-100', exc_interhemispheric=True)
    sim_group.score(emp_fc_tril, emp_fcd_tril)
    sim_group.save()
    return sim_group

def run_grid():
    gs = optimize.GridSearch(
        model = 'rWW',
        params = {
            'G': (0.5, 2.5, 4),
            'wEE': 0.15,
            'wEI': 0.21
        },
        # duration = 450,
        # TR = 3,
        duration = 60,
        TR = 1,
        sc = datasets.load_sc('strength', 'schaefer-100'),
        out_dir='./grid',
    )
    emp_fc_tril = datasets.load_functional('FC', 'schaefer-100', exc_interhemispheric=True)
    emp_fcd_tril = datasets.load_functional('FCD', 'schaefer-100', exc_interhemispheric=True)
    scores = gs.evaluate(emp_fc_tril, emp_fcd_tril)
    gs.sim_group.save()
    return gs, scores

def run_grid_delay():
    gs = optimize.GridSearch(
        model = 'rWW',
        params = {
            'G': 0.5,
            'wEE': (0.05, 1, 2),
            'wEI': (0.07, 0.75, 2),
            'v': (1, 5, 2)
        },
        duration = 60,
        TR = 1,
        sc = datasets.load_sc('strength', 'schaefer-100'),
        sc_dist = datasets.load_sc('length', 'schaefer-100'),
        out_dir='./grid_delay',
    )
    emp_fc_tril = datasets.load_functional('FC', 'schaefer-100', exc_interhemispheric=True)
    emp_fcd_tril = datasets.load_functional('FCD', 'schaefer-100', exc_interhemispheric=True)
    scores = gs.evaluate(emp_fc_tril, emp_fcd_tril)
    return gs, scores

def run_problem():
    emp_fc_tril = datasets.load_functional('FC', 'schaefer-100', exc_interhemispheric=True)
    emp_fcd_tril = datasets.load_functional('FCD', 'schaefer-100', exc_interhemispheric=True)
    problem = optimize.BNMProblem(
        model = 'rWW',
        params = {
            'G': (1.0, 3.0),
            'wEE': (0.05, 0.5),
            'wEI': 0.15,
        },
        emp_fc_tril = emp_fc_tril,
        emp_fcd_tril = emp_fcd_tril,
        duration = 60,
        TR = 1,
        sc = datasets.load_sc('strength', 'schaefer-100'),
        out_dir='./problem'
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
    problem = optimize.BNMProblem(
        model = 'rWW',
        params = {
            'G': (1.0, 3.0),
            'wEE': (0.05, 0.5),
            'wEI': 0.15,
        },
        emp_fc_tril = emp_fc_tril,
        emp_fcd_tril = emp_fcd_tril,
        duration = 60,
        TR = 1,
        sc_path = datasets.load_sc('strength', 'schaefer-100'),
        out_dir = './cmaes'
    )
    cmaes = optimize.CMAESOptimizer(popsize=10, n_iter=2)
    cmaes.setup_problem(problem, seed=1)
    cmaes.optimize()
    return cmaes

def run_bayes_optimizer():
    emp_fc_tril = datasets.load_functional('FC', 'schaefer-100', exc_interhemispheric=True)
    emp_fcd_tril = datasets.load_functional('FCD', 'schaefer-100', exc_interhemispheric=True)
    problem = optimize.BNMProblem(
        model = 'rWW',
        params = {
            'G': (1.0, 3.0),
            'wEE': (0.05, 0.5),
            'wEI': 0.15,
        },
        emp_fc_tril = emp_fc_tril,
        emp_fcd_tril = emp_fcd_tril,
        duration = 60,
        TR = 1,
        sc = datasets.load_sc('strength', 'schaefer-100'),
        out_dir = './bayes'
    )
    bo = optimize.BayesOptimizer(popsize=10, n_iter=7)
    bo.setup_problem(problem, seed=1)
    bo.optimize()
    return bo

def run_cmaes_optimizer_het(force_cpu=False, use_bound_penalty=False):
    emp_fc_tril = datasets.load_functional('FC', 'schaefer-100', exc_interhemispheric=True)
    emp_fcd_tril = datasets.load_functional('FCD', 'schaefer-100', exc_interhemispheric=True)
    problem = optimize.BNMProblem(
        model = 'rWW',
        params = {
            'G': (1.0, 3.0),
            'wEE': (0.05, 0.5),
            'wEI': 0.15,
            'v': (0.5, 8.0)
        },
        emp_fc_tril = emp_fc_tril,
        emp_fcd_tril = emp_fcd_tril,
        het_params = ['wEE', 'wEI'],
        maps = datasets.load_maps(
            ['myelinmap', 'fcgradient01'],
            'schaefer-100', norm='minmax'
        ),
        duration = 60,
        TR = 1,
        sc = datasets.load_sc('strength', 'schaefer-100'),
        sc_dist = datasets.load_sc('length', 'schaefer-100'),
        force_cpu = force_cpu,
        out_dir = './cmaes_het',
    )
    cmaes = optimize.CMAESOptimizer(popsize=10, n_iter=2, seed=1, 
                                    use_bound_penalty=use_bound_penalty,
                                    algorithm_kws=dict(tolfun=5e-3))
    cmaes.setup_problem(problem)
    cmaes.optimize()
    cmaes.save()
    return cmaes

def run_cmaes_optimizer_regional(node_grouping='sym'):
    if node_grouping == 'yeo':
        node_grouping = datasets.load_maps('yeo7', 'schaefer-100', norm=None)
    emp_fc_tril = datasets.load_functional('FC', 'schaefer-100', exc_interhemispheric=True)
    emp_fcd_tril = datasets.load_functional('FCD', 'schaefer-100', exc_interhemispheric=True)
    problem = optimize.BNMProblem(
        model = 'rWW',
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
        sc = datasets.load_sc('strength', 'schaefer-100'),
        out_dir = './cmaes_regional'
    )
    cmaes = optimize.CMAESOptimizer(popsize=10, n_iter=2, seed=1)
    cmaes.setup_problem(problem)
    cmaes.optimize()
    cmaes.save()
    return cmaes

def run_nsga2_optimizer_het(force_cpu=False):
    emp_fc_tril = datasets.load_functional('FC', 'schaefer-100', exc_interhemispheric=True)
    emp_fcd_tril = datasets.load_functional('FCD', 'schaefer-100', exc_interhemispheric=True)
    problem = optimize.BNMProblem(
        model = 'rWW',
        params = {
            'G': (1.0, 3.0),
            'wEE': (0.05, 0.5),
            'wEI': 0.15,
            'v': (0.5, 8.0)
        },
        emp_fc_tril = emp_fc_tril,
        emp_fcd_tril = emp_fcd_tril,
        het_params = ['wEE', 'wEI'],
        maps = datasets.load_maps(
            ['myelinmap', 'fcgradient01'], 
            'schaefer-100', norm='minmax'),
        duration = 60,
        TR = 1,
        sc = datasets.load_sc('strength', 'schaefer-100'),
        sc_dist = datasets.load_sc('length', 'schaefer-100'),
        force_cpu = force_cpu,
        gof_terms = ['-fc_normec', '-fcd_ks'],
        multiobj = True,
        out_dir = './nsga2'
    )
    optimizer = optimize.NSGA2Optimizer(popsize=10, n_iter=3, seed=1)
    optimizer.setup_problem(problem)
    optimizer.optimize()
    optimizer.save()
    return optimizer

if __name__ == '__main__':
    # sim_bolds, sim_fc_trils, sim_fcd_trils = run_sims(2, force_cpu=False)
    sg = run_sim_group(force_cpu=False)
    # sg = run_sim_group_400(force_cpu=False)
    # sg = run_sim_group_rWWEx(force_cpu=False)
    # sg_kuramoto = run_sim_group_kuramoto(force_cpu=False)
    # gs, scores = run_grid()
    # problem, out = run_problem()
    # cmaes = run_cmaes_optimizer_het()
    # cmaes = run_cmaes_optimizer_regional()
    # run_grid_many_nodes()
    # nsga2 = run_nsga2_optimizer_het()
