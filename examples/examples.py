import numpy as np
from cubnm import optimize, sim, datasets
import copy

def run_sim_group(N_SIMS=2, force_cpu=False):
    nodes = 100
    sim_group = sim.rWWSimGroup(
        duration=60,
        TR=1,
        sc=datasets.load_sc('strength', 'schaefer-100'),
        sim_verbose=True,
        force_cpu=force_cpu,
        states_ts=False,
        states_sampling=1,
    )
    sim_group.N = N_SIMS
    sim_group.param_lists['G'] = np.repeat(0.5, N_SIMS)
    sim_group.param_lists['wEE'] = np.full((N_SIMS, nodes), 0.21)
    sim_group.param_lists['wEI'] = np.full((N_SIMS, nodes), 0.15)
    import time
    print("Running Python FIC...")
    start = time.time()
    sim_group._fic_analytical()
    print("took", time.time() - start, "seconds")
    python_wIE = sim_group.param_lists['wIE'].copy()
    sim_group.param_lists['wIE'][:] = 0.0
    sim_group.run()
    assert np.allclose(sim_group.param_lists['wIE'], python_wIE)
    emp_fc_tril = datasets.load_functional('FC', 'schaefer-100')
    emp_fcd_tril = datasets.load_functional('FCD', 'schaefer-100')
    sim_group.score(emp_fc_tril, emp_fcd_tril)
    return sim_group

def run_sim_group_co(nodes, N_SIMS=1):
    sc = np.ones((nodes, nodes), dtype=float)
    sc[np.diag_indices(nodes)] = 0
    sim_group = sim.rWWExSimGroup(
        duration=60,
        TR=1,
        sc=sc,
        sim_verbose=True,
        states_ts=False,
        bold_remove_s=0,
        noise_segment_length=1,
        dt='1.0',
        do_fc = False,
        do_fcd = False,
        gof_terms=[],
    )
    sim_group.N = N_SIMS
    sim_group._set_default_params()
    sim_group.run()
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
    # emp_fc_tril = datasets.load_functional('FC', 'schaefer-100')
    # emp_fcd_tril = datasets.load_functional('FCD', 'schaefer-100')
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
    emp_fc_tril = datasets.load_functional('FC', 'schaefer-100')
    emp_fcd_tril = datasets.load_functional('FCD', 'schaefer-100')
    sim_group.score(emp_fc_tril, emp_fcd_tril)
    sim_group.save()
    return sim_group

def run_sim_group_kuramoto(N_SIMS=2, force_cpu=False):
    nodes = 100
    sim_group = sim.KuramotoSimGroup(
        duration=30,
        TR=0.5,
        sc=datasets.load_sc('strength', 'schaefer-100'),
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
    emp_fc_tril = datasets.load_functional('FC', 'schaefer-100')
    emp_fcd_tril = datasets.load_functional('FCD', 'schaefer-100')
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
    )
    emp_fc_tril = datasets.load_functional('FC', 'schaefer-100')
    emp_fcd_tril = datasets.load_functional('FCD', 'schaefer-100')
    scores = gs.evaluate(emp_fc_tril, emp_fcd_tril)
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
    )
    emp_fc_tril = datasets.load_functional('FC', 'schaefer-100')
    emp_fcd_tril = datasets.load_functional('FCD', 'schaefer-100')
    scores = gs.evaluate(emp_fc_tril, emp_fcd_tril)
    return gs, scores

def run_problem():
    emp_fc_tril = datasets.load_functional('FC', 'schaefer-100')
    emp_fcd_tril = datasets.load_functional('FCD', 'schaefer-100')
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
    emp_fc_tril = datasets.load_functional('FC', 'schaefer-100')
    emp_fcd_tril = datasets.load_functional('FCD', 'schaefer-100')
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
    )
    cmaes = optimize.CMAESOptimizer(popsize=10, n_iter=2, seed=1)
    cmaes.setup_problem(problem)
    cmaes.optimize()
    return cmaes

def run_cmaes_optimizer_rWWEx():
    emp_fc_tril = datasets.load_functional('FC', 'schaefer-100')
    emp_fcd_tril = datasets.load_functional('FCD', 'schaefer-100')
    problem = optimize.BNMProblem(
        model = 'rWWEx',
        params = {
            'G': (1.0, 3.0),
            'w': (0.05, 1.5),
            'I0': (0.05, 1.5),
            'sigma': 0.001
        },
        emp_fc_tril = emp_fc_tril,
        emp_fcd_tril = emp_fcd_tril,
        duration = 60,
        TR = 1,
        sc = datasets.load_sc('strength', 'schaefer-100'),
    )
    cmaes = optimize.CMAESOptimizer(popsize=10, n_iter=2, seed=1)
    cmaes.setup_problem(problem)
    cmaes.optimize()
    return cmaes

def run_bayes_optimizer():
    emp_fc_tril = datasets.load_functional('FC', 'schaefer-100')
    emp_fcd_tril = datasets.load_functional('FCD', 'schaefer-100')
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
    )
    bo = optimize.BayesOptimizer(popsize=10, n_iter=2, seed=0)
    bo.setup_problem(problem)
    bo.optimize()
    return bo

def run_cmaes_optimizer_het(force_cpu=False, use_bound_penalty=False):
    emp_fc_tril = datasets.load_functional('FC', 'schaefer-100')
    emp_fcd_tril = datasets.load_functional('FCD', 'schaefer-100')
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
    )
    cmaes = optimize.CMAESOptimizer(popsize=10, n_iter=2, seed=1, 
                                    use_bound_penalty=use_bound_penalty,
                                    algorithm_kws=dict(tolfun=5e-3))
    cmaes.setup_problem(problem)
    cmaes.optimize()
    return cmaes

def run_cmaes_optimizer_regional(node_grouping='sym'):
    if node_grouping == 'yeo':
        node_grouping = datasets.load_maps('yeo7', 'schaefer-100', norm=None)
    emp_fc_tril = datasets.load_functional('FC', 'schaefer-100')
    emp_fcd_tril = datasets.load_functional('FCD', 'schaefer-100')
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
    )
    cmaes = optimize.CMAESOptimizer(popsize=10, n_iter=2, seed=1)
    cmaes.setup_problem(problem)
    cmaes.optimize()
    cmaes.save()
    return cmaes

def run_nsga2_optimizer_het(force_cpu=False):
    emp_fc_tril = datasets.load_functional('FC', 'schaefer-100')
    emp_fcd_tril = datasets.load_functional('FCD', 'schaefer-100')
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
    )
    optimizer = optimize.NSGA2Optimizer(popsize=10, n_iter=3, seed=1)
    optimizer.setup_problem(problem)
    optimizer.optimize()
    return optimizer

def run_batch_optimize_diff_sub():
    sc_sub1 = datasets.load_sc('strength', 'schaefer-100')
    sc_sub2 = sc_sub1[::-1, ::-1].copy()
    bold_sub1 = datasets.load_functional('bold', 'schaefer-100')
    bold_sub2 = bold_sub1[::-1, ::-1].copy()
    # shared problem configuration
    problem_kwargs = dict(
        model = 'rWW',
        params = {
            'G': (1.0, 3.0),
            'wEE': (0.05, 0.5),
            'wEI': 0.15,
        },
        duration = 60,
        TR = 1,
    )
    # problem for subject 1
    p_sub1 = optimize.BNMProblem(
        sc = sc_sub1,
        emp_bold = bold_sub1,
        out_dir = tempfile.mkdtemp(),
        **problem_kwargs
    )
    # problem for subject 2
    p_sub2 = optimize.BNMProblem(
        sc = sc_sub2,
        emp_bold = bold_sub2,
        out_dir = tempfile.mkdtemp(),
        **problem_kwargs
    )
    # optimizers
    cmaes = optimize.CMAESOptimizer(popsize=20, n_iter=2, seed=1)
    # batch optimization
    optimizers = optimize.batch_optimize(cmaes, [p_sub1, p_sub2])
    # print optima
    print(optimizers[0].opt)
    print(optimizers[1].opt)
    return optimizers

def run_batch_optimize_identical():
    sc = datasets.load_sc('strength', 'schaefer-100')
    bold = datasets.load_functional('bold', 'schaefer-100')
    # shared problem configuration
    problem_kwargs = dict(
        model = 'rWW',
        params = {
            'G': (1.0, 3.0),
            'wEE': (0.05, 0.5),
            'wEI': 0.15,
        },
        duration = 60,
        TR = 1,
        sc = sc,
        emp_bold = bold,
        out_dir = tempfile.mkdtemp(),
    )
    problem = optimize.BNMProblem(
        **problem_kwargs
    )
    # optimizer
    cmaes = optimize.CMAESOptimizer(popsize=20, n_iter=2, seed=1)
    # batch optimization
    optimizers = optimize.batch_optimize([cmaes, cmaes], [problem, problem])
    # print optima
    print(optimizers[0].opt)
    print(optimizers[1].opt)
    
    # serial optimizer
    cmaes_ind = optimize.CMAESOptimizer(popsize=20, n_iter=2, seed=1)
    cmaes_ind.setup_problem(problem)
    cmaes_ind.optimize()
    print(cmaes_ind.opt)
    return optimizers[0], optimizers[1], cmaes_ind