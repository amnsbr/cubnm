import numpy as np
from cubnm import optimize, sim, datasets
import copy

def run_sim_group(N_SIMS=2, force_cpu=False):
    nodes = 100
    sim_group = sim.rWWSimGroup(
        duration=60,
        TR=1,
        window_size=10,
        window_step=2,
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
    sim_group.run()
    emp_bold = datasets.load_bold('schaefer-100')
    sim_group.score(emp_bold=emp_bold)
    return sim_group

def run_sim_group_delay(N_SIMS=2, force_cpu=False):
    nodes = 100
    sim_group = sim.rWWSimGroup(
        duration=60,
        TR=1,
        sc=datasets.load_sc('strength', 'schaefer-100'),
        sc_dist=datasets.load_sc('length', 'schaefer-100'),
        window_size=10,
        window_step=2,
        sim_verbose=True,
        force_cpu=force_cpu,
        states_ts=False,
        states_sampling=1,
        dt='1.0',
    )
    sim_group.N = N_SIMS
    sim_group.param_lists['G'] = np.repeat(0.5, N_SIMS)
    sim_group.param_lists['v'] = np.repeat(0.5, N_SIMS)
    sim_group.param_lists['wEE'] = np.full((N_SIMS, nodes), 0.21)
    sim_group.param_lists['wEI'] = np.full((N_SIMS, nodes), 0.15)
    sim_group.run()
    emp_bold = datasets.load_bold('schaefer-100')
    sim_group.score(emp_bold=emp_bold)
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
        window_size=10,
        window_step=2,
        ext_out=False
    )
    sim_group.N = N_SIMS
    sim_group.param_lists['G'] = np.repeat(0.5, N_SIMS)
    sim_group.param_lists['w'] = np.full((N_SIMS, nodes), 0.9)
    sim_group.param_lists['I0'] = np.full((N_SIMS, nodes), 0.3)
    sim_group.param_lists['sigma'] = np.full((N_SIMS, nodes), 0.001)
    sim_group.run()
    emp_bold = datasets.load_bold('schaefer-100')
    sim_group.score(emp_bold=emp_bold)
    sim_group.save()
    return sim_group

def run_sim_group_kuramoto(N_SIMS=2, force_cpu=False):
    nodes = 100
    sim_group = sim.KuramotoSimGroup(
        duration=30,
        TR=0.5,
        window_size=10,
        window_step=2,
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
    emp_bold = datasets.load_bold('schaefer-100')
    sim_group.score(emp_bold=emp_bold)
    sim_group.save()
    return sim_group

def run_sim_group_jr(N_SIMS=2, force_cpu=False):
    nodes = 100
    sim_group = sim.JRSimGroup(
        duration=60,
        TR=1,
        window_size=10,
        window_step=2,
        sc=datasets.load_sc('strength', 'schaefer-100'),
        sim_verbose=True,
        force_cpu=force_cpu,
        states_ts=False,
        states_sampling=1,
    )
    sim_group.N = N_SIMS
    sim_group._set_default_params()
    sim_group.run()
    emp_bold = datasets.load_bold('schaefer-100')
    sim_group.score(emp_bold=emp_bold)
    return sim_group

def run_sim_group_many_nodes(nodes=9684, force_cpu=False):
    # generate reproducible random sc
    np.random.seed(0)
    sc = np.random.rand(nodes, nodes)
    # make it symmetric
    sc += sc.T
    # make its sum 1
    sc = sc / sc.sum()
    # set diagonal to 0
    sc[np.diag_indices(nodes)] = 0
    sim_group = sim.rWWExSimGroup(
        duration=60,
        TR=1,
        sc=sc,
        sim_verbose=True,
        states_ts=True,
        states_sampling=1,
        noise_segment_length=1,
        do_fc=True,
        do_fcd=False,
        gof_terms=[],
        progress_interval=500,
        dt='1.0',
        force_cpu=force_cpu,
    )
    sim_group.N = 1
    sim_group._set_default_params()
    sim_group.param_lists['G'][:] = 0.005
    sim_group.run()
    return sim_group

def run_grid():
    problem = optimize.BNMProblem(
        model = 'rWW',
        params = {
            'G': (0.5, 2.5),
            'wEE': 0.15,
            'wEI': (0.21, 0.22),
        },
        duration = 60,
        TR = 1,
        window_size=10,
        window_step=2,
        sc = datasets.load_sc('strength', 'schaefer-100'),
        emp_bold = datasets.load_bold('schaefer-100'),
        out_dir = './grid_optimizer',
    )
    go = optimize.GridOptimizer()
    go.optimize(problem, grid_shape={'G': 4, 'wEI': 2})
    return go

def run_grid_het():
    problem = optimize.BNMProblem(
        model = 'rWW',
        params = {
            'G': (0.5, 2.5),
            'wEE': 0.15,
            'wEI': (0.21, 0.42),
        },
        duration = 60,
        TR = 1,
        window_size=10,
        window_step=2,
        sc = datasets.load_sc('strength', 'schaefer-100'),
        emp_bold = datasets.load_bold('schaefer-100'),
        het_params = ['wEI'],
        maps = datasets.load_maps(
            ['myelinmap'],
            'schaefer-100', norm='minmax'
        ),
        out_dir = './grid_optimizer',
    )
    go = optimize.GridOptimizer()
    go.optimize(problem, grid_shape={'G': 2, 'wEI': 2, 'wEIscale0': 2})
    return go

def run_grid_delay():
    problem = optimize.BNMProblem(
        model = 'rWW',
        params = {
            'G': (0.5, 2.5),
            'wEE': 0.15,
            'wEI': (0.21, 0.22),
            'v': (0.5, 2.5),
        },
        duration = 60,
        TR = 1,
        window_size=10,
        window_step=2,
        sc = datasets.load_sc('strength', 'schaefer-100'),
        sc_dist = datasets.load_sc('length', 'schaefer-100'),
        dt = '1.0',
        emp_bold = datasets.load_bold('schaefer-100'),
        out_dir = './grid_optimizer',
    )
    go = optimize.GridOptimizer()
    go.optimize(problem, grid_shape={'G': 2, 'wEI': 2, 'v': 2})
    return go

def run_problem():
    emp_bold = datasets.load_bold('schaefer-100')
    problem = optimize.BNMProblem(
        model = 'rWW',
        params = {
            'G': (1.0, 3.0),
            'wEE': (0.05, 0.5),
            'wEI': 0.15,
        },
        emp_bold = emp_bold,
        duration = 60,
        TR = 1,
        window_size=10,
        window_step=2,
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
    emp_bold = datasets.load_bold('schaefer-100')
    problem = optimize.BNMProblem(
        model = 'rWW',
        params = {
            'G': (1.0, 3.0),
            'wEE': (0.05, 0.5),
            'wEI': 0.15,
        },
        emp_bold = emp_bold,
        duration = 60,
        TR = 1,
        window_size=10,
        window_step=2,
        sc = datasets.load_sc('strength', 'schaefer-100'),
    )
    cmaes = optimize.CMAESOptimizer(popsize=10, n_iter=2, seed=1)
    cmaes.setup_problem(problem)
    cmaes.optimize()
    return cmaes

def run_cmaes_optimizer_rWWEx():
    emp_bold = datasets.load_bold('schaefer-100')
    problem = optimize.BNMProblem(
        model = 'rWWEx',
        params = {
            'G': (1.0, 3.0),
            'w': (0.05, 1.5),
            'I0': (0.05, 1.5),
            'sigma': 0.001
        },
        emp_bold = emp_bold,
        duration = 60,
        TR = 1,
        window_size=10,
        window_step=2,
        sc = datasets.load_sc('strength', 'schaefer-100'),
    )
    cmaes = optimize.CMAESOptimizer(popsize=10, n_iter=2, seed=1)
    cmaes.setup_problem(problem)
    cmaes.optimize()
    return cmaes

def run_cmaes_optimizer_het(force_cpu=False, use_bound_penalty=False):
    emp_bold = datasets.load_bold('schaefer-100')
    problem = optimize.BNMProblem(
        model = 'rWW',
        params = {
            'G': (1.0, 3.0),
            'wEE': (0.05, 0.5),
            'wEI': 0.15,
            'v': (0.5, 8.0)
        },
        emp_bold = emp_bold,
        het_params = ['wEE', 'wEI'],
        maps = datasets.load_maps(
            ['myelinmap', 'fcgradient01'],
            'schaefer-100', norm='minmax'
        ),
        duration = 60,
        TR = 1,
        window_size=10,
        window_step=2,
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
    emp_bold = datasets.load_bold('schaefer-100')
    problem = optimize.BNMProblem(
        model = 'rWW',
        params = {
            'G': (1.0, 3.0),
            'wEE': (0.05, 0.5),
            'wEI': (0.05, 0.75),
        },
        emp_bold = emp_bold,
        het_params = ['wEE', 'wEI'],
        node_grouping = node_grouping,
        duration = 60,
        TR = 1,
        window_size=10,
        window_step=2,
        sc = datasets.load_sc('strength', 'schaefer-100'),
    )
    cmaes = optimize.CMAESOptimizer(popsize=10, n_iter=2, seed=1)
    cmaes.setup_problem(problem)
    cmaes.optimize()
    cmaes.save()
    return cmaes

def run_nsga2_optimizer_het(force_cpu=False):
    emp_bold = datasets.load_bold('schaefer-100')
    problem = optimize.BNMProblem(
        model = 'rWW',
        params = {
            'G': (1.0, 3.0),
            'wEE': (0.05, 0.5),
            'wEI': 0.15,
            'v': (0.5, 8.0)
        },
        emp_bold = emp_bold,
        het_params = ['wEE', 'wEI'],
        maps = datasets.load_maps(
            ['myelinmap', 'fcgradient01'], 
            'schaefer-100', norm='minmax'),
        duration = 60,
        TR = 1,
        window_size=10,
        window_step=2,
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
    bold_sub1 = datasets.load_bold('schaefer-100')
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
        window_size=10,
        window_step=2,
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
    bold = datasets.load_bold('schaefer-100')
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
        window_size=10,
        window_step=2,
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