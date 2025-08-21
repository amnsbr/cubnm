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
    sim_group.param_lists['w_p'] = np.full((N_SIMS, nodes), 1.2)
    sim_group.param_lists['J_N'] = np.full((N_SIMS, nodes), 0.15)
    sim_group.run()
    emp_bold = datasets.load_bold('schaefer-100')
    sim_group.score(emp_bold=emp_bold)
    return sim_group

def run_sim_group_pFF(N_SIMS=2, force_cpu=False):
    nodes = 100
    # create a random proportion of FF matrix
    np.random.seed(0)
    pFF_tril = np.random.rand((nodes*(nodes-1))//2)
    pFF = np.zeros((nodes, nodes))
    pFF[np.tril_indices(nodes, -1)] = pFF_tril
    pFF[np.triu_indices(nodes, 1)] = (1 - pFF.T)[np.triu_indices(100, 1)]
    sim_group = sim.rWWSimGroup(
        duration=60,
        TR=1,
        sc=datasets.load_sc('strength', 'schaefer-100'),
        pFF=pFF,
        window_size=10,
        window_step=2,
        sim_verbose=True,
        force_cpu=force_cpu,
        states_ts=False,
        states_sampling=1,
    )
    sim_group.N = N_SIMS
    sim_group.param_lists['G'] = np.repeat(0.5, N_SIMS)
    sim_group.param_lists['w_p'] = np.full((N_SIMS, nodes), 1.2)
    sim_group.param_lists['J_N'] = np.full((N_SIMS, nodes), 0.15)
    sim_group.run()
    emp_bold = datasets.load_bold('schaefer-100')
    sim_group.score(emp_bold=emp_bold)
    return sim_group

def run_sim_group_pFF_two_nodes(N_SIMS=2, force_cpu=False):
    nodes = 2
    sc = np.ones((nodes, nodes), dtype=float) * 0.01
    sc[np.diag_indices(nodes)] = 0
    # create a pFF matrix with all FF from 0->1 and 
    # all FB from 1->0
    pFF = np.array([
        [0, 1],
        [0, 0]
    ])
    sim_group = sim.rWWSimGroup(
        duration=60,
        TR=1,
        sc=sc,
        pFF=pFF,
        window_size=10,
        window_step=2,
        sim_verbose=True,
        force_cpu=force_cpu,
        states_ts=False,
        states_sampling=1,
    )
    sim_group.N = N_SIMS
    sim_group.param_lists['G'] = np.repeat(1000.0, N_SIMS) # exaggerated G
    sim_group.param_lists['w_p'] = np.full((N_SIMS, nodes), 1.2)
    sim_group.param_lists['J_N'] = np.full((N_SIMS, nodes), 0.15)
    sim_group.run()
    return sim_group

def run_sim_group_pFF_separate_G(N_SIMS=2, force_cpu=False):
    nodes = 100
    # create a random proportion of FF matrix
    np.random.seed(0)
    pFF_tril = np.random.rand((nodes*(nodes-1))//2)
    pFF = np.zeros((nodes, nodes))
    pFF[np.tril_indices(nodes, -1)] = pFF_tril
    pFF[np.triu_indices(nodes, 1)] = (1 - pFF.T)[np.triu_indices(100, 1)]
    sim_group = sim.rWWSimGroup(
        duration=60,
        TR=1,
        sc=datasets.load_sc('strength', 'schaefer-100'),
        pFF=pFF,
        window_size=10,
        window_step=2,
        sim_verbose=True,
        force_cpu=force_cpu,
        states_ts=False,
        states_sampling=1,
        separate_G_fb=True
    )
    sim_group.N = N_SIMS
    sim_group.param_lists['G'] = np.repeat(0.5, N_SIMS)
    sim_group.param_lists['G_fb'] = np.repeat(0.25, N_SIMS)
    sim_group.param_lists['w_p'] = np.full((N_SIMS, nodes), 1.2)
    sim_group.param_lists['J_N'] = np.full((N_SIMS, nodes), 0.15)
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
    sim_group.param_lists['w_p'] = np.full((N_SIMS, nodes), 1.2)
    sim_group.param_lists['J_N'] = np.full((N_SIMS, nodes), 0.15)
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
            'G': (0.001, 10.0),
            'w_p': 1.2,
            'J_N': (0.001, 0.5),
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
    go.optimize(problem, grid_shape={'G': 4, 'J_N': 2})
    return go

def run_grid_het():
    problem = optimize.BNMProblem(
        model = 'rWW',
        params = {
            'G': (0.001, 10.0),
            'w_p': 1.2,
            'J_N': (0.001, 0.5),
        },
        duration = 60,
        TR = 1,
        window_size=10,
        window_step=2,
        sc = datasets.load_sc('strength', 'schaefer-100'),
        emp_bold = datasets.load_bold('schaefer-100'),
        het_params = ['J_N'],
        maps = datasets.load_maps(
            ['myelinmap'],
            'schaefer-100', norm='minmax'
        ),
        out_dir = './grid_optimizer',
    )
    go = optimize.GridOptimizer()
    go.optimize(problem, grid_shape={'G': 2, 'J_N': 2, 'J_Nscale0': 2})
    return go

def run_grid_delay():
    problem = optimize.BNMProblem(
        model = 'rWW',
        params = {
            'G': (0.001, 10.0),
            'w_p': 1.2,
            'J_N': (0.001, 0.5),
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
    go.optimize(problem, grid_shape={'G': 2, 'J_N': 2, 'v': 2})
    return go

def run_cmaes_optimizer():
    emp_bold = datasets.load_bold('schaefer-100')
    problem = optimize.BNMProblem(
        model = 'rWW',
        params = {
            'G': (0.001, 10.0),
            'w_p': (0, 2.0),
            'J_N': (0.001, 0.5),
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
            'G': (0.001, 10.0),
            'w_p': (0, 2.0),
            'J_N': (0.001, 0.5),
            'v': (0.5, 8.0)
        },
        emp_bold = emp_bold,
        het_params = ['w_p', 'J_N'],
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
        pFF = pFF,
        separate_G_fb = separate_G_fb,
        force_cpu = force_cpu,
    )
    cmaes = optimize.CMAESOptimizer(popsize=10, n_iter=2, seed=1, 
                                    algorithm_kws=dict(tolfun=5e-3))
    cmaes.setup_problem(problem)
    cmaes.optimize()
    return cmaes

def run_cmaes_optimizer_pFF(separate_G_fb=True):
    nodes = 100
    # create a random proportion of FF matrix
    np.random.seed(0)
    pFF_tril = np.random.rand((nodes*(nodes-1))//2)
    pFF = np.zeros((nodes, nodes))
    pFF[np.tril_indices(nodes, -1)] = pFF_tril
    pFF[np.triu_indices(nodes, 1)] = (1 - pFF.T)[np.triu_indices(100, 1)]
    emp_bold = datasets.load_bold('schaefer-100')
    params = {
        'G': (0.001, 10.0),
        'w_p': (0, 2.0),
        'J_N': (0.001, 0.5),
    }
    if separate_G_fb:
        params['G_fb'] = (0.001, 10.0)
    problem = optimize.BNMProblem(
        model = 'rWW',
        params = params,
        emp_bold = emp_bold,
        gof_terms = ["+fc_corr", "+var_corr", "-nr_diff"],
        het_params = ['w_p', 'J_N'],
        maps = datasets.load_maps(
            ['myelinmap', 'fcgradient01'],
            'schaefer-100', norm='minmax'
        ),
        duration = 60,
        TR = 1,
        window_size=10,
        window_step=2,
        sc = datasets.load_sc('strength', 'schaefer-100'),
        pFF = pFF,
        separate_G_fb = separate_G_fb,
        out_dir = './cmaes_rWW_pFF',
    )
    cmaes = optimize.CMAESOptimizer(popsize=10, n_iter=2, seed=1, 
                                    algorithm_kws=dict(tolfun=5e-3))
    cmaes.setup_problem(problem)
    cmaes.optimize()
    cmaes.save()
    return cmaes

def run_cmaes_optimizer_regional(node_grouping='sym'):
    if node_grouping == 'yeo':
        node_grouping = datasets.load_maps('yeo7', 'schaefer-100', norm=None)
    emp_bold = datasets.load_bold('schaefer-100')
    problem = optimize.BNMProblem(
        model = 'rWW',
        params = {
            'G': (0.001, 10.0),
            'w_p': (0, 2.0),
            'J_N': (0.001, 0.5),
        },
        emp_bold = emp_bold,
        het_params = ['w_p', 'J_N'],
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
            'G': (0.001, 10.0),
            'w_p': (0, 2.0),
            'J_N': (0.001, 0.5),
            'v': (0.5, 8.0)
        },
        emp_bold = emp_bold,
        het_params = ['w_p', 'J_N'],
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
            'G': (0.001, 10.0),
            'w_p': (0, 2.0),
            'J_N': (0.001, 0.5),
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
            'G': (0.001, 10.0),
            'w_p': (0, 2.0),
            'J_N': (0.001, 0.5),
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