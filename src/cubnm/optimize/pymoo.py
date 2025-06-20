import os
import copy
import random
import numpy as np
import pandas as pd

from pymoo.algorithms.soo.nonconvex.cmaes import CMAES
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.termination import get_termination
import cma

from cubnm import sim
from cubnm.optimize.base import Optimizer

class PymooOptimizer(Optimizer):
    def __init__(
        self, 
        termination=None, 
        n_iter=2,
        seed=1, 
        print_history=True, 
        save_history_sim=False, 
        **kwargs
    ):
        """
        Generic wrapper for `pymoo` optimizers.

        Parameters:
        ----------
        termination: :obj:`pymoo.termination.Termination`, optional
            The termination object that defines the stopping criteria for 
            the optimization process.
            If not provided, the termination criteria will be based on the 
            number of iterations (`n_iter`).
        n_iter: :obj:`int`, optional
            The maximum number of iterations for the optimization process.
            This parameter is only used if `termination` is not provided.
        seed: :obj:`int`, optional
            The seed value for the random number generator used by the optimizer.
        print_history: :obj:`bool`, optional
            Whether to print the optimization history during the 
            optimization process.
        save_history_sim: :obj:`bool`, optional
            Whether to save the simulation data of each iteration.
            Default is False to avoid consuming too much memory across iterations.
        **kwargs
            Additional keyword arguments that can be passed to the `pymoo` optimizer.        
        """
        # set optimizer seed
        self.seed = seed
        # set termination and n_iter (aka n_max_gen)
        self.termination = termination
        if self.termination:
            self.n_iter = self.termination.n_max_gen
        else:
            self.n_iter = n_iter
            self.termination = get_termination("n_iter", self.n_iter)
        self.print_history = print_history
        self.save_history_sim = save_history_sim
        # TODO: some of these options are valid for other
        # non-pymoo optimizers as well, move them to the base class

    def setup_problem(self, problem, pymoo_verbose=False, **kwargs):
        """
        Registers a :class:`cubnm.optimizer.BNMProblem` 
        with the optimizer, so that the optimizer can optimize
        its free parameters.

        Parameters
        ----------
        problem: :obj:`cubnm.optimizer.BNMProblem`
            The problem to be set up with the algorithm.
        pymoo_verbose: :obj:`bool`, optional
            Flag indicating whether to enable verbose output from pymoo. Default is False.
        **kwargs
            Additional keyword arguments to be passed to the algorithm setup method.
        """
        # setup the algorithm with the problem
        self.problem = problem
        if self.problem.n_obj > self.max_obj:
            raise ValueError(
                f"Maximum number of objectives for {type(self.algorithm)} = {self.max_obj} exceeds the number of problem objectives = {self.problem.n_obj}"
            )
        self.algorithm.setup(
            problem,
            termination=self.termination,
            seed=self.seed,
            verbose=pymoo_verbose,
            save_history=True,
            **kwargs,
        )

    def optimize(self):
        """
        Optimizes the associated :class:`cubnm.optimizer.BNMProblem`
        free parameters through an evolutionary optimization approach by
        running multiple generations of parallel simulations until the
        termination criteria is met or maximum number of iterations is reached.
        """
        self.history = []
        self.opt_history = []
        while self.algorithm.has_next():
            # ask the algorithm for the next solution to be evaluated
            pop = self.algorithm.ask()
            # evaluate the individuals using the algorithm's evaluator (necessary to count evaluations for termination)
            scores = []  # pass on an empty scores list to the evaluator to store the individual GOF measures
            self.algorithm.evaluator.eval(self.problem, pop, scores=scores)
            if not self.save_history_sim:
                # clear current simulation data
                # it has to be here before .tell
                self.problem.sim_group.clear()
            # returned the evaluated individuals which have been evaluated or even modified
            self.algorithm.tell(infills=pop)
            # TODO: do same more things, printing, logging, storing or even modifying the algorithm object
            X = np.array([p.x for p in pop])
            Xt = pd.DataFrame(self.problem._get_Xt(X), columns=self.problem.free_params)
            Xt.index.name = "sim_idx"
            F = np.array([p.F for p in pop])
            F = pd.DataFrame(F, columns=self.problem.obj_names)
            if self.problem.multiobj:
                res = pd.concat([Xt, F], axis=1)
            else:
                res = pd.concat([Xt, F, scores[0]], axis=1)
            res["gen"] = self.algorithm.n_gen - 1
            if self.print_history:
                print(res.to_string())
            self.history.append(res)
            if self.problem.n_obj == 1:
                # a single optimum can be defined from each population
                self.opt_history.append(res.loc[res["cost"].argmin()])
        self.history = pd.concat(self.history, axis=0).reset_index(drop=False)
        if self.problem.n_obj == 1:
            # a single optimum can be defined
            self.opt_history = pd.DataFrame(self.opt_history).reset_index(drop=True)
            self.opt = self.history.loc[self.history["cost"].argmin()]
        self.is_fit = True

    @property
    def opt_X(self):
        """
        Returns the optimal parameters in the normalized space.

        Returns
        -------
        :obj:`np.ndarray`
            The optimal parameters in the normalized space.
        """
        if not self.is_fit:
            raise ValueError(
                "Cannot get the optimal parameters before the optimizer is fit. "
                "Call `optimize` method first."
            )
        res = self.algorithm.result()
        return np.atleast_2d(res.X)

class CMAESOptimizer(PymooOptimizer):
    max_obj = 1
    def __init__(
        self,
        popsize,
        x0=None,
        sigma=0.5,
        use_bound_penalty=False,
        algorithm_kws={},
        **kwargs,
    ):
        """
        Covariance Matrix Adaptation Evolution Strategy (CMA-ES) optimizer

        Parameters
        ----------
        popsize: :obj:`int`
            The population size for the optimizer
        x0: array-like, optional
            The initial guess for the optimization.
            If None (default), the initial guess will be estimated based on
            `popsize` random samples as the first generation
        sigma: :obj:`float`, optional
            The initial step size for the optimization
        use_bound_penalty: :obj:`bool`, optional
            Whether to use a bound penalty for the optimization
        algorithm_kws: :obj:`dict`, optional
            Additional keyword arguments for the CMAES algorithm
        **kwargs
            Additional keyword arguments

        Example
        -------
        Run a CMAES optimization for 10 iterations with 
        a population size of 20: ::

            from cubnm import datasets, optimize

            problem = optimize.BNMProblem(
                model = 'rWW',
                params = {
                    'G': (0.5, 2.5),
                    'wEE': (0.05, 0.75),
                    'wEI': 0.15,
                },
                emp_bold = datasets.load_bold('schaefer-100'),
                duration = 60,
                TR = 1,
                sc_path = datasets.load_sc('strength', 'schaefer-100'),
                states_ts = True
            )
            cmaes = optimize.CMAESOptimizer(popsize=20, n_iter=10, seed=1)
            cmaes.setup_problem(problem)
            cmaes.optimize()
            cmaes.save()
        """
        super().__init__(**kwargs)
        if self.seed == 0:
            print(
                "Warning: Seed value for CMAES is set to 0."
                " This will result in different random initializations for each run"
                " due to how `cma` works. Setting seed to 100 instead."
            )
            self.seed = 100
        self.use_bound_penalty = (
            use_bound_penalty  # this option will be set after problem is set up
        )
        self.popsize = popsize
        self.algorithm = CMAES(
            x0=x0, 
            sigma=sigma,
            popsize=popsize,
            **algorithm_kws,
        )
        if x0 is None:
            # overwrite the default of 20 sample points
            # (in pymoo) to popsize
            self.algorithm.n_sample_points = self.popsize

    def setup_problem(self, problem, **kwargs):
        """
        Extends :meth:`cubnm.optimizer.PymooOptimizer.setup_problem` to
        set up the optimizer with the problem and set the bound penalty
        option based on the optimizer's `use_bound_penalty` attribute.

        Parameters
        ----------
        problem: :obj:`cubnm.optimizer.BNMProblem`
            The problem to be set up with the algorithm.
        **kwargs
            Additional keyword arguments to be passed to 
            :meth:`cubnm.optimizer.PymooOptimizer.setup_problem`
        """
        super().setup_problem(problem, **kwargs)
        self.algorithm.options["bounds"] = [0, 1]
        if self.use_bound_penalty:
            self.algorithm.options["BoundaryHandler"] = cma.BoundPenalty
        else:
            self.algorithm.options["BoundaryHandler"] = cma.BoundTransform


class NSGA2Optimizer(PymooOptimizer):
    max_obj = 3
    def __init__(self, popsize, algorithm_kws={}, **kwargs):
        """
        Non-dominated Sorting Genetic Algorithm II (NSGA-II) optimizer

        Parameters
        ----------
        popsize: int
            The population size for the optimizer
        algorithm_kws: dict, optional
            Additional keyword arguments for the NSGA2 algorithm
        kwargs: dict
            Additional keyword arguments for the base class
        """
        super().__init__(**kwargs)
        self.popsize = popsize
        self.algorithm = NSGA2(pop_size=popsize, **algorithm_kws)

def batch_optimize(optimizers, problems, save=True, setup_kwargs={}):
    """
    Optimize a batch of optimizers in parallel (without requiring
    multiple CPU cores when using GPUs).

    Parameters
    ----------
    optimizers: :obj:`list` of :obj:`cubnm.optimize.PymooOptimizer` or :obj:`cubnm.optimize.PymooOptimizer`
        A (list of) optimizer instance(s) to be run in parallel.
        If not a list, the same optimizer will be used for all problems
        and problems must be a list.
    problems: :obj:`list` of :obj:`cubnm.optimize.BNMProblem` or :obj:`cubnm.optimize.BNMProblem`
        A (list of) problem instance(s) to be set up with the optimizers.
        Will be mapped one-to-one with the optimizers.
        If not a list, the same problem will be used in all optimizers
        and optimizers must be a list.
    save: :obj:`bool`, optional
        save the optimizers and their results. This is more efficient
        than saving each optimizer separately as saving involves
        rerunning the optimal simulations, which is done in a batch in
        this function.
    setup_kwargs: :obj:`dict`, optional
        kwargs passed on to :meth:`cubnm.optimize.PymooOptimizer.setup_problem`
        
    Returns
    -------
    optimizers: :obj:`list` of :obj:`cubnm.optimize.PymooOptimizer`
        A list of optimizer instances which have been run in parallel


    Example
    -------
    Run CMAES for two subjects (with different SC and functional data) in a batch: ::

        from cubnm import datasets, optimize

        # assuming sub1 and sub2 SC and BOLD are available as
        # numpy arrays `sc_sub1`, `sc_sub2`, `bold_sub1`, `bold_sub2`

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
            **problem_kwargs
        )
        # problem for subject 2
        p_sub2 = optimize.BNMProblem(
            sc = sc_sub2,
            emp_bold = bold_sub2,
            **problem_kwargs
        )
        # optimizer
        cmaes = optimize.CMAESOptimizer(popsize=20, n_iter=10, seed=1)
        # batch optimization
        optimizers = optimize.batch_optimize(cmaes, [p_sub1, p_sub2])
        # print optima
        print(optimizers[0].opt)
        print(optimizers[1].opt)
    """
    assert any([isinstance(optimizers, list), isinstance(problems, list)]), (
        "At least one of optimizers or problems must be a list"
    )
    # repeat optimizers/problems across problems/optimizers
    # if a single one rather than a list is provided
    if not isinstance(optimizers, list):
        optimizers = [optimizers] * len(problems)
    if not isinstance(problems, list):
        problems = [problems] * len(optimizers)
    assert len(optimizers) == len(problems), (
        "Same number of optimizers and problems must be provided."
    )
    # assert all(isinstance(o, PymooOptimizer) for o in optimizers), (
    #     "Batch optimization is only supported for PymooOptimizer derived classes"
    # )
    assert len(set([type(p.sim_group) for p in problems])) == 1, (
        "All optimizations must be done on the same type of model."
    )
    # create deep copies of the optimizer to ensure that 
    # (in case the same optimizer object is used) they
    # do not conflict with each other
    optimizers = [copy.deepcopy(o) for o in optimizers]
    # do the same for problems to ensure that
    # (if same problem object is used, e.g. when running
    # multiple optimizer runs of the same problem with
    # different seeds) the problem sim groups do not conflict
    problems = [copy.deepcopy(p) for p in problems]
    # dynamically define a MultiSimGroup of the type
    # matching the SimGroup of the first optimizer in the list
    # this will be used to create instances of MultiSimGroup
    # containing ongoing optimizers sim_groups in each iteration
    MultiSimGroup = sim.create_multi_sim_group(type(problems[0].sim_group))
    # to avoid conflict between random states of optimizers we must manually
    # keep track of the random states of each optimizer
    # just to be sure we also keep track of the base random states
    # and switch to it before setting up each optimizer to its problem
    base_np_rand_state = np.random.get_state()
    base_py_rand_state = random.getstate()
    # initialization: set-up problem, keep track of random states and
    # initialize history
    for i, optimizer in enumerate(optimizers):
        # recall base random state
        np.random.set_state(base_np_rand_state)
        random.setstate(base_py_rand_state)
        # set up the problem
        optimizer.setup_problem(problems[i], **setup_kwargs)
        # store optimizer's np and python random states
        optimizer.np_rand_state = np.random.get_state()
        optimizer.py_rand_state = random.getstate()
        # initialize history
        optimizer.history = []
        optimizer.opt_history = []
    # optimization loop continues until all optimizers have finished
    it = 0
    last_N = 0
    last_config = None
    while any([optimizer.algorithm.has_next() for optimizer in optimizers]):
        # get the ongoing optimizers
        ongoing_optimizers = [o for o in optimizers if o.algorithm.has_next()]
        # create a multi-sim group by concatenating the sim groups of the optimizers
        msg = MultiSimGroup([o.problem.sim_group for o in ongoing_optimizers])
        if it > 0:
            # this is to avoid reinitialization of MultiSimGroup
            # as it is created and destroyed in each iteration
            # TODO: avoid destroying and recreating MultiSimGroup 
            # in each iteration
            msg.last_N = last_N
            msg.last_config = last_config
        # get the next populations for each optimizer
        pops = []
        for optimizer in ongoing_optimizers:
            # recall optimizer's latest random state
            np.random.set_state(optimizer.np_rand_state)
            random.setstate(optimizer.py_rand_state)
            # ask the algorithm for the next solution to be evaluated
            pop = optimizer.algorithm.ask()
            # set the optimizer's own sim_group parameters based on pop
            X = np.array([p.x for p in pop])
            optimizer.problem.sim_group.N = X.shape[0]
            optimizer.problem._set_sim_params(X)
            pops.append(pop)
            # store the random state for the next iteration
            optimizer.np_rand_state = np.random.get_state()
            optimizer.py_rand_state = random.getstate()
        # run the simulations of all ongoing optimizers
        # in parallel
        msg.run()
        # evaluate gof (without running simulations) and tell the algorithm
        for i, optimizer in enumerate(ongoing_optimizers):
            # recall optimizer's latest random state
            np.random.set_state(optimizer.np_rand_state)
            random.setstate(optimizer.py_rand_state)
            scores = []  # pass on an empty scores list to the evaluator to store the individual GOF measures
            # evalulate pop without running the simulation (as it is already run)
            optimizer.algorithm.evaluator.eval(optimizer.problem, pops[i], scores=scores, skip_run=True)
            if not optimizer.save_history_sim:
                # clear current simulation data
                optimizer.problem.sim_group.clear()
            # returned the evaluated individuals which have been evaluated or even modified
            optimizer.algorithm.tell(infills=pops[i])
            # store the history
            X = np.array([p.x for p in pops[i]])
            Xt = pd.DataFrame(optimizer.problem._get_Xt(X), columns=optimizer.problem.free_params)
            Xt.index.name = "sim_idx"
            F = np.array([p.F for p in pops[i]])
            F = pd.DataFrame(F, columns=optimizer.problem.obj_names)
            if optimizer.problem.multiobj:
                res = pd.concat([Xt, F], axis=1)
            else:
                res = pd.concat([Xt, F, scores[0]], axis=1)
            res["gen"] = optimizer.algorithm.n_gen - 1
            if optimizer.print_history:
                print(res.to_string())
            optimizer.history.append(res)
            if optimizer.problem.n_obj == 1:
                # a single optimum can be defined from each population
                optimizer.opt_history.append(res.loc[res["cost"].argmin()])
            # store the random state for the next iteration
            optimizer.np_rand_state = np.random.get_state()
            optimizer.py_rand_state = random.getstate()
        # since MultiSimGroup is created and destroyed in each
        # iteration, keep track of its last_N and last_config
        # to avoid reinitialization (except when N is different)
        last_N = copy.deepcopy(msg.last_N)
        last_config = copy.deepcopy(msg.last_config)
        it+=1
        # clear and delete MultiSimGroup instance created for this iteration
        msg.clear()
        del msg
    # organize history and assign optima
    for optimizer in optimizers:
        optimizer.history = pd.concat(optimizer.history, axis=0).reset_index(drop=False)
        if optimizer.problem.n_obj == 1:
            # a single optimum can be defined
            optimizer.opt_history = pd.DataFrame(optimizer.opt_history).reset_index(drop=True)
            optimizer.opt = optimizer.history.loc[optimizer.history["cost"].argmin()]
    if save:
        print("Rerunning and saving the optimal simulation(s)")
        # create a multi-sim group by concatenating the sim groups of the optimizers
        msg = MultiSimGroup([o.problem.sim_group for o in optimizers])
        optimizer_dirs = []
        Xs = []
        # save metadata and set up optimal simulations
        for optimizer in optimizers:
            ## save history and metadata by calling save method of each
            ## optimizer independently
            optimizer_dir = optimizer.save(save_opt=False, save_obj=False)
            optimizer_dirs.append(optimizer_dir)
            # TODO: integrate code repetition here and in `Optimizer.save`
            ## reuse the original problem used throughout the optimization
            ## but change its out_dir, N and parameters to the optimum
            optimizer.problem.sim_group.out_dir = os.path.join(optimizer_dir, "opt_sim")
            os.makedirs(optimizer.problem.sim_group.out_dir, exist_ok=True)
            res = optimizer.algorithm.result()
            X = np.atleast_2d(res.X)
            optimizer.problem.sim_group.N = X.shape[0]
            optimizer.problem._set_sim_params(X)
            Xs.append(X)
        # batch run optimal simulations
        msg.run()
        # evaluate gof and save the optimal simulations
        for i, optimizer in enumerate(optimizers):
            opt_scores = optimizer.problem.eval(Xs[i], skip_run=True)
            Xt = pd.DataFrame(optimizer.problem._get_Xt(Xs[i]), columns=optimizer.problem.free_params)
            pd.concat([Xt, opt_scores], axis=1).to_csv(
                os.path.join(optimizer.problem.sim_group.out_dir, "res.csv")
            )
            ## save simulations data (including BOLD, FC and FCD, extended output, and param maps)
            optimizer.problem.sim_group.save()
        # clear msg
        msg.clear()
        del msg
    return optimizers