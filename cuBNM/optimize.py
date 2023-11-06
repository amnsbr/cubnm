import itertools
from abc import ABC, abstractmethod
import numpy as np
import pandas as pd
from pymoo.core.problem import Problem
from pymoo.algorithms.soo.nonconvex.cmaes import CMAES
from pymoo.optimize import minimize
from pymoo.termination import get_termination
import cma
from bayes_opt import BayesianOptimization, UtilityFunction


from cuBNM import sim

class GridSearch():
    """
    Simple grid search
    """
    #TODO: GridSearch should also be an Optimizer
    def __init__(self, params, **kwargs):
        """
        Parameters
        ---------
        params: (dict)
            a dictionary with G, wEE and wEI (+- wIE, v) keys of
            floats (fixed values) or tuples (min, max, n)
        **kwargs to SimGroup
        """
        self.sim_group = sim.SimGroup(**kwargs)
        param_ranges = {}
        for param, v in params.items():
            if isinstance(v, tuple):
                param_ranges[param] = np.linspace(v[0], v[1], v[2])
            else:
                param_ranges[param] = np.array([v])
        self.param_combs = pd.DataFrame(
            list(itertools.product(*param_ranges.values())),
            columns=param_ranges.keys(), dtype=float
            )
        self.sim_group.N = self.param_combs.shape[0]
        for param in params.keys():
            self.sim_group.param_lists[param] = []
            for sim_idx in range(self.sim_group.N):
                if param in ['G', 'v']:
                    self.sim_group.param_lists[param].append(
                        self.param_combs.iloc[sim_idx].loc[param]
                    )
                else:
                    self.sim_group.param_lists[param].append(
                        np.repeat(self.param_combs.iloc[sim_idx].loc[param], self.sim_group.nodes)
                    )
            self.sim_group.param_lists[param] = np.array(self.sim_group.param_lists[param])

    def evaluate(self, emp_fc_tril, emp_fcd_tril):
        self.sim_group.run()
        return self.sim_group.score(emp_fc_tril, emp_fcd_tril)
        

class RWWProblem(Problem):
    def __init__(self, params, emp_fc_tril, emp_fcd_tril, maps_path=None, **kwargs):
        """
        The "Problem" of reduced Wong-Wang model optimal parameters fitted
        to the provided empirical FC and FCDs

        params: (dict)
            a dictionary with G, wEE and wEI (+- wIE, v) keys of
            floats (fixed values) or tuples (min, max)
        maps_path: (str)
            path to heterogeneity maps. If provided one free parameter
            per map-localparam combination will be added
        """
        self.sim_group = sim.SimGroup(**kwargs)
        self.params = params
        self.emp_fc_tril = emp_fc_tril
        self.emp_fcd_tril = emp_fcd_tril
        self.free_params = []
        self.lb = []
        self.ub = []
        for param, v in params.items():
            if isinstance(v, tuple):
                self.free_params.append(param)
                self.lb.append(v[0])
                self.ub.append(v[1])
        self.lb = np.array(self.lb)
        self.ub = np.array(self.ub)
        self.ndim = len(self.free_params)
        self.fixed_params = list(set(self.params.keys()) - set(self.free_params))
        self.maps_path = maps_path
        self.is_heterogeneous = False
        if self.maps_path is not None:
            self.is_heterogeneous = True
            raise NotImplementedError("Heterogeneous models are not implemented yet")
        super().__init__(
            n_var=len(self.free_params), 
            n_obj=1, 
            n_ieq_constr=0, # TODO: consider using this for enforcing FIC success
            xl=np.zeros(len(self.free_params), dtype=float), 
            xu=np.ones(len(self.free_params), dtype=float)
        )

    def _get_Xt(self, X):
        """
        Transforms X from normalized [0, 1] range to [self.lb, self.ub]
        """
        return (X * (self.ub - self.lb)) + self.lb

    def _set_sim_params(self, X):
        global_params = ['G','v']
        local_params = ['wEE', 'wEI', 'wIE']
        # transform X from [0, 1] range to the actual
        # parameter range
        Xt = self._get_Xt(X)
        if self.is_heterogeneous:
            raise NotImplementedError("Heterogeneous models are not implemented yet")
        else:
            for i, param in enumerate(self.free_params):
                if param in global_params:
                    self.sim_group.param_lists[param] = Xt[:, i]
                else:
                    self.sim_group.param_lists[param] = np.repeat(Xt[:, i], self.sim_group.nodes)
            for param in self.fixed_params:
                if param in global_params:
                    self.sim_group.param_lists[param] = np.repeat(self.params[param], self.sim_group.N)
                else:
                    self.sim_group.param_lists[param] = np.tile(self.params[param], (self.sim_group.N, self.sim_group.nodes))

    def _evaluate(self, X, out, *args, **kwargs):
        # set N to current iteration population size
        # which might be variable, e.g. from evaluating
        # CMAES initial guess to its next iterations
        self.sim_group.N = X.shape[0]
        self._set_sim_params(X)
        self.sim_group.run()
        scores = self.sim_group.score(self.emp_fc_tril, self.emp_fcd_tril)
        if self.sim_group.do_fic:
            out["F"] = (scores.loc[:, 'fic_penalty']-scores.loc[:, 'gof']).values
        else:
            out["F"] = -scores.loc[:, 'gof'].values
        # out["F"] = X.sum(axis=1)
        # out["G"] = ... # TODO: consider using this for enforcing FIC success

class Optimizer(ABC):
    @abstractmethod
    def setup_problem(self, problem, **kwargs):
        pass

    @abstractmethod
    def optimize(self):
        pass

class PymooOptimizer(Optimizer):
    """
    General purpose wrapper for pymoo and other optimizers
    """
    def __init__(self, **kwargs):
        """
        Initialize pymoo optimizers by setting up the
        termination rule based on `n_iter` or `termination`
        """
        # set termination and n_iter (aka n_max_gen)
        self.termination = kwargs.pop('termination', None)
        if self.termination:
            self.n_iter = self.termination.n_max_gen
        else:
            self.n_iter = kwargs.pop('n_iter', 2)
            self.termination = get_termination('n_iter', self.n_iter)

    def setup_problem(self, problem, **kwargs):
        """
        Sets up the problem with the algorithm
        """
        # setup the algorithm with the problem
        self.seed = kwargs.pop('seed', 0)
        self.problem = problem
        self.algorithm.setup(problem, termination=self.termination, 
            seed=self.seed, verbose=True, save_history=True, **kwargs)

    def optimize(self):
        while self.algorithm.has_next():
            # ask the algorithm for the next solution to be evaluated
            pop = self.algorithm.ask()
            # evaluate the individuals using the algorithm's evaluator (necessary to count evaluations for termination)
            self.algorithm.evaluator.eval(self.problem, pop)
            # returned the evaluated individuals which have been evaluated or even modified
            self.algorithm.tell(infills=pop)
            # TODO: do same more things, printing, logging, storing or even modifying the algorithm object


class CMAESOptimizer(PymooOptimizer):
    def __init__(self, popsize, x0=None, sigma=0.5, 
            use_bound_penalty=False, **kwargs):
        """
        Sets up a CMAES optimizer with some defaults
        """
        super().setup_problem(problem, **kwargs)
        if use_bound_penalty:
            bound_penalty = cma.constraints_handler.BoundPenalty([0, 1])
            kwargs['BoundaryHandler'] = bound_penalty
        kwargs['eval_initial_x'] = False
        self.algorithm = CMAES(
            x0=x0, # will estimate the initial guess based on 20 random samples
            sigma=sigma,
            popsize=popsize, # from second generation
            **kwargs
        )

    def setup_problem(self, problem, **kwargs):
        super().setup_problem(problem, **kwargs)
        if self.algorithm.options.get('BoundaryHandler') is not None:
            # the following is to avoid an error caused by pymoo interfering with cma
            # after problem is registered with the algorithm
            # the bounds will be enforced by bound_penalty
            self.algorithm.options['bounds'] = None 

class BayesOptimizer(Optimizer):
    # this does not have the mechanics of PymooOptimizer but
    # uses similar API as much as possible
    def __init__(self, popsize, n_iter):
        """
        Sets up a Bayesian optimizer
        """
        # does not initialize BayesianOptimization because
        # the number of dimensions and therefore pbounds
        # are not known yet
        self.popsize = popsize
        self.n_iter = n_iter
        # setting up utility function
        # based on the default in BayesianOptimization.maximize function
        self.utility = UtilityFunction(kind='ucb',
                                   kappa=2.576,
                                   xi=0.0,
                                   kappa_decay=1,
                                   kappa_decay_delay=0)
        
    def setup_problem(self, problem, **kwargs):
        """
        Initializes the algorithm based on problem ndim
        """
        self.problem = problem
        self.seed = kwargs.pop('seed', 0)
        pbounds = dict([(f'p{i}', (0, 1)) for i in range(self.problem.ndim)])
        self.algorithm = BayesianOptimization(
            f=None,
            pbounds=pbounds,
            verbose=2,
            random_state=self.seed,
        )


    def optimize(self):
        for it in range(self.n_iter):
            # get the next popsize suggestions
            params = []
            for sim_idx in range(self.popsize):
                params.append(self.algorithm.suggest(self.utility))
            # evaluate them
            # convert params dictionaries to a numpy array
            # while making sure that the order is preserved
            X = pd.DataFrame(params).T.sort_index().T.values
            out = {}
            self.problem._evaluate(X, out)
            out['F'] = -out['F'] # this optimizer maximizes the target
            # register the results
            for sim_idx in range(self.popsize):
                self.algorithm.register(params[sim_idx], out['F'][sim_idx])
            # update the utility function
            # note that by default because kappa_decay is 1
            # this doesn't do anything
            self.utility.update_params()
            # TODO: consider bound modification similar to BayesianOptimization.maximize function
            # TODO: consider adding constraints for FIC failure
            # print results
            res = pd.DataFrame(self.problem._get_Xt(X), columns=self.problem.free_params)
            res['F'] = out['F']
            print(res)

