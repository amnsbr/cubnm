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
    global_params = ['G','v']
    local_params = ['wEE', 'wEI', 'wIE']
    def __init__(self, params, emp_fc_tril, emp_fcd_tril, het_params=[], maps_path=None,
            reject_negative=False, node_grouping=None, **kwargs):
        """
        The "Problem" of reduced Wong-Wang model optimal parameters fitted
        to the provided empirical FC and FCDs

        params: (dict)
            a dictionary with G, wEE and wEI (+- wIE, v) keys of
            floats (fixed values) or tuples (min, max)
        emp_fc_tril, emp_fcd_tril: (np.ndarrray) (n_pairs,)
            target empirical FC and FCD
        het_params: (list of str)
            which local parameters of 'wEE', 'wEI' and 'wIE' should be regionally variable
            wIE must not be used when do_fic is true
        maps_path: (str)
            path to heterogeneity maps. If provided one free parameter
            per map-localparam combination will be added
            it is expected to have (n_maps, nodes) dimension (for historical reasons)
        reject_negative: (bool)
            rejects particles with negative (or actually < 0.001) local parameters
            after applying heterogeneity. If False, instead of rejecting shifts the parameter map to
            have a min of 0.001
        node_grouping: (str)
            - None: does not use region-/group-specific parameters
            - 'node'
            - 'sym'
            - path to node grouping array
        **kwargs to sim.SimGroup
        """
        # set opts
        self.params = params
        self.emp_fc_tril = emp_fc_tril
        self.emp_fcd_tril = emp_fcd_tril
        self.het_params = kwargs.pop('het_params', het_params)
        self.maps_path = kwargs.pop('maps_path', maps_path)
        self.reject_negative = kwargs.pop('reject_negative', reject_negative)
        self.node_grouping = kwargs.pop('node_grouping', node_grouping)
        # initialize sim_group (N not known yet)
        self.sim_group = sim.SimGroup(**kwargs)
        # raise errors if opts are impossible
        if (self.sim_group.do_fic) & ('wIE' in self.het_params):
            raise ValueError("wIE should not be specified as a heterogeneous parameter when FIC is done")
        if (self.node_grouping is not None) & (self.maps_path is not None):
            raise ValueError("Both node_groups and maps_path cannot be used")
        # identify free and fixed parameters
        self.free_params = []
        self.lb = []
        self.ub = []
        # decide if parameters can be variable across nodes/groups
        # note that this is different from map-based heterogeneity
        self.is_regional = self.node_grouping is not None
        if self.is_regional:
            for param in self.het_params:
                if not isinstance(params[param], tuple):
                    raise ValueError(f"{param} is set to be heterogeneous based on groups but no range is provided")
        # set up node_groups (ordered index of all unique groups)
        # and memberships (from node i to N, which group they belong to)
        if self.is_regional:
            if self.node_grouping=='node':
                # each node gets its own local free parameters
                # therefore each node has its own group and 
                # is the only member of it
                self.node_groups = np.arange(self.sim_group.nodes)
                self.memberships = np.arange(self.sim_group.nodes)
            elif self.node_grouping=='sym':
                print("Warning: `sym` node grouping assumes symmetry of parcels between L and R hemispheres")
                # nodes i & rh_idx+i belong to the same group
                # and will have similar parameters
                assert (self.sim_group.nodes % 2 == 0), "Number of nodes must be even"
                rh_idx = int(self.sim_group.nodes / 2)
                self.node_groups = np.arange(rh_idx)
                self.memberships = np.tile(np.arange(rh_idx), 2)
            else:
                self.memberships = np.loadtxt(self.node_grouping).astype('int')
                self.node_groups = np.unique(self.memberships)
        # set up global and local (incl. bias) free parameters
        for param, v in params.items():
            if isinstance(v, tuple):
                if self.is_regional & (param in self.local_params) & (param in self.het_params):
                    # set up local parameters which are regionally variable based on groups
                    for group in self.node_groups:
                        param_name = f'{param}{group}'
                        self.free_params.append(param_name)
                        self.lb.append(v[0])
                        self.ub.append(v[1])
                else:
                    # is a global or bias parameter
                    self.free_params.append(param)
                    self.lb.append(v[0])
                    self.ub.append(v[1])
        self.fixed_params = list(set(self.params.keys()) - set(self.free_params))
        # set up map scaler parameters
        self.is_heterogeneous = False
        if self.maps_path is not None:
            self.is_heterogeneous = True
            self.maps = np.loadtxt(self.maps_path)
            scale_max_minmax = kwargs.pop('scale_max_minmax', 1)
            assert self.maps.shape[1] == self.sim_group.nodes, f"Maps second dimension {self.maps.shape[1]} != nodes {self.sim_group.nodes}"
            for param in self.het_params:
                for map_idx in range(self.maps.shape[0]):
                    # identify the scaler range
                    map_max = self.maps[map_idx, :].max()
                    map_min = self.maps[map_idx, :].min()
                    if ((map_min == 0) & (map_max == 1)):
                        # map is min-max normalized
                        scale_min = 0
                        scale_max = scale_max_minmax # defined in constants
                    elif ((map_min < 0) & (map_max > 0)):
                        # e.g. z-scored
                        scale_min = -1 / map_max
                        scale_min = np.ceil(scale_min / 0.1) * 0.1 # round up
                        scale_max = -1 / map_min
                        scale_max = np.floor(scale_max / 0.1) * 0.1 # round down 
                    # add the scaler as a free parameter
                    scaler_name = f"{param}scale{map_idx}"
                    self.free_params.append(scaler_name)
                    self.lb.append(scale_min)
                    self.ub.append(scale_max)
        # convert bounds to arrays
        self.lb = np.array(self.lb)
        self.ub = np.array(self.ub)
        # determine ndim of the problem
        self.ndim = len(self.free_params)
        # initialize pymoo Problem
        super().__init__(
            n_var=self.ndim, 
            n_obj=1, 
            n_ieq_constr=0, # TODO: consider using this for enforcing FIC success
            xl=np.zeros(self.ndim, dtype=float), 
            xu=np.ones(self.ndim, dtype=float)
        )

    def _get_Xt(self, X):
        """
        Transforms X from normalized [0, 1] range to [self.lb, self.ub]
        """
        return (X * (self.ub - self.lb)) + self.lb

    def _set_sim_params(self, X):
        # transform X from [0, 1] range to the actual
        # parameter range and label them
        Xt = pd.DataFrame(self._get_Xt(X), columns=self.free_params, dtype=float)
        print(Xt)
        # set fixed parameter lists
        # these are not going to vary across iterations
        # but must be specified here as in elsewhere N is unknown
        for param in self.fixed_params:
            if param in self.global_params:
                self.sim_group.param_lists[param] = np.repeat(self.params[param], self.sim_group.N)
            else:
                self.sim_group.param_lists[param] = np.tile(self.params[param], (self.sim_group.N, self.sim_group.nodes))
        # first determine the global parameters and bias terms
        for param in self.free_params:
            if param in self.global_params:
                self.sim_group.param_lists[param] = Xt.loc[:, param].values
            elif param in self.local_params:
                self.sim_group.param_lists[param] = np.tile(Xt.loc[:, param].values[:, np.newaxis], self.sim_group.nodes)
        # then multiply the local parameters by their map-based scalers
        if self.is_heterogeneous:
            for param in self.het_params:
                for sim_idx in range(Xt.shape[0]):
                    # not doing this vectorized for better code readability
                    # determine scaler map of current simulation-param
                    param_scalers = np.ones(self.sim_group.nodes)
                    for map_idx in range(self.maps.shape[0]):
                        scaler_name = f"{param}scale{map_idx}"
                        param_scalers += self.maps[map_idx, :] * Xt.iloc[sim_idx].loc[scaler_name]
                    # multiply it by the bias which is already put in param_lists
                    self.sim_group.param_lists[param][sim_idx, :] *= param_scalers
                    # fix/reject negatives
                    if (self.sim_group.param_lists[param][sim_idx, :].min() < 0.001):
                        print("Found negative local parameter...Shifting min to 0.001")
                        if self.reject_negative:
                            # TODO
                            raise NotImplementedError("Rejecting particles due to negative local parameter is not implemented")
                        else:
                            self.sim_group.param_lists[param][sim_idx, :] -= self.sim_group.param_lists[param][sim_idx, :].min() - 0.001
        # determine regional parameters that are variable based on groups
        if self.is_regional:
            for param in self.het_params:
                curr_param_maps = np.zeros((Xt.shape[0], self.sim_group.nodes))
                for group in self.node_groups:
                    param_name = f"{param}{group}"
                    curr_param_maps[:, self.memberships==group] = Xt.loc[:, param_name].values[:, np.newaxis]
                self.sim_group.param_lists[param] = curr_param_maps


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

