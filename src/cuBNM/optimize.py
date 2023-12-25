import os
import itertools
from abc import ABC, abstractmethod
import json
import pickle
import numpy as np
import pandas as pd
from pymoo.core.problem import Problem
from pymoo.algorithms.soo.nonconvex.cmaes import CMAES
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.termination import get_termination
import cma
import skopt

from cuBNM import sim


class GridSearch:
    """
    Simple grid search
    """

    # TODO: GridSearch should also be an Optimizer
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
            columns=param_ranges.keys(),
            dtype=float,
        )
        self.sim_group.N = self.param_combs.shape[0]
        for param in params.keys():
            self.sim_group.param_lists[param] = []
            for sim_idx in range(self.sim_group.N):
                if param in ["G", "v"]:
                    self.sim_group.param_lists[param].append(
                        self.param_combs.iloc[sim_idx].loc[param]
                    )
                else:
                    self.sim_group.param_lists[param].append(
                        np.repeat(
                            self.param_combs.iloc[sim_idx].loc[param],
                            self.sim_group.nodes,
                        )
                    )
            self.sim_group.param_lists[param] = np.array(
                self.sim_group.param_lists[param]
            )

    def evaluate(self, emp_fc_tril, emp_fcd_tril):
        self.sim_group.run()
        return self.sim_group.score(emp_fc_tril, emp_fcd_tril)


class RWWProblem(Problem):
    global_params = ["G", "v"]
    local_params = ["wEE", "wEI", "wIE"]

    def __init__(
        self,
        params,
        emp_fc_tril,
        emp_fcd_tril,
        het_params=[],
        maps_path=None,
        reject_negative=False,
        node_grouping=None,
        multiobj=False,
        **kwargs,
    ):
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
        multiobj: (bool)
            instead of combining the objectives into a single objective function
            (via summation) defines each objective separately. This must not be used
            with single-objective optimizers
        **kwargs to sim.SimGroup
        """
        # set opts
        self.params = params
        self.emp_fc_tril = emp_fc_tril
        self.emp_fcd_tril = emp_fcd_tril
        self.het_params = kwargs.pop("het_params", het_params)
        self.maps_path = kwargs.pop("maps_path", maps_path)
        self.reject_negative = kwargs.pop("reject_negative", reject_negative)
        self.node_grouping = kwargs.pop("node_grouping", node_grouping)
        self.multiobj = kwargs.pop("multiobj", multiobj)
        # initialize sim_group (N not known yet)
        self.sim_group = sim.SimGroup(**kwargs)
        # raise errors if opts are impossible
        if (self.sim_group.do_fic) & ("wIE" in self.het_params):
            raise ValueError(
                "wIE should not be specified as a heterogeneous parameter when FIC is done"
            )
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
                    raise ValueError(
                        f"{param} is set to be heterogeneous based on groups but no range is provided"
                    )
        # set up node_groups (ordered index of all unique groups)
        # and memberships (from node i to N, which group they belong to)
        if self.is_regional:
            if self.node_grouping == "node":
                # each node gets its own local free parameters
                # therefore each node has its own group and
                # is the only member of it
                self.node_groups = np.arange(self.sim_group.nodes)
                self.memberships = np.arange(self.sim_group.nodes)
            elif self.node_grouping == "sym":
                print(
                    "Warning: `sym` node grouping assumes symmetry of parcels between L and R hemispheres"
                )
                # nodes i & rh_idx+i belong to the same group
                # and will have similar parameters
                assert self.sim_group.nodes % 2 == 0, "Number of nodes must be even"
                rh_idx = int(self.sim_group.nodes / 2)
                self.node_groups = np.arange(rh_idx)
                self.memberships = np.tile(np.arange(rh_idx), 2)
            else:
                self.memberships = np.loadtxt(self.node_grouping).astype("int")
                self.node_groups = np.unique(self.memberships)
        # set up global and local (incl. bias) free parameters
        for param, v in params.items():
            if isinstance(v, tuple):
                if (
                    self.is_regional
                    & (param in self.local_params)
                    & (param in self.het_params)
                ):
                    # set up local parameters which are regionally variable based on groups
                    for group in self.node_groups:
                        param_name = f"{param}{group}"
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
            scale_max_minmax = kwargs.pop("scale_max_minmax", 1)
            assert (
                self.maps.shape[1] == self.sim_group.nodes
            ), f"Maps second dimension {self.maps.shape[1]} != nodes {self.sim_group.nodes}"
            for param in self.het_params:
                for map_idx in range(self.maps.shape[0]):
                    # identify the scaler range
                    map_max = self.maps[map_idx, :].max()
                    map_min = self.maps[map_idx, :].min()
                    if (map_min == 0) & (map_max == 1):
                        # map is min-max normalized
                        scale_min = 0
                        scale_max = scale_max_minmax  # defined in constants
                    elif (map_min < 0) & (map_max > 0):
                        # e.g. z-scored
                        scale_min = -1 / map_max
                        scale_min = np.ceil(scale_min / 0.1) * 0.1  # round up
                        scale_max = -1 / map_min
                        scale_max = np.floor(scale_max / 0.1) * 0.1  # round down
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
        # determine the number of objectives
        if self.multiobj:
            self.obj_names = []
            for term in self.sim_group.gof_terms:
                # these are cost values which are minimized by
                # the optimizers, therefore if the aim is to
                # maximize a measure (e.g. +fc_corr) its cost
                # will be -fc_corr
                if term.startswith("-"):
                    self.obj_names.append(term.replace("-", "+"))
                elif term.startswith("+"):
                    self.obj_names.append(term.replace("+", "-"))
            if self.sim_group.fic_penalty:
                self.obj_names.append("+fic_penalty")
        else:
            self.obj_names = ["cost"]
        n_obj = len(self.obj_names)
        # initialize pymoo Problem
        super().__init__(
            n_var=self.ndim,
            n_obj=n_obj,
            n_ieq_constr=0,  # TODO: consider using this for enforcing FIC success
            xl=np.zeros(self.ndim, dtype=float),
            xu=np.ones(self.ndim, dtype=float),
        )

    def get_config(self, include_sim_group=True, include_N=False):
        config = {
            "params": self.params,
            "het_params": self.het_params,
            "maps_path": self.maps_path,
            "reject_negative": self.reject_negative,
            "node_grouping": self.node_grouping,
        }
        if include_N:
            config["N"] = self.sim_group.N
        if include_sim_group:
            config.update(self.sim_group.get_config())
        return config

    def _get_Xt(self, X):
        """
        Transforms X from normalized [0, 1] range to [self.lb, self.ub]
        """
        return (X * (self.ub - self.lb)) + self.lb

    def _get_X(self, Xt):
        """
        Transforms Xt from [self.lb, self.ub] to [0, 1]
        """
        return (Xt - self.lb) / (self.ub - self.lb)

    def _set_sim_params(self, X):
        # transform X from [0, 1] range to the actual
        # parameter range and label them
        Xt = pd.DataFrame(self._get_Xt(X), columns=self.free_params, dtype=float)
        # set fixed parameter lists
        # these are not going to vary across iterations
        # but must be specified here as in elsewhere N is unknown
        for param in self.fixed_params:
            if param in self.global_params:
                self.sim_group.param_lists[param] = np.repeat(
                    self.params[param], self.sim_group.N
                )
            else:
                self.sim_group.param_lists[param] = np.tile(
                    self.params[param], (self.sim_group.N, self.sim_group.nodes)
                )
        # first determine the global parameters and bias terms
        for param in self.free_params:
            if param in self.global_params:
                self.sim_group.param_lists[param] = Xt.loc[:, param].values
            elif param in self.local_params:
                self.sim_group.param_lists[param] = np.tile(
                    Xt.loc[:, param].values[:, np.newaxis], self.sim_group.nodes
                )
        # then multiply the local parameters by their map-based scalers
        if self.is_heterogeneous:
            for param in self.het_params:
                for sim_idx in range(Xt.shape[0]):
                    # not doing this vectorized for better code readability
                    # determine scaler map of current simulation-param
                    param_scalers = np.ones(self.sim_group.nodes)
                    for map_idx in range(self.maps.shape[0]):
                        scaler_name = f"{param}scale{map_idx}"
                        param_scalers += (
                            self.maps[map_idx, :] * Xt.iloc[sim_idx].loc[scaler_name]
                        )
                    # multiply it by the bias which is already put in param_lists
                    self.sim_group.param_lists[param][sim_idx, :] *= param_scalers
                    # fix/reject negatives
                    if self.sim_group.param_lists[param][sim_idx, :].min() < 0.001:
                        if self.reject_negative:
                            # TODO
                            raise NotImplementedError(
                                "Rejecting particles due to negative local parameter is not implemented"
                            )
                        else:
                            self.sim_group.param_lists[param][sim_idx, :] -= (
                                self.sim_group.param_lists[param][sim_idx, :].min()
                                - 0.001
                            )
        # determine regional parameters that are variable based on groups
        if self.is_regional:
            for param in self.het_params:
                curr_param_maps = np.zeros((Xt.shape[0], self.sim_group.nodes))
                for group in self.node_groups:
                    param_name = f"{param}{group}"
                    curr_param_maps[:, self.memberships == group] = Xt.loc[
                        :, param_name
                    ].values[:, np.newaxis]
                self.sim_group.param_lists[param] = curr_param_maps

    def _evaluate(self, X, out, *args, **kwargs):
        """
        This is used by PyMoo optimizers
        """
        scores = self.eval(X)
        if "scores" in kwargs:
            kwargs["scores"].append(scores)
        if self.multiobj:
            out["F"] = -scores.loc[:, self.sim_group.gof_terms].values
            if self.sim_group.do_fic & self.sim_group.fic_penalty:
                out["F"] = np.concatenate(
                    [out["F"], -scores.loc[:, ["-fic_penalty"]].values], axis=1
                )
        else:
            if self.sim_group.do_fic & self.sim_group.fic_penalty:
                out["F"] = (
                    -scores.loc[:, "-fic_penalty"] - scores.loc[:, "+gof"]
                ).values
            else:
                out["F"] = -scores.loc[:, "+gof"].values
            # out["G"] = ... # TODO: consider using this for enforcing FIC success

    def eval(self, X):
        # set N to current iteration population size
        # which might be variable, e.g. from evaluating
        # CMAES initial guess to its next iterations
        self.sim_group.N = X.shape[0]
        self._set_sim_params(X)
        self.sim_group.run()
        return self.sim_group.score(self.emp_fc_tril, self.emp_fcd_tril)


class Optimizer(ABC):
    @abstractmethod
    def setup_problem(self, problem, **kwargs):
        pass

    @abstractmethod
    def optimize(self):
        pass

    def save(self, save_obj=False):
        """
        Saves the output of the optimizer, including history
        of particles, history of optima, the optimal point,
        and its simulation data

        Parameters
        ---------
        save_obj: (bool)
            saves the optimizer object which also includes the simulation
            data of all simulations and therefore can be large file
        """
        # specify the directory
        run_idx = 0
        while True:
            dirname = (
                type(self).__name__.lower().replace("optimizer", "") + f"_run-{run_idx}"
            )
            optimizer_dir = os.path.join(self.problem.sim_group.out_dir, dirname)
            if not os.path.exists(optimizer_dir):
                break
            else:
                run_idx += 1
        os.makedirs(optimizer_dir, exist_ok=True)
        # save the optimizer object
        if save_obj:
            with open(os.path.join(optimizer_dir, "optimizer.pickle"), "wb") as f:
                pickle.dump(self, f)
        # save the optimizer history
        self.history.to_csv(os.path.join(optimizer_dir, "history.csv"))
        if self.problem.n_obj == 1:
            self.opt_history.to_csv(os.path.join(optimizer_dir, "opt_history.csv"))
            self.opt.to_csv(os.path.join(optimizer_dir, "opt.csv"))
        # save the configs of simulations and optimizer
        with open(os.path.join(optimizer_dir, "problem.json"), "w") as f:
            json.dump(
                self.problem.get_config(include_sim_group=True, include_N=True),
                f,
                indent=4,
            )
        with open(os.path.join(optimizer_dir, "optimizer.json"), "w") as f:
            json.dump(self.get_config(), f, indent=4)
        # rerun optimum simulation(s) and save it
        print("Rerunning and saving the optimal simulation(s)")
        ## reuse the original problem used throughout the optimization
        ## but change its out_dir, N and parameters to the optimum
        self.problem.sim_group.out_dir = os.path.join(optimizer_dir, "opt_sim")
        os.makedirs(self.problem.sim_group.out_dir, exist_ok=True)
        res = self.algorithm.result()
        X = np.atleast_2d(res.X)
        Xt = pd.DataFrame(self.problem._get_Xt(X), columns=self.problem.free_params)
        opt_scores = self.problem.eval(X)
        pd.concat([Xt, opt_scores], axis=1).to_csv(
            os.path.join(self.problem.sim_group.out_dir, "res.csv")
        )
        ## save simulations data (including BOLD, FC and FCD, extended output, and param maps)
        self.problem.sim_group.save()
        # TODO: add option to save everything, including all the
        # SimGroup's of the different iterations

    def get_config(self):
        # TODO: add optimizer-specific get_config funcitons
        return {"n_iter": self.n_iter, "popsize": self.popsize, "seed": self.seed}


class PymooOptimizer(Optimizer):
    """
    General purpose wrapper for pymoo and other optimizers
    """

    def __init__(
        self, termination=None, n_iter=2, seed=0, print_history=True, **kwargs
    ):
        """
        Initialize pymoo optimizers by setting up the termination rule based on `n_iter` or `termination`

        Parameters:
        ----------
        termination : object, optional
            The termination object that defines the stopping criteria for the optimization process.
            If not provided, the termination criteria will be based on the number of iterations (`n_iter`).
        n_iter : int, optional
            The maximum number of iterations for the optimization process.
            This parameter is only used if `termination` is not provided.
        seed : int, optional
            The seed value for the random number generator used by the optimizer.
        print_history : bool, optional
            Flag indicating whether to print the optimization history during the optimization process.
        **kwargs : dict, optional
            Additional keyword arguments that can be passed to the optimizer.
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

    def setup_problem(self, problem, pymoo_verbose=False, **kwargs):
        """
        Sets up the problem with the algorithm.

        Parameters
        ----------
        problem : object
            The problem to be set up with the algorithm.
        pymoo_verbose : bool, optional
            Flag indicating whether to enable verbose output from pymoo. Default is False.
        **kwargs : dict
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
        self.history = []
        self.opt_history = []
        while self.algorithm.has_next():
            # ask the algorithm for the next solution to be evaluated
            pop = self.algorithm.ask()
            # evaluate the individuals using the algorithm's evaluator (necessary to count evaluations for termination)
            scores = []  # pass on an empty scores list to the evaluator to store the individual GOF measures
            self.algorithm.evaluator.eval(self.problem, pop, scores=scores)
            # returned the evaluated individuals which have been evaluated or even modified
            self.algorithm.tell(infills=pop)
            # TODO: do same more things, printing, logging, storing or even modifying the algorithm object
            X = np.array([p.x for p in self.algorithm.pop])
            Xt = pd.DataFrame(self.problem._get_Xt(X), columns=self.problem.free_params)
            Xt.index.name = "sim_idx"
            F = np.array([p.F for p in self.algorithm.pop])
            F = pd.DataFrame(F, columns=self.problem.obj_names)
            if self.problem.multiobj:
                # in multiobj optimizers simulations are reordered after scores are calculated
                # therefore don't concatenate scores to them
                # TODO: make sure this doesn't happen with the other single-objective optimizers
                # (other than CMAES)
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


class CMAESOptimizer(PymooOptimizer):
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
        Sets up a CMAES optimizer with some defaults

        Parameters
        ----------
        popsize : int
            The population size for the optimizer
        x0 : array-like, optional
            The initial guess for the optimization.
            If None (default), the initial guess will be estimated based on
            20 random samples as the first generation
        sigma : float, optional
            The initial step size for the optimization
        use_bound_penalty : bool, optional
            Whether to use a bound penalty for the optimization
        algorithm_kws : dict, optional
            Additional keyword arguments for the CMAES algorithm
        kwargs : dict
            Additional keyword arguments
        """
        super().__init__(**kwargs)
        self.max_obj = 1
        self.use_bound_penalty = (
            use_bound_penalty  # this option will be set after problem is set up
        )
        self.popsize = popsize
        self.algorithm = CMAES(
            x0=x0,  # will estimate the initial guess based on 20 random samples
            sigma=sigma,
            popsize=popsize,  # from second generation
            **algorithm_kws,
        )

    def setup_problem(self, problem, **kwargs):
        super().setup_problem(problem, **kwargs)
        self.algorithm.options["bounds"] = [0, 1]
        if self.use_bound_penalty:
            self.algorithm.options["BoundaryHandler"] = cma.BoundPenalty
        else:
            self.algorithm.options["BoundaryHandler"] = cma.BoundTransform


class NSGA2Optimizer(PymooOptimizer):
    def __init__(self, popsize, algorithm_kws={}, **kwargs):
        """
        Sets up a NSGA-II optimizer with some defaults
        """
        super().__init__(**kwargs)
        self.max_obj = 3
        self.popsize = popsize
        self.algorithm = NSGA2(pop_size=popsize, **algorithm_kws)


class BayesOptimizer(Optimizer):
    # this does not have the mechanics of PymooOptimizer but
    # uses similar API as much as possible
    def __init__(self, popsize, n_iter, seed=0):
        """
        Sets up a Bayesian optimizer
        """
        # does not initialize the optimizer yet because
        # the number of dimensions are not known yet
        # TODO: consider defining Problem before optimizer
        # and then passing it on to the optimizer
        self.max_obj = 1
        self.popsize = popsize
        self.n_iter = n_iter
        self.seed = seed

    def setup_problem(self, problem, **kwargs):
        """
        Initializes the algorithm based on problem ndim
        """
        self.problem = problem
        self.algorithm = skopt.Optimizer(
            dimensions=[skopt.space.Real(0.0, 1.0)] * self.problem.ndim,
            random_state=self.seed,
            base_estimator="gp",
            acq_func="LCB",  # TODO: see if this is the best option
            **kwargs,
        )

    def optimize(self):
        self.history = []
        self.opt_history = []
        for it in range(self.n_iter):
            # ask for next popsize suggestions
            X = np.array(self.algorithm.ask(n_points=self.popsize))
            # evaluate them
            out = {}
            self.problem._evaluate(X, out)
            # tell the results to the optimizer
            self.algorithm.tell(X.tolist(), out["F"].tolist())
            # print results
            res = pd.DataFrame(
                self.problem._get_Xt(X), columns=self.problem.free_params
            )
            res.index.name = "sim_idx"
            res["F"] = out["F"]
            print(res)
            res["gen"] = it + 1
            self.history.append(res)
            self.opt_history.append(res.loc[res["F"].argmin()])
        self.history = pd.concat(self.history, axis=0).reset_index(drop=False)
        self.opt_history = pd.DataFrame(self.opt_history).reset_index(drop=True)
        self.opt = self.opt_history.loc[self.opt_history["F"].argmin()]
