"""
Optimizers of the model free parameters
"""
import os
import itertools
import copy
from abc import ABC, abstractmethod
import json
import pickle
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from pymoo.core.problem import Problem
from pymoo.algorithms.soo.nonconvex.cmaes import CMAES
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.termination import get_termination
import cma

from cubnm import sim, utils

# define evaluation metric labels used in plots
METRIC_LABELS = {
    'cost': 'Cost',
    '-cost': '- Cost',
    '+gof': 'Goodness-of-fit',
    '+fc_corr': r'FC$_{corr}$',
    '-fcd_ks': r'- FCD$_{KS}$',
    '-fc_diff': r'- FC$_{diff}$',
    '-fc_normec': r'- FC$_{normEC}'
}

class BNMProblem(Problem):
    def __init__(
        self,
        model,
        params,
        emp_fc_tril=None,
        emp_fcd_tril=None,
        emp_bold=None,
        het_params=[],
        het_params_range='same',
        maps=None,
        maps_coef_range='auto',
        node_grouping=None,
        multiobj=False,
        **kwargs,
    ):
        """
        Brain network model problem. A :class:`pymoo.core.problem.Problem` 
        that defines the model, free parameters and their ranges, and target empirical
        data (FC and FCD), and the simulation configurations (through 
        :class:`cubnm.sim.SimGroup`). 
        :class:`cubnm.optimize.Optimizer` classes can be
        used to optimize the free parameters of this problem.

        Parameters
        ----------
        model: :obj:`str`, {'rWW', 'rWWEx', 'Kuramoto'}
            model name
        params: :obj:`dict` of :obj:`tuple` or :obj:`float`
            a dictionary including parameter names as keys and their
            fixed values (:obj:`float`) or continuous range of
            values (:obj:`tuple` of (min, max)) as values.
        emp_fc_tril: :obj:`np.ndarray` or :obj:`None`
            lower triangular part of empirical FC. Shape: (edges,)
        emp_fcd_tril: :obj:`np.ndarray` or :obj:`None`
            lower triangular part of empirical FCD. Shape: (window_pairs,)
        emp_bold: :obj:`np.ndarray` or None
            cleaned and parcellated empirical BOLD time series. Shape: (nodes, volumes)
            Motion outliers can either be excluded (not recommended as it disrupts
            the temporal structure) or replaced with zeros.
            If provided emp_fc_tril and emp_fcd_tril will be ignored.
        het_params: :obj:`list` of :obj:`str`
            which regional parameters are heterogeneous across nodes
        maps: :obj:`str`
            path to heterogeneity maps as a text file or a numpy array.
            Shape: (n_maps, nodes).
            If provided one free parameter per regional parameter per 
            each map will be added.
        maps_coef_range: 'auto' or :obj:`tuple` or :obj:`list` of :obj:`tuple`
            Range of coefficients for the maps in map-based heterogeneity 
            (i.e., when ``maps`` is provided).

            - ``'auto'``: uses (-1/max, -1/min) for maps with positive and negative
              values (assuming they are z-scored) and (0, 1) otherwise
            - :obj:`tuple`: uses the same range for all maps
            - :obj:`list` of :obj:`tuple`: n-map element list specifying the range
              of coefficients for each map

        het_params_range: 'same' or :obj:`dict` of :obj:`tuple` or None
            Forced range of regional parameters (after map-based
            heterogeneity is applied). The regional parameter values
            across nodes will be normalized into this range (if out
            of range).

            - ``'same'``: uses the same range as provided in `params`.
            - :obj:`dict` of :obj:`tuple`: uses the specified range
              for each regional parameter. The keys must be the all
              ``het_params`` and the values must be tuples of (min, max).
            - ``None``: does not normalize the regional parameters.
              This may lead to infeasible parameters, e.g. negative
              values, with some combinations of maps and map coeficients.

        node_grouping: {None, 'node', 'sym', :obj:`str`, :obj:`np.ndarray`}
            Defines groups of nodes which have the same regional
            parameters.

            - ``None``: does not use region-/group-specific parameters
            - ``'node'``: each node has its own regional free parameters
            - ``'sym'``: uses the same regional free parameters for each pair of symmetric nodes
              (e.g. L and R hemispheres). Assumes symmetry  of parcels between L and R
              hemispheres.
            - :obj:`str`: path to a text file including node grouping array. Shape: (nodes,)
            - :obj:`np.ndarray`: a numpy array. Shape: (nodes,)

        multiobj: :obj:`bool`
            instead of combining the objectives into a single objective function
            (via summation) defines each objective separately. This must not be used
            with single-objective optimizers
        **kwargs
            Keyword arguments passed to :class:`cubnm.sim.SimGroup`
        """
        # TODO: break down __init__ into smaller methods
        # set opts
        self.model = model 
        self.params = params
        self.emp_fc_tril = emp_fc_tril
        self.emp_fcd_tril = emp_fcd_tril
        self.emp_bold=emp_bold
        self.het_params = kwargs.pop("het_params", het_params)
        self.input_maps = kwargs.pop("maps", maps)
        self.maps_coef_range = kwargs.pop("maps_coef_range", maps_coef_range)
        self.het_params_range = kwargs.pop("het_params_range", het_params_range)
        self.node_grouping = kwargs.pop("node_grouping", node_grouping)
        self.multiobj = kwargs.pop("multiobj", multiobj)
        # initialize sim_group (N not known yet)
        sim_group_cls = getattr(sim, f"{self.model}SimGroup")
        self.sim_group = sim_group_cls(**kwargs)
        # calculate empirical FC and FCD if BOLD is provided
        # and do_fc/do_fcd are set to True
        if emp_bold is not None:
            if (emp_fc_tril is not None) or (emp_fcd_tril is not None):
                print(
                    "Warning: Both empirical BOLD and empirical FC/FCD are"
                    " provided. Empirical FC/FCD will be calculated based on"
                    " BOLD and will be overwritten."
                )
            if self.sim_group.do_fc:
                self.emp_fc_tril = utils.calculate_fc(
                    self.emp_bold, 
                    self.sim_group.exc_interhemispheric, 
                    return_tril=True)
            if self.sim_group.do_fcd:
                self.emp_fcd_tril = utils.calculate_fcd(
                    self.emp_bold, 
                    self.sim_group.window_size_TRs, 
                    self.sim_group.window_step_TRs, 
                    drop_edges=self.sim_group.fcd_drop_edges,
                    exc_interhemispheric=self.sim_group.exc_interhemispheric,
                    return_tril=True,
                    return_dfc = False
                )
        else:
            # when BOLD is not provided but empirical FC/FCD are
            # set do_fc and do_fcd based on the provided data
            if (self.emp_fc_tril is not None):
                self.sim_group.do_fc = True
                if (self.emp_fcd_tril is not None):
                    self.sim_group.do_fcd = True
            else:
                self.sim_group.do_fc = False
            if (self.emp_fcd_tril is not None):
                # note: as separate fc and fcd calculation is not
                # supported on simulation side, if fcd is provided
                # set do_fc to true as well (but then simulated fc
                # will be ignored)
                self.sim_group.do_fc = True
                self.sim_group.do_fcd = True
            else:
                self.sim_group.do_fcd = False
        # node grouping and input maps cannot be used together
        if (self.node_grouping is not None) & (self.input_maps is not None):
            raise ValueError("Both `node_grouping` and `maps` cannot be used")

        # identify free and fixed parameters
        self.free_params = []
        self.lb = []
        self.ub = []
        self.global_params = self.sim_group.global_param_names + ["v"]
        self.regional_params = self.sim_group.regional_param_names

        # decide if parameters can be variable across nodes/groups
        # note that this is different from map-based heterogeneity
        self.is_node_based = self.node_grouping is not None
        if self.is_node_based:
            for param in self.het_params:
                if not isinstance(params[param], tuple):
                    raise ValueError(
                        f"{param} is set to be heterogeneous based on groups but no range is provided"
                    )
        # set up node_groups (ordered index of all unique groups)
        # and memberships (from node i to N, which group they belong to)
        if self.is_node_based:
            if isinstance(self.node_grouping, (str, os.PathLike)):
                if self.node_grouping == "node":
                    # each node gets its own regional free parameters
                    # therefore each node has its own group and
                    # is the only member of it
                    self.memberships = np.arange(self.sim_group.nodes)
                elif self.node_grouping == "sym":
                    print(
                        "Warning: `sym` node grouping assumes symmetry of parcels between L and R hemispheres"
                    )
                    # nodes i & rh_idx+i belong to the same group
                    # and will have similar parameters
                    assert self.sim_group.nodes % 2 == 0, "Number of nodes must be even"
                    rh_idx = int(self.sim_group.nodes / 2)
                    self.memberships = np.tile(np.arange(rh_idx), 2)
                else:
                    self.memberships = np.loadtxt(self.node_grouping).astype("int")
            elif isinstance(self.node_grouping, np.ndarray):
                self.memberships = self.node_grouping.astype("int")
            else:
                raise ValueError(
                    "Invalid node_grouping provided."
                )
            self.node_groups = np.unique(self.memberships)

        # set up global and regional (incl. bias) free parameters
        for param, v in params.items():
            if isinstance(v, tuple):
                if (
                    self.is_node_based
                    & (param in self.regional_params)
                    & (param in self.het_params)
                ):
                    # set up regional parameters which are regionally variable based on groups
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
        self.is_map_based = False
        if self.input_maps is not None:
            self.is_map_based = True
            # load maps
            if isinstance(self.input_maps, (str, os.PathLike)):
                self.maps = np.loadtxt(self.input_maps)
            else:
                self.maps = self.input_maps
            assert (
                self.maps.shape[1] == self.sim_group.nodes
            ), f"Maps second dimension {self.maps.shape[1]} != nodes {self.sim_group.nodes}"

            # define map coefficients as free parameter
            for param in self.het_params:
                for map_idx in range(self.maps.shape[0]):
                    if self.maps_coef_range == 'auto':
                        # identify the scaler range
                        map_max = self.maps[map_idx, :].max()
                        map_min = self.maps[map_idx, :].min()
                        if (map_min < 0) & (map_max > 0):
                            # e.g. z-scored
                            scale_min = -1 / map_max
                            scale_min = np.ceil(scale_min / 0.1) * 0.1  # round up
                            scale_max = -1 / map_min
                            scale_max = np.floor(scale_max / 0.1) * 0.1  # round down
                        else:
                            scale_min = 0.0
                            scale_max = 1.0
                    elif isinstance(self.maps_coef_range, tuple):
                        scale_min, scale_max = self.maps_coef_range
                    elif isinstance(self.maps_coef_range, list):
                        scale_min, scale_max = self.maps_coef_range[map_idx]
                    else:
                        raise ValueError("Invalid maps_coef_range provided")
                    # add the scaler as a free parameter
                    scaler_name = f"{param}scale{map_idx}"
                    self.free_params.append(scaler_name)
                    self.lb.append(scale_min)
                    self.ub.append(scale_max)

            # define regional ranges
            if self.het_params_range == "same":
                self._het_params_range = {}
                for param in self.het_params:
                    # use the same range as provided in params
                    if param in self.params:
                        if isinstance(self.params[param], tuple):
                            self._het_params_range[param] = self.params[param]
                        else:
                            raise ValueError(
                                f"Parameter {param} baseline cannot be fixed"
                                " when map-based heterogeneity is applied and"
                                " het_params_range is set to 'same'"
                            )
                    else:
                        raise ValueError(
                            f"Parameter {param} is not defined in params"
                        )
            elif isinstance(self.het_params_range, dict):
                assert all(
                    p in self.het_params_range for p in self.het_params
                ), "All het_params must be defined in het_params_range"
                self._het_params_range = self.het_params_range
            elif self.het_params_range is None:
                # do not normalize the regional parameters
                self._het_params_range = None

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
        else:
            self.obj_names = ["cost"]
        self.n_obj = len(self.obj_names)
        # model-specific initialization of problem
        # (e.g. including FIC penalty in rWW cost function)
        self.sim_group._problem_init(self)
        # initialize pymoo Problem
        super().__init__(
            n_var=self.ndim,
            n_obj=self.n_obj,
            n_ieq_constr=0,  # TODO: consider using this for enforcing FIC success
            xl=np.zeros(self.ndim, dtype=float),
            xu=np.ones(self.ndim, dtype=float),
        )

    def get_config(self, include_sim_group=True, include_N=False):
        """
        Get the problem configuration

        Parameters
        ----------
        include_sim_group: :obj:`bool`
            whether to include the configuration of the
            associated :class:`cubnm.sim.SimGroup`
        include_N: :obj:`bool`
            whether to include the current population size
            in the configuration

        Returns
        -------
        :obj:`dict`
            the configuration of the problem
        """
        config = {
            "model": self.model,
            "params": self.params,
            "het_params": self.het_params,
            "maps": self.input_maps,
            "node_grouping": self.node_grouping,
            "maps_coef_range": self.maps_coef_range,
            "het_params_range": self.het_params_range,
            "emp_fc_tril": self.emp_fc_tril,
            "emp_fcd_tril": self.emp_fcd_tril,
            "emp_bold": self.emp_bold,
        }
        if include_N:
            config["N"] = self.sim_group.N
        if include_sim_group:
            config.update(self.sim_group.get_config())
        return config

    def _get_Xt(self, X):
        """
        Transforms normalized parameters in range [0, 1] to
        the actual parameter ranges

        Parameters
        ----------
        X: :obj:`np.ndarray`
            the normalized parameters of current population. 
            Shape: (N, ndim)

        Returns
        -------
        :obj:`np.ndarray`
            the transformed parameters of current population. 
            Shape: (N, ndim)
        """
        return (X * (self.ub - self.lb)) + self.lb

    def _get_X(self, Xt):
        """
        Normalizes parameters to range [0, 1]

        Parameters
        ----------
        Xt: :obj:`np.ndarray`
            the parameters of current population. 
            Shape: (N, ndim)

        Returns
        -------
        :obj:`np.ndarray`
            the normalized parameters current population. 
            Shape: (N, ndim)
        """
        return (Xt - self.lb) / (self.ub - self.lb)

    def _get_sim_params(self, X):
        """
        Gets the global and regional parameters of the problem's 
        :class:`cubnm.sim.SimGroup` based on the
        problem free and fixed parameters and type of regional parameter
        heterogeneity (map-based, group-based or none).

        Parameters
        ----------
        X: :obj:`np.ndarray`
            the normalized parameters of current population in range [0, 1]. 
            Shape: (N, ndim)
        Returns
        -------
        :obj:`dict` of :obj:`np.ndarray`
            keys correspond to model parameters
        """
        param_lists = {}
        N = X.shape[0]
        # transform X from [0, 1] range to the actual
        # parameter range and label them
        Xt = pd.DataFrame(self._get_Xt(X), columns=self.free_params, dtype=float)
        # set fixed parameter lists
        # these are not going to vary across iterations
        # but must be specified here as in elsewhere N is unknown
        for param in self.fixed_params:
            if param in self.global_params:
                param_lists[param] = np.repeat(
                    self.params[param], N
                )
            else:
                param_lists[param] = np.tile(
                    self.params[param], (N, self.sim_group.nodes)
                )
        # first determine the global parameters and bias terms
        for param in self.free_params:
            if param in self.global_params:
                param_lists[param] = Xt.loc[:, param].values
            elif param in self.regional_params:
                param_lists[param] = np.tile(
                    Xt.loc[:, param].values[:, np.newaxis], self.sim_group.nodes
                )
        # then multiply the regional parameters by their map-based scalers
        if self.is_map_based:
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
                    param_lists[param][sim_idx, :] *= param_scalers
                    # fix out-of-range parameter values if indicated
                    if self._het_params_range is not None:
                        # min-max normalize regional parameters to the
                        # provided range, if the map-based heterogeneity
                        # has led to out-of-range values (in either end)
                        min_lim, max_lim = self._het_params_range[param]
                        uncorr_min = param_lists[param][sim_idx, :].min()
                        uncorr_max = param_lists[param][sim_idx, :].max()
                        # Note that if one end is within range we want
                        # to keep it as is, therefore we define the target
                        # min and max to be the limits, or the within-range
                        # min or max values (whichever is smaller or larger)
                        # We also make sure that the target min is not larger than
                        # the target max and vice versa. As a result, 
                        # if both ends are outside the range, the parameter
                        # map will effecitvely be homogeneously set to the
                        # target min or max
                        target_min = min(max(min_lim, uncorr_min), max_lim)
                        target_max = max(min(max_lim, uncorr_max), min_lim)
                        # normalization done only if at least one of the ends is out of range
                        if ((uncorr_min < target_min) or 
                            (uncorr_max > target_max)):
                            # normalize the regional parameters to the provided range
                            # by first min-max normalizing them to [0, 1]
                            # and then scaling to the target range
                            param_lists[param][sim_idx, :] = (
                                (param_lists[param][sim_idx, :] - uncorr_min)
                                / (uncorr_max - uncorr_min)
                            ) * (target_max - target_min) + target_min

        # determine regional parameters that are variable based on groups
        if self.is_node_based:
            for param in self.het_params:
                curr_param_maps = np.zeros((Xt.shape[0], self.sim_group.nodes))
                for group in self.node_groups:
                    param_name = f"{param}{group}"
                    curr_param_maps[:, self.memberships == group] = Xt.loc[
                        :, param_name
                    ].values[:, np.newaxis]
                param_lists[param] = curr_param_maps
        return param_lists

    def _set_sim_params(self, X):
        """
        Sets the global and regional parameters of the problem's 
        :class:`cubnm.sim.SimGroup` based on the
        problem free and fixed parameters and type of regional parameter
        heterogeneity (map-based, group-based or none).

        Parameters
        ----------
        X: :obj:`np.ndarray`
            the normalized parameters of current population in range [0, 1]. 
            Shape: (N, ndim)
        """
        self.sim_group.param_lists.update(self._get_sim_params(X))

    def _evaluate(self, X, out, *args, **kwargs):
        """
        Ovewrites the :meth:`pymoo.core.problem.Problem._evaluate` method
        to evaluate the goodness of fit of the simulations based on parameters
        `X` and store the results in `out`.

        Parameters
        ----------
        X: :obj:`np.ndarray`
            the normalized parameters of current population in range [0, 1]. 
            Shape: (N, ndim)
        out: :obj:`dict`
            the output dictionary to store the results with keys ``'F'`` and ``'G'``.
            Currently only ``'F'`` (cost) is used.
        *args, **kwargs
            additional arguments passed to the evaluation function,
            which may include:

            - scores: :obj:`list`
                an empty list passed on to evaluation function to store
                the individual goodness of fit measures
            - skip_run: :obj:`bool`
                will only be true in batch optimization where the simulations
                are already run and only the GOF calculation is needed
        """
        skip_run = kwargs.pop("skip_run", False)
        scores = self.eval(X, skip_run=skip_run)
        kwargs["scores"].append(scores)
        if self.multiobj:
            out["F"] = -scores.loc[:, self.sim_group.gof_terms].values
        else:
            out["F"] = -scores.loc[:, "+gof"].values
        # extend cost function by model-specific costs (e.g. FIC penalty in rWW)
        self.sim_group._problem_evaluate(self, X, out, *args, **kwargs)

    def eval(self, X, skip_run=False):
        """
        Runs the simulations based on normalized candidate free
        parameters `X` and evaluates their goodness of fit to
        the empirical FC and FCD of the problem.

        Parameters
        ----------
        X: :obj:`np.ndarray`
            the normalized parameters of current population in range [0, 1]. 
            Shape: (N, ndim)
        skip_run: :obj:`bool`
            will only be true in batch optimization where the simulations
            are already run and only the GOF calculation is needed

        
        Returns
        -------
        :obj:`pd.DataFrame`
            The goodness of fit measures (columns) of each simulation (rows)
        """
        # set N to current iteration population size
        # which might be variable, e.g. from evaluating
        # CMAES initial guess to its next iterations
        self.sim_group.N = X.shape[0]
        self._set_sim_params(X)
        if not skip_run:
            self.sim_group.run()
        return self.sim_group.score(self.emp_fc_tril, self.emp_fcd_tril)


class Optimizer(ABC):
    @abstractmethod
    def __init__(self, **kwargs):
        # defining __init__ as abstractmethod
        # for the docs to conform with subclasses
        """
        Base class for evolutionary optimizers
        """
        pass

    @abstractmethod
    def optimize(self):
        pass

    def save(self, save_opt=True, save_obj=False):
        """
        Saves the output of the optimizer, including history
        of particles, history of optima, the optimal point,
        and its simulation data. The output will be saved
        to `out_dir` of the problem's :class:`cubnm.sim.SimGroup`.
        If a directory with the same type of optimizer already
        exists, a new directory with a new index will be created.

        Parameters
        ---------
        save_opt: :obj:`bool`
            reruns and saves the optimal simulation(s) data
        save_obj: :obj:`bool`
            saves the optimizer object which also includes the simulation
            data of all simulations and therefore can be large file.
            Warning: this file is very large.
        """
        assert self.is_fit, (
            "Optimizer must be fitted before saving. "
            "Call `optimize()` method to fit the optimizer."
        )
        # specify the directory
        # TODO: avoid saving if an identical optimizer directory already exists
        # by checking the optimizer and problem jsons in previous directories
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
            with open(os.path.join(optimizer_dir, "optimizer.pkl"), "wb") as f:
                pickle.dump(self, f)
        # save the optimizer history
        self.history.to_csv(os.path.join(optimizer_dir, "history.csv"))
        if self.problem.n_obj == 1:
            self.opt.to_csv(os.path.join(optimizer_dir, "opt.csv"))
            if type(self).__name__ != "GridOptimizer":
                self.opt_history.to_csv(os.path.join(optimizer_dir, "opt_history.csv"))
        # save the configs of simulations and optimizer
        problem_config = self.problem.get_config(include_sim_group=True, include_N=True)
        ## problem config might include numpy arrays (for sc, sc_dist, etc.)
        ## which cannot be saved in the json file. If that's the case save
        ## these arrays as txt files within the optimizer directory and
        ## save their path instead of the arrays
        ## Also convert all relative paths to absolute paths
        for k, v in problem_config.items():
            if isinstance(v, np.ndarray):
                v_path = os.path.join(optimizer_dir, k+".txt")
                np.savetxt(v_path, v)
                problem_config[k] = v_path
            # convert relative to absoulte paths
            if (
                isinstance(problem_config[k], (str, os.PathLike)) and 
                os.path.exists(problem_config[k])
            ):
                problem_config[k] = os.path.abspath(problem_config[k])
        with open(os.path.join(optimizer_dir, "problem.json"), "w") as f:
            json.dump(
                problem_config,
                f,
                indent=4,
            )
        with open(os.path.join(optimizer_dir, "optimizer.json"), "w") as f:
            json.dump(self.get_config(), f, indent=4)
        if save_opt:
            # (rerun) optimum simulation(s) and save it
            opt_sim_dir = os.path.join(optimizer_dir, "opt_sim")
            os.makedirs(opt_sim_dir, exist_ok=True)
            if type(self).__name__ == "GridOptimizer":
                print("Saving the optimal simulation")
                ## slice sim_group at the index of the optimal simulation
                opt_idx = self.opt.name
                self.problem.opt_sim_group = self.problem.sim_group.slice(opt_idx, inplace=False)
            else:
                print("Rerunning and saving the optimal simulation(s)")
                ## reuse the original problem used throughout the optimization
                ## but change its out_dir, N and parameters to the optimum
                res = self.algorithm.result()
                X = np.atleast_2d(res.X)
                Xt = pd.DataFrame(self.problem._get_Xt(X), columns=self.problem.free_params)
                opt_scores = self.problem.eval(X)
                pd.concat([Xt, opt_scores], axis=1).to_csv(
                    os.path.join(opt_sim_dir, "scores.csv")
                )
                self.problem.opt_sim_group = self.problem.sim_group
            ## save simulations data (including BOLD, FC and FCD, extended output, and param maps)
            self.problem.opt_sim_group.out_dir = opt_sim_dir
            self.problem.opt_sim_group.save()
        return optimizer_dir

    def get_config(self):
        """
        Get the optimizer configuration

        Returns
        -------
        :obj:`dict`
            the configuration of the optimizer
        """
        # TODO: add optimizer-specific get_config funcitons
        return {"n_iter": self.n_iter, "popsize": self.popsize, "seed": self.seed}

    def _plot_space_3d(
            self, 
            plot_df, 
            measure, 
            title='default', 
            opt_params=None, 
            config={}, 
            ax=None
        ):
        """
        Plot 3D parameter space colored by `measure`
        """
        # set plotting options based on defaults
        # and options provided by user
        _config = dict(
            figsize = (4.5, 4.5),
            elev = 15,
            azim = -15,
            roll = None,
            aspects = (1, 1, 1),
            zoom = 0.85,
            size = 30,
            alpha = 0.4,
            cmap = 'viridis',
            vmin = None,
            vmax = None,
            color = 'red',
            opt_facecolor = 'none',
            opt_edgecolor = 'black',
            full_lim = True,
        )
        _config.update(config)
        if ax is None:
            # create a figure with 3d projection
            fig, ax = plt.subplots(figsize=_config['figsize'], subplot_kw={'projection': '3d'})
        # plot grid
        if measure is None:
            c = _config['color']
            cmap = None
        else:
            c = plot_df.loc[:, measure]
            cmap = _config['cmap']
        ax.scatter(
            plot_df.loc[:, self.problem.free_params[0]],
            plot_df.loc[:, self.problem.free_params[1]],
            plot_df.loc[:, self.problem.free_params[2]],
            s=_config['size'],
            alpha=_config['alpha'],
            c=c,
            cmap=cmap,
            vmin=_config['vmin'],
            vmax=_config['vmax'],
        )
        # mark the optimum if indicated
        if opt_params is not None:
            ax.scatter(
                *opt_params.values,
                alpha=1.0,
                s=_config['size'],
                facecolors=_config['opt_facecolor'],
                edgecolors=_config['opt_edgecolor'],
            )
        # set the x y z lim
        if _config['full_lim']:
            ax.set_xlim(self.problem.lb[0], self.problem.ub[0])
            ax.set_ylim(self.problem.lb[1], self.problem.ub[1])
            ax.set_zlim(self.problem.lb[2], self.problem.ub[2])
        # aesthetics
        labels = [self.problem.sim_group.labels.get(p, p) for p in self.problem.free_params]
        ax.set_xlabel(labels[0])
        ax.set_ylabel(labels[1])
        ax.set_zlabel(labels[2])
        ax.view_init(elev=_config['elev'], azim=_config['azim'], roll=_config['roll'])
        ax.set_box_aspect(_config['aspects'], zoom=_config['zoom'])
        if title == 'default':
            if measure in self.problem.sim_group.state_names:
                title = self.problem.sim_group.labels.get(measure, measure)
            else:
                title = METRIC_LABELS.get(measure, measure)
        if title is not None:
            ax.set_title(title)
        return ax

    def _plot_space_2d(
            self, 
            plot_df, 
            measure, 
            title='default', 
            opt_params=None, 
            config={}, 
            ax=None
        ):
        """
        Plot 2D parameter space colored by `measure`
        """
        # set plotting options based on defaults
        # and options provided by user
        _config = dict(
            figsize = (4.5, 4.5),
            size = 30,
            marker = 'o',
            alpha = 1.0,
            cmap = 'viridis',
            vmin = None,
            vmax = None,
            color = 'red',
            opt_facecolor = 'none',
            opt_edgecolor = 'black',
            full_lim = True,
        )
        _config.update(config)
        # plot
        if ax is None:
            fig, ax = plt.subplots(figsize=_config['figsize'])
        if measure is None:
            c = _config['color']
            cmap = None
        else:
            c = plot_df.loc[:, measure]
            cmap = _config['cmap']
        ax.scatter(
            plot_df.loc[:, self.problem.free_params[0]],
            plot_df.loc[:, self.problem.free_params[1]],
            s=_config['size'],
            marker=_config['marker'],
            c=c,
            cmap=cmap,
            vmin=_config['vmin'],
            vmax=_config['vmax'],
        )
        # mark the optimum if indicated
        if opt_params is not None:
            ax.scatter(
                *opt_params.values,
                alpha=1.0,
                s=_config['size'],
                marker=_config['marker'],
                facecolors=_config['opt_facecolor'],
                edgecolors=_config['opt_edgecolor'],
            )
        # set the x y lim
        if _config['full_lim']:
            # TODO: fix the edge simulations in grid which are cut
            ax.set_xlim(self.problem.lb[0], self.problem.ub[0])
            ax.set_ylim(self.problem.lb[1], self.problem.ub[1])
        # aesthetics
        labels = [self.problem.sim_group.labels.get(p, p) for p in self.problem.free_params]
        ax.set_xlabel(labels[0])
        ax.set_ylabel(labels[1])
        sns.despine(ax=ax)
        if title == 'default':
            if measure in self.problem.sim_group.state_names:
                title = self.problem.sim_group.labels.get(measure, measure)
            else:
                title = METRIC_LABELS.get(measure, measure)
        if title is not None:
            ax.set_title(title)
        return ax

    def _plot_space_1d(
            self, 
            plot_df, 
            measure, 
            title='default', 
            opt_params=None, 
            config={}, 
            ax=None
        ):
        """
        Plot 1D parameter space colored by `measure`
        """
        # set plotting options based on defaults
        # and options provided by user
        _config = dict(
            figsize = None,
            size = 30,
            marker = 'o',
            alpha = 1.0,
            color = 'red',
            opt_facecolor = 'none',
            opt_edgecolor = 'black',
            full_lim = True,
        )
        _config.update(config)
        # plot
        if ax is None:
            fig, ax = plt.subplots(figsize=_config['figsize'])
        if measure is None:
            y = 1
            if opt_params is not None:
                opt_y = 1
        else:
            y = plot_df.loc[:, measure]
            if opt_params is not None:
                opt_y = plot_df.loc[
                    plot_df.loc[:,self.problem.free_params[0]]==\
                    opt_params.values[0], measure
                ]
        ax.scatter(
            plot_df.loc[:, self.problem.free_params[0]],
            y,
            s=_config['size'],
            marker=_config['marker'],
            c=_config['color'],
        )
        # mark the optimum if indicated
        if opt_params is not None:
            ax.scatter(
                opt_params.values[0],
                opt_y,
                alpha=1.0,
                s=_config['size'],
                marker=_config['marker'],
                facecolors=_config['opt_facecolor'],
                edgecolors=_config['opt_edgecolor'],
            )
        # set the x lim
        if _config['full_lim']:
            # TODO: fix the edge simulations in grid which are cut
            ax.set_xlim(self.problem.lb[0], self.problem.ub[0])
        # aesthetics
        labels = [self.problem.sim_group.labels.get(p, p) for p in self.problem.free_params]
        ax.set_xlabel(labels[0])
        sns.despine(ax=ax)
        if title == 'default':
            if measure in self.problem.sim_group.state_names:
                title = self.problem.sim_group.labels.get(measure, measure)
            else:
                title = METRIC_LABELS.get(measure, measure)
        if title is not None:
            ax.set_ylabel(title)
        return ax

    def plot_space(
            self, 
            measure=None, 
            title='default', 
            opt=False, 
            gen=None, 
            config={}, 
            ax=None
        ):
        """
        Plot parameter space colored by ``measure``. Can only be used
        for 1D, 2D and 3D parameter spaces.

        Parameters
        ----------
        measure: :obj:`str` or :obj:`None`
            the measure to color the points by. If None, the points
            will be colored by the ``config['color']`` parameter and
            have the same color
        title: :obj:`str`
            the title of the plot. If 'default', the title will be
            set to the measure clean label. If None, no title will be set.
        opt: :obj:`bool`
            whether to mark the optimum in the plot. Will
            be ignored if ``gen`` is provided.
        gen: :obj:`int` or :obj:`None`
            the generation to plot. If None, the entire history
            will be plotted. When using a :class:`cubnm.optimize.GridOptimizer`
            this is ignored.
        config: :obj:`dict`
            plotting configuration. The following keys are available
            with default values:
            
            - figsize: :obj:`tuple`, (4.5, 4.5) for 2D and 3D plots, None for 1D
                figure size. Ignored if ``ax`` is provided
            - size: :obj:`float`, 30
                size of the sphere
            - alpha: :obj:`float`, 0.2 
                transparency of the sphere
            - color: :obj:`str`, 'red'
                color of the points if measure is None or space is 1D
            - opt_facecolor: :obj:`str`, 'none'
                face color of the optimum point
            - opt_edgecolor: :obj:`str`, 'black'
                edge color of the optimum point
            - full_lim: :obj:`bool`, True
                whether to set the axes limits to the full range
                of the parameters defined by the problem

            Specific to 2D and 3D plots:
            
            - cmap: :obj:`str`, 'viridis'
                colormap to use for the measure
            - vmin: :obj:`float`, None
                minimum value of the color range
            - vmax: :obj:`float`, None
                maximum value of the color range

            Specific to 1D and 2D plots:

            - marker: :obj:`str`, 'o'
            
            Specific to 3D plots:

            - elev: :obj:`float`, 15
                elevation angle in degrees
            - azim: :obj:`float`, -15
                azimuth angle in degrees
            - roll: :obj:`float` or :obj:`None`, None
                roll angle in degrees
            - aspects: :obj:`tuple`, (1, 1, 1)
                aspect ratio of the axes
            - zoom: :obj:`float`, 0.85
                zoom level of the plot

        ax: :obj:`matplotlib.axes.Axes`
            the axes to plot on. For 3D must have
            ``projection='3d'``.
            If None, a new figure and axes will be created. 

        Returns
        -------
        :obj:`tuple`
            the axes of the plot
        """
        assert self.is_fit, (
            "Cannot plot parameter space before the optimizer is fit. "
            "Call `optimize` method first."
        )
        # check if measure can be plotted
        if measure in self.problem.sim_group.state_names:
            if not isinstance(self, GridOptimizer):
                raise NotImplementedError(
                    "Plotting state variables is not supported "
                    "using evolutionary optimizaters"
                )
        ndim = len(self.problem.free_params)
        if ndim > 3:
            raise ValueError("Cannot plot samples in >3D parameter spaces")
        # prepare plotting data
        # (limit history to a given generation
        # if indicated)
        history = self.history
        if (gen is not None) and (not isinstance(self, GridOptimizer)):
            history = history.loc[history['gen']==gen]
        if measure in self.problem.sim_group.state_names:
            state_averages = self.problem.sim_group.get_state_averages()
            plot_df = pd.concat([
                history.loc[:,self.problem.free_params], 
                state_averages.loc[:, measure]
            ], axis=1)
        elif measure == '-cost':
            plot_df = history.set_index(self.problem.free_params)['cost'].reset_index()
            plot_df.loc[:, '-cost'] = -plot_df.loc[:, 'cost']
        elif measure is not None:
            plot_df = history.set_index(self.problem.free_params)[measure].reset_index()
        else:
            plot_df = history.loc[:,self.problem.free_params]
        if opt and (gen is None):
            opt_params = self.opt[self.problem.free_params]
        else:
            opt_params = None
        # plot based on the number of dimensions
        if ndim == 3:
            return self._plot_space_3d(plot_df, measure, title, opt_params, config, ax=ax)
        elif ndim == 2:
            return self._plot_space_2d(plot_df, measure, title, opt_params, config, ax=ax)
        else:
            return self._plot_space_1d(plot_df, measure, title, opt_params, config, ax=ax)

    def plot_history(self, measure, legend=True, ax=None, line_kws={}, scatter_kws={}):
        """
        Plot the history of ``measure`` across the optimization generations.

        Parameters
        ----------
        measure: :obj:`str`
            the measure to plot
        legend: :obj:`bool`
            whether to show the legend
        ax: :obj:`matplotlib.axes.Axes`
            the axes to plot on. If None, a new figure and axes will be created.
        line_kws: :obj:`dict`
            additional keyword arguments passed to the line plot
            of the median across generations
        scatter_kws: :obj:`dict`
            additional keyword arguments passed to the scatter plot
            of the individual particles across generations
        """
        assert not isinstance(self, GridOptimizer), (
            "Not applicable to GridOptimizer."
        )
        assert self.is_fit, (
            "Cannot plot history before the optimizer is fit. "
            "Call `optimize` method first."
        )
        # calculate negative cost
        plot_data = self.history.copy()
        if measure == '-cost':
            plot_data['-cost'] = -plot_data['cost']
        # calculate median in each generation
        median = plot_data.groupby("gen")[measure].median()
        # plot
        if ax is None:
            fig, ax = plt.subplots(figsize=(6, 4))
        # median
        _line_kws = dict(
            color = 'red',
            alpha = 0.5,
        )
        _line_kws.update(line_kws)
        ax.plot(median, label='Median', **_line_kws)
        # individual particles
        _scatter_kws = dict(
            s = 4,
            alpha = 0.5,
            color = 'black',
        )
        _scatter_kws.update(scatter_kws)
        # TODO: consider using matplotlib for all
        # plotting to avoid dependency on seaborn
        sns.scatterplot(data=plot_data, x='gen', y=measure, label='Particles', ax=ax, **_scatter_kws)
        # aesthetics
        ax.set_ylabel(METRIC_LABELS.get(measure, measure))
        ax.set_xlabel('Generation')
        if legend:
            ax.legend()
        return ax

class GridOptimizer(Optimizer):
    def __init__(self, **kwargs):
        """
        Grid search optimizer

        Example
        -------
        Run a grid search of rWW model with 10 G and 10 wEE values
        with fixed wEI: ::

            from cubnm import datasets, optimize

            problem = optimize.BNMProblem(
                model = 'rWW',
                params = {
                    'G': (0.5, 2.5),
                    'wEE': (0.05, 0.75),
                    'wEI': 0.21,
                },
                duration = 60,
                TR = 1,
                window_size=10,
                window_step=2,
                sc = datasets.load_sc('strength', 'schaefer-100'),
                emp_bold = datasets.load_bold('schaefer-100'),
            )
            go = optimize.GridOptimizer()
            go.optimize(problem, grid_shape=10)
            go.save()
        """
        pass

    def optimize(self, problem, grid_shape):
        """
        Runs a grid search optimization on the given problem.

        Parameters
        ----------
        problem: :obj:`cubnm.optimizer.BNMProblem`
            The problem to be set up with the algorithm.

        grid_shape: :obj:`int` or :obj:`dict`
            Shape of the grid search. If an integer is provided
            the same number of points are used for each parameter. 
            If a dictionary is provided, the keys should be the 
            parameter names and the values should be the number of points 
            within the range of each parameter.
        """
        self.problem = problem
        if isinstance(grid_shape, int):
            grid_shape = {param: grid_shape for param in self.problem.free_params}
        self.grid_shape = grid_shape
        param_ranges = {}
        for free_param in self.problem.free_params:
            param_ranges[free_param] = np.linspace(0, 1, grid_shape[free_param])
        X = np.array(list(itertools.product(*param_ranges.values())))
        self.popsize = X.shape[0]
        out = {} # will include 'F' (cost)
        scores = [] # will include individual GOF measures
        self.problem._evaluate(X, out, scores=scores)
        scores = scores[0]
        scores['cost'] = out['F']
        Xt = pd.DataFrame(self.problem._get_Xt(X), columns=self.problem.free_params)
        self.history = pd.concat([Xt, scores], axis=1)
        self.opt = self.history.loc[self.history["cost"].idxmin()]
        self.is_fit = True

    def save(self, save_avg_states=True, **kwargs):
        """
        Saves the output of the optimizer, including history
        of particles, history of optima, the optimal point,
        and its simulation data. The output will be saved
        to `out_dir` of the problem's :class:`cubnm.sim.SimGroup`.

        Parameters
        ---------
        save_avg_states: :obj:`bool`
            saves the average states of all simulations
        **kwargs
            additional keyword arguments passed to the :meth:`cubnm.optimizer.Optimizer.save`
        """
        optimizer_dir = super().save(**kwargs)
        # save the average states of the optimal simulation
        if save_avg_states:
            state_averages = self.problem.sim_group.get_state_averages()
            state_averages.to_csv(os.path.join(optimizer_dir, "state_averages.csv"))
        return optimizer_dir

    def get_config(self):
        """
        Get the optimizer configuration

        Returns
        -------
        :obj:`dict`
            the configuration of the optimizer
        """
        return {'grid_shape': self.grid_shape}

class PymooOptimizer(Optimizer):
    def __init__(
        self, termination=None, n_iter=2, seed=0, print_history=True, 
        save_history_sim=False, **kwargs
    ):
        """
        Generic wrapper for `pymoo` optimizers.

        Parameters:
        ----------
        termination: :obj:`pymoo.termination.Termination`
            The termination object that defines the stopping criteria for 
            the optimization process.
            If not provided, the termination criteria will be based on the 
            number of iterations (`n_iter`).
        n_iter: :obj:`int`
            The maximum number of iterations for the optimization process.
            This parameter is only used if `termination` is not provided.
        seed: :obj:`int`
            The seed value for the random number generator used by the optimizer.
        print_history: :obj:`bool`
            Flag indicating whether to print the optimization history during the 
            optimization process.
        save_history_sim: :obj:`bool`
            Flag indicating whether to save the simulation data of each iteration.
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
        pymoo_verbose: :obj:`bool`
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
        x0: array-like
            The initial guess for the optimization.
            If None (default), the initial guess will be estimated based on
            `popsize` random samples as the first generation
        sigma: :obj:`float`
            The initial step size for the optimization
        use_bound_penalty: :obj:`bool`
            Whether to use a bound penalty for the optimization
        algorithm_kws: :obj:`dict`
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
        algorithm_kws: dict
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
    save: :obj:`bool`
        save the optimizers and their results. This is more efficient
        than saving each optimizer separately as saving involves
        rerunning the optimal simulations, which is done in a batch in
        this function.
    setup_kwargs: :obj:`dict`
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
    assert all(isinstance(o, PymooOptimizer) for o in optimizers), (
        "Batch optimization is only supported for PymooOptimizer derived classes"
    )
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

    for optimizer in optimizers:
        optimizer.history = pd.concat(optimizer.history, axis=0).reset_index(drop=False)
        if optimizer.problem.n_obj == 1:
            # a single optimum can be defined
            optimizer.opt_history = pd.DataFrame(optimizer.opt_history).reset_index(drop=True)
            optimizer.opt = optimizer.history.loc[optimizer.history["cost"].argmin()]
        optimizer.is_fit = True

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