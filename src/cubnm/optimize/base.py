"""
Optimizers of the model free parameters
"""
import os
from abc import ABC, abstractmethod
import json
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pymoo.core.problem import Problem

from cubnm import sim, utils

# define evaluation metric labels used in plots
METRIC_LABELS = {
    'cost': 'Cost',
    '-cost': '- Cost',
    '+gof': 'Goodness-of-fit',
    '+fc_corr': r'FC$_corr$',
    '-fcd_ks': r'- FCD$_KS$',
    '-fc_diff': r'- FC$_diff$',
    '-fc_normec': r'- FC$_normEC'
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
        maps=None,
        maps_coef_range='auto',
        node_grouping=None,
        multiobj=False,
        **kwargs,
    ):
        """
        Brain network model problem. A :class:`pymoo.core.problem.Problem` 
        It defines the model, free parameters and their ranges, and target empirical
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
        het_params: :obj:`list` of :obj:`str`, optional
            which regional parameters are heterogeneous across nodes
        maps: :obj:`str`, optional
            path to heterogeneity maps as a text file or a numpy array.
            Shape: (n_maps, nodes).
            If provided one free parameter per regional parameter per 
            each map will be added.
        maps_coef_range: 'auto' or :obj:`tuple` or :obj:`list` of :obj:`tuple`
            - 'auto': uses (-1/max, -1/min) for maps with positive and negative
              values (assuming they are z-scored) and (0, 1) otherwise
            - :obj:`tuple`: uses the same range for all maps
            - :obj:`list` of :obj:`tuple`: n-map element list specifying the range
                of coefficients for each map
        node_grouping: {None, 'node', 'sym', :obj:`str`, :obj:`np.ndarray`}, optional
            - None: does not use region-/group-specific parameters
            - 'node': each node has its own regional free parameters
            - 'sym': uses the same regional free parameters for each pair of symmetric nodes
              (e.g. L and R hemispheres). Assumes symmetry  of parcels between L and R
              hemispheres.
            - :obj:`str`: path to a text file including node grouping array. Shape: (nodes,)
            - :obj:`np.ndarray`: a numpy array. Shape: (nodes,)
        multiobj: :obj:`bool`, optional
            instead of combining the objectives into a single objective function
            (via summation) defines each objective separately. This must not be used
            with single-objective optimizers
        **kwargs
            Keyword arguments passed to :class:`cubnm.sim.SimGroup`
        """
        # set opts
        self.model = model 
        self.params = params
        self.emp_fc_tril = emp_fc_tril
        self.emp_fcd_tril = emp_fcd_tril
        self.emp_bold=emp_bold
        self.het_params = kwargs.pop("het_params", het_params)
        self.input_maps = kwargs.pop("maps", maps)
        self.maps_coef_range = kwargs.pop("maps_coef_range", maps_coef_range)
        self.reject_negative = False # not implemented yet
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
            self.sim_group.do_fc = False
            self.sim_group.do_fcd = False
            if (self.emp_fc_tril is not None):
                self.sim_group.do_fc = True
                if (self.emp_fcd_tril is not None):
                    self.sim_group.do_fcd = True
            elif (self.emp_fcd_tril is not None):
                # note: as separate fc and fcd calculation is not
                # supported on simulation side, if fcd is provided
                # set do_fc to true as well (but then simulated fc
                # will be ignored)
                self.sim_group.do_fc = True
                self.sim_group.do_fcd = True
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
                # each node gets its own regional free parameters
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
                if isinstance(self.node_grouping, (str, os.PathLike)):
                    self.memberships = np.loadtxt(self.node_grouping).astype("int")
                else:
                    self.memberships = self.node_grouping.astype("int")
                self.node_groups = np.unique(self.memberships)
        # set up global and regional (incl. bias) free parameters
        for param, v in params.items():
            if isinstance(v, tuple):
                if (
                    self.is_regional
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
        self.is_heterogeneous = False
        if self.input_maps is not None:
            self.is_heterogeneous = True
            if isinstance(self.input_maps, (str, os.PathLike)):
                self.maps = np.loadtxt(self.input_maps)
            else:
                self.maps = self.input_maps
            assert (
                self.maps.shape[1] == self.sim_group.nodes
            ), f"Maps second dimension {self.maps.shape[1]} != nodes {self.sim_group.nodes}"
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
            n_ieq_constr=0,
            xl=np.zeros(self.ndim, dtype=float),
            xu=np.ones(self.ndim, dtype=float),
        )

    def get_config(self, include_sim_group=True, include_N=False):
        """
        Get the problem configuration

        Parameters
        ----------
        include_sim_group: :obj:`bool`, optional
            whether to include the configuration of the
            associated :class:`cubnm.sim.SimGroup`
        include_N: :obj:`bool`, optional
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

    def _is_feasible(self, x):
        """
        Checks if a set of candidate parameters for the
        next particle are feasible before running the
        simulations. Currently this only involves checking
        if regional parameters are non-negative after applying
        map-based heterogeneity. In addition it is only currently
        used by the Nevergrad optimizers.
        
        Parameters
        ----------
        x: :obj:`np.ndarray`
            the normalized parameters of next candidate in range [0, 1]. 
            Shape: (ndim,)
        """
        # get regional parameters based on higher-level optimization
        # parameters (bias and coefficient terms)
        # TODO: Avoid this for model parameters where negative
        # values are valid
        param_lists = self._get_sim_params(x[None, :], fix_negatives=False)
        for param, param_arr in param_lists.items():
            if param_arr.min() < 0:
                return False
        return True
    
    def _get_sim_params(self, X, fix_negatives=True):
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
        fix_negatives: :obj:`bool`
            if regional parameter maps include negative values
            shift them so that the lowest value is ``np.finfo(float).eps``
            
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
                    param_lists[param][sim_idx, :] *= param_scalers
                    # fix negatives if indicated
                    if fix_negatives:
                        if param_lists[param][sim_idx, :].min() < 0:
                            param_lists[param][sim_idx, :] -= (
                                param_lists[param][sim_idx, :].min()
                                - np.finfo(float).eps
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
                param_lists[param] = curr_param_maps
        return param_lists
    
    def _set_sim_params(self, X, fix_negatives=True):
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
        fix_negatives: :obj:`bool`
            if regional parameter maps include negative values
            shift them so that the lowest value is ``np.finfo(float).eps``
        """
        self.sim_group.param_lists.update(self._get_sim_params(X, fix_negatives))

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
            the output dictionary to store the results with keys 'F' and 'G'.
            Currently only 'F' (cost) is used.
        *args, **kwargs
            additional arguments passed to the evaluation function
            kwargs may include:
            - scores: :obj:`list`
                an empty list passed on to evaluation function to store
                the individual goodness of fit measures
            - skip_run: :obj:`bool`, optional
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
        skip_run: :obj:`bool`, optional
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
        save_opt: :obj:`bool`, optional
            reruns and saves the optimal simulation(s) data
        save_obj: :obj:`bool`, optional
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
            if hasattr(self, "opt_history"):
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
            if getattr(self, "is_grid", False):
                print("Saving the optimal simulation")
                ## slice sim_group at the index of the optimal simulation
                opt_idx = self.opt.name
                self.problem.opt_sim_group = self.problem.sim_group.slice(opt_idx, inplace=False)
            else:
                ## reuse the original problem used throughout the optimization
                ## but change its out_dir, N and parameters to the optimum
                print("Rerunning and saving the optimal simulation(s)")
                X = self.opt_X
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
        title: :obj:`str`, optional
            the title of the plot. If 'default', the title will be
            set to the measure clean label. If None, no title will be set.
        opt: :obj:`bool`
            whether to mark the optimum in the plot. Will
            be ignored if ``gen`` is provided.
        gen: :obj:`int` or :obj:`None`
            the generation to plot. If None, the entire history
            will be plotted. When using a :class:`cubnm.optimize.grid.GridOptimizer`
            this is ignored.
        config: :obj:`dict`, optional
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

        ax: :obj:`matplotlib.axes.Axes`, optional
            the axes to plot on. For 3D must have
            `projection='3d'`.
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
            if not getattr(self, "has_state_history", False):
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
        if (gen is not None) and ('gen' in history.columns):
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
        legend: :obj:`bool`, optional
            whether to show the legend
        ax: :obj:`matplotlib.axes.Axes`, optional
            the axes to plot on. If None, a new figure and axes will be created.
        line_kws: :obj:`dict`, optional
            additional keyword arguments passed to the line plot
            of the median across generations
        scatter_kws: :obj:`dict`, optional
            additional keyword arguments passed to the scatter plot
            of the individual particles across generations
        """
        assert self.is_fit, (
            "Cannot plot history before the optimizer is fit. "
            "Call `optimize` method first."
        )
        assert "gen" in self.history.columns, (
            "Not applicable to GridOptimizer."
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
        sns.scatterplot(data=plot_data, x='gen', y=measure, label='Particles', **_scatter_kws)
        # aesthetics
        ax.set_ylabel(METRIC_LABELS.get(measure, measure))
        ax.set_xlabel('Generation')
        if legend:
            ax.legend()
        return ax