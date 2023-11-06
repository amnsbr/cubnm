import itertools
import numpy as np
import pandas as pd
from pymoo.core.problem import Problem

from cuBNM import sim

class GridSearch():
    """
    Simple grid search
    """
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

    def _set_sim_params(self, X):
        global_params = ['G','v']
        local_params = ['wEE', 'wEI', 'wIE']
        # transform X from [0, 1] range to the actual
        # parameter range
        Xt = (X * (self.ub - self.lb)) + self.lb
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