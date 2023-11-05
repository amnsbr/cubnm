import itertools
import numpy as np
import pandas as pd

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
        self.sim_group = sim.SimGroup(N=self.param_combs.shape[0], **kwargs)
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
        
