import os
import itertools
import numpy as np
import pandas as pd

from cubnm.optimize.base import Optimizer

class GridOptimizer(Optimizer):
    has_state_history = True
    is_grid = True
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
        save_avg_states: :obj:`bool`, optional
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
