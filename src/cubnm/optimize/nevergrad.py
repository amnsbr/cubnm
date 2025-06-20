import warnings
import numpy as np
import pandas as pd

import nevergrad as ng
import cma

from cubnm.optimize.base import Optimizer

class NevergradOptimizer(Optimizer):
    def __init__(
            self, 
            problem, 
            popsize, 
            n_iter, 
            seed=0, 
            max_sampling_per_particle=10,
            print_history=True,
            save_history_sim=False, 
            **kwargs
        ):
        """
        Generic wrapper for `nevergrad` optimizers.

        Parameters
        ----------
        problem: :obj:`cubnm.optimizer.BNMProblem`
            The problem to be set up with the algorithm.
        popsize: :obj:`int`
            The number of individuals in the population.
        n_iter: :obj:`int`, optional
            The maximum number of iterations for the optimization process.
        seed: :obj:`int`, optional
            The seed value for the random number generator used by the optimizer.
        print_history: :obj:`bool`, optional
            Whether to print the optimization history during the 
            optimization process.
        save_history_sim: :obj:`bool`, optional
            Whether to save the simulation data of each iteration.
            Default is False to avoid consuming too much memory across iterations.
        **kwargs
            Additional keyword arguments that can be passed to the `nevergrad` optimizer.
        """
        self.problem = problem
        self.popsize = popsize
        self.n_iter = n_iter
        self.seed = seed
        self.max_sampling_per_pop = max_sampling_per_particle * self.popsize
        self.print_history = print_history
        self.save_history_sim = save_history_sim
        if self.save_history_sim:
            raise NotImplementedError
        if self.problem.multiobj:
            raise NotImplementedError(
                "Multi-objective optimization is not supported "
                "with `nevergrad`. Use `pymoo` optimizers instead"
            )
        # define budget
        self.budget = self.popsize * self.n_iter
        # define parameterization
        _params = []
        for _ in range(self.problem.ndim):
            _params.append(
                ng.p.Scalar(lower=0, upper=1)
            )
        self.parametrization = ng.p.Instrumentation(*_params)
        # set its random state
        self.parametrization.random_state.seed(self.seed)
        
    @property
    def stop(self):
        return self.gen+1 > self.n_iter
        
    def optimize(self):
        """
        Optimizes the associated :class:`cubnm.optimizer.BNMProblem`
        free parameters through an evolutionary optimization approach by
        running multiple generations of parallel simulations until the
        termination criteria is met or maximum number of iterations is reached.
        """
        self.gen = 0
        self.history = []
        self.opt_history = []
        while not self.stop:
            # TODO: the early stopping (with CMAES) does not work sometimes
            # figure out why and fix it
            # sample current generation
            pop = []
            X = []
            attempted_samples = 0
            while ((len(pop) < self.popsize) & (not self.stop) & (attempted_samples < self.max_sampling_per_pop)):
                sample = self.algorithm.ask()
                x = np.array(sample.args)
                if self.problem._is_feasible(x):
                    pop.append(sample)
                    X.append(x)
                attempted_samples += 1
            # evaluate the samples (in parallel)
            X = np.array(X)
            out = {} # will include 'F' (cost)
            scores = [] # will include individual GOF measures
            self.problem._evaluate(X, out, scores=scores)
            scores = scores[0]
            scores.loc[:, 'cost'] = out['F']
            # tell the algorithm
            for i, particle in enumerate(pop):
                self.algorithm.tell(particle, out['F'][i])
            # record (and print) history
            Xt = pd.DataFrame(self.problem._get_Xt(X), columns=self.problem.free_params)
            Xt.index.name = "sim_idx"
            res = pd.concat([Xt, scores], axis=1)
            res["gen"] = self.gen
            if self.print_history:
                print(res.to_string())
            self.history.append(res)
            # define optimum up to this point
            self.opt_history.append(res.loc[res["cost"].argmin()])
            self.gen += 1
        self.history = pd.concat(self.history, axis=0).reset_index(drop=False)
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
        res = self.algorithm.recommend()
        return np.atleast_2d(np.array(res.args))

class CMAESOptimizer(NevergradOptimizer):
    def __init__(self, fcmaes=True, random_init=True, algorithm_kws={}, *args, **kwargs):
        """
        CMA-ES optimizer of `nevergrad` library.

        Parameters
        ----------
        fcmaes: :obj:`bool`, optional
            whether to use fcmaes backend
        random_init: :obj:`bool`, optional
            whether to use random initialization for the first generation
        algorithm_kws: :obj:`dict`, optional
            Additional keyword arguments passed to :class:`ng.families.ParameterizedCMA`
            constructor.
        *args, **kwargs
            Additional arguments passed to the `NevergradOptimizer` constructor.
        """
        super().__init__(*args, **kwargs)
        self.fcmaes = fcmaes
        self.random_init = random_init
        algorithm_kws.pop('fcmaes', None)
        algorithm_kws.pop('random_init', None)
        if not self.fcmaes:
            # cma backend raises a warning when multiple single-particle asks are called
            warnings.simplefilter("ignore", cma.evolution_strategy.InjectionWarning)
            if self.seed == 0:
                print(
                    "Warning: Seed value for CMAES is set to 0."
                    " This will result in different random initializations for each run"
                    " due to how `cma` works. Setting seed to 100 instead."
                )
                self.seed = 100
                self.parameterization.random_state.seed(self.seed)
        else:
            if 'inopts' in algorithm_kws:
                raise ValueError(
                    "The 'inopts' argument is not supported with fcmaes backend."
                )
        self.algorithm = ng.families.ParametrizedCMA(fcmaes=self.fcmaes, **algorithm_kws)(
            parametrization=self.parametrization, 
            budget=self.budget,
            num_workers=self.popsize,
        )

    @property
    def stop(self):
        if self.fcmaes:
            return super().stop or (self.algorithm.es.stop > 0)
        else:
            return super().stop or (len(self.algorithm.es.stop()) > 0)
