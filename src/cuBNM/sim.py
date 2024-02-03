"""
Simulation of the model
"""
import numpy as np
import scipy.stats
import pandas as pd
import os
import gc

from cuBNM._core import run_simulations, set_const, set_conf
from cuBNM._setup_flags import many_nodes_flag, gpu_enabled_flag
from cuBNM import utils


class SimGroup:
    def __init__(
        self,
        duration,
        TR,
        sc_path,
        sc_dist_path=None,
        out_dir="same",
        do_fic=True,
        extended_output=True,
        extended_output_ts=False,
        window_size=10,
        window_step=2,
        rand_seed=410,
        exc_interhemispheric=True,
        force_cpu=False,
        gof_terms=["+fc_corr", "-fc_diff", "-fcd_ks"],
        fic_penalty=True,
        bw_params="friston2003",
    ):
        """
        Group of simulations that are executed in parallel
        (as possible depending on the available hardware)
        on GPU/CPU.

        Parameters
        ---------
        duration: :obj:`float`
            simulation duration (in seconds)
        TR: :obj:`float`
            BOLD TR and sampling rate of extended output (in seconds)
        sc_path: :obj:`str`
            path to structural connectome strengths (as an unlabled .txt)
            Shape: (nodes, nodes)
        sc_dist_path: :obj:`str`, optional
            path to structural connectome distances (as an unlabled .txt)
            Shape: (nodes, nodes)
            If provided v (velocity) will be a free parameter and there
            will be delay in inter-regional connections
        out_dir: {'same' or :obj:`str`}, optional
            - 'same': will create a directory named based on sc_path
            - :obj:`str`: will create a directory in the provided path
        do_fic: :obj:`bool`, optional
            do analytical-numerical Feedback Inhibition Control
            if provided wIE parameters will be ignored
        extended_output: :obj:`bool`, optional
            return mean internal model variables to self.ext_out
        extended_output_ts: :obj:`bool`, optional
            return time series of internal model variables to self.ext_out
            Note that this will increase the memory usage and is not
            recommended for large number of simulations (e.g. in a grid search)
        window_size: :obj:`int`, optional
            dynamic FC window size (in TR)
        window_step: :obj:`int`, optional
            dynamic FC window step (in TR)
        rand_seed: :obj:`int`, optional
            seed used for the noise simulation
        exc_interhemispheric: :obj:`bool`, optional
            excluded interhemispheric connections from sim FC and FCD calculations
        force_cpu: :obj:`bool`, optional
            use CPU for the simulations (even if GPU is available). If set
            to False the program might use GPU or CPU depending on GPU
            availability
        gof_terms: :obj:`list` of :obj:`str`, optional
            list of goodness-of-fit terms to be used for scoring. May include:
            - '-fcd_ks': negative Kolmogorov-Smirnov distance of FCDs
            - '+fc_corr': Pearson correlation of FCs
            - '-fc_diff': negative absolute difference of FC means
            - '-fc_normec': negative Euclidean distance of FCs \
                divided by max EC [sqrt(n_pairs*4)]
        fic_penalty: :obj:`bool`, optional
            penalize deviation from FIC target mean rE of 3 Hz
        bw_params: {'friston2003' or 'heinzle2016-3T' or :obj:`dict`}, optional
            see :func:`cuBNM.utils.get_bw_params` for details

        Attributes
        ---------
        param_lists: :obj:`dict` of :obj:`np.ndarray`
            dictionary of parameter lists, including
                - 'G': global coupling strength. Shape: (N_SIMS,)
                - 'wEE': local excitatory self-connection strength. Shape: (N_SIMS, nodes)
                - 'wEI': local inhibitory self-connection strength. Shape: (N_SIMS, nodes)
                - 'wIE': local excitatory to inhibitory connection strength. Shape: (N_SIMS, nodes)
                - 'v': conduction velocity. Shape: (N_SIMS,)

        Example
        -------
        To see example usage in grid search and evolutionary algorithms
        see :mod:`cuBNM.optimize`.

        Here, as an example on how to use SimGroup independently, we
        will run a single simulation and save the outputs to disk. ::

            from cuBNM import sim, datasets

            sim_group = sim.SimGroup(
                duration=60,
                TR=1,
                sc_path=datasets.load_sc('strength', 'schaefer-100', return_path=True),
            )
            sim_group.N = 1
            sim_group.param_lists['G'] = np.array([0.5])
            sim_group.param_lists['wEE'] = np.repeat(0.21, nodes*N_SIMS)
            sim_group.param_lists['wEI'] = np.repeat(0.15, nodes*N_SIMS)
            sim_group.run()
        """
        self.duration = duration
        self.TR = TR  # in msec
        self.sc_path = sc_path
        self.sc_dist_path = sc_dist_path
        self._out_dir = out_dir
        self.sc = np.loadtxt(self.sc_path)
        self.do_fic = do_fic
        self.extended_output = (
            extended_output | do_fic
        )  # extended output is needed for FIC penalty calculations
        self.extended_output_ts = self.extended_output & extended_output_ts
        self.window_size = window_size
        self.window_step = window_step
        self.rand_seed = rand_seed
        self.exc_interhemispheric = exc_interhemispheric
        self.force_cpu = force_cpu
        self.gof_terms = gof_terms
        self.fic_penalty = fic_penalty
        # get time and TR in msec
        self.duration_msec = int(duration * 1000)  # in msec
        self.TR_msec = int(TR * 1000)
        # determine number of nodes based on sc dimensions
        self.nodes = self.sc.shape[0]
        if (self.nodes > 500) & (not many_nodes_flag):
            # TODO: try to get an estimate of max nodes based on the device's
            # __shared__ memory size
            # TODO: with higher number of nodes simulation may still fail
            # even if __device__ memory is used because the "real" max number of threads
            # on a block is usually maxed out beyond 600 nodes
            print(
                f"""
            Warning: In the current version with {self.nodes} nodes the simulation might
            fail due to limits in __shared__ GPU memory. If the simulations fail
            reinstall the package with MANY_NODES support (which uses slower but larger
            __device__ memory) via: 
            > export CUBNM_MANY_NODES=1 && pip install --ignore-installed cuBNM
            But if it doesn't fail there is no need to reinstall the package.
            """
            )
        # inter-regional delay will be added to the simulations
        # if SC distance matrix is provided
        if self.sc_dist_path:
            self.sc_dist = np.loadtxt(self.sc_dist_path)
            self.do_delay = True
        else:
            self.sc_dist = np.zeros_like(self.sc, dtype=float)
            self.do_delay = False
        # set Ballon-Windkessel parameters
        self.bw_params = bw_params
        # initialze w_IE_list as all 0s if do_fic
        self.param_lists = dict([(k, None) for k in ["G", "wEE", "wEI", "wIE", "v"]])
        # determine and create output directory
        if out_dir == "same":
            self.out_dir = self.sc_path.replace(".txt", "")
        else:
            self.out_dir = out_dir
        os.makedirs(out_dir, exist_ok=True)
        # keep track of the last N and config to determine if force_reinit
        # is needed in the next run if any are changed
        self.last_N = 0
        self.last_config = self.get_config(for_reinit=True)
        # keep track of the iterations for iterative algorithms
        self.it = 0

    @property
    def N(self):
        return self._N

    @N.setter
    def N(self, N):
        self._N = N
        # TODO: maybe it's more appropriate to the following elewhere
        if self.do_fic:
            self.param_lists["wIE"] = np.zeros((self._N, self.nodes), dtype=float)
        if not self.do_delay:
            self.param_lists["v"] = np.zeros(self._N, dtype=float)

    @property
    def bw_params(self):
        return self._bw_params

    @bw_params.setter
    def bw_params(self, bw_params):
        self._bw_params = bw_params
        params = utils.get_bw_params(self._bw_params)
        set_const("k1", params["k1"])
        set_const("k2", params["k2"])
        set_const("k3", params["k3"])

    @property
    def exc_interhemispheric(self):
        return self._exc_interhemispheric

    @exc_interhemispheric.setter
    def exc_interhemispheric(self, exc_interhemispheric):
        self._exc_interhemispheric = exc_interhemispheric
        set_conf("exc_interhemispheric", self._exc_interhemispheric)

    @property
    def do_delay(self):
        return self._do_delay

    @do_delay.setter
    def do_delay(self, do_delay):
        self._do_delay = do_delay
        self.sync_msec = self._do_delay
        if self._do_delay:
            print(
                "Delay is enabled...will sync nodes every 1 msec\n"
                "to do syncing every 0.1 msec set self.sync_msec to False\n"
                "but note that this will increase the simulation time\n"
            )

    @property
    def sync_msec(self):
        return self._sync_msec

    @sync_msec.setter
    def sync_msec(self, sync_msec):
        self._sync_msec = sync_msec
        set_conf("sync_msec", self._sync_msec)

    def get_config(self, include_N=False, for_reinit=False):
        """
        Get the configuration of the simulation group

        Parameters
        ----------
        include_N: :obj:`bool`, optional
            include N in the output config
            is ignored when for_reinit is True
        for_reinit: :obj:`bool`, optional
            include the parameters that need reinitialization if changed

        Returns
        -------
        config: :obj:`dict`
            dictionary of simulation group configuration
        """
        config = {
            "duration": self.duration,
            "TR": self.TR,
            "sc_path": self.sc_path,
            "sc_dist_path": self.sc_dist_path,
            "do_fic": self.do_fic,
            "extended_output": self.extended_output,
            "extended_output_ts": self.extended_output_ts,
            "window_size": self.window_size,
            "window_step": self.window_step,
            "rand_seed": self.rand_seed,
            "exc_interhemispheric": self.exc_interhemispheric,
            "force_cpu": self.force_cpu,
            "bw_params": self.bw_params,
        }
        if not for_reinit:
            config["out_dir"] = self.out_dir
            config["gof_terms"] = self.gof_terms
            if include_N:
                config["N"] = self.N
        return config

    def run(self, force_reinit=False):
        """
        Run the simulations in parallel (as possible) on GPU/CPU
        through the :func:`cuBNM._core.run_simulations` function which runs
        compiled C++/CUDA code.

        Parameters
        ----------
        force_reinit: :obj:`bool`, optional
            force reinitialization of the session.
            At the beginning of each session (when `cuBNM` is imported)
            some variables are initialized on CPU/GPU and reused in
            every run. Set this to True if you want to reinitialize
            these variables. This is rarely needed.

        Notes
        -----
        The simulation outputs are assigned to the following object attributes:
            - sim_bold : :obj:`np.ndarray`
                simulated BOLD time series. Shape: (N_SIMS, duration/TR, nodes)
            - sim_fc_trils : :obj:`np.ndarray`
                simulated FC lower triangle. Shape: (N_SIMS, n_pairs)
            - sim_fcd_trils : :obj:`np.ndarray`
                simulated FCD lower triangle. Shape: (N_SIMS, n_pairs)
            - ext_out: :obj:`dict` of :obj:`np.ndarray`
                dictionary of time-averaged model state variables.
                Shape of each array: (N_SIMS, nodes)
                Only available if `extended_output` or `do_fic` is True.
            - fic_unstable: :obj:`np.ndarray`
                boolean array indicating if the FIC analytical solution
                was unstable. Shape: (N_SIMS,)
                Only available if `do_fic` is True.
        """
        # TODO: add assertions to make sure all the input data is complete
        # check if reinitialization is needed (user request, change of N,
        # of change of other config)
        curr_config = self.get_config(for_reinit=True)
        force_reinit = (
            force_reinit
            | (self.N != self.last_N)
            | (curr_config != self.last_config)
        )
        use_cpu = self.force_cpu | (not gpu_enabled_flag) | (utils.avail_gpus() == 0)
        # set wIE to its flattened copy and pass this
        # array to run_simulations so that in the case
        # of do_fic it is overwritten with the actual wIE
        # used in the simulations
        # Note that the 2D arrays are flattened to a 1D (pseudo-2D) array
        # and all arrays are made C-contiguous using .ascontiguousarray
        # .ascontiguousarray may be redundant with .flatten but I'm doing
        # it to be sure
        self.param_lists["wIE"] = np.ascontiguousarray(
            self.param_lists["wIE"].flatten()
        )
        global_params = np.ascontiguousarray(self.param_lists["G"])[np.newaxis, :]
        regional_params = np.vstack([
            np.ascontiguousarray(self.param_lists["wEE"].flatten()),
            np.ascontiguousarray(self.param_lists["wEI"].flatten()),
            self.param_lists["wIE"],
        ])
        model_config = {
            'do_fic': str(int(self.do_fic)),
            # 'max_fic_trials': '5',
            # 'fic_verbose': '0',
        }
        out = run_simulations(
            'rWW',
            np.ascontiguousarray(self.sc.flatten()),
            np.ascontiguousarray(self.sc_dist.flatten()),
            global_params,
            regional_params,
            np.ascontiguousarray(self.param_lists["v"]),
            model_config,
            self.extended_output,
            self.extended_output_ts,
            self.do_delay,
            force_reinit,
            use_cpu,
            self.N,
            self.nodes,
            self.duration_msec,
            self.TR_msec,
            self.window_size,
            self.window_step,
            self.rand_seed,
        )
        # avoid reinitializing GPU in the next runs
        # of the same group
        self.last_N = self.N
        self.last_config = curr_config
        self.it += 1
        # assign the output to object properties
        # and reshape them to (N_SIMS, ...)
        self.ext_out = {}
        if self.extended_output:
            (
                sim_bold,
                sim_fc_trils,
                sim_fcd_trils,
                sim_states,
                global_bools,
                global_ints,
            ) = out
            self.ext_out = {
                'I_E': sim_states[0],
                'I_I': sim_states[1],
                'r_E': sim_states[2],
                'r_I': sim_states[3],
                'S_E': sim_states[4],
                'S_I': sim_states[5],
            }
            if self.extended_output_ts:
                for k in self.ext_out:
                    self.ext_out[k] = self.ext_out[k].reshape(self.N, -1, self.nodes)
            self.fic_unstable = global_bools[0]
            self.fic_failed = global_bools[1]
            self.fic_ntrials = global_ints[0]
        else:
            sim_bold, sim_fc_trils, sim_fcd_trils = out
        self.sim_bold = sim_bold.reshape(self.N, -1, self.nodes)
        self.sim_fc_trils = sim_fc_trils.reshape(self.N, -1)
        self.sim_fcd_trils = sim_fcd_trils.reshape(self.N, -1)
        self.param_lists["wIE"] = self.param_lists["wIE"].reshape(self.N, -1)

    def clear(self):
        """
        Clear the simulation outputs
        """
        for attr in ["sim_bold", "sim_fc_trils", "sim_fcd_trils", "ext_out", "fic_unstable"]:
            if hasattr(self, attr):
                delattr(self, attr)
        gc.collect()

    def score(self, emp_fc_tril, emp_fcd_tril, fic_penalty_scale=2):
        """
        Calcualates individual goodness-of-fit terms and aggregates them.
        In FIC models also calculates fic_penalty.

        Parameters
        --------
        emp_fc_tril: :obj:`np.ndarray`
            1D array of empirical FC lower triangle. Shape: (edges,)
        emp_fcd_tril: :obj:`np.ndarray`
            1D array of empirical FCD lower triangle. Shape: (window_pairs,)
        fic_penalty_scale: :obj:`float`, optional
            scale of the FIC penalty term.
            Set to 0 to disable the FIC penalty term.
            Note that while it is included in the cost function of
            optimizer, it is not included in the aggregate GOF

        Returns
        -------
        scores: :obj:`pd.DataFrame`
            The goodness of fit measures (columns) of each simulation (rows)
        """
        # + => aim to maximize; - => aim to minimize
        # TODO: add the option to provide empirical BOLD as input
        columns = ["+fc_corr", "-fc_diff", "-fcd_ks", "-fc_normec", "+gof"]
        if self.do_fic:
            columns.append("-fic_penalty")
        scores = pd.DataFrame(columns=columns, dtype=float)
        # calculate GOF
        for idx in range(self.N):
            scores.loc[idx, "+fc_corr"] = scipy.stats.pearsonr(
                self.sim_fc_trils[idx], emp_fc_tril
            ).statistic
            scores.loc[idx, "-fc_diff"] = -np.abs(
                self.sim_fc_trils[idx].mean() - emp_fc_tril.mean()
            )
            scores.loc[idx, "-fcd_ks"] = -scipy.stats.ks_2samp(
                self.sim_fcd_trils[idx], emp_fcd_tril
            ).statistic
            scores.loc[idx, "-fc_normec"] = -utils.fc_norm_euclidean(
                self.sim_fc_trils[idx], emp_fc_tril
            )
            # combined the selected terms into gof (that should be maximized)
            scores.loc[idx, "+gof"] = 0
            for term in self.gof_terms:
                scores.loc[idx, "+gof"] += scores.loc[idx, term]
        # calculate FIC penalty
        if self.do_fic & self.fic_penalty:
            for idx in range(self.N):
                diff_r_E = np.abs(self.ext_out["r_E"][idx, :] - 3)
                if (diff_r_E > 1).sum() > 0:
                    diff_r_E[diff_r_E <= 1] = np.NaN
                    scores.loc[idx, "-fic_penalty"] = (
                        -np.nansum(1 - np.exp(-0.05 * (diff_r_E - 1)))
                        * fic_penalty_scale / self.nodes
                    )
                else:
                    scores.loc[idx, "-fic_penalty"] = 0
        return scores

    def save(self, save_as="npz"):
        """
        Save simulation outputs to disk.

        Parameters
        ---------
        save_as: {'npz' or 'txt'}, optional
            - 'npz': all the output of all sims will be written to a npz file
            - 'txt': outputs of simulations will be written to separate files,\
                recommended when N = 1 (e.g. rerunning the best simulation)
        """
        sims_dir = self.out_dir
        os.makedirs(sims_dir, exist_ok=True)
        if save_as == "npz":
            out_data = dict(
                sim_bold=self.sim_bold,
                sim_fc_trils=self.sim_fc_trils,
                sim_fcd_trils=self.sim_fcd_trils,
            )
            out_data.update(self.param_lists)
            if self.extended_output:
                out_data.update(self.ext_out)
            # TODO: use more informative filenames
            np.savez_compressed(os.path.join(sims_dir, f"it{self.it}.npz"), **out_data)
        elif save_as == "txt":
            raise NotImplementedError
