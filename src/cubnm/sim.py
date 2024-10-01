"""
Simulation of the models
"""
import numpy as np
import scipy.stats
import pandas as pd
import os
import gc
import copy
from decimal import Decimal

from cubnm._core import run_simulations, set_const
from cubnm._setup_opts import (
    many_nodes_flag, gpu_enabled_flag, 
    max_nodes_reg, max_nodes_many,
    noise_segment_flag
)
from cubnm import utils, datasets


class SimGroup:
    def __init__(
        self,
        duration,
        TR,
        sc,
        sc_dist=None,
        out_dir=None,
        dt='0.1',
        bw_dt='1.0',
        ext_out=True,
        states_ts=False,
        states_sampling=None,
        noise_out=False,
        do_fc=True,
        do_fcd=True,
        window_size=10,
        window_step=2,
        rand_seed=410,
        exc_interhemispheric=False,
        force_cpu=False,
        force_gpu=False,
        serial_nodes=False,
        gof_terms=["+fc_corr", "-fcd_ks"],
        bw_params="friston2003",
        bold_remove_s=30,
        fcd_drop_edges=True,
        noise_segment_length=30,
        sim_verbose=False,
        progress_interval=500
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
            BOLD TR (in seconds)
        sc: :obj:`str` or :obj:`np.ndarray`
            path to structural connectome strengths (as an unlabled .txt)
            or a numpy array
            Shape: (nodes, nodes)
            If asymmetric, rows are sources and columns are targets.
        sc_dist: :obj:`str` or :obj:`np.ndarray`, optional
            path to structural connectome distances (as an unlabled .txt)
            or a numpy array
            Shape: (nodes, nodes)
            If provided v (velocity) will be a free parameter and there
            will be delay in inter-regional connections
            If asymmetric, rows are sources and columns are targets.
        out_dir: {:obj:`str` or 'same' or None}, optional
            - :obj:`str`: will create a directory in the provided path
            - 'same': will create a directory named based on sc
                (only should be used when sc is a path and not a numpy array)
            - None: will not have an output directory (and cannot save outputs)
        dt: :obj:`decimal.Decimal` or :obj:`str`, optional
            model integration time step (in msec)
        bw_dt: :obj:`decimal.Decimal` or :obj:`str`, optional
            Ballon-Windkessel integration time step (in msec)
        ext_out: :obj:`bool`, optional
            return model state variables to self.sim_states
        states_ts: :obj:`bool`, optional
            return time series of model states to self.sim_states
            Note that this will increase the memory usage and is not
            recommended for large number of simulations (e.g. in a grid search)
        states_sampling: :obj:`float`, optional
            sampling rate of model states in seconds.
            Default is None, which uses BOLD TR as the sampling rate.
        noise_out: :obj:`bool`, optional
            return noise time series
        do_fc: :obj:`bool`, optional
            calculate simulated functional connectivity (FC)
        do_fcd: :obj:`bool`, optional
            calculate simulated functional connectivity dynamics (FCD)
        window_size: :obj:`int`, optional
            dynamic FC window size (in TR)
            Must be even. The actual window size is +1 (including center)
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
        force_gpu: :obj:`bool`, optional
            on some HPC/HTC systems occasionally GPUtil might not detect an 
            available GPU device. Use this if there is a GPU available but
            is not being used for the simulation. If set to True but a GPU
            is not available will lead to errors.
        serial_nodes: :obj:`bool`, optional
            only applicable to GPUs; uses one thread per simulation and do calculation
            of nodes serially. This is an experimental feature which is generally not 
            recommended and has significantly slower performance in typical use cases. 
            Only may provide performance benefits with very large grids as computing 
            time does not scale with the number of simulations as much as the 
            parallel (default) mode.
        gof_terms: :obj:`list` of :obj:`str`, optional
            list of goodness-of-fit terms to be used for scoring. May include:
            - '-fcd_ks': negative Kolmogorov-Smirnov distance of FCDs
            - '+fc_corr': Pearson correlation of FCs
            - '-fc_diff': negative absolute difference of FC means
            - '-fc_normec': negative Euclidean distance of FCs \
                divided by max EC [sqrt(n_pairs*4)]
        bw_params: {'friston2003' or 'heinzle2016-3T' or :obj:`dict`}, optional
            see :func:`cubnm.utils.get_bw_params` for details
        bold_remove_s: :obj:`float`, optional
            remove the first bold_remove_s seconds from the simulated BOLD
            in FC, FCD and mean state calculations (but the entire BOLD will
            be returned to .sim_bold)
        fcd_drop_edges: :obj:`bool`, optional
            drop the edge windows in FCD calculations
        noise_segment_length: :obj:`float` or None, optional
            in seconds, length of the noise segments in the simulations
            The noise segment will be repeated after shuffling
            of nodes and time points. To generate noise for the entire
            simulation without repetition, set this to None.
            Note that varying the noise segment length will result
            in a different noise array even if seed is fixed (but
            fixed combination of seed and noise_segment_length will
            result in reproducible noise)
        sim_verbose: :obj:`bool`, optional
            verbose output of the simulation including details of
            simulations and a progress bar. This may slightly make
            the simulations slower.
        progress_interval: :obj:`int`, optional
            msec; interval of progress updates in the simulation
            Only used if sim_verbose is True

        Attributes
        ---------
        param_lists: :obj:`dict` of :obj:`np.ndarray`
            dictionary of parameter lists, including
                - global parameters with shape (N_SIMS,)
                - regional parameters with shape (N_SIMS, nodes)
                - 'v': conduction velocity. Shape: (N_SIMS,)
        Additional attributes will be added after running the simulations.
        See :func:`cubnm.sim.SimGroup._process_out` for details.

        Notes
        -----
        Derived classes must set the following attributes:
            model_name: :obj:`str`
                name of the model used in the simulations
            global_param_names: :obj:`list` of :obj:`str`
                names of global parameters
            regional_param_names: :obj:`list` of :obj:`str`
                names of regional parameters
            sel_state_var: :obj:`str`
                name of the state variable used in the tests
            n_noise: :obj:`int`
                number of noise elements per node per time point
                (e.g. 2 if there are noise to E and I neuronal populations)
        And they must implement the following methods:
            _set_default_params: set default (example) parameters for the simulations
        """
        self.duration = duration
        self.TR = TR
        self.input_sc = sc
        if isinstance(sc, (str, os.PathLike)):
            self.sc = np.loadtxt(self.input_sc)
        else:
            self.sc = self.input_sc
        self._dt = Decimal(dt)
        self._bw_dt = Decimal(bw_dt)
        self.ext_out = ext_out
        self.states_ts = self.ext_out & states_ts
        if states_sampling is None:
            self.states_sampling = self.TR
        else:
            self.states_sampling = states_sampling
        self.do_fc = do_fc
        if self.do_fc:
            self.do_fcd = do_fcd
        elif do_fcd:
            raise ValueError("Cannot calculate FCD without FC")
        else:
            self.do_fcd = False
        if not self.do_fc:
            if ('+fc_corr' in gof_terms) or ('-fc_diff' in gof_terms) or ('-fc_normec' in gof_terms):
                raise ValueError("Cannot calculate FC goodness-of-fit terms without FC."
                                 " Set do_fc to True or remove FC-related goodness-of-fit"
                                 " terms.")
        if not self.do_fcd:
            if '-fcd_ks' in gof_terms:
                raise ValueError("Cannot calculate FCD goodness-of-fit terms without FCD."
                                 " Set do_fcd to True or remove FCD-related goodness-of-fit"
                                 " terms.")
        self.window_size = window_size
        assert self.window_size % 2 == 0, "Window size must be even"
        self.window_step = window_step
        self.rand_seed = rand_seed
        self.exc_interhemispheric = exc_interhemispheric
        self.force_cpu = force_cpu
        self.force_gpu = force_gpu
        self.serial_nodes = serial_nodes
        self.gof_terms = gof_terms
        self.bold_remove_s = bold_remove_s
        self.fcd_drop_edges = fcd_drop_edges
        self.noise_segment_length = noise_segment_length
        if self.noise_segment_length is None:
            self.noise_segment_length = self.duration
        self.noise_out = noise_out
        self.sim_verbose = sim_verbose
        self.progress_interval = progress_interval
        # get duration, TR and extended sampling rate in msec
        self.duration_msec = int(duration * 1000)  # in msec
        self.TR_msec = int(TR * 1000)
        self.states_sampling_msec = int(self.states_sampling * 1000)
        # ensure bw_dt > dt and divisible by dt + duration is divisible by bw_dt
        self._check_dt()
        # warn user if serial is set to True
        # + revert unsupported options to default
        if self.serial_nodes and not self.force_cpu:
            print(
                "Warning: Running simulations serially on GPU is an experimental "
                "feature which is generally not recommended and has "
                "significantly slower performance. Consider setting "
                "serial_nodes to False."
            )
            if self.states_sampling != self.TR:
                print(
                    "In serial different states_sampling from TR is not "
                    "supported. Reseting it to TR."
                )
                self.states_sampling = self.TR
        # warn user if SC diagonal is not 0
        if np.any(np.diag(self.sc)):
            print("Warning: The diagonal of the SC matrix is not 0. "
                "Self-connections are not ignored by default in "
                "the simulations. If you want to ignore them set "
                "the diagonal of the SC matrix to 0.")
        # raise an error if sampling rate is below model's outer loop dt
        if (self.states_sampling < 0.001):
            raise ValueError(
                "states_sampling cannot be lower than 0.001s "
                "which is model dt"
            )
        # determine number of nodes based on sc dimensions
        self.nodes = self.sc.shape[0]
        self.use_cpu = (self.force_cpu | (not gpu_enabled_flag) | (utils.avail_gpus() == 0))
        max_nodes = max_nodes_many if many_nodes_flag else max_nodes_reg
        if (self.nodes > max_nodes) and (not self.use_cpu) and (not self.serial_nodes):
            if not many_nodes_flag:
                raise NotImplementedError(
                    f"With {self.nodes} nodes in the current installation of the toolbox"
                    " simulations will fail. Please reinstall the package from source after"
                    " `export CUBNM_MANY_NODES=1`")
            if self.do_fc or self.do_fcd:
                raise NotImplementedError(
                    "With many nodes, FC and FCD calculations are not supported."
                    " Set do_fc and do_fcd to False."
                )
            if self.dt < 1.0:
                print(
                    "Warning: When runnig simulations with large number"
                    " of nodes it is recommended to set the model dt to"
                    " >=1.0 msec for better performance."
                )
        # inter-regional delay will be added to the simulations
        # if SC distance matrix is provided
        self.input_sc_dist = sc_dist
        if self.input_sc_dist is None:
            self.sc_dist = np.zeros_like(self.sc, dtype=float)
            self.do_delay = False
        else:
            if isinstance(self.input_sc_dist, (str, os.PathLike)):
                self.sc_dist = np.loadtxt(self.input_sc_dist)
            else:
                self.sc_dist = self.input_sc_dist
            self.do_delay = True
        # set Ballon-Windkessel parameters
        self.bw_params = bw_params
        # initialze w_IE_list as all 0s if do_fic
        self.param_lists = dict([(k, None) for k in self.global_param_names + self.regional_param_names + ['v']])
        # determine output directory
        self.input_out_dir = out_dir
        if self.input_out_dir == "same":
            assert isinstance(self.input_sc, (str, os.PathLike)), (
                "When `out_dir` is set to 'same' `sc` must be"
                "a path-like string"
            )
            self.out_dir = self.input_sc.replace(".txt", "")
        else:
            self.out_dir = self.input_out_dir
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
        if not self.do_delay:
            self.param_lists["v"] = np.zeros(self._N, dtype=float)

    @property
    def dt(self):
        return self._dt
    
    @dt.setter
    def dt(self, dt):
        self._dt = Decimal(dt)
        self._check_dt()

    @property
    def bw_dt(self):
        return self._bw_dt
    
    @bw_dt.setter
    def bw_dt(self, bw_dt):
        self._bw_dt = Decimal(bw_dt)
        self._check_dt()

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
    def do_delay(self):
        return self._do_delay

    @do_delay.setter
    def do_delay(self, do_delay):
        self._do_delay = do_delay
        if self.do_delay and (self.dt < 1.0):
            print(
                "Warning: When using delay in the simulations"
                " it is recommended to set the model dt to >="
                " 1.0 msec for better performance."
            )

    def _check_dt(self):
        """
        Check if integrations steps are valid
        """
        assert self.bw_dt >= self.dt, (
            "Ballon-Windkessel dt must be greater than or equal to model dt"
        )
        assert (self.bw_dt % self.dt) == Decimal(0), (
            "Ballon-Windkessel dt must be divisible by model dt"
        )
        assert (self.duration_msec % self.bw_dt) == Decimal(0), (
            "Duration must be divisible by Ballon-Windkessel dt"
        )

    def _set_default_params(self):
        """
        Set default parameters for the simulations.
        This is used in tests.
        """
        if self.do_delay:
            # a low velocity is chosen to make the delay more visible
            # in the tests
            self.param_lists["v"] = np.repeat(0.5, self.N)

    def get_config(self, include_N=False, for_reinit=False):
        """
        Get the configuration of the simulation group

        Parameters
        ----------
        include_N: :obj:`bool`, optional
            include N in the output config
            is ignored when for_reinit is True
        for_reinit: :obj:`bool`, optional
            include the parameters that need reinitialization of the
            simulation core session if changed

        Returns
        -------
        config: :obj:`dict`
            dictionary of simulation group configuration
        """
        config = {
            "duration": self.duration,
            "TR": self.TR,
            "sc": self.input_sc,
            "sc_dist": self.input_sc_dist,
            "ext_out": self.ext_out,
            "states_ts": self.states_ts,
            "states_sampling": self.states_sampling,
            "do_fc": self.do_fc,
            "do_fcd": self.do_fcd,
            "window_size": self.window_size,
            "window_step": self.window_step,
            "rand_seed": self.rand_seed,
            "exc_interhemispheric": self.exc_interhemispheric,
            "force_cpu": self.force_cpu,
            "force_gpu": self.force_gpu,
            "bw_params": self.bw_params,
            "bold_remove_s": self.bold_remove_s,
            "fcd_drop_edges": self.fcd_drop_edges,
            "noise_segment_length": self.noise_segment_length,
        }
        if not for_reinit:
            config["out_dir"] = self.input_out_dir
            config["gof_terms"] = self.gof_terms
            if include_N:
                config["N"] = self.N
        return config
    
    @property
    def _model_config(self):
        """
        Internal model configuration used in the simulation
        """
        model_config = {
            'do_fc': str(int(self.do_fc)),
            'do_fcd': str(int(self.do_fcd)),
            'bold_remove_s': str(self.bold_remove_s),
            'exc_interhemispheric': str(int(self.exc_interhemispheric)),
            'window_size': str(self.window_size),
            'window_step': str(self.window_step),
            'drop_edges': str(int(self.fcd_drop_edges)),
            'noise_time_steps': str(int(self.noise_segment_length*1000)), 
            'verbose': str(int(self.sim_verbose)),
            'progress_interval': str(int(self.progress_interval)),
            'serial': str(int(self.serial_nodes)),
        }
        return model_config

    def run(self, force_reinit=False):
        """
        Run the simulations in parallel (as possible) on GPU/CPU
        through the :func:`cubnm._core.run_simulations` function which runs
        compiled C++/CUDA code.

        Parameters
        ----------
        force_reinit: :obj:`bool`, optional
            force reinitialization of the session.
            At the beginning of each session (when `cubnm` is imported)
            some variables are initialized on CPU/GPU and reused in
            every run. Set this to True if you want to reinitialize
            these variables. This is rarely needed.
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
        self.use_cpu = (
            (self.force_cpu | (not gpu_enabled_flag) | (utils.avail_gpus() == 0))
            & (not self.force_gpu)
        )
        # check if running on Jupyter and print warning that
        # simulations are running but progress will not be shown
        if utils.is_jupyter():
            print(
                "Warning: Progress cannot be shown in real time"
                " when using Jupyter, but simulations are running"
                " in background. Please wait...", flush=True
            )
        # convert self.param_lists to flattened and contiguous arrays of global
        # and local parameters
        global_params_arrays = []
        regional_params_arrays = []
        for param in self.global_param_names:
            global_params_arrays.append(np.ascontiguousarray(self.param_lists[param])[np.newaxis, :])
        for param in self.regional_param_names:
            regional_params_arrays.append(np.ascontiguousarray(self.param_lists[param].flatten()))
        self._global_params = np.vstack(global_params_arrays)
        self._regional_params = np.vstack(regional_params_arrays)
        # specify fixed or variable SCs
        if ((self.sc.ndim == 2) or (self.sc.shape[0] == 1)):
            sc = np.ascontiguousarray(self.sc.flatten())[None, :]
            sc_indices = np.repeat(0, self.N).astype(np.intc)
        else:
            sc = np.ascontiguousarray(self.sc.reshape(self.sc.shape[0], -1))
            sc_indices = np.ascontiguousarray(self.sc_indices.astype(np.intc))
        # make sure sc indices are correctly in {0, ..., N_SCs-1}
        assert sc.shape[0] == np.unique(sc_indices).size
        assert (np.sort(np.unique(sc_indices)) == np.arange(np.unique(sc_indices).size)).all()
        assert sc_indices.size == self.N
        out = run_simulations(
            self.model_name,
            sc,
            sc_indices,
            np.ascontiguousarray(self.sc_dist.flatten()),
            self._global_params,
            self._regional_params,
            np.ascontiguousarray(self.param_lists["v"]),
            self._model_config,
            self.ext_out,
            self.states_ts,
            self.noise_out,
            self.do_delay,
            force_reinit,
            self.use_cpu,
            self.N,
            self.nodes,
            self.duration_msec,
            self.TR_msec,
            self.states_sampling_msec,
            self.rand_seed,
            float(self.dt),
            float(self.bw_dt),
        )
        # avoid reinitializing GPU in the next runs
        # of the same group
        self.last_N = self.N
        self.last_config = curr_config
        self.it += 1
        # process output
        self._process_out(out)

    def _process_out(self, out):
        """
        Assigns model outputs (as arrays) to object attributes
        with correct shapes, names and types.

        Parameters
        ----------
        out: :obj:`tuple`
            output of `run_simulations` function

        Notes
        -----
        The simulation outputs are assigned to the following object attributes:
            - init_time: :obj:`float`
                initialization time of the simulations
            - run_time: :obj:`float`
                run time of the simulations
            - sim_bold : :obj:`np.ndarray`
                simulated BOLD time series. Shape: (N_SIMS, duration/TR, nodes)
        If `do_fc` is True, additionally includes:
            - sim_fc_trils : :obj:`np.ndarray`
                simulated FC lower triangle. Shape: (N_SIMS, n_pairs)
        If `do_fcd` is True, additionally includes:
            - sim_fcd_trils : :obj:`np.ndarray`
                simulated FCD lower triangle. Shape: (N_SIMS, n_window_pairs)
        If `ext_out` is True, additionally includes:
            - _sim_states: :obj:`np.ndarray`
                Model state variables. Shape: (n_vars, N_SIMS*nodes[*duration/TR])
            - _global_bools: :obj:`np.ndarray`
                Global boolean variables. Shape: (n_bools, N_SIMS)
            - _global_ints: :obj:`np.ndarray`
                Global integer variables. Shape: (n_ints, N_SIMS)
        If `noise_out` is True, additionally includes:
            - _noise: :obj:`np.ndarray`
                Noise segment array. Shape: (nodes*noise_time_steps*10*Model.n_noise,)
        If `noise_out` is True and noise segmenting is on, additionally includes:
            - _shuffled_nodes: :obj: `np.ndarray`
                Node shufflings in each noise repeate. Shape: (noise_repeats, nodes)
            - _shuffled_ts: :obj: `np.ndarray`
                Time step shufflings in each noise repeate. Shape: (noise_repeats, noise_time_steps)                
        """
        # assign the output to object attributes
        # and reshape them to (N, ...)
        # in all cases init time, run time, bold, fc and fcd will be returned
        (
            self.init_time, 
            self.run_time, 
            sim_bold, 
         ) = out[:3]
        self.sim_bold = sim_bold.reshape(self.N, -1, self.nodes)
        idx = 3
        if self.do_fc:
            sim_fc_trils = out[idx]
            self.sim_fc_trils = sim_fc_trils.reshape(self.N, -1)
            idx += 1
            if self.do_fcd:
                sim_fcd_trils = out[idx]
                self.sim_fcd_trils = sim_fcd_trils.reshape(self.N, -1)
                idx += 1
        # assign the additional outputs in indices 5: if
        # they are returned depending on `ext_out`, `noise_out`
        # and `noise_segment_flag`
        if self.ext_out:
            (
                sim_states,
                global_bools,
                global_ints,
            ) = out[idx:idx+3]
            idx += 3
            self._sim_states = sim_states
            self._global_bools = global_bools
            self._global_ints = global_ints
        if self.noise_out:
            self._noise = out[idx]
            idx += 1
            if noise_segment_flag:
                (
                    self._shuffled_nodes, 
                    self._shuffled_ts
                ) = out[idx:]

    def get_noise(self):
        """
        Get the (recreated) noise time series. This requires recreation
        of the noise array if noise segmenting is on. Noise will be
        recreated based on shuffling indices of nodes and time steps,
        similar to how it is done in the core code.

        Returns
        -------
        noise: :obj:`np.ndarray`
            Noise segment array. Shape: (n_noise, nodes, time_steps, 10)
        """
        # raise an error if noise was not saved in the simulation
        if not self.noise_out:
            raise ValueError(
                "Noise was not saved in the simulation."
                " Set `noise_out` to true and redo the simulation."
                )
        # raise an error if simulation is not run yet
        if not hasattr(self, "_noise"):
            raise ValueError(
                "Simulation must be run for the noise to be calculated"
                )
        # reshape the whole noise array into (nodes, ts) when
        # noise segmenting is off
        if not noise_segment_flag:
            return (
                self._noise.
                reshape(int(self.duration*1000), 10, self.nodes, -1)
                .transpose(3, 2, 0, 1)
            )
        # otherwise recreate the noise time series otherwise
        # first reshape the noise into (noise_segment_length in msec, 10, nodes, n_noise)
        noise_segment = self._noise.reshape(
            int(self.noise_segment_length*1000), 10, self.nodes, -1
        )
        # then shuffle the noise in each repeat based on the shuffling indices
        noise_all = []
        for r in range(self._shuffled_nodes.shape[0]):
            noise_all.append(
                noise_segment[
                    self._shuffled_ts[r] # shuffle time steps
                ][
                    :, :, self._shuffled_nodes[r] # shuffle nodes
                ]
            )
        # conctaenate the shuffled noise repeats
        noise_all = np.concatenate(noise_all, axis=0)
        # crop the last part of the noise that is not used
        noise_all = noise_all[:int(self.duration*1000)]
        # transpose to (n_noise, nodes, time_steps, 10)
        return noise_all.transpose(3, 2, 0, 1)

    def clear(self):
        """
        Clear the simulation outputs
        """
        for attr in ["sim_bold", "sim_fc_trils", "sim_fcd_trils", "sim_states"]:
            if hasattr(self, attr):
                delattr(self, attr)
        gc.collect()

    def _problem_init(self, problem):
        """
        Extends BNMProblem initialization if needed.
        By default it doesn't do anything.

        Parameters
        ----------
        problem: :obj:`cubnm.optimize.BNMProblem`
            optimization problem object
        """
        pass

    def _problem_evaluate(self, problem, X, out, *args, **kwargs):
        """
        Extends BNMProblem evaluation if needed.
        By default it doesn't do anything.

        Parameters
        ----------
        X : :obj:`np.ndarray`
            the normalized parameters of current population in range [0, 1]. 
            Shape: (N, ndim)
        out : :obj:`dict`
            the output dictionary to store the results with keys 'F' and 'G'.
            Currently only 'F' (cost) is used.
        *args, **kwargs
        """
        pass

    def score(self, emp_fc_tril=None, emp_fcd_tril=None, emp_bold=None):
        """
        Calcualates individual goodness-of-fit terms and aggregates them.

        Parameters
        --------
        emp_fc_tril: :obj:`np.ndarray` or None
            1D array of empirical FC lower triangle. Shape: (edges,)
        emp_fcd_tril: :obj:`np.ndarray` or None
            1D array of empirical FCD lower triangle. Shape: (window_pairs,)
        emp_bold: :obj:`np.ndarray` or None
            cleaned and parcellated empirical BOLD time series. Shape: (nodes, volumes)
            Motion outliers should either be excluded (not recommended as it disrupts
            the temporal structure) or replaced with zeros.
            If provided emp_fc_tril and emp_fcd_tril will be ignored.

        Returns
        -------
        scores: :obj:`pd.DataFrame`
            The goodness of fit measures (columns) of each simulation (rows)
        """
        # calculate empirical FC and FCD if BOLD is provided
        if emp_bold is not None:
            if (emp_fc_tril is not None) or (emp_fcd_tril is not None):
                print(
                    "Warning: Both empirical BOLD and empirical FC/FCD are"
                    " provided. Empirical FC/FCD will be calculated based on"
                    " BOLD and will be overwritten."
                )
            if self.do_fc:
                emp_fc_tril = utils.calculate_fc(emp_bold, self.exc_interhemispheric, return_tril=True)
            if self.do_fcd:
                emp_fcd_tril = utils.calculate_fcd(
                    emp_bold, 
                    self.window_size, 
                    self.window_step, 
                    drop_edges=self.fcd_drop_edges,
                    exc_interhemispheric=self.exc_interhemispheric,
                    return_tril=True,
                    return_dfc = False
                )
        # + => aim to maximize; - => aim to minimize
        # get list of columns to be calculated
        # based on availability of FC and FCD (simulated and empirical)
        # and gof_terms
        columns = []
        if self.do_fc and (emp_fc_tril is not None):
            columns += list(set(["+fc_corr", "-fc_diff", "-fc_normec"]) & set(self.gof_terms))
        if self.do_fcd and (emp_fcd_tril is not None):
            columns += list(set(["-fcd_ks"]) & set(self.gof_terms))
        columns += ["+gof"]
        scores = pd.DataFrame(columns=columns, dtype=float)
        # calculate GOF
        for idx in range(self.N):
            for column in columns:
                if column == "+fc_corr":                    
                    scores.loc[idx, column] = scipy.stats.pearsonr(
                        self.sim_fc_trils[idx], emp_fc_tril
                    ).statistic
                elif column == "-fc_diff":
                    scores.loc[idx, column] = -np.abs(
                        self.sim_fc_trils[idx].mean() - emp_fc_tril.mean()
                    )
                elif column == "-fc_normec":
                    scores.loc[idx, column] = -utils.fc_norm_euclidean(
                        self.sim_fc_trils[idx], emp_fc_tril
                    )
                elif column == "-fcd_ks":
                    scores.loc[idx, column] = -scipy.stats.ks_2samp(
                        self.sim_fcd_trils[idx], emp_fcd_tril
                    ).statistic
            # combined the selected terms into gof (that should be maximized)
            scores.loc[idx, "+gof"] = 0
            for term in self.gof_terms:
                scores.loc[idx, "+gof"] += scores.loc[idx, term]
        return scores
    
    def _get_save_data(self):
        """
        Get the simulation outputs and parameters to be saved to disk

        Returns
        -------
        out_data: :obj:`dict`
            dictionary of simulation outputs and parameters
        """
        out_data = dict(
            sim_bold=self.sim_bold,
        )
        if self.do_fc:
            out_data['sim_fc_trils'] = self.sim_fc_trils
        if self.do_fcd:
            out_data['sim_fcd_trils'] = self.sim_fcd_trils
        if self.ext_out:
            out_data['sim_states'] = self.sim_states
        out_data.update(self.param_lists)
        return out_data

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
        assert self.out_dir is not None, (
            "Cannot save the simulations when `.out_dir`"
            "is not set"
        )
        sims_dir = self.out_dir
        os.makedirs(sims_dir, exist_ok=True)
        out_data = self._get_save_data()
        if save_as == "npz":
            # TODO: use more informative filenames
            np.savez_compressed(os.path.join(sims_dir, f"it{self.it}.npz"), **out_data)
        elif save_as == "txt":
            raise NotImplementedError

    @classmethod
    def _get_test_configs(cls, cpu_gpu_identity=False):
        """
        Get configs for testing the simulations

        Parameters
        ----------
        cpu_gpu_identity: :obj:`bool`, optional
            indicates whether configs are for CPU/GPU identity tests
            in which case force_cpu is not included in the configs
            since tests will be done on both CPU and GPU

        Returns
        -------
        configs: :obj:`dict` of :obj:`list`
        """
        configs = {}
        # for compatibality with previously-stored
        # expected simulation files force_cpu must
        # be the first item
        # TODO: fix this and recompute the expected files
        if not cpu_gpu_identity:
            configs['force_cpu'] = [0, 1]
        configs['do_delay'] = [0, 1]
        return configs

    @classmethod
    def _get_test_instance(cls, opts):
        """
        Initializes an instance that is used in tests

        Parameters
        ----------
        opts: :obj:`dict`
            dictionary of test options

        Returns
        -------
        sim_group: :obj:`cubnm.sim.SimGroup`
            simulation group object of the test simulation
            which is not run yet
        """
        do_delay = bool(opts.pop('do_delay'))
        if do_delay:
            sc_dist = datasets.load_sc('length', 'schaefer-100')
            dt = '1.0'
        else:
            sc_dist = None
            dt = '0.1'
        sim_group = cls(
            duration=60,
            TR=1,
            sc=datasets.load_sc('strength', 'schaefer-100'),
            sc_dist=sc_dist,
            dt = dt,
            sim_verbose=False,
            **opts
        )
        return sim_group

class rWWSimGroup(SimGroup):
    model_name = "rWW"
    global_param_names = ["G"]
    regional_param_names = ["wEE", "wEI", "wIE"]
    sel_state_var = 'r_E' # TODO: use all states
    n_noise = 2
    def __init__(self, 
                 *args, 
                 do_fic = True,
                 max_fic_trials = 0,
                 fic_penalty = True,
                 fic_i_sampling_start = 1000,
                 fic_i_sampling_end = 10000,
                 fic_init_delta = 0.02,
                 **kwargs):
        """
        Group of reduced Wong-Wang simulations (Deco 2014) 
        that are executed in parallel

        Parameters
        ---------
        do_fic: :obj:`bool`, optional
            do analytical (Demirtas 2019) & numerical (Deco 2014) 
            Feedback Inhibition Control.
            If provided wIE parameters will be ignored
        max_fic_trials: :obj:`int`, optional
            maximum number of trials for FIC numerical adjustment.
            If set to 0, FIC will be done only analytically
        fic_penalty: :obj:`bool`, optional
            penalize deviation from FIC target mean rE of 3 Hz
        fic_i_sampling_start: :obj:`int`, optional
            starting time of numerical FIC I_E sampling (msec)
        fic_i_sampling_end: :obj:`int`, optional
            end time of numerical FIC I_E sampling (msec)
        fic_init_delta: :obj:`float`, optional
        *args, **kwargs:
            see :class:`cubnm.sim.SimGroup` for details

        Attributes
        ----------
        param_lists: :obj:`dict` of :obj:`np.ndarray`
            dictionary of parameter lists, including
                - 'G': global coupling. Shape: (N_SIMS,)
                - 'wEE': local excitatory self-connection strength. Shape: (N_SIMS, nodes)
                - 'wEI': local inhibitory self-connection strength. Shape: (N_SIMS, nodes)
                - 'wIE': local excitatory to inhibitory connection strength. Shape: (N_SIMS, nodes)
                - 'v': conduction velocity. Shape: (N_SIMS,)

        Example
        -------
        To see example usage in grid search and evolutionary algorithms
        see :mod:`cubnm.optimize`.

        Here, as an example on how to use SimGroup independently, we
        will run a single simulation and save the outputs to disk. ::

            from cubnm import sim, datasets

            sim_group = sim.rWWSimGroup(
                duration=60,
                TR=1,
                sc=datasets.load_sc('strength', 'schaefer-100'),
            )
            sim_group.N = 1
            sim_group.param_lists['G'] = np.repeat(0.5, N_SIMS)
            sim_group.param_lists['wEE'] = np.full((N_SIMS, nodes), 0.21)
            sim_group.param_lists['wEI'] = np.full((N_SIMS, nodes), 0.15)
            sim_group.run()
        """
        self.do_fic = do_fic
        self.max_fic_trials = max_fic_trials
        self.fic_penalty = fic_penalty
        self.fic_i_sampling_start = fic_i_sampling_start
        self.fic_i_sampling_end = fic_i_sampling_end
        self.fic_init_delta = fic_init_delta
        # parent init must be called here because
        # it sets .last_config which may include some
        # model-specific attributes (e.g. self.do_fic)
        super().__init__(*args, **kwargs)
        self.ext_out = self.ext_out | self.do_fic
        if self.serial_nodes and (self.max_fic_trials > 0):
            print(
                "Numerical FIC is not supported in serial_nodes mode. "
                "Setting max_fic_trials to 0."
            )
            self.max_fic_trials = 0

    @SimGroup.N.setter
    def N(self, N):
        super(rWWSimGroup, rWWSimGroup).N.__set__(self, N)
        if self.do_fic:
            self.param_lists["wIE"] = np.zeros((self._N, self.nodes), dtype=float)

    def get_config(self, *args, **kwargs):
        config = super().get_config(*args, **kwargs)
        config.update(
            do_fic=self.do_fic, 
            max_fic_trials=self.max_fic_trials,
            fic_penalty=self.fic_penalty,
            fic_i_sampling_start=self.fic_i_sampling_start,
            fic_i_sampling_end=self.fic_i_sampling_end,
            fic_init_delta=self.fic_init_delta,
        )
        return config
    
    @property
    def _model_config(self):
        """
        Internal model configuration used in the simulation
        """
        model_config = super()._model_config
        model_config.update({
            'do_fic': str(int(self.do_fic)),
            'max_fic_trials': str(self.max_fic_trials),
            'I_SAMPLING_START': str(self.fic_i_sampling_start),
            'I_SAMPLING_END': str(self.fic_i_sampling_end),
            'init_delta': str(self.fic_init_delta),
        })
        return model_config
    
    def _set_default_params(self):
        """
        Set default parameters for the simulations.
        This is used in tests.
        """
        super()._set_default_params()
        self.param_lists["G"] = np.repeat(0.5, self.N)
        self.param_lists["wEE"] = np.full((self.N, self.nodes), 0.21)
        self.param_lists["wEI"] = np.full((self.N, self.nodes), 0.15)
        if not self.do_fic:
            self.param_lists["wIE"] = np.full((self.N, self.nodes), 1.0)

    def _process_out(self, out):
        super()._process_out(out)
        if self.ext_out:
            # name and reshape the states
            self.sim_states = {
                'I_E': self._sim_states[0],
                'I_I': self._sim_states[1],
                'r_E': self._sim_states[2],
                'r_I': self._sim_states[3],
                'S_E': self._sim_states[4],
                'S_I': self._sim_states[5],
            }
            if self.states_ts:
                for k in self.sim_states:
                    self.sim_states[k] = self.sim_states[k].reshape(self.N, -1, self.nodes)
        if self.do_fic:
            # FIC stats
            self.fic_unstable = self._global_bools[0]
            self.fic_failed = self._global_bools[1]
            self.fic_ntrials = self._global_ints[0]
            # write FIC (+ numerical adjusted) wIE to param_lists
            self.param_lists["wIE"] = self._regional_params[2].reshape(self.N, -1)
    
    def score(self, emp_fc_tril=None, emp_fcd_tril=None, emp_bold=None, fic_penalty_scale=2):
        """
        Calcualates individual goodness-of-fit terms and aggregates them.
        In FIC models also calculates fic_penalty.

        Parameters
        --------
        emp_fc_tril: :obj:`np.ndarray`
            1D array of empirical FC lower triangle. Shape: (edges,)
        emp_fcd_tril: :obj:`np.ndarray`
            1D array of empirical FCD lower triangle. Shape: (window_pairs,)
        emp_bold: :obj:`np.ndarray` or None
            cleaned and parcellated empirical BOLD time series. Shape: (nodes, volumes)
            Motion outliers should either be excluded (not recommended as it disrupts
            the temporal structure) or replaced with zeros.
            If provided emp_fc_tril and emp_fcd_tril will be ignored.
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
        scores = super().score(emp_fc_tril, emp_fcd_tril, emp_bold)
        # calculate FIC penalty
        if self.do_fic & self.fic_penalty:
            if self.states_ts:
                mean_r_E = self.sim_states["r_E"].mean(axis=1)
            else:
                mean_r_E = self.sim_states["r_E"]
            for idx in range(self.N):
                diff_r_E = np.abs(mean_r_E[idx, :] - 3)
                if (diff_r_E > 1).sum() > 0:
                    diff_r_E[diff_r_E <= 1] = np.NaN
                    scores.loc[idx, "-fic_penalty"] = (
                        -np.nansum(1 - np.exp(-0.05 * (diff_r_E - 1)))
                        * fic_penalty_scale / self.nodes
                    )
                else:
                    scores.loc[idx, "-fic_penalty"] = 0
        return scores
    
    def _get_save_data(self):
        """
        Get the simulation outputs and parameters to be saved to disk

        Returns
        -------
        out_data: :obj:`dict`
            dictionary of simulation outputs and parameters
        """
        out_data = super()._get_save_data()
        if self.do_fic:
            out_data.update(
                fic_unstable=self.fic_unstable,
                fic_failed=self.fic_failed,
                fic_ntrials=self.fic_ntrials,
            )
        return out_data
    
    def clear(self):
        """
        Clear the simulation outputs
        """
        super().clear()
        for attr in ["fic_unstable", "fic_failed", "fic_ntrials"]:
            if hasattr(self, attr):
                delattr(self, attr)
        gc.collect()

    def _problem_init(self, problem):
        """
        Extends BNMProblem initialization and includes FIC penalty
        if indicated.

        Parameters
        ----------
        problem: :obj:`cubnm.optimize.BNMProblem`
            optimization problem object
        """
        if (self.do_fic) & ("wIE" in problem.het_params):
            raise ValueError(
                "In rWW wIE should not be specified as a heterogeneous parameter when FIC is done"
            )
        if problem.multiobj:
            if self.do_fic & self.fic_penalty:
                problem.obj_names.append("+fic_penalty")
                problem.n_obj += 1

    def _problem_evaluate(self, problem, X, out, *args, **kwargs):
        """
        Extends BNMProblem evaluation and includes FIC penalty
        in the cost function if indicated.

        Parameters
        ----------
        X : :obj:`np.ndarray`
            the normalized parameters of current population in range [0, 1]. 
            Shape: (N, ndim)
        out : :obj:`dict`
            the output dictionary to store the results with keys 'F' and 'G'.
            Currently only 'F' (cost) is used.
        *args, **kwargs
        """
        # Note: scores (inidividual GOF measures) is passed on in kwargs 
        # in the internal mechanism of pymoo evaluation function
        scores = kwargs["scores"][-1]
        if self.do_fic & self.fic_penalty:
            if problem.multiobj:
                out["F"] = np.concatenate(
                    [out["F"], -scores.loc[:, ["-fic_penalty"]].values], axis=1
                )
            else:
                out["F"] -= scores.loc[:, "-fic_penalty"].values
                

    @classmethod
    def _get_test_configs(cls, cpu_gpu_identity=False):
        """
        Get configs for testing the simulations

        Parameters
        ----------
        cpu_gpu_identity: :obj:`bool`, optional
            indicates whether configs are for CPU/GPU identity tests
            in which case force_cpu is not included in the configs
            since tests will be done on both CPU and GPU

        Returns
        -------
        configs: :obj:`dict` of :obj:`list`
        """
        configs = super()._get_test_configs(cpu_gpu_identity)
        configs.update({
            'do_fic': [0, 1]
        })
        return configs

    @classmethod
    def _get_test_instance(cls, opts):
        """
        Initializes an instance that is used in tests

        Parameters
        ----------
        opts: :obj:`dict`
            dictionary of test options

        Returns
        -------
        sim_group: :obj:`cubnm.sim.rWWSimGroup`
            simulation group object of the test simulation
            which is not run yet
        """
        # initialze sim group
        sim_group = super()._get_test_instance(opts)
        # set do_fic
        sim_group.do_fic = bool(opts['do_fic'])
        return sim_group

class rWWExSimGroup(SimGroup):
    model_name = "rWWEx"
    global_param_names = ["G"]
    regional_param_names = ["w", "I0", "sigma"]
    sel_state_var = 'r' # TODO: use all states
    n_noise = 1
    def __init__(self, *args, **kwargs):
        """
        Group of reduced Wong-Wang simulations (excitatory only, Deco 2013) 
        that are executed in parallel

        Parameters
        ---------
        *args, **kwargs:
            see :class:`cubnm.sim.SimGroup` for details

        Attributes
        ----------
        param_lists: :obj:`dict` of :obj:`np.ndarray`
            dictionary of parameter lists, including
                - 'G': global coupling. Shape: (N_SIMS,)
                - 'w': local excitatory self-connection strength. Shape: (N_SIMS, nodes)
                - 'I0': local external input current. Shape: (N_SIMS, nodes)
                - 'sigma': local noise sigma. Shape: (N_SIMS, nodes)
                - 'v': conduction velocity. Shape: (N_SIMS,)

        Example
        -------
        To see example usage in grid search and evolutionary algorithms
        see :mod:`cubnm.optimize`.

        Here, as an example on how to use SimGroup independently, we
        will run a single simulation and save the outputs to disk. ::

            from cubnm import sim, datasets

            sim_group = sim.rWWExSimGroup(
                duration=60,
                TR=1,
                sc=datasets.load_sc('strength', 'schaefer-100'),
            )
            sim_group.N = 1
            sim_group.param_lists['G'] = np.repeat(0.5, N_SIMS)
            sim_group.param_lists['w'] = np.full((N_SIMS, nodes), 0.9)
            sim_group.param_lists['I0'] = np.full((N_SIMS, nodes), 0.3)
            sim_group.param_lists['sigma'] = np.full((N_SIMS, nodes), 0.001)
            sim_group.run()
        """
        super().__init__(*args, **kwargs)

    def _set_default_params(self):
        """
        Set default parameters for the simulations.
        This is used in tests.
        """
        super()._set_default_params()
        self.param_lists["G"] = np.repeat(0.5, self.N)
        self.param_lists["w"] = np.full((self.N, self.nodes), 0.9)
        self.param_lists["I0"] = np.full((self.N, self.nodes), 0.3)
        self.param_lists["sigma"] = np.full((self.N, self.nodes), 0.001)
    
    def _process_out(self, out):
        super()._process_out(out)
        if self.ext_out:
            # name and reshape the states
            self.sim_states = {
                'x': self._sim_states[0],
                'r': self._sim_states[1],
                'S': self._sim_states[2],
            }
            if self.states_ts:
                for k in self.sim_states:
                    self.sim_states[k] = self.sim_states[k].reshape(self.N, -1, self.nodes)

class KuramotoSimGroup(SimGroup):
    model_name = "Kuramoto"
    global_param_names = ["G"]
    regional_param_names = ["init_theta", "omega", "sigma"]
    sel_state_var = 'theta'
    n_noise = 1
    def __init__(self, 
                 *args, 
                 random_init_theta = True,
                 **kwargs):
        """
        Group of Kuramoto simulations that are executed in parallel

        Parameters
        ---------
        *args, **kwargs:
            see :class:`cubnm.sim.SimGroup` for details
        random_init_theta : :obj:`bool`, optional
            Set initial theta by randomly sampling from a uniform distribution 
            [0, 2*pi].

        Attributes
        ----------
        param_lists: :obj:`dict` of :obj:`np.ndarray`
            dictionary of parameter lists, including
                - 'G': global coupling. Shape: (N_SIMS,)
                - 'init_theta': initial theta. Randomly sampled from a uniform distribution 
                    [0, 2*pi] by default. Shape: (N_SIMS, nodes)
                - 'omega': intrinsic frequency. Shape: (N_SIMS, nodes)
                - 'sigma': local noise sigma. Shape: (N_SIMS, nodes)
                - 'v': conduction velocity. Shape: (N_SIMS,)

        Example
        -------
        To see example usage in grid search and evolutionary algorithms
        see :mod:`cubnm.optimize`.

        Here, as an example on how to use SimGroup independently, we
        will run a single simulation and save the outputs to disk. ::

            from cubnm import sim, datasets

            sim_group = sim.KuramotoSimGroup(
                duration=60,
                TR=1,
                sc=datasets.load_sc('strength', 'schaefer-100'),
            )
            sim_group.N = 1
            sim_group.param_lists['G'] = np.repeat(0.5, N_SIMS)
            sim_group.param_lists['omega'] = np.full((N_SIMS, nodes), np.pi)
            sim_group.param_lists['sigma'] = np.full((N_SIMS, nodes), 0.17)
            sim_group.run()
        """
        super().__init__(*args, **kwargs)
        self.random_init_theta = random_init_theta

    @SimGroup.N.setter
    def N(self, N):
        super(KuramotoSimGroup, KuramotoSimGroup).N.__set__(self, N)
        if self.random_init_theta:
            # sample from uniform distribution of [0, 2*pi] across nodes and
            # repeat it across simulations
            # use the same random seed as the simulation noise
            rng = np.random.default_rng(self.rand_seed)
            self.param_lists["init_theta"] = np.tile(rng.uniform(0, 2 * np.pi, self.nodes), (self._N, 1))
        else:
            self.param_lists["init_theta"] = np.zeros((self._N, self.nodes), dtype=float)
            print("Warning: init_theta is set to zero")

    def _set_default_params(self):
        """
        Set default parameters for the simulations.
        This is used in tests.
        """
        super()._set_default_params()
        self.param_lists["G"] = np.repeat(0.5, self.N)
        self.param_lists["omega"] = np.full((self.N, self.nodes), np.pi)
        self.param_lists["sigma"] = np.full((self.N, self.nodes), 0.17)
    
    def _process_out(self, out):
        super()._process_out(out)
        if self.ext_out:
            # name and reshape the states
            self.sim_states = {
                'theta': self._sim_states[0],
            }
            if self.states_ts:
                for k in self.sim_states:
                    self.sim_states[k] = self.sim_states[k].reshape(self.N, -1, self.nodes)

class MultiSimGroupMixin:
    def __init__(self, sim_groups):
        """
        Mixin for combining multiple simulation groups into one,
        which is intended for batch optimization, i.e. running multiple
        optimizations at the same time on GPU.
        """
        self.children = sim_groups
        # copy all attributes of the first child
        # it is important to deep copy the attributes as
        # some (e.g. param_lists) are modifiable
        # this also assumes that all children have the same attributes
        # except SC which can be variable
        # TODO: add a check that this is the case
        self.__dict__.update(copy.deepcopy(sim_groups[0].__dict__))

    @property
    def N(self):
        return sum([child.N for child in self.children])
    
    def run(self, **kwargs):
        """
        Runs merged simulations of all children
        by concatenating SCs and parameters
        and then running all simulations in parallel
        as a single merged simulation group.

        Parameters
        ----------
        **kwargs
            keyword arguments to be passed to `cubnm.sim.SimGroup.run` method
        """
        # concatenate SCs while allowing variable SCs
        # across children
        for child_i, child in enumerate(self.children):
            # make a copy of current child's SCs
            child_scs = child.sc.copy()
            # make scs 3D (n_scs, nodes, nodes)
            if (child_scs.ndim == 2):
                child_scs = child_scs[None, :, :]
            # create sc_indices as all zeros if only one SC is used
            # in current child
            if (child_scs.shape[0] == 1):
                child_sc_indices = np.zeros(child.N, dtype=int)
            else:
                child_sc_indices = np.array(child.sc_indices).copy()
            # initialize merged sim group SC as the first child SC(s)
            if child_i == 0:
                self.sc = child_scs
                self.sc_indices = child_sc_indices
            else:
                # for subsequent children check if SC is already in self.sc
                # (and then update the indices to point to correct sc for each
                # simulation), otherwise add the SC to self.sc (and update the
                # indices with a new index pointing to the new SC)
                sc_indices = child_sc_indices
                for sc_i, sc in enumerate(child_scs):
                    if ~np.isclose(sc, self.sc).all(axis=(1, 2)).any():
                        # if SC is not already in self.sc, add it
                        # as a new SC with a new index
                        self.sc = np.concatenate([self.sc, sc[None, :, :]], axis=0)
                        new_sc_idx = self.sc_indices.max()+1
                        sc_indices[sc_indices == sc_i] = new_sc_idx
                    else:
                        # if SC is already in self.sc, find its index
                        # and just update the sc_indices to reflect the duplicate
                        # which already exist. In this case there is no need to
                        # add the SC to self.sc as it already exists there
                        dupl_sc_idx = np.where(np.isclose(sc, self.sc).all(axis=(1, 2)))[0][0]
                        sc_indices[sc_indices == sc_i] = dupl_sc_idx
                    # update self.sc_indices
                    self.sc_indices = np.concatenate([self.sc_indices, sc_indices], axis=0)
        # convert sc back to 2D if same SC is used in all simulations
        self.sc = np.squeeze(self.sc)
        # concatenate parameters
        for param in self.param_lists:
            self.param_lists[param] = np.concatenate(
                [child.param_lists[param] for child in self.children]
            )
        # run the simulations
        super().run(**kwargs)

    def _process_out(self, out):
        """
        Divides simulations outputs across individual
        children SimGroup objects which will respectively
        convert the output to attributes with correct shapes,
        names and types. See `cubnm.sim.SimGroup._process_out`
        for details.

        Parameters
        ----------
        out: :obj:`tuple`
            output of `run_simulations` function
        """
        start_idx = 0
        for child in self.children:
            end_idx = start_idx + child.N
            # break down self._regional_params and self._global_params
            # (which may be modified in run) into children as they may
            # be used in _process_out of children (e.g. in rWW)
            child._regional_params = self._regional_params[:, start_idx:end_idx]
            child._global_params = self._global_params[:, start_idx:end_idx]
            # break down simulation output elements
            out_child = []
            out_child.append(out[0]) # init time
            out_child.append(out[1]) # run time
            for var_idx in range(2, len(out)): # other outputs
                # determine which index has N elements
                # TODO: this is very "hacky". Ideally sim_idx should always be at
                # the first axis. It can easily break if e.g. BOLD TRs is equal to N!
                sim_axis = np.where(np.array(out[var_idx].shape) == self.N)[0][0]
                # break down the output into children based on the axis
                if sim_axis == 0:
                    out_child.append(out[var_idx][start_idx:end_idx])
                elif sim_axis == 1:
                    out_child.append(out[var_idx][:, start_idx:end_idx])
                elif sim_axis == 2:
                    out_child.append(out[var_idx][:, :, start_idx:end_idx])
            # update start_idx for the next child
            start_idx += child.N
            # process the output of the child
            child._process_out(out_child)

def create_multi_sim_group(sim_group_cls):
    """
    Dynamically creates a MultiSimGroup class by combining
    a model's specific <Model>SimGroup class with MultiSimGroupMixin,
    which can be used in batch optimization.

    Parameters
    ----------
    sim_group_cls: :obj:`type`
        e.g. rWWSimGroup, rWWExSimGroup, KuramotoSimGroup
    
    Returns
    -------
    MultiSimGroup: :obj:`type`
        MultiSimGroup class that can be used in batch optimization
    """
    MultiSimGroup = type(
        'MultiSimGroup',
        (MultiSimGroupMixin, sim_group_cls),
        {}
    )
    return MultiSimGroup