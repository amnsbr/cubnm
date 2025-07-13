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
import multiprocessing

from cubnm._core import run_simulations, set_const
from cubnm._setup_opts import (
    many_nodes_flag, gpu_enabled_flag, 
    max_nodes_reg, max_nodes_many,
    noise_segment_flag
)
from cubnm import utils, datasets
from . import _version
__version__ = _version.get_versions()['version']

try:
    import cupy as cp
    has_cupy = True
except ImportError:
    cp = None
    has_cupy = False



# NaN values in the scores are replaced with the worst possible scores
# listed below.
WORST_SCORES = {
    "+fc_corr": -1.0,
    "-fc_diff": -2.0,
    "-fc_normec": -1.0,
    "-fcd_ks": -1.0,
}

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
        window_size=30,
        window_step=5,
        sim_seed=0,
        exc_interhemispheric=False,
        force_cpu=False,
        force_gpu=False,
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
            path to structural connectome strengths (as an unlabled .txt/.npy)
            or a numpy array. Shape: (nodes, nodes)
            If asymmetric, rows are sources and columns are targets.
        sc_dist: :obj:`str` or :obj:`np.ndarray`
            path to structural connectome distances (as an unlabled .txt)
            or a numpy array. Shape: (nodes, nodes)
            If provided ``'v'`` (velocity) will be a free parameter and there
            will be delay in inter-regional connections.
            If asymmetric, rows are sources and columns are targets.
        out_dir: {:obj:`str` or 'same' or None}
            output directory

            - :obj:`str`: will create a directory in the provided path
            - ``'same'``: will create a directory named based on sc
                (only should be used when sc is a path and not a numpy array)
            - ``None``: will not have an output directory (and cannot save outputs)

        dt: :obj:`decimal.Decimal` or :obj:`str`
            model integration time step (in msec)
        bw_dt: :obj:`decimal.Decimal` or :obj:`str`
            Ballon-Windkessel integration time step (in msec)
        ext_out: :obj:`bool`
            return model state variables to self.sim_states
        states_ts: :obj:`bool`
            return time series of model states to self.sim_states
            Note that this will increase the memory usage and is not
            recommended for large number of simulations (e.g. in a grid search)
        states_sampling: :obj:`float`
            sampling rate of model states in seconds.
            Default is None, which uses BOLD TR as the sampling rate.
        noise_out: :obj:`bool`
            return noise time series
        do_fc: :obj:`bool`
            calculate simulated functional connectivity (FC)
        do_fcd: :obj:`bool`
            calculate simulated functional connectivity dynamics (FCD)
        window_size: :obj:`int`
            dynamic FC window size (in seconds)
            will be converted to N TRs (nearest even number)
            The actual window size is number of TRs + 1 (including center)
        window_step: :obj:`int`
            dynamic FC window step (in seconds)
            will be converted to N TRs
        sim_seed: :obj:`int`
            seed used for the noise simulation
        exc_interhemispheric: :obj:`bool`
            excluded interhemispheric connections from sim FC and FCD calculations
        force_cpu: :obj:`bool`
            use CPU for the simulations (even if GPU is available). If set
            to False the program might use GPU or CPU depending on GPU
            availability
        force_gpu: :obj:`bool`
            on some HPC/HTC systems occasionally GPUtil might not detect an 
            available GPU device. Use this if there is a GPU available but
            is not being used for the simulation. If set to True but a GPU
            is not available will lead to errors.
        gof_terms: :obj:`list` of :obj:`str`
            list of goodness-of-fit terms to be used for scoring. May include:

            - ``'-fcd_ks'``: negative Kolmogorov-Smirnov distance of FCDs
            - ``'+fc_corr'``: Pearson correlation of FCs
            - ``'-fc_diff'``: negative absolute difference of FC means
            - ``'-fc_normec'``: negative Euclidean distance of FCs \
                divided by max EC [sqrt(n_pairs*4)]

        bw_params: {'friston2003' or 'heinzle2016-3T' or :obj:`dict`}
            see :func:`cubnm.utils.get_bw_params` for details
        bold_remove_s: :obj:`float`
            remove the first bold_remove_s seconds from the simulated BOLD
            in FC, FCD and mean state calculations (but the entire BOLD will
            be returned to .sim_bold)
        fcd_drop_edges: :obj:`bool`
            drop the edge windows in FCD calculations
        noise_segment_length: :obj:`float` or None
            in seconds, length of the noise segments in the simulations
            The noise segment will be repeated after shuffling
            of nodes and time points. To generate noise for the entire
            simulation without repetition, set this to None.
            Note that varying the noise segment length will result
            in a different noise array even if seed is fixed (but
            fixed combination of seed and noise_segment_length will
            result in reproducible noise)
        sim_verbose: :obj:`bool`
            verbose output of the simulation including details of
            simulations and a progress bar.
        progress_interval: :obj:`int`
            msec; interval of progress updates in the simulation
            Only used if ``sim_verbose`` is ``True``

        Attributes
        ---------
        param_lists: :obj:`dict` of :obj:`np.ndarray`
            dictionary of parameter lists, including

            - global parameters with shape (N_SIMS,)
            - regional parameters with shape (N_SIMS, nodes)
            - ``'v'``: conduction velocity. Shape: (N_SIMS,)

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
            state_names: :obj:`list` of :obj:`str`
                names of the state variables
            sel_state_var: :obj:`str`
                name of the state variable used in the tests
            n_noise: :obj:`int`
                number of noise elements per node per time point
                (e.g. 2 if there are noise to E and I neuronal populations)
        And they must implement the following methods:
            _set_default_params: set default (example) parameters for the simulations
        """
        self.input_sc = sc
        if isinstance(sc, (str, os.PathLike)):
            if sc.endswith(".txt"):
                self.sc = np.loadtxt(self.input_sc)
            elif sc.endswith(".npy"):
                self.sc = np.load(self.input_sc)
        else:
            self.sc = self.input_sc
        if np.any(np.diag(self.sc)):
            print("Warning: The diagonal of the SC matrix is not 0. "
                "Self-connections are not ignored by default in "
                "the simulations. If you want to ignore them set "
                "the diagonal of the SC matrix to 0.")
        self.duration = duration
        self.TR = TR
        self._dt = Decimal(dt)
        self._bw_dt = Decimal(bw_dt)
        self._check_dt()
        self.ext_out = ext_out
        self.states_sampling = states_sampling
        self.states_ts = self.ext_out & states_ts
        self.gof_terms = gof_terms
        self.do_fc = do_fc
        self.do_fcd = do_fcd
        self.window_size = window_size
        self.window_step = window_step
        self.sim_seed = sim_seed
        self.exc_interhemispheric = exc_interhemispheric
        self.force_cpu = force_cpu
        self.force_gpu = force_gpu
        self.bold_remove_s = bold_remove_s
        self.fcd_drop_edges = fcd_drop_edges
        self.noise_segment_length = noise_segment_length
        self.noise_out = noise_out
        self.sim_verbose = sim_verbose
        self.progress_interval = progress_interval
        self.nodes = self.sc.shape[0]
        # load sc distance if provided, and set delay flag accordingly
        self.input_sc_dist = sc_dist
        if self.input_sc_dist is None:
            self.sc_dist = np.zeros_like(self.sc, dtype=float)
            self.do_delay = False
        else:
            if isinstance(self.input_sc_dist, (str, os.PathLike)):
                if self.input_sc_dist.endswith(".txt"):
                    self.sc_dist = np.loadtxt(self.input_sc_dist)
                elif self.input_sc_dist.endswith(".npy"):
                    self.sc_dist = np.load(self.input_sc_dist)
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
            self.out_dir = self.input_sc.replace(".txt", "").replace(".npy", "")
        else:
            self.out_dir = self.input_out_dir
        # get number of available hardware
        self.avail_gpus = utils.avail_gpus()
        self.avail_cpus = multiprocessing.cpu_count()
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
        if hasattr(self, "_do_delay") and (not self.do_delay):
            self.param_lists["v"] = np.zeros(self._N, dtype=float)

    @property
    def nodes(self):
        return self._nodes
    
    @nodes.setter
    def nodes(self, nodes):
        self._nodes = nodes
        max_nodes = max_nodes_many if many_nodes_flag else max_nodes_reg
        # determine if with this number of nodes co-launch mode will be enabled
        # in co-launch mode multiple blocks are occupied by a single (massive) simulation
        self._co_launch = (self.nodes > max_nodes) and (not self.use_cpu)
        if self._co_launch:
            if not many_nodes_flag:
                raise NotImplementedError(
                    f"With {self.nodes} nodes in the current installation of the toolbox"
                    " simulations will fail. Please reinstall the package from source after"
                    " `export CUBNM_MANY_NODES=1`")
            if self.do_fcd:
                raise NotImplementedError(
                    "With many nodes, FCD calculation is not supported."
                    " Set do_fcd to False."
                )
            if self.dt < 1.0:
                print(
                    "Warning: When runnig simulations with large number"
                    " of nodes it is recommended to set the model dt to"
                    " >=1.0 msec for better performance."
                )

    @property
    def duration(self):
        return self._duration
    
    @duration.setter
    def duration(self, duration):
        self._duration = duration
        self.duration_msec = int(duration * 1000)
        if hasattr(self, "_noise_segment_length") and (self.noise_segment_length is None):
            self.noise_segment_length = self.duration

    @property
    def TR(self):
        return self._TR

    @TR.setter
    def TR(self, TR):
        self._TR = TR
        self.TR_msec = int(TR * 1000)
        if hasattr(self, "_states_sampling") and ((self.states_sampling is None) | (self.states_sampling == self.TR)):
            self.states_sampling = self.TR
        if hasattr(self, "_window_size"):
            self.window_size_TRs = int(np.round(self.window_size / (self.TR*2))) * 2
        if hasattr(self, "_window_step"):
            self.window_step_TRs = int(np.round(self.window_step / self.TR))

    @property
    def states_sampling(self):
        return self._states_sampling

    @states_sampling.setter
    def states_sampling(self, states_sampling):
        # raise an error if sampling rate is below model's outer loop dt
        if (states_sampling is not None) and (states_sampling < 0.001):
            raise ValueError(
                "states_sampling cannot be lower than 0.001s "
                "which is model dt"
            )
        if hasattr(self, "_TR") and (states_sampling is None):
            # also set it to TR when None
            self._states_sampling = self.TR
        else:
            self._states_sampling = states_sampling
        # convert states_sampling to msec
        self.states_sampling_msec = int(self.states_sampling * 1000)

    @property
    def n_states_samples_remove(self):
        return int(self.bold_remove_s / self.states_sampling)

    @property
    def noise_segment_length(self):
        return self._noise_segment_length

    @noise_segment_length.setter
    def noise_segment_length(self, noise_segment_length):
        if hasattr(self, "_duration") and (noise_segment_length is None):
            self._noise_segment_length = self.duration
        else:
            self._noise_segment_length = noise_segment_length

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
        if hasattr(self, "_dt") and self.do_delay and (self.dt < 1.0):
            print(
                "Warning: When using delay in the simulations"
                " it is recommended to set the model dt to >="
                " 1.0 msec for better performance."
            )

    @property
    def window_size(self):
        return self._window_size

    @window_size.setter
    def window_size(self, window_size):
        self._window_size = window_size
        # calculate nearest even number of TRs
        # TODO: enable odd window sizes
        self.window_size_TRs = int(np.round(window_size / (self.TR*2))) * 2

    @property
    def window_step(self):
        return self._window_step
    
    @window_step.setter
    def window_step(self, window_step):
        self._window_step = window_step
        self.window_step_TRs = int(np.round(window_step / self.TR))

    @property
    def do_fc(self):
        return self._do_fc

    @do_fc.setter
    def do_fc(self, do_fc):
        if hasattr(self, "gof_terms") and (not do_fc):
            if ('+fc_corr' in self.gof_terms) or ('-fc_diff' in self.gof_terms) or ('-fc_normec' in self.gof_terms):
                raise ValueError("Cannot calculate FC goodness-of-fit terms without FC."
                                 " Set do_fc to True or remove FC-related goodness-of-fit"
                                 " terms.")
        if hasattr(self, "_do_fcd") and (not do_fc):
            raise ValueError("Cannot calculate FCD without FC. Set do_fcd to False.")
        self._do_fc = do_fc

    @property
    def do_fcd(self):
        return self._do_fcd

    @do_fcd.setter
    def do_fcd(self, do_fcd):
        if hasattr(self, "_do_fc") and self.do_fc:
            self._do_fcd = do_fcd
        elif do_fcd:
            raise ValueError("Cannot calculate FCD without FC")
        else:
            self._do_fcd = False
        if hasattr(self, "gof_terms") and (not self._do_fcd) and ('-fcd_ks' in self.gof_terms):
            raise ValueError("Cannot calculate FCD goodness-of-fit terms without FCD."
                                " Set do_fcd to True or remove FCD-related goodness-of-fit"
                                " terms.")

    @property
    def labels(self):
        """
        Labels of parameters and state variables
        to use in plots and reports
        """
        return {}

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
        include_N: :obj:`bool`
            include N in the output config
            is ignored when for_reinit is True
        for_reinit: :obj:`bool`
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
            "nodes": self.nodes,
            "ext_out": self.ext_out,
            "states_ts": self.states_ts,
            "states_sampling": self.states_sampling,
            "do_fc": self.do_fc,
            "do_fcd": self.do_fcd,
            "window_size": self.window_size,
            "window_step": self.window_step,
            "window_size_TRs": self.window_size_TRs,
            "window_step_TRs": self.window_step_TRs,
            "sim_seed": self.sim_seed,
            "exc_interhemispheric": self.exc_interhemispheric,
            "force_cpu": self.force_cpu,
            "force_gpu": self.force_gpu,
            "bold_remove_s": self.bold_remove_s,
            "fcd_drop_edges": self.fcd_drop_edges,
            "noise_segment_length": self.noise_segment_length,
        }
        if not for_reinit:
            # these are features that if changed
            # will not require reinitialization in core
            # (reallocation of memory and/or recalculation
            # of noise)
            config["out_dir"] = self.input_out_dir
            config["gof_terms"] = self.gof_terms
            config["sc"] = self.input_sc
            config["sc_dist"] = self.input_sc_dist
            config["bw_params"] = self.bw_params
            config["version"] = __version__
            if include_N:
                config["N"] = self.N
        return config
    
    @property
    def _model_config(self):
        """
        Internal model configuration used in the simulation

        Returns
        -------
        model_config: :obj:`dict`
            Dictionary of internal model configurations
        """
        model_config = {
            'do_fc': str(int(self.do_fc)),
            'do_fcd': str(int(self.do_fcd)),
            'bold_remove_s': str(self.bold_remove_s),
            'exc_interhemispheric': str(int(self.exc_interhemispheric)),
            'window_size': str(self.window_size_TRs),
            'window_step': str(self.window_step_TRs),
            'drop_edges': str(int(self.fcd_drop_edges)),
            'noise_time_steps': str(int(self.noise_segment_length*1000)), 
            'verbose': str(int(self.sim_verbose)),
            'progress_interval': str(int(self.progress_interval)),
        }
        return model_config

    @property
    def use_cpu(self):
        """
        Check if CPU should be used for the simulations
        based on the available hardware, compilation options, 
        and user settings
        """
        return (
            (self.force_cpu | 
            (not gpu_enabled_flag) | 
            (self.avail_gpus == 0)) & 
            (not self.force_gpu)
        )

    def run(self, force_reinit=False):
        """
        Run the simulations in parallel (as possible) on GPU/CPU
        through the :func:`cubnm._core.run_simulations` function which runs
        compiled C++/CUDA code.

        Parameters
        ----------
        force_reinit: :obj:`bool`
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
        # run the simulations
        args = (
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
            self.sim_seed,
            float(self.dt),
            float(self.bw_dt),
        )
        try:
            out = run_simulations(*args)
        except RuntimeError as e:
            if "CUDA Error" in str(e):
                print(
                    "CUDA error occurred. This might be due to the data exceeding "
                    "the available GPU memory. Try reducing the number of simulations or "
                    "sampling rate of states,",
                    end=" "
                )
                if self.states_ts:
                    print("or set `states_ts` to False", end=" ")
                print("to reduce memory usage")
            raise e
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
            output of ``cubnm._core.run_simulations`` function

        Notes
        -----
        The simulation outputs are assigned to the following object attributes:

        - init_time: :obj:`float`
            initialization time of the simulations
        - run_time: :obj:`float`
            run time of the simulations
        - sim_bold: :obj:`np.ndarray`
            simulated BOLD time series. Shape: (N_SIMS, duration/TR, nodes)

        If ``do_fc`` is ``True``, additionally includes:

        - sim_fc_trils: :obj:`np.ndarray`
            simulated FC lower triangle. Shape: (N_SIMS, n_pairs)

        If ``do_fcd`` is ``True``, additionally includes:

        - sim_fcd_trils: :obj:`np.ndarray`
            simulated FCD lower triangle. Shape: (N_SIMS, n_window_pairs)

        If ``ext_out`` is True, additionally includes:

        - sim_states: :obj:`dict` of :obj:`np.ndarray`
            simulated state variables with keys as state names
            and values as arrays with the shape (N_SIMS, nodes)
            when ``states_ts`` is False, and (N_SIMS, duration/TR, nodes)
            when ``states_ts`` is True
        """
        # assign the output to object attributes
        self.__dict__.update(out)
        # reshape the simulated BOLD, FC and FCD to (N, ...)
        self.sim_bold = self.sim_bold.reshape(self.N, -1, self.nodes)
        if self.do_fc:
            self.sim_fc_trils = self.sim_fc_trils.reshape(self.N, -1)
            if self.do_fcd:
                self.sim_fcd_trils = self.sim_fcd_trils.reshape(self.N, -1)
        if self.ext_out:
            # process sim states into a dictionary
            # TODO: ensure each state's array is a view and not a copy of the original
            self.sim_states = {}
            for state_i, state_name in enumerate(self.state_names):
                self.sim_states[state_name] = self._sim_states[state_i, :, :]
                if self.states_ts:
                    self.sim_states[state_name] = self.sim_states[state_name].reshape(self.N, -1, self.nodes)

    def get_state_averages(self):
        """
        Get the averages of state variables across time and nodes
        for each simulation.

        Returns
        -------
        state_averages: :obj:`pd.DataFrame`
            DataFrame of state averages with columns as state names
            and rows as simulations
        """
        state_averages = {}
        for state_var in self.state_names:
            if self.states_ts:
                state_averages[state_var] = (
                    self.sim_states[state_var]
                    [:, self.n_states_samples_remove:, :]
                    .mean(axis=1).mean(axis=2)
                )
            else:
                state_averages[state_var] = self.sim_states[state_var].mean(axis=1)
        return pd.DataFrame(state_averages)

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

    def get_sim_fc(self, idx):
        """
        Get the simulated FC of a given simulation ``idx``
        as a square matrix.

        Parameters
        ----------
        idx: :obj:`int`
            index of the simulation to get the FC for

        Returns
        -------
        fc: :obj:`np.ndarray`
            simulated FC matrix of shape (nodes, nodes)
            for the simulation with index ``idx``
        """
        if not self.do_fc:
            raise ValueError("FC is not calculated in this simulation group")
        if idx >= self.N:
            raise IndexError("Index out of range")
        # get the FC lower triangle
        fc_tril = self.sim_fc_trils[idx].copy()
        # convert it to a square matrix
        sim_fc = np.zeros((self.nodes, self.nodes), dtype=float)
        if self.exc_interhemispheric:
            # when interhemispheric connections are excluded
            # the FC matrix is split into two halves
            # and the lower triangle is filled in each half
            half_nodes = self.nodes // 2
            sim_fc[:half_nodes, :half_nodes][
                np.tril_indices(half_nodes,-1)
            ] = fc_tril[:fc_tril.shape[0] // 2]
            sim_fc[half_nodes:, half_nodes:][
                np.tril_indices(half_nodes,-1)
            ] = fc_tril[fc_tril.shape[0] // 2:]
            # set the rest to NaNs
            sim_fc[:half_nodes, half_nodes:] = np.NaN
            sim_fc[half_nodes:, :half_nodes] = np.NaN
        else:
            # fill the lower triangle of the square matrix
            sim_fc[np.tril_indices(self.nodes,-1)] = fc_tril
        # make the matrix symmetric
        sim_fc += sim_fc.T
        # fill the diagonal with 1s
        np.fill_diagonal(sim_fc, 1.0)
        return sim_fc

    def get_sim_fcd(self, idx):
        """
        Get the simulated FCD of a given simulation ``idx``
        as a square matrix.

        Parameters
        ----------
        idx: :obj:`int`
            index of the simulation to get the FC for
        
        Returns
        -------
        fcd: :obj:`np.ndarray`
            simulated FC matrix of shape (n_windows, n_windows)
            for the simulation with index ``idx``
        """
        if not self.do_fcd:
            raise ValueError("FCD is not calculated in this simulation group")
        if idx >= self.N:
            raise IndexError("Index out of range")
        # get the FCD lower triangle
        fcd_tril = self.sim_fcd_trils[idx].copy()
        n_pairs = fcd_tril.shape[0]
        # determine number of windows
        # this is not returned from the core
        # and its direct calculation without using
        # a loop isn't trivial (because of different
        # conditions etc.; or at least I don't know
        # how to do it).
        # therefore, we use the (known) number of pairs
        # and solve for n_pairs = n_windows * (n_windows - 1) / 2
        # aka:
        n_windows = (1 + np.sqrt(1 + 8 * n_pairs)) / 2
        assert n_windows.is_integer() # just a sanity check
        n_windows = int(n_windows)
        # convert it to a square matrix
        sim_fcd = np.zeros((n_windows, n_windows), dtype=float)
        # fill the lower triangle of the square matrix
        sim_fcd[np.tril_indices(n_windows,-1)] = fcd_tril
        # make the matrix symmetric
        sim_fcd += sim_fcd.T
        # fill the diagonal with 1s
        np.fill_diagonal(sim_fcd, 1.0)
        return sim_fcd

    def slice(self, key, inplace=False):
        """
        Slice the simulation group to a single simulation

        Parameters
        ----------
        key: :obj:`int`
            index of the simulation to slice
        inplace: :obj:`bool`
            the object will be sliced in place
            and therefore the data of other simulations
            will be removed. Otherwise a new object
            copied from the current object will be returned.
        
        Returns
        -------
        obj: :obj:`cubnm.sim.SimGroup`
            sliced simulation group
        """
        if not isinstance(key, int):
            raise ValueError("Only integer indexing is supported")
        if key >= self.N:
            raise IndexError("Index out of range")
        if inplace:
            obj = self
        else:
            # create a (shallow) copy
            # and not a deep copy as the
            # data may be very large
            obj = copy.copy(self)
        obj.param_lists = {k: v[key][np.newaxis, ...] for k, v in self.param_lists.items()}
        obj.sim_bold = self.sim_bold[key][np.newaxis, ...]
        if obj.do_fc:
            obj.sim_fc_trils = self.sim_fc_trils[key][np.newaxis, ...]
            if obj.do_fcd:
                obj.sim_fcd_trils = self.sim_fcd_trils[key][np.newaxis, ...]
        if obj.ext_out:
            obj.sim_states = {k: v[key][np.newaxis, ...] for k, v in self.sim_states.items()}
        obj.N = 1
        return obj

    def clear(self):
        """
        Clear the simulation outputs
        """
        for attr in [
            "sim_bold", 
            "sim_fc_trils", 
            "sim_fcd_trils", 
            "sim_states",
            "_sim_states",
            "_noise",
            "_shuffled_nodes",
            "_shuffled_ts",
        ]:
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
        X: :obj:`np.ndarray`
            the normalized parameters of current population in range [0, 1]. 
            Shape: (N, ndim)
        out: :obj:`dict`
            the output dictionary to store the results with keys 'F' and 'G'.
            Currently only 'F' (cost) is used.
        *args, **kwargs
        """
        pass

    def score(
            self, 
            emp_fc_tril=None, 
            emp_fcd_tril=None, 
            emp_bold=None, 
            force_cpu=False,
            usable_mem=None
        ):
        """
        Calcualates individual goodness-of-fit terms and aggregates them.

        .. note::
            If `emp_bold` is provided, `emp_fc_tril` and `emp_fcd_tril` will be ignored.

        .. note::
            For each measure, if the value is NaN, it will be set to the "worst" possible value.
            NaNs may occur in simulated FCD or FC. For example, in the rWWEx model, when excitation
            is too high and noise is low, `S` and in turn `BOLD` in some areas may become saturated
            and show no variability. This can result in correlations of their BOLD signals with
            other nodes (within certain dynamic windows) being NaN.

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
        force_cpu: :obj:`bool`
            force CPU for the calculations. Otherwise if GPU is available
            some of the scores (including "+fc_corr", "-fcd_ks") 
            will be calculated on GPU.
        usable_mem: :obj:`int`
            amount of available GPU memory to be used in bytes.
            If None, 80% of the free memory will be used.

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
                    self.window_size_TRs, 
                    self.window_step_TRs, 
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
            assert emp_fc_tril.size == self.sim_fc_trils.shape[1], (
                "Empirical FC lower triangle size does not match the simulated FC size"
            )
        if self.do_fcd and (emp_fcd_tril is not None):
            columns += list(set(["-fcd_ks"]) & set(self.gof_terms))
        columns += ["+gof"]
        scores = pd.DataFrame(index=range(self.N), columns=columns, dtype=float)
        # vectorized calculation of some measures on CPU
        for column in columns:
            if column == "-fc_diff":
                scores.loc[:, column] = -np.abs(
                    self.sim_fc_trils.mean(axis=1) - emp_fc_tril.mean()
                )
            elif column == "-fc_normec":
                scores.loc[:, column] = -np.linalg.norm(
                    self.sim_fc_trils - emp_fc_tril[None, :],
                    axis=1,
                ) / (2 * np.sqrt(emp_fc_tril.size))
        # calculation of fc_corr and fcd_ks per simulation on CPU
        # or in parallel batches on GPU (if available and CuPy is installled)
        if (self.use_cpu or force_cpu or (not has_cupy)):
            # TODO: run in parallel on CPU cores when
            # self.avail_cpus > 1
            for idx in range(self.N):
                for column in columns:
                    if column == "+fc_corr":                    
                        # TODO: vectorize FC corr calculation
                        scores.loc[idx, column] = scipy.stats.pearsonr(
                            self.sim_fc_trils[idx], emp_fc_tril
                        ).statistic
                    elif column == "-fcd_ks":
                        scores.loc[idx, column] = -scipy.stats.ks_2samp(
                            self.sim_fcd_trils[idx], emp_fcd_tril
                        ).statistic
        else:
            # calculate on GPU
            if usable_mem is None:
                # get amount of available GPU memory as
                # 80% of the free memory
                free_mem, _ = cp.cuda.runtime.memGetInfo()
                usable_mem = int(free_mem * 0.8)
            for column in columns:
                if column == "+fc_corr":
                    scores.loc[:, column] = utils.fc_corr_device(self.sim_fc_trils, emp_fc_tril, usable_mem)
                elif column == "-fcd_ks":
                    scores.loc[:, column] = -utils.fcd_ks_device(self.sim_fcd_trils, emp_fcd_tril, usable_mem)
        # fill NaNs with the worst possible value
        for column in columns:
            scores.loc[:, column] = scores.loc[:, column].fillna(WORST_SCORES.get(column, np.nan))
        # combined the selected terms into gof (that should be maximized)
        scores.loc[:, "+gof"] = scores.loc[:, self.gof_terms].sum(axis=1)
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
            out_data.update(self.sim_states)
        out_data.update(self.param_lists)
        return out_data

    def save(self):
        """
        Save simulation outputs to disk as an npz file.
        """
        assert self.out_dir is not None, (
            "Cannot save the simulations when `.out_dir`"
            "is not set"
        )
        os.makedirs(self.out_dir, exist_ok=True)
        out_data = self._get_save_data()
        np.savez_compressed(os.path.join(self.out_dir, "sim_data.npz"), **out_data)

    @classmethod
    def _get_test_configs(cls, cpu_gpu_identity=False):
        """
        Get configs for testing the simulations

        Parameters
        ----------
        cpu_gpu_identity: :obj:`bool`
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
            window_size=10,
            window_step=2,
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
    state_names = ["I_E", "I_I", "r_E", "r_I", "S_E", "S_I"]
    sel_state_var = "r_E" # TODO: use all states
    n_noise = 2
    def __init__(self, 
                 *args, 
                 do_fic = True,
                 max_fic_trials = 0,
                 fic_penalty_scale = 0.5,
                 fic_i_sampling_start = 1000,
                 fic_i_sampling_end = 10000,
                 fic_init_delta = 0.02,
                 **kwargs):
        """
        Group of reduced Wong-Wang simulations (Deco 2014) 
        that are executed in parallel

        Parameters
        ---------
        do_fic: :obj:`bool`
            do analytical (Demirtas 2019) & numerical (Deco 2014) 
            Feedback Inhibition Control.
            If provided wIE parameters will be ignored
        max_fic_trials: :obj:`int`
            maximum number of trials for FIC numerical adjustment.
            If set to 0, FIC will be done only analytically
        fic_penalty_scale: :obj:`bool`
            how much deviation from FIC target mean rE of 3 Hz
            is penalized. Set to 0 to disable FIC penalty.
        fic_i_sampling_start: :obj:`int`
            starting time of numerical FIC I_E sampling (msec)
        fic_i_sampling_end: :obj:`int`
            end time of numerical FIC I_E sampling (msec)
        fic_init_delta: :obj:`float`
            initial delta for numerical FIC adjustment.
        *args, **kwargs:
            see :class:`cubnm.sim.SimGroup` for details

        Attributes
        ----------
        param_lists: :obj:`dict` of :obj:`np.ndarray`
            dictionary of parameter lists, including
                - ``'G'``: global coupling. Shape: (N_SIMS,)
                - ``'wEE'``: local excitatory self-connection strength. Shape: (N_SIMS, nodes)
                - ``'wEI'``: local inhibitory self-connection strength. Shape: (N_SIMS, nodes)
                - ``'wIE'``: local excitatory to inhibitory connection strength. Shape: (N_SIMS, nodes)
                - ``'v'``: conduction velocity. Shape: (N_SIMS,)

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
        self.fic_penalty_scale = fic_penalty_scale
        self.fic_i_sampling_start = fic_i_sampling_start
        self.fic_i_sampling_end = fic_i_sampling_end
        self.fic_init_delta = fic_init_delta
        # parent init must be called here because
        # it sets .last_config which may include some
        # model-specific attributes (e.g. self.do_fic)
        super().__init__(*args, **kwargs)
        self.ext_out = self.ext_out | self.do_fic

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
            fic_penalty_scale=self.fic_penalty_scale,
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

    @property
    def labels(self):
        """
        Labels of parameters and state variables
        to use in plots and reports
        """
        labels = super().labels
        labels.update({
            'G': 'G',
            'wEE': r'$w^{EE}$',
            'wEI': r'$w^{EI}$',
            'wIE': r'$w^{IE}$',
            'I_E': r'$I^E$',
            'I_I': r'$I^I$',
            'r_E': r'$r^E$',
            'r_I': r'$r^I$',
            'S_E': r'$S^E$',
            'S_I': r'$S^I$',
        })
        return labels
    
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
        if self.do_fic:
            # FIC stats
            self.fic_unstable = self._global_bools[0]
            self.fic_failed = self._global_bools[1]
            self.fic_ntrials = self._global_ints[0]
            # write FIC (+ numerical adjusted) wIE to param_lists
            self.param_lists["wIE"] = self._regional_params[2].reshape(self.N, -1)
    
    def score(self, emp_fc_tril=None, emp_fcd_tril=None, emp_bold=None, **kwargs):
        """
        Calcualates individual goodness-of-fit terms and aggregates them.
        In FIC models also calculates fic_penalty. Note that while 
        FIC penalty is included in the cost function of optimizer, 
        it is not included in the aggregate GOF.

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
        **kwargs:
            keyword arguments passed to `cubnm.sim.SimGroup.score`

        Returns
        -------
        scores: :obj:`pd.DataFrame`
            The goodness of fit measures (columns) of each simulation (rows)
        """
        scores = super().score(emp_fc_tril, emp_fcd_tril, emp_bold, **kwargs)
        # calculate FIC penalty
        if self.do_fic & (self.fic_penalty_scale > 0):
            # calculate time-averaged r_E in each simulation-node
            if self.states_ts:
                mean_r_E = self.sim_states["r_E"][:, self.n_states_samples_remove:].mean(axis=1)
            else:
                mean_r_E = self.sim_states["r_E"]
            for idx in range(self.N):
                # absolute deviation from target of 3 Hz
                diff_r_E = np.abs(mean_r_E[idx, :] - 3)
                if (diff_r_E > 1).sum() > 0:
                    # at least one node has a deviation greater than 1
                    # only consider deviations greater than 1
                    # in the penalty
                    diff_r_E[diff_r_E <= 1] = np.NaN
                    # within each node the FIC penalty decreases
                    # exponentially as the deviations gets smaller
                    # the penalty max value in each node is 1
                    # which is then scaled by fic_penalty_scale
                    # and averaged across nodes (with 0s/NaNs)
                    # in node with no deviation > 1 Hz
                    scores.loc[idx, "-fic_penalty"] = (
                        -np.nansum(1 - np.exp(-0.05 * (diff_r_E - 1)))
                        * self.fic_penalty_scale / self.nodes
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
            if self.do_fic & (self.fic_penalty_scale > 0):
                problem.obj_names.append("+fic_penalty")
                problem.n_obj += 1

    def _problem_evaluate(self, problem, X, out, *args, **kwargs):
        """
        Extends BNMProblem evaluation and includes FIC penalty
        in the cost function if indicated.

        Parameters
        ----------
        X: :obj:`np.ndarray`
            the normalized parameters of current population in range [0, 1]. 
            Shape: (N, ndim)
        out: :obj:`dict`
            the output dictionary to store the results with keys ``'F'`` and ``'G'``.
            Currently only ``'F'`` (cost) is used.
        *args, **kwargs
        """
        # Note: scores (inidividual GOF measures) is passed on in kwargs 
        # in the internal mechanism of pymoo evaluation function
        scores = kwargs["scores"][-1]
        if self.do_fic & (self.fic_penalty_scale > 0):
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
        cpu_gpu_identity: :obj:`bool`
            indicates whether configs are for CPU/GPU identity tests
            in which case force_cpu is not included in the configs
            since tests will be done on both CPU and GPU

        Returns
        -------
        configs: :obj:`dict` of :obj:`list`
        """
        configs = super()._get_test_configs(cpu_gpu_identity)
        configs.update({
            # 0: no FIC
            # 1: analytical FIC
            # 2: analytical + numerical FIC
            'do_fic': [0, 1, 2]
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
        sim_group.do_fic = opts['do_fic'] > 0
        if opts['do_fic'] == 2:
            sim_group.max_fic_trials = 5
        return sim_group

class rWWExSimGroup(SimGroup):
    model_name = "rWWEx"
    global_param_names = ["G"]
    regional_param_names = ["w", "I0", "sigma"]
    state_names = ["x", "r", "S"]
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
                - ``'G'``: global coupling. Shape: (N_SIMS,)
                - ``'w'``: local excitatory self-connection strength. Shape: (N_SIMS, nodes)
                - ``'I0'``: local external input current. Shape: (N_SIMS, nodes)
                - ``'sigma'``: local noise sigma. Shape: (N_SIMS, nodes)
                - ``'v'``: conduction velocity. Shape: (N_SIMS,)

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

class KuramotoSimGroup(SimGroup):
    model_name = "Kuramoto"
    global_param_names = ["G"]
    regional_param_names = ["init_theta", "omega", "sigma"]
    state_names = ["theta"]
    sel_state_var = "theta"
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
        random_init_theta: :obj:`bool`
            Set initial theta by randomly sampling from a uniform distribution 
            [0, 2*pi].

        Attributes
        ----------
        param_lists: :obj:`dict` of :obj:`np.ndarray`
            dictionary of parameter lists, including
                - ``'G'``: global coupling. Shape: (N_SIMS,)
                - ``'init_theta'``: initial theta. Randomly sampled from a uniform distribution 
                  [0, 2*pi] by default. Shape: (N_SIMS, nodes)
                - ``'omega'``: intrinsic frequency. Shape: (N_SIMS, nodes)
                - ``'sigma'``: local noise sigma. Shape: (N_SIMS, nodes)
                - ``'v'``: conduction velocity. Shape: (N_SIMS,)

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
            rng = np.random.default_rng(self.sim_seed)
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
        names and types. See :func:`cubnm.sim.SimGroup._process_out`
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
            out_child = {}
            for k in [
                "init_time",
                "run_time",
                "sim_bold", 
                "sim_fc_trils", 
                "sim_fcd_trils", 
                "_sim_states",
                "_global_bools",
                "_global_ints",
                "_noise",
                "_shuffled_nodes",
                "_shuffled_ts",
            ]:
                if k not in out:
                    continue
                if k in ["init_time", "run_time", "_noise", "_shuffled_nodes", "_shuffled_ts"]:
                    # shared between all simulations
                    out_child[k] = out[k]
                elif k in ["sim_bold", "sim_fc_trils", "sim_fcd_trils"]:
                    # simulation-specific outputs with shape (N, ...)
                    out_child[k] = out[k][start_idx:end_idx]
                else:
                    # simulation-specific outputs with shape (..., N, ...)
                    # TODO: it'll be easier to do this if all
                    # simulation-specific outputs have N_SIMS as
                    # the first axis
                    out_child[k] = out[k][:, start_idx:end_idx]
            # update start_idx for the next child
            start_idx += child.N
            # process the output of the child
            child._process_out(out_child)

def create_multi_sim_group(sim_group_cls):
    """
    Dynamically creates a MultiSimGroup class by combining
    a model's specific ``<Model>SimGroup`` class with 
    :class:`cubnm.sim.MultiSimGroupMixin`,
    which can be used in batch optimization.

    Parameters
    ----------
    sim_group_cls: :obj:`type`
        :class:`cubnm.sim.SimGroup` subclass, e.g. :class:`cubnm.sim.rWWSimGroup`
    
    Returns
    -------
    MultiSimGroup: :obj:`type`
        :class:`MultiSimGroup` class that can be used in batch optimization
    """
    MultiSimGroup = type(
        'MultiSimGroup',
        (MultiSimGroupMixin, sim_group_cls),
        {}
    )
    return MultiSimGroup