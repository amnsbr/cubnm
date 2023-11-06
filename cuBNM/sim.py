import numpy as np
import scipy.stats
import pandas as pd
import os

from cuBNM.core import run_simulations


class SimGroup:
    def __init__(self, duration, TR, sc_path, sc_dist_path=None, out_dir='same',
            do_fic=True, extended_output=True, window_size=10, window_step=2, rand_seed=410,
            exc_interhemispheric=True):
        """
        Group of simulations that will be executed in parallel

        Parameters
        ---------
        duration: (float)
            simulation duration (in seconds)
        TR: (float)
            BOLD TR and sampling rate of extended output (in seconds)
        sc_path: (str)
            path to structural connectome strengths (as an unlabled .txt)
        sc_dist_path: (str)
            path to structural connectome distances
            if provided v (velocity) will be a free parameter and there
            will be delay in inter-regional connections
        out_dir: (str)
            if 'same' will create a directory named based on sc_path
        do_fic: (bool)
            do analytical-numerical Feedback Inhibition Control
            if provided wIE parameters will be ignored
        extended_output: (bool)
            return mean internal model variables to self.ext_out
        window_size: (int)
            dynamic FC window size (in TR)
        window_step: (int)
            dynamic FC window step (in TR)
        rand_seed: (int)
            seed used for the noise simulation
        exc_interhemispheric: (bool)
            excluded interhemispheric connections from sim FC and FCD calculations
        """
        self.time_steps = int(duration * 1000) # in msec
        self.TR = int(TR * 1000) # in msec
        self.sc_path = sc_path
        self.sc_dist_path = sc_dist_path
        self.sc = np.loadtxt(self.sc_path)
        self.do_fic = do_fic
        self.extended_output = (extended_output | do_fic)  # extended output is needed for FIC penalty calculations
        self.window_size = window_size
        self.window_step = window_step
        self.rand_seed = rand_seed
        self.exc_interhemispheric = exc_interhemispheric
        os.environ['BNM_EXC_INTERHEMISPHERIC'] = str(int(self.exc_interhemispheric))
        # determine number of nodes based on sc dimensions
        self.nodes = self.sc.shape[0]
        # inter-regional delay will be added to the simulations 
        # if SC distance matrix is provided
        if self.sc_dist_path:
            self.sc_dist = np.loadtxt(self.sc_dist_path)
            self.do_delay = True
        else:
            self.sc_dist = np.zeros_like(self.sc, dtype=float)
            self.do_delay = False
        os.environ['BNM_SYNC_MSEC'] = str(int(self.do_delay))
        # TODO: get other model configs (see constants.hpp) from user and
        # change BNM_ env variables accordingly
        # initialze w_IE_list as all 0s if do_fic
        self.param_lists = dict([(k, None) for k in ['G', 'wEE', 'wEI', 'wIE', 'v']])
        # determine and create output directory
        if out_dir == 'same':
            self.out_dir = self.sc_path.replace('.txt', '')
        else:
            self.out_dir = out_dir
        os.makedirs(out_dir, exist_ok=True)
        # keep track of the last N run to determine if force_reinit
        # is needed (when last_N is different from current N)
        self.last_N = 0
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
            self.param_lists['wIE'] = np.zeros((self._N,self.nodes), dtype=float)
        if not self.do_delay:
            self.param_lists['v'] = np.zeros(self._N, dtype=float)


    def run(self):
        """
        Run the simulations in parallel on GPU
        """
        # TODO: add assertions to make sure all the data is complete
        force_reinit = (self.N != self.last_N)
        # set wIE to its flattened copy and pass this
        # array to run_simulations so that in the case
        # of do_fic it is overwritten with the actual wIE
        # used in the simulations
        self.param_lists['wIE'] = self.param_lists['wIE'].flatten()
        out = run_simulations(
            self.sc.flatten(), 
            self.sc_dist.flatten(), 
            self.param_lists['G'], 
            self.param_lists['wEE'].flatten(), 
            self.param_lists['wEI'].flatten(), 
            self.param_lists['wIE'], 
            self.param_lists['v'],
            self.do_fic, self.extended_output, self.do_delay, force_reinit, 
            self.N, self.nodes, self.time_steps, self.TR,
            self.window_size, self.window_step, self.rand_seed
        )
        # avoid reinitializing GPU in the next runs
        # of the same group
        self.last_N = self.N
        self.it += 1
        # assign the output to object properties
        # and reshape them to (N_SIMS, ...)
        ext_out = {}
        if self.extended_output:
            sim_bold, sim_fc_trils, sim_fcd_trils, \
            ext_out['S_E'], ext_out['S_I'], ext_out['S_ratio'],\
            ext_out['r_E'], ext_out['r_I'], ext_out['r_ratio'],\
            ext_out['I_E'], ext_out['I_I'], ext_out['I_ratio'],\
            self.fic_unstable\
              = out
            self.ext_out = ext_out
            for k in self.ext_out:
                self.ext_out[k] = self.ext_out[k].reshape(self.N, -1)
        else:
            sim_bold, sim_fc_trils, sim_fcd_trils, self.fic_unstable = out
        self.sim_bold = sim_bold.reshape(self.N, -1, self.nodes)
        self.sim_fc_trils = sim_fc_trils.reshape(self.N, -1)
        self.sim_fcd_trils = sim_fcd_trils.reshape(self.N, -1)
        self.param_lists['wIE'] = self.param_lists['wIE'].reshape(self.N, -1)

    def score(self, emp_fc_tril, emp_fcd_tril, fic_penalty_scale=2):
        """
        Calcualates fc_corr, fc_diff, fcd_ks and their aggregate (gof).
        In FIC models also calculates fic_penalty. To ignore fic_penalty
        set `fic_penalty_scale` to 0.

        Parameters
        --------
        emp_fc_tril: (np.array)
            1D array of empirical FC lower triangle
        emp_fcd_tril: (np.array)
            1D array of empirical FCD lower triangle
        fic_penalty_scale: (float)
        """
        #TODO: add the option to provide empirical BOLD as input
        columns = ['fc_corr', 'fc_diff', 'fcd_ks', 'gof']
        if self.do_fic:
            columns.append('fic_penalty')
        scores = pd.DataFrame(columns=columns, dtype=float)
        for idx in range(self.N):
            if self.do_fic & self.fic_unstable[idx]:
                # in unfeasible simulations set GOF to max
                # TODO: instead, reject these particles and get
                # substitutes
                scores.loc[idx, 'fc_corr'] = -1
                scores.loc[idx, 'fc_diff'] = 1
                scores.loc[idx, 'fcd_ks'] = 1
            else:
                # otherwise calculate FC corr, FC diff and FCD ks
                scores.loc[idx, 'fc_corr'] = scipy.stats.pearsonr(self.sim_fc_trils[idx], emp_fc_tril).statistic
                scores.loc[idx, 'fc_diff'] = np.abs(self.sim_fc_trils[idx].mean() - emp_fc_tril.mean())
                scores.loc[idx, 'fcd_ks'] = scipy.stats.ks_2samp(self.sim_fcd_trils[idx], emp_fcd_tril).statistic
        # aggeegate GOF
        scores.loc[:, 'gof'] = scores.loc[:, 'fc_corr'] - 1 - scores.loc[:, 'fc_diff'] - scores.loc[:, 'fcd_ks']
        # calculate FIC penalty
        if self.do_fic:
            for idx in range(self.N):
                if self.fic_unstable[idx]:
                    # in unfeasible simulations set GOF to max
                    scores.loc[idx, 'fic_penalty'] = 1
                    continue
                diff_r_E = np.abs(self.ext_out['r_E'][idx,:] - 3)
                if (diff_r_E > 1).sum() > 1:
                    diff_r_E[diff_r_E <= 1] = np.NaN
                    scores.loc[idx, 'fic_penalty'] = np.nanmean(1 - np.exp(-0.05 * diff_r_E)) * fic_penalty_scale
                else:
                    scores.loc[idx, 'fic_penalty'] = 0
        return scores

    def save(self, save_as='npz'):
        """
        Save current simulation outputs to disk

        Parameters
        ---------
        save_as: (str)
            - npz: all the output of all sims will be written to a npz file
            - txt: outputs of simulations will be written to separate files,
                recommended when N = 1 (e.g. rerunning the best simulation) 
        """
        sims_dir = os.path.join(self.out_dir, 'sims')
        os.makedirs(sims_dir, exist_ok=True)
        if save_as == 'npz':
            out_data = dict(
                sim_bold = self.sim_bold,
                sim_fc_trils = self.sim_fc_trils,
                sim_fcd_trils = self.sim_fcd_trils
            )
            out_data.update(self.param_lists)
            if self.extended_output:
                out_data.update(self.ext_out)
            # TODO: use more informative filenames
            np.savez_compressed(
                os.path.join(sims_dir, f'it{self.it}.npz'),
                **out_data
            )
        elif save_as == 'txt':
            raise NotImplementedError

