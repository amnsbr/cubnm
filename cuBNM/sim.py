import numpy as np
import scipy.stats
import pandas as pd
import os

from cuBNM.core import run_simulations


class SimGroup:
    """
    Group of simulations that will be executed in parallel
    """
    def __init__(self, N, duration, TR, sc_path, sc_dist_path=None, maps_path=None, out_dir='same',
            do_fic=True, extended_output=True, window_size=10, window_step=2, rand_seed=410,
            exc_interhemispheric=True):
        self.N = N
        self.time_steps = duration * 1000 # in msec
        self.TR = TR * 1000 # in msec
        self.sc_path = sc_path
        self.sc_dist_path = sc_dist_path
        self.maps_path = maps_path
        self.sc = np.loadtxt(self.sc_path)
        self.do_fic = do_fic
        self.extended_output = extended_output
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
        # determine heterogeneity and load the maps
        if self.maps_path:
            self.maps = np.loadtxt(self.maps_path)
            self.is_heterogeneous = True
        else:
            self.maps = None
            self.is_heterogeneous = False
        # initialze w_IE_list as all 0s if do_fic
        self.param_lists = dict([(k, None) for k in ['G', 'wEE', 'wEI', 'wIE', 'v']])
        if self.do_fic:
            self.param_lists['wIE'] = np.zeros((self.N,self.nodes), dtype=float)
        if not self.do_delay:
            self.param_lists['v'] = np.zeros(self.N, dtype=float)
        # determine and create output directory
        if out_dir == 'same':
            self.out_dir = self.sc_path.replace('.txt', '')
        else:
            self.out_dir = out_dir
        os.makedirs(out_dir, exist_ok=True)

    def run(self):
        # TODO: add assertions to make sure all the data is complete
        out = run_simulations(
            self.sc.flatten(), 
            self.sc_dist.flatten(), 
            self.param_lists['G'], 
            self.param_lists['wEE'].flatten(), 
            self.param_lists['wEI'].flatten(), 
            self.param_lists['wIE'].flatten(), 
            self.param_lists['v'],
            self.do_fic, self.extended_output, self.do_delay, self.N, 
            self.nodes, self.time_steps, self.TR,
            self.window_size, self.window_step, self.rand_seed
        )
        ext_out = {}
        if self.extended_output:
            sim_bold, sim_fc_trils, sim_fcd_trils, \
            ext_out['S_E'], ext_out['S_I'], ext_out['S_ratio'],\
            ext_out['r_E'], ext_out['r_I'], ext_out['r_ratio'],\
            ext_out['I_E'], ext_out['I_I'], ext_out['I_ratio'],\
              = out
            self.ext_out = ext_out
            for k in self.ext_out:
                self.ext_out[k] = self.ext_out[k].reshape(self.N, -1)
        else:
            sim_bold, sim_fc_trils, sim_fcd_trils = out
        self.sim_bold = sim_bold.reshape(self.N, -1, self.nodes)
        self.sim_fc_trils = sim_fc_trils.reshape(self.N, -1)
        self.sim_fcd_trils = sim_fcd_trils.reshape(self.N, -1)

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
        columns = ['fc_corr', 'fc_diff', 'fcd_ks', 'gof']
        if self.do_fic:
            columns.append('fic_penalty')
        scores = pd.DataFrame(columns=columns, dtype=float)
        for idx in range(self.N):
            scores.loc[idx, 'fc_corr'] = scipy.stats.pearsonr(self.sim_fc_trils[idx], emp_fc_tril).statistic
            scores.loc[idx, 'fc_diff'] = np.abs(self.sim_fc_trils[idx].mean() - emp_fc_tril.mean())
            scores.loc[idx, 'fcd_ks'] = scipy.stats.ks_2samp(self.sim_fcd_trils[idx], emp_fcd_tril).statistic
        scores.loc[:, 'gof'] = scores.loc[:, 'fc_corr'] - 1 - scores.loc[:, 'fc_diff'] - scores.loc[:, 'fcd_ks']
        if self.do_fic:
            for idx in range(self.N):
                diff_r_E = np.abs(self.ext_out['r_E'][idx,:] - 3)
                if (diff_r_E > 1).sum() > 1:
                    diff_r_E[diff_r_E <= 1] = np.NaN
                    scores.loc[idx, 'fic_penalty'] = np.nanmean(1 - np.exp(-0.05 * diff_r_E)) * fic_penalty_scale
                else:
                    scores.loc[idx, 'fic_penalty'] = 0
        return scores
