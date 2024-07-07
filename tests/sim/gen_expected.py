"""
Generates expected simulation outputs used to test stability and reproducibility of
the simulations. This should be run ideally once for each model on a stable version
of the code. The expected outputs will be tracked in the repository. They should only
be updated if the simulation code is considerably changed and it is expected to have
a different output. 
"""
import os
import sys
import gzip
import pickle
import itertools

import numpy as np
from cubnm import sim, datasets, utils

test_data_dir = os.path.join(os.path.dirname(__file__), '..', 'expected', 'sim')

def gen_expected(model):
    if os.path.exists(os.path.join(test_data_dir, f'{model}.pkl.gz')):
        prev_test_data = pickle.load(gzip.open(os.path.join(test_data_dir, f'{model}.pkl.gz'), 'rb'))
    else:
        prev_test_data = {}
    ModelSimGroup = getattr(sim, f'{model}SimGroup')
    model_opts = {
        'force_cpu': [False, True],
        'do_delay': [False, True],
    }
    if model == 'rWW':
        model_opts['do_fic'] = [False, True]
    if model == 'rWW':
        sel_state_var = 'r_E'
    elif model == 'rWWEx':
        sel_state_var = 'r'
    test_data = {}
    # loop through combinat
    for opts in itertools.product(*[[(k, v) for v in vs] for k, vs in model_opts.items()]):
        opts_str = ','.join([f'{k}:{int(v)}' for k, v in opts])
        opts_dict = {k: v for k, v in opts}
        print(opts_str)
        print(opts_dict)
        if (not opts_dict['force_cpu']) & (utils.avail_gpus() == 0):
            print("Warning: no GPU available, skipping GPU")
            continue
        if opts_dict['do_delay']:
            sc_dist_path = datasets.load_sc('length', 'schaefer-100', return_path=True)
        else:
            sc_dist_path = None
        opts_dict.pop('do_delay')
        sg = ModelSimGroup(
            duration=60,
            TR=1,
            sc_path=datasets.load_sc('strength', 'schaefer-100', return_path=True),
            sc_dist_path=sc_dist_path,
            sim_verbose=True,
            **opts_dict
        )
        sg.N = 1
        sg._set_default_params()
        sg.run()
        test_data[opts_str] = {
            'sim_bold': sg.sim_bold,
            'sim_fc_trils': sg.sim_fc_trils,
            'sim_fcd_trils': sg.sim_fcd_trils,
            'sim_sel_state': sg.sim_states[sel_state_var],
        }
        # print a warning if the output has changed from the previous version
        if opts in prev_test_data:
            for k in ['sim_bold', 'sim_fc_trils', 'sim_fcd_trils', 'sim_sel_state']:
                if k in prev_test_data.get(opts_str, {}):
                    if not np.isclose(test_data[opts_str][k], prev_test_data[opts_str][k], atol=1e-12).all():
                        print(f"Warning: {opts_str} output has changed from the previous version")
    # save all
    with gzip.open(os.path.join(test_data_dir, f'{model}.pkl.gz'), 'wb') as f:
        pickle.dump(test_data, f)

if __name__ == "__main__":
    model = sys.argv[1]
    os.makedirs(test_data_dir, exist_ok=True)
    gen_expected(model)