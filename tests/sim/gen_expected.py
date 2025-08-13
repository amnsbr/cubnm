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
from cubnm import sim, utils

test_data_dir = os.path.join(os.path.dirname(__file__), '..', 'expected', 'sim')

def gen_expected(model):
    if os.path.exists(os.path.join(test_data_dir, f'{model}.pkl.gz')):
        prev_test_data = pickle.load(gzip.open(os.path.join(test_data_dir, f'{model}.pkl.gz'), 'rb'))
    else:
        prev_test_data = {}
    ModelSimGroup = getattr(sim, f'{model}SimGroup')
    test_configs = ModelSimGroup._get_test_configs(cpu_gpu_identity=False)
    test_data = {}
    # loop through combinat
    for opts in itertools.product(*[[(k, v) for v in vs] for k, vs in test_configs.items()]):
        opts_str = ','.join([f'{k}:{int(v)}' for k, v in opts])
        opts_dict = {k: v for k, v in opts}
        print(opts_str)
        print(opts_dict)
        if (not opts_dict['force_cpu']) & (utils.avail_gpus() == 0):
            print("Warning: no GPU available, skipping GPU")
            continue
        sg = ModelSimGroup._get_test_instance(opts_dict)
        sg.N = 1
        sg._set_default_params()
        sg.run()
        test_data[opts_str] = {
            'sim_bold': sg.sim_bold,
            'sim_fc_trils': sg.sim_fc_trils,
            'sim_fcd_trils': sg.sim_fcd_trils,
            'sim_sel_state': sg.sim_states[ModelSimGroup.sel_state_var], # TODO: use all state variables
        }
        # print a warning if the output has changed from the previous version
        if opts_str in prev_test_data:
            for k in ['sim_bold', 'sim_fc_trils', 'sim_fcd_trils', 'sim_sel_state']:
                if k in prev_test_data.get(opts_str, {}):
                    if test_data[opts_str][k].shape != prev_test_data[opts_str][k].shape:
                        print(f"Warning: {opts_str} {k} shape has changed from the previous version")
                    elif not np.isclose(test_data[opts_str][k], prev_test_data[opts_str][k], atol=1e-12).all():
                        print(f"Warning: {opts_str} {k} has changed from the previous version")
    # save all
    with gzip.open(os.path.join(test_data_dir, f'{model}.pkl.gz'), 'wb') as f:
        pickle.dump(test_data, f)

if __name__ == "__main__":
    model = sys.argv[1]
    os.makedirs(test_data_dir, exist_ok=True)
    gen_expected(model)