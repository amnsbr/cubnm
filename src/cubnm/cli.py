"""
Command line interface for cuBNM
"""
import argparse
import os
import pandas as pd

from cubnm import sim, optimize, datasets


def parse_params(param_str):
    """
    Parses input string of parameters and their
    ranges into a dictionary recognizable by GridSearch
    and BNMProblem

    Parameters
    ----------
    param_str : :obj:`str`
        a string with the format "<param_name>:<start>[:<end>[:<step>]],..."

    Returns
    -------
    :obj:`dict` with the format {<param_name>:(start[,end[,step]]),...}
    """
    params = {}
    for item in param_str.split(','):
        key, value = item.split('=')
        if ':' in value:
            params[key] = list(map(float, value.split(':')))
            assert len(params[key]) <= 3
            if len(params[key]) == 3:
                params[key][2] = int(params[key][2])
            params[key] = tuple(params[key])
        else:
            params[key] = float(value)
    return params

def parse_maps_coef_range(value):
    """
    Parses input string of map coefficient ranges

    Parameters
    ----------
    value : :obj:`str`
        - "auto"
        - "<start>:<end>": same range for all maps
        - "<start1>:<end1>,<start2>:<end2>,...": different range for each map

    Returns
    -------
    "auto", :obj:`tuple` or :obj:`list` of :obj:`tuple`
    """
    if value == 'auto':
        return value
    try:
        maps_coef_range = [tuple(map(float, item.split(':'))) for item in value.split(',')]
    except:
        raise argparse.ArgumentTypeError("Value should be 'auto' or a list of tuples")
    if len(maps_coef_range) == 1:
        return maps_coef_range[0]
    else:
        return maps_coef_range

def get_optimizer(optimizer_str):
    """
    Gets an Optimizer class based on its name

    Parameters
    ----------
    optimizer_str : :obj:`str`

    Returns
    -------
    :class:`cubnm.optimize.Optimizer` derived class
    """
    return getattr(optimize, f'{optimizer_str}Optimizer')

def main():
    parser = argparse.ArgumentParser(
        description='cuBNM command line interface',
    )

    # simgroup arguments shared between all commands
    parser.add_argument('-m', '--model', type=str, required=True, 
                        choices=['rWW', 'rWWEx', 'Kuramoto'], #TODO: get this from sim module
                        help='Model (required)')
    parser.add_argument('-p', '--params', type=parse_params, required=True, 
                        help='Parameters in custom format inside'
                        ' quotation marks (required). Examples:'
                         ' grid: "G=0.5,wEE=0.05:1:2,wEI=0.07:0.75:2,v=1:5:2"'
                         ' optimizer: "G=0.5,wEE=0.05:1,wEI=0.07:0.075,v=1:5"'
                       )
    parser.add_argument('--duration', type=int, required=True, 
                        help='Duration of the simulation (required)')
    parser.add_argument('--TR', type=float, required=True, 
                        help='Repetition time (required)')
    parser.add_argument('--sc', type=str, required=True, 
                        help='Structural connectivity .txt file or "example" (required)')
    parser.add_argument('--sc_dist', type=str, 
                        help='Structural connectivity distribution .txt file or "example"')
    parser.add_argument('-d', '--out_dir', type=str, default="same", 
                        help='Output directory')
    # TODO: emp FC and FCD do not have to be required for grid search
    # TODO: FCD does not have to be required when fcd_ks is not in GOF terms
    parser.add_argument('--emp_fc_tril', type=str, required=True,
                        help='Functional connectivity lower triangle .txt file or "example"')
    parser.add_argument('--emp_fcd_tril', type=str, required=True,
                        help='Functional connectivity dynamics lower triangle .txt file or "example"')
    parser.add_argument('--rand_seed', type=int, default=410, 
                        help='Simulation noise seed')
    parser.add_argument('--noise_segment_length', type=int, default=30, 
                        help='Noise segment length (seconds)')
    parser.add_argument('--no_ext_out', action='store_true', 
                        help='Do not return model state variables')
    parser.add_argument('--states_ts', action='store_true',
                        help='Return model states timeseries')
    parser.add_argument('--states_sampling', type=float, 
                        help='Sampling rate of model state variables')
    parser.add_argument('--bw_params', type=str, default='friston2003', 
                        choices=['friston2003', 'heinzle2016-3T'],
                        help='Balloon Windkessel parameters')
    parser.add_argument('--bold_remove_s', type=int, default=30, 
                        help='Remove initial n seconds of the simulation from BOLD'
                           ' and average of state variables'
                       )
    parser.add_argument('--window_size', type=int, default=10, 
                        help='FCD window size')
    parser.add_argument('--window_step', type=int, default=2, 
                        help='FCD window step')
    parser.add_argument('--inc_interhemispheric', action='store_true', 
                        help='Include interhemispheric connections in cost')
    parser.add_argument('--fcd_keep_edges', action='store_true', 
                        help='Keep edge windows from FCD calculations')
    parser.add_argument('--gof_terms',
                        type=lambda s: s.split(','),
                        default=["+fc_corr","-fc_diff","-fcd_ks"], 
                        help='Goodness of fit terms (comma separated in quotation marks)'
                            ' e.g. "+fc_corr,-fcd_ks"'
                       )
    parser.add_argument('--force_cpu', action='store_true', 
                        help='Force CPU')
    parser.add_argument('--force_gpu', action='store_true', 
                        help='Force GPU')
    parser.add_argument('--serial_nodes', action='store_true', 
                        help='Only applicable to GPUs; Uses one thread'
                             ' per simulation and do calculation of nodes serially; Experimental'
                       )
    parser.add_argument('-v', '--sim_verbose', action='store_true',
                        help='Show simulation progress')
    parser.add_argument('--progress_interval', type=int, default=500, 
                        help='Interval of progress updates (in msec).'
                            ' Only used if sim_verbose is True'
                       )
    parser.add_argument('--no_print_args', action='store_true', 
                        help='Do not print command line arguments table')

    # add specific commands and their options
    subparsers = parser.add_subparsers(dest='cmd')
    
    # grid command
    parser_grid = subparsers.add_parser('grid', help='Grid search')
    # has no specific options
    
    # optimizer command
    parser_optimize = subparsers.add_parser('optimize', help='Optimize parameters using evolutionary algorithms')
    # Optimizer arguments
    parser_optimize.add_argument('-o', '--optimizer', type=get_optimizer, 
                                  help='Optimizer type')
    parser_optimize.add_argument('--optimizer_seed', type=int, default=0, 
                                  help='Optimizer random seed')
    parser_optimize.add_argument('--n_iter', type=int, default=80, 
                                  help='Optimizer max iterations')
    parser_optimize.add_argument('--popsize', type=int, default=24, 
                                  help='Optimizer population size')
    # BNMProblem arguments
    parser_optimize.add_argument('--het_params', nargs='+', 
                                  help='List of heterogeneous regional parameters (space separated)')
    parser_optimize.add_argument('--maps', type=str, 
                                  help='Path to heterogeneity maps or "example"')
    parser_optimize.add_argument('--maps_coef_range', type=parse_maps_coef_range, 
                                  default='auto',
                                  help='Coefficient range for maps.'
                                  ' Options: "auto", "min:max" (same for all maps),'
                                  ' "min1:max1,min2:max2,..." (different for each map)'
                                 )
    parser_optimize.add_argument('--node_grouping', type=str, 
                                  default=None,
                                  help='Path to node grouping array or special values: node, sym')
    parser_optimize.add_argument('--multiobj', action='store_true', 
                                  help='Instead of combining the objectives into a single objective'
                                  ' function (via summation) defines each objective separately.'
                                  ' This must not be used with single-objective optimizers'
                                 )

    # TODO: add model specific options (e.g. do_fic in rWW)
    # TODO: add optimizer specific options
    
    # TODO: add run command
    # parser_run = subparsers.add_parser('run', help='Run one simulation with given parameters')
    
    # parse the arguments
    args = parser.parse_args()

    if not args.no_print_args:
        # pretty print the command and its arguments
        if args.cmd == 'grid':
            print("Running grid search with the following options:")
        if args.cmd == 'optimize':
            print("Running evolutionary optimization with the following options:")
        args_df = (
            pd.Series(vars(args))
            .fillna('-')
            .rename('Value').to_frame()
            .drop(index='cmd')
            .rename_axis(index='Option')
        )
        print(args_df.to_markdown(tablefmt="simple_outline"))
    del args.no_print_args
    
    # additional refinement of arguments
    args.fcd_drop_edges = not args.fcd_keep_edges
    args.ext_out = not args.no_ext_out
    args.exc_interhemispheric = not args.inc_interhemispheric
    if args.het_params is None:
        args.het_params = []
    del args.fcd_keep_edges, args.no_ext_out, args.inc_interhemispheric

    # use example input data if requested
    if args.sc == 'example':
        args.sc = datasets.load_sc('strength', 'schaefer-100')
    if args.sc_dist == 'example':
        args.sc_dist = datasets.load_sc('length', 'schaefer-100')
    if args.emp_fc_tril == 'example':
        args.emp_fc_tril = datasets.load_functional(
            'FC', 'schaefer-100', 
             exc_interhemispheric=args.exc_interhemispheric
        )
    if args.emp_fcd_tril == 'example':
        args.emp_fcd_tril = datasets.load_functional(
            'FCD', 'schaefer-100', 
             exc_interhemispheric=args.exc_interhemispheric
        )
    if (args.cmd == 'optimize') and (args.maps == 'example'):
        args.maps = datasets.load_maps(
            ['myelinmap', 'fcgradient01'],
            'schaefer-100'
        )

    if args.cmd == 'grid':
        run_grid(args)
    elif args.cmd == 'optimize':
        run_optimize(args)
    else:
        parser.print_help()

def run_grid(args):
    """
    Runs a grid search based on CLI inputs

    Parameters
    ----------
    args : :obj:`argparse.NameSpace`
    """
    # Note: using vars(args) instead of repeating every argument. 
    # It must be ensured that extra arguments are not passed on
    # via the argument parser validations. 
    # Remove extra arguments
    emp_fc_tril = args.emp_fc_tril
    emp_fcd_tril = args.emp_fcd_tril
    del args.cmd, args.emp_fc_tril, args.emp_fcd_tril
    # initialize GridSearch and run it
    gs = optimize.GridSearch(
        **vars(args)
    )
    scores = gs.evaluate(emp_fc_tril, emp_fcd_tril)
    # save the grid and scores
    gs.sim_group.save()
    params_scores = pd.concat([gs.param_combs, scores], axis=1)
    params_scores.to_csv(os.path.join(gs.sim_group.out_dir, 'scores.csv'))

def run_optimize(args):
    """
    Runs an evolutionary optimization based on CLI inputs

    Parameters
    ----------
    args : :obj:`argparse.NameSpace`
    """
    # initialize optimization problem
    problem = optimize.BNMProblem(
        model=args.model,
        params=args.params,
        emp_fc_tril=args.emp_fc_tril,
        emp_fcd_tril=args.emp_fcd_tril,
        het_params=args.het_params,
        maps=args.maps,
        maps_coef_range=args.maps_coef_range,
        node_grouping=args.node_grouping,
        multiobj=args.multiobj,
        sc=args.sc,
        sc_dist=args.sc_dist,
        out_dir=args.out_dir,
        duration=args.duration,
        TR=args.TR,
        states_ts=args.states_ts,
        states_sampling=args.states_sampling,
        window_size=args.window_size,
        window_step=args.window_step,
        rand_seed=args.rand_seed,
        exc_interhemispheric=args.exc_interhemispheric,
        force_gpu=args.force_gpu,
        force_cpu=args.force_cpu,
        serial_nodes=args.serial_nodes,
        gof_terms=args.gof_terms,
        bw_params=args.bw_params,
        bold_remove_s=args.bold_remove_s,
        fcd_drop_edges=args.fcd_drop_edges,
        noise_segment_length=args.noise_segment_length,
        sim_verbose=args.sim_verbose,
        progress_interval=args.progress_interval,
    )
    # initialize optimizer
    optimizer = args.optimizer(
        popsize=args.popsize,
        n_iter=args.n_iter,
        seed=args.optimizer_seed
    )
    # register problem with optimizer
    optimizer.setup_problem(problem)
    # run optimizer and save it
    optimizer.optimize()
    optimizer.save()
    print("Optimal simulation:", optimizer.opt, sep="\n")


if __name__ == '__main__':
    main()

