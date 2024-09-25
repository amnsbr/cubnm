"""
Command line interface for cuBNM
"""
import os
import pandas as pd
from cubnm import datasets, optimize, cli_parser

def main():
    parser = cli_parser.get_parser()
    # parse the arguments
    args = parser.parse_args()

    if args.cmd not in ['grid', 'optimize']:
        parser.print_help()
        return
    
    # additional validations
    # make sure either bold, fc or fcd are provided
    if (args.emp_fc_tril is None) and (args.emp_fcd_tril is None) and (args.emp_bold is None):
        parser.error("At least one of the following must be provided: --emp_fc_tril, --emp_fcd_tril, --emp_bold")

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
    args.do_fc = not args.no_fc
    args.do_fcd = not args.no_fcd
    args.fcd_drop_edges = not args.fcd_keep_edges
    args.ext_out = not args.no_ext_out
    del args.no_fc, args.no_fcd, args.fcd_keep_edges, args.no_ext_out

    # add +/- to gof terms
    for i, term in enumerate(args.gof_terms):
        if term == 'fc_corr':
            args.gof_terms[i] = '+' + term
        else:
            args.gof_terms[i] = '-' + term

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
    if args.emp_bold == 'example':
        args.emp_bold = datasets.load_functional(
            'bold', 'schaefer-100'
        )
    if (args.cmd == 'optimize') and (args.maps == 'example'):
        args.maps = datasets.load_maps(
            ['myelinmap', 'fcgradient01'],
            'schaefer-100'
        )

    if args.cmd == 'grid':
        run_grid(args)
    elif args.cmd == 'optimize':
        # additional refinement of specific arguments
        if args.het_params is None:
            args.het_params = []
        run_optimize(args)

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
    emp_bold = args.emp_bold
    del args.cmd, args.emp_fc_tril, args.emp_fcd_tril, args.emp_bold
    # initialize GridSearch and run it
    gs = optimize.GridSearch(
        **vars(args)
    )
    scores = gs.evaluate(emp_fc_tril, emp_fcd_tril, emp_bold)
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
        emp_bold=args.emp_bold,
        do_fc=args.do_fc,
        do_fcd=args.do_fcd,
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
        dt=args.dt,
        bw_dt=args.bw_dt,
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
    optimizer_cls = getattr(optimize, f'{args.optimizer}Optimizer')
    optimizer = optimizer_cls(
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

