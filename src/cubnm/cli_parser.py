"""
CLI parser

Note: This is in a separate file so that sphinx doesn't require pandas and cubnm
for autogenerating the CLI docs
"""
import argparse
import os
import ast
import sys
if sys.version_info >= (3, 9):
    from importlib.resources import files
else:
    from importlib_resources import files

def parse_params(param_str):
    """
    Parses input string of parameters and their
    ranges into a dictionary recognizable by GridSearch
    and BNMProblem

    Parameters
    ----------
    param_str : :obj:`str`
        a string with the format "<param_name>=<start>[:<end>[:<step>]],..."

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
    except ValueError:
        raise argparse.ArgumentTypeError("Value should be 'auto' or a list of tuples")
    if len(maps_coef_range) == 1:
        return maps_coef_range[0]
    else:
        return maps_coef_range

def parse_grid_shape(value):
    """
    Parses input string of grid shape

    Parameters
    ----------
    value : :obj:`str`
        - "<int>": same number of points for each parameter
        - "<param_name=size>,...": different number of points for each parameter
    
    Returns
    -------
    :obj:`int` or :obj:`dict`
    """
    try:
        grid_shape = int(value)
    except ValueError:
        grid_shape = {}
        for item in value.split(','):
            key, value = item.split('=')
            grid_shape[key] = int(value)
    return grid_shape

def get_class_names(file_path):
    """
    Returns a list of class names in a python file
    without importing it

    Note: This is important for generating the documentation
    as in the docs environment we don't have the package
    installed (and don't want to have it installed as that
    would require a complicated setup of readthedocs)

    Parameters
    ----------
    file_path : :obj:`str`
        path to the python file
    
    Returns
    -------
    :obj:`list` of :obj:`str`
    """
    with open(file_path, "r", encoding="utf-8") as file:
        tree = ast.parse(file.read(), filename=file_path)
    class_names = [node.name for node in ast.walk(tree) if isinstance(node, ast.ClassDef)]
    return class_names

def get_models():
    """
    Returns a list of available models

    Returns
    -------
    :obj:`list` of :obj:`str`
    """
    if os.environ.get("READTHEDOCS") == "True":
        # as I'm not sure what is the cwd when docs are being built
        # on readthedocs, I'm using the repository path as a more
        # certain way to get the path to sim.py
        sim_py_path = os.path.join(os.environ["READTHEDOCS_REPOSITORY_PATH"], "src", "cubnm", "sim.py")
    else:
        try:
            # when user runs the CLI
            sim_py_path = os.path.abspath(files("cubnm").joinpath("sim.py").as_posix())
        except ModuleNotFoundError:
            # when running sphinx locally current directory is docs
            sim_py_path = os.path.abspath(os.path.join("..", "src", "cubnm", "sim.py"))
    # get class names of sim.py without importing it
    class_names = get_class_names(sim_py_path)
    # get model names from SimGroup classes
    model_names = [c.replace("SimGroup", "") for c in class_names if c.endswith("SimGroup")]
    # remove the empty string corresponding to the base class
    model_names.remove('')
    return model_names


def add_shared_arguments(parser):
    # simgroup arguments shared between all commands
    parser.add_argument('-m', '--model', type=str, required=True, 
                        choices=get_models(),
                        help='Model (required)')
    parser.add_argument('-p', '--params', type=parse_params, required=True, 
                        help='Parameters in custom format inside'
                        ' quotation marks (required). Format:'
                         ' "<param_name>=<start>[:<end>],..."'
                       )
    parser.add_argument('--duration', type=int, required=True, 
                        help='Duration of the simulation (required)')
    parser.add_argument('--TR', type=float, required=True, 
                        help='Repetition time (required)')
    parser.add_argument('--sc', type=str, required=True, 
                        help='Structural connectivity .txt file or "example" (required)')
    parser.add_argument('--sc_dist', type=str, 
                        help='Structural connectivity distribution .txt file or "example"')
    parser.add_argument('--dt', type=str, default='0.1', 
                        help='Neuronal model integration step (msec)')
    parser.add_argument('--bw_dt', type=str, default='1.0', 
                        help='Balloon-Windkessel model integration step (msec)')
    parser.add_argument('-d', '--out_dir', type=str, default="same", 
                        help='Output directory')
    parser.add_argument('--emp_fc_tril', type=str,
                    help='Functional connectivity lower triangle as space-separated .txt file')
    parser.add_argument('--emp_fcd_tril', type=str,
                    help='Functional connectivity dynamics lower triangle as space-separated .txt file')
    parser.add_argument('--emp_bold', type=str,
                    help='Cleaned and parcellated BOLD signal as space-separated .txt file or "example"'
                        ' BOLD signal should be in the shape (nodes, volumes).'
                        ' Motion outliers can either be excluded (not recommended as it disrupts'
                        ' the temporal structure) or replaced with zeros.'
                        ' If provided emp_fc_tril and emp_fcd_tril will be ignored.')
    parser.add_argument('--no_fc', action='store_true', 
                        help='Do not calculate simulated/empirical FC')
    parser.add_argument('--no_fcd', action='store_true', 
                        help='Do not calculate simulated/empirical FCD')
    parser.add_argument('--sim_seed', type=int, default=0, 
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
                        help='Remove initial n seconds of the simulation from FC/FCD calculations'
                           ' and average of state variables')
    parser.add_argument('--window_size', type=int, default=30, 
                        help='FCD window size (s)')
    parser.add_argument('--window_step', type=int, default=5, 
                        help='FCD window step (s)')
    parser.add_argument('--exc_interhemispheric', action='store_true', 
                        help='Exclude interhemispheric connections in FC/FCD calculations')
    parser.add_argument('--fcd_keep_edges', action='store_true', 
                        help='Keep edge windows from FCD calculations')
    parser.add_argument('--gof_terms',
                        type=lambda s: s.split(','),
                        default=["fc_corr","fcd_ks"],
                        help='Goodness of fit terms (comma separated in quotation marks)'
                            ' e.g. "fc_corr,fcd_ks".'
                       )
    parser.add_argument('--force_cpu', action='store_true', 
                        help='Force CPU')
    parser.add_argument('--force_gpu', action='store_true', 
                        help='Force GPU')
    parser.add_argument('-v', '--sim_verbose', action='store_true',
                        help='Show simulation progress')
    parser.add_argument('--progress_interval', type=int, default=500, 
                        help='Interval of progress updates (in msec).'
                            ' Only used if sim_verbose is True'
                       )
    parser.add_argument('--no_print_args', action='store_true', 
                        help='Do not print command line arguments table')
    # BNMProblem arguments
    parser.add_argument('--het_params', nargs='+', default=[],
                        help='List of heterogeneous regional parameters (space separated)')
    parser.add_argument('--maps', type=str, 
                        help='Path to heterogeneity maps or "example"')
    parser.add_argument('--maps_coef_range', type=parse_maps_coef_range, 
                        default='auto',
                        help='Coefficient range for maps.'
                        ' Options: "auto", "min:max" (same for all maps),'
                        ' "min1:max1,min2:max2,..." (different for each map)'
                        )
    parser.add_argument('--node_grouping', type=str, 
                        default=None,
                        help='Path to node grouping array or special values: node, sym')
    parser.add_argument('--multiobj', action='store_true', 
                        help='Instead of combining the objectives into a single objective'
                        ' function (via summation) defines each objective separately.'
                        ' This must not be used with single-objective optimizers'
                        )


def get_parser():
    parser = argparse.ArgumentParser(
        description='cuBNM command line interface',
    )
    # add specific commands and their options
    subparsers = parser.add_subparsers(dest='cmd')

    # optimizer command
    parser_optimize = subparsers.add_parser('optimize', help='Optimize parameters using evolutionary algorithms')
    add_shared_arguments(parser_optimize)
    # Optimizer arguments
    parser_optimize.add_argument('-o', '--optimizer', type=str, required=True,
                                help='Optimizer type')
    parser_optimize.add_argument('--optimizer_seed', type=int, default=0, 
                                help='Optimizer random seed')
    parser_optimize.add_argument('--n_iter', type=int, default=80, 
                                help='Optimizer max iterations')
    parser_optimize.add_argument('--popsize', type=int, default=24, 
                                help='Optimizer population size')

    # grid command
    parser_grid = subparsers.add_parser('grid', help='Grid search')
    add_shared_arguments(parser_grid)
    parser_grid.add_argument('--grid_shape', type=parse_grid_shape, required=True,
                            help='Shape of the grid search: <int> (same number of'
                                'points for each parameter) or "<param_name=size>,..."'
                                ' (different number of points for each parameter)'
                            )
    
    # TODO: add model specific options (e.g. do_fic in rWW)
    # TODO: add optimizer specific options
    
    # TODO: add run command
    # parser_run = subparsers.add_parser('run', help='Run one simulation with given parameters')

    return parser