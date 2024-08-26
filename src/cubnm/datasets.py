"""
Example datasets
"""
import sys
import numpy as np

if sys.version_info >= (3, 10):
    from importlib.resources import files
else:
    from importlib_resources import files


def load_sc(what, parc, return_path=False):
    """
    Load example structural connectivity matrix

    Parameters
    ----------
    what : {'strength', 'length'}
        - 'strength': SC strength (normalized tract counts)
        - 'length': SC tracts length
    parc : {'schaefer-[100, 200, 400, 600]', 'aparc', 'glasser-360'}
        parcellation. For Schaefer, specify number of parcels.
    return_path : :obj:`bool`, optional
        If True, returns path to the file
        Otherwise, returns the matrix

    Returns
    -------
    :obj:`np.ndarray` or :obj:`str`
        Structural connectivity matrix or path to its
        text file. Shape: (nodes, nodes)
    """
    filename = f"ctx_parc-{parc}_approach-median_mean001_desc-{what}.txt"
    path = files("cubnm.data").joinpath(filename).as_posix()
    if return_path:
        return path
    try:
        mat = np.loadtxt(path)
    except FileNotFoundError:
        print(f"SC {what} for {parc} parcellation does not exist")
        return
    return mat


def load_functional(what, parc, exc_interhemispheric=True, return_path=False):
    """
    Load example lower triangle of FC/FCD

    Parameters
    --------
    what: {'FC', 'FCD'}
        - 'FC': functional connectivity
        - 'FCD': functional connectivity dynamics
    parc: 'schaefer-100'
        parcellation
    exc_interhemispheric: :obj:`bool`, optional
        whether to exclude interhemispheric connections
    return_path : :obj:`bool`, optional
        If True, returns path to the file
        Otherwise, returns the matrix
        

    Returns
    --------
    :obj:`np.ndarray` or :obj:`str`
        Lower triangle of FC/FCD matrix or path to its
        text file. Shape: (n_pairs,)
    """
    # TODO: Add other parcellations
    filename = f"ctx_parc-{parc}_hemi-LR"
    if exc_interhemispheric:
        filename += "_exc-inter"
    filename += f"_desc-{what}tril.txt"
    path = files("cubnm.data").joinpath(filename).as_posix()
    if return_path:
        return path
    try:
        tril = np.loadtxt(path)
    except FileNotFoundError:
        print(
            f"{what} for {parc} parcellation (exc_interhemispheric {exc_interhemispheric}) does not exist"
        )
        return
    return tril


def load_maps(names, parc, norm="minmax"):
    """
    Loads example heterogeneity maps

    Parameters
    ----------
    names: :obj:`str` or :obj:`list`
        One or more maps selected from this list:
        - 'myelinmap'
        - 'thickness'
        - 'fcgradient01'
        - 'genepc1'
        - 'nmda'
        - 'gabaa'
        - 'yeo7'
    parc: {'schaefer-100'}
        parcellation
    norm: {'zscore', 'minmax', None}
        - 'zscore': maps are z-score normalized
        - 'minmax': maps are min-max normalized to [0, 1]
    return_path : :obj:`bool`, optional
        If True, returns path to the file
        Otherwise, returns the matrix

    Returns
    --------
    :obj:`np.ndarray` or :obj:`str`
        Maps arrays or path to their
        text file. Shape: (maps, nodes)

    Notes
    -----
    For more information and code on how these maps were
    obtained and parcellated see `utils.datasets.load_maps`
    in https://github.com/amnsbr/eidev. The set of maps included
    here are limited and provided just as examples. We recommend 
    users to use `neuromaps` and similar tools to obtain and 
    parcellate further maps.
    """
    # TODO: Add other parcellations
    if isinstance(names, str):
        names = [names]
    maps = []
    for name in names:
        if name not in ['yeo7']:
            filename = f"ctx_parc-{parc}_desc-{name}_zscore.txt"
        else:
            norm = None
            filename = f"ctx_parc-{parc}_desc-{name}.txt"
        path = files("cubnm.data").joinpath(filename).as_posix()
        curr_map = np.loadtxt(path)
        if norm=='minmax':
            curr_map = (curr_map - np.min(curr_map)) / (np.max(curr_map) - np.min(curr_map))
        maps.append(curr_map.reshape(1, -1))
    maps = np.vstack(maps)
    return maps
