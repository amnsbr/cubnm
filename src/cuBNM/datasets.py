import sys
import numpy as np

if sys.version_info >= (3, 10):
    from importlib.resources import files
else:
    from importlib_resources import files


def load_sc(what, parc, return_path=False):
    """
    Loads example structural connectivity matrix

    Parameters
    --------
    what: (str)
        'strength' or 'length'
    parc: (str)
        'schaefer-N', 'aparc', or 'glasser-360'
    return_path: (bool)

    Returns
    -------
    mat: (np.ndarray) (nodes, nodes) | (str)
    """
    filename = f"ctx_parc-{parc}_approach-median_mean001_desc-{what}.txt"
    path = files("cuBNM.data").joinpath(filename).as_posix()
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
    Loads example lower triangle of FC/FCD

    Parameters
    --------
    what: (str)
        'FC' or 'FCD'
    parc: (str)
        'schaefer-N', 'aparc', or 'glasser-360'
    exc_interhemispheric: (bool)
    return_path: (bool)

    Returns
    --------
    tril: (np.ndarray) (pairs,) | (str)
    """
    filename = f"ctx_parc-{parc}_hemi-LR"
    if exc_interhemispheric:
        filename += "_exc-inter"
    filename += f"_desc-{what}tril.txt"
    path = files("cuBNM.data").joinpath(filename).as_posix()
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


def load_maps(what, parc, norm="minmax", return_path=False):
    """
    Loads example heterogeneity maps

    Parameters
    --------
    what: (str)
        '6maps', 'yeo7'
    parc: (str)
        'schaefer-N', 'aparc', or 'glasser-360'
    norm: (str | None)
        how maps are normalized: 'zscore', 'minmax'
    return_path: (bool)

    Returns
    -------
    maps: (np.ndarray) (maps, nodes) | (str)
    """
    # TODO: construct the maps in this function based on
    # a list
    filename = f"ctx_parc-{parc}_desc-{what}"
    if norm:
        filename += f"_{norm}"
    filename += ".txt"
    path = files("cuBNM.data").joinpath(filename).as_posix()
    if return_path:
        return path
    try:
        maps = np.loadtxt(path)
    except FileNotFoundError:
        print(f"Maps {what} ({norm}) for {parc} parcellation does not exist")
        return
    return maps
