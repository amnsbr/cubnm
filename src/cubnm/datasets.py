"""
Example datasets
"""
import sys
import os
import numpy as np
import pandas as pd

if sys.version_info >= (3, 9):
    from importlib.resources import files
else:
    from importlib_resources import files

from cubnm import utils

def load_sc(
        measure, 
        parc="schaefer-100", 
        sub="group-train706", 
        norm="mean",
    ):
    """
    Load example structural connectivity matrix from HCP-YA dataset

    Parameters
    ----------
    measure: {'strength', 'length'}
        Structural connectivity measure.

        - ``'strength'``: SC strength (tract counts)
        - ``'length'``: SC tracts length

    parc: :obj:`str`
        Parcellation. Currently only ``'schaefer-100'`` is supported.
    sub: :obj:`str`
        Subject or group ID. Two subjects and two group-averaged 
        matrices are currently included:

        - ``'100206'``
        - ``'100307'``
        - ``'group-train706'``
        - ``'group-test303'``

    norm: {'mean', None}
        SC strength normalization method.
        Only used when ``measure`` is ``'strength'``.

        - ``'mean'``: normalize to mean 0.01. Default
        - ``None``: no normalization

    Returns
    -------
    :obj:`np.ndarray`
        Structural connectivity matrix. Shape: (nodes, nodes)
    """
    path = (
        files("cubnm.data").joinpath('hcp')
        .joinpath('sc').joinpath(sub)
        .joinpath(f'ctx_parc-{parc}_desc-{measure}.npz')
    ).as_posix()
    try:
        mat = np.load(path)['arr_0']
    except FileNotFoundError as e:
        raise ValueError(
            f"Structural connectivity {measure} matrix in {parc} "
            f"for {sub} is not included in the toolbox."
        ) from e
    # normalization
    if measure == 'strength':
        if norm == 'mean':
            mat /= (mat.mean() * 100)
    return mat

def load_bold(parc="schaefer-100", sub="100206", ses="REST1_LR"):
    """
    Load example BOLD data from the HCP-YA dataset

    Parameters
    --------
    parc: :obj:`str`
        Parcellation. Currently only ``'schaefer-100'`` is available.
    sub: :obj:`str`
        Subject ID. Currently two subjects are included:

        - ``'100206'``
        - ``'100307'``

    ses: {'REST1_LR', 'REST2_LR'}
        Imaging session.

    Returns
    --------
    :obj:`np.ndarray`
        The BOLD signal. Shape: (nodes, volumes)
    """
    path = (
        files("cubnm.data").joinpath('hcp').joinpath('bold')
        .joinpath(sub).joinpath(ses)
        .joinpath(f'ctx_parc-{parc}_desc-bold.npz')
    ).as_posix()
    try:
        out = np.load(path)['arr_0']
    except FileNotFoundError as e:
        raise ValueError(
            f"BOLD data for {parc} and {sub} in session {ses} "
            f"is not included in the toolbox."
        ) from e
    return out

def load_fc(
        parc="schaefer-100", 
        sub="group-train706", 
        ses="REST",
        exc_interhemispheric=False,
        return_tril=True,
    ):
    """
    Load example FC from the HCP-YA dataset

    Parameters
    --------
    parc: :obj:`str`
        Parcellation. Currently only ``'schaefer-100'`` is available.
    sub: :obj:`str`
        Subject or group ID. Two subjects and two group-averaged 
        matrices are currently included:

        - ``'100206'``
        - ``'100307'``
        - ``'group-train706'``
        - ``'group-test303'``

    ses: {'REST1_LR', 'REST2_LR', 'REST'}
        Imaging session. For subject-level data currently
        only ``'REST1_LR'``, and ``'REST2_LR'`` are included and
        for group-level data only ``'REST'`` is included.
        ``'REST'`` includes the data averaged across
        all sessions.
    exc_interhemispheric: :obj:`bool`
        Whether to exclude interhemispheric connections. Default: False
    return_tril: :obj:`bool`
        Whether to return the lower triangular part of the FC matrix.

    Returns
    --------
    :obj:`np.ndarray`
        Functional connectivity matrix or its lower triangular part.
        Shape: (nodes, nodes) or (node_pairs,)
    """
    if "group-" in sub:
        if exc_interhemispheric:
            tags = '_exc-inter'
        else:
            tags = ''
        path = (
            files("cubnm.data").joinpath('hcp').joinpath('fc')
            .joinpath(sub).joinpath(ses)
            .joinpath(f'ctx_parc-{parc}{tags}_desc-fc.npz')
        ).as_posix()
        try:
            fc = np.load(path)['arr_0']
        except FileNotFoundError as e:
            raise ValueError(
                f"Functional connectivity matrix for {parc} "
                f"and {sub} in session {ses} is not included in the toolbox."
            ) from e
        if return_tril:
            fc = fc[np.tril_indices(fc.shape[0], -1)]
            if exc_interhemispheric:
                fc = fc[~np.isnan(fc)]
        return fc
    else:
        # load bold
        bold = load_bold(
            parc=parc, 
            sub=sub,
            ses=ses, 
        )
        # calculate FC
        fc = utils.calculate_fc(
            bold, 
            exc_interhemispheric=exc_interhemispheric,
            return_tril=return_tril
        )
        return fc

def load_fcd(
        parc="schaefer-100", 
        sub="group-train706",
        ses="REST", 
        window_size=30,
        window_step=5,
        drop_edges=True,
        exc_interhemispheric=False,
        return_tril=True,
    ):
    """
    Load example FCD from the HCP-YA dataset

    Parameters
    --------
    parc: :obj:`str`
        Parcellation. Currently only ``'schaefer-100'`` is available.
    sub: :obj:`str`
        Subject or group ID. Two subjects and two group-averaged 
        matrices are currently included:

        - ``'100206'``
        - ``'100307'``
        - ``'group-train706'``
        - ``'group-test303'``

    ses: {'REST1_LR', 'REST2_LR', 'REST'}
        Imaging session. For subject-level data currently
        only ``'REST1_LR'``, and ``'REST2_LR'`` are included and
        for group-level data only ``'REST'`` is included.
        ``'REST'`` includes the data pooled across
        all sessions.
    window_size: :obj:`int`
        dynamic FC window size (in seconds)
        will be converted to N TRs (nearest even number)
        The actual window size is number of TRs + 1 (including center)
    window_step: :obj:`int`
        dynamic FC window step (in seconds)
        will be converted to N TRs
    drop_edges: :obj:`bool`
        drop edge windows which have less than window_size volumes
    exc_interhemispheric: :obj:`bool`
        Whether to exclude interhemispheric connections. Default: False
    return_tril: :obj:`bool`
        Whether to return the lower triangular part of the FCD matrix.
        Ignored when ``sub`` is a group (in this case
        the return matrix is pooled from FCD lower triangles of
        individual subjects).

    Returns
    --------
    :obj:`np.ndarray`
        Functional connectivity dynamics matrix or its lower triangular part.
        Shape: (nodes, nodes) or (node_pairs,)
    """
    if "group-" in sub:
        if not return_tril:
            print(
                "Warning: return_tril is ignored when sub='group-all'. "
                "Returning lower triangle."
            )
        if exc_interhemispheric:
            tags = '_exc-inter'
        else:
            tags = ''
        tags += f'_window-{window_size}_step-{window_step}'
        path = (
            files("cubnm.data").joinpath('hcp').joinpath('fcd')
            .joinpath(sub).joinpath(ses)
            .joinpath(f'ctx_parc-{parc}{tags}_desc-fcdtril.npz')
        ).as_posix()
        try:
            fcd = np.load(path)['arr_0']
        except FileNotFoundError as e:
            raise ValueError(
                f"Functional connectivity dynamics for {parc} "
                f"and {sub} in session {ses} using window size {window_size} "
                f"and step {window_step} is not included in the toolbox."
            ) from e
        return fcd
    else:
        # create it from bold
        bold = load_bold(
            parc=parc, 
            sub=sub, 
            ses=ses
        )
        # convert window size and step to TRs
        TR = 0.72
        window_size_TRs = int(np.round(window_size / (TR*2))) * 2
        window_step_TRs = int(np.round(window_step / TR))
        # calculate FCD
        fcd = utils.calculate_fcd(
            bold, 
            window_size=window_size_TRs, 
            window_step=window_step_TRs,
            drop_edges=drop_edges,
            exc_interhemispheric=exc_interhemispheric,
            return_tril=return_tril
        )
        return fcd

def load_maps(names, parc="schaefer-100", norm="minmax"):
    """
    Loads example heterogeneity maps

    Parameters
    ----------
    names: :obj:`str` or :obj:`list`
        One or more maps selected from this list:

        - ``'myelinmap'``
        - ``'thickness'``
        - ``'fcgradient01'``
        - ``'genepc1'``
        - ``'nmda'``
        - ``'gabaa'``
        - ``'yeo7'``

    parc: :obj:`str`
        Parcellation. Currently only ``'schaefer-100'`` is supported.
    norm: {'zscore', 'minmax', None}
        Map normalization method applied across nodes.

        - ``'zscore'``: maps are z-score normalized
        - ``'minmax'``: maps are min-max normalized to [0, 1]
        - ``None``: no normalization

    Returns
    --------
    :obj:`np.ndarray` or :obj:`str`
        Maps arrays or path to their
        text file. Shape: (maps, nodes)

    Notes
    -----
    For more information and code on how these maps were
    obtained and parcellated see ``utils.datasets.load_maps``
    in https://github.com/amnsbr/eidev. The set of maps included
    here are limited and provided just as examples. We recommend 
    users to use ``neuromaps`` and similar tools to obtain and 
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
        path = files("cubnm.data").joinpath('maps').joinpath(filename).as_posix()
        curr_map = np.loadtxt(path)
        if norm=='minmax':
            curr_map = (curr_map - np.min(curr_map)) / (np.max(curr_map) - np.min(curr_map))
        maps.append(curr_map.reshape(1, -1))
    maps = np.vstack(maps)
    return maps
