"""
Example datasets
"""
import sys
import os
import numpy as np
import pandas as pd

if sys.version_info >= (3, 10):
    from importlib.resources import files
else:
    from importlib_resources import files

def _get_lut_full(parc):
    """
    Load the full (cortical+subcortical) lookup table for the given parcellation

    Parameters
    ----------
    parc: {'schaefer-[100, 200, ... 1000]', 'aparc', 'glasser-360'}
        Parcellation

    Returns
    -------
    :obj:`pandas.DataFrame`
        Full lookup table
    """
    parcs_dir = files("cubnm.data").joinpath("parcellations").as_posix()
    lut_sctx_mics = pd.read_csv(os.path.join(parcs_dir, 'lut', 'lut_subcortical-cerebellum_mics.csv'))
    lut_parc_mics = pd.read_csv(os.path.join(parcs_dir, 'lut', f'lut_{parc}_mics.csv'))
    # order them similar to  micapipe/functions/connectome_slicer.R 
    # (which is the order of parcels in full_sc)
    lut_full = pd.concat([lut_parc_mics, lut_sctx_mics], axis=0).sort_values(by='mics')
    # in micamics the subcortical+cortical SC is published and cerebellum parcels are excluded
    # from the SC file => they should also be excluded from lut_full (which later determines
    # labels of full_SC dataframe)
    lut_full = lut_full.loc[~((lut_full['mics'] >= 100) & (lut_full['mics'] < 1000))]
    return lut_full

def _clean_micamics_sc(
        micamics_dir, measure, parc, sub, exc_subcortex=True, 
        norm='mean001', out_dir=None
    ):
    """
    Clean SC matrix of the given subject and parcellation

    Parameters
    ----------
    micamics_dir: :obj:`str`
        Path to the directory containing the micapipe outputs of micamics
        (https://osf.io/x7qr2 unzipped)
    measure: {'strength', 'length'}
        - 'strength': SC strength (normalized tract counts)
        - 'length': SC tracts length
    parc: {'schaefer-[100, 200, ... 1000]', 'aparc', 'glasser-360'}
        Parcellation
    sub: :obj:`str`
        Subject ID, e.g. "sub-HC001"
    exc_subcortex: :obj:`bool`, optional
        Whether to exclude subcortical regions. Default: True
    norm: {'mean001', 'none'}
        SC strength normalization method
        - 'mean001': normalize to mean 0.01. Default
        - 'none': no normalization
    out_dir: :obj:`str`, optional
        Path to save the cleaned SC matrix
    """
    sc_file_prefix = os.path.join(
        micamics_dir, sub, 'ses-01', 'dwi', 
        f'{sub}_ses-01_space-dwinative_atlas-{parc.replace("-", "")}_desc'
    )
    if measure == 'strength':
        sc_file_path = sc_file_prefix + '-sc'
    elif measure == 'length':
        sc_file_path = sc_file_prefix + '-edgeLength'
    full_sc = np.loadtxt(sc_file_path+'.txt', delimiter=',')
    # label it and select indicated parcels
    lut_full = _get_lut_full(parc)
    # specify parcels to exclude
    ## midline
    exc_parcels = [1000, 2000]
    ## for glasser-360 exclude the parcel L_H and R_H (hippocampus)
    if parc == 'glasser-360':
        exc_parcels += [1120, 2120]
    ## for aparc remove L/R_corpuscallosum
    elif parc == 'aparc':
        exc_parcels += [1004, 2004]
    ## exclude subcortex (mics 10-100) if indicated
    if exc_subcortex:
        exc_parcels += lut_full.loc[(lut_full['mics']) < 100, 'mics'].values.tolist()
    exc_parcels = list(set(exc_parcels)) # drops duplicates
    # exclude specified parcles and label
    full_sc = pd.DataFrame(full_sc, index=lut_full['mics'], columns=lut_full['mics'])
    sc = full_sc.drop(index=exc_parcels, columns=exc_parcels)
    # make symmetric
    sym_sc = (sc + sc.T)
    # remove diagonal
    sym_sc.values[np.diag_indices_from(sym_sc)] = 0
    # order by hemispheres
    all_hemi_parcs = {
        'L': lut_full.loc[
                ((lut_full['mics']>1000) & (lut_full['mics']<2000)) | (lut_full['mics']<49)
                ,'mics'].values.tolist(),
        'R': lut_full.loc[
                (lut_full['mics']>2000) | ((lut_full['mics']>=49) & (lut_full['mics']<100))
                ,'mics'].values.tolist()
    }
    hemi_parcs = {
        'L': sym_sc.index.intersection(lut_full.loc[lut_full['mics'].isin(all_hemi_parcs['L']), 'mics']),
        'R': sym_sc.index.intersection(lut_full.loc[lut_full['mics'].isin(all_hemi_parcs['R']), 'mics']),
    }
    if not exc_subcortex:
        # order L -> R (instead of ctx -> sctx)
        ordered_parcs = hemi_parcs['L'].tolist() + hemi_parcs['R'].tolist()
        sym_sc = sym_sc.loc[ordered_parcs, ordered_parcs]
    if measure == 'strength':
        if norm == 'mean001':
            # normalize to mean 0.01
            sym_sc = (sym_sc / sym_sc.values.mean()) * 0.01
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        out_fname = f'{sub}_ses-01_space-dwinative_atlas-{parc.replace("-", "")}'
        if not exc_subcortex:
            out_fname += '_sctx'
        if measure == 'strength':
            out_fname += f'_norm-{norm}'
        out_fname += f'_desc-{measure}.npz'
        np.savez_compressed(os.path.join(out_dir, out_fname), sym_sc.values)
    return sym_sc.values

def _clean_micamics_bold(micamics_dir, parc, sub, exc_subcortex=True, out_dir=None):
    """
    Clean BOLD time series of the given subject and parcellation

    Parameters
    ----------
    micamics_dir: :obj:`str`
        Path to the directory containing the micapipe outputs of micamics
        (https://osf.io/x7qr2 unzipped)
    parc: {'schaefer-[100, 200, ... 1000]', 'aparc', 'glasser-360'}
        Parcellation
    sub: :obj:`str`
        Subject ID, e.g. "sub-HC001"
    exc_subcortex: :obj:`bool`, optional
        Whether to exclude subcortical regions. Default: True
    out_dir: :obj:`str`, optional
        Path to save the cleaned BOLD time series
    """
    bold_file_path = os.path.join(
        micamics_dir, sub, 'ses-01', 'func', 
        f'{sub}_ses-01_space-fsnative_atlas-{parc.replace("-", "")}_desc-timeseries.txt'
    )
    bold = np.loadtxt(bold_file_path, delimiter=',')
    # label it and select indicated parcels
    lut_full = _get_lut_full(parc)
    # specify parcels to exclude
    ## midline
    exc_parcels = [1000, 2000]
    ## for glasser-360 exclude the parcel L_H and R_H (hippocampus)
    if parc == 'glasser-360':
        exc_parcels += [1120, 2120]
    ## for aparc remove L/R_corpuscallosum
    elif parc == 'aparc':
        exc_parcels += [1004, 2004]
    ## exclude subcortex (mics 10-100) if indicated
    if exc_subcortex:
        exc_parcels += lut_full.loc[(lut_full['mics']) < 100, 'mics'].values.tolist()
    exc_parcels = list(set(exc_parcels)) # drops duplicates
    # transpose to (nodes,time), exclude specified parcles and label
    bold = pd.DataFrame(bold.T, index=lut_full['mics'])
    bold = bold.drop(index=exc_parcels)
    # order by hemispheres
    all_hemi_parcs = {
        'L': lut_full.loc[
                ((lut_full['mics']>1000) & (lut_full['mics']<2000)) | (lut_full['mics']<49)
                ,'mics'].values.tolist(),
        'R': lut_full.loc[
                (lut_full['mics']>2000) | ((lut_full['mics']>=49) & (lut_full['mics']<100))
                ,'mics'].values.tolist()
    }
    hemi_parcs = {
        'L': bold.index.intersection(lut_full.loc[lut_full['mics'].isin(all_hemi_parcs['L']), 'mics']),
        'R': bold.index.intersection(lut_full.loc[lut_full['mics'].isin(all_hemi_parcs['R']), 'mics']),
    }
    if not exc_subcortex:
        # order L -> R (instead of ctx -> sctx)
        ordered_parcs = hemi_parcs['L'].tolist() + hemi_parcs['R'].tolist()
        bold = bold.loc[ordered_parcs]
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        out_fname = f'{sub}_ses-01_space-fsnative_atlas-{parc.replace("-", "")}'
        if not exc_subcortex:
            out_fname += '_sctx'
        out_fname += '_desc-bold.npz'
        np.savez_compressed(os.path.join(out_dir, out_fname), bold.values)
    return bold.values

def load_sc(measure, parc, sub="sub-HC001", exc_subcortex=True, 
        norm="mean001", micamics_dir=None):
    """
    Load example structural connectivity matrix from MICA-MICs dataset
    (https://www.nature.com/articles/s41597-022-01682-y)

    Parameters
    ----------
    measure: {'strength', 'length'}
        - 'strength': SC strength (normalized tract counts)
        - 'length': SC tracts length
    parc: {'schaefer-[100, 200, ... 1000]', 'aparc', 'glasser-360'}
        Parcellation
    sub: :obj:`str`
        Subject ID, e.g. "sub-HC001"
    exc_subcortex: :obj:`bool`, optional
        Whether to exclude subcortical regions. Default: True
    norm: {'mean001', 'none'}
        SC strength normalization method
        - 'mean001': normalize to mean 0.01. Default
        - 'none': no normalization
    micamics_dir: :obj:`str`, optional
        Path to the directory containing the micapipe outputs of micamics
        (https://osf.io/x7qr2 unzipped). Required if subjects other than 
        'sub-HC001' are requested and/or exc_subcortex is False and/or
        norm is 'none'.

    Returns
    -------
    :obj:`np.ndarray` or :obj:`str`
        Structural connectivity matrix or path to its
        text file. Shape: (nodes, nodes)
    """
    if (sub != "sub-HC001") | (not exc_subcortex) | (norm == 'none'):
        if not micamics_dir:
            raise ValueError("micamics_dir is required")
        return _clean_micamics_sc(
            micamics_dir, measure, parc, sub, exc_subcortex=exc_subcortex, out_dir=None
        )
    fname = f'{sub}_ses-01_space-dwinative_atlas-{parc.replace("-", "")}'
    if not exc_subcortex:
        fname += '_sctx'
    if measure == 'strength':
        fname += f'_norm-{norm}'
    fname += f'_desc-{measure}.npz'
    path = files("cubnm.data").joinpath('micamics', 'sc', fname).as_posix()
    mat = np.load(path)['arr_0']
    return mat


def load_bold(parc, sub="sub-HC001", exc_subcortex=True, micamics_dir=None):
    """
    Load example BOLD data from the MICA-MICs dataset
    (https://www.nature.com/articles/s41597-022-01682-y)

    Parameters
    --------
    parc: 'schaefer-100'
        parcellation
    sub: :obj:`str`
        Subject ID, e.g. "sub-HC001"
    exc_subcortex: :obj:`bool`, optional
        Whether to exclude subcortical regions. Default: True
    micamics_dir: :obj:`str`, optional
        Path to the directory containing the micapipe outputs of micamics
        (https://osf.io/x7qr2 unzipped). Required if subjects other than 
        'sub-HC001' are requested and/or exc_subcortex is False.

    Returns
    --------
    :obj:`np.ndarray` or :obj:`str`
        Path to a text file or numpy array including
        the BOLD signal. Shape: (nodes, volumes)
    """
    if (sub != "sub-HC001") | (not exc_subcortex):
        if not micamics_dir:
            raise ValueError("micamics_dir is required")
        return _clean_micamics_bold(
            micamics_dir, parc, sub, exc_subcortex=exc_subcortex, out_dir=None
        )
    fname = f'{sub}_ses-01_space-fsnative_atlas-{parc.replace("-", "")}'
    if not exc_subcortex:
        fname += '_sctx'
    fname += '_desc-bold.npz'
    path = files("cubnm.data").joinpath('micamics', 'bold', fname).as_posix()
    out = np.load(path)['arr_0']
    return out

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
        path = files("cubnm.data").joinpath('maps', filename).as_posix()
        curr_map = np.loadtxt(path)
        if norm=='minmax':
            curr_map = (curr_map - np.min(curr_map)) / (np.max(curr_map) - np.min(curr_map))
        maps.append(curr_map.reshape(1, -1))
    maps = np.vstack(maps)
    return maps
