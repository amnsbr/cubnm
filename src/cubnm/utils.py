"""
Utility functions
"""
import numpy as np
import scipy
import subprocess

def avail_gpus():
    """
    Get the number of available GPUs

    Returns
    -------
    :obj:`int`
        Number of available GPUs
    """
    try:
        output = subprocess.check_output(['nvidia-smi', '-L'])
        output = output.decode('utf-8').strip()
        gpu_list = output.split('\n')
        gpu_count = len(gpu_list)
        return gpu_count
    except subprocess.CalledProcessError:
        return 0
    except FileNotFoundError:
        return 0

def is_jupyter():
    """
    This function checks if the current environment is a Jupyter notebook.

    Returns:
        bool: True if the current environment is a Jupyter notebook, False otherwise.
    """
    try:
        from IPython import get_ipython
    except ImportError:
        return False
    if hasattr(get_ipython(), 'config'):
        if 'IPKernelApp' in get_ipython().config:  
            return True
    return False

def fc_norm_euclidean(x, y):
    """
    Calculates Euclidean distance of two FC arrays
    divided by their maximum possible distance, equal
    to the distance of np.ones(n_pairs) and -np.ones(n_pairs)
    or 2 * np.sqrt(n_pairs)

    Parameters
    ----------
    x, y: :obj:`np.ndarray`
        FC arrays. Shape: (n_pairs,)

    Returns
    -------
    :obj:`float`
        Normalized Euclidean distance
    """
    euclidean = scipy.spatial.distance.euclidean(x, y)
    max_euclidean = 2 * np.sqrt(x.size)
    return euclidean / max_euclidean

def get_bw_params(src):
    """
    Get Balloon-Windkessel model parameters

    Parameters
    ----------
    src: {'friston2003', 'heinzle2016-3T'}
        - 'friston2003': Friston et al. 2003
        - 'heinzle2016-3T': Heinzle et al. 2016, 3T parameters

    Returns
    -------
    :obj:`dict`
        Balloon-Windkessel model parameters
    """
    if src == 'friston2003':
        rho = 0.34
        return {
            'k1': 7*rho,
            'k2': 2.0,
            'k3': 2*rho - 0.2,
        }
    elif src == 'heinzle2016-3T':
        return {
            'k1': 3.72,
            'k2': 0.527,
            'k3': 0.53,
        }
    else:
        raise ValueError(f'Unknown BW parameters source: {src}')

def calculate_fc(
    bold,
    exc_interhemispheric=False,
    return_tril=True,
):
    """
    Calculates functional connectivity matrix

    Parameters
    ---------
    bold: :obj:`np.ndarray`
        cleaned and parcellated empirical BOLD time series. Shape: (nodes, volumes)
        Motion outliers should either be excluded or replaced with zeros.
    exc_interhemispheric: :obj:`bool`, optional
        exclude interhemispheric connections
    return_tril: :obj:`bool`, optional
        return only the lower triangular part of the FCD matrix

    Returns
    -------
    fc: :obj:`np.ndarray`
        FC dynamics matrix. 
        Shape: (nodes, nodes) or (n_node_pairs,)
        if return_tril is True
    """
    # z-score non-zero volumes in each node
    outlier_vols = bold.sum(axis=0) == 0
    bold[:, ~outlier_vols] = scipy.stats.zscore(bold[:, ~outlier_vols], axis=1)
    # calculate FC
    fc = np.corrcoef(bold)
    if exc_interhemispheric:
        # set interhemispheric connections to NaN
        rh_idx = bold.shape[0] // 2
        fc[:rh_idx, rh_idx:] = np.NaN
        fc[rh_idx:, :rh_idx] = np.NaN
    if return_tril:
        fc = fc[np.tril_indices(fc.shape[0], -1)]
        # drop NaNs (interhemispheric connections)
        fc = fc[~np.isnan(fc)]
    return fc


def calculate_fcd(
    bold,
    window_size,
    window_step,
    drop_edges=True,
    outlier_threshold=0.5,
    exc_interhemispheric=False,
    return_tril=True,
    return_dfc=False,
):
    """
    Calculates functional connectivity dynamics matrix
    and dynamic functional connectivity matrices

    Parameters
    ---------
    bold: :obj:`np.ndarray`
        cleaned and parcellated empirical BOLD time series. Shape: (nodes, volumes)
        Motion outliers should either be excluded (not recommended as it disrupts
        the temporal structure) or replaced with zeros.
    window_size: :obj:`int`, optional
        dynamic FC window size (in TR)
        Must be even. The actual window size is +1 (including center).
    window_step: :obj:`int`, optional
        dynamic FC window step (in TR)
    drop_edges: :obj:`bool`, optional
        drop edge windows which have less than window_size volumes
    outlier_threshold: :obj:`float`, optional
        threshold for the proportion of motion outliers in a window
        that would lead to discarding the window
    exc_interhemispheric: :obj:`bool`, optional
        exclude interhemispheric connections
    return_tril: :obj:`bool`, optional
        return only the lower triangular part of the FCD matrix
    return_dfc: :obj:`bool`, optional
        return dynamic FCs as well

    Returns
    -------
    fcd_matrix: :obj:`np.ndarray`
        FC dynamics matrix. 
        Shape: (n_windows, n_windows) or (n_window_pairs,)
        if return_tril is True
    window_fcs: :obj:`np.ndarray`
        dynamic FCs. Shape: (nodes, nodes, n_windows)
        Returned only if return_dfc is True
    """
    # TODO: make non-even window size work
    assert window_size % 2 == 0, "Window size must be even"
    # z-score non-zero volumes in each node
    outlier_vols = bold.sum(axis=0) == 0
    bold[:, ~outlier_vols] = scipy.stats.zscore(bold[:, ~outlier_vols], axis=1)
    # calculate dynamic FC
    nodes = bold.shape[0]
    n_vols = bold.shape[1]
    window_fc_trils = []
    window_fcs = []
    if drop_edges:
        first_center = int(window_size / 2)
        last_center = n_vols - 1 - int(window_size / 2)
    else:
        first_center = 0
        last_center = n_vols - 1
    window_center = first_center
    while window_center <= last_center:
        window_start = window_center - int(window_size / 2)
        if window_start < 0:
            window_start = 0
        window_end = window_center + int(window_size / 2)
        if window_end >= n_vols:
            window_end = n_vols - 1
        window_bold = bold[:, window_start:window_end+1]
        window_center += window_step
        # discard the window if more than `outlier_threshold`
        # (default: 50%) of its volumes are motion outliers
        if (window_bold.sum(axis=0) == 0).mean() > outlier_threshold:
            continue
        window_fc = np.corrcoef(window_bold)
        if exc_interhemispheric:
            # set interhemispheric connections to NaN
            rh_idx = nodes // 2
            window_fc[:rh_idx, rh_idx:] = np.NaN
            window_fc[rh_idx:, :rh_idx] = np.NaN
        window_fcs.append(window_fc[:, :, np.newaxis])
        # calculate lower triangular part of the window FC
        # and drop NaNs (interhemispheric connections)
        # to be used for FCD calculation
        window_fc_tril = window_fcs[-1][np.tril_indices(nodes, -1)]
        window_fc_tril = window_fc_tril[~np.isnan(window_fc_tril), np.newaxis]
        window_fc_trils.append(window_fc_tril)
    # calculate FCD matrix
    window_fc_trils = np.concatenate(window_fc_trils, axis=1)
    fcd_matrix = np.corrcoef(window_fc_trils.T)
    if return_tril:
        fcd_matrix = fcd_matrix[np.tril_indices(fcd_matrix.shape[0], -1)]
    # concatenate window FCs
    window_fcs = np.concatenate(window_fcs, axis=2)
    if return_dfc:
        return fcd_matrix, window_fcs
    else:
        return fcd_matrix