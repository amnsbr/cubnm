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
