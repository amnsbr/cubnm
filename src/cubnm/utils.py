"""
Utility functions
"""
import numpy as np
import scipy
import subprocess
import json
import gc
from cubnm._setup_opts import gpu_model_flag
try:
    import cupy as cp
    from numba import cuda
    has_cupy = True
except ImportError:
    cp = None
    cuda = None
    has_cupy = False

if has_cupy:
    @cuda.jit
    def cdf_2d_2d_right(A_sorted, x_values, out):
        """
        CUDA kernel for calculating CDF of matching rows in 2D arrays.
        For each row i, performs searchsorted (right-sided) of 
        x_values[i] in A_sorted[i] divided by the length of A_sorted[i]
        resulting in empirical CDF of A_sorted[i] at x_values[i].
        Each row of A_sorted[i] must be sorted in ascending order.

        Parameters
        ----------
        A_sorted: :obj:`cp.ndarray`
            2D array of sorted values. Shape: (n_rows, m)
        x_values: :obj:`cp.ndarray`
            2D array of values to calculate CDF for. Shape: (n_rows, k)
        out: :obj:`cp.ndarray`
            2D array to store the CDF values. Shape: (n_rows, k)
        """
        # get current row and x_value index (col)
        row, col = cuda.grid(2)
        # get dimensions
        n_rows, m = A_sorted.shape
        _, k = x_values.shape
        # calculate CDF of A_sorted[row]
        # at x_values[row, col]
        if row < n_rows and col < k:
            x = x_values[row, col]
            left = 0
            right = m
            while left < right:
                mid = (left + right) // 2
                if x < A_sorted[row, mid]:
                    right = mid
                else:
                    left = mid + 1
            out[row, col] = left / m
else:
    def cdf_2d_2d_right(*args, **kwargs):
        raise RuntimeError(
            "CuPy is not available. Install `cubnm[cupy-cuda11x]` or `cubnm[cupy-cuda12x]`."
        )

def avail_gpus():
    """
    Get the number of available GPUs

    Returns
    -------
    :obj:`int`
        Number of available GPUs
    """
    gpu_counts = {'nvidia': 0, 'rocm': 0}
    # nvidia
    try:
        output = subprocess.check_output(['nvidia-smi', '-L'])
        output = output.decode('utf-8').strip()
        gpu_list = output.split('\n')
        gpu_counts['nvidia'] = len(gpu_list)
    except subprocess.CalledProcessError:
        gpu_counts['nvidia'] = 0
    except FileNotFoundError:
        gpu_counts['nvidia'] = 0
    # rocm
    try:
        output = subprocess.check_output(['rocm-smi', '-i', '--json'])
        output = output.decode('utf-8').strip()
        gpu_dict = json.loads(output)
        gpu_counts['rocm'] = len(gpu_dict)
    except subprocess.CalledProcessError:
        gpu_counts['rocm'] = 0
    except FileNotFoundError:
        gpu_counts['rocm'] = 0
    # warn if there is mismatch between the type of
    # gpu for which the code is complied vs available gpus
    if (
        ((gpu_model_flag == 'nvidia') and (gpu_counts['rocm'] > 0)) |
        ((gpu_model_flag == 'rocm') and (gpu_counts['nvidia'] > 0))
    ):
        print(
            f"Warning: Toolbox compiled for {gpu_model_flag}"
            f" but {(set(gpu_counts.keys()) - set([gpu_model_flag]))}"
            " GPUs are available. Reinstall for the correct GPU model."
        )
    # but only return the number of gpus for which
    # the code was compiled
    return gpu_counts[gpu_model_flag]

def is_jupyter():
    """
    This function checks if the current environment is a Jupyter notebook.

    Returns
    -------
    :obj:`bool`
    """
    try:
        from IPython import get_ipython
    except ImportError:
        return False
    if hasattr(get_ipython(), 'config'):
        if 'IPKernelApp' in get_ipython().config:  
            return True
    return False

def get_bw_params(src):
    """
    Get Balloon-Windkessel model parameters

    Parameters
    ----------
    src: {'friston2003', 'heinzle2016-3T'}
        Source of the Balloon-Windkessel model parameters.
        - ``'friston2003'``: Friston et al. 2003
        - ``'heinzle2016-3T'``: Heinzle et al. 2016, 3T parameters

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
    exc_interhemispheric: :obj:`bool`
        exclude interhemispheric connections
    return_tril: :obj:`bool`
        return only the lower triangular part of the FCD matrix

    Returns
    -------
    :obj:`np.ndarray`
        FC dynamics matrix. 
        Shape: (nodes, nodes) or (n_node_pairs,)
        if ``return_tril`` is ``True``
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
    window_size: :obj:`int`
        dynamic FC window size (in TR)
        Must be even. The actual window size is +1 (including center).
    window_step: :obj:`int`
        dynamic FC window step (in TR)
    drop_edges: :obj:`bool`
        drop edge windows which have less than window_size volumes
    outlier_threshold: :obj:`float`
        threshold for the proportion of motion outliers in a window
        that would lead to discarding the window
    exc_interhemispheric: :obj:`bool`
        exclude interhemispheric connections
    return_tril: :obj:`bool`
        return only the lower triangular part of the FCD matrix
    return_dfc: :obj:`bool`
        return dynamic FCs as well

    Returns
    -------
    :obj:`np.ndarray`
        FC dynamics matrix. 
        Shape: (n_windows, n_windows) or (n_window_pairs,)
        if ``return_tril`` is ``True``
    :obj:`np.ndarray`
        dynamic FCs. Shape: (nodes, nodes, n_windows)
        Returned only if ``return_dfc`` is ``True``
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

def fcd_ks_device(sim_fcd_trils, emp_fcd_tril, usable_mem=None):
    """
    Calculates Kolmogorov-Smirnov distance of provided simulated FCDs
    to the empirical FCD on GPU. The calculation will be done in batches
    depending on the available GPU memory (`usable_mem`) and the memory
    required for each simulation.

    Parameters
    ----------
    sim_fcd_trils: :obj:`cp.ndarray`
        2D array of simulated FCDs. Shape: (n_simulations, m1)
    emp_fcd_tril: :obj:`cp.ndarray`
        1D array of empirical FCD. Shape: (m2)
    usable_mem: :obj:`int`
        available GPU memory in bytes
        If not provided, 80% of the free GPU memory will be used.
    
    Returns
    -------
    :obj:`np.ndarray`
        Kolmogorov-Smirnov distances. Shape: (n_simulations,)
    """
    if not has_cupy:
        raise RuntimeError(
            "CuPy is not available. Install `cubnm[cupy-cuda11x]` or `cubnm[cupy-cuda12x]`."
        )

    if usable_mem is None:
        free_mem, _ = cp.cuda.runtime.memGetInfo()
        usable_mem = int(free_mem * 0.8)
    n_simulations, m1 = sim_fcd_trils.shape
    m2 = emp_fcd_tril.shape[0]

    # sort emp_fcd_tril and copy to device
    emp_sorted = cp.asarray(emp_fcd_tril, dtype=cp.float64)
    emp_sorted.sort()

    # memory estimation to determine batch size
    mem_emp = emp_sorted.nbytes # invariant to number of simulations
    mem_per_sim = (
        # n_cols * n_bytes * n_variables
        m1 * 8 * 1 + # sim_sorted
        m2 * 8 * 1 + # emp_tiled
        (m1 + m2) * 8 * 3 + # merged_sorted, cdf_sim (also cdf_diff, abs_cdf_diff), cdf_emp
        1 * 8 * 1 # max
    )
    mem_batches = usable_mem - mem_emp
    sims_per_batch = mem_batches // mem_per_sim
    ## this will rarely happen, but just in case:
    assert sims_per_batch > 0, \
        "Not enough GPU memory available for calculating"\
        " FCD KS of even a single simulation."
    n_batches = int(np.ceil(n_simulations / sims_per_batch))
    if n_batches > 1:
        print(f"Calculating FCD KS in {n_batches} batches to reduce GPU memory usage.")
    all_ks = []
    # allocate memory to arrays that are reused across batches
    sim_sorted = cp.zeros((sims_per_batch, m1), dtype=cp.float64)
    merged_sorted = cp.zeros((sims_per_batch, m1 + m2), dtype=cp.float64)
    cdf_sim = cp.zeros((sims_per_batch, m1 + m2), dtype=cp.float64) # also will be used for cdf_diff and abs_cdf_diff
    cdf_emp = cp.zeros((sims_per_batch, m1 + m2), dtype=cp.float64)
    maxes = cp.zeros(sims_per_batch, dtype=cp.float64)
    # repeat emp_sorted for n_batch rows to compare to sim_sorted
    emp_tiled = cp.tile(emp_sorted[None, :], (sims_per_batch, 1))

    for i in range(n_batches):
        # clean up memory before each batch
        gc.collect()
        cp.get_default_memory_pool().free_all_blocks()        
        # calculate batch start and end indices + size
        start = i * sims_per_batch
        end = min((i + 1) * sims_per_batch, n_simulations)
        batch_size = end - start
        # copy sim_fcd_trils to device and sort
        sim_sorted[:batch_size] = cp.asarray(sim_fcd_trils[start:end], cp.float64)
        sim_sorted.sort(axis=1)   
        # merge sim_sorted with emp_sorted and then re-sort
        merged_sorted[:batch_size, :m1] = sim_sorted[:batch_size]
        merged_sorted[:batch_size, m1:] = emp_tiled[:batch_size]
        merged_sorted.sort(axis=1)
        # calculate grid dimensions for kernel launch
        threadsperblock = (16, 16)
        blockspergrid_x = int(np.ceil(batch_size / threadsperblock[0]))
        blockspergrid_y = int(np.ceil((m1 + m2) / threadsperblock[1]))
        blockspergrid = (blockspergrid_x, blockspergrid_y)                
        # launch cdf kernels on GPU to calculate
        # CDFs of sim and emp FCDs for current batch
        cdf_2d_2d_right[blockspergrid, threadsperblock](
            sim_sorted[:batch_size], merged_sorted[:batch_size], cdf_sim[:batch_size]
        )
        cdf_2d_2d_right[blockspergrid, threadsperblock](
            emp_tiled[:batch_size], merged_sorted[:batch_size], cdf_emp[:batch_size]
        )
        cuda.synchronize()
        # calculate KS distance for each simulation as max(abs(cdf_sim - cdf_emp))
        # note: reusing the same array (cdf_sim) for intermediate steps to reduce
        # GPU memory usage
        cp.subtract(cdf_sim[:batch_size], cdf_emp[:batch_size], out=cdf_sim[:batch_size])
        cp.abs(cdf_sim[:batch_size], out=cdf_sim[:batch_size])
        cp.max(cdf_sim[:batch_size], axis=1, out=maxes[:batch_size])
        ks = maxes[:batch_size].get()
        all_ks.append(ks)

    # clean up GPU memory
    del emp_sorted, emp_tiled, sim_sorted, merged_sorted, cdf_emp, cdf_sim, maxes
    gc.collect()
    cp.get_default_memory_pool().free_all_blocks()
    # combine batches and return
    return np.concatenate(all_ks)

def fc_corr_device(sim_fc_trils, emp_fc_tril, usable_mem=None):
    """
    Calculates Pearson correlation between simulated FCs and empirical FCs
    on GPU. The calculation will be done in batches depending on the available
    GPU memory (`usable_mem`) and the memory required for each simulation.

    Parameters
    ----------
    sim_fc_trils: :obj:`cp.ndarray`
        2D array of simulated FCs. Shape: (n_simulations, n_pairs)
    emp_fc_tril: :obj:`cp.ndarray`
        1D array of empirical FC. Shape: (n_pairs)
    usable_mem: :obj:`int`
        available GPU memory in bytes
        If not provided, 80% of the free GPU memory will be used.
    
    Returns
    -------
    :obj:`np.ndarray`
        Pearson correlation coefficients. Shape: (n_simulations,)
    """
    if not has_cupy:
        raise RuntimeError(
            "CuPy is not available. Install `cubnm[cupy-cuda11x]` or `cubnm[cupy-cuda12x]`."
        )

    if usable_mem is None:
        free_mem, _ = cp.cuda.runtime.memGetInfo()
        usable_mem = int(free_mem * 0.8)
    n_simulations, n_pairs = sim_fc_trils.shape

    # copy empirical FC to device and z-score
    # note that ddof=1 is necessary for the same
    # results as scipy.stats.pearsonr
    emp_fc = cp.asarray(emp_fc_tril[None, :])
    mean = emp_fc.mean(axis=1, keepdims=True)
    sd = emp_fc.std(axis=1, ddof=1, keepdims=True)
    # z-scoring is done in-place to reduce memory usage
    emp_fc -= mean
    emp_fc /= sd

    # memory estimation to determine batch size
    mem_emp = emp_fc.nbytes
    mem_per_sim = (
        # n_cols * n_bytes * n_variables
        n_pairs * 8 * 1 + # sim_fc
        1 * 8 * 3 # mean and sd and corr
    )
    mem_batches = usable_mem - mem_emp
    sims_per_batch = max(1, mem_batches // mem_per_sim)
    ## this will rarely happen, but just in case:
    assert sims_per_batch > 0, \
        "Not enough GPU memory available for calculating"\
        " FCD KS of even a single simulation."
    n_batches = int(np.ceil(n_simulations / sims_per_batch))
    if n_batches > 1:
        print(f"Calculating FC corr in {n_batches} batches to reduce GPU memory usage.")

    all_corr = []
    # allocate memory to arrays that are reused across batches
    sim_fc = cp.zeros((sims_per_batch, n_pairs), dtype=cp.float64)
    corr = cp.zeros((sims_per_batch, 1), dtype=cp.float64)
    mean = cp.zeros((sims_per_batch, 1), dtype=cp.float64)
    sd = cp.zeros((sims_per_batch, 1), dtype=cp.float64)
    
    for i in range(n_batches):
        # clean up memory before each batch
        gc.collect()
        cp.get_default_memory_pool().free_all_blocks()        
        # calculate batch start and end indices + size
        start = i * sims_per_batch
        end = min((i + 1) * sims_per_batch, n_simulations)
        batch_size = end - start
        # zscore sim_fc
        sim_fc[:batch_size] = cp.asarray(sim_fc_trils[start:end], cp.float64)
        sim_fc[:batch_size].mean(axis=1, keepdims=True, out=mean[:batch_size])
        sim_fc[:batch_size].std(axis=1, ddof=1, keepdims=True, out=sd[:batch_size])
        sim_fc[:batch_size] -= mean[:batch_size]
        sim_fc[:batch_size] /= sd[:batch_size]
        # matrix multiplication to calculate correlation
        # divide by n_pairs-1 to get unbiased estimate
        # (following scipy.stats.pearsonr)
        cp.matmul(sim_fc[:batch_size], emp_fc.T, out=corr[:batch_size])
        corr[:batch_size] /= n_pairs-1
        all_corr.append(corr[:batch_size].get().flatten())

    # clean up GPU memory
    del sim_fc, emp_fc, corr, mean, sd
    gc.collect()
    cp.get_default_memory_pool().free_all_blocks()
    # combine batches and return
    return np.concatenate(all_corr)