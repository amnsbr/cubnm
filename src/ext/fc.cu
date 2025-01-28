#include "cubnm/includes.cuh"
#include "cubnm/defines.h"
#include "cubnm/fc.cuh"

__global__ void bold_stats(
        u_real **mean_bold, u_real **ssd_bold,
        u_real **BOLD, int N_SIMS, int nodes,
        int bold_len, int corr_len, int n_vols_remove,
        bool co_launch
    ) {
    // get simulation and node indices
    int sim_idx;
    int j;
    if (co_launch) {
        // in co_launch mode, sim_idx is the second index of the grid
        // and j is determined based on grid and block first indices
        // Note: co_launch refers to how the simulation kernel is launched
        // This kernel is launched normally either way
        sim_idx = blockIdx.y;
        j = blockIdx.x * blockDim.x + threadIdx.x;
    } else {
        // in normal mode, sim_idx is the first index of the grid
        // and j is the first index of the block
        sim_idx = blockIdx.x;
        j = threadIdx.x;
    }
    // safe-guard against out-of-bound indices
    if (sim_idx >= N_SIMS) return;
    if (j >= nodes) return;

    // mean
    u_real _mean_bold = 0;
    int vol;
    for (vol=n_vols_remove; vol<bold_len; vol++) {
        _mean_bold += BOLD[sim_idx][vol*nodes+j];
    }
    _mean_bold /= corr_len;
    // ssd
    u_real _ssd_bold = 0;
    for (vol=n_vols_remove; vol<bold_len; vol++) {
        _ssd_bold += POW(BOLD[sim_idx][vol*nodes+j] - _mean_bold, 2);
    }
    // save to memory
    mean_bold[sim_idx][j] = _mean_bold;
    ssd_bold[sim_idx][j] = SQRT(_ssd_bold);
}

__global__ void window_bold_stats(
    u_real **BOLD, int N_SIMS, int nodes,
    int n_windows, int window_size_1, int *window_starts, int *window_ends,
    u_real **windows_mean_bold, u_real **windows_ssd_bold) {
        // get simulation index
        int sim_idx = blockIdx.x;
        if (sim_idx >= N_SIMS) return;
        // get window index
        int w = blockIdx.y;
        if (w >= n_windows) return;
        // get node index
        int j = threadIdx.x;
        if (j >= nodes) return;
        // calculate mean of window
        u_real _mean_bold = 0;
        int vol;
        for (vol=window_starts[w]; vol<=window_ends[w]; vol++) {
            _mean_bold += BOLD[sim_idx][vol*nodes+j];
        }
        _mean_bold /= window_size_1;
        // calculate sd of window
        u_real _ssd_bold = 0;
        for (vol=window_starts[w]; vol<=window_ends[w]; vol++) {
            _ssd_bold += POW(BOLD[sim_idx][vol*nodes+j] - _mean_bold, 2);
        }
        // save to memory
        windows_mean_bold[sim_idx][w*nodes+j] = _mean_bold;
        windows_ssd_bold[sim_idx][w*nodes+j] = SQRT(_ssd_bold);
}

__global__ void fc(u_real **fc_trils, u_real **windows_fc_trils,
    u_real **BOLD, int N_SIMS, int nodes, int n_pairs, int *pairs_i,
    int *pairs_j, int bold_len, int n_vols_remove, 
    int corr_len, u_real **mean_bold, u_real **ssd_bold, 
    int n_windows, int window_size_1, u_real **windows_mean_bold, u_real **windows_ssd_bold,
    int *window_starts, int *window_ends,
    int maxThreadsPerBlock) {
        // get simulation index
        int sim_idx = blockIdx.x;
        if (sim_idx >= N_SIMS) return;
        // get pair index
        int pair_idx = threadIdx.x + (maxThreadsPerBlock * blockIdx.y);
        if (pair_idx >= n_pairs) return;
        int i = pairs_i[pair_idx];
        int j = pairs_j[pair_idx];
        // get window index
        int w = blockIdx.z - 1; // -1 indicates total FC
        if (w >= n_windows) return;
        int vol_start, vol_end;
        u_real _mean_bold_i, _mean_bold_j, _ssd_bold_i, _ssd_bold_j;
        if (w == -1) {
            vol_start = n_vols_remove;
            vol_end = bold_len;
            _mean_bold_i = mean_bold[sim_idx][i];
            _ssd_bold_i = ssd_bold[sim_idx][i];
            _mean_bold_j = mean_bold[sim_idx][j];
            _ssd_bold_j = ssd_bold[sim_idx][j];
        } else {
            vol_start = window_starts[w];
            vol_end = window_ends[w]+1; // +1 because end is non-inclusive
            _mean_bold_i = windows_mean_bold[sim_idx][w*nodes+i];
            _ssd_bold_i = windows_ssd_bold[sim_idx][w*nodes+i];
            _mean_bold_j = windows_mean_bold[sim_idx][w*nodes+j];
            _ssd_bold_j = windows_ssd_bold[sim_idx][w*nodes+j];
        }
        // calculate sigma(x_i * x_j)
        int vol;
        u_real cov = 0;
        for (vol=vol_start; vol<vol_end; vol++) {
            cov += (BOLD[sim_idx][vol*nodes+i] - _mean_bold_i) * (BOLD[sim_idx][vol*nodes+j] - _mean_bold_j);
        }
        // calculate corr(i, j)
        u_real corr = cov / (_ssd_bold_i * _ssd_bold_j);
        if (w == -1) {
            fc_trils[sim_idx][pair_idx] = corr;
        } else {
            windows_fc_trils[sim_idx][w*n_pairs+pair_idx] = corr;
        }
    }

__global__ void window_fc_stats(
    u_real **windows_mean_fc, u_real **windows_ssd_fc,
    u_real **L_windows_mean_fc, u_real **L_windows_ssd_fc,
    u_real **R_windows_mean_fc, u_real **R_windows_ssd_fc,
    u_real **windows_fc_trils, int N_SIMS, int n_windows, int n_pairs,
    bool save_hemis, int n_pairs_hemi) {
        // get simulation index
        int sim_idx = blockIdx.x;
        if (sim_idx >= N_SIMS) return;
        // get window index
        int w = threadIdx.x;
        if (w >= n_windows) return;
        // get hemi
        int hemi = blockIdx.z;
        if (!save_hemis) {
            if (hemi > 0) return;
        } else {
            if (hemi > 2) return;
        }
        // calculate mean fc of window
        u_real _mean_fc = 0;
        int pair_idx_start = 0;
        int pair_idx_end = n_pairs; // non-inclusive
        int pair_idx;
        int _curr_n_pairs = n_pairs;
        // for left and right specify start and end indices
        // that belong to current hemi. Note that this will work
        // regardless of exc_interhemispheric true or false
        if (hemi == 1) { // left
            pair_idx_end = n_pairs_hemi;
            _curr_n_pairs = n_pairs_hemi;
        } else if (hemi == 2) { // right
            pair_idx_start = n_pairs - n_pairs_hemi;
            _curr_n_pairs = n_pairs_hemi;
        }
        for (pair_idx=pair_idx_start; pair_idx<pair_idx_end; pair_idx++) {
            _mean_fc += windows_fc_trils[sim_idx][w*n_pairs+pair_idx];
        }
        _mean_fc /= _curr_n_pairs;
        // calculate ssd fc of window
        u_real _ssd_fc = 0;
        for (pair_idx=pair_idx_start; pair_idx<pair_idx_end; pair_idx++) {
            _ssd_fc += POW(windows_fc_trils[sim_idx][w*n_pairs+pair_idx] - _mean_fc, 2);
        }
        // save to memory
        if (hemi == 0) {
            windows_mean_fc[sim_idx][w] = _mean_fc;
            windows_ssd_fc[sim_idx][w] = SQRT(_ssd_fc);
        } else if (hemi == 1) {
            L_windows_mean_fc[sim_idx][w] = _mean_fc;
            L_windows_ssd_fc[sim_idx][w] = SQRT(_ssd_fc);
        } else if (hemi == 2) {
            R_windows_mean_fc[sim_idx][w] = _mean_fc;
            R_windows_ssd_fc[sim_idx][w] = SQRT(_ssd_fc);
        }
    }

__global__ void fcd(
    u_real **fcd_trils, u_real **L_fcd_trils, u_real **R_fcd_trils,
    u_real **windows_fc_trils,
    u_real **windows_mean_fc, u_real **windows_ssd_fc,
    u_real **L_windows_mean_fc, u_real **L_windows_ssd_fc,
    u_real **R_windows_mean_fc, u_real **R_windows_ssd_fc,
    int N_SIMS, int n_pairs, int n_windows, int n_window_pairs, 
    int *window_pairs_i, int *window_pairs_j, int maxThreadsPerBlock,
    bool save_hemis, int n_pairs_hemi) {
        // get simulation index
        int sim_idx = blockIdx.x;
        if (sim_idx >= N_SIMS) return;
        // get window pair index
        int window_pair_idx = threadIdx.x + (maxThreadsPerBlock * blockIdx.y);
        if (window_pair_idx >= n_window_pairs) return;
        int w_i = window_pairs_i[window_pair_idx];
        int w_j = window_pairs_j[window_pair_idx];
        // get hemi
        int hemi = blockIdx.z;
        if (!save_hemis) {
            if (hemi > 0) return;
        } else {
            if (hemi > 2) return;
        }
        // calculate cov
        int pair_idx;
        u_real cov = 0;
        // pair_idx_start = 0;
        // pair_idx_end = n_pairs; // non-inclusive
        // if (hemi == 1) { // left
        //     pair_idx_end = n_pairs_hemi;
        // } else if (hemi == 2) { // right
        //     pair_idx_start = n_pairs - n_pairs_hemi;
        // }
        if (hemi == 0) {
            for (pair_idx=0; pair_idx<n_pairs; pair_idx++) {
                cov += 
                    (windows_fc_trils[sim_idx][w_i*n_pairs+pair_idx] - windows_mean_fc[sim_idx][w_i]) 
                    * (windows_fc_trils[sim_idx][w_j*n_pairs+pair_idx] - windows_mean_fc[sim_idx][w_j]);
            }
            fcd_trils[sim_idx][window_pair_idx] = cov / (windows_ssd_fc[sim_idx][w_i] * windows_ssd_fc[sim_idx][w_j]);
        } else if (hemi == 1) {
            for (pair_idx=0; pair_idx<n_pairs_hemi; pair_idx++) {
                cov += 
                    (windows_fc_trils[sim_idx][w_i*n_pairs+pair_idx] - L_windows_mean_fc[sim_idx][w_i]) 
                    * (windows_fc_trils[sim_idx][w_j*n_pairs+pair_idx] - L_windows_mean_fc[sim_idx][w_j]);
            }
            L_fcd_trils[sim_idx][window_pair_idx] = cov / (L_windows_ssd_fc[sim_idx][w_i] * L_windows_ssd_fc[sim_idx][w_j]);
        } else if (hemi == 2) {
            for (pair_idx=n_pairs-n_pairs_hemi; pair_idx<n_pairs; pair_idx++) {
                cov += 
                    (windows_fc_trils[sim_idx][w_i*n_pairs+pair_idx] - R_windows_mean_fc[sim_idx][w_i]) 
                    * (windows_fc_trils[sim_idx][w_j*n_pairs+pair_idx] - R_windows_mean_fc[sim_idx][w_j]);
            }
            R_fcd_trils[sim_idx][window_pair_idx] = cov / (R_windows_ssd_fc[sim_idx][w_i] * R_windows_ssd_fc[sim_idx][w_j]);
        }
    }

void run_fc_calculation_gpu(
    double *fc_trils_out, double *BOLD, 
    int N_BOLD, int nodes, int bold_len, int n_pairs, 
    bool exc_interhemispheric, int rh_idx
) {
    /*
        Standalone function to run the FC calculation on the GPU
        for user-provided (empirical) BOLD data.
    */
    /* Initializations */
    // get device properties without printing device name
    cudaDeviceProp prop = get_device_prop(0);
    // copy BOLD data from np array on host to managed memory
    // while converting it from 3D to 2D array
    u_real **d_BOLD;
    int bold_size = nodes * bold_len;
    CUDA_CHECK_RETURN(cudaMallocManaged(&d_BOLD, N_BOLD * sizeof(double *)));
    for (int bold_idx=0; bold_idx<N_BOLD; bold_idx++) {
        CUDA_CHECK_RETURN(cudaMallocManaged(&(d_BOLD[bold_idx]), bold_size * sizeof(double)));
        CUDA_CHECK_RETURN(cudaMemcpy(
            d_BOLD[bold_idx],
            BOLD+(bold_idx*bold_size),
            bold_size * sizeof(double), 
            cudaMemcpyHostToDevice
            ));
    }
    // specify n_vols_remove as 0
    int n_vols_remove = 0;
    int corr_len = bold_len - n_vols_remove;
    if (corr_len < 2) {
        std::cerr << "Number of BOLD volumes is too low for FC calculations" << std::endl;
        exit(1);
    }
    // create a mapping between pair_idx and i and j
    int curr_idx = 0;
    int *pairs_i, *pairs_j;
    CUDA_CHECK_RETURN(cudaMallocManaged(&pairs_i, sizeof(int) * n_pairs));
    CUDA_CHECK_RETURN(cudaMallocManaged(&pairs_j, sizeof(int) * n_pairs));
    for (int i=0; i < nodes; i++) {
        for (int j=0; j < nodes; j++) {
            if (i > j) {
                if (exc_interhemispheric) {
                    // skip if each node belongs to a different hemisphere
                    if ((i < rh_idx) ^ (j < rh_idx)) {
                        continue;
                    }
                }
                pairs_i[curr_idx] = i;
                pairs_j[curr_idx] = j;
                curr_idx++;
            }
        }
    }
    // allocate memory for mean_bold, ssd_bold and fc_trils
    double **mean_bold, **ssd_bold, **fc_trils;
    CUDA_CHECK_RETURN(cudaMallocManaged(&mean_bold, sizeof(double*) * N_BOLD));
    CUDA_CHECK_RETURN(cudaMallocManaged(&ssd_bold, sizeof(double*) * N_BOLD));
    CUDA_CHECK_RETURN(cudaMallocManaged(&fc_trils, sizeof(double*) * N_BOLD));
    for (int bold_idx=0; bold_idx<N_BOLD; bold_idx++) {
        CUDA_CHECK_RETURN(cudaMallocManaged(&mean_bold[bold_idx], sizeof(double) * nodes));
        CUDA_CHECK_RETURN(cudaMallocManaged(&ssd_bold[bold_idx], sizeof(double) * nodes));
        CUDA_CHECK_RETURN(cudaMallocManaged(&fc_trils[bold_idx], sizeof(double) * n_pairs));
    }
    // Note: since FC and FCD calculations are entangled in some
    // kernels (e.g. `fc` kernel calculates both static and dynamic FCs)
    // skipping FCD calculation can be done by setting n_windows to 0
    // in which case window fc and fcd calculation kernels are called
    // but don't do anything
    int n_windows = 0;
    int n_window_pairs = 0;
    /* Running kernels */
    // set co_launch to true so that many nodes can be supported
    bool co_launch = true;
    // run bold_stats kernel
    // setting grid and block dimensions similar to the
    // co_launch mode when running simulations
    dim3 numBlocks;
    dim3 threadsPerBlock;
    threadsPerBlock.x = 256;
    numBlocks.x = ceil((float)nodes / (float)threadsPerBlock.x);
    numBlocks.y = N_BOLD;
    bold_stats<<<numBlocks, threadsPerBlock>>>(
        mean_bold, ssd_bold,
        d_BOLD, N_BOLD, nodes,
        bold_len, corr_len, n_vols_remove,
        co_launch);
    CUDA_CHECK_LAST_ERROR();
    CUDA_CHECK_RETURN(cudaDeviceSynchronize());
    // run FC calculation kernel
    int maxThreadsPerBlock = prop.maxThreadsPerBlock;
    numBlocks.x = N_BOLD;
    numBlocks.y = ceil((float)n_pairs / (float)maxThreadsPerBlock);
    numBlocks.z = n_windows + 1; // +1 for total FC
    if (numBlocks.y > prop.maxGridSize[1]) {
        std::cerr << "Error: Number of pairs " << n_pairs 
            << " exceeds the capacity of the device for FC calculation" << std::endl;
        exit(1);
    }
    if (prop.maxThreadsPerBlock!=prop.maxThreadsDim[0]) {
        std::cerr << "Error: Code not implemented for GPUs in which maxThreadsPerBlock!=maxThreadsDim[0]" << std::endl;
        exit(1);
    }
    threadsPerBlock.x = maxThreadsPerBlock;
    threadsPerBlock.y = 1;
    threadsPerBlock.z = 1;
    // launch kernel, passing FCD-related arrays as NULL
    fc<<<numBlocks, threadsPerBlock>>>(
        fc_trils, NULL, d_BOLD, 
        N_BOLD, nodes, n_pairs, 
        pairs_i, pairs_j,
        bold_len, n_vols_remove, 
        corr_len, mean_bold, ssd_bold,
        n_windows, 1, NULL, NULL,
        NULL, NULL,
        maxThreadsPerBlock
    );
    CUDA_CHECK_LAST_ERROR();
    CUDA_CHECK_RETURN(cudaDeviceSynchronize());

    /* Copy results to host np array */
    for (int bold_idx=0; bold_idx<N_BOLD; bold_idx++) {
        CUDA_CHECK_RETURN(cudaMemcpy(
            fc_trils_out+(bold_idx*n_pairs), 
            fc_trils[bold_idx], 
            n_pairs * sizeof(double), 
            cudaMemcpyDeviceToHost
        ));
    }

    /* Free memory */
    // free mean_bold, ssd_bold and fc_trils
    for (int bold_idx=0; bold_idx<N_BOLD; bold_idx++) {
        CUDA_CHECK_RETURN(cudaFree(fc_trils[bold_idx]));
        CUDA_CHECK_RETURN(cudaFree(ssd_bold[bold_idx]));
        CUDA_CHECK_RETURN(cudaFree(mean_bold[bold_idx]));
    }
    // free pairs_i and pairs_j
    CUDA_CHECK_RETURN(cudaFree(pairs_i));
    CUDA_CHECK_RETURN(cudaFree(pairs_j));
    // free BOLD copy
    for (int bold_idx=0; bold_idx<N_BOLD; bold_idx++) {
        CUDA_CHECK_RETURN(cudaFree(d_BOLD[bold_idx]));
    }
    CUDA_CHECK_RETURN(cudaFree(d_BOLD));
}