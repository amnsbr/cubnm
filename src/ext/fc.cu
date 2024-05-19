#include "cubnm/includes.cuh"
#include "cubnm/defines.h"
#include "cubnm/fc.cuh"
__global__ void bold_stats(
    u_real **mean_bold, u_real **ssd_bold,
    u_real **BOLD, int N_SIMS, int nodes,
    int bold_len, int corr_len, int n_vols_remove) {
    // TODO: consider combining this with window_bold_stats
    // get simulation index
    int sim_idx = blockIdx.x;
    if (sim_idx >= N_SIMS) return;
    // get node index
    int j = threadIdx.x;
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