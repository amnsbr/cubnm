__global__ void bold_stats(
    u_real **mean_bold, u_real **ssd_bold,
    u_real **BOLD, int N_SIMS, int nodes,
    int bold_len, int corr_len, int n_vols_remove);

__global__ void window_bold_stats(
    u_real **BOLD, int N_SIMS, int nodes,
    int n_windows, int window_size_1, int *window_starts, int *window_ends,
    u_real **windows_mean_bold, u_real **windows_ssd_bold);

__global__ void fc(u_real **fc_trils, u_real **windows_fc_trils,
    u_real **BOLD, int N_SIMS, int nodes, int n_pairs, int *pairs_i,
    int *pairs_j, int bold_len, int n_vols_remove, 
    int corr_len, u_real **mean_bold, u_real **ssd_bold, 
    int n_windows, int window_size_1, u_real **windows_mean_bold, u_real **windows_ssd_bold,
    int *window_starts, int *window_ends,
    int maxThreadsPerBlock);

__global__ void window_fc_stats(
    u_real **windows_mean_fc, u_real **windows_ssd_fc,
    u_real **L_windows_mean_fc, u_real **L_windows_ssd_fc,
    u_real **R_windows_mean_fc, u_real **R_windows_ssd_fc,
    u_real **windows_fc_trils, int N_SIMS, int n_windows, int n_pairs,
    bool save_hemis, int n_pairs_hemi);

__global__ void fcd(
    u_real **fcd_trils, u_real **L_fcd_trils, u_real **R_fcd_trils,
    u_real **windows_fc_trils,
    u_real **windows_mean_fc, u_real **windows_ssd_fc,
    u_real **L_windows_mean_fc, u_real **L_windows_ssd_fc,
    u_real **R_windows_mean_fc, u_real **R_windows_ssd_fc,
    int N_SIMS, int n_pairs, int n_windows, int n_window_pairs, 
    int *window_pairs_i, int *window_pairs_j, int maxThreadsPerBlock,
    bool save_hemis, int n_pairs_hemi);