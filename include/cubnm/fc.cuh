__global__ void bold_stats(
    double **mean_bold, double **ssd_bold,
    double **BOLD, int N_SIMS, int nodes,
    int bold_len, int corr_len, int n_vols_remove,
    bool co_launch);

__global__ void window_bold_stats(
    double **BOLD, int N_SIMS, int nodes,
    int n_windows, int window_size_1, int *window_starts, int *window_ends,
    double **windows_mean_bold, double **windows_ssd_bold);

__global__ void fc(double **fc_trils, double **windows_fc_trils,
    double **BOLD, int N_SIMS, int nodes, int n_pairs, int *pairs_i,
    int *pairs_j, int bold_len, int n_vols_remove, 
    int corr_len, double **mean_bold, double **ssd_bold, 
    int n_windows, int window_size_1, double **windows_mean_bold, double **windows_ssd_bold,
    int *window_starts, int *window_ends,
    int maxThreadsPerBlock);

__global__ void window_fc_stats(
    double **windows_mean_fc, double **windows_ssd_fc,
    double **L_windows_mean_fc, double **L_windows_ssd_fc,
    double **R_windows_mean_fc, double **R_windows_ssd_fc,
    double **windows_fc_trils, int N_SIMS, int n_windows, int n_pairs,
    bool save_hemis, int n_pairs_hemi);

__global__ void fcd(
    double **fcd_trils, double **L_fcd_trils, double **R_fcd_trils,
    double **windows_fc_trils,
    double **windows_mean_fc, double **windows_ssd_fc,
    double **L_windows_mean_fc, double **L_windows_ssd_fc,
    double **R_windows_mean_fc, double **R_windows_ssd_fc,
    int N_SIMS, int n_pairs, int n_windows, int n_window_pairs, 
    int *window_pairs_i, int *window_pairs_j, int maxThreadsPerBlock,
    bool save_hemis, int n_pairs_hemi);