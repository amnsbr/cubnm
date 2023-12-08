#ifndef UTILS_HPP
#define UTILS_HPP

extern int get_fc_n_pairs(int nodes);

extern int get_dfc_windows(
        std::vector<int> *window_starts_p, std::vector<int> *window_ends_p,
        int corr_len, int output_ts, int n_vols_remove,
        int window_step, int window_size);

extern int get_dfc_windows(
        int **window_starts_p, int **window_ends_p,
        int corr_len, int output_ts, int n_vols_remove,
        int window_step, int window_size);

#endif