#ifndef UTILS_HPP
#define UTILS_HPP

extern int get_fc_n_pairs(int nodes, bool exc_interhemispheric);

extern int get_dfc_windows(
        std::vector<int> *window_starts_p, std::vector<int> *window_ends_p,
        int corr_len, int bold_len, int n_vols_remove,
        int window_step, int window_size, bool drop_edges);

extern int get_dfc_windows(
        int **window_starts_p, int **window_ends_p,
        int corr_len, int bold_len, int n_vols_remove,
        int window_step, int window_size, bool drop_edges);

extern void get_shuffled_nodes_ts(
        int **shuffled_nodes_p, int **shuffled_ts_p,
        int nodes, int noise_time_steps, int noise_repeats,
        std::mt19937 *rand_gen_p);

#endif