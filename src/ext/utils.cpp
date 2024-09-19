/*
Includes utility functions used by both CPU and GPU
*/
#include "cubnm/utils.hpp"

int get_fc_n_pairs(int nodes, bool exc_interhemispheric) {
    int n_pairs = ((nodes) * (nodes - 1)) / 2;
    int rh_idx;
    if (exc_interhemispheric) {
        if ((nodes % 2) != 0) {
            std::cerr << "Error: exc_interhemispheric is set but number of nodes is not even" << std::endl;
            exit(1);
        }
        rh_idx = nodes / 2; // assumes symmetric number of parcels and L->R order
        n_pairs -= pow(rh_idx, 2); // exc the middle square
    }
    return n_pairs;
}

int get_dfc_windows(
        std::vector<int> *window_starts_p, std::vector<int> *window_ends_p,
        int corr_len, int bold_len, int n_vols_remove,
        int window_step, int window_size, bool drop_edges
    ) {
    int n_windows = 0;
    int first_center, last_center, window_center, window_start, window_end;
    if (drop_edges) {
        first_center = window_size / 2;
        last_center = corr_len - 1 - (window_size / 2);
    } else {
        first_center = 0;
        last_center = corr_len - 1;
    }
    first_center += n_vols_remove;
    last_center += n_vols_remove;
    window_center = first_center;
    while (window_center <= last_center) {
        window_start = window_center - (window_size/2);
        if (window_start < 0)
            window_start = 0;
        window_end = window_center + (window_size/2);
        if (window_end >= bold_len)
            window_end = bold_len-1;
        (*window_starts_p).push_back(window_start);
        (*window_ends_p).push_back(window_end);
        window_center += window_step;
        n_windows ++;
    }
    if (n_windows == 0) {
        std::cerr << "Error: Number of dynamic FC windows is 0" << std::endl;
        exit(1);
    }
    return n_windows;
}

int get_dfc_windows(
        int **window_starts_p, int **window_ends_p,
        int corr_len, int bold_len, int n_vols_remove,
        int window_step, int window_size, bool drop_edges
    ) {
    std::vector<int> _window_starts, _window_ends;
    int n_windows = get_dfc_windows(
        &_window_starts, &_window_ends, 
        corr_len, bold_len, n_vols_remove,
        window_step, window_size, drop_edges);
    // copy the vectors to arrays
    *window_starts_p = (int *)malloc(sizeof(int) * n_windows);
    *window_ends_p = (int *)malloc(sizeof(int) * n_windows);
    std::copy(_window_starts.begin(), _window_starts.end(), *window_starts_p);
    std::copy(_window_ends.begin(), _window_ends.end(), *window_ends_p);
    return n_windows;
}

void get_shuffled_nodes_ts(
        int **shuffled_nodes_p, int **shuffled_ts_p,
        int nodes, int noise_bw_it, int noise_repeats,
        std::mt19937 *rand_gen_p
    ) {
    // create shuffled nodes indices for each repeat of the 
    // precalculaed noise  (row shuffling)
    std::vector<int> node_indices(nodes);
    std::iota(node_indices.begin(), node_indices.end(), 0);
    for (int i = 0; i < noise_repeats; i++) {
        std::shuffle(node_indices.begin(), node_indices.end(), *rand_gen_p);
        std::copy(node_indices.begin(), node_indices.end(), *shuffled_nodes_p+(i*nodes));
    }
    // similarly create shuffled time point indices (msec)
    // for each repeat (column shuffling)
    std::vector<int> ts_indices(noise_bw_it);
    std::iota(ts_indices.begin(), ts_indices.end(), 0);
    for (int i = 0; i < noise_repeats; i++) {
        std::shuffle(ts_indices.begin(), ts_indices.end(), *rand_gen_p);
        std::copy(ts_indices.begin(), ts_indices.end(), *shuffled_ts_p+(i*noise_bw_it));
    }
}