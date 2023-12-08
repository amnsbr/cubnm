/*
Includes utility functions used by both CPU and GPU
*/
#ifndef UTILS_CPP
#define UTILS_CPP

int get_fc_n_pairs(int nodes) {
    int n_pairs = ((nodes) * (nodes - 1)) / 2;
    int rh_idx;
    if (conf.exc_interhemispheric) {
        assert((nodes % 2) == 0);
        rh_idx = nodes / 2; // assumes symmetric number of parcels and L->R order
        n_pairs -= pow(rh_idx, 2); // exc the middle square
    }
    return n_pairs;
}

int get_dfc_windows(
        std::vector<int> *window_starts_p, std::vector<int> *window_ends_p,
        int corr_len, int output_ts, int n_vols_remove,
        int window_step, int window_size
    ) {
    int n_windows = 0;
    int first_center, last_center, window_center, window_start, window_end;
    if (conf.drop_edges) {
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
        if (window_end >= output_ts)
            window_end = output_ts-1;
        (*window_starts_p).push_back(window_start);
        (*window_ends_p).push_back(window_end);
        window_center += window_step;
        n_windows ++;
    }
    if (n_windows == 0) {
        printf("Error: Number of dynamic FC windows is 0\n");
        exit(1);
    }
    return n_windows;
}

int get_dfc_windows(
        int **window_starts_p, int **window_ends_p,
        int corr_len, int output_ts, int n_vols_remove,
        int window_step, int window_size
    ) {
    std::vector<int> _window_starts, _window_ends;
    int n_windows = get_dfc_windows(
        &_window_starts, &_window_ends, 
        corr_len, output_ts, n_vols_remove,
        window_step, window_size);
    // copy the vectors to arrays
    *window_starts_p = (int *)malloc(sizeof(int) * n_windows);
    *window_ends_p = (int *)malloc(sizeof(int) * n_windows);
    std::copy(_window_starts.begin(), _window_starts.end(), *window_starts_p);
    std::copy(_window_ends.begin(), _window_ends.end(), *window_ends_p);
    return n_windows;
}

#endif