gsl_vector * BaseModel::calculate_fc_tril(gsl_matrix * bold) {
    /*
     Given empirical/simulated bold (n_vols x nodes) returns
     the lower triangle of the FC
    */
    int nodes = bold->size2;
    int n_vols = bold->size1;
    int rh_idx = bold->size2 / 2; // assumes symmetric number of parcels and L->R order
    int i, j;
    double corr;
    gsl_vector * FC_tril = gsl_vector_alloc(this->n_pairs);
    int curr_idx = 0;
    for (i = 0; i<(bold->size2); i++) {
        for (j = 0; j<(bold->size2); j++) {
            if (i > j) {
                if (this->base_conf.exc_interhemispheric) {
                    // skip if each node belongs to a different hemisphere
                    if ((i < rh_idx) ^ (j < rh_idx)) {
                        continue;
                    }
                }
                gsl_vector_view ts_i = gsl_matrix_column(bold, i);
                gsl_vector_view ts_j = gsl_matrix_column(bold, j);
                corr = gsl_stats_correlation(
                    ts_i.vector.data, ts_i.vector.stride,
                    ts_j.vector.data, ts_j.vector.stride,
                    (bold->size1)
                );
                if (std::isnan(corr)) {
                    std::cerr << "FC[" << i << "," << j << "] is NaN" << std::endl;
                    return NULL;
                }
                gsl_vector_set(FC_tril, curr_idx, corr);
                curr_idx ++;
            }
        }
    }
    return FC_tril;
}

gsl_vector * BaseModel::calculate_fcd_tril(gsl_matrix * bold, int * window_starts, int * window_ends) {
    /*
     Calculates the functional connectivity dynamics matrix (lower triangle)
     given BOLD, step and window size. Note that the actual window size is +1 higher.
     The FCD matrix shows similarity of FC patterns between the windows.
    */
    int n_vols = bold->size1;
    int nodes = bold->size2;
    gsl_vector * FCD_tril = gsl_vector_alloc(this->n_window_pairs);
    gsl_matrix * window_FC_trils = gsl_matrix_alloc(this->n_pairs, this->n_windows);
    if (this->n_windows < 10) {
        std::cout << "Warning: Too few FC windows: " << this->n_windows << std::endl;
    }
    // calculate dynamic FC
    for (int i=0; i<this->n_windows; i++) {
        gsl_matrix_view bold_window =  gsl_matrix_submatrix(
            bold, 
            window_starts[i], 0, 
            window_ends[i]-window_starts[i]+1, this->nodes);
        gsl_vector * window_FC_tril = this->calculate_fc_tril(&bold_window.matrix);
        if (window_FC_tril==NULL) {
            std::cerr << "Error: Dynamic FC calculation failed" << std::endl;
            return NULL;
        }
        gsl_matrix_set_col(window_FC_trils, i, window_FC_tril);
        gsl_vector_free(window_FC_tril);
    }
    // calculate the FCD matrix (lower triangle)
    int window_i, window_j;
    double corr;
    int curr_idx = 0;
    for (window_i=0; window_i<this->n_windows; window_i++) {
        for (window_j=0; window_j<this->n_windows; window_j++) {
            if (window_i > window_j) {
                gsl_vector_view FC_i = gsl_matrix_column(window_FC_trils, window_i);
                gsl_vector_view FC_j = gsl_matrix_column(window_FC_trils, window_j);
                corr = gsl_stats_correlation(
                    FC_i.vector.data, FC_i.vector.stride,
                    FC_j.vector.data, FC_j.vector.stride,
                    this->n_pairs
                );
                if (std::isnan(corr)) {
                    std::cerr << "Error: FCD[" << window_i << "," << window_j << "] is NaN" << std::endl;
                    return NULL;
                }
                gsl_vector_set(FCD_tril, curr_idx, corr);
                curr_idx ++;
            }
        }
    }
    return FCD_tril;
}