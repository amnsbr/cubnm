/*
Reduced Wong-Wang model (Deco 2014) simulation on CPU

Parts of this code are based on https://github.com/BrainModes/The-Hybrid-Virtual-Brain, 
https://github.com/murraylab/hbnm & https://github.com/decolab/cb-neuromod

Author: Amin Saberi, Feb 2023
*/

gsl_vector * fc_tril(gsl_matrix * bold, bool exc_interhemispheric) {
    /*
     Given empirical/simulated bold (n_vols x n_regions) returns
     the lower triangle of the FC
    */
    int n_regions = bold->size2;
    int n_vols = bold->size1;
    int n_pairs = ((n_regions) * (n_regions - 1)) / 2;
    int rh_idx;
    if (exc_interhemispheric) {
        assert((bold->size2 % 2) == 0);
        rh_idx = bold->size2 / 2; // assumes symmetric number of parcels and L->R order
        n_pairs -= pow(rh_idx, 2); // exc the middle square
    }
    int i, j;
    double corr;
    gsl_vector * FC_tril = gsl_vector_alloc(n_pairs);
    int curr_idx = 0;
    for (i = 0; i<(bold->size2); i++) {
        for (j = 0; j<(bold->size2); j++) {
            if (i > j) {
                if (exc_interhemispheric) {
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
                    return NULL;
                }
                gsl_vector_set(FC_tril, curr_idx, corr);
                curr_idx ++;
            }
        }
    }
    return FC_tril;
}

gsl_vector * fcd_tril(gsl_matrix * bold, const int step, const int window_size, bool drop_edges, bool exc_interhemispheric) {
    /*
     Calculates the functional connectivity dynamics matrix (lower triangle)
     given BOLD, step and window size. Note that the actual window size is +1 higher.
     The FCD matrix shows similarity of FC patterns between the windows.
    */
    gsl_vector * FCD_tril;
    int n_vols = bold->size1;
    int n_regions = bold->size2;
    int n_pairs = ((n_regions) * (n_regions - 1)) / 2;
    if (exc_interhemispheric) {
        assert((bold->size2 % 2) == 0);
        int rh_idx = bold->size2 / 2; // assumes symmetric number of parcels and L->R order
        n_pairs -= pow(rh_idx, 2); // exc the middle square
    }
    int first_center, last_center, window_center, window_start, window_end;
    if (drop_edges) {
        first_center = window_size / 2;
        last_center = n_vols - 1 - (window_size / 2);
    } else {
        first_center = 0;
        last_center = n_vols - 1;
    }

    // too lazy to calculate the exact number of windows, but this should be the maximum bound (not sure though)
    gsl_matrix * window_FC_trils = gsl_matrix_alloc(n_pairs, ((n_vols + window_size) / step));
    // gsl_matrix_set_all(window_FC_trils, -1.5);
    // calculate the FC of each window
    int n_windows = 0;
    window_center = first_center;
    while (window_center <= last_center) {
        window_start = window_center - (window_size/2);
        if (window_start < 0)
            window_start = 0;
        window_end = window_center + (window_size/2);
        if (window_end >= n_vols)
            window_end = n_vols-1;
        gsl_matrix_view bold_window =  gsl_matrix_submatrix(bold, window_start, 0, window_end-window_start+1, n_regions);
        gsl_vector * window_FC_tril = fc_tril(&bold_window.matrix, exc_interhemispheric);
        if (window_FC_tril==NULL) {
            return NULL;
        }
        gsl_matrix_set_col(window_FC_trils, n_windows, window_FC_tril);
        window_center += step;
        n_windows ++;
    }
    if (n_windows < 10) {
        std::cout << "Warning: Too few FC windows: " << n_windows << "\n";
    }
    // calculate the FCD matrix (lower triangle)
    int window_i, window_j;
    double corr;
    int n_window_pairs = ((n_windows) * (n_windows - 1)) / 2;
    FCD_tril = gsl_vector_alloc(n_window_pairs);
    int curr_idx = 0;
    for (window_i=0; window_i<n_windows; window_i++) {
        for (window_j=0; window_j<n_windows; window_j++) {
            if (window_i > window_j) {
                gsl_vector_view FC_i = gsl_matrix_column(window_FC_trils, window_i);
                gsl_vector_view FC_j = gsl_matrix_column(window_FC_trils, window_j);
                // gsl_vector_fprintf(stdout, &FC_i.vector, "%f");
                corr = gsl_stats_correlation(
                    FC_i.vector.data, FC_i.vector.stride,
                    FC_j.vector.data, FC_j.vector.stride,
                    n_pairs
                );
                if (std::isnan(corr)) {
                    return NULL;
                }
                gsl_vector_set(FCD_tril, curr_idx, corr);
                curr_idx ++;
            }
        }
    }
    return FCD_tril;
}

double bnm(int nodes, u_real * SC, gsl_matrix * SC_gsl,
        u_real G, u_real * w_EE,  u_real * w_EI, u_real * w_IE, 
        int time_steps, int BOLD_TR, int rand_seed, int _max_fic_trials,
        int window_step, int window_size, 
        bool verbose, bool extended_output) {

    time_t start = time(NULL);

    /* random number generator */
    std::mt19937 rand_gen(rand_seed);
    // generating random numbers as floats regardless of USE_FLOATS
    // for better performance and consistency of the noise for the same
    // seed regardless of USE_FLOATS
    std::normal_distribution<float> normal_dist(0, 1);

    u_real TR        = (u_real)BOLD_TR / 1000; // (s) TR of fMRI data
    int   output_ts = (time_steps / (TR / model_dt))+1; // Length of BOLD time-series written to HDD    
    /*
     Allocate and Initialize arrays
     */
    size_t bold_size = nodes * output_ts;
    u_real *BOLD_ex                  = (u_real *)malloc((bold_size * sizeof(u_real)));
    u_real *S_i_E                    = (u_real *)malloc(nodes * sizeof(u_real));
    u_real *S_i_I                    = (u_real *)malloc(nodes * sizeof(u_real));
    u_real *_w_EE                    = (u_real *)malloc(nodes * sizeof(u_real));  // regional w_EE
    u_real *_w_EI                    = (u_real *)malloc(nodes * sizeof(u_real));  // regional w_EI
    u_real *_w_IE                    = (u_real *)malloc(nodes * sizeof(u_real));  // regional w_IE
    u_real *bw_x_ex                  = (u_real *)malloc(nodes * sizeof(u_real));  // State-variable 1 of BW-model (exc. pop.)
    u_real *bw_f_ex                  = (u_real *)malloc(nodes * sizeof(u_real));  // State-variable 2 of BW-model (exc. pop.)
    u_real *bw_nu_ex                 = (u_real *)malloc(nodes * sizeof(u_real));  // State-variable 3 of BW-model (exc. pop.)
    u_real *bw_q_ex                  = (u_real *)malloc(nodes * sizeof(u_real));  // State-variable 4 of BW-model (exc. pop.)
    u_real *r_i_E                    = (u_real *)malloc(nodes * sizeof(u_real));
    u_real *r_i_I                    = (u_real *)malloc(nodes * sizeof(u_real));
    u_real *I_i_E                    = (u_real *)malloc(nodes * sizeof(u_real));
    u_real *I_i_I                    = (u_real *)malloc(nodes * sizeof(u_real));

    if (S_i_E == NULL || S_i_I == NULL || _w_EE == NULL || _w_EI == NULL || _w_IE == NULL || 
        bw_x_ex == NULL || bw_f_ex == NULL || bw_nu_ex == NULL || bw_q_ex == NULL ||
        r_i_E == NULL || r_i_I == NULL || I_i_E == NULL || I_i_I == NULL) {
        printf("Error: Failed to allocate memory\n");
        exit(1);
    }

    for (int j = 0; j < nodes; j++) {
        S_i_E[j] = 0.001;
        S_i_I[j] = 0.001;
        _w_EE[j] = w_EE[j]; // this is redundant for wee and wei but needed for wie (in FIC+ case)
        _w_EI[j] = w_EI[j];
        _w_IE[j] = w_IE[j];
        bw_x_ex[j] = 0.0;
        bw_f_ex[j] = 1.0;
        bw_nu_ex[j] = 1.0;
        bw_q_ex[j] = 1.0;
        r_i_E[j] = 0;
        r_i_I[j] = 0;
        I_i_E[j] = 0;
        I_i_I[j] = 0;
    }

    // do FIC if wie (of the first region) is 0
    bool do_fic = false;
    double *_dw_EE, *_dw_EI;
    gsl_vector * _w_IE_vector;
    if (w_IE[0]==0) {
        if (verbose) std::cout << "Running FIC\n";
        do_fic = true;
        extended_output = true;
        _w_IE_vector = gsl_vector_alloc(nodes);
        // make a double copy of the wEE and wEI arrays (if needed in USE_FLOATS)
        #ifdef USE_FLOATS
            _dw_EE = (double *)malloc(nodes * sizeof(double));
            _dw_EI = (double *)malloc(nodes * sizeof(double));
            for (int j = 0; j < nodes; j++) {
                _dw_EE[j] = (double)_w_EE[j];
                _dw_EI[j] = (double)_w_EI[j];
            }
        #else
            #define _dw_EE _w_EE
            #define _dw_EI _w_EI
        #endif
        bool _unstable; // TODO: remove it as it is not used
        #pragma omp critical
        analytical_fic_het(
            SC_gsl, G, _dw_EE, _dw_EI,
            _w_IE_vector, &_unstable);
        // analytical fic is run as a critical step to avoid racing conditions
        // between the OMP threads
        // (this is a more cost-efficient solution than trying to
        // fix it with memory management)
        // copy the vector back to the _w_IE array
        for (int j = 0; j < nodes; j++) {
            _w_IE[j] = (u_real)(gsl_vector_get(_w_IE_vector, j));
        }
    }

    u_real *S_E_ex, *S_I_ex, *r_E_ex, *r_I_ex, *I_E_ex, *I_I_ex;
    if (extended_output) {
        S_E_ex                  = (u_real *)malloc((bold_size * sizeof(u_real)));
        S_I_ex                  = (u_real *)malloc((bold_size * sizeof(u_real)));
        r_E_ex                  = (u_real *)malloc((bold_size * sizeof(u_real)));
        r_I_ex                  = (u_real *)malloc((bold_size * sizeof(u_real)));
        I_E_ex                  = (u_real *)malloc((bold_size * sizeof(u_real)));
        I_I_ex                  = (u_real *)malloc((bold_size * sizeof(u_real)));
        if (S_E_ex == NULL || S_I_ex == NULL || r_E_ex == NULL || r_I_ex == NULL ||
            I_E_ex == NULL || I_I_ex == NULL) {
                printf("Error: Failed to allocate memory\n");
                exit(1);
        }
    }

    /*
    Integration
    */
    bool adjust_fic = (do_fic && numerical_fic);
    u_real *S_i_1_E = (u_real *)malloc(nodes * sizeof(u_real));
    u_real *mean_I_E = (u_real *)malloc(nodes * sizeof(u_real));
    u_real *delta = (u_real *)malloc(nodes * sizeof(u_real));
    if (S_i_1_E == NULL || mean_I_E == NULL || delta == NULL) {
        printf("Error: Failed to allocate memory\n");
        exit(1);
    }
    for (int j = 0; j < nodes; j++) {
        S_i_1_E[j] = S_i_E[j];
        mean_I_E[j] = 0;
        delta[j] = init_delta;
    }
    int fic_trial = 0;
    bool fic_failed = false;
    bool _adjust_fic = adjust_fic;
    u_real tmp_globalinput, tmp_rand_E, tmp_rand_I, 
        dSdt_E, dSdt_I, tmp_f, tmp_exp_E, tmp_exp_I,
        I_E_ba_diff;
    int j, k;
    int BOLD_len_i = 0;
    int ts_bold = 0;
    int bold_idx = 0;
    while (ts_bold <= time_steps) {
        if (verbose) printf("%.1f %% \r", ((u_real)ts_bold / (u_real)time_steps) * 100.0f );
        for (int int_i = 0; int_i < 10; int_i++) {
            // create a copy of S_E in the previous time point
            for (j=0; j<nodes; j++) {
                S_i_1_E[j] = S_i_E[j];
            }
            for (j=0; j<nodes; j++) {
                // calculate global input
                tmp_globalinput = 0;
                for (k=0; k<nodes; k++) {
                    tmp_globalinput += SC[j*nodes+k] * S_i_1_E[k];
                }
                // equations
                I_i_E[j] = w_E__I_0 + _w_EE[j] * S_i_E[j] + tmp_globalinput * G * J_NMDA - _w_IE[j]*S_i_I[j];
                I_i_I[j] = w_I__I_0 + _w_EI[j] * S_i_E[j] - S_i_I[j];
                tmp_exp_E = EXP(-d_E * (a_E * I_i_E[j] - b_E));
                tmp_exp_I = EXP(-d_I * (a_I * I_i_I[j] - b_I));
                r_i_E[j] = (a_E * I_i_E[j] - b_E) / (1 - tmp_exp_E);
                r_i_I[j] = (a_I * I_i_I[j] - b_I) / (1 - tmp_exp_I);
                tmp_rand_E = normal_dist(rand_gen);
                tmp_rand_I = normal_dist(rand_gen);
                dSdt_E = tmp_rand_E * sigma_model * sqrt_dt + dt * ((1 - S_i_E[j]) * gamma_E * r_i_E[j] - (S_i_E[j] / tau_E));
                dSdt_I = tmp_rand_I * sigma_model * sqrt_dt + dt * (gamma_I * r_i_I[j] - (S_i_I[j] / tau_I));
                S_i_E[j] += dSdt_E;
                S_i_I[j] += dSdt_I;
                // clip S to 0-1
                #ifdef USE_FLOATS
                S_i_E[j] = std::max(0.0f, std::min(S_i_E[j], 1.0f));
                S_i_I[j] = std::max(0.0f, std::min(S_i_I[j], 1.0f));
                #else
                S_i_E[j] = std::max(0.0, std::min(S_i_E[j], 1.0));
                S_i_I[j] = std::max(0.0, std::min(S_i_I[j], 1.0));
                #endif
            }
            // printf("\n");
        }

        /*
        Compute BOLD for that time-step (subsampled to 1 ms)
        save BOLD in addition to S_E, S_I, and r_E, r_I if requested
        */
        for (j=0; j<nodes; j++) {
            bw_x_ex[j]  = bw_x_ex[j]  +  model_dt * (S_i_E[j] - kappa * bw_x_ex[j] - y * (bw_f_ex[j] - 1.0));
            tmp_f       = bw_f_ex[j]  +  model_dt * bw_x_ex[j];
            bw_nu_ex[j] = bw_nu_ex[j] +  model_dt * itau * (bw_f_ex[j] - POW(bw_nu_ex[j], ialpha));
            bw_q_ex[j]  = bw_q_ex[j]  +  model_dt * itau * (bw_f_ex[j] * (1.0 - POW(oneminrho,(1.0/bw_f_ex[j]))) / rho  - POW(bw_nu_ex[j],ialpha) * bw_q_ex[j] / bw_nu_ex[j]);
            bw_f_ex[j]  = tmp_f;   
        }
        if (ts_bold % BOLD_TR == 0) {
            for (j = 0; j<nodes; j++) {
                bold_idx = BOLD_len_i*nodes+j;
                BOLD_ex[bold_idx] = 100 / rho * V_0 * (k1 * (1 - bw_q_ex[j]) + k2 * (1 - bw_q_ex[j]/bw_nu_ex[j]) + k3 * (1 - bw_nu_ex[j]));
                if (extended_output) {
                    S_E_ex[bold_idx] = S_i_E[j];
                    S_I_ex[bold_idx] = S_i_I[j];
                    r_E_ex[bold_idx] = r_i_E[j];
                    r_I_ex[bold_idx] = r_i_I[j];
                    I_E_ex[bold_idx] = I_i_E[j];
                    I_I_ex[bold_idx] = I_i_I[j];
                }
            }
            BOLD_len_i++;
        }
        ts_bold++;

        // adjust FIC according to Deco2014
        if (_adjust_fic) {
            if ((ts_bold >= I_SAMPLING_START) && (ts_bold <= I_SAMPLING_END)) {
                for (j = 0; j<nodes; j++) {
                    mean_I_E[j] += I_i_E[j];
                }
            }
            if (ts_bold == I_SAMPLING_END) {
                bool needs_fic_adjustment = false;
                if (verbose) printf("FIC adjustment trial %d\nnode\tIE_ba_diff\tdelta\tnew_w_IE\n", fic_trial);
                for (j = 0; j<nodes; j++) {
                    mean_I_E[j] /= I_SAMPLING_DURATION;
                    I_E_ba_diff = mean_I_E[j] - b_a_ratio_E;
                    if (abs(I_E_ba_diff + 0.026) > 0.005) {
                        needs_fic_adjustment = true;
                        if (fic_trial < _max_fic_trials) { // only do the adjustment if max trials is not exceeded
                            // up- or downregulate inhibition
                            if ((I_E_ba_diff) < -0.026) {
                                _w_IE[j] -= delta[j];
                                if (verbose) printf("%d\t%f\t-%f\t%f\n", j, I_E_ba_diff, delta[j], _w_IE[j]);
                                delta[j] -= 0.001;
                                delta[j] = fmaxf(delta[j], 0.001);
                            } else {
                                _w_IE[j] += delta[j];
                                if (verbose) printf("%d\t%f\t+%f\t%f\n", j, I_E_ba_diff, delta[j], _w_IE[j]);
                            }
                        }
                    }
                }
                if (needs_fic_adjustment) {
                    if (fic_trial < _max_fic_trials) {
                        // reset states
                        for (j = 0; j < nodes; j++) {
                            S_i_E[j] = 0.001;
                            S_i_I[j] = 0.001;
                            bw_x_ex[j] = 0.0;
                            bw_f_ex[j] = 1.0;
                            bw_nu_ex[j] = 1.0;
                            bw_q_ex[j] = 1.0;
                            mean_I_E[j] = 0;
                        }
                        // reset time and random sequence
                        ts_bold = 0;
                        BOLD_len_i = 0;
                        rand_gen = std::mt19937(rand_seed);
                        normal_dist = std::normal_distribution<float>(0, 1);
                        fic_trial++;
                    } else {
                        // continue the simulation but
                        // declare FIC failed
                        fic_failed = true;
                        _adjust_fic = false;
                    }
                }
                else {
                    // if no node needs fic adjustment don't run
                    // this block of code any more
                    _adjust_fic = false;
                }
            }
        }
    }

    // write to HDD
    FILE *BOLDout, *w_IE_file, *fic_trials_file, *fic_failed_file, *meanFRout, 
        *S_E_out, *S_I_out, *r_E_out, *r_I_out, *I_E_out, *I_I_out;
    if (save_output) {
        std::string bold_filename = out_prefix + "_bold.txt";
        BOLDout = fopen(bold_filename.c_str(), "w");
        if (do_fic) {
            std::string w_IE_filename = out_prefix + "_w_IE.txt";
            std::string fic_trials_filename = out_prefix + "_fic_ntrials.txt";
            std::string fic_failed_filename = out_prefix + "_fic_failed.txt";
            w_IE_file = fopen(w_IE_filename.c_str(), "w");
            fic_trials_file = fopen(fic_trials_filename.c_str(), "w");
            fic_failed_file = fopen(fic_failed_filename.c_str(), "w");
            for (j = 0; j < nodes; j++) {
                fprintf(w_IE_file, "%f\n", _w_IE[j]);
            }
            fprintf(fic_trials_file, "%d of max %d", fic_trial, _max_fic_trials);
            fprintf(fic_failed_file, "%d", fic_failed);
            fclose(w_IE_file);
            fclose(fic_trials_file);
            fclose(fic_failed_file);
        }
        // std::string fr_filename = out_prefix + "_meanFR.txt";
        // meanFRout = fopen(fr_filename.c_str(), "w");
        if (extended_output) {
            std::string S_E_filename = out_prefix + "_S_E.txt";
            S_E_out = fopen(S_E_filename.c_str(), "w");
            std::string S_I_filename = out_prefix + "_S_I.txt";
            S_I_out = fopen(S_I_filename.c_str(), "w");
            std::string r_E_filename = out_prefix + "_r_E.txt";
            r_E_out = fopen(r_E_filename.c_str(), "w");
            std::string r_I_filename = out_prefix + "_r_I.txt";
            r_I_out = fopen(r_I_filename.c_str(), "w");
            std::string I_E_filename = out_prefix + "_I_E.txt";
            I_E_out = fopen(I_E_filename.c_str(), "w");
            std::string I_I_filename = out_prefix + "_I_I.txt";
            I_I_out = fopen(I_I_filename.c_str(), "w");
        }
    }
    /*
        Print BOLD time series in addition to S_E, S_I, and r_E, r_I if requested
        Also save BOLD to gsl_matrix for using in check_fit
    */
    int n_vols_remove = bold_remove_s * 1000 / BOLD_TR; // // "BNM_BOLD_REMOVE" 30 seconds by default (only removed from gsl_matrix and not the saved txt)
    gsl_matrix * bold;
    if (!(sim_only)) {
        bold = gsl_matrix_alloc(BOLD_len_i-n_vols_remove, nodes);
    }
    u_real *mean_r_E;
    if (calculate_fic_penalty) {
        mean_r_E = (u_real *)malloc(nodes * sizeof(u_real)); // for fic_penalty calculation
        for (int j=0; j<nodes; j++) {
            mean_r_E[j] = 0;
        }
    }

    // copy bold[n_vols_remove:,:] to a GSL matrix for
    // FC and FCD calculation
    for (int i=n_vols_remove; i<output_ts; i++) {
        for (j=0; j<nodes; j++) {
            gsl_matrix_set(bold, i-n_vols_remove, j, BOLD_ex[i*nodes+j]);
        }
    }

    // Calculate FC and FCD
    gsl_vector * sim_FC_tril = fc_tril(bold, conf.exc_interhemispheric);
    gsl_vector * sim_FCD_tril = fcd_tril(bold, window_step, window_size, conf.drop_edges, conf.exc_interhemispheric);

    // Free memory
    free(delta); free(mean_I_E); free(S_i_1_E);
    if (extended_output) {
        free(I_I_ex); free(I_E_ex); free(r_I_ex);  
        free(r_E_ex); free(S_I_ex); free(S_E_ex); 
    }
    free(I_i_I); free(I_i_E); free(r_i_I); free(r_i_E);
    free(bw_q_ex); free(bw_nu_ex); free(bw_f_ex); free(bw_x_ex); 
    free(_w_IE); free(_w_EI); free(_w_EE); 
    free(S_i_I); free(S_i_E); free(BOLD_ex); 
}

void run_simulations_cpu(
    double * BOLD_ex_out, double * fc_trils_out, double * fcd_trils_out,
    u_real * G_list, u_real * w_EE_list, u_real * w_EI_list, u_real * w_IE_list,
    u_real * SC, gsl_matrix * SC_gsl, int nodes, int N_SIMS,
    int time_steps, int BOLD_TR, int rand_seed, int _max_fic_trials,
    int window_step, int window_size, bool sim_verbose, bool extended_output
)
{
    #pragma omp parallel
	#pragma omp for
    for(int IndPar = 0; IndPar < N_SIMS; IndPar++) {
        // write thread info with the time
        std::chrono::system_clock::time_point now = std::chrono::system_clock::now();
        std::time_t current_time = std::chrono::system_clock::to_time_t(now);
        std::tm* timeinfo = std::localtime(&current_time);
        char time_str[9];
        std::strftime(time_str, sizeof(time_str), "%T", timeinfo);
        ::printf("Thread %d (of %d) is executing particle %d [%s]\n", omp_get_thread_num(), omp_get_num_threads(), IndPar, time_str);
        // run the simulation, calcualte FC and FCD and gof
        // (all in bnm function)
        bnm(nodes, SC, SC_gsl,
            G_list[IndPar], 
            w_EE_list+(IndPar*nodes), // the starting index of the w_EE_list slice of current particle
            w_EI_list+(IndPar*nodes),
            w_IE_list+(IndPar*nodes),
            time_steps, BOLD_TR, rand_seed, _max_fic_trials,
            window_step, window_size,
            sim_verbose, extended_output);
    }
}
