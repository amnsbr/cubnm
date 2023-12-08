/*
Reduced Wong-Wang model (Deco 2014) simulation on CPU

Parts of this code are based on https://github.com/BrainModes/The-Hybrid-Virtual-Brain, 
https://github.com/murraylab/hbnm & https://github.com/decolab/cb-neuromod

Author: Amin Saberi, Feb 2023
*/

// use a namespace dedicated to cpu implementation
// to avoid overlaps with similar variables on gpu
namespace bnm_cpu {
    bool is_initialized = false;
    int last_time_steps = 0; // to avoid recalculating noise in subsequent calls of the function with force_reinit
    int last_nodes = 0;
    #ifdef NOISE_SEGMENT
    int *shuffled_nodes, *shuffled_ts;
    // set a default length of noise (msec)
    // (+1 to avoid having an additional repeat for a single time point
    // when time_steps can be divided by 30(000), as the actual duration of
    // simulation (in msec) is always user request time steps + 1)
    int noise_time_steps = 30001; 
    int noise_repeats; // number of noise segment repeats
    #endif
    u_real *noise;
    int n_pairs, n_window_pairs, n_windows, output_ts, corr_len, n_vols_remove;
    size_t bold_size;
    int *window_starts, *window_ends;
}

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

gsl_vector * calculate_fc_tril(gsl_matrix * bold) {
    /*
     Given empirical/simulated bold (n_vols x nodes) returns
     the lower triangle of the FC
    */
    using namespace bnm_cpu; // n_pairs is in here
    int nodes = bold->size2;
    int n_vols = bold->size1;
    int rh_idx = bold->size2 / 2; // assumes symmetric number of parcels and L->R order
    int i, j;
    double corr;
    gsl_vector * FC_tril = gsl_vector_alloc(n_pairs);
    int curr_idx = 0;
    for (i = 0; i<(bold->size2); i++) {
        for (j = 0; j<(bold->size2); j++) {
            if (i > j) {
                if (conf.exc_interhemispheric) {
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

gsl_vector * calculate_fcd_tril(gsl_matrix * bold) {
    /*
     Calculates the functional connectivity dynamics matrix (lower triangle)
     given BOLD, step and window size. Note that the actual window size is +1 higher.
     The FCD matrix shows similarity of FC patterns between the windows.
    */
    using namespace bnm_cpu; // includes window info which is already calculated in init_cpu
    int n_vols = bold->size1;
    int nodes = bold->size2;
    gsl_vector * FCD_tril = gsl_vector_alloc(n_window_pairs);
    gsl_matrix * window_FC_trils = gsl_matrix_alloc(n_pairs, n_windows);
    if (n_windows < 10) {
        printf("Warning: Too few FC windows: %d\n", n_windows);
    }
    // calculate dynamic FC
    for (int i=0; i<n_windows; i++) {
        gsl_matrix_view bold_window =  gsl_matrix_submatrix(
            bold, 
            window_starts[i], 0, 
            window_ends[i]-window_starts[i]+1, nodes);
        gsl_vector * window_FC_tril = calculate_fc_tril(&bold_window.matrix);
        if (window_FC_tril==NULL) {
            printf("Error: Dynamic FC calculation failed\n");
            return NULL;
        }
        gsl_matrix_set_col(window_FC_trils, i, window_FC_tril);
        gsl_vector_free(window_FC_tril);
    }
    // calculate the FCD matrix (lower triangle)
    int window_i, window_j;
    double corr;
    int curr_idx = 0;
    for (window_i=0; window_i<n_windows; window_i++) {
        for (window_j=0; window_j<n_windows; window_j++) {
            if (window_i > window_j) {
                gsl_vector_view FC_i = gsl_matrix_column(window_FC_trils, window_i);
                gsl_vector_view FC_j = gsl_matrix_column(window_FC_trils, window_j);
                corr = gsl_stats_correlation(
                    FC_i.vector.data, FC_i.vector.stride,
                    FC_j.vector.data, FC_j.vector.stride,
                    n_pairs
                );
                if (std::isnan(corr)) {
                    printf("Error: FCD[%d,%d] is NaN\n", window_i, window_j);
                    return NULL;
                }
                gsl_vector_set(FCD_tril, curr_idx, corr);
                curr_idx ++;
            }
        }
    }
    return FCD_tril;
}

void bnm(double * BOLD_ex, double * fc_tril_out, double * fcd_tril_out,
        double * S_E_mean, double * S_I_mean, double * S_ratio_mean,
        double * r_E_mean, double * r_I_mean, double * r_ratio_mean,
        double * I_E_mean, double * I_I_mean, double * I_ratio_mean,
        bool * fic_unstable_p, bool * fic_failed_p,
        int nodes, u_real * SC, gsl_matrix * SC_gsl,
        u_real G, u_real * w_EE,  u_real * w_EI, u_real * w_IE,
        bool do_fic, int time_steps, int BOLD_TR, int rand_seed, int _max_fic_trials,
        int window_step, int window_size, 
        bool extended_output, int sim_idx,
        struct ModelConstants mc, struct ModelConfigs conf) {
    using namespace bnm_cpu;
    time_t start = time(NULL);
    /*
     Allocate and Initialize arrays
     */
    u_real *_w_EE                    = (u_real *)malloc(nodes * sizeof(u_real));  // regional w_EE
    u_real *_w_EI                    = (u_real *)malloc(nodes * sizeof(u_real));  // regional w_EI
    u_real *_w_IE                    = (u_real *)malloc(nodes * sizeof(u_real));  // regional w_IE
    // size_t bold_size = nodes * output_ts;
    // u_real *BOLD_ex                  = (u_real *)malloc((bold_size * sizeof(u_real)));
    u_real *S_i_E                    = (u_real *)malloc(nodes * sizeof(u_real));
    u_real *S_i_I                    = (u_real *)malloc(nodes * sizeof(u_real));
    u_real *r_i_E                    = (u_real *)malloc(nodes * sizeof(u_real));
    u_real *r_i_I                    = (u_real *)malloc(nodes * sizeof(u_real));
    u_real *I_i_E                    = (u_real *)malloc(nodes * sizeof(u_real));
    u_real *I_i_I                    = (u_real *)malloc(nodes * sizeof(u_real));
    u_real *bw_x_ex                  = (u_real *)malloc(nodes * sizeof(u_real));  // State-variable 1 of BW-model (exc. pop.)
    u_real *bw_f_ex                  = (u_real *)malloc(nodes * sizeof(u_real));  // State-variable 2 of BW-model (exc. pop.)
    u_real *bw_nu_ex                 = (u_real *)malloc(nodes * sizeof(u_real));  // State-variable 3 of BW-model (exc. pop.)
    u_real *bw_q_ex                  = (u_real *)malloc(nodes * sizeof(u_real));  // State-variable 4 of BW-model (exc. pop.)

    if (S_i_E == NULL || S_i_I == NULL || _w_EE == NULL || _w_EI == NULL || _w_IE == NULL || 
        bw_x_ex == NULL || bw_f_ex == NULL || bw_nu_ex == NULL || bw_q_ex == NULL ||
        r_i_E == NULL || r_i_I == NULL || I_i_E == NULL || I_i_I == NULL) {
        printf("Error: Failed to allocate memory to internal simulation variables\n");
        exit(1);
    }
    bool _extended_output = (do_fic || extended_output);
    for (int j = 0; j < nodes; j++) {
        _w_EE[j] = w_EE[j]; // this is redundant for wee and wei but needed for wie (in FIC+ case)
        _w_EI[j] = w_EI[j];
        _w_IE[j] = w_IE[j];
        S_i_E[j] = 0.001;
        S_i_I[j] = 0.001;
        r_i_E[j] = 0;
        r_i_I[j] = 0;
        I_i_E[j] = 0;
        I_i_I[j] = 0;
        bw_x_ex[j] = 0.0;
        bw_f_ex[j] = 1.0;
        bw_nu_ex[j] = 1.0;
        bw_q_ex[j] = 1.0;
        if (_extended_output) {
            S_E_mean[j] = 0; // initialize sum (mean) of extended output to 0
            S_I_mean[j] = 0;
            S_ratio_mean[j] = 0;
            r_E_mean[j] = 0;
            r_I_mean[j] = 0;
            r_ratio_mean[j] = 0;
            I_E_mean[j] = 0;
            I_I_mean[j] = 0;
            I_ratio_mean[j] = 0;
        }
    }
    // do FIC if indicated
    double *_dw_EE, *_dw_EI;
    gsl_vector * _w_IE_vector;
    if (do_fic) {
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
        *fic_unstable_p = false;
        #ifdef OMP_ENABLED
        #pragma omp critical
        #endif
        analytical_fic_het(
            SC_gsl, G, _dw_EE, _dw_EI,
            _w_IE_vector, fic_unstable_p);
        // analytical fic is run as a critical step to avoid racing conditions
        // between the OMP threads
        if (*fic_unstable_p) {
            printf("In simulation #%d FIC solution is unstable. Setting wIE to 1 in all nodes\n", sim_idx);
            for (int j=0; j<nodes; j++) {
                _w_IE[j] = 1.0;
            }
        } else {
            // copy to w_IE_fic which will be passed on to the device
            for (int j=0; j<nodes; j++) {
                _w_IE[j] = (u_real)(gsl_vector_get(_w_IE_vector, j));
            }
        }
        gsl_vector_free(_w_IE_vector);
    }

    u_real *S_E_ts, *S_I_ts, *r_E_ts, *r_I_ts, *I_E_ts, *I_I_ts;
    if (_extended_output && conf.extended_output_ts) {
        S_E_ts                  = (u_real *)malloc((bold_size * sizeof(u_real)));
        S_I_ts                  = (u_real *)malloc((bold_size * sizeof(u_real)));
        r_E_ts                  = (u_real *)malloc((bold_size * sizeof(u_real)));
        r_I_ts                  = (u_real *)malloc((bold_size * sizeof(u_real)));
        I_E_ts                  = (u_real *)malloc((bold_size * sizeof(u_real)));
        I_I_ts                  = (u_real *)malloc((bold_size * sizeof(u_real)));
        if (S_E_ts == NULL || S_I_ts == NULL || r_E_ts == NULL || r_I_ts == NULL ||
            I_E_ts == NULL || I_I_ts == NULL) {
                printf("Error: Failed to allocate memory to extended output\n");
                exit(1);
        }
    }

    // allocate memory to BOLD gsl matrix used for FC and FCD calculation
    gsl_matrix * bold_gsl = gsl_matrix_alloc(output_ts, nodes);

    // Integration
    bool adjust_fic = (do_fic && conf.numerical_fic);
    u_real *S_i_1_E, *mean_I_E, *delta;
    S_i_1_E = (u_real *)malloc(nodes * sizeof(u_real));
    for (int j = 0; j < nodes; j++) {
        S_i_1_E[j] = S_i_E[j];
    }
    if (adjust_fic) {
        mean_I_E = (u_real *)malloc(nodes * sizeof(u_real)); // this is different from I_E_mean, which starts after bold_remove_s
        delta = (u_real *)malloc(nodes * sizeof(u_real));
        for (int j = 0; j < nodes; j++) {
            mean_I_E[j] = 0;
            delta[j] = conf.init_delta;
        }
    }
    int fic_trial = 0;
    *fic_failed_p = false;
    bool _adjust_fic = adjust_fic; // whether to adjust FIC in the next trial
    bool needs_fic_adjustment;
    u_real tmp_globalinput, tmp_rand_E, tmp_rand_I, 
        dSdt_E, dSdt_I, tmp_f, tmp_aIb_E, tmp_aIb_I,
        I_E_ba_diff;
    int j, k, noise_idx;
    int BOLD_len_i = 0;
    int ts_bold = 0;
    int bold_idx = 0;
    // set up noise shuffling if indicated
    #ifdef NOISE_SEGMENT
    /* 
    How noise shuffling works?
    At each time point we will have `ts_bold` which is the real time (in msec) 
    from the start of simulation, `ts_noise` which is the real time passed within 
    each repeat of the noise segment (`curr_noise_repeat`), `sh_ts_noise` which is 
    the shuffled timepoint (column of the noise segment) that will be used for getting 
    the noise of nodes * 10 int_i * 2 neurons for the current msec. 
    Similarly, in each thread we have `j` which is mapped to a `sh_j` which will 
    vary in each repeat.
    */
    // get position of the node
    // in shuffled nodes for the first
    // repeat of noise segment
    int curr_noise_repeat = 0;
    int ts_noise = 0;
    // keep track of shuffled position of
    // current node, ts
    int sh_j, sh_ts_noise;
    #endif
    while (ts_bold <= time_steps) {
        if (conf.sim_verbose) printf("%.1f %% \r", ((u_real)ts_bold / (u_real)time_steps) * 100.0f );
        #ifdef NOISE_SEGMENT
        sh_ts_noise = shuffled_ts[ts_noise+curr_noise_repeat*noise_time_steps];
        #endif
        for (int int_i = 0; int_i < 10; int_i++) {
            // copy S_E of previous time point to S_i_1_E
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
                I_i_E[j] = mc.w_E__I_0 + _w_EE[j] * S_i_E[j] + tmp_globalinput * G * mc.J_NMDA - _w_IE[j]*S_i_I[j];
                I_i_I[j] = mc.w_I__I_0 + _w_EI[j] * S_i_E[j] - S_i_I[j];
                tmp_aIb_E = mc.a_E * I_i_E[j] - mc.b_E;
                tmp_aIb_I = mc.a_I * I_i_I[j] - mc.b_I;
                #ifdef USE_FLOATS
                // to avoid firing rate approaching infinity near I = b/a
                if (abs(tmp_aIb_E) < 1e-4) tmp_aIb_E = 1e-4;
                if (abs(tmp_aIb_I) < 1e-4) tmp_aIb_I = 1e-4;
                #endif
                r_i_E[j] = tmp_aIb_E / (1 - EXP(-mc.d_E * tmp_aIb_E));
                r_i_I[j] = tmp_aIb_I / (1 - EXP(-mc.d_I * tmp_aIb_I));
                #ifdef NOISE_SEGMENT
                // * 10 for 0.1 msec steps, nodes * 2 and [sh_]j*2 for two E and I neurons
                sh_j = shuffled_nodes[curr_noise_repeat*nodes+j];
                noise_idx = (((sh_ts_noise * 10 + int_i) * nodes * 2) + (sh_j * 2));
                #else
                noise_idx = (((ts_bold * 10 + int_i) * nodes * 2) + (j * 2));
                #endif
                tmp_rand_E = noise[noise_idx];
                tmp_rand_I = noise[noise_idx+1];
                dSdt_E = tmp_rand_E * mc.sigma_model * mc.sqrt_dt + mc.dt * ((1 - S_i_E[j]) * mc.gamma_E * r_i_E[j] - (S_i_E[j] * mc.itau_E));
                dSdt_I = tmp_rand_I * mc.sigma_model * mc.sqrt_dt + mc.dt * (mc.gamma_I * r_i_I[j] - (S_i_I[j] * mc.itau_I));
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
        }

        /*
        Compute BOLD for that time-step (subsampled to 1 ms)
        save BOLD in addition to S_E, S_I, and r_E, r_I if requested
        */
        for (j=0; j<nodes; j++) {
            bw_x_ex[j]  = bw_x_ex[j]  +  mc.model_dt * (S_i_E[j] - mc.kappa * bw_x_ex[j] - mc.y * (bw_f_ex[j] - 1.0));
            tmp_f       = bw_f_ex[j]  +  mc.model_dt * bw_x_ex[j];
            bw_nu_ex[j] = bw_nu_ex[j] +  mc.model_dt * mc.itau * (bw_f_ex[j] - POW(bw_nu_ex[j], mc.ialpha));
            bw_q_ex[j]  = bw_q_ex[j]  +  mc.model_dt * mc.itau * (bw_f_ex[j] * (1.0 - POW(mc.oneminrho,(1.0/bw_f_ex[j]))) / mc.rho  - POW(bw_nu_ex[j],mc.ialpha) * bw_q_ex[j] / bw_nu_ex[j]);
            bw_f_ex[j]  = tmp_f;   
        }
        if (ts_bold % BOLD_TR == 0) {
            for (j = 0; j<nodes; j++) {
                bold_idx = BOLD_len_i*nodes+j;
                BOLD_ex[bold_idx] = 100 / mc.rho * mc.V_0 * (mc.k1 * (1 - bw_q_ex[j]) + mc.k2 * (1 - bw_q_ex[j]/bw_nu_ex[j]) + mc.k3 * (1 - bw_nu_ex[j]));
                gsl_matrix_set(bold_gsl, BOLD_len_i, j, BOLD_ex[bold_idx]);
                if (conf.extended_output_ts) {
                    S_E_ts[bold_idx] = S_i_E[j];
                    S_I_ts[bold_idx] = S_i_I[j];
                    r_E_ts[bold_idx] = r_i_E[j];
                    r_I_ts[bold_idx] = r_i_I[j];
                    I_E_ts[bold_idx] = I_i_E[j];
                    I_I_ts[bold_idx] = I_i_I[j];
                }
                // save data to extended output means
                // only after n_vols_remove
                if ((BOLD_len_i>=n_vols_remove)) {
                    if (_extended_output) {
                        S_E_mean[j] += S_i_E[j];
                        S_I_mean[j] += S_i_I[j];
                        S_ratio_mean[j] += S_i_E[j]/S_i_I[j];
                        r_E_mean[j] += r_i_E[j];
                        r_I_mean[j] += r_i_I[j];
                        r_ratio_mean[j] += r_i_E[j]/r_i_I[j];
                        I_E_mean[j] += I_i_E[j];
                        I_I_mean[j] += I_i_I[j];
                        I_ratio_mean[j] += I_i_E[j]/I_i_I[j];
                    }
                }
            }
            BOLD_len_i++;
        }
        ts_bold++;

        #ifdef NOISE_SEGMENT
        // update noise segment time
        ts_noise++;
        // reset noise segment time 
        // and shuffle nodes if the segment
        // has reached to the end
        if (ts_noise % noise_time_steps == 0) {
            curr_noise_repeat++;
            ts_noise = 0;
        }
        #endif

        // adjust FIC according to Deco2014
        if (_adjust_fic) {
            if ((ts_bold >= conf.I_SAMPLING_START) && (ts_bold <= conf.I_SAMPLING_END)) {
                for (j = 0; j<nodes; j++) {
                    mean_I_E[j] += I_i_E[j];
                }
            }
            if (ts_bold == conf.I_SAMPLING_END) {
                needs_fic_adjustment = false;
                if (conf.fic_verbose) printf("FIC adjustment trial %d\nnode\tIE_ba_diff\tdelta\tnew_w_IE\n", fic_trial);
                for (j = 0; j<nodes; j++) {
                    mean_I_E[j] /= conf.I_SAMPLING_DURATION;
                    I_E_ba_diff = mean_I_E[j] - mc.b_a_ratio_E;
                    if (abs(I_E_ba_diff + 0.026) > 0.005) {
                        needs_fic_adjustment = true;
                        if (fic_trial < _max_fic_trials) { // only do the adjustment if max trials is not exceeded
                            // up- or downregulate inhibition
                            if ((I_E_ba_diff) < -0.026) {
                                _w_IE[j] -= delta[j];
                                if (conf.fic_verbose) printf("%d\t%f\t-%f\t%f\n", j, I_E_ba_diff, delta[j], _w_IE[j]);
                                delta[j] -= 0.001;
                                delta[j] = fmaxf(delta[j], 0.001);
                            } else {
                                _w_IE[j] += delta[j];
                                if (conf.fic_verbose) printf("%d\t%f\t+%f\t%f\n", j, I_E_ba_diff, delta[j], _w_IE[j]);
                            }
                        }
                    }
                }
                if (needs_fic_adjustment) {
                    if (fic_trial < _max_fic_trials) {
                        // reset time
                        ts_bold = 0;
                        BOLD_len_i = 0;
                        fic_trial++;
                        #ifdef NOISE_SEGMENT
                        curr_noise_repeat = 0;
                        ts_noise = 0;
                        #endif
                        // reset states
                        for (j = 0; j < nodes; j++) {
                            S_i_E[j] = 0.001;
                            S_i_I[j] = 0.001;
                            bw_x_ex[j] = 0.0;
                            bw_f_ex[j] = 1.0;
                            bw_nu_ex[j] = 1.0;
                            bw_q_ex[j] = 1.0;
                            mean_I_E[j] = 0;
                            if (extended_output) {
                                S_E_mean[j] = 0;
                                S_I_mean[j] = 0;
                                S_ratio_mean[j] = 0;
                                r_E_mean[j] = 0;
                                r_I_mean[j] = 0;
                                r_ratio_mean[j] = 0;
                                I_E_mean[j] = 0;
                                I_I_mean[j] = 0;
                                I_ratio_mean[j] = 0;
                            }
                        }
                    } else {
                        // continue the simulation but
                        // declare FIC failed
                        *fic_failed_p = true;
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

    // Calculate FC and FCD
    // for FC discard first n_vols_remove of BOLD
    // (for FCD this is not needed as window_starts
    // starts after n_vols_remove, as calcualated in get_dfc_windows)
    gsl_matrix_view bold_window =  gsl_matrix_submatrix(
        bold_gsl, 
        n_vols_remove, 0, 
        output_ts-n_vols_remove, nodes);
    gsl_vector * fc_tril = calculate_fc_tril(&bold_window.matrix);
    gsl_vector * fcd_tril = calculate_fcd_tril(bold_gsl);

    // copy FC and FCD to numpy arrays
    memcpy(fc_tril_out, gsl_vector_ptr(fc_tril, 0), sizeof(double) * n_pairs);
    memcpy(fcd_tril_out, gsl_vector_ptr(fcd_tril, 0), sizeof(double) * n_window_pairs);

    // Free memory
    gsl_vector_free(fcd_tril); gsl_vector_free(fc_tril); gsl_matrix_free(bold_gsl);
    free(delta); free(mean_I_E); free(S_i_1_E);
    if (extended_output && conf.extended_output_ts) {
        free(I_I_ts); free(I_E_ts); free(r_I_ts);  
        free(r_E_ts); free(S_I_ts); free(S_E_ts); 
    }
    free(I_i_I); free(I_i_E); free(r_i_I); free(r_i_E);
    free(bw_q_ex); free(bw_nu_ex); free(bw_f_ex); free(bw_x_ex); 
    free(_w_IE); free(_w_EI); free(_w_EE); 
    free(S_i_I); free(S_i_E); 
}

void run_simulations_cpu(
    double * BOLD_ex_out, double * fc_trils_out, double * fcd_trils_out,
    double * S_E_out, double * S_I_out, double * S_ratio_out,
    double * r_E_out, double * r_I_out, double * r_ratio_out,
    double * I_E_out, double * I_I_out, double * I_ratio_out,
    bool * fic_unstable_out, bool * fic_failed_out,
    u_real * G_list, u_real * w_EE_list, u_real * w_EI_list, u_real * w_IE_list, u_real * v_list,
    u_real * SC, gsl_matrix * SC_gsl, u_real * SC_dist, bool do_delay,
    int nodes, int time_steps, int BOLD_TR, int _max_fic_trials, 
    int window_size, int window_step,
    int N_SIMS, bool do_fic, bool only_wIE_free, bool extended_output,
    struct ModelConstants mc, struct ModelConfigs conf, int rand_seed
) {
    using namespace bnm_cpu;
    // run the simulations
    #ifdef OMP_ENABLED
    #pragma omp parallel
	#pragma omp for
    #endif
    for(int sim_idx = 0; sim_idx < N_SIMS; sim_idx++) {
        // write thread info with the time
        std::chrono::system_clock::time_point now = std::chrono::system_clock::now();
        std::time_t current_time = std::chrono::system_clock::to_time_t(now);
        std::tm* timeinfo = std::localtime(&current_time);
        char time_str[9];
        std::strftime(time_str, sizeof(time_str), "%T", timeinfo);
        #ifdef OMP_ENABLED
            printf("Thread %d (of %d) is executing particle %d [%s]\n", omp_get_thread_num(), omp_get_num_threads(), sim_idx, time_str);
        #else
            printf("Executing particle %d [%s]\n", sim_idx, time_str);
        #endif
        // run the simulation and calcualte FC and FC
        // write the output to chunks of output variables
        // dedicated to the current simulation
        bnm(
            BOLD_ex_out+(sim_idx*bold_size), 
            fc_trils_out+(sim_idx*n_pairs), 
            fcd_trils_out+(sim_idx*n_window_pairs),
            S_E_out+(sim_idx*nodes),
            S_I_out+(sim_idx*nodes),
            S_ratio_out+(sim_idx*nodes),
            r_E_out+(sim_idx*nodes),
            r_I_out+(sim_idx*nodes),
            r_ratio_out+(sim_idx*nodes),
            I_E_out+(sim_idx*nodes),
            I_I_out+(sim_idx*nodes),
            I_ratio_out+(sim_idx*nodes),
            fic_unstable_out+sim_idx,
            fic_failed_out+sim_idx,
            nodes, SC, SC_gsl,
            G_list[sim_idx], 
            w_EE_list+(sim_idx*nodes),
            w_EI_list+(sim_idx*nodes),
            w_IE_list+(sim_idx*nodes),
            do_fic, time_steps, BOLD_TR, rand_seed, _max_fic_trials,
            window_step, window_size,
            extended_output,
            sim_idx, mc, conf);
    }
}

void init_cpu(
        int *output_ts_p, int *n_pairs_p, int *n_window_pairs_p,
        int nodes, int rand_seed,
        int BOLD_TR, int time_steps, int window_size, int window_step,
        struct ModelConstants mc, struct ModelConfigs conf) {
    using namespace bnm_cpu;
    // precalculate noise (segments) similar to GPU
    // to have a similar noise array given the same seed
    // between CPU and GPU + for better performance
    #ifndef NOISE_SEGMENT
    // precalculate the entire noise needed; can use up a lot of memory
    // with high N of nodes and longer durations leads maxes out the memory
    int noise_size = nodes * (time_steps+1) * 10 * 2; // +1 for inclusive last time point, 2 for E and I
    #else
    // otherwise precalculate a noise segment and arrays of shuffled
    // nodes and time points and reuse-shuffle the noise segment
    // throughout the simulation for `noise_repeats`
    int noise_size = nodes * (noise_time_steps) * 10 * 2;
    noise_repeats = ceil((float)(time_steps+1) / (float)noise_time_steps); // +1 for inclusive last time point
    #endif
    if ((time_steps != last_time_steps) || (nodes != last_nodes)) {
        printf("Precalculating %d noise elements...\n", noise_size);
        last_time_steps = time_steps;
        last_nodes = nodes;
        std::mt19937 rand_gen(rand_seed);
        // generating random numbers as floats regardless of USE_FLOATS
        // for better performance and consistency of the noise for the same
        // seed regardless of USE_FLOATS
        std::normal_distribution<float> normal_dist(0, 1);
        noise = (u_real *)malloc(sizeof(u_real) * noise_size);
        for (int i = 0; i < noise_size; i++) {
            #ifdef USE_FLOATS
            noise[i] = normal_dist(rand_gen);
            #else
            noise[i] = (double)normal_dist(rand_gen);
            #endif
        }
        #ifdef NOISE_SEGMENT
        // create shuffled nodes indices for each repeat of the 
        // precalculaed noise (row shuffling)
        printf("noise will be repeated %d times (nodes [rows] and timepoints [columns] will be shuffled in each repeat)\n", noise_repeats);
        std::vector<int> node_indices(nodes);
        std::iota(node_indices.begin(), node_indices.end(), 0);
        shuffled_nodes = (int *)malloc(sizeof(int) * noise_repeats * nodes);
        for (int i = 0; i < noise_repeats; i++) {
            std::shuffle(node_indices.begin(), node_indices.end(), rand_gen);
            std::copy(node_indices.begin(), node_indices.end(), shuffled_nodes+(i*nodes));
        }
        // similarly create shuffled time point indices (msec)
        // for each repeat (column shuffling)
        std::vector<int> ts_indices(noise_time_steps);
        std::iota(ts_indices.begin(), ts_indices.end(), 0);
        shuffled_ts = (int *)malloc(sizeof(int) * noise_repeats * noise_time_steps);
        for (int i = 0; i < noise_repeats; i++) {
            std::shuffle(ts_indices.begin(), ts_indices.end(), rand_gen);
            std::copy(ts_indices.begin(), ts_indices.end(), shuffled_ts+(i*noise_time_steps));
        }
        #endif
    } else {
        printf("Noise already precalculated\n");
    }
    // calculate bold size for properly allocating parts
    // of BOLD_ex_out to the simulations
    u_real TR = (u_real)BOLD_TR / 1000; // (s) TR of fMRI data
    output_ts = (time_steps / (TR / mc.model_dt))+1; // Length of BOLD time-series written to HDD
    bold_size = nodes * output_ts;
    // specify n_vols_remove (for extended output and FC calculations)
    n_vols_remove = conf.bold_remove_s * 1000 / BOLD_TR; // 30 seconds
    // calculate length of BOLD after removing initial volumes
    corr_len = output_ts - n_vols_remove;
    if (corr_len < 2) {
        printf("Number of BOLD timepoints (after removing initial %ds of the simulation) is too low for FC calculations\n", conf.bold_remove_s);
        exit(1);
    }
    // calculate the number of FC pairs
    n_pairs = get_fc_n_pairs(nodes);
    // calculate the number of windows and their start-ends
    std::vector<int> _window_starts, _window_ends;
    n_windows = get_dfc_windows(
        &_window_starts, &_window_ends, 
        corr_len, output_ts, n_vols_remove,
        window_step, window_size);
    // copy the vectors to arrays
    window_starts = (int *)malloc(sizeof(int) * n_windows);
    window_ends = (int *)malloc(sizeof(int) * n_windows);
    std::copy(_window_starts.begin(), _window_starts.end(), window_starts);
    std::copy(_window_ends.begin(), _window_ends.end(), window_ends);
    // calculate the number of window pairs
    n_window_pairs = (n_windows * (n_windows-1)) / 2;

    // pass on output_ts etc. to the run_simulations_interface
    *output_ts_p = output_ts;
    *n_pairs_p = n_pairs;
    *n_window_pairs_p = n_window_pairs;

    is_initialized = true;
}
