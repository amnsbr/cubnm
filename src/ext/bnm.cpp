/*
Reduced Wong-Wang model (Deco 2014) simulation on CPU

Parts of this code are based on https://github.com/BrainModes/The-Hybrid-Virtual-Brain, 
https://github.com/murraylab/hbnm & https://github.com/decolab/cb-neuromod

Author: Amin Saberi, Feb 2023
*/

// use a namespace dedicated to cpu implementation
// to avoid overlaps with similar variables on gpu
#include "cubnm/bnm.hpp"

template <typename Model>
void bnm(
        Model* model, int sim_idx,
        double * BOLD_ex, double * fc_tril_out, double * fcd_tril_out,
        u_real *SC, u_real **global_params, u_real **regional_params
    ) {
    // copy parameters to local memory as vectors
    // note that to mimick the GPU implementation
    // in regional arrays the first index is node index
    // and second index is the variable or parameter index
    u_real* _global_params = (u_real*)malloc(Model::n_global_params * sizeof(u_real));
    for (int ii = 0; ii < Model::n_global_params; ii++) {
        _global_params[ii] = global_params[ii][sim_idx];
    }
    u_real** _regional_params = (u_real**)malloc(model->nodes * sizeof(u_real*));
    for (int j = 0; j < model->nodes; j++) {
        _regional_params[j] = (u_real*)malloc(Model::n_regional_params * sizeof(u_real));
        for (int ii = 0; ii < Model::n_regional_params; ii++) {
            _regional_params[j][ii] = regional_params[ii][sim_idx * model->nodes + j];
        }
    }
    // create vectors for state variables, intermediate variables
    // and additional ints and bools
    u_real** _state_vars = (u_real**)malloc(model->nodes * sizeof(u_real*));
    for (int j = 0; j < model->nodes; j++) {
        _state_vars[j] = (u_real*)malloc(Model::n_state_vars * sizeof(u_real));
    }
    u_real** _intermediate_vars = (u_real**)malloc(model->nodes * sizeof(u_real*));
    for (int j = 0; j < model->nodes; j++) {
        _intermediate_vars[j] = (u_real*)malloc(Model::n_intermediate_vars * sizeof(u_real));
    }
    int** _ext_int = (int**)malloc(model->nodes * sizeof(int*));
    for (int j = 0; j < model->nodes; j++) {
        _ext_int[j] = (int*)malloc(Model::n_ext_int * sizeof(int));
    }
    bool** _ext_bool = (bool**)malloc(model->nodes * sizeof(bool*));
    for (int j = 0; j < model->nodes; j++) {
        _ext_bool[j] = (bool*)malloc(Model::n_ext_bool * sizeof(bool));
    }

    // initialze the sum (eventually mean) of states_out to 0
    // if not asked to return timeseries
    // initialize extended output sums
    if (model->base_conf.extended_output && (!model->base_conf.extended_output_ts)) {
        for (int j=0; j<model->nodes; j++) {
            for (int ii=0; ii<Model::n_state_vars; ii++) {
                model->states_out[ii][sim_idx][j] = 0;
            }
        }
    }

    // initialze simulation variables
    for (int j=0; j<model->nodes; j++) {
        model->h_init(
            _state_vars[j], _intermediate_vars[j],
            _ext_int[j], _ext_bool[j]
        );
    }

    // Balloon-Windkessel model variables
    u_real* bw_x = (u_real*)malloc(model->nodes * sizeof(u_real));
    u_real* bw_f = (u_real*)malloc(model->nodes * sizeof(u_real));
    u_real* bw_nu = (u_real*)malloc(model->nodes * sizeof(u_real));
    u_real* bw_q = (u_real*)malloc(model->nodes * sizeof(u_real));
    u_real tmp_f;
    for (int j=0; j<model->nodes; j++) {
        bw_x[j] = 0.0;
        bw_f[j] = 1.0;
        bw_nu[j] = 1.0;
        bw_q[j] = 1.0;
    }

    // TODO: delay setup

    // store history of conn_state_vars
    u_real* conn_state_var_1 = (u_real*)malloc(model->nodes * sizeof(u_real));
    for (int j=0; j<model->nodes; j++) {
        conn_state_var_1[j] = _state_vars[j][Model::conn_state_var_idx];
    }

    // allocate memory to BOLD gsl matrix used for FC and FCD calculation
    gsl_matrix * bold_gsl = gsl_matrix_alloc(model->output_ts, model->nodes);

    // Integration
    bool restart = false;
    u_real tmp_globalinput = 0.0;
    int j = 0;
    int k = 0;
    long noise_idx = 0;
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
    while (ts_bold <= model->time_steps) {
        if (model->base_conf.verbose) printf("%.1f %% \r", ((u_real)ts_bold / (u_real)model->time_steps) * 100.0f );
        // TODO: similar to CPU add the option for sync_msec
        #ifdef NOISE_SEGMENT
        sh_ts_noise = model->shuffled_ts[ts_noise+curr_noise_repeat*model->base_conf.noise_time_steps];
        #endif
        for (int int_i = 0; int_i < 10; int_i++) {
            // copy conn_sate of previous time point to conn_state_var_1
            for (j=0; j<model->nodes; j++) {
                conn_state_var_1[j] = _state_vars[j][Model::conn_state_var_idx];
            }
            for (j=0; j<model->nodes; j++) {
                // calculate global input
                tmp_globalinput = 0;
                for (k=0; k<model->nodes; k++) {
                    // TODO: after adding delay make this a separate function
                    tmp_globalinput += SC[j*model->nodes+k] * conn_state_var_1[k];
                }
                // equations
                #ifdef NOISE_SEGMENT
                // * 10 for 0.1 msec steps, nodes * 2 and [sh_]j*2 for two E and I neurons
                sh_j = model->shuffled_nodes[curr_noise_repeat*model->nodes+j];
                noise_idx = (((sh_ts_noise * 10 + int_i) * model->nodes * Model::n_noise) + (sh_j * Model::n_noise));
                #else
                noise_idx = (((ts_bold * 10 + int_i) * model->nodes * Model::n_noise) + (j * Model::n_noise));
                #endif
                model->h_step(
                    _state_vars[j], _intermediate_vars[j],
                    _global_params, _regional_params[j],
                    tmp_globalinput, model->noise, noise_idx
                );
            }
        }

        /*
        Compute BOLD for that time-step (subsampled to 1 ms)
        save BOLD in addition to S_E, S_I, and r_E, r_I if requested
        */
        for (j=0; j<model->nodes; j++) {
            h_bw_step(
                bw_x[j], bw_f[j], bw_nu[j], bw_q[j], tmp_f,
                _state_vars[j][Model::bold_state_var_idx]
            );
            // bw_x[j]  = bw_x[j]  +  _mc.bw_dt * (S_i_E[j] - _mc.kappa * bw_x[j] - _mc.y * (bw_f[j] - 1.0));
            // tmp_f       = bw_f[j]  +  _mc.bw_dt * bw_x[j];
            // bw_nu[j] = bw_nu[j] +  _mc.bw_dt * _mc.itau * (bw_f[j] - POW(bw_nu[j], _mc.ialpha));
            // bw_q[j]  = bw_q[j]  +  _mc.bw_dt * _mc.itau * (bw_f[j] * (1.0 - POW(_mc.oneminrho,(1.0/bw_f[j]))) / _mc.rho  - POW(bw_nu[j],_mc.ialpha) * bw_q[j] / bw_nu[j]);
            // bw_f[j]  = tmp_f;   
        }
        if (ts_bold % model->BOLD_TR == 0) {
            for (j = 0; j<model->nodes; j++) {
                bold_idx = BOLD_len_i*model->nodes+j;
                BOLD_ex[bold_idx] = bwc.V_0 * (bwc.k1 * (1 - bw_q[j]) + bwc.k2 * (1 - bw_q[j]/bw_nu[j]) + bwc.k3 * (1 - bw_nu[j]));
                gsl_matrix_set(bold_gsl, BOLD_len_i, j, BOLD_ex[bold_idx]);
                if (model->base_conf.extended_output && model->base_conf.extended_output_ts) {
                    for (int ii=0; ii<Model::n_state_vars; ii++) {
                        model->states_out[ii][sim_idx][bold_idx] = _state_vars[j][ii];
                    }
                }
                if ((BOLD_len_i>=model->n_vols_remove) && model->base_conf.extended_output && (!model->base_conf.extended_output_ts)) {
                    for (int ii=0; ii<Model::n_state_vars; ii++) {
                        model->states_out[ii][sim_idx][j] += _state_vars[j][ii];
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
        if (ts_noise % model->base_conf.noise_time_steps == 0) {
            curr_noise_repeat++;
            ts_noise = 0;
        }
        #endif

        if (Model::has_post_bw_step) {
            for (j=0; j<model->nodes; j++) {
                bool j_restart = false;
                model->h_post_bw_step(
                    _state_vars[j], _intermediate_vars[j],
                    _ext_int[j], _ext_bool[j], j_restart,
                    _global_params, _regional_params[j],
                    ts_bold
                );
                // restart if any node needs to restart
                restart = restart || j_restart;
            }
        }

        // if restart is indicated (e.g. FIC failed in rWW)
        // reset the simulation and start from the beginning
        if (restart) {
            for (j=0; j<model->nodes; j++) {
                // model-specific restart
                model->h_restart(
                    _state_vars[j], _intermediate_vars[j],
                    _ext_int[j], _ext_bool[j]
                );
                // reset Balloon-Windkessel model variables
                bw_x[j] = 0.0;
                bw_f[j] = 1.0;
                bw_nu[j] = 1.0;
                bw_q[j] = 1.0;
                // reset conn_state_var_1
                conn_state_var_1[j] = _state_vars[j][Model::conn_state_var_idx];
            }
            // // subtract progress of current simulation
            // if (model->base_conf.verbose && (j==0)) {
            //     atomicAdd(progress, -BOLD_len_i);
            // }
            // reset indices
            BOLD_len_i = 0;
            ts_bold = 0;
            // // reset delay buffer index
            // if (has_delay) {
            //     // initialize conn_state_var_hist in every time point at initial value
            //     for (buff_idx=0; buff_idx<max_delay; buff_idx++) {
            //         conn_state_var_hist[sim_idx][buff_idx*nodes+j] = _state_vars[Model::conn_state_var_idx];
            //     }
            //     buff_idx = max_delay-1;
            // }
            #ifdef NOISE_SEGMENT
            curr_noise_repeat = 0;
            ts_noise = 0;
            #endif
            restart = false; // restart is done
        }
    }

    if (Model::has_post_integration) {
        for (j=0; j<model->nodes; j++) {
            model->h_post_integration(
                model->states_out, model->global_out_int, model->global_out_bool,
                _state_vars[j], _intermediate_vars[j], 
                _ext_int[j], _ext_bool[j], 
                global_params, regional_params,
                _global_params, _regional_params[j],
                sim_idx, model->nodes, j
            );
        }
    }

    // divide sum of extended output by number of time points
    // to calculate the mean
    if (model->base_conf.extended_output && (!model->base_conf.extended_output_ts)) {
        int extended_output_time_points = BOLD_len_i - model->n_vols_remove;
        for (int j=0; j<model->nodes; j++) {
            for (int ii=0; ii<Model::n_state_vars; ii++) {
                model->states_out[ii][sim_idx][j] /= extended_output_time_points;
            }
        }
    }

    // Calculate FC and FCD
    // for FC discard first n_vols_remove of BOLD
    // (for FCD this is not needed as window_starts
    // starts after n_vols_remove, as calcualated in get_dfc_windows)
    gsl_matrix_view bold_window =  gsl_matrix_submatrix(
        bold_gsl, 
        model->n_vols_remove, 0, 
        model->output_ts-model->n_vols_remove, model->nodes);
    gsl_vector * fc_tril = model->calculate_fc_tril(&bold_window.matrix);
    gsl_vector * fcd_tril = model->calculate_fcd_tril(bold_gsl, model->window_starts, model->window_ends);

    // copy FC and FCD to numpy arrays
    memcpy(fc_tril_out, gsl_vector_ptr(fc_tril, 0), sizeof(double) * model->n_pairs);
    memcpy(fcd_tril_out, gsl_vector_ptr(fcd_tril, 0), sizeof(double) * model->n_window_pairs);

    // Free memory
    gsl_vector_free(fcd_tril); gsl_vector_free(fc_tril); gsl_matrix_free(bold_gsl);
    free(bw_x); free(bw_f); free(bw_nu); free(bw_q); free(conn_state_var_1);
    for (int j=0; j<model->nodes; j++) {
        free(_state_vars[j]); free(_intermediate_vars[j]);
        free(_ext_int[j]); free(_ext_bool[j]);
    }
    free(_state_vars); free(_intermediate_vars);
    free(_ext_int); free(_ext_bool);
    for (int j=0; j<model->nodes; j++) {
        free(_regional_params[j]);
    }
    free(_regional_params);
    free(_global_params);
    // other variables are freed automatically
    // or should not be freed
}

template <typename Model>
void _run_simulations_cpu(
    double * BOLD_ex_out, double * fc_trils_out, double * fcd_trils_out,
    u_real ** global_params, u_real ** regional_params, u_real * v_list,
    u_real * SC, u_real * SC_dist, BaseModel* m
) {
    if (m->base_conf.verbose) {
        m->print_config();
    }

    // The following currently only does analytical FIC for rWW
    // but in theory can be used for any model that requires
    // parameter modifications
    // TODO: consider doing this in a separate function
    // called from Python, therefore final params are passed
    // to _run_simulations_cpu (except that they might be
    // modified during the simulation, e.g. in numerical FIC)
    m->prep_params(global_params, regional_params, v_list, SC, 
        SC_dist, m->global_out_bool, m->global_out_int);

    // run the simulations
    size_t ext_out_size;
    if (m->base_conf.extended_output_ts) {
        ext_out_size = m->bold_size;
    } else {
        ext_out_size = m->nodes;
    }
    // TODO keep track of a global progress
    // similar to the GPU implementation
    #ifdef OMP_ENABLED
    #pragma omp parallel
	#pragma omp for
    #endif
    for(int sim_idx = 0; sim_idx < m->N_SIMS; sim_idx++) {
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
        bnm<Model>(
            (Model*)m, sim_idx,
            BOLD_ex_out+(sim_idx*m->bold_size), 
            fc_trils_out+(sim_idx*m->n_pairs), 
            fcd_trils_out+(sim_idx*m->n_window_pairs),
            SC, global_params, regional_params
        );
    }
}

template <typename Model>
void _init_cpu(BaseModel *m) {
    // set up global int and bool outputs
    if (Model::n_global_out_int > 0) {
        m->global_out_int = (int**)malloc(Model::n_global_out_int * sizeof(int*));
        for (int i = 0; i < Model::n_global_out_int; i++) {
            m->global_out_int[i] = (int*)malloc(m->N_SIMS * sizeof(int));
        }
    }
    
    if (Model::n_global_out_bool > 0) {
        m->global_out_bool = (bool**)malloc(Model::n_global_out_bool * sizeof(bool*));
        for (int i = 0; i < Model::n_global_out_bool; i++) {
            m->global_out_bool[i] = (bool*)malloc(m->N_SIMS * sizeof(bool));
        }
    }

    // allocate memory for extended output
    size_t ext_out_size = m->nodes;
    if (m->base_conf.extended_output_ts) {
        ext_out_size *= m->output_ts;
    }
    if (m->base_conf.extended_output) {
        m->states_out = (u_real***)(malloc(Model::n_state_vars * sizeof(u_real**)));
        for (int i = 0; i < Model::n_state_vars; i++) {
            m->states_out[i] = (u_real**)(malloc(m->N_SIMS * sizeof(u_real*)));
            for (int sim_idx = 0; sim_idx < m->N_SIMS; sim_idx++) {
                m->states_out[i][sim_idx] = (u_real*)(malloc(ext_out_size * sizeof(u_real)));
            }
        }
    }

    // specify n_vols_remove (for extended output and FC calculations)
    m->n_vols_remove = m->base_conf.bold_remove_s * 1000 / m->BOLD_TR; 
    // calculate length of BOLD after removing initial volumes
    m->corr_len = m->output_ts - m->n_vols_remove;
    if (m->corr_len < 2) {
        printf("Number of BOLD volumes (after removing initial volumes) is too low for FC calculations\n");
        exit(1);
    }
    // calculate the number of FC pairs
    m->n_pairs = get_fc_n_pairs(m->nodes, m->base_conf.exc_interhemispheric);
    // calculate the number of windows and their start-ends
    m->n_windows = get_dfc_windows(
        &(m->window_starts), &(m->window_ends), 
        m->corr_len, m->output_ts, m->n_vols_remove,
        m->window_step, m->window_size,
        m->base_conf.drop_edges
        );
    // calculate the number of window pairs
    m->n_window_pairs = (m->n_windows * (m->n_windows-1)) / 2;

    // check if noise needs to be calculated
    if (
        (m->rand_seed != m->last_rand_seed) ||
        (m->time_steps != m->last_time_steps) ||
        (m->nodes != m->last_nodes) ||
        (m->base_conf.noise_time_steps != m->last_noise_time_steps) ||
        (!m->cpu_initialized)
        ) {
        // precalculate noise (segments) similar to GPU
        #ifndef NOISE_SEGMENT
        // precalculate the entire noise needed; can use up a lot of memory
        // with high N of nodes and longer durations leads maxes out the memory
        m->noise_size = m->nodes * (m->time_steps+1) * 10 * Model::n_noise;
            // +1 for inclusive last time point, *10 for 0.1msec
        #else
        // otherwise precalculate a noise segment and arrays of shuffled
        // nodes and time points and reuse-shuffle the noise segment
        // throughout the simulation for `noise_repeats`
        m->noise_size = m->nodes * (m->base_conf.noise_time_steps) * 10 * Model::n_noise;
        m->noise_repeats = ceil((float)(m->time_steps+1) / (float)(m->base_conf.noise_time_steps)); // +1 for inclusive last time point
        #endif
        printf("Precalculating %d noise elements...\n", m->noise_size);
        if (m->last_nodes != 0) {
            // noise is being recalculated, free the previous one
            free(m->noise);
            #ifdef NOISE_SEGMENT
            free(m->shuffled_nodes);
            free(m->shuffled_ts);
            #endif
        }
        m->last_time_steps = m->time_steps;
        m->last_nodes = m->nodes;
        m->last_rand_seed = m->rand_seed;
        m->last_noise_time_steps = m->base_conf.noise_time_steps;
        std::mt19937 rand_gen(m->rand_seed);
        std::normal_distribution<float> normal_dist(0, 1);
        m->noise = (u_real*)malloc(m->noise_size * sizeof(u_real));
        for (int i = 0; i < m->noise_size; i++) {
            #ifdef USE_FLOATS
            m->noise[i] = normal_dist(rand_gen);
            #else
            m->noise[i] = (double)normal_dist(rand_gen);
            #endif
        }
        #ifdef NOISE_SEGMENT
        // create shuffled nodes and ts indices for each repeat of the 
        // precalculaed noise 
        printf("noise will be repeated %d times (nodes [rows] and "
            "timepoints [columns] will be shuffled in each repeat)\n", m->noise_repeats);
        m->shuffled_nodes = (int*)malloc(m->noise_repeats * m->nodes * sizeof(int));
        m->shuffled_ts = (int*)malloc(m->noise_repeats * m->base_conf.noise_time_steps * sizeof(int));
        get_shuffled_nodes_ts(&(m->shuffled_nodes), &(m->shuffled_ts),
            m->nodes, m->base_conf.noise_time_steps, m->noise_repeats, &rand_gen);
        #endif
    } else {
        printf("Noise already precalculated\n");
    }
    
    m->cpu_initialized = true;
}


void BaseModel::free_cpu() {
    if (strcmp(this->get_name(), "Base")==0) {
        // skip freeing memory for BaseModel
        // though free_gpu normally is not called for BaseModel
        // but keeping it here for safety
        return;
    }
    if (!this->cpu_initialized) {
        // if cpu not initialized, skip freeing memory
        return;
    }
    if (this->base_conf.verbose) {
        printf("Freeing CPU memory (%s)\n", this->get_name());
    }
    #ifdef NOISE_SEGMENT
    free(this->shuffled_nodes);
    free(this->shuffled_ts);
    #endif
    free(this->noise);
    free(this->window_ends);
    free(this->window_starts);
    if (this->base_conf.extended_output) {
        for (int var_idx=0; var_idx<this->get_n_state_vars(); var_idx++) {
            for (int sim_idx=0; sim_idx<this->N_SIMS; sim_idx++) {
                free(this->states_out[var_idx][sim_idx]);
            }
            free(this->states_out[var_idx]);
        }
        free(this->states_out);
    }
    if (this->get_n_global_out_bool() > 0) {
        for (int i=0; i<this->get_n_global_out_bool(); i++) {
            free(this->global_out_bool[i]);
        }
        free(this->global_out_bool);
    }
    if (this->get_n_global_out_int() > 0) {
        for (int i=0; i<this->get_n_global_out_int(); i++) {
            free(this->global_out_int[i]);
        }
        free(this->global_out_int);
    }
}