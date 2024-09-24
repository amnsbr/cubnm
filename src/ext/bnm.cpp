/*
Reduced Wong-Wang model (Deco 2014) simulation on CPU

Parts of this code are based on https://github.com/BrainModes/The-Hybrid-Virtual-Brain, 
https://github.com/murraylab/hbnm & https://github.com/decolab/cb-neuromod

Author: Amin Saberi, Feb 2023
*/

// use a namespace dedicated to cpu implementation
// to avoid overlaps with similar variables on gpu
#include "cubnm/bnm.hpp"

inline void h_global_input_cond(
        u_real& tmp_globalinput, int& k_buff_idx,
        const int& nodes, const int& j, 
        int& k, int& buff_idx, u_real* SC, 
        int* delay, const bool& has_delay, const int& max_delay,
        u_real* conn_state_var_hist, u_real* conn_state_var_1
        ) {
    // calculates global input from other nodes `k` to current node `j`
    // Note: this will not skip over self-connections
    // if they should be ignored, their SC should be set to 0
    // Note: inlining considerably improves performance for this function
    // Note: in case of directed SCs, rows are the source and columns are the target
    tmp_globalinput = 0;
    if (has_delay) {
        for (k=0; k<nodes; k++) {
            // calculate correct index of the other region in the buffer based on j-k delay
            // buffer is moving backward, therefore the delay timesteps back in history
            // will be in +delay time steps in the buffer (then modulo max_delay as it is circular buffer)
            k_buff_idx = (buff_idx + delay[k*nodes+j]) % max_delay;
            tmp_globalinput += SC[k*nodes+j] * conn_state_var_hist[k_buff_idx*nodes+k];
        }
    } else {
        for (k=0; k<nodes; k++) {
            tmp_globalinput += SC[k*nodes+j] * conn_state_var_1[k];
        }            
    }
}

inline void h_global_input_osc(
        u_real& tmp_globalinput, int& k_buff_idx,
        const int& nodes, const int& j, 
        int& k, int& buff_idx, u_real* SC, 
        int* delay, const bool& has_delay, const int& max_delay,
        u_real* conn_state_var_hist, u_real* conn_state_var_1
        ) {
    // calculates global input from other nodes `k` to current node `j`
    // Note: this will not skip over self-connections
    // if they should be ignored, their SC should be set to 0
    // Note: inlining considerably improves performance for this function
    tmp_globalinput = 0;
    if (has_delay) {
        for (k=0; k<nodes; k++) {
            // calculate correct index of the other region in the buffer based on j-k delay
            // buffer is moving backward, therefore the delay timesteps back in history
            // will be in +delay time steps in the buffer (then modulo max_delay as it is circular buffer)
            k_buff_idx = (buff_idx + delay[k*nodes+j]) % max_delay;
            tmp_globalinput += SC[k*nodes+j] * SIN(conn_state_var_hist[k_buff_idx*nodes+k] - conn_state_var_hist[buff_idx*nodes+j]);
        }
    } else {
        for (k=0; k<nodes; k++) {
            tmp_globalinput += SC[k*nodes+j] * SIN(conn_state_var_1[k] - conn_state_var_1[j]);
        }
    }
}

template <typename Model>
void bnm(
        Model* model, int sim_idx,
        double * BOLD_ex, double * fc_tril_out, double * fcd_tril_out,
        u_real **global_params, u_real **regional_params, u_real *v_list,
        u_real *SC, u_real *SC_dist, uint & progress, const uint & progress_final
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
    int* _ext_int_shared = (int*)malloc(Model::n_ext_int_shared * sizeof(int));
    bool* _ext_bool_shared = (bool*)malloc(Model::n_ext_bool_shared * sizeof(bool));

    // initialze the sum (eventually mean) of states_out to 0
    // if not asked to return timeseries
    // initialize extended output sums
    if (model->base_conf.ext_out && (!model->base_conf.states_ts)) {
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
            _global_params, _regional_params[j],
            _ext_int[j], _ext_bool[j],
            _ext_int_shared, _ext_bool_shared
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

    // if indicated, calculate delay matrix of this simulation and allocate
    // memory to conn_state_var_hist according to the max_delay
    int *delay;
    u_real *conn_state_var_hist, *conn_state_var_1;
    float sim_velocity = v_list[sim_idx] * model->dt; // how much signal travels in each integration step (mm)
    int max_delay{0};
    float max_length{0.0};
    float curr_length{0.0};
    int curr_delay{0};
    if (model->do_delay) {
    // note that do_delay is user asking for delay to be considered, has_delay indicates
    // if user has asked for delay AND there would be any delay between nodes given
    // velocity and distance matrix
        delay = (int*)malloc(sizeof(int) * model->nodes * model->nodes);
        for (int i = 0; i < model->nodes; i++) {
            for (int j = 0; j < model->nodes; j++) {
                curr_length = SC_dist[i*model->nodes+j];
                if (i > j) {
                    curr_delay = (int)round(curr_length/sim_velocity); // how many integration steps between i and j
                    // set minimum delay to 1 because a node
                    // cannot access instantaneous states of 
                    // other nodes, as they might not have been
                    // calculated yet
                    curr_delay = std::max(curr_delay, 1);
                    delay[i*model->nodes + j] = curr_delay;
                    delay[j*model->nodes + i] = curr_delay;
                    if (curr_delay > max_delay) {
                        max_delay = curr_delay;
                        max_length = curr_length;
                    }
                } else if (i == j) {
                    delay[i*model->nodes + j] = 1;
                }
            }
        }
    }
    bool has_delay = (max_delay > 0);
    // when there is delay
    // conn_state_var_hist will be used as a circular buffer with a buff_idx
    // to keep track of the current position in the buffer, startgin from the
    // end of the buffer and moving backwards
    int buff_idx = max_delay - 1;
    // allocate memory for history of conn_state_var
    if (has_delay) {
        if (model->base_conf.verbose) {
            std::cout << "Max distance " << max_length << " (mm) with a minimum velocity of " 
                << sim_velocity << " (mm/dt) => Max delay: " 
                << max_delay << " (dt)" << std::endl;
        }
        // allocate memory to conn_state_var_hist for (nodes * max_delay)
        conn_state_var_hist = (u_real*)malloc(sizeof(u_real) * model->nodes * max_delay);
    } else {
        // allocated memory only for the immediate history
        // note: a different variable is used for conssistency with
        // the GPU implementation
        conn_state_var_1 = (u_real*)malloc(sizeof(u_real) * model->nodes);
    }

    for (int j=0; j<model->nodes; j++) {
        if (has_delay) {
            // initialize conn_state_var_hist in every time point of the buffer
            // at the initial value
            for (int bi=0; bi<max_delay; bi++) {
                conn_state_var_hist[bi*model->nodes+j] = _state_vars[j][Model::conn_state_var_idx];
            }
        } else {
            // initialize immediate history of conn_state_var_1
            // at the initial value
            conn_state_var_1[j] = _state_vars[j][Model::conn_state_var_idx];
        }
    }

    // allocate memory to BOLD gsl matrix used for FC and FCD calculation
    gsl_matrix * bold_gsl;
    if (model->base_conf.do_fc) {
        bold_gsl = gsl_matrix_alloc(model->bold_len, model->nodes);
    }

    // define global input function
    HGlobalInputFunc h_global_input_func;
    if (Model::is_osc) {
        h_global_input_func = &h_global_input_osc;
    } else {
        h_global_input_func = &h_global_input_cond;
    }

    // allocate memory for globalinput
    u_real *tmp_globalinput = (u_real*)malloc(sizeof(u_real) * model->nodes);

    // Integration
    bool restart = false;
    int j{0}, k{0}, k_buff_idx{0}, inner_i{0},
        bw_i{0}, bold_i{0}, states_i{0};
    long noise_idx{0};
    // set up noise shuffling if indicated
    #ifdef NOISE_SEGMENT
    /* 
    How noise shuffling works?
    At each outer iteration we will have `i_bw` which is the real time (in units of bw_dt) 
    from the start of simulation, `i_bw % noise_bw_it` which is the real 
    time passed within  each repeat of the noise segment (`curr_noise_repeat`), 
    `sh_ts_noise` which is the shuffled timepoint (column of the noise segment). 
    Similarly, in each thread we have `j` which is mapped to a `sh_j` which will 
    vary in each repeat.
    */
    int curr_noise_repeat{0}, sh_j{0}, sh_ts_noise{0};
    #endif
    // outer loop (of bw iterations, default: 1 msec)
    // TODO: define number of steps for outer
    // and inner loops based on model dt and BW dt from user input 
    while (bw_i < model->bw_it) {
        #ifdef NOISE_SEGMENT
        // get shuffled timepoint corresponding to
        // current noise repeat and the amount of time
        // past in the repeat
        sh_ts_noise = model->shuffled_ts[
            (bw_i % model->noise_bw_it)
            +(curr_noise_repeat*model->noise_bw_it)
        ];
        #endif
        // inner loop (of model iterations, default: 0.1 msec)
        for (inner_i=0; inner_i<model->inner_it; inner_i++) {
            // calculate global input
            for (j=0; j<model->nodes; j++) {
                h_global_input_func(
                    tmp_globalinput[j], k_buff_idx,
                    model->nodes, j, 
                    k, buff_idx, SC, 
                    delay, has_delay, max_delay,
                    conn_state_var_hist, conn_state_var_1
                );
            }
            // run the model step function
            for (j=0; j<model->nodes; j++) {
                #ifdef NOISE_SEGMENT
                sh_j = model->shuffled_nodes[curr_noise_repeat*model->nodes+j];
                noise_idx = (((sh_ts_noise * model->inner_it + inner_i) * model->nodes * Model::n_noise) + (sh_j * Model::n_noise));
                #else
                noise_idx = (((bw_i * model->inner_it + inner_i) * model->nodes * Model::n_noise) + (j * Model::n_noise));
                #endif
                model->h_step(
                    _state_vars[j], _intermediate_vars[j],
                    _global_params, _regional_params[j],
                    tmp_globalinput[j], model->noise, noise_idx
                );
            }
            // update states of nodes in history every 0.1 msec
            if (has_delay) {
                // save the activity of current time point in the buffer
                for (j=0; j<model->nodes; j++) {
                    conn_state_var_hist[buff_idx*model->nodes+j] = _state_vars[j][Model::conn_state_var_idx];
                }
                // move buffer index 1 step back for the next time point
                buff_idx = (buff_idx + max_delay - 1) % max_delay;
            } else {
                // save the activity of current time point in the immediate history
                for (j=0; j<model->nodes; j++) {
                    conn_state_var_1[j] = _state_vars[j][Model::conn_state_var_idx];
                }
            }
        }

        // Balloon-Windkessel model equations
        for (j=0; j<model->nodes; j++) {
            h_bw_step(
                bw_x[j], bw_f[j], bw_nu[j], bw_q[j], tmp_f,
                _state_vars[j][Model::bold_state_var_idx]
            );
        }
        // Calculate and write BOLD to memory every TR
        if ((bw_i+1) % model->BOLD_TR_iters == 0) {
            for (j = 0; j<model->nodes; j++) {
                BOLD_ex[(bold_i*model->nodes)+j] = bwc.V_0 * (bwc.k1 * (1 - bw_q[j]) + bwc.k2 * (1 - bw_q[j]/bw_nu[j]) + bwc.k3 * (1 - bw_nu[j]));
                if (model->base_conf.do_fc) {
                    gsl_matrix_set(bold_gsl, bold_i, j, BOLD_ex[(bold_i*model->nodes)+j]);
                }
            }
            bold_i++;
            // update progress
            if (model->base_conf.verbose) {
                #ifdef OMP_ENABLED
                #pragma omp critical 
                #endif
                {
                    progress++;
                    std::cout << std::fixed << std::setprecision(2) 
                        << ((double)progress / progress_final) * 100 << "%\r" << std::flush;
                }
            }
        }

        // Write states to memory or update their sum
        if (model->base_conf.ext_out) {
            if ((bw_i+1) % model->states_sampling_iters == 0) {
                for (j = 0; j<model->nodes; j++) {
                    if (model->base_conf.states_ts) {
                        for (int ii=0; ii<Model::n_state_vars; ii++) {
                            model->states_out[ii][sim_idx][(states_i*model->nodes)+j] = _state_vars[j][ii];
                        }
                    } else if (states_i >= model->n_states_samples_remove) {
                        // update sum (later mean) of extended
                        // output only after n_samples_remove_states
                        for (int ii=0; ii<Model::n_state_vars; ii++) {
                            model->states_out[ii][sim_idx][j] += _state_vars[j][ii];
                        }                    
                    }
                }
                states_i++;
            }
        }

        #ifdef NOISE_SEGMENT
        // reset noise segment time 
        // and shuffle nodes if the segment
        // has reached to the end
        if ((bw_i+1) % model->noise_bw_it == 0) {
            // at the last time point don't do this
            // to avoid going over the extent of shuffled_nodes
            if (bw_i+1 < model->bw_it) {
                curr_noise_repeat++;
            }
        }
        #endif

        if (Model::has_post_bw_step) {
            model->h_post_bw_step(
                _state_vars, _intermediate_vars,
                _ext_int, _ext_bool, 
                _ext_int_shared, _ext_bool_shared,
                restart,
                _global_params, _regional_params,
                bw_i
            );
        }

        // move forward outer bw loop
        // this has to be before restart
        // because restart will reset bw_i to 0
        bw_i++;
        
        // if restart is indicated (e.g. FIC failed in rWW)
        // reset the simulation and start from the beginning
        if (restart) {
            // model-specific restart
            model->h_restart(
                _state_vars, _intermediate_vars,
                _global_params, _regional_params,
                _ext_int, _ext_bool,
                _ext_int_shared, _ext_bool_shared
            );
            // regional generic resets
            for (j=0; j<model->nodes; j++) {
                // reset Balloon-Windkessel model variables
                bw_x[j] = 0.0;
                bw_f[j] = 1.0;
                bw_nu[j] = 1.0;
                bw_q[j] = 1.0;
                if (has_delay) {
                    // reset conn_state_var_hist in every time point of the buffer
                    // at the initial value
                    for (int bi=0; bi<max_delay; bi++) {
                        conn_state_var_hist[bi*model->nodes+j] = _state_vars[j][Model::conn_state_var_idx];
                    }                    
                } else {
                    // reset conn_state_var_1
                    conn_state_var_1[j] = _state_vars[j][Model::conn_state_var_idx];
                }

            }
            // subtract progress of current simulation
            if (model->base_conf.verbose) {
                #ifdef OMP_ENABLED
                #pragma omp critical 
                #endif
                {
                    progress -= bold_i;
                }
            }
            // reset indices
            bold_i = 0;
            states_i = 0;
            bw_i = 0;
            // reset delay buffer index
            buff_idx = max_delay-1;
            #ifdef NOISE_SEGMENT
            curr_noise_repeat = 0;
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
                _ext_int_shared, _ext_bool_shared,
                global_params, regional_params,
                _global_params, _regional_params[j],
                sim_idx, model->nodes, j
            );
        }
    }

    // divide sum of extended output by number of time points
    // after n_states_samples_remove to calculate the mean
    if (model->base_conf.ext_out && (!model->base_conf.states_ts)) {
        int ext_out_time_points = states_i - model->n_states_samples_remove;
        for (int j=0; j<model->nodes; j++) {
            for (int ii=0; ii<Model::n_state_vars; ii++) {
                model->states_out[ii][sim_idx][j] /= ext_out_time_points;
            }
        }
    }

    if (model->base_conf.do_fc) {
        // Calculate FC and FCD
        // for FC discard first n_vols_remove of BOLD
        // (for FCD this is not needed as window_starts
        // starts after n_vols_remove, as calcualated in get_dfc_windows)
        gsl_matrix_view bold_window =  gsl_matrix_submatrix(
            bold_gsl, 
            model->n_vols_remove, 0, 
            model->bold_len-model->n_vols_remove, model->nodes);
        // calculate FC and copy to numpy arrays
        gsl_vector * fc_tril = model->calculate_fc_tril(&bold_window.matrix);
        memcpy(fc_tril_out, gsl_vector_ptr(fc_tril, 0), sizeof(double) * model->n_pairs);
        if (model->base_conf.do_fcd) {
            // calculate FCD and copy to numpy arrays
            gsl_vector * fcd_tril = model->calculate_fcd_tril(bold_gsl, model->window_starts, model->window_ends);
            memcpy(fcd_tril_out, gsl_vector_ptr(fcd_tril, 0), sizeof(double) * model->n_window_pairs);
            // free memory
            gsl_vector_free(fcd_tril);
        }
        // free memory
        gsl_vector_free(fc_tril);
        gsl_matrix_free(bold_gsl);
    }
    // free memory
    free(bw_x); free(bw_f); free(bw_nu); free(bw_q); 
    if (has_delay) {
        free(conn_state_var_hist);
    } else {
        free(conn_state_var_1);
    }
    if (model->do_delay) {
        free(delay);
    }
    for (int j=0; j<model->nodes; j++) {
        if (Model::n_ext_bool > 0) {
            free(_ext_bool[j]);
        }
        if (Model::n_ext_int > 0) {
            free(_ext_int[j]);
        }
        free(_intermediate_vars[j]);
        free(_state_vars[j]);
        if (Model::n_regional_params > 0) {
            free(_regional_params[j]);
        }
    }
    free(_state_vars); free(_intermediate_vars);
    free(_ext_int); free(_ext_bool);
    free(_regional_params);
    if (Model::n_ext_bool_shared > 0) {
        free(_ext_bool_shared);
    }
    if (Model::n_ext_int_shared > 0) {
        free(_ext_int_shared);
    }
    if (Model::n_global_params > 0) {
        free(_global_params);
    }
    // other variables are freed automatically
    // or should not be freed
}

template <typename Model>
void _run_simulations_cpu(
    double * BOLD_ex_out, double * fc_trils_out, double * fcd_trils_out,
    u_real ** global_params, u_real ** regional_params, u_real * v_list,
    u_real ** SC, int * SC_indices, u_real * SC_dist, BaseModel* m
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
    m->prep_params(global_params, regional_params, v_list, 
        SC, SC_indices, SC_dist, 
        m->global_out_bool, m->global_out_int);

    // run the simulations
    // keep track of a global progress
    uint progress{0};
    uint progress_final{m->bold_len * m->N_SIMS};
    // run the simulations
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
        std::cout << "Thread " << omp_get_thread_num() << " (of " << omp_get_num_threads() << 
            ") is executing particle " << sim_idx << " [" << time_str << "]" << std::endl;
        #else
        std::cout << "Executing particle " << sim_idx << " [" << time_str << "]" << std::endl;
        #endif
        // run the simulation and calcualte FC and FC
        // write the output to chunks of output variables
        // dedicated to the current simulation
        bnm<Model>(
            (Model*)m, sim_idx,
            BOLD_ex_out+(sim_idx*m->bold_size), 
            fc_trils_out+(sim_idx*m->n_pairs), 
            fcd_trils_out+(sim_idx*m->n_window_pairs),
            global_params, regional_params, v_list,
            SC[SC_indices[sim_idx]], SC_dist, 
            progress, progress_final
        );
    }
    if (m->base_conf.verbose) {
        std::cout << "Simulations completed" << std::endl;
    }
}

template <typename Model>
void _init_cpu(BaseModel *m, bool force_reinit) {
    // set up constants (based on dt and bw dt)
    Model::init_constants(m->dt);
    init_bw_constants(&bwc, m->bw_dt);

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
    if (m->base_conf.states_ts) {
        ext_out_size *= m->states_len;
    }
    if (m->base_conf.ext_out) {
        m->states_out = (u_real***)(malloc(Model::n_state_vars * sizeof(u_real**)));
        for (int i = 0; i < Model::n_state_vars; i++) {
            m->states_out[i] = (u_real**)(malloc(m->N_SIMS * sizeof(u_real*)));
            for (int sim_idx = 0; sim_idx < m->N_SIMS; sim_idx++) {
                m->states_out[i][sim_idx] = (u_real*)(malloc(ext_out_size * sizeof(u_real)));
            }
        }
    }
    // specifiy n_states_samples_remove (for states mean calculations)
    m->n_states_samples_remove = m->base_conf.bold_remove_s * 1000 / m->states_sampling;

    if (m->base_conf.do_fc) {
        // specify n_vols_remove (for FC(D) calculations)
        m->n_vols_remove = m->base_conf.bold_remove_s * 1000 / m->BOLD_TR; 
        // calculate length of BOLD after removing initial volumes
        m->corr_len = m->bold_len - m->n_vols_remove;
        if (m->corr_len < 2) {
            std::cerr << "Number of BOLD volumes (after removing initial volumes) is too low for FC calculations" << std::endl;
            exit(1);
        }
        // calculate the number of FC pairs
        m->n_pairs = get_fc_n_pairs(m->nodes, m->base_conf.exc_interhemispheric);
        if (!m->base_conf.do_fcd) {
            m->n_windows = 0;
            m->n_window_pairs = 0;
        } else {
            // calculate the number of windows and their start-ends
            m->n_windows = get_dfc_windows(
                &(m->window_starts), &(m->window_ends), 
                m->corr_len, m->bold_len, m->n_vols_remove,
                m->base_conf.window_step, m->base_conf.window_size,
                m->base_conf.drop_edges
                );
            // calculate the number of window pairs
            m->n_window_pairs = (m->n_windows * (m->n_windows-1)) / 2;
        }
    }

    // check if noise needs to be calculated
    if (
        (m->rand_seed != m->last_rand_seed) ||
        (m->time_steps != m->last_time_steps) ||
        (m->nodes != m->last_nodes) ||
        (m->base_conf.noise_time_steps != m->last_noise_time_steps) ||
        (!m->cpu_initialized) ||
        force_reinit
        ) {
        // precalculate noise (segments) similar to GPU
        #ifndef NOISE_SEGMENT
        // precalculate the entire noise needed; can use up a lot of memory
        // with high N of nodes and longer durations leads maxes out the memory
        m->noise_size = m->nodes * m->bw_it * m->inner_it * Model::n_noise; // *10 for 0.1msec
        #else
        // otherwise precalculate a noise segment and arrays of shuffled
        // nodes and time points and reuse-shuffle the noise segment
        // throughout the simulation for `noise_repeats`
        m->noise_bw_it = (((u_real)(m->base_conf.noise_time_steps) / 1000.0)/ m->bw_dt);
        m->noise_size = m->nodes * m->noise_bw_it * m->inner_it * Model::n_noise;
        m->noise_repeats = ceil((float)(m->bw_it) / (float)(m->noise_bw_it));
        #endif
        if (m->base_conf.verbose) {
            std::cout << "Precalculating " << m->noise_size << " noise elements..." << std::endl;
        }
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
        if (m->base_conf.verbose) {
            std::cout << "noise will be repeated " << m->noise_repeats << " times (nodes [rows] and "
                "timepoints [columns] will be shuffled in each repeat)" << std::endl;
        }
        m->shuffled_nodes = (int*)malloc(m->noise_repeats * m->nodes * sizeof(int));
        m->shuffled_ts = (int*)malloc(m->noise_repeats * m->base_conf.noise_time_steps * sizeof(int));
        get_shuffled_nodes_ts(&(m->shuffled_nodes), &(m->shuffled_ts),
            m->nodes, m->noise_bw_it, m->noise_repeats, &rand_gen);
        #endif
    } else {
        if (m->base_conf.verbose) {
            std::cout << "Noise already precalculated" << std::endl;
        }
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
        std::cout << "Freeing CPU memory (" << this->get_name() << ")" << std::endl;
    }
    #ifdef NOISE_SEGMENT
    free(this->shuffled_nodes);
    free(this->shuffled_ts);
    #endif
    free(this->noise);
    free(this->window_ends);
    free(this->window_starts);
    if (this->base_conf.ext_out) {
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
