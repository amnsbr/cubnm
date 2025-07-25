#include "cubnm/models/rww.hpp"

rWWModel::Constants rWWModel::mc;

void rWWModel::init_constants(double dt) {
    mc.dt  = dt; // Time-step of synaptic activity model (msec)
    mc.sqrt_dt = SQRT(mc.dt); 
    mc.a_E = 310; // (n/C)
    mc.b_E = 125; // (Hz)
    mc.d_E = 0.16; // (s)
    mc.a_I = 615; // (n/C)
    mc.b_I = 177; // (Hz)
    mc.d_I = 0.087; // (s)
    mc.gamma_E = (double)0.641/(double)1000.0; // factor 1000 for expressing everything in ms
    mc.gamma_I = (double)1.0/(double)1000.0; // factor 1000 for expressing everything in ms
    mc.tau_E = 100; // (ms) Time constant of NMDA (excitatory)
    mc.tau_I = 10; // (ms) Time constant of GABA (inhibitory)
    mc.sigma_model = 0.01; // (nA) Noise amplitude (named sigma_model to avoid confusion with CMAES sigma)
    mc.I_0 = 0.382; // (nA) overall effective external input
    mc.w_E = 1.0; // scaling of external input for excitatory pool
    mc.w_I = 0.7; // scaling of external input for inhibitory pool
    mc.w_II = 1.0; // I.I self-coupling
    mc.I_ext = 0.0; // [nA] external input
    mc.w_E__I_0 = mc.w_E * mc.I_0; // pre-calculating some multiplications/divisions
    mc.w_I__I_0 = mc.w_I * mc.I_0;
    mc.b_a_ratio_E = mc.b_E / mc.a_E;
    mc.itau_E = 1.0/mc.tau_E;
    mc.itau_I = 1.0/mc.tau_I;
    mc.sigma_model_sqrt_dt = mc.sigma_model * mc.sqrt_dt;
    mc.dt_itau_E = mc.dt * mc.itau_E;
    mc.dt_gamma_E = mc.dt * mc.gamma_E;
    mc.dt_itau_I = mc.dt * mc.itau_I;
    mc.dt_gamma_I = mc.dt * mc.gamma_I;

    /*
    FIC parameters
    */
    // tau and gamma in seconds (for FIC)
    mc.tau_E_s = 0.1; // [s] (NMDA)
    mc.tau_I_s = 0.01; // [s] (GABA)
    mc.gamma_E_s = 0.641; // kinetic conversion factor (typo in text)
    mc.gamma_I_s = 1.0;
    // Steady-state solutions in isolated case (for FIC)
    mc.r_I_ss = 3.9218448633; // Hz
    mc.r_E_ss = 3.0773270642; // Hz
    mc.I_I_ss = 0.2528951325; // nA
    mc.I_E_ss = 0.3773805650; // nA
    mc.S_I_ss = 0.0392184486; // dimensionless
    mc.S_E_ss = 0.1647572075; // dimensionless
}

void rWWModel::set_conf(std::map<std::string, std::string> config_map) {
    set_base_conf(config_map);
    for (const auto& pair : config_map) {
        if (pair.first == "do_fic") {
            this->conf.do_fic = (bool)std::stoi(pair.second);
            this->base_conf.ext_out = this->conf.do_fic || this->base_conf.ext_out;
            this->modifies_params = true;
        } else if (pair.first == "max_fic_trials") {
            this->conf.max_fic_trials = std::stoi(pair.second);
        } else if (pair.first == "fic_verbose") {
            this->conf.fic_verbose = (bool)std::stoi(pair.second);
        } else if (pair.first == "I_SAMPLING_START") {
            this->conf.I_SAMPLING_START = (std::stoi(pair.second) / 1000) / this->bw_dt;
        } else if (pair.first == "I_SAMPLING_END") {
            this->conf.I_SAMPLING_END = (std::stoi(pair.second) / 1000) / this->bw_dt;
        } else if (pair.first == "init_delta") {
            this->conf.init_delta = std::stod(pair.second);
        }
        this->conf.I_SAMPLING_DURATION = this->conf.I_SAMPLING_END - this->conf.I_SAMPLING_START + 1;
    }
}

void rWWModel::prep_params(
        double ** global_params, double ** regional_params, double * v_list,
        double ** SC, int * SC_indices, double * SC_dist,
        bool ** global_out_bool, int ** global_out_int
        ) {
    // Set wIE to output of analytical FIC
    // if FIC is enabled
    if (!(this->conf.do_fic)) {
        return;
    }
    // copy SC to SCs_gsl
    std::vector<gsl_matrix*> SCs_gsl;
    #ifdef OMP_ENABLED
    #pragma omp critical
    #endif
    {
        for (int SC_idx=0; SC_idx<this->N_SCs; SC_idx++) {
            gsl_matrix* SC_gsl = gsl_matrix_alloc(this->nodes, this->nodes);
            for (int j = 0; j < this->nodes; j++) {
                for (int k = 0; k < this->nodes; k++) {
                    // while copying transpose it from the shape (source, target) to (target, source)
                    // as this is the format expected by the FIC function
                    gsl_matrix_set(SC_gsl, j, k, SC[SC_idx][k*nodes + j]);
                }
            }
            SCs_gsl.push_back(SC_gsl);
        }
    }
    gsl_vector * curr_w_IE = gsl_vector_alloc(this->nodes);
    double *curr_w_EE = (double *)malloc(this->nodes * sizeof(double));
    double *curr_w_EI = (double *)malloc(this->nodes * sizeof(double));
    for (int sim_idx=0; sim_idx<this->N_SIMS; sim_idx++) {
        // assign regional wEE and wEI
        // i.e., convert TVB-style parameters to
        // Demirtas et al. style parameters which
        // are used for the FIC
        // TODO: adapt the code in FIC to use TVB-style
        // parameters instead.
        for (int j=0; j<this->nodes; j++) {
            // w_EI = J_N
            curr_w_EI[j] = (double)(regional_params[1][sim_idx*this->nodes+j]);
            // w_EE = w_p * J_N
            curr_w_EE[j] = (double)(regional_params[0][sim_idx*this->nodes+j]) * curr_w_EI[j];
        }
        // do FIC for the current particle
        global_out_bool[0][sim_idx] = false;
        analytical_fic_het(
            SCs_gsl[SC_indices[sim_idx]], global_params[0][sim_idx], 
            curr_w_EE, curr_w_EI,
            curr_w_IE, global_out_bool[0]+sim_idx);
        if (global_out_bool[0][sim_idx]) {
            std::cerr << "In simulation #" << sim_idx << 
                " analytical FIC failed. Setting wIE to 1 in all nodes" << std::endl;
            for (int j=0; j<this->nodes; j++) {
                regional_params[2][sim_idx*this->nodes+j] = 1.0;
            }
        } else {
            // copy to w_IE_fic which will be passed on to the device
            for (int j=0; j<this->nodes; j++) {
                regional_params[2][sim_idx*this->nodes+j] = (double)gsl_vector_get(curr_w_IE, j);
            }
        }
    }
    #ifdef OMP_ENABLED
    #pragma omp critical
    #endif
    {
        // free SCs_gsl and empty the vector
        for (auto &SC_gsl : SCs_gsl) {
            gsl_matrix_free(SC_gsl);
        }
        SCs_gsl.clear();
    }
}

void rWWModel::h_init(
    double* _state_vars, double* _intermediate_vars, 
    double* _global_params, double* _regional_params,
    int* _ext_int, bool* _ext_bool,
    int* _ext_int_shared, bool* _ext_bool_shared
) {
    // Note that rather than harcoding the variable
    // indices it is also possible to do indexing via
    // strings, but that will be less efficient
    _state_vars[4] = 0.001; // S_E
    _state_vars[5] = 0.001; // S_I
    // numerical FIC initializations
    _intermediate_vars[4] = 0.0; // mean_I_E
    _intermediate_vars[5] = this->conf.init_delta; // delta
    _ext_int_shared[0] = 0; // fic_trial
    _ext_bool_shared[0] = this->conf.do_fic & (this->conf.max_fic_trials > 0); // _adjust_fic in current sim
    _ext_bool_shared[1] = false; // fic_failed
}

void rWWModel::_j_restart(
    double* _state_vars, double* _intermediate_vars, 
    double* _global_params, double* _regional_params,
    int* _ext_int, bool* _ext_bool,
    int* _ext_int_shared, bool* _ext_bool_shared
) {
    // this is different from init in that it doesn't
    // reset the numerical FIC variables delta, fic_trial
    // and _adjust_fic
    _state_vars[4] = 0.001; // S_E
    _state_vars[5] = 0.001; // S_I
    // numerical FIC reset
    _intermediate_vars[4] = 0.0; // mean_I_E
}

void rWWModel::h_step(
        double* _state_vars, double* _intermediate_vars,
        double* _global_params, double* _regional_params,
        double& tmp_globalinput,
        double* noise, long& noise_idx
        ) {
    // I_E = w_E * I_0 + (w_p * J_N) * S_E + global_input * G * J_N - w_IE * S_I
    _state_vars[0] =
        rWWModel::mc.w_E__I_0 
        + _regional_params[0] * _regional_params[1] * _state_vars[4] 
        + tmp_globalinput * _global_params[0] * _regional_params[1] 
        - _regional_params[2] * _state_vars[5];
    // I_I = w_I * I_0 + J_N * S_E - w_II * S_I
    _state_vars[1] = 
        rWWModel::mc.w_I__I_0 
        + _regional_params[1] * _state_vars[4] 
        - rWWModel::mc.w_II * _state_vars[5];
    // aIb_E = a_E * I_E - b_E
    _intermediate_vars[0] = rWWModel::mc.a_E * _state_vars[0] - rWWModel::mc.b_E;
    // aIb_I = a_I * I_I - b_I
    _intermediate_vars[1] = rWWModel::mc.a_I * _state_vars[1] - rWWModel::mc.b_I;
    // r_E = aIb_E / (1 - exp(-d_E * aIb_E))
    _state_vars[2] = _intermediate_vars[0] / (1 - EXP(-rWWModel::mc.d_E * _intermediate_vars[0]));
    // r_I = aIb_I / (1 - exp(-d_I * aIb_I))
    _state_vars[3] = _intermediate_vars[1] / (1 - EXP(-rWWModel::mc.d_I * _intermediate_vars[1]));
    // dS_E = noise * sigma * sqrt(dt) + dt * gamma_E * ((1 - S_E) * (r_E)) - dt * itau_E * S_E;
    _intermediate_vars[2] = 
        noise[noise_idx] * rWWModel::mc.sigma_model_sqrt_dt 
        + rWWModel::mc.dt_gamma_E * ((1 - _state_vars[4]) * _state_vars[2]) 
        - rWWModel::mc.dt_itau_E * _state_vars[4];
    // dS_I = noise * sigma * sqrt(dt) + dt * gamma_I * r_I - dt * itau_I * S_I;
    _intermediate_vars[3] = 
        noise[noise_idx+1] * rWWModel::mc.sigma_model_sqrt_dt 
        + rWWModel::mc.dt_gamma_I * _state_vars[3] 
        - rWWModel::mc.dt_itau_I * _state_vars[5];
    // S_E += dS_E;
    _state_vars[4] += _intermediate_vars[2];
    // S_I += dS_I;
    _state_vars[5] += _intermediate_vars[3];
    // clip S to 0-1
    _state_vars[4] = fmax(0.0f, fmin(1.0f, _state_vars[4]));
    _state_vars[5] = fmax(0.0f, fmin(1.0f, _state_vars[5]));
}

void rWWModel::_j_post_bw_step(
        double* _state_vars, double* _intermediate_vars,
        int* _ext_int, bool* _ext_bool, 
        int* _ext_int_shared, bool* _ext_bool_shared,
        bool& restart,
        double* _global_params, double* _regional_params,
        int& bw_i
        ) {
    if (_ext_bool_shared[0]) {
        if (((bw_i+1) >= this->conf.I_SAMPLING_START) & ((bw_i+1) <= this->conf.I_SAMPLING_END)) {
            _intermediate_vars[4] += _state_vars[0];
        }
        if ((bw_i+1) == this->conf.I_SAMPLING_END) {
            restart = false;
            _intermediate_vars[4] /= this->conf.I_SAMPLING_DURATION;
            _intermediate_vars[6] = _intermediate_vars[4] - rWWModel::mc.b_a_ratio_E;
            if (abs(_intermediate_vars[6] + 0.026) > 0.005) {
                restart = true;
                if (_ext_int_shared[0] < this->conf.max_fic_trials) { // only do the adjustment if max trials is not exceeded
                    // up- or downregulate inhibition
                    if ((_intermediate_vars[6]) < -0.026) {
                        _regional_params[2] -= _intermediate_vars[5];
                        // printf("sim %d node %d (trial %d): %f ==> adjusting w_IE by -%f ==> %f\n", sim_idx, j, fic_trial, I_E_ba_diff, delta, w_IE);
                        _intermediate_vars[5] -= 0.001;
                        _intermediate_vars[5] = fmax(_intermediate_vars[5], 0.001);
                    } else {
                        _regional_params[2] += _intermediate_vars[5];
                        // printf("sim %d node %d (trial %d): %f ==> adjusting w_IE by +%f ==> %f\n", sim_idx, j, fic_trial, I_E_ba_diff, delta, w_IE);
                    }
                }
            }
        }
    }
}

void rWWModel::h_post_bw_step(double** _state_vars, double** _intermediate_vars,
        int** _ext_int, bool** _ext_bool, 
        int* _ext_int_shared, bool* _ext_bool_shared,
        bool& restart,
        double* _global_params, double** _regional_params,
        int& bw_i) {
    BaseModel::h_post_bw_step(
        _state_vars, _intermediate_vars, 
        _ext_int, _ext_bool, 
        _ext_int_shared, _ext_bool_shared,
        restart, 
        _global_params, _regional_params, 
        bw_i);
    // if needs_fic_adjustment in any node do another trial or declare fic failure and continue
    // the simulation until the end
    if ((_ext_bool_shared[0]) && ((bw_i+1) == this->conf.I_SAMPLING_END)) {
        if (restart) {
            if (_ext_int_shared[0] < (this->conf.max_fic_trials)) {
                _ext_int_shared[0]++; // increment fic_trial
            } else {
                // continue the simulation and
                // declare FIC failed
                restart = false;
                _ext_bool_shared[0] = false; // _adjust_fic
                _ext_bool_shared[1] = true; // fic_failed
            }
        } else {
            // if no node needs fic adjustment don't run
            // this block of code any more
            _ext_bool_shared[0] = false;
        }
    }
}


void rWWModel::h_post_integration(
        double ***state_vars_out, 
        int **global_out_int, bool **global_out_bool,
        double* _state_vars, double* _intermediate_vars, 
        int* _ext_int, bool* _ext_bool, 
        int* _ext_int_shared, bool* _ext_bool_shared,
        double** global_params, double** regional_params,
        double* _global_params, double* _regional_params,
        int& sim_idx, const int& nodes, int& j
    ) {
    if (this->conf.do_fic) {
        // save the wIE adjustment results
        // modified wIE array
        regional_params[2][sim_idx*nodes+j] = _regional_params[2];
        if (j == 0) {
            // number of trials and fic failure
            global_out_int[0][sim_idx] = _ext_int_shared[0];
            global_out_bool[1][sim_idx] = _ext_bool_shared[1];
        }
    }
}
