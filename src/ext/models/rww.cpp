#include "cubnm/models/rww.hpp"

rWWModel::Constants rWWModel::mc;

void rWWModel::init_constants() {
    mc.dt  = 0.1; // Time-step of synaptic activity model (msec)
    mc.sqrt_dt = SQRT(mc.dt); 
    mc.J_NMDA  = 0.15;
    mc.a_E = 310; // (n/C)
    mc.b_E = 125; // (Hz)
    mc.d_E = 0.16; // (s)
    mc.a_I = 615; // (n/C)
    mc.b_I = 177; // (Hz)
    mc.d_I = 0.087; // (s)
    mc.gamma_E = (u_real)0.641/(u_real)1000.0; // factor 1000 for expressing everything in ms
    mc.gamma_I = (u_real)1.0/(u_real)1000.0; // factor 1000 for expressing everything in ms
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

    /*
    Config
    */
    mc.I_SAMPLING_START = 1000;
    mc.I_SAMPLING_END = 10000;
    mc.I_SAMPLING_DURATION = mc.I_SAMPLING_END - mc.I_SAMPLING_START + 1;
    mc.init_delta = 0.02;
}

void rWWModel::set_conf(std::map<std::string, std::string> config_map) {
    set_base_conf(config_map);
    for (const auto& pair : config_map) {
        if (pair.first == "do_fic") {
            this->conf.do_fic = (bool)std::stoi(pair.second);
            this->base_conf.extended_output = this->conf.do_fic || this->base_conf.extended_output;
            this->modifies_params = true;
        } else if (pair.first == "max_fic_trials") {
            this->conf.max_fic_trials = std::stoi(pair.second);
        } else if (pair.first == "fic_verbose") {
            this->conf.fic_verbose = (bool)std::stoi(pair.second);
        }
    }
}

void rWWModel::prep_params(
        u_real ** global_params, u_real ** regional_params, u_real * v_list,
        u_real * SC, u_real * SC_dist,
        bool ** global_out_bool, int ** global_out_int
        ) {
    // Set wIE to output of analytical FIC
    // if FIC is enabled
    // copy SC to SC_gsl if FIC is needed
    static gsl_matrix *SC_gsl;
    if (this->conf.do_fic) {
        // copy SC to SC_gsl (only once)
        // TODO: handle following cases:
        // 1. SC is modified by the user
        // 2. This function is being called in multiple threads
        if (SC_gsl == nullptr) {
            SC_gsl = gsl_matrix_alloc(this->nodes, this->nodes);
            for (int i = 0; i < this->nodes; i++) {
                for (int j = 0; j < this->nodes; j++) {
                    gsl_matrix_set(SC_gsl, i, j, SC[i*nodes + j]);
                }
            }
        }
        gsl_vector * curr_w_IE = gsl_vector_alloc(this->nodes);
        double *curr_w_EE = (double *)malloc(this->nodes * sizeof(double));
        double *curr_w_EI = (double *)malloc(this->nodes * sizeof(double));
        for (int sim_idx=0; sim_idx<this->N_SIMS; sim_idx++) {
            // make a copy of regional wEE and wEI
            for (int j=0; j<this->nodes; j++) {
                curr_w_EE[j] = (double)(regional_params[0][sim_idx*this->nodes+j]);
                curr_w_EI[j] = (double)(regional_params[1][sim_idx*this->nodes+j]);
            }
            // do FIC for the current particle
            global_out_bool[0][sim_idx] = false;
            // bool* _fic_unstable;
            analytical_fic_het(
                SC_gsl, global_params[0][sim_idx], curr_w_EE, curr_w_EI,
                // curr_w_IE, _fic_unstable);
                curr_w_IE, global_out_bool[0]+sim_idx);
            if (global_out_bool[0][sim_idx]) {
                printf("In simulation #%d FIC solution is unstable. Setting wIE to 1 in all nodes\n", sim_idx);
                for (int j=0; j<this->nodes; j++) {
                    regional_params[2][sim_idx*this->nodes+j] = 1.0;
                }
            } else {
                // copy to w_IE_fic which will be passed on to the device
                for (int j=0; j<this->nodes; j++) {
                    regional_params[2][sim_idx*this->nodes+j] = (u_real)gsl_vector_get(curr_w_IE, j);
                }
            }
        }
    }
}

