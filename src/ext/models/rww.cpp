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
        } else if (pair.first == "max_fic_trials") {
            this->conf.max_fic_trials = std::stoi(pair.second);
        } else if (pair.first == "fic_verbose") {
            this->conf.fic_verbose = (bool)std::stoi(pair.second);
        }
    }
}

