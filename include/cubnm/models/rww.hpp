#ifndef RWW_HPP
#define RWW_HPP
class rWWModel {
public:
    static constexpr char *name = "rWW";
    static constexpr int n_state_vars = 6; // number of state variables (u_real)
    static constexpr int n_intermediate_vars = 7; // number of intermediate/extra u_real variables
    static constexpr int n_noise = 2; // number of noise elements needed for each node
    static constexpr int n_global_params = 1;
    static constexpr int n_regional_params = 3;
    static constexpr char* state_var_names[] = {"I_E", "I_I", "r_E", "r_I", "S_E", "S_I"};
    static constexpr char* intermediate_var_names[] = {"aIb_E", "aIb_I", "dSdt_E", "dSdt_I", "mean_I_E", "delta", "I_E_ba_diff"};
    static constexpr char* conn_state_var_name = "S_E"; // name of the state variable which sends input to connected nodes
    static constexpr int conn_state_var_idx = 4;
    static constexpr char* bold_state_var_name = "S_E"; // name of the state variable which is used for BOLD calculation
    static constexpr int bold_state_var_idx = 4;
    // the following are needed for numerical FIC
    static constexpr int n_ext_int = 1; // number of additional int variables for each node; fic_trial
    static constexpr int n_ext_bool = 2; // number of additional bool variables for each node; _adjust_fic, fic_failed
    // static constexpr int n_ext_int_shared = 1; // number of additional int variables shared; TODO: make the fic int and bools shared
    // static constexpr int n_ext_bool_shared = 2; // number of additional bool variables shared
    static constexpr int n_global_out_int = 1; // fic_trials
    static constexpr int n_global_out_bool = 2; // fic_unstable, fic_failed
    static constexpr int n_global_out_u_real = 0; // not implemented
    static constexpr int n_regional_out_int = 0; // not implemented
    static constexpr int n_regional_out_bool = 0; // not implemented
    static constexpr int n_regional_out_u_real = 0; // not implemented
    static constexpr bool has_post_bw_step = true;
    static constexpr bool has_post_integration = true;

    struct Constants {
        u_real dt;
        u_real sqrt_dt;
        u_real J_NMDA;
        u_real a_E;
        u_real b_E;
        u_real d_E;
        u_real a_I;
        u_real b_I;
        u_real d_I;
        u_real gamma_E;
        u_real gamma_I;
        u_real tau_E;
        u_real tau_I;
        u_real sigma_model;
        u_real I_0;
        u_real w_E;
        u_real w_I;
        u_real w_II;
        u_real I_ext;
        u_real w_E__I_0;
        u_real w_I__I_0;
        u_real b_a_ratio_E;
        u_real itau_E;
        u_real itau_I;
        u_real sigma_model_sqrt_dt;
        u_real dt_itau_E;
        u_real dt_gamma_E;
        u_real dt_itau_I;
        u_real dt_gamma_I;
        double tau_E_s;
        double tau_I_s;
        double gamma_E_s;
        double gamma_I_s;
        double r_I_ss;
        double r_E_ss;
        double I_I_ss;
        double I_E_ss;
        double S_I_ss;
        double S_E_ss;
        unsigned int I_SAMPLING_START;
        unsigned int I_SAMPLING_END;
        unsigned int I_SAMPLING_DURATION;
        u_real init_delta;
    };

    struct Config {
        bool do_fic;
        int max_fic_trials;
        bool fic_verbose;
    };

    static Constants mc;
    Config conf; // TODO: consider making it static

    // static void init_constants(rWWModel::Constants* mc);
    static void init_constants(Constants* mc) {
        mc->dt  = 0.1; // Time-step of synaptic activity model (msec)
        mc->sqrt_dt = SQRT(mc->dt); 
        mc->J_NMDA  = 0.15;
        mc->a_E = 310; // (n/C)
        mc->b_E = 125; // (Hz)
        mc->d_E = 0.16; // (s)
        mc->a_I = 615; // (n/C)
        mc->b_I = 177; // (Hz)
        mc->d_I = 0.087; // (s)
        mc->gamma_E = (u_real)0.641/(u_real)1000.0; // factor 1000 for expressing everything in ms
        mc->gamma_I = (u_real)1.0/(u_real)1000.0; // factor 1000 for expressing everything in ms
        mc->tau_E = 100; // (ms) Time constant of NMDA (excitatory)
        mc->tau_I = 10; // (ms) Time constant of GABA (inhibitory)
        mc->sigma_model = 0.01; // (nA) Noise amplitude (named sigma_model to avoid confusion with CMAES sigma)
        mc->I_0 = 0.382; // (nA) overall effective external input
        mc->w_E = 1.0; // scaling of external input for excitatory pool
        mc->w_I = 0.7; // scaling of external input for inhibitory pool
        mc->w_II = 1.0; // I->I self-coupling
        mc->I_ext = 0.0; // [nA] external input
        mc->w_E__I_0 = mc->w_E * mc->I_0; // pre-calculating some multiplications/divisions
        mc->w_I__I_0 = mc->w_I * mc->I_0;
        mc->b_a_ratio_E = mc->b_E / mc->a_E;
        mc->itau_E = 1.0/mc->tau_E;
        mc->itau_I = 1.0/mc->tau_I;
        mc->sigma_model_sqrt_dt = mc->sigma_model * mc->sqrt_dt;
        mc->dt_itau_E = mc->dt * mc->itau_E;
        mc->dt_gamma_E = mc->dt * mc->gamma_E;
        mc->dt_itau_I = mc->dt * mc->itau_I;
        mc->dt_gamma_I = mc->dt * mc->gamma_I;

        /*
        FIC parameters
        */
        // tau and gamma in seconds (for FIC)
        mc->tau_E_s = 0.1; // [s] (NMDA)
        mc->tau_I_s = 0.01; // [s] (GABA)
        mc->gamma_E_s = 0.641; // kinetic conversion factor (typo in text)
        mc->gamma_I_s = 1.0;
        // Steady-state solutions in isolated case (for FIC)
        mc->r_I_ss = 3.9218448633; // Hz
        mc->r_E_ss = 3.0773270642; // Hz
        mc->I_I_ss = 0.2528951325; // nA
        mc->I_E_ss = 0.3773805650; // nA
        mc->S_I_ss = 0.0392184486; // dimensionless
        mc->S_E_ss = 0.1647572075; // dimensionless

        /*
        Config
        */
        mc->I_SAMPLING_START = 1000;
        mc->I_SAMPLING_END = 10000;
        mc->I_SAMPLING_DURATION = mc->I_SAMPLING_END - mc->I_SAMPLING_START + 1;
        mc->init_delta = 0.02;
    }

    void init_config(Config* conf) {
        conf->do_fic = true;
        conf->max_fic_trials = 5;
        conf->fic_verbose = false;
    }

    void set_config(std::map<std::string, std::string> config_map, Config* conf) {
        for (const auto& pair : config_map) {
            if (pair.first == "do_fic") {
                conf->do_fic = (bool)std::stoi(pair.second);
            } else if (pair.first == "max_fic_trials") {
                conf->max_fic_trials = std::stoi(pair.second);
            } else if (pair.first == "fic_verbose") {
                conf->fic_verbose = (bool)std::stoi(pair.second);
            }
        }
    }

    CUDA_CALLABLE_MEMBER void init(
        u_real* _state_vars, u_real* _intermediate_vars, 
        int* _ext_int, bool* _ext_bool, rWWModel* model
        );
    CUDA_CALLABLE_MEMBER void step(
        u_real* _state_vars, u_real* _intermediate_vars,
        u_real* _global_params, u_real* _regional_params,
        u_real* tmp_globalinput,
        u_real* noise, long* noise_idx, rWWModel* model
    );
    CUDA_CALLABLE_MEMBER void post_bw_step(
        u_real* _state_vars, u_real* _intermediate_vars,
        int* _ext_int, bool* _ext_bool, bool* restart,
        u_real* _global_params, u_real* _regional_params,
        int* ts_bold, rWWModel* model
    );
    CUDA_CALLABLE_MEMBER void restart(
        u_real* _state_vars, u_real* _intermediate_vars, 
        int* _ext_int, bool* _ext_bool, rWWModel* model
        );
    CUDA_CALLABLE_MEMBER void post_integration(
        u_real **BOLD, u_real ***state_vars_out, 
        int **global_out_int, bool **global_out_bool,
        u_real* _state_vars, u_real* _intermediate_vars, 
        int* _ext_int, bool* _ext_bool, 
        u_real** global_params, u_real** regional_params,
        u_real* _global_params, u_real* _regional_params,
        int sim_idx, int nodes, int j,
        rWWModel* model
    );
};

#endif